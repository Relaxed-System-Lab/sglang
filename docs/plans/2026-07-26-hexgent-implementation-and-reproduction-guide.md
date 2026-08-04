# HexAgenT on SGLang v0.5.9: Implementation and Reproduction Guide

## 1. Scope and Design

HexAgenT is implemented as a workflow-aware control plane on top of SGLang
P-D disaggregated serving. The target upstream base is SGLang release v0.5.9.
The design keeps high-frequency model execution inside SGLang and places
workflow state, latency prediction, and scheduling decisions in Python.

    Client / DAG-aware replay
      -> Rust PD router: admission, worker dispatch, bootstrap, fallback
      -> Python HexAgenT planner: state, roofline, scheduling decision
      -> SGLang prefill workers: prompt processing and KV production
      -> Mooncake: P-to-D KV transfer
      -> SGLang decode workers: continuous decode batching and completion
      -> router event -> planner state update

The planner never assumes the full future workflow DAG. A call is known only
when it is revealed. Its metadata contains the parent calls that are already
known to gate this call, allowing the planner to maintain a revealed-DAG
critical path without an oracle for future LLM calls.

## 2. Components and Interfaces

### 2.1 Client metadata and DAG replay

Implementation: python/sglang/hexgent/workflow_trace.py and
python/sglang/hexgent/schemas.py.

Each routed request carries a hexgent object:

    {
      "workflow_id": "workflow-17",
      "call_id": "workflow-17-step-3",
      "workflow_arrival_ms": 1730000000000,
      "parent_call_ids": ["workflow-17-step-1", "workflow-17-step-2"],
      "input_tokens": 1024,
      "predicted_output_tokens": 192
    }

Required fields are workflow_id, call_id, and workflow_arrival_ms.
parent_call_ids is an empty list for a root. The replay uses it to wait until
dependencies complete; the planner uses it to derive its revealed critical
path. input_tokens and predicted_output_tokens are strongly recommended. In a
fixed-length replay they come from the profile trace. In online serving, a
proxy/BERT predictor can provide output length and a tokenizer can estimate
input length.

Deprecated compatibility fields such as workflow_horizon_ms are accepted on the
wire but ignored by the planner. Client metadata must not set the online
horizon.

Input: a workflow DAG and fixed request/output length data.

Output: dependency-respecting OpenAI-compatible requests, one for each
revealed LLM call, plus per-call replay records containing actual response,
latency, selected P/D workers, and scheduling headers.

### 2.2 Hardware and roofline latency model

Implementation: python/sglang/hexgent/hardware.py and
python/sglang/hexgent/latency_estimator.py.

hardware.py packages the GPU compute, HBM bandwidth, on-chip memory, and VRAM
profiles normalized from LLM-simulator-main/simulator/configs/hardware.py.
Supported labels include H100 SXM/PCIe, A100 40G/80G, H20, and L40S.

RooflineLatencyEstimator evaluates prefill and decode operator work using the
maximum of compute time and HBM time. It models:

- prefill TTFT as compute/memory work for the prompt;
- packed prefill batches by sharing model-weight traffic while adding
  per-request activation, attention, and KV work;
- decode as a sequence of iterations over the generated tokens;
- continuous decode batches by sharing model-weight traffic but scaling dynamic
  activation/KV work by the active batch size;
- KV transfer bytes from model layer/head/KV dimensions, plus separately
  configurable link bandwidth and startup cost.

HybridLatencyEstimator first uses a measured profile row when supplied and uses
roofline batch scaling to adapt that measurement to current decode occupancy.
If no model/hardware configuration is available, it falls back to the worker's
linear labels. This makes first-request planning possible before runtime
telemetry is available.

Input: model dimensions, precision/TP information, GPU profile, request token
lengths, current decode batch size, and optional calibrated stage times.

Output: per-P-worker prefill time, per-D-worker transfer time, and per-D-worker
decode time.

### 2.3 Live snapshot capacity model

Implementation: python/sglang/hexgent/snapshots.py.

The Rust router fetches GET /hexgent/pd_snapshot from every worker before an
admission attempt. The snapshot identifies P bootstrap/waiting/inflight work,
D preallocation/transfer/waiting work, D running requests, memory-token use,
and worker limits.

The planner converts these snapshots into:

- prefill worker availability from packed batches constrained by token and
  request limits;
- decode worker running batch size and remaining compute time;
- decode token-capacity intervals, including requests already in the D batch;
- transfer-pipeline work, respecting
  SGLANG_DISAGGREGATION_MAX_INFLIGHT_TRANSFERS when configured.

This is a capacity estimate, not an exact token-by-token emulator. The scheduler
is therefore re-invoked against fresh snapshots at admission time.

Input: real worker queues/running state and worker limits.

Output: P and D availability times, D capacity intervals, active decode batch
sizes, and mutable waiting-request records.

### 2.4 Stateful Python planner

Implementation: python/sglang/hexgent/planner.py,
python/sglang/hexgent/scheduler.py, and python/sglang/hexgent/sidecar.py.

The sidecar exposes:

| Endpoint | Input | Output |
|---|---|---|
| GET /health | none | status, revision, workflow/pending counts |
| POST /schedule_request or /plan | a request, workers, and snapshots | P/D worker selection, priorities, reservation, revision |
| POST /admit_request | same request plus P admission-window policy | dispatch-ready or retry decision; schedule if ready |
| POST /event | completion/cancellation and stage telemetry | updated planner health/state |

For every revealed call, the planner:

1. Estimates P, transfer, and D service on each eligible worker.
2. Adds/updates the call in WorkflowRuntimeState.
3. Computes workflow_horizon_ms as the longest root-to-revealed-node service
   path. A parent edge contributes only when it has been revealed.
4. Uses live snapshots plus reservations to schedule P and D jointly.
5. Stores a reservation whose D occupancy starts after estimated P completion,
   rather than at request arrival.
6. Returns P/D workers, priorities, a planned decode instance, predicted output
   length, schedule revision, and optional waiting-queue mutation actions.

The scheduler ranks calls by projected completion and horizon. For a revealed
fan-out it additionally uses known critical-path pressure, whereas ordinary
serial workflows retain the prior laxity/projected-ratio behavior. Decode
placement observes token capacity intervals and current continuous-batch
occupancy.

On a completion event, the planner keeps two separate values:

- end-to-end wall latency, for observed workflow completion;
- measured exclusive stage service, for horizon/model calibration.

attained_service_ms is the completed DAG critical path, never the sum of
parallel branch service. This preserves serial behavior and corrects fan-out
accounting.

### 2.5 Rust PD router and global admission

Implementation: sgl-model-gateway/src/routers/http/pd_router.rs and
sgl-model-gateway/src/routers/http/pd_types.rs.

When --hexgent-router is enabled, the router:

1. extracts the original hexgent JSON without changing its dependency metadata;
2. discovers P and D workers from the existing registry;
3. fetches their snapshots and calls the Python planner;
4. keeps an undisbursed request in the global admission queue until the chosen
   P worker has admission credit;
5. creates the Mooncake bootstrap only at dispatch time, so queued requests can
   be replanned without KV migration;
6. validates the planner-selected workers and injects scheduling revision,
   priorities, and the planned D instance into the actual worker request;
7. uses native PD routing when the planner fails and fallback policy is
   pd-policy, or fails closed when configured;
8. records pair-selection duration immediately after pair choice, then posts
   completion telemetry back to POST /event.

The global queue controls cross-worker choice before P dispatch. It does not
replace SGLang's local schedulers: after dispatch, P and especially D workers
still maintain local queues and continuous batches. A request is not migrated
after prefill/KV transfer/decode starts because its bootstrap/KV state is then
bound to its selected P/D pair.

### 2.6 Worker control and timing interfaces

Implementation: python/sglang/srt/entrypoints/http_server.py,
python/sglang/srt/managers/scheduler.py, scheduler_metrics_mixin.py, and
related I/O structs.

Worker interfaces:

| Endpoint | Purpose |
|---|---|
| GET /hexgent/pd_snapshot | returns P/D queues, running requests, token state, and limits |
| POST /hexgent/update_waiting_priority | changes a safe local waiting-request priority |
| POST /hexgent/reassign_waiting | changes an eligible waiting target before execution |
| GET /hexgent/request_timing | returns bounded P/D stage timing for a completed call |

The timing record separates prefill execution, transfer, and decode execution
from queue wait. The router constructs measured_service_ms only from complete
stage records; if a record is missing, the planner falls back to roofline.

## 3. Reproduction Procedure

### 3.1 Build and install

Start from a clean Linux checkout at SGLang v0.5.9, then apply the HexAgenT
Python, worker, and sgl-model-gateway changes in this repository. Build the
Rust router in the same Linux/container environment as SGLang; local Windows
validation alone is insufficient because the router needs the platform linker
and CUDA/Mooncake runtime used by the workers.

Install the Python serving dependencies and the SGLang package in editable mode
from the same checkout. Verify:

    python -m pytest test/unit/utils/test_hexgent_planner.py \
      test/unit/utils/test_hexgent_trace_tools.py \
      test/unit/srt/test_hexgent_worker_bridge.py -q
    cd sgl-model-gateway && cargo fmt --all -- --check

### 3.2 Start a homogeneous 4P+4D deployment

Use eight GPUs with the same model replica on each worker. Start four P workers
and four D workers with the same model/precision/TP setting. P workers should
have an admission/running limit sufficient for packed prefill batches; D
workers require a larger running pool for continuous batching. Configure
Mooncake P-D transfer and verify the worker logs state the actual selected
transport. Do not infer direct NVLink merely from nvidia-smi topo; require a
runtime log such as Using Intra-Node NVLink transport.

Start the Python sidecar with the model and hardware descriptor used by the
workers. Start two router configurations against the same P/D worker pool only
when runs are serialized: one native baseline and one HexAgenT router. Record
the exact launch environment, including transfer bandwidth/startup overrides,
admission window, token limits, CUDA graph setting, and random seed.

### 3.3 Run replay

Build WorkflowTrace JSONL files from ShareGPT, BFCL-v3, and LATS/HotpotQA
collectors. Preserve the true dependency edges. Build a same-model/same-GPU
exclusive service profile and pass it as a fixed-length replay profile. The
replayer waits for parent completion, submits independent fan-out calls
concurrently, and forces each call's recorded output length.

For a valid comparison:

1. Use identical workflows, fixed input/output lengths, arrival seed, model,
   worker topology, and cache policy in both modes.
2. Run at least three paired trials with alternating order.
3. Save calls, planner invocations, snapshot samples, GPU telemetry, worker
   logs, router logs, and transfer telemetry in one immutable archive.
4. Compute workflow SLO from the exclusive profile over the DAG longest path;
   do not sum parallel branches.
5. Keep router pair-selection time separate from end-to-end request latency;
   it must be captured before P-D dispatch, while planner overhead comes from
   the sidecar's per-invocation telemetry.

## 4. Current Boundaries

- The planner supports one logical LLM call per routed request. Multi-request
  OpenAI batches use native routing unless explicitly split before admission.
- Exact future DAG shape is not assumed. Only parent IDs known at reveal are
  scheduled.
- Snapshot/roofline capacity is intentionally approximate; real P/D timing is
  used to calibrate it after completion.
- Cross-worker migration is supported only before dispatch/bootstrap. Live KV
  migration after P, transfer, or D starts is not part of the current design.
- The direct NVLink run proves runtime transport selection, but needs
  per-request transfer telemetry before the planner's link model can be called
  calibrated.

## 5. File Map

| Area | Main files |
|---|---|
| Metadata and sidecar schema | python/sglang/hexgent/schemas.py, sidecar.py |
| Planner and state | python/sglang/hexgent/planner.py |
| Scheduling policy | python/sglang/hexgent/scheduler.py |
| Roofline/hardware | hardware.py, latency_estimator.py |
| Snapshot capacity model | snapshots.py |
| Trace conversion/replay | workflow_trace.py, service_profiler.py |
| Rust execution plane | sgl-model-gateway/src/routers/http/pd_router.rs, pd_types.rs |
| Worker snapshot/timing | python/sglang/srt/entrypoints/http_server.py and python/sglang/srt/managers |
| Planner/replay tests | test/unit/utils/test_hexgent_planner.py and test/unit/utils/test_hexgent_trace_tools.py |

The NVLink evidence and the associated debugging decisions are recorded in
docs/plans/2026-07-26-llama8b-h100-4p4d-nvlink-p2p-replay-report.md and
docs/plans/2026-07-26-hexgent-nvlink-replay-debug-and-optimization.md.
