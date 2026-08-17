# HexAgenT on SGLang

HexAgenT is a workflow-aware scheduler for prefill-decode (P-D) disaggregated
serving. The `Relaxed-System-Lab/sglang` fork maintains two independent
implementation branches:

- `hexgent-v0.5.9-main` preserves the implementation used by the retained
  H100/A100 experiments.
- `hexgent-v0.5.15` ports the same control-plane contract onto the upstream
  SGLang `v0.5.15` release tag (`f63458b5be`).

This document describes the current `hexgent-v0.5.15` branch. Historical
profiles remain valid evidence for the v0.5.9 branch; formal v0.5.15 latency
comparisons must regenerate the exclusive profile because the serving runtime
has changed.

This document is the authoritative entry point. Dated files under
`docs/plans/hexgent-migration/` are development evidence, not current setup
instructions.

## Architecture

| Module | Input | Output / responsibility |
| --- | --- | --- |
| Client or DAG replayer | A revealed LLM call, its workflow/call identity, parents, input length and expected output length | An OpenAI-compatible request carrying `hexgent` metadata |
| Rust PD router | Request, registered P/D workers, live snapshots and the latest planner revision | Global admission, placement execution, bootstrap creation at dispatch time, fallback routing and completion events |
| Python planner sidecar | Revealed workflow state, pending calls, worker snapshots, hardware/model parameters and reservations | Prefill/decode placement, stage priorities, revision and safe waiting-request actions |
| SGLang workers | Routed request and stage priority | Continuous-batched prefill/decode, queue snapshots and Mooncake KV transfer |
| Profiler and replayer | Fixed workflow DAG plus per-call input/output lengths | Exclusive service profile, concurrent replay telemetry and workflow SLO ratios |

Production scheduling uses the Rust router in
`sgl-model-gateway/src/routers/http/pd_router.rs` and the sidecar in
`python/sglang/hexgent/`. The removed Python `HexgentLoadBalancer` and vendored
simulator were legacy prototypes and are not part of this data path.

## v0.5.15 Scheduling Boundary

SGLang v0.5.15 makes the integration safer through native priority-aware local
scheduling, improved bootstrap conflict/timeout cleanup, and newer queue and
control-plane primitives. It does not add transparent live migration of a
partially executed request.

HexAgenT therefore uses the following boundary:

1. Before dispatch, a request stays in the router-owned global admission queue.
   Every published planner revision may change its prefill/decode placement and
   global order. Bootstrap state is created only when admission capacity exists
   and the request is actually dispatched.
2. After dispatch, the prefill/decode worker pair is immutable. A newer revision
   may still apply a revision-guarded priority update while the request is in a
   compatible local waiting queue.
3. After prefill, KV transfer, or decode starts, HexAgenT does not attempt
   cross-worker migration. Doing so would require cancellation, KV ownership
   transfer, and a new bootstrap protocol that SGLang v0.5.15 does not provide.

Consequently, v0.5.15 is more convenient for global scheduling before dispatch
and for safe local reprioritization, but it does not remove the fundamental
execution boundary for cross-worker reassignment.

## Reproducibility Rules

1. Replay the same DAG and fixed input/output token lengths for baseline and
   HexAgenT. Do not regenerate a different workload for each scheduler.
2. Build the SLO denominator from an isolated profile with the same model,
   dtype, GPU, TP size and cache settings. e.g. A Llama-70B/H100 profile is not a
   valid SLO for Llama-8B/A100.
3. For serial workflows, SLO is the sum of exclusive call latency. For fan-out
   workflows, SLO is the exclusive-profile latency on the DAG critical path.
4. Disable radix cache for the published comparisons. Otherwise repeated
   prefixes change TTFT and invalidate the exclusive profile.
5. Warm up each scheduler mode separately and give every replay a unique
   `--run-namespace`; otherwise mode order and stale planner DAG state can
   reverse the conclusion.
6. Report planner revision coverage. A run with zero coverage only measures the
   native fallback path, not HexAgenT scheduling.

## Curated Data

See [the asset index](../../real_trace_profiles/README.md). The retained
archives contain input/output lengths and measured TTFT/total latency:

- `h100_real_trace_profiles_final.tar.gz`: H100 TP4 profiles for
  Llama-3.1-70B and Qwen3-235B-A22B FP8.
- `a100_hexgent_final_valid_assets_20260810.tar.gz`: A100 TP1 Llama-8B BF16
  exclusive profiles, fixed traces, ShareGPT/BFCL A/B telemetry and NVLink
  calibration.
- `a100_hotpotqa_hexgent_valid_20260811.tar.gz`: the final two paired HotpotQA
  runs after per-mode warmup and runtime-ID isolation.
- `a100_llama8b_v0515_hexgent_validation_20260817.tar.gz`: release-matched
  v0.5.15 A100/Llama-8B exclusive profile, three-trace A/B results, GPU and
  planner telemetry, and the updated timing calibration artifacts.

## RunPod Environment

Use a persistent `/workspace` volume and keep the repository at
`/workspace/sglang`. On a fresh Pod:

```bash
cd /workspace/sglang
git switch hexgent-v0.5.15
bash scripts/hexgent_real_trace_collectors/setup_hexgent_e2e_pod_env.sh
source /workspace/hexgent_e2e/env.sh
python scripts/hexgent_real_trace_collectors/verify_hexgent_serving_env.py \
  --require-gpu --output /workspace/hexgent_e2e/preflight/manual.json
```

If Hugging Face authentication is required:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

The environment script installs the Python runtime, Rust toolchain, router
extension, Mooncake 0.3.9 and records a platform manifest. For the published
A100 measurements, also build Mooncake with the patches under
`scripts/hexgent_real_trace_collectors/patches/` and verify `NV12` P2P topology.
Do not infer NVLink activation from topology alone; require the Mooncake
transport log and a P2P calibration.

## Prepare Profiles and Traces

```bash
mkdir -p /workspace/hexgent_e2e/assets
tar -xzf real_trace_profiles/a100_hexgent_final_valid_assets_20260810.tar.gz \
  -C /workspace/hexgent_e2e/assets
tar -xzf real_trace_profiles/a100_hotpotqa_hexgent_valid_20260811.tar.gz \
  -C /workspace/hexgent_e2e/assets
```

The A100 archive contains:

```text
a100_llama8b_bf16_profile_final_20260805/exclusive_profile.jsonl
formal_traces/h100_llama70b_profiled_lengths/sharegpt_100.jsonl
formal_traces/h100_llama70b_profiled_lengths/bfcl_50.jsonl
formal_traces/h100_llama70b_profiled_lengths/hotpotqa.jsonl
```

Use `run_a100_llama8b_exclusive_profile.sh` when changing the model, dtype,
hardware or cache policy. Reusing only the old token lengths is allowed; reusing
the old latency as SLO is not.

## Start a P-D Deployment

`start_hexgent_pd_smoke_4p4d.sh` is configurable despite its historical name.
The validated six-GPU A100 topologies are:

| Trace | Topology | Arrival rate |
| --- | ---: | ---: |
| ShareGPT | 1P+5D | 24 workflows/s |
| BFCL-v3 | 4P+2D | 0.3 workflows/s |
| HotpotQA/LATS | 1P+5D | 0.19 workflows/s |

Example for ShareGPT:

```bash
source /workspace/hexgent_e2e/env.sh
export MODEL=meta-llama/Llama-3.1-8B-Instruct
export SERVED_MODEL_NAME=hexgent-llama8b-a100-bf16
export PREFILL_WORKER_COUNT=1
export DECODE_WORKER_COUNT=5
export DECODE_GPU_OFFSET=1
export PREFILL_MAX_RUNNING_REQUESTS=32
export DECODE_MAX_RUNNING_REQUESTS=128
export MOONCAKE_INTRANODE_NVLINK=1
export MOONCAKE_ENGINE_OVERLAY_DIR=/workspace/mooncake_nvlink_overlay
bash scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_4p4d.sh
```

For BFCL set `PREFILL_WORKER_COUNT=4`, `DECODE_WORKER_COUNT=2` and
`DECODE_GPU_OFFSET=4`. Both routers use the same workers:

- HexAgenT: `http://127.0.0.1:33000`
- Native PD baseline: `http://127.0.0.1:33001`

## Replay and Summarize

```bash
export ASSET_ROOT=/workspace/hexgent_e2e/assets
export TRACE_DIR=$ASSET_ROOT/formal_traces/h100_llama70b_profiled_lengths
export EXCLUSIVE_PROFILE=$ASSET_ROOT/a100_llama8b_bf16_profile_final_20260805/exclusive_profile.jsonl
export PROFILE_PATH=$EXCLUSIVE_PROFILE
export MODEL=hexgent-llama8b-a100-bf16
export MODE_ORDER=baseline,hexgent
export TRACE_ORDER=sharegpt
export SHAREGPT_TRACE_PATH=$TRACE_DIR/sharegpt_100.jsonl
export SHAREGPT_ARRIVAL_RATE=24
export PREFILL_WORKER_COUNT=1
export DECODE_WORKER_COUNT=5
bash scripts/hexgent_real_trace_collectors/run_h100_formal_comparison.sh
```

Change `TRACE_ORDER`, trace path, arrival rate and topology for BFCL or
HotpotQA. For final paired evidence, run both `baseline,hexgent` and
`hexgent,baseline`. The runner records calls, summaries, GPU samples, worker
snapshots and planner invocations under `RESULT_ROOT`.

## Validation

```bash
python -m pytest \
  test/unit/utils/test_hexgent_planner.py \
  test/unit/utils/test_hexgent_scheduler.py \
  test/unit/utils/test_hexgent_trace_tools.py \
  test/unit/utils/test_hexgent_scheduling_summary.py \
  test/unit/utils/test_hexgent_worker_bridge.py -q

cd sgl-model-gateway
cargo check
```

Before interpreting performance, require zero replay errors, non-zero planner
coverage, valid snapshots, all worker PIDs alive, and a model-matched SLO file.
