# HexAgenT Tail SLO-Ratio Correction Plan

**Goal:** Make HexAgenT optimize the published global execution plan without adding a per-call control-plane wait, use an honest component transfer model, and prioritize known fan-out critical paths without changing the serial-workflow ordering rule.

**Observed problem:** The 2P+2D H100 replay does not show a robust HexAgenT gain. ShareGPT is worse in mean and P95 SLO ratio, BFCL-v3 is slightly worse in mean but better at P95 in one non-repeated run, and HotpotQA is worse in both. Therefore the corrective work cannot be limited to LATS fan-out scheduling.

## Root Causes To Address

1. The router waits for a per-request `/wait_admission` result before it dispatches. This adds 12--30 ms to every call even when the planner's batch solve is only milliseconds.
2. The planner ranks individual calls greedily by a local projected ratio. It does not publish a durable global plan which the router can follow while the next solve is pending.
3. The current real LATS fan-out branch returns `call_laxity` before it can apply `critical_path_ms / workflow_horizon_ms`. The existing fan-out priority rule is consequently unreachable for long-prompt LATS calls.
4. `10 ms + payload / 7 GB/s` was fitted to a request-level Mooncake pipeline. It conflates NVLink payload copy, transfer-engine queueing, producer/consumer coordination, and decode-ready notification. It is not a valid raw-link roofline parameter.

## Target Semantics

- The sidecar owns an atomically published, revisioned execution plan for all globally admitted requests.
- A new request is immediately assigned a native P-D fallback decision. Until bootstrap/dispatch, a newer published HexAgenT plan may replace that decision.
- At prefill admission capacity, the router selects the newest valid plan entry. If no plan entry exists, it dispatches the fallback in global FCFS/priority order; it does not wait for a planner HTTP response.
- The planner receives cached live worker snapshots and models three distinct transfer components: payload copy, fixed producer/consumer coordination, and snapshot/telemetry-derived contention delay.
- Serial workflows retain projected-ratio ordering. A known same-workflow fan-out applies critical-path pressure before generic laxity, independently of prompt length.

## Implementation Tasks

### Task 1: Pod preflight and reproducibility

Files:
- Modify: `scripts/hexgent_real_trace_collectors/setup_hexgent_e2e_pod_env.sh`
- Modify: `scripts/hexgent_real_trace_collectors/verify_hexgent_serving_env.py`
- Test: `test/unit/utils/test_hexgent_real_trace_collectors.py`

Add a non-mutating preflight that validates the persistent source/cache paths, Python/venv, Torch CUDA ABI, `sgl_kernel`, Mooncake, router extension, Cargo/maturin/protoc, CUDA shared libraries, GPU topology, and writable telemetry directory. Persist the JSON result in `/workspace/hexgent_e2e/preflight/` and run it at the end of setup.

### Task 2: Correct fan-out priority regression

Files:
- Modify: `python/sglang/hexgent/scheduler.py`
- Test: `test/unit/utils/test_hexgent_scheduler.py`

Write a failing test for a same-workflow, multi-child, long-prompt fan-out batch where the larger `critical_path_ms / workflow_horizon_ms` must be scheduled first. Then move the fan-out priority branch before the generic `same_workflow_fanout_batch` laxity return in both prefill and decode paths. Keep the existing serial test unchanged.

### Task 3: Published execution-plan cache

Files:
- Modify: `python/sglang/hexgent/sidecar.py`
- Modify: `sgl-model-gateway/src/routers/http/pd_router.rs`
- Modify: `sgl-model-gateway/src/routers/http/pd_types.rs`
- Tests: `test/unit/utils/test_hexgent_planner.py`, Rust router unit tests

Define a plan-entry state with request key, revision, decision, fallback decision, state (`planned`, `dispatching`, `dispatched`, `cancelled`), and timestamps. The sidecar publishes the complete batch as one revision. The router enqueues asynchronously, returns/retains a native fallback immediately, and resolves the final worker pair only at admission. Add timestamps for enqueue, solve start/end, plan publication, fallback selection, override, and dispatch. Preserve FCFS/priority order when no plan entry is available.

### Task 4: Component transfer model and live telemetry

Files:
- Modify: `python/sglang/hexgent/latency_estimator.py`
- Modify: `python/sglang/hexgent/hardware.py`
- Modify: `python/sglang/srt/disaggregation/mooncake/conn.py`
- Modify: worker snapshot producer/consumer code
- Modify: `scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_2p2d.sh`
- Tests: `test/unit/utils/test_pd_telemetry_calibration.py`, latency estimator tests

Represent raw payload-copy bandwidth separately from coordination overhead and transfer contention. The raw-copy field uses topology-specific measured CUDA-event data only. Worker telemetry/snapshots supply inflight transfer count, queue depth, producer-ready percentiles, and decode-ready delay; the planner uses these for the non-copy terms. Remove the `7 GB/s + 10 ms` default from the launcher.

### Task 5: Controlled validation

On a fresh RunPod pod, run preflight before any model start. Run a 1P+1D transfer component measurement, then 2P+2D baseline/HexAgenT replay with the same model, exact input/output lengths, workflow DAG, and exclusive-profile SLO. Report P95/P99 workflow SLO ratio, not only mean latency. Run repeated orders before claiming a scheduler gain.
