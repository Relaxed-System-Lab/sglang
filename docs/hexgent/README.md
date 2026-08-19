# HexAgenT on SGLang v0.5.15

HexAgenT is a workflow-aware scheduler for prefill-decode (P-D)
disaggregated serving. This branch, `hexgent-v0.5.15`, ports HexAgenT onto the
upstream SGLang v0.5.15 runtime. The older `hexgent-v0.5.9-main` branch is
retained only for the historical H100/A100 experiments that used that runtime.

This README is the setup and reproduction entry point for v0.5.15. The current
performance evidence is in
`docs/plans/hexgent-migration/2026-08-18-v0.5.15-a100-stable-topology-rate-study.md`.

## Architecture

| Component | Inputs | Responsibility |
| --- | --- | --- |
| Client or DAG replayer | A newly revealed call, its workflow/call identity, parents, input length and fixed target output length | Sends an OpenAI-compatible request with `hexgent` metadata. |
| Rust PD router | Request, registered P/D workers, worker snapshots and current planner revision | Maintains global admission, dispatches to selected workers, creates bootstrap at dispatch, applies fallback and reports completion. |
| Python planner sidecar | Revealed DAG state, pending calls, worker snapshots, hardware/model parameters and reservations | Estimates service time, orders runnable calls by projected SLO ratio, chooses P/D placement and publishes a revisioned execution plan. |
| SGLang workers | Routed request and priority | Executes continuous-batched prefill/decode, exposes queue snapshots and performs Mooncake KV transfer. |
| Profile/replay tools | Fixed DAG plus per-call input/output lengths | Builds exclusive SLO profiles and records workflow latency, SLO ratio, GPU, snapshot and planner telemetry. |

The production path is `sgl-model-gateway/src/routers/http/pd_router.rs` plus
`python/sglang/hexgent/`.

## Scheduling Boundary

Before dispatch, a request remains in the router-owned global admission queue.
A new planner revision may change its prefill/decode placement and priority.
The router creates the Mooncake bootstrap only when a prefill worker has
admission capacity and it dispatches the request.

After dispatch, the prefill/decode pair is fixed. A revision-guarded update may
change priority inside a compatible local waiting queue, but it does not
migrate an already dispatched request across workers. Prefill, KV transfer
and decode ownership are therefore never live-migrated.

## Reproduction Rules

1. Baseline and HexAgenT must replay the same DAG and the same fixed input and
   output lengths.
2. Build the SLO denominator from an exclusive profile with the same model,
   dtype, GPU, TP, cache policy and SGLang release.
3. Serial workflow SLO is the sum of exclusive call latency. Fan-out workflow
   SLO is the exclusive critical-path latency of the known DAG.
4. Disable radix cache for published comparisons and warm up each mode
   independently with a unique run namespace.
5. A performance run requires zero replay errors, non-zero planner coverage,
   live worker/snapshot telemetry and all configured P/D workers doing
   meaningful work.
6. Use projected-ratio `high_first` ordering and disable pairwise local search.

## Retained Profile Assets

The repository retains validated H100/A100 profile assets under
`real_trace_profiles/`. See `real_trace_profiles/README.md` for hashes and
contents. The A100 v0.5.15 archive supplies the Llama-8B BF16 exclusive profile
and fixed traces needed for the basic replay; older H100 profiles are retained
for the Llama-70B/Qwen-235B service-model studies.

## Container and GPU Environment Setup

Run the experiments inside a Linux container with NVIDIA GPUs exposed to the
container, a compatible CUDA/PyTorch stack, network access for dependencies and
model downloads, and enough writable storage for the checkout, Hugging Face
cache and result artifacts. Persistent storage is recommended so that model
weights and completed profiles survive container replacement.

The supplied helper scripts currently use `/workspace` as their in-container
working-data convention. This is not a RunPod requirement: mount or bind any
local or persistent host directory at `/workspace` before setup. To use another
in-container path, update the `/workspace` cache, environment and artifact
paths in the helper scripts consistently.

Inside the container:

```bash
export HEXGENT_WORKSPACE=/workspace
mkdir -p "$HEXGENT_WORKSPACE"
git clone --branch hexgent-v0.5.15 \
  https://github.com/Relaxed-System-Lab/sglang.git \
  "$HEXGENT_WORKSPACE/sglang"
export HEXGENT_SRC_DIR="$HEXGENT_WORKSPACE/sglang"

bash "$HEXGENT_SRC_DIR/scripts/hexgent_real_trace_collectors/setup_hexgent_e2e_pod_env.sh"
source "$HEXGENT_WORKSPACE/hexgent_e2e/env.sh"
python "$HEXGENT_SRC_DIR/scripts/hexgent_real_trace_collectors/verify_hexgent_serving_env.py" \
  --source-dir "$HEXGENT_SRC_DIR" \
  --workspace "$HEXGENT_WORKSPACE" \
  --require-gpu --output "$HEXGENT_WORKSPACE/hexgent_e2e/preflight/manual.json"
```

Log in to Hugging Face if the model is not already available:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

The setup script installs the Python runtime, Rust toolchain, router extension
and Mooncake. For intra-node A100 experiments, also use the Mooncake overlay
under `scripts/hexgent_real_trace_collectors/patches/` and verify both the GPU
P2P topology and the Mooncake transport log.

## Prepare Trace and Profile Inputs

Make the retained profile archives available inside the container, then extract
them outside the Git checkout. The following commands use the repository's
tracked archives and the `/workspace` convention above:

```bash
export HEXGENT_ASSET_DIR="$HEXGENT_WORKSPACE/hexgent_e2e/assets"
mkdir -p "$HEXGENT_ASSET_DIR"
tar -xzf "$HEXGENT_SRC_DIR/real_trace_profiles/a100_llama8b_v0515_hexgent_validation_20260817.tar.gz" \
  -C "$HEXGENT_ASSET_DIR"
tar -xzf "$HEXGENT_SRC_DIR/real_trace_profiles/a100_hexgent_final_valid_assets_20260810.tar.gz" \
  -C "$HEXGENT_ASSET_DIR"
```

When changing model, dtype, hardware, TP or cache policy, regenerate the
exclusive profile with:

```bash
bash "$HEXGENT_SRC_DIR/scripts/hexgent_real_trace_collectors/run_a100_llama8b_exclusive_profile.sh"
```

Reusing historical token lengths is valid for controlled replay. Reusing their
latency as the SLO denominator on different serving hardware is not.

## Launch P-D Serving

The historical launcher name `start_hexgent_pd_smoke_4p4d.sh` is configurable.
Set the current scheduling policy before launch:

```bash
source /workspace/hexgent_e2e/env.sh
export MODEL=meta-llama/Llama-3.1-8B-Instruct
export SERVED_MODEL_NAME=hexgent-llama8b-a100-bf16
export PREFILL_MAX_RUNNING_REQUESTS=32
export DECODE_MAX_RUNNING_REQUESTS=128
export HEXGENT_PROJECTED_RATIO_ORDER=high_first
export HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH=0
export MOONCAKE_INTRANODE_NVLINK=1
export MOONCAKE_ENGINE_OVERLAY_DIR=/workspace/mooncake_nvlink_overlay
```

The final A100 pressure points are workload-specific, not universal topology
recommendations:

| Trace | Workflows | Topology | Arrival rate |
| --- | ---: | ---: | ---: |
| ShareGPT | 100 | 1P+5D | 48 workflow/s |
| BFCL-v3 | 50 | 2P+2D | 0.5 workflow/s |
| HotpotQA/LATS | 60 (20 DAGs x3) | 1P+2D | 0.3 workflow/s |

For example, ShareGPT uses:

```bash
export PREFILL_WORKER_COUNT=1
export DECODE_WORKER_COUNT=5
export DECODE_GPU_OFFSET=1
bash scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_4p4d.sh
```

Use the corresponding worker counts for BFCL-v3 or HotpotQA, then run baseline
and HexAgenT with identical settings. The native baseline router listens on
`http://127.0.0.1:33001`; HexAgenT listens on `http://127.0.0.1:33000`.

## Replay and Evaluate

```bash
export ASSET_ROOT=/workspace/hexgent_e2e/assets
export TRACE_DIR=$ASSET_ROOT/formal_traces/h100_llama70b_profiled_lengths
export PROFILE_PATH=/workspace/hexgent_e2e/profiles/exclusive_profile.jsonl
export MODE_ORDER=baseline,hexgent
export TRACE_ORDER=sharegpt
export SHAREGPT_TRACE_PATH=$TRACE_DIR/sharegpt_100.jsonl
export SHAREGPT_ARRIVAL_RATE=48
bash scripts/hexgent_real_trace_collectors/run_h100_formal_comparison.sh
```

Report SLO ratio, workflow latency, planner coverage, GPU telemetry and
snapshot health. The matched A/B result must use the same trace, topology,
arrival process, exclusive profile and run namespace isolation.

For the ShareGPT 100-workflow result, report P99 as the sample maximum. The
current stable A100 result is baseline `4.574x` versus HexAgenT `3.439x` for
that metric; see the pressure-study report for all metrics and limits.

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

The authoritative design and validation references are:

- `docs/plans/2026-08-12-hexgent-v0.5.15-port-design.md`
- `docs/plans/hexgent-migration/2026-08-17-v0.5.15-a100-validation.md`
- `docs/plans/hexgent-migration/2026-08-18-v0.5.15-a100-stable-topology-rate-study.md`
