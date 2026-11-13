# Simulator232 to live SGLang: source and launch guide

This branch aligns the live planner with the stable four-component controller. The accompanying validation receipt identifies the exact source commit, unique CPU suite result and Linux gateway build results. GPU inference, contention and actual cache restoration on the changed branch remain for the next hardware run.

## Controller and capability boundaries

Serving base: `4877d352c07f45eb05515c26f25a267dd5a4666f` (v15). Decision reference: simulator `232cbd06e7bfef308dc38c87caffe35d337b8e14`, canonical `fairclock-tooloff-llama-four-trace-v1/final` A constructor. `scheduler.py` copies that pure core with only its live factory changed; `snapshot_quantile_guard.py` is unchanged.

The factory enables joint P/D, queue/capacity forecasts, known-priority insertion and recomputation above 32 calls. Worker priority is a stable integer rank; risk inputs retain unrounded seconds. Migration, pairwise refinement and completed-tool scheduling signals are OFF. The first live topology supports DP1, including TP2/TP4 workers. D token capacity and request slots come from actual snapshots.

Online H uses full stored service labels and the revealed DAG. Bound device-prefix observations adjust P planning costs and prefill ideal service. Decode ideal service stays full. Tools do not enter the scheduling horizon. Offline evaluator solo-reference H is a separate quantity. Main216 simulator Ours rows use `232/eb9/56da/e95` and migration ON; this branch targets canonical AB migration OFF.

| Capability | Behavior and evidence |
|---|---|
| Four-component controller | Paired CPU decisions against frozen232 cover continuous risk, joint P/D, real D slots/capacity, released queue jobs and 33-call recomputation. |
| Undispatched pending request | Gateway-owned admission can receive a revised P/D choice; snapshot events wake existing pending work without a new arrival. |
| Already dispatched waiting request | Existing placement stays locked; mutable local queues receive revision-CAS priority updates. Cross-worker resend is not implemented. |
| Running D | Observed occupancy is fixed input; running-D migration OFF. |
| P-local prefix cache | Exact token/namespace/cap probes call the actual worker match API. Only observed device tokens discount P; actual model hits still require hardware evidence. |
| Normal P-to-D KV transfer | Native SGLang/Mooncake lifecycle. v15 exercised this with Radix disabled; the changed branch needs another inference smoke. |
| D-generated KV through shared storage to P | Opt-in SGLang D offload, host/HiCache storage and P restoration exist in source. Backend deployment/restoration remain unvalidated; simulator owner/lease/copy-lane semantics are separate. |
| Replay tools | Async wait after HTTP completion, before releasing children; terminal tools count in workflow C. HTTP/LLM and tool clocks stay separate. |

A cache probe can update backend Radix access/node metadata. It provides a planning observation; runtime admission performs the actual match and locking. Host/storage counts grant no device timing credit.

## Reproduce CPU validation

Use Python 3.12 and the packet's `cpu-dependencies.txt`. Core/planner tests load without GPU runtime imports. HTTP/body tests additionally need FastAPI, Pydantic, Starlette, msgspec, orjson and the TestClient transport dependency. Runtime tokenization uses SGLang's ordinary `httpx` dependency.

```bash
cd "$SOURCE_DIR"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HEXGENT_SIM232_SOURCE="$SIM232_SOURCE_DIR"
tests=()
for test_file in test/unit/utils/test_hexgent*.py; do
  [[ "$test_file" == */test_hexgent_matched_h20_simulator.py ]] || tests+=("$test_file")
done
python -B -m pytest -q -p no:cacheprovider "${tests[@]}"
```

The excluded historical `matched_h20_simulator` harness requires its accepted C1+C2 cb714 core hash and intentionally rejects the new232 core. Its source-integrity guard is preserved. The current232 comparisons are `test_hexgent_sim232_decision_parity.py` and `test_hexgent_sim232_batch_parity.py`; the old matched physical-evaluation harness needs a separately scoped update before use with this branch. The final target log supplies one unique suite count; earlier slice counts overlap.

Numerical examples: a 400-token request joins another 400-token D on a 1000-token worker when its second slot is available; a one-slot limit changes the chosen D. In the 33-call case, recomputation moves B ahead of A after the anchor and changes B's predicted elapsed time from 121 to 101 ms. These are CPU forecast cases, not measured GPU gains.

## Build the Linux gateway

Use the packet's pinned `Cargo.lock` and SHA receipt. The repository ignores that lock; the packet preserves the exact graph used for validation. Required tools include Rust/Cargo 1.90.0, C/C++, CMake, Perl, OpenSSL development headers and `protoc`. Put toolchain, cache, target and temp directories on a filesystem with build space.

```bash
export CARGO_HOME="$BUILD_ROOT/cargo" RUSTUP_HOME="$BUILD_ROOT/rustup"
export CARGO_TARGET_DIR="$BUILD_ROOT/target" TMPDIR="$BUILD_ROOT/tmp"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME" "$CARGO_TARGET_DIR" "$TMPDIR"
cd "$SOURCE_DIR/sgl-model-gateway"
# Copy the packet's pinned Cargo.lock here before using --locked.
cargo check --lib --locked -j8
cargo build --locked --bin sgl-model-gateway -j8
cargo test --lib --locked hexgent -j8
```

The packet reports crate checking, test execution and binary linking separately. `cargo check` produces no serving executable. A parser check or std-only helper test does not establish a full gateway build.

## Later RunPod or Hyperstack launch

The commands below illustrate Llama-3.1-70B BF16 on a single host with eight H200 GPUs, using 2P2D TP2. The preserved H20 v15 run used the same functional topology on GPUs with at least 96 GB capacity. Confirm weight and KV fit on the actual allocation. Provider provisioning and model downloads are separate. Set `SOURCE_DIR`, `MODEL_PATH`, `SERVED_MODEL_NAME`, `GATEWAY_BINARY` and a fresh `RUN_DIR` to that host's actual paths. Keep this branch's Python and rebuilt gateway together.

For A100/H100 80 GB, use TP4 per 70B BF16 worker: 2P2D requires 16 physical GPUs. The TP2 `mem-fraction-static=0.82` example leaves insufficient room for 70B BF16 weights plus KV on 80 GB GPUs. Change each worker's TP size and GPU list to four assigned devices. When workers are on multiple hosts, bind their HTTP/bootstrap endpoints to reachable private interfaces and replace gateway/sidecar worker URLs with those actual addresses; the single-host `127.0.0.1` endpoints below do not describe that topology. Size a Qwen235B deployment from its total 235B parameters, not the approximately 22B active parameters, and select its own TP/KV configuration.

Start each worker in its own terminal. This initial transport profile keeps Radix disabled, matching v15. Choose free ports and explicit GPU IDs on the new host.

```bash
export PYTHONPATH="$SOURCE_DIR/python"
export SGLANG_MOONCAKE_TRANSFER_PROTOCOL=tcp MOONCAKE_PROTOCOL=tcp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
common=(--model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME"
  --host 127.0.0.1 --tp-size 2 --dtype bfloat16 --kv-cache-dtype bfloat16
  --context-length 8192 --max-running-requests 16 --mem-fraction-static 0.82
  --disable-radix-cache --disable-cuda-graph --disaggregation-transfer-backend mooncake
  --enable-priority-scheduling --schedule-low-priority-values-first
  --enable-request-time-stats-logging)
# P0 terminal (repeat the exports/common array in each worker terminal):
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server "${common[@]}" \
  --port 31000 --disaggregation-mode prefill --disaggregation-bootstrap-port 8998 \
  --prefill-max-requests 4
# P1 terminal:
CUDA_VISIBLE_DEVICES=2,3 python -m sglang.launch_server "${common[@]}" \
  --port 31001 --disaggregation-mode prefill --disaggregation-bootstrap-port 8999 \
  --prefill-max-requests 4
# D0 terminal:
CUDA_VISIBLE_DEVICES=4,5 python -m sglang.launch_server "${common[@]}" \
  --port 32000 --disaggregation-mode decode
# D1 terminal:
CUDA_VISIBLE_DEVICES=6,7 python -m sglang.launch_server "${common[@]}" \
  --port 32001 --disaggregation-mode decode
```

Wait for all four `/v1/models` endpoints. Inspect GET `/hexgent/pd_snapshot` for DP1, positive `max_total_num_tokens` and `max_running_requests`, correct P/D mode and queues. Limits reflect the actual allocation; the example does not force simulator capacity.

Sidecar terminal:

```bash
export PYTHONPATH="$SOURCE_DIR/python" CUDA_VISIBLE_DEVICES=
export HEXGENT_ENABLE_WORKER_SNAPSHOTS=1 HEXGENT_ENABLE_PREFIX_CACHE_PROBES=0
export HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH=0 HEXGENT_ENABLE_LOCAL_WAITING_MUTATIONS=1
export HEXGENT_REVISION_STATE_PATH="$RUN_DIR/planner_revision"
python -m sglang.hexgent.sidecar --host 127.0.0.1 --port 30999 \
  --telemetry-path "$RUN_DIR/planner.jsonl"
```

Gateway terminal:

```bash
"$GATEWAY_BINARY" --host 127.0.0.1 --port 33000 \
  --prometheus-host 127.0.0.1 --prometheus-port 29001 --pd-disaggregation \
  --prefill http://127.0.0.1:31000 8998 --prefill http://127.0.0.1:31001 8999 \
  --decode http://127.0.0.1:32000 --decode http://127.0.0.1:32001 \
  --hexgent-router --hexgent-planner-url http://127.0.0.1:30999 \
  --hexgent-planner-timeout-ms 10000 --hexgent-fallback-policy fail_closed \
  --hexgent-admission-queue --hexgent-max-prefill-queue-per-worker 16 --json-log
```

Replay a prepared trace:

```bash
python -m sglang.hexgent.workflow_trace replay \
  --trace-path "$TRACE_PATH" --router-url http://127.0.0.1:33000 \
  --model "$SERVED_MODEL_NAME" --output-path "$RUN_DIR/replay.jsonl"
```

`tool_latency_s` is a per-call trace field. The replayer records HTTP response before awaiting the tool and releasing children. Workflow C includes terminal tool completion. Supply measured/model service labels through the existing trace/profile interfaces for scheduling evaluation. Fallback linear worker labels support wiring checks; performance claims require matched physical measurements. The inherited single-call P estimator excludes returned weight bytes from its memory term, while the batch P estimator includes them; this branch preserves those formulas.

## Enable and verify P-local cache

For the first cache profile use identical tokenizers, unsalted no-LoRA text requests. Restart P workers without `--disable-radix-cache` and the sidecar with snapshot/probe flags both `1`. Original gateway JSON goes through backend `/v1/tokenize`, preserving messages, tools and template options. Flattened `request_text` never supplies fabricated IDs.

Automatic binding supports text `/v1/chat/completions` and native `/generate` text or exact IDs. Unknown namespace, multimodal/embedding and unproven completion-template paths receive no credit. Scientific labels remain unchanged: a modeled input-token count differing from the backend ID count receives no prefix timing discount.

Each P receives POST `/hexgent/pd_snapshot` with bounded probes. Credit requires matching workflow/call ID, digest, runtime source and device count. Each solve covers at most 64 probes and 262144 submitted tokens; uncovered calls remain unknown. Failed refresh clears old cache observations, retains the caller's observed occupancy and performs no immediate second query. Automatic probing OFF still permits supplied offline observations for deterministic CPU tests.

Compare cold, repeated exact-prefix and unrelated requests using actual P0/P1 probe receipts and worker behavior. `disabled` means known zero; missing or rejected probes mean unknown. P suffix work can decrease while full D context, memory demand and full P-to-D transfer remain intact. Shared D-generated KV requires separate backend configuration plus publication/restoration receipts.

For the next inference run retain source/executable hashes, original requests, tokenizer/probe receipts, P/D snapshots, mutation revisions/outcomes, dispatch identities, bootstrap transfers and replay timestamps. Verify queue competition and an obsolete-revision event without duplicate dispatch. Keep the new outcome separate from preserved v15 (10/10 calls, 2/2 workflows, cache disabled) and the v13 failure packet.
