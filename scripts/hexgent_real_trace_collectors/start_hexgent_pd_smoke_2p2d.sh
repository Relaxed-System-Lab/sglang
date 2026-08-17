#!/usr/bin/env bash
set -euo pipefail

# Start a 4-GPU functional PD-disaggregated deployment for HexAgenT E2E
# debugging. Each worker occupies one GPU:
# GPU0/GPU1 are prefill workers, GPU2/GPU3 are decode workers.
# Two routers are exposed:
#   33000: HexAgenT planner-backed router
#   33001: baseline SGLang PD router

VENV_DIR="${HEXGENT_VENV_DIR:-/workspace/hexgent_sglang_venv}"
SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang}"
RUN_DIR="${HEXGENT_RUN_DIR:-/workspace/hexgent_e2e/run}"
LOG_DIR="${HEXGENT_LOG_DIR:-/workspace/hexgent_e2e/logs}"
export HEXGENT_REVISION_STATE_PATH="${HEXGENT_REVISION_STATE_PATH:-$RUN_DIR/planner_revision}"
PLANNER_TELEMETRY_PATH="${HEXGENT_PLANNER_TELEMETRY_PATH:-$LOG_DIR/planner_invocations.jsonl}"

MODEL="${MODEL:-neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama8b-pd-2p2d}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
# Retain headroom for true prefill batching and a long-lived decode batch. KV
# capacity is bounded by max-total-tokens rather than a four-request ceiling.
PREFILL_MAX_RUNNING_REQUESTS="${PREFILL_MAX_RUNNING_REQUESTS:-32}"
PREFILL_MAX_BATCH_REQUESTS="${PREFILL_MAX_BATCH_REQUESTS:-32}"
DECODE_MAX_RUNNING_REQUESTS="${DECODE_MAX_RUNNING_REQUESTS:-128}"
DECODE_MAX_INFLIGHT_TRANSFERS="${DECODE_MAX_INFLIGHT_TRANSFERS:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
# This is deliberately an end-to-end P-D estimate, not the raw NVLink copy
# bandwidth. It includes Mooncake producer-ready and receiver-side overhead.
TRANSFER_BANDWIDTH_BYTES_PER_S="${TRANSFER_BANDWIDTH_BYTES_PER_S:-7000000000}"
TRANSFER_STARTUP_MS="${TRANSFER_STARTUP_MS:-10}"
HEXGENT_ADMISSION_MAX_WAIT_MS="${HEXGENT_ADMISSION_MAX_WAIT_MS:-300000}"
HEXGENT_ADMISSION_POLL_INTERVAL_MS="${HEXGENT_ADMISSION_POLL_INTERVAL_MS:-10}"
HEXGENT_MAX_PREFILL_QUEUE_PER_WORKER="${HEXGENT_MAX_PREFILL_QUEUE_PER_WORKER:-${PREFILL_MAX_RUNNING_REQUESTS}}"
HEXGENT_PREFILL_BATCHING_HEADROOM="${HEXGENT_PREFILL_BATCHING_HEADROOM:-16}"
HEXGENT_PREFILL_TOKEN_HEADROOM_PERCENT="${HEXGENT_PREFILL_TOKEN_HEADROOM_PERCENT:-25}"
HEXGENT_PLANNER_TIMEOUT_MS="${HEXGENT_PLANNER_TIMEOUT_MS:-10000}"
HEXGENT_FALLBACK_POLICY="${HEXGENT_FALLBACK_POLICY:-pd_policy}"
MOONCAKE_ENGINE_OVERLAY_DIR="${MOONCAKE_ENGINE_OVERLAY_DIR:-}"

PLANNER_PORT="${PLANNER_PORT:-30999}"
PREFILL0_PORT="${PREFILL0_PORT:-31000}"
PREFILL1_PORT="${PREFILL1_PORT:-31001}"
DECODE0_PORT="${DECODE0_PORT:-32000}"
DECODE1_PORT="${DECODE1_PORT:-32001}"
HEXGENT_ROUTER_PORT="${HEXGENT_ROUTER_PORT:-33000}"
BASELINE_ROUTER_PORT="${BASELINE_ROUTER_PORT:-33001}"
BOOTSTRAP0_PORT="${BOOTSTRAP0_PORT:-8998}"
BOOTSTRAP1_PORT="${BOOTSTRAP1_PORT:-8999}"
PREFILL0_GPU="${PREFILL0_GPU:-0}"
PREFILL1_GPU="${PREFILL1_GPU:-1}"
DECODE0_GPU="${DECODE0_GPU:-2}"
DECODE1_GPU="${DECODE1_GPU:-3}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
if [[ -f /workspace/hexgent_e2e/env.sh ]]; then
  # shellcheck source=/dev/null
  source /workspace/hexgent_e2e/env.sh
fi

export CARGO_HOME="${CARGO_HOME:-/workspace/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/workspace/.rustup}"
export PATH="$CARGO_HOME/bin:$PATH"
if [[ "${HEXGENT_USE_SOURCE_TREE:-0}" == "1" ]]; then
  export PYTHONPATH="$SRC_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
fi
export SGLANG_KV_TRANSFER_TELEMETRY=1
export SGLANG_HEXGENT_CUDA_BATCH_TELEMETRY=1
export MC_TE_METRIC=1
export MC_TE_METRIC_INTERVAL_SECONDS=1
export SGLANG_DISAGGREGATION_MAX_INFLIGHT_TRANSFERS="$DECODE_MAX_INFLIGHT_TRANSFERS"
export SGLANG_MOONCAKE_TRANSFER_PROTOCOL=nvlink_intra
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
export SGLANG_MOONCAKE_SEND_AUX_TCP=1
export MC_INTRANODE_NVLINK=1
export HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S="$TRANSFER_BANDWIDTH_BYTES_PER_S"
export HEXGENT_TRANSFER_STARTUP_MS="$TRANSFER_STARTUP_MS"
export HEXGENT_ENABLE_LOCAL_WAITING_MUTATIONS=0
if [[ -n "$MOONCAKE_ENGINE_OVERLAY_DIR" ]]; then
  export LD_LIBRARY_PATH="$MOONCAKE_ENGINE_OVERLAY_DIR/mooncake:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$MOONCAKE_ENGINE_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

stop_pid_file() {
  local pid_file="$1"
  if [[ -s "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
      for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

if [[ "${HEXGENT_RESTART:-1}" == "1" ]]; then
  for name in \
    router baseline_router \
    prefill prefill0 prefill1 \
    decode decode0 decode1; do
    stop_pid_file "$RUN_DIR/${name}.pid"
  done
fi

if [[ "${HEXGENT_RESTART_PLANNER:-0}" == "1" ]]; then
  stop_pid_file "$RUN_DIR/planner.pid"
fi

if ! curl -fsS "http://127.0.0.1:${PLANNER_PORT}/health" >/dev/null 2>&1; then
  python -m sglang.hexgent.sidecar \
    --host 127.0.0.1 \
    --port "$PLANNER_PORT" \
    --telemetry-path "$PLANNER_TELEMETRY_PATH" \
    --log-level warning \
    >"$LOG_DIR/planner.log" 2>&1 &
  echo $! >"$RUN_DIR/planner.pid"
fi

start_prefill() {
  local gpu="$1"
  local port="$2"
  local bootstrap_port="$3"
  local name="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$port" \
    --tp-size 1 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
    --disaggregation-bootstrap-port "$bootstrap_port" \
    --context-length "$CONTEXT_LENGTH" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-running-requests "$PREFILL_MAX_RUNNING_REQUESTS" \
    --prefill-max-requests "$PREFILL_MAX_BATCH_REQUESTS" \
    --chunked-prefill-size "$CONTEXT_LENGTH" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" --disable-radix-cache \
    --enable-priority-scheduling --schedule-low-priority-values-first \
    --enable-request-time-stats-logging \
    --trust-remote-code \
    >"$LOG_DIR/${name}.log" 2>&1 &
  echo $! >"$RUN_DIR/${name}.pid"
}

start_decode() {
  local gpu="$1"
  local port="$2"
  local name="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$port" \
    --tp-size 1 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
    --context-length "$CONTEXT_LENGTH" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-running-requests "$DECODE_MAX_RUNNING_REQUESTS" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" --disable-radix-cache \
    --enable-priority-scheduling --schedule-low-priority-values-first \
    --enable-request-time-stats-logging \
    --trust-remote-code \
    >"$LOG_DIR/${name}.log" 2>&1 &
  echo $! >"$RUN_DIR/${name}.pid"
}

wait_endpoint() {
  local name="$1"
  local url="$2"
  local log_file="$3"
  local pid_file="$4"
  for _ in $(seq 1 300); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name ready: $url"
      return 0
    fi
    if [[ -s "$pid_file" ]]; then
      local pid
      pid="$(cat "$pid_file")"
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "$name process exited. Tail of $log_file:" >&2
        tail -120 "$log_file" >&2
        return 1
      fi
    fi
    if grep -Eiq 'Scheduler hit an exception|Received sigquit|Traceback|Killed$|RuntimeError:' "$log_file"; then
      echo "$name failed. Tail of $log_file:" >&2
      tail -120 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  echo "$name did not become ready. Tail of $log_file:" >&2
  tail -120 "$log_file" >&2
  return 1
}

start_prefill "$PREFILL0_GPU" "$PREFILL0_PORT" "$BOOTSTRAP0_PORT" prefill0
start_prefill "$PREFILL1_GPU" "$PREFILL1_PORT" "$BOOTSTRAP1_PORT" prefill1
start_decode "$DECODE0_GPU" "$DECODE0_PORT" decode0
start_decode "$DECODE1_GPU" "$DECODE1_PORT" decode1

wait_endpoint prefill0 "http://127.0.0.1:${PREFILL0_PORT}/v1/models" "$LOG_DIR/prefill0.log" "$RUN_DIR/prefill0.pid"
wait_endpoint prefill1 "http://127.0.0.1:${PREFILL1_PORT}/v1/models" "$LOG_DIR/prefill1.log" "$RUN_DIR/prefill1.pid"
wait_endpoint decode0 "http://127.0.0.1:${DECODE0_PORT}/v1/models" "$LOG_DIR/decode0.log" "$RUN_DIR/decode0.pid"
wait_endpoint decode1 "http://127.0.0.1:${DECODE1_PORT}/v1/models" "$LOG_DIR/decode1.log" "$RUN_DIR/decode1.pid"

python -m sglang_router.launch_router \
  --host 127.0.0.1 \
  --port "$HEXGENT_ROUTER_PORT" \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:${PREFILL0_PORT}" "$BOOTSTRAP0_PORT" \
  --prefill "http://127.0.0.1:${PREFILL1_PORT}" "$BOOTSTRAP1_PORT" \
  --decode "http://127.0.0.1:${DECODE0_PORT}" \
  --decode "http://127.0.0.1:${DECODE1_PORT}" \
  --hexgent-router \
  --hexgent-planner-url "http://127.0.0.1:${PLANNER_PORT}" \
  --hexgent-planner-timeout-ms "$HEXGENT_PLANNER_TIMEOUT_MS" \
  --hexgent-fallback-policy "$HEXGENT_FALLBACK_POLICY" \
  --hexgent-admission-queue \
  --hexgent-admission-max-wait-ms "$HEXGENT_ADMISSION_MAX_WAIT_MS" \
  --hexgent-admission-poll-interval-ms "$HEXGENT_ADMISSION_POLL_INTERVAL_MS" \
  --hexgent-max-prefill-queue-per-worker "$HEXGENT_MAX_PREFILL_QUEUE_PER_WORKER" \
  --hexgent-prefill-batching-headroom "$HEXGENT_PREFILL_BATCHING_HEADROOM" \
  --hexgent-prefill-token-headroom-percent "$HEXGENT_PREFILL_TOKEN_HEADROOM_PERCENT" \
  >"$LOG_DIR/router.log" 2>&1 &
echo $! >"$RUN_DIR/router.pid"

python -m sglang_router.launch_router \
  --host 127.0.0.1 \
  --port "$BASELINE_ROUTER_PORT" \
  --prometheus-port 29001 \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:${PREFILL0_PORT}" "$BOOTSTRAP0_PORT" \
  --prefill "http://127.0.0.1:${PREFILL1_PORT}" "$BOOTSTRAP1_PORT" \
  --decode "http://127.0.0.1:${DECODE0_PORT}" \
  --decode "http://127.0.0.1:${DECODE1_PORT}" \
  >"$LOG_DIR/baseline_router.log" 2>&1 &
echo $! >"$RUN_DIR/baseline_router.pid"

wait_endpoint hexgent-router "http://127.0.0.1:${HEXGENT_ROUTER_PORT}/v1/models" "$LOG_DIR/router.log" "$RUN_DIR/router.pid"
wait_endpoint baseline-router "http://127.0.0.1:${BASELINE_ROUTER_PORT}/v1/models" "$LOG_DIR/baseline_router.log" "$RUN_DIR/baseline_router.pid"

echo "HexAgenT 2P2D smoke deployment is ready."
echo "Planner:        http://127.0.0.1:${PLANNER_PORT}"
echo "HexGENT router: http://127.0.0.1:${HEXGENT_ROUTER_PORT}"
echo "Baseline router:http://127.0.0.1:${BASELINE_ROUTER_PORT}"
