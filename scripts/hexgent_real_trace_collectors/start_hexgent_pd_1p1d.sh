#!/usr/bin/env bash
set -euo pipefail

# Start a real-model, one-prefill/one-decode H100 deployment. The global
# HexAgenT admission queue owns requests until the prefill worker has room;
# model execution and KV transfer remain native SGLang/Mooncake operations.

VENV_DIR="${HEXGENT_VENV_DIR:-/workspace/hexgent_sglang_venv}"
SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang}"
RUN_DIR="${HEXGENT_RUN_DIR:-/workspace/hexgent_e2e/run_1p1d_llama8b}"
LOG_DIR="${HEXGENT_LOG_DIR:-/workspace/hexgent_e2e/logs_1p1d_llama8b}"
MODEL_PATH="${MODEL_PATH:-/workspace/hub/models--neuralmagic--Meta-Llama-3.1-8B-Instruct-FP8/snapshots/12fd6884d2585dd4d020373e7f39f74507b31866}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama8b-pd-1p1d}"

PREFILL_PORT="${PREFILL_PORT:-31100}"
DECODE_PORT="${DECODE_PORT:-31101}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-9110}"
PLANNER_PORT="${PLANNER_PORT:-30998}"
ROUTER_PORT="${ROUTER_PORT:-33100}"
# Avoid the gateway's shared default (29000), which can be owned by an
# unrelated router process on long-lived development pods.
PROMETHEUS_PORT="${PROMETHEUS_PORT:-29100}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
# Keep sufficient headroom for a 32-request prefill batch without imposing the
# old four-request ceiling. KV capacity remains bounded by max-total-tokens.
PREFILL_MAX_RUNNING_REQUESTS="${PREFILL_MAX_RUNNING_REQUESTS:-64}"
PREFILL_MAX_BATCH_REQUESTS="${PREFILL_MAX_BATCH_REQUESTS:-32}"
# SGLang otherwise selects an 8K chunk for this GPU. The profiled HotpotQA
# inputs are about 1.6K--2.1K tokens, so 16K permits a real 7--10 request
# prefill batch without weakening the 32-request admission ceiling.
PREFILL_CHUNKED_PREFILL_SIZE="${PREFILL_CHUNKED_PREFILL_SIZE:-16384}"
# Decode requests are long lived. The model/KV token budget, not this request
# count, is the intended occupancy constraint during continuous batching.
DECODE_MAX_RUNNING_REQUESTS="${DECODE_MAX_RUNNING_REQUESTS:-256}"
DECODE_MAX_INFLIGHT_TRANSFERS="${DECODE_MAX_INFLIGHT_TRANSFERS:-8}"
HEXGENT_MAX_PREFILL_QUEUE_PER_WORKER="${HEXGENT_MAX_PREFILL_QUEUE_PER_WORKER:-64}"
HEXGENT_ADMISSION_MAX_WAIT_MS="${HEXGENT_ADMISSION_MAX_WAIT_MS:-300000}"
HEXGENT_ADMISSION_POLL_INTERVAL_MS="${HEXGENT_ADMISSION_POLL_INTERVAL_MS:-10}"
# Enqueue/status operations are bounded separately inside the router. A stale
# plan follows the native PD policy rather than turning a control-plane delay
# into an HTTP 503 for a ready request.
HEXGENT_PLANNER_TIMEOUT_MS="${HEXGENT_PLANNER_TIMEOUT_MS:-10000}"
HEXGENT_FALLBACK_POLICY="${HEXGENT_FALLBACK_POLICY:-pd_policy}"

mkdir -p "$RUN_DIR" "$LOG_DIR"
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$SRC_DIR/python:$SRC_DIR/sgl-model-gateway/bindings/python/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/lib:/workspace/third_party/Mooncake-v0.3.9/build-intra-nvlink/mooncake-asio${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SGLANG_MOONCAKE_TRANSFER_PROTOCOL=nvlink_intra
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
export MC_INTRANODE_NVLINK=1
# Auxiliary metadata is CPU-resident; only bulk KV uses the NVLink path.
export SGLANG_MOONCAKE_SEND_AUX_TCP=1
export SGLANG_KV_TRANSFER_TELEMETRY=1
export MC_TE_METRIC=1
export SGLANG_DISAGGREGATION_MAX_INFLIGHT_TRANSFERS="$DECODE_MAX_INFLIGHT_TRANSFERS"
# Do not force synchronization here: this is a throughput/occupancy run.
export SGLANG_HEXGENT_CUDA_BATCH_TELEMETRY=1
unset SGLANG_HEXGENT_CUDA_BATCH_TELEMETRY_SYNC

stop_pid_file() {
  local pid_file="$1"
  [[ -s "$pid_file" ]] || return 0
  local pid
  pid="$(cat "$pid_file")"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  rm -f "$pid_file"
}

# A result directory may change between runs. In that case the previous PID
# files are not under the new RUN_DIR, but the fixed localhost ports still
# identify this dedicated 1P1D deployment. Clear those listeners before
# starting so that a successful readiness probe cannot accidentally target an
# old service with different telemetry or timeout settings.
stop_listener_port() {
  local port="$1"
  local pids
  if command -v fuser >/dev/null 2>&1; then
    pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
  else
    # Minimal cloud images may not ship fuser. ss exposes the listener PID
    # when executed as root.
    pids="$(ss -ltnp "sport = :$port" 2>/dev/null | sed -nE 's/.*pid=([0-9]+).*/\1/p' | sort -u)"
  fi
  [[ -n "$pids" ]] || return 0

  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 20); do
    local alive=false
    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=true
        break
      fi
    done
    "$alive" || return 0
    sleep 1
  done
  kill -KILL $pids 2>/dev/null || true
}

for name in router planner prefill decode; do
  stop_pid_file "$RUN_DIR/$name.pid"
done
stop_listener_port "$ROUTER_PORT"
stop_listener_port "$PROMETHEUS_PORT"
stop_listener_port "$PLANNER_PORT"
stop_listener_port "$BOOTSTRAP_PORT"
stop_listener_port "$PREFILL_PORT"
stop_listener_port "$DECODE_PORT"

setsid python -m sglang.hexgent.sidecar \
  --host 127.0.0.1 --port "$PLANNER_PORT" --log-level warning \
  --telemetry-path "$LOG_DIR/planner_invocations.jsonl" \
  >"$LOG_DIR/planner.log" 2>&1 &
echo $! >"$RUN_DIR/planner.pid"

setsid env CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 --port "$PREFILL_PORT" --tp-size 1 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" --disaggregation-decode-tp 1 \
  --context-length "$CONTEXT_LENGTH" --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --max-running-requests "$PREFILL_MAX_RUNNING_REQUESTS" \
  --prefill-max-requests "$PREFILL_MAX_BATCH_REQUESTS" \
  --chunked-prefill-size "$PREFILL_CHUNKED_PREFILL_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" --disable-radix-cache \
  --enable-priority-scheduling \
  --enable-request-time-stats-logging --trust-remote-code \
  >"$LOG_DIR/prefill.log" 2>&1 &
echo $! >"$RUN_DIR/prefill.pid"

setsid env CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 --port "$DECODE_PORT" --tp-size 1 --base-gpu-id 0 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --disaggregation-prefill-pp 1 \
  --context-length "$CONTEXT_LENGTH" --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --max-running-requests "$DECODE_MAX_RUNNING_REQUESTS" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" --disable-radix-cache \
  --enable-priority-scheduling \
  --enable-request-time-stats-logging --trust-remote-code \
  >"$LOG_DIR/decode.log" 2>&1 &
echo $! >"$RUN_DIR/decode.pid"

wait_endpoint() {
  local name="$1" url="$2" log_file="$3" pid_file="$4"
  for _ in $(seq 1 360); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name ready: $url"
      return 0
    fi
    if [[ -s "$pid_file" ]] && ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      tail -120 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  tail -120 "$log_file" >&2
  return 1
}

wait_endpoint prefill "http://127.0.0.1:$PREFILL_PORT/v1/models" \
  "$LOG_DIR/prefill.log" "$RUN_DIR/prefill.pid"
wait_endpoint decode "http://127.0.0.1:$DECODE_PORT/v1/models" \
  "$LOG_DIR/decode.log" "$RUN_DIR/decode.pid"

setsid python -m sglang_router.launch_router \
  --host 127.0.0.1 --port "$ROUTER_PORT" \
  --prometheus-port "$PROMETHEUS_PORT" \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:$PREFILL_PORT" "$BOOTSTRAP_PORT" \
  --decode "http://127.0.0.1:$DECODE_PORT" \
  --hexgent-router --hexgent-planner-url "http://127.0.0.1:$PLANNER_PORT" \
  --hexgent-planner-timeout-ms "$HEXGENT_PLANNER_TIMEOUT_MS" --hexgent-fallback-policy "$HEXGENT_FALLBACK_POLICY" \
  --hexgent-admission-queue \
  --hexgent-admission-max-wait-ms "$HEXGENT_ADMISSION_MAX_WAIT_MS" \
  --hexgent-admission-poll-interval-ms "$HEXGENT_ADMISSION_POLL_INTERVAL_MS" \
  --hexgent-max-prefill-queue-per-worker "$HEXGENT_MAX_PREFILL_QUEUE_PER_WORKER" \
  >"$LOG_DIR/router.log" 2>&1 &
echo $! >"$RUN_DIR/router.pid"

wait_endpoint router "http://127.0.0.1:$ROUTER_PORT/v1/models" \
  "$LOG_DIR/router.log" "$RUN_DIR/router.pid"

cat <<EOF
HexAgenT 1P1D deployment is ready.
  Router:  http://127.0.0.1:$ROUTER_PORT
  Planner: http://127.0.0.1:$PLANNER_PORT
  Logs:    $LOG_DIR
EOF
