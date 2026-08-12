#!/usr/bin/env bash
set -euo pipefail

# Start a minimal local PD-disaggregated SGLang deployment with HexAgenT router
# integration. This is for functional online debugging; it intentionally uses a
# small model by default so it can run on a 4-GPU H100 pod.

VENV_DIR="${HEXGENT_VENV_DIR:-/workspace/hexgent_sglang_venv}"
SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang_hexgent_debug}"
RUN_DIR="${HEXGENT_RUN_DIR:-/workspace/hexgent_e2e/run}"
LOG_DIR="${HEXGENT_LOG_DIR:-/workspace/hexgent_e2e/logs}"

MODEL="${MODEL:-/workspace/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-hexgent-smoke-model}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"

PLANNER_PORT="${PLANNER_PORT:-30999}"
PREFILL_PORT="${PREFILL_PORT:-31000}"
DECODE_PORT="${DECODE_PORT:-32000}"
ROUTER_PORT="${ROUTER_PORT:-33000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"

PREFILL_CUDA_DEVICES="${PREFILL_CUDA_DEVICES:-0}"
DECODE_CUDA_DEVICES="${DECODE_CUDA_DEVICES:-1}"
PREFILL_TP="${PREFILL_TP:-1}"
DECODE_TP="${DECODE_TP:-1}"

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
export PYTHONPATH="$SRC_DIR/python:${PYTHONPATH:-}"

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
  stop_pid_file "$RUN_DIR/router.pid"
  stop_pid_file "$RUN_DIR/prefill.pid"
  stop_pid_file "$RUN_DIR/decode.pid"
fi

if ! curl -fsS "http://127.0.0.1:${PLANNER_PORT}/health" >/dev/null 2>&1; then
  python -m sglang.hexgent.sidecar \
    --host 127.0.0.1 \
    --port "$PLANNER_PORT" \
    --log-level warning \
    >"$LOG_DIR/planner.log" 2>&1 &
  echo $! >"$RUN_DIR/planner.pid"
fi

CUDA_VISIBLE_DEVICES="$PREFILL_CUDA_DEVICES" python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$PREFILL_PORT" \
  --tp-size "$PREFILL_TP" \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
  --disaggregation-decode-tp "$DECODE_TP" \
  --context-length "$CONTEXT_LENGTH" \
  --max-running-requests 4 \
  --mem-fraction-static 0.70 \
  --trust-remote-code \
  >"$LOG_DIR/prefill.log" 2>&1 &
echo $! >"$RUN_DIR/prefill.pid"

CUDA_VISIBLE_DEVICES="$DECODE_CUDA_DEVICES" python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$DECODE_PORT" \
  --tp-size "$DECODE_TP" \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
  --disaggregation-prefill-pp 1 \
  --context-length "$CONTEXT_LENGTH" \
  --max-running-requests 4 \
  --mem-fraction-static 0.70 \
  --trust-remote-code \
  >"$LOG_DIR/decode.log" 2>&1 &
echo $! >"$RUN_DIR/decode.pid"

wait_endpoint() {
  local name="$1"
  local url="$2"
  local log_file="$3"
  local pid_file="$4"
  for _ in $(seq 1 240); do
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

wait_endpoint prefill "http://127.0.0.1:${PREFILL_PORT}/v1/models" "$LOG_DIR/prefill.log" "$RUN_DIR/prefill.pid"
wait_endpoint decode "http://127.0.0.1:${DECODE_PORT}/v1/models" "$LOG_DIR/decode.log" "$RUN_DIR/decode.pid"

python -m sglang_router.launch_router \
  --host 127.0.0.1 \
  --port "$ROUTER_PORT" \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:${PREFILL_PORT}" "$BOOTSTRAP_PORT" \
  --decode "http://127.0.0.1:${DECODE_PORT}" \
  --hexgent-router \
  --hexgent-planner-url "http://127.0.0.1:${PLANNER_PORT}" \
  --hexgent-planner-timeout-ms 2000 \
  --hexgent-fallback-policy pd_policy \
  >"$LOG_DIR/router.log" 2>&1 &
echo $! >"$RUN_DIR/router.pid"

wait_endpoint router "http://127.0.0.1:${ROUTER_PORT}/v1/models" "$LOG_DIR/router.log" "$RUN_DIR/router.pid"

echo "HexAgenT PD smoke deployment is ready."
echo "Planner: http://127.0.0.1:${PLANNER_PORT}"
echo "Router:  http://127.0.0.1:${ROUTER_PORT}"
