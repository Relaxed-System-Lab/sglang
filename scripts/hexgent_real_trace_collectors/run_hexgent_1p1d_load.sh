#!/usr/bin/env bash
set -euo pipefail

# Replay a fixed-length workflow trace through the real 1P1D deployment while
# preserving the telemetry needed to diagnose planner, queue, GPU, and KV
# transfer bottlenecks. The trace must contain fixed input/output token hints.

SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang}"
VENV_DIR="${HEXGENT_VENV_DIR:-/workspace/hexgent_sglang_venv}"
TRACE_PATH="${TRACE_PATH:?set TRACE_PATH to a fixed-length workflow JSONL}"
RUN_DIR="${RUN_DIR:?set RUN_DIR for this replay}"
LOG_DIR="${HEXGENT_LOG_DIR:-/workspace/hexgent_e2e/logs_1p1d_llama8b}"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:33100}"
MODEL="${MODEL:-llama8b-pd-1p1d}"
PREFILL_PORT="${PREFILL_PORT:-31100}"
DECODE_PORT="${DECODE_PORT:-31101}"
ARRIVAL_RATE="${ARRIVAL_RATE:-12}"
SEED="${SEED:-42}"
TIMEOUT_S="${TIMEOUT_S:-1200}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-0.2}"

mkdir -p "$RUN_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$SRC_DIR/python:$SRC_DIR/sgl-model-gateway/bindings/python/src${PYTHONPATH:+:$PYTHONPATH}"

snapshot_workers() {
  python - "$1" "$PREFILL_PORT" "$DECODE_PORT" <<'PY'
import json
import sys
import time
import urllib.request

output, prefill_port, decode_port = sys.argv[1:]
rows = []
for stage, port in (("prefill", prefill_port), ("decode", decode_port)):
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/hexgent/pd_snapshot", timeout=0.5
        ) as response:
            payload = json.load(response)
        error = None
    except Exception as exc:
        payload = None
        error = f"{type(exc).__name__}: {exc}"
    rows.append(
        {
            "timestamp_s": time.time(),
            "stage": stage,
            "snapshot": payload,
            "error": error,
        }
    )
with open(output, "a", encoding="utf-8") as sink:
    for row in rows:
        sink.write(json.dumps(row) + "\n")
PY
}

monitor_run() {
  while true; do
    printf '%s,' "$(date +%s.%N)" >>"$RUN_DIR/gpu.csv"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits | paste -sd ';' - >>"$RUN_DIR/gpu.csv"
    snapshot_workers "$RUN_DIR/snapshots.jsonl"
    sleep "$MONITOR_INTERVAL_S"
  done
}

stop_monitor() {
  kill "$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
}

TELEMETRY_PATH="$LOG_DIR/planner_invocations.jsonl"
TELEMETRY_START=0
if [[ -f "$TELEMETRY_PATH" ]]; then
  TELEMETRY_START="$(wc -l <"$TELEMETRY_PATH")"
fi
PREFILL_LOG_PATH="$LOG_DIR/prefill.log"
PREFILL_LOG_START=0
if [[ -f "$PREFILL_LOG_PATH" ]]; then
  PREFILL_LOG_START="$(wc -l <"$PREFILL_LOG_PATH")"
fi
DECODE_LOG_PATH="$LOG_DIR/decode.log"
DECODE_LOG_START=0
if [[ -f "$DECODE_LOG_PATH" ]]; then
  DECODE_LOG_START="$(wc -l <"$DECODE_LOG_PATH")"
fi

: >"$RUN_DIR/gpu.csv"
: >"$RUN_DIR/snapshots.jsonl"
monitor_run >"$RUN_DIR/monitor.log" 2>&1 &
MONITOR_PID=$!
STARTED_S="$(date +%s.%N)"
set +e
python -m sglang.hexgent.workflow_trace replay \
  --trace-path "$TRACE_PATH" \
  --router-url "$ROUTER_URL" \
  --model "$MODEL" \
  --arrival-rate "$ARRIVAL_RATE" \
  --seed "$SEED" \
  --timeout-s "$TIMEOUT_S" \
  --output-path "$RUN_DIR/calls.jsonl" \
  >"$RUN_DIR/replay.log" 2>&1
STATUS=$?
set -e
COMPLETED_S="$(date +%s.%N)"
stop_monitor "$MONITOR_PID"

if [[ -f "$TELEMETRY_PATH" ]]; then
  tail -n "+$((TELEMETRY_START + 1))" "$TELEMETRY_PATH" \
    >"$RUN_DIR/planner_invocations.jsonl"
fi
if [[ -f "$PREFILL_LOG_PATH" ]]; then
  tail -n "+$((PREFILL_LOG_START + 1))" "$PREFILL_LOG_PATH" \
    >"$RUN_DIR/prefill_worker.log"
fi
if [[ -f "$DECODE_LOG_PATH" ]]; then
  tail -n "+$((DECODE_LOG_START + 1))" "$DECODE_LOG_PATH" \
    >"$RUN_DIR/decode_worker.log"
fi

python - "$RUN_DIR/run.json" "$STATUS" "$STARTED_S" "$COMPLETED_S" <<'PY'
import json
import sys

path, status, started, completed = sys.argv[1:]
open(path, "w", encoding="utf-8").write(
    json.dumps(
        {
            "status": int(status),
            "started_s": float(started),
            "completed_s": float(completed),
            "elapsed_s": float(completed) - float(started),
        },
        indent=2,
    )
)
PY

exit "$STATUS"
