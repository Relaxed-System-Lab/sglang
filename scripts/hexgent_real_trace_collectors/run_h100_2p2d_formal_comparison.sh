#!/usr/bin/env bash
set -euo pipefail

# Run the 2P+2D real-serving comparison. EXCLUSIVE_PROFILE is deliberately
# mandatory: it is a serial profile from this exact model/GPU/PD topology and
# is reduced to a workflow DAG critical path by the summary program.

SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang}"
VENV_DIR="${HEXGENT_VENV_DIR:-/workspace/hexgent_sglang_venv}"
TRACE_DIR="${TRACE_DIR:-/workspace/hexgent_e2e/formal_traces}"
LOG_DIR="${HEXGENT_LOG_DIR:-/workspace/hexgent_e2e/logs}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/hexgent_e2e/formal_results_2p2d}"
MODEL="${MODEL:-llama8b-pd-2p2d}"
HEXGENT_ROUTER_URL="${HEXGENT_ROUTER_URL:-http://127.0.0.1:33000}"
BASELINE_ROUTER_URL="${BASELINE_ROUTER_URL:-http://127.0.0.1:33001}"
EXCLUSIVE_PROFILE="${EXCLUSIVE_PROFILE:?set EXCLUSIVE_PROFILE to the merged 2P2D per-call profile JSONL}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RESULT_ROOT/$RUN_ID"
MODE_ORDER="${MODE_ORDER:-baseline,hexgent}"
TRACE_WORKFLOW_COUNT="${TRACE_WORKFLOW_COUNT:-10}"
SHAREGPT_ARRIVAL_RATE="${SHAREGPT_ARRIVAL_RATE:-3.0}"
BFCL_ARRIVAL_RATE="${BFCL_ARRIVAL_RATE:-0.6}"
HOTPOTQA_ARRIVAL_RATE="${HOTPOTQA_ARRIVAL_RATE:-0.2}"
TIMEOUT_S="${TIMEOUT_S:-1200}"

mkdir -p "$RUN_DIR"
source "$VENV_DIR/bin/activate"
if [[ "${HEXGENT_USE_SOURCE_TREE:-0}" == "1" ]]; then
  export PYTHONPATH="$SRC_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
fi

declare -A ARRIVAL_RATES=(
  [sharegpt]="$SHAREGPT_ARRIVAL_RATE"
  [bfcl]="$BFCL_ARRIVAL_RATE"
  [hotpotqa]="$HOTPOTQA_ARRIVAL_RATE"
)

snapshot_workers() {
  local output="$1"
  # Prefill workers: for port in 31000 31001. Decode workers: for port in 32000 32001.
  python - "$output" <<'PY'
import json
import sys
import time
import urllib.request

rows = []
for stage, ports in (("prefill", (31000, 31001)), ("decode", (32000, 32001))):
    for index, port in enumerate(ports):
        url = f"http://127.0.0.1:{port}/hexgent/pd_snapshot"
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                payload = json.load(response)
            error = None
        except Exception as exc:
            payload = None
            error = f"{type(exc).__name__}: {exc}"
        rows.append({"timestamp_ms": time.time() * 1000, "stage": stage,
                     "index": index, "url": url, "snapshot": payload,
                     "error": error})
with open(sys.argv[1], "a", encoding="utf-8") as sink:
    for row in rows:
        sink.write(json.dumps(row) + "\n")
PY
}

monitor_run() {
  local prefix="$1"
  while true; do
    date +%s.%N | tr '\n' ',' >>"${prefix}_gpu.csv"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw \
      --format=csv,noheader,nounits | paste -sd ';' - >>"${prefix}_gpu.csv"
    snapshot_workers "${prefix}_snapshots.jsonl"
    sleep 0.25
  done
}

stop_monitor() {
  kill "$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
}

record_log_positions() {
  local directory="$1"
  mkdir -p "$directory"
  for name in prefill0 prefill1 decode0 decode1 router baseline_router planner; do
    if [[ -f "$LOG_DIR/${name}.log" ]]; then
      wc -l <"$LOG_DIR/${name}.log" >"$directory/${name}.start_line"
    else
      echo 0 >"$directory/${name}.start_line"
    fi
  done
}

copy_new_log_lines() {
  local directory="$1"
  for name in prefill0 prefill1 decode0 decode1 router baseline_router planner; do
    local source="$LOG_DIR/${name}.log"
    [[ -f "$source" ]] || continue
    local start
    start=$(<"$directory/${name}.start_line")
    tail -n "+$((start + 1))" "$source" >"$directory/${name}.log"
  done
}

run_one() {
  local trace="$1" mode="$2" router_url="$3"
  local prefix="$RUN_DIR/${trace}_${mode}"
  local worker_logs="${prefix}_worker_logs"
  local planner_telemetry="$LOG_DIR/planner_invocations.jsonl"
  local planner_start=0
  [[ -f "$planner_telemetry" ]] && planner_start=$(wc -l <"$planner_telemetry")
  record_log_positions "$worker_logs"

  : >"${prefix}_gpu.csv"
  : >"${prefix}_snapshots.jsonl"
  monitor_run "$prefix" >"${prefix}_monitor.log" 2>&1 &
  local monitor_pid=$!
  local started
  started=$(date +%s.%N)
  set +e
  python -m sglang.hexgent.workflow_trace replay \
    --trace-path "$TRACE_DIR/${trace}_workflows${TRACE_WORKFLOW_COUNT}.jsonl" \
    --router-url "$router_url" --model "$MODEL" \
    --arrival-rate "${ARRIVAL_RATES[$trace]}" --seed 42 \
    --timeout-s "$TIMEOUT_S" --output-path "${prefix}_calls.jsonl" \
    >"${prefix}_replay.log" 2>&1
  local status=$?
  set -e
  local completed
  completed=$(date +%s.%N)
  stop_monitor "$monitor_pid"
  copy_new_log_lines "$worker_logs"
  if [[ -f "$planner_telemetry" ]]; then
    tail -n "+$((planner_start + 1))" "$planner_telemetry" \
      >"${prefix}_planner_invocations.jsonl"
  else
    : >"${prefix}_planner_invocations.jsonl"
  fi

  python "$SRC_DIR/scripts/hexgent_real_trace_collectors/summarize_scheduling_replay.py" \
    --replay "${prefix}_calls.jsonl" \
    --planner-telemetry "${prefix}_planner_invocations.jsonl" \
    --exclusive-profile "$EXCLUSIVE_PROFILE" \
    --output "${prefix}_summary.json" >"${prefix}_summary.log"
  python - "$prefix" "$status" "$started" "$completed" <<'PY'
import json
import sys

prefix, status, started, completed = sys.argv[1:]
path = prefix + "_summary.json"
data = json.load(open(path, encoding="utf-8"))
data.update({"process_status": int(status), "experiment_started_s": float(started),
             "experiment_completed_s": float(completed),
             "experiment_elapsed_s": float(completed) - float(started)})
open(path, "w", encoding="utf-8").write(json.dumps(data, indent=2) + "\n")
PY
  return "$status"
}

curl -fsS "$BASELINE_ROUTER_URL/v1/models" >/dev/null
curl -fsS "$HEXGENT_ROUTER_URL/v1/models" >/dev/null

IFS=',' read -r -a modes <<<"$MODE_ORDER"
for trace in sharegpt bfcl hotpotqa; do
  for mode in "${modes[@]}"; do
    case "$mode" in
      baseline) run_one "$trace" baseline "$BASELINE_ROUTER_URL" ;;
      hexgent) run_one "$trace" hexgent "$HEXGENT_ROUTER_URL" ;;
      *) echo "Unsupported MODE_ORDER entry: $mode" >&2; exit 2 ;;
    esac
  done
done

cp "$EXCLUSIVE_PROFILE" "$RUN_DIR/exclusive_profile.jsonl"
cp "$TRACE_DIR/trace_summary.json" "$RUN_DIR/trace_summary.json" 2>/dev/null || true
tar -czf "$RESULT_ROOT/hexgent_h100_2p2d_${RUN_ID}.tar.gz" -C "$RESULT_ROOT" "$RUN_ID"
echo "$RUN_DIR"
