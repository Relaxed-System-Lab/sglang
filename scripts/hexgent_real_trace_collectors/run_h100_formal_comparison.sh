#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang}"
VENV_DIR="${HEXGENT_VENV_DIR:-/workspace/hexgent_sglang_venv}"
TRACE_DIR="${TRACE_DIR:-/workspace/hexgent_e2e/formal_traces}"
LOG_DIR="${HEXGENT_LOG_DIR:-/workspace/hexgent_e2e/logs_4p4d}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/hexgent_e2e/formal_results}"
# The replay model must match the served model registered in the router.  Keep
# an explicit MODEL override for real-model runs, but default to the same name
# used by the local 4P4D launcher.
MODEL="${MODEL:-${SERVED_MODEL_NAME:-hexgent-smoke-model}}"
HEXGENT_ROUTER_URL="${HEXGENT_ROUTER_URL:-http://127.0.0.1:33000}"
BASELINE_ROUTER_URL="${BASELINE_ROUTER_URL:-http://127.0.0.1:33001}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RESULT_ROOT/$RUN_ID"
MODE_ORDER="${MODE_ORDER:-baseline,hexgent}"
TRACE_WORKFLOW_COUNT="${TRACE_WORKFLOW_COUNT:-10}"
SHAREGPT_ARRIVAL_RATE="${SHAREGPT_ARRIVAL_RATE:-3.0}"
BFCL_ARRIVAL_RATE="${BFCL_ARRIVAL_RATE:-0.6}"
HOTPOTQA_ARRIVAL_RATE="${HOTPOTQA_ARRIVAL_RATE:-0.2}"
PROFILE_PATH="${PROFILE_PATH:-}"
# Replay length overrides and the SLO denominator are intentionally separate.
# The latter must be an isolated profile from this exact model/GPU/PD topology.
EXCLUSIVE_PROFILE="${EXCLUSIVE_PROFILE:?set EXCLUSIVE_PROFILE to the merged 4P4D per-call profile JSONL}"

PROFILE_ARGS=()
if [[ -n "$PROFILE_PATH" ]]; then
  PROFILE_ARGS=(--profile-path "$PROFILE_PATH")
fi

mkdir -p "$RUN_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$SRC_DIR/python:${PYTHONPATH:-}"

declare -A ARRIVAL_RATES=(
  [sharegpt]="$SHAREGPT_ARRIVAL_RATE"
  [bfcl]="$BFCL_ARRIVAL_RATE"
  [hotpotqa]="$HOTPOTQA_ARRIVAL_RATE"
)

snapshot_workers() {
  local output="$1"
  python - "$output" <<'PY'
import json, sys, time, urllib.request
output = sys.argv[1]
rows = []
for stage, base in (("prefill", 31000), ("decode", 32000)):
    for index in range(4):
        url = f"http://127.0.0.1:{base + index}/hexgent/pd_snapshot"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.load(response)
            error = None
        except Exception as exc:
            payload = None
            error = f"{type(exc).__name__}: {exc}"
        rows.append({"timestamp_ms": time.time() * 1000, "stage": stage,
                     "index": index, "url": url, "snapshot": payload,
                     "error": error})
with open(output, "a", encoding="utf-8") as sink:
    for row in rows:
        sink.write(json.dumps(row) + "\n")
PY
}

monitor_run() {
  local prefix="$1"
  while true; do
    date +%s.%N | tr '\n' ',' >>"${prefix}_gpu.csv"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits | paste -sd ';' - >>"${prefix}_gpu.csv"
    snapshot_workers "${prefix}_snapshots.jsonl"
    sleep 0.5
  done
}

stop_monitor() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

run_one() {
  local trace="$1" mode="$2" router_url="$3"
  local prefix="$RUN_DIR/${trace}_${mode}"
  local telemetry="$LOG_DIR/planner_invocations.jsonl"
  local telemetry_start
  telemetry_start=$(wc -l <"$telemetry")

  : >"${prefix}_snapshots.jsonl"
  : >"${prefix}_gpu.csv"
  monitor_run "$prefix" &
  local monitor_pid=$!
  local started
  started=$(date +%s.%N)
  set +e
  python -m sglang.hexgent.workflow_trace replay \
    --trace-path "$TRACE_DIR/${trace}_workflows${TRACE_WORKFLOW_COUNT}.jsonl" \
    --router-url "$router_url" --model "$MODEL" \
    "${PROFILE_ARGS[@]}" \
    --arrival-rate "${ARRIVAL_RATES[$trace]}" --seed 42 \
    --timeout-s 1200 --output-path "${prefix}_calls.jsonl" \
    >"${prefix}_replay.log" 2>&1
  local status=$?
  set -e
  local completed
  completed=$(date +%s.%N)
  stop_monitor "$monitor_pid"
  tail -n "+$((telemetry_start + 1))" "$telemetry" \
    >"${prefix}_planner_invocations.jsonl"

  python "$SRC_DIR/scripts/hexgent_real_trace_collectors/summarize_scheduling_replay.py" \
    --replay "${prefix}_calls.jsonl" \
    --planner-telemetry "${prefix}_planner_invocations.jsonl" \
    --exclusive-profile "$EXCLUSIVE_PROFILE" \
    --output "${prefix}_summary.json" \
    >"${prefix}_summary.log"
  python - "$prefix" "$status" "$started" "$completed" <<'PY'
import json, sys
prefix, status, started, completed = sys.argv[1:]
path = prefix + "_summary.json"
data = json.load(open(path, encoding="utf-8"))
data["process_status"] = int(status)
data["experiment_started_s"] = float(started)
data["experiment_completed_s"] = float(completed)
data["experiment_elapsed_s"] = float(completed) - float(started)
open(path, "w", encoding="utf-8").write(json.dumps(data, indent=2))
PY
  return "$status"
}

# Warm up model execution and Mooncake bootstrap before measured runs.
curl -fsS "$BASELINE_ROUTER_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":8}" \
  >"$RUN_DIR/warmup.json"

# The Rust router registers workers asynchronously after /v1/models becomes
# ready. Exercise the HexAgenT path before measurement so the first measured
# request cannot race an empty model-index snapshot.
python - "$HEXGENT_ROUTER_URL" "$MODEL" "$RUN_DIR/hexgent_warmup.json" <<'PY'
import json
import sys
import time

import requests

router_url, model, output_path = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "hexgent readiness probe"}],
    "max_tokens": 1,
    "hexgent": {
        "workflow_id": "__hexgent_readiness_probe__",
        "call_id": "__hexgent_readiness_probe__",
        "workflow_arrival_ms": int(time.time() * 1000),
        "workflow_horizon_ms": 60000,
        "input_tokens": 4,
        "predicted_output_tokens": 1,
    },
}
last = None
probe_prefix = f"__hexgent_readiness_probe__-{int(time.time() * 1000)}"
for attempt in range(120):
    # A failed PD/bootstrap attempt may leave a short-lived room on a worker.
    # Never retry with the same identity, otherwise Mooncake can reject the
    # retry as a duplicate metadata/bootstrap allocation.
    probe_id = f"{probe_prefix}-{attempt}"
    payload["hexgent"]["workflow_id"] = probe_id
    payload["hexgent"]["call_id"] = probe_id
    try:
        response = requests.post(
            f"{router_url}/v1/chat/completions", json=payload, timeout=5
        )
        if response.ok:
            with open(output_path, "w", encoding="utf-8") as sink:
                json.dump(
                    {"status": response.status_code, "response": response.json()},
                    sink,
                )
            break
        last = f"HTTP {response.status_code}: {response.text[:300]}"
    except Exception as exc:
        last = f"{type(exc).__name__}: {exc}"
    time.sleep(1)
else:
    raise SystemExit(f"HexAgenT readiness probe failed: {last}")
PY

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

cp "$TRACE_DIR/trace_summary.json" "$RUN_DIR/trace_summary.json"
cp "$EXCLUSIVE_PROFILE" "$RUN_DIR/exclusive_profile.jsonl"
cp "$LOG_DIR"/*.log "$RUN_DIR/" 2>/dev/null || true
tar -czf "$RESULT_ROOT/hexgent_h100_4p4d_${RUN_ID}.tar.gz" -C "$RESULT_ROOT" "$RUN_ID"
echo "$RUN_DIR"
