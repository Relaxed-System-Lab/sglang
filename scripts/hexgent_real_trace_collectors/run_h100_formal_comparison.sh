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
TRACE_ORDER="${TRACE_ORDER:-sharegpt,bfcl,hotpotqa}"
SHAREGPT_TRACE_PATH="${SHAREGPT_TRACE_PATH:-$TRACE_DIR/sharegpt_workflows50.jsonl}"
BFCL_TRACE_PATH="${BFCL_TRACE_PATH:-$TRACE_DIR/bfcl_workflows50.jsonl}"
HOTPOTQA_TRACE_PATH="${HOTPOTQA_TRACE_PATH:-$TRACE_DIR/hotpotqa_workflows20.jsonl}"
SHAREGPT_ARRIVAL_RATE="${SHAREGPT_ARRIVAL_RATE:-3.0}"
BFCL_ARRIVAL_RATE="${BFCL_ARRIVAL_RATE:-0.6}"
HOTPOTQA_ARRIVAL_RATE="${HOTPOTQA_ARRIVAL_RATE:-0.2}"
PREFILL_WORKER_COUNT="${PREFILL_WORKER_COUNT:-4}"
DECODE_WORKER_COUNT="${DECODE_WORKER_COUNT:-4}"
PREFILL_PORT_BASE="${PREFILL_PORT_BASE:-31000}"
DECODE_PORT_BASE="${DECODE_PORT_BASE:-32000}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-32}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-8}"
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
declare -A TRACE_PATHS=(
  [sharegpt]="$SHAREGPT_TRACE_PATH"
  [bfcl]="$BFCL_TRACE_PATH"
  [hotpotqa]="$HOTPOTQA_TRACE_PATH"
)

snapshot_workers() {
  local output="$1"
  python - "$output" "$PREFILL_WORKER_COUNT" "$DECODE_WORKER_COUNT" "$PREFILL_PORT_BASE" "$DECODE_PORT_BASE" <<'PY'
import json, sys, time, urllib.request
output = sys.argv[1]
prefill_count, decode_count, prefill_base, decode_base = map(int, sys.argv[2:])
rows = []
for stage, count, base in (
    ("prefill", prefill_count, prefill_base),
    ("decode", decode_count, decode_base),
):
    for index in range(count):
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

warmup_mode() {
  local trace="$1" mode="$2" router_url="$3"
  local output="$RUN_DIR/${trace}_${mode}_warmup.json"
  python - "$router_url" "$MODEL" "$mode" "$WARMUP_REQUESTS" \
    "$WARMUP_CONCURRENCY" "$output" <<'PY'
import concurrent.futures
import json
import sys
import time

import requests

router_url, model, mode, request_count, concurrency, output = sys.argv[1:]
request_count = int(request_count)
concurrency = int(concurrency)
started = time.perf_counter()


def issue(index):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"warmup request {index}. " + "warmup token " * 256,
            }
        ],
        "max_tokens": 64,
        "ignore_eos": True,
        "stream": False,
    }
    if mode == "hexgent":
        now_ms = int(time.time() * 1000)
        request_id = f"__hexgent_mode_warmup__-{now_ms}-{index}"
        payload["hexgent"] = {
            "workflow_id": request_id,
            "call_id": request_id,
            "workflow_arrival_ms": now_ms,
            "input_tokens": 520,
            "predicted_output_tokens": 64,
        }
    response = requests.post(
        f"{router_url}/v1/chat/completions", json=payload, timeout=180
    )
    response.raise_for_status()
    return {
        "index": index,
        "status": response.status_code,
        "completion_tokens": response.json().get("usage", {}).get("completion_tokens"),
    }


with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
    results = list(executor.map(issue, range(request_count)))

with open(output, "w", encoding="utf-8") as sink:
    json.dump(
        {
            "mode": mode,
            "request_count": request_count,
            "concurrency": concurrency,
            "elapsed_s": time.perf_counter() - started,
            "results": results,
        },
        sink,
        indent=2,
    )
PY
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
    --trace-path "${TRACE_PATHS[$trace]}" \
    --router-url "$router_url" --model "$MODEL" \
    --run-namespace "$RUN_ID::$trace::$mode" \
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
IFS=',' read -r -a traces <<<"$TRACE_ORDER"
for trace in "${traces[@]}"; do
  if [[ -z "${TRACE_PATHS[$trace]+x}" ]]; then
    echo "Unsupported TRACE_ORDER entry: $trace" >&2
    exit 2
  fi
  for mode in "${modes[@]}"; do
    case "$mode" in
      baseline) router_url="$BASELINE_ROUTER_URL" ;;
      hexgent) router_url="$HEXGENT_ROUTER_URL" ;;
      *) echo "Unsupported MODE_ORDER entry: $mode" >&2; exit 2 ;;
    esac
    warmup_mode "$trace" "$mode" "$router_url"
    run_one "$trace" "$mode" "$router_url"
  done
done

cp "$TRACE_DIR/trace_summary.json" "$RUN_DIR/trace_summary.json"
cp "$EXCLUSIVE_PROFILE" "$RUN_DIR/exclusive_profile.jsonl"
cp "$LOG_DIR"/*.log "$RUN_DIR/" 2>/dev/null || true
tar -czf "$RESULT_ROOT/hexgent_h100_4p4d_${RUN_ID}.tar.gz" -C "$RESULT_ROOT" "$RUN_ID"
echo "$RUN_DIR"
