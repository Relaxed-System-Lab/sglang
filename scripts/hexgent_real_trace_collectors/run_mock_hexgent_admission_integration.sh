#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang}"
VENV_DIR="${HEXGENT_ROUTER_TEST_VENV:-/root/hexgent-router-test-venv}"
RUN_DIR="${HEXGENT_MOCK_RUN_DIR:-/workspace/hexgent_admission_mock}"
export HEXGENT_REVISION_STATE_PATH="${HEXGENT_REVISION_STATE_PATH:-$RUN_DIR/planner_revision}"

source /root/.cargo/env
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$SRC_DIR/python:$SRC_DIR/sgl-model-gateway/bindings/python/src"
mkdir -p "$RUN_DIR"

stop_pid() {
  local file="$1"
  if [[ -s "$file" ]]; then
    local pid
    pid="$(cat "$file")"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.02
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    rm -f "$file"
  fi
}

cleanup() {
  for name in router planner workers; do
    stop_pid "$RUN_DIR/$name.pid"
  done
}

trap cleanup EXIT

for name in workers planner router; do
  stop_pid "$RUN_DIR/$name.pid"
done

python "$SRC_DIR/scripts/hexgent_real_trace_collectors/mock_hexgent_admission_workers.py" \
  --event-log "$RUN_DIR/events.jsonl" --busy-snapshot-count 3 \
  >"$RUN_DIR/workers.log" 2>&1 &
echo $! >"$RUN_DIR/workers.pid"

python "$SRC_DIR/scripts/hexgent_real_trace_collectors/run_hexgent_sidecar_overlay.py" \
  --host 127.0.0.1 --port 30999 --log-level warning \
  >"$RUN_DIR/planner.log" 2>&1 &
echo $! >"$RUN_DIR/planner.pid"

for _ in $(seq 1 50); do
  if curl -fsS http://127.0.0.1:31000/health >/dev/null 2>&1 && \
    curl -fsS http://127.0.0.1:30999/health >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

curl -fsS http://127.0.0.1:31000/health >/dev/null
curl -fsS http://127.0.0.1:30999/health >/dev/null

python -m sglang_router.launch_router \
  --host 127.0.0.1 --port 33000 \
  --pd-disaggregation \
  --prefill http://127.0.0.1:31000 8998 \
  --decode http://127.0.0.1:32000 \
  --hexgent-router \
  --hexgent-planner-url http://127.0.0.1:30999 \
  --hexgent-planner-timeout-ms 2000 \
  --hexgent-fallback-policy fail_closed \
  --hexgent-admission-queue \
  --hexgent-admission-max-wait-ms 10000 \
  --hexgent-admission-poll-interval-ms 20 \
  --hexgent-max-prefill-queue-per-worker 1 \
  --hexgent-prefill-batching-headroom 1 \
  >"$RUN_DIR/router.log" 2>&1 &
echo $! >"$RUN_DIR/router.pid"

for _ in $(seq 1 150); do
  if curl -fsS http://127.0.0.1:33000/v1/models >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$(cat "$RUN_DIR/router.pid")" 2>/dev/null; then
    tail -100 "$RUN_DIR/router.log" >&2
    exit 1
  fi
  sleep 0.2
done

curl -fsS http://127.0.0.1:33000/v1/models >/dev/null

curl -fsS http://127.0.0.1:33000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "hexgent-mock-model",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 8,
    "stream": false,
    "hexgent": {
      "workflow_id": "wf-live-refresh",
      "call_id": "call-0",
      "workflow_arrival_ms": 0,
      "workflow_horizon_ms": 10000,
      "input_tokens": 8,
      "predicted_output_tokens": 1
    }
  }' >"$RUN_DIR/response.json"

RUN_DIR="$RUN_DIR" python - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
prefill_snapshots = [
    event
    for event in events
    if event["event"] == "snapshot" and event["stage"] == "prefill"
]
dispatches = [event for event in events if event["event"] == "dispatch"]
assert len(prefill_snapshots) >= 4, prefill_snapshots
assert any(event["busy"] is False for event in prefill_snapshots), prefill_snapshots
first_dispatch = next(index for index, event in enumerate(events) if event["event"] == "dispatch")
first_available = next(
    index
    for index, event in enumerate(events)
    if event["event"] == "snapshot"
    and event["stage"] == "prefill"
    and event["busy"] is False
)
assert first_available < first_dispatch, events
assert {event["stage"] for event in dispatches} == {"prefill", "decode"}, dispatches
rooms = {event["bootstrap_room"] for event in dispatches}
assert len(rooms) == 1 and None not in rooms, dispatches
print(
    json.dumps(
        {
            "prefill_snapshot_polls": len(prefill_snapshots),
            "dispatches": dispatches,
            "response": json.loads((run / "response.json").read_text()),
        },
        indent=2,
    )
)
PY
