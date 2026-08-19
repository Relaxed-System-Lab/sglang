#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang}"
TRACE_DIR="${TRACE_DIR:-/workspace/hexgent_e2e/formal_traces}"
SHAREGPT_TRACE_PATH="${SHAREGPT_TRACE_PATH:-$TRACE_DIR/sharegpt_workflows128.jsonl}"
SHAREGPT_WORKFLOW_COUNT="${SHAREGPT_WORKFLOW_COUNT:-128}"
SHAREGPT_RATE_STAIRCASE="${SHAREGPT_RATE_STAIRCASE:-3,6,9,12}"

python - "$SHAREGPT_TRACE_PATH" "$SHAREGPT_WORKFLOW_COUNT" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
workflow_ids = []
with open(path, encoding="utf-8") as source:
    for line in source:
        if not line.strip():
            continue
        row = json.loads(line)
        workflow_id = row.get("workflow_id")
        if workflow_id is not None:
            workflow_ids.append(str(workflow_id))

distinct = len(set(workflow_ids))
if distinct < expected:
    raise SystemExit(
        f"ShareGPT staircase needs at least {expected} distinct workflows; found {distinct}"
    )
PY

IFS=',' read -r -a rates <<<"$SHAREGPT_RATE_STAIRCASE"
for rate in "${rates[@]}"; do
  rate_id="${rate//./p}"
  TRACE_ORDER=sharegpt \
  MODE_ORDER=hexgent \
  SHAREGPT_TRACE_PATH="$SHAREGPT_TRACE_PATH" \
  SHAREGPT_ARRIVAL_RATE="$rate" \
  RUN_ID="sharegpt_rate_${rate_id}_$(date -u +%Y%m%dT%H%M%SZ)" \
    bash "$SRC_DIR/scripts/hexgent_real_trace_collectors/run_h100_formal_comparison.sh"
done
