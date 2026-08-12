#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${SRC_DIR:-/workspace/sglang}"
ENV_FILE="${ENV_FILE:-/workspace/hexgent_e2e/env.sh}"
TRACE_DIR="${TRACE_DIR:-/workspace/hexgent_assets_curated/traces/formal_traces_20260804}"
RESULT_DIR="${RESULT_DIR:-/workspace/hexgent_e2e/a100_llama8b_bf16_profile_50_50_20_20260805}"
SHAREGPT_TRACE_PATH="${SHAREGPT_TRACE_PATH:-$TRACE_DIR/sharegpt_workflows50_valid.jsonl}"
BFCL_TRACE_PATH="${BFCL_TRACE_PATH:-$TRACE_DIR/bfcl_workflows50.jsonl}"
HOTPOTQA_TRACE_PATH="${HOTPOTQA_TRACE_PATH:-$TRACE_DIR/hotpotqa_workflows20.jsonl}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
SERVED_MODEL="${SERVED_MODEL:-hexgent-llama8b-a100-bf16}"
WORKER_CLASS="${WORKER_CLASS:-A100_SXM4_80GB_TP1_LLAMA8B_BF16}"
PORT="${PORT:-30000}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
TRACE_ORDER="${TRACE_ORDER:-sharegpt,bfcl,hotpotqa}"

source "$ENV_FILE"
export PYTHONPATH="$SRC_DIR/python:$SRC_DIR/sgl-model-gateway/bindings/python/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$RESULT_DIR"
if compgen -G "$RESULT_DIR/raw_*.jsonl" >/dev/null; then
  echo "Refusing to overwrite an existing profile in $RESULT_DIR" >&2
  exit 2
fi

server_pid=""
profile_pid=""
cleanup() {
  status=$?
  printf '%s\n' "$status" >"$RESULT_DIR/profile_driver.status"
  if [[ -n "$profile_pid" ]] && kill -0 "$profile_pid" 2>/dev/null; then
    kill "$profile_pid" || true
    wait "$profile_pid" || true
  fi
  if [[ -n "$server_pid" ]] && kill -0 -- -"$server_pid" 2>/dev/null; then
    kill -- -"$server_pid" || true
    for _ in $(seq 1 10); do
      kill -0 -- -"$server_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- -"$server_pid" 2>/dev/null; then
      kill -KILL -- -"$server_pid" || true
    fi
    wait "$server_pid" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

radix_cache_args=()
if [[ "$DISABLE_RADIX_CACHE" == "1" ]]; then
  radix_cache_args=(--disable-radix-cache)
fi

setsid python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --tp-size 1 \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --mem-fraction-static 0.85 \
  --max-running-requests 256 \
  "${radix_cache_args[@]}" \
  >"$RESULT_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$RESULT_DIR/server.pid"

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "SGLang server exited before becoming healthy" >&2
    tail -n 200 "$RESULT_DIR/server.log" >&2
    exit 1
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

IFS=',' read -r -a traces <<<"$TRACE_ORDER"
profile_parts=()
for trace in "${traces[@]}"; do
  case "$trace" in
    sharegpt) trace_path="$SHAREGPT_TRACE_PATH" ;;
    bfcl) trace_path="$BFCL_TRACE_PATH" ;;
    hotpotqa) trace_path="$HOTPOTQA_TRACE_PATH" ;;
    *) echo "Unknown trace in TRACE_ORDER: $trace" >&2; exit 2 ;;
  esac

  python -m sglang.hexgent.service_profiler profile \
    --trace-path "$trace_path" \
    --endpoint-url "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$SERVED_MODEL" \
    --worker-class "$WORKER_CLASS" \
    --phase both \
    --prefill-output "$RESULT_DIR/prefill_${trace}.jsonl" \
    --decode-output "$RESULT_DIR/decode_${trace}.jsonl" \
    --raw-output "$RESULT_DIR/raw_${trace}.jsonl" \
    --trials 1 \
    --warmup-calls 3 \
    --timeout-s 3600 &
  profile_pid=$!
  wait "$profile_pid"
  profile_pid=""
  profile_parts+=("$RESULT_DIR/raw_${trace}.jsonl")
done

cat "${profile_parts[@]}" >"$RESULT_DIR/exclusive_profile.jsonl"

python - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
for path in sorted(result_dir.glob("raw_*.jsonl")):
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    errors = sum(bool(row.get("error")) for row in rows)
    print(f"{path.name}: rows={len(rows)} errors={errors}")
    if errors:
        raise SystemExit(f"profile contains errors: {path}")
PY
