#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-/workspace/hexgent_sglang_venv}"
SRC_DIR="${SRC_DIR:-/workspace/sglang}"
TRACE_DIR="${TRACE_DIR:-/workspace/hexgent_e2e/formal_traces}"
RESULT_DIR="${RESULT_DIR:-/workspace/hexgent_e2e/llama8b_exclusive_profile_20260722}"
ENDPOINT_URL="${ENDPOINT_URL:-http://127.0.0.1:30000/v1/chat/completions}"
MODEL="${MODEL:-hexgent-llama8b-fp8}"
WORKER_CLASS="${WORKER_CLASS:-H100_80G_TP1_LLAMA8B_FP8}"
SHAREGPT_TRACE_PATH="${SHAREGPT_TRACE_PATH:-$TRACE_DIR/sharegpt_workflows50.jsonl}"
BFCL_TRACE_PATH="${BFCL_TRACE_PATH:-$TRACE_DIR/bfcl_workflows50.jsonl}"
HOTPOTQA_TRACE_PATH="${HOTPOTQA_TRACE_PATH:-$TRACE_DIR/hotpotqa_workflows20.jsonl}"

export PATH="$VENV_DIR/bin:$PATH"
export PYTHONPATH="$SRC_DIR/python:$SRC_DIR/sgl-model-gateway/bindings/python/src:${PYTHONPATH:-}"

mkdir -p "$RESULT_DIR"
rm -f \
  "$RESULT_DIR"/raw_{sharegpt,bfcl,hotpotqa}.jsonl \
  "$RESULT_DIR"/prefill_{sharegpt,bfcl,hotpotqa}.jsonl \
  "$RESULT_DIR"/decode_{sharegpt,bfcl,hotpotqa}.jsonl \
  "$RESULT_DIR/profile_driver.status"

on_exit() {
  local status=$?
  printf '%s\n' "$status" >"$RESULT_DIR/profile_driver.status"
}
trap on_exit EXIT

for trace in sharegpt bfcl hotpotqa; do
  case "$trace" in
    sharegpt) trace_path="$SHAREGPT_TRACE_PATH" ;;
    bfcl) trace_path="$BFCL_TRACE_PATH" ;;
    hotpotqa) trace_path="$HOTPOTQA_TRACE_PATH" ;;
  esac
  python -m sglang.hexgent.service_profiler profile \
    --trace-path "$trace_path" \
    --endpoint-url "$ENDPOINT_URL" \
    --model "$MODEL" \
    --worker-class "$WORKER_CLASS" \
    --phase both \
    --prefill-output "$RESULT_DIR/prefill_${trace}.jsonl" \
    --decode-output "$RESULT_DIR/decode_${trace}.jsonl" \
    --raw-output "$RESULT_DIR/raw_${trace}.jsonl" \
    --trials 1 \
    --warmup-calls 3 \
    --timeout-s 3600
done
