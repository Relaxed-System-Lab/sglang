#!/usr/bin/env bash
set -euo pipefail

# Replace this value or pass HF_TOKEN in the environment.
HF_TOKEN="${HF_TOKEN:-hf_REPLACE_WITH_YOUR_TOKEN}"

GPU_KIND="${GPU_KIND:-A100_80G}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-31000}"
HOST="${HOST:-0.0.0.0}"
TRIALS="${TRIALS:-3}"
WARMUP_CALLS="${WARMUP_CALLS:-1}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
MAX_CALLS="${MAX_CALLS:-}"

PROFILE_DIR="${PROFILE_DIR:-/workspace/profiles/${GPU_KIND}}"
HF_HOME="${HF_HOME:-/workspace/hf_cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf_cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/hf_cache/hub}"

LLAMA_MODEL="${LLAMA_MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
LLAMA_TP="${LLAMA_TP:-4}"
LLAMA_CUDA_DEVICES="${LLAMA_CUDA_DEVICES:-0,1,2,3}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-235B-A22B-FP8}"
QWEN_TP="${QWEN_TP:-4}"
QWEN_CUDA_DEVICES="${QWEN_CUDA_DEVICES:-0,1,2,3}"
QWEN_EXTRA_ARGS="${QWEN_EXTRA_ARGS:---reasoning-parser qwen3 --moe-runner-backend cutlass --disable-piecewise-cuda-graph --disable-cuda-graph}"

# Format: trace_name:/path/to/converted_workflow_trace.jsonl
# Override this env var if your three traces use different names or paths.
TRACE_SPECS="${TRACE_SPECS:-trace1:/workspace/data/trace1_workflows.jsonl,trace2:/workspace/data/trace2_workflows.jsonl,trace3:/workspace/data/trace3_workflows.jsonl}"
AUTO_CONVERT_TRACES="${AUTO_CONVERT_TRACES:-1}"
TRACE_MAX_WORKFLOWS="${TRACE_MAX_WORKFLOWS:-100}"

export HF_HOME TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
mkdir -p "$PROFILE_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

if [[ "$HF_TOKEN" != "hf_REPLACE_WITH_YOUR_TOKEN" ]]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null
else
  echo "HF_TOKEN is still the placeholder. Set HF_TOKEN before running if the model is gated."
fi

sanitize_name() {
  echo "$1" | tr '/:.' '___'
}

ensure_default_traces() {
  if [[ "$AUTO_CONVERT_TRACES" != "1" ]]; then
    return 0
  fi

  mkdir -p /workspace/data
  if [[ ! -f /workspace/data/sharegpt_workflows.jsonl ]]; then
    python profile_traces.py convert \
      --input-format sharegpt_hf \
      --output-path /workspace/data/sharegpt_workflows.jsonl \
      --max-calls "$TRACE_MAX_WORKFLOWS" \
      --hf-cache-dir /workspace/data/hf_trace_cache
  fi
  if [[ ! -f /workspace/data/bfcl_workflows.jsonl ]]; then
    python profile_traces.py convert \
      --input-format bfcl_hf \
      --output-path /workspace/data/bfcl_workflows.jsonl \
      --max-calls "$TRACE_MAX_WORKFLOWS" \
      --hf-cache-dir /workspace/data/hf_trace_cache
  fi
  if [[ ! -f /workspace/data/hotpotqa_workflows.jsonl ]]; then
    python profile_traces.py convert \
      --input-format hotpotqa_hf \
      --output-path /workspace/data/hotpotqa_workflows.jsonl \
      --max-calls "$TRACE_MAX_WORKFLOWS"
  fi

  if [[ "$TRACE_SPECS" == "trace1:/workspace/data/trace1_workflows.jsonl,trace2:/workspace/data/trace2_workflows.jsonl,trace3:/workspace/data/trace3_workflows.jsonl" ]]; then
    TRACE_SPECS="sharegpt:/workspace/data/sharegpt_workflows.jsonl,bfcl:/workspace/data/bfcl_workflows.jsonl,hotpotqa:/workspace/data/hotpotqa_workflows.jsonl"
  fi
}

wait_for_server() {
  local base_url="$1"
  local deadline_seconds="${SERVER_WAIT_S:-2400}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "${base_url}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start > deadline_seconds )); then
      echo "Timed out waiting for ${base_url}" >&2
      return 1
    fi
    sleep 10
  done
}

ACTIVE_SERVER_PID=""

stop_server() {
  if [[ -n "$ACTIVE_SERVER_PID" ]] && kill -0 "$ACTIVE_SERVER_PID" >/dev/null 2>&1; then
    kill "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
    wait "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
  fi
  ACTIVE_SERVER_PID=""
}

run_model_suite() {
  local model="$1"
  local tp="$2"
  local cuda_devices="$3"
  local extra_args="$4"
  local model_key
  local worker_class
  local server_log
  local server_pid

  model_key="$(sanitize_name "$model")"
  worker_class="${GPU_KIND}_TP${tp}"
  server_log="${PROFILE_DIR}/server_${model_key}_${worker_class}.log"

  export CUDA_VISIBLE_DEVICES="$cuda_devices"

  echo "Starting ${model} on ${GPU_KIND}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, tp=${tp}"
  # shellcheck disable=SC2086
  python -m sglang.launch_server \
    --model-path "$model" \
    --tp "$tp" \
    --host "$HOST" \
    --port "$PORT" \
    $extra_args >"$server_log" 2>&1 &
  ACTIVE_SERVER_PID="$!"
  trap 'stop_server' EXIT

  wait_for_server "http://127.0.0.1:${PORT}"

  IFS=',' read -ra traces <<< "$TRACE_SPECS"
  for trace_spec in "${traces[@]}"; do
    local trace_name="${trace_spec%%:*}"
    local trace_path="${trace_spec#*:}"
    local max_call_args=()

    if [[ ! -f "$trace_path" ]]; then
      echo "Missing trace file: ${trace_path}" >&2
      stop_server
      return 1
    fi
    if [[ -n "$MAX_CALLS" ]]; then
      max_call_args=(--max-calls "$MAX_CALLS")
    fi

    echo "Profiling model=${model} trace=${trace_name} gpu=${GPU_KIND}"
    python profile_traces.py profile \
      --trace-path "$trace_path" \
      --endpoint-url "http://127.0.0.1:${PORT}/v1/chat/completions" \
      --model "$model" \
      --worker-class "$worker_class" \
      --phase both \
      --prefill-output "${PROFILE_DIR}/prefill_${trace_name}_${model_key}_${worker_class}.jsonl" \
      --decode-output "${PROFILE_DIR}/decode_${trace_name}_${model_key}_${worker_class}.jsonl" \
      --raw-output "${PROFILE_DIR}/raw_${trace_name}_${model_key}_${worker_class}.jsonl" \
      --trials "$TRIALS" \
      --warmup-calls "$WARMUP_CALLS" \
      --timeout-s "$TIMEOUT_S" \
      "${max_call_args[@]}"
  done

  stop_server
  trap - EXIT
}

ensure_default_traces

run_model_suite "$LLAMA_MODEL" "$LLAMA_TP" "$LLAMA_CUDA_DEVICES" "$LLAMA_EXTRA_ARGS"
run_model_suite "$QWEN_MODEL" "$QWEN_TP" "$QWEN_CUDA_DEVICES" "$QWEN_EXTRA_ARGS"

cat "${PROFILE_DIR}"/prefill_*.jsonl > "${PROFILE_DIR}/prefill_all.jsonl"
cat "${PROFILE_DIR}"/decode_*.jsonl > "${PROFILE_DIR}/decode_all.jsonl"

python profile_traces.py fit-latency-db \
  --prefill-path "${PROFILE_DIR}/prefill_all.jsonl" \
  --decode-path "${PROFILE_DIR}/decode_all.jsonl" \
  --output-path "${PROFILE_DIR}/latency_db_measured_${GPU_KIND}.json"

echo "Wrote ${PROFILE_DIR}/latency_db_measured_${GPU_KIND}.json"
