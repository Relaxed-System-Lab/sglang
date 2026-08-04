#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR=${PROFILE_DIR:-/workspace/hexgent_profiles/H100_QWEN_FP8_TP8}
ARCHIVE_PATH=${ARCHIVE_PATH:-/workspace/hexgent_profiles/qwen_real_traces_H100_TP8.tar.gz}
MODEL=${MODEL:-Qwen/Qwen3-235B-A22B-FP8}
SERVED_MODEL=${SERVED_MODEL:-$MODEL}
BFCL_MODEL=${BFCL_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FC}
MODEL_CACHE_SLUG=${MODEL_CACHE_SLUG:-models--${MODEL//\//--}}
WORKER_CLASS=${WORKER_CLASS:-H100_TP8}
PORT=${PORT:-8200}
TP=${TP:-8}
EP_SIZE=${EP_SIZE:-}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
VENV=${VENV:-/workspace/hexgent_sglang_venv}
LATS_VENV=${LATS_VENV:-/workspace/hexgent_lats_venv}

mkdir -p "$PROFILE_DIR" /workspace/hub

activate_sglang() {
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  export HF_HOME=/workspace
  export HUGGINGFACE_HUB_CACHE=/workspace/hub
  export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
  export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
  export PYTHONUNBUFFERED=1
  export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
  local venv_libs
  venv_libs=$(python -c 'import glob, os, site; print(":".join(p for root in site.getsitepackages() for p in glob.glob(os.path.join(root, "nvidia", "*", "lib")) if os.path.isdir(p)))')
  export LD_LIBRARY_PATH="$venv_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

wait_for_download() {
  activate_sglang
  if [ -f "$PROFILE_DIR/qwen_download.pid" ] && kill -0 "$(cat "$PROFILE_DIR/qwen_download.pid")" 2>/dev/null; then
    echo "[qwen-suite] waiting for existing download pid=$(cat "$PROFILE_DIR/qwen_download.pid")"
    while kill -0 "$(cat "$PROFILE_DIR/qwen_download.pid")" 2>/dev/null; do
      sleep 30
    done
  fi
  local attempt=1
  while ! download_complete; do
    echo "[qwen-suite] downloading $MODEL attempt=$attempt"
    hf download "$MODEL" --cache-dir /workspace/hub
    attempt=$((attempt + 1))
  done
  echo "[qwen-suite] download complete for $MODEL"
}

download_complete() {
  find "/workspace/hub/${MODEL_CACHE_SLUG}/snapshots" -maxdepth 1 -mindepth 1 -type d >/dev/null 2>&1 || return 1
  if find "/workspace/hub/${MODEL_CACHE_SLUG}" -type f -name '*.incomplete' 2>/dev/null | grep -q .; then
    return 1
  fi
  return 0
}

find_snapshot() {
  find "/workspace/hub/${MODEL_CACHE_SLUG}/snapshots" -maxdepth 1 -mindepth 1 -type d | head -n 1
}

server_ready() {
  curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}

start_server() {
  activate_sglang
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
  if [ -f "$PROFILE_DIR/server.pid" ] && kill -0 "$(cat "$PROFILE_DIR/server.pid")" 2>/dev/null && server_ready; then
    echo "[qwen-suite] server already ready pid=$(cat "$PROFILE_DIR/server.pid")"
    return
  fi
  if [ -f "$PROFILE_DIR/server.pid" ]; then
    kill -TERM "$(cat "$PROFILE_DIR/server.pid")" 2>/dev/null || true
    sleep 5
    kill -KILL "$(cat "$PROFILE_DIR/server.pid")" 2>/dev/null || true
  fi
  local ep_args=()
  if [ -n "$EP_SIZE" ]; then
    ep_args=(--ep-size "$EP_SIZE")
  fi
  local extra_args=()
  if [ -n "$SERVER_EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    extra_args=($SERVER_EXTRA_ARGS)
  fi
  echo "[qwen-suite] starting server $MODEL tp=$TP ep_size=${EP_SIZE:-none} context=$CONTEXT_LENGTH port=$PORT extra_args=${SERVER_EXTRA_ARGS:-none}"
  nohup python -u -m sglang.launch_server \
    --model-path "$MODEL" \
    --context-length "$CONTEXT_LENGTH" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tp "$TP" \
    "${ep_args[@]}" \
    --reasoning-parser qwen3 \
    "${extra_args[@]}" \
    > "$PROFILE_DIR/server_qwen.log" 2>&1 &
  echo $! > "$PROFILE_DIR/server.pid"

  for _ in $(seq 1 720); do
    if server_ready; then
      echo "[qwen-suite] server ready"
      return
    fi
    if ! kill -0 "$(cat "$PROFILE_DIR/server.pid")" 2>/dev/null; then
      echo "[qwen-suite] server exited early"
      tail -n 240 "$PROFILE_DIR/server_qwen.log" || true
      exit 1
    fi
    sleep 10
  done
  echo "[qwen-suite] server did not become ready"
  tail -n 240 "$PROFILE_DIR/server_qwen.log" || true
  exit 1
}

count_trace() {
  local path=$1
  python - "$path" <<'PY'
import json, sys
path = sys.argv[1]
rows = errors = bad = 0
workflows = set()
with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rows += 1
        row = json.loads(line)
        workflows.add(row.get("workflow_id"))
        errors += bool(row.get("error"))
        ttft = row.get("ttft_ms")
        total = row.get("total_latency_ms")
        bad += bool(ttft is None or total is None or (ttft is not None and total is not None and float(total) < float(ttft)))
print(f"{path}: rows={rows} workflows={len(workflows)} errors={errors} bad_latency={bad}")
PY
}

run_sharegpt() {
  activate_sglang
  local out="$PROFILE_DIR/sharegpt_qwen_h100_tp8.jsonl"
  rm -f "$out"
  echo "[qwen-suite] running ShareGPT"
  PYTHONPATH=/workspace python -m hexgent_real_trace_collectors sharegpt \
    --max-workflows 100 \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --model "$SERVED_MODEL" \
    --stream \
    --max-tokens 512 \
    --timeout-s 1200 \
    --output "$out" \
    > "$PROFILE_DIR/sharegpt_run.log" 2>&1
  count_trace "$out"
}

start_proxy() {
  local port=$1
  local dataset=$2
  local out=$3
  local log=$4
  local pid_file=$5
  activate_sglang
  if [ -f "$pid_file" ]; then
    kill -TERM "$(cat "$pid_file")" 2>/dev/null || true
  fi
  rm -f "$out"
  PYTHONPATH=/workspace nohup python -m hexgent_real_trace_collectors proxy \
    --listen-host 127.0.0.1 \
    --listen-port "$port" \
    --target-endpoint "http://127.0.0.1:${PORT}/v1" \
    --trace-dataset "$dataset" \
    --model-override "$SERVED_MODEL" \
    --output "$out" \
    > "$log" 2>&1 &
  echo $! > "$pid_file"
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "[qwen-suite] proxy on port $port did not become healthy"
  cat "$log" || true
  exit 1
}

stop_proxy() {
  local pid_file=$1
  if [ -f "$pid_file" ]; then
    kill -TERM "$(cat "$pid_file")" 2>/dev/null || true
  fi
}

run_bfcl() {
  local out="$PROFILE_DIR/bfcl_qwen_h100_tp8.jsonl"
  local port=18184
  local snapshot
  snapshot=$(find_snapshot)
  echo "[qwen-suite] running BFCL using snapshot=$snapshot"
  start_proxy "$port" bfcl-v4-multi_turn_base "$out" "$PROFILE_DIR/bfcl_proxy.log" "$PROFILE_DIR/bfcl_proxy.pid"
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  cd /workspace/gorilla/berkeley-function-call-leaderboard
  export REMOTE_OPENAI_BASE_URL="http://127.0.0.1:${port}/v1"
  export REMOTE_OPENAI_API_KEY=EMPTY
  export REMOTE_OPENAI_TOKENIZER_PATH="$snapshot"
  export BFCL_PROJECT_ROOT=/workspace/gorilla/berkeley-function-call-leaderboard
  export HEXGENT_SERVER_CONTEXT_LENGTH="$CONTEXT_LENGTH"
  export PYTHONUNBUFFERED=1
  bfcl generate \
    --model "$BFCL_MODEL" \
    --test-category multi_turn_base \
    --skip-server-setup \
    --run-ids \
    --num-threads 1 \
    --include-input-log \
    --allow-overwrite \
    --result-dir "$PROFILE_DIR/bfcl_result" \
    > "$PROFILE_DIR/bfcl_generate.log" 2>&1
  stop_proxy "$PROFILE_DIR/bfcl_proxy.pid"
  count_trace "$out"
}

run_lats() {
  local out="$PROFILE_DIR/lats_hotpotqa_qwen_h100_tp8.jsonl"
  local port=18187
  echo "[qwen-suite] running LATS HotpotQA"
  start_proxy "$port" lats-hotpotqa "$out" "$PROFILE_DIR/lats_proxy.log" "$PROFILE_DIR/lats_proxy.pid"
  # shellcheck disable=SC1091
  . "$LATS_VENV/bin/activate"
  cd /workspace/LanguageAgentTreeSearch/hotpot
  export OPENAI_API_BASE="http://127.0.0.1:${port}/v1"
  export OPENAI_API_KEY=EMPTY
  export HF_HOME=/workspace
  export HUGGINGFACE_HUB_CACHE=/workspace/hub
  export PYTHONPATH="/workspace:${PYTHONPATH:-}"
  export HEXGENT_SPLIT_N_REQUESTS="${HEXGENT_SPLIT_N_REQUESTS:-1}"
  export PYTHONUNBUFFERED=1
  python -m hexgent_real_trace_collectors.lats_split_requests \
    --models-path /workspace/LanguageAgentTreeSearch/hotpot/models.py
  python run.py \
    --backend gpt-3.5-turbo \
    --task_start_index "${LATS_TASK_START_INDEX:-0}" \
    --task_end_index "${LATS_TASK_END_INDEX:-100}" \
    --n_generate_sample "${LATS_N_GENERATE_SAMPLE:-8}" \
    --n_evaluate_sample "${LATS_N_EVALUATE_SAMPLE:-1}" \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 1 \
    --algorithm lats \
    --log "$PROFILE_DIR/lats_task.log" \
    > "$PROFILE_DIR/lats_run.log" 2>&1
  stop_proxy "$PROFILE_DIR/lats_proxy.pid"
  count_trace "$out"
}

build_profiles() {
  activate_sglang
  echo "[qwen-suite] building phase profiles"
  PYTHONPATH=/workspace python -m hexgent_real_trace_collectors.build_phase_profiles \
    --profile-dir "$PROFILE_DIR" \
    --model "$SERVED_MODEL" \
    --worker-class "$WORKER_CLASS" \
    --output-tag qwen_h100_tp8 \
    --source "sharegpt=$PROFILE_DIR/sharegpt_qwen_h100_tp8.jsonl" \
    --source "bfcl=$PROFILE_DIR/bfcl_qwen_h100_tp8.jsonl" \
    --source "hotpotqa=$PROFILE_DIR/lats_hotpotqa_qwen_h100_tp8.jsonl" \
    > "$PROFILE_DIR/build_phase_profiles.log" 2>&1
  cat "$PROFILE_DIR/build_phase_profiles.log"
}

package_results() {
  echo "[qwen-suite] packaging $ARCHIVE_PATH"
  tar -C "$(dirname "$PROFILE_DIR")" -czf "$ARCHIVE_PATH" "$(basename "$PROFILE_DIR")"
  ls -lh "$ARCHIVE_PATH"
}

main() {
  wait_for_download
  start_server
  run_sharegpt
  run_bfcl
  run_lats
  build_profiles
  package_results
  echo "[qwen-suite] complete"
}

main "$@"
