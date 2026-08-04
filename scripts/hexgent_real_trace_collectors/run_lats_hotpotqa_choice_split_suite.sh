#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR=${PROFILE_DIR:-/workspace/hexgent_profiles/H100_LATS_CHOICE_SPLIT}
ARCHIVE_PATH=${ARCHIVE_PATH:-${PROFILE_DIR}.tar.gz}
MODEL=${MODEL:-NousResearch/Meta-Llama-3.1-70B-Instruct}
SERVED_MODEL=${SERVED_MODEL:-$MODEL}
WORKER_CLASS=${WORKER_CLASS:-H100_TP4}
PORT=${PORT:-8230}
PROXY_PORT=${PROXY_PORT:-18193}
TP=${TP:-4}
TASK_START_INDEX=${TASK_START_INDEX:-0}
TASK_END_INDEX=${TASK_END_INDEX:-20}
N_GENERATE_SAMPLE=${N_GENERATE_SAMPLE:-5}
N_EVALUATE_SAMPLE=${N_EVALUATE_SAMPLE:-1}
LATS_ROLLOUT_MAX_DEPTH=${LATS_ROLLOUT_MAX_DEPTH:-3}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
VENV=${VENV:-/workspace/hexgent_sglang_venv}
LATS_VENV=${LATS_VENV:-/workspace/hexgent_lats_venv}
SERVER_QUANTIZATION=${SERVER_QUANTIZATION:-}
SERVER_REASONING_PARSER=${SERVER_REASONING_PARSER:-}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "$PROFILE_DIR" /workspace/hub

activate_sglang() {
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  export HF_HOME=/workspace
  export HUGGINGFACE_HUB_CACHE=/workspace/hub
  export HF_HUB_CACHE=/workspace/hub
  export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
  export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
  export SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION=${SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION:-1}
  export PYTHONUNBUFFERED=1
  export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
  local venv_libs
  venv_libs=$(python -c 'import glob, os, site; print(":".join(p for root in site.getsitepackages() for p in glob.glob(os.path.join(root, "nvidia", "*", "lib")) if os.path.isdir(p)))')
  export LD_LIBRARY_PATH="$venv_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

server_ready() {
  curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}

start_server() {
  activate_sglang
  export CUDA_VISIBLE_DEVICES
  if [ -f "$PROFILE_DIR/server.pid" ] && kill -0 "$(cat "$PROFILE_DIR/server.pid")" 2>/dev/null && server_ready; then
    echo "[lats-choice] server already ready pid=$(cat "$PROFILE_DIR/server.pid")"
    return
  fi
  if [ -f "$PROFILE_DIR/server.pid" ]; then
    kill -TERM "$(cat "$PROFILE_DIR/server.pid")" 2>/dev/null || true
    sleep 5
    kill -KILL "$(cat "$PROFILE_DIR/server.pid")" 2>/dev/null || true
  fi

  quantization_args=()
  if [ -n "$SERVER_QUANTIZATION" ]; then
    quantization_args=(--quantization "$SERVER_QUANTIZATION")
  fi
  reasoning_args=()
  if [ -n "$SERVER_REASONING_PARSER" ]; then
    reasoning_args=(--reasoning-parser "$SERVER_REASONING_PARSER")
  fi

  echo "[lats-choice] starting server model=$MODEL tp=$TP port=$PORT"
  # shellcheck disable=SC2086
  nohup python -u -m sglang.launch_server \
    --model-path "$MODEL" \
    --context-length "$CONTEXT_LENGTH" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tp "$TP" \
    --mem-fraction-static 0.837 \
    --cuda-graph-bs 1 \
    "${quantization_args[@]}" \
    "${reasoning_args[@]}" \
    $SERVER_EXTRA_ARGS \
    > "$PROFILE_DIR/server.log" 2>&1 &
  echo $! > "$PROFILE_DIR/server.pid"

  for _ in $(seq 1 720); do
    if server_ready; then
      echo "[lats-choice] server ready"
      return
    fi
    if ! kill -0 "$(cat "$PROFILE_DIR/server.pid")" 2>/dev/null; then
      echo "[lats-choice] server exited early"
      tail -n 240 "$PROFILE_DIR/server.log" || true
      exit 1
    fi
    sleep 10
  done
  echo "[lats-choice] server did not become ready"
  tail -n 240 "$PROFILE_DIR/server.log" || true
  exit 1
}

start_proxy() {
  activate_sglang
  if [ -f "$PROFILE_DIR/proxy.pid" ]; then
    kill -TERM "$(cat "$PROFILE_DIR/proxy.pid")" 2>/dev/null || true
  fi
  rm -f "$PROFILE_DIR/lats_hotpotqa.jsonl"
  PYTHONPATH=/workspace nohup python -m hexgent_real_trace_collectors proxy \
    --listen-host 127.0.0.1 \
    --listen-port "$PROXY_PORT" \
    --target-endpoint "http://127.0.0.1:${PORT}/v1" \
    --trace-dataset lats-hotpotqa \
    --model-override "$SERVED_MODEL" \
    --output "$PROFILE_DIR/lats_hotpotqa.jsonl" \
    > "$PROFILE_DIR/proxy.log" 2>&1 &
  echo $! > "$PROFILE_DIR/proxy.pid"
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
      echo "[lats-choice] proxy ready"
      return
    fi
    sleep 1
  done
  echo "[lats-choice] proxy did not become healthy"
  cat "$PROFILE_DIR/proxy.log" || true
  exit 1
}

stop_proxy() {
  if [ -f "$PROFILE_DIR/proxy.pid" ]; then
    kill -TERM "$(cat "$PROFILE_DIR/proxy.pid")" 2>/dev/null || true
  fi
}

run_lats() {
  # shellcheck disable=SC1091
  . "$LATS_VENV/bin/activate"
  cd /workspace/LanguageAgentTreeSearch/hotpot
  export OPENAI_API_BASE="http://127.0.0.1:${PROXY_PORT}/v1"
  export OPENAI_API_KEY=EMPTY
  export HF_HOME=/workspace
  export HUGGINGFACE_HUB_CACHE=/workspace/hub
  export PYTHONPATH="/workspace:${PYTHONPATH:-}"
  export HEXGENT_SPLIT_N_REQUESTS=0
  export HEXGENT_RECORD_N_CHOICES=1
  export HEXGENT_LATS_MAX_DEPTH="$LATS_ROLLOUT_MAX_DEPTH"
  export HEXGENT_LATS_ROLLOUT_MAX_DEPTH="$LATS_ROLLOUT_MAX_DEPTH"
  export HEXGENT_LATS_MAX_EMPTY_EXPANSION_RETRIES=${HEXGENT_LATS_MAX_EMPTY_EXPANSION_RETRIES:-2}
  export PYTHONUNBUFFERED=1
  python -m hexgent_real_trace_collectors.lats_split_requests \
    --models-path /workspace/LanguageAgentTreeSearch/hotpot/models.py \
    --lats-path /workspace/LanguageAgentTreeSearch/hotpot/lats.py
  python run.py \
    --backend gpt-3.5-turbo \
    --task_start_index "$TASK_START_INDEX" \
    --task_end_index "$TASK_END_INDEX" \
    --n_generate_sample "$N_GENERATE_SAMPLE" \
    --n_evaluate_sample "$N_EVALUATE_SAMPLE" \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 1 \
    --algorithm lats \
    --log "$PROFILE_DIR/lats_task.log" \
    > "$PROFILE_DIR/lats_run.log" 2>&1
}

count_trace() {
  python - "$PROFILE_DIR/lats_hotpotqa.jsonl" <<'PY'
import json, sys
path = sys.argv[1]
rows = errors = missing_ttft = 0
workflows = set()
fanouts = set()
with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        workflows.add(row.get("workflow_id"))
        meta = row.get("metadata") or {}
        if meta.get("fanout_group_id"):
            fanouts.add(meta.get("fanout_group_id"))
        errors += bool(row.get("error"))
        missing_ttft += row.get("ttft_ms") is None
print(f"{path}: rows={rows} workflows={len(workflows)} fanout_groups={len(fanouts)} errors={errors} missing_ttft={missing_ttft}")
PY
}

build_profiles() {
  activate_sglang
  PYTHONPATH=/workspace python -m hexgent_real_trace_collectors.build_phase_profiles \
    --profile-dir "$PROFILE_DIR" \
    --model "$SERVED_MODEL" \
    --worker-class "$WORKER_CLASS" \
    --output-tag lats_hotpotqa \
    --source "hotpotqa=$PROFILE_DIR/lats_hotpotqa.jsonl" \
    > "$PROFILE_DIR/build_phase_profiles.log" 2>&1
  cat "$PROFILE_DIR/build_phase_profiles.log"
}

package_results() {
  tar -C "$(dirname "$PROFILE_DIR")" -czf "$ARCHIVE_PATH" "$(basename "$PROFILE_DIR")"
  ls -lh "$ARCHIVE_PATH"
}

main() {
  start_server
  start_proxy
  run_lats
  stop_proxy
  count_trace
  build_profiles
  package_results
  echo "[lats-choice] complete"
}

main "$@"
