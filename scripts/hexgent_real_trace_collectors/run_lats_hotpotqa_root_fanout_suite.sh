#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR=${PROFILE_DIR:-/workspace/hexgent_profiles/H100_LATS_ROOT_FANOUT}
ARCHIVE_PATH=${ARCHIVE_PATH:-${PROFILE_DIR}.tar.gz}
MODEL=${MODEL:-Qwen/Qwen3-235B-A22B-FP8}
SERVED_MODEL=${SERVED_MODEL:-$MODEL}
WORKER_CLASS=${WORKER_CLASS:-H100_TP4_FP8}
PORT=${PORT:-8240}
PROXY_PORT=${PROXY_PORT:-18195}
TP=${TP:-4}
TASK_START_INDEX=${TASK_START_INDEX:-0}
TASK_END_INDEX=${TASK_END_INDEX:-20}
N_GENERATE_SAMPLE=${N_GENERATE_SAMPLE:-5}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
VENV=${VENV:-/workspace/hexgent_sglang_venv}
LATS_VENV=${LATS_VENV:-/workspace/hexgent_lats_venv}
SERVER_REASONING_PARSER=${SERVER_REASONING_PARSER:-}
SERVER_QUANTIZATION=${SERVER_QUANTIZATION:-}
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
    echo "[lats-root] server already ready pid=$(cat "$PROFILE_DIR/server.pid")"
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

  echo "[lats-root] starting server model=$MODEL tp=$TP port=$PORT"
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
    > "$PROFILE_DIR/server.log" 2>&1 &
  echo $! > "$PROFILE_DIR/server.pid"

  for _ in $(seq 1 720); do
    if server_ready; then
      echo "[lats-root] server ready"
      return
    fi
    if ! kill -0 "$(cat "$PROFILE_DIR/server.pid")" 2>/dev/null; then
      echo "[lats-root] server exited early"
      tail -n 240 "$PROFILE_DIR/server.log" || true
      exit 1
    fi
    sleep 10
  done
  echo "[lats-root] server did not become ready"
  tail -n 240 "$PROFILE_DIR/server.log" || true
  exit 1
}

start_proxy() {
  activate_sglang
  if [ -f "$PROFILE_DIR/proxy.pid" ]; then
    kill -TERM "$(cat "$PROFILE_DIR/proxy.pid")" 2>/dev/null || true
  fi
  rm -f "$PROFILE_DIR/lats_hotpotqa_root.jsonl"
  PYTHONPATH=/workspace nohup python -m hexgent_real_trace_collectors proxy \
    --listen-host 127.0.0.1 \
    --listen-port "$PROXY_PORT" \
    --target-endpoint "http://127.0.0.1:${PORT}/v1" \
    --trace-dataset lats-hotpotqa-root \
    --model-override "$SERVED_MODEL" \
    --output "$PROFILE_DIR/lats_hotpotqa_root.jsonl" \
    > "$PROFILE_DIR/proxy.log" 2>&1 &
  echo $! > "$PROFILE_DIR/proxy.pid"
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
      echo "[lats-root] proxy ready"
      return
    fi
    sleep 1
  done
  echo "[lats-root] proxy did not become healthy"
  cat "$PROFILE_DIR/proxy.log" || true
  exit 1
}

run_root_fanout() {
  # shellcheck disable=SC1091
  . "$LATS_VENV/bin/activate"
  PYTHONPATH=/workspace python -m hexgent_real_trace_collectors.lats_root_fanout \
    --lats-hotpot-dir /workspace/LanguageAgentTreeSearch/hotpot \
    --endpoint "http://127.0.0.1:${PROXY_PORT}/v1" \
    --model "$SERVED_MODEL" \
    --task-start-index "$TASK_START_INDEX" \
    --task-end-index "$TASK_END_INDEX" \
    --n-generate-sample "$N_GENERATE_SAMPLE" \
    --max-tokens 100 \
    --timeout-s 1200 \
    > "$PROFILE_DIR/root_fanout_run.log" 2>&1
}

count_trace() {
  python - "$PROFILE_DIR/lats_hotpotqa_root.jsonl" <<'PY'
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
    --output-tag lats_hotpotqa_root \
    --source "hotpotqa=$PROFILE_DIR/lats_hotpotqa_root.jsonl" \
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
  run_root_fanout
  kill -TERM "$(cat "$PROFILE_DIR/proxy.pid")" 2>/dev/null || true
  count_trace
  build_profiles
  package_results
  echo "[lats-root] complete"
}

main "$@"
