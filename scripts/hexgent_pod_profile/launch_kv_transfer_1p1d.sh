#!/usr/bin/env bash
set -euo pipefail

# Measure the production P-D Mooncake path on a two-GPU, single-node setup.
# The caller must activate the SGLang environment before running this script.

TASK_DIR="${TASK_DIR:-/workspace/hexgent_e2e/kv_transfer_1p1d_20260727}"
MODEL_PATH="${MODEL_PATH:-/workspace/hub/models--neuralmagic--Meta-Llama-3.1-8B-Instruct-FP8/snapshots/12fd6884d2585dd4d020373e7f39f74507b31866}"

mkdir -p "$TASK_DIR"

export PYTHONPATH="/workspace/sglang/python:/workspace/sglang/sgl-model-gateway/bindings/python/src${PYTHONPATH:+:$PYTHONPATH}"
# The persistent Mooncake build is compiled with USE_INTRA_NVLINK=ON.  Its
# runtime dependencies are installed under /usr/local/lib on this pod.
export LD_LIBRARY_PATH="/usr/local/lib:/workspace/third_party/Mooncake-v0.3.9/build-intra-nvlink/mooncake-asio${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SGLANG_MOONCAKE_TRANSFER_PROTOCOL=nvlink_intra
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
# Mooncake otherwise selects cross-node MNNVL when no RDMA HCA is detected.
# This two-GPU experiment explicitly requires its intra-node NVLink transport.
export MC_INTRANODE_NVLINK=1
export SGLANG_MOONCAKE_SEND_AUX_TCP=1
export SGLANG_KV_TRANSFER_TELEMETRY=1
export MC_TE_METRIC=1

# GPU event measurements are easiest to interpret when a scheduler iteration
# maps to one model-forward stream interval. Keep this enabled for the isolated
# 1P+1D benchmark; production throughput experiments can set it to 0.
MEASUREMENT_SERVER_ARGS=()
if [[ "${MEASUREMENT_DISABLE_OVERLAP_SCHEDULE:-1}" == "1" ]]; then
  MEASUREMENT_SERVER_ARGS+=(--disable-overlap-schedule)
fi

CUDA_VISIBLE_DEVICES=0 nohup python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 31000 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 9100 \
  --mem-fraction-static 0.70 \
  --max-running-requests 16 \
  --disable-radix-cache \
  "${MEASUREMENT_SERVER_ARGS[@]}" \
  > "$TASK_DIR/prefill.log" 2>&1 &
echo $! > "$TASK_DIR/prefill.pid"

CUDA_VISIBLE_DEVICES=1 nohup python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 31001 \
  --base-gpu-id 0 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --mem-fraction-static 0.70 \
  --max-running-requests 64 \
  "${MEASUREMENT_SERVER_ARGS[@]}" \
  > "$TASK_DIR/decode.log" 2>&1 &
echo $! > "$TASK_DIR/decode.pid"

nohup python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:31000 9100 \
  --decode http://127.0.0.1:31001 \
  --host 127.0.0.1 \
  --port 31081 \
  --prometheus-port 29091 \
  > "$TASK_DIR/router.log" 2>&1 &
echo $! > "$TASK_DIR/router.pid"

printf 'Logs: %s\n' "$TASK_DIR"
