#!/usr/bin/env bash
set -euo pipefail

# Prepare a RunPod container for HexAgenT online PD/e2e debugging.
#
# This script is intentionally idempotent. It assumes the persistent network
# volume is mounted at /workspace and the checked-out/debug source lives at:
#   /workspace/sglang
#
# It does not download models. Existing Hugging Face cache is expected at:
#   /workspace/hub

export DEBIAN_FRONTEND=noninteractive

SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang}"
VENV_DIR="${HEXGENT_VENV_DIR:-/workspace/hexgent_sglang_venv}"
export CARGO_HOME="${CARGO_HOME:-/workspace/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/workspace/.rustup}"

if [[ -f "$CARGO_HOME/env" ]]; then
  # shellcheck source=/dev/null
  source "$CARGO_HOME/env"
fi

if [[ ! -d "$SRC_DIR/python/sglang/hexgent" ]]; then
  echo "Missing HexAgenT source at $SRC_DIR/python/sglang/hexgent" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python -m venv --system-site-packages "$VENV_DIR"
fi

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  libgoogle-glog0v5 \
  libibverbs1 \
  libjsoncpp25 \
  libnuma1 \
  libprotobuf-dev \
  librdmacm1 \
  libssl-dev \
  patchelf \
  pkg-config \
  protobuf-compiler \
  rdma-core

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
VENV_CUDA_RUNTIME_DIR="$SITE_PACKAGES/nvidia/cuda_runtime/lib"
if [[ -f "$VENV_CUDA_RUNTIME_DIR/libcudart.so.12" ]]; then
  # The stock image can expose an older toolkit through ldconfig. Prefer the
  # CUDA runtime installed beside the Torch wheel so Torch and sgl_kernel use
  # the same ABI.
  export LD_LIBRARY_PATH="$VENV_CUDA_RUNTIME_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if ! python - <<'PY' >/dev/null 2>&1
import sglang
import sglang.srt.entrypoints.http_server
import torch

assert torch.__version__.split("+")[0] == "2.9.1"
PY
then
  # This source tree pins Torch 2.9.1 and its compatible binary packages.
  # Install the complete declared stack, then prefer its CUDA runtime below.
  python -m pip install -e "$SRC_DIR/python"
fi

# The editable installation can have installed a new CUDA runtime after the
# initial lookup above. Reapply the search path before importing native code.
VENV_CUDA_RUNTIME_DIR="$SITE_PACKAGES/nvidia/cuda_runtime/lib"
if [[ -f "$VENV_CUDA_RUNTIME_DIR/libcudart.so.12" ]]; then
  export LD_LIBRARY_PATH="$VENV_CUDA_RUNTIME_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python -m pip install --upgrade \
  pytest \
  fastapi \
  maturin \
  uvicorn \
  mooncake-transfer-engine==0.3.9 \
  >/dev/null

mkdir -p "$SITE_PACKAGES/sglang/hexgent"
cp -a "$SRC_DIR/python/sglang/hexgent/." "$SITE_PACKAGES/sglang/hexgent/"

# The editable install resolves ``sglang`` from site-packages on the stock
# RunPod image. Sync the complete runtime tree rather than a hand-maintained
# file list: HexAgenT's disaggregation changes have transitive imports such as
# ``disaggregation.utils -> environ -> debug_utils``.
mkdir -p "$SITE_PACKAGES/sglang/srt"
cp -a "$SRC_DIR/python/sglang/srt/." "$SITE_PACKAGES/sglang/srt/"

mkdir -p /workspace/hexgent_e2e/logs /workspace/hexgent_e2e/run
mkdir -p /workspace/hexgent_e2e/preflight

cat >/workspace/hexgent_e2e/env.sh <<'EOF'
export HF_HOME=/workspace
export HUGGINGFACE_HUB_CACHE=/workspace/hub
export TRANSFORMERS_CACHE=/workspace/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HEXGENT_SRC_DIR=/workspace/sglang
export HEXGENT_VENV_DIR=/workspace/hexgent_sglang_venv
export CARGO_HOME=/workspace/.cargo
export RUSTUP_HOME=/workspace/.rustup
export PATH=/workspace/.cargo/bin:$PATH
if [[ -f "$HEXGENT_VENV_DIR/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12" ]]; then
  export LD_LIBRARY_PATH="$HEXGENT_VENV_DIR/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
EOF

# Keep the setup process and every subsequent SSH pane on the same persistent
# cache/toolchain configuration before probing binary extensions.
# shellcheck source=/dev/null
source /workspace/hexgent_e2e/env.sh

python - <<'PY'
from mooncake.engine import TransferEngine
import sgl_kernel
import sglang.hexgent.sidecar as sidecar
import sglang.srt.entrypoints.http_server as http_server
print("mooncake transfer engine import ok:", TransferEngine)
print("sgl_kernel import ok")
print("hexgent sidecar:", sidecar.__file__)
print("http_server:", http_server.__file__)
PY

if ! python -c 'import sglang_router.launch_router' >/dev/null 2>&1; then
  ROUTER_BINDINGS="$SRC_DIR/sgl-model-gateway/bindings/python"
  if [[ ! -f "$ROUTER_BINDINGS/pyproject.toml" ]]; then
    echo "Missing router Python packaging file: $ROUTER_BINDINGS/pyproject.toml" >&2
    exit 1
  fi
  (
    cd "$ROUTER_BINDINGS"
    python -m maturin develop --release
  )
fi

python "$SRC_DIR/scripts/hexgent_real_trace_collectors/verify_hexgent_serving_env.py" \
  --require-gpu \
  --output /workspace/hexgent_e2e/preflight/latest.json

echo "HexAgenT pod environment is ready."
echo "Run: source /workspace/hexgent_e2e/env.sh"
