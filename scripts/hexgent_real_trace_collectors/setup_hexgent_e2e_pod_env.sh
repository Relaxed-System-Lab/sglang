#!/usr/bin/env bash
set -euo pipefail

# Prepare a RunPod container for HexAgenT online PD/e2e debugging.
#
# This script is intentionally idempotent. It assumes the persistent network
# volume is mounted at /workspace and the checked-out/debug source lives at:
#   /workspace/sglang
#
# The runtime venv defaults to the container disk because installing thousands
# of wheel files directly on RunPod's network volume can block in NFS I/O.
# Download caches, manifests, source, and experiment outputs remain persistent.
# It does not download models. Existing Hugging Face cache is expected at:
#   /workspace/hub

export DEBIAN_FRONTEND=noninteractive

SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang}"
VENV_DIR="${HEXGENT_VENV_DIR:-/opt/hexgent_sglang_venv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export CARGO_HOME="${CARGO_HOME:-/workspace/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/workspace/.rustup}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/opt/hexgent-router-target}"
ROUTER_BUILD_SRC="${HEXGENT_ROUTER_BUILD_SRC:-/opt/hexgent-router-src}"
ROUTER_CARGO_HOME="${HEXGENT_ROUTER_CARGO_HOME:-/opt/hexgent-router-cargo}"
ROUTER_RUSTUP_HOME="${HEXGENT_ROUTER_RUSTUP_HOME:-/opt/hexgent-router-rustup}"
LOCAL_BUILD_ROOT="${HEXGENT_LOCAL_BUILD_ROOT:-/opt/hexgent-build}"

mkdir -p "$PIP_CACHE_DIR" "$LOCAL_BUILD_ROOT"

if [[ -x /usr/local/cuda-12.4/bin/nvcc ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

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
  build-essential \
  ca-certificates \
  cmake \
  curl \
  git \
  libaio-dev \
  libboost-all-dev \
  libcurl4-openssl-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  libgoogle-glog0v5 \
  libgrpc++-dev \
  libgtest-dev \
  libibverbs1 \
  libjsoncpp-dev \
  libjsoncpp25 \
  libnuma1 \
  libnuma-dev \
  libprotobuf-dev \
  librdmacm1 \
  libssl-dev \
  ninja-build \
  patchelf \
  pkg-config \
  protobuf-compiler \
  rdma-core \
  libyaml-cpp-dev

YALANTINGLIBS_VERSION="0.5.7"
YALANTINGLIBS_DIR="$LOCAL_BUILD_ROOT/yalantinglibs-$YALANTINGLIBS_VERSION"
if [[ ! -f /usr/local/lib/cmake/yalantinglibs/yalantinglibsConfig.cmake ]]; then
  if [[ ! -d "$YALANTINGLIBS_DIR/.git" ]]; then
    git clone --depth 1 --branch "$YALANTINGLIBS_VERSION" \
      https://github.com/alibaba/yalantinglibs.git "$YALANTINGLIBS_DIR"
  fi
  cmake -S "$YALANTINGLIBS_DIR" -B "$YALANTINGLIBS_DIR/build" \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARK=OFF \
    -DBUILD_UNIT_TESTS=OFF
  cmake --build "$YALANTINGLIBS_DIR/build" -j"$(nproc)"
  cmake --install "$YALANTINGLIBS_DIR/build"
fi

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
fi
# shellcheck source=/dev/null
source "$CARGO_HOME/env"

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

# Install the complete stack declared by the checked-out branch. This is
# intentionally unconditional: a persistent volume can contain a valid but
# incompatible v0.5.9 environment, and import-only checks cannot detect that
# the router/runtime dependencies belong to another release.
python -m pip install -e "$SRC_DIR/python"

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

printf 'export HEXGENT_SRC_DIR=%q\n' "$SRC_DIR" >/workspace/hexgent_e2e/env.sh
cat >>/workspace/hexgent_e2e/env.sh <<'EOF'
export HF_HOME=/workspace
export HUGGINGFACE_HUB_CACHE=/workspace/hub
export TRANSFORMERS_CACHE=/workspace/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HEXGENT_VENV_DIR=/opt/hexgent_sglang_venv
export PIP_CACHE_DIR=/workspace/.cache/pip
export CARGO_HOME=/workspace/.cargo
export RUSTUP_HOME=/workspace/.rustup
export CARGO_TARGET_DIR=/opt/hexgent-router-target
export PATH="$HEXGENT_VENV_DIR/bin:/workspace/.cargo/bin:$PATH"
if [[ -x /usr/local/cuda-12.4/bin/nvcc ]]; then
  export CUDA_HOME=/usr/local/cuda-12.4
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
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
  # Cargo metadata and source traversal can stall on RunPod's network volume.
  # Build from a disposable container-local mirror; the authoritative source
  # and all experiment outputs remain under /workspace.
  rm -rf "$ROUTER_BUILD_SRC"
  mkdir -p "$ROUTER_BUILD_SRC"
  cp -a "$SRC_DIR/sgl-model-gateway/." "$ROUTER_BUILD_SRC/"
  ROUTER_BINDINGS="$ROUTER_BUILD_SRC/bindings/python"
  if [[ ! -f "$ROUTER_BINDINGS/pyproject.toml" ]]; then
    echo "Missing router Python packaging file: $ROUTER_BINDINGS/pyproject.toml" >&2
    exit 1
  fi
  mkdir -p "$ROUTER_CARGO_HOME"
  if [[ ! -x "$ROUTER_CARGO_HOME/bin/cargo" ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | env CARGO_HOME="$ROUTER_CARGO_HOME" RUSTUP_HOME="$ROUTER_RUSTUP_HOME" \
        sh -s -- -y --profile minimal --default-toolchain stable
  fi
  (
    cd "$ROUTER_BINDINGS"
    env CARGO_HOME="$ROUTER_CARGO_HOME" RUSTUP_HOME="$ROUTER_RUSTUP_HOME" \
      PATH="$ROUTER_CARGO_HOME/bin:$PATH" python -m maturin develop --release
  )
fi

python "$SRC_DIR/scripts/hexgent_real_trace_collectors/verify_hexgent_serving_env.py" \
  --require-gpu \
  --output /workspace/hexgent_e2e/preflight/latest.json

MANIFEST_DIR=/workspace/hexgent_e2e/environment-manifest
mkdir -p "$MANIFEST_DIR"
python -m pip freeze | sort >"$MANIFEST_DIR/pip-freeze.txt"
{
  cat /etc/os-release
  echo
  uname -a
  echo
  python --version
  cargo --version
  rustc --version
  nvcc --version 2>/dev/null || true
} >"$MANIFEST_DIR/platform.txt" 2>&1
nvidia-smi topo -m >"$MANIFEST_DIR/nvidia-topology.txt"
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader >"$MANIFEST_DIR/gpus.csv"

echo "HexAgenT pod environment is ready."
echo "Run: source /workspace/hexgent_e2e/env.sh"
