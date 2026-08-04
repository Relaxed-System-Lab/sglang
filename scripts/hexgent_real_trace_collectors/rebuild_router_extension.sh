#!/usr/bin/env bash
set -euo pipefail

# Build the router extension on pod-local storage. Building on the mounted
# network volume has triggered rust-lld bus errors on some RunPod images.
VENV_DIR="${HEXGENT_VENV_DIR:-/workspace/hexgent_sglang_venv}"
SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang}"
TARGET_DIR="${HEXGENT_ROUTER_TARGET_DIR:-/tmp/hexgent_router_target}"
BUILD_JOBS="${HEXGENT_ROUTER_BUILD_JOBS:-6}"
PROFILE="${HEXGENT_ROUTER_BUILD_PROFILE:-ci}"

source "$VENV_DIR/bin/activate"
if [[ -f /workspace/hexgent_e2e/env.sh ]]; then
  source /workspace/hexgent_e2e/env.sh
fi

export CARGO_TARGET_DIR="$TARGET_DIR"
export CARGO_BUILD_JOBS="$BUILD_JOBS"
# The bundled rust-lld has crashed while linking this extension on RunPod.
export RUSTFLAGS="${RUSTFLAGS:--C link-arg=-fuse-ld=bfd}"

mkdir -p "$CARGO_TARGET_DIR"
cd "$SRC_DIR/sgl-model-gateway/bindings/python"
python -m maturin develop --profile "$PROFILE"
