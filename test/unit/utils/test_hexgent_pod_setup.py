from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SETUP_SCRIPT = (
    ROOT
    / "scripts"
    / "hexgent_real_trace_collectors"
    / "setup_hexgent_e2e_pod_env.sh"
)


def test_native_build_trees_use_container_local_storage():
    script = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert 'LOCAL_BUILD_ROOT="${HEXGENT_LOCAL_BUILD_ROOT:-/opt/hexgent-build}"' in script
    assert 'YALANTINGLIBS_DIR="$LOCAL_BUILD_ROOT/yalantinglibs-$YALANTINGLIBS_VERSION"' in script
    assert 'YALANTINGLIBS_DIR="/workspace/third_party/' not in script


def test_generated_environment_preserves_selected_source_tree():
    script = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "printf 'export HEXGENT_SRC_DIR=%q\\n' \"$SRC_DIR\"" in script
    assert "export HEXGENT_SRC_DIR=/workspace/sglang\n" not in script


def test_python_wheel_is_built_from_container_local_mirror():
    script = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert 'PYTHON_BUILD_SRC="${HEXGENT_PYTHON_BUILD_SRC:-$LOCAL_BUILD_ROOT/sglang-src}"' in script
    assert 'python -m pip install "$PYTHON_BUILD_SRC/python"' in script
    assert 'python -m pip install -e "$SRC_DIR/python"' not in script
    assert 'LOCAL_CARGO_HOME="${HEXGENT_LOCAL_CARGO_HOME:-/opt/hexgent-cargo}"' in script
    assert 'LOCAL_RUSTUP_HOME="${HEXGENT_LOCAL_RUSTUP_HOME:-/opt/hexgent-rustup}"' in script
    assert 'PYTHON_CARGO_HOME="${HEXGENT_PYTHON_CARGO_HOME:-$LOCAL_CARGO_HOME}"' in script
    assert 'CARGO_HOME="$PYTHON_CARGO_HOME"' in script
    assert 'RUSTUP_HOME="$LOCAL_RUSTUP_HOME"' in script


def test_generated_environment_uses_container_local_rust_toolchain():
    script = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "export CARGO_HOME=/workspace/.cargo" not in script
    assert "export RUSTUP_HOME=/workspace/.rustup" not in script
    assert "export CARGO_HOME=/opt/hexgent-cargo" in script
    assert "export RUSTUP_HOME=/opt/hexgent-rustup" in script
