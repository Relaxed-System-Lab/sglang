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
