from pathlib import Path


def test_mooncake_native_copy_patch_uses_cuda_events_without_changing_copy_api():
    patch_path = (
        Path(__file__).parents[3]
        / "scripts"
        / "hexgent_real_trace_collectors"
        / "patches"
        / "mooncake-v0.3.9-intranode-copy-cuda-events.patch"
    )

    patch = patch_path.read_text(encoding="utf-8")

    assert "MC_HEXGENT_CUDA_COPY_TELEMETRY" in patch
    assert "cudaEventRecord(start_event, 0)" in patch
    assert "cudaEventRecord(end_event, 0)" in patch
    assert "cudaEventElapsedTime" in patch
    assert "[HEXGENT_CUDA_COPY]" in patch
    assert "[HEXGENT_CUDA_COPY_BATCH]" in patch
    assert patch.count("timedCudaMemcpy(") >= 3
    assert "cudaMemcpyAsync" not in patch


def test_intranode_priority_patch_matches_v039_no_hca_branch():
    patch_path = (
        Path(__file__).parents[3]
        / "scripts"
        / "hexgent_real_trace_collectors"
        / "patches"
        / "mooncake-v0.3.9-intranode-nvlink-priority.patch"
    )

    patch = patch_path.read_text(encoding="utf-8")

    assert (
        "-        if (force_mnnvl || local_topology_->getHcaList().empty()) {"
        in patch
    )
    assert "+        if (intra_env) {" in patch
    assert "+        } else if (force_mnnvl ||" in patch


def test_peer_access_patch_restores_callers_cuda_device():
    patch_path = (
        Path(__file__).parents[3]
        / "scripts"
        / "hexgent_real_trace_collectors"
        / "patches"
        / "mooncake-v0.3.9-intranode-nvlink-enable-peer-access.patch"
    )

    patch = patch_path.read_text(encoding="utf-8")

    assert "+    int original_device = -1;" in patch
    assert "+    cudaGetDevice(&original_device);" in patch
    assert "+        cudaSetDevice(original_device);" in patch
