from pathlib import Path


def test_cuda_peer_probe_times_and_verifies_the_same_peer_copy():
    source = (
        Path(__file__).parents[3]
        / "scripts"
        / "hexgent_pod_profile"
        / "cuda_peer_copy_probe.cu"
    ).read_text(encoding="utf-8")

    assert "cudaDeviceCanAccessPeer" in source
    assert "cudaDeviceEnablePeerAccess" in source
    assert "cudaMemcpyPeerAsync" in source
    assert "cudaEventElapsedTime" in source
    assert "verification failed" in source
