import ctypes

from scripts.hexgent_pod_profile import probe_cuda_ipc_tensor as probe


def test_cuda_ipc_handle_serialization_preserves_all_handle_bytes():
    handle = probe.cudaIpcMemHandle_t()
    payload = bytes(range(ctypes.sizeof(handle)))
    ctypes.memmove(ctypes.byref(handle), payload, len(payload))

    restored = probe.deserialize_ipc_handle(probe.serialize_ipc_handle(handle))

    assert ctypes.string_at(ctypes.byref(restored), ctypes.sizeof(restored)) == payload
