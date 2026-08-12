from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.disaggregation import utils as disaggregation_utils
from sglang.srt.disaggregation.mooncake.conn import (
    KVTransferTiming,
    MooncakeKVManager,
    kv_payload_bytes,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    get_mooncake_transfer_protocol,
)


@pytest.mark.parametrize(
    "configured, expected",
    [
        (None, "rdma"),
        ("tcp", "tcp"),
        ("NVLINK_INTRA", "nvlink_intra"),
        ("nvlink", "nvlink"),
    ],
)
def test_mooncake_transfer_protocol_selection(monkeypatch, configured, expected):
    key = "SGLANG_MOONCAKE_TRANSFER_PROTOCOL"
    if configured is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, configured)

    assert get_mooncake_transfer_protocol() == expected


def test_mooncake_transfer_protocol_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("SGLANG_MOONCAKE_TRANSFER_PROTOCOL", "not-a-protocol")

    with pytest.raises(ValueError, match="Unsupported Mooncake transfer protocol"):
        get_mooncake_transfer_protocol()


def test_intranode_nvlink_keeps_auxiliary_metadata_on_cpu(monkeypatch):
    """TCP/ZMQ auxiliary messages must never dereference CUDA pointers."""
    devices = []

    monkeypatch.setenv("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "INTRA_NODE_NVLINK")
    monkeypatch.setattr(disaggregation_utils, "is_npu", lambda: False)
    monkeypatch.setattr(
        disaggregation_utils.torch,
        "zeros",
        lambda *args, **kwargs: devices.append(kwargs["device"]) or object(),
    )

    disaggregation_utils.MetadataBuffers(
        size=1,
        hidden_size=1,
        hidden_states_dtype=disaggregation_utils.torch.float16,
    )

    assert devices == ["cpu"] * 10


def test_intranode_nvlink_sends_auxiliary_metadata_over_tcp(monkeypatch):
    """Keep bulk KV on NVLink but use the CPU-safe control-message path."""
    manager = object.__new__(MooncakeKVManager)
    manager.enable_custom_mem_pool = False
    manager.custom_mem_pool_type = None
    manager.kv_args = SimpleNamespace(aux_data_ptrs=[100], aux_item_lens=[8])
    manager.send_aux_tcp = Mock(return_value=0)
    manager._transfer_data = Mock(return_value=0)
    request = SimpleNamespace(
        room=7,
        mooncake_session_id="session",
        dst_aux_index=0,
    )

    monkeypatch.setenv("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "INTRA_NODE_NVLINK")
    monkeypatch.delenv("SGLANG_MOONCAKE_SEND_AUX_TCP", raising=False)

    assert manager.send_aux(request, prefill_aux_index=0, dst_aux_ptrs=[200]) == 0
    manager.send_aux_tcp.assert_called_once_with(request, 0, [200])
    manager._transfer_data.assert_not_called()


def test_kv_transfer_timing_separates_queue_and_active_time():
    timing = KVTransferTiming(first_queued_ns=100)

    timing.record_transfer(
        queued_ns=100,
        started_ns=140,
        completed_ns=200,
        payload_bytes=4096,
    )
    timing.record_transfer(
        queued_ns=220,
        started_ns=250,
        completed_ns=300,
        payload_bytes=2048,
    )

    assert timing.queue_wait_ns == 40
    assert timing.active_ns == 110
    assert timing.payload_bytes == 6144
    assert timing.chunk_count == 2
    assert timing.end_to_end_ns(340) == 240


def test_kv_transfer_timing_tracks_native_sync_critical_path():
    timing = KVTransferTiming(first_queued_ns=100)

    # Two overlapping per-layer calls: critical path is not their sum.
    timing.record_sync_call(started_ns=150, completed_ns=230)
    timing.record_sync_call(started_ns=170, completed_ns=260)

    assert timing.sync_call_count == 2
    assert timing.sync_sum_ns == 170
    assert timing.sync_critical_path_ns() == 110


def test_kv_payload_bytes_uses_the_actual_page_count_for_all_kv_buffers():
    assert kv_payload_bytes([1024, 1024, 512, 512], page_count=3) == 9216
