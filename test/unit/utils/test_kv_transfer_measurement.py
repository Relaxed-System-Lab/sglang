import sys

import pytest

from scripts.hexgent_pod_profile.measure_kv_transfer_1p1d import build_request_payload
from scripts.hexgent_pod_profile import measure_mooncake_cuda_p2p_peer as peer_measurement
from scripts.hexgent_real_trace_collectors import run_pd_transfer_stress


def test_measurement_payload_can_force_a_fixed_generation_length():
    payload = build_request_payload(
        model="model",
        content="prompt",
        max_tokens=64,
        min_tokens=64,
    )

    assert payload["max_tokens"] == 64
    assert payload["min_tokens"] == 64


def test_peer_cuda_measurement_accepts_an_explicit_visible_device(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_mooncake_cuda_p2p_peer.py",
            "--role",
            "target",
            "--coord-dir",
            str(tmp_path),
            "--device-index",
            "1",
        ],
    )

    args = peer_measurement.parse_args()

    assert args.device_index == 1


def test_peer_cuda_measurement_exposes_the_sync_transfer_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_mooncake_cuda_p2p_peer.py",
            "--role",
            "initiator",
            "--coord-dir",
            str(tmp_path),
            "--transfer-mode",
            "sync",
        ],
    )

    args = peer_measurement.parse_args()

    assert args.transfer_mode == "sync"


def test_peer_cuda_measurement_accepts_a_segmented_kv_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_mooncake_cuda_p2p_peer.py",
            "--role",
            "initiator",
            "--coord-dir",
            str(tmp_path),
            "--size-mib",
            "256",
            "--segments",
            "64",
        ],
    )

    args = peer_measurement.parse_args()

    assert args.segments == 64


def test_peer_cuda_measurement_can_submit_from_a_worker_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_mooncake_cuda_p2p_peer.py",
            "--role",
            "initiator",
            "--coord-dir",
            str(tmp_path),
            "--submit-in-thread",
        ],
    )

    args = peer_measurement.parse_args()

    assert args.submit_in_thread is True


def test_peer_target_selects_its_device_before_initializing_mooncake(monkeypatch, tmp_path):
    events = []

    class ExpectedStop(Exception):
        pass

    monkeypatch.setattr(peer_measurement, "selected_device", lambda _: "cuda:1")
    monkeypatch.setattr(
        peer_measurement.torch.cuda,
        "set_device",
        lambda device: events.append(("device", device)),
    )

    def stop_at_engine_initialization():
        events.append(("engine",))
        raise ExpectedStop

    monkeypatch.setattr(peer_measurement, "build_engine", stop_at_engine_initialization)
    args = peer_measurement.argparse.Namespace(
        coord_dir=tmp_path,
        size_mib=64,
        repeats=1,
        timeout_seconds=1.0,
        device_index=1,
    )

    with pytest.raises(ExpectedStop):
        peer_measurement.run_target(args)

    assert events == [("device", "cuda:1"), ("engine",)]


def test_pd_transfer_stress_defaults_to_a_fixed_long_generation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pd_transfer_stress.py", "--output", "stress.jsonl"],
    )

    args = run_pd_transfer_stress.parse_args()

    assert args.router_url == "http://127.0.0.1:33001"
    assert args.concurrency == 12
    assert args.output_tokens == 128


def test_pd_transfer_stress_payload_prevents_early_eos():
    payload = run_pd_transfer_stress.build_payload(
        model="llama8b-pd-2p2d",
        prompt="test prompt",
        output_tokens=128,
    )

    assert payload["min_tokens"] == 128
    assert payload["max_tokens"] == 128
    assert payload["ignore_eos"] is True
