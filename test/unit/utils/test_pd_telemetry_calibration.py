import importlib.util
import json
from pathlib import Path
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = (
    _ROOT
    / "scripts"
    / "hexgent_real_trace_collectors"
    / "calibrate_pd_telemetry.py"
)


def _load_calibrator():
    spec = importlib.util.spec_from_file_location(
        "calibrate_pd_telemetry", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_workflow_trace():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "python"
        / "sglang"
        / "hexgent"
        / "workflow_trace.py"
    )
    spec = importlib.util.spec_from_file_location("workflow_trace", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_kv_bytes_per_token_is_derived_from_model_config():
    calibrator = _load_calibrator()

    qwen025b_config = {
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "num_hidden_layers": 24,
        "head_dim": 64,
    }

    assert (
        calibrator.kv_bytes_per_token_from_config(qwen025b_config, kv_dtype_bytes=2)
        == 12_288
    )


def test_explicit_kv_bytes_override_takes_precedence():
    calibrator = _load_calibrator()

    assert calibrator.resolve_bytes_per_input_token(
        explicit_bytes=777,
        model_config={
            "hidden_size": 896,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "num_hidden_layers": 24,
            "head_dim": 64,
        },
        kv_dtype_bytes=2,
    ) == (777, "explicit")


def test_request_telemetry_preserves_admission_queue_duration(tmp_path: Path):
    calibrator = _load_calibrator()
    log_path = tmp_path / "worker.log"
    log_path.write_text(
        "[x] Req Time Stats(rid=req-1, bootstrap_room=1, input len=10, "
        "output len=2, type=prefill): "
        "bootstrap_queue_duration(37.5ms) = alloc_wait(0.0ms) + "
        "bootstrap(0.0ms) + other(37.5ms); queue_duration=1.5ms, "
        "forward_duration=12.0ms, start=1\n"
    )

    rows = calibrator.read_rows([log_path], bytes_per_input_token=100)

    assert rows[0]["admission_queue_ms"] == 37.5
    assert rows[0]["forward_ms"] == 12.0


def test_request_telemetry_can_filter_short_probe_rows(tmp_path: Path):
    calibrator = _load_calibrator()
    log_path = tmp_path / "worker.log"
    log_path.write_text(
        "Req Time Stats(rid=probe, input len=4, output len=1, type=prefill): "
        "bootstrap_queue_duration(0.1ms); queue_duration=0.1ms, "
        "forward_duration=9000.0ms\n"
        "Req Time Stats(rid=real, input len=100, output len=10, type=prefill): "
        "bootstrap_queue_duration(0.1ms); queue_duration=0.1ms, "
        "forward_duration=20.0ms\n"
    )

    rows = calibrator.read_rows(
        [log_path], bytes_per_input_token=100, min_input_tokens=8
    )

    assert [row["rid"] for row in rows] == ["real"]


def test_request_telemetry_parses_v0515_pipeline_and_prefill_transfer_fields(
    tmp_path: Path,
):
    calibrator = _load_calibrator()
    log_path = tmp_path / "worker.log"
    log_path.write_text(
        "ReqTimeStats(rid=prefill-rid, bootstrap_room=42, input_len=6510, "
        "cached_input_len=0, output_len=1, type=prefill): "
        "bootstrap_duration=0.27ms, queue_duration=0.25ms, "
        "forward_duration=443.23ms, entry_time=1, "
        "transfer_speed=123.90 GB/s, transfer_total=813.75 MB, #retries=0\n"
        "ReqTimeStats(rid=decode-rid, bootstrap_room=42, input_len=6510, "
        "cached_input_len=0, output_len=20, type=decode): "
        "bootstrap_duration=0.09ms, alloc_wait_duration=0.22ms, "
        "transfer_duration=444.96ms, queue_duration=0.14ms, "
        "forward_duration=215.29ms, entry_time=1\n"
    )

    rows = calibrator.read_rows([log_path], bytes_per_input_token=131_072)

    assert len(rows) == 2
    expected_prefill_transfer_ms = 813.75 / 1024 / 123.90 * 1000
    assert rows[0]["prefill_transfer_observed_ms"] == pytest.approx(
        expected_prefill_transfer_ms
    )
    assert rows[0]["prefill_transfer_observed_bandwidth_bytes_per_s"] == pytest.approx(
        123.90 * 1024**3
    )
    assert rows[1]["transfer_ms"] == 444.96
    assert rows[1]["bytes"] == 6510 * 131_072

    calibration = calibrator.build_calibration(
        rows,
        bytes_per_input_token=131_072,
        max_inflight_transfers=1,
    )
    observation = calibration["prefill_transfer_observation"]
    assert observation["rows"] == 1
    assert observation["boundary"] == "prefill_transfer_queue_to_completion_last_chunk"
    assert observation["duration_ms"]["median"] == pytest.approx(
        expected_prefill_transfer_ms
    )


def test_request_telemetry_parses_v0515_rows_without_bootstrap_room(
    tmp_path: Path,
):
    calibrator = _load_calibrator()
    log_path = tmp_path / "worker.log"
    log_path.write_text(
        "ReqTimeStats(rid=regular-rid, input_len=128, cached_input_len=0, "
        "output_len=32, type=decode): queue_duration=1.25ms, "
        "forward_duration=12.50ms, entry_time=1\n"
    )

    rows = calibrator.read_rows([log_path], bytes_per_input_token=131_072)

    assert len(rows) == 1
    assert rows[0]["rid"] == "regular-rid"
    assert rows[0]["bootstrap_room"] is None
    assert rows[0]["queue_ms"] == 1.25
    assert rows[0]["forward_ms"] == 12.50


def test_calibration_emits_service_profile_rows_from_request_telemetry():
    calibrator = _load_calibrator()
    rows = [
        {
            "stage": "prefill",
            "input_tokens": 100,
            "output_tokens": 10,
            "forward_ms": 40.0,
            "transfer_ms": None,
        },
        {
            "stage": "prefill",
            "input_tokens": 100,
            "output_tokens": 10,
            "forward_ms": 60.0,
            "transfer_ms": None,
        },
        {
            "stage": "decode",
            "input_tokens": 100,
            "output_tokens": 10,
            "forward_ms": 20.0,
            "transfer_ms": 200.0,
        },
    ]

    calibration = calibrator.build_calibration(
        rows,
        bytes_per_input_token=100,
        model="test-model",
        worker_class="H100_TP1",
    )

    assert calibration["profiles"] == [
        {
            "model": "test-model",
            "worker_class": "H100_TP1",
            "input_len": 100,
            "output_len": 10,
            "estimated_prefill_time": 50.0,
            "estimated_decode_time": 20.0,
            "estimated_transfer_time": 200.0,
        }
    ]


def test_calibration_records_transfer_concurrency_contract():
    calibrator = _load_calibrator()
    rows = [
        {
            "stage": "decode",
            "input_tokens": 100,
            "output_tokens": 10,
            "forward_ms": 20.0,
            "transfer_ms": 400.0,
            "effective_bandwidth_bytes_per_s": 250000.0,
        }
    ]

    calibration = calibrator.build_calibration(
        rows,
        bytes_per_input_token=100,
        max_inflight_transfers=2,
    )

    assert calibration["transfer"]["max_inflight_transfers"] == 2
    assert calibration["transfer"]["recommended_bandwidth_bytes_per_s"] == 250000.0


def test_service_profile_table_loads_calibration_artifact_with_summary_count(tmp_path):
    workflow_trace = _load_workflow_trace()
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "rows": 3,
                "profiles": [
                    {
                        "model": "test-model",
                        "worker_class": "H100_TP1",
                        "input_len": 100,
                        "output_len": 10,
                        "estimated_prefill_time": 50.0,
                        "estimated_decode_time": 20.0,
                        "estimated_transfer_time": 200.0,
                    }
                ],
            }
        )
    )

    table = workflow_trace.ServiceProfileTable.from_path(path)

    assert table.lookup("test-model", "H100_TP1", 100, 10)[
        "estimated_transfer_time"
    ] == 200.0
