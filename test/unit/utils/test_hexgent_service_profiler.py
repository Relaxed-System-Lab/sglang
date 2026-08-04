import importlib.util
from pathlib import Path
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "python" / "sglang" / "hexgent" / "service_profiler.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hexgent_service_profiler", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sse_parser_extracts_chat_content_and_usage():
    module = _load_module()
    parsed = list(
        module.iter_sse_json_events(
            [
                b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"B"}}],"usage":{"completion_tokens":2}}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
    )

    assert [event["choices"][0]["delta"]["content"] for event in parsed] == ["A", "B"]
    assert parsed[-1]["usage"]["completion_tokens"] == 2


def test_build_phase_rows_keeps_prefill_and_decode_tables_separate():
    module = _load_module()
    measurement = module.CallMeasurement(
        workflow_id="wf-1",
        call_id="call-1",
        model="m",
        worker_class="A100_80G",
        input_len=128,
        output_len=32,
        trial=0,
        ttft_ms=25.0,
        total_latency_ms=90.0,
        output_tokens=32,
        inter_token_latencies_ms=[2.0, 3.0, 4.0],
    )

    prefill = module.build_prefill_row(measurement)
    decode = module.build_decode_row(measurement)

    assert prefill["phase"] == "prefill"
    assert prefill["prefill_time_ms"] == pytest.approx(25.0)
    assert "decode_time_per_token_ms" not in prefill
    assert decode["phase"] == "decode"
    assert decode["decode_time_ms"] == pytest.approx(65.0)
    assert decode["decode_time_per_token_ms"] == pytest.approx(65.0 / 31.0)


def test_decode_row_uses_total_latency_minus_ttft_when_stream_has_single_chunk():
    module = _load_module()
    measurement = module.CallMeasurement(
        workflow_id="wf-1",
        call_id="call-1",
        model="m",
        worker_class="A100_80G",
        input_len=128,
        output_len=32,
        trial=0,
        ttft_ms=25.0,
        total_latency_ms=90.0,
        output_tokens=32,
        inter_token_latencies_ms=[],
    )

    decode = module.build_decode_row(measurement)

    assert decode["decode_time_ms"] == pytest.approx(65.0)
    assert decode["decode_time_per_token_ms"] == pytest.approx(65.0 / 31.0)


def test_decode_row_uses_wall_clock_decode_time_for_chunked_streaming():
    module = _load_module()
    measurement = module.CallMeasurement(
        workflow_id="wf-1",
        call_id="call-1",
        model="qwen",
        worker_class="H100_TP4",
        input_len=128,
        output_len=64,
        trial=0,
        ttft_ms=100.0,
        total_latency_ms=730.0,
        output_tokens=64,
        inter_token_latencies_ms=[90.0, 90.0, 90.0],
    )

    decode = module.build_decode_row(measurement)

    assert decode["decode_time_ms"] == pytest.approx(630.0)
    assert decode["decode_time_per_token_ms"] == pytest.approx(630.0 / 63.0)
    assert decode["decode_p50_ms"] == pytest.approx(630.0 / 63.0)
    assert decode["decode_p95_ms"] == pytest.approx(630.0 / 63.0)
    assert decode["timing_source"] == "wall_clock_total_latency_minus_ttft_ms"


def test_latency_db_fit_uses_separate_prefill_and_decode_rows():
    module = _load_module()
    prefill_rows = [
        {"worker_class": "A100_80G", "input_len": 100, "output_len": 10, "prefill_time_ms": 20.0},
        {"worker_class": "A100_80G", "input_len": 200, "output_len": 10, "prefill_time_ms": 40.0},
    ]
    decode_rows = [
        {"worker_class": "A100_80G", "input_len": 100, "output_len": 10, "decode_time_ms": 30.0},
        {"worker_class": "A100_80G", "input_len": 100, "output_len": 20, "decode_time_ms": 60.0},
    ]

    latency_db = module.build_latency_db(
        prefill_rows,
        decode_rows,
        transfer_same_ms=12.0,
        transfer_cross_ms=26.0,
    )

    hardware = latency_db["hardware"]["A100_80G"]
    assert hardware["prefill_cold"]["input_coeff_ms"] > 0.0
    assert hardware["prefill_hot"]["input_coeff_ms"] > 0.0
    assert hardware["decode"]["output_coeff_ms"] > 0.0
    assert latency_db["transfer"]["same_kind"]["base_ms"] == pytest.approx(12.0)
