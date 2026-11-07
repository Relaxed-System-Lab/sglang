"""CPU-only profiler usage provenance tests using synthetic SSE chunks."""

import asyncio
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "python/sglang/hexgent/service_profiler.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hexgent_profile_usage", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, chunks, status=200):
        self.chunks = chunks
        self.status = status
        self.content = self._content()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def _content(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def text(self):
        return "synthetic HTTP error"


class _Session:
    def __init__(self, chunks, status=200):
        self.response = _Response(chunks, status=status)
        self.payload = None

    def post(self, endpoint_url, *, json, timeout):
        self.payload = json
        return self.response


def _trace_call(module, request=None):
    return module.TraceCall(
        workflow_id="wf-profile-usage",
        call=module.WorkflowCall(
            call_id="c0", parents=[],
            request=request or {"messages": [{"role": "user", "content": "synthetic input"}]},
            input_len=128, output_len=32,
        ),
    )


def _event(event, *, final_newline=True):
    return ("data: " + json.dumps(event) + ("\n\n" if final_newline else "")).encode()


def _measure(module, chunks, status=200):
    session = _Session(chunks, status=status)
    result = asyncio.run(module.measure_call(
        session, "http://synthetic-profile/v1/chat/completions", _trace_call(module),
        model="synthetic-model", worker_class="synthetic-H20", trial=0,
        exact_output_length=True, timeout_s=5,
    ))
    return result, session


@pytest.mark.parametrize("final_newline", [False, True])
def test_profiler_preserves_actual_usage_instead_of_target_lengths(final_newline):
    module = _load_module()
    measurement, session = _measure(module, [
        _event({"choices": [{"delta": {"content": "arbitrary text length"}}]}),
        _event({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
               final_newline=final_newline),
    ])

    raw = asdict(measurement)
    assert raw["input_len"] == 128
    assert raw["output_len"] == 32
    assert raw["actual_prompt_tokens"] == 7
    assert raw["actual_completion_tokens"] == 3
    assert raw["usage_observed"] is True
    assert raw["output_tokens"] == 3
    assert raw["output_tokens_provenance"] == "sse_usage_completion_tokens"
    assert session.payload["max_tokens"] == 32
    assert session.payload["ignore_eos"] is True
    assert session.payload["stream_options"]["include_usage"] is True


@pytest.mark.parametrize("event", [
    {"choices": []},
    {"choices": [{"delta": {"content": "content is not a token count"}}], "usage": None},
    {"choices": [{"meta_info": {"completion_tokens": 4}}]},
])
def test_missing_usage_keeps_actual_counts_unknown(event):
    module = _load_module()
    measurement, _session = _measure(module, [_event(event)])

    assert measurement.actual_prompt_tokens is None
    assert measurement.actual_completion_tokens is None
    assert measurement.usage_observed is False
    assert measurement.output_tokens_provenance == "legacy_stream_or_target_fallback"


@pytest.mark.parametrize("usage,prompt,completion", [
    ({"prompt_tokens": 7}, 7, None),
    ({"completion_tokens": 3}, None, 3),
    ({"prompt_tokens": 7, "completion_tokens": None}, 7, None),
])
def test_incomplete_usage_preserves_only_observed_values(usage, prompt, completion):
    module = _load_module()
    measurement, _session = _measure(module, [_event({"choices": [], "usage": usage})])

    assert measurement.actual_prompt_tokens == prompt
    assert measurement.actual_completion_tokens == completion
    assert measurement.usage_observed is False
    if completion is not None:
        assert measurement.output_tokens == completion
        assert measurement.output_tokens_provenance == "sse_usage_completion_tokens"
    else:
        assert measurement.output_tokens == 32
        assert measurement.output_tokens_provenance == "legacy_stream_or_target_fallback"


def test_profiler_forces_usage_without_mutating_original_stream_options():
    module = _load_module()
    original_options = {"include_usage": False, "continuous_usage_stats": True}
    trace_call = _trace_call(module, request={
        "messages": [{"role": "user", "content": "synthetic input"}],
        "stream_options": original_options,
    })

    payload = module.build_request_payload(
        trace_call, model="synthetic-model", exact_output_length=True,
    )

    assert payload["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    assert original_options == {"include_usage": False, "continuous_usage_stats": True}
    assert payload["max_tokens"] == 32
    assert payload["ignore_eos"] is True


def test_error_measurement_defaults_do_not_invent_usage():
    module = _load_module()
    measurement = module.CallMeasurement(
        workflow_id="wf", call_id="c", model="synthetic", worker_class="synthetic",
        input_len=128, output_len=32, trial=0, ttft_ms=0, total_latency_ms=0,
        output_tokens=32, inter_token_latencies_ms=[], error="synthetic failure",
    )

    assert measurement.actual_prompt_tokens is None
    assert measurement.actual_completion_tokens is None
    assert measurement.usage_observed is False
    assert measurement.output_tokens_provenance == "legacy_unspecified"


@pytest.mark.parametrize("status,chunks,error_text", [
    (500, [], "HTTP 500"),
    (200, [_event({"usage": {"prompt_tokens": 7, "completion_tokens": 3}}),
           RuntimeError("synthetic stream failure")], "synthetic stream failure"),
    (200, [_event({"error": {"message": "synthetic SSE failure"}})], "synthetic SSE failure"),
])
def test_http_and_stream_errors_preserve_failed_raw_without_usage(status, chunks, error_text):
    module = _load_module()
    measurement, _session = _measure(module, chunks, status=status)

    assert measurement.workflow_id == "wf-profile-usage"
    assert measurement.call_id == "c0"
    assert measurement.error.startswith("RuntimeError:")
    assert error_text in measurement.error
    assert measurement.actual_prompt_tokens is None
    assert measurement.actual_completion_tokens is None
    assert measurement.usage_observed is False
    assert measurement.output_tokens_provenance == "unavailable_error"


def test_zero_observed_completion_does_not_fall_back_to_target():
    module = _load_module()
    measurement, _session = _measure(module, [
        _event({"usage": {"prompt_tokens": 7, "completion_tokens": 0}}),
    ])

    assert measurement.actual_prompt_tokens == 7
    assert measurement.actual_completion_tokens == 0
    assert measurement.usage_observed is True
    assert measurement.output_tokens == 0
    assert measurement.output_tokens_provenance == "sse_usage_completion_tokens"


def test_latency_db_excludes_error_rows_from_fits_and_worker_table():
    module = _load_module()
    prefill = [
        {"worker_class": "valid", "input_len": 100, "output_len": 10, "prefill_time_ms": 20},
        {"worker_class": "valid", "input_len": 200, "output_len": 10, "prefill_time_ms": 40},
        {"worker_class": "valid", "input_len": 300, "output_len": 10, "prefill_time_ms": 9999,
         "error": "synthetic failure"},
        {"worker_class": "failed-only", "input_len": 100, "output_len": 10, "prefill_time_ms": 9999,
         "error": "synthetic failure"},
    ]
    decode = [
        {"worker_class": "valid", "input_len": 100, "output_len": 10, "decode_time_ms": 30},
        {"worker_class": "valid", "input_len": 100, "output_len": 20, "decode_time_ms": 60},
        {"worker_class": "valid", "input_len": 100, "output_len": 30, "decode_time_ms": 9999,
         "error": "synthetic failure"},
        {"worker_class": "failed-only", "input_len": 100, "output_len": 10, "decode_time_ms": 9999,
         "error": "synthetic failure"},
    ]

    table = module.build_latency_db(prefill, decode, transfer_same_ms=12, transfer_cross_ms=26)
    valid = module.build_latency_db(prefill[:2], decode[:2], transfer_same_ms=12, transfer_cross_ms=26)

    assert table == valid
    assert set(table["hardware"]) == {"valid"}


def test_run_profile_preserves_failed_raw_and_excludes_failed_phase_rows(monkeypatch, tmp_path):
    from types import SimpleNamespace
    module = _load_module()
    trace_call = _trace_call(module)

    class SessionContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(ClientSession=SessionContext))
    monkeypatch.setattr(module, "load_trace_calls", lambda *args, **kwargs: [trace_call])

    async def failed_measurement(*args, **kwargs):
        return module.CallMeasurement(
            workflow_id=trace_call.workflow_id, call_id=trace_call.call.call_id,
            model="synthetic", worker_class="synthetic", input_len=128, output_len=32,
            trial=0, ttft_ms=0, total_latency_ms=1, output_tokens=32,
            inter_token_latencies_ms=[], error="RuntimeError: synthetic failure",
        )

    monkeypatch.setattr(module, "measure_call", failed_measurement)
    raw_path, prefill_path, decode_path = (tmp_path / name for name in ("raw.jsonl", "prefill.jsonl", "decode.jsonl"))
    args = SimpleNamespace(
        trace_path="synthetic", max_calls=None, phase="both", endpoint_url="http://synthetic",
        warmup_calls=0, trials=2, model="synthetic", worker_class="synthetic",
        disable_exact_output_length=False, timeout_s=5,
        raw_output=str(raw_path), prefill_output=str(prefill_path), decode_output=str(decode_path),
    )

    asyncio.run(module.run_profile(args))

    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert len(raw) == 2
    assert all(row["error"] and row["actual_completion_tokens"] is None for row in raw)
    assert not prefill_path.exists()
    assert not decode_path.exists()
