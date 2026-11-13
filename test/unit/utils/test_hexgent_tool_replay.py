"""CPU-only replay/tool boundary regressions; HTTP is a local coroutine stub."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_path(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def trace_module():
    return load_path("tool_replay_trace_tests", "python/sglang/hexgent/workflow_trace.py")


def call(module, identity, *, parents=(), delay=0.0):
    return module.WorkflowCall.from_dict({
        "call_id": identity, "parents": list(parents),
        "request": {"messages": [{"role": "user", "content": identity}]},
        "input_len": 1, "output_len": 1, "tool_latency_s": delay,
    })


def test_tool_duration_roundtrips_without_becoming_a_planner_hint(tmp_path):
    module = trace_module()
    item = call(module, "root", delay=.025)
    workflow = module.WorkflowTrace(workflow_id="wf", calls=[item])
    path = tmp_path / "tools.jsonl"
    module.dump_workflow_traces_jsonl([workflow], path)
    loaded = module.load_workflow_traces_jsonl(path)[0].calls[0]
    assert getattr(loaded, "tool_latency_s", None) == .025
    assert "tool_latency_s" not in loaded.planner_hints
    assert getattr(call(module, "old-zero"), "tool_latency_s", None) == 0.0
    converted = module.convert_call_jsonl_records([{
        **item.to_dict(), "workflow_id": "wf",
    }])[0].calls[0]
    assert converted.tool_latency_s == .025


def test_tool_wait_keeps_response_service_separate_and_other_workflow_running(monkeypatch):
    module = trace_module()
    payloads = []
    async def post(_session, _url, payload, timeout_s):
        payloads.append(payload)
        await asyncio.sleep(.002)
        return {"choices": []}, {}
    monkeypatch.setattr(module, "_post_json", post)

    async def scenario():
        rows = []
        replay = module.WorkflowTraceReplayer(router_url="http://unused")
        delayed = module.WorkflowTrace("delayed", [call(module, "root", delay=.06)])
        independent = module.WorkflowTrace("independent", [call(module, "other")])
        start = module.time.perf_counter()
        delayed_task = asyncio.create_task(replay._run_workflow(None, delayed, start, rows))
        independent_task = asyncio.create_task(replay._run_workflow(None, independent, start, rows))
        await asyncio.sleep(.02)
        assert not delayed_task.done(), "root must remain incomplete during its actual tool wait"
        assert independent_task.done(), "tool wait must yield to other workflows"
        root = next(row for row in rows if row["call_id"] == "root")
        assert root["response_completed_at_ms"] is not None
        assert root["call_completed_at_ms"] is None
        assert root["total_latency_ms"] < 50
        await asyncio.gather(delayed_task, independent_task)
        assert root["tool_started_at_ms"] >= root["response_completed_at_ms"]
        assert root["tool_completed_at_ms"] >= root["tool_started_at_ms"] + 50
        assert root["call_completed_at_ms"] == root["tool_completed_at_ms"]
        assert root["total_latency_ms"] < root["tool_completed_at_ms"] - root["tool_started_at_ms"]
    asyncio.run(scenario())
    for payload in payloads:
        assert "tool_latency_s" not in json.dumps(payload["hexgent"])
        assert "tool_completed" not in json.dumps(payload["hexgent"])


def test_fanin_waits_for_all_parent_tools_and_terminal_tool_finishes_workflow(monkeypatch):
    module = trace_module()
    submitted = {}
    async def post(_session, _url, payload, timeout_s):
        submitted[payload["hexgent"]["call_id"]] = module.time.time() * 1000
        return {"choices": []}, {}
    monkeypatch.setattr(module, "_post_json", post)
    async def scenario():
        rows = []
        workflow = module.WorkflowTrace("wf", [
            call(module, "left", delay=.02), call(module, "right", delay=.05),
            call(module, "join", parents=("left", "right"), delay=.02),
        ])
        replay = module.WorkflowTraceReplayer(router_url="http://unused")
        await replay._run_workflow(None, workflow, module.time.perf_counter(), rows)
        by_id = {row["call_id"]: row for row in rows}
        assert submitted["join"] >= max(by_id[parent]["tool_completed_at_ms"] for parent in ("left", "right"))
        assert by_id["join"]["call_completed_at_ms"] > by_id["join"]["response_completed_at_ms"] + 10
        assert module.time.time() * 1000 >= by_id["join"]["tool_completed_at_ms"]
    asyncio.run(scenario())


def test_workflow_latency_consumers_include_leaf_tool_and_preserve_response_time(tmp_path, monkeypatch):
    rows = [{
        "workflow_id": "wf", "call_id": "leaf", "parents": [],
        "workflow_arrival_ms": 1000, "submitted_at": 1.0,
        "request_started_at_ms": 1000, "response_completed_at_ms": 1100,
        "call_completed_at_ms": 1250, "tool_completed_at_ms": 1250,
        "total_latency_ms": 100, "exclusive_profile_latency_ms": 100,
        "input_tokens": 1, "output_tokens": 1, "error": None,
        "response": {"model": "model", "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    }]
    summary_tests = load_path("tool_summary_helpers", "test/unit/utils/test_hexgent_scheduling_summary.py")
    summary = summary_tests._run_summary(rows, tmp_path, monkeypatch)
    observed = summary["workflows"][0]
    assert observed["latency_ms"] == 250
    assert observed["response_completed_at_ms"] == 1100
    assert observed["call_completed_at_ms"] == 1250

    support = load_path("tool_reference_support", "scripts/hexgent_real_trace_collectors/h20_pilot/h20_run_support.py")
    workflow = SimpleNamespace(workflow_id="wf", calls=[SimpleNamespace(call_id="leaf", input_len=1, output_len=1)])
    gate = support.validate_workflow_rows(workflow, rows)
    assert gate["latency_ms"] == 250
    assert gate["response_completed_at_ms"] == 1100
    assert gate["call_completed_at_ms"] == 1250

    aggregate = load_path("tool_final_aggregate", "scripts/hexgent_real_trace_collectors/h20_pilot/aggregate_final.py")
    final = aggregate.workflow_record(
        {"workflow_id": "wf", "calls": [{"call_id": "leaf", "input_len": 1, "output_len": 1}]},
        rows, {"latency_ms": 100}, "COMPLETE", "model",
    )
    assert final["observed_latency_ms"] == 250
    assert final["observed_response_completed_at_ms"] == 1100
    assert final["observed_call_completed_at_ms"] == 1250

    analysis = load_path("tool_1p1d_analysis", "scripts/hexgent_real_trace_collectors/analyze_hexgent_1p1d_run.py")
    metrics = analysis.workflow_metrics(rows)
    assert metrics["workflow_latency_ms"]["mean"] == 250

    old_rows = [{key: value for key, value in rows[0].items() if key not in {"call_completed_at_ms", "tool_completed_at_ms"}}]
    assert support.validate_workflow_rows(workflow, old_rows)["latency_ms"] == 100



def test_incremental_recorder_persists_its_own_completed_row_after_tool_wait(monkeypatch):
    module = trace_module()
    support = load_path("tool_incremental_support", "scripts/hexgent_real_trace_collectors/h20_pilot/h20_run_support.py")
    recorded = []
    async def post(_session, _url, payload, timeout_s):
        return {"choices": []}, {}
    monkeypatch.setattr(module, "_post_json", post)
    async def scenario():
        rows = []
        replayer = support.incremental_replayer_class(module, lambda row: recorded.append(dict(row)))(
            router_url="http://unused",
        )
        await asyncio.gather(
            replayer._run_workflow(None, module.WorkflowTrace("slow", [call(module, "slow-tool", delay=.04)]),
                                   module.time.perf_counter(), rows),
            replayer._run_workflow(None, module.WorkflowTrace("fast", [call(module, "fast-call")]),
                                   module.time.perf_counter(), rows),
        )
        by_id = {row["call_id"]: row for row in recorded}
        assert set(by_id) == {"slow-tool", "fast-call"}
        assert by_id["slow-tool"]["call_completed_at_ms"] >= by_id["slow-tool"]["response_completed_at_ms"] + 30
        assert recorded[-1]["call_id"] == "slow-tool"
    asyncio.run(scenario())
