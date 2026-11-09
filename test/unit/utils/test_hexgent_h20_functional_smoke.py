"""Synthetic metadata fixtures; these tests do not run a model or prove runtime behavior."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts/hexgent_real_trace_collectors/h20_functional_smoke.py"
SIDECAR = ROOT / "python/sglang/hexgent/sidecar.py"


def load_helper():
    assert HELPER.exists(), "functional evidence helper is not implemented"
    spec = importlib.util.spec_from_file_location("functional_evidence_under_test", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    # Synthetic lengths preserve the accepted aggregate; these are not real requests.
    instances = [
        {"instance_id": identity, "role": role, "worker_url": f"http://127.0.0.1:{port}"}
        for identity, role, port in (
            ("P0", "prefill", 37110), ("P1", "prefill", 37111),
            ("D0", "decode", 37130), ("D1", "decode", 37131))
    ]
    namespace, model = "synthetic-smoke::ours", "synthetic-70b"
    workflows, raw, planner, events = [], [], [], []
    for w in range(2):
        workflow_id = f"synthetic-workflow-{w}"
        calls = []
        for c in range(5):
            i = w * 5 + c
            call_id = f"synthetic-call-{i}"
            input_len, output_len = (676 if i == 9 else 500), (197 if i == 9 else 195)
            calls.append(SimpleNamespace(call_id=call_id, input_len=input_len, output_len=output_len,
                                         parents=[] if c == 0 else [f"synthetic-call-{i - 1}"]))
            revision = i + 1
            raw.append({
                "workflow_id": workflow_id, "call_id": call_id,
                "workflow_arrival_ms": 1000.0, "request_started_at_ms": 1100.0 + c * 100,
                "response_completed_at_ms": 1150.0 + c * 100, "error": None,
                "input_tokens": input_len, "output_tokens": output_len,
                "prefill_worker": instances[0]["worker_url"], "decode_worker": instances[2]["worker_url"],
                "router_mode": "hexgent", "schedule_revision": revision,
                "response": {"model": model, "choices": [{"finish_reason": "length", "message": {"content": "synthetic"}}],
                             "usage": {"prompt_tokens": input_len, "completion_tokens": output_len}},
            })
            planner.append({
                "endpoint": "/schedule_request",
                "workflow_id": namespace + "::" + workflow_id, "call_id": namespace + "::" + call_id,
                "revision": revision, "prefill_worker_url": instances[0]["worker_url"],
                "decode_worker_url": instances[2]["worker_url"],
                "prefill_priority": i + 10, "decode_priority": i + 20,
                "planned_decode_instance": "127.0.0.1:37130/dp0", "fallback_reason": None, "error": None,
                "phase_timings_ms": {"prefill_solver": 0.1, "decode_solver": 0.2},
            })
            for role, url, priority in (
                ("prefill", instances[0]["worker_url"], i + 10),
                ("decode", instances[2]["worker_url"], i + 20),
            ):
                req = {
                    "rid": f"synthetic-rid-{i}", "workflow_id": namespace + "::" + workflow_id,
                    "call_id": namespace + "::" + call_id, "priority": priority,
                    "prefill_priority": i + 10, "decode_priority": i + 20,
                    "schedule_revision": revision, "planned_decode_instance": "127.0.0.1:37130/dp0",
                }
                events.append({"event": "dequeue", "role": role, "worker_url": url, "tp_rank": 0,
                               "waiting_count": 1, "selected_count": 1,
                               "waiting": [copy.deepcopy(req)], "selected": [req],
                               "record_cap": 64, "record_byte_cap": 16384})
        workflows.append(SimpleNamespace(workflow_id=workflow_id, calls=calls))
    kwargs = {
        "model": model, "run_namespace": namespace, "worker_instances": instances,
        "observer_summary": {"observation_complete": True, "first_error_unix_s": None,
                             "stop_requested": False, "incomplete_reasons": []},
        "capture_complete": True, "worker_capture_complete": True,
    }
    return workflows, raw, planner, events, kwargs


def run_gate(data):
    workflows, raw, planner, events, kwargs = data
    return load_helper().functional_smoke_gate(workflows, raw, planner, events, **kwargs)


def test_complete_join_is_functional_only_and_reports_actual_replica_use():
    result = run_gate(fixture())
    assert result["successful"] is True
    assert result["status"] == "PASSED"
    assert result["completed_calls"] == result["solver_joined_calls"] == result["worker_joined_calls"] == 10
    assert result["workflow_completed_count"] == 2
    assert result["observed_prompt_tokens"] == 5176
    assert result["observed_completion_tokens"] == 1952
    assert result["selected_replica_call_counts"] == {"P0": 10, "P1": 0, "D0": 10, "D1": 0}
    assert result["queue_contention"]["multi_candidate_dequeue_events"] == 0
    assert result["actor_batch_owner_validation"] is False
    assert result["performance_claims"] is False
    assert "inferred" in result["kv_transfer_evidence"]


@pytest.mark.parametrize("change", [
    "missing_raw", "duplicate_raw", "usage", "model", "request_error", "router_fallback",
    "response_assignment", "unfinished_generation", "parent_order",
])
def test_response_or_workflow_failure_cannot_pass(change):
    data = fixture()
    _, rows, _, _, _ = data
    if change == "missing_raw":
        rows.pop()
    elif change == "duplicate_raw":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "usage":
        rows[0]["response"]["usage"]["completion_tokens"] -= 1
    elif change == "model":
        rows[0]["response"]["model"] = "wrong-model"
    elif change == "request_error":
        rows[0]["error"] = "synthetic HTTP failure"
    elif change == "router_fallback":
        rows[0]["router_mode"] = "pd_policy"
    elif change == "response_assignment":
        rows[0]["decode_worker"] = "http://127.0.0.1:37131"
    elif change == "unfinished_generation":
        rows[0]["response"]["choices"][0]["finish_reason"] = None
    elif change == "parent_order":
        rows[1]["request_started_at_ms"] = rows[0]["response_completed_at_ms"] - 1
    result = run_gate(data)
    assert result["successful"] is False
    assert result["failure_reasons"] or result["incomplete_reasons"]


@pytest.mark.parametrize("change", [
    "missing_solver", "duplicate_solver", "zero_revision", "different_revision",
    "fallback", "unknown_fallback", "missing_priority", "missing_solver_phase", "solver_error",
])
def test_missing_or_unmatched_solver_result_cannot_pass(change):
    data = fixture()
    planner = data[2]
    if change == "missing_solver":
        planner.pop()
    elif change == "duplicate_solver":
        planner.append(copy.deepcopy(planner[0]))
    elif change == "zero_revision":
        planner[0]["revision"] = 0
    elif change == "different_revision":
        planner[0]["revision"] = 999
    elif change == "fallback":
        planner[0]["fallback_reason"] = "synthetic fallback"
    elif change == "unknown_fallback":
        del planner[0]["fallback_reason"]
    elif change == "missing_priority":
        del planner[0]["prefill_priority"]
    elif change == "missing_solver_phase":
        del planner[0]["phase_timings_ms"]["decode_solver"]
    elif change == "solver_error":
        planner[0]["error"] = "synthetic solver failure"
    assert run_gate(data)["successful"] is False


@pytest.mark.parametrize("change", [
    "missing_dequeue", "wrong_priority", "wrong_rank", "wrong_revision", "wrong_worker",
    "not_waiting", "candidate_only", "record_cap", "byte_cap", "identities_truncated",
])
def test_candidate_or_incomplete_worker_evidence_cannot_prove_execution(change):
    data = fixture()
    events = data[3]
    if change == "missing_dequeue":
        events.pop()
    elif change == "wrong_priority":
        events[0]["selected"][0]["priority"] += 1
    elif change == "wrong_rank":
        events[0]["selected"][0]["decode_priority"] += 1
    elif change == "wrong_revision":
        events[0]["selected"][0]["schedule_revision"] += 1
    elif change == "wrong_worker":
        events[0]["worker_url"] = "http://127.0.0.1:37111"
    elif change == "not_waiting":
        events[0]["waiting"] = []
    elif change == "candidate_only":
        events[0]["selected"] = []
        events[0]["selected_count"] = 0
    elif change == "record_cap":
        events.append({"event": "record_cap_reached"})
    elif change == "byte_cap":
        events.append({"event": "dequeue", "record_byte_cap_reached": True})
    elif change == "identities_truncated":
        events[0]["identities_truncated"] = True
    assert run_gate(data)["successful"] is False


def test_reports_contention_without_requiring_it_or_claiming_an_effect():
    data = fixture()
    waiting = copy.deepcopy(data[3][0]["waiting"][0])
    waiting.update(rid="other-synthetic-rid", call_id="unselected-candidate", priority=-1)
    data[3][0]["waiting"].append(waiting)
    data[3][0]["waiting_count"] = 2
    result = run_gate(data)
    assert result["successful"] is True
    assert result["queue_contention"]["multi_candidate_dequeue_events"] == 1
    assert result["queue_contention"]["events_with_unselected_candidates"] == 1
    assert result["queue_contention"]["distinct_priority_competition_events"] == 1


@pytest.mark.parametrize("source", ["observer", "sidecar", "worker"])
def test_incomplete_capture_is_unknown(source):
    data = fixture()
    if source == "observer":
        data[4]["observer_summary"]["observation_complete"] = False
    elif source == "sidecar":
        data[4]["capture_complete"] = False
    else:
        data[4]["worker_capture_complete"] = False
    result = run_gate(data)
    assert result["status"] == "UNKNOWN"
    assert result["successful"] is False


def test_log_reader_ignores_other_logs_and_retains_malformed_evidence(tmp_path):
    path = tmp_path / "prefill.log"
    event = fixture()[3][0]
    path.write_text("ordinary log\n[worker] HexgentDequeueEvidence " + json.dumps(event) +
                    "\nHexgentDequeueEvidence {broken\n")
    events = load_helper().read_dequeue_events([path])
    assert events[0]["event"] == "dequeue"
    assert events[1]["event"] == "capture_incomplete"


def test_missing_log_file_is_explicitly_incomplete(tmp_path):
    events = load_helper().read_dequeue_events([tmp_path / "missing.log"])
    assert events[0]["event"] == "capture_incomplete"


def test_real_telemetry_record_keeps_solver_ranks_and_explicit_no_fallback(tmp_path):
    # Execute only the real stdlib telemetry class, without importing FastAPI/SGLang.
    tree = ast.parse(SIDECAR.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PlannerTelemetry")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
                        type_ignores=[])
    namespace = {"Path": Path, "json": json, "threading": threading, "time": time}
    exec(compile(ast.fix_missing_locations(module), str(SIDECAR), "exec"), namespace)
    path = tmp_path / "telemetry.jsonl"
    telemetry = namespace["PlannerTelemetry"](path)
    telemetry.record(endpoint="/schedule_request",
                     payload={"hexgent": {"workflow_id": "synthetic-w", "call_id": "synthetic-c"}},
                     result={"revision": 7, "prefill_priority": 2, "decode_priority": 3,
                             "fallback_reason": None},
                     wall_started_ns=time.perf_counter_ns(), cpu_started_ns=time.process_time_ns())
    row = json.loads(path.read_text())
    assert row.get("prefill_priority") == 2, "real telemetry omits solver prefill rank"
    assert row.get("decode_priority") == 3
    assert "fallback_reason" in row and row["fallback_reason"] is None
