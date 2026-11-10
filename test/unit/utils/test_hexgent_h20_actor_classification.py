"""CPU reproduction of event-maintenance telemetry and private evidence classification."""
import ast
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import test_hexgent_h20_functional_smoke as evidence
import test_hexgent_h20_transport_window as window

ACTOR_REASON = "actor_admission_endpoint_observed_in_compatible_scope"


def event_maintenance_row(tmp_path):
    # Execute the real stdlib class bodies; no FastAPI/model/GPU imports.
    tree = ast.parse(evidence.SIDECAR.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"PlannerTelemetry", "AdmissionCoordinator"}]
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *classes], type_ignores=[])
    namespace = {"Path": Path, "json": json, "threading": threading, "time": time}
    exec(compile(ast.fix_missing_locations(module), str(evidence.SIDECAR), "exec"), namespace)
    handled = []
    planner = SimpleNamespace(handle_event=handled.append, plan_pending_admissions=lambda: {})
    path = tmp_path / "real-coordinator-telemetry.jsonl"
    coordinator = namespace["AdmissionCoordinator"](planner, namespace["PlannerTelemetry"](path))
    event = SimpleNamespace(workflow_id="synthetic-w", call_id="synthetic-c", event_type="call_completed")
    coordinator.enqueue_event(event)
    assert coordinator.plan_next_batch() == 0
    assert handled == [event]
    row = json.loads(path.read_text())
    assert (row["admission_count"], row["event_count"], row["result_count"]) == (0, 1, 0)
    return row


def http_event():
    return {"method": "POST", "path": "/event", "status_code": 200,
            "request": {"event_type": "call_completed"}, "response": {"ok": True}}


def recorded_gate(tmp_path, monkeypatch, *, extra_rows=(), captures=(), capture_complete=True,
                  observer_complete=True):
    workflows, raw, planner, events, arguments = evidence.fixture()
    namespace = "synthetic-smoke::functional-smoke-ours"
    old_prefix = arguments["run_namespace"] + "::"
    for row in [*planner, *(req for event in events for key in ("waiting", "selected") for req in event[key])]:
        for key in ("workflow_id", "call_id"):
            row[key] = row[key].replace(old_prefix, namespace + "::", 1)
    module = window.load_window()
    runner = SimpleNamespace(load_module=lambda *args: evidence.load_helper(),
                             workflow_gate=lambda *args: {"successful": True})
    monkeypatch.setattr(module, "_pilot", lambda name: runner)
    ops = object.__new__(module.C1RuntimeOperations)
    ops.root = tmp_path / "runtime"
    directory = ops.root / "functional-smoke-ours"
    directory.mkdir(parents=True)
    worker_dir = ops.root / "workers"
    worker_dir.mkdir()
    ops.plan = {"run_id": "synthetic-smoke"}
    ops.calibration, ops.model = workflows, arguments["model"]
    ops._instances = lambda: arguments["worker_instances"]
    names = ("prefill", "prefill_1", "decode", "decode_1")
    ops._worker_components = lambda: names
    ops.functional_observer_summary = {**arguments["observer_summary"], "observation_complete": observer_complete}
    ops.capture_status = directory / "capture-status.latest.json"
    ops.capture_status.write_text(json.dumps({"observation_complete": capture_complete}))
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_jsonl(directory / "replay.raw.jsonl", raw)
    write_jsonl(directory / "planner-telemetry.jsonl", [*planner, *extra_rows])
    if captures is not None:
        write_jsonl(directory / "identity-capture.jsonl", captures)
    statuses = []
    for name, instance in zip(names, arguments["worker_instances"]):
        path = worker_dir / (name + ".log")
        selected = [row for row in events if row["worker_url"] == instance["worker_url"]]
        path.write_text("".join("HexgentDequeueEvidence " + json.dumps(row) + "\n" for row in selected))
        statuses.append({"name": name, "returncode": 0, "drain_finished": True,
                         "bytes_saved": path.stat().st_size, "bytes_drained": path.stat().st_size,
                         "truncated": False, "write_error": None})
    (worker_dir / "status.json").write_text(json.dumps({"processes": statuses}))
    return ops._functional_evidence(directory)


def test_real_event_only_coordinator_record_is_not_actor_admission(tmp_path, monkeypatch):
    row = event_maintenance_row(tmp_path)
    result = recorded_gate(tmp_path, monkeypatch, extra_rows=[row], captures=[http_event()])
    assert result["successful"] is True, result
    assert ACTOR_REASON not in result["failure_reasons"]
    assert result["actor_batch_owner_validation"] is False


@pytest.mark.parametrize("field", ["admission_count", "result_count"])
def test_positive_actor_activity_is_rejected(tmp_path, monkeypatch, field):
    row = {"endpoint": "/plan_pending_admissions", "admission_count": 0,
           "event_count": 1, "result_count": 0, field: 1}
    result = recorded_gate(tmp_path, monkeypatch, extra_rows=[row], captures=[http_event()])
    assert result["successful"] is False
    assert ACTOR_REASON in result["failure_reasons"]


@pytest.mark.parametrize("field", ["admission_count", "event_count", "result_count"])
def test_missing_internal_semantic_counts_are_unknown(tmp_path, monkeypatch, field):
    row = {"endpoint": "/plan_pending_admissions", "admission_count": 0,
           "event_count": 1, "result_count": 0}
    del row[field]
    result = recorded_gate(tmp_path, monkeypatch, extra_rows=[row], captures=[http_event()])
    assert result["status"] == "UNKNOWN", result
    assert "actor_activity_evidence_incomplete" in result["incomplete_reasons"]


def test_zero_event_count_cannot_explain_internal_batch(tmp_path, monkeypatch):
    row = {"endpoint": "/plan_pending_admissions", "admission_count": 0,
           "event_count": 0, "result_count": 0}
    result = recorded_gate(tmp_path, monkeypatch, extra_rows=[row], captures=[http_event()])
    assert result["status"] == "UNKNOWN"


def test_actual_enqueue_http_is_rejected_even_without_enqueue_telemetry(tmp_path, monkeypatch):
    result = recorded_gate(tmp_path, monkeypatch, captures=[{"method": "POST", "path": "/enqueue_admission"}])
    assert result["successful"] is False
    assert ACTOR_REASON in result["failure_reasons"]


def test_empty_revision_query_is_not_admission_dispatch(tmp_path, monkeypatch):
    result = recorded_gate(tmp_path, monkeypatch, captures=[{"method": "GET", "path": "/wait_plan_revision",
                                                           "response": {"revision": 0, "entries": []}}])
    assert result["successful"] is True


def test_missing_http_capture_is_unknown(tmp_path, monkeypatch):
    result = recorded_gate(tmp_path, monkeypatch, captures=None)
    assert result["status"] == "UNKNOWN"
    assert "sidecar_http_evidence_missing" in result["incomplete_reasons"]


def test_event_only_label_requires_recorded_event_http(tmp_path, monkeypatch):
    row = event_maintenance_row(tmp_path)
    result = recorded_gate(tmp_path, monkeypatch, extra_rows=[row], captures=[])
    assert result["status"] == "UNKNOWN"
    assert "actor_activity_evidence_incomplete" in result["incomplete_reasons"]


def test_event_label_does_not_override_incomplete_capture(tmp_path, monkeypatch):
    row = event_maintenance_row(tmp_path)
    result = recorded_gate(tmp_path, monkeypatch, extra_rows=[row], captures=[http_event()], capture_complete=False)
    assert result["status"] == "UNKNOWN"
    assert result["successful"] is False


def test_event_label_does_not_override_incomplete_process_observation(tmp_path, monkeypatch):
    row = event_maintenance_row(tmp_path)
    result = recorded_gate(tmp_path, monkeypatch, extra_rows=[row], captures=[http_event()], observer_complete=False)
    assert result["status"] == "UNKNOWN"
    assert "transport_observer_incomplete_or_unsafe" in result["incomplete_reasons"]
