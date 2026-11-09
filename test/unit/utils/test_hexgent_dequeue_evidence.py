"""CPU-only proof that worker dequeue evidence observes actual Req metadata."""
import ast
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_HELPER = _ROOT / "python/sglang/srt/managers/hexgent_worker.py"
_SCHEDULER = _ROOT / "python/sglang/srt/managers/scheduler.py"
_DECODE = _ROOT / "python/sglang/srt/disaggregation/decode.py"


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setenv("HEXGENT_FUNCTIONAL_DEQUEUE_NAMESPACE", "smoke::")
    spec = importlib.util.spec_from_file_location("hexgent_dequeue_under_test", _HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(rid="r0", *, priority=3):
    return SimpleNamespace(
        rid=rid, workflow_id="smoke::wf", hexgent_call_id="smoke::call",
        priority=priority, hexgent_prefill_priority=3, hexgent_decode_priority=7,
        hexgent_schedule_revision=9, planned_decode_instance="http://127.0.0.1:9012",
        origin_input_text="PROMPT_MUST_NEVER_BE_LOGGED", origin_input_ids=[999, 998],
        output_ids=[777],
    )


def _emit(worker, caplog, *, waiting=None, selected=None, rank=0):
    helper = getattr(worker, "log_hexgent_dequeue", None)
    assert callable(helper), "worker has no actual dequeue evidence helper"
    waiting = [_req()] if waiting is None else waiting
    selected = waiting[:1] if selected is None else selected
    with caplog.at_level(logging.INFO, logger="hexgent-dequeue-test"):
        helper(
            logger=logging.getLogger("hexgent-dequeue-test"), role="prefill",
            worker_url="http://127.0.0.1:9010", worker_port=9010, tp_rank=rank,
            waiting=waiting, selected=selected,
        )
    return [json.loads(record.message.removeprefix("HexgentDequeueEvidence "))
            for record in caplog.records]


def test_dequeue_logging_is_disabled_by_default(worker, caplog, monkeypatch):
    monkeypatch.delenv("HEXGENT_FUNCTIONAL_DEQUEUE_LOG", raising=False)
    assert _emit(worker, caplog) == []


def test_dequeue_logging_is_rank_zero_only(worker, caplog, monkeypatch):
    monkeypatch.setenv("HEXGENT_FUNCTIONAL_DEQUEUE_LOG", "1")
    assert _emit(worker, caplog, rank=1) == []


def test_empty_selection_does_not_log_or_consume_budget(worker, caplog, monkeypatch):
    monkeypatch.setenv("HEXGENT_FUNCTIONAL_DEQUEUE_LOG", "1")
    assert _emit(worker, caplog, selected=[]) == []
    assert worker._hexgent_dequeue_record_count == 0


def test_dequeue_records_actual_metadata_and_candidate_selection(worker, caplog, monkeypatch):
    monkeypatch.setenv("HEXGENT_FUNCTIONAL_DEQUEUE_LOG", "1")
    first, chosen = _req("waiting", priority=8), _req("chosen", priority=2)
    original_waiting = [first, chosen]
    selected = [chosen]
    rows = _emit(worker, caplog, waiting=original_waiting, selected=selected)
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "dequeue"
    assert (row["role"], row["worker_port"], row["tp_rank"]) == ("prefill", 9010, 0)
    assert row["worker_url"] == "http://127.0.0.1:9010"
    assert (row["waiting_count"], row["selected_count"]) == (2, 1)
    assert [item["rid"] for item in row["waiting"]] == ["waiting", "chosen"]
    assert row["selected"] == [{
        "rid": "chosen", "workflow_id": "smoke::wf", "call_id": "smoke::call",
        "priority": 2, "prefill_priority": 3, "decode_priority": 7,
        "schedule_revision": 9, "planned_decode_instance": "http://127.0.0.1:9012",
    }]
    assert row["identities_truncated"] is False
    assert original_waiting == [first, chosen] and selected == [chosen]
    assert first.priority == 8 and chosen.priority == 2
    encoded = json.dumps(row)
    assert "PROMPT_MUST_NEVER_BE_LOGGED" not in encoded
    assert "origin_input" not in encoded and "output_ids" not in encoded


def test_dequeue_event_count_cap_is_visible_once(worker, caplog, monkeypatch):
    monkeypatch.setenv("HEXGENT_FUNCTIONAL_DEQUEUE_LOG", "1")
    helper = getattr(worker, "log_hexgent_dequeue", None)
    assert callable(helper), "worker has no bounded dequeue evidence helper"
    cap = worker.HEXGENT_DEQUEUE_MAX_RECORDS
    assert cap == 64
    for _ in range(cap + 3):
        _emit(worker, caplog)
    rows = [json.loads(record.message.removeprefix("HexgentDequeueEvidence "))
            for record in caplog.records]
    assert len(rows) == cap + 1
    assert all(row["event"] == "dequeue" for row in rows[:cap])
    assert rows[-1]["record_cap_reached"] is True
    assert rows[-1]["record_cap"] == cap


def test_dequeue_record_byte_cap_is_visible(worker, caplog, monkeypatch):
    monkeypatch.setenv("HEXGENT_FUNCTIONAL_DEQUEUE_LOG", "1")
    rows = _emit(worker, caplog, waiting=[_req("x" * 20000)])
    assert rows[0]["record_byte_cap_reached"] is True
    assert rows[0]["record_byte_cap"] == 16384
    assert len(caplog.records[0].message.encode()) <= 16384


def test_dequeue_identity_cap_preserves_counts_and_reports_truncation(worker, caplog, monkeypatch):
    monkeypatch.setenv("HEXGENT_FUNCTIONAL_DEQUEUE_LOG", "1")
    requests = [_req(str(i)) for i in range(65)]
    row = _emit(worker, caplog, waiting=requests, selected=requests[:1])[0]
    assert row["waiting_count"] == 65
    assert len(row["waiting"]) == 64
    assert row["identities_truncated"] is True



def test_missing_namespace_keeps_dequeue_evidence_disabled(worker, caplog, monkeypatch):
    monkeypatch.setenv("HEXGENT_FUNCTIONAL_DEQUEUE_LOG", "1")
    monkeypatch.delenv("HEXGENT_FUNCTIONAL_DEQUEUE_NAMESPACE", raising=False)
    assert _emit(worker, caplog) == []


def test_startup_request_without_workflow_does_not_consume_owned_budget(worker, caplog, monkeypatch):
    monkeypatch.setenv("HEXGENT_FUNCTIONAL_DEQUEUE_LOG", "1")
    startup = _req("builtin-startup")
    startup.workflow_id = None
    startup.hexgent_call_id = None
    assert _emit(worker, caplog, waiting=[startup], selected=[startup]) == []
    assert worker._hexgent_dequeue_record_count == 0
    rows = _emit(worker, caplog)
    assert len(rows) == 1
    assert rows[0]["selected"][0]["workflow_id"] == "smoke::wf"


def test_nonmatching_namespace_is_ignored(worker, caplog, monkeypatch):
    monkeypatch.setenv("HEXGENT_FUNCTIONAL_DEQUEUE_LOG", "1")
    unrelated = _req("other-scope")
    unrelated.workflow_id = "another-run::wf"
    assert _emit(worker, caplog, waiting=[unrelated], selected=[unrelated]) == []
    assert worker._hexgent_dequeue_record_count == 0


def test_owned_selection_preserves_mixed_actual_identities(worker, caplog, monkeypatch):
    monkeypatch.setenv("HEXGENT_FUNCTIONAL_DEQUEUE_LOG", "1")
    owned, unexpected = _req("owned"), _req("builtin-startup")
    unexpected.workflow_id = None
    unexpected.hexgent_call_id = None
    row = _emit(worker, caplog, waiting=[unexpected, owned], selected=[owned, unexpected])[0]
    assert row["waiting_count"] == row["selected_count"] == 2
    assert [item["rid"] for item in row["waiting"]] == ["builtin-startup", "owned"]
    assert [item["rid"] for item in row["selected"]] == ["owned", "builtin-startup"]
    assert row["selected"][1]["workflow_id"] is None


def _hook(path, method_name):
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    hooks = [node for node in ast.walk(method)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "log_hexgent_dequeue"]
    assert len(hooks) == 1, "actual worker selection path must log exactly once"
    assert any(isinstance(node, ast.ImportFrom)
               and node.module == "sglang.srt.managers.hexgent_worker"
               and any(alias.name == "log_hexgent_dequeue" for alias in node.names)
               for node in tree.body), "dequeue helper must be imported by actual worker"
    return method, hooks[0]


@pytest.mark.parametrize("path,method_name,role", [
    (_SCHEDULER, "_get_new_batch_prefill_raw", "prefill"),
    (_DECODE, "get_new_prebuilt_batch", "decode"),
])
def test_actual_selection_hooks_observe_before_waiting_removal(path, method_name, role):
    method, hook = _hook(path, method_name)
    waiting_removals = [node for node in ast.walk(method) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self" and target.attr == "waiting_queue"
                                for target in node.targets)]
    assert any(node.lineno > hook.lineno for node in waiting_removals)
    selected_assignments = [node for node in ast.walk(method) if isinstance(node, ast.AnnAssign)
                            and isinstance(node.target, ast.Name) and node.target.id == "can_run_list"]
    assert any(node.lineno < hook.lineno for node in selected_assignments)
    captured = []
    self = SimpleNamespace(server_args=SimpleNamespace(host="127.0.0.1", port=9010),
                           ps=SimpleNamespace(tp_rank=0), waiting_queue=[_req("candidate")])
    selected = [self.waiting_queue[0]]
    statement = ast.Expr(value=hook)
    module = ast.fix_missing_locations(ast.Module(body=[statement], type_ignores=[]))
    exec(compile(module, str(path), "exec"), {
        "log_hexgent_dequeue": lambda **kwargs: captured.append(kwargs),
        "logger": logging.getLogger("hook-test"), "self": self, "can_run_list": selected,
    })
    assert len(captured) == 1
    row = captured[0]
    assert row["role"] == role and row["tp_rank"] == 0
    assert row["waiting"] is self.waiting_queue and row["selected"] is selected
    assert row["worker_port"] == 9010
    assert row["worker_url"] == "http://127.0.0.1:9010"
