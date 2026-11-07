"""CPU-only endpoint AST/log-summary checks, not live HTTP/IPC evidence.

The fake TokenizerManager returns dict-backed rank fixtures. Tests execute the
real converter's dict/list traversal; native msgspec Struct serialization is a
separate H20-container runtime probe.
"""

import ast
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SERVER = _ROOT / "python/sglang/srt/entrypoints/http_server.py"
_CONVERTER = _ROOT / "python/sglang/srt/utils/msgspec_utils.py"
_SUMMARY = _ROOT / "scripts/hexgent_real_trace_collectors/summarize_worker_mutations.py"
_PREFIX = "HexgentMutationOutcome "


class _RankResult(dict):
    @property
    def success(self):
        return self["success"]


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args)


def _load_routes(results):
    logger = _Logger()
    calls = []

    async def priority(obj):
        calls.append(("priority_update", obj))
        return results

    async def reassign(obj):
        calls.append(("reassign_waiting", obj))
        return results

    manager = SimpleNamespace(
        server_args=SimpleNamespace(port=31000),
        update_hexgent_waiting_priority=priority,
        reassign_hexgent_waiting=reassign,
    )
    namespace = {
        "json": json, "logger": logger, "Any": Any,
        "_global_state": SimpleNamespace(tokenizer_manager=manager),
        # Dict-backed fixtures take the real converter's non-Struct branches.
        "msgspec": SimpleNamespace(Struct=()),
    }
    converter_tree = ast.parse(_CONVERTER.read_text())
    converter = copy.deepcopy(next(n for n in converter_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "msgspec_to_builtins"))
    tree = ast.parse(_SERVER.read_text())
    names = {"_log_hexgent_mutation_outcome", "update_hexgent_waiting_priority", "reassign_hexgent_waiting"}
    nodes = [copy.deepcopy(n) for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    code = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        converter, *nodes,
    ], type_ignores=[])
    ast.fix_missing_locations(code)
    exec(compile(code, str(_SERVER), "exec"), namespace)
    return namespace, logger, calls


@pytest.mark.parametrize("route,action_type,queue_attr", [
    ("update_hexgent_waiting_priority", "priority_update", "queue_name"),
    ("reassign_hexgent_waiting", "reassign_waiting", "target_queue_name"),
])
def test_real_route_preserves_payload_and_emits_one_complete_outcome(route, action_type, queue_attr):
    results = [
        _RankResult(success=True, reason_code="applied", current_revision=7),
        _RankResult(success=False, reason_code="stale_revision", current_revision=8),
    ]
    namespace, logger, calls = _load_routes(results)
    obj = SimpleNamespace(request_id="r1", schedule_revision=7, expected_revision=6,
                          **{queue_attr: "decode_waiting"}, prompt="not-to-be-logged")

    payload = asyncio.run(namespace[route](obj))

    assert payload == {"results": [dict(result) for result in results], "success": True}
    assert calls == [(action_type, obj)]
    assert len(logger.messages) == 1
    assert logger.messages[0].startswith(_PREFIX)
    event = json.loads(logger.messages[0][len(_PREFIX):])
    assert event == {
        "request_id": "r1", "action_type": action_type, "queue": "decode_waiting",
        "schedule_revision": 7, "expected_revision": 6, "worker_port": 31000,
        "results": payload["results"], "success": True,
    }
    assert "not-to-be-logged" not in logger.messages[0]
    assert "prompt" not in event


def _load_summary():
    assert _SUMMARY.exists(), "worker mutation summary tool is not implemented"
    spec = importlib.util.spec_from_file_location("worker_mutation_summary", _SUMMARY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(results, **changes):
    event = dict(
        worker_port=31000, request_id="r1", action_type="priority_update", queue="decode_waiting",
        schedule_revision=7, expected_revision=6,
        results=results, success=any(result["success"] for result in results),
    )
    event.update(changes)
    return event


def test_mixed_rank_attempts_and_unique_actions_keep_stale_and_applied_nonexclusive():
    module = _load_summary()
    events = [
        _event([{"success": True, "reason_code": "applied"},
                {"success": False, "reason_code": "stale_revision"}]),
        _event([{"success": True, "reason_code": "applied"},
                {"success": False, "reason_code": "expected_revision_mismatch"}]),
        _event([{"success": False, "reason_code": "not_found"},
                {"success": False, "message": "Incoming revision is stale."}], schedule_revision=8),
    ]

    report = module.summarize_events(events)

    assert report["boundary"] == "worker_endpoint_received_requests"
    assert report["received_http_attempts"] == 3
    assert report["unique_action_count"] == 2
    assert report["rank_outcome_count"] == 6
    assert report["rank_outcomes"] == {
        "success": 2, "stale_revision": 1, "expected_mismatch": 1,
        "not_found": 1, "legacy_unknown": 1, "other_rejection": 0,
    }
    assert report["attempt_any_stale_revision_count"] == 1
    assert report["attempt_any_stale_revision_rate"] == pytest.approx(1 / 3)
    assert report["rank_outcome_rates"]["stale_revision"] == pytest.approx(1 / 6)
    assert report["rank_outcome_rates"]["expected_mismatch"] == pytest.approx(1 / 6)
    assert report["unique_any_stale_revision_count"] == 1
    assert report["unique_eventual_applied_count"] == 1
    assert report["unique_any_stale_revision_rate"] == 0.5
    assert report["unique_eventual_applied_rate"] == 0.5
    action = next(a for a in report["unique_actions"] if a["schedule_revision"] == 7)
    assert action["http_attempts"] == 2
    assert action["any_stale_revision"] is True
    assert action["any_expected_mismatch"] is True
    assert action["eventual_applied"] is True
    assert report["transport_failure_count"] is None
    assert report["transport_failure_rate"] is None
    assert "not independent workflow samples" in report["statistical_unit"]


def test_expected_mismatch_and_unknown_legacy_are_not_called_stale():
    module = _load_summary()
    report = module.summarize_events([_event([
        {"success": False, "reason_code": "expected_revision_mismatch"},
        {"success": False, "message": "newer revision than incoming"},
        {"success": False, "reason_code": "unsupported_queue"},
    ])])

    assert report["rank_outcomes"]["expected_mismatch"] == 1
    assert report["rank_outcomes"]["legacy_unknown"] == 1
    assert report["rank_outcomes"]["other_rejection"] == 1
    assert report["rank_outcomes"]["stale_revision"] == 0
    assert report["attempt_any_stale_revision_count"] == 0


def test_unique_action_key_includes_worker_request_action_queue_and_both_revisions():
    module = _load_summary()
    changes = [
        {}, {"worker_port": 31001}, {"request_id": "r2"},
        {"action_type": "reassign_waiting"}, {"queue": "prefill_waiting"},
        {"schedule_revision": 8}, {"expected_revision": None},
    ]
    report = module.summarize_events([_event([], **change) for change in changes])

    assert report["received_http_attempts"] == 7
    assert report["unique_action_count"] == 7
    assert report["rank_outcome_count"] == 0
    assert all(rate is None for rate in report["rank_outcome_rates"].values())


def test_zero_attempt_rates_are_unknown_instead_of_zero():
    module = _load_summary()
    report = module.summarize_events([])

    assert report["received_http_attempts"] == 0
    assert report["unique_action_count"] == 0
    assert report["rank_outcome_count"] == 0
    assert report["attempt_any_stale_revision_rate"] is None
    assert report["unique_any_stale_revision_rate"] is None
    assert report["unique_eventual_applied_rate"] is None
    assert all(rate is None for rate in report["rank_outcome_rates"].values())
    assert report["transport_failure_count"] is None


def test_cli_reads_only_prefixed_events_from_explicit_logs(tmp_path):
    _load_summary()
    first, second = tmp_path / "worker-0.log", tmp_path / "worker-1.log"
    event = _event([{"success": False, "reason_code": "stale_revision"}])
    first.write_text("unrelated worker warning\n[2026-11-07 INFO] " + _PREFIX + json.dumps(event) + "\n")
    second.write_text(_PREFIX + json.dumps(event) + "\n")
    output = tmp_path / "mutations.json"

    result = subprocess.run([
        sys.executable, str(_SUMMARY), "--log", str(first), "--log", str(second),
        "--output", str(output),
    ], cwd=_ROOT, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["received_http_attempts"] == 2
    assert report["unique_action_count"] == 1
    assert report["unique_actions"][0]["http_attempts"] == 2
