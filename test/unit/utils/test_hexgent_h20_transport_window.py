"""Offline C1 window contract tests; all runtime operations are injected."""

import asyncio
import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/hexgent_real_trace_collectors/h20_transport_window.py"
PILOT = ROOT / "scripts/hexgent_real_trace_collectors/h20_pilot"
OWNER = "a" * 64
RUN = "synthetic-c1-window"
MODEL = "synthetic-model"
REVISION = "b" * 40
EPOCH = "c" * 64
PLAN_SHA = "d" * 64
NAMESPACE = RUN + "::calibration-ours"
BIG_REVISION = 1794070240492103252


def load_path(name, path):
    assert path.is_file(), "C1 transport-window implementation is not present yet"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_window():
    return load_path("_h20_transport_window_test", SCRIPT)


def workflow_api():
    support = load_path("_c1_frozen_support_test", PILOT / "h20_run_support.py")
    return support.load_workflow_api(ROOT)


def workflows(ids=("cal::one", "cal::two"), *, fanout=False):
    api = workflow_api()
    graph = [("root", [])]
    if fanout:
        graph += [("left", ["root"]), ("right", ["root"])]
    return [api.WorkflowTrace(workflow_id=wid, workflow_type="synthetic",
        arrival_offset=0.0, calls=[api.WorkflowCall(call_id=cid, parents=parents,
            request={"messages": [{"role": "user", "content": "retained prompt"}]},
            input_len=10, output_len=2,
            planner_hints={"model_config": {}, "hardware_params": {}})
            for cid, parents in graph]) for wid in ids]


def raw_rows(items, revision=BIG_REVISION):
    ends = {"root": 1010.0, "left": 1030.0, "right": 1050.0}
    return [{"workflow_id": item.workflow_id, "call_id": call.call_id,
        "parents": list(call.parents), "workflow_arrival_ms": 1000.0,
        "request_started_at_ms": 1000.0 if call.call_id == "root" else 1010.0,
        "response_completed_at_ms": ends[call.call_id],
        "input_tokens": call.input_len, "output_tokens": call.output_len,
        "prefill_worker": "http://prefill", "decode_worker": "http://decode",
        "schedule_revision": revision + index if revision > 0 else revision, "error": None,
        "response": {"model": MODEL, "usage": {"prompt_tokens": 10, "completion_tokens": 2}}}
        for index, (item, call) in enumerate((item, call) for item in items for call in item.calls)]


def capture_records(items, *, namespace=NAMESPACE):
    records = []
    entries = []
    for item in items:
        for call in item.calls:
            wf, cid = namespace + "::" + item.workflow_id, namespace + "::" + call.call_id
            entry_revision = BIG_REVISION + len(entries)
            schedule = {"revision": entry_revision, "prefill_worker_url": "http://prefill",
                        "decode_worker_url": "http://decode", "prefill_priority": 0,
                        "decode_priority": 0}
            records.append({"method": "POST", "path": "/enqueue_admission", "status_code": 200,
                "request": {"request": {"hexgent": {"workflow_id": wf, "call_id": cid}}},
                "response": {"dispatch_ready": False, "schedule": None}})
            entries.append({"workflow_id": wf, "call_id": cid, "schedule": schedule})
            records.append({"method": "POST", "path": "/event", "status_code": 200,
                "request": {"event_type": "request_dispatched", "workflow_id": wf,
                    "call_id": cid, "timestamp_ms": 1001,
                    "payload": {"schedule_revision": entry_revision,
                        "prefill_worker_url": "http://prefill", "decode_worker_url": "http://decode"}},
                "response": {"ok": True}})
    # Only the per-entry revision joins; the batch maximum need not equal it.
    records.append({"method": "GET", "path": "/wait_plan_revision", "status_code": 200,
        "request": None, "response": {"revision": max(entry["schedule"]["revision"] for entry in entries), "entries": entries}})
    return records


def observer(**changes):
    value = {"schema_version": 1, "run_id": RUN, "owner_id": OWNER,
        "sampled_unix_s": 1100.0, "observation_complete": True,
        "first_error_unix_s": None, "stop_requested": False, "incomplete_reasons": [],
        "anonymous_namespace_counts": {"TIME_WAIT": 1000}, "owned_details": {}}
    value.update(changes)
    return value


def approval():
    return {"approved_plan_sha256": PLAN_SHA, "window_run_id": RUN}


def new_window(module, *, clock=lambda: 1100.0):
    return module.TransportWindow(run_id=RUN, budget=module.WindowBudget(anchor_unix_s=1000.0),
        plan_sha256=PLAN_SHA, clock=clock)


def cal_gate(module, items, rows, records=(), **kwargs):
    return module.calibration_gate("ours", items, rows, model=MODEL,
        run_namespace=NAMESPACE, capture_records=records, capture_complete=True, **kwargs)


def test_global_deadline_is_frozen_and_children_use_remaining_original_budget():
    module = load_window()
    budget = module.WindowBudget(anchor_unix_s=1000.0)
    assert budget.hard_deadline_unix_s == 3700.0
    assert budget.work_deadline_unix_s == 3520.0
    assert budget.child_timeout(3400.0, requested_s=180) == 120.0
    assert budget.child_timeout(3510.0, requested_s=180) == 10.0
    with pytest.raises(FrozenInstanceError):
        budget.anchor_unix_s = 3510.0
    with pytest.raises(TimeoutError):
        budget.child_timeout(3520.0)


@pytest.mark.parametrize("kwargs", [{"duration_s": 2701}, {"cleanup_s": 179}])
def test_window_cannot_expand_the_45minute_cap_or_consume_cleanup_reserve(kwargs):
    with pytest.raises(ValueError):
        load_window().WindowBudget(anchor_unix_s=1000, **kwargs)


def test_pair_budget_keeps_original_formula_and_separate_180second_cleanup():
    module = load_window()
    budget = module.WindowBudget(anchor_unix_s=1000)
    assert module.pair_budget_s([80000, 110000], last_arrival_s=0) == 335.0
    assert budget.pair_fits(3185.0, [80000, 110000], last_arrival_s=0)
    assert not budget.pair_fits(3185.001, [80000, 110000], last_arrival_s=0)


@pytest.mark.parametrize("dry_run", [False, True])
def test_default_plan_only_and_dryrun_never_execute_startup_or_cleanup(dry_run):
    module = load_window()
    window = new_window(module)
    window.register_owned_container(OWNER)
    calls = []
    async def operation(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("plan-only must never execute an operation")
    asyncio.run(window.run_stage("startup", operation, execute=dry_run,
        dry_run=dry_run, approval=approval()))
    asyncio.run(window.cleanup(OWNER, operation, execute=dry_run,
        dry_run=dry_run, approval=approval()))
    assert calls == []


def test_execute_requires_explicit_matching_frozen_plan_approval():
    module = load_window()
    window = new_window(module)
    calls = []
    async def operation(**kwargs):
        calls.append(kwargs)
    with pytest.raises(PermissionError):
        asyncio.run(window.run_stage("startup", operation, execute=True, approval=None))
    with pytest.raises(PermissionError):
        asyncio.run(window.run_stage("startup", operation, execute=True,
            approval={"approved_plan_sha256": "e" * 64, "window_run_id": RUN}))
    assert calls == []


def test_stage_timeout_and_operation_deadline_share_the_single_window_anchor(monkeypatch):
    module = load_window()
    window = new_window(module, clock=lambda: 3400.0)
    window.register_owned_container(OWNER)
    seen = []
    async def fake_wait_for(awaitable, timeout):
        seen.append(("timeout", timeout))
        return await awaitable
    async def operation(*, deadline_unix_s):
        seen.append(("deadline", deadline_unix_s))
        return {"successful": True}
    monkeypatch.setattr(module.asyncio, "wait_for", fake_wait_for)
    asyncio.run(window.run_stage("calibration-native", operation,
        observer_summary=observer(sampled_unix_s=3400), execute=True, approval=approval()))
    assert seen == [("timeout", 120.0), ("deadline", 3520.0)]


@pytest.mark.parametrize("changes", [
    {"first_error_unix_s": 1099.0}, {"stop_requested": True},
    {"observation_complete": False, "incomplete_reasons": ["permission denied"]},
    {"run_id": "other-window"}, {"owner_id": "f" * 64},
])
def test_first_error_or_incomplete_observation_stickily_stops_new_stages(changes):
    module = load_window()
    window = new_window(module)
    window.register_owned_container(OWNER)
    calls = []
    async def operation(**kwargs):
        calls.append(kwargs)
    with pytest.raises(module.WindowStopped):
        asyncio.run(window.run_stage("calibration-native", operation,
            observer_summary=observer(**changes), execute=True, approval=approval()))
    with pytest.raises(module.WindowStopped):
        asyncio.run(window.run_stage("calibration-ours", operation,
            observer_summary=observer(), execute=True, approval=approval()))
    assert calls == []


@pytest.mark.parametrize("alarm_changes", [
    {"first_error_unix_s": 1101.0, "stop_requested": True},
    {"observation_complete": False, "stop_requested": True, "incomplete_reasons": ["observer cap"]},
])
def test_error_or_incomplete_alarm_during_replay_cancels_inflight_then_exact_owned_cleanup(alarm_changes):
    module = load_window()
    window = new_window(module)
    window.register_owned_container(OWNER)
    calls = []
    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        async def replay(*, deadline_unix_s):
            calls.append("replay-started")
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        async def alarm():
            await started.wait()
            return observer(sampled_unix_s=1101.0, **alarm_changes)
        with pytest.raises(module.WindowStopped):
            await window.run_stage("calibration-native", replay, observer_summary=observer(),
                observer_watch=alarm, execute=True, approval=approval())
        assert cancelled.is_set()
        with pytest.raises(module.WindowStopped):
            await window.run_stage("calibration-ours", replay, observer_summary=observer(),
                execute=True, approval=approval())
        async def cleanup(*, owner_id, deadline_unix_s):
            calls.append((owner_id, deadline_unix_s))
        await window.cleanup(OWNER, cleanup, execute=True, approval=approval())
    asyncio.run(scenario())
    assert calls == ["replay-started", (OWNER, 3700.0)]


def test_stage_failure_is_not_retried_and_cleanup_keeps_hard_deadline():
    module = load_window()
    window = new_window(module)
    window.register_owned_container(OWNER)
    calls = []
    async def fail(*, deadline_unix_s):
        calls.append("failed-once")
        raise RuntimeError("synthetic first transport failure")
    async def later(*, deadline_unix_s):
        calls.append("unapproved later stage")
    async def cleanup(*, owner_id, deadline_unix_s):
        calls.append((owner_id, deadline_unix_s))
    with pytest.raises(RuntimeError, match="synthetic first transport failure"):
        asyncio.run(window.run_stage("calibration-native", fail,
            observer_summary=observer(), execute=True, approval=approval()))
    with pytest.raises(module.WindowStopped):
        asyncio.run(window.run_stage("calibration-ours", later,
            observer_summary=observer(), execute=True, approval=approval()))
    asyncio.run(window.cleanup(OWNER, cleanup, execute=True, approval=approval()))
    assert calls == ["failed-once", (OWNER, 3700.0)]


@pytest.mark.parametrize("target", ["a" * 12, "f" * 64, "pkill", "kill -9 123", ""])
def test_cleanup_rejects_prefix_foreign_or_ambiguous_targets_before_operation(target):
    module = load_window()
    window = new_window(module)
    window.register_owned_container(OWNER)
    calls = []
    async def operation(**kwargs):
        calls.append(kwargs)
    with pytest.raises(ValueError):
        asyncio.run(window.cleanup(target, operation, execute=True, approval=approval()))
    assert calls == []


def test_container_owner_cannot_be_replaced_or_registered_by_prefix():
    module = load_window()
    window = new_window(module)
    with pytest.raises(ValueError):
        window.register_owned_container(OWNER[:12])
    window.register_owned_container(OWNER)
    with pytest.raises(ValueError):
        window.register_owned_container("f" * 64)


def test_native_calibration_accepts_complete_zero_revision_without_waiting_or_plans():
    module = load_window()
    items = workflows()
    result = module.calibration_gate("native", items, raw_rows(items, revision=0),
        model=MODEL, run_namespace=RUN + "::calibration-native", capture_records=[],
        capture_complete=True, worker_snapshots=[])
    assert result["successful"]
    assert result["canonical_required"] is False
    assert result["expected_calls"] == result["completed_calls"] == 2


def test_ours_calibration_joins_exact_entry_revision_dispatch_and_raw_with_real_competition():
    module = load_window()
    items = workflows()
    result = cal_gate(module, items, raw_rows(items), capture_records(items))
    assert result["successful"]
    assert result["canonical_required"] is True
    assert result["canonical_joined_calls"] == 2
    assert result["canonical_denominator"] == 2
    assert result["pressure_observed"] is True
    assert result["fallback_calls"] == result["missing_revision_calls"] == 0


@pytest.mark.parametrize("change", ["counts_only", "wrong_namespace", "wrong_workflow",
    "wrong_call", "wrong_revision", "wrong_pair", "no_enqueue", "no_plan", "no_dispatch"])
def test_ours_canonical_gate_rejects_unjoined_fields_counts_and_wrong_identities(change):
    module = load_window()
    items = workflows()
    rows = raw_rows(items)
    records = capture_records(items)
    if change == "counts_only":
        records = [{"endpoint": "/plan_pending_admissions", "admission_count": 2, "result_count": 2}]
    elif change == "wrong_namespace":
        records = capture_records(items, namespace=RUN + "::warmup")
    elif change in {"wrong_workflow", "wrong_call", "wrong_revision", "wrong_pair"}:
        for record in records:
            if record.get("path") == "/event":
                event = record["request"]
                if change == "wrong_workflow": event["workflow_id"] += "-other"
                elif change == "wrong_call": event["call_id"] += "-other"
                elif change == "wrong_revision": event["payload"]["schedule_revision"] += 1
                else: event["payload"]["decode_worker_url"] = "http://other-decode"
    else:
        path = {"no_enqueue": "/enqueue_admission", "no_plan": "/wait_plan_revision", "no_dispatch": "/event"}[change]
        records = [record for record in records if record.get("path") != path]
    result = cal_gate(module, items, rows, records)
    assert not result["successful"]
    assert result["canonical_joined_calls"] == 0


def test_one_joined_call_needs_actual_waiting_not_overlap_running_or_field_presence():
    module = load_window()
    items = workflows(ids=("cal::one",))
    records = capture_records(items)
    result = cal_gate(module, items, raw_rows(items), records)
    assert result["canonical_joined_calls"] == 1
    assert not result["pressure_observed"]
    assert not result["successful"]
    snapshots = [scoped_waiting_evidence()]
    waiting = cal_gate(module, items, raw_rows(items), records, worker_snapshots=snapshots,
        owned_worker_urls={"http://prefill", "http://decode"}, phase_bounds_unix_s=(1.0, 1.01))
    assert waiting["successful"] and waiting["pressure_observed"]


@pytest.mark.parametrize("change", ["capture_incomplete", "raw_error", "wrong_usage"])
def test_ours_capture_or_usage_failure_preserves_failed_gate(change):
    module = load_window()
    items = workflows()
    rows = raw_rows(items)
    records = capture_records(items)
    if change == "raw_error": rows[0].update(error="synthetic HTTP failure", response=None)
    elif change == "wrong_usage": rows[0]["response"]["usage"]["completion_tokens"] = 1
    result = module.calibration_gate("ours", items, rows, model=MODEL,
        run_namespace=NAMESPACE, capture_records=records,
        capture_complete=change != "capture_incomplete", worker_snapshots=[])
    assert not result["successful"]
    assert result["expected_calls"] == 2


def reference_fixture(module):
    items = workflows(ids=tuple("hotpotqa_" + str(index) for index in range(10, 14)), fanout=True)
    rows = raw_rows(items, revision=0)
    by_workflow = {item.workflow_id: [row for row in rows if row["workflow_id"] == item.workflow_id]
                   for item in items}
    raw_hashes = {item.workflow_id: "e" * 64 for item in items}
    contract = {"model": MODEL, "implementation_revision": REVISION, "runtime_evidence_sha256": EPOCH}
    references = module.hot_reference_rows(items, by_workflow,
        raw_sha256_by_workflow=raw_hashes, **contract)
    return items, references, contract


def test_four_hot_references_use_actual_whole_workflow_elapsed_and_current_epoch():
    module = load_window()
    items, references, contract = reference_fixture(module)
    assert len(references) == 4
    assert all(row["latency_ms"] == 50.0 for row in references)
    assert all(row["reference_kind"] == "isolated_workflow_replay" for row in references)
    assert all(row["expected_calls"] == row["usage_verified_calls"] == 3 for row in references)
    assert module.validate_hot_references(items, references, contract)["successful"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "old_epoch", "old_source", "failed", "partial", "nonfinite"])
def test_four_hot_reference_gate_rejects_incomplete_or_mixed_epoch_without_proxy_fallback(change):
    module = load_window()
    items, references, contract = reference_fixture(module)
    if change == "missing": references.pop()
    elif change == "duplicate": references[-1] = copy.deepcopy(references[0])
    elif change == "old_epoch": references[0]["runtime_evidence_sha256"] = "f" * 64
    elif change == "old_source": references[0]["implementation_revision"] = "f" * 40
    elif change == "failed": references[0].update(status="failed", latency_ms=None)
    elif change == "partial": references[0]["usage_verified_calls"] -= 1
    else: references[0]["latency_ms"] = float("inf")
    result = module.validate_hot_references(items, references, contract)
    assert not result["successful"]
    assert result["failure_reasons"]


def test_replay_wrapper_preserves_actual_workflows_and_uses_frozen_helper_with_global_timeout(monkeypatch, tmp_path):
    module = load_window()
    api = workflow_api()
    items = workflows(fanout=True)
    before = copy.deepcopy([item.to_dict() for item in items])
    seen = []
    async def replay(api_arg, workflows_arg, **kwargs):
        seen.append((api_arg, workflows_arg, kwargs))
        return raw_rows(items, revision=0)
    async def wait_for(awaitable, timeout):
        seen.append(timeout)
        return await awaitable
    monkeypatch.setattr(module.asyncio, "wait_for", wait_for)
    rows = asyncio.run(module.replay_workflows(api, items, raw_path=tmp_path / "raw.jsonl",
        router_url="synthetic://never-contact", model=MODEL, run_namespace=NAMESPACE,
        budget=module.WindowBudget(anchor_unix_s=1000), now_unix_s=3510,
        replay_to_raw=replay))
    assert seen[0] == 10.0
    assert seen[1][0] is api and seen[1][1] is items
    assert seen[1][2]["worker_class"] == "H20-BF16-TP2"
    assert seen[1][2]["run_namespace"] == NAMESPACE
    assert [item.to_dict() for item in items] == before
    assert len(rows) == 6


def asgi_roundtrip(module, path, request, response, *, method="POST", max_body_bytes=1024 * 1024, status_writer=None):
    records = []
    body = json.dumps(request).encode() if request is not None else b""
    incoming = [{"type": "http.request", "body": body[:7], "more_body": True},
                {"type": "http.request", "body": body[7:], "more_body": False}]
    outgoing = [{"type": "http.response.start", "status": 200, "headers": []},
                {"type": "http.response.body", "body": json.dumps(response).encode(), "more_body": False}]
    received = []
    forwarded = []
    async def receive():
        return incoming[len(received)]
    async def send(message):
        forwarded.append(message)
    async def app(scope, receive, send):
        for _ in incoming:
            received.append(await receive())
        for message in outgoing:
            await send(message)
    capture = module.ASGIEvidenceCapture(app, records.append, max_body_bytes=max_body_bytes,
        max_saved_bytes=1024 * 1024, status_writer=status_writer)
    scope = {"type": "http", "path": path, "method": method, "query_string": b""}
    asyncio.run(capture(scope, receive, send))
    assert all(actual is expected for actual, expected in zip(received, incoming))
    assert all(actual is expected for actual, expected in zip(forwarded, outgoing))
    return capture, records


def test_passive_capture_preserves_asgi_messages_and_exact_integer_plan_without_prompt_leak():
    module = load_window()
    request = {"request": {"hexgent": {"workflow_id": "runtime-wf", "call_id": "runtime-call"},
        "request_text": "secret-prompt", "request": {"messages": [{"content": "secret-prompt"}]}}}
    response = {"revision": BIG_REVISION, "entries": [{"workflow_id": "runtime-wf",
        "call_id": "runtime-call", "schedule": {"revision": BIG_REVISION,
            "prefill_worker_url": "http://prefill", "decode_worker_url": "http://decode"}}]}
    enqueue, enqueue_records = asgi_roundtrip(module, "/enqueue_admission", request,
        {"dispatch_ready": False, "schedule": None})
    assert enqueue.observation_complete
    assert enqueue_records[0]["request"]["request"]["hexgent"]["call_id"] == "runtime-call"
    assert "secret-prompt" not in json.dumps(enqueue_records)
    capture, records = asgi_roundtrip(module, "/wait_plan_revision", None, response, method="GET")
    assert capture.observation_complete
    assert records[0]["response"]["entries"][0]["schedule"]["revision"] == BIG_REVISION
    assert isinstance(records[0]["response"]["entries"][0]["schedule"]["revision"], int)


def test_passive_capture_cap_marks_incomplete_while_forwarding_real_messages():
    module = load_window()
    capture, _ = asgi_roundtrip(module, "/event", {"event_type": "request_dispatched",
        "workflow_id": "runtime-wf", "call_id": "runtime-call", "payload": {}}, {}, max_body_bytes=8)
    assert capture.observation_complete is False
    assert capture.incomplete_reasons


def plan_config(tmp_path):
    inputs = tmp_path / "runtime-inputs-v2"
    inputs.mkdir()
    def save(dataset, split, counts, ids):
        records = []
        for wf_id, count in zip(ids, counts):
            calls = [{"call_id": f"{wf_id}-call-{index}",
                "parents": [] if index == 0 else [f"{wf_id}-call-0"],
                "request": {"messages": [{"role": "user", "content": "unchanged"}]},
                "input_len": 10, "output_len": 2,
                "planner_hints": {"model_config": {}, "hardware_params": {}}}
                for index in range(count)]
            records.append({"workflow_id": wf_id, "workflow_type": "synthetic",
                "arrival_offset": 0.0, "calls": calls})
        path = inputs / f"{dataset}.{split}.sanitized.runtime.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
    save("sharegpt", "calibration", [5, 5], ["sg-cal0", "sg-cal1"])
    save("hotpotqa", "calibration", [29, 29], ["hot-cal0", "hot-cal1"])
    save("hotpotqa", "heldout", [33, 33, 33, 34], [f"hotpotqa_{i}" for i in range(10, 14)])
    return {"run_id": RUN, "source_root": str(ROOT), "runtime_input_dir": str(inputs),
        "output_dir": str(tmp_path / "future-run"), "window_anchor_unix_s": 1000.0,
        "window_s": 2700, "cleanup_s": 180, "model": MODEL,
        "implementation_revision": REVISION, "runtime_evidence_sha256": EPOCH,
        "image_digest": "sha256:" + "f" * 64, "gateway_elf_sha256": "e" * 64,
        "model_revision": "945c8663693130f8be2ee66210e062158b2a9693",
        "transfer_parameters_sha256": "9" * 64}


def test_prepared_recipe_freezes_only_mixed68_hot4_references_and_original07_08_repeat(tmp_path):
    module = load_window()
    config = plan_config(tmp_path)
    inputs = Path(config["runtime_input_dir"])
    before = {path.name: path.read_bytes() for path in inputs.iterdir()}
    plan = module.prepare_plan(config)
    assert plan["seed"] == 20261107
    assert plan["calibration"]["methods"] == ["native", "ours"]
    assert plan["calibration"]["workflow_count"] == 4
    assert plan["calibration"]["expected_calls"] == 68
    assert plan["calibration"]["arrival_offsets_s"] == [0.0] * 4
    assert plan["calibration"]["service_fit_input"] is False
    assert plan["references"]["workflow_ids"] == [f"hotpotqa_{i}" for i in range(10, 14)]
    assert plan["references"]["concurrency"] == "one_workflow_at_a_time"
    assert [(run["original_ordinal"], run["method"]) for run in plan["formal_runs"]] == [(7, "native"), (8, "ours")]
    assert all(run["dataset"] == "hotpotqa" and run["expected_calls"] == 133
        and run["arrival_offsets_s"] == [0.0] * 4 for run in plan["formal_runs"])
    assert plan["prior_sg03"] == "unchanged"
    assert plan["future_execution_requires_approval"] is True
    assert plan["input_sha256"] == {name: hashlib.sha256(data).hexdigest() for name, data in before.items()}
    assert {path.name: path.read_bytes() for path in inputs.iterdir()} == before


@pytest.mark.parametrize("flag", [None, "--dry-run"])
def test_default_cli_and_dryrun_prepare_only_never_touch_injected_runtime_operations(tmp_path, capsys, flag):
    module = load_window()
    config = plan_config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    class ForbiddenOperations:
        def __getattr__(self, name):
            raise AssertionError("no startup/cleanup/runtime operation in plan-only: " + name)
    argv = ["--config", str(path)] + ([flag] if flag else [])
    assert module.main(argv, operations=ForbiddenOperations()) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == RUN


def feature(module, name):
    assert hasattr(module, name), "C1 readiness feature is not implemented: " + name
    return getattr(module, name)


@pytest.mark.parametrize("failure", ["stale", "observer_exit", "capture_incomplete", "capture_missing"])
def test_actual_file_watch_alarm_cancels_active_replay_and_preserves_cleanup(failure, tmp_path):
    module = load_window()
    stamp = [1100.0]
    alive = [True]
    latest = tmp_path / "summary.latest.json"
    capture = tmp_path / "capture-status.latest.json"
    latest.write_text(json.dumps(observer()))
    capture.write_text(json.dumps({"observation_complete": True, "incomplete_reasons": []}))
    window = new_window(module, clock=lambda: stamp[0])
    window.register_owned_container(OWNER)
    calls = []
    async def scenario():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        async def replay(*, deadline_unix_s):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        async def change_after_start(delay):
            await started.wait()
            if failure == "stale": stamp[0] = 1103.001
            elif failure == "observer_exit": alive[0] = False
            elif failure == "capture_missing": capture.unlink()
            else: capture.write_text(json.dumps({"observation_complete": False, "incomplete_reasons": ["cap"]}))
        async def alarm():
            return await module.watch_observer(latest, run_id=RUN, owner_id=OWNER,
                deadline_unix_s=3520, capture_status_path=capture, clock=lambda: stamp[0],
                sleep=change_after_start, observer_is_alive=lambda: alive[0])
        with pytest.raises(module.WindowStopped):
            await window.run_stage("calibration-native", replay, observer_summary=observer(),
                observer_watch=alarm, execute=True, approval=approval())
        assert cancelled.is_set()
        async def cleanup(*, owner_id, deadline_unix_s):
            calls.append((owner_id, deadline_unix_s))
        await window.cleanup(OWNER, cleanup, execute=True, approval=approval())
    asyncio.run(scenario())
    assert calls == [(OWNER, 3700.0)]


def test_missing_or_stale_observer_cannot_admit_a_new_stage(tmp_path):
    module = load_window()
    window = new_window(module, clock=lambda: 1103.001)
    window.register_owned_container(OWNER)
    async def forbidden(**kwargs):
        pytest.fail("stale summary must not start any runtime operation")
    with pytest.raises(module.WindowStopped):
        asyncio.run(window.run_stage("calibration-native", forbidden,
            observer_summary=observer(published_unix_s=1103.001), execute=True, approval=approval()))
    with pytest.raises(module.WindowStopped):
        asyncio.run(module.watch_observer(tmp_path / "missing.json", run_id=RUN, owner_id=OWNER,
            deadline_unix_s=3520, clock=lambda: 1100, observer_is_alive=lambda: True))


def epoch_fixture(tmp_path):
    static = {"model": MODEL, "implementation_revision": REVISION,
        "image_digest": "sha256:" + "f" * 64, "gateway_elf_sha256": "e" * 64,
        "model_revision": "945c8663693130f8be2ee66210e062158b2a9693", "model_config_sha256": "8" * 64}
    actual = {"model_alias": MODEL, "implementation_revision": REVISION,
        "image_id": static["image_digest"], "elf_sha256": static["gateway_elf_sha256"],
        "model_revision": static["model_revision"], "model_config_sha256": static["model_config_sha256"],
        "container_id": OWNER, "window_anchor_unix_s": 1000.0, "hard_deadline_unix_s": 3700.0,
        "endpoint_observation_ready": True, "kv_registration_status": "unobserved_until_warmup"}
    path = tmp_path / "actual-runtime-evidence.json"
    path.write_text(json.dumps(actual))
    return static, actual, path


def test_static_plan_keeps_actual_epoch_hash_pending_instead_of_requiring_prestart_evidence(tmp_path):
    module = load_window()
    config = plan_config(tmp_path)
    config.pop("runtime_evidence_sha256")
    try:
        plan = module.prepare_plan(config)
    except KeyError as error:
        pytest.fail("static plan still requires an unknowable pre-start actual epoch: " + str(error))
    assert plan["runtime_evidence_binding"] == "pending_current_runtime"
    assert "runtime_evidence_sha256" not in plan["frozen_metadata"]


def test_epoch_binds_fresh_actual_artifact_once_before_warmup_and_references(tmp_path):
    module = load_window()
    cls = feature(module, "RuntimeEpoch")
    static, actual, path = epoch_fixture(tmp_path)
    epoch = cls(static, run_id=RUN, owner_id=OWNER, budget=module.WindowBudget(anchor_unix_s=1000))
    with pytest.raises(module.WindowStopped):
        epoch.require_bound()
    contract = epoch.bind_file(path)
    assert contract == {"model": MODEL, "implementation_revision": REVISION,
        "runtime_evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert epoch.require_bound() == contract
    assert epoch.bind_file(path) == contract
    # Engine endpoint observation precedes lazy KVArgs registration/warmup.
    assert actual["kv_registration_status"] == "unobserved_until_warmup"
    actual["timestamp_unix_s"] = 1101.0
    changed = tmp_path / "changed-epoch.json"
    changed.write_text(json.dumps(actual))
    with pytest.raises(ValueError, match="bound|changed|epoch"):
        epoch.bind_file(changed)
    assert epoch.require_bound() == contract


@pytest.mark.parametrize("key,value", [
    ("container_id", "f" * 64), ("implementation_revision", "f" * 40),
    ("model_alias", "foreign-model"), ("image_id", "sha256:" + "9" * 64),
    ("elf_sha256", "9" * 64), ("model_revision", "9" * 40),
    ("model_config_sha256", "9" * 64), ("window_anchor_unix_s", 1001.0),
    ("hard_deadline_unix_s", 3701.0), ("endpoint_observation_ready", False),
])
def test_epoch_rejects_foreign_or_unready_actual_source_owner_model_image_anchor(key, value, tmp_path):
    module = load_window()
    cls = feature(module, "RuntimeEpoch")
    static, actual, path = epoch_fixture(tmp_path)
    actual[key] = value
    path.write_text(json.dumps(actual))
    epoch = cls(static, run_id=RUN, owner_id=OWNER, budget=module.WindowBudget(anchor_unix_s=1000))
    with pytest.raises(ValueError):
        epoch.bind_file(path)
    with pytest.raises(module.WindowStopped):
        epoch.require_bound()


def test_unscoped_waiting_evidence_cannot_unlock_one_joined_calibration_call():
    module = load_window()
    items = workflows(ids=("cal::one",))
    result = cal_gate(module, items, raw_rows(items), capture_records(items),
        worker_snapshots=[{"snapshots": [{"queues": {"waiting": [{"request_id": "ambient"}]}}]}])
    assert result["canonical_joined_calls"] == 1
    assert not result["pressure_observed"]
    assert not result["successful"]


def scoped_waiting_evidence():
    return {"worker_url": "http://prefill", "run_namespace": NAMESPACE, "captured_unix_s": 1.006,
        "payload": {"snapshots": [{"instance_id": "prefill", "mode": "prefill", "timestamp": 1.005,
            "queues": {"waiting": [{"workflow_id": NAMESPACE + "::cal::one", "call_id": NAMESPACE + "::root"}]}}]}}


@pytest.mark.parametrize("change", [None, "old_namespace", "old_timestamp", "warmup_call", "foreign_worker"])
def test_waiting_pressure_requires_current_calibration_identity_owned_worker_and_phase(change):
    module = load_window()
    items = workflows(ids=("cal::one",))
    evidence = scoped_waiting_evidence()
    if change == "old_namespace": evidence["run_namespace"] = "old-window"
    elif change == "old_timestamp": evidence["payload"]["snapshots"][0]["timestamp"] = 0.5
    elif change == "warmup_call": evidence["payload"]["snapshots"][0]["queues"]["waiting"][0]["call_id"] = RUN + "::warmup::root"
    elif change == "foreign_worker": evidence["worker_url"] = "http://foreign-worker"
    result = cal_gate(module, items, raw_rows(items), capture_records(items), worker_snapshots=[evidence],
        owned_worker_urls={"http://prefill", "http://decode"}, phase_bounds_unix_s=(1.0, 1.01))
    assert result["canonical_joined_calls"] == 1
    assert result["pressure_observed"] is (change is None)
    assert result["successful"] is (change is None)


def test_capture_status_publication_failure_removes_old_healthy_file_and_keeps_asgi_bytes(monkeypatch, tmp_path):
    module = load_window()
    cls = feature(module, "CaptureStatusPublisher")
    path = tmp_path / "capture-status.latest.json"
    path.write_text(json.dumps({"observation_complete": True, "incomplete_reasons": []}))
    publisher = cls(path)
    def fail(*args, **kwargs):
        raise OSError("synthetic atomic publication failure")
    monkeypatch.setattr(publisher, "_replace", fail)
    capture, _ = asgi_roundtrip(module, "/event", {"event_type": "request_dispatched",
        "workflow_id": "runtime-wf", "call_id": "runtime-call", "payload": {}}, {},
        status_writer=publisher.publish)
    assert capture.observation_complete is False
    assert not path.exists()


def registration_fixture():
    source_paths = (
        "python/sglang/srt/environ.py", "python/sglang/srt/disaggregation/common/conn.py",
        "python/sglang/srt/disaggregation/mooncake/conn.py",
        "python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py")
    roots = [("prefill", 10), ("decode", 20), ("baseline_gateway", 30)]
    status = {"phase": "ready", "started_unix_s": 1001.0, "timestamp_unix_s": 1100.0,
        "processes": [{"name": name, "pid": pid, "pgid": pid, "returncode": None} for name, pid in roots]}
    facts = [{"pid": pid, "ppid": 1, "pgrp": pid, "start_identity": str(pid + 100),
        "start_identity_after": str(pid + 100), "pid_namespace": "pid:[42]", "net_namespace": "net:[43]",
        "task_ids": [pid], "fd_socket_inodes": []} for _, pid in roots]
    sockets, records = [], []
    for stream, root in roots[:2]:
        for pid in (root + 1, root + 2):
            identity = str(pid + 100)
            tid = pid + 1000  # A C++ numeric ID is deliberately a TID, not this PID.
            facts.append({"pid": pid, "ppid": root, "pgrp": root, "start_identity": identity,
                "start_identity_after": identity, "pid_namespace": "pid:[42]", "net_namespace": "net:[43]",
                "task_ids": [pid, tid], "fd_socket_inodes": [str(10000 + pid), str(20000 + pid)]})
            for kind, port, inode in (("session", 40000 + pid, str(10000 + pid)),
                                      ("tcp", 45000 + pid, str(20000 + pid))):
                line = (f"I1107 17:00:00.000001 {tid} transfer_engine_impl.cpp:182] Transfer Engine RPC using P2P handshake, listening on 192.0.2.1:{port}"
                    if kind == "session" else
                    f"I1107 17:00:00.000002 {tid} tcp_transport.cpp:553] TcpTransport: listen on port {port}")
                records.append({"stream": stream, "offset": len(records) * 200, "line": line})
                sockets.append({"pid": pid, "start_identity": identity, "socket_inode": inode,
                    "local_ip": "192.0.2.1", "local_port": port, "remote_ip": "0.0.0.0",
                    "remote_port": 0, "state": "LISTEN"})
    return {"run_id": RUN, "owner_id": OWNER, "budget": None, "launch_status": status,
        "process_facts": facts, "socket_facts": sockets, "endpoint_records": records,
        "pid_namespace": "pid:[42]", "net_namespace": "net:[43]", "cgroup_path": "/synthetic/cgroup",
        "source_hashes": {name: "7" * 64 for name in source_paths},
        "log_paths": ["/synthetic/run/prefill.log", "/synthetic/run/decode.log"]}


def test_startup_registration_joins_actual_log_roles_tid_descendants_and_listen_inodes():
    module = load_window()
    builder = feature(module, "build_observer_registration")
    fixture = registration_fixture()
    fixture["budget"] = module.WindowBudget(anchor_unix_s=1000)
    registration = builder(**fixture)
    assert registration["owner_id"] == OWNER
    assert registration["deadline_unix_s"] == 3700.0
    assert registration["pid_namespace"] == "pid:[42]"
    assert registration["net_namespace"] == "net:[43]"
    assert len(registration["peers"]) == 8
    assert {item["pid"] for item in registration["processes"]} == {10, 11, 12, 20, 21, 22, 30}
    assert registration["endpoint_observation_ready"] is True
    assert registration["kv_registration_status"] == "unobserved_until_warmup"
    assert all(item["port"] != 16537 for item in registration["peers"])


@pytest.mark.parametrize("change", ["empty", "missing_rank_endpoint", "unowned_inode", "tid_not_pid",
    "broken_parent_chain", "pid_reused", "foreign_netns", "missing_source"])
def test_registration_never_guesses_peer_or_ownership_from_cpp_number_or_old_port(change):
    module = load_window()
    builder = feature(module, "build_observer_registration")
    fixture = registration_fixture()
    fixture["budget"] = module.WindowBudget(anchor_unix_s=1000)
    child = next(item for item in fixture["process_facts"] if item["pid"] == 11)
    if change == "empty": fixture["endpoint_records"] = []
    elif change == "missing_rank_endpoint": fixture["endpoint_records"].pop()
    elif change == "unowned_inode": child["fd_socket_inodes"] = []
    elif change == "tid_not_pid": child["task_ids"] = [11]
    elif change == "broken_parent_chain": child["ppid"] = 999
    elif change == "pid_reused": child["start_identity_after"] = "reused"
    elif change == "foreign_netns": child["net_namespace"] = "net:[foreign]"
    else: fixture["source_hashes"].pop("python/sglang/srt/environ.py")
    with pytest.raises(ValueError):
        builder(**fixture)


def runtime_config(tmp_path):
    config = plan_config(tmp_path)
    config.pop("runtime_evidence_sha256")
    config["model_config_sha256"] = "8" * 64
    config["runtime"] = {"container_name": RUN, "source_host_path": str(ROOT),
        "model_host_path": "/synthetic/models/Llama", "artifacts_host_path": "/synthetic/artifacts",
        "inputs_host_path": config["runtime_input_dir"], "source_dir": "/src",
        "run_dir": "/artifacts/c1", "model_path": "/model-cache/snapshots/" + config["model_revision"],
        "gateway_binary": "/artifacts/r2-build/bin/sgl-model-gateway", "uid_gid": "1002:1002",
        "gpu_uuids": [f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(4)],
        "cpuset_cpus": "5,7,9,11,13,15", "config_container_path": "/artifacts/c1-config.json",
        "approval_container_path": "/artifacts/c1-approval.json", "owner_container_path": "/artifacts/c1-owner.json",
        "registration_container_path": "/artifacts/c1-registration.json",
        "ports": {"prefill": 37100, "decode": 37110, "bootstrap": 37120, "native": 37130,
            "native_metrics": 37131, "sidecar": 37140, "ours": 37150, "ours_metrics": 37151}}
    return config


def test_concrete_c1_recipe_reuses_unmodified_reference_launcher_and_original_absolute_deadline(tmp_path):
    module = load_window()
    render = feature(module, "render_c1_runtime_recipe")
    config = runtime_config(tmp_path)
    reference = ROOT / "scripts/hexgent_real_trace_collectors/start_h20_tp2_reference.sh"
    before = reference.read_bytes()
    recipe = render(config, now_unix_s=1200.0)
    assert recipe["reference_argv"] == ["bash", "/src/scripts/hexgent_real_trace_collectors/start_h20_tp2_reference.sh"]
    env = recipe["reference_environment"]
    assert float(env["RUN_TIMEOUT_S"]) == 2320.0
    assert float(env["READY_TIMEOUT_S"]) == 600.0
    assert env["MODEL_PATH"] == config["runtime"]["model_path"]
    assert recipe["hard_deadline_unix_s"] == 3700.0
    assert recipe["work_deadline_unix_s"] == 3520.0
    assert recipe["docker_argv"][:2] == ["docker", "run"]
    assert "--pull=never" in recipe["docker_argv"]
    assert config["image_digest"] in recipe["docker_argv"]
    assert all(uuid in recipe["docker_argv"][recipe["docker_argv"].index("--gpus") + 1]
        for uuid in config["runtime"]["gpu_uuids"])
    assert recipe["worker_log_budget_mib"] == 32
    assert recipe["native_gateway_log_budget_mib"] == 16
    assert recipe["ours_stdout_caps_mib"] == [4, 4, 4, 4]
    assert recipe["raw_reference_capture_telemetry_qc_budget_mib"] == 32
    assert recipe["observer_budget_mib"] == 16 and recipe["persistent_total_budget_mib"] == 128
    assert recipe["endpoint_observation_ready_required"] is True
    assert recipe["kv_registration_required_before_warmup"] is False
    assert reference.read_bytes() == before
    assert recipe["source_sha256"][str(reference.relative_to(ROOT))] == hashlib.sha256(before).hexdigest()


def test_concrete_owned_launcher_is_deferred_and_cleanup_uses_full_returned_id(tmp_path):
    module = load_window()
    cls = feature(module, "C1OwnedLauncher")
    render = feature(module, "render_c1_runtime_recipe")
    config = runtime_config(tmp_path)
    plan = module.prepare_plan(config)
    recipe = render(config, now_unix_s=1200.0)
    calls = []
    class FixtureExecutor:
        async def run(self, argv, *, deadline_unix_s):
            calls.append((list(argv), deadline_unix_s))
            return {"returncode": 0, "stdout": OWNER + "\n", "stderr": ""}
    launcher = cls(plan, recipe, executor=FixtureExecutor(), clock=lambda: 1200.0)
    assert asyncio.run(launcher.start())["executed"] is False
    assert calls == []
    approve = {"approved_plan_sha256": feature(module, "plan_sha256")(plan), "window_run_id": RUN}
    record = asyncio.run(launcher.start(execute=True, approval=approve))
    assert record["owner_id"] == OWNER
    with pytest.raises(ValueError):
        asyncio.run(launcher.cleanup(OWNER[:12], execute=True, approval=approve))
    asyncio.run(launcher.cleanup(OWNER, execute=True, approval=approve))
    assert calls == [(recipe["docker_argv"], 3520.0), (["docker", "stop", "--time", "20", OWNER], 3700.0)]


def test_first_replay_error_cancels_only_owned_source_dag_tasks_before_later_calls(monkeypatch, tmp_path):
    module = load_window()
    api = workflow_api()
    items = workflows(fanout=True)
    tracked = set()
    original = api.WorkflowTraceReplayer._run_workflow
    async def track_workflow(self, *args, **kwargs):
        tracked.add(asyncio.current_task())
        return await original(self, *args, **kwargs)
    monkeypatch.setattr(api.WorkflowTraceReplayer, "_run_workflow", track_workflow)
    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return None
    import types
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(ClientSession=FakeSession))
    async def scenario():
        second_started = asyncio.Event()
        second_cancelled = asyncio.Event()
        release = asyncio.Event()
        later_calls = []
        async def post(session, url, payload, timeout_s):
            tracked.add(asyncio.current_task())
            wf, cid = payload["conversation_id"], payload["hexgent_call_id"]
            if wf == NAMESPACE + "::cal::one":
                await second_started.wait()
                raise RuntimeError("synthetic first HTTP/transport failure")
            if cid == NAMESPACE + "::root":
                second_started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    second_cancelled.set()
                    raise
            else:
                later_calls.append(cid)
            return {"model": MODEL, "usage": {"prompt_tokens": 10, "completion_tokens": 2}}, {}
        monkeypatch.setattr(api, "_post_json", post)
        try:
            with pytest.raises(module.WindowStopped):
                await asyncio.wait_for(module.C1RawBudget().replay(api, items, raw_path=tmp_path / "raw.jsonl",
                    router_url="synthetic://never-contact", model=MODEL, worker_class="H20-BF16-TP2",
                    run_namespace=NAMESPACE), timeout=2)
            assert second_cancelled.is_set(), "first failure must cancel the sibling in-flight actual source task"
            release.set()
            for _ in range(3):
                await asyncio.sleep(0)
            assert later_calls == []
        finally:
            for task in tracked:
                if not task.done(): task.cancel()
            await asyncio.gather(*tracked, return_exceptions=True)
        rows = [json.loads(line) for line in (tmp_path / "raw.jsonl").read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["error"]
    asyncio.run(scenario())


def test_long_quiet_passive_capture_is_valid_with_live_publisher_and_fresh_sampler(tmp_path):
    module = load_window()
    latest = tmp_path / "summary.latest.json"
    status = tmp_path / "capture-status.latest.json"
    now = [1100.0]
    polls = []
    latest.write_text(json.dumps(observer()))
    status.write_text(json.dumps({"observation_complete": True, "incomplete_reasons": [],
        "last_event_unix_s": 1000.0}))
    async def publish_fresh_sampler(delay):
        polls.append(delay)
        now[0] += 10.0
        value = observer(sampled_unix_s=now[0])
        if len(polls) == 2:
            value.update(first_error_unix_s=now[0], stop_requested=True)
        latest.write_text(json.dumps(value))
    result = asyncio.run(module.watch_observer(latest, run_id=RUN, owner_id=OWNER,
        deadline_unix_s=3520, capture_status_path=status, observer_is_alive=lambda: True,
        clock=lambda: now[0], sleep=publish_fresh_sampler))
    assert len(polls) == 2
    assert result["first_error_unix_s"] == 1120.0


def test_controlled_ours_cleanup_is_not_reported_as_unexpected_observer_exit(monkeypatch):
    module = load_window()
    import os
    import types
    children = module.RuntimeChildren()
    state = {"code": None}
    class FakeProcess:
        pid = 12345
        def poll(self): return state["code"]
        def wait(self, timeout=None): return state["code"]
    entry = {"name": "ours-sidecar", "process": FakeProcess(), "pgid": 12345,
        "truncated": False, "write_error": None, "thread": types.SimpleNamespace(join=lambda timeout: None, is_alive=lambda: False)}
    children.entries.append(entry)
    health_during_owned_stop = []
    monkeypatch.setattr(os, "getpgid", lambda pid: 12345)
    def signal_owned(pgid, signal):
        assert pgid == 12345
        state["code"] = 0
        health_during_owned_stop.append(children.healthy())
    monkeypatch.setattr(os, "killpg", signal_owned)
    asyncio.run(children.stop(module.time.time() + 30, [entry]))
    assert health_during_owned_stop == [True]
    assert children.entries == []


def configured_listener_fixture():
    fixture = registration_fixture()
    facts = fixture["process_facts"]
    sockets = list(fixture["socket_facts"])
    for pid in (40, 50):
        facts.append({"pid": pid, "ppid": 1, "pgrp": pid, "start_identity": str(pid + 100),
            "start_identity_after": str(pid + 100), "pid_namespace": "pid:[42]", "net_namespace": "net:[43]",
            "task_ids": [pid], "fd_socket_inodes": []})
    roles = {}
    for role, pid, port in (("prefill_http", 10, 37100), ("prefill_bootstrap", 10, 37120),
            ("decode_http", 20, 37110), ("native_http", 30, 37130),
            ("sidecar_http", 40, 37140), ("ours_http", 50, 37150)):
        process = next(item for item in facts if item["pid"] == pid)
        inode = str(50000 + port)
        process["fd_socket_inodes"].append(inode)
        roles[role] = {"port": port, "local_ip": "127.0.0.1", "root_pid": pid, "root_start_identity": process["start_identity"]}
        sockets.append({"pid": pid, "start_identity": process["start_identity"], "socket_inode": inode,
            "local_ip": "127.0.0.1", "local_port": port, "remote_ip": "0.0.0.0", "remote_port": 0, "state": "LISTEN"})
    return roles, facts, sockets


def test_all_configured_frontends_require_owned_listen_fd_start_and_descendant_proof():
    module = load_window()
    prove = feature(module, "prove_owned_listeners")
    roles, facts, sockets = configured_listener_fixture()
    result = prove(roles, facts, sockets, pid_namespace="pid:[42]", net_namespace="net:[43]")
    assert set(result) == set(roles)
    assert all(result[role]["port"] == declared["port"] for role, declared in roles.items())


@pytest.mark.parametrize("change", ["engine_only", "foreign_same_port", "wrong_start", "inode_not_owned", "not_listen", "wrong_local_ip", "foreign_namespace", "foreign_parent"])
def test_engine_endpoints_or_same_alias_foreign_http_are_not_frontend_ownership(change):
    module = load_window()
    prove = feature(module, "prove_owned_listeners")
    roles, facts, sockets = configured_listener_fixture()
    row = next(item for item in sockets if item["local_port"] == 37100)
    if change == "engine_only": sockets = [item for item in sockets if item["local_port"] >= 40000]
    elif change == "foreign_same_port": row["pid"] = 999
    elif change == "wrong_start": row["start_identity"] = "reused"
    elif change == "inode_not_owned": row["socket_inode"] = "foreign-inode"
    elif change == "not_listen": row["state"] = "ESTABLISHED"
    elif change == "wrong_local_ip": row["local_ip"] = "192.0.2.99"
    elif change == "foreign_namespace": next(item for item in facts if item["pid"] == 10)["net_namespace"] = "net:[foreign]"
    else:
        row["pid"] = 11
        child = next(item for item in facts if item["pid"] == 11)
        child["ppid"] = 999; child["fd_socket_inodes"].append(row["socket_inode"])
        row["start_identity"] = child["start_identity"]
    with pytest.raises(ValueError):
        prove(roles, facts, sockets, pid_namespace="pid:[42]", net_namespace="net:[43]")


def test_cleanup_catches_actual_subprocess_timeout_and_still_finishes_drain(monkeypatch):
    module = load_window()
    import os
    import subprocess
    import types
    class PendingProcess:
        pid = 12345
        def poll(self): return None
        def wait(self, timeout=None): raise subprocess.TimeoutExpired("synthetic-owned", timeout)
    joined = []
    entry = {"name": "ours-sidecar", "process": PendingProcess(), "pgid": 12345,
        "truncated": False, "write_error": None,
        "thread": types.SimpleNamespace(join=lambda timeout: joined.append(True), is_alive=lambda: False)}
    clock = [1000.0]
    def advance(): clock[0] += 10; return clock[0]
    monkeypatch.setattr(module.time, "time", advance)
    monkeypatch.setattr(os, "getpgid", lambda pid: 12345)
    monkeypatch.setattr(os, "killpg", lambda *args: None)
    children = module.RuntimeChildren(); children.entries.append(entry)
    try:
        asyncio.run(children.stop(1300.0, [entry]))
    except subprocess.TimeoutExpired:
        pytest.fail("actual subprocess.TimeoutExpired bypasses the current built-in TimeoutError handler")
    except module.WindowStopped:
        pass  # A typed incomplete-cleanup result is acceptable after the drain is handled.
    assert joined == [True]


def test_observer_teardown_finally_runs_when_other_owned_cleanup_fails(monkeypatch, tmp_path):
    module = load_window()
    import types
    ops = object.__new__(module.C1RuntimeOperations)
    ops.runtime = {}  # Real constructor supplies the valid legacy runtime mapping.
    ops.plan = {"run_id": RUN}; ops.root = tmp_path; ops.phase_records = []
    ops.budget = module.WindowBudget(anchor_unix_s=module.time.time())
    nonobserver = {"name": "reference-supervisor"}; observer_entry = {"name": "transport-observer"}
    calls = []
    class Children:
        entries = [nonobserver, observer_entry]
        async def stop(self, deadline, entries=None):
            selected = self.entries if entries is None else entries
            calls.append([entry["name"] for entry in selected])
            if nonobserver in selected:
                raise RuntimeError("synthetic owned cleanup failure")
    ops.children = Children()
    async def fail_startup(): raise module.WindowStopped("synthetic startup observation failure")
    ops.startup = fail_startup
    monkeypatch.setattr(module, "_bounded_qc_json", lambda path, value: None)
    with pytest.raises(RuntimeError):
        asyncio.run(ops.execute())
    assert any("transport-observer" in names for names in calls[1:]), "observer must be attempted in a finally after other cleanup fails"


def test_created_container_cidfile_is_registered_when_executor_raises_then_exact_cleanup(monkeypatch, tmp_path):
    module = load_window()
    config = runtime_config(tmp_path)
    plan = module.prepare_plan(config)
    recipe = module.render_c1_runtime_recipe(config, now_unix_s=1200)
    cid = tmp_path / "fresh.cid"
    recipe["docker_argv"][recipe["docker_argv"].index("--cidfile") + 1] = str(cid)
    calls = []
    class Executor:
        async def run(self, argv, *, deadline_unix_s):
            calls.append(list(argv))
            if argv[:2] == ["docker", "run"]:
                cid.write_text(OWNER + "\n")
                raise TimeoutError("synthetic transport timeout after container creation")
            return {"returncode": 0, "stdout": OWNER, "stderr": ""}
    launcher = module.C1OwnedLauncher(plan, recipe, executor=Executor(), clock=lambda: 1200)
    approved = {"window_run_id": RUN, "approved_plan_sha256": module.plan_sha256(plan)}
    async def scenario():
        try:
            with pytest.raises(TimeoutError):
                await launcher.start(execute=True, approval=approved)
            assert launcher.window.owner_id == OWNER
        finally:
            if launcher.window.owner_id is not None:
                await launcher.cleanup(launcher.window.owner_id, execute=True, approval=approved)
    asyncio.run(scenario())
    assert calls[-1] == ["docker", "stop", "--time", "20", OWNER]


def test_child_process_manifest_exports_exact_whitelist_not_arbitrary_prefixed_secrets(monkeypatch, tmp_path):
    module = load_window()
    import os
    import subprocess
    import types
    class Thread:
        def __init__(self, target, **kwargs): self.target = target
        def start(self): self.target()
    process = types.SimpleNamespace(pid=12345, stdout=types.SimpleNamespace(fileno=lambda: 123), poll=lambda: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(os, "read", lambda *args: b"")
    monkeypatch.setattr(module.threading, "Thread", Thread)
    records = []
    monkeypatch.setattr(module, "_bounded_qc_json", lambda path, value: records.append(value))
    module.RuntimeChildren().start("owned", ["synthetic-no-exec"],
        {"HEXGENT_SECRET": "never-export-this", "AGENTIC_PD_PRIVATE_TOKEN": "never-export-this",
         "HEXGENT_PROJECTED_RATIO_ORDER": "high_first", "HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH": "1",
         "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}, tmp_path / "child.log", cap_bytes=1024)
    exported = records[0]["configuration"]
    assert "HEXGENT_SECRET" not in exported and "AGENTIC_PD_PRIVATE_TOKEN" not in exported
    assert exported["HEXGENT_PROJECTED_RATIO_ORDER"] == "high_first"
    assert exported["HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH"] == "1"
    assert exported["OMP_NUM_THREADS"] == "1"


@pytest.mark.parametrize("missing_http", [False, True])
def test_actual_startup_reader_decodes_owned_tasks_fd_tcp_and_configured_frontends(monkeypatch, tmp_path, missing_http, ipv6_data=False, observed=None):
    module = load_window()
    import ipaddress
    import types
    fixture = registration_fixture()
    roles, all_facts, sockets = configured_listener_fixture()
    facts = {row["pid"]: row for row in all_facts}
    status = fixture["launch_status"]
    source_hashes = {name: module.sha256(ROOT / name) for name in module.CONFIG_SOURCE_PATHS}
    files = {"/proc/sys/net/ipv4/ip_local_port_range": "32768 60999\n",
        "/proc/sys/net/ipv4/ip_local_reserved_ports": "\n",
        "/sys/fs/cgroup/memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
        "/sys/fs/cgroup/memory.peak": "1048576\n"}
    for pid, row in facts.items():
        fields = ["S"] + ["0"] * 19
        fields[1], fields[2], fields[19] = str(row["ppid"]), str(row["pgrp"]), row["start_identity"]
        files[f"/proc/{pid}/stat"] = str(pid) + " (synthetic owned worker) " + " ".join(fields)
        files[f"/proc/{pid}/status"] = f"Name:\tsynthetic-worker\nTgid:\t{pid}\n"
        files[f"/proc/{pid}/cmdline"] = "python\0synthetic-worker.py\0"
        files[f"/proc/{pid}/task/{pid}/children"] = " ".join(str(item["pid"]) for item in all_facts if item["ppid"] == pid)
        files[f"/proc/{pid}/limits"] = "Max open files 1024 4096 files\n"
        files[f"/proc/{pid}/environ"] = "SGLANG_MOONCAKE_TRANSFER_PROTOCOL=tcp\0"
        for tid in row["task_ids"]:
            task_fields=["S"]+["0"]*19;task_fields[19]=str(tid+500)
            files[f"/proc/{pid}/task/{tid}/stat"]=str(tid)+" (owned task) "+" ".join(task_fields)
    def endpoint(ip, port):
        return ipaddress.IPv4Address(ip).packed[::-1].hex().upper() + f":{port:04X}"
    lines = ["sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"]
    lines6=[lines[0]]
    for index, row in enumerate(sockets):
        if missing_http and row["local_port"] == 37100:
            continue
        if ipv6_data and row["local_port"]>=45000:
            lines6.append(f"{index}: {'0'*32}:{row['local_port']:04X} {'0'*32}:0000 0A 00000000:00000000 00:00000000 00000000 1002 0 {row['socket_inode']} 1\n")
        else:
            lines.append(f"{index}: {endpoint(row['local_ip'], row['local_port'])} 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1002 0 {row['socket_inode']} 1\n")
    files["/proc/net/tcp"] = "".join(lines)
    files["/proc/net/tcp6"] = "".join(lines6)
    for stream in ("prefill", "decode"):
        files[str(tmp_path / "workers" / (stream + ".log"))] = "\n".join(row["line"] for row in fixture["endpoint_records"] if row["stream"] == stream)
    calls = []
    class Reader:
        def read_text(self, path):
            calls.append(("read", str(path)))
            assert str(path) in files, "unexpected fixture read: " + str(path)
            return files[str(path)]
        def readlink(self, path):
            calls.append(("link", str(path)))
            if str(path).endswith("/ns/pid"): return "pid:[42]"
            if str(path).endswith("/ns/net"): return "net:[43]"
            raise AssertionError("unexpected fixture link: " + str(path))
        def list_fd_targets(self, pid, proc_root="/proc"):
            calls.append(("fd", pid))
            return ["socket:[" + inode + "]" for inode in facts[pid]["fd_socket_inodes"]]
        def file_metadata(self,path):
            return {"path":path,"sha256":"31124eb6385718f551751a580c7e17203bc307efa207ab5315b66b4ad5793e8c","is_regular":True}
    original_iterdir = Path.iterdir
    def task_ids(path):
        if str(path).startswith("/proc/"):
            pid = int(path.parts[2])
            assert path.name == "task" and pid in facts
            return iter(Path(str(path) + "/" + str(tid)) for tid in facts[pid]["task_ids"])
        return original_iterdir(path)
    monkeypatch.setattr(Path, "iterdir", task_ids)
    now = module.time.time()
    plan = {"run_id": RUN, "window_anchor_unix_s": now - 1, "hard_deadline_unix_s": now + 2699,
        "cleanup_reserve_s": 180, "source_file_sha256": source_hashes}
    config = {"runtime": {"run_dir": str(tmp_path), "source_dir": str(ROOT),
        "ports": {"prefill": 37100, "decode": 37110, "bootstrap": 37120, "native": 37130}}}
    queries=[]
    if ipv6_data:
        api=module._observer_api()
        def query(argv,*,timeout_s,max_capture_bytes):
            queries.append((list(argv),timeout_s,max_capture_bytes))
            if argv[-1]=="-V":raw=b"ss utility, iproute2-6.1.0\n"
            else:
                port=int(argv[-1].split(":")[1].split()[0])
                row=next(row for row in sockets if row["local_port"]==port)
                raw=f"LISTEN 0 4096 *:{port} *:* ino:{row['socket_inode']} sk:abcd v6only:0\n".encode()
            return {"stdout":raw,"stderr":b"","returncode":0,"timed_out":False,"truncated":False,
                "drain_finished":True,"capture_complete":True,"capture_errors":[]}
        monkeypatch.setattr(api,"_ss_query",query)
        monkeypatch.setattr(module,"_observer_api",lambda:api)
        config["runtime"]["ss_tool"]={"path":"/usr/bin/ss","sha256":"31124eb6385718f551751a580c7e17203bc307efa207ab5315b66b4ad5793e8c","version":"ss utility, iproute2-6.1.0"}
    if missing_http:
        with pytest.raises((ValueError, module.WindowStopped)):
            module.collect_startup_registration(config, plan, OWNER, status, supervisor_pid=1, reader=Reader())
    else:
        result = module.collect_startup_registration(config, plan, OWNER, status, supervisor_pid=1, reader=Reader())
        assert result.get("configured_listeners"), "engine-only registration must also bind configured HTTP/bootstrap listeners"
        assert {"prefill_http", "prefill_bootstrap", "decode_http", "native_http"} <= set(result["configured_listeners"])
        if observed is not None:observed.update(result=result,queries=queries,plan=plan,module=module,reader_class=Reader,files=files)
    assert any(kind == "fd" for kind, value in calls)
    assert ("read", "/proc/net/tcp") in calls


@pytest.mark.parametrize("scope", ["phase", "references"])
def test_actual_thread_idle_does_not_issue_a_later_snapshot_poll_after_cancellation(monkeypatch, tmp_path, scope):
    module = load_window()
    runner = module._pilot("run_paired_pilot")
    import threading
    calls = []
    release = threading.Event()
    used_thread = [False]
    original_wait = runner.wait_idle
    async def scenario():
        loop = asyncio.get_running_loop()
        at_pause, thread_done = asyncio.Event(), asyncio.Event()
        payload = {"snapshots": [{"limits": {}, "running": [], "queues": {"waiting": []}}]}
        def get(url, deadline):
            calls.append(url)
            return payload
        def pause(delay):
            loop.call_soon_threadsafe(at_pause.set)
            release.wait(timeout=1)
        def actual_wait(*args, **kwargs):
            used_thread[0] = True
            try: return original_wait(*args, **kwargs)
            finally: loop.call_soon_threadsafe(thread_done.set)
        async def async_get(url, deadline):
            calls.append(url)
            if len(calls) == 2: at_pause.set()
            return payload
        monkeypatch.setattr(runner, "get_json", get)
        monkeypatch.setattr(runner.time, "sleep", pause)
        monkeypatch.setattr(runner, "wait_idle", actual_wait)
        monkeypatch.setattr(module, "async_get_json", async_get, raising=False)
        budget = module.WindowBudget(anchor_unix_s=module.time.time())
        if scope == "phase":
            operation = module._idle(tmp_path / "idle.json", {"prefill": {}, "decode": {}}, budget)
        else:
            items = workflows(ids=module.HOT_IDS, fanout=True)
            async def forbidden_replay(*args, **kwargs):
                pytest.fail("cancelled pre-reference idle must not issue a model request")
            operation = module.collect_hot_references(workflow_api(), items, output_dir=tmp_path / "refs",
                router_url="synthetic://never-contact", model=MODEL, implementation_revision=REVISION,
                runtime_evidence_sha256=EPOCH, run_namespace=RUN + "::refs", expected_limits={"prefill": {}, "decode": {}},
                budget=budget, replay_to_raw=forbidden_replay)
        task = asyncio.create_task(operation)
        try:
            await asyncio.wait_for(at_pause.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        finally:
            release.set()
            if used_thread[0]: await asyncio.wait_for(thread_done.wait(), timeout=1)
        assert len(calls) == 2, "cancelled legacy to_thread poll issued new P/D HTTP requests"
    asyncio.run(scenario())


def test_fixed_async_idle_preserves_configured_endpoints_two_empty_polls_and_spacing(tmp_path):
    module = load_window()
    idle = feature(module, "wait_idle_async")
    calls, sleeps = [], []
    async def get(url, deadline):
        calls.append(url)
        return {"snapshots": [{"limits": {"max_running_requests": 16}, "running": [], "queues": {"waiting": []}}]}
    async def sleep(delay): sleeps.append(delay)
    urls = {"prefill": "http://synthetic-p:38100", "decode": "http://synthetic-d:38110"}
    result = asyncio.run(idle(tmp_path / "idle.json", {mode: {"max_running_requests": 16} for mode in urls},
        module.time.time() + 30, worker_urls=urls, getter=get, sleep=sleep))
    assert result["success"] and result["poll_count"] == 2
    assert calls == [urls["prefill"] + "/hexgent/pd_snapshot", urls["decode"] + "/hexgent/pd_snapshot"] * 2
    assert sleeps == [0.25]


def fake_ours_start(module, monkeypatch, tmp_path):
    import types
    config = runtime_config(tmp_path)
    transfer = tmp_path / "transfer.json"; transfer.write_text("{}")
    config["runtime"]["transfer_parameters_path"] = str(transfer)
    ops = object.__new__(module.C1RuntimeOperations)
    ops.runtime = config["runtime"]
    ops.plan = {"run_id": RUN, "frozen_metadata": {"transfer_parameters_sha256": module.sha256(transfer)}}
    ops.budget = module.WindowBudget(anchor_unix_s=module.time.time())
    ops.capture_status = None; ops.ours_entries = []
    class Children:
        def healthy(self): return True
        def start(self, name, argv, env, path, **kwargs):
            if name == "ours-sidecar":
                target = Path(argv[argv.index("--capture-status-path") + 1])
                target.write_text(json.dumps({"observation_complete": True, "incomplete_reasons": []}))
            return {"process": types.SimpleNamespace(poll=lambda: None, pid=123), "name": name}
    ops.children = Children()
    gateway = ["synthetic-gateway", "--port", "37150", "--prometheus-port", "37151",
        "--prefill", "http://p", "37120", "--decode", "http://d", "--hexgent-planner-url", "http://s"]
    runner = module._pilot("run_paired_pilot")
    monkeypatch.setattr(runner, "method_commands", lambda *args: ({}, [], list(gateway), tmp_path / "telemetry.jsonl"))
    monkeypatch.setattr(ops, "verify_ours_frontends", lambda: {"synthetic_verified": True}, raising=False)
    return ops, runner


def test_actual_readiness_thread_does_not_repeat_models_get_after_cancellation(monkeypatch, tmp_path):
    module = load_window()
    ops, runner = fake_ours_start(module, monkeypatch, tmp_path)
    import threading
    release = threading.Event(); calls = []; used_thread = [False]
    original_wait = runner.wait_ready
    async def scenario():
        loop = asyncio.get_running_loop()
        at_pause, thread_done = asyncio.Event(), asyncio.Event()
        def get(url, deadline):
            calls.append(url)
            if len(calls) == 1: raise ConnectionRefusedError("synthetic not-ready")
            return {"data": [{"id": MODEL}]}
        def pause(delay):
            loop.call_soon_threadsafe(at_pause.set)
            release.wait(timeout=1)
        def actual_wait(*args, **kwargs):
            used_thread[0] = True
            try: return original_wait(*args, **kwargs)
            finally: loop.call_soon_threadsafe(thread_done.set)
        async def async_get(url, deadline):
            calls.append(url); at_pause.set()
            raise ConnectionRefusedError("synthetic not-ready")
        monkeypatch.setattr(runner, "get_json", get)
        monkeypatch.setattr(runner.time, "sleep", pause)
        monkeypatch.setattr(runner, "wait_ready", actual_wait)
        monkeypatch.setattr(module, "async_get_json", async_get, raising=False)
        task = asyncio.create_task(ops.start_ours(tmp_path))
        try:
            await asyncio.wait_for(at_pause.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        finally:
            release.set()
            if used_thread[0]: await asyncio.wait_for(thread_done.wait(), timeout=1)
        assert len(calls) == 1, "cancelled source wait_ready thread performed another models GET"
    asyncio.run(scenario())


def test_actual_ours_start_requires_frontend_ownership_after_models_response(monkeypatch, tmp_path):
    module = load_window()
    ops, runner = fake_ours_start(module, monkeypatch, tmp_path)
    async def good_get(url, deadline): return {"data": [{"id": MODEL}]}
    monkeypatch.setattr(runner, "get_json", lambda *args: {"data": [{"id": MODEL}]})
    monkeypatch.setattr(module, "async_get_json", good_get, raising=False)
    def foreign(): raise module.WindowStopped("configured frontend owned by foreign PID")
    ops.verify_ours_frontends = foreign
    with pytest.raises(module.WindowStopped, match="foreign"):
        asyncio.run(ops.start_ours(tmp_path))


def test_startup_post_sample_idle_is_actively_cancelled_on_observer_error(monkeypatch, tmp_path):
    module = load_window()
    import types
    import os
    monkeypatch.setattr(os, "environ", {})
    root = tmp_path / "run"; root.mkdir(); (root / "workers").mkdir()
    elf = tmp_path / "elf"; elf.write_bytes(b"synthetic-elf")
    now = module.time.time()
    ops = object.__new__(module.C1RuntimeOperations)
    ops.root = root
    ops.runtime = {"source_dir": str(ROOT), "gateway_binary": str(elf), "run_dir": str(root),
        "approval_container_path": str(tmp_path / "approval.json"), "cpuset_cpus": "1,3",
        "ports": {"prefill": 38100, "decode": 38110}}
    ops.config = {"runtime": ops.runtime}
    ops.plan = {"run_id": RUN, "window_anchor_unix_s": now, "hard_deadline_unix_s": now + 2700,
        "cleanup_reserve_s": 180, "source_file_sha256": {}, "input_sha256": {},
        "frozen_metadata": {"gateway_elf_sha256": module.sha256(elf)}}
    ops.budget = module.WindowBudget(anchor_unix_s=now); ops.owner_id = OWNER; ops.expected = {}
    ops.window = module.TransportWindow(run_id=RUN, budget=ops.budget, plan_sha256=module.plan_sha256(ops.plan))
    ops.window.register_owned_container(OWNER)
    ops.approval = {"window_run_id": RUN, "approved_plan_sha256": module.plan_sha256(ops.plan)}
    ops.capture_status = None
    class Children:
        def healthy(self): return True
        def start(self, name, argv, env, path, **kwargs):
            if name == "reference-supervisor":
                (root / "workers" / "status.json").write_text(json.dumps({"phase": "ready", "processes": []}))
            else:
                (root / "observer").mkdir()
                (root / "observer" / "summary.latest.json").write_text(json.dumps(observer(sampled_unix_s=module.time.time())))
            return {"process": types.SimpleNamespace(pid=123, poll=lambda: None), "name": name}
    ops.children = Children()
    monkeypatch.setattr(module, "render_c1_runtime_recipe", lambda *args, **kwargs:
        {"reference_environment": {"READY_TIMEOUT_S": "30"}, "reference_argv": ["synthetic-never-exec"]})
    monkeypatch.setattr(module, "collect_startup_registration", lambda *args, **kwargs:
        {"log_paths": [], "endpoint_observation_ready": True})
    monkeypatch.setattr(module, "_bounded_qc_json", lambda *args: None)
    async def bad_idle(*args, **kwargs):
        (root / "observer" / "summary.latest.json").write_text(json.dumps(observer(
            sampled_unix_s=module.time.time(), first_error_unix_s=module.time.time(), stop_requested=True)))
        await asyncio.Future()
    monkeypatch.setattr(module, "_idle", bad_idle)
    async def scenario():
        try:
            with pytest.raises(module.WindowStopped):
                await asyncio.wait_for(ops.startup(), timeout=0.5)
        except asyncio.TimeoutError:
            pytest.fail("startup idle ignored the first-error observer alarm")
    asyncio.run(scenario())


def test_capped_command_pipes_keep_draining_until_owned_child_termination(monkeypatch):
    module = load_window()
    async def scenario():
        terminated = asyncio.Event()
        chunks = [b"x" * 65536 for _ in range(20)]
        class Stream:
            def __init__(self, data): self.data = list(data); self.drained = 0
            async def read(self, size):
                block = self.data.pop(0) if self.data else b""
                assert len(block) <= size
                self.drained += len(block)
                return block
        class Process:
            returncode = None
            stdout = Stream(chunks); stderr = Stream([])
            def terminate(self): self.returncode = -15; terminated.set()
            def kill(self): self.returncode = -9; terminated.set()
            async def wait(self): await terminated.wait(); return self.returncode
        process = Process()
        async def create(*args, **kwargs): return process
        monkeypatch.setattr(module.asyncio, "create_subprocess_exec", create)
        with pytest.raises(module.WindowStopped):
            await module.DeferredCommandExecutor().run(["synthetic-never-exec"], deadline_unix_s=module.time.time() + 2)
        assert terminated.is_set()
        assert process.stdout.drained == sum(map(len, chunks)), "capped stdout abandoned buffered producer bytes instead of draining"
    asyncio.run(scenario())


@pytest.mark.parametrize("late_failure", ["truncated", "write_error", "unresolved_thread"])
def test_late_owned_drain_failure_remains_sticky_after_retirement(late_failure):
    module = load_window()
    import types
    class Process:
        pid = 12345
        def poll(self): return 0
        def wait(self, timeout=None): return 0
    entry = {"name": "ours-sidecar", "process": Process(), "pgid": 12345,
        "stopping": True, "truncated": False, "write_error": None}
    class Thread:
        def join(self, timeout):
            if late_failure == "truncated": entry["truncated"] = True
            elif late_failure == "write_error": entry["write_error"] = "synthetic late disk failure"
        def is_alive(self): return late_failure == "unresolved_thread"
    entry["thread"] = Thread()
    children = module.RuntimeChildren(); children.entries.append(entry)
    with pytest.raises(module.WindowStopped):
        asyncio.run(children.stop(module.time.time() + 30, [entry]))
    assert not children.healthy(), "retired incomplete capture must not disappear from the stage guard"


def minimal_method_ops(module, tmp_path):
    import types
    ops = object.__new__(module.C1RuntimeOperations)
    ops.root = tmp_path; ops.plan = {"run_id": RUN}; ops.model = MODEL
    ops.runtime = {"source_dir": str(ROOT)}; ops.expected = {}
    ops.budget = module.WindowBudget(anchor_unix_s=module.time.time())
    ops.epoch = types.SimpleNamespace(require_bound=lambda: {"model": MODEL,
        "implementation_revision": REVISION, "runtime_evidence_sha256": EPOCH})
    ops.phase_records = []; ops.ours_entries = []; ops.capture_status = None
    ops.endpoint = lambda mode: {"prefill": "http://prefill", "decode": "http://decode",
        "ours": "synthetic://ours", "native": "synthetic://native"}[mode]
    ops.api = workflow_api(); ops.raw_budget = types.SimpleNamespace(replay=None)
    return ops


def test_final_drain_failure_rewrites_previously_successful_actual_phase_gate(monkeypatch, tmp_path):
    module = load_window(); ops = minimal_method_ops(module, tmp_path)
    items = workflows()
    async def start_ours(directory):
        ops.ours_entries = [{"name": "ours-sidecar"}]
        ops.capture_status = directory / "capture-status.latest.json"
        ops.capture_status.write_text(json.dumps({"observation_complete": True, "incomplete_reasons": []}))
        (directory / "identity-capture.jsonl").write_text("".join(json.dumps(row) + "\n" for row in capture_records(items)))
    ops.start_ours = start_ours
    class Children:
        async def stop(self, *args): raise module.WindowStopped("late final capture drain incomplete")
    ops.children = Children()
    async def idle(*args, **kwargs): return {"success": True}
    async def replay(api, selected, **kwargs):
        rows = [{"workflow_id": wf.workflow_id, "call_id": call.call_id, "parents": call.parents,
            "workflow_arrival_ms": 1000, "request_started_at_ms": 1000, "response_completed_at_ms": 1010,
            "prefill_worker": "http://prefill", "decode_worker": "http://decode", "schedule_revision": BIG_REVISION + index,
            "error": None, "response": {"model": MODEL, "usage": {"prompt_tokens": call.input_len, "completion_tokens": call.output_len}}}
            for index, (wf, call) in enumerate((wf, call) for wf in selected for call in wf.calls)]
        kwargs["raw_path"].write_text("".join(json.dumps(row) + "\n" for row in rows))
        return rows
    monkeypatch.setattr(module, "_idle", idle)
    monkeypatch.setattr(module, "replay_workflows", replay)
    monkeypatch.setattr(module, "_bounded_qc_json", lambda path, value: module.write_json(path, value, fresh=not Path(path).exists()))
    with pytest.raises(module.WindowStopped, match="drain"):
        asyncio.run(ops.method_phase("calibration-ours", "ours", items, calibration=True))
    gate = json.loads((tmp_path / "calibration-ours" / "measurement-gate.json").read_text())
    assert gate["successful"] is False
    assert ops.phase_records[-1]["successful"] is False


def test_formal_warmup_failure_stays_not_started_with_zero_of_133_calls(monkeypatch, tmp_path):
    module = load_window()
    import types
    config = runtime_config(tmp_path)
    ops = minimal_method_ops(module, tmp_path / "run")
    ops.root.mkdir()
    ops.hot = ops.api.load_workflow_traces_jsonl(Path(config["runtime_input_dir"]) / "hotpotqa.heldout.sanitized.runtime.jsonl")
    ops.calibration = []; ops.children = types.SimpleNamespace(entries=[], stop=None)
    async def stop(*args, **kwargs): return None
    ops.children.stop = stop
    async def startup(): return None
    ops.startup = startup
    async def run_stage(name, operation): return await operation(deadline_unix_s=ops.budget.work_deadline_unix_s)
    ops.run_stage = run_stage
    original_method = module.C1RuntimeOperations.method_phase
    async def method(name, selected_method, selected, *, calibration):
        if name.startswith("formal-08"):
            return await original_method(ops, name, selected_method, selected, calibration=calibration)
        (ops.root / name).mkdir()
        result = {"successful": True, "phase": name, "actual_phase_started_unix_s": module.time.time()}
        ops.phase_records.append(result); return result
    ops.method_phase = method
    async def start_ours(directory): return None
    ops.start_ours = start_ours
    async def idle(*args, **kwargs): return {"success": True}
    async def failed_warmup(api, selected, **kwargs):
        assert len(selected) == 1 and selected[0].workflow_type == "uniform_warmup"
        kwargs["raw_path"].write_text(json.dumps({"workflow_id": selected[0].workflow_id,
            "call_id": selected[0].calls[0].call_id, "error": "synthetic KV warmup failure"}) + "\n")
        raise module.WindowStopped("synthetic KV warmup failure before formal input")
    async def references(api, selected, **kwargs):
        directory = kwargs["output_dir"]; directory.mkdir()
        contract = {"model": MODEL, "implementation_revision": REVISION, "runtime_evidence_sha256": EPOCH}
        rows = [{"workflow_id": wf.workflow_id, "reference_kind": "isolated_workflow_replay", "latency_ms": 100,
            "status": "successful", "expected_calls": len(wf.calls), "completed_calls": len(wf.calls),
            "usage_verified_calls": len(wf.calls), "errors": 0, "raw_replay_sha256": "e" * 64, **contract} for wf in selected]
        (directory / "workflow-references.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return {"successful": True}
    class Executor:
        async def run(self, *args, **kwargs): return {"returncode": 0, "stdout": "", "stderr": ""}
    monkeypatch.setattr(module, "DeferredCommandExecutor", Executor)
    monkeypatch.setattr(module, "collect_hot_references", references)
    monkeypatch.setattr(module, "_idle", idle)
    monkeypatch.setattr(module, "replay_workflows", failed_warmup)
    monkeypatch.setattr(module, "_bounded_qc_json", lambda path, value: module.write_json(path, value, fresh=not Path(path).exists()))
    assert asyncio.run(ops.execute()) == 2
    final = json.loads((ops.root / "window-results.json").read_text())
    failed = next(row for row in final["phase_records"] if row["phase"].startswith("formal-08"))
    assert failed["actual_phase_started_unix_s"] is None
    assert failed["expected_calls"] == 133 and failed["recorded_calls"] == 0
    assert 8 in final["not_started_formal_ordinals"], "a warmup attempt is not a started formal133-call replay"


def test_actual_w2_parser_never_persists_oversized_duplicate_before_bounded_writer(monkeypatch, tmp_path):
    module = load_window()
    runner = module._pilot("run_paired_pilot")
    parser = load_path("_c1_actual_mutation_parser_test", ROOT / "scripts/hexgent_real_trace_collectors/summarize_worker_mutations.py")
    ops = object.__new__(module.C1RuntimeOperations)
    ops.root = tmp_path; ops.runtime = {"source_dir": str(ROOT)}; ops.w2_saved_bytes = 0
    ops.budget = module.WindowBudget(anchor_unix_s=module.time.time())
    workers = tmp_path / "workers"; workers.mkdir()
    directory = tmp_path / "formal"; directory.mkdir()
    events = [{"worker_port": 37100, "request_id": f"synthetic-{index}", "action_type": "update_waiting_priority",
        "queue": "prefill_waiting", "schedule_revision": index + 1, "expected_revision": 0,
        "success": True, "results": [{"rank": 0, "success": True, "reason_code": "applied"}]} for index in range(512)]
    (workers / "prefill.log").write_text("".join("HexgentMutationOutcome " + json.dumps(event) + "\n" for event in events))
    (workers / "decode.log").write_text("")
    status = {"processes": [{"name": name, "truncated": False, "write_error": None} for name in ("prefill", "decode")]}
    (workers / "status.json").write_text(json.dumps(status))
    before = {"cpu": {"cpu.stat": "usage_usec 10\n"}, "status": status, "offsets": {"prefill": 0, "decode": 0}}
    monkeypatch.setattr(runner, "cpu_snapshot", lambda: {"cpu.stat": "usage_usec 20\n"})
    async def no_sleep(delay): return None
    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)
    commands = []
    class Executor:
        async def run(self, argv, **kwargs):
            commands.append(argv)
            logs = [Path(argv[index + 1]) for index, value in enumerate(argv) if value == "--log"]
            report = parser.summarize_events(parser.read_events(logs))
            report["input_logs"] = [str(path) for path in logs]
            Path(argv[argv.index("--output") + 1]).write_text(json.dumps(report))
            return {"returncode": 0, "stdout": "", "stderr": ""}
    monkeypatch.setattr(module, "DeferredCommandExecutor", Executor)
    limit = 32 * 1024
    def bounded(path, value):
        if len(json.dumps(value).encode()) > limit:
            raise module.WindowStopped("synthetic bounded QC cap")
        module.write_json(path, value, fresh=not Path(path).exists())
    monkeypatch.setattr(module, "_bounded_qc_json", bounded)
    try:
        complete = asyncio.run(ops.formal_auxiliary(directory, before))
    except module.WindowStopped:
        complete = False
    assert complete is False
    assert not (directory / "worker-mutation-summary.observed.json").exists(), "unbounded CLI duplicate already escaped the prewrite cap"
    assert all(path.stat().st_size <= limit for path in directory.glob("*.json"))


def test_four_complete_reference_rows_obey_one_mib_aggregate_prewrite_cap(monkeypatch, tmp_path):
    module = load_window()
    items = workflows(ids=module.HOT_IDS, fanout=True)
    long_model = "synthetic-" + "x" * (300 * 1024)
    class IdleDone:
        def __await__(self):
            async def done(): return {"success": True}
            return done().__await__()
    def idle(*args): return IdleDone()
    async def replay(api, selected, **kwargs):
        rows = raw_rows(selected, revision=0)
        for row in rows: row["response"]["model"] = long_model
        kwargs["raw_path"].write_text("".join(json.dumps(row) + "\n" for row in rows))
        return rows
    output = tmp_path / "refs"
    with pytest.raises(module.WindowStopped, match="cap|budget|reference"):
        asyncio.run(module.collect_hot_references(workflow_api(), items, output_dir=output,
            router_url="synthetic://never-contact", model=long_model, implementation_revision=REVISION,
            runtime_evidence_sha256=EPOCH, run_namespace=RUN + "::refs", expected_limits={},
            budget=module.WindowBudget(anchor_unix_s=1000), clock=lambda: 1100,
            replay_to_raw=replay, idle_checker=idle))
    table = output / "workflow-references.jsonl"
    assert table.stat().st_size <= 1024 * 1024
    assert [row["workflow_id"] for row in map(json.loads, table.read_text().splitlines())] == list(module.HOT_IDS[:3])
    assert len(list(output.glob("standalone-*.raw.jsonl"))) == 4


def test_host_control_aggregate_cap_precedes_any_owned_docker_start(monkeypatch, tmp_path):
    module = load_window()
    import types
    config = runtime_config(tmp_path)
    artifacts = tmp_path / "artifacts"
    config["output_dir"] = str(artifacts / "c1")
    runtime = config["runtime"]
    runtime.update(artifacts_host_path=str(artifacts), run_dir="/artifacts/c1", owner_container_path="/artifacts/c1/owner.json",
        plan_container_path="/artifacts/c1/approved-plan.json", transfer_parameters_path="/artifacts/frozen-transfer.json",
        expected_worker_limits={"prefill": {"max_total_num_tokens": 73197, "max_running_requests": 16,
            "prefill_max_requests": 4, "max_prefill_tokens": 16384},
            "decode": {"max_total_num_tokens": 73197, "max_running_requests": 16,
                "prefill_max_requests": None, "max_prefill_tokens": 16384}})
    plan = module.prepare_plan(config)
    plan["review_metadata"] = "x" * (4 * 1024 * 1024)
    render = module.render_c1_runtime_recipe
    def large_recipe(*args, **kwargs):
        result = render(*args, **kwargs); result["review_metadata"] = "y" * (3 * 1024 * 1024)
        return result
    monkeypatch.setattr(module, "render_c1_runtime_recipe", large_recipe)
    monkeypatch.setattr(module.time, "time", lambda: 1200.0)
    starts = []
    class Launcher:
        def __init__(self, *args): self.window = types.SimpleNamespace(owner_id=None); self.executor = self
        async def start(self, **kwargs): starts.append(True); self.window.owner_id = OWNER; return {"owner_id": OWNER}
        async def run(self, *args, **kwargs): return {"returncode": 0, "stdout": "0\n", "stderr": ""}
        async def cleanup(self, *args, **kwargs): return {"returncode": 0}
    monkeypatch.setattr(module, "C1OwnedLauncher", Launcher)
    approved = {"window_run_id": RUN, "approved_plan_sha256": module.plan_sha256(plan)}
    with pytest.raises((module.WindowStopped, ValueError), match="cap|budget|control"):
        module.C1DefaultOperations(config).execute_c1(plan, approved)
    assert starts == []
    assert sum(path.stat().st_size for path in Path(config["output_dir"]).rglob("*") if path.is_file()) <= 6 * 1024 * 1024


def test_configured_listener_allows_owned_inherited_inode_and_labels_wildcard_binding():
    module = load_window()
    prove = feature(module, "prove_owned_listeners")
    roles, facts, sockets = configured_listener_fixture()
    original = next(item for item in sockets if item["local_port"] == 37100)
    original["local_ip"] = "0.0.0.0"
    child = next(item for item in facts if item["pid"] == 11)
    child["fd_socket_inodes"].append(original["socket_inode"])
    sockets.append({**original, "pid": 11, "start_identity": child["start_identity"]})
    result = prove(roles, facts, sockets, pid_namespace="pid:[42]", net_namespace="net:[43]")
    assert result["prefill_http"]["binding"] == "wildcard_listener"


def test_actual_publisher_double_io_failure_fatally_exits_without_waiting_for_active_longpoll(monkeypatch, tmp_path):
    module = load_window()
    import types
    captured = tmp_path / "identity.jsonl"
    status = tmp_path / "capture-status.latest.json"
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval()))
    fatal_codes, fatal_while_active, forwarded = [], [], []
    active_task = [None]; fatal_event = [None]
    def fatal(code):
        fatal_codes.append(code)
        fatal_while_active.append(active_task[0] is not None and not active_task[0].done())
        fatal_event[0].set()
    monkeypatch.setattr(module, "owned_publisher_fatal", fatal, raising=False)
    start_message = {"type": "http.response.start", "status": 200, "headers": []}
    body_message = {"type": "http.response.body", "body": b"{}", "more_body": False}
    class Telemetry:
        def __init__(self, path): self.path = Path(path)
    started = [None]
    async def actual_app(scope, receive, send):
        if scope["path"] == "/wait_plan_revision":
            started[0].set()
            await asyncio.Future()  # Existing valid longpoll remains active.
        await receive()
        await send(start_message)
        await send(body_message)
    sidecar = types.ModuleType("sglang.hexgent.sidecar")
    sidecar.PlannerTelemetry = Telemetry
    sidecar.build_app = lambda **kwargs: actual_app
    for name in ("sglang", "sglang.hexgent"):
        package = types.ModuleType(name); package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "sglang.hexgent.sidecar", sidecar)
    original_unlink = Path.unlink
    def unlink(path, *args, **kwargs):
        if path == status or path == status.with_suffix(".tmp"):
            raise OSError("synthetic unlink denied; previous healthy file survives")
        return original_unlink(path, *args, **kwargs)
    class Server:
        def __init__(self, config): self.app = config.app; self.should_exit = False
        async def serve(self):
            started[0] = asyncio.Event(); fatal_event[0] = asyncio.Event()
            publisher = self.app.status_writer.__self__
            def replace(temporary): raise OSError("synthetic atomic replace denied")
            monkeypatch.setattr(publisher, "_replace", replace)
            monkeypatch.setattr(Path, "unlink", unlink)
            async def receive_empty(): return {"type": "http.request", "body": b"", "more_body": False}
            async def send(message): forwarded.append(message)
            active_task[0] = asyncio.create_task(self.app({"type": "http", "method": "GET",
                "path": "/wait_plan_revision", "query_string": b""}, receive_empty, send))
            await started[0].wait()
            event_body = json.dumps({"event_type": "request_dispatched", "workflow_id": "runtime-wf",
                "call_id": "runtime-call", "payload": {}}).encode()
            async def receive_event(): return {"type": "http.request", "body": event_body, "more_body": False}
            try:
                await self.app({"type": "http", "method": "POST", "path": "/event",
                    "query_string": b""}, receive_event, send)
                try:
                    await asyncio.wait_for(fatal_event[0].wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass  # A graceful should_exit flag cannot finish this active longpoll.
            finally:
                active_task[0].cancel()
                await asyncio.gather(active_task[0], return_exceptions=True)
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.Config = lambda app, **kwargs: types.SimpleNamespace(app=app)
    uvicorn.Server = Server
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    module.passive_sidecar_main(["--capture-path", str(captured), "--capture-status-path", str(status),
        "--telemetry-path", str(tmp_path / "telemetry.jsonl"), "--approval-file", str(approval_path),
        "--plan-sha256", PLAN_SHA, "--run-id", RUN, "--deadline-unix-s", str(module.time.time() + 2)])
    assert status.exists() and json.loads(status.read_text())["observation_complete"] is True
    assert forwarded[0] is start_message and forwarded[1] is body_message
    assert fatal_codes and all(code != 0 for code in fatal_codes), "failed owned publisher stayed alive behind an active graceful longpoll"
    assert any(fatal_while_active), "fatal publication signal must not await the existing active request"


@pytest.mark.parametrize("io_phase", ["open", "write", "flush", "close"])
def test_actual_actor_telemetry_io_failure_marks_capture_incomplete_and_unlocks(io_phase, tmp_path):
    module = load_window()
    import ast
    import threading
    import types
    from typing import Optional
    # Execute only the real writer class, without FastAPI/SGLang/GPU imports or
    # __init__ filesystem work. Its actor record_admission_batch body is unchanged.
    source = ROOT / "python/sglang/hexgent/sidecar.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PlannerTelemetry")
    namespace = {"json": json, "threading": threading, "Path": Path, "Optional": Optional,
        "os": types.SimpleNamespace(environ={}), "time": types.SimpleNamespace(time=lambda: 1100,
            perf_counter_ns=lambda: 1, process_time_ns=lambda: 1)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), namespace)
    capture = module.ASGIEvidenceCapture(None, lambda row: None)
    notifications = []
    def incomplete(reason):
        capture._incomplete(reason)
        notifications.append(capture.status())
        if io_phase == "close":
            raise RuntimeError("synthetic callback publication failure")
    class File:
        def tell(self): return 0
        def write(self, payload):
            if io_phase == "write": raise OSError("synthetic telemetry write failure")
            return len(payload)
        def flush(self):
            if io_phase == "flush": raise OSError("synthetic telemetry flush failure")
        def close(self):
            if io_phase == "close": raise OSError("synthetic telemetry close failure")
    class FakePath:
        def open(self, *args, **kwargs):
            if io_phase == "open": raise OSError("synthetic telemetry open failure")
            return File()
    wrapped = module._TelemetryPath(tmp_path / "never-created", 1024, incomplete)
    wrapped.path = FakePath()
    telemetry_class = namespace["PlannerTelemetry"]
    telemetry = telemetry_class.__new__(telemetry_class)
    telemetry.path = wrapped; telemetry._lock = threading.Lock(); telemetry._process_thread_count = lambda: 1
    with pytest.raises(OSError):
        telemetry.record_admission_batch(admission_count=1, event_count=0, result_count=1,
            wall_started_ns=0, cpu_started_ns=0)
    assert notifications and notifications[-1]["observation_complete"] is False
    acquired = wrapped.lock.acquire(blocking=False)
    if acquired: wrapped.lock.release()
    assert acquired, "telemetry I/O/publication failure leaked the actor writer lock"



def test_preexisting_owned_publisher_exit_is_latched_before_retirement_and_other_drains_finish():
    module = load_window()
    import types
    joined = []
    class ExitedProcess:
        def __init__(self, pid, code): self.pid, self.code = pid, code
        def poll(self): return self.code
        def wait(self, timeout=None): return self.code
    failed = {"name": "ours-sidecar", "process": ExitedProcess(12345, 70), "pgid": 12345,
        "truncated": False, "write_error": None,
        "thread": types.SimpleNamespace(join=lambda timeout: joined.append("failed-publisher"), is_alive=lambda: False)}
    other = {"name": "ours-gateway", "process": ExitedProcess(12346, 0), "pgid": 12346,
        "stopping": True, "truncated": False, "write_error": None,
        "thread": types.SimpleNamespace(join=lambda timeout: joined.append("other-owned"), is_alive=lambda: False)}
    children = module.RuntimeChildren(); children.entries = [failed, other]
    with pytest.raises(module.WindowStopped):
        asyncio.run(children.stop(module.time.time() + 30, [failed, other]))
    assert joined == ["failed-publisher", "other-owned"]
    assert not children.healthy(), "retirement erased the already-fatal owned publisher exit"



def test_owned_publisher_fatal_exit_during_retirement_survives_the_expected_signal(monkeypatch):
    module = load_window()
    import os
    import types
    state = {"code": None}; joined = []
    class Process:
        pid = 12345
        def poll(self): return state["code"]
        def wait(self, timeout=None): return state["code"]
    entry = {"name": "ours-sidecar", "process": Process(), "pgid": 12345,
        "truncated": False, "write_error": None,
        "thread": types.SimpleNamespace(join=lambda timeout: joined.append(True), is_alive=lambda: False)}
    children = module.RuntimeChildren(); children.entries.append(entry)
    monkeypatch.setattr(os, "getpgid", lambda pid: 12345)
    def publication_wins_signal_race(pgid, signal):
        assert pgid == 12345
        state["code"] = 70
    monkeypatch.setattr(os, "killpg", publication_wins_signal_race)
    with pytest.raises(module.WindowStopped):
        asyncio.run(children.stop(module.time.time() + 30, [entry]))
    assert joined == [True]
    assert not children.healthy(), "expected retirement masked the explicit owned-publisher fatal exit70"



def test_docker_recipe_sets_startup_identity_without_changing_numeric_user_or_runtime_paths(tmp_path):
    module = load_window()
    config = runtime_config(tmp_path)
    argv = module.render_c1_runtime_recipe(config, now_unix_s=1200.0)["docker_argv"]
    environment = dict(argv[index + 1].split("=", 1) for index, value in enumerate(argv) if value == "--env")
    assert argv[argv.index("--user") + 1] == "1002:1002"
    assert {key: environment[key] for key in ("HOME", "TMPDIR", "HF_HOME")} == {
        "HOME": "/runtime/home", "TMPDIR": "/runtime/tmp", "HF_HOME": "/runtime/hf"}
    assert {key: environment.get(key) for key in ("USER", "LOGNAME")} == {
        "USER": "fred", "LOGNAME": "fred"}

def _blocked_startup_listener_diagnostic(module, invoke):
    with pytest.raises((ValueError, module.WindowStopped)) as stopped:
        invoke()
    message = str(stopped.value)
    assert message.startswith("startup role is not uniquely joined to an owned LISTEN FD inode")
    marker = "; owned_listener_diagnostic="
    assert marker in message, "blocked startup did not retain structured owned-listener failure evidence"
    evidence = json.loads(message.split(marker, 1)[1])
    assert evidence["schema_version"] == 1
    assert evidence["run_id"] == RUN and evidence["owner_id"] == OWNER
    assert evidence["stream"] == "prefill" and evidence["kind"] == "session"
    assert evidence["log_tid"] == 1011 and evidence["log_offset"] == 0
    assert evidence["registered_peer"] == {"ip": "192.0.2.1", "port": 40011, "kind": "session"}
    assert evidence["owned_pid"] == 11 and evidence["owned_start_identity"] == "111"
    assert evidence["candidate_limit"] == 16
    assert len(evidence["candidates"]) <= 16
    assert evidence["candidates_omitted_count"] == evidence["owned_fd_candidate_row_count"] - len(evidence["candidates"])
    assert len(json.dumps(evidence).encode()) < 8192
    return evidence


def test_startup_listener_zero_match_reports_scoped_owned_evidence_and_stays_blocked(monkeypatch, tmp_path):
    module = load_window()
    import types
    fixture = registration_fixture()
    missing = fixture["socket_facts"].pop(0)
    source_status = {name: "ok" for name in ("processes", "sockets", "namespace_counts",
        "port_range", "cgroup", "allowed_env")}
    actual = {"source_status": source_status, "read_errors": [], "sockets": fixture["socket_facts"]}
    class Producer:
        def __init__(self, registration, reader): pass
        def collect(self, now): return actual
    observer = types.SimpleNamespace(LinuxIPv4FactProducer=Producer, REQUIRED_SOURCES=tuple(source_status))
    monkeypatch.setattr(module, "_observer_api", lambda: observer)
    monkeypatch.setattr(module, "_read_owned_processes", lambda *args, **kwargs:
        (fixture["process_facts"], fixture["pid_namespace"], fixture["net_namespace"]))
    class Reader:
        def read_text(self, path):
            stream = Path(path).stem
            assert stream in {"prefill", "decode"}
            return "\n".join(row["line"] for row in fixture["endpoint_records"] if row["stream"] == stream)
    plan = {"run_id": RUN, "window_anchor_unix_s": 1000, "hard_deadline_unix_s": 3700,
        "cleanup_reserve_s": 180, "source_file_sha256":
        {name: module.sha256(ROOT / name) for name in module.CONFIG_SOURCE_PATHS}}
    config = {"runtime": {"run_dir": str(tmp_path), "source_dir": str(ROOT)}}
    evidence = _blocked_startup_listener_diagnostic(module, lambda:
        module.collect_startup_registration(config, plan, OWNER, fixture["launch_status"],
            supervisor_pid=1, reader=Reader()))
    assert missing["local_port"] == 40011
    assert evidence["raw_row_count"] == 0
    assert evidence["distinct_socket_identity_count"] == 0
    assert evidence["owned_fd_candidate_row_count"] == 0
    assert evidence["owned_fd_candidate_distinct_socket_identity_count"] == 0
    assert evidence["candidates"] == [] and evidence["candidates_truncated"] is False
    assert evidence["known_source_status"] == source_status


def test_startup_listener_identical_rows_report_raw_vs_distinct_and_stay_blocked():
    module = load_window()
    fixture = registration_fixture()
    fixture["budget"] = module.WindowBudget(anchor_unix_s=1000)
    fixture["socket_facts"].insert(0, copy.deepcopy(fixture["socket_facts"][0]))
    evidence = _blocked_startup_listener_diagnostic(module, lambda:
        module.build_observer_registration(**fixture))
    assert evidence["raw_row_count"] == 2
    assert evidence["distinct_socket_identity_count"] == 1
    assert evidence["owned_fd_candidate_row_count"] == 2
    assert evidence["owned_fd_candidate_distinct_socket_identity_count"] == 1
    assert len(evidence["candidates"]) == 2 and evidence["candidates_truncated"] is False
    assert all(row["family"] is None for row in evidence["candidates"]), "family must not be inferred from an address"
    assert evidence["known_source_status"] == {}


def test_startup_listener_distinct_ambiguous_rows_report_two_identities_and_stay_blocked():
    module = load_window()
    fixture = registration_fixture()
    fixture["budget"] = module.WindowBudget(anchor_unix_s=1000)
    process = next(row for row in fixture["process_facts"] if row["pid"] == 11)
    process["fd_socket_inodes"].append("30011")
    fixture["socket_facts"].insert(0, {**fixture["socket_facts"][0], "socket_inode": "30011"})
    evidence = _blocked_startup_listener_diagnostic(module, lambda:
        module.build_observer_registration(**fixture))
    assert evidence["raw_row_count"] == 2 and evidence["distinct_socket_identity_count"] == 2
    assert evidence["owned_fd_candidate_row_count"] == 2
    assert evidence["owned_fd_candidate_distinct_socket_identity_count"] == 2
    assert {row["socket_inode"] for row in evidence["candidates"]} == {"10011", "30011"}


def test_startup_listener_diagnostic_excludes_foreign_unowned_and_unrelated_fds():
    module = load_window()
    fixture = registration_fixture()
    fixture["budget"] = module.WindowBudget(anchor_unix_s=1000)
    original = fixture["socket_facts"].pop(0)
    process = next(row for row in fixture["process_facts"] if row["pid"] == 11)
    other = next(row for row in fixture["process_facts"] if row["pid"] == 21)
    process["fd_socket_inodes"].append("30011")
    fixture["socket_facts"].extend([
        {**original, "pid": 999, "socket_inode": "foreign-pid", "remote_ip": "FOREIGN_REMOTE"},
        {**original, "start_identity": "reused", "remote_ip": "FOREIGN_REMOTE"},
        {**original, "socket_inode": "not-an-owned-fd", "remote_ip": "FOREIGN_REMOTE"},
        {**original, "pid": 21, "start_identity": other["start_identity"],
            "socket_inode": other["fd_socket_inodes"][0], "remote_ip": "FOREIGN_REMOTE"},
        {**original, "socket_inode": "30011", "local_ip": "::", "family": "AF_INET6",
            "remote_ip": "UNRETAINED_REMOTE", "environment": {"SECRET": "UNRETAINED_ENV"}},
    ])
    evidence = _blocked_startup_listener_diagnostic(module, lambda:
        module.build_observer_registration(**fixture))
    assert evidence["raw_row_count"] == 0 and evidence["distinct_socket_identity_count"] == 0
    assert evidence["owned_fd_candidate_row_count"] == 1
    assert evidence["owned_fd_candidate_distinct_socket_identity_count"] == 1
    assert evidence["candidates"] == [{"pid": 11, "start_identity": "111", "socket_inode": "30011",
        "local_ip": "::", "local_port": 40011, "state": "LISTEN", "family": "AF_INET6"}]
    encoded = json.dumps(evidence)
    assert "FOREIGN_REMOTE" not in encoded and "UNRETAINED_REMOTE" not in encoded and "UNRETAINED_ENV" not in encoded
    assert "45011" not in encoded and "40021" not in encoded


def test_startup_listener_diagnostic_caps_verified_owned_candidates_without_accepting():
    module = load_window()
    fixture = registration_fixture()
    fixture["budget"] = module.WindowBudget(anchor_unix_s=1000)
    original = fixture["socket_facts"].pop(0)
    process = next(row for row in fixture["process_facts"] if row["pid"] == 11)
    for index in range(20):
        inode = str(30000 + index)
        process["fd_socket_inodes"].append(inode)
        fixture["socket_facts"].append({**original, "socket_inode": inode,
            "local_ip": "192.0.2.99", "family": "AF_INET", "remote_ip": "UNRETAINED_REMOTE"})
    evidence = _blocked_startup_listener_diagnostic(module, lambda:
        module.build_observer_registration(**fixture))
    assert evidence["raw_row_count"] == 0
    assert evidence["owned_fd_candidate_row_count"] == 20
    assert evidence["owned_fd_candidate_distinct_socket_identity_count"] == 20
    assert len(evidence["candidates"]) == 16 and evidence["candidates_truncated"] is True
    assert evidence["candidates_omitted_count"] == 4
    assert all(row["pid"] == 11 and row["start_identity"] == "111" and row["local_port"] == 40011
        and row["family"] == "AF_INET" for row in evidence["candidates"])
    assert "UNRETAINED_REMOTE" not in json.dumps(evidence)


def v7_registration_fixture(module):
    fixture=registration_fixture();fixture["budget"]=module.WindowBudget(anchor_unix_s=1000)
    properties=[]
    for row in fixture["socket_facts"]:
        row["family"]="AF_INET"
        if row["local_port"]<45000:continue
        row.update(family="AF_INET6",local_ip="::",remote_ip="::")
        properties.append({**row,"kind":"tcp","run_id":RUN,"owner_id":OWNER,
            "net_namespace":"net:[43]","observed_unix_s":1001.5,"tool_sha256":"3"*64,
            "ss_v6only":0,"raw_owned_record":f"LISTEN 0 4096 *:{row['local_port']} *:* ino:{row['socket_inode']} v6only:0",
            "observation_scope":"held_listener_lifetime"})
    fixture["listener_properties"]=properties
    return fixture

def test_current_ipv6_data_v6only_zero_joins_ipv4_advert_without_changing_rpc_identity():
    module=load_window();fixture=v7_registration_fixture(module)
    actual=module.build_observer_registration(**fixture)
    assert actual["endpoint_observation_ready"] is True
    by_pid={}
    for row in actual["endpoint_binding_evidence"]:by_pid.setdefault(row["pid"],[]).append(row)
    for rows in by_pid.values():
        assert len(rows)==2
        rpc=next(row for row in rows if row["peer"]["kind"]=="session")
        data=next(row for row in rows if row["peer"]["kind"]=="tcp")
        assert rpc["socket_inode"]!=data["socket_inode"]
        assert rpc.get("listener_property") is None
        assert data["family"]=="AF_INET6" and data["local_ip"]=="::"
        assert data["listener_property"]["ss_v6only"]==0
        assert data["binding"]=="ipv6_wildcard_v6only_0"
    assert len(actual["listener_properties"])==4

@pytest.mark.parametrize("change",["one","missing","pid","inode","port","netns","owner"])
def test_ipv6_data_join_requires_current_exact_zero_property_and_no_other_role_inference(change):
    module=load_window();fixture=v7_registration_fixture(module)
    row=fixture["listener_properties"][0]
    if change=="one":row["ss_v6only"]=1
    elif change=="missing":fixture["listener_properties"].pop(0)
    elif change=="pid":row["start_identity"]="old"
    elif change=="inode":row["socket_inode"]="other"
    elif change=="port":row["local_port"]+=1
    elif change=="netns":row["net_namespace"]="net:[foreign]"
    else:row["owner_id"]="b"*64
    with pytest.raises(ValueError):module.build_observer_registration(**fixture)

def test_identical_ipv6_listener_rows_remain_ambiguous_without_deduplication():
    import copy
    module=load_window();fixture=v7_registration_fixture(module)
    row=next(row for row in fixture["socket_facts"] if row["family"]=="AF_INET6")
    fixture["socket_facts"].append(copy.deepcopy(row))
    with pytest.raises(ValueError):module.build_observer_registration(**fixture)


def test_rejected_ipv6_only_join_retains_bounded_current_property_and_missing_is_unknown():
    import copy
    module=load_window()
    for missing in (False,True):
        fixture=v7_registration_fixture(module)
        owned=fixture["listener_properties"][0]
        owned.update(ss_v6only=1,source="fixed_ss_-6_exact_port_owned_inode")
        foreign=copy.deepcopy(owned)
        foreign.update(pid=999,socket_inode="foreign",source="FOREIGN_SECRET",raw_owned_record="FOREIGN_SECRET")
        if missing:fixture["listener_properties"].pop(0)
        fixture["listener_properties"].append(foreign)
        with pytest.raises(ValueError) as error:module.build_observer_registration(**fixture)
        text=str(error.value);assert "FOREIGN_SECRET" not in text
        diagnostic=json.loads(text.split("; owned_listener_diagnostic=",1)[1])
        assert len(diagnostic["candidates"])<=16
        prop=diagnostic["candidates"][0]["listener_property"]
        assert prop["status"] == ("unknown" if missing else "observed")
        assert prop["ss_v6only"] == (None if missing else 1)
        if not missing:
            assert prop["source"]=="fixed_ss_-6_exact_port_owned_inode"
            assert prop["tool_sha256"]=="3"*64
            assert prop["observed_unix_s"]==1001.5
        assert "raw_owned_record" not in prop


def test_ours_frontend_wrapper_scopes_worker_listener_proofs_out_of_real_producer(monkeypatch,tmp_path):
    import types
    module=load_window();observer_api=module._observer_api()
    fixture=v7_registration_fixture(module)
    reg={"run_id":RUN,"owner_id":OWNER,"deadline_unix_s":module.time.time()+120,
        "pid_namespace":"pid:[42]","net_namespace":"net:[43]","cgroup_path":"/fake-cgroup",
        "processes":[{"pid":11,"start_identity":"111"}],
        "peers":[{"ip":"192.0.2.1","port":45011,"kind":"tcp"}],
        "source_hashes":{name:"7"*64 for name in module.CONFIG_SOURCE_PATHS},
        "listener_properties":[fixture["listener_properties"][0]]}
    persisted=tmp_path/"observer-registration.json"
    original=json.dumps(reg,sort_keys=True).encode();persisted.write_bytes(original)
    process_facts=[{"pid":pid,"ppid":1,"pgrp":pid,"start_identity":str(pid+100),
        "start_identity_after":str(pid+100),"pid_namespace":"pid:[42]","net_namespace":"net:[43]",
        "task_ids":[pid],"fd_socket_inodes":[str(pid+1000)]} for pid in (50,60)]
    header="sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
    files={"/proc/net/tcp":header,"/proc/net/tcp6":header,
        "/proc/sys/net/ipv4/ip_local_port_range":"32768 60999\n",
        "/proc/sys/net/ipv4/ip_local_reserved_ports":"\n",
        "/fake-cgroup/memory.events":"low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
        "/fake-cgroup/memory.peak":"1048576\n"}
    for pid,port in ((50,37140),(60,37150)):
        fields=["S"]+["0"]*19;fields[19]=str(pid+100)
        files[f"/proc/{pid}/stat"]=str(pid)+" (owned frontend) "+" ".join(fields)
        files[f"/proc/{pid}/limits"]="Max open files 1024 4096 files\n"
        files[f"/proc/{pid}/environ"]="SGLANG_MOONCAKE_TRANSFER_PROTOCOL=tcp\0"
        files["/proc/net/tcp"]+=f"0: 0100007F:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1002 0 {pid+1000} 1\n"
    calls=[]
    class Reader:
        def read_text(self,path):
            calls.append(("read",str(path)));assert str(path) in files
            return files[str(path)]
        def readlink(self,path):
            calls.append(("link",str(path)))
            if str(path)=="/proc/self/ns/pid":return "pid:[42]"
            if str(path).endswith("/ns/net"):return "net:[43]"
            raise AssertionError("unexpected namespace read")
        def list_fd_targets(self,pid,proc_root="/proc"):
            calls.append(("fd",pid));assert pid in (50,60)
            return ["socket:["+str(pid+1000)+"]"]
    monkeypatch.setattr(observer_api,"LinuxProcReader",Reader)
    monkeypatch.setattr(module,"_observer_api",lambda:observer_api)
    monkeypatch.setattr(module,"_read_owned_processes",lambda *args,**kwargs:(process_facts,"pid:[42]","net:[43]"))
    ops=object.__new__(module.C1RuntimeOperations);ops.root=tmp_path
    ops.runtime={"ports":{"sidecar":37140,"ours":37150}}
    ops.ours_entries=[{"process":types.SimpleNamespace(pid=pid,poll=lambda:None)} for pid in (50,60)]
    actual=ops.verify_ours_frontends()
    assert set(actual)=={"sidecar_http","ours_http"}
    assert persisted.read_bytes()==original
    assert json.loads(persisted.read_text())["listener_properties"]==reg["listener_properties"]
    assert not any("/11/" in str(call) for call in calls)


def test_actual_ipv6_startup_collector_wires_four_current_log_tid_properties(monkeypatch,tmp_path):
    observed={}
    test_actual_startup_reader_decodes_owned_tasks_fd_tcp_and_configured_frontends(
        monkeypatch,tmp_path,missing_http=False,ipv6_data=True,observed=observed)
    queries=observed["queries"];actual=observed["result"]
    assert len(queries)==5 and queries[0][0]==["/usr/bin/ss","-V"]
    proofs=actual["listener_properties"];assert len(proofs)==4
    assert {row["local_port"] for row in proofs}=={45011,45012,45021,45022}
    for row in proofs:
        assert row["log_tid"]==row["pid"]+1000
        assert row["log_tid_start_identity"]==str(row["log_tid"]+500)
        assert row["socket_inode"]==str(row["pid"]+20000)
        assert row["ss_v6only"]==0
    for argv,timeout,cap in queries[1:]:
        assert argv[:8]==["/usr/bin/ss","-H","-O","-n","-t","-l","-e","-6"]
        assert timeout<=3 and cap==65536
    assert all(row.get("listener_property") is None for row in actual["endpoint_binding_evidence"] if row["peer"]["kind"]=="session")


# Startup-only CPU contracts reuse the existing constructor/input fixtures and
# actual startup orchestration; only process, proc, ss and HTTP I/O are fake.
def startup_only_runtime_fixture(monkeypatch,tmp_path,scenario="success"):
    import types
    observed={};setup=tmp_path/"registration-fixture";setup.mkdir()
    test_actual_startup_reader_decodes_owned_tasks_fd_tcp_and_configured_frontends(
        monkeypatch,setup,missing_http=False,ipv6_data=True,observed=observed)
    module=observed["module"];config=runtime_config(tmp_path)
    clock=[module.time.time()];monkeypatch.setattr(module.time,"time",lambda:clock[0])
    config["window_anchor_unix_s"]=clock[0]
    config["runtime"].update(source_dir=str(ROOT),run_dir=str(tmp_path/"runtime-run"),
        expected_worker_limits={"prefill":{"max_running_requests":16},"decode":{"max_running_requests":16}},
        ss_tool={"path":"/usr/bin/ss","sha256":"31124eb6385718f551751a580c7e17203bc307efa207ab5315b66b4ad5793e8c","version":"ss utility, iproute2-6.1.0"})
    gateway=tmp_path/"gateway";gateway.write_bytes(b"synthetic existing ELF identity")
    model_dir=tmp_path/"model";model_dir.mkdir();(model_dir/"config.json").write_text("{}")
    config["runtime"].update(gateway_binary=str(gateway),model_path=str(model_dir))
    config["gateway_elf_sha256"]=module.sha256(gateway);config["model_config_sha256"]=module.sha256(model_dir/"config.json")
    plan=module.prepare_plan(config,startup_only=True)
    approved={"window_run_id":RUN,"approved_plan_sha256":module.plan_sha256(plan)}
    original_sha=module.sha256
    def mapped_sha(path):
        path=Path(path)
        if str(path).startswith("/inputs/"):path=Path(config["runtime_input_dir"])/path.name
        return original_sha(path)
    monkeypatch.setattr(module,"sha256",mapped_sha)
    support=module._pilot("h20_run_support");api=support.load_workflow_api(ROOT)
    original_load=api.load_workflow_traces_jsonl
    def mapped_load(path):return original_load(Path(config["runtime_input_dir"])/Path(path).name)
    monkeypatch.setattr(api,"load_workflow_traces_jsonl",mapped_load)
    async def forbidden_replay(*args,**kwargs):pytest.fail("startup-only reached experimental model replay")
    monkeypatch.setattr(support,"replay_to_raw",forbidden_replay)
    ops=module.C1RuntimeOperations(config,plan,approved,OWNER)
    for stream in ("prefill","decode"):
        observed["files"][str(ops.root/"workers"/(stream+".log"))]=observed["files"][str(setup/"workers"/(stream+".log"))]
    api_observer=module._observer_api();monkeypatch.setattr(api_observer,"LinuxProcReader",observed["reader_class"])
    status=registration_fixture()["launch_status"]
    class Process:
        def __init__(self,pid):self.pid=pid;self.code=None
        def poll(self):return self.code
    class Children:
        def __init__(self):self.entries=[];self.stop_calls=[]
        def start(self,name,argv,env,path,**kwargs):
            process=Process(1 if name=="reference-supervisor" else 2)
            entry={"name":name,"process":process,"pgid":process.pid,"truncated":False,"write_error":None,
                "thread":types.SimpleNamespace(is_alive=lambda:False)}
            self.entries.append(entry)
            if name=="reference-supervisor":
                directory=ops.root/"workers";directory.mkdir();(directory/"status.json").write_text(json.dumps(status))
            elif name=="transport-observer":
                if scenario=="deadline":clock[0]=ops.budget.work_deadline_unix_s+.01
                directory=ops.root/"observer";directory.mkdir()
                summary=observer(sampled_unix_s=clock[0])
                if scenario=="observer_failure":summary.update(observation_complete=False,stop_requested=True,incomplete_reasons=["synthetic owned sample incomplete"])
                (directory/"summary.latest.json").write_text(json.dumps(summary))
            else:pytest.fail("startup-only started an Ours/experimental child: "+name)
            return entry
        def healthy(self):return all(row["process"].poll() is None for row in self.entries)
        async def stop(self,hard_deadline,entries=None):
            selected=list(self.entries if entries is None else entries)
            self.stop_calls.append(([row["name"] for row in selected],hard_deadline))
            for row in selected:row["process"].code=0
            if scenario=="child_cleanup_failure" and any(row["name"]=="reference-supervisor" for row in selected):
                raise RuntimeError("synthetic owned reference drain failed")
    ops.children=Children()
    async def get(url,deadline):
        assert url.endswith("/hexgent/pd_snapshot"),"startup-only issued a model/experimental request"
        return {"snapshots":[{"limits":{"max_running_requests":16},"running":[],"queues":{"waiting":[]}}]}
    monkeypatch.setattr(module,"async_get_json",get)
    async def forbidden_phase(*args,**kwargs):pytest.fail("startup-only entered calibration/warmup/formal")
    ops.method_phase=forbidden_phase
    return module,ops,clock


def test_startup_only_plan_binds_scope_same_old_constructor_hashes_and_exact_child_flag(tmp_path):
    module=load_window();config=runtime_config(tmp_path)
    old=module.prepare_plan(config);plan=module.prepare_plan(config,startup_only=True)
    assert plan["execution_scope"]=="startup_only"
    assert plan["input_sha256"]==old["input_sha256"]
    assert plan["frozen_helper_sha256"]==old["frozen_helper_sha256"]
    assert plan["calibration"]==old["calibration"] and plan["references"]==old["references"]
    assert plan["unused_workload_metadata"]=="constructor_inputs_only"
    recipe=module.render_c1_runtime_recipe(config,now_unix_s=1200,startup_only=True)
    assert "--startup-only" in recipe["docker_argv"]
    assert recipe["hard_deadline_unix_s"]==3700 and recipe["work_deadline_unix_s"]==3520
    assert module.plan_sha256(plan)!=module.plan_sha256(old)


def test_startup_only_default_dryrun_has_no_runtime_operation(monkeypatch,tmp_path,capsys):
    module=load_window();config=runtime_config(tmp_path);path=tmp_path/"config.json";path.write_text(json.dumps(config))
    class Forbidden:
        def execute_c1(self,*args,**kwargs):pytest.fail("dryrun started owned resources")
    assert module.main(["--config",str(path),"--startup-only","--dry-run"],operations=Forbidden())==0
    assert json.loads(capsys.readouterr().out)["execution_scope"]=="startup_only"
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("scope,flag",[("startup_only",True),("startup_only",False),("full_c1",True)])
def test_startup_only_runtime_dispatch_cannot_fall_through_to_full_pipeline(monkeypatch,tmp_path,scope,flag):
    module=load_window();config=runtime_config(tmp_path)
    plan=module.prepare_plan(config,startup_only=scope=="startup_only")
    approved={"window_run_id":RUN,"approved_plan_sha256":module.plan_sha256(plan)}
    owner={"run_id":RUN,"owner_id":OWNER,"plan_sha256":module.plan_sha256(plan),
        "window_anchor_unix_s":plan["window_anchor_unix_s"],"hard_deadline_unix_s":plan["hard_deadline_unix_s"]}
    paths={}
    for name,value in (("config",config),("plan",plan),("approval-file",approved),("owner-record",owner)):
        paths[name]=tmp_path/(name+".json");paths[name].write_text(json.dumps(value))
    calls=[]
    class Operations:
        def __init__(self,*args):calls.append(("constructor",args))
        async def execute(self):pytest.fail("startup-only scope reached full experimental pipeline")
        async def execute_startup_only(self):calls.append(("startup",));return 0
    monkeypatch.setattr(module,"C1RuntimeOperations",Operations)
    original_mkdir=Path.mkdir
    def mkdir(path,*args,**kwargs):
        if str(path) in ("/runtime/home","/runtime/tmp"):return None
        return original_mkdir(path,*args,**kwargs)
    monkeypatch.setattr(Path,"mkdir",mkdir)
    monkeypatch.setattr(module.time,"time",lambda:1200.)
    argv=[item for name,path in paths.items() for item in ("--"+name,str(path))]+(["--startup-only"] if flag else [])
    if scope=="startup_only" and flag:
        assert module.runtime_supervisor_main(argv)==0
        assert calls[0][1]==(config,plan,approved,OWNER) and calls[1]==("startup",)
    else:
        with pytest.raises(PermissionError):module.runtime_supervisor_main(argv)
        assert calls==[]


@pytest.mark.parametrize("scenario",["success","observer_failure","deadline","child_cleanup_failure"])
def test_startup_only_actual_orchestration_never_replays_and_always_retires_owned_children(monkeypatch,tmp_path,scenario):
    module,ops,clock=startup_only_runtime_fixture(monkeypatch,tmp_path,scenario)
    code=asyncio.run(ops.execute_startup_only())
    report=json.loads((ops.root/"startup-only-results.json").read_text())
    assert code==(0 if scenario=="success" else 2)
    assert report["execution_scope"]=="startup_only" and report["phase_records"]==[]
    assert report["constructor_input_sha256"]==ops.plan["input_sha256"]
    assert report["workload_statuses"]=={name:"NOT_STARTED" for name in ("warmup","calibration","references","replay","formal")}
    assert report["metrics"] is None
    names=[name for group,deadline in ops.children.stop_calls for name in group]
    assert names==["reference-supervisor","transport-observer"]
    assert all(deadline==ops.budget.hard_deadline_unix_s for group,deadline in ops.children.stop_calls)
    assert all(row["process"].poll()==0 for row in ops.children.entries)
    assert report["children_cleanup_complete"] is (scenario!="child_cleanup_failure")
    assert {row["pid"] for row in report["owned_children"]}=={1,2}


@pytest.mark.parametrize("failure",[None,"stop","absence"])
def test_startup_only_exact_container_cleanup_records_absence_without_foreign_targets(tmp_path,failure):
    module=load_window();config=runtime_config(tmp_path);plan=module.prepare_plan(config,startup_only=True)
    calls=[]
    class Executor:
        async def run(self,argv,*,deadline_unix_s):
            calls.append((list(argv),deadline_unix_s))
            if argv[:2]==["docker","stop"] and failure=="stop":raise TimeoutError("synthetic stop lost")
            if argv[:3]==["docker","container","inspect"]:
                if failure=="absence":return {"returncode":0,"stdout":OWNER,"stderr":""}
                return {"returncode":1,"stdout":"","stderr":"Error: No such container: "+OWNER}
            return {"returncode":0,"stdout":OWNER,"stderr":""}
    launcher=module.C1OwnedLauncher(plan,module.render_c1_runtime_recipe(config,now_unix_s=1200,startup_only=True),executor=Executor(),clock=lambda:1200.)
    launcher.window.register_owned_container(OWNER)
    approved={"window_run_id":RUN,"approved_plan_sha256":module.plan_sha256(plan)}
    record=asyncio.run(launcher.cleanup(OWNER,execute=True,approval=approved,verify_absence=True))
    assert record["absence_verified"] is (failure!="absence")
    assert all(argv[-1]==OWNER and deadline==3700 for argv,deadline in calls)
    assert sum(argv[:3]==["docker","rm","--force"] for argv,deadline in calls)==1
    previous=len(calls)
    with pytest.raises(ValueError):asyncio.run(launcher.cleanup(OWNER[:12],execute=True,approval=approved,verify_absence=True))
    assert len(calls)==previous


@pytest.mark.parametrize("failure",[None,"create","wait","absence"])
def test_startup_only_host_always_cleans_exact_created_container_and_requires_absence(monkeypatch,tmp_path,failure):
    module=load_window();config=runtime_config(tmp_path);config["window_anchor_unix_s"]=module.time.time()
    config["output_dir"]=str(tmp_path/"owned-run")
    config["runtime"].update(artifacts_host_path=str(tmp_path),run_dir="/artifacts/owned-run",
        owner_container_path="/artifacts/owned-run/owner.json",expected_worker_limits={},transfer_parameters_path="frozen-unused.json")
    plan=module.prepare_plan(config,startup_only=True)
    approved={"window_run_id":RUN,"approved_plan_sha256":module.plan_sha256(plan)}
    calls=[]
    async def run(executor,argv,*,deadline_unix_s):
        calls.append((list(argv),deadline_unix_s))
        if argv[:2]==["docker","run"]:
            Path(argv[argv.index("--cidfile")+1]).write_text(OWNER)
            if failure=="create":raise OSError("synthetic created then lost acknowledgement")
            return {"returncode":0,"stdout":OWNER,"stderr":""}
        if argv[:2]==["docker","wait"]:
            if failure=="wait":raise TimeoutError("synthetic startup deadline")
            return {"returncode":0,"stdout":"0","stderr":""}
        if argv[:3]==["docker","container","inspect"]:
            if failure=="absence":return {"returncode":0,"stdout":OWNER,"stderr":""}
            return {"returncode":1,"stdout":"","stderr":"Error: No such container: "+OWNER}
        return {"returncode":0,"stdout":OWNER,"stderr":""}
    monkeypatch.setattr(module.DeferredCommandExecutor,"run",run)
    code=module.C1DefaultOperations(config).execute_c1(plan,approved)
    assert code==(0 if failure is None else 2)
    cleanup=json.loads((Path(config["output_dir"])/"owned-cleanup.json").read_text())
    assert cleanup["absence_verified"] is (failure!="absence")
    assert all(argv[-1]==OWNER for argv,deadline in calls if argv[:2]!=["docker","run"])
    assert all(deadline==plan["hard_deadline_unix_s"] for argv,deadline in calls if argv[:2] in (["docker","stop"],["docker","rm"]) or argv[:3]==["docker","container","inspect"])
    assert "--startup-only" in calls[0][0]


# V10: offline four-instance wiring, not Rust/HTTP/GPU execution.
def v10_window(tmp_path):
    module = load_window()
    # Match C1RuntimeOperations.__init__ without replacing its bounded writer.
    module._bounded_qc_json.root = tmp_path
    return module


def v10_window_config(tmp_path):
    config = runtime_config(tmp_path)
    runtime = config["runtime"]
    runtime.update(pd_topology="2p2d_tp2", source_dir=str(ROOT),
        gpu_uuids=[f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(8)],
        cpuset_cpus="4,5,6,7,8,9,10,11,12,13,14,15")
    runtime["ports"].update(prefill_1=37200, bootstrap_1=37220, decode_1=37210)
    runtime["expected_worker_limits"] = {role: {"max_running_requests": 16} for role in ("prefill", "decode")}
    return config


def v10_window_instances(config):
    helper = load_path("_v10_window_actual_layout", SCRIPT.parent / "h20_reference_components.py")
    return helper.reference_worker_layout(config["runtime"]["ports"], topology="2p2d_tp2")


def test_v10_recipe_and_plan_bind_eight_gpus_layout_source_and_explicit_budget(tmp_path):
    module = v10_window(tmp_path); config = v10_window_config(tmp_path)
    recipe = module.render_c1_runtime_recipe(config, now_unix_s=1200.0, startup_only=True)
    plan = module.prepare_plan(config, startup_only=True)
    argv, env = recipe["docker_argv"], recipe["reference_environment"]
    assert argv[argv.index("--gpus") + 1] == '"device=' + ",".join(config["runtime"]["gpu_uuids"]) + '"'
    assert argv[argv.index("--cpus") + 1] == "12"
    assert argv[argv.index("--memory") + 1] == argv[argv.index("--memory-swap") + 1] == "512g"
    assert argv[argv.index("--shm-size") + 1] == "2g"
    assert "size=12g" in argv[argv.index("--tmpfs") + 1]
    assert recipe["reference_argv"][-2:] == ["--layout", "2p2d"]
    assert {key: env[key] for key in ("PREFILL_1_PORT", "BOOTSTRAP_1_PORT", "DECODE_1_PORT")} == {
        "PREFILL_1_PORT": "37200", "BOOTSTRAP_1_PORT": "37220", "DECODE_1_PORT": "37210"}
    assert float(env["READY_TIMEOUT_S"]) == 900 and float(env["RUN_TIMEOUT_S"]) == 2320
    assert (recipe["worker_log_budget_mib"], recipe["observer_budget_mib"],
        recipe["raw_reference_capture_telemetry_qc_budget_mib"], recipe["persistent_total_budget_mib"]) == (64, 64, 64, 256)
    assert plan["output_budget_mib"] == {"worker_logs": 64, "gateway_planner": 32, "observer": 64,
        "raw_reference_telemetry_qc": 64, "reserve": 32, "total": 256}
    helper = "scripts/hexgent_real_trace_collectors/h20_reference_components.py"
    assert plan["source_file_sha256"][helper] == recipe["source_sha256"][helper] == module.sha256(ROOT / helper)
    assert plan["execution_scope"] == "startup_only"


@pytest.mark.parametrize("change", ["duplicate_gpu", "missing_port", "colliding_port"])
def test_v10_recipe_rejects_incomplete_or_overlapping_fixed_allocation(tmp_path, change):
    module = v10_window(tmp_path); config = v10_window_config(tmp_path)
    if change == "duplicate_gpu": config["runtime"]["gpu_uuids"][-1] = config["runtime"]["gpu_uuids"][0]
    elif change == "missing_port": config["runtime"]["ports"].pop("bootstrap_1")
    else: config["runtime"]["ports"]["decode_1"] = config["runtime"]["ports"]["decode"]
    with pytest.raises((ValueError, KeyError)):
        module.render_c1_runtime_recipe(config, now_unix_s=1200.0)


def test_v10_actual_start_ours_registers_both_prefills_bootstraps_and_decodes(monkeypatch, tmp_path):
    import types
    module = v10_window(tmp_path); config = v10_window_config(tmp_path)
    transfer = tmp_path / "transfer.json"
    transfer.write_text(json.dumps({"HEXGENT_TRANSFER_STARTUP_MS": 5.4,
        "HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S": 2379754520.5}))
    config["runtime"]["transfer_parameters_path"] = str(transfer)
    ops = object.__new__(module.C1RuntimeOperations)
    ops.runtime = config["runtime"]; ops.ours_entries = []; ops.capture_status = None
    ops.plan = {"run_id": RUN, "frozen_metadata": {"transfer_parameters_sha256": module.sha256(transfer)}}
    ops.budget = module.WindowBudget(anchor_unix_s=module.time.time())
    recorded = []
    class Children:
        def healthy(self): return True
        def start(self, name, argv, env, path, **kwargs):
            recorded.append((name, list(argv), dict(env)))
            if name == "ours-sidecar":
                Path(argv[argv.index("--capture-status-path") + 1]).write_text(json.dumps({"observation_complete": True}))
            return {"name": name, "process": types.SimpleNamespace(pid=123, poll=lambda: None)}
    ops.children = Children()
    async def get(url, deadline): return {"data": [{"id": MODEL}]}
    monkeypatch.setattr(module, "async_get_json", get)
    monkeypatch.setattr(ops, "verify_ours_frontends", lambda: {"owned_frontends": True})
    # Real frozen method_commands; only external process/HTTP boundaries are replaced.
    asyncio.run(ops.start_ours(tmp_path))
    argv = next(row[1] for row in recorded if row[0] == "ours-gateway")
    assert [(argv[i+1], argv[i+2]) for i, flag in enumerate(argv) if flag == "--prefill"] == [
        ("http://127.0.0.1:37100", "37120"), ("http://127.0.0.1:37200", "37220")]
    assert [argv[i+1] for i, flag in enumerate(argv) if flag == "--decode"] == [
        "http://127.0.0.1:37110", "http://127.0.0.1:37210"]
    env = next(row[2] for row in recorded if row[0] == "ours-gateway")
    assert env["HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH"] == "1"
    assert env["HEXGENT_PROJECTED_RATIO_ORDER"] == "high_first"
    assert env["AGENTIC_PD_OURS_PLACEMENT_STRATEGY"] == "projected_finish"


def v10_registration_fixture(module, config):
    instances = v10_window_instances(config)
    roots = {"prefill": 10, "prefill_1": 40, "decode": 20, "decode_1": 50, "baseline_gateway": 30}
    facts = [{"pid": pid, "ppid": 1, "pgrp": pid, "start_identity": str(pid+100),
        "start_identity_after": str(pid+100), "pid_namespace": "pid:[42]", "net_namespace": "net:[43]",
        "task_ids": [pid], "fd_socket_inodes": []} for pid in roots.values()]
    sockets, records = [], []
    for instance in instances:
        root = roots[instance["component"]]
        for pid in (root+1, root+2):
            tid = pid+1000
            facts.append({"pid": pid, "ppid": root, "pgrp": root, "start_identity": str(pid+100),
                "start_identity_after": str(pid+100), "pid_namespace": "pid:[42]", "net_namespace": "net:[43]",
                "task_ids": [pid, tid], "fd_socket_inodes": [str(10000+pid), str(20000+pid)]})
            for kind, port, inode in (("session", 40000+pid, str(10000+pid)), ("tcp", 45000+pid, str(20000+pid))):
                line = (f"I1107 17:00:00.000001 {tid} transfer_engine_impl.cpp:182] Transfer Engine RPC using P2P handshake, listening on 192.0.2.1:{port}"
                    if kind == "session" else f"I1107 17:00:00.000002 {tid} tcp_transport.cpp:553] TcpTransport: listen on port {port}")
                records.append({"stream": instance["component"], "offset": len(records)*200, "line": line})
                sockets.append({"pid": pid, "start_identity": str(pid+100), "socket_inode": inode,
                    "local_ip": "192.0.2.1", "local_port": port, "remote_ip": "0.0.0.0", "remote_port": 0,
                    "state": "LISTEN", "family": "AF_INET"})
    return {"run_id": RUN, "owner_id": OWNER, "budget": module.WindowBudget(anchor_unix_s=1000),
        "launch_status": {"phase": "ready", "processes": [
            {"name": name, "pid": pid, "pgid": pid, "returncode": None} for name, pid in roots.items()]},
        "process_facts": facts, "socket_facts": sockets, "endpoint_records": records,
        "pid_namespace": "pid:[42]", "net_namespace": "net:[43]", "cgroup_path": "/sys/fs/cgroup",
        "source_hashes": {name: module.sha256(ROOT/name) for name in module.CONFIG_SOURCE_PATHS},
        "log_paths": ["/synthetic/"+row["component"]+".log" for row in instances],
        "pd_topology": "2p2d_tp2", "worker_instances": instances}


def test_v10_registration_retains_four_instances_and_two_rpc_two_data_each(tmp_path):
    module = v10_window(tmp_path); fixture = v10_registration_fixture(module, v10_window_config(tmp_path))
    result = module.build_observer_registration(**fixture)
    assert result["pd_topology"] == "2p2d_tp2" and result["worker_instances"] == fixture["worker_instances"]
    for instance in fixture["worker_instances"]:
        bindings = [row for row in result["endpoint_binding_evidence"] if row["instance_id"] == instance["instance_id"]]
        assert [row["peer"]["kind"] for row in bindings].count("session") == 2
        assert [row["peer"]["kind"] for row in bindings].count("tcp") == 2
        assert all(row["role"] == instance["role"] and row["worker_url"] == instance["worker_url"] for row in bindings)
    assert len(result["endpoint_binding_evidence"]) == 16
    module._observer_api().TransportObserver(result)  # Actual T2 validator, no reader.



def test_v10_actual_collector_attributes_provisional_roots_before_eight_property_queries(monkeypatch, tmp_path):
    import ipaddress
    module = v10_window(tmp_path); config = v10_window_config(tmp_path)
    fixture = v10_registration_fixture(module, config); instances = fixture["worker_instances"]
    facts = {row["pid"]: row for row in fixture["process_facts"]}; sockets = copy.deepcopy(fixture["socket_facts"])
    for instance in instances:
        root = next(row["pid"] for row in fixture["launch_status"]["processes"] if row["name"] == instance["component"])
        for port in [instance["http_port"]] + ([instance["bootstrap_port"]] if instance["role"] == "prefill" else []):
            inode = str(30000+port); facts[root]["fd_socket_inodes"].append(inode)
            sockets.append({"pid": root, "start_identity": facts[root]["start_identity"], "socket_inode": inode,
                "local_ip": "127.0.0.1", "local_port": port, "state": "LISTEN", "family": "AF_INET"})
    facts[30]["fd_socket_inodes"].append("99999")
    sockets.append({"pid": 30, "start_identity": "130", "socket_inode": "99999", "local_ip": "127.0.0.1",
        "local_port": config["runtime"]["ports"]["native"], "state": "LISTEN", "family": "AF_INET"})
    files = {"/proc/sys/net/ipv4/ip_local_port_range": "32768 60999\n", "/proc/sys/net/ipv4/ip_local_reserved_ports": "\n",
        "/sys/fs/cgroup/memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n", "/sys/fs/cgroup/memory.peak": "1048576\n"}
    for pid, row in facts.items():
        fields = ["S"]+["0"]*19; fields[1], fields[2], fields[19] = str(row["ppid"]), str(row["pgrp"]), row["start_identity"]
        files[f"/proc/{pid}/stat"] = str(pid)+" (owned worker) "+" ".join(fields)
        files[f"/proc/{pid}/task/{pid}/children"] = " ".join(str(item["pid"]) for item in facts.values() if item["ppid"] == pid)
        files[f"/proc/{pid}/limits"] = "Max open files 1024 4096 files\n"
        files[f"/proc/{pid}/environ"] = "SGLANG_MOONCAKE_TRANSFER_PROTOCOL=tcp\0"
        for tid in row["task_ids"]:
            task = ["S"]+["0"]*19; task[19] = str(tid+500)
            files[f"/proc/{pid}/task/{tid}/stat"] = str(tid)+" (owned task) "+" ".join(task)
    header = "sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
    ipv4, ipv6 = [header], [header]
    for index, row in enumerate(sockets):
        if 45000 <= row["local_port"] < 45100:
            ipv6.append(f"{index}: {'0'*32}:{row['local_port']:04X} {'0'*32}:0000 0A 00000000:00000000 00:00000000 00000000 1002 0 {row['socket_inode']} 1\n")
        else:
            address = ipaddress.IPv4Address(row["local_ip"]).packed[::-1].hex().upper()
            ipv4.append(f"{index}: {address}:{row['local_port']:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1002 0 {row['socket_inode']} 1\n")
    files["/proc/net/tcp"], files["/proc/net/tcp6"] = "".join(ipv4), "".join(ipv6)
    for instance in instances:
        files[str(tmp_path/"workers"/(instance["component"]+".log"))] = "\n".join(
            row["line"] for row in fixture["endpoint_records"] if row["stream"] == instance["component"])
    tool = {"path": "/usr/bin/ss", "sha256": "31124eb6385718f551751a580c7e17203bc307efa207ab5315b66b4ad5793e8c",
        "version": "ss utility, iproute2-6.1.0"}
    class Reader:
        def read_text(self, path):
            assert str(path) in files, "unexpected fixture read: "+str(path)
            return files[str(path)]
        def readlink(self, path):
            if str(path).endswith("/ns/pid"): return "pid:[42]"
            if str(path).endswith("/ns/net"): return "net:[43]"
            raise AssertionError("unexpected fixture namespace")
        def list_fd_targets(self, pid, proc_root="/proc"):
            return ["socket:["+inode+"]" for inode in facts[pid]["fd_socket_inodes"]]
        def file_metadata(self, path): return {"path": path, "sha256": tool["sha256"], "is_regular": True}
    original_iterdir = Path.iterdir
    def tasks(path):
        if str(path).startswith("/proc/"):
            assert path.name == "task"
            return iter(path/str(tid) for tid in facts[int(path.parts[2])]["task_ids"])
        return original_iterdir(path)
    monkeypatch.setattr(Path, "iterdir", tasks)
    api = module._observer_api(); queries, provisional = [], []
    actual_producer = api.LinuxIPv4FactProducer
    class ObservedProducer(actual_producer):
        def __init__(self, registration, **kwargs):
            provisional.append(copy.deepcopy(registration))
            super().__init__(registration, **kwargs)  # Strict T2 validator is not mocked.
    def query(argv, *, timeout_s, max_capture_bytes):
        queries.append(list(argv))
        if argv[-1] == "-V": raw = (tool["version"]+"\n").encode()
        else:
            port = int(argv[-1].split(":")[1].split()[0]); row = next(row for row in sockets if row["local_port"] == port)
            raw = f"LISTEN 0 4096 *:{port} *:* ino:{row['socket_inode']} sk:abcd v6only:0\n".encode()
        return {"stdout": raw, "stderr": b"", "returncode": 0, "timed_out": False,
            "truncated": False, "drain_finished": True, "capture_complete": True, "capture_errors": []}
    monkeypatch.setattr(api, "LinuxIPv4FactProducer", ObservedProducer)
    monkeypatch.setattr(api, "_ss_query", query); monkeypatch.setattr(module, "_observer_api", lambda: api)
    now = module.time.time()
    plan = {"run_id": RUN, "window_anchor_unix_s": now-1, "hard_deadline_unix_s": now+2699,
        "cleanup_reserve_s": 180, "source_file_sha256": fixture["source_hashes"]}
    config["runtime"].update(run_dir=str(tmp_path), source_dir=str(ROOT), ss_tool=tool)
    result = module.collect_startup_registration(config, plan, OWNER, fixture["launch_status"], supervisor_pid=1, reader=Reader())
    assert provisional and provisional[0]["pd_topology"] == "2p2d_tp2"
    model_rows = [row for row in provisional[0]["processes"] if row.get("stream_role") != "baseline_gateway"]
    assert model_rows and all(key in row for row in model_rows for key in ("instance_id", "role", "worker_url"))
    assert len(queries) == 9 and queries[0] == [tool["path"], "-V"]
    assert len(result["endpoint_binding_evidence"]) == 16 and len(result["configured_listeners"]) == 7
    assert {row["stream"] for row in result["endpoint_binding_evidence"]} == {"prefill", "prefill_1", "decode", "decode_1"}
    api.TransportObserver(result)


def test_v10_four_worker_idle_polls_fourth_busy_worker_and_derives_role_limits(tmp_path):
    module = v10_window(tmp_path); config = v10_window_config(tmp_path); instances = v10_window_instances(config)
    calls, sleeps = [], []
    async def get(url, deadline):
        calls.append(url); busy = url.startswith(instances[3]["worker_url"]) and len(calls) <= 4
        return {"snapshots": [{"limits": {"max_running_requests": 16},
            "running": [{"request_id": "D1-busy"}] if busy else [], "queues": {"waiting": []}}]}
    async def sleep(delay): sleeps.append(delay)
    result = asyncio.run(module.wait_idle_async(tmp_path/"idle4.json", config["runtime"]["expected_worker_limits"],
        module.time.time()+30, worker_urls={row["component"]: row["worker_url"] for row in instances},
        worker_instances=instances, getter=get, sleep=sleep))
    assert result["success"] and result["poll_count"] == 3
    assert calls == [row["worker_url"]+"/hexgent/pd_snapshot" for row in instances]*3
    assert sleeps == [0.25, 0.25] and set(result["final"]) == {"prefill", "prefill_1", "decode", "decode_1"}


def test_v10_frontend_only_proof_clears_model_registration_and_preserves_saved_main(monkeypatch, tmp_path):
    import types
    module = v10_window(tmp_path); api = module._observer_api(); config = v10_window_config(tmp_path)
    reg = {"run_id": RUN, "owner_id": OWNER, "deadline_unix_s": module.time.time()+120,
        "pid_namespace": "pid:[42]", "net_namespace": "net:[43]", "cgroup_path": "/fake",
        "source_hashes": {name: "7"*64 for name in module.CONFIG_SOURCE_PATHS},
        "processes": [{"pid": 11, "start_identity": "111"}], "peers": [],
        "pd_topology": "2p2d_tp2", "worker_instances": v10_window_instances(config),
        "endpoint_binding_evidence": [{"instance_id": "P0"}], "listener_properties": [{"pid": 11}]}
    path = tmp_path/"observer-registration.json"; original = json.dumps(reg).encode(); path.write_bytes(original)
    facts = [{"pid": pid, "ppid": 1, "pgrp": pid, "start_identity": str(pid+100), "start_identity_after": str(pid+100),
        "pid_namespace": "pid:[42]", "net_namespace": "net:[43]", "task_ids": [pid], "fd_socket_inodes": [str(pid+1000)]}
        for pid in (50, 60)]
    seen = []
    class Producer:
        def __init__(self, registration, **kwargs):
            seen.append(registration)
            api._worker_registration(registration)  # Actual strict metadata validator.
            assert not any(key in registration for key in ("pd_topology", "worker_instances", "endpoint_binding_evidence"))
        def collect(self, now):
            return {"read_errors": [], "source_status": {key: "ok" for key in api.REQUIRED_SOURCES},
                "sockets": [{"pid": pid, "start_identity": str(pid+100), "socket_inode": str(pid+1000),
                    "state": "LISTEN", "local_ip": "127.0.0.1", "local_port": port}
                    for pid, port in ((50, 37140), (60, 37150))]}
    monkeypatch.setattr(api, "LinuxProcReader", lambda: object()); monkeypatch.setattr(api, "LinuxIPv4FactProducer", Producer)
    monkeypatch.setattr(module, "_observer_api", lambda: api)
    monkeypatch.setattr(module, "_read_owned_processes", lambda *args, **kwargs: (facts, "pid:[42]", "net:[43]"))
    ops = object.__new__(module.C1RuntimeOperations); ops.root = tmp_path; ops.runtime = config["runtime"]
    ops.ours_entries = [{"process": types.SimpleNamespace(pid=pid, poll=lambda: None)} for pid in (50, 60)]
    assert set(ops.verify_ours_frontends()) == {"sidecar_http", "ours_http"}
    assert seen and path.read_bytes() == original



def test_v10_actual_planner_admission_keeps_two_worker_lists_and_existing_pair_locks(monkeypatch, tmp_path):
    # Actual Python admission/planner/core boundary, without sidecar HTTP or Rust.
    existing = load_path("_v10_existing_planner_fixture", ROOT/"test/unit/utils/test_hexgent_planner.py")
    planner_module, schemas = existing._load("planner"), existing._load("schemas")
    monkeypatch.setenv("HEXGENT_REVISION_STATE_PATH", str(tmp_path/"revision"))
    planner = planner_module.HexgentPlanner(); config = v10_window_config(tmp_path)
    instances = v10_window_instances(config)
    workers = {role: [schemas.WorkerDescriptor(id=row["instance_id"], url=row["worker_url"], worker_type=role,
        capacity_tokens=10000, prefill_ms_per_input_token=1 if row["instance_id"] == "P0" else 100,
        decode_ms_per_output_token=1 if row["instance_id"] == "D1" else 100)
        for row in instances if row["role"] == role] for role in ("prefill", "decode")}
    item = {"request_id": "locked-request", "workflow_id": "locked-wf", "call_id": "locked",
        "input_tokens": 10, "max_new_tokens": 2, "planned_decode_instance": "D0",
        "estimated_prefill_time": 10, "estimated_decode_time": 20, "schedule_revision": BIG_REVISION}
    snapshots = {row["worker_url"]: {"mode": row["role"], "running": [], "queues": {"waiting": []}} for row in instances}
    snapshots[instances[1]["worker_url"]]["queues"]["waiting"] = [item]
    request = schemas.PlannerScheduleRequest(route="/v1/chat/completions", model=MODEL, request_text="fixed request",
        max_tokens=2, hexgent=schemas.HexgentRequestMetadata(workflow_id="new-wf", call_id="new", input_tokens=10),
        prefill_workers=workers["prefill"], decode_workers=workers["decode"],
        state=schemas.PlannerClusterState(current_time_ms=100, worker_snapshots=snapshots))
    observed, decisions = [], []; original = planner.scheduler.schedule_prefill
    def schedule(calls, stage):
        observed.extend(calls); result = original(calls, stage); decisions.extend(result); return result
    monkeypatch.setattr(planner.scheduler, "schedule_prefill", schedule)
    planner.enqueue_admission(schemas.PlannerAdmissionRequest(request=request))
    result = planner.plan_pending_admissions()
    assert [row.url for row in request.prefill_workers] == [row["worker_url"] for row in instances if row["role"] == "prefill"]
    assert [row.url for row in request.decode_workers] == [row["worker_url"] for row in instances if row["role"] == "decode"]
    locked = next(call for call in observed if call.call_id == "locked")
    assert locked.locked_prefill_instance == 1 and locked.locked_decode_instance == 0
    selected = next(row for row in decisions if row.call_id == "locked")
    assert selected.assigned_instance == 1 and selected.planned_decode_instance == 0
    assert len(locked.prefill_ms_by_instance) == len(locked.decode_ms_by_instance) == 2
    assert result[("new-wf", "new")].schedule.prefill_worker_url in {row.url for row in request.prefill_workers}
    assert result[("new-wf", "new")].schedule.decode_worker_url in {row.url for row in request.decode_workers}


def v10_assignment_rows(items, instances, revision=0):
    rows = []
    for index, (wf, call) in enumerate((wf, call) for wf in items for call in wf.calls):
        p, d = ((0, 2), (0, 3), (1, 2), (1, 3))[index % 4]
        start = 1000 if not call.parents else 1010
        rows.append({"workflow_id": wf.workflow_id, "call_id": call.call_id, "parents": list(call.parents),
            "workflow_arrival_ms": 1000, "request_started_at_ms": start, "response_completed_at_ms": start+10,
            "input_tokens": call.input_len, "output_tokens": call.output_len, "error": None,
            "prefill_worker": instances[p]["worker_url"], "decode_worker": instances[d]["worker_url"],
            "schedule_revision": revision+index if revision else 0,
            "response": {"model": MODEL, "usage": {"prompt_tokens": call.input_len, "completion_tokens": call.output_len}}})
    return rows


def v10_method_ops(module, monkeypatch, tmp_path):
    import types
    config = v10_window_config(tmp_path); ops = minimal_method_ops(module, tmp_path)
    ops.runtime.update(config["runtime"]); ops.expected = config["runtime"]["expected_worker_limits"]
    ops.endpoint = module.C1RuntimeOperations.endpoint.__get__(ops)
    ops.worker_sampler_byte_budget = {"saved_bytes": 0}
    async def idle(*args, **kwargs): return {"success": True}
    async def sampler(path, instances, **kwargs):
        path.write_text(json.dumps(v10_sample_record(instances))+"\n")
        await asyncio.Event().wait()
    async def stop(*args, **kwargs): return None
    async def start_ours(directory):
        ops.ours_entries = [{"name": "ours-sidecar"}]
        ops.capture_status = directory/"capture-status.latest.json"
        ops.capture_status.write_text(json.dumps({"observation_complete": True}))
        (directory/"identity-capture.jsonl").write_text("")
    ops.children = types.SimpleNamespace(stop=stop); ops.start_ours = start_ours
    monkeypatch.setattr(module, "_idle", idle)
    monkeypatch.setattr(module, "_sample_phase_workers", sampler, raising=False)
    monkeypatch.setattr(module._pilot("run_paired_pilot"), "cpu_snapshot", lambda: {})
    return ops, v10_window_instances(config)



@pytest.mark.parametrize("method", ["native", "ours"])
@pytest.mark.parametrize("change", ["role_crossed", "foreign", "missing"])
def test_v10_calibration_ownership_precedes_native_or_unjoined_revision_zero_shortcuts(tmp_path, method, change):
    module = v10_window(tmp_path); instances = v10_window_instances(v10_window_config(tmp_path))
    items = workflows(); rows = v10_assignment_rows(items, instances)
    if change == "role_crossed": rows[0]["prefill_worker"] = instances[2]["worker_url"]
    elif change == "foreign": rows[0]["decode_worker"] = "http://foreign:9999"
    else: rows[0].pop("prefill_worker")
    result = module.calibration_gate(method, items, rows, model=MODEL, run_namespace=NAMESPACE, worker_instances=instances)
    assert result["successful"] is False and result["assignment_ownership_valid"] is False


def test_v10_exact_four_pair_join_requires_same_raw_plan_dispatch_urls_and_bigint_revision(tmp_path):
    module = v10_window(tmp_path); instances = v10_window_instances(v10_window_config(tmp_path))
    items = workflows(ids=("pair00", "pair01", "pair10", "pair11"))
    rows = v10_assignment_rows(items, instances, BIG_REVISION); records = capture_records(items)
    pairs = {(NAMESPACE+"::"+row["workflow_id"], NAMESPACE+"::"+row["call_id"]): row for row in rows}
    for record in records:
        if record["path"] == "/event":
            body = record["request"]; row = pairs[(body["workflow_id"], body["call_id"])]
            body["payload"].update(prefill_worker_url=row["prefill_worker"], decode_worker_url=row["decode_worker"])
        elif record["path"] == "/wait_plan_revision":
            for entry in record["response"]["entries"]:
                row = pairs[(entry["workflow_id"], entry["call_id"])]
                entry["schedule"].update(prefill_worker_url=row["prefill_worker"], decode_worker_url=row["decode_worker"])
    good = module.calibration_gate("ours", items, rows, model=MODEL, run_namespace=NAMESPACE,
        capture_records=records, worker_instances=instances)
    assert good["successful"] and good["canonical_joined_calls"] == 4
    wrong = copy.deepcopy(records)
    for record in wrong:
        if record["path"] == "/event":
            payload = record["request"]["payload"]
            payload["decode_worker_url"] = instances[3]["worker_url"] if payload["decode_worker_url"] == instances[2]["worker_url"] else instances[2]["worker_url"]
    bad = module.calibration_gate("ours", items, rows, model=MODEL, run_namespace=NAMESPACE,
        capture_records=wrong, worker_instances=instances)
    assert bad["canonical_joined_calls"] == 0 and bad["successful"] is False


@pytest.mark.parametrize("method", ["native", "ours"])
@pytest.mark.parametrize("change", ["role_crossed", "foreign", "missing"])
def test_v10_formal_actual_method_does_not_replace_bad_ownership_with_usage_validity(monkeypatch, tmp_path, method, change):
    module = v10_window(tmp_path); ops, instances = v10_method_ops(module, monkeypatch, tmp_path); items = workflows()
    async def replay(api, selected, **kwargs):
        rows = v10_assignment_rows(selected, instances)
        if selected[0].workflow_type != "uniform_warmup":
            if change == "role_crossed": rows[0]["decode_worker"] = instances[1]["worker_url"]
            elif change == "foreign": rows[0]["prefill_worker"] = "http://foreign:9999"
            else: rows[0].pop("decode_worker")
        kwargs["raw_path"].write_text("".join(json.dumps(row)+"\n" for row in rows)); return rows
    monkeypatch.setattr(module, "replay_workflows", replay)
    async def auxiliary(*args): return True
    monkeypatch.setattr(ops, "formal_auxiliary", auxiliary)
    workers = tmp_path/"workers"; workers.mkdir(); (workers/"status.json").write_text("{}")
    for row in instances: (workers/(row["component"]+".log")).write_text("")
    try:
        gate = asyncio.run(ops.method_phase("formal-07-hotpotqa-burst-"+method, method, items, calibration=False))
    except module.WindowStopped:
        gate = json.loads((tmp_path/("formal-07-hotpotqa-burst-"+method)/"measurement-gate.json").read_text())
    assert gate["successful"] is False and gate["formal_status"] == "FAILED"
    assert gate["assignment_ownership_valid"] is False


def test_v10_p1_d1_warmup_and_four_selected_pair_counts_are_actual_request_rows(monkeypatch, tmp_path):
    module = v10_window(tmp_path); ops, instances = v10_method_ops(module, monkeypatch, tmp_path)
    items = workflows(ids=("pair00", "pair01", "pair10", "pair11"))
    async def replay(api, selected, **kwargs):
        rows = v10_assignment_rows(selected, instances)
        if selected[0].workflow_type == "uniform_warmup":
            rows[0].update(prefill_worker=instances[1]["worker_url"], decode_worker=instances[3]["worker_url"])
        kwargs["raw_path"].write_text("".join(json.dumps(row)+"\n" for row in rows)); return rows
    monkeypatch.setattr(module, "replay_workflows", replay)
    gate = asyncio.run(ops.method_phase("calibration-native", "native", items, calibration=True))
    assert gate["successful"] is True
    summary = gate["assignment_summary"]
    assert summary["selected_pair_call_counts"] == {"P0/D0": 1, "P0/D1": 1, "P1/D0": 1, "P1/D1": 1}
    assert summary["assigned_calls"] == 4 and summary["unreturned_calls"] == 0 and summary["unknown_assignment_calls"] == 0


def test_v10_pair_summary_separates_unknown_and_unreturned_from_four_pair_counts(tmp_path):
    module = v10_window(tmp_path); instances = v10_window_instances(v10_window_config(tmp_path))
    items = workflows(ids=("pair00", "pair01", "pair10", "missing"))
    rows = v10_assignment_rows(items, instances)[:3]; rows[1]["prefill_worker"] = None
    summary = feature(module, "_assignment_summary")(items, rows, instances)
    assert summary["selected_pair_call_counts"] == {"P0/D0": 1, "P0/D1": 0, "P1/D0": 1, "P1/D1": 0}
    assert summary["assigned_calls"] == 2 and summary["unknown_assignment_calls"] == 1 and summary["unreturned_calls"] == 1


@pytest.mark.parametrize("change", ["role_crossed", "foreign", "missing"])
def test_v10_complete_standalone_reference_rejects_bad_assignment(tmp_path, change):
    module = v10_window(tmp_path); config = v10_window_config(tmp_path); instances = v10_window_instances(config)
    items = workflows(ids=module.HOT_IDS, fanout=True)
    async def idle(*args, **kwargs): return {"success": True}
    async def replay(api, selected, **kwargs):
        rows = v10_assignment_rows(selected, instances)
        if change == "role_crossed": rows[0]["prefill_worker"] = instances[2]["worker_url"]
        elif change == "foreign": rows[0]["decode_worker"] = "http://foreign:9999"
        else: rows[0].pop("decode_worker")
        kwargs["raw_path"].write_text("".join(json.dumps(row)+"\n" for row in rows)); return rows
    with pytest.raises(module.WindowStopped, match="ownership|assignment"):
        asyncio.run(module.collect_hot_references(workflow_api(), items, output_dir=tmp_path/"refs4",
            router_url="synthetic://never-contact", model=MODEL, implementation_revision=REVISION,
            runtime_evidence_sha256=EPOCH, run_namespace=RUN+"::refs", expected_limits=config["runtime"]["expected_worker_limits"],
            budget=module.WindowBudget(anchor_unix_s=1000), clock=lambda: 1100, replay_to_raw=replay,
            idle_checker=idle, worker_instances=instances, worker_urls={row["component"]: row["worker_url"] for row in instances}))
    table = tmp_path/"refs4"/"workflow-references.jsonl"
    assert not table.exists() or not any(json.loads(line).get("status") == "successful" for line in table.read_text().splitlines())



def test_v10_w2_actual_auxiliary_retains_four_logs_with_new_eight_mib_shared_cap(monkeypatch, tmp_path):
    module = v10_window(tmp_path); config = v10_window_config(tmp_path); instances = v10_window_instances(config)
    ops = object.__new__(module.C1RuntimeOperations)
    ops.root = tmp_path; ops.runtime = config["runtime"]; ops.w2_saved_bytes = 0
    ops.budget = module.WindowBudget(anchor_unix_s=module.time.time())
    directory = tmp_path/"formal"; directory.mkdir(); workers = tmp_path/"workers"; workers.mkdir()
    (workers/"status.json").write_text("{}")
    text = ("I synthetic bounded diagnostic "+("x"*1000)+"\n")*1280
    for row in instances: (workers/(row["component"]+".log")).write_text(text)
    runner = module._pilot("run_paired_pilot")
    monkeypatch.setattr(runner, "cpu_snapshot", lambda: {})
    monkeypatch.setattr(runner, "cgroup_cpu_delta", lambda *args: {})
    monkeypatch.setattr(runner, "worker_observation_complete", lambda *args: True)
    async def no_sleep(delay): return None
    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)
    assert asyncio.run(ops.formal_auxiliary(directory, {"cpu": {}, "status": {},
        "offsets": {row["component"]: 0 for row in instances}})) is True
    report = json.loads((directory/"worker-mutation-summary.json").read_text())
    assert {Path(path).name for path in report["input_logs"]} == {row["component"]+".worker-slice.log" for row in instances}
    provenance = json.loads((directory/"formal-observation-provenance.json").read_text())
    assert set(provenance["worker_log_offsets_end"]) == {row["component"] for row in instances}
    assert 4*1024**2 < ops.w2_saved_bytes < 8*1024**2


@pytest.mark.parametrize("dp_ranks", [(None,), (0, 1)])
def test_v10_sampler_preserves_rank_identity_sequential_times_and_observed_counts(tmp_path, dp_ranks):
    module = v10_window(tmp_path); sample = feature(module, "_sample_phase_workers")
    config = v10_window_config(tmp_path); instances = v10_window_instances(config)
    clock, calls, delays, byte_budget = [1100.0], [], [], {"saved_bytes": 0}
    async def getter(url, deadline):
        calls.append((url, deadline, clock[0])); clock[0] += .01
        instance = next(row for row in instances if url.startswith(row["worker_url"]+"/"))
        netloc = instance["worker_url"].split("://",1)[1]
        return {"dp_rank_count": len(dp_ranks), "snapshots": [
            {"instance_id": netloc+("" if rank is None else "/dp"+str(rank)), "dp_rank": rank,
             "timestamp": clock[0], "mode": instance["role"], "limits": {"max_running_requests": 16},
             "running": [{"request_id": "one", "prompt": "NEVER_EXPORT"}],
             "queues": {"waiting": [{"request_id": "two", "prompt": "NEVER_EXPORT"}] if index == 0 else []}}
            for index,rank in enumerate(dp_ranks)]}
    async def sleep(delay): delays.append(delay); raise asyncio.CancelledError
    path = tmp_path/"pressure.jsonl"
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(sample(path, instances, expected_by_role=config["runtime"]["expected_worker_limits"],
            deadline_unix_s=1200, byte_budget=byte_budget, getter=getter, sleep=sleep, clock=lambda: clock[0]))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1 and delays == [1.0]
    assert [row[0] for row in calls] == [row["worker_url"]+"/hexgent/pd_snapshot" for row in instances]
    assert all(row[1] == 1200 for row in calls)
    record = records[0]
    assert record["observation_complete"] and record["sampling_scope"] == "sequential_round_not_simultaneous"
    assert set(record["workers"]) == {row["component"] for row in instances}
    for instance in instances:
        actual = record["workers"][instance["component"]]; netloc=instance["worker_url"].split("://",1)[1]
        assert (actual["instance_id"], actual["role"], actual["worker_url"]) == (instance["instance_id"], instance["role"], instance["worker_url"])
        assert actual["dp_rank_count"] == len(dp_ranks)
        assert [(row["instance_id"], row["dp_rank"], row["running_count"]) for row in actual["snapshots"]] == [
            (netloc+("" if rank is None else "/dp"+str(rank)),rank,1) for rank in dp_ranks]
        assert actual["snapshots"][0]["queue_counts"] == {"waiting": 1}
    assert "NEVER_EXPORT" not in path.read_text()
    assert byte_budget["saved_bytes"] == path.stat().st_size <= 8*1024**2



@pytest.mark.parametrize("failure", ["get", "cap", "write", "partial_write"])
def test_v10_sampler_failure_keeps_valid_prefix_and_shared_budget(monkeypatch, tmp_path, failure):
    module = v10_window(tmp_path); sample = feature(module, "_sample_phase_workers")
    config = v10_window_config(tmp_path); instances = v10_window_instances(config); path = tmp_path/"pressure.jsonl"
    initial = 8*1024**2-1 if failure == "cap" else 0
    budget, calls, rounds = {"saved_bytes": initial}, [], []
    async def getter(url, deadline):
        calls.append(url)
        if failure == "get" and len(calls) == 8: raise OSError("synthetic fourth-worker capture unavailable")
        instance = next(row for row in instances if url.startswith(row["worker_url"]+"/"))
        return {"dp_rank_count": 1, "snapshots": [{"instance_id": instance["worker_url"].split("://",1)[1],
            "dp_rank": None, "timestamp": 1100, "mode": instance["role"], "limits": {"max_running_requests": 16},
            "running": [], "queues": {"waiting": []}}]}
    if failure in ("write", "partial_write"):
        original = Path.open
        class Partial:
            def __init__(self, target): self.target = target
            def __enter__(self): self.target.__enter__(); return self
            def __exit__(self, *args): return self.target.__exit__(*args)
            def __getattr__(self, key): return getattr(self.target, key)
            def write(self, data):
                self.target.write(data[:max(1,len(data)//2)]); self.target.flush()
                raise OSError("synthetic partial output write")
        def altered(self, mode="r", *args, **kwargs):
            if self == path and any(flag in mode for flag in "wax"):
                if failure == "write": raise OSError("synthetic output denied")
                return Partial(original(self, mode, *args, **kwargs))
            return original(self, mode, *args, **kwargs)
        monkeypatch.setattr(Path, "open", altered)
    async def sleep(delay):
        rounds.append(delay)
        assert failure == "get" and len(rounds) == 1, "failed sampler retried"
    with pytest.raises((module.WindowStopped, OSError)):
        asyncio.run(sample(path, instances, expected_by_role=config["runtime"]["expected_worker_limits"],
            deadline_unix_s=1200, byte_budget=budget, getter=getter, sleep=sleep, clock=lambda: 1100))
    if path.exists():
        assert path.stat().st_size <= 8*1024**2
        for line in path.read_text().splitlines(): json.loads(line)
        if failure == "get": assert path.stat().st_size > 0
    assert budget["saved_bytes"] >= initial


def test_v10_sampler_expired_original_deadline_never_issues_get(tmp_path):
    module = v10_window(tmp_path); sample = feature(module, "_sample_phase_workers")
    config = v10_window_config(tmp_path); calls = []
    async def get(*args): calls.append(args); pytest.fail("expired sampler issued HTTP")
    with pytest.raises(module.WindowStopped):
        asyncio.run(sample(tmp_path/"pressure.jsonl", v10_window_instances(config),
            expected_by_role=config["runtime"]["expected_worker_limits"], deadline_unix_s=1100,
            byte_budget={"saved_bytes": 0}, getter=get, clock=lambda: 1100))
    assert not calls



def v10_sample_record(instances):
    # One actual non-DP backend record is valid for a TP2 model instance.
    return {"sampled_unix_s": 1100, "observation_complete": True,
        "sampling_scope": "sequential_round_not_simultaneous", "workers": {
            row["component"]: {"instance_id": row["instance_id"], "role": row["role"], "worker_url": row["worker_url"],
                "dp_rank_count": 1, "snapshots": [{"instance_id": row["worker_url"].split("://",1)[1], "dp_rank": None,
                    "timestamp": 1100, "running_count": 1, "queue_counts": {"waiting": 0}}]} for row in instances}}



@pytest.mark.parametrize("failure", ["none", "active_error", "simultaneous_error"])
def test_v10_actual_method_owns_sampler_start_cancel_and_completed_error(monkeypatch, tmp_path, failure):
    module = v10_window(tmp_path); ops, instances = v10_method_ops(module, monkeypatch, tmp_path)
    items = workflows(); events = []; sampler_started = asyncio.Event(); replay_started = asyncio.Event(); sampler_failed = asyncio.Event()
    async def sampler(path, selected, **kwargs):
        events.append("sampler-start"); sampler_started.set()
        path.write_text(json.dumps(v10_sample_record(selected))+"\n")
        try:
            if failure != "none":
                await replay_started.wait(); sampler_failed.set()
                raise module.WindowStopped("sampler capture incomplete")
            await asyncio.Event().wait()
        finally: events.append("sampler-finally")
    async def replay(api, selected, **kwargs):
        rows = v10_assignment_rows(selected, instances)
        if selected[0].workflow_type == "uniform_warmup":
            kwargs["raw_path"].write_text("".join(json.dumps(row)+"\n" for row in rows)); return rows
        assert sampler_started.is_set(), "replay entered before its phase sampler"
        events.append("replay-start")
        kwargs["raw_path"].write_text(json.dumps(rows[0])+"\n"); replay_started.set()
        if failure == "active_error":
            try: await asyncio.Event().wait()
            finally: events.append("replay-finally")
        elif failure == "simultaneous_error":
            await sampler_failed.wait()
        kwargs["raw_path"].write_text("".join(json.dumps(row)+"\n" for row in rows)); return rows
    monkeypatch.setattr(module, "_sample_phase_workers", sampler, raising=False)
    monkeypatch.setattr(module, "replay_workflows", replay)
    async def invoke():
        return await asyncio.wait_for(ops.method_phase("calibration-native", "native", items, calibration=True), 1.0)
    if failure != "none":
        with pytest.raises(module.WindowStopped, match="sampler"): asyncio.run(invoke())
        gate = json.loads((tmp_path/"calibration-native"/"measurement-gate.json").read_text())
        assert gate["successful"] is False
        assert (tmp_path/"calibration-native"/"replay.raw.jsonl").read_text().strip()
        if failure == "active_error": assert "replay-finally" in events
    else:
        gate = asyncio.run(invoke()); assert gate["successful"] is True
        evidence = gate["worker_queue_observation"]
        assert evidence["scope"] == "sampled_maxima_not_true_peak"
        assert evidence["max_observed_running_by_instance"] == {"P0": 1, "P1": 1, "D0": 1, "D1": 1}
        assert evidence["capture_gaps"] == []  # Backend DP records are not synthetic TP-rank samples.
    assert events[0] == "sampler-start" and "sampler-finally" in events
    assert events.count("replay-start") == 1


def test_v10_actual_phase_reuses_one_sampler_byte_budget_across_methods(monkeypatch, tmp_path):
    module = v10_window(tmp_path); ops, instances = v10_method_ops(module, monkeypatch, tmp_path)
    budgets = []
    async def sampler(path, selected, **kwargs):
        budgets.append(kwargs["byte_budget"])
        kwargs["byte_budget"]["saved_bytes"] += 100
        path.write_text(json.dumps(v10_sample_record(selected))+"\n")
        await asyncio.Event().wait()
    async def replay(api, selected, **kwargs):
        rows = v10_assignment_rows(selected, instances)
        kwargs["raw_path"].write_text("".join(json.dumps(row)+"\n" for row in rows)); return rows
    monkeypatch.setattr(module, "_sample_phase_workers", sampler, raising=False)
    monkeypatch.setattr(module, "replay_workflows", replay)
    async def invoke():
        for method in ("native", "ours"):
            gate = await asyncio.wait_for(ops.method_phase("calibration-"+method, method, workflows(), calibration=True), 1.0)
            # Ours with no canonical capture is intentionally not activated.
            assert gate["assignment_ownership_valid"] is True
    asyncio.run(invoke())
    assert len(budgets) == 2 and budgets[0] is budgets[1] is ops.worker_sampler_byte_budget
    assert ops.worker_sampler_byte_budget["saved_bytes"] == 200


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_v10_startup_only_retires_all_four_owned_worker_entries_before_observer(monkeypatch, tmp_path, cleanup_error):
    import types
    module = v10_window(tmp_path); monkeypatch.setattr(module.time, "time", lambda: 1100)
    ops = object.__new__(module.C1RuntimeOperations)
    ops.window = new_window(module); ops.window.register_owned_container(OWNER)
    ops.budget = ops.window.budget; ops.approval = approval()
    ops.root = tmp_path; ops.owner_id = OWNER; ops.plan = {"run_id": RUN, "execution_scope": "startup_only", "input_sha256": {}}
    ops.summary = lambda: observer(); ops.healthy = lambda: True
    rows, calls = [], []
    for index, name in enumerate(("prefill", "prefill_1", "decode", "decode_1", "baseline_gateway", "transport-observer")):
        state = {"code": None}
        process = types.SimpleNamespace(pid=500+index, poll=lambda state=state: state["code"])
        rows.append({"name": name, "process": process, "pgid": process.pid,
            "thread": types.SimpleNamespace(is_alive=lambda: False), "truncated": False, "write_error": None, "state": state})
    async def startup(): return None
    async def stop(deadline, selected=None):
        calls.append((deadline, [row["name"] for row in selected]))
        for row in selected: row["state"]["code"] = -15
        if cleanup_error and len(calls) == 1: raise module.WindowStopped("one owned nonobserver retirement failed")
    ops.startup = startup; ops.children = types.SimpleNamespace(entries=rows, stop=stop)
    result = asyncio.run(ops.execute_startup_only())
    assert result == (2 if cleanup_error else 0)
    assert calls == [(3700, ["prefill", "prefill_1", "decode", "decode_1", "baseline_gateway"]), (3700, ["transport-observer"])]



def test_v10_full_primary_pipeline_is_unready_before_startup_without_h100_reference_contract(tmp_path):
    import types
    module = v10_window(tmp_path); ops = object.__new__(module.C1RuntimeOperations)
    ops.plan = {"run_id": RUN}; ops.runtime = {"pd_topology": "2p2d_tp2"}
    ops.root = tmp_path; ops.phase_records = []
    ops.budget = module.WindowBudget(anchor_unix_s=module.time.time())
    calls = []
    async def startup():
        calls.append("startup")
        raise module.WindowStopped("fixture reached startup before primary reference guard")
    async def stop(*args, **kwargs): return None
    ops.startup = startup; ops.children = types.SimpleNamespace(entries=[], stop=stop)
    with pytest.raises(module.WindowStopped, match="H100|primary reference"):
        asyncio.run(ops.execute())
    assert calls == []



@pytest.mark.parametrize("invalid_extra", ["none", "duplicate", "unexpected"])
def test_v10_pair_counts_use_unique_expected_calls_and_check_ownership_on_every_raw_row(tmp_path, invalid_extra):
    module = v10_window(tmp_path); instances = v10_window_instances(v10_window_config(tmp_path))
    items = workflows(ids=("pair00", "pair01", "pair10", "pair11"))
    rows = v10_assignment_rows(items, instances)
    duplicate = copy.deepcopy(rows[0]); unexpected = copy.deepcopy(rows[1])
    unexpected["workflow_id"] = "foreign-logical-workflow"
    if invalid_extra == "duplicate": duplicate["prefill_worker"] = instances[2]["worker_url"]
    elif invalid_extra == "unexpected": unexpected["decode_worker"] = instances[0]["worker_url"]
    rows += [duplicate, unexpected]
    summary = module._assignment_summary(items, rows, instances)
    assert summary["selected_pair_call_counts"] == {"P0/D0": 1, "P0/D1": 1, "P1/D0": 1, "P1/D1": 1}
    assert summary["assigned_calls"] == 4
    assert summary["duplicate_raw_rows"] == 1 and summary["unexpected_raw_rows"] == 1
    assert summary["unknown_assignment_calls"] == 0 and summary["unreturned_calls"] == 0
    assert summary["assignment_ownership_valid"] is (invalid_extra == "none")
    gate = module.calibration_gate("native", items, rows, model=MODEL, run_namespace=NAMESPACE, worker_instances=instances)
    assert gate["successful"] is False  # Duplicate/foreign logical rows never form a valid measurement.


def test_v10_legacy_plan_and_recipe_pin_the_shared_helper_used_by_actual_launcher(tmp_path):
    module = v10_window(tmp_path); config = runtime_config(tmp_path)
    plan = module.prepare_plan(config, startup_only=True)
    recipe = module.render_c1_runtime_recipe(config, now_unix_s=1200, startup_only=True)
    helper = "scripts/hexgent_real_trace_collectors/h20_reference_components.py"
    assert plan["source_file_sha256"].get(helper) == module.sha256(ROOT/helper)
    assert recipe["source_sha256"].get(helper) == module.sha256(ROOT/helper)
    assert recipe["reference_argv"] == ["bash", "/src/scripts/hexgent_real_trace_collectors/start_h20_tp2_reference.sh"]
    assert recipe["persistent_total_budget_mib"] == 128 and recipe["worker_log_budget_mib"] == 32
    assert recipe["docker_argv"][recipe["docker_argv"].index("--cpus")+1] == "6"



@pytest.mark.parametrize("failure", ["borrowed_d0_at_d1", "missing_dp_rank", "missing_instance_id", "wrong_dp_suffix"])
def test_v10_sampler_rejects_unbound_or_missing_actual_backend_dp_identity(tmp_path, failure):
    module = v10_window(tmp_path); instances = v10_window_instances(v10_window_config(tmp_path))
    path = tmp_path/"pressure.jsonl"
    async def getter(url, deadline):
        instance = next(row for row in instances if url.startswith(row["worker_url"]+"/"))
        snapshot = {"instance_id": instance["worker_url"].split("://",1)[1], "dp_rank": None,
            "timestamp": 1100, "mode": instance["role"], "limits": {"max_running_requests": 16},
            "running": [], "queues": {"waiting": []}}
        if instance["instance_id"] == "D1":
            if failure == "borrowed_d0_at_d1": snapshot["instance_id"] = instances[2]["worker_url"].split("://",1)[1]
            elif failure == "missing_dp_rank": snapshot.pop("dp_rank")
            elif failure == "missing_instance_id": snapshot.pop("instance_id")
            else:
                snapshot["dp_rank"] = 1
                snapshot["instance_id"] += "/dp0"
        return {"dp_rank_count": 1, "snapshots": [snapshot]}
    async def forbidden_pause(delay): pytest.fail("unbound backend identity reached a complete sampler round")
    with pytest.raises(module.WindowStopped):
        asyncio.run(module._sample_phase_workers(path, instances,
            expected_by_role={"prefill": {"max_running_requests": 16}, "decode": {"max_running_requests": 16}},
            deadline_unix_s=1200, byte_budget={"saved_bytes": 0},
            getter=getter, sleep=forbidden_pause, clock=lambda: 1100))
    assert not path.exists() or not path.read_text().strip()
