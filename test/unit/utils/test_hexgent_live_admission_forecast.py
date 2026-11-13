"""CPU regressions for the live snapshot to canonical admission forecast."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[3] / "python" / "sglang"


def _load(module_name):
    for name, path in (("sglang", _PKG_ROOT), ("sglang.hexgent", _PKG_ROOT / "hexgent")):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    name = f"sglang.hexgent.{module_name}"
    spec = importlib.util.spec_from_file_location(name, _PKG_ROOT / "hexgent" / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules():
    schemas = _load("schemas")
    scheduler = _load("scheduler")
    snapshots = _load("snapshots")
    return schemas, scheduler, snapshots


def _item(call="old", workflow="wf", **extra):
    return dict(rid=f"{workflow}/{call}", call_id=call, workflow_id=workflow,
                input_tokens=300, max_new_tokens=100, remaining_service_tokens=50,
                output_tokens_generated=50, memory_tokens=350,
                estimated_decode_time=1000.0, **extra)


def _request(schemas, *, running=(), queues=None, limits=None):
    return schemas.PlannerScheduleRequest(
        route="/v1/chat/completions", model="cpu-forecast", request_text="observed work",
        max_tokens=100, hexgent=schemas.HexgentRequestMetadata(workflow_id="new-wf", call_id="new"),
        prefill_workers=[schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")],
        decode_workers=[schemas.WorkerDescriptor(id="d0", url="http://d0", worker_type="decode")],
        state=schemas.PlannerClusterState(current_time_ms=100, worker_snapshots={"http://d0": {"dp_rank_count": 1, "snapshots": [{
            "mode": "decode", "dp_rank": 0,
            "limits": {"max_total_num_tokens": 1000, "max_running_requests": 2} if limits is None else limits,
            "queues": {} if queues is None else queues, "running": list(running),
        }]}}),
    )


def _call(scheduler, call="old", workflow="wf", *, ready=100, duration=500):
    return scheduler.DecodeCallInfo(
        call_id=call, workflow_id=workflow, input_tokens=300, output_tokens=100,
        workflow_arrival_ms=0, reveal_sequence=0, workflow_horizon_ms=1000,
        workflow_completion_ms=0, workflow_attained_service_ms=0,
        ideal_service_ms=duration, ready_at_ms=ready, decode_demand_tokens=400,
        decode_ms_by_instance=[duration], locked_decode_instance=0,
    )


def _build(snapshots, request, **kwargs):
    assert hasattr(snapshots, "build_live_stage_states"), "missing live admission stage builder"
    return snapshots.build_live_stage_states(request, **kwargs)


def test_running_sequences_have_independent_frozen_intervals(modules):
    schemas, _, snapshots = modules
    request = _request(schemas, running=[_item("a"), _item("b")])
    frozen_raw = copy.deepcopy(request.state.worker_snapshots)
    snapshots.apply_worker_snapshots(request, include_mutable_waiting=False)
    assert len(request.state.decode_scheduled_intervals[0]) == 2
    assert [item[2] for item in request.state.decode_scheduled_intervals[0]] == [400, 400]
    assert request.state.worker_snapshots == frozen_raw


def test_partial_running_work_retains_full_context_demand(modules):
    schemas, _, snapshots = modules
    request = _request(schemas, running=[_item()])
    snapshots.apply_worker_snapshots(request, include_mutable_waiting=False)
    assert request.state.decode_scheduled_intervals == [[(100, 600, 400)]]
    p_state, d_state = _build(snapshots, request)
    assert p_state.decode_admission_intervals == [[(100, 600, 400)]]
    assert d_state.scheduled_intervals == [[(100, 600, 400)]]
    assert p_state.decode_max_batch_size_by_instance == [2]
    assert d_state.max_batch_size_by_instance == [2]
    assert p_state.decode_priority_calls == ()
    assert p_state.decode_priority_job_ids == ((),)
    assert d_state.capacity_max_tokens == [1000]


def test_snapshot_metadata_covers_all_decode_states_but_only_waiting_is_mutable(modules):
    schemas, _, snapshots = modules
    queues = {name: [_item(name)] for name in ("prealloc", "transfer", "waiting", "retracted")}
    request = _request(schemas, running=[_item("running")], queues=queues)
    assert hasattr(snapshots, "decode_snapshot_calls"), "missing complete decode queue metadata"
    records = snapshots.decode_snapshot_calls(request)
    assert {record.queue_name for record in records} == {
        "decode_prealloc", "decode_transfer", "decode_waiting", "decode_retracted", "decode_running"
    }
    assert [record.call_id for record in records if record.is_mutable_waiting] == ["waiting"]
    assert [record.call_id for record in records if record.is_running] == ["running"]


def test_all_queued_jobs_use_supplied_predictions_and_transfer_release(modules):
    schemas, scheduler, snapshots = modules
    queues = {name: [_item(name, estimated_transfer_time=25.0)]
              for name in ("prealloc", "transfer", "waiting", "retracted")}
    request = _request(schemas, queues=queues)
    calls = [_call(scheduler, name, duration=20+i) for i, name in enumerate(queues)]
    p_state, d_state = _build(snapshots, request, decode_calls=calls)
    assert p_state.decode_admission_jobs == [[(125, 20, 400), (125, 21, 400), (100, 22, 400), (100, 23, 400)]]
    assert len(p_state.decode_priority_job_ids[0]) == 4
    assert len(p_state.decode_priority_calls) == 4
    assert d_state.scheduled_intervals == [[]]


def test_queued_calls_keep_global_identity_when_local_call_ids_repeat(modules):
    schemas, scheduler, snapshots = modules
    request = _request(schemas, queues={"waiting": [_item("same", "wf-a"), _item("same", "wf-b")]})
    calls = [_call(scheduler, "same", "wf-a", duration=20), _call(scheduler, "same", "wf-b", duration=30)]
    p_state, _ = _build(snapshots, request, decode_calls=calls)
    ids = p_state.decode_priority_job_ids[0]
    assert len(set(ids)) == 2
    assert {tuple(json.loads(value)) for value in ids} == {("wf-a", "same"), ("wf-b", "same")}
    assert [call.call_id for call in p_state.decode_priority_calls] == ["same", "same"]
    assert [call.call_id for call in calls] == ["same", "same"]


@pytest.mark.parametrize("limits", [{"max_running_requests": 2}, {"max_total_num_tokens": 1000}])
def test_missing_live_capacity_or_slots_is_reported(modules, limits):
    schemas, _, snapshots = modules
    request = _request(schemas, limits=limits)
    with pytest.raises(ValueError, match="max_total_num_tokens|max_running_requests"):
        _build(snapshots, request)


def test_independent_dp_replicas_are_not_collapsed_into_one_worker(modules):
    schemas, _, snapshots = modules
    request = _request(schemas)
    payload = request.state.worker_snapshots["http://d0"]
    payload["dp_rank_count"] = 2
    payload["snapshots"].append(copy.deepcopy(payload["snapshots"][0]))
    payload["snapshots"][1]["dp_rank"] = 1
    with pytest.raises(ValueError, match="DP|replica|rank"):
        _build(snapshots, request)


def test_prefix_probe_round_trips_without_changing_legacy_metadata(modules):
    schemas, _, _ = modules
    request = _request(schemas)
    payload = schemas.to_jsonable(request)
    probe = {"probe_id": "wf/new", "input_ids": [11, 12, 13], "extra_key": "salt-lora", "max_prefix_tokens": 2}
    payload["prefix_cache_probe"] = probe
    restored = schemas.PlannerScheduleRequest.from_dict(payload)
    assert getattr(restored, "prefix_cache_probe", None) == probe
    assert restored.hexgent.input_tokens is None


def test_prefix_probe_fetch_posts_only_exact_probe_fields_to_prefill(modules, monkeypatch):
    schemas, _, snapshots = modules
    request = _request(schemas)
    probe = {"probe_id": "wf/new", "input_ids": [11, 12, 13], "extra_key": "salt-lora", "max_prefix_tokens": 2}
    calls = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"snapshot":{"mode":"prefill"}}'

    def urlopen(req, *, timeout):
        calls.append((req.full_url, req.method, req.data))
        return Response()

    monkeypatch.setattr(snapshots.urllib.request, "urlopen", urlopen)
    snapshots.fetch_worker_snapshots([*request.prefill_workers, *request.decode_workers], timeout_s=0.1,
                                     prefix_cache_probe={**probe, "unrelated": "ignored"})
    assert [(url, method) for url, method, _ in calls] == [
        ("http://p0/hexgent/pd_snapshot", "POST"), ("http://d0/hexgent/pd_snapshot", "GET")]
    assert json.loads(calls[0][2]) == {"prefix_probes": [probe]}
    assert calls[1][2] is None


def test_live_scheduler_normalizes_repeated_ids_without_mutating_calls(modules):
    _, scheduler, _ = modules
    forecast_path = _PKG_ROOT / "hexgent" / "forecast.py"
    assert forecast_path.exists(), "missing workflow identity adapter"
    forecast = _load("forecast")
    calls = [_call(scheduler, "same", "wf-a", duration=20),
             _call(scheduler, "same", "wf-b", duration=40)]
    before = copy.deepcopy(calls)
    state = scheduler.DecodeStageState(100, [1000], [[]], [2], (100,))
    live = forecast.LiveStableScheduler(decode_admission_projection=True, pairwise_refinement=False)
    actual = live.schedule_decode(calls, state)
    canonical_calls = [copy.deepcopy(call) for call in calls]
    for call in canonical_calls:
        call.call_id = forecast.global_call_id(call.workflow_id, call.call_id)
    canonical = scheduler.AgenticPDExactScheduler(decode_admission_projection=True, pairwise_refinement=False)
    expected = canonical.schedule_decode(canonical_calls, state)
    assert [(d.workflow_id, d.call_id, d.assigned_instance, d.priority_score) for d in actual] == [
        (d.workflow_id, json.loads(d.call_id)[1], d.assigned_instance, d.priority_score) for d in expected]
    assert calls == before


def test_original_request_body_round_trips_for_exact_tokenization(modules):
    schemas, _, _ = modules
    payload = schemas.to_jsonable(_request(schemas))
    body = {"model": "cpu-forecast", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 100}
    payload["request_body"] = body
    restored = schemas.PlannerScheduleRequest.from_dict(payload)
    assert getattr(restored, "request_body", None) == body


def test_batch_prefix_probes_are_preserved_in_one_prefill_post(modules, monkeypatch):
    schemas, _, snapshots = modules
    request = _request(schemas)
    probes = [dict(probe_id=f"wf/{i}", input_ids=[i, i+1], extra_key=None, max_prefix_tokens=1)
              for i in (3, 8)]
    captured = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"snapshot":{"mode":"prefill"}}'

    def urlopen(req, *, timeout):
        captured.append((req.method, req.data))
        return Response()

    monkeypatch.setattr(snapshots.urllib.request, "urlopen", urlopen)
    snapshots.fetch_worker_snapshots(request.prefill_workers, timeout_s=0.1, prefix_probes=probes)
    assert len(captured) == 1
    assert captured[0][0] == "POST"
    assert json.loads(captured[0][1]) == {"prefix_probes": probes}


@pytest.mark.parametrize("running_slots,expected_decode", [(2, 0), (1, 1)])
def test_actual_free_decode_slot_changes_prefill_placement(modules, running_slots, expected_decode):
    schemas, scheduler, snapshots = modules
    request = _request(schemas, running=[_item()], limits={"max_total_num_tokens": 1000, "max_running_requests": running_slots})
    request.decode_workers.append(schemas.WorkerDescriptor(id="d1", url="http://d1", worker_type="decode"))
    request.state.worker_snapshots["http://d1"] = {"snapshot": {
        "mode": "decode", "limits": {"max_total_num_tokens": 1000, "max_running_requests": 2}, "running": [], "queues": {}}}
    candidate = scheduler.PrefillCallInfo(
        call_id="new", workflow_id="new-wf", input_tokens=300, output_tokens=100,
        workflow_arrival_ms=100, reveal_sequence=0, workflow_horizon_ms=1000,
        workflow_completion_ms=0, workflow_attained_service_ms=0, ideal_service_ms=11,
        prefill_ms_by_instance=[1], decode_ms_by_instance=[10, 30], transfer_ms_by_decode_instance=[0, 0],
        decode_demand_tokens=400,
    )
    p_state, _ = _build(snapshots, request)
    live = _load("forecast").LiveStableScheduler(
        prefill_decode_admission_projection=True, known_priority_insertion_projection=True,
        known_priority_insertion_backlog_gate=True, pairwise_refinement=False,
    )
    result = live.schedule_prefill([candidate], p_state)
    assert result[0].planned_decode_instance == expected_decode


def test_missing_decode_payload_in_partial_live_snapshot_is_reported(modules):
    schemas, _, snapshots = modules
    request = _request(schemas)
    request.decode_workers.append(schemas.WorkerDescriptor(id="d1", url="http://d1", worker_type="decode"))
    with pytest.raises(ValueError, match="d1.*snapshot|snapshot.*d1"):
        _build(snapshots, request)


def test_direct_calendar_legacy_callers_keep_explicit_intervals_and_limits(modules):
    schemas, _, snapshots = modules
    request = _request(schemas)
    request.state.worker_snapshots = {}
    request.decode_workers[0].capacity_tokens = 777
    request.state.decode_scheduled_intervals = [[schemas.ScheduledInterval(100, 130, 333)]]
    request.state.decode_max_batch_size_by_instance = [4]
    request.state.decode_admission_not_before_ms = [111]
    p_state, d_state = _build(snapshots, request)
    assert p_state.decode_admission_intervals == [[(100, 130, 333)]]
    assert p_state.decode_admission_not_before_ms == (111,)
    assert d_state.capacity_max_tokens == [777]
    assert d_state.max_batch_size_by_instance == [4]


def test_known_priority_backlog_and_incoming_repeated_ids_use_same_global_keys(modules):
    schemas, scheduler, snapshots = modules
    request = _request(schemas, queues={"waiting": [_item("same", "queued-wf")]})
    queued = _call(scheduler, "same", "queued-wf", duration=40)
    candidate = scheduler.PrefillCallInfo(
        call_id="same", workflow_id="new-wf", input_tokens=300, output_tokens=100,
        workflow_arrival_ms=100, reveal_sequence=0, workflow_horizon_ms=1000,
        workflow_completion_ms=0, workflow_attained_service_ms=0, ideal_service_ms=11,
        prefill_ms_by_instance=[1], decode_ms_by_instance=[10], transfer_ms_by_decode_instance=[0],
        decode_demand_tokens=400,
    )
    p_state, _ = _build(snapshots, request, decode_calls=[queued, candidate])
    original = copy.deepcopy((queued, candidate, p_state))
    live = _load("forecast").LiveStableScheduler(
        prefill_decode_admission_projection=True, known_priority_insertion_projection=True,
        known_priority_insertion_backlog_gate=True, pairwise_refinement=False,
    )
    result = live.schedule_prefill([candidate], p_state)
    assert [(d.workflow_id, d.call_id, d.planned_decode_instance) for d in result] == [("new-wf", "same", 0)]
    assert (queued, candidate, p_state) == original


def test_identity_adapter_preserves_canonical_mapping_call_container(modules):
    _, scheduler, _ = modules
    call = _call(scheduler)
    live = _load("forecast").LiveStableScheduler(decode_admission_projection=True, pairwise_refinement=False)
    decisions = live.schedule_decode({"wf/old": call}, scheduler.DecodeStageState(100, [1000], [[]], [2], (100,)))
    assert [(decision.workflow_id, decision.call_id) for decision in decisions] == [("wf", "old")]


def test_anonymous_running_health_check_still_occupies_decode_memory_and_slot(modules):
    schemas, _, snapshots = modules
    anonymous = _item()
    anonymous.pop("workflow_id")
    anonymous.pop("call_id")
    anonymous["rid"] = "HEALTH_CHECK"
    request = _request(schemas, running=[anonymous])
    records = snapshots.decode_snapshot_calls(request)
    assert len(records) == 1
    assert records[0].is_running
    assert records[0].workflow_id == records[0].call_id == ""
    p_state, d_state = _build(snapshots, request)
    assert p_state.decode_admission_intervals == [[(100, 600, 400)]]
    assert d_state.scheduled_intervals == [[(100, 600, 400)]]
    assert d_state.max_batch_size_by_instance == [2]
    assert p_state.decode_admission_jobs == [[]]
    assert p_state.decode_priority_calls == ()


def test_anonymous_queued_request_is_reported_instead_of_omitted(modules):
    schemas, _, snapshots = modules
    anonymous = _item()
    anonymous.pop("workflow_id")
    anonymous.pop("call_id")
    request = _request(schemas, queues={"waiting": [anonymous]})
    with pytest.raises(ValueError, match="identity"):
        _build(snapshots, request)
