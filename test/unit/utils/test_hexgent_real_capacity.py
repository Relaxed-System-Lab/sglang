from test_hexgent_planner import _complete_live_snapshots
"""Real-wrapper C1 capacity admission and fixed-pair regressions (CPU only)."""

import importlib.util
import math
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


@pytest.fixture(autouse=True)
def policy_environment(monkeypatch):
    monkeypatch.setenv("AGENTIC_PD_OURS_PLACEMENT_STRATEGY", "projected_finish")
    monkeypatch.setenv("HEXGENT_PROJECTED_RATIO_ORDER", "high_first")
    monkeypatch.setenv("HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH", "1")


def _prefill_call(core, *, n=2, demand=1000, locked=None, short=False):
    call = core.PrefillCallInfo(
        "c", "wf", 128 if short else 500, 64 if short else 500,
        0, 0, 5000, 0, 0, 1000, [100] * n,
        [200] * (n // 2) + [100] * (n - n // 2), [0] * n,
        locked_decode_instance=locked,
    )
    call.decode_demand_tokens = demand
    return call


@pytest.mark.parametrize("stage", ["prefill", "decode"])
@pytest.mark.parametrize("strategy", ["projected_finish", "weighted"])
@pytest.mark.parametrize("n,short", [(2, False), (6, True), (16, True)])
def test_core_uses_the_only_feasible_destination(stage, strategy, n, short, monkeypatch):
    monkeypatch.setenv("AGENTIC_PD_OURS_PLACEMENT_STRATEGY", strategy)
    core = _load("scheduler")
    capacities = [100] * (n - 1) + [1000]
    call = _prefill_call(core, n=n, short=short)
    if stage == "prefill":
        state = core.PrefillStageState(0, [0] * n, [0] * n)
        state.decode_capacity_max_tokens = capacities
    else:
        call = core.DecodeCallInfo(
            "c", "wf", call.input_tokens, call.output_tokens, 0, 0, 5000,
            0, 0, 1000, 0, 1000, call.decode_ms_by_instance, None,
        )
        state = core.DecodeStageState(0, capacities, [[] for _ in capacities])

    decision = getattr(core.AgenticPDExactScheduler(), f"schedule_{stage}")([call], state)[0]

    destination = decision.planned_decode_instance if stage == "prefill" else decision.assigned_instance
    assert destination == n - 1


@pytest.mark.parametrize("locked,capacities", [(None, [100, 999]), (0, [100, 1000])])
def test_core_prefill_all_infeasible_or_infeasible_lock_stays_waiting(locked, capacities):
    core = _load("scheduler")
    state = core.PrefillStageState(0, [0, 0], [0, 0])
    state.decode_capacity_max_tokens = capacities

    assert core.AgenticPDExactScheduler().schedule_prefill(
        [_prefill_call(core, locked=locked)], state
    ) == []


def test_core_preserves_an_existing_feasible_pair():
    core = _load("scheduler")
    call = _prefill_call(core, locked=0)
    call.locked_prefill_instance = 0
    state = core.PrefillStageState(0, [0, 0], [0, 0])
    state.decode_capacity_max_tokens = [1000, 1000]

    decision = core.AgenticPDExactScheduler().schedule_prefill([call], state)[0]

    assert (decision.assigned_instance, decision.planned_decode_instance) == (0, 0)


@pytest.mark.parametrize("locked,capacities", [(None, [100, 999]), (0, [100, 1000])])
def test_core_decode_all_infeasible_or_infeasible_lock_stays_waiting(locked, capacities):
    core = _load("scheduler")
    call = core.DecodeCallInfo(
        "c", "wf", 500, 500, 0, 0, 5000, 0, 0, 1000,
        0, 1000, [200, 100], locked,
    )

    assert core.AgenticPDExactScheduler().schedule_decode(
        [call], core.DecodeStageState(0, capacities, [[], []])
    ) == []


def test_core_decode_preserves_a_feasible_existing_lock():
    core = _load("scheduler")
    call = core.DecodeCallInfo(
        "c", "wf", 500, 500, 0, 0, 5000, 0, 0, 1000,
        0, 1000, [200, 100], 0,
    )

    decision = core.AgenticPDExactScheduler().schedule_decode(
        [call], core.DecodeStageState(0, [1000, 1000], [[], []])
    )[0]

    assert decision.assigned_instance == 0


@pytest.mark.parametrize("intervals", [[], [(0, 101, 400)]])
def test_core_rejects_an_intrinsically_oversized_interval(intervals):
    core = _load("scheduler")

    assert math.isinf(core.find_earliest_capacity_start(
        ready_at_ms=0, duration_ms=1, demand_tokens=1001,
        scheduled_intervals=intervals, capacity_tokens=1000,
    ))


def _planner_request(capacities, *, predicted=500, max_tokens=500, snapshots=None):
    planner_mod = _load("planner")
    schemas = _load("schemas")
    request = _complete_live_snapshots(schemas.PlannerScheduleRequest(
        route="/v1/chat/completions", model="cpu-real-capacity-audit",
        request_text="known reservation demand", max_tokens=max_tokens,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf", call_id="new", input_tokens=500,
            predicted_output_tokens=predicted,
        ),
        prefill_workers=[schemas.WorkerDescriptor(
            id="p0", url="http://p0", worker_type="prefill",
        )],
        decode_workers=[schemas.WorkerDescriptor(
            id=f"d{index}", url=f"http://d{index}", worker_type="decode",
            capacity_tokens=capacity,
            decode_ms_per_output_token=1 + index,
        ) for index, capacity in enumerate(capacities)],
        state=schemas.PlannerClusterState(current_time_ms=0, worker_snapshots=snapshots or {}),
    ))
    return planner_mod.HexgentPlanner(), schemas, request


@pytest.mark.parametrize("endpoint", ["direct", "admission"])
def test_real_wrapper_chooses_the_only_feasible_decode(endpoint):
    planner, schemas, request = _planner_request([100, 1000])

    if endpoint == "direct":
        schedule = planner.schedule_request(request)
    else:
        planner.enqueue_admission(schemas.PlannerAdmissionRequest(request=request))
        schedule = planner.plan_pending_admissions()[("wf", "new")].schedule

    assert schedule.decode_worker_id == "d1"
    assert planner._reservations[("wf", "new")].decode_demand_tokens == 1000


@pytest.mark.parametrize("predicted,max_tokens", [(500, 20), (None, 500)])
def test_real_wrapper_passes_only_known_predicted_or_max_demand(predicted, max_tokens, monkeypatch):
    planner, schemas, request = _planner_request([100, 1000], predicted=predicted, max_tokens=max_tokens)
    observed = []
    original = planner.scheduler.schedule_prefill

    def capture(calls, state):
        observed.append((getattr(calls[0], "decode_demand_tokens", None),
                         getattr(state, "decode_capacity_max_tokens", None)))
        return original(calls, state)

    monkeypatch.setattr(planner.scheduler, "schedule_prefill", capture)
    planner.enqueue_admission(schemas.PlannerAdmissionRequest(request=request))
    planner.plan_pending_admissions()

    assert observed == [(1000, [100, 1000])]


def test_real_wrapper_leaves_all_infeasible_admissions_pending_without_reservation():
    planner, schemas, request = _planner_request([100, 999])
    planner.enqueue_admission(schemas.PlannerAdmissionRequest(request=request))

    assert planner.plan_pending_admissions() == {}
    assert not planner.admission_status(request).dispatch_ready
    assert ("wf", "new") not in planner._reservations


def test_real_direct_wrapper_reports_an_infeasible_request():
    planner, _, request = _planner_request([100, 999])

    with pytest.raises(ValueError, match="capacity-feasible decode"):
        planner.schedule_request(request)


def test_real_wrapper_keeps_an_infeasible_dispatched_pair_without_migration():
    snapshots = {"http://p0": {"snapshot": {
        "mode": "prefill", "queues": {"waiting": [{
            "rid": "rid-locked", "call_id": "locked", "workflow_id": "old-wf",
            "input_tokens": 500, "predicted_output_tokens": 500,
            "planned_decode_instance": "d0",
        }]},
    }}}
    planner, schemas, request = _planner_request([100, 1000], snapshots=snapshots)
    planner.enqueue_admission(schemas.PlannerAdmissionRequest(request=request))

    schedule = planner.plan_pending_admissions()[("wf", "new")].schedule

    assert schedule.decode_worker_id == "d1"
    assert not any(action.request_id == "rid-locked" for action in schedule.actions)


@pytest.mark.parametrize("demand,expected_start", [(300, 0), (700, 101), (1001, math.inf)])
def test_c1_with_total_snapshot_capacity_releases_or_rejects_correctly(demand, expected_start):
    snapshots = {"http://d0": {"snapshot": {
        "mode": "decode", "limits": {"max_total_num_tokens": 1000},
        "queues": {},
        "running": [{"memory_tokens": 400, "remaining_service_tokens": 10}],
    }}}
    _, _, request = _planner_request([1000], snapshots=snapshots)
    request.decode_workers[0].decode_base_ms = 1
    request.decode_workers[0].decode_ms_per_output_token = 10
    _load("snapshots").apply_worker_snapshots(request, include_mutable_waiting=False)
    core = _load("scheduler")

    assert core.find_earliest_capacity_start(
        ready_at_ms=0, duration_ms=1, demand_tokens=demand,
        scheduled_intervals=request.state.decode_scheduled_intervals[0],
        capacity_tokens=request.decode_workers[0].capacity_tokens,
    ) == expected_start
