"""Paired public-input decisions against frozen source232 (CPU only)."""
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from test_hexgent_planner import _load

FROZEN = Path(os.environ.get("HEXGENT_SIM232_SOURCE",
    "/mnt/data/fred/hexagent-rebuttal/worktrees/rebuttal-priority-snapshot-consistency-v1"))


def frozen_core():
    name = "hexgent_frozen232"
    package = ModuleType(name)
    package.__path__ = [str(FROZEN / "pd_disaggregated")]
    sys.modules[name] = package
    return importlib.import_module(name + ".sglang_scheduler_core")


def profile(module):
    core = module.AgenticPDExactScheduler()
    core.decode_admission_projection = True
    core.prefill_decode_admission_projection = True
    core.known_priority_insertion_projection = True
    core.known_priority_insertion_backlog_gate = True
    core.refinement_projection = "faithful"
    core.pairwise_refinement = False
    core.recompute_large_ratio_batches = True
    return core


def decode_call(module, name="new", *, times=(10, 100), demand=400):
    return module.DecodeCallInfo(call_id=name, workflow_id="wf-" + name,
        input_tokens=200, output_tokens=200, workflow_arrival_ms=0, reveal_sequence=0,
        workflow_horizon_ms=100, workflow_completion_ms=99, workflow_attained_service_ms=0,
        ideal_service_ms=1, ready_at_ms=0, decode_demand_tokens=demand,
        decode_ms_by_instance=list(times), locked_decode_instance=None)


def signature(rows):
    return [(x.call_id, x.assigned_instance, x.priority_score) for x in rows]


def decode(module, calls, *, intervals=None, slots=(1, 1)):
    core = profile(module)
    state = module.DecodeStageState(0, [1000, 1000], intervals or [[], []])
    state.max_batch_size_by_instance = list(slots)
    for call in calls:
        core.on_workflow_arrival(call.workflow_id, call.workflow_arrival_ms, call.workflow_horizon_ms)
    return core.schedule_decode(calls, state)


def test_hard_decode_slot_limit_changes_the_real_choice():
    def decide(module):
        return decode(module, [decode_call(module)], intervals=[[(0, 1000, 100)], []])
    expected = decide(frozen_core())
    assert expected[0].assigned_instance == 1  # D0 memory fits, but its only slot is occupied.
    assert signature(decide(_load("scheduler"))) == signature(expected)


def test_two_decode_slots_allow_real_parallel_admission():
    def decide(module):
        return decode(module, [decode_call(module)], intervals=[[(0, 1000, 400)], []], slots=(2, 1))
    expected = decide(frozen_core())
    assert expected[0].assigned_instance == 0  # 400+400 <=1000 and one of two slots is free.
    assert signature(decide(_load("scheduler"))) == signature(expected)


def test_float_priority_risk_is_preserved_before_integer_worker_ranks():
    def decide(module):
        calls = [decode_call(module, name, times=(1, 1), demand=10) for name in ("a", "b")]
        for call, horizon in zip(calls, (.1004, .1001)):
            call.priority_seconds = SimpleNamespace(current_time_s=0., workflow_arrival_s=0.,
                workflow_horizon_s=horizon, workflow_completion_s=.0999,
                prefill_s_by_instance=(), decode_s_by_instance=(.0004, .0004),
                transfer_s_by_decode_instance=())
        return decode(module, calls, slots=(8, 8))
    expected = decide(frozen_core())
    assert expected[0].call_id == "b"  # Integer-ms inputs tie; continuous risk does not.
    assert signature(decide(_load("scheduler"))) == signature(expected)


def test_default_live_planner_selects_the_frozen_stable_profile():
    core = _load("planner").HexgentPlanner().scheduler
    for name in ("decode_admission_projection", "prefill_decode_admission_projection",
                 "known_priority_insertion_projection", "known_priority_insertion_backlog_gate",
                 "recompute_large_ratio_batches"):
        assert getattr(core, name, False) is True, name
    assert getattr(core, "pairwise_refinement", True) is False
