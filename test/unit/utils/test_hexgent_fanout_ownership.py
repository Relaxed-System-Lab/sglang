"""CPU-only real-core regressions for accepted C2 fanout ownership."""

import importlib.util
import itertools
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def core():
    path = Path(__file__).resolve().parents[3] / "python/sglang/hexgent/scheduler.py"
    name = "_hexgent_real_fanout_ownership_core"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def fixed_policy(monkeypatch):
    monkeypatch.setenv("AGENTIC_PD_OURS_PLACEMENT_STRATEGY", "projected_finish")
    monkeypatch.setenv("HEXGENT_PROJECTED_RATIO_ORDER", "high_first")
    monkeypatch.setenv("HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH", "1")


def call(core, stage, call_id, workflow_id, parents=(), child_count=0):
    fields = dict(
        call_id=call_id, workflow_id=workflow_id, input_tokens=1000,
        output_tokens=100, workflow_arrival_ms=0, reveal_sequence=0,
        workflow_horizon_ms=1000, workflow_completion_ms=0,
        workflow_attained_service_ms=0, ideal_service_ms=300,
        decode_ms_by_instance=[200, 200], parent_call_ids=parents,
        child_count=child_count,
    )
    if stage == "prefill":
        return core.PrefillCallInfo(
            **fields, prefill_ms_by_instance=[100, 100],
            transfer_ms_by_decode_instance=[0, 0], decode_demand_tokens=1100,
        )
    return core.DecodeCallInfo(
        **fields, ready_at_ms=0, decode_demand_tokens=1100,
        locked_decode_instance=None,
    )


def schedule(core, scheduler, stage, calls):
    state = (
        core.PrefillStageState(0, [0, 0], [0, 0], [10000, 10000])
        if stage == "prefill"
        else core.DecodeStageState(0, [10000, 10000], [[], []])
    )
    return getattr(scheduler, "schedule_" + stage)(calls, state)


@pytest.mark.parametrize("stage", ["prefill", "decode"])
@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_mixed_batch_marks_only_its_sibling_owner(core, stage, order):
    scheduler = core.AgenticPDExactScheduler()
    calls = [
        call(core, stage, "branch-a", "tree", ("parent",)),
        call(core, stage, "branch-b", "tree", ("parent",)),
        call(core, stage, "chain-call", "chain", ("chain-parent",)),
    ]

    schedule(core, scheduler, stage, [calls[index] for index in order])

    assert scheduler._fanout_workflows == {"tree"}


@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_unrelated_chain_later_keeps_refinement(core, stage):
    scheduler = core.AgenticPDExactScheduler()
    schedule(core, scheduler, stage, [
        call(core, stage, "branch-a", "tree", ("parent",)),
        call(core, stage, "branch-b", "tree", ("parent",)),
        call(core, stage, "chain-first", "chain", ("chain-parent",)),
    ])
    before = getattr(scheduler.stats, stage + "_local_search_evaluations")

    schedule(core, scheduler, stage, [
        call(core, stage, "chain-later", "chain", ("chain-first",)),
        call(core, stage, "other-chain", "other", ("other-parent",)),
    ])

    assert getattr(scheduler.stats, stage + "_local_search_evaluations") - before == 1
    assert scheduler._fanout_workflows == {"tree"}


@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_count_and_child_hint_do_not_combine_across_workflows(core, stage):
    scheduler = core.AgenticPDExactScheduler()

    schedule(core, scheduler, stage, [
        call(core, stage, "a0", "two-calls", ("p0",)),
        call(core, stage, "a1", "two-calls", ("p1",)),
        call(core, stage, "b0", "one-call", ("p2",), child_count=2),
    ])

    assert scheduler._fanout_workflows == set()


@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_legacy_child_hint_marks_only_its_own_workflow(core, stage):
    scheduler = core.AgenticPDExactScheduler()

    schedule(core, scheduler, stage, [
        call(core, stage, "a0", "legacy-tree", ("p0",), child_count=2),
        call(core, stage, "a1", "legacy-tree", ("p1",)),
        call(core, stage, "chain-call", "chain", ("p2",)),
    ])

    assert scheduler._fanout_workflows == {"legacy-tree"}


@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_same_parent_text_in_distinct_workflows_is_not_siblings(core, stage):
    scheduler = core.AgenticPDExactScheduler()

    schedule(core, scheduler, stage, [
        call(core, stage, "a", "workflow-a", ("same-parent-text",)),
        call(core, stage, "b", "workflow-b", ("same-parent-text",)),
    ])

    assert scheduler._fanout_workflows == set()


@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_empty_parent_sets_do_not_imply_siblings(core, stage):
    scheduler = core.AgenticPDExactScheduler()

    schedule(core, scheduler, stage, [
        call(core, stage, "a", "workflow", ()),
        call(core, stage, "b", "workflow", ()),
    ])

    assert scheduler._fanout_workflows == set()


@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_mixed_batch_refinement_gate_is_preserved(core, stage):
    scheduler = core.AgenticPDExactScheduler()

    schedule(core, scheduler, stage, [
        call(core, stage, "a", "tree", ("parent",)),
        call(core, stage, "b", "tree", ("parent",)),
        call(core, stage, "chain-call", "chain", ("chain-parent",)),
    ])

    assert getattr(scheduler.stats, stage + "_local_search_evaluations") == 0
