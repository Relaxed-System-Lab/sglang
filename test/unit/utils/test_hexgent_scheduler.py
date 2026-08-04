import importlib.util
from pathlib import Path
import sys
import types


_ROOT = Path(__file__).resolve().parents[3]
_PKG_ROOT = _ROOT / "python"


def _load_scheduler():
    if "sglang" not in sys.modules:
        package = types.ModuleType("sglang")
        package.__path__ = [str(_PKG_ROOT / "sglang")]
        sys.modules["sglang"] = package
    module_path = _PKG_ROOT / "sglang" / "hexgent" / "scheduler.py"
    spec = importlib.util.spec_from_file_location("sglang.hexgent.scheduler", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _call(
    scheduler_mod,
    *,
    call_id,
    output_tokens,
    completion_ms,
    critical_path_ms,
    input_tokens=64,
    child_count=1,
    parent_call_ids=(),
    decode_ms=100,
):
    kwargs = {}
    if parent_call_ids:
        kwargs["parent_call_ids"] = parent_call_ids
    return scheduler_mod.PrefillCallInfo(
        call_id=call_id,
        workflow_id="wf",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        workflow_arrival_ms=0,
        reveal_sequence=0,
        workflow_horizon_ms=1000,
        workflow_completion_ms=completion_ms,
        workflow_attained_service_ms=0,
        ideal_service_ms=10,
        prefill_ms_by_instance=[10],
        decode_ms_by_instance=[decode_ms],
        transfer_ms_by_decode_instance=[0],
        critical_path_ms=critical_path_ms,
        child_count=child_count,
        **kwargs,
    )


def _schedule_prefill(scheduler_mod, calls):
    scheduler = scheduler_mod.AgenticPDExactScheduler()
    scheduler.on_workflow_arrival("wf", arrival_ms=0, horizon_ms=1000)
    return scheduler.schedule_prefill(
        calls,
        scheduler_mod.PrefillStageState(
            current_time_ms=0,
            instance_available_at_ms=[0],
            decode_available_hint_ms=[0],
        ),
    )


def _decode_call(
    scheduler_mod,
    *,
    call_id,
    output_tokens,
    completion_ms,
    critical_path_ms,
    input_tokens=64,
    child_count=1,
    parent_call_ids=(),
    decode_ms=100,
):
    kwargs = {}
    if parent_call_ids:
        kwargs["parent_call_ids"] = parent_call_ids
    return scheduler_mod.DecodeCallInfo(
        call_id=call_id,
        workflow_id="wf",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        workflow_arrival_ms=0,
        reveal_sequence=0,
        workflow_horizon_ms=1000,
        workflow_completion_ms=completion_ms,
        workflow_attained_service_ms=0,
        ideal_service_ms=10,
        ready_at_ms=0,
        decode_demand_tokens=1,
        decode_ms_by_instance=[decode_ms],
        locked_decode_instance=None,
        critical_path_ms=critical_path_ms,
        child_count=child_count,
        **kwargs,
    )


def _schedule_decode(scheduler_mod, calls):
    scheduler = scheduler_mod.AgenticPDExactScheduler()
    scheduler.on_workflow_arrival("wf", arrival_ms=0, horizon_ms=1000)
    return scheduler.schedule_decode(
        calls,
        scheduler_mod.DecodeStageState(
            current_time_ms=0,
            capacity_max_tokens=[10000],
            scheduled_intervals=[[]],
        ),
    )


def test_fanout_prefill_orders_by_critical_path_pressure():
    scheduler_mod = _load_scheduler()
    calls = [
        _call(
            scheduler_mod,
            call_id="critical",
            output_tokens=128,
            completion_ms=100,
            critical_path_ms=900,
        ),
        _call(
            scheduler_mod,
            call_id="projected",
            output_tokens=128,
            completion_ms=900,
            critical_path_ms=100,
        ),
    ]

    decisions = _schedule_prefill(scheduler_mod, calls)

    assert [decision.call_id for decision in decisions] == ["critical", "projected"]


def test_sequential_prefill_keeps_projected_ratio_ordering():
    scheduler_mod = _load_scheduler()
    calls = [
        _call(
            scheduler_mod,
            call_id="critical",
            output_tokens=100,
            completion_ms=100,
            critical_path_ms=900,
        ),
        _call(
            scheduler_mod,
            call_id="projected",
            output_tokens=100,
            completion_ms=900,
            critical_path_ms=100,
        ),
    ]

    decisions = _schedule_prefill(scheduler_mod, calls)

    assert [decision.call_id for decision in decisions] == ["projected", "critical"]


def test_prefill_rotates_exact_homogeneous_worker_ties():
    scheduler_mod = _load_scheduler()
    scheduler = scheduler_mod.AgenticPDExactScheduler()
    scheduler.on_workflow_arrival("wf", arrival_ms=0, horizon_ms=1000)

    placements = []
    for index in range(4):
        call = scheduler_mod.PrefillCallInfo(
            call_id=f"call-{index}",
            workflow_id="wf",
            input_tokens=64,
            output_tokens=64,
            workflow_arrival_ms=0,
            reveal_sequence=index,
            workflow_horizon_ms=1000,
            workflow_completion_ms=0,
            workflow_attained_service_ms=0,
            ideal_service_ms=10,
            prefill_ms_by_instance=[10, 10, 10, 10],
            decode_ms_by_instance=[20, 20, 20, 20],
            transfer_ms_by_decode_instance=[5, 5, 5, 5],
        )
        decision = scheduler.schedule_prefill(
            [call],
            scheduler_mod.PrefillStageState(
                current_time_ms=index * 1000,
                instance_available_at_ms=[index * 1000] * 4,
                decode_available_hint_ms=[index * 1000] * 4,
            ),
        )[0]
        placements.append(
            (decision.assigned_instance, decision.planned_decode_instance)
        )

    assert placements == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_fanout_decode_orders_by_critical_path_pressure():
    scheduler_mod = _load_scheduler()
    calls = [
        _decode_call(
            scheduler_mod,
            call_id="critical",
            output_tokens=128,
            completion_ms=100,
            critical_path_ms=900,
        ),
        _decode_call(
            scheduler_mod,
            call_id="projected",
            output_tokens=128,
            completion_ms=900,
            critical_path_ms=100,
        ),
    ]

    decisions = _schedule_decode(scheduler_mod, calls)

    assert [decision.call_id for decision in decisions] == ["critical", "projected"]


def test_sequential_decode_keeps_projected_ratio_ordering():
    scheduler_mod = _load_scheduler()
    calls = [
        _decode_call(
            scheduler_mod,
            call_id="critical",
            output_tokens=100,
            completion_ms=100,
            critical_path_ms=900,
        ),
        _decode_call(
            scheduler_mod,
            call_id="projected",
            output_tokens=100,
            completion_ms=900,
            critical_path_ms=100,
        ),
    ]

    decisions = _schedule_decode(scheduler_mod, calls)

    assert [decision.call_id for decision in decisions] == ["projected", "critical"]


def test_long_prompt_same_workflow_fanout_prefill_prioritizes_critical_path():
    scheduler_mod = _load_scheduler()
    calls = [
        _call(
            scheduler_mod,
            call_id="critical",
            input_tokens=1_600,
            output_tokens=256,
            completion_ms=100,
            critical_path_ms=900,
            child_count=5,
            decode_ms=100,
        ),
        _call(
            scheduler_mod,
            call_id="short_laxity",
            input_tokens=1_600,
            output_tokens=256,
            completion_ms=100,
            critical_path_ms=100,
            child_count=5,
            decode_ms=600,
        ),
    ]

    decisions = _schedule_prefill(scheduler_mod, calls)

    assert [decision.call_id for decision in decisions] == ["critical", "short_laxity"]


def test_long_prompt_same_workflow_fanout_decode_prioritizes_critical_path():
    scheduler_mod = _load_scheduler()
    calls = [
        _decode_call(
            scheduler_mod,
            call_id="critical",
            input_tokens=1_600,
            output_tokens=256,
            completion_ms=100,
            critical_path_ms=900,
            child_count=5,
            decode_ms=100,
        ),
        _decode_call(
            scheduler_mod,
            call_id="short_laxity",
            input_tokens=1_600,
            output_tokens=256,
            completion_ms=100,
            critical_path_ms=100,
            child_count=5,
            decode_ms=600,
        ),
    ]

    decisions = _schedule_decode(scheduler_mod, calls)

    assert [decision.call_id for decision in decisions] == ["critical", "short_laxity"]


def test_revealed_sibling_prefill_fanout_does_not_require_future_child_count():
    scheduler_mod = _load_scheduler()
    calls = [
        _call(
            scheduler_mod,
            call_id="critical",
            input_tokens=1_600,
            output_tokens=256,
            completion_ms=100,
            critical_path_ms=900,
            child_count=0,
            parent_call_ids=("completed-parent",),
            decode_ms=100,
        ),
        _call(
            scheduler_mod,
            call_id="short_laxity",
            input_tokens=1_600,
            output_tokens=256,
            completion_ms=100,
            critical_path_ms=100,
            child_count=0,
            parent_call_ids=("completed-parent",),
            decode_ms=600,
        ),
    ]

    decisions = _schedule_prefill(scheduler_mod, calls)

    assert [decision.call_id for decision in decisions] == ["critical", "short_laxity"]


def test_revealed_sibling_decode_fanout_does_not_require_future_child_count():
    scheduler_mod = _load_scheduler()
    calls = [
        _decode_call(
            scheduler_mod,
            call_id="critical",
            input_tokens=1_600,
            output_tokens=256,
            completion_ms=100,
            critical_path_ms=900,
            child_count=0,
            parent_call_ids=("completed-parent",),
            decode_ms=100,
        ),
        _decode_call(
            scheduler_mod,
            call_id="short_laxity",
            input_tokens=1_600,
            output_tokens=256,
            completion_ms=100,
            critical_path_ms=100,
            child_count=0,
            parent_call_ids=("completed-parent",),
            decode_ms=600,
        ),
    ]

    decisions = _schedule_decode(scheduler_mod, calls)

    assert [decision.call_id for decision in decisions] == ["critical", "short_laxity"]
