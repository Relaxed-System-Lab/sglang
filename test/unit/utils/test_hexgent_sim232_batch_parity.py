"""Source232 parity for a numerical change in a revealed batch above 32 calls."""

from test_hexgent_sim232_decision_parity import _load, frozen_core, profile, signature


NOW_MS = 1000


def _batch(module):
    # These lengths avoid the short-decode and fan-out dispatch branches.
    # With one serial D slot, the anchor adds 100ms of observable queue wait.
    specs = [
        ("anchor", 0, 100, 100),
        ("a", 900, 100, 20),
        ("b", 990, 10, 1),
        *[(f"filler-{index:02d}", NOW_MS, 1000, 1) for index in range(30)],
    ]
    return [module.DecodeCallInfo(
        call_id=name, workflow_id=f"wf-{name}", input_tokens=512,
        output_tokens=100, workflow_arrival_ms=arrival, reveal_sequence=0,
        workflow_horizon_ms=horizon, workflow_completion_ms=0,
        workflow_attained_service_ms=0, ideal_service_ms=duration,
        ready_at_ms=NOW_MS, decode_demand_tokens=1000,
        decode_ms_by_instance=[duration], locked_decode_instance=None,
    ) for name, arrival, horizon, duration in specs]


def _decide(module, *, recompute):
    calls = _batch(module)
    assert len(calls) == 33 > module.GREEDY_RECOMPUTE_LIMIT
    core = profile(module)
    core.recompute_large_ratio_batches = recompute
    for call in calls:
        core.on_workflow_arrival(
            call.workflow_id, call.workflow_arrival_ms, call.workflow_horizon_ms)
    state = module.DecodeStageState(NOW_MS, [1000], [[]])
    state.max_batch_size_by_instance = [1]
    decisions = core.schedule_decode(calls, state)
    # The public one-slot decisions uniquely determine their finish times.
    by_id = {call.call_id: call for call in calls}
    finish = NOW_MS
    elapsed = {}
    for decision in decisions:
        call = by_id[decision.call_id]
        finish = max(finish, call.ready_at_ms) + call.decode_ms_by_instance[decision.assigned_instance]
        elapsed[decision.call_id] = finish - NOW_MS
    return decisions, elapsed


def test_large_ratio_batch_recomputes_the_frozen_numeric_order(monkeypatch):
    monkeypatch.setenv("HEXGENT_PROJECTED_RATIO_ORDER", "high_first")
    monkeypatch.setenv("AGENTIC_PD_OURS_PLACEMENT_STRATEGY", "projected_finish")
    expected = {}
    for recompute in (False, True):
        frozen_rows, elapsed = _decide(frozen_core(), recompute=recompute)
        live_rows, live_elapsed = _decide(_load("scheduler"), recompute=recompute)
        assert signature(live_rows) == signature(frozen_rows)
        assert live_elapsed == elapsed
        expected[recompute] = frozen_rows, elapsed

    static_rows, static_elapsed = expected[False]
    refreshed_rows, refreshed_elapsed = expected[True]
    # Initial ratios are anchor=11, A=1.2, B=1.1. After the anchor runs,
    # queue-aware ratios become A=2.2 and B=11.1, so B moves ahead of A.
    assert [row.call_id for row in static_rows[:3]] == ["anchor", "a", "b"]
    assert [row.call_id for row in refreshed_rows[:3]] == ["anchor", "b", "a"]
    assert static_elapsed["b"] == 121
    assert refreshed_elapsed["b"] == 101
    assert sum(static_elapsed.values()) == 4436
    assert sum(refreshed_elapsed.values()) == 4417
