"""Online public input invariants, independent of the frozen placement calendar."""
from types import SimpleNamespace
import pytest
from test_hexgent_planner import _load


def request(schemas, *, future=False):
    return schemas.PlannerScheduleRequest(route="/v1/chat/completions", model="m", request_text="revealed",
        max_tokens=1, hexgent=schemas.HexgentRequestMetadata(workflow_id="wf", call_id="root",
            input_tokens=10, predicted_output_tokens=1, workflow_horizon_ms=999999 if future else 0,
            critical_path_ms=999999 if future else 0, child_count=999 if future else 0),
        prefill_workers=[schemas.WorkerDescriptor(id="p",url="http://p",worker_type="prefill")],
        decode_workers=[schemas.WorkerDescriptor(id="d",url="http://d",worker_type="decode")],
        state=schemas.PlannerClusterState(current_time_ms=0))


def test_client_critical_hint_cannot_restore_unrevealed_future_path():
    module = _load("planner")
    wf = module.WorkflowRuntimeState("wf", 0)
    wf.revealed_calls["p"] = module.RevealedCallRuntimeState((), 10)
    wf.revealed_calls["c"] = module.RevealedCallRuntimeState(("p",), 20)
    assert module.HexgentPlanner._call_critical_path_ms(wf,"c",999999) == 30


def test_future_critical_child_and_private_horizon_hints_do_not_change_revealed_input():
    module, schemas = _load("planner"), _load("schemas")
    calls=[]
    for future in [False, True]:
        planner=module.HexgentPlanner(); req=request(schemas,future=future)
        call=planner._prefill_call(request=req,input_tokens=10,output_tokens=1,
                                   workflow=planner._workflow_state(req.hexgent))
        calls.append((call.workflow_horizon_ms,call.critical_path_ms,call.child_count))
    assert calls[0] == calls[1]


def test_unrounded_forecast_service_reaches_priority_and_revealed_horizon():
    module, schemas, latency = _load("planner"), _load("schemas"), _load("latency_estimator")
    value=latency.LatencyEstimate([1],[1],[1],
        prefill_ms_unrounded_by_instance=[.11], decode_ms_unrounded_by_instance=[.026],
        transfer_ms_unrounded_by_decode_instance=[.07])
    estimator=SimpleNamespace(estimate=lambda *args,**kwargs:value)
    planner=module.HexgentPlanner(latency_estimator=estimator);req=request(schemas)
    call=planner._prefill_call(request=req,input_tokens=10,output_tokens=1,
                               workflow=planner._workflow_state(req.hexgent))
    assert call.priority_seconds is not None
    assert call.priority_seconds.prefill_s_by_instance == pytest.approx((.00011,))
    assert call.priority_seconds.decode_s_by_instance == pytest.approx((.000026,))
    assert call.priority_seconds.transfer_s_by_decode_instance == pytest.approx((.00007,))
    assert call.priority_seconds.workflow_horizon_s == pytest.approx(.000206)


def test_same_workflow_siblings_share_the_final_float_horizon_before_ranking():
    module, schemas, latency = _load("planner"), _load("schemas"), _load("latency_estimator")
    def estimate(req, *, input_tokens, output_tokens):
        return latency.LatencyEstimate([1], [1], [1], [input_tokens * .01], [.026], [.07])
    planner = module.HexgentPlanner(latency_estimator=SimpleNamespace(estimate=estimate))
    first = request(schemas)
    from dataclasses import replace
    second = replace(first, hexgent=replace(first.hexgent, call_id="child", input_tokens=20,
        parent_call_ids=["root"]))
    seen = []
    original = planner.scheduler.schedule_prefill
    def capture(calls, state):
        seen.extend(calls)
        return original(calls, state)
    planner.scheduler.schedule_prefill = capture
    for req in [first, second]:
        planner.enqueue_admission(schemas.PlannerAdmissionRequest(request=req))
    planner.plan_pending_admissions()
    assert {call.call_id for call in seen} == {"root", "child"}
    assert all(call.workflow_horizon_ms == 6 for call in seen)
    assert all(call.priority_seconds.workflow_horizon_s == pytest.approx(.000492) for call in seen)
    assert {call.call_id: call.child_count for call in seen} == {"root": 1, "child": 0}


def test_snapshot_first_prefill_recovery_preserves_cold_float_horizon():
    from test_hexgent_live_planner_alignment import request, snapshot
    planner = _load("planner").HexgentPlanner()
    req = request()
    req.state.worker_snapshots["http://p0"]["queues"]["waiting"] = [{
        "rid": "recovered", "workflow_id": "recovered-wf", "call_id": "recovered",
        "input_tokens": 100, "predicted_output_tokens": 20,
        "estimated_prefill_time": 10.4, "estimated_decode_time": 7.4,
        "estimated_transfer_time": .4}]
    record = _load("snapshots").waiting_prefill_calls(req)[0]
    call = planner._prefill_call_from_snapshot(record, req)
    assert call.priority_seconds.workflow_horizon_s == pytest.approx(.0182)
