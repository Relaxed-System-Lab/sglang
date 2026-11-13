"""Exercise the real sidecar coordinator event/actor paths without HTTP services."""
import asyncio
from dataclasses import replace

import pytest

from test_hexgent_planner import _load, _complete_live_snapshots


def setup(tmp_path):
    schemas = _load("schemas")
    planner_module = _load("planner")
    sidecar = _load("sidecar")
    planner = planner_module.HexgentPlanner(revision_state_path=tmp_path / "revision")
    planner.scheduler.known_priority_insertion_projection = False
    planner.scheduler.decode_admission_projection = False
    request = schemas.PlannerAdmissionRequest(
        request=schemas.PlannerScheduleRequest(
            route="/v1/chat/completions", model="test", request_text="original", max_tokens=4,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf-pending", call_id="original", input_tokens=4, predicted_output_tokens=4,
            ),
            prefill_workers=[schemas.WorkerDescriptor(id="p", url="http://prefill", worker_type="prefill")],
            decode_workers=[schemas.WorkerDescriptor(id="d", url="http://decode", worker_type="decode", capacity_tokens=1000)],
            state=schemas.PlannerClusterState(current_time_ms=1, worker_snapshots={
                "http://decode": {"mode": "decode", "queues": {"waiting": []}, "running": []}}),
        ),
        policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
    )
    _complete_live_snapshots(request.request)
    return schemas, planner, sidecar, request


@pytest.mark.parametrize("event_type", ["admission_cancelled", "request_cancelled", "request_dispatched", "call_completed"])
def test_terminal_event_during_solve_cannot_resurrect_a_cached_admission(tmp_path, monkeypatch, event_type):
    schemas, planner, sidecar, admission = setup(tmp_path)
    coordinator = sidecar.AdmissionCoordinator(planner)
    original = planner.plan_pending_admissions
    def complete_during_solve():
        result = original()
        coordinator.enqueue_event(schemas.PlannerEvent(
            event_type=event_type, workflow_id="wf-pending", call_id="original",
        ))
        return result
    monkeypatch.setattr(planner, "plan_pending_admissions", complete_during_solve)
    coordinator.enqueue(admission)
    coordinator.plan_next_batch()
    assert not coordinator.status(admission).dispatch_ready
    revision = asyncio.run(coordinator.wait_for_revision_plan(timeout_ms=0))
    assert revision.entries == []


def snapshot_event(schemas, *, pending=True):
    payload = {
        "worker_url": "http://prefill",
        "snapshot": {"snapshot": {
            "mode": "prefill", "timestamp": 7,
            "limits": {"max_total_num_tokens": 10000, "max_running_requests": 16}, "queues": {"waiting": []}, "running": [],
        }},
    }
    if pending:
        payload["pending_admissions"] = [{"workflow_id": "wf-pending", "call_id": "original"}]
    return schemas.PlannerEvent(event_type="worker_snapshot_updated", timestamp_ms=7, payload=payload)


def test_fresh_snapshot_event_replans_the_same_pending_request_without_new_arrival(tmp_path, monkeypatch):
    schemas, planner, sidecar, admission = setup(tmp_path)
    app = sidecar.build_app(planner=planner)
    coordinator = app.state.admission_coordinator
    endpoint = next(route.endpoint for route in app.routes if route.path == "/event")
    observed = []
    original = planner.plan_pending_admissions
    def record_solve():
        observed.append(planner._pending_admissions[("wf-pending", "original")].state.worker_snapshots)
        return original()
    monkeypatch.setattr(planner, "plan_pending_admissions", record_solve)

    async def scenario():
        actor = asyncio.create_task(coordinator.run(coalesce_ms=0))
        await asyncio.sleep(0)
        try:
            coordinator.enqueue(admission)
            coordinator.wake()
            first = await coordinator.wait_for_status(admission, timeout_ms=1000)
            assert first.dispatch_ready
            event = snapshot_event(schemas)
            await endpoint(schemas.to_jsonable(event))
            second = await coordinator.wait_for_status(
                admission, timeout_ms=1000, after_revision=first.schedule.revision,
            )
            assert second.dispatch_ready, "fresh snapshot must wake the existing actor and publish a new plan"
            assert second.schedule.revision > first.schedule.revision
            assert set(coordinator._admissions) == {("wf-pending", "original")}
            assert observed[-1]["http://prefill"]["snapshot"]["timestamp"] == 7
        finally:
            coordinator.stop()
            await actor
    asyncio.run(scenario())
    assert len(observed) == 2


def test_ready_cached_admission_requires_gateway_pending_proof_before_refresh(tmp_path):
    schemas, planner, sidecar, admission = setup(tmp_path)
    coordinator = sidecar.AdmissionCoordinator(planner)
    coordinator.enqueue(admission)
    coordinator.plan_next_batch()
    revision = coordinator.status(admission).schedule.revision
    coordinator.enqueue_event(snapshot_event(schemas, pending=False))
    assert coordinator.status(admission).schedule.revision == revision
    assert not coordinator._inbox


def test_snapshot_refresh_during_solve_discards_the_old_generation(tmp_path, monkeypatch):
    schemas, planner, sidecar, admission = setup(tmp_path)
    coordinator = sidecar.AdmissionCoordinator(planner)
    original = planner.plan_pending_admissions
    def refresh_during_solve():
        result = original()
        coordinator.enqueue_event(snapshot_event(schemas))
        return result
    monkeypatch.setattr(planner, "plan_pending_admissions", refresh_during_solve)
    coordinator.enqueue(admission)
    coordinator.plan_next_batch()
    assert not coordinator.status(admission).dispatch_ready
    assert ("wf-pending", "original") in coordinator._inbox
