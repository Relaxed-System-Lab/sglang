"""FastAPI sidecar for the HexAgenT planner."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
import time
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException

from sglang.hexgent.cache import prepare_prefix_cache_probe
from sglang.hexgent.planner import HexgentPlanner
from sglang.hexgent.predictor import OutputLengthPredictor
from sglang.hexgent.schemas import (
    PlannerEvent,
    PlannerAdmissionRequest,
    PlannerAdmissionResponse,
    PlannerRevisionPlanEntry,
    PlannerRevisionPlanResponse,
    PlannerScheduleRequest,
    to_jsonable,
)


class PlannerTelemetry:
    """Append one compact timing record for every planner invocation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> Optional["PlannerTelemetry"]:
        path = os.environ.get("HEXGENT_PLANNER_TELEMETRY_PATH")
        return cls(path) if path else None

    @staticmethod
    def _process_thread_count() -> int:
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("Threads:"):
                    return max(1, int(line.split(":", 1)[1].strip()))
        except (OSError, ValueError):
            pass
        return max(1, threading.active_count())

    def record(
        self,
        *,
        endpoint: str,
        payload: dict,
        result: Optional[dict],
        wall_started_ns: int,
        cpu_started_ns: int,
        error: Optional[str] = None,
    ) -> None:
        request_payload = payload.get("request", payload)
        metadata = request_payload.get("hexgent") or {}
        nested_schedule = (result or {}).get("schedule")
        schedule = nested_schedule or result or {}
        phase_timings_ms = dict((result or {}).get("phase_timings_ms") or {})
        if isinstance(nested_schedule, dict):
            phase_timings_ms.update(
                {
                    f"schedule.{name}": value
                    for name, value in (
                        nested_schedule.get("phase_timings_ms") or {}
                    ).items()
                }
            )
        row = {
            "timestamp_ms": time.time() * 1000.0,
            "endpoint": endpoint,
            "workflow_id": metadata.get("workflow_id"),
            "call_id": metadata.get("call_id"),
            "wall_ms": (time.perf_counter_ns() - wall_started_ns) / 1_000_000.0,
            "cpu_ms": (time.process_time_ns() - cpu_started_ns) / 1_000_000.0,
            "process_thread_count": self._process_thread_count(),
            "dispatch_ready": (result or {}).get("dispatch_ready"),
            "queue_position": (result or {}).get("queue_position"),
            "revision": schedule.get("revision"),
            "prefill_worker_url": schedule.get("prefill_worker_url"),
            "decode_worker_url": schedule.get("decode_worker_url"),
            "prefill_priority": schedule.get("prefill_priority"),
            "decode_priority": schedule.get("decode_priority"),
            "planned_decode_instance": schedule.get("planned_decode_instance"),
            "fallback_reason": schedule.get("fallback_reason"),
            "phase_timings_ms": phase_timings_ms,
            "error": error,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, sort_keys=True) + "\n")

    def record_admission_batch(
        self,
        *,
        admission_count: int,
        event_count: int,
        result_count: int,
        wall_started_ns: int,
        cpu_started_ns: int,
    ) -> None:
        """Record the actor's complete batch solve, separate from HTTP enqueue."""
        row = {
            "timestamp_ms": time.time() * 1000.0,
            "endpoint": "/plan_pending_admissions",
            "wall_ms": (time.perf_counter_ns() - wall_started_ns) / 1_000_000.0,
            "cpu_ms": (time.process_time_ns() - cpu_started_ns) / 1_000_000.0,
            "process_thread_count": self._process_thread_count(),
            "admission_count": admission_count,
            "event_count": event_count,
            "result_count": result_count,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, sort_keys=True) + "\n")


class AdmissionCoordinator:
    """Single-writer admission planner with a cheap submission path.

    HTTP handlers only mutate the coordinator inbox.  The background actor is
    the only code that mutates ``HexgentPlanner`` admission state, so a slow
    solve cannot block a later submission or status read.
    """

    def __init__(
        self, planner: HexgentPlanner, telemetry: Optional[PlannerTelemetry] = None
    ) -> None:
        self.planner = planner
        self.telemetry = telemetry
        self._inbox: Dict[Tuple[str, str], PlannerAdmissionRequest] = {}
        self._admissions: Dict[Tuple[str, str], PlannerAdmissionRequest] = {}
        self._admission_generations: Dict[Tuple[str, str], int] = {}
        self._next_admission_generation = 0
        self._snapshot_refresh_keys: set[Tuple[str, str]] = set()
        self._events: list[PlannerEvent] = []
        self._results: Dict[Tuple[str, str], object] = {}
        self._latest_revision_plan: Optional[PlannerRevisionPlanResponse] = None
        self._planning_keys: set[Tuple[str, str]] = set()
        self._inbox_lock = threading.RLock()
        self._planner_lock = threading.Lock()
        self._wake_event: Optional[asyncio.Event] = None
        self._status_changed_event: Optional[asyncio.Event] = None
        self._stopped = False

    @staticmethod
    def _key(request: PlannerAdmissionRequest) -> Tuple[str, str]:
        metadata = request.request.hexgent
        return str(metadata.workflow_id), str(metadata.call_id)

    @staticmethod
    def _event_key(event: PlannerEvent) -> Optional[Tuple[str, str]]:
        if not event.workflow_id or not event.call_id:
            return None
        return str(event.workflow_id), str(event.call_id)

    def enqueue(self, request: PlannerAdmissionRequest):
        key = self._key(request)
        with self._inbox_lock:
            cached = self._results.get(key)
            # Network retries after a successful enqueue must not discard a
            # concrete plan. The router can continue to dispatch using this
            # cached revision without re-entering the solver.
            if cached is not None and cached.dispatch_ready:
                return cached
            self._inbox[key] = request
            self._admissions[key] = request
            self._next_admission_generation += 1
            self._admission_generations[key] = self._next_admission_generation
            self._results.pop(key, None)
        return self.status(request)

    def _discard_plan_entries(self, keys: set[Tuple[str, str]]) -> None:
        plan = self._latest_revision_plan
        if plan is not None:
            entries = [
                entry for entry in plan.entries
                if (entry.workflow_id, entry.call_id) not in keys
            ]
            self._latest_revision_plan = replace(plan, entries=entries) if entries else None

    def _refresh_pending_snapshot(self, event: PlannerEvent) -> None:
        worker_url = event.payload["worker_url"]
        snapshot = event.payload["snapshot"]
        # A cached-ready plan can still await router admission. Only explicit
        # gateway queue ownership allows refreshing such a plan.
        gateway_pending = {
            (str(item["workflow_id"]), str(item["call_id"]))
            for item in event.payload.get("pending_admissions", [])
        }
        refreshed = set()
        for key, admission in list(self._admissions.items()):
            cached = self._results.get(key)
            if cached is not None and cached.dispatch_ready and key not in gateway_pending:
                continue
            request = admission.request
            if not any(worker.url == worker_url for worker in request.prefill_workers + request.decode_workers):
                continue
            state = replace(
                request.state,
                current_time_ms=max(request.state.current_time_ms, int(event.timestamp_ms or 0)),
                worker_snapshots={**request.state.worker_snapshots, worker_url: snapshot},
            )
            updated = replace(admission, request=replace(request, state=state))
            self._inbox[key] = updated
            self._admissions[key] = updated
            self._results.pop(key, None)
            self._next_admission_generation += 1
            self._admission_generations[key] = self._next_admission_generation
            self._snapshot_refresh_keys.add(key)
            refreshed.add(key)
        self._discard_plan_entries(refreshed)

    def enqueue_event(self, event: PlannerEvent) -> None:
        with self._inbox_lock:
            if event.event_type == "worker_snapshot_updated":
                self._refresh_pending_snapshot(event)
                return
            self._events.append(event)
            key = self._event_key(event)
            if key is not None and event.event_type in {
                "admission_cancelled",
                "request_cancelled",
                "request_dispatched",
                "call_completed",
            }:
                self._inbox.pop(key, None)
                self._admissions.pop(key, None)
                self._admission_generations.pop(key, None)
                self._snapshot_refresh_keys.discard(key)
                self._results.pop(key, None)
                self._planning_keys.discard(key)
                self._discard_plan_entries({key})

    def status(self, request: PlannerAdmissionRequest):
        key = self._key(request)
        with self._inbox_lock:
            cached = self._results.get(key)
            pending = key in self._inbox or key in self._planning_keys
            pending_count = len(self._inbox)
        if cached is not None:
            return cached
        return self._planning_pending_response(request, pending_count if pending else 0)

    @staticmethod
    def _planning_pending_response(
        request: PlannerAdmissionRequest, queue_position: int
    ):
        from sglang.hexgent.schemas import PlannerAdmissionResponse

        return PlannerAdmissionResponse(
            dispatch_ready=False,
            retry_after_ms=max(1, int(request.policy.retry_after_ms)),
            queue_position=queue_position,
            reason="planning_pending",
        )

    def plan_next_batch(self) -> int:
        """Drain the newest inbox contents and update cached execution decisions."""

        wall_started_ns = time.perf_counter_ns()
        cpu_started_ns = time.process_time_ns()

        with self._inbox_lock:
            admissions = list(self._inbox.values())
            generations = dict(self._admission_generations)
            refresh_keys = self._snapshot_refresh_keys.intersection(self._inbox)
            self._snapshot_refresh_keys.difference_update(refresh_keys)
            self._inbox.clear()
            self._planning_keys.update(self._key(admission) for admission in admissions)
            events = self._events
            self._events = []
        if not admissions and not events:
            return 0

        with self._planner_lock:
            for event in events:
                self.planner.handle_event(event)
            for admission in admissions:
                if self._key(admission) in refresh_keys:
                    self.planner.enqueue_admission(admission, refresh_pending=True)
                else:
                    self.planner.enqueue_admission(admission)
            results = self.planner.plan_pending_admissions()

        with self._inbox_lock:
            self._planning_keys.difference_update(results)
            results = {
                key: result for key, result in results.items()
                if key in self._admissions
                and self._admission_generations.get(key) == generations.get(key)
            }
            self._results.update(results)
            ready = [
                PlannerRevisionPlanEntry(
                    workflow_id=workflow_id,
                    call_id=call_id,
                    schedule=result.schedule,
                )
                for (workflow_id, call_id), result in results.items()
                if result.dispatch_ready and result.schedule is not None
            ]
            if ready:
                self._latest_revision_plan = PlannerRevisionPlanResponse(
                    revision=max(entry.schedule.revision for entry in ready),
                    entries=ready,
                )
        if self.telemetry is not None:
            self.telemetry.record_admission_batch(
                admission_count=len(admissions),
                event_count=len(events),
                result_count=len(results),
                wall_started_ns=wall_started_ns,
                cpu_started_ns=cpu_started_ns,
            )
        return len(admissions)

    async def wait_for_revision_plan(
        self, *, timeout_ms: int, after_revision: Optional[int] = None
    ) -> PlannerRevisionPlanResponse:
        """Return one complete execution-plan revision to the router."""

        deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0
        while True:
            with self._inbox_lock:
                plan = self._latest_revision_plan
            if plan is not None and (
                after_revision is None or plan.revision > int(after_revision)
            ):
                return plan
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return PlannerRevisionPlanResponse(
                    revision=int(after_revision or 0),
                    entries=[],
                    reason="revision_not_updated",
                )
            event = self._status_changed_event
            if event is None:
                await asyncio.sleep(0)
                continue
            event.clear()
            with self._inbox_lock:
                plan = self._latest_revision_plan
            if plan is not None and (
                after_revision is None or plan.revision > int(after_revision)
            ):
                return plan
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining_s)
            except asyncio.TimeoutError:
                return PlannerRevisionPlanResponse(
                    revision=int(after_revision or 0),
                    entries=[],
                    reason="revision_not_updated",
                )

    def wake(self) -> None:
        if self._wake_event is not None:
            self._wake_event.set()

    def _notify_status_changed(self) -> None:
        """Wake long-polling routers after a new planner-visible event."""

        if self._status_changed_event is not None:
            self._status_changed_event.set()

    async def wait_for_status(
        self,
        request: PlannerAdmissionRequest,
        *,
        timeout_ms: int,
        after_revision: Optional[int] = None,
    ) -> PlannerAdmissionResponse:
        """Wait for a published admission result without periodic polling.

        ``dispatch_ready=True`` means a plan revision is available; router-side
        live credit still controls actual worker dispatch. Arrival, completion,
        cancellation, and plan publication wake waiters through
        ``_status_changed_event``.
        """

        deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0
        while True:
            response = self.status(request)
            revision = (
                response.schedule.revision
                if response.schedule is not None
                else -1
            )
            if response.dispatch_ready and (
                after_revision is None or revision > int(after_revision)
            ):
                return response
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                if response.dispatch_ready and after_revision is not None:
                    return replace(
                        response,
                        dispatch_ready=False,
                        reason="revision_not_updated",
                    )
                return response
            event = self._status_changed_event
            if event is None:
                await asyncio.sleep(0)
                continue
            # This task performs no await between status() and clear(). A plan
            # publication therefore either appears in the status we just read
            # or sets the event after it has been cleared.
            event.clear()
            response = self.status(request)
            revision = (
                response.schedule.revision
                if response.schedule is not None
                else -1
            )
            if response.dispatch_ready and (
                after_revision is None or revision > int(after_revision)
            ):
                return response
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining_s)
            except asyncio.TimeoutError:
                response = self.status(request)
                revision = (
                    response.schedule.revision
                    if response.schedule is not None
                    else -1
                )
                if (
                    response.dispatch_ready
                    and after_revision is not None
                    and revision <= int(after_revision)
                ):
                    return replace(
                        response,
                        dispatch_ready=False,
                        reason="revision_not_updated",
                    )
                return response

    async def wait_for_key(
        self,
        workflow_id: str,
        call_id: str,
        *,
        timeout_ms: int,
        after_revision: Optional[int] = None,
    ) -> Optional[PlannerAdmissionResponse]:
        """Look up an enqueued admission and wait for its next plan revision."""

        key = str(workflow_id), str(call_id)
        with self._inbox_lock:
            request = self._admissions.get(key)
        if request is None:
            return None
        return await self.wait_for_status(
            request,
            timeout_ms=timeout_ms,
            after_revision=after_revision,
        )

    async def run(self, *, coalesce_ms: int = 2) -> None:
        self._wake_event = asyncio.Event()
        self._status_changed_event = asyncio.Event()
        while not self._stopped:
            await self._wake_event.wait()
            self._wake_event.clear()
            if coalesce_ms:
                await asyncio.sleep(max(0, int(coalesce_ms)) / 1000.0)
            await asyncio.to_thread(self.plan_next_batch)
            self._notify_status_changed()

    def stop(self) -> None:
        self._stopped = True
        self.wake()


def build_app(
    *,
    planner: Optional[HexgentPlanner] = None,
    predictor: Optional[OutputLengthPredictor] = None,
    telemetry: Optional[PlannerTelemetry] = None,
) -> FastAPI:
    app = FastAPI(title="HexAgenT Planner", version="0.1")
    planner = planner or HexgentPlanner(predictor=predictor)
    telemetry = telemetry or PlannerTelemetry.from_environment()
    admission_coordinator = AdmissionCoordinator(planner, telemetry=telemetry)
    app.state.admission_coordinator = admission_coordinator
    app.state.admission_task = None

    @app.on_event("startup")
    async def start_admission_coordinator():
        app.state.admission_task = asyncio.create_task(admission_coordinator.run())

    @app.on_event("shutdown")
    async def stop_admission_coordinator():
        admission_coordinator.stop()
        task = app.state.admission_task
        if task is not None:
            await task

    @app.get("/health")
    async def health():
        return planner.health()

    @app.post("/schedule_request")
    async def schedule_request(payload: dict):
        wall_started_ns = time.perf_counter_ns()
        cpu_started_ns = time.process_time_ns()
        try:
            request = PlannerScheduleRequest.from_dict(payload)
            if os.environ.get("HEXGENT_ENABLE_PREFIX_CACHE_PROBES", "0") == "1":
                request = await prepare_prefix_cache_probe(request)
            response = planner.schedule_request(request)
        except Exception as exc:
            if telemetry is not None:
                telemetry.record(
                    endpoint="/schedule_request",
                    payload=payload,
                    result=None,
                    wall_started_ns=wall_started_ns,
                    cpu_started_ns=cpu_started_ns,
                    error=str(exc),
                )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = to_jsonable(response)
        if telemetry is not None:
            telemetry.record(
                endpoint="/schedule_request",
                payload=payload,
                result=result,
                wall_started_ns=wall_started_ns,
                cpu_started_ns=cpu_started_ns,
            )
        return result

    @app.post("/plan")
    async def plan(payload: dict):
        # Alias for clients that batch planning through a generic route.
        return await schedule_request(payload)

    @app.post("/admit_request")
    async def admit_request(payload: dict):
        wall_started_ns = time.perf_counter_ns()
        cpu_started_ns = time.process_time_ns()
        try:
            request = PlannerAdmissionRequest.from_dict(payload)
            if os.environ.get("HEXGENT_ENABLE_PREFIX_CACHE_PROBES", "0") == "1":
                request = replace(request, request=await prepare_prefix_cache_probe(request.request))
            response = planner.admit_request(request)
        except Exception as exc:
            if telemetry is not None:
                telemetry.record(
                    endpoint="/admit_request",
                    payload=payload,
                    result=None,
                    wall_started_ns=wall_started_ns,
                    cpu_started_ns=cpu_started_ns,
                    error=str(exc),
                )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = to_jsonable(response)
        if telemetry is not None:
            telemetry.record(
                endpoint="/admit_request",
                payload=payload,
                result=result,
                wall_started_ns=wall_started_ns,
                cpu_started_ns=cpu_started_ns,
            )
        return result

    @app.post("/enqueue_admission")
    async def enqueue_admission(payload: dict):
        """Accept an admission immediately; the actor computes its next plan."""

        wall_started_ns = time.perf_counter_ns()
        cpu_started_ns = time.process_time_ns()
        try:
            request = PlannerAdmissionRequest.from_dict(payload)
            if os.environ.get("HEXGENT_ENABLE_PREFIX_CACHE_PROBES", "0") == "1":
                request = replace(request, request=await prepare_prefix_cache_probe(request.request))
            response = admission_coordinator.enqueue(request)
            admission_coordinator.wake()
        except Exception as exc:
            if telemetry is not None:
                telemetry.record(
                    endpoint="/enqueue_admission",
                    payload=payload,
                    result=None,
                    wall_started_ns=wall_started_ns,
                    cpu_started_ns=cpu_started_ns,
                    error=str(exc),
                )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = to_jsonable(response)
        if telemetry is not None:
            telemetry.record(
                endpoint="/enqueue_admission",
                payload=payload,
                result=result,
                wall_started_ns=wall_started_ns,
                cpu_started_ns=cpu_started_ns,
            )
        return result

    @app.get("/admission_status")
    async def admission_status(workflow_id: str, call_id: str):
        key = str(workflow_id), str(call_id)
        with admission_coordinator._inbox_lock:
            result = admission_coordinator._results.get(key)
            pending = (
                key in admission_coordinator._inbox
                or key in admission_coordinator._planning_keys
            )
        if result is not None:
            return to_jsonable(result)
        return {
            "dispatch_ready": False,
            "retry_after_ms": 1,
            "queue_position": 1 if pending else 0,
            "reason": "planning_pending" if pending else "unknown_admission",
        }

    @app.get("/wait_admission")
    async def wait_admission(
        workflow_id: str,
        call_id: str,
        timeout_ms: int = 300_000,
        after_revision: Optional[int] = None,
    ):
        response = await admission_coordinator.wait_for_key(
            workflow_id,
            call_id,
            timeout_ms=timeout_ms,
            after_revision=after_revision,
        )
        if response is None:
            return {
                "dispatch_ready": False,
                "retry_after_ms": 0,
                "queue_position": 0,
                "reason": "unknown_admission",
            }
        return to_jsonable(response)

    @app.get("/wait_plan_revision")
    async def wait_plan_revision(
        timeout_ms: int = 300_000,
        after_revision: Optional[int] = None,
    ):
        response = await admission_coordinator.wait_for_revision_plan(
            timeout_ms=timeout_ms,
            after_revision=after_revision,
        )
        return to_jsonable(response)

    @app.post("/event")
    async def event(payload: dict):
        try:
            planner_event = PlannerEvent.from_dict(payload)
            admission_coordinator.enqueue_event(planner_event)
            admission_coordinator.wake()
            return {"status": "ok", "queued": True}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HexAgenT planner sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30999)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--telemetry-path")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        build_app(
            telemetry=(
                PlannerTelemetry(args.telemetry_path)
                if args.telemetry_path
                else None
            )
        ),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
