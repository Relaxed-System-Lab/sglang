"""Stateful HexAgenT planner used by the sidecar service."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from sglang.hexgent.cache import observed_prefix_tokens
from sglang.hexgent.latency_estimator import (
    HybridLatencyEstimator,
    LatencyEstimate,
    LatencyEstimator,
)
from sglang.hexgent.predictor import (
    OutputLengthPredictor,
    build_output_length_predictor,
)
from sglang.hexgent.schemas import (
    HexgentRequestMetadata,
    PlannerAdmissionPolicy,
    PlannerAdmissionRequest,
    PlannerAdmissionResponse,
    PlannerClusterState,
    PlannerEvent,
    PlannerScheduleRequest,
    PlannerScheduleResponse,
    WorkerDescriptor,
    WorkerMutationAction,
)
from sglang.hexgent.scheduler import (
    AgenticPDExactScheduler,
    build_scheduler,
    DecodeCallInfo,
    DecodeStageState,
    find_earliest_fifo_admission_start,
    PrefillCallInfo,
    PrefillStageState,
    PrioritySeconds,
)
from sglang.hexgent.snapshots import (
    SnapshotWaitingCall,
    build_live_stage_states,
    decode_snapshot_calls,
    maybe_apply_live_worker_snapshots,
    snapshot_request_keys,
    waiting_decode_calls,
    waiting_prefill_calls,
)


LOCAL_PRIORITY_UPDATES_ENV = "HEXGENT_ENABLE_LOCAL_PRIORITY_UPDATES"


def _local_priority_updates_enabled() -> bool:
    """Enable safe in-place ordering updates by default.

    Unlike the legacy mutation flag, this never moves a request across a
    worker or into bootstrap/transfer state.
    """

    return os.environ.get(LOCAL_PRIORITY_UPDATES_ENV, "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _estimate_input_tokens(text: str) -> int:
    return max(1, len(str(text).split()))


def _latency_ms(base_ms: float, coeff_ms: float, tokens: int) -> int:
    return max(1, int(round(float(base_ms) + float(coeff_ms) * max(1, tokens))))


@dataclass
class RevealedCallRuntimeState:
    parent_call_ids: tuple[str, ...]
    estimated_service_ms: int
    measured_service_ms: Optional[int] = None
    end_to_end_latency_ms: Optional[int] = None
    completed: bool = False
    estimated_service_ms_unrounded: Optional[float] = None
    measured_service_ms_unrounded: Optional[float] = None

    @property
    def horizon_service_ms_unrounded(self) -> float:
        value = self.measured_service_ms_unrounded
        if value is None:
            value = self.estimated_service_ms_unrounded
        return float(self.horizon_service_ms if value is None else value)

    @property
    def horizon_service_ms(self) -> int:
        return max(
            1,
            int(
                self.measured_service_ms
                if self.measured_service_ms is not None
                else self.estimated_service_ms
            ),
        )


@dataclass
class WorkflowRuntimeState:
    workflow_id: str
    arrival_ms: int
    arrival_ms_unrounded: Optional[float] = None
    horizon_ms_unrounded: Optional[float] = None
    completion_ms_unrounded: float = 0.0
    horizon_ms: int = 1
    completion_ms: int = 0
    attained_service_ms: int = 0
    completed_calls: set[str] = field(default_factory=set)
    revealed_calls: Dict[str, RevealedCallRuntimeState] = field(default_factory=dict)


@dataclass
class PlannedCallReservation:
    workflow_id: str
    call_id: str
    prefill_worker_id: str
    prefill_worker_url: str
    prefill_end_ms: int
    decode_worker_id: str
    decode_worker_url: str
    decode_start_ms: int
    decode_end_ms: int
    decode_demand_tokens: int


class HexgentPlanner:
    _REVISION_BLOCK_SIZE = 1_000_000

    def __init__(
        self,
        *,
        predictor: Optional[OutputLengthPredictor] = None,
        scheduler: Optional[AgenticPDExactScheduler] = None,
        latency_estimator: Optional[LatencyEstimator] = None,
        revision_state_path: Optional[str | Path] = None,
    ) -> None:
        self.predictor = predictor or build_output_length_predictor()
        self.scheduler = scheduler or build_scheduler()
        self.latency_estimator = latency_estimator or HybridLatencyEstimator()
        configured_revision_path = revision_state_path or os.environ.get(
            "HEXGENT_REVISION_STATE_PATH"
        )
        self._revision_state_path = Path(
            configured_revision_path
            or (Path(tempfile.gettempdir()) / "sglang_hexgent_revision")
        )
        self._revision = self._load_revision_epoch()
        self._workflows: Dict[str, WorkflowRuntimeState] = {}
        self._reservations: Dict[Tuple[str, str], PlannedCallReservation] = {}
        self._pending_admissions: Dict[Tuple[str, str], PlannerScheduleRequest] = {}
        self._pending_admission_policies: Dict[
            Tuple[str, str], PlannerAdmissionPolicy
        ] = {}
        self._pending_admission_order: Dict[Tuple[str, str], int] = {}
        self._next_admission_order = 0
        self._latest_event_time_ms = 0
        self._admission_results: Dict[
            Tuple[str, str], PlannerAdmissionResponse
        ] = {}

    @property
    def revision(self) -> int:
        return self._revision

    def _load_revision_epoch(self) -> int:
        stored = 0
        try:
            stored = int(self._revision_state_path.read_text().strip())
        except FileNotFoundError:
            pass
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid HexAgenT revision state: {self._revision_state_path}"
            ) from exc
        revision = max(time.time_ns(), stored)
        self._revision_limit = revision + self._REVISION_BLOCK_SIZE
        # Persist a high-water mark once per process lifetime/block rather than
        # adding filesystem latency to every solve.
        self._persist_revision(self._revision_limit)
        return revision

    def _persist_revision(self, revision: int) -> None:
        path = self._revision_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(str(int(revision)))
        os.replace(temporary, path)

    def _next_revision(self) -> int:
        candidate = self._revision + 1
        if candidate >= self._revision_limit:
            self._revision_limit = candidate + self._REVISION_BLOCK_SIZE
            self._persist_revision(self._revision_limit)
        self._revision = candidate
        return self._revision

    def health(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "revision": self._revision,
            "workflows": len(self._workflows),
            "pending_admissions": len(self._pending_admissions),
        }

    def _workflow_state(self, meta: HexgentRequestMetadata) -> WorkflowRuntimeState:
        return self._workflow_state_values(
            workflow_id=meta.workflow_id,
            arrival_ms=int(meta.workflow_arrival_ms),
        )

    def _workflow_state_values(
        self,
        *,
        workflow_id: str,
        arrival_ms: int,
        horizon_ms: Optional[int] = None,
    ) -> WorkflowRuntimeState:
        workflow_id = str(workflow_id)
        # `workflow_horizon_ms` was supplied by older trace replayers. The
        # online planner owns horizon construction, so a client value cannot
        # override the revealed-DAG estimate.
        del horizon_ms
        state = self._workflows.get(workflow_id)
        if state is None:
            state = WorkflowRuntimeState(
                workflow_id=workflow_id,
                arrival_ms=int(arrival_ms),
                arrival_ms_unrounded=float(arrival_ms),
            )
            self._workflows[workflow_id] = state
            self.scheduler.on_workflow_arrival(
                workflow_id,
                arrival_ms=state.arrival_ms,
                horizon_ms=state.horizon_ms,
            )
        return state

    def _refresh_workflow_horizon(self, workflow: WorkflowRuntimeState) -> None:
        """One revealed-DAG pass retains both calendar and original precision."""
        memo = {}
        def path_service(call_id):
            if call_id not in memo:
                call = workflow.revealed_calls[call_id]
                parents = [path_service(parent) for parent in call.parent_call_ids
                           if parent in workflow.revealed_calls]
                memo[call_id] = (
                    max((value[0] for value in parents), default=0) + call.horizon_service_ms,
                    max((value[1] for value in parents), default=0.0) + call.horizon_service_ms_unrounded,
                )
            return memo[call_id]
        paths = [path_service(call_id) for call_id in workflow.revealed_calls]
        horizon_ms = max(1, max((value[0] for value in paths), default=1))
        workflow.horizon_ms_unrounded = max(1e-9, max((value[1] for value in paths), default=1.0))
        if horizon_ms != workflow.horizon_ms:
            workflow.horizon_ms = horizon_ms
            self.scheduler.on_workflow_arrival(workflow.workflow_id,
                arrival_ms=workflow.arrival_ms, horizon_ms=workflow.horizon_ms)

    @staticmethod
    def _completed_critical_path_ms(workflow: WorkflowRuntimeState) -> int:
        """Return completed service on the DAG critical path, not all branches.

        Summing completed calls turns a fan-out into artificial serial work.  It
        makes the remaining-service calculation pessimistic after parallel
        branches finish, which in turn distorts later admission priorities.
        Missing parent records can occur after a sidecar restart; those are
        treated as an unknown root rather than blocking progress forever.
        """

        memo: Dict[str, int] = {}

        def completed_path_service(call_id: str) -> int:
            if call_id in memo:
                return memo[call_id]
            call = workflow.revealed_calls.get(call_id)
            if (
                call is None
                or not call.completed
                or call.measured_service_ms is None
            ):
                return 0
            known_parents = [
                parent_id
                for parent_id in call.parent_call_ids
                if parent_id in workflow.revealed_calls
            ]
            if any(
                not workflow.revealed_calls[parent_id].completed
                or workflow.revealed_calls[parent_id].measured_service_ms is None
                for parent_id in known_parents
            ):
                return 0
            memo[call_id] = call.horizon_service_ms + max(
                (completed_path_service(parent_id) for parent_id in known_parents),
                default=0,
            )
            return memo[call_id]

        return max(
            (completed_path_service(call_id) for call_id in workflow.revealed_calls),
            default=0,
        )

    def _reveal_call(self, workflow: WorkflowRuntimeState, meta: HexgentRequestMetadata,
                     ideal_service_ms: int, ideal_service_ms_unrounded: Optional[float] = None) -> None:
        call_id = str(meta.call_id)
        existing = workflow.revealed_calls.get(call_id)
        workflow.revealed_calls[call_id] = RevealedCallRuntimeState(
            parent_call_ids=tuple(str(parent) for parent in meta.parent_call_ids),
            estimated_service_ms=max(1, int(ideal_service_ms)),
            measured_service_ms=existing.measured_service_ms if existing else None,
            end_to_end_latency_ms=existing.end_to_end_latency_ms if existing else None,
            completed=existing.completed if existing else False,
            estimated_service_ms_unrounded=(float(ideal_service_ms) if ideal_service_ms_unrounded is None
                                             else float(ideal_service_ms_unrounded)),
            measured_service_ms_unrounded=existing.measured_service_ms_unrounded if existing else None,
        )
        self._refresh_workflow_horizon(workflow)

    def _reveal_snapshot_call(
        self,
        workflow: WorkflowRuntimeState,
        *,
        call_id: str,
        ideal_service_ms: int,
        ideal_service_ms_unrounded: Optional[float] = None,
        parent_call_ids: tuple[str, ...] = (),
    ) -> None:
        """Recover a queued call after sidecar restart when parents are unknown."""

        workflow.revealed_calls[str(call_id)] = RevealedCallRuntimeState(
            parent_call_ids=tuple(str(parent) for parent in parent_call_ids),
            estimated_service_ms=max(1, int(ideal_service_ms)),
            estimated_service_ms_unrounded=ideal_service_ms_unrounded,
        )
        self._refresh_workflow_horizon(workflow)

    @staticmethod
    def _known_critical_path_ms(
        workflow: WorkflowRuntimeState, call_id: str
    ) -> int:
        """Return this call's downstream path across the revealed subgraph."""

        children: Dict[str, List[str]] = {}
        for child_id, call in workflow.revealed_calls.items():
            for parent_id in call.parent_call_ids:
                children.setdefault(parent_id, []).append(child_id)
        memo: Dict[str, int] = {}

        def visit(node_id: str) -> int:
            if node_id in memo:
                return memo[node_id]
            call = workflow.revealed_calls.get(node_id)
            if call is None:
                return 0
            memo[node_id] = call.horizon_service_ms + max(
                (visit(child_id) for child_id in children.get(node_id, [])), default=0
            )
            return memo[node_id]

        return visit(str(call_id))

    @staticmethod
    def _known_path_through_call_ms(
        workflow: WorkflowRuntimeState, call_id: str
    ) -> int:
        """Return the longest revealed parent path ending at ``call_id``.

        Newly revealed fan-out siblings have no known descendants yet, so a
        downstream-only path gives them only their own service time.  Their
        known parent edges are enough to calculate the path accumulated before
        this call without assuming any future workflow structure.
        """

        memo: Dict[str, int] = {}

        def visit(node_id: str) -> int:
            if node_id in memo:
                return memo[node_id]
            call = workflow.revealed_calls.get(node_id)
            if call is None:
                return 0
            memo[node_id] = call.horizon_service_ms + max(
                (
                    visit(parent_id)
                    for parent_id in call.parent_call_ids
                    if parent_id in workflow.revealed_calls
                ),
                default=0,
            )
            return memo[node_id]

        return visit(str(call_id))

    @classmethod
    def _call_critical_path_ms(cls, workflow: WorkflowRuntimeState, call_id: str,
                               client_hint_ms: int = 0) -> int:
        """The wire hint cannot add unrevealed descendants to the online path."""
        del client_hint_ms
        return max(cls._known_critical_path_ms(workflow, call_id),
                   cls._known_path_through_call_ms(workflow, call_id))

    def _reservation_key(self, workflow_id: str, call_id: str) -> Tuple[str, str]:
        return str(workflow_id), str(call_id)

    def _request_key(self, request: PlannerScheduleRequest) -> Tuple[str, str]:
        return self._reservation_key(
            request.hexgent.workflow_id,
            request.hexgent.call_id,
        )

    @staticmethod
    def _decision_key(decision) -> Tuple[str, str]:
        if decision.workflow_id is None:
            raise ValueError("Scheduler decision is missing workflow_id")
        return str(decision.workflow_id), str(decision.call_id)

    def _prune_reservations(self, current_time_ms: int) -> None:
        expired = [
            key
            for key, reservation in self._reservations.items()
            if reservation.prefill_end_ms <= current_time_ms
            and reservation.decode_end_ms <= current_time_ms
        ]
        for key in expired:
            self._reservations.pop(key, None)

    def _worker_matches(
        self,
        worker: WorkerDescriptor,
        *,
        worker_id: str,
        worker_url: str,
    ) -> bool:
        return worker.id == worker_id or worker.url == worker_url

    def _base_prefill_available(self, request: PlannerScheduleRequest) -> List[int]:
        provided = list(request.state.prefill_available_at_ms)
        if provided:
            return provided
        return [int(worker.available_at_ms) for worker in request.prefill_workers]

    def _base_decode_available(self, request: PlannerScheduleRequest) -> List[int]:
        provided = list(request.state.decode_available_at_ms)
        if provided:
            return provided
        return [int(worker.available_at_ms) for worker in request.decode_workers]

    def _prefill_available(self, request: PlannerScheduleRequest) -> List[int]:
        current_time_ms = int(request.state.current_time_ms)
        self._prune_reservations(current_time_ms)
        available = self._base_prefill_available(request)
        represented = snapshot_request_keys(request, stage="prefill")
        represented.update(snapshot_request_keys(request, stage="decode"))
        for key, reservation in self._reservations.items():
            if key in represented:
                continue
            if reservation.prefill_end_ms <= current_time_ms:
                continue
            for index, worker in enumerate(request.prefill_workers):
                if self._worker_matches(
                    worker,
                    worker_id=reservation.prefill_worker_id,
                    worker_url=reservation.prefill_worker_url,
                ):
                    available[index] = max(available[index], reservation.prefill_end_ms)
                    break
        return available

    def _decode_available(self, request: PlannerScheduleRequest) -> List[int]:
        current_time_ms = int(request.state.current_time_ms)
        self._prune_reservations(current_time_ms)
        available = self._base_decode_available(request)
        represented = snapshot_request_keys(request, stage="decode")
        for key, reservation in self._reservations.items():
            if key in represented:
                continue
            if reservation.decode_end_ms <= current_time_ms:
                continue
            for index, worker in enumerate(request.decode_workers):
                if self._worker_matches(
                    worker,
                    worker_id=reservation.decode_worker_id,
                    worker_url=reservation.decode_worker_url,
                ):
                    available[index] = max(available[index], reservation.decode_end_ms)
                    break
        return available

    def _base_decode_intervals(self, state: PlannerClusterState, decode_count: int):
        if state.decode_scheduled_intervals:
            return [
                [
                    tuple(item)
                    if isinstance(item, (list, tuple))
                    else (item.start_ms, item.end_ms, item.demand_tokens)
                    for item in bucket
                ]
                for bucket in state.decode_scheduled_intervals
            ]
        return [[] for _ in range(decode_count)]

    def _decode_intervals(self, request: PlannerScheduleRequest):
        current_time_ms = int(request.state.current_time_ms)
        self._prune_reservations(current_time_ms)
        intervals = self._base_decode_intervals(
            request.state,
            len(request.decode_workers),
        )
        represented = snapshot_request_keys(request, stage="decode")
        for key, reservation in self._reservations.items():
            if key in represented:
                continue
            if reservation.decode_end_ms <= current_time_ms:
                continue
            for index, worker in enumerate(request.decode_workers):
                if self._worker_matches(
                    worker,
                    worker_id=reservation.decode_worker_id,
                    worker_url=reservation.decode_worker_url,
                ):
                    intervals[index].append(
                        (
                            reservation.decode_start_ms,
                            reservation.decode_end_ms,
                            reservation.decode_demand_tokens,
                        )
                    )
                    break
        return intervals

    def _request_with_live_cluster(
        self,
        request: PlannerScheduleRequest,
        cluster_source: PlannerScheduleRequest,
    ) -> PlannerScheduleRequest:
        return replace(
            request,
            prefill_workers=cluster_source.prefill_workers,
            decode_workers=cluster_source.decode_workers,
            state=cluster_source.state,
        )

    def _request_at_latest_event_time(
        self, request: PlannerScheduleRequest
    ) -> PlannerScheduleRequest:
        """Advance stale admission state to the latest worker lifecycle event."""

        current_time_ms = max(
            int(request.state.current_time_ms), int(self._latest_event_time_ms)
        )
        if current_time_ms == int(request.state.current_time_ms):
            return request
        return replace(
            request,
            state=replace(request.state, current_time_ms=current_time_ms),
        )

    def _compatible_pending_requests(
        self, cluster_source: PlannerScheduleRequest
    ) -> List[PlannerScheduleRequest]:
        prefill_urls = {worker.url for worker in cluster_source.prefill_workers}
        decode_urls = {worker.url for worker in cluster_source.decode_workers}
        model = cluster_source.model
        route = cluster_source.route
        compatible = []
        for request in self._pending_admissions.values():
            if request.model != model or request.route != route:
                continue
            request_prefill_urls = {worker.url for worker in request.prefill_workers}
            request_decode_urls = {worker.url for worker in request.decode_workers}
            if not request_prefill_urls.issubset(prefill_urls):
                continue
            if not request_decode_urls.issubset(decode_urls):
                continue
            compatible.append(self._request_with_live_cluster(request, cluster_source))
        return compatible

    def _active_prefill_reservation_counts(
        self, request: PlannerScheduleRequest
    ) -> List[int]:
        current_time_ms = int(request.state.current_time_ms)
        self._prune_reservations(current_time_ms)
        counts = [0 for _ in request.prefill_workers]
        represented = snapshot_request_keys(request, stage="prefill")
        represented.update(snapshot_request_keys(request, stage="decode"))
        for key, reservation in self._reservations.items():
            if key in represented:
                continue
            if reservation.prefill_end_ms <= current_time_ms:
                continue
            for index, worker in enumerate(request.prefill_workers):
                if self._worker_matches(
                    worker,
                    worker_id=reservation.prefill_worker_id,
                    worker_url=reservation.prefill_worker_url,
                ):
                    counts[index] += 1
                    break
        return counts

    def _snapshot_prefill_waiting_counts(
        self, request: PlannerScheduleRequest
    ) -> List[int]:
        counts = [0 for _ in request.prefill_workers]
        for record in waiting_prefill_calls(request):
            if 0 <= record.worker_index < len(counts):
                counts[record.worker_index] += 1
        return counts

    @staticmethod
    def _revealed_child_count(workflow, call_id):
        return sum(call_id in call.parent_call_ids for call in workflow.revealed_calls.values())

    @staticmethod
    def _raw_latency_vectors(latency):
        def vector(raw, calendar):
            return tuple(float(value) for value in (calendar if raw is None else raw))
        return (
            vector(latency.prefill_ms_unrounded_by_instance, latency.prefill_ms_by_instance),
            vector(latency.decode_ms_unrounded_by_instance, latency.decode_ms_by_instance),
            vector(latency.transfer_ms_unrounded_by_decode_instance, latency.transfer_ms_by_decode_instance),
        )

    def _priority_seconds(self, workflow, request, latency):
        prefill, decode, transfer = self._raw_latency_vectors(latency)
        return PrioritySeconds(
            float(request.state.current_time_ms) / 1000.0,
            float(workflow.arrival_ms if workflow.arrival_ms_unrounded is None else workflow.arrival_ms_unrounded) / 1000.0,
            float(workflow.horizon_ms if workflow.horizon_ms_unrounded is None else workflow.horizon_ms_unrounded) / 1000.0,
            float(workflow.completion_ms_unrounded or workflow.completion_ms) / 1000.0,
            tuple(value / 1000.0 for value in prefill),
            tuple(value / 1000.0 for value in decode),
            tuple(value / 1000.0 for value in transfer),
        )

    def _refresh_call_priority_views(self, calls):
        # Finish the revealed generation before freezing any call's denominator.
        for call in calls:
            workflow = self._workflows[call.workflow_id]
            call.workflow_horizon_ms = workflow.horizon_ms
            call.critical_path_ms = self._call_critical_path_ms(workflow, call.call_id)
            call.child_count = self._revealed_child_count(workflow, call.call_id)
            if call.priority_seconds is not None:
                call.priority_seconds = replace(call.priority_seconds,
                    workflow_horizon_s=float(workflow.horizon_ms_unrounded or workflow.horizon_ms) / 1000.0,
                    workflow_completion_s=float(workflow.completion_ms_unrounded or workflow.completion_ms) / 1000.0)

    @staticmethod
    def _decode_candidate_view(call, ready_at_ms):
        values = {name: getattr(call, name) for name in DecodeCallInfo.__dataclass_fields__
                  if hasattr(call, name)}
        values.update(ready_at_ms=int(ready_at_ms),
            decode_demand_tokens=max(1, int(call.decode_demand_tokens or call.input_tokens + call.output_tokens)))
        return DecodeCallInfo(**values)

    @staticmethod
    def _snapshot_request(record, request):
        metadata = replace(request.hexgent,
            workflow_id=record.workflow_id, call_id=record.call_id,
            workflow_arrival_ms=record.workflow_arrival_ms, parent_call_ids=list(record.parent_call_ids),
            input_tokens=record.input_tokens, predicted_output_tokens=record.output_tokens,
            estimated_prefill_time=record.estimated_prefill_time,
            estimated_decode_time=record.estimated_decode_time,
            estimated_transfer_time=record.estimated_transfer_time,
            critical_path_ms=0, child_count=0,
        )
        # Another queued request cannot borrow the incoming call's body/probe/profile.
        return replace(request, hexgent=metadata, prefix_cache_probe=None, request_body=None)

    def _live_stage_states(self, request, prefill_calls=(), decode_calls=()):
        known = {self._decision_key(call): call for call in decode_calls}
        records = [record for record in decode_snapshot_calls(request) if not record.is_running]
        for record in records:
            key = self._reservation_key(record.workflow_id, record.call_id)
            if key not in known:
                known[key] = self._decode_call_from_snapshot(record, request)
        for call in prefill_calls:
            known.setdefault(self._decision_key(call),
                self._decode_candidate_view(call, request.state.current_time_ms))
        self._refresh_call_priority_views([*prefill_calls, *known.values()])
        p_state, d_state = build_live_stage_states(request, decode_calls=list(known.values()),
            prefill_available_at_ms=self._prefill_available(request),
            decode_available_hint_ms=self._decode_available(request))
        represented = snapshot_request_keys(request, stage="decode") | set(known)
        now = int(request.state.current_time_ms)
        # Keep existing committed-but-not-yet-observed plans in the calendar.
        # These are live reservations, not simulated physical allocations.
        self._prune_reservations(now)
        for key, reservation in self._reservations.items():
            if key in represented or reservation.decode_end_ms <= now:
                continue
            for index, worker in enumerate(request.decode_workers):
                if self._worker_matches(worker, worker_id=reservation.decode_worker_id,
                                        worker_url=reservation.decode_worker_url):
                    interval = (reservation.decode_start_ms, reservation.decode_end_ms,
                                reservation.decode_demand_tokens)
                    d_state.scheduled_intervals[index].append(interval)
                    p_state.decode_admission_intervals[index].append(interval)
                    break
        return p_state, d_state

    @staticmethod
    def _decode_admission_start(call, state, worker_index, intervals):
        return find_earliest_fifo_admission_start(
            current_time_ms=int(state.current_time_ms),
            ready_at_ms=max(int(call.ready_at_ms),
                int((state.admission_not_before_ms or (0,) * len(state.capacity_max_tokens))[worker_index])),
            demand_tokens=int(call.decode_demand_tokens), scheduled_intervals=intervals,
            capacity_tokens=int(state.capacity_max_tokens[worker_index]),
            max_batch_size=(state.max_batch_size_by_instance[worker_index]
                if state.max_batch_size_by_instance is not None else None))

    def _decode_bounds(self, calls, decisions, state):
        """Replay every ranked D job, including immutable in-flight transfers."""
        calls_by_key = {self._decision_key(call): call for call in calls}
        intervals = [list(row) for row in state.scheduled_intervals]
        bounds = {}
        for decision in sorted(decisions, key=lambda row: (int(row.priority_score),
                                                           self._decision_key(row))):
            key = self._decision_key(decision)
            call = calls_by_key[key]
            index = int(decision.assigned_instance)
            start = self._decode_admission_start(call, state, index, intervals[index])
            end = start + int(call.decode_ms_by_instance[index])
            intervals[index].append((start, end, int(call.decode_demand_tokens)))
            bounds[key] = (index, start, end)
        return bounds

    def _planning_latency(self, request, cold_latency, *, input_tokens, output_tokens):
        probe = request.prefix_cache_probe
        # Forecast labels and cache counts must use the same token domain.
        if probe is None or len(probe["input_ids"]) != input_tokens:
            return cold_latency
        prefixes = observed_prefix_tokens(request)
        if not any(prefix is not None and prefix > 0 for prefix in prefixes):
            return cold_latency
        estimate_with_prefixes = getattr(self.latency_estimator, "estimate_with_prefixes", None)
        if estimate_with_prefixes is None:
            return cold_latency
        return estimate_with_prefixes(request, input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_prefix_tokens_by_prefill_instance=prefixes)

    def _estimate_latency(
        self,
        request: PlannerScheduleRequest,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> LatencyEstimate:
        return self.latency_estimator.estimate(
            request,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _prefill_call(
        self,
        *,
        request: PlannerScheduleRequest,
        output_tokens: int,
        input_tokens: int,
        workflow: WorkflowRuntimeState,
    ) -> PrefillCallInfo:
        meta = request.hexgent
        latency = self._estimate_latency(
            request,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        cold_latency = latency
        cold_ideal_ms = (min(latency.prefill_ms_by_instance)
            + min(latency.transfer_ms_by_decode_instance or [0])
            + min(latency.decode_ms_by_instance))
        raw_p, raw_d, raw_t = self._raw_latency_vectors(cold_latency)
        self._reveal_call(workflow, meta, cold_ideal_ms,
            min(raw_p) + min(raw_t or (0.0,)) + min(raw_d))
        latency = self._planning_latency(request, cold_latency,
            input_tokens=input_tokens, output_tokens=output_tokens)
        ideal_service_ms = (
            min(latency.prefill_ms_by_instance)
            + min(latency.transfer_ms_by_decode_instance or [0])
            + min(latency.decode_ms_by_instance)
        )
        return PrefillCallInfo(
            call_id=meta.call_id,
            workflow_id=meta.workflow_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            workflow_arrival_ms=workflow.arrival_ms,
            reveal_sequence=int(meta.reveal_sequence),
            workflow_horizon_ms=workflow.horizon_ms,
            workflow_completion_ms=int(meta.workflow_completion_ms or workflow.completion_ms),
            workflow_attained_service_ms=int(
                meta.workflow_attained_service_ms or workflow.attained_service_ms
            ),
            ideal_service_ms=ideal_service_ms,
            prefill_ms_by_instance=latency.prefill_ms_by_instance,
            decode_ms_by_instance=latency.decode_ms_by_instance,
            priority_seconds=self._priority_seconds(workflow, request, latency),
            transfer_ms_by_decode_instance=latency.transfer_ms_by_decode_instance,
            locked_prefill_instance=None,
            locked_decode_instance=None,
            critical_path_ms=self._call_critical_path_ms(
                workflow, meta.call_id, meta.critical_path_ms
            ),
            child_count=self._revealed_child_count(workflow, meta.call_id),
            parent_call_ids=tuple(str(parent) for parent in meta.parent_call_ids),
            decode_demand_tokens=max(1, input_tokens + output_tokens),
        )

    def _prefill_call_from_snapshot(
        self,
        record: SnapshotWaitingCall,
        request: PlannerScheduleRequest,
    ) -> PrefillCallInfo:
        request = self._snapshot_request(record, request)
        workflow = self._workflow_state_values(
            workflow_id=record.workflow_id,
            arrival_ms=record.workflow_arrival_ms,
            horizon_ms=record.workflow_horizon_ms,
        )
        latency = self._estimate_latency(
            request,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
        )
        ideal_service_ms = (
            min(latency.prefill_ms_by_instance)
            + min(latency.transfer_ms_by_decode_instance or [0])
            + min(latency.decode_ms_by_instance)
        )
        if record.call_id not in workflow.revealed_calls:
            raw_p, raw_d, raw_t = self._raw_latency_vectors(latency)
            self._reveal_snapshot_call(
                workflow,
                call_id=record.call_id,
                ideal_service_ms=ideal_service_ms,
                ideal_service_ms_unrounded=min(raw_p) + min(raw_t or (0.0,)) + min(raw_d),
                parent_call_ids=record.parent_call_ids,
            )
        return PrefillCallInfo(
            call_id=record.call_id,
            workflow_id=record.workflow_id,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            workflow_arrival_ms=workflow.arrival_ms,
            reveal_sequence=int(record.reveal_sequence),
            workflow_horizon_ms=workflow.horizon_ms,
            workflow_completion_ms=int(
                record.workflow_completion_ms or workflow.completion_ms
            ),
            workflow_attained_service_ms=int(
                record.workflow_attained_service_ms or workflow.attained_service_ms
            ),
            ideal_service_ms=ideal_service_ms,
            prefill_ms_by_instance=latency.prefill_ms_by_instance,
            decode_ms_by_instance=latency.decode_ms_by_instance,
            priority_seconds=self._priority_seconds(workflow, request, latency),
            transfer_ms_by_decode_instance=latency.transfer_ms_by_decode_instance,
            locked_prefill_instance=int(record.worker_index),
            locked_decode_instance=record.locked_decode_instance,
            critical_path_ms=self._call_critical_path_ms(
                workflow, record.call_id, record.critical_path_ms
            ),
            child_count=self._revealed_child_count(workflow, record.call_id),
            parent_call_ids=tuple(str(parent) for parent in record.parent_call_ids),
            decode_demand_tokens=max(1, record.input_tokens + record.output_tokens),
        )

    def _decode_call_from_snapshot(
        self,
        record: SnapshotWaitingCall,
        request: PlannerScheduleRequest,
    ) -> DecodeCallInfo:
        request = self._snapshot_request(record, request)
        workflow = self._workflow_state_values(
            workflow_id=record.workflow_id,
            arrival_ms=record.workflow_arrival_ms,
            horizon_ms=record.workflow_horizon_ms,
        )
        latency = self._estimate_latency(
            request,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
        )
        cold_ideal_ms = (min(latency.prefill_ms_by_instance)
            + min(latency.transfer_ms_by_decode_instance or [0])
            + min(latency.decode_ms_by_instance))
        if record.call_id not in workflow.revealed_calls:
            raw_p, raw_d, raw_t = self._raw_latency_vectors(latency)
            self._reveal_snapshot_call(
                workflow, call_id=record.call_id, ideal_service_ms=cold_ideal_ms,
                ideal_service_ms_unrounded=min(raw_p) + min(raw_t or (0.0,)) + min(raw_d),
                parent_call_ids=record.parent_call_ids)
        remaining = max(0, record.remaining_service_tokens)
        if record.output_tokens_generated > 0:
            # Forecast the unfinished suffix with its real current context;
            # keep full request lengths and memory demand in the priority view.
            remaining_request = replace(request, hexgent=replace(request.hexgent,
                estimated_decode_time=(None if record.estimated_decode_time is None else
                    record.estimated_decode_time * remaining / max(1, record.output_tokens))))
            latency = self._estimate_latency(remaining_request,
                input_tokens=record.input_tokens + record.output_tokens_generated,
                output_tokens=remaining)
        return DecodeCallInfo(
            call_id=record.call_id,
            workflow_id=record.workflow_id,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            workflow_arrival_ms=workflow.arrival_ms,
            reveal_sequence=int(record.reveal_sequence),
            workflow_horizon_ms=workflow.horizon_ms,
            workflow_completion_ms=int(
                record.workflow_completion_ms or workflow.completion_ms
            ),
            workflow_attained_service_ms=int(
                record.workflow_attained_service_ms or workflow.attained_service_ms
            ),
            ideal_service_ms=cold_ideal_ms,
            ready_at_ms=int(record.ready_at_ms),
            decode_demand_tokens=max(1, record.memory_tokens, record.input_tokens + record.output_tokens),
            decode_ms_by_instance=latency.decode_ms_by_instance,
            priority_seconds=self._priority_seconds(workflow, request, latency),
            locked_decode_instance=record.locked_decode_instance,
            critical_path_ms=self._call_critical_path_ms(
                workflow, record.call_id, record.critical_path_ms
            ),
            child_count=self._revealed_child_count(workflow, record.call_id),
            parent_call_ids=tuple(str(parent) for parent in record.parent_call_ids),
        )

    def schedule_request(
        self,
        request: PlannerScheduleRequest,
        *,
        apply_live_snapshot: bool = True,
        prefill_decision_override=None,
    ) -> PlannerScheduleResponse:
        if not request.prefill_workers:
            raise ValueError("At least one prefill worker is required")
        if not request.decode_workers:
            raise ValueError("At least one decode worker is required")
        if apply_live_snapshot:
            maybe_apply_live_worker_snapshots(request, include_mutable_waiting=False)

        request = self._request_at_latest_event_time(request)
        timing_started_ns = time.perf_counter_ns()
        phase_timings_ms = {
            "prefill_queue_model": 0.0,
            "prefill_solver": 0.0,
            "decode_queue_model": 0.0,
            "decode_solver": 0.0,
            "reservation": 0.0,
        }
        meta = request.hexgent
        workflow = self._workflow_state(meta)
        phase_started_ns = time.perf_counter_ns()
        input_tokens = int(meta.input_tokens or _estimate_input_tokens(request.request_text))
        output_tokens = self.predictor.predict(
            request.request_text,
            predicted_output_tokens=meta.predicted_output_tokens,
            max_tokens=request.max_tokens,
        )
        call = self._prefill_call(
            request=request,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            workflow=workflow,
        )
        request_key = self._reservation_key(meta.workflow_id, meta.call_id)
        waiting_prefill_records = [
            record
            for record in waiting_prefill_calls(request)
            if self._reservation_key(record.workflow_id, record.call_id) != request_key
            and record.queue_name == "prefill_waiting"
        ]
        waiting_prefill_by_call = {
            self._reservation_key(record.workflow_id, record.call_id): record
            for record in waiting_prefill_records
        }
        prefill_calls = [
            self._prefill_call_from_snapshot(record, request)
            for record in waiting_prefill_records
        ]
        prefill_calls.append(call)
        phase_timings_ms["prefill_queue_model"] = (
            time.perf_counter_ns() - phase_started_ns
        ) / 1_000_000.0

        phase_started_ns = time.perf_counter_ns()
        current_time_ms = int(request.state.current_time_ms)
        prefill_available = self._prefill_available(request)
        decode_available = self._decode_available(request)
        if prefill_decision_override is None:
            state, _ = self._live_stage_states(request, prefill_calls)
            prefill_decisions = self.scheduler.schedule_prefill(prefill_calls, state)
            prefill_decision_by_call = {
                self._decision_key(decision): decision for decision in prefill_decisions
            }
            prefill_decision = prefill_decision_by_call.get(request_key)
            if prefill_decision is None:
                raise ValueError("No capacity-feasible decode worker for request")
            phase_timings_ms["prefill_solver"] = (
                time.perf_counter_ns() - phase_started_ns
            ) / 1_000_000.0
        else:
            prefill_decisions = [prefill_decision_override]
            prefill_decision = prefill_decision_override

        phase_started_ns = time.perf_counter_ns()
        decode_index = int(prefill_decision.planned_decode_instance or 0)
        prefill_index = int(prefill_decision.assigned_instance)
        prefill_start_ms = max(current_time_ms, int(prefill_available[prefill_index]))
        prefill_end_ms = prefill_start_ms + int(call.prefill_ms_by_instance[prefill_index])
        transfer_ready_ms = prefill_end_ms + int(
            call.transfer_ms_by_decode_instance[decode_index]
            if decode_index < len(call.transfer_ms_by_decode_instance)
            else 0
        )

        decode_call = DecodeCallInfo(
            call_id=meta.call_id,
            workflow_id=meta.workflow_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            workflow_arrival_ms=workflow.arrival_ms,
            reveal_sequence=int(meta.reveal_sequence),
            workflow_horizon_ms=workflow.horizon_ms,
            workflow_completion_ms=int(meta.workflow_completion_ms or workflow.completion_ms),
            workflow_attained_service_ms=int(
                meta.workflow_attained_service_ms or workflow.attained_service_ms
            ),
            ideal_service_ms=workflow.revealed_calls[meta.call_id].estimated_service_ms,
            ready_at_ms=transfer_ready_ms,
            decode_demand_tokens=max(1, input_tokens + output_tokens),
            decode_ms_by_instance=call.decode_ms_by_instance,
            priority_seconds=call.priority_seconds,
            locked_decode_instance=decode_index,
            critical_path_ms=self._call_critical_path_ms(
                workflow, meta.call_id, meta.critical_path_ms
            ),
            child_count=self._revealed_child_count(workflow, meta.call_id),
            parent_call_ids=tuple(str(parent) for parent in meta.parent_call_ids),
        )
        waiting_decode_records = [
            record
            for record in decode_snapshot_calls(request)
            if not record.is_running
            and self._reservation_key(record.workflow_id, record.call_id) != request_key
        ]
        waiting_decode_by_call = {
            self._reservation_key(record.workflow_id, record.call_id): record
            for record in waiting_decode_records
            if record.queue_name in {"decode_waiting", "decode_retracted"}
        }
        decode_calls = [
            self._decode_call_from_snapshot(record, request)
            for record in waiting_decode_records
        ]
        decode_calls.append(decode_call)
        _, decode_state = self._live_stage_states(request, prefill_calls, decode_calls)
        decode_intervals = decode_state.scheduled_intervals
        phase_timings_ms["decode_queue_model"] = (
            time.perf_counter_ns() - phase_started_ns
        ) / 1_000_000.0

        phase_started_ns = time.perf_counter_ns()
        decode_decisions = self.scheduler.schedule_decode(decode_calls, decode_state)
        decode_decision_by_call = {
            self._decision_key(decision): decision for decision in decode_decisions
        }
        decode_decision = decode_decision_by_call[
            self._reservation_key(meta.workflow_id, meta.call_id)
        ]
        decode_index = int(decode_decision.assigned_instance)
        phase_timings_ms["decode_solver"] = (
            time.perf_counter_ns() - phase_started_ns
        ) / 1_000_000.0

        phase_started_ns = time.perf_counter_ns()
        revision = self._next_revision()
        prefill_worker = request.prefill_workers[prefill_index]
        decode_worker = request.decode_workers[decode_index]
        priority = int(prefill_decision.priority_score)
        decode_priority = int(decode_decision.priority_score)
        _, decode_start_ms, decode_end_ms = self._decode_bounds(
            decode_calls, decode_decisions, decode_state)[request_key]
        self._reservations[self._reservation_key(meta.workflow_id, meta.call_id)] = (
            PlannedCallReservation(
                workflow_id=meta.workflow_id,
                call_id=meta.call_id,
                prefill_worker_id=prefill_worker.id,
                prefill_worker_url=prefill_worker.url,
                prefill_end_ms=prefill_end_ms,
                decode_worker_id=decode_worker.id,
                decode_worker_url=decode_worker.url,
                decode_start_ms=decode_start_ms,
                decode_end_ms=decode_end_ms,
                decode_demand_tokens=int(decode_call.decode_demand_tokens),
            )
        )
        actions: List[WorkerMutationAction] = []
        if _local_priority_updates_enabled():
            for decision in prefill_decisions:
                record = waiting_prefill_by_call.get(self._decision_key(decision))
                if record is None:
                    continue
                current_prefill_worker = request.prefill_workers[record.worker_index]
                actions.append(
                    WorkerMutationAction(
                        worker_url=record.worker_url,
                        request_id=record.request_id,
                        target_queue_name=record.queue_name,
                        priority=int(decision.priority_score),
                        action_type="priority_update",
                        # Placement is locked after router dispatch. A safe
                        # local update therefore preserves the existing decode
                        # target even when the new global optimum differs.
                        planned_decode_instance=None,
                        schedule_revision=revision,
                        expected_revision=record.schedule_revision,
                        target_worker_id=current_prefill_worker.id,
                        target_worker_url=record.worker_url,
                    )
                )
            for decision in decode_decisions:
                record = waiting_decode_by_call.get(self._decision_key(decision))
                if record is None:
                    continue
                current_decode_worker = request.decode_workers[record.worker_index]
                actions.append(
                    WorkerMutationAction(
                        worker_url=record.worker_url,
                        request_id=record.request_id,
                        target_queue_name=record.queue_name,
                        priority=int(decision.priority_score),
                        action_type="priority_update",
                        planned_decode_instance=current_decode_worker.id,
                        schedule_revision=revision,
                        expected_revision=record.schedule_revision,
                        target_worker_id=current_decode_worker.id,
                        target_worker_url=record.worker_url,
                    )
                )
        phase_timings_ms["reservation"] = (
            time.perf_counter_ns() - phase_started_ns
        ) / 1_000_000.0
        phase_timings_ms["total"] = (
            time.perf_counter_ns() - timing_started_ns
        ) / 1_000_000.0
        return PlannerScheduleResponse(
            revision=revision,
            workflow_horizon_ms=int(workflow.horizon_ms),
            prefill_worker_id=prefill_worker.id,
            prefill_worker_url=prefill_worker.url,
            decode_worker_id=decode_worker.id,
            decode_worker_url=decode_worker.url,
            prefill_priority=priority,
            decode_priority=decode_priority,
            planned_decode_instance=decode_worker.id,
            predicted_output_tokens=output_tokens,
            actions=actions,
            phase_timings_ms=phase_timings_ms,
        )

    def enqueue_admission(
        self, admission: PlannerAdmissionRequest, *, refresh_pending: bool = False
    ) -> None:
        """Record an admission without running a solver.

        The sidecar coordinator is the sole caller that drains this map.  This
        keeps an HTTP enqueue cheap even while a previous batch is solving.
        """

        request = admission.request
        key = self._request_key(request)
        cached = self._admission_results.get(key)
        # A retry can arrive after this request already received a concrete
        # dispatch plan. Preserve that plan instead of turning a duplicate
        # enqueue into a second reservation and a different worker choice.
        if refresh_pending:
            # Only an existing coordinator-owned, gateway-confirmed pending
            # admission may replace its cached plan. A dispatched/cancelled
            # identity cannot be resurrected by a late snapshot update.
            if key not in self._pending_admissions:
                return
            self._reservations.pop(key, None)
        if cached is not None and cached.dispatch_ready and not refresh_pending:
            return
        if not request.prefill_workers:
            raise ValueError("At least one prefill worker is required")
        if not request.decode_workers:
            raise ValueError("At least one decode worker is required")
        self._next_admission_order += 1
        self._pending_admissions[key] = request
        self._pending_admission_policies[key] = admission.policy
        self._pending_admission_order[key] = self._next_admission_order
        self._admission_results.pop(key, None)

    def admission_status(
        self, request: PlannerScheduleRequest
    ) -> PlannerAdmissionResponse:
        key = self._request_key(request)
        cached = self._admission_results.get(key)
        if cached is not None:
            return cached
        policy = self._pending_admission_policies.get(key, PlannerAdmissionPolicy())
        return PlannerAdmissionResponse(
            dispatch_ready=False,
            retry_after_ms=max(1, int(policy.retry_after_ms)),
            queue_position=len(self._pending_admissions),
            reason="planning_pending",
        )

    def plan_pending_admissions(
        self,
    ) -> Dict[Tuple[str, str], PlannerAdmissionResponse]:
        """Plan each compatible pending set once using its newest cluster state.

        One invocation computes a complete, revisioned prefill/decode plan for
        each compatible pending set. Router-side live credit controls actual
        dispatch; status reads return cached plans and never re-enter either
        solver.
        """

        results: Dict[Tuple[str, str], PlannerAdmissionResponse] = {}
        seen: set[Tuple[str, str]] = set()
        while True:
            candidates = [
                key
                for key in self._pending_admissions
                if key not in seen
            ]
            if not candidates:
                break
            # All compatible calls use the newest live cluster snapshot rather
            # than the first request that happened to enter the dictionary.
            candidate_key = max(
                candidates,
                key=lambda key: self._pending_admission_order.get(key, 0),
            )
            candidate_request = self._request_at_latest_event_time(
                self._pending_admissions[candidate_key]
            )
            timing_started_ns = time.perf_counter_ns()
            phase_timings_ms = {
                "snapshot": 0.0,
                "pending": 0.0,
                "prefill_rank": 0.0,
                "schedule": 0.0,
            }

            phase_started_ns = time.perf_counter_ns()
            maybe_apply_live_worker_snapshots(
                candidate_request, include_mutable_waiting=False,
                prefix_probes=[pending.prefix_cache_probe
                    for pending in self._compatible_pending_requests(candidate_request)
                    if pending.prefix_cache_probe is not None])
            phase_timings_ms["snapshot"] = (
                time.perf_counter_ns() - phase_started_ns
            ) / 1_000_000.0

            phase_started_ns = time.perf_counter_ns()
            current_time_ms = int(candidate_request.state.current_time_ms)
            self._prune_reservations(current_time_ms)
            represented_dispatches = snapshot_request_keys(
                candidate_request, stage="prefill"
            )
            represented_dispatches.update(
                snapshot_request_keys(candidate_request, stage="decode")
            )
            # A live worker snapshot is an authoritative dispatch boundary.
            # This also repairs planner state if the fire-and-forget router
            # event was delayed or lost.
            for dispatched_key in represented_dispatches:
                self._pending_admissions.pop(dispatched_key, None)
                self._pending_admission_policies.pop(dispatched_key, None)
                self._pending_admission_order.pop(dispatched_key, None)
                self._admission_results.pop(dispatched_key, None)
            pending_requests = self._compatible_pending_requests(candidate_request)
            phase_timings_ms["pending"] = (
                time.perf_counter_ns() - phase_started_ns
            ) / 1_000_000.0
            pending_keys = {
                self._request_key(request) for request in pending_requests
            }
            seen.update(pending_keys)
            if not pending_requests:
                continue

            # Pending requests are speculative until the router grants actual
            # prefill admission. Remove their previous revision's reservations
            # before rebuilding one coherent execution plan for the whole set.
            for key in pending_keys:
                self._reservations.pop(key, None)

            phase_started_ns = time.perf_counter_ns()
            waiting_prefill_records = [
                record
                for record in waiting_prefill_calls(candidate_request)
                if record.queue_name == "prefill_waiting"
            ]
            waiting_prefill_by_call = {
                self._reservation_key(record.workflow_id, record.call_id): record
                for record in waiting_prefill_records
            }
            prefill_calls: List[PrefillCallInfo] = [
                self._prefill_call_from_snapshot(record, candidate_request)
                for record in waiting_prefill_records
            ]
            prefill_calls_by_key: Dict[Tuple[str, str], PrefillCallInfo] = {}
            for pending_request in pending_requests:
                meta = pending_request.hexgent
                workflow = self._workflow_state(meta)
                input_tokens = int(
                    meta.input_tokens or _estimate_input_tokens(pending_request.request_text)
                )
                output_tokens = self.predictor.predict(
                    pending_request.request_text,
                    predicted_output_tokens=meta.predicted_output_tokens,
                    max_tokens=pending_request.max_tokens,
                )
                call = self._prefill_call(
                    request=pending_request,
                    output_tokens=output_tokens,
                    input_tokens=input_tokens,
                    workflow=workflow,
                )
                prefill_calls.append(call)
                key = self._request_key(pending_request)
                prefill_calls_by_key[key] = call

            waiting_decode_records = [
                record
                for record in decode_snapshot_calls(candidate_request)
                if not record.is_running
            ]
            waiting_decode_by_call = {
                self._reservation_key(record.workflow_id, record.call_id): record
                for record in waiting_decode_records
                if record.queue_name in {"decode_waiting", "decode_retracted"}
            }
            waiting_decode_calls_for_plan = [
                self._decode_call_from_snapshot(record, candidate_request)
                for record in waiting_decode_records
            ]
            # Finish revealing this plan's observed pending and waiting calls
            # before either rank uses their workflow denominator. Earlier call
            # objects otherwise retain a partial horizon from construction.
            for call in prefill_calls + waiting_decode_calls_for_plan:
                call.workflow_horizon_ms = self._workflows[call.workflow_id].horizon_ms

            prefill_available = self._prefill_available(candidate_request)
            decode_available = self._decode_available(candidate_request)
            state, _ = self._live_stage_states(candidate_request, prefill_calls,
                waiting_decode_calls_for_plan)
            decisions = self.scheduler.schedule_prefill(prefill_calls, state)
            phase_timings_ms["prefill_rank"] = (
                time.perf_counter_ns() - phase_started_ns
            ) / 1_000_000.0

            active_counts = self._active_prefill_reservation_counts(candidate_request)
            snapshot_counts = self._snapshot_prefill_waiting_counts(candidate_request)
            ordered_decisions = sorted(
                decisions,
                key=lambda decision: (
                    int(decision.priority_score),
                    str(decision.call_id),
                ),
            )
            selected: List[Tuple[Tuple[str, str], object]] = []
            queue_positions: Dict[Tuple[str, str], int] = {}
            per_worker_seen = [0 for _ in candidate_request.prefill_workers]
            for decision in ordered_decisions:
                key = self._decision_key(decision)
                if key not in pending_keys:
                    continue
                worker_index = int(decision.assigned_instance)
                if worker_index < 0 or worker_index >= len(candidate_request.prefill_workers):
                    continue
                queue_positions[key] = (
                    active_counts[worker_index]
                    + snapshot_counts[worker_index]
                    + per_worker_seen[worker_index]
                )
                # Router credit controls dispatch. The planner must still
                # publish placement and local order for every pending call so
                # later arrivals/events can revise the complete execution plan.
                selected.append((key, decision))
                per_worker_seen[worker_index] += 1

            # The global admission queue needs all selected calls to receive a
            # plan promptly. Re-running schedule_request for every selection
            # repeated the decode capacity solver O(selected_calls) times and
            # made otherwise-ready requests wait behind the full batch solve.
            phase_started_ns = time.perf_counter_ns()
            selected_order = sorted(
                selected,
                key=lambda item: (int(item[1].priority_score), str(item[1].call_id)),
            )
            selected_requests = {
                key: self._request_with_live_cluster(
                    self._pending_admissions[key], candidate_request
                )
                for key, _ in selected_order
                if key in self._pending_admissions
            }
            prefill_available_for_batch = self._prefill_available(candidate_request)
            workflows_by_key = {}
            prefill_end_by_key = {}
            decode_calls_by_key = {}
            for key, decision in selected_order:
                request = selected_requests.get(key)
                call = prefill_calls_by_key.get(key)
                if request is None or call is None:
                    continue
                prefill_index = int(decision.assigned_instance)
                prefill_start_ms = max(
                    current_time_ms, int(prefill_available_for_batch[prefill_index])
                )
                prefill_end_ms = prefill_start_ms + int(
                    call.prefill_ms_by_instance[prefill_index]
                )
                prefill_available_for_batch[prefill_index] = prefill_end_ms
                decode_index = int(
                    decision.planned_decode_instance
                    if decision.planned_decode_instance is not None
                    else 0
                )
                transfer_ready_ms = prefill_end_ms + int(
                    call.transfer_ms_by_decode_instance[decode_index]
                    if decode_index < len(call.transfer_ms_by_decode_instance)
                    else 0
                )
                workflow = self._workflow_state(request.hexgent)
                workflows_by_key[key] = workflow
                prefill_end_by_key[key] = prefill_end_ms
                decode_calls_by_key[key] = DecodeCallInfo(
                    call_id=request.hexgent.call_id,
                    workflow_id=request.hexgent.workflow_id,
                    input_tokens=int(call.input_tokens),
                    output_tokens=int(call.output_tokens),
                    workflow_arrival_ms=workflow.arrival_ms,
                    reveal_sequence=int(request.hexgent.reveal_sequence),
                    workflow_horizon_ms=workflow.horizon_ms,
                    workflow_completion_ms=int(
                        request.hexgent.workflow_completion_ms or workflow.completion_ms
                    ),
                    workflow_attained_service_ms=int(
                        request.hexgent.workflow_attained_service_ms
                        or workflow.attained_service_ms
                    ),
                    ideal_service_ms=workflow.revealed_calls[request.hexgent.call_id].estimated_service_ms,
                    ready_at_ms=transfer_ready_ms,
                    decode_demand_tokens=max(
                        1, int(call.input_tokens) + int(call.output_tokens)
                    ),
                    decode_ms_by_instance=call.decode_ms_by_instance,
                    priority_seconds=call.priority_seconds,
                    locked_decode_instance=decode_index,
                    critical_path_ms=self._call_critical_path_ms(
                        workflow,
                        request.hexgent.call_id,
                        request.hexgent.critical_path_ms,
                    ),
                    child_count=self._revealed_child_count(workflow, request.hexgent.call_id),
                    parent_call_ids=tuple(
                        str(parent) for parent in request.hexgent.parent_call_ids
                    ),
                )

            decode_queue_started_ns = time.perf_counter_ns()
            all_decode_calls = waiting_decode_calls_for_plan + list(decode_calls_by_key.values())
            _, decode_state = self._live_stage_states(candidate_request, prefill_calls, all_decode_calls)
            decode_intervals = decode_state.scheduled_intervals
            decode_decisions = self.scheduler.schedule_decode(all_decode_calls, decode_state)
            decode_solver_ms = (
                time.perf_counter_ns() - decode_queue_started_ns
            ) / 1_000_000.0
            decode_decisions_by_call = {
                self._decision_key(decision): decision for decision in decode_decisions
            }
            decode_bounds_by_key = self._decode_bounds(
                all_decode_calls, decode_decisions, decode_state)
            reservation_started_ns = time.perf_counter_ns()
            revision = self._next_revision()
            priority_actions: List[WorkerMutationAction] = []
            if _local_priority_updates_enabled():
                for prefill_decision in decisions:
                    record = waiting_prefill_by_call.get(
                        self._decision_key(prefill_decision)
                    )
                    if record is None:
                        continue
                    current_worker = candidate_request.prefill_workers[record.worker_index]
                    priority_actions.append(
                        WorkerMutationAction(
                            worker_url=record.worker_url,
                            request_id=record.request_id,
                            target_queue_name=record.queue_name,
                            priority=int(prefill_decision.priority_score),
                            action_type="priority_update",
                            schedule_revision=revision,
                            expected_revision=record.schedule_revision,
                            target_worker_id=current_worker.id,
                            target_worker_url=current_worker.url,
                        )
                    )
                for decode_decision in decode_decisions:
                    record = waiting_decode_by_call.get(
                        self._decision_key(decode_decision)
                    )
                    if record is None:
                        continue
                    current_worker = candidate_request.decode_workers[record.worker_index]
                    priority_actions.append(
                        WorkerMutationAction(
                            worker_url=record.worker_url,
                            request_id=record.request_id,
                            target_queue_name=record.queue_name,
                            priority=int(decode_decision.priority_score),
                            action_type="priority_update",
                            planned_decode_instance=current_worker.id,
                            schedule_revision=revision,
                            expected_revision=record.schedule_revision,
                            target_worker_id=current_worker.id,
                            target_worker_url=current_worker.url,
                        )
                    )
            schedule_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
            for key, decision in selected_order:
                request = selected_requests.get(key)
                call = prefill_calls_by_key.get(key)
                decode_bounds = decode_bounds_by_key.get(key)
                workflow = workflows_by_key.get(key)
                if request is None or call is None or decode_bounds is None or workflow is None:
                    continue
                decode_index, decode_start_ms, decode_end_ms = decode_bounds
                prefill_index = int(decision.assigned_instance)
                prefill_worker = candidate_request.prefill_workers[prefill_index]
                decode_worker = candidate_request.decode_workers[decode_index]
                decode_decision = decode_decisions_by_call[key]
                self._reservations[
                    self._reservation_key(
                        request.hexgent.workflow_id, request.hexgent.call_id
                    )
                ] = PlannedCallReservation(
                    workflow_id=request.hexgent.workflow_id,
                    call_id=request.hexgent.call_id,
                    prefill_worker_id=prefill_worker.id,
                    prefill_worker_url=prefill_worker.url,
                    prefill_end_ms=prefill_end_by_key[key],
                    decode_worker_id=decode_worker.id,
                    decode_worker_url=decode_worker.url,
                    decode_start_ms=decode_start_ms,
                    decode_end_ms=decode_end_ms,
                    decode_demand_tokens=int(
                        decode_calls_by_key[key].decode_demand_tokens
                    ),
                )
                schedule = PlannerScheduleResponse(
                    revision=revision,
                    workflow_horizon_ms=int(workflow.horizon_ms),
                    prefill_worker_id=prefill_worker.id,
                    prefill_worker_url=prefill_worker.url,
                    decode_worker_id=decode_worker.id,
                    decode_worker_url=decode_worker.url,
                    prefill_priority=int(decision.priority_score),
                    decode_priority=int(decode_decision.priority_score),
                    planned_decode_instance=decode_worker.id,
                    predicted_output_tokens=int(call.output_tokens),
                    # Every waiter receives the revision-wide action batch.
                    # Router-side per-request revision gates deduplicate RPCs
                    # without relying on one possibly-dispatched owner.
                    actions=list(priority_actions),
                    phase_timings_ms={},
                )
                result_timings = dict(phase_timings_ms)
                result_timings["schedule"] = schedule_ms
                result_timings["total"] = (
                    time.perf_counter_ns() - timing_started_ns
                ) / 1_000_000.0
                result = PlannerAdmissionResponse(
                    dispatch_ready=True,
                    retry_after_ms=0,
                    queue_position=queue_positions.get(key, 0),
                    reason="dispatch_ready",
                    schedule=schedule,
                    phase_timings_ms=result_timings,
                )
                self._admission_results[key] = result
                results[key] = result

            reservation_ms = (
                time.perf_counter_ns() - reservation_started_ns
            ) / 1_000_000.0
            schedule_phase_timings = {
                # The pending batch already constructed all queue-model inputs
                # before the one global prefill rank. Preserve the public
                # per-schedule timing schema while exposing that work once.
                "prefill_queue_model": 0.0,
                "prefill_solver": phase_timings_ms["prefill_rank"],
                "decode_queue_model": 0.0,
                "decode_solver": decode_solver_ms,
                "reservation": reservation_ms,
                "total": phase_timings_ms["prefill_rank"]
                + decode_solver_ms
                + reservation_ms,
            }
            for result in results.values():
                if result.schedule is not None:
                    result.schedule.phase_timings_ms = dict(schedule_phase_timings)

        return results

    def admit_request(self, admission: PlannerAdmissionRequest) -> PlannerAdmissionResponse:
        """Synchronous compatibility entrypoint used by legacy callers/tests."""

        self.enqueue_admission(admission)
        self.plan_pending_admissions()
        return self.admission_status(admission.request)

    def handle_event(self, event: PlannerEvent) -> Dict[str, object]:
        if event.timestamp_ms is not None:
            self._latest_event_time_ms = max(
                self._latest_event_time_ms, int(event.timestamp_ms)
            )
        admission_key = (
            self._reservation_key(event.workflow_id, event.call_id)
            if event.workflow_id and event.call_id
            else None
        )
        if (
            event.workflow_id
            and event.call_id
            and event.event_type in {"admission_cancelled", "request_cancelled"}
        ):
            # A request_cancelled event is emitted only after a concrete plan
            # may have been published. Unlike an admission timeout, it must
            # remove the outstanding decode reservation as well.
            if event.event_type == "request_cancelled":
                self._reservations.pop(admission_key, None)
            self._pending_admissions.pop(admission_key, None)
            self._pending_admission_policies.pop(admission_key, None)
            self._pending_admission_order.pop(admission_key, None)
            self._admission_results.pop(admission_key, None)
        elif (
            event.workflow_id
            and event.call_id
            and event.event_type == "request_dispatched"
        ):
            admission_key = self._reservation_key(event.workflow_id, event.call_id)
            # Placement becomes immutable at bootstrap creation. Keep the
            # reservation for capacity accounting, but stop globally
            # replanning this request.
            self._pending_admissions.pop(admission_key, None)
            self._pending_admission_policies.pop(admission_key, None)
            self._pending_admission_order.pop(admission_key, None)
            self._admission_results.pop(admission_key, None)
        elif event.workflow_id and event.call_id and event.event_type == "prefill_completed":
            reservation = self._reservations.get(admission_key)
            if reservation is not None and event.timestamp_ms is not None:
                actual_prefill_end_ms = max(0, int(event.timestamp_ms))
                decode_duration_ms = max(
                    0, reservation.decode_end_ms - reservation.decode_start_ms
                )
                reservation.prefill_end_ms = actual_prefill_end_ms
                reservation.decode_start_ms = max(
                    reservation.decode_start_ms, actual_prefill_end_ms
                )
                reservation.decode_end_ms = max(
                    reservation.decode_end_ms,
                    reservation.decode_start_ms + decode_duration_ms,
                )
        elif event.workflow_id and event.call_id and event.event_type == "call_completed":
            self._reservations.pop(admission_key, None)
            self._pending_admissions.pop(admission_key, None)
            self._pending_admission_policies.pop(admission_key, None)
            self._pending_admission_order.pop(admission_key, None)
            self._admission_results.pop(admission_key, None)
            workflow = self._workflows.get(event.workflow_id)
            if workflow is None or event.call_id not in workflow.revealed_calls:
                fallback_metadata = event.payload.get("hexgent_metadata")
                if isinstance(fallback_metadata, dict):
                    metadata = HexgentRequestMetadata.from_dict(fallback_metadata)
                    if (
                        str(metadata.workflow_id) == str(event.workflow_id)
                        and str(metadata.call_id) == str(event.call_id)
                    ):
                        workflow = self._workflow_state(metadata)
                        measured = event.measured_service_ms
                        if measured is None:
                            measured = event.payload.get("measured_service_ms")
                        try:
                            estimated_service_ms = max(1, int(measured or 1))
                        except (TypeError, ValueError):
                            estimated_service_ms = 1
                        self._reveal_call(
                            workflow,
                            metadata,
                            ideal_service_ms=estimated_service_ms,
                        )
            if workflow is not None:
                workflow.completed_calls.add(event.call_id)
                revealed_call = workflow.revealed_calls.get(event.call_id)
                if revealed_call is not None:
                    measured_service_ms = event.measured_service_ms
                    if measured_service_ms is None:
                        measured_service_ms = event.payload.get("measured_service_ms")
                    try:
                        parsed_measured_service_ms = float(measured_service_ms)
                    except (TypeError, ValueError):
                        parsed_measured_service_ms = 0.0
                    if parsed_measured_service_ms > 0:
                        revealed_call.measured_service_ms_unrounded = parsed_measured_service_ms
                        revealed_call.measured_service_ms = max(1, int(parsed_measured_service_ms))
                    observed_wall_ms = event.end_to_end_latency_ms
                    if observed_wall_ms is None:
                        observed_wall_ms = event.payload.get("end_to_end_latency_ms")
                    if observed_wall_ms is None:
                        observed_wall_ms = event.service_ms
                    try:
                        parsed_observed_wall_ms = int(observed_wall_ms)
                    except (TypeError, ValueError):
                        parsed_observed_wall_ms = 0
                    if parsed_observed_wall_ms > 0:
                        revealed_call.end_to_end_latency_ms = parsed_observed_wall_ms
                    revealed_call.completed = True
                    workflow.attained_service_ms = self._completed_critical_path_ms(
                        workflow
                    )
                    self._refresh_workflow_horizon(workflow)
                if event.timestamp_ms is not None:
                    workflow.completion_ms_unrounded = max(workflow.completion_ms_unrounded,
                        float(event.timestamp_ms) - float(workflow.arrival_ms_unrounded or workflow.arrival_ms))
                    workflow.completion_ms = max(
                        workflow.completion_ms,
                        int(event.timestamp_ms or time.time() * 1000) - workflow.arrival_ms,
                    )
            measured_service_ms = event.measured_service_ms
            if measured_service_ms is None:
                measured_service_ms = event.payload.get("measured_service_ms")
            self.scheduler.on_call_completed(
                event.call_id,
                event.workflow_id,
                int(measured_service_ms or 0),
            )
        elif event.call_id and event.event_type == "call_completed":
            for key, reservation in list(self._reservations.items()):
                if reservation.call_id == event.call_id:
                    self._reservations.pop(key, None)
        return self.health()
