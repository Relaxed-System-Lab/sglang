"""Wire-level schemas for the HexAgenT planner sidecar.

The Rust router and the Python planner exchange plain JSON. These dataclasses
keep the Python side explicit while remaining easy to convert from/to dicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Literal, Optional


def _coerce_dataclass(cls, value):
    if value is None or isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls.from_dict(value)
    raise TypeError(f"Cannot coerce {type(value)!r} to {cls.__name__}")


def _drop_unknown(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in payload.items() if key in names}


@dataclass
class HexgentRequestMetadata:
    workflow_id: str
    call_id: str
    workflow_arrival_ms: int = 0
    # Deprecated client input retained for wire compatibility. Normal online
    # scheduling derives horizon from the currently revealed DAG and ignores a
    # caller-supplied value.
    workflow_horizon_ms: Optional[int] = None
    # Parents are the dependency edges known when this call is revealed. This
    # must not contain unrevealed descendants or a full future workflow DAG.
    parent_call_ids: List[str] = field(default_factory=list)
    reveal_sequence: int = 0
    critical_path_ms: int = 0
    child_count: int = 0
    predicted_output_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    workflow_completion_ms: int = 0
    workflow_attained_service_ms: int = 0
    # Optional measured service times for this input/output length.  These
    # values are supplied by the replay profile and override the estimator's
    # absolute single-request calibration while roofline still models batch
    # occupancy when model parameters are available.
    estimated_prefill_time: Optional[float] = None
    estimated_decode_time: Optional[float] = None
    estimated_transfer_time: Optional[float] = None
    model_config: Optional[Dict[str, Any]] = None
    hardware_params: Optional[Dict[str, Any]] = None
    hardware_params_by_name: Optional[Dict[str, Dict[str, Any]]] = None
    roofline: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HexgentRequestMetadata":
        return cls(**_drop_unknown(cls, payload))


@dataclass
class WorkerDescriptor:
    id: str
    url: str
    worker_type: Literal["prefill", "decode"]
    hardware: Optional[str] = None
    hardware_params: Optional[Dict[str, Any]] = None
    model_config: Optional[Dict[str, Any]] = None
    roofline: Optional[Dict[str, Any]] = None
    tp_size: int = 1
    weight_bits: int = 16
    activation_bits: int = 16
    kv_bits: int = 16
    available_at_ms: int = 0
    capacity_tokens: int = 1_000_000
    prefill_base_ms: float = 2.0
    prefill_ms_per_input_token: float = 0.5
    decode_base_ms: float = 1.0
    decode_ms_per_output_token: float = 10.0
    transfer_ms_per_token: float = 0.0
    # Optional limits used when a worker exposes prefill batching through
    # router metadata rather than the live snapshot.
    prefill_batch_token_limit: int = 0
    prefill_batch_request_limit: int = 0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "WorkerDescriptor":
        return cls(**_drop_unknown(cls, payload))


@dataclass
class ScheduledInterval:
    start_ms: int
    end_ms: int
    demand_tokens: int

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScheduledInterval":
        return cls(**_drop_unknown(cls, payload))


@dataclass
class PlannerClusterState:
    current_time_ms: int = 0
    prefill_available_at_ms: List[int] = field(default_factory=list)
    decode_available_at_ms: List[int] = field(default_factory=list)
    # Number of sequences currently executing on each decode worker. This is
    # derived from live worker snapshots and used by the roofline estimator.
    decode_running_batch_sizes: List[int] = field(default_factory=list)
    decode_scheduled_intervals: List[List[ScheduledInterval]] = field(default_factory=list)
    worker_snapshots: Dict[str, Any] = field(default_factory=dict)
    # Optional explicit runtime limits for callers supplying a calendar directly.
    decode_max_batch_size_by_instance: Optional[List[Optional[int]]] = None
    decode_admission_not_before_ms: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannerClusterState":
        payload = _drop_unknown(cls, payload)
        intervals = payload.get("decode_scheduled_intervals")
        if intervals is not None:
            payload["decode_scheduled_intervals"] = [
                [
                    ScheduledInterval.from_dict(item) if isinstance(item, dict) else item
                    for item in bucket
                ]
                for bucket in intervals
            ]
        return cls(**payload)


@dataclass
class PlannerScheduleRequest:
    route: str
    model: Optional[str]
    request_text: str
    max_tokens: Optional[int]
    hexgent: HexgentRequestMetadata
    prefill_workers: List[WorkerDescriptor]
    decode_workers: List[WorkerDescriptor]
    state: PlannerClusterState = field(default_factory=PlannerClusterState)
    # Exact tokenizer IDs and runtime namespace; omitted by legacy callers.
    prefix_cache_probe: Optional[Dict[str, Any]] = None
    # Original JSON for backend-compatible chat-template tokenization.
    request_body: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannerScheduleRequest":
        payload = _drop_unknown(cls, payload)
        payload["hexgent"] = _coerce_dataclass(
            HexgentRequestMetadata, payload.get("hexgent")
        )
        payload["prefill_workers"] = [
            WorkerDescriptor.from_dict(item) if isinstance(item, dict) else item
            for item in payload.get("prefill_workers", [])
        ]
        payload["decode_workers"] = [
            WorkerDescriptor.from_dict(item) if isinstance(item, dict) else item
            for item in payload.get("decode_workers", [])
        ]
        payload["state"] = _coerce_dataclass(
            PlannerClusterState, payload.get("state") or {}
        )
        return cls(**payload)


@dataclass
class PlannerAdmissionPolicy:
    max_prefill_queue_per_worker: int = 1
    retry_after_ms: int = 20

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannerAdmissionPolicy":
        return cls(**_drop_unknown(cls, payload))


@dataclass
class PlannerAdmissionRequest:
    request: PlannerScheduleRequest
    policy: PlannerAdmissionPolicy = field(default_factory=PlannerAdmissionPolicy)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannerAdmissionRequest":
        payload = _drop_unknown(cls, payload)
        payload["request"] = _coerce_dataclass(
            PlannerScheduleRequest, payload.get("request")
        )
        payload["policy"] = _coerce_dataclass(
            PlannerAdmissionPolicy, payload.get("policy") or {}
        )
        return cls(**payload)


@dataclass
class WorkerMutationAction:
    worker_url: str
    request_id: str
    target_queue_name: str
    priority: int
    # Priority-only updates are safe after dispatch because they neither move
    # KV state nor change the established prefill/decode bootstrap pairing.
    action_type: str = "priority_update"
    planned_decode_instance: Optional[str] = None
    schedule_revision: int = 0
    expected_revision: Optional[int] = None
    target_worker_id: Optional[str] = None
    target_worker_url: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "WorkerMutationAction":
        return cls(**_drop_unknown(cls, payload))


@dataclass
class PlannerScheduleResponse:
    revision: int
    # The planner owns horizon construction from the revealed workflow DAG.
    # Return the current value so the router can propagate it to worker state.
    workflow_horizon_ms: int
    prefill_worker_id: str
    prefill_worker_url: str
    decode_worker_id: str
    decode_worker_url: str
    prefill_priority: int
    decode_priority: int
    planned_decode_instance: str
    predicted_output_tokens: int
    actions: List[WorkerMutationAction] = field(default_factory=list)
    fallback_reason: Optional[str] = None
    phase_timings_ms: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannerScheduleResponse":
        payload = _drop_unknown(cls, payload)
        payload["actions"] = [
            WorkerMutationAction.from_dict(item) if isinstance(item, dict) else item
            for item in payload.get("actions", [])
        ]
        return cls(**payload)


@dataclass
class PlannerAdmissionResponse:
    dispatch_ready: bool
    retry_after_ms: int
    queue_position: int
    reason: str
    schedule: Optional[PlannerScheduleResponse] = None
    phase_timings_ms: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannerAdmissionResponse":
        payload = _drop_unknown(cls, payload)
        payload["schedule"] = _coerce_dataclass(
            PlannerScheduleResponse, payload.get("schedule")
        )
        return cls(**payload)


@dataclass
class PlannerRevisionPlanEntry:
    workflow_id: str
    call_id: str
    schedule: PlannerScheduleResponse


@dataclass
class PlannerRevisionPlanResponse:
    revision: int
    entries: List[PlannerRevisionPlanEntry] = field(default_factory=list)
    reason: str = "plan_ready"


@dataclass
class PlannerEvent:
    event_type: str
    workflow_id: Optional[str] = None
    call_id: Optional[str] = None
    # Deprecated compatibility field. Older routers may populate it with
    # client-visible elapsed wall time, which includes waiting and must not be
    # used as an exclusive service-time measurement.
    service_ms: Optional[int] = None
    end_to_end_latency_ms: Optional[int] = None
    # Worker-stage telemetry: prefill + transfer + decode, excluding queueing.
    # Only this value can update the planner's service-time/horizon model.
    measured_service_ms: Optional[int] = None
    timestamp_ms: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannerEvent":
        return cls(**_drop_unknown(cls, payload))


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            key: to_jsonable(item)
            for key, item in asdict(value).items()
            if item is not None
        }
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value
