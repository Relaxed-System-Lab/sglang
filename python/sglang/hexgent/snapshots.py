"""Worker snapshot helpers for HexAgenT live-state planning."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sglang.hexgent.schemas import PlannerScheduleRequest, WorkerDescriptor


@dataclass
class SnapshotWaitingCall:
    request_id: str
    call_id: str
    workflow_id: str
    worker_url: str
    worker_index: int
    queue_name: str
    schedule_revision: int
    input_tokens: int
    output_tokens: int
    workflow_arrival_ms: int
    workflow_horizon_ms: int
    reveal_sequence: int
    workflow_completion_ms: int
    workflow_attained_service_ms: int
    critical_path_ms: int
    child_count: int
    ready_at_ms: int
    locked_decode_instance: Optional[int] = None
    parent_call_ids: Tuple[str, ...] = ()
    is_running: bool = False
    is_mutable_waiting: bool = False
    remaining_service_tokens: int = 0
    memory_tokens: int = 0
    output_tokens_generated: int = 0
    estimated_prefill_time: Optional[float] = None
    estimated_decode_time: Optional[float] = None
    estimated_transfer_time: Optional[float] = None


def _queue_items(
    snapshot: Dict[str, Any],
    *,
    include_mutable_waiting: bool = True,
) -> Iterable[Dict[str, Any]]:
    for queue_key, items in (snapshot.get("queues") or {}).items():
        for item in items or []:
            if isinstance(item, dict):
                queue_name = str(item.get("queue_name") or queue_key)
                if not include_mutable_waiting and queue_name.endswith("_waiting"):
                    continue
                yield item
    for item in snapshot.get("running") or []:
        if isinstance(item, dict):
            yield item


def _worker_snapshot_items(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    snapshots = payload.get("snapshots")
    if isinstance(snapshots, list):
        for snapshot in snapshots:
            if isinstance(snapshot, dict):
                yield snapshot
        return
    snapshot = payload.get("snapshot")
    if isinstance(snapshot, dict):
        yield snapshot
        return
    if isinstance(payload, dict):
        yield payload


def snapshot_request_keys(
    request: PlannerScheduleRequest,
    *,
    stage: str,
) -> set[Tuple[str, str]]:
    """Return workflow/call identities represented by a live stage snapshot."""

    workers = request.prefill_workers if stage == "prefill" else request.decode_workers
    snapshots = request.state.worker_snapshots or {}
    keys: set[Tuple[str, str]] = set()
    for worker in workers:
        payload = snapshots.get(worker.url) or snapshots.get(worker.id)
        if not isinstance(payload, dict):
            continue
        for snapshot in _worker_snapshot_items(payload):
            if snapshot.get("mode") not in {None, stage, "unified"}:
                continue
            for item in _queue_items(snapshot):
                workflow_id = str(
                    item.get("workflow_id")
                    or item.get("hexgent_workflow_id")
                    or ""
                )
                call_id = str(
                    item.get("call_id") or item.get("hexgent_call_id") or ""
                )
                if workflow_id and call_id:
                    keys.add((workflow_id, call_id))
    return keys


def _sum_int(items: Iterable[Dict[str, Any]], key: str) -> int:
    total = 0
    for item in items:
        try:
            total += max(0, int(item.get(key) or 0))
        except (TypeError, ValueError):
            continue
    return total


def _snapshot_input_tokens(item: Dict[str, Any]) -> int:
    """Prefer the fixed trace length over tokenizer-derived bookkeeping."""

    for key in ("hexgent_input_tokens", "input_tokens"):
        try:
            value = int(item.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 1


def _snapshot_parent_call_ids(item: Dict[str, Any]) -> Tuple[str, ...]:
    """Read dependency metadata injected into a worker request, if present."""

    raw_parent_ids = item.get("parent_call_ids") or item.get(
        "hexgent_parent_call_ids"
    )
    if isinstance(raw_parent_ids, str):
        return (raw_parent_ids,)
    if isinstance(raw_parent_ids, (list, tuple)):
        return tuple(str(parent_id) for parent_id in raw_parent_ids)
    return ()


def _snapshot_float(item: Dict[str, Any], key: str) -> Optional[float]:
    try:
        value = float(item.get(key))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _snapshot_memory_tokens(item: Dict[str, Any]) -> int:
    try:
        reported = int(item.get("memory_tokens") or 0)
    except (TypeError, ValueError):
        reported = 0
    try:
        output_tokens = int(item.get("output_tokens_generated") or 0)
    except (TypeError, ValueError):
        output_tokens = 0
    return max(reported, _snapshot_input_tokens(item) + max(0, output_tokens))


def _snapshot_transfer_ms(
    request: PlannerScheduleRequest,
    worker: WorkerDescriptor,
    item: Dict[str, Any],
) -> int:
    profiled = _snapshot_float(item, "estimated_transfer_time")
    if profiled is not None:
        return max(1, int(round(profiled)))
    input_tokens = _snapshot_input_tokens(item)
    if worker.transfer_ms_per_token > 0:
        return max(1, int(round(worker.transfer_ms_per_token * input_tokens)))

    # Worker descriptors from the Rust router may not carry a transfer label.
    # Reuse the request's model/hardware metadata for the common-model case.
    try:
        from sglang.hexgent.latency_estimator import (
            RooflineLatencyEstimator,
            _hardware_params,
            _model_params,
        )

        model = _model_params(request, worker)
        hardware = _hardware_params(request, worker)
        if model is not None and hardware is not None:
            estimate = RooflineLatencyEstimator().estimate_transfer_ms(
                model=model,
                hardware=hardware,
                input_tokens=input_tokens,
            )
            if estimate is not None:
                return estimate
    except (ImportError, TypeError, ValueError, ZeroDivisionError):
        pass
    return 0


def _snapshot_queue_items(
    snapshot: Dict[str, Any],
    *,
    include_mutable_waiting: bool,
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for queue_key, items in (snapshot.get("queues") or {}).items():
        for item in items or []:
            if not isinstance(item, dict):
                continue
            queue_name = str(item.get("queue_name") or queue_key)
            if not include_mutable_waiting and queue_name.endswith("_waiting"):
                continue
            yield queue_name, item
    for item in snapshot.get("running") or []:
        if isinstance(item, dict):
            yield "running", item


def _estimate_prefill_available_ms(
    worker: WorkerDescriptor,
    snapshot_payload: Dict[str, Any],
    current_time_ms: int,
    *,
    request: PlannerScheduleRequest,
    include_mutable_waiting: bool,
) -> int:
    queued_tokens = 0
    queued_count = 0
    work_ms = 0.0
    for snapshot in _worker_snapshot_items(snapshot_payload):
        if snapshot.get("mode") not in {None, "prefill", "unified"}:
            continue
        items = list(
            _queue_items(
                snapshot,
                include_mutable_waiting=include_mutable_waiting,
            )
        )
        queued_tokens += sum(_snapshot_input_tokens(item) for item in items)
        queued_count += len(items)
        limits = snapshot.get("limits") or {}
        try:
            token_limit = int(
                limits.get("max_prefill_tokens")
                or worker.prefill_batch_token_limit
                or 0
            )
        except (TypeError, ValueError):
            token_limit = 0
        try:
            request_limit = int(
                limits.get("prefill_max_requests")
                or worker.prefill_batch_request_limit
                or 0
            )
        except (TypeError, ValueError):
            request_limit = 0

        if token_limit > 0 or request_limit > 0:
            # Requests in one prefill batch share the forward pass. Pack the
            # reported queue in scheduler order and charge the slowest request
            # in each batch. This is a capacity estimate, not a kernel model.
            batches: List[List[Dict[str, Any]]] = []
            current_batch: List[Dict[str, Any]] = []
            current_tokens = 0
            for item in items:
                item_tokens = _snapshot_input_tokens(item)
                exceeds_tokens = token_limit > 0 and current_batch and (
                    current_tokens + item_tokens > token_limit
                )
                exceeds_requests = request_limit > 0 and current_batch and (
                    len(current_batch) >= request_limit
                )
                if exceeds_tokens or exceeds_requests:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                current_batch.append(item)
                current_tokens += item_tokens
            if current_batch:
                batches.append(current_batch)

            for batch in batches:
                work_ms += _estimate_prefill_batch_ms(request, worker, batch)
        else:
            for item in items:
                profiled = _snapshot_float(item, "estimated_prefill_time")
                if profiled is not None:
                    work_ms += profiled
                else:
                    work_ms += float(worker.prefill_base_ms)
                    work_ms += _snapshot_input_tokens(item) * float(
                        worker.prefill_ms_per_input_token
                    )
    if queued_count and work_ms == 0.0:
        work_ms = queued_count * float(worker.prefill_base_ms)
        work_ms += queued_tokens * float(worker.prefill_ms_per_input_token)
    return int(round(current_time_ms + max(0.0, work_ms)))


def _estimate_prefill_batch_ms(
    request: PlannerScheduleRequest,
    worker: WorkerDescriptor,
    batch: List[Dict[str, Any]],
) -> float:
    """Estimate one queued prefill batch using roofline plus calibration."""

    if not batch:
        return 0.0
    token_counts = [_snapshot_input_tokens(item) for item in batch]
    try:
        from sglang.hexgent.latency_estimator import (
            RooflineLatencyEstimator,
            _hardware_params,
            _model_params,
        )

        model = _model_params(request, worker)
        hardware = _hardware_params(request, worker)
        if model is not None and hardware is not None:
            estimator = RooflineLatencyEstimator()
            batch_ms = float(
                estimator.estimate_prefill_batch_ms(
                    model=model,
                    hardware=hardware,
                    input_tokens=token_counts,
                )
            )
            # Profiled rows correct the roofline's absolute error while the
            # roofline preserves the batch-shape scaling. Use the mean scale
            # because each request contributes work to the same batch.
            scales = []
            for item, input_tokens in zip(batch, token_counts):
                profiled = _snapshot_float(item, "estimated_prefill_time")
                if profiled is None:
                    continue
                single_ms = estimator.estimate_prefill_ms(
                    model=model,
                    hardware=hardware,
                    input_tokens=input_tokens,
                )
                if single_ms > 0:
                    scales.append(profiled / float(single_ms))
            if scales:
                return batch_ms * (sum(scales) / len(scales))
            return batch_ms
    except (ImportError, TypeError, ValueError, ZeroDivisionError):
        pass

    # Without model/hardware metadata, a serial calibrated sum is the only
    # conservative estimate. Do not retain the old max-item approximation.
    return sum(
        _snapshot_float(item, "estimated_prefill_time")
        or float(worker.prefill_base_ms)
        + _snapshot_input_tokens(item) * float(worker.prefill_ms_per_input_token)
        for item in batch
    )


def _estimate_decode_state(
    worker: WorkerDescriptor,
    snapshot_payload: Dict[str, Any],
    current_time_ms: int,
    *,
    request: PlannerScheduleRequest,
    include_mutable_waiting: bool,
) -> Tuple[int, int, Optional[Tuple[int, int, int]]]:
    remaining_tokens = 0
    used_memory_tokens = 0
    queued_count = 0
    transfer_durations_ms: List[int] = []
    running_items: List[Dict[str, Any]] = []
    live_capacity_tokens = 0
    for snapshot in _worker_snapshot_items(snapshot_payload):
        if snapshot.get("mode") not in {None, "decode", "unified"}:
            continue
        limits = snapshot.get("limits") or {}
        try:
            live_capacity_tokens = max(
                live_capacity_tokens,
                int(limits.get("max_total_num_tokens") or 0),
            )
        except (TypeError, ValueError):
            pass
        for queue_name, item in _snapshot_queue_items(
            snapshot,
            include_mutable_waiting=include_mutable_waiting,
        ):
            used_memory_tokens += _snapshot_memory_tokens(item)
            if "transfer" in queue_name or queue_name in {
                "prealloc",
                "decode_prealloc",
            }:
                # ``prealloc`` entries have reserved metadata/KV slots but
                # have not entered the transfer queue yet. They still consume
                # the same transfer pipeline once admitted. With an explicit
                # runtime cap, approximate the pipeline as equally balanced
                # slots; without a cap, preserve the historical overlap model.
                transfer_durations_ms.append(
                    _snapshot_transfer_ms(request, worker, item)
                )
                continue
            try:
                remaining_tokens += max(
                    0, int(item.get("remaining_service_tokens") or 0)
                )
            except (TypeError, ValueError):
                pass
            queued_count += 1
            if queue_name == "running" or item.get("is_running") is True:
                running_items.append(item)

    compute_ms = max(
        (_snapshot_decode_duration_ms(request, worker, item, len(running_items))
         for item in running_items),
        default=0.0,
    )

    configured_transfer_slots = 0
    try:
        configured_transfer_slots = max(
            0,
            int(
                os.environ.get(
                    "SGLANG_DISAGGREGATION_MAX_INFLIGHT_TRANSFERS", "0"
                )
            ),
        )
    except (TypeError, ValueError):
        configured_transfer_slots = 0
    if transfer_durations_ms:
        transfer_work_ms = sum(transfer_durations_ms)
        if configured_transfer_slots <= 0:
            # Transfers overlap when the runtime has no explicit cap.
            transfer_ms = max(transfer_durations_ms)
        else:
            transfer_ms = (
                transfer_work_ms + configured_transfer_slots - 1
            ) // configured_transfer_slots
    else:
        transfer_ms = 0

    # A running decode batch is one continuous computation interval. Its
    # completion is governed by the longest remaining sequence, not the sum of
    # all output tokens across the batch.
    # KV transfer overlaps with decode compute, but a newly admitted request
    # cannot start before both the compute-side queue and its transfer queue
    # have made progress.  Taking the maximum avoids double-counting overlap
    # while still preventing the old zero-cost transfer underestimation.
    work_ms = max(compute_ms, float(transfer_ms))
    end_ms = int(round(current_time_ms + max(0.0, work_ms)))
    capacity_tokens = live_capacity_tokens or int(worker.capacity_tokens)
    interval = None
    if queued_count > 0 or remaining_tokens > 0 or used_memory_tokens > 0:
        demand_tokens = max(1, used_memory_tokens)
        interval = (current_time_ms, max(current_time_ms + 1, end_ms), demand_tokens)
    # The interval already accounts for occupied KV and releases it at end_ms.
    # Keep the constant stage limit equal to the total pool to count it once.
    return end_ms, capacity_tokens, interval


def _decode_running_batch_size(snapshot_payload: Dict[str, Any]) -> int:
    """Return active decode sequences without counting transfer/waiting work."""

    running = 0
    for snapshot in _worker_snapshot_items(snapshot_payload):
        if snapshot.get("mode") not in {None, "decode", "unified"}:
            continue
        for item in snapshot.get("running") or []:
            if not isinstance(item, dict):
                continue
            if item.get("is_running") is False:
                continue
            running += 1
    return running


def apply_worker_snapshots(
    request: PlannerScheduleRequest,
    *,
    include_mutable_waiting: bool = True,
) -> None:
    """Fold supplied worker snapshots into planner availability/capacity state.

    The worker endpoint reports queue/running records, not exact future finish
    times. This helper uses each worker's calibrated latency labels to convert
    outstanding token work into conservative availability hints.
    """

    snapshots = request.state.worker_snapshots or {}
    if not snapshots:
        return
    current_time_ms = int(request.state.current_time_ms)

    prefill_available: List[int] = []
    for worker in request.prefill_workers:
        payload = snapshots.get(worker.url) or snapshots.get(worker.id)
        if isinstance(payload, dict):
            prefill_available.append(
                _estimate_prefill_available_ms(
                    worker,
                    payload,
                    current_time_ms,
                    request=request,
                    include_mutable_waiting=include_mutable_waiting,
                )
            )
        else:
            prefill_available.append(int(worker.available_at_ms))
    request.state.prefill_available_at_ms = prefill_available

    decode_available: List[int] = []
    decode_running_batch_sizes: List[int] = []
    decode_intervals: List[List[Tuple[int, int, int]]] = []
    for worker in request.decode_workers:
        payload = snapshots.get(worker.url) or snapshots.get(worker.id)
        if isinstance(payload, dict):
            available_ms, capacity_tokens, interval = _estimate_decode_state(
                worker,
                payload,
                current_time_ms,
                request=request,
                include_mutable_waiting=include_mutable_waiting,
            )
            worker.capacity_tokens = capacity_tokens
            decode_available.append(available_ms)
            decode_running_batch_sizes.append(_decode_running_batch_size(payload))
            # Old snapshot callers without a runtime sequence limit retain
            # their historical aggregate calendar. The explicit live adapter
            # requires that limit and keeps one interval per running request.
            has_sequence_limit = any(
                (snapshot.get("limits") or {}).get("max_running_requests") is not None
                for snapshot in _worker_snapshot_items(payload)
            )
            decode_intervals.append(
                _running_decode_intervals(request, worker, payload)
                if has_sequence_limit else ([interval] if interval else [])
            )
        else:
            decode_available.append(int(worker.available_at_ms))
            decode_running_batch_sizes.append(0)
            decode_intervals.append([])
    request.state.decode_available_at_ms = decode_available
    request.state.decode_running_batch_sizes = decode_running_batch_sizes
    request.state.decode_scheduled_intervals = decode_intervals


def fetch_worker_snapshot(
    worker_url: str,
    timeout_s: float,
    *,
    prefix_cache_probe: Optional[Dict[str, Any]] = None,
    prefix_probes: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    url = worker_url.rstrip("/") + "/hexgent/pd_snapshot"
    probes = list(prefix_probes) if prefix_probes is not None else (
        [prefix_cache_probe] if prefix_cache_probe is not None else [])
    if not probes:
        request = urllib.request.Request(url, method="GET")
    else:
        exact_probes = [
            {key: probe[key] for key in ("probe_id", "input_ids", "extra_key", "max_prefix_tokens")}
            for probe in probes
        ]
        request = urllib.request.Request(
            url, method="POST",
            data=json.dumps({"prefix_probes": exact_probes}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def fetch_worker_snapshots(
    workers: Iterable[WorkerDescriptor],
    *,
    timeout_s: float,
    prefix_cache_probe: Optional[Dict[str, Any]] = None,
    prefix_probes: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    snapshots: Dict[str, Dict[str, Any]] = {}
    for worker in workers:
        if worker.worker_type == "prefill" and prefix_probes is not None:
            payload = fetch_worker_snapshot(worker.url, timeout_s, prefix_probes=prefix_probes)
        elif worker.worker_type == "prefill" and prefix_cache_probe is not None:
            payload = fetch_worker_snapshot(worker.url, timeout_s, prefix_cache_probe=prefix_cache_probe)
        else:
            payload = fetch_worker_snapshot(worker.url, timeout_s)
        if payload:
            snapshots[worker.url] = payload
    return snapshots


def _int_item(item: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = item.get(key, default)
        return int(default if value is None else value)
    except (TypeError, ValueError):
        return default


def _optional_int_item(item: Dict[str, Any], key: str) -> Optional[int]:
    value = item.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _snapshot_queue_records(
    *,
    request: PlannerScheduleRequest,
    workers: List[WorkerDescriptor],
    stage: str,
    include_running: bool = False,
    require_identities: bool = False,
) -> List[SnapshotWaitingCall]:
    records: List[SnapshotWaitingCall] = []
    snapshots = request.state.worker_snapshots or {}
    current_time_ms = int(request.state.current_time_ms)
    for worker_index, worker in enumerate(workers):
        payload = snapshots.get(worker.url) or snapshots.get(worker.id)
        if not isinstance(payload, dict):
            continue
        for snapshot in _worker_snapshot_items(payload):
            if snapshot.get("mode") not in {None, stage, "unified"}:
                continue
            for queue_key, item in _snapshot_queue_items(snapshot, include_mutable_waiting=True):
                if stage == "decode":
                    queue_name = _decode_role(queue_key, item)
                    is_running = queue_name == "decode_running"
                    if require_identities and queue_name not in {
                        "decode_prealloc", "decode_transfer", "decode_waiting",
                        "decode_retracted", "decode_running",
                    }:
                        raise ValueError(f"Unsupported decode snapshot queue {queue_name!r}")
                else:
                    queue_name = queue_key if queue_key.startswith("prefill_") else "prefill_" + queue_key
                    is_running = queue_key == "running" or item.get("is_running") is True
                if is_running and not include_running:
                    continue
                if bool(item.get("prefill_started")) and stage == "prefill":
                    continue
                request_id = str(item.get("rid") or item.get("request_id") or "")
                call_id = str(item.get("call_id") or item.get("hexgent_call_id") or "")
                workflow_id = str(item.get("workflow_id") or item.get("hexgent_workflow_id") or "")
                if not request_id or not call_id or not workflow_id:
                    if require_identities and not is_running:
                        raise ValueError(f"Decode snapshot {queue_name} lacks request/workflow/call identity")
                    if not is_running:
                        continue
                    # Runtime health checks can run without HexAgenT identity.
                    # Their observed occupancy still belongs in the calendar.
                output_tokens = (
                    _optional_int_item(item, "predicted_output_tokens")
                    or _optional_int_item(item, "max_new_tokens")
                    or _optional_int_item(item, "remaining_service_tokens") or 1
                )
                locked_decode_instance = None
                planned = item.get("planned_decode_instance")
                for decode_index, decode_worker in enumerate(request.decode_workers):
                    if planned in {decode_worker.id, decode_worker.url}:
                        locked_decode_instance = decode_index
                        break
                if stage == "decode" and locked_decode_instance is None:
                    locked_decode_instance = worker_index
                ready_at_ms = current_time_ms
                if stage == "decode" and queue_name in {"decode_prealloc", "decode_transfer"}:
                    ready_at_ms += _snapshot_transfer_ms(request, worker, item)
                records.append(SnapshotWaitingCall(
                    request_id=request_id, call_id=call_id, workflow_id=workflow_id,
                    worker_url=worker.url, worker_index=worker_index, queue_name=queue_name,
                    schedule_revision=_int_item(item, "schedule_revision"),
                    input_tokens=_snapshot_input_tokens(item), output_tokens=max(1, int(output_tokens)),
                    workflow_arrival_ms=_int_item(item, "workflow_arrival_ms", current_time_ms),
                    workflow_horizon_ms=max(1, _int_item(item, "workflow_horizon_ms", 1)),
                    reveal_sequence=_int_item(item, "reveal_sequence"),
                    workflow_completion_ms=_int_item(item, "workflow_completion_ms"),
                    workflow_attained_service_ms=_int_item(item, "workflow_attained_service_ms"),
                    critical_path_ms=_int_item(item, "critical_path_ms"),
                    child_count=_int_item(item, "child_count"), ready_at_ms=ready_at_ms,
                    locked_decode_instance=locked_decode_instance,
                    parent_call_ids=_snapshot_parent_call_ids(item), is_running=is_running,
                    is_mutable_waiting=queue_name == f"{stage}_waiting" and not is_running,
                    remaining_service_tokens=max(0, _int_item(item, "remaining_service_tokens")),
                    memory_tokens=_snapshot_memory_tokens(item),
                    output_tokens_generated=max(0, _int_item(item, "output_tokens_generated")),
                    estimated_prefill_time=_snapshot_float(item, "estimated_prefill_time"),
                    estimated_decode_time=_snapshot_float(item, "estimated_decode_time"),
                    estimated_transfer_time=_snapshot_float(item, "estimated_transfer_time"),
                ))
    return records


def decode_snapshot_calls(request: PlannerScheduleRequest) -> List[SnapshotWaitingCall]:
    """Return every currently observed D role, with immutable running flags."""
    return _snapshot_queue_records(
        request=request, workers=request.decode_workers, stage="decode",
        include_running=True, require_identities=True,
    )


def waiting_prefill_calls(request: PlannerScheduleRequest) -> List[SnapshotWaitingCall]:
    return _snapshot_queue_records(
        request=request,
        workers=request.prefill_workers,
        stage="prefill",
    )


def waiting_decode_calls(request: PlannerScheduleRequest) -> List[SnapshotWaitingCall]:
    return _snapshot_queue_records(
        request=request,
        workers=request.decode_workers,
        stage="decode",
    )


def maybe_apply_live_worker_snapshots(
    request: PlannerScheduleRequest,
    *,
    include_mutable_waiting: bool = True,
    prefix_probes: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    probes = list(prefix_probes if prefix_probes is not None else
                  ([request.prefix_cache_probe] if request.prefix_cache_probe else []))
    if (probes and os.environ.get("HEXGENT_ENABLE_PREFIX_CACHE_PROBES", "0") == "1"
            and os.environ.get("HEXGENT_ENABLE_WORKER_SNAPSHOTS", "0") == "1"):
        from sglang.hexgent.cache import MAX_PREFIX_PROBES, MAX_PREFIX_PROBE_TOTAL_TOKENS
        bounded, total = [], 0
        for probe in probes:
            size = len(probe["input_ids"])
            if len(bounded) >= MAX_PREFIX_PROBES or total + size > MAX_PREFIX_PROBE_TOTAL_TOKENS:
                continue
            bounded.append(probe)
            total += size
        # One fresh snapshot set binds all covered pending probes to this solve.
        # Calls beyond the bounded probe budget retain unknown cache state.
        from copy import deepcopy
        snapshots = deepcopy(request.state.worker_snapshots)
        for worker in request.prefill_workers:
            payload = snapshots.get(worker.url) or snapshots.get(worker.id)
            if isinstance(payload, dict):
                for snapshot in _worker_snapshot_items(payload):
                    snapshot.pop("prefix_observations", None)
        snapshots.update(fetch_worker_snapshots(
            [*request.prefill_workers, *request.decode_workers],
            timeout_s=float(os.environ.get("HEXGENT_WORKER_SNAPSHOT_TIMEOUT_S", "0.05")),
            prefix_probes=bounded))
        # A failed optional cache query grants no old credit; retain the
        # caller's observed occupancy and do not issue an immediate retry.
        request.state.worker_snapshots = snapshots
        apply_worker_snapshots(request, include_mutable_waiting=include_mutable_waiting)
        return
    if request.state.worker_snapshots:
        apply_worker_snapshots(
            request,
            include_mutable_waiting=include_mutable_waiting,
        )
        return
    if os.environ.get("HEXGENT_ENABLE_WORKER_SNAPSHOTS", "0") != "1":
        return
    timeout_s = float(os.environ.get("HEXGENT_WORKER_SNAPSHOT_TIMEOUT_S", "0.05"))
    workers = [*request.prefill_workers, *request.decode_workers]
    request.state.worker_snapshots = fetch_worker_snapshots(
        workers,
        timeout_s=timeout_s,
        prefix_cache_probe=(request.prefix_cache_probe
            if os.environ.get("HEXGENT_ENABLE_PREFIX_CACHE_PROBES", "0") == "1" else None),
    )
    apply_worker_snapshots(
        request,
        include_mutable_waiting=include_mutable_waiting,
    )


def _snapshot_decode_demand(item: Dict[str, Any]) -> int:
    """Reserve full predicted context, regardless of prefill prefix reuse."""
    generated = max(0, _int_item(item, "output_tokens_generated"))
    output = max(0, _int_item(item, "predicted_output_tokens"))
    if output <= 0:
        output = max(0, _int_item(item, "max_new_tokens"))
    if output <= 0:
        output = generated + max(0, _int_item(item, "remaining_service_tokens"))
    return max(_snapshot_memory_tokens(item), _snapshot_input_tokens(item) + output)


def _snapshot_decode_duration_ms(request, worker, item, batch_size=1) -> float:
    """Predict only unfinished decode work using worker labels/model metadata."""
    remaining = max(0, _int_item(item, "remaining_service_tokens",
                                _int_item(item, "output_tokens_remaining", 0)))
    if remaining <= 0:
        return 0.0
    input_tokens = _snapshot_input_tokens(item)
    generated = max(0, _int_item(item, "output_tokens_generated"))
    profiled = _snapshot_float(item, "estimated_decode_time")
    original_output = max(_int_item(item, "max_new_tokens"),
                          _int_item(item, "predicted_output_tokens"), generated + remaining)
    try:
        from sglang.hexgent.latency_estimator import RooflineLatencyEstimator, _hardware_params, _model_params
        model = _model_params(request, worker)
        hardware = _hardware_params(request, worker)
        if profiled is not None:
            item_ms = profiled * min(1.0, remaining / max(1, original_output))
            if model is not None and hardware is not None:
                estimator = RooflineLatencyEstimator()
                single = estimator.estimate_decode_total_ms(
                    model=model, hardware=hardware, input_tokens=input_tokens + generated,
                    output_tokens=remaining, batch_size=1)
                batch = estimator.estimate_decode_total_ms(
                    model=model, hardware=hardware, input_tokens=input_tokens + generated,
                    output_tokens=remaining, batch_size=batch_size)
                item_ms *= max(1.0, batch / max(1, single))
            else:
                # Retain the existing calibrated fallback slowdown when the
                # snapshot has no model/hardware shape for batch estimation.
                item_ms *= 1.0 + 0.15 * max(0, batch_size - 1)
            return item_ms
        if model is not None and hardware is not None:
            return float(RooflineLatencyEstimator().estimate_decode_total_ms(
                model=model, hardware=hardware, input_tokens=input_tokens + generated,
                output_tokens=remaining, batch_size=batch_size))
    except (ImportError, TypeError, ValueError, ZeroDivisionError):
        pass
    return float(worker.decode_base_ms) + float(worker.decode_ms_per_output_token) * remaining


def _decode_role(queue_name: str, item: Dict[str, Any]) -> str:
    if item.get("is_running") is True or queue_name in {"running", "decode_running"}:
        return "decode_running"
    return queue_name if queue_name.startswith("decode_") else "decode_" + queue_name


def _running_decode_intervals(request, worker, payload):
    now = int(request.state.current_time_ms)
    running = []
    for snapshot in _worker_snapshot_items(payload):
        if snapshot.get("mode") not in {None, "decode", "unified"}:
            continue
        for queue_name, item in _snapshot_queue_items(snapshot, include_mutable_waiting=True):
            if _decode_role(queue_name, item) == "decode_running" and item.get("is_running") is not False:
                running.append(item)
    return [
        (now, now + max(1, int(round(_snapshot_decode_duration_ms(request, worker, item, len(running))))),
         _snapshot_decode_demand(item))
        for item in running
    ]


def _single_live_snapshot(payload, stage, worker):
    snapshots = [snapshot for snapshot in _worker_snapshot_items(payload)
                 if snapshot.get("mode") in {None, stage, "unified"}]
    if int(payload.get("dp_rank_count", 1)) != 1 or len(snapshots) != 1:
        raise ValueError(f"Worker {worker.id} requires one explicit DP replica snapshot")
    return snapshots[0]


def build_live_stage_states(
    request: PlannerScheduleRequest,
    *,
    decode_calls: Sequence[Any] = (),
    prefill_available_at_ms: Optional[Sequence[int]] = None,
    decode_available_hint_ms: Optional[Sequence[int]] = None,
):
    """Build canonical public calendars from one frozen DP1 worker snapshot.

    Supplied call views retain workflow-local IDs and their full context. Their
    priority metadata must cover every revealed D queue job and every incoming
    P candidate when a D backlog exists. Job IDs use global_call_id; the live
    scheduler adapter normalizes cloned views before invoking the frozen core.
    Transfer release is a remaining-work forecast, not an allocator event.
    """
    from sglang.hexgent.forecast import global_call_id
    from sglang.hexgent.scheduler import DecodeStageState, PrefillStageState

    now = int(request.state.current_time_ms)
    snapshots = request.state.worker_snapshots or {}
    for worker in request.prefill_workers:
        payload = snapshots.get(worker.url) or snapshots.get(worker.id)
        if isinstance(payload, dict):
            _single_live_snapshot(payload, "prefill", worker)
    capacity = []
    slots = []
    floors = []
    intervals = []
    jobs = [[] for _ in request.decode_workers]
    job_ids = [[] for _ in request.decode_workers]
    calls_by_key = {(str(call["workflow_id"]), str(call["call_id"])): call for call in decode_calls}
    explicit_slots = request.state.decode_max_batch_size_by_instance
    explicit_floors = request.state.decode_admission_not_before_ms
    for index, worker in enumerate(request.decode_workers):
        payload = snapshots.get(worker.url) or snapshots.get(worker.id)
        floor = int(explicit_floors[index]) if explicit_floors is not None else now
        if isinstance(payload, dict):
            snapshot = _single_live_snapshot(payload, "decode", worker)
            limits = snapshot.get("limits") or {}
            for name in ("max_total_num_tokens", "max_running_requests"):
                if name not in limits or limits[name] is None or int(limits[name]) <= 0:
                    raise ValueError(f"Worker {worker.id} snapshot lacks positive {name}")
            capacity.append(int(limits["max_total_num_tokens"]))
            slots.append(int(limits["max_running_requests"]))
            floor = max(floor, int(snapshot.get("admission_not_before_ms", 0)),
                        int(limits.get("decode_admission_not_before_ms", 0)))
            intervals.append(_running_decode_intervals(request, worker, payload))
        else:
            if snapshots:
                raise ValueError(f"Worker {worker.id} is missing its live decode snapshot")
            # Direct-calendar callers retain their explicit legacy semantics.
            capacity.append(int(worker.capacity_tokens))
            slots.append(explicit_slots[index] if explicit_slots is not None else None)
            raw = request.state.decode_scheduled_intervals
            intervals.append([
                (int(item.start_ms), int(item.end_ms), int(item.demand_tokens))
                if hasattr(item, "start_ms") else tuple(item)
                for item in (raw[index] if index < len(raw) else [])
            ])
        floors.append(max(now, floor))
    for record in decode_snapshot_calls(request):
        if record.is_running:
            continue
        key = (record.workflow_id, record.call_id)
        call = calls_by_key.get(key)
        if call is None:
            raise ValueError(f"Missing revealed decode prediction for {key!r}")
        index = record.worker_index
        jobs[index].append((
            max(now, int(record.ready_at_ms), int(call["ready_at_ms"])),
            max(1, int(call["decode_ms_by_instance"][index])),
            max(record.memory_tokens, record.input_tokens + record.output_tokens,
                int(call["decode_demand_tokens"])),
        ))
        job_ids[index].append(global_call_id(*key))
    prefill_available = list(prefill_available_at_ms) if prefill_available_at_ms is not None else list(
        request.state.prefill_available_at_ms or [worker.available_at_ms for worker in request.prefill_workers])
    decode_available = list(decode_available_hint_ms) if decode_available_hint_ms is not None else list(
        request.state.decode_available_at_ms or [worker.available_at_ms for worker in request.decode_workers])
    has_jobs = any(jobs)
    p_state = PrefillStageState(
        current_time_ms=now, instance_available_at_ms=prefill_available,
        decode_available_hint_ms=decode_available, decode_capacity_max_tokens=capacity,
        decode_admission_not_before_ms=tuple(floors), decode_admission_intervals=intervals,
        decode_max_batch_size_by_instance=slots, decode_admission_jobs=jobs,
        decode_priority_calls=tuple(decode_calls) if has_jobs else (),
        decode_priority_job_ids=tuple(tuple(row) for row in job_ids),
    )
    d_state = DecodeStageState(now, list(capacity), [list(row) for row in intervals], list(slots), tuple(floors))
    return p_state, d_state
