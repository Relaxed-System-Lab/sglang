"""HexAgenT worker-side queue inspection and safe mutation helpers."""

from __future__ import annotations

import json
import os
import time


HEXGENT_DEQUEUE_MAX_RECORDS = 64
HEXGENT_DEQUEUE_MAX_REQUESTS = 64
HEXGENT_DEQUEUE_MAX_BYTES = 16 * 1024
_hexgent_dequeue_record_count = 0


def log_hexgent_dequeue(
    *, logger, role, worker_url, worker_port, tp_rank, waiting, selected
):
    """Log bounded metadata from actual selected requests in the owned smoke."""
    if (
        os.environ.get("HEXGENT_FUNCTIONAL_DEQUEUE_LOG") != "1"
        or tp_rank != 0
        or not selected
    ):
        return
    namespace = os.environ.get("HEXGENT_FUNCTIONAL_DEQUEUE_NAMESPACE")
    if not namespace or not any(
        (req.workflow_id or "").startswith(namespace) for req in selected
    ):
        return
    global _hexgent_dequeue_record_count
    if _hexgent_dequeue_record_count > HEXGENT_DEQUEUE_MAX_RECORDS:
        return
    payload = {
        "event": "dequeue",
        "role": role,
        "worker_url": worker_url,
        "worker_port": worker_port,
        "tp_rank": tp_rank,
        "record_cap": HEXGENT_DEQUEUE_MAX_RECORDS,
        "record_byte_cap": HEXGENT_DEQUEUE_MAX_BYTES,
    }
    if _hexgent_dequeue_record_count == HEXGENT_DEQUEUE_MAX_RECORDS:
        payload["event"] = "evidence_cap"
        payload["record_cap_reached"] = True
    else:

        def metadata(req):
            return {
                "rid": req.rid,
                "workflow_id": req.workflow_id,
                "call_id": req.hexgent_call_id,
                "priority": req.priority,
                "prefill_priority": req.hexgent_prefill_priority,
                "decode_priority": req.hexgent_decode_priority,
                "schedule_revision": req.hexgent_schedule_revision,
                "planned_decode_instance": req.planned_decode_instance,
            }

        payload.update(
            waiting_count=len(waiting),
            selected_count=len(selected),
            waiting=[metadata(req) for req in waiting[:HEXGENT_DEQUEUE_MAX_REQUESTS]],
            selected=[metadata(req) for req in selected[:HEXGENT_DEQUEUE_MAX_REQUESTS]],
            identities_truncated=(
                len(waiting) > HEXGENT_DEQUEUE_MAX_REQUESTS
                or len(selected) > HEXGENT_DEQUEUE_MAX_REQUESTS
            ),
        )
    prefix = "HexgentDequeueEvidence "
    record = json.dumps(payload, separators=(",", ":"))
    if len((prefix + record).encode("utf-8")) > HEXGENT_DEQUEUE_MAX_BYTES:
        record = json.dumps(
            {
                "event": "evidence_cap",
                "role": role,
                "worker_port": worker_port,
                "tp_rank": tp_rank,
                "record_byte_cap_reached": True,
                "record_byte_cap": HEXGENT_DEQUEUE_MAX_BYTES,
            },
            separators=(",", ":"),
        )
    logger.info("%s%s", prefix, record)
    _hexgent_dequeue_record_count += 1


def hexgent_stage_priority(disaggregation_mode, recv_req):
    mode = getattr(disaggregation_mode, "value", disaggregation_mode)
    if mode == "prefill":
        return (
            recv_req.hexgent_prefill_priority
            if recv_req.hexgent_prefill_priority is not None
            else recv_req.priority
        )
    if mode == "decode":
        return (
            recv_req.hexgent_decode_priority
            if recv_req.hexgent_decode_priority is not None
            else recv_req.priority
        )
    return recv_req.priority


def hexgent_update_waiting_request_priority(
    queue,
    *,
    request_id: str,
    new_priority: int,
    new_revision: int,
    expected_revision=None,
    reorder: bool = False,
    schedule_low_priority_values_first: bool = False,
    planned_decode_instance=None,
):
    matched = [
        getattr(entry, "req", entry)
        for entry in queue
        if getattr(getattr(entry, "req", entry), "rid", None) == request_id
    ]
    if not matched:
        return {
            "success": False,
            "message": f"Request {request_id} was not found.",
            "reason_code": "not_found",
            "current_revision": None,
        }

    current_revision = getattr(matched[0], "hexgent_schedule_revision", 0)
    if expected_revision is not None and current_revision != expected_revision:
        return {
            "success": False,
            "message": f"Request {request_id} has revision {current_revision}, expected {expected_revision}.",
            "reason_code": "expected_revision_mismatch",
            "current_revision": current_revision,
        }
    if new_revision < current_revision:
        return {
            "success": False,
            "message": f"Request {request_id} has newer revision {current_revision} than incoming revision {new_revision}.",
            "reason_code": "stale_revision",
            "current_revision": current_revision,
        }

    for req in matched:
        req.priority = new_priority
        req.hexgent_schedule_revision = new_revision
        if planned_decode_instance is not None:
            req.planned_decode_instance = planned_decode_instance

    if reorder:
        queue.sort(
            key=lambda entry: (
                getattr(getattr(entry, "req", entry), "priority", 0),
                getattr(getattr(entry, "req", entry), "hexgent_queue_entered_at", 0.0),
                getattr(getattr(entry, "req", entry), "rid", ""),
            ),
            reverse=not schedule_low_priority_values_first,
        )
    return {
        "success": True,
        "message": f"Updated request {request_id}.",
        "applied_revision": new_revision,
        "reason_code": "success",
        "current_revision": new_revision,
    }


def hexgent_reassign_waiting_request(
    source_queues,
    *,
    target_queue,
    target_queue_name: str,
    request_id: str,
    new_priority: int,
    new_revision: int,
    expected_revision=None,
    planned_decode_instance=None,
    reorder_target: bool = False,
    schedule_low_priority_values_first: bool = False,
    target_append_fn=None,
    now=None,
):
    matched = []
    for queue_name, queue in source_queues.items():
        for index, entry in enumerate(queue):
            req = getattr(entry, "req", entry)
            if getattr(req, "rid", None) == request_id:
                matched.append((queue_name, queue, index, entry, req))
    if not matched:
        return {"success": False, "message": f"Request {request_id} was not found."}

    canonical_req = matched[0][4]
    current_revision = getattr(canonical_req, "hexgent_schedule_revision", 0)
    if expected_revision is not None and current_revision != expected_revision:
        return {"success": False, "message": "Schedule revision mismatch."}
    if new_revision < current_revision:
        return {"success": False, "message": "Incoming schedule revision is stale."}
    if bool(getattr(canonical_req, "hexgent_prefill_started", False)):
        return {"success": False, "message": f"Request {request_id} has already started prefill."}

    for _name, queue, index, _entry, _req in sorted(
        matched, key=lambda item: (id(item[1]), item[2]), reverse=True
    ):
        queue.pop(index)
    target_queue[:] = [
        entry
        for entry in target_queue
        if getattr(getattr(entry, "req", entry), "rid", None) != request_id
    ]
    canonical_req.priority = new_priority
    canonical_req.hexgent_schedule_revision = new_revision
    canonical_req.hexgent_queue_state = target_queue_name
    canonical_req.hexgent_queue_entered_at = time.time() if now is None else now
    if planned_decode_instance is not None:
        canonical_req.planned_decode_instance = planned_decode_instance
    (target_append_fn or target_queue.append)(canonical_req)
    if reorder_target:
        target_queue.sort(
            key=lambda entry: getattr(getattr(entry, "req", entry), "priority", 0),
            reverse=not schedule_low_priority_values_first,
        )
    return {
        "success": True,
        "message": f"Reassigned request {request_id}.",
        "source_queue_names": [item[0] for item in matched],
        "target_queue_name": target_queue_name,
        "applied_revision": new_revision,
        "request": canonical_req,
        "removed_entries": matched,
    }


def build_hexgent_stage_timing_payload(req, *, mode: str):
    import math

    workflow_id = getattr(req, "workflow_id", None)
    call_id = getattr(req, "hexgent_call_id", None) or getattr(req, "rid", None)
    stats = getattr(req, "time_stats", None)
    if not workflow_id or not call_id or stats is None:
        return None
    if mode == "prefill":
        start = getattr(stats, "forward_entry_time", 0.0) or 0.0
        execution_end = getattr(stats, "prefill_finished_time", 0.0) or 0.0
        last_enqueue = getattr(stats, "prefill_transfer_queue_entry_time", 0.0) or 0.0
        completion = getattr(stats, "completion_time", 0.0) or 0.0
        timestamps = (start, execution_end, last_enqueue, completion)
        if (
            not all(math.isfinite(value) and value > 0.0 for value in timestamps)
            or not start <= execution_end <= last_enqueue <= completion
        ):
            return None
        return {
            "workflow_id": workflow_id,
            "call_id": call_id,
            "stage": "prefill",
            # Pure P execution remains diagnostic; last enqueue can follow it.
            "prefill_execution_ms": round((execution_end - start) * 1000),
            # Both additive intervals use P's own clock. The tail includes
            # queue/send/control/poll completion, not an isolated link transfer.
            "prefill_service_prefix_ms": (last_enqueue - start) * 1000,
            "prefill_transfer_tail_ms": (completion - last_enqueue) * 1000,
            "prefill_service_prefix_boundary": "forward_entry_to_last_chunk_enqueue_return",
            "prefill_transfer_tail_boundary": "last_chunk_enqueue_return_to_completion",
        }
    if mode == "decode":
        transfer_start = getattr(stats, "decode_transfer_queue_entry_time", 0.0) or 0.0
        transfer_end = getattr(stats, "wait_queue_entry_time", 0.0) or 0.0
        decode_start = getattr(stats, "forward_entry_time", 0.0) or 0.0
        decode_end = getattr(stats, "completion_time", 0.0) or 0.0
        timestamps = (transfer_start, transfer_end, decode_start, decode_end)
        if (
            not all(math.isfinite(value) and value > 0.0 for value in timestamps)
            or not transfer_start <= transfer_end <= decode_start <= decode_end
        ):
            return None
        return {
            "workflow_id": workflow_id,
            "call_id": call_id,
            "stage": "decode",
            # This interval can include P execution and is never added as
            # independent transfer service by the gateway.
            "decode_pipeline_wait_ms": (transfer_end - transfer_start) * 1000,
            "decode_pipeline_wait_boundary": "decode_transfer_queue_entry_to_wait_queue_entry",
            "decode_execution_ms": (decode_end - decode_start) * 1000,
        }
    return None


def build_hexgent_pd_snapshot_payload(
    *, mode, instance_id, dp_rank, waiting_queue, running_reqs,
    prefill_bootstrap_queue=(), prefill_inflight_queue=(),
    decode_prealloc_queue=(), decode_transfer_queue=(), decode_retracted_queue=(),
    limits=None, now=None,
):
    now = time.time() if now is None else now

    def item(entry, queue_name, running=False):
        req = getattr(entry, "req", entry)
        output_tokens = len(getattr(req, "output_ids", []) or [])
        input_tokens = len(getattr(req, "origin_input_ids", []) or [])
        max_new = getattr(getattr(req, "sampling_params", None), "max_new_tokens", 0)
        prefill_started = bool(getattr(req, "hexgent_prefill_started", False))
        if running or queue_name in {
            "prefill_inflight",
            "decode_transfer",
            "decode_waiting",
            "decode_retracted",
            "decode_running",
        }:
            prefill_started = True
        return {
            "rid": req.rid,
            "call_id": getattr(req, "hexgent_call_id", None) or req.rid,
            "workflow_id": getattr(req, "workflow_id", None),
            "queue_name": queue_name,
            "current_assignment": instance_id,
            "current_priority_score": getattr(req, "priority", None),
            "schedule_revision": getattr(req, "hexgent_schedule_revision", 0),
            "planned_decode_instance": getattr(req, "planned_decode_instance", None),
            "prefill_started": prefill_started,
            "input_tokens": input_tokens,
            "output_tokens_generated": output_tokens,
            "max_new_tokens": max_new,
            "memory_tokens": input_tokens + output_tokens,
            "remaining_service_tokens": max(max_new - output_tokens, 0),
            "is_running": running,
            "workflow_arrival_ms": getattr(req, "hexgent_workflow_arrival_ms", 0),
            "workflow_horizon_ms": getattr(req, "hexgent_workflow_horizon_ms", 1),
        }

    if mode == "prefill":
        queues = {
            "bootstrap": [item(x, "prefill_bootstrap") for x in prefill_bootstrap_queue],
            "waiting": [item(x, "prefill_waiting") for x in waiting_queue],
            "inflight": [item(x, "prefill_inflight") for x in prefill_inflight_queue],
        }
    elif mode == "decode":
        queues = {
            "prealloc": [item(x, "decode_prealloc") for x in decode_prealloc_queue],
            "transfer": [item(x, "decode_transfer") for x in decode_transfer_queue],
            "retracted": [item(x, "decode_retracted") for x in decode_retracted_queue],
            "waiting": [item(x, "decode_waiting") for x in waiting_queue],
        }
    else:
        queues = {"waiting": [item(x, "waiting") for x in waiting_queue]}
    return {
        "mode": mode,
        "instance_id": instance_id,
        "dp_rank": dp_rank,
        "timestamp": now,
        "limits": dict(limits or {}),
        "queues": queues,
        "running": [item(x, f"{mode}_running", True) for x in running_reqs],
    }
