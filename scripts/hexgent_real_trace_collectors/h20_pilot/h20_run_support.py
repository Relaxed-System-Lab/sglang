"""Shared H20 replay evidence helpers; runtime execution requires manual start."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
_WORKFLOW_API = None


def load_workflow_api(source_root):
    """Load the actual standalone replay source once, without SGLang/GPU init."""
    global _WORKFLOW_API
    path = (Path(source_root) / "python/sglang/hexgent/workflow_trace.py").resolve()
    if _WORKFLOW_API is None:
        spec = importlib.util.spec_from_file_location("_h20_actual_workflow_trace", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _WORKFLOW_API = module
    elif Path(_WORKFLOW_API.__file__).resolve() != path:
        raise ValueError("one process must use one frozen workflow replay source")
    return _WORKFLOW_API


def incremental_replayer_class(api, append_row):
    """Persist the native just-appended row before returning control to the DAG."""
    class IncrementalReplayer(api.WorkflowTraceReplayer):
        async def _run_call(self, session, workflow_id, call, results,
                            workflow_arrival_ms, workflow_horizon_ms):
            call_id = await super()._run_call(session, workflow_id, call, results,
                                              workflow_arrival_ms, workflow_horizon_ms)
            row = results[-1]
            if row["workflow_id"] != workflow_id or row["call_id"] != call_id:
                raise RuntimeError("native replay append contract changed; expected just-completed matching row")
            append_row(row)
            return call_id
    return IncrementalReplayer


async def replay_to_raw(api, workflows, *, raw_path, router_url, model, worker_class, run_namespace):
    """Replay supplied offsets with exact targets and incremental fresh JSONL.

    A flush after every row preserves completed-call evidence across process
    termination. In-flight requests may have no row after a hard stop.
    """
    for workflow in workflows:
        for call in workflow.calls:
            if set(call.planner_hints) - {"model_config", "hardware_params"}:
                raise ValueError("replay accepts only current static model/hardware hints")
    with Path(raw_path).open("x", encoding="utf-8") as target:
        def append_row(row):
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            target.flush()
        replayer = incremental_replayer_class(api, append_row)(router_url=router_url,
            profile_table=None, model=model, worker_class=worker_class,
            exact_output_length=True, timeout_s=180, endpoint="/v1/chat/completions",
            run_namespace=run_namespace)
        return await replayer.replay(workflows, arrival_rate=None)


def validate_workflow_rows(workflow, rows):
    """Check one original workflow's complete observed rows, without a proxy SLO."""
    expected = {call.call_id: call for call in workflow.calls}
    reasons = []
    if len(expected) != len(workflow.calls) or not expected:
        reasons.append("expected call IDs must be unique and nonempty")
    ids = [str(row["call_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        reasons.append("duplicate call IDs")
    if set(ids) != set(expected):
        reasons.append("observed call IDs differ from expected workflow")
    if any(str(row["workflow_id"]) != workflow.workflow_id for row in rows):
        reasons.append("observed workflow ID differs from expected")
    errors = sum(bool(row.get("error")) for row in rows)
    if errors:
        reasons.append("request errors")
    verified = set()
    for row in rows:
        call = expected.get(str(row["call_id"]))
        if call is None or row.get("error"):
            continue
        usage = (row.get("response") or {}).get("usage") or {}
        if usage.get("prompt_tokens") == call.input_len and usage.get("completion_tokens") == call.output_len:
            verified.add(call.call_id)
        else:
            reasons.append(f"missing or mismatched observed token usage: {call.call_id}")
    arrivals = [row.get("workflow_arrival_ms") for row in rows]
    arrival = None
    if (arrivals and all(value is not None and math.isfinite(float(value)) for value in arrivals)
            and len(set(float(value) for value in arrivals)) == 1):
        arrival = float(arrivals[0])
    else:
        reasons.append("one finite workflow arrival is required on every row")
    ends = [row.get("response_completed_at_ms") for row in rows]
    last = None
    if ends and all(value is not None and math.isfinite(float(value)) for value in ends):
        last = max(float(value) for value in ends)
    else:
        reasons.append("finite observed response completion timestamps are required")
    elapsed = last - arrival if last is not None and arrival is not None else None
    if elapsed is None or not math.isfinite(elapsed) or elapsed <= 0:
        reasons.append("positive finite workflow elapsed latency is required")
    return {"successful": not reasons, "expected_calls": len(workflow.calls),
            "completed_calls": len(set(ids) & set(expected)), "errors": errors,
            "usage_verified_calls": len(verified), "workflow_arrival_ms": arrival,
            "response_completed_at_ms": last, "latency_ms": elapsed if not reasons else None,
            "failure_reasons": reasons}


def workflow_reference_from_rows(workflow, rows, *, model, implementation_revision,
                                  runtime_evidence_sha256, raw_replay_sha256):
    gate = validate_workflow_rows(workflow, rows)
    if any(not row.get("error") and (row.get("response") or {}).get("model") != model for row in rows):
        gate["failure_reasons"].append("returned model differs from frozen served model")
        gate["successful"] = False; gate["latency_ms"] = None
    return {"workflow_id": workflow.workflow_id, "reference_kind": "isolated_workflow_replay",
            "latency_ms": gate["latency_ms"], "status": "successful" if gate["successful"] else "failed",
            "expected_calls": gate["expected_calls"], "completed_calls": gate["completed_calls"],
            "errors": gate["errors"], "usage_verified_calls": gate["usage_verified_calls"],
            "model": model, "implementation_revision": implementation_revision,
            "raw_replay_sha256": raw_replay_sha256, "runtime_evidence_sha256": runtime_evidence_sha256,
            "failure_reasons": gate["failure_reasons"]}
