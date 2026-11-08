#!/usr/bin/env python3
"""C1 preparation only by default; runtime operations require separate approval."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import threading

SEED = 20261107
HOT_IDS = tuple(f"hotpotqa_{index}" for index in range(10, 14))
PILOT_DIR = Path(__file__).with_name("h20_pilot")
CAPTURE_BYTES = 8 * 1024 * 1024
OBSERVATION_MAX_AGE_S = 3.0
_PILOT_MODULES = {}
_STAGES = {"startup", "calibration-native", "calibration-ours",
    *("reference-" + workflow_id for workflow_id in HOT_IDS),
    "formal-07-hotpotqa-burst-native", "formal-08-hotpotqa-burst-ours", "references-hot4", "startup-idle"}


def _pilot(name):
    if name not in _PILOT_MODULES:
        path = PILOT_DIR / (name + ".py")
        spec = importlib.util.spec_from_file_location("_c1_frozen_" + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _PILOT_MODULES[name] = module
    return _PILOT_MODULES[name]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, *, fresh=True):
    with Path(path).open("x" if fresh else "w", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write("\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class WindowBudget:
    anchor_unix_s: float
    duration_s: float = 2700.0
    cleanup_s: float = 180.0

    def __post_init__(self):
        if (not all(math.isfinite(value) for value in
                    (self.anchor_unix_s, self.duration_s, self.cleanup_s))
                or self.duration_s > 2700 or self.cleanup_s < 180
                or self.duration_s <= self.cleanup_s):
            raise ValueError("finite <=45-minute window and >=180 seconds cleanup required")

    @property
    def hard_deadline_unix_s(self):
        return self.anchor_unix_s + self.duration_s

    @property
    def work_deadline_unix_s(self):
        return self.hard_deadline_unix_s - self.cleanup_s

    def child_timeout(self, now_unix_s, requested_s=180.0):
        remaining = self.work_deadline_unix_s - now_unix_s
        if remaining <= 0:
            raise TimeoutError("immutable work deadline reached; cleanup reserve retained")
        return min(float(requested_s), remaining)

    def pair_fits(self, now_unix_s, reference_latencies_ms, *, last_arrival_s=0):
        return pair_budget_s(reference_latencies_ms, last_arrival_s=last_arrival_s) <= self.work_deadline_unix_s - now_unix_s


def pair_budget_s(reference_latencies_ms, *, last_arrival_s=0):
    values = list(reference_latencies_ms)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("positive current-epoch references required")
    return 1.25 * 2 * (last_arrival_s + max(values) / 1000.0) + 60.0


class WindowStopped(RuntimeError):
    pass


class TransportWindow:
    """Fixed C1 stage, deadline, and ownership gate; no implicit runtime operation."""

    def __init__(self, *, run_id, budget, plan_sha256, clock=time.time):
        self.run_id = run_id
        self.budget = budget
        self.plan_sha256 = plan_sha256
        self.clock = clock
        self.owner_id = None
        self.stop_reason = None
        self.started_stages = set()

    def register_owned_container(self, owner_id):
        if not isinstance(owner_id, str) or re.fullmatch("[0-9a-f]{64}", owner_id) is None:
            raise ValueError("one exact full 64-hex owned container ID required")
        if self.owner_id is not None and self.owner_id != owner_id:
            raise ValueError("registered owned container cannot be replaced")
        self.owner_id = owner_id

    def _approve(self, approval):
        if (not isinstance(approval, dict) or approval.get("approved_plan_sha256") != self.plan_sha256
                or approval.get("window_run_id") != self.run_id):
            raise PermissionError("live execution needs explicit approval of this frozen plan/run")

    def _observe(self, summary):
        if (not isinstance(summary, dict) or summary.get("schema_version") != 1
                or summary.get("run_id") != self.run_id or summary.get("owner_id") != self.owner_id
                or summary.get("observation_complete") is not True
                or summary.get("first_error_unix_s") is not None
                or summary.get("stop_requested") is not False
                or summary.get("incomplete_reasons") != []):
            self.stop_reason = "observer first error, incomplete evidence, or identity mismatch"
            raise WindowStopped(self.stop_reason)
        sampled = summary.get("sampled_unix_s")
        if (not isinstance(sampled, (int, float)) or not math.isfinite(sampled)
                or self.clock() - sampled > OBSERVATION_MAX_AGE_S):
            self.stop_reason = "missing/stale actual observer sample (fixed3s bound)"
            raise WindowStopped(self.stop_reason)
        # Publication/log activity never refreshes the facts timestamp.

    async def _with_alarm(self, operation, observer_watch):
        if observer_watch is None:
            return await operation
        work = asyncio.create_task(operation)
        alarm = asyncio.create_task(observer_watch())
        try:
            done, _ = await asyncio.wait((work, alarm), return_when=asyncio.FIRST_COMPLETED)
            # An alarm wins a simultaneous completion. No new stage is released.
            if alarm in done:
                self._observe(await alarm)
                raise WindowStopped("observer alarm returned without an unsafe summary")
            return await work
        finally:
            for task in (work, alarm):
                if not task.done():
                    task.cancel()
            await asyncio.gather(work, alarm, return_exceptions=True)

    async def run_stage(self, name, operation, *, observer_summary=None,
                        observer_watch=None, execute=False, dry_run=False, approval=None):
        if name not in _STAGES:
            raise ValueError("stage is outside the fixed C1 recipe")
        if not execute or dry_run:
            return {"stage": name, "status": "planned", "executed": False}
        self._approve(approval)
        if self.stop_reason is not None:
            raise WindowStopped(self.stop_reason)
        if name in self.started_stages:
            self.stop_reason = "single-attempt C1 stage; no controller retry"
            raise WindowStopped(self.stop_reason)
        if name != "startup":
            if self.owner_id is None:
                raise ValueError("register exact owned container before runtime stages")
            self._observe(observer_summary)
        timeout = self.budget.child_timeout(self.clock(), requested_s=self.budget.duration_s)
        self.started_stages.add(name)
        try:
            result = await asyncio.wait_for(self._with_alarm(
                operation(deadline_unix_s=self.budget.work_deadline_unix_s), observer_watch), timeout=timeout)
            if isinstance(result, dict) and result.get("successful") is False:
                self.stop_reason = name + " incomplete/failed gate"
                raise WindowStopped(self.stop_reason)
            return result
        except BaseException as failure:
            self.stop_reason = self.stop_reason or f"{name}: {type(failure).__name__}: {failure}"
            raise

    async def cleanup(self, target_id, operation, *, execute=False, dry_run=False, approval=None):
        if (self.owner_id is None or target_id != self.owner_id
                or re.fullmatch("[0-9a-f]{64}", str(target_id)) is None):
            raise ValueError("cleanup accepts only exact registered owned container ID")
        if not execute or dry_run:
            return {"status": "planned", "executed": False, "owner_id": target_id}
        self._approve(approval)
        remaining = self.budget.hard_deadline_unix_s - self.clock()
        if remaining <= 0:
            raise TimeoutError("hard deadline reached; no new cleanup subprocess started")
        return await asyncio.wait_for(operation(owner_id=target_id,
            deadline_unix_s=self.budget.hard_deadline_unix_s), timeout=remaining)


def owned_cleanup_argv(owner_id):
    if not isinstance(owner_id, str) or re.fullmatch("[0-9a-f]{64}", owner_id) is None:
        raise ValueError("full owned container ID required")
    return ["docker", "stop", "--time", "20", owner_id]


def _positive_revision(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _record_ok(record, method, path):
    return record.get("method") == method and record.get("path") == path and record.get("status_code") == 200


def calibration_gate(method, workflows, raw_rows, *, model, run_namespace,
                     capture_records=(), capture_complete=True, worker_snapshots=(),
                     owned_worker_urls=(), phase_bounds_unix_s=None):
    """Actual producer shapes: count-only/event-only records cannot activate Ours."""
    support, runner = _pilot("h20_run_support"), _pilot("run_paired_pilot")
    gates = {workflow.workflow_id: runner.workflow_gate(support, workflow,
        [row for row in raw_rows if row["workflow_id"] == workflow.workflow_id], model)
        for workflow in workflows}
    expected = {(workflow.workflow_id, call.call_id): call for workflow in workflows for call in workflow.calls}
    measured = runner.run_gate(gates, expected_calls=len(expected), rows=raw_rows)
    result = {"successful": measured["valid"], "canonical_required": method == "ours",
        "expected_calls": len(expected), "completed_calls": sum(gate["completed_calls"] for gate in gates.values()),
        "workflow_gates": gates, "canonical_denominator": len(expected), "canonical_joined_calls": 0,
        "fallback_calls": 0, "missing_revision_calls": 0, "pressure_observed": False,
        "capture_complete": capture_complete, "failure_reasons": []}
    if method not in {"native", "ours"}:
        raise ValueError("C1 has only native and ours methods")
    if not measured["valid"]:
        result["failure_reasons"].append("incomplete model/usage/error/call-identity gate")
    for row in raw_rows:
        revision = row.get("schedule_revision")
        if revision == 0:
            result["fallback_calls"] += 1
        elif not _positive_revision(revision):
            result["missing_revision_calls"] += 1
    if method == "native":
        return result
    runtime_ids = {(run_namespace + "::" + wid, run_namespace + "::" + cid): (wid, cid) for wid, cid in expected}
    enqueued, dispatches, planned = set(), set(), set()
    for record in capture_records:
        if _record_ok(record, "POST", "/enqueue_admission"):
            metadata = ((record.get("request") or {}).get("request") or {}).get("hexgent") or {}
            identity = metadata.get("workflow_id"), metadata.get("call_id")
            if identity in runtime_ids:
                enqueued.add(identity)
        elif _record_ok(record, "POST", "/event"):
            event = record.get("request") or {}
            payload = event.get("payload") or {}
            if event.get("event_type") == "request_dispatched":
                dispatches.add((event.get("workflow_id"), event.get("call_id"), payload.get("schedule_revision"),
                    payload.get("prefill_worker_url"), payload.get("decode_worker_url")))
        elif _record_ok(record, "GET", "/wait_plan_revision"):
            ready = set()
            for entry in (record.get("response") or {}).get("entries") or []:
                identity = entry.get("workflow_id"), entry.get("call_id")
                schedule = entry.get("schedule") or {}
                revision = schedule.get("revision")
                if identity in runtime_ids and _positive_revision(revision):
                    ready.add(identity)
                    planned.add((*identity, revision, schedule.get("prefill_worker_url"), schedule.get("decode_worker_url")))
            if len(ready) >= 2:
                result["pressure_observed"] = True
    # Only a current phase/owned-worker wrapper can turn a saved queue into
    # pressure evidence. Anonymous ambient snapshots are never calibration proof.
    if phase_bounds_unix_s is not None:
        phase_start, phase_end = phase_bounds_unix_s
        for evidence in worker_snapshots:
            captured = evidence.get("captured_unix_s")
            if (evidence.get("run_namespace") != run_namespace
                    or evidence.get("worker_url") not in owned_worker_urls
                    or not isinstance(captured, (int, float))
                    or not phase_start <= captured <= phase_end):
                continue
            from urllib.parse import urlsplit
            instance = urlsplit(evidence["worker_url"]).netloc
            for snapshot in (evidence.get("payload") or {}).get("snapshots") or []:
                stamp = snapshot.get("timestamp")
                if (snapshot.get("instance_id") != instance
                        or not isinstance(stamp, (int, float)) or not phase_start <= stamp <= phase_end):
                    continue
                for waiting in (snapshot.get("queues") or {}).get("waiting") or []:
                    identity = (waiting.get("workflow_id") or waiting.get("hexgent_workflow_id"),
                                waiting.get("call_id") or waiting.get("hexgent_call_id"))
                    if identity in runtime_ids:
                        result["pressure_observed"] = True
    joined = set()
    for row in raw_rows:
        identity = row.get("workflow_id"), row.get("call_id")
        call, revision = expected.get(identity), row.get("schedule_revision")
        usage = (row.get("response") or {}).get("usage") or {}
        if (call is None or row.get("error") or not _positive_revision(revision)
                or (row.get("response") or {}).get("model") != model
                or usage.get("prompt_tokens") != call.input_len or usage.get("completion_tokens") != call.output_len):
            continue
        runtime_identity = run_namespace + "::" + identity[0], run_namespace + "::" + identity[1]
        key = (*runtime_identity, revision, row.get("prefill_worker"), row.get("decode_worker"))
        if runtime_identity in enqueued and key in planned and key in dispatches:
            joined.add(identity)
    result["canonical_joined_calls"] = len(joined)
    if not capture_complete:
        result["failure_reasons"].append("identity capture incomplete; canonical coverage unknown")
    if not joined:
        result["failure_reasons"].append("no exact calibration raw/plan/dispatch join")
    if not result["pressure_observed"]:
        result["failure_reasons"].append("no actual waiting or same-batch ready competition")
    result["successful"] = measured["valid"] and capture_complete and bool(joined) and result["pressure_observed"]
    return result


def hot_reference_rows(workflows, rows_by_workflow, *, raw_sha256_by_workflow,
                       model, implementation_revision, runtime_evidence_sha256):
    support = _pilot("h20_run_support")
    return [support.workflow_reference_from_rows(workflow, rows_by_workflow.get(workflow.workflow_id, []),
        model=model, implementation_revision=implementation_revision, runtime_evidence_sha256=runtime_evidence_sha256,
        raw_replay_sha256=raw_sha256_by_workflow[workflow.workflow_id]) for workflow in workflows]


def validate_hot_references(workflows, references, contract):
    expected = {workflow.workflow_id: len(workflow.calls) for workflow in workflows}
    reasons = []
    if len(workflows) != 4 or set(expected) != set(HOT_IDS):
        reasons.append("expected exactly original four Hot workflows")
    ids = [row.get("workflow_id") for row in references]
    if len(references) != 4 or len(set(ids)) != len(ids) or set(ids) != set(expected):
        reasons.append("four unique complete current-epoch Hot references required")
    for row in references:
        latency, count = row.get("latency_ms"), expected.get(row.get("workflow_id"))
        if (row.get("reference_kind") != "isolated_workflow_replay" or row.get("status") != "successful"
                or row.get("errors") != 0 or count is None
                or any(row.get(key) != count for key in ("expected_calls", "completed_calls", "usage_verified_calls"))
                or any(row.get(key) != contract[key] for key in ("model", "implementation_revision", "runtime_evidence_sha256"))
                or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0
                or not row.get("raw_replay_sha256")):
            reasons.append("reference fails complete/model/source/runtime/usage/elapsed contract: " + str(row.get("workflow_id")))
    return {"successful": not reasons, "expected_workflows": 4, "recorded_workflows": len(references),
        "failure_reasons": reasons, "reference_contract": contract}


async def replay_workflows(api, workflows, *, raw_path, router_url, model, run_namespace,
                           budget, now_unix_s, replay_to_raw=None):
    """Frozen incremental replay; its per-call180s never extends the global window."""
    replay = replay_to_raw or _pilot("h20_run_support").replay_to_raw
    return await asyncio.wait_for(replay(api, workflows, raw_path=raw_path, router_url=router_url,
        model=model, worker_class="H20-BF16-TP2", run_namespace=run_namespace),
        timeout=budget.child_timeout(now_unix_s, requested_s=budget.duration_s))


async def collect_hot_references(api, workflows, *, output_dir, router_url, model,
                                 implementation_revision, runtime_evidence_sha256, run_namespace,
                                 expected_limits, budget, clock=time.time, replay_to_raw=None,
                                 idle_checker=None, worker_urls=None):
    """Four-Hot producer: original DAG alone, failed/partial references remain unknown."""
    if len(workflows) != 4 or {workflow.workflow_id for workflow in workflows} != set(HOT_IDS):
        raise ValueError("original Hot10/11/12/13 workflows required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    contract = {"model": model, "implementation_revision": implementation_revision,
                "runtime_evidence_sha256": runtime_evidence_sha256}
    write_json(output_dir / "workflow-reference-contract.json", contract)
    support = _pilot("h20_run_support")
    idle = idle_checker
    references = []
    reference_bytes = 0
    with (output_dir / "workflow-references.jsonl").open("x", encoding="utf-8") as table:
        for index, workflow in enumerate(sorted(workflows, key=lambda item: item.workflow_id)):
            budget.child_timeout(clock())
            idle_path = output_dir / f"idle-before-standalone-{index:02d}.json"
            if idle is None:
                await wait_idle_async(idle_path, expected_limits, budget.work_deadline_unix_s, worker_urls=worker_urls)
            else:
                await idle(idle_path, expected_limits, budget.work_deadline_unix_s)
            raw = output_dir / f"standalone-{index:02d}.raw.jsonl"
            failure = None
            try:
                await replay_workflows(api, [replace(workflow, arrival_offset=0.0)], raw_path=raw,
                    router_url=router_url, model=model, run_namespace=f"{run_namespace}::{workflow.workflow_id}",
                    budget=budget, now_unix_s=clock(), replay_to_raw=replay_to_raw)
            except BaseException as error:
                failure = error
            rows = read_jsonl(raw) if raw.exists() else []
            reference = support.workflow_reference_from_rows(workflow, rows, model=model,
                implementation_revision=implementation_revision, runtime_evidence_sha256=runtime_evidence_sha256,
                raw_replay_sha256=sha256(raw) if raw.exists() else None)
            if failure is not None:
                reference.update(status="failed", latency_ms=None)
                reference["failure_reasons"].append(f"replay interrupted: {type(failure).__name__}: {failure}")
            line = json.dumps(reference) + "\n"
            size = len(line.encode("utf-8"))
            if reference_bytes + size > 1024 * 1024:
                gate = validate_hot_references(workflows, references, contract)
                gate.update(successful=False, pending_reference_workflow_id=workflow.workflow_id,
                    reference_table_cap_bytes=1024 * 1024,
                    workflow_reference_sha256=sha256(output_dir / "workflow-references.jsonl"))
                gate["failure_reasons"].append("four-reference JSONL aggregate cap reached before append")
                write_json(output_dir / "collection-gate.json", gate,
                    fresh=not (output_dir / "collection-gate.json").exists())
                raise WindowStopped("reference JSONL aggregate cap reached; raw workflow evidence preserved")
            table.write(line)
            table.flush()
            reference_bytes += size
            references.append(reference)
            gate = validate_hot_references(workflows, references, contract)
            gate["workflow_reference_sha256"] = sha256(output_dir / "workflow-references.jsonl")
            write_json(output_dir / "collection-gate.json", gate, fresh=index == 0)
            if failure is not None:
                raise failure
            if reference["status"] != "successful":
                raise WindowStopped("incomplete standalone reference; later stages stopped")
    return gate


class ASGIEvidenceCapture:
    """Whitelisted passive capture. Original receive/send objects pass through unchanged."""

    def __init__(self, app, append_record, *, max_body_bytes=1024 * 1024,
                 max_saved_bytes=CAPTURE_BYTES, status_writer=None):
        self.app = app
        self.append_record = append_record
        self.max_body_bytes = max_body_bytes
        self.max_saved_bytes = max_saved_bytes
        self.status_writer = status_writer
        self.saved_bytes = 0
        self.observation_complete = True
        self.incomplete_reasons = []

    def status(self):
        return {"observation_complete": self.observation_complete,
            "incomplete_reasons": list(self.incomplete_reasons), "saved_bytes": self.saved_bytes,
            "cap_bytes": self.max_saved_bytes}

    def _incomplete(self, reason):
        self.observation_complete = False
        if reason not in self.incomplete_reasons:
            self.incomplete_reasons.append(reason)

    @staticmethod
    def _schedule(schedule):
        return {key: schedule[key] for key in ("revision", "prefill_worker_url", "decode_worker_url",
            "prefill_priority", "decode_priority") if key in schedule}

    def _record(self, scope, request, response, status_code):
        path = scope["path"]
        if path == "/enqueue_admission":
            metadata = ((request or {}).get("request") or {}).get("hexgent") or {}
            state = (((request or {}).get("request") or {}).get("state") or {})
            snapshots = {}
            for worker, payload in (state.get("worker_snapshots") or {}).items():
                snapshots[worker] = {"snapshots": [{"instance_id": snapshot.get("instance_id"),
                    "timestamp": snapshot.get("timestamp"), "mode": snapshot.get("mode"),
                    "queues": {"waiting": [{key: item.get(key) for key in
                        ("workflow_id", "call_id", "hexgent_workflow_id", "hexgent_call_id")}
                        for item in (snapshot.get("queues") or {}).get("waiting") or []]}}
                    for snapshot in payload.get("snapshots") or []]}
            request = {"request": {"hexgent": {key: metadata.get(key) for key in ("workflow_id", "call_id")},
                                   "state": {"worker_snapshots": snapshots}}}
            response = {key: response[key] for key in ("dispatch_ready", "queue_position", "reason") if key in response}
        elif path == "/wait_plan_revision":
            request = None
            response = {"revision": response.get("revision"), "entries": [
                {"workflow_id": entry.get("workflow_id"), "call_id": entry.get("call_id"),
                 "schedule": self._schedule(entry.get("schedule") or {})}
                for entry in response.get("entries") or []]}
        else:
            payload = (request or {}).get("payload") or {}
            request = {key: (request or {}).get(key) for key in ("event_type", "workflow_id", "call_id", "timestamp_ms",
                "measured_service_ms", "end_to_end_latency_ms", "service_ms")}
            request["payload"] = {key: payload[key] for key in
                ("schedule_revision", "prefill_worker_url", "decode_worker_url", "stage_timing") if key in payload}
            metadata = payload.get("hexgent_metadata") or {}
            if metadata:
                request["payload"]["hexgent_metadata"] = {key: metadata[key] for key in
                    ("workflow_id", "call_id", "parent_call_ids", "workflow_arrival_ms", "input_tokens", "predicted_output_tokens") if key in metadata}
            response = {"ok": status_code == 200}
        return {"captured_unix_ns": time.time_ns(), "method": scope["method"], "path": path,
            "status_code": status_code, "request": request, "response": response}

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") not in {"/enqueue_admission", "/wait_plan_revision", "/event"}:
            return await self.app(scope, receive, send)
        request_body, response_body = bytearray(), bytearray()
        request_complete = scope.get("method") == "GET"
        response_complete, overflow = False, False
        status_code = None
        def retain(target, data):
            nonlocal overflow
            if overflow:
                return
            if len(target) + len(data) > self.max_body_bytes:
                overflow = True
                request_body.clear()
                response_body.clear()
                self._incomplete("passive capture body cap reached")
            else:
                target.extend(data)
        async def capture_receive():
            nonlocal request_complete
            message = await receive()
            if message["type"] == "http.request":
                retain(request_body, message.get("body", b""))
                request_complete = not message.get("more_body", False)
            return message
        async def capture_send(message):
            nonlocal response_complete, status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                retain(response_body, message.get("body", b""))
                response_complete = not message.get("more_body", False)
            await send(message)
        try:
            return await self.app(scope, capture_receive, capture_send)
        finally:
            try:
                if not request_complete or not response_complete:
                    self._incomplete("passive capture incomplete ASGI body")
                elif not overflow:
                    request = json.loads(request_body) if request_body else None
                    response = json.loads(response_body)
                    record = self._record(scope, request, response, status_code)
                    encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode()
                    if self.saved_bytes + len(encoded) > self.max_saved_bytes:
                        self._incomplete("passive capture total cap reached")
                    else:
                        self.append_record(record)
                        self.saved_bytes += len(encoded)
            except Exception as error:
                self._incomplete("passive capture write/JSON failure: " + type(error).__name__)
            if self.status_writer is not None:
                try:
                    self.status_writer(self.status())
                except Exception:
                    self._incomplete("passive capture status write failed")


async def watch_observer(summary_path, *, run_id, owner_id, deadline_unix_s,
                         capture_status_path=None, observer_is_alive=None,
                         clock=time.time, sleep=asyncio.sleep):
    """File-only active alarm, fixed3s facts age; no extra HTTP/TCP probes."""
    while clock() < deadline_unix_s:
        if observer_is_alive is not None and not observer_is_alive():
            raise WindowStopped("owned observer controller exited")
        try:
            summary = json.loads(Path(summary_path).read_text())
            active_capture = capture_status_path() if callable(capture_status_path) else capture_status_path
            if active_capture is not None:
                capture = json.loads(Path(active_capture).read_text())
                if capture.get("observation_complete") is not True:
                    raise WindowStopped("passive identity capture incomplete: " + str(capture.get("incomplete_reasons")))
        except (OSError, ValueError) as error:
            raise WindowStopped("missing/unreadable required evidence publication: " + str(error)) from error
        if summary.get("run_id") != run_id or summary.get("owner_id") != owner_id:
            raise WindowStopped("observer identity differs from exact owned runtime")
        sampled = summary.get("sampled_unix_s")
        if (not isinstance(sampled, (int, float)) or not math.isfinite(sampled)
                or clock() - sampled > OBSERVATION_MAX_AGE_S):
            raise WindowStopped("stale actual observer facts (fixed3s bound)")
        if (summary.get("observation_complete") is not True or summary.get("stop_requested") is not False
                or summary.get("first_error_unix_s") is not None or summary.get("incomplete_reasons") != []):
            return summary
        await sleep(max(0, min(0.25, deadline_unix_s - clock())))
    raise TimeoutError("immutable deadline reached while monitoring")


def owned_publisher_fatal(code):
    """Terminate only this approved owned sidecar; active longpolls cannot delay failure."""
    import os
    os._exit(code)


class CaptureStatusPublisher:
    """Failure invalidates the previous healthy publication, without changing ASGI bytes."""
    def __init__(self, path, *, fatal_exit=None):
        self.path = Path(path)
        self.fatal_exit = fatal_exit
        self.failed = False
        self._lock = threading.Lock()
        self._incomplete = False

    def _replace(self, temporary):
        temporary.replace(self.path)

    def publish(self, value):
        with self._lock:
            value = dict(value)
            self._incomplete = self._incomplete or value.get("observation_complete") is not True or self.failed
            if self._incomplete:
                value["observation_complete"] = False
                if not value.get("incomplete_reasons"):
                    value["incomplete_reasons"] = ["earlier capture/publication failure"]
            temporary = self.path.with_suffix(".tmp")
            try:
                write_json(temporary, value, fresh=False)
                self._replace(temporary)
            except BaseException:
                self.failed = True
                try:
                    self.path.unlink(missing_ok=True)
                    temporary.unlink(missing_ok=True)
                finally:
                    # File invalidation itself can fail. The approved sidecar
                    # must still become observably non-live immediately.
                    if self.fatal_exit is not None:
                        self.fatal_exit(70)
                raise


class RuntimeEpoch:
    """Approve static expectations first; bind one saved actual runtime before requests."""
    def __init__(self, expected_static, *, run_id, owner_id, budget):
        if re.fullmatch("[0-9a-f]{64}", owner_id) is None:
            raise ValueError("exact current owned container required")
        self.expected = dict(expected_static)
        self.run_id, self.owner_id, self.budget = run_id, owner_id, budget
        self._contract = None

    def bind_file(self, path):
        actual = json.loads(Path(path).read_text())
        digest = sha256(path)
        if self._contract is not None:
            if digest != self._contract["runtime_evidence_sha256"]:
                raise ValueError("actual epoch already bound; changed artifact cannot replace it")
            return dict(self._contract)
        mapping = {"model": "model_alias", "implementation_revision": "implementation_revision",
            "image_digest": "image_id", "gateway_elf_sha256": "elf_sha256",
            "model_revision": "model_revision", "model_config_sha256": "model_config_sha256"}
        if (any(actual.get(target) != self.expected.get(source) for source, target in mapping.items())
                or actual.get("container_id") != self.owner_id
                or actual.get("window_anchor_unix_s") != self.budget.anchor_unix_s
                or actual.get("hard_deadline_unix_s") != self.budget.hard_deadline_unix_s
                or actual.get("endpoint_observation_ready") is not True):
            raise ValueError("actual epoch differs from frozen static/source/model/image/owner/anchor contract")
        self._contract = {"model": self.expected["model"], "implementation_revision": self.expected["implementation_revision"],
            "runtime_evidence_sha256": digest}
        return dict(self._contract)

    def require_bound(self):
        if self._contract is None:
            raise WindowStopped("actual current runtime epoch must be bound before warmup/references")
        return dict(self._contract)


def prepare_plan(config):
    """Freeze only the C1 cohort and resource recipe, without starting anything."""
    directory = Path(config["runtime_input_dir"])
    names = ("sharegpt.calibration.sanitized.runtime.jsonl",
             "hotpotqa.calibration.sanitized.runtime.jsonl", "hotpotqa.heldout.sanitized.runtime.jsonl")
    records = {name: read_jsonl(directory / name) for name in names}
    sg_cal, hot_cal, hot = (records[name] for name in names)
    counts = lambda rows: sum(len(row["calls"]) for row in rows)
    if len(sg_cal) != 2 or counts(sg_cal) != 10 or len(hot_cal) != 2 or counts(hot_cal) != 58:
        raise ValueError("fixed mixed four-workflow/68-call calibration cohort required")
    if len(hot) != 4 or counts(hot) != 133 or {row["workflow_id"] for row in hot} != set(HOT_IDS):
        raise ValueError("original Hot10..13/133-call formal/reference cohort required")
    budget = WindowBudget(config["window_anchor_unix_s"], config.get("window_s", 2700), config.get("cleanup_s", 180))
    plan = {"schema_version": 1, "run_id": config["run_id"], "seed": SEED,
        "window_anchor_unix_s": budget.anchor_unix_s, "hard_deadline_unix_s": budget.hard_deadline_unix_s,
        "work_deadline_unix_s": budget.work_deadline_unix_s, "cleanup_reserve_s": budget.cleanup_s,
        "frozen_metadata": {key: config[key] for key in ("model", "implementation_revision",
            "image_digest", "gateway_elf_sha256", "model_revision", "transfer_parameters_sha256")},
        "runtime_evidence_binding": "pending_current_runtime",
        "source_root": config["source_root"], "runtime_input_dir": str(directory), "output_dir": config["output_dir"],
        "input_sha256": {name: sha256(directory / name) for name in names},
        "frozen_helper_sha256": {name: sha256(PILOT_DIR / name) for name in
            ("h20_run_support.py", "run_paired_pilot.py")},
        "calibration": {"methods": ["native", "ours"], "workflow_count": 4, "expected_calls": 68,
            "workflow_ids": [row["workflow_id"] for row in sg_cal + hot_cal], "arrival_offsets_s": [0.0] * 4,
            "mixed_trace_cohort": True, "cross_trace_concurrency": True, "service_fit_input": False,
            "native_requires_canonical": False,
            "ours_requires_exact_raw_plan_dispatch_join_and_actual_competition": True},
        "references": {"workflow_ids": list(HOT_IDS), "expected_workflows": 4,
            "reference_kind": "isolated_workflow_replay", "concurrency": "one_workflow_at_a_time",
            "original_within_workflow_fanout": True, "old_epoch_reuse": False},
        "formal_runs": [{"original_ordinal": ordinal, "method": method, "dataset": "hotpotqa",
            "load": "burst", "declared_repeat": "C1", "expected_calls": 133,
            "workflow_ids": list(HOT_IDS), "arrival_offsets_s": [0.0] * 4}
            for ordinal, method in ((7, "native"), (8, "ours"))],
        "prior_sg03": "unchanged", "future_execution_requires_approval": True,
        "budget_formula": "1.25*2*(last_arrival_s+max_current_epoch_reference_s)+60; separate180s cleanup",
        "warmup": {"scope": "once before each calibration/formal method phase", "prompt_tokens": 39,
            "completion_tokens": 8, "fresh_namespace": True, "then_frozen_wait_idle": True},
        "output_budget_mib": {"worker_logs": 32, "gateway_planner": 32, "observer": 16,
            "raw_reference_telemetry_qc": 32, "reserve": 16, "total": 128},
        "gateway_planner_budget_mib": {"native_gateway": 16, "ours_calibration_sidecar": 4,
            "ours_calibration_gateway": 4, "ours_formal_sidecar": 4, "ours_formal_gateway": 4},
        "identity_capture_scope": "inside32MiB raw/reference/capture/telemetry/QC aggregate",
        "runtime_binding_status": "UNREADY: reviewed owned startup/current peer registration and aggregate log-cap adapters required",
        "no_runtime_parity_claim": True}
    if "model_config_sha256" in config:
        plan["frozen_metadata"]["model_config_sha256"] = config["model_config_sha256"]
    if "runtime" in config:
        plan["runtime_configuration"] = config["runtime"]
        plan["source_file_sha256"] = {name: sha256(Path(config["source_root"]) / name) for name in (
            "scripts/hexgent_real_trace_collectors/start_h20_tp2_reference.sh",
            "scripts/hexgent_real_trace_collectors/h20_transport_window.py",
            "scripts/hexgent_real_trace_collectors/h20_transport_observer.py",
            "scripts/hexgent_real_trace_collectors/h20_pilot/h20_run_support.py",
            "scripts/hexgent_real_trace_collectors/h20_pilot/run_paired_pilot.py",
            "scripts/hexgent_real_trace_collectors/summarize_scheduling_replay.py",
            "scripts/hexgent_real_trace_collectors/summarize_worker_mutations.py",
            "python/sglang/hexgent/planner.py", "python/sglang/hexgent/scheduler.py",
            "python/sglang/hexgent/workflow_trace.py", "python/sglang/hexgent/latency_estimator.py",
            "python/sglang/hexgent/snapshots.py", "python/sglang/hexgent/sidecar.py",
            "python/sglang/hexgent/schemas.py", "python/sglang/hexgent/predictor.py",
            "python/sglang/srt/managers/scheduler.py", "python/sglang/srt/managers/schedule_policy.py", *CONFIG_SOURCE_PATHS)}
        plan["runtime_binding_status"] = "bound_deferred_C1_adapter; future actual owner/endpoints/epoch still required"
    return plan


CONFIG_SOURCE_PATHS = (
    "python/sglang/srt/environ.py", "python/sglang/srt/disaggregation/common/conn.py",
    "python/sglang/srt/disaggregation/mooncake/conn.py",
    "python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py")


def prove_owned_listeners(expected_roles, process_facts, socket_facts, *, pid_namespace, net_namespace):
    """Configured frontend ownership only; this is not KV registration/health proof."""
    facts = {int(row["pid"]): row for row in process_facts}
    result = {}
    for role, expected in expected_roles.items():
        root = int(expected["root_pid"])
        if root not in facts or facts[root]["start_identity"] != expected["root_start_identity"]:
            raise ValueError("configured listener launch-root identity differs: " + role)
        def owned(pid):
            seen = set()
            original = pid
            while True:
                if pid in seen or pid not in facts:
                    return False
                seen.add(pid); row = facts[pid]
                if (row.get("start_identity") != row.get("start_identity_after")
                        or row.get("pid_namespace") != pid_namespace or row.get("net_namespace") != net_namespace
                        or int(row["pgrp"]) != root):
                    return False
                if pid == root:
                    return True
                pid = int(row["ppid"])
        matches = []
        local_ip = expected.get("local_ip", "127.0.0.1")
        for socket in socket_facts:
            pid = socket.get("pid")
            inode = socket.get("socket_inode")
            if (pid in facts and owned(pid) and socket.get("state") == "LISTEN"
                    and socket.get("local_port") == expected["port"] and socket.get("local_ip") in {local_ip, "0.0.0.0"}
                    and socket.get("start_identity") == facts[pid]["start_identity"]
                    and inode not in {None, 0, "0"} and inode in facts[pid].get("fd_socket_inodes", [])):
                matches.append(socket)
        if not matches:
            raise ValueError("no current owned configured LISTEN FD: " + role)
        # Forked owned processes may share one inherited listening inode.
        result[role] = {"port": expected["port"], "local_ip": local_ip, "root_pid": root,
            "root_start_identity": expected["root_start_identity"], "pids": sorted({row["pid"] for row in matches}),
            "socket_inodes": sorted({row["socket_inode"] for row in matches}),
            "binding": "wildcard_listener" if any(row["local_ip"] == "0.0.0.0" for row in matches) else "exact_listener"}
    return result


def _startup_endpoint_roles(endpoint_records):
    rpc = re.compile(r"^I\d+\s+\S+\s+(\d+)\s+transfer_engine_impl\.cpp:\d+\] Transfer Engine RPC using P2P handshake, listening on ([0-9.]+):(\d+)")
    tcp = re.compile(r"^I\d+\s+\S+\s+(\d+)\s+tcp_transport\.cpp:\d+\] TcpTransport: listen on port (\d+)")
    roles, rpc_ips = [], {}
    for record in endpoint_records:
        stream, line = record["stream"], record["line"]
        match = rpc.match(line)
        if match:
            tid, ip, port = int(match[1]), match[2], int(match[3])
            rpc_ips[(stream, tid)] = ip
            roles.append((stream, tid, ip, port, "session", record))
        else:
            match = tcp.match(line)
            if match:
                tid = int(match[1])
                if (stream, tid) not in rpc_ips:
                    raise ValueError("TCP role has no fresh same-task RPC/source address")
                roles.append((stream, tid, rpc_ips[(stream, tid)], int(match[2]), "tcp", record))
    return roles


def build_observer_registration(*, run_id, owner_id, budget, launch_status, process_facts,
        socket_facts, endpoint_records, pid_namespace, net_namespace, cgroup_path, source_hashes, log_paths,
        known_source_status=None, listener_properties=()):
    """Bind fresh log roles to verified owned task/PID/start/FD/LISTEN facts."""
    if re.fullmatch("[0-9a-f]{64}", owner_id) is None:
        raise ValueError("full current owned container required")
    if (not all(name in source_hashes for name in CONFIG_SOURCE_PATHS)
            or not pid_namespace or not net_namespace or not log_paths):
        raise ValueError("source and namespace/log provenance required")
    roots = {row["name"]: int(row["pid"]) for row in launch_status["processes"]}
    if set(roots) != {"prefill", "decode", "baseline_gateway"} or launch_status.get("phase") != "ready":
        raise ValueError("fresh actual owned reference supervisor readiness required")
    facts = {int(row["pid"]): row for row in process_facts}
    for row in launch_status["processes"]:
        if row.get("returncode") is not None or row["pgid"] != row["pid"] or row["pid"] not in facts:
            raise ValueError("owned launch root/PGID identity differs")
    def root_of(pid):
        seen = set()
        while pid not in roots.values():
            if pid in seen or pid not in facts:
                raise ValueError("process is not a verified descendant of an owned launch root")
            seen.add(pid)
            pid = int(facts[pid]["ppid"])
        return pid
    for pid, row in facts.items():
        root = root_of(pid)
        if (row.get("start_identity") != row.get("start_identity_after")
                or row.get("pid_namespace") != pid_namespace or row.get("net_namespace") != net_namespace
                or int(row["pgrp"]) != root):
            raise ValueError("owned process start/namespace/group changed")
    roles = _startup_endpoint_roles(endpoint_records)
    peers, evidence, bound_properties = [], [], []
    def property_for(row, kind):
        if kind != "tcp" or row.get("family") != "AF_INET6" or row["local_ip"] != "::":
            return None
        matches = [proof for proof in listener_properties if all(proof.get(k) == row.get(k)
            for k in ("pid","start_identity","socket_inode","local_port","local_ip","family"))
            and proof.get("kind") == "tcp" and proof.get("run_id") == run_id and proof.get("owner_id") == owner_id
            and proof.get("net_namespace") == net_namespace and type(proof.get("ss_v6only")) is int
            and proof["ss_v6only"] == 0 and proof.get("observation_scope") == "held_listener_lifetime"
            and budget.anchor_unix_s <= proof.get("observed_unix_s",-1) < budget.work_deadline_unix_s
            and re.fullmatch("[0-9a-f]{64}",proof.get("tool_sha256",""))]
        return matches[0] if len(matches) == 1 else None
    by_role = {(name, kind): set() for name in ("prefill", "decode") for kind in ("session", "tcp")}
    for stream, tid, ip, port, kind, record in roles:
        candidates = [row for pid, row in facts.items()
                      if root_of(pid) == roots.get(stream) and tid in row.get("task_ids", [])]
        if len(candidates) != 1:
            raise ValueError("CPP numeric ID is not uniquely bound to an owned task/PID")
        process = candidates[0]
        sockets = [row for row in socket_facts if row.get("pid") == process["pid"]
            and row.get("start_identity") == process["start_identity"] and row.get("state") == "LISTEN"
            and row.get("socket_inode") in process.get("fd_socket_inodes", [])
            and row.get("socket_inode") not in {None, "0", 0}
            and row.get("local_port") == port and (row.get("local_ip") in {ip,"0.0.0.0"}
                or property_for(row,kind) is not None)]
        if len(sockets) != 1:
            # Evidence only: the original one-row acceptance condition above is unchanged.
            candidates = [row for row in socket_facts if row.get("pid") == process["pid"]
                and row.get("start_identity") == process["start_identity"] and row.get("state") == "LISTEN"
                and row.get("socket_inode") in process.get("fd_socket_inodes", [])
                and row.get("socket_inode") not in {None, "0", 0} and row.get("local_port") == port]
            def distinct_socket_count(rows):
                return len({(row["pid"], row["start_identity"], row["socket_inode"]) for row in rows})
            retained = []
            for row in candidates[:16]:
                item = {key: row[key] for key in ("pid", "start_identity", "socket_inode",
                    "local_ip", "local_port", "state")}
                family = row.get("family")
                item["family"] = family if family in ("AF_INET", "AF_INET6", 2, 10) else None
                if kind == "tcp" and family == "AF_INET6":
                    proofs = [proof for proof in listener_properties if all(proof.get(k) == row.get(k)
                        for k in ("pid","start_identity","socket_inode","local_port","local_ip","family"))
                        and proof.get("kind") == "tcp" and proof.get("run_id") == run_id and proof.get("owner_id") == owner_id
                        and proof.get("net_namespace") == net_namespace and type(proof.get("ss_v6only")) is int
                        and proof["ss_v6only"] in (0,1) and proof.get("source") == "fixed_ss_-6_exact_port_owned_inode"
                        and proof.get("observation_scope") == "held_listener_lifetime"
                        and budget.anchor_unix_s <= proof.get("observed_unix_s",-1) < budget.work_deadline_unix_s
                        and re.fullmatch("[0-9a-f]{64}",proof.get("tool_sha256",""))]
                    item["listener_property"] = {"status":"unknown","ss_v6only":None}
                    if len(proofs) == 1:
                        item["listener_property"] = {"status":"observed",**{key:proofs[0][key] for key in
                            ("ss_v6only","source","tool_sha256","observed_unix_s","observation_scope")}}
                retained.append(item)
            status_keys = ("processes", "sockets", "namespace_counts", "port_range", "cgroup", "allowed_env")
            status_values = {"ok", "denied", "sampling_race", "incomplete",
                "not_collected_wrong_namespace", "unbound_or_invalid_configuration"}
            source_status = known_source_status or {}
            diagnostic = {"schema_version": 1, "run_id": run_id, "owner_id": owner_id,
                "stream": stream, "kind": kind, "log_tid": tid, "log_offset": record["offset"],
                "registered_peer": {"ip": ip, "port": port, "kind": kind},
                "owned_pid": process["pid"], "owned_start_identity": process["start_identity"],
                "raw_row_count": len(sockets), "distinct_socket_identity_count": distinct_socket_count(sockets),
                "owned_fd_candidate_row_count": len(candidates),
                "owned_fd_candidate_distinct_socket_identity_count": distinct_socket_count(candidates),
                "candidate_limit": 16, "candidates_truncated": len(candidates) > 16,
                "candidates_omitted_count": len(candidates) - len(retained), "candidates": retained,
                "known_source_status": {key: source_status[key] for key in status_keys
                    if source_status.get(key) in status_values}}
            raise ValueError("startup role is not uniquely joined to an owned LISTEN FD inode"
                + "; owned_listener_diagnostic=" + json.dumps(diagnostic, separators=(",", ":"), sort_keys=True))
        by_role[(stream, kind)].add((ip, port))
        peer = {"ip": ip, "port": port, "kind": kind}
        if peer not in peers:
            peers.append(peer)
        property_row = property_for(sockets[0],kind)
        item = {"stream": stream, "pid": process["pid"], "start_identity": process["start_identity"],
            "log_tid": tid, "log_offset": record["offset"], "log_line": record["line"],
            "socket_inode": sockets[0]["socket_inode"],"family":sockets[0].get("family"),"local_ip":sockets[0]["local_ip"],
            "binding": "wildcard_listener" if sockets[0]["local_ip"] == "0.0.0.0" else "exact_listener", "peer":peer}
        if property_row is not None:
            item.update(binding="ipv6_wildcard_v6only_0",listener_property=property_row)
            bound_properties.append(property_row)
        evidence.append(item)
    if any(len(values) != 2 for values in by_role.values()):
        raise ValueError("TP2 needs two fresh RPC and two TCP endpoints per P/D instance")
    return {"run_id": run_id, "owner_id": owner_id, "deadline_unix_s": budget.hard_deadline_unix_s,
        "pid_namespace": pid_namespace, "net_namespace": net_namespace, "cgroup_path": cgroup_path,
        "processes": [{"pid": pid, "start_identity": row["start_identity"]} for pid, row in facts.items()],
        "peers": peers, "source_hashes": source_hashes, "log_paths": log_paths,
        "endpoint_observation_ready": True, "kv_registration_status": "unobserved_until_warmup",
        "endpoint_binding_evidence": evidence,"listener_properties":bound_properties}


def plan_sha256(plan):
    return hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def render_c1_runtime_recipe(config, *, now_unix_s):
    """Pure concrete argv/environment recipe; no resource discovery or process launch."""
    runtime = config["runtime"]
    budget = WindowBudget(config["window_anchor_unix_s"], config.get("window_s", 2700), config.get("cleanup_s", 180))
    remaining = budget.child_timeout(now_unix_s, requested_s=budget.duration_s)
    ports = runtime["ports"]
    if len(set(ports.values())) != len(ports) or len(runtime["gpu_uuids"]) != 4:
        raise ValueError("four explicitly allocated GPU UUIDs and distinct frozen ports required")
    source_dir = runtime["source_dir"]
    reference = Path(config["source_root"]) / "scripts/hexgent_real_trace_collectors/start_h20_tp2_reference.sh"
    environment = {"SOURCE_DIR": source_dir, "MODEL_PATH": runtime["model_path"], "SERVED_MODEL_NAME": config["model"],
        "GATEWAY_BINARY": runtime["gateway_binary"], "RUN_DIR": runtime["run_dir"] + "/runtime/workers",
        "PREFILL_PORT": str(ports["prefill"]), "DECODE_PORT": str(ports["decode"]),
        "BOOTSTRAP_PORT": str(ports["bootstrap"]), "BASELINE_ROUTER_PORT": str(ports["native"]),
        "BASELINE_METRICS_PORT": str(ports["native_metrics"]), "RUN_TIMEOUT_S": str(remaining),
        "READY_TIMEOUT_S": str(min(600.0, remaining)), "IMPLEMENTATION_REVISION": config["implementation_revision"]}
    argv = ["docker", "run", "--pull=never", "--rm", "-d", "--init", "--name", runtime["container_name"],
        "--cidfile", str(Path(config["output_dir"]) / "container.id"), "--label", "c1-run=" + config["run_id"],
        "--log-driver", "none", "--read-only", "--network", "host", "--user", runtime["uid_gid"],
        "--gpus", '\"device=' + ",".join(runtime["gpu_uuids"]) + '\"', "--cpuset-cpus", runtime["cpuset_cpus"],
        "--cpus", "6", "--memory", "256g", "--memory-swap", "256g", "--shm-size", "1g",
        "--tmpfs", "/runtime:rw,exec,size=7g,mode=1777"]
    for host, container, readonly in ((runtime["source_host_path"], source_dir, True),
            (runtime["model_host_path"], "/model-cache", True), (runtime["artifacts_host_path"], "/artifacts", False),
            (runtime["inputs_host_path"], "/inputs", True)):
        argv += ["--mount", f"type=bind,src={host},dst={container}" + (",readonly" if readonly else "")]
    for key, value in {"USER": "fred", "LOGNAME": "fred", "HOME": "/runtime/home", "TMPDIR": "/runtime/tmp", "HF_HOME": "/runtime/hf",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1", "PYTHONPATH": source_dir + "/python", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}.items():
        argv += ["--env", key + "=" + value]
    argv += ["--entrypoint", "python3", config["image_digest"], source_dir + "/scripts/hexgent_real_trace_collectors/h20_transport_window.py",
        "runtime-supervisor", "--config", runtime["config_container_path"], "--approval-file", runtime["approval_container_path"],
        "--owner-record", runtime["owner_container_path"], "--plan", runtime.get("plan_container_path", runtime["run_dir"] + "/approved-plan.json")]
    return {"docker_argv": argv, "reference_argv": ["bash", source_dir + "/scripts/hexgent_real_trace_collectors/start_h20_tp2_reference.sh"],
        "reference_environment": environment, "hard_deadline_unix_s": budget.hard_deadline_unix_s,
        "work_deadline_unix_s": budget.work_deadline_unix_s, "worker_log_budget_mib": 32,
        "native_gateway_log_budget_mib": 16, "ours_stdout_caps_mib": [4, 4, 4, 4],
        "raw_reference_capture_telemetry_qc_budget_mib": 32, "observer_budget_mib": 16,
        "persistent_total_budget_mib": 128, "endpoint_observation_ready_required": True,
        "kv_registration_required_before_warmup": False,
        "source_sha256": {str(reference.relative_to(config["source_root"])): sha256(reference)}}


class DeferredCommandExecutor:
    """Only called by explicit live execution; constructor/import do not sample or start."""
    async def run(self, argv, *, deadline_unix_s):
        remaining = deadline_unix_s - time.time()
        if remaining <= 0:
            raise TimeoutError("immutable command deadline reached")
        process = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        capped = False
        terminated = False
        def terminate_owned():
            nonlocal terminated
            if not terminated and process.returncode is None:
                terminated = True
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        async def drain(stream):
            nonlocal capped
            saved = bytearray(); drained = 0
            while True:
                block = await stream.read(65536)
                if not block:
                    return bytes(saved).decode(errors="replace"), drained
                drained += len(block)
                keep = min(len(block), max(0, 1024 * 1024 - len(saved)))
                saved.extend(block[:keep])
                if keep != len(block):
                    capped = True
                    terminate_owned()
                # Continue draining even after retention cap/termination. No
                # producer pipe is abandoned or called complete after truncation.
        stdout_task = asyncio.create_task(drain(process.stdout))
        stderr_task = asyncio.create_task(drain(process.stderr))
        wait_task = asyncio.create_task(process.wait())
        try:
            stdout, stderr, code = await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task, wait_task), remaining)
            if capped:
                raise WindowStopped("owned command output cap reached; child terminated and buffered pipes drained")
            return {"returncode": code, "stdout": stdout[0], "stderr": stderr[0],
                "stdout_drained_bytes": stdout[1], "stderr_drained_bytes": stderr[1]}
        finally:
            if process.returncode is None:
                terminate_owned()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            for task in (stdout_task, stderr_task, wait_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stdout_task, stderr_task, wait_task, return_exceptions=True)


class C1OwnedLauncher:
    def __init__(self, plan, recipe, *, executor=None, clock=time.time):
        self.plan, self.recipe = plan, recipe
        self.executor = executor or DeferredCommandExecutor()
        self.window = TransportWindow(run_id=plan["run_id"], budget=WindowBudget(plan["window_anchor_unix_s"],
            plan["hard_deadline_unix_s"] - plan["window_anchor_unix_s"], plan["cleanup_reserve_s"]),
            plan_sha256=plan_sha256(plan), clock=clock)

    async def start(self, *, execute=False, dry_run=False, approval=None):
        async def start(*, deadline_unix_s):
            cid_path = Path(self.recipe["docker_argv"][self.recipe["docker_argv"].index("--cidfile") + 1])
            if cid_path.exists():
                raise ValueError("owned launch CID file must be fresh")
            try:
                result = await self.executor.run(self.recipe["docker_argv"], deadline_unix_s=deadline_unix_s)
            except BaseException:
                if cid_path.exists():
                    self.window.register_owned_container(cid_path.read_text().strip())
                raise
            if result["returncode"] != 0:
                if cid_path.exists():
                    self.window.register_owned_container(cid_path.read_text().strip())
                raise WindowStopped("owned docker startup failed: " + result["stderr"])
            owner_id = result["stdout"].strip()
            self.window.register_owned_container(owner_id)
            return {"owner_id": owner_id, "executed": True}
        return await self.window.run_stage("startup", start, execute=execute, dry_run=dry_run, approval=approval)

    async def cleanup(self, target_id, *, execute=False, dry_run=False, approval=None):
        async def stop(*, owner_id, deadline_unix_s):
            return await self.executor.run(owned_cleanup_argv(owner_id), deadline_unix_s=deadline_unix_s)
        return await self.window.cleanup(target_id, stop, execute=execute, dry_run=dry_run, approval=approval)


def _observer_api():
    path = Path(__file__).with_name("h20_transport_observer.py")
    name = "_c1_actual_observer"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _stat_fields(text, expected_pid):
    actual = int(text.split("(", 1)[0])
    fields = text.rsplit(")", 1)[1].split()
    if actual != expected_pid:
        raise WindowStopped("owned PID identity differs")
    return {"pid": actual, "ppid": int(fields[1]), "pgrp": int(fields[2]), "start_identity": fields[19]}


def _read_owned_processes(root_pids, reader, *, supervisor_pid):
    proc_root = "/proc"
    pidns, netns = reader.readlink(proc_root + "/self/ns/pid"), reader.readlink(proc_root + "/self/ns/net")
    facts, visited = [], set()
    def visit(pid, parent=None):
        if pid in visited:
            return
        visited.add(pid)
        base = f"{proc_root}/{pid}"
        record = _stat_fields(reader.read_text(base + "/stat"), pid)
        if parent is not None and record["ppid"] != parent:
            raise WindowStopped("owned descendant parent changed during registration")
        record.update(pid_namespace=reader.readlink(base + "/ns/pid"), net_namespace=reader.readlink(base + "/ns/net"),
            task_ids=[int(item.name) for item in (Path(base) / "task").iterdir()],
            fd_socket_inodes=[match[1] for target in reader.list_fd_targets(pid)
                if (match := re.fullmatch(r"socket:\[(\d+)\]", target))])
        record["start_identity_after"] = _stat_fields(reader.read_text(base + "/stat"), pid)["start_identity"]
        facts.append(record)
        for child in reader.read_text(base + f"/task/{pid}/children").split():
            visit(int(child), pid)
    for pid in root_pids:
        visit(int(pid), supervisor_pid)
    return facts, pidns, netns

def collect_startup_registration(config, plan, owner_id, status, *, supervisor_pid, reader=None):
    """Deferred owned-only /proc traversal, fresh startup roles, and actual FD/listener join."""
    observer = _observer_api()
    reader = reader or observer.LinuxProcReader()
    runtime = config["runtime"]
    facts, pidns, netns = _read_owned_processes([row["pid"] for row in status["processes"]], reader,
        supervisor_pid=supervisor_pid)
    log_paths = [str(Path(runtime["run_dir"]) / "workers" / (name + ".log")) for name in ("prefill", "decode")]
    records = []
    for stream, path in zip(("prefill", "decode"), log_paths):
        offset = 0
        for line in reader.read_text(path).splitlines(keepends=True):
            records.append({"stream": stream, "offset": offset, "line": line.rstrip("\r\n")})
            offset += len(line.encode())
    source_root = Path(runtime["source_dir"])
    source_hashes = {name: sha256(source_root / name) for name in CONFIG_SOURCE_PATHS}
    if any(source_hashes[name] != plan["source_file_sha256"][name] for name in CONFIG_SOURCE_PATHS):
        raise WindowStopped("actual configuration parser source differs from approved source")
    # Candidate ports are private inputs only. No readiness is claimed until
    # the final owned-task/FD/listener join below succeeds.
    candidates = []
    for record in records:
        match = re.search(r"Transfer Engine RPC .* listening on ([0-9.]+):(\d+)", record["line"])
        if match:
            candidates.append({"ip": match[1], "port": int(match[2]), "kind": "session"})
    if not candidates:
        raise WindowStopped("no fresh owned startup RPC endpoints")
    provisional = {"run_id": plan["run_id"], "owner_id": owner_id,
        "deadline_unix_s": plan["hard_deadline_unix_s"], "pid_namespace": pidns, "net_namespace": netns,
        "processes": [{"pid": item["pid"], "start_identity": item["start_identity"]} for item in facts],
        "peers": candidates, "source_hashes": source_hashes, "cgroup_path": "/sys/fs/cgroup", "log_paths": log_paths}
    actual = observer.LinuxIPv4FactProducer(provisional, reader=reader).collect(time.time())
    if actual.get("read_errors") or any(actual["source_status"].get(name) != "ok" for name in observer.REQUIRED_SOURCES):
        raise WindowStopped("startup ownership/socket/configuration observation incomplete: " + str(actual.get("read_errors")))
    budget = WindowBudget(plan["window_anchor_unix_s"], plan["hard_deadline_unix_s"] - plan["window_anchor_unix_s"], plan["cleanup_reserve_s"])
    targets = []
    roots = {row["name"]:row["pid"] for row in status["processes"]}
    for stream,tid,ip,port,kind,record in _startup_endpoint_roles(records):
        if kind != "tcp":
            continue
        processes = [row for row in facts if row["pgrp"] == roots.get(stream) and tid in row["task_ids"]]
        if len(processes) != 1:
            raise WindowStopped("DATA role does not identify one current owned task")
        process = processes[0]
        rows = [row for row in actual["sockets"] if row["pid"] == process["pid"]
            and row["start_identity"] == process["start_identity"] and row["state"] == "LISTEN"
            and row["socket_inode"] in process["fd_socket_inodes"] and row["local_port"] == port]
        if any(row.get("family") == "AF_INET6" for row in rows):
            if len(rows) != 1:
                raise WindowStopped("DATA listener rows remain ambiguous without deduplication")
            targets.append({**rows[0],"stream":stream,"kind":kind,"net_namespace":netns,"log_tid":tid})
    properties = (observer.collect_ipv6_listener_properties(targets,registration=provisional,reader=reader,
        ss_tool=runtime.get("ss_tool"),deadline_unix_s=budget.work_deadline_unix_s) if targets else [])
    registration = build_observer_registration(run_id=plan["run_id"], owner_id=owner_id, budget=budget, launch_status=status,
        process_facts=facts, socket_facts=actual["sockets"], endpoint_records=records, pid_namespace=pidns,
        net_namespace=netns, cgroup_path="/sys/fs/cgroup", source_hashes=source_hashes, log_paths=log_paths,
        known_source_status=actual["source_status"],listener_properties=properties)
    roots = {row["name"]: row["pid"] for row in status["processes"]}
    identities = {row["pid"]: row["start_identity"] for row in facts}
    ports = runtime["ports"]
    roles = {role: {"port": ports[port], "local_ip": "127.0.0.1", "root_pid": roots[name],
        "root_start_identity": identities[roots[name]]} for role, name, port in (
            ("prefill_http", "prefill", "prefill"), ("prefill_bootstrap", "prefill", "bootstrap"),
            ("decode_http", "decode", "decode"), ("native_http", "baseline_gateway", "native"))}
    registration["configured_listeners"] = prove_owned_listeners(roles, facts, actual["sockets"],
        pid_namespace=pidns, net_namespace=netns)
    return registration


class RuntimeChildren:
    """Only processes created by this C1 supervisor; fixed per-stream drain caps."""
    def __init__(self):
        self.entries = []
        self.incomplete_reasons = []

    def start(self, name, argv, env, path, *, cap_bytes):
        import os
        import subprocess
        process = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, bufsize=0)
        entry = {"name": name, "process": process, "pgid": process.pid, "saved_bytes": 0,
            "drained_bytes": 0, "truncated": False, "write_error": None, "log_path": str(path)}
        def drain():
            try:
                with Path(path).open("xb") as sink:
                    while True:
                        block = os.read(process.stdout.fileno(), 65536)
                        if not block:
                            break
                        entry["drained_bytes"] += len(block)
                        keep = min(len(block), max(0, cap_bytes - entry["saved_bytes"]))
                        sink.write(block[:keep]); sink.flush()
                        entry["saved_bytes"] += keep
                        entry["truncated"] = entry["drained_bytes"] > cap_bytes
            except OSError as error:
                entry["write_error"] = str(error)
                # Continue draining after write failure so child pipes cannot block.
                while os.read(process.stdout.fileno(), 65536):
                    pass
        thread = threading.Thread(target=drain, daemon=True)
        entry["thread"] = thread
        self.entries.append(entry); thread.start()
        _bounded_qc_json(Path(path).with_suffix(".process.json"), {"name": name, "pid": process.pid,
            "pgid": process.pid, "argv": argv, "log_path": str(path), "cap_bytes": cap_bytes,
            "configuration": {key: value for key, value in env.items() if key in (_observer_api().ALLOWED_ENV | {
                "AGENTIC_PD_OURS_PLACEMENT_STRATEGY", "HEXGENT_REVISION_STATE_PATH", "CUDA_VISIBLE_DEVICES", "PYTHONPATH",
                "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"})}})
        return entry

    def healthy(self):
        return not self.incomplete_reasons and all((entry.get("stopping", False) or entry["process"].poll() is None)
            and not entry["truncated"] and entry["write_error"] is None for entry in self.entries)

    async def stop(self, hard_deadline, entries=None):
        import os
        import signal
        import subprocess
        selected = list(self.entries if entries is None else entries)
        failures = []
        for entry in selected:
            code = entry["process"].poll()
            if not entry.get("stopping", False) and code is not None:
                reason = entry["name"] + ": unexpected owned exit before retirement: " + str(code)
                failures.append(reason)
                if reason not in self.incomplete_reasons:
                    self.incomplete_reasons.append(reason)
            entry["stopping"] = True
        for entry in reversed(selected):
            process = entry["process"]
            if process.poll() is None:
                if os.getpgid(process.pid) != entry["pgid"]:
                    raise WindowStopped("owned process group identity changed; no signal sent")
                os.killpg(entry["pgid"], signal.SIGTERM)
        grace = 35 if any(entry["name"] == "reference-supervisor" for entry in selected) else 5
        until = min(hard_deadline, time.time() + grace)
        while time.time() < until and any(entry["process"].poll() is None for entry in selected):
            await asyncio.sleep(0.1)
        for entry in reversed(selected):
            if entry["process"].poll() is None:
                os.killpg(entry["pgid"], signal.SIGKILL)
        for entry in selected:
            remaining = max(0.01, min(2, hard_deadline - time.time()))
            waited_code = None
            try:
                waited_code = await asyncio.to_thread(entry["process"].wait, timeout=remaining)
            except subprocess.TimeoutExpired:
                failures.append(entry["name"] + ": owned process wait timed out")
            entry["thread"].join(timeout=max(0, min(1, hard_deadline - time.time())))
            if entry["name"] == "ours-sidecar" and (waited_code == 70 or entry["process"].poll() == 70):
                reason = "ours-sidecar: owned publisher fatal exit70 during retirement"
                failures.append(reason)
                if reason not in self.incomplete_reasons:
                    self.incomplete_reasons.append(reason)
            if entry["truncated"] or entry["write_error"] is not None or entry["thread"].is_alive():
                failures.append(entry["name"] + ": final owned capture drain incomplete")

        if entries is not None:
            self.entries = [entry for entry in self.entries if entry not in selected]
        if failures:
            self.incomplete_reasons.extend(reason for reason in failures if reason not in self.incomplete_reasons)
            raise WindowStopped("; ".join(failures))


class C1RawBudget:
    """12MiB aggregate raw allocation; W2 slices retain4MiB inside the32MiB category."""
    def __init__(self):
        self.saved_bytes = 0

    async def replay(self, api, workflows, *, raw_path, router_url, model, worker_class, run_namespace):
        support = _pilot("h20_run_support")
        for workflow in workflows:
            for call in workflow.calls:
                if set(call.planner_hints) != {"model_config", "hardware_params"}:
                    raise ValueError("current static hints only; no service/future profile oracle")
        owned, stopped = set(), False
        with Path(raw_path).open("x", encoding="utf-8") as target:
            def append(row):
                nonlocal stopped
                line = json.dumps(row, ensure_ascii=False) + "\n"
                size = len(line.encode())
                if self.saved_bytes + size > 12 * 1024 * 1024:
                    stopped = True
                    raise WindowStopped("aggregate raw evidence cap reached")
                target.write(line); target.flush(); self.saved_bytes += size
                if row.get("error"):
                    stopped = True
                    raise WindowStopped("first client error persisted; stop remaining requests/stages")
            incremental = support.incremental_replayer_class(api, append)
            class Guarded(incremental):
                async def _run_workflow(self, *args, **kwargs):
                    if stopped:
                        raise WindowStopped("first error latched before pending workflow entry")
                    task = asyncio.current_task(); owned.add(task)
                    try:
                        return await super()._run_workflow(*args, **kwargs)
                    finally:
                        owned.discard(task)
                async def _run_call(self, *args, **kwargs):
                    if stopped:
                        raise WindowStopped("first error latched before pending call/HTTP entry")
                    task = asyncio.current_task(); owned.add(task)
                    try:
                        return await super()._run_call(*args, **kwargs)
                    finally:
                        owned.discard(task)
            replayer = Guarded(router_url=router_url, profile_table=None, model=model, worker_class=worker_class,
                exact_output_length=True, timeout_s=180, endpoint="/v1/chat/completions", run_namespace=run_namespace)
            try:
                return await replayer.replay(workflows, arrival_rate=None)
            finally:
                stopped = True
                pending = tuple(owned)
                for task in pending:
                    if not task.done():
                        task.cancel()
                # Explicit source-created task ownership only. The observer alarm
                # survives; completed raw rows are flushed before this file closes.
                await asyncio.gather(*pending, return_exceptions=True)


class _TelemetryPath:
    """Bound the real actor writer and publish every actual telemetry I/O failure."""
    def __init__(self, path, cap_bytes, incomplete):
        self.path, self.cap_bytes, self.incomplete = Path(path), cap_bytes, incomplete
        self.lock = threading.Lock()

    def _io_failure(self, operation, error):
        try:
            self.incomplete("planner telemetry " + operation + " failed: " + type(error).__name__)
        except BaseException as publication_error:
            # The I/O failure remains the raised error; notification failure is
            # retained as its explicit cause rather than replacing it.
            raise error from publication_error

    def open(self, mode, encoding=None):
        parent = self
        class Sink:
            def __enter__(self):
                parent.lock.acquire()
                try:
                    self.target = parent.path.open("ab")
                except OSError as error:
                    try:
                        parent._io_failure("open", error)
                    finally:
                        parent.lock.release()
                    raise
                return self

            def write(self, value):
                data = value.encode("utf-8")
                try:
                    position = self.target.tell()
                except OSError as error:
                    parent._io_failure("tell", error)
                    raise
                if position + len(data) > parent.cap_bytes:
                    parent.incomplete("planner telemetry cap reached")
                else:
                    try:
                        self.target.write(data)
                        self.target.flush()
                    except OSError as error:
                        parent._io_failure("write/flush", error)
                        raise
                return len(value)

            def __exit__(self, exc_type, exc, traceback):
                try:
                    try:
                        self.target.close()
                    except OSError as error:
                        parent._io_failure("close", error)
                        if isinstance(exc, OSError):
                            raise exc.with_traceback(traceback) from error
                        raise
                finally:
                    parent.lock.release()
        return Sink()


def _bounded_qc_json(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    root = _bounded_qc_json.root
    used = sum(item.stat().st_size for item in root.rglob("*.json") if item.is_file())
    previous = Path(path).stat().st_size if Path(path).exists() else 0
    if used - previous + len(encoded) > 4 * 1024 * 1024:
        raise WindowStopped("aggregate JSON QC/reference metadata cap reached")
    write_json(path, value, fresh=not Path(path).exists())


async def async_get_json(url, deadline):
    """The existing fixed GET boundary, asynchronously cancellable in flight."""
    import aiohttp
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("immutable GET deadline reached")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=min(5.0, remaining))) as response:
            if response.status != 200:
                raise ValueError("snapshot/readiness HTTP status differs from200")
            return await response.json(content_type=None)


async def wait_idle_async(path, expected, deadline, *, worker_urls=None, getter=None, sleep=None):
    """Same frozen gate: two empty snapshots0.25s apart, max30s/original deadline."""
    runner = _pilot("run_paired_pilot")
    worker_urls = worker_urls or {"prefill": runner.PREFILL, "decode": runner.DECODE}
    getter, sleep = getter or async_get_json, sleep or asyncio.sleep
    started = time.monotonic(); until = min(deadline, time.time() + 30)
    record = {"poll_count": 0, "first": None, "final": None, "success": False}
    consecutive = 0
    try:
        while time.time() < until:
            actual = {}; record["poll_count"] += 1; record["final"] = actual
            if record["first"] is None:
                record["first"] = actual
            for mode in ("prefill", "decode"):
                actual[mode] = await getter(worker_urls[mode] + "/hexgent/pd_snapshot", until)
            empty = all(runner.snapshot_is_empty(actual[mode], expected[mode]) for mode in ("prefill", "decode"))
            consecutive = consecutive + 1 if empty else 0
            if consecutive == 2:
                record["success"] = True
                return record
            remaining = until - time.time()
            if remaining <= 0:
                break
            await sleep(min(0.25, remaining))
        raise TimeoutError("workers not idle within30 seconds/original deadline")
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["elapsed_ms"] = (time.monotonic() - started) * 1000
        write_json(path, record, fresh=not Path(path).exists())


async def wait_ready_async(url, processes, deadline, *, getter=None, sleep=None):
    """The original models readiness GET/0.25s/max30s policy without a thread poller."""
    getter, sleep = getter or async_get_json, sleep or asyncio.sleep
    until = min(deadline, time.time() + 30); last = None
    while time.time() < until:
        for process, _, _ in processes.entries:
            if process.poll() is not None:
                raise WindowStopped("owned service exited before readiness")
        try:
            return await getter(url, until)
        except Exception as error:
            last = f"{type(error).__name__}: {error}"
        remaining = until - time.time()
        if remaining <= 0:
            break
        await sleep(min(0.25, remaining))
    raise RuntimeError("owned service readiness failed: " + str(last))


async def _idle(path, expected, budget, *, worker_urls=None):
    budget.child_timeout(time.time())
    return await wait_idle_async(path, expected, budget.work_deadline_unix_s, worker_urls=worker_urls)


class C1RuntimeOperations:
    """Concrete fixed C1 replay stages inside the future approved owned container."""
    def __init__(self, config, plan, approval, owner_id):
        self.config, self.plan, self.approval, self.owner_id = config, plan, approval, owner_id
        self.runtime = config["runtime"]
        self.root = Path(self.runtime["run_dir"]) / "runtime"
        self.root.mkdir(parents=True, exist_ok=False)
        _bounded_qc_json.root = self.root
        self.budget = WindowBudget(plan["window_anchor_unix_s"], plan["hard_deadline_unix_s"] - plan["window_anchor_unix_s"], plan["cleanup_reserve_s"])
        self.window = TransportWindow(run_id=plan["run_id"], budget=self.budget, plan_sha256=plan_sha256(plan))
        self.window.register_owned_container(owner_id)
        self.children = RuntimeChildren()
        self.raw_budget = C1RawBudget()
        self.api = _pilot("h20_run_support").load_workflow_api(self.runtime["source_dir"])
        self.expected = self.runtime["expected_worker_limits"]
        self.model = plan["frozen_metadata"]["model"]
        self.calibration = [replace(item, arrival_offset=0.0) for dataset in ("sharegpt", "hotpotqa")
            for item in self.api.load_workflow_traces_jsonl(Path("/inputs") / (dataset + ".calibration.sanitized.runtime.jsonl"))]
        self.hot = [replace(item, arrival_offset=0.0) for item in self.api.load_workflow_traces_jsonl(
            "/inputs/hotpotqa.heldout.sanitized.runtime.jsonl")]
        self.observer_entry = None
        self.capture_status = None
        self.ours_entries = []
        self.phase_records = []
        self.w2_saved_bytes = 0

    def endpoint(self, name):
        return "http://127.0.0.1:" + str(self.runtime["ports"][name])

    def summary(self):
        return json.loads((self.root / "observer" / "summary.latest.json").read_text())

    def healthy(self):
        if not self.children.healthy():
            return False
        worker = json.loads((self.root / "workers" / "status.json").read_text())
        return (worker.get("phase") == "ready" and not worker.get("error") and
            all(not row.get("truncated") and not row.get("write_error") and row.get("returncode") is None
                for row in worker.get("processes", [])))

    async def alarm(self):
        return await watch_observer(self.root / "observer" / "summary.latest.json", run_id=self.plan["run_id"],
            owner_id=self.owner_id, deadline_unix_s=self.budget.work_deadline_unix_s,
            capture_status_path=lambda: self.capture_status, observer_is_alive=self.healthy)

    async def run_stage(self, name, operation):
        return await self.window.run_stage(name, operation, observer_summary=self.summary(), observer_watch=self.alarm,
            execute=True, approval=self.approval)

    async def startup(self):
        import os
        source = Path(self.runtime["source_dir"])
        for name, digest in self.plan["source_file_sha256"].items():
            if sha256(source / name) != digest:
                raise WindowStopped("actual source differs from approved snapshot: " + name)
        for name, digest in self.plan["input_sha256"].items():
            if sha256(Path("/inputs") / name) != digest:
                raise WindowStopped("actual immutable input differs: " + name)
        if sha256(self.runtime["gateway_binary"]) != self.plan["frozen_metadata"]["gateway_elf_sha256"]:
            raise WindowStopped("actual ELF differs from approved R2 binary")
        recipe = render_c1_runtime_recipe({**self.config, "source_root": str(source)}, now_unix_s=time.time())
        env = os.environ.copy(); env.update(recipe["reference_environment"])
        self.reference = self.children.start("reference-supervisor", recipe["reference_argv"], env,
            self.root / "reference-supervisor.log", cap_bytes=4 * 1024 * 1024)
        status_path = self.root / "workers" / "status.json"
        until = min(self.budget.work_deadline_unix_s, time.time() + float(env["READY_TIMEOUT_S"]))
        while time.time() < until:
            if not self.children.healthy():
                raise WindowStopped("owned reference supervisor exited or capture incomplete during startup")
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status["phase"] == "ready":
                    break
                if status.get("error"):
                    raise WindowStopped("owned reference startup failed: " + str(status["error"]))
            await asyncio.sleep(0.25)
        else:
            raise TimeoutError("original window startup deadline reached")
        registration_config = {**self.config, "runtime": {**self.runtime, "run_dir": str(self.root)}}
        registration = collect_startup_registration(registration_config, self.plan, self.owner_id, status,
            supervisor_pid=self.reference["process"].pid)
        registration_path = self.root / "observer-registration.json"
        _bounded_qc_json(registration_path, registration)
        observer_argv = [sys.executable, str(Path(__file__)), "observer-child", "--registration-file", str(registration_path),
            "--output-dir", str(self.root / "observer"), "--approval-file", self.runtime["approval_container_path"],
            "--plan-sha256", plan_sha256(self.plan), "--run-id", self.plan["run_id"],
            "--cpu", self.runtime["cpuset_cpus"].split(",")[-1]]
        for path in registration["log_paths"]:
            observer_argv += ["--log", path]
        self.observer_entry = self.children.start("transport-observer", observer_argv, env,
            self.root / "observer-controller.log", cap_bytes=1024 * 1024)
        until = min(self.budget.work_deadline_unix_s, time.time() + 3)
        while not (self.root / "observer" / "summary.latest.json").exists():
            if time.time() >= until or not self.children.healthy():
                raise WindowStopped("owned observer did not publish complete first sample")
            await asyncio.sleep(0.05)
        self.window._observe(self.summary())
        async def startup_idle(*, deadline_unix_s):
            return await _idle(self.root / "idle-startup.json", self.expected, self.budget,
                worker_urls={mode: self.endpoint(mode) for mode in ("prefill", "decode")})
        idle = await self.run_stage("startup-idle", startup_idle)
        actual = {"implementation_revision": self.plan["frozen_metadata"]["implementation_revision"],
            "container_id": self.owner_id, "window_anchor_unix_s": self.budget.anchor_unix_s,
            "hard_deadline_unix_s": self.budget.hard_deadline_unix_s,
            "image_id": self.plan["frozen_metadata"]["image_digest"],
            "image_identity_source": "immutable docker-run image argument in exact-owned host launch record",
            "elf_sha256": sha256(self.runtime["gateway_binary"]), "model_alias": self.model,
            "model_revision": self.plan["frozen_metadata"]["model_revision"],
            "model_config_sha256": sha256(Path(self.runtime["model_path"]) / "config.json"),
            "source_file_sha256": {name: sha256(source / name) for name in self.plan["source_file_sha256"]},
            "worker_snapshots": idle["final"], "endpoint_observation_ready": registration["endpoint_observation_ready"],
            "kv_registration_status": "unobserved_until_warmup", "timestamp_unix_s": time.time()}
        path = self.root / "actual-runtime-evidence.json"
        _bounded_qc_json(path, actual)
        self.epoch = RuntimeEpoch(self.plan["frozen_metadata"], run_id=self.plan["run_id"], owner_id=self.owner_id, budget=self.budget)
        self.contract = self.epoch.bind_file(path)
        _bounded_qc_json(self.root / "workflow-reference-contract.json", self.contract)

    def verify_ours_frontends(self):
        import os
        observer = _observer_api(); reader = observer.LinuxProcReader()
        if any(entry["process"].poll() is not None for entry in self.ours_entries):
            raise WindowStopped("owned Ours frontend exited before ownership proof")
        facts, pidns, netns = _read_owned_processes([entry["process"].pid for entry in self.ours_entries],
            reader, supervisor_pid=os.getpid())
        registration = json.loads((self.root / "observer-registration.json").read_text())
        if pidns != registration["pid_namespace"] or netns != registration["net_namespace"]:
            raise WindowStopped("Ours frontends differ from the owned runtime namespace")
        provisional = {**registration, "listener_properties": [],
            "processes": [{"pid": row["pid"], "start_identity": row["start_identity"]} for row in facts]}
        actual = observer.LinuxIPv4FactProducer(provisional, reader=reader).collect(time.time())
        if actual.get("read_errors") or any(actual["source_status"].get(name) != "ok" for name in observer.REQUIRED_SOURCES):
            raise WindowStopped("owned Ours frontend FD observation incomplete")
        if any(entry["process"].poll() is not None for entry in self.ours_entries):
            raise WindowStopped("Ours frontend exited during ownership proof")
        identities = {row["pid"]: row["start_identity"] for row in facts}
        roles = {role: {"port": self.runtime["ports"][port], "local_ip": "127.0.0.1",
            "root_pid": entry["process"].pid, "root_start_identity": identities[entry["process"].pid]}
            for role, port, entry in (("sidecar_http", "sidecar", self.ours_entries[0]),
                                     ("ours_http", "ours", self.ours_entries[1]))}
        return prove_owned_listeners(roles, facts, actual["sockets"], pid_namespace=pidns, net_namespace=netns)

    async def start_ours(self, directory):
        import os
        transfer_path = self.runtime["transfer_parameters_path"]
        if sha256(transfer_path) != self.plan["frozen_metadata"]["transfer_parameters_sha256"]:
            raise WindowStopped("approved transfer parameters changed; no fit/fallback")
        runner = _pilot("run_paired_pilot")
        transfer = json.loads(Path(transfer_path).read_text())
        env, _, gateway, _ = runner.method_commands(Path(self.runtime["source_dir"]), directory, transfer)
        ports = self.runtime["ports"]
        def replace_flag(flag, value):
            gateway[gateway.index(flag) + 1] = str(value)
        for flag, value in (("--port", ports["ours"]), ("--prometheus-port", ports["ours_metrics"]),
                ("--prefill", self.endpoint("prefill")), ("--decode", self.endpoint("decode")),
                ("--hexgent-planner-url", self.endpoint("sidecar"))):
            replace_flag(flag, value)
        gateway[gateway.index("--prefill") + 2] = str(ports["bootstrap"])
        gateway[0] = self.runtime["gateway_binary"]
        capture_status = directory / "capture-status.latest.json"
        sidecar = [sys.executable, str(Path(__file__)), "sidecar-capture", "--capture-path", str(directory / "identity-capture.jsonl"),
            "--capture-status-path", str(capture_status), "--telemetry-path", str(directory / "planner-telemetry.jsonl"),
            "--approval-file", self.runtime["approval_container_path"], "--plan-sha256", plan_sha256(self.plan),
            "--run-id", self.plan["run_id"], "--deadline-unix-s", str(self.budget.work_deadline_unix_s),
            "--port", str(ports["sidecar"]), "--capture-cap-bytes", str(4 * 1024 * 1024),
            "--telemetry-cap-bytes", str(1024 * 1024)]
        self.ours_entries = [self.children.start("ours-sidecar", sidecar, env, directory / "sidecar.log", cap_bytes=4 * 1024 * 1024)]
        until = min(self.budget.work_deadline_unix_s, time.time() + 30)
        while not capture_status.exists():
            if time.time() >= until or not self.children.healthy():
                raise WindowStopped("owned passive sidecar startup failed")
            await asyncio.sleep(0.05)
        self.capture_status = capture_status
        # Original readiness helper sends only the existing models check. It is
        # never treated as KV health; the uniform approved warmup proves that.
        self.ours_entries.append(self.children.start("ours-gateway", gateway, env, directory / "gateway.log", cap_bytes=4 * 1024 * 1024))
        from types import SimpleNamespace
        actual_processes = SimpleNamespace(entries=[(entry["process"], None, {}) for entry in self.ours_entries])
        await wait_ready_async(self.endpoint("ours") + "/v1/models", actual_processes, self.budget.work_deadline_unix_s)
        proof = self.verify_ours_frontends()
        _bounded_qc_json(directory / "configured-listeners.json", proof)


    async def formal_auxiliary(self, directory, before):
        runner = _pilot("run_paired_pilot")
        record = {"cpu_before": before["cpu"], "cpu_after": runner.cpu_snapshot(),
            "worker_capture_status_start": before["status"], "worker_log_offsets_start": before["offsets"],
            "scope": "post-warmup replay+drain cgroup delta includes common observer; no invocation CPU sums"}
        record["cgroup_cpu_delta"] = runner.cgroup_cpu_delta(record["cpu_before"], record["cpu_after"])
        delay = max(0, min(2.0, self.budget.work_deadline_unix_s - time.time()))
        if delay:
            await asyncio.sleep(delay)
        record["bounded_log_tail_s"] = delay
        after = json.loads((self.root / "workers" / "status.json").read_text())
        record["worker_capture_status_end"] = after
        complete = runner.worker_observation_complete(before["status"]) and runner.worker_observation_complete(after)
        ends, slices = {}, []
        try:
            for name in ("prefill", "decode"):
                source = self.root / "workers" / (name + ".log")
                end = source.stat().st_size
                size = end - before["offsets"][name]
                if size < 0 or self.w2_saved_bytes + size > 4 * 1024 * 1024:
                    raise WindowStopped("formal W2 slice allocation exhausted/truncated")
                target = directory / (name + ".worker-slice.log")
                runner.copy_slice(source, target, before["offsets"][name], end)
                self.w2_saved_bytes += size; slices.append(target); ends[name] = end
            parser_path = Path(self.runtime["source_dir"]) / "scripts/hexgent_real_trace_collectors/summarize_worker_mutations.py"
            parser = runner.load_module("_c1_worker_mutation_parser", parser_path)
            observed = parser.summarize_events(parser.read_events(slices))
            observed["input_logs"] = [str(path) for path in slices]
            record["mutation_summary_source"] = {"path": str(parser_path), "sha256": sha256(parser_path),
                "functions": ["read_events", "summarize_events"], "execution": "unchanged pure parser API; no CLI/stdout duplicate"}
            report = runner.label_mutation_report(observed, complete)
            # The one final report is bounded BEFORE any persistent write.
            _bounded_qc_json(directory / "worker-mutation-summary.json", report)
        except BaseException as error:
            complete = False
            record["artifact_error"] = f"{type(error).__name__}: {error}"
            report = {"observation_complete": False, "coverage_status": "unknown_or_incomplete_worker_log_capture",
                "received_http_attempts": None, "attempt_any_stale_revision_rate": None, "error": record["artifact_error"]}
        record["worker_log_offsets_end"] = ends
        record["worker_log_observation_complete"] = complete
        if not complete:
            _bounded_qc_json(directory / "worker-mutation-summary.json", report)
        _bounded_qc_json(directory / "formal-observation-provenance.json", record)
        return complete

    async def method_phase(self, name, method, workflows, *, calibration):
        self.epoch.require_bound()
        directory = self.root / name
        directory.mkdir()
        caught = None
        started, finished = None, None
        formal_before = None
        raw = directory / "replay.raw.jsonl"
        try:
            if method == "ours":
                await self.start_ours(directory)
            await _idle(directory / "idle-before-warmup.json", self.expected, self.budget, worker_urls={mode: self.endpoint(mode) for mode in ("prefill", "decode")})
            warmup = self.api.WorkflowTrace(workflow_id="warmup-" + name, workflow_type="uniform_warmup", arrival_offset=0.0,
                calls=[self.api.WorkflowCall(call_id="warmup-call-" + name, parents=[], input_len=39, output_len=8,
                    request={"messages": [{"role": "user", "content": "Check your python version"}]},
                    planner_hints=dict(workflows[0].calls[0].planner_hints))])
            warm_raw = directory / "warmup.raw.jsonl"
            await replay_workflows(self.api, [warmup], raw_path=warm_raw, router_url=self.endpoint(method), model=self.model,
                run_namespace=self.plan["run_id"] + "::" + name + "::warmup", budget=self.budget,
                now_unix_s=time.time(), replay_to_raw=self.raw_budget.replay)
            warm_rows = read_jsonl(warm_raw)
            gate = _pilot("run_paired_pilot").workflow_gate(_pilot("h20_run_support"), warmup, warm_rows, self.model)
            if any(row.get("prefill_worker") != self.endpoint("prefill") or row.get("decode_worker") != self.endpoint("decode") for row in warm_rows):
                gate["successful"] = False
                gate["failure_reasons"].append("warmup response P/D identity differs from owned runtime")
            _bounded_qc_json(directory / "warmup-gate.json", gate)
            if not gate["successful"]:
                raise WindowStopped("uniform real-KV warmup failed; no retry")
            await _idle(directory / "idle-after-warmup.json", self.expected, self.budget, worker_urls={mode: self.endpoint(mode) for mode in ("prefill", "decode")})
            if not calibration:
                formal_before = {"cpu": _pilot("run_paired_pilot").cpu_snapshot(),
                    "status": json.loads((self.root / "workers" / "status.json").read_text()),
                    "offsets": {mode: (self.root / "workers" / (mode + ".log")).stat().st_size for mode in ("prefill", "decode")}}
            started = time.time()
            telemetry_path = directory / "planner-telemetry.jsonl"
            telemetry_start = telemetry_path.stat().st_size if telemetry_path.exists() else 0
            await replay_workflows(self.api, workflows, raw_path=raw, router_url=self.endpoint(method), model=self.model,
                run_namespace=self.plan["run_id"] + "::" + name, budget=self.budget,
                now_unix_s=time.time(), replay_to_raw=self.raw_budget.replay)
            await _idle(directory / "idle-after-replay.json", self.expected, self.budget, worker_urls={mode: self.endpoint(mode) for mode in ("prefill", "decode")})
        except BaseException as error:
            caught = error
        finally:
            if formal_before is not None:
                try:
                    if not await self.formal_auxiliary(directory, formal_before):
                        caught = caught or WindowStopped("formal worker capture/parser incomplete")
                except BaseException as error:
                    caught = caught or error
            finished = time.time()
            rows = read_jsonl(raw) if raw.exists() else []
            telemetry_path = directory / "planner-telemetry.jsonl"
            if started is not None and telemetry_path.exists():
                _pilot("run_paired_pilot").copy_slice(telemetry_path, directory / "planner-telemetry.formal.jsonl",
                    telemetry_start, telemetry_path.stat().st_size)
            captures_path = directory / "identity-capture.jsonl"
            captures = read_jsonl(captures_path) if captures_path.exists() else []
            capture_complete = method == "native"
            if method == "ours" and self.capture_status is not None and self.capture_status.exists():
                capture_complete = json.loads(self.capture_status.read_text()).get("observation_complete") is True
            namespace = self.plan["run_id"] + "::" + name
            waiting = []
            for record in captures:
                request = (record.get("request") or {}).get("request") or {}
                metadata = request.get("hexgent") or {}
                if not str(metadata.get("workflow_id", "")).startswith(namespace + "::"):
                    continue
                for worker_url, payload in (request.get("state") or {}).get("worker_snapshots", {}).items():
                    waiting.append({"worker_url": worker_url, "run_namespace": namespace,
                        "captured_unix_s": record["captured_unix_ns"] / 1e9, "payload": payload})
            report = calibration_gate(method, workflows, rows, model=self.model, run_namespace=namespace,
                capture_records=captures, capture_complete=capture_complete, worker_snapshots=waiting,
                owned_worker_urls={self.endpoint("prefill"), self.endpoint("decode")},
                phase_bounds_unix_s=(started, finished) if started is not None else None)
            if calibration:
                gate = report
            else:
                runner, support = _pilot("run_paired_pilot"), _pilot("h20_run_support")
                workflow_gates = {workflow.workflow_id: runner.workflow_gate(support, workflow,
                    [row for row in rows if row["workflow_id"] == workflow.workflow_id], self.model) for workflow in workflows}
                gate = runner.run_gate(workflow_gates, expected_calls=sum(len(item.calls) for item in workflows), rows=rows)
                gate["successful"] = gate["valid"] and capture_complete
                gate["activation_report"] = report
                gate["activation_status"] = "observed" if report["canonical_joined_calls"] else "not_observed"
            if caught is not None:
                gate["successful"] = False
                gate["interruption"] = f"{type(caught).__name__}: {caught}"
            if not calibration:
                gate["formal_status"] = "NOT_STARTED" if started is None else ("COMPLETED" if gate["successful"] else "FAILED")
            gate.update(phase=name, method=method, service_fit_input=False, run_namespace=namespace,
                actual_phase_started_unix_s=started, actual_phase_finished_unix_s=finished,
                raw_sha256=sha256(raw) if raw.exists() else None)
            _bounded_qc_json(directory / "measurement-gate.json", gate)
            self.phase_records.append(gate)
            if self.ours_entries:
                try:
                    await self.children.stop(self.budget.hard_deadline_unix_s, self.ours_entries)
                    if (self.capture_status is None or not self.capture_status.exists()
                            or json.loads(self.capture_status.read_text()).get("observation_complete") is not True):
                        raise WindowStopped("final passive capture publication incomplete")
                except BaseException as cleanup_error:
                    gate["successful"] = False
                    if not calibration:
                        gate["formal_status"] = "NOT_STARTED" if started is None else "FAILED"
                    gate["interruption"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                    _bounded_qc_json(directory / "measurement-gate.json", gate)
                    raise
                finally:
                    self.ours_entries = []; self.capture_status = None
        if caught is not None:
            raise caught
        return gate

    async def execute(self):
        final = {"run_id": self.plan["run_id"], "expected_formal_ordinals": [7, 8], "phase_records": [],
            "prior_sg03": "unchanged", "simulations_launched": False, "service_fit_input": False}
        try:
            await self.startup()
            for method in ("native", "ours"):
                name = "calibration-" + method
                async def run_calibration(*, deadline_unix_s, method=method, name=name):
                    return await self.method_phase(name, method, self.calibration, calibration=True)
                await self.run_stage(name, run_calibration)
            contract = self.epoch.require_bound()
            async def references(*, deadline_unix_s):
                return await collect_hot_references(self.api, self.hot, output_dir=self.root / "reference-collection",
                    router_url=self.endpoint("native"), model=self.model, implementation_revision=contract["implementation_revision"],
                    runtime_evidence_sha256=contract["runtime_evidence_sha256"], run_namespace=self.plan["run_id"] + "::references",
                    expected_limits=self.expected, budget=self.budget, replay_to_raw=self.raw_budget.replay,
                    worker_urls={mode: self.endpoint(mode) for mode in ("prefill", "decode")})
            await self.run_stage("references-hot4", references)
            reference_path = self.root / "reference-collection" / "workflow-references.jsonl"
            table = read_jsonl(reference_path)
            reference_gate = validate_hot_references(self.hot, table, contract)
            if not reference_gate["successful"]:
                raise WindowStopped("four current-epoch whole-workflow references incomplete")
            estimate = pair_budget_s([row["latency_ms"] for row in table])
            allowed = self.budget.pair_fits(time.time(), [row["latency_ms"] for row in table])
            _bounded_qc_json(self.root / "pair-budget.json", {"allowed": allowed, "required_s": estimate,
                "work_deadline_unix_s": self.budget.work_deadline_unix_s, "cleanup_reserve_s": self.budget.cleanup_s,
                "checked_unix_s": time.time(), "last_arrival_s": 0})
            if not allowed:
                raise WindowStopped("NOT_RUN_budget; original pair formula and cleanup reserve retained")
            trace = self.root / "hotpotqa-burst.trace.jsonl"
            data = "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in self.hot)
            if len(data.encode()) > 1024 * 1024:
                raise WindowStopped("frozen scenario trace exceeds QC allocation")
            trace.write_text(data)
            for ordinal, method in ((7, "native"), (8, "ours")):
                name = f"formal-{ordinal:02d}-hotpotqa-burst-{method}"
                async def formal(*, deadline_unix_s, name=name, method=method):
                    result = await self.method_phase(name, method, self.hot, calibration=False)
                    if result["successful"]:
                        argv = [sys.executable, str(Path(self.runtime["source_dir"]) / "scripts/hexgent_real_trace_collectors/summarize_scheduling_replay.py"),
                            "--replay", str(self.root / name / "replay.raw.jsonl"), "--trace", str(trace),
                            "--workflow-reference", str(reference_path), "--workflow-reference-contract",
                            str(self.root / "reference-collection" / "workflow-reference-contract.json"), "--slo-scales", "1", "2", "4",
                            "--kv-bytes-per-input-token", "327680", "--output", str(self.root / name / "summary.json")]
                        formal_telemetry = self.root / name / "planner-telemetry.formal.jsonl"
                        if formal_telemetry.exists():
                            argv += ["--planner-telemetry", str(formal_telemetry)]
                        command = await DeferredCommandExecutor().run(argv, deadline_unix_s=deadline_unix_s)
                        _bounded_qc_json(self.root / name / "summary-command.json", {"argv": argv, **command})
                        if command["returncode"] != 0:
                            raise WindowStopped("unchanged summary CLI failed")
                    return result
                await self.run_stage(name, formal)
            final["successful"] = True
        except BaseException as error:
            final.update(successful=False, stopped_reason=f"{type(error).__name__}: {error}")
        finally:
            final["phase_records"] = self.phase_records
            final["attempted_phases"] = [row["phase"] for row in self.phase_records]
            final["not_started_formal_ordinals"] = [ordinal for ordinal, method in ((7, "native"), (8, "ours"))
                if not any(row["phase"] == f"formal-{ordinal:02d}-hotpotqa-burst-{method}"
                    and row.get("actual_phase_started_unix_s") is not None for row in self.phase_records)]
            try:
                _bounded_qc_json(self.root / "window-results.json", final)
            finally:
                nonobserver = [entry for entry in self.children.entries if entry["name"] != "transport-observer"]
                try:
                    await self.children.stop(self.budget.hard_deadline_unix_s, nonobserver)
                finally:
                    observers = [entry for entry in self.children.entries if entry["name"] == "transport-observer"]
                    await self.children.stop(self.budget.hard_deadline_unix_s, observers)
        return 0 if final.get("successful") else 2


def _host_artifact_path(runtime, container_path):
    relative = Path(container_path).relative_to("/artifacts")
    return Path(runtime["artifacts_host_path"]) / relative


class C1DefaultOperations:
    """Host-owned Docker start/wait/final cleanup, never called in plan/dry-run."""
    def __init__(self, config):
        self.config = config

    def execute_c1(self, plan, approval):
        config, runtime = self.config, self.config["runtime"]
        # Required frozen actual inputs are checked before any resource is started.
        for name in ("expected_worker_limits", "transfer_parameters_path"):
            if name not in runtime:
                raise ValueError("concrete C1 runtime configuration requires " + name)
        if "model_config_sha256" not in plan["frozen_metadata"]:
            raise ValueError("frozen actual model config hash required")
        output = Path(plan["output_dir"]).resolve()
        mapped = _host_artifact_path(runtime, runtime["run_dir"]).resolve()
        if output != mapped:
            raise ValueError("fresh host output directory must map exactly to owned container RUN directory")
        owner_path = _host_artifact_path(runtime, runtime["owner_container_path"]).resolve()
        plan_path = _host_artifact_path(runtime, runtime.get("plan_container_path", runtime["run_dir"] + "/approved-plan.json")).resolve()
        for path in (owner_path, plan_path):
            if not path.is_relative_to(output) or path == output:
                raise ValueError("dynamic owner/plan records must stay inside the fresh owned output directory")
        host_control_bytes = 65
        def write_control(path, value):
            nonlocal host_control_bytes
            payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            if host_control_bytes + len(payload) > 6 * 1024 * 1024:
                raise WindowStopped("owned host control aggregate6MiB budget reached before write")
            with Path(path).open("xb") as target:
                target.write(payload)
            host_control_bytes += len(payload)
        output.mkdir(parents=True, exist_ok=False)
        write_control(plan_path, plan)
        recipe = render_c1_runtime_recipe(config, now_unix_s=time.time())
        write_control(output / "actual-launch-recipe.json", recipe)
        launcher = C1OwnedLauncher(plan, recipe)
        async def execute():
            record = None
            try:
                record = await launcher.start(execute=True, approval=approval)
                temporary_owner = owner_path.with_suffix(".tmp")
                write_control(temporary_owner, {"run_id": plan["run_id"], "owner_id": record["owner_id"],
                    "window_anchor_unix_s": plan["window_anchor_unix_s"], "hard_deadline_unix_s": plan["hard_deadline_unix_s"],
                    "image_digest": plan["frozen_metadata"]["image_digest"], "plan_sha256": plan_sha256(plan)})
                temporary_owner.replace(owner_path)
                result = await launcher.executor.run(["docker", "wait", record["owner_id"]],
                    deadline_unix_s=plan["work_deadline_unix_s"])
                write_control(output / "container-exit.json", result)
                return 0 if result["returncode"] == 0 and result["stdout"].strip() == "0" else 2
            finally:
                if launcher.window.owner_id is not None:
                    cleanup = await launcher.cleanup(launcher.window.owner_id, execute=True, approval=approval)
                    write_control(output / "owned-cleanup.json", cleanup)
        return asyncio.run(execute())


def _approved(approval, plan):
    if approval.get("approved_plan_sha256") != plan_sha256(plan) or approval.get("window_run_id") != plan["run_id"]:
        raise PermissionError("C1 runtime differs from explicit frozen plan approval")


def runtime_supervisor_main(argv):
    parser = argparse.ArgumentParser()
    for name in ("config", "plan", "approval-file", "owner-record"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    config, plan = json.loads(args.config.read_text()), json.loads(args.plan.read_text())
    approval = json.loads(args.approval_file.read_text())
    _approved(approval, plan)
    if config["runtime"] != plan["runtime_configuration"]:
        raise PermissionError("runtime configuration changed after approval")
    for name, expected in plan["frozen_metadata"].items():
        if config.get(name) != expected:
            raise PermissionError("static configuration changed after approval: " + name)
    async def execute():
        while not args.owner_record.exists():
            if time.time() >= plan["work_deadline_unix_s"]:
                raise TimeoutError("original window expired before exact owner registration")
            await asyncio.sleep(0.05)
        owner = json.loads(args.owner_record.read_text())
        if (owner.get("run_id") != plan["run_id"] or owner.get("plan_sha256") != plan_sha256(plan)
                or owner.get("window_anchor_unix_s") != plan["window_anchor_unix_s"]
                or owner.get("hard_deadline_unix_s") != plan["hard_deadline_unix_s"]):
            raise PermissionError("dynamic exact owner/anchor registration differs")
        for path in ("/runtime/home", "/runtime/tmp"):
            Path(path).mkdir(parents=True, exist_ok=True)
        return await C1RuntimeOperations(config, plan, approval, owner["owner_id"]).execute()
    return asyncio.run(execute())


def observer_child_main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval-file", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cpu", type=int, required=True)
    args, observer_args = parser.parse_known_args(argv)
    approval = json.loads(args.approval_file.read_text())
    if approval.get("approved_plan_sha256") != args.plan_sha256 or approval.get("window_run_id") != args.run_id:
        raise PermissionError("observer child needs exact C1 approval")
    import os
    import resource
    os.sched_setaffinity(0, {args.cpu})
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    return _observer_api().main(["--observe", *observer_args])


def passive_sidecar_main(argv):
    """Deferred real build_app producer; caller must separately approve/freeze runtime."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-path", type=Path, required=True)
    parser.add_argument("--capture-status-path", type=Path, required=True)
    parser.add_argument("--telemetry-path", type=Path, required=True)
    parser.add_argument("--approval-file", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deadline-unix-s", type=float, required=True)
    parser.add_argument("--port", type=int, default=37140)
    parser.add_argument("--capture-cap-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--telemetry-cap-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args(argv)
    approval = json.loads(args.approval_file.read_text())
    if (approval.get("approved_plan_sha256") != args.plan_sha256
            or approval.get("window_run_id") != args.run_id):
        raise PermissionError("passive sidecar needs explicit matching C1 plan approval")
    remaining = args.deadline_unix_s - time.time()
    if remaining <= 0:
        raise TimeoutError("immutable C1 work deadline already reached")
    from sglang.hexgent.sidecar import PlannerTelemetry, build_app
    import uvicorn
    with args.capture_path.open("x", encoding="utf-8") as target:
        def append(record):
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
        publisher = CaptureStatusPublisher(args.capture_status_path, fatal_exit=owned_publisher_fatal)
        status = publisher.publish
        if not 0 < args.capture_cap_bytes <= 4 * 1024 * 1024 or not 0 < args.telemetry_cap_bytes <= 1024 * 1024:
            raise ValueError("C1 capture/telemetry phase subcaps exceeded")
        telemetry = PlannerTelemetry(args.telemetry_path)
        captured = None
        def incomplete(reason):
            captured._incomplete(reason)
            status(captured.status())
        telemetry.path = _TelemetryPath(args.telemetry_path, args.telemetry_cap_bytes, incomplete)
        captured = ASGIEvidenceCapture(build_app(telemetry=telemetry), append,
            max_saved_bytes=args.capture_cap_bytes, status_writer=status)
        status(captured.status())
        server = uvicorn.Server(uvicorn.Config(captured, host="127.0.0.1", port=args.port, log_level="warning"))
        async def serve():
            try:
                await asyncio.wait_for(server.serve(), timeout=max(0, args.deadline_unix_s - time.time()))
            finally:
                server.should_exit = True
        asyncio.run(serve())


def main(argv=None, *, operations=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-recipe", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-file", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    plan = prepare_plan(config)
    if not args.execute or args.dry_run:
        output = {"plan": plan, "plan_sha256": plan_sha256(plan), "recipe": render_c1_runtime_recipe(config,
            now_unix_s=plan["window_anchor_unix_s"])} if args.print_recipe else plan
        print(json.dumps(output, indent=2))
        return 0
    # A reviewed concrete binding is required. No old60min launcher fallback,
    # inherited live service, default peer, or implicit container operation.
    if operations is None:
        if "runtime" not in config:
            raise RuntimeError(plan["runtime_binding_status"])
        operations = C1DefaultOperations(config)
    if args.approval_file is None:
        raise PermissionError("live C1 execution needs a separate explicit approval file")
    approval = json.loads(args.approval_file.read_text())
    plan_bytes = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    frozen_sha = hashlib.sha256(plan_bytes).hexdigest()
    if approval.get("approved_plan_sha256") != frozen_sha or approval.get("window_run_id") != plan["run_id"]:
        raise PermissionError("live C1 execution plan hash/run differs from explicit approval")
    return operations.execute_c1(plan, approval)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sidecar-capture":
        passive_sidecar_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "runtime-supervisor":
        raise SystemExit(runtime_supervisor_main(sys.argv[2:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "observer-child":
        raise SystemExit(observer_child_main(sys.argv[2:]))
    else:
        raise SystemExit(main())
