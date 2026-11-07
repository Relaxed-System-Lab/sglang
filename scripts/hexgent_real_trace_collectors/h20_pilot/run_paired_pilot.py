#!/usr/bin/env python3
"""One prospective, counterbalanced eight-run H20 pilot block.

Artifact-only driver. No action occurs on import. --execute is required for live
HTTP/process work; otherwise only the fresh plan/artifacts are written.
"""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
from urllib.error import HTTPError
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

SEED = 20261107
DATASETS = (("sharegpt", 16), ("hotpotqa", 133))
PREFILL = "http://127.0.0.1:37100"
DECODE = "http://127.0.0.1:37110"
ENDPOINTS = {"native": "http://127.0.0.1:37130", "ours": "http://127.0.0.1:37150"}
LIMIT = 16 * 1024 * 1024


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_order():
    return [
        ("sharegpt", "low", "native"), ("sharegpt", "low", "ours"),
        ("sharegpt", "burst", "ours"), ("sharegpt", "burst", "native"),
        ("hotpotqa", "low", "ours"), ("hotpotqa", "low", "native"),
        ("hotpotqa", "burst", "native"), ("hotpotqa", "burst", "ours"),
    ]


def arrival_offsets(count, *, rate, seed, burst):
    if burst:
        return [0.0] * count
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(count - 1):
        offsets.append(offsets[-1] + rng.expovariate(rate))
    return offsets


def low_rate(record):
    makespan = record["makespan_ms"]
    if record["workflow_count"] != 2 or makespan is None or not math.isfinite(makespan) or makespan <= 0:
        raise ValueError("new two-calibration-workflow burst must have positive complete makespan")
    return 0.5 * (2 / (makespan / 1000))


def freeze_arrivals(workflows, offsets):
    ordered = sorted(copy.deepcopy(workflows), key=lambda w: w["workflow_id"])
    if len(ordered) != len(offsets):
        raise ValueError("arrival vector length differs from workflow count")
    for workflow, offset in zip(ordered, offsets):
        workflow["arrival_offset"] = offset
    return ordered


def snapshot_is_empty(payload, expected_limits):
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("missing/nonempty actual snapshots required")
    empty = True
    for snapshot in snapshots:
        if any(k not in snapshot["limits"] or snapshot["limits"][k] != v for k, v in expected_limits.items()):
            raise ValueError("actual worker limits differ from frozen observations")
        if not isinstance(snapshot["running"], list) or not isinstance(snapshot["queues"], dict) or not snapshot["queues"]:
            raise ValueError("actual running/queues schema differs")
        if any(not isinstance(queue, list) for queue in snapshot["queues"].values()):
            raise ValueError("actual queue values must be lists")
        empty = empty and not snapshot["running"] and all(not q for q in snapshot["queues"].values())
    return empty


def run_gate(gates, *, expected_calls, rows):
    ids = [(row["workflow_id"], row["call_id"]) for row in rows]
    return {
        "valid": len(rows) == expected_calls and len(ids) == len(set(ids))
                 and all(gate["successful"] for gate in gates.values()),
        "expected_calls": expected_calls, "recorded_calls": len(rows),
        "unique_recorded_calls": len(set(ids)), "workflow_gates": gates,
        "note": "Failures/missing calls retain their raw evidence and cannot be presented as valid pressure measurements.",
    }


def copy_slice(source, target, start, end):
    if start < 0 or end < start or Path(source).stat().st_size < end:
        raise ValueError("worker/telemetry file truncated or invalid byte interval")
    with Path(source).open("rb") as src, Path(target).open("xb") as out:
        src.seek(start)
        left = end - start
        while left:
            chunk = src.read(min(left, 65536))
            if not chunk:
                raise ValueError("source shortened while slicing")
            out.write(chunk)
            left -= len(chunk)


def ensure_time(deadline):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("approved hard deadline reached")
    return remaining


def get_json(url, deadline):
    with urlopen(url, timeout=min(5.0, ensure_time(deadline))) as response:
        if response.status != 200:
            raise ValueError("snapshot/readiness HTTP status differs from 200")
        return json.load(response)


def wait_idle(path, expected, deadline):
    started = time.monotonic()
    idle_deadline = min(deadline, time.time() + 30)
    record = {"poll_count": 0, "first": None, "final": None, "success": False}
    consecutive = 0
    try:
        while time.time() < idle_deadline:
            actual = {}
            record["poll_count"] += 1
            record["final"] = actual
            if record["first"] is None:
                record["first"] = actual
            for mode, url in (("prefill", PREFILL), ("decode", DECODE)):
                try:
                    actual[mode] = get_json(url + "/hexgent/pd_snapshot", idle_deadline)
                except HTTPError as exc:
                    actual[mode] = {"http_status": exc.code,
                                    "response_body": exc.read().decode("utf-8", errors="replace")}
                    raise
            empty = all(snapshot_is_empty(actual[mode], expected[mode]) for mode in actual)
            consecutive = consecutive + 1 if empty else 0
            if consecutive == 2:
                record["success"] = True
                return record
            time.sleep(min(0.25, ensure_time(idle_deadline)))
        raise TimeoutError("workers not idle within30 seconds")
    except Exception as exc:
        record["error"] = repr(exc)
        raise
    finally:
        record["elapsed_ms"] = (time.monotonic() - started) * 1000
        save_json(path, record)


def cpu_snapshot():
    value = {"timestamp_unix_s": time.time()}
    for name in ("cpu.stat", "memory.current"):
        try:
            value[name] = Path("/sys/fs/cgroup", name).read_text()
        except OSError as exc:
            value[name + "_error"] = repr(exc)
    return value


def block_completion(records):
    attempted = len(records) == 8
    return {"all_runs_attempted": attempted,
            "complete_balanced_block": attempted and all(r["valid_measurement"] for r in records)}


def cgroup_cpu_delta(before, after):
    def parse(value):
        return {key: int(number) for key, number in
                (line.split() for line in value["cpu.stat"].splitlines())}
    first, last = parse(before), parse(after)
    delta = {key: last[key] - first[key] for key in first}
    if any(value < 0 for value in delta.values()):
        raise ValueError("actual cgroup CPU counter decreased")
    return delta


def worker_observation_complete(status):
    workers = {p["name"]: p for p in status["processes"] if p["name"] in {"prefill", "decode"}}
    return set(workers) == {"prefill", "decode"} and all(
        p.get("truncated") is False and "write_error" in p and p["write_error"] is None for p in workers.values())


def label_mutation_report(report, complete):
    result = copy.deepcopy(report)
    result["observation_complete"] = complete
    result["backend_count_note"] = "Observed backend outputs; actual DP aggregation count need not equal TP ranks."
    if not complete:
        result["coverage_status"] = "unknown_or_incomplete_worker_log_capture"
        for key in list(result):
            if key.endswith("_rate"):
                result[key] = None
        if "rank_outcome_rates" in result:
            result["rank_outcome_rates"] = dict.fromkeys(result["rank_outcome_rates"], None)
    else:
        result["coverage_status"] = "bounded_endpoint_log_capture_complete; upstream transport remains unobserved"
    return result


def require_free_ports():
    for port in (37140, 37150, 37151):
        try:
            connection = socket.create_connection(("127.0.0.1", port), timeout=0.25)
        except ConnectionRefusedError:
            continue
        else:
            connection.close()
            raise RuntimeError("existing listener on owned Ours port %s; no service stopped" % port)


def workflow_gate(support, workflow, rows, model):
    gate = support.validate_workflow_rows(workflow, rows)
    if any(not row.get("error") and (row.get("response") or {}).get("model") != model for row in rows):
        gate["failure_reasons"].append("returned model differs from frozen served model")
        gate["successful"] = False
        gate["latency_ms"] = None
    return gate


class OwnProcesses:
    def __init__(self):
        self.entries = []

    def start(self, name, argv, env, directory):
        proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
        entry = {"name": name, "pid": proc.pid, "pgid": proc.pid, "argv": argv,
                 "retained_stdout_bytes": 0, "dropped_stdout_bytes": 0}
        log_path = directory / (name + ".log")
        def drain():
            with log_path.open("xb") as output:
                while True:
                    data = proc.stdout.read(65536)
                    if not data:
                        break
                    keep = min(len(data), LIMIT - entry["retained_stdout_bytes"])
                    output.write(data[:keep])
                    output.flush()
                    entry["retained_stdout_bytes"] += keep
                    entry["dropped_stdout_bytes"] += len(data) - keep
        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        self.entries.append((proc, thread, entry))
        return entry

    def stop(self, deadline):
        for proc, thread, entry in reversed(self.entries):
            if proc.poll() is None:
                try:
                    actual_pgid = os.getpgid(proc.pid)
                except ProcessLookupError:
                    actual_pgid = None
                if actual_pgid is not None and actual_pgid != entry["pgid"]:
                    entry["cleanup_error"] = "owned process group identity changed; no signal sent"
                    continue
                if actual_pgid is not None:
                    try:
                        os.killpg(entry["pgid"], signal.SIGTERM if time.time() < deadline else signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                try:
                    proc.wait(timeout=max(0.01, min(3.0, deadline - time.time())))
                except subprocess.TimeoutExpired:
                    os.killpg(entry["pgid"], signal.SIGKILL)
                    proc.wait(timeout=2)
            thread.join(timeout=2)
            entry["exit_code"] = proc.returncode
        return [entry for _, _, entry in self.entries]


def wait_ready(url, processes, deadline):
    started = time.monotonic()
    last = None
    while time.monotonic() - started < 30:
        for proc, _, _ in processes.entries:
            if proc.poll() is not None:
                raise RuntimeError("owned service exited before readiness")
        try:
            return get_json(url, deadline)
        except Exception as exc:
            last = repr(exc)
        time.sleep(min(0.25, ensure_time(deadline)))
    raise RuntimeError("owned service readiness failed: " + str(last))


def method_commands(source_root, directory, transfer):
    env = os.environ.copy()
    env.pop("HEXGENT_PROXY_MODEL_PATH", None)
    env.pop("SGLANG_DISAGGREGATION_MAX_INFLIGHT_TRANSFERS", None)
    env.update({"PYTHONPATH": str(source_root / "python"), "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
                "AGENTIC_PD_OURS_PLACEMENT_STRATEGY": "projected_finish",
                "HEXGENT_PROJECTED_RATIO_ORDER": "high_first",
                "HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH": "1", "HEXGENT_ENABLE_LOCAL_PRIORITY_UPDATES": "1",
                "HEXGENT_TRANSFER_STARTUP_MS": str(transfer["HEXGENT_TRANSFER_STARTUP_MS"]),
                "HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S": str(transfer["HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S"]),
                "HEXGENT_REVISION_STATE_PATH": str(directory / "planner-revision")})
    telemetry = directory / "planner-telemetry.full.jsonl"
    sidecar = [sys.executable, "-m", "sglang.hexgent.sidecar", "--host", "127.0.0.1",
               "--port", "37140", "--telemetry-path", str(telemetry)]
    gateway = ["/artifacts/r2-build/bin/sgl-model-gateway", "--host", "127.0.0.1", "--port", "37150",
               "--prometheus-host", "127.0.0.1", "--prometheus-port", "37151",
               "--pd-disaggregation", "--prefill", PREFILL, "37120", "--decode", DECODE,
               "--policy", "round_robin", "--json-log", "--hexgent-router",
               "--hexgent-planner-url", "http://127.0.0.1:37140", "--hexgent-planner-timeout-ms", "10000",
               "--hexgent-fallback-policy", "pd_policy", "--hexgent-admission-queue",
               "--hexgent-admission-max-wait-ms", "300000", "--hexgent-admission-poll-interval-ms", "10",
               "--hexgent-max-prefill-queue-per-worker", "64", "--hexgent-prefill-batching-headroom", "16",
               "--hexgent-prefill-token-headroom-percent", "25"]
    return env, sidecar, gateway, telemetry


def prepare(args, contract):
    root = args.run_dir
    root.mkdir(parents=True, exist_ok=False)
    summary = read_json(args.reference_dir / "calibration-burst.summary.json")
    plan = {"seed": SEED, "repeat": 0, "run_count": 8, "cells": [], "runs": [],
            "reference_contract": contract, "hard_deadline_unix_s": args.deadline_unix_s,
            "calibration_provenance": "new four-calibration-workflow mixed native burst; load heuristic, not pure capacity",
            "poisson_convention": "random.Random(20261107) independently per dataset; first arrival conditioned0; cumulative exponential gaps",
            "warmup": "one identical39-prompt/8-target request before every method; then two idle observations",
            "baseline_workers": "same persistent P/D as Ours; no process CPU invocation-time sum"}
    for dataset, count in DATASETS:
        source = args.runtime_input_dir / (dataset + ".heldout.sanitized.runtime.jsonl")
        workflows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if len(workflows) != 4 or sum(len(w["calls"]) for w in workflows) != count or len({w["workflow_id"] for w in workflows}) != 4:
            raise ValueError("frozen heldout workflow/call set changed")
        cal = summary["datasets"][dataset]
        if not cal["all_workflows_complete"] or cal["successful_workflows"] != 2:
            raise ValueError("incomplete new calibration burst cannot set offered rate")
        rate = low_rate(cal)
        for load in ("low", "burst"):
            offsets = arrival_offsets(4, rate=rate, seed=SEED, burst=load == "burst")
            frozen = freeze_arrivals(workflows, offsets)
            trace = root / (dataset + "-" + load + ".trace.jsonl")
            trace.write_text("".join(json.dumps(w, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for w in frozen))
            cell = {"dataset": dataset, "load": load, "expected_calls": count,
                    "trace": str(trace), "trace_sha256": sha(trace), "source_trace_sha256": sha(source),
                    "workflow_ids": [w["workflow_id"] for w in frozen], "arrival_offsets_s": offsets,
                    "low_poisson_rate_per_s": rate, "calibration_burst_makespan_ms": cal["makespan_ms"]}
            plan["cells"].append(cell)
    for ordinal, (dataset, load, method) in enumerate(run_order(), 1):
        cell = next(c for c in plan["cells"] if c["dataset"] == dataset and c["load"] == load)
        plan["runs"].append({"ordinal": ordinal, "dataset": dataset, "load": load, "method": method,
                             "trace": cell["trace"], "trace_sha256": cell["trace_sha256"],
                             "arrival_offsets_s": cell["arrival_offsets_s"]})
    save_json(root / "run-plan.json", plan)
    return plan


def validate_inputs(args):
    contract = read_json(args.reference_dir / "workflow-reference-contract.json")
    expected = {"model": os.environ["SERVED_MODEL"],
                "implementation_revision": os.environ["IMPLEMENTATION_REVISION"],
                "runtime_evidence_sha256": sha(os.environ["RUNTIME_EVIDENCE_JSON"])}
    if contract != expected:
        raise ValueError("new workflow reference contract differs from model/revision/runtime evidence")
    gate = read_json(args.reference_dir / "collection-gate.json")
    if not (gate["calibration_workflows_complete"] and gate["all_heldout_references_complete"]
            and gate["recorded_heldout_workflows"] == 8 and gate["successful_heldout_workflows"] == 8):
        raise ValueError("new reference collection incomplete; formal pilot forbidden")
    if gate["reference_contract"] != contract:
        raise ValueError("reference collection gate contract differs")
    table = args.reference_dir / "workflow-references.jsonl"
    if sha(table) != gate["workflow_reference_sha256"]:
        raise ValueError("reference table hash changed")
    references = [json.loads(line) for line in table.read_text().splitlines() if line.strip()]
    expected_calls = {workflow["workflow_id"]: len(workflow["calls"]) for dataset, _ in DATASETS
                      for workflow in (json.loads(line) for line in
                      (args.runtime_input_dir / (dataset + ".heldout.sanitized.runtime.jsonl")).read_text().splitlines()
                      if line.strip())}
    if len(references) != 8 or {r["workflow_id"] for r in references} != set(expected_calls):
        raise ValueError("reference IDs do not exactly match the eight heldout workflows")
    for reference in references:
        count = expected_calls[reference["workflow_id"]]
        latency = reference["latency_ms"]
        if (reference["status"] != "successful" or reference["errors"] != 0
                or any(reference[key] != count for key in
                       ("expected_calls", "completed_calls", "usage_verified_calls"))
                or any(reference[key] != value for key, value in contract.items())
                or latency is None or not math.isfinite(latency) or latency <= 0):
            raise ValueError("reference row fails success/model/usage/completeness/latency gate")
    return contract


async def execute_run(args, run, cell, support, api, expected, transfer, logs, model):
    directory = args.run_dir / ("%02d-%s-%s-%s" % (run["ordinal"], run["dataset"], run["load"], run["method"]))
    directory.mkdir()
    processes = OwnProcesses()
    record = {**run, "status": "started", "namespace": args.run_dir.name + "::" + directory.name,
              "cpu_before": cpu_snapshot(), "worker_log_offsets_start": None, "artifact_errors": []}
    workers = api.load_workflow_traces_jsonl(Path(run["trace"]))
    raw = directory / "replay.raw.jsonl"
    telemetry = None
    start_offsets = None
    telemetry_start = 0
    status_path = Path(os.environ["RUN_DIR"]) / "status.json"
    try:
        ensure_time(args.deadline_unix_s)
        if run["method"] == "ours":
            require_free_ports()
            env, sidecar, gateway, telemetry = method_commands(args.source_root, directory, transfer)
            record["owned_configuration"] = {
                "sidecar_argv": sidecar, "gateway_argv": gateway,
                "environment": {k: v for k, v in env.items() if k.startswith(("HEXGENT_", "AGENTIC_PD_")) or k in (
                    "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
            processes.start("planner", sidecar, env, directory)
            wait_ready("http://127.0.0.1:37140/health", processes, args.deadline_unix_s)
            processes.start("gateway", gateway, env, directory)
            wait_ready(ENDPOINTS["ours"] + "/v1/models", processes, args.deadline_unix_s)
        wait_idle(directory / "idle-before-warmup.json", expected, args.deadline_unix_s)
        warm = api.WorkflowTrace(
            workflow_id="warmup-" + directory.name,
            calls=[api.WorkflowCall(call_id="warmup-call", parents=[],
                request={"messages": [{"role": "user", "content": "Check your python version"}]},
                input_len=39, output_len=8, planner_hints=copy.deepcopy(workers[0].calls[0].planner_hints))],
            workflow_type="uniform_warmup", arrival_offset=0.0)
        warm_rows = await asyncio.wait_for(support.replay_to_raw(
            api, [warm], raw_path=directory / "warmup.raw.jsonl", router_url=ENDPOINTS[run["method"]],
            model=model, worker_class="H20-BF16-TP2", run_namespace=record["namespace"] + "::warmup"),
            timeout=ensure_time(args.deadline_unix_s))
        warm_gate = workflow_gate(support, warm, warm_rows, model)
        save_json(directory / "warmup-gate.json", warm_gate)
        if not warm_gate["successful"]:
            raise ValueError("uniform warmup failed model/usage/completeness gate")
        wait_idle(directory / "idle-after-warmup.json", expected, args.deadline_unix_s)
        record["cpu_formal_before"] = cpu_snapshot()
        record["worker_capture_status_start"] = read_json(status_path)
        start_offsets = {name: Path(path).stat().st_size for name, path in logs.items()}
        record["worker_log_offsets_start"] = start_offsets
        telemetry_start = telemetry.stat().st_size if telemetry and telemetry.exists() else 0
        save_json(directory / "run.json", record)
        await asyncio.wait_for(support.replay_to_raw(
            api, workers, raw_path=raw, router_url=ENDPOINTS[run["method"]], model=model,
            worker_class="H20-BF16-TP2", run_namespace=record["namespace"]),
            timeout=ensure_time(args.deadline_unix_s))
        wait_idle(directory / "idle-after-replay.json", expected, args.deadline_unix_s)
        record["cpu_formal_after"] = cpu_snapshot()
        record["cgroup_cpu_formal_delta"] = cgroup_cpu_delta(record["cpu_formal_before"], record["cpu_formal_after"])
        record["cpu_scope"] = "Actual containing cgroup delta over formal replay+drain; not a sum of overlapping invocation CPU durations."
        record["status"] = "replay_returned"
    except Exception as exc:
        record["error"] = repr(exc)
        record["status"] = "failed"
    finally:
        if "cpu_formal_before" in record and "cpu_formal_after" not in record:
            record["cpu_formal_after"] = cpu_snapshot()
        delay = max(0.0, min(2.0, args.deadline_unix_s - time.time())) if start_offsets is not None else 0
        if delay:
            await asyncio.sleep(delay)
        try:
            record["owned_processes"] = processes.stop(args.deadline_unix_s)
            record["artifact_errors"].extend(p["cleanup_error"] for p in record["owned_processes"] if "cleanup_error" in p)
        except Exception as exc:
            record["artifact_errors"].append("owned cleanup: " + repr(exc))
        if start_offsets is not None:
            try:
                record["worker_capture_status_end"] = read_json(status_path)
                observation_complete = all(worker_observation_complete(record[key]) for key in
                                           ("worker_capture_status_start", "worker_capture_status_end"))
                record["worker_log_observation_complete"] = observation_complete
                end_offsets = {name: Path(path).stat().st_size for name, path in logs.items()}
                record["worker_log_offsets_end"] = end_offsets
                slices = []
                for name, path in logs.items():
                    target = directory / (name + ".worker-slice.log")
                    copy_slice(path, target, start_offsets[name], end_offsets[name])
                    slices.append(target)
                command = [sys.executable, str(args.source_root / "scripts/hexgent_real_trace_collectors/summarize_worker_mutations.py")]
                for path in slices:
                    command += ["--log", str(path)]
                observed = directory / "worker-mutation-summary.observed.json"
                command += ["--output", str(observed)]
                record["mutation_summary_exit"] = subprocess.run(command, stdout=subprocess.DEVNULL).returncode
                if record["mutation_summary_exit"] == 0:
                    save_json(directory / "worker-mutation-summary.json",
                              label_mutation_report(read_json(observed), observation_complete))
                else:
                    record["artifact_errors"].append("mutation summary CLI failed")
            except Exception as exc:
                record["worker_log_observation_complete"] = False
                record["artifact_errors"].append("worker log capture: " + repr(exc))
                save_json(directory / "worker-mutation-summary.json", {
                    "observation_complete": False, "coverage_status": "unknown_or_incomplete_worker_log_capture",
                    "received_http_attempts": None, "attempt_any_stale_revision_rate": None,
                    "error": repr(exc)})
        record["cpu_after"] = cpu_snapshot()
        save_json(directory / "run.json", record)
    if not raw.exists():
        raw.touch()
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    gates = {wf.workflow_id: workflow_gate(
        support, wf, [row for row in rows if row["workflow_id"] == wf.workflow_id], model) for wf in workers}
    gate = run_gate(gates, expected_calls=cell["expected_calls"], rows=rows)
    gate["valid"] = gate["valid"] and record["status"] == "replay_returned" and not record["artifact_errors"]
    save_json(directory / "measurement-gate.json", gate)
    actual = {wid: value.get("workflow_arrival_ms") for wid, value in gates.items()}
    available = [v for v in actual.values() if v is not None]
    origin = min(available) if available else None
    save_json(directory / "actual-workflow-arrivals.json", {
        "workflow_arrival_ms": actual, "planned_offsets_s": cell["arrival_offsets_s"],
        "actual_offsets_ms": {wid: None if value is None or origin is None else value - origin for wid, value in actual.items()},
        "jitter_ms": {wid: None if actual[wid] is None or origin is None else
                      actual[wid] - origin - 1000 * planned
                      for wid, planned in zip(cell["workflow_ids"], cell["arrival_offsets_s"])}})
    command = [sys.executable, str(args.source_root / "scripts/hexgent_real_trace_collectors/summarize_scheduling_replay.py"),
               "--replay", str(raw), "--trace", run["trace"],
               "--workflow-reference", str(args.reference_dir / "workflow-references.jsonl"),
               "--workflow-reference-contract", str(args.reference_dir / "workflow-reference-contract.json"),
               "--slo-scales", "1", "2", "4", "--kv-bytes-per-input-token", "327680",
               "--output", str(directory / "summary.json")]
    if telemetry and telemetry.exists():
        formal = directory / "planner-telemetry.formal.jsonl"
        copy_slice(telemetry, formal, telemetry_start, telemetry.stat().st_size)
        command += ["--planner-telemetry", str(formal)]
    record["summary_argv"] = command
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (directory / "summary-command.log").write_text(result.stdout[:LIMIT])
    record["summary_exit"] = result.returncode
    record["valid_measurement"] = gate["valid"] and record["summary_exit"] == 0
    save_json(directory / "run.json", record)
    return record


async def execute(args, plan, support, api, expected, transfer, logs):
    records = []
    try:
        for run in plan["runs"]:
            ensure_time(args.deadline_unix_s)
            cell = next(c for c in plan["cells"] if c["dataset"] == run["dataset"] and c["load"] == run["load"])
            records.append(await execute_run(args, run, cell, support, api, expected, transfer, logs,
                                             plan["reference_contract"]["model"]))
            save_json(args.run_dir / "block-results.json", {"runs": records, **block_completion(records)})
    except Exception as exc:
        save_json(args.run_dir / "block-results.json", {
            "runs": records, **block_completion(records), "error": repr(exc),
            "not_executed_ordinals": [run["ordinal"] for run in plan["runs"][len(records):]]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-input-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, choices=[0], default=0)
    parser.add_argument("--deadline-unix-s", type=float, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    contract = validate_inputs(args)
    plan = prepare(args, contract)
    if not args.execute:
        return
    expected = read_json(os.environ["H20_EXPECTED_LIMITS_JSON"])
    transfer = read_json(os.environ["TRANSFER_PARAMETERS_JSON"])
    logs = json.loads(os.environ["H20_WORKER_LOG_PATHS_JSON"])
    if set(logs) != {"prefill", "decode"}:
        raise ValueError("exact explicit prefill/decode worker log paths required")
    support = load_module("h20_run_support", Path(__file__).with_name("h20_run_support.py"))
    api = support.load_workflow_api(args.source_root)
    asyncio.run(execute(args, plan, support, api, expected, transfer, logs))


if __name__ == "__main__":
    main()
