#!/usr/bin/env python3
"""Standard-library aggregation of one sealed H20 eight-case pilot.

Reads saved artifacts only. Writes ledger.json, cases.csv, workflows.csv and
report.md into a fresh output location; existing analysis outputs are refused.
No source imports, subprocesses, networking, GPU calls or driver mutation.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, median

ALPHAS = (1.0, 2.0, 4.0)


def distribution(values):
    values = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("nonfinite metric observation")
    ordered = sorted(values)
    n = len(ordered)
    def rank(q):
        return math.ceil(q * n) if n else None
    return {"count": n, "mean": mean(ordered) if n else None,
            "median": median(ordered) if n else None,
            "p95": ordered[rank(.95) - 1] if n else None,
            "p99": ordered[rank(.99) - 1] if n else None,
            "max": ordered[-1] if n else None,
            "p95_rank": rank(.95), "p99_rank": rank(.99)}


def case_state(directory_exists, rows, warmup_gate, record):
    if not directory_exists:
        return "NOT_RUN_budget"
    if not rows and warmup_gate.get("successful") is False and not record.get("cpu_formal_before"):
        return "NOT_STARTED_warmup_failure"
    return "FORMAL_STARTED"


def workflow_record(workflow, rows, reference, state, model):
    expected = {call["call_id"]: call for call in workflow["calls"]}
    observed = [row["call_id"] for row in rows]
    if len(observed) != len(set(observed)) or set(observed) - set(expected):
        raise ValueError("duplicate/unexpected raw call identity")
    missing = sorted(set(expected) - set(observed))
    errors = sum(bool(row.get("error")) for row in rows)
    usage_verified = 0
    for row in rows:
        if row.get("error"):
            continue
        response = row.get("response") or {}
        usage = response.get("usage") or {}
        call = expected[row["call_id"]]
        usage_verified += (usage.get("prompt_tokens") == call["input_len"]
                           and usage.get("completion_tokens") == call["output_len"]
                           and response.get("model") == model)
    arrivals = [row.get("workflow_arrival_ms") for row in rows]
    ends = [row.get("call_completed_at_ms", row.get("response_completed_at_ms")) for row in rows]
    response_end = max((float(row["response_completed_at_ms"]) for row in rows
                        if row.get("response_completed_at_ms") is not None), default=None)
    valid_clock = bool(rows) and all(v is not None and math.isfinite(float(v)) for v in arrivals + ends) and len(set(arrivals)) == 1
    origin = float(arrivals[0]) if valid_clock else None
    end = max(map(float, ends)) if valid_clock else None
    elapsed = end - origin if valid_clock else None
    if state.startswith("NOT_"):
        status = state
    elif missing:
        status = "incomplete"
    elif errors:
        status = "failed"
    elif usage_verified != len(expected) or elapsed is None or elapsed <= 0:
        status = "invalid_observation"
    else:
        status = "successful"
    base = reference["latency_ms"]
    ratio = elapsed / base if status == "successful" else None
    result = {"workflow_id": workflow["workflow_id"], "status": status,
              "expected_calls": len(expected), "observed_calls": len(rows),
              "missing_call_ids": missing, "errors": errors, "usage_verified_calls": usage_verified,
              "workflow_arrival_ms": origin, "observed_response_completed_at_ms": response_end,
              "observed_call_completed_at_ms": end,
              "observed_latency_ms": elapsed,
              "successful_latency_ms": elapsed if status == "successful" else None,
              "base_slo_ms": base, "slo_ratio": ratio,
              "reference_raw_sha256": reference.get("raw_replay_sha256")}
    for alpha in ALPHAS:
        result["threshold_alpha_" + str(int(alpha)) + "_ms"] = base * alpha
        result["violation_alpha_" + str(int(alpha))] = (
            True if status == "failed" else ratio > alpha if ratio is not None else None)
    return result


def alpha_metrics(workflows, *, formal_started):
    failed = sum(w["status"] == "failed" for w in workflows)
    known = [w["slo_ratio"] for w in workflows if w.get("slo_ratio") is not None]
    unknown = len(workflows) - len(known) - failed
    result = []
    for alpha in ALPHAS:
        violations = failed + sum(ratio > alpha for ratio in known)
        result.append({"alpha": alpha, "registered_workflows": len(workflows),
                       "successful_ratio_n": len(known), "failed_workflows": failed,
                       "unknown_workflows": unknown,
                       "violations_known": violations if formal_started else None,
                       "violation_rate_lower": violations / len(workflows) if formal_started else None,
                       "violation_rate_upper": (violations + unknown) / len(workflows) if formal_started else None,
                       "population": "all expected workflows; failures violate every alpha; incomplete outcomes bound"
                                     if formal_started else "not an executed formal experiment"})
    return result


def planner_coverage(rows):
    successful = [r for r in rows if r.get("router_mode") == "hexgent" and not r.get("error")]
    planned = sum(isinstance(r.get("schedule_revision"), int) and r["schedule_revision"] > 0 for r in successful)
    fallback = sum(r.get("schedule_revision") == 0 for r in successful)
    return {"applicable": bool(successful), "successful_hexgent_calls": len(successful),
            "planner_revision_calls": planned, "fallback_revision_calls": fallback,
            "missing_revision_calls": len(successful) - planned - fallback,
            "planner_revision_coverage": planned / len(successful) if successful else None,
            "planner_effect_observed": planned > 0}


def planner_metrics(rows, *, formal_started):
    formal = rows if formal_started else []
    actor = [r for r in formal if r.get("endpoint") == "/plan_pending_admissions"]
    nonempty = [r for r in formal if r.get("endpoint") in ("/schedule_request", "/plan")
                or r.get("admission_count", 0) > 0 or r.get("result_count", 0) > 0]
    return {"formal_invocation_count": len(formal), "excluded_nonformal_records": len(rows) - len(formal),
            "endpoints": dict(Counter(r.get("endpoint", "missing") for r in formal)),
            "nonempty_planning_invocations": len(nonempty), "event_actor_invocations": len(actor),
            "admission_count": sum(r.get("admission_count", 0) for r in formal),
            "result_count": sum(r.get("result_count", 0) for r in formal),
            "event_count": sum(r.get("event_count", 0) for r in formal),
            "wall_ms": distribution(r.get("wall_ms") for r in formal),
            "cpu_ms": distribution(r.get("cpu_ms") for r in formal),
            "cpu_aggregation": "per-invocation distribution; overlapping invocation CPU durations are not summed",
            "formal_started": formal_started}


class Inputs:
    def __init__(self):
        self.sources = {}

    def bytes(self, path, optional=False):
        path = Path(path)
        if not path.exists():
            if not optional:
                raise FileNotFoundError(path)
            self.sources[str(path)] = {"exists": False}
            return None
        data = path.read_bytes()
        self.sources[str(path)] = {"exists": True, "bytes": len(data),
                                   "sha256": hashlib.sha256(data).hexdigest()}
        return data

    def json(self, path, optional=False):
        data = self.bytes(path, optional)
        return json.loads(data) if data is not None else None

    def rows(self, path, optional=False):
        data = self.bytes(path, optional)
        return [json.loads(line) for line in data.splitlines() if line.strip()] if data is not None else []


def worker_capture(record, directory, run_root, io, formal_started):
    if not formal_started:
        return {"formal_capture_status": "not_applicable_formal_not_started",
                "complete": None, "received_http_attempts": None, "stale_rate": None}
    complete = record.get("worker_log_observation_complete") is True
    evidence = {}
    for boundary in ("start", "end"):
        snapshot = record.get("worker_capture_status_" + boundary, {})
        workers = {p["name"]: p for p in snapshot.get("processes", [])
                   if p["name"] in ("prefill", "decode")}
        okay = set(workers) == {"prefill", "decode"} and all(
            p.get("truncated") is False and "write_error" in p and p["write_error"] is None
            for p in workers.values())
        complete = complete and okay
        evidence[boundary] = workers
    slices = {}
    for name in ("prefill", "decode"):
        data = io.bytes(directory / (name + ".worker-slice.log"), optional=True)
        original = io.bytes(run_root / (name + ".log"))
        start = record.get("worker_log_offsets_start", {}).get(name)
        end = record.get("worker_log_offsets_end", {}).get(name)
        matches = data is not None and start is not None and end is not None and original[start:end] == data
        complete = complete and matches
        slices[name] = {"start": start, "end": end, "exact_original_bytes_match": matches,
                        "slice_bytes": len(data) if data is not None else None}
    summary = io.json(directory / "worker-mutation-summary.json", optional=True)
    observed = io.json(directory / "worker-mutation-summary.observed.json", optional=True)
    if summary is None:
        complete = False
    final = dict(summary or {})
    final.update({"complete": complete, "capture_boundary_evidence": evidence,
                  "byte_slices": slices,
                  "formal_capture_status": "complete_bounded_endpoint_window" if complete else "incomplete",
                  "unit": "received HTTP attempts and observed backend outputs; DP aggregation is not assumed to equal TP ranks",
                  "upstream_transport_and_prehandler_failures": "unobserved"})
    if not complete:
        for key in list(final):
            if key.endswith("_rate"):
                final[key] = None
        if "rank_outcome_rates" in final:
            final["rank_outcome_rates"] = dict.fromkeys(final["rank_outcome_rates"], None)
    final["observed_report_present"] = observed is not None
    return final


def stage_logs(directory, io, formal_started):
    if not formal_started:
        return {"status": "not_applicable_formal_not_started"}
    result = {"status": "sampled_formal_worker_log_window", "stages": {}}
    for stage, filename in (("prefill", "prefill.worker-slice.log"), ("decode", "decode.worker-slice.log")):
        data = io.bytes(directory / filename, optional=True)
        if data is None:
            result["stages"][stage] = {"status": "unavailable"}
            continue
        lines = data.decode(errors="replace").splitlines()
        failed = set()
        for line in lines:
            if "transfer failed for request" in line.lower():
                match = re.search(r"req\.rid='([^']+)'", line)
                if match:
                    failed.add(match.group(1))
        counters = {}
        for key in ("new-seq", "running-req", "queue-req", "pending-token",
                    "bootstrap-req", "inflight-req", "prealloc-req", "transfer-req", "retracted-req"):
            values = [int(m.group(1)) for line in lines
                      for m in [re.search(r"#" + key + r":\s*(\d+)", line)] if m]
            if values:
                counters[key] = {"samples": len(values), "max": max(values),
                                 "positive_samples": sum(v > 0 for v in values)}
        timings = {}
        failed_records = health_records = record_count = 0
        for line in lines:
            match = re.search(r"ReqTimeStats\(rid=([^,]+),.*type=(prefill|decode)\): (.*)", line)
            if not match or match.group(2) != stage:
                continue
            rid = match.group(1)
            if rid.startswith("HEALTH_CHECK"):
                health_records += 1
                continue
            if rid in failed:
                failed_records += 1
                continue
            record_count += 1
            for key, value in re.findall(r"(\w+_duration)=([0-9.]+)ms", match.group(3)):
                timings.setdefault(key, []).append(float(value))
        result["stages"][stage] = {
            "logged_nonfailed_stage_records": record_count, "excluded_known_failed_records": failed_records,
            "excluded_health_records": health_records, "sampled_counters": counters,
            "logged_duration_ms": {key: distribution(values) for key, values in timings.items()},
            "unit": "logged worker stage records, not independent workflow samples"}
    result["boundaries"] = {
        "prefill_queue_duration": "P wait_queue_entry to forward_entry",
        "prefill_forward_duration": "P forward_entry to completion; includes execution and post-enqueue transfer completion",
        "decode_transfer_duration": "D transfer_queue_entry to successful wait_queue_entry; includes waiting for P queue/compute",
        "decode_queue_duration": "D wait_queue_entry to forward_entry",
        "failure_zero": "Missing success timestamps can log0ms; known failed records excluded from duration distributions",
        "sampling": "Batch counters are sampled log observations; absence of a positive queue sample is not continuous absence.",
        "GPU_utilization": "not measured by these counters"}
    return result


def concurrency(rows):
    events = []
    for row in rows:
        start, end = row.get("request_started_at_ms"), row.get("response_completed_at_ms")
        if start is not None and end is not None and end > start:
            events.extend([(start, 1, row["workflow_id"]), (end, -1, row["workflow_id"])])
    active = Counter()
    calls = peak_calls = peak_workflows = 0
    for _, change, wid in sorted(events, key=lambda event: (event[0], event[1])):
        calls += change
        active[wid] += change
        if active[wid] == 0:
            del active[wid]
        peak_calls, peak_workflows = max(peak_calls, calls), max(peak_workflows, len(active))
    return {"max_overlapping_client_calls": peak_calls, "max_overlapping_client_workflows": peak_workflows,
            "boundary": "half-open [request_started_at_ms,response_completed_at_ms); ends before starts at ties"}


def flatten_case(case):
    ratio, req = case["workflow_slo_ratio_successful"], case["request_latency_ms_all_attempts"]
    return {"ordinal": case["ordinal"], "dataset": case["dataset"], "load": case["load"],
            "configured_method": case["method"], "formal_status": case["formal_status"],
            "valid_measurement": case["saved_valid_measurement"],
            "expected_calls": case["expected_calls"], "formal_rows": case["formal_rows"],
            "request_errors": case["request_errors"], "usage_verified_calls": case["usage_verified_calls"],
            "successful_workflows": case["workflow_status_counts"].get("successful", 0),
            "failed_workflows": case["workflow_status_counts"].get("failed", 0),
            "ratio_n": ratio["count"], "ratio_p95": ratio["p95"], "ratio_p99": ratio["p99"],
            "Req_n": req["count"], "Req95_ms": req["p95"], "Req99_ms": req["p99"],
            "planner_revision_coverage": case["planner_coverage"]["planner_revision_coverage"],
            "canonical_planned_calls": case["planner_coverage"]["planner_revision_calls"],
            "formal_planner_actor_n": case["planner_overhead"]["formal_invocation_count"],
            "formal_nonempty_planning_n": case["planner_overhead"]["nonempty_planning_invocations"],
            "W2_received_attempts": case["worker_mutations"].get("received_http_attempts"),
            "W2_complete_capture": case["worker_mutations"].get("complete"),
            **{"alpha_" + str(int(r["alpha"])) + "_violation_lower": r["violation_rate_lower"]
               for r in case["failure_inclusive_alphas"]}}


def aggregate(run_root, source_root):
    io = Inputs()
    paired = run_root / "paired-pilot"
    plan = io.json(paired / "run-plan.json")
    block = io.json(paired / "block-results.json")
    references = io.rows(run_root / "reference-collection/workflow-references.jsonl")
    refs = {row["workflow_id"]: row for row in references}
    contract = io.json(run_root / "reference-collection/workflow-reference-contract.json")
    for path in [run_root / "actual-runtime-observations.json", run_root / "expected-worker-limits.json",
                 run_root / "protocol-budget-amendment.json", paired / "continuation-decisions.jsonl",
                 paired / "block-results.before-continuation.json",
                 run_root / "audit-metrics/qc-reference.json", run_root / "audit-metrics/transport-failure-audit.json"]:
        io.bytes(path, optional=True)
    for relative in ["scripts/hexgent_real_trace_collectors/summarize_scheduling_replay.py",
                     "scripts/hexgent_real_trace_collectors/summarize_worker_mutations.py",
                     "python/sglang/hexgent/workflow_trace.py", "python/sglang/hexgent/sidecar.py",
                     "python/sglang/srt/managers/scheduler_components/metrics_reporter.py",
                     "python/sglang/srt/observability/req_time_stats.py"]:
        io.bytes(source_root / relative)
    prep = run_root.parent / "r2-runtime-preparation"
    for filename in ("run_paired_pilot.py", "h20_run_support.py", "run_budgeted_block.py",
                     "continue_unattempted_pairs.py"):
        io.bytes(prep / filename)
    cells = {}
    for cell in plan["cells"]:
        trace = paired / Path(cell["trace"]).name
        workflows = io.rows(trace)
        if hashlib.sha256(trace.read_bytes()).hexdigest() != cell["trace_sha256"]:
            raise ValueError("frozen planned trace hash changed")
        cells[(cell["dataset"], cell["load"])] = (cell, workflows)
    cases, ledger, mismatches, corrections = [], [], [], []
    for run in plan["runs"]:
        ordinal = run["ordinal"]
        name = "%02d-%s-%s-%s" % (ordinal, run["dataset"], run["load"], run["method"])
        directory = paired / name
        record = io.json(directory / "run.json", optional=True) or {}
        saved = io.json(directory / "summary.json", optional=True)
        measurement = io.json(directory / "measurement-gate.json", optional=True)
        warm_gate = io.json(directory / "warmup-gate.json", optional=True) or {}
        rows = io.rows(directory / "replay.raw.jsonl", optional=True)
        warm_rows = io.rows(directory / "warmup.raw.jsonl", optional=True)
        state = case_state(directory.exists(), rows, warm_gate, record)
        formal_started = state == "FORMAL_STARTED"
        cell, expected = cells[(run["dataset"], run["load"])]
        expected_ids = {w["workflow_id"] for w in expected}
        if {r["workflow_id"] for r in rows} - expected_ids:
            raise ValueError("unexpected formal workflow ID")
        workflows = []
        raw_source = io.sources[str(directory / "replay.raw.jsonl")]
        for workflow in expected:
            wid = workflow["workflow_id"]
            row = workflow_record(workflow, [r for r in rows if r["workflow_id"] == wid],
                                  refs[wid], state, contract["model"])
            row.update({"ordinal": ordinal, "dataset": run["dataset"], "load": run["load"],
                        "configured_method": run["method"], "case_state": state,
                        "formal_raw_path": str(directory / "replay.raw.jsonl"),
                        "formal_raw_sha256": raw_source.get("sha256"),
                        "reference_table_path": str(run_root / "reference-collection/workflow-references.jsonl")})
            workflows.append(row)
            ledger.append(row)
        counts = Counter(w["status"] for w in workflows)
        if formal_started:
            formal_status = "COMPLETE_SUCCESS" if counts["successful"] == 4 else (
                "COMPLETE_WITH_FAILURES" if all(not w["missing_call_ids"] for w in workflows) else "INCOMPLETE")
        else:
            formal_status = state
        successes = [r for r in rows if not r.get("error")]
        coverage = planner_coverage(rows)
        telemetry = io.rows(directory / "planner-telemetry.formal.jsonl", optional=True)
        full_telemetry = io.rows(directory / "planner-telemetry.full.jsonl", optional=True)
        planner = planner_metrics(telemetry, formal_started=formal_started)
        capture = worker_capture(record, directory, run_root, io, formal_started)
        case = {**run, "directory": str(directory), "formal_status": formal_status,
                "formal_started": formal_started, "saved_run_status": record.get("status"),
                "saved_valid_measurement": record.get("valid_measurement"),
                "saved_measurement_gate": measurement,
                "expected_workflows": len(expected), "expected_calls": sum(len(w["calls"]) for w in expected),
                "formal_rows": len(rows), "request_errors": sum(bool(r.get("error")) for r in rows) if formal_started else None,
                "usage_verified_calls": sum(w["usage_verified_calls"] for w in workflows),
                "workflow_status_counts": dict(counts),
                "workflow_latency_ms_successful": distribution(w["successful_latency_ms"] for w in workflows),
                "workflow_slo_ratio_successful": distribution(w["slo_ratio"] for w in workflows),
                "request_latency_ms_all_attempts": distribution(r.get("total_latency_ms") for r in rows),
                "request_latency_ms_successful_HTTP": distribution(r.get("total_latency_ms") for r in successes),
                "failure_inclusive_alphas": alpha_metrics(workflows, formal_started=formal_started),
                "planner_coverage": coverage, "planner_overhead": planner,
                "planner_full_file_records": len(full_telemetry),
                "router_mode_counts": dict(Counter(str(r.get("router_mode")) for r in rows)),
                "schedule_revision_counts": dict(Counter(str(r.get("schedule_revision")) for r in rows)),
                "router_overhead_ms": {key: distribution(r.get(key) for r in rows)
                                       for key in ("router_selection_wall_ms", "router_snapshot_fetch_ms", "router_admission_wait_ms")},
                "worker_mutations": capture, "stage_logs": stage_logs(directory, io, formal_started),
                "client_concurrency": concurrency(rows) if formal_started else None,
                "cgroup_cpu_formal_delta": record.get("cgroup_cpu_formal_delta"),
                "cpu_scope": record.get("cpu_scope"),
                "warmup": {"rows": len(warm_rows), "errors": sum(bool(r.get("error")) for r in warm_rows),
                           "gate": warm_gate, "raw_path": str(directory / "warmup.raw.jsonl"),
                           "excluded_from_formal": True},
                "KV_bytes": {"kind": "observed-request input-token byte estimate",
                             "bytes_per_input_token": 327680,
                             "estimate_bytes": sum(r["input_tokens"] * 327680 for r in rows) if formal_started else None,
                             "actual_wire_transfer_bytes": None},
                "GPU_utilization": None}
        if run["method"] == "ours" and formal_started and coverage["planner_revision_coverage"] == 0:
            case["interpretation"] = "Configured Ours with revision0 native admission fastpath; canonical planner placement effect unobserved"
        elif not formal_started:
            case["interpretation"] = formal_status
        elif formal_status != "COMPLETE_SUCCESS":
            case["interpretation"] = "Retained failure-inclusive formal trajectory; successful workflow ratios conditional on success"
        else:
            case["interpretation"] = "Complete saved formal trajectory"
        if saved:
            if saved["calls"] != len(rows):
                mismatches.append({"ordinal": ordinal, "field": "saved calls"})
            if formal_started:
                for item in saved["workflows"]:
                    calculated = next(w for w in workflows if w["workflow_id"] == item["workflow_id"])
                    for key, field in (("latency_ms", "observed_latency_ms"), ("slo_ratio", "slo_ratio")):
                        if item[key] != calculated[field]:
                            mismatches.append({"ordinal": ordinal, "workflow_id": item["workflow_id"], "field": key})
                for key in ("count", "p95", "p99"):
                    if saved["call_latency_ms"][key] != case["request_latency_ms_all_attempts"][key]:
                        mismatches.append({"ordinal": ordinal, "field": "request_" + key})
                if saved["planner_plan_execution"] != coverage:
                    mismatches.append({"ordinal": ordinal, "field": "planner coverage"})
            elif telemetry:
                corrections.append({"ordinal": ordinal, "field": "formal planner overhead",
                                    "saved_summary_invocations": saved["planner_invocations"]["count"],
                                    "excluded_warmup_only_records": len(telemetry),
                                    "final_formal_invocations": 0,
                                    "evidence": "No cpu_formal_before/worker_log_offsets_start, unsuccessful warmup, formal raw empty"})
        cases.append(case)
    if len(cases) != 8 or len(ledger) != 32:
        raise ValueError("all eight cases and32 registered workflow rows required")
    status_counts = dict(Counter(w["status"] for w in ledger))
    result = {"analysis_version": 1, "run_root": str(run_root),
              "reference_contract": contract, "percentile_definition": "nearest rank ceil(q*n),1-based; median uses middle/average",
              "workflow_ratio_population": "successful complete workflows only; errors retain null ratios",
              "request_population": "all observed client attempts including errors; successful-HTTP population separately retained",
              "overall": {"registered_cases": 8, "case_directories": sum(Path(c["directory"]).exists() for c in cases),
                          "formal_started_cases": sum(c["formal_started"] for c in cases),
                          "valid_saved_cases": sum(c["saved_valid_measurement"] is True for c in cases),
                          "registered_workflow_rows": 32, "workflow_status_counts": status_counts,
                          "formal_request_rows": sum(c["formal_rows"] for c in cases),
                          "formal_request_errors": sum(c["request_errors"] or 0 for c in cases),
                          "warmup_client_rows_excluded": sum(c["warmup"]["rows"] for c in cases),
                          "all_runs_attempted": block["all_runs_attempted"],
                          "complete_balanced_block": block["complete_balanced_block"]},
              "cases": cases, "workflows": ledger,
              "validation": {"recomputed_vs_saved_summary_mismatches": mismatches,
                             "intentional_context_corrections": corrections},
              "sources": io.sources,
              "limitations": [
                  "One registered block; correlated workflow/call observations have no repeat-based confidence interval.",
                  "SG configuredOurs revision0 fastpath has canonical planner coverage0; metrics describe that effective execution.",
                  "Hot low cases05/06 were not run for fixed budget. Hot07 retains transport failures; Hot08 failed warmup before formal calls.",
                  "Planner event-only actor timing and router timing have different boundaries. Cgroup CPU delta includes the containing runtime; invocation CPU intervals are not summed.",
                  "W2 measures received completed endpoint outcomes in exact captured slices; upstream transport/prehandler attempts remain unobserved. Zero attempts gives undefined rates.",
                  "Worker stage logs are sampled and rounded. P forward includes transfer completion; D transfer pipeline can include Pqueue/compute; failed missing-endpoint zeros are diagnostic.",
                  "GPU utilization time series and actual wire KV bytes were not measured; input-token byte estimates retain explicit labels."]}
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def render_report(data):
    def fmt(value, digits=6):
        return "N/A" if value is None else str(value) if isinstance(value, int) else ("%.*f" % (digits, value))
    lines = [
        "The saved block contains four complete SG cases, one failure-inclusive Hot burst case, two budget-skipped Hot low cases, and one Hot burst case blocked during warmup.",
        "",
        "SG configured Ours used the revision0 admission fastpath for32/32 successful calls: canonical planner coverage is0. Hot07 completed one workflow with ratio112567.46606445312/68896.486328125=1.6338636709039356; its other three workflows failed and retain null ratios. Hot08 has zero formal calls and ten warmup-only actor records excluded from formal overhead.",
        "",
        "| Case | Effective status | Calls/errors | WF success/fail | Req95/99 ms (n) | Successful ratio95/99 (n) | Canonical coverage |",
        "|---|---|---:|---:|---|---|---|"]
    for c in data["cases"]:
        req, ratio = c["request_latency_ms_all_attempts"], c["workflow_slo_ratio_successful"]
        counts = c["workflow_status_counts"]
        lines.append("| %02d %s %s %s | %s | %s/%s | %s/%s | %s / %s (%s) | %s / %s (%s) | %s |" %
                     (c["ordinal"], c["dataset"], c["load"], c["method"], c["formal_status"],
                      c["formal_rows"], c["request_errors"] if c["request_errors"] is not None else "N/A",
                      counts.get("successful", 0), counts.get("failed", 0),
                      fmt(req["p95"], 3), fmt(req["p99"], 3), req["count"],
                      fmt(ratio["p95"]), fmt(ratio["p99"]), ratio["count"],
                      fmt(c["planner_coverage"]["planner_revision_coverage"], 3)))
    lines += ["", "Percentiles use nearest rank ceil(q*n). For the16-call SG cases, Req95 and Req99 both select the maximum; four-workflow ratio95/99 select the maximum. Hot07 request statistics include38 HTTP503 outcomes; its successful-HTTP request distribution is stored separately. The one successful Hot workflow gives ratio95=ratio99 with n=1.",
              "", "| Case | Failure-inclusive V@1 | V@2 | V@4 | Formal actor records / nonempty planning | W2 received attempts / complete capture |",
              "|---|---|---|---|---|---|"]
    for c in data["cases"]:
        a = c["failure_inclusive_alphas"]
        rates = [fmt(r["violation_rate_lower"], 3) if r["violation_rate_lower"] == r["violation_rate_upper"]
                 else "%s–%s" % (fmt(r["violation_rate_lower"], 3), fmt(r["violation_rate_upper"], 3)) for r in a]
        w2 = c["worker_mutations"]
        lines.append("| %02d | %s | %s | %s | %d / %d | %s / %s |" %
                     (c["ordinal"], *rates, c["planner_overhead"]["formal_invocation_count"],
                      c["planner_overhead"]["nonempty_planning_invocations"],
                      w2.get("received_http_attempts", "N/A"), w2.get("complete")))
    lines += ["", "Failed workflows count as violations at every alpha. Unexecuted cases have N/A rates. Hot07 has V@1=4/4 and V@2=V@4=3/4. All five captured formal worker windows retain complete bounded log evidence and zero received mutation attempts; stale/apply rates are N/A with zero denominator.",
              "", "The SG formal planner files contain38 and33 event-only actor records, respectively, with48 events each and zero admissions/results. Router selection and actor wall/CPU distributions, actual containing-cgroup CPU deltas, exact worker byte-slice checks, sampled P/D counters and stage durations are in ledger.json. Warmup is recorded separately for all six attempted controllers.",
              "", "| Case/WF | Status | Observed latency ms | Reference base ms | Valid ratio |",
              "|---|---|---:|---:|---:|"]
    for w in data["workflows"]:
        lines.append("| %02d / %s | %s | %s | %s | %s |" %
                     (w["ordinal"], w["workflow_id"], w["status"], fmt(w["observed_latency_ms"], 6),
                      fmt(w["base_slo_ms"], 6), fmt(w["slo_ratio"], 12)))
    lines += ["", "CSV and JSON retain full floating-point values. Failed observed latency ends at the last observed terminal response and remains diagnostic; only successful workflow latency enters normalized ratio distributions.",
              "", "Evidence: " + str(Path(data["run_root"]) / "paired-pilot/run-plan.json") + "; " +
              str(Path(data["run_root"]) / "reference-collection/workflow-references.jsonl") + "; per-case replay.raw.jsonl, run.json, gates, planner-telemetry.formal.jsonl and exact worker slices. Every consumed source path, byte count and SHA256 is listed in ledger.json.sources.",
              "", "Transport chronology and source boundaries are preserved in " +
              str(Path(data["run_root"]) / "audit-metrics/transport-failure-audit.json") + ".",
              "", "Interpretation limits: " + " ".join(data["limitations"]), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    names = ("ledger.json", "cases.csv", "workflows.csv", "report.md")
    if any((output / name).exists() for name in names):
        raise FileExistsError("analysis outputs already exist; use a fresh output directory")
    data = aggregate(args.run_root, args.source_root)
    data["aggregator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if data["validation"]["recomputed_vs_saved_summary_mismatches"]:
        raise ValueError(data["validation"]["recomputed_vs_saved_summary_mismatches"])
    with (output / "ledger.json").open("x") as target:
        json.dump(data, target, ensure_ascii=False, indent=2)
        target.write("\n")
    write_csv(output / "cases.csv", [flatten_case(case) for case in data["cases"]])
    write_csv(output / "workflows.csv", data["workflows"])
    with (output / "report.md").open("x") as target:
        target.write(render_report(data))
    print(json.dumps({"output_dir": str(output), "overall": data["overall"], "validation": data["validation"]}, indent=2))


if __name__ == "__main__":
    main()
