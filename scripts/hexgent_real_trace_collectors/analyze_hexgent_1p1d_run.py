#!/usr/bin/env python3
"""Summarize a fixed-length HexAgenT 1P1D replay from persisted telemetry."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    values = list(values)
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def workflow_metrics(calls: list[dict]) -> dict[str, object]:
    by_workflow: dict[str, list[dict]] = defaultdict(list)
    for call in calls:
        if not call.get("error"):
            by_workflow[str(call["workflow_id"])].append(call)

    workflow_latencies = []
    workflow_slos = []
    slo_ratios = []
    for workflow_calls in by_workflow.values():
        calls_by_id = {str(call["call_id"]): call for call in workflow_calls}
        actual_start = min(float(call["submitted_at"]) for call in workflow_calls)
        actual_end = max(float(call["response_completed_at_ms"]) / 1000 for call in workflow_calls)
        workflow_latency_ms = (actual_end - actual_start) * 1000

        critical_slo: dict[str, float] = {}
        for call in sorted(workflow_calls, key=lambda row: row["submitted_at"]):
            parent_slo = max(
                (critical_slo.get(str(parent), 0.0) for parent in call.get("parents", [])),
                default=0.0,
            )
            critical_slo[str(call["call_id"])] = parent_slo + float(
                call.get("exclusive_profile_latency_ms") or call["total_latency_ms"]
            )
        slo_ms = max(critical_slo.values(), default=0.0)
        workflow_latencies.append(workflow_latency_ms)
        workflow_slos.append(slo_ms)
        if slo_ms:
            slo_ratios.append(workflow_latency_ms / slo_ms)

    return {
        "completed_workflows": len(by_workflow),
        "workflow_latency_ms": summarize(workflow_latencies),
        "critical_path_slo_ms": summarize(workflow_slos),
        "workflow_latency_to_slo": summarize(slo_ratios),
    }


def gpu_metrics(path: Path) -> dict[str, object]:
    samples: dict[str, list[float]] = defaultdict(list)
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        for gpu in parts[1].split(";"):
            fields = [item.strip() for item in gpu.split(",")]
            if len(fields) < 3:
                continue
            try:
                samples[f"gpu{fields[0]}_utilization_pct"].append(float(fields[2]))
            except ValueError:
                continue
    return {name: summarize(values) for name, values in sorted(samples.items())}


def transfer_metrics(
    log_path: Path, startup_log_path: Path | None = None
) -> dict[str, object]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    startup_text = (
        startup_log_path.read_text(encoding="utf-8", errors="replace")
        if startup_log_path is not None and startup_log_path.exists()
        else ""
    )
    durations = [float(match) for match in re.findall(r"transfer_duration=([0-9.]+)ms", text)]
    return {
        "nvlink_transport_selected": "Using Intra-Node NVLink transport" in (
            text + startup_text
        ),
        "transfer_duration_ms": summarize(durations),
    }


def select_decode_log_path(run_dir: Path, log_dir: Path) -> Path:
    """Use the run-local worker log to avoid mixing prior replay telemetry."""
    run_log = run_dir / "decode_worker.log"
    return run_log if run_log.exists() else log_dir / "decode.log"


def planner_metrics(planner_rows: list[dict], router_log: str) -> dict[str, object]:
    """Separate cheap HTTP enqueue work from the actor's actual batch solve."""
    plan_batches = [
        row
        for row in planner_rows
        if row.get("endpoint") == "/plan_pending_admissions"
    ]
    return {
        "endpoint_counts": dict(Counter(row.get("endpoint") for row in planner_rows)),
        "enqueue_wall_ms": summarize(
            row["wall_ms"]
            for row in planner_rows
            if row.get("endpoint") == "/enqueue_admission"
        ),
        "plan_batch_wall_ms": summarize(
            row["wall_ms"] for row in plan_batches if row.get("wall_ms") is not None
        ),
        "plan_batch_cpu_ms": summarize(
            row["cpu_ms"] for row in plan_batches if row.get("cpu_ms") is not None
        ),
        "plan_batch_admission_count": summarize(
            row["admission_count"]
            for row in plan_batches
            if row.get("admission_count") is not None
        ),
        "plan_batch_event_count": summarize(
            row["event_count"]
            for row in plan_batches
            if row.get("event_count") is not None
        ),
        "plan_batch_result_count": summarize(
            row["result_count"]
            for row in plan_batches
            if row.get("result_count") is not None
        ),
        "planner_fallbacks": router_log.count("planner decision unavailable after"),
    }


def call_metrics(calls: list[dict]) -> dict[str, object]:
    completed = [call for call in calls if not call.get("error")]
    return {
        "total": len(calls),
        "errors": len(calls) - len(completed),
        "request_total_latency_ms": summarize(
            call["total_latency_ms"] for call in completed
        ),
        "router_selection_wall_ms": summarize(
            call.get("router_selection_wall_ms", 0.0) for call in completed
        ),
        "router_snapshot_fetch_ms": summarize(
            call["router_snapshot_fetch_ms"]
            for call in completed
            if call.get("router_snapshot_fetch_ms") is not None
        ),
        "router_admission_wait_ms": summarize(
            call["router_admission_wait_ms"]
            for call in completed
            if call.get("router_admission_wait_ms") is not None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    calls = read_jsonl(args.run_dir / "calls.jsonl")
    planner_rows = read_jsonl(args.run_dir / "planner_invocations.jsonl")
    router_log = (args.log_dir / "router.log").read_text(
        encoding="utf-8", errors="replace"
    ) if (args.log_dir / "router.log").exists() else ""

    report = {
        "run": json.loads((args.run_dir / "run.json").read_text())
        if (args.run_dir / "run.json").exists()
        else {},
        "calls": call_metrics(calls),
        "workflows": workflow_metrics(calls),
        "planner": planner_metrics(planner_rows, router_log),
        "gpu": gpu_metrics(args.run_dir / "gpu.csv"),
        "transfer": transfer_metrics(
            select_decode_log_path(args.run_dir, args.log_dir),
            args.log_dir / "decode.log",
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
