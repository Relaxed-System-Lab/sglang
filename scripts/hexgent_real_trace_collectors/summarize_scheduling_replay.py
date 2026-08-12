#!/usr/bin/env python3
"""Summarize route, workflow SLO, planner overhead, and KV transfer evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # Profiles are commonly assembled on Windows, where PowerShell may emit a
    # UTF-8 BOM.  Accept both BOM and non-BOM JSONL so SLO reporting does not
    # fail after a valid data transfer.
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def planner_plan_execution(rows: list[dict]) -> dict:
    successful = [
        row
        for row in rows
        if row.get("router_mode") == "hexgent" and not row.get("error")
    ]
    planner_revision_calls = sum(
        isinstance(row.get("schedule_revision"), int)
        and int(row["schedule_revision"]) > 0
        for row in successful
    )
    fallback_revision_calls = sum(
        row.get("schedule_revision") == 0 for row in successful
    )
    missing_revision_calls = (
        len(successful) - planner_revision_calls - fallback_revision_calls
    )
    coverage = planner_revision_calls / len(successful) if successful else None
    return {
        "applicable": bool(successful),
        "successful_hexgent_calls": len(successful),
        "planner_revision_calls": planner_revision_calls,
        "fallback_revision_calls": fallback_revision_calls,
        "missing_revision_calls": missing_revision_calls,
        "planner_revision_coverage": coverage,
        "planner_effect_observed": planner_revision_calls > 0,
    }


def load_exclusive_profile(path: Path) -> dict[tuple[str, str], float]:
    profile: dict[tuple[str, str], float] = {}
    for row in read_jsonl(path):
        if row.get("error"):
            continue
        key = (str(row["workflow_id"]), str(row["call_id"]))
        if key in profile:
            raise ValueError(f"duplicate exclusive profile row: {key}")
        profile[key] = float(row["total_latency_ms"])
    return profile


def critical_path_profile_latency_ms(
    calls: list[dict],
    *,
    exclusive_profile: dict[tuple[str, str], float] | None = None,
) -> float | None:
    if not calls:
        return None

    calls_by_id = {str(row["call_id"]): row for row in calls}
    if len(calls_by_id) != len(calls):
        raise ValueError("duplicate call_id in workflow replay rows")

    latency_by_id: dict[str, float] = {}
    for call_id, row in calls_by_id.items():
        if exclusive_profile is None:
            value = row.get("exclusive_profile_latency_ms")
        else:
            key = (str(row["workflow_id"]), call_id)
            if key not in exclusive_profile:
                raise ValueError(f"missing exclusive profile row: {key}")
            value = exclusive_profile[key]
        if value is None:
            return None
        latency_by_id[call_id] = float(value)

    memo: dict[str, float] = {}
    visiting: set[str] = set()

    def longest_path_to(call_id: str) -> float:
        if call_id in memo:
            return memo[call_id]
        if call_id in visiting:
            raise ValueError(f"cycle in workflow dependencies at call_id={call_id}")
        visiting.add(call_id)
        parents = [str(parent) for parent in calls_by_id[call_id].get("parents") or []]
        missing = [parent for parent in parents if parent not in calls_by_id]
        if missing:
            raise ValueError(
                f"missing parent(s) for call_id={call_id}: {', '.join(missing)}"
            )
        parent_path = max((longest_path_to(parent) for parent in parents), default=0.0)
        visiting.remove(call_id)
        memo[call_id] = parent_path + latency_by_id[call_id]
        return memo[call_id]

    return max(longest_path_to(call_id) for call_id in calls_by_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--planner-telemetry", type=Path)
    parser.add_argument(
        "--exclusive-profile",
        type=Path,
        help="Per-call profile JSONL used to compute model-matched DAG critical-path SLOs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-bytes-per-input-token", type=int, default=131072)
    args = parser.parse_args()

    rows = read_jsonl(args.replay)
    exclusive_profile = (
        load_exclusive_profile(args.exclusive_profile)
        if args.exclusive_profile
        else None
    )
    by_workflow = defaultdict(list)
    for row in rows:
        by_workflow[str(row["workflow_id"])].append(row)

    workflows = []
    for workflow_id, calls in sorted(by_workflow.items()):
        starts = [float(row["request_started_at_ms"]) for row in calls]
        ends = [float(row["response_completed_at_ms"]) for row in calls]
        latency = max(ends) - min(starts)
        slo = critical_path_profile_latency_ms(
            calls, exclusive_profile=exclusive_profile
        )
        workflows.append(
            {
                "workflow_id": workflow_id,
                "calls": len(calls),
                "latency_ms": latency,
                "profile_slo_ms": slo,
                "slo_ratio": latency / slo if slo and slo > 0 else None,
                "errors": sum(bool(row.get("error")) for row in calls),
            }
        )

    route_pairs = Counter(
        (row.get("prefill_worker"), row.get("decode_worker")) for row in rows
    )
    kv_bytes = [
        int(row.get("input_tokens") or 0) * args.kv_bytes_per_input_token
        for row in rows
    ]
    planner_rows = (
        read_jsonl(args.planner_telemetry) if args.planner_telemetry else []
    )
    output = {
        "calls": len(rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "slo_definition": "dag_critical_path_of_exclusive_call_latency",
        "exclusive_profile_path": (
            str(args.exclusive_profile) if args.exclusive_profile else None
        ),
        "workflows": workflows,
        "workflow_latency_ms": distribution(
            [float(row["latency_ms"]) for row in workflows]
        ),
        "workflow_slo_ratio": distribution(
            [float(row["slo_ratio"]) for row in workflows if row["slo_ratio"] is not None]
        ),
        "call_latency_ms": distribution(
            [float(row["total_latency_ms"]) for row in rows]
        ),
        "router_selection_wall_ms": distribution(
            [
                float(row["router_selection_wall_ms"])
                for row in rows
                if row.get("router_selection_wall_ms") is not None
            ]
        ),
        "route_pairs": [
            {"prefill_worker": pair[0], "decode_worker": pair[1], "calls": count}
            for pair, count in sorted(route_pairs.items(), key=lambda item: str(item[0]))
        ],
        "kv_transfer": {
            "bytes_per_input_token": args.kv_bytes_per_input_token,
            "total_bytes": sum(kv_bytes),
            "per_call_bytes": distribution([float(value) for value in kv_bytes]),
        },
        "planner_invocations": {
            "count": len(planner_rows),
            "wall_ms": distribution([float(row["wall_ms"]) for row in planner_rows]),
            "cpu_ms": distribution([float(row["cpu_ms"]) for row in planner_rows]),
            "max_process_thread_count": max(
                (int(row["process_thread_count"]) for row in planner_rows),
                default=None,
            ),
            "dispatch_ready": sum(row.get("dispatch_ready") is True for row in planner_rows),
            "deferred": sum(row.get("dispatch_ready") is False for row in planner_rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
