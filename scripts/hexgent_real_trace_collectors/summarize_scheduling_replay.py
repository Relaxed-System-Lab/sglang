#!/usr/bin/env python3
"""Summarize route, workflow SLO, planner overhead, and KV transfer evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
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
        return {
            "count": 0, "mean": None, "median": None,
            "p95": None, "p99": None, "max": None,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
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
    coverage = (
        planner_revision_calls / len(successful) if successful else None
    )
    return {
        "applicable": bool(successful),
        "successful_hexgent_calls": len(successful),
        "planner_revision_calls": planner_revision_calls,
        "fallback_revision_calls": fallback_revision_calls,
        "missing_revision_calls": missing_revision_calls,
        "planner_revision_coverage": coverage,
        "planner_effect_observed": planner_revision_calls > 0,
    }


def fixed_scale_violations(workflows: list[dict], scales: list[float]) -> list[dict]:
    """Bound violations while keeping incomplete outcomes and missing SLOs unknown."""
    ratios = [float(row["slo_ratio"]) for row in workflows
              if not row["errors"] and row.get("status") != "incomplete"
              and row["slo_ratio"] is not None]
    failed = sum(bool(row["errors"]) for row in workflows)
    incomplete = sum(row.get("status") == "incomplete" and not row["errors"]
                     for row in workflows)
    unknown = len(workflows) - failed - len(ratios)
    total = len(workflows)
    result = []
    for scale in scales:
        violations = sum(ratio > scale for ratio in ratios)
        result.append({
            "scale": scale,
            "total_workflows": total,
            "profiled_successful_workflows": len(ratios),
            "profiled_successful_violations": violations,
            "failed_workflows": failed,
            "unknown_successful_workflows": unknown - incomplete,
            "incomplete_workflows": incomplete,
            "unknown_workflows": unknown,
            "violation_rate_lower_bound": (failed + violations) / total if total else None,
            "violation_rate_upper_bound": (failed + violations + unknown) / total if total else None,
        })
    return result


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



WORKFLOW_REFERENCE_KIND = "isolated_workflow_replay"
WORKFLOW_SLO_DEFINITION = "isolated_workflow_replay_on_finite_1p1d"
CP_PROXY_SLO_DEFINITION = "dag_critical_path_of_exclusive_call_latency_proxy"


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdefABCDEF" for character in value))


def workflow_references(workflows: list[dict], rows: list[dict], frozen_contract: dict) -> dict:
    """Select isolated workflow references; unavailable measurements stay unknown.

    Hash fields bind recorded provenance. The producer/root verifies the raw
    replay and runtime evidence contents before freezing this table/contract.
    """
    contract_fields = ("model", "implementation_revision", "runtime_evidence_sha256")
    if (not isinstance(frozen_contract, dict)
            or any(not isinstance(frozen_contract.get(key), str)
                   or not frozen_contract[key] for key in contract_fields)
            or not _is_sha256(frozen_contract["runtime_evidence_sha256"])):
        raise ValueError("workflow reference contract needs model, implementation revision and runtime SHA256")
    expected = {str(workflow["workflow_id"]): len(workflow["calls"]) for workflow in workflows}
    if any(count <= 0 for count in expected.values()):
        raise ValueError("workflow reference requires nonempty expected workflows")
    required = {
        "workflow_id", "reference_kind", "latency_ms", "status", "expected_calls",
        "completed_calls", "errors", "usage_verified_calls", "model",
        "implementation_revision", "raw_replay_sha256", "runtime_evidence_sha256",
    }
    indexed = {}
    for row in rows:
        if required - row.keys():
            raise ValueError(f"workflow reference row lacks fields: {sorted(required - row.keys())}")
        workflow_id = str(row["workflow_id"])
        if workflow_id in indexed:
            raise ValueError(f"duplicate workflow reference: {workflow_id}")
        if row["reference_kind"] != WORKFLOW_REFERENCE_KIND:
            raise ValueError(f"workflow reference kind is not isolated workflow replay: {workflow_id}")
        if any(row[key] != frozen_contract[key] for key in contract_fields):
            raise ValueError(f"workflow reference metadata differs from frozen contract: {workflow_id}")
        if not _is_sha256(row["raw_replay_sha256"]) or not _is_sha256(row["runtime_evidence_sha256"]):
            raise ValueError(f"workflow reference provenance needs valid SHA256 fields: {workflow_id}")
        if not isinstance(row["status"], str) or not row["status"]:
            raise ValueError(f"workflow reference status is missing: {workflow_id}")
        latency = None
        if row["status"] == "successful":
            counts = [row[key] for key in ("expected_calls", "completed_calls", "usage_verified_calls")]
            if (any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in counts)
                    or len(set(counts)) != 1 or row["errors"] != 0
                    or (workflow_id in expected and counts[0] != expected[workflow_id])):
                raise ValueError(f"workflow reference successful counts disagree: {workflow_id}")
            try:
                latency = float(row["latency_ms"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"workflow reference latency is not numeric: {workflow_id}") from error
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"workflow reference latency must be positive and finite: {workflow_id}")
        indexed[workflow_id] = {"latency_ms": latency, "status": row["status"], "row": row}
    return {workflow_id: indexed.get(workflow_id, {"latency_ms": None, "status": "missing", "row": None})
            for workflow_id in expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--trace", type=Path,
        help="Expected workflow JSONL (workflow_id, calls) used to verify replay completeness.",
    )
    parser.add_argument("--planner-telemetry", type=Path)
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--exclusive-profile",
        type=Path,
        help="Per-call exclusive latency JSONL for the DAG critical-path proxy denominator.",
    )
    reference_group.add_argument(
        "--workflow-reference", type=Path,
        help="Isolated whole-workflow replay JSONL; finite 1P1D elapsed denominator, including fanout contention.",
    )
    parser.add_argument(
        "--workflow-reference-contract", type=Path,
        help="Frozen model, implementation_revision and runtime_evidence_sha256 JSON for workflow references.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slo-scales", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument(
        "--kv-bytes-per-input-token", type=int, default=131072,
        help="Model-specific KV bytes per input token for a byte estimate, not measured transfer.",
    )
    args = parser.parse_args()
    if args.workflow_reference is not None:
        if args.trace is None or args.workflow_reference_contract is None:
            parser.error("--workflow-reference requires --trace and --workflow-reference-contract")
        if not args.workflow_reference.exists():
            raise FileNotFoundError(args.workflow_reference)
    elif args.workflow_reference_contract is not None:
        parser.error("--workflow-reference-contract requires --workflow-reference")

    rows = read_jsonl(args.replay)
    exclusive_profile = (
        load_exclusive_profile(args.exclusive_profile)
        if args.exclusive_profile
        else None
    )
    by_workflow = defaultdict(list)
    for row in rows:
        by_workflow[str(row["workflow_id"])].append(row)

    expected_calls = None
    expected_workflows = []
    if args.trace is not None:
        if not args.trace.exists():
            raise FileNotFoundError(args.trace)
        expected_calls = {}
        for workflow in read_jsonl(args.trace):
            expected_workflows.append(workflow)
            workflow_id = str(workflow["workflow_id"])
            if workflow_id in expected_calls:
                raise ValueError(f"duplicate workflow_id in expected trace: {workflow_id}")
            call_ids = [str(call["call_id"]) for call in workflow["calls"]]
            if len(set(call_ids)) != len(call_ids):
                raise ValueError(f"duplicate call_id in expected trace: {workflow_id}")
            expected_calls[workflow_id] = set(call_ids)
        unexpected_workflows = sorted(by_workflow.keys() - expected_calls.keys())
        if unexpected_workflows:
            raise ValueError(f"unexpected workflow_id(s) in replay: {unexpected_workflows}")

    workflow_reference_contract = None
    selected_references = None
    if args.workflow_reference is not None:
        workflow_reference_contract = json.loads(args.workflow_reference_contract.read_text(encoding="utf-8-sig"))
        selected_references = workflow_references(
            expected_workflows, read_jsonl(args.workflow_reference), workflow_reference_contract
        )

    workflows = []
    workflow_ids = expected_calls.keys() if expected_calls is not None else by_workflow.keys()
    for workflow_id in sorted(workflow_ids):
        calls = by_workflow.get(workflow_id, [])
        observed_call_ids = {str(row["call_id"]) for row in calls}
        if len(observed_call_ids) != len(calls):
            raise ValueError(f"duplicate call_id in workflow replay rows: {workflow_id}")
        missing_call_ids = None
        expected_count = None
        if expected_calls is not None:
            unexpected_calls = sorted(observed_call_ids - expected_calls[workflow_id])
            if unexpected_calls:
                raise ValueError(f"unexpected call_id(s) in {workflow_id}: {unexpected_calls}")
            missing_call_ids = sorted(expected_calls[workflow_id] - observed_call_ids)
            expected_count = len(expected_calls[workflow_id])
        starts = [float(row["request_started_at_ms"]) for row in calls]
        ends = [float(row["response_completed_at_ms"]) for row in calls]
        earliest_start = min(starts, default=None)
        latest_response = max(ends, default=None)
        arrivals = {float(row["workflow_arrival_ms"]) for row in calls
                    if row.get("workflow_arrival_ms") is not None}
        if len(arrivals) > 1:
            raise ValueError(f"inconsistent workflow arrivals: {workflow_id}")
        arrival = next(iter(arrivals), None)
        origin = ("unobserved" if earliest_start is None
                  else "workflow_arrival_ms" if arrival is not None
                  else "earliest_request_started_at_ms")
        latency = (latest_response - (arrival if arrival is not None else earliest_start)
                   if latest_response is not None else None)
        # A partial graph cannot supply an observed completion or a critical
        # path. Keep its evidence without trying to traverse missing parents.
        if selected_references is not None:
            reference = selected_references[workflow_id]
            slo = reference["latency_ms"]
            reference_kind = WORKFLOW_REFERENCE_KIND
            reference_status = reference["status"]
            reference_raw_row = reference["row"]
        else:
            slo = None if missing_call_ids else critical_path_profile_latency_ms(
                calls, exclusive_profile=exclusive_profile
            )
            reference_kind = CP_PROXY_SLO_DEFINITION
            reference_status = "successful" if slo is not None else "missing"
            reference_raw_row = None
        errors = sum(bool(row.get("error")) for row in calls)
        status = "failed" if errors else "incomplete" if missing_call_ids else "successful"
        workflows.append(
            {
                "workflow_id": workflow_id,
                "calls": len(calls),
                "expected_calls": expected_count,
                "missing_call_ids": missing_call_ids,
                "completeness_verified": expected_calls is not None,
                "call_coverage": len(calls) / expected_count if expected_count else None,
                "workflow_arrival_ms": arrival,
                "latency_origin": origin,
                "earliest_request_started_at_ms": earliest_start,
                "response_completed_at_ms": latest_response,
                "latency_ms": latency,
                "base_slo_ms": slo,
                "profile_slo_ms": slo,
                "reference_kind": reference_kind,
                "reference_status": reference_status,
                "reference_raw_row": reference_raw_row,
                "slo_ratio": latency / slo if status == "successful" and slo and slo > 0 else None,
                "errors": errors,
                "status": status,
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
    successful_workflows = [row for row in workflows if row["status"] == "successful"]
    successful_ratios = [
        float(row["slo_ratio"])
        for row in successful_workflows
        if row["slo_ratio"] is not None
    ]
    latency_origins = {row["latency_origin"] for row in workflows}
    expected_call_count = (
        sum(len(call_ids) for call_ids in expected_calls.values())
        if expected_calls is not None else None
    )
    output = {
        "calls": len(rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "slo_definition": (WORKFLOW_SLO_DEFINITION if selected_references is not None else CP_PROXY_SLO_DEFINITION),
        "workflow_reference_path": str(args.workflow_reference) if args.workflow_reference is not None else None,
        "workflow_reference_sha256": (hashlib.sha256(args.workflow_reference.read_bytes()).hexdigest()
                                      if args.workflow_reference is not None else None),
        "workflow_reference_contract_path": (str(args.workflow_reference_contract)
                                             if args.workflow_reference_contract is not None else None),
        "workflow_reference_contract_sha256": (hashlib.sha256(args.workflow_reference_contract.read_bytes()).hexdigest()
                                               if args.workflow_reference_contract is not None else None),
        "workflow_reference_contract": workflow_reference_contract,
        "workflow_reference_provenance_validation": (
            "row metadata/hash fields checked against frozen contract; raw/evidence contents verified by producer/root"
            if selected_references is not None else None
        ),
        "exclusive_profile_path": (
            str(args.exclusive_profile) if args.exclusive_profile else None
        ),
        "expected_trace_path": str(args.trace) if args.trace is not None else None,
        "completeness_verified": expected_calls is not None,
        "call_coverage": {
            "expected": expected_call_count,
            "observed": len(rows),
            "missing": expected_call_count - len(rows) if expected_call_count is not None else None,
            "observed_fraction": len(rows) / expected_call_count if expected_call_count else None,
        },
        "workflows": workflows,
        "workflow_counts": {
            "total": len(workflows),
            "observed": len(by_workflow),
            "successful": len(successful_workflows),
            "failed": sum(row["status"] == "failed" for row in workflows),
            "incomplete": sum(row["status"] == "incomplete" for row in workflows),
            "entirely_missing": sum(row["calls"] == 0 for row in workflows),
            "successful_with_reference_slo": len(successful_ratios),
            "successful_missing_reference_slo": len(successful_workflows) - len(successful_ratios),
        },
        "workflow_latency_origin": (
            next(iter(latency_origins)) if len(latency_origins) == 1 else "mixed_or_empty"
        ),
        "percentile_definition": "nearest_rank_ceil_q_times_n",
        "workflow_distribution_population": "successful_workflows",
        "workflow_latency_ms": distribution(
            [float(row["latency_ms"]) for row in successful_workflows]
        ),
        "workflow_slo_ratio": distribution(successful_ratios),
        "fixed_scale_slo_violations": fixed_scale_violations(workflows, args.slo_scales),
        "call_latency_population": "all_attempts",
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
        "planner_plan_execution": planner_plan_execution(rows),
        "route_pairs": [
            {"prefill_worker": pair[0], "decode_worker": pair[1], "calls": count}
            for pair, count in sorted(route_pairs.items(), key=lambda item: str(item[0]))
        ],
        "kv_transfer": {
            "measurement_kind": "input_token_byte_estimate",
            "actual_transfer_bytes": None,
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
