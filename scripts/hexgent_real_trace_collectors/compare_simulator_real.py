#!/usr/bin/env python3
"""Compare declared matching simulator predictions and real workflow records.

Inputs are JSON objects with comparison_contract, implementation_revision and
workflows. Errors are predicted minus measured. Missing and failed records stay
in the report, while aggregates use only successful pairs with observed values.
These are descriptive record comparisons; repeated trajectories do not become
independent samples. Synthetic fixtures validate tooling, not H20 calibration.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from statistics import mean


CONTRACT_FIELDS = (
    "trace_sha256", "profile_sha256", "model", "model_revision", "tp",
    "weight_dtype", "kv_dtype", "hardware", "topology", "cache_policy",
    "tool_policy", "arrival", "scheduler_flags", "observation_horizon",
)
METRICS = ("latency_ms", "slo_ratio")


def _index_workflows(document: dict) -> dict[str, dict]:
    rows = {}
    for row in document["workflows"]:
        workflow_id = str(row["workflow_id"])
        if workflow_id in rows:
            raise ValueError(f"duplicate workflow_id: {workflow_id}")
        rows[workflow_id] = row
    return rows


def _metric_errors(measured, predicted, eligible: bool) -> dict:
    error = (
        float(predicted) - float(measured)
        if eligible and measured is not None and predicted is not None
        else None
    )
    relative = error / float(measured) if error is not None and measured != 0 else None
    return {
        "signed_error": error,
        "absolute_error": abs(error) if error is not None else None,
        "relative_error": relative,
        "absolute_relative_error": abs(relative) if relative is not None else None,
    }


def _pairwise_ordering(values: list[tuple[float, float]]) -> dict:
    counts = {
        "concordant": 0, "discordant": 0, "both_tied": 0,
        "measured_only_tied": 0, "predicted_only_tied": 0,
    }
    for (measured_a, predicted_a), (measured_b, predicted_b) in combinations(values, 2):
        measured_sign = (measured_a > measured_b) - (measured_a < measured_b)
        predicted_sign = (predicted_a > predicted_b) - (predicted_a < predicted_b)
        if measured_sign == 0 and predicted_sign == 0:
            category = "both_tied"
        elif measured_sign == 0:
            category = "measured_only_tied"
        elif predicted_sign == 0:
            category = "predicted_only_tied"
        else:
            category = "concordant" if measured_sign == predicted_sign else "discordant"
        counts[category] += 1
    total = sum(counts.values())
    strict = counts["concordant"] + counts["discordant"]
    return {
        "pairs": total,
        **counts,
        "strict_order_agreement": counts["concordant"] / strict if strict else None,
        "exact_sign_agreement": (
            (counts["concordant"] + counts["both_tied"]) / total if total else None
        ),
        "tie_definition": "exact_numeric_equality",
    }


def compare_reports(
    measured: dict, predicted: dict, *, evidence_kind: str = "unclassified"
) -> dict:
    """Return paired errors after checking the shared experimental contract."""
    for name, document in (("measured", measured), ("predicted", predicted)):
        missing = sorted(set(CONTRACT_FIELDS) - document["comparison_contract"].keys())
        if missing:
            raise ValueError(f"{name} comparison_contract missing fields: {', '.join(missing)}")
    if measured["comparison_contract"] != predicted["comparison_contract"]:
        raise ValueError("comparison_contract mismatch between measured and predicted")

    measured_by_id = _index_workflows(measured)
    predicted_by_id = _index_workflows(predicted)
    workflows = []
    for workflow_id in sorted(measured_by_id.keys() | predicted_by_id.keys()):
        actual = measured_by_id.get(workflow_id)
        estimate = predicted_by_id.get(workflow_id)
        actual_failed = actual is not None and bool(actual.get("errors"))
        estimate_failed = estimate is not None and bool(estimate.get("errors"))
        actual_incomplete = actual is not None and actual.get("status") == "incomplete"
        estimate_incomplete = estimate is not None and estimate.get("status") == "incomplete"
        if actual is None:
            status = "missing_measured"
        elif estimate is None:
            status = "missing_predicted"
        elif actual_failed and estimate_failed:
            status = "failed_both"
        elif actual_failed:
            status = "failed_measured"
        elif estimate_failed:
            status = "failed_predicted"
        elif actual_incomplete and estimate_incomplete:
            status = "incomplete_both"
        elif actual_incomplete:
            status = "incomplete_measured"
        elif estimate_incomplete:
            status = "incomplete_predicted"
        else:
            status = "paired_successful"
        workflows.append({
            "workflow_id": workflow_id,
            "status": status,
            "measured": actual,
            "predicted": estimate,
            "metrics": {
                metric: _metric_errors(
                    actual.get(metric) if actual is not None else None,
                    estimate.get(metric) if estimate is not None else None,
                    status == "paired_successful",
                )
                for metric in METRICS
            },
        })

    metrics = {}
    for metric in METRICS:
        paired = [
            row for row in workflows
            if row["metrics"][metric]["signed_error"] is not None
        ]
        absolute_errors = [row["metrics"][metric]["absolute_error"] for row in paired]
        relative_errors = [
            row["metrics"][metric]["absolute_relative_error"] for row in paired
            if row["metrics"][metric]["absolute_relative_error"] is not None
        ]
        metrics[metric] = {
            "paired_value_count": len(paired),
            "relative_error_count": len(relative_errors),
            "mae": mean(absolute_errors) if absolute_errors else None,
            "mape_percent": mean(relative_errors) * 100 if relative_errors else None,
            "pairwise_ordering": _pairwise_ordering([
                (float(row["measured"][metric]), float(row["predicted"][metric]))
                for row in paired
            ]),
        }
    return {
        "evidence_kind": evidence_kind,
        "comparison_contract": measured["comparison_contract"],
        "measured_implementation_revision": measured["implementation_revision"],
        "predicted_implementation_revision": predicted["implementation_revision"],
        "error_definition": "predicted - measured; relative_error divides by measured",
        "statistical_scope": (
            "descriptive workflow records; no independent-sample or uncertainty claim; "
            "records may repeat the same underlying trajectory"
        ),
        "workflow_counts": {
            "measured": len(measured_by_id),
            "predicted": len(predicted_by_id),
            "union": len(workflows),
            "paired_successful": sum(row["status"] == "paired_successful" for row in workflows),
            "missing_measured": len(predicted_by_id.keys() - measured_by_id.keys()),
            "missing_predicted": len(measured_by_id.keys() - predicted_by_id.keys()),
            "failed_measured": sum(bool(row.get("errors")) for row in measured_by_id.values()),
            "failed_predicted": sum(bool(row.get("errors")) for row in predicted_by_id.values()),
            "incomplete_measured": sum(row.get("status") == "incomplete" for row in measured_by_id.values()),
            "incomplete_predicted": sum(row.get("status") == "incomplete" for row in predicted_by_id.values()),
        },
        "workflows": workflows,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence-kind", default="unclassified",
        choices=("unclassified", "synthetic_tooling_check", "measured_comparison"),
    )
    args = parser.parse_args()
    measured = json.loads(args.measured.read_text(encoding="utf-8-sig"))
    predicted = json.loads(args.predicted.read_text(encoding="utf-8-sig"))
    report = compare_reports(measured, predicted, evidence_kind=args.evidence_kind)
    payload = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
