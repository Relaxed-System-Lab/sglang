"""Synthetic tooling checks; these fixtures are not H20 calibration evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = (
    _ROOT / "scripts" / "hexgent_real_trace_collectors" / "compare_simulator_real.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("hexgent_simulator_real", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _document(rows, revision="synthetic-real"):
    return {
        "comparison_contract": {
            "trace_sha256": "synthetic-trace-hash",
            "profile_sha256": "synthetic-isolated-profile-hash",
            "model": "synthetic-model", "model_revision": "synthetic-weights",
            "tp": 1, "weight_dtype": "bf16", "kv_dtype": "bf16",
            "hardware": "synthetic-hardware",
            "topology": {"prefill": 1, "decode": 1},
            "cache_policy": "disabled", "tool_policy": "fixed_no_tools",
            "arrival": {"kind": "fixed", "seed": 42},
            "scheduler_flags": {"pairwise": False},
            "observation_horizon": "all_workflows_terminal",
        },
        "implementation_revision": revision,
        "workflows": rows,
    }


def _row(workflow_id, latency_ms, slo_ratio, errors=0):
    return {
        "workflow_id": workflow_id, "latency_ms": latency_ms,
        "slo_ratio": slo_ratio, "errors": errors,
    }


def test_paired_errors_preserve_sign_and_distinct_implementation_revisions():
    module = _load_module()
    measured = _document([_row("a", 100, 1), _row("b", 200, 2)])
    predicted = _document([_row("a", 120, 1.2), _row("b", 150, 1.5)], "synthetic-sim")

    report = module.compare_reports(
        measured, predicted, evidence_kind="synthetic_tooling_check"
    )
    rows = {row["workflow_id"]: row for row in report["workflows"]}
    assert rows["a"]["metrics"]["latency_ms"] == {
        "signed_error": 20, "absolute_error": 20,
        "relative_error": 0.2, "absolute_relative_error": 0.2,
    }
    assert rows["b"]["metrics"]["latency_ms"]["signed_error"] == -50
    assert rows["b"]["metrics"]["latency_ms"]["relative_error"] == -0.25
    assert report["metrics"]["latency_ms"]["mae"] == 35
    assert report["metrics"]["latency_ms"]["mape_percent"] == 22.5
    assert report["metrics"]["slo_ratio"]["mae"] == pytest.approx(0.35)
    assert report["measured_implementation_revision"] == "synthetic-real"
    assert report["predicted_implementation_revision"] == "synthetic-sim"
    assert report["evidence_kind"] == "synthetic_tooling_check"
    assert "no independent-sample" in report["statistical_scope"]


def test_missing_failed_and_zero_reference_records_remain_explicit():
    module = _load_module()
    measured = _document([
        _row("paired", 100, 1), _row("failed", 200, 2, errors=1),
        _row("only_real", 30, 0.3), _row("unknown", None, None),
        _row("zero", 0, 0),
    ])
    predicted = _document([
        _row("paired", 110, 1.1), _row("failed", 230, 2.3),
        _row("only_sim", 50, 0.5), _row("unknown", 20, 1),
        _row("zero", 10, 0.1),
    ])

    report = module.compare_reports(measured, predicted)
    rows = {row["workflow_id"]: row for row in report["workflows"]}
    assert len(rows) == 6
    assert rows["only_real"]["status"] == "missing_predicted"
    assert rows["only_sim"]["status"] == "missing_measured"
    assert rows["failed"]["status"] == "failed_measured"
    assert rows["failed"]["measured"]["errors"] == 1
    assert rows["failed"]["metrics"]["latency_ms"]["signed_error"] is None
    assert rows["unknown"]["metrics"]["latency_ms"]["signed_error"] is None
    assert rows["zero"]["metrics"]["latency_ms"]["absolute_error"] == 10
    assert rows["zero"]["metrics"]["latency_ms"]["relative_error"] is None
    assert report["metrics"]["latency_ms"]["paired_value_count"] == 2
    assert report["metrics"]["latency_ms"]["relative_error_count"] == 1
    assert report["metrics"]["latency_ms"]["mape_percent"] == 10
    assert report["workflow_counts"]["missing_measured"] == 1
    assert report["workflow_counts"]["missing_predicted"] == 1
    assert report["workflow_counts"]["failed_measured"] == 1


@pytest.mark.parametrize(
    ("measured_values", "predicted_values", "category", "strict", "exact"),
    [
        ([1, 2], [1, 2], "concordant", 1, 1),
        ([1, 2], [2, 1], "discordant", 0, 0),
        ([1, 1], [1, 1], "both_tied", None, 1),
        ([1, 1], [1, 2], "measured_only_tied", None, 0),
        ([1, 2], [1, 1], "predicted_only_tied", None, 0),
    ],
)
def test_pairwise_ordering_distinguishes_exact_ties(
    measured_values, predicted_values, category, strict, exact
):
    module = _load_module()
    measured = _document([_row(str(i), value, value) for i, value in enumerate(measured_values)])
    predicted = _document([_row(str(i), value, value) for i, value in enumerate(predicted_values)])

    report = module.compare_reports(measured, predicted)
    ordering = report["metrics"]["latency_ms"]["pairwise_ordering"]
    assert ordering["pairs"] == 1
    assert ordering[category] == 1
    assert ordering["strict_order_agreement"] == strict
    assert ordering["exact_sign_agreement"] == exact
    assert ordering["tie_definition"] == "exact_numeric_equality"


def test_comparison_contract_mismatch_is_rejected():
    module = _load_module()
    measured = _document([])
    predicted = copy.deepcopy(measured)
    predicted["comparison_contract"]["profile_sha256"] = "different-synthetic-profile"

    with pytest.raises(ValueError, match="comparison_contract mismatch"):
        module.compare_reports(measured, predicted)


def test_comparison_contract_requires_all_declared_fields():
    module = _load_module()
    measured = _document([])
    predicted = _document([])
    del measured["comparison_contract"]["hardware"]
    del predicted["comparison_contract"]["hardware"]

    with pytest.raises(ValueError, match="hardware"):
        module.compare_reports(measured, predicted)


def test_duplicate_workflow_ids_are_rejected_instead_of_overwritten():
    module = _load_module()
    measured = _document([_row("repeat", 1, 1), _row("repeat", 2, 2)])
    predicted = _document([_row("repeat", 1, 1)])

    with pytest.raises(ValueError, match="duplicate workflow_id"):
        module.compare_reports(measured, predicted)


def test_empty_comparison_does_not_fill_missing_errors_with_zero():
    module = _load_module()
    report = module.compare_reports(_document([]), _document([]))

    assert report["metrics"]["latency_ms"]["mae"] is None
    assert report["metrics"]["latency_ms"]["mape_percent"] is None
    assert report["metrics"]["latency_ms"]["pairwise_ordering"]["exact_sign_agreement"] is None


def test_incomplete_workflow_is_preserved_without_becoming_a_successful_pair():
    module = _load_module()
    measured_row = _row("incomplete", 100, None)
    measured_row["status"] = "incomplete"
    measured_row["missing_call_ids"] = ["child"]
    measured = _document([measured_row])
    predicted = _document([_row("incomplete", 200, 2)])

    report = module.compare_reports(measured, predicted)

    row = report["workflows"][0]
    assert row["status"] == "incomplete_measured"
    assert row["measured"]["missing_call_ids"] == ["child"]
    assert row["metrics"]["latency_ms"]["signed_error"] is None
    assert report["workflow_counts"]["paired_successful"] == 0
    assert report["workflow_counts"]["incomplete_measured"] == 1
    assert report["metrics"]["latency_ms"]["mae"] is None


def test_cli_marks_synthetic_tooling_fixture(tmp_path):
    measured_path = tmp_path / "measured.synthetic.json"
    predicted_path = tmp_path / "predicted.synthetic.json"
    output_path = tmp_path / "comparison.synthetic.json"
    measured_path.write_text(json.dumps(_document([_row("wf", 100, 1)])), encoding="utf-8")
    predicted_path.write_text(json.dumps(_document([_row("wf", 120, 1.2)])), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable, str(_MODULE_PATH), "--measured", str(measured_path),
            "--predicted", str(predicted_path), "--output", str(output_path),
            "--evidence-kind", "synthetic_tooling_check",
        ],
        capture_output=True, text=True, cwd=_ROOT,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["evidence_kind"] == "synthetic_tooling_check"
    assert report["metrics"]["latency_ms"]["mae"] == 20
