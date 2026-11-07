import importlib.util
import json
from pathlib import Path
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = (
    _ROOT / "scripts" / "hexgent_real_trace_collectors" / "summarize_scheduling_replay.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "hexgent_scheduling_summary", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_read_jsonl_accepts_utf8_bom(tmp_path):
    module = _load_module()
    profile = tmp_path / "profile.jsonl"
    profile.write_text(
        '\ufeff{"workflow_id": "wf", "call_id": "call", "total_latency_ms": 1.0}\n',
        encoding="utf-8",
    )

    assert module.read_jsonl(profile) == [
        {"workflow_id": "wf", "call_id": "call", "total_latency_ms": 1.0}
    ]


def test_profile_slo_uses_dag_critical_path_instead_of_sum():
    module = _load_module()
    calls = [
        {
            "call_id": "root",
            "parents": [],
            "exclusive_profile_latency_ms": 10.0,
        },
        {
            "call_id": "short",
            "parents": ["root"],
            "exclusive_profile_latency_ms": 5.0,
        },
        {
            "call_id": "long",
            "parents": ["root"],
            "exclusive_profile_latency_ms": 20.0,
        },
        {
            "call_id": "join",
            "parents": ["short", "long"],
            "exclusive_profile_latency_ms": 7.0,
        },
    ]

    assert module.critical_path_profile_latency_ms(calls) == pytest.approx(37.0)


def test_profile_override_replaces_old_model_latency():
    module = _load_module()
    calls = [
        {
            "workflow_id": "wf",
            "call_id": "first",
            "parents": [],
            "exclusive_profile_latency_ms": 100.0,
        },
        {
            "workflow_id": "wf",
            "call_id": "second",
            "parents": ["first"],
            "exclusive_profile_latency_ms": 200.0,
        },
    ]
    override = {("wf", "first"): 11.0, ("wf", "second"): 13.0}

    assert module.critical_path_profile_latency_ms(
        calls, exclusive_profile=override
    ) == pytest.approx(24.0)


def test_profile_slo_rejects_missing_parent():
    module = _load_module()
    calls = [
        {
            "call_id": "child",
            "parents": ["missing"],
            "exclusive_profile_latency_ms": 10.0,
        }
    ]

    with pytest.raises(ValueError, match="missing parent"):
        module.critical_path_profile_latency_ms(calls)


def test_planner_plan_execution_detects_fallback_only_hexgent_run():
    module = _load_module()
    rows = [
        {"router_mode": "hexgent", "schedule_revision": 0, "error": None},
        {"router_mode": "hexgent", "schedule_revision": 0, "error": None},
        {"router_mode": "hexgent", "schedule_revision": None, "error": "HTTP 400"},
    ]

    assert module.planner_plan_execution(rows) == {
        "applicable": True,
        "successful_hexgent_calls": 2,
        "planner_revision_calls": 0,
        "fallback_revision_calls": 2,
        "missing_revision_calls": 0,
        "planner_revision_coverage": 0.0,
        "planner_effect_observed": False,
    }


def test_planner_plan_execution_reports_nonzero_revision_coverage():
    module = _load_module()
    rows = [
        {"router_mode": "hexgent", "schedule_revision": 0, "error": None},
        {"router_mode": "hexgent", "schedule_revision": 7, "error": None},
        {"router_mode": "hexgent", "schedule_revision": None, "error": None},
    ]

    result = module.planner_plan_execution(rows)

    assert result["successful_hexgent_calls"] == 3
    assert result["planner_revision_calls"] == 1
    assert result["fallback_revision_calls"] == 1
    assert result["missing_revision_calls"] == 1
    assert result["planner_revision_coverage"] == pytest.approx(1 / 3)
    assert result["planner_effect_observed"] is True


@pytest.mark.parametrize(
    ("values", "expected_p99"),
    [(list(range(1, 101)), 99), ([], None)],
)
def test_distribution_reports_nearest_rank_p99(values, expected_p99):
    module = _load_module()

    assert module.distribution(values)["p99"] == expected_p99


def _run_summary(rows, tmp_path, monkeypatch, *, trace=None, profile=None):
    module = _load_module()
    replay_path = tmp_path / "calls.jsonl"
    replay_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output_path = tmp_path / "summary.json"
    argv = [
        "summarize_scheduling_replay.py",
        "--replay", str(replay_path),
        "--output", str(output_path),
        "--kv-bytes-per-input-token", "32",
    ]
    if trace is not None:
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8"
        )
        argv.extend(["--trace", str(trace_path)])
    if profile is not None:
        profile_path = tmp_path / "profile.jsonl"
        profile_path.write_text(
            "".join(json.dumps(row) + "\n" for row in profile), encoding="utf-8"
        )
        argv.extend(["--exclusive-profile", str(profile_path)])
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_summary_preserves_failed_workflows_and_excludes_failed_ratios(
    tmp_path, monkeypatch
):
    rows = [
        {
            "workflow_id": "success", "call_id": "call", "parents": [],
            "request_started_at_ms": 1000, "response_completed_at_ms": 1100,
            "total_latency_ms": 100, "exclusive_profile_latency_ms": 50,
            "input_tokens": 10, "error": None,
        },
        {
            "workflow_id": "failure", "call_id": "call", "parents": [],
            "request_started_at_ms": 2000, "response_completed_at_ms": 2200,
            "total_latency_ms": 200, "exclusive_profile_latency_ms": 50,
            "input_tokens": 10, "error": "HTTP 500",
        },
        {
            "workflow_id": "unprofiled", "call_id": "call", "parents": [],
            "request_started_at_ms": 3000, "response_completed_at_ms": 3500,
            "total_latency_ms": 500, "input_tokens": 10, "error": None,
        },
    ]

    summary = _run_summary(rows, tmp_path, monkeypatch)
    workflows = {row["workflow_id"]: row for row in summary["workflows"]}
    assert summary["workflow_counts"] == {
        "total": 3, "observed": 3, "successful": 2, "failed": 1,
        "incomplete": 0, "entirely_missing": 0,
        "successful_with_reference_slo": 1,
        "successful_missing_reference_slo": 1,
    }
    assert summary["workflow_latency_origin"] == "earliest_request_started_at_ms"
    assert workflows["success"]["earliest_request_started_at_ms"] == 1000
    assert workflows["success"]["response_completed_at_ms"] == 1100
    assert workflows["success"]["slo_ratio"] == 2
    assert workflows["failure"]["errors"] == 1
    assert workflows["failure"]["status"] == "failed"
    assert workflows["failure"]["latency_ms"] == 200
    assert workflows["failure"]["slo_ratio"] is None
    assert workflows["unprofiled"]["profile_slo_ms"] is None
    assert workflows["unprofiled"]["slo_ratio"] is None
    assert summary["workflow_slo_ratio"]["count"] == 1
    assert summary["workflow_slo_ratio"]["mean"] == 2
    assert summary["workflow_latency_ms"]["count"] == 2
    assert summary["workflow_latency_ms"]["max"] == 500


def test_summary_labels_token_based_kv_bytes_as_estimated(tmp_path, monkeypatch):
    rows = [
        {
            "workflow_id": "wf", "call_id": "call", "parents": [],
            "request_started_at_ms": 1000, "response_completed_at_ms": 1100,
            "total_latency_ms": 100, "input_tokens": 10, "error": None,
        }
    ]

    summary = _run_summary(rows, tmp_path, monkeypatch)
    assert summary["kv_transfer"]["measurement_kind"] == "input_token_byte_estimate"
    assert summary["kv_transfer"]["total_bytes"] == 320
    assert summary["kv_transfer"]["actual_transfer_bytes"] is None


def test_summary_measures_from_recorded_workflow_arrival(tmp_path, monkeypatch):
    rows = [{
        "workflow_id": "wf", "call_id": "root", "parents": [],
        "workflow_arrival_ms": 900,
        "request_started_at_ms": 1000, "response_completed_at_ms": 1100,
        "total_latency_ms": 100, "exclusive_profile_latency_ms": 50,
        "error": None,
    }]

    summary = _run_summary(rows, tmp_path, monkeypatch)

    assert summary["workflow_latency_origin"] == "workflow_arrival_ms"
    assert summary["workflows"][0]["workflow_arrival_ms"] == 900
    assert summary["workflows"][0]["latency_ms"] == 200
    assert summary["workflows"][0]["slo_ratio"] == 4


def test_fixed_scale_violations_preserve_failures_and_unknown_slo():
    module = _load_module()
    rows = [
        {"errors": 0, "slo_ratio": 2},
        {"errors": 1, "slo_ratio": None},
        {"errors": 0, "slo_ratio": None},
    ]

    result = module.fixed_scale_violations(rows, [1, 2])

    assert result[0] == {
        "scale": 1, "total_workflows": 3, "profiled_successful_workflows": 1,
        "profiled_successful_violations": 1, "failed_workflows": 1,
        "unknown_successful_workflows": 1,
        "incomplete_workflows": 0, "unknown_workflows": 1,
        "violation_rate_lower_bound": 2 / 3, "violation_rate_upper_bound": 1,
    }
    assert result[1]["profiled_successful_violations"] == 0
    assert result[1]["violation_rate_lower_bound"] == 1 / 3
    assert result[1]["violation_rate_upper_bound"] == 2 / 3


def _observed_call(workflow_id, call_id, parents=(), *, start=1000, end=1100, reference=100):
    return {
        "workflow_id": workflow_id, "call_id": call_id, "parents": list(parents),
        "request_started_at_ms": start, "response_completed_at_ms": end,
        "total_latency_ms": end - start, "exclusive_profile_latency_ms": reference,
        "error": None,
    }


def test_trace_preserves_missing_workflows_and_children_without_using_profile_as_expected(
    tmp_path, monkeypatch
):
    trace = [
        {"workflow_id": "partial", "calls": [
            {"call_id": "root", "parents": []},
            {"call_id": "child", "parents": ["root"]},
        ]},
        {"workflow_id": "missing", "calls": [{"call_id": "only", "parents": []}]},
    ]
    profile = [
        {"workflow_id": workflow_id, "call_id": call_id, "total_latency_ms": 100}
        for workflow_id, call_id in [
            ("partial", "root"), ("partial", "child"), ("missing", "only"),
            ("library_workflow_not_in_this_run", "other"),
        ]
    ]

    summary = _run_summary(
        [_observed_call("partial", "root")], tmp_path, monkeypatch,
        trace=trace, profile=profile,
    )
    workflows = {row["workflow_id"]: row for row in summary["workflows"]}
    assert set(workflows) == {"partial", "missing"}
    assert summary["completeness_verified"] is True
    assert summary["workflow_counts"]["total"] == 2
    assert summary["workflow_counts"]["observed"] == 1
    assert summary["workflow_counts"]["successful"] == 0
    assert summary["workflow_counts"]["failed"] == 0
    assert summary["workflow_counts"]["incomplete"] == 2
    assert summary["workflow_counts"]["entirely_missing"] == 1
    assert summary["call_coverage"] == {
        "expected": 3, "observed": 1, "missing": 2, "observed_fraction": 1 / 3,
    }
    assert workflows["partial"]["status"] == "incomplete"
    assert workflows["partial"]["missing_call_ids"] == ["child"]
    assert workflows["partial"]["slo_ratio"] is None
    assert workflows["missing"]["status"] == "incomplete"
    assert workflows["missing"]["missing_call_ids"] == ["only"]
    assert workflows["missing"]["latency_ms"] is None
    assert workflows["missing"]["response_completed_at_ms"] is None
    assert summary["workflow_latency_ms"]["count"] == 0
    assert summary["workflow_slo_ratio"]["count"] == 0
    violations = summary["fixed_scale_slo_violations"][0]
    assert violations["incomplete_workflows"] == 2
    assert violations["unknown_workflows"] == 2
    assert violations["violation_rate_lower_bound"] == 0
    assert violations["violation_rate_upper_bound"] == 1


def test_partial_child_with_missing_parent_keeps_report_evidence(tmp_path, monkeypatch):
    trace = [{"workflow_id": "wf", "calls": [
        {"call_id": "root", "parents": []},
        {"call_id": "child", "parents": ["root"]},
    ]}]

    summary = _run_summary(
        [_observed_call("wf", "child", ["root"])], tmp_path, monkeypatch, trace=trace
    )
    workflow = summary["workflows"][0]
    assert workflow["status"] == "incomplete"
    assert workflow["missing_call_ids"] == ["root"]
    assert workflow["profile_slo_ms"] is None
    assert workflow["slo_ratio"] is None
    assert workflow["latency_ms"] == 100


def test_complete_fanout_trace_has_verified_coverage_and_critical_path_slo(
    tmp_path, monkeypatch
):
    rows = [
        _observed_call("wf", "root", start=1000, end=1010, reference=10),
        _observed_call("wf", "short", ["root"], start=1010, end=1015, reference=5),
        _observed_call("wf", "long", ["root"], start=1010, end=1030, reference=20),
        _observed_call("wf", "join", ["short", "long"], start=1030, end=1037, reference=7),
    ]
    trace = [{"workflow_id": "wf", "calls": [
        {"call_id": row["call_id"], "parents": row["parents"]} for row in rows
    ]}]

    summary = _run_summary(rows, tmp_path, monkeypatch, trace=trace)
    workflow = summary["workflows"][0]
    assert workflow["status"] == "successful"
    assert workflow["completeness_verified"] is True
    assert workflow["expected_calls"] == 4
    assert workflow["missing_call_ids"] == []
    assert workflow["call_coverage"] == 1
    assert workflow["profile_slo_ms"] == 37
    assert workflow["slo_ratio"] == 1
    assert summary["workflow_counts"]["successful"] == 1
    assert summary["workflow_slo_ratio"]["count"] == 1


def test_no_trace_preserves_observed_only_legacy_semantics(tmp_path, monkeypatch):
    profile = [
        {"workflow_id": "wf", "call_id": "root", "total_latency_ms": 100},
        {"workflow_id": "wf", "call_id": "missing_child", "total_latency_ms": 100},
    ]

    summary = _run_summary(
        [_observed_call("wf", "root")], tmp_path, monkeypatch, profile=profile
    )
    workflow = summary["workflows"][0]
    assert summary["completeness_verified"] is False
    assert workflow["completeness_verified"] is False
    assert workflow["expected_calls"] is None
    assert workflow["missing_call_ids"] is None
    assert workflow["status"] == "successful"
    assert workflow["slo_ratio"] == 1
    assert summary["call_coverage"]["expected"] is None
    assert summary["call_coverage"]["observed_fraction"] is None


@pytest.mark.parametrize(
    "row", [_observed_call("unexpected_workflow", "root"), _observed_call("wf", "unexpected_call")]
)
def test_trace_rejects_unexpected_workflow_or_call_ids(row, tmp_path, monkeypatch):
    trace = [{"workflow_id": "wf", "calls": [{"call_id": "root", "parents": []}]}]

    with pytest.raises(ValueError, match="unexpected"):
        _run_summary([row], tmp_path, monkeypatch, trace=trace)


def test_fixed_scale_incomplete_is_unknown_and_observed_failure_is_violation():
    module = _load_module()
    workflows = [
        {"errors": 0, "slo_ratio": None, "status": "incomplete"},
        {"errors": 1, "slo_ratio": None, "status": "failed"},
        {"errors": 0, "slo_ratio": 1, "status": "successful"},
    ]

    result = module.fixed_scale_violations(workflows, [1])[0]
    assert result["unknown_successful_workflows"] == 0
    assert result["incomplete_workflows"] == 1
    assert result["unknown_workflows"] == 1
    assert result["failed_workflows"] == 1
    assert result["violation_rate_lower_bound"] == 1 / 3
    assert result["violation_rate_upper_bound"] == 2 / 3
