"""Synthetic CLI tests for artifact-only workflow-reference summary correction."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/hexgent_real_trace_collectors/summarize_scheduling_replay.py"
MODEL = "synthetic-served-model"
REVISION = "a" * 40
RAW_SHA = "1" * 64
RUNTIME_SHA = "2" * 64
CONTRACT = {"model": MODEL, "implementation_revision": REVISION, "runtime_evidence_sha256": RUNTIME_SHA}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path):
    parents = {"root": [], "left": ["root"], "right": ["root"], "join": ["left", "right"]}
    trace = [{"workflow_id": "wf-a", "calls": [
        {"call_id": name, "parents": upstream, "input_len": 10, "output_len": 1}
        for name, upstream in parents.items()
    ]}]
    starts = [1000, 1010, 1010, 1040]
    ends = [1010, 1025, 1040, 1060]
    replay = [{
        "workflow_id": "wf-a", "call_id": name, "parents": upstream,
        "request_started_at_ms": start, "response_completed_at_ms": end,
        "workflow_arrival_ms": 1000, "total_latency_ms": end - start,
        "exclusive_profile_latency_ms": 10, "input_tokens": 10, "error": None,
        "router_mode": "hexgent", "schedule_revision": 1,
    } for (name, upstream), start, end in zip(parents.items(), starts, ends)]
    exclusive = [{"workflow_id": "wf-a", "call_id": name,
                  "total_latency_ms": 10, "error": None} for name in parents]
    reference = [{
        "workflow_id": "wf-a", "reference_kind": "isolated_workflow_replay",
        "latency_ms": 55, "status": "successful", "expected_calls": 4,
        "completed_calls": 4, "errors": 0, "usage_verified_calls": 4,
        "model": MODEL, "implementation_revision": REVISION,
        "raw_replay_sha256": RAW_SHA, "runtime_evidence_sha256": RUNTIME_SHA,
    }]
    return {"root": tmp_path, "trace": trace, "replay": replay,
            "exclusive": exclusive, "reference": reference, "contract": copy.deepcopy(CONTRACT)}


def run_summary(inputs, *, mode="workflow", changes=()):
    root = inputs["root"]
    for key in ("trace", "replay", "exclusive", "reference"):
        write_jsonl(root / (key + ".jsonl"), inputs[key])
    (root / "contract.json").write_text(json.dumps(inputs["contract"]), encoding="utf-8")
    output = root / "summary.json"
    command = [sys.executable, str(SCRIPT), "--replay", str(root / "replay.jsonl"),
               "--trace", str(root / "trace.jsonl"), "--output", str(output)]
    if mode == "workflow":
        command += ["--workflow-reference", str(root / "reference.jsonl"),
                    "--workflow-reference-contract", str(root / "contract.json")]
    elif mode == "exclusive":
        command += ["--exclusive-profile", str(root / "exclusive.jsonl")]
    for operation, value in changes:
        if operation == "remove":
            index = command.index(value)
            del command[index:index + 2]
        elif operation == "append":
            command += value
    result = subprocess.run(command, capture_output=True, text=True)
    return result, json.loads(output.read_text()) if result.returncode == 0 else None


def rejected_reference(inputs):
    result, _ = run_summary(inputs)
    assert result.returncode != 0
    assert "workflow reference" in result.stderr.lower()


def test_fanout_reference_uses_standalone_elapsed_instead_of_cp_sum(inputs):
    result, output = run_summary(inputs)
    assert result.returncode == 0, result.stderr
    row = output["workflows"][0]
    assert row["latency_ms"] == 60
    assert row["base_slo_ms"] == row["profile_slo_ms"] == 55
    assert row["slo_ratio"] == pytest.approx(60 / 55)
    assert row["reference_kind"] == "isolated_workflow_replay"
    assert row["reference_status"] == "successful"
    assert row["reference_raw_row"] == inputs["reference"][0]
    assert output["slo_definition"] == "isolated_workflow_replay_on_finite_1p1d"
    assert output["workflow_reference_path"] == str(inputs["root"] / "reference.jsonl")
    assert output["workflow_reference_sha256"] == hashlib.sha256((inputs["root"] / "reference.jsonl").read_bytes()).hexdigest()
    assert output["workflow_reference_contract"] == CONTRACT


@pytest.mark.parametrize("status", ["failed", "incomplete"])
def test_failed_reference_is_preserved_as_unknown_without_cp_fallback(inputs, status):
    row = inputs["reference"][0]
    row.update(status=status, completed_calls=1, usage_verified_calls=1, errors=1, latency_ms=999)
    result, output = run_summary(inputs)
    assert result.returncode == 0, result.stderr
    actual = output["workflows"][0]
    assert actual["reference_status"] == status
    assert actual["reference_raw_row"] == row
    assert actual["base_slo_ms"] is None and actual["slo_ratio"] is None
    assert output["workflow_counts"]["successful_missing_reference_slo"] == 1
    assert output["fixed_scale_slo_violations"][0]["unknown_workflows"] == 1


def test_missing_reference_is_unknown_without_embedded_cp_fallback(inputs):
    inputs["reference"] = []
    result, output = run_summary(inputs)
    assert result.returncode == 0, result.stderr
    row = output["workflows"][0]
    assert row["reference_status"] == "missing" and row["reference_raw_row"] is None
    assert row["base_slo_ms"] is None and row["profile_slo_ms"] is None and row["slo_ratio"] is None


def test_whole_table_can_cover_a_selected_trace_subset(inputs):
    extra = dict(inputs["reference"][0], workflow_id="wf-b", expected_calls=2,
                 completed_calls=2, usage_verified_calls=2)
    inputs["reference"].append(extra)
    result, output = run_summary(inputs)
    assert result.returncode == 0, result.stderr
    assert len(output["workflows"]) == 1
    assert output["workflows"][0]["base_slo_ms"] == 55


@pytest.mark.parametrize("changes", [
    {"expected_calls": 3, "completed_calls": 3, "usage_verified_calls": 3},
    {"completed_calls": 3}, {"usage_verified_calls": 3}, {"errors": 1},
    {"expected_calls": 0, "completed_calls": 0, "usage_verified_calls": 0},
])
def test_successful_reference_counts_must_match_the_expected_trace(inputs, changes):
    inputs["reference"][0].update(changes)
    rejected_reference(inputs)


def test_unselected_successful_rows_must_have_self_consistent_counts(inputs):
    inputs["reference"].append(dict(inputs["reference"][0], workflow_id="wf-b", completed_calls=3))
    rejected_reference(inputs)


@pytest.mark.parametrize("changes", [
    {"reference_kind": "critical_path_proxy"}, {"model": "another-model"},
    {"implementation_revision": "b" * 40}, {"runtime_evidence_sha256": "3" * 64},
    {"raw_replay_sha256": "not-a-sha256"}, {"runtime_evidence_sha256": "bad"},
])
def test_reference_metadata_must_match_the_frozen_contract(inputs, changes):
    inputs["reference"][0].update(changes)
    rejected_reference(inputs)


def test_failed_rows_still_require_correct_model_and_runtime_provenance(inputs):
    inputs["reference"][0].update(status="failed", errors=1, model="another-model", latency_ms=None)
    rejected_reference(inputs)


def test_duplicate_workflow_reference_is_rejected(inputs):
    inputs["reference"].append(copy.deepcopy(inputs["reference"][0]))
    rejected_reference(inputs)


@pytest.mark.parametrize("latency", [0, -1, float("nan"), float("inf")])
def test_successful_reference_denominator_must_be_positive_finite(inputs, latency):
    inputs["reference"][0]["latency_ms"] = latency
    rejected_reference(inputs)


def test_reference_modes_are_mutually_exclusive(inputs):
    result, _ = run_summary(inputs, changes=[("append", ["--exclusive-profile", str(inputs["root"] / "exclusive.jsonl")])])
    assert result.returncode != 0


@pytest.mark.parametrize("flag", ["--trace", "--workflow-reference-contract"])
def test_workflow_reference_requires_trace_and_contract(inputs, flag):
    result, _ = run_summary(inputs, changes=[("remove", flag)])
    assert result.returncode != 0


def test_contract_without_workflow_reference_is_rejected(inputs):
    result, _ = run_summary(inputs, changes=[("remove", "--workflow-reference")])
    assert result.returncode != 0


def test_failed_partial_pressure_replay_keeps_its_valid_workflow_denominator(inputs):
    inputs["replay"] = inputs["replay"][:3]
    inputs["replay"][1]["error"] = "synthetic failed call"
    result, output = run_summary(inputs)
    assert result.returncode == 0, result.stderr
    row = output["workflows"][0]
    assert row["status"] == "failed" and row["errors"] == 1
    assert row["missing_call_ids"] == ["join"]
    assert row["base_slo_ms"] == 55 and row["reference_status"] == "successful"
    assert row["slo_ratio"] is None


def test_entirely_missing_pressure_workflow_stays_incomplete(inputs):
    inputs["replay"] = []
    result, output = run_summary(inputs)
    assert result.returncode == 0, result.stderr
    row = output["workflows"][0]
    assert row["status"] == "incomplete" and row["calls"] == 0
    assert row["latency_ms"] is None and row["slo_ratio"] is None
    assert row["base_slo_ms"] == 55


@pytest.mark.parametrize("mode", ["exclusive", "embedded"])
def test_old_critical_path_proxy_mode_keeps_its_numeric_behavior(inputs, mode):
    result, output = run_summary(inputs, mode=mode)
    assert result.returncode == 0, result.stderr
    row = output["workflows"][0]
    assert row["profile_slo_ms"] == 30 and row["slo_ratio"] == 2
    assert output["slo_definition"] == "dag_critical_path_of_exclusive_call_latency_proxy"
