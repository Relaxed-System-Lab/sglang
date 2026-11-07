import importlib.util
import json
from pathlib import Path

import pytest


def load():
    path = Path(__file__).with_name("run_paired_pilot.py")
    spec = importlib.util.spec_from_file_location("h20_paired_pilot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_single_counterbalanced_block_has_exact_eight_runs():
    m = load()
    assert m.run_order() == [
        ("sharegpt", "low", "native"), ("sharegpt", "low", "ours"),
        ("sharegpt", "burst", "ours"), ("sharegpt", "burst", "native"),
        ("hotpotqa", "low", "ours"), ("hotpotqa", "low", "native"),
        ("hotpotqa", "burst", "native"), ("hotpotqa", "burst", "ours"),
    ]


def test_conditioned_poisson_is_frozen_and_burst_is_zero():
    m = load()
    a = m.arrival_offsets(4, rate=0.1, seed=20261107, burst=False)
    assert a[0] == 0
    assert a == m.arrival_offsets(4, rate=0.1, seed=20261107, burst=False)
    assert all(x < y for x, y in zip(a, a[1:]))
    assert m.arrival_offsets(4, rate=0.1, seed=20261107, burst=True) == [0.0] * 4


def test_rate_uses_only_new_calibration_burst_makespan():
    m = load()
    assert m.low_rate({"workflow_count": 2, "makespan_ms": 20000}) == pytest.approx(0.05)
    for record in [{"workflow_count": 1, "makespan_ms": 20},
                   {"workflow_count": 2, "makespan_ms": None},
                   {"workflow_count": 2, "makespan_ms": 0}]:
        with pytest.raises(ValueError):
            m.low_rate(record)


def test_planned_trace_changes_only_explicit_arrivals():
    m = load()
    original = [
        {"workflow_id": "b", "arrival_offset": None, "calls": [{"call_id": "b0", "parents": [],
         "input_len": 39, "output_len": 8, "request": {"messages": [{"role": "user", "content": "x"}]},
         "planner_hints": {"model_config": {"tp_size": 2}}}]},
        {"workflow_id": "a", "calls": [{"call_id": "a0", "parents": [], "input_len": 10,
         "output_len": 8, "request": {"messages": [{"role": "user", "content": "y"}]}}]},
    ]
    frozen = m.freeze_arrivals(original, [0.0, 2.0])
    assert [w["workflow_id"] for w in frozen] == ["a", "b"]
    assert [w["arrival_offset"] for w in frozen] == [0.0, 2.0]
    for w in frozen:
        source = next(v for v in original if v["workflow_id"] == w["workflow_id"])
        assert {k: v for k, v in w.items() if k != "arrival_offset"} == {
            k: v for k, v in source.items() if k != "arrival_offset"}
    assert original[0]["arrival_offset"] is None


def test_actual_snapshot_limits_and_all_queues_are_checked():
    m = load()
    limits = {"max_total_num_tokens": 73197, "max_running_requests": 16,
              "prefill_max_requests": 4, "max_prefill_tokens": 16384}
    value = {"snapshots": [{"limits": limits, "running": [],
                           "queues": {"bootstrap": [], "waiting": [], "inflight": []}}]}
    assert m.snapshot_is_empty(value, limits)
    value["snapshots"][0]["queues"]["inflight"] = [{"rid": "busy"}]
    assert not m.snapshot_is_empty(value, limits)
    with pytest.raises(ValueError):
        m.snapshot_is_empty({"snapshots": []}, limits)
    with pytest.raises(ValueError):
        m.snapshot_is_empty(value, dict(limits, max_total_num_tokens=1))
    value["snapshots"][0]["queues"]["inflight"] = None
    with pytest.raises(ValueError):
        m.snapshot_is_empty(value, limits)


def test_missing_workflow_gate_is_retained_and_invalid():
    m = load()
    gates = {"a": {"successful": True}, "b": {"successful": False, "errors": ["missing"]}}
    assert m.run_gate(gates, expected_calls=2, rows=[{"workflow_id": "a", "call_id": "a0"}])["valid"] is False
    assert m.run_gate(gates, expected_calls=2, rows=[{"workflow_id": "a", "call_id": "a0"}])["workflow_gates"] == gates


def test_byte_slices_preserve_exact_raw_and_reject_truncation(tmp_path):
    m = load()
    source, target = tmp_path / "worker.log", tmp_path / "slice.log"
    source.write_bytes(b"warmup\nformal\nretry\n")
    m.copy_slice(source, target, 7, 20)
    assert target.read_bytes() == b"formal\nretry\n"
    with pytest.raises(ValueError):
        m.copy_slice(source, tmp_path / "bad.log", 7, 100)


def test_block_completion_requires_all_eight_valid_measurements():
    m = load()
    assert m.block_completion([{"valid_measurement": True}] * 8) == {
        "all_runs_attempted": True, "complete_balanced_block": True}
    result = m.block_completion([{"valid_measurement": True}] * 7 + [{"valid_measurement": False}])
    assert result == {"all_runs_attempted": True, "complete_balanced_block": False}


def test_cpu_delta_uses_actual_nonoverlapping_cgroup_counters():
    m = load()
    assert m.cgroup_cpu_delta({"cpu.stat": "usage_usec 100\nuser_usec 80\n"},
                              {"cpu.stat": "usage_usec 250\nuser_usec 180\n"}) == {
                                  "usage_usec": 150, "user_usec": 100}
    with pytest.raises(ValueError):
        m.cgroup_cpu_delta({"cpu.stat": "usage_usec 100\n"}, {"cpu.stat": "usage_usec 90\n"})


def test_incomplete_worker_capture_makes_rates_unknown_but_preserves_counts():
    m = load()
    status = {"processes": [
        {"name": "prefill", "truncated": False, "write_error": None},
        {"name": "decode", "truncated": True, "write_error": None}]}
    assert not m.worker_observation_complete(status)
    report = m.label_mutation_report(
        {"received_http_attempts": 3, "attempt_any_stale_revision_rate": 0.0,
         "rank_outcome_rates": {"success": 1.0}}, False)
    assert report["received_http_attempts"] == 3
    assert report["attempt_any_stale_revision_rate"] is None
    assert report["rank_outcome_rates"] == {"success": None}
    assert report["observation_complete"] is False


def test_actual_null_limit_must_be_present_not_silently_missing():
    m = load()
    with pytest.raises(ValueError):
        m.snapshot_is_empty(
            {"snapshots": [{"limits": {"max_running_requests": 16}, "running": [], "queues": {"waiting": []}}]},
            {"max_running_requests": 16, "prefill_max_requests": None})


def test_method_env_unsets_unapproved_proxy_and_projection_cap(monkeypatch, tmp_path):
    m = load()
    monkeypatch.setenv("HEXGENT_PROXY_MODEL_PATH", "/old/oracle")
    monkeypatch.setenv("SGLANG_DISAGGREGATION_MAX_INFLIGHT_TRANSFERS", "1")
    env, _, gateway, _ = m.method_commands(Path("/src"), tmp_path,
        {"HEXGENT_TRANSFER_STARTUP_MS": 5, "HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S": 1000})
    assert "HEXGENT_PROXY_MODEL_PATH" not in env
    assert "SGLANG_DISAGGREGATION_MAX_INFLIGHT_TRANSFERS" not in env
    assert "--hexgent-admission-queue" in gateway


def test_idle_requires_two_consecutive_empty_polls_and_keeps_schema_failure(tmp_path, monkeypatch):
    m = load()
    limits = {"max_running_requests": 16}
    calls = []
    def get(url, deadline):
        poll = len(calls) // 2
        calls.append(url)
        return {"snapshots": [{"limits": limits, "running": [],
                               "queues": {"waiting": [{"rid": "busy"}] if poll == 0 else []}}]}
    monkeypatch.setattr(m, "get_json", get)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    record = m.wait_idle(tmp_path / "idle.json", {"prefill": limits, "decode": limits}, m.time.time() + 30)
    assert record["poll_count"] == 3
    assert record["success"] is True
    def bad(url, deadline):
        return {"snapshots": [{"limits": limits, "running": [], "queues": {"waiting": None}}]}
    monkeypatch.setattr(m, "get_json", bad)
    with pytest.raises(ValueError):
        m.wait_idle(tmp_path / "bad-idle.json", {"prefill": limits, "decode": limits}, m.time.time() + 30)
    saved = json.loads((tmp_path / "bad-idle.json").read_text())
    assert saved["success"] is False
    assert saved["final"]["prefill"]["snapshots"][0]["queues"]["waiting"] is None


def test_new_reference_artifact_gate_and_shared_trace_plan(tmp_path, monkeypatch):
    from types import SimpleNamespace
    m = load()
    inputs, reference = tmp_path / "inputs", tmp_path / "reference"
    inputs.mkdir()
    reference.mkdir()
    evidence = tmp_path / "runtime.json"
    evidence.write_text('{"fixture":"synthetic tooling only; no H20 results"}')
    rows = []
    for dataset, total in [("sharegpt", 16), ("hotpotqa", 133)]:
        workflows = []
        for index in range(4):
            n = total // 4 + (index < total % 4)
            wid = dataset + str(index)
            workflows.append({"workflow_id": wid, "calls": [
                {"call_id": wid + "-" + str(j), "parents": [], "input_len": 39, "output_len": 8,
                 "request": {"messages": [{"role": "user", "content": "synthetic"}]}}
                for j in range(n)]})
            rows.append({"workflow_id": wid, "status": "successful", "latency_ms": 10,
                         "expected_calls": n, "completed_calls": n, "usage_verified_calls": n, "errors": 0})
        (inputs / (dataset + ".heldout.sanitized.runtime.jsonl")).write_text(
            "".join(json.dumps(w) + "\n" for w in workflows))
    contract = {"model": "synthetic-model", "implementation_revision": "synthetic-revision",
                "runtime_evidence_sha256": m.sha(evidence)}
    for row in rows:
        row.update(contract)
    table = reference / "workflow-references.jsonl"
    table.write_text("".join(json.dumps(row) + "\n" for row in rows))
    gate = {"calibration_workflows_complete": True, "all_heldout_references_complete": True,
            "recorded_heldout_workflows": 8, "successful_heldout_workflows": 8,
            "workflow_reference_sha256": m.sha(table), "reference_contract": contract}
    m.save_json(reference / "collection-gate.json", gate)
    m.save_json(reference / "workflow-reference-contract.json", contract)
    m.save_json(reference / "calibration-burst.summary.json", {"datasets": {
        dataset: {"all_workflows_complete": True, "successful_workflows": 2,
                  "workflow_count": 2, "makespan_ms": 20000}
        for dataset in ["sharegpt", "hotpotqa"]}})
    monkeypatch.setenv("SERVED_MODEL", contract["model"])
    monkeypatch.setenv("IMPLEMENTATION_REVISION", contract["implementation_revision"])
    monkeypatch.setenv("RUNTIME_EVIDENCE_JSON", str(evidence))
    args = SimpleNamespace(reference_dir=reference, runtime_input_dir=inputs,
                           run_dir=tmp_path / "fresh-plan", deadline_unix_s=2000000000)
    assert m.validate_inputs(args) == contract
    plan = m.prepare(args, contract)
    assert len(plan["runs"]) == 8
    for cell in plan["cells"]:
        paired = [r for r in plan["runs"] if r["dataset"] == cell["dataset"] and r["load"] == cell["load"]]
        assert len(paired) == 2
        assert paired[0]["trace"] == paired[1]["trace"]
        assert paired[0]["trace_sha256"] == paired[1]["trace_sha256"]
        assert paired[0]["arrival_offsets_s"] == paired[1]["arrival_offsets_s"]
    rows[0]["status"] = "failed"
    table.write_text("".join(json.dumps(row) + "\n" for row in rows))
    gate["workflow_reference_sha256"] = m.sha(table)
    m.save_json(reference / "collection-gate.json", gate)
    with pytest.raises(ValueError):
        m.validate_inputs(args)
