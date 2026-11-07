"""Synthetic-only tests; no actual H20 workload or service is executed."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/hexgent_real_trace_collectors/matched_h20_simulator.py"
MODEL = {"hidden_size": 8192, "intermediate_size": 28672, "num_hidden_layers": 80,
         "num_attention_heads": 64, "num_key_value_heads": 8, "vocab_size": 128256,
         "head_dim": 128, "tp_size": 2, "w_bit": 16, "a_bit": 16, "kv_bit": 16}
HARDWARE = {"memory_bandwidth_bytes_per_s": 4096e9, "FP16": 148e12}
FLAGS = {"AGENTIC_PD_OURS_PLACEMENT_STRATEGY": "projected_finish",
         "HEXGENT_PROJECTED_RATIO_ORDER": "high_first", "HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH": "0"}
BOUNDARIES = {"prefill_service": "forward_ms_minus_prefill_last_chunk_ms",
              "decode_service": "decode_forward_ms",
              "transfer": "prefill_last_chunk_queue_to_tp_consensus",
              "transfer_payload_scope": "per_tp_rank",
              "transfer_payload_rank": 0,
              "transfer_precision": "rounded_log_reconstruction",
              "single_chunk_verified": True,
              "serial_additive_model_boundary_verified": True,
              "bandwidth_interpretation": "full_group_effective_not_physical"}


def load_candidate():
    name = "_h20_matched_candidate"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def apis(module):
    sim = Path(os.environ["MATCHED_SIM_TEST_SIM_ROOT"])
    real = Path(os.environ["MATCHED_SIM_TEST_REAL_ROOT"])
    revision = lambda path: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    return module.load_apis(sim, real, sim_revision=revision(sim), real_revision=revision(real))


def call(name, tokens, output=3, parents=()):
    return {"call_id": name, "input_len": tokens, "output_len": output,
            "parents": list(parents), "request": {"messages": [{"role": "user", "content": name}]},
            "planner_hints": {"model_config": copy.deepcopy(MODEL), "hardware_params": dict(HARDWARE)}}


def heldout():
    return [{"workflow_id": "held-a", "workflow_type": "synthetic", "calls": [
                call("a-root", 120), call("a-child", 140, parents=("a-root",))]},
            {"workflow_id": "held-b", "workflow_type": "synthetic", "calls": [
                call("b-root", 160), call("b-left", 180, parents=("b-root",)),
                call("b-right", 200, parents=("b-root",)),
                call("b-join", 220, parents=("b-left", "b-right"))]}]


def calibration_trace():
    return [{"workflow_id": "cal-only", "workflow_type": "synthetic", "calls": [
        call("cal-0", 100, 2), call("cal-1", 200, 4), call("cal-2", 300, 6)]}]


def real_rows(workflows):
    rows = []
    for index, workflow in enumerate(workflows):
        for item in workflow["calls"]:
            rows.append({"workflow_id": workflow["workflow_id"], "call_id": item["call_id"],
                         "workflow_arrival_ms": 1000 + index * 234, "error": None,
                         "response": {"usage": {"prompt_tokens": item["input_len"],
                                                  "completion_tokens": item["output_len"]}}})
    return rows


def runtime():
    return {"model_config": dict(MODEL), "hardware_params": dict(HARDWARE),
            "prefill_workers": 1, "decode_capacity_tokens": [10000],
            "decode_max_batch_size": [1], "scheduler_overhead_mode": "off", "seed": 42,
            "scheduler_flags": dict(FLAGS), "cache_policy": "disabled",
            "tool_policy": "fixed_llm_dag_no_external_tools"}


def fitted(module, api):
    nominal = module.NominalCosts(api, MODEL, HARDWARE)
    rows = []
    for item in calibration_trace()[0]["calls"]:
        p, d = nominal.estimate_call_times(item["input_len"], item["output_len"])
        rows.append({"workflow_id": "cal-only", "call_id": item["call_id"],
                     "input_tokens": item["input_len"], "output_tokens": item["output_len"],
                     "prefill_before_lastchunk_handoff_ms": p * 1000 * 2, "decode_forward_ms": d * 1000 * 3,
                     "transfer_surrogate_ms": 5 + 0.1 * item["input_len"],
                     "transfer_rank_bytes_exact": item["input_len"] * 163840,
                     "transfer_fullgroup_bytes": item["input_len"] * 327680})
    document = {"source_split": "calibration", "model_config": dict(MODEL),
                "boundaries": dict(BOUNDARIES), "rows": rows}
    fit = module.fit_calibration(document, calibration_trace(), heldout(), nominal)
    return nominal, document, fit


def test_lossless_dag_conversion_has_no_profile_oracle():
    module = load_candidate(); api = apis(module)
    source = heldout(); original = copy.deepcopy(source)

    traces, call_ids = module.convert_workflows(source, api, model="synthetic-served")

    assert source == original
    assert [trace.workflow_id for trace in traces] == ["held-a", "held-b"]
    assert traces[1].dag_edges == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert call_ids == {"held-a": ["a-root", "a-child"],
                        "held-b": ["b-root", "b-left", "b-right", "b-join"]}
    for trace in traces:
        for item in trace.calls:
            assert item.profiled_prefill_time_s is None and item.profiled_decode_time_s is None


def test_dag_missing_parent_is_reported_instead_of_silently_dropped():
    module = load_candidate(); api = apis(module)
    source = heldout(); source[0]["calls"][1]["parents"] = ["missing"]

    with pytest.raises(module.CandidateBlocked, match="parent"):
        module.convert_workflows(source, api, model="synthetic-served")


def test_saved_arrivals_are_exact_no_rng_and_usage_is_checked():
    module = load_candidate(); source = heldout(); rows = real_rows(source)

    offsets, evidence = module.arrivals_from_replay(source, list(reversed(rows)))

    assert offsets == {"held-a": 0.0, "held-b": 0.234}
    assert evidence["origin_ms"] == 1000
    assert evidence["usage_verified_calls"] == 6
    rows[0]["response"]["usage"]["prompt_tokens"] = 2
    with pytest.raises(module.CandidateBlocked, match="token.*mismatch"):
        module.arrivals_from_replay(source, rows)


def test_entirely_missing_arrival_is_blocked_without_guess():
    module = load_candidate(); source = heldout()
    rows = [row for row in real_rows(source) if row["workflow_id"] != "held-b"]

    with pytest.raises(module.CandidateBlocked, match="arrival"):
        module.arrivals_from_replay(source, rows)


def test_calibration_only_global_fit_separates_prediction_and_execution():
    module = load_candidate(); api = apis(module)
    nominal, document, fit = fitted(module, api)

    assert fit["prefill_scale"] == pytest.approx(2)
    assert fit["decode_scale"] == pytest.approx(3)
    assert fit["transfer_startup_ms"] == pytest.approx(5)
    assert fit["transfer_ms_per_input_token"] == pytest.approx(0.1)
    assert fit["calibration_workflows"] == 1 and fit["calibration_calls"] == 3
    assert fit["boundaries"] == BOUNDARIES
    assert fit["transfer_observed_bytes_per_input_token_per_rank"] == 163840
    estimator = module.CalibratedCosts(nominal, fit)
    predicted = estimator.estimate_call_times(150, 3)
    actual = estimator.estimate_actual_call_times(150, 3)
    assert actual == pytest.approx((2 * predicted[0], 3 * predicted[1]))
    assert estimator.estimate_transfer_time(150) == pytest.approx(0.020)
    assert "service_by_call_id" not in fit


def test_heldout_calibration_or_wrong_transfer_boundary_is_blocked():
    module = load_candidate(); api = apis(module)
    nominal, document, _ = fitted(module, api)
    poisoned = copy.deepcopy(document); poisoned["rows"][0]["workflow_id"] = "held-a"
    with pytest.raises(module.CandidateBlocked, match="calibration"):
        module.fit_calibration(poisoned, calibration_trace(), heldout(), nominal)
    poisoned = copy.deepcopy(document); poisoned["boundaries"]["transfer"] = "decode_transfer_duration"
    with pytest.raises(module.CandidateBlocked, match="boundary"):
        module.fit_calibration(poisoned, calibration_trace(), heldout(), nominal)


def test_unfit_transfer_data_is_blocked_without_default_bandwidth():
    module = load_candidate(); api = apis(module)
    nominal, document, _ = fitted(module, api)
    source = calibration_trace()
    for row, item in zip(document["rows"], source[0]["calls"]):
        row["input_tokens"] = item["input_len"] = 100
        row["transfer_rank_bytes_exact"] = 163840 * 100
        row["transfer_fullgroup_bytes"] = 327680 * 100
    with pytest.raises(module.CandidateBlocked, match="distinct payload"):
        module.fit_calibration(document, source, heldout(), nominal)


def test_native_synthetic_replay_injects_every_workflow_once():
    module = load_candidate(); api = apis(module)
    nominal, _, fit = fitted(module, api)
    source = heldout(); traces, call_ids = module.convert_workflows(source, api, model="synthetic-served")
    offsets, _ = module.arrivals_from_replay(source, real_rows(source))

    sim, mapping = module.run_native(api, traces, offsets, runtime(), module.CalibratedCosts(nominal, fit))

    assert sim.total_workflows == 2 and sim.completed_workflows == 2
    assert {entry["source_workflow_id"] for entry in mapping.values()} == {"held-a", "held-b"}
    for native_id, entry in mapping.items():
        assert sim.workflows[native_id].arrival_time == offsets[entry["source_workflow_id"]]
        assert len(sim.workflows[native_id].all_calls) == len(call_ids[entry["source_workflow_id"]])
    assert len(sim.completed_calls) == 6


def reference_contract(runtime_hash="f" * 64):
    return {"model": "synthetic-served", "implementation_revision": "synthetic_runtime_revision",
            "runtime_evidence_sha256": runtime_hash}


def reference_rows(workflows, multiplier=1, runtime_hash="f" * 64):
    return [{"workflow_id": w["workflow_id"], "reference_kind": "isolated_workflow_replay",
             "latency_ms": (index + 1) * 100 * multiplier, "status": "successful",
             "expected_calls": len(w["calls"]), "completed_calls": len(w["calls"]),
             "errors": 0, "usage_verified_calls": len(w["calls"]), "model": "synthetic-served",
             "implementation_revision": "synthetic_runtime_revision", "raw_replay_sha256": "a" * 64,
             "runtime_evidence_sha256": runtime_hash}
            for index, w in enumerate(workflows)]


def test_workflow_reference_changes_only_offline_ratios():
    module = load_candidate(); api = apis(module)
    nominal, _, fit = fitted(module, api)
    source = heldout(); traces, _ = module.convert_workflows(source, api, model="synthetic-served")
    offsets, _ = module.arrivals_from_replay(source, real_rows(source))
    sim, mapping = module.run_native(api, traces, offsets, runtime(), module.CalibratedCosts(nominal, fit))
    refs_a = module.workflow_references(source, reference_rows(source), reference_contract())
    refs_b = module.workflow_references(source, reference_rows(source, 10), reference_contract())
    a = module.build_workflow_report(sim, mapping, refs_a)
    b = module.build_workflow_report(sim, mapping, refs_b)
    assert [row["base_slo_ms"] for row in a] == [100, 200]
    assert a[1]["base_slo_ms"] != 30  # Legacy isolated-call CP proxy for this fanout fixture.
    assert [row["latency_ms"] for row in a] == [row["latency_ms"] for row in b]
    for left, right in zip(a, b):
        assert right["slo_ratio"] == pytest.approx(left["slo_ratio"] / 10)
        assert left["reference_kind"] == "isolated_workflow_replay"
        assert left["base_slo_ms"] != left["native_model_slo_ms"]


def test_incomplete_workflows_remain_null_and_expected():
    module = load_candidate(); api = apis(module)
    nominal, _, fit = fitted(module, api)
    source = heldout(); traces, _ = module.convert_workflows(source, api, model="synthetic-served")
    offsets, _ = module.arrivals_from_replay(source, real_rows(source))
    config = runtime(); config["decode_capacity_tokens"] = [1]
    sim, mapping = module.run_native(api, traces, offsets, config, module.CalibratedCosts(nominal, fit))
    refs = module.workflow_references(source, reference_rows(source), reference_contract())
    rows = module.build_workflow_report(sim, mapping, refs)
    assert len(rows) == 2 and sim.completed_workflows == 0
    assert all(row["status"] == "incomplete" and row["latency_ms"] is None
               and row["slo_ratio"] is None for row in rows)
    assert sum(row["expected_calls"] for row in rows) == 6


def test_partial_or_failed_real_replay_is_not_a_matched_call_set():
    module = load_candidate(); source = heldout(); rows = real_rows(source)
    with pytest.raises(module.CandidateBlocked, match="every expected call"):
        module.arrivals_from_replay(source, rows[:-1])
    rows[0]["error"] = "synthetic real failure"
    with pytest.raises(module.CandidateBlocked, match="failed real"):
        module.arrivals_from_replay(source, rows)


def test_cycle_is_rejected_before_native_execution():
    module = load_candidate(); api = apis(module); source = heldout()
    source[0]["calls"][0]["parents"] = ["a-child"]
    with pytest.raises(module.CandidateBlocked, match="cycle"):
        module.convert_workflows(source, api, model="synthetic-served")


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_fresh_denominator_requires_positive_finite_evidence(bad):
    module = load_candidate(); source = heldout(); rows = reference_rows(source)
    rows[0]["latency_ms"] = bad
    with pytest.raises(module.CandidateBlocked, match="positive finite"):
        module.workflow_references(source, rows, reference_contract())


def comparison(runtime):
    return {"model": "synthetic-served", "model_revision": "frozen-synthetic-revision",
            "tp": 2, "weight_dtype": "bfloat16", "kv_dtype": "bfloat16",
            "hardware": "NVDA:H20",
            "topology": {"prefill_workers": runtime["prefill_workers"],
                         "decode_workers": len(runtime["decode_capacity_tokens"]), "tp_size": 2,
                         "decode_capacity_tokens": runtime["decode_capacity_tokens"],
                         "decode_max_batch_size": runtime["decode_max_batch_size"]},
            "cache_policy": runtime["cache_policy"], "tool_policy": runtime["tool_policy"],
            "scheduler_flags": runtime["scheduler_flags"], "observation_horizon": "revealed_calls_only",
            "arrival": {"source": "saved_real_workflow_arrival_ms", "completion_policy": "drain_all_expected_calls"}}


def test_effective_runtime_contract_cannot_contradict_frozen_metadata():
    module = load_candidate(); actual = runtime()
    actual.update(model="synthetic-served", model_revision="frozen-synthetic-revision")
    frozen = comparison(actual)
    module.validate_runtime_contract(frozen, actual)
    for key, value in (("tp", 4), ("hardware", "NVDA:H100"),
                       ("cache_policy", "enabled"), ("scheduler_flags", {}),
                       ("topology", {"prefill_workers": 2}),
                       ("observation_horizon", "future_full_DAG")):
        poisoned = copy.deepcopy(frozen); poisoned[key] = value
        with pytest.raises(module.CandidateBlocked, match="effective"):
            module.validate_runtime_contract(poisoned, actual)


@pytest.mark.parametrize("field,value", [("model", "different-model"),
    ("implementation_revision", "different-runtime"), ("runtime_evidence_sha256", "e" * 64),
    ("reference_kind", "isolated_call_critical_path"), ("completed_calls", 1),
    ("usage_verified_calls", 0), ("raw_replay_sha256", "not-a-hash")])
def test_reference_table_must_match_frozen_complete_workflow(field, value):
    module = load_candidate(); source = heldout(); rows = reference_rows(source)
    rows[0][field] = value
    with pytest.raises(module.CandidateBlocked):
        module.workflow_references(source, rows, reference_contract())


def test_reference_failures_and_missing_rows_remain_unknown_without_proxy_fallback():
    module = load_candidate(); source = heldout(); rows = reference_rows(source)
    rows[0].update(status="failed", errors=1, completed_calls=0, usage_verified_calls=0, latency_ms=None)
    refs = module.workflow_references(source, rows[:1], reference_contract())
    assert refs["held-a"]["latency_ms"] is None and refs["held-a"]["row"] == rows[0]
    assert refs["held-a"]["status"] == "failed"
    assert refs["held-b"]["latency_ms"] is None and refs["held-b"]["status"] == "missing"
    with pytest.raises(module.CandidateBlocked, match="duplicate"):
        module.workflow_references(source, rows + rows[:1], reference_contract())


def test_candidate_file_contract_end_to_end_uses_reference_only_for_report(tmp_path):
    module = load_candidate(); api = apis(module)
    _, calibration, _ = fitted(module, api)
    def write_json(name, value):
        path = tmp_path / name; path.write_text(json.dumps(value)); return path
    def write_jsonl(name, value):
        path = tmp_path / name; path.write_text("".join(json.dumps(row) + "\n" for row in value)); return path
    source = heldout()
    trace = write_jsonl("heldout.jsonl", source)
    caltrace = write_jsonl("calibration.jsonl", calibration_trace())
    stages = write_jsonl("stages.jsonl", calibration.pop("rows"))
    calibration.update(source_revision="synthetic_calibration_revision",
                       stage_records_sha256=module.file_hash(stages),
                       calibration_trace_sha256={str(caltrace): module.file_hash(caltrace)})
    boundary = write_json("boundary.json", calibration)
    replay = write_jsonl("replay.jsonl", real_rows(source))
    evidence = write_json("observed-runtime.json", {"synthetic": True, "decode_total_capacity": 10000})
    ref_contract = reference_contract(module.file_hash(evidence))
    profile = write_jsonl("workflow-reference.jsonl", reference_rows(source, runtime_hash=module.file_hash(evidence)))
    actual = runtime(); actual.update(model="synthetic-served", model_revision="frozen-synthetic-revision",
                                      implementation_revision="synthetic_runtime_revision")
    frozen = comparison(actual)
    frozen.update(trace_sha256=module.file_hash(trace), profile_sha256=module.file_hash(profile))
    frozen["arrival"]["replay_sha256"] = module.file_hash(replay)
    contract = write_json("contract.json", {"comparison_contract": frozen, "runtime": actual, "workflow_reference_contract": ref_contract,
                          "runtime_evidence": {"path": str(evidence), "sha256": module.file_hash(evidence)}})
    args = SimpleNamespace(sim_root=Path(os.environ["MATCHED_SIM_TEST_SIM_ROOT"]),
                           real_root=Path(os.environ["MATCHED_SIM_TEST_REAL_ROOT"]),
                           sim_revision=api.provenance["sim_revision"], real_revision=api.provenance["real_prediction_revision"],
                           trace=trace, calibration_trace=[caltrace], calibration_records=stages,
                           calibration_boundary=boundary, real_replay=replay, workflow_reference=profile,
                           contract=contract, evidence_kind="synthetic_tooling_check")
    first = module.run_candidate(args)
    profile.write_text("".join(json.dumps(row) + "\n" for row in reference_rows(source, 10, runtime_hash=module.file_hash(evidence))))
    frozen["profile_sha256"] = module.file_hash(profile)
    contract.write_text(json.dumps({"comparison_contract": frozen, "runtime": actual, "workflow_reference_contract": ref_contract,
                                   "runtime_evidence": {"path": str(evidence), "sha256": module.file_hash(evidence)}}))
    second = module.run_candidate(args)
    assert first["counts"] == {"expected_workflows": 2, "simulated_workflows": 2, "completed_workflows": 2,
                               "expected_calls": 6, "completed_calls": 6}
    assert first["calibration_fit"] == second["calibration_fit"]
    assert [r["latency_ms"] for r in first["workflows"]] == [r["latency_ms"] for r in second["workflows"]]
    for left, right in zip(first["workflows"], second["workflows"]):
        assert right["slo_ratio"] == pytest.approx(left["slo_ratio"] / 10)
    assert first["test_service_oracle_used"] is False
    assert first["runtime_complete_parity_claimed"] is False
    boundary.write_text(json.dumps({**calibration, "stage_records_sha256": "wrong"}))
    with pytest.raises(module.CandidateBlocked, match="frozen hash"):
        module.run_candidate(args)


def test_unknown_flag_value_cannot_freeze_a_silent_core_fallback():
    module = load_candidate(); flags = dict(FLAGS)
    flags["HEXGENT_PROJECTED_RATIO_ORDER"] = "typo_not_a_direction"
    with pytest.raises(module.CandidateBlocked, match="flag value"):
        with module._policy_environment(flags):
            pass


def test_failed_or_missing_reference_keeps_successful_execution_and_unknown_ratio():
    module = load_candidate(); api = apis(module)
    nominal, _, fit = fitted(module, api); source = heldout()
    traces, _ = module.convert_workflows(source, api, model="synthetic-served")
    offsets, _ = module.arrivals_from_replay(source, real_rows(source))
    sim, mapping = module.run_native(api, traces, offsets, runtime(), module.CalibratedCosts(nominal, fit))
    baseline = module.build_workflow_report(sim, mapping, module.workflow_references(source, reference_rows(source), reference_contract()))
    failed = reference_rows(source)[:1]
    failed[0].update(status="failed", errors=1, completed_calls=0, usage_verified_calls=0, latency_ms=None)
    unknown = module.build_workflow_report(sim, mapping, module.workflow_references(source, failed, reference_contract()))
    assert [r["latency_ms"] for r in baseline] == [r["latency_ms"] for r in unknown]
    assert [r["reference_status"] for r in unknown] == ["failed", "missing"]
    assert all(r["status"] == "successful" and r["base_slo_ms"] is None and r["slo_ratio"] is None for r in unknown)
