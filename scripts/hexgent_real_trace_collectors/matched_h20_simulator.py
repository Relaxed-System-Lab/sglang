#!/usr/bin/env python3
"""Standalone matched-input native simulator adaptation candidate; review required.

Only frozen v2 lengths/targets, saved real arrivals, calibration-only global
cost fits and evaluation-only standalone workflow references are accepted. This runner
preserves the canonical scheduler/C3 adapter and exposes native-model limits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import heapq
import importlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import types

sys.dont_write_bytecode = True
CORE_SHA256 = "cb71482755ba8f27c5234effe2e4e1c0897763d45e00001acfe9f592b3d139a3"
CONTRACT_FIELDS = ("trace_sha256", "profile_sha256", "model", "model_revision", "tp",
                   "weight_dtype", "kv_dtype", "hardware", "topology", "cache_policy",
                   "tool_policy", "arrival", "scheduler_flags", "observation_horizon")
LIMITATIONS = [
    "Native prefill has one running call per worker, without packed prefill batches.",
    "Native decode durations are fixed per call; concurrent batch size does not change progress.",
    "Each call executes serial P then transfer then D; transfers overlap freely across calls without shared-link contention.",
    "C3 uses exact simulated running remainders and scalar non-running commitments; real snapshots estimate intervals.",
    "Native revealed horizons retain predicted service; real completed history can replace it with observed service.",
    "Children reveal immediately at dependency completion, without client/HTTP round-trip gaps.",
    "Native-generated IDs can change lexical ties; equal-time Event ordering has no explicit sequence key.",
    "Native scheduling/fallback timing excludes actual router RPC and admission execution semantics.",
    "Real local-waiting mutation gating has no equivalent native adapter control; core equality does not establish queue-mutation parity.",
    "Saved arrivals have the precision of persisted workflow_arrival_ms; sub-millisecond ordering is unavailable.",
    "Transfer fits use a declared last-chunk surrogate and effective byte conventions, not pure-DMA bandwidth.",
]


class CandidateBlocked(ValueError):
    """A necessary observed contract or identifiable calibration fit is missing."""


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _repo_sources(root, revision, names):
    root = Path(root).resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise CandidateBlocked(f"source revision mismatch: {root}: {actual} != {revision}")
    changed = subprocess.run(["git", "-C", str(root), "diff", "--quiet", revision, "--", *names])
    if changed.returncode:
        raise CandidateBlocked(f"required source files have pending changes: {root}")
    return {name: hashlib.sha256((root / name).read_text().encode()).hexdigest() for name in names}


def load_apis(sim_root, real_root, *, sim_revision, real_revision):
    sim_root = Path(sim_root).resolve(); real_root = Path(real_root).resolve()
    sim_names = ["pd_disaggregated/sglang_scheduler_core.py", "pd_disaggregated/sglang_scheduler.py",
                 "pd_disaggregated/simulator.py", "pd_disaggregated/scheduler_interface.py"]
    real_names = ["python/sglang/hexgent/latency_estimator.py", "python/sglang/hexgent/schemas.py"]
    provenance = {"sim_revision": sim_revision, "real_prediction_revision": real_revision,
                  "sim_sources": _repo_sources(sim_root, sim_revision, sim_names),
                  "real_sources": _repo_sources(real_root, real_revision, real_names)}
    if provenance["sim_sources"][sim_names[0]] != CORE_SHA256:
        raise CandidateBlocked("canonical core is not the accepted C1+C2 cb714 reference")
    interface_text = (sim_root / sim_names[3]).read_text()
    if "running_call_ids" not in interface_text or "decode_reservations" not in interface_text:
        raise CandidateBlocked("C3 complete-snapshot interface is absent")
    sys.path.insert(0, str(sim_root))
    sim = importlib.import_module("pd_disaggregated.simulator")
    sched = importlib.import_module("pd_disaggregated.sglang_scheduler")
    trace = importlib.import_module("utils.trace_loader")
    if Path(sim.__file__).resolve() != sim_root / sim_names[2]:
        raise CandidateBlocked("another simulator checkout was already imported")
    # Load only standalone real estimator/schema modules, avoiding SGLang init.
    package_root = real_root / "python/sglang"
    for name, path in (("sglang", package_root), ("sglang.hexgent", package_root / "hexgent")):
        if name not in sys.modules:
            package = types.ModuleType(name); package.__path__ = [str(path)]
            sys.modules[name] = package
    schemas = importlib.import_module("sglang.hexgent.schemas")
    latency = importlib.import_module("sglang.hexgent.latency_estimator")
    if Path(latency.__file__).resolve() != real_root / real_names[0]:
        raise CandidateBlocked("another real estimator checkout was already imported")
    return types.SimpleNamespace(Simulator=sim.PDDisaggregatedSimulator, Event=sim.Event,
                                 EventType=sim.EventType, Scheduler=sched.SGLangHexAgenTScheduler,
                                 TraceCall=trace.TraceCall, TraceWorkflow=trace.TraceWorkflow,
                                 latency=latency, schemas=schemas, provenance=provenance)


def _calls_by_key(workflows):
    result = {}
    workflow_ids = set()
    for workflow in workflows:
        wid = str(workflow["workflow_id"])
        if wid in workflow_ids:
            raise CandidateBlocked(f"duplicate workflow ID: {wid}")
        workflow_ids.add(wid)
        for call in workflow["calls"]:
            key = (wid, str(call["call_id"]))
            if key in result:
                raise CandidateBlocked(f"duplicate call identity: {key}")
            result[key] = call
    return result


def convert_workflows(workflows, api, *, model):
    _calls_by_key(workflows)
    # Validate graph integrity before entering the native event loop.
    _validate_dags(workflows)
    traces = []; call_ids = {}
    for workflow in workflows:
        wid = str(workflow["workflow_id"])
        ids = [str(call["call_id"]) for call in workflow["calls"]]
        indices = {name: index for index, name in enumerate(ids)}
        edges = []; calls = []
        for index, call in enumerate(workflow["calls"]):
            if set(call.get("planner_hints", {})) != {"model_config", "hardware_params"}:
                raise CandidateBlocked("runtime input hints must contain only static model_config/hardware_params")
            for parent in call["parents"]:
                if str(parent) not in indices:
                    raise CandidateBlocked(f"missing parent {parent} in {wid}")
                edges.append((indices[str(parent)], index))
            calls.append(api.TraceCall(input_length=int(call["input_len"]),
                                       output_length=int(call["output_len"]),
                                       tool_name=ids[index], model_name=model))
        traces.append(api.TraceWorkflow(wid, str(workflow.get("workflow_type", "fixed_dag")), calls, edges))
        call_ids[wid] = ids
    return traces, call_ids


def arrivals_from_replay(workflows, rows):
    expected = _calls_by_key(workflows)
    arrivals = {str(workflow["workflow_id"]): set() for workflow in workflows}
    seen = set(); verified = 0; errors = 0
    for row in rows:
        key = (str(row["workflow_id"]), str(row["call_id"]))
        if key not in expected or key in seen:
            raise CandidateBlocked(f"unexpected or duplicate real replay identity: {key}")
        seen.add(key)
        value = row.get("workflow_arrival_ms")
        if value is None or not math.isfinite(float(value)):
            raise CandidateBlocked(f"missing/invalid saved arrival: {key}")
        arrivals[key[0]].add(float(value))
        if row.get("error"):
            raise CandidateBlocked(f"failed real call prevents matched executed-call comparison: {key}")
        usage = (row.get("response") or {}).get("usage") or {}
        if "prompt_tokens" not in usage or "completion_tokens" not in usage:
            raise CandidateBlocked(f"server token usage unavailable for successful call: {key}")
        call = expected[key]
        if int(usage["prompt_tokens"]) != int(call["input_len"]) or int(usage["completion_tokens"]) != int(call["output_len"]):
            raise CandidateBlocked(f"server token length mismatch: {key}; frozen input/target remain unchanged")
        verified += 1
    if any(len(values) != 1 for values in arrivals.values()):
        raise CandidateBlocked("each expected workflow needs one unique saved real arrival")
    if seen != set(expected):
        raise CandidateBlocked("saved real replay must contain every expected call for matched comparison")
    times = {wid: next(iter(values)) for wid, values in arrivals.items()}
    origin = min(times.values())
    return {wid: (value - origin) / 1000.0 for wid, value in times.items()}, {
        "origin_ms": origin, "usage_verified_calls": verified, "real_error_calls": errors,
        "missing_real_call_ids": [list(key) for key in sorted(set(expected) - seen)],
        "arrival_source": "saved_real_workflow_arrival_ms", "arrival_precision": "persisted_ms",
    }


class NominalCosts:
    """Real pure roofline at batch size one; no measured per-call lookup."""

    def __init__(self, api, model_config, hardware_params):
        self.api = api; self.model_config = copy.deepcopy(model_config)
        self.hardware_params = copy.deepcopy(hardware_params)
        if model_config["tp_size"] != 2 or any(model_config[key] != 16 for key in ("w_bit", "a_bit", "kv_bit")):
            raise CandidateBlocked("candidate scope requires TP2 and 16-bit weights/activations/KV")
        worker = api.schemas.WorkerDescriptor(id="static", url="static", worker_type="decode", tp_size=2)
        self.model = api.latency._model_params_from_payload(model_config, worker)
        if self.model is None:
            raise CandidateBlocked("complete actual model configuration is required")
        self.roofline = api.latency.RooflineLatencyEstimator()
        self.full_kv_bytes_per_token = (2 * self.model.num_hidden_layers * self.model.head_dim
                                        * self.model.num_key_value_heads * self.model.kv_byte)

    def estimate_call_times(self, input_length, output_length, hardware=None):
        p = self.roofline.estimate_prefill_ms(model=self.model, hardware=self.hardware_params,
                                             input_tokens=int(input_length))
        d = self.roofline.estimate_decode_total_ms(model=self.model, hardware=self.hardware_params,
                                                  input_tokens=int(input_length),
                                                  output_tokens=int(output_length), batch_size=1)
        return p / 1000.0, d / 1000.0


def fit_calibration(document, calibration_workflows, heldout_workflows, nominal):
    calibration = _calls_by_key(calibration_workflows)
    heldout = _calls_by_key(heldout_workflows)
    if set(key[0] for key in calibration) & set(key[0] for key in heldout):
        raise CandidateBlocked("calibration and heldout workflow identities must be disjoint")
    if document.get("source_split") != "calibration" or document["model_config"] != nominal.model_config:
        raise CandidateBlocked("calibration split/model configuration mismatch")
    boundaries = document["boundaries"]
    required = {"prefill_service": "forward_ms_minus_prefill_last_chunk_ms",
                "decode_service": "decode_forward_ms",
                "transfer": "prefill_last_chunk_queue_to_tp_consensus",
                "transfer_payload_scope": "per_tp_rank",
                "single_chunk_verified": True, "serial_additive_model_boundary_verified": True,
                "bandwidth_interpretation": "full_group_effective_not_physical"}
    if any(boundaries.get(key) != value for key, value in required.items()):
        raise CandidateBlocked("unsupported or unverified calibration boundary")
    p_ratios = []; d_ratios = []; xs = []; ys = []; byte_ratios = []; seen = set()
    for row in document["rows"]:
        key = (str(row["workflow_id"]), str(row["call_id"]))
        if key not in calibration or key in seen or key in heldout:
            raise CandidateBlocked(f"calibration contains heldout/unexpected/duplicate identity: {key}")
        seen.add(key); expected = calibration[key]
        if int(row["input_tokens"]) != int(expected["input_len"]) or int(row["output_tokens"]) != int(expected["output_len"]):
            raise CandidateBlocked(f"calibration token budget mismatch: {key}")
        values = [float(row[name]) for name in ("prefill_before_lastchunk_handoff_ms", "decode_forward_ms", "transfer_surrogate_ms", "transfer_rank_bytes_exact")]
        if row.get("error") or any(not math.isfinite(value) or value <= 0 for value in values):
            raise CandidateBlocked(f"calibration has failed/missing/nonpositive stage evidence: {key}")
        expected_full_bytes = nominal.full_kv_bytes_per_token * int(row["input_tokens"])
        if (int(row["transfer_fullgroup_bytes"]) != expected_full_bytes
                or values[3] * nominal.model_config["tp_size"] != expected_full_bytes):
            raise CandidateBlocked("observed rank/fullgroup transfer payload differs from frozen TP2 model geometry")
        p, d = nominal.estimate_call_times(row["input_tokens"], row["output_tokens"])
        p_ratios.append(values[0] / (p * 1000)); d_ratios.append(values[1] / (d * 1000))
        xs.append(float(row["input_tokens"])); ys.append(values[2])
        byte_ratios.append(values[3] / float(row["input_tokens"]))
    if seen != set(calibration):
        raise CandidateBlocked("calibration evidence does not cover the frozen calibration call set")
    if len(set(xs)) < 2:
        raise CandidateBlocked("transfer startup/bandwidth fit requires two distinct payload sizes")
    xbar = mean(xs); ybar = mean(ys)
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum((x - xbar) ** 2 for x in xs)
    startup = ybar - slope * xbar
    if slope <= 0 or startup < 0 or not math.isfinite(slope + startup):
        raise CandidateBlocked("positive-bandwidth/nonnegative-startup transfer fit is unavailable")
    return {"prefill_scale": median(p_ratios), "decode_scale": median(d_ratios),
            "transfer_startup_ms": startup, "transfer_ms_per_input_token": slope,
            "transfer_observed_bytes_per_input_token_per_rank": median(byte_ratios),
            "transfer_observed_rank_effective_bandwidth_bytes_per_s": median(byte_ratios) * 1000 / slope,
            "transfer_model_effective_full_logical_bandwidth_bytes_per_s": nominal.full_kv_bytes_per_token * 1000 / slope,
            "transfer_full_logical_bytes_per_input_token": nominal.full_kv_bytes_per_token,
            "calibration_calls": len(seen), "calibration_workflows": len({key[0] for key in seen}),
            "calibration_ids": [list(key) for key in sorted(seen)],
            "input_token_range": [min(xs), max(xs)], "boundaries": copy.deepcopy(boundaries),
            "fit_scope": "calibration-only global scales and additive transfer surrogate; no per-call heldout service"}


class CalibratedCosts:
    """Separate nominal scheduler predictions from fitted native execution."""

    def __init__(self, nominal, fit):
        self.nominal = nominal; self.fit = fit

    def estimate_call_times(self, input_length, output_length, hardware=None):
        return self.nominal.estimate_call_times(input_length, output_length, hardware)

    def estimate_actual_call_times(self, input_length, output_length, hardware=None):
        p, d = self.estimate_call_times(input_length, output_length, hardware)
        return p * self.fit["prefill_scale"], d * self.fit["decode_scale"]

    def estimate_transfer_time(self, input_length):
        return (self.fit["transfer_startup_ms"] + self.fit["transfer_ms_per_input_token"] * input_length) / 1000.0


@contextmanager
def _policy_environment(flags):
    names = ("AGENTIC_PD_OURS_PLACEMENT_STRATEGY", "HEXGENT_PROJECTED_RATIO_ORDER", "HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH")
    if set(flags) != set(names):
        raise CandidateBlocked("all three canonical scheduler flags must be frozen explicitly")
    allowed = ({"projected_finish", "weighted"}, {"high_first", "low_first"}, {"0", "1"})
    if any(flags[name] not in values for name, values in zip(names, allowed)):
        raise CandidateBlocked("scheduler flag value must be explicit canonical spelling, without core fallback")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: str(flags[name]) for name in names})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_native(api, traces, offsets, runtime, estimator):
    if runtime["cache_policy"] != "disabled" or runtime["tool_policy"] != "fixed_llm_dag_no_external_tools":
        raise CandidateBlocked("candidate supports cache-disabled fixed LLM DAG replay only")

    class ExactArrivalSimulator(api.Simulator):
        def generate_arrivals(self, duration, arrival_rate, arrival_mode="poisson", burst_jitter=0):
            for trace in traces:
                arrival = offsets[trace.workflow_id]
                workflow = self._build_workflow(trace, arrival)
                self.total_workflows += 1
                mapping[workflow.workflow_id] = {
                    "source_workflow_id": trace.workflow_id,
                    "source_call_ids": [call.tool_name for call in trace.calls],
                    "native_call_ids": [call.call_id for call in workflow.all_calls],
                }
                heapq.heappush(self.event_queue, api.Event(
                    time=arrival, event_type=api.EventType.WORKFLOW_ARRIVAL,
                    data={"workflow_id": workflow.workflow_id, "workflow": workflow},
                ))

    mapping = {}
    with _policy_environment(runtime["scheduler_flags"]):
        sim = ExactArrivalSimulator(
            scheduler=api.Scheduler(), estimator=estimator,
            trace_loader=types.SimpleNamespace(traces={"frozen": traces}),
            num_prefill_instances=int(runtime["prefill_workers"]),
            num_decode_instances=len(runtime["decode_capacity_tokens"]),
            decode_capacity_tokens=list(runtime["decode_capacity_tokens"]),
            decode_max_batch_size=list(runtime["decode_max_batch_size"]),
            prefill_hardware_types=["NVDA:H20"] * int(runtime["prefill_workers"]),
            decode_hardware_types=["NVDA:H20"] * len(runtime["decode_capacity_tokens"]),
            slo_reference_hardware=None, scheduler_overhead_mode=runtime["scheduler_overhead_mode"],
            simulate_tool_latency=False, tool_aware_horizon=False,
            simulate_kv_reuse=False, kv_aware_planning=False, seed=int(runtime["seed"]),
        )
        sim.run_simulation(duration=0, arrival_rate=0, verbose=False)
    return sim, mapping


def workflow_references(workflows, rows, contract):
    """Use measured whole-workflow standalone elapsed time only for evaluation."""
    expected = {str(w["workflow_id"]): len(w["calls"]) for w in workflows}
    references = {}
    for row in rows:
        wid = str(row["workflow_id"])
        if wid in references:
            raise CandidateBlocked(f"duplicate standalone workflow reference: {wid}")
        if row["reference_kind"] != "isolated_workflow_replay":
            raise CandidateBlocked("reference must be an actual isolated workflow replay, not a call critical-path proxy")
        if any(row[name] != contract[name] for name in ("model", "implementation_revision", "runtime_evidence_sha256")):
            raise CandidateBlocked("workflow reference model/version/runtime differs from frozen metadata")
        for name in ("raw_replay_sha256", "runtime_evidence_sha256"):
            value = row[name]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise CandidateBlocked("workflow reference requires frozen raw/runtime SHA256 provenance")
        count = expected.get(wid, int(row["expected_calls"]))
        if int(row["expected_calls"]) != count or count <= 0:
            raise CandidateBlocked("workflow reference expected call count differs from frozen DAG")
        latency = None
        if row["status"] == "successful":
            if row["errors"] != 0 or row["completed_calls"] != count or row["usage_verified_calls"] != count:
                raise CandidateBlocked("successful workflow reference must be complete and usage verified")
            latency = float(row["latency_ms"])
            if not math.isfinite(latency) or latency <= 0:
                raise CandidateBlocked("standalone workflow reference needs positive finite elapsed latency")
        references[wid] = {"latency_ms": latency, "status": row["status"], "row": copy.deepcopy(row)}
    return {wid: references.get(wid, {"latency_ms": None, "status": "missing", "row": None}) for wid in expected}


def _validate_dags(workflows):
    for workflow in workflows:
        calls = {str(c["call_id"]): c for c in workflow["calls"]}
        visited = set(); visiting = set()
        def visit(call_id):
            if call_id in visited:
                return
            if call_id in visiting:
                raise CandidateBlocked("cycle in frozen DAG")
            if call_id not in calls:
                raise CandidateBlocked("frozen DAG has missing parent")
            visiting.add(call_id)
            for parent in calls[call_id]["parents"]:
                visit(str(parent))
            visiting.remove(call_id); visited.add(call_id)
        for call_id in calls:
            visit(call_id)


def build_workflow_report(sim, mapping, references):
    native_records = {row["workflow_id"]: row for row in sim.workflow_records}
    output = []
    for native_id, identity in mapping.items():
        wid = identity["source_workflow_id"]; workflow = sim.workflows[native_id]
        native = native_records.get(native_id)
        completed = sum(call.call_id in sim.completed_calls and call.tool_completed for call in workflow.all_calls)
        complete = native is not None and completed == len(workflow.all_calls)
        latency_ms = float(native["duration_s"]) * 1000 if complete else None
        reference = references[wid]
        slo = reference["latency_ms"]
        output.append({"workflow_id": wid, "native_workflow_id": native_id,
                       "arrival_ms_relative": workflow.arrival_time * 1000,
                       "completion_ms_relative": float(native["completion_s"]) * 1000 if complete else None,
                       "latency_ms": latency_ms, "base_slo_ms": slo, "profile_slo_ms": slo,
                       "reference_kind": "isolated_workflow_replay",
                       "reference_status": reference["status"], "reference_raw_row": reference["row"],
                       "slo_ratio": latency_ms / slo if complete and slo is not None else None,
                       "expected_calls": len(workflow.all_calls), "completed_calls": completed,
                       "errors": 0, "status": "successful" if complete else "incomplete",
                       "native_model_slo_ms": float(native["slo_s"]) * 1000 if native else None,
                       "native_model_scaled_slo": float(native["scaled_slo"]) if native else None})
    return sorted(output, key=lambda row: row["workflow_id"])


def validate_runtime_contract(comparison, runtime):
    """Bind shared comparison metadata to parameters actually passed to native APIs."""
    expected = {"model": runtime["model"], "model_revision": runtime["model_revision"],
                "tp": runtime["model_config"]["tp_size"], "weight_dtype": "bfloat16", "kv_dtype": "bfloat16",
                "hardware": "NVDA:H20",
                "topology": {"prefill_workers": runtime["prefill_workers"],
                             "decode_workers": len(runtime["decode_capacity_tokens"]),
                             "tp_size": runtime["model_config"]["tp_size"],
                             "decode_capacity_tokens": runtime["decode_capacity_tokens"],
                             "decode_max_batch_size": runtime["decode_max_batch_size"]},
                "cache_policy": runtime["cache_policy"], "tool_policy": runtime["tool_policy"],
                "scheduler_flags": runtime["scheduler_flags"], "observation_horizon": "revealed_calls_only"}
    if any(comparison.get(key) != value for key, value in expected.items()):
        raise CandidateBlocked("comparison metadata contradicts effective native runtime parameters")
    if (runtime["prefill_workers"] != 1 or len(runtime["decode_capacity_tokens"]) != 1
            or comparison["arrival"].get("source") != "saved_real_workflow_arrival_ms"
            or comparison["arrival"].get("completion_policy") != "drain_all_expected_calls"):
        raise CandidateBlocked("effective candidate scope requires native 1P1D with saved arrivals and complete drain")


def load_calibration_document(records_path, boundary_path, calibration_paths):
    document = json.loads(Path(boundary_path).read_text())
    hashes = {str(path): file_hash(path) for path in calibration_paths}
    if document["stage_records_sha256"] != file_hash(records_path) or document["calibration_trace_sha256"] != hashes:
        raise CandidateBlocked("calibration stage/trace frozen hash mismatch")
    document["rows"] = read_jsonl(records_path)
    return document


def run_candidate(args):
    contract = json.loads(args.contract.read_text())
    comparison = contract["comparison_contract"]; runtime = contract["runtime"]
    if set(CONTRACT_FIELDS) - comparison.keys():
        raise CandidateBlocked("comparison_contract is incomplete")
    if comparison["trace_sha256"] != file_hash(args.trace) or comparison["profile_sha256"] != file_hash(args.workflow_reference):
        raise CandidateBlocked("frozen trace/workflow-reference table hash mismatch")
    if comparison["arrival"]["replay_sha256"] != file_hash(args.real_replay):
        raise CandidateBlocked("frozen saved-arrival replay hash mismatch")
    validate_runtime_contract(comparison, runtime)
    evidence = contract["runtime_evidence"]
    if file_hash(evidence["path"]) != evidence["sha256"]:
        raise CandidateBlocked("frozen observed runtime evidence hash mismatch")
    reference_contract = contract["workflow_reference_contract"]
    if (reference_contract["model"] != comparison["model"]
            or reference_contract["implementation_revision"] != runtime["implementation_revision"]
            or reference_contract["runtime_evidence_sha256"] != evidence["sha256"]):
        raise CandidateBlocked("workflow reference contract differs from frozen real runtime metadata")
    workflows = read_jsonl(args.trace)
    calibration_workflows = [workflow for path in args.calibration_trace for workflow in read_jsonl(path)]
    calibration_doc = load_calibration_document(args.calibration_records, args.calibration_boundary, args.calibration_trace)
    api = load_apis(args.sim_root, args.real_root, sim_revision=args.sim_revision, real_revision=args.real_revision)
    nominal = NominalCosts(api, runtime["model_config"], runtime["hardware_params"])
    for item in _calls_by_key(workflows + calibration_workflows).values():
        if item["planner_hints"] != {"model_config": runtime["model_config"], "hardware_params": runtime["hardware_params"]}:
            raise CandidateBlocked("runtime v2 static configuration differs from declared simulation config")
    offsets, arrival_evidence = arrivals_from_replay(workflows, read_jsonl(args.real_replay))
    fit = fit_calibration(calibration_doc, calibration_workflows, workflows, nominal)
    traces, _ = convert_workflows(workflows, api, model=comparison["model"])
    references = workflow_references(workflows, read_jsonl(args.workflow_reference), reference_contract)
    sim, mapping = run_native(api, traces, offsets, runtime, CalibratedCosts(nominal, fit))
    # Standalone elapsed references enter only reporting, never the estimator,
    # TraceCall.profiled_* fields or any scheduler horizon.
    rows = build_workflow_report(sim, mapping, references)
    return {"status": "candidate_native_adaptation", "evidence_kind": args.evidence_kind,
            "implementation_revision": api.provenance["sim_revision"],
            "comparison_contract": comparison, "workflows": rows,
            "candidate_sha256": file_hash(__file__),
            "source_provenance": api.provenance, "runtime": runtime,
            "runtime_evidence": evidence, "workflow_reference_contract": reference_contract,
            "workflow_reference_path": str(args.workflow_reference),
            "slo_definition": "isolated_workflow_replay_on_finite_1p1d",
            "input_hashes": {"workflow_reference": file_hash(args.workflow_reference),
                             "calibration_records": file_hash(args.calibration_records),
                             "calibration_boundary": file_hash(args.calibration_boundary),
                             "calibration_traces": {str(path): file_hash(path) for path in args.calibration_trace}},
            "arrival_evidence": arrival_evidence, "calibration_fit": fit,
            "source_native_id_mapping": mapping, "limitations": LIMITATIONS,
            "counts": {"expected_workflows": len(workflows), "simulated_workflows": sim.total_workflows,
                       "completed_workflows": sim.completed_workflows,
                       "expected_calls": sum(len(w["calls"]) for w in workflows),
                       "completed_calls": len(sim.completed_calls)},
            "scheduler_invocation_ms": [value * 1000 for value in sim.scheduler_times],
            "reference_use": "evaluation-only actual standalone whole-workflow elapsed latency on finite 1P1D; no isolated-call CP sum; native_model_slo fields are separate",
            "reference_validation": "frozen table hash, unique workflow IDs, complete usage-verified counts, model/runtime revision/evidence hash, positive finite successful latency; raw replay elapsed calculation is externally verified",
            "effective_observation_horizon": {"visibility": "revealed_calls_only",
                                              "completed_history_service": "predicted_estimates",
                                              "real_R2_difference": "real planner refreshes completed history with measured service"},
            "runtime_complete_parity_claimed": False,
            "test_service_oracle_used": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sim-root", "real-root", "trace", "real-replay", "calibration-records", "calibration-boundary", "workflow-reference", "contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sim-revision", required=True)
    parser.add_argument("--real-revision", required=True)
    parser.add_argument("--calibration-trace", type=Path, nargs="+", required=True)
    parser.add_argument("--evidence-kind", choices=("synthetic_tooling_check", "native_simulator_adaptation"), required=True)
    args = parser.parse_args(argv)
    try:
        report = run_candidate(args); status = 0
    except CandidateBlocked as error:
        report = {"status": "BLOCKED", "reason": str(error), "evidence_kind": args.evidence_kind}; status = 2
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2); target.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
