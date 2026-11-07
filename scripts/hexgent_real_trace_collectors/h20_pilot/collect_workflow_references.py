#!/usr/bin/env python3
"""Collect native finite-1P1D workflow references after an explicit runtime start.

The four calibration workflows form one zero-offset burst for uniform warmup
and load calibration only. Held-out workflows run one at a time; original DAG
fanout remains concurrent through the actual WorkflowTraceReplayer API.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import h20_run_support as support

DATASETS = ("sharegpt", "hotpotqa")


def idle_check(path, expected, deadline):
    # One implementation and evidence contract for reference and formal runs.
    from run_paired_pilot import wait_idle
    return wait_idle(path, expected, deadline)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2); target.write("\n")


def load_split(api, directory, split):
    result = []
    for dataset in DATASETS:
        path = directory / f"{dataset}.{split}.sanitized.runtime.jsonl"
        workflows = sorted(api.load_workflow_traces_jsonl(path), key=lambda workflow: workflow.workflow_id)
        for workflow in workflows:
            for call in workflow.calls:
                if set(call.planner_hints) != {"model_config", "hardware_params"}:
                    raise ValueError("only frozen static model_config/hardware_params hints are accepted")
            result.append((dataset, replace(workflow, arrival_offset=0.0)))
    return result


async def replay_phase(api, workflows, *, raw_path, router_url, model, worker_class, run_namespace):
    error = None
    try:
        await support.replay_to_raw(api, workflows, raw_path=raw_path, router_url=router_url,
                                    model=model, worker_class=worker_class, run_namespace=run_namespace)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    return read_jsonl(raw_path), error


def calibration_summary(calibration, rows, *, raw_path, phase_error):
    datasets = {}
    for dataset in DATASETS:
        workflows = [workflow for label, workflow in calibration if label == dataset]
        gates = [{"workflow_id": workflow.workflow_id,
                  **support.validate_workflow_rows(workflow, [row for row in rows if row["workflow_id"] == workflow.workflow_id])}
                 for workflow in workflows]
        complete = phase_error is None and all(gate["successful"] for gate in gates)
        first = min((gate["workflow_arrival_ms"] for gate in gates if gate["workflow_arrival_ms"] is not None), default=None)
        last = max((gate["response_completed_at_ms"] for gate in gates if gate["response_completed_at_ms"] is not None), default=None)
        datasets[dataset] = {"workflow_ids": [workflow.workflow_id for workflow in workflows],
                             "workflow_count": len(workflows), "successful_workflows": sum(gate["successful"] for gate in gates),
                             "expected_calls": sum(gate["expected_calls"] for gate in gates),
                             "completed_calls": sum(gate["completed_calls"] for gate in gates),
                             "usage_verified_calls": sum(gate["usage_verified_calls"] for gate in gates),
                             "errors": sum(gate["errors"] for gate in gates),
                             "first_workflow_arrival_ms": first, "last_response_completed_at_ms": last,
                             "makespan_ms": last - first if complete else None,
                             "all_workflows_complete": complete, "workflows": gates}
    return {"phase": "calibration_warmup_load_only", "service_fit_input": False,
            "mixed_trace_cohort": True, "cross_trace_concurrency": True,
            "workflow_ids": [workflow.workflow_id for _, workflow in calibration],
            "makespan_interpretation": "per-dataset window during one concurrent mixed-trace calibration burst; not pure capacity",
            "raw_replay_path": str(raw_path), "raw_replay_sha256": sha256(raw_path),
            "scheduled_arrival_offsets_s": [0.0] * len(calibration),
            "replay_exception": phase_error, "datasets": datasets,
            "all_workflows_complete": all(value["all_workflows_complete"] for value in datasets.values())}


async def collect_references(api, *, runtime_input_dir, output_dir, router_url, namespace, model,
                             worker_class, implementation_revision, runtime_evidence_path, expected_limits_path):
    runtime_input_dir = Path(runtime_input_dir); output_dir = Path(output_dir)
    evidence_hash = sha256(runtime_evidence_path)
    expected_limits = json.loads(Path(expected_limits_path).read_text())
    calibration = load_split(api, runtime_input_dir, "calibration")
    heldout = load_split(api, runtime_input_dir, "heldout")
    if len(calibration) != 4 or len(heldout) != 8:
        raise ValueError("the frozen collection requires four calibration and eight held-out workflows")
    workflow_ids = [workflow.workflow_id for _, workflow in calibration + heldout]
    if len(set(workflow_ids)) != len(workflow_ids):
        raise ValueError("collection workflow IDs must be unique across the frozen splits")
    if not namespace or not model or not implementation_revision:
        raise ValueError("namespace, served model and frozen implementation revision are required")
    output_dir.mkdir(parents=True, exist_ok=False)
    reference_contract = {"model": model, "implementation_revision": implementation_revision,
                          "runtime_evidence_sha256": evidence_hash}
    write_json(output_dir / "workflow-reference-contract.json", reference_contract)
    write_json(output_dir / "run-plan.json", {
        "implementation_revision": implementation_revision, "model": model, "worker_class": worker_class,
        "runtime_evidence_path": str(runtime_evidence_path), "runtime_evidence_sha256": evidence_hash,
        "expected_limits_path": str(expected_limits_path), "expected_limits_sha256": sha256(expected_limits_path),
        "idle_helper_path": str(Path(__file__).with_name("run_paired_pilot.py")),
        "idle_helper_sha256": sha256(Path(__file__).with_name("run_paired_pilot.py")),
        "idle_evidence_contract": "same paired-run wait_idle: two empty snapshots 0.25s apart, maximum30s, frozen limits",
        "workflow_api_path": str(api.__file__), "workflow_api_sha256": sha256(api.__file__),
        "producer_sha256": sha256(__file__), "support_sha256": sha256(support.__file__),
        "python": sys.version, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "planned_method_policy": {"router": "native", "policy": "fixed_round_robin", "pd_pairs": 1,
                                  "runtime_flags_link": str(runtime_evidence_path)},
        "router_url": router_url, "namespace": namespace, "profile_table": None,
        "exact_output_length": True, "request_timeout_s": 180,
        "calibration_scope": "uniform warmup plus load calibration only; never service-fit input",
        "heldout_order": [workflow.workflow_id for _, workflow in heldout],
        "heldout_concurrency": "one workflow at a time, original within-workflow DAG fanout",
        "raw_logging": "per-call synchronous append and flush after native record creation, before child release",
        "input_sha256": {path.name: sha256(path) for dataset in DATASETS
                         for split in ("calibration", "heldout")
                         for path in [runtime_input_dir / f"{dataset}.{split}.sanitized.runtime.jsonl"]},
        "formal_execution": "not launched by this collector; root reviews completion gate"})
    references = []
    blocked_phase = None; blocking_error = None
    with (output_dir / "workflow-references.jsonl").open("x", encoding="utf-8") as table:
        raw_path = output_dir / "calibration-burst.raw.jsonl"
        rows, phase_error = await replay_phase(api, [workflow for _, workflow in calibration], raw_path=raw_path,
            router_url=router_url, model=model, worker_class=worker_class, run_namespace=f"{namespace}:calibration-burst")
        warmup = calibration_summary(calibration, rows, raw_path=raw_path, phase_error=phase_error)
        write_json(output_dir / "calibration-burst.summary.json", warmup)
        if warmup["all_workflows_complete"]:
            try:
                blocked_phase = "idle-after-calibration"
                idle_check(output_dir / "idle-after-calibration.json", expected_limits, time.time() + 30)
                for index, (_, workflow) in enumerate(heldout):
                    blocked_phase = f"idle-before-standalone-{index:02d}"
                    idle_check(output_dir / f"idle-before-standalone-{index:02d}.json", expected_limits, time.time() + 30)
                    blocked_phase = None
                    raw_path = output_dir / f"standalone-{index:02d}.raw.jsonl"
                    rows, phase_error = await replay_phase(api, [workflow], raw_path=raw_path, router_url=router_url,
                        model=model, worker_class=worker_class,
                        run_namespace=f"{namespace}:standalone:{index:02d}:{workflow.workflow_id}")
                    reference = support.workflow_reference_from_rows(workflow, rows, model=model,
                        implementation_revision=implementation_revision, runtime_evidence_sha256=evidence_hash,
                        raw_replay_sha256=sha256(raw_path))
                    if phase_error is not None:
                        reference["status"] = "failed"; reference["latency_ms"] = None
                        reference["failure_reasons"].append(f"replay_exception: {phase_error}")
                    table.write(json.dumps(reference, ensure_ascii=False) + "\n"); table.flush()
                    references.append(reference)
            except Exception as failure:
                blocking_error = failure
    successful = sum(row["status"] == "successful" for row in references)
    gate = {"expected_calibration_workflows": 4, "calibration_workflows_complete": warmup["all_workflows_complete"],
            "expected_heldout_workflows": 8, "recorded_heldout_workflows": len(references),
            "successful_heldout_workflows": successful,
            "all_heldout_references_complete": len(references) == 8 and successful == 8,
            "workflow_reference_sha256": sha256(output_dir / "workflow-references.jsonl"),
            "blocked_phase": blocked_phase if blocking_error is not None else None,
            "blocking_error": repr(blocking_error) if blocking_error is not None else None,
            "reference_contract": reference_contract,
            "formal_execution": "requires root review and separate manual start"}
    write_json(output_dir / "collection-gate.json", gate)
    if blocking_error is not None:
        raise blocking_error
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "runtime-input-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    api = support.load_workflow_api(args.source_root)
    gate = asyncio.run(collect_references(api, runtime_input_dir=args.runtime_input_dir, output_dir=args.output_dir,
        router_url=args.router_url, namespace=args.namespace, model=os.environ["SERVED_MODEL"],
        worker_class=os.environ.get("WORKER_CLASS", "H20-BF16-TP2"),
        implementation_revision=os.environ["IMPLEMENTATION_REVISION"],
        runtime_evidence_path=Path(os.environ["RUNTIME_EVIDENCE_JSON"]),
        expected_limits_path=Path(os.environ["H20_EXPECTED_LIMITS_JSON"])))
    print(json.dumps(gate, indent=2))
    return 0 if gate["all_heldout_references_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
