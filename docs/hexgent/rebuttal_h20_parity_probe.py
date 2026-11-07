"""CPU-only, standalone parity evidence; does not launch or measure GPU serving.

Run from the H20 worktree with an explicit simulator source, for example:
python docs/hexgent/rebuttal_h20_parity_probe.py --simulator-core D:/Github/HexAGenT/pd_disaggregated/sglang_scheduler_core.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types


FLAG_NAMES = ("HEXGENT_PROJECTED_RATIO_ORDER", "HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH")


def load_core(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def decision_probe(module, order, pairwise):
    for key, value in zip(FLAG_NAMES, (order, pairwise)):
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    scheduler = module.AgenticPDExactScheduler()
    calls = []
    for index, completion in enumerate((900, 100)):
        workflow_id = f"wf-{index}"
        scheduler.on_workflow_arrival(workflow_id, 0, 1000)
        calls.append(module.PrefillCallInfo(
            call_id=("high-risk", "low-risk")[index], workflow_id=workflow_id,
            input_tokens=1000, output_tokens=100, workflow_arrival_ms=0,
            reveal_sequence=index, workflow_horizon_ms=1000,
            workflow_completion_ms=completion, workflow_attained_service_ms=0,
            ideal_service_ms=110, prefill_ms_by_instance=[10],
            decode_ms_by_instance=[100], transfer_ms_by_decode_instance=[0],
            child_count=1,
        ))
    decisions = scheduler.schedule_prefill(calls, module.PrefillStageState(
        current_time_ms=0, instance_available_at_ms=[0], decode_available_hint_ms=[0],
    ))
    stats = scheduler.get_stats()
    return {
        "flags": dict(zip(FLAG_NAMES, (order, pairwise))),
        "effective_high_first": module._projected_ratio_priority(2) < module._projected_ratio_priority(1),
        "effective_pairwise": module._pairwise_local_search_enabled(),
        "decisions": [vars(decision) for decision in decisions],
        "pair_evaluations": stats["prefill_local_search_evaluations"],
        "pair_swaps": stats["prefill_local_search_swaps"],
    }


def horizon_probe(real_root, services):
    # Import only the standalone planner modules, avoiding the GPU package init.
    package_root = real_root / "python" / "sglang"
    for name, path in (("sglang", package_root), ("sglang.hexgent", package_root / "hexgent")):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    from sglang.hexgent.planner import HexgentPlanner
    from sglang.hexgent.schemas import (
        HexgentRequestMetadata, PlannerAdmissionPolicy, PlannerAdmissionRequest,
        PlannerClusterState, PlannerScheduleRequest, WorkerDescriptor,
    )
    planner = HexgentPlanner()
    captured = []
    original_schedule = planner.scheduler.schedule_prefill

    def capture(calls, state):
        captured.append([
            {"call_id": call.call_id, "workflow_horizon_ms": call.workflow_horizon_ms,
             "ideal_service_ms": call.ideal_service_ms}
            for call in calls
        ])
        return original_schedule(calls, state)

    planner.scheduler.schedule_prefill = capture
    for service in services:
        request = PlannerScheduleRequest(
            route="/v1/chat/completions", model="cpu-audit", request_text="audit", max_tokens=1,
            hexgent=HexgentRequestMetadata(
                workflow_id="same-workflow", call_id=f"root-{service}",
                workflow_arrival_ms=0, input_tokens=1000, predicted_output_tokens=1,
                estimated_prefill_time=service, estimated_decode_time=1, estimated_transfer_time=0,
            ),
            prefill_workers=[WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")],
            decode_workers=[WorkerDescriptor(id="d0", url="http://d0", worker_type="decode", capacity_tokens=10000)],
            state=PlannerClusterState(current_time_ms=0),
        )
        planner.enqueue_admission(PlannerAdmissionRequest(
            request=request, policy=PlannerAdmissionPolicy(max_prefill_queue_per_worker=4),
        ))
    result = planner.plan_pending_admissions()
    return {
        "enqueue_prefill_services_ms": list(services), "captured_prefill_inputs": captured,
        "final_revealed_horizon_ms": planner._workflows["same-workflow"].horizon_ms,
        "published_plan_horizons_ms": [item.schedule.workflow_horizon_ms for item in result.values()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-core", type=Path, required=True)
    args = parser.parse_args()
    real_root = Path(__file__).resolve().parents[2]
    paths = {"real": real_root / "python/sglang/hexgent/scheduler.py", "simulator": args.simulator_core.resolve()}
    original_flags = {key: os.environ.get(key) for key in FLAG_NAMES}
    original_revision_path = os.environ.pop("HEXGENT_REVISION_STATE_PATH", None)
    try:
        text_by_source = {key: path.read_text(encoding="utf-8") for key, path in paths.items()}
        report = {
            "provenance": "CPU-only audit of source and deterministic decisions; no GPU/remote experiment",
            "source_paths": {key: str(path) for key, path in paths.items()},
            "normalized_source_sha256": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in text_by_source.items()},
            "normalized_source_equal": text_by_source["real"] == text_by_source["simulator"],
            "original_local_env": original_flags,
            "decision_probes": {},
        }
        for key, path in paths.items():
            module = load_core(f"audit_{key}", path)
            report["decision_probes"][key] = [
                decision_probe(module, order, pairwise)
                for order, pairwise in ((None, None), ("high_first", "0"), ("low_first", "0"), ("high_first", "1"))
            ]
        report["sibling_horizon_probes"] = [horizon_probe(real_root, services) for services in ((10, 100), (100, 10))]
        print(json.dumps(report, indent=2))
    finally:
        for key, value in original_flags.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if original_revision_path is not None:
            os.environ["HEXGENT_REVISION_STATE_PATH"] = original_revision_path


if __name__ == "__main__":
    main()
