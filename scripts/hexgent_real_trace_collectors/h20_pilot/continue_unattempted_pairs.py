#!/usr/bin/env python3
"""Manual one-shot continuation of original registered, unattempted pairs.

No action on import. Run only inside the root-selected existing runtime
container, after the original paired job has finished with NOT_RUN_budget.
No frozen driver/support/wrapper is changed and no arrival vector is generated.
"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

FROZEN_RUNNER_SHA256 = "2ade58d47148ff1abe021d700671e5dbe6eaec9cfbd1f7c71c65a11aad724077"
FROZEN_WRAPPER_SHA256 = "9e146c390e800b6af6e8d21f65b1bd421f47f9f5cecc63693b325eff561a70ab"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_stopped(job, block):
    if job.get("phase") != "finished":
        raise ValueError("original paired job must be finished")
    if block.get("stopped_reason") != "NOT_RUN_budget":
        raise ValueError("continuation only applies to original NOT_RUN_budget stop")


def fixed_deadline(block, plan):
    deadline = block["effective_driver_deadline_unix_s"]
    if deadline != plan["hard_deadline_unix_s"]:
        raise ValueError("old block and original plan deadlines differ; never refresh deadline")
    return deadline


def original_pairs(plan):
    if len(plan["runs"]) != 8 or [r["ordinal"] for r in plan["runs"]] != list(range(1, 9)):
        raise ValueError("exact original eight ordinal runs required")
    for index in range(0, 8, 2):
        pair = plan["runs"][index:index + 2]
        if (pair[0]["dataset"], pair[0]["load"]) != (pair[1]["dataset"], pair[1]["load"]):
            raise ValueError("pair cell identity differs")
        cell = next(c for c in plan["cells"] if
                    (c["dataset"], c["load"]) == (pair[0]["dataset"], pair[0]["load"]))
        yield pair, cell


def attempted_ordinals(run_dir, plan, block):
    attempted = {int(r["ordinal"]) for r in block["runs"]}
    for run in plan["runs"]:
        if any(p.name.startswith("%02d-" % run["ordinal"]) for p in Path(run_dir).iterdir()):
            attempted.add(run["ordinal"])
    return attempted


def pair_decision(pair, cell, references, *, deadline, now, attempted):
    ordinals = [run["ordinal"] for run in pair]
    if any(ordinal in attempted for ordinal in ordinals):
        return {"pair_ordinals": ordinals, "status": "INELIGIBLE_existing_attempt"}
    max_reference = max(references[wid]["latency_ms"] for wid in cell["workflow_ids"]) / 1000
    last_arrival = max(cell["arrival_offsets_s"])
    required = 1.25 * 2 * (last_arrival + max_reference) + 60
    remaining = deadline - now
    return {
        "pair_ordinals": ordinals, "dataset": pair[0]["dataset"], "load": pair[0]["load"],
        "checked_unix_s": now, "effective_deadline_unix_s": deadline,
        "remaining_s": remaining, "required_heuristic_s": required,
        "max_reference_s": max_reference, "last_planned_arrival_s": last_arrival,
        "allowed": remaining >= required,
        "status": "READY" if remaining >= required else "NOT_RUN_budget",
        "selection_inputs": "fixed same-cell standalone reference, original planned arrivals, remaining clock only",
    }


class AppendLedger:
    def __init__(self, path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
        self.file = os.fdopen(fd, "w")

    def append(self, value):
        self.file.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def exclusive_start(run_dir, original_block_bytes, metadata):
    run_dir = Path(run_dir)
    with (run_dir / "continuation.marker.json").open("x") as target:
        json.dump(metadata, target, indent=2)
        target.write("\n")
    with (run_dir / "block-results.before-continuation.json").open("xb") as target:
        target.write(original_block_bytes)
    return AppendLedger(run_dir / "continuation-decisions.jsonl")


def require_gone(pids, group_ids, *, pid_exists=None, group_exists=None):
    pid_exists = pid_exists or (lambda pid: Path("/proc", str(pid)).exists())
    if any(pid_exists(pid) for pid in sorted(pids)):
        raise RuntimeError("original/owned child PID still exists or was reused; no signal sent")
    if group_exists is None:
        def group_exists(pgid):
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return False
            return True
    if any(group_exists(pgid) for pgid in sorted(group_ids)):
        raise RuntimeError("owned child process group still exists; no signal sent")


def child_processes(run_dir, plan, block):
    records = list(block["runs"])
    unknown = []
    for run in plan["runs"]:
        paths = [p for p in Path(run_dir).iterdir() if p.name.startswith("%02d-" % run["ordinal"])]
        for path in paths:
            evidence = path / "run.json"
            if not path.is_dir() or not evidence.exists():
                unknown.append(run["ordinal"])
                continue
            record = read(evidence)
            if "owned_processes" not in record:
                unknown.append(run["ordinal"])
            records.append(record)
    pids, groups = set(), set()
    for record in records:
        for child in record.get("owned_processes", []):
            pids.add(int(child["pid"]))
            groups.add(int(child["pgid"]))
    if unknown:
        raise RuntimeError("partial ordinal child-quiescence evidence is incomplete: %s" % sorted(set(unknown)))
    return pids, groups


def validate_frozen_plan(plan, module, args):
    if (plan["seed"], plan["repeat"], plan["run_count"]) != (20261107, 0, 8):
        raise ValueError("original one-block seed/repeat/count changed")
    actual = [(r["dataset"], r["load"], r["method"]) for r in plan["runs"]]
    if actual != module.run_order():
        raise ValueError("original counterbalanced method order changed")
    for cell in plan["cells"]:
        path = Path(cell["trace"])
        if sha(path) != cell["trace_sha256"]:
            raise ValueError("original saved trace bytes changed")
        workflows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if [w["workflow_id"] for w in workflows] != cell["workflow_ids"]:
            raise ValueError("original workflow order changed")
        if [w["arrival_offset"] for w in workflows] != cell["arrival_offsets_s"]:
            raise ValueError("original saved arrival vector changed")
        source = args.runtime_input_dir / (cell["dataset"] + ".heldout.sanitized.runtime.jsonl")
        if sha(source) != cell["source_trace_sha256"]:
            raise ValueError("frozen runtime input changed")
        original = {w["workflow_id"]: w for w in
                    (json.loads(line) for line in source.read_text().splitlines() if line.strip())}
        for workflow in workflows:
            before = original[workflow["workflow_id"]]
            if {k: v for k, v in workflow.items() if k != "arrival_offset"} != {
                    k: v for k, v in before.items() if k != "arrival_offset"}:
                raise ValueError("original workflow/calls/request/hints changed")
    for run in plan["runs"]:
        cell = next(c for c in plan["cells"] if
                    (c["dataset"], c["load"]) == (run["dataset"], run["load"]))
        if any(run[key] != cell[key] for key in ("trace", "trace_sha256", "arrival_offsets_s")):
            raise ValueError("paired run no longer points to exact original trace/vector")


async def continue_pairs(module, args, plan, old_block, references, support, api,
                         expected, transfer, logs, ledger):
    records = list(old_block["runs"])
    decisions = []
    initial_attempted = attempted_ordinals(args.run_dir, plan, old_block)
    attempted = set(initial_attempted)

    def save(extra=None):
        ordinals = {r["ordinal"] for r in records}
        value = dict(old_block)
        value.update({
            "runs": records, "continuation_pair_budget_checks": decisions,
            "original_stopped_reason": old_block["stopped_reason"],
            "original_block_snapshot": "block-results.before-continuation.json",
            "continuation_ledger": "continuation-decisions.jsonl",
            "continuation": True,
            "effective_driver_deadline_unix_s": args.deadline_unix_s,
            "all_runs_attempted": len(attempted) == 8,
            "complete_balanced_block": len(records) == 8 and ordinals == set(range(1, 9))
                                       and all(r.get("valid_measurement") for r in records),
            "not_run_ordinals": [r["ordinal"] for r in plan["runs"] if r["ordinal"] not in attempted],
            "attempted_without_complete_block_row": sorted(attempted - ordinals),
        })
        if extra:
            value.update(extra)
        module.save_json(args.run_dir / "block-results.json", value)

    try:
        for pair, cell in original_pairs(plan):
            attempted |= attempted_ordinals(args.run_dir, plan, {"runs": records})
            decision = pair_decision(pair, cell, references, deadline=args.deadline_unix_s,
                                     now=time.time(), attempted=attempted)
            decisions.append(decision)
            ledger.append({"event": "PAIR_DECISION", **decision})
            save()
            if decision["status"] != "READY":
                continue
            for run in pair:
                if run["ordinal"] in attempted_ordinals(args.run_dir, plan, {"runs": records}):
                    raise RuntimeError("ordinal became attempted after budget decision; no rerun")
                module.ensure_time(args.deadline_unix_s)
                ledger.append({"event": "RUN_ATTEMPTING", "ordinal": run["ordinal"],
                               "checked_unix_s": time.time(), "effective_deadline_unix_s": args.deadline_unix_s})
                attempted.add(run["ordinal"])
                save()
                record = await module.execute_run(
                    args, run, cell, support, api, expected, transfer, logs, plan["reference_contract"]["model"])
                records.append(record)
                ledger.append({"event": "RUN_RETURNED", "ordinal": run["ordinal"],
                               "valid_measurement": record.get("valid_measurement"),
                               "finished_unix_s": time.time()})
                save()
        save({"continuation_finished": True, "continuation_finished_unix_s": time.time()})
    except BaseException as exc:
        ledger.append({"event": "CONTINUATION_FAILED", "error": repr(exc), "recorded_unix_s": time.time()})
        save({"continuation_error": repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/src"))
    parser.add_argument("--runtime-input-dir", type=Path, default=Path("/inputs/runtime-inputs-v2"))
    parser.add_argument("--original-driver-pid", type=int, required=True)
    parser.add_argument("--container-id", required=True,
                        help="Root must bind this manual invocation to the exact existing container via docker exec.")
    parser.add_argument("--execute", action="store_true", required=True)
    cli = parser.parse_args()
    root = cli.run_root
    prep = Path(__file__).resolve().parent
    run_dir = root / "paired-pilot"
    old_bytes = (run_dir / "block-results.json").read_bytes()
    old = json.loads(old_bytes)
    plan = read(run_dir / "run-plan.json")
    validate_stopped(read(root / "paired-job.json"), old)
    deadline = fixed_deadline(old, plan)
    guard_path = root / "original-driver-process-guard.json"
    guard = read(guard_path)
    if guard["original_driver"]["pid"] != cli.original_driver_pid:
        raise ValueError("explicit original driver PID differs from captured process guard")
    launch_path = root.parent / "reference-r2-container-launch.json"
    launch = read(launch_path)
    if launch["container_id"] != cli.container_id:
        raise ValueError("root-selected container ID differs from actual launch manifest")
    amendment_path = root / "protocol-budget-amendment.json"
    amendment = read(amendment_path)
    if amendment["current_primary_driver_sha256"] != FROZEN_WRAPPER_SHA256:
        raise ValueError("budget amendment does not bind original root wrapper")
    if sha(prep / "run_budgeted_block.py") != FROZEN_WRAPPER_SHA256:
        raise ValueError("original root wrapper changed")
    runner_path = prep / "run_paired_pilot.py"
    if sha(runner_path) != FROZEN_RUNNER_SHA256:
        raise ValueError("frozen runner changed")
    spec = importlib.util.spec_from_file_location("continuation_frozen_pilot", runner_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = types.SimpleNamespace(
        source_root=cli.source_root, runtime_input_dir=cli.runtime_input_dir,
        reference_dir=root / "reference-collection", run_dir=run_dir, repeat=0,
        deadline_unix_s=deadline)
    validate_frozen_plan(plan, module, args)
    if module.validate_inputs(args) != plan["reference_contract"]:
        raise ValueError("frozen reference contract changed")
    pids, groups = child_processes(run_dir, plan, old)
    pids.add(cli.original_driver_pid)
    require_gone(pids, groups)
    module.require_free_ports()
    module.ensure_time(deadline)
    metadata = {
        "created_unix_s": time.time(), "controller_pid": os.getpid(),
        "effective_deadline_unix_s": deadline, "original_block_sha256": hashlib.sha256(old_bytes).hexdigest(),
        "original_plan_sha256": sha(run_dir / "run-plan.json"),
        "original_process_guard_sha256": sha(guard_path), "original_driver_pid": cli.original_driver_pid,
        "checked_absent_pids": sorted(pids), "checked_absent_owned_pgids": sorted(groups),
        "root_selected_container_id": cli.container_id, "launch_manifest_sha256": sha(launch_path),
        "actual_pid_namespace": os.readlink("/proc/self/ns/pid"),
        "amendment_sha256": sha(amendment_path), "frozen_runner_sha256": FROZEN_RUNNER_SHA256,
        "selection": "Original ordinal order; insufficient registered pair skipped, later unattempted pairs checked using clock/reference formula only.",
    }
    ledger = exclusive_start(run_dir, old_bytes, metadata)
    try:
        ledger.append({"event": "CONTINUATION_CLAIMED", **metadata})
        expected = module.read_json(os.environ["H20_EXPECTED_LIMITS_JSON"])
        module.wait_idle(run_dir / "idle-before-continuation.json", expected, deadline)
        # Recheck the same known process guards after the idle observations.
        require_gone(pids, groups)
        module.require_free_ports()
        support = module.load_module("h20_run_support", prep / "h20_run_support.py")
        api = support.load_workflow_api(args.source_root)
        transfer = module.read_json(os.environ["TRANSFER_PARAMETERS_JSON"])
        logs = json.loads(os.environ["H20_WORKER_LOG_PATHS_JSON"])
        if set(logs) != {"prefill", "decode"}:
            raise ValueError("exact original worker log paths required")
        references = {r["workflow_id"]: r for r in
                      (json.loads(line) for line in
                      (args.reference_dir / "workflow-references.jsonl").read_text().splitlines() if line.strip())}
        asyncio.run(continue_pairs(module, args, plan, old, references, support, api,
                                   expected, transfer, logs, ledger))
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
