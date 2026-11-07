"""Pure synthetic continuation tests; no process, socket, HTTP, or GPU work."""
import importlib.util
import json
from pathlib import Path

import pytest


def load():
    path = Path(__file__).with_name("continue_unattempted_pairs.py")
    spec = importlib.util.spec_from_file_location("h20_continuation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_plan():
    runs = []
    cells = []
    for dataset, load_name, methods, offsets in [
        ("sharegpt", "low", ["native", "ours"], [0, 300]),
        ("sharegpt", "burst", ["ours", "native"], [0, 0]),
        ("hotpotqa", "low", ["ours", "native"], [0, 400]),
        ("hotpotqa", "burst", ["native", "ours"], [0, 0]),
    ]:
        cells.append({"dataset": dataset, "load": load_name, "workflow_ids": [dataset],
                      "arrival_offsets_s": offsets})
        for method in methods:
            runs.append({"ordinal": len(runs) + 1, "dataset": dataset, "load": load_name, "method": method})
    return {"runs": runs, "cells": cells, "hard_deadline_unix_s": 5000}


def test_skipped_budget_pair_does_not_hide_later_registered_pair():
    m = load()
    plan = fixture_plan()
    refs = {"sharegpt": {"latency_ms": 80000}, "hotpotqa": {"latency_ms": 100000}}
    decisions = []
    for pair, cell in m.original_pairs(plan):
        decisions.append(m.pair_decision(pair, cell, refs, deadline=5000, now=4600, attempted={1, 2, 3, 4}))
    assert [d["status"] for d in decisions] == [
        "INELIGIBLE_existing_attempt", "INELIGIBLE_existing_attempt", "NOT_RUN_budget", "READY"]
    assert decisions[2]["pair_ordinals"] == [5, 6]
    assert decisions[3]["pair_ordinals"] == [7, 8]
    assert decisions[3]["required_heuristic_s"] == 310


def test_same_formula_deadline_and_within_pair_order_are_preserved():
    m = load()
    plan = fixture_plan()
    pair, cell = list(m.original_pairs(plan))[0]
    decision = m.pair_decision(pair, cell, {"sharegpt": {"latency_ms": 80000}},
                              deadline=5000, now=4000, attempted=set())
    assert decision["required_heuristic_s"] == 1.25 * 2 * (300 + 80) + 60
    assert decision["remaining_s"] == 1000
    assert [run["method"] for run in pair] == ["native", "ours"]
    assert m.fixed_deadline({"effective_driver_deadline_unix_s": 5000}, plan) == 5000
    with pytest.raises(ValueError):
        m.fixed_deadline({"effective_driver_deadline_unix_s": 5001}, plan)


def test_partial_directory_is_attempted_and_entire_pair_ineligible(tmp_path):
    m = load()
    plan = fixture_plan()
    (tmp_path / "03-sharegpt-burst-ours").mkdir()
    attempted = m.attempted_ordinals(tmp_path, plan, {"runs": [{"ordinal": 1}, {"ordinal": 2}]})
    assert attempted == {1, 2, 3}
    pair, cell = list(m.original_pairs(plan))[1]
    assert m.pair_decision(pair, cell, {"sharegpt": {"latency_ms": 1}},
                           deadline=5000, now=0, attempted=attempted)["status"] == "INELIGIBLE_existing_attempt"


def test_any_same_ordinal_artifact_is_not_rerun_even_if_not_a_directory(tmp_path):
    m = load()
    plan = fixture_plan()
    (tmp_path / "07-partial").write_text("partial")
    assert 7 in m.attempted_ordinals(tmp_path, plan, {"runs": []})


def test_one_marker_exact_snapshot_and_append_only_ledger(tmp_path):
    m = load()
    old = b'{ "runs": [], "stopped_reason": "NOT_RUN_budget" }\n'
    metadata = {"effective_deadline_unix_s": 5000}
    ledger = m.exclusive_start(tmp_path, old, metadata)
    ledger.append({"status": "NOT_RUN_budget", "pair_ordinals": [5, 6]})
    ledger.append({"status": "READY", "pair_ordinals": [7, 8]})
    ledger.close()
    assert (tmp_path / "block-results.before-continuation.json").read_bytes() == old
    entries = [json.loads(line) for line in (tmp_path / "continuation-decisions.jsonl").read_text().splitlines()]
    assert [entry["status"] for entry in entries] == ["NOT_RUN_budget", "READY"]
    with pytest.raises(FileExistsError):
        m.exclusive_start(tmp_path, old, metadata)
    assert (tmp_path / "block-results.before-continuation.json").read_bytes() == old


def test_active_original_job_or_nonbudget_stop_is_rejected():
    m = load()
    with pytest.raises(ValueError):
        m.validate_stopped({"phase": "running"}, {"stopped_reason": "NOT_RUN_budget"})
    with pytest.raises(ValueError):
        m.validate_stopped({"phase": "finished"}, {"stopped_reason": "error"})
    m.validate_stopped({"phase": "finished"}, {"stopped_reason": "NOT_RUN_budget"})


def test_pid_reuse_or_live_owned_group_blocks_without_signalling():
    m = load()
    calls = []
    with pytest.raises(RuntimeError):
        m.require_gone({904}, {1200}, pid_exists=lambda pid: pid == 904,
                       group_exists=lambda pgid: calls.append(pgid) or False)
    assert calls == []
    with pytest.raises(RuntimeError):
        m.require_gone({904}, {1200}, pid_exists=lambda pid: False,
                       group_exists=lambda pgid: True)
    m.require_gone({904}, {1200}, pid_exists=lambda pid: False, group_exists=lambda pgid: False)


def test_outcome_values_cannot_change_prospective_budget_selection():
    m = load()
    plan = fixture_plan()
    pair, cell = list(m.original_pairs(plan))[3]
    refs = {"hotpotqa": {"latency_ms": 100000}}
    baseline = m.pair_decision(pair, cell, refs, deadline=5000, now=4600, attempted={1, 2, 3, 4})
    plan["prior_pressure_results"] = {"boost": -999, "winner": "native", "mae": 999999}
    changed = m.pair_decision(pair, cell, refs, deadline=5000, now=4600, attempted={1, 2, 3, 4})
    assert changed == baseline
