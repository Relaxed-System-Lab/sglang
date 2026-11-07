"""Accepted R1_REAL_CAPACITY snapshot contract regressions.

These CPU checks use an observed live KV limit and explicit occupancy under
accepted real commit 92e563764b11fad6df360352fdb232e3d547b7a8.
Physical GPU measurements and missing-snapshot behavior are outside this fixture.
"""

import importlib.util
from pathlib import Path
import sys
import types

import pytest


_PKG_ROOT = Path(__file__).resolve().parents[3] / "python" / "sglang"


def _load(module_name):
    for name, path in (
        ("sglang", _PKG_ROOT),
        ("sglang.hexgent", _PKG_ROOT / "hexgent"),
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    name = f"sglang.hexgent.{module_name}"
    spec = importlib.util.spec_from_file_location(
        name, _PKG_ROOT / "hexgent" / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def live_decode_state():
    schemas = _load("schemas")
    snapshots = _load("snapshots")
    scheduler = _load("scheduler")

    def build(*, used_tokens=400, current_time_ms=0):
        running = []
        if used_tokens:
            running.append({
                "memory_tokens": used_tokens,
                "remaining_service_tokens": 10,
            })
        request = schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="cpu-snapshot-contract-audit",
            request_text="explicit observed live capacity",
            max_tokens=1,
            hexgent=schemas.HexgentRequestMetadata(workflow_id="wf", call_id="new"),
            prefill_workers=[
                schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")
            ],
            decode_workers=[
                schemas.WorkerDescriptor(id="d0", url="http://d0", worker_type="decode")
            ],
            state=schemas.PlannerClusterState(
                current_time_ms=current_time_ms,
                worker_snapshots={"http://d0": {"snapshot": {
                    "mode": "decode",
                    "limits": {"max_total_num_tokens": 1000},
                    "queues": {},
                    "running": running,
                }}},
            ),
        )
        snapshots.apply_worker_snapshots(request, include_mutable_waiting=False)
        state = scheduler.DecodeStageState(
            current_time_ms=current_time_ms,
            capacity_max_tokens=[request.decode_workers[0].capacity_tokens],
            scheduled_intervals=request.state.decode_scheduled_intervals,
        )
        return scheduler, state

    return build


def test_empty_live_snapshot_preserves_total_capacity(live_decode_state):
    scheduler, state = live_decode_state(used_tokens=0)

    assert state.capacity_max_tokens == [1000]
    assert state.scheduled_intervals == [[]]
    assert scheduler.find_earliest_capacity_start(
        ready_at_ms=0, duration_ms=1, demand_tokens=700,
        scheduled_intervals=state.scheduled_intervals[0],
        capacity_tokens=state.capacity_max_tokens[0],
    ) == 0


def test_occupied_live_snapshot_keeps_total_capacity_with_occupancy_intervals(live_decode_state):
    _, state = live_decode_state()

    assert state.scheduled_intervals == [[(0, 101, 400)]]
    assert state.capacity_max_tokens == [1000]


def test_live_snapshot_charges_running_kv_once_when_new_demand_fits(live_decode_state):
    scheduler, state = live_decode_state()

    # At t=0 the explicit running demand and new demand use 400+300 of 1000.
    assert scheduler.find_earliest_capacity_start(
        ready_at_ms=0, duration_ms=1, demand_tokens=300,
        scheduled_intervals=state.scheduled_intervals[0],
        capacity_tokens=state.capacity_max_tokens[0],
    ) == 0


def test_live_snapshot_releases_all_occupied_capacity_at_interval_end(live_decode_state):
    scheduler, state = live_decode_state()
    end_ms = state.scheduled_intervals[0][0][1]

    # The constant stage limit is the total pool. C1's intrinsic-feasibility
    # check must therefore accept 700 after the 400-token interval expires.
    assert 700 <= state.capacity_max_tokens[0]
    assert scheduler.find_earliest_capacity_start(
        ready_at_ms=end_ms, duration_ms=1, demand_tokens=700,
        scheduled_intervals=state.scheduled_intervals[0],
        capacity_tokens=state.capacity_max_tokens[0],
    ) == end_ms


def test_later_empty_live_snapshot_restores_the_same_total_pool(live_decode_state):
    _, busy_state = live_decode_state()
    scheduler, later_state = live_decode_state(used_tokens=0, current_time_ms=101)

    assert later_state.capacity_max_tokens == [1000]
    assert later_state.scheduled_intervals == [[]]
    assert scheduler.find_earliest_capacity_start(
        ready_at_ms=101, duration_ms=1, demand_tokens=700,
        scheduled_intervals=later_state.scheduled_intervals[0],
        capacity_tokens=later_state.capacity_max_tokens[0],
    ) == 101
    assert busy_state.capacity_max_tokens == later_state.capacity_max_tokens
