import ast
import asyncio
import copy
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, Generic, List, Optional, TypeVar

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SCHEDULER_PATH = _ROOT / "python" / "sglang" / "srt" / "managers" / "scheduler.py"
_METRICS_PATH = (
    _ROOT / "python" / "sglang" / "srt" / "managers" / "scheduler_metrics_mixin.py"
)
_DECODE_PATH = _ROOT / "python" / "sglang" / "srt" / "disaggregation" / "decode.py"
_COMMUNICATOR_PATH = (
    _ROOT / "python" / "sglang" / "srt" / "managers" / "tokenizer_communicator_mixin.py"
)


def _load_functions(path: Path, names: list[str], extra_globals=None):
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        copy.deepcopy(node)
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    missing = [name for name in names if name not in {node.name for node in selected}]
    if missing:
        raise AssertionError(f"Missing functions in {path.name}: {missing}")

    ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"time": time}
    if extra_globals:
        namespace.update(extra_globals)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return [namespace[name] for name in names]


def _load_communicator_class():
    module = ast.parse(
        _COMMUNICATOR_PATH.read_text(encoding="utf-8"),
        filename=str(_COMMUNICATOR_PATH),
    )
    communicator = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "_Communicator"
    )
    future_annotations = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations")],
        level=0,
    )
    selected = ast.Module(
        body=[future_annotations, copy.deepcopy(communicator)], type_ignores=[]
    )
    ast.fix_missing_locations(selected)
    namespace = {
        "asyncio": asyncio,
        "copy": copy,
        "deque": deque,
        "Deque": Deque,
        "Generic": Generic,
        "List": List,
        "Optional": Optional,
        "T": TypeVar("T"),
        "zmq": SimpleNamespace(Socket=object),
    }
    exec(compile(selected, str(_COMMUNICATOR_PATH), "exec"), namespace)
    return namespace["_Communicator"]


def test_communicator_serializes_continuous_concurrent_calls():
    communicator_cls = _load_communicator_class()

    class Sender:
        def __init__(self):
            self.sent = []

        def send_pyobj(self, value):
            self.sent.append(value)

    async def exercise():
        sender = Sender()
        communicator = communicator_cls(sender, fan_out=1)
        tasks = []
        request_count = 100

        async def produce():
            for request_id in range(1, request_count + 1):
                tasks.append(asyncio.create_task(communicator(request_id)))
                await asyncio.sleep(0)

        async def respond():
            response_count = 0
            while response_count < request_count:
                while len(sender.sent) <= response_count:
                    await asyncio.sleep(0)
                communicator.handle_recv(response_count + 1)
                response_count += 1
                await asyncio.sleep(0)

        await asyncio.wait_for(asyncio.gather(produce(), respond()), timeout=2)
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
        assert sender.sent == list(range(1, request_count + 1))
        assert results == [[request_id] for request_id in range(1, request_count + 1)]

    asyncio.run(exercise())


def _make_req(
    rid: str,
    *,
    priority: int,
    revision: int,
    workflow_id: str = "wf-1",
    planned_decode_instance: str = "decode-a",
    queue_state: str = "prefill_waiting",
    prefill_started: bool = False,
    input_tokens: int = 8,
    output_tokens: int = 0,
    max_new_tokens: int = 32,
):
    return SimpleNamespace(
        rid=rid,
        priority=priority,
        workflow_id=workflow_id,
        hexgent_prefill_priority=priority,
        hexgent_decode_priority=priority,
        hexgent_schedule_revision=revision,
        planned_decode_instance=planned_decode_instance,
        hexgent_queue_state=queue_state,
        hexgent_queue_entered_at=1234.5,
        hexgent_prefill_started=prefill_started,
        origin_input_ids=list(range(input_tokens)),
        output_ids=list(range(output_tokens)),
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        time_stats=SimpleNamespace(forward_entry_time=0.0),
    )


def _make_decode_entry(req):
    return SimpleNamespace(req=req, waiting_for_input=False, metadata_buffer_index=-1)


def test_decode_metadata_room_allows_short_aux_arrival_race():
    (classify_room,) = _load_functions(
        _DECODE_PATH,
        ["classify_metadata_room"],
    )

    state, mismatch_since = classify_room(
        actual_room=11,
        expected_room=22,
        mismatch_since=None,
        now=100.0,
        grace_s=1.0,
    )
    assert state == "waiting"
    assert mismatch_since == 100.0

    state, mismatch_since = classify_room(
        actual_room=11,
        expected_room=22,
        mismatch_since=mismatch_since,
        now=100.5,
        grace_s=1.0,
    )
    assert state == "waiting"
    assert mismatch_since == 100.0

    state, mismatch_since = classify_room(
        actual_room=22,
        expected_room=22,
        mismatch_since=mismatch_since,
        now=100.6,
        grace_s=1.0,
    )
    assert state == "ready"
    assert mismatch_since is None


def test_decode_metadata_room_rejects_persistent_mismatch():
    (classify_room,) = _load_functions(
        _DECODE_PATH,
        ["classify_metadata_room"],
    )

    state, mismatch_since = classify_room(
        actual_room=11,
        expected_room=22,
        mismatch_since=100.0,
        now=101.0,
        grace_s=1.0,
    )
    assert state == "corrupt"
    assert mismatch_since == 100.0


def test_hexgent_stage_priority_selects_prefill_or_decode_priority():
    class FakeDisaggregationMode:
        PREFILL = "prefill"
        DECODE = "decode"

    (stage_priority,) = _load_functions(
        _SCHEDULER_PATH,
        ["hexgent_stage_priority"],
        extra_globals={"DisaggregationMode": FakeDisaggregationMode},
    )
    recv_req = SimpleNamespace(
        priority=9,
        hexgent_prefill_priority=3,
        hexgent_decode_priority=7,
    )

    assert stage_priority(FakeDisaggregationMode.PREFILL, recv_req) == 3
    assert stage_priority(FakeDisaggregationMode.DECODE, recv_req) == 7
    assert stage_priority("null", recv_req) == 9


def test_hexgent_pd_snapshot_includes_required_fields():
    (build_snapshot,) = _load_functions(
        _METRICS_PATH,
        ["build_hexgent_pd_snapshot_payload"],
    )

    bootstrap_req = _make_req(
        "pref-bootstrap",
        priority=2,
        revision=1,
        queue_state="prefill_bootstrap",
    )
    waiting_req = _make_req(
        "pref-waiting",
        priority=4,
        revision=2,
        queue_state="prefill_waiting",
    )
    inflight_req = _make_req(
        "pref-inflight",
        priority=5,
        revision=3,
        queue_state="prefill_inflight",
        prefill_started=True,
        output_tokens=1,
    )
    running_req = _make_req(
        "pref-running",
        priority=7,
        revision=4,
        queue_state="prefill_running",
        prefill_started=True,
        output_tokens=2,
    )

    snapshot = build_snapshot(
        mode="prefill",
        instance_id="prefill-instance-a",
        dp_rank=0,
        waiting_queue=[waiting_req],
        running_reqs=[running_req],
        prefill_bootstrap_queue=[bootstrap_req],
        prefill_inflight_queue=[inflight_req],
        decode_prealloc_queue=[],
        decode_transfer_queue=[],
        decode_retracted_queue=[],
        now=321.0,
    )

    assert snapshot["mode"] == "prefill"
    assert snapshot["instance_id"] == "prefill-instance-a"
    assert snapshot["queues"]["bootstrap"][0]["rid"] == "pref-bootstrap"
    assert snapshot["queues"]["waiting"][0]["queue_name"] == "prefill_waiting"
    assert snapshot["queues"]["inflight"][0]["prefill_started"] is True

    running = snapshot["running"][0]
    assert running["rid"] == "pref-running"
    assert running["workflow_id"] == "wf-1"
    assert running["current_assignment"] == "prefill-instance-a"
    assert running["current_priority_score"] == 7
    assert running["schedule_revision"] == 4
    assert running["planned_decode_instance"] == "decode-a"
    assert running["input_tokens"] == 8
    assert running["output_tokens_generated"] == 2
    assert running["max_new_tokens"] == 32
    assert running["memory_tokens"] == 10
    assert running["remaining_service_tokens"] == 30
    assert running["is_running"] is True


def test_hexgent_stage_timing_separates_execution_from_waiting_time():
    (build_stage_timing,) = _load_functions(
        _METRICS_PATH,
        ["build_hexgent_stage_timing_payload"],
    )
    req = _make_req("req-timing", priority=2, revision=4)
    req.hexgent_call_id = "call-7"
    req.time_stats = SimpleNamespace(
        prefill_start_time_host=10.000,
        prefill_end_time_host=10.024,
        decode_transfer_queue_entry_time=20.000,
        wait_queue_entry_time=20.125,
        forward_entry_time=21.000,
        completion_time=21.080,
    )

    prefill = build_stage_timing(req, mode="prefill")
    decode = build_stage_timing(req, mode="decode")

    assert prefill == {
        "workflow_id": "wf-1",
        "call_id": "call-7",
        "stage": "prefill",
        "prefill_execution_ms": 24,
    }
    assert decode == {
        "workflow_id": "wf-1",
        "call_id": "call-7",
        "stage": "decode",
        "transfer_ms": 125,
        "decode_execution_ms": 80,
    }


def test_hexgent_update_waiting_priority_reorders_queue():
    (
        update_priority,
    ) = _load_functions(
        _SCHEDULER_PATH,
        ["hexgent_update_waiting_request_priority"],
    )

    low = _make_req("req-low", priority=1, revision=1)
    high = _make_req("req-high", priority=5, revision=1)
    queue = [low, high]

    result = update_priority(
        queue,
        request_id="req-low",
        new_priority=9,
        new_revision=2,
        expected_revision=1,
        reorder=True,
        schedule_low_priority_values_first=False,
    )

    assert result["success"] is True
    assert [req.rid for req in queue] == ["req-low", "req-high"]
    assert queue[0].priority == 9
    assert queue[0].hexgent_schedule_revision == 2


def test_hexgent_priority_updates_converge_to_newest_revision_out_of_order():
    (update_priority,) = _load_functions(
        _SCHEDULER_PATH,
        ["hexgent_update_waiting_request_priority"],
    )
    req = _make_req("req", priority=1, revision=1)
    queue = [req]

    newest = update_priority(
        queue,
        request_id="req",
        new_priority=3,
        new_revision=3,
        expected_revision=None,
    )
    stale = update_priority(
        queue,
        request_id="req",
        new_priority=2,
        new_revision=2,
        expected_revision=None,
    )

    assert newest["success"] is True
    assert stale["success"] is False
    assert req.priority == 3
    assert req.hexgent_schedule_revision == 3


def test_hexgent_reassign_waiting_moves_request_without_duplicate():
    (
        reassign_request,
    ) = _load_functions(
        _SCHEDULER_PATH,
        ["hexgent_reassign_waiting_request"],
    )

    req = _make_req("req-a", priority=3, revision=1)
    source_waiting = [req]
    target_bootstrap = []

    result = reassign_request(
        source_queues={"prefill_waiting": source_waiting},
        target_queue=target_bootstrap,
        target_queue_name="prefill_bootstrap",
        request_id="req-a",
        new_priority=6,
        new_revision=2,
        expected_revision=1,
    )

    assert result["success"] is True
    assert source_waiting == []
    assert [entry.rid for entry in target_bootstrap] == ["req-a"]
    assert target_bootstrap[0].hexgent_queue_state == "prefill_bootstrap"
    assert target_bootstrap[0].priority == 6


def test_hexgent_reassign_waiting_rejects_after_prefill_started():
    (
        reassign_request,
    ) = _load_functions(
        _SCHEDULER_PATH,
        ["hexgent_reassign_waiting_request"],
    )

    req = _make_req(
        "req-locked",
        priority=3,
        revision=1,
        prefill_started=True,
        queue_state="decode_waiting",
    )
    source_waiting = [req]
    target_queue = []

    result = reassign_request(
        source_queues={"decode_waiting": source_waiting},
        target_queue=target_queue,
        target_queue_name="decode_prealloc",
        request_id="req-locked",
        new_priority=6,
        new_revision=2,
        expected_revision=1,
    )

    assert result["success"] is False
    assert source_waiting == [req]
    assert target_queue == []


def test_hexgent_reassign_waiting_rejects_stale_revision():
    (
        reassign_request,
    ) = _load_functions(
        _SCHEDULER_PATH,
        ["hexgent_reassign_waiting_request"],
    )

    req = _make_req("req-stale", priority=3, revision=5)
    source_waiting = [req]
    target_queue = []

    result = reassign_request(
        source_queues={"prefill_waiting": source_waiting},
        target_queue=target_queue,
        target_queue_name="prefill_bootstrap",
        request_id="req-stale",
        new_priority=6,
        new_revision=4,
        expected_revision=5,
    )

    assert result["success"] is False
    assert source_waiting == [req]
    assert target_queue == []


def test_hexgent_reassign_waiting_cleans_duplicate_source_entries():
    (
        reassign_request,
    ) = _load_functions(
        _SCHEDULER_PATH,
        ["hexgent_reassign_waiting_request"],
    )

    req1 = _make_req("req-dup", priority=3, revision=1)
    req2 = _make_req("req-dup", priority=3, revision=1)
    source_bootstrap = [req1]
    source_waiting = [req2]
    target_queue = [_make_req("req-dup", priority=2, revision=1)]

    result = reassign_request(
        source_queues={
            "prefill_bootstrap": source_bootstrap,
            "prefill_waiting": source_waiting,
        },
        target_queue=target_queue,
        target_queue_name="prefill_bootstrap",
        request_id="req-dup",
        new_priority=8,
        new_revision=2,
        expected_revision=1,
    )

    assert result["success"] is True
    assert source_bootstrap == []
    assert source_waiting == []
    assert [entry.rid for entry in target_queue] == ["req-dup"]
    assert target_queue[0].priority == 8
