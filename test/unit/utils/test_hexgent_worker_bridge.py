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
_METRICS_PATH = _ROOT / "python" / "sglang" / "srt" / "managers" / "hexgent_worker.py"
_DECODE_PATH = _ROOT / "python" / "sglang" / "srt" / "disaggregation" / "decode.py"
_COMMUNICATOR_PATH = _ROOT / "python" / "sglang" / "srt" / "managers" / "communicator.py"
_SCHEDULER_PATH = _METRICS_PATH
_ACTUAL_SCHEDULER_PATH = _ROOT / "python" / "sglang" / "srt" / "managers" / "scheduler.py"


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
        if isinstance(node, ast.ClassDef) and node.name == "FanOutCommunicator"
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
    return namespace["FanOutCommunicator"]


def test_communicator_serializes_continuous_concurrent_calls():
    communicator_cls = _load_communicator_class()

    class Sender:
        def __init__(self):
            self.sent = []

        def send_pyobj(self, value):
            self.sent.append(value)

    async def exercise():
        sender = Sender()
        communicator = communicator_cls(sender.send_pyobj, fan_out=1)
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
    prefill_req = _make_req("req-timing-prefill", priority=2, revision=4)
    prefill_req.hexgent_call_id = "call-7"
    prefill_req.time_stats = SimpleNamespace(
        forward_entry_time=10.000,
        prefill_finished_time=10.024,
        prefill_transfer_queue_entry_time=10.026,
        completion_time=10.030,
    )
    decode_req = _make_req("req-timing-decode", priority=2, revision=4)
    decode_req.hexgent_call_id = "call-7"
    decode_req.time_stats = SimpleNamespace(
        decode_transfer_queue_entry_time=20.000,
        wait_queue_entry_time=20.125,
        forward_entry_time=21.000,
        completion_time=21.080,
    )

    prefill = build_stage_timing(prefill_req, mode="prefill")
    decode = build_stage_timing(decode_req, mode="decode")

    assert prefill == {
        "workflow_id": "wf-1",
        "call_id": "call-7",
        "stage": "prefill",
        "prefill_execution_ms": 24,
        "prefill_service_prefix_ms": pytest.approx(26),
        "prefill_transfer_tail_ms": pytest.approx(4),
        "prefill_service_prefix_boundary": "forward_entry_to_last_chunk_enqueue_return",
        "prefill_transfer_tail_boundary": "last_chunk_enqueue_return_to_completion",
    }
    assert decode == {
        "workflow_id": "wf-1",
        "call_id": "call-7",
        "stage": "decode",
        "decode_pipeline_wait_ms": pytest.approx(125),
        "decode_pipeline_wait_boundary": "decode_transfer_queue_entry_to_wait_queue_entry",
        "decode_execution_ms": pytest.approx(80),
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


def _load_actual_scheduler(queues):
    """Execute the real wrapper methods without importing GPU runtime modules."""
    tree = ast.parse(_ACTUAL_SCHEDULER_PATH.read_text(encoding="utf-8"))
    scheduler = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    names = {"update_hexgent_waiting_priority", "reassign_hexgent_waiting"}
    methods = [copy.deepcopy(node) for node in scheduler.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, *methods], type_ignores=[])
    ast.fix_missing_locations(module)

    def output(**fields):
        return SimpleNamespace(**{
            "reason_code": None, "current_revision": None,
            "applied_revision": None, "queue_name": None, **fields,
        })

    (helper,) = _load_functions(_METRICS_PATH, ["hexgent_update_waiting_request_priority"])
    namespace = {
        "hexgent_update_waiting_request_priority": helper,
        "HexgentUpdateWaitingPriorityReqInput": SimpleNamespace,
        "HexgentUpdateWaitingPriorityReqOutput": output,
        "HexgentReassignWaitingReqOutput": output,
    }
    exec(compile(module, str(_ACTUAL_SCHEDULER_PATH), "exec"), namespace)
    scheduler_type = type("ExtractedScheduler", (), {name: namespace[name] for name in names})
    instance = scheduler_type()
    instance._hexgent_mutable_queues = lambda: queues
    instance.schedule_low_priority_values_first = False
    return instance


def _priority_input(*, revision=4, expected=None, queue_name=None, request_id="req"):
    return SimpleNamespace(
        request_id=request_id, priority=9, schedule_revision=revision,
        expected_revision=expected, queue_name=queue_name,
        planned_decode_instance="decode-b",
    )


@pytest.mark.parametrize(
    ("new_revision", "expected", "code", "current"),
    [(6, 4, "expected_revision_mismatch", 5), (4, None, "stale_revision", 5),
     (6, 5, "success", 6)],
)
def test_priority_helper_reports_revision_reason_without_mutating_rejections(
    new_revision, expected, code, current
):
    (helper,) = _load_functions(_METRICS_PATH, ["hexgent_update_waiting_request_priority"])
    req = _make_req("req", priority=3, revision=5)
    before = copy.deepcopy(vars(req))

    result = helper(
        [req], request_id="req", new_priority=9, new_revision=new_revision,
        expected_revision=expected, planned_decode_instance="decode-b",
    )
    assert result["reason_code"] == code
    assert result["current_revision"] == current
    if code != "success":
        assert result["success"] is False
        assert vars(req) == before
    else:
        assert result["success"] is True
        assert req.priority == 9
        assert req.hexgent_schedule_revision == 6


def test_priority_helper_not_found_has_no_stale_revision():
    (helper,) = _load_functions(_METRICS_PATH, ["hexgent_update_waiting_request_priority"])

    result = helper([], request_id="missing", new_priority=9, new_revision=4)
    assert result["success"] is False
    assert result["reason_code"] == "not_found"
    assert result["current_revision"] is None


@pytest.mark.parametrize(
    ("revision", "expected", "code"),
    [(4, None, "stale_revision"), (6, 4, "expected_revision_mismatch")],
)
def test_actual_scheduler_preserves_rejection_after_scanning_other_queues(revision, expected, code):
    req = _make_req("req", priority=3, revision=5)
    before = copy.deepcopy(vars(req))
    scheduler = _load_actual_scheduler([
        ("decode_prealloc", [req], False), ("decode_waiting", [], True),
    ])

    result = scheduler.update_hexgent_waiting_priority(
        _priority_input(revision=revision, expected=expected)
    )
    assert result.success is False
    assert result.reason_code == code
    assert result.current_revision == 5
    assert result.queue_name == "decode_prealloc"
    assert vars(req) == before


def test_actual_scheduler_reports_not_found_without_claiming_stale():
    scheduler = _load_actual_scheduler([("prefill_waiting", [], True)])

    result = scheduler.update_hexgent_waiting_priority(_priority_input())
    assert result.success is False
    assert result.reason_code == "not_found"
    assert result.current_revision is None


def test_actual_scheduler_keeps_later_success_over_earlier_rejection():
    rejected = _make_req("req", priority=3, revision=5)
    accepted = _make_req("req", priority=2, revision=3)
    before_rejected = copy.deepcopy(vars(rejected))
    scheduler = _load_actual_scheduler([
        ("decode_prealloc", [rejected], False), ("decode_waiting", [accepted], True),
    ])

    result = scheduler.update_hexgent_waiting_priority(_priority_input(revision=4))
    assert result.success is True
    assert result.reason_code == "success"
    assert result.current_revision == result.applied_revision == 4
    assert result.queue_name == "decode_waiting"
    assert vars(rejected) == before_rejected
    assert accepted.hexgent_schedule_revision == 4
    assert accepted.priority == 9


@pytest.mark.parametrize(
    ("revision", "expected", "code", "current"),
    [(4, None, "stale_revision", 5), (6, 4, "expected_revision_mismatch", 5),
     (6, 5, "success", 6)],
)
def test_actual_reassign_endpoint_wrapper_preserves_priority_outcome(revision, expected, code, current):
    req = _make_req("req", priority=3, revision=5)
    queue = [req]
    before = copy.deepcopy(vars(req))
    scheduler = _load_actual_scheduler([("prefill_waiting", queue, True)])

    result = scheduler.reassign_hexgent_waiting(SimpleNamespace(
        request_id="req", priority=9, schedule_revision=revision,
        expected_revision=expected, target_queue_name="prefill_waiting",
        planned_decode_instance="decode-b",
    ))
    assert result.reason_code == code
    assert result.current_revision == current
    assert result.target_queue_name == "prefill_waiting"
    assert queue == [req]
    if code != "success":
        assert result.success is False
        assert vars(req) == before
    else:
        assert result.success is True
        assert result.applied_revision == 6
        assert req.priority == 9


@pytest.mark.parametrize("class_name", [
    "HexgentUpdateWaitingPriorityReqOutput", "HexgentReassignWaitingReqOutput",
])
def test_mutation_response_schema_appends_optional_diagnostic_fields(class_name):
    path = _ROOT / "python" / "sglang" / "srt" / "managers" / "io_struct.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    schema = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == class_name)
    fields = [node for node in schema.body if isinstance(node, ast.AnnAssign)]
    assert [node.target.id for node in fields[-2:]] == ["reason_code", "current_revision"]
    assert [ast.unparse(node.annotation) for node in fields[-2:]] == ["Optional[str]", "Optional[int]"]
    assert all(isinstance(node.value, ast.Constant) and node.value.value is None
               for node in fields[-2:])


def test_hexgent_stage_timing_avoids_prefill_decode_pipeline_overlap():
    (build,) = _load_functions(_METRICS_PATH, ["build_hexgent_stage_timing_payload"])
    p = _make_req("p", priority=2, revision=4)
    p.time_stats = SimpleNamespace(
        forward_entry_time=10.0, prefill_finished_time=10.1,
        prefill_transfer_queue_entry_time=10.11, completion_time=10.14,
    )
    d = _make_req("d", priority=2, revision=4)
    d.time_stats = SimpleNamespace(
        decode_transfer_queue_entry_time=9.99, wait_queue_entry_time=10.14,
        forward_entry_time=10.145, completion_time=10.345,
    )
    prefill, decode = build(p, mode="prefill"), build(d, mode="decode")
    assert prefill["prefill_execution_ms"] == 100
    assert prefill["prefill_service_prefix_ms"] == pytest.approx(110)
    assert prefill["prefill_transfer_tail_ms"] == pytest.approx(30)
    assert decode["decode_pipeline_wait_ms"] == pytest.approx(150)
    assert decode["decode_execution_ms"] == pytest.approx(200)
    # Synthetic tooling example only: the historical sum double-counted P work.
    assert 100 + 150 + 200 == 450
    assert round(prefill["prefill_service_prefix_ms"]
                 + prefill["prefill_transfer_tail_ms"]
                 + decode["decode_execution_ms"]) == 340
    assert "transfer_ms" not in decode
    assert decode["decode_pipeline_wait_boundary"] == "decode_transfer_queue_entry_to_wait_queue_entry"


@pytest.mark.parametrize("missing", [
    "forward_entry_time", "prefill_finished_time",
    "prefill_transfer_queue_entry_time", "completion_time",
])
def test_hexgent_prefill_timing_requires_complete_new_boundaries(missing):
    (build,) = _load_functions(_METRICS_PATH, ["build_hexgent_stage_timing_payload"])
    values = dict(forward_entry_time=10.0, prefill_finished_time=10.1,
                  prefill_transfer_queue_entry_time=10.11, completion_time=10.14)
    del values[missing]
    req = _make_req("p", priority=2, revision=4)
    req.time_stats = SimpleNamespace(**values)
    assert build(req, mode="prefill") is None


def test_hexgent_stage_timing_preserves_zero_and_submillisecond_intervals():
    (build,) = _load_functions(_METRICS_PATH, ["build_hexgent_stage_timing_payload"])
    p = _make_req("p", priority=2, revision=4)
    p.time_stats = SimpleNamespace(
        forward_entry_time=10.0, prefill_finished_time=10.0,
        prefill_transfer_queue_entry_time=10.0, completion_time=10.0004,
    )
    d = _make_req("d", priority=2, revision=4)
    d.time_stats = SimpleNamespace(
        decode_transfer_queue_entry_time=9.0, wait_queue_entry_time=9.0,
        forward_entry_time=10.0, completion_time=10.0,
    )
    prefill, decode = build(p, mode="prefill"), build(d, mode="decode")
    assert prefill["prefill_execution_ms"] == 0
    assert prefill["prefill_service_prefix_ms"] == 0
    assert prefill["prefill_transfer_tail_ms"] == pytest.approx(0.4)
    assert decode["decode_pipeline_wait_ms"] == 0
    assert decode["decode_execution_ms"] == 0
