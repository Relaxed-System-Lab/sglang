import importlib.util
from pathlib import Path
from types import SimpleNamespace


_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "srt"
    / "disaggregation"
    / "priority_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("priority_utils", _MODULE_PATH)
priority_utils = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(priority_utils)


class _FakePolicy:
    def calc_priority(self, waiting_queue, running_batch):
        waiting_queue.sort(key=lambda req: (-req.priority, req.rid))


class _FakeScheduler(SimpleNamespace):
    pass


def test_sort_decode_waiting_queue_uses_scheduler_policy():
    scheduler = _FakeScheduler(
        waiting_queue=[
            SimpleNamespace(rid="req-low", priority=1),
            SimpleNamespace(rid="req-high", priority=5),
            SimpleNamespace(rid="req-mid", priority=3),
        ],
        running_batch=SimpleNamespace(reqs=[]),
        policy=_FakePolicy(),
    )

    priority_utils.sort_waiting_queue_by_policy(
        scheduler.waiting_queue, scheduler.policy, scheduler.running_batch
    )

    assert [req.rid for req in scheduler.waiting_queue] == [
        "req-high",
        "req-mid",
        "req-low",
    ]


def test_sort_decode_waiting_queue_is_noop_without_policy():
    scheduler = _FakeScheduler(
        waiting_queue=[
            SimpleNamespace(rid="req-a", priority=1),
            SimpleNamespace(rid="req-b", priority=5),
        ],
        running_batch=SimpleNamespace(reqs=[]),
    )

    priority_utils.sort_waiting_queue_by_policy(
        scheduler.waiting_queue, getattr(scheduler, "policy", None), scheduler.running_batch
    )

    assert [req.rid for req in scheduler.waiting_queue] == ["req-a", "req-b"]
