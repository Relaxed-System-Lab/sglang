from types import SimpleNamespace

from sglang.srt.utils.cuda_batch_timing import (
    CudaBatchTimer,
    batch_timing_metadata,
    ensure_cuda_batch_timer,
)


class FakeEvent:
    def __init__(self, elapsed_ms=0.0):
        self.elapsed_ms = elapsed_ms
        self.recorded = False
        self.synchronized = False

    def record(self):
        self.recorded = True

    def query(self):
        return self.recorded

    def synchronize(self):
        self.synchronized = True

    def elapsed_time(self, other):
        return other.elapsed_ms


def test_batch_timing_metadata_classifies_prefill_and_decode():
    prefill = SimpleNamespace(
        forward_mode=SimpleNamespace(name="EXTEND"),
        extend_num_tokens=96,
        seq_lens_sum=96,
        batch_size=lambda: 3,
    )
    decode = SimpleNamespace(
        forward_mode=SimpleNamespace(name="DECODE"),
        extend_num_tokens=0,
        seq_lens_sum=384,
        batch_size=lambda: 3,
    )

    assert batch_timing_metadata(prefill) == {
        "phase": "prefill",
        "forward_mode": "EXTEND",
        "batch_size": 3,
        "input_tokens": 96,
        "sequence_tokens": 96,
        "decode_tokens": 0,
    }
    assert batch_timing_metadata(decode) == {
        "phase": "decode",
        "forward_mode": "DECODE",
        "batch_size": 3,
        "input_tokens": 0,
        "sequence_tokens": 384,
        "decode_tokens": 3,
    }


def test_cuda_batch_timer_reports_completed_event_with_metadata():
    events = iter([FakeEvent(), FakeEvent(elapsed_ms=1.75)])
    reported = []
    timer = CudaBatchTimer(
        event_factory=lambda **_: next(events),
        reporter=reported.append,
        force_synchronize=True,
    )

    with timer.time({"phase": "decode", "batch_size": 2}):
        pass

    assert reported == [{"phase": "decode", "batch_size": 2, "gpu_time_ms": 1.75}]


def test_ensure_cuda_batch_timer_creates_timer_after_device_initialization():
    events = iter([FakeEvent(), FakeEvent(elapsed_ms=2.5)])
    reported = []
    timer = ensure_cuda_batch_timer(
        None,
        event_factory=lambda **_: next(events),
        reporter=reported.append,
        force_synchronize=True,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(name="DECODE"),
        extend_num_tokens=0,
        seq_lens_sum=64,
        batch_size=lambda: 1,
    )
    with timer.time(batch_timing_metadata(batch)):
        pass

    assert reported == [
        {
            "phase": "decode",
            "forward_mode": "DECODE",
            "batch_size": 1,
            "input_tokens": 0,
            "sequence_tokens": 64,
            "decode_tokens": 1,
            "gpu_time_ms": 2.5,
        }
    ]
