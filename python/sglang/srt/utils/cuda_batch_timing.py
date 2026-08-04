"""Asynchronous CUDA-event timing for scheduler model-forward batches."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict


def _as_int(value: Any) -> int:
    """Convert Python or scalar tensor values without importing torch."""
    if hasattr(value, "item"):
        value = value.item()
    return int(value or 0)


def batch_timing_metadata(batch: Any) -> Dict[str, int | str]:
    """Describe the actual model-forward work represented by a schedule batch."""
    forward_mode = getattr(getattr(batch, "forward_mode", None), "name", "UNKNOWN")
    batch_size = _as_int(batch.batch_size())
    sequence_tokens = _as_int(getattr(batch, "seq_lens_sum", 0))
    if forward_mode == "EXTEND":
        return {
            "phase": "prefill",
            "forward_mode": forward_mode,
            "batch_size": batch_size,
            "input_tokens": _as_int(getattr(batch, "extend_num_tokens", 0)),
            "sequence_tokens": sequence_tokens,
            "decode_tokens": 0,
        }
    if forward_mode == "DECODE":
        return {
            "phase": "decode",
            "forward_mode": forward_mode,
            "batch_size": batch_size,
            "input_tokens": 0,
            "sequence_tokens": sequence_tokens,
            # One generated token is advanced for each live sequence per iteration.
            "decode_tokens": batch_size,
        }
    return {
        "phase": "other",
        "forward_mode": forward_mode,
        "batch_size": batch_size,
        "input_tokens": 0,
        "sequence_tokens": sequence_tokens,
        "decode_tokens": 0,
    }


@dataclass
class _CudaTimingInterval:
    start_event: Any
    end_event: Any
    metadata: Dict[str, Any]


class CudaBatchTimer:
    """Collect forward GPU durations without timing CPU scheduling work.

    The caller records both events on the stream that executes the model forward.
    Normal serving only reports intervals that have already completed. Measurement
    mode can synchronize the end event to flush every sample deterministically.
    """

    def __init__(
        self,
        *,
        event_factory: Callable[..., Any],
        reporter: Callable[[Dict[str, Any]], None],
        force_synchronize: bool = False,
    ) -> None:
        self._event_factory = event_factory
        self._reporter = reporter
        self._force_synchronize = force_synchronize
        self._intervals: Deque[_CudaTimingInterval] = deque()

    @contextmanager
    def time(self, metadata: Dict[str, Any]):
        start_event = self._event_factory(enable_timing=True)
        start_event.record()
        try:
            yield
        finally:
            end_event = self._event_factory(enable_timing=True)
            end_event.record()
            self._intervals.append(
                _CudaTimingInterval(start_event, end_event, dict(metadata))
            )
            self.report_completed()

    def report_completed(self) -> None:
        while self._intervals:
            interval = self._intervals[0]
            if self._force_synchronize:
                interval.end_event.synchronize()
            elif not interval.end_event.query():
                return

            self._intervals.popleft()
            payload = dict(interval.metadata)
            payload["gpu_time_ms"] = round(
                float(interval.start_event.elapsed_time(interval.end_event)), 4
            )
            self._reporter(payload)


def ensure_cuda_batch_timer(
    timer: CudaBatchTimer | None,
    *,
    event_factory: Callable[..., Any],
    reporter: Callable[[Dict[str, Any]], None],
    force_synchronize: bool,
) -> CudaBatchTimer:
    """Create a timer only after the scheduler has initialized its device."""
    if timer is not None:
        return timer
    return CudaBatchTimer(
        event_factory=event_factory,
        reporter=reporter,
        force_synchronize=force_synchronize,
    )
