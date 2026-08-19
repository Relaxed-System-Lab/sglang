from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from .workflow_trace import WorkflowCall, load_workflow_traces_jsonl
except ImportError:  # Allows direct invocation by file path on a pod.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from workflow_trace import WorkflowCall, load_workflow_traces_jsonl  # type: ignore


@dataclass
class TraceCall:
    workflow_id: str
    call: WorkflowCall


@dataclass
class CallMeasurement:
    workflow_id: str
    call_id: str
    model: str
    worker_class: str
    input_len: int
    output_len: int
    trial: int
    ttft_ms: float
    total_latency_ms: float
    output_tokens: int
    inter_token_latencies_ms: List[float]
    error: Optional[str] = None

    @property
    def coarse_decode_time_ms(self) -> float:
        return float(max(0.0, self.total_latency_ms - self.ttft_ms))

    @property
    def decode_time_ms(self) -> float:
        if self.decode_token_count <= 0:
            return 0.0
        return self.coarse_decode_time_ms

    @property
    def decode_token_count(self) -> int:
        return max(0, int(self.output_tokens) - 1)

    @property
    def has_token_level_latencies(self) -> bool:
        return bool(self.inter_token_latencies_ms) and len(self.inter_token_latencies_ms) == self.decode_token_count

    @property
    def decode_time_per_token_ms(self) -> float:
        if self.decode_token_count <= 0:
            return 0.0
        return float(self.decode_time_ms / self.decode_token_count)

    @property
    def decode_p50_ms(self) -> float:
        if not self.has_token_level_latencies:
            return self.decode_time_per_token_ms
        return _percentile(self.inter_token_latencies_ms, 50.0)

    @property
    def decode_p95_ms(self) -> float:
        if not self.has_token_level_latencies:
            return self.decode_time_per_token_ms
        return _percentile(self.inter_token_latencies_ms, 95.0)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    lower_weight = upper - position
    upper_weight = position - lower
    return ordered[lower] * lower_weight + ordered[upper] * upper_weight


class SSEJsonEventParser:
    """Incrementally parse SSE JSON while preserving partial network chunks."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: bytes | str) -> Iterator[Dict[str, Any]]:
        if isinstance(chunk, bytes):
            text = chunk.decode("utf-8", errors="replace")
        else:
            text = chunk
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                continue
            yield json.loads(payload)

    def finish(self) -> Iterator[Dict[str, Any]]:
        tail = self._buffer.strip()
        self._buffer = ""
        if not tail.startswith("data:"):
            return
        payload = tail[len("data:") :].strip()
        if payload and payload != "[DONE]":
            yield json.loads(payload)


def iter_sse_json_events(chunks: Iterable[bytes | str]) -> Iterator[Dict[str, Any]]:
    parser = SSEJsonEventParser()
    for chunk in chunks:
        yield from parser.feed(chunk)
    yield from parser.finish()


def load_trace_calls(trace_path: str | Path, *, max_calls: Optional[int] = None) -> List[TraceCall]:
    calls: List[TraceCall] = []
    for workflow in load_workflow_traces_jsonl(trace_path):
        for call in workflow.calls:
            calls.append(TraceCall(workflow_id=workflow.workflow_id, call=call))
            if max_calls is not None and len(calls) >= max_calls:
                return calls
    return calls


def build_request_payload(
    trace_call: TraceCall,
    *,
    model: Optional[str],
    exact_output_length: bool,
) -> Dict[str, Any]:
    call = trace_call.call
    payload = dict(call.request)
    if model:
        payload["model"] = model
    payload["conversation_id"] = trace_call.workflow_id
    payload["hexgent_call_id"] = call.call_id
    payload["max_tokens"] = int(call.output_len)
    payload["stream"] = True
    payload.setdefault("temperature", 0.0)
    payload.setdefault("stream_options", {"include_usage": True})
    if exact_output_length:
        payload["ignore_eos"] = True
    return payload


def _event_content(event: Dict[str, Any]) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    choice = choices[0] or {}
    delta = choice.get("delta") or {}
    message = choice.get("message") or {}
    for container in (delta, message, choice):
        for key in ("content", "reasoning_content", "text"):
            value = container.get(key)
            if value:
                return str(value)
        tool_calls = container.get("tool_calls")
        if tool_calls:
            return json.dumps(tool_calls, ensure_ascii=False)
    return ""


def _event_output_tokens(event: Dict[str, Any]) -> Optional[int]:
    usage = event.get("usage") or {}
    for key in ("completion_tokens", "output_tokens"):
        if usage.get(key) is not None:
            return int(usage[key])
    choices = event.get("choices") or []
    if choices:
        meta = choices[0].get("meta_info") or {}
        if meta.get("completion_tokens") is not None:
            return int(meta["completion_tokens"])
    meta = event.get("meta_info") or {}
    if meta.get("completion_tokens") is not None:
        return int(meta["completion_tokens"])
    return None


async def measure_call(
    session,
    endpoint_url: str,
    trace_call: TraceCall,
    *,
    model: Optional[str],
    worker_class: str,
    trial: int,
    exact_output_length: bool,
    timeout_s: float,
) -> CallMeasurement:
    payload = build_request_payload(
        trace_call,
        model=model,
        exact_output_length=exact_output_length,
    )
    call = trace_call.call
    start = time.perf_counter()
    first_token_at: Optional[float] = None
    last_token_at: Optional[float] = None
    itls_ms: List[float] = []
    output_tokens = 0
    parser = SSEJsonEventParser()

    async with session.post(endpoint_url, json=payload, timeout=timeout_s) as response:
        if response.status >= 400:
            text = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {text}")
        async for chunk in response.content:
            for event in parser.feed(chunk):
                content = _event_content(event)
                tokens = _event_output_tokens(event)
                if tokens is not None:
                    output_tokens = tokens
                if not content:
                    continue
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                elif last_token_at is not None:
                    itls_ms.append((now - last_token_at) * 1000.0)
                last_token_at = now
                if output_tokens == 0:
                    output_tokens += 1
        for event in parser.finish():
            tokens = _event_output_tokens(event)
            if tokens is not None:
                output_tokens = tokens

    end = time.perf_counter()
    ttft_ms = ((first_token_at or end) - start) * 1000.0
    return CallMeasurement(
        workflow_id=trace_call.workflow_id,
        call_id=call.call_id,
        model=str(model or payload.get("model") or ""),
        worker_class=worker_class,
        input_len=int(call.input_len),
        output_len=int(call.output_len),
        trial=int(trial),
        ttft_ms=ttft_ms,
        total_latency_ms=(end - start) * 1000.0,
        output_tokens=int(output_tokens or call.output_len),
        inter_token_latencies_ms=itls_ms,
    )


def build_prefill_row(measurement: CallMeasurement) -> Dict[str, Any]:
    row = asdict(measurement)
    row.pop("inter_token_latencies_ms", None)
    row["phase"] = "prefill"
    row["prefill_time_ms"] = float(measurement.ttft_ms)
    row["timing_source"] = "stream_ttft_ms"
    return row


def build_decode_row(measurement: CallMeasurement) -> Dict[str, Any]:
    row = asdict(measurement)
    row["phase"] = "decode"
    row["decode_time_ms"] = float(measurement.decode_time_ms)
    row["decode_time_per_token_ms"] = float(measurement.decode_time_per_token_ms)
    row["decode_p50_ms"] = float(measurement.decode_p50_ms)
    row["decode_p95_ms"] = float(measurement.decode_p95_ms)
    row["timing_source"] = "wall_clock_total_latency_minus_ttft_ms"
    row["stream_chunk_count"] = len(measurement.inter_token_latencies_ms)
    row["stream_chunk_latency_sum_ms"] = float(sum(measurement.inter_token_latencies_ms))
    row["stream_timing_granularity"] = (
        "token" if measurement.has_token_level_latencies else "chunk_or_missing"
    )
    return row


def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fit_linear(rows: Sequence[Dict[str, Any]], target_key: str) -> Dict[str, float]:
    points = [
        (float(row.get("input_len", 0.0)), float(row.get("output_len", 0.0)), float(row[target_key]))
        for row in rows
        if row.get(target_key) is not None
    ]
    if not points:
        return {"base_ms": 0.0, "input_coeff_ms": 0.0, "output_coeff_ms": 0.0}
    if len(points) < 3:
        y_mean = statistics.mean(point[2] for point in points)
        input_values = [point[0] for point in points]
        output_values = [point[1] for point in points]
        input_coeff = _simple_slope(input_values, [point[2] for point in points])
        output_coeff = _simple_slope(output_values, [point[2] for point in points])
        base = max(0.0, y_mean - input_coeff * statistics.mean(input_values) - output_coeff * statistics.mean(output_values))
        return {
            "base_ms": float(base),
            "input_coeff_ms": float(max(0.0, input_coeff)),
            "output_coeff_ms": float(max(0.0, output_coeff)),
        }

    # Solve normal equations for y = b + i*x1 + o*x2 using Gaussian elimination.
    sx1 = sum(p[0] for p in points)
    sx2 = sum(p[1] for p in points)
    sy = sum(p[2] for p in points)
    sx1x1 = sum(p[0] * p[0] for p in points)
    sx2x2 = sum(p[1] * p[1] for p in points)
    sx1x2 = sum(p[0] * p[1] for p in points)
    sx1y = sum(p[0] * p[2] for p in points)
    sx2y = sum(p[1] * p[2] for p in points)
    matrix = [
        [float(len(points)), sx1, sx2, sy],
        [sx1, sx1x1, sx1x2, sx1y],
        [sx2, sx1x2, sx2x2, sx2y],
    ]
    base, input_coeff, output_coeff = _solve_3x3(matrix)
    return {
        "base_ms": float(max(0.0, base)),
        "input_coeff_ms": float(max(0.0, input_coeff)),
        "output_coeff_ms": float(max(0.0, output_coeff)),
    }


def _simple_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(set(xs)) < 2:
        return 0.0
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def _solve_3x3(matrix: List[List[float]]) -> Tuple[float, float, float]:
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot][col]) < 1e-9:
            return (0.0, 0.0, 0.0)
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        divisor = matrix[col][col]
        matrix[col] = [value / divisor for value in matrix[col]]
        for row in range(3):
            if row == col:
                continue
            factor = matrix[row][col]
            matrix[row] = [value - factor * matrix[col][idx] for idx, value in enumerate(matrix[row])]
    return (matrix[0][3], matrix[1][3], matrix[2][3])


def build_latency_db(
    prefill_rows: Sequence[Dict[str, Any]],
    decode_rows: Sequence[Dict[str, Any]],
    *,
    transfer_same_ms: float,
    transfer_cross_ms: float,
    hot_prefill_scale: float = 1.0,
) -> Dict[str, Any]:
    hardware_names = sorted(
        {str(row["worker_class"]) for row in prefill_rows}
        | {str(row["worker_class"]) for row in decode_rows}
    )
    payload: Dict[str, Any] = {"hardware": {}, "transfer": {}}
    for hardware in hardware_names:
        hw_prefill = [row for row in prefill_rows if str(row["worker_class"]) == hardware]
        hw_decode = [row for row in decode_rows if str(row["worker_class"]) == hardware]
        prefill_cold = _fit_linear(hw_prefill, "prefill_time_ms")
        prefill_hot = {
            key: (value * float(hot_prefill_scale) if key.endswith("_ms") or key.endswith("coeff_ms") else value)
            for key, value in prefill_cold.items()
        }
        payload["hardware"][hardware] = {
            "prefill_cold": prefill_cold,
            "prefill_hot": prefill_hot,
            "decode": _fit_linear(hw_decode, "decode_time_ms"),
        }
    payload["transfer"] = {
        "same_kind": {"base_ms": float(transfer_same_ms), "input_coeff_ms": 0.0},
        "cross_kind": {"base_ms": float(transfer_cross_ms), "input_coeff_ms": 0.0},
    }
    return payload


async def run_profile(args: argparse.Namespace) -> None:
    import aiohttp

    trace_calls = load_trace_calls(args.trace_path, max_calls=args.max_calls)
    if not trace_calls:
        raise RuntimeError(f"No calls found in {args.trace_path}")

    phase = args.phase
    endpoint_url = args.endpoint_url.rstrip("/")
    raw_rows: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession() as session:
        for warmup_idx in range(args.warmup_calls):
            trace_call = trace_calls[warmup_idx % len(trace_calls)]
            await measure_call(
                session,
                endpoint_url,
                trace_call,
                model=args.model,
                worker_class=args.worker_class,
                trial=-1,
                exact_output_length=not args.disable_exact_output_length,
                timeout_s=args.timeout_s,
            )
        for trial in range(args.trials):
            for index, trace_call in enumerate(trace_calls):
                measurement = await measure_call(
                    session,
                    endpoint_url,
                    trace_call,
                    model=args.model,
                    worker_class=args.worker_class,
                    trial=trial,
                    exact_output_length=not args.disable_exact_output_length,
                    timeout_s=args.timeout_s,
                )
                raw_rows.append(asdict(measurement))
                if args.raw_output:
                    append_jsonl(args.raw_output, [asdict(measurement)])
                if phase in {"prefill", "both"} and args.prefill_output:
                    append_jsonl(args.prefill_output, [build_prefill_row(measurement)])
                if phase in {"decode", "both"} and args.decode_output:
                    append_jsonl(args.decode_output, [build_decode_row(measurement)])
                print(
                    f"{index + 1}/{len(trace_calls)} trial={trial} "
                    f"call={measurement.call_id} ttft_ms={measurement.ttft_ms:.3f} "
                    f"decode_ms={measurement.decode_time_ms:.3f}",
                    flush=True,
                )


def run_fit_latency_db(args: argparse.Namespace) -> None:
    prefill_rows = read_jsonl(args.prefill_path)
    decode_rows = read_jsonl(args.decode_path)
    payload = build_latency_db(
        prefill_rows,
        decode_rows,
        transfer_same_ms=args.transfer_same_ms,
        transfer_cross_ms=args.transfer_cross_ms,
        hot_prefill_scale=args.hot_prefill_scale,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Hexgent workflow calls against a live SGLang endpoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Run trace calls one by one and write phase profile JSONL.")
    profile.add_argument("--trace-path", required=True)
    profile.add_argument("--endpoint-url", required=True, help="Full OpenAI-compatible chat completions URL.")
    profile.add_argument("--model", required=True)
    profile.add_argument("--worker-class", required=True, help="Hardware label, e.g. A100_80G or H20.")
    profile.add_argument("--phase", choices=["prefill", "decode", "both"], default="both")
    profile.add_argument("--prefill-output")
    profile.add_argument("--decode-output")
    profile.add_argument("--raw-output")
    profile.add_argument("--max-calls", type=int)
    profile.add_argument("--trials", type=int, default=1)
    profile.add_argument("--warmup-calls", type=int, default=1)
    profile.add_argument("--timeout-s", type=float, default=3600.0)
    profile.add_argument("--disable-exact-output-length", action="store_true")

    fit = subparsers.add_parser("fit-latency-db", help="Fit optimizer-compatible latency_db.json from phase JSONL.")
    fit.add_argument("--prefill-path", required=True)
    fit.add_argument("--decode-path", required=True)
    fit.add_argument("--output-path", required=True)
    fit.add_argument("--transfer-same-ms", type=float, default=12.0)
    fit.add_argument("--transfer-cross-ms", type=float, default=26.0)
    fit.add_argument("--hot-prefill-scale", type=float, default=1.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "profile":
        asyncio.run(run_profile(args))
        return 0
    if args.command == "fit-latency-db":
        run_fit_latency_db(args)
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
