#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


SHAREGPT_HF_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_HF_FILE = "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
BFCL_HF_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_V3_FILES = (
    "BFCL_v3_exec_simple.json",
    "BFCL_v3_exec_multiple.json",
    "BFCL_v3_exec_parallel.json",
    "BFCL_v3_exec_parallel_multiple.json",
    "BFCL_v3_multi_turn_base.json",
    "BFCL_v3_multi_turn_composite.json",
    "BFCL_v3_multi_turn_long_context.json",
)
HOTPOTQA_HF_DATASET = "hotpotqa/hotpot_qa"
HOTPOTQA_HF_CONFIG = "distractor"
HOTPOTQA_HF_SPLIT = "validation"
HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"


@dataclass
class WorkflowCall:
    call_id: str
    request: Dict[str, Any]
    input_len: int
    output_len: int
    parents: List[str]


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


def _iter_json_lines(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _iter_json_array(path: Path) -> Iterator[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    for value in data:
        if isinstance(value, dict):
            yield value


def _hf_dataset_resolve_url(repo_id: str, filename: str) -> str:
    repo = quote(repo_id, safe="/")
    file_part = "/".join(quote(part, safe="") for part in filename.split("/"))
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{file_part}"


def _download_url_to_cache(url: str, cache_path: Path, timeout_s: float = 120.0) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path
    request = Request(url, headers={"User-Agent": "hexgent-pod-profile"})
    with urlopen(request, timeout=timeout_s) as response, cache_path.open("wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    return cache_path


def _fetch_json(url: str, timeout_s: float = 60.0) -> Dict[str, Any]:
    request = Request(url, headers={"User-Agent": "hexgent-pod-profile"})
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def load_sharegpt_hf_records(cache_dir: Path, max_records: Optional[int]) -> List[Dict[str, Any]]:
    cache_path = cache_dir / "sharegpt" / SHAREGPT_HF_FILE
    downloaded = _download_url_to_cache(
        _hf_dataset_resolve_url(SHAREGPT_HF_REPO, SHAREGPT_HF_FILE),
        cache_path,
    )
    rows: List[Dict[str, Any]] = []
    for record in _iter_json_array(downloaded):
        if _count_sharegpt_user_turns(record) <= 1:
            continue
        rows.append(record)
        if max_records is not None and len(rows) >= max_records:
            break
    return rows


def load_bfcl_hf_records(
    cache_dir: Path,
    max_records: Optional[int],
    bfcl_files: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for filename in bfcl_files or BFCL_V3_FILES:
        cache_path = cache_dir / "bfcl" / filename
        downloaded = _download_url_to_cache(
            _hf_dataset_resolve_url(BFCL_HF_REPO, filename),
            cache_path,
        )
        for record in _iter_json_lines(downloaded):
            if len(_bfcl_turns(record)) <= 1:
                continue
            rows.append(record)
            if max_records is not None and len(rows) >= max_records:
                return rows
    return rows


def load_hotpotqa_hf_rows(
    max_rows: Optional[int],
    *,
    config: str = HOTPOTQA_HF_CONFIG,
    split: str = HOTPOTQA_HF_SPLIT,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    wanted = max_rows if max_rows is not None else 100
    offset = 0
    while len(rows) < wanted:
        params = urlencode(
            {
                "dataset": HOTPOTQA_HF_DATASET,
                "config": config,
                "split": split,
                "offset": offset,
                "length": min(page_size, wanted - len(rows)),
            }
        )
        payload = _fetch_json(f"{HF_DATASETS_SERVER}/rows?{params}")
        page_rows = payload.get("rows") or []
        if not page_rows:
            break
        for row in page_rows:
            if isinstance(row, dict) and isinstance(row.get("row"), dict):
                rows.append(row["row"])
                if len(rows) >= wanted:
                    return rows
        offset += len(page_rows)
    return rows


def _count_text_tokens(text: Any) -> int:
    return max(1, len(str(text or "").split()))


def _count_messages_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    return sum(_count_text_tokens(message.get("content", "")) for message in messages) + 4


def _normalized_sharegpt_turns(record: Dict[str, Any]) -> List[Dict[str, str]]:
    role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}
    turns: List[Dict[str, str]] = []
    for turn in record.get("conversations") or []:
        role = role_map.get(str(turn.get("from", "")).lower())
        if role is None:
            continue
        turns.append({"role": role, "content": str(turn.get("value", ""))})
    return turns


def _count_sharegpt_user_turns(record: Dict[str, Any]) -> int:
    return sum(1 for turn in _normalized_sharegpt_turns(record) if turn["role"] == "user")


def _bfcl_turns(record: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    questions = record.get("question") or []
    if not isinstance(questions, list):
        return []
    turns: List[List[Dict[str, Any]]] = []
    for turn in questions:
        if isinstance(turn, list):
            turns.append([message for message in turn if isinstance(message, dict)])
        elif isinstance(turn, dict):
            turns.append([turn])
    return [turn for turn in turns if turn]


def _normalize_request(record: Dict[str, Any]) -> Dict[str, Any]:
    request = copy.deepcopy(record.get("request") or {})
    if "messages" not in request:
        messages = record.get("messages")
        if messages is None:
            prompt = record.get("prompt") or record.get("input") or record.get("question") or ""
            messages = [{"role": "user", "content": str(prompt)}]
        request["messages"] = messages
    return request


def _workflow_to_dict(workflow_id: str, workflow_type: str, calls: Sequence[WorkflowCall]) -> Dict[str, Any]:
    return {"workflow_id": workflow_id, "workflow_type": workflow_type, "calls": [asdict(call) for call in calls]}


def _workflow_call_from_record(record: Dict[str, Any], index: int) -> Tuple[str, WorkflowCall]:
    workflow_id = str(record.get("workflow_id") or record.get("id") or f"workflow-{index}")
    call_id = str(record.get("call_id") or f"{workflow_id}-call-0")
    request = _normalize_request(record)
    input_len = int(record.get("input_len") or record.get("prompt_tokens") or _count_messages_tokens(request["messages"]))
    output_len = int(record.get("output_len") or record.get("max_tokens") or request.get("max_tokens") or 128)
    parents = [str(parent) for parent in record.get("parents", [])]
    return workflow_id, WorkflowCall(call_id, request, input_len, output_len, parents)


def _sharegpt_messages(record: Dict[str, Any]) -> Tuple[List[Dict[str, str]], int]:
    messages: List[Dict[str, str]] = []
    output_len = 128
    for turn in _normalized_sharegpt_turns(record):
        if turn["role"] == "assistant" and messages:
            output_len = _count_text_tokens(turn["content"])
            break
        messages.append(turn)
    if not messages:
        messages = [{"role": "user", "content": ""}]
    return messages, output_len


def convert_sharegpt_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    workflows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        workflow_id = str(record.get("id") or f"sharegpt-{index}")
        turns = _normalized_sharegpt_turns(record)
        calls: List[WorkflowCall] = []
        previous_call_id: Optional[str] = None
        for turn_index, turn in enumerate(turns):
            if turn["role"] != "user":
                continue
            messages = copy.deepcopy(turns[: turn_index + 1])
            output_len = 128
            for response_turn in turns[turn_index + 1 :]:
                if response_turn["role"] == "assistant":
                    output_len = _count_text_tokens(response_turn["content"])
                    break
                if response_turn["role"] == "user":
                    break
            call_id = f"{workflow_id}-call-{len(calls)}"
            calls.append(
                WorkflowCall(
                    call_id=call_id,
                    request={"messages": messages},
                    input_len=_count_messages_tokens(messages),
                    output_len=output_len,
                    parents=[previous_call_id] if previous_call_id else [],
                )
            )
            previous_call_id = call_id
        if len(calls) > 1:
            workflows.append(_workflow_to_dict(workflow_id, "sharegpt", calls))
    return workflows


def convert_bfcl_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    workflows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        turns = _bfcl_turns(record)
        if len(turns) <= 1:
            continue

        workflow_id = str(record.get("id") or f"bfcl-{index}")
        tools = record.get("function") or []
        calls: List[WorkflowCall] = []
        previous_call_id: Optional[str] = None
        tool_tokens = _count_text_tokens(json.dumps(tools, ensure_ascii=False))
        for turn_index, messages in enumerate(turns):
            call_id = f"{workflow_id}-call-{turn_index}"
            plain_messages = [dict(message) for message in messages]
            if tools and plain_messages:
                tool_text = json.dumps(tools, ensure_ascii=False)
                plain_messages[0]["content"] = (
                    f"{plain_messages[0].get('content', '')}\n\n"
                    f"Available tools:\n{tool_text}\n\n"
                    "Respond with the tool call you would make."
                )
            request = {"messages": plain_messages}
            calls.append(
                WorkflowCall(
                    call_id=call_id,
                    parents=[previous_call_id] if previous_call_id else [],
                    request=request,
                    input_len=_count_messages_tokens(plain_messages) + tool_tokens,
                    output_len=64,
                )
            )
            previous_call_id = call_id
        workflows.append(_workflow_to_dict(workflow_id, "bfcl", calls))
    return workflows


def convert_hotpotqa_rows_to_lats_style(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    workflows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = str(row.get("question") or "")
        if not question:
            continue
        answer = str(row.get("answer") or "")
        workflow_id = str(row.get("id") or f"hotpotqa-{index}")
        context = row.get("context") or {}
        snippets: List[str] = []
        if isinstance(context, dict):
            titles = context.get("title") or []
            sentences = context.get("sentences") or []
            for title_index, title in enumerate(titles[:4] if isinstance(titles, list) else []):
                sentence_group = sentences[title_index] if isinstance(sentences, list) and title_index < len(sentences) else []
                if isinstance(sentence_group, list):
                    snippet = " ".join(str(sentence) for sentence in sentence_group[:2])
                else:
                    snippet = str(sentence_group)
                snippets.append(f"{title}: {snippet}".strip())
        if not snippets:
            snippets = [question]

        calls: List[WorkflowCall] = []
        root_id = f"{workflow_id}-root"
        root_messages = [{"role": "user", "content": question}]
        calls.append(
            WorkflowCall(
                call_id=root_id,
                parents=[],
                request={"messages": root_messages},
                input_len=_count_messages_tokens(root_messages),
                output_len=max(128, min(512, _count_text_tokens(answer) * 6 or 128)),
            )
        )
        expand_ids: List[str] = []
        for snippet_index, snippet in enumerate(snippets[:4]):
            call_id = f"{workflow_id}-expand-{snippet_index}"
            messages = [
                {
                    "role": "user",
                    "content": f"Question: {question}\nEvidence: {snippet}\nExpand the reasoning tree.",
                }
            ]
            calls.append(
                WorkflowCall(
                    call_id=call_id,
                    parents=[root_id],
                    request={"messages": messages},
                    input_len=_count_messages_tokens(messages),
                    output_len=max(128, min(512, _count_text_tokens(snippet) + 120)),
                )
            )
            expand_ids.append(call_id)
        aggregate_messages = [{"role": "user", "content": f"Answer the question using the explored evidence: {question}"}]
        calls.append(
            WorkflowCall(
                call_id=f"{workflow_id}-aggregate",
                parents=expand_ids or [root_id],
                request={"messages": aggregate_messages},
                input_len=_count_messages_tokens(aggregate_messages),
                output_len=max(128, min(512, _count_text_tokens(answer) * 4 or 128)),
            )
        )
        workflows.append(_workflow_to_dict(workflow_id, "lats", calls))
    return workflows


def convert_trace(
    input_path: Optional[Path],
    input_format: str,
    output_path: Path,
    max_calls: Optional[int],
    *,
    cache_dir: Optional[Path] = None,
    bfcl_files: Optional[Sequence[str]] = None,
    hotpot_config: str = HOTPOTQA_HF_CONFIG,
    hotpot_split: str = HOTPOTQA_HF_SPLIT,
    hotpot_page_size: int = 100,
) -> int:
    grouped: Dict[str, List[WorkflowCall]] = {}
    count = 0

    cache_root = cache_dir or Path.home() / ".cache" / "hexgent_pod_profile"
    if input_format == "sharegpt_hf":
        workflows = convert_sharegpt_records(load_sharegpt_hf_records(cache_root, max_calls))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in workflows) + "\n", encoding="utf-8")
        return sum(len(row["calls"]) for row in workflows)
    if input_format == "bfcl_hf":
        workflows = convert_bfcl_records(load_bfcl_hf_records(cache_root, max_calls, bfcl_files))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in workflows) + "\n", encoding="utf-8")
        return sum(len(row["calls"]) for row in workflows)
    if input_format == "hotpotqa_hf":
        workflows = convert_hotpotqa_rows_to_lats_style(
            load_hotpotqa_hf_rows(
                max_calls,
                config=hotpot_config,
                split=hotpot_split,
                page_size=hotpot_page_size,
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in workflows) + "\n", encoding="utf-8")
        return sum(len(row["calls"]) for row in workflows)

    if input_path is None:
        raise ValueError("--input-path is required for this input format")
    if input_format == "sharegpt_json":
        records = _iter_json_array(input_path)
    else:
        records = _iter_json_lines(input_path)

    for index, record in enumerate(records):
        if max_calls is not None and count >= max_calls:
            break
        if input_format == "workflow_jsonl":
            calls = record.get("calls") or []
            workflow_id = str(record.get("workflow_id") or record.get("id") or f"workflow-{index}")
            for call_index, call_record in enumerate(calls):
                _, call = _workflow_call_from_record(
                    {"workflow_id": workflow_id, **dict(call_record)},
                    call_index,
                )
                grouped.setdefault(workflow_id, []).append(call)
                count += 1
                if max_calls is not None and count >= max_calls:
                    break
        elif input_format == "sharegpt_json":
            for workflow in convert_sharegpt_records([record]):
                grouped.setdefault(workflow["workflow_id"], []).extend(
                    WorkflowCall(**call) for call in workflow["calls"]
                )
                count += len(workflow["calls"])
        else:
            workflow_id, call = _workflow_call_from_record(record, index)
            grouped.setdefault(workflow_id, []).append(call)
            count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for workflow_id, calls in grouped.items():
            fh.write(
                json.dumps(
                    {"workflow_id": workflow_id, "calls": [asdict(call) for call in calls]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return count


def load_trace_calls(trace_path: Path, max_calls: Optional[int]) -> List[TraceCall]:
    calls: List[TraceCall] = []
    for workflow in _iter_json_lines(trace_path):
        workflow_id = str(workflow.get("workflow_id") or workflow.get("id") or len(calls))
        for raw_call in workflow.get("calls") or []:
            _, call = _workflow_call_from_record({"workflow_id": workflow_id, **dict(raw_call)}, len(calls))
            calls.append(TraceCall(workflow_id=workflow_id, call=call))
            if max_calls is not None and len(calls) >= max_calls:
                return calls
    return calls


class SSEJsonEventParser:
    """Incrementally parse SSE JSON while retaining incomplete chunks."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: bytes) -> Iterator[Dict[str, Any]]:
        self._buffer += chunk.decode("utf-8", errors="ignore")
        while "\n\n" in self._buffer:
            event, self._buffer = self._buffer.split("\n\n", 1)
            for line in event.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data or data == "[DONE]":
                    continue
                yield json.loads(data)


def iter_sse_json_events(chunks: Iterable[bytes]) -> Iterator[Dict[str, Any]]:
    parser = SSEJsonEventParser()
    for chunk in chunks:
        yield from parser.feed(chunk)


def _event_content(event: Dict[str, Any]) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
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
    value = usage.get("completion_tokens") or usage.get("output_tokens")
    return int(value) if value is not None else None


def format_http_error(status: int, body: str) -> str:
    body = body.strip()
    return f"HTTP {status}: {body}" if body else f"HTTP {status}"


def build_request_payload(trace_call: TraceCall, model: str, exact_output_length: bool) -> Dict[str, Any]:
    payload = copy.deepcopy(trace_call.call.request)
    payload["model"] = model
    payload["stream"] = True
    payload.setdefault("temperature", 0.0)
    payload.setdefault("stream_options", {"include_usage": True})
    payload["max_tokens"] = max(1, int(trace_call.call.output_len if exact_output_length else payload.get("max_tokens", 128)))
    payload.setdefault("ignore_eos", True)
    return payload


async def measure_call(
    trace_call: TraceCall,
    *,
    endpoint_url: str,
    model: str,
    worker_class: str,
    trial: int,
    timeout_s: float,
    exact_output_length: bool,
) -> CallMeasurement:
    import aiohttp

    payload = build_request_payload(trace_call, model, exact_output_length)
    token_times: List[float] = []
    output_tokens: Optional[int] = None
    parser = SSEJsonEventParser()
    start = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint_url, json=payload) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(format_http_error(response.status, body))
                async for chunk in response.content.iter_any():
                    now = time.perf_counter()
                    for event in parser.feed(chunk):
                        token_count = _event_output_tokens(event)
                        if token_count is not None:
                            output_tokens = token_count
                        if _event_content(event):
                            token_times.append(now)
        end = time.perf_counter()
        ttft_ms = (token_times[0] - start) * 1000.0 if token_times else (end - start) * 1000.0
        inter_token_ms = [
            (current - previous) * 1000.0 for previous, current in zip(token_times, token_times[1:])
        ]
        return CallMeasurement(
            workflow_id=trace_call.workflow_id,
            call_id=trace_call.call.call_id,
            model=model,
            worker_class=worker_class,
            input_len=trace_call.call.input_len,
            output_len=trace_call.call.output_len,
            trial=trial,
            ttft_ms=ttft_ms,
            total_latency_ms=(end - start) * 1000.0,
            output_tokens=output_tokens or len(token_times),
            inter_token_latencies_ms=inter_token_ms,
        )
    except Exception as exc:
        end = time.perf_counter()
        return CallMeasurement(
            workflow_id=trace_call.workflow_id,
            call_id=trace_call.call.call_id,
            model=model,
            worker_class=worker_class,
            input_len=trace_call.call.input_len,
            output_len=trace_call.call.output_len,
            trial=trial,
            ttft_ms=0.0,
            total_latency_ms=(end - start) * 1000.0,
            output_tokens=0,
            inter_token_latencies_ms=[],
            error=str(exc),
        )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    return float(sorted_values[low] * (high - rank) + sorted_values[high] * (rank - low))


def build_prefill_row(measurement: CallMeasurement) -> Dict[str, Any]:
    row = asdict(measurement)
    row.pop("inter_token_latencies_ms", None)
    row["phase"] = "prefill"
    row["prefill_time_ms"] = measurement.ttft_ms
    row["timing_source"] = "stream_ttft_ms"
    return row


def build_decode_row(measurement: CallMeasurement) -> Dict[str, Any]:
    row = asdict(measurement)
    row["phase"] = "decode"
    row["decode_time_ms"] = measurement.decode_time_ms
    row["decode_time_per_token_ms"] = measurement.decode_time_per_token_ms
    row["decode_p50_ms"] = measurement.decode_p50_ms
    row["decode_p95_ms"] = measurement.decode_p95_ms
    row["timing_source"] = "wall_clock_total_latency_minus_ttft_ms"
    row["stream_chunk_count"] = len(measurement.inter_token_latencies_ms)
    row["stream_chunk_latency_sum_ms"] = float(sum(measurement.inter_token_latencies_ms))
    row["stream_timing_granularity"] = (
        "token" if measurement.has_token_level_latencies else "chunk_or_missing"
    )
    return row


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(_iter_json_lines(path))


def _solve_3x3(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    matrix = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot][col]) < 1e-9:
            return None
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        scale = matrix[col][col]
        matrix[col] = [value / scale for value in matrix[col]]
        for row in range(3):
            if row == col:
                continue
            factor = matrix[row][col]
            matrix[row] = [value - factor * base for value, base in zip(matrix[row], matrix[col])]
    return [matrix[row][3] for row in range(3)]


def _simple_slope(rows: Sequence[Dict[str, Any]], y_key: str, x_key: str) -> Dict[str, float]:
    if not rows:
        return {"base_ms": 0.0, "input_coeff_ms": 0.0, "output_coeff_ms": 0.0}
    xs = [float(row.get(x_key, 0.0)) for row in rows]
    ys = [float(row.get(y_key, 0.0)) for row in rows]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0 if var_x < 1e-9 else sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
    base = max(0.0, mean_y - slope * mean_x)
    return {
        "base_ms": float(base),
        "input_coeff_ms": float(max(0.0, slope if x_key == "input_len" else 0.0)),
        "output_coeff_ms": float(max(0.0, slope if x_key == "output_len" else 0.0)),
    }


def _fit_linear(rows: Sequence[Dict[str, Any]], y_key: str) -> Dict[str, float]:
    usable = [row for row in rows if row.get(y_key) is not None]
    if len(usable) < 3:
        x_key = "output_len" if y_key.startswith("decode") else "input_len"
        return _simple_slope(usable, y_key, x_key)
    n = float(len(usable))
    sx = sum(float(row.get("input_len", 0.0)) for row in usable)
    sz = sum(float(row.get("output_len", 0.0)) for row in usable)
    sy = sum(float(row.get(y_key, 0.0)) for row in usable)
    sxx = sum(float(row.get("input_len", 0.0)) ** 2 for row in usable)
    szz = sum(float(row.get("output_len", 0.0)) ** 2 for row in usable)
    sxz = sum(float(row.get("input_len", 0.0)) * float(row.get("output_len", 0.0)) for row in usable)
    sxy = sum(float(row.get("input_len", 0.0)) * float(row.get(y_key, 0.0)) for row in usable)
    szy = sum(float(row.get("output_len", 0.0)) * float(row.get(y_key, 0.0)) for row in usable)
    solution = _solve_3x3([[n, sx, sz], [sx, sxx, sxz], [sz, sxz, szz]], [sy, sxy, szy])
    if solution is None:
        x_key = "output_len" if y_key.startswith("decode") else "input_len"
        return _simple_slope(usable, y_key, x_key)
    base, input_coeff, output_coeff = solution
    return {
        "base_ms": float(max(0.0, base)),
        "input_coeff_ms": float(max(0.0, input_coeff)),
        "output_coeff_ms": float(max(0.0, output_coeff)),
    }


def build_latency_db(
    prefill_rows: Sequence[Dict[str, Any]],
    decode_rows: Sequence[Dict[str, Any]],
    *,
    transfer_same_ms: float = 0.0,
    transfer_cross_ms: float = 0.0,
    hot_prefill_scale: float = 1.0,
) -> Dict[str, Any]:
    hardware: Dict[str, Dict[str, Any]] = {}
    worker_classes = sorted(
        {str(row["worker_class"]) for row in prefill_rows if row.get("worker_class")}
        | {str(row["worker_class"]) for row in decode_rows if row.get("worker_class")}
    )
    for worker_class in worker_classes:
        class_prefill = [row for row in prefill_rows if row.get("worker_class") == worker_class]
        class_decode = [row for row in decode_rows if row.get("worker_class") == worker_class]
        prefill_cold = _fit_linear(class_prefill, "prefill_time_ms")
        prefill_hot = {key: value * hot_prefill_scale for key, value in prefill_cold.items()}
        hardware[worker_class] = {
            "prefill_cold": prefill_cold,
            "prefill_hot": prefill_hot,
            "decode": _fit_linear(class_decode, "decode_time_ms"),
        }
    return {
        "hardware": hardware,
        "transfer": {
            "same_kind": {"base_ms": transfer_same_ms, "per_token_ms": 0.0},
            "cross_kind": {"base_ms": transfer_cross_ms, "per_token_ms": 0.0},
        },
    }


async def profile_main(args: argparse.Namespace) -> int:
    trace_calls = load_trace_calls(Path(args.trace_path), args.max_calls)
    if not trace_calls:
        raise ValueError(f"No calls found in {args.trace_path}")
    for warmup_call in trace_calls[: args.warmup_calls]:
        await measure_call(
            warmup_call,
            endpoint_url=args.endpoint_url,
            model=args.model,
            worker_class=args.worker_class,
            trial=-1,
            timeout_s=args.timeout_s,
            exact_output_length=not args.disable_exact_output_length,
        )
    for trial in range(args.trials):
        for index, trace_call in enumerate(trace_calls, start=1):
            measurement = await measure_call(
                trace_call,
                endpoint_url=args.endpoint_url,
                model=args.model,
                worker_class=args.worker_class,
                trial=trial,
                timeout_s=args.timeout_s,
                exact_output_length=not args.disable_exact_output_length,
            )
            if args.raw_output:
                append_jsonl(Path(args.raw_output), asdict(measurement))
            if measurement.error is None and args.phase in ("prefill", "both"):
                append_jsonl(Path(args.prefill_output), build_prefill_row(measurement))
            if measurement.error is None and args.phase in ("decode", "both"):
                append_jsonl(Path(args.decode_output), build_decode_row(measurement))
            status = "error" if measurement.error else "ok"
            print(f"[{status}] trial={trial} call={index}/{len(trace_calls)} id={measurement.call_id}", flush=True)
    return 0


def fit_latency_db_main(args: argparse.Namespace) -> int:
    latency_db = build_latency_db(
        read_jsonl(Path(args.prefill_path)),
        read_jsonl(Path(args.decode_path)),
        transfer_same_ms=args.transfer_same_ms,
        transfer_cross_ms=args.transfer_cross_ms,
        hot_prefill_scale=args.hot_prefill_scale,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(latency_db, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Hexgent trace profiler for plain Pod containers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--input-path")
    convert.add_argument(
        "--input-format",
        required=True,
        choices=["call_jsonl", "chat_jsonl", "workflow_jsonl", "sharegpt_json", "sharegpt_hf", "bfcl_hf", "hotpotqa_hf"],
    )
    convert.add_argument("--output-path", required=True)
    convert.add_argument("--max-calls", type=int)
    convert.add_argument("--hf-cache-dir")
    convert.add_argument("--bfcl-files", nargs="*")
    convert.add_argument("--hotpot-config", default=HOTPOTQA_HF_CONFIG)
    convert.add_argument("--hotpot-split", default=HOTPOTQA_HF_SPLIT)
    convert.add_argument("--hotpot-page-size", type=int, default=100)

    profile = subparsers.add_parser("profile")
    profile.add_argument("--trace-path", required=True)
    profile.add_argument("--endpoint-url", required=True)
    profile.add_argument("--model", required=True)
    profile.add_argument("--worker-class", required=True)
    profile.add_argument("--phase", choices=["prefill", "decode", "both"], default="both")
    profile.add_argument("--prefill-output", required=True)
    profile.add_argument("--decode-output", required=True)
    profile.add_argument("--raw-output")
    profile.add_argument("--max-calls", type=int)
    profile.add_argument("--trials", type=int, default=1)
    profile.add_argument("--warmup-calls", type=int, default=1)
    profile.add_argument("--timeout-s", type=float, default=600.0)
    profile.add_argument("--disable-exact-output-length", action="store_true")

    fit = subparsers.add_parser("fit-latency-db")
    fit.add_argument("--prefill-path", required=True)
    fit.add_argument("--decode-path", required=True)
    fit.add_argument("--output-path", required=True)
    fit.add_argument("--transfer-same-ms", type=float, default=0.0)
    fit.add_argument("--transfer-cross-ms", type=float, default=0.0)
    fit.add_argument("--hot-prefill-scale", type=float, default=1.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "convert":
        written = convert_trace(
            Path(args.input_path) if args.input_path else None,
            args.input_format,
            Path(args.output_path),
            args.max_calls,
            cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
            bfcl_files=args.bfcl_files,
            hotpot_config=args.hotpot_config,
            hotpot_split=args.hotpot_split,
            hotpot_page_size=args.hotpot_page_size,
        )
        print(f"Wrote {written} calls to {args.output_path}")
        return 0
    if args.command == "profile":
        return asyncio.run(profile_main(args))
    if args.command == "fit-latency-db":
        return fit_latency_db_main(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
