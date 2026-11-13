from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence
from urllib.parse import urlencode, quote
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


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "sglang" / "hexgent_traces"


def _hf_dataset_resolve_url(repo_id: str, filename: str) -> str:
    repo = quote(repo_id, safe="/")
    file_part = "/".join(quote(part, safe="") for part in filename.split("/"))
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{file_part}"


def _download_url_to_cache(
    url: str,
    *,
    cache_path: Path,
    timeout_s: float = 120.0,
) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path
    request = Request(url, headers={"User-Agent": "sglang-hexgent-trace-tool"})
    with urlopen(request, timeout=timeout_s) as response, cache_path.open("wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    return cache_path


def _iter_json_array(path: Path) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    idx = 0
    in_array = False
    eof = False

    with path.open("r", encoding="utf-8") as fh:
        while True:
            if idx >= len(buffer) and eof:
                return

            while True:
                if idx >= len(buffer):
                    if eof:
                        return
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        eof = True
                        continue
                    buffer += chunk
                    continue
                ch = buffer[idx]
                if ch in " \r\n\t,":
                    idx += 1
                    continue
                if not in_array:
                    if ch != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    in_array = True
                    idx += 1
                    continue
                if ch == "]":
                    return
                break

            try:
                obj, next_idx = decoder.raw_decode(buffer, idx)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = fh.read(1 << 20)
                if chunk:
                    buffer += chunk
                    continue
                eof = True
                continue

            if isinstance(obj, dict):
                yield obj
            idx = next_idx
            if idx > (1 << 20):
                buffer = buffer[idx:]
                idx = 0


def _iter_json_lines(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                yield record


def _fetch_json(url: str, *, timeout_s: float = 60.0) -> Dict[str, Any]:
    request = Request(url, headers={"User-Agent": "sglang-hexgent-trace-tool"})
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def load_sharegpt_hf_records(
    *,
    cache_dir: Optional[str | Path] = None,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    cache_root = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache_path = cache_root / "sharegpt" / SHAREGPT_HF_FILE
    downloaded = _download_url_to_cache(
        _hf_dataset_resolve_url(SHAREGPT_HF_REPO, SHAREGPT_HF_FILE),
        cache_path=cache_path,
    )
    rows: List[Dict[str, Any]] = []
    for record in _iter_json_array(downloaded):
        rows.append(record)
        if max_records is not None and len(rows) >= max_records:
            break
    return rows


def load_bfcl_hf_records(
    *,
    cache_dir: Optional[str | Path] = None,
    max_records: Optional[int] = None,
    bfcl_files: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    cache_root = Path(cache_dir) if cache_dir else _default_cache_dir()
    rows: List[Dict[str, Any]] = []
    for filename in bfcl_files or BFCL_V3_FILES:
        cache_path = cache_root / "bfcl" / filename
        downloaded = _download_url_to_cache(
            _hf_dataset_resolve_url(BFCL_HF_REPO, filename),
            cache_path=cache_path,
        )
        for record in _iter_json_lines(downloaded):
            rows.append(record)
            if max_records is not None and len(rows) >= max_records:
                return rows
    return rows


def load_hotpotqa_hf_rows(
    *,
    config: str = HOTPOTQA_HF_CONFIG,
    split: str = HOTPOTQA_HF_SPLIT,
    max_rows: Optional[int] = None,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    wanted = max_rows if max_rows is not None else float("inf")
    while len(rows) < wanted:
        params = urlencode(
            {
                "dataset": HOTPOTQA_HF_DATASET,
                "config": config,
                "split": split,
                "offset": offset,
                "length": min(page_size, int(wanted - len(rows)) if wanted != float("inf") else page_size),
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


def _default_call_id(prefix: str, index: int) -> str:
    return f"{prefix}-call-{index}"


def _coerce_messages(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [dict(message) for message in value]
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    raise ValueError(f"Unsupported message payload: {type(value)!r}")


def _count_message_tokens(messages: Sequence[Dict[str, Any]], tokenizer=None) -> int:
    if tokenizer is not None:
        tokenized = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        return len(tokenized)
    return sum(len(str(message.get("content", "")).split()) for message in messages) + 1


def _count_text_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(str(text).split())


def _maybe_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() == ".json":
            return json.load(fh)
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class WorkflowCall:
    call_id: str
    parents: List[str]
    request: Dict[str, Any]
    input_len: int
    output_len: int
    planner_hints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Replay-only wait; never sent as planner service or horizon input.
    tool_latency_s: float = 0.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "WorkflowCall":
        return cls(
            call_id=str(payload["call_id"]),
            parents=[str(parent) for parent in payload.get("parents", [])],
            request=dict(payload["request"]),
            input_len=int(payload["input_len"]),
            output_len=int(payload["output_len"]),
            planner_hints=dict(payload.get("planner_hints") or {}),
            metadata=dict(payload.get("metadata") or {}),
            tool_latency_s=float(payload.get("tool_latency_s", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowTrace:
    workflow_id: str
    calls: List[WorkflowCall]
    workflow_type: str = "generic"
    arrival_offset: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "WorkflowTrace":
        return cls(
            workflow_id=str(payload["workflow_id"]),
            calls=[WorkflowCall.from_dict(call) for call in payload.get("calls", [])],
            workflow_type=str(payload.get("workflow_type", "generic")),
            arrival_offset=(
                float(payload["arrival_offset"])
                if payload.get("arrival_offset") is not None
                else None
            ),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "arrival_offset": self.arrival_offset,
            "metadata": dict(self.metadata),
            "calls": [call.to_dict() for call in self.calls],
        }


@dataclass(frozen=True)
class ServiceProfileRow:
    model: str
    worker_class: str
    input_len: int
    output_len: int
    estimated_prefill_time: float
    estimated_decode_time: float
    estimated_transfer_time: float = 0.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ServiceProfileRow":
        prefill = payload.get("estimated_prefill_time")
        if prefill is None:
            prefill = payload.get("prefill_time_ms", payload.get("ttft_ms", 0.0))
        decode = payload.get("estimated_decode_time")
        if decode is None:
            decode = payload.get("decode_time_ms")
        if decode is None:
            total = payload.get("total_latency_ms")
            ttft = payload.get("ttft_ms")
            if total is not None and ttft is not None:
                decode = max(0.0, float(total) - float(ttft))
            else:
                decode = 0.0
        return cls(
            model=str(payload.get("model", "")),
            worker_class=str(payload.get("worker_class", "default")),
            input_len=int(payload["input_len"]),
            output_len=int(payload["output_len"]),
            estimated_prefill_time=float(prefill),
            estimated_decode_time=float(decode),
            estimated_transfer_time=float(
                payload.get(
                    "estimated_transfer_time",
                    payload.get("transfer_time_ms", 0.0),
                )
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ServiceProfileTable:
    def __init__(self, rows: Sequence[ServiceProfileRow]):
        self.rows = list(rows)

    @classmethod
    def from_rows(cls, rows: Sequence[Dict[str, Any] | ServiceProfileRow]):
        parsed = [
            row if isinstance(row, ServiceProfileRow) else ServiceProfileRow.from_dict(row)
            for row in rows
        ]
        return cls(parsed)

    @classmethod
    def from_path(cls, path: str | Path):
        data = _maybe_load_json(Path(path))
        if isinstance(data, dict):
            rows = data.get("profiles")
            if not isinstance(rows, list):
                rows = data.get("rows")
            if not isinstance(rows, list):
                rows = []
        else:
            rows = data
        return cls.from_rows(rows)

    def lookup(
        self,
        model: str,
        worker_class: str,
        input_len: int,
        output_len: int,
    ) -> Dict[str, float]:
        if not self.rows:
            raise KeyError("Service profile table is empty")

        candidates = [
            row for row in self.rows if row.model == model and row.worker_class == worker_class
        ]
        if not candidates:
            candidates = [row for row in self.rows if row.model == model]
        if not candidates:
            candidates = [row for row in self.rows if row.worker_class == worker_class]
        if not candidates:
            candidates = list(self.rows)

        exact = [
            row
            for row in candidates
            if row.input_len == int(input_len) and row.output_len == int(output_len)
        ]
        chosen = exact[0] if exact else min(
            candidates,
            key=lambda row: abs(row.input_len - int(input_len))
            + abs(row.output_len - int(output_len)),
        )
        return {
            "estimated_prefill_time": float(chosen.estimated_prefill_time),
            "estimated_decode_time": float(chosen.estimated_decode_time),
            "estimated_transfer_time": float(chosen.estimated_transfer_time),
        }

    def dump(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def convert_sharegpt_records(
    records: Sequence[Dict[str, Any]],
    *,
    tokenizer=None,
    model: Optional[str] = None,
) -> List[WorkflowTrace]:
    workflows: List[WorkflowTrace] = []
    for wf_index, record in enumerate(records):
        turns = record.get("conversations") or record.get("conversation") or []
        if len(turns) < 2:
            continue

        workflow_id = str(record.get("id") or record.get("workflow_id") or f"sharegpt-{wf_index}")
        calls: List[WorkflowCall] = []
        messages: List[Dict[str, str]] = []
        previous_call_id: Optional[str] = None

        for turn in turns:
            role = str(turn.get("from") or turn.get("role") or "").lower()
            content = str(turn.get("value") or turn.get("content") or "")
            if role in {"human", "user"}:
                messages.append({"role": "user", "content": content})
                continue
            if role not in {"gpt", "assistant"} or not messages:
                continue

            call_index = len(calls)
            call_id = _default_call_id(workflow_id, call_index)
            request = {
                "messages": [dict(message) for message in messages],
            }
            if model:
                request["model"] = model

            calls.append(
                WorkflowCall(
                    call_id=call_id,
                    parents=[previous_call_id] if previous_call_id else [],
                    request=request,
                    input_len=_count_message_tokens(request["messages"], tokenizer=tokenizer),
                    output_len=max(1, _count_text_tokens(content, tokenizer=tokenizer)),
                    planner_hints={},
                    metadata={"assistant_reference": content},
                )
            )
            previous_call_id = call_id
            messages.append({"role": "assistant", "content": content})

        if calls:
            workflows.append(
                WorkflowTrace(
                    workflow_id=workflow_id,
                    calls=calls,
                    workflow_type="sharegpt",
                    metadata={"source_id": record.get("id")},
                )
            )
    return workflows


def convert_openai_request_records(
    records: Sequence[Dict[str, Any]],
    *,
    tokenizer=None,
) -> List[WorkflowTrace]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for index, record in enumerate(records):
        workflow_id = str(
            record.get("conversation_id")
            or record.get("workflow_id")
            or record.get("session_id")
            or f"openai-{index}"
        )
        if workflow_id not in grouped:
            grouped[workflow_id] = []
            order.append(workflow_id)
        grouped[workflow_id].append(dict(record))

    workflows: List[WorkflowTrace] = []
    for workflow_id in order:
        calls: List[WorkflowCall] = []
        previous_call_id: Optional[str] = None
        for index, record in enumerate(grouped[workflow_id]):
            messages = _coerce_messages(record.get("messages") or record.get("prompt") or [])
            output_len = int(
                record.get("max_tokens")
                or record.get("max_completion_tokens")
                or record.get("output_len")
                or 256
            )
            call_id = str(record.get("call_id") or _default_call_id(workflow_id, index))
            excluded = {
                "conversation_id",
                "workflow_id",
                "session_id",
                "call_id",
                "messages",
                "prompt",
                "max_tokens",
                "max_completion_tokens",
                "output_len",
            }
            request = {k: v for k, v in record.items() if k not in excluded}
            request["messages"] = messages
            calls.append(
                WorkflowCall(
                    call_id=call_id,
                    parents=[previous_call_id] if previous_call_id else [],
                    request=request,
                    input_len=_count_message_tokens(messages, tokenizer=tokenizer),
                    output_len=output_len,
                    planner_hints=dict(record.get("hexgent") or {}),
                )
            )
            previous_call_id = call_id
        workflows.append(
            WorkflowTrace(
                workflow_id=workflow_id,
                calls=calls,
                workflow_type="openai_jsonl",
            )
        )
    return workflows


def convert_bfcl_records(
    records: Sequence[Dict[str, Any]],
    *,
    tokenizer=None,
    model: str = "bfcl-v3",
) -> List[WorkflowTrace]:
    workflows: List[WorkflowTrace] = []
    for wf_index, record in enumerate(records):
        questions = record.get("question") or []
        if not isinstance(questions, list):
            continue
        turns: List[List[Dict[str, Any]]] = []
        for turn in questions:
            if isinstance(turn, list):
                turns.append([message for message in turn if isinstance(message, dict)])
            elif isinstance(turn, dict):
                turns.append([turn])
        turns = [turn for turn in turns if turn]
        if not turns:
            continue

        function_payload = record.get("function") or []
        ground_truth = record.get("ground_truth") or []
        path_seq = record.get("path") or []
        functions_text = _as_text(function_payload)

        calls: List[WorkflowCall] = []
        previous_call_id: Optional[str] = None
        for turn_index, turn_messages in enumerate(turns):
            think_call_id = f"{record.get('id', f'bfcl-{wf_index}')}-think-{turn_index}"
            think_request = {
                "messages": [dict(message) for message in turn_messages],
                "model": model,
                "tools": function_payload,
            }
            think_call = WorkflowCall(
                call_id=think_call_id,
                parents=[previous_call_id] if previous_call_id else [],
                request=think_request,
                input_len=_count_message_tokens(think_request["messages"], tokenizer=tokenizer)
                + _count_text_tokens(functions_text, tokenizer=tokenizer),
                output_len=64,
                planner_hints={},
                metadata={"stage": "think", "turn_index": turn_index},
            )
            calls.append(think_call)

            tool_texts: List[str]
            if len(turns) == 1 and isinstance(ground_truth, list) and ground_truth:
                tool_texts = [_as_text(item) for item in ground_truth]
            elif turn_index < len(path_seq):
                tool_texts = [_as_text(path_seq[turn_index])]
            elif turn_index < len(ground_truth):
                tool_texts = [_as_text(ground_truth[turn_index])]
            else:
                tool_texts = ["tool_call"]

            stage_parent = think_call_id
            if len(tool_texts) == 1:
                tool_request = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Execute tool call: {tool_texts[0]}",
                        }
                    ],
                    "model": model,
                }
                tool_call_id = f"{record.get('id', f'bfcl-{wf_index}')}-tool-{turn_index}"
                tool_call = WorkflowCall(
                    call_id=tool_call_id,
                    parents=[think_call_id],
                    request=tool_request,
                    input_len=_count_message_tokens(tool_request["messages"], tokenizer=tokenizer),
                    output_len=max(20, _count_text_tokens(tool_texts[0], tokenizer=tokenizer)),
                    planner_hints={},
                    metadata={"stage": "tool", "turn_index": turn_index},
                )
                calls.append(tool_call)
                stage_parent = tool_call_id
            else:
                tool_ids: List[str] = []
                for tool_index, tool_text in enumerate(tool_texts):
                    tool_request = {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Execute parallel tool call: {tool_text}",
                            }
                        ],
                        "model": model,
                    }
                    tool_call_id = (
                        f"{record.get('id', f'bfcl-{wf_index}')}-tool-{turn_index}-{tool_index}"
                    )
                    calls.append(
                        WorkflowCall(
                            call_id=tool_call_id,
                            parents=[think_call_id],
                            request=tool_request,
                            input_len=_count_message_tokens(tool_request["messages"], tokenizer=tokenizer),
                            output_len=max(20, _count_text_tokens(tool_text, tokenizer=tokenizer)),
                            planner_hints={},
                            metadata={
                                "stage": "tool",
                                "turn_index": turn_index,
                                "tool_index": tool_index,
                            },
                        )
                    )
                    tool_ids.append(tool_call_id)

                synth_request = {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Synthesize the tool results into the next dialogue state.",
                        }
                    ],
                    "model": model,
                }
                synth_call_id = f"{record.get('id', f'bfcl-{wf_index}')}-synthesize-{turn_index}"
                calls.append(
                    WorkflowCall(
                        call_id=synth_call_id,
                        parents=tool_ids,
                        request=synth_request,
                        input_len=_count_message_tokens(synth_request["messages"], tokenizer=tokenizer),
                        output_len=96,
                        planner_hints={},
                        metadata={"stage": "synthesize", "turn_index": turn_index},
                    )
                )
                stage_parent = synth_call_id

            previous_call_id = stage_parent

        workflows.append(
            WorkflowTrace(
                workflow_id=str(record.get("id") or f"bfcl-{wf_index}"),
                calls=calls,
                workflow_type="bfcl",
                metadata={"source": "bfcl_hf"},
            )
        )
    return workflows


def convert_hotpotqa_rows_to_lats_style(
    rows: Sequence[Dict[str, Any]],
    *,
    tokenizer=None,
    model: str = "hotpotqa-lats",
    max_calls_per_workflow: int = 64,
) -> List[WorkflowTrace]:
    workflows: List[WorkflowTrace] = []
    for wf_index, row in enumerate(rows):
        question = _as_text(row.get("question"))
        answer = _as_text(row.get("answer"))
        if not question:
            continue

        level = str(row.get("level", "")).lower()
        depth = 4 if level == "hard" else 3 if level == "medium" else 2
        support = row.get("supporting_facts") or {}
        titles = support.get("title") if isinstance(support, dict) else []
        if not isinstance(titles, list):
            titles = []
        branch = min(4, max(2, len({str(title) for title in titles if str(title).strip()}) or 2))

        context = row.get("context") or {}
        snippets: List[str] = []
        if isinstance(context, dict):
            context_titles = context.get("title") or []
            context_sentences = context.get("sentences") or []
            if isinstance(context_titles, list) and isinstance(context_sentences, list):
                for index, title in enumerate(context_titles):
                    sentence_group = context_sentences[index] if index < len(context_sentences) else []
                    if isinstance(sentence_group, list):
                        snippet = " ".join(str(sentence) for sentence in sentence_group[:2])
                    else:
                        snippet = str(sentence_group)
                    snippets.append(f"{title}: {snippet}".strip())
        if not snippets:
            snippets = [question]

        workflow_id = str(row.get("id") or f"hotpotqa-{wf_index}")
        calls: List[WorkflowCall] = []
        root_request = {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "model": model,
        }
        root_id = f"{workflow_id}-root"
        calls.append(
            WorkflowCall(
                call_id=root_id,
                parents=[],
                request=root_request,
                input_len=_count_message_tokens(root_request["messages"], tokenizer=tokenizer),
                output_len=max(200, min(600, _count_text_tokens(answer, tokenizer=tokenizer) * 6 or 200)),
                planner_hints={},
                metadata={"stage": "root"},
            )
        )

        current_level = [root_id]
        stop = False
        for depth_index in range(depth):
            next_level: List[str] = []
            for parent_id in current_level:
                for branch_index in range(branch):
                    if len(calls) >= max_calls_per_workflow - 1:
                        stop = True
                        break
                    snippet = snippets[(depth_index * branch + branch_index) % len(snippets)]
                    expand_id = f"{workflow_id}-expand-{depth_index}-{branch_index}-{len(calls)}"
                    expand_request = {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Question: {question}\nEvidence: {snippet}\nExpand the reasoning tree.",
                            }
                        ],
                        "model": model,
                    }
                    calls.append(
                        WorkflowCall(
                            call_id=expand_id,
                            parents=[parent_id],
                            request=expand_request,
                            input_len=_count_message_tokens(expand_request["messages"], tokenizer=tokenizer),
                            output_len=max(150, min(600, _count_text_tokens(snippet, tokenizer=tokenizer) + 120)),
                            planner_hints={},
                            metadata={
                                "stage": "expand",
                                "depth": depth_index,
                                "branch": branch_index,
                            },
                        )
                    )
                    next_level.append(expand_id)
                if stop:
                    break
            current_level = next_level
            if stop or not current_level:
                break

        if current_level:
            aggregate_id = f"{workflow_id}-aggregate"
            aggregate_request = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Aggregate the explored reasoning tree for question: {question}",
                    }
                ],
                "model": model,
            }
            calls.append(
                WorkflowCall(
                    call_id=aggregate_id,
                    parents=list(current_level),
                    request=aggregate_request,
                    input_len=_count_message_tokens(aggregate_request["messages"], tokenizer=tokenizer),
                    output_len=max(200, min(500, _count_text_tokens(answer, tokenizer=tokenizer) * 4 or 200)),
                    planner_hints={},
                    metadata={"stage": "aggregate"},
                )
            )

        workflows.append(
            WorkflowTrace(
                workflow_id=workflow_id,
                calls=calls,
                workflow_type="lats",
                metadata={"source": "hotpotqa_hf", "level": level or "unknown"},
            )
        )
    return workflows


def convert_call_jsonl_records(records: Sequence[Dict[str, Any]]) -> List[WorkflowTrace]:
    grouped: Dict[str, List[WorkflowCall]] = {}
    workflow_meta: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for index, record in enumerate(records):
        workflow_id = str(record.get("workflow_id") or f"workflow-{index}")
        if workflow_id not in grouped:
            grouped[workflow_id] = []
            order.append(workflow_id)
            workflow_meta[workflow_id] = {
                "workflow_type": str(record.get("workflow_type", "call_jsonl")),
                "arrival_offset": record.get("arrival_offset"),
                "metadata": dict(record.get("workflow_metadata") or {}),
            }
        grouped[workflow_id].append(
            WorkflowCall(
                call_id=str(record.get("call_id") or _default_call_id(workflow_id, len(grouped[workflow_id]))),
                parents=[str(parent) for parent in record.get("parents", [])],
                request=dict(record["request"]),
                input_len=int(record["input_len"]),
                output_len=int(record["output_len"]),
                planner_hints=dict(record.get("planner_hints") or record.get("hexgent") or {}),
                metadata=dict(record.get("metadata") or {}),
                tool_latency_s=float(record.get("tool_latency_s", 0.0)),
            )
        )

    workflows: List[WorkflowTrace] = []
    for workflow_id in order:
        meta = workflow_meta[workflow_id]
        workflows.append(
            WorkflowTrace(
                workflow_id=workflow_id,
                calls=grouped[workflow_id],
                workflow_type=meta["workflow_type"],
                arrival_offset=(
                    float(meta["arrival_offset"])
                    if meta["arrival_offset"] is not None
                    else None
                ),
                metadata=meta["metadata"],
            )
        )
    return workflows


def _profile_row_value(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row and row.get(key) is not None:
        return row.get(key)
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return default


def _unique_call_id(raw_call_id: str, used: Dict[str, int]) -> str:
    count = used.get(raw_call_id, 0)
    used[raw_call_id] = count + 1
    if count == 0:
        return raw_call_id
    return f"{raw_call_id}__dup{count}"


def _request_from_profile_row(row: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
    request: Dict[str, Any]
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        request = {"messages": messages}
    elif row.get("prompt") is not None:
        request = {"messages": [{"role": "user", "content": str(row.get("prompt") or "")}]}
    else:
        request = {"messages": [{"role": "user", "content": ""}]}
    if model:
        request["model"] = model
    return request


def _profile_row_has_replayable_request(row: Dict[str, Any]) -> bool:
    """Return whether a profiled row can be sent to chat completions unchanged."""
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return all(
            isinstance(message, dict)
            and str(message.get("content") or "").strip()
            for message in messages
        )
    if row.get("prompt") is not None:
        return bool(str(row.get("prompt") or "").strip())
    return False


def _profile_planner_hints(
    row: Dict[str, Any], *, input_tokens: int, output_tokens: int
) -> Dict[str, Any]:
    hints: Dict[str, Any] = {
        "input_tokens": int(input_tokens),
        "predicted_output_tokens": int(output_tokens),
    }
    exclusive_latency = row.get("total_latency_ms")
    if exclusive_latency is not None:
        try:
            hints["exclusive_profile_latency_ms"] = float(exclusive_latency)
        except (TypeError, ValueError):
            pass
    return hints


def convert_profile_rows_to_workflows(
    records: Sequence[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    max_output_len: Optional[int] = None,
) -> List[WorkflowTrace]:
    """Convert profiled per-call rows into canonical replay workflows.

    The profiling rows are already ordered by trace execution. Rows without
    explicit dependency metadata are treated as a serial stage. Rows sharing a
    fanout_group_id are treated as parallel calls that depend on the previous
    stage barrier or on parent_call_id when provided by the LATS collector.
    """

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for index, record in enumerate(records):
        if record.get("error"):
            continue
        workflow_id = str(record.get("workflow_id") or f"workflow-{index}")
        row = dict(record)
        row["_profile_row_index"] = index
        if workflow_id not in grouped:
            grouped[workflow_id] = []
            order.append(workflow_id)
        grouped[workflow_id].append(row)

    workflows: List[WorkflowTrace] = []
    for workflow_id in order:
        rows = grouped[workflow_id]
        # A blank turn is rejected by SGLang's chat-completions validator. Do
        # not retain a partial conversation: later turns depend on its history
        # and would no longer reproduce the profiled workflow faithfully.
        if any(not _profile_row_has_replayable_request(row) for row in rows):
            continue
        calls: List[WorkflowCall] = []
        used_call_ids: Dict[str, int] = {}
        raw_to_unique: Dict[str, str] = {}
        previous_barrier: List[str] = []
        i = 0
        while i < len(rows):
            row = rows[i]
            fanout_group_id = _profile_row_value(row, "fanout_group_id")
            if fanout_group_id:
                group_rows = [row]
                i += 1
                while i < len(rows) and _profile_row_value(rows[i], "fanout_group_id") == fanout_group_id:
                    group_rows.append(rows[i])
                    i += 1

                parent_call_id = _profile_row_value(group_rows[0], "parent_call_id")
                if parent_call_id:
                    parents = [raw_to_unique.get(str(parent_call_id), str(parent_call_id))]
                else:
                    parents = list(previous_barrier)

                group_call_ids: List[str] = []
                for group_row in group_rows:
                    raw_call_id = str(
                        group_row.get("call_id")
                        or _default_call_id(workflow_id, int(group_row["_profile_row_index"]))
                    )
                    call_id = _unique_call_id(raw_call_id, used_call_ids)
                    raw_to_unique[raw_call_id] = call_id
                    group_call_ids.append(call_id)
                    output_len = int(group_row.get("output_tokens") or group_row.get("output_len") or 1)
                    if max_output_len is not None:
                        output_len = max(1, min(output_len, int(max_output_len)))
                    calls.append(
                        WorkflowCall(
                            call_id=call_id,
                            parents=list(parents),
                            request=_request_from_profile_row(group_row, model),
                            input_len=int(group_row.get("input_tokens") or group_row.get("input_len") or 1),
                            output_len=output_len,
                            planner_hints=_profile_planner_hints(
                                group_row,
                                input_tokens=int(
                                    group_row.get("input_tokens")
                                    or group_row.get("input_len")
                                    or 1
                                ),
                                output_tokens=output_len,
                            ),
                            metadata=dict(group_row.get("metadata") or {}),
                            tool_latency_s=float(_profile_row_value(group_row, "tool_latency_s", 0.0)),
                        )
                    )
                previous_barrier = group_call_ids
                continue

            raw_call_id = str(
                row.get("call_id") or _default_call_id(workflow_id, int(row["_profile_row_index"]))
            )
            call_id = _unique_call_id(raw_call_id, used_call_ids)
            raw_to_unique[raw_call_id] = call_id
            output_len = int(row.get("output_tokens") or row.get("output_len") or 1)
            if max_output_len is not None:
                output_len = max(1, min(output_len, int(max_output_len)))
            calls.append(
                WorkflowCall(
                    call_id=call_id,
                    parents=list(previous_barrier),
                    request=_request_from_profile_row(row, model),
                    input_len=int(row.get("input_tokens") or row.get("input_len") or 1),
                    output_len=output_len,
                    planner_hints=_profile_planner_hints(
                        row,
                        input_tokens=int(
                            row.get("input_tokens") or row.get("input_len") or 1
                        ),
                        output_tokens=output_len,
                    ),
                    metadata=dict(row.get("metadata") or {}),
                    tool_latency_s=float(_profile_row_value(row, "tool_latency_s", 0.0)),
                )
            )
            previous_barrier = [call_id]
            i += 1

        workflows.append(
            WorkflowTrace(
                workflow_id=workflow_id,
                calls=calls,
                workflow_type=str(rows[0].get("dataset") or "profile_jsonl"),
                metadata={"source": "profile_jsonl"},
            )
        )
    return workflows


def load_workflow_traces_jsonl(path: str | Path) -> List[WorkflowTrace]:
    rows = _maybe_load_json(Path(path))
    if isinstance(rows, dict):
        rows = rows.get("workflows") or []
    return [WorkflowTrace.from_dict(row) for row in rows]


def dump_workflow_traces_jsonl(workflows: Sequence[WorkflowTrace], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for workflow in workflows:
            fh.write(json.dumps(workflow.to_dict(), ensure_ascii=False) + "\n")


def generate_poisson_arrival_offsets(
    *,
    num_workflows: int,
    rate: float,
    seed: int = 42,
) -> List[float]:
    if num_workflows < 0:
        raise ValueError("num_workflows must be non-negative")
    if rate <= 0.0:
        return [0.0 for _ in range(num_workflows)]

    rng = random.Random(seed)
    offsets: List[float] = []
    current = 0.0
    for _ in range(num_workflows):
        current += rng.expovariate(rate)
        offsets.append(current)
    return offsets


def build_chat_completion_payload(
    *,
    workflow_id: str,
    call_id: str,
    request: Dict[str, Any],
    output_len: int,
    input_len: Optional[int] = None,
    workflow_arrival_ms: Optional[int] = None,
    workflow_horizon_ms: Optional[int] = None,
    parent_call_ids: Optional[Sequence[str]] = None,
    planner_hints: Optional[Dict[str, Any]] = None,
    exact_output_length: bool = True,
) -> Dict[str, Any]:
    payload = dict(request)
    payload["conversation_id"] = workflow_id
    payload["hexgent_call_id"] = call_id
    payload["max_tokens"] = int(output_len)
    if exact_output_length:
        payload["ignore_eos"] = True
    merged_hints = dict(planner_hints or {})
    merged_hints.setdefault("estimated_output_length", int(output_len))
    merged_hints.setdefault("workflow_id", workflow_id)
    merged_hints.setdefault("call_id", call_id)
    merged_hints.setdefault(
        "workflow_arrival_ms",
        int(workflow_arrival_ms if workflow_arrival_ms is not None else time.time() * 1000),
    )
    if workflow_horizon_ms is not None:
        merged_hints.setdefault("workflow_horizon_ms", int(workflow_horizon_ms))
    if parent_call_ids is not None:
        merged_hints.setdefault("parent_call_ids", [str(parent) for parent in parent_call_ids])
    merged_hints.setdefault("predicted_output_tokens", int(output_len))
    if input_len is not None:
        merged_hints.setdefault("input_tokens", int(input_len))
    if merged_hints:
        payload["hexgent"] = merged_hints
        # Keep calibrated stage times as top-level fields too. The Rust router
        # forwards these fields to workers, while the nested object remains the
        # canonical planner metadata for compatibility.
        for hint_key, request_key in (
            ("estimated_prefill_time", "hexgent_estimated_prefill_time"),
            ("estimated_decode_time", "hexgent_estimated_decode_time"),
            ("estimated_transfer_time", "hexgent_estimated_transfer_time"),
        ):
            if merged_hints.get(hint_key) is not None:
                payload[request_key] = float(merged_hints[hint_key])
    return payload


def _build_call_planner_hints(
    call: WorkflowCall,
    *,
    model: str,
    worker_class: str,
    profile_table: Optional[ServiceProfileTable],
) -> Dict[str, Any]:
    hints = dict(call.planner_hints)
    hints.setdefault("estimated_output_length", int(call.output_len))
    if profile_table is not None:
        looked_up = profile_table.lookup(model, worker_class, call.input_len, call.output_len)
        hints = {**looked_up, **hints}
    return hints


async def _post_json(
    session, url: str, payload: Dict[str, Any], timeout_s: float
) -> tuple[Dict[str, Any], Dict[str, str]]:
    async with session.post(url, json=payload, timeout=timeout_s) as response:
        text = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status} for {url}: {text}")
        headers = {key.lower(): value for key, value in response.headers.items()}
        if not text:
            return {}, headers
        try:
            return json.loads(text), headers
        except json.JSONDecodeError:
            return {"raw_text": text}, headers


class WorkflowTraceReplayer:
    def __init__(
        self,
        *,
        router_url: str,
        profile_table: Optional[ServiceProfileTable] = None,
        model: Optional[str] = None,
        worker_class: str = "default",
        exact_output_length: bool = True,
        timeout_s: float = 3600.0,
        endpoint: str = "/v1/chat/completions",
        run_namespace: Optional[str] = None,
    ):
        self.router_url = router_url.rstrip("/")
        self.profile_table = profile_table
        self.model = model
        self.worker_class = worker_class
        self.exact_output_length = exact_output_length
        self.timeout_s = timeout_s
        self.endpoint = endpoint
        self.run_namespace = str(run_namespace or "").strip()

    def _runtime_id(self, value: str) -> str:
        if not self.run_namespace:
            return str(value)
        return f"{self.run_namespace}::{value}"

    async def replay(
        self,
        workflows: Sequence[WorkflowTrace],
        *,
        arrival_rate: Optional[float] = None,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        import aiohttp

        offsets = self._resolve_arrival_offsets(workflows, arrival_rate=arrival_rate, seed=seed)
        start = time.perf_counter()
        results: List[Dict[str, Any]] = []
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(
                    self._run_workflow(session, workflow, start + offsets[index], results)
                )
                for index, workflow in enumerate(workflows)
            ]
            await asyncio.gather(*tasks)
        return sorted(results, key=lambda row: row["submitted_at"])

    def _resolve_arrival_offsets(
        self,
        workflows: Sequence[WorkflowTrace],
        *,
        arrival_rate: Optional[float],
        seed: int,
    ) -> List[float]:
        if arrival_rate is not None:
            return generate_poisson_arrival_offsets(
                num_workflows=len(workflows),
                rate=float(arrival_rate),
                seed=seed,
            )
        resolved = []
        for workflow in workflows:
            resolved.append(float(workflow.arrival_offset or 0.0))
        return resolved

    async def _run_workflow(
        self,
        session,
        workflow: WorkflowTrace,
        scheduled_start: float,
        results: List[Dict[str, Any]],
    ) -> None:
        await asyncio.sleep(max(0.0, scheduled_start - time.perf_counter()))
        workflow_arrival_ms = int(time.time() * 1000)
        completed: set[str] = set()
        launched: Dict[str, asyncio.Task] = {}
        calls_by_id = {call.call_id: call for call in workflow.calls}

        while len(completed) < len(workflow.calls):
            for call in workflow.calls:
                if call.call_id in launched or call.call_id in completed:
                    continue
                if all(parent in completed for parent in call.parents):
                    launched[call.call_id] = asyncio.create_task(
                        self._run_call(
                            session,
                            workflow.workflow_id,
                            call,
                            results,
                            workflow_arrival_ms=workflow_arrival_ms,
                            workflow_horizon_ms=None,
                        )
                    )

            if not launched:
                missing = sorted(set(calls_by_id) - completed)
                raise RuntimeError(
                    f"Workflow {workflow.workflow_id} has unresolved dependencies: {missing}"
                )

            done, _ = await asyncio.wait(
                tuple(launched.values()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                call_id = task.result()
                completed.add(call_id)
                launched.pop(call_id, None)

    async def _run_call(
        self,
        session,
        workflow_id: str,
        call: WorkflowCall,
        results: List[Dict[str, Any]],
        workflow_arrival_ms: int,
        workflow_horizon_ms: Optional[int],
    ) -> str:
        request = dict(call.request)
        # A replay model override must replace the model embedded in a trace;
        # otherwise the router may reject the request as an unknown worker model.
        if self.model:
            request["model"] = self.model
        model = str(request.get("model") or "")
        hints = _build_call_planner_hints(
            call,
            model=model,
            worker_class=self.worker_class,
            profile_table=self.profile_table,
        )
        runtime_workflow_id = self._runtime_id(workflow_id)
        runtime_call_id = self._runtime_id(call.call_id)
        payload = build_chat_completion_payload(
            workflow_id=runtime_workflow_id,
            call_id=runtime_call_id,
            request=request,
            output_len=call.output_len,
            input_len=call.input_len,
            workflow_arrival_ms=workflow_arrival_ms,
            # The planner derives its own horizon from revealed dependencies.
            # Keep the argument for API compatibility but never replay it.
            workflow_horizon_ms=None,
            parent_call_ids=[self._runtime_id(parent) for parent in call.parents],
            planner_hints=hints,
            exact_output_length=self.exact_output_length,
        )
        submitted_at = time.time()
        started = time.perf_counter()
        response_headers: Dict[str, str] = {}
        try:
            post_result = await _post_json(
                session,
                f"{self.router_url}{self.endpoint}",
                payload,
                timeout_s=self.timeout_s,
            )
            if isinstance(post_result, tuple):
                response, response_headers = post_result
            else:
                # Preserve compatibility with external test/replay adapters.
                response = post_result
            error = None
        except Exception as exc:
            response = None
            error = f"{type(exc).__name__}: {exc}"
        completed_at = time.time()
        total_latency_ms = (time.perf_counter() - started) * 1000.0

        def optional_int_header(name: str) -> Optional[int]:
            value = response_headers.get(name)
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        selection_wall_us = optional_int_header("x-hexgent-selection-wall-us")
        snapshot_fetch_us = optional_int_header("x-hexgent-snapshot-fetch-us")
        admission_wait_us = optional_int_header("x-hexgent-admission-wait-us")
        results.append(
            {
                "workflow_id": workflow_id,
                "call_id": call.call_id,
                "workflow_arrival_ms": workflow_arrival_ms,
                "submitted_at": submitted_at,
                "request_started_at_ms": submitted_at * 1000.0,
                "response_completed_at_ms": completed_at * 1000.0,
                "total_latency_ms": total_latency_ms,
                "input_tokens": int(call.input_len),
                "output_tokens": int(call.output_len),
                "parents": list(call.parents),
                "exclusive_profile_latency_ms": call.planner_hints.get(
                    "exclusive_profile_latency_ms"
                ),
                "prefill_worker": response_headers.get(
                    "x-sglang-prefill-worker"
                ),
                "decode_worker": response_headers.get("x-sglang-decode-worker"),
                "router_mode": response_headers.get("x-hexgent-router-mode"),
                "schedule_revision": optional_int_header(
                    "x-hexgent-schedule-revision"
                ),
                "router_selection_wall_ms": (
                    selection_wall_us / 1000.0
                    if selection_wall_us is not None
                    else None
                ),
                "router_snapshot_fetch_ms": (
                    snapshot_fetch_us / 1000.0
                    if snapshot_fetch_us is not None
                    else None
                ),
                "router_admission_wait_ms": (
                    admission_wait_us / 1000.0
                    if admission_wait_us is not None
                    else None
                ),
                "response": response,
                "error": error,
            }
        )
        # Capture our row before yielding to other workflows during a tool wait.
        result = results[-1]
        result["tool_latency_s"] = call.tool_latency_s
        result["tool_started_at_ms"] = None
        result["tool_completed_at_ms"] = None
        result["call_completed_at_ms"] = None
        if error is None and call.tool_latency_s > 0.0:
            result["tool_started_at_ms"] = time.time() * 1000.0
            await asyncio.sleep(call.tool_latency_s)
            result["tool_completed_at_ms"] = time.time() * 1000.0
            result["call_completed_at_ms"] = result["tool_completed_at_ms"]
        else:
            result["call_completed_at_ms"] = result["response_completed_at_ms"]
        return call.call_id


def _load_tokenizer(tokenizer_path: Optional[str]):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _read_input_records(path: str | Path) -> Any:
    return _maybe_load_json(Path(path))


def _convert_records(
    records: Any,
    *,
    input_format: str,
    tokenizer=None,
    model: Optional[str] = None,
) -> List[WorkflowTrace]:
    if input_format in {"sharegpt", "sharegpt_hf"}:
        if isinstance(records, dict):
            records = records.get("data") or records.get("conversations") or []
        return convert_sharegpt_records(records, tokenizer=tokenizer, model=model)
    if input_format == "bfcl_hf":
        if isinstance(records, dict):
            records = records.get("rows") or records.get("records") or []
        return convert_bfcl_records(records, tokenizer=tokenizer)
    if input_format == "hotpotqa_hf":
        if isinstance(records, dict):
            records = records.get("rows") or records.get("records") or []
        return convert_hotpotqa_rows_to_lats_style(records, tokenizer=tokenizer)
    if input_format == "openai_jsonl":
        if isinstance(records, dict):
            records = records.get("requests") or []
        return convert_openai_request_records(records, tokenizer=tokenizer)
    if input_format == "call_jsonl":
        if isinstance(records, dict):
            records = records.get("calls") or []
        return convert_call_jsonl_records(records)
    if input_format == "profile_jsonl":
        if isinstance(records, dict):
            records = records.get("rows") or records.get("calls") or []
        return convert_profile_rows_to_workflows(records, model=model)
    if input_format == "workflow_jsonl":
        if isinstance(records, dict):
            records = records.get("workflows") or []
        return [WorkflowTrace.from_dict(row) for row in records]
    raise ValueError(f"Unsupported input format: {input_format}")


def _profile_rows_from_records(records: Any) -> List[ServiceProfileRow]:
    if isinstance(records, dict):
        records = records.get("rows") or records.get("profiles") or []
    return [ServiceProfileRow.from_dict(record) for record in records]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hexgent trace conversion and replay tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert a dataset into canonical workflow JSONL.")
    convert.add_argument("--input-path")
    convert.add_argument(
        "--input-format",
        required=True,
        choices=[
            "sharegpt",
            "sharegpt_hf",
            "bfcl_hf",
            "hotpotqa_hf",
            "openai_jsonl",
            "call_jsonl",
            "profile_jsonl",
            "workflow_jsonl",
        ],
    )
    convert.add_argument("--output-path", required=True)
    convert.add_argument("--tokenizer-path")
    convert.add_argument("--model")
    convert.add_argument("--hf-cache-dir")
    convert.add_argument("--max-workflows", type=int)
    convert.add_argument("--bfcl-files", nargs="*")
    convert.add_argument("--hotpot-config", default=HOTPOTQA_HF_CONFIG)
    convert.add_argument("--hotpot-split", default=HOTPOTQA_HF_SPLIT)
    convert.add_argument("--hotpot-page-size", type=int, default=100)
    convert.add_argument(
        "--max-output-len",
        type=int,
        help="Optional output_len cap when converting profiling rows for smoke replay.",
    )

    profile = subparsers.add_parser("build-profile", help="Normalize service profile rows into JSONL.")
    profile.add_argument("--input-path", required=True)
    profile.add_argument("--output-path", required=True)

    replay = subparsers.add_parser("replay", help="Replay canonical workflow traces against a live router.")
    replay.add_argument("--trace-path", required=True)
    replay.add_argument("--router-url", required=True)
    replay.add_argument("--profile-path")
    replay.add_argument("--arrival-rate", type=float)
    replay.add_argument("--seed", type=int, default=42)
    replay.add_argument("--model")
    replay.add_argument("--worker-class", default="default")
    replay.add_argument("--endpoint", default="/v1/chat/completions")
    replay.add_argument("--timeout-s", type=float, default=3600.0)
    replay.add_argument("--output-path")
    replay.add_argument(
        "--run-namespace",
        help=(
            "Namespace runtime workflow/call IDs while preserving original IDs "
            "in replay output and profile lookup."
        ),
    )
    replay.add_argument(
        "--disable-exact-output-length",
        action="store_true",
        help="Do not force ignore_eos=true with the trace max_tokens budget.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert":
        tokenizer = _load_tokenizer(args.tokenizer_path)
        if args.input_format == "sharegpt_hf":
            records = load_sharegpt_hf_records(
                cache_dir=args.hf_cache_dir,
                max_records=args.max_workflows,
            )
        elif args.input_format == "bfcl_hf":
            records = load_bfcl_hf_records(
                cache_dir=args.hf_cache_dir,
                max_records=args.max_workflows,
                bfcl_files=args.bfcl_files,
            )
        elif args.input_format == "hotpotqa_hf":
            records = load_hotpotqa_hf_rows(
                config=args.hotpot_config,
                split=args.hotpot_split,
                max_rows=args.max_workflows,
                page_size=args.hotpot_page_size,
            )
        else:
            if not args.input_path:
                parser.error("--input-path is required for this input format")
            records = _read_input_records(args.input_path)
        if args.input_format == "profile_jsonl":
            if isinstance(records, dict):
                records = records.get("rows") or records.get("calls") or []
            workflows = convert_profile_rows_to_workflows(
                records,
                model=args.model,
                max_output_len=args.max_output_len,
            )
            if args.max_workflows is not None:
                workflows = workflows[: args.max_workflows]
        else:
            workflows = _convert_records(
                records,
                input_format=args.input_format,
                tokenizer=tokenizer,
                model=args.model,
            )
        dump_workflow_traces_jsonl(workflows, args.output_path)
        print(f"Wrote {len(workflows)} workflows to {args.output_path}")
        return 0

    if args.command == "build-profile":
        records = _read_input_records(args.input_path)
        table = ServiceProfileTable(_profile_rows_from_records(records))
        table.dump(args.output_path)
        print(f"Wrote {len(table.rows)} profile rows to {args.output_path}")
        return 0

    if args.command == "replay":
        workflows = load_workflow_traces_jsonl(args.trace_path)
        profile_table = (
            ServiceProfileTable.from_path(args.profile_path)
            if args.profile_path
            else None
        )
        replayer = WorkflowTraceReplayer(
            router_url=args.router_url,
            profile_table=profile_table,
            model=args.model,
            worker_class=args.worker_class,
            exact_output_length=not args.disable_exact_output_length,
            timeout_s=args.timeout_s,
            endpoint=args.endpoint,
            run_namespace=args.run_namespace,
        )
        results = asyncio.run(
            replayer.replay(
                workflows,
                arrival_rate=args.arrival_rate,
                seed=args.seed,
            )
        )
        if args.output_path:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fh:
                for row in results:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Replayed {len(results)} calls from {len(workflows)} workflows")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
