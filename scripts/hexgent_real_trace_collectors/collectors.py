from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from .openai_compat import OpenAIChatClient
from .schema import estimate_call_input_tokens, estimate_text_tokens, make_trace_row


JsonDict = Dict[str, Any]


def parse_sharegpt_records(
    records: Iterable[JsonDict],
    *,
    max_workflows: Optional[int] = None,
    output_token_counter: Optional[Callable[[str], int]] = None,
    min_calls_per_workflow: int = 1,
) -> List[JsonDict]:
    workflows = []
    for index, record in enumerate(records):
        if max_workflows is not None and len(workflows) >= max_workflows:
            break
        conversation = record.get("conversations") or record.get("conversation") or record.get("messages") or []
        history: List[JsonDict] = []
        calls: List[JsonDict] = []
        workflow_id = str(record.get("id") or record.get("conversation_id") or f"sharegpt_{index}")
        for item in conversation:
            role = _sharegpt_role(item.get("from") or item.get("role"))
            content = item.get("value") if "value" in item else item.get("content")
            if role is None or content is None:
                continue
            if role == "assistant":
                if history:
                    expected_response = str(content)
                    expected_output_tokens = (
                        output_token_counter(expected_response)
                        if output_token_counter is not None
                        else estimate_text_tokens(expected_response) or 0
                    )
                    calls.append(
                        {
                            "call_id": f"{workflow_id}_turn_{len(calls)}",
                            "turn_index": len(calls),
                            "messages": [dict(message) for message in history],
                            "max_tokens": expected_output_tokens,
                            "min_tokens": expected_output_tokens,
                            "ignore_eos": True,
                            "metadata": {
                                "expected_response": expected_response,
                                "expected_output_tokens": expected_output_tokens,
                                "response_source": "sharegpt_dataset",
                            },
                        }
                    )
                history.append({"role": "assistant", "content": str(content)})
            else:
                history.append({"role": role, "content": str(content)})
        if len(calls) >= min_calls_per_workflow:
            workflows.append({"dataset": "sharegpt", "workflow_id": workflow_id, "calls": calls})
    return workflows


def parse_bfcl_records(records: Iterable[JsonDict], *, max_workflows: Optional[int] = None) -> List[JsonDict]:
    workflows = []
    for index, record in enumerate(records):
        if max_workflows is not None and len(workflows) >= max_workflows:
            break
        workflow_id = str(record.get("id") or record.get("question_id") or record.get("test_id") or f"bfcl_{index}")
        tools = _normalize_tools(record.get("tools") or record.get("function") or record.get("functions"))
        turns = record.get("turns") or record.get("messages")
        native_questions = record.get("question")
        calls: List[JsonDict]
        if _is_bfcl_native_question(native_questions):
            calls = [
                {
                    "call_id": f"{workflow_id}_turn_{turn_index}",
                    "turn_index": turn_index,
                    "messages": [_normalize_message(message) for message in conversation],
                    "tools": tools,
                }
                for turn_index, conversation in enumerate(native_questions)
            ]
        elif isinstance(turns, list) and turns and all(isinstance(turn, dict) and "role" in turn for turn in turns):
            calls = [
                {
                    "call_id": f"{workflow_id}_turn_0",
                    "turn_index": 0,
                    "messages": [_normalize_message(message) for message in turns],
                    "tools": tools,
                }
            ]
        elif isinstance(turns, list) and turns:
            calls = []
            history: List[JsonDict] = []
            for turn in turns:
                prompt = _record_prompt(turn)
                if prompt:
                    history.append({"role": "user", "content": prompt})
                    calls.append(
                        {
                            "call_id": f"{workflow_id}_turn_{len(calls)}",
                            "turn_index": len(calls),
                            "messages": [dict(message) for message in history],
                            "tools": tools,
                        }
                    )
        else:
            calls = [
                {
                    "call_id": f"{workflow_id}_turn_0",
                    "turn_index": 0,
                    "prompt": _record_prompt(record),
                    "tools": tools,
                }
            ]
        workflows.append({"dataset": "bfcl-v3", "workflow_id": workflow_id, "calls": calls})
    return workflows


def parse_lats_hotpotqa_records(records: Iterable[JsonDict], *, max_workflows: Optional[int] = None) -> List[JsonDict]:
    workflows = []
    for index, record in enumerate(records):
        if max_workflows is not None and len(workflows) >= max_workflows:
            break
        workflow_id = str(record.get("id") or record.get("_id") or f"hotpotqa_{index}")
        prompt = record.get("prompt") or _hotpotqa_prompt(record)
        call = {
            "call_id": f"{workflow_id}_turn_0",
            "turn_index": 0,
            "prompt": prompt,
            "metadata": {"answer": record.get("answer"), "source": record.get("source")},
        }
        workflows.append({"dataset": "lats-hotpotqa", "workflow_id": workflow_id, "calls": [call]})
    return workflows


def replay_workflows(
    workflows: Iterable[JsonDict],
    *,
    client: Optional[OpenAIChatClient],
    model: Optional[str],
    endpoint: Optional[str],
    stream: bool = False,
    max_tokens: Optional[int] = None,
) -> Iterator[JsonDict]:
    for workflow in workflows:
        for call in workflow.get("calls", []):
            if max_tokens is not None and call.get("max_tokens") is None:
                call = dict(call)
                call["max_tokens"] = max_tokens
            messages = call.get("messages")
            prompt = call.get("prompt")
            extra = {"metadata": call.get("metadata")} if call.get("metadata") is not None else None
            if call.get("tools") is not None:
                extra = dict(extra or {})
                extra["tools"] = call.get("tools")
            if client is None:
                yield make_trace_row(
                    dataset=workflow["dataset"],
                    workflow_id=workflow["workflow_id"],
                    call_id=call["call_id"],
                    turn_index=int(call["turn_index"]),
                    messages=messages,
                    prompt=prompt,
                    response_text=None,
                    input_tokens=estimate_call_input_tokens(messages, prompt),
                    output_tokens=None,
                    model=model,
                    endpoint=endpoint,
                    extra=extra,
                )
                continue
            try:
                result = client.complete(call, stream=stream)
                metadata = call.get("metadata") or {}
                expected_response = metadata.get("expected_response")
                expected_output_tokens = metadata.get("expected_output_tokens")
                yield make_trace_row(
                    dataset=workflow["dataset"],
                    workflow_id=workflow["workflow_id"],
                    call_id=call["call_id"],
                    turn_index=int(call["turn_index"]),
                    messages=messages,
                    prompt=prompt,
                    response_text=expected_response if expected_response is not None else result.get("response_text"),
                    input_tokens=result.get("input_tokens") or estimate_call_input_tokens(messages, prompt),
                    output_tokens=expected_output_tokens
                    if expected_output_tokens is not None
                    else result.get("output_tokens") or estimate_text_tokens(result.get("response_text")),
                    ttft_ms=result.get("ttft_ms"),
                    total_latency_ms=result.get("total_latency_ms"),
                    model=model,
                    endpoint=endpoint,
                    extra=extra,
                )
            except Exception as exc:
                yield make_trace_row(
                    dataset=workflow["dataset"],
                    workflow_id=workflow["workflow_id"],
                    call_id=call["call_id"],
                    turn_index=int(call["turn_index"]),
                    messages=messages,
                    prompt=prompt,
                    response_text=None,
                    input_tokens=estimate_call_input_tokens(messages, prompt),
                    output_tokens=None,
                    error=str(exc),
                    model=model,
                    endpoint=endpoint,
                    extra=extra,
                )


def load_json_records(path: Path) -> List[JsonDict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for key in ("data", "examples", "records", "questions"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return data[key]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"unsupported JSON records in {path}")


def write_jsonl(rows: Iterable[JsonDict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _sharegpt_role(role: Optional[str]) -> Optional[str]:
    if role is None:
        return None
    normalized = role.lower()
    if normalized in ("human", "user"):
        return "user"
    if normalized in ("gpt", "assistant", "bot"):
        return "assistant"
    if normalized in ("system",):
        return "system"
    return None


def _record_prompt(record: JsonDict) -> str:
    for key in ("question", "prompt", "instruction", "user_prompt"):
        if record.get(key):
            return str(record[key])
    return json.dumps(record, ensure_ascii=False)


def _normalize_message(message: JsonDict) -> JsonDict:
    return {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}


def _is_bfcl_native_question(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(conversation, list)
            and all(isinstance(message, dict) and "role" in message for message in conversation)
            for conversation in value
        )
    )


def _normalize_tools(raw_tools: Any) -> List[JsonDict]:
    if not raw_tools:
        return []
    tools = raw_tools if isinstance(raw_tools, list) else [raw_tools]
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            normalized.append(tool)
        else:
            normalized.append({"type": "function", "function": tool})
    return normalized


def _hotpotqa_prompt(record: JsonDict) -> str:
    question = str(record.get("question") or "")
    context = _format_hotpot_context(record.get("context"))
    return (
        "Answer the HotpotQA question using a reasoning and acting trace similar to LATS.\n"
        "At each step, write Thought:, then Action: if more evidence is needed, and finish with Answer:.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Thought:"
    )


def _format_hotpot_context(context: Any) -> str:
    if not context:
        return "(no local context provided; use model knowledge or connected search tool if available)"
    lines = []
    for item in context:
        if isinstance(item, list) and len(item) >= 2:
            title, sentences = item[0], item[1]
            if isinstance(sentences, list):
                lines.extend(f"{title}: {sentence}" for sentence in sentences)
            else:
                lines.append(f"{title}: {sentences}")
        else:
            lines.append(str(item))
    return "\n".join(lines)
