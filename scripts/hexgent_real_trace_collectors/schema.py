from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


REQUIRED_FIELDS = (
    "dataset",
    "workflow_id",
    "call_id",
    "turn_index",
    "messages",
    "prompt",
    "response_text",
    "input_tokens",
    "output_tokens",
    "ttft_ms",
    "total_latency_ms",
    "error",
    "model",
    "endpoint",
    "created_at",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_text_tokens(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    # Portable fallback for request-only rows. Endpoint usage wins when present.
    return max(1, (len(text) + 3) // 4) if text else 0


def estimate_call_input_tokens(messages: Optional[Iterable[Dict[str, Any]]], prompt: Optional[str]) -> Optional[int]:
    if messages:
        return estimate_text_tokens("\n".join(str(message.get("content", "")) for message in messages))
    return estimate_text_tokens(prompt)


def make_trace_row(
    *,
    dataset: str,
    workflow_id: str,
    call_id: str,
    turn_index: int,
    messages: Optional[Iterable[Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    response_text: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    ttft_ms: Optional[float] = None,
    total_latency_ms: Optional[float] = None,
    error: Optional[str] = None,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    created_at: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": dataset,
        "workflow_id": workflow_id,
        "call_id": call_id,
        "turn_index": turn_index,
        "messages": list(messages) if messages is not None else None,
        "prompt": prompt,
        "response_text": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "error": error,
        "model": model,
        "endpoint": endpoint,
        "created_at": created_at or utc_now_iso(),
    }
    if extra:
        row.update(extra)
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"trace row missing required fields: {missing}")
    return row
