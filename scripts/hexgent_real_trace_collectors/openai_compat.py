from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, Union


JsonDict = Dict[str, Any]
PostJson = Callable[[str, JsonDict, Dict[str, str]], JsonDict]
StreamEvent = Union[JsonDict, Tuple[float, JsonDict]]
StreamJson = Callable[[str, JsonDict, Dict[str, str]], Iterable[StreamEvent]]


class OpenAIChatClient:
    """Small OpenAI-compatible /v1/chat/completions client."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
        post_json: Optional[PostJson] = None,
        stream_json: Optional[StreamJson] = None,
        clock_ms: Optional[Callable[[], float]] = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._post_json = post_json or self._urllib_post_json
        self._stream_json = stream_json or self._urllib_stream_json
        self._clock_ms = clock_ms or (lambda: time.perf_counter() * 1000.0)

    def complete(self, call: JsonDict, *, stream: bool = False) -> JsonDict:
        payload = self._payload_for_call(call, stream=stream)
        headers = self._headers()
        if stream:
            return self._complete_stream(payload, headers)
        start_ms = self._clock_ms()
        response = self._post_json("/chat/completions", payload, headers)
        end_ms = self._clock_ms()
        usage = response.get("usage") or {}
        return {
            "response_text": _extract_response_text(response),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "ttft_ms": None,
            "total_latency_ms": end_ms - start_ms,
            "raw_response": response,
        }

    def _complete_stream(self, payload: JsonDict, headers: Dict[str, str]) -> JsonDict:
        start_ms = self._clock_ms()
        last_ms = start_ms
        first_token_ms: Optional[float] = None
        chunks = []
        usage: JsonDict = {}
        for event in self._stream_json("/chat/completions", payload, headers):
            event_ms, event_payload = event if isinstance(event, tuple) else (self._clock_ms(), event)
            last_ms = event_ms
            content = _extract_delta_text(event_payload)
            if content:
                if first_token_ms is None:
                    first_token_ms = event_ms
                chunks.append(content)
            if event_payload.get("usage"):
                usage = event_payload["usage"]
        return {
            "response_text": "".join(chunks),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "ttft_ms": None if first_token_ms is None else first_token_ms - start_ms,
            "total_latency_ms": last_ms - start_ms,
            "raw_response": {"usage": usage},
        }

    def _payload_for_call(self, call: JsonDict, *, stream: bool) -> JsonDict:
        messages = call.get("messages")
        if not messages:
            messages = [{"role": "user", "content": call.get("prompt", "")}]
        payload: JsonDict = {
            "model": self.model or call.get("model") or "default",
            "messages": messages,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        for key in (
            "tools",
            "tool_choice",
            "temperature",
            "max_tokens",
            "min_tokens",
            "ignore_eos",
            "top_p",
        ):
            if key in call and call[key] is not None:
                payload[key] = call[key]
        return payload

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _urllib_post_json(self, path: str, payload: JsonDict, headers: Dict[str, str]) -> JsonDict:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.endpoint + path, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from endpoint: {body}") from exc

    def _urllib_stream_json(self, path: str, payload: JsonDict, headers: Dict[str, str]) -> Iterator[Tuple[float, JsonDict]]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.endpoint + path, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_text = line[len("data:") :].strip()
                    if data_text == "[DONE]":
                        break
                    yield self._clock_ms(), json.loads(data_text)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from endpoint: {body}") from exc


def _extract_response_text(response: JsonDict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") or {}
    message_text = _join_text_fields(message, ("reasoning_content", "reasoning", "content"))
    if message_text:
        return message_text + _extract_tool_call_text(message)
    tool_call_text = _extract_tool_call_text(message)
    if tool_call_text:
        return tool_call_text
    if first.get("text") is not None:
        return _normalize_text_value(first.get("text"))
    return ""


def _extract_delta_text(event: JsonDict) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    delta = first.get("delta") or {}
    delta_text = _join_text_fields(delta, ("reasoning_content", "reasoning", "content"))
    if delta_text:
        return delta_text + _extract_tool_call_text(delta)
    tool_call_text = _extract_tool_call_text(delta)
    if tool_call_text:
        return tool_call_text
    if first.get("text") is not None:
        return _normalize_text_value(first.get("text"))
    return ""


def _join_text_fields(source: JsonDict, field_names: Tuple[str, ...]) -> str:
    parts = []
    for field_name in field_names:
        value = source.get(field_name)
        text = _normalize_text_value(value)
        if text:
            parts.append(text)
    return "".join(parts)


def _extract_tool_call_text(source: JsonDict) -> str:
    parts = []
    function_call = source.get("function_call")
    if isinstance(function_call, dict):
        parts.append(_normalize_text_value(function_call.get("name")))
        parts.append(_normalize_text_value(function_call.get("arguments")))
    tool_calls = source.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                parts.append(_normalize_text_value(tool_call))
                continue
            function = tool_call.get("function")
            if isinstance(function, dict):
                parts.append(_normalize_text_value(function.get("name")))
                parts.append(_normalize_text_value(function.get("arguments")))
            else:
                parts.append(_normalize_text_value(tool_call))
    return "".join(parts)


def _normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(_normalize_text_value(item.get("text") or item.get("content") or item.get("value")))
            else:
                parts.append(_normalize_text_value(item))
        return "".join(parts)
    if isinstance(value, dict):
        return _normalize_text_value(value.get("text") or value.get("content") or value.get("value"))
    return str(value)
