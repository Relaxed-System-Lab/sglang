from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .schema import estimate_call_input_tokens, estimate_text_tokens, make_trace_row


JsonDict = Dict[str, Any]


class RecordingTraceSink:
    """Append OpenAI-compatible completion calls to HexGen trace JSONL."""

    def __init__(self, *, output_path: Path, dataset: str, endpoint: str) -> None:
        self.output_path = output_path
        self.dataset = dataset
        self.endpoint = endpoint
        self._lock = threading.Lock()
        self._counter = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def record_chat_completion(
        self,
        *,
        request_payload: JsonDict,
        response_payload: Optional[JsonDict],
        start_ms: float,
        first_token_ms: Optional[float],
        end_ms: float,
        error: Optional[str] = None,
    ) -> None:
        metadata = _request_metadata(request_payload)
        with self._lock:
            index = self._counter
            self._counter += 1
        workflow_id = str(metadata.get("workflow_id") or f"{self.dataset}_{index}")
        call_id = str(metadata.get("call_id") or f"{workflow_id}_call_0")
        messages = request_payload.get("messages")
        prompt = request_payload.get("prompt")
        usage = (response_payload or {}).get("usage") or {}
        rows = _choice_split_rows(
            dataset=self.dataset,
            workflow_id=workflow_id,
            base_call_id=call_id,
            metadata=metadata,
            messages=messages,
            prompt=prompt,
            response_payload=response_payload or {},
            usage=usage,
            first_token_ms=first_token_ms,
            start_ms=start_ms,
            end_ms=end_ms,
            error=error,
            model=request_payload.get("model"),
            endpoint=self.endpoint,
        )
        if rows is not None:
            with self._lock, self.output_path.open("a", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
            return

        response_text = _extract_response_text(response_payload or {})
        row = make_trace_row(
            dataset=self.dataset,
            workflow_id=workflow_id,
            call_id=call_id,
            turn_index=int(metadata.get("turn_index") or 0),
            messages=messages,
            prompt=prompt,
            response_text=response_text,
            input_tokens=usage.get("prompt_tokens")
            or metadata.get("input_tokens")
            or estimate_call_input_tokens(messages, prompt),
            output_tokens=usage.get("completion_tokens")
            or metadata.get("output_tokens")
            or estimate_text_tokens(response_text),
            ttft_ms=None if first_token_ms is None else first_token_ms - start_ms,
            total_latency_ms=end_ms - start_ms,
            error=error,
            model=request_payload.get("model"),
            endpoint=self.endpoint,
            extra={"metadata": metadata} if metadata else None,
        )
        with self._lock, self.output_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record_completion(
        self,
        *,
        request_payload: JsonDict,
        response_payload: Optional[JsonDict],
        start_ms: float,
        first_token_ms: Optional[float],
        end_ms: float,
        error: Optional[str] = None,
    ) -> None:
        self.record_chat_completion(
            request_payload=request_payload,
            response_payload=response_payload,
            start_ms=start_ms,
            first_token_ms=first_token_ms,
            end_ms=end_ms,
            error=error,
        )


def run_recording_proxy(
    *,
    listen_host: str,
    listen_port: int,
    target_endpoint: str,
    output_path: Path,
    dataset: str,
    model_override: Optional[str] = None,
) -> None:
    sink = RecordingTraceSink(
        output_path=output_path,
        dataset=dataset,
        endpoint=target_endpoint.rstrip("/"),
    )
    handler_cls = _make_proxy_handler(target_endpoint.rstrip("/"), sink, model_override=model_override)
    server = ThreadingHTTPServer((listen_host, listen_port), handler_cls)
    print(
        f"recording proxy listening on http://{listen_host}:{listen_port}; "
        f"forwarding to {target_endpoint.rstrip('/')} and writing {output_path}",
        flush=True,
    )
    server.serve_forever()


def _make_proxy_handler(target_endpoint: str, sink: RecordingTraceSink, *, model_override: Optional[str] = None):
    class RecordingProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if self.path.rstrip("/") in {"/health", "/v1/health"}:
                self._send_json(200, {"status": "ok"})
                return
            self._forward_raw(method="GET", payload=None)

        def do_POST(self) -> None:
            payload = self._read_json()
            if self.path.endswith("/chat/completions"):
                self._forward_chat_completion(payload)
            elif self.path.endswith("/completions"):
                self._forward_completion(payload)
            else:
                self._forward_raw(method="POST", payload=payload)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _forward_chat_completion(self, payload: JsonDict) -> None:
            forwarded_payload = _forward_payload(payload, model_override=model_override)
            start_ms = _clock_ms()
            first_token_ms: Optional[float] = None
            response_payload: Optional[JsonDict] = None
            error: Optional[str] = None
            try:
                if payload.get("stream"):
                    first_token_ms, response_payload = self._forward_stream(
                        forwarded_payload,
                        target_path="/chat/completions",
                        relay_to_client=True,
                    )
                else:
                    first_token_ms, response_payload = self._collect_stream_as_chat_completion(forwarded_payload)
                    self._send_json(200, response_payload)
            except Exception as exc:
                error = str(exc)
                self._send_json(502, {"error": error})
            finally:
                sink.record_chat_completion(
                    request_payload=payload,
                    response_payload=response_payload,
                    start_ms=start_ms,
                    first_token_ms=first_token_ms,
                    end_ms=_clock_ms(),
                    error=error,
                )

        def _forward_completion(self, payload: JsonDict) -> None:
            forwarded_payload = _forward_payload(payload, model_override=model_override)
            start_ms = _clock_ms()
            first_token_ms: Optional[float] = None
            response_payload: Optional[JsonDict] = None
            error: Optional[str] = None
            try:
                if payload.get("stream"):
                    first_token_ms, response_payload = self._forward_stream(
                        forwarded_payload,
                        target_path="/completions",
                        relay_to_client=True,
                    )
                else:
                    first_token_ms, response_payload = self._collect_stream_as_completion(forwarded_payload)
                    self._send_json(200, response_payload)
            except Exception as exc:
                error = str(exc)
                self._send_json(502, {"error": error})
            finally:
                sink.record_completion(
                    request_payload=payload,
                    response_payload=response_payload,
                    start_ms=start_ms,
                    first_token_ms=first_token_ms,
                    end_ms=_clock_ms(),
                    error=error,
                )

        def _post_json(self, payload: JsonDict, *, target_path: str) -> JsonDict:
            request = urllib.request.Request(
                target_endpoint + target_path,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._forward_headers("application/json"),
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))

        def _forward_stream(
            self,
            payload: JsonDict,
            *,
            target_path: str,
            relay_to_client: bool,
        ) -> Tuple[Optional[float], JsonDict]:
            request = urllib.request.Request(
                target_endpoint + target_path,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._forward_headers("application/json"),
                method="POST",
            )
            if relay_to_client:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
            chunks_by_index: Dict[int, list[str]] = {}
            usage: JsonDict = {}
            first_token_ms: Optional[float] = None
            finish_reason_by_index: Dict[int, Optional[str]] = {}
            with urllib.request.urlopen(request) as response:
                for raw_line in response:
                    if relay_to_client:
                        self.wfile.write(raw_line)
                        self.wfile.flush()
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_text = line[len("data:") :].strip()
                    if data_text == "[DONE]":
                        continue
                    event = json.loads(data_text)
                    choices = event.get("choices") or []
                    for choice in choices:
                        index = int(choice.get("index") or 0)
                        content = _extract_choice_delta_text(choice)
                        if content:
                            if first_token_ms is None:
                                first_token_ms = _clock_ms()
                            chunks_by_index.setdefault(index, []).append(content)
                        if choice.get("finish_reason"):
                            finish_reason_by_index[index] = str(choice["finish_reason"])
                    if event.get("usage"):
                        usage = event["usage"]
            if not chunks_by_index:
                chunks_by_index[0] = []
            if target_path.endswith("/chat/completions"):
                response_payload = {
                    "choices": [
                        {
                            "index": index,
                            "message": {
                                "role": "assistant",
                                "content": "".join(chunks_by_index[index]),
                            },
                            "finish_reason": finish_reason_by_index.get(index),
                        }
                        for index in sorted(chunks_by_index)
                    ],
                    "usage": usage,
                }
            else:
                response_payload = {
                    "id": f"hexgent-proxy-{time.time_ns()}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": payload.get("model"),
                    "choices": [
                        {
                            "text": "".join(chunks_by_index[index]),
                            "index": index,
                            "logprobs": None,
                            "finish_reason": finish_reason_by_index.get(index),
                        }
                        for index in sorted(chunks_by_index)
                    ],
                    "usage": usage,
                }
            return first_token_ms, response_payload

        def _collect_stream_as_completion(self, payload: JsonDict) -> Tuple[Optional[float], JsonDict]:
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            stream_payload.setdefault("stream_options", {"include_usage": True})
            return self._forward_stream(
                stream_payload,
                target_path="/completions",
                relay_to_client=False,
            )

        def _collect_stream_as_chat_completion(self, payload: JsonDict) -> Tuple[Optional[float], JsonDict]:
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            stream_payload.setdefault("stream_options", {"include_usage": True})
            first_token_ms, response_payload = self._forward_stream(
                stream_payload,
                target_path="/chat/completions",
                relay_to_client=False,
            )
            response_payload.setdefault("id", f"hexgent-proxy-{time.time_ns()}")
            response_payload.setdefault("object", "chat.completion")
            response_payload.setdefault("created", int(time.time()))
            response_payload.setdefault("model", payload.get("model"))
            return first_token_ms, response_payload

        def _forward_raw(self, *, method: str, payload: Optional[JsonDict]) -> None:
            body = None if payload is None else json.dumps(payload).encode("utf-8")
            path = self.path
            if path.startswith("/v1/"):
                path = path[len("/v1") :]
            request = urllib.request.Request(
                target_endpoint + path,
                data=body,
                headers=self._forward_headers("application/json"),
                method=method,
            )
            try:
                with urllib.request.urlopen(request) as response:
                    data = response.read()
                    self.send_response(response.status)
                    for key, value in response.headers.items():
                        if key.lower() not in {"connection", "transfer-encoding"}:
                            self.send_header(key, value)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            except urllib.error.HTTPError as exc:
                data = exc.read()
                self.send_response(exc.code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def _read_json(self) -> JsonDict:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, status_code: int, payload: JsonDict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _forward_headers(self, content_type: str) -> Dict[str, str]:
            headers = {"Content-Type": content_type}
            if self.headers.get("Authorization"):
                headers["Authorization"] = self.headers["Authorization"]
            return headers

    return RecordingProxyHandler


def _clock_ms() -> float:
    return time.perf_counter() * 1000.0


def _request_metadata(payload: JsonDict) -> JsonDict:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    hexgent = metadata.get("hexgent")
    if isinstance(hexgent, dict):
        return dict(hexgent)
    return dict(metadata)


def _choice_split_rows(
    *,
    dataset: str,
    workflow_id: str,
    base_call_id: str,
    metadata: JsonDict,
    messages: Any,
    prompt: Any,
    response_payload: JsonDict,
    usage: JsonDict,
    first_token_ms: Optional[float],
    start_ms: float,
    end_ms: float,
    error: Optional[str],
    model: Optional[str],
    endpoint: str,
) -> Optional[list[JsonDict]]:
    if not metadata.get("split_batched_choices"):
        return None
    choices = response_payload.get("choices") or []
    if len(choices) <= 1:
        return None

    base_turn_index = int(metadata.get("turn_index") or 0)
    ttft_ms = None if first_token_ms is None else first_token_ms - start_ms
    request_total_ms = end_ms - start_ms
    decode_total_ms = max(0.0, request_total_ms - (ttft_ms or 0.0))
    response_texts = [
        _extract_response_text({"choices": [choice], "usage": usage})
        for choice in choices
    ]
    output_tokens = [estimate_text_tokens(text) or 0 for text in response_texts]
    max_output_tokens = max(output_tokens) if output_tokens else 0

    rows = []
    for offset, choice in enumerate(choices):
        choice_metadata = dict(metadata)
        choice_metadata["call_id"] = f"{workflow_id}-call-{base_turn_index + offset}"
        choice_metadata["turn_index"] = base_turn_index + offset
        choice_metadata["fanout_index"] = offset
        choice_metadata.setdefault("fanout_width", len(choices))
        choice_metadata["batched_choice_index"] = int(choice.get("index") or offset)
        choice_metadata["batched_choice_count"] = len(choices)
        choice_metadata.pop("logical_call_count", None)
        choice_metadata.pop("split_batched_choices", None)

        if ttft_ms is None or max_output_tokens <= 0:
            total_latency_ms = request_total_ms
        else:
            total_latency_ms = ttft_ms + decode_total_ms * (output_tokens[offset] / max_output_tokens)

        rows.append(
            make_trace_row(
                dataset=dataset,
                workflow_id=workflow_id,
                call_id=str(choice_metadata.get("call_id") or base_call_id),
                turn_index=int(choice_metadata.get("turn_index") or 0),
                messages=messages,
                prompt=prompt,
                response_text=response_texts[offset],
                input_tokens=usage.get("prompt_tokens") or metadata.get("input_tokens") or estimate_call_input_tokens(messages, prompt),
                output_tokens=output_tokens[offset],
                ttft_ms=ttft_ms,
                total_latency_ms=total_latency_ms,
                error=error,
                model=model,
                endpoint=endpoint,
                extra={"metadata": choice_metadata},
            )
        )
    return rows


def _forward_payload(payload: JsonDict, *, model_override: Optional[str]) -> JsonDict:
    if not model_override:
        return payload
    forwarded_payload = dict(payload)
    forwarded_payload["model"] = model_override
    return forwarded_payload


def _extract_response_text(response: JsonDict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    message_text = _join_text_fields(message, ("reasoning_content", "reasoning", "content"))
    if message_text:
        return message_text + _extract_tool_call_text(message)
    tool_call_text = _extract_tool_call_text(message)
    if tool_call_text:
        return tool_call_text
    if choices[0].get("text") is not None:
        return _normalize_text_value(choices[0]["text"])
    return ""


def _extract_delta_text(event: JsonDict) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    return _extract_choice_delta_text(choices[0])


def _extract_choice_delta_text(choice: JsonDict) -> str:
    delta = choice.get("delta") or {}
    delta_text = _join_text_fields(delta, ("reasoning_content", "reasoning", "content"))
    if delta_text:
        return delta_text + _extract_tool_call_text(delta)
    tool_call_text = _extract_tool_call_text(delta)
    if tool_call_text:
        return tool_call_text
    if choice.get("text") is not None:
        return _normalize_text_value(choice["text"])
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
