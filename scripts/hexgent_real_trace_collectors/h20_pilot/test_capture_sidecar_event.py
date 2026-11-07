"""Synthetic CPU tests for the preflight-only ASGI event capture wrapper."""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).with_name("capture_sidecar_event.py")
    spec = importlib.util.spec_from_file_location("r2_capture_sidecar_event", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exercise(wrapper, messages, path="/event"):
    seen, sent = [], []
    pending = iter(messages)
    async def receive():
        return next(pending)
    async def send(message):
        sent.append(message)
    async def downstream(scope, receive, send):
        while True:
            message = await receive()
            seen.append(message)
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b"unchanged"})
    wrapper.app = downstream
    asyncio.run(wrapper({"type": "http", "method": "POST", "path": path}, receive, send))
    assert all(actual is original for actual, original in zip(seen, messages))
    assert sent == [
        {"type": "http.response.start", "status": 201, "headers": []},
        {"type": "http.response.body", "body": b"unchanged"},
    ]


def test_split_body_is_captured_exactly_and_messages_forwarded_unchanged(tmp_path):
    module = _load()
    path = tmp_path / "events.jsonl"
    capture = module.EventBodyCapture(None, path)
    messages = [
        {"type": "http.request", "body": b'{ "event_type":', "more_body": True},
        {"type": "http.request", "body": b' "synthetic" }\n', "more_body": False},
    ]
    _exercise(capture, messages)
    capture.close()
    record = json.loads(path.read_text())
    assert record["event_body_utf8"] == '{ "event_type": "synthetic" }\n'


def test_other_paths_are_forwarded_without_capture(tmp_path):
    module = _load()
    path = tmp_path / "events.jsonl"
    capture = module.EventBodyCapture(None, path)
    _exercise(capture, [{"type": "http.request", "body": b"{}", "more_body": False}], "/plan")
    capture.close()
    assert path.read_bytes() == b""


def test_capture_budget_does_not_truncate_or_change_downstream_request(tmp_path):
    module = _load()
    path = tmp_path / "events.jsonl"
    capture = module.EventBodyCapture(None, path, max_bytes=100)
    _exercise(capture, [{"type": "http.request", "body": b"{}", "more_body": False}])
    _exercise(capture, [{"type": "http.request", "body": b"x" * 150, "more_body": False}])
    _exercise(capture, [{"type": "http.request", "body": b"{}", "more_body": False}])
    capture.close()
    assert len(path.read_bytes()) <= 100
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event_body_utf8"] for record in records] == ["{}"]


def test_fastapi_receives_identical_event_body_and_response_is_preserved(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from starlette.requests import Request
    from starlette.responses import Response
    module = _load()
    app = fastapi.FastAPI()
    received = []
    async def event(request: Request):
        body = await request.body()
        received.append(body)
        return Response(body, status_code=201, headers={"x-synthetic": "unchanged"})
    app.add_api_route("/event", event, methods=["POST"])
    path = tmp_path / "events.jsonl"
    capture = module.EventBodyCapture(app, path)
    raw = b'{ "event_type" : "synthetic", "payload": { "x": 1 } }\n'
    with TestClient(capture) as client:
        response = client.post("/event", content=raw, headers={"content-type": "application/json"})
    capture.close()
    assert response.status_code == 201
    assert response.headers["x-synthetic"] == "unchanged"
    assert response.content == raw
    assert received == [raw]
    assert json.loads(path.read_text())["event_body_utf8"].encode() == raw


def test_fastapi_non_event_route_is_not_recorded(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    module = _load()
    app = fastapi.FastAPI()
    async def health():
        return {"status": "synthetic"}
    app.add_api_route("/health", health)
    path = tmp_path / "events.jsonl"
    capture = module.EventBodyCapture(app, path)
    with TestClient(capture) as client:
        assert client.get("/health").json() == {"status": "synthetic"}
    capture.close()
    assert path.read_bytes() == b""
