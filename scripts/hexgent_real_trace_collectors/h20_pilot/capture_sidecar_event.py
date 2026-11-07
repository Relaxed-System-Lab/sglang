#!/usr/bin/env python3
"""Preflight-only /event body capture around the real HexAgenT sidecar.

This is not the formal-run sidecar entrypoint. ASGI receive messages are passed
through unchanged; only complete POST /event bodies are copied into a fresh,
bounded JSONL evidence file. Normal PlannerTelemetry is independently enabled.
"""

import argparse
import json
import logging
from pathlib import Path
import threading
import time


_CAPTURE_LIMIT = 1024 * 1024
_LOG = logging.getLogger("r2_preflight_event_capture")


class EventBodyCapture:
    def __init__(self, app, path, *, max_bytes=_CAPTURE_LIMIT):
        if not 0 < max_bytes <= _CAPTURE_LIMIT:
            raise ValueError("capture limit must be within 1MiB")
        self.app = app
        self.max_bytes = max_bytes
        self._file = Path(path).open("xb")
        self._lock = threading.Lock()
        self._warned = False

    def _warning(self):
        if not self._warned:
            _LOG.warning("Preflight event capture skipped a record: 1MiB/file budget or incomplete body")
            self._warned = True

    def _record(self, body):
        # Rust sends UTF-8 JSON. Keeping its text retains its exact original
        # whitespace and numbers; downstream receives the original bytes.
        line = (json.dumps({
            "captured_unix_ns": time.time_ns(),
            "event_body_utf8": body.decode("utf-8"),
        }, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            if self._file.tell() + len(line) > self.max_bytes:
                self._warning()
                return
            self._file.write(line)
            self._file.flush()

    async def __call__(self, scope, receive, send):
        if not (scope["type"] == "http" and scope["method"] == "POST"
                and scope["path"] == "/event"):
            return await self.app(scope, receive, send)
        body = bytearray()
        complete = False
        overflow = False

        async def capture_receive():
            nonlocal complete, overflow
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                if not overflow:
                    if len(body) + len(chunk) <= self.max_bytes:
                        body.extend(chunk)
                    else:
                        overflow = True
                        body.clear()
                complete = not message.get("more_body", False)
            return message

        try:
            await self.app(scope, capture_receive, send)
        finally:
            # Capture must not replace the real app response on an I/O error.
            try:
                if complete and not overflow:
                    self._record(bytes(body))
                else:
                    self._warning()
            except (OSError, UnicodeDecodeError):
                _LOG.exception("Preflight event evidence could not be recorded")

    def close(self):
        self._file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37140)
    parser.add_argument("--capture-path", required=True)
    parser.add_argument("--telemetry-path", required=True)
    args = parser.parse_args()
    import uvicorn
    from sglang.hexgent.sidecar import PlannerTelemetry, build_app

    app = EventBodyCapture(
        build_app(telemetry=PlannerTelemetry(args.telemetry_path)), args.capture_path
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        app.close()


if __name__ == "__main__":
    main()
