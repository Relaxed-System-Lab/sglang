#!/usr/bin/env python3
"""Mock PD workers for validating HexAgenT global admission dispatch."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aiohttp import web

MODEL_ID = "hexgent-mock-model"


class EventSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = asyncio.Lock()

    async def write(self, event: dict) -> None:
        async with self.lock:
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(event, sort_keys=True) + "\n")


def build_worker(
    *, stage: str, busy_snapshot_count: int, sink: EventSink
) -> web.Application:
    app = web.Application()
    state = {"snapshot_count": 0}

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def models(_: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [{"id": MODEL_ID}]}
        )

    async def server_info(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "model_id": MODEL_ID,
                "model_path": MODEL_ID,
                "served_model_name": MODEL_ID,
                "tp_size": 1,
                "dp_size": 1,
                "disaggregation_mode": stage,
                "version": "mock",
                "max_running_requests": 4,
            }
        )

    async def model_info(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "model_path": MODEL_ID,
                "tokenizer_path": MODEL_ID,
                "is_generation": True,
                "model_type": "mock",
                "architectures": ["MockForCausalLM"],
            }
        )

    async def snapshot(_: web.Request) -> web.Response:
        state["snapshot_count"] += 1
        busy = stage == "prefill" and state["snapshot_count"] <= busy_snapshot_count
        await sink.write(
            {
                "event": "snapshot",
                "stage": stage,
                "count": state["snapshot_count"],
                "busy": busy,
            }
        )
        queues = {}
        if busy:
            queues["prefill_waiting"] = [
                {
                    "rid": "existing-request",
                    "call_id": "existing-call",
                    "workflow_id": "existing-workflow",
                    "queue_name": "prefill_waiting",
                    "input_tokens": 128,
                    "is_running": False,
                    "prefill_started": False,
                }
            ]
        return web.json_response(
            {"mode": stage, "queues": queues, "running": []}
        )

    async def completion(request: web.Request) -> web.Response:
        payload = await request.json()
        await sink.write(
            {
                "event": "dispatch",
                "stage": stage,
                "bootstrap_room": payload.get("bootstrap_room"),
                "workflow_id": payload.get("workflow_id"),
                "schedule_revision": payload.get("hexgent_schedule_revision"),
            }
        )
        if stage == "prefill":
            return web.json_response({"meta_info": {}})
        return web.json_response(
            {
                "id": "mock-completion",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 1,
                    "total_tokens": 9,
                },
            }
        )

    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.router.add_get("/health_generate", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/server_info", server_info)
    app.router.add_get("/get_server_info", server_info)
    app.router.add_get("/model_info", model_info)
    app.router.add_get("/get_model_info", model_info)
    app.router.add_get("/hexgent/pd_snapshot", snapshot)
    app.router.add_post("/v1/chat/completions", completion)
    return app


async def main_async(args: argparse.Namespace) -> None:
    event_path = Path(args.event_log)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.write_text("", encoding="utf-8")
    sink = EventSink(event_path)
    runners = []
    for stage, port in (("prefill", args.prefill_port), ("decode", args.decode_port)):
        runner = web.AppRunner(
            build_worker(
                stage=stage,
                busy_snapshot_count=args.busy_snapshot_count,
                sink=sink,
            )
        )
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        runners.append(runner)
    await asyncio.Event().wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-port", type=int, default=31000)
    parser.add_argument("--decode-port", type=int, default=32000)
    parser.add_argument("--busy-snapshot-count", type=int, default=3)
    parser.add_argument("--event-log", required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
