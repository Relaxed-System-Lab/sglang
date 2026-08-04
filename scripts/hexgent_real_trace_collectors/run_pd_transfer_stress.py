#!/usr/bin/env python3
"""Drive a reproducible concurrent fixed-length P-D transfer calibration.

This deliberately uses the native PD router by default.  It is not a
workflow/SLO benchmark: it creates a burst of long requests so the worker
telemetry can measure how ``max_inflight_transfers`` changes real transfer,
queue, and CUDA batch behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiohttp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", default="http://127.0.0.1:33001")
    parser.add_argument("--model", default="llama8b-pd-2p2d")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--prompt-words", type=int, default=1800)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_payload(*, model: str, prompt: str, output_tokens: int) -> dict[str, Any]:
    if output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "stream": False,
    }


async def submit_one(
    session: aiohttp.ClientSession,
    *,
    index: int,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    try:
        async with session.post(endpoint, json=payload) as response:
            body = await response.text()
            completed = time.time()
            result: dict[str, Any] = {
                "request_index": index,
                "submitted_at_s": started,
                "completed_at_s": completed,
                "wall_latency_ms": (completed - started) * 1000.0,
                "status": response.status,
            }
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {"body": body[:1000]}
            result["response"] = parsed
            if response.status != 200:
                result["error"] = f"HTTP {response.status}"
            return result
    except Exception as exc:
        completed = time.time()
        return {
            "request_index": index,
            "submitted_at_s": started,
            "completed_at_s": completed,
            "wall_latency_ms": (completed - started) * 1000.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    prompt = " ".join(["calibration"] * args.prompt_words)
    payload = build_payload(
        model=args.model, prompt=prompt, output_tokens=args.output_tokens
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    endpoint = f"{args.router_url.rstrip('/')}/v1/chat/completions"
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await asyncio.gather(
            *(
                submit_one(
                    session,
                    index=index,
                    endpoint=endpoint,
                    payload=payload,
                )
                for index in range(args.concurrency)
            )
        )


def main() -> None:
    args = parse_args()
    rows = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    failures = sum("error" in row for row in rows)
    print(json.dumps({"requests": len(rows), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
