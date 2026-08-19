"""Issue serial P-D requests while Mooncake emits native-sync transfer telemetry.

The SGLang prefill worker logs ``HEXGENT_KV_TRANSFER`` for each request.  This
client records the matching request order and serving E2E latency; it does not
use client wall time as the KV-transfer measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:31081")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-words", type=int, nargs="+", default=[64, 256, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Require at least this many output tokens before accepting EOS.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def build_request_payload(
    *, model: str, content: str, max_tokens: int, min_tokens: int
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if min_tokens:
        payload["min_tokens"] = min_tokens
    return payload


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output_file:
        sequence = 0
        for prompt_words in args.prompt_words:
            for repeat in range(args.repeats):
                # The unique suffix prevents prefix-cache reuse across samples.
                content = " ".join(["transfer"] * prompt_words)
                content += f" unique-request-{sequence}-{time.time_ns()}"
                payload = build_request_payload(
                    model=args.model,
                    content=content,
                    max_tokens=args.max_tokens,
                    min_tokens=args.min_tokens,
                )
                started_ns = time.perf_counter_ns()
                response = requests.post(
                    f"{args.endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=args.timeout_seconds,
                )
                completed_ns = time.perf_counter_ns()
                row = {
                    "sequence": sequence,
                    "requested_prompt_words": prompt_words,
                    "repeat": repeat,
                    "client_e2e_ms": (completed_ns - started_ns) / 1e6,
                    "http_status": response.status_code,
                }
                try:
                    body = response.json()
                    row["usage"] = body.get("usage")
                    row["error"] = body.get("error")
                except ValueError:
                    row["response_text"] = response.text[:1000]
                output_file.write(json.dumps(row) + "\n")
                output_file.flush()
                response.raise_for_status()
                sequence += 1


if __name__ == "__main__":
    main()
