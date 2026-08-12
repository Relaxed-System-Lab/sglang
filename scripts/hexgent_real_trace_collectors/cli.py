from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .collectors import (
    load_json_records,
    parse_bfcl_records,
    parse_lats_hotpotqa_records,
    parse_sharegpt_records,
    replay_workflows,
    write_jsonl,
)
from .openai_compat import OpenAIChatClient
from .proxy import run_recording_proxy


SHAREGPT_DATASET = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_DEFAULT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect and replay real trace responses for HexGen/SGLang.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    _add_common_args(_add_parser(subparsers, "sharegpt", "Collect ShareGPT workflows from HF or local JSON/JSONL."))
    _add_common_args(_add_parser(subparsers, "bfcl", "Collect BFCL-v3 prompts/questions from local JSON/JSONL."))
    _add_common_args(_add_parser(subparsers, "lats-hotpotqa", "Collect LATS/HotpotQA workflows from local JSON/JSONL."))
    _add_proxy_args(_add_parser(subparsers, "proxy", "Record any OpenAI-compatible evaluation while forwarding to SGLang."))
    args = parser.parse_args(argv)

    if args.dataset == "proxy":
        run_recording_proxy(
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            target_endpoint=args.target_endpoint,
            output_path=Path(args.output),
            dataset=args.trace_dataset,
            model_override=args.model_override or os.environ.get("HEXGENT_PROXY_MODEL_OVERRIDE"),
        )
        return 0

    workflows = _load_workflows(args)
    client = None
    if args.endpoint:
        client = OpenAIChatClient(
            endpoint=args.endpoint,
            model=args.model,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            timeout_s=args.timeout_s,
        )
    rows = replay_workflows(
        workflows,
        client=client,
        model=args.model,
        endpoint=args.endpoint,
        stream=args.stream,
        max_tokens=args.max_tokens,
    )
    count = write_jsonl(rows, Path(args.output))
    print(f"wrote {count} rows to {args.output}")
    return 0


def _add_parser(subparsers: argparse._SubParsersAction, name: str, help_text: str) -> argparse.ArgumentParser:
    return subparsers.add_parser(name, help=help_text)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", help="Local JSON/JSONL file. Required for BFCL and LATS fallback collection.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--max-workflows", type=int, default=None, help="Limit workflows for smoke tests.")
    parser.add_argument(
        "--min-calls-per-workflow",
        type=int,
        default=1,
        help="Minimum logical LLM calls per workflow. Useful for filtering ShareGPT to multi-turn sessions.",
    )
    parser.add_argument("--endpoint", help="OpenAI-compatible endpoint base URL, for example http://127.0.0.1:30000/v1.")
    parser.add_argument("--model", help="Model name to send in requests.")
    parser.add_argument("--api-key", help="API key. Defaults to OPENAI_API_KEY if unset.")
    parser.add_argument("--stream", action="store_true", help="Use streaming chat completions and record ttft_ms.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional max_tokens value for every replayed call.")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="HTTP timeout per request.")
    parser.add_argument(
        "--tokenizer",
        help=(
            "Optional Hugging Face tokenizer path/name for exact ShareGPT assistant "
            "response token lengths. Defaults to --model for ShareGPT when available."
        ),
    )


def _add_proxy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--listen-host", default="127.0.0.1", help="Proxy listen host.")
    parser.add_argument("--listen-port", type=int, default=18080, help="Proxy listen port.")
    parser.add_argument(
        "--target-endpoint",
        required=True,
        help="Target OpenAI-compatible endpoint base URL, for example http://127.0.0.1:30000/v1.",
    )
    parser.add_argument("--trace-dataset", required=True, help="Dataset label to write in trace rows, e.g. bfcl-v3.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--model-override",
        help="Optional model name to send to the target server while preserving the original request in trace rows.",
    )


def _load_workflows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dataset == "sharegpt":
        records = _load_sharegpt_records(args.input)
        token_counter = _load_output_token_counter(args.tokenizer or args.model)
        return parse_sharegpt_records(
            records,
            max_workflows=args.max_workflows,
            output_token_counter=token_counter,
            min_calls_per_workflow=args.min_calls_per_workflow,
        )
    if not args.input:
        raise SystemExit(f"{args.dataset} requires --input pointing at a local JSON/JSONL file")
    records = load_json_records(Path(args.input))
    if args.dataset == "bfcl":
        return parse_bfcl_records(records, max_workflows=args.max_workflows)
    if args.dataset == "lats-hotpotqa":
        return parse_lats_hotpotqa_records(records, max_workflows=args.max_workflows)
    raise SystemExit(f"unknown dataset: {args.dataset}")


def _load_output_token_counter(tokenizer_name: str | None):
    if not tokenizer_name:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("--tokenizer/--model token counting requires: pip install transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return lambda text: len(tokenizer.encode(str(text), add_special_tokens=False))


def _load_sharegpt_records(input_path: str | None) -> Iterable[Dict[str, Any]]:
    if input_path:
        return load_json_records(Path(input_path))
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("ShareGPT HF collection without --input requires: pip install huggingface_hub") from exc
    dataset_path = hf_hub_download(
        repo_id=SHAREGPT_DATASET,
        repo_type="dataset",
        filename=SHAREGPT_DEFAULT_FILE,
    )
    return load_json_records(Path(dataset_path))


if __name__ == "__main__":
    raise SystemExit(main())
