#!/usr/bin/env python3
"""Build fixed-length DAG replay traces from the final H100 Llama profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types

repo_root = Path(__file__).resolve().parents[2]
sglang_package = types.ModuleType("sglang")
sglang_package.__path__ = [str(repo_root / "python" / "sglang")]
sglang_package.__package__ = "sglang"
sys.modules["sglang"] = sglang_package

from sglang.hexgent.workflow_trace import (
    convert_profile_rows_to_workflows,
    dump_workflow_traces_jsonl,
)


MODEL_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "vocab_size": 128256,
    "w_bit": 8,
    "a_bit": 16,
    "kv_bit": 16,
    "tp_size": 1,
}

H100_HARDWARE_PARAMS = {
    "memory_bandwidth_bytes_per_s": 3.35e12,
    "FP16": 9.89e14,
    # Initial same-node estimate. Actual Mooncake timing is reported separately.
    "transfer_bandwidth_bytes_per_s": 50e9,
}

SOURCES = {
    "sharegpt": "raw_sharegpt_llama_h100_tp4_final.clean.jsonl",
    "bfcl": "raw_bfcl_llama_h100_tp4_final.clean.jsonl",
    "hotpotqa": "raw_hotpotqa_llama_h100_tp4_final.clean.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="hexgent-llama8b-fp8")
    parser.add_argument("--max-workflows", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for trace, filename in SOURCES.items():
        rows = read_jsonl(args.profile_dir / filename)
        workflows = convert_profile_rows_to_workflows(rows, model=args.model)
        workflows = workflows[: args.max_workflows]
        for workflow in workflows:
            for call in workflow.calls:
                call.planner_hints["model_config"] = dict(MODEL_CONFIG)
                call.planner_hints["hardware_params"] = dict(
                    H100_HARDWARE_PARAMS
                )
        output = args.output_dir / f"{trace}_workflows{len(workflows)}.jsonl"
        dump_workflow_traces_jsonl(workflows, output)
        summary[trace] = {
            "path": str(output),
            "workflows": len(workflows),
            "calls": sum(len(workflow.calls) for workflow in workflows),
            "workflow_ids": [workflow.workflow_id for workflow in workflows],
        }

    (args.output_dir / "trace_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
