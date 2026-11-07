#!/usr/bin/env python3
"""Prepare H20 replay inputs using a local tokenizer and static runtime config.

Input traces must already be sanitized. IDs, dependency edges, messages and
output targets are preserved. Prompt counts are local chat-template counts;
server-observed counts and transfer calibration are recorded separately.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


H20_HARDWARE_PARAMS = {
    "memory_bandwidth_bytes_per_s": 4096e9,
    "FP16": 148e12,
}


def build_model_config(config, *, tp_size):
    fields = (
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size",
    )
    model = {key: int(config[key]) for key in fields}
    explicit_head_dim = config.get("head_dim")
    model.update(
        head_dim=(int(explicit_head_dim) if explicit_head_dim is not None
                  else model["hidden_size"] // model["num_attention_heads"]),
        tp_size=int(tp_size), w_bit=16, a_bit=16, kv_bit=16,
    )
    return model


def load_tokenizer(model_path):
    # Keep transformers and its optional runtime dependencies out of pure tests.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)


def prepare_workflows(workflows, config, tokenizer, *, tp_size):
    prepared = copy.deepcopy(workflows)
    static_hints = {
        "model_config": build_model_config(config, tp_size=tp_size),
        "hardware_params": H20_HARDWARE_PARAMS,
    }
    counts = []
    for workflow in prepared:
        for call in workflow["calls"]:
            declared_input_len = call["input_len"]
            token_ids = tokenizer.apply_chat_template(
                call["request"]["messages"], tokenize=True,
                add_generation_prompt=True, return_dict=False,
            )
            call["input_len"] = len(token_ids)
            # Replace the complete hint object so historical per-call timing,
            # workflow SLO/progress and future-DAG hints cannot take precedence.
            call["planner_hints"] = copy.deepcopy(static_hints)
            counts.append({
                "workflow_id": workflow["workflow_id"],
                "call_id": call["call_id"],
                "historical_declared_input_len": declared_input_len,
                "local_tokenizer_input_len": call["input_len"],
            })
    return prepared, counts


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def prepare_files(trace_paths, model_path, output_dir, *, tp_size, tokenizer=None):
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    # Every invocation receives a fresh directory; previous evidence is retained.
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = model_path / "config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if tokenizer is None:
        tokenizer = load_tokenizer(model_path)
    tokenizer_files = sorted(model_path.glob("tokenizer*"))
    tokenizer_files += [model_path / name for name in ("special_tokens_map.json", "chat_template.jinja")]
    manifest = {
        "source_model_path": str(model_path),
        "model_config_source": str(config_path),
        "model_config_sha256": sha256_bytes(config_bytes),
        "model_config": build_model_config(config, tp_size=tp_size),
        "declared_runtime_precision": {"weights": "BF16", "activations": "BF16", "kv": "BF16"},
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_files_sha256": {
            path.name: sha256_bytes(path.read_bytes())
            for path in tokenizer_files if path.is_file()
        },
        "preparer_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "input_length_source": "local_tokenizer.apply_chat_template",
        "server_observed_input_tokens": False,
        "tokenization": {
            "tokenize": True, "add_generation_prompt": True,
            "return_dict": False, "local_files_only": True, "truncation": False,
        },
        "hardware_profile_key": "NVDA:H20",
        "hardware_source": "packaged_nominal_H20_catalog",
        "hardware_params": dict(H20_HARDWARE_PARAMS),
        "transfer_in_trace": False,
        "transfer_calibration": "freeze separately in planner HEXGENT_TRANSFER environment",
        "traces": [],
    }
    for trace_path in trace_paths:
        trace_path = Path(trace_path).resolve()
        source_bytes = trace_path.read_bytes()
        workflows = [json.loads(line) for line in source_bytes.decode("utf-8-sig").splitlines() if line.strip()]
        prepared, counts = prepare_workflows(workflows, config, tokenizer, tp_size=tp_size)
        output_path = output_dir / (trace_path.stem + ".runtime.jsonl")
        output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in prepared).encode("utf-8")
        with output_path.open("xb") as target:
            target.write(output_bytes)
        manifest["traces"].append({
            "source_path": str(trace_path),
            "source_sha256": sha256_bytes(source_bytes),
            "output_path": str(output_path),
            "output_sha256": sha256_bytes(output_bytes),
            "workflows": len(prepared), "calls": len(counts),
            "historical_declared_input_len_total": sum(row["historical_declared_input_len"] for row in counts),
            "local_tokenizer_input_len_total": sum(row["local_tokenizer_input_len"] for row in counts),
            "input_lengths": counts,
        })
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", type=Path, nargs="+", required=True,
                        help="Sanitized JSONL traces with distinct basenames.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; existing directories are never overwritten.")
    args = parser.parse_args(argv)
    manifest = prepare_files(args.trace_path, args.model_path, args.output_dir, tp_size=args.tp_size)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "traces": len(manifest["traces"]),
                      "calls": sum(row["calls"] for row in manifest["traces"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
