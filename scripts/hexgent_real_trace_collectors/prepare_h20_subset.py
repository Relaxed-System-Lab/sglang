#!/usr/bin/env python3
"""Prepare reproducible fixed-token subsets from a verified local trace archive.

For each dataset, sort workflow_id lexicographically, take the calibration
prefix, then the heldout prefix immediately after it. Original subsets preserve
source JSONL lines. Sanitized subsets retain messages, IDs, parents and lengths;
they contain no historical model, metadata or planner hints. The output
directory must be new. This prepares inputs, not H20 serving measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


DEFAULT_MEMBERS = {
    "sharegpt": "formal_traces/h100_llama70b_profiled_lengths/sharegpt_100.jsonl",
    "hotpotqa": "formal_traces/h100_llama70b_profiled_lengths/hotpotqa.jsonl",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8") + b"\n"


def _sanitize(workflow: dict) -> dict:
    return {
        "workflow_id": workflow["workflow_id"],
        "workflow_type": workflow["workflow_type"],
        "arrival_offset": workflow.get("arrival_offset"),
        "calls": [
            {
                "call_id": call["call_id"],
                "parents": call.get("parents", []),
                "request": {"messages": call["request"]["messages"]},
                "input_len": call["input_len"],
                "output_len": call["output_len"],
            }
            for call in workflow["calls"]
        ],
    }


def prepare_subsets(
    *, source_archive: Path, source_sha256: str, output_dir: Path,
    members: dict[str, str] | None = None,
    calibration_count: int = 2, heldout_count: int = 4,
) -> dict:
    """Verify the source and write fixed, disjoint subsets into a new directory."""
    actual_sha256 = _file_sha256(source_archive)
    if actual_sha256 != source_sha256:
        raise ValueError(f"source archive SHA256 mismatch: {actual_sha256}")
    members = DEFAULT_MEMBERS if members is None else members
    outputs = {}
    source_members = {}
    selections = {}
    files = []
    with tarfile.open(source_archive, "r:gz") as archive:
        for dataset, member in sorted(members.items()):
            data = archive.extractfile(member).read()
            workflows = {}
            original_lines = {}
            for line in data.splitlines(keepends=True):
                if not line.strip():
                    continue
                workflow = json.loads(line)
                workflow_id = str(workflow["workflow_id"])
                if workflow_id in workflows:
                    raise ValueError(f"duplicate workflow_id in {member}: {workflow_id}")
                workflows[workflow_id] = workflow
                original_lines[workflow_id] = line
            ordered = sorted(workflows)
            required = calibration_count + heldout_count
            if len(ordered) < required:
                raise ValueError(f"{dataset} requires {required} workflows; found {len(ordered)}")
            source_members[dataset] = {
                "member": member, "sha256": _sha256(data), "bytes": len(data),
                "workflows": len(workflows),
                "calls": sum(len(workflow["calls"]) for workflow in workflows.values()),
            }
            splits = {
                "calibration": ordered[:calibration_count],
                "heldout": ordered[calibration_count:required],
            }
            selections[dataset] = {}
            for split, workflow_ids in splits.items():
                selected = [workflows[key] for key in workflow_ids]
                calls = [call for workflow in selected for call in workflow["calls"]]
                selections[dataset][split] = {
                    "workflow_ids": workflow_ids, "workflows": len(workflow_ids),
                    "calls": len(calls),
                    "total_recorded_input_tokens": sum(call["input_len"] for call in calls),
                    "total_target_output_tokens": sum(call["output_len"] for call in calls),
                    "max_recorded_input_plus_output": max(
                        (call["input_len"] + call["output_len"] for call in calls),
                        default=None,
                    ),
                }
                payloads = {
                    "original": b"".join(original_lines[key] for key in workflow_ids),
                    "sanitized": b"".join(_json_line(_sanitize(workflow)) for workflow in selected),
                }
                for version, payload in payloads.items():
                    name = f"{dataset}.{split}.{version}.jsonl"
                    outputs[name] = payload
                    files.append({
                        "path": name, "dataset": dataset, "split": split,
                        "version": version, "sha256": _sha256(payload),
                        "bytes": len(payload), "workflow_ids": workflow_ids,
                        "workflows": len(workflow_ids), "calls": len(calls),
                    })
    manifest = {
        "schema_version": 1,
        "evidence_kind": "input_preparation_only_no_H20_serving_result",
        "source_archive": {
            "path": str(source_archive.resolve()), "sha256": actual_sha256,
            "bytes": source_archive.stat().st_size,
        },
        "source_members": source_members,
        "preparation_tool": "prepare_h20_subset.py",
        "preparation_tool_sha256": _file_sha256(Path(__file__)),
        "selection_rule": {
            "ordering": "workflow_id_lexicographic",
            "calibration_count": calibration_count, "heldout_count": heldout_count,
            "performance_based_selection": False,
        },
        "sanitization": {
            "preserved": [
                "workflow_id", "workflow_type", "arrival_offset", "call_id",
                "parents", "request.messages verbatim", "input_len", "output_len",
            ],
            "request_keys_after": ["messages"],
            "all_metadata_and_planner_hints_removed": True,
            "future_children_metadata_added": False,
            "profile_or_hardware_hints_added": False,
            "original_subsets_are_provenance_only": True,
        },
        "target_tokenizer_length_validation_pending": True,
        "actual_served_model_config_pending": True,
        "unique_original_tasks": None,
        "independence_claim": False,
        "execution_scope": "fixed-token request DAG inputs, not live agent/tool execution",
        "selections": selections,
        "files": files,
    }
    outputs["manifest.json"] = json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8") + b"\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in outputs.items():
        (output_dir / name).write_bytes(payload)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--sharegpt-member", default=DEFAULT_MEMBERS["sharegpt"])
    parser.add_argument("--hotpotqa-member", default=DEFAULT_MEMBERS["hotpotqa"])
    parser.add_argument("--calibration-count", type=int, default=2)
    parser.add_argument("--heldout-count", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_subsets(
        source_archive=args.source_archive, source_sha256=args.source_sha256,
        output_dir=args.output_dir, calibration_count=args.calibration_count,
        heldout_count=args.heldout_count,
        members={"sharegpt": args.sharegpt_member, "hotpotqa": args.hotpotqa_member},
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "manifest_sha256": _file_sha256(args.output_dir / "manifest.json"),
        "evidence_kind": "input_preparation_only_no_H20_serving_result",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
