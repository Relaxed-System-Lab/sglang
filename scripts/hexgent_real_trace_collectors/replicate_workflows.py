"""Replicate workflow traces without changing request payloads or their DAGs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def _replicated_id(value: str, replica: int) -> str:
    return f"{value}__replica_{replica}"


def replicate_workflows(source: Path, destination: Path, replicas: int) -> int:
    """Write ``replicas`` independent copies of each workflow in ``source``."""
    if replicas < 1:
        raise ValueError("replicas must be at least one")

    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open(encoding="utf-8") as input_file, destination.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line in input_file:
            if not line.strip():
                continue
            workflow = json.loads(line)
            workflow_id = str(workflow["workflow_id"])
            for replica in range(replicas):
                clone = copy.deepcopy(workflow)
                clone["workflow_id"] = _replicated_id(workflow_id, replica)
                for call in clone.get("calls", []):
                    call["call_id"] = _replicated_id(str(call["call_id"]), replica)
                    call["parents"] = [
                        _replicated_id(str(parent), replica)
                        for parent in call.get("parents", [])
                    ]
                output_file.write(json.dumps(clone, separators=(",", ":")) + "\n")
                written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    args = parser.parse_args()
    print(replicate_workflows(args.source, args.destination, args.replicas))


if __name__ == "__main__":
    main()
