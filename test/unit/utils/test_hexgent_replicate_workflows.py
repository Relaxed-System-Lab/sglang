import json
from pathlib import Path


def test_replicate_workflows_rewrites_ids_and_preserves_dependencies(tmp_path: Path):
    from scripts.hexgent_real_trace_collectors.replicate_workflows import (
        replicate_workflows,
    )

    source = tmp_path / "source.jsonl"
    destination = tmp_path / "replicated.jsonl"
    source.write_text(
        json.dumps(
            {
                "workflow_id": "workflow-1",
                "calls": [
                    {"call_id": "root", "parents": [], "output_len": 7},
                    {"call_id": "child", "parents": ["root"], "output_len": 11},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count = replicate_workflows(source, destination, replicas=2)

    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert count == 2
    assert [row["workflow_id"] for row in rows] == [
        "workflow-1__replica_0",
        "workflow-1__replica_1",
    ]
    assert rows[0]["calls"][0]["call_id"] == "root__replica_0"
    assert rows[0]["calls"][1]["call_id"] == "child__replica_0"
    assert rows[0]["calls"][1]["parents"] == ["root__replica_0"]
    assert rows[1]["calls"][1]["parents"] == ["root__replica_1"]
    assert rows[1]["calls"][1]["output_len"] == 11
