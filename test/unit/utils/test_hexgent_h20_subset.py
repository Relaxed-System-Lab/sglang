"""Synthetic input-preparation tests; no model-serving measurements."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "scripts/hexgent_real_trace_collectors/prepare_h20_subset.py"
_MEMBERS = {
    "sharegpt": "inputs/sharegpt.jsonl",
    "hotpotqa": "inputs/hotpotqa.jsonl",
}


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_h20_subset", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow(workflow_id):
    calls = []
    for suffix, parents in [
        ("root", []), ("left", ["root"]), ("right", ["root"]),
        ("join", ["left", "right"]),
    ]:
        calls.append({
            "call_id": f"{workflow_id}-{suffix}",
            "parents": [f"{workflow_id}-{parent}" for parent in parents],
            "input_len": 7, "output_len": 3,
            "request": {
                "messages": [{"role": "user", "content": f"café {workflow_id} {suffix}"}],
                "model": "historical-model", "hexgent": {"old_hint": 10},
                "timing": {"old_ms": 99}, "priority": 4,
            },
            "planner_hints": {"exclusive_profile_latency_ms": 100, "hardware": "old"},
            "metadata": {"future_child_count": 2, "expected_response": "old"},
        })
    return {
        "workflow_id": workflow_id, "workflow_type": "synthetic",
        "arrival_offset": None, "metadata": {"historical_profile": "old"},
        "calls": calls,
    }


def _archive(tmp_path, count=8):
    path = tmp_path / "source.tar.gz"
    payloads = {}
    with tarfile.open(path, "w:gz") as archive:
        for dataset, member in _MEMBERS.items():
            rows = [_workflow(f"{dataset}-{i}") for i in reversed(range(count))]
            data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
            info = tarfile.TarInfo(member)
            info.size = len(data)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
            payloads[dataset] = data
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), payloads


def _prepare(module, archive, archive_sha, output, **kwargs):
    return module.prepare_subsets(
        source_archive=archive, source_sha256=archive_sha,
        output_dir=output, members=_MEMBERS, **kwargs,
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_default_split_sorts_ids_and_keeps_calibration_heldout_disjoint(tmp_path):
    module = _load_module()
    source, digest, _ = _archive(tmp_path)
    output = tmp_path / "prepared"
    manifest = _prepare(module, source, digest, output)

    for dataset in _MEMBERS:
        calibration = manifest["selections"][dataset]["calibration"]["workflow_ids"]
        heldout = manifest["selections"][dataset]["heldout"]["workflow_ids"]
        assert calibration == [f"{dataset}-0", f"{dataset}-1"]
        assert heldout == [f"{dataset}-{i}" for i in range(2, 6)]
        assert set(calibration).isdisjoint(heldout)
        assert manifest["selections"][dataset]["calibration"]["calls"] == 8
        assert manifest["selections"][dataset]["heldout"]["calls"] == 16


def test_sanitization_preserves_messages_dag_and_lengths_without_old_hints(tmp_path):
    module = _load_module()
    source, digest, _ = _archive(tmp_path)
    output = tmp_path / "prepared"
    _prepare(module, source, digest, output)

    for dataset in _MEMBERS:
        for split in ("calibration", "heldout"):
            originals = _read_jsonl(output / f"{dataset}.{split}.original.jsonl")
            sanitized = _read_jsonl(output / f"{dataset}.{split}.sanitized.jsonl")
            for raw, clean in zip(originals, sanitized):
                assert set(clean) == {"workflow_id", "workflow_type", "arrival_offset", "calls"}
                assert clean["workflow_id"] == raw["workflow_id"]
                assert len(clean["calls"]) == len(raw["calls"])
                for original_call, clean_call in zip(raw["calls"], clean["calls"]):
                    assert set(clean_call) == {"call_id", "parents", "request", "input_len", "output_len"}
                    assert set(clean_call["request"]) == {"messages"}
                    assert clean_call["request"]["messages"] == original_call["request"]["messages"]
                    for key in ("call_id", "parents", "input_len", "output_len"):
                        assert clean_call[key] == original_call[key]


def test_manifest_tracks_archive_member_and_output_hashes(tmp_path):
    module = _load_module()
    source, digest, payloads = _archive(tmp_path)
    output = tmp_path / "prepared"
    manifest = _prepare(module, source, digest, output)

    assert manifest["source_archive"]["sha256"] == digest
    assert manifest["source_archive"]["bytes"] == source.stat().st_size
    for dataset, data in payloads.items():
        assert manifest["source_members"][dataset]["member"] == _MEMBERS[dataset]
        assert manifest["source_members"][dataset]["sha256"] == hashlib.sha256(data).hexdigest()
    for record in manifest["files"]:
        data = (output / record["path"]).read_bytes()
        assert record["sha256"] == hashlib.sha256(data).hexdigest()
        assert record["bytes"] == len(data)
    assert manifest["evidence_kind"] == "input_preparation_only_no_H20_serving_result"
    assert manifest["unique_original_tasks"] is None
    assert manifest["independence_claim"] is False


def test_repeated_preparation_has_identical_jsonl_and_manifest_bytes(tmp_path):
    module = _load_module()
    source, digest, _ = _archive(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    _prepare(module, source, digest, first)
    _prepare(module, source, digest, second)

    names = sorted(path.name for path in first.iterdir())
    assert names == sorted(path.name for path in second.iterdir())
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in names)


def test_explicit_counts_change_only_fixed_sorted_cut_points(tmp_path):
    module = _load_module()
    source, digest, _ = _archive(tmp_path)
    manifest = _prepare(module, source, digest, tmp_path / "prepared", calibration_count=1, heldout_count=2)

    assert manifest["selections"]["sharegpt"]["calibration"]["workflow_ids"] == ["sharegpt-0"]
    assert manifest["selections"]["sharegpt"]["heldout"]["workflow_ids"] == ["sharegpt-1", "sharegpt-2"]


def test_source_hash_mismatch_is_rejected_before_writing(tmp_path):
    module = _load_module()
    source, _, _ = _archive(tmp_path)
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="archive SHA256 mismatch"):
        _prepare(module, source, "0" * 64, output)
    assert not output.exists()


def test_insufficient_workflows_do_not_silently_shrink_the_split(tmp_path):
    module = _load_module()
    source, digest, _ = _archive(tmp_path, count=3)

    with pytest.raises(ValueError, match="requires 6 workflows"):
        _prepare(module, source, digest, tmp_path / "prepared")


def test_existing_evidence_directory_is_never_overwritten(tmp_path):
    module = _load_module()
    source, digest, _ = _archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing evidence")

    with pytest.raises(FileExistsError):
        _prepare(module, source, digest, output)
    assert sentinel.read_text() == "existing evidence"
    assert list(output.iterdir()) == [sentinel]
