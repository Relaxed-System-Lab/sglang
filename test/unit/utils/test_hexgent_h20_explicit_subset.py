"""Explicit SG subset selection contracts; synthetic archive only, no tokenizer."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "scripts/hexgent_real_trace_collectors/prepare_h20_subset.py"
_MEMBER = "inputs/sharegpt.jsonl"
_CAL = [
    ("1HuVef4_0", 4), ("2wtLdZm_0", 6), ("BmS3AX0_0", 5),
    ("BmS3AX0_10", 4), ("CmD5m6T_0", 5), ("CmD5m6T_9", 3),
    ("DhelrJT_0", 3), ("Dk6qZaf_0", 2),
]
_EVAL = [
    ("50SbvAt_0", 2), ("6CWcc2w_0", 2), ("6YMEjv4_0", 6),
    ("88iCu0j_0", 6), ("88iCu0j_11", 2), ("9UIByXH_0", 11),
    ("ACt45V7_0", 6), ("ACt45V7_11", 8),
]
_ASSISTANT_FIRST = {"CmD5m6T_9", "88iCu0j_11", "ACt45V7_11"}


def _load():
    spec = importlib.util.spec_from_file_location("h20_explicit_subset", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection():
    return {"sharegpt": {
        "calibration": [workflow_id for workflow_id, _ in _CAL],
        "heldout": [workflow_id for workflow_id, _ in _EVAL],
    }}


def _workflow(workflow_id, count):
    calls = []
    for index in range(count):
        messages = [
            {"role": "assistant", "content": f"prior café\n{workflow_id}:{index}"},
            {"role": "user", "content": f"next question {workflow_id}:{index}\n"},
        ]
        if workflow_id not in _ASSISTANT_FIRST:
            messages.insert(0, {"role": "user", "content": "original opening"})
        calls.append({
            "call_id": f"{workflow_id}_turn_{index}",
            "parents": [] if index == 0 else [f"{workflow_id}_turn_{index - 1}"],
            "request": {"messages": messages, "model": "historical-model"},
            "input_len": 14 + index * 7, "output_len": 1 + index * 3,
            "planner_hints": {
                "exclusive_profile_latency_ms": 1000,
                "input_tokens": 14 + index * 7, "predicted_output_tokens": 1 + index * 3,
            },
            "metadata": {"future_child_count": 20, "expected_response": "historical"},
        })
    return {
        "workflow_id": workflow_id, "workflow_type": "sharegpt",
        "arrival_offset": None, "metadata": {"source": "profile_jsonl"},
        "calls": calls,
    }


def _archive(tmp_path):
    path = tmp_path / "source.tar.gz"
    rows = [_workflow(workflow_id, count) for workflow_id, count in _CAL + _EVAL]
    rows.append(_workflow("ZZ-not-selected", 2))
    rows.reverse()
    raw_rows, data = {}, bytearray(b"\n")
    for line_number, row in enumerate(rows, start=2):
        raw = b" \t" + json.dumps(row, ensure_ascii=False).encode() + b"\r\n"
        raw_rows[row["workflow_id"]] = (row, raw, line_number)
        data.extend(raw)
    payload = bytes(data)
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(_MEMBER)
        info.size = len(payload)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(payload))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), payload, raw_rows


def _prepare(module, source, digest, output, **kwargs):
    return module.prepare_subsets(
        source_archive=source, source_sha256=digest, output_dir=output,
        members={"sharegpt": _MEMBER}, **kwargs,
    )


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_explicit_sg_lists_keep_complete_disjoint_32_and_43_calls(tmp_path):
    module = _load()
    source, digest, _, _ = _archive(tmp_path)
    output = tmp_path / "prepared"
    manifest = _prepare(module, source, digest, output, workflow_ids=_selection())

    for split, expected, count in (("calibration", _CAL, 32), ("heldout", _EVAL, 43)):
        rows = _read(output / f"sharegpt.{split}.sanitized.jsonl")
        assert [row["workflow_id"] for row in rows] == [item[0] for item in expected]
        assert [len(row["calls"]) for row in rows] == [item[1] for item in expected]
        assert sum(len(row["calls"]) for row in rows) == count
        assert manifest["selections"]["sharegpt"][split]["calls"] == count
        for row in rows:
            call_ids = {call["call_id"] for call in row["calls"]}
            assert len(call_ids) == len(row["calls"])
            assert sum(not call["parents"] for call in row["calls"]) == 1
            assert all(set(call["parents"]) <= call_ids for call in row["calls"])
    assert set(_selection()["sharegpt"]["calibration"]).isdisjoint(
        _selection()["sharegpt"]["heldout"])
    assert manifest["independence_claim"] is False
    assert manifest["unique_original_tasks"] is None


def test_explicit_sanitizer_preserves_13_assistant_histories_nulls_and_targets(tmp_path):
    module = _load()
    source, digest, _, raw_rows = _archive(tmp_path)
    output = tmp_path / "prepared"
    _prepare(module, source, digest, output, workflow_ids=_selection())

    assistant_first = 0
    for split in ("calibration", "heldout"):
        for clean in _read(output / f"sharegpt.{split}.sanitized.jsonl"):
            original = raw_rows[clean["workflow_id"]][0]
            assert set(clean) == {"workflow_id", "workflow_type", "arrival_offset", "calls"}
            assert clean["arrival_offset"] is original["arrival_offset"] is None
            assert clean["workflow_type"] == original["workflow_type"]
            for call, raw_call in zip(clean["calls"], original["calls"], strict=True):
                assert set(call) == {"call_id", "parents", "request", "input_len", "output_len"}
                assert set(call["request"]) == {"messages"}
                assert call["request"]["messages"] == raw_call["request"]["messages"]
                assert call["request"]["messages"][-1]["role"] == "user"
                assistant_first += call["request"]["messages"][0]["role"] == "assistant"
                for key in ("call_id", "parents", "input_len", "output_len"):
                    assert call[key] == raw_call[key]
    assert assistant_first == 13


def test_explicit_manifest_hashes_exact_selected_raw_rows_and_call_ids(tmp_path):
    module = _load()
    source, digest, payload, raw_rows = _archive(tmp_path)
    output = tmp_path / "prepared"
    manifest = _prepare(module, source, digest, output, workflow_ids=_selection())

    assert manifest["source_archive"]["sha256"] == digest
    assert manifest["source_members"]["sharegpt"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["preparation_tool_sha256"] == hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest()
    assert manifest["selection_rule"] == {
        "ordering": "explicit_workflow_ids", "performance_based_selection": False,
    }
    for split, ids in _selection()["sharegpt"].items():
        expected_raw = b"".join(raw_rows[workflow_id][1] for workflow_id in ids)
        assert (output / f"sharegpt.{split}.original.jsonl").read_bytes() == expected_raw
        records = manifest["selected_source_rows"]["sharegpt"][split]
        assert [record["workflow_id"] for record in records] == ids
        for record in records:
            row, raw, line_number = raw_rows[record["workflow_id"]]
            assert record["member_line_1based"] == line_number
            assert record["sha256"] == hashlib.sha256(raw).hexdigest()
            assert record["bytes"] == len(raw)
            assert record["call_ids"] == [call["call_id"] for call in row["calls"]]
    for record in manifest["files"]:
        actual = (output / record["path"]).read_bytes()
        assert record["sha256"] == hashlib.sha256(actual).hexdigest()
        assert record["bytes"] == len(actual)
    assert manifest["target_tokenizer_length_validation_pending"] is True
    assert manifest["actual_served_model_config_pending"] is True


def test_omitting_explicit_ids_keeps_original_default_prefix_contract(tmp_path):
    module = _load()
    source, digest, _, raw_rows = _archive(tmp_path)
    output = tmp_path / "prepared"
    manifest = _prepare(module, source, digest, output)
    ordered = sorted(raw_rows)

    assert manifest["selection_rule"] == {
        "ordering": "workflow_id_lexicographic", "calibration_count": 2,
        "heldout_count": 4, "performance_based_selection": False,
    }
    assert manifest["selections"]["sharegpt"]["calibration"]["workflow_ids"] == ordered[:2]
    assert manifest["selections"]["sharegpt"]["heldout"]["workflow_ids"] == ordered[2:6]
    assert "selected_source_rows" not in manifest
    for split, ids in (("calibration", ordered[:2]), ("heldout", ordered[2:6])):
        assert (output / f"sharegpt.{split}.original.jsonl").read_bytes() == b"".join(
            raw_rows[workflow_id][1] for workflow_id in ids)
        assert _read(output / f"sharegpt.{split}.sanitized.jsonl") == [
            module._sanitize(raw_rows[workflow_id][0]) for workflow_id in ids]


@pytest.mark.parametrize("case", ["missing", "duplicate", "overlap"])
def test_explicit_ids_reject_missing_repeated_or_shared_workflows_before_writing(tmp_path, case):
    module = _load()
    source, digest, _, _ = _archive(tmp_path)
    selection = _selection()
    if case == "missing":
        selection["sharegpt"]["heldout"][-1] = "not-in-source"
    elif case == "duplicate":
        selection["sharegpt"]["calibration"].append(selection["sharegpt"]["calibration"][0])
    else:
        selection["sharegpt"]["heldout"][-1] = selection["sharegpt"]["calibration"][0]
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="missing|unknown|duplicate|overlap|disjoint"):
        _prepare(module, source, digest, output, workflow_ids=selection)
    assert not output.exists()


def test_explicit_ids_never_overwrite_existing_evidence(tmp_path):
    module = _load()
    source, digest, _, _ = _archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("frozen evidence")

    with pytest.raises(FileExistsError):
        _prepare(module, source, digest, output, workflow_ids=_selection())
    assert sentinel.read_text() == "frozen evidence"
    assert list(output.iterdir()) == [sentinel]


def test_explicit_id_cli_prepares_only_requested_sharegpt_member(monkeypatch, tmp_path):
    module = _load()
    source, digest, _, _ = _archive(tmp_path)
    selection_path = tmp_path / "ids.json"
    selection_path.write_text(json.dumps(_selection()))
    output = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", [
        str(_MODULE_PATH), "--source-archive", str(source), "--source-sha256", digest,
        "--sharegpt-member", _MEMBER, "--workflow-ids-json", str(selection_path),
        "--output-dir", str(output),
    ])

    module.main()

    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["source_members"]) == {"sharegpt"}
    assert manifest["selections"]["sharegpt"]["calibration"]["calls"] == 32
    assert manifest["selections"]["sharegpt"]["heldout"]["calls"] == 43
    assert len(list(output.iterdir())) == 5
