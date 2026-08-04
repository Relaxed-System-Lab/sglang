import importlib.util
from pathlib import Path
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = (
    _ROOT / "scripts" / "hexgent_real_trace_collectors" / "summarize_scheduling_replay.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "hexgent_scheduling_summary", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_read_jsonl_accepts_utf8_bom(tmp_path):
    module = _load_module()
    profile = tmp_path / "profile.jsonl"
    profile.write_text(
        '\ufeff{"workflow_id": "wf", "call_id": "call", "total_latency_ms": 1.0}\n',
        encoding="utf-8",
    )

    assert module.read_jsonl(profile) == [
        {"workflow_id": "wf", "call_id": "call", "total_latency_ms": 1.0}
    ]


def test_profile_slo_uses_dag_critical_path_instead_of_sum():
    module = _load_module()
    calls = [
        {
            "call_id": "root",
            "parents": [],
            "exclusive_profile_latency_ms": 10.0,
        },
        {
            "call_id": "short",
            "parents": ["root"],
            "exclusive_profile_latency_ms": 5.0,
        },
        {
            "call_id": "long",
            "parents": ["root"],
            "exclusive_profile_latency_ms": 20.0,
        },
        {
            "call_id": "join",
            "parents": ["short", "long"],
            "exclusive_profile_latency_ms": 7.0,
        },
    ]

    assert module.critical_path_profile_latency_ms(calls) == pytest.approx(37.0)


def test_profile_override_replaces_old_model_latency():
    module = _load_module()
    calls = [
        {
            "workflow_id": "wf",
            "call_id": "first",
            "parents": [],
            "exclusive_profile_latency_ms": 100.0,
        },
        {
            "workflow_id": "wf",
            "call_id": "second",
            "parents": ["first"],
            "exclusive_profile_latency_ms": 200.0,
        },
    ]
    override = {("wf", "first"): 11.0, ("wf", "second"): 13.0}

    assert module.critical_path_profile_latency_ms(
        calls, exclusive_profile=override
    ) == pytest.approx(24.0)


def test_profile_slo_rejects_missing_parent():
    module = _load_module()
    calls = [
        {
            "call_id": "child",
            "parents": ["missing"],
            "exclusive_profile_latency_ms": 10.0,
        }
    ]

    with pytest.raises(ValueError, match="missing parent"):
        module.critical_path_profile_latency_ms(calls)
