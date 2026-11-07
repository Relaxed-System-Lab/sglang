"""CPU-only tests for H20 inputs; tokenizer fixtures are not server observations."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/hexgent_real_trace_collectors/prepare_h20_runtime_inputs.py"
CONFIG = {
    "hidden_size": 8192, "intermediate_size": 28672, "num_hidden_layers": 80,
    "num_attention_heads": 64, "num_key_value_heads": 8, "vocab_size": 128256,
    "torch_dtype": "bfloat16",
}


def load_script():
    name = "_hexgent_h20_runtime_inputs"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def workflows():
    parents = [[], ["root"], ["root"], ["left", "right"]]
    calls = []
    for index, (name, edges) in enumerate(zip(["root", "left", "right", "join"], parents)):
        calls.append({
            "call_id": name, "parents": edges, "input_len": index + 100,
            "output_len": index + 5,
            "request": {"messages": [{"role": "user", "content": f"unchanged {name}"}]},
        })
    return [{"workflow_id": "wf", "workflow_type": "fanout", "arrival_offset": 0.25,
             "calls": calls}]


class TokenizerFixture:
    def __init__(self, counts=(11, 23, 37, 49)):
        self.counts = list(counts)
        self.seen = []
        self.seen_kwargs = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=True):
        assert tokenize is True
        assert add_generation_prompt is True
        self.seen.append(copy.deepcopy(messages))
        self.seen_kwargs.append({"tokenize": tokenize, "add_generation_prompt": add_generation_prompt,
                                 "return_dict": return_dict})
        token_ids = list(range(self.counts[len(self.seen) - 1]))
        # The installed TokenizersBackend defaults to a mapping with two keys;
        # len(mapping) is not the prompt token count.
        if return_dict:
            return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}
        return token_ids


def test_dag_messages_and_output_targets_are_preserved_without_mutating_input():
    module = load_script()
    source = workflows()
    original = copy.deepcopy(source)
    tokenizer = TokenizerFixture()

    prepared, counts = module.prepare_workflows(source, CONFIG, tokenizer, tp_size=2)

    assert source == original
    assert prepared[0]["workflow_id"] == source[0]["workflow_id"]
    assert prepared[0]["arrival_offset"] == source[0]["arrival_offset"]
    for old, new in zip(source[0]["calls"], prepared[0]["calls"]):
        for key in ("call_id", "parents", "request", "output_len"):
            assert new[key] == old[key]
    assert tokenizer.seen == [call["request"]["messages"] for call in source[0]["calls"]]
    assert len(counts) == 4


def test_local_tokenizer_counts_are_separate_from_historical_declared_counts():
    module = load_script()
    tokenizer = TokenizerFixture((131073, 23, 37, 49))

    prepared, counts = module.prepare_workflows(workflows(), CONFIG, tokenizer, tp_size=2)

    assert [call["input_len"] for call in prepared[0]["calls"]] == [131073, 23, 37, 49]
    assert tokenizer.seen_kwargs == [
        {"tokenize": True, "add_generation_prompt": True, "return_dict": False}
    ] * 4
    assert [row["historical_declared_input_len"] for row in counts] == [100, 101, 102, 103]
    assert [row["local_tokenizer_input_len"] for row in counts] == [131073, 23, 37, 49]
    assert {(row["workflow_id"], row["call_id"]) for row in counts} == {
        ("wf", "root"), ("wf", "left"), ("wf", "right"), ("wf", "join")
    }


def test_only_static_model_and_nominal_hardware_hints_are_injected():
    module = load_script()
    source = workflows()
    source[0]["calls"][0]["planner_hints"] = {
        "workflow_horizon_ms": 999999, "critical_path_ms": 999999, "child_count": 80,
        "estimated_prefill_time": 1, "estimated_decode_time": 2,
        "estimated_transfer_time": 3, "exclusive_profile_latency_ms": 4,
        "workflow_completion_ms": 5, "workflow_attained_service_ms": 6,
    }

    prepared, _ = module.prepare_workflows(source, CONFIG, TokenizerFixture(), tp_size=2)

    for call in prepared[0]["calls"]:
        hints = call["planner_hints"]
        assert set(hints) == {"model_config", "hardware_params"}
        assert hints["model_config"] == {
            "hidden_size": 8192, "intermediate_size": 28672, "num_hidden_layers": 80,
            "num_attention_heads": 64, "num_key_value_heads": 8, "vocab_size": 128256,
            "head_dim": 128, "tp_size": 2, "w_bit": 16, "a_bit": 16, "kv_bit": 16,
        }
        assert hints["hardware_params"] == {
            "memory_bandwidth_bytes_per_s": 4096e9, "FP16": 148e12,
        }


@pytest.mark.parametrize("explicit,expected", [(None, 128), (160, 160)])
def test_actual_config_head_dimension_is_explicit_first_then_derived(explicit, expected):
    config = dict(CONFIG, head_dim=explicit)

    assert load_script().build_model_config(config, tp_size=2)["head_dim"] == expected


def make_files(tmp_path):
    trace = tmp_path / "fixture.sanitized.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in workflows()))
    model = tmp_path / "local-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(CONFIG))
    (model / "tokenizer_config.json").write_text('{"chat_template": "fixture"}')
    return trace, model


def test_manifest_records_provenance_and_fresh_output_files(tmp_path):
    module = load_script()
    trace, model = make_files(tmp_path)
    before = trace.read_bytes()
    output = tmp_path / "new-runtime-inputs"

    manifest = module.prepare_files([trace], model, output, tp_size=2, tokenizer=TokenizerFixture())

    assert trace.read_bytes() == before
    assert manifest["input_length_source"] == "local_tokenizer.apply_chat_template"
    assert manifest["server_observed_input_tokens"] is False
    assert manifest["tokenization"] == {"tokenize": True, "add_generation_prompt": True,
                                         "return_dict": False, "local_files_only": True,
                                         "truncation": False}
    assert manifest["hardware_source"] == "packaged_nominal_H20_catalog"
    assert manifest["transfer_in_trace"] is False
    assert manifest["model_config"]["tp_size"] == 2
    part = manifest["traces"][0]
    assert part["workflows"] == 1 and part["calls"] == 4
    assert len(part["source_sha256"]) == len(part["output_sha256"]) == 64
    assert part["historical_declared_input_len_total"] == 406
    assert part["local_tokenizer_input_len_total"] == 120
    assert json.loads((output / "manifest.json").read_text()) == manifest
    prepared = [json.loads(row) for row in Path(part["output_path"]).read_text().splitlines()]
    assert [call["input_len"] for call in prepared[0]["calls"]] == [11, 23, 37, 49]


def test_existing_output_directory_is_preserved(tmp_path):
    module = load_script()
    trace, model = make_files(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "retain.txt"
    sentinel.write_text("preserve")

    with pytest.raises(FileExistsError):
        module.prepare_files([trace], model, output, tp_size=2, tokenizer=TokenizerFixture())

    assert sentinel.read_text() == "preserve"
    assert list(output.iterdir()) == [sentinel]


def test_tokenizer_loader_is_lazy_and_local_files_only(monkeypatch, tmp_path):
    module = load_script()
    seen = []
    fixture = TokenizerFixture()

    class AutoTokenizerFixture:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen.append((path, kwargs))
            return fixture

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizerFixture))

    assert module.load_tokenizer(tmp_path) is fixture
    assert seen == [(str(tmp_path), {"local_files_only": True})]


def test_generated_static_config_reaches_roofline_without_linear_fallback(monkeypatch):
    module = load_script()
    prepared, _ = module.prepare_workflows(workflows(), CONFIG, TokenizerFixture(), tp_size=2)
    package_root = ROOT / "python/sglang"
    for name, path in (("sglang", package_root), ("sglang.hexgent", package_root / "hexgent")):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            monkeypatch.setitem(sys.modules, name, package)
    from sglang.hexgent.schemas import HexgentRequestMetadata, PlannerClusterState, PlannerScheduleRequest, WorkerDescriptor
    from sglang.hexgent.latency_estimator import HybridLatencyEstimator

    hints = prepared[0]["calls"][0]["planner_hints"]
    request = PlannerScheduleRequest(
        route="/v1/chat/completions", model="verified-served-model", request_text="current prompt",
        max_tokens=5, hexgent=HexgentRequestMetadata(workflow_id="wf", call_id="root", **hints),
        prefill_workers=[WorkerDescriptor(id="p", url="http://p", worker_type="prefill", hardware="H20")],
        decode_workers=[WorkerDescriptor(id="d", url="http://d", worker_type="decode", hardware="H20")],
        state=PlannerClusterState(current_time_ms=0),
    )
    estimator = HybridLatencyEstimator()

    def fail_linear(*args, **kwargs):
        raise AssertionError("unexpected linear fallback")

    for name in ("prefill_ms", "decode_ms", "transfer_ms"):
        monkeypatch.setattr(estimator.linear, name, fail_linear)
    # Synthetic test-only link values exercise the transfer branch. They are
    # never written to prepared traces and are not H20 calibration evidence.
    monkeypatch.setenv("HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S", "1000000000")
    monkeypatch.setenv("HEXGENT_TRANSFER_STARTUP_MS", "3")

    result = estimator.estimate(request, input_tokens=11, output_tokens=5)

    assert result.prefill_ms_by_instance[0] > 0
    assert result.decode_ms_by_instance[0] > 0
    assert result.transfer_ms_by_decode_instance[0] > 0
