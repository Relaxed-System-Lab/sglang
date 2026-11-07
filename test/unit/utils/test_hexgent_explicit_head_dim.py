"""Offline Q1-real dimension tests; these are not model-serving measurements.

Qwen attention dimensions follow the official Qwen3-235B-A22B-Instruct-2507
config at revision ac9c66cc9b46af7306746a9250f23d47083d689e. Projection tests
isolate attention matrix work from this estimator's existing dense FFN term;
they do not establish a complete MoE service model.
"""

import copy
import importlib.util
from pathlib import Path
import sys
import types

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = _ROOT / "python" / "sglang"


def _load(name):
    for package_name, path in (
        ("sglang", _PACKAGE_ROOT), ("sglang.hexgent", _PACKAGE_ROOT / "hexgent")
    ):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(path)]
            sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"sglang.hexgent.{name}", _PACKAGE_ROOT / "hexgent" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _qwen_config():
    # BF16 dimensions only: no simulator-specific INT8/MoE execution assumptions.
    return {
        "hidden_size": 4096, "num_attention_heads": 64,
        "num_key_value_heads": 4, "head_dim": 128,
        "num_hidden_layers": 94, "intermediate_size": 12288,
        "vocab_size": 151936,
    }


def _worker(schemas, **kwargs):
    return schemas.WorkerDescriptor(
        id="d", url="http://d", worker_type="decode", **kwargs
    )


def _request(schemas, worker, **kwargs):
    return schemas.PlannerScheduleRequest(
        route="/v1/chat/completions", model="dimension-test", request_text="test",
        max_tokens=16,
        hexgent=schemas.HexgentRequestMetadata(workflow_id="wf", call_id="c", **kwargs),
        prefill_workers=[worker], decode_workers=[worker],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )


@pytest.mark.parametrize("source", ["worker_config", "worker_roofline", "request_config", "request_roofline"])
def test_explicit_head_dim_is_read_from_existing_model_config_entrypoints(source):
    module, schemas = _load("latency_estimator"), _load("schemas")
    worker = _worker(schemas)
    request = _request(schemas, worker)
    if source == "worker_config":
        worker.model_config = _qwen_config()
    elif source == "worker_roofline":
        worker.roofline = {"model_config": _qwen_config()}
    elif source == "request_config":
        request.hexgent.model_config = _qwen_config()
    else:
        request.hexgent.roofline = {"model_config": _qwen_config()}

    model = module._model_params(request, worker)

    assert model.head_dim == 128
    assert model.w_bit == model.a_bit == model.kv_bit == 16


def test_missing_head_dim_keeps_derived_dimension():
    module, schemas = _load("latency_estimator"), _load("schemas")
    config = _qwen_config()
    del config["head_dim"]

    model = module._model_params_from_payload(config, _worker(schemas))

    assert model.head_dim == 64


@pytest.mark.parametrize("stage,token_factor", [("prefill", 7), ("decode", 1)])
@pytest.mark.parametrize("tp", [1, 4])
def test_qwen_attention_projection_work_uses_true_query_and_kv_width(stage, token_factor, tp):
    module, schemas = _load("latency_estimator"), _load("schemas")
    model = module._model_params_from_payload(_qwen_config(), _worker(schemas, tp_size=tp))
    estimator = module.RooflineLatencyEstimator()
    ops, weights, _dynamic = estimator._linear_work_components(model, 7, stage)

    # Q=4096x8192, K/V=4096x512, O=8192x4096. Subtract the unchanged
    # dense-FFN term so the assertion targets attention projection geometry.
    ffn_parameters_per_rank = 3 * 4096 * 12288 / tp
    attention_parameters_per_rank = (2 * 4096 * 8192 + 2 * 4096 * 512) / tp
    assert ops - 2 * ffn_parameters_per_rank * token_factor == 2 * attention_parameters_per_rank * token_factor
    assert weights - 2 * ffn_parameters_per_rank == 2 * attention_parameters_per_rank


@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_qwen_attention_compute_and_kv_memory_use_explicit_dimension(stage):
    module, schemas = _load("latency_estimator"), _load("schemas")
    model = module._model_params_from_payload(_qwen_config(), _worker(schemas))
    sequence = 7
    query_tokens = sequence if stage == "prefill" else 1
    expected_ops = 4 * query_tokens * sequence * 128 * 64 + 5 * query_tokens * sequence * 64
    expected_bytes = 2 * query_tokens * 128 * 64 * 2 + 2 * sequence * 128 * 4 * 2

    assert module.RooflineLatencyEstimator()._attention_work(model, sequence, stage) == (
        expected_ops, expected_bytes
    )


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_qwen_transfer_agrees_with_calibrator_full_tp_group_bytes(tp, monkeypatch):
    monkeypatch.delenv("HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S", raising=False)
    monkeypatch.delenv("HEXGENT_TRANSFER_STARTUP_MS", raising=False)
    module, schemas = _load("latency_estimator"), _load("schemas")
    spec = importlib.util.spec_from_file_location(
        "head_dim_calibrator", _ROOT / "scripts/hexgent_real_trace_collectors/calibrate_pd_telemetry.py"
    )
    calibrator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibrator)
    config = _qwen_config()
    model = module._model_params_from_payload(config, _worker(schemas, tp_size=tp))
    expected_bytes_per_token = 2 * 94 * 4 * 128 * 2
    assert calibrator.kv_bytes_per_token_from_config(config, kv_dtype_bytes=2) == expected_bytes_per_token

    latency = module.RooflineLatencyEstimator().estimate_transfer_ms(
        model=model,
        hardware={"transfer_bandwidth_bytes_per_s": 1e9, "transfer_startup_ms": 5},
        input_tokens=1000,
    )

    assert latency == round(5 + expected_bytes_per_token * 1000 / 1e9 * 1000) == 198


@pytest.mark.parametrize("source", ["worker", "request"])
def test_hybrid_cache_distinguishes_explicit_head_dimensions(source, monkeypatch):
    monkeypatch.delenv("HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S", raising=False)
    monkeypatch.delenv("HEXGENT_TRANSFER_STARTUP_MS", raising=False)
    module, schemas = _load("latency_estimator"), _load("schemas")
    config = _qwen_config()
    config["head_dim"] = 64
    worker = _worker(schemas, hardware_params={"bandwidth": 1e12, "FP16": 1e14, "transfer_bandwidth_bytes_per_s": 1e9})
    request = _request(schemas, worker)
    if source == "worker":
        worker.model_config = config
    else:
        request.hexgent.model_config = config
    estimator = module.HybridLatencyEstimator()
    narrow = estimator.estimate(request, input_tokens=1000, output_tokens=16)
    config["head_dim"] = 128
    wide = estimator.estimate(request, input_tokens=1000, output_tokens=16)

    assert narrow.transfer_ms_by_decode_instance == [96]
    assert wide.transfer_ms_by_decode_instance == [193]
    assert len(estimator._transfer_cache) == 2


# Captured from real revision 33e34d5ccc before Q1-real implementation. Values:
# prefill(512), batched decode(512,16,batch3), transfer(512), prefill batch(128,512).
_LLAMA_GOLDEN = [
    ("8b", 1, "NVDA:A100_80G:SXM", (23, 144, 72, 29)),
    ("8b", 1, "NVDA:H100:SXM", (7, 96, 72, 9)),
    ("8b", 1, "NVDA:H20", (49, 80, 72, 61)),
    ("8b", 2, "NVDA:A100_80G:SXM", (12, 96, 72, 15)),
    ("8b", 2, "NVDA:H100:SXM", (4, 64, 72, 5)),
    ("8b", 2, "NVDA:H20", (25, 48, 72, 31)),
    ("8b", 4, "NVDA:A100_80G:SXM", (6, 64, 72, 8)),
    ("8b", 4, "NVDA:H100:SXM", (2, 48, 72, 3)),
    ("8b", 4, "NVDA:H20", (13, 32, 72, 16)),
    ("70b", 1, "NVDA:A100_80G:SXM", (227, 1223, 173, 283)),
    ("70b", 1, "NVDA:H100:SXM", (72, 816, 173, 89)),
    ("70b", 1, "NVDA:H20", (478, 608, 173, 597)),
    ("70b", 2, "NVDA:A100_80G:SXM", (115, 688, 173, 143)),
    ("70b", 2, "NVDA:H100:SXM", (36, 453, 173, 45)),
    ("70b", 2, "NVDA:H20", (242, 336, 173, 301)),
    ("70b", 4, "NVDA:A100_80G:SXM", (58, 416, 173, 73)),
    ("70b", 4, "NVDA:H100:SXM", (18, 272, 173, 23)),
    ("70b", 4, "NVDA:H20", (123, 208, 173, 153)),
]


@pytest.mark.parametrize("model_size,tp,hardware_name,expected", _LLAMA_GOLDEN)
@pytest.mark.parametrize("explicit_dimension", [False, True])
def test_llama_estimates_preserve_original_revision(
    model_size, tp, hardware_name, expected, explicit_dimension, monkeypatch
):
    monkeypatch.delenv("HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S", raising=False)
    monkeypatch.delenv("HEXGENT_TRANSFER_STARTUP_MS", raising=False)
    module, schemas, hardware_module = _load("latency_estimator"), _load("schemas"), _load("hardware")
    configs = {
        "8b": dict(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
                   num_hidden_layers=32, intermediate_size=14336, vocab_size=128256),
        "70b": dict(hidden_size=8192, num_attention_heads=64, num_key_value_heads=8,
                    num_hidden_layers=80, intermediate_size=28672, vocab_size=128256),
    }
    config = copy.deepcopy(configs[model_size])
    if explicit_dimension:
        config["head_dim"] = 128
    model = module._model_params_from_payload(config, _worker(schemas, tp_size=tp))
    hardware = hardware_module.resolve_hardware_profile(hardware_name)
    hardware.update(transfer_bandwidth_bytes_per_s=1e9, transfer_startup_ms=5)
    estimator = module.RooflineLatencyEstimator()

    actual = (
        estimator.estimate_prefill_ms(model=model, hardware=hardware, input_tokens=512),
        estimator.estimate_decode_total_ms(model=model, hardware=hardware, input_tokens=512, output_tokens=16, batch_size=3),
        estimator.estimate_transfer_ms(model=model, hardware=hardware, input_tokens=512),
        estimator.estimate_prefill_batch_ms(model=model, hardware=hardware, input_tokens=[128, 512]),
    )

    assert model.head_dim == 128
    assert actual == expected
