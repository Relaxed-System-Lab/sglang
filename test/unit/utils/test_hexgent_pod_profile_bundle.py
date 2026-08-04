import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "scripts" / "hexgent_pod_profile" / "profile_traces.py"
_IMPUTE_MODULE_PATH = _ROOT / "scripts" / "hexgent_pod_profile" / "impute_decode_tarball.py"
_COMPACT_QWEN_MODULE_PATH = _ROOT / "scripts" / "hexgent_pod_profile" / "compact_qwen_profile_tarball.py"
_REPAIR_QWEN_MODULE_PATH = _ROOT / "scripts" / "hexgent_pod_profile" / "repair_qwen_tarball_from_server_log.py"
_REPAIR_QWEN_STREAM_MODULE_PATH = _ROOT / "scripts" / "hexgent_pod_profile" / "repair_qwen_decode_from_stream_rows.py"
_AUTO_SCRIPT_PATH = _ROOT / "scripts" / "hexgent_pod_profile" / "run_gpu_profile_suite.sh"
_SETUP_SCRIPT_PATH = _ROOT / "scripts" / "hexgent_pod_profile" / "setup_pod_env.sh"
_SMOKE_SCRIPT_PATH = _ROOT / "scripts" / "hexgent_pod_profile" / "smoke_test_server.sh"
_HEXGENT_1P1D_LAUNCHER_PATH = (
    _ROOT / "scripts" / "hexgent_real_trace_collectors" / "start_hexgent_pd_1p1d.sh"
)
_HEXGENT_1P1D_LOAD_RUNNER_PATH = (
    _ROOT / "scripts" / "hexgent_real_trace_collectors" / "run_hexgent_1p1d_load.sh"
)
_HEXGENT_1P1D_ANALYZER_PATH = (
    _ROOT / "scripts" / "hexgent_real_trace_collectors" / "analyze_hexgent_1p1d_run.py"
)
_HEXGENT_E2E_SETUP_PATH = (
    _ROOT / "scripts" / "hexgent_real_trace_collectors" / "setup_hexgent_e2e_pod_env.sh"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("hexgent_pod_profile", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_impute_module():
    spec = importlib.util.spec_from_file_location("hexgent_decode_imputer", _IMPUTE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_compact_qwen_module():
    spec = importlib.util.spec_from_file_location("hexgent_qwen_compactor", _COMPACT_QWEN_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_repair_qwen_module():
    spec = importlib.util.spec_from_file_location("hexgent_qwen_repair", _REPAIR_QWEN_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_repair_qwen_stream_module():
    spec = importlib.util.spec_from_file_location("hexgent_qwen_stream_repair", _REPAIR_QWEN_STREAM_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_hexgent_1p1d_analyzer_module():
    spec = importlib.util.spec_from_file_location(
        "hexgent_1p1d_analyzer", _HEXGENT_1P1D_ANALYZER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_convert_call_jsonl_to_workflow_trace(tmp_path):
    module = _load_module()
    raw_path = tmp_path / "calls.jsonl"
    output_path = tmp_path / "workflows.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "workflow_id": "wf-1",
                "call_id": "call-1",
                "parents": [],
                "request": {"messages": [{"role": "user", "content": "hello"}]},
                "input_len": 11,
                "output_len": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    written = module.convert_trace(raw_path, "call_jsonl", output_path, max_calls=None)

    assert written == 1
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["workflow_id"] == "wf-1"
    assert rows[0]["calls"][0]["call_id"] == "call-1"
    assert rows[0]["calls"][0]["input_len"] == 11


def test_convert_dataset_records_for_bfcl_and_hotpotqa():
    module = _load_module()

    bfcl = module.convert_bfcl_records(
        [
            {
                "id": "bfcl-1",
                "question": [[{"role": "user", "content": "call a tool"}]],
                "function": [{"name": "search"}],
                "ground_truth": [{"name": "search", "arguments": {"q": "x"}}],
            }
        ]
    )
    hotpot = module.convert_hotpotqa_rows_to_lats_style(
        [
            {
                "id": "hotpot-1",
                "question": "Who founded the company?",
                "answer": "Alice",
                "level": "easy",
                "context": {"title": ["A"], "sentences": [["Alice founded it."]]},
            }
        ]
    )

    assert bfcl == []
    assert hotpot[0]["workflow_type"] == "lats"
    assert len(hotpot[0]["calls"]) > 1


def test_sharegpt_conversation_turns_become_sequential_workflow_calls():
    module = _load_module()

    workflows = module.convert_sharegpt_records(
        [
            {
                "id": "sharegpt-multi",
                "conversations": [
                    {"from": "human", "value": "question one"},
                    {"from": "gpt", "value": "answer one with several words"},
                    {"from": "human", "value": "question two"},
                    {"from": "gpt", "value": "answer two"},
                ],
            }
        ]
    )

    calls = workflows[0]["calls"]
    assert workflows[0]["workflow_id"] == "sharegpt-multi"
    assert len(calls) == 2
    assert calls[0]["call_id"] == "sharegpt-multi-call-0"
    assert calls[0]["parents"] == []
    assert calls[0]["request"]["messages"] == [{"role": "user", "content": "question one"}]
    assert calls[1]["call_id"] == "sharegpt-multi-call-1"
    assert calls[1]["parents"] == ["sharegpt-multi-call-0"]
    assert calls[1]["request"]["messages"][-1] == {"role": "user", "content": "question two"}
    assert calls[1]["output_len"] == 2


def test_bfcl_converter_keeps_only_multi_call_workflows():
    module = _load_module()

    workflows = module.convert_bfcl_records(
        [
            {
                "id": "single-turn",
                "question": [[{"role": "user", "content": "one"}]],
                "function": [{"name": "search"}],
            },
            {
                "id": "multi-turn",
                "question": [
                    [{"role": "user", "content": "first"}],
                    [{"role": "user", "content": "second"}],
                ],
                "function": [{"name": "search"}],
            },
        ]
    )

    assert [workflow["workflow_id"] for workflow in workflows] == ["multi-turn"]
    calls = workflows[0]["calls"]
    assert len(calls) == 2
    assert calls[1]["parents"] == ["multi-turn-call-0"]
    assert "tools" not in calls[0]["request"]
    assert "Available tools" in calls[0]["request"]["messages"][0]["content"]


def test_http_error_message_includes_response_body():
    module = _load_module()

    assert module.format_http_error(400, "bad schema") == "HTTP 400: bad schema"


def test_auto_suite_can_generate_default_hf_traces():
    script = _AUTO_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "ensure_default_traces" in script
    assert "--input-format sharegpt_hf" in script
    assert "--input-format bfcl_hf" in script
    assert "--input-format hotpotqa_hf" in script


def test_setup_script_installs_plain_pytorch_pod_runtime_dependencies():
    script = _SETUP_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "libnuma1" in script
    assert "sglang-kernel" in script
    assert "SGLANG_KERNEL_SPEC" in script


def test_smoke_test_script_checks_server_before_profiling():
    script = _SMOKE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "python -m sglang.launch_server" in script
    assert "/v1/models" in script
    assert "/v1/chat/completions" in script
    assert "SMOKE_MODEL" in script
    assert "SMOKE_CUDA_DEVICES" in script


def test_1p1d_launcher_keeps_batching_capacity_and_telemetry_enabled():
    script = _HEXGENT_1P1D_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "PREFILL_MAX_RUNNING_REQUESTS:-64" in script
    assert "PREFILL_MAX_BATCH_REQUESTS:-32" in script
    assert "PREFILL_CHUNKED_PREFILL_SIZE:-16384" in script
    assert "DECODE_MAX_RUNNING_REQUESTS:-256" in script
    assert "HEXGENT_PLANNER_TIMEOUT_MS:-10000" in script
    assert "HEXGENT_FALLBACK_POLICY:-pd_policy" in script
    assert '--hexgent-fallback-policy "$HEXGENT_FALLBACK_POLICY"' in script
    assert "PROMETHEUS_PORT:-29100" in script
    assert "--hexgent-admission-queue" in script
    assert "SGLANG_HEXGENT_CUDA_BATCH_TELEMETRY=1" in script
    assert "SGLANG_KV_TRANSFER_TELEMETRY=1" in script
    assert "MC_INTRANODE_NVLINK=1" in script
    assert "stop_listener_port()" in script
    assert "ss -ltnp" in script
    assert 'stop_listener_port "$BOOTSTRAP_PORT"' in script
    assert 'stop_listener_port "$PREFILL_PORT"' in script
    assert 'stop_listener_port "$DECODE_PORT"' in script
    assert 'stop_listener_port "$ROUTER_PORT"' in script
    assert 'stop_listener_port "$PROMETHEUS_PORT"' in script


def test_hexgent_e2e_setup_copies_the_complete_srt_runtime_tree():
    script = _HEXGENT_E2E_SETUP_PATH.read_text(encoding="utf-8")

    assert 'cp -a "$SRC_DIR/python/sglang/srt/." "$SITE_PACKAGES/sglang/srt/"' in script
    assert "libgoogle-glog0v5" in script
    assert "libjsoncpp25" in script
    assert "from mooncake.engine import TransferEngine" in script
    assert "VENV_CUDA_RUNTIME_DIR" in script
    assert 'assert torch.__version__.split("+")[0] == "2.9.1"' in script
    assert 'python -m pip install -e "$SRC_DIR/python"' in script
    assert '--no-deps' not in script
    assert script.count("import sglang.srt.entrypoints.http_server") >= 2


def test_1p1d_load_runner_preserves_replay_and_telemetry_artifacts():
    script = _HEXGENT_1P1D_LOAD_RUNNER_PATH.read_text(encoding="utf-8")

    assert "TRACE_PATH=" in script
    assert 'ARRIVAL_RATE:-12' in script
    assert 'snapshot_workers "$RUN_DIR/snapshots.jsonl"' in script
    assert "nvidia-smi --query-gpu=index,memory.used,utilization.gpu" in script
    assert '--timeout-s "$TIMEOUT_S"' in script
    assert '--output-path "$RUN_DIR/calls.jsonl"' in script
    assert "planner_invocations.jsonl" in script
    assert "PREFILL_LOG_START" in script
    assert "DECODE_LOG_START" in script
    assert '"$RUN_DIR/prefill_worker.log"' in script
    assert '"$RUN_DIR/decode_worker.log"' in script


def test_1p1d_analyzer_prefers_the_current_runs_decode_log(tmp_path):
    module = _load_hexgent_1p1d_analyzer_module()
    run_dir = tmp_path / "run"
    log_dir = tmp_path / "shared_logs"
    run_dir.mkdir()
    log_dir.mkdir()
    (run_dir / "decode_worker.log").write_text(
        "transfer_duration=7.5ms\n",
        encoding="utf-8",
    )
    (log_dir / "decode.log").write_text(
        "Using Intra-Node NVLink transport\ntransfer_duration=99.0ms\n",
        encoding="utf-8",
    )

    assert module.select_decode_log_path(run_dir, log_dir) == run_dir / "decode_worker.log"
    metrics = module.transfer_metrics(
        module.select_decode_log_path(run_dir, log_dir), log_dir / "decode.log"
    )
    assert metrics["nvlink_transport_selected"] is True
    assert metrics[
        "transfer_duration_ms"
    ]["p50"] == 7.5


def test_1p1d_analyzer_reports_planner_batch_solve_timing():
    module = _load_hexgent_1p1d_analyzer_module()
    rows = [
        {"endpoint": "/enqueue_admission", "wall_ms": 0.2},
        {
            "endpoint": "/plan_pending_admissions",
            "wall_ms": 12.5,
            "cpu_ms": 11.0,
            "admission_count": 3,
            "event_count": 1,
            "result_count": 3,
        },
    ]

    metrics = module.planner_metrics(rows, "")

    assert metrics["endpoint_counts"] == {
        "/enqueue_admission": 1,
        "/plan_pending_admissions": 1,
    }
    assert metrics["plan_batch_wall_ms"]["p50"] == 12.5
    assert metrics["plan_batch_cpu_ms"]["p50"] == 11.0
    assert metrics["plan_batch_admission_count"]["p50"] == 3
    assert metrics["plan_batch_event_count"]["p50"] == 1
    assert metrics["plan_batch_result_count"]["p50"] == 3


def test_1p1d_analyzer_reports_router_selection_phase_timings():
    module = _load_hexgent_1p1d_analyzer_module()
    metrics = module.call_metrics(
        [
            {
                "error": None,
                "total_latency_ms": 20.0,
                "router_selection_wall_ms": 5.0,
                "router_snapshot_fetch_ms": 1.5,
                "router_admission_wait_ms": 3.0,
            }
        ]
    )

    assert metrics["total"] == 1
    assert metrics["errors"] == 0
    assert metrics["router_selection_wall_ms"]["p50"] == 5.0
    assert metrics["router_snapshot_fetch_ms"]["p50"] == 1.5
    assert metrics["router_admission_wait_ms"]["p50"] == 3.0


def test_sse_parser_extracts_content_and_usage():
    module = _load_module()

    events = list(
        module.iter_sse_json_events(
            [
                b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"y"}}],"usage":{"completion_tokens":2}}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
    )

    assert [event["choices"][0]["delta"]["content"] for event in events] == ["x", "y"]
    assert events[-1]["usage"]["completion_tokens"] == 2


def test_decode_row_falls_back_to_total_latency_minus_ttft_for_coarse_streaming():
    module = _load_module()

    measurement = module.CallMeasurement(
        workflow_id="wf-1",
        call_id="call-1",
        model="m",
        worker_class="A100_80G_TP4",
        input_len=128,
        output_len=32,
        trial=0,
        ttft_ms=25.0,
        total_latency_ms=90.0,
        output_tokens=32,
        inter_token_latencies_ms=[],
    )

    decode = module.build_decode_row(measurement)

    assert decode["decode_time_ms"] == pytest.approx(65.0)
    assert decode["decode_time_per_token_ms"] == pytest.approx(65.0 / 31.0)
    assert decode["decode_p50_ms"] == pytest.approx(65.0 / 31.0)
    assert decode["decode_p95_ms"] == pytest.approx(65.0 / 31.0)


def test_decode_row_uses_wall_clock_decode_time_for_chunked_streaming():
    module = _load_module()

    measurement = module.CallMeasurement(
        workflow_id="wf-1",
        call_id="call-1",
        model="qwen",
        worker_class="H100_TP4",
        input_len=128,
        output_len=64,
        trial=0,
        ttft_ms=100.0,
        total_latency_ms=730.0,
        output_tokens=64,
        inter_token_latencies_ms=[90.0, 90.0, 90.0],
    )

    decode = module.build_decode_row(measurement)

    assert decode["decode_time_ms"] == pytest.approx(630.0)
    assert decode["decode_time_per_token_ms"] == pytest.approx(630.0 / 63.0)
    assert decode["decode_p50_ms"] == pytest.approx(630.0 / 63.0)
    assert decode["decode_p95_ms"] == pytest.approx(630.0 / 63.0)
    assert decode["timing_source"] == "wall_clock_total_latency_minus_ttft_ms"


def test_decode_tarball_imputer_fills_zero_decode_rows(tmp_path):
    module = _load_impute_module()
    input_path = tmp_path / "input.tar.gz"
    output_path = tmp_path / "output.tar.gz"

    decode_rows = [
        {
            "call_id": "a",
            "output_tokens": 11,
            "decode_time_ms": 50.0,
            "decode_time_per_token_ms": 5.0,
            "decode_p50_ms": 5.0,
            "decode_p95_ms": 5.0,
        },
        {
            "call_id": "b",
            "output_tokens": 21,
            "decode_time_ms": 0.0,
            "decode_time_per_token_ms": 0.0,
            "decode_p50_ms": 0.0,
            "decode_p95_ms": 0.0,
        },
    ]
    prefill_rows = [{"call_id": "a", "prefill_time_ms": 10.0}]

    with tarfile.open(input_path, "w:gz") as tf:
        for name, rows in {
            "decode_sharegpt_qwen_bf16.jsonl": decode_rows,
            "prefill_sharegpt_qwen_bf16.jsonl": prefill_rows,
        }.items():
            payload = "\n".join(json.dumps(row) for row in rows) + "\n"
            data = payload.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    changed = module.impute_tarball(input_path, output_path)

    assert changed["decode_sharegpt_qwen_bf16.jsonl"] == 1
    with tarfile.open(output_path, "r:gz") as tf:
        rows = [
            json.loads(line)
            for line in tf.extractfile("decode_sharegpt_qwen_bf16.jsonl").read().decode("utf-8").splitlines()
            if line.strip()
        ]
    assert rows[1]["decode_time_ms"] == pytest.approx(100.0)
    assert rows[1]["decode_time_per_token_ms"] == pytest.approx(5.0)
    assert rows[1]["timing_source"] == "imputed_from_recorded_qwen_decode_rows"


def test_compact_qwen_tarball_drops_raw_and_compacts_decode_rows(tmp_path):
    module = _load_compact_qwen_module()
    input_path = tmp_path / "input.tar.gz"
    output_path = tmp_path / "output.tar.gz"

    decode_rows = [
        {
            "workflow_id": "wf",
            "call_id": "a",
            "model": "qwen",
            "worker_class": "A100",
            "input_len": 10,
            "output_len": 11,
            "trial": 0,
            "ttft_ms": 20.0,
            "decode_time_ms": 50.0,
            "output_tokens": 11,
            "extra": "x",
        },
        {
            "workflow_id": "wf",
            "call_id": "b",
            "model": "qwen",
            "worker_class": "A100",
            "input_len": 10,
            "output_len": 11,
            "trial": 0,
            "ttft_ms": 21.0,
            "decode_time_ms": 0.0,
            "output_tokens": 11,
            "extra": "y",
        },
    ]
    prefill_rows = [{"call_id": "a", "prefill_time_ms": 10.0, "extra": "keep"}]
    raw_rows = [{"call_id": "a", "error": None}]

    with tarfile.open(input_path, "w:gz") as tf:
        for name, rows in {
            "decode_bfcl_qwen_bf16.jsonl": decode_rows,
            "prefill_bfcl_qwen_bf16.jsonl": prefill_rows,
            "raw_bfcl_qwen_bf16.jsonl": raw_rows,
        }.items():
            payload = "\n".join(json.dumps(row) for row in rows) + "\n"
            data = payload.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    changed = module.compact_tarball(input_path, output_path)

    assert changed["decode_bfcl_qwen_bf16.jsonl"] >= 1
    with tarfile.open(output_path, "r:gz") as tf:
        names = tf.getnames()
        assert "raw_bfcl_qwen_bf16.jsonl" not in names
        rows = [
            json.loads(line)
            for line in tf.extractfile("decode_bfcl_qwen_bf16.jsonl").read().decode("utf-8").splitlines()
            if line.strip()
        ]
        assert list(rows[0].keys()) == [
            "workflow_id",
            "call_id",
            "model",
            "worker_class",
            "input_len",
            "output_len",
            "trial",
            "ttft_ms",
            "decode_time_ms",
        ]
        assert rows[1]["decode_time_ms"] > 0.0
        assert rows[0]["decode_time_ms"] != rows[1]["decode_time_ms"]


def test_repair_qwen_tarball_uses_server_log_then_imputes_tail(tmp_path):
    module = _load_repair_qwen_module()
    input_path = tmp_path / "input.tar.gz"
    log_path = tmp_path / "server_qwen.log"
    output_path = tmp_path / "output.tar.gz"

    decode_rows = [
        {
            "workflow_id": "wf",
            "call_id": "a",
            "model": "qwen",
            "worker_class": "H200",
            "input_len": 10,
            "output_len": 8,
            "trial": 0,
            "ttft_ms": 20.0,
            "total_latency_ms": 20.0,
            "output_tokens": 8,
            "inter_token_latencies_ms": [],
            "error": None,
            "phase": "decode",
            "decode_time_ms": 0.0,
            "decode_time_per_token_ms": 0.0,
            "decode_p50_ms": 0.0,
            "decode_p95_ms": 0.0,
            "timing_source": "broken",
        },
        {
            "workflow_id": "wf",
            "call_id": "b",
            "model": "qwen",
            "worker_class": "H200",
            "input_len": 10,
            "output_len": 6,
            "trial": 0,
            "ttft_ms": 21.0,
            "total_latency_ms": 21.0,
            "output_tokens": 6,
            "inter_token_latencies_ms": [],
            "error": None,
            "phase": "decode",
            "decode_time_ms": 0.0,
            "decode_time_per_token_ms": 0.0,
            "decode_p50_ms": 0.0,
            "decode_p95_ms": 0.0,
            "timing_source": "broken",
        },
    ]

    with tarfile.open(input_path, "w:gz") as tf:
        payload = "\n".join(json.dumps(row) for row in decode_rows) + "\n"
        data = payload.encode("utf-8")
        info = tarfile.TarInfo(name="H200_QWEN_TP4/decode_bfcl_qwen.jsonl")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    log_path.write_text(
        "\n".join(
            [
                "[2026-05-06 20:33:02 TP0] Prefill batch, #new-seq: 1, #new-token: 35, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, cuda graph: True, input throughput (token/s): 0.03",
                "[2026-05-06 20:33:04 TP0] Decode batch, #running-req: 1, #token: 68, token usage: 0.00, cuda graph: False, gen throughput (token/s): 0.13, #queue-req: 0",
                "[2026-05-06 20:33:07 TP0] Decode batch, #running-req: 1, #token: 108, token usage: 0.00, cuda graph: False, gen throughput (token/s): 13.58, #queue-req: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = module.repair_tarball(input_path, log_path, output_path)

    assert result["blocks"] == 1
    assert result["applied_rows"] == 1
    assert result["imputed_rows"] == 1
    with tarfile.open(output_path, "r:gz") as tf:
        rows = [
            json.loads(line)
            for line in tf.extractfile("H200_QWEN_TP4/decode_bfcl_qwen.jsonl").read().decode("utf-8").splitlines()
            if line.strip()
        ]
    assert rows[0]["timing_source"] == "server_log_decode_batches"
    assert rows[0]["decode_time_ms"] > 0.0
    assert rows[1]["timing_source"] == "imputed_after_server_log_coverage"
    assert rows[1]["decode_time_ms"] > 0.0


def test_repair_qwen_decode_from_full_stream_rows(tmp_path):
    module = _load_repair_qwen_stream_module()
    input_path = tmp_path / "input.tar.gz"
    output_path = tmp_path / "output.tar.gz"

    rows = [
        {
            "call_id": "full",
            "output_tokens": 4,
            "inter_token_latencies_ms": [10.0, 12.0, 14.0],
            "decode_time_ms": 0.0,
            "decode_time_per_token_ms": 0.0,
            "decode_p50_ms": 0.0,
            "decode_p95_ms": 0.0,
        },
        {
            "call_id": "missing",
            "output_tokens": 6,
            "inter_token_latencies_ms": [],
            "decode_time_ms": 0.0,
            "decode_time_per_token_ms": 0.0,
            "decode_p50_ms": 0.0,
            "decode_p95_ms": 0.0,
        },
    ]

    with tarfile.open(input_path, "w:gz") as tf:
        payload = "\n".join(json.dumps(row) for row in rows) + "\n"
        data = payload.encode("utf-8")
        info = tarfile.TarInfo(name="H200_QWEN_TP4/decode_hotpotqa_qwen.jsonl")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    stats = module.repair_tarball(input_path, output_path, jitter=0.0)

    assert stats["fallback_per_token_ms"] == pytest.approx(12.0)
    with tarfile.open(output_path, "r:gz") as tf:
        repaired = [
            json.loads(line)
            for line in tf.extractfile("H200_QWEN_TP4/decode_hotpotqa_qwen.jsonl").read().decode("utf-8").splitlines()
            if line.strip()
        ]
    assert repaired[0]["decode_time_ms"] == pytest.approx(36.0)
    assert repaired[0]["timing_source"] == "full_stream_inter_token_latency_ms"
    assert repaired[1]["decode_time_ms"] == pytest.approx(60.0)
    assert repaired[1]["timing_source"] == "imputed_from_full_stream_qwen_rows"


def test_fit_latency_db_uses_separate_prefill_and_decode_tables():
    module = _load_module()
    prefill_rows = [
        {"worker_class": "A100_80G_TP4", "input_len": 100, "output_len": 10, "prefill_time_ms": 20.0},
        {"worker_class": "A100_80G_TP4", "input_len": 200, "output_len": 10, "prefill_time_ms": 40.0},
    ]
    decode_rows = [
        {"worker_class": "A100_80G_TP4", "input_len": 100, "output_len": 10, "decode_time_ms": 30.0},
        {"worker_class": "A100_80G_TP4", "input_len": 100, "output_len": 20, "decode_time_ms": 60.0},
    ]

    latency_db = module.build_latency_db(prefill_rows, decode_rows)

    hardware = latency_db["hardware"]["A100_80G_TP4"]
    assert hardware["prefill_cold"]["input_coeff_ms"] > 0.0
    assert hardware["decode"]["output_coeff_ms"] > 0.0
    assert latency_db["transfer"]["same_kind"]["base_ms"] == pytest.approx(0.0)


def test_auto_profile_script_runs_models_and_traces_contract():
    script = _AUTO_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "meta-llama/Llama-3.1-70B-Instruct" in script
    assert "Qwen/Qwen3-235B-A22B-FP8" in script
    assert 'QWEN_TP="${QWEN_TP:-4}"' in script
    assert "hf_REPLACE_WITH_YOUR_TOKEN" in script
    assert "TRACE_SPECS" in script
    assert "wait_for_server" in script
    assert "profile_traces.py profile" in script
    assert "profile_traces.py fit-latency-db" in script
    assert "CUDA_DEVICES" in script
    assert "LLAMA_CUDA_DEVICES" in script
    assert "QWEN_CUDA_DEVICES" in script
    assert "--reasoning-parser qwen3" in script
    assert "--disable-piecewise-cuda-graph" in script
    assert "--moe-runner-backend cutlass" in script
    assert "hf auth login" in script
    assert "huggingface-cli login" not in script


def test_smoke_script_uses_current_hf_login_command():
    script = _SMOKE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "hf auth login" in script
    assert "huggingface-cli login" not in script
