import json
from pathlib import Path
import sys
import types

import pytest

from scripts.hexgent_real_trace_collectors.collectors import (
    parse_bfcl_records,
    parse_lats_hotpotqa_records,
    parse_sharegpt_records,
    replay_workflows,
)
from scripts.hexgent_real_trace_collectors.build_phase_profiles import build_phase_profiles
from scripts.hexgent_real_trace_collectors import build_phase_profiles as phase_profiles
from scripts.hexgent_real_trace_collectors.openai_compat import OpenAIChatClient
from scripts.hexgent_real_trace_collectors.schema import REQUIRED_FIELDS
from scripts.hexgent_real_trace_collectors.proxy import RecordingTraceSink, _forward_payload
from scripts.hexgent_real_trace_collectors import cli as collectors_cli
from scripts.hexgent_real_trace_collectors import proxy as collector_proxy


def test_pd_smoke_launchers_enable_global_prefill_admission():
    script_dir = Path("scripts/hexgent_real_trace_collectors")
    launchers = [
        script_dir / "start_hexgent_pd_smoke_2p2d.sh",
        script_dir / "start_hexgent_pd_smoke_4p4d.sh",
    ]

    for launcher in launchers:
        contents = launcher.read_text(encoding="utf-8")
        assert "--hexgent-admission-queue" in contents
        assert "--hexgent-max-prefill-queue-per-worker" in contents
        assert "--hexgent-prefill-token-headroom-percent" in contents


def test_sharegpt_arrival_staircase_uses_distinct_workflows_and_hexgent_only():
    runner = Path(
        "scripts/hexgent_real_trace_collectors/run_h100_sharegpt_rate_staircase.sh"
    )
    contents = runner.read_text(encoding="utf-8")

    assert 'SHAREGPT_WORKFLOW_COUNT="${SHAREGPT_WORKFLOW_COUNT:-128}"' in contents
    assert 'SHAREGPT_RATE_STAIRCASE="${SHAREGPT_RATE_STAIRCASE:-3,6,9,12}"' in contents
    assert "TRACE_ORDER=sharegpt" in contents
    assert "MODE_ORDER=hexgent" in contents


def test_2p2d_launcher_defaults_to_real_model_batching_and_nvlink_telemetry():
    launcher = Path(
        "scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_2p2d.sh"
    )
    contents = launcher.read_text(encoding="utf-8")

    assert 'CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"' in contents
    assert 'PREFILL_MAX_RUNNING_REQUESTS="${PREFILL_MAX_RUNNING_REQUESTS:-32}"' in contents
    assert 'DECODE_MAX_RUNNING_REQUESTS="${DECODE_MAX_RUNNING_REQUESTS:-128}"' in contents
    assert 'PREFILL_MAX_BATCH_REQUESTS="${PREFILL_MAX_BATCH_REQUESTS:-32}"' in contents
    assert 'export SGLANG_KV_TRANSFER_TELEMETRY=1' in contents
    assert 'export SGLANG_HEXGENT_CUDA_BATCH_TELEMETRY=1' in contents
    assert 'export SGLANG_MOONCAKE_TRANSFER_PROTOCOL=nvlink_intra' in contents
    assert '--prefill-max-requests "$PREFILL_MAX_BATCH_REQUESTS"' in contents
    assert 'PLANNER_TELEMETRY_PATH="${HEXGENT_PLANNER_TELEMETRY_PATH:-$LOG_DIR/planner_invocations.jsonl}"' in contents
    assert '--telemetry-path "$PLANNER_TELEMETRY_PATH"' in contents
    assert 'TRANSFER_BANDWIDTH_BYTES_PER_S="${TRANSFER_BANDWIDTH_BYTES_PER_S:-7000000000}"' in contents
    assert 'TRANSFER_STARTUP_MS="${TRANSFER_STARTUP_MS:-10}"' in contents
    assert 'export HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S="$TRANSFER_BANDWIDTH_BYTES_PER_S"' in contents
    assert 'export HEXGENT_TRANSFER_STARTUP_MS="$TRANSFER_STARTUP_MS"' in contents


def test_2p2d_launcher_allows_explicit_physical_gpu_assignment():
    launcher = Path(
        "scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_2p2d.sh"
    )
    contents = launcher.read_text(encoding="utf-8")

    assert 'PREFILL0_GPU="${PREFILL0_GPU:-0}"' in contents
    assert 'PREFILL1_GPU="${PREFILL1_GPU:-1}"' in contents
    assert 'DECODE0_GPU="${DECODE0_GPU:-2}"' in contents
    assert 'DECODE1_GPU="${DECODE1_GPU:-3}"' in contents
    assert 'start_prefill "$PREFILL0_GPU"' in contents
    assert 'start_prefill "$PREFILL1_GPU"' in contents
    assert 'start_decode "$DECODE0_GPU"' in contents
    assert 'start_decode "$DECODE1_GPU"' in contents


def test_pod_environment_preflight_checks_runtime_and_persistent_paths():
    verifier = Path(
        "scripts/hexgent_real_trace_collectors/verify_hexgent_serving_env.py"
    )
    setup = Path(
        "scripts/hexgent_real_trace_collectors/setup_hexgent_e2e_pod_env.sh"
    )
    verifier_contents = verifier.read_text(encoding="utf-8")
    setup_contents = setup.read_text(encoding="utf-8")

    assert "--output" in verifier_contents
    assert "nvidia-smi" in verifier_contents
    assert '["nvidia-smi", "topo", "-m"]' in verifier_contents
    assert "sgl_kernel" in verifier_contents
    assert "mooncake" in verifier_contents
    assert "sglang_router" in verifier_contents
    assert "HUGGINGFACE_HUB_CACHE" in verifier_contents
    assert "verify_hexgent_serving_env.py" in setup_contents
    assert "/workspace/hexgent_e2e/preflight" in setup_contents


def test_pod_setup_keeps_runtime_venv_off_the_network_volume():
    setup = Path(
        "scripts/hexgent_real_trace_collectors/setup_hexgent_e2e_pod_env.sh"
    ).read_text(encoding="utf-8")

    assert 'VENV_DIR="${HEXGENT_VENV_DIR:-/opt/hexgent_sglang_venv}"' in setup
    assert 'export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"' in setup
    assert "export HEXGENT_VENV_DIR=/opt/hexgent_sglang_venv" in setup
    assert "export CARGO_HOME=/opt/hexgent-cargo" in setup
    assert "export RUSTUP_HOME=/opt/hexgent-rustup" in setup
    assert 'export PATH="$HEXGENT_VENV_DIR/bin:/opt/hexgent-cargo/bin:$PATH"' in setup


def test_pod_setup_bootstraps_native_build_toolchain_before_router_build():
    setup = Path(
        "scripts/hexgent_real_trace_collectors/setup_hexgent_e2e_pod_env.sh"
    ).read_text(encoding="utf-8")

    for package in (
        "build-essential",
        "cmake",
        "ninja-build",
        "libaio-dev",
        "libgtest-dev",
        "libjsoncpp-dev",
        "libyaml-cpp-dev",
    ):
        assert package in setup
    assert 'LOCAL_CARGO_HOME="${HEXGENT_LOCAL_CARGO_HOME:-/opt/hexgent-cargo}"' in setup
    assert 'LOCAL_RUSTUP_HOME="${HEXGENT_LOCAL_RUSTUP_HOME:-/opt/hexgent-rustup}"' in setup
    assert 'if [[ ! -x "$LOCAL_CARGO_HOME/bin/cargo" ]]' in setup
    assert "https://sh.rustup.rs" in setup
    assert "yalantinglibs.git" in setup
    assert "0.5.7" in setup
    assert 'export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/opt/hexgent-router-target}"' in setup
    assert "export CARGO_TARGET_DIR=/opt/hexgent-router-target" in setup
    assert '/usr/local/cuda-12.4/bin' in setup
    assert "environment-manifest" in setup


def test_pod_setup_builds_rust_router_from_container_local_source():
    setup = Path(
        "scripts/hexgent_real_trace_collectors/setup_hexgent_e2e_pod_env.sh"
    ).read_text(encoding="utf-8")

    assert (
        'ROUTER_BUILD_SRC="${HEXGENT_ROUTER_BUILD_SRC:-/opt/hexgent-router-src}"'
        in setup
    )
    assert 'rm -rf "$ROUTER_BUILD_SRC"' in setup
    assert 'cp -a "$SRC_DIR/sgl-model-gateway/." "$ROUTER_BUILD_SRC/"' in setup
    assert 'ROUTER_BINDINGS="$ROUTER_BUILD_SRC/bindings/python"' in setup
    assert 'ROUTER_CARGO_HOME="${HEXGENT_ROUTER_CARGO_HOME:-$LOCAL_CARGO_HOME}"' in setup
    assert 'ROUTER_RUSTUP_HOME="${HEXGENT_ROUTER_RUSTUP_HOME:-$LOCAL_RUSTUP_HOME}"' in setup
    assert 'CARGO_HOME="$ROUTER_CARGO_HOME" RUSTUP_HOME="$ROUTER_RUSTUP_HOME"' in setup
    assert 'PATH="$ROUTER_CARGO_HOME/bin:$PATH" python -m maturin develop --release' in setup


def test_pd_launchers_use_v0515_disaggregation_arguments():
    launchers = [
        Path("scripts/hexgent_real_trace_collectors/start_hexgent_pd_1p1d.sh"),
        Path("scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke.sh"),
        Path("scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_2p2d.sh"),
        Path("scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_4p4d.sh"),
    ]

    for launcher in launchers:
        contents = launcher.read_text(encoding="utf-8")
        assert "--disaggregation-decode-tp" not in contents, launcher
        assert "--disaggregation-prefill-pp" not in contents, launcher


def test_2p2d_formal_runner_requires_model_matched_slo_and_monitors_all_workers():
    runner = Path(
        "scripts/hexgent_real_trace_collectors/run_h100_2p2d_formal_comparison.sh"
    )
    contents = runner.read_text(encoding="utf-8")

    assert 'EXCLUSIVE_PROFILE="${EXCLUSIVE_PROFILE:?set EXCLUSIVE_PROFILE' in contents
    assert 'SHAREGPT_ARRIVAL_RATE="${SHAREGPT_ARRIVAL_RATE:-3.0}"' in contents
    assert 'BFCL_ARRIVAL_RATE="${BFCL_ARRIVAL_RATE:-0.6}"' in contents
    assert 'HOTPOTQA_ARRIVAL_RATE="${HOTPOTQA_ARRIVAL_RATE:-0.2}"' in contents
    assert "for port in 31000 31001" in contents
    assert "for port in 32000 32001" in contents
    assert '"$SRC_DIR/scripts/hexgent_real_trace_collectors/summarize_scheduling_replay.py"' in contents
    assert '"$EXCLUSIVE_PROFILE"' in contents


def test_4p4d_formal_runner_requires_model_matched_slo_profile():
    runner = Path(
        "scripts/hexgent_real_trace_collectors/run_h100_formal_comparison.sh"
    )
    contents = runner.read_text(encoding="utf-8")

    assert 'EXCLUSIVE_PROFILE="${EXCLUSIVE_PROFILE:?set EXCLUSIVE_PROFILE' in contents
    assert '--exclusive-profile "$EXCLUSIVE_PROFILE"' in contents
    assert 'cp "$EXCLUSIVE_PROFILE" "$RUN_DIR/exclusive_profile.jsonl"' in contents


def test_4p4d_formal_runner_warms_each_mode_before_measurement():
    runner = Path(
        "scripts/hexgent_real_trace_collectors/run_h100_formal_comparison.sh"
    )
    contents = runner.read_text(encoding="utf-8")

    assert 'WARMUP_REQUESTS="${WARMUP_REQUESTS:-32}"' in contents
    assert 'warmup_mode "$trace" "$mode" "$router_url"' in contents
    assert contents.index('warmup_mode "$trace" "$mode" "$router_url"') < contents.index(
        'run_one "$trace" "$mode" "$router_url"'
    )


def test_4p4d_formal_runner_namespaces_each_measured_replay():
    contents = Path(
        "scripts/hexgent_real_trace_collectors/run_h100_formal_comparison.sh"
    ).read_text(encoding="utf-8")

    assert '--run-namespace "$RUN_ID::$trace::$mode"' in contents


def test_llama8b_profile_and_4p4d_runner_support_distinct_trace_paths():
    script_dir = Path("scripts/hexgent_real_trace_collectors")
    profile_runner = (script_dir / "run_llama8b_exclusive_profile.sh").read_text(
        encoding="utf-8"
    )
    comparison_runner = (script_dir / "run_h100_formal_comparison.sh").read_text(
        encoding="utf-8"
    )

    for contents in (profile_runner, comparison_runner):
        assert 'SHAREGPT_TRACE_PATH="${SHAREGPT_TRACE_PATH:-' in contents
        assert 'BFCL_TRACE_PATH="${BFCL_TRACE_PATH:-' in contents
        assert 'HOTPOTQA_TRACE_PATH="${HOTPOTQA_TRACE_PATH:-' in contents


def test_a100_exclusive_profiler_terminates_the_server_process_group():
    runner = Path(
        "scripts/hexgent_real_trace_collectors/run_a100_llama8b_exclusive_profile.sh"
    )
    contents = runner.read_text(encoding="utf-8")

    assert "setsid python -m sglang.launch_server" in contents
    assert 'kill -- -"$server_pid"' in contents
    assert 'kill -KILL -- -"$server_pid"' in contents
    assert 'profile_pid=""' in contents
    assert 'kill "$profile_pid"' in contents


def test_a100_exclusive_profiler_can_run_selected_traces_independently():
    runner = Path(
        "scripts/hexgent_real_trace_collectors/run_a100_llama8b_exclusive_profile.sh"
    )
    contents = runner.read_text(encoding="utf-8")

    assert 'TRACE_ORDER="${TRACE_ORDER:-sharegpt,bfcl,hotpotqa}"' in contents
    assert "IFS=',' read -r -a traces <<<\"$TRACE_ORDER\"" in contents
    assert 'for trace in "${traces[@]}"; do' in contents
    assert 'profile_parts+=("$RESULT_DIR/raw_${trace}.jsonl")' in contents
    assert 'cat "${profile_parts[@]}" >"$RESULT_DIR/exclusive_profile.jsonl"' in contents
    assert "profile_pid=$!" in contents
    assert 'wait "$profile_pid"' in contents


def test_a100_exclusive_profiler_matches_formal_radix_cache_configuration():
    runner = Path(
        "scripts/hexgent_real_trace_collectors/run_a100_llama8b_exclusive_profile.sh"
    )
    contents = runner.read_text(encoding="utf-8")

    assert 'DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"' in contents
    assert 'radix_cache_args=(--disable-radix-cache)' in contents
    assert '"${radix_cache_args[@]}"' in contents


def test_4p4d_intranode_nvlink_uses_cpu_safe_auxiliary_metadata_path():
    launcher = Path(
        "scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_4p4d.sh"
    )
    contents = launcher.read_text(encoding="utf-8")

    # Bulk KV is transferred through NVLink. The tiny bootstrap metadata is
    # CPU-resident and sent over ZMQ/TCP, avoiding a raw CUDA-pointer dereference.
    intranode_block = contents.split(
        'if [[ "$MOONCAKE_INTRANODE_NVLINK" == "1" ]]; then', 1
    )[1].split("else", 1)[0]
    assert "export SGLANG_MOONCAKE_SEND_AUX_TCP=1" in intranode_block
    assert "unset SGLANG_MOONCAKE_SEND_AUX_TCP" not in intranode_block
    assert "TRANSFER_BANDWIDTH_BYTES_PER_S=7000000000" in contents
    assert 'export HEXGENT_TRANSFER_STARTUP_MS="$TRANSFER_STARTUP_MS"' in contents
    assert "export HEXGENT_ENABLE_LOCAL_WAITING_MUTATIONS=0" in contents


def test_multi_worker_launcher_supports_non_symmetric_pd_worker_counts():
    launcher = Path(
        "scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_4p4d.sh"
    )
    contents = launcher.read_text(encoding="utf-8")

    assert 'PREFILL_WORKER_COUNT="${PREFILL_WORKER_COUNT:-4}"' in contents
    assert 'DECODE_WORKER_COUNT="${DECODE_WORKER_COUNT:-4}"' in contents
    assert 'PREFILL_GPU_OFFSET="${PREFILL_GPU_OFFSET:-0}"' in contents
    assert 'DECODE_GPU_OFFSET="${DECODE_GPU_OFFSET:-$PREFILL_WORKER_COUNT}"' in contents
    assert 'seq 0 "$((PREFILL_WORKER_COUNT - 1))"' in contents
    assert 'seq 0 "$((DECODE_WORKER_COUNT - 1))"' in contents
    assert 'local physical_gpu=$((index + PREFILL_GPU_OFFSET))' in contents
    assert 'local gpu=$((index + DECODE_GPU_OFFSET))' in contents


def test_formal_runner_monitors_configured_non_symmetric_pd_worker_counts():
    runner = Path(
        "scripts/hexgent_real_trace_collectors/run_h100_formal_comparison.sh"
    )
    contents = runner.read_text(encoding="utf-8")

    assert 'PREFILL_WORKER_COUNT="${PREFILL_WORKER_COUNT:-4}"' in contents
    assert 'DECODE_WORKER_COUNT="${DECODE_WORKER_COUNT:-4}"' in contents
    assert 'PREFILL_PORT_BASE="${PREFILL_PORT_BASE:-31000}"' in contents
    assert 'DECODE_PORT_BASE="${DECODE_PORT_BASE:-32000}"' in contents
    assert (
        'python - "$output" "$PREFILL_WORKER_COUNT" "$DECODE_WORKER_COUNT" '
        '"$PREFILL_PORT_BASE" "$DECODE_PORT_BASE"'
    ) in contents
    assert (
        'prefill_count, decode_count, prefill_base, decode_base = '
        'map(int, sys.argv[2:])'
    ) in contents
    assert '("prefill", prefill_count, prefill_base)' in contents
    assert '("decode", decode_count, decode_base)' in contents
    assert 'for index in range(count)' in contents


def test_multi_worker_launcher_stops_workers_from_previous_topology():
    launcher = Path(
        "scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_4p4d.sh"
    )
    contents = launcher.read_text(encoding="utf-8")

    assert 'PID_DIR="${HEXGENT_PID_DIR:-/workspace/hexgent_e2e/active_service_pids}"' in contents
    assert 'for pid_file in "$PID_DIR"/*.pid; do' in contents
    assert 'stop_pid_file "$pid_file"' in contents
    assert "prefill0 prefill1 prefill2 prefill3" not in contents
    assert 'echo $! >"$PID_DIR/planner.pid"' in contents
    assert 'echo $! >"$PID_DIR/router.pid"' in contents


def test_4p4d_launcher_allows_an_explicit_nonnvlink_mooncake_protocol():
    launcher = Path(
        "scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_4p4d.sh"
    )
    contents = launcher.read_text(encoding="utf-8")

    assert 'MOONCAKE_TRANSFER_PROTOCOL="${MOONCAKE_TRANSFER_PROTOCOL:-}"' in contents
    assert (
        'export SGLANG_MOONCAKE_TRANSFER_PROTOCOL="$MOONCAKE_TRANSFER_PROTOCOL"'
        in contents
    )


def test_4p4d_global_admission_uses_native_pd_fallback_until_a_plan_is_published():
    launcher = Path(
        "scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_4p4d.sh"
    )
    contents = launcher.read_text(encoding="utf-8")

    assert 'HEXGENT_FALLBACK_POLICY="${HEXGENT_FALLBACK_POLICY:-pd_policy}"' in contents
    assert '--hexgent-fallback-policy "$HEXGENT_FALLBACK_POLICY"' in contents


def test_4p4d_launcher_enables_worker_priority_scheduling_for_hexgent():
    launcher = Path(
        "scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_4p4d.sh"
    )
    contents = launcher.read_text(encoding="utf-8")

    # The router injects stage-specific HexAgenT priorities. Both the prefill
    # and decode schedulers must opt in or SGLang falls back to FCFS locally.
    assert contents.count("--enable-priority-scheduling") == 2
    # HexAgenT rank zero is the most urgent call, whereas SGLang's default
    # priority convention treats larger integers as more urgent.
    assert contents.count("--schedule-low-priority-values-first") == 2


def test_mock_admission_integration_uses_discoverable_workers_and_cleans_up():
    script_dir = Path("scripts/hexgent_real_trace_collectors")
    worker = (script_dir / "mock_hexgent_admission_workers.py").read_text(
        encoding="utf-8"
    )
    integration = (script_dir / "run_mock_hexgent_admission_integration.sh").read_text(
        encoding="utf-8"
    )
    sidecar_runner = (script_dir / "run_hexgent_sidecar_overlay.py").read_text(
        encoding="utf-8"
    )

    assert 'app.router.add_get("/server_info"' in worker
    assert 'app.router.add_get("/model_info"' in worker
    assert '"model_id": MODEL_ID' in worker
    assert '"disaggregation_mode": stage' in worker
    assert "trap cleanup EXIT" in integration
    assert 'wait "$pid"' in integration
    assert "run_hexgent_sidecar_overlay.py" in integration
    assert 'curl -fsS http://127.0.0.1:30999/health' in integration
    assert 'runpy.run_module("sglang.hexgent.sidecar"' in sidecar_runner


def test_parse_sharegpt_preserves_multiturn_workflow():
    records = [
        {
            "id": "sg_1",
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi"},
                {"from": "human", "value": "Tell me a joke"},
                {"from": "gpt", "value": "A short joke."},
            ],
        }
    ]

    workflows = parse_sharegpt_records(records)

    assert len(workflows) == 1
    assert workflows[0]["dataset"] == "sharegpt"
    assert workflows[0]["workflow_id"] == "sg_1"
    assert len(workflows[0]["calls"]) == 2
    assert workflows[0]["calls"][0]["messages"] == [
        {"role": "user", "content": "Hello"},
    ]
    assert workflows[0]["calls"][1]["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Tell me a joke"},
    ]
    assert workflows[0]["calls"][0]["metadata"]["expected_response"] == "Hi"
    assert workflows[0]["calls"][0]["max_tokens"] == workflows[0]["calls"][0]["min_tokens"]
    assert workflows[0]["calls"][0]["ignore_eos"] is True


def test_parse_sharegpt_uses_token_counter_for_expected_output_length():
    records = [
        {
            "id": "sg_tok",
            "conversations": [
                {"from": "human", "value": "Prompt"},
                {"from": "gpt", "value": "one two three"},
            ],
        }
    ]

    workflows = parse_sharegpt_records(
        records,
        output_token_counter=lambda text: 7 if text == "one two three" else 0,
    )

    call = workflows[0]["calls"][0]
    assert call["max_tokens"] == 7
    assert call["min_tokens"] == 7
    assert call["metadata"]["expected_output_tokens"] == 7


def test_parse_sharegpt_can_filter_single_call_workflows():
    records = [
        {
            "id": "single",
            "conversations": [
                {"from": "human", "value": "One prompt"},
                {"from": "gpt", "value": "One answer"},
            ],
        },
        {
            "id": "multi",
            "conversations": [
                {"from": "human", "value": "First"},
                {"from": "gpt", "value": "Answer 1"},
                {"from": "human", "value": "Second"},
                {"from": "gpt", "value": "Answer 2"},
            ],
        },
    ]

    workflows = parse_sharegpt_records(records, min_calls_per_workflow=2)

    assert [workflow["workflow_id"] for workflow in workflows] == ["multi"]
    assert len(workflows[0]["calls"]) == 2


def test_sharegpt_replay_uses_dataset_assistant_response_and_fixed_generation_length():
    requests = []

    def fake_stream_json(path, payload, headers):
        requests.append(payload)
        assert payload["max_tokens"] == 3
        assert payload["min_tokens"] == 3
        assert payload["ignore_eos"] is True
        yield 1010.0, {"choices": [{"delta": {"content": "generated"}}]}
        yield 1040.0, {"usage": {"prompt_tokens": 8, "completion_tokens": 3}, "choices": [{"delta": {}}]}

    client = OpenAIChatClient(
        endpoint="http://fake-server/v1",
        model="fake-model",
        stream_json=fake_stream_json,
        clock_ms=lambda: 1000.0,
    )
    workflows = [
        {
            "dataset": "sharegpt",
            "workflow_id": "sg",
            "calls": [
                {
                    "call_id": "sg_turn_0",
                    "turn_index": 0,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 3,
                    "min_tokens": 3,
                    "ignore_eos": True,
                    "metadata": {
                        "expected_response": "dataset answer",
                        "expected_output_tokens": 3,
                    },
                }
            ],
        }
    ]

    rows = list(
        replay_workflows(
            workflows,
            client=client,
            model="fake-model",
            endpoint="http://fake-server/v1",
            stream=True,
        )
    )

    assert rows[0]["response_text"] == "dataset answer"
    assert rows[0]["output_tokens"] == 3
    assert rows[0]["ttft_ms"] == 10.0
    assert rows[0]["total_latency_ms"] == 40.0
    assert rows[0]["metadata"]["expected_response"] == "dataset answer"
    assert requests[0]["messages"] == [{"role": "user", "content": "Hi"}]


def test_phase_profiles_accept_single_token_rows_with_missing_ttft():
    row = {
        "workflow_id": "wf",
        "call_id": "call",
        "dataset": "sharegpt",
        "input_tokens": 10,
        "output_tokens": 1,
        "ttft_ms": None,
        "total_latency_ms": 123.0,
        "error": None,
    }

    normalized = phase_profiles._normalize_for_profile(row)

    assert phase_profiles._valid(normalized)
    assert normalized["ttft_ms"] == 123.0


def test_load_sharegpt_records_downloads_explicit_hf_json_when_dataset_auto_discovery_fails(tmp_path, monkeypatch):
    downloaded = tmp_path / "sharegpt.json"
    downloaded.write_text(
        json.dumps(
            [
                {
                    "id": "sg_hf",
                    "conversations": [
                        {"from": "human", "value": "Hi"},
                        {"from": "gpt", "value": "Hello"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    fake_hub = types.SimpleNamespace(
        hf_hub_download=lambda **kwargs: str(downloaded),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    records = list(collectors_cli._load_sharegpt_records(None))

    assert records[0]["id"] == "sg_hf"


def test_parse_bfcl_preserves_tool_question_and_tools():
    records = [
        {
            "id": "bfcl_1",
            "question": "Call the weather API for Boston.",
            "function": [
                {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
        }
    ]

    workflows = parse_bfcl_records(records)

    call = workflows[0]["calls"][0]
    assert workflows[0]["dataset"] == "bfcl-v3"
    assert call["prompt"] == "Call the weather API for Boston."
    assert call["tools"][0]["function"]["name"] == "get_weather"


def test_parse_bfcl_native_question_conversations_as_workflow_calls():
    records = [
        {
            "id": "multi_turn_0",
            "question": [
                [{"role": "user", "content": "Create a calendar event."}],
                [
                    {"role": "user", "content": "Create a calendar event."},
                    {"role": "assistant", "content": "Which date?"},
                    {"role": "user", "content": "Tomorrow at noon."},
                ],
            ],
            "function": [{"name": "create_event", "parameters": {"type": "object"}}],
        }
    ]

    workflows = parse_bfcl_records(records)

    assert len(workflows[0]["calls"]) == 2
    assert workflows[0]["calls"][0]["messages"] == [{"role": "user", "content": "Create a calendar event."}]
    assert workflows[0]["calls"][1]["messages"][-1] == {"role": "user", "content": "Tomorrow at noon."}
    assert workflows[0]["calls"][1]["tools"][0]["function"]["name"] == "create_event"


def test_parse_lats_hotpotqa_creates_agent_style_prompt_with_context():
    records = [
        {
            "id": "hp_1",
            "question": "Who founded the company that made the iPod?",
            "context": [["Apple Inc.", ["Apple made the iPod.", "Apple was founded by Steve Jobs."]]],
            "answer": "Steve Jobs",
        }
    ]

    workflows = parse_lats_hotpotqa_records(records)

    call = workflows[0]["calls"][0]
    assert workflows[0]["dataset"] == "lats-hotpotqa"
    assert "Thought:" in call["prompt"]
    assert "Apple made the iPod." in call["prompt"]
    assert call["metadata"]["answer"] == "Steve Jobs"


def test_fake_endpoint_replay_emits_required_schema_and_usage():
    requests = []

    def fake_post_json(path, payload, headers):
        requests.append((path, payload, headers))
        return {
            "choices": [{"message": {"content": "fake response"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    client = OpenAIChatClient(
        endpoint="http://fake-server/v1",
        api_key="token",
        model="fake-model",
        post_json=fake_post_json,
        clock_ms=lambda: 1000.0,
    )
    workflows = [
        {
            "dataset": "bfcl-v3",
            "workflow_id": "bfcl_1",
            "calls": [{"call_id": "bfcl_1_turn_0", "turn_index": 0, "messages": [{"role": "user", "content": "Hi"}]}],
        }
    ]

    rows = list(
        replay_workflows(
            workflows,
            client=client,
            model="fake-model",
            endpoint="http://fake-server/v1",
            max_tokens=128,
        )
    )

    assert requests[0][0] == "/chat/completions"
    assert requests[0][1]["messages"] == [{"role": "user", "content": "Hi"}]
    assert requests[0][1]["max_tokens"] == 128
    assert set(REQUIRED_FIELDS).issubset(rows[0])
    assert rows[0]["response_text"] == "fake response"
    assert rows[0]["input_tokens"] == 7
    assert rows[0]["output_tokens"] == 3
    assert rows[0]["total_latency_ms"] == 0.0
    json.dumps(rows[0])


def test_stream_replay_records_ttft_from_first_delta():
    events = [
        (1005.0, {"choices": [{"delta": {"content": "hel"}}]}),
        (1017.0, {"choices": [{"delta": {"content": "lo"}}]}),
        (1020.0, {"usage": {"prompt_tokens": 2, "completion_tokens": 2}, "choices": [{"delta": {}}]}),
    ]

    def fake_stream_json(path, payload, headers):
        assert payload["stream"] is True
        yield from events

    client = OpenAIChatClient(
        endpoint="http://fake-server/v1",
        model="fake-model",
        stream_json=fake_stream_json,
        clock_ms=lambda: 1000.0,
    )
    workflows = [
        {
            "dataset": "bfcl-v3",
            "workflow_id": "bfcl_1",
            "calls": [{"call_id": "bfcl_1_turn_0", "turn_index": 0, "prompt": "Use a tool"}],
        }
    ]

    rows = list(replay_workflows(workflows, client=client, model="fake-model", endpoint="http://fake-server/v1", stream=True))

    assert rows[0]["response_text"] == "hello"
    assert rows[0]["ttft_ms"] == 5.0
    assert rows[0]["total_latency_ms"] == 20.0
    assert rows[0]["input_tokens"] == 2
    assert rows[0]["output_tokens"] == 2


def test_stream_replay_records_qwen_reasoning_delta_as_response_and_ttft():
    events = [
        (1007.0, {"choices": [{"delta": {"reasoning_content": "reason"}}]}),
        (1011.0, {"choices": [{"delta": {"content": " answer"}}]}),
        (1018.0, {"usage": {"prompt_tokens": 4, "completion_tokens": 3}, "choices": [{"delta": {}}]}),
    ]

    def fake_stream_json(path, payload, headers):
        assert payload["stream"] is True
        yield from events

    client = OpenAIChatClient(
        endpoint="http://fake-server/v1",
        model="fake-model",
        stream_json=fake_stream_json,
        clock_ms=lambda: 1000.0,
    )
    workflows = [
        {
            "dataset": "sharegpt",
            "workflow_id": "sharegpt_1",
            "calls": [{"call_id": "sharegpt_1_turn_0", "turn_index": 0, "messages": [{"role": "user", "content": "Hi"}]}],
        }
    ]

    rows = list(replay_workflows(workflows, client=client, model="fake-model", endpoint="http://fake-server/v1", stream=True))

    assert rows[0]["response_text"] == "reason answer"
    assert rows[0]["ttft_ms"] == 7.0
    assert rows[0]["total_latency_ms"] == 18.0
    assert rows[0]["input_tokens"] == 4
    assert rows[0]["output_tokens"] == 3


def test_replay_without_endpoint_collects_request_only_rows():
    workflows = parse_bfcl_records([{"id": "bfcl_1", "prompt": "Return a function call."}])

    rows = list(replay_workflows(workflows, client=None, model=None, endpoint=None))

    assert rows[0]["dataset"] == "bfcl-v3"
    assert rows[0]["prompt"] == "Return a function call."
    assert rows[0]["response_text"] is None
    assert rows[0]["error"] is None


def test_recording_proxy_sink_writes_forwarded_chat_completion_rows(tmp_path):
    output = tmp_path / "proxy_trace.jsonl"
    sink = RecordingTraceSink(
        output_path=output,
        dataset="bfcl-v3",
        endpoint="http://127.0.0.1:30000/v1",
    )
    sink.record_chat_completion(
        request_payload={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Call the tool."}],
            "metadata": {"hexgent": {"workflow_id": "wf", "call_id": "call-0"}},
        },
        response_payload={
            "choices": [{"message": {"content": '{"name":"tool"}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        },
        start_ms=100.0,
        first_token_ms=None,
        end_ms=160.0,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["dataset"] == "bfcl-v3"
    assert row["workflow_id"] == "wf"
    assert row["call_id"] == "call-0"
    assert row["input_tokens"] == 12
    assert row["output_tokens"] == 5
    assert row["response_text"] == '{"name":"tool"}'
    assert row["total_latency_ms"] == 60.0


def test_recording_proxy_does_not_fabricate_ttft_for_empty_stream_completion(tmp_path):
    output = tmp_path / "proxy_trace.jsonl"
    sink = RecordingTraceSink(
        output_path=output,
        dataset="lats-hotpotqa",
        endpoint="http://127.0.0.1:30000/v1",
    )
    sink.record_chat_completion(
        request_payload={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Reflect briefly."}],
            "metadata": {"hexgent": {"workflow_id": "wf", "call_id": "call-0"}},
        },
        response_payload={
            "choices": [{"message": {"content": ""}}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 2},
        },
        start_ms=100.0,
        first_token_ms=None,
        end_ms=154.0,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["response_text"] == ""
    assert row["ttft_ms"] is None
    assert row["total_latency_ms"] == 54.0


def test_recording_proxy_extracts_structured_reasoning_and_tool_call_text():
    response = {
        "choices": [
            {
                "message": {
                    "reasoning_content": [{"type": "text", "text": "think"}],
                    "content": [{"type": "text", "text": " answer"}],
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"query\":\"Arthur\"}",
                            }
                        }
                    ],
                }
            }
        ]
    }
    stream_choice = {
        "delta": {
            "reasoning_content": [{"type": "text", "text": "partial"}],
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": "{\"q\":\"Oberoi\"}",
                    }
                }
            ],
        }
    }

    assert collector_proxy._extract_response_text(response) == 'think answerlookup{"query":"Arthur"}'
    assert collector_proxy._extract_choice_delta_text(stream_choice) == 'partialsearch{"q":"Oberoi"}'


def test_recording_proxy_splits_batched_lats_choices_into_logical_rows(tmp_path):
    output = tmp_path / "proxy_trace.jsonl"
    sink = RecordingTraceSink(
        output_path=output,
        dataset="lats-hotpotqa",
        endpoint="http://127.0.0.1:30000/v1",
    )
    sink.record_chat_completion(
        request_payload={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Question: Q? Thought 1:"}],
            "metadata": {
                "hexgent": {
                    "workflow_id": "hotpotqa_0",
                    "call_id": "hotpotqa_0-call-10",
                    "turn_index": 10,
                    "fanout_group_id": "hotpotqa_0-fanout-10",
                    "fanout_width": 2,
                    "fanout_index": 0,
                    "logical_call_count": 2,
                    "split_batched_choices": True,
                }
            },
        },
        response_payload={
            "choices": [
                {"index": 0, "message": {"content": "short"}},
                {"index": 1, "message": {"content": "a much longer trajectory"}},
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 9},
        },
        start_ms=100.0,
        first_token_ms=120.0,
        end_ms=200.0,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["call_id"] for row in rows] == ["hotpotqa_0-call-10", "hotpotqa_0-call-11"]
    assert [row["metadata"]["fanout_index"] for row in rows] == [0, 1]
    assert [row["response_text"] for row in rows] == ["short", "a much longer trajectory"]
    assert {row["ttft_ms"] for row in rows} == {20.0}
    assert rows[0]["total_latency_ms"] < rows[1]["total_latency_ms"]
    assert rows[1]["total_latency_ms"] == 100.0


def test_recording_proxy_sink_writes_forwarded_completion_rows(tmp_path):
    output = tmp_path / "proxy_trace.jsonl"
    sink = RecordingTraceSink(
        output_path=output,
        dataset="bfcl-v3",
        endpoint="http://127.0.0.1:30000/v1",
    )
    sink.record_completion(
        request_payload={
            "model": "test-model",
            "prompt": "Return a function call.",
            "metadata": {"hexgent": {"workflow_id": "wf", "call_id": "call-0"}},
        },
        response_payload={
            "choices": [{"text": '{"name":"tool"}'}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        },
        start_ms=100.0,
        first_token_ms=125.0,
        end_ms=160.0,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["dataset"] == "bfcl-v3"
    assert row["workflow_id"] == "wf"
    assert row["call_id"] == "call-0"
    assert row["prompt"] == "Return a function call."
    assert row["input_tokens"] == 12
    assert row["output_tokens"] == 5
    assert row["ttft_ms"] == 25.0
    assert row["response_text"] == '{"name":"tool"}'
    assert row["total_latency_ms"] == 60.0


def test_recording_proxy_model_override_only_changes_forward_payload():
    original = {
        "model": "official-eval-model",
        "messages": [{"role": "user", "content": "hi"}],
    }

    forwarded = _forward_payload(original, model_override="served-local-model")

    assert original["model"] == "official-eval-model"
    assert forwarded["model"] == "served-local-model"
    assert forwarded["messages"] == original["messages"]


def test_build_phase_profiles_keeps_valid_rows_and_splits_prefill_decode(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [
        {
            "dataset": "sharegpt",
            "workflow_id": "wf",
            "call_id": "call-0",
            "input_tokens": 10,
            "output_tokens": 4,
            "ttft_ms": 20.0,
            "total_latency_ms": 50.0,
            "error": None,
        },
        {
            "dataset": "sharegpt",
            "workflow_id": "bad",
            "call_id": "bad-0",
            "input_tokens": 1,
            "output_tokens": 1,
            "ttft_ms": None,
            "total_latency_ms": 5.0,
            "error": None,
        },
    ]
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = build_phase_profiles(
        profile_dir=tmp_path,
        model="served-model",
        worker_class="H100_TP8",
        sources=[("sharegpt", raw)],
        output_tag="qwen_h100_tp8",
    )

    assert summary["sharegpt"]["raw_rows"] == 2
    assert summary["sharegpt"]["clean_rows"] == 2
    prefill = json.loads(
        (tmp_path / "prefill_sharegpt_qwen_h100_tp8.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    decode = json.loads(
        (tmp_path / "decode_sharegpt_qwen_h100_tp8.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert prefill["prefill_time_ms"] == 20.0
    assert decode["decode_time_ms"] == 30.0
    assert decode["decode_time_per_token_ms"] == 10.0
    assert decode["model"] == "served-model"
