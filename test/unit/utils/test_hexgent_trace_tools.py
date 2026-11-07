import importlib.util
import asyncio
from pathlib import Path
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "python" / "sglang" / "hexgent" / "workflow_trace.py"
_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_module():
    spec = importlib.util.spec_from_file_location("hexgent_workflow_trace", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    def encode(self, text):
        return [tok for tok in str(text).split() if tok]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
    ):
        text = " ".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            text += " assistant:"
        if tokenize:
            return self.encode(text)
        return text


def test_convert_sharegpt_records_builds_sequential_workflow_calls():
    module = _load_module()
    records = [
        {
            "id": "conv-1",
            "conversations": [
                {"from": "human", "value": "Hello there"},
                {"from": "gpt", "value": "General Kenobi"},
                {"from": "human", "value": "How are you"},
                {"from": "gpt", "value": "I am fine"},
            ],
        }
    ]

    workflows = module.convert_sharegpt_records(
        records,
        tokenizer=_FakeTokenizer(),
        model="test-model",
    )

    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow.workflow_id == "conv-1"
    assert len(workflow.calls) == 2
    assert workflow.calls[0].parents == []
    assert workflow.calls[1].parents == [workflow.calls[0].call_id]
    assert workflow.calls[0].request["messages"][0]["role"] == "user"
    assert workflow.calls[0].output_len == 2
    assert workflow.calls[1].input_len > workflow.calls[0].input_len


def test_convert_openai_request_records_groups_by_workflow_id():
    module = _load_module()
    records = [
        {
            "conversation_id": "wf-1",
            "messages": [{"role": "user", "content": "First"}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
        {
            "conversation_id": "wf-1",
            "messages": [{"role": "user", "content": "Second"}],
            "max_tokens": 8,
        },
    ]

    workflows = module.convert_openai_request_records(records, tokenizer=_FakeTokenizer())

    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow.workflow_id == "wf-1"
    assert [call.output_len for call in workflow.calls] == [16, 8]
    assert workflow.calls[1].parents == [workflow.calls[0].call_id]


def test_convert_call_jsonl_records_keeps_explicit_parents():
    module = _load_module()
    records = [
        {
            "workflow_id": "wf-explicit",
            "call_id": "root",
            "parents": [],
            "request": {"messages": [{"role": "user", "content": "root"}]},
            "input_len": 5,
            "output_len": 7,
        },
        {
            "workflow_id": "wf-explicit",
            "call_id": "child",
            "parents": ["root"],
            "request": {"messages": [{"role": "user", "content": "child"}]},
            "input_len": 6,
            "output_len": 9,
        },
    ]

    workflows = module.convert_call_jsonl_records(records)

    assert len(workflows) == 1
    assert workflows[0].calls[1].parents == ["root"]


def test_service_profile_table_uses_exact_then_nearest_lookup():
    module = _load_module()
    table = module.ServiceProfileTable.from_rows(
        [
            {
                "model": "m",
                "worker_class": "A100",
                "input_len": 128,
                "output_len": 32,
                "estimated_prefill_time": 1.25,
                "estimated_decode_time": 3.5,
                "estimated_transfer_time": 0.2,
            },
            {
                "model": "m",
                "worker_class": "A100",
                "input_len": 256,
                "output_len": 64,
                "estimated_prefill_time": 2.5,
                "estimated_decode_time": 7.0,
                "estimated_transfer_time": 0.4,
            },
        ]
    )

    exact = table.lookup("m", "A100", 128, 32)
    nearest = table.lookup("m", "A100", 130, 40)

    assert exact["estimated_prefill_time"] == pytest.approx(1.25)
    assert nearest["estimated_prefill_time"] == pytest.approx(1.25)
    assert nearest["estimated_decode_time"] == pytest.approx(3.5)


def test_build_chat_completion_payload_forces_trace_length_and_hexgent_hints():
    module = _load_module()
    payload = module.build_chat_completion_payload(
        workflow_id="wf-1",
        call_id="call-1",
        request={
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
        },
        output_len=24,
        input_len=7,
        workflow_arrival_ms=1234,
        workflow_horizon_ms=5000,
        planner_hints={
            "estimated_prefill_time": 1.0,
            "estimated_decode_time": 2.0,
            "estimated_transfer_time": 0.1,
        },
        exact_output_length=True,
    )

    assert payload["conversation_id"] == "wf-1"
    assert payload["hexgent_call_id"] == "call-1"
    assert payload["max_tokens"] == 24
    assert payload["ignore_eos"] is True
    assert payload["temperature"] == 0.7
    assert payload["hexgent"]["workflow_id"] == "wf-1"
    assert payload["hexgent"]["call_id"] == "call-1"
    assert payload["hexgent"]["workflow_arrival_ms"] == 1234
    assert payload["hexgent"]["workflow_horizon_ms"] == 5000
    assert payload["hexgent"]["input_tokens"] == 7
    assert payload["hexgent"]["predicted_output_tokens"] == 24
    assert payload["hexgent"]["estimated_decode_time"] == pytest.approx(2.0)
    assert payload["hexgent_estimated_prefill_time"] == pytest.approx(1.0)
    assert payload["hexgent_estimated_decode_time"] == pytest.approx(2.0)
    assert payload["hexgent_estimated_transfer_time"] == pytest.approx(0.1)


def test_build_chat_completion_payload_omits_default_horizon_and_carries_revealed_parents():
    module = _load_module()

    payload = module.build_chat_completion_payload(
        workflow_id="wf-1",
        call_id="child-1",
        request={"messages": [{"role": "user", "content": "follow up"}]},
        output_len=24,
        input_len=7,
        workflow_arrival_ms=1234,
        parent_call_ids=["root-1"],
    )

    assert payload["hexgent"]["parent_call_ids"] == ["root-1"]
    assert "workflow_horizon_ms" not in payload["hexgent"]


def test_generate_poisson_arrival_offsets_is_monotonic():
    module = _load_module()
    offsets = module.generate_poisson_arrival_offsets(num_workflows=4, rate=2.0, seed=7)

    assert len(offsets) == 4
    assert offsets == sorted(offsets)
    assert offsets[0] >= 0.0


def test_dump_and_load_workflow_traces_round_trip(tmp_path: Path):
    module = _load_module()
    workflows = module.convert_call_jsonl_records(
        [
            {
                "workflow_id": "wf-1",
                "call_id": "call-1",
                "parents": [],
                "request": {"messages": [{"role": "user", "content": "hello"}]},
                "input_len": 4,
                "output_len": 6,
            }
        ]
    )
    out_path = tmp_path / "trace.jsonl"

    module.dump_workflow_traces_jsonl(workflows, out_path)
    loaded = module.load_workflow_traces_jsonl(out_path)

    assert len(loaded) == 1
    assert loaded[0].workflow_id == "wf-1"
    assert loaded[0].calls[0].call_id == "call-1"


def test_convert_profile_rows_recovers_fanout_barrier_dependencies():
    module = _load_module()
    workflows = module.convert_profile_rows_to_workflows(
        [
            {
                "workflow_id": "wf-lats",
                "call_id": "root-0",
                "messages": [{"role": "user", "content": "q"}],
                "input_tokens": 10,
                "output_tokens": 20,
                "total_latency_ms": 123.5,
                "metadata": {"fanout_group_id": "g0", "fanout_index": 0, "fanout_width": 2},
            },
            {
                "workflow_id": "wf-lats",
                "call_id": "root-1",
                "messages": [{"role": "user", "content": "q"}],
                "input_tokens": 10,
                "output_tokens": 22,
                "metadata": {"fanout_group_id": "g0", "fanout_index": 1, "fanout_width": 2},
            },
            {
                "workflow_id": "wf-lats",
                "call_id": "eval",
                "messages": [{"role": "user", "content": "eval"}],
                "input_tokens": 12,
                "output_tokens": 8,
                "metadata": {},
            },
            {
                "workflow_id": "wf-lats",
                "call_id": "child-0",
                "messages": [{"role": "user", "content": "next"}],
                "input_tokens": 13,
                "output_tokens": 16,
                "metadata": {
                    "fanout_group_id": "g1",
                    "fanout_index": 0,
                    "fanout_width": 2,
                    "parent_call_id": "eval",
                },
            },
            {
                "workflow_id": "wf-lats",
                "call_id": "child-1",
                "messages": [{"role": "user", "content": "next"}],
                "input_tokens": 13,
                "output_tokens": 18,
                "metadata": {
                    "fanout_group_id": "g1",
                    "fanout_index": 1,
                    "fanout_width": 2,
                    "parent_call_id": "eval",
                },
            },
        ],
        model="hexgent-smoke-model",
        max_output_len=12,
    )

    calls = {call.call_id: call for call in workflows[0].calls}
    assert calls["root-0"].parents == []
    assert calls["root-1"].parents == []
    assert calls["eval"].parents == ["root-0", "root-1"]
    assert calls["child-0"].parents == ["eval"]
    assert calls["child-1"].parents == ["eval"]
    assert calls["root-0"].output_len == 12
    assert calls["root-0"].planner_hints["exclusive_profile_latency_ms"] == 123.5
    assert calls["eval"].output_len == 8
    assert calls["root-0"].request["model"] == "hexgent-smoke-model"


def test_profile_jsonl_cli_limits_complete_workflows(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "profile.jsonl"
    destination = tmp_path / "workflows.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"workflow_id":"wf-0","call_id":"c0","messages":[{"role":"user","content":"a"}],"input_tokens":1,"output_tokens":2}',
                '{"workflow_id":"wf-1","call_id":"c1","messages":[{"role":"user","content":"b"}],"input_tokens":1,"output_tokens":2}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert module.main(
        [
            "convert",
            "--input-format",
            "profile_jsonl",
            "--input-path",
            str(source),
            "--output-path",
            str(destination),
            "--max-workflows",
            "1",
        ]
    ) == 0

    assert [workflow.workflow_id for workflow in module.load_workflow_traces_jsonl(destination)] == ["wf-0"]


def test_convert_profile_rows_skips_workflows_with_empty_message_content():
    module = _load_module()
    workflows = module.convert_profile_rows_to_workflows(
        [
            {
                "workflow_id": "valid-before",
                "call_id": "call-0",
                "messages": [{"role": "user", "content": "valid"}],
                "input_tokens": 1,
                "output_tokens": 2,
            },
            {
                "workflow_id": "invalid",
                "call_id": "call-1",
                "messages": [{"role": "user", "content": ""}],
                "input_tokens": 1,
                "output_tokens": 2,
            },
            {
                "workflow_id": "invalid",
                "call_id": "call-2",
                "messages": [{"role": "user", "content": "later"}],
                "input_tokens": 1,
                "output_tokens": 2,
            },
            {
                "workflow_id": "valid-after",
                "call_id": "call-3",
                "messages": [{"role": "user", "content": "also valid"}],
                "input_tokens": 1,
                "output_tokens": 2,
            },
        ]
    )

    assert [workflow.workflow_id for workflow in workflows] == [
        "valid-before",
        "valid-after",
    ]


def test_replayer_records_call_error_without_cancelling_workflow(monkeypatch):
    module = _load_module()
    results = []

    async def fail_post_json(*args, **kwargs):
        raise TimeoutError("synthetic timeout")

    monkeypatch.setattr(module, "_post_json", fail_post_json)
    replayer = module.WorkflowTraceReplayer(
        router_url="http://router",
        model="test-model",
        timeout_s=1,
    )
    call = module.WorkflowCall(
        call_id="call-timeout",
        parents=[],
        request={"messages": [{"role": "user", "content": "hello"}]},
        input_len=3,
        output_len=5,
    )

    returned = asyncio.run(
        replayer._run_call(
            None,
            "wf-timeout",
            call,
            results,
            workflow_arrival_ms=123,
            workflow_horizon_ms=456,
        )
    )

    assert returned == "call-timeout"
    assert len(results) == 1
    assert results[0]["workflow_id"] == "wf-timeout"
    assert results[0]["call_id"] == "call-timeout"
    assert "synthetic timeout" in results[0]["error"]
    assert results[0]["response"] is None


def test_replayer_model_override_replaces_trace_model(monkeypatch):
    module = _load_module()
    captured = {}

    async def capture_post_json(session, url, payload, timeout_s):
        captured["payload"] = payload
        return ({"choices": [{"message": {"content": "ok"}}]}, {})

    monkeypatch.setattr(module, "_post_json", capture_post_json)
    replayer = module.WorkflowTraceReplayer(
        router_url="http://router",
        model="served-model",
    )
    call = module.WorkflowCall(
        call_id="call-model-override",
        parents=[],
        request={
            "model": "trace-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
        input_len=3,
        output_len=5,
    )

    asyncio.run(
        replayer._run_call(
            None,
            "wf-model-override",
            call,
            [],
            workflow_arrival_ms=123,
            workflow_horizon_ms=456,
        )
    )

    assert captured["payload"]["model"] == "served-model"
    assert "workflow_horizon_ms" not in captured["payload"]["hexgent"]


def test_replayer_namespaces_runtime_ids_without_changing_result_ids(monkeypatch):
    module = _load_module()
    captured = {}
    results = []

    async def capture_post_json(session, url, payload, timeout_s):
        captured["payload"] = payload
        return ({"choices": [{"message": {"content": "ok"}}]}, {})

    monkeypatch.setattr(module, "_post_json", capture_post_json)
    replayer = module.WorkflowTraceReplayer(
        router_url="http://router",
        model="served-model",
        run_namespace="run-a",
    )
    call = module.WorkflowCall(
        call_id="call-1",
        parents=["parent-0"],
        request={"messages": [{"role": "user", "content": "hello"}]},
        input_len=3,
        output_len=5,
    )

    asyncio.run(
        replayer._run_call(
            None,
            "wf-1",
            call,
            results,
            workflow_arrival_ms=123,
            workflow_horizon_ms=None,
        )
    )

    metadata = captured["payload"]["hexgent"]
    assert metadata["workflow_id"] == "run-a::wf-1"
    assert metadata["call_id"] == "run-a::call-1"
    assert metadata["parent_call_ids"] == ["run-a::parent-0"]
    assert results[0]["workflow_id"] == "wf-1"
    assert results[0]["call_id"] == "call-1"


def test_replayer_records_latency_route_and_schedule_metadata(monkeypatch):
    module = _load_module()
    results = []

    async def fake_post_json(*args, **kwargs):
        return (
            {"choices": [{"message": {"content": "ok"}}]},
            {
                "x-sglang-prefill-worker": "http://prefill-2",
                "x-sglang-decode-worker": "http://decode-1",
                "x-hexgent-router-mode": "hexgent",
                "x-hexgent-schedule-revision": "17",
                "x-hexgent-selection-wall-us": "2500",
                "x-hexgent-snapshot-fetch-us": "800",
                "x-hexgent-admission-wait-us": "1600",
            },
        )

    monkeypatch.setattr(module, "_post_json", fake_post_json)
    replayer = module.WorkflowTraceReplayer(
        router_url="http://router",
        model="test-model",
    )
    call = module.WorkflowCall(
        call_id="call-observed",
        parents=["parent-0"],
        request={"messages": [{"role": "user", "content": "hello"}]},
        input_len=32,
        output_len=12,
        planner_hints={"exclusive_profile_latency_ms": 45.0},
    )

    returned = asyncio.run(
        replayer._run_call(
            None,
            "wf-observed",
            call,
            results,
            workflow_arrival_ms=123,
            workflow_horizon_ms=456,
        )
    )

    assert returned == "call-observed"
    row = results[0]
    assert row["workflow_arrival_ms"] == 123
    assert row["request_started_at_ms"] <= row["response_completed_at_ms"]
    assert row["total_latency_ms"] >= 0
    assert row["prefill_worker"] == "http://prefill-2"
    assert row["decode_worker"] == "http://decode-1"
    assert row["router_mode"] == "hexgent"
    assert row["schedule_revision"] == 17
    assert row["router_selection_wall_ms"] == 2.5
    assert row["router_snapshot_fetch_ms"] == 0.8
    assert row["router_admission_wait_ms"] == 1.6
    assert row["input_tokens"] == 32
    assert row["output_tokens"] == 12
    assert row["parents"] == ["parent-0"]
    assert row["exclusive_profile_latency_ms"] == 45.0


def test_convert_bfcl_records_builds_tool_using_workflow_from_real_schema():
    module = _load_module()
    records = [
        module._maybe_load_json(_DATA_DIR / "bfcl_exec_simple_sample.jsonl")[0],
        module._maybe_load_json(_DATA_DIR / "bfcl_multi_turn_base_sample.jsonl")[0],
    ]

    workflows = module.convert_bfcl_records(records, tokenizer=_FakeTokenizer())

    assert len(workflows) == 2
    simple = workflows[0]
    assert simple.workflow_type == "bfcl"
    assert simple.calls[0].metadata["stage"] == "think"
    assert any(call.metadata["stage"] == "tool" for call in simple.calls)
    multi = workflows[1]
    assert len(multi.calls) >= 3
    assert multi.calls[1].parents == [multi.calls[0].call_id]


def test_convert_hotpotqa_rows_builds_lats_style_tree_from_real_schema():
    module = _load_module()
    rows = module._maybe_load_json(_DATA_DIR / "hotpotqa_sample_rows.jsonl")

    workflows = module.convert_hotpotqa_rows_to_lats_style(
        rows,
        max_calls_per_workflow=12,
    )

    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow.workflow_type == "lats"
    assert workflow.calls[0].metadata["stage"] == "root"
    assert len(workflow.calls) > 1
    assert any(call.metadata["stage"] == "aggregate" for call in workflow.calls)


def test_convert_records_supports_real_dataset_presets():
    module = _load_module()
    sharegpt_records = module._maybe_load_json(_DATA_DIR / "sharegpt_sample.json")
    bfcl_records = module._maybe_load_json(_DATA_DIR / "bfcl_exec_simple_sample.jsonl")
    hotpot_rows = module._maybe_load_json(_DATA_DIR / "hotpotqa_sample_rows.jsonl")

    sharegpt = module._convert_records(
        sharegpt_records,
        input_format="sharegpt_hf",
        tokenizer=_FakeTokenizer(),
        model="test-model",
    )
    bfcl = module._convert_records(
        bfcl_records,
        input_format="bfcl_hf",
        tokenizer=_FakeTokenizer(),
    )
    lats = module._convert_records(
        hotpot_rows,
        input_format="hotpotqa_hf",
        tokenizer=_FakeTokenizer(),
    )

    assert sharegpt[0].workflow_type == "sharegpt"
    assert bfcl[0].workflow_type == "bfcl"
    assert lats[0].workflow_type == "lats"
