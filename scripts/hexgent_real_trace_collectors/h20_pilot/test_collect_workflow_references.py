"""Synthetic real-replayer support tests: fake HTTP only, no endpoint/workload."""

import asyncio
import copy
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parent
SUPPORT = ROOT / "h20_run_support.py"
MODEL = "synthetic-served-model"
REVISION = "a" * 40
RUNTIME_SHA = "b" * 64
RAW_SHA = "c" * 64


@lru_cache(maxsize=1)
def support():
    assert SUPPORT.is_file(), "h20_run_support implementation is not present yet"
    name = "_workflow_reference_support_tests"
    spec = importlib.util.spec_from_file_location(name, SUPPORT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def api():
    return support().load_workflow_api(Path(os.environ["H20_REPLAY_TEST_SOURCE"]))


def workflow(*, workflow_id="wf-one", fanout=True):
    module = api()
    parents = {"root": [], "left": ["root"], "right": ["root"], "join": ["left", "right"]} if fanout else {"root": [], "slow": ["root"]}
    calls = [module.WorkflowCall(
        call_id=name, parents=upstream,
        request={"model": "embedded-old", "messages": [{"role": "user", "content": name}]},
        input_len=10, output_len=2,
        planner_hints={"model_config": {"frozen": True}, "hardware_params": {"nominal": True}},
    ) for name, upstream in parents.items()]
    return module.WorkflowTrace(workflow_id=workflow_id, calls=calls, workflow_type="synthetic", arrival_offset=0.0)


def response_for(payload):
    return {"model": payload["model"], "usage": {
        "prompt_tokens": payload["hexgent"]["input_tokens"],
        "completion_tokens": payload["max_tokens"],
    }}


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def post(self, *args, **kwargs):
        raise AssertionError("Network access is forbidden in synthetic producer tests")


def fake_http_session(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(ClientSession=FakeSession))


def saved_rows():
    item = workflow()
    starts = {"root": 1010, "left": 1020, "right": 1020, "join": 1080}
    ends = {"root": 1020, "left": 1060, "right": 1080, "join": 1090}
    return [{
        "workflow_id": item.workflow_id, "call_id": call.call_id,
        "parents": list(call.parents), "workflow_arrival_ms": 1000,
        "request_started_at_ms": starts[call.call_id],
        "response_completed_at_ms": ends[call.call_id],
        "total_latency_ms": 99999, "exclusive_profile_latency_ms": 99999,
        "input_tokens": call.input_len, "output_tokens": call.output_len,
        "response": {"model": MODEL, "usage": {"prompt_tokens": call.input_len, "completion_tokens": call.output_len}},
        "error": None,
    } for call in item.calls]


def reference(rows):
    return support().workflow_reference_from_rows(
        workflow(), rows, model=MODEL, implementation_revision=REVISION,
        runtime_evidence_sha256=RUNTIME_SHA, raw_replay_sha256=RAW_SHA,
    )


def assert_failed(row):
    assert row["status"] != "successful"
    assert row["latency_ms"] is None
    assert row["failure_reasons"]
    assert row["workflow_id"] == "wf-one" and row["expected_calls"] == 4
    assert row["raw_replay_sha256"] == RAW_SHA
    assert row["runtime_evidence_sha256"] == RUNTIME_SHA


def test_support_loads_the_actual_workflow_api_once():
    module = api()
    expected = Path(os.environ["H20_REPLAY_TEST_SOURCE"]) / "python/sglang/hexgent/workflow_trace.py"
    assert Path(module.__file__).resolve() == expected.resolve()
    assert support().load_workflow_api(Path(os.environ["H20_REPLAY_TEST_SOURCE"])) is module


def test_reference_elapsed_uses_workflow_arrival_and_parallel_completion():
    rows = saved_rows()
    before = copy.deepcopy(rows)
    result = reference(list(reversed(rows)))
    assert rows == before
    assert result["status"] == "successful"
    assert result["reference_kind"] == "isolated_workflow_replay"
    assert result["latency_ms"] == 90  # 1090 - workflow arrival 1000, not earliest request 1010.
    assert result["expected_calls"] == result["completed_calls"] == result["usage_verified_calls"] == 4
    assert result["errors"] == 0
    assert result["model"] == MODEL and result["implementation_revision"] == REVISION
    assert result["raw_replay_sha256"] == RAW_SHA and result["runtime_evidence_sha256"] == RUNTIME_SHA


def test_missing_call_reference_remains_failed():
    assert_failed(reference(saved_rows()[:-1]))


def test_error_call_reference_remains_failed_and_preserves_error_count():
    rows = saved_rows(); rows[1].update(error="synthetic HTTP failure", response=None)
    result = reference(rows)
    assert_failed(result)
    assert result["errors"] >= 1


def test_duplicate_call_is_rejected_as_failed_evidence():
    rows = saved_rows(); rows.append(copy.deepcopy(rows[0]))
    assert_failed(reference(rows))


@pytest.mark.parametrize("change", ["unexpected_call", "wrong_workflow"])
def test_reference_requires_the_exact_workflow_call_identities(change):
    rows = saved_rows()
    if change == "unexpected_call":
        rows[1]["call_id"] = "not-an-expected-call"
    else:
        rows[1]["workflow_id"] = "not-the-expected-workflow"
    assert_failed(reference(rows))


@pytest.mark.parametrize("change", ["no_usage", "no_prompt", "no_completion", "wrong_prompt", "wrong_completion", "wrong_model"])
def test_reference_requires_observed_exact_usage_and_model(change):
    rows = saved_rows(); reply = rows[1]["response"]
    if change == "no_usage": reply.pop("usage")
    elif change == "no_prompt": reply["usage"].pop("prompt_tokens")
    elif change == "no_completion": reply["usage"].pop("completion_tokens")
    elif change == "wrong_prompt": reply["usage"]["prompt_tokens"] = 11
    elif change == "wrong_completion": reply["usage"]["completion_tokens"] = 3
    else: reply["model"] = "another-served-model"
    assert_failed(reference(rows))


@pytest.mark.parametrize("change", ["missing", "inconsistent", "nan", "inf"])
def test_reference_requires_one_finite_workflow_arrival(change):
    rows = saved_rows()
    values = {"missing": None, "inconsistent": 1001, "nan": float("nan"), "inf": float("inf")}
    rows[1]["workflow_arrival_ms"] = values[change]
    assert_failed(reference(rows))


@pytest.mark.parametrize("end", [1000, 999, float("nan"), float("inf")])
def test_reference_requires_positive_finite_elapsed(end):
    rows = saved_rows()
    for row in rows: row["response_completed_at_ms"] = end
    assert_failed(reference(rows))


def test_incremental_subclass_captures_each_concurrent_workflow_identity(monkeypatch):
    module = api(); fake_http_session(monkeypatch)
    captured = []
    async def fake_post(session, url, payload, timeout_s):
        return response_for(payload), {}
    monkeypatch.setattr(module, "_post_json", fake_post)
    subclass = support().incremental_replayer_class(module, lambda row: captured.append(copy.deepcopy(row)))
    assert issubclass(subclass, module.WorkflowTraceReplayer)
    first = workflow(workflow_id="wf-first"); first.calls = first.calls[:1]
    second = workflow(workflow_id="wf-second"); second.calls = second.calls[:1]
    replayer = subclass(router_url="fake://never-contact", model=MODEL, run_namespace="capture")
    returned = asyncio.run(replayer.replay([first, second], arrival_rate=None))
    assert len(returned) == len(captured) == 2
    assert {(row["workflow_id"], row["call_id"]) for row in captured} == {("wf-first", "root"), ("wf-second", "root")}


def test_actual_replayer_keeps_fanout_parallel_and_fixed_replay_settings(monkeypatch, tmp_path):
    module = api(); fake_http_session(monkeypatch)
    observed = []; settings = []
    original_init = module.WorkflowTraceReplayer.__init__
    def capture_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs); settings.append(dict(vars(self)))
    monkeypatch.setattr(module.WorkflowTraceReplayer, "__init__", capture_init)
    async def scenario():
        left = asyncio.Event(); right = asyncio.Event(); done = set()
        async def fake_post(session, url, payload, timeout_s):
            name = payload["hexgent"]["call_id"].split("::")[-1]
            observed.append((url, copy.deepcopy(payload), timeout_s))
            if name == "left":
                left.set(); await asyncio.wait_for(right.wait(), 1)
            elif name == "right":
                right.set(); await asyncio.wait_for(left.wait(), 1)
            elif name == "join":
                assert {"left", "right"} <= done
            done.add(name)
            return response_for(payload), {}
        monkeypatch.setattr(module, "_post_json", fake_post)
        return await support().replay_to_raw(
            module, [workflow()], raw_path=tmp_path / "fanout.raw.jsonl",
            router_url="fake://never-contact", model=MODEL,
            worker_class="H20-BF16-TP2", run_namespace="fanout-test",
        )
    rows = asyncio.run(scenario())
    persisted = [json.loads(line) for line in (tmp_path / "fanout.raw.jsonl").read_text().splitlines()]
    assert len(rows) == len(persisted) == 4
    assert {row["call_id"] for row in persisted} == {"root", "left", "right", "join"}
    assert all(row["error"] is None for row in persisted)
    assert len({row["workflow_arrival_ms"] for row in persisted}) == 1
    assert settings[0]["profile_table"] is None
    assert settings[0]["exact_output_length"] is True and settings[0]["timeout_s"] == 180
    assert settings[0]["endpoint"] == "/v1/chat/completions"
    assert settings[0]["run_namespace"] == "fanout-test" and settings[0]["model"] == MODEL
    for url, payload, timeout in observed:
        assert url == "fake://never-contact/v1/chat/completions"
        assert timeout == 180 and payload["model"] == MODEL
        assert payload["max_tokens"] == 2 and payload["ignore_eos"] is True
        assert payload["hexgent"]["workflow_id"] == "fanout-test::wf-one"
        assert not {"workflow_horizon_ms", "estimated_prefill_time", "estimated_decode_time", "estimated_transfer_time"} & payload["hexgent"].keys()


def test_incremental_raw_survives_cancellation_before_workflow_completion(monkeypatch, tmp_path):
    module = api(); fake_http_session(monkeypatch); raw = tmp_path / "cancel.raw.jsonl"
    async def scenario():
        blocked = asyncio.Event(); forever = asyncio.Event()
        async def fake_post(session, url, payload, timeout_s):
            if payload["hexgent"]["call_id"].endswith("::slow"):
                blocked.set(); await forever.wait()
            return response_for(payload), {}
        monkeypatch.setattr(module, "_post_json", fake_post)
        task = asyncio.create_task(support().replay_to_raw(
            module, [workflow(fanout=False)], raw_path=raw,
            router_url="fake://never-contact", model=MODEL,
            worker_class="H20-BF16-TP2", run_namespace="cancel-test",
        ))
        await asyncio.wait_for(blocked.wait(), 1)
        on_disk = [json.loads(line) for line in raw.read_text().splitlines()]
        assert [row["call_id"] for row in on_disk] == ["root"]
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    asyncio.run(scenario())
    assert [json.loads(line)["call_id"] for line in raw.read_text().splitlines()] == ["root"]


def test_incremental_raw_survives_hard_stop_of_our_fake_http_subprocess(tmp_path):
    api()  # Verify support is implemented before launching the owned child.
    raw = tmp_path / "hardstop.raw.jsonl"
    child_code = r"""
import asyncio, importlib.util, json, pathlib, sys, types
support_path, source_root, raw_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location('_hardstop_owned_support', support_path)
support = importlib.util.module_from_spec(spec); sys.modules[spec.name] = support; spec.loader.exec_module(support)
api = support.load_workflow_api(pathlib.Path(source_root))
class FakeSession:
 async def __aenter__(self): return self
 async def __aexit__(self, *args): return None
 def post(self, *args, **kwargs): raise AssertionError('No network in synthetic test')
sys.modules['aiohttp'] = types.SimpleNamespace(ClientSession=FakeSession)
async def fake_post(session, url, payload, timeout_s):
 if payload['hexgent']['call_id'].endswith('::slow'):
  rows = [json.loads(line) for line in pathlib.Path(raw_path).read_text().splitlines()]
  assert [row['call_id'] for row in rows] == ['root']
  print('BLOCKED_AFTER_COMPLETED_ROW_FLUSH', flush=True)
  await asyncio.Event().wait()
 return {'model':'synthetic-served-model','usage':{'prompt_tokens':10,'completion_tokens':2}}, {}
api._post_json = fake_post
calls = [api.WorkflowCall(call_id=name,parents=parents,request={'messages':[{'role':'user','content':name}]},input_len=10,output_len=2)
         for name,parents in [('root',[]),('slow',['root'])]]
workflow = api.WorkflowTrace(workflow_id='wf-hardstop',calls=calls,arrival_offset=0.0)
asyncio.run(support.replay_to_raw(api,[workflow],raw_path=pathlib.Path(raw_path),router_url='fake://never-contact',
                                 model='synthetic-served-model',worker_class='H20-BF16-TP2',run_namespace='hardstop-test'))
"""
    process = subprocess.Popen([sys.executable, "-c", child_code, str(SUPPORT), os.environ["H20_REPLAY_TEST_SOURCE"], str(raw)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=3), "Owned fake child did not reach the blocked call"
        line = process.stdout.readline()
        assert line.strip() == "BLOCKED_AFTER_COMPLETED_ROW_FLUSH", line
        process.kill()  # This test owns this new child; no service or unrelated process is touched.
        process.wait(timeout=2)
    finally:
        selector.close()
        if process.poll() is None:
            process.kill(); process.wait(timeout=2)
        process.stdout.close(); process.stderr.close()
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["call_id"] == "root" and rows[0]["error"] is None
