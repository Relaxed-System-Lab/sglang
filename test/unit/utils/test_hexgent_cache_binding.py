"""CPU-only binding of original requests, real tokenization replies and cache facts."""
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = _ROOT / "python/sglang"


def _load(name):
    for package_name, path in (("sglang", _PACKAGE), ("sglang.hexgent", _PACKAGE / "hexgent")):
        if package_name not in sys.modules:
            module = types.ModuleType(package_name)
            module.__path__ = [str(path)]
            sys.modules[package_name] = module
    spec = importlib.util.spec_from_file_location(
        "sglang.hexgent." + name, _PACKAGE / "hexgent" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _request(schemas, body=None, route="/v1/chat/completions"):
    workers = [schemas.WorkerDescriptor(id="p"+str(i), url="http://p"+str(i),
                                        worker_type="prefill") for i in range(2)]
    return schemas.PlannerScheduleRequest(
        route=route, model="synthetic", request_text="flattened text must not be used",
        max_tokens=8, request_body=body,
        hexgent=schemas.HexgentRequestMetadata(workflow_id="wf", call_id="c", input_tokens=999),
        prefill_workers=workers, decode_workers=[])


def _probe():
    return dict(probe_id="wf/c", input_ids=list(range(17)), extra_key=None, max_prefix_tokens=16)


def _observation(cache, probe, device=8):
    return dict(probe_id=probe["probe_id"], probe_digest=cache.prefix_probe_digest(probe),
                input_tokens=len(probe["input_ids"]), max_prefix_tokens=probe["max_prefix_tokens"],
                status="observed", source="tree_cache.match_prefix", device_prefix_tokens=device,
                host_prefix_tokens=100, storage_prefix_tokens=100, page_size=4)


def _snapshot(observation=None, enabled=True):
    result = dict(mode="prefill", prefix_cache=dict(enabled=enabled, page_size=4))
    if observation is not None:
        result["prefix_observations"] = [observation]
    return result


class ObservedPrefixBindingTests(unittest.TestCase):
    def setUp(self):
        self.cache = _load("cache")
        self.schemas = _load("schemas")
        self.request = _request(self.schemas)
        self.request.prefix_cache_probe = _probe()

    def read(self):
        self.assertTrue(hasattr(self.cache, "observed_prefix_tokens"), "Missing observed-prefix binding helper")
        return self.cache.observed_prefix_tokens(self.request)

    def test_per_worker_credit_requires_exact_worker_observations(self):
        probe = self.request.prefix_cache_probe
        self.request.state.worker_snapshots = {
            "http://p0": {"snapshots": [_snapshot(_observation(self.cache, probe, 8))]},
            "p1": {"snapshot": _snapshot(_observation(self.cache, probe, 4))},
        }
        self.assertEqual(self.read(), [8, 4])

    def test_probe_id_digest_and_runtime_source_must_all_match(self):
        probe = self.request.prefix_cache_probe
        for field, wrong in (("probe_id", "other/c"), ("probe_digest", "wrong"),
                             ("source", "simulator_hits"), ("device_prefix_tokens", 20)):
            with self.subTest(field=field):
                row = dict(_observation(self.cache, probe), **{field: wrong})
                self.request.state.worker_snapshots = {"p0": {"snapshots": [_snapshot(row)]}}
                self.assertEqual(self.read(), [None, None])
        self.request.prefix_cache_probe = dict(probe, probe_id="other/c")
        row = _observation(self.cache, self.request.prefix_cache_probe)
        self.request.state.worker_snapshots = {"p0": {"snapshots": [_snapshot(row)]}}
        self.assertEqual(self.read(), [None, None])

    def test_disabled_is_known_zero_without_ids_and_missing_is_unknown(self):
        self.request.prefix_cache_probe = None
        self.request.state.worker_snapshots = {"p0": {"snapshots": [_snapshot(enabled=False)]},
                                               "p1": {"snapshots": [_snapshot()]}}
        self.assertEqual(self.read(), [0, None])

    def test_all_dp_replicas_must_support_the_conservative_credit(self):
        probe = self.request.prefix_cache_probe
        replicas = [_snapshot(_observation(self.cache, probe, 8)),
                    _snapshot(_observation(self.cache, probe, 4))]
        self.request.state.worker_snapshots = {"p0": {"dp_rank_count": 2, "snapshots": replicas}}
        self.assertEqual(self.read(), [4, None])
        replicas[1] = _snapshot()
        self.assertEqual(self.read(), [None, None])

    def test_host_and_storage_residency_never_become_device_credit(self):
        probe = self.request.prefix_cache_probe
        self.request.state.worker_snapshots = {"p0": {"snapshots": [_snapshot(_observation(self.cache, probe, 0))]}}
        self.assertEqual(self.read(), [0, None])


class _Response:
    def __init__(self, tokens):
        self.tokens = tokens
    def raise_for_status(self):
        return None
    def json(self):
        return dict(tokens=self.tokens, count=len(self.tokens), max_model_len=8192)


class _Client:
    def __init__(self, tokens=None, error=None):
        self.tokens = [10, 20, 30, 40] if tokens is None else tokens
        self.error = error
        self.calls = []
    async def post(self, url, *, json, timeout):
        self.calls.append((url, copy.deepcopy(json), timeout))
        if self.error:
            raise self.error
        return _Response(self.tokens)


class PreparePrefixBindingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cache, self.schemas = _load("cache"), _load("schemas")

    async def prepare(self, request, client):
        self.assertTrue(hasattr(self.cache, "prepare_prefix_cache_probe"), "Missing backend tokenizer binding helper")
        return await self.cache.prepare_prefix_cache_probe(request, client=client, timeout_s=0.5)

    async def test_original_chat_body_tools_and_template_options_reach_backend(self):
        body = dict(model="synthetic", messages=[{"role": "user", "content": "hello"}],
                    tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
                    tool_choice="auto", chat_template_kwargs={"enable_thinking": False},
                    continue_final_message=False, reasoning_effort="low", max_tokens=8)
        original = copy.deepcopy(body)
        request, client = _request(self.schemas, body), _Client()
        prepared = await self.prepare(request, client)
        self.assertIsNot(prepared, request)
        self.assertIsNone(request.prefix_cache_probe)
        self.assertEqual(prepared.hexgent.input_tokens, 999)
        self.assertEqual(request.request_body, original)
        self.assertEqual(client.calls, [("http://p0/v1/tokenize", original, 0.5)])
        self.assertEqual(prepared.prefix_cache_probe,
                         dict(probe_id="wf/c", input_ids=[10,20,30,40], extra_key=None, max_prefix_tokens=3))

    async def test_native_text_uses_backend_default_special_token_behavior(self):
        body = dict(text="hello", return_logprob=True, logprob_start_len=0)
        request, client = _request(self.schemas, body, "/generate"), _Client()
        prepared = await self.prepare(request, client)
        self.assertEqual(client.calls[0][1], dict(model="synthetic", prompt="hello", add_special_tokens=True))
        self.assertEqual(prepared.prefix_cache_probe["max_prefix_tokens"], 0)
        self.assertEqual(prepared.hexgent.input_tokens, 999)

    async def test_direct_input_ids_keep_native_logprob_cap_without_http(self):
        request = _request(self.schemas, dict(input_ids=[1,2,3,4], return_logprob=True,
                                              logprob_start_len=2), "/generate")
        client = _Client()
        prepared = await self.prepare(request, client)
        self.assertEqual(prepared.prefix_cache_probe["input_ids"], [1,2,3,4])
        self.assertEqual(prepared.prefix_cache_probe["max_prefix_tokens"], 2)
        self.assertEqual(client.calls, [])

    async def test_chat_input_ids_follow_actual_chat_logprob_start_minus_one(self):
        request = _request(self.schemas, dict(messages=[], input_ids=[1,2,3,4], logprobs=True,
                                              logprob_start_len=0))
        client = _Client()
        prepared = await self.prepare(request, client)
        self.assertEqual(prepared.prefix_cache_probe["max_prefix_tokens"], 3)
        self.assertEqual(client.calls, [])

    async def test_unsupported_namespaces_leave_unknown_and_clear_stale_probe(self):
        for field, value in (("cache_salt", "salt"), ("cache_salt", ""),
                             ("extra_key", "key"), ("lora_path", "adapter")):
            with self.subTest(field=field, value=value):
                request = _request(self.schemas, dict(messages=[{"role":"user","content":"x"}], **{field:value}))
                request.prefix_cache_probe = _probe()
                client = _Client()
                prepared = await self.prepare(request, client)
                self.assertIsNone(prepared.prefix_cache_probe)
                self.assertEqual(client.calls, [])

    async def test_multimodal_or_embeddings_leave_unknown(self):
        for body in (dict(messages=[{"role":"user","content":[{"type":"image_url","image_url":{"url":"image"}}]}]),
                     dict(input_ids=[1,2,3], input_embeds=[[0.1]])):
            request, client = _request(self.schemas, body), _Client()
            self.assertIsNone((await self.prepare(request, client)).prefix_cache_probe)
            self.assertEqual(client.calls, [])

    async def test_network_failure_or_batched_ids_leave_unknown(self):
        request = _request(self.schemas, dict(messages=[{"role":"user","content":"x"}]))
        for client in (_Client(error=RuntimeError("synthetic unavailable tokenizer")), _Client(tokens=[[1,2]])):
            self.assertIsNone((await self.prepare(request, client)).prefix_cache_probe)

    async def test_missing_original_body_never_uses_flattened_request_text(self):
        request, client = _request(self.schemas), _Client()
        self.assertIsNone((await self.prepare(request, client)).prefix_cache_probe)
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
