"""CPU source/API cache tests; these do not establish live cache performance."""
import importlib.util
import ast
import asyncio
import copy
import dataclasses
import time
from array import array
from typing import Any, Dict, List, Optional
from unittest.mock import patch
import inspect
import os
from pathlib import Path
import sys
import types
import unittest
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = _ROOT / "python/sglang"


def _load(name):
    for package_name, path in (("sglang", _PACKAGE), ("sglang.hexgent", _PACKAGE / "hexgent")):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(path)]
            sys.modules[package_name] = package
    assert (_PACKAGE / "hexgent" / (name + ".py")).exists(), "Missing cache observation helper"
    spec = importlib.util.spec_from_file_location(
        "sglang.hexgent." + name, _PACKAGE / "hexgent" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _probe(**kwargs):
    return dict(probe_id="wf/c", input_ids=list(range(17)), extra_key="saltadapter",
                max_prefix_tokens=15, **kwargs)


class PrefixProbeTests(unittest.TestCase):
    def test_actual_match_lengths_and_exact_namespace_are_reported(self):
        cache = _load("cache")
        calls = []
        def match(input_ids, extra_key, max_prefix_tokens):
            calls.append((input_ids, extra_key, max_prefix_tokens))
            return SimpleNamespace(device_indices=[4] * 8, host_hit_length=4)
        result = cache.probe_prefix_cache([_probe()], cache_enabled=True,
                                          match_prefix=match, page_size=4, now=7.0)[0]
        self.assertEqual(calls, [(list(range(17)), "saltadapter", 15)])
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["device_prefix_tokens"], 8)
        self.assertEqual(result["host_prefix_tokens"], 4)
        self.assertIsNone(result["storage_prefix_tokens"])
        self.assertEqual(result["observed_at"], 7.0)
        self.assertEqual(result["probe_digest"], cache.prefix_probe_digest(_probe()))

    def test_disabled_unknown_and_observed_miss_are_distinct(self):
        cache = _load("cache")
        def match(*args):
            return SimpleNamespace(device_indices=[], host_hit_length=0)
        disabled = cache.probe_prefix_cache([_probe()], cache_enabled=False,
                                            match_prefix=None, page_size=4)[0]
        unknown = cache.probe_prefix_cache([_probe()], cache_enabled=None,
                                           match_prefix=None, page_size=4)[0]
        miss = cache.probe_prefix_cache([_probe()], cache_enabled=True,
                                        match_prefix=match, page_size=4)[0]
        self.assertEqual([disabled["status"], unknown["status"], miss["status"]],
                         ["disabled", "unknown", "observed"])
        self.assertEqual(disabled["device_prefix_tokens"], 0)
        self.assertIsNone(unknown["device_prefix_tokens"])
        self.assertEqual(miss["device_prefix_tokens"], 0)

    def test_digest_binds_tokens_namespace_and_prefix_cap(self):
        cache = _load("cache")
        original = _probe()
        digest = cache.prefix_probe_digest(original)
        for field, value in (("input_ids", [99] + original["input_ids"][1:]),
                             ("extra_key", "other"), ("max_prefix_tokens", 12)):
            changed = dict(original, **{field: value})
            self.assertNotEqual(digest, cache.prefix_probe_digest(changed))

    def test_http_probe_work_is_bounded(self):
        cache = _load("cache")
        with self.assertRaises(ValueError):
            cache.validate_prefix_probes([_probe()] * (cache.MAX_PREFIX_PROBES + 1))
        invalid = dict(_probe(), max_prefix_tokens=17)
        with self.assertRaises(ValueError):
            cache.validate_prefix_probes([invalid])


class PrefixLatencyTests(unittest.TestCase):
    def setUp(self):
        self.schemas = _load("schemas")
        self.estimator_module = _load("latency_estimator")
        self.workers = [self.schemas.WorkerDescriptor(
            id="p" + str(i), url="http://p" + str(i), worker_type="prefill",
            prefill_base_ms=2, prefill_ms_per_input_token=1,
            decode_ms_per_output_token=3, transfer_ms_per_token=2) for i in range(2)]
        self.request = self.schemas.PlannerScheduleRequest(
            route="/generate", model="synthetic", request_text="", max_tokens=8,
            hexgent=self.schemas.HexgentRequestMetadata(workflow_id="wf", call_id="c"),
            prefill_workers=self.workers, decode_workers=[self.workers[0]])

    def test_per_prefill_prefix_only_changes_prefill_service(self):
        estimator = self.estimator_module.HybridLatencyEstimator()
        baseline = estimator.estimate(self.request, input_tokens=100, output_tokens=8)
        self.assertTrue(hasattr(estimator, "estimate_with_prefixes"), "Missing prefix-aware estimator API")
        actual = estimator.estimate_with_prefixes(
            self.request, input_tokens=100, output_tokens=8,
            cached_prefix_tokens_by_prefill_instance=[60, None])
        self.assertEqual(actual.prefill_ms_by_instance, [42, 102])
        self.assertEqual(actual.decode_ms_by_instance, baseline.decode_ms_by_instance)
        self.assertEqual(actual.transfer_ms_by_decode_instance, [200])
        self.assertEqual(self.request.hexgent.input_tokens, None)

    def test_full_prompt_measurement_does_not_become_suffix_measurement(self):
        self.request.hexgent.estimated_prefill_time = 77
        estimator = self.estimator_module.HybridLatencyEstimator()
        self.assertTrue(hasattr(estimator, "estimate_with_prefixes"), "Missing prefix-aware estimator API")
        actual = estimator.estimate_with_prefixes(
            self.request, input_tokens=100, output_tokens=8,
            cached_prefix_tokens_by_prefill_instance=[60, 0])
        self.assertEqual(actual.prefill_ms_by_instance, [77, 77])

    def test_fractional_worker_labels_survive_in_unrounded_companions(self):
        for worker in self.workers:
            worker.prefill_base_ms = 0.01
            worker.prefill_ms_per_input_token = 0.001
            worker.decode_base_ms = 0.002
            worker.decode_ms_per_output_token = 0.003
            worker.transfer_ms_per_token = 0.0007
        for estimator in (self.estimator_module.LinearWorkerLatencyEstimator(),
                          self.estimator_module.RooflineLatencyEstimator(),
                          self.estimator_module.HybridLatencyEstimator()):
            value = estimator.estimate(self.request, input_tokens=100, output_tokens=8)
            self.assertTrue(hasattr(value, "prefill_ms_unrounded_by_instance"))
            self.assertEqual(value.prefill_ms_by_instance, [1, 1])
            self.assertAlmostEqual(value.prefill_ms_unrounded_by_instance[0], 0.11)
            self.assertAlmostEqual(value.decode_ms_unrounded_by_instance[0], 0.026)
            self.assertAlmostEqual(value.transfer_ms_unrounded_by_decode_instance[0], 0.07)
        suffix = self.estimator_module.HybridLatencyEstimator().estimate_with_prefixes(
            self.request, input_tokens=100, output_tokens=8,
            cached_prefix_tokens_by_prefill_instance=[60, 0])
        self.assertAlmostEqual(suffix.prefill_ms_unrounded_by_instance[0], 0.05)
        self.assertAlmostEqual(suffix.prefill_ms_unrounded_by_instance[1], 0.11)
        self.assertAlmostEqual(suffix.transfer_ms_unrounded_by_decode_instance[0], 0.07)

    def test_fractional_profile_values_are_not_rounded_before_companions(self):
        self.request.hexgent.estimated_prefill_time = 0.234
        self.request.hexgent.estimated_decode_time = 0.456
        self.request.hexgent.estimated_transfer_time = 0.078
        value = self.estimator_module.HybridLatencyEstimator().estimate_with_prefixes(
            self.request, input_tokens=100, output_tokens=8,
            cached_prefix_tokens_by_prefill_instance=[60, None])
        self.assertTrue(hasattr(value, "prefill_ms_unrounded_by_instance"))
        self.assertEqual(value.prefill_ms_by_instance, [1, 1])
        self.assertEqual(value.decode_ms_by_instance, [1])
        self.assertEqual(value.transfer_ms_by_decode_instance, [1])
        self.assertEqual(value.prefill_ms_unrounded_by_instance, [0.234, 0.234])
        self.assertEqual(value.decode_ms_unrounded_by_instance, [0.456])
        self.assertEqual(value.transfer_ms_unrounded_by_decode_instance, [0.078])

    def test_roofline_companions_sum_raw_operator_times(self):
        worker = self.workers[0]
        worker.model_config = dict(hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
        worker.hardware_params = dict(memory_bandwidth=1e12, FP16=1e15,
                                     transfer_bandwidth_bytes_per_s=1e12)
        roofline = self.estimator_module.RooflineLatencyEstimator()
        value = self.estimator_module.HybridLatencyEstimator().estimate(
            self.request, input_tokens=100, output_tokens=8)
        self.assertTrue(hasattr(value, "prefill_ms_unrounded_by_instance"))
        model = self.estimator_module._model_params(self.request, worker)
        def raw(sequence_tokens, stage, include_weights=True):
            ops, weights, dynamic = roofline._stage_work_components(
                model=model, sequence_tokens=sequence_tokens, stage=stage)
            memory = weights + dynamic if include_weights else dynamic
            return roofline._roofline_time_ms(ops, memory, model, worker.hardware_params)
        # Preserve the existing single-call prefill model; batching separately
        # includes shared weight traffic in its real work calculation.
        self.assertAlmostEqual(value.prefill_ms_unrounded_by_instance[0],
                               raw(100, "prefill", include_weights=False))
        self.assertAlmostEqual(value.decode_ms_unrounded_by_instance[0],
                               sum(raw(100+i, "decode") for i in range(8)))
        self.assertLess(value.decode_ms_unrounded_by_instance[0], 1)
        self.assertEqual(value.decode_ms_by_instance[0], 8)
        self.assertTrue(hasattr(roofline, "prefill_batch_ms_unrounded"))
        self.assertTrue(hasattr(roofline, "decode_iteration_ms_unrounded"))
        self.assertAlmostEqual(roofline.prefill_batch_ms_unrounded(
            worker, input_tokens=[100], request=self.request), raw(100, "prefill"))
        self.assertAlmostEqual(roofline.decode_iteration_ms_unrounded(
            worker, sequence_length=100, request=self.request), raw(100, "decode"))

    def test_suffix_attention_keeps_the_full_key_context(self):
        roofline = self.estimator_module.RooflineLatencyEstimator()
        model = self.estimator_module._model_params_from_payload(dict(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8), self.workers[0])
        full = roofline._stage_work_components(model=model, sequence_tokens=100, stage="prefill")
        self.assertIn("cached_prefix_tokens", inspect.signature(roofline._stage_work_components).parameters)
        suffix = roofline._stage_work_components(
            model=model, sequence_tokens=100, stage="prefill", cached_prefix_tokens=60)
        shortened = roofline._stage_work_components(model=model, sequence_tokens=40, stage="prefill")
        self.assertLess(suffix[0], full[0])
        self.assertGreater(suffix[0], shortened[0])
        self.assertEqual(suffix[1], full[1])
        self.assertGreater(suffix[2], shortened[2])


def _definitions(path, names, namespace):
    nodes = [copy.deepcopy(n) for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names), "Missing worker prefix-query API"
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)


class PrefixWorkerAPITests(unittest.TestCase):
    def test_worker_passes_actual_radix_key_namespace_cap_and_page_alignment(self):
        worker_path = _PACKAGE / "srt/managers/hexgent_worker.py"
        spec = importlib.util.spec_from_file_location("prefix_worker_helper", worker_path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        self.assertTrue(hasattr(worker, "build_hexgent_prefix_observations"))
        namespace = dict(__name__=__name__, dataclasses=dataclasses, array=array, Optional=Optional)
        _definitions(_PACKAGE / "srt/mem_cache/radix_cache.py", ["RadixKey"], namespace)
        _definitions(_PACKAGE / "srt/mem_cache/base_prefix_cache.py", ["MatchPrefixParams"], namespace)
        keys = []
        def match(params):
            keys.append(params.key)
            aligned = params.key.page_aligned(4)
            return SimpleNamespace(device_indices=list(range(len(aligned))), host_hit_length=0)
        tree_cache = SimpleNamespace(match_prefix=match, page_size=4)
        with patch.dict(sys.modules, {
            "sglang.srt.mem_cache.radix_cache": SimpleNamespace(RadixKey=namespace["RadixKey"]),
            "sglang.srt.mem_cache.base_prefix_cache": SimpleNamespace(MatchPrefixParams=namespace["MatchPrefixParams"]),
        }):
            result = worker.build_hexgent_prefix_observations(
                tree_cache, [_probe()], cache_enabled=True, now=11)[0]
        self.assertEqual(keys[0].extra_key, "saltadapter")
        self.assertEqual(keys[0].limit, 15)
        self.assertEqual(keys[0].raw_token_ids(), array("q", range(15)))
        self.assertEqual(result["device_prefix_tokens"], 12)
        self.assertEqual(result["page_size"], 4)

    def test_snapshot_schema_and_http_routes_use_existing_communicator(self):
        source = ast.parse((_PACKAGE / "srt/entrypoints/http_server.py").read_text())
        self.assertIn("query_hexgent_pd_snapshot", [n.name for n in source.body if isinstance(n, ast.AsyncFunctionDef)])
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.testclient import TestClient
        import msgspec
        cache = _load("cache")
        namespace = dict(__name__=__name__, msgspec=msgspec, Any=Any, Dict=Dict,
                         List=List, Optional=Optional)
        _definitions(_PACKAGE / "srt/managers/io_struct.py",
                     ["BaseReq", "HexgentPDSnapshotReqInput", "HexgentPDSnapshotReqOutput"], namespace)
        control_tree = ast.parse((_PACKAGE / "srt/managers/tokenizer_control_mixin.py").read_text())
        control_class = next(n for n in control_tree.body if isinstance(n, ast.ClassDef)
                             and any(isinstance(m, ast.AsyncFunctionDef) and m.name == "get_hexgent_pd_snapshot" for m in n.body))
        method = copy.deepcopy(next(m for m in control_class.body
                                    if isinstance(m, ast.AsyncFunctionDef) and m.name == "get_hexgent_pd_snapshot"))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])),
            "actual_tokenizer_control", "exec"), namespace)
        ipc_calls = []
        async def communicator(message):
            ipc_calls.append(message)
            return [namespace["HexgentPDSnapshotReqOutput"](snapshot={"mode": "prefill"})]
        manager = SimpleNamespace(hexgent_pd_snapshot_communicator=communicator)
        async def get_snapshot(*args, **kwargs):
            return await namespace["get_hexgent_pd_snapshot"](manager, *args, **kwargs)
        manager.get_hexgent_pd_snapshot = get_snapshot
        routes = dict(__name__=__name__, app=FastAPI(), Body=Body, Dict=Dict, Any=Any,
                      HTTPException=HTTPException, time=time, validate_prefix_probes=cache.validate_prefix_probes,
                      _global_state=SimpleNamespace(tokenizer_manager=manager))
        _definitions(_PACKAGE / "srt/entrypoints/http_server.py",
                     ["get_hexgent_pd_snapshot", "query_hexgent_pd_snapshot"], routes)
        client = TestClient(routes["app"])
        self.assertEqual(client.get("/hexgent/pd_snapshot").status_code, 200)
        response = client.post("/hexgent/pd_snapshot", json={"prefix_probes": [_probe()]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(ipc_calls[0].prefix_probes, [])
        self.assertEqual(ipc_calls[1].prefix_probes, [_probe()])
        bad = client.post("/hexgent/pd_snapshot", json={"prefix_probes": [dict(_probe(), max_prefix_tokens=17)]})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(len(ipc_calls), 2)

    def test_scheduler_reports_runtime_disabled_capability_and_probe_result(self):
        worker_namespace = dict(__name__=__name__, time=time)
        _definitions(_PACKAGE / "srt/managers/hexgent_worker.py",
                     ["build_hexgent_pd_snapshot_payload"], worker_namespace)
        scheduler_path = Path(os.environ.get("HEXGENT_PREFIX_SCHEDULER_SOURCE",
                                             str(_PACKAGE / "srt/managers/scheduler.py")))
        tree = ast.parse(scheduler_path.read_text())
        scheduler_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        method = copy.deepcopy(next(n for n in scheduler_class.body
                                    if isinstance(n, ast.FunctionDef) and n.name == "get_hexgent_pd_snapshot"))
        cache = _load("cache")
        namespace = dict(
            HexgentPDSnapshotReqOutput=SimpleNamespace,
            build_hexgent_pd_snapshot_payload=worker_namespace["build_hexgent_pd_snapshot_payload"],
            build_hexgent_prefix_observations=lambda tree_cache, probes, cache_enabled:
                cache.probe_prefix_cache(probes, cache_enabled=cache_enabled,
                                         match_prefix=None, page_size=tree_cache.page_size),
            envs=SimpleNamespace(SGLANG_RADIX_FORCE_MISS=SimpleNamespace(get=lambda: False)))
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(scheduler_path), "exec"), namespace)
        scheduler = SimpleNamespace(
            _hexgent_mode_name=lambda: "prefill", ps=SimpleNamespace(dp_rank=0),
            server_args=SimpleNamespace(host="127.0.0.1", port=30000, prefill_max_requests=4),
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]), disagg_prefill_inflight_queue=[],
            waiting_queue=[], running_batch=SimpleNamespace(reqs=[]),
            max_prefill_tokens=8192, max_running_requests=16, max_total_num_tokens=100000,
            tree_cache=SimpleNamespace(disable=True, page_size=4))
        snapshot = namespace["get_hexgent_pd_snapshot"](scheduler, SimpleNamespace(prefix_probes=[])).snapshot
        self.assertEqual(snapshot.get("prefix_cache"), {"enabled": False, "page_size": 4})
        observed = namespace["get_hexgent_pd_snapshot"](scheduler, SimpleNamespace(prefix_probes=[_probe()])).snapshot
        self.assertEqual(observed["prefix_observations"][0]["status"], "disabled")
        self.assertEqual(observed["prefix_observations"][0]["device_prefix_tokens"], 0)

    def test_snapshot_can_report_disabled_without_any_input_ids(self):
        worker_path = _PACKAGE / "srt/managers/hexgent_worker.py"
        namespace = dict(__name__=__name__, time=time)
        _definitions(worker_path, ["build_hexgent_pd_snapshot_payload"], namespace)
        helper = namespace["build_hexgent_pd_snapshot_payload"]
        self.assertIn("prefix_cache", inspect.signature(helper).parameters)
        result = helper(mode="prefill", instance_id="P0", dp_rank=0,
                        waiting_queue=[], running_reqs=[],
                        prefix_cache={"enabled": False, "page_size": 4})
        self.assertEqual(result["prefix_cache"]["enabled"], False)
        self.assertNotIn("prefix_observations", result)


if __name__ == "__main__":
    unittest.main()
