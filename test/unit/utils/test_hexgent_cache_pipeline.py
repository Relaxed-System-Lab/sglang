"""CPU-only original-body, probe refresh and snapshot occupancy wiring tests."""
import ast
import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

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


def _probe(call="a", offset=0):
    return dict(probe_id="wf/"+call, input_ids=list(range(offset, offset+8)),
                extra_key=None, max_prefix_tokens=7)


def _request(schemas, call="a"):
    return schemas.PlannerScheduleRequest(
        route="/v1/chat/completions", model="synthetic", request_text="unused flattened text",
        max_tokens=2, request_body=dict(model="synthetic", messages=[{"role":"user","content":call}],
            tools=[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
            chat_template_kwargs={"enable_thinking":False}),
        prefix_cache_probe=_probe(call),
        hexgent=schemas.HexgentRequestMetadata(workflow_id="wf", call_id=call, input_tokens=8),
        prefill_workers=[schemas.WorkerDescriptor(id="p",url="http://p",worker_type="prefill")],
        decode_workers=[schemas.WorkerDescriptor(id="d",url="http://d",worker_type="decode")],
        state=schemas.PlannerClusterState(current_time_ms=10,
            decode_scheduled_intervals=[[schemas.ScheduledInterval(40,60,5)]]))


def _positive(cache, probe, tokens):
    return dict(probe_id=probe["probe_id"],probe_digest=cache.prefix_probe_digest(probe),
                input_tokens=len(probe["input_ids"]),max_prefix_tokens=probe["max_prefix_tokens"],
                status="observed",source="tree_cache.match_prefix",device_prefix_tokens=tokens,
                host_prefix_tokens=0,storage_prefix_tokens=None,page_size=1)


def _snapshots(cache, probe):
    return {
        "http://p":dict(dp_rank_count=1,snapshots=[dict(mode="prefill",instance_id="p",timestamp=1,
            prefix_cache=dict(enabled=True,page_size=1),
            prefix_observations=[_positive(cache,probe,4)],
            queues={"waiting":[dict(rid="queued-p",call_id="queued-p",workflow_id="queued",
                input_tokens=9,output_tokens_generated=0,max_new_tokens=2,prefill_started=False)]},
            running=[],limits={"max_prefill_tokens":8192,"prefill_max_requests":4})]),
        "http://d":dict(dp_rank_count=1,snapshots=[dict(mode="decode",instance_id="d",timestamp=1,
            queues={},running=[dict(rid="running-d",call_id="running-d",workflow_id="running",
                input_tokens=13,output_tokens_generated=2,max_new_tokens=8,memory_tokens=15)],
            limits={"max_running_requests":4,"max_total_num_tokens":100})]),
    }


@pytest.mark.parametrize("endpoint", ["schedule_request", "admit_request", "enqueue_admission"])
def test_sidecar_automatic_prepare_receives_original_body_and_probe(endpoint, monkeypatch):
    schemas = _load("schemas")
    request = _request(schemas)
    original_body, original_probe = copy.deepcopy(request.request_body), copy.deepcopy(request.prefix_cache_probe)
    prepared, consumed = [], []
    replacement_probe = _probe("a",100)
    async def prepare(value):
        prepared.append(value)
        return replace(value,prefix_cache_probe=replacement_probe)
    def schedule(value):
        consumed.append(value)
        return {"ok":True}
    def admit(value):
        consumed.append(value.request)
        return {"ok":True}
    coordinator = SimpleNamespace(enqueue=admit,wake=lambda:None)
    namespace = dict(app=FastAPI(),os=os,time=time,replace=replace,telemetry=None,
        HTTPException=HTTPException,PlannerScheduleRequest=schemas.PlannerScheduleRequest,
        PlannerAdmissionRequest=schemas.PlannerAdmissionRequest,to_jsonable=schemas.to_jsonable,
        prepare_prefix_cache_probe=prepare,
        planner=SimpleNamespace(schedule_request=schedule,admit_request=admit),
        admission_coordinator=coordinator)
    path = _PACKAGE / "hexgent/sidecar.py"
    tree = ast.parse(path.read_text())
    build = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="build_app")
    routes = [copy.deepcopy(node) for node in build.body
              if isinstance(node,ast.AsyncFunctionDef) and node.name==endpoint]
    assert len(routes)==1
    exec(compile(ast.fix_missing_locations(ast.Module(body=routes,type_ignores=[])),str(path),"exec"),namespace)
    monkeypatch.setenv("HEXGENT_ENABLE_PREFIX_CACHE_PROBES","1")
    payload = schemas.to_jsonable(request)
    if endpoint != "schedule_request":
        payload = {"request":payload}
    with TestClient(namespace["app"]) as client:
        response = client.post("/"+endpoint,json=payload)
    assert response.status_code==200,response.text
    assert len(prepared)==len(consumed)==1
    assert prepared[0].request_body==original_body
    assert prepared[0].prefix_cache_probe==original_probe
    assert consumed[0].request_body==original_body
    assert consumed[0].prefix_cache_probe==replacement_probe
    assert consumed[0].hexgent.input_tokens==8


def test_plural_pending_probes_share_one_fresh_fetch_and_keep_own_binding(monkeypatch):
    schemas, cache, snapshots = _load("schemas"),_load("cache"),_load("snapshots")
    request = _request(schemas)
    other = _probe("b",100)
    calls = []
    payload = _snapshots(cache,request.prefix_cache_probe)
    payload["http://p"]["snapshots"][0]["prefix_observations"].append(_positive(cache,other,0))
    def fetch(workers,**kwargs):
        calls.append((list(workers),kwargs))
        return payload
    monkeypatch.setattr(snapshots,"fetch_worker_snapshots",fetch)
    monkeypatch.setenv("HEXGENT_ENABLE_WORKER_SNAPSHOTS","1")
    monkeypatch.setenv("HEXGENT_ENABLE_PREFIX_CACHE_PROBES","1")
    snapshots.maybe_apply_live_worker_snapshots(request,prefix_probes=[request.prefix_cache_probe,other])
    assert len(calls)==1
    assert calls[0][1]["prefix_probes"]==[request.prefix_cache_probe,other]
    assert cache.observed_prefix_tokens(request)==[4]
    other_request = replace(request,hexgent=replace(request.hexgent,call_id="b"),prefix_cache_probe=other)
    assert cache.observed_prefix_tokens(other_request)==[0]


def test_failed_fresh_fetch_happens_once_retains_occupancy_and_drops_old_credit(monkeypatch):
    schemas, cache, snapshots = _load("schemas"),_load("cache"),_load("snapshots")
    request = _request(schemas)
    previous = _snapshots(cache,request.prefix_cache_probe)
    request.state.worker_snapshots = copy.deepcopy(previous)
    calls = []
    def fetch(workers,**kwargs):
        calls.append(kwargs)
        return {}
    monkeypatch.setattr(snapshots,"fetch_worker_snapshots",fetch)
    monkeypatch.setenv("HEXGENT_ENABLE_WORKER_SNAPSHOTS","1")
    monkeypatch.setenv("HEXGENT_ENABLE_PREFIX_CACHE_PROBES","1")
    snapshots.maybe_apply_live_worker_snapshots(request,prefix_probes=[request.prefix_cache_probe,_probe("b",100)])
    assert len(calls)==1,"A failed authoritative refresh must not trigger a second single-probe fetch"
    assert request.state.worker_snapshots["http://p"]["snapshots"][0]["queues"]==previous["http://p"]["snapshots"][0]["queues"]
    assert request.state.worker_snapshots["http://d"]["snapshots"][0]["running"]==previous["http://d"]["snapshots"][0]["running"]
    assert request.state.decode_scheduled_intervals[0],"Existing decode occupancy must not become an empty calendar"
    assert cache.observed_prefix_tokens(request)==[None]


def test_automatic_probe_flag_off_keeps_explicit_probe_fetch_get_only(monkeypatch):
    schemas, snapshots = _load("schemas"),_load("snapshots")
    request = _request(schemas)
    calls = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self,*args):
            return False
        def read(self):
            return json.dumps({"snapshots":[]}).encode()
    def urlopen(message,timeout):
        calls.append((message.get_method(),message.data))
        return Response()
    monkeypatch.setattr(snapshots.urllib.request,"urlopen",urlopen)
    monkeypatch.setenv("HEXGENT_ENABLE_WORKER_SNAPSHOTS","1")
    monkeypatch.setenv("HEXGENT_ENABLE_PREFIX_CACHE_PROBES","0")
    snapshots.maybe_apply_live_worker_snapshots(request)
    assert calls==[("GET",None),("GET",None)]
