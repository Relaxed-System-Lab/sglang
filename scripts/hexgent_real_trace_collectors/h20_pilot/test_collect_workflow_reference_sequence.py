"""Synthetic sequence test; never calls an endpoint."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import pytest

ROOT=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location(name, ROOT/(name+".py"))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize("fail_idle_at", [None, 2])
def test_four_calibration_burst_then_eight_standalone_workflows(tmp_path,monkeypatch,fail_idle_at):
    support=load("h20_run_support"); producer=load("collect_workflow_references")
    api_path=Path(os.environ["H20_REPLAY_TEST_SOURCE"])/"python/sglang/hexgent/workflow_trace.py"
    spec=importlib.util.spec_from_file_location("_sequence_actual_workflow_trace",api_path)
    api=importlib.util.module_from_spec(spec);sys.modules[spec.name]=api;spec.loader.exec_module(api)
    inputs=tmp_path/"inputs";inputs.mkdir()
    def workflow(name):
        return {"workflow_id":name,"arrival_offset":99,"calls":[{"call_id":name+"-root","parents":[],"input_len":4,"output_len":2,"request":{"messages":[{"role":"user","content":name}]},"planner_hints":{"model_config":{},"hardware_params":{}}}]}
    files={"sharegpt.calibration.sanitized.runtime.jsonl":[workflow("sg-cal1"),workflow("sg-cal0")],
           "hotpotqa.calibration.sanitized.runtime.jsonl":[workflow("hot-cal1"),workflow("hot-cal0")],
           "sharegpt.heldout.sanitized.runtime.jsonl":[workflow("sg3"),workflow("sg1"),workflow("sg0"),workflow("sg2")],
           "hotpotqa.heldout.sanitized.runtime.jsonl":[workflow("hot3"),workflow("hot1"),workflow("hot0"),workflow("hot2")]}
    for name,rows in files.items():
        (inputs/name).write_text("".join(json.dumps(r)+"\n" for r in rows))
    calls=[]
    async def fake_replay(api, workflows, **kwargs):
        calls.append({"ids":[w.workflow_id for w in workflows],"offsets":[w.arrival_offset for w in workflows],"kwargs":kwargs})
        rows=[]
        for workflow in workflows:
            for call in workflow.calls:
                rows.append({"workflow_id":workflow.workflow_id,"call_id":call.call_id,"workflow_arrival_ms":1000,"request_started_at_ms":1010,"response_completed_at_ms":1055,"error":None,"response":{"model":"served-test","usage":{"prompt_tokens":call.input_len,"completion_tokens":call.output_len}}})
        Path(kwargs["raw_path"]).write_text("".join(json.dumps(r)+"\n" for r in rows))
        return rows
    monkeypatch.setattr(support,"replay_to_raw",fake_replay)
    evidence=tmp_path/"runtime.json";evidence.write_text('{"synthetic":true}')
    limits=tmp_path/"limits.json";limits.write_text("{}")
    idle=[]
    def fake_idle(path, expected, deadline):
        assert time.time()+25 < deadline < time.time()+35, "idle helper deadline is UNIX epoch seconds"
        index=len(idle);idle.append((Path(path).name,len(calls)))
        record={"success":index!=fail_idle_at,"error":"synthetic idle failure" if index==fail_idle_at else None}
        Path(path).write_text(json.dumps(record))
        if index==fail_idle_at:raise RuntimeError("synthetic idle failure")
        return record
    monkeypatch.setattr(producer,"idle_check",fake_idle,raising=False)
    out=tmp_path/"output"
    coroutine=producer.collect_references(api,runtime_input_dir=inputs,output_dir=out,router_url="http://no-network",namespace="unique-run",model="served-test",worker_class="H20-BF16-TP2",implementation_revision="synthetic-revision",runtime_evidence_path=evidence,expected_limits_path=limits)
    if fail_idle_at is not None:
        with pytest.raises(RuntimeError,match="synthetic idle failure"):
            asyncio.run(coroutine)
        gate=json.loads((out/"collection-gate.json").read_text())
        assert gate["all_heldout_references_complete"] is False
        assert gate["blocked_phase"]=="idle-before-standalone-01"
        assert len(calls)==2 and len(idle)==3
        assert len((out/"workflow-references.jsonl").read_text().splitlines())==1
        assert json.loads((out/"idle-before-standalone-01.json").read_text())["success"] is False
        return
    gate=asyncio.run(coroutine)
    assert calls[0]["ids"]==["sg-cal0","sg-cal1","hot-cal0","hot-cal1"]
    assert calls[0]["offsets"]==[0.0]*4
    assert [x["ids"] for x in calls[1:]]==[[x] for x in ["sg0","sg1","sg2","sg3","hot0","hot1","hot2","hot3"]]
    assert all(x["offsets"]==[0.0] for x in calls[1:])
    assert len({x["kwargs"]["run_namespace"] for x in calls})==9
    assert len(idle)==9
    assert idle[:2]==[("idle-after-calibration.json",1),("idle-before-standalone-00.json",1)]
    assert idle[2:]==[(f"idle-before-standalone-{index:02d}.json",index+1) for index in range(1,8)]
    assert gate["all_heldout_references_complete"] is True
    refs=[json.loads(r) for r in (out/"workflow-references.jsonl").read_text().splitlines()]
    assert len(refs)==8 and all(r["latency_ms"]==55 for r in refs)
    summary=json.loads((out/"calibration-burst.summary.json").read_text())
    assert summary["service_fit_input"] is False
    assert summary["mixed_trace_cohort"] is True and summary["cross_trace_concurrency"] is True
    assert summary["workflow_ids"]==["sg-cal0","sg-cal1","hot-cal0","hot-cal1"]
    assert summary["datasets"]["sharegpt"]["workflow_count"]==2
    assert summary["datasets"]["sharegpt"]["makespan_ms"]==55
    assert summary["datasets"]["hotpotqa"]["makespan_ms"]==55
    assert evidence.read_text()=='{"synthetic":true}'
