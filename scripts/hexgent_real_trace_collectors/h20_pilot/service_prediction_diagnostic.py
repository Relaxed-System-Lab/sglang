#!/usr/bin/env python3
"""Fixed-parameter CPU stage/service diagnostic; never runs a scheduler or fits."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
import subprocess
import sys
import time
import types

sys.dont_write_bytecode=True
OLD_REV="a65d6f91acfc38f3f86c210b6d17f214e657a04f"
CURRENT_REV="08885ec27573ee1807b822e786ce95ebd06ec884"
ESTIMATOR_SHA="42cded95817e809924b65090992d3772b24008c0ec95ef6165983a501ddd3407"
MODEL_ALIAS="hexagent-llama70b-bf16-tp2-01a1b627"
FULLGROUP_BYTES_PER_TOKEN=327680
EXPECTED={"sharegpt":16,"hotpotqa":133}
HEADER=re.compile(r"ReqTimeStats\(rid=(?P<rid>[^,]+), bootstrap_room=(?P<room>[^,]+), input_len=(?P<input>\d+), cached_input_len=(?P<cached>\d+), output_len=(?P<output>\d+), type=(?P<stage>\w+)\):(?P<details>.*)")
METRICS={"prefill_prefix":("predicted_prefill_prefix_ms","observed_prefill_prefix_ms"),
         "transfer_surrogate":("predicted_transfer_surrogate_ms","observed_transfer_surrogate_ms"),
         "decode_forward":("predicted_decode_forward_ms","observed_decode_forward_ms"),
         "additive_service":("predicted_additive_service_ms","observed_additive_service_ms"),
         "service_vs_client_total":("predicted_additive_service_ms","observed_client_total_ms")}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def write_json(path,value):
    with Path(path).open("x",encoding="utf-8") as target:
        json.dump(value,target,indent=2,allow_nan=False);target.write("\n")


def field(details,name,unit="ms"):
    match=re.search(r"(?:^|, )"+re.escape(name)+r"=([\d.]+)"+re.escape(unit),details.strip())
    if match is None:raise ValueError("missing logged field: "+name)
    return float(match.group(1))


def parse_stats(text,stage):
    parsed=[];errors=[]
    for number,line in enumerate(text.splitlines(),1):
        if "ReqTimeStats(" not in line:continue
        match=HEADER.search(line)
        if match is None:
            errors.append({"line":number,"reason":"unparsed ReqTimeStats header"});continue
        item=match.groupdict()
        if item["rid"].startswith("HEALTH_CHECK"):continue
        try:
            if item["stage"]!=stage:raise ValueError("stage mismatch")
            value={"rid":item["rid"],"room":item["room"],"input_tokens":int(item["input"]),
                   "output_tokens":int(item["output"]),"cached_tokens":int(item["cached"]),
                   "forward_ms":field(item["details"],"forward_duration"),"line":number}
            if stage=="prefill":
                value.update(transfer_mib=field(item["details"],"transfer_total"," MB"),
                             speed_gib_per_s=field(item["details"],"transfer_speed"," GB/s"))
            else:value["transfer_wait_ms"]=field(item["details"],"transfer_duration")
            parsed.append(value)
        except ValueError as error:errors.append({"line":number,"reason":str(error)})
    return parsed,errors


def pair_stages(raw,prefill,decode,parse_errors=()):
    reasons=list(parse_errors)
    if len(prefill)!=len(raw) or len(decode)!=len(raw):
        reasons.append({"reason":"serial stage count mismatch","raw":len(raw),"prefill":len(prefill),"decode":len(decode)})
    if reasons:return None,reasons
    rooms=[];result=[]
    for index,(actual,p,d) in enumerate(zip(raw,prefill,decode)):
        if (p["room"]!=d["room"] or p["room"] in {"None","null",""}
                or p["input_tokens"]!=d["input_tokens"] or p["input_tokens"]!=actual["input_len"]
                or p["output_tokens"]!=1 or d["output_tokens"]!=actual["output_len"]
                or p["cached_tokens"]!=0 or d["cached_tokens"]!=0
                or p["input_tokens"]>=8192):
            reasons.append({"index":index,"reason":"serial order/room/token/cache/single-chunk contract mismatch"});continue
        mib=p["transfer_mib"];speed=p["speed_gib_per_s"]
        if mib<=0.005 or speed<=0.005 or abs(mib*1024**2-p["input_tokens"]*163840)>0.0051*1024**2:
            reasons.append({"index":index,"reason":"rank payload geometry/rounded speed cannot establish transfer duration"});continue
        tail=mib/(1024*speed)*1000
        low=(mib-0.005)/(1024*(speed+0.005))*1000
        high=(mib+0.005)/(1024*(speed-0.005))*1000
        rooms.append(p["room"])
        result.append({"prefill_worker_rid":p["rid"],"decode_worker_rid":d["rid"],"bootstrap_room":p["room"],
            "prefill_slice_line":p["line"],"decode_slice_line":d["line"],
            "observed_prefill_forward_including_handoff_ms":p["forward_ms"],
            "observed_prefill_prefix_ms":p["forward_ms"]-tail,
            "prefill_prefix_interval_low_ms":p["forward_ms"]-0.005-high,
            "prefill_prefix_interval_high_ms":p["forward_ms"]+0.005-low,
            "observed_transfer_surrogate_ms":tail,"transfer_interval_low_ms":low,"transfer_interval_high_ms":high,
            "transfer_rounding_interval_width_ms":high-low,
            "transfer_reported_rank_mib":mib,"transfer_reported_rank_gib_per_s":speed,
            "transfer_rank_bytes_exact":p["input_tokens"]*163840,
            "transfer_fullgroup_bytes_exact":p["input_tokens"]*FULLGROUP_BYTES_PER_TOKEN,
            "observed_decode_forward_ms":d["forward_ms"],"decode_transfer_wait_ms":d["transfer_wait_ms"],
            "observed_additive_service_ms":p["forward_ms"]+d["forward_ms"],
            "additive_service_interval_low_ms":p["forward_ms"]+d["forward_ms"]-0.01,
            "additive_service_interval_high_ms":p["forward_ms"]+d["forward_ms"]+0.01})
    if len(set(rooms))!=len(rooms):reasons.append({"reason":"bootstrap room reused within serial slice"})
    return (None,reasons) if reasons else (result,[])


def predict_components(nominal,input_tokens,output_tokens,startup_ms,bandwidth_bytes_per_s):
    prefill,decode=nominal.estimate_ms(input_tokens,output_tokens)
    transfer=startup_ms+FULLGROUP_BYTES_PER_TOKEN*input_tokens/bandwidth_bytes_per_s*1000
    return {"predicted_prefill_prefix_ms":prefill,"predicted_decode_forward_ms":decode,
            "predicted_transfer_surrogate_ms":transfer,"predicted_additive_service_ms":prefill+decode+transfer}


def error_statistics(pairs,expected_n):
    valid=[(float(p),float(a)) for p,a in pairs if p is not None and a is not None and math.isfinite(p) and math.isfinite(a)]
    differences=[p-a for p,a in valid];percent=[abs(p-a)/abs(a)*100 for p,a in valid if a>0]
    return {"expected_n":expected_n,"n":len(valid),"missing_n":expected_n-len(valid),
        "signed_bias_ms":mean(differences) if valid else None,"mae_ms":mean(map(abs,differences)) if valid else None,
        "rmse_ms":math.sqrt(mean(e*e for e in differences)) if valid else None,
        "mape_percent":mean(percent) if percent else None,"mape_n":len(percent),
        "mape_missing_or_nonpositive_n":expected_n-len(percent),
        "maximum_absolute_error_ms":max(map(abs,differences),default=None),"outliers_excluded":0}


class OnlineNominalRoofline:
    def __init__(self,source_root,model_config,hardware_params):
        source_root=Path(source_root);relative="python/sglang/hexgent/latency_estimator.py"
        self.source_hashes={revision:hashlib.sha256(subprocess.check_output(["git","-C",str(source_root),"show",revision+":"+relative])).hexdigest()
                            for revision in (OLD_REV,CURRENT_REV)}
        self.source_hashes["actual_file"]=sha(source_root/relative)
        if set(self.source_hashes.values())!={ESTIMATOR_SHA}:raise ValueError("online estimator source differs from frozen a65/08885")
        for name,path in (("sglang",source_root/"python/sglang"),("sglang.hexgent",source_root/"python/sglang/hexgent")):
            package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
        latency=importlib.import_module("sglang.hexgent.latency_estimator")
        schemas=importlib.import_module("sglang.hexgent.schemas")
        if Path(latency.__file__).resolve()!=(source_root/relative).resolve():raise ValueError("wrong online estimator imported")
        required={"hidden_size","intermediate_size","num_hidden_layers","num_attention_heads","num_key_value_heads","vocab_size","head_dim","tp_size","w_bit","a_bit","kv_bit"}
        if required-model_config.keys() or model_config["tp_size"]!=2 or any(model_config[k]!=16 for k in ("w_bit","a_bit","kv_bit")):
            raise ValueError("actual complete TP2 BF16 geometry required")
        self.model=latency._model_params_from_payload(model_config,schemas.WorkerDescriptor(id="static",url="static",worker_type="decode",tp_size=2))
        self.roofline=latency.RooflineLatencyEstimator();self.hardware=dict(hardware_params)
        if self.model is None or 2*self.model.num_hidden_layers*self.model.head_dim*self.model.num_key_value_heads*self.model.kv_byte!=FULLGROUP_BYTES_PER_TOKEN:
            raise ValueError("actual model KV geometry mismatch")
    def estimate_ms(self,input_tokens,output_tokens):
        return (self.roofline.estimate_prefill_ms(model=self.model,hardware=self.hardware,input_tokens=input_tokens),
                self.roofline.estimate_decode_total_ms(model=self.model,hardware=self.hardware,input_tokens=input_tokens,output_tokens=output_tokens,batch_size=1))


def run(args):
    started=time.perf_counter();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    old=args.old_run;held=old/"exclusive-heldout-v1";status=read_json(held/"status.json");seal=read_json(old/"native-phase-seal.json")
    if status["phase"]!="heldout_collected_gate_passed" or status["source_revision"]!=OLD_REV:raise ValueError("old held-out profile provenance mismatch")
    features=[];trace_hashes={}
    for dataset,expected_n in EXPECTED.items():
        path=args.trace_dir/(dataset+".heldout.sanitized.runtime.jsonl");trace_hashes[str(path)]=sha(path)
        workflows=read_jsonl(path);items=[(w["workflow_id"],c) for w in workflows for c in w["calls"]]
        if len(items)!=expected_n:raise ValueError("expected held-out trace counts changed")
        features.extend((dataset,wid,call) for wid,call in items)
    hints=features[0][2]["planner_hints"]
    if any(call["planner_hints"]!=hints or set(hints)!={"model_config","hardware_params"} for _,_,call in features):raise ValueError("static hint configuration mismatch")
    if hints["hardware_params"]!={"memory_bandwidth_bytes_per_s":4096e9,"FP16":148e12}:raise ValueError("nominal H20 catalog parameters changed")
    approved=read_json(args.approved_transfer)
    if approved["HEXGENT_TRANSFER_STARTUP_MS"]!=5.435876377145576 or approved["HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S"]!=2379754520.5106893 or approved["heldout_used"]:
        raise ValueError("approved frozen calibration-only transfer parameters changed")
    nominal=OnlineNominalRoofline(args.source_root,hints["model_config"],hints["hardware_params"])
    # Predictions are materialized before reading held-out service measurements.
    predictions={(wid,c["call_id"]):predict_components(nominal,c["input_len"],c["output_len"],approved["HEXGENT_TRANSFER_STARTUP_MS"],approved["HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S"])
                 for _,wid,c in features}
    chronology=[]
    for label in ("exclusive-profile-v1","exclusive-profile-v2"):
        s=read_json(old/label/"status.json")
        chronology.append({"phase":label,"raw_call_count":sum(len(read_jsonl(p)) for p in (old/label).glob("raw.*.jsonl")),
                           "status_phase":s.get("phase"),"started_unix_s":s.get("started_unix_s"),"finished_unix_s":s.get("finished_unix_s"),
                           "before_heldout":s.get("finished_unix_s",math.inf)<=status["started_unix_s"],"used_in_this_predictor":False})
    parameters={"diagnostic_kind":"fixed_online_roofline_stage_and_service_error_no_scheduler_rollout",
        "predictor_input":"frozen runtime-v2 current-call prompt length and fixed output target; no measured service inputs",
        "model":MODEL_ALIAS,"model_config":hints["model_config"],"resolved_model_params":asdict(nominal.model),"hardware_params":hints["hardware_params"],
        "hardware_kind":"nominal catalog, not measured HBM/OPS","batch_size":1,"P_D_scales_fitted_or_applied":False,"transfer_refitted":False,
        "frozen_transfer_startup_ms":approved["HEXGENT_TRANSFER_STARTUP_MS"],"frozen_transfer_fullgroup_effective_bandwidth_bytes_per_s":approved["HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S"],
        "fullgroup_KV_bytes_per_input_token":FULLGROUP_BYTES_PER_TOKEN,"TP0_rank_KV_bytes_per_input_token":163840,
        "approved_transfer_path":str(args.approved_transfer),"approved_transfer_sha256":sha(args.approved_transfer),
        "online_estimator_source_sha256":nominal.source_hashes,"physical_profile_revision":OLD_REV,"native_profile_ELF_sha256":seal["elf_sha256"],
        "expected_native_profile_ELF_sha256":"a96b9c62a8bfaafc89703a2eb8dc2cdf801ef79ac4e6bede0c287261a9b76148",
        "prior_calibration_passes":chronology,"heldout_started_unix_s":status["started_unix_s"],"explicit_heldout_warmup_calls":0,
        "cold_start_claim":False,"new_R2_scheduler_claim":False,"scheduler_or_simulator_run":False,"all_outliers_preserved":True,
        "boundaries":{"P":"logged forward F minus reconstructed P last-chunk transfer T; send/enqueue before timestamp remains in Pprefix",
                      "T":"rounded MiB/GiB_per_s reconstruction, queue-entry after send_kv_chunk(last=True) to P TP-consensus completion; effective, not physical DMA",
                      "D":"logged decode forward_duration; upstream-waiting D transfer_duration excluded",
                      "additive_service":"Pprefix+T+D = logged P forward F + D forward; T/P uncertainty anticorrelates and cancels",
                      "client":"streaming client total_latency_ms includes network/bootstrap/queue/framework costs outside additive model"},
        "rounding":{"transfer_MiB_halfwidth":0.005,"transfer_GiB_per_s_halfwidth":0.005,"forward_ms_halfwidth":0.005},
        "trace_sha256":trace_hashes,"python":sys.version,"cpu_affinity":sorted(os.sched_getaffinity(0)),"threads":1,"script_sha256":sha(__file__)}
    if parameters["native_profile_ELF_sha256"]!=parameters["expected_native_profile_ELF_sha256"]:raise ValueError("old physical native ELF changed")
    write_json(out/"parameters.json",parameters)
    all_records=[];pairing=[];input_hashes={str(held/"status.json"):sha(held/"status.json"),str(old/"native-phase-seal.json"):sha(old/"native-phase-seal.json")}
    slices=out/"log-slices";slices.mkdir()
    for dataset,expected_n in EXPECTED.items():
        stem=dataset+".heldout";rawpath=held/("raw."+stem+".jsonl");raw=read_jsonl(rawpath);input_hashes[str(rawpath)]=sha(rawpath)
        source={(wid,c["call_id"]):c for label,wid,c in features if label==dataset};keys=[(r["workflow_id"],r["call_id"])for r in raw]
        if len(raw)!=expected_n or len(set(keys))!=expected_n or set(keys)!=set(source):raise ValueError("raw held-out identity completeness mismatch")
        for actual in raw:
            call=source[(actual["workflow_id"],actual["call_id"])]
            if (actual.get("error") is not None or actual["model"]!=MODEL_ALIAS or actual["trial"]!=0 or not actual["usage_observed"]
                    or actual["input_len"]!=call["input_len"] or actual["output_len"]!=call["output_len"]
                    or actual["actual_prompt_tokens"]!=call["input_len"] or actual["actual_completion_tokens"]!=call["output_len"]
                    or not math.isfinite(actual["total_latency_ms"]) or actual["total_latency_ms"]<=0):raise ValueError("raw successful usage/model/target gate failed")
        sealed=next(item for item in seal["raw_files"]if item["path"]==str(rawpath))
        if sealed["sha256"]!=sha(rawpath):raise ValueError("sealed held-out raw hash mismatch")
        item=next(value for value in status["completed"]if value["stem"]==stem);parsed={};parse_errors=[];slice_meta={}
        for name,stage in (("prefill.log","prefill"),("decode.log","decode")):
            fullpath=old/name;data=fullpath.read_bytes();start=item["log_offsets_start"][name];end=item["log_offsets_end"][name]
            if not 0<=start<=end<=len(data):raise ValueError("recorded stage log range unavailable")
            chunk=data[start:end];target=slices/(stem+"."+name);target.write_bytes(chunk)
            parsed[stage],errors=parse_stats(chunk.decode(errors="replace"),stage);parse_errors.extend({"stage":stage,**error}for error in errors)
            slice_meta[stage]={"source_path":str(fullpath),"source_sha256":sha(fullpath),"start_byte":start,"end_byte":end,"slice_sha256":sha(target),"parsed_stats":len(parsed[stage])}
        paired,reasons=pair_stages(raw,parsed["prefill"],parsed["decode"],parse_errors)
        pairing.append({"dataset":dataset,"expected_n":expected_n,"paired_n":len(paired)if paired is not None else 0,"pairing_status":"exact_serial_order_plus_PD_room_and_usage_budget"if paired is not None else"UNAVAILABLE_NO_GUESS",
                        "reasons":reasons,"slices":slice_meta})
        observed={}if paired is None else{key:value for key,value in zip(keys,paired)}
        for actual in raw:
            key=(actual["workflow_id"],actual["call_id"])
            value={"dataset":dataset,"workflow_id":key[0],"call_id":key[1],"input_tokens":source[key]["input_len"],"output_tokens":source[key]["output_len"],
                   "observed_client_total_ms":actual["total_latency_ms"],"observed_client_ttft_ms":actual["ttft_ms"],
                   "stage_pairing_available":paired is not None,"raw_error":actual["error"],"usage_verified":True,**predictions[key],**observed.get(key,{})}
            for metric,(pred,obs) in METRICS.items():
                p=value.get(pred);a=value.get(obs);value[metric+"_signed_error_ms"]=p-a if p is not None and a is not None else None
                value[metric+"_absolute_percentage_error"]=abs(p-a)/abs(a)*100 if p is not None and a is not None and a>0 else None
            all_records.append(value)
    aggregates={}
    for dataset in ("sharegpt","hotpotqa","all"):
        subset=[r for r in all_records if dataset=="all"or r["dataset"]==dataset]
        expected_n=149 if dataset=="all"else EXPECTED[dataset]
        aggregates[dataset]={metric:error_statistics([(r.get(pred),r.get(obs))for r in subset],expected_n)for metric,(pred,obs)in METRICS.items()}
    result={"diagnostic_kind":parameters["diagnostic_kind"],"status":"COMPLETE_STAGE_AND_CLIENT_SERVICE_DIAGNOSTIC"if all(r["stage_pairing_available"]for r in all_records)else"CLIENT_SERVICE_DIAGNOSTIC_WITH_STAGE_PAIRING_MISSING",
            "counts":{"expected":149,"recorded":len(all_records),"successful_usage_verified":sum(r["usage_verified"]for r in all_records),"raw_errors":sum(bool(r["raw_error"])for r in all_records),
                      "paired_stage_calls":sum(r["stage_pairing_available"]for r in all_records),"missing_client_calls":149-len(all_records),"outliers_excluded":0},
            "aggregates":aggregates,"pairing":pairing,"input_sha256":input_hashes,"parameters_sha256":sha(out/"parameters.json"),
            "maximum_transfer_rounding_interval_width_ms":max((r.get("transfer_rounding_interval_width_ms",0)for r in all_records),default=None),
            "physical_profile_revision":OLD_REV,"native_profile_ELF_sha256":seal["elf_sha256"],"new_R2_scheduler_claim":False,"scheduler_or_simulator_run":False,"fitting_performed":False,
            "elapsed_cpu_driver_wall_s":time.perf_counter()-started,"elapsed_seconds_includes_source_import_and_io":True}
    write_json(out/"results.json",result)
    with(out/"per-call.jsonl").open("x")as target:
        for value in all_records:target.write(json.dumps(value,allow_nan=False)+"\n")
    columns=sorted(set().union(*(value.keys()for value in all_records)))
    with(out/"per-call.csv").open("x",newline="")as target:
        writer=csv.DictWriter(target,fieldnames=columns);writer.writeheader();writer.writerows(all_records)
    lines=["Fixed-parameter held-out stage/service diagnostic: 149 isolated calls (ShareGPT16, HotpotQA133), all observed outputs retained. No scheduler/native rollout or new fitting.\n",
           "Predictions use the unchanged actual online TP2 BF16 roofline, full Llama geometry, nominal H20 catalog4096e9 B/s and148e12 FP16 OPS, batch1. Approved transfer startup/BW are fixed. Negative bias means underprediction.\n",
           "| Dataset | Component / comparison | N | Bias ms | MAE ms | RMSE ms | MAPE % |","| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset,metrics in aggregates.items():
        for metric,stats in metrics.items():
            fmt=lambda x:"unknown"if x is None else f"{x:.3f}"
            lines.append(f"| {dataset} | {metric} | {stats['n']} | {fmt(stats['signed_bias_ms'])} | {fmt(stats['mae_ms'])} | {fmt(stats['rmse_ms'])} | {fmt(stats['mape_percent'])} |")
    lines.extend(["\nObserved Pprefix=logged P forward F minus rounded last-chunk T; T is sender-queue-to-TP-consensus surrogate, not pure DMA. D is logged decode forward; D transfer waiting is excluded. Additive observed service=F+D, so P/T reconstruction uncertainty cancels in that total. Client streaming total includes unmodeled transport/bootstrap/queue/framework overhead.",
                  "\nPairing uses the collector's serial client order plus matching P/D bootstrap room, input/output usage budgets, zero cached tokens and single-chunk prompts. Missing/ambiguous pairing is unknown, never guessed. Full reported rounding intervals and all per-call errors are preserved in CSV/JSONL.",
                  "\nPhysical profiles are old a65d6f9 native ELF a96b9c..., not R2 scheduler data. Two prior68-call calibration passes (v1 retained separately, v2 approved) preceded these149 held-out calls; this is not a cold-start sample. The estimator file is identical at a65d6f9,08885ec and the actual loaded source. This diagnostic does not validate scheduler decisions, workflow latency, pressure-case matching or runtime parity."])
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":result["status"],"counts":result["counts"],"aggregates":aggregates,"elapsed_wall_s":result["elapsed_cpu_driver_wall_s"]},indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in("source-root","old-run","trace-dir","approved-transfer","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())

if __name__=="__main__":main()
