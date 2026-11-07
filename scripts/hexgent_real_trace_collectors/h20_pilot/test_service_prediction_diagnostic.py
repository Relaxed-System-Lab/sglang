"""Synthetic CPU tests; no GPU, service calls, scheduler, fits or installs."""
import importlib.util
import math
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("service_prediction_diagnostic",ROOT/"service_prediction_diagnostic.py")
MOD=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=MOD;SPEC.loader.exec_module(MOD)

P="[TP0] ReqTimeStats(rid=p0, bootstrap_room=room0, input_len=100, cached_input_len=0, output_len=1, type=prefill): forward_duration=100.00ms, transfer_speed=1.00 GB/s, transfer_total=15.62 MB"
D="[TP0] ReqTimeStats(rid=d0, bootstrap_room=room0, input_len=100, cached_input_len=0, output_len=2, type=decode): transfer_duration=101.00ms, forward_duration=200.00ms"
RAW=[{"workflow_id":"wf","call_id":"call","input_len":100,"output_len":2,"total_latency_ms":320}]

def parsed():
    p,pe=MOD.parse_stats(P,"prefill");d,de=MOD.parse_stats(D,"decode")
    return p,d,pe+de

def test_logged_component_pairing_preserves_units_and_boundaries():
    p,d,errors=parsed();pairs,reasons=MOD.pair_stages(RAW,p,d,errors)
    assert not reasons and len(pairs)==1
    item=pairs[0]
    assert item["observed_transfer_surrogate_ms"]==pytest.approx(15.25390625)
    assert item["observed_prefill_prefix_ms"]==pytest.approx(84.74609375)
    assert item["observed_decode_forward_ms"]==200
    assert item["observed_additive_service_ms"]==300
    assert item["decode_transfer_wait_ms"]==101  # Excluded from additive total.
    assert item["transfer_interval_low_ms"]<item["observed_transfer_surrogate_ms"]<item["transfer_interval_high_ms"]
    assert item["prefill_prefix_interval_low_ms"]<item["observed_prefill_prefix_ms"]<item["prefill_prefix_interval_high_ms"]

@pytest.mark.parametrize("failure",["room","count","budget","cached","chunk"])
def test_stage_mismatch_is_unavailable_instead_of_guessed(failure):
    p,d,errors=parsed()
    if failure=="room":d[0]["room"]="wrong-room"
    elif failure=="count":p=[]
    elif failure=="budget":d[0]["output_tokens"]=3
    elif failure=="cached":p[0]["cached_tokens"]=1
    else:p[0]["input_tokens"]=8192
    pairs,reasons=MOD.pair_stages(RAW,p,d,errors)
    assert pairs is None and reasons

def test_unparsed_stats_are_not_silently_ignored():
    _,errors=MOD.parse_stats(P.replace("forward_duration=100.00ms","forward_duration=bad"),"prefill")
    assert errors

def test_fixed_transfer_uses_fullgroup_bytes_and_never_observations():
    class FixedNominal:
        def estimate_ms(self,input_tokens,output_tokens):return 10,20
    first=MOD.predict_components(FixedNominal(),100,2,5,327680000)
    assert first=={"predicted_prefill_prefix_ms":10,"predicted_decode_forward_ms":20,"predicted_transfer_surrogate_ms":105,"predicted_additive_service_ms":135}
    second=MOD.predict_components(FixedNominal(),100,2,5,327680000)
    assert first==second

def test_error_summary_keeps_outliers_and_reports_missing():
    result=MOD.error_statistics([(2,1),(1100,100),(None,2)],expected_n=4)
    assert result["n"]==2 and result["missing_n"]==2
    assert result["signed_bias_ms"]==pytest.approx(500.5)
    assert result["mae_ms"]==pytest.approx(500.5)
    assert result["rmse_ms"]==pytest.approx(math.sqrt((1+1000000)/2))
    assert result["mape_percent"]==pytest.approx(550)
    assert result["maximum_absolute_error_ms"]==1000
