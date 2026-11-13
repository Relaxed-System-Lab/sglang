"""Real planner inputs reach the stable pure core; no runtime/model is launched."""
import copy
import pytest
from test_hexgent_planner import _load


def snapshot(mode, *, slots=2, capacity=1000, running=()):
    return {"mode":mode,"limits":{"max_total_num_tokens":capacity,"max_running_requests":slots},
            "queues":{"waiting":[],"prealloc":[],"transfer":[],"retracted":[]},"running":list(running)}


def request(slots=2):
    schemas=_load("schemas")
    running={"rid":"HEALTH_CHECK-observed","input_tokens":200,"predicted_output_tokens":200,
             "max_new_tokens":200,"remaining_service_tokens":200,"memory_tokens":400,
             "estimated_decode_time":1000,"is_running":True}
    return schemas.PlannerScheduleRequest(route="/v1/chat/completions",model="m",request_text="public",
        max_tokens=200,hexgent=schemas.HexgentRequestMetadata(workflow_id="wf",call_id="new",
            input_tokens=200,predicted_output_tokens=200),
        prefill_workers=[schemas.WorkerDescriptor(id="p0",url="http://p0",worker_type="prefill",
            prefill_base_ms=0,prefill_ms_per_input_token=.01)],
        decode_workers=[schemas.WorkerDescriptor(id="d0",url="http://d0",worker_type="decode",
            decode_base_ms=0,decode_ms_per_output_token=.05),schemas.WorkerDescriptor(id="d1",
            url="http://d1",worker_type="decode",decode_base_ms=0,decode_ms_per_output_token=.5)],
        state=schemas.PlannerClusterState(current_time_ms=0,worker_snapshots={
            "http://p0":snapshot("prefill"),"http://d0":snapshot("decode",slots=slots,running=[running]),
            "http://d1":snapshot("decode",slots=1)}))


@pytest.mark.parametrize("slots,expected",[(2,"d0"),(1,"d1")])
def test_real_planner_parallel_slot_choice(slots, expected):
    planner=_load("planner").HexgentPlanner()
    result=planner.schedule_request(request(slots))
    assert result.decode_worker_id == expected


@pytest.mark.parametrize("probe_tokens", [1000, 999])
def test_worker_observed_prefix_changes_P_choice_without_changing_cold_horizon(probe_tokens):
    schemas,cache=_load("schemas"),_load("cache")
    probe={"probe_id":"wf/new","input_ids":list(range(probe_tokens)),"extra_key":None,"max_prefix_tokens":probe_tokens-1}
    inputs=request();inputs.hexgent.input_tokens=1000;inputs.hexgent.predicted_output_tokens=10;inputs.max_tokens=10
    inputs.prefill_workers=[schemas.WorkerDescriptor(id="p0",url="http://p0",worker_type="prefill",
        prefill_base_ms=0,prefill_ms_per_input_token=.5),schemas.WorkerDescriptor(id="p1",url="http://p1",
        worker_type="prefill",prefill_base_ms=0,prefill_ms_per_input_token=1.)]
    inputs.state.worker_snapshots={"http://p0":snapshot("prefill",capacity=10000),
        "http://p1":snapshot("prefill",capacity=10000),"http://d0":snapshot("decode",capacity=10000),
        "http://d1":snapshot("decode",capacity=10000)}
    for url,hit in [("http://p0",0),("http://p1",900)]:
        payload=inputs.state.worker_snapshots[url];payload['prefix_cache']={"enabled":True,"page_size":1}
        payload['prefix_observations']=[{"probe_id":probe['probe_id'],"probe_digest":cache.prefix_probe_digest(probe),
            "status":"observed","device_prefix_tokens":hit,"host_prefix_tokens":None,
            "storage_prefix_tokens":None,"source":"tree_cache.match_prefix","input_tokens":1000,"max_prefix_tokens":999}]
    cold=_load("planner").HexgentPlanner().schedule_request(copy.deepcopy(inputs))
    inputs.prefix_cache_probe=probe
    cached=_load("planner").HexgentPlanner().schedule_request(inputs)
    assert cold.prefill_worker_id == "p0"
    assert cached.prefill_worker_id == ("p1" if probe_tokens == 1000 else "p0")
    assert cached.workflow_horizon_ms == cold.workflow_horizon_ms


@pytest.mark.parametrize("batched", [False, True])
def test_waiting_P_and_D_mutations_compare_against_the_observed_revision(batched):
    schemas = _load("schemas")
    req = request()
    for url, queue, revision in [("http://p0", "waiting", 7), ("http://d1", "waiting", 11)]:
        req.state.worker_snapshots[url]["queues"][queue] = [{
            "rid": url + "/old", "workflow_id": url, "call_id": "old",
            "input_tokens": 10, "predicted_output_tokens": 10,
            "remaining_service_tokens": 10, "schedule_revision": revision}]
    planner = _load("planner").HexgentPlanner()
    if batched:
        planner.enqueue_admission(schemas.PlannerAdmissionRequest(request=req))
        result = planner.plan_pending_admissions()[("wf", "new")].schedule
    else:
        result = planner.schedule_request(req)
    actions = {action.request_id: action for action in result.actions}
    assert set(actions) == {"http://p0/old", "http://d1/old"}
    for url, revision in [("http://p0", 7), ("http://d1", 11)]:
        assert actions[url + "/old"].expected_revision == revision
        assert actions[url + "/old"].action_type == "priority_update"


def test_retracted_D_uses_remaining_decode_service_but_keeps_full_context_and_demand():
    schemas = _load("schemas")
    planner = _load("planner").HexgentPlanner()
    req = request()
    req.state.worker_snapshots["http://d1"]["queues"]["retracted"] = [{
        "rid": "rid", "workflow_id": "wf-old", "call_id": "old", "input_tokens": 100,
        "predicted_output_tokens": 100, "remaining_service_tokens": 10,
        "output_tokens_generated": 90, "memory_tokens": 200, "schedule_revision": 3}]
    record = next(row for row in _load("snapshots").decode_snapshot_calls(req) if row.call_id == "old")
    call = planner._decode_call_from_snapshot(record, req)
    assert call.input_tokens == 100 and call.output_tokens == 100
    assert call.decode_demand_tokens == 200
    assert call.decode_ms_by_instance == [1, 5]
    assert call.priority_seconds.decode_s_by_instance == pytest.approx((.0005, .005))


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("queue", ["prealloc", "transfer", "waiting"])
def test_final_D_reservation_includes_every_observed_nonrunning_release_job(batched, queue):
    schemas = _load("schemas")
    req = request()
    req.state.current_time_ms = 1000
    req.hexgent.workflow_arrival_ms = 1000
    req.prefill_workers[0].prefill_ms_per_input_token = .5
    req.decode_workers = req.decode_workers[:1]
    req.state.worker_snapshots.pop("http://d1")
    req.state.worker_snapshots["http://d0"] = snapshot("decode", slots=1, capacity=10000)
    req.state.worker_snapshots["http://d0"]["queues"][queue] = [{
        "rid": "old", "workflow_id": "old-wf", "call_id": "old",
        "workflow_arrival_ms": 0, "input_tokens": 10, "predicted_output_tokens": 1000,
        "remaining_service_tokens": 1000, "estimated_prefill_time": 1,
        "estimated_decode_time": 1000, "estimated_transfer_time": 10}]
    planner = _load("planner").HexgentPlanner()
    if batched:
        planner.enqueue_admission(schemas.PlannerAdmissionRequest(request=req))
        result = planner.plan_pending_admissions()[("wf", "new")].schedule
    else:
        result = planner.schedule_request(req)
    reservation = planner._reservations[("wf", "new")]
    assert reservation.decode_start_ms == (2000 if queue == "waiting" else 2010)
    if queue != "waiting":
        assert result.actions == []  # Preallocation/transfer is immutable once dispatched.
