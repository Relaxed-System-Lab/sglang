import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
import types


_ROOT = Path(__file__).resolve().parents[3]
_PKG_ROOT = _ROOT / "python"


def _load(module_name: str):
    if "sglang" not in sys.modules:
        package = types.ModuleType("sglang")
        package.__path__ = [str(_PKG_ROOT / "sglang")]
        sys.modules["sglang"] = package
    if "sglang.hexgent" not in sys.modules:
        package = types.ModuleType("sglang.hexgent")
        package.__path__ = [str(_PKG_ROOT / "sglang" / "hexgent")]
        sys.modules["sglang.hexgent"] = package
    module_path = _PKG_ROOT / "sglang" / "hexgent" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"sglang.hexgent.{module_name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_predictor_uses_request_prediction_then_max_tokens_then_text_fallback():
    predictor_mod = _load("predictor")
    predictor = predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)

    assert predictor.predict("hello", predicted_output_tokens=37, max_tokens=11) == 37
    assert predictor.predict("hello", predicted_output_tokens=None, max_tokens=11) == 11

    short = predictor.predict("one two three", predicted_output_tokens=None, max_tokens=None)
    long = predictor.predict(
        " ".join(["token"] * 200), predicted_output_tokens=None, max_tokens=None
    )
    assert short >= 1
    assert long > short


def test_decode_reservation_starts_after_prefill_and_kv_transfer():
    """A decode worker cannot consume KV before the transfer has completed."""
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=1)
    )
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="transfer-aware reservation",
        max_tokens=1,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-transfer-ready",
            call_id="call-0",
            workflow_arrival_ms=0,
            input_tokens=1,
            predicted_output_tokens=1,
            estimated_prefill_time=10,
            estimated_transfer_time=35,
            estimated_decode_time=20,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0", url="http://d0", worker_type="decode", capacity_tokens=100
            )
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )

    planner.schedule_request(request)

    reservation = planner._reservations[("wf-transfer-ready", "call-0")]
    assert reservation.prefill_end_ms == 10
    assert reservation.decode_start_ms == 45
    assert reservation.decode_end_ms == 65


def test_live_snapshot_replaces_matching_dispatched_reservation():
    """A dispatched call must not consume snapshot and reservation capacity twice."""
    planner_mod = _load("planner")
    schemas = _load("schemas")
    snapshots_mod = _load("snapshots")
    planner = planner_mod.HexgentPlanner()
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="new call",
        max_tokens=1,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-new",
            call_id="call-new",
            workflow_arrival_ms=0,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0", url="http://prefill0", worker_type="prefill"
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                capacity_tokens=1000,
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=0,
            worker_snapshots={
                "http://prefill0": {
                    "mode": "prefill",
                    "running": [
                        {
                            "workflow_id": "wf-live",
                            "call_id": "call-live",
                            "is_running": True,
                            "input_tokens": 10,
                            "estimated_prefill_time": 20.0,
                        }
                    ],
                },
                "http://decode0": {
                    "mode": "decode",
                    "running": [
                        {
                            "workflow_id": "wf-live",
                            "call_id": "call-live",
                            "is_running": True,
                            "input_tokens": 10,
                            "memory_tokens": 20,
                            "remaining_service_tokens": 2,
                            "max_new_tokens": 2,
                            "estimated_decode_time": 30.0,
                        }
                    ],
                },
            },
        ),
    )
    planner._reservations[("wf-live", "call-live")] = (
        planner_mod.PlannedCallReservation(
            workflow_id="wf-live",
            call_id="call-live",
            prefill_worker_id="p0",
            prefill_worker_url="http://prefill0",
            prefill_end_ms=100,
            decode_worker_id="d0",
            decode_worker_url="http://decode0",
            decode_start_ms=100,
            decode_end_ms=130,
            decode_demand_tokens=20,
        )
    )

    snapshots_mod.apply_worker_snapshots(request, include_mutable_waiting=False)

    assert planner._prefill_available(request) == [20]
    assert planner._decode_available(request) == [30]
    assert planner._active_prefill_reservation_counts(request) == [0]
    assert len(planner._decode_intervals(request)[0]) == 1


def test_prefill_snapshot_preserves_future_decode_reservation():
    """A call still in prefill must retain its planned future decode capacity."""
    planner_mod = _load("planner")
    schemas = _load("schemas")
    snapshots_mod = _load("snapshots")
    planner = planner_mod.HexgentPlanner()
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="new call",
        max_tokens=1,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-new",
            call_id="call-new",
            workflow_arrival_ms=0,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0", url="http://prefill0", worker_type="prefill"
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                capacity_tokens=1000,
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=0,
            worker_snapshots={
                "http://prefill0": {
                    "mode": "prefill",
                    "running": [
                        {
                            "workflow_id": "wf-live",
                            "call_id": "call-live",
                            "is_running": True,
                            "input_tokens": 10,
                            "estimated_prefill_time": 20.0,
                        }
                    ],
                }
            },
        ),
    )
    planner._reservations[("wf-live", "call-live")] = (
        planner_mod.PlannedCallReservation(
            workflow_id="wf-live",
            call_id="call-live",
            prefill_worker_id="p0",
            prefill_worker_url="http://prefill0",
            prefill_end_ms=100,
            decode_worker_id="d0",
            decode_worker_url="http://decode0",
            decode_start_ms=25,
            decode_end_ms=55,
            decode_demand_tokens=20,
        )
    )

    snapshots_mod.apply_worker_snapshots(request, include_mutable_waiting=False)

    assert planner._prefill_available(request) == [20]
    assert planner._decode_available(request) == [55]
    assert len(planner._decode_intervals(request)[0]) == 1


def test_live_snapshot_closes_pending_admission_after_lost_dispatch_event():
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=1)
    )
    workers = (
        [schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")],
        [schemas.WorkerDescriptor(id="d0", url="http://d0", worker_type="decode")],
    )

    def request(workflow_id, *, snapshots=None):
        return schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text=workflow_id,
            max_tokens=1,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id=workflow_id,
                call_id="call-0",
                workflow_arrival_ms=0,
                input_tokens=1,
                predicted_output_tokens=1,
            ),
            prefill_workers=workers[0],
            decode_workers=workers[1],
            state=schemas.PlannerClusterState(
                current_time_ms=0,
                worker_snapshots=snapshots or {},
            ),
        )

    dispatched = request("wf-dispatched")
    fresh = request(
        "wf-fresh",
        snapshots={
            "http://p0": {
                "mode": "prefill",
                "queues": {
                    "waiting": [
                        {
                            "workflow_id": "wf-dispatched",
                            "call_id": "call-0",
                            "request_id": "rid-dispatched",
                            "queue_name": "prefill_waiting",
                        }
                    ]
                },
                "running": [],
            }
        },
    )
    for item in (dispatched, fresh):
        planner.enqueue_admission(schemas.PlannerAdmissionRequest(request=item))

    results = planner.plan_pending_admissions()

    assert set(results) == {("wf-fresh", "call-0")}
    assert ("wf-dispatched", "call-0") not in planner._pending_admissions


def test_planner_derives_horizon_from_revealed_dependency_critical_path():
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=1)
    )

    def request(call_id, *, parents=(), prefill_ms=10, transfer_ms=5, decode_ms=20):
        return schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text=call_id,
            max_tokens=1,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf-dynamic-horizon",
                call_id=call_id,
                workflow_arrival_ms=0,
                parent_call_ids=list(parents),
                input_tokens=1,
                predicted_output_tokens=1,
                estimated_prefill_time=prefill_ms,
                estimated_transfer_time=transfer_ms,
                estimated_decode_time=decode_ms,
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")
            ],
            decode_workers=[
                schemas.WorkerDescriptor(id="d0", url="http://d0", worker_type="decode")
            ],
            state=schemas.PlannerClusterState(current_time_ms=0),
        )

    root_response = planner.schedule_request(request("root"))
    assert planner._workflows["wf-dynamic-horizon"].horizon_ms == 35
    assert root_response.workflow_horizon_ms == 35

    child_response = planner.schedule_request(
        request("child", parents=["root"], prefill_ms=7, transfer_ms=3, decode_ms=10)
    )
    assert planner._workflows["wf-dynamic-horizon"].horizon_ms == 55
    assert child_response.workflow_horizon_ms == 55

    planner.schedule_request(request("short-sibling", parents=["root"], prefill_ms=1, transfer_ms=1, decode_ms=1))
    assert planner._workflows["wf-dynamic-horizon"].horizon_ms == 55


def test_planner_uses_revealed_parent_path_for_current_call_pressure():
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=1)
    )

    def request(call_id, *, parents=(), prefill_ms=10, transfer_ms=5, decode_ms=20):
        return schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text=call_id,
            max_tokens=1,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf-revealed-parent-path",
                call_id=call_id,
                workflow_arrival_ms=0,
                parent_call_ids=list(parents),
                input_tokens=1,
                predicted_output_tokens=1,
                estimated_prefill_time=prefill_ms,
                estimated_transfer_time=transfer_ms,
                estimated_decode_time=decode_ms,
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")
            ],
            decode_workers=[
                schemas.WorkerDescriptor(id="d0", url="http://d0", worker_type="decode")
            ],
            state=schemas.PlannerClusterState(current_time_ms=0),
        )

    planner.schedule_request(request("root"))
    planner.schedule_request(
        request("sibling", parents=["root"], prefill_ms=7, transfer_ms=3, decode_ms=10)
    )

    workflow = planner._workflows["wf-revealed-parent-path"]
    assert planner._known_path_through_call_ms(workflow, "sibling") == 55


def test_planner_ignores_legacy_client_horizon_and_derives_its_own():
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=1)
    )
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="root",
        max_tokens=1,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-legacy-horizon",
            call_id="root",
            workflow_arrival_ms=0,
            workflow_horizon_ms=60_000,
            input_tokens=1,
            predicted_output_tokens=1,
            estimated_prefill_time=10,
            estimated_transfer_time=5,
            estimated_decode_time=20,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")
        ],
        decode_workers=[
            schemas.WorkerDescriptor(id="d0", url="http://d0", worker_type="decode")
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )

    planner.schedule_request(request)

    assert planner._workflows["wf-legacy-horizon"].horizon_ms == 35


def test_planner_keeps_wall_latency_out_of_horizon_but_uses_worker_service_telemetry():
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=1)
    )

    def schedule(workflow_id, call_id):
        planner.schedule_request(
            schemas.PlannerScheduleRequest(
                route="/v1/chat/completions",
                model="test-model",
                request_text=call_id,
                max_tokens=1,
                hexgent=schemas.HexgentRequestMetadata(
                    workflow_id=workflow_id,
                    call_id=call_id,
                    workflow_arrival_ms=100,
                    input_tokens=1,
                    predicted_output_tokens=1,
                    estimated_prefill_time=10,
                    estimated_transfer_time=5,
                    estimated_decode_time=20,
                ),
                prefill_workers=[
                    schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")
                ],
                decode_workers=[
                    schemas.WorkerDescriptor(id="d0", url="http://d0", worker_type="decode")
                ],
                state=schemas.PlannerClusterState(current_time_ms=100),
            )
        )

    schedule("wf-wall", "call-wall")
    planner.handle_event(
        schemas.PlannerEvent(
            event_type="call_completed",
            workflow_id="wf-wall",
            call_id="call-wall",
            service_ms=500,
            timestamp_ms=600,
        )
    )
    wall_state = planner._workflows["wf-wall"]
    assert wall_state.horizon_ms == 35
    assert wall_state.completion_ms == 500
    assert wall_state.attained_service_ms == 0

    schedule("wf-telemetry", "call-telemetry")
    planner.handle_event(
        schemas.PlannerEvent(
            event_type="call_completed",
            workflow_id="wf-telemetry",
            call_id="call-telemetry",
            service_ms=500,
            timestamp_ms=600,
            payload={"measured_service_ms": 47},
        )
    )
    telemetry_state = planner._workflows["wf-telemetry"]
    assert telemetry_state.horizon_ms == 47
    assert telemetry_state.completion_ms == 500
    assert telemetry_state.attained_service_ms == 47


def test_planner_uses_completed_critical_path_not_parallel_service_sum():
    """Parallel fan-out must not consume workflow service as serial work."""

    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=1)
    )

    def schedule(call_id, *, parents=(), prefill_ms=10, transfer_ms=5, decode_ms=20):
        planner.schedule_request(
            schemas.PlannerScheduleRequest(
                route="/v1/chat/completions",
                model="test-model",
                request_text=call_id,
                max_tokens=1,
                hexgent=schemas.HexgentRequestMetadata(
                    workflow_id="wf-fanout-attained",
                    call_id=call_id,
                    workflow_arrival_ms=0,
                    parent_call_ids=list(parents),
                    input_tokens=1,
                    predicted_output_tokens=1,
                    estimated_prefill_time=prefill_ms,
                    estimated_transfer_time=transfer_ms,
                    estimated_decode_time=decode_ms,
                ),
                prefill_workers=[
                    schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")
                ],
                decode_workers=[
                    schemas.WorkerDescriptor(id="d0", url="http://d0", worker_type="decode")
                ],
                state=schemas.PlannerClusterState(current_time_ms=0),
            )
        )

    def complete(call_id, service_ms):
        planner.handle_event(
            schemas.PlannerEvent(
                event_type="call_completed",
                workflow_id="wf-fanout-attained",
                call_id=call_id,
                measured_service_ms=service_ms,
                timestamp_ms=service_ms,
            )
        )

    schedule("root-a")  # Estimated service: 35 ms.
    schedule("root-b", prefill_ms=30)  # Estimated service: 55 ms.
    complete("root-a", 31)
    complete("root-b", 49)

    state = planner._workflows["wf-fanout-attained"]
    assert state.horizon_ms == 49
    assert state.attained_service_ms == 49

    schedule("join", parents=("root-a", "root-b"), prefill_ms=2, transfer_ms=3, decode_ms=4)
    complete("join", 7)

    # The join follows the slower branch: max(31, 49) + 7, not 31 + 49 + 7.
    assert state.horizon_ms == 56
    assert state.attained_service_ms == 56


def test_bert_proxy_predictor_supports_multiclass_bucket_outputs(monkeypatch):
    predictor_mod = _load("predictor")

    class FakeLogits:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *_shape):
            return self

        def tolist(self):
            return self.values

    class FakeOutput:
        logits = FakeLogits([0.1, 2.0, 0.3])

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, **_inputs):
            return FakeOutput()

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": [1, 2, 3]}

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_path):
            assert model_path == "proxy-model"
            return FakeTokenizer()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_path):
            assert model_path == "proxy-model"
            return FakeModel()

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoConfig=types.SimpleNamespace(),
            AutoModelForSequenceClassification=FakeAutoModel,
            AutoTokenizer=FakeAutoTokenizer,
            BertModel=types.SimpleNamespace(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(no_grad=lambda: FakeNoGrad()),
    )

    predictor = predictor_mod.BertProxyOutputLengthPredictor(
        "proxy-model",
        task_type="multiclass",
        class_values=[16, 64, 256],
    )

    assert predictor.predict("hello") == 64


def test_hybrid_latency_estimator_uses_roofline_kv_bandwidth_for_decode_total():
    schemas = _load("schemas")
    estimator_mod = _load("latency_estimator")

    model_config = {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "w_bit": 16,
        "a_bit": 16,
        "kv_bit": 16,
    }
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=4,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
            model_config=model_config,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0",
                url="http://prefill0",
                worker_type="prefill",
                hardware_params={
                    "bandwidth": 1_000_000_000_000,
                    "FP16": 1_000_000_000_000_000,
                    "onchip_buffer": 32_000_000,
                },
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d-slow",
                url="http://decode-slow",
                worker_type="decode",
                hardware_params={
                    "bandwidth": 50_000_000_000,
                    "FP16": 1_000_000_000_000_000,
                    "onchip_buffer": 32_000_000,
                },
            ),
            schemas.WorkerDescriptor(
                id="d-fast",
                url="http://decode-fast",
                worker_type="decode",
                hardware_params={
                    "bandwidth": 1_000_000_000_000,
                    "FP16": 1_000_000_000_000_000,
                    "onchip_buffer": 32_000_000,
                },
            ),
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )

    estimator = estimator_mod.HybridLatencyEstimator()
    one_token = estimator.estimate(request, input_tokens=4096, output_tokens=1)
    four_tokens = estimator.estimate(request, input_tokens=4096, output_tokens=4)

    assert one_token.prefill_ms_by_instance[0] > 0
    assert one_token.decode_ms_by_instance[0] > one_token.decode_ms_by_instance[1]
    assert four_tokens.decode_ms_by_instance == [
        value * 4 for value in one_token.decode_ms_by_instance
    ]


def test_roofline_transfer_calibration_includes_request_startup_overhead(monkeypatch):
    schemas = _load("schemas")
    estimator_mod = _load("latency_estimator")

    model_config = {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "w_bit": 16,
        "a_bit": 16,
        "kv_bit": 16,
    }
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=4,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf", call_id="call-0", workflow_arrival_ms=0,
            workflow_horizon_ms=1000, model_config=model_config,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://d0",
                worker_type="decode",
                hardware_params={
                    "bandwidth": 1_000_000_000_000,
                    "FP16": 1_000_000_000_000_000,
                    "onchip_buffer": 32_000_000,
                    "transfer_bandwidth_bytes_per_s": 7_000_000_000,
                },
            )
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )

    estimator = estimator_mod.HybridLatencyEstimator()
    baseline = estimator.estimate(request, input_tokens=4096, output_tokens=1)
    monkeypatch.setenv("HEXGENT_TRANSFER_STARTUP_MS", "10")
    calibrated = estimator.estimate(request, input_tokens=4096, output_tokens=1)

    assert calibrated.transfer_ms_by_decode_instance == [
        baseline.transfer_ms_by_decode_instance[0] + 10
    ]


def test_hybrid_latency_estimator_falls_back_to_worker_linear_labels():
    schemas = _load("schemas")
    estimator_mod = _load("latency_estimator")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=4,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0",
                url="http://prefill0",
                worker_type="prefill",
                prefill_base_ms=3.0,
                prefill_ms_per_input_token=2.0,
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                decode_base_ms=5.0,
                decode_ms_per_output_token=7.0,
                transfer_ms_per_token=0.25,
            )
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )

    estimate = estimator_mod.HybridLatencyEstimator().estimate(
        request, input_tokens=10, output_tokens=4
    )

    assert estimate.prefill_ms_by_instance == [23]
    assert estimate.decode_ms_by_instance == [33]
    assert estimate.transfer_ms_by_decode_instance == [2]


def test_hybrid_latency_estimator_caches_static_stages_without_staling_decode_batch(
    monkeypatch,
):
    schemas = _load("schemas")
    estimator_mod = _load("latency_estimator")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=8,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(id="p0", url="http://prefill0", worker_type="prefill")
        ],
        decode_workers=[
            schemas.WorkerDescriptor(id="d0", url="http://decode0", worker_type="decode")
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=0,
            decode_running_batch_sizes=[0],
        ),
    )
    estimator = estimator_mod.HybridLatencyEstimator()
    calls = {"prefill": 0, "decode": 0, "transfer": 0}

    def prefill(*_args, **_kwargs):
        calls["prefill"] += 1
        return 10

    def decode(*args, **kwargs):
        calls["decode"] += 1
        batch_size = kwargs.get("batch_size", args[-1])
        return 20 * batch_size

    def transfer(*_args, **_kwargs):
        calls["transfer"] += 1
        return 5

    monkeypatch.setattr(estimator, "_prefill_ms", prefill)
    monkeypatch.setattr(estimator, "_decode_ms", decode)
    monkeypatch.setattr(estimator, "_transfer_ms", transfer)

    first = estimator.estimate(request, input_tokens=128, output_tokens=16)
    second = estimator.estimate(request, input_tokens=128, output_tokens=16)

    assert first == second
    assert calls == {"prefill": 1, "decode": 1, "transfer": 1}

    request.state.decode_running_batch_sizes = [1]
    occupied = estimator.estimate(request, input_tokens=128, output_tokens=16)

    assert occupied.decode_ms_by_instance == [40]
    assert calls == {"prefill": 1, "decode": 2, "transfer": 1}


def test_snapshot_state_uses_live_worker_queue_availability():
    snapshots_mod = _load("snapshots")
    schemas = _load("schemas")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=32,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0",
                url="http://prefill0",
                worker_type="prefill",
                prefill_ms_per_input_token=2.0,
            ),
            schemas.WorkerDescriptor(
                id="p1",
                url="http://prefill1",
                worker_type="prefill",
                prefill_ms_per_input_token=2.0,
            ),
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                capacity_tokens=1000,
                decode_base_ms=0.0,
                decode_ms_per_output_token=10.0,
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=1000,
            worker_snapshots={
                "http://prefill0": {
                    "snapshots": [
                        {
                            "mode": "prefill",
                            "queues": {
                                "waiting": [
                                    {"input_tokens": 100, "memory_tokens": 100}
                                ]
                            },
                            "running": [
                                {"input_tokens": 50, "memory_tokens": 50}
                            ],
                        }
                    ]
                },
                "http://decode0": {
                    "snapshots": [
                        {
                            "mode": "decode",
                            "queues": {
                                "waiting": [
                                    {
                                        "remaining_service_tokens": 20,
                                        "memory_tokens": 200,
                                    }
                                ]
                            },
                            "running": [
                                {
                                    "remaining_service_tokens": 30,
                                    "memory_tokens": 300,
                                }
                            ],
                        }
                    ]
                },
            },
        ),
    )

    snapshots_mod.apply_worker_snapshots(request)

    assert request.state.prefill_available_at_ms[0] > request.state.prefill_available_at_ms[1]
    # Waiting requests already own KV capacity but do not add compute time to
    # the active continuous batch. Availability is governed by the longest
    # currently running sequence: 30 * 10 ms.
    assert request.state.decode_available_at_ms == [1300]
    assert request.state.decode_scheduled_intervals == [[(1000, 1300, 500)]]
    assert request.decode_workers[0].capacity_tokens == 500


def test_snapshot_state_tracks_only_running_decode_batch_occupancy():
    snapshots_mod = _load("snapshots")
    schemas = _load("schemas")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=32,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0", url="http://prefill0", worker_type="prefill"
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0", url="http://decode0", worker_type="decode"
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=1000,
            worker_snapshots={
                "http://decode0": {
                    "snapshots": [
                        {
                            "mode": "decode",
                            "queues": {
                                "transfer": [
                                    {
                                        "is_running": False,
                                        "remaining_service_tokens": 10,
                                    }
                                ],
                                "waiting": [
                                    {
                                        "is_running": False,
                                        "remaining_service_tokens": 10,
                                    }
                                ],
                            },
                            "running": [
                                {"is_running": True, "remaining_service_tokens": 20},
                                {"remaining_service_tokens": 30},
                            ],
                        }
                    ]
                }
            },
        ),
    )

    snapshots_mod.apply_worker_snapshots(request)

    assert request.state.decode_running_batch_sizes == [2]


def test_hybrid_roofline_decode_estimate_uses_live_running_batch_size():
    schemas = _load("schemas")
    estimator_mod = _load("latency_estimator")

    model_config = {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "w_bit": 16,
        "a_bit": 16,
        "kv_bit": 16,
    }
    hardware = {
        "bandwidth": 1_000_000_000_000,
        "FP16": 1_000_000_000_000_000,
        "onchip_buffer": 32_000_000,
    }

    def make_request(running_batch_size):
        return schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text="hello",
            max_tokens=8,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf",
                call_id="call-0",
                workflow_arrival_ms=0,
                workflow_horizon_ms=1000,
                model_config=model_config,
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(
                    id="p0",
                    url="http://prefill0",
                    worker_type="prefill",
                    hardware_params=hardware,
                )
            ],
            decode_workers=[
                schemas.WorkerDescriptor(
                    id="d0",
                    url="http://decode0",
                    worker_type="decode",
                    hardware_params=hardware,
                    decode_base_ms=3.0,
                    decode_ms_per_output_token=7.0,
                )
            ],
            state=schemas.PlannerClusterState(
                current_time_ms=0,
                decode_running_batch_sizes=[running_batch_size],
            ),
        )

    estimator = estimator_mod.HybridLatencyEstimator()
    idle = estimator.estimate(make_request(0), input_tokens=1024, output_tokens=8)
    occupied = estimator.estimate(make_request(2), input_tokens=1024, output_tokens=8)

    assert occupied.decode_ms_by_instance[0] > idle.decode_ms_by_instance[0]
    assert occupied.decode_ms_by_instance[0] < idle.decode_ms_by_instance[0] * 3


def test_hybrid_linear_decode_estimate_ignores_unmodeled_batch_size():
    schemas = _load("schemas")
    estimator_mod = _load("latency_estimator")

    def make_request(running_batch_size):
        return schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text="hello",
            max_tokens=8,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf",
                call_id="call-0",
                workflow_arrival_ms=0,
                workflow_horizon_ms=1000,
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(
                    id="p0", url="http://prefill0", worker_type="prefill"
                )
            ],
            decode_workers=[
                schemas.WorkerDescriptor(
                    id="d0",
                    url="http://decode0",
                    worker_type="decode",
                    decode_base_ms=3.0,
                    decode_ms_per_output_token=7.0,
                )
            ],
            state=schemas.PlannerClusterState(
                current_time_ms=0,
                decode_running_batch_sizes=[running_batch_size],
            ),
        )

    estimator = estimator_mod.HybridLatencyEstimator()
    idle = estimator.estimate(make_request(0), input_tokens=1024, output_tokens=8)
    occupied = estimator.estimate(make_request(2), input_tokens=1024, output_tokens=8)

    assert idle.decode_ms_by_instance == occupied.decode_ms_by_instance == [59]


def test_snapshot_state_accounts_for_transfer_queue_and_profiled_input_length():
    snapshots_mod = _load("snapshots")
    schemas = _load("schemas")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=32,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0", url="http://prefill0", worker_type="prefill"
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                capacity_tokens=1000,
                decode_base_ms=0.0,
                decode_ms_per_output_token=1.0,
                transfer_ms_per_token=2.0,
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=1000,
            worker_snapshots={
                "http://decode0": {
                    "snapshots": [
                        {
                            "mode": "decode",
                            "queues": {
                                "transfer": [
                                    {
                                        "queue_name": "decode_transfer",
                                        "hexgent_input_tokens": 100,
                                        "input_tokens": 1,
                                        "memory_tokens": 100,
                                        "remaining_service_tokens": 0,
                                    }
                                ]
                            },
                            "running": [],
                        }
                    ]
                }
            },
        ),
    )

    snapshots_mod.apply_worker_snapshots(request)

    assert request.state.decode_available_at_ms == [1200]
    assert request.decode_workers[0].capacity_tokens == 900


def test_snapshot_decode_capacity_uses_live_kv_pool_limit():
    snapshots_mod = _load("snapshots")
    schemas = _load("schemas")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=32,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(id="p0", url="http://prefill0", worker_type="prefill")
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                capacity_tokens=1_000_000,
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=1000,
            worker_snapshots={
                "http://decode0": {
                    "snapshot": {
                        "mode": "decode",
                        "limits": {
                            "max_total_num_tokens": 500,
                            "max_running_requests": 128,
                        },
                        "queues": {
                            "waiting": [
                                {"memory_tokens": 120, "remaining_service_tokens": 40}
                            ]
                        },
                        "running": [],
                    }
                }
            },
        ),
    )

    snapshots_mod.apply_worker_snapshots(request)

    assert request.decode_workers[0].capacity_tokens == 380


def test_profiled_absolute_latency_calibrates_hybrid_estimator():
    schemas = _load("schemas")
    estimator_mod = _load("latency_estimator")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=4,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
            estimated_prefill_time=17.0,
            estimated_decode_time=23.0,
            estimated_transfer_time=5.0,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0", url="http://prefill0", worker_type="prefill"
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0", url="http://decode0", worker_type="decode"
            )
        ],
    )

    estimate = estimator_mod.HybridLatencyEstimator().estimate(
        request, input_tokens=100, output_tokens=4
    )

    assert estimate.prefill_ms_by_instance == [17]
    assert estimate.decode_ms_by_instance == [23]
    assert estimate.transfer_ms_by_decode_instance == [5]


def test_snapshot_state_accepts_worker_endpoint_wrapper_payload():
    snapshots_mod = _load("snapshots")
    schemas = _load("schemas")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=32,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0",
                url="http://prefill0",
                worker_type="prefill",
                prefill_base_ms=0.0,
                prefill_ms_per_input_token=2.0,
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(id="d0", url="http://decode0", worker_type="decode")
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=1000,
            worker_snapshots={
                "http://prefill0": {
                    "snapshot": {
                        "mode": "prefill",
                        "queues": {
                            "waiting": [
                                {"input_tokens": 100, "memory_tokens": 100}
                            ]
                        },
                        "running": [],
                    }
                }
            },
        ),
    )

    snapshots_mod.apply_worker_snapshots(request)

    assert request.state.prefill_available_at_ms == [1200]


def test_snapshot_state_uses_profiled_stage_times_from_worker_items():
    snapshots_mod = _load("snapshots")
    schemas = _load("schemas")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=32,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0", url="http://prefill0", worker_type="prefill"
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0", url="http://decode0", worker_type="decode", capacity_tokens=1000
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=1000,
            worker_snapshots={
                "http://prefill0": {
                    "mode": "prefill",
                    "queues": {
                        "waiting": [
                            {
                                "input_tokens": 100,
                                "memory_tokens": 100,
                                "estimated_prefill_time": 37.0,
                            }
                        ]
                    },
                    "running": [],
                },
                "http://decode0": {
                    "mode": "decode",
                    "queues": {
                        "decode_transfer": [
                            {
                                "input_tokens": 100,
                                "memory_tokens": 100,
                                "estimated_transfer_time": 41.0,
                            }
                        ]
                    },
                    "running": [
                        {
                            "is_running": True,
                            "input_tokens": 100,
                            "memory_tokens": 100,
                            "remaining_service_tokens": 20,
                            "max_new_tokens": 20,
                            "estimated_decode_time": 53.0,
                        }
                    ],
                },
            },
        ),
    )

    snapshots_mod.apply_worker_snapshots(request)

    assert request.state.prefill_available_at_ms == [1037]


def test_snapshot_prefill_without_roofline_uses_conservative_serial_batch_work():
    snapshots_mod = _load("snapshots")
    schemas = _load("schemas")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=32,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0", url="http://prefill0", worker_type="prefill"
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(id="d0", url="http://decode0", worker_type="decode")
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=1000,
            worker_snapshots={
                "http://prefill0": {
                    "snapshot": {
                        "mode": "prefill",
                        "limits": {
                            "max_prefill_tokens": 300,
                            "prefill_max_requests": 2,
                        },
                        "queues": {
                            "waiting": [
                                {
                                    "input_tokens": 100,
                                    "memory_tokens": 100,
                                    "estimated_prefill_time": 40,
                                },
                                {
                                    "input_tokens": 100,
                                    "memory_tokens": 100,
                                    "estimated_prefill_time": 60,
                                },
                            ]
                        },
                        "running": [],
                    }
                }
            },
        ),
    )

    snapshots_mod.apply_worker_snapshots(request)

    # Without model/hardware metadata, the snapshot cannot estimate the
    # batching gain. It must keep the calibrated serial total rather than
    # underestimating the batch as its longest member.
    assert request.state.prefill_available_at_ms == [1100]


def test_snapshot_decode_transfer_queue_models_overlap_not_serial_sum():
    snapshots_mod = _load("snapshots")
    schemas = _load("schemas")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=32,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(id="p0", url="http://prefill0", worker_type="prefill")
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                transfer_ms_per_token=1.0,
                capacity_tokens=10_000,
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=1000,
            worker_snapshots={
                "http://decode0": {
                    "mode": "decode",
                    "queues": {
                        "decode_transfer": [
                            {"input_tokens": 100, "memory_tokens": 100},
                            {"input_tokens": 200, "memory_tokens": 200},
                        ]
                    },
                    "running": [],
                }
            },
        ),
    )

    snapshots_mod.apply_worker_snapshots(request)

    # Two transfers are concurrent; the worker is blocked by the longest one,
    # not by their sum.
    assert request.state.decode_available_at_ms == [1200]


def test_snapshot_decode_transfer_pipeline_respects_inflight_cap(monkeypatch):
    monkeypatch.setenv("SGLANG_DISAGGREGATION_MAX_INFLIGHT_TRANSFERS", "1")
    snapshots_mod = _load("snapshots")
    schemas = _load("schemas")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=32,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(id="p0", url="http://prefill0", worker_type="prefill")
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                transfer_ms_per_token=1.0,
                capacity_tokens=10_000,
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=1000,
            worker_snapshots={
                "http://decode0": {
                    "mode": "decode",
                    "queues": {
                        "prealloc": [
                            {"input_tokens": 100, "memory_tokens": 100},
                            {"input_tokens": 200, "memory_tokens": 200},
                        ],
                        "transfer": [
                            {"input_tokens": 300, "memory_tokens": 300},
                        ],
                    },
                    "running": [],
                }
            },
        ),
    )

    snapshots_mod.apply_worker_snapshots(request)

    # The runtime allows one transfer at a time, so prealloc and transfer
    # entries form a serialized 600 ms pipeline.
    assert request.state.decode_available_at_ms == [1600]


def test_snapshot_decode_capacity_uses_longest_active_sequence():
    snapshots_mod = _load("snapshots")
    schemas = _load("schemas")

    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=32,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(id="p0", url="http://prefill0", worker_type="prefill")
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                decode_base_ms=10.0,
                decode_ms_per_output_token=2.0,
                capacity_tokens=10_000,
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=1000,
            worker_snapshots={
                "http://decode0": {
                    "mode": "decode",
                    "queues": {},
                    "running": [
                        {
                            "is_running": True,
                            "input_tokens": 100,
                            "memory_tokens": 100,
                            "remaining_service_tokens": 50,
                        },
                        {
                            "is_running": True,
                            "input_tokens": 100,
                            "memory_tokens": 100,
                            "remaining_service_tokens": 20,
                        },
                    ],
                }
            },
        ),
    )

    snapshots_mod.apply_worker_snapshots(request)

    # The running batch completes when the longest sequence completes: the
    # fallback estimate is 10 + 2 * 50 = 110 ms, not the sum of both rows.
    assert request.state.decode_available_at_ms == [1110]


def test_roofline_latency_estimator_uses_model_and_hardware_limits():
    latency_mod = _load("latency")
    schemas = _load("schemas")

    model_config = {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
    }
    slow_worker = schemas.WorkerDescriptor(
        id="slow",
        url="http://slow",
        worker_type="prefill",
        model_config=model_config,
        hardware_params={
            "bandwidth": 900e9,
            "FP16": 100e12,
            "onchip_buffer": 20e6,
        },
        tp_size=4,
    )
    fast_worker = schemas.WorkerDescriptor(
        id="fast",
        url="http://fast",
        worker_type="prefill",
        model_config=model_config,
        hardware_params={
            "bandwidth": 3_000e9,
            "FP16": 900e12,
            "onchip_buffer": 32e6,
        },
        tp_size=4,
    )
    estimator = latency_mod.RooflineLatencyEstimator()

    slow_prefill = estimator.prefill_ms(slow_worker, input_tokens=512, output_tokens=64)
    fast_prefill = estimator.prefill_ms(fast_worker, input_tokens=512, output_tokens=64)
    slow_decode = estimator.decode_ms(slow_worker, input_tokens=512, output_tokens=64)
    fast_decode = estimator.decode_ms(fast_worker, input_tokens=512, output_tokens=64)

    assert fast_prefill < slow_prefill
    assert fast_decode < slow_decode
    assert estimator.decode_ms(fast_worker, input_tokens=512, output_tokens=128) > fast_decode
    assert (
        estimator.decode_iteration_ms(fast_worker, sequence_length=1024, batch_size=1)
        > estimator.decode_iteration_ms(fast_worker, sequence_length=128, batch_size=1)
    )
    expected_long_decode = sum(
        estimator.decode_iteration_ms(fast_worker, sequence_length=512 + offset, batch_size=1)
        for offset in range(200)
    )
    assert (
        estimator.decode_ms(fast_worker, input_tokens=512, output_tokens=200)
        == expected_long_decode
    )


def test_roofline_resolves_h100_sxm_from_worker_label_without_request_hardware():
    hardware_mod = _load("hardware")
    latency_mod = _load("latency_estimator")
    schemas = _load("schemas")

    worker = schemas.WorkerDescriptor(
        id="h100",
        url="http://h100",
        worker_type="prefill",
        hardware="H100_TP1",
    )

    profile = hardware_mod.resolve_hardware_profile(worker.hardware)
    assert profile is not None
    assert profile["memory_bandwidth_bytes_per_s"] == 3.072e12
    assert latency_mod._worker_hardware_params(worker) == profile


def test_prefill_snapshot_uses_roofline_batch_time_not_max_or_serial_sum():
    schemas = _load("schemas")
    latency_mod = _load("latency_estimator")
    snapshots_mod = _load("snapshots")

    model_config = {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "vocab_size": 128256,
        "w_bit": 8,
        "a_bit": 16,
        "kv_bit": 16,
    }
    worker = schemas.WorkerDescriptor(
        id="prefill0",
        url="http://prefill0",
        worker_type="prefill",
        hardware="H100_TP1",
        model_config=model_config,
        prefill_batch_token_limit=2048,
        prefill_batch_request_limit=2,
    )
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="llama-8b",
        request_text="new request",
        max_tokens=64,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call",
            workflow_arrival_ms=0,
            workflow_horizon_ms=10_000,
            input_tokens=512,
            predicted_output_tokens=64,
            model_config=model_config,
        ),
        prefill_workers=[worker],
        decode_workers=[],
        state=schemas.PlannerClusterState(
            current_time_ms=1000,
            worker_snapshots={
                "http://prefill0": {
                    "mode": "prefill",
                    "limits": {
                        "max_prefill_tokens": 2048,
                        "prefill_max_requests": 2,
                    },
                    "queues": {
                        "prefill_waiting": [
                            {"input_tokens": 512},
                            {"input_tokens": 512},
                        ]
                    },
                    "running": [],
                }
            },
        ),
    )

    snapshots_mod.apply_worker_snapshots(request)

    estimator = latency_mod.RooflineLatencyEstimator()
    single_ms = estimator.prefill_ms(worker, input_tokens=512, request=request)
    batch_ms = estimator.prefill_batch_ms(
        worker,
        input_tokens=[512, 512],
        request=request,
    )
    availability_delta = request.state.prefill_available_at_ms[0] - 1000
    # Integer millisecond rounding can add one millisecond to the packed batch.
    assert single_ms < batch_ms <= 2 * single_ms + 1
    assert availability_delta == batch_ms


def test_planner_uses_roofline_latency_when_worker_has_model_and_hardware_params():
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")
    latency_mod = _load("latency")

    model_config = {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
    }
    slow_hardware = {
        "bandwidth": 900e9,
        "FP16": 100e12,
        "onchip_buffer": 20e6,
    }
    fast_hardware = {
        "bandwidth": 3_000e9,
        "FP16": 900e12,
        "onchip_buffer": 32e6,
    }
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64),
        latency_estimator=latency_mod.RooflineLatencyEstimator(),
    )
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello",
        max_tokens=64,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-roofline",
            call_id="call-roofline",
            workflow_arrival_ms=0,
            workflow_horizon_ms=10000,
            input_tokens=512,
            predicted_output_tokens=64,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p-slow",
                url="http://prefill-slow",
                worker_type="prefill",
                model_config=model_config,
                hardware_params=slow_hardware,
                tp_size=4,
                prefill_base_ms=0.0,
                prefill_ms_per_input_token=1.0,
            ),
            schemas.WorkerDescriptor(
                id="p-fast",
                url="http://prefill-fast",
                worker_type="prefill",
                model_config=model_config,
                hardware_params=fast_hardware,
                tp_size=4,
                prefill_base_ms=0.0,
                prefill_ms_per_input_token=10.0,
            ),
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d-slow",
                url="http://decode-slow",
                worker_type="decode",
                model_config=model_config,
                hardware_params=slow_hardware,
                tp_size=4,
                decode_base_ms=0.0,
                decode_ms_per_output_token=1.0,
                capacity_tokens=100000,
            ),
            schemas.WorkerDescriptor(
                id="d-fast",
                url="http://decode-fast",
                worker_type="decode",
                model_config=model_config,
                hardware_params=fast_hardware,
                tp_size=4,
                decode_base_ms=0.0,
                decode_ms_per_output_token=10.0,
                capacity_tokens=100000,
            ),
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )

    response = planner.schedule_request(request)

    assert response.prefill_worker_id == "p-fast"
    assert response.decode_worker_id == "d-fast"


def test_planner_replans_waiting_prefill_requests_from_snapshots(monkeypatch):
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)
    )
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="new request",
        max_tokens=64,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-new",
            call_id="call-new",
            workflow_arrival_ms=0,
            workflow_horizon_ms=10000,
            reveal_sequence=1,
            input_tokens=100,
            predicted_output_tokens=64,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0",
                url="http://prefill0",
                worker_type="prefill",
                prefill_base_ms=0.0,
                prefill_ms_per_input_token=1.0,
            ),
            schemas.WorkerDescriptor(
                id="p1",
                url="http://prefill1",
                worker_type="prefill",
                prefill_base_ms=0.0,
                prefill_ms_per_input_token=1.0,
            ),
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                decode_base_ms=0.0,
                decode_ms_per_output_token=1.0,
                capacity_tokens=10000,
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=0,
            worker_snapshots={
                "http://prefill0": {
                    "snapshot": {
                        "mode": "prefill",
                        "instance_id": "p0",
                        "queues": {
                            "waiting": [
                                {
                                    "rid": "rid-existing",
                                    # The same call ID in a different workflow
                                    # must remain independently schedulable.
                                    "call_id": "call-new",
                                    "workflow_id": "wf-existing",
                                    "queue_name": "prefill_waiting",
                                    "current_assignment": "p0",
                                    "schedule_revision": 0,
                                    "workflow_arrival_ms": 0,
                                    "workflow_horizon_ms": 10000,
                                    "reveal_sequence": 0,
                                    "workflow_completion_ms": 0,
                                    "workflow_attained_service_ms": 0,
                                    "input_tokens": 100,
                                    "predicted_output_tokens": 64,
                                    "max_new_tokens": 64,
                                    "remaining_service_tokens": 64,
                                }
                            ]
                        },
                        "running": [],
                    }
                }
            },
        ),
    )

    response = planner.schedule_request(request)

    action_by_request = {action.request_id: action for action in response.actions}
    assert "rid-existing" in action_by_request
    assert action_by_request["rid-existing"].worker_url == "http://prefill0"
    assert action_by_request["rid-existing"].target_queue_name == "prefill_waiting"
    assert action_by_request["rid-existing"].schedule_revision == response.revision
    assert action_by_request["rid-existing"].action_type == "priority_update"
    assert action_by_request["rid-existing"].target_worker_url == "http://prefill0"

    batch_planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)
    )
    batch_planner.enqueue_admission(
        schemas.PlannerAdmissionRequest(
            request=request,
            policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
        )
    )
    second_request = replace(
        request,
        request_text="second pending call",
        hexgent=replace(
            request.hexgent,
            workflow_id="wf-second",
            # Call IDs are commonly reused across workflows; identity is the
            # composite (workflow_id, call_id).
            call_id="call-new",
        ),
    )
    batch_planner.enqueue_admission(
        schemas.PlannerAdmissionRequest(
            request=second_request,
            policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
        )
    )
    batch_results = batch_planner.plan_pending_admissions()
    assert set(batch_results) == {
        ("wf-new", "call-new"),
        ("wf-second", "call-new"),
    }
    batch_response = batch_results[("wf-new", "call-new")]
    assert any(
        action.request_id == "rid-existing"
        and action.action_type == "priority_update"
        for action in batch_response.schedule.actions
    )
    assert all(result.schedule.actions for result in batch_results.values())


def test_planner_mutation_actions_skip_started_prefill_and_carry_target_revision(monkeypatch):
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)
    )
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="new request",
        max_tokens=64,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-new",
            call_id="call-new",
            workflow_arrival_ms=0,
            workflow_horizon_ms=10000,
            reveal_sequence=1,
            input_tokens=32,
            predicted_output_tokens=64,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0",
                url="http://prefill0",
                worker_type="prefill",
                prefill_base_ms=0.0,
                prefill_ms_per_input_token=10.0,
            ),
            schemas.WorkerDescriptor(
                id="p1",
                url="http://prefill1",
                worker_type="prefill",
                prefill_base_ms=0.0,
                prefill_ms_per_input_token=1.0,
            ),
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                decode_base_ms=0.0,
                decode_ms_per_output_token=1.0,
                capacity_tokens=10000,
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=0,
            worker_snapshots={
                "http://prefill0": {
                    "snapshot": {
                        "mode": "prefill",
                        "instance_id": "p0",
                        "queues": {
                            "bootstrap": [
                                {
                                    "rid": "rid-bootstrap",
                                    "call_id": "call-bootstrap",
                                    "workflow_id": "wf-bootstrap",
                                    "queue_name": "prefill_bootstrap",
                                    "schedule_revision": 2,
                                    "input_tokens": 64,
                                    "predicted_output_tokens": 64,
                                }
                            ],
                            "waiting": [
                                {
                                    "rid": "rid-waiting",
                                    "call_id": "call-waiting",
                                    "workflow_id": "wf-waiting",
                                    "queue_name": "prefill_waiting",
                                    "schedule_revision": 4,
                                    "workflow_arrival_ms": 0,
                                    "workflow_horizon_ms": 10000,
                                    "input_tokens": 64,
                                    "predicted_output_tokens": 64,
                                },
                                {
                                    "rid": "rid-started",
                                    "call_id": "call-started",
                                    "workflow_id": "wf-started",
                                    "queue_name": "prefill_waiting",
                                    "schedule_revision": 9,
                                    "workflow_arrival_ms": 0,
                                    "workflow_horizon_ms": 10000,
                                    "input_tokens": 64,
                                    "predicted_output_tokens": 64,
                                    "prefill_started": True,
                                },
                            ]
                        },
                        "running": [],
                    }
                }
            },
        ),
    )

    response = planner.schedule_request(request)

    action_by_request = {action.request_id: action for action in response.actions}
    assert "rid-started" not in action_by_request
    assert "rid-bootstrap" not in action_by_request
    waiting_action = action_by_request["rid-waiting"]
    assert waiting_action.schedule_revision == response.revision
    # Monotonic revisions let overlapping N/N+1 updates converge to N+1.
    assert waiting_action.expected_revision is None
    assert waiting_action.action_type == "priority_update"
    assert waiting_action.target_worker_id == "p0"
    assert waiting_action.target_worker_url == "http://prefill0"
    assert waiting_action.planned_decode_instance is None


def test_worker_mutation_action_defaults_to_safe_priority_update():
    schemas = _load("schemas")

    action = schemas.WorkerMutationAction.from_dict(
        {
            "worker_url": "http://prefill0",
            "request_id": "rid-0",
            "target_queue_name": "prefill_waiting",
            "priority": 7,
        }
    )

    assert action.action_type == "priority_update"


def test_planner_can_disable_safe_local_priority_updates(monkeypatch):
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)
    )
    monkeypatch.setenv("HEXGENT_ENABLE_LOCAL_PRIORITY_UPDATES", "0")
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="new request",
        max_tokens=64,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-new",
            call_id="call-new",
            workflow_arrival_ms=0,
            workflow_horizon_ms=10_000,
            input_tokens=32,
            predicted_output_tokens=64,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0",
                url="http://prefill0",
                worker_type="prefill",
                prefill_ms_per_input_token=1.0,
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                decode_ms_per_output_token=1.0,
                capacity_tokens=10_000,
            )
        ],
        state=schemas.PlannerClusterState(
            current_time_ms=0,
            worker_snapshots={
                "http://prefill0": {
                    "snapshot": {
                        "mode": "prefill",
                        "queues": {
                            "waiting": [
                                {
                                    "rid": "rid-dispatched",
                                    "call_id": "call-dispatched",
                                    "workflow_id": "wf-dispatched",
                                    "queue_name": "prefill_waiting",
                                    "schedule_revision": 0,
                                    "workflow_arrival_ms": 0,
                                    "workflow_horizon_ms": 10_000,
                                    "input_tokens": 32,
                                    "predicted_output_tokens": 64,
                                }
                            ]
                        },
                        "running": [],
                    }
                }
            },
        ),
    )

    assert planner.schedule_request(request).actions == []


def test_exact_scheduler_prefill_jointly_picks_prefill_and_planned_decode():
    scheduler_mod = _load("scheduler")
    scheduler = scheduler_mod.AgenticPDExactScheduler()
    scheduler.on_workflow_arrival("wf", arrival_ms=0, horizon_ms=1000)

    calls = [
        scheduler_mod.PrefillCallInfo(
            call_id="c0",
            workflow_id="wf",
            input_tokens=128,
            output_tokens=64,
            workflow_arrival_ms=0,
            reveal_sequence=0,
            workflow_horizon_ms=1000,
            workflow_completion_ms=0,
            workflow_attained_service_ms=0,
            ideal_service_ms=100,
            prefill_ms_by_instance=[90, 10],
            decode_ms_by_instance=[10, 80],
            transfer_ms_by_decode_instance=[0, 0],
            critical_path_ms=100,
            child_count=0,
        )
    ]
    state = scheduler_mod.PrefillStageState(
        current_time_ms=0,
        instance_available_at_ms=[0, 0],
        decode_available_hint_ms=[0, 0],
    )

    decisions = scheduler.schedule_prefill(calls, state)

    assert len(decisions) == 1
    assert decisions[0].call_id == "c0"
    assert decisions[0].assigned_instance == 1
    assert decisions[0].planned_decode_instance == 0


def test_planner_uses_roofline_latency_estimator_when_model_and_hardware_are_provided():
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=8)
    )
    model_config = {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "w_bit": 16,
        "a_bit": 16,
        "kv_bit": 16,
    }
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello world",
        max_tokens=8,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-roofline",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=100000,
            input_tokens=4096,
            predicted_output_tokens=8,
            model_config=model_config,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0",
                url="http://prefill0",
                worker_type="prefill",
                hardware_params={
                    "bandwidth": 1_000_000_000_000,
                    "FP16": 1_000_000_000_000_000,
                    "onchip_buffer": 32_000_000,
                },
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d-slow",
                url="http://decode-slow",
                worker_type="decode",
                decode_base_ms=0.0,
                decode_ms_per_output_token=1.0,
                hardware_params={
                    "bandwidth": 50_000_000_000,
                    "FP16": 1_000_000_000_000_000,
                    "onchip_buffer": 32_000_000,
                },
            ),
            schemas.WorkerDescriptor(
                id="d-fast",
                url="http://decode-fast",
                worker_type="decode",
                decode_base_ms=0.0,
                decode_ms_per_output_token=1.0,
                hardware_params={
                    "bandwidth": 1_000_000_000_000,
                    "FP16": 1_000_000_000_000_000,
                    "onchip_buffer": 32_000_000,
                },
            ),
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )

    response = planner.schedule_request(request)

    assert response.decode_worker_id == "d-fast"


def test_planner_keeps_linear_worker_label_fallback_without_roofline_config():
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=16)
    )
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello world",
        max_tokens=16,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-linear",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=10000,
            input_tokens=100,
            predicted_output_tokens=16,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p-slow",
                url="http://prefill-slow",
                worker_type="prefill",
                prefill_base_ms=0.0,
                prefill_ms_per_input_token=10.0,
            ),
            schemas.WorkerDescriptor(
                id="p-fast",
                url="http://prefill-fast",
                worker_type="prefill",
                prefill_base_ms=0.0,
                prefill_ms_per_input_token=1.0,
            ),
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                decode_base_ms=0.0,
                decode_ms_per_output_token=1.0,
                capacity_tokens=10000,
            )
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )

    response = planner.schedule_request(request)

    assert response.prefill_worker_id == "p-fast"


def test_planner_builds_schedule_response_with_worker_urls_and_priorities():
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)
    )
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="hello world",
        max_tokens=32,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=1000,
            reveal_sequence=0,
            critical_path_ms=100,
            child_count=0,
            input_tokens=128,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0", url="http://prefill0", worker_type="prefill"
            ),
            schemas.WorkerDescriptor(
                id="p1", url="http://prefill1", worker_type="prefill"
            ),
        ],
        decode_workers=[
            schemas.WorkerDescriptor(id="d0", url="http://decode0", worker_type="decode"),
            schemas.WorkerDescriptor(id="d1", url="http://decode1", worker_type="decode"),
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )

    response = planner.schedule_request(request)

    assert response.revision > 0
    assert response.prefill_worker_url in {"http://prefill0", "http://prefill1"}
    assert response.decode_worker_url in {"http://decode0", "http://decode1"}
    assert isinstance(response.prefill_priority, int)
    assert isinstance(response.decode_priority, int)
    assert response.predicted_output_tokens == 32
    assert response.planned_decode_instance in {"d0", "d1"}


def test_planner_revision_epoch_advances_across_sidecar_restart(monkeypatch, tmp_path):
    planner_mod = _load("planner")

    epochs = iter((1_000_000, 2_000_000))
    monkeypatch.setattr(planner_mod.time, "time_ns", lambda: next(epochs))

    state_path = tmp_path / "revision"
    first = planner_mod.HexgentPlanner(revision_state_path=state_path)
    second = planner_mod.HexgentPlanner(revision_state_path=state_path)

    assert second.revision > first.revision


def test_planner_updates_horizon_from_revealed_service_not_client_value():
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)
    )

    def make_request(call_id, horizon_ms):
        return schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text="hello world",
            max_tokens=16,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf-online",
                call_id=call_id,
                workflow_arrival_ms=0,
                workflow_horizon_ms=horizon_ms,
                input_tokens=32,
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(
                    id="p0", url="http://prefill0", worker_type="prefill"
                )
            ],
            decode_workers=[
                schemas.WorkerDescriptor(
                    id="d0", url="http://decode0", worker_type="decode"
                )
            ],
            state=schemas.PlannerClusterState(current_time_ms=0),
        )

    planner.schedule_request(make_request("call-0", 1000))
    first_horizon = planner._workflows["wf-online"].horizon_ms
    planner.schedule_request(make_request("call-1", 2500))

    assert first_horizon > 0
    assert planner._workflows["wf-online"].horizon_ms == first_horizon
    assert planner.scheduler.workflow_horizons["wf-online"] == first_horizon


def test_planner_reserves_planned_prefill_and_decode_until_completion_event():
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)
    )

    def make_request(call_id):
        return schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text="hello world",
            max_tokens=None,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf-reserve",
                call_id=call_id,
                workflow_arrival_ms=0,
                workflow_horizon_ms=10000,
                input_tokens=100,
                predicted_output_tokens=100,
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(
                    id="p0",
                    url="http://prefill0",
                    worker_type="prefill",
                    prefill_base_ms=0.0,
                    prefill_ms_per_input_token=10.0,
                ),
                schemas.WorkerDescriptor(
                    id="p1",
                    url="http://prefill1",
                    worker_type="prefill",
                    prefill_base_ms=0.0,
                    prefill_ms_per_input_token=10.0,
                ),
            ],
            decode_workers=[
                schemas.WorkerDescriptor(
                    id="d0",
                    url="http://decode0",
                    worker_type="decode",
                    capacity_tokens=200,
                    decode_base_ms=0.0,
                    decode_ms_per_output_token=20.0,
                ),
                schemas.WorkerDescriptor(
                    id="d1",
                    url="http://decode1",
                    worker_type="decode",
                    capacity_tokens=200,
                    decode_base_ms=0.0,
                    decode_ms_per_output_token=20.0,
                ),
            ],
            state=schemas.PlannerClusterState(current_time_ms=0),
        )

    first = planner.schedule_request(make_request("call-0"))
    second = planner.schedule_request(make_request("call-1"))

    assert first.prefill_worker_id == "p0"
    assert first.decode_worker_id == "d0"
    assert second.prefill_worker_id == "p1"
    assert second.decode_worker_id == "d1"
    first_reservation = planner._reservations[("wf-reserve", "call-0")]
    assert first_reservation.decode_start_ms >= first_reservation.prefill_end_ms

    planner.handle_event(
        schemas.PlannerEvent(
            event_type="call_completed",
            workflow_id="wf-reserve",
            call_id="call-0",
            service_ms=2000,
            timestamp_ms=2000,
        )
    )
    third = planner.schedule_request(make_request("call-2"))

    assert third.prefill_worker_id == "p0"
    assert third.decode_worker_id == "d0"


def test_admission_plans_every_pending_request_independent_of_router_credit():
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)
    )

    def make_request(call_id):
        return schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text=f"request {call_id}",
            max_tokens=None,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf-admit",
                call_id=call_id,
                workflow_arrival_ms=0,
                workflow_horizon_ms=10000,
                input_tokens=100,
                predicted_output_tokens=64,
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(
                    id="p0",
                    url="http://prefill0",
                    worker_type="prefill",
                    prefill_base_ms=0.0,
                    prefill_ms_per_input_token=10.0,
                )
            ],
            decode_workers=[
                schemas.WorkerDescriptor(
                    id="d0",
                    url="http://decode0",
                    worker_type="decode",
                    decode_base_ms=0.0,
                    decode_ms_per_output_token=10.0,
                    capacity_tokens=10000,
                )
            ],
            state=schemas.PlannerClusterState(current_time_ms=0),
        )

    first = planner.admit_request(
        schemas.PlannerAdmissionRequest(
            request=make_request("call-0"),
            policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
        )
    )
    second = planner.admit_request(
        schemas.PlannerAdmissionRequest(
            request=make_request("call-1"),
            policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
        )
    )

    assert first.dispatch_ready is True
    assert first.schedule is not None
    assert second.dispatch_ready is True
    assert second.schedule is not None
    assert second.queue_position == 1
    assert ("wf-admit", "call-0") in planner._pending_admissions
    assert ("wf-admit", "call-1") in planner._pending_admissions

    planner.handle_event(
        schemas.PlannerEvent(
            event_type="admission_cancelled",
            workflow_id="wf-admit",
            call_id="call-1",
            payload={"reason": "test_cancel"},
        )
    )
    assert ("wf-admit", "call-1") not in planner._pending_admissions

    second = planner.admit_request(
        schemas.PlannerAdmissionRequest(
            request=make_request("call-1"),
            policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
        )
    )
    assert second.dispatch_ready is True

    planner.handle_event(
        schemas.PlannerEvent(
            event_type="call_completed",
            workflow_id="wf-admit",
            call_id="call-0",
            service_ms=1000,
            timestamp_ms=1000,
        )
    )
    released = planner.admit_request(
        schemas.PlannerAdmissionRequest(
            request=make_request("call-1"),
            policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
        )
    )

    assert released.dispatch_ready is True
    assert released.schedule is not None
    assert released.schedule.prefill_worker_id == "p0"


def test_prefill_completion_event_releases_only_prefill_admission_capacity():
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=8)
    )

    def request(call_id):
        return schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text=call_id,
            max_tokens=8,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf-prefill-event",
                call_id=call_id,
                workflow_arrival_ms=0,
                input_tokens=8,
                predicted_output_tokens=8,
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(
                    id="p0",
                    url="http://prefill0",
                    worker_type="prefill",
                    prefill_base_ms=100.0,
                )
            ],
            decode_workers=[
                schemas.WorkerDescriptor(
                    id="d0",
                    url="http://decode0",
                    worker_type="decode",
                    decode_base_ms=1_000.0,
                    capacity_tokens=10000,
                )
            ],
            state=schemas.PlannerClusterState(current_time_ms=0),
        )

    policy = schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1)
    first = planner.admit_request(
        schemas.PlannerAdmissionRequest(request=request("call-0"), policy=policy)
    )
    planned = planner.admit_request(
        schemas.PlannerAdmissionRequest(request=request("call-1"), policy=policy)
    )
    assert first.dispatch_ready is True
    assert planned.dispatch_ready is True
    first_revision = planned.schedule.revision

    planner.handle_event(
        schemas.PlannerEvent(
            event_type="request_dispatched",
            workflow_id="wf-prefill-event",
            call_id="call-0",
            timestamp_ms=1,
        )
    )
    planner.handle_event(
        schemas.PlannerEvent(
            event_type="prefill_completed",
            workflow_id="wf-prefill-event",
            call_id="call-0",
            timestamp_ms=25,
        )
    )
    planner.plan_pending_admissions()

    reservation = planner._reservations[("wf-prefill-event", "call-0")]
    assert reservation.prefill_end_ms == 25
    assert reservation.decode_end_ms > 25

    replanned = planner.admission_status(request("call-1"))
    assert replanned.dispatch_ready is True
    assert replanned.schedule is not None
    assert replanned.schedule.revision > first_revision


def test_request_cancellation_releases_a_prefill_completed_decode_reservation():
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=8)
    )
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="request",
        max_tokens=8,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-cancel",
            call_id="call-0",
            workflow_arrival_ms=0,
            input_tokens=8,
            predicted_output_tokens=8,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0",
                url="http://prefill0",
                worker_type="prefill",
                prefill_base_ms=100.0,
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                decode_base_ms=1_000.0,
                capacity_tokens=10000,
            )
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )
    planner.admit_request(schemas.PlannerAdmissionRequest(request=request))
    planner.handle_event(
        schemas.PlannerEvent(
            event_type="prefill_completed",
            workflow_id="wf-cancel",
            call_id="call-0",
            timestamp_ms=25,
        )
    )
    assert ("wf-cancel", "call-0") in planner._reservations

    planner.handle_event(
        schemas.PlannerEvent(
            event_type="request_cancelled",
            workflow_id="wf-cancel",
            call_id="call-0",
            timestamp_ms=30,
        )
    )

    assert ("wf-cancel", "call-0") not in planner._reservations


def test_admission_reuses_the_snapshot_it_already_applied(monkeypatch):
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)
    )
    snapshot_calls = []

    def record_snapshot(request, *, include_mutable_waiting=True):
        snapshot_calls.append(include_mutable_waiting)

    monkeypatch.setattr(planner_mod, "maybe_apply_live_worker_snapshots", record_snapshot)
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="request",
        max_tokens=None,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-admit-once",
            call_id="call-0",
            workflow_arrival_ms=0,
            workflow_horizon_ms=10000,
            input_tokens=100,
            predicted_output_tokens=64,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(
                id="p0",
                url="http://prefill0",
                worker_type="prefill",
                prefill_base_ms=0.0,
                prefill_ms_per_input_token=10.0,
            )
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0",
                url="http://decode0",
                worker_type="decode",
                decode_base_ms=0.0,
                decode_ms_per_output_token=10.0,
                capacity_tokens=10000,
            )
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )

    response = planner.admit_request(
        schemas.PlannerAdmissionRequest(
            request=request,
            policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
        )
    )

    assert response.dispatch_ready is True
    assert snapshot_calls == [False]


def test_admission_queue_uses_deterministic_tiebreak_without_client_horizon():
    planner_mod = _load("planner")
    schemas = _load("schemas")
    predictor_mod = _load("predictor")

    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)
    )

    def make_request(call_id, horizon_ms):
        return schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text=f"request {call_id}",
            max_tokens=None,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id=f"wf-{call_id}",
                call_id=call_id,
                workflow_arrival_ms=0,
                workflow_horizon_ms=horizon_ms,
                input_tokens=100,
                predicted_output_tokens=64,
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(
                    id="p0",
                    url="http://prefill0",
                    worker_type="prefill",
                    prefill_base_ms=0.0,
                    prefill_ms_per_input_token=10.0,
                )
            ],
            decode_workers=[
                schemas.WorkerDescriptor(
                    id="d0",
                    url="http://decode0",
                    worker_type="decode",
                    decode_base_ms=0.0,
                    decode_ms_per_output_token=10.0,
                    capacity_tokens=10000,
                )
            ],
            state=schemas.PlannerClusterState(current_time_ms=0),
        )

    policy = schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=0)
    first = planner.admit_request(
        schemas.PlannerAdmissionRequest(request=make_request("slow", 10000), policy=policy)
    )
    urgent = planner.admit_request(
        schemas.PlannerAdmissionRequest(request=make_request("urgent", 1000), policy=policy)
    )

    assert first.dispatch_ready is True
    assert urgent.dispatch_ready is True

    release_policy = schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1)
    urgent_release = planner.admit_request(
        schemas.PlannerAdmissionRequest(
            request=make_request("urgent", 1000),
            policy=release_policy,
        )
    )
    slow_release = planner.admit_request(
        schemas.PlannerAdmissionRequest(
            request=make_request("slow", 10000),
            policy=release_policy,
        )
    )

    assert urgent_release.dispatch_ready is True
    assert slow_release.dispatch_ready is True
    assert slow_release.schedule is not None
    latest_urgent = planner.admission_status(make_request("urgent", 1000))
    latest_slow = planner.admission_status(make_request("slow", 10000))
    # Client-provided horizon is intentionally ignored; equal revealed work
    # uses the scheduler's stable call-id tiebreak.
    assert latest_slow.queue_position < latest_urgent.queue_position


def test_sidecar_exposes_health_schedule_and_event_endpoints():
    import pytest

    pytest.importorskip("fastapi")
    sidecar_mod = _load("sidecar")
    predictor_mod = _load("predictor")

    app = sidecar_mod.build_app(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=64)
    )

    from fastapi.testclient import TestClient

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    payload = {
        "route": "/v1/chat/completions",
        "model": "test-model",
        "request_text": "hello world",
        "max_tokens": 12,
        "hexgent": {
            "workflow_id": "wf-sidecar",
            "call_id": "call-0",
            "workflow_arrival_ms": 0,
            "workflow_horizon_ms": 1000,
            "input_tokens": 10,
        },
        "prefill_workers": [
            {"id": "p0", "url": "http://prefill0", "worker_type": "prefill"}
        ],
        "decode_workers": [
            {"id": "d0", "url": "http://decode0", "worker_type": "decode"}
        ],
        "state": {"current_time_ms": 0},
    }
    scheduled = client.post("/schedule_request", json=payload)
    assert scheduled.status_code == 200
    assert scheduled.json()["prefill_worker_url"] == "http://prefill0"
    assert scheduled.json()["decode_worker_url"] == "http://decode0"
    assert scheduled.json()["predicted_output_tokens"] == 12

    admit_payload = {
        "request": {
            **payload,
            "hexgent": {
                **payload["hexgent"],
                "workflow_id": "wf-sidecar-admit",
                "call_id": "call-admit-0",
            },
            "state": {"current_time_ms": 100000},
        },
        "policy": {"max_prefill_queue_per_worker": 1, "retry_after_ms": 5},
    }
    admitted = client.post("/admit_request", json=admit_payload)
    assert admitted.status_code == 200
    assert admitted.json()["dispatch_ready"] is True
    assert admitted.json()["schedule"]["prefill_worker_url"] == "http://prefill0"
    assert set(admitted.json()["phase_timings_ms"]) == {
        "snapshot",
        "pending",
        "prefill_rank",
        "schedule",
        "total",
    }
    assert set(admitted.json()["schedule"]["phase_timings_ms"]) == {
        "prefill_queue_model",
        "prefill_solver",
        "decode_queue_model",
        "decode_solver",
        "reservation",
        "total",
    }

    event = client.post(
        "/event",
        json={
            "event_type": "call_completed",
            "workflow_id": "wf-sidecar",
            "call_id": "call-0",
            "service_ms": 25,
            "timestamp_ms": 50,
        },
    )
    assert event.status_code == 200
    assert event.json()["status"] == "ok"


def test_sidecar_telemetry_records_each_planner_invocation(tmp_path):
    import json
    import pytest

    pytest.importorskip("fastapi")
    sidecar_mod = _load("sidecar")
    predictor_mod = _load("predictor")
    telemetry_path = tmp_path / "planner_invocations.jsonl"
    app = sidecar_mod.build_app(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=8),
        telemetry=sidecar_mod.PlannerTelemetry(telemetry_path),
    )

    from fastapi.testclient import TestClient

    payload = {
        "route": "/v1/chat/completions",
        "model": "test-model",
        "request_text": "hello",
        "max_tokens": 8,
        "hexgent": {
            "workflow_id": "wf-telemetry",
            "call_id": "call-0",
            "workflow_horizon_ms": 1000,
            "input_tokens": 4,
        },
        "prefill_workers": [
            {"id": "p0", "url": "http://prefill0", "worker_type": "prefill"}
        ],
        "decode_workers": [
            {"id": "d0", "url": "http://decode0", "worker_type": "decode"}
        ],
        "state": {"current_time_ms": 0},
    }
    client = TestClient(app)

    assert client.post("/schedule_request", json=payload).status_code == 200
    assert client.post(
        "/admit_request",
        json={"request": payload, "policy": {"max_prefill_queue_per_worker": 1}},
    ).status_code == 200

    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert [row["endpoint"] for row in rows] == [
        "/schedule_request",
        "/admit_request",
    ]
    assert {row["workflow_id"] for row in rows} == {"wf-telemetry"}
    assert {row["call_id"] for row in rows} == {"call-0"}
    assert all(row["wall_ms"] >= 0 for row in rows)
    assert all(row["cpu_ms"] >= 0 for row in rows)
    assert all(row["process_thread_count"] >= 1 for row in rows)
    assert rows[0]["prefill_worker_url"] == "http://prefill0"
    assert rows[0]["decode_worker_url"] == "http://decode0"
    assert {
        "snapshot",
        "pending",
        "prefill_rank",
        "schedule",
        "total",
    }.issubset(rows[1]["phase_timings_ms"])
    assert set(rows[0]["phase_timings_ms"]) == {
        "prefill_queue_model",
        "prefill_solver",
        "decode_queue_model",
        "decode_solver",
        "reservation",
        "total",
    }
    assert "schedule.decode_solver" in rows[1]["phase_timings_ms"]


def test_admission_coordinator_records_solver_batch_telemetry(tmp_path):
    import json

    sidecar_mod = _load("sidecar")
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    telemetry_path = tmp_path / "planner_invocations.jsonl"
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=8)
    )
    coordinator = sidecar_mod.AdmissionCoordinator(
        planner, telemetry=sidecar_mod.PlannerTelemetry(telemetry_path)
    )
    admission = schemas.PlannerAdmissionRequest(
        request=schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text="queued",
            max_tokens=8,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf-batch-telemetry", call_id="call-0", input_tokens=8
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(
                    id="p0", url="http://prefill0", worker_type="prefill"
                )
            ],
            decode_workers=[
                schemas.WorkerDescriptor(
                    id="d0", url="http://decode0", worker_type="decode"
                )
            ],
            state=schemas.PlannerClusterState(current_time_ms=0),
        ),
        policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
    )

    coordinator.enqueue(admission)
    assert coordinator.plan_next_batch() == 1

    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["endpoint"] == "/plan_pending_admissions"
    assert row["admission_count"] == 1
    assert row["event_count"] == 0
    assert row["result_count"] == 1
    assert row["wall_ms"] >= 0
    assert row["cpu_ms"] >= 0


def test_admission_coordinator_coalesces_pending_requests_without_solving_on_enqueue(
    monkeypatch,
):
    """Admissions are queued cheaply; one actor drains the newest pending set."""
    sidecar_mod = _load("sidecar")
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=8)
    )
    coordinator = sidecar_mod.AdmissionCoordinator(planner)
    solver_calls = []
    original = planner.plan_pending_admissions

    def record_solver_call():
        solver_calls.append("solve")
        return original()

    monkeypatch.setattr(planner, "plan_pending_admissions", record_solver_call)

    def admission(call_id):
        return schemas.PlannerAdmissionRequest(
            request=schemas.PlannerScheduleRequest(
                route="/v1/chat/completions",
                model="test-model",
                request_text=call_id,
                max_tokens=8,
                hexgent=schemas.HexgentRequestMetadata(
                    workflow_id="wf-coalesced",
                    call_id=call_id,
                    workflow_arrival_ms=0,
                    input_tokens=8,
                    predicted_output_tokens=8,
                ),
                prefill_workers=[
                    schemas.WorkerDescriptor(
                        id="p0", url="http://prefill0", worker_type="prefill"
                    )
                ],
                decode_workers=[
                    schemas.WorkerDescriptor(
                        id="d0", url="http://decode0", worker_type="decode"
                    )
                ],
                state=schemas.PlannerClusterState(current_time_ms=0),
            ),
            policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=2),
        )

    first = admission("call-0")
    second = admission("call-1")
    assert coordinator.enqueue(first).reason == "planning_pending"
    assert coordinator.enqueue(second).reason == "planning_pending"
    assert solver_calls == []
    assert coordinator.status(first).reason == "planning_pending"
    assert solver_calls == []

    coordinator.plan_next_batch()

    assert solver_calls == ["solve"]
    assert coordinator.status(first).dispatch_ready is True
    assert coordinator.status(second).dispatch_ready is True
    revision_plan = __import__("asyncio").run(
        coordinator.wait_for_revision_plan(timeout_ms=1)
    )
    assert {
        (entry.workflow_id, entry.call_id) for entry in revision_plan.entries
    } == {("wf-coalesced", "call-0"), ("wf-coalesced", "call-1")}

    # Router retries after a concrete plan must keep using that plan rather
    # than clearing the cache and making the request wait for another solve.
    assert coordinator.enqueue(first).dispatch_ready is True
    assert coordinator.status(first).dispatch_ready is True
    assert solver_calls == ["solve"]

    # A prior execution plan only belongs to requests that it explicitly
    # contains. A newly revealed call remains pending until the next batched
    # solve publishes its own plan; the router may use native PD fallback in
    # the meantime, but it must never inherit another call's assignment.
    newly_revealed = coordinator.enqueue(admission("call-2"))
    assert newly_revealed.dispatch_ready is False
    assert newly_revealed.reason == "planning_pending"
    assert solver_calls == ["solve"]


def test_pending_admission_batch_runs_decode_capacity_solver_once():
    """A coalesced batch must not re-solve decode capacity per selected call."""
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=32)
    )

    for index in range(4):
        planner.enqueue_admission(
            schemas.PlannerAdmissionRequest(
                request=schemas.PlannerScheduleRequest(
                    route="/v1/chat/completions",
                    model="test-model",
                    request_text=f"queued-{index}",
                    max_tokens=32,
                    hexgent=schemas.HexgentRequestMetadata(
                        workflow_id=f"wf-batch-{index}",
                        call_id=f"call-{index}",
                        workflow_arrival_ms=0,
                        input_tokens=32,
                        predicted_output_tokens=32,
                    ),
                    prefill_workers=[
                        schemas.WorkerDescriptor(
                            id="p0", url="http://prefill0", worker_type="prefill"
                        ),
                        schemas.WorkerDescriptor(
                            id="p1", url="http://prefill1", worker_type="prefill"
                        ),
                    ],
                    decode_workers=[
                        schemas.WorkerDescriptor(
                            id="d0",
                            url="http://decode0",
                            worker_type="decode",
                            capacity_tokens=10_000,
                        ),
                        schemas.WorkerDescriptor(
                            id="d1",
                            url="http://decode1",
                            worker_type="decode",
                            capacity_tokens=10_000,
                        ),
                    ],
                    state=schemas.PlannerClusterState(current_time_ms=0),
                ),
                policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=4),
            )
        )

    results = planner.plan_pending_admissions()

    assert len(results) == 4
    assert all(result.dispatch_ready for result in results.values())
    assert planner.scheduler.get_stats()["decode_invocations"] == 1


def test_pending_admission_batch_waits_for_kv_transfer_before_decode_reservation():
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=1)
    )
    request = schemas.PlannerScheduleRequest(
        route="/v1/chat/completions",
        model="test-model",
        request_text="queued transfer-aware reservation",
        max_tokens=1,
        hexgent=schemas.HexgentRequestMetadata(
            workflow_id="wf-batch-transfer-ready",
            call_id="call-0",
            workflow_arrival_ms=0,
            input_tokens=1,
            predicted_output_tokens=1,
            estimated_prefill_time=10,
            estimated_transfer_time=35,
            estimated_decode_time=20,
        ),
        prefill_workers=[
            schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")
        ],
        decode_workers=[
            schemas.WorkerDescriptor(
                id="d0", url="http://d0", worker_type="decode", capacity_tokens=100
            )
        ],
        state=schemas.PlannerClusterState(current_time_ms=0),
    )
    planner.enqueue_admission(
        schemas.PlannerAdmissionRequest(
            request=request,
            policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
        )
    )

    result = planner.plan_pending_admissions()[("wf-batch-transfer-ready", "call-0")]

    assert result.dispatch_ready is True
    reservation = planner._reservations[("wf-batch-transfer-ready", "call-0")]
    assert reservation.prefill_end_ms == 10
    assert reservation.decode_start_ms == 45
    assert reservation.decode_end_ms == 65


def test_admission_waiter_is_released_by_plan_publication_event(monkeypatch):
    """Admission waiting is signalled by a published plan, not a poll timer."""
    import asyncio
    import threading

    sidecar_mod = _load("sidecar")
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=8)
    )
    coordinator = sidecar_mod.AdmissionCoordinator(planner)
    entered_solver = threading.Event()
    release_solver = threading.Event()
    original = planner.plan_pending_admissions

    def delayed_plan():
        entered_solver.set()
        assert release_solver.wait(timeout=1)
        return original()

    monkeypatch.setattr(planner, "plan_pending_admissions", delayed_plan)
    admission = schemas.PlannerAdmissionRequest(
        request=schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text="queued",
            max_tokens=8,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf-wait", call_id="call-0", input_tokens=8
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(
                    id="p0", url="http://prefill0", worker_type="prefill"
                )
            ],
            decode_workers=[
                schemas.WorkerDescriptor(
                    id="d0", url="http://decode0", worker_type="decode"
                )
            ],
            state=schemas.PlannerClusterState(current_time_ms=0),
        ),
        policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
    )

    async def scenario():
        actor = asyncio.create_task(coordinator.run(coalesce_ms=0))
        await asyncio.sleep(0)
        coordinator.enqueue(admission)
        coordinator.wake()
        assert await asyncio.to_thread(entered_solver.wait, 1)

        waiter = asyncio.create_task(
            coordinator.wait_for_status(admission, timeout_ms=1_000)
        )
        await asyncio.sleep(0.01)
        assert not waiter.done()

        release_solver.set()
        result = await waiter
        assert result.dispatch_ready is True
        assert result.schedule is not None

        coordinator.stop()
        await actor

    asyncio.run(scenario())


def test_admission_waiter_requires_a_newer_plan_revision():
    import asyncio

    sidecar_mod = _load("sidecar")
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=8)
    )
    coordinator = sidecar_mod.AdmissionCoordinator(planner)
    admission = schemas.PlannerAdmissionRequest(
        request=schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text="queued",
            max_tokens=8,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf-revision", call_id="call-0", input_tokens=8
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(
                    id="p0", url="http://prefill0", worker_type="prefill"
                )
            ],
            decode_workers=[
                schemas.WorkerDescriptor(
                    id="d0", url="http://decode0", worker_type="decode"
                )
            ],
            state=schemas.PlannerClusterState(current_time_ms=0),
        )
    )

    async def scenario():
        actor = asyncio.create_task(coordinator.run(coalesce_ms=0))
        await asyncio.sleep(0)
        coordinator.enqueue(admission)
        coordinator.wake()
        first = await coordinator.wait_for_status(admission, timeout_ms=1_000)
        first_revision = first.schedule.revision
        revision_plan = await coordinator.wait_for_revision_plan(timeout_ms=1_000)
        assert revision_plan.revision == first_revision
        assert [
            (entry.workflow_id, entry.call_id) for entry in revision_plan.entries
        ] == [("wf-revision", "call-0")]

        unchanged = await coordinator.wait_for_status(
            admission, timeout_ms=5, after_revision=first_revision
        )
        assert unchanged.reason == "revision_not_updated"

        coordinator.enqueue_event(
            schemas.PlannerEvent(event_type="cluster_state_changed", timestamp_ms=1)
        )
        coordinator.wake()
        updated = await coordinator.wait_for_status(
            admission, timeout_ms=1_000, after_revision=first_revision
        )
        assert updated.schedule.revision > first_revision
        updated_plan = await coordinator.wait_for_revision_plan(
            timeout_ms=1_000, after_revision=first_revision
        )
        assert updated_plan.revision == updated.schedule.revision

        coordinator.stop()
        await actor

    asyncio.run(scenario())


def test_prefill_completion_event_replans_and_wakes_revision_waiter():
    import asyncio

    sidecar_mod = _load("sidecar")
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=8)
    )
    coordinator = sidecar_mod.AdmissionCoordinator(planner)

    def admission(call_id):
        return schemas.PlannerAdmissionRequest(
            request=schemas.PlannerScheduleRequest(
                route="/v1/chat/completions",
                model="test-model",
                request_text=call_id,
                max_tokens=8,
                hexgent=schemas.HexgentRequestMetadata(
                    workflow_id="wf-prefill-wait",
                    call_id=call_id,
                    workflow_arrival_ms=0,
                    input_tokens=8,
                    predicted_output_tokens=8,
                ),
                prefill_workers=[
                    schemas.WorkerDescriptor(
                        id="p0",
                        url="http://prefill0",
                        worker_type="prefill",
                        prefill_base_ms=100.0,
                    )
                ],
                decode_workers=[
                    schemas.WorkerDescriptor(
                        id="d0",
                        url="http://decode0",
                        worker_type="decode",
                        decode_base_ms=1_000.0,
                        capacity_tokens=10_000,
                    )
                ],
                state=schemas.PlannerClusterState(current_time_ms=0),
            ),
            policy=schemas.PlannerAdmissionPolicy(max_prefill_queue_per_worker=1),
        )

    async def scenario():
        actor = asyncio.create_task(coordinator.run(coalesce_ms=0))
        await asyncio.sleep(0)
        first = admission("call-0")
        coordinator.enqueue(first)
        coordinator.wake()
        assert (await coordinator.wait_for_status(first, timeout_ms=1_000)).dispatch_ready

        second = admission("call-1")
        coordinator.enqueue(second)
        coordinator.wake()
        initial = await coordinator.wait_for_status(second, timeout_ms=1_000)
        waiter = asyncio.create_task(
            coordinator.wait_for_status(
                second,
                timeout_ms=1_000,
                after_revision=initial.schedule.revision,
            )
        )

        coordinator.enqueue_event(
            schemas.PlannerEvent(
                event_type="prefill_completed",
                workflow_id="wf-prefill-wait",
                call_id="call-0",
                timestamp_ms=25,
            )
        )
        coordinator.wake()
        assert (await waiter).dispatch_ready

        coordinator.stop()
        await actor

    asyncio.run(scenario())


def test_completion_event_reconstructs_fallback_call_for_future_workflow_horizon():
    planner_mod = _load("planner")
    predictor_mod = _load("predictor")
    schemas = _load("schemas")
    planner = planner_mod.HexgentPlanner(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=1)
    )

    planner.handle_event(
        schemas.PlannerEvent(
            event_type="call_completed",
            workflow_id="wf-fallback",
            call_id="root",
            measured_service_ms=40,
            timestamp_ms=40,
            payload={
                "hexgent_metadata": {
                    "workflow_id": "wf-fallback",
                    "call_id": "root",
                    "workflow_arrival_ms": 0,
                    "input_tokens": 1,
                    "predicted_output_tokens": 1,
                }
            },
        )
    )

    planner.schedule_request(
        schemas.PlannerScheduleRequest(
            route="/v1/chat/completions",
            model="test-model",
            request_text="child",
            max_tokens=1,
            hexgent=schemas.HexgentRequestMetadata(
                workflow_id="wf-fallback",
                call_id="child",
                workflow_arrival_ms=0,
                parent_call_ids=["root"],
                input_tokens=1,
                predicted_output_tokens=1,
                estimated_prefill_time=5,
                estimated_transfer_time=5,
                estimated_decode_time=5,
            ),
            prefill_workers=[
                schemas.WorkerDescriptor(id="p0", url="http://p0", worker_type="prefill")
            ],
            decode_workers=[
                schemas.WorkerDescriptor(id="d0", url="http://d0", worker_type="decode")
            ],
            state=schemas.PlannerClusterState(current_time_ms=40),
        )
    )

    assert planner._workflows["wf-fallback"].horizon_ms == 55


def test_sidecar_enqueue_and_status_use_the_background_admission_actor():
    import pytest

    pytest.importorskip("fastapi")
    sidecar_mod = _load("sidecar")
    predictor_mod = _load("predictor")
    from fastapi.testclient import TestClient

    payload = {
        "request": {
            "route": "/v1/chat/completions",
            "model": "test-model",
            "request_text": "queued request",
            "max_tokens": 8,
            "hexgent": {
                "workflow_id": "wf-async-admission",
                "call_id": "call-0",
                "workflow_arrival_ms": 0,
                "input_tokens": 8,
                "predicted_output_tokens": 8,
            },
            "prefill_workers": [
                {"id": "p0", "url": "http://prefill0", "worker_type": "prefill"}
            ],
            "decode_workers": [
                {"id": "d0", "url": "http://decode0", "worker_type": "decode"}
            ],
            "state": {"current_time_ms": 0},
        },
        "policy": {"max_prefill_queue_per_worker": 1, "retry_after_ms": 1},
    }
    app = sidecar_mod.build_app(
        predictor=predictor_mod.DeterministicOutputLengthPredictor(default_tokens=8)
    )

    with TestClient(app) as client:
        enqueued = client.post("/enqueue_admission", json=payload)
        assert enqueued.status_code == 200
        assert enqueued.json()["dispatch_ready"] is False

        status = client.get(
            "/wait_admission",
            params={
                "workflow_id": "wf-async-admission",
                "call_id": "call-0",
                "timeout_ms": 1_000,
            },
        )

    assert status.status_code == 200
    assert status.json()["dispatch_ready"] is True
