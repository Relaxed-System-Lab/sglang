"""Offline regressions for the one bounded H20 functional observation phase."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_hexgent_h20_transport_observer as obs
import test_hexgent_h20_transport_window as win
import test_hexgent_h20_functional_window as functional

PROFILE = {"normal_interval_s": 2.0, "error_burst_seconds": 10.0, "post_ready_window_s": 300.0}


def configured_registration():
    registration = obs.registration(deadline=obs.T0 + 1200)
    registration["observation"] = {"normal_interval_s": 2.0, "error_burst_seconds": 10.0,
                                   "deadline_unix_s": obs.T0 + 120}
    return registration


def test_registered_normal_interval_reaches_the_actual_collector():
    observer = obs.load().TransportObserver(configured_registration())
    observer.accept_sample(obs.facts())
    assert observer.next_sample_delay(obs.T0) == 2.0


def test_error_burst_stops_after_ten_seconds():
    observer = obs.load().TransportObserver(configured_registration())
    observer.accept_sample(obs.facts())
    observer.consume_stream([obs.TCP_ERROR.encode()], observed_unix_s=obs.T0 + 1)
    assert observer.next_sample_delay(obs.T0 + 2) == .25
    assert observer.next_sample_delay(obs.T0 + 11) is None


def test_observation_deadline_never_moves_global_deadline():
    observer = obs.load().TransportObserver(configured_registration())
    observer.accept_sample(obs.facts())
    assert observer.deadline == obs.T0 + 1200
    assert observer.next_sample_delay(obs.T0 + 120) is None
    assert not observer.can_start_new_stage(obs.T0 + 120)


def test_real_loop_obeys_ready_window_and_bounded_error_sampling():
    module = obs.load(); observer = module.TransportObserver(configured_registration())
    now, samples, packets = [obs.T0], [], []
    class Producer:
        def collect(self, at):
            samples.append(at)
            return obs.facts(at)
    delivered = []
    def logs():
        if now[0] >= obs.T0 + 110 and not delivered:
            delivered.append(True)
            return {"worker": [obs.TCP_ERROR.encode()]}
        return {}
    result = module.collect_loop(observer, Producer(), read_logs=logs,
        clock=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay),
        emit_summary=lambda value: True, emit_record=lambda value: packets.append(value) or True)
    assert now[0] == obs.T0 + 120
    assert len(samples) == len(packets) <= 102
    assert samples[1] - samples[0] == 2
    assert result["stop_requested"] is True
    assert observer.deadline == obs.T0 + 1200


def test_functional_profile_is_frozen_in_the_real_plan(tmp_path):
    module = win.load_window(); config = functional.functional_config(tmp_path)
    config["window_s"] = 1200
    plan = module.prepare_plan(config, functional_smoke=True)
    assert plan["functional_smoke"].get("observer") == PROFILE
    assert plan["hard_deadline_unix_s"] - plan["window_anchor_unix_s"] == 1200
    assert plan["cleanup_reserve_s"] == 180
    assert plan["output_budget_mib"]["observer"] == 64
    assert plan["output_budget_mib"]["total"] == 256


@pytest.mark.parametrize("at", [1100., 1900.])
def test_actual_startup_binds_observer_only_after_worker_ready(monkeypatch, tmp_path, at):
    module = win.load_window(); monkeypatch.setattr(module.time, "time", lambda: at)
    ops = object.__new__(module.C1RuntimeOperations)
    module._bounded_qc_json.root = tmp_path
    ops.root = tmp_path; ops.root.joinpath("workers").mkdir()
    ops.root.joinpath("workers/status.json").write_text(json.dumps({"phase": "ready"}))
    gateway = tmp_path/"gateway"; gateway.write_bytes(b"synthetic")
    model = tmp_path/"model"; model.mkdir(); (model/"config.json").write_text("{}")
    ops.runtime = {"source_dir": str(tmp_path), "gateway_binary": str(gateway),
        "model_path": str(model), "approval_container_path": "synthetic-approval", "cpuset_cpus": "1"}
    frozen = {"implementation_revision": "synthetic", "image_digest": "synthetic",
        "gateway_elf_sha256": module.sha256(gateway), "model_revision": "synthetic"}
    ops.plan = {"run_id": win.RUN, "execution_scope": "functional_smoke", "source_file_sha256": {},
        "input_sha256": {}, "frozen_metadata": frozen, "functional_smoke": {"observer": PROFILE}}
    ops.config = {"runtime": ops.runtime}; ops.owner_id = win.OWNER
    ops.budget = module.WindowBudget(1000, 1200, 180); ops.ready_deadline_unix_s = None
    ops.window = SimpleNamespace(_observe=lambda value: None)
    ops.summary = lambda: win.observer(); ops._idle_arguments = lambda: {}
    ops.expected = {}; ops.model = "synthetic"
    captured, order = [], []
    class Children:
        def healthy(self): return True
        def start(self, name, argv, env, path, **kwargs):
            order.append(name)
            if name == "transport-observer":
                captured.append(json.loads((tmp_path/"observer-registration.json").read_text()))
                folder = tmp_path/"observer"; folder.mkdir(); (folder/"summary.latest.json").write_text("{}")
            return {"process": SimpleNamespace(pid=123)}
    ops.children = Children()
    monkeypatch.setattr(module, "render_c1_runtime_recipe", lambda *a, **k:
        {"reference_environment": {"READY_TIMEOUT_S": "900"}, "reference_argv": ["synthetic"]})
    def registration(config, plan, owner, status, **kwargs):
        assert status["phase"] == "ready"
        order.append("actual-ready-registration")
        return {"log_paths": [], "endpoint_observation_ready": True,
                "deadline_unix_s": ops.budget.hard_deadline_unix_s}
    monkeypatch.setattr(module, "collect_startup_registration", registration)
    async def idle(*a, **k): return {"final": {}}
    monkeypatch.setattr(module, "_idle", idle)
    monkeypatch.setattr(module, "RuntimeEpoch", lambda *a, **k: SimpleNamespace(bind_file=lambda p: {}))
    async def stage(name, operation): return await operation(deadline_unix_s=2020)
    ops.run_stage = stage
    if at == 1900.:
        with pytest.raises(TimeoutError, match="startup deadline"):
            asyncio.run(ops.startup())
        assert captured == []
        assert order == ["reference-supervisor"]
        return
    asyncio.run(ops.startup())
    assert order == ["reference-supervisor", "actual-ready-registration", "transport-observer"]
    assert captured[0].get("observation") == {"normal_interval_s": 2.0,
        "error_burst_seconds": 10.0, "deadline_unix_s": 1400.0}
    assert captured[0]["deadline_unix_s"] == 2200
    assert ops.ready_deadline_unix_s == 1400
    assert ops.budget.work_deadline_unix_s == 2020


def test_ready_timeout_cancels_replay_but_runs_owned_cleanup(monkeypatch, tmp_path):
    module = win.load_window()
    assert hasattr(module.C1RuntimeOperations, "_within_ready_deadline"), "ready-phase timeout is not implemented"
    monkeypatch.setattr(module.time, "time", lambda: 1100.)
    ops = object.__new__(module.C1RuntimeOperations)
    ops.plan = {"run_id": win.RUN, "execution_scope": "functional_smoke", "functional_smoke": {"observer": PROFILE}}
    module._bounded_qc_json.root = tmp_path
    ops.root = tmp_path; ops.owner_id = win.OWNER; ops.approval = {}; ops.ready_deadline_unix_s = None
    ops.budget = module.WindowBudget(1000, 1200, 180)
    ops.api = object(); ops.calibration = []; ops.model = "synthetic"; ops.expected = {}
    ops.raw_budget = SimpleNamespace(replay=None); ops.endpoint = lambda role: "synthetic://"+role
    ops.summary = lambda: {}; ops.healthy = lambda: True; ops._idle_arguments = lambda: {}
    cancelled, cleaned = [], []
    async def outer_stage(name, operation, **kwargs): return await operation(deadline_unix_s=2020)
    ops.window = SimpleNamespace(run_stage=outer_stage, _observe=lambda value: None)
    async def alarm(): await asyncio.Event().wait()
    ops.alarm = alarm
    async def startup(): ops.ready_deadline_unix_s = 1100.01
    async def ours(directory): pass
    async def replay(*a, **k):
        try: await asyncio.Event().wait()
        finally: cancelled.append(True)
    async def cleanup(final):
        cleaned.append(ops.budget.hard_deadline_unix_s)
        final["children_cleanup_complete"] = True
    ops.startup = startup; ops.start_ours = ours; ops._retire_owned_children = cleanup
    ops._functional_evidence = lambda directory: {"successful": True}
    monkeypatch.setattr(module, "replay_workflows", replay)
    code = asyncio.run(ops.execute_functional_smoke())
    report = json.loads((tmp_path/"functional-smoke-results.json").read_text())
    assert code == 2 and report["successful"] is False
    assert report["replay_status"] == "FAILED"
    assert "TimeoutError" in report["stopped_reason"]
    assert cancelled == [True] and cleaned == [2200]
    assert ops.budget.work_deadline_unix_s == 2020
