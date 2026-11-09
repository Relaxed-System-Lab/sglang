"""Offline execution-boundary tests for the fixed compatible solver smoke."""
import asyncio
import json
from pathlib import Path
import types

import pytest
import test_hexgent_h20_transport_window as base


def functional_config(tmp_path):
    config = base.v10_window_config(tmp_path)
    path = Path(config["runtime_input_dir"]) / "sharegpt.calibration.sanitized.runtime.jsonl"
    inputs = [39, 300, 300, 501, 400, 400, 400, 400, 1200, 1236]
    outputs = [100, 100, 100, 163, 200, 200, 200, 200, 300, 389]
    records = []
    offset = 0
    for wid, count in (("1HuVef4_0", 4), ("2wtLdZm_0", 6)):
        calls = [{"call_id": f"{wid}_{i}", "parents": [] if i == 0 else [f"{wid}_0"],
            "request": {"messages": [{"role": "user", "content": "synthetic fixed-cohort test"}]},
            "input_len": inputs[offset+i], "output_len": outputs[offset+i],
            "planner_hints": {"model_config": {}, "hardware_params": {}}} for i in range(count)]
        records.append({"workflow_id": wid, "workflow_type": "synthetic", "arrival_offset": 0, "calls": calls})
        offset += count
    path.write_text("".join(json.dumps(row)+"\n" for row in records))
    config["functional_input_sha256"] = base.hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("hotpotqa.calibration.sanitized.runtime.jsonl", "hotpotqa.heldout.sanitized.runtime.jsonl"):
        (path.parent/name).unlink()
    return config


def test_functional_plan_needs_only_fixed_ten_call_input_and_omits_performance_phases(tmp_path):
    module = base.load_window(); config = functional_config(tmp_path)
    plan = module.prepare_plan(config, functional_smoke=True)
    assert plan["execution_scope"] == "functional_smoke"
    assert set(plan["input_sha256"]) == {"sharegpt.calibration.sanitized.runtime.jsonl"}
    fixed = plan["functional_smoke"]
    assert (fixed["expected_workflows"], fixed["expected_calls"], fixed["expected_input_tokens"], fixed["expected_output_tokens"]) == (2, 10, 5176, 1952)
    assert fixed["workflow_ids"] == ["1HuVef4_0", "2wtLdZm_0"]
    assert fixed["validation_scope"] == "compatible_per_request_solver"
    assert fixed["actor_batch_owner_validation"] is False
    assert fixed["baseline_workload_calls"] == 0
    assert not any(key in plan for key in ("calibration", "references", "formal_runs", "warmup"))
    assert plan["output_budget_mib"]["total"] == 256


def test_functional_recipe_and_owner_bind_scope_and_bounded_worker_logging(tmp_path):
    module = base.load_window(); config = functional_config(tmp_path)
    plan = module.prepare_plan(config, functional_smoke=True)
    recipe = module.render_c1_runtime_recipe(config, now_unix_s=1200, functional_smoke=True)
    assert "--functional-smoke" in recipe["docker_argv"]
    assert "--startup-only" not in recipe["docker_argv"]
    assert "HEXGENT_FUNCTIONAL_DEQUEUE_LOG=1" in recipe["docker_argv"]
    assert "HEXGENT_FUNCTIONAL_DEQUEUE_NAMESPACE="+base.RUN+"::functional-smoke-ours::" in recipe["docker_argv"]
    assert recipe["work_deadline_unix_s"] == 3520
    assert recipe["reference_environment"]["READY_TIMEOUT_S"] == "900.0"
    recipe["docker_argv"].remove("--functional-smoke")
    owner = module.C1OwnedLauncher(plan, recipe, clock=lambda: 1200)
    with pytest.raises(PermissionError, match="scope"):
        asyncio.run(owner.start(dry_run=True))


def test_functional_input_pin_and_scope_cannot_change_before_launch(tmp_path):
    module = base.load_window(); config = functional_config(tmp_path)
    config["functional_input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="input"):
        module.prepare_plan(config, functional_smoke=True)
    with pytest.raises(ValueError, match="scope"):
        module.prepare_plan(config, functional_smoke=True, startup_only=True)


def test_functional_constructor_loads_only_the_pinned_sharegpt_cohort(monkeypatch, tmp_path):
    module = base.load_window(); config = functional_config(tmp_path)
    config["runtime"]["run_dir"] = str(tmp_path/"owned-run")
    config["runtime"]["source_dir"] = str(base.ROOT)
    plan = module.prepare_plan(config, functional_smoke=True)
    api = module._pilot("h20_run_support").load_workflow_api(base.ROOT)
    original = api.load_workflow_traces_jsonl
    loaded = []
    def load(path):
        loaded.append(Path(path).name)
        return original(Path(config["runtime_input_dir"])/Path(path).name)
    monkeypatch.setattr(api, "load_workflow_traces_jsonl", load)
    ops = module.C1RuntimeOperations(config, plan, {}, base.OWNER)
    assert loaded == ["sharegpt.calibration.sanitized.runtime.jsonl"]
    assert [row.workflow_id for row in ops.calibration] == ["1HuVef4_0", "2wtLdZm_0"]
    assert sum(len(row.calls) for row in ops.calibration) == 10
    assert ops.hot == []


def test_functional_cli_dryrun_binds_recipe_and_never_executes(tmp_path, capsys):
    module = base.load_window(); config = functional_config(tmp_path)
    path = tmp_path/"config.json"; path.write_text(json.dumps(config))
    def forbidden(*args): pytest.fail("dry run reached owned runtime")
    assert module.main(["--config", str(path), "--functional-smoke", "--dry-run", "--print-recipe"],
        operations=types.SimpleNamespace(execute_c1=forbidden)) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["plan"]["execution_scope"] == "functional_smoke"
    assert "--functional-smoke" in rendered["recipe"]["docker_argv"]


@pytest.mark.parametrize("functional", [False, True])
def test_real_gateway_command_uses_frozen_compatible_route_only_in_functional_scope(monkeypatch, tmp_path, functional):
    module = base.v10_window(tmp_path); config = base.v10_window_config(tmp_path)
    transfer = tmp_path/"transfer.json"; transfer.write_text(json.dumps({"HEXGENT_TRANSFER_STARTUP_MS": 5.4, "HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S": 2379754520.5}))
    ops = object.__new__(module.C1RuntimeOperations)
    ops.runtime = config["runtime"]; ops.runtime["transfer_parameters_path"] = str(transfer)
    ops.ours_entries = []; ops.capture_status = None
    ops.plan = {"run_id": base.RUN, "frozen_metadata": {"transfer_parameters_sha256": module.sha256(transfer)}}
    if functional: ops.plan["execution_scope"] = "functional_smoke"
    ops.budget = module.WindowBudget(anchor_unix_s=module.time.time())
    recorded = []
    class Children:
        def healthy(self): return True
        def start(self, name, argv, env, path, **kwargs):
            recorded.append((name, list(argv), dict(env)))
            if name == "ours-sidecar": Path(argv[argv.index("--capture-status-path")+1]).write_text(json.dumps({"observation_complete": True}))
            return {"name": name, "process": types.SimpleNamespace(pid=123, poll=lambda: None)}
    ops.children = Children()
    async def get(url, deadline): return {"data": [{"id": base.MODEL}]}
    monkeypatch.setattr(module, "async_get_json", get)
    ops.verify_ours_frontends = lambda: {"owned_frontends": True}
    asyncio.run(ops.start_ours(tmp_path))
    argv = next(row[1] for row in recorded if row[0] == "ours-gateway")
    assert "--hexgent-router" in argv
    assert ("--hexgent-admission-queue" in argv) is (not functional)
    assert argv[argv.index("--hexgent-fallback-policy")+1] == ("fail_closed" if functional else "pd_policy")
    if functional:
        assert not any("admission" in flag for flag in argv)


@pytest.mark.parametrize("failure", [None, "startup", "replay", "cleanup"])
def test_functional_execution_replays_once_without_warmup_and_drains_owned_children(monkeypatch, tmp_path, failure):
    module = base.v10_window(tmp_path); monkeypatch.setattr(module.time, "time", lambda: 1100.)
    ops = object.__new__(module.C1RuntimeOperations)
    ops.plan = {"run_id": base.RUN, "execution_scope": "functional_smoke", "input_sha256": {}}
    ops.runtime = {"pd_topology": "2p2d_tp2"}; ops.root = tmp_path; ops.owner_id = base.OWNER
    ops.window = base.new_window(module); ops.window.register_owned_container(base.OWNER)
    ops.budget = ops.window.budget; ops.approval = base.approval()
    ops.calibration = base.workflows(); ops.api = object(); ops.model = base.MODEL
    ops.raw_budget = types.SimpleNamespace(replay=None); ops.phase_records = []; ops.ours_entries = []; ops.capture_status = None
    ops.expected = {}; ops.summary = lambda: base.observer(); ops.healthy = lambda: True
    ops.endpoint = lambda name: "synthetic://"+name
    ops._idle_arguments = lambda: {}; ops._instances = lambda: []
    observed = []
    async def startup():
        observed.append("startup")
        if failure == "startup": raise module.WindowStopped("synthetic startup error")
    async def start_ours(directory):
        observed.append("ours")
        for index, name in enumerate(("ours-sidecar", "ours-gateway")):
            state = {"code": None}
            ops.children.entries.append({"name": name, "state": state, "process": types.SimpleNamespace(pid=600+index, poll=lambda state=state: state["code"]), "pgid": 600+index, "thread": types.SimpleNamespace(is_alive=lambda: False), "truncated": False, "write_error": None})
    async def replay(api, workflows, **kwargs):
        observed.append(("replay", len(workflows), kwargs["router_url"]))
        kwargs["raw_path"].write_text("")
        if failure == "replay": raise module.WindowStopped("synthetic replay error")
    async def idle(*args, **kwargs): observed.append("idle"); return {"success": True}
    async def stop(deadline, selected):
        observed.append(("stop", deadline, [row["name"] for row in selected]))
        for row in selected: row["state"]["code"] = 0
        if failure == "cleanup" and selected[0]["name"] != "transport-observer": raise module.WindowStopped("synthetic cleanup error")
    entries = []
    for index, name in enumerate(("reference-supervisor", "transport-observer")):
        state = {"code": None}
        entries.append({"name": name, "state": state, "process": types.SimpleNamespace(pid=500+index, poll=lambda state=state: state["code"]), "pgid": 500+index, "thread": types.SimpleNamespace(is_alive=lambda: False), "truncated": False, "write_error": None})
    ops.children = types.SimpleNamespace(entries=entries, stop=stop)
    ops.startup = startup; ops.start_ours = start_ours
    async def stage(name, operation): return await operation(deadline_unix_s=3520)
    ops.run_stage = stage
    if failure == "startup":
        import test_hexgent_h20_functional_smoke as metadata_fixture
        helper = metadata_fixture.load_helper()
        cohort, _, _, _, arguments = metadata_fixture.fixture()
        ops._functional_evidence = lambda directory: helper.functional_smoke_gate(
            cohort, [], [], [], **{**arguments, "observer_summary": ops.functional_observer_summary,
                                   "capture_complete": False})
    else:
        ops._functional_evidence = lambda directory: {"successful": True, "expected_calls": 10}
    monkeypatch.setattr(module, "replay_workflows", replay); monkeypatch.setattr(module, "_idle", idle)
    async def forbidden(*args, **kwargs): pytest.fail("functional path entered old measurement/warmup/reference phase")
    ops.method_phase = forbidden
    code = asyncio.run(ops.execute())
    report = json.loads((tmp_path/"functional-smoke-results.json").read_text())
    assert code == (0 if failure is None else 2)
    assert report["successful"] is (failure is None)
    expected_replays = [] if failure == "startup" else [("replay", 2, "synthetic://ours")]
    assert [row for row in observed if isinstance(row, tuple) and row[0] == "replay"] == expected_replays
    expected_nonobserver = ["reference-supervisor"] if failure == "startup" else ["reference-supervisor", "ours-sidecar", "ours-gateway"]
    assert [row[2] for row in observed if isinstance(row, tuple) and row[0] == "stop"] == [expected_nonobserver, ["transport-observer"]]
    assert report["metrics"] is None
    if failure == "startup":
        assert report["stopped_reason"] == "WindowStopped: synthetic startup error"
        assert report["replay_status"] == "NOT_STARTED"
        assert "evidence_error" not in report
        assert report.get("evidence", {}).get("status") == "UNKNOWN"
        assert report["evidence"]["request_attempt_rows"] == 0
        assert report["evidence"]["completed_calls"] == 0
