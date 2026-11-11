"""CPU checks for strict owned-process diagnostics and the full-workflow recipe."""
import os
from pathlib import Path

import test_hexgent_h20_transport_window as base
import test_hexgent_h20_functional_window as functional


def test_functional_recipe_uses_synchronous_inductor_compilation(tmp_path):
    module = base.load_window()
    config = functional.functional_config(tmp_path)
    config["window_s"] = 1200
    plan = module.prepare_plan(config, functional_smoke=True)
    recipe = module.render_c1_runtime_recipe(config, now_unix_s=1200, functional_smoke=True)
    assert "TORCHINDUCTOR_COMPILE_THREADS=1" in recipe["docker_argv"]
    assert plan["functional_smoke"]["compile_threads"] == 1
    assert plan["functional_smoke"]["observer"] == {
        "normal_interval_s": 2.0, "error_burst_seconds": 10.0, "post_ready_window_s": 300.0}
    assert plan["hard_deadline_unix_s"] - plan["window_anchor_unix_s"] == 1200
    assert plan["cleanup_reserve_s"] == 180


def test_other_recipe_does_not_force_compiler_environment(tmp_path):
    module = base.v10_window(tmp_path)
    recipe = module.render_c1_runtime_recipe(base.v10_window_config(tmp_path), now_unix_s=1200)
    assert "TORCHINDUCTOR_COMPILE_THREADS=1" not in recipe["docker_argv"]


def test_registration_preserves_parent_and_executable_identity(tmp_path):
    module = base.v10_window(tmp_path)
    fixture = base.v10_registration_fixture(module, base.v10_window_config(tmp_path))
    for row in fixture["process_facts"]:
        row.update(tgid=row["pid"], process_name="synthetic-worker", argv=["python", "synthetic-worker.py"])
    registration = module.build_observer_registration(**fixture)
    actual = {row["pid"]: row for row in registration.get("process_identity_metadata", [])}
    assert set(actual) == {row["pid"] for row in fixture["process_facts"]}
    for row in fixture["process_facts"]:
        for key in ("start_identity", "ppid", "pgrp", "tgid", "process_name", "argv", "task_ids"):
            assert actual[row["pid"]][key] == row[key]
    assert len(registration["processes"]) == len(fixture["process_facts"])


def test_actual_proc_reader_records_owned_command_and_tgid():
    module = base.load_window()
    class Reader:
        def read_text(self, path):
            return "" if path.endswith("/children") else Path(path).read_text()
        def readlink(self, path): return os.readlink(path)
        def list_fd_targets(self, pid): return []
    facts, _, _ = module._read_owned_processes([os.getpid()], Reader(), supervisor_pid=os.getppid())
    assert len(facts) == 1
    row = facts[0]
    assert row["ppid"] == os.getppid() and row["tgid"] == os.getpid()
    assert row["process_name"] and row["argv"] and "python" in row["argv"][0]
    assert row["start_identity"] == row["start_identity_after"]
