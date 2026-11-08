"""Fixed H20 layouts exercised through the real embedded launcher, without workers."""
import builtins
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts/hexgent_real_trace_collectors"
LAYOUT = SCRIPTS / "h20_reference_components.py"
LAUNCHER = SCRIPTS / "start_h20_tp2_reference.sh"
MODEL = "/model-cache/snapshots/945c8663693130f8be2ee66210e062158b2a9693"
SERVED = "hexagent-llama70b-bf16-tp2-01a1b627"


def ports(four=False):
    values = {"prefill": 37100, "bootstrap": 37120, "decode": 37110,
              "native": 37130, "native_metrics": 37131}
    if four:
        values.update(prefill_1=37200, bootstrap_1=37220, decode_1=37210)
    return values


def expected_instances(four=False):
    rows = [("P0", "prefill", "prefill", 37100, 37120, [0, 1])]
    if four:
        rows.append(("P1", "prefill_1", "prefill", 37200, 37220, [2, 3]))
    rows.append(("D0", "decode", "decode", 37110, None, [4, 5] if four else [2, 3]))
    if four:
        rows.append(("D1", "decode_1", "decode", 37210, None, [6, 7]))
    return [{"instance_id": identity, "component": component, "role": role,
             "worker_url": f"http://127.0.0.1:{port}", "http_port": port,
             "bootstrap_port": bootstrap, "cuda_devices": devices, "tp_size": 2}
            for identity, component, role, port, bootstrap, devices in rows]


def load_layout(monkeypatch):
    # Missing implementation is a targeted assertion failure, not collection failure.
    assert LAYOUT.is_file(), "canonical h20_reference_components.py is not implemented"
    spec = importlib.util.spec_from_file_location("h20_reference_components", LAYOUT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def expected_components(four=False, model=MODEL, gateway="/synthetic/gateway"):
    common = ["python3", "-m", "sglang.launch_server", "--model-path", model,
        "--served-model-name", SERVED, "--host", "127.0.0.1", "--tp-size", "2",
        "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16",
        "--context-length", "8192", "--max-running-requests", "16",
        "--mem-fraction-static", "0.82", "--disable-radix-cache", "--disable-cuda-graph",
        "--enable-priority-scheduling", "--schedule-low-priority-values-first",
        "--log-level", "info", "--disaggregation-transfer-backend", "mooncake",
        "--enable-request-time-stats-logging"]
    instances = expected_instances(four)
    components = {}
    for row in instances:
        argv = common + ["--port", str(row["http_port"]), "--disaggregation-mode", row["role"]]
        if row["role"] == "prefill":
            argv += ["--disaggregation-bootstrap-port", str(row["bootstrap_port"]),
                     "--prefill-max-requests", "4"]
        components[row["component"]] = {"argv": argv,
            "environment": {"CUDA_VISIBLE_DEVICES": ",".join(map(str, row["cuda_devices"]))},
            "ready_url": row["worker_url"] + "/v1/models"}
    argv = [gateway, "--host", "127.0.0.1", "--port", "37130",
            "--prometheus-host", "127.0.0.1", "--prometheus-port", "37131",
            "--pd-disaggregation"]
    for row in instances:
        if row["role"] == "prefill":
            argv += ["--prefill", row["worker_url"], str(row["bootstrap_port"])]
    for row in instances:
        if row["role"] == "decode":
            argv += ["--decode", row["worker_url"]]
    argv += ["--policy", "round_robin", "--json-log"]
    components["baseline_gateway"] = {"argv": argv, "environment": {"CUDA_VISIBLE_DEVICES": ""},
                                    "ready_url": "http://127.0.0.1:37130/v1/models"}
    return components


def embedded_program(flags):
    header, body = LAUNCHER.read_text().split("<<'PY'\n", 1)
    source = body.rsplit("\nPY", 1)[0]
    line = next(row for row in header.splitlines() if row.startswith("exec python3 - "))
    # python3 - sets __file__ to <stdin>. Derive its real extra argument from
    # the actual shell wire, rather than inventing a working SOURCE_DIR.
    argv = ["-"]
    if '"${BASH_SOURCE[0]}"' in line:
        argv.append(str(LAUNCHER))
    else:
        assert 'exec python3 - "$@"' in line, "unmodeled launcher stdin bootstrap"
    return source, argv + flags


def copy_module(module, **overrides):
    result = types.ModuleType(module.__name__)
    result.__dict__.update(vars(module))
    result.__dict__.update(overrides)
    return result


def run_embedded(tmp_path, monkeypatch, *, four=False, dry=False, fail_on=None):
    """Execute production supervisor logic; replace only external I/O boundaries."""
    values = ports(four)
    environment = {"SOURCE_DIR": "/synthetic/source", "MODEL_PATH": MODEL,
        "SERVED_MODEL_NAME": SERVED, "GATEWAY_BINARY": "/synthetic/gateway",
        "RUN_DIR": "/synthetic/run", "READY_TIMEOUT_S": "30", "RUN_TIMEOUT_S": "60",
        "PREFILL_PORT": str(values["prefill"]), "DECODE_PORT": str(values["decode"]),
        "BOOTSTRAP_PORT": str(values["bootstrap"]), "BASELINE_ROUTER_PORT": "37130",
        "BASELINE_METRICS_PORT": "37131"}
    if four:
        environment.update(PREFILL_1_PORT="37200", BOOTSTRAP_1_PORT="37220", DECODE_1_PORT="37210")
    run_dir = tmp_path / "owned-run"
    if not dry:
        source_dir, model_dir, gateway = (tmp_path / name for name in ("source", "model", "gateway"))
        source_dir.mkdir(); model_dir.mkdir(); gateway.write_text("fixture executable\n")
        environment.update(SOURCE_DIR=str(source_dir), MODEL_PATH=str(model_dir),
                           GATEWAY_BINARY=str(gateway), RUN_DIR=str(run_dir))
    state = types.SimpleNamespace(events=[], processes=[], ready=set(), model_calls={},
        gateway_ready=False, gateway_ready_at_start=None, signals=[], handlers={}, now=1000.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run reached an external runtime boundary")

    class NoRuntimePath:
        def __init__(self, value): self.value = str(value)
        def __str__(self): return self.value
        def __truediv__(self, value): return NoRuntimePath(self.value + "/" + str(value))
        is_dir = is_file = exists = mkdir = open = write_text = forbidden

    runtime_paths = {environment[key] for key in ("SOURCE_DIR", "MODEL_PATH", "GATEWAY_BINARY", "RUN_DIR")}
    def path(value):
        return NoRuntimePath(value) if dry and str(value) in runtime_paths else Path(value)

    class Pipe:
        closed = False
        def fileno(self): return 99999  # Never passed to the OS: fake os.read below.
        def close(self): self.closed = True

    class Process:
        def __init__(self, name):
            self.name, self.pid, self.returncode, self.stdout = name, 51001 + len(state.processes), None, Pipe()
        def poll(self): return self.returncode
        def wait(self, timeout=None):
            state.events.append(("wait", self.name))
            if self.returncode is None:
                raise subprocess.TimeoutExpired(self.name, timeout)
            return self.returncode

    def popen(argv, **kwargs):
        if dry: return forbidden()
        if argv[0] == environment["GATEWAY_BINARY"]:
            name = "baseline_gateway"
            state.gateway_ready_at_start = set(state.ready)
        else:
            port = int(argv[argv.index("--port") + 1])
            name = next(row["component"] for row in expected_instances(four) if row["http_port"] == port)
        state.events.append(("start", name))
        assert kwargs["start_new_session"] is True
        assert kwargs["cwd"] == environment["SOURCE_DIR"]
        if name == fail_on:
            raise OSError("fixture process creation failed")
        process = Process(name)
        state.processes.append(process)
        return process

    def killpg(pgid, signum):
        state.signals.append((pgid, signum))
        process = next((row for row in state.processes if row.pid == pgid), None)
        assert process is not None, "supervisor signaled a PGID it did not create"
        if process.returncode is not None:
            raise ProcessLookupError(pgid)
        process.returncode = -signum

    class Thread:
        def __init__(self, *, target, args, daemon): self.target, self.args, self.finished = target, args, False
        def start(self): pass
        def join(self, timeout=None):
            if not self.finished:
                self.target(*self.args)
                self.finished = True

    class Opener:
        def open(self, url, timeout):
            if dry: return forbidden()
            name = "baseline_gateway" if ":37130/" in url else next(
                row["component"] for row in expected_instances(four) if row["worker_url"] + "/v1/models" == url)
            state.model_calls[name] = state.model_calls.get(name, 0) + 1
            actual = "another-model" if name == "decode_1" and state.model_calls[name] == 1 else SERVED
            state.events.append(("model", name, actual))
            if actual == SERVED:
                if name == "baseline_gateway": state.gateway_ready = True
                else: state.ready.add(name)
            return io.BytesIO(json.dumps({"data": [{"id": actual}]}).encode())

    def sleep(seconds):
        state.now += seconds
        if state.gateway_ready:
            state.handlers[signal.SIGTERM](signal.SIGTERM, None)

    fake_os = copy_module(os, environ=environment, access=lambda *args: True,
        getpid=lambda: 51000, read=lambda *args: b"", killpg=killpg)
    fake_time = copy_module(time, time=lambda: state.now, monotonic=lambda: state.now, sleep=sleep)
    fake_signal = copy_module(signal, signal=lambda number, handler: state.handlers.__setitem__(number, handler))
    fake_subprocess = copy_module(subprocess, Popen=popen)
    fake_threading = copy_module(threading, Thread=Thread)
    fake_request = copy_module(urllib.request, build_opener=lambda *args: Opener(), ProxyHandler=lambda value: value)
    fake_urllib = types.ModuleType("urllib")
    fake_urllib.request, fake_urllib.error = fake_request, urllib.error
    fake_pathlib = types.ModuleType("pathlib"); fake_pathlib.Path = path
    replacements = {"os": fake_os, "time": fake_time, "signal": fake_signal,
        "subprocess": fake_subprocess, "threading": fake_threading, "pathlib": fake_pathlib}
    original_import = builtins.__import__
    def boundary_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in replacements: return replacements[name]
        if name.startswith("urllib"):
            if fromlist: return fake_request if name == "urllib.request" else urllib.error
            return fake_urllib
        if name.split(".")[0] in {"transformers", "tokenizers", "jinja2", "torch", "sglang"}:
            raise AssertionError("supervisor imported a provider/runtime")
        return original_import(name, globals, locals, fromlist, level)
    flags = (["--dry-run"] if dry else []) + (["--layout", "2p2d"] if four else [])
    source, argv = embedded_program(flags)
    monkeypatch.setattr(sys, "argv", argv)
    # Restore any sys.path insertion by the actual stdin bootstrap after this test.
    monkeypatch.syspath_prepend(str(tmp_path / "unrelated-import-root"))
    namespace = {"__name__": "__main__", "__file__": "<stdin>",
                 "__builtins__": {**vars(builtins), "__import__": boundary_import}}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            exec(compile(source, str(LAUNCHER), "exec"), namespace)
        except SystemExit as error:
            code = error.code
        else:
            raise AssertionError("actual launcher must terminate through SystemExit")
    return types.SimpleNamespace(code=code, state=state, output=output.getvalue(),
                                 run_dir=run_dir, environment=environment)


def test_default_layout_preserves_the_original_two_workers(monkeypatch):
    module = load_layout(monkeypatch)
    assert module.reference_worker_layout(ports()) == expected_instances()


def test_explicit_layout_has_four_fixed_nonoverlapping_tp2_instances(monkeypatch):
    module = load_layout(monkeypatch)
    rows = module.reference_worker_layout(ports(True), topology="2p2d_tp2")
    assert rows == expected_instances(True)
    assert [gpu for row in rows for gpu in row["cuda_devices"]] == list(range(8))


@pytest.mark.parametrize("missing", ["prefill_1", "bootstrap_1", "decode_1"])
def test_four_worker_layout_requires_each_new_owned_endpoint(monkeypatch, missing):
    module = load_layout(monkeypatch)
    values = ports(True); values.pop(missing)
    with pytest.raises((ValueError, KeyError)):
        module.reference_worker_layout(values, topology="2p2d_tp2")


def test_four_worker_layout_rejects_a_colliding_worker_port(monkeypatch):
    module = load_layout(monkeypatch)
    values = ports(True); values["prefill_1"] = values["prefill"]
    with pytest.raises(ValueError):
        module.reference_worker_layout(values, topology="2p2d_tp2")


def test_layout_import_reads_code_without_starting_a_runtime(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("layout import has runtime side effects")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(urllib.request, "build_opener", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    assert callable(load_layout(monkeypatch).reference_worker_layout)


def test_stdin_launcher_wires_its_real_source_path_before_user_arguments():
    header = LAUNCHER.read_text().split("<<'PY'\n", 1)[0]
    line = next(row for row in header.splitlines() if row.startswith("exec python3 - "))
    assert '"${BASH_SOURCE[0]}"' in line, "stdin entry lacks its real source-script argument"
    assert line.index('"${BASH_SOURCE[0]}"') < line.index('"$@"')


def test_actual_default_dry_run_keeps_legacy_commands_and_metadata(tmp_path, monkeypatch):
    result = run_embedded(tmp_path, monkeypatch, dry=True)
    assert result.code == 0
    plan = json.loads(result.output)
    assert plan["components"] == expected_components()
    assert plan["model_instances"] == 2 and plan["physical_container_gpus"] == 4
    assert plan["container_gpu_mapping"] == {"prefill": [0, 1], "decode": [2, 3]}
    assert plan["tp_per_model_instance"] == 2
    assert plan["maximum_component_log_bytes"] == 48 * 1024**2
    assert not result.state.events and not result.run_dir.exists()


def test_actual_four_worker_dry_run_consumes_the_canonical_layout(tmp_path, monkeypatch):
    module = load_layout(monkeypatch)
    original = module.reference_worker_layout
    calls = []
    def record(values, *, topology="1p1d_tp2"):
        calls.append((dict(values), topology))
        return original(values, topology=topology)
    monkeypatch.setattr(module, "reference_worker_layout", record)
    result = run_embedded(tmp_path, monkeypatch, four=True, dry=True)
    assert result.code == 0
    plan = json.loads(result.output)
    assert calls == [(ports(True), "2p2d_tp2")]
    assert plan["components"] == expected_components(True)
    assert plan["model_instances"] == 4 and plan["physical_container_gpus"] == 8
    assert plan["container_gpu_mapping"] == {row["component"]: row["cuda_devices"] for row in expected_instances(True)}
    assert plan["tp_per_model_instance"] == 2
    assert plan["log_cap_bytes_per_process"] == 16 * 1024**2
    assert plan["maximum_component_log_bytes"] == 80 * 1024**2
    assert not result.state.events and not result.run_dir.exists()


def test_gateway_starts_only_after_all_four_same_model_readiness_results(tmp_path, monkeypatch):
    result = run_embedded(tmp_path, monkeypatch, four=True)
    assert result.code == 0
    workers = {row["component"] for row in expected_instances(True)}
    assert result.state.gateway_ready_at_start == workers
    assert result.state.model_calls["decode_1"] == 2
    wrong = result.state.events.index(("model", "decode_1", "another-model"))
    ready = result.state.events.index(("model", "decode_1", SERVED))
    gateway = result.state.events.index(("start", "baseline_gateway"))
    assert wrong < ready < gateway
    status = json.loads((result.run_dir / "status.json").read_text())
    assert status["phase"] == "stopped" and status["signal"] == signal.SIGTERM
    assert len(status["processes"]) == 5
    assert all(row["drain_finished"] for row in status["processes"])


def test_third_worker_creation_failure_retires_only_returned_owned_groups(tmp_path, monkeypatch):
    result = run_embedded(tmp_path, monkeypatch, four=True, fail_on="decode")
    assert result.code == 1
    assert [process.name for process in result.state.processes] == ["prefill", "prefill_1"]
    owned = {process.pid for process in result.state.processes}
    assert {pgid for pgid, signum in result.state.signals} == owned
    assert {pgid for pgid, signum in result.state.signals if signum == signal.SIGTERM} == owned
    assert ("start", "baseline_gateway") not in result.state.events
    assert all(process.returncode is not None and process.stdout.closed for process in result.state.processes)
    status = json.loads((result.run_dir / "status.json").read_text())
    assert status["phase"] == "failed" and "OSError" in status["error"]
    assert {row["pid"] for row in status["processes"]} == owned
    assert all(row["drain_finished"] and not row["truncated"] and row["write_error"] is None
               for row in status["processes"])
