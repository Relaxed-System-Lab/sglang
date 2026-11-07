#!/usr/bin/env bash
# Run inside the approved runtime container. The caller supplies paths/ports and
# verifies that container GPUs 0..3 map to the authorized host GPUs 4..7.
# Stop this foreground supervisor with SIGTERM; it cleans only its own groups.
# --dry-run prints the plan without accessing paths, creating files or launching.
set -euo pipefail
exec python3 - "$@" <<'PY'
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description="Native 1P+1D TP2 reference; no planner/Ours.")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
required = (
    "SOURCE_DIR", "MODEL_PATH", "SERVED_MODEL_NAME", "GATEWAY_BINARY", "RUN_DIR",
    "PREFILL_PORT", "DECODE_PORT", "BOOTSTRAP_PORT",
    "BASELINE_ROUTER_PORT", "BASELINE_METRICS_PORT",
)
missing = [name for name in required if not os.environ.get(name)]
if missing:
    parser.error("required environment variables: " + ", ".join(missing))
config = {name: os.environ[name] for name in required}
port_names = required[5:]
ports = {name: int(config[name]) for name in port_names}
if len(set(ports.values())) != len(ports) or any(not 0 < port < 65536 for port in ports.values()):
    parser.error("the five caller-supplied ports must be distinct and within 1..65535")
ready_timeout = float(os.environ.get("READY_TIMEOUT_S", "1800"))
run_timeout = float(os.environ.get("RUN_TIMEOUT_S", "7200"))
if not 0 < ready_timeout <= run_timeout:
    parser.error("require 0 < READY_TIMEOUT_S <= RUN_TIMEOUT_S")
log_cap = 16 * 1024 * 1024
source = config["SOURCE_DIR"]
model = config["MODEL_PATH"]
served = config["SERVED_MODEL_NAME"]
run_dir = Path(config["RUN_DIR"])
prefill_url = f"http://127.0.0.1:{ports['PREFILL_PORT']}"
decode_url = f"http://127.0.0.1:{ports['DECODE_PORT']}"
gateway_url = f"http://127.0.0.1:{ports['BASELINE_ROUTER_PORT']}"
environment = {
    "PYTHONPATH": source + "/python",
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1",
    "SGLANG_MOONCAKE_TRANSFER_PROTOCOL": "tcp", "MOONCAKE_PROTOCOL": "tcp",
    "MC_TE_METRIC": "1",
}
unset_environment = [
    "SGLANG_MOONCAKE_CUSTOM_MEM_POOL",
    "SGLANG_DISAGGREGATION_MAX_INFLIGHT_TRANSFERS",
]
worker_common = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", model, "--served-model-name", served,
    "--host", "127.0.0.1", "--tp-size", "2",
    "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16",
    "--context-length", "8192", "--max-running-requests", "16",
    "--mem-fraction-static", "0.82",
    "--disable-radix-cache", "--disable-cuda-graph",
    "--enable-priority-scheduling", "--schedule-low-priority-values-first",
    "--log-level", "info",
    "--disaggregation-transfer-backend", "mooncake",
    "--enable-request-time-stats-logging",
]
components = {
    "prefill": {
        "argv": worker_common + [
            "--port", str(ports["PREFILL_PORT"]), "--disaggregation-mode", "prefill",
            "--disaggregation-bootstrap-port", str(ports["BOOTSTRAP_PORT"]),
            "--prefill-max-requests", "4",
        ],
        "environment": {"CUDA_VISIBLE_DEVICES": "0,1"},
        "ready_url": prefill_url + "/v1/models",
    },
    "decode": {
        "argv": worker_common + [
            "--port", str(ports["DECODE_PORT"]), "--disaggregation-mode", "decode",
        ],
        "environment": {"CUDA_VISIBLE_DEVICES": "2,3"},
        "ready_url": decode_url + "/v1/models",
    },
    "baseline_gateway": {
        "argv": [
            config["GATEWAY_BINARY"], "--host", "127.0.0.1",
            "--port", str(ports["BASELINE_ROUTER_PORT"]),
            "--prometheus-host", "127.0.0.1",
            "--prometheus-port", str(ports["BASELINE_METRICS_PORT"]),
            "--pd-disaggregation",
            "--prefill", prefill_url, str(ports["BOOTSTRAP_PORT"]),
            "--decode", decode_url, "--policy", "round_robin", "--json-log",
        ],
        "environment": {"CUDA_VISIBLE_DEVICES": ""},
        "ready_url": gateway_url + "/v1/models",
    },
}
plan = {
    "scope": "native_reference_only_no_planner_or_ours",
    "config": config, "components": components,
    "environment_overrides": environment, "environment_unset": unset_environment,
    "ready_timeout_s": ready_timeout, "run_timeout_s": run_timeout,
    "process_creation": "setsid via subprocess.Popen(start_new_session=True)",
    "cleanup_scope": "only process groups created by this supervisor",
    "log_cap_bytes_per_process": log_cap,
    "maximum_component_log_bytes": len(components) * log_cap,
    "persistent_budget_bytes": 512 * 1024 * 1024,
    "container_gpu_mapping": {"prefill": [0, 1], "decode": [2, 3]},
    "host_gpu_mapping": "must be verified by caller before launch",
    "tp_per_model_instance": 2, "model_instances": 2, "physical_container_gpus": 4,
    "runtime_transfer_cap": "not_configured_or_verified",
    "snapshot_projection_slots": 0,
    "isolated_profile_concurrency": "serial client must ensure max outstanding requests = 1",
    "tcp_protocol": "configured explicitly; confirm actual transport in worker logs",
    "implementation_revision": os.environ.get("IMPLEMENTATION_REVISION"),
}
if args.dry_run:
    print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
    raise SystemExit(0)

if not Path(source).is_dir() or not Path(model).is_dir():
    raise RuntimeError("SOURCE_DIR and MODEL_PATH must exist inside the runtime container")
if not Path(config["GATEWAY_BINARY"]).is_file() or not os.access(config["GATEWAY_BINARY"], os.X_OK):
    raise RuntimeError("GATEWAY_BINARY must be an executable inside the runtime container")
run_dir.mkdir(parents=True, exist_ok=False)
entries = []
received_signal = None
started_wall = time.time()
started_mono = time.monotonic()
deadline = started_mono + run_timeout
startup_deadline = started_mono + ready_timeout


def save_json(name, value):
    temporary = run_dir / (name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(run_dir / name)


def snapshot():
    return [
        {
            "name": entry["name"], "pid": entry["process"].pid,
            "pgid": entry["process"].pid, "returncode": entry["process"].poll(),
            **dict(entry["capture"]),
        }
        for entry in entries
    ]


def status(phase, error=None):
    save_json("status.json", {
        "phase": phase, "error": error, "timestamp_unix_s": time.time(),
        "started_unix_s": started_wall, "supervisor_pid": os.getpid(),
        "signal": received_signal, "processes": snapshot(),
        "endpoints": {"prefill": prefill_url, "decode": decode_url, "baseline": gateway_url},
        "runtime_transfer_cap": plan["runtime_transfer_cap"],
        "snapshot_projection_slots": 0,
    })


def drain_log(entry):
    capture = entry["capture"]
    sink = None
    try:
        sink = (run_dir / capture["log_file"]).open("wb")
    except OSError as exc:
        capture["write_error"] = str(exc)
    try:
        while True:
            chunk = os.read(entry["process"].stdout.fileno(), 65536)
            if not chunk:
                break
            capture["bytes_drained"] += len(chunk)
            remaining = log_cap - capture["bytes_saved"]
            if sink is not None and remaining > 0:
                try:
                    capture["bytes_saved"] += sink.write(chunk[:remaining])
                    sink.flush()
                except OSError as exc:
                    capture["write_error"] = str(exc)
                    try:
                        sink.close()
                    except OSError:
                        pass
                    sink = None
            capture["truncated"] = capture["bytes_drained"] > log_cap
    finally:
        if sink is not None:
            try:
                sink.close()
            except OSError as exc:
                capture["write_error"] = str(exc)
        entry["process"].stdout.close()
        capture["drain_finished"] = True


def start(name):
    component = components[name]
    child_env = os.environ.copy()
    for key in unset_environment:
        child_env.pop(key, None)
    child_env.update(environment)
    child_env.update(component["environment"])
    process = subprocess.Popen(
        component["argv"], cwd=source, env=child_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
        start_new_session=True,
    )
    entry = {
        "name": name, "process": process,
        "capture": {
            "log_file": name + ".log", "bytes_drained": 0, "bytes_saved": 0,
            "truncated": False, "write_error": None, "drain_finished": False,
        },
    }
    entries.append(entry)
    (run_dir / (name + ".pid")).write_text(str(process.pid) + "\n")
    (run_dir / (name + ".pgid")).write_text(str(process.pid) + "\n")
    entry["thread"] = threading.Thread(target=drain_log, args=(entry,), daemon=True)
    entry["thread"].start()
    save_json("manifest.json", {**plan, "started_unix_s": started_wall,
                               "supervisor_pid": os.getpid(), "processes": snapshot()})


def check_running():
    if received_signal is not None:
        raise InterruptedError(signal.Signals(received_signal).name)
    if time.monotonic() >= deadline:
        raise TimeoutError("RUN_TIMEOUT_S reached")
    for entry in entries:
        if entry["process"].poll() is not None:
            raise RuntimeError(f"{entry['name']} exited with {entry['process'].returncode}")
        if entry["capture"]["write_error"]:
            raise RuntimeError(f"log capture failed for {entry['name']}: {entry['capture']['write_error']}")


def model_ready(name):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(components[name]["ready_url"], timeout=2) as response:
            payload = json.load(response)
        return any(item.get("id") == served for item in payload.get("data", []))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def wait_ready(names, phase):
    ready = set()
    while len(ready) != len(names):
        check_running()
        if time.monotonic() >= startup_deadline:
            raise TimeoutError("READY_TIMEOUT_S reached")
        ready.update(name for name in names if name not in ready and model_ready(name))
        status(phase)
        time.sleep(1)


def stop_owned_groups():
    # PGIDs are exactly the PIDs returned by our own setsid launches. Never scan
    # shared PID files, reuse RUN_DIR, or signal a service found outside this list.
    for entry in reversed(entries):
        try:
            os.killpg(entry["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    grace_deadline = time.monotonic() + 10
    for entry in entries:
        try:
            entry["process"].wait(timeout=max(0.01, grace_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for entry in reversed(entries):
        try:
            os.killpg(entry["process"].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for entry in entries:
        entry["process"].wait(timeout=5)
        if "thread" in entry:
            entry["thread"].join(timeout=2)


def on_signal(signum, frame):
    global received_signal
    received_signal = signum


signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)
(run_dir / "supervisor.pid").write_text(str(os.getpid()) + "\n")
save_json("manifest.json", {**plan, "started_unix_s": started_wall,
                           "supervisor_pid": os.getpid(), "processes": []})
terminal_phase, error, exit_code = "stopped", None, 0
try:
    status("starting")
    start("prefill")
    start("decode")
    wait_ready(["prefill", "decode"], "loading_workers")
    start("baseline_gateway")
    wait_ready(["baseline_gateway"], "starting_gateway")
    status("ready")
    print(json.dumps({"phase": "ready", "run_dir": str(run_dir),
                      "baseline_url": gateway_url}), flush=True)
    while True:
        check_running()
        status("ready")
        time.sleep(1)
except InterruptedError as exc:
    error = str(exc)
except TimeoutError as exc:
    terminal_phase, error, exit_code = "timed_out", str(exc), 124
except Exception as exc:
    terminal_phase, error, exit_code = "failed", f"{type(exc).__name__}: {exc}", 1
finally:
    stop_owned_groups()
    status(terminal_phase, error)
print(json.dumps({"phase": terminal_phase, "error": error,
                  "run_dir": str(run_dir)}), flush=True)
raise SystemExit(exit_code)
PY
