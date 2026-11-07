#!/usr/bin/env python3
"""Bounded transport observer core; default CLI is plan-only.

Runtime facts and streams are supplied explicitly. Import and plan mode perform
no process/socket/FD/environment sampling and start no operation. The summary
contract is independent of the transport-window controller.
"""
import argparse
import copy
import json
import math
import re
import sys

SCHEMA_VERSION = 1
MAX_CAPTURE_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 64 * 1024
REQUIRED_SOURCES = ("processes", "sockets", "namespace_counts", "port_range", "cgroup", "allowed_env")
ALLOWED_ENV = frozenset({
    "SGLANG_MOONCAKE_TRANSFER_PROTOCOL", "MOONCAKE_PROTOCOL",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", "SGLANG_DISAGGREGATION_WAITING_TIMEOUT",
    "SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", "SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE",
    "SGLANG_ENABLE_FAILED_SESSION_PROBE", "SGLANG_FAILED_SESSION_PROBE_INTERVAL_S",
    "SGLANG_KV_TRANSFER_TELEMETRY", "AGENTIC_PD_OURS_PLACEMENT_STRATEGY",
    "HEXGENT_PROJECTED_RATIO_ORDER", "HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH",
    "HEXGENT_ENABLE_LOCAL_PRIORITY_UPDATES", "HEXGENT_TRANSFER_STARTUP_MS",
    "HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S", "HEXGENT_PROXY_MODEL_PATH",
    "SGLANG_DISAGGREGATION_MAX_INFLIGHT_TRANSFERS",
})
_CPP_ERROR = re.compile(
    r"^E\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+.*"
    r"(?:tcp_transport|transfer_engine(?:_impl)?)\.cpp:\d+\].*"
    r"(?:\bfailed\b|\berror\b|\bcannot\b)", re.IGNORECASE)
_KV_ERROR = re.compile(
    r"^\[\d{4}-\d{2}-\d{2}[^\]]*\].*"
    r"(?:prefill|decode) transfer failed.*KVTransferError\(", re.IGNORECASE)
_BACKEND_TIME = re.compile(r"^(E\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?|\[[^\]]+\])")


class TransportObserver:
    def __init__(self, registration, *, cap_bytes=MAX_CAPTURE_BYTES, burst_seconds=60):
        self.registration = copy.deepcopy(registration)
        self.run_id = registration["run_id"]
        self.owner_id = registration["owner_id"]
        if not re.fullmatch(r"[0-9a-f]{64}", self.owner_id):
            raise ValueError("owner_id must be the exact full registered container ID")
        self.deadline = float(registration["deadline_unix_s"])
        if not math.isfinite(self.deadline):
            raise ValueError("finite immutable global deadline required")
        if not 0 < cap_bytes <= MAX_CAPTURE_BYTES or not 0 < burst_seconds <= 60:
            raise ValueError("capture/burst exceeds approved observer bounds")
        self.cap_bytes = int(cap_bytes)
        self.burst_seconds = float(burst_seconds)
        if not registration["processes"] or not registration["peers"]:
            raise ValueError("live observation requires nonempty actual process and peer registration")
        self._processes = {int(p["pid"]): str(p["start_identity"]) for p in registration["processes"]}
        if len(self._processes) != len(registration["processes"]):
            raise ValueError("registered PID identities must be unique")
        self._peers = {(p["ip"], int(p["port"])) for p in registration["peers"]}
        self._sampled_unix_s = None
        self._reasons = []
        self._stop = False
        self._counts = {}
        self._details = []
        self._first_error = None
        self._first_error_evidence = None
        self._burst_end = None
        self._retained = bytearray()
        self._streams = {}
        self.drained_bytes = 0

    def request_stop(self):
        self._stop = True

    def mark_incomplete(self, reason):
        if reason not in self._reasons:
            self._reasons.append(reason)
        self._stop = True

    @property
    def retained_log(self):
        return bytes(self._retained)

    @property
    def first_error_evidence(self):
        return copy.deepcopy(self._first_error_evidence)

    @property
    def buffered_line_bytes(self):
        return sum(len(state["buffer"]) for state in self._streams.values())

    def summary(self):
        reasons = list(self._reasons)
        if self._sampled_unix_s is None:
            reasons.insert(0, "not_sampled")
        return {
            "schema_version": SCHEMA_VERSION, "run_id": self.run_id, "owner_id": self.owner_id,
            "sampled_unix_s": self._sampled_unix_s,
            "observation_complete": self._sampled_unix_s is not None and not self._reasons,
            "first_error_unix_s": self._first_error, "stop_requested": self._stop,
            "incomplete_reasons": reasons,
            "anonymous_namespace_counts": copy.deepcopy(self._counts),
            "owned_details": copy.deepcopy(self._details),
        }

    def accept_sample(self, facts):
        if facts.get("run_id") != self.run_id:
            self.mark_incomplete("provenance_mismatch:run_id")
            return self.summary()
        if facts.get("owner_id") != self.owner_id:
            self.mark_incomplete("provenance_mismatch:owner_id")
            return self.summary()
        now = facts.get("sampled_unix_s")
        if now is None or not math.isfinite(float(now)):
            self.mark_incomplete("missing_sample_time")
            return self.summary()
        now = float(now)
        if self._sampled_unix_s is not None and now < self._sampled_unix_s:
            self.mark_incomplete("sample_time_regressed")
            return self.summary()
        self._sampled_unix_s = now
        payloads = {
            "processes": "processes", "sockets": "sockets",
            "namespace_counts": "anonymous_namespace_counts", "port_range": "port_range",
            "cgroup": "cgroup_memory_events", "allowed_env": "processes"}
        status = facts.get("source_status", {})
        for name in REQUIRED_SOURCES:
            if status.get(name) != "ok":
                state = status.get(name)
                reason = ("denied_source:" + name) if state == "denied" else (
                    "missing_source:" + name if state in (None, "missing") else
                    "incomplete_source:" + name + ":" + str(state))
                self.mark_incomplete(reason)
            if payloads[name] not in facts:
                self.mark_incomplete("missing_payload:" + name)
        supplied_counts = facts.get("anonymous_namespace_counts", {})
        # Only anonymous numeric counts are accepted; no peer/PID strings.
        counts = {}
        if isinstance(supplied_counts.get("socket_total"), int):
            counts["socket_total"] = supplied_counts["socket_total"]
        states = supplied_counts.get("tcp_state_counts", {})
        if isinstance(states, dict):
            counts["tcp_state_counts"] = {key: value for key, value in states.items()
                                          if isinstance(value, int) and value >= 0}
        self._counts = counts
        observed = facts.get("processes", [])
        verified = {}
        for pid, identity in self._processes.items():
            candidates = [p for p in observed if int(p["pid"]) == pid]
            if len(candidates) != 1:
                self.mark_incomplete("registered_pid_missing_or_duplicate:" + str(pid))
                continue
            process = candidates[0]
            if str(process.get("start_identity")) != identity:
                self.mark_incomplete("pid_identity_mismatch:" + str(pid))
                continue
            for field in ("fd_count", "nofile_soft", "nofile_hard", "environment"):
                if field not in process or process[field] is None:
                    self.mark_incomplete("missing_process_field:%s:%s" % (pid, field))
            env = process.get("environment") or {}
            verified[pid] = {
                "resource_id": self.owner_id + "/pid/" + str(pid) + "/start/" + identity,
                "pid": pid, "start_identity": identity,
                "fd_count": process.get("fd_count"), "nofile_soft": process.get("nofile_soft"),
                "nofile_hard": process.get("nofile_hard"),
                "allowed_env": {key: copy.deepcopy(value) for key, value in env.items()
                                if key in ALLOWED_ENV and (value is None or isinstance(value, str))},
                "connections": [],
            }
        for connection in facts.get("sockets", []):
            pid = int(connection["pid"])
            if pid not in verified:
                continue
            if str(connection.get("start_identity")) != self._processes[pid]:
                self.mark_incomplete("socket_identity_mismatch:" + str(pid))
                continue
            outbound = (connection.get("remote_ip"), connection.get("remote_port")) in self._peers
            inbound = ((connection.get("local_ip"), connection.get("local_port")) in self._peers
                       and connection.get("remote_ip") in {ip for ip, _ in self._peers})
            if not (outbound or inbound):
                continue
            verified[pid]["connections"].append({
                key: connection[key] for key in ("local_ip", "local_port", "remote_ip",
                "remote_port", "state", "socket_inode") if key in connection})
        if self.registration.get("source_hashes"):
            for detail in verified.values():
                process = next(p for p in observed if int(p["pid"]) == detail["pid"])
                detail["configuration_evidence"] = configuration_evidence(
                    process.get("environment") or {}, self.registration["source_hashes"],
                    observed=isinstance(process.get("environment"), dict))
        self._details = list(verified.values())
        return self.summary()

    def _line(self, line, *, observed_unix_s, source_id, byte_offset):
        text = line.decode("utf-8", errors="replace")
        kind = "native_tcp_engine" if _CPP_ERROR.search(text) else "KVTransferError" if _KV_ERROR.search(text) else None
        if kind is not None and self._first_error is None:
            # Detector clock is the first OBSERVATION time. Backend log time
            # remains raw evidence and is not promoted to a kernel timestamp.
            self._first_error = float(observed_unix_s)
            self._stop = True
            self._burst_end = min(self.deadline, self._first_error + self.burst_seconds)
            stamp = _BACKEND_TIME.search(text)
            self._first_error_evidence = {
                "observed_unix_s": self._first_error, "source_id": source_id,
                "byte_offset": byte_offset, "kind": kind,
                "backend_raw_time": stamp.group(1) if stamp else None,
                "backend_line": text[:MAX_LINE_BYTES],
            }

    def consume_stream(self, chunks, *, observed_unix_s, source_id="stream", final=False):
        if not math.isfinite(float(observed_unix_s)):
            raise ValueError("finite detector observation clock required")
        state = self._streams.setdefault(source_id, {
            "buffer": bytearray(), "offset": 0, "line_start": 0, "discarding": False})
        for chunk in chunks:
            self.drained_bytes += len(chunk)
            remaining = self.cap_bytes - len(self._retained)
            self._retained.extend(chunk[:remaining])
            if len(chunk) >= remaining and chunk:
                self.mark_incomplete("log_cap_reached")
            # Keep draining/parsing after retention caps; never close upstream.
            for part in chunk.splitlines(keepends=True):
                newline = part.endswith(b"\n")
                piece = part.rstrip(b"\r\n") if newline else part
                if not state["discarding"]:
                    if len(state["buffer"]) + len(piece) > MAX_LINE_BYTES:
                        self.mark_incomplete("line_cap_reached")
                        state["buffer"].clear()
                        state["discarding"] = True
                    else:
                        state["buffer"].extend(piece)
                state["offset"] += len(part)
                if newline:
                    if not state["discarding"]:
                        self._line(bytes(state["buffer"]), observed_unix_s=observed_unix_s,
                                   source_id=source_id, byte_offset=state["line_start"])
                    state["buffer"].clear()
                    state["discarding"] = False
                    state["line_start"] = state["offset"]
        if final and state["buffer"] and not state["discarding"]:
            self._line(bytes(state["buffer"]), observed_unix_s=observed_unix_s,
                       source_id=source_id, byte_offset=state["line_start"])
            state["buffer"].clear()
            state["line_start"] = state["offset"]
        return self.summary()

    def next_sample_delay(self, now_unix_s):
        end = self._burst_end if self._burst_end is not None else self.deadline
        remaining = min(end, self.deadline) - float(now_unix_s)
        if remaining <= 0:
            return None
        return min(.25 if self._first_error is not None else 1.0, remaining)

    def can_start_new_stage(self, now_unix_s):
        return (self.summary()["observation_complete"] and not self._stop
                and self._first_error is None and float(now_unix_s) < self.deadline)


def main(argv=None, *, readers=None, output=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="UNREGISTERED")
    parser.add_argument("--owner-id", default="UNREGISTERED")
    parser.add_argument("--deadline-unix-s", type=float)
    parser.add_argument("--registration-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--log", action="append", default=[])
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.observe and not args.dry_run:
        return _run_observe(args, readers)
    plan = {
        "mode": "plan", "live_collection_started": False,
        "run_id": args.run_id, "owner_id": args.owner_id,
        "deadline_unix_s": args.deadline_unix_s,
        "summary_schema_version": SCHEMA_VERSION,
        "registration_file": args.registration_file, "output_dir": args.output_dir,
        "logs": args.log, "capture_limit_bytes": MAX_CAPTURE_BYTES,
        "normal_interval_s": 1.0, "error_interval_s": .25, "error_burst_max_s": 60,
    }
    (output or sys.stdout).write(json.dumps(plan, indent=2) + "\n")
    return 0


CONFIG_SOURCE_PATHS = (
    "python/sglang/srt/environ.py",
    "python/sglang/srt/disaggregation/common/conn.py",
    "python/sglang/srt/disaggregation/mooncake/conn.py",
    "python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py",
)


def configuration_evidence(raw_env, source_hashes, *, observed):
    """Derive config from captured keys and registered source hashes.

    Known-unset uses source defaults. Denied/unknown capture uses no defaults.
    These are source-derived interpretations, not independently sampled flags.
    """
    sources = {p: source_hashes[p] for p in CONFIG_SOURCE_PATHS if p in source_hashes}
    bound = len(sources) == len(CONFIG_SOURCE_PATHS) and all(
        isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in sources.values())
    values = {}
    specs = (
        ("bootstrap_timeout_s", "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 300, int),
        ("waiting_timeout_s", "SGLANG_DISAGGREGATION_WAITING_TIMEOUT", 300, int),
        ("heartbeat_interval_s", "SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0, float),
        ("heartbeat_max_failure", "SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2, int),
        ("failed_session_probe", "SGLANG_ENABLE_FAILED_SESSION_PROBE", False, "bool"),
        ("failed_session_probe_interval_s", "SGLANG_FAILED_SESSION_PROBE_INTERVAL_S", 30.0, float),
    )
    def boolean(raw):
        if raw.lower() in {"true", "1", "yes", "y"}:
            return True
        if raw.lower() in {"false", "0", "no", "n"}:
            return False
        raise ValueError("invalid EnvBool; source does not strip whitespace")
    for label, name, default, parser in specs:
        raw = raw_env.get(name) if observed else None
        entry = {"env_key": name, "raw": raw, "parsed": None, "effective": None,
                 "origin": "capture_unknown", "parse_warning": None}
        if observed:
            if raw is None:
                parsed, origin = default, "source_default"
            else:
                try:
                    parsed = boolean(raw) if parser == "bool" else parser(raw)
                    origin = "worker_env"
                except (ValueError, TypeError):
                    parsed, origin = default, "invalid_worker_env_source_default"
                    entry["parse_warning"] = "SGLang EnvField uses declared default after invalid parse"
            effective = max(parsed, 2.0) if label == "heartbeat_interval_s" else (
                max(parsed, 1) if label == "heartbeat_max_failure" else parsed)
            entry.update(parsed=parsed, effective=effective, origin=origin)
        values[label] = entry
    raw = raw_env.get("SGLANG_KV_TRANSFER_TELEMETRY") if observed else None
    values["kv_transfer_telemetry"] = {
        "env_key": "SGLANG_KV_TRANSFER_TELEMETRY", "raw": raw,
        "parsed": bool(raw) if observed else None, "effective": bool(raw) if observed else None,
        "origin": ("worker_env" if raw is not None else "consumer_unset_semantics") if observed else "capture_unknown",
        "semantics": "bool(os.environ.get); nonempty string0 is true"}
    override = raw_env.get("SGLANG_MOONCAKE_TRANSFER_PROTOCOL") if observed else None
    raw = override if override is not None else raw_env.get("MOONCAKE_PROTOCOL") if observed else None
    protocol = {"env_key": "SGLANG_MOONCAKE_TRANSFER_PROTOCOL" if override is not None else "MOONCAKE_PROTOCOL",
                "raw": raw, "parsed": None, "effective": None, "origin": "capture_unknown", "error": None}
    valid = bool(observed)
    if observed:
        parsed = ("rdma" if raw is None else raw).strip().lower()
        protocol.update(parsed=parsed, origin="source_default" if raw is None else "worker_env")
        if parsed in {"rdma", "efa", "tcp", "nvlink", "nvlink_intra"}:
            protocol["effective"] = parsed
        else:
            protocol["error"] = "unsupported_protocol"
            valid = False
    values["transfer_protocol"] = protocol
    return {"capture_complete": bool(observed and bound), "configuration_valid": valid,
            "source_hashes": sources, "source_binding": "registered source hashes; not independently sampled",
            "values": values}


class SamplingRace(ValueError):
    """Process/FD/log changed during sampling; not evidence of a port cause."""


class LinuxProcReader:
    """Deferred OS primitives; construction and imports do not sample."""
    def __init__(self, *, max_read_bytes=MAX_CAPTURE_BYTES, max_fds=65536):
        self.max_read_bytes, self.max_fds = max_read_bytes, max_fds

    def read_text(self, path):
        with open(path, "rb") as target:
            data = target.read(self.max_read_bytes + 1)
        if len(data) > self.max_read_bytes:
            raise ValueError("reader_cap_reached")
        return data.decode("utf-8", errors="replace")

    def readlink(self, path):
        import os
        return os.readlink(path)

    def list_fd_targets(self, pid, proc_root="/proc"):
        from pathlib import Path
        import os
        targets = []
        try:
            for entry in (Path(proc_root) / str(pid) / "fd").iterdir():
                if len(targets) >= self.max_fds:
                    raise ValueError("fd_detail_cap_reached")
                try:
                    targets.append(os.readlink(entry))
                except FileNotFoundError as exc:
                    raise SamplingRace("fd_closed_during_sample") from exc
        except FileNotFoundError as exc:
            raise SamplingRace("process_closed_during_fd_sample") from exc
        return targets

    def iter_log_chunks(self, path, offset):
        import os
        with open(path, "rb") as target:
            if os.fstat(target.fileno()).st_size < offset:
                raise SamplingRace("log_truncated")
            target.seek(offset)
            while True:
                chunk = target.read(65536)
                if not chunk:
                    break
                yield chunk


_TCP_STATES = {
    "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV", "04": "FIN_WAIT1",
    "05": "FIN_WAIT2", "06": "TIME_WAIT", "07": "CLOSE", "08": "CLOSE_WAIT",
    "09": "LAST_ACK", "0A": "LISTEN", "0B": "CLOSING", "0C": "NEW_SYN_RECV",
}


def _stat_identity(text, expected_pid):
    prefix, tail = text.rsplit(")", 1)
    if int(prefix.split("(", 1)[0].strip()) != expected_pid:
        raise SamplingRace("stat_pid_changed")
    return tail.split()[19]


def _ipv4_endpoint(value):
    address, port = value.split(":")
    return ".".join(str(v) for v in bytes.fromhex(address)[::-1]), int(port, 16)


class LinuxIPv4FactProducer:
    """Deferred sampling of registered identities; no inferred ownership/ports."""
    def __init__(self, registration, *, reader=None):
        self.registration = copy.deepcopy(registration)
        self.reader = reader if reader is not None else LinuxProcReader()
        self.proc_root = registration.get("proc_root", "/proc").rstrip("/")
        self.cgroup = registration.get("cgroup_path", "/sys/fs/cgroup").rstrip("/")

    def collect(self, sampled_unix_s):
        reg, reader = self.registration, self.reader
        result = {
            "run_id": reg["run_id"], "owner_id": reg["owner_id"],
            "sampled_unix_s": float(sampled_unix_s),
            "source_status": {name: "ok" for name in REQUIRED_SOURCES},
            "processes": [], "sockets": [], "anonymous_namespace_counts": {},
            "port_range": None, "reserved_ports": None, "cgroup_memory_events": {},
            "cgroup_memory_peak_bytes": None, "read_errors": [],
        }
        def failed(source, error):
            kind = "denied" if isinstance(error, PermissionError) else (
                "sampling_race" if isinstance(error, (SamplingRace, FileNotFoundError)) else "incomplete")
            if result["source_status"].get(source) == "ok":
                result["source_status"][source] = kind
            result["read_errors"].append({"source": source, "kind": kind,
                                          "error_type": type(error).__name__})
        try:
            namespace = reader.readlink(self.proc_root + "/self/ns/pid")
            if not reg.get("pid_namespace") or namespace != reg["pid_namespace"]:
                raise SamplingRace("pid_namespace_mismatch")
            if not reg.get("net_namespace"):
                raise SamplingRace("network_namespace_registration_missing")
            if reader.readlink(self.proc_root + "/self/ns/net") != reg["net_namespace"]:
                raise SamplingRace("observer_network_namespace_mismatch")
        except (OSError, ValueError) as exc:
            failed("processes", exc)
            for source in REQUIRED_SOURCES:
                if result["source_status"][source] == "ok":
                    result["source_status"][source] = "not_collected_wrong_namespace"
            return result
        inode_owners = {}
        for registered in reg["processes"]:
            pid, expected = int(registered["pid"]), str(registered["start_identity"])
            base = self.proc_root + "/" + str(pid)
            try:
                identity = _stat_identity(reader.read_text(base + "/stat"), pid)
                if identity != expected:
                    raise SamplingRace("registered_pid_reused")
                if reader.readlink(base + "/ns/net") != reg["net_namespace"]:
                    raise SamplingRace("owned_pid_network_namespace_mismatch")
            except (OSError, ValueError, IndexError) as exc:
                failed("processes", exc)
                continue
            process = {"pid": pid, "start_identity": identity, "fd_count": None,
                       "nofile_soft": None, "nofile_hard": None, "environment": None}
            targets = []
            try:
                targets = (reader.list_fd_targets(pid, self.proc_root)
                           if isinstance(reader, LinuxProcReader) else reader.list_fd_targets(pid))
                process["fd_count"] = len(targets)
            except (OSError, ValueError) as exc:
                failed("sockets", SamplingRace("fd_closed_during_sample")
                       if isinstance(exc, FileNotFoundError) else exc)
            try:
                line = next(line for line in reader.read_text(base + "/limits").splitlines()
                            if line.startswith("Max open files"))
                parts = line.split()
                process["nofile_soft"], process["nofile_hard"] = int(parts[3]), int(parts[4])
            except (OSError, ValueError, StopIteration) as exc:
                failed("processes", exc)
            try:
                raw = reader.read_text(base + "/environ")
                env = dict.fromkeys(ALLOWED_ENV, None)
                for entry in raw.split("\0"):
                    key, separator, value = entry.partition("=")
                    if separator and key in ALLOWED_ENV:
                        env[key] = value
                process["environment"] = env
                evidence = configuration_evidence(env, reg.get("source_hashes", {}), observed=True)
                process["configuration_evidence"] = evidence
                if not evidence["capture_complete"] or not evidence["configuration_valid"]:
                    result["source_status"]["allowed_env"] = "unbound_or_invalid_configuration"
            except (OSError, ValueError) as exc:
                failed("allowed_env", exc)
            try:
                if _stat_identity(reader.read_text(base + "/stat"), pid) != expected:
                    raise SamplingRace("pid_changed_during_fd_join")
                if reader.readlink(base + "/ns/net") != reg["net_namespace"]:
                    raise SamplingRace("pid_network_namespace_changed_during_fd_join")
            except (OSError, ValueError, IndexError) as exc:
                failed("processes", exc)
                continue
            result["processes"].append(process)
            for target in targets:
                match = re.fullmatch(r"socket:\[(\d+)\]", target)
                if match and match.group(1) != "0":
                    inode_owners.setdefault(match.group(1), []).append((pid, identity))
        try:
            counts, total = {}, 0
            for line in reader.read_text(self.proc_root + "/net/tcp").splitlines()[1:]:
                if not line.strip():
                    continue
                fields = line.split()
                local_ip, local_port = _ipv4_endpoint(fields[1])
                remote_ip, remote_port = _ipv4_endpoint(fields[2])
                state = _TCP_STATES.get(fields[3], "UNKNOWN_" + fields[3])
                inode = fields[9]
                total += 1
                counts[state] = counts.get(state, 0) + 1
                # No owned liveFD match (including inode0/TIME_WAIT) means
                # anonymous namespace evidence, never a PID-owned claim.
                for pid, identity in inode_owners.get(inode, []):
                    result["sockets"].append({
                        "pid": pid, "start_identity": identity, "local_ip": local_ip,
                        "local_port": local_port, "remote_ip": remote_ip,
                        "remote_port": remote_port, "state": state, "socket_inode": inode})
            result["anonymous_namespace_counts"] = {"tcp_state_counts": counts, "socket_total": total}
        except (OSError, ValueError, IndexError) as exc:
            failed("sockets", exc)
            failed("namespace_counts", exc)
        try:
            result["port_range"] = list(map(int, reader.read_text(
                self.proc_root + "/sys/net/ipv4/ip_local_port_range").split()))
            if len(result["port_range"]) != 2:
                raise ValueError("invalid_port_range")
            result["reserved_ports"] = reader.read_text(
                self.proc_root + "/sys/net/ipv4/ip_local_reserved_ports").strip()
        except (OSError, ValueError) as exc:
            failed("port_range", exc)
        try:
            result["cgroup_memory_events"] = {
                key: int(value) for key, value in (line.split() for line in
                reader.read_text(self.cgroup + "/memory.events").splitlines())}
            result["cgroup_memory_peak_bytes"] = int(reader.read_text(self.cgroup + "/memory.peak").strip())
        except (OSError, ValueError) as exc:
            failed("cgroup", exc)
        return result


def collect_loop(observer, producer, *, read_logs, clock, sleep, emit_summary, emit_record):
    """Future synchronous loop with all readers/clocks/sinks injectable.

    Error flags stop new stages immediately through the published summary.
    Draining/sampling continues only through the bounded error/global window.
    """
    last_clock = None
    while True:
        now = float(clock())
        if last_clock is not None and now < last_clock:
            observer.mark_incomplete("collector_clock_regressed")
            emit_summary(observer.summary())
            return observer.summary()
        last_clock = now
        if now >= observer.deadline:
            observer.request_stop()
            emit_summary(observer.summary())
            return observer.summary()
        sample = producer.collect(now)
        observer.accept_sample(sample)
        try:
            for source_id, chunks in read_logs().items():
                observer.consume_stream(chunks, observed_unix_s=float(clock()), source_id=source_id)
        except (OSError, ValueError) as exc:
            observer.mark_incomplete("log_read_incomplete:" + type(exc).__name__)
        packet = {
            "summary": observer.summary(), "first_error_evidence": observer.first_error_evidence,
            "context": {key: sample.get(key) for key in
                        ("port_range", "reserved_ports", "cgroup_memory_events",
                         "cgroup_memory_peak_bytes", "source_status", "read_errors")},
            "drained_log_bytes": observer.drained_bytes,
        }
        try:
            saved = emit_record(packet)
        except OSError:
            saved = False
        if saved is False:
            observer.mark_incomplete("output_cap_or_write_incomplete")
        published = emit_summary(observer.summary())
        if published is False:
            observer.mark_incomplete("summary_cap_reached")
            minimal = observer.summary()
            minimal["owned_details"] = []
            if emit_summary(minimal) is False:
                raise RuntimeError("incomplete terminal summary publication failed")
        delay = observer.next_sample_delay(float(clock()))
        if delay is None:
            observer.request_stop()
            emit_summary(observer.summary())
            return observer.summary()
        sleep(delay)


class _LogTail:
    def __init__(self, paths, reader):
        self.paths, self.reader = list(paths), reader
        self.offsets = {path: 0 for path in paths}

    def _chunks(self, path):
        for chunk in self.reader.iter_log_chunks(path, self.offsets[path]):
            self.offsets[path] += len(chunk)
            yield chunk

    def read_logs(self):
        return {path: self._chunks(path) for path in self.paths}


class _ObserverFiles:
    """Observe-only writer; reserved latest+temporary fit within16MiB total."""
    def __init__(self, output_dir, cap_bytes=MAX_CAPTURE_BYTES):
        from pathlib import Path
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=False)
        self.summary_limit = min(128 * 1024, cap_bytes // 4)
        self.record_limit = cap_bytes - 2 * self.summary_limit
        self.saved_record_bytes = 0
        self.records = (self.root / "observations.jsonl").open("xb")

    def emit_record(self, packet):
        encoded = (json.dumps(packet, separators=(",", ":")) + "\n").encode("utf-8")
        if self.saved_record_bytes + len(encoded) > self.record_limit:
            return False
        self.records.write(encoded)
        self.records.flush()
        self.saved_record_bytes += len(encoded)
        return True

    def emit_summary(self, summary):
        import os
        encoded = (json.dumps(summary, separators=(",", ":")) + "\n").encode("utf-8")
        if len(encoded) > self.summary_limit:
            return False
        latest, temporary = self.root / "summary.latest.json", self.root / ".summary.latest.tmp"
        try:
            temporary.write_bytes(encoded)
            os.replace(temporary, latest)
        except OSError:
            # A missing summary and controller failure are fail-closed.
            # Do not leave the last healthy snapshot authoritative.
            try:
                latest.unlink(missing_ok=True)
            finally:
                raise
        return True

    def close(self):
        self.records.close()


def _run_observe(args, readers):
    """Explicit future collection only; unreachable in default/dry-run mode.

    The owned startup controller supplies verified endpoint/PID registration.
    Fresh engine endpoints do not prove lazy D->P KVArgs registration or health;
    the separately authorized realKV warmup establishes that later boundary.
    """
    import time
    dependencies = readers or {}
    reader = dependencies.get("reader") or LinuxProcReader()
    if not args.registration_file or not args.output_dir:
        raise ValueError("observe requires explicit registration-file and fresh output-dir")
    registration = json.loads(reader.read_text(args.registration_file))
    for key in ("run_id", "owner_id", "deadline_unix_s", "pid_namespace", "net_namespace", "processes",
                "peers", "source_hashes", "cgroup_path", "log_paths"):
        if key not in registration:
            raise ValueError("missing actual runtime registration: " + key)
    for option, key in ((args.run_id, "run_id"), (args.owner_id, "owner_id")):
        if option != "UNREGISTERED" and option != registration[key]:
            raise ValueError("CLI and registered runtime identity differ")
    if args.deadline_unix_s is not None and args.deadline_unix_s != registration["deadline_unix_s"]:
        raise ValueError("CLI may not replace the immutable registered deadline")
    paths = args.log or registration["log_paths"]
    if not paths or set(paths) != set(registration["log_paths"]):
        raise ValueError("observe only the exact registered owned log paths")
    observer = TransportObserver(registration)
    producer = dependencies.get("producer") or LinuxIPv4FactProducer(registration, reader=reader)
    tail = _LogTail(paths, reader)
    files = _ObserverFiles(args.output_dir)
    try:
        collect_loop(
            observer, producer,
            read_logs=dependencies.get("read_logs") or tail.read_logs,
            clock=dependencies.get("clock") or time.time,
            sleep=dependencies.get("sleep") or time.sleep,
            emit_summary=dependencies.get("emit_summary") or files.emit_summary,
            emit_record=dependencies.get("emit_record") or files.emit_record)
    finally:
        files.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
