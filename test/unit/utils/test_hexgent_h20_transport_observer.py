"""Offline observer contracts. All runtime facts/streams are synthetic.

Root's sole bounded worker runs these tests; they never sample sockets, /proc,
FDs, environment, endpoints, containers, or GPUs.
"""
import argparse
import copy
import datetime
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/hexgent_real_trace_collectors/h20_transport_observer.py"
OWNER = "a" * 64
RUN = "c1-offline-fixture"
T0 = 1794072752.0
TCP_ERROR = (
    "E1107 17:32:32.268730   404 tcp_transport.cpp:720] "
    "TcpTransport::getConnection failed to create connection to 203.0.113.20:14000. "
    "Error: connect: Cannot assign requested address\n"
)
KV_ERROR = (
    "[2026-11-07 17:32:46 TP0] Decode transfer failed for request rank=0 "
    "decode_req.req.rid='fixture' decode_req.req.bootstrap_room=1 with exception "
    "KVTransferError(bootstrap_room=1): Failed to get kvcache from prefill instance, "
    "it might be dead\n"
)


def load():
    spec = importlib.util.spec_from_file_location("h20_transport_observer_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registration(deadline=None):
    return {
        "run_id": RUN, "owner_id": OWNER,
        "deadline_unix_s": T0 + 120 if deadline is None else deadline,
        "processes": [{"pid": 101, "start_identity": "111"},
                      {"pid": 102, "start_identity": "222"}],
        "peers": [{"ip": "203.0.113.20", "port": 14000, "kind": "tcp"},
                  {"ip": "203.0.113.20", "port": 15000, "kind": "session"}],
    }


def facts(at=T0):
    return {
        "run_id": RUN, "owner_id": OWNER, "sampled_unix_s": at,
        "source_status": {name: "ok" for name in
                          ("processes", "sockets", "namespace_counts",
                           "port_range", "cgroup", "allowed_env")},
        "processes": [
            {"pid": 101, "start_identity": "111", "fd_count": 12,
             "nofile_soft": 1048576, "nofile_hard": 1048576,
             "environment": {"SGLANG_MOONCAKE_TRANSFER_PROTOCOL": "tcp",
                             "SGLANG_ENABLE_FAILED_SESSION_PROBE": None,
                             "OPENAI_API_KEY": "DO_NOT_EXPORT_SECRET",
                             "AWS_SECRET_ACCESS_KEY": "DO_NOT_EXPORT_SECRET"}},
            {"pid": 102, "start_identity": "222", "fd_count": 14,
             "nofile_soft": 1048576, "nofile_hard": 1048576,
             "environment": {"SGLANG_MOONCAKE_TRANSFER_PROTOCOL": "tcp"}},
        ],
        "sockets": [
            {"pid": 101, "start_identity": "111", "local_ip": "192.0.2.1",
             "local_port": 41000, "remote_ip": "203.0.113.20", "remote_port": 14000,
             "state": "ESTABLISHED", "socket_inode": "501"},
            {"pid": 101, "start_identity": "111", "local_ip": "192.0.2.1",
             "local_port": 41001, "remote_ip": "198.51.100.88", "remote_port": 14000,
             "state": "ESTABLISHED", "socket_inode": "502"},
            {"pid": 101, "start_identity": "111", "local_ip": "192.0.2.1",
             "local_port": 41002, "remote_ip": "203.0.113.20", "remote_port": 16537,
             "state": "TIME_WAIT", "socket_inode": "503"},
            {"pid": 999, "start_identity": "999", "local_ip": "192.0.2.1",
             "local_port": 41003, "remote_ip": "203.0.113.20", "remote_port": 14000,
             "state": "ESTABLISHED", "socket_inode": "504"},
        ],
        "anonymous_namespace_counts": {
            "tcp_state_counts": {"ESTABLISHED": 8, "TIME_WAIT": 30},
            "socket_total": 38,
        },
        "port_range": [32768, 60999],
        "cgroup_memory_events": {"max": 0, "oom": 0, "oom_kill": 0},
    }


class DrainProbe:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.consumed = 0
        self.exhausted = False

    def __iter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk
        self.exhausted = True


class TransportObserverContracts(unittest.TestCase):
    def test_exact_owned_pid_start_identity_and_actual_peer_ports(self):
        observer = load().TransportObserver(registration())
        summary = observer.accept_sample(facts())
        self.assertTrue(summary["observation_complete"])
        self.assertFalse(summary["stop_requested"])
        self.assertEqual({row["pid"] for row in summary["owned_details"]}, {101, 102})
        connections = [connection for row in summary["owned_details"]
                       for connection in row["connections"]]
        self.assertEqual(len(connections), 1)
        self.assertEqual(connections[0]["remote_port"], 14000)
        self.assertEqual(connections[0]["remote_ip"], "203.0.113.20")
        # The old run's16537 is not silently registered in this fixture.
        self.assertNotIn("198.51.100.88", json.dumps(summary["owned_details"]))
        self.assertNotIn('"pid": 999', json.dumps(summary["owned_details"]))

    def test_pid_reuse_is_excluded_and_incompleteness_is_sticky(self):
        observer = load().TransportObserver(registration())
        changed = facts()
        changed["processes"][0]["start_identity"] = "REUSED"
        summary = observer.accept_sample(changed)
        self.assertFalse(summary["observation_complete"])
        self.assertTrue(summary["stop_requested"])
        self.assertNotIn(101, {row["pid"] for row in summary["owned_details"]})
        self.assertTrue(any("identity" in reason for reason in summary["incomplete_reasons"]))
        recovered = observer.accept_sample(facts(T0 + 1))
        self.assertFalse(recovered["observation_complete"])
        self.assertFalse(observer.can_start_new_stage(T0 + 1))

    def test_anonymous_namespace_counts_are_separate_immutable_diagnostics(self):
        observer = load().TransportObserver(registration())
        sample = facts()
        summary = observer.accept_sample(sample)
        self.assertEqual(summary["anonymous_namespace_counts"]["tcp_state_counts"]["TIME_WAIT"], 30)
        self.assertEqual(len(summary["owned_details"][0]["connections"]), 1)
        # Nonzero namespace totals do not become owned socket counts or thresholds.
        self.assertTrue(observer.can_start_new_stage(T0 + .1))
        sample["anonymous_namespace_counts"]["tcp_state_counts"]["TIME_WAIT"] = 999
        self.assertEqual(observer.summary()["anonymous_namespace_counts"]["tcp_state_counts"]["TIME_WAIT"], 30)

    def test_environment_is_only_explicit_allowed_keys_never_raw_dump(self):
        observer = load().TransportObserver(registration())
        summary = observer.accept_sample(facts())
        env = summary["owned_details"][0]["allowed_env"]
        self.assertEqual(env["SGLANG_MOONCAKE_TRANSFER_PROTOCOL"], "tcp")
        self.assertIn("SGLANG_ENABLE_FAILED_SESSION_PROBE", env)
        self.assertIsNone(env["SGLANG_ENABLE_FAILED_SESSION_PROBE"])
        self.assertNotIn("SGLANG_FAILED_SESSION_PROBE_INTERVAL_S", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("DO_NOT_EXPORT_SECRET", json.dumps(summary))

    def test_denied_missing_or_incomplete_required_data_never_allows_stage(self):
        for kind in ("denied", "missing_payload", "missing_status"):
            with self.subTest(kind=kind):
                observer = load().TransportObserver(registration())
                sample = facts()
                if kind == "denied":
                    sample["source_status"]["sockets"] = "denied"
                elif kind == "missing_payload":
                    del sample["sockets"]
                else:
                    del sample["source_status"]["cgroup"]
                summary = observer.accept_sample(sample)
                self.assertFalse(summary["observation_complete"])
                self.assertTrue(summary["stop_requested"])
                self.assertTrue(summary["incomplete_reasons"])
                self.assertFalse(observer.can_start_new_stage(T0 + .1))

    def test_cap_still_drains_every_chunk_and_detects_later_real_error(self):
        observer = load().TransportObserver(registration(), cap_bytes=32)
        observer.accept_sample(facts())
        stream = DrainProbe([b"ordinary log\n" * 5, TCP_ERROR[:70].encode(),
                             TCP_ERROR[70:].encode(), b"after error\n"])
        summary = observer.consume_stream(stream, observed_unix_s=T0 + 1)
        self.assertTrue(stream.exhausted)
        self.assertEqual(stream.consumed, 4)
        self.assertLessEqual(len(observer.retained_log), 32)
        self.assertFalse(summary["observation_complete"])
        self.assertTrue(summary["stop_requested"])
        self.assertTrue(any("cap" in reason for reason in summary["incomplete_reasons"]))
        self.assertIsNotNone(summary["first_error_unix_s"])

    def test_split_first_real_error_is_sticky_and_no_new_stage_or_retry(self):
        observer = load().TransportObserver(registration())
        observer.accept_sample(facts())
        stream = DrainProbe([TCP_ERROR[:41].encode(), TCP_ERROR[41:].encode(), KV_ERROR.encode()])
        summary = observer.consume_stream(stream, observed_unix_s=T0 + 20)
        self.assertTrue(stream.exhausted)
        self.assertEqual(summary["first_error_unix_s"], T0 + 20)
        self.assertEqual(observer.first_error_evidence["backend_raw_time"], "E1107 17:32:32.268730")
        self.assertTrue(summary["stop_requested"])
        self.assertFalse(observer.can_start_new_stage(T0 + 21))
        # New good samples cannot reset the first error or authorize a retry.
        again = observer.accept_sample(facts(T0 + 21))
        self.assertEqual(again["first_error_unix_s"], summary["first_error_unix_s"])
        self.assertTrue(again["stop_requested"])
        self.assertFalse(observer.can_start_new_stage(T0 + 22))

    def test_bootstrap_room_startup_warning_and_http503_are_not_tcp_or_kv_error(self):
        observer = load().TransportObserver(registration())
        observer.accept_sample(facts())
        harmless = [
            b"server_args=bootstrap_room timeout=300 no_oom\n",
            b"E1107 16:48:06.780401 57 transfer_metadata.cpp:877] Local segment descriptor not found\n",
            b'{"level":"ERROR","message":"request failed with server error","status":503}\n',
            b"[2026-11-07 17:32:32] No IB devices specified, auto discovery.\n",
        ]
        summary = observer.consume_stream(harmless, observed_unix_s=T0 + 1)
        self.assertIsNone(summary["first_error_unix_s"])
        self.assertFalse(summary["stop_requested"])
        self.assertTrue(observer.can_start_new_stage(T0 + 1))

    def test_first_kv_error_alone_stops_immediately(self):
        observer = load().TransportObserver(registration())
        observer.accept_sample(facts())
        summary = observer.consume_stream([KV_ERROR.encode()], observed_unix_s=T0 + 14)
        self.assertEqual(summary["first_error_unix_s"], T0 + 14)
        self.assertTrue(summary["stop_requested"])

    def test_four_hz_burst_is_clipped_to_global_deadline(self):
        observer = load().TransportObserver(registration(deadline=T0 + 10))
        observer.accept_sample(facts())
        self.assertEqual(observer.next_sample_delay(T0), 1.0)
        observer.consume_stream([TCP_ERROR.encode()], observed_unix_s=T0 + .3)
        self.assertEqual(observer.next_sample_delay(T0 + 1), .25)
        self.assertAlmostEqual(observer.next_sample_delay(T0 + 9.9), .1, places=5)
        self.assertIsNone(observer.next_sample_delay(T0 + 10))
        self.assertFalse(observer.can_start_new_stage(T0 + 10))

    def test_burst_does_not_run_beyond_sixty_seconds(self):
        observer = load().TransportObserver(registration(deadline=T0 + 200))
        observer.accept_sample(facts())
        observer.consume_stream([TCP_ERROR.encode()], observed_unix_s=T0 + .3)
        self.assertIsNone(observer.next_sample_delay(T0 + 60.3))
        self.assertFalse(observer.can_start_new_stage(T0 + 61))

    def test_run_owner_resource_identity_and_sample_time_never_move_backwards(self):
        original = registration()
        observer = load().TransportObserver(original)
        first = observer.accept_sample(facts(T0 + 2))
        original["owner_id"] = "b" * 64
        original["processes"][0]["start_identity"] = "CHANGED"
        wrong = facts(T0 + 3)
        wrong["owner_id"] = "b" * 64
        bad = observer.accept_sample(wrong)
        self.assertEqual(bad["run_id"], RUN)
        self.assertEqual(bad["owner_id"], OWNER)
        self.assertFalse(bad["observation_complete"])
        self.assertEqual(first["owned_details"][0]["resource_id"], OWNER + "/pid/101/start/111")
        regressed = observer.accept_sample(facts(T0 + 1))
        self.assertGreaterEqual(regressed["sampled_unix_s"], first["sampled_unix_s"])
        self.assertFalse(regressed["observation_complete"])

    def test_container_prefix_is_rejected_and_unsampled_observer_is_not_ready(self):
        module = load()
        short = registration()
        short["owner_id"] = "a" * 12
        with self.assertRaises(ValueError):
            module.TransportObserver(short)
        observer = module.TransportObserver(registration())
        self.assertFalse(observer.can_start_new_stage(T0))

    def test_default_cli_is_plan_only_and_never_calls_injected_readers(self):
        module = load()
        forbidden = mock.Mock(side_effect=AssertionError("runtime reader called in plan mode"))
        output = io.StringIO()
        with mock.patch("socket.socket", side_effect=AssertionError("socket called")), \
             mock.patch("subprocess.Popen", side_effect=AssertionError("subprocess called")), \
             mock.patch("os.getenv", side_effect=AssertionError("env called")), \
             mock.patch("os.readlink", side_effect=AssertionError("proc called")):
            result = module.main(
                ["--run-id", RUN, "--owner-id", OWNER, "--deadline-unix-s", str(T0 + 120)],
                readers={"sample": forbidden}, output=output)
        self.assertEqual(result, 0)
        forbidden.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["mode"], "plan")
        self.assertFalse(plan["live_collection_started"])
        self.assertEqual(plan["run_id"], RUN)
        self.assertEqual(plan["owner_id"], OWNER)

    def test_import_has_no_runtime_os_or_reader_side_effects(self):
        code = compile(SOURCE.read_text(), str(SOURCE), "exec")
        namespace = {"__name__": "observer_import_contract"}
        with mock.patch("builtins.open", side_effect=AssertionError("file opened")), \
             mock.patch("os.open", side_effect=AssertionError("fd opened")), \
             mock.patch("os.getenv", side_effect=AssertionError("env sampled")), \
             mock.patch("os.getpid", side_effect=AssertionError("pid sampled")), \
             mock.patch("os.readlink", side_effect=AssertionError("proc sampled")), \
             mock.patch("socket.socket", side_effect=AssertionError("socket sampled")), \
             mock.patch("subprocess.Popen", side_effect=AssertionError("process started")):
            exec(code, namespace)
        self.assertIn("TransportObserver", namespace)
        self.assertIn("main", namespace)



SOURCE_HASHES = {
    "python/sglang/srt/environ.py": "b" * 64,
    "python/sglang/srt/disaggregation/common/conn.py": "c" * 64,
    "python/sglang/srt/disaggregation/mooncake/conn.py": "d" * 64,
    "python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py": "e" * 64,
}


def proc_stat(pid, identity):
    # Linux stat field22 is index19 after the parenthesized comm field.
    after = ["S"] + ["0"] * 19
    after[19] = str(identity)
    return str(pid) + " (worker name with spaces) " + " ".join(after) + "\n"


def tcp_row(index, local, remote, state, inode):
    return ("%d: %s %s %s 00000000:00000000 00:00000000 00000000 1000 0 %s 1\n"
            % (index, local, remote, state, inode))


class FakeProcReader:
    """Only synthetic paths exist; unexpected reads fail the fixture."""
    def __init__(self):
        self.calls = []
        self.files = {
            "/fake-proc/101/stat": proc_stat(101, "111"),
            "/fake-proc/102/stat": proc_stat(102, "222"),
            "/fake-proc/101/limits": "Limit Soft Limit Hard Limit Units\nMax open files 1048576 1048576 files\n",
            "/fake-proc/102/limits": "Limit Soft Limit Hard Limit Units\nMax open files 1048576 1048576 files\n",
            "/fake-proc/101/environ": "SGLANG_MOONCAKE_TRANSFER_PROTOCOL=tcp\0SGLANG_KV_TRANSFER_TELEMETRY=0\0OPENAI_API_KEY=DO_NOT_EXPORT_SECRET\0",
            "/fake-proc/102/environ": "SGLANG_MOONCAKE_TRANSFER_PROTOCOL=tcp\0",
            "/fake-proc/net/tcp": (
                "sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
                + tcp_row(0, "010200C0:A028", "147100CB:36B0", "01", "501")
                + tcp_row(1, "010200C0:A029", "147100CB:36B0", "06", "0")),
            "/fake-proc/sys/net/ipv4/ip_local_port_range": "32768 60999\n",
            "/fake-proc/sys/net/ipv4/ip_local_reserved_ports": "40000-40010\n",
            "/fake-cgroup/memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
            "/fake-cgroup/memory.peak": "1000000\n",
        }
        self.fds = {101: ["socket:[501]", "/dev/null"], 102: ["socket:[601]"]}
        self.namespace = "pid:[4026532222]"
        self.net_namespace = "net:[4026533333]"
        self.pid_net_namespaces = {101: "net:[4026533333]", 102: "net:[4026533333]"}
        self.denied = set()

    def read_text(self, path):
        key = str(path)
        self.calls.append(("read_text", key))
        if key in self.denied:
            raise PermissionError(key)
        if key not in self.files:
            raise AssertionError("unexpected fixture read: " + key)
        return self.files[key]

    def list_fd_targets(self, pid):
        self.calls.append(("list_fd_targets", pid))
        return list(self.fds[pid])

    def readlink(self, path):
        self.calls.append(("readlink", str(path)))
        key = str(path)
        if key == "/fake-proc/self/ns/pid":
            return self.namespace
        if key == "/fake-proc/self/ns/net":
            return self.net_namespace
        for pid, namespace in self.pid_net_namespaces.items():
            if key == "/fake-proc/" + str(pid) + "/ns/net":
                return namespace
        raise AssertionError("unexpected fixture readlink")


def producer_registration(deadline=None):
    result = registration(deadline)
    result.update({
        "proc_root": "/fake-proc", "cgroup_path": "/fake-cgroup",
        "pid_namespace": "pid:[4026532222]", "net_namespace": "net:[4026533333]",
        "source_hashes": dict(SOURCE_HASHES),
        "log_paths": ["/fake-run/prefill.log", "/fake-run/decode.log"],
    })
    return result


def same_host_peer_reader(*, owned_reverse):
    """Synthetic accepted socket and its same-sample reverse TCP tuple."""
    reader = FakeProcReader()
    reader.fds[101] = ["socket:[500]", "socket:[501]"]
    reader.fds[102] = ["socket:[601]"] if owned_reverse else []
    reader.files["/fake-proc/net/tcp"] = (
        "sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
        + tcp_row(0, "147100CB:36B0", "00000000:0000", "0A", "500")
        + tcp_row(1, "147100CB:36B0", "147100CB:A028", "01", "501")
        + tcp_row(2, "147100CB:A028", "147100CB:36B0", "01", "601"))
    return reader


class DeferredFactProducerContracts(unittest.TestCase):
    def test_ipv4_inode_join_uses_only_registered_pid_identity_and_peer(self):
        module = load()
        reader = FakeProcReader()
        producer = module.LinuxIPv4FactProducer(producer_registration(), reader=reader)
        sample = producer.collect(T0)
        summary = module.TransportObserver(producer_registration()).accept_sample(sample)
        self.assertTrue(summary["observation_complete"], summary["incomplete_reasons"])
        self.assertEqual(summary["anonymous_namespace_counts"]["socket_total"], 2)
        self.assertEqual(summary["anonymous_namespace_counts"]["tcp_state_counts"]["TIME_WAIT"], 1)
        owned = summary["owned_details"][0]
        self.assertEqual(owned["start_identity"], "111")
        self.assertEqual(owned["fd_count"], 2)
        self.assertEqual(owned["connections"][0]["remote_ip"], "203.0.113.20")
        self.assertEqual(owned["connections"][0]["remote_port"], 14000)
        self.assertEqual(owned["connections"][0]["socket_inode"], "501")
        self.assertNotIn("0", [c["socket_inode"] for row in summary["owned_details"] for c in row["connections"]])
        # TIME_WAIT inode0 still contributes to anonymous namespace pressure.
        self.assertNotIn("198.51.100.88", json.dumps(summary["owned_details"]))
        self.assertNotIn("DO_NOT_EXPORT_SECRET", json.dumps(sample))
        self.assertFalse(any("/999/" in str(call) for call in reader.calls))
        # An accepted inbound connection also has a current owned reverse FD.
        inbound = same_host_peer_reader(owned_reverse=True)
        sample = module.LinuxIPv4FactProducer(producer_registration(), reader=inbound).collect(T0)
        joined = module.TransportObserver(producer_registration()).accept_sample(sample)
        self.assertEqual(len(joined["owned_details"][0]["connections"]), 1)
        self.assertEqual(joined["owned_details"][0]["connections"][0]["local_port"], 14000)
        self.assertNotIn("198.51.100.88", json.dumps(joined["owned_details"]))

    def test_same_host_foreign_inbound_is_anonymous_without_owned_reverse_tuple(self):
        module = load()
        reader = same_host_peer_reader(owned_reverse=False)
        reg = producer_registration()
        sample = module.LinuxIPv4FactProducer(reg, reader=reader).collect(T0)
        summary = module.TransportObserver(reg).accept_sample(sample)
        self.assertTrue(summary["observation_complete"], summary["incomplete_reasons"])
        # The accepted server FD is ours. The ephemeral client FD is not.
        self.assertIn("501", {row["socket_inode"] for row in sample["sockets"]})
        self.assertNotIn("601", {row["socket_inode"] for row in sample["sockets"]})
        self.assertEqual(summary["anonymous_namespace_counts"]["socket_total"], 3)
        self.assertEqual(summary["anonymous_namespace_counts"]["tcp_state_counts"]["ESTABLISHED"], 2)
        self.assertEqual([connection for row in summary["owned_details"]
                          for connection in row["connections"]], [])
        self.assertNotIn('"remote_port": 41000', json.dumps(summary["owned_details"]))
        self.assertFalse(any("/999/" in str(call) for call in reader.calls))

    def test_same_sample_owned_reverse_tuple_keeps_legitimate_inbound_and_outbound(self):
        module = load()
        reader = same_host_peer_reader(owned_reverse=True)
        reg = producer_registration()
        sample = module.LinuxIPv4FactProducer(reg, reader=reader).collect(T0)
        summary = module.TransportObserver(reg).accept_sample(sample)
        self.assertTrue(summary["observation_complete"], summary["incomplete_reasons"])
        connections = {row["pid"]: row["connections"] for row in summary["owned_details"]}
        self.assertEqual([row["socket_inode"] for row in connections[101]], ["501"])
        self.assertEqual([row["socket_inode"] for row in connections[102]], ["601"])
        inbound, outbound = connections[101][0], connections[102][0]
        self.assertEqual((inbound["local_ip"], inbound["local_port"], inbound["remote_ip"], inbound["remote_port"]),
                         (outbound["remote_ip"], outbound["remote_port"], outbound["local_ip"], outbound["local_port"]))
        self.assertEqual(summary["anonymous_namespace_counts"]["socket_total"], 3)
        self.assertNotIn("500", {row["socket_inode"] for values in connections.values() for row in values})
        self.assertFalse(any("/999/" in str(call) for call in reader.calls))

    def test_inbound_reverse_tuple_is_not_reused_after_pid_or_fd_loses_ownership(self):
        module = load()
        for change in ("reused_pid", "missing_fd", "closed_fd"):
            with self.subTest(change=change):
                reader = same_host_peer_reader(owned_reverse=True)
                reg = producer_registration()
                producer = module.LinuxIPv4FactProducer(reg, reader=reader)
                observer = module.TransportObserver(reg)
                first = observer.accept_sample(producer.collect(T0))
                self.assertEqual(len(first["owned_details"][0]["connections"]), 1)
                if change == "reused_pid":
                    reader.files["/fake-proc/102/stat"] = proc_stat(102, "REUSED")
                elif change == "missing_fd":
                    reader.fds[102] = []
                else:
                    original = reader.list_fd_targets
                    def closing(pid):
                        if pid == 102:
                            raise FileNotFoundError("synthetic registered reverse FD closed")
                        return original(pid)
                    reader.list_fd_targets = closing
                sample = producer.collect(T0 + 1)
                summary = observer.accept_sample(sample)
                self.assertNotIn("601", {row["socket_inode"] for row in sample["sockets"]})
                self.assertEqual(summary["anonymous_namespace_counts"]["socket_total"], 3)
                detail = next(row for row in summary["owned_details"] if row["pid"] == 101)
                self.assertEqual(detail["connections"], [])
                self.assertEqual(summary["observation_complete"], change == "missing_fd")
                self.assertFalse(any("/999/" in str(call) for call in reader.calls))

    def test_denied_environment_is_unknown_not_an_unset_default(self):
        module = load()
        reader = FakeProcReader()
        reader.denied.add("/fake-proc/101/environ")
        sample = module.LinuxIPv4FactProducer(producer_registration(), reader=reader).collect(T0)
        summary = module.TransportObserver(producer_registration()).accept_sample(sample)
        self.assertFalse(summary["observation_complete"])
        self.assertTrue(summary["stop_requested"])
        self.assertIsNone(sample["processes"][0]["environment"])
        self.assertNotIn("DO_NOT_EXPORT_SECRET", json.dumps(sample))

    def test_pid_namespace_mismatch_never_becomes_complete(self):
        module = load()
        reader = FakeProcReader()
        reader.namespace = "pid:[FOREIGN]"
        sample = module.LinuxIPv4FactProducer(producer_registration(), reader=reader).collect(T0)
        summary = module.TransportObserver(producer_registration()).accept_sample(sample)
        self.assertFalse(summary["observation_complete"])
        self.assertTrue(summary["stop_requested"])

    def test_observer_network_namespace_mismatch_blocks_network_reads(self):
        module = load()
        for mode in ("mismatch", "missing_registration"):
            with self.subTest(mode=mode):
                reader = FakeProcReader()
                dynamic = producer_registration()
                if mode == "mismatch":
                    reader.net_namespace = "net:[FOREIGN]"
                else:
                    del dynamic["net_namespace"]
                sample = module.LinuxIPv4FactProducer(dynamic, reader=reader).collect(T0)
                summary = module.TransportObserver(dynamic).accept_sample(sample)
                self.assertFalse(summary["observation_complete"])
                self.assertTrue(summary["stop_requested"])
                self.assertTrue(summary["incomplete_reasons"])
                self.assertEqual(sample["anonymous_namespace_counts"], {})
                self.assertFalse(any(call[0] == "read_text" and
                                     ("/net/tcp" in str(call[1]) or "ip_local_port_range" in str(call[1]))
                                     for call in reader.calls))

    def test_owned_pid_in_another_network_namespace_is_not_joined(self):
        module = load()
        reader = FakeProcReader()
        reader.pid_net_namespaces[101] = "net:[FOREIGN-WORKER]"
        dynamic = producer_registration()
        sample = module.LinuxIPv4FactProducer(dynamic, reader=reader).collect(T0)
        summary = module.TransportObserver(dynamic).accept_sample(sample)
        self.assertFalse(summary["observation_complete"])
        self.assertTrue(summary["stop_requested"])
        self.assertNotIn(101, {row["pid"] for row in summary["owned_details"]})
        self.assertFalse(any(call == ("list_fd_targets", 101) for call in reader.calls))

    def test_pid_reuse_during_fd_join_is_detected(self):
        module = load()
        reader = FakeProcReader()
        original_read = reader.read_text
        stat_reads = [0]
        def changing(path):
            if str(path) == "/fake-proc/101/stat":
                stat_reads[0] += 1
                if stat_reads[0] >= 2:
                    return proc_stat(101, "REUSED")
            return original_read(path)
        reader.read_text = changing
        sample = module.LinuxIPv4FactProducer(producer_registration(), reader=reader).collect(T0)
        summary = module.TransportObserver(producer_registration()).accept_sample(sample)
        self.assertFalse(summary["observation_complete"])
        self.assertNotIn(101, {row["pid"] for row in summary["owned_details"]})
        closing = FakeProcReader()
        original_fds = closing.list_fd_targets
        def fd_closed(pid):
            if pid == 101:
                raise FileNotFoundError("synthetic FD closed between listing and readlink")
            return original_fds(pid)
        closing.list_fd_targets = fd_closed
        raced = module.LinuxIPv4FactProducer(producer_registration(), reader=closing).collect(T0)
        incomplete = module.TransportObserver(producer_registration()).accept_sample(raced)
        self.assertFalse(incomplete["observation_complete"])
        self.assertTrue(any("race" in r for r in incomplete["incomplete_reasons"]))

    def test_config_evidence_matches_exact_source_semantics_and_provenance(self):
        module = load()
        raw = {
            "SGLANG_MOONCAKE_TRANSFER_PROTOCOL": " TCP ",
            "MOONCAKE_PROTOCOL": "rdma",
            "SGLANG_KV_TRANSFER_TELEMETRY": "0",
            "SGLANG_ENABLE_FAILED_SESSION_PROBE": "0",
            "SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL": "0.5",
            "SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE": "0",
        }
        evidence = module.configuration_evidence(raw, SOURCE_HASHES, observed=True)
        self.assertTrue(evidence["capture_complete"])
        self.assertEqual(evidence["source_hashes"], SOURCE_HASHES)
        self.assertEqual(evidence["values"]["transfer_protocol"]["effective"], "tcp")
        self.assertEqual(evidence["values"]["transfer_protocol"]["raw"], " TCP ")
        self.assertTrue(evidence["values"]["kv_transfer_telemetry"]["effective"])
        self.assertFalse(evidence["values"]["failed_session_probe"]["effective"])
        self.assertEqual(evidence["values"]["heartbeat_interval_s"]["effective"], 2.0)
        self.assertEqual(evidence["values"]["heartbeat_max_failure"]["effective"], 1)
        self.assertEqual(evidence["values"]["waiting_timeout_s"]["origin"], "source_default")
        self.assertEqual(evidence["values"]["waiting_timeout_s"]["effective"], 300)

    def test_unknown_capture_has_no_assumed_defaults_and_empty_protocol_does_not_fall_back(self):
        module = load()
        unknown = module.configuration_evidence({}, SOURCE_HASHES, observed=False)
        self.assertFalse(unknown["capture_complete"])
        self.assertIsNone(unknown["values"]["failed_session_probe"]["effective"])
        self.assertIsNone(unknown["values"]["waiting_timeout_s"]["effective"])
        bad = module.configuration_evidence(
            {"SGLANG_MOONCAKE_TRANSFER_PROTOCOL": "", "MOONCAKE_PROTOCOL": "tcp"},
            SOURCE_HASHES, observed=True)
        self.assertIsNone(bad["values"]["transfer_protocol"]["effective"])
        self.assertEqual(bad["values"]["transfer_protocol"]["error"], "unsupported_protocol")
        spaced_bool = module.configuration_evidence(
            {"SGLANG_ENABLE_FAILED_SESSION_PROBE": " true "}, SOURCE_HASHES, observed=True)
        self.assertFalse(spaced_bool["values"]["failed_session_probe"]["effective"])
        self.assertEqual(spaced_bool["values"]["failed_session_probe"]["origin"], "invalid_worker_env_source_default")

    def test_missing_source_hash_never_claims_provenance_complete(self):
        evidence = load().configuration_evidence(
            {"SGLANG_MOONCAKE_TRANSFER_PROTOCOL": "tcp"}, {}, observed=True)
        self.assertFalse(evidence["capture_complete"])

    def test_fake_loop_publishes_exact_summary_and_observed_clock_burst(self):
        module = load()
        reg = producer_registration(deadline=T0 + 1)
        observer = module.TransportObserver(reg)
        now = [T0]
        sleeps, summaries, packets = [], [], []
        class FakeProducer:
            def collect(self, sampled):
                return facts(sampled)
        logs = [DrainProbe([TCP_ERROR[:40].encode(), TCP_ERROR[40:].encode(), b"tail\n"])]
        def read_logs():
            return {"prefill": logs.pop(0)} if logs else {}
        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds
        result = module.collect_loop(
            observer, FakeProducer(), read_logs=read_logs,
            clock=lambda: now[0], sleep=sleep,
            emit_summary=lambda summary: summaries.append(copy.deepcopy(summary)) or True,
            emit_record=lambda packet: packets.append(copy.deepcopy(packet)) or True)
        self.assertEqual(result["first_error_unix_s"], T0)
        self.assertTrue(result["stop_requested"])
        self.assertEqual(sleeps, [.25] * 4)
        self.assertEqual(now[0], T0 + 1)
        self.assertTrue(summaries)
        self.assertEqual(set(summaries[-1]), {
            "schema_version", "run_id", "owner_id", "sampled_unix_s",
            "observation_complete", "first_error_unix_s", "stop_requested",
            "incomplete_reasons", "anonymous_namespace_counts", "owned_details"})
        self.assertEqual(packets[0]["first_error_evidence"]["backend_raw_time"], "E1107 17:32:32.268730")

    def test_output_cap_marks_incomplete_and_still_drains_future_chunks(self):
        module = load()
        reg = producer_registration(deadline=T0 + 2)
        observer = module.TransportObserver(reg, cap_bytes=16)
        now = [T0]
        probes, summaries = [], []
        class FakeProducer:
            def collect(self, sampled):
                return facts(sampled)
        def read_logs():
            stream = DrainProbe([b"first log\n", b"second log\n", b"third log\n"])
            probes.append(stream)
            return {"prefill": stream}
        result = module.collect_loop(
            observer, FakeProducer(), read_logs=read_logs,
            clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            emit_summary=lambda summary: summaries.append(copy.deepcopy(summary)) or True,
            emit_record=lambda packet: False)
        self.assertTrue(all(probe.exhausted and probe.consumed == 3 for probe in probes))
        self.assertFalse(result["observation_complete"])
        self.assertTrue(result["stop_requested"])
        self.assertTrue(any("cap" in reason for reason in result["incomplete_reasons"]))
        self.assertFalse(summaries[-1]["observation_complete"])

    def test_observe_plus_dry_run_never_reads_registration_or_starts_collector(self):
        module = load()
        forbidden = mock.Mock(side_effect=AssertionError("live operation called"))
        output = io.StringIO()
        with mock.patch("builtins.open", side_effect=AssertionError("file opened")), \
             mock.patch("os.open", side_effect=AssertionError("fd opened")), \
             mock.patch("os.getenv", side_effect=AssertionError("env read")), \
             mock.patch("socket.socket", side_effect=AssertionError("socket read")):
            rc = module.main(
                ["--observe", "--dry-run", "--registration-file", "/DO_NOT_READ",
                 "--output-dir", "/DO_NOT_CREATE", "--log", "/DO_NOT_TAIL"],
                readers={"sample": forbidden}, output=output)
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "plan")
        forbidden.assert_not_called()


class RegistryAndPacketPrivacyContracts(unittest.TestCase):
    def test_empty_dynamic_process_or_peer_registry_is_rejected(self):
        module = load()
        for field in ("processes", "peers"):
            with self.subTest(field=field):
                dynamic = registration()
                dynamic[field] = []
                with self.assertRaises(ValueError):
                    module.TransportObserver(dynamic)

    def test_static_plan_cli_without_dynamic_registration_stays_inert(self):
        module = load()
        forbidden = mock.Mock(side_effect=AssertionError("reader called by static plan"))
        output = io.StringIO()
        result = module.main([], readers={"sample": forbidden}, output=output)
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "plan")
        self.assertFalse(json.loads(output.getvalue())["live_collection_started"])
        forbidden.assert_not_called()

    def test_loop_packets_never_dump_unwhitelisted_normalized_environment(self):
        module = load()
        observer = module.TransportObserver(registration(deadline=T0 + .5))
        now = [T0]
        summaries, packets = [], []
        class FullKeyProducer:
            def collect(self, sampled):
                result = facts(sampled)
                result["raw_environment"] = {"PRIVATE_TOKEN": "DO_NOT_EXPORT_SECRET"}
                result["processes"][0]["raw_environ"] = "DO_NOT_EXPORT_SECRET"
                result["processes"].append({
                    "pid": 999, "start_identity": "foreign", "fd_count": 100,
                    "nofile_soft": 1000, "nofile_hard": 1000,
                    "environment": {"PRIVATE_TOKEN": "DO_NOT_EXPORT_SECRET"},
                })
                return result
        module.collect_loop(
            observer, FullKeyProducer(), read_logs=lambda: {},
            clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            emit_summary=lambda summary: summaries.append(copy.deepcopy(summary)) or True,
            emit_record=lambda packet: packets.append(copy.deepcopy(packet)) or True)
        emitted = json.dumps({"summaries": summaries, "jsonl_packets": packets})
        self.assertNotIn("DO_NOT_EXPORT_SECRET", emitted)
        self.assertNotIn("OPENAI_API_KEY", emitted)
        self.assertNotIn("PRIVATE_TOKEN", emitted)
        self.assertNotIn("raw_environ", emitted)


class PrefixOrFlushFailureSink:
    """An I/O error may leave a physical prefix even though write did not return."""
    def __init__(self, phase):
        self.phase = phase
        self.persisted = bytearray()
        self.write_calls = 0
        self.flush_calls = 0

    def write(self, data):
        self.write_calls += 1
        if self.phase == "write" and self.write_calls == 1:
            self.persisted.extend(data[:17])
            raise OSError("synthetic prefix persisted before write error")
        self.persisted.extend(data)
        return len(data)

    def flush(self):
        self.flush_calls += 1
        if self.phase == "flush" and self.flush_calls == 1:
            raise OSError("synthetic flush error after complete write")

    def close(self):
        pass


class ObserverFileFailureContracts(unittest.TestCase):
    def test_record_write_or_flush_failure_reserves_bytes_and_disables_later_appends(self):
        module = load()
        packet = {"synthetic_payload": "x" * 512}
        encoded = (json.dumps(packet, separators=(",", ":")) + "\n").encode("utf-8")
        for phase in ("write", "flush"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                files = module._ObserverFiles(Path(directory) / "output", cap_bytes=4096)
                files.records.close()
                sink = PrefixOrFlushFailureSink(phase)
                files.records = sink
                try:
                    with self.assertRaises(OSError):
                        files.emit_record(packet)
                    self.assertEqual(files.saved_record_bytes, len(encoded))
                    writes, flushes = sink.write_calls, sink.flush_calls
                    for _ in range(10):
                        self.assertIs(files.emit_record(packet), False)
                    self.assertEqual((sink.write_calls, sink.flush_calls), (writes, flushes))
                    self.assertEqual(files.saved_record_bytes, len(encoded))
                    self.assertLessEqual(len(sink.persisted), files.saved_record_bytes)
                    self.assertLessEqual(files.saved_record_bytes, files.record_limit)
                    summary = module.TransportObserver(registration()).summary()
                    self.assertTrue(files.emit_summary(summary))
                    saved = json.loads((files.root / "summary.latest.json").read_text())
                    self.assertFalse(saved["observation_complete"])
                    self.assertLessEqual(files.saved_record_bytes + 2 * files.summary_limit, 4096)
                finally:
                    files.close()

    def test_record_sink_failure_keeps_incomplete_summary_and_later_log_draining(self):
        module = load()
        for phase in ("write", "flush"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                files = module._ObserverFiles(Path(directory) / "output", cap_bytes=16384)
                files.records.close()
                sink = PrefixOrFlushFailureSink(phase)
                files.records = sink
                observer = module.TransportObserver(registration(deadline=T0 + 2))
                now = [T0]
                probes = []
                class FakeProducer:
                    def collect(self, sampled):
                        return facts(sampled)
                def logs():
                    probe = DrainProbe([b"first log\n", b"second log\n", b"third log\n"])
                    probes.append(probe)
                    return {"prefill": probe}
                try:
                    result = module.collect_loop(observer, FakeProducer(), read_logs=logs,
                        clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
                        emit_summary=files.emit_summary, emit_record=files.emit_record)
                    self.assertEqual(len(probes), 2)
                    self.assertTrue(all(probe.exhausted and probe.consumed == 3 for probe in probes))
                    self.assertFalse(result["observation_complete"])
                    self.assertTrue(result["stop_requested"])
                    self.assertTrue(any("write" in reason for reason in result["incomplete_reasons"]))
                    self.assertEqual(sink.write_calls, 1)
                    self.assertEqual(sink.flush_calls, 0 if phase == "write" else 1)
                    self.assertGreaterEqual(files.saved_record_bytes, len(sink.persisted))
                    self.assertLessEqual(files.saved_record_bytes + 2 * files.summary_limit, 16384)
                    saved = json.loads((files.root / "summary.latest.json").read_text())
                    self.assertFalse(saved["observation_complete"])
                    self.assertTrue(saved["stop_requested"])
                finally:
                    files.close()


if __name__ == "__main__":
    unittest.main()
