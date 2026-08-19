"""Measure Mooncake's production synchronous CUDA IPC copy path.

The native Mooncake engine must be built with the HexAgenT CUDA-event patch.
Those C++ events are the authoritative pure-copy measurement. This driver also
records Python API wall time and verifies samples spanning the target buffer.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import socket
import statistics
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("target", "initiator"), required=True)
    parser.add_argument("--coord-dir", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--size-mib", type=int, default=250)
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--sample-bytes", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def sample_offsets(
    *, size_bytes: int, sample_bytes: int, sample_count: int
) -> list[int]:
    if sample_bytes <= 0 or sample_bytes > size_bytes:
        raise ValueError("sample size must fit inside the transfer buffer")
    if sample_count < 2:
        raise ValueError("at least two samples are required")
    last = size_bytes - sample_bytes
    return sorted(
        {
            round(index * last / (sample_count - 1))
            for index in range(sample_count)
        }
    )


def segment_layout(size_bytes: int, segments: int) -> tuple[int, list[int]]:
    if segments <= 0 or size_bytes % segments:
        raise ValueError("transfer size must divide evenly across segments")
    segment_bytes = size_bytes // segments
    return segment_bytes, [index * segment_bytes for index in range(segments)]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def wait_for(path: Path, timeout_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        time.sleep(0.01)
    raise TimeoutError(f"timed out waiting for {path}")


def local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]


class CudaRuntime:
    DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        library = ctypes.util.find_library("cudart") or "libcudart.so.12"
        self.lib = ctypes.CDLL(library)
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]

    def check(self, status: int, operation: str) -> None:
        if status != 0:
            message = self.lib.cudaGetErrorString(status).decode("utf-8")
            raise RuntimeError(f"{operation} failed: {message} ({status})")

    def set_device(self, device: int) -> None:
        self.check(self.lib.cudaSetDevice(device), "cudaSetDevice")

    def memset(self, pointer: int, value: int, length: int) -> None:
        self.check(
            self.lib.cudaMemset(ctypes.c_void_p(pointer), value, length),
            "cudaMemset",
        )

    def synchronize(self) -> None:
        self.check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def read(self, pointer: int, length: int) -> bytes:
        output = ctypes.create_string_buffer(length)
        self.check(
            self.lib.cudaMemcpy(
                ctypes.cast(output, ctypes.c_void_p),
                ctypes.c_void_p(pointer),
                length,
                self.DEVICE_TO_HOST,
            ),
            "cudaMemcpyDeviceToHost",
        )
        return output.raw


def build_engine():
    from mooncake.engine import TransferEngine

    engine = TransferEngine()
    hostname = local_ip()
    if engine.initialize(hostname, "P2PHANDSHAKE", "nvlink_intra", "") != 0:
        raise RuntimeError("Mooncake intra-node engine initialization failed")
    return engine, f"{hostname}:{engine.get_rpc_port()}"


def verify_samples(
    runtime: CudaRuntime,
    pointer: int,
    value: int,
    offsets: list[int],
    sample_bytes: int,
) -> None:
    expected = bytes([value]) * sample_bytes
    for offset in offsets:
        actual = runtime.read(pointer + offset, sample_bytes)
        if actual != expected:
            raise RuntimeError(
                f"sample verification failed at offset={offset}: "
                f"first_bytes={actual[:16].hex()} expected={value:02x}"
            )


def run_target(args: argparse.Namespace) -> None:
    runtime = CudaRuntime()
    runtime.set_device(args.device_index)
    engine, target_name = build_engine()
    size_bytes = args.size_mib * 1024 * 1024
    pointer = engine.allocate_managed_buffer(size_bytes)
    if not pointer:
        raise RuntimeError("Mooncake target allocation failed")
    runtime.memset(pointer, 0, size_bytes)
    runtime.synchronize()
    atomic_json(
        args.coord_dir / "target.json",
        {"target": target_name, "destination_ptr": pointer, "bytes": size_bytes},
    )
    offsets = sample_offsets(
        size_bytes=size_bytes,
        sample_bytes=args.sample_bytes,
        sample_count=args.sample_count,
    )
    for iteration in range(args.warmup + args.repeats):
        request = wait_for(
            args.coord_dir / f"verify-{iteration}.json", args.timeout_seconds
        )
        runtime.synchronize()
        verify_samples(
            runtime,
            pointer,
            request["value"],
            offsets,
            args.sample_bytes,
        )
        atomic_json(
            args.coord_dir / f"verified-{iteration}.json",
            {
                "iteration": iteration,
                "checked_samples": len(offsets),
                "checked_bytes": len(offsets) * args.sample_bytes,
            },
        )
    engine.free_managed_buffer(pointer, size_bytes)


def run_initiator(args: argparse.Namespace) -> None:
    target = wait_for(args.coord_dir / "target.json", args.timeout_seconds)
    size_bytes = args.size_mib * 1024 * 1024
    if target["bytes"] != size_bytes:
        raise ValueError("initiator and target transfer sizes differ")
    runtime = CudaRuntime()
    runtime.set_device(args.device_index)
    engine, _ = build_engine()
    pointer = engine.allocate_managed_buffer(size_bytes)
    if not pointer:
        raise RuntimeError("Mooncake initiator allocation failed")
    segment_bytes, offsets = segment_layout(size_bytes, args.segments)
    sources = [pointer + offset for offset in offsets]
    destinations = [target["destination_ptr"] + offset for offset in offsets]
    lengths = [segment_bytes] * args.segments
    api_ms = []
    for iteration in range(args.warmup + args.repeats):
        value = (iteration + 31) % 251
        runtime.memset(pointer, value, size_bytes)
        runtime.synchronize()
        started_ns = time.perf_counter_ns()
        result = engine.batch_transfer_sync_write(
            target["target"], sources, destinations, lengths
        )
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        if result != 0:
            raise RuntimeError(f"Mooncake synchronous transfer failed: {result}")
        atomic_json(args.coord_dir / f"verify-{iteration}.json", {"value": value})
        verified = wait_for(
            args.coord_dir / f"verified-{iteration}.json", args.timeout_seconds
        )
        if verified["checked_samples"] < 2:
            raise RuntimeError(f"insufficient target verification: {verified}")
        if iteration >= args.warmup:
            api_ms.append(elapsed_ms)

    median_ms = statistics.median(api_ms)
    p95_ms = sorted(api_ms)[max(0, int(0.95 * len(api_ms)) - 1)]
    gib_per_second = size_bytes / (1024**3) / (median_ms / 1000)
    print(
        "MOONCAKE_NATIVE_COPY "
        f"size_mib={args.size_mib} segments={args.segments} "
        f"warmup={args.warmup} repeats={args.repeats} "
        f"api_median_ms={median_ms:.4f} api_p95_ms={p95_ms:.4f} "
        f"api_gib_s={gib_per_second:.2f}",
        flush=True,
    )
    engine.free_managed_buffer(pointer, size_bytes)


def main() -> None:
    args = parse_args()
    args.coord_dir.mkdir(parents=True, exist_ok=True)
    if args.role == "target":
        run_target(args)
    else:
        run_initiator(args)


if __name__ == "__main__":
    main()
