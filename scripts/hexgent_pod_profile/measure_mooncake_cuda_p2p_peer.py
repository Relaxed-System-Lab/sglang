"""Two-process CUDA-event benchmark for Mooncake intra-node NVLink.

Run one ``target`` process on GPU 1 and one ``initiator`` process on GPU 0.
The target publishes a registered VRAM address.  The initiator records CUDA
events around ``transfer_write_on_cuda`` on the same stream supplied to
Mooncake.  The callback used by Mooncake waits for transfer completion before
the end event can execute; the target independently verifies the copied bytes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import socket
import statistics
import time
from pathlib import Path

import torch
from mooncake.engine import TransferEngine


def local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("target", "initiator"), required=True)
    parser.add_argument("--coord-dir", type=Path, required=True)
    parser.add_argument("--size-mib", type=int, default=64)
    parser.add_argument(
        "--segments",
        type=int,
        default=1,
        help="Split the registered buffer into this many equal transfer requests.",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--transfer-mode",
        choices=("stream-callback", "sync"),
        default="stream-callback",
        help=(
            "Mooncake API to benchmark. SGLang's current Mooncake KV path uses "
            "the synchronous API; stream-callback is retained to expose its "
            "separate CUDA callback behavior."
        ),
    )
    parser.add_argument(
        "--submit-in-thread",
        action="store_true",
        help="Run the synchronous Mooncake call in a Python worker thread.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index within the process-visible device set.",
    )
    return parser.parse_args()


def build_engine() -> tuple[TransferEngine, str]:
    engine = TransferEngine()
    hostname = local_ip()
    if engine.initialize(hostname, "P2PHANDSHAKE", "nvlink_intra", "") != 0:
        raise RuntimeError("Mooncake nvlink_intra engine initialization failed")
    return engine, f"{hostname}:{engine.get_rpc_port()}"


def selected_device(index: int) -> torch.device:
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"--device-index={index} is outside the {torch.cuda.device_count()} visible CUDA devices"
        )
    return torch.device(f"cuda:{index}")


def segment_layout(size_bytes: int, segments: int) -> tuple[int, list[int]]:
    if segments <= 0:
        raise ValueError("--segments must be positive")
    if size_bytes % segments:
        raise ValueError(
            f"{size_bytes} bytes cannot be split evenly across {segments} segments"
        )
    segment_bytes = size_bytes // segments
    return segment_bytes, [index * segment_bytes for index in range(segments)]


def run_target(args: argparse.Namespace) -> None:
    device = selected_device(args.device_index)
    torch.cuda.set_device(device)
    size_bytes = args.size_mib * 1024 * 1024
    engine, target_name = build_engine()
    destination = torch.zeros(size_bytes, dtype=torch.uint8, device=device)
    if engine.register_memory(destination.data_ptr(), destination.nbytes) != 0:
        raise RuntimeError("failed to register target VRAM")
    atomic_json(
        args.coord_dir / "target.json",
        {"target": target_name, "destination_ptr": destination.data_ptr(), "bytes": size_bytes},
    )
    for iteration in range(args.repeats):
        request = wait_for(args.coord_dir / f"verify-{iteration}.json", args.timeout_seconds)
        expected = request["value"] * size_bytes
        torch.cuda.synchronize(device)
        actual = destination.to(dtype=torch.int64).sum().item()
        atomic_json(
            args.coord_dir / f"verified-{iteration}.json",
            {"iteration": iteration, "actual": actual, "expected": expected},
        )
    engine.unregister_memory(destination.data_ptr())


def run_initiator(args: argparse.Namespace) -> None:
    target = wait_for(args.coord_dir / "target.json", args.timeout_seconds)
    size_bytes = args.size_mib * 1024 * 1024
    if target["bytes"] != size_bytes:
        raise ValueError("initiator and target size mismatch")
    segment_bytes, offsets = segment_layout(size_bytes, args.segments)
    device = selected_device(args.device_index)
    torch.cuda.set_device(device)
    engine, _ = build_engine()
    source = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    if engine.register_memory(source.data_ptr(), source.nbytes) != 0:
        raise RuntimeError("failed to register source VRAM")
    stream = torch.cuda.Stream(device)
    elapsed_ms = []
    api_elapsed_ms = []
    for iteration in range(args.repeats):
        value = (iteration + 31) % 251
        source.fill_(value)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream.wait_event(ready)
        api_started_at = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            start.record()
            source_ptrs = [source.data_ptr() + offset for offset in offsets]
            destination_ptrs = [target["destination_ptr"] + offset for offset in offsets]
            lengths = [segment_bytes] * args.segments
            if args.transfer_mode == "stream-callback":
                engine.batch_transfer_write_on_cuda(
                    target["target"],
                    source_ptrs,
                    destination_ptrs,
                    lengths,
                    stream.cuda_stream,
                )
            else:
                transfer_args = (
                    target["target"],
                    source_ptrs,
                    destination_ptrs,
                    lengths,
                )
                if args.submit_in_thread:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        result = executor.submit(
                            engine.batch_transfer_sync_write, *transfer_args
                        ).result()
                else:
                    result = engine.batch_transfer_sync_write(*transfer_args)
                if result != 0:
                    raise RuntimeError(f"Mooncake synchronous transfer failed: {result}")
            end.record()
        end.synchronize()
        api_elapsed_ms.append((time.perf_counter_ns() - api_started_at) / 1_000_000)
        atomic_json(args.coord_dir / f"verify-{iteration}.json", {"value": value})
        verified = wait_for(args.coord_dir / f"verified-{iteration}.json", args.timeout_seconds)
        if verified["actual"] != verified["expected"]:
            raise RuntimeError(f"destination verification failed: {verified}")
        elapsed_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(elapsed_ms)
    p95_ms = sorted(elapsed_ms)[max(0, int(0.95 * len(elapsed_ms)) - 1)]
    median_api_ms = statistics.median(api_elapsed_ms)
    api_gib_s = size_bytes / (1024**3) / (median_api_ms / 1000)
    print(
        f"CUDA_P2P_PEER mode={args.transfer_mode} size_mib={args.size_mib} "
        f"segments={args.segments} submit_in_thread={args.submit_in_thread} "
        f"repeats={args.repeats} "
        f"cuda_event_ms={median_ms:.4f} cuda_event_p95_ms={p95_ms:.4f} "
        f"api_median_ms={median_api_ms:.4f} api_gib_s={api_gib_s:.2f}",
        flush=True,
    )
    engine.unregister_memory(source.data_ptr())


def main() -> None:
    args = parse_args()
    args.coord_dir.mkdir(parents=True, exist_ok=True)
    if args.role == "target":
        run_target(args)
    else:
        run_initiator(args)


if __name__ == "__main__":
    main()
