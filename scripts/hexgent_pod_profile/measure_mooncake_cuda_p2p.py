"""Measure Mooncake's CUDA-stream P2P API with CUDA events.

This is deliberately separate from the production ``batch_transfer_sync``
measurement.  It times ``transfer_write_on_cuda`` on the exact CUDA stream
passed to Mooncake, then makes the destination stream wait on the end event
before checking destination data.  A successful check establishes that the
timed stream event covers data visibility at the destination for this API.
"""

from __future__ import annotations

import argparse
import socket
import statistics

import torch
from mooncake.engine import TransferEngine


def local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This microbenchmark requires two CUDA devices")

    engine = TransferEngine()
    hostname = local_ip()
    if engine.initialize(hostname, "P2PHANDSHAKE", "nvlink_intra", "") != 0:
        raise RuntimeError("Mooncake nvlink_intra engine initialization failed")
    target = f"{hostname}:{engine.get_rpc_port()}"

    source_device = torch.device("cuda:0")
    destination_device = torch.device("cuda:1")
    source_stream = torch.cuda.Stream(source_device)
    destination_stream = torch.cuda.Stream(destination_device)

    for size_mib in args.sizes_mib:
        size_bytes = size_mib * 1024 * 1024
        source = torch.empty(size_bytes, dtype=torch.uint8, device=source_device)
        destination = torch.empty(size_bytes, dtype=torch.uint8, device=destination_device)
        if engine.register_memory(source.data_ptr(), source.nbytes) != 0:
            raise RuntimeError("failed to register source memory")
        if engine.register_memory(destination.data_ptr(), destination.nbytes) != 0:
            raise RuntimeError("failed to register destination memory")

        elapsed_ms = []
        for iteration in range(args.repeats + 1):
            value = (iteration + 17) % 251
            with torch.cuda.device(source_device):
                source.fill_(value)
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(source_device))
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                source_stream.wait_event(ready)
                with torch.cuda.stream(source_stream):
                    start.record()
                    status = engine.transfer_write_on_cuda(
                        target,
                        source.data_ptr(),
                        destination.data_ptr(),
                        source.nbytes,
                        source_stream.cuda_stream,
                    )
                    if status != 0:
                        raise RuntimeError(f"transfer_write_on_cuda failed: {status}")
                    end.record()
            # This dependency and check ensure the timed end event is not just
            # launch submission: destination data must be visible after it.
            with torch.cuda.device(destination_device), torch.cuda.stream(destination_stream):
                destination_stream.wait_event(end)
                checksum = destination.to(dtype=torch.int64).sum()
            destination_stream.synchronize()
            expected = value * size_bytes
            if checksum.item() != expected:
                raise RuntimeError(
                    f"destination verification failed: got={checksum.item()} expected={expected}"
                )
            if iteration:
                elapsed_ms.append(start.elapsed_time(end))

        median_ms = statistics.median(elapsed_ms)
        p95_ms = sorted(elapsed_ms)[max(0, int(0.95 * len(elapsed_ms)) - 1)]
        gib_per_s = size_bytes / (1024**3) / (median_ms / 1000)
        print(
            f"CUDA_P2P size_mib={size_mib} repeats={len(elapsed_ms)} "
            f"median_ms={median_ms:.4f} p95_ms={p95_ms:.4f} gib_s={gib_per_s:.2f}",
            flush=True,
        )
        engine.unregister_memory(source.data_ptr())
        engine.unregister_memory(destination.data_ptr())


if __name__ == "__main__":
    main()
