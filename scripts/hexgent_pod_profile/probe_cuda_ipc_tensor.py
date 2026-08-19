"""Verify CUDA IPC for a PyTorch-allocated tensor across two processes.

Mooncake registers PyTorch-owned KV-cache memory. This probe isolates CUDA IPC
from Mooncake by exporting the exact allocation class with ``cudaIpcGetMemHandle``
and opening it from a peer process on another visible GPU.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import time
from pathlib import Path

import torch

from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)


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


def serialize_ipc_handle(handle: cudaIpcMemHandle_t) -> str:
    raw = ctypes.string_at(ctypes.byref(handle), ctypes.sizeof(handle))
    return base64.b64encode(raw).decode("ascii")


def deserialize_ipc_handle(payload: str) -> cudaIpcMemHandle_t:
    raw = base64.b64decode(payload.encode("ascii"))
    if len(raw) != ctypes.sizeof(cudaIpcMemHandle_t):
        raise ValueError("invalid CUDA IPC handle length")
    return cudaIpcMemHandle_t.from_buffer_copy(raw)


def selected_device(index: int) -> torch.device:
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"--device-index={index} is outside the {torch.cuda.device_count()} visible CUDA devices"
        )
    return torch.device(f"cuda:{index}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("target", "initiator"), required=True)
    parser.add_argument("--coord-dir", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--size-mib", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def run_target(args: argparse.Namespace) -> None:
    device = selected_device(args.device_index)
    torch.cuda.set_device(device)
    size_bytes = args.size_mib * 1024 * 1024
    tensor = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    tensor.fill_(7)
    torch.cuda.synchronize(device)
    handle = CudaRTLibrary().cudaIpcGetMemHandle(tensor.data_ptr())
    atomic_json(
        args.coord_dir / "target.json",
        {
            "handle": serialize_ipc_handle(handle),
            "size_bytes": tensor.nbytes,
            "device_index": args.device_index,
        },
    )
    result = wait_for(args.coord_dir / "result.json", args.timeout_seconds)
    print(f"CUDA_IPC_TENSOR_TARGET result={json.dumps(result, sort_keys=True)}", flush=True)


def run_initiator(args: argparse.Namespace) -> None:
    device = selected_device(args.device_index)
    torch.cuda.set_device(device)
    target = wait_for(args.coord_dir / "target.json", args.timeout_seconds)
    try:
        pointer = CudaRTLibrary().cudaIpcOpenMemHandle(
            deserialize_ipc_handle(target["handle"])
        )
    except RuntimeError as exc:
        result = {"ok": False, "error": str(exc), "device_index": args.device_index}
    else:
        result = {"ok": True, "pointer": int(pointer.value), "device_index": args.device_index}
    atomic_json(args.coord_dir / "result.json", result)
    print(f"CUDA_IPC_TENSOR_INITIATOR result={json.dumps(result, sort_keys=True)}", flush=True)
    if not result["ok"]:
        raise RuntimeError(result["error"])


def main() -> None:
    args = parse_args()
    args.coord_dir.mkdir(parents=True, exist_ok=True)
    if args.role == "target":
        run_target(args)
    else:
        run_initiator(args)


if __name__ == "__main__":
    main()
