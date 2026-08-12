#!/usr/bin/env python3
"""Verify a persistent RunPod environment before starting HexAgenT serving.

The check intentionally performs no installation or model download.  It
captures the container-local runtime state that is frequently lost when a pod
is replaced, while checking that source code and Hugging Face cache remain on
the mounted network volume.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any


REQUIRED_IMPORTS = (
    "torch",
    "sglang",
    "sgl_kernel",
    "mooncake",
    "sglang_router",
)
REQUIRED_TOOLS = ("cargo", "maturin", "protoc", "nvidia-smi")


def _command(args: list[str], *, timeout_s: int = 10) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _import_module(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # Native-extension import errors are diagnostic data.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        version = importlib.metadata.version(name.replace("_", "-"))
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "unknown")
    return {
        "ok": True,
        "version": version,
        "path": getattr(module, "__file__", None),
    }


def _cuda_library_report() -> dict[str, Any]:
    ldconfig = _command(["ldconfig", "-p"])
    listing = ldconfig.get("stdout", "") if ldconfig.get("ok") else ""
    return {
        "ldconfig": ldconfig,
        "visible": {
            library: library in listing
            for library in ("libcudart.so", "libnvrtc.so", "libibverbs.so", "librdmacm.so")
        },
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
    }


def _paths_report(source_dir: Path, workspace: Path) -> dict[str, Any]:
    cache_dir = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", workspace / "hub"))
    return {
        "workspace": {"path": str(workspace), "exists": workspace.is_dir()},
        "source": {
            "path": str(source_dir),
            "exists": source_dir.is_dir(),
            "hexgent_exists": (source_dir / "python" / "sglang" / "hexgent").is_dir(),
        },
        "huggingface": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "cache_path": str(cache_dir),
            "cache_exists": cache_dir.is_dir(),
        },
        "venv": {
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
            "python": sys.executable,
        },
    }


def build_report(*, source_dir: Path, workspace: Path) -> dict[str, Any]:
    imports = {name: _import_module(name) for name in REQUIRED_IMPORTS}
    torch_report: dict[str, Any] = {}
    if imports["torch"]["ok"]:
        import torch

        torch_report = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }

    commands = {
        tool: {"path": shutil.which(tool), "probe": _command([tool, "--version"])}
        for tool in REQUIRED_TOOLS
    }
    commands["nvidia-smi-query"] = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    commands["nvidia-smi-topology"] = _command(["nvidia-smi", "topo", "-m"])

    return {
        "timestamp_unix_s": time.time(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "paths": _paths_report(source_dir, workspace),
        "imports": imports,
        "torch": torch_report,
        "commands": commands,
        "cuda_libraries": _cuda_library_report(),
    }


def failures(report: dict[str, Any], *, require_gpu: bool) -> list[str]:
    result: list[str] = []
    paths = report["paths"]
    if not paths["workspace"]["exists"]:
        result.append("network workspace is unavailable")
    if not paths["source"]["hexgent_exists"]:
        result.append("HexAgenT source is unavailable on the network workspace")
    if not paths["huggingface"]["cache_exists"]:
        result.append("Hugging Face cache directory is unavailable")
    for name, item in report["imports"].items():
        if not item["ok"]:
            result.append(f"Python import failed: {name}: {item['error']}")
    for tool in REQUIRED_TOOLS:
        if not report["commands"][tool]["path"]:
            result.append(f"required tool is unavailable: {tool}")
    if require_gpu:
        torch = report["torch"]
        if not torch.get("cuda_available"):
            result.append("Torch cannot access CUDA")
        if int(torch.get("cuda_device_count", 0)) < 1:
            result.append("no CUDA device is visible")
        if not report["commands"]["nvidia-smi-query"]["ok"]:
            result.append("nvidia-smi GPU query failed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", default=os.environ.get("HEXGENT_SRC_DIR", "/workspace/sglang")
    )
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        source_dir=Path(args.source_dir), workspace=Path(args.workspace)
    )
    report["failures"] = failures(report, require_gpu=args.require_gpu)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
