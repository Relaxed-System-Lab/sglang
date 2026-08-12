#!/usr/bin/env python3
"""Run the HexAgenT sidecar from a source checkout without importing SGLang APIs."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "python" / "sglang"
    if not package_root.is_dir():
        raise SystemExit(f"SGLang package not found at {package_root}")

    # Importing sglang normally loads frontend and serving dependencies that the
    # standalone planner does not use. A namespace package keeps overlay tests
    # lightweight while resolving the real sglang.hexgent modules from source.
    package = types.ModuleType("sglang")
    package.__path__ = [str(package_root)]
    package.__package__ = "sglang"
    sys.modules["sglang"] = package
    runpy.run_module("sglang.hexgent.sidecar", run_name="__main__")


if __name__ == "__main__":
    main()
