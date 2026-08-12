"""Packaged GPU roofline profiles used before serving telemetry is available.

Values are normalized from ``LLM-simulator-main/simulator/configs/hardware.py``
so the online planner and the offline simulator start from the same HBM and
compute assumptions. Runtime snapshots and measured profiles remain the source
of truth once a worker is serving requests.
"""

from __future__ import annotations

from typing import Dict, Optional


def _profile(
    *,
    bandwidth: float,
    fp16: float,
    int8: float,
    onchip_buffer: float,
    vmemory: float,
) -> Dict[str, float]:
    """Expose simulator values under the planner's accepted field aliases."""

    return {
        "bandwidth": bandwidth,
        "memory_bandwidth_bytes_per_s": bandwidth,
        "FP16": fp16,
        "INT8": int8,
        "onchip_buffer": onchip_buffer,
        "onchip_buffer_bytes": onchip_buffer,
        "vmemory": vmemory,
        "vmemory_bytes": vmemory,
    }


# Keep this catalogue intentionally small and traceable. Unknown labels must
# provide explicit metadata rather than silently inheriting a different GPU.
HARDWARE_PROFILES: Dict[str, Dict[str, float]] = {
    "NVDA:A100": _profile(
        bandwidth=1555e9,
        fp16=312e12,
        int8=624e12,
        onchip_buffer=27648e3,
        vmemory=40e9,
    ),
    "NVDA:A100_40G:SXM": _profile(
        bandwidth=2039e9,
        fp16=312e12,
        int8=624e12,
        onchip_buffer=27648e3,
        vmemory=40e9,
    ),
    "NVDA:A100_80G:SXM": _profile(
        bandwidth=2039e9,
        fp16=312e12,
        int8=624e12,
        onchip_buffer=27648e3,
        vmemory=80e9,
    ),
    "NVDA:A40": _profile(
        bandwidth=696e9,
        fp16=149.7e12,
        int8=299.3e12,
        onchip_buffer=21504e3,
        vmemory=48e9,
    ),
    "NVDA:H100:SXM": _profile(
        bandwidth=3072e9,
        fp16=1979e12 / 2,
        int8=3958e12 / 2,
        onchip_buffer=33792e3,
        vmemory=80e9,
    ),
    "NVDA:H100:PCIe": _profile(
        bandwidth=2048e9,
        fp16=1513e12 / 2,
        int8=3026e12 / 2,
        onchip_buffer=29184e3,
        vmemory=80e9,
    ),
    "NVDA:H20": _profile(
        bandwidth=4096e9,
        fp16=148e12,
        int8=296e12,
        onchip_buffer=34560e3,
        vmemory=96e9,
    ),
    "NVDA:L40S": _profile(
        bandwidth=864e9,
        fp16=181e12,
        int8=362e12,
        onchip_buffer=36352e3,
        vmemory=48e9,
    ),
}


def resolve_hardware_profile(hardware: Optional[str]) -> Optional[Dict[str, float]]:
    """Resolve common worker labels without silently guessing unknown hardware."""

    if not hardware:
        return None
    normalized = hardware.upper().replace("-", "_").replace(" ", "_")
    if "H100" in normalized:
        key = "NVDA:H100:PCIe" if "PCIE" in normalized else "NVDA:H100:SXM"
    elif "A100" in normalized:
        if "80" in normalized:
            key = "NVDA:A100_80G:SXM"
        elif "40" in normalized:
            key = "NVDA:A100_40G:SXM"
        else:
            key = "NVDA:A100"
    elif "H20" in normalized:
        key = "NVDA:H20"
    elif "L40S" in normalized:
        key = "NVDA:L40S"
    else:
        return None
    return dict(HARDWARE_PROFILES[key])
