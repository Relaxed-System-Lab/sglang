"""The two fixed H20 BF16/TP2 worker layouts shared by launcher and window."""


_PRESETS = {
    "1p1d_tp2": (
        ("P0", "prefill", "prefill", "prefill", "bootstrap", (0, 1)),
        ("D0", "decode", "decode", "decode", None, (2, 3)),
    ),
    "2p2d_tp2": (
        ("P0", "prefill", "prefill", "prefill", "bootstrap", (0, 1)),
        ("P1", "prefill_1", "prefill", "prefill_1", "bootstrap_1", (2, 3)),
        ("D0", "decode", "decode", "decode", None, (4, 5)),
        ("D1", "decode_1", "decode", "decode_1", None, (6, 7)),
    ),
}


def reference_worker_layout(ports, *, topology="1p1d_tp2"):
    """Return explicit instance identities for one of the two approved presets."""
    if topology not in _PRESETS:
        raise ValueError("expected 1p1d_tp2 or 2p2d_tp2")
    values = {name: int(value) for name, value in ports.items()}
    if (len(set(values.values())) != len(values)
            or any(not 0 < value < 65536 for value in values.values())):
        raise ValueError("caller-supplied ports must be distinct and within 1..65535")
    rows = []
    for identity, component, role, http, bootstrap, devices in _PRESETS[topology]:
        port = values[http]
        rows.append({
            "instance_id": identity, "component": component, "role": role,
            "worker_url": f"http://127.0.0.1:{port}", "http_port": port,
            "bootstrap_port": values[bootstrap] if bootstrap is not None else None,
            "cuda_devices": list(devices), "tp_size": 2,
        })
    return rows
