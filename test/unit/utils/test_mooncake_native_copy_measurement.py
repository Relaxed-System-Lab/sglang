import sys

from scripts.hexgent_pod_profile import measure_mooncake_native_copy


def test_native_copy_measurement_defaults_to_production_sync_api(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_mooncake_native_copy.py",
            "--role",
            "initiator",
            "--coord-dir",
            str(tmp_path),
        ],
    )

    args = measure_mooncake_native_copy.parse_args()

    assert args.size_mib == 250
    assert args.segments == 64
    assert args.sample_count == 64


def test_sample_offsets_cover_start_middle_and_end():
    offsets = measure_mooncake_native_copy.sample_offsets(
        size_bytes=16 * 1024 * 1024,
        sample_bytes=4096,
        sample_count=9,
    )

    assert offsets[0] == 0
    assert offsets[-1] == 16 * 1024 * 1024 - 4096
    assert len(offsets) == 9
    assert offsets == sorted(set(offsets))
