# 2026-08-04 RunPod Network Volume Retirement

## Purpose

This session audited the CPU-only RunPod network volume before retirement. The
local checkout is the authoritative source tree because it contains changes
newer than `/workspace/sglang`; remote source files must not overwrite local
files.

No serving, replay, router, or planner process was running during the backup.

## Directly Verified Remote Assets

The following assets were present under `/workspace`:

- `hexgent_e2e`: profiles, formal and stress results, calibration data, worker
  logs, planner telemetry, replay traces, and launch evidence.
- `hexgent_profiles/H100_REAL_TRACE_PROFILES_FINAL`: the retained H100 Llama and
  Qwen real-trace profiles.
- `h100_real_trace_profiles_final.tar.gz`: the corresponding packaged profile.
- `hexgent_admission_mock`: admission integration evidence.
- `third_party/mooncake_engine_overlays`, Mooncake binary backups, install
  artifacts, and the two intra-node NVLink patches.
- The Python serving virtual environment, Cargo/Rust caches, and Hugging Face
  model cache.

The latest experiment assets included:

- `formal_results_20260804`
- `formal_traces_20260804`
- `llama8b_profile_50_50_20_20260804`
- `logs_4p4d_revisioned_20260804`
- `slo_profiles`

## Local Backup

The non-reconstructible assets were initially archived remotely and copied to:

```text
real_trace_profiles/runpod_workspace_backup_20260804/
  runpod_hexgent_assets_20260804.tar.gz
  runpod_hexgent_assets_20260804.tar.gz.sha256
```

Verification:

- Archive size: 267,351,865 bytes.
- Archive members: 3,750.
- Remote and local SHA-256:
  `dbac2a8c6812367bb0466fc7f7bd1aa4a5ac4af1f101b4769f1bbe18e042e500`.
- The local archive was opened successfully and all required directories listed
  above were found.

The archive also contains:

- Base and serving-venv `pip freeze` manifests.
- OS and Python platform information.
- SHA-256 values for the retained Mooncake patches and binary overlays.
- Launch scripts and historical logs needed to audit the experiments.

This broad retirement archive was superseded on 2026-08-05 by the curated
archive described in `2026-08-05-runpod-asset-curation.md`. It must no longer be
used as an experiment source. Its physical deletion remains pending because the
current execution policy rejected filesystem delete operations.

## Intentionally Excluded Assets

The following assets were not copied because they are reconstructible or
superseded:

- `/workspace/hub`: Hugging Face model cache; models will be downloaded again
  for the A100 environment.
- `/workspace/hexgent_sglang_venv`: the virtual environment; exact package
  manifests are retained instead.
- `/workspace/.cargo` and `/workspace/.rustup`: tool caches/toolchains.
- `/workspace/hexgent_e2e/source_snapshots`: 5.3 GiB of old full source
  snapshots.
- `/workspace/hexgent_e2e/source_backups`: old source backups superseded by the
  local checkout.
- `/workspace/hexgent_e2e/tmp`: temporary files.

The remote source archive and `sglang-sync` helpers were retained inside the
backup only as historical evidence. They are not authoritative implementation
inputs.

## Environment Evidence

The retired Pod used Ubuntu 22.04.5 and Python 3.11.10. The serving environment
included editable local installs of SGLang, the model gateway, and Python router
bindings, together with SGLang kernel 0.3.21 and Mooncake transfer engine 0.3.9.
The complete package list is inside the archive and should be used for version
comparison, not installed blindly on A100.

## Next A100 Pod

The next 8xA100 run must proceed in this order:

1. Sync the authoritative local source tree to persistent storage on the new
   Pod.
2. Run the environment preflight before starting model downloads or workers.
3. Install CUDA/A100-compatible SGLang kernel, Mooncake, router build, and
   system dependencies; do not reuse Hopper-specific binaries blindly.
4. Download the selected Llama/Qwen model variants and verify model precision,
   TP, GPU memory, and transport backend.
5. Profile the same retained workflows exclusively on A100 to construct
   model/GPU-matched SLO and service profiles.
6. Run 1P+1D smoke tests, then the intended multi-worker topology.
7. Calibrate arrival rates and token-aware admission using prefill batch tokens,
   decode running batch, global wait, planner revision coverage, GPU telemetry,
   and KV-transfer telemetry.
8. Compare baseline and HexAgenT only after confirming nonzero planner plan
   execution and matching workflow DAG/input/output lengths.
