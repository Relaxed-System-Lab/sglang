# 2026-08-05 RunPod Asset Curation

## Objective

The first retirement archive preserved nearly every non-source artifact from
the old RunPod volume. It also retained historical scheduling runs that must not
be interpreted as current HexAgenT performance. This session produced a smaller
curated archive that preserves reusable measurements but removes obsolete or
invalid scheduler experiments.

## Profile Validation

All retained profile JSONL files were parsed during curation:

- Files: 61.
- Rows: 47,020.
- Rows carrying an error: 0.
- Rows with zero output tokens: 0.

The retained 50/50/20 Llama-8B profile contains 50 ShareGPT workflows, 50 BFCL
workflows, and 20 HotpotQA workflows. Original H100 Llama/Qwen real-trace
profiles, older 10-workflow Llama-8B exclusive profiles, SLO profile inputs, and
formal workflow traces were also retained.

## Retained Evidence

- H100 and Llama-8B profiles and traces.
- Raw KV-transfer GPU-event/no-cache measurements and CUDA peer-access probes.
- 2P+2D transfer measurements and capacity/snapshot calibration evidence.
- Mooncake NVLink patches, exact binary overlays/backups, and their hashes.
- Python/platform manifests and admission integration evidence.

## Removed Evidence

All historical baseline-versus-HexAgenT performance runs were removed from the
curated archive. Those runs predate the current combination of revisioned plans,
nonblocking fallback, safe priority updates, and token-aware admission. Their
known confounders include revision-0-only fallback, incorrect queue/request
caps, duplicated workflows, old capacity/transfer models, partial launches,
run-order effects, and cache-state contamination.

The following broad categories were therefore removed:

- `formal_results*`, `stress_results`, and old high-load comparisons.
- Old 1P+1D/2P+2D/4P+4D scheduling comparison directories.
- Standalone worker/router logs tied to those comparisons.
- Intermediate admission, warm-start, horizon-priority, timeout, and transfer
  tuning runs.
- Duplicate Mooncake source/build trees, old source snapshots/backups, model and
  package caches, PID files outside retained raw measurements, and temporary
  files.

High-level debugging conclusions remain in repository reports and migration
notes, but the removed data must not be used for scheduler performance claims.

## Curated Archive

```text
real_trace_profiles/runpod_workspace_backup_20260804/
  runpod_hexgent_assets_curated_20260805.tar.gz
  runpod_hexgent_assets_curated_20260805.tar.gz.sha256
```

The archive is verified by listing all members, checking the required profile,
trace, measurement, environment, and integration paths, confirming removed
performance-result categories are absent, and comparing the archive with the
adjacent `.sha256` file. Exact size and hash are recorded outside the archive in
the main migration pack to avoid a self-referential checksum.

The original broad archive and temporary extraction directories are superseded
and safe to delete. The current execution environment rejected filesystem
delete commands, so their physical deletion is pending; only the curated
archive is authoritative.
