# HexAgenT v0.5.15 Evidence Index

This directory keeps only the current v0.5.15 design and real-GPU evidence.
Historical session notes, provisional topology sweeps and superseded experiment
summaries were removed from the working tree; their Git history remains
available when needed.

- [v0.5.15 port design](../2026-08-12-hexgent-v0.5.15-port-design.md):
  architecture, ownership and compatibility decisions.
- [A100 functional validation](2026-08-17-v0.5.15-a100-validation.md):
  release-matched implementation and end-to-end correctness validation.
- [A100 stable topology/rate study](2026-08-18-v0.5.15-a100-stable-topology-rate-study.md):
  final high-pressure matched baseline/HexAgenT comparisons.

For setup and reproduction, use `docs/hexgent/README.md`. The cross-worker
experiment is documented only on its own branch and is not part of the stable
v0.5.15 evidence set.
