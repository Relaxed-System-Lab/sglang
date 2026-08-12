# HexAgenT Migration Session Index

This directory stores incremental implementation and experiment evidence. The
authoritative setup entry point is `docs/hexgent/README.md`; session notes must
not override it.

## Update Rules

Each HexAgenT work session must record the code branch, relevant configuration,
direct measurements, code-backed conclusions, remaining assumptions, and next
verification step. Credentials and transient SSH addresses must not be stored.

## Current Session

- [2026-08-12 v0.5.15 port](2026-08-12-v0.5.15-port.md): ports the planner,
  Rust PD router integration, worker controls, replay tooling, and curated
  assets from the validated v0.5.9 line to upstream SGLang v0.5.15.

## Historical Sessions

- [Repository curation and three-trace summary](2026-08-12-repository-curation-and-three-trace-summary.md)
- [A100 HotpotQA topology and scheduling](2026-08-11-a100-hotpotqa-topology-and-scheduling.md)
- [Live snapshot alignment and planner coverage](2026-08-05-live-snapshot-alignment-and-planner-coverage.md)
- [Token-aware admission recovery](2026-08-05-token-aware-admission-recovery-and-full-profile-replay.md)
- [A100 algorithm audit and bring-up](2026-08-05-a100-algorithm-audit-and-bringup.md)
- [Mooncake segmentation interpretation](2026-08-05-mooncake-segmentation-bandwidth-interpretation.md)
- [RunPod asset curation](2026-08-05-runpod-asset-curation.md)
- [4P+4D scheduling audit](2026-08-04-4p4d-scheduling-audit.md)
- [Admission credit and greedy semantics](2026-08-04-admission-credit-and-greedy-semantics.md)
- [Token-aware prefill admission](2026-08-04-token-aware-prefill-admission.md)
- [RunPod volume retirement](2026-08-04-runpod-volume-retirement.md)

Historical notes describe the v0.5.9 implementation and experiments unless a
note explicitly states otherwise.
