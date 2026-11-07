# H20 Measurement and Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepare scientifically valid, failure-preserving real-P/D measurements and a matched simulator comparison for the authorized H20 subset.

**Architecture:** Preserve scheduler and router semantics at the starting revision. Extend the existing replay/summary path with observed workflow arrival and explicit metric provenance. Add a small comparison utility for matched workflow predictions and measurements; the comparison requires declared matching configuration and retains unmatched/failed records.

**Tech Stack:** Python 3.10, pytest, SGLang v0.5.15 Python planner and Rust PD router; existing JSONL traces and telemetry.

---

## Scope and decisions

The assigned worktree already exists and starts clean at `af74dbfbb1561147a3853b1ae145ede06de91f13`. Implementation is authorized. Algorithm changes require coordinator review under PROTOCOL.md. This plan changes measurement only. An additional standalone analyzer would duplicate most existing summary logic; extending the existing tested summary is simpler. Remote host/root are required inputs and will be verified before access. Historical H100/A100 latency profiles remain historical data.

## Task 1: Summary correctness (delegated, two files)

Files: `scripts/hexgent_real_trace_collectors/summarize_scheduling_replay.py`, `test/unit/utils/test_hexgent_scheduling_summary.py`.

1. Write tests showing missing p99, failed workflows entering successful distributions, and KV estimates lacking provenance; run to confirm failure.
2. Add nearest-rank p99; retain failed workflow rows and explicit population counts; use null failure ratios; preserve missing isolated SLO as missing.
3. Label KV input-token estimates and expose actual transfer as unmeasured unless measured telemetry is supplied.
4. Run the summary, calibration and trace-tool unit tests. Save red/green evidence.

## Task 2: Persist replay arrival

Files: `python/sglang/hexgent/workflow_trace.py`, `test/unit/utils/test_hexgent_trace_tools.py`.

1. Extend the existing metadata-recording test to require `workflow_arrival_ms` in each raw result; run to see the missing-field failure.
2. Persist the already available arrival value; do not change reveal or request behavior.
3. Require the summary to use the persisted arrival as latency origin when present; label old results as earliest-request origin. Test an artificial gap between arrival and first request.
4. Run trace and summary tests.

## Task 3: Matched simulator-vs-real report

Files: new `scripts/hexgent_real_trace_collectors/compare_simulator_real.py`, new `test/unit/utils/test_hexgent_simulator_real.py`.

1. Write tests for paired absolute/relative errors, sign-preserving errors, missing/failed workflows, tie-aware ordering, and declared configuration mismatch rejection.
2. Implement one function and a JSON CLI over measured/predicted workflow rows, each carrying `workflow_id`, `latency_ms`, `slo_ratio`, and optional `errors`. Require an identical declared comparison contract (input hash, reference profile hash, model/revision, TP, dtype/KV dtype, hardware, topology, cache/tool policy, arrival and scheduling configuration). Keep simulator implementation revision distinct from real implementation revision.
3. Emit per-workflow errors and aggregate latency/ratio MAE and relative error; preserve unmatched and failed IDs. These are descriptive paired errors, with no independence or uncertainty claim inferred from repeats.
4. Run unit tests and CLI on explicitly synthetic fixtures; label these as tooling checks, not real calibration.

## Task 4: Preflight and bounded pilot

1. Obtain exact authorized SSH alias and Fred-owned root through coordinator; run read-only identity, directory ownership, GPU inventory/process/topology, model configuration and runtime checks.
2. Record actual attention head dimension, KV heads, total/active parameters for MoE, TP, weight/KV precision, physical GPUs versus model instances, memory budget and effective scheduler flags.
3. Select a feasible already-available model; pilot 1P+1D proves the path and calibrates stages, while placement needs multiple workers per stage. Freeze trace subset and arrival rates before comparison.
4. Build isolated model-matched SLO profile; use identical cache and tool conventions. Run bounded fixed-token chain and fan-out/join replays, saving complete failures and telemetry.
5. Feed measured H20 stages and transfer into simulator under matching settings, then report held-out errors and limitations. Missing hardware remains missing evidence.

## Validation and delivery

Use an isolated `.venv-h20` for FastAPI tests; do not alter shared simulator virtualenv. Run bounded Python tests and Rust checks if the available platform supports the gateway. Record commands, package versions, file hashes, seed, input/profile hashes and revision. Update shared `H20_STATUS.md` and send coordinator concrete evidence and change proposals. No remote processes or files are created before target/root verification.

## 2026-11-07 remote execution and synchronization update

This update preserves the original measurement requirements and the historical
Windows commands above. Development, validation and experiments now run on H20;
local Windows worktrees are synchronization mirrors. The assigned real-serving
worktree is `/home/fred/hexagent-rebuttal/worktrees/rebuttal-h20`.

Follow the current ownership, CPU/memory allocations, raw-data storage and
completion checks in
[SERVER_WORKFLOW.md](/home/fred/hexagent-rebuttal/coordination/SERVER_WORKFLOW.md).
GitHub synchronization is authoritative. The coordinator handles the current
shared-index staging/commit sequence and authentication, and verifies the H20,
GitHub branch and local mirror HEADs before reporting synchronization complete.
No worker changes shared authentication or another task's worktree.

Accepted correctness changes remain separate from measurement updates and from
historical runs. Use their exact revision ledger in
[the parity audit](/home/fred/hexagent-rebuttal/worktrees/rebuttal-h20/docs/hexgent/rebuttal_h20_parity.md)
when recording later H20 results. Preserve workload identity, reference-profile
hash, observed token counts, TP/precision, cache/tool policy and all failed or
incomplete outcomes required by the original plan.
