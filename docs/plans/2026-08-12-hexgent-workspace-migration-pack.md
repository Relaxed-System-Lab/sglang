# HexAgenT Workspace Migration Pack

## Repository State

HexAgenT is maintained in the `Relaxed-System-Lab/sglang` fork with two branches:

- `hexgent-v0.5.9-main`: experiment-compatible implementation used for the
  retained H100 and A100 results.
- `hexgent-v0.5.15`: forward port based directly on upstream SGLang v0.5.15
  (`f63458b5be`).

Do not merge the two SGLang runtimes wholesale. Port HexAgenT interfaces and
semantics onto each release branch separately.

## System Model

The Python sidecar owns workflow state, service estimation, projected-ratio
ordering, P/D placement, reservations, and plan revisions. The Rust router owns
the global admission queue, live worker snapshots, native fallback, placement
execution, dispatch-time bootstrap creation, and completion reporting. Workers
own continuous batching and expose snapshot/timing/local-priority controls.

The planner never assumes knowledge of unrevealed future calls. A newly revealed
call updates the known DAG and workflow horizon. Fixed profiled output lengths
are currently used for controlled replay; a predictor remains a fallback.

## Scheduling and Migration Boundary

Every undispatched request has an executable native fallback and may receive a
new HexAgenT placement while it remains in the router global queue. Planner work
must not block an idle worker. Token-aware admission keeps enough work near each
prefill worker to sustain batching while preserving global replanning freedom.

At dispatch, the router creates the bootstrap relationship and commits the P/D
worker pair. A later revision may change local waiting priority when safe, but
cross-worker live migration is not supported after dispatch or after prefill/KV
transfer/decode begins.

## Experimental Rules Learned

- Replay identical DAGs and fixed per-call token lengths for both schedulers.
- Build SLOs from isolated profiles with identical model, dtype, GPU, TP, cache,
  and serving release.
- Serial workflow SLO is the sum of exclusive call latency; fan-out workflow SLO
  is the exclusive critical-path latency.
- Disable radix cache for controlled comparisons and use unique run namespaces.
- Require zero errors and non-zero planner revision coverage before interpreting
  a run.
- Treat Mooncake CUDA-event copy time, synchronous API wall-clock, and end-to-end
  producer-ready latency as different measurement boundaries.

## Current Validation Status

The v0.5.15 Python/control-plane port has focused test coverage and formatting
validation. Full Rust compilation and GPU P-D acceptance remain pending on a
fresh Linux Pod because the local Windows Cargo dependency cache is incomplete.
See `docs/plans/hexgent-migration/2026-08-12-v0.5.15-port.md` for exact evidence
and the Pod checklist.

## Authoritative References

- `docs/hexgent/README.md`: setup and reproduction entry point.
- `docs/plans/2026-08-12-hexgent-v0.5.15-port-design.md`: port architecture.
- `docs/hexgent/2026-08-12-a100-three-trace-results.md`: retained A100 summary.
- `real_trace_profiles/README.md`: curated profile/result asset index.
- `docs/plans/hexgent-migration/`: session-level evidence and decisions.
