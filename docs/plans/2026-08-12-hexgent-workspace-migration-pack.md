# HexAgenT Workspace Migration Pack

## Repository State

HexAgenT is maintained in the `Relaxed-System-Lab/sglang` fork with two branches:

- `hexgent-v0.5.9-main`: experiment-compatible implementation used for the
  retained H100 and A100 results.
- `hexgent-v0.5.15`: forward port based directly on upstream SGLang v0.5.15
  (`f63458b5be`).

Do not merge the two SGLang runtimes wholesale. Port HexAgenT interfaces and
semantics onto each release branch separately.

The v0.5.15 README uses a provider-neutral Linux-container setup. Its helper
scripts retain `/workspace` as an in-container working-data convention, not as
a RunPod dependency; bind any suitable local or persistent storage at that
path, or update the helper paths consistently before setup.

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

The v0.5.15 port has completed Linux Rust compilation, Python/Rust tests, and
real 6x A100 P-D acceptance with Llama-3.1-8B-Instruct BF16. The retained test
evidence is 343 Python tests passed (2 skipped), 36 Rust tests passed, and 57
telemetry/runner target tests passed (including 4 new regression tests).
ShareGPT, BFCL-v3, and HotpotQA
completed fixed-length DAG replay with zero request errors and non-zero planner
revision coverage.

The retained A100 runs prove functional end-to-end execution but are not valid
performance evidence. BFCL left one prefill worker at 0%/1.3% average
utilization, HotpotQA concentrated work on a subset of decode workers, and
planner revision coverage was only 6.16%-23.74%. Re-run topology/arrival-rate
search before claiming any scheduler win. Formal comparisons must demonstrate
queue pressure, meaningful planner coverage, and non-trivial utilization on
every configured P/D worker. See
`docs/plans/hexgent-migration/2026-08-17-v0.5.15-a100-validation.md`.

The v0.5.15 worker timing format is now parsed separately from v0.5.9. Its
`transfer_speed` field is a prefill transfer-window observation, not a pure
CUDA-event DMA measurement. The safe cross-worker boundary is unchanged:
placement can change while a request remains in router global admission;
after dispatch only same-worker waiting-priority updates are supported.

The reproducible checkpoint for this state is the annotated Git tag
`hexgent-v0.5.15-a100-validated-20260817`. Any experiment that changes bootstrap
ownership, request cancellation, or cross-worker waiting-request reassignment
must branch from this checkpoint and must not rewrite the retained A100 result.

## Authoritative References

- `docs/hexgent/README.md`: setup and reproduction entry point.
- `docs/plans/2026-08-12-hexgent-v0.5.15-port-design.md`: port architecture.
- `docs/plans/hexgent-migration/2026-08-17-v0.5.15-a100-validation.md`: real
  v0.5.15 A100 validation and retained result asset.
- `docs/plans/hexgent-migration/2026-08-18-v0.5.15-a100-stable-topology-rate-study.md`:
  current matched high-pressure A100 performance evidence.
- `real_trace_profiles/README.md`: curated profile/result asset index.
- `docs/plans/hexgent-migration/`: session-level evidence and decisions.

## 2026-08-18 A100 Stable Topology/Rate Study

The experimental cross-worker branch was preserved, but it produced no valid
safe-pair, reassignment action, or completed migration on the real A100 run.
Performance work therefore returned to the validated `hexgent-v0.5.15` global
admission implementation. Do not mix cross-worker results into stable-branch
performance claims.

All final stable runs used projected-ratio `high_first` ordering and disabled
pairwise-swap local search. This is now the default experimental policy; do not
continue ordering-direction searches unless a new algorithmic hypothesis is
explicitly approved.

At matched high pressure, the retained results are:

- ShareGPT, 1P+5D at 48 workflow/s: planner coverage 26.6%. P95 ratio was
  effectively neutral/slightly worse (1.659x baseline vs 1.670x HexAgenT), but
  the 100-workflow P99 convention is the sample maximum and improved from
  4.574x to 3.439x (-24.8%).
- BFCL-v3, 2P+2D at 0.5 workflow/s: coverage 20.7%; P95 improved from 9.388x
  to 7.387x and empirical P99 from 9.536x to 7.624x.
- HotpotQA, 1P+2D at 0.3 workflow/s with 20 real DAGs replicated three times:
  coverage 53.6%; P95 improved slightly from 5.096x to 5.057x, while P99
  worsened from 5.175x to 5.682x. Fan-out extreme-tail scheduling remains open.

Absolute SLO ratios may increase under intentionally higher pressure. A run is
judged by matched baseline-versus-HexAgenT ratios at the same trace, topology,
arrival process, exclusive SLO, and runtime configuration, not by requiring the
absolute ratio to remain small.

The curated final-evidence archive is
`/workspace/hexgent_v0515_stable_a100_final_evidence_20260818.tar.gz`; its local
copy is
`real_trace_profiles/hexgent_v0515_stable_a100_final_evidence_20260818.tar.gz`
(9,953,517 bytes, SHA-256
`acd56a9bce8203f700b7a0f1accd09664e56d43d50f5756aafee5a65744674b3`).
It contains only the three final matched A/B runs, their telemetry/log evidence,
the exclusive profile, and the replicated Hotpot trace. It is gitignored and
must not be committed to normal Git history. The 87 MB full remote archive also
contains topology-search and intermediate runs and is not required locally.
See
`docs/plans/hexgent-migration/2026-08-18-v0.5.15-a100-stable-topology-rate-study.md`
for the full evidence and result paths.
