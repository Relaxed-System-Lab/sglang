# C2 proposal: one accounting convention for decode snapshot capacity

Status: `R1_REAL_CAPACITY` candidate implemented locally and awaiting coordinator
review. The initial red-test audit used H20 revision `33e34d5ccc` (accepted
sibling-horizon H1). C1 was subsequently accepted in the simulator as
`9be157f9c1c8e3d2271e1e52808d3b1bd97778c0`; its exact scheduling-core hunks and
the real-wrapper interface are now a separate local candidate. Both candidates
were tested together. Neither real candidate has been committed or accepted.

## Observed failure and root cause

For a supplied live decode snapshot with total KV pool 1000 tokens, running KV
occupancy 400 tokens and estimated running interval `[0, 101)`,
`apply_worker_snapshots()` produces both:

- `DecodeStageState.capacity_max_tokens = [600]`;
- `scheduled_intervals = [[(0, 101, 400)]]`.

`_estimate_decode_state()` in `python/sglang/hexgent/snapshots.py:507` computes
`available_capacity = total_capacity - used_memory_tokens` and returns an
interval that also carries `used_memory_tokens`. `apply_worker_snapshots()`
stores the reduced capacity in `worker.capacity_tokens`; the planner passes
that scalar and the occupied interval to the decode capacity solver.

The pre-R1 capacity check charges the same 400 tokens twice. A new request
with demand 300 and duration 1 ms receives start time 101 ms, although its
combined occupancy at time zero is `400 + 300 = 700 <= 1000` tokens. The running
duration in this reproducer is deterministic: the existing default decode
labels estimate `1 + 10 * 10 = 101` ms for ten remaining service tokens.

Within this plan, the scalar remains 600 after the interval ends. A later empty
live snapshot restores 1000. The defect therefore persists for the frozen plan
until a new snapshot/replan, rather than permanently across subsequent snapshots.

## Contract choice

Two coherent representations are possible:

| Representation | Constant scalar | Occupancy handling | Consequence |
| --- | --- | --- | --- |
| Total pool plus occupied intervals, recommended | Observed total KV limit, 1000 | Charge 400 during `[0, 101)` exactly once | Existing interval solver releases that occupancy at 101; intrinsic feasibility uses the physical total. |
| Free pool without duplicate occupancy | Current free tokens, 600 | Omit the already-subtracted 400 from concurrent reservations | Requires a time-dependent restoration of the subtracted capacity, or restricting the free-capacity estimate to the present instant and obtaining a fresh snapshot before later placement. |

Recommend the first representation because `DecodeStageState.capacity_max_tokens`
is the constant maximum used by interval placement. It preserves the existing
occupied intervals and their release times. The local candidate returns the
observed total pool from `_estimate_decode_state()` while retaining the existing
interval, demand, timing and placement-lock semantics. It also updates three
existing test assertions from free capacity to the corresponding total pool.

The scope is an explicit, same-cutoff live snapshot with a reported total limit.
The proposal makes no estimate for absent snapshots and introduces no fallback
capacity. Existing tests that assert reduced `worker.capacity_tokens` must be
reviewed as contract assertions if C2 is accepted; they currently encode the
free-pool interpretation.

## Independent red tests

Command from `D:/HexAGenT-wt/rebuttal-h20`:

```powershell
& '.venv-h20/Scripts/python.exe' -m pytest -q test/unit/utils/test_hexgent_snapshot_capacity_contract.py
```

Actual result on H1 revision `33e34d5ccc`: **4 failed, 1 passed in 0.65 s**, exit
code 1. The file is deliberately separate from the accepted regression suite
until review.

| Test | Expected under proposed contract | Current observation |
| --- | --- | --- |
| Empty live snapshot | Total 1000; no intervals; demand 700 starts at 0 | Pass. |
| Busy live snapshot | Total 1000 plus interval `(0, 101, 400)` | Scalar is 600; interval is correct. |
| New demand 300 | Starts at 0 because 700 total tokens fit | Starts at 101. |
| Release at interval end | Demand 700 is intrinsically feasible and starts at 101 | State scalar is 600, so the intrinsic-fit assertion fails. |
| Later empty snapshot | Same total 1000, occupied intervals removed | Later state is 1000, but preceding busy-state scalar is 600. |

The release test fails on the represented capacity before calling the current
helper. This distinction matters because the pre-C1 helper can return a time
even when request demand exceeds its supplied capacity, concealing the bad
snapshot contract.

## Interaction with the separate C1 proposal

C1 rejects demand greater than a worker's total capacity. Applied
to the current snapshot representation, demand 700 would be compared with 600
and marked intrinsically infeasible, including after time 101 when the observed
400-token interval has ended. Against the observed total limit of 1000 it is
feasible after that interval. At the initial red-test audit this was a deduction,
because C1 was not yet available. The subsequent exact C1 core port plus R1 now
passes the combined regression cases below.

Review C1 and C2 independently, then verify their composition using this same
fixture. Expected combined results are: demand 300 starts at 0; demand 700 starts
at 101; demand 1001 is intrinsically infeasible; an empty snapshot preserves
capacity 1000. H1's horizon fix remains a separate accepted revision.

## Local candidate validation

The real C1 candidate supplies total decode capacities at both prefill planning
entry points, explicitly passes each call's visible input-plus-output reservation
demand, and retains existing assignment locks. Its output demand follows the
existing predictor interface (provided prediction, then request max tokens),
without reading future output. All-infeasible global admissions retain their
pending status; the direct API reports an explicit infeasibility error.

The initial real C1 tests recorded **19 failed, 5 passed** before implementation.
After the two candidates were combined, explicit tests confirm demand 300 starts
at 0, demand 700 waits until 101, and demand 1001 has no finite start under the
same observed 1000-token pool with 400 occupied until 101. Empty snapshots retain
total capacity. Additional cases cover sole-feasible placement, all-infeasible
placement, visible prediction/max-token demand, and fixed existing P/D pairs.

```powershell
& '.venv-h20/Scripts/python.exe' -m pytest -q `
  test/unit/utils/test_hexgent_planner.py `
  test/unit/utils/test_hexgent_scheduler.py `
  test/unit/utils/test_hexgent_real_capacity.py `
  test/unit/utils/test_hexgent_snapshot_capacity_contract.py
```

Actual combined result: **121 passed, 13 existing FastAPI/Starlette deprecation
warnings in 3.37 s**, exit 0. The three C1-plus-snapshot composition cases in
`test_hexgent_real_capacity.py` require both candidates. R1 and the real C1 port
are exported as separate patches for review; their combined validation does not
make either real candidate an accepted comparison version.

## Scientific impact and acceptance checks

C2 can change decode admission order, projected completion, P/D decisions that
use those projections, workflow latency and Req95/Req99. It can also change
planner rejection/fallback coverage after C1. Comparative runs must record a
separate accepted method revision and preserve earlier results.

Before acceptance, review live-snapshot units and aggregate occupancy semantics,
update the existing free-capacity assertions deliberately, run this independent
contract file plus planner/scheduler/snapshot regressions, and verify the C1+C2
composition. H20 measurements remain necessary for real performance or
simulator-to-real claims; these tests establish a CPU accounting inconsistency.
