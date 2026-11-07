# R1_REAL_CAPACITY: total decode capacity in live snapshots

Status: **accepted R1_REAL_CAPACITY**, real commit
`92e563764b11fad6df360352fdb232e3d547b7a8`. The matching real C1 feasibility
interface is accepted separately in `77400e9d6d40bb982f1decfc87856dd279598b70`;
the simulator C1 reference is `9be157f9c1c8e3d2271e1e52808d3b1bd97778c0`.

The historical audit below records the pre-R1 snapshot contract at H1 revision
`33e34d5ccc`. Original red/green counts, arithmetic fixtures and Windows commands
remain provenance for the correction; they are CPU checks rather than H20
serving measurements.

## Historical failure and root cause

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
| Total pool plus occupied intervals, accepted | Observed total KV limit, 1000 | Charge 400 during `[0, 101)` exactly once | Existing interval solver releases that occupancy at 101; intrinsic feasibility uses the physical total. |
| Free pool without duplicate occupancy | Current free tokens, 600 | Omit the already-subtracted 400 from concurrent reservations | Requires a time-dependent restoration of the subtracted capacity, or restricting the free-capacity estimate to the present instant and obtaining a fresh snapshot before later placement. |

R1 uses the first representation because `DecodeStageState.capacity_max_tokens`
is the constant maximum used by interval placement. It preserves the existing
occupied intervals and their release times. The accepted implementation returns the
observed total pool from `_estimate_decode_state()` while retaining the existing
interval, demand, timing and placement-lock semantics. It also updates three
existing test assertions from free capacity to the corresponding total pool.

The scope is an explicit, same-cutoff live snapshot with a reported total limit.
R1 preserves the existing behavior for absent snapshots. Three existing
free-capacity assertions were updated to the total-pool contract; the occupied
intervals continue to carry the corresponding reservations.

## Historical red-test evidence

Command from `D:/HexAGenT-wt/rebuttal-h20`:

```powershell
& '.venv-h20/Scripts/python.exe' -m pytest -q test/unit/utils/test_hexgent_snapshot_capacity_contract.py
```

Actual result on H1 revision `33e34d5ccc`: **4 failed, 1 passed in 0.65 s**, exit
code 1. These failures preceded the accepted R1 implementation. The same
contract file is now part of the accepted regression suite.

| Test | Expected under total-pool contract | Pre-R1 observation |
| --- | --- | --- |
| Empty live snapshot | Total 1000; no intervals; demand 700 starts at 0 | Pass. |
| Busy live snapshot | Total 1000 plus interval `(0, 101, 400)` | Scalar is 600; interval is correct. |
| New demand 300 | Starts at 0 because 700 total tokens fit | Starts at 101. |
| Release at interval end | Demand 700 is intrinsically feasible and starts at 101 | State scalar is 600, so the intrinsic-fit assertion fails. |
| Later empty snapshot | Same total 1000, occupied intervals removed | Later state is 1000, but preceding busy-state scalar is 600. |

The release test fails on the represented capacity before calling the pre-C1
helper. This distinction matters because the pre-C1 helper can return a time
even when request demand exceeds its supplied capacity, concealing the bad
snapshot contract.

## Interaction with accepted C1

C1 rejects demand greater than a worker's total capacity. Applied
to the pre-R1 snapshot representation, demand 700 would be compared with 600
and marked intrinsically infeasible, including after time 101 when the observed
400-token interval has ended. Against the observed total limit of 1000 it is
feasible after that interval. At the initial red-test audit this was a deduction,
because C1 was not yet available. The subsequent exact C1 core port plus R1 now
passes the combined regression cases below.

R1 and C1 were accepted as separate commits and checked together using this
fixture. Combined regression expectations are: demand 300 starts at 0; demand 700 starts
at 101; demand 1001 is intrinsically infeasible; an empty snapshot preserves
capacity 1000. H1's horizon fix remains a separate accepted revision.

## Historical Windows CPU validation

The accepted real C1 interface supplies total decode capacities at both prefill planning
entry points, explicitly passes each call's visible input-plus-output reservation
demand, and retains existing assignment locks. Its output demand follows the
existing predictor interface (provided prediction, then request max tokens),
without reading future output. All-infeasible global admissions retain their
pending status; the direct API reports an explicit infeasibility error.

The initial real C1 tests recorded **19 failed, 5 passed** before implementation.
After the two corrections were combined, explicit tests confirm demand 300 starts
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
`test_hexgent_real_capacity.py` require both R1 and C1. Their historical
candidate patches remain separate provenance artifacts; the accepted commits
above identify the production implementation versions.

## Scientific impact and measurement scope

R1 can change decode admission order, projected completion, P/D decisions that
use those projections, workflow latency and Req95/Req99. It can also change
planner rejection/fallback coverage after C1. Comparative runs must record a
separate accepted method revision and preserve earlier results.

The accepted correction establishes the total-pool snapshot contract and its
composition with C1. H20 performance and simulator-to-real error claims require
separate serving measurements under those recorded implementation revisions.
