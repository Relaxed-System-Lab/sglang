# HexAgenT Event-Driven Global Admission Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `test-driven-development` for each behavior change.

**Goal:** Replace the router's per-request 10 ms planner grace fallback with event-driven global admission, so requests are dispatched only after the latest applicable HexAgenT plan grants prefill admission capacity.

**Architecture:** The router retains the original HTTP request while it is not dispatched. The planner sidecar owns the global pending-admission state and publishes per-request plans after arrival, `prefill_completed`, full-call completion, cancellation, capacity, and plan-publication events. The router waits on a long-poll admission endpoint rather than periodically re-enqueuing or immediately dispatching a fallback. A native PD pair remains a failure-only fallback for planner transport/unavailability or the configured maximum admission wait, not a normal timeout path.

**Tech Stack:** Rust Tokio/reqwest, Python 3.11/FastAPI/asyncio, existing HexAgenT planner and admission reservations.

---

### Task 1: Event-driven admission wait

**Files:**
- Modify: `python/sglang/hexgent/sidecar.py`
- Test: `test/unit/utils/test_hexgent_planner.py`

1. Add a failing test showing that an admission waiter remains pending until the background actor publishes its cached result.
2. Add a coordinator change notification and a `/wait_admission` endpoint.
3. Make arrival, prefill completion, full-call completion, cancellation, and plan publication notify waiters; do not use a fixed polling timer. `retry_after_ms` is compatibility metadata only and must not drive router-side replanning.
4. Verify the focused planner tests pass.

### Task 2: Router dispatch-time admission

**Files:**
- Modify: `sgl-model-gateway/src/routers/http/pd_router.rs`
- Modify: `sgl-model-gateway/src/routers/http/pd_types.rs`
- Test: `sgl-model-gateway/src/routers/http/pd_router.rs`

1. Add a failing Rust test for the `/wait_admission` endpoint construction.
2. Replace `wait_for_hexgent_admission` status polling and planner-grace fallback with one enqueue and one long-poll wait.
3. Retain the request in the router handler until a `dispatch_ready` schedule is received, then inject priority and create bootstrap immediately before dual dispatch.
4. Use native PD only if the planner enqueue/wait transport fails or the configured maximum admission wait expires.
5. Verify focused Rust tests and the Python planner suite.

### Task 3: Validate revision semantics

**Files:**
- Modify: `test/unit/utils/test_hexgent_planner.py`
- Modify: `docs/plans/2026-07-30-hexgent-nonblocking-admission-plan.md`

1. Test that a newly revealed call has no inherited schedule from an earlier call.
2. Test that an already published request retains its reservation while a new batch is solved.
3. Document that incomplete plans are mutable until dispatch; no bootstrap room is created for pending requests.

### Task 4: Stage-boundary capacity release

**Files:**
- Modify: `python/sglang/hexgent/planner.py`
- Modify: `sgl-model-gateway/src/routers/http/pd_router.rs`
- Test: `test/unit/utils/test_hexgent_planner.py`

1. Treat a successful prefill response as a `prefill_completed` planner event.
2. On this event, release only the completed request's prefill reservation. Keep its decode reservation, including its planned decode start/end, until final completion.
3. Re-evaluate global pending admissions immediately. A request can enter a prefill worker as soon as that stage has real credit; it never waits for another request's decode completion merely because both stages were reserved together.
4. Keep worker-local priority injection at dispatch time. Global ranking orders not-yet-dispatched requests; the SGLang local scheduler still owns continuous batching among admitted local requests.
5. If dispatch fails after a plan is published, send `request_cancelled` and remove both reservations. Apply the same rule to a streaming response that terminates without `[DONE]`. This prevents a failed decode from consuming artificial future decode capacity.
