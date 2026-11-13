# Live Tool and Revision Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve real replay tool waits and discard obsolete asynchronous worker actions without losing pending admissions.

**Architecture:** Replay-only tool duration delays dependency completion after HTTP response metrics are recorded. Worker CAS outcomes distinguish obsolete actions from transient failures; fresh worker snapshots return through the existing sidecar event/actor path. Coordinator publication checks admission generations so cancellation or refresh during a solve cannot publish an obsolete result.

**Tech Stack:** Python asyncio, FastAPI sidecar, Rust HTTP gateway; CPU-only pytest and rustc tests.

---

### Task 1: Replay duration and completion boundaries
- Create focused tests in test/unit/utils/test_hexgent_tool_replay.py for duration roundtrip, concurrent workflows, parent-tool barriers, terminal tools, and actual workflow-C consumers.
- Run focused tests after CPU3/isolated-venv grant and confirm missing-duration/boundary failures.
- Update workflow_trace.py and minimal workflow summary consumers; retain HTTP response/service boundaries and zero-tool input behavior.
- Re-run focused and existing replay/summary tests.

### Task 2: Obsolete mutation delivery
- Add std-only gateway mutation-outcome tests and observe RED before implementation.
- Classify worker stale revision/CAS mismatch as obsolete, retain network/HTTP retries, and pass fresh snapshots to the existing sidecar event endpoint.
- Preserve the original queued request; never modify expected_revision to retry an old action.
- Run helper tests and attempt the existing gateway build only within the authorized CPU/resource budget.

### Task 3: Coordinator freshness and generation
- Add regressions for cancellation/refresh during a solve and pending admission replan from the real event/wake path without a new arrival.
- Observe RED, then add generation-filtered publication and snapshot-driven requeue of only gateway-confirmed pending admissions.
- Re-run relevant coordinator tests.

### Task 4: Original request body for cache tokenization
- Add optional request_body JSON to the gateway planner request and clone the actual original request at the sole runtime constructor.
- Preserve request_text as the existing length fallback and omit request_body for old None fixtures.
- Add a native serialization regression retaining messages, tool calls, tools, and chat template arguments.

### Validation
- Replay and actual workflow-latency consumers: 114 CPU tests passed.
- Coordinator generation and fresh-state event/wake paths: 7 CPU tests passed.
- Rust parser accepted pd_router.rs, pd_types.rs, and the three new helper/test files.
- Rust outcome helper tests typechecked with rustc --test --emit=metadata on Windows. Executable tests remain blocked by the absent MSVC linker/Windows SDK libraries; Linux cargo validation is required before cloud inference readiness.
- Native request_body regression execution is pending the same compatible Rust build environment.

No commits, GPU execution, inference, shared-environment installations, or services are authorized for this work.
