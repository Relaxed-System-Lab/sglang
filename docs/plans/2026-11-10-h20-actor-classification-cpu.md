# H20 actor evidence classification implementation plan

**Goal:** Classify event-only coordinator telemetry without calling it actor admission, while rejecting real admission and missing evidence.

**Architecture:** Read the existing completed HTTP capture in the private functional evidence caller. Combine recorded paths with admission/event/result counts in the private gate. Preserve the public sidecar/planner, process identity gate, observer profile, source63 and original FAILED artifacts.

**Tech stack:** Python3.12 stdlib and CPU-only pytest in the existing sim environment.

1. Establish the existing functional gate/window baseline (48 tests).
2. Add CPU regression tests in `test/unit/utils/test_hexgent_h20_actor_classification.py`. Generate the event-only row using the actual AdmissionCoordinator/PlannerTelemetry class bodies, and exercise the real private `_functional_evidence` file reader. Confirm RED for the original false actor flag, absent HTTP evidence, real enqueue HTTP without telemetry, and missing counter evidence.
3. Add `capture_records` to `h20_functional_smoke.functional_smoke_gate`; reject missing capture, positive admission/result counts and real enqueue activity. Accept internal maintenance only with explicit zero admission/result, positive event count and recorded POST/event evidence. Endpoint existence or an empty revision query alone is not admission dispatch.
4. Pass the existing `identity-capture.jsonl` records from `h20_transport_window._functional_evidence`. Update the synthetic complete-evidence fixture to include an explicit empty capture. No capture collection changes.
5. Run the new tests and the two focused existing files. Review the exact diff; verify observer/public sources and original outcome hashes unchanged. Commit a clean CPU-only pin, retain RED/GREEN logs and a concise receipt. No GPU, services or paid API calls.
