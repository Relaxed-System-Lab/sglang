# Worker mutation observation (W2)

Accepted scoped instrumentation on 2026-11-07. This adds one INFO JSON event after either existing mutation endpoint obtains its tokenizer results, then returns the original HTTP payload. Queue mutation, revision guards, retries and application order are unchanged.

The boundary is worker-endpoint received requests that returned result objects. Requests lost upstream, transport failures and handler errors before results are unobserved; their counts/rates remain null. A worker result is a diagnostic subresult, not an independent workflow or physical GPU sample. Actual result cardinality must be recorded at deployment.

The parser separates received HTTP attempts, unique actions and result outcomes. The unique key is worker_port/request_id/action_type/queue/schedule_revision/expected_revision within the supplied run logs. Never merge reused IDs or ports across runs. Stale_revision, expected_revision_mismatch, not_found and missing legacy reason codes remain distinct. Any-stale and at-least-once-applied action categories can overlap; any-rank success is not all-rank success. Zero denominators are null.

INFO JSON serialization and logging consume real time. Keep the worker logging configuration identical for all methods. End-to-end serving results include this overhead; mutation event counts and saved log bytes quantify its volume. Per-event logging latency is not separately measured by this observer. Runtime logs are capped and drained by the pilot supervisor; truncated logs invalidate a complete-count claim.

CPU tests execute extracted actual endpoint/helper code and preserve payloads, mixed results, repeated actions, legacy unknowns and zero-attempt behavior. The author observed 7 new tests red-to-green and 32 tests with the existing bridge suite passing; main independently repeated 32 tests on H20. Separate actual msgspec serialization probes cover W1 response fields. Full live endpoint and TP/IPC behavior still require the pilot; these CPU checks are not network or GPU performance evidence.

Use summarize_worker_mutations.py with explicit per-run worker log slices. Preserve full logs separately, offsets/time window, implementation revision, source hashes and supervisor capture/truncation metadata. Exclude calibration/control probes by taking run windows rather than silently filtering rejected experiment actions.
