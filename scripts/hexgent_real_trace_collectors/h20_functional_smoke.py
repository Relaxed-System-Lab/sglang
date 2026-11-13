"""Pure evidence QC for the fixed ten-call H20 compatible solver smoke.

This checks joined recorded facts. It does not run requests, validate the actor
admission owner, measure a performance effect, or independently measure KV bytes.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

DEQUEUE_PREFIX = "HexgentDequeueEvidence "


def read_dequeue_events(paths):
    """Read only the fixed worker event; preserve missing/malformed capture."""
    events = []
    for path in paths:
        try:
            with Path(path).open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if DEQUEUE_PREFIX not in line:
                        continue
                    try:
                        event = json.loads(line.split(DEQUEUE_PREFIX, 1)[1])
                    except json.JSONDecodeError:
                        event = {"event": "capture_incomplete", "reason": "malformed_dequeue_record"}
                    event["_source_log"] = str(path)
                    event["_source_line"] = line_number
                    events.append(event)
        except (OSError, UnicodeError) as error:
            events.append({"event": "capture_incomplete", "reason": type(error).__name__,
                           "_source_log": str(path)})
    return events


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def functional_smoke_gate(workflows, raw_rows, planner_rows, worker_events, *,
                          model, run_namespace, worker_instances, observer_summary,
                          capture_complete, worker_capture_complete, capture_records=None):
    """Join exact responses, direct solver results and actual P/D removals.

    None records startup/replay ending before a functional summary was saved.
    """
    failures, incomplete = [], []

    def fail(reason):
        if reason not in failures:
            failures.append(reason)

    def unknown(reason):
        if reason not in incomplete:
            incomplete.append(reason)

    expected = {(workflow.workflow_id, call.call_id): call
                for workflow in workflows for call in workflow.calls}
    runtime_ids = {(run_namespace + "::" + wf, run_namespace + "::" + call): (wf, call)
                   for wf, call in expected}
    if (len(workflows) != 2 or len(expected) != 10
            or sum(call.input_len for call in expected.values()) != 5176
            or sum(call.output_len for call in expected.values()) != 1952):
        fail("fixed_two_workflow_ten_call_token_contract_differs")
    workers = {row["worker_url"]: row for row in worker_instances}
    replica_counts = {row["instance_id"]: 0 for row in worker_instances}
    raw_index, solver_index, selected_index = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in raw_rows:
        identity = (row.get("workflow_id"), row.get("call_id"))
        raw_index[identity].append(row)
        if identity not in expected:
            fail("unexpected_response_identity")
    schedules = [row for row in planner_rows if row.get("endpoint") == "/schedule_request"]
    for row in schedules:
        identity = (row.get("workflow_id"), row.get("call_id"))
        solver_index[identity].append(row)
        if identity not in runtime_ids:
            fail("unexpected_direct_solver_identity")
    if capture_records is None:
        unknown("sidecar_http_evidence_missing")
    captures = capture_records if capture_records is not None else ()
    if any(row.get("method") == "POST" and row.get("path") == "/enqueue_admission"
           for row in captures):
        fail("actor_admission_endpoint_observed_in_compatible_scope")
    event_http_observed = any(row.get("method") == "POST" and row.get("path") == "/event"
                              for row in captures)
    for row in planner_rows:
        if row.get("endpoint") == "/enqueue_admission":
            fail("actor_admission_endpoint_observed_in_compatible_scope")
        elif row.get("endpoint") == "/plan_pending_admissions":
            counts = [row.get(key) for key in ("admission_count", "event_count", "result_count")]
            if any(type(count) is not int or count < 0 for count in counts):
                unknown("actor_activity_evidence_incomplete")
                continue
            admissions, events, results = counts
            if admissions or results:
                fail("actor_admission_endpoint_observed_in_compatible_scope")
            elif not events or not event_http_observed:
                unknown("actor_activity_evidence_incomplete")
            # Event-only maintenance invokes this internal method too; its
            # telemetry label alone is not an external admission or dispatch.

    contention = {"multi_candidate_dequeue_events": 0, "events_with_unselected_candidates": 0,
                  "distinct_priority_competition_events": 0,
                  "max_waiting_by_role": {"prefill": 0, "decode": 0}}
    dequeue_count = 0
    for event in worker_events:
        if (event.get("event") != "dequeue" or event.get("record_cap_reached")
                or event.get("record_byte_cap_reached") or event.get("identities_truncated")):
            unknown("worker_evidence_missing_malformed_or_capped")
            continue
        dequeue_count += 1
        role, url = event.get("role"), event.get("worker_url")
        if url not in workers or workers[url]["role"] != role or event.get("tp_rank") != 0:
            fail("dequeue_worker_or_role_not_owned_rank_zero")
            continue
        waiting, selected = event.get("waiting", []), event.get("selected", [])
        if event.get("waiting_count") != len(waiting) or event.get("selected_count") != len(selected):
            unknown("worker_candidate_identity_counts_incomplete")
        count = event.get("waiting_count", 0)
        contention["max_waiting_by_role"][role] = max(contention["max_waiting_by_role"][role], count)
        contention["multi_candidate_dequeue_events"] += count > 1
        contention["events_with_unselected_candidates"] += count > event.get("selected_count", 0)
        priorities = {row.get("priority") for row in waiting}
        contention["distinct_priority_competition_events"] += count > 1 and len(priorities) > 1
        for req in selected:
            identity = (req.get("workflow_id"), req.get("call_id"))
            if identity not in runtime_ids:
                fail("unexpected_actual_worker_selected_identity")
                continue
            # Candidates are not execution proof: this is the actual selected list,
            # joined back to the same pre-removal queue identity and Req revision.
            signature = tuple(req.get(key) for key in ("rid", "workflow_id", "call_id", "schedule_revision"))
            if not req.get("rid") or not any(
                    tuple(candidate.get(key) for key in ("rid", "workflow_id", "call_id", "schedule_revision"))
                    == signature for candidate in waiting):
                fail("actual_selected_request_missing_from_waiting_candidates")
            selected_index[(identity, role)].append((event, req))

    if observer_summary is None:
        unknown("transport_observer_summary_missing")
    elif not (observer_summary.get("observation_complete") is True
            and observer_summary.get("first_error_unix_s") is None
            and observer_summary.get("stop_requested") is False
            and observer_summary.get("incomplete_reasons") == []):
        unknown("transport_observer_incomplete_or_unsafe")
    if capture_complete is not True:
        unknown("sidecar_final_capture_incomplete")
    if worker_capture_complete is not True:
        unknown("all_four_worker_final_drains_incomplete")

    completed, usage_verified, solver_joined, worker_joined = set(), set(), set(), set()
    good_calls = set()
    prompt_tokens = completion_tokens = 0
    pairs = Counter()
    for identity, call in expected.items():
        label = identity[0] + "/" + identity[1]
        runtime_identity = (run_namespace + "::" + identity[0], run_namespace + "::" + identity[1])
        before = len(failures), len(incomplete)
        responses = raw_index[identity]
        if len(responses) != 1:
            (unknown if not responses else fail)("response_missing_or_duplicated:" + label)
            continue
        row = responses[0]
        if row.get("error"):
            fail("request_error:" + label)
        response = row.get("response") or {}
        usage = response.get("usage") or {}
        if response.get("model") != model:
            fail("returned_model_differs:" + label)
        if usage.get("prompt_tokens") != call.input_len or usage.get("completion_tokens") != call.output_len:
            fail("observed_token_usage_differs:" + label)
        else:
            usage_verified.add(identity)
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
        choices = response.get("choices") or []
        if not choices or any(choice.get("finish_reason") is None for choice in choices):
            fail("generation_completion_missing:" + label)
        elif not row.get("error"):
            completed.add(identity)
        if row.get("router_mode") != "hexgent":
            fail("returned_router_mode_not_hexgent:" + label)
        chosen = {}
        for role, field in (("prefill", "prefill_worker"), ("decode", "decode_worker")):
            url = row.get(field)
            if url not in workers or workers[url]["role"] != role:
                fail("returned_assignment_not_owned:" + label)
            else:
                chosen[role] = url
                replica_counts[workers[url]["instance_id"]] += 1
        if len(chosen) == 2:
            pairs[workers[chosen["prefill"]]["instance_id"] + "/" + workers[chosen["decode"]]["instance_id"]] += 1
        solver_rows = solver_index[runtime_identity]
        if len(solver_rows) != 1:
            (unknown if not solver_rows else fail)("direct_solver_missing_or_duplicated:" + label)
            continue
        solver = solver_rows[0]
        required = ("revision", "prefill_worker_url", "decode_worker_url", "prefill_priority",
                    "decode_priority", "planned_decode_instance", "fallback_reason")
        if any(key not in solver or (solver[key] is None and key != "fallback_reason") for key in required):
            unknown("solver_result_fields_missing:" + label)
            continue
        solver_before = len(failures), len(incomplete)
        revision = solver["revision"]
        if not isinstance(revision, int) or isinstance(revision, bool) or revision <= 0:
            fail("solver_revision_not_positive:" + label)
        if row.get("schedule_revision") != revision:
            fail("returned_revision_differs_from_solver:" + label)
        if solver.get("error") or solver["fallback_reason"] is not None:
            fail("solver_error_or_fallback:" + label)
        timings = solver.get("phase_timings_ms") or {}
        if any(not _finite(timings.get(key)) or timings[key] < 0
               for key in ("prefill_solver", "decode_solver")):
            unknown("real_solver_phase_evidence_missing:" + label)
        for role in ("prefill", "decode"):
            if solver[role + "_worker_url"] != chosen.get(role):
                fail("returned_assignment_differs_from_solver:" + label)
        if solver_before == (len(failures), len(incomplete)):
            solver_joined.add(identity)
        worker_before = len(failures), len(incomplete)
        for role in ("prefill", "decode"):
            selections = selected_index[(runtime_identity, role)]
            if len(selections) != 1:
                (unknown if not selections else fail)("actual_" + role + "_dequeue_missing_or_duplicated:" + label)
                continue
            event, req = selections[0]
            if event["worker_url"] != solver[role + "_worker_url"]:
                fail("actual_" + role + "_worker_differs_from_solver:" + label)
            if req.get("schedule_revision") != revision:
                fail("actual_" + role + "_revision_differs_from_solver:" + label)
            if (req.get("priority") != solver[role + "_priority"]
                    or req.get("prefill_priority") != solver["prefill_priority"]
                    or req.get("decode_priority") != solver["decode_priority"]):
                fail("actual_" + role + "_priority_differs_from_solver:" + label)
            if req.get("planned_decode_instance") != solver["planned_decode_instance"]:
                fail("actual_" + role + "_decode_target_differs_from_solver:" + label)
        if worker_before == (len(failures), len(incomplete)):
            worker_joined.add(identity)
        if before == (len(failures), len(incomplete)):
            good_calls.add(identity)

    workflow_completed = 0
    for workflow in workflows:
        identities = [(workflow.workflow_id, call.call_id) for call in workflow.calls]
        rows = [raw_index[key][0] for key in identities if len(raw_index[key]) == 1]
        arrivals = [row.get("workflow_arrival_ms") for row in rows]
        if len(rows) != len(identities) or not all(_finite(value) for value in arrivals):
            unknown("whole_workflow_arrival_evidence_missing:" + workflow.workflow_id)
            continue
        if len(set(arrivals)) != 1:
            fail("workflow_arrival_inconsistent:" + workflow.workflow_id)
        valid = True
        for call in workflow.calls:
            identity = (workflow.workflow_id, call.call_id)
            row = raw_index[identity][0]
            start, end = row.get("request_started_at_ms"), row.get("response_completed_at_ms")
            if not _finite(start) or not _finite(end):
                unknown("request_completion_timestamps_missing:" + call.call_id)
                valid = False
                continue
            if not arrivals[0] <= start < end:
                fail("request_completion_timestamps_inconsistent:" + call.call_id)
                valid = False
            for parent in call.parents:
                parent_rows = raw_index[(workflow.workflow_id, parent)]
                parent_end = (parent_rows[0].get("call_completed_at_ms", parent_rows[0].get("response_completed_at_ms"))
                              if len(parent_rows) == 1 else None)
                if not _finite(parent_end):
                    unknown("parent_completion_evidence_missing:" + call.call_id)
                    valid = False
                elif start < parent_end:
                    fail("child_started_before_parent_completed:" + call.call_id)
                    valid = False
        if valid and all(key in good_calls for key in identities):
            workflow_completed += 1

    successful = not failures and not incomplete and len(good_calls) == 10 and workflow_completed == 2
    return {
        "schema_version": 1, "scope": "compatible_per_request_solver_functional_smoke",
        "status": "PASSED" if successful else ("FAILED" if failures else "UNKNOWN"),
        "successful": successful, "actor_batch_owner_validation": False, "performance_claims": False,
        "expected_workflows": 2, "expected_calls": 10,
        "request_attempt_rows": len(raw_rows), "planner_schedule_rows": len(schedules),
        "worker_dequeue_events": dequeue_count, "completed_calls": len(completed),
        "usage_verified_calls": len(usage_verified), "solver_joined_calls": len(solver_joined),
        "worker_joined_calls": len(worker_joined), "workflow_completed_count": workflow_completed,
        "observed_prompt_tokens": prompt_tokens, "observed_completion_tokens": completion_tokens,
        "selected_replica_call_counts": replica_counts, "selected_pair_call_counts": dict(sorted(pairs.items())),
        "queue_contention": contention,
        "kv_transfer_evidence": ("inferred_from_joined_actual_P_D_dequeues_and_completed_generation;"
                                 "transfer_bytes_and_independent_KV_health_not_measured")
                                if successful else "incomplete_or_unmatched_end_to_end_evidence",
        "failure_reasons": failures, "incomplete_reasons": incomplete,
    }
