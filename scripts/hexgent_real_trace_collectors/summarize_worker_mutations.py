#!/usr/bin/env python3
"""Summarize worker-endpoint mutation outcomes from explicit worker logs.

Events cover received HTTP requests that returned per-rank tokenizer results.
They do not cover upstream transport errors or handler failures before results.
Rank outcomes are diagnostic subresults, not independent workflow samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFIX = "HexgentMutationOutcome "
ACTION_KEY = ("worker_port", "request_id", "action_type", "queue", "schedule_revision", "expected_revision")
OUTCOMES = ("success", "stale_revision", "expected_mismatch", "not_found", "legacy_unknown", "other_rejection")


def _rank_outcome(result: dict) -> str:
    if result["success"]:
        return "success"
    reason = result.get("reason_code")
    return {
        "stale_revision": "stale_revision",
        "expected_revision_mismatch": "expected_mismatch",
        "not_found": "not_found",
    }.get(reason, "legacy_unknown" if reason is None else "other_rejection")


def summarize_events(events: list[dict]) -> dict:
    rank_counts = dict.fromkeys(OUTCOMES, 0)
    actions = {}
    stale_attempts = 0
    for event in events:
        key = tuple(event[field] for field in ACTION_KEY)
        if key not in actions:
            actions[key] = {
                **dict(zip(ACTION_KEY, key)),
                "http_attempts": 0,
                "rank_outcomes": dict.fromkeys(OUTCOMES, 0),
                "any_stale_revision": False,
                "any_expected_mismatch": False,
                "eventual_applied": False,
            }
        action = actions[key]
        action["http_attempts"] += 1
        outcomes = [_rank_outcome(result) for result in event["results"]]
        for outcome in outcomes:
            rank_counts[outcome] += 1
            action["rank_outcomes"][outcome] += 1
        stale = "stale_revision" in outcomes
        stale_attempts += stale
        action["any_stale_revision"] |= stale
        action["any_expected_mismatch"] |= "expected_mismatch" in outcomes
        action["eventual_applied"] |= bool(event["success"])
    attempts, unique = len(events), len(actions)
    rank_total = sum(rank_counts.values())
    stale_unique = sum(action["any_stale_revision"] for action in actions.values())
    applied_unique = sum(action["eventual_applied"] for action in actions.values())
    return {
        "boundary": "worker_endpoint_received_requests",
        "scope_detail": "received requests with completed per-rank results; upstream transport and handler failures before results are unobserved",
        "statistical_unit": "HTTP attempts and unique action keys; rank outcomes are not independent workflow samples",
        "received_http_attempts": attempts,
        "unique_action_key_fields": list(ACTION_KEY),
        "unique_action_count": unique,
        "rank_outcome_count": rank_total,
        "rank_outcomes": rank_counts,
        "rank_outcome_rates": {name: count / rank_total if rank_total else None for name, count in rank_counts.items()},
        "rate_definition": "explicit observed outcome labels over all received rank results; legacy unknowns stay separate and are never inferred from messages",
        "attempt_any_stale_revision_count": stale_attempts,
        "attempt_any_stale_revision_rate": stale_attempts / attempts if attempts else None,
        "unique_any_stale_revision_count": stale_unique,
        "unique_any_stale_revision_rate": stale_unique / unique if unique else None,
        "unique_eventual_applied_count": applied_unique,
        "unique_eventual_applied_rate": applied_unique / unique if unique else None,
        "eventual_applied_definition": "at least one received attempt reported any-rank success; not an all-ranks-applied guarantee; may overlap any-stale",
        "transport_failure_count": None,
        "transport_failure_rate": None,
        "unique_actions": list(actions.values()),
    }


def read_events(log_paths: list[Path]) -> list[dict]:
    events = []
    for path in log_paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if PREFIX in line:
                    events.append(json.loads(line.split(PREFIX, 1)[1]))
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_events(read_events(args.log))
    report["input_logs"] = [str(path) for path in args.log]
    payload = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
