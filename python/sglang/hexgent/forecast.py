"""Live identity adapter around the unchanged canonical scheduling core."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
import json

from sglang.hexgent.scheduler import AgenticPDExactScheduler


def global_call_id(workflow_id: str, call_id: str) -> str:
    """Encode workflow-local IDs without separator collisions or ID guessing."""
    return json.dumps([str(workflow_id), str(call_id)], ensure_ascii=False, separators=(",", ":"))


def _normalized_call(call):
    workflow_id = str(call["workflow_id"])
    changes = {
        "call_id": global_call_id(workflow_id, call["call_id"]),
        "parent_call_ids": tuple(global_call_id(workflow_id, parent) for parent in call.get("parent_call_ids", ())),
    }
    return replace(call, **changes) if is_dataclass(call) else {**call, **changes}


def _local_decisions(decisions):
    result = []
    for decision in decisions:
        workflow_id, call_id = json.loads(decision.call_id)
        result.append(replace(decision, call_id=call_id, workflow_id=workflow_id))
    return result


class LiveStableScheduler(AgenticPDExactScheduler):
    """Keep the live API's workflow-local IDs outside the frozen core."""

    def schedule_prefill(self, calls, state, *, snapshot_for_refinement=None):
        normalized_state = state
        priority_calls = state.get("decode_priority_calls")
        if priority_calls is not None:
            changes = {"decode_priority_calls": tuple(_normalized_call(call) for call in priority_calls)}
            normalized_state = replace(state, **changes) if is_dataclass(state) else {**state, **changes}
        return _local_decisions(super().schedule_prefill(
            [_normalized_call(call) for call in (calls.values() if isinstance(calls, dict) else calls)], normalized_state,
            snapshot_for_refinement=snapshot_for_refinement,
        ))

    def schedule_decode(self, calls, state, *, snapshot_for_refinement=None):
        return _local_decisions(super().schedule_decode(
            [_normalized_call(call) for call in (calls.values() if isinstance(calls, dict) else calls)], state,
            snapshot_for_refinement=snapshot_for_refinement,
        ))
