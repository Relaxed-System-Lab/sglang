"""HexAgenT workflow-aware PD scheduler.

This is a standalone port of the fixed handoff scheduler. It keeps the same
dict-like dataclass access pattern used by the simulator while removing the
OpenEvolve/runtime dependencies.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Sequence, Tuple


GREEDY_RECOMPUTE_LIMIT = 32
UNIFORM_SHORT_DECODE_OUTPUT_TOKENS = 96
UNIFORM_SHORT_DECODE_INPUT_TOKENS = 256
LARGE_CLUSTER_BALANCE_THRESHOLD = 16
FANOUT_DECODE_INPUT_TOKENS = 256
FANOUT_DECODE_OUTPUT_TOKENS = 128
PREFILL_DECODE_SPEED_PENALTY = 1.0
SHORT_DECODE_SPEED_BIAS = 0.25
SHORT_DECODE_SLOW_INSTANCE_CUTOFF = 1.55
OURS_PLACEMENT_STRATEGY_ENV_VAR = "AGENTIC_PD_OURS_PLACEMENT_STRATEGY"


def _ours_placement_strategy() -> str:
    strategy = os.environ.get(OURS_PLACEMENT_STRATEGY_ENV_VAR, "projected_finish")
    strategy = str(strategy).strip().lower()
    if strategy not in {"projected_finish", "weighted"}:
        return "projected_finish"
    return strategy


class MappingViewMixin:
    def as_dict(self) -> Dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> object:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()

    def values(self):
        return self.as_dict().values()


@dataclass
class PrefillCallInfo(MappingViewMixin):
    call_id: str
    workflow_id: str
    input_tokens: int
    output_tokens: int
    workflow_arrival_ms: int
    reveal_sequence: int
    workflow_horizon_ms: int
    workflow_completion_ms: int
    workflow_attained_service_ms: int
    ideal_service_ms: int
    prefill_ms_by_instance: List[int]
    decode_ms_by_instance: List[int]
    transfer_ms_by_decode_instance: List[int]
    critical_path_ms: int = 0
    child_count: int = 0
    # Parents are dependency edges that are known when this call is revealed.
    # They let the scheduler recognize sibling fan-out without assuming future
    # descendants through `child_count`.
    parent_call_ids: Tuple[str, ...] = ()


@dataclass
class DecodeCallInfo(MappingViewMixin):
    call_id: str
    workflow_id: str
    input_tokens: int
    output_tokens: int
    workflow_arrival_ms: int
    reveal_sequence: int
    workflow_horizon_ms: int
    workflow_completion_ms: int
    workflow_attained_service_ms: int
    ideal_service_ms: int
    ready_at_ms: int
    decode_demand_tokens: int
    decode_ms_by_instance: List[int]
    locked_decode_instance: Optional[int]
    critical_path_ms: int = 0
    child_count: int = 0
    parent_call_ids: Tuple[str, ...] = ()


@dataclass
class PrefillStageState(MappingViewMixin):
    current_time_ms: int
    instance_available_at_ms: List[int]
    decode_available_hint_ms: List[int]


@dataclass
class DecodeStageState(MappingViewMixin):
    current_time_ms: int
    capacity_max_tokens: List[int]
    scheduled_intervals: List[List[Tuple[int, int, int]]]


@dataclass
class StageDecision:
    call_id: str
    assigned_instance: int
    priority_score: float
    planned_decode_instance: Optional[int] = None


@dataclass
class SchedulerStats:
    prefill_invocations: int = 0
    decode_invocations: int = 0
    prefill_solver_ms: float = 0.0
    decode_solver_ms: float = 0.0


def _has_revealed_sibling_fanout(calls: Sequence[MappingViewMixin]) -> bool:
    """Return whether two revealed calls share the same known parent set.

    A child-count hint describes potential future work and is therefore not a
    valid requirement for online fan-out detection.  Sibling calls that have
    already arrived expose their common parent IDs, which is sufficient to
    identify a currently schedulable fan-out group.
    """

    sibling_counts: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    for call in calls:
        raw_parent_ids = call.get("parent_call_ids", ())
        if isinstance(raw_parent_ids, str):
            parent_ids = (raw_parent_ids,)
        elif raw_parent_ids:
            parent_ids = tuple(sorted(str(parent_id) for parent_id in raw_parent_ids))
        else:
            continue
        key = (str(call["workflow_id"]), parent_ids)
        sibling_counts[key] = sibling_counts.get(key, 0) + 1
        if sibling_counts[key] >= 2:
            return True
    return False


def find_earliest_capacity_start(
    *,
    ready_at_ms: int,
    duration_ms: int,
    demand_tokens: int,
    scheduled_intervals: Sequence[Tuple[int, int, int]],
    capacity_tokens: int,
) -> int:
    if not scheduled_intervals:
        return ready_at_ms
    candidate = ready_at_ms
    checkpoints = sorted(
        {
            ready_at_ms,
            *[start for start, _, _ in scheduled_intervals],
            *[end for _, end, _ in scheduled_intervals],
        }
    )
    for point in checkpoints:
        candidate = max(candidate, point)
        end = candidate + duration_ms
        valid = True
        overlap_points = sorted(
            {
                candidate,
                end,
                *[start for start, _, _ in scheduled_intervals],
                *[finish for _, finish, _ in scheduled_intervals],
            }
        )
        for left, right in zip(overlap_points, overlap_points[1:]):
            if right <= candidate or left >= end:
                continue
            used = sum(
                demand
                for start, finish, demand in scheduled_intervals
                if start < right and finish > left
            )
            if used + demand_tokens > capacity_tokens:
                valid = False
                break
        if valid:
            return candidate
    return max(candidate, max(end for _, end, _ in scheduled_intervals))


class AgenticPDExactScheduler:
    def __init__(self) -> None:
        self.stats = SchedulerStats()
        self.workflow_arrivals: Dict[str, int] = {}
        self.workflow_horizons: Dict[str, int] = {}
        # Identical same-host workers frequently have equal predicted finish
        # times. Rotate only those exact ties to avoid a permanent index-zero
        # bias while retaining all latency-based heterogeneous choices.
        self._prefill_tie_cursor = 0
        self._decode_tie_cursor = 0

    def on_workflow_arrival(self, workflow_id: str, arrival_ms: int, horizon_ms: int) -> None:
        self.workflow_arrivals[workflow_id] = arrival_ms
        self.workflow_horizons[workflow_id] = horizon_ms

    def on_call_completed(self, call_id: str, workflow_id: str, service_ms: int) -> None:
        return None

    @staticmethod
    def _rotating_tie_rank(index: int, count: int, cursor: int) -> int:
        return (int(index) - int(cursor)) % max(1, int(count))

    def _assignment_tie_key(
        self,
        *,
        prefill_instance: int,
        decode_instance: int,
        prefill_count: int,
        decode_count: int,
    ) -> Tuple[int, int]:
        return (
            self._rotating_tie_rank(
                prefill_instance, prefill_count, self._prefill_tie_cursor
            ),
            self._rotating_tie_rank(
                decode_instance, decode_count, self._decode_tie_cursor
            ),
        )

    def _record_assignment(
        self,
        *,
        prefill_instance: int,
        decode_instance: int,
        prefill_count: int,
        decode_count: int,
    ) -> None:
        self._prefill_tie_cursor = (int(prefill_instance) + 1) % max(
            1, int(prefill_count)
        )
        self._decode_tie_cursor = (int(decode_instance) + 1) % max(
            1, int(decode_count)
        )

    def schedule_prefill(self, calls, state):
        started = time.perf_counter()
        call_list = list(calls.values()) if isinstance(calls, dict) else list(calls)
        prefill_available = [int(value) for value in state["instance_available_at_ms"]]
        decode_available = [int(value) for value in state["decode_available_hint_ms"]]
        current_time_ms = int(state["current_time_ms"])
        uniform_short_decode = (
            bool(call_list)
            and max(int(call["output_tokens"]) for call in call_list)
            <= UNIFORM_SHORT_DECODE_OUTPUT_TOKENS
            and max(int(call["input_tokens"]) for call in call_list)
            <= UNIFORM_SHORT_DECODE_INPUT_TOKENS
        )
        fanout_decode_batch = (
            bool(call_list)
            and min(int(call["output_tokens"]) for call in call_list)
            >= FANOUT_DECODE_OUTPUT_TOKENS
            and max(int(call["input_tokens"]) for call in call_list)
            <= FANOUT_DECODE_INPUT_TOKENS
        )
        small_sequential_batch = (
            bool(call_list)
            and max(int(call["output_tokens"]) for call in call_list)
            <= UNIFORM_SHORT_DECODE_OUTPUT_TOKENS
            and max(int(call["input_tokens"]) for call in call_list)
            <= UNIFORM_SHORT_DECODE_INPUT_TOKENS
            and max(int(call["child_count"]) for call in call_list) <= 1
            and max(int(call["ideal_service_ms"]) for call in call_list) <= 6500
        )
        workflow_batch_counts = {}
        for call in call_list:
            workflow_batch_counts[str(call["workflow_id"])] = (
                workflow_batch_counts.get(str(call["workflow_id"]), 0) + 1
            )
        same_workflow_fanout_batch = (
            max(workflow_batch_counts.values(), default=0) >= 2
            and max(int(call["child_count"]) for call in call_list) >= 2
        ) or _has_revealed_sibling_fanout(call_list)
        high_service_batch = (
            bool(call_list)
            and max(int(call["ideal_service_ms"]) for call in call_list) > 6500
        )
        large_uniform_cluster = (
            uniform_short_decode
            and len(prefill_available) >= LARGE_CLUSTER_BALANCE_THRESHOLD
            and len(decode_available) >= LARGE_CLUSTER_BALANCE_THRESHOLD
        )
        fastest_decode_late = (
            bool(call_list)
            and min(
                range(len(decode_available)),
                key=lambda idx: min(
                    int(call["decode_ms_by_instance"][idx]) for call in call_list
                ),
            )
            >= len(decode_available) // 2
        )
        large_short_decode_cluster = (
            uniform_short_decode
            and len(prefill_available) >= 6
            and len(decode_available) >= 6
            and fastest_decode_late
        )
        max_prefill_ms = max(
            (
                max(int(value) for value in call["prefill_ms_by_instance"])
                for call in call_list
            ),
            default=1,
        )
        max_decode_ms = max(
            (
                max(int(value) for value in call["decode_ms_by_instance"])
                for call in call_list
            ),
            default=0,
        )
        decode_dominated_uniform = (
            False
            and uniform_short_decode
            and not large_uniform_cluster
            and len(prefill_available) >= 4
            and max_decode_ms >= 4 * max(1, max_prefill_ms)
        )

        def best_assignment(call):
            if _ours_placement_strategy() == "weighted":
                prefill_inst = min(
                    range(len(prefill_available)),
                    key=lambda idx: (
                        max(current_time_ms, prefill_available[idx])
                        + int(call["prefill_ms_by_instance"][idx]),
                        int(call["prefill_ms_by_instance"][idx]),
                        self._rotating_tie_rank(
                            idx,
                            len(prefill_available),
                            self._prefill_tie_cursor,
                        ),
                    ),
                )
                prefill_start = max(current_time_ms, prefill_available[prefill_inst])
                prefill_end = prefill_start + int(
                    call["prefill_ms_by_instance"][prefill_inst]
                )
                decode_inst = min(
                    range(len(decode_available)),
                    key=lambda idx: (
                        max(current_time_ms, decode_available[idx])
                        + int(call["transfer_ms_by_decode_instance"][idx])
                        + int(call["decode_ms_by_instance"][idx]),
                        int(call["decode_ms_by_instance"][idx]),
                        self._rotating_tie_rank(
                            idx,
                            len(decode_available),
                            self._decode_tie_cursor,
                        ),
                    ),
                )
                decode_start = max(
                    int(decode_available[decode_inst]),
                    prefill_end
                    + int(call["transfer_ms_by_decode_instance"][decode_inst]),
                )
                decode_end = decode_start + int(call["decode_ms_by_instance"][decode_inst])
                return (
                    decode_end,
                    decode_end,
                    decode_start,
                    prefill_end,
                    prefill_start,
                    *self._assignment_tie_key(
                        prefill_instance=prefill_inst,
                        decode_instance=decode_inst,
                        prefill_count=len(prefill_available),
                        decode_count=len(decode_available),
                    ),
                    prefill_inst,
                    decode_inst,
                )
            best = None
            fastest_decode_ms = min(int(value) for value in call["decode_ms_by_instance"])
            for prefill_inst, available_ms in enumerate(prefill_available):
                prefill_start = max(current_time_ms, available_ms)
                prefill_end = prefill_start + int(
                    call["prefill_ms_by_instance"][prefill_inst]
                )
                for decode_inst in range(len(decode_available)):
                    decode_duration_ms = int(call["decode_ms_by_instance"][decode_inst])
                    decode_start = max(
                        int(decode_available[decode_inst]),
                        prefill_end
                        + int(call["transfer_ms_by_decode_instance"][decode_inst]),
                    )
                    decode_end = decode_start + decode_duration_ms
                    speed_penalty = (
                        int(
                            PREFILL_DECODE_SPEED_PENALTY
                            * max(0, decode_duration_ms - fastest_decode_ms)
                        )
                        if large_short_decode_cluster
                        else 0
                    )
                    assignment_score = (
                        decode_duration_ms
                        if large_short_decode_cluster
                        else decode_end + speed_penalty
                    )
                    candidate = (
                        assignment_score,
                        decode_end,
                        decode_start,
                        prefill_end,
                        prefill_start,
                        *self._assignment_tie_key(
                            prefill_instance=prefill_inst,
                            decode_instance=decode_inst,
                            prefill_count=len(prefill_available),
                            decode_count=len(decode_available),
                        ),
                        prefill_inst,
                        decode_inst,
                    )
                    if best is None or candidate < best:
                        best = candidate
            return best

        def projected_score(call):
            best = best_assignment(call)
            projected_call_elapsed = max(0, int(best[1]) - current_time_ms)
            workflow_age_ms = max(
                0,
                current_time_ms
                - int(self.workflow_arrivals.get(call["workflow_id"], current_time_ms)),
            )
            projected_completion = max(
                int(call["workflow_completion_ms"]),
                workflow_age_ms,
            ) + projected_call_elapsed
            projected_ratio = projected_completion / max(
                1, int(call["workflow_horizon_ms"])
            )
            return projected_ratio, -int(call["ideal_service_ms"])

        def workflow_laxity(call):
            deadline_ms = int(call["workflow_arrival_ms"]) + int(
                call["workflow_horizon_ms"]
            )
            remaining_known_service = max(
                0,
                int(call["workflow_horizon_ms"])
                - int(call["workflow_attained_service_ms"]),
            )
            return deadline_ms - current_time_ms - remaining_known_service

        def call_laxity(call):
            deadline_ms = int(call["workflow_arrival_ms"]) + int(
                call["workflow_horizon_ms"]
            )
            stage_service = (
                min(int(value) for value in call["prefill_ms_by_instance"])
                + min(int(value) for value in call["transfer_ms_by_decode_instance"])
                + min(int(value) for value in call["decode_ms_by_instance"])
            )
            return deadline_ms - current_time_ms - stage_service

        def critical_path_pressure(call):
            return max(0, int(call["critical_path_ms"])) / max(
                1, int(call["workflow_horizon_ms"])
            )

        def priority_key(call):
            if same_workflow_fanout_batch:
                projected_ratio, service_tiebreak = projected_score(call)
                return (
                    -critical_path_pressure(call),
                    -projected_ratio,
                    -int(call["critical_path_ms"]),
                    -int(call["child_count"]),
                    service_tiebreak,
                    str(call["call_id"]),
                )
            if small_sequential_batch or decode_dominated_uniform:
                return (call_laxity(call), int(call["reveal_sequence"]), str(call["call_id"]))
            projected_ratio, service_tiebreak = projected_score(call)
            if fanout_decode_batch:
                return (
                    -critical_path_pressure(call),
                    -projected_ratio,
                    -int(call["critical_path_ms"]),
                    -int(call["child_count"]),
                    service_tiebreak,
                    str(call["call_id"]),
                )
            return (-projected_ratio, -service_tiebreak, str(call["call_id"]))

        decisions = []
        if large_short_decode_cluster and not fanout_decode_batch:
            ordered_calls = sorted(
                call_list,
                key=lambda call: (
                    workflow_laxity(call),
                    -(
                        max(
                            int(call["workflow_completion_ms"]),
                            max(
                                0,
                                current_time_ms
                                - int(
                                    self.workflow_arrivals.get(
                                        call["workflow_id"], current_time_ms
                                    )
                                ),
                            ),
                        )
                        + int(call["ideal_service_ms"])
                    )
                    / max(1, int(call["workflow_horizon_ms"])),
                    int(call["reveal_sequence"]),
                    str(call["call_id"]),
                ),
            )
            decode_virtual_load = [
                max(0.0, float(value - current_time_ms) * 0.10)
                for value in decode_available
            ]
            for rank, call in enumerate(ordered_calls):
                prefill_inst = min(
                    range(len(prefill_available)),
                    key=lambda idx: (
                        max(current_time_ms, prefill_available[idx])
                        + int(call["prefill_ms_by_instance"][idx]),
                        int(call["prefill_ms_by_instance"][idx]),
                        idx,
                    ),
                )
                demand_tokens = max(
                    1, int(call["input_tokens"]) + int(call["output_tokens"])
                )
                fastest_decode_ms = min(int(value) for value in call["decode_ms_by_instance"])
                eligible_decode_instances = [
                    idx
                    for idx, value in enumerate(call["decode_ms_by_instance"])
                    if int(value) <= int(fastest_decode_ms * SHORT_DECODE_SLOW_INSTANCE_CUTOFF)
                ]
                if len(eligible_decode_instances) < 4:
                    eligible_decode_instances = list(range(len(decode_available)))
                decode_inst = min(
                    eligible_decode_instances,
                    key=lambda idx: (
                        decode_virtual_load[idx]
                        + SHORT_DECODE_SPEED_BIAS
                        * int(call["decode_ms_by_instance"][idx]),
                        int(call["decode_ms_by_instance"][idx]),
                        idx,
                    ),
                )
                prefill_start = max(current_time_ms, prefill_available[prefill_inst])
                prefill_end = prefill_start + int(
                    call["prefill_ms_by_instance"][prefill_inst]
                )
                decode_virtual_load[decode_inst] += int(
                    call["decode_ms_by_instance"][decode_inst]
                ) * min(1.0, demand_tokens / 4096.0)
                prefill_available[prefill_inst] = prefill_end
                decisions.append(
                    StageDecision(call["call_id"], prefill_inst, float(rank), decode_inst)
                )
            self.stats.prefill_invocations += 1
            self.stats.prefill_solver_ms += (time.perf_counter() - started) * 1000.0
            return decisions

        if (
            len(call_list) <= GREEDY_RECOMPUTE_LIMIT
            and not decode_dominated_uniform
            and not large_uniform_cluster
            and not fanout_decode_batch
        ):
            ordered_calls = []
            pending = list(call_list)
            while pending:
                call = min(pending, key=priority_key)
                pending.remove(call)
                ordered_calls.append(call)
                best = best_assignment(call)
                _, decode_end, _, prefill_end, _, _, _, prefill_inst, decode_inst = best
                prefill_available[prefill_inst] = prefill_end
                decode_available[decode_inst] = decode_end
            prefill_available = [int(value) for value in state["instance_available_at_ms"]]
            decode_available = [int(value) for value in state["decode_available_hint_ms"]]
        else:
            ordered_calls = sorted(call_list, key=priority_key)

        for rank, call in enumerate(ordered_calls):
            if large_uniform_cluster:
                prefill_inst = min(
                    range(len(prefill_available)),
                    key=lambda idx: (prefill_available[idx], idx),
                )
                decode_inst = min(
                    range(len(decode_available)),
                    key=lambda idx: (decode_available[idx], idx),
                )
                prefill_start = max(current_time_ms, prefill_available[prefill_inst])
                prefill_end = prefill_start + int(
                    call["prefill_ms_by_instance"][prefill_inst]
                )
                decode_start = max(
                    decode_available[decode_inst],
                    prefill_end
                    + int(call["transfer_ms_by_decode_instance"][decode_inst]),
                )
                decode_end = decode_start + int(call["decode_ms_by_instance"][decode_inst])
            else:
                best = best_assignment(call)
                (
                    _,
                    decode_end,
                    _,
                    prefill_end,
                    prefill_start,
                    _,
                    _,
                    prefill_inst,
                    decode_inst,
                ) = best
            prefill_available[prefill_inst] = prefill_end
            decode_available[decode_inst] = decode_end
            self._record_assignment(
                prefill_instance=prefill_inst,
                decode_instance=decode_inst,
                prefill_count=len(prefill_available),
                decode_count=len(decode_available),
            )
            decisions.append(
                StageDecision(call["call_id"], prefill_inst, float(rank), decode_inst)
            )
        self.stats.prefill_invocations += 1
        self.stats.prefill_solver_ms += (time.perf_counter() - started) * 1000.0
        return decisions

    def schedule_decode(self, calls, state):
        started = time.perf_counter()
        call_list = list(calls.values()) if isinstance(calls, dict) else list(calls)
        intervals = [list(items) for items in state["scheduled_intervals"]]
        current_time_ms = int(state["current_time_ms"])
        uniform_short_decode = (
            bool(call_list)
            and max(int(call["output_tokens"]) for call in call_list)
            <= UNIFORM_SHORT_DECODE_OUTPUT_TOKENS
            and max(int(call["input_tokens"]) for call in call_list)
            <= UNIFORM_SHORT_DECODE_INPUT_TOKENS
        )
        fanout_decode_batch = (
            bool(call_list)
            and min(int(call["output_tokens"]) for call in call_list)
            >= FANOUT_DECODE_OUTPUT_TOKENS
            and max(int(call["input_tokens"]) for call in call_list)
            <= FANOUT_DECODE_INPUT_TOKENS
        )
        small_sequential_batch = (
            bool(call_list)
            and max(int(call["output_tokens"]) for call in call_list)
            <= UNIFORM_SHORT_DECODE_OUTPUT_TOKENS
            and max(int(call["input_tokens"]) for call in call_list)
            <= UNIFORM_SHORT_DECODE_INPUT_TOKENS
            and max(int(call["child_count"]) for call in call_list) <= 1
            and max(int(call["ideal_service_ms"]) for call in call_list) <= 6500
        )
        workflow_batch_counts = {}
        for call in call_list:
            workflow_batch_counts[str(call["workflow_id"])] = (
                workflow_batch_counts.get(str(call["workflow_id"]), 0) + 1
            )
        same_workflow_fanout_batch = (
            max(workflow_batch_counts.values(), default=0) >= 2
            and max(int(call["child_count"]) for call in call_list) >= 2
        ) or _has_revealed_sibling_fanout(call_list)
        high_service_batch = (
            bool(call_list)
            and max(int(call["ideal_service_ms"]) for call in call_list) > 6500
        )
        large_uniform_cluster = uniform_short_decode and len(intervals) >= LARGE_CLUSTER_BALANCE_THRESHOLD
        fastest_decode_late = (
            bool(call_list)
            and min(
                range(len(intervals)),
                key=lambda idx: min(
                    int(call["decode_ms_by_instance"][idx]) for call in call_list
                ),
            )
            >= len(intervals) // 2
        )
        large_short_decode_cluster = (
            uniform_short_decode and len(intervals) >= 6 and fastest_decode_late
        )
        max_decode_ms = max(
            (
                max(int(value) for value in call["decode_ms_by_instance"])
                for call in call_list
            ),
            default=0,
        )
        min_ideal_service_ms = min(
            (int(call["ideal_service_ms"]) for call in call_list), default=1
        )
        decode_dominated_uniform = (
            False
            and uniform_short_decode
            and not large_uniform_cluster
            and len(intervals) >= 4
            and 4 * max_decode_ms >= 3 * max(1, min_ideal_service_ms)
        )

        def best_assignment(call):
            candidate_instances = (
                [int(call["locked_decode_instance"])]
                if call["locked_decode_instance"] is not None
                else list(range(len(state["capacity_max_tokens"])))
            )
            if _ours_placement_strategy() == "weighted":
                inst = min(
                    candidate_instances,
                    key=lambda idx: (
                        sum(
                            max(0, int(end_ms) - current_time_ms)
                            for _, end_ms, _ in intervals[idx]
                        )
                        + int(call["decode_ms_by_instance"][idx]),
                        int(call["decode_ms_by_instance"][idx]),
                        idx,
                    ),
                )
                start_ms = find_earliest_capacity_start(
                    ready_at_ms=int(call["ready_at_ms"]),
                    duration_ms=int(call["decode_ms_by_instance"][inst]),
                    demand_tokens=int(call["decode_demand_tokens"]),
                    scheduled_intervals=intervals[inst],
                    capacity_tokens=int(state["capacity_max_tokens"][inst]),
                )
                finish_ms = start_ms + int(call["decode_ms_by_instance"][inst])
                return (finish_ms, start_ms, inst)
            best = None
            for decode_inst in candidate_instances:
                start_ms = find_earliest_capacity_start(
                    ready_at_ms=int(call["ready_at_ms"]),
                    duration_ms=int(call["decode_ms_by_instance"][decode_inst]),
                    demand_tokens=int(call["decode_demand_tokens"]),
                    scheduled_intervals=intervals[decode_inst],
                    capacity_tokens=int(state["capacity_max_tokens"][decode_inst]),
                )
                finish_ms = start_ms + int(call["decode_ms_by_instance"][decode_inst])
                candidate = (finish_ms, start_ms, decode_inst)
                if best is None or candidate < best:
                    best = candidate
            return best

        def projected_score(call):
            if fanout_decode_batch:
                candidate_instances = (
                    [int(call["locked_decode_instance"])]
                    if call["locked_decode_instance"] is not None
                    else list(range(len(state["capacity_max_tokens"])))
                )
                estimated_finish_ms = min(
                    max(current_time_ms, int(call["ready_at_ms"]))
                    + int(call["decode_ms_by_instance"][decode_inst])
                    for decode_inst in candidate_instances
                )
                projected_call_elapsed = max(0, estimated_finish_ms - current_time_ms)
            else:
                best = best_assignment(call)
                projected_call_elapsed = max(0, int(best[0]) - current_time_ms)
            workflow_age_ms = max(
                0,
                current_time_ms
                - int(self.workflow_arrivals.get(call["workflow_id"], current_time_ms)),
            )
            projected_completion = max(
                int(call["workflow_completion_ms"]),
                workflow_age_ms,
            ) + projected_call_elapsed
            projected_ratio = projected_completion / max(
                1, int(call["workflow_horizon_ms"])
            )
            return projected_ratio, -int(call["ideal_service_ms"])

        def workflow_laxity(call):
            deadline_ms = int(call["workflow_arrival_ms"]) + int(
                call["workflow_horizon_ms"]
            )
            remaining_known_service = max(
                0,
                int(call["workflow_horizon_ms"])
                - int(call["workflow_attained_service_ms"]),
            )
            return deadline_ms - current_time_ms - remaining_known_service

        def call_laxity(call):
            deadline_ms = int(call["workflow_arrival_ms"]) + int(
                call["workflow_horizon_ms"]
            )
            stage_service = min(int(value) for value in call["decode_ms_by_instance"])
            return deadline_ms - current_time_ms - stage_service

        def critical_path_pressure(call):
            return max(0, int(call["critical_path_ms"])) / max(
                1, int(call["workflow_horizon_ms"])
            )

        def priority_key(call):
            if same_workflow_fanout_batch:
                projected_ratio, service_tiebreak = projected_score(call)
                return (
                    -critical_path_pressure(call),
                    -projected_ratio,
                    -int(call["critical_path_ms"]),
                    -int(call["child_count"]),
                    service_tiebreak,
                    str(call["call_id"]),
                )
            if small_sequential_batch or decode_dominated_uniform:
                return (call_laxity(call), int(call["reveal_sequence"]), str(call["call_id"]))
            projected_ratio, service_tiebreak = projected_score(call)
            if fanout_decode_batch:
                return (
                    -critical_path_pressure(call),
                    -projected_ratio,
                    -int(call["critical_path_ms"]),
                    -int(call["child_count"]),
                    service_tiebreak,
                    str(call["call_id"]),
                )
            return (-projected_ratio, -service_tiebreak, str(call["call_id"]))

        decisions = []
        if large_short_decode_cluster and not fanout_decode_batch:
            ordered_calls = sorted(
                call_list,
                key=lambda call: (
                    workflow_laxity(call),
                    -(
                        max(
                            int(call["workflow_completion_ms"]),
                            max(
                                0,
                                current_time_ms
                                - int(
                                    self.workflow_arrivals.get(
                                        call["workflow_id"], current_time_ms
                                    )
                                ),
                            ),
                        )
                        + int(call["ideal_service_ms"])
                    )
                    / max(1, int(call["workflow_horizon_ms"])),
                    int(call["reveal_sequence"]),
                    str(call["call_id"]),
                ),
            )
            virtual_load = [
                sum(
                    max(0, int(end_ms) - current_time_ms)
                    * min(
                        1.0,
                        max(1, int(tokens))
                        / max(1, int(state["capacity_max_tokens"][idx])),
                    )
                    for _, end_ms, tokens in items
                )
                for idx, items in enumerate(intervals)
            ]
            for rank, call in enumerate(ordered_calls):
                candidate_instances = (
                    [int(call["locked_decode_instance"])]
                    if call["locked_decode_instance"] is not None
                    else list(range(len(state["capacity_max_tokens"])))
                )
                fastest_decode_ms = min(int(value) for value in call["decode_ms_by_instance"])
                fast_instances = [
                    idx
                    for idx in candidate_instances
                    if int(call["decode_ms_by_instance"][idx])
                    <= int(fastest_decode_ms * SHORT_DECODE_SLOW_INSTANCE_CUTOFF)
                ]
                if len(fast_instances) >= 4:
                    candidate_instances = fast_instances
                decode_inst = min(
                    candidate_instances,
                    key=lambda idx: (
                        virtual_load[idx]
                        + SHORT_DECODE_SPEED_BIAS
                        * int(call["decode_ms_by_instance"][idx]),
                        int(call["decode_ms_by_instance"][idx]),
                        idx,
                    ),
                )
                demand_tokens = max(1, int(call["decode_demand_tokens"]))
                virtual_load[decode_inst] += int(
                    call["decode_ms_by_instance"][decode_inst]
                ) * min(
                    1.0,
                    demand_tokens / max(1, int(state["capacity_max_tokens"][decode_inst])),
                )
                decisions.append(StageDecision(call["call_id"], decode_inst, float(rank)))
            self.stats.decode_invocations += 1
            self.stats.decode_solver_ms += (time.perf_counter() - started) * 1000.0
            return decisions

        if (
            len(call_list) <= GREEDY_RECOMPUTE_LIMIT
            and not decode_dominated_uniform
            and not large_uniform_cluster
            and not fanout_decode_batch
        ):
            ordered_calls = []
            pending = list(call_list)
            while pending:
                call = min(pending, key=priority_key)
                pending.remove(call)
                ordered_calls.append(call)
                finish_ms, start_ms, decode_inst = best_assignment(call)
                intervals[decode_inst].append(
                    (start_ms, finish_ms, int(call["decode_demand_tokens"]))
                )
            intervals = [list(items) for items in state["scheduled_intervals"]]
        else:
            ordered_calls = sorted(call_list, key=priority_key)

        for rank, call in enumerate(ordered_calls):
            finish_ms, start_ms, decode_inst = best_assignment(call)
            intervals[decode_inst].append(
                (start_ms, finish_ms, int(call["decode_demand_tokens"]))
            )
            decisions.append(StageDecision(call["call_id"], decode_inst, float(rank)))
        self.stats.decode_invocations += 1
        self.stats.decode_solver_ms += (time.perf_counter() - started) * 1000.0
        return decisions

    def get_stats(self):
        return asdict(self.stats)


def build_scheduler():
    return AgenticPDExactScheduler()
