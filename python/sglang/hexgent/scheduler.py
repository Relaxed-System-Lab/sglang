"""HexAgenT workflow-aware PD scheduler.

This is a standalone port of the fixed handoff scheduler. It keeps the same
dict-like dataclass access pattern used by the simulator while removing the
OpenEvolve/runtime dependencies.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
import os
import time
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Sequence, Tuple

from . import snapshot_quantile_guard as quantile_guard


GREEDY_RECOMPUTE_LIMIT = 32
LOCAL_SEARCH_MAX_CALLS = 8
LOCAL_SEARCH_MAX_PAIR_EVALUATIONS = (
    LOCAL_SEARCH_MAX_CALLS * (LOCAL_SEARCH_MAX_CALLS - 1) // 2
)
UNIFORM_SHORT_DECODE_OUTPUT_TOKENS = 96
UNIFORM_SHORT_DECODE_INPUT_TOKENS = 256
LARGE_CLUSTER_BALANCE_THRESHOLD = 16
FANOUT_DECODE_INPUT_TOKENS = 256
FANOUT_DECODE_OUTPUT_TOKENS = 128
PREFILL_DECODE_SPEED_PENALTY = 1.0
SHORT_DECODE_SPEED_BIAS = 0.25
SHORT_DECODE_SLOW_INSTANCE_CUTOFF = 1.55
OURS_PLACEMENT_STRATEGY_ENV_VAR = "AGENTIC_PD_OURS_PLACEMENT_STRATEGY"
PAIRWISE_LOCAL_SEARCH_ENV_VAR = "HEXGENT_ENABLE_PAIRWISE_LOCAL_SEARCH"
PROJECTED_RATIO_ORDER_ENV_VAR = "HEXGENT_PROJECTED_RATIO_ORDER"


def _ours_placement_strategy() -> str:
    strategy = os.environ.get(OURS_PLACEMENT_STRATEGY_ENV_VAR, "projected_finish")
    strategy = str(strategy).strip().lower()
    if strategy not in {"projected_finish", "weighted"}:
        return "projected_finish"
    return strategy


def _pairwise_local_search_enabled() -> bool:
    value = os.environ.get(PAIRWISE_LOCAL_SEARCH_ENV_VAR, "1")
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _projected_ratio_priority(projected_ratio: float) -> float:
    order = os.environ.get(PROJECTED_RATIO_ORDER_ENV_VAR, "high_first")
    if str(order).strip().lower() == "low_first":
        return projected_ratio
    return -projected_ratio


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


@dataclass(frozen=True)
class PrioritySeconds:
    """Public seconds retained independently of integer placement calendars."""
    current_time_s: float
    workflow_arrival_s: float
    workflow_horizon_s: float
    workflow_completion_s: float
    prefill_s_by_instance: Tuple[float, ...] = ()
    decode_s_by_instance: Tuple[float, ...] = ()
    transfer_s_by_decode_instance: Tuple[float, ...] = ()


def _continuous_projected_ratio(seconds, elapsed_ms, intrinsic_ms, intrinsic_s):
    wait_s = max(0, elapsed_ms - intrinsic_ms) / 1000.0
    age_s = max(0.0, seconds.current_time_s - seconds.workflow_arrival_s)
    completion_s = max(seconds.workflow_completion_s, age_s) + (wait_s + intrinsic_s)
    return completion_s / seconds.workflow_horizon_s


def decode_priority_projection(call, *, current_time_ms, workflow_arrival_ms,
                               projected_call_elapsed_ms, decode_instance):
    """Native D ratio arithmetic over supplied revealed-call primitives."""
    workflow_age_ms = max(0, current_time_ms - int(workflow_arrival_ms))
    projected_completion = max(
        int(call["workflow_completion_ms"]), workflow_age_ms,
    ) + projected_call_elapsed_ms
    projected_ratio = projected_completion / max(1, int(call["workflow_horizon_ms"]))
    seconds = call.get("priority_seconds")
    if seconds is not None:
        projected_ratio = _continuous_projected_ratio(
            seconds, projected_call_elapsed_ms,
            int(call["decode_ms_by_instance"][decode_instance]),
            seconds.decode_s_by_instance[decode_instance])
    return projected_completion, projected_ratio


def decode_priority_key(call, *, current_time_ms, projected_ratio=None, use_laxity=False,
                        use_workflow_laxity=False, workflow_arrival_ms=None):
    """Ordinary native D key; caller retains override and branch selection."""
    if use_workflow_laxity:
        deadline = int(call["workflow_arrival_ms"]) + int(call["workflow_horizon_ms"])
        remaining = max(0, int(call["workflow_horizon_ms"]) - int(call["workflow_attained_service_ms"]))
        age = max(0, current_time_ms - int(workflow_arrival_ms))
        return (deadline-current_time_ms-remaining,
                -(max(int(call["workflow_completion_ms"]), age) + int(call["ideal_service_ms"]))
                / max(1, int(call["workflow_horizon_ms"])),
                int(call["reveal_sequence"]), str(call["call_id"]))
    if use_laxity:
        deadline_ms = int(call["workflow_arrival_ms"]) + int(call["workflow_horizon_ms"])
        stage_service = min(int(value) for value in call["decode_ms_by_instance"])
        return (deadline_ms - current_time_ms - stage_service,
                int(call["reveal_sequence"]), str(call["call_id"]))
    return (_projected_ratio_priority(projected_ratio),
            int(call["ideal_service_ms"]), str(call["call_id"]))


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
    # A request already dispatched to a worker-local queue keeps its P/D pair.
    # Global admissions leave both fields unset and remain freely placeable.
    locked_prefill_instance: Optional[int] = None
    locked_decode_instance: Optional[int] = None
    critical_path_ms: int = 0
    child_count: int = 0
    # Parents are dependency edges that are known when this call is revealed.
    # They let the scheduler recognize sibling fan-out without assuming future
    # descendants through `child_count`.
    parent_call_ids: Tuple[str, ...] = ()
    # Explicit runtime demand may differ from the output-length prediction.
    decode_demand_tokens: Optional[int] = None
    prefill_read_not_before_ms_by_instance: Optional[List[int]] = None
    priority_seconds: Optional[PrioritySeconds] = None


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
    priority_seconds: Optional[PrioritySeconds] = None


@dataclass
class PrefillStageState(MappingViewMixin):
    current_time_ms: int
    instance_available_at_ms: List[int]
    decode_available_hint_ms: List[int]
    # Total per-instance capacity, not capacity remaining after running work.
    decode_capacity_max_tokens: Optional[List[int]] = None
    # Full-invocation resource certificate; A2 only, never a runtime guarantee.
    decode_headroom_certified: Optional[List[bool]] = None
    # Independent hard admission release per D; None/0 is neutral.
    decode_admission_not_before_ms: Optional[Tuple[int, ...]] = None
    # Existing public D load replayed with actual slot/nominal limits.
    decode_admission_intervals: Optional[List[List[Tuple[int, int, int]]]] = None
    decode_max_batch_size_by_instance: Optional[List[Optional[int]]] = None
    # Unadmitted public work as (D-ready release, whole D service, demand).
    # Existing D-ready FIFO order precedes incoming release ties.
    decode_admission_jobs: Optional[List[List[Tuple[int, int, int]]]] = None
    # Fresh revealed D views and aligned existing job IDs; feature OFF leaves None.
    decode_priority_calls: Optional[Tuple[DecodeCallInfo, ...]] = None
    decode_priority_overrides: Optional[Dict[str, Tuple[object, ...]]] = None
    decode_priority_job_ids: Optional[Tuple[Tuple[str, ...], ...]] = None


@dataclass
class DecodeStageState(MappingViewMixin):
    current_time_ms: int
    capacity_max_tokens: List[int]
    scheduled_intervals: List[List[Tuple[int, int, int]]]
    # Outer None retains legacy callers; A1 requires explicit runtime limits.
    max_batch_size_by_instance: Optional[List[Optional[int]]] = None
    # Independent hard admission release per D; None/0 is neutral.
    admission_not_before_ms: Optional[Tuple[int, ...]] = None


@dataclass
class StageDecision:
    call_id: str
    assigned_instance: int
    priority_score: float
    planned_decode_instance: Optional[int] = None
    workflow_id: Optional[str] = None


@dataclass
class SchedulerStats:
    prefill_invocations: int = 0
    decode_invocations: int = 0
    prefill_solver_ms: float = 0.0
    decode_solver_ms: float = 0.0
    prefill_local_search_evaluations: int = 0
    decode_local_search_evaluations: int = 0
    prefill_local_search_swaps: int = 0
    decode_local_search_swaps: int = 0


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
) -> int | float:
    if demand_tokens > capacity_tokens:
        return math.inf
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


def find_earliest_fifo_admission_start(
    *,
    current_time_ms: int,
    ready_at_ms: int,
    demand_tokens: int,
    scheduled_intervals: Sequence[Tuple[int, int, int]],
    capacity_tokens: int,
    max_batch_size: Optional[int],
) -> int | float:
    """Project strict FIFO admission using known memory and batch-slot releases.

    Every earlier admitted head starts no later than the admission barrier.
    Consequently occupancy only decreases after that barrier: release events,
    rather than interval backfilling, determine the next head's start.
    """
    if demand_tokens > capacity_tokens:
        return math.inf
    barrier = max(
        int(current_time_ms),
        int(ready_at_ms),
        max((int(start) for start, _, _ in scheduled_intervals), default=int(current_time_ms)),
    )
    releases = {}
    used_tokens = 0
    running_count = 0
    for _, finish, tokens in scheduled_intervals:
        if finish <= barrier:
            continue
        used_tokens += int(tokens)
        running_count += 1
        old_tokens, old_count = releases.get(int(finish), (0, 0))
        releases[int(finish)] = (old_tokens + int(tokens), old_count + 1)

    def fits():
        return (
            used_tokens + demand_tokens <= capacity_tokens
            and (max_batch_size is None or running_count < max_batch_size)
        )

    if fits():
        return barrier
    for finish, (released_tokens, released_count) in sorted(releases.items()):
        used_tokens -= released_tokens
        running_count -= released_count
        if fits():
            return finish
    return math.inf


def _bounded_pairwise_order_search(order, objective):
    """Apply at most one best-improving swap to a small scheduling order."""

    initial = list(order)
    if len(initial) < 2 or len(initial) > LOCAL_SEARCH_MAX_CALLS:
        return initial, 0, False

    best_order = initial
    best_score = objective(initial)
    evaluations = 0
    for left in range(len(initial) - 1):
        for right in range(left + 1, len(initial)):
            if evaluations >= LOCAL_SEARCH_MAX_PAIR_EVALUATIONS:
                return best_order, evaluations, best_order != initial
            candidate = list(initial)
            candidate[left], candidate[right] = candidate[right], candidate[left]
            score = objective(candidate)
            evaluations += 1
            if score < best_score:
                best_order = candidate
                best_score = score
    return best_order, evaluations, best_order != initial


@dataclass(frozen=True)
class RefinementRecord:
    """Creation-snapshot choice; it makes no application-time quantile claim."""
    criterion: str
    projection: str
    stage: str
    invocation_time_ms: int
    surviving_call_count: int
    snapshot_time_ms: Optional[int] = None
    snapshot_collected: bool = False
    skip_reason: Optional[str] = None
    population_count: Optional[int] = None
    rank95: Optional[int] = None
    rank99: Optional[int] = None
    moving_workflow_count: Optional[int] = None
    projector_evaluations: int = 0
    score_evaluations: int = 0
    swap_evaluations: int = 0
    eligible_swaps: int = 0
    selected_swap: bool = False
    baseline_q95: Optional[float] = None
    baseline_q99: Optional[float] = None
    selected_q95: Optional[float] = None
    selected_q99: Optional[float] = None
    delta_q95: Optional[float] = None
    delta_q99: Optional[float] = None
    background_provenance: Optional[str] = None
    q95_quantile_changed_any_tested: bool = False
    q99_quantile_changed_any_tested: bool = False
    q95_mover_at_rank_any_tested: bool = False
    q99_mover_at_rank_any_tested: bool = False
    q95_mover_crossed_or_touched_initial_boundary: bool = False
    q99_mover_crossed_or_touched_initial_boundary: bool = False


def find_earliest_fifo_release_admission_start(
    *, current_time_ms, ready_at_ms, duration_ms, demand_tokens,
    running_intervals, queued_jobs, capacity_tokens, max_batch_size,
    admission_not_before_ms=0,
):
    """Jointly replay released work; a new long call can delay later arrivals.

    Running service is fixed. Unadmitted work enters FIFO at its public release,
    preserving the supplied order for ties. Replaying after each insertion keeps
    later incoming service feasible instead of freezing future occupied slots.
    """
    jobs = [*queued_jobs, (ready_at_ms, duration_ms, demand_tokens)]
    intervals = list(running_intervals)
    candidate = len(jobs) - 1
    candidate_start = math.inf
    for index, (release, duration, demand) in sorted(
        enumerate(jobs), key=lambda item: (max(current_time_ms, item[1][0]), item[0])
    ):
        start = find_earliest_fifo_admission_start(
            current_time_ms=current_time_ms, ready_at_ms=max(release, admission_not_before_ms),
            demand_tokens=demand, scheduled_intervals=intervals,
            capacity_tokens=capacity_tokens, max_batch_size=max_batch_size)
        intervals.append((start, start + duration, demand))
        if index == candidate:
            candidate_start = start
    return candidate_start


@dataclass(frozen=True)
class PriorityAdmissionJob:
    ready_at_ms: int
    duration_ms: int
    demand_tokens: int
    priority_key: Tuple[object, ...]

    def __iter__(self):
        # Existing constructive copies and FIFO helpers retain their 3-value contract.
        return iter((self.ready_at_ms, self.duration_ms, self.demand_tokens))


def find_earliest_priority_release_admission_start(
    *, current_time_ms, ready_at_ms, duration_ms, demand_tokens, priority_key,
    running_intervals, queued_jobs, capacity_tokens, max_batch_size,
    admission_not_before_ms=0,
):
    """Admit released known jobs using frozen initial native-priority keys."""
    jobs = [*queued_jobs, PriorityAdmissionJob(ready_at_ms, duration_ms, demand_tokens, priority_key)]
    candidate = len(jobs)-1
    pending = list(enumerate(jobs))
    intervals = list(running_intervals)
    now = current_time_ms
    while pending:
        ready = [(index, job) for index, job in pending if job.ready_at_ms <= now]
        if not ready:
            now = min(job.ready_at_ms for _, job in pending)
            continue
        index, job = min(ready, key=lambda row: (row[1].priority_key, row[0]))
        start = find_earliest_fifo_admission_start(
            current_time_ms=current_time_ms,
            ready_at_ms=max(now, admission_not_before_ms),
            demand_tokens=job.demand_tokens, scheduled_intervals=intervals,
            capacity_tokens=capacity_tokens, max_batch_size=max_batch_size)
        next_release = min((other.ready_at_ms for _, other in pending
                            if other.ready_at_ms > now), default=math.inf)
        if next_release < math.inf and next_release <= start:
            now = next_release
            continue
        if index == candidate:
            return start
        intervals.append((start, start+job.duration_ms, job.demand_tokens))
        pending.remove((index, job))
        now = start


class AgenticPDExactScheduler:
    def __init__(self, *, decode_admission_projection: bool = False,
                 prefill_headroom_projection: bool = False,
                 prefill_decode_admission_projection: bool = False,
                 pairwise_refinement: bool = True,
                 refinement_projection: str = "legacy",
                 refinement_criterion: str = "legacy_max", control_config=None,
                 known_priority_insertion_projection: bool = False,
                 known_priority_insertion_backlog_gate: bool = False) -> None:
        if refinement_criterion not in {"legacy_max", "snapshot_guard"}:
            raise ValueError("unknown refinement criterion")
        if refinement_criterion == "snapshot_guard" and refinement_projection != "faithful":
            raise ValueError("snapshot_guard requires faithful refinement projection")
        self.control_config = control_config
        self.known_priority_insertion_projection = bool(known_priority_insertion_projection)
        self.known_priority_insertion_backlog_gate = bool(known_priority_insertion_backlog_gate)
        # Stable ratio controls opt in; other entry points retain the size gate.
        self.recompute_large_ratio_batches = False
        self._control_stats = {"queue_blind_score_evaluations":0}
        self.refinement_criterion = refinement_criterion
        self.decode_admission_projection = bool(decode_admission_projection)
        self.prefill_headroom_projection = bool(prefill_headroom_projection)
        self.prefill_decode_admission_projection = bool(prefill_decode_admission_projection)
        self.pairwise_refinement = bool(pairwise_refinement)
        # Independent projection factor: "legacy" retains historical replay;
        # "faithful" scores the existing emitted P placement with local cursors.
        self.refinement_projection = refinement_projection
        self.stats = SchedulerStats()
        self.last_refinement_record = None
        self.workflow_arrivals: Dict[str, int] = {}
        self.workflow_horizons: Dict[str, int] = {}
        # Identical same-host workers frequently have equal predicted finish
        # times. Rotate only those exact ties to avoid a permanent index-zero
        # bias while retaining all latency-based heterogeneous choices.
        self._prefill_tie_cursor = 0
        self._decode_tie_cursor = 0
        self._fanout_workflows: set[str] = set()

    def _queue_aware_placement(self, *, for_priority=False):
        # The one Q switch controls projected queue costs in BOTH priority and
        # placement. The Hex ratio formula and hard ready/hold floors remain.
        return self.control_config is None or self.control_config.queue_aware_placement

    def _placement_score_observed(self, for_priority=False):
        if not self._queue_aware_placement(for_priority=for_priority):
            self._control_stats["queue_blind_score_evaluations"] += 1

    def get_control_stats(self):
        return dict(self._control_stats)

    def _priority_key_override(self, stage, call):
        """Optional complete ordering key, checked before native priority work."""
        return None

    def _projected_priority_ratio(self, call, projected_completion, original_ratio):
        """Default-neutral seam for a priority-only observed-state overlay."""
        return original_ratio

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
        tie_cursors: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        if tie_cursors is None:
            tie_cursors = (self._prefill_tie_cursor, self._decode_tie_cursor)
        return (
            self._rotating_tie_rank(
                prefill_instance, prefill_count, tie_cursors[0]
            ),
            self._rotating_tie_rank(
                decode_instance, decode_count, tie_cursors[1]
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

    def _refinement_skip(self, stage, current_time_ms, count, reason):
        self.last_refinement_record = RefinementRecord(
            self.refinement_criterion, self.refinement_projection,
            stage, current_time_ms, count, skip_reason=reason)

    def _refine_order(self, order, project_order, legacy_objective, *, stage,
                      current_time_ms, contains_fanout_workflow,
                      snapshot_for_refinement):
        """Reuse the native bounded search with one frozen public objective."""
        reason = (
            "pairwise_off" if not self.pairwise_refinement else
            "global_off" if not _pairwise_local_search_enabled() else
            "revealed_fanout" if contains_fanout_workflow else
            "too_few_calls" if len(order) < 2 else
            "too_many_calls" if len(order) > LOCAL_SEARCH_MAX_CALLS else None
        )
        public = initial_score = selected_score = initial_ratios = None
        arrivals = ownership = movers = None
        objective_evaluations = eligible_swaps = 0
        predicates = {f"q{q}_{name}": False for q in (95, 99) for name in (
            "quantile_changed_any_tested", "mover_at_rank_any_tested",
            "mover_crossed_or_touched_initial_boundary")}

        def objective(candidate_order):
            nonlocal public, initial_score, selected_score, initial_ratios
            nonlocal arrivals, ownership, movers, objective_evaluations, eligible_swaps
            objective_evaluations += 1
            if self.refinement_criterion == "legacy_max":
                return legacy_objective(candidate_order)
            if public is None:
                eligible_ids = tuple(str(call["call_id"]) for call in candidate_order)
                public = snapshot_for_refinement(eligible_ids)
                ownership = dict(public.snapshot.movable_call_workflows)
                if ownership.keys() != set(eligible_ids):
                    raise ValueError("snapshot movable call IDs must equal eligible call IDs")
                arrivals = dict(public.arrival_ms_by_workflow)
                movers = set(ownership.values())
            finishes, total = project_order(candidate_order)
            candidate = quantile_guard.Candidate(
                {cid: finish - arrivals[ownership[cid]] for cid, finish in finishes.items()},
                total,
            )
            candidate_score, ratio_items = quantile_guard.score_with_ratios(public.snapshot, candidate)
            ratios = dict(ratio_items)
            baseline = initial_score is None
            if baseline:
                initial_score = selected_score = candidate_score
                initial_ratios = ratios
            ordered_ratios = sorted(ratios.values())
            for q, rank in ((95, public.rank95), (99, public.rank99)):
                initial_q = getattr(initial_score, f"q{q}")
                predicates[f"q{q}_quantile_changed_any_tested"] |= (
                    getattr(candidate_score, f"q{q}") != initial_q)
                for workflow in movers:
                    value = ratios[workflow]
                    predicates[f"q{q}_mover_at_rank_any_tested"] |= (
                        bisect_left(ordered_ratios, value) < rank <=
                        bisect_right(ordered_ratios, value))
                    before = initial_ratios[workflow]
                    predicates[f"q{q}_mover_crossed_or_touched_initial_boundary"] |= (
                        before != value and min(before, value) <= initial_q <= max(before, value))
            admissible = quantile_guard.eligible(candidate_score, initial_score)
            if not baseline:
                eligible_swaps += int(admissible)
            if admissible and candidate_score.selection_key < selected_score.selection_key:
                selected_score = candidate_score
            return (candidate_score.selection_key if admissible
                    else initial_score.selection_key)

        if reason in {"pairwise_off", "global_off", "revealed_fanout"}:
            result, evaluations, swapped = list(order), 0, False
        else:
            result, evaluations, swapped = _bounded_pairwise_order_search(order, objective)
        metadata = {}
        if public is not None:
            metadata = dict(
                snapshot_time_ms=public.collected_at_ms, snapshot_collected=True,
                population_count=public.active_workflow_count,
                rank95=public.rank95, rank99=public.rank99,
                moving_workflow_count=public.moving_workflow_count,
                baseline_q95=initial_score.q95, baseline_q99=initial_score.q99,
                selected_q95=selected_score.q95, selected_q99=selected_score.q99,
                delta_q95=selected_score.q95-initial_score.q95,
                delta_q99=selected_score.q99-initial_score.q99,
                background_provenance=public.background_provenance,
            )
        self.last_refinement_record = RefinementRecord(
            self.refinement_criterion, self.refinement_projection, stage,
            current_time_ms, len(order),
            skip_reason=reason,
            projector_evaluations=objective_evaluations,
            score_evaluations=objective_evaluations, swap_evaluations=evaluations,
            eligible_swaps=eligible_swaps, selected_swap=swapped,
            **metadata, **predicates,
        )
        return result, evaluations, swapped

    def _legacy_order_score(self, candidate_order, finishes, total, current_time_ms):
        """Historical reducer, preserving elapsed-first arithmetic and order."""
        ratios_by_workflow = {}
        for candidate_call in candidate_order:
            projected_elapsed = max(0, finishes[str(candidate_call["call_id"])] - current_time_ms)
            workflow_age_ms = max(
                0,
                current_time_ms - int(self.workflow_arrivals.get(
                    candidate_call["workflow_id"], current_time_ms)),
            )
            projected_completion = max(
                int(candidate_call["workflow_completion_ms"]), workflow_age_ms,
            ) + projected_elapsed
            ratio = projected_completion / max(1, int(candidate_call["workflow_horizon_ms"]))
            workflow_id = str(candidate_call["workflow_id"])
            ratios_by_workflow[workflow_id] = max(
                ratio, ratios_by_workflow.get(workflow_id, 0.0))
        ratios = list(ratios_by_workflow.values())
        return (max(ratios, default=0.0), sum(ratios) / max(1, len(ratios)), total)

    def schedule_prefill(self, calls, state, *, snapshot_for_refinement=None):
        if self.refinement_criterion == "snapshot_guard" and snapshot_for_refinement is None:
            raise ValueError("snapshot_guard requires a bound snapshot provider callback")
        started = time.perf_counter()
        call_list = list(calls.values()) if isinstance(calls, dict) else list(calls)
        prefill_available = [int(value) for value in state["instance_available_at_ms"]]
        decode_available = [int(value) for value in state["decode_available_hint_ms"]]
        current_time_ms = int(state["current_time_ms"])
        priority_projection = (self.known_priority_insertion_projection
                               and self._queue_aware_placement())
        if (priority_projection and self.known_priority_insertion_backlog_gate
                and state["decode_priority_calls"] == ()):
            priority_projection = False
        decode_admission_not_before = tuple(
            state.get("decode_admission_not_before_ms") or (0,) * len(decode_available)
        )

        initial_decode_intervals = state.get("decode_admission_intervals")
        slot_projection = (self.prefill_decode_admission_projection
                           and initial_decode_intervals is not None)
        initial_decode_jobs = state.get("decode_admission_jobs")
        release_projection = slot_projection and initial_decode_jobs is not None
        initial_decode_schedules = (initial_decode_jobs if release_projection else initial_decode_intervals)
        decode_intervals = ([list(row) for row in initial_decode_schedules]
                            if slot_projection else None)

        def admission_start(call, worker, p_end, scheduled, *, queue_aware=True):
            arrival = p_end + int(call["transfer_ms_by_decode_instance"][worker])
            if release_projection:
                if priority_projection and queue_aware:
                    return find_earliest_priority_release_admission_start(
                        current_time_ms=current_time_ms, ready_at_ms=arrival,
                        duration_ms=int(call["decode_ms_by_instance"][worker]),
                        demand_tokens=int(call["decode_demand_tokens"]),
                        priority_key=candidate_priority_key(call, worker, arrival),
                        running_intervals=initial_decode_intervals[worker],
                        queued_jobs=scheduled[worker],
                        capacity_tokens=int(state["decode_capacity_max_tokens"][worker]),
                        max_batch_size=state["decode_max_batch_size_by_instance"][worker],
                        admission_not_before_ms=decode_admission_not_before[worker])
                return find_earliest_fifo_release_admission_start(
                    current_time_ms=current_time_ms, ready_at_ms=arrival,
                    duration_ms=int(call["decode_ms_by_instance"][worker]),
                    demand_tokens=int(call["decode_demand_tokens"]),
                    running_intervals=initial_decode_intervals[worker] if queue_aware else (),
                    queued_jobs=scheduled[worker],
                    capacity_tokens=int(state["decode_capacity_max_tokens"][worker]),
                    max_batch_size=state["decode_max_batch_size_by_instance"][worker],
                    admission_not_before_ms=decode_admission_not_before[worker])
            return find_earliest_fifo_admission_start(
                current_time_ms=current_time_ms,
                ready_at_ms=max(arrival, decode_admission_not_before[worker]),
                demand_tokens=int(call["decode_demand_tokens"]),
                scheduled_intervals=scheduled[worker],
                capacity_tokens=int(state["decode_capacity_max_tokens"][worker]),
                max_batch_size=state["decode_max_batch_size_by_instance"][worker])

        def append_decode(scheduled, call, worker, start, finish, p_end):
            if scheduled is not None:
                if release_projection:
                    ready = p_end + int(call["transfer_ms_by_decode_instance"][worker])
                    job = (ready, finish-start, int(call["decode_demand_tokens"]))
                    if priority_projection:
                        job = PriorityAdmissionJob(*job, candidate_priority_key(call, worker, ready))
                    scheduled[worker].append(job)
                else:
                    scheduled[worker].append((start, finish, int(call["decode_demand_tokens"])))

        def prefill_finish(call, worker, start, *, queue_aware=True):
            boundary = call.get("prefill_read_not_before_ms_by_instance")
            not_before = int(boundary[worker]) if queue_aware and boundary is not None else 0
            return max(start, not_before) + int(call["prefill_ms_by_instance"][worker])

        def prefill_candidates(call):
            locked = call["locked_prefill_instance"]
            return (
                [int(locked)]
                if locked is not None
                else list(range(len(prefill_available)))
            )

        def decode_candidates(call):
            locked = call["locked_decode_instance"]
            candidates = (
                [int(locked)]
                if locked is not None
                else list(range(len(decode_available)))
            )
            capacities = state.get("decode_capacity_max_tokens")
            if capacities is None:
                return candidates
            demand = call.get("decode_demand_tokens")
            if demand is None:
                demand = int(call["input_tokens"]) + int(call["output_tokens"])
            return [index for index in candidates if int(demand) <= int(capacities[index])]

        # An infeasible existing lock remains waiting; never migrate it here.
        call_list = [call for call in call_list if decode_candidates(call)]
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
        calls_by_workflow = {}
        for call in call_list:
            calls_by_workflow.setdefault(str(call["workflow_id"]), []).append(call)
        self._fanout_workflows.update(
            workflow_id
            for workflow_id, workflow_calls in calls_by_workflow.items()
            if (
                len(workflow_calls) >= 2
                and max(int(call["child_count"]) for call in workflow_calls) >= 2
            ) or _has_revealed_sibling_fanout(workflow_calls)
        )
        contains_fanout_workflow = any(
            str(call["workflow_id"]) in self._fanout_workflows for call in call_list
        )
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

        # Both native virtual-load routes retain their complete frozen invocation.
        certificate = state.get("decode_headroom_certified")
        headroom = (
            list(certificate)
            if self.prefill_headroom_projection and certificate is not None
            and len(certificate) == len(decode_available)
            and not large_uniform_cluster
            and not (large_short_decode_cluster and not fanout_decode_batch)
            else [False] * len(decode_available)
        )

        if priority_projection:
            if not release_projection:
                raise ValueError("known-priority insertion requires public release jobs")
            priority_calls = {str(call["call_id"]): call for call in state["decode_priority_calls"]}
            overrides = state["decode_priority_overrides"]
            priority_batch = list(priority_calls.values())
            short = bool(priority_batch) and max(int(c["output_tokens"]) for c in priority_batch) <= UNIFORM_SHORT_DECODE_OUTPUT_TOKENS and max(int(c["input_tokens"]) for c in priority_batch) <= UNIFORM_SHORT_DECODE_INPUT_TOKENS
            small = short and max(int(c["child_count"]) for c in priority_batch) <= 1 and max(int(c["ideal_service_ms"]) for c in priority_batch) <= 6500
            fanout = any(str(c["workflow_id"]) in self._fanout_workflows for c in priority_batch)
            fastest_late = bool(priority_batch) and min(range(len(decode_available)), key=lambda d: min(int(c["decode_ms_by_instance"][d]) for c in priority_batch)) >= len(decode_available)//2
            workflow_laxity_priority = short and len(decode_available) >= 6 and fastest_late
            candidate_keys = {}

            def snapshot_priority_key(call_id, worker, ready):
                if overrides is not None:
                    return overrides[call_id]
                view = priority_calls[call_id]
                arrival = self.workflow_arrivals.get(view["workflow_id"], current_time_ms)
                if workflow_laxity_priority:
                    return decode_priority_key(view, current_time_ms=current_time_ms,
                        use_workflow_laxity=True, workflow_arrival_ms=arrival)
                if small and not fanout:
                    return decode_priority_key(view, current_time_ms=current_time_ms, use_laxity=True)
                start = find_earliest_fifo_admission_start(
                    current_time_ms=current_time_ms,
                    ready_at_ms=max(ready, decode_admission_not_before[worker]),
                    demand_tokens=int(view["decode_demand_tokens"]),
                    scheduled_intervals=initial_decode_intervals[worker],
                    capacity_tokens=int(state["decode_capacity_max_tokens"][worker]),
                    max_batch_size=state["decode_max_batch_size_by_instance"][worker])
                elapsed = max(0, start+int(view["decode_ms_by_instance"][worker])-current_time_ms)
                completion, ratio = decode_priority_projection(view,
                    current_time_ms=current_time_ms, workflow_arrival_ms=arrival,
                    projected_call_elapsed_ms=elapsed, decode_instance=worker)
                ratio = self._projected_priority_ratio(view, completion, ratio)
                return decode_priority_key(view, current_time_ms=current_time_ms, projected_ratio=ratio)

            def candidate_priority_key(call, worker, ready):
                key = (str(call["call_id"]), worker, ready)
                if key not in candidate_keys:
                    candidate_keys[key] = snapshot_priority_key(key[0], worker, ready)
                    name = "candidate_priority_keys_prepared"
                    self._control_stats[name] = self._control_stats.get(name, 0)+1
                return candidate_keys[key]

            prepared = []
            for worker, (jobs, ids) in enumerate(zip(initial_decode_jobs, state["decode_priority_job_ids"])):
                row = []
                for job, call_id in zip(jobs, ids):
                    key = snapshot_priority_key(call_id, worker, job[0])
                    row.append(PriorityAdmissionJob(*job, key))
                    name = "known_priority_keys_prepared"
                    self._control_stats[name] = self._control_stats.get(name, 0)+1
                prepared.append(tuple(row))
            initial_decode_schedules = tuple(prepared)
            decode_intervals = [list(row) for row in initial_decode_schedules]

        def best_assignment(
            call,
            prefill_slots=None,
            decode_slots=None,
            tie_cursors=None,
            *, for_priority=False, decode_schedules=None,
        ):
            if tie_cursors is None:
                tie_cursors = (self._prefill_tie_cursor, self._decode_tie_cursor)
            active_prefill = (
                prefill_available if prefill_slots is None else prefill_slots
            )
            active_decode = decode_available if decode_slots is None else decode_slots
            actual_prefill,actual_decode=active_prefill,active_decode
            actual_schedules = decode_intervals if decode_schedules is None else decode_schedules
            active_schedules = actual_schedules
            queue_aware=self._queue_aware_placement(for_priority=for_priority)
            self._placement_score_observed(for_priority)
            if not queue_aware:
                active_prefill=[current_time_ms]*len(actual_prefill)
                active_decode=[current_time_ms]*len(actual_decode)
                if slot_projection:
                    active_schedules=[[] for _ in actual_schedules]

            def actual_finish(result):
                # Priority consumes the blinded cost. Constructive replay keeps
                # actual waits on the chosen route for legal state updates.
                if queue_aware or for_priority:return result
                p,d=result[-2:]
                p_start=max(current_time_ms,actual_prefill[p])
                p_end=prefill_finish(call,p,p_start)
                d_start=(admission_start(call,d,p_end,actual_schedules) if slot_projection else
                         max(current_time_ms if headroom[d] else actual_decode[d],
                             p_end+int(call["transfer_ms_by_decode_instance"][d]),
                             decode_admission_not_before[d]))
                return (result[0],d_start+int(call["decode_ms_by_instance"][d]),d_start,p_end,p_start,*result[5:])

            if _ours_placement_strategy() == "weighted":
                prefill_inst = min(
                    prefill_candidates(call),
                    key=lambda idx: (
                        prefill_finish(call,idx,max(current_time_ms,active_prefill[idx]),
                                       queue_aware=queue_aware),
                        int(call["prefill_ms_by_instance"][idx]),
                        self._rotating_tie_rank(
                            idx,
                            len(active_prefill),
                            tie_cursors[0],
                        ),
                    ),
                )
                prefill_start = max(current_time_ms, active_prefill[prefill_inst])
                prefill_end = prefill_finish(call,prefill_inst,prefill_start,
                                             queue_aware=queue_aware)
                decode_inst = min(
                    decode_candidates(call),
                    key=lambda idx: (
                        (admission_start(call,idx,prefill_end,active_schedules,queue_aware=queue_aware) if slot_projection else
                         max(current_time_ms, current_time_ms if headroom[idx] else active_decode[idx],
                             decode_admission_not_before[idx])
                         + int(call["transfer_ms_by_decode_instance"][idx]))
                        + int(call["decode_ms_by_instance"][idx]),
                        int(call["decode_ms_by_instance"][idx]),
                        self._rotating_tie_rank(
                            idx,
                            len(active_decode),
                            tie_cursors[1],
                        ),
                    ),
                )
                decode_start = (admission_start(call,decode_inst,prefill_end,active_schedules,queue_aware=queue_aware)
                                if slot_projection else max(
                    current_time_ms if headroom[decode_inst] else int(active_decode[decode_inst]),
                    prefill_end + int(call["transfer_ms_by_decode_instance"][decode_inst]),
                    decode_admission_not_before[decode_inst]))
                decode_end = decode_start + int(call["decode_ms_by_instance"][decode_inst])
                return actual_finish((
                    decode_end,
                    decode_end,
                    decode_start,
                    prefill_end,
                    prefill_start,
                    *self._assignment_tie_key(
                        prefill_instance=prefill_inst,
                        decode_instance=decode_inst,
                        prefill_count=len(active_prefill),
                        decode_count=len(active_decode),
                        tie_cursors=tie_cursors,
                    ),
                    prefill_inst,
                    decode_inst,
                ))
            best = None
            candidate_decode_instances = decode_candidates(call)
            fastest_decode_ms = min(
                int(call["decode_ms_by_instance"][idx])
                for idx in candidate_decode_instances
            )
            for prefill_inst in prefill_candidates(call):
                available_ms = active_prefill[prefill_inst]
                prefill_start = max(current_time_ms, available_ms)
                prefill_end = prefill_finish(call,prefill_inst,prefill_start,
                                             queue_aware=queue_aware)
                for decode_inst in candidate_decode_instances:
                    decode_duration_ms = int(call["decode_ms_by_instance"][decode_inst])
                    decode_start = (admission_start(call,decode_inst,prefill_end,active_schedules,queue_aware=queue_aware)
                                    if slot_projection else max(
                        current_time_ms if headroom[decode_inst] else int(active_decode[decode_inst]),
                        prefill_end + int(call["transfer_ms_by_decode_instance"][decode_inst]),
                        decode_admission_not_before[decode_inst]))
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
                            prefill_count=len(active_prefill),
                            decode_count=len(active_decode),
                            tie_cursors=tie_cursors,
                        ),
                        prefill_inst,
                        decode_inst,
                    )
                    if best is None or candidate < best:
                        best = candidate
            return actual_finish(best)

        def placement_assignment(call, prefill_slots=None, decode_slots=None,
                                 tie_cursors=None, decode_schedules=None):
            """Replay the existing emitted placement for one call."""
            if not large_uniform_cluster:
                return best_assignment(call, prefill_slots, decode_slots, tie_cursors,
                                       decode_schedules=decode_schedules)
            active_prefill = prefill_available if prefill_slots is None else prefill_slots
            active_decode = decode_available if decode_slots is None else decode_slots
            self._placement_score_observed()
            prefill_inst = min(
                prefill_candidates(call),
                key=lambda idx: (active_prefill[idx] if self._queue_aware_placement() else current_time_ms, idx),
            )
            decode_inst = min(
                decode_candidates(call),
                key=lambda idx: (active_decode[idx] if self._queue_aware_placement() else current_time_ms, idx),
            )
            prefill_start = max(current_time_ms, active_prefill[prefill_inst])
            prefill_end = prefill_finish(call,prefill_inst,prefill_start)
            decode_start = max(
                active_decode[decode_inst],
                prefill_end + int(call["transfer_ms_by_decode_instance"][decode_inst]),
                decode_admission_not_before[decode_inst],
            )
            decode_end = decode_start + int(call["decode_ms_by_instance"][decode_inst])
            # The balancing branch uses availability/index, so rotated tie ranks
            # do not participate in its assignment. Keep the common tuple shape.
            return (decode_end, decode_end, decode_start, prefill_end,
                    prefill_start, 0, 0, prefill_inst, decode_inst)

        def projected_score(call):
            best = best_assignment(call,for_priority=True)
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
            seconds = call.get("priority_seconds")
            if seconds is not None:
                p, d = best[-2:]
                intrinsic_ms = (int(call["prefill_ms_by_instance"][p])
                                + int(call["transfer_ms_by_decode_instance"][d])
                                + int(call["decode_ms_by_instance"][d]))
                intrinsic_s = (seconds.prefill_s_by_instance[p]
                               + seconds.transfer_s_by_decode_instance[d]) + seconds.decode_s_by_instance[d]
                projected_ratio = _continuous_projected_ratio(
                    seconds, projected_call_elapsed, intrinsic_ms, intrinsic_s)
            projected_ratio = self._projected_priority_ratio(
                call, projected_completion, projected_ratio
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

        def priority_key(call):
            override = self._priority_key_override("prefill", call)
            if override is not None:
                return override
            if (
                small_sequential_batch or decode_dominated_uniform
            ) and not contains_fanout_workflow:
                return (call_laxity(call), int(call["reveal_sequence"]), str(call["call_id"]))
            projected_ratio, service_tiebreak = projected_score(call)
            return (
                _projected_ratio_priority(projected_ratio),
                -service_tiebreak,
                str(call["call_id"]),
            )

        decisions = []
        if large_short_decode_cluster and not fanout_decode_batch:
            def large_short_priority_key(call):
                override = self._priority_key_override("prefill", call)
                if override is not None:
                    return override
                return (
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
                )

            ordered_calls = sorted(call_list, key=large_short_priority_key)
            decode_virtual_load = [
                max(0.0, float(value - current_time_ms) * 0.10) if self._queue_aware_placement() else 0.
                for value in decode_available
            ]
            for rank, call in enumerate(ordered_calls):
                prefill_inst = min(
                    prefill_candidates(call),
                    key=lambda idx: (
                        prefill_finish(call,idx,max(current_time_ms,
                            prefill_available[idx] if self._queue_aware_placement() else current_time_ms),
                            queue_aware=self._queue_aware_placement()),
                        int(call["prefill_ms_by_instance"][idx]),
                        idx,
                    ),
                )
                demand_tokens = max(
                    1, int(call["input_tokens"]) + int(call["output_tokens"])
                )
                candidate_decode_instances = decode_candidates(call)
                fastest_decode_ms = min(
                    int(call["decode_ms_by_instance"][idx])
                    for idx in candidate_decode_instances
                )
                eligible_decode_instances = [
                    idx
                    for idx in candidate_decode_instances
                    if int(call["decode_ms_by_instance"][idx])
                    <= int(fastest_decode_ms * SHORT_DECODE_SLOW_INSTANCE_CUTOFF)
                ]
                if len(eligible_decode_instances) < 4:
                    eligible_decode_instances = candidate_decode_instances
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
                prefill_end = prefill_finish(call,prefill_inst,prefill_start)
                self._placement_score_observed()
                if self._queue_aware_placement():
                    decode_virtual_load[decode_inst] += int(
                        call["decode_ms_by_instance"][decode_inst]
                    ) * min(1.0, demand_tokens / 4096.0)
                prefill_available[prefill_inst] = prefill_end
                decisions.append(
                    StageDecision(
                        call["call_id"],
                        prefill_inst,
                        float(rank),
                        decode_inst,
                        workflow_id=str(call["workflow_id"]),
                    )
                )
            self._refinement_skip("prefill", current_time_ms, len(call_list), "early_dispatch")
            self.stats.prefill_invocations += 1
            self.stats.prefill_solver_ms += (time.perf_counter() - started) * 1000.0
            return decisions

        if (
            (len(call_list) <= GREEDY_RECOMPUTE_LIMIT
             or (self.recompute_large_ratio_batches
                 and not ((small_sequential_batch or decode_dominated_uniform)
                          and not contains_fanout_workflow)))
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
                _, decode_end, decode_start, prefill_end, _, _, _, prefill_inst, decode_inst = best
                prefill_available[prefill_inst] = prefill_end
                append_decode(decode_intervals,call,decode_inst,decode_start,decode_end,prefill_end)
                if not headroom[decode_inst]:
                    decode_available[decode_inst] = decode_end
            prefill_available = [int(value) for value in state["instance_available_at_ms"]]
            decode_available = [int(value) for value in state["decode_available_hint_ms"]]
            if slot_projection:
                decode_intervals = [list(row) for row in initial_decode_schedules]
        else:
            ordered_calls = sorted(call_list, key=priority_key)

        def project_order(candidate_order):
            simulated_prefill = [
                int(value) for value in state["instance_available_at_ms"]
            ]
            simulated_decode = [
                int(value) for value in state["decode_available_hint_ms"]
            ]
            simulated_schedules = ([list(row) for row in initial_decode_schedules]
                                   if slot_projection else None)
            faithful = self.refinement_projection == "faithful"
            simulated_cursors = (self._prefill_tie_cursor, self._decode_tie_cursor)
            finish_ms_by_call = {}
            total_projected_elapsed = 0
            for candidate_call in candidate_order:
                if faithful:
                    assignment = placement_assignment(
                        candidate_call,
                        prefill_slots=simulated_prefill,
                        decode_slots=simulated_decode,
                        tie_cursors=simulated_cursors,
                        decode_schedules=simulated_schedules,
                    )
                else:
                    assignment = best_assignment(
                        candidate_call,
                        prefill_slots=simulated_prefill,
                        decode_slots=simulated_decode,
                        decode_schedules=simulated_schedules,
                    )
                (
                    _,
                    decode_end,
                    decode_start,
                    prefill_end,
                    _,
                    _,
                    _,
                    prefill_inst,
                    decode_inst,
                ) = assignment
                simulated_prefill[prefill_inst] = prefill_end
                append_decode(simulated_schedules,candidate_call,decode_inst,decode_start,decode_end,prefill_end)
                if not headroom[decode_inst]:
                    simulated_decode[decode_inst] = decode_end
                if faithful:
                    simulated_cursors = (
                        (int(prefill_inst) + 1) % max(1, len(simulated_prefill)),
                        (int(decode_inst) + 1) % max(1, len(simulated_decode)),
                    )
                projected_elapsed = max(0, int(decode_end) - current_time_ms)
                finish_ms_by_call[str(candidate_call["call_id"])] = int(decode_end)
                total_projected_elapsed += projected_elapsed
            return finish_ms_by_call, total_projected_elapsed

        def prefill_order_objective(candidate_order):
            finishes, total = project_order(candidate_order)
            return self._legacy_order_score(candidate_order, finishes, total, current_time_ms)

        ordered_calls, evaluations, swapped = self._refine_order(
            ordered_calls, project_order, prefill_order_objective,
            stage="prefill", current_time_ms=current_time_ms,
            contains_fanout_workflow=contains_fanout_workflow,
            snapshot_for_refinement=snapshot_for_refinement,
        )
        self.stats.prefill_local_search_evaluations += evaluations
        self.stats.prefill_local_search_swaps += int(swapped)

        for rank, call in enumerate(ordered_calls):
            if large_uniform_cluster or self.refinement_projection == "faithful":
                best = placement_assignment(call)
            else:
                best = best_assignment(call)
            (
                _,
                decode_end,
                decode_start,
                prefill_end,
                prefill_start,
                _,
                _,
                prefill_inst,
                decode_inst,
            ) = best
            prefill_available[prefill_inst] = prefill_end
            append_decode(decode_intervals,call,decode_inst,decode_start,decode_end,prefill_end)
            if not headroom[decode_inst]:
                decode_available[decode_inst] = decode_end
            self._record_assignment(
                prefill_instance=prefill_inst,
                decode_instance=decode_inst,
                prefill_count=len(prefill_available),
                decode_count=len(decode_available),
            )
            decisions.append(
                StageDecision(
                    call["call_id"],
                    prefill_inst,
                    float(rank),
                    decode_inst,
                    workflow_id=str(call["workflow_id"]),
                )
            )
        self.stats.prefill_invocations += 1
        self.stats.prefill_solver_ms += (time.perf_counter() - started) * 1000.0
        return decisions

    def schedule_decode(self, calls, state, *, snapshot_for_refinement=None):
        if self.refinement_criterion == "snapshot_guard" and snapshot_for_refinement is None:
            raise ValueError("snapshot_guard requires a bound snapshot provider callback")
        started = time.perf_counter()
        call_list = list(calls.values()) if isinstance(calls, dict) else list(calls)
        intervals = [list(items) for items in state["scheduled_intervals"]]
        current_time_ms = int(state["current_time_ms"])
        admission_not_before = tuple(
            state.get("admission_not_before_ms") or (0,) * len(intervals)
        )

        def decode_candidates(call):
            locked = call["locked_decode_instance"]
            candidates = (
                [int(locked)] if locked is not None
                else list(range(len(state["capacity_max_tokens"])))
            )
            return [index for index in candidates
                    if int(call["decode_demand_tokens"]) <= int(state["capacity_max_tokens"][index])]

        batch_limits = state.get("max_batch_size_by_instance")
        if self.decode_admission_projection and call_list and batch_limits is None:
            raise ValueError("A1 requires known max_batch_size_by_instance")
        call_list = [call for call in call_list if decode_candidates(call)]

        def admission_start(call, inst, scheduled):
            ready_lower = max(int(call["ready_at_ms"]), admission_not_before[inst])
            if self.decode_admission_projection:
                return find_earliest_fifo_admission_start(
                    current_time_ms=current_time_ms,
                    ready_at_ms=ready_lower,
                    demand_tokens=int(call["decode_demand_tokens"]),
                    scheduled_intervals=scheduled[inst],
                    capacity_tokens=int(state["capacity_max_tokens"][inst]),
                    max_batch_size=batch_limits[inst],
                )
            return find_earliest_capacity_start(
                ready_at_ms=ready_lower,
                duration_ms=int(call["decode_ms_by_instance"][inst]),
                demand_tokens=int(call["decode_demand_tokens"]),
                scheduled_intervals=scheduled[inst],
                capacity_tokens=int(state["capacity_max_tokens"][inst]),
            )

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
        calls_by_workflow = {}
        for call in call_list:
            calls_by_workflow.setdefault(str(call["workflow_id"]), []).append(call)
        self._fanout_workflows.update(
            workflow_id
            for workflow_id, workflow_calls in calls_by_workflow.items()
            if (
                len(workflow_calls) >= 2
                and max(int(call["child_count"]) for call in workflow_calls) >= 2
            ) or _has_revealed_sibling_fanout(workflow_calls)
        )
        contains_fanout_workflow = any(
            str(call["workflow_id"]) in self._fanout_workflows for call in call_list
        )
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

        def best_assignment(call, scheduled=None, *, for_priority=False):
            active_intervals = intervals if scheduled is None else scheduled
            actual_intervals=active_intervals
            queue_aware=self._queue_aware_placement(for_priority=for_priority)
            self._placement_score_observed(for_priority)
            if not queue_aware:
                active_intervals=[[] for _ in active_intervals]

            def actual_finish(result):
                # Priority consumes the blinded cost. Constructive replay keeps
                # actual waits on the chosen route for legal state updates.
                if queue_aware or for_priority:return result
                inst=result[-1]
                start=admission_start(call,inst,actual_intervals)
                return (start+int(call["decode_ms_by_instance"][inst]),start,inst)

            candidate_instances = decode_candidates(call)
            if _ours_placement_strategy() == "weighted":
                inst = min(
                    candidate_instances,
                    key=lambda idx: (
                        sum(
                            max(0, int(end_ms) - current_time_ms)
                            for _, end_ms, _ in active_intervals[idx]
                        )
                        + int(call["decode_ms_by_instance"][idx]),
                        int(call["decode_ms_by_instance"][idx]),
                        idx,
                    ),
                )
                start_ms = admission_start(call, inst, active_intervals)
                finish_ms = start_ms + int(call["decode_ms_by_instance"][inst])
                return actual_finish((finish_ms, start_ms, inst))
            best = None
            for decode_inst in candidate_instances:
                start_ms = admission_start(call, decode_inst, active_intervals)
                finish_ms = start_ms + int(call["decode_ms_by_instance"][decode_inst])
                candidate = (finish_ms, start_ms, decode_inst)
                if best is None or candidate < best:
                    best = candidate
            return actual_finish(best)

        def projected_score(call):
            if fanout_decode_batch and not self.decode_admission_projection:
                candidate_instances = decode_candidates(call)
                estimated_finish_ms, priority_decode_inst = min(
                    (max(current_time_ms, int(call["ready_at_ms"]), admission_not_before[decode_inst])
                     + int(call["decode_ms_by_instance"][decode_inst]), decode_inst)
                    for decode_inst in candidate_instances)
                projected_call_elapsed = max(0, estimated_finish_ms - current_time_ms)
            else:
                best = best_assignment(call,for_priority=True)
                projected_call_elapsed = max(0, int(best[0]) - current_time_ms)
                priority_decode_inst = best[-1]
            projected_completion, projected_ratio = decode_priority_projection(
                call, current_time_ms=current_time_ms,
                workflow_arrival_ms=self.workflow_arrivals.get(call["workflow_id"], current_time_ms),
                projected_call_elapsed_ms=projected_call_elapsed,
                decode_instance=priority_decode_inst)
            projected_ratio = self._projected_priority_ratio(
                call, projected_completion, projected_ratio
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

        def priority_key(call):
            override = self._priority_key_override("decode", call)
            if override is not None:
                return override
            if (
                small_sequential_batch or decode_dominated_uniform
            ) and not contains_fanout_workflow:
                return decode_priority_key(call, current_time_ms=current_time_ms, use_laxity=True)
            projected_ratio, service_tiebreak = projected_score(call)
            return decode_priority_key(call, current_time_ms=current_time_ms,
                                       projected_ratio=projected_ratio)

        decisions = []
        if large_short_decode_cluster and not fanout_decode_batch:
            def large_short_priority_key(call):
                override = self._priority_key_override("decode", call)
                if override is not None:
                    return override
                return decode_priority_key(call, current_time_ms=current_time_ms,
                    use_workflow_laxity=True,
                    workflow_arrival_ms=self.workflow_arrivals.get(call["workflow_id"], current_time_ms))

            ordered_calls = sorted(call_list, key=large_short_priority_key)
            # This excluded dispatch path retains the legacy adapter's
            # one-ms load for a current-time zero marker; A1 finish paths do not.
            virtual_load = [
                sum(
                    (
                        1
                        if self.decode_admission_projection
                        and int(start_ms) == int(end_ms) == current_time_ms
                        else max(0, int(end_ms) - current_time_ms)
                    )
                    * min(
                        1.0,
                        max(1, int(tokens))
                        / max(1, int(state["capacity_max_tokens"][idx])),
                    )
                    for start_ms, end_ms, tokens in items
                )
                for idx, items in enumerate(intervals)
            ]
            if not self._queue_aware_placement():
                virtual_load=[0.]*len(intervals)
            for rank, call in enumerate(ordered_calls):
                self._placement_score_observed()
                candidate_instances = decode_candidates(call)
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
                if self._queue_aware_placement():
                    virtual_load[decode_inst] += int(
                        call["decode_ms_by_instance"][decode_inst]
                    ) * min(
                        1.0,
                        demand_tokens / max(1, int(state["capacity_max_tokens"][decode_inst])),
                    )
                decisions.append(
                    StageDecision(
                        call["call_id"],
                        decode_inst,
                        float(rank),
                        workflow_id=str(call["workflow_id"]),
                    )
                )
            self._refinement_skip("decode", current_time_ms, len(call_list), "early_dispatch")
            self.stats.decode_invocations += 1
            self.stats.decode_solver_ms += (time.perf_counter() - started) * 1000.0
            return decisions

        if (
            (len(call_list) <= GREEDY_RECOMPUTE_LIMIT
             or (self.recompute_large_ratio_batches
                 and not ((small_sequential_batch or decode_dominated_uniform)
                          and not contains_fanout_workflow)))
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

        def project_order(candidate_order):
            simulated_intervals = [
                list(items) for items in state["scheduled_intervals"]
            ]
            finish_ms_by_call = {}
            total_projected_elapsed = 0
            for candidate_call in candidate_order:
                finish_ms, start_ms, decode_inst = best_assignment(
                    candidate_call, scheduled=simulated_intervals
                )
                simulated_intervals[decode_inst].append(
                    (
                        start_ms,
                        finish_ms,
                        int(candidate_call["decode_demand_tokens"]),
                    )
                )
                projected_elapsed = max(0, int(finish_ms) - current_time_ms)
                finish_ms_by_call[str(candidate_call["call_id"])] = int(finish_ms)
                total_projected_elapsed += projected_elapsed
            return finish_ms_by_call, total_projected_elapsed

        def decode_order_objective(candidate_order):
            finishes, total = project_order(candidate_order)
            return self._legacy_order_score(candidate_order, finishes, total, current_time_ms)

        ordered_calls, evaluations, swapped = self._refine_order(
            ordered_calls, project_order, decode_order_objective,
            stage="decode", current_time_ms=current_time_ms,
            contains_fanout_workflow=contains_fanout_workflow,
            snapshot_for_refinement=snapshot_for_refinement,
        )
        self.stats.decode_local_search_evaluations += evaluations
        self.stats.decode_local_search_swaps += int(swapped)

        for rank, call in enumerate(ordered_calls):
            finish_ms, start_ms, decode_inst = best_assignment(call)
            intervals[decode_inst].append(
                (start_ms, finish_ms, int(call["decode_demand_tokens"]))
            )
            decisions.append(
                StageDecision(
                    call["call_id"],
                    decode_inst,
                    float(rank),
                    workflow_id=str(call["workflow_id"]),
                )
            )
        self.stats.decode_invocations += 1
        self.stats.decode_solver_ms += (time.perf_counter() - started) * 1000.0
        return decisions

    def get_stats(self):
        return asdict(self.stats)


def build_scheduler() -> AgenticPDExactScheduler:
    """Frozen source232 stable-A live profile; no pairwise refinement."""
    from .forecast import LiveStableScheduler
    scheduler = LiveStableScheduler(
        decode_admission_projection=True,
        prefill_decode_admission_projection=True,
        known_priority_insertion_projection=True,
        known_priority_insertion_backlog_gate=True,
        pairwise_refinement=False,
        refinement_projection="faithful",
    )
    scheduler.recompute_large_ratio_batches = True
    return scheduler
