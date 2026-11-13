"""Pure fixed-population snapshot-proxy mathematics used by opt-in refinement.

Completion contributions are milliseconds relative to each workflow's arrival.
Public horizons, fixed contributions and movable-call ownership are frozen once.
The guarded quantiles describe this proxy, not final evaluator C/S quantiles.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import fsum, isfinite


@dataclass(frozen=True, init=False)
class Snapshot:
    """Immutable public inputs; each workflow has exactly one quantile vote."""

    horizon_ms_by_workflow: tuple[tuple[str, float], ...]
    fixed_completion_ms_by_workflow: tuple[tuple[str, float], ...]
    movable_call_workflows: tuple[tuple[str, str], ...]

    def __init__(
        self,
        horizon_ms_by_workflow: Mapping[str, float],
        fixed_completion_ms_by_workflow: Mapping[str, float],
        movable_call_workflows: Mapping[str, str],
    ) -> None:
        horizons = tuple(horizon_ms_by_workflow.items())
        fixed = tuple(fixed_completion_ms_by_workflow.items())
        calls = tuple(movable_call_workflows.items())
        population = {workflow for workflow, _ in horizons}
        if not population:
            raise ValueError('workflow population must be nonempty')
        if {workflow for workflow, _ in fixed} != population:
            raise ValueError('fixed completion workflow IDs must equal the population')
        if any(workflow not in population for _, workflow in calls):
            raise ValueError('movable calls must belong to the workflow population')
        if any(not isfinite(value) or value <= 0 for _, value in horizons):
            raise ValueError('public horizons must be positive and finite')
        if any(not isfinite(value) or value < 0 for _, value in fixed):
            raise ValueError('fixed completion contributions must be nonnegative and finite')
        object.__setattr__(self, 'horizon_ms_by_workflow', horizons)
        object.__setattr__(self, 'fixed_completion_ms_by_workflow', fixed)
        object.__setattr__(self, 'movable_call_workflows', calls)


@dataclass(frozen=True, init=False)
class Candidate:
    """Only movable completion values and the total projected call elapsed time."""

    completion_ms_by_call: tuple[tuple[str, float], ...]
    total_call_elapsed_ms: float

    def __init__(
        self,
        completion_ms_by_call: Mapping[str, float],
        total_call_elapsed_ms: float,
    ) -> None:
        completions = tuple(completion_ms_by_call.items())
        if any(not isfinite(value) or value < 0 for _, value in completions):
            raise ValueError('call completion contributions must be nonnegative and finite')
        if not isfinite(total_call_elapsed_ms) or total_call_elapsed_ms < 0:
            raise ValueError('total call elapsed time must be nonnegative and finite')
        object.__setattr__(self, 'completion_ms_by_call', completions)
        object.__setattr__(self, 'total_call_elapsed_ms', total_call_elapsed_ms)


@dataclass(frozen=True)
class Score:
    q95: float
    q99: float
    max_ratio: float
    mean_ratio: float
    total_call_elapsed_ms: float

    @property
    def selection_key(self) -> tuple[float, float, float, float, float]:
        """Predeclared order: tails, workflow mean, call elapsed, maximum."""
        return (self.q95, self.q99, self.mean_ratio,
                self.total_call_elapsed_ms, self.max_ratio)


@dataclass(frozen=True)
class Selection:
    candidate: Candidate
    score: Score
    candidate_index: int | None


def nearest_rank(values: Iterable[float], percentile: int) -> float:
    """Nearest rank for nonempty finite values and an integer percentile 1..100."""
    ordered = sorted(values)
    rank = (percentile * len(ordered) + 99) // 100
    return ordered[rank - 1]


def score_with_ratios(snapshot: Snapshot, candidate: Candidate):
    """Return the score and its already-computed workflow vector for diagnostics."""
    ownership = dict(snapshot.movable_call_workflows)
    completions = dict(candidate.completion_ms_by_call)
    if completions.keys() != ownership.keys():
        raise ValueError('candidate call IDs must equal the frozen movable call IDs')
    workflow_completion = dict(snapshot.fixed_completion_ms_by_workflow)
    for call_id, completion in completions.items():
        workflow = ownership[call_id]
        workflow_completion[workflow] = max(workflow_completion[workflow], completion)
    ratio_items = tuple((workflow, workflow_completion[workflow] / horizon)
                        for workflow, horizon in snapshot.horizon_ms_by_workflow)
    ratios = [value for _, value in ratio_items]
    result = Score(
        q95=nearest_rank(ratios, 95),
        q99=nearest_rank(ratios, 99),
        max_ratio=max(ratios),
        mean_ratio=fsum(ratios) / len(ratios),
        total_call_elapsed_ms=candidate.total_call_elapsed_ms,
    )

    return result, ratio_items


def score(snapshot: Snapshot, candidate: Candidate) -> Score:
    """Aggregate fixed and movable contributions once per frozen workflow."""
    return score_with_ratios(snapshot, candidate)[0]


def eligible(candidate_score: Score, initial_score: Score) -> bool:
    """Non-degradation of both tails; scores must come from the same snapshot."""
    return (candidate_score.q95 <= initial_score.q95
            and candidate_score.q99 <= initial_score.q99)


def select(
    snapshot: Snapshot,
    baseline: Candidate,
    candidates: Iterable[Candidate],
) -> Selection:
    """Choose a strict improvement, always guarding against the initial baseline.

    Candidate enumeration order resolves exact score ties. The baseline is kept
    on an exact tie; its returned candidate_index is None.
    """
    initial_score = score(snapshot, baseline)
    best = Selection(baseline, initial_score, None)
    for index, candidate in enumerate(candidates):
        candidate_score = score(snapshot, candidate)
        if (eligible(candidate_score, initial_score)
                and candidate_score.selection_key < best.score.selection_key):
            best = Selection(candidate, candidate_score, index)
    return best
