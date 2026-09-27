"""Per-stage latency budgets with graceful degradation.

A cascade is only useful if its tail latency is bounded. This module owns that decision: before a
stage runs, the controller *predicts* how long the planned amount of work will take from what past
requests actually cost, compares the prediction with the time left in the request budget, and
either lets the stage run, shrinks its work, or skips it.

Cost model. Each stage declares how many "units" of work it plans to do (candidates to fetch,
candidates to score). A stage's cost is modelled as ``fixed + rate * units``, inflated by
``safety_factor`` so a prediction is a pessimistic estimate rather than an average.

Only ``rate`` is fitted online, as an exponentially weighted moving average of
``(seconds - fixed) / units`` over the samples a stage reports; this is a one-parameter fit, not
two. ``fixed`` is a *measured constant* for the reference CPU, seeded per stage in
:class:`BudgetController` and never moved by live traffic, because no serving call site ever reports
zero units: retrieval always plans at least
``max(ef, k) >= 16`` nodes, ranking at least one row, the rules stage at least one item. Time above
the seeded intercept is therefore folded into ``rate``, which is the term the ladder divides by, so
an under- or over-stated intercept shows up as a slightly conservative or optimistic per-unit rate
rather than as a wrong shape. ``observe(0.0, seconds)`` does update ``fixed``, for callers (tests,
and any future stage with a genuine no-work path) that can measure the intercept directly. Fitting
both terms would need a running least squares over ``(units, seconds)`` or bucketed EWMAs; it is not
implemented, and ``test_budget.py`` pins the current behaviour so this description cannot drift.

Degradation ladder, cheapest loss of quality first:

1. ``lower_ef_search`` - halve the ANN ``efSearch`` (and, if needed, the candidate count) until
   the predicted retrieval cost fits. Recall drops; the ranking stage still runs.
2. ``lower_rerank_depth`` - rerank only the best-scoring prefix of the candidate set. Stage 2's
   cost is linear in rows (measured at about 13 microseconds per row for a 200-tree model), so
   halving the depth halves the cost, and since ``k`` is far smaller than the depth the items that
   fall off the end were never going to be shown.
3. ``skip_rerank`` - return the ANN order instead of the LambdaMART order. Precision drops a lot,
   the response is still relevant, and it is flagged ``degraded`` so callers and dashboards can
   see it.

Two inputs make overload visible, and both are needed.

*Queue time.* A request that waited 30 ms for a worker thread has already spent most of a 50 ms
budget before any stage runs, so :meth:`BudgetController.start` takes the elapsed time and the
request begins with what is left.

*Contention.* Queue time alone is not enough, because most of the waiting happens outside the
process - in the accept backlog and in clients' connection pools - where nothing can be measured
from here. What *can* be measured is how many accepted requests are in flight. With ``C`` requests
sharing ``N`` service slots each one should expect roughly ``max(1, C / N)`` times its solo cost, so
predictions are multiplied by that factor. Under load this makes the expensive stage look as
expensive as it really is, the ladder engages, responses get cheaper, and the backlog drains -
instead of every request optimistically planning full work and all of them missing the budget.

Nothing here sleeps, forks or cancels: a stage that has already started is never interrupted, so
a single pathological call can still overshoot. The controller bounds *planned* work, which is
what actually keeps p99 in range when load rises.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "DEGRADE_LOWER_DEPTH",
    "DEGRADE_LOWER_EF",
    "DEGRADE_SKIP_RERANK",
    "BudgetController",
    "CostEstimator",
    "RankingPlan",
    "RequestBudget",
    "RetrievalPlan",
    "StageTimings",
]

DEGRADE_LOWER_EF: Final = "lower_ef_search"
DEGRADE_LOWER_DEPTH: Final = "lower_rerank_depth"
DEGRADE_SKIP_RERANK: Final = "skip_rerank"

_MIN_EF_SEARCH: Final = 16
_MIN_CANDIDATES: Final = 16
_MIN_RERANK_DEPTH: Final = 32


class CostEstimator:
    """Thread-safe EWMA model of one stage's cost: ``fixed + rate * units`` seconds."""

    def __init__(
        self,
        *,
        fixed_seconds: float,
        seconds_per_unit: float,
        alpha: float = 0.2,
        safety_factor: float = 1.3,
    ) -> None:
        if fixed_seconds < 0.0 or seconds_per_unit < 0.0:
            raise ValueError("initial cost estimates must be non-negative")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if safety_factor < 1.0:
            raise ValueError("safety_factor must be >= 1")
        self._fixed = fixed_seconds
        self._rate = seconds_per_unit
        self._alpha = alpha
        self.safety_factor = safety_factor
        self._samples = 0
        self._lock = threading.Lock()

    @property
    def samples(self) -> int:
        with self._lock:
            return self._samples

    @property
    def fixed_seconds(self) -> float:
        with self._lock:
            return self._fixed

    @property
    def seconds_per_unit(self) -> float:
        with self._lock:
            return self._rate

    def predict(self, units: float) -> float:
        """Pessimistic estimate of the seconds needed for ``units`` of work."""
        if units < 0.0:
            raise ValueError("units must be non-negative")
        with self._lock:
            return self.safety_factor * (self._fixed + self._rate * units)

    def max_units(self, seconds: float) -> float:
        """Largest ``units`` still predicted to fit in ``seconds`` (0 if even no work fits)."""
        with self._lock:
            budget = seconds / self.safety_factor - self._fixed
            if budget <= 0.0:
                return 0.0
            if self._rate <= 0.0:
                return math.inf
            return budget / self._rate

    def observe(self, units: float, seconds: float) -> None:
        """Fold one measurement into the model. Non-finite or negative samples are ignored."""
        if units < 0.0 or seconds < 0.0 or not math.isfinite(seconds) or not math.isfinite(units):
            return
        with self._lock:
            self._samples += 1
            a = self._alpha
            if units <= 0.0:
                self._fixed = (1.0 - a) * self._fixed + a * seconds
                return
            # Attribute the part above the current fixed cost to the per-unit rate.
            variable = max(seconds - self._fixed, 0.0)
            self._rate = (1.0 - a) * self._rate + a * (variable / units)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "fixed_seconds": self._fixed,
                "seconds_per_unit": self._rate,
                "safety_factor": self.safety_factor,
                "samples": float(self._samples),
            }


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """How much ANN work stage 1 may do, after the budget has had its say."""

    candidate_k: int
    ef_search: int
    requested_candidate_k: int
    requested_ef_search: int
    predicted_seconds: float

    @property
    def lowered(self) -> bool:
        return (
            self.ef_search < self.requested_ef_search
            or self.candidate_k < self.requested_candidate_k
        )


@dataclass(frozen=True, slots=True)
class RankingPlan:
    """Whether stage 2 runs, and over how many of the retrieved candidates."""

    rerank: bool
    candidates: int
    requested_candidates: int
    predicted_seconds: float
    remaining_seconds: float

    @property
    def lowered(self) -> bool:
        return self.rerank and self.candidates < self.requested_candidates


@dataclass(slots=True)
class StageTimings:
    """Measured seconds per stage for one request, in the order the stages ran."""

    seconds: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, seconds: float) -> None:
        self.seconds[stage] = self.seconds.get(stage, 0.0) + seconds

    @property
    def total(self) -> float:
        return sum(self.seconds.values())

    def as_dict(self) -> dict[str, float]:
        return dict(self.seconds)


class RequestBudget:
    """One request's slice of the budget. Not thread-safe; create one per request."""

    def __init__(
        self,
        controller: BudgetController,
        clock: Callable[[], float],
        spent_seconds: float = 0.0,
        contention: float = 1.0,
    ) -> None:
        if spent_seconds < 0.0:
            raise ValueError("spent_seconds must be non-negative")
        if contention < 1.0 or not math.isfinite(contention):
            raise ValueError("contention must be a finite value >= 1")
        self._controller = controller
        self._clock = clock
        self._started = clock() - spent_seconds
        self.spent_before_start = spent_seconds
        self.contention = contention
        self.timings = StageTimings()
        self.degraded_reasons: list[str] = []

    def _predict(self, stage: str, units: float) -> float:
        """Predicted wall time for ``units`` of ``stage``, inflated by the current contention."""
        return self._controller.estimator(stage).predict(units) * self.contention

    def _affordable(self, stage: str, seconds: float) -> float:
        """Units of ``stage`` that fit in ``seconds`` at the current contention."""
        return self._controller.estimator(stage).max_units(seconds / self.contention)

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._started

    @property
    def remaining_seconds(self) -> float:
        return self._controller.total_seconds - self.elapsed_seconds

    @property
    def degraded(self) -> bool:
        return bool(self.degraded_reasons)

    def _degrade(self, reason: str) -> None:
        if reason not in self.degraded_reasons:
            self.degraded_reasons.append(reason)

    @contextmanager
    def stage(self, name: str, units: float) -> Iterator[None]:
        """Time a stage and feed the measurement back into its cost estimator."""
        start = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - start
            self.timings.record(name, elapsed)
            self._controller.estimator(name).observe(units, elapsed)

    def plan_retrieval(self, candidate_k: int, ef_search: int) -> RetrievalPlan:
        """Shrink ``ef_search`` (then ``candidate_k``) until stage 1 is predicted to fit."""
        if candidate_k < 1 or ef_search < 1:
            raise ValueError("candidate_k and ef_search must be >= 1")
        allowance = min(self._controller.retrieval_seconds, self.remaining_seconds)
        current_k, current_ef = candidate_k, ef_search
        predicted = self._predict(
            "retrieval", self._controller.retrieval_units(current_k, current_ef)
        )
        while predicted > allowance and (
            current_ef > _MIN_EF_SEARCH or current_k > _MIN_CANDIDATES
        ):
            if current_ef > max(_MIN_EF_SEARCH, current_k):
                current_ef = max(_MIN_EF_SEARCH, current_ef // 2)
            elif current_k > _MIN_CANDIDATES:
                current_k = max(_MIN_CANDIDATES, current_k // 2)
                # A search that visits more nodes than the caller keeps is waste: units are
                # max(ef, k), so follow the candidate count down - but only ever down. A request
                # may legitimately ask for an efSearch below the floor, and the plan must never
                # hand back more work than was asked for.
                current_ef = min(current_ef, max(_MIN_EF_SEARCH, current_k))
            else:
                current_ef = max(_MIN_EF_SEARCH, current_ef // 2)
            predicted = self._predict(
                "retrieval", self._controller.retrieval_units(current_k, current_ef)
            )
        plan = RetrievalPlan(
            candidate_k=current_k,
            ef_search=current_ef,
            requested_candidate_k=candidate_k,
            requested_ef_search=ef_search,
            predicted_seconds=predicted,
        )
        if plan.lowered:
            self._degrade(DEGRADE_LOWER_EF)
        return plan

    def plan_ranking(self, candidates: int, *, minimum: int = _MIN_RERANK_DEPTH) -> RankingPlan:
        """Decide how many of the ``candidates`` stage 2 may score in the time that is left.

        The depth is reduced towards ``minimum`` while that keeps the prediction inside the
        allowance; if even ``minimum`` rows do not fit, stage 2 is skipped altogether.
        """
        if candidates < 0:
            raise ValueError("candidates must be non-negative")
        remaining = self.remaining_seconds
        allowance = min(self._controller.ranking_seconds, remaining)
        if candidates == 0:
            return RankingPlan(
                rerank=False,
                candidates=0,
                requested_candidates=0,
                predicted_seconds=0.0,
                remaining_seconds=remaining,
            )
        affordable = int(min(float(candidates), max(self._affordable("ranking", allowance), 0.0)))
        if affordable >= candidates:
            return RankingPlan(
                rerank=True,
                candidates=candidates,
                requested_candidates=candidates,
                predicted_seconds=self._predict("ranking", candidates),
                remaining_seconds=remaining,
            )
        floor = min(minimum, candidates)
        if affordable >= floor:
            self._degrade(DEGRADE_LOWER_DEPTH)
            return RankingPlan(
                rerank=True,
                candidates=affordable,
                requested_candidates=candidates,
                predicted_seconds=self._predict("ranking", affordable),
                remaining_seconds=remaining,
            )
        self._degrade(DEGRADE_SKIP_RERANK)
        return RankingPlan(
            rerank=False,
            candidates=0,
            requested_candidates=candidates,
            predicted_seconds=self._predict("ranking", candidates),
            remaining_seconds=remaining,
        )


class BudgetController:
    """Holds the budgets and the per-stage cost estimators shared by all requests."""

    def __init__(
        self,
        *,
        total_seconds: float,
        retrieval_seconds: float,
        ranking_seconds: float,
        estimators: Mapping[str, CostEstimator] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if min(total_seconds, retrieval_seconds, ranking_seconds) <= 0.0:
            raise ValueError("budgets must be positive")
        if retrieval_seconds + ranking_seconds > total_seconds:
            raise ValueError("stage budgets must fit inside the total budget")
        self.total_seconds = total_seconds
        self.retrieval_seconds = retrieval_seconds
        self.ranking_seconds = ranking_seconds
        self._clock = clock
        # Seeds measured on the reference CPU (see bench/bench_load.py). The per-unit rates only
        # matter for the first few requests - observe() replaces them with what this box does - but
        # the fixed terms stay as they are, because serving never reports a zero-unit sample (see
        # the module docstring).
        defaults = {
            "retrieval": CostEstimator(fixed_seconds=2e-4, seconds_per_unit=4e-7),
            "ranking": CostEstimator(fixed_seconds=3e-4, seconds_per_unit=6e-7),
            "rules": CostEstimator(fixed_seconds=1e-4, seconds_per_unit=2e-7),
        }
        self._estimators: dict[str, CostEstimator] = {**defaults, **dict(estimators or {})}

    def estimator(self, stage: str) -> CostEstimator:
        if stage not in self._estimators:
            raise KeyError(f"unknown stage {stage!r}")
        return self._estimators[stage]

    @staticmethod
    def retrieval_units(candidate_k: int, ef_search: int) -> float:
        """Work unit for HNSW: the graph visits about ``max(efSearch, k)`` nodes per query."""
        return float(max(ef_search, candidate_k))

    def start(self, spent_seconds: float = 0.0, contention: float = 1.0) -> RequestBudget:
        """Begin a budget already ``spent_seconds`` in, with predictions scaled by contention."""
        return RequestBudget(self, self._clock, spent_seconds, contention)

    @staticmethod
    def contention(in_flight: int, slots: int) -> float:
        """``max(1, in_flight / slots)``: the slowdown a request should expect when sharing."""
        if slots < 1:
            raise ValueError("slots must be >= 1")
        return max(1.0, float(max(in_flight, 1)) / float(slots))

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {name: estimator.snapshot() for name, estimator in self._estimators.items()}
