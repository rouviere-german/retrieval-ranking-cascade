"""Per-stage latency budgets, the cost model and the degradation ladder."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval_ranking_cascade.serving.budget import (
    DEGRADE_LOWER_DEPTH,
    DEGRADE_LOWER_EF,
    DEGRADE_SKIP_RERANK,
    BudgetController,
    CostEstimator,
    StageTimings,
)

pytestmark = pytest.mark.unit


class FakeClock:
    """A clock the test advances by hand, so no assertion depends on real wall time."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _controller(
    clock: Callable[[], float] | None = None,
    *,
    total: float = 0.050,
    retrieval: float = 0.020,
    ranking: float = 0.020,
    estimators: dict[str, CostEstimator] | None = None,
) -> BudgetController:
    return BudgetController(
        total_seconds=total,
        retrieval_seconds=retrieval,
        ranking_seconds=ranking,
        estimators=estimators,
        clock=clock or FakeClock(),
    )


# --- cost estimator ---


def test_the_estimator_predicts_fixed_plus_rate_times_units_with_a_safety_margin() -> None:
    estimator = CostEstimator(fixed_seconds=0.001, seconds_per_unit=1e-5, safety_factor=1.5)
    assert estimator.predict(0) == pytest.approx(1.5 * 0.001)
    assert estimator.predict(100) == pytest.approx(1.5 * (0.001 + 1e-3))
    assert estimator.fixed_seconds == pytest.approx(0.001)
    assert estimator.seconds_per_unit == pytest.approx(1e-5)
    assert estimator.samples == 0


def test_max_units_is_the_inverse_of_predict() -> None:
    estimator = CostEstimator(fixed_seconds=0.002, seconds_per_unit=2e-5, safety_factor=1.25)
    units = estimator.max_units(0.010)
    assert estimator.predict(units) == pytest.approx(0.010)
    # Nothing fits once even the fixed cost is over budget.
    assert estimator.max_units(0.001) == 0.0


def test_max_units_is_unbounded_when_work_is_free_per_unit() -> None:
    estimator = CostEstimator(fixed_seconds=0.001, seconds_per_unit=0.0)
    assert estimator.max_units(0.010) == math.inf
    assert estimator.max_units(0.0) == 0.0


def test_observing_a_zero_unit_sample_moves_the_fixed_term_only() -> None:
    estimator = CostEstimator(fixed_seconds=0.001, seconds_per_unit=1e-5, alpha=0.5)
    estimator.observe(0.0, 0.003)
    assert estimator.fixed_seconds == pytest.approx(0.002)
    assert estimator.seconds_per_unit == pytest.approx(1e-5)
    assert estimator.samples == 1


def test_observing_a_loaded_sample_moves_the_rate_only() -> None:
    estimator = CostEstimator(fixed_seconds=0.001, seconds_per_unit=1e-5, alpha=1.0)
    estimator.observe(100.0, 0.011)
    assert estimator.fixed_seconds == pytest.approx(0.001)
    # (0.011 - 0.001) / 100 = 1e-4
    assert estimator.seconds_per_unit == pytest.approx(1e-4)


def test_serving_traffic_never_refits_the_fixed_term() -> None:
    """The documented shape of the model: one fitted parameter, one measured constant.

    Every serving call site reports at least one unit of work (``retrieval_units`` floors at
    ``max(ef, k) >= 16``, ranking at one row), so no live sample reaches the ``units <= 0`` branch.
    Whatever the fixed seed gets wrong is absorbed by the per-unit rate.
    """
    estimator = CostEstimator(fixed_seconds=3e-4, seconds_per_unit=6e-7, alpha=0.5)
    for _ in range(500):
        estimator.observe(150.0, 0.020)
    assert estimator.fixed_seconds == 3e-4
    assert estimator.seconds_per_unit == pytest.approx((0.020 - 3e-4) / 150.0, rel=1e-6)
    assert estimator.samples == 500


def test_a_sample_faster_than_the_fixed_cost_drives_the_rate_to_zero_not_negative() -> None:
    estimator = CostEstimator(fixed_seconds=0.005, seconds_per_unit=1e-4, alpha=1.0)
    estimator.observe(50.0, 0.001)
    assert estimator.seconds_per_unit == 0.0


def test_the_estimator_converges_on_a_stable_cost() -> None:
    estimator = CostEstimator(fixed_seconds=0.010, seconds_per_unit=1e-3, alpha=0.3)
    for _ in range(80):
        estimator.observe(0.0, 0.002)
        estimator.observe(500.0, 0.002 + 500 * 2e-6)
    assert estimator.fixed_seconds == pytest.approx(0.002, rel=0.05)
    assert estimator.seconds_per_unit == pytest.approx(2e-6, rel=0.1)


@pytest.mark.parametrize(
    "sample", [(-1.0, 0.001), (1.0, -0.001), (1.0, math.inf), (math.nan, 0.001)]
)
def test_nonsense_samples_are_ignored(sample: tuple[float, float]) -> None:
    estimator = CostEstimator(fixed_seconds=0.001, seconds_per_unit=1e-5)
    estimator.observe(*sample)
    assert estimator.samples == 0
    assert estimator.fixed_seconds == pytest.approx(0.001)


def test_the_estimator_refuses_invalid_construction_and_negative_predictions() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CostEstimator(fixed_seconds=-1.0, seconds_per_unit=0.0)
    with pytest.raises(ValueError, match="alpha"):
        CostEstimator(fixed_seconds=0.0, seconds_per_unit=0.0, alpha=0.0)
    with pytest.raises(ValueError, match="safety_factor"):
        CostEstimator(fixed_seconds=0.0, seconds_per_unit=0.0, safety_factor=0.9)
    with pytest.raises(ValueError, match="units must be non-negative"):
        CostEstimator(fixed_seconds=0.0, seconds_per_unit=0.0).predict(-1.0)


def test_the_snapshot_reports_the_whole_model() -> None:
    estimator = CostEstimator(fixed_seconds=0.001, seconds_per_unit=2e-6, safety_factor=1.4)
    estimator.observe(10.0, 0.002)
    snapshot = estimator.snapshot()
    assert snapshot["safety_factor"] == pytest.approx(1.4)
    assert snapshot["samples"] == 1.0
    assert set(snapshot) == {"fixed_seconds", "seconds_per_unit", "safety_factor", "samples"}


@settings(max_examples=100, deadline=None)
@given(
    fixed=st.floats(min_value=0.0, max_value=0.05),
    rate=st.floats(min_value=1e-9, max_value=1e-3),
    safety=st.floats(min_value=1.0, max_value=3.0),
    seconds=st.floats(min_value=0.0, max_value=1.0),
)
def test_max_units_never_predicts_over_the_allowance(
    fixed: float, rate: float, safety: float, seconds: float
) -> None:
    """The invariant the whole ladder rests on: what ``max_units`` allows must fit."""
    estimator = CostEstimator(fixed_seconds=fixed, seconds_per_unit=rate, safety_factor=safety)
    units = estimator.max_units(seconds)
    if units > 0.0:
        assert estimator.predict(units) <= seconds + 1e-12


# --- controller construction ---


def test_the_controller_refuses_budgets_that_do_not_fit() -> None:
    with pytest.raises(ValueError, match="budgets must be positive"):
        _controller(total=0.0)
    with pytest.raises(ValueError, match="must fit inside"):
        _controller(total=0.010, retrieval=0.008, ranking=0.008)


def test_unknown_stages_are_a_key_error_and_known_ones_are_not() -> None:
    controller = _controller()
    for stage in ("retrieval", "ranking", "rules"):
        assert controller.estimator(stage) is controller.estimator(stage)
    with pytest.raises(KeyError, match="unknown stage"):
        controller.estimator("nope")


def test_supplied_estimators_override_the_defaults() -> None:
    mine = CostEstimator(fixed_seconds=0.9, seconds_per_unit=0.0)
    controller = _controller(estimators={"retrieval": mine})
    assert controller.estimator("retrieval") is mine
    assert controller.estimator("ranking") is not mine
    assert set(controller.snapshot()) == {"retrieval", "ranking", "rules"}


def test_retrieval_units_are_the_visited_node_count() -> None:
    assert BudgetController.retrieval_units(500, 768) == 768.0
    assert BudgetController.retrieval_units(500, 100) == 500.0


def test_contention_is_the_ratio_of_in_flight_requests_to_slots() -> None:
    assert BudgetController.contention(1, 8) == pytest.approx(1.0)
    assert BudgetController.contention(4, 8) == pytest.approx(1.0)
    assert BudgetController.contention(16, 8) == pytest.approx(2.0)
    assert BudgetController.contention(0, 8) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="slots"):
        BudgetController.contention(1, 0)


# --- timings ---


def test_stage_timings_accumulate_per_stage() -> None:
    timings = StageTimings()
    timings.record("retrieval", 0.001)
    timings.record("retrieval", 0.002)
    timings.record("ranking", 0.004)
    assert timings.as_dict() == {"retrieval": pytest.approx(0.003), "ranking": pytest.approx(0.004)}
    assert timings.total == pytest.approx(0.007)


def test_a_stage_context_manager_times_the_body_and_feeds_the_estimator() -> None:
    clock = FakeClock()
    controller = _controller(clock)
    budget = controller.start()
    with budget.stage("ranking", 150.0):
        clock.advance(0.004)
    assert budget.timings.as_dict()["ranking"] == pytest.approx(0.004)
    assert controller.estimator("ranking").samples == 1


def test_a_stage_that_raises_is_still_measured() -> None:
    clock = FakeClock()
    controller = _controller(clock)
    budget = controller.start()

    def boom() -> None:
        with budget.stage("rules", 20.0):
            clock.advance(0.002)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        boom()
    assert budget.timings.as_dict()["rules"] == pytest.approx(0.002)


# --- request budget bookkeeping ---


def test_queue_time_is_already_spent_when_the_budget_starts() -> None:
    clock = FakeClock()
    controller = _controller(clock, total=0.050)
    budget = controller.start(spent_seconds=0.030)
    assert budget.elapsed_seconds == pytest.approx(0.030)
    assert budget.remaining_seconds == pytest.approx(0.020)
    clock.advance(0.005)
    assert budget.remaining_seconds == pytest.approx(0.015)


def test_a_budget_refuses_negative_queue_time_and_sub_unit_contention() -> None:
    controller = _controller()
    with pytest.raises(ValueError, match="spent_seconds"):
        controller.start(spent_seconds=-0.001)
    with pytest.raises(ValueError, match="contention"):
        controller.start(contention=0.5)
    with pytest.raises(ValueError, match="contention"):
        controller.start(contention=math.inf)


def test_a_fresh_budget_is_not_degraded() -> None:
    budget = _controller().start()
    assert not budget.degraded
    assert budget.degraded_reasons == []


# --- the degradation ladder ---


def _cheap() -> dict[str, CostEstimator]:
    """Costs low enough that a full plan always fits, so only the tested knob matters."""
    return {
        "retrieval": CostEstimator(fixed_seconds=1e-6, seconds_per_unit=1e-9, safety_factor=1.0),
        "ranking": CostEstimator(fixed_seconds=1e-6, seconds_per_unit=1e-9, safety_factor=1.0),
    }


def test_a_comfortable_budget_plans_the_full_request() -> None:
    controller = _controller(estimators=_cheap())
    budget = controller.start()
    plan = budget.plan_retrieval(500, 768)
    assert (plan.candidate_k, plan.ef_search) == (500, 768)
    assert not plan.lowered
    ranking = budget.plan_ranking(150)
    assert ranking.rerank
    assert ranking.candidates == 150
    assert not ranking.lowered
    assert not budget.degraded


def test_a_tight_retrieval_budget_lowers_ef_search_first() -> None:
    """Step 1 of the ladder: recall drops a little, the reranker still runs."""
    estimators = {
        "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-5, safety_factor=1.0),
        **{k: v for k, v in _cheap().items() if k == "ranking"},
    }
    controller = _controller(retrieval=0.006, estimators=estimators)
    budget = controller.start()
    plan = budget.plan_retrieval(500, 2048)
    assert plan.ef_search < 2048
    assert plan.ef_search >= 16
    assert plan.candidate_k == 500
    assert plan.predicted_seconds <= 0.006
    assert plan.lowered
    assert budget.degraded_reasons == [DEGRADE_LOWER_EF]


def test_an_extremely_tight_retrieval_budget_also_lowers_the_candidate_count() -> None:
    estimators = {
        "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-4, safety_factor=1.0),
        "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9, safety_factor=1.0),
    }
    controller = _controller(total=0.050, retrieval=0.003, estimators=estimators)
    plan = controller.start().plan_retrieval(500, 768)
    assert plan.candidate_k < 500
    assert plan.candidate_k >= 16
    assert plan.ef_search >= 16


def test_lowering_the_candidate_count_also_lowers_a_now_oversized_ef_search() -> None:
    """A graph walk that visits more nodes than the caller will keep is pure waste.

    ``retrieval_units`` is ``max(ef, k)``, so once the ladder has cut ``candidate_k`` an
    ``efSearch`` above it costs time and buys nothing: the extra neighbours are found, then dropped.
    """
    estimators = {
        "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-4, safety_factor=1.0),
        "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9, safety_factor=1.0),
    }
    controller = _controller(total=0.050, retrieval=0.003, estimators=estimators)
    plan = controller.start().plan_retrieval(500, 768)
    assert plan.candidate_k < 500
    assert plan.ef_search <= plan.candidate_k


def test_the_ladder_stops_at_the_floors_even_when_nothing_fits() -> None:
    """A budget nothing can satisfy must terminate at the minimum, not loop forever."""
    estimators = {
        "retrieval": CostEstimator(fixed_seconds=1.0, seconds_per_unit=1.0, safety_factor=1.0),
        "ranking": CostEstimator(fixed_seconds=1.0, seconds_per_unit=1.0, safety_factor=1.0),
    }
    controller = _controller(estimators=estimators)
    plan = controller.start().plan_retrieval(500, 768)
    assert (plan.candidate_k, plan.ef_search) == (16, 16)
    assert plan.predicted_seconds > controller.retrieval_seconds


def test_a_tight_ranking_budget_lowers_the_rerank_depth() -> None:
    """Step 2: score the best-scoring prefix only; ``k`` is far below the depth, so it fits."""
    estimators = {
        "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9, safety_factor=1.0),
        "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-5, safety_factor=1.0),
    }
    controller = _controller(ranking=0.001, estimators=estimators)
    budget = controller.start()
    ranking = budget.plan_ranking(500)
    assert ranking.rerank
    assert ranking.candidates == pytest.approx(100, abs=1)
    assert ranking.lowered
    assert budget.degraded_reasons == [DEGRADE_LOWER_DEPTH]


def test_a_budget_that_cannot_fit_the_minimum_depth_skips_reranking_altogether() -> None:
    """Step 3: return ANN order, flagged. Precision drops a lot; the response is still relevant."""
    estimators = {
        "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9, safety_factor=1.0),
        "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-3, safety_factor=1.0),
    }
    controller = _controller(ranking=0.002, estimators=estimators)
    budget = controller.start()
    ranking = budget.plan_ranking(500)
    assert not ranking.rerank
    assert ranking.candidates == 0
    assert ranking.requested_candidates == 500
    assert not ranking.lowered
    assert budget.degraded
    assert budget.degraded_reasons == [DEGRADE_SKIP_RERANK]


def test_time_already_spent_shrinks_the_ranking_allowance() -> None:
    clock = FakeClock()
    estimators = {
        "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9, safety_factor=1.0),
        "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-5, safety_factor=1.0),
    }
    controller = _controller(clock, total=0.050, ranking=0.020, estimators=estimators)
    relaxed = controller.start().plan_ranking(500)
    assert relaxed.candidates == 500
    late = controller.start(spent_seconds=0.048)
    tight = late.plan_ranking(500)
    assert tight.candidates < 500
    assert tight.remaining_seconds == pytest.approx(0.002)


def test_a_minimum_lower_than_the_candidate_count_is_honoured() -> None:
    estimators = {
        "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9, safety_factor=1.0),
        "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-4, safety_factor=1.0),
    }
    controller = _controller(ranking=0.001, estimators=estimators)
    plan = controller.start().plan_ranking(500, minimum=5)
    assert plan.rerank
    assert plan.candidates == pytest.approx(10, abs=1)


def test_planning_zero_candidates_skips_the_stage_without_flagging_degradation() -> None:
    """An empty candidate set is not a quality loss; a heavy user simply has nothing left."""
    budget = _controller(estimators=_cheap()).start()
    plan = budget.plan_ranking(0)
    assert not plan.rerank
    assert plan.candidates == 0
    assert plan.requested_candidates == 0
    assert not budget.degraded


def test_plan_arguments_are_validated() -> None:
    budget = _controller().start()
    with pytest.raises(ValueError, match="must be >= 1"):
        budget.plan_retrieval(0, 100)
    with pytest.raises(ValueError, match="must be >= 1"):
        budget.plan_retrieval(100, 0)
    with pytest.raises(ValueError, match="non-negative"):
        budget.plan_ranking(-1)


def test_a_reason_is_never_recorded_twice() -> None:
    estimators = {
        "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-4, safety_factor=1.0),
        "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9, safety_factor=1.0),
    }
    budget = _controller(retrieval=0.002, estimators=estimators).start()
    budget.plan_retrieval(500, 2048)
    budget.plan_retrieval(500, 2048)
    assert budget.degraded_reasons == [DEGRADE_LOWER_EF]


# --- contention ---


def test_contention_makes_the_expensive_stage_look_as_expensive_as_it_is() -> None:
    """Queue time alone misses the backlog outside the process; the in-flight count does not."""
    estimators = {
        "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9, safety_factor=1.0),
        "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-5, safety_factor=1.0),
    }
    controller = _controller(ranking=0.010, estimators=estimators)
    solo = controller.start(contention=1.0).plan_ranking(500)
    assert solo.candidates == 500
    assert not solo.lowered
    crowded = controller.start(contention=4.0)
    plan = crowded.plan_ranking(500)
    # Solo the allowance pays for 1000 rows, twice the requested depth. At contention 4 a quarter
    # of it is left per request, which is 250 rows: half the request, so the depth is lowered.
    assert plan.candidates == pytest.approx(250, abs=1)
    assert plan.lowered
    assert crowded.degraded_reasons == [DEGRADE_LOWER_DEPTH]


def test_contention_scales_the_retrieval_prediction_too() -> None:
    estimators = {
        "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-5, safety_factor=1.0),
        "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9, safety_factor=1.0),
    }
    controller = _controller(retrieval=0.020, estimators=estimators)
    solo = controller.start().plan_retrieval(500, 1024)
    crowded = controller.start(contention=8.0).plan_retrieval(500, 1024)
    assert solo.ef_search == 1024
    assert crowded.ef_search < 1024


@settings(max_examples=80, deadline=None)
@given(
    candidate_k=st.integers(min_value=1, max_value=4_000),
    ef_search=st.integers(min_value=1, max_value=4_096),
    rate=st.floats(min_value=1e-9, max_value=1e-3),
    allowance=st.floats(min_value=1e-5, max_value=0.05),
    contention=st.floats(min_value=1.0, max_value=16.0),
)
def test_a_retrieval_plan_never_grows_the_request_and_respects_the_floors(
    candidate_k: int, ef_search: int, rate: float, allowance: float, contention: float
) -> None:
    controller = _controller(
        total=0.1,
        retrieval=allowance,
        ranking=0.001,
        estimators={
            "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=rate),
            "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=rate),
        },
    )
    plan = controller.start(contention=contention).plan_retrieval(candidate_k, ef_search)
    assert plan.candidate_k <= candidate_k
    assert plan.ef_search <= ef_search
    assert plan.candidate_k >= min(16, candidate_k)
    assert plan.ef_search >= min(16, ef_search)
    if plan.candidate_k < candidate_k:
        # Once the candidate count has been cut, efSearch above it is work nobody reads.
        assert plan.ef_search <= max(plan.candidate_k, 16)
    assert plan.lowered == (plan.ef_search < ef_search or plan.candidate_k < candidate_k)


@settings(max_examples=80, deadline=None)
@given(
    candidates=st.integers(min_value=0, max_value=4_000),
    rate=st.floats(min_value=1e-9, max_value=1e-2),
    allowance=st.floats(min_value=1e-6, max_value=0.05),
    contention=st.floats(min_value=1.0, max_value=16.0),
)
def test_a_ranking_plan_is_a_prefix_and_its_prediction_fits_when_it_reranks(
    candidates: int, rate: float, allowance: float, contention: float
) -> None:
    controller = _controller(
        total=0.1,
        retrieval=0.001,
        ranking=allowance,
        estimators={
            "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=rate),
            "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=rate),
        },
    )
    budget = controller.start(contention=contention)
    plan = budget.plan_ranking(candidates)
    assert 0 <= plan.candidates <= candidates
    assert plan.requested_candidates == candidates
    if plan.rerank and plan.candidates < candidates:
        # ``predicted_seconds`` already carries the contention factor, and a lowered depth must
        # actually fit in what the allowance and the time left allow.
        assert plan.predicted_seconds <= min(allowance, plan.remaining_seconds) + 1e-9
    if candidates > 0:
        assert budget.degraded == (plan.candidates != candidates)
