"""The offline feedback simulator that closes the serve-observe-CTR loop."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from retrieval_ranking_cascade.data.generator import generate, world_config
from retrieval_ranking_cascade.evaluation.simulator import FeedbackSimulator, SimulationResult

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def world() -> Any:
    config = world_config("tiny", seed=11, n_users=60, n_items=120, pool_size=20, slots=5)
    generated, _, _ = generate(config)
    return generated


# --- result shape ---


def test_a_zero_impression_result_reports_a_zero_ctr_rather_than_dividing_by_zero() -> None:
    result = SimulationResult(0, 0, 0, 0.0, ())
    assert result.ctr == 0.0
    assert result.as_dict()["impressions"] == 0


def test_the_result_reports_ctr_and_rounds_for_json(world: Any) -> None:
    simulator = FeedbackSimulator(world, seed=1)
    result = simulator.simulate([(user, list(range(5))) for user in range(40)])
    assert result.users == 40
    assert result.impressions == 200
    assert 0 <= result.clicks <= result.impressions
    assert result.ctr == pytest.approx(result.clicks / result.impressions)
    payload = result.as_dict()
    assert payload["users"] == 40
    assert len(payload["position_ctr"]) == 5  # type: ignore[arg-type]
    assert 0.0 <= float(payload["mean_relevance"]) <= 1.0  # type: ignore[arg-type]


def test_simulating_nothing_returns_zeroes(world: Any) -> None:
    assert FeedbackSimulator(world, seed=1).simulate([]).impressions == 0


def test_an_empty_list_contributes_no_impressions(world: Any) -> None:
    simulator = FeedbackSimulator(world, seed=1)
    result = simulator.simulate([(0, []), (1, [3, 4])])
    assert result.users == 2
    assert result.impressions == 2


def test_simulate_list_returns_one_flag_per_position(world: Any) -> None:
    simulator = FeedbackSimulator(world, seed=2)
    outcome = simulator.simulate_list(0, [1, 2, 3, 4])
    assert outcome.shape == (4,)
    assert set(outcome.tolist()) <= {0, 1}
    assert simulator.simulate_list(0, []).shape == (0,)


# --- the click model it applies ---


def test_position_ctr_decays_down_the_page(world: Any) -> None:
    """The same items in the same order: only the examination curve can produce the decay."""
    simulator = FeedbackSimulator(world, seed=3)
    items = list(range(8))
    result = simulator.simulate([(user % world.config.n_users, items) for user in range(2_000)])
    head = float(np.mean(result.position_ctr[:2]))
    tail = float(np.mean(result.position_ctr[-2:]))
    assert head > tail
    # With eta = 1 the ratio between rank 1 and rank 8 is about 8.
    assert result.position_ctr[0] > 3.0 * result.position_ctr[-1]


def test_a_more_relevant_list_earns_more_clicks(world: Any) -> None:
    """A better ranker genuinely gains here; this is not a replay of logged clicks."""
    users = list(range(world.config.n_users))
    relevance = world.relevance_rows(np.asarray(users))
    best = [(user, np.argsort(-relevance[index])[:6].tolist()) for index, user in enumerate(users)]
    worst = [(user, np.argsort(relevance[index])[:6].tolist()) for index, user in enumerate(users)]
    simulator = FeedbackSimulator(world, seed=4)
    good = simulator.simulate(best)
    bad = simulator.simulate(worst)
    assert good.mean_relevance > bad.mean_relevance
    assert good.ctr > bad.ctr


def test_clicks_are_reproducible_for_a_seed_and_differ_across_seeds(world: Any) -> None:
    served = [(user, [1, 2, 3]) for user in range(30)]
    first = FeedbackSimulator(world, seed=5).simulate(served)
    again = FeedbackSimulator(world, seed=5).simulate(served)
    other = FeedbackSimulator(world, seed=6).simulate(served)
    assert first.clicks == again.clicks
    assert first.position_ctr == again.position_ctr
    assert (first.clicks, other.clicks) != (0, 0)


def test_the_simulated_ctr_stays_below_the_mean_relevance(world: Any) -> None:
    """``P(click) = P(examine) * rel`` and examination is at most 1, so CTR cannot exceed it."""
    simulator = FeedbackSimulator(world, seed=7)
    result = simulator.simulate([(user, list(range(6))) for user in range(world.config.n_users)])
    assert result.ctr <= result.mean_relevance + 1e-9


def test_lists_of_different_lengths_are_aggregated_by_position(world: Any) -> None:
    simulator = FeedbackSimulator(world, seed=8)
    result = simulator.simulate([(0, [1]), (1, [1, 2, 3])])
    assert len(result.position_ctr) == 3
    assert result.impressions == 4
