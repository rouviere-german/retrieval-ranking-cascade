"""Propensity estimation from the randomized slice, and the IPS/SNIPS click estimators."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval_ranking_cascade.data.generator import Interactions
from retrieval_ranking_cascade.evaluation.propensity import (
    PropensityModel,
    estimate_eta,
    estimate_propensities,
    ips_estimate,
    position_click_rates,
    snips_estimate,
)

pytestmark = pytest.mark.unit


def _interactions(
    positions: npt.NDArray[np.int64],
    clicked: npt.NDArray[np.int64],
    randomized: npt.NDArray[np.bool_],
) -> Interactions:
    n = positions.shape[0]
    return Interactions(
        session_id=np.arange(n, dtype=np.int32) // 10,
        user_id=np.zeros(n, dtype=np.int32),
        item_id=np.arange(n, dtype=np.int32),
        position=positions.astype(np.int8),
        clicked=clicked.astype(np.int8),
        randomized=randomized,
        day=np.zeros(n, dtype=np.int16),
        period=np.zeros(n, dtype=np.int8),
    )


def _randomized_log(
    *,
    slots: int,
    per_rank: int,
    eta: float,
    base_relevance: float = 0.4,
    seed: int = 0,
    randomized: bool = True,
) -> Interactions:
    """A log in which rank is independent of relevance, so ``CTR(r)`` measures ``p_r * rel``."""
    rng = np.random.default_rng(seed)
    positions = np.repeat(np.arange(1, slots + 1, dtype=np.int64), per_rank)
    examine = positions.astype(np.float64) ** -eta
    clicked = (rng.random(positions.shape[0]) < examine * base_relevance).astype(np.int64)
    flags = np.full(positions.shape[0], randomized, dtype=np.bool_)
    return _interactions(positions, clicked, flags)


# --- per-rank rates ---


def test_position_click_rates_counts_impressions_clicks_and_rates() -> None:
    positions = np.asarray([1, 1, 2, 3, 3, 3], dtype=np.int64)
    clicked = np.asarray([1, 0, 1, 0, 0, 1], dtype=np.int64)
    rates, impressions, clicks = position_click_rates(positions, clicked, 4)
    assert impressions.tolist() == [2, 1, 3, 0]
    assert clicks.tolist() == [1, 1, 1, 0]
    np.testing.assert_allclose(rates[:3], [0.5, 1.0, 1 / 3])
    assert np.isnan(rates[3])


def test_position_click_rates_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="slots must be"):
        position_click_rates([1], [0], 0)
    with pytest.raises(ValueError, match="same shape"):
        position_click_rates([1, 2], [0], 3)
    with pytest.raises(ValueError, match=r"within 1\.\.slots"):
        position_click_rates([0], [0], 3)
    with pytest.raises(ValueError, match=r"within 1\.\.slots"):
        position_click_rates([4], [0], 3)


# --- the model itself ---


def test_propensity_model_validates_its_array() -> None:
    impressions = np.ones(3, dtype=np.int64)
    clicks = np.ones(3, dtype=np.int64)
    with pytest.raises(ValueError, match="non-empty"):
        PropensityModel(np.zeros(0), impressions, clicks, 10.0)
    with pytest.raises(ValueError, match="finite and positive"):
        PropensityModel(np.asarray([1.0, 0.0, 0.5]), impressions, clicks, 10.0)
    with pytest.raises(ValueError, match="max_weight"):
        PropensityModel(np.asarray([1.0, 0.5, 0.25]), impressions, clicks, 0.5)


def test_propensity_lookup_is_one_based_and_saturates_past_the_slate() -> None:
    model = PropensityModel(
        np.asarray([1.0, 0.5, 0.25]), np.ones(3, dtype=np.int64), np.ones(3, dtype=np.int64), 8.0
    )
    assert model.slots == 3
    np.testing.assert_allclose(model.propensity([1, 2, 3]), [1.0, 0.5, 0.25])
    # Anything shown below the last measured rank inherits that rank's propensity.
    np.testing.assert_allclose(model.propensity([4, 99]), [0.25, 0.25])
    with pytest.raises(ValueError, match="1-based"):
        model.propensity([0])


def test_weights_are_inverse_propensities_clipped_at_max_weight() -> None:
    model = PropensityModel(
        np.asarray([1.0, 0.5, 0.05]), np.ones(3, dtype=np.int64), np.ones(3, dtype=np.int64), 10.0
    )
    np.testing.assert_allclose(model.weight([1, 2, 3]), [1.0, 2.0, 10.0])


def test_as_dict_is_json_friendly() -> None:
    model = PropensityModel(
        np.asarray([1.0, 0.4]),
        np.asarray([9, 8], dtype=np.int64),
        np.asarray([3, 1], dtype=np.int64),
        5.0,
    )
    payload = model.as_dict()
    assert payload == {"propensities": [1.0, 0.4], "impressions": [9, 8], "clicks": [3, 1]}


# --- estimation from the randomized slice ---


@pytest.mark.parametrize("eta", [0.6, 1.0, 1.4])
def test_estimated_propensities_track_the_true_examination_curve(eta: float) -> None:
    log = _randomized_log(slots=10, per_rank=40_000, eta=eta, seed=1)
    model = estimate_propensities(log, slots=10, min_impressions=1_000)
    truth = np.arange(1, 11, dtype=np.float64) ** -eta
    assert model.propensities[0] == pytest.approx(1.0)
    np.testing.assert_allclose(model.propensities, truth, atol=0.06)
    assert estimate_eta(model) == pytest.approx(eta, abs=0.08)


def test_only_the_randomized_slice_is_used() -> None:
    """Non-randomized rows carry the logging policy's bias; including them would poison the fit."""
    randomized = _randomized_log(slots=6, per_rank=8_000, eta=1.0, seed=2)
    # A biased slice where every impression at rank 6 is a click and nothing else is.
    biased_positions = np.full(40_000, 6, dtype=np.int64)
    biased = _interactions(
        biased_positions,
        np.ones(40_000, dtype=np.int64),
        np.zeros(40_000, dtype=np.bool_),
    )
    combined = Interactions(
        **{
            name: np.concatenate([getattr(randomized, name), getattr(biased, name)])
            for name in randomized.arrays()
        }
    )
    model = estimate_propensities(combined, slots=6, min_impressions=500)
    assert model.propensities[5] < 0.35


def test_ranks_with_too_few_randomized_impressions_inherit_the_nearest_reliable_rank() -> None:
    log = _randomized_log(slots=8, per_rank=6_000, eta=1.0, seed=3)
    # Keep everything up to rank 4 plus a handful of rank-8 rows.
    keep = (log.position <= 4) | ((log.position == 8) & (np.arange(len(log)) % 900 == 0))
    thin = log.select(keep)
    model = estimate_propensities(thin, slots=8, min_impressions=1_000)
    # Ranks 5..8 have no reliable estimate, so they all copy rank 4's ratio.
    assert len(set(np.round(model.propensities[3:], 9).tolist())) == 1


def test_estimation_fails_loudly_when_rank_one_cannot_be_identified() -> None:
    log = _randomized_log(slots=5, per_rank=50, eta=1.0, seed=4)
    with pytest.raises(ValueError, match="identify the propensity scale"):
        estimate_propensities(log, slots=5, min_impressions=1_000)
    # No randomized sessions at all is the same failure.
    none_randomized = _randomized_log(slots=5, per_rank=5_000, eta=1.0, randomized=False, seed=5)
    with pytest.raises(ValueError, match="identify the propensity scale"):
        estimate_propensities(none_randomized, slots=5, min_impressions=100)


def test_estimated_propensities_stay_inside_the_clipping_range() -> None:
    log = _randomized_log(slots=10, per_rank=20_000, eta=2.5, seed=6)
    model = estimate_propensities(log, slots=10, min_impressions=500, max_weight=4.0)
    assert model.propensities.min() >= 0.25 - 1e-12
    assert model.propensities.max() <= 1.0


def test_estimate_eta_needs_more_than_one_rank() -> None:
    model = PropensityModel(
        np.asarray([1.0, 0.5]), np.ones(2, dtype=np.int64), np.ones(2, dtype=np.int64), 5.0
    )
    with pytest.raises(ValueError, match="at least two ranks"):
        estimate_eta(model)


def test_estimate_eta_recovers_an_exact_power_law() -> None:
    ranks = np.arange(1, 11, dtype=np.float64)
    model = PropensityModel(
        ranks**-1.3, np.ones(10, dtype=np.int64), np.ones(10, dtype=np.int64), 20.0
    )
    assert estimate_eta(model) == pytest.approx(1.3, abs=1e-9)


# --- IPS and SNIPS ---


def test_ips_recovers_the_relevance_level_that_the_naive_ctr_understates() -> None:
    relevance, eta, slots = 0.35, 1.0, 10
    log = _randomized_log(slots=slots, per_rank=30_000, eta=eta, base_relevance=relevance, seed=7)
    model = estimate_propensities(log, slots=slots, min_impressions=1_000)
    naive = float(log.clicked.mean())
    corrected = ips_estimate(log.clicked, log.position, model)
    assert naive < 0.5 * relevance
    assert corrected == pytest.approx(relevance, abs=0.05)
    assert abs(corrected - relevance) < abs(naive - relevance)


def test_snips_is_ips_divided_by_the_mean_weight() -> None:
    """The two differ by exactly the mean weight, which is the whole content of the ratio form.

    That identity is worth pinning: it says SNIPS and IPS estimate the same quantity up to a scale
    the sample itself supplies, so SNIPS cannot be read as "IPS but larger" or "IPS but smaller" -
    which way it moves depends on the weight mean, and on a slate whose propensities decay it moves
    down. What SNIPS buys is that its value is a convex combination of the click indicators, so it
    stays a probability whatever the clipping does; IPS can and does exceed 1.
    """
    log = _randomized_log(slots=10, per_rank=4_000, eta=1.6, base_relevance=0.5, seed=8)
    model = estimate_propensities(log, slots=10, min_impressions=200, max_weight=3.0)
    weights = model.weight(log.position)
    expected = float(np.dot(log.clicked, weights) / weights.sum())
    snips = snips_estimate(log.clicked, log.position, model)
    assert snips == pytest.approx(expected)
    assert snips * float(weights.mean()) == pytest.approx(
        ips_estimate(log.clicked, log.position, model)
    )
    assert 0.0 <= snips <= 1.0


def test_snips_equals_ips_when_every_position_is_examined_with_certainty() -> None:
    model = PropensityModel(np.ones(4), np.ones(4, dtype=np.int64), np.ones(4, dtype=np.int64), 5.0)
    clicked = np.asarray([1, 0, 1, 1])
    positions = np.asarray([1, 2, 3, 4])
    assert snips_estimate(clicked, positions, model) == pytest.approx(
        ips_estimate(clicked, positions, model)
    )


def test_both_estimators_return_zero_on_an_empty_input() -> None:
    model = PropensityModel(
        np.asarray([1.0, 0.5]), np.ones(2, dtype=np.int64), np.ones(2, dtype=np.int64), 5.0
    )
    empty = np.zeros(0, dtype=np.int64)
    assert ips_estimate(empty, empty, model) == 0.0
    assert snips_estimate(empty, empty, model) == 0.0


def test_snips_of_an_all_zero_weight_vector_is_zero() -> None:
    """A degenerate model whose weights vanish must not divide by zero."""
    model = PropensityModel(
        np.asarray([1.0]), np.ones(1, dtype=np.int64), np.ones(1, dtype=np.int64), 1.0
    )
    assert snips_estimate([0, 0], [1, 1], model) == 0.0


@settings(max_examples=60, deadline=None)
@given(
    slots=st.integers(min_value=2, max_value=12),
    eta=st.floats(min_value=0.1, max_value=2.0),
    max_weight=st.floats(min_value=1.0, max_value=50.0),
)
def test_weights_are_bounded_and_monotone_in_rank(
    slots: int, eta: float, max_weight: float
) -> None:
    ranks = np.arange(1, slots + 1, dtype=np.float64)
    model = PropensityModel(
        ranks**-eta,
        np.ones(slots, dtype=np.int64),
        np.ones(slots, dtype=np.int64),
        max_weight,
    )
    weights = model.weight(ranks.astype(np.int64))
    assert weights.min() >= 1.0 - 1e-12
    assert weights.max() <= max_weight + 1e-12
    assert np.all(np.diff(weights) >= -1e-12)


@settings(max_examples=60, deadline=None)
@given(
    clicks=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=40),
    seed=st.integers(min_value=0, max_value=200),
)
def test_ips_inflates_the_naive_rate_by_between_one_and_max_weight(
    clicks: list[int], seed: int
) -> None:
    """Every weight is in ``[1, max_weight]``, so the estimate is bracketed by the naive rate."""
    rng = np.random.default_rng(seed)
    slots = 8
    positions = rng.integers(1, slots + 1, size=len(clicks))
    ranks = np.arange(1, slots + 1, dtype=np.float64)
    model = PropensityModel(
        ranks**-1.0, np.ones(slots, dtype=np.int64), np.ones(slots, dtype=np.int64), float(slots)
    )
    naive = float(np.mean(clicks))
    estimate = ips_estimate(clicks, positions, model)
    assert naive - 1e-12 <= estimate <= naive * slots + 1e-12


@settings(max_examples=60, deadline=None)
@given(
    clicks=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=40),
    seed=st.integers(min_value=0, max_value=200),
)
def test_snips_stays_a_probability_because_it_is_a_convex_combination(
    clicks: list[int], seed: int
) -> None:
    rng = np.random.default_rng(seed)
    slots = 8
    positions = rng.integers(1, slots + 1, size=len(clicks))
    ranks = np.arange(1, slots + 1, dtype=np.float64)
    model = PropensityModel(
        ranks**-1.5, np.ones(slots, dtype=np.int64), np.ones(slots, dtype=np.int64), 20.0
    )
    estimate = snips_estimate(clicks, positions, model)
    assert min(clicks) - 1e-12 <= estimate <= max(clicks) + 1e-12
