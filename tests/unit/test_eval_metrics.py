"""Ranking metrics: DCG, NDCG, MRR, set recall, catalog coverage."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval_ranking_cascade.evaluation.metrics import (
    catalog_coverage,
    dcg,
    discounts,
    mean_reciprocal_rank,
    ndcg,
    ndcg_at_k,
    recall_at_k_sets,
)

pytestmark = pytest.mark.unit

_GAINS = st.lists(
    st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=20,
)


# --- discounts ---


def test_discounts_are_the_standard_log2_positions() -> None:
    np.testing.assert_allclose(discounts(3), [1.0, 1.0 / np.log2(3.0), 0.5])
    assert discounts(0).shape == (0,)
    with pytest.raises(ValueError, match="non-negative"):
        discounts(-1)


# --- DCG ---


def test_dcg_is_the_discounted_sum_and_truncates_at_k() -> None:
    gains = [3.0, 2.0, 1.0]
    assert dcg(gains) == pytest.approx(3.0 + 2.0 / np.log2(3.0) + 0.5)
    assert dcg(gains, 2) == pytest.approx(3.0 + 2.0 / np.log2(3.0))
    assert dcg(gains, 99) == pytest.approx(dcg(gains))


def test_exponential_gain_mode_uses_two_to_the_power_minus_one() -> None:
    assert dcg([1.0, 1.0], mode="exponential") == pytest.approx(1.0 + 1.0 / np.log2(3.0))
    assert dcg([2.0], mode="exponential") == pytest.approx(3.0)


def test_dcg_refuses_negative_or_non_finite_gains() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        dcg([1.0, -0.5])
    with pytest.raises(ValueError, match="finite"):
        dcg([1.0, np.nan])


@settings(max_examples=80, deadline=None)
@given(gains=_GAINS)
def test_dcg_is_maximized_by_the_descending_order(gains: list[float]) -> None:
    best = dcg(sorted(gains, reverse=True))
    assert dcg(gains) <= best + 1e-9


# --- NDCG ---


def test_ndcg_of_a_perfectly_sorted_list_is_one_and_of_a_reversed_one_is_lower() -> None:
    gains = [3.0, 2.0, 1.0]
    assert ndcg(gains) == pytest.approx(1.0)
    assert ndcg(list(reversed(gains))) < 1.0
    assert ndcg_at_k(gains, 2) == pytest.approx(1.0)


def test_ndcg_of_an_all_zero_list_is_zero_rather_than_undefined() -> None:
    assert ndcg([0.0, 0.0, 0.0]) == 0.0
    assert ndcg([1.0, 0.0], ideal_gains=[0.0, 0.0]) == 0.0


def test_ideal_gains_normalize_against_a_larger_pool() -> None:
    """Full-corpus NDCG divides by the best k items of the catalogue, not of the served list."""
    served = [0.4, 0.3]
    assert ndcg(served, 2) == pytest.approx(1.0)
    assert ndcg(served, 2, ideal_gains=[0.9, 0.8]) < 0.6


def test_ndcg_is_invariant_to_a_positive_rescaling_of_linear_gains() -> None:
    gains = [0.5, 0.2, 0.9, 0.1]
    assert ndcg(gains, 3) == pytest.approx(ndcg([7.0 * g for g in gains], 3))


@settings(max_examples=80, deadline=None)
@given(gains=_GAINS, k=st.integers(min_value=1, max_value=20))
def test_ndcg_is_in_the_unit_interval(gains: list[float], k: int) -> None:
    value = ndcg(gains, k)
    assert 0.0 <= value <= 1.0 + 1e-9


@settings(max_examples=60, deadline=None)
@given(gains=_GAINS)
def test_sorting_a_list_never_lowers_its_ndcg(gains: list[float]) -> None:
    assert ndcg(sorted(gains, reverse=True)) >= ndcg(gains) - 1e-9


@settings(max_examples=60, deadline=None)
@given(
    gains=st.lists(
        st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=12,
    ),
    seed=st.integers(min_value=0, max_value=200),
)
def test_swapping_an_adjacent_pair_into_descending_order_cannot_hurt(
    gains: list[float], seed: int
) -> None:
    """The pairwise-swap monotonicity LambdaMART's gradients are built on."""
    rng = np.random.default_rng(seed)
    index = int(rng.integers(0, len(gains) - 1))
    swapped = list(gains)
    low, high = sorted(swapped[index : index + 2], reverse=True)
    swapped[index], swapped[index + 1] = low, high
    assert ndcg(swapped) >= ndcg(gains) - 1e-9


# --- MRR ---


def test_mrr_is_the_reciprocal_of_the_first_hit() -> None:
    assert mean_reciprocal_rank([False, True, True]) == pytest.approx(0.5)
    assert mean_reciprocal_rank([True]) == pytest.approx(1.0)
    assert mean_reciprocal_rank([False, False]) == 0.0
    # A hit outside the cut-off does not count.
    assert mean_reciprocal_rank([False, False, True], 2) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


# --- set recall and coverage ---


def test_recall_at_k_sets_handles_duplicates_and_an_empty_relevant_set() -> None:
    assert recall_at_k_sets([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / 3)
    assert recall_at_k_sets([], [1]) == 0.0
    assert recall_at_k_sets([1, 2], []) == 1.0


def test_catalog_coverage_counts_distinct_items_and_ignores_padding() -> None:
    assert catalog_coverage([1, 1, 2, -1], 10) == pytest.approx(0.2)
    assert catalog_coverage([], 10) == 0.0
    with pytest.raises(ValueError, match="n_items"):
        catalog_coverage([1], 0)


@settings(max_examples=60, deadline=None)
@given(
    items=st.lists(st.integers(min_value=0, max_value=19), min_size=0, max_size=40),
)
def test_catalog_coverage_is_a_share_of_the_catalog(items: list[int]) -> None:
    value = catalog_coverage(items, 20)
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(len(set(items)) / 20)
