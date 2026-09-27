"""Stage 3 rules: the seen filter, MMR diversity and the category cap."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval_ranking_cascade.ranking.rules import (
    allowed_positions,
    cap_per_group,
    filter_seen,
    intra_list_similarity,
    mmr_objective,
    mmr_select,
    normalize_scores,
)

pytestmark = pytest.mark.unit


def _unit_rows(rng: np.random.Generator, n: int, dim: int) -> npt.NDArray[np.float64]:
    matrix = rng.normal(size=(n, dim))
    return np.asarray(matrix / np.linalg.norm(matrix, axis=1, keepdims=True))


# --- seen filter ---


def test_allowed_positions_drops_seen_items_padding_and_repeats() -> None:
    candidates = np.asarray([7, -1, 3, 7, 5, 3], dtype=np.int64)
    keep = allowed_positions(candidates, np.asarray([5], dtype=np.int64))
    # -1 is faiss padding, position 3 repeats item 7 and position 5 repeats item 3.
    assert keep.tolist() == [0, 2]
    assert filter_seen(candidates, [5]).tolist() == [7, 3]


def test_allowed_positions_preserves_the_ann_order() -> None:
    """The cascade reranks a prefix of the survivors, so the order has to be the input's."""
    candidates = np.asarray([9, 2, 8, 1], dtype=np.int64)
    keep = allowed_positions(candidates, np.asarray([8], dtype=np.int64))
    assert keep.tolist() == [0, 1, 3]
    assert np.all(np.diff(keep) > 0)


def test_allowed_positions_with_nothing_seen_and_with_everything_seen() -> None:
    candidates = np.asarray([4, 5, 6], dtype=np.int64)
    assert allowed_positions(candidates, np.empty(0, dtype=np.int64)).tolist() == [0, 1, 2]
    assert allowed_positions(candidates, candidates).size == 0
    assert filter_seen(candidates, candidates).size == 0


@settings(max_examples=80, deadline=None)
@given(
    candidates=st.lists(st.integers(min_value=-1, max_value=30), min_size=0, max_size=25),
    seen=st.lists(st.integers(min_value=0, max_value=30), min_size=0, max_size=10),
)
def test_filter_seen_output_is_a_deduplicated_subsequence_disjoint_from_seen(
    candidates: list[int], seen: list[int]
) -> None:
    kept = filter_seen(candidates, seen).tolist()
    assert len(set(kept)) == len(kept)
    assert not set(kept) & set(seen)
    assert all(item >= 0 for item in kept)
    # A subsequence: walking the input once must consume the whole output in order.
    iterator = iter(candidates)
    assert all(any(item == found for found in iterator) for item in kept)


# --- score normalization ---


def test_normalize_scores_maps_to_the_unit_interval_and_handles_a_flat_input() -> None:
    np.testing.assert_allclose(normalize_scores([2.0, 4.0, 6.0]), [0.0, 0.5, 1.0])
    # A constant input carries no relevance signal, so every candidate ties at the top.
    np.testing.assert_allclose(normalize_scores([3.0, 3.0]), [1.0, 1.0])
    assert normalize_scores([]).size == 0


def test_normalize_scores_refuses_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        normalize_scores([1.0, np.inf])


@settings(max_examples=80, deadline=None)
@given(
    scores=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30,
    )
)
def test_normalize_scores_is_monotone_and_bounded(scores: list[float]) -> None:
    normalized = normalize_scores(scores)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    order = np.argsort(np.asarray(scores), kind="stable")
    assert np.all(np.diff(normalized[order]) >= -1e-12)


# --- MMR ---


def test_mmr_at_lambda_one_is_pure_relevance_order() -> None:
    rng = np.random.default_rng(0)
    vectors = _unit_rows(rng, 6, 4)
    scores = np.asarray([0.1, 0.9, 0.5, 0.2, 0.7, 0.3])
    chosen = mmr_select(scores, vectors, 4, 1.0)
    assert chosen.tolist() == np.argsort(-scores, kind="stable")[:4].tolist()


def test_mmr_at_lambda_zero_still_starts_from_the_most_relevant_item() -> None:
    """With an empty selection the diversity term is undefined; defining it as 0 would tie."""
    rng = np.random.default_rng(1)
    vectors = _unit_rows(rng, 5, 3)
    scores = np.asarray([0.2, 0.4, 0.95, 0.1, 0.3])
    assert int(mmr_select(scores, vectors, 3, 0.0)[0]) == 2


def test_lambda_trades_a_near_duplicate_for_a_dissimilar_item() -> None:
    """Item 1 duplicates item 0 and is nearly as relevant; item 2 is orthogonal and weaker.

    Which one follows item 0 is exactly what ``lambda`` buys, and the switch point is where the
    relevance gap stops paying for the similarity. Here the gap is 0.95 of the normalized range
    against a similarity difference of 0.5, so relevance wins until lambda drops below about 0.35.
    """
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9998, 0.02, 0.0],  # a near-duplicate of item 0
            [0.0, 0.0, 1.0],  # orthogonal, slightly less relevant
        ]
    )
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = np.asarray([1.0, 0.98, 0.6])
    assert mmr_select(scores, vectors, 2, 1.0).tolist() == [0, 1]
    assert mmr_select(scores, vectors, 2, 0.5).tolist() == [0, 1]
    assert mmr_select(scores, vectors, 2, 0.2).tolist() == [0, 2]
    assert mmr_select(scores, vectors, 2, 0.0).tolist() == [0, 2]


def test_mmr_returns_a_permutation_prefix_without_repeats() -> None:
    rng = np.random.default_rng(2)
    vectors = _unit_rows(rng, 12, 5)
    scores = rng.random(12)
    chosen = mmr_select(scores, vectors, 12, 0.6)
    assert sorted(chosen.tolist()) == list(range(12))


def test_mmr_k_is_clamped_to_the_candidate_count_and_zero_is_allowed() -> None:
    rng = np.random.default_rng(3)
    vectors = _unit_rows(rng, 4, 3)
    scores = rng.random(4)
    assert mmr_select(scores, vectors, 99, 0.5).shape == (4,)
    assert mmr_select(scores, vectors, 0, 0.5).shape == (0,)
    assert mmr_select(np.zeros(0), np.zeros((0, 3)), 3, 0.5).shape == (0,)


def test_mmr_rejects_an_out_of_range_lambda_a_negative_k_and_misaligned_vectors() -> None:
    rng = np.random.default_rng(4)
    vectors = _unit_rows(rng, 3, 2)
    scores = rng.random(3)
    with pytest.raises(ValueError, match="lam"):
        mmr_select(scores, vectors, 2, 1.5)
    with pytest.raises(ValueError, match="k must be"):
        mmr_select(scores, vectors, -1, 0.5)
    with pytest.raises(ValueError, match="one row per score"):
        mmr_select(scores, vectors[:2], 2, 0.5)


def test_mmr_objective_is_relevance_until_something_is_selected() -> None:
    relevance = np.asarray([0.2, 0.8, 0.5])
    similarity = np.asarray([[1.0, 0.4, 0.1], [0.4, 1.0, 0.9], [0.1, 0.9, 1.0]])
    np.testing.assert_allclose(mmr_objective(relevance, similarity, [], 0.5), relevance)
    marginal = mmr_objective(relevance, similarity, [1], 0.5)
    np.testing.assert_allclose(marginal, 0.5 * relevance - 0.5 * similarity[:, 1])


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=12),
    k=st.integers(min_value=0, max_value=12),
    lam=st.floats(min_value=0.0, max_value=1.0),
    seed=st.integers(min_value=0, max_value=500),
)
def test_mmr_selection_is_deterministic_and_a_valid_index_set(
    n: int, k: int, lam: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    vectors = _unit_rows(rng, n, 4)
    scores = rng.normal(size=n)
    first = mmr_select(scores, vectors, k, lam)
    second = mmr_select(scores, vectors, k, lam)
    assert first.tolist() == second.tolist()
    assert first.shape[0] == min(k, n)
    assert len(set(first.tolist())) == first.shape[0]
    assert all(0 <= int(index) < n for index in first)


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=500),
)
def test_each_mmr_pick_maximizes_the_marginal_objective(n: int, seed: int) -> None:
    """The greedy invariant, checked step by step against a recomputed objective."""
    lam = 0.6
    rng = np.random.default_rng(seed)
    vectors = _unit_rows(rng, n, 4)
    scores = rng.normal(size=n)
    chosen = mmr_select(scores, vectors, n, lam).tolist()
    relevance = normalize_scores(scores)
    similarity = np.clip(0.5 * (vectors @ vectors.T + 1.0), 0.0, 1.0)
    for step in range(1, len(chosen)):
        marginal = mmr_objective(relevance, similarity, chosen[:step], lam)
        remaining = [index for index in range(n) if index not in chosen[:step]]
        best = max(float(marginal[index]) for index in remaining)
        assert float(marginal[chosen[step]]) == pytest.approx(best)


# --- category cap ---


def test_cap_per_group_keeps_the_earliest_items_of_each_group() -> None:
    order = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    groups = np.asarray([1, 1, 1, 2, 2, 3], dtype=np.int64)
    assert cap_per_group(order, groups, 2).tolist() == [0, 1, 3, 4, 5]


def test_cap_per_group_truncates_at_k_and_respects_rank_order() -> None:
    order = np.asarray([4, 2, 0, 1, 3], dtype=np.int64)
    groups = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    # Group 1 holds indices 2, 3, 4; only the first two in rank order survive.
    assert cap_per_group(order, groups, 2, k=3).tolist() == [4, 2, 0]


def test_cap_per_group_can_return_fewer_than_k_when_the_cap_binds() -> None:
    order = np.arange(5, dtype=np.int64)
    groups = np.zeros(5, dtype=np.int64)
    assert cap_per_group(order, groups, 2, k=5).tolist() == [0, 1]


def test_cap_per_group_refuses_a_zero_cap() -> None:
    with pytest.raises(ValueError, match="max_per_group"):
        cap_per_group(np.arange(3), np.zeros(3), 0)


@settings(max_examples=80, deadline=None)
@given(
    groups=st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=25),
    cap=st.integers(min_value=1, max_value=4),
    k=st.integers(min_value=1, max_value=25),
)
def test_cap_per_group_never_exceeds_the_cap_or_k(groups: list[int], cap: int, k: int) -> None:
    labels = np.asarray(groups, dtype=np.int64)
    kept = cap_per_group(np.arange(labels.shape[0]), labels, cap, k)
    assert kept.shape[0] <= k
    counts = np.bincount(labels[kept], minlength=5)
    assert counts.max(initial=0) <= cap
    assert np.all(np.diff(kept) > 0)


# --- intra-list similarity ---


def test_intra_list_similarity_is_one_for_identical_items_and_zero_for_orthogonal_ones() -> None:
    same = np.tile(np.asarray([[1.0, 0.0]]), (3, 1))
    assert intra_list_similarity(same) == pytest.approx(1.0)
    assert intra_list_similarity(np.eye(3)) == pytest.approx(0.0)


def test_intra_list_similarity_of_a_short_list_is_zero() -> None:
    assert intra_list_similarity(np.zeros((1, 3))) == 0.0
    assert intra_list_similarity(np.zeros((0, 3))) == 0.0
    assert intra_list_similarity(np.zeros(3)) == 0.0


def test_intra_list_similarity_equals_the_mean_of_the_explicit_pairs() -> None:
    rng = np.random.default_rng(5)
    vectors = _unit_rows(rng, 6, 4)
    pairs = [
        float(vectors[i] @ vectors[j])
        for i in range(vectors.shape[0])
        for j in range(vectors.shape[0])
        if i != j
    ]
    assert intra_list_similarity(vectors) == pytest.approx(float(np.mean(pairs)))
