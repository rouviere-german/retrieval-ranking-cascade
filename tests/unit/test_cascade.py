"""The three-stage cascade itself: retrieval, reranking, rules, degradation and cold start."""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any

import numpy as np
import pytest

from retrieval_ranking_cascade.serving.budget import (
    DEGRADE_SKIP_RERANK,
    BudgetController,
    CostEstimator,
)
from retrieval_ranking_cascade.serving.bundle import Bundle
from retrieval_ranking_cascade.serving.cascade import (
    STAGE_RANKING,
    STAGE_RETRIEVAL,
    STAGE_RULES,
    Cascade,
    CascadeConfig,
)

pytestmark = pytest.mark.unit


def _controller(
    *,
    total: float = 0.5,
    retrieval: float = 0.2,
    ranking: float = 0.2,
    estimators: dict[str, CostEstimator] | None = None,
) -> BudgetController:
    return BudgetController(
        total_seconds=total,
        retrieval_seconds=retrieval,
        ranking_seconds=ranking,
        estimators=estimators,
    )


@pytest.fixture(scope="module")
def bundle(tiny_training: Any) -> Bundle:
    loaded: Bundle = tiny_training.bundle
    return loaded


@pytest.fixture
def cascade(bundle: Bundle) -> Cascade:
    return Cascade(bundle, _controller(), CascadeConfig(candidate_k=150, rerank_depth=80))


# --- configuration ---


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"candidate_k": 0}, "sizes must be"),
        ({"default_k": 0}, "sizes must be"),
        ({"rerank_depth": 600, "candidate_k": 500}, "rerank_depth must not exceed"),
        ({"mmr_lambda": 1.5}, "mmr_lambda"),
        ({"mmr_lambda": -0.1}, "mmr_lambda"),
    ],
)
def test_the_cascade_config_is_validated(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CascadeConfig(**overrides)


# --- the happy path ---


def test_a_known_user_gets_k_ranked_items_from_every_stage(cascade: Cascade) -> None:
    result = cascade.recommend(3, k=10)
    assert len(result.items) == 10
    assert [item.rank for item in result.items] == list(range(1, 11))
    assert not result.cold_start
    assert result.reranked
    assert not result.degraded
    assert result.candidates > 0
    assert result.reranked_candidates > 0
    assert result.ef_search == cascade.config.ef_search
    assert set(result.stage_seconds) == {STAGE_RETRIEVAL, STAGE_RANKING, STAGE_RULES}
    assert result.total_seconds > 0.0
    assert result.bundle_id == cascade.bundle.bundle_id


def test_returned_items_are_distinct_valid_ids_with_their_real_categories(
    cascade: Cascade,
) -> None:
    result = cascade.recommend(5, k=15)
    ids = result.item_ids
    assert len(set(ids)) == len(ids)
    assert all(0 <= item < cascade.bundle.n_items for item in ids)
    for item in result.items:
        assert item.category == int(cascade.bundle.snapshot.item_category[item.item_id])


def test_items_the_user_already_clicked_are_never_recommended(cascade: Cascade) -> None:
    """The seen filter runs between the stages, so a clicked item cannot even reach the ranker."""
    snapshot = cascade.bundle.snapshot
    heavy = int(np.argmax(np.diff(snapshot.seen_indptr)))
    seen = set(snapshot.seen(heavy).tolist())
    assert seen
    assert not set(cascade.recommend(heavy, k=20).item_ids) & seen


def test_the_category_cap_is_enforced(bundle: Bundle) -> None:
    cascade = Cascade(
        bundle,
        _controller(),
        CascadeConfig(candidate_k=150, rerank_depth=80, max_per_category=2),
    )
    counts = Counter(item.category for item in cascade.recommend(7, k=20).items)
    assert counts
    assert max(counts.values()) <= 2


def test_k_is_honoured_and_a_non_positive_k_is_refused(cascade: Cascade) -> None:
    for k in (1, 5, 25):
        assert len(cascade.recommend(2, k=k).items) <= k
    assert len(cascade.recommend(2).items) <= cascade.config.default_k
    with pytest.raises(ValueError, match="k must be"):
        cascade.recommend(2, k=0)


def test_lambda_one_gives_pure_relevance_order_and_lower_lambda_diversifies(
    cascade: Cascade,
) -> None:
    relevance = cascade.recommend(4, k=12, mmr_lambda=1.0)
    diverse = cascade.recommend(4, k=12, mmr_lambda=0.2)
    assert relevance.item_ids != diverse.item_ids
    assert cascade.list_similarity(diverse.item_ids) <= cascade.list_similarity(relevance.item_ids)


def test_the_requested_ann_parameters_are_echoed_back_and_bound_the_candidate_set(
    cascade: Cascade,
) -> None:
    narrow = cascade.recommend(6, k=10, ef_search=32, candidate_k=64)
    wide = cascade.recommend(6, k=10, ef_search=512, candidate_k=150)
    assert (narrow.ef_search, wide.ef_search) == (32, 512)
    assert narrow.candidates <= 64
    assert wide.candidates <= 150
    assert narrow.candidates <= wide.candidates
    for result in (narrow, wide):
        assert 0 < len(result.items) <= 10


def test_the_category_cap_can_return_a_short_page_rather_than_break_the_constraint(
    bundle: Bundle,
) -> None:
    """A hard business constraint wins over filling the page; that is the point of a hard cap."""
    cascade = Cascade(
        bundle,
        _controller(),
        CascadeConfig(candidate_k=150, rerank_depth=80, max_per_category=1),
    )
    result = cascade.recommend(6, k=40)
    assert len(result.items) < 40
    assert len({item.category for item in result.items}) == len(result.items)


def test_recommendations_are_deterministic_for_the_same_request(cascade: Cascade) -> None:
    first = cascade.recommend(8, k=10)
    second = cascade.recommend(8, k=10)
    assert first.item_ids == second.item_ids
    assert [item.score for item in first.items] == [item.score for item in second.items]


def test_list_similarity_of_a_short_list_is_zero(cascade: Cascade) -> None:
    assert cascade.list_similarity([]) == 0.0
    assert cascade.list_similarity([3]) == 0.0
    assert cascade.list_similarity([3, 4]) != 0.0


# --- cold start ---


def test_an_unknown_user_is_served_from_the_popularity_list_and_flagged(cascade: Cascade) -> None:
    result = cascade.recommend(10**9, k=10)
    assert result.cold_start
    assert not result.reranked
    assert not result.degraded
    assert result.ef_search == 0
    assert result.reranked_candidates == 0
    assert len(result.items) == 10
    assert set(result.stage_seconds) == {STAGE_RULES}
    popular = set(cascade.bundle.snapshot.popular_items.tolist())
    assert set(result.item_ids) <= popular


def test_a_negative_user_id_also_takes_the_cold_start_path(cascade: Cascade) -> None:
    assert cascade.recommend(-1, k=5).cold_start


def test_the_cold_start_list_respects_the_category_cap(bundle: Bundle) -> None:
    cascade = Cascade(bundle, _controller(), CascadeConfig(max_per_category=2))
    counts = Counter(item.category for item in cascade.recommend(10**9, k=20).items)
    assert max(counts.values()) <= 2


def test_a_known_user_with_no_click_history_still_goes_through_the_towers(
    cascade: Cascade,
) -> None:
    """An empty history is not cold start: the id embedding is zero, so the tower is content-only.

    That distinction matters because the two paths answer differently - popularity against the
    user's own content-based neighbours.
    """
    snapshot = cascade.bundle.snapshot
    quiet = np.flatnonzero(snapshot.user_clicks == 0)
    if quiet.size == 0:
        pytest.skip("this tiny world has no user without clicks")
    result = cascade.recommend(int(quiet[0]), k=10)
    assert not result.cold_start
    assert result.reranked


# --- degradation ---


def test_a_budget_too_small_for_the_reranker_returns_ann_order_flagged_degraded(
    bundle: Bundle,
) -> None:
    cascade = Cascade(
        bundle,
        _controller(
            total=0.002,
            retrieval=0.001,
            ranking=0.001,
            estimators={
                "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9),
                "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1.0),
            },
        ),
        CascadeConfig(candidate_k=150, rerank_depth=80),
    )
    result = cascade.recommend(3, k=10)
    assert result.degraded
    assert DEGRADE_SKIP_RERANK in result.degraded_reasons
    assert not result.reranked
    assert result.reranked_candidates == 0
    # The response is still a real recommendation list, just less precise.
    assert len(result.items) == 10
    assert STAGE_RANKING not in result.stage_seconds


def test_the_degraded_order_is_the_ann_order_and_differs_from_the_reranked_one(
    bundle: Bundle,
) -> None:
    healthy = Cascade(bundle, _controller(), CascadeConfig(candidate_k=150, rerank_depth=80))
    starved = Cascade(
        bundle,
        _controller(
            total=0.002,
            retrieval=0.001,
            ranking=0.001,
            estimators={
                "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9),
                "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1.0),
            },
        ),
        CascadeConfig(candidate_k=150, rerank_depth=80, mmr_lambda=1.0),
    )
    reranked = healthy.recommend(3, k=10, mmr_lambda=1.0).item_ids
    degraded = starved.recommend(3, k=10).item_ids
    vector = bundle.user_vectors(np.asarray([3]))
    _, ann = bundle.index.search(vector, 150, ef_search=starved.config.ef_search)
    seen = set(bundle.snapshot.seen(3).tolist())
    ann_order = [item for item in ann[0].tolist() if item >= 0 and item not in seen]
    # With lambda 1 stage 3 only applies the category cap, which is an order-preserving filter, so
    # the degraded response is a subsequence of the ANN order rather than an arbitrary reshuffle.
    remaining = iter(ann_order)
    assert all(any(item == found for found in remaining) for item in degraded)
    assert degraded[0] == ann_order[0]
    assert degraded != reranked


def test_a_tight_retrieval_budget_lowers_ef_search_but_keeps_reranking(bundle: Bundle) -> None:
    cascade = Cascade(
        bundle,
        _controller(
            total=0.5,
            retrieval=0.002,
            ranking=0.2,
            estimators={
                "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-4),
                "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9),
            },
        ),
        CascadeConfig(candidate_k=150, rerank_depth=80, ef_search=1024),
    )
    result = cascade.recommend(3, k=10)
    assert result.degraded
    assert result.ef_search < 1024
    assert result.reranked


def test_queue_time_and_contention_are_carried_into_the_result(cascade: Cascade) -> None:
    result = cascade.recommend(3, k=5, queued_seconds=0.004, contention=2.5)
    assert result.queued_seconds == pytest.approx(0.004)
    assert result.contention == pytest.approx(2.5)


def test_contention_alone_can_force_degradation(bundle: Bundle) -> None:
    """Predictions are scaled by the slowdown a request should expect when sharing the pool."""
    controller = _controller(
        total=0.100,
        retrieval=0.005,
        ranking=0.020,
        estimators={
            "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9),
            "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-4, safety_factor=1.0),
        },
    )
    cascade = Cascade(bundle, controller, CascadeConfig(candidate_k=150, rerank_depth=80))
    assert not cascade.recommend(3, k=10, contention=1.0).degraded
    assert cascade.recommend(3, k=10, contention=16.0).degraded


# --- slate reranking helper ---


def test_rerank_slate_scores_a_fixed_slate(cascade: Cascade) -> None:
    slate = np.asarray([1, 2, 3, 4, 5], dtype=np.int64)
    scores = cascade.rerank_slate(3, slate)
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
    assert cascade.rerank_slate(3, np.empty(0, dtype=np.int64)).shape == (0,)


def test_rerank_slate_agrees_with_itself_across_calls(cascade: Cascade) -> None:
    slate = [7, 8, 9, 10]
    np.testing.assert_allclose(
        cascade.rerank_slate(4, slate), cascade.rerank_slate(4, slate), rtol=0
    )


# --- live swaps ---


def test_swapping_the_index_changes_what_retrieval_returns(bundle: Bundle) -> None:
    cascade = Cascade(bundle, _controller(), CascadeConfig(candidate_k=150, rerank_depth=80))
    before = cascade.recommend(3, k=10).item_ids
    from retrieval_ranking_cascade.retrieval.canary import corrupt_index

    damaged = bundle.index.copy()
    corrupt_index(damaged, 1.0, seed=2)
    cascade.swap_index(damaged)
    assert cascade.index is damaged
    assert cascade.recommend(3, k=10).item_ids != before


def test_swapping_the_bundle_returns_the_new_id_and_rebinds_the_index(bundle: Bundle) -> None:
    cascade = Cascade(bundle, _controller())
    assert cascade.swap_bundle(bundle) == bundle.bundle_id
    assert cascade.bundle is bundle
    assert cascade.index is bundle.index
    np.testing.assert_allclose(cascade.item_vectors(), bundle.item_vectors)


def test_concurrent_requests_all_return_complete_results(cascade: Cascade) -> None:
    """The cascade is shared between worker threads; nothing per-request may be shared state."""
    results: list[list[int]] = []
    lock = threading.Lock()

    def work(user: int) -> None:
        items = cascade.recommend(user, k=8).item_ids
        with lock:
            results.append(items)

    threads = [threading.Thread(target=work, args=(user,)) for user in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 8
    assert all(0 < len(items) <= 8 for items in results)
    assert all(len(set(items)) == len(items) for items in results)
    # Sequential answers must match the concurrent ones exactly: no per-request state is shared.
    sequential = [cascade.recommend(user, k=8).item_ids for user in range(8)]
    assert sorted(results) == sorted(sequential)
