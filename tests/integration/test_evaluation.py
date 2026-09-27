"""Offline evaluation: slate reranking, full-corpus ranking, position-bias correction, shilling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from retrieval_ranking_cascade.evaluation.offline import (
    SLATE_POLICIES,
    corpus_evaluation,
    make_user_sample,
    position_bias_report,
    slate_evaluation,
)
from retrieval_ranking_cascade.evaluation.shilling import (
    AttackConfig,
    inject_shilling_profiles,
    mean_item_rank,
    run_shilling_attack,
)
from retrieval_ranking_cascade.pipeline import held_out_sessions
from retrieval_ranking_cascade.serving.budget import BudgetController
from retrieval_ranking_cascade.serving.cascade import Cascade, CascadeConfig

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def cascade(tiny_training: Any) -> Cascade:
    controller = BudgetController(total_seconds=1.0, retrieval_seconds=0.4, ranking_seconds=0.4)
    return Cascade(
        tiny_training.bundle,
        controller,
        CascadeConfig(candidate_k=200, rerank_depth=120, default_k=10),
    )


@pytest.fixture(scope="module")
def slate_report(tiny_training: Any, cascade: Cascade) -> Any:
    return slate_evaluation(
        sessions=held_out_sessions(tiny_training),
        cascade=cascade,
        world=tiny_training.world,
        propensity=tiny_training.propensity,
        k=10,
        max_sessions=200,
        seed=1,
    )


@pytest.fixture(scope="module")
def corpus_report(tiny_training: Any, cascade: Cascade) -> dict[str, Any]:
    users = make_user_sample(
        held_out_sessions(tiny_training), tiny_training.bundle.n_users, 60, seed=2
    )
    return corpus_evaluation(
        cascade=cascade, world=tiny_training.world, users=users, k=10, upper_bound_users=20
    )


# --- user sampling ---


def test_the_user_sample_is_distinct_sorted_and_known_to_the_bundle(tiny_training: Any) -> None:
    sessions = held_out_sessions(tiny_training)
    users = make_user_sample(sessions, tiny_training.bundle.n_users, 40, seed=3)
    assert users.size == 40
    assert np.all(np.diff(users) > 0)
    assert users.max() < tiny_training.bundle.n_users
    assert set(users.tolist()) <= set(sessions.user_id.astype(np.int64).tolist())


def test_a_predicate_narrows_the_sample_and_an_empty_one_is_refused(tiny_training: Any) -> None:
    sessions = held_out_sessions(tiny_training)
    clicks = tiny_training.bundle.snapshot.user_clicks
    heavy = make_user_sample(
        sessions, tiny_training.bundle.n_users, 20, seed=4, predicate=lambda u: clicks[u] > 0
    )
    assert np.all(clicks[heavy] > 0)
    with pytest.raises(ValueError, match="no eligible users"):
        make_user_sample(sessions, tiny_training.bundle.n_users, 5, predicate=lambda u: u < 0)


def test_the_sample_is_clamped_to_the_eligible_population(tiny_training: Any) -> None:
    sessions = held_out_sessions(tiny_training)
    users = make_user_sample(sessions, tiny_training.bundle.n_users, 10**6, seed=5)
    assert users.size <= tiny_training.bundle.n_users


# --- position bias ---


def test_ips_shrinks_the_gap_between_estimated_and_true_per_rank_relevance(
    tiny_training: Any,
) -> None:
    """The cleanest bias measurement: no ranking and no gain scale, just level per rank."""
    report = position_bias_report(
        tiny_training.interactions,
        tiny_training.world,
        tiny_training.propensity,
        slots=tiny_training.world_config.slots,
    )
    assert len(report.ranks) == tiny_training.world_config.slots
    # The naive estimate decays down the page even though relevance per rank is roughly flat.
    assert report.naive[0] > report.naive[-1]
    assert report.ips_bias < report.naive_bias
    payload = report.as_dict()
    assert float(str(payload["ips_position_bias"])) < float(str(payload["naive_position_bias"]))


def test_the_randomized_slice_is_excluded_by_default(tiny_training: Any) -> None:
    """Including it would let the estimator grade its own homework."""
    both = position_bias_report(
        tiny_training.interactions,
        tiny_training.world,
        tiny_training.propensity,
        slots=tiny_training.world_config.slots,
        exclude_randomized=False,
    )
    excluded = position_bias_report(
        tiny_training.interactions,
        tiny_training.world,
        tiny_training.propensity,
        slots=tiny_training.world_config.slots,
    )
    assert both.naive != excluded.naive


def test_an_empty_impression_set_is_refused(tiny_training: Any) -> None:
    empty = tiny_training.interactions.select(np.zeros(len(tiny_training.interactions), dtype=bool))
    with pytest.raises(ValueError, match="no impressions left"):
        position_bias_report(
            empty,
            tiny_training.world,
            tiny_training.propensity,
            slots=tiny_training.world_config.slots,
        )


# --- protocol A: slate reranking ---


def test_every_policy_is_scored_three_ways_on_the_same_slates(slate_report: Any) -> None:
    assert slate_report.sessions > 0
    assert set(slate_report.metrics) == set(SLATE_POLICIES)
    for values in slate_report.metrics.values():
        assert set(values) == {
            "ndcg_naive",
            "ndcg_ips",
            "ndcg_truth",
            "mrr_naive",
            "mrr_truth",
        }
        for value in values.values():
            assert 0.0 <= value <= 1.0


def test_the_cascade_beats_popularity_on_the_ground_truth_slate_metric(slate_report: Any) -> None:
    metrics = slate_report.metrics
    assert metrics["cascade"]["ndcg_truth"] > metrics["popularity"]["ndcg_truth"]
    assert metrics["cascade"]["ndcg_truth"] > 0.5


def test_advantage_is_reported_against_the_logged_policy_for_every_other_policy(
    slate_report: Any,
) -> None:
    assert set(slate_report.advantage) == set(SLATE_POLICIES) - {"logged"}
    for values in slate_report.advantage.values():
        assert set(values) == {"naive", "ips", "truth"}


def test_the_ips_advantage_is_closer_to_the_truth_than_the_naive_one(slate_report: Any) -> None:
    """The relative question offline evaluation is used for, and the bias in answering it."""
    assert slate_report.ips_bias < slate_report.naive_bias
    payload = slate_report.as_dict()
    assert payload["ips_advantage_bias"] < payload["naive_advantage_bias"]
    assert payload["sessions"] == slate_report.sessions


def test_slate_evaluation_validates_its_arguments(tiny_training: Any, cascade: Cascade) -> None:
    with pytest.raises(ValueError, match="k must be"):
        slate_evaluation(
            sessions=held_out_sessions(tiny_training),
            cascade=cascade,
            world=tiny_training.world,
            propensity=tiny_training.propensity,
            k=0,
        )
    sessions = held_out_sessions(tiny_training)
    all_clicked = sessions.select(sessions.clicked == 1)
    with pytest.raises(ValueError, match="no held-out session"):
        slate_evaluation(
            sessions=all_clicked,
            cascade=cascade,
            world=tiny_training.world,
            propensity=tiny_training.propensity,
            k=10,
        )


def test_max_sessions_caps_the_work(tiny_training: Any, cascade: Cascade) -> None:
    small = slate_evaluation(
        sessions=held_out_sessions(tiny_training),
        cascade=cascade,
        world=tiny_training.world,
        propensity=tiny_training.propensity,
        k=10,
        max_sessions=25,
        seed=6,
    )
    assert small.sessions == 25


# --- protocol B: full corpus ---


def test_the_corpus_report_covers_every_policy_with_its_quality_and_diversity(
    corpus_report: dict[str, Any],
) -> None:
    for name in ("popularity", "tower", "cascade_relevance", "cascade", "exact_rank"):
        block = corpus_report[name]
        assert 0.0 <= block["ndcg_at_k"] <= 1.0
        assert 0.0 <= block["intra_list_similarity"] <= 1.0
        assert 0.0 < block["catalog_coverage"] <= 1.0
        assert 0.0 <= block["cold_item_share"] <= 1.0
        assert block["users"] > 0


def test_the_cascade_beats_popularity_on_true_relevance(corpus_report: dict[str, Any]) -> None:
    """The baseline every recommender has to clear, on the protocol where retrieval matters.

    The tower comparison is deliberately not asserted here. At this test scale the label period
    holds a few dozen usable sessions, which is not enough to fit a ranker that improves on the
    tower's own score - and the ``exact_rank`` row confirms it is the ranker, not the cascade, that
    is the limit. ``bench/bench_quality.py --check`` enforces the margin over both baselines on the
    slate protocol at ``small`` scale, where the ranker has a training set.
    """
    cascade_ndcg = corpus_report["cascade_ndcg_at_k"]
    assert cascade_ndcg > corpus_report["popularity"]["ndcg_at_k"]
    assert cascade_ndcg > 0.3
    assert corpus_report["upper_bound_retained"] > 0.8


def test_the_gap_to_the_unservable_upper_bound_is_reported(
    corpus_report: dict[str, Any],
) -> None:
    assert corpus_report["upper_bound_gap"] >= 0.0
    assert 0.0 < corpus_report["upper_bound_retained"] <= 1.0
    assert corpus_report["upper_bound_ndcg"] >= corpus_report["cascade_ndcg_at_k"] - 1e-9


def test_the_diversity_rules_cost_relevance_and_buy_coverage(
    corpus_report: dict[str, Any],
) -> None:
    """Reported rather than hidden: MMR and the category cap are a trade, not a free win."""
    assert corpus_report["diversity_cost"] >= 0.0
    assert corpus_report["cascade_served_ndcg_at_k"] <= corpus_report["cascade_ndcg_at_k"] + 1e-9
    assert (
        corpus_report["cascade"]["intra_list_similarity"]
        <= corpus_report["cascade_relevance"]["intra_list_similarity"] + 1e-9
    )


def test_cold_user_slices_are_reported_separately(corpus_report: dict[str, Any]) -> None:
    cascade = corpus_report["cascade"]
    assert "cold_user_ndcg_at_k" in cascade
    assert cascade["cold_users"] >= 0


def test_corpus_evaluation_validates_its_arguments(tiny_training: Any, cascade: Cascade) -> None:
    with pytest.raises(ValueError, match="k must be"):
        corpus_evaluation(cascade=cascade, world=tiny_training.world, users=[0], k=0)
    with pytest.raises(ValueError, match="must not be empty"):
        corpus_evaluation(cascade=cascade, world=tiny_training.world, users=[], k=5)


def test_ef_search_changes_the_candidate_set_the_cascade_can_choose_from(
    tiny_training: Any, cascade: Cascade
) -> None:
    """The recall ceiling is real: items the ANN stage misses can never be recommended.

    What is asserted here is the mechanism - a narrow search retrieves a measurably worse candidate
    set and the served lists change. Whether that costs NDCG depends on how much better than the
    tower the ranker is, which is a scale question the bench's efSearch ablation answers.
    """
    from retrieval_ranking_cascade.retrieval.ann import exact_top_k, recall_at_k

    bundle = tiny_training.bundle
    users = make_user_sample(held_out_sessions(tiny_training), bundle.n_users, 30, seed=7)
    queries = bundle.user_vectors(users)
    _, truth = exact_top_k(queries, bundle.item_vectors, 200)
    recalls = {}
    for ef in (4, 512):
        _, approx = bundle.index.search(queries, 200, ef_search=ef)
        recalls[ef] = float(recall_at_k(approx, truth).mean())
    assert recalls[4] < 0.5 < recalls[512]

    wide = corpus_evaluation(
        cascade=cascade,
        world=tiny_training.world,
        users=users,
        k=10,
        upper_bound_users=0,
        ef_search=512,
    )
    narrow = corpus_evaluation(
        cascade=cascade,
        world=tiny_training.world,
        users=users,
        k=10,
        upper_bound_users=0,
        ef_search=4,
    )
    assert narrow["cascade_ndcg_at_k"] != wide["cascade_ndcg_at_k"]
    # The upper bound block is only computed when it is asked for.
    assert "upper_bound_gap" not in narrow


# --- shilling robustness ---


def test_injected_profiles_land_as_new_accounts_that_all_click_the_target(
    tiny_training: Any,
) -> None:
    config = AttackConfig(target_item=17, attackers=30, filler_items=4, seed=1)
    before = tiny_training.interactions
    after = inject_shilling_profiles(
        before,
        config=config,
        n_users=tiny_training.bundle.n_users,
        slots=tiny_training.world_config.slots,
        popular_items=tiny_training.bundle.snapshot.popular_items,
    )
    added = len(after) - len(before)
    assert added == 30 * (1 + 4)
    injected = after.select(np.arange(len(before), len(after)))
    assert int(injected.user_id.min()) >= tiny_training.bundle.n_users
    assert np.unique(injected.user_id).size == 30
    # Every injected session clicks the target, and nothing else was touched.
    assert int(np.sum(injected.item_id == 17)) == 30
    assert int(injected.clicked.sum()) == added
    assert not bool(injected.randomized.any())


def test_the_target_sits_in_a_plausible_slot_rather_than_always_first(
    tiny_training: Any,
) -> None:
    config = AttackConfig(target_item=17, attackers=40, filler_items=5, seed=2)
    after = inject_shilling_profiles(
        tiny_training.interactions,
        config=config,
        n_users=tiny_training.bundle.n_users,
        slots=tiny_training.world_config.slots,
        popular_items=tiny_training.bundle.snapshot.popular_items,
    )
    injected = after.select(np.arange(len(tiny_training.interactions), len(after)))
    slots = injected.position[injected.item_id == 17]
    assert np.unique(slots).size > 1


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"target_item": -1}, "target_item"),
        ({"attackers": 0}, "attackers"),
        ({"sessions_per_attacker": 0}, "attackers"),
        ({"sample_users": 0}, "sample_users"),
    ],
)
def test_the_attack_config_is_validated(overrides: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AttackConfig(**{"target_item": 1, **overrides})


def test_the_full_catalogue_rank_of_an_item_is_measurable(tiny_training: Any) -> None:
    bundle = tiny_training.bundle
    rank = mean_item_rank(
        snapshot=bundle.snapshot,
        model=bundle.model,
        ranker=bundle.ranker,
        item=17,
        users=np.asarray([0, 1, 2, 3], dtype=np.int64),
    )
    assert 1.0 <= rank <= bundle.n_items


def test_robust_aggregation_damps_a_shilling_attack_the_naive_one_falls_for(
    tiny_training: Any,
) -> None:
    """Fake clicks are legitimate-looking events; only the aggregation decides whether they work."""
    result = tiny_training
    clickers = result.bundle.snapshot.item_clickers
    target = int(np.argsort(clickers)[len(clickers) // 2])
    report = run_shilling_attack(
        interactions=result.interactions,
        catalog=result.catalog,
        model=result.bundle.model,
        ranker=result.bundle.ranker,
        n_users=result.bundle.n_users,
        slots=result.world_config.slots,
        config=AttackConfig(
            target_item=target, attackers=150, filler_items=5, sample_users=8, seed=3
        ),
    )
    payload = report.as_dict()
    assert payload["target_item"] == target
    assert set(report.robust.as_dict()) == set(report.naive.as_dict())
    # The naive aggregation moves the item up; the robust one moves it far less.
    assert report.naive.lift > 1.0
    assert report.robust.lift < report.naive.lift
    assert report.damping is not None
    assert report.damping > 1.0
    # The clicker count the attack buys is what the tenure weight is supposed to flatten.
    naive_gain = report.naive.clickers_after - report.naive.clickers_before
    robust_gain = report.robust.clickers_after - report.robust.clickers_before
    assert robust_gain < 0.5 * naive_gain


def test_an_attack_that_achieves_nothing_reports_no_damping_ratio(tiny_training: Any) -> None:
    """Dividing by a lift of 1 would invent a number; ``None`` says the attack simply failed."""
    result = tiny_training
    report = run_shilling_attack(
        interactions=result.interactions,
        catalog=result.catalog,
        model=result.bundle.model,
        ranker=result.bundle.ranker,
        n_users=result.bundle.n_users,
        slots=result.world_config.slots,
        config=AttackConfig(
            target_item=int(result.bundle.snapshot.popular_items[0]),
            attackers=1,
            filler_items=1,
            sample_users=4,
            seed=4,
        ),
    )
    assert report.naive.lift <= 1.0 or report.damping is not None
    if report.naive.lift <= 1.0:
        assert report.damping is None
