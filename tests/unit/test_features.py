"""The DuckDB feature snapshot, its contract and the ranker's feature assembly."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval_ranking_cascade.data.generator import Catalog, Interactions, generate, world_config
from retrieval_ranking_cascade.ranking.features import (
    FEATURE_NAMES,
    FeatureSnapshot,
    SnapshotConfig,
    assemble_features,
    build_snapshot,
    item_side_features,
    user_side_features,
)

pytestmark = pytest.mark.unit

N_USERS = 6
N_ITEMS = 8
N_CATEGORIES = 3


def _catalog(
    *, n_items: int = N_ITEMS, n_categories: int = N_CATEGORIES, launch: list[int] | None = None
) -> Catalog:
    categories = (np.arange(n_items) % n_categories).astype(np.int32)
    launch_day = (
        np.zeros(n_items, dtype=np.int32) if launch is None else np.asarray(launch, dtype=np.int32)
    )
    return Catalog(item_category=categories, item_launch_day=launch_day, n_categories=n_categories)


def _log(rows: list[tuple[int, int, int, int, int]]) -> Interactions:
    """Rows of (session, user, item, clicked, day) as an impression log at position 1..n."""
    n = len(rows)
    sessions = np.asarray([row[0] for row in rows], dtype=np.int32)
    positions = np.ones(n, dtype=np.int8)
    for index in range(1, n):
        if sessions[index] == sessions[index - 1]:
            positions[index] = positions[index - 1] + 1
    return Interactions(
        session_id=sessions,
        user_id=np.asarray([row[1] for row in rows], dtype=np.int32),
        item_id=np.asarray([row[2] for row in rows], dtype=np.int32),
        position=positions,
        clicked=np.asarray([row[3] for row in rows], dtype=np.int8),
        randomized=np.zeros(n, dtype=np.bool_),
        day=np.asarray([row[4] for row in rows], dtype=np.int16),
        period=np.zeros(n, dtype=np.int8),
    )


@pytest.fixture(scope="module")
def tiny_snapshot() -> FeatureSnapshot:
    log = _log(
        [
            (0, 0, 0, 1, 0),
            (0, 0, 1, 0, 0),
            (0, 0, 2, 1, 0),
            (1, 0, 3, 0, 2),
            (1, 0, 0, 0, 2),
            (2, 1, 0, 1, 1),
            (2, 1, 4, 0, 1),
            (3, 2, 5, 0, 3),
            (3, 2, 6, 1, 3),
        ]
    )
    return build_snapshot(log, _catalog(), n_users=N_USERS)


# --- shape and content ---


def test_the_snapshot_reports_its_dimensions_and_day(tiny_snapshot: FeatureSnapshot) -> None:
    assert tiny_snapshot.n_users == N_USERS
    assert tiny_snapshot.n_items == N_ITEMS
    assert tiny_snapshot.n_categories == N_CATEGORIES
    # The snapshot day defaults to one past the last impression day.
    assert tiny_snapshot.day == 4


def test_user_aggregates_count_only_that_user_s_impressions(
    tiny_snapshot: FeatureSnapshot,
) -> None:
    assert tiny_snapshot.user_impressions.tolist()[:3] == [5.0, 2.0, 2.0]
    assert tiny_snapshot.user_clicks.tolist()[:3] == [2.0, 1.0, 1.0]
    # Users with no impressions are zeroed, not missing.
    assert tiny_snapshot.user_impressions[5] == 0.0


def test_the_seen_list_holds_clicked_items_only_and_is_sorted(
    tiny_snapshot: FeatureSnapshot,
) -> None:
    assert tiny_snapshot.seen(0).tolist() == [0, 2]
    assert tiny_snapshot.seen(1).tolist() == [0]
    assert tiny_snapshot.seen(5).tolist() == []
    # Out-of-range users get an empty array rather than an exception.
    assert tiny_snapshot.seen(-1).tolist() == []
    assert tiny_snapshot.seen(N_USERS + 10).tolist() == []


def test_a_user_s_category_shares_sum_to_about_one(tiny_snapshot: FeatureSnapshot) -> None:
    # The Dirichlet-style prior keeps the row normalized even for a user with no clicks.
    np.testing.assert_allclose(tiny_snapshot.user_cat_share.sum(axis=1), 1.0, atol=1e-5)


def test_recency_is_days_since_the_last_click_and_saturates_without_one(
    tiny_snapshot: FeatureSnapshot,
) -> None:
    # User 0 clicked item 0 (category 0) on day 0; the snapshot day is 4.
    assert tiny_snapshot.user_cat_recency[0, 0] == pytest.approx(4.0)
    # Category 1 was shown to user 0 but never clicked.
    assert tiny_snapshot.user_cat_recency[0, 1] == pytest.approx(60.0)


def test_the_popular_list_is_ordered_by_clicks_then_ctr(tiny_snapshot: FeatureSnapshot) -> None:
    popular = tiny_snapshot.popular_items.tolist()
    clicks = tiny_snapshot.item_clickers
    assert popular[0] == 0
    assert all(clicks[popular[i]] >= clicks[popular[i + 1]] for i in range(len(popular) - 1))


def test_the_popular_list_is_capped_at_the_configured_size() -> None:
    log = _log([(0, 0, item, item % 2, 0) for item in range(N_ITEMS)])
    snapshot = build_snapshot(
        log, _catalog(), n_users=N_USERS, config=SnapshotConfig(popular_list_size=3)
    )
    assert snapshot.popular_items.shape == (3,)


def test_an_explicit_snapshot_day_overrides_the_log_s_last_day() -> None:
    log = _log([(0, 0, 0, 1, 0)])
    snapshot = build_snapshot(log, _catalog(), n_users=N_USERS, snapshot_day=42)
    assert snapshot.day == 42


def test_users_beyond_n_users_feed_item_statistics_but_get_no_row() -> None:
    """New accounts still move item popularity; that is why the shilling defence lives there."""
    without = build_snapshot(_log([(0, 0, 3, 1, 0)]), _catalog(), n_users=3)
    with_stranger = build_snapshot(_log([(0, 0, 3, 1, 0), (1, 99, 3, 1, 0)]), _catalog(), n_users=3)
    assert with_stranger.n_users == 3
    assert with_stranger.user_clicks.shape == (3,)
    # The out-of-range account has no snapshot row but still lifted the item's clicker count.
    assert with_stranger.item_clickers[3] > without.item_clickers[3]


# --- robust vs naive aggregation ---


def test_robust_item_counts_weight_a_fresh_account_below_an_established_one() -> None:
    """Twenty accounts click the same item. Their tenure is the only difference between the logs."""
    fresh = _log([(user, 10 + user, 3, 1, 0) for user in range(20)])
    established = _log(
        [(user, 10 + user, 3, 1, 0) for user in range(20)]
        # The same twenty accounts, each with two further sessions of their own.
        + [(100 + 2 * user + extra, 10 + user, 4, 0, 0) for user in range(20) for extra in (0, 1)]
    )
    catalog = _catalog()
    robust_fresh = float(build_snapshot(fresh, catalog, n_users=5).item_clickers[3])
    robust_established = float(build_snapshot(established, catalog, n_users=5).item_clickers[3])
    # One session out of the three the trust threshold asks for, squared: (1/3) ** 2 each.
    assert robust_fresh == pytest.approx(20.0 / 9.0, rel=1e-3)
    assert robust_established == pytest.approx(20.0)
    assert robust_fresh < 0.2 * robust_established
    # The naive aggregation cannot tell the two logs apart at all.
    naive = SnapshotConfig(robust=False)
    assert (
        build_snapshot(fresh, catalog, n_users=5, config=naive).item_clickers[3]
        == build_snapshot(established, catalog, n_users=5, config=naive).item_clickers[3]
    )


def test_a_repeated_click_on_the_same_item_counts_once_under_robust_aggregation() -> None:
    repeated = _log([(session, 1, 3, 1, 0) for session in range(50)])
    naive = build_snapshot(repeated, _catalog(), n_users=5, config=SnapshotConfig(robust=False))
    robust = build_snapshot(repeated, _catalog(), n_users=5)
    assert naive.item_clickers[3] == pytest.approx(50.0)
    assert robust.item_clickers[3] == pytest.approx(1.0)


def test_the_naive_build_reports_a_raw_unshrunk_ctr() -> None:
    log = _log([(0, 0, 3, 1, 0), (0, 0, 4, 0, 0)])
    naive = build_snapshot(log, _catalog(), n_users=5, config=SnapshotConfig(robust=False))
    assert naive.item_ctr[3] == pytest.approx(1.0)
    assert naive.item_ctr[4] == pytest.approx(0.0)


def test_the_robust_build_shrinks_item_ctr_towards_the_global_rate() -> None:
    """A perfect 1-of-1 click rate is what a shilled item shows; the prior has to flatten it."""
    # Background traffic with a 5% click rate, plus item 3 seen once and clicked.
    rows = [(session, 0, session % 3, int(session % 20 == 0), 0) for session in range(200)]
    rows.append((999, 1, 3, 1, 0))
    log = _log(rows)
    global_ctr = float(build_snapshot(log, _catalog(), n_users=5).global_ctr[0])
    assert global_ctr < 0.1
    naive = build_snapshot(log, _catalog(), n_users=5, config=SnapshotConfig(robust=False))
    robust = build_snapshot(
        log, _catalog(), n_users=5, config=SnapshotConfig(ctr_prior_strength=100.0)
    )
    assert naive.item_ctr[3] == pytest.approx(1.0)
    assert robust.item_ctr[3] == pytest.approx(global_ctr, abs=0.02)


def test_an_item_nobody_saw_falls_back_to_the_global_ctr() -> None:
    log = _log([(0, 0, 0, 1, 0), (0, 0, 1, 0, 0)])
    for robust in (True, False):
        snapshot = build_snapshot(log, _catalog(), n_users=5, config=SnapshotConfig(robust=robust))
        assert snapshot.item_impressions[7] == 0.0
        assert snapshot.item_ctr[7] == pytest.approx(float(snapshot.global_ctr[0]))


@pytest.mark.parametrize("exponent", [1.0, 2.0, 3.0])
def test_a_higher_tenure_exponent_damps_a_fresh_account_harder(exponent: float) -> None:
    fresh = _log([(session, 10 + session, 3, 1, 0) for session in range(10)])
    snapshot = build_snapshot(
        fresh,
        _catalog(),
        n_users=5,
        config=SnapshotConfig(trust_sessions=4, tenure_exponent=exponent),
    )
    # Each attacker has one session out of four, so the weight is (1/4) ** exponent.
    assert snapshot.item_clickers[3] == pytest.approx(10.0 * 0.25**exponent)


# --- contract ---


def test_validate_accepts_a_real_snapshot_and_arrays_round_trip(
    tiny_snapshot: FeatureSnapshot,
) -> None:
    tiny_snapshot.validate()
    restored = FeatureSnapshot.from_arrays(tiny_snapshot.arrays())
    assert restored.sha256() == tiny_snapshot.sha256()
    assert restored.n_items == tiny_snapshot.n_items


def test_from_arrays_names_the_missing_arrays(tiny_snapshot: FeatureSnapshot) -> None:
    arrays = dict(tiny_snapshot.arrays())
    del arrays["item_ctr"]
    with pytest.raises(ValueError, match="item_ctr"):
        FeatureSnapshot.from_arrays(arrays)


def _mutate(snapshot: FeatureSnapshot, **changes: Any) -> FeatureSnapshot:
    return dataclasses.replace(snapshot, **changes)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("user_ctr", np.zeros(99, dtype=np.float32), "user_ctr has the wrong shape"),
        ("user_cat_ctr", np.zeros((N_USERS, 99), dtype=np.float32), "user_cat_ctr has the wrong"),
        ("item_ctr", np.zeros(99, dtype=np.float32), "item_ctr has the wrong shape"),
        ("seen_indptr", np.zeros(2, dtype=np.int64), "seen-item index is inconsistent"),
        ("popular_items", np.asarray([999], dtype=np.int32), "popular_items references unknown"),
        ("item_category", np.full(N_ITEMS, 99, dtype=np.int32), "categories are out of range"),
        ("item_ctr", np.full(N_ITEMS, 2.0, dtype=np.float32), "item_ctr exceeds 1"),
        ("item_ctr", np.full(N_ITEMS, -1.0, dtype=np.float32), "item_ctr has invalid values"),
        ("item_ctr", np.full(N_ITEMS, np.nan, dtype=np.float32), "item_ctr has invalid values"),
    ],
)
def test_the_snapshot_contract_rejects_broken_arrays(
    tiny_snapshot: FeatureSnapshot,
    field: str,
    value: npt.NDArray[np.generic],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _mutate(tiny_snapshot, **{field: value}).validate()


def test_a_non_monotone_seen_index_is_rejected(tiny_snapshot: FeatureSnapshot) -> None:
    broken = tiny_snapshot.seen_indptr.copy()
    broken[1] = broken[-1]
    broken[-1] = len(tiny_snapshot.seen_indices)
    with pytest.raises(ValueError, match="non-decreasing"):
        _mutate(tiny_snapshot, seen_indptr=broken).validate()


@pytest.mark.parametrize("scale", ["tiny", "smoke"])
def test_the_same_log_always_produces_the_same_snapshot_bytes(scale: str) -> None:
    """The snapshot hash is in the bundle manifest, so the aggregation has to be reproducible.

    DuckDB sums a group in whatever order its morsels finish in and floating-point addition is not
    associative, so an unpinned connection produces sums that differ in the last bits - and the
    popularity ``argsort`` turns a last-bit difference into a swapped pair of items.
    """
    config = world_config(scale, seed=4)
    _, catalog, interactions = generate(config)
    history = interactions.periods(0, 1)
    hashes = {build_snapshot(history, catalog, n_users=config.n_users).sha256() for _ in range(6)}
    assert len(hashes) == 1


def test_the_popularity_order_ignores_differences_below_the_rounding_floor() -> None:
    """A difference a parallel sum could invent must not reorder the list."""
    log = _log([(session, 0, session % N_ITEMS, 1, 0) for session in range(40)])
    snapshot = build_snapshot(log, _catalog(), n_users=N_USERS)
    nudged = snapshot.item_clickers.astype(np.float64)
    nudged += 1e-12
    reordered = np.lexsort((np.arange(N_ITEMS), -np.round(nudged, 6)))
    assert (
        reordered[: len(snapshot.popular_items)].tolist()
        == [
            int(item)
            for item in np.lexsort((np.arange(N_ITEMS), -np.round(snapshot.item_clickers, 6)))
        ][: len(snapshot.popular_items)]
    )


def test_the_content_hash_changes_with_any_array(tiny_snapshot: FeatureSnapshot) -> None:
    """The bundle manifest pins this hash, so it has to cover the values, not just the shapes."""
    nudged = tiny_snapshot.item_ctr.copy()
    nudged[0] = min(1.0, float(nudged[0]) + 0.01)
    assert _mutate(tiny_snapshot, item_ctr=nudged).sha256() != tiny_snapshot.sha256()


# --- side features ---


def test_side_feature_widths_match_the_tower_inputs(tiny_snapshot: FeatureSnapshot) -> None:
    user_side = user_side_features(tiny_snapshot)
    item_side = item_side_features(tiny_snapshot)
    assert user_side.shape == (N_USERS, N_CATEGORIES + 2)
    assert item_side.shape == (N_ITEMS, N_CATEGORIES + 2)
    assert user_side.dtype == np.float32
    assert item_side.dtype == np.float32


def test_item_side_features_start_with_a_category_one_hot(
    tiny_snapshot: FeatureSnapshot,
) -> None:
    item_side = item_side_features(tiny_snapshot)
    onehot = item_side[:, :N_CATEGORIES]
    np.testing.assert_allclose(onehot.sum(axis=1), 1.0)
    assert np.argmax(onehot, axis=1).tolist() == tiny_snapshot.item_category.tolist()


def test_side_features_are_finite_and_bounded(tiny_snapshot: FeatureSnapshot) -> None:
    for side in (user_side_features(tiny_snapshot), item_side_features(tiny_snapshot)):
        assert np.all(np.isfinite(side))
        assert side.min() >= 0.0


# --- assemble_features ---


def test_assembled_features_have_one_column_per_feature_name(
    tiny_snapshot: FeatureSnapshot,
) -> None:
    items = np.asarray([0, 1, 2], dtype=np.int64)
    rows = assemble_features(tiny_snapshot, 0, items, np.asarray([0.5, 0.4, 0.3]))
    assert rows.shape == (3, len(FEATURE_NAMES))
    assert rows.dtype == np.float32
    np.testing.assert_allclose(rows[:, 0], [0.5, 0.4, 0.3], rtol=1e-6)


def test_a_scalar_user_and_a_user_array_give_the_same_rows(
    tiny_snapshot: FeatureSnapshot,
) -> None:
    """Training passes an array and serving a scalar; they must not drift apart."""
    items = np.asarray([0, 1, 2, 3], dtype=np.int64)
    scores = np.asarray([0.9, 0.5, 0.1, 0.0], dtype=np.float32)
    scalar = assemble_features(tiny_snapshot, 1, items, scores)
    array = assemble_features(tiny_snapshot, np.full(items.shape, 1), items, scores)
    np.testing.assert_allclose(scalar, array)


def test_item_age_is_clipped_and_reads_from_the_catalog_launch_day() -> None:
    log = _log([(0, 0, 0, 1, 0), (0, 0, 1, 0, 0)])
    catalog = _catalog(launch=[0, 2, 0, 0, 0, 0, 0, 0])
    snapshot = build_snapshot(log, catalog, n_users=N_USERS, snapshot_day=100)
    rows = assemble_features(snapshot, 0, np.asarray([0, 1]), np.zeros(2, dtype=np.float32))
    age = FEATURE_NAMES.index("item_age_days")
    # Both ages saturate at the 60-day cap the feature uses.
    np.testing.assert_allclose(rows[:, age], [60.0, 60.0])


def test_assembled_features_are_finite_and_use_log_scaled_counts(
    tiny_snapshot: FeatureSnapshot,
) -> None:
    items = np.arange(N_ITEMS, dtype=np.int64)
    rows = assemble_features(tiny_snapshot, 0, items, np.zeros(N_ITEMS, dtype=np.float32))
    assert np.all(np.isfinite(rows))
    clickers = FEATURE_NAMES.index("item_log_clickers")
    np.testing.assert_allclose(
        rows[:, clickers], np.log1p(tiny_snapshot.item_clickers[items]), rtol=1e-6
    )


def test_an_empty_item_array_gives_an_empty_feature_matrix(
    tiny_snapshot: FeatureSnapshot,
) -> None:
    rows = assemble_features(
        tiny_snapshot, 0, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    )
    assert rows.shape == (0, len(FEATURE_NAMES))


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=50),
    robust=st.booleans(),
    trust_sessions=st.integers(min_value=1, max_value=6),
)
def test_a_snapshot_of_a_generated_world_always_satisfies_its_own_contract(
    seed: int, robust: bool, trust_sessions: int
) -> None:
    config = world_config("tiny", seed=seed, n_users=40, n_items=90, pool_size=20, slots=5)
    _, catalog, interactions = generate(config)
    snapshot = build_snapshot(
        interactions,
        catalog,
        n_users=config.n_users,
        config=SnapshotConfig(robust=robust, trust_sessions=trust_sessions),
    )
    snapshot.validate()
    assert snapshot.n_users == config.n_users
    assert snapshot.n_items == config.n_items
    assert float(snapshot.item_ctr.max()) <= 1.0
    assert int(snapshot.seen_indptr[-1]) == snapshot.seen_indices.shape[0]
    rows = assemble_features(
        snapshot,
        0,
        np.arange(snapshot.n_items, dtype=np.int64),
        np.zeros(snapshot.n_items, dtype=np.float32),
    )
    assert np.all(np.isfinite(rows))
