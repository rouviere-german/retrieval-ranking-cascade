"""The LambdaMART reranker: its training-set builder, IPS row weights and integrity gate."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval_ranking_cascade.data.generator import Interactions
from retrieval_ranking_cascade.evaluation.propensity import PropensityModel
from retrieval_ranking_cascade.ranking.features import FEATURE_NAMES
from retrieval_ranking_cascade.ranking.ranker import (
    Ranker,
    RankerConfig,
    RankerDataset,
    RankerIntegrityError,
    build_ranker_dataset,
    ips_row_weights,
)

pytestmark = pytest.mark.unit

SLOTS = 6


def _propensity(slots: int = SLOTS, eta: float = 1.0, max_weight: float = 10.0) -> PropensityModel:
    ranks = np.arange(1, slots + 1, dtype=np.float64)
    return PropensityModel(
        ranks**-eta,
        np.full(slots, 1_000, dtype=np.int64),
        np.full(slots, 100, dtype=np.int64),
        max_weight,
    )


def _label_sessions(n_sessions: int = 24, seed: int = 0) -> Interactions:
    """Label-period sessions with a mix of clicked and unclicked slots."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int, int, int, int]] = []
    for session in range(n_sessions):
        user = session % 8
        items = rng.choice(60, size=SLOTS, replace=False)
        clicks = np.zeros(SLOTS, dtype=int)
        clicks[rng.integers(0, SLOTS)] = 1
        for position, (item, click) in enumerate(zip(items, clicks, strict=True), start=1):
            rows.append((session, user, int(item), position, int(click)))
    n = len(rows)
    return Interactions(
        session_id=np.asarray([r[0] for r in rows], dtype=np.int32),
        user_id=np.asarray([r[1] for r in rows], dtype=np.int32),
        item_id=np.asarray([r[2] for r in rows], dtype=np.int32),
        position=np.asarray([r[3] for r in rows], dtype=np.int8),
        clicked=np.asarray([r[4] for r in rows], dtype=np.int8),
        randomized=np.zeros(n, dtype=np.bool_),
        day=np.ones(n, dtype=np.int16),
        period=np.ones(n, dtype=np.int8),
    )


# --- config ---


def test_the_config_round_trips_and_builds_lightgbm_params() -> None:
    config = RankerConfig(num_boost_round=10, num_leaves=7, seed=3)
    assert config.to_dict()["num_leaves"] == 7
    params = config.lgb_params()
    assert params["objective"] == "lambdarank"
    assert params["deterministic"] is True
    assert params["seed"] == 3
    assert "lambdarank_position_bias_regularization" not in params


def test_position_debias_adds_the_unbiased_lambdarank_knob() -> None:
    params = RankerConfig(position_debias=True).lgb_params()
    assert params["lambdarank_position_bias_regularization"] == 0.0


# --- IPS row weights ---


def test_shown_rows_get_clipped_inverse_propensities_and_unshown_rows_get_one() -> None:
    positions = np.asarray([0, 2, 5, SLOTS], dtype=np.int32)
    weights = ips_row_weights(positions, _propensity(), SLOTS)
    # 0-based positions; the last row is the "never shown" bucket.
    np.testing.assert_allclose(weights, [1.0, 3.0, 6.0, 1.0])


def test_weights_are_all_one_when_nothing_was_shown() -> None:
    positions = np.full(4, SLOTS, dtype=np.int32)
    np.testing.assert_allclose(ips_row_weights(positions, _propensity(), SLOTS), 1.0)


def test_clipping_bounds_the_weight_of_the_lowest_slot() -> None:
    weights = ips_row_weights(
        np.asarray([SLOTS - 1], dtype=np.int32), _propensity(max_weight=2.0), SLOTS
    )
    assert weights[0] == pytest.approx(2.0)


@settings(max_examples=60, deadline=None)
@given(
    positions=st.lists(st.integers(min_value=0, max_value=SLOTS), min_size=1, max_size=40),
    max_weight=st.floats(min_value=1.0, max_value=25.0),
)
def test_every_row_weight_is_between_one_and_max_weight(
    positions: list[int], max_weight: float
) -> None:
    weights = ips_row_weights(
        np.asarray(positions, dtype=np.int32), _propensity(max_weight=max_weight), SLOTS
    )
    assert weights.min() >= 1.0 - 1e-12
    assert weights.max() <= max_weight + 1e-12


# --- dataset ---


@pytest.fixture(scope="module")
def dataset(tiny_training: Any) -> RankerDataset:
    result: RankerDataset = tiny_training.ranker_dataset
    return result


def test_the_dataset_groups_sum_to_the_row_count(dataset: RankerDataset) -> None:
    assert int(dataset.groups.sum()) == dataset.labels.shape[0]
    assert dataset.n_queries == dataset.groups.shape[0] > 0
    assert dataset.features.shape == (dataset.labels.shape[0], len(FEATURE_NAMES))


def test_every_query_has_both_a_clicked_and_an_unclicked_row(dataset: RankerDataset) -> None:
    """A group with no contrast contributes no lambda gradient, so it is dropped at build time."""
    start = 0
    for size in dataset.groups:
        labels = dataset.labels[start : start + int(size)]
        assert 0 < int(labels.sum()) < int(size)
        start += int(size)


def test_extra_negatives_are_unshown_rows_at_the_slots_bucket(dataset: RankerDataset) -> None:
    unshown = dataset.positions >= 1
    assert bool(np.any(unshown))
    # Extra negatives are never labelled 1 and always carry weight 1.
    extra = dataset.positions == dataset.positions.max()
    assert int(dataset.labels[extra].sum()) == 0


def test_ips_weights_are_above_one_for_rows_shown_low_on_the_page(
    dataset: RankerDataset,
) -> None:
    assert float(dataset.weights.min()) == pytest.approx(1.0)
    assert float(dataset.weights.max()) > 1.0


def test_a_mismatched_dataset_is_refused() -> None:
    features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)
    good: dict[str, Any] = {
        "features": features,
        "labels": np.zeros(4, dtype=np.int32),
        "groups": np.asarray([4], dtype=np.int32),
        "positions": np.zeros(4, dtype=np.int32),
        "weights": np.ones(4, dtype=np.float64),
        "users": np.zeros(4, dtype=np.int32),
        "items": np.arange(4, dtype=np.int32),
    }
    with pytest.raises(ValueError, match="mismatched lengths"):
        RankerDataset(**{**good, "labels": np.zeros(3, dtype=np.int32)})
    with pytest.raises(ValueError, match="group sizes"):
        RankerDataset(**{**good, "groups": np.asarray([3], dtype=np.int32)})


def test_building_a_dataset_without_a_propensity_model_is_refused(tiny_training: Any) -> None:
    result = tiny_training
    with pytest.raises(ValueError, match="PropensityModel"):
        build_ranker_dataset(
            snapshot=result.train_snapshot,
            label_sessions=result.period(1),
            user_vectors=np.zeros((result.world_config.n_users, 4), dtype=np.float32),
            item_vectors=np.zeros((result.world_config.n_items, 4), dtype=np.float32),
            retrieved=np.zeros((result.world_config.n_users, 1), dtype=np.int64),
            config=RankerConfig(ips_weights=True),
            slots=result.world_config.slots,
            propensity=None,
        )


def test_a_label_period_with_no_usable_session_is_refused(tiny_training: Any) -> None:
    result = tiny_training
    label = result.period(1)
    all_clicked = Interactions(**{**label.arrays(), "clicked": np.ones(len(label), dtype=np.int8)})
    with pytest.raises(ValueError, match="no label-period session"):
        build_ranker_dataset(
            snapshot=result.train_snapshot,
            label_sessions=all_clicked,
            user_vectors=np.zeros((result.world_config.n_users, 4), dtype=np.float32),
            item_vectors=np.zeros((result.world_config.n_items, 4), dtype=np.float32),
            retrieved=np.zeros((result.world_config.n_users, 1), dtype=np.int64),
            config=RankerConfig(ips_weights=False),
            slots=result.world_config.slots,
        )


def test_turning_ips_off_gives_uniform_weights(tiny_training: Any) -> None:
    result = tiny_training
    plain = build_ranker_dataset(
        snapshot=result.train_snapshot,
        label_sessions=result.period(1),
        user_vectors=np.asarray(
            result.bundle.model.user.embed(
                np.arange(result.world_config.n_users), result.bundle.user_side
            )[0],
            dtype=np.float32,
        ),
        item_vectors=result.bundle.item_vectors,
        retrieved=np.full((result.world_config.n_users, 1), -1, dtype=np.int64),
        config=RankerConfig(ips_weights=False, extra_negatives=0),
        slots=result.world_config.slots,
    )
    np.testing.assert_allclose(plain.weights, 1.0)


def test_a_session_never_proposes_an_item_the_user_already_clicked(tiny_training: Any) -> None:
    """Extra negatives come from the retrieved pool minus the shown slate and the seen list."""
    result = tiny_training
    dataset = result.ranker_dataset
    snapshot = result.train_snapshot
    for user in np.unique(dataset.users)[:20]:
        rows = dataset.users == user
        proposed = set(dataset.items[rows & (dataset.positions == dataset.positions.max())])
        assert not proposed & set(snapshot.seen(int(user)).tolist())


# --- training and serving ---


def test_a_trained_ranker_exposes_its_trees_features_and_importances(tiny_training: Any) -> None:
    ranker: Ranker = tiny_training.bundle.ranker
    assert ranker.num_trees > 0
    importance = ranker.feature_importance()
    assert set(importance) == set(FEATURE_NAMES)
    assert sum(importance.values()) > 0.0


def test_predictions_are_finite_one_per_row_and_order_dependent_on_features(
    tiny_training: Any,
) -> None:
    ranker: Ranker = tiny_training.bundle.ranker
    dataset: RankerDataset = tiny_training.ranker_dataset
    rows = dataset.features[:200]
    scores = ranker.predict(rows)
    assert scores.shape == (rows.shape[0],)
    assert np.all(np.isfinite(scores))
    assert float(np.ptp(scores)) > 0.0


def test_predicting_nothing_returns_an_empty_array(tiny_training: Any) -> None:
    ranker: Ranker = tiny_training.bundle.ranker
    assert ranker.predict(np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)).shape == (0,)


def test_predictions_are_deterministic(tiny_training: Any) -> None:
    ranker: Ranker = tiny_training.bundle.ranker
    rows = tiny_training.ranker_dataset.features[:64]
    np.testing.assert_allclose(ranker.predict(rows), ranker.predict(rows), rtol=0)


def test_a_feature_matrix_of_the_wrong_width_is_refused(tiny_training: Any) -> None:
    ranker: Ranker = tiny_training.bundle.ranker
    with pytest.raises(ValueError, match=f"{len(FEATURE_NAMES)} columns"):
        ranker.predict(np.zeros((3, len(FEATURE_NAMES) - 1), dtype=np.float32))
    with pytest.raises(ValueError, match="columns"):
        ranker.predict(np.zeros(len(FEATURE_NAMES), dtype=np.float32))


def test_each_thread_gets_its_own_parsed_booster(tiny_training: Any) -> None:
    """Serving is multi-threaded; a shared booster behind a mutex would serialize stage 2."""
    import threading

    ranker: Ranker = tiny_training.bundle.ranker
    rows = tiny_training.ranker_dataset.features[:32]
    expected = ranker.predict(rows)
    seen: list[npt.NDArray[np.float64]] = []
    # The booster objects themselves, not their ids: a thread-local booster becomes garbage when its
    # thread ends, and CPython happily reuses the address, so ids alone collide.
    boosters: list[Any] = []
    lock = threading.Lock()

    def work() -> None:
        scores = ranker.predict(rows)
        with lock:
            seen.append(scores)
            boosters.append(ranker._thread_booster())

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(seen) == 4
    for scores in seen:
        np.testing.assert_allclose(scores, expected, rtol=0)
    assert len({id(booster) for booster in boosters}) == 4


# --- integrity ---


def test_the_model_text_round_trips_through_a_hash_verified_load(tiny_training: Any) -> None:
    ranker: Ranker = tiny_training.bundle.ranker
    text = ranker.to_text()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    restored = Ranker.from_text(text, expected_sha256=digest)
    rows = tiny_training.ranker_dataset.features[:16]
    np.testing.assert_allclose(restored.predict(rows), ranker.predict(rows), rtol=1e-12)


def test_a_tampered_model_text_never_reaches_the_lightgbm_parser(tiny_training: Any) -> None:
    ranker: Ranker = tiny_training.bundle.ranker
    text = ranker.to_text()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    with pytest.raises(RankerIntegrityError, match="sha256"):
        Ranker.from_text(text.replace("leaf_value", "leaf_valve", 1), expected_sha256=digest)


def test_a_model_whose_features_do_not_match_is_refused(tiny_training: Any) -> None:
    """A booster trained on other columns would silently reinterpret the feature matrix."""
    ranker: Ranker = tiny_training.bundle.ranker
    renamed = ranker.to_text().replace("tower_score", "not_tower_score", 1)
    digest = hashlib.sha256(renamed.encode("utf-8")).hexdigest()
    with pytest.raises(RankerIntegrityError, match="feature names"):
        Ranker.from_text(renamed, expected_sha256=digest)
