"""The synthetic world: determinism, the contracts it must satisfy, and the click model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval_ranking_cascade.data.generator import (
    PERIOD_HISTORY,
    PERIOD_LABEL,
    PERIOD_TEST,
    SCALES,
    Interactions,
    generate,
    world_config,
)
from retrieval_ranking_cascade.ml.contracts import (
    catalog_schema,
    impressions_schema,
    validate,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def tiny() -> tuple[object, object, Interactions]:
    return generate(world_config("tiny"))


def test_generation_is_a_pure_function_of_the_seed() -> None:
    config = world_config("tiny", seed=11)
    first = generate(config)[2]
    second = generate(config)[2]
    assert first.sha256() == second.sha256()
    other = generate(world_config("tiny", seed=12))[2]
    assert other.sha256() != first.sha256()


def test_the_batch_size_is_part_of_the_data_identity() -> None:
    """The draws are shaped per batch, so the batch size changes the log - and is in the config.

    Keeping ``chunk_users`` in ``WorldConfig`` is what makes that visible: it lands in the run
    manifest, so a change to a memory knob can never quietly produce a different dataset under the
    same recorded parameters.
    """
    whole = generate(world_config("tiny", seed=3, chunk_users=10_000))[2]
    chunked = generate(world_config("tiny", seed=3, chunk_users=64))[2]
    assert whole.sha256() != chunked.sha256()
    again = generate(world_config("tiny", seed=3, chunk_users=64))[2]
    assert again.sha256() == chunked.sha256()
    with pytest.raises(ValueError, match="chunk_users"):
        world_config("tiny", chunk_users=0)


def test_log_and_catalog_satisfy_their_contracts(tiny: tuple[object, object, Interactions]) -> None:
    _, catalog, interactions = tiny
    config = world_config("tiny")
    validate(
        interactions.to_frame(),
        impressions_schema(n_users=config.n_users, n_items=config.n_items, slots=config.slots),
    )
    validate(
        catalog.to_frame(),  # type: ignore[attr-defined]
        catalog_schema(n_items=config.n_items, n_categories=config.n_categories),
    )


def test_every_user_has_exactly_one_test_and_one_label_session(
    tiny: tuple[object, object, Interactions],
) -> None:
    _, _, interactions = tiny
    for period in (PERIOD_LABEL, PERIOD_TEST):
        rows = interactions.periods(period)
        users, counts = np.unique(rows.user_id, return_counts=True)
        sessions = np.unique(rows.session_id)
        assert users.size == world_config("tiny").n_users
        assert sessions.size == users.size
        assert counts.min() >= 1
    assert len(interactions.periods(PERIOD_HISTORY)) > 0


def test_a_user_is_never_shown_an_item_they_already_clicked(
    tiny: tuple[object, object, Interactions],
) -> None:
    _, _, interactions = tiny
    order = np.lexsort((interactions.position, interactions.session_id))
    rows = interactions.select(order)
    clicked: set[tuple[int, int]] = set()
    seen_after_click = 0
    for user, item, click in zip(rows.user_id, rows.item_id, rows.clicked, strict=True):
        key = (int(user), int(item))
        if key in clicked:
            seen_after_click += 1
        if click:
            clicked.add(key)
    assert seen_after_click == 0


def test_the_randomized_slice_is_a_whole_number_of_sessions(
    tiny: tuple[object, object, Interactions],
) -> None:
    _, _, interactions = tiny
    frame = interactions.to_frame()
    per_session = frame.groupby("session_id")["randomized"].nunique()
    assert set(per_session.unique()) == {1}
    assert 0.0 < float(interactions.randomized.mean()) < 0.2


def test_clicks_decay_with_position_outside_the_randomized_slice(
    tiny: tuple[object, object, Interactions],
) -> None:
    """The logging policy plus the examination model must produce a decaying click curve."""
    _, _, interactions = tiny
    rows = interactions.select(~interactions.randomized)
    rates = [
        float(rows.clicked[rows.position == rank].mean())
        for rank in range(1, world_config("tiny").slots + 1)
    ]
    assert rates[0] > rates[-1] * 3
    # Monotone in the large: the first half always beats the second half.
    half = len(rates) // 2
    assert float(np.mean(rates[:half])) > float(np.mean(rates[half:]))


def _relevance_slope_by_rank(world: object, rows: Interactions, slots: int) -> float:
    """Least-squares slope of mean true relevance against rank."""
    relevance = world.relevance(rows.user_id, rows.item_id)  # type: ignore[attr-defined]
    ranks = []
    means = []
    for rank in range(1, slots + 1):
        mask = rows.position == rank
        if mask.any():
            ranks.append(float(rank))
            means.append(float(relevance[mask].mean()))
    x = np.asarray(ranks) - np.mean(ranks)
    y = np.asarray(means)
    return float(np.dot(x, y) / np.dot(x, x))


def test_randomization_removes_the_relevance_trend_across_positions() -> None:
    """This is the property the propensity estimator depends on, so it is tested on its own.

    In the logged slice the policy puts its best guesses first, so true relevance falls with rank.
    In the randomized slice the order is uniform, so that slope must essentially vanish. The test
    runs at ``small`` scale because at ``tiny`` the randomized slice is a few hundred rows and the
    sampling noise is larger than the effect being measured.
    """
    config = world_config("small")
    world, _, interactions = generate(config)
    randomized = _relevance_slope_by_rank(
        world, interactions.select(interactions.randomized), config.slots
    )
    logged = _relevance_slope_by_rank(
        world, interactions.select(~interactions.randomized), config.slots
    )
    assert logged < 0, "the logging policy should rank more relevant items higher"
    assert abs(randomized) < abs(logged) / 5.0, (
        f"randomized slope {randomized:.5f} is not flat next to logged slope {logged:.5f}"
    )


@settings(max_examples=25, deadline=None)
@given(
    users=st.lists(st.integers(min_value=0, max_value=299), min_size=1, max_size=8),
    items=st.lists(st.integers(min_value=0, max_value=799), min_size=1, max_size=8),
)
def test_relevance_is_a_probability_and_matches_the_row_form(
    users: list[int], items: list[int]
) -> None:
    world, _, _ = generate(world_config("tiny"))
    size = min(len(users), len(items))
    u = np.asarray(users[:size])
    i = np.asarray(items[:size])
    pointwise = world.relevance(u, i)
    assert np.all((pointwise >= 0.0) & (pointwise <= 1.0))
    rows = world.relevance_rows(u)
    assert rows.shape == (size, 800)
    np.testing.assert_allclose(pointwise, rows[np.arange(size), i], rtol=1e-6)


def test_examination_is_decreasing_and_starts_at_one(
    tiny: tuple[object, object, Interactions],
) -> None:
    world, _, _ = tiny
    values = world.examination(np.arange(1, 11))  # type: ignore[attr-defined]
    assert values[0] == pytest.approx(1.0)
    assert np.all(np.diff(values) < 0)
    with pytest.raises(ValueError, match="1-based"):
        world.examination([0])  # type: ignore[attr-defined]


def test_select_accepts_a_mask_or_an_index_array(
    tiny: tuple[object, object, Interactions],
) -> None:
    _, _, interactions = tiny
    mask = interactions.clicked == 1
    by_mask = interactions.select(mask)
    by_index = interactions.select(np.flatnonzero(mask))
    assert by_mask.sha256() == by_index.sha256()
    assert len(by_mask) == int(mask.sum())


def test_npz_round_trip_preserves_the_log(tiny: tuple[object, object, Interactions]) -> None:
    import io

    _, _, interactions = tiny
    with np.load(io.BytesIO(interactions.to_npz_bytes()), allow_pickle=False) as archive:
        restored = Interactions(**{name: archive[name] for name in archive.files})
    assert restored.sha256() == interactions.sha256()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("n_users", 1, "2 users"),
        ("n_categories", 1, "2 users"),
        ("slots", 0, "slots"),
        ("min_sessions", 2, "min_sessions"),
        ("randomized_fraction", 1.5, "randomized_fraction"),
        ("max_sessions", 2, "max_sessions"),
    ],
)
def test_impossible_world_configurations_are_refused(field: str, value: object, match: str) -> None:
    overrides: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match=match):
        world_config("tiny", **overrides)


def test_unknown_scale_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown scale"):
        world_config("enormous")
    assert set(SCALES) == {"tiny", "smoke", "small", "full"}


def test_click_rate_is_zero_for_an_empty_selection(
    tiny: tuple[object, object, Interactions],
) -> None:
    _, _, interactions = tiny
    empty = interactions.select(np.zeros(len(interactions), dtype=bool))
    assert len(empty) == 0
    assert empty.click_rate == 0.0
