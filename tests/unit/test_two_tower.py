"""The hand-derived two-tower gradients, checked against finite differences and by invariant."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from retrieval_ranking_cascade.retrieval.two_tower import (
    Adam,
    Tower,
    TowerConfig,
    TwoTowerModel,
    gradient_check,
    sampled_softmax,
    train_two_tower,
)

pytestmark = pytest.mark.unit

FArray = npt.NDArray[np.floating[Any]]
IArray = npt.NDArray[np.integer[Any]]


class Batch(TypedDict):
    """One training batch, shaped exactly like :meth:`TwoTowerModel.loss_and_grads`' arguments."""

    users: IArray
    user_side: FArray
    positives: IArray
    negatives: IArray
    item_side: FArray
    log_q: FArray | None


def _model(
    *,
    batch: int = 3,
    negatives: int = 2,
    id_dim: int = 3,
    hidden: int = 4,
    out: int = 3,
    side: int = 2,
    logq: bool = True,
    seed: int = 0,
) -> tuple[TwoTowerModel, Batch]:
    """A float64 model small enough for a full finite-difference sweep, plus one batch."""
    rng = np.random.default_rng(seed)
    n_users = batch + 1
    n_items = batch + negatives + 1
    config = TowerConfig(
        id_dim=id_dim,
        hidden_dim=hidden,
        output_dim=out,
        batch_size=batch,
        uniform_negatives=negatives,
        epochs=1,
        logq_correction=logq,
        seed=seed,
    )
    model = TwoTowerModel.initialize(
        n_users=n_users,
        n_items=n_items,
        user_side_dim=side,
        item_side_dim=side,
        config=config,
        dtype=np.float64,
    )
    # Id embeddings start at zero by design, which would hide their gradient path.
    for tower in (model.user, model.item):
        tower.params["emb"][:] = rng.normal(size=tower.params["emb"].shape) * 0.4
    positives = np.arange(1, batch + 1)
    negatives_ids = np.arange(batch + 1, batch + 1 + negatives)
    candidates = np.concatenate([positives, negatives_ids])
    item_side = rng.normal(size=(n_items, side))
    batch_data: Batch = {
        "users": np.arange(batch),
        "user_side": rng.normal(size=(n_users, side))[np.arange(batch)],
        "positives": positives,
        "negatives": negatives_ids,
        "item_side": item_side[candidates],
        "log_q": np.log(np.full(candidates.shape[0], 0.3)) if logq else None,
    }
    return model, batch_data


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    batch=st.integers(min_value=2, max_value=4),
    negatives=st.integers(min_value=0, max_value=3),
    id_dim=st.integers(min_value=1, max_value=3),
    hidden=st.integers(min_value=4, max_value=6),
    out=st.integers(min_value=2, max_value=3),
    logq=st.booleans(),
    seed=st.integers(min_value=0, max_value=400),
)
def test_analytic_gradients_match_central_differences(
    batch: int, negatives: int, id_dim: int, hidden: int, out: int, logq: bool, seed: int
) -> None:
    """Every analytic gradient, over a range of shapes and both logQ settings.

    Two regimes are excluded, because a central difference is not a valid reference in either, and
    each has its own test below:

    * a near-zero pre-normalization norm - the normalization has a near-singularity there
      (``test_a_dead_relu_layer_gives_zero_embeddings_and_finite_gradients``);
    * a collapsed, or nearly collapsed, embedding space. When the candidate logits almost tie, the
      softmax sits at the uniform point, the gradient that reaches the weights is tiny, and every
      entry falls below the floor a central difference can resolve - so the sweep would compare
      nothing at all. The precondition below is on the *cosine* spread rather than on the gradients,
      so it states something about the example rather than about the answer: a spread of 0.05 in
      cosine is a logit spread of 1.0 at this temperature, which is a real signal. The degenerate
      end of that regime has its own tests
      (``test_a_rank_one_hidden_layer_with_zero_output_bias_collapses_the_space`` and
      ``test_a_collapsed_space_gives_a_loss_of_exactly_log_c``).
    """
    model, data = _model(
        batch=batch,
        negatives=negatives,
        id_dim=id_dim,
        hidden=hidden,
        out=out,
        logq=logq,
        seed=seed,
    )
    candidates = np.concatenate([data["positives"], data["negatives"]])
    user_cache = model.user.forward(data["users"], data["user_side"])
    item_cache = model.item.forward(candidates, data["item_side"])
    assume(float(min(user_cache.norm.min(), item_cache.norm.min())) > 1e-3)
    assume(float(np.ptp(user_cache.y @ item_cache.y.T)) > 0.05)
    result = gradient_check(model, **data)
    assert result.checked > 0, "the sweep compared nothing"
    assert result.max_relative_error < 1e-4, (
        f"worst relative error {result.max_relative_error:.2e} at {result.worst_entry} "
        f"({result.checked} entries checked, {result.skipped} below the noise floor)"
    )


def test_tower_output_is_unit_norm_where_the_layer_is_alive() -> None:
    model, data = _model(hidden=32, seed=4)
    cache = model.user.forward(data["users"], data["user_side"])
    assert np.all(cache.norm > 1e-3), "a healthy tower should not produce a near-zero embedding"
    np.testing.assert_allclose(np.linalg.norm(cache.y, axis=1), 1.0, rtol=1e-9)
    # The bound holds unconditionally, alive or dead.
    assert np.all(np.linalg.norm(cache.y, axis=1) <= 1.0 + 1e-12)


def test_a_dead_relu_layer_gives_zero_embeddings_and_finite_gradients() -> None:
    """Forcing every hidden unit negative must not produce NaNs or 1e12-scale gradients."""
    model, data = _model(hidden=4, seed=7)
    for tower in (model.user, model.item):
        tower.params["w1"][:] = 0.0
        tower.params["b1"][:] = -1.0
    cache = model.user.forward(data["users"], data["user_side"])
    assert np.all(cache.h == 0.0)
    np.testing.assert_allclose(np.linalg.norm(cache.y, axis=1), 0.0, atol=1e-9)
    loss, grads = model.loss_and_grads(**data)
    assert np.isfinite(loss)
    for tower_grads in grads.values():
        for name, value in tower_grads.items():
            array = value[1] if name == "emb" else value
            assert np.all(np.isfinite(array)), f"{name} gradient is not finite"
            assert float(np.abs(array).max()) < 1e6, f"{name} gradient exploded"


def test_a_rank_one_hidden_layer_with_zero_output_bias_collapses_the_space() -> None:
    """A real degeneracy, worth naming: normalization erases the only varying degree of freedom.

    If all but one hidden unit is dead, ``h`` is a single non-negative scalar times a fixed basis
    vector, so ``z = c * w2[0] + b2``. With ``b2 == 0`` that is a positive multiple of a fixed
    vector, and normalizing it returns the same unit vector whatever ``c`` is - so every entity gets
    the *same* embedding, every logit ties, and the loss is exactly ``log(C)``. Nothing is
    numerically wrong; the model has lost its only signal. A non-zero ``b2`` breaks the collapse.

    The tower here is built by hand rather than trained, because the point is the arithmetic.
    """
    hand_built = Tower(
        {
            # Three entities with distinct, positive id embeddings.
            "emb": np.asarray([[0.5], [1.0], [2.0]]),
            # Only the first hidden unit is reachable; the second is switched off by its bias.
            "w1": np.asarray([[1.0, 0.0], [0.0, 0.0]]),
            "b1": np.asarray([0.0, -1.0]),
            "w2": np.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
            "b2": np.zeros(3),
        }
    )
    ids = np.arange(3)
    side = np.zeros((3, 1))
    cache = hand_built.forward(ids, side)
    assert np.allclose(cache.h[:, 1], 0.0)
    np.testing.assert_allclose(cache.h[:, 0], [0.5, 1.0, 2.0])
    # Distinct hidden activations, identical embeddings.
    assert np.unique(np.round(cache.y, 9), axis=0).shape[0] == 1

    hand_built.params["b2"][:] = np.asarray([0.5, 0.0, 0.0])
    recovered = hand_built.forward(ids, side).y
    assert np.unique(np.round(recovered, 9), axis=0).shape[0] == 3


def test_a_collapsed_space_gives_a_loss_of_exactly_log_c() -> None:
    """The consequence of the collapse above, on the loss the trainer actually minimizes."""
    model, data = _model(batch=4, negatives=0, hidden=3, out=3, seed=2)
    for tower in (model.user, model.item):
        tower.params["w1"][:, 1:] = 0.0
        tower.params["b1"][1:] = -1.0
        tower.params["b2"][:] = 0.0
    loss, _ = model.loss_and_grads(**data)
    candidates = data["positives"].shape[0] + data["negatives"].shape[0]
    assert loss == pytest.approx(float(np.log(candidates)))


def test_embed_chunking_gives_the_same_vectors() -> None:
    model, _ = _model(batch=4)
    rng = np.random.default_rng(1)
    ids = np.arange(model.item.n_entities)
    side = rng.normal(size=(ids.shape[0], model.item.side_dim))
    whole, norms = model.item.embed(ids, side, chunk=1024)
    pieces, piece_norms = model.item.embed(ids, side, chunk=2)
    np.testing.assert_allclose(whole, pieces, rtol=1e-12)
    np.testing.assert_allclose(norms, piece_norms, rtol=1e-12)


def test_embed_of_nothing_returns_empty_arrays_of_the_right_width() -> None:
    model, _ = _model()
    vectors, norms = model.item.embed(np.empty(0, dtype=np.int64), np.empty((0, 2)))
    assert vectors.shape == (0, model.output_dim)
    assert norms.shape == (0,)


def test_sampled_softmax_equals_log_c_when_every_candidate_looks_identical() -> None:
    """With identical candidate embeddings all logits tie, so the loss is exactly log(C)."""
    batch, candidates = 4, 7
    user_y = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (batch, 1))
    item_y = np.tile(np.asarray([[0.0, 1.0, 0.0]]), (candidates, 1))
    positives = np.arange(batch)
    candidate_ids = np.arange(candidates)
    loss, d_user, d_item = sampled_softmax(
        user_y, item_y, positives, candidate_ids, None, temperature=0.05
    )
    assert loss == pytest.approx(float(np.log(candidates)))
    assert d_user.shape == user_y.shape
    assert d_item.shape == item_y.shape


def test_sampled_softmax_masks_duplicate_positives() -> None:
    """A repeated positive is not a negative for the row that owns it."""
    rng = np.random.default_rng(2)
    user_y = rng.normal(size=(2, 3))
    user_y /= np.linalg.norm(user_y, axis=1, keepdims=True)
    item_y = rng.normal(size=(3, 3))
    item_y /= np.linalg.norm(item_y, axis=1, keepdims=True)
    positives = np.asarray([5, 7], dtype=np.int64)
    # Candidate 2 repeats row 0's positive; masking it must lower row 0's loss.
    with_duplicate = np.asarray([5, 7, 5], dtype=np.int64)
    loss_masked, _, _ = sampled_softmax(
        user_y, item_y, positives, with_duplicate, None, temperature=0.1
    )
    distinct = np.asarray([5, 7, 9], dtype=np.int64)
    loss_plain, _, _ = sampled_softmax(user_y, item_y, positives, distinct, None, temperature=0.1)
    assert loss_masked < loss_plain


def test_sampled_softmax_requires_the_positives_first() -> None:
    user_y = np.eye(2)
    item_y = np.eye(2)
    with pytest.raises(ValueError, match="first B candidates"):
        sampled_softmax(
            user_y,
            item_y,
            np.asarray([1, 0], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            None,
            temperature=0.1,
        )


def test_a_uniform_logq_is_a_no_op_but_a_varying_one_is_not() -> None:
    """Subtracting a constant from every logit cannot change a softmax; that is the whole point.

    The correction only does work when candidates differ in sampling frequency, which is exactly
    when popular items would otherwise be over-represented as negatives.
    """
    model, data = _model(logq=True)
    plain: Batch = {**data, "log_q": None}
    loss_plain, _ = model.loss_and_grads(**plain)
    loss_uniform, _ = model.loss_and_grads(**data)
    assert loss_uniform == pytest.approx(loss_plain)

    n_candidates = data["positives"].shape[0] + data["negatives"].shape[0]
    varying: Batch = {**data, "log_q": np.log(np.linspace(0.05, 0.9, n_candidates))}
    loss_varying, _ = model.loss_and_grads(**varying)
    assert loss_varying != pytest.approx(loss_plain)


def test_lazy_adam_matches_a_dense_update_on_touched_rows() -> None:
    """The embedding table is updated row-wise; the arithmetic must equal the dense form."""
    model, _ = _model()
    tower = model.user
    dense_copy = Tower({name: value.copy() for name, value in tower.params.items()})
    lazy = Adam(tower, learning_rate=0.01)
    dense = Adam(dense_copy, learning_rate=0.01)
    rows = np.asarray([0, 2])
    row_grad = np.full((rows.shape[0], tower.id_dim), 0.5)
    other = {
        "w1": np.zeros_like(tower.params["w1"]),
        "b1": np.zeros_like(tower.params["b1"]),
        "w2": np.zeros_like(tower.params["w2"]),
        "b2": np.zeros_like(tower.params["b2"]),
    }
    dense_grad = np.zeros_like(tower.params["emb"])
    dense_grad[rows] = row_grad
    for _ in range(3):
        lazy.step({"emb": (rows, row_grad), **other})
        dense.step({"emb": (np.arange(dense_copy.n_entities), dense_grad), **other})
    np.testing.assert_allclose(
        tower.params["emb"][rows], dense_copy.params["emb"][rows], rtol=1e-10
    )
    # Rows that never received a gradient are untouched by the lazy optimizer.
    untouched = np.asarray([1, 3])
    np.testing.assert_allclose(
        tower.params["emb"][untouched], dense_copy.params["emb"][untouched], rtol=1e-10
    )


def test_training_reduces_the_loss_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    n_users, n_items = 40, 60
    users = rng.integers(0, n_users, size=600, dtype=np.int64)
    items = rng.integers(0, n_items, size=600, dtype=np.int64)
    user_side = rng.normal(size=(n_users, 3)).astype(np.float32)
    item_side = rng.normal(size=(n_items, 3)).astype(np.float32)
    config = TowerConfig(epochs=4, batch_size=64, uniform_negatives=64, seed=5)
    first_model, first = train_two_tower(
        users=users,
        items=items,
        user_side=user_side,
        item_side=item_side,
        known_items=np.unique(items),
        config=config,
    )
    second_model, second = train_two_tower(
        users=users,
        items=items,
        user_side=user_side,
        item_side=item_side,
        known_items=np.unique(items),
        config=config,
    )
    assert first.epoch_loss[-1] < first.epoch_loss[0]
    assert first.epoch_loss == second.epoch_loss
    assert first.steps == second.steps > 0
    for name, value in first_model.to_arrays().items():
        np.testing.assert_allclose(value, second_model.to_arrays()[name], rtol=0)


def test_on_epoch_callback_sees_every_epoch() -> None:
    rng = np.random.default_rng(0)
    seen: list[int] = []
    train_two_tower(
        users=rng.integers(0, 10, size=100, dtype=np.int64),
        items=rng.integers(0, 12, size=100, dtype=np.int64),
        user_side=rng.normal(size=(10, 2)).astype(np.float32),
        item_side=rng.normal(size=(12, 2)).astype(np.float32),
        known_items=np.arange(12),
        config=TowerConfig(epochs=3, batch_size=16, uniform_negatives=8),
        on_epoch=lambda epoch, _loss: seen.append(epoch),
    )
    assert seen == [0, 1, 2]


def test_arrays_round_trip_through_from_arrays() -> None:
    model, _ = _model()
    restored = TwoTowerModel.from_arrays(model.to_arrays(), model.config)
    for name, value in model.to_arrays().items():
        np.testing.assert_allclose(restored.to_arrays()[name], value, rtol=0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"id_dim": 0}, "dimensions"),
        ({"temperature": 0.0}, "temperature"),
        ({"learning_rate": -1.0}, "temperature"),
        ({"uniform_negatives": -1}, "uniform_negatives"),
    ],
)
def test_invalid_tower_configs_are_refused(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TowerConfig(**overrides)  # type: ignore[arg-type]


def test_mismatched_tower_shapes_are_refused() -> None:
    model, _ = _model()
    with pytest.raises(ValueError, match="parameters missing"):
        Tower({"emb": np.zeros((2, 2))})
    bad = {name: value.copy() for name, value in model.item.params.items()}
    bad["w2"] = np.zeros((model.item.params["w1"].shape[1], model.output_dim + 1))
    with pytest.raises(ValueError, match="same output dimension"):
        TwoTowerModel(
            model.user, Tower({**bad, "b2": np.zeros(model.output_dim + 1)}), model.config
        )


def test_forward_rejects_side_features_of_the_wrong_width() -> None:
    model, _ = _model()
    with pytest.raises(ValueError, match="side features"):
        model.user.forward(np.asarray([0, 1], dtype=np.int64), np.zeros((2, 99)))


def test_training_refuses_empty_inputs() -> None:
    config = TowerConfig(epochs=1)
    with pytest.raises(ValueError, match="non-empty"):
        train_two_tower(
            users=np.empty(0, dtype=np.int64),
            items=np.empty(0, dtype=np.int64),
            user_side=np.zeros((1, 1), dtype=np.float32),
            item_side=np.zeros((1, 1), dtype=np.float32),
            known_items=np.asarray([0], dtype=np.int64),
            config=config,
        )
    with pytest.raises(ValueError, match="known_items"):
        train_two_tower(
            users=np.asarray([0], dtype=np.int64),
            items=np.asarray([0], dtype=np.int64),
            user_side=np.zeros((1, 1), dtype=np.float32),
            item_side=np.zeros((1, 1), dtype=np.float32),
            known_items=np.empty(0, dtype=np.int64),
            config=config,
        )
