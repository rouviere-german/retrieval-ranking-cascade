"""Two-tower retrieval model in NumPy with hand-derived gradients.

Each tower maps an entity to a unit-norm embedding::

    x = [E[id], side]              id embedding (learned) + side features (from the snapshot)
    h = relu(x W1 + b1)
    z = h W2 + b2
    y = z / ||z||

and the score of a (user, item) pair is ``<y_u, y_i> / tau``.

The normalization is ``||z|| = sqrt(sum(z ** 2) + eps ** 2)`` rather than ``max(||z||, eps)``. Both
avoid dividing by zero when every ReLU unit is dead for a given input, but only the first is smooth:
with the clamped form the derivative is discontinuous at the clamp, and a dead layer produces a
gradient of magnitude ``1 / 1e-12``. With eps inside the square root the map is differentiable
everywhere, a dead layer yields the zero embedding and a finite gradient, and - conveniently - the
backward formula is unchanged, because ``z_i z_j / n ** 3 == y_i y_j / n`` holds for this ``n`` too.
``b1`` also starts at a small positive value so that no unit is dead at initialization.

Training uses a sampled softmax over the in-batch positives plus ``M`` uniformly sampled
negatives. With logQ correction (Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large
Corpus Item Recommendations", RecSys 2019) every candidate logit is reduced by ``log q_j``, the log
of the expected number of times item ``j`` appears in a batch's candidate set
(``B * p_j + M / N_known`` where ``p_j`` is its share of training positives). Without the
correction popular items are over-represented as negatives and get pushed down. Serving scores
never include the correction.

For row ``b`` of the logit matrix ``Z`` (B x C) the loss is ``logsumexp(Z_b) - Z_bb``; entries
where a candidate equals row ``b``'s positive item (a duplicate, not a negative) are masked out.
Gradients: ``dZ = (softmax(Z) - I) / B``, ``dU = dZ V / tau``, ``dV = dZ^T U / tau``; through the
normalization ``dz = (dy - y <y, dy>) / ||z||``; then the usual dense/ReLU chain rule, and
id-embedding rows receive the first ``id_dim`` columns of ``dx`` (summed over repeats).
:func:`gradient_check` compares all of this against central finite differences.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "Adam",
    "GradientCheck",
    "Tower",
    "TowerCache",
    "TowerConfig",
    "TrainingHistory",
    "TwoTowerModel",
    "gradient_check",
    "sampled_softmax",
    "train_two_tower",
]

FArray = npt.NDArray[np.floating[Any]]
IArray = npt.NDArray[np.integer[Any]]
# Large enough that a dead layer's gradient stays in a sane range, small enough that it does not
# perturb a healthy embedding: ||y|| differs from 1 by under 1e-12 once ||z|| exceeds 1e-3.
_EPS = 1e-6
# Gradient entries below this fraction of their own tensor's largest entry are beyond what a finite
# difference can resolve; see gradient_check.
_RELATIVE_FLOOR = 1e-6
# A central difference of a loss of order 1 with step ``eps`` carries absolute noise of about
# ``machine_epsilon / eps``. To read a *relative* error of 1e-4 off such a comparison the derivative
# has to stand this many noise levels above it; see gradient_check.
_NOISE_MULTIPLIER = 1e5


@dataclass(frozen=True, slots=True)
class TowerConfig:
    id_dim: int = 32
    hidden_dim: int = 64
    output_dim: int = 32
    temperature: float = 0.05
    learning_rate: float = 0.005
    batch_size: int = 1024
    uniform_negatives: int = 1024
    epochs: int = 10
    logq_correction: bool = True
    init_scale: float = 1.0
    # A small positive first-layer bias, so no hidden unit is dead at initialization.
    init_bias: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.id_dim, self.hidden_dim, self.output_dim, self.batch_size, self.epochs) < 1:
            raise ValueError("tower dimensions, batch size and epochs must be positive")
        if self.temperature <= 0 or self.learning_rate <= 0:
            raise ValueError("temperature and learning rate must be positive")
        if self.uniform_negatives < 0:
            raise ValueError("uniform_negatives must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TowerCache:
    ids: npt.NDArray[np.int64]
    x: FArray
    h_pre: FArray
    h: FArray
    norm: FArray
    y: FArray


class Tower:
    """One tower: id embedding + side features -> 2-layer MLP -> L2 normalization."""

    PARAMS = ("emb", "w1", "b1", "w2", "b2")

    def __init__(self, params: Mapping[str, FArray]) -> None:
        missing = set(self.PARAMS) - set(params)
        if missing:
            raise ValueError(f"tower parameters missing: {sorted(missing)}")
        self.params: dict[str, FArray] = {name: params[name] for name in self.PARAMS}
        emb, w1, b1, w2, b2 = (self.params[name] for name in self.PARAMS)
        if emb.ndim != 2 or w1.ndim != 2 or w2.ndim != 2:
            raise ValueError("tower weight matrices must be 2-D")
        if w1.shape[0] < emb.shape[1] or b1.shape != (w1.shape[1],):
            raise ValueError("first layer shape does not match the id embedding")
        if w2.shape[0] != w1.shape[1] or b2.shape != (w2.shape[1],):
            raise ValueError("second layer shape does not match the first")

    @classmethod
    def initialize(
        cls,
        n_entities: int,
        side_dim: int,
        config: TowerConfig,
        rng: np.random.Generator,
        dtype: npt.DTypeLike = np.float32,
    ) -> Tower:
        in_dim = config.id_dim + side_dim
        scale = config.init_scale
        # Id embeddings start at zero: entities never seen in training keep a content-only
        # representation instead of random noise.
        params = {
            "emb": np.zeros((n_entities, config.id_dim), dtype=dtype),
            "w1": (rng.normal(size=(in_dim, config.hidden_dim)) * scale * math.sqrt(2.0 / in_dim)),
            "b1": np.full(config.hidden_dim, config.init_bias),
            "w2": (
                rng.normal(size=(config.hidden_dim, config.output_dim))
                * scale
                * math.sqrt(1.0 / config.hidden_dim)
            ),
            "b2": np.zeros(config.output_dim),
        }
        return cls({name: np.asarray(value, dtype=dtype) for name, value in params.items()})

    @property
    def id_dim(self) -> int:
        return int(self.params["emb"].shape[1])

    @property
    def side_dim(self) -> int:
        return int(self.params["w1"].shape[0] - self.params["emb"].shape[1])

    @property
    def n_entities(self) -> int:
        return int(self.params["emb"].shape[0])

    def forward(self, ids: IArray, side: FArray) -> TowerCache:
        index = np.asarray(ids, dtype=np.int64)
        p = self.params
        if side.shape != (index.shape[0], self.side_dim):
            raise ValueError(f"side features must have shape ({index.shape[0]}, {self.side_dim})")
        x = np.concatenate([p["emb"][index], side.astype(p["w1"].dtype, copy=False)], axis=1)
        h_pre = x @ p["w1"] + p["b1"]
        h = np.maximum(h_pre, 0.0)
        z = h @ p["w2"] + p["b2"]
        norm = np.sqrt(np.sum(z * z, axis=1, keepdims=True) + _EPS * _EPS)
        return TowerCache(ids=index, x=x, h_pre=h_pre, h=h, norm=norm, y=z / norm)

    def backward(self, cache: TowerCache, dy: FArray) -> dict[str, Any]:
        """Gradients of the loss w.r.t. parameters; ``emb`` is (unique_ids, row_gradients)."""
        p = self.params
        dz = (dy - cache.y * np.sum(cache.y * dy, axis=1, keepdims=True)) / cache.norm
        dw2 = cache.h.T @ dz
        db2 = dz.sum(axis=0)
        dh_pre = (dz @ p["w2"].T) * (cache.h_pre > 0)
        dw1 = cache.x.T @ dh_pre
        db1 = dh_pre.sum(axis=0)
        dx_id = (dh_pre @ p["w1"].T)[:, : self.id_dim]
        unique, inverse = np.unique(cache.ids, return_inverse=True)
        demb = np.zeros((unique.shape[0], self.id_dim), dtype=dx_id.dtype)
        np.add.at(demb, inverse, dx_id)
        return {"emb": (unique, demb), "w1": dw1, "b1": db1, "w2": dw2, "b2": db2}

    def embed(self, ids: IArray, side: FArray, *, chunk: int = 65_536) -> tuple[FArray, FArray]:
        """Unit-norm embeddings and the pre-normalization norms, computed in chunks."""
        index = np.asarray(ids, dtype=np.int64)
        outputs: list[FArray] = []
        norms: list[FArray] = []
        for start in range(0, index.shape[0], chunk):
            cache = self.forward(index[start : start + chunk], side[start : start + chunk])
            outputs.append(cache.y)
            norms.append(cache.norm[:, 0])
        if not outputs:
            dim = int(self.params["w2"].shape[1])
            dtype = self.params["w2"].dtype
            return np.zeros((0, dim), dtype=dtype), np.zeros(0, dtype=dtype)
        return np.concatenate(outputs), np.concatenate(norms)


def sampled_softmax(
    user_y: FArray,
    item_y: FArray,
    positives: IArray,
    candidates: IArray,
    log_q: FArray | None,
    temperature: float,
) -> tuple[float, FArray, FArray]:
    """Loss and gradients w.r.t. user and candidate embeddings.

    ``candidates[:B]`` must be the positives (row ``b``'s target is column ``b``).
    """
    batch = user_y.shape[0]
    if candidates.shape[0] < batch or not np.array_equal(candidates[:batch], positives):
        raise ValueError("the first B candidates must be the batch positives")
    logits = (user_y @ item_y.T) / temperature
    if log_q is not None:
        logits = logits - log_q[None, :]
    duplicate = candidates[None, :] == positives[:, None]
    duplicate[np.arange(batch), np.arange(batch)] = False
    logits = np.where(duplicate, -np.inf, logits)
    row_max = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - row_max)
    total = exp.sum(axis=1, keepdims=True)
    log_sum = np.log(total) + row_max
    diag = logits[np.arange(batch), np.arange(batch)]
    loss = float(np.mean(log_sum[:, 0] - diag))
    d_logits = exp / total
    d_logits[np.arange(batch), np.arange(batch)] -= 1.0
    d_logits /= batch
    d_user = (d_logits @ item_y) / temperature
    d_item = (d_logits.T @ user_y) / temperature
    return loss, d_user, d_item


class TwoTowerModel:
    def __init__(self, user: Tower, item: Tower, config: TowerConfig) -> None:
        if user.params["w2"].shape[1] != item.params["w2"].shape[1]:
            raise ValueError("user and item towers must have the same output dimension")
        self.user = user
        self.item = item
        self.config = config

    @classmethod
    def initialize(
        cls,
        *,
        n_users: int,
        n_items: int,
        user_side_dim: int,
        item_side_dim: int,
        config: TowerConfig,
        dtype: npt.DTypeLike = np.float32,
    ) -> TwoTowerModel:
        rng = np.random.default_rng(config.seed)
        user = Tower.initialize(n_users, user_side_dim, config, rng, dtype)
        item = Tower.initialize(n_items, item_side_dim, config, rng, dtype)
        return cls(user, item, config)

    @property
    def output_dim(self) -> int:
        return int(self.user.params["w2"].shape[1])

    def loss_and_grads(
        self,
        users: IArray,
        user_side: FArray,
        positives: IArray,
        negatives: IArray,
        item_side: FArray,
        log_q: FArray | None,
    ) -> tuple[float, dict[str, dict[str, Any]]]:
        """One sampled-softmax step. ``item_side`` rows follow ``[positives, negatives]``."""
        candidates = np.concatenate([np.asarray(positives), np.asarray(negatives)]).astype(np.int64)
        user_cache = self.user.forward(users, user_side)
        item_cache = self.item.forward(candidates, item_side)
        loss, d_user, d_item = sampled_softmax(
            user_cache.y,
            item_cache.y,
            np.asarray(positives, dtype=np.int64),
            candidates,
            log_q,
            self.config.temperature,
        )
        grads = {
            "user": self.user.backward(user_cache, d_user),
            "item": self.item.backward(item_cache, d_item),
        }
        return loss, grads

    def to_arrays(self) -> dict[str, FArray]:
        arrays: dict[str, FArray] = {}
        for prefix, tower in (("user", self.user), ("item", self.item)):
            for name, value in tower.params.items():
                arrays[f"{prefix}_{name}"] = value
        return arrays

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, FArray], config: TowerConfig) -> TwoTowerModel:
        towers = {
            prefix: Tower({name: arrays[f"{prefix}_{name}"] for name in Tower.PARAMS})
            for prefix in ("user", "item")
        }
        return cls(towers["user"], towers["item"], config)


class Adam:
    """Adam (Kingma and Ba 2015) for dense tensors, lazy (row-wise) for embedding tables.

    Embedding rows are only updated when they receive a gradient, as in TensorFlow's LazyAdam, so
    a step costs O(batch) instead of O(table size).
    """

    def __init__(
        self,
        tower: Tower,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.tower = tower
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0
        self.m = {name: np.zeros_like(value) for name, value in tower.params.items()}
        self.v = {name: np.zeros_like(value) for name, value in tower.params.items()}

    def step(self, grads: Mapping[str, Any]) -> None:
        self.step_count += 1
        t = self.step_count
        correction = math.sqrt(1.0 - self.beta2**t) / (1.0 - self.beta1**t)
        lr = self.lr * correction
        for name, grad in grads.items():
            param = self.tower.params[name]
            if name == "emb":
                rows, row_grad = grad
                m = self.beta1 * self.m[name][rows] + (1.0 - self.beta1) * row_grad
                v = self.beta2 * self.v[name][rows] + (1.0 - self.beta2) * row_grad**2
                self.m[name][rows] = m
                self.v[name][rows] = v
                param[rows] -= (lr * m / (np.sqrt(v) + self.eps)).astype(param.dtype)
            else:
                self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
                self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * grad**2
                param -= (lr * self.m[name] / (np.sqrt(self.v[name]) + self.eps)).astype(
                    param.dtype
                )


@dataclass(slots=True)
class TrainingHistory:
    epoch_loss: list[float] = field(default_factory=list)
    steps: int = 0


def train_two_tower(
    *,
    users: IArray,
    items: IArray,
    user_side: FArray,
    item_side: FArray,
    known_items: IArray,
    config: TowerConfig,
    on_epoch: Callable[[int, float], None] | None = None,
) -> tuple[TwoTowerModel, TrainingHistory]:
    """Train on positive (user, item) pairs. Side features are indexed by entity id.

    Uniform negatives are drawn only from ``known_items`` (items with impressions in the training
    data), so items that launched later keep an untouched, content-only id embedding.
    """
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    known = np.asarray(known_items, dtype=np.int64)
    if users.shape != items.shape or users.size == 0:
        raise ValueError("users and items must be non-empty arrays of equal length")
    if known.size == 0:
        raise ValueError("known_items must not be empty")
    model = TwoTowerModel.initialize(
        n_users=user_side.shape[0],
        n_items=item_side.shape[0],
        user_side_dim=user_side.shape[1],
        item_side_dim=item_side.shape[1],
        config=config,
    )
    rng = np.random.default_rng(config.seed + 1)
    counts = np.bincount(items, minlength=item_side.shape[0]).astype(np.float64)
    share = counts / counts.sum()
    optimizers = {
        "user": Adam(model.user, config.learning_rate),
        "item": Adam(model.item, config.learning_rate),
    }
    history = TrainingHistory()
    batch = min(config.batch_size, users.shape[0])
    n_neg = config.uniform_negatives
    for epoch in range(config.epochs):
        order = rng.permutation(users.shape[0])
        losses: list[float] = []
        for start in range(0, order.shape[0] - batch + 1, batch):
            rows = order[start : start + batch]
            b_users, b_items = users[rows], items[rows]
            negatives = known[rng.integers(0, known.shape[0], size=n_neg)]
            candidates = np.concatenate([b_items, negatives])
            log_q = None
            if config.logq_correction:
                expected = batch * share[candidates] + n_neg / known.shape[0]
                log_q = np.log(np.maximum(expected, _EPS)).astype(user_side.dtype)
            loss, grads = model.loss_and_grads(
                b_users,
                user_side[b_users],
                b_items,
                negatives,
                item_side[candidates],
                log_q,
            )
            optimizers["user"].step(grads["user"])
            optimizers["item"].step(grads["item"])
            losses.append(loss)
            history.steps += 1
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.epoch_loss.append(mean_loss)
        if on_epoch is not None:
            on_epoch(epoch, mean_loss)
    return model, history


@dataclass(frozen=True, slots=True)
class GradientCheck:
    """Outcome of a finite-difference sweep over every parameter entry."""

    max_relative_error: float
    checked: int
    skipped: int
    worst_entry: str | None

    def __float__(self) -> float:
        return self.max_relative_error


def gradient_check(
    model: TwoTowerModel,
    *,
    users: IArray,
    user_side: FArray,
    positives: IArray,
    negatives: IArray,
    item_side: FArray,
    log_q: FArray | None,
    eps: float = 1e-6,
    min_magnitude: float | None = None,
) -> GradientCheck:
    """Compare every analytic gradient entry with a central difference.

    Use a float64 model. Every parameter entry is perturbed, so keep the model small.

    Negligible entries are skipped rather than compared, for a reason worth stating: a central
    difference of a loss of order 1 with step ``eps`` carries absolute noise of about
    ``machine_epsilon / eps``, so two numbers inside that band differ because of floating-point
    cancellation, not because of the derivation. A *relative* error is only readable off the
    comparison when the derivative stands well above that noise, which is why the default floor is
    ``_NOISE_MULTIPLIER`` (1e5) times it - enough that a compared entry's finite difference is
    accurate to roughly 1e-5 relative, and a 1e-4 assertion therefore measures the derivation and
    not the arithmetic. An entry counts as negligible when both the analytic and the numeric value
    fall below

        ``max(min_magnitude, relative_floor * max|analytic| over that parameter)``

    The second term covers tensors whose entries span many orders of magnitude. Real ones do: an
    id-embedding row barely touched by the batch, or a weight feeding a mostly-dead unit, has a true
    gradient a millionth of its neighbours', and no finite difference can say anything about it. A
    mistake in a derivation moves gradients that are large relative to their own tensor, which is
    exactly what is still compared. ``skipped`` is reported so a sweep that compared nothing cannot
    pass silently.
    """
    floor = (
        min_magnitude
        if min_magnitude is not None
        else _NOISE_MULTIPLIER * float(np.finfo(np.float64).eps) / eps
    )
    _, grads = model.loss_and_grads(users, user_side, positives, negatives, item_side, log_q)
    worst = 0.0
    worst_entry: str | None = None
    checked = 0
    skipped = 0
    for tower_name, tower in (("user", model.user), ("item", model.item)):
        analytic: dict[str, FArray] = {}
        for name, grad in grads[tower_name].items():
            if name == "emb":
                rows, row_grad = grad
                dense = np.zeros_like(tower.params["emb"])
                dense[rows] = row_grad
                analytic[name] = dense
            else:
                analytic[name] = grad
        for name, param in tower.params.items():
            flat = param.reshape(-1)
            expected = analytic[name].reshape(-1)
            negligible = max(floor, _RELATIVE_FLOOR * float(np.abs(expected).max(initial=0.0)))
            for index in range(flat.shape[0]):
                original = flat[index]
                flat[index] = original + eps
                plus, _ = model.loss_and_grads(
                    users, user_side, positives, negatives, item_side, log_q
                )
                flat[index] = original - eps
                minus, _ = model.loss_and_grads(
                    users, user_side, positives, negatives, item_side, log_q
                )
                flat[index] = original
                numeric = (plus - minus) / (2.0 * eps)
                analytic_value = float(expected[index])
                if abs(numeric) < negligible and abs(analytic_value) < negligible:
                    skipped += 1
                    continue
                checked += 1
                scale = abs(numeric) + abs(analytic_value)
                error = abs(numeric - analytic_value) / scale
                if error > worst:
                    worst = error
                    worst_entry = f"{tower_name}.{name}[{index}]"
    return GradientCheck(
        max_relative_error=worst, checked=checked, skipped=skipped, worst_entry=worst_entry
    )
