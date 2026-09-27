"""Seeded synthetic recommendation world with a position-based click model.

World. Users and items share a latent space built from category centres. Item popularity follows
a power law, and a fraction of items launches after day 0 (cold-start items). The ground-truth
affinity ``rel(u, i) = sigmoid(a * <p_u, q_i> / k + w * z_pop(i) + b)`` is the probability of a
click once the user has examined the item. The world is hidden from the system: training only
sees the impression log and the item catalog, while evaluation may query the world directly.

Logging policy. Every session, the previous production system scores a pool of candidates (half
drawn by global popularity, half from the user's preferred categories) with a noisy,
popularity-biased version of the true score and shows the top ``slots`` items in order. Items the
user already clicked are never shown again. In a ``randomized_fraction`` slice of sessions the
same items are shown in a uniformly random order (the RandTop-n intervention of Joachims,
Swaminathan and Schnabel, "Unbiased Learning-to-Rank with Biased Feedback", WSDM 2017). In that
slice position is independent of relevance, so click rates per position estimate the examination
propensities.

Clicks follow the position-based model (Richardson et al. 2007; Craswell et al. 2008):
``P(click) = P(examine | position) * rel(u, i)`` with ``P(examine | k) = k ** -eta``.

Periods. Per user, sessions are ordered by day: the last one is the held-out *test* session, the
one before it the *label* session (ranker training labels), and the rest are *history*.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "PERIOD_HISTORY",
    "PERIOD_LABEL",
    "PERIOD_TEST",
    "SCALES",
    "Catalog",
    "Interactions",
    "Scale",
    "World",
    "WorldConfig",
    "generate",
    "world_config",
]

PERIOD_HISTORY = 0
PERIOD_LABEL = 1
PERIOD_TEST = 2

Scale = Literal["tiny", "smoke", "small", "full"]
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """Parameters of the synthetic world, the logging policy and the click model."""

    n_users: int
    n_items: int
    n_categories: int
    latent_dim: int = 16
    slots: int = 10
    pool_size: int = 40
    min_sessions: int = 3
    extra_sessions_mean: float = 1.0
    max_sessions: int = 12
    n_days: int = 30
    popularity_exponent: float = 0.8
    category_exponent: float = 0.7
    within_category_popularity_power: float = 0.5
    cold_item_fraction: float = 0.08
    randomized_fraction: float = 0.05
    examination_eta: float = 1.0
    user_concentration: float = 0.3
    latent_noise: float = 0.6
    affinity_scale: float = 8.0
    popularity_weight: float = 0.5
    relevance_bias: float = -6.0
    logging_affinity_weight: float = 0.6
    logging_popularity_weight: float = 0.8
    logging_noise: float = 1.0
    seed: int = 7
    # Users processed per vectorized batch. This is a memory knob, but the random draws are shaped
    # per batch, so changing it changes the log. It lives in the config (and therefore in the run
    # manifest) rather than in a call argument, so a change can never be silent.
    chunk_users: int = 8_192

    def __post_init__(self) -> None:
        if self.n_users < 2 or self.n_items < self.pool_size or self.n_categories < 2:
            raise ValueError("world needs >= 2 users, >= pool_size items and >= 2 categories")
        if not 1 <= self.slots <= self.pool_size // 2:
            raise ValueError("slots must be between 1 and pool_size // 2")
        if self.min_sessions < 3:
            raise ValueError("min_sessions must be >= 3 (history, label and test periods)")
        if not 0.0 <= self.randomized_fraction <= 1.0:
            raise ValueError("randomized_fraction must be in [0, 1]")
        if self.max_sessions < self.min_sessions:
            raise ValueError("max_sessions must be >= min_sessions")
        if self.chunk_users < 1:
            raise ValueError("chunk_users must be >= 1")


SCALES: dict[str, WorldConfig] = {
    # Unit and integration tests.
    "tiny": WorldConfig(n_users=300, n_items=800, n_categories=8),
    # CI training smoke: 1% of the full world.
    "smoke": WorldConfig(n_users=500, n_items=1_000, n_categories=10),
    # CI benchmarks and the demo: 10% of the full world.
    "small": WorldConfig(n_users=5_000, n_items=10_000, n_categories=30),
    # The specification's world: 50k users, 100k items, about 2M impressions.
    "full": WorldConfig(n_users=50_000, n_items=100_000, n_categories=50),
}


def world_config(scale: str, *, seed: int | None = None, **overrides: Any) -> WorldConfig:
    """Return the preset for ``scale`` with an optional seed and field overrides."""
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; choose from {sorted(SCALES)}")
    config = SCALES[scale]
    if seed is not None:
        overrides["seed"] = seed
    return replace(config, **overrides) if overrides else config


def _sigmoid(x: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64))))


@dataclass(frozen=True, slots=True)
class World:
    """Ground truth. Only the generator and offline evaluation may look at it."""

    config: WorldConfig
    user_latent: FloatArray
    item_latent: FloatArray
    user_category_pref: FloatArray
    item_category: IntArray
    item_popularity: npt.NDArray[np.float64]
    item_popularity_z: FloatArray
    item_launch_day: IntArray

    def relevance_logit(self, users: npt.ArrayLike, items: npt.ArrayLike) -> FloatArray:
        """Element-wise true relevance logit for broadcastable user and item index arrays."""
        u = np.asarray(users, dtype=np.int64)
        i = np.asarray(items, dtype=np.int64)
        cfg = self.config
        dot = np.einsum("...k,...k->...", self.user_latent[u], self.item_latent[i])
        logit = (
            cfg.affinity_scale * dot / cfg.latent_dim
            + cfg.popularity_weight * self.item_popularity_z[i]
            + cfg.relevance_bias
        )
        return np.asarray(logit, dtype=np.float32)

    def relevance(self, users: npt.ArrayLike, items: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """True click probability given examination, element-wise."""
        return _sigmoid(self.relevance_logit(users, items))

    def relevance_rows(self, users: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Relevance of every item for each user in ``users``: shape (len(users), n_items)."""
        u = np.asarray(users, dtype=np.int64)
        cfg = self.config
        logits = (
            cfg.affinity_scale * (self.user_latent[u] @ self.item_latent.T) / cfg.latent_dim
            + cfg.popularity_weight * self.item_popularity_z[None, :]
            + cfg.relevance_bias
        )
        return _sigmoid(logits)

    def examination(self, positions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """True examination propensity ``P(examine | position)`` for 1-based positions."""
        pos = np.asarray(positions, dtype=np.float64)
        if np.any(pos < 1):
            raise ValueError("positions are 1-based")
        return np.asarray(pos ** (-self.config.examination_eta))


@dataclass(frozen=True, slots=True)
class Catalog:
    """What the system knows about items: category and launch day (no latent factors)."""

    item_category: IntArray
    item_launch_day: IntArray
    n_categories: int

    @property
    def n_items(self) -> int:
        return int(self.item_category.shape[0])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "item_id": np.arange(self.n_items, dtype=np.int64),
                "category": self.item_category.astype(np.int64),
                "launch_day": self.item_launch_day.astype(np.int64),
            }
        )


@dataclass(frozen=True, slots=True)
class Interactions:
    """Impression log in session order: one row per shown item."""

    session_id: IntArray
    user_id: IntArray
    item_id: IntArray
    position: npt.NDArray[np.int8]
    clicked: npt.NDArray[np.int8]
    randomized: npt.NDArray[np.bool_]
    day: npt.NDArray[np.int16]
    period: npt.NDArray[np.int8]

    def __len__(self) -> int:
        return int(self.session_id.shape[0])

    def select(
        self, selector: npt.NDArray[np.bool_] | npt.NDArray[np.integer[Any]]
    ) -> Interactions:
        """Rows picked by a boolean mask or an index array, keeping every column aligned."""
        return Interactions(**{f.name: getattr(self, f.name)[selector] for f in fields(self)})

    def periods(self, *periods: int) -> Interactions:
        return self.select(np.isin(self.period, np.asarray(periods, dtype=np.int8)))

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: values.astype(np.int64) for name, values in self.arrays().items()}
            | {"randomized": self.randomized.astype(bool)}
        )

    def sha256(self) -> str:
        """Content hash over column names, dtypes, shapes and bytes (order-sensitive)."""
        digest = hashlib.sha256()
        for name, values in self.arrays().items():
            array = np.ascontiguousarray(values)
            digest.update(f"{name}:{array.dtype.str}:{array.shape}".encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def to_npz_bytes(self) -> bytes:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, allow_pickle=False, **self.arrays())
        return buffer.getvalue()

    @property
    def click_rate(self) -> float:
        return float(self.clicked.mean()) if len(self) else 0.0


def _make_world(cfg: WorldConfig, rng: np.random.Generator) -> World:
    k = cfg.latent_dim
    centres = rng.normal(size=(cfg.n_categories, k))
    category_weights = (np.arange(cfg.n_categories) + 1.0) ** -cfg.category_exponent
    item_category = rng.choice(
        cfg.n_categories, size=cfg.n_items, p=category_weights / category_weights.sum()
    ).astype(np.int32)
    item_latent = centres[item_category] + cfg.latent_noise * rng.normal(size=(cfg.n_items, k))

    ranks = rng.permutation(cfg.n_items) + 1.0
    popularity = np.asarray(ranks**-cfg.popularity_exponent, dtype=np.float64)
    popularity /= popularity.sum()
    log_pop = np.log(popularity)
    popularity_z = (log_pop - log_pop.mean()) / log_pop.std()

    launch = np.zeros(cfg.n_items, dtype=np.int32)
    cold = rng.random(cfg.n_items) < cfg.cold_item_fraction
    launch[cold] = rng.integers(1, cfg.n_days, size=int(cold.sum()))

    pref = rng.dirichlet(np.full(cfg.n_categories, cfg.user_concentration), size=cfg.n_users)
    user_latent = pref @ centres
    # Rescale so a user's taste vector is as long as a category centre on average.
    norms = np.linalg.norm(user_latent, axis=1, keepdims=True)
    user_latent = user_latent / np.maximum(norms, 1e-9) * np.sqrt(k)
    user_latent += cfg.latent_noise * rng.normal(size=(cfg.n_users, k))

    return World(
        config=cfg,
        user_latent=user_latent.astype(np.float32),
        item_latent=item_latent.astype(np.float32),
        user_category_pref=pref.astype(np.float32),
        item_category=item_category,
        item_popularity=popularity,
        item_popularity_z=popularity_z.astype(np.float32),
        item_launch_day=launch,
    )


def _session_plan(
    cfg: WorldConfig, rng: np.random.Generator
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Per-user session counts and a (users, max_sessions) matrix of sorted days (-1 = none)."""
    counts = np.minimum(
        cfg.min_sessions + rng.poisson(cfg.extra_sessions_mean, size=cfg.n_users),
        cfg.max_sessions,
    ).astype(np.int64)
    width = int(counts.max())
    raw = rng.integers(0, cfg.n_days, size=(cfg.n_users, width)).astype(np.int64)
    valid = np.arange(width)[None, :] < counts[:, None]
    raw[~valid] = np.iinfo(np.int64).max
    days = np.sort(raw, axis=1)
    days[~valid] = -1
    return counts, days


def _category_sampler(
    world: World,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Items sorted by category and a cumulative key ``category + within-category CDF``.

    Sampling category ``c`` then ``u ~ U[0, 1)`` and searching ``c + u`` in the key picks an item
    of category ``c`` with probability proportional to its popularity, fully vectorized.
    """
    order = np.lexsort((np.arange(world.item_category.shape[0]), world.item_category))
    categories = world.item_category[order].astype(np.int64)
    weights = world.item_popularity[order] ** world.config.within_category_popularity_power
    totals = np.bincount(categories, weights=weights, minlength=world.config.n_categories)
    cumulative = np.zeros_like(weights)
    for category in range(world.config.n_categories):
        members = categories == category
        if members.any():
            cumulative[members] = np.cumsum(weights[members]) / totals[category]
    key = np.asarray(categories + np.minimum(cumulative, 1.0 - 1e-12), dtype=np.float64)
    return order.astype(np.int64), key


def _sample_pool(
    world: World,
    users: npt.NDArray[np.int64],
    rng: np.random.Generator,
    sampler: tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]],
    popularity_cdf: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    cfg = world.config
    half = cfg.pool_size // 2
    m = users.shape[0]
    popular = np.searchsorted(popularity_cdf, rng.random((m, half)), side="right")
    popular = np.minimum(popular, cfg.n_items - 1)

    pref_cdf = np.cumsum(world.user_category_pref[users].astype(np.float64), axis=1)
    pref_cdf /= pref_cdf[:, -1:]
    draws = rng.random((m, cfg.pool_size - half))
    categories = (pref_cdf[:, None, :] < draws[:, :, None]).sum(axis=2)
    categories = np.minimum(categories, cfg.n_categories - 1)
    order, key = sampler
    within = np.searchsorted(key, categories + rng.random(categories.shape), side="right")
    within = np.minimum(within, order.shape[0] - 1)
    return np.concatenate([popular, order[within]], axis=1).astype(np.int64)


def _duplicate_mask(pool: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
    """True for every repeated occurrence of an item within its row (first one kept)."""
    order = np.argsort(pool, axis=1, kind="stable")
    sorted_pool = np.take_along_axis(pool, order, axis=1)
    repeated = np.zeros_like(sorted_pool, dtype=bool)
    repeated[:, 1:] = sorted_pool[:, 1:] == sorted_pool[:, :-1]
    mask = np.zeros_like(repeated)
    np.put_along_axis(mask, order, repeated, axis=1)
    return mask


def generate(  # noqa: PLR0915 - one vectorized pass over the session rounds
    config: WorldConfig,
) -> tuple[World, Catalog, Interactions]:
    """Generate the world, the item catalog and the impression log.

    Deterministic in the whole config, which includes ``chunk_users``: the draws are shaped per
    batch, so the batch size is part of the data's identity, not an implementation detail.
    """
    chunk_users = config.chunk_users
    rng = np.random.default_rng(config.seed)
    world = _make_world(config, rng)
    counts, days = _session_plan(config, rng)
    sampler = _category_sampler(world)
    popularity_cdf = np.cumsum(world.item_popularity)
    popularity_cdf /= popularity_cdf[-1]
    examine = world.examination(np.arange(1, config.slots + 1))

    clicked_keys = np.empty(0, dtype=np.int64)
    columns: dict[str, list[npt.NDArray[Any]]] = {
        name: [] for name in ("user", "item", "pos", "click", "rand", "day", "period", "order")
    }
    n_items = np.int64(config.n_items)
    for round_index in range(int(counts.max())):
        active = np.flatnonzero(counts > round_index).astype(np.int64)
        new_keys: list[npt.NDArray[np.int64]] = []
        for start in range(0, active.shape[0], chunk_users):
            users = active[start : start + chunk_users]
            m = users.shape[0]
            session_day = days[users, round_index]
            pool = _sample_pool(world, users, rng, sampler, popularity_cdf)
            logit = world.relevance_logit(users[:, None], pool)
            score = (
                config.logging_affinity_weight * logit
                + config.logging_popularity_weight * world.item_popularity_z[pool]
                + config.logging_noise * rng.normal(size=pool.shape)
            )
            invalid = _duplicate_mask(pool) | (world.item_launch_day[pool] > session_day[:, None])
            if clicked_keys.size:
                invalid |= np.isin(users[:, None] * n_items + pool, clicked_keys)
            score = np.where(invalid, -np.inf, score)
            top = np.argsort(-score, axis=1, kind="stable")[:, : config.slots]
            shown = np.take_along_axis(pool, top, axis=1)
            shown_ok = np.isfinite(np.take_along_axis(score, top, axis=1))

            randomized = np.asarray(rng.random(m) < config.randomized_fraction)
            if randomized.any():
                shuffle = np.argsort(rng.random((int(randomized.sum()), config.slots)), axis=1)
                shown[randomized] = np.take_along_axis(shown[randomized], shuffle, axis=1)
                shown_ok[randomized] = np.take_along_axis(shown_ok[randomized], shuffle, axis=1)

            rel = world.relevance(users[:, None], shown)
            clicks = rng.random(shown.shape) < examine[None, :] * rel
            clicks &= shown_ok
            new_keys.append((users[:, None] * n_items + shown)[clicks])

            period = np.full(m, PERIOD_HISTORY, dtype=np.int8)
            period[round_index == counts[users] - 1] = PERIOD_TEST
            period[round_index == counts[users] - 2] = PERIOD_LABEL

            rows, cols = np.nonzero(shown_ok)
            columns["user"].append(users[rows])
            columns["item"].append(shown[rows, cols])
            columns["pos"].append(cols + 1)
            columns["click"].append(clicks[rows, cols])
            columns["rand"].append(randomized[rows])
            columns["day"].append(session_day[rows])
            columns["period"].append(period[rows])
            columns["order"].append(users[rows] * config.max_sessions + round_index)
        if new_keys:
            clicked_keys = np.union1d(clicked_keys, np.concatenate(new_keys))

    order_key = np.concatenate(columns["order"])
    position = np.concatenate(columns["pos"])
    sort = np.lexsort((position, order_key))
    session_key = order_key[sort]
    boundaries = np.concatenate([[True], session_key[1:] != session_key[:-1]])
    interactions = Interactions(
        session_id=(np.cumsum(boundaries) - 1).astype(np.int32),
        user_id=np.concatenate(columns["user"])[sort].astype(np.int32),
        item_id=np.concatenate(columns["item"])[sort].astype(np.int32),
        position=position[sort].astype(np.int8),
        clicked=np.concatenate(columns["click"])[sort].astype(np.int8),
        randomized=np.concatenate(columns["rand"])[sort].astype(np.bool_),
        day=np.concatenate(columns["day"])[sort].astype(np.int16),
        period=np.concatenate(columns["period"])[sort].astype(np.int8),
    )
    catalog = Catalog(
        item_category=world.item_category.copy(),
        item_launch_day=world.item_launch_day.copy(),
        n_categories=config.n_categories,
    )
    return world, catalog, interactions
