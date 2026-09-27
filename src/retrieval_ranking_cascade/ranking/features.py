"""Feature snapshot: user, user x category and item aggregates built with DuckDB.

A snapshot is computed once from an impression log cut at a point in time and loaded at startup
(online feature stores are a non-goal). The ranker is trained on a snapshot built from the history
period and served with one built from history + label, so training features never see the
labels they are trained on.

Item statistics are *robust* by default, which is the model-side half of the shilling defence:

- each user counts at most once per item (a user clicking an item 50 times is one clicker);
- each user's contribution is weighted by tenure,
  ``min(1, (sessions / trust_sessions) ** tenure_exponent)``. The exponent matters: a linear weight
  damps a one-session account by ``trust_sessions``, a quadratic one by its square, and an attacker
  pays for the difference in accounts. It never reaches zero, because roughly a third of legitimate
  users in this world also have a single session in the history period and their signal is real;
- item CTR is shrunk towards the global CTR with a prior of ``ctr_prior_strength`` *weighted*
  impressions. This is the only defence that touches a ratio: tenure weighting scales a shilled
  item's clicks and impressions by the same factor and so cannot damp its CTR at all, which is why
  the prior has to be strong enough to matter next to a plausible number of fake accounts.

``robust=False`` builds the naive version - raw counts, one row per impression, and an unshrunk
``clicks / impressions`` CTR - which is what an implementation that never considered the attack
would produce. The pair is what ``evaluation.shilling`` compares.

Determinism. A snapshot's SHA-256 is part of the bundle manifest, so the aggregation has to be
bit-reproducible. Two things would otherwise break that, and both are handled here:

* DuckDB sums a group in whatever order its morsels finish in, and floating-point addition is not
  associative, so ``sum(t.w)`` varies in the last bits between runs. The connection is pinned to one
  thread with insertion order preserved, which also keeps the aggregation polite on a small box.
* the popularity list is an ``argsort`` over those sums, where a last-bit difference does not stay
  in the last bits: it swaps two items. The sort keys are rounded well above the noise and ties are
  broken by item id, so the order is a function of the counts rather than of the schedule.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import duckdb
import numpy as np
import numpy.typing as npt
import pandas as pd

from retrieval_ranking_cascade.data.generator import Catalog, Interactions

__all__ = [
    "FEATURE_NAMES",
    "FeatureSnapshot",
    "SnapshotConfig",
    "assemble_features",
    "build_snapshot",
    "item_side_features",
    "user_side_features",
]

FEATURE_NAMES: tuple[str, ...] = (
    "tower_score",
    "item_log_clickers",
    "item_ctr",
    "item_age_days",
    "user_cat_click_share",
    "user_cat_ctr",
    "user_cat_recency_days",
    "user_log_clicks",
    "user_ctr",
    "item_log_impressions",
)
_MAX_RECENCY = 60.0
_LOG_SCALE = 5.0
# Decimals the popularity sort keys are rounded to. Weighted counts reach about 1e6 in the full
# world, where float64 noise is around 1e-10, so this is far above the noise and far below any
# difference that should change an item's rank.
_POPULARITY_DECIMALS = 6


@dataclass(frozen=True, slots=True)
class SnapshotConfig:
    robust: bool = True
    trust_sessions: int = 3
    tenure_exponent: float = 2.0
    ctr_prior_strength: float = 200.0
    user_ctr_prior_strength: float = 10.0
    category_ctr_prior_strength: float = 5.0
    share_prior_strength: float = 1.0
    popular_list_size: int = 2_000


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Precomputed arrays indexed by user id, item id and category id."""

    snapshot_day: npt.NDArray[np.int32]  # shape (1,)
    global_ctr: npt.NDArray[np.float32]  # shape (1,)
    user_clicks: npt.NDArray[np.float32]
    user_impressions: npt.NDArray[np.float32]
    user_ctr: npt.NDArray[np.float32]
    user_cat_share: npt.NDArray[np.float32]
    user_cat_ctr: npt.NDArray[np.float32]
    user_cat_recency: npt.NDArray[np.float32]
    item_impressions: npt.NDArray[np.float32]
    item_clickers: npt.NDArray[np.float32]
    item_ctr: npt.NDArray[np.float32]
    item_category: npt.NDArray[np.int32]
    item_launch_day: npt.NDArray[np.int32]
    seen_indptr: npt.NDArray[np.int64]
    seen_indices: npt.NDArray[np.int32]
    popular_items: npt.NDArray[np.int32]

    @property
    def n_users(self) -> int:
        return int(self.user_clicks.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_category.shape[0])

    @property
    def n_categories(self) -> int:
        return int(self.user_cat_share.shape[1])

    @property
    def day(self) -> int:
        return int(self.snapshot_day[0])

    def seen(self, user: int) -> npt.NDArray[np.int32]:
        """Items ``user`` already clicked (sorted); empty for unknown users."""
        if not 0 <= user < self.n_users:
            return np.empty(0, dtype=np.int32)
        return self.seen_indices[self.seen_indptr[user] : self.seen_indptr[user + 1]]

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, npt.NDArray[Any]]) -> FeatureSnapshot:
        names = {f.name for f in fields(cls)}
        missing = names - set(arrays)
        if missing:
            raise ValueError(f"snapshot arrays missing: {sorted(missing)}")
        snapshot = cls(**{name: arrays[name] for name in names})
        snapshot.validate()
        return snapshot

    def validate(self) -> None:  # noqa: PLR0912 - a flat list of shape and range checks
        """Shape and range checks run on every load (a snapshot is a data contract too)."""
        u, i, c = self.n_users, self.n_items, self.n_categories
        for name in ("user_impressions", "user_ctr"):
            if getattr(self, name).shape != (u,):
                raise ValueError(f"snapshot {name} has the wrong shape")
        for name in ("user_cat_ctr", "user_cat_recency"):
            if getattr(self, name).shape != (u, c):
                raise ValueError(f"snapshot {name} has the wrong shape")
        for name in ("item_impressions", "item_clickers", "item_ctr", "item_launch_day"):
            if getattr(self, name).shape != (i,):
                raise ValueError(f"snapshot {name} has the wrong shape")
        if self.seen_indptr.shape != (u + 1,) or int(self.seen_indptr[-1]) != len(
            self.seen_indices
        ):
            raise ValueError("snapshot seen-item index is inconsistent")
        if np.any(np.diff(self.seen_indptr) < 0):
            raise ValueError("snapshot seen-item offsets must be non-decreasing")
        for name in ("seen_indices", "popular_items"):
            values = getattr(self, name)
            if values.size and (values.min() < 0 or values.max() >= i):
                raise ValueError(f"snapshot {name} references unknown items")
        if self.item_category.size and (
            self.item_category.min() < 0 or self.item_category.max() >= c
        ):
            raise ValueError("snapshot item categories are out of range")
        for name in ("user_ctr", "item_ctr", "user_cat_ctr", "user_cat_share"):
            values = getattr(self, name)
            if not np.all(np.isfinite(values)) or values.min(initial=0.0) < 0:
                raise ValueError(f"snapshot {name} has invalid values")
            if values.max(initial=0.0) > 1.0 + 1e-6:
                raise ValueError(f"snapshot {name} exceeds 1")

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for name, values in sorted(self.arrays().items()):
            array = np.ascontiguousarray(values)
            digest.update(f"{name}:{array.dtype.str}:{array.shape}".encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


def _frame(interactions: Interactions) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": interactions.session_id.astype(np.int64),
            "user_id": interactions.user_id.astype(np.int64),
            "item_id": interactions.item_id.astype(np.int64),
            "clicked": interactions.clicked.astype(np.int64),
            "day": interactions.day.astype(np.int64),
        }
    )


def build_snapshot(  # noqa: PLR0915 - one aggregation pass, split would hide the SQL
    interactions: Interactions,
    catalog: Catalog,
    *,
    n_users: int,
    snapshot_day: int | None = None,
    config: SnapshotConfig | None = None,
) -> FeatureSnapshot:
    """Aggregate ``interactions`` into a snapshot. Users with id >= ``n_users`` (for example
    accounts created after the tower was trained) contribute to item statistics only."""
    cfg = config or SnapshotConfig()
    n_items, n_cat = catalog.n_items, catalog.n_categories
    day = int(interactions.day.max()) + 1 if snapshot_day is None else int(snapshot_day)
    con = duckdb.connect(database=":memory:")
    try:
        # Single-threaded, insertion-ordered: the aggregates below must be bit-reproducible,
        # because the snapshot's content hash is part of the bundle manifest.
        con.execute("SET threads TO 1")
        con.execute("SET preserve_insertion_order TO true")
        con.register("imp", _frame(interactions))
        con.register("cat", catalog.to_frame())
        users = con.execute(
            """
            SELECT user_id, count(*) AS impressions, sum(clicked) AS clicks
            FROM imp WHERE user_id < ? GROUP BY user_id
            """,
            [n_users],
        ).fetchnumpy()
        user_cat = con.execute(
            """
            SELECT i.user_id, c.category, count(*) AS impressions, sum(i.clicked) AS clicks,
                   coalesce(max(CASE WHEN i.clicked = 1 THEN i.day END), -1) AS last_click_day
            FROM imp i JOIN cat c ON i.item_id = c.item_id
            WHERE i.user_id < ? GROUP BY i.user_id, c.category
            """,
            [n_users],
        ).fetchnumpy()
        if cfg.robust:
            items = con.execute(
                """
                WITH tenure AS (
                    SELECT user_id,
                           least(1.0, pow(count(DISTINCT session_id) / ?, ?)) AS w
                    FROM imp GROUP BY user_id
                ), pairs AS (
                    SELECT user_id, item_id, max(clicked) AS clicked
                    FROM imp GROUP BY user_id, item_id
                )
                SELECT p.item_id, sum(t.w) AS impressions, sum(t.w * p.clicked) AS clicks
                FROM pairs p JOIN tenure t ON p.user_id = t.user_id
                GROUP BY p.item_id
                """,
                [float(cfg.trust_sessions), float(cfg.tenure_exponent)],
            ).fetchnumpy()
        else:
            items = con.execute(
                """
                SELECT item_id, count(*) AS impressions, sum(clicked) AS clicks
                FROM imp GROUP BY item_id
                """
            ).fetchnumpy()
        seen = con.execute(
            """
            SELECT DISTINCT user_id, item_id FROM imp
            WHERE clicked = 1 AND user_id < ? ORDER BY user_id, item_id
            """,
            [n_users],
        ).fetchnumpy()
    finally:
        con.close()

    user_impr = np.zeros(n_users, dtype=np.float64)
    user_clicks = np.zeros(n_users, dtype=np.float64)
    user_impr[users["user_id"]] = users["impressions"]
    user_clicks[users["user_id"]] = users["clicks"]
    total_impr = max(1.0, float(user_impr.sum()))
    global_ctr = float(user_clicks.sum()) / total_impr
    m_user = cfg.user_ctr_prior_strength
    user_ctr = (user_clicks + m_user * global_ctr) / (user_impr + m_user)

    uc_impr = np.zeros((n_users, n_cat), dtype=np.float64)
    uc_clicks = np.zeros((n_users, n_cat), dtype=np.float64)
    uc_last = np.full((n_users, n_cat), -1.0)
    rows, cols = user_cat["user_id"], user_cat["category"]
    uc_impr[rows, cols] = user_cat["impressions"]
    uc_clicks[rows, cols] = user_cat["clicks"]
    uc_last[rows, cols] = user_cat["last_click_day"]
    a = cfg.share_prior_strength
    uc_share = (uc_clicks + a / n_cat) / (user_clicks[:, None] + a)
    m_cat = cfg.category_ctr_prior_strength
    uc_ctr = (uc_clicks + m_cat * user_ctr[:, None]) / (uc_impr + m_cat)
    uc_recency = np.where(uc_last >= 0, np.clip(day - uc_last, 0, _MAX_RECENCY), _MAX_RECENCY)

    item_impr = np.zeros(n_items, dtype=np.float64)
    item_clicks = np.zeros(n_items, dtype=np.float64)
    item_impr[items["item_id"]] = items["impressions"]
    item_clicks[items["item_id"]] = items["clicks"]
    # Only the robust build shrinks item CTR; the naive one reports the raw ratio, falling back to
    # the global rate for items nobody has seen so the column stays finite.
    m_item = cfg.ctr_prior_strength if cfg.robust else 0.0
    item_ctr = np.clip(
        np.divide(
            item_clicks + m_item * global_ctr,
            item_impr + m_item,
            out=np.full(n_items, global_ctr),
            where=(item_impr + m_item) > 0,
        ),
        0.0,
        1.0,
    )

    seen_users = np.asarray(seen["user_id"], dtype=np.int64)
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(np.bincount(seen_users, minlength=n_users), out=indptr[1:])
    order = np.lexsort(
        (
            np.arange(n_items),
            -np.round(item_ctr, _POPULARITY_DECIMALS),
            -np.round(item_clicks, _POPULARITY_DECIMALS),
        )
    )
    popular = order[: min(cfg.popular_list_size, n_items)]

    snapshot = FeatureSnapshot(
        snapshot_day=np.array([day], dtype=np.int32),
        global_ctr=np.array([global_ctr], dtype=np.float32),
        user_clicks=user_clicks.astype(np.float32),
        user_impressions=user_impr.astype(np.float32),
        user_ctr=user_ctr.astype(np.float32),
        user_cat_share=uc_share.astype(np.float32),
        user_cat_ctr=uc_ctr.astype(np.float32),
        user_cat_recency=uc_recency.astype(np.float32),
        item_impressions=item_impr.astype(np.float32),
        item_clickers=item_clicks.astype(np.float32),
        item_ctr=item_ctr.astype(np.float32),
        item_category=catalog.item_category.astype(np.int32),
        item_launch_day=catalog.item_launch_day.astype(np.int32),
        seen_indptr=indptr,
        seen_indices=np.asarray(seen["item_id"], dtype=np.int32),
        popular_items=popular.astype(np.int32),
    )
    snapshot.validate()
    return snapshot


def user_side_features(snapshot: FeatureSnapshot) -> npt.NDArray[np.float32]:
    """User-tower side features: category click shares plus activity levels."""
    activity = np.stack(
        [
            np.log1p(snapshot.user_clicks) / _LOG_SCALE,
            np.log1p(snapshot.user_impressions) / _LOG_SCALE,
        ],
        axis=1,
    )
    return np.concatenate([snapshot.user_cat_share, activity], axis=1).astype(np.float32)


def item_side_features(snapshot: FeatureSnapshot) -> npt.NDArray[np.float32]:
    """Item-tower side features: category one-hot plus popularity levels."""
    onehot = np.zeros((snapshot.n_items, snapshot.n_categories), dtype=np.float32)
    onehot[np.arange(snapshot.n_items), snapshot.item_category] = 1.0
    popularity = np.stack(
        [
            np.log1p(snapshot.item_impressions) / _LOG_SCALE,
            np.log1p(snapshot.item_clickers) / _LOG_SCALE,
        ],
        axis=1,
    )
    return np.concatenate([onehot, popularity], axis=1).astype(np.float32)


def assemble_features(
    snapshot: FeatureSnapshot,
    users: npt.ArrayLike,
    items: npt.ArrayLike,
    tower_scores: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Ranker features for (user, item) rows in :data:`FEATURE_NAMES` order.

    ``users`` may be a scalar (one request) or an array aligned with ``items`` (training). The
    same function serves both paths, so training and serving features cannot drift apart.
    """
    item = np.asarray(items, dtype=np.int64)
    user = np.broadcast_to(np.asarray(users, dtype=np.int64), item.shape)
    category = snapshot.item_category[item]
    out = np.empty((item.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    out[:, 0] = np.asarray(tower_scores, dtype=np.float32)
    out[:, 1] = np.log1p(snapshot.item_clickers[item])
    out[:, 2] = snapshot.item_ctr[item]
    out[:, 3] = np.clip(snapshot.day - snapshot.item_launch_day[item], 0, _MAX_RECENCY)
    out[:, 4] = snapshot.user_cat_share[user, category]
    out[:, 5] = snapshot.user_cat_ctr[user, category]
    out[:, 6] = snapshot.user_cat_recency[user, category]
    out[:, 7] = np.log1p(snapshot.user_clicks[user])
    out[:, 8] = snapshot.user_ctr[user]
    out[:, 9] = np.log1p(snapshot.item_impressions[item])
    return out
