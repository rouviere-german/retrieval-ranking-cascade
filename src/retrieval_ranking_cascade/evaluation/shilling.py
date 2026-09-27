"""Shilling-attack robustness: can fake accounts push a chosen item up the ranking?

The attack. An adversary creates ``attackers`` fresh accounts. Each one clicks the target item and
a handful of popular *filler* items, which is the classic bandwagon profile (Lam and Riedl,
"Shilling Recommender Systems for Fun and Profit", WWW 2004; Mobasher et al., "Toward Trustworthy
Recommender Systems", ACM TOIT 2007). The clicks are legitimate-looking events, so no input
validation can reject them; only the way statistics are aggregated decides whether they work.

The defence lives in :func:`ranking.features.build_snapshot` with ``robust=True``:

* one click per (user, item) pair, so repeating a click buys nothing;
* per-user tenure weight ``min(1, sessions / trust_sessions)``, so a brand-new account counts for a
  fraction of an established one;
* item CTR shrunk towards the global CTR with a prior, so a small number of impressions cannot
  produce an extreme rate.

``robust=False`` rebuilds the same snapshot with raw counts and no CTR shrinkage. Running the attack
against both, with everything else identical, isolates the effect of the aggregation choice.

What to measure. The obvious quantity, rank *shift*, is misleading: the two aggregations produce
different baseline rankings, so equal shifts can end in completely different places. What an
attacker wants is to reach the top, so the primary number here is the target item's **rank after**
the attack, and the comparison between aggregations is the ratio of *lifts*
``(rank_before / rank_after)``: how many times closer to the top the attack got the item. Rank shift
is still reported, for continuity with how these attacks are usually described.

The rank is the full-catalogue ranker rank, not the served position, because a successful attack
shows up as a lift in the underlying score long before the item reaches a page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from retrieval_ranking_cascade.data.generator import Catalog, Interactions
from retrieval_ranking_cascade.ranking.features import (
    FeatureSnapshot,
    SnapshotConfig,
    assemble_features,
    build_snapshot,
    item_side_features,
    user_side_features,
)
from retrieval_ranking_cascade.ranking.ranker import Ranker
from retrieval_ranking_cascade.retrieval.two_tower import TwoTowerModel

__all__ = [
    "AttackConfig",
    "AttackOutcome",
    "ShillingReport",
    "inject_shilling_profiles",
    "mean_item_rank",
    "run_shilling_attack",
]


@dataclass(frozen=True, slots=True)
class AttackConfig:
    """Shape of the injected profiles."""

    target_item: int
    attackers: int = 200
    filler_items: int = 8
    sessions_per_attacker: int = 1
    sample_users: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.target_item < 0:
            raise ValueError("target_item must be a valid item id")
        if self.attackers < 1 or self.filler_items < 0 or self.sessions_per_attacker < 1:
            raise ValueError("attackers and sessions_per_attacker must be >= 1")
        if self.sample_users < 1:
            raise ValueError("sample_users must be >= 1")


@dataclass(frozen=True, slots=True)
class AttackOutcome:
    """What the attack achieved against one aggregation policy."""

    aggregation: str
    rank_before: float
    rank_after: float
    clickers_before: float
    clickers_after: float
    ctr_before: float
    ctr_after: float

    @property
    def rank_shift(self) -> float:
        """Positions gained by the target item (positive means the attack moved it up)."""
        return self.rank_before - self.rank_after

    @property
    def lift(self) -> float:
        """How many times closer to rank 1 the attack got the item."""
        return float(self.rank_before / max(self.rank_after, 1.0))

    def as_dict(self) -> dict[str, float | str]:
        return {
            "aggregation": self.aggregation,
            "rank_before": round(self.rank_before, 2),
            "rank_after": round(self.rank_after, 2),
            "rank_shift": round(self.rank_shift, 2),
            "lift": round(self.lift, 3),
            "clickers_before": round(self.clickers_before, 3),
            "clickers_after": round(self.clickers_after, 3),
            "ctr_before": round(self.ctr_before, 6),
            "ctr_after": round(self.ctr_after, 6),
        }


@dataclass(frozen=True, slots=True)
class ShillingReport:
    config: AttackConfig
    robust: AttackOutcome
    naive: AttackOutcome

    @property
    def damping(self) -> float | None:
        """Naive lift divided by robust lift (None when the naive attack achieved nothing)."""
        if self.naive.lift <= 1.0:
            return None
        return float(self.naive.lift / max(self.robust.lift, 1e-9))

    def as_dict(self) -> dict[str, object]:
        return {
            "target_item": self.config.target_item,
            "attackers": self.config.attackers,
            "sessions_per_attacker": self.config.sessions_per_attacker,
            "filler_items": self.config.filler_items,
            "sample_users": self.config.sample_users,
            "robust": self.robust.as_dict(),
            "naive": self.naive.as_dict(),
            "damping": None if self.damping is None else round(self.damping, 2),
        }


def inject_shilling_profiles(
    interactions: Interactions,
    *,
    config: AttackConfig,
    n_users: int,
    slots: int,
    popular_items: npt.NDArray[np.int32],
) -> Interactions:
    """Append fake sessions from new accounts (ids ``>= n_users``) that all click the target."""
    rng = np.random.default_rng(config.seed)
    filler_pool = np.asarray(
        [int(item) for item in popular_items if int(item) != config.target_item], dtype=np.int64
    )
    n_filler = min(config.filler_items, filler_pool.shape[0], slots - 1)
    if n_filler < 0:
        raise ValueError("slots must be >= 1")
    rows_per_session = 1 + n_filler
    total_sessions = config.attackers * config.sessions_per_attacker
    total_rows = total_sessions * rows_per_session
    next_session = int(interactions.session_id.max()) + 1 if len(interactions) else 0
    day = int(interactions.day.max()) if len(interactions) else 0

    users = np.repeat(
        n_users + np.arange(config.attackers, dtype=np.int64),
        config.sessions_per_attacker * rows_per_session,
    )
    session_ids = np.repeat(
        next_session + np.arange(total_sessions, dtype=np.int64), rows_per_session
    )
    items = np.empty(total_rows, dtype=np.int64)
    positions = np.empty(total_rows, dtype=np.int64)
    clicked = np.zeros(total_rows, dtype=np.int64)
    for index in range(total_sessions):
        start = index * rows_per_session
        # The target sits in a plausible middle slot, not always first.
        target_slot = int(rng.integers(0, rows_per_session))
        filler = (
            rng.choice(filler_pool, size=n_filler, replace=False) if n_filler else filler_pool[:0]
        )
        slate = np.insert(filler, target_slot, config.target_item)
        items[start : start + rows_per_session] = slate
        positions[start : start + rows_per_session] = np.arange(1, rows_per_session + 1)
        clicked[start : start + rows_per_session] = 1

    injected = Interactions(
        session_id=session_ids.astype(np.int32),
        user_id=users.astype(np.int32),
        item_id=items.astype(np.int32),
        position=positions.astype(np.int8),
        clicked=clicked.astype(np.int8),
        randomized=np.zeros(total_rows, dtype=np.bool_),
        day=np.full(total_rows, day, dtype=np.int16),
        period=np.zeros(total_rows, dtype=np.int8),
    )
    return Interactions(
        **{
            name: np.concatenate([getattr(interactions, name), getattr(injected, name)])
            for name in interactions.arrays()
        }
    )


def mean_item_rank(
    *,
    snapshot: FeatureSnapshot,
    model: TwoTowerModel,
    ranker: Ranker,
    item: int,
    users: npt.NDArray[np.int64],
) -> float:
    """Mean full-catalogue rank (1 = best) of ``item`` across ``users`` under ``snapshot``."""
    item_side = item_side_features(snapshot)
    user_side = user_side_features(snapshot)
    item_vectors, _ = model.item.embed(np.arange(snapshot.n_items), item_side)
    user_vectors, _ = model.user.embed(users, user_side[users])
    all_items = np.arange(snapshot.n_items, dtype=np.int64)
    ranks = np.empty(users.shape[0], dtype=np.float64)
    for position, user in enumerate(users):
        tower = (item_vectors @ user_vectors[position]).astype(np.float32)
        scores = ranker.predict(assemble_features(snapshot, int(user), all_items, tower))
        ranks[position] = float(1 + np.sum(scores > scores[item]))
    return float(ranks.mean())


def _outcome(
    *,
    aggregation: str,
    before: FeatureSnapshot,
    after: FeatureSnapshot,
    model: TwoTowerModel,
    ranker: Ranker,
    config: AttackConfig,
    users: npt.NDArray[np.int64],
) -> AttackOutcome:
    target = config.target_item
    return AttackOutcome(
        aggregation=aggregation,
        rank_before=mean_item_rank(
            snapshot=before, model=model, ranker=ranker, item=target, users=users
        ),
        rank_after=mean_item_rank(
            snapshot=after, model=model, ranker=ranker, item=target, users=users
        ),
        clickers_before=float(before.item_clickers[target]),
        clickers_after=float(after.item_clickers[target]),
        ctr_before=float(before.item_ctr[target]),
        ctr_after=float(after.item_ctr[target]),
    )


def run_shilling_attack(
    *,
    interactions: Interactions,
    catalog: Catalog,
    model: TwoTowerModel,
    ranker: Ranker,
    n_users: int,
    slots: int,
    config: AttackConfig,
    snapshot_config: SnapshotConfig | None = None,
) -> ShillingReport:
    """Rebuild the snapshot with and without the injected profiles, robust and naive."""
    base_config = snapshot_config or SnapshotConfig()
    robust_config = SnapshotConfig(**{**_as_dict(base_config), "robust": True})
    naive_config = SnapshotConfig(**{**_as_dict(base_config), "robust": False})
    clean_robust = build_snapshot(interactions, catalog, n_users=n_users, config=robust_config)
    attacked = inject_shilling_profiles(
        interactions,
        config=config,
        n_users=n_users,
        slots=slots,
        popular_items=clean_robust.popular_items,
    )
    rng = np.random.default_rng(config.seed + 1)
    users = np.sort(
        rng.choice(n_users, size=min(config.sample_users, n_users), replace=False)
    ).astype(np.int64)
    snapshots = {
        "robust": (
            clean_robust,
            build_snapshot(attacked, catalog, n_users=n_users, config=robust_config),
        ),
        "naive": (
            build_snapshot(interactions, catalog, n_users=n_users, config=naive_config),
            build_snapshot(attacked, catalog, n_users=n_users, config=naive_config),
        ),
    }
    outcomes = {
        name: _outcome(
            aggregation=name,
            before=before,
            after=after,
            model=model,
            ranker=ranker,
            config=config,
            users=users,
        )
        for name, (before, after) in snapshots.items()
    }
    return ShillingReport(config=config, robust=outcomes["robust"], naive=outcomes["naive"])


def _as_dict(config: SnapshotConfig) -> dict[str, Any]:
    return {name: getattr(config, name) for name in SnapshotConfig.__dataclass_fields__}
