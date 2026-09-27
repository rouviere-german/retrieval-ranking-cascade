"""LambdaMART reranker (LightGBM ``lambdarank``) and its training-set builder.

LambdaMART (Burges, "From RankNet to LambdaRank to LambdaMART: An Overview", MSR-TR-2010-82)
boosts regression trees on lambda gradients that weight each pairwise swap by its change in
NDCG. Queries are the users' *label-period* sessions: the items the logging policy showed, with
click labels, plus ``extra_negatives`` items the tower retrieved but that were never shown (label
0). The extra negatives make the training distribution resemble the serving distribution (ANN
candidates) instead of only the logging policy's choices.

Clicks are position-biased, and training on them naively produces a ranker that is *worse* than
the retrieval tower it is supposed to improve (see the ablation in ``bench/bench_quality.py``): the
gradient rewards whatever the logging policy happened to show near the top, which is mostly
popularity. Two corrections are available:

* ``ips_weights`` (the default) multiplies each shown row's contribution by the clipped inverse of
  its logged position's examination propensity, so the lambdarank objective estimates NDCG over
  *relevance* rather than over clicks (Joachims et al., WSDM 2017). Unshown extra negatives keep
  weight 1 - they were never examined, so there is no propensity to invert.
* ``position_debias`` instead hands the logged positions to LightGBM's unbiased lambdarank
  (``Dataset(position=...)``, after Hu et al., "Unbiased LambdaMART", WWW 2019), which fits position
  factors jointly. Measured here it helps the slate metric and badly hurts full-corpus ranking, so
  it is off by default and kept for the ablation.

The model is stored as LightGBM's text format and parsed only after its SHA-256 has been checked.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
from dataclasses import asdict, dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

from retrieval_ranking_cascade.data.generator import Interactions
from retrieval_ranking_cascade.evaluation.propensity import PropensityModel
from retrieval_ranking_cascade.ranking.features import (
    FEATURE_NAMES,
    FeatureSnapshot,
    assemble_features,
)

__all__ = [
    "Ranker",
    "RankerConfig",
    "RankerDataset",
    "RankerIntegrityError",
    "build_ranker_dataset",
    "ips_row_weights",
]


class RankerIntegrityError(ValueError):
    """The model text does not match its hash or its feature list."""


@dataclass(frozen=True, slots=True)
class RankerConfig:
    num_boost_round: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.9
    lambdarank_truncation_level: int = 20
    extra_negatives: int = 20
    candidate_pool: int = 100
    ips_weights: bool = True
    position_debias: bool = False
    threads: int = 4
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def lgb_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [10],
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "feature_fraction_seed": self.seed,
            "lambdarank_truncation_level": self.lambdarank_truncation_level,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": self.threads,
            "seed": self.seed,
            "verbose": -1,
        }
        if self.position_debias:
            params["lambdarank_position_bias_regularization"] = 0.0
        return params


@dataclass(frozen=True, slots=True)
class RankerDataset:
    features: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int32]
    groups: npt.NDArray[np.int32]
    positions: npt.NDArray[np.int32]
    weights: npt.NDArray[np.float64]
    users: npt.NDArray[np.int32]
    items: npt.NDArray[np.int32]

    @property
    def n_queries(self) -> int:
        return int(self.groups.shape[0])

    def __post_init__(self) -> None:
        lengths = {
            name: getattr(self, name).shape[0]
            for name in ("features", "labels", "positions", "weights", "users", "items")
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"ranker dataset columns have mismatched lengths: {lengths}")
        if int(np.sum(self.groups)) != self.labels.shape[0]:
            raise ValueError("group sizes do not sum to the number of rows")


def ips_row_weights(
    positions: npt.NDArray[np.int32], propensity: PropensityModel, slots: int
) -> npt.NDArray[np.float64]:
    """Clipped inverse propensity per row; rows in the ``slots`` bucket were never shown."""
    weights = np.ones(positions.shape[0], dtype=np.float64)
    shown = positions < slots
    if shown.any():
        weights[shown] = propensity.weight(positions[shown].astype(np.int64) + 1)
    return weights


def build_ranker_dataset(
    *,
    snapshot: FeatureSnapshot,
    label_sessions: Interactions,
    user_vectors: npt.NDArray[np.float32],
    item_vectors: npt.NDArray[np.float32],
    retrieved: npt.NDArray[np.int64],
    config: RankerConfig,
    slots: int,
    propensity: PropensityModel | None = None,
) -> RankerDataset:
    """One query per label-period session with at least one click.

    ``user_vectors[u]`` and ``retrieved[u]`` (tower top candidates, -1 padded) are indexed by
    user id; ``snapshot`` must be built from data strictly before the label period.
    """
    rng = np.random.default_rng(config.seed)
    order = np.lexsort((label_sessions.position, label_sessions.session_id))
    sessions = label_sessions.select(order)
    starts = np.flatnonzero(np.r_[True, sessions.session_id[1:] != sessions.session_id[:-1]])
    ends = np.r_[starts[1:], len(sessions)]
    users_out: list[npt.NDArray[np.int32]] = []
    items_out: list[npt.NDArray[np.int32]] = []
    labels_out: list[npt.NDArray[np.int32]] = []
    positions_out: list[npt.NDArray[np.int32]] = []
    groups: list[int] = []
    for start, end in zip(starts, ends, strict=True):
        clicked = sessions.clicked[start:end]
        if clicked.sum() == 0 or clicked.sum() == end - start:
            continue
        user = int(sessions.user_id[start])
        shown = sessions.item_id[start:end].astype(np.int32)
        pool = retrieved[user]
        pool = pool[pool >= 0]
        excluded = np.concatenate([shown, snapshot.seen(user)])
        pool = pool[~np.isin(pool, excluded)]
        n_extra = min(config.extra_negatives, pool.shape[0])
        extra = (
            rng.choice(pool, size=n_extra, replace=False).astype(np.int32)
            if n_extra
            else pool[:0].astype(np.int32)
        )
        items = np.concatenate([shown, extra])
        users_out.append(np.full(items.shape[0], user, dtype=np.int32))
        items_out.append(items)
        labels_out.append(
            np.concatenate([clicked.astype(np.int32), np.zeros(n_extra, dtype=np.int32)])
        )
        positions_out.append(
            np.concatenate(
                [
                    sessions.position[start:end].astype(np.int32) - 1,
                    np.full(n_extra, slots, dtype=np.int32),
                ]
            )
        )
        groups.append(int(items.shape[0]))
    if not groups:
        raise ValueError("no label-period session has both clicked and unclicked items")
    users = np.concatenate(users_out)
    items = np.concatenate(items_out)
    scores = np.einsum("ij,ij->i", user_vectors[users], item_vectors[items])
    positions = np.concatenate(positions_out)
    if config.ips_weights and propensity is None:
        raise ValueError("ips_weights needs a PropensityModel estimated from the randomized slice")
    weights = (
        ips_row_weights(positions, propensity, slots)
        if config.ips_weights and propensity is not None
        else np.ones(positions.shape[0], dtype=np.float64)
    )
    return RankerDataset(
        features=assemble_features(snapshot, users, items, scores),
        labels=np.concatenate(labels_out),
        groups=np.asarray(groups, dtype=np.int32),
        positions=positions,
        weights=weights,
        users=users,
        items=items,
    )


class Ranker:
    """A trained LightGBM booster plus the feature names it expects.

    Serving is multi-threaded, so each thread gets its own parsed ``Booster`` built from the same
    model text, held in a :class:`threading.local`. The alternative - one shared booster behind a
    mutex - serializes stage 2 and was measured at roughly 4.5x the per-call latency under eight
    concurrent callers (``bench/bench_baseline.py::bench_ranker_predict_threads``). A parsed copy of
    a 200-tree model is a few hundred kilobytes, which is the price of that.
    """

    def __init__(self, booster: Any) -> None:
        names = tuple(booster.feature_name())
        if names != FEATURE_NAMES:
            raise RankerIntegrityError("model feature names do not match FEATURE_NAMES")
        self._booster = booster
        self._text = str(booster.model_to_string())
        self._local = threading.local()

    @classmethod
    def train(cls, dataset: RankerDataset, config: RankerConfig) -> Ranker:
        train_set = lgb.Dataset(
            dataset.features,
            label=dataset.labels,
            group=dataset.groups,
            weight=dataset.weights if config.ips_weights else None,
            position=dataset.positions if config.position_debias else None,
            feature_name=list(FEATURE_NAMES),
            free_raw_data=True,
        )
        booster = lgb.train(config.lgb_params(), train_set, num_boost_round=config.num_boost_round)
        return cls(booster)

    @property
    def num_trees(self) -> int:
        return int(self._booster.num_trees())

    def _thread_booster(self) -> Any:
        booster = getattr(self._local, "booster", None)
        if booster is None:
            booster = lgb.Booster(model_str=self._text)
            self._local.booster = booster
        return booster

    def predict(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected features with {len(FEATURE_NAMES)} columns")
        if features.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        scores = self._thread_booster().predict(features, num_threads=1)
        return np.asarray(scores, dtype=np.float64)

    def feature_importance(self) -> dict[str, float]:
        gains = self._booster.feature_importance(importance_type="gain")
        return {name: float(value) for name, value in zip(FEATURE_NAMES, gains, strict=True)}

    def to_text(self) -> str:
        return self._text

    @classmethod
    def from_text(cls, text: str, *, expected_sha256: str) -> Ranker:
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(actual, expected_sha256):
            raise RankerIntegrityError("ranker model text does not match the expected sha256")
        return cls(lgb.Booster(model_str=text))
