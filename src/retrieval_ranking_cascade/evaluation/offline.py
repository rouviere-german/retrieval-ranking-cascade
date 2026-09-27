"""Offline evaluation under two protocols, with the baselines the cascade has to beat.

**Protocol A, slate reranking.** Take each held-out session, keep exactly the items the logging
policy showed, and ask each policy to reorder them. This is the only protocol click data can
support honestly: an item the logging policy never showed has no click to be right or wrong about.
Every metric is reported three ways - naive (clicks as gains), IPS (clicks weighted by the inverse
of their logged position's propensity) and ground truth (the world's true relevance as gains). The
naive and IPS numbers are what a real team would have; the ground-truth number is the answer key
this synthetic world can provide, and the distance between an estimate and it is the bias.

**Protocol B, full corpus.** Sample users, let each policy pick k items out of the whole catalogue,
and score the picks against true relevance. This is where retrieval matters: a policy can only
recommend what it retrieves. Four policies are compared -

* ``popularity`` - the snapshot's popularity list, the cheapest sensible baseline;
* ``tower`` - ANN order, i.e. the cascade with reranking switched off (its degraded path);
* ``cascade_relevance`` - the full cascade with MMR disabled (``lambda = 1``), which is what the
  reranker is worth before any diversity is traded for it;
* ``cascade`` - the served configuration: ANN, LambdaMART, then MMR and the category cap. It scores
  *below* ``cascade_relevance`` on relevance and above it on coverage and diversity; that
  difference is the price of the diversity rules, reported rather than hidden;
* ``exact_rank`` - the LambdaMART model applied to *every* item in the catalogue. Unservable at
  50 ms, it is the upper bound the cascade is trying to approximate, and the gap between it and
  ``cascade`` is the price of the cascade's two approximations (a finite candidate set and an
  approximate search inside it).

Because the ideal ranking here is over the full catalogue, NDCG is normalized by the best k items
a perfect policy could have chosen, so an NDCG of 1.0 would mean "exactly the k most relevant
items in the catalogue, in order".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from retrieval_ranking_cascade.data.generator import Interactions, World
from retrieval_ranking_cascade.evaluation.metrics import (
    catalog_coverage,
    mean_reciprocal_rank,
    ndcg,
)
from retrieval_ranking_cascade.evaluation.propensity import PropensityModel
from retrieval_ranking_cascade.ranking.features import assemble_features
from retrieval_ranking_cascade.ranking.rules import filter_seen, intra_list_similarity
from retrieval_ranking_cascade.serving.cascade import Cascade

__all__ = [
    "SLATE_POLICIES",
    "PositionBiasReport",
    "SlateReport",
    "corpus_evaluation",
    "make_user_sample",
    "position_bias_report",
    "slate_evaluation",
]

SLATE_POLICIES = ("logged", "popularity", "tower", "cascade")
FArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SlateReport:
    """Per-policy metrics plus how well each click-based estimator recovers relative lift.

    Comparing the *level* of a click-based NDCG with a relevance-based one is meaningless: binary
    clicks and continuous relevance have different gain scales, and the difference between them is
    dominated by that, not by position bias. What offline evaluation is actually used for is the
    relative question - is this ranker better than the one in production, and by how much - so the
    bias reported here is over relative lift against the logged policy,

        advantage(P) = ndcg(P) / ndcg(logged) - 1

    computed once from the estimator and once from ground truth. It is scale-free, and a smaller
    value means the estimator would have led a team to the conclusion the truth supports.
    """

    sessions: int
    metrics: dict[str, dict[str, float]]
    naive_bias: float
    ips_bias: float
    advantage: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, object]:
        return {
            "sessions": self.sessions,
            "metrics": {
                policy: {name: round(value, 6) for name, value in sorted(values.items())}
                for policy, values in sorted(self.metrics.items())
            },
            "advantage_over_logged": {
                policy: {name: round(value, 6) for name, value in sorted(values.items())}
                for policy, values in sorted(self.advantage.items())
            },
            "naive_advantage_bias": round(self.naive_bias, 6),
            "ips_advantage_bias": round(self.ips_bias, 6),
        }


@dataclass(frozen=True, slots=True)
class PositionBiasReport:
    """Per-rank relevance, estimated from clicks naively and with IPS, against the truth.

    This is the cleanest of the two bias measurements: under the position-based model the click
    rate at rank ``r`` equals ``p_r * E[rel]``, so the naive estimate decays down the page even when
    the items at every rank are equally relevant, while dividing by ``p_r`` recovers the relevance
    level. No ranking and no gain scale are involved.
    """

    ranks: tuple[int, ...]
    naive: tuple[float, ...]
    ips: tuple[float, ...]
    truth: tuple[float, ...]

    @property
    def naive_bias(self) -> float:
        return float(np.mean(np.abs(np.asarray(self.naive) - np.asarray(self.truth))))

    @property
    def ips_bias(self) -> float:
        return float(np.mean(np.abs(np.asarray(self.ips) - np.asarray(self.truth))))

    def as_dict(self) -> dict[str, object]:
        return {
            "ranks": list(self.ranks),
            "naive_relevance": [round(value, 6) for value in self.naive],
            "ips_relevance": [round(value, 6) for value in self.ips],
            "true_relevance": [round(value, 6) for value in self.truth],
            "naive_position_bias": round(self.naive_bias, 6),
            "ips_position_bias": round(self.ips_bias, 6),
        }


def position_bias_report(
    interactions: Interactions,
    world: World,
    propensity: PropensityModel,
    *,
    slots: int,
    exclude_randomized: bool = True,
) -> PositionBiasReport:
    """Estimate mean relevance per rank from clicks, naively and with IPS, and compare to truth.

    The randomized slice is excluded by default: it is the data the propensities were fitted on, so
    including it would let the estimator grade its own homework.
    """
    rows = interactions.select(~interactions.randomized) if exclude_randomized else interactions
    if len(rows) == 0:
        raise ValueError("no impressions left to measure position bias on")
    positions = rows.position.astype(np.int64)
    clicked = rows.clicked.astype(np.float64)
    weights = propensity.weight(positions)
    truth = world.relevance(rows.user_id.astype(np.int64), rows.item_id.astype(np.int64))
    ranks: list[int] = []
    naive: list[float] = []
    ips: list[float] = []
    actual: list[float] = []
    for rank in range(1, slots + 1):
        mask = positions == rank
        if not mask.any():
            continue
        ranks.append(rank)
        naive.append(float(clicked[mask].mean()))
        ips.append(float((clicked[mask] * weights[mask]).mean()))
        actual.append(float(truth[mask].mean()))
    return PositionBiasReport(
        ranks=tuple(ranks), naive=tuple(naive), ips=tuple(ips), truth=tuple(actual)
    )


def _sessions(interactions: Interactions) -> list[tuple[int, int, int]]:
    """(user_id, start, end) for each session, sorted by session then position."""
    order = np.lexsort((interactions.position, interactions.session_id))
    ordered = interactions.select(order)
    starts = np.flatnonzero(np.r_[True, ordered.session_id[1:] != ordered.session_id[:-1]])
    ends = np.r_[starts[1:], len(ordered)]
    return [
        (int(ordered.user_id[start]), int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
    ]


def slate_evaluation(
    *,
    sessions: Interactions,
    cascade: Cascade,
    world: World,
    propensity: PropensityModel,
    k: int = 10,
    max_sessions: int = 2_000,
    seed: int = 0,
) -> SlateReport:
    """Rerank each held-out slate with every policy and score it three ways."""
    if k < 1:
        raise ValueError("k must be >= 1")
    ordered = sessions.select(np.lexsort((sessions.position, sessions.session_id)))
    groups = _sessions(sessions)
    bundle = cascade.bundle
    rng = np.random.default_rng(seed)
    usable = [
        (user, start, end)
        for user, start, end in groups
        if 0 <= user < bundle.n_users and 0 < int(ordered.clicked[start:end].sum()) < end - start
    ]
    if not usable:
        raise ValueError("no held-out session has both clicked and unclicked items")
    if len(usable) > max_sessions:
        picked = rng.choice(len(usable), size=max_sessions, replace=False)
        usable = [usable[int(index)] for index in sorted(picked)]

    names = ("ndcg_naive", "ndcg_ips", "ndcg_truth", "mrr_naive", "mrr_truth")
    totals: dict[str, dict[str, float]] = {
        policy: dict.fromkeys(names, 0.0) for policy in SLATE_POLICIES
    }
    for user, start, end in usable:
        items = ordered.item_id[start:end].astype(np.int64)
        clicked = ordered.clicked[start:end].astype(np.float64)
        positions = ordered.position[start:end].astype(np.int64)
        weights = clicked * propensity.weight(positions)
        truth = world.relevance(np.full(items.shape, user), items)
        user_vector = bundle.user_vectors(np.asarray([user]))
        tower = (bundle.item_vectors[items] @ user_vector[0]).astype(np.float64)
        orders = {
            "logged": np.arange(items.shape[0]),
            "popularity": np.argsort(-bundle.snapshot.item_clickers[items], kind="stable"),
            "tower": np.argsort(-tower, kind="stable"),
            "cascade": np.argsort(-cascade.rerank_slate(user, items), kind="stable"),
        }
        for policy, order in orders.items():
            bucket = totals[policy]
            bucket["ndcg_naive"] += ndcg(clicked[order], k)
            bucket["ndcg_ips"] += ndcg(weights[order], k)
            bucket["ndcg_truth"] += ndcg(truth[order], k)
            bucket["mrr_naive"] += mean_reciprocal_rank(clicked[order] > 0, k)
            # Ground-truth MRR: where the slate's single most relevant item ended up.
            bucket["mrr_truth"] += mean_reciprocal_rank(truth[order] >= truth.max(), k)
    count = float(len(usable))
    metrics = {
        policy: {name: value / count for name, value in values.items()}
        for policy, values in totals.items()
    }

    def advantage(metric: str, policy: str) -> float:
        base = metrics["logged"][metric]
        return metrics[policy][metric] / base - 1.0 if base > 0.0 else 0.0

    others = [policy for policy in SLATE_POLICIES if policy != "logged"]
    advantages = {
        policy: {
            "naive": advantage("ndcg_naive", policy),
            "ips": advantage("ndcg_ips", policy),
            "truth": advantage("ndcg_truth", policy),
        }
        for policy in others
    }
    naive_bias = float(np.mean([abs(row["naive"] - row["truth"]) for row in advantages.values()]))
    ips_bias = float(np.mean([abs(row["ips"] - row["truth"]) for row in advantages.values()]))
    return SlateReport(
        sessions=len(usable),
        metrics=metrics,
        naive_bias=naive_bias,
        ips_bias=ips_bias,
        advantage=advantages,
    )


def _exact_rank_top_k(
    cascade: Cascade, user: int, k: int, blocked: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Score every catalogue item with the ranker: the unservable upper bound."""
    bundle = cascade.bundle
    items = np.arange(bundle.n_items, dtype=np.int64)
    if blocked.size:
        items = items[~np.isin(items, blocked)]
    user_vector = bundle.user_vectors(np.asarray([user]))
    tower = (bundle.item_vectors[items] @ user_vector[0]).astype(np.float32)
    scores = bundle.ranker.predict(assemble_features(bundle.snapshot, user, items, tower))
    return items[np.argsort(-scores, kind="stable")[:k]]


def corpus_evaluation(
    *,
    cascade: Cascade,
    world: World,
    users: Sequence[int] | npt.NDArray[np.int64],
    k: int = 10,
    upper_bound_users: int = 40,
    cold_click_threshold: int = 1,
    ef_search: int | None = None,
) -> dict[str, object]:
    """Full-corpus top-k for each policy, scored against the world's true relevance."""
    if k < 1:
        raise ValueError("k must be >= 1")
    bundle = cascade.bundle
    sample = np.asarray(users, dtype=np.int64)
    if sample.size == 0:
        raise ValueError("users must not be empty")
    policies = ("popularity", "tower", "cascade_relevance", "cascade", "exact_rank")
    ndcgs: dict[str, list[float]] = {name: [] for name in policies}
    similarity: dict[str, list[float]] = {name: [] for name in policies}
    served: dict[str, list[int]] = {name: [] for name in policies}
    cold_item_share: dict[str, list[float]] = {name: [] for name in policies}
    cold_user_ndcg: dict[str, list[float]] = {name: [] for name in policies}
    launch = bundle.snapshot.item_launch_day
    for position, user in enumerate(sample):
        user_index = int(user)
        relevance = world.relevance_rows(np.asarray([user_index]))[0]
        seen = bundle.snapshot.seen(user_index).astype(np.int64)
        allowed = np.ones(bundle.n_items, dtype=bool)
        allowed[seen] = False
        ideal = np.sort(relevance[allowed])[::-1][:k]
        is_cold_user = int(bundle.snapshot.user_clicks[user_index]) <= cold_click_threshold

        lists: dict[str, npt.NDArray[np.int64]] = {
            "popularity": filter_seen(bundle.snapshot.popular_items, seen)[:k],
            "tower": _tower_top_k(cascade, user_index, k, seen, ef_search),
            # lambda = 1 switches MMR off, isolating the reranker from the diversity trade-off.
            "cascade_relevance": np.asarray(
                cascade.recommend(user_index, k=k, ef_search=ef_search, mmr_lambda=1.0).item_ids,
                dtype=np.int64,
            ),
            "cascade": np.asarray(
                cascade.recommend(user_index, k=k, ef_search=ef_search).item_ids, dtype=np.int64
            ),
        }
        if position < upper_bound_users:
            lists["exact_rank"] = _exact_rank_top_k(cascade, user_index, k, seen)
        for name, items in lists.items():
            if items.size == 0:
                continue
            score = ndcg(relevance[items], k, ideal_gains=ideal)
            ndcgs[name].append(score)
            similarity[name].append(intra_list_similarity(bundle.item_vectors[items]))
            served[name].extend(int(item) for item in items)
            cold_item_share[name].append(float(np.mean(launch[items] > 0)))
            if is_cold_user:
                cold_user_ndcg[name].append(score)

    report: dict[str, object] = {"users": int(sample.size), "k": k}
    for name in policies:
        if not ndcgs[name]:
            continue
        report[name] = {
            "ndcg_at_k": round(float(np.mean(ndcgs[name])), 6),
            "users": len(ndcgs[name]),
            "intra_list_similarity": round(float(np.mean(similarity[name])), 6),
            "catalog_coverage": round(catalog_coverage(served[name], bundle.n_items), 6),
            "cold_item_share": round(float(np.mean(cold_item_share[name])), 6),
            "cold_user_ndcg_at_k": (
                round(float(np.mean(cold_user_ndcg[name])), 6) if cold_user_ndcg[name] else None
            ),
            "cold_users": len(cold_user_ndcg[name]),
        }
    cascade_ndcg = float(np.mean(ndcgs["cascade_relevance"]))
    if ndcgs["exact_rank"]:
        bound = float(np.mean(ndcgs["exact_rank"]))
        shared = min(len(ndcgs["cascade_relevance"]), len(ndcgs["exact_rank"]))
        paired_cascade = float(np.mean(ndcgs["cascade_relevance"][:shared]))
        paired_bound = float(np.mean(ndcgs["exact_rank"][:shared]))
        report["upper_bound_gap"] = round(paired_bound - paired_cascade, 6)
        report["upper_bound_retained"] = (
            round(paired_cascade / paired_bound, 6) if paired_bound > 0 else None
        )
        report["upper_bound_ndcg"] = round(bound, 6)
    # The headline number compares like with like: the reranker without the diversity rules.
    report["cascade_ndcg_at_k"] = round(cascade_ndcg, 6)
    report["cascade_served_ndcg_at_k"] = round(float(np.mean(ndcgs["cascade"])), 6)
    report["diversity_cost"] = round(cascade_ndcg - float(np.mean(ndcgs["cascade"])), 6)
    return report


def _tower_top_k(
    cascade: Cascade,
    user: int,
    k: int,
    seen: npt.NDArray[np.int64],
    ef_search: int | None,
) -> npt.NDArray[np.int64]:
    """ANN order with the seen filter but no reranking: the cascade's degraded path."""
    bundle = cascade.bundle
    vector = bundle.user_vectors(np.asarray([user]))
    _, ids = cascade.index.search(
        vector,
        cascade.config.candidate_k,
        ef_search=ef_search if ef_search is not None else cascade.config.ef_search,
    )
    return filter_seen(ids[0], seen)[:k]


def make_user_sample(
    interactions: Interactions,
    n_users: int,
    size: int,
    *,
    seed: int = 0,
    predicate: Callable[[npt.NDArray[np.int64]], npt.NDArray[np.bool_]] | None = None,
) -> npt.NDArray[np.int64]:
    """Sample distinct users that appear in ``interactions`` and are known to the bundle."""
    candidates = np.unique(interactions.user_id.astype(np.int64))
    candidates = candidates[candidates < n_users]
    if predicate is not None:
        candidates = candidates[predicate(candidates)]
    if candidates.size == 0:
        raise ValueError("no eligible users to sample")
    rng = np.random.default_rng(seed)
    take = min(size, candidates.size)
    return np.sort(rng.choice(candidates, size=take, replace=False))
