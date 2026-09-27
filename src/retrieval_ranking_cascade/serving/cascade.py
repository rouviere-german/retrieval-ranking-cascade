"""The three-stage cascade: ANN retrieval, LambdaMART reranking, business rules.

    user id
      |  stage 1  user tower -> HNSW top-C, minus items the user already clicked
      v            (cheap, high recall, C ~ 500)
    candidates
      |  stage 2  feature lookup -> LambdaMART over the best D of them
      v            (expensive, high precision, D ~ 150)
    scored
      |  stage 3  MMR diversity -> category cap    (trivial, k rows)
      v
    k items

Why a cascade at all: ranking the whole catalogue with the LambdaMART model would cost
``n_items`` feature rows per request (100k rows at roughly 13 microseconds each), which no 50 ms
budget survives. Retrieval narrows 100k items to 500 for a fraction of a millisecond and the
expensive model only sees the best 150 of those. The price is a recall ceiling: an item the ANN
stage misses can never be recommended, which is why recall@C is measured and watched by a canary
rather than assumed.

Why retrieve C but rerank only D: the seen filter runs between the two stages, so the extra
candidates are what protects a heavy user - whose top results are mostly items they have already
clicked - from ending up with a short list. Spending stage 2 on items that were going to be
filtered out anyway is the one part of the cost that buys nothing. ``bench/bench_quality.py
--ablations`` reports NDCG against D so the default is a measured choice, not a guess.

Every stage is timed and every stage is optional under pressure. The budget controller
(:mod:`serving.budget`) decides before each stage whether the planned work fits in the time that
is left, lowering ``efSearch`` first and dropping the reranker second. A response that lost
quality carries ``degraded=True`` and the reasons, so a dashboard can tell a slow-but-correct
system from a fast-but-degraded one.

Cold start. A user id the bundle has never seen has no snapshot row, so stages 1 and 2 have
nothing to condition on; the request is answered from the snapshot's popularity list and flagged
``cold_start``. Known users always go through the towers, including users whose history is empty
(their id embedding is zero, so the tower output is content-only).
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from retrieval_ranking_cascade.core.logging import get_logger
from retrieval_ranking_cascade.ranking.features import assemble_features
from retrieval_ranking_cascade.ranking.rules import (
    allowed_positions,
    cap_per_group,
    intra_list_similarity,
    mmr_select,
)
from retrieval_ranking_cascade.serving.budget import BudgetController
from retrieval_ranking_cascade.serving.bundle import Bundle

__all__ = [
    "STAGE_RANKING",
    "STAGE_RETRIEVAL",
    "STAGE_RULES",
    "Cascade",
    "CascadeConfig",
    "CascadeResult",
    "Recommendation",
]

STAGE_RETRIEVAL: Final = "retrieval"
STAGE_RANKING: Final = "ranking"
STAGE_RULES: Final = "rules"

_log = get_logger("retrieval_ranking_cascade.cascade")


@dataclass(frozen=True, slots=True)
class CascadeConfig:
    """Serving-side knobs. Bounds are enforced by the API layer before they reach here."""

    candidate_k: int = 500
    rerank_depth: int = 150
    default_k: int = 20
    ef_search: int = 768
    mmr_lambda: float = 0.7
    max_per_category: int = 5

    def __post_init__(self) -> None:
        if (
            min(
                self.candidate_k,
                self.rerank_depth,
                self.default_k,
                self.ef_search,
                self.max_per_category,
            )
            < 1
        ):
            raise ValueError("cascade sizes must be >= 1")
        if self.rerank_depth > self.candidate_k:
            raise ValueError("rerank_depth must not exceed candidate_k")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class Recommendation:
    item_id: int
    rank: int
    score: float
    category: int


@dataclass(frozen=True, slots=True)
class CascadeResult:
    """One response, with enough provenance to explain how it was produced."""

    items: tuple[Recommendation, ...]
    bundle_id: str
    degraded: bool
    degraded_reasons: tuple[str, ...]
    reranked: bool
    cold_start: bool
    candidates: int
    reranked_candidates: int
    ef_search: int
    queued_seconds: float = 0.0
    contention: float = 1.0
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def item_ids(self) -> list[int]:
        return [item.item_id for item in self.items]

    @property
    def total_seconds(self) -> float:
        return sum(self.stage_seconds.values())


class Cascade:
    """Serves recommendations from one bundle. Safe to call from several threads."""

    def __init__(
        self,
        bundle: Bundle,
        controller: BudgetController,
        config: CascadeConfig | None = None,
    ) -> None:
        self._bundle = bundle
        self._controller = controller
        self.config = config or CascadeConfig()
        self._index = bundle.index
        self._lock = threading.Lock()

    # --- live bundle / index state ---

    @property
    def bundle(self) -> Bundle:
        with self._lock:
            return self._bundle

    @property
    def index(self) -> Any:
        """The index currently serving traffic (a reindex replaces it atomically)."""
        with self._lock:
            return self._index

    def item_vectors(self) -> npt.NDArray[np.float32]:
        return self.bundle.item_vectors

    def swap_bundle(self, bundle: Bundle) -> str:
        """Point the cascade at a new bundle. Returns the new bundle id."""
        with self._lock:
            self._bundle = bundle
            self._index = bundle.index
        return bundle.bundle_id

    def swap_index(self, index: Any) -> None:
        """Replace only the ANN index (used by ``/v1/admin/reindex`` and the corruption drill)."""
        with self._lock:
            self._index = index

    # --- the cascade ---

    def recommend(
        self,
        user_id: int,
        *,
        k: int | None = None,
        ef_search: int | None = None,
        candidate_k: int | None = None,
        mmr_lambda: float | None = None,
        queued_seconds: float = 0.0,
        contention: float = 1.0,
    ) -> CascadeResult:
        bundle = self.bundle
        index = self.index
        cfg = self.config
        want = cfg.default_k if k is None else int(k)
        if want < 1:
            raise ValueError("k must be >= 1")
        lam = cfg.mmr_lambda if mmr_lambda is None else float(mmr_lambda)
        budget = self._controller.start(queued_seconds, contention)

        if not 0 <= user_id < bundle.n_users:
            return self._cold_start(bundle, budget, user_id, want, lam)

        plan = budget.plan_retrieval(
            cfg.candidate_k if candidate_k is None else int(candidate_k),
            cfg.ef_search if ef_search is None else int(ef_search),
        )
        with budget.stage(
            STAGE_RETRIEVAL, BudgetController.retrieval_units(plan.candidate_k, plan.ef_search)
        ):
            user_vector = bundle.user_vectors(np.asarray([user_id]))
            scores, ids = index.search(user_vector, plan.candidate_k, ef_search=plan.ef_search)
            ann_items = np.asarray(ids[0], dtype=np.int64)
            ann_scores = np.asarray(scores[0], dtype=np.float32)
            # The seen filter belongs here, before the expensive stage: reranking an item the user
            # has already clicked is work that can never appear in the response.
            keep = allowed_positions(ann_items, bundle.snapshot.seen(user_id))
            candidates = ann_items[keep]
            tower_scores = ann_scores[keep]

        retrieved_count = int(candidates.shape[0])
        # faiss returns candidates in descending score, and allowed_positions preserves that order,
        # so a prefix is the best-scoring prefix.
        depth = min(retrieved_count, cfg.rerank_depth)
        ranking = budget.plan_ranking(depth)
        final_scores = tower_scores.astype(np.float64)
        if ranking.rerank:
            candidates = candidates[: ranking.candidates]
            tower_scores = tower_scores[: ranking.candidates]
            with budget.stage(STAGE_RANKING, float(candidates.shape[0])):
                features = assemble_features(bundle.snapshot, user_id, candidates, tower_scores)
                final_scores = bundle.ranker.predict(features)
        else:
            candidates = candidates[:depth]
            final_scores = final_scores[:depth]

        with budget.stage(STAGE_RULES, float(want)):
            items = self._apply_rules(bundle, candidates, final_scores, want, lam)

        result = CascadeResult(
            items=items,
            bundle_id=bundle.bundle_id,
            degraded=budget.degraded,
            degraded_reasons=tuple(budget.degraded_reasons),
            reranked=ranking.rerank,
            cold_start=False,
            candidates=int(retrieved_count),
            reranked_candidates=int(ranking.candidates),
            ef_search=plan.ef_search,
            queued_seconds=queued_seconds,
            contention=contention,
            stage_seconds=budget.timings.as_dict(),
        )
        if result.degraded:
            _log.info(
                "response_degraded",
                reasons=list(result.degraded_reasons),
                ef_search=result.ef_search,
                candidates=result.candidates,
            )
        return result

    def _apply_rules(
        self,
        bundle: Bundle,
        allowed: npt.NDArray[np.int64],
        kept_scores: npt.NDArray[np.float64],
        k: int,
        lam: float,
    ) -> tuple[Recommendation, ...]:
        """Stage 3: diversify with MMR, then cap each category.

        Everything is vectorized: a Python pass over the candidate set would be the most
        expensive GIL-holding work in the request, and that is what serializes a worker pool.
        """
        if allowed.size == 0:
            return ()
        vectors = bundle.item_vectors[allowed]
        # MMR over a bounded window: diversity only ever reorders what could be shown anyway, and
        # the O(w^2) similarity matrix must not grow with candidate_k. The whole window is ordered,
        # not just the first k, so the category cap below has spare candidates to reach k with.
        # It is still a hard constraint: if the window holds fewer than k items the cap allows, the
        # page comes back short rather than breaking the constraint.
        window = min(allowed.shape[0], max(4 * k, 32))
        order = np.argsort(-kept_scores, kind="stable")[:window]
        chosen = mmr_select(kept_scores[order], vectors[order], window, lam)
        ranked = order[chosen]
        categories = bundle.snapshot.item_category[allowed]
        final = cap_per_group(ranked, categories, self.config.max_per_category, k)
        return tuple(
            Recommendation(
                item_id=int(allowed[position]),
                rank=rank,
                score=float(kept_scores[position]),
                category=int(categories[position]),
            )
            for rank, position in enumerate(final, start=1)
        )

    def _cold_start(
        self, bundle: Bundle, budget: Any, user_id: int, k: int, lam: float
    ) -> CascadeResult:
        """Unknown user: serve the snapshot's popularity list, diversified and capped."""
        with budget.stage(STAGE_RULES, float(k)):
            popular = np.asarray(bundle.snapshot.popular_items[: max(4 * k, 32)], dtype=np.int64)
            scores = np.arange(popular.shape[0], 0, -1, dtype=np.float64)
            chosen = mmr_select(scores, bundle.item_vectors[popular], popular.shape[0], lam)
            categories = bundle.snapshot.item_category[popular]
            final = cap_per_group(chosen, categories, self.config.max_per_category, k)
            items = tuple(
                Recommendation(
                    item_id=int(popular[position]),
                    rank=rank,
                    score=float(scores[position]),
                    category=int(categories[position]),
                )
                for rank, position in enumerate(final, start=1)
            )
        _log.info("cold_start_response", known_users=bundle.n_users, requested=k)
        return CascadeResult(
            items=items,
            bundle_id=bundle.bundle_id,
            degraded=False,
            degraded_reasons=(),
            reranked=False,
            cold_start=True,
            candidates=int(popular.shape[0]),
            reranked_candidates=0,
            ef_search=0,
            stage_seconds=budget.timings.as_dict(),
        )

    # --- helpers used by evaluation and monitoring ---

    def rerank_slate(
        self, user_id: int, items: Sequence[int] | npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Ranker scores for a fixed slate (the offline slate-reranking protocol)."""
        bundle = self.bundle
        slate = np.asarray(items, dtype=np.int64)
        if slate.size == 0:
            return np.zeros(0, dtype=np.float64)
        user_vector = bundle.user_vectors(np.asarray([user_id]))
        tower_scores = (bundle.item_vectors[slate] @ user_vector[0]).astype(np.float32)
        features = assemble_features(bundle.snapshot, user_id, slate, tower_scores)
        return bundle.ranker.predict(features)

    def list_similarity(self, item_ids: Sequence[int]) -> float:
        """Mean pairwise cosine similarity of a returned list (lower means more diverse)."""
        if len(item_ids) < 2:
            return 0.0
        return intra_list_similarity(self.bundle.item_vectors[np.asarray(item_ids, dtype=np.int64)])
