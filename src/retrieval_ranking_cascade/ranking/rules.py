"""Stage 3: business rules over the reranked list - seen filter, MMR diversity, category caps.

Maximal Marginal Relevance (Carbonell and Goldstein, "The Use of MMR, Diversity-Based Reranking
for Reordering Documents and Producing Summaries", SIGIR 1998) greedily picks

    argmax_i  lam * rel(i) - (1 - lam) * max_{j in selected} sim(i, j)

Relevance and similarity have to live on the same scale for ``lam`` to mean anything, so scores
are min-max normalized to [0, 1] over the candidate set and similarity is the cosine of the
(unit-norm) item embeddings, also mapped to [0, 1]. The first pick is always the most relevant
candidate: with an empty selection the diversity term is undefined, and defining it as 0 would
make every candidate tie at ``lam = 0``.

The category cap is a hard business constraint applied last, as an order-preserving filter: it
keeps the first ``max_per_group`` items of each category and drops later ones, so a run of
near-identical items cannot fill the whole page even at ``lam = 1``.

:func:`mmr_objective` states the formula as written above - the diversity term is a ``max`` over the
selected set - and :func:`mmr_select` keeps that ``max`` as a running vector instead of re-slicing
the selected columns on every pick. The two agree exactly (``tests/unit/test_rules.py::
test_each_mmr_pick_maximizes_the_marginal_objective`` checks the greedy step against the formula),
but the incremental form costs O(k n) rather than O(k^2 n) and, more importantly here, allocates
nothing per iteration: the growing fancy-index slice was worth a p95 of 29 ms on 200 candidates
against a p50 of 2 ms, which is not a tail a 50 ms budget can absorb.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "allowed_positions",
    "cap_per_group",
    "filter_seen",
    "intra_list_similarity",
    "mmr_objective",
    "mmr_select",
    "normalize_scores",
]

FArray = npt.NDArray[np.float32]
IArray = npt.NDArray[np.int64]


def allowed_positions(candidates: npt.ArrayLike, seen: npt.ArrayLike) -> IArray:
    """Positions in ``candidates`` that survive the seen filter, de-duplicated, in order.

    Returning positions rather than items lets the caller index the aligned score array without
    building a Python lookup table over the whole candidate set.
    """
    items = np.asarray(candidates, dtype=np.int64).ravel()
    blocked = np.asarray(seen, dtype=np.int64).ravel()
    keep = items >= 0
    if blocked.size:
        keep &= ~np.isin(items, blocked)
    positions = np.flatnonzero(keep)
    if positions.size == 0:
        return positions.astype(np.int64)
    _, first = np.unique(items[positions], return_index=True)
    return np.asarray(positions[np.sort(first)], dtype=np.int64)


def filter_seen(candidates: npt.ArrayLike, seen: npt.ArrayLike) -> IArray:
    """Candidates that survive the seen filter (see :func:`allowed_positions`)."""
    items = np.asarray(candidates, dtype=np.int64).ravel()
    return np.asarray(items[allowed_positions(items, seen)], dtype=np.int64)


def normalize_scores(scores: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Min-max normalize to [0, 1]; a constant input maps to all ones (no relevance signal)."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        return values
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    low, high = float(values.min()), float(values.max())
    if high - low <= 0.0:
        return np.ones_like(values)
    return (values - low) / (high - low)


def mmr_objective(
    relevance: npt.NDArray[np.float64],
    similarity: npt.NDArray[np.float64],
    selected: list[int],
    lam: float,
) -> npt.NDArray[np.float64]:
    """Marginal MMR value of every candidate given the items already ``selected``."""
    if not selected:
        return relevance.copy()
    worst = similarity[:, selected].max(axis=1)
    return lam * relevance - (1.0 - lam) * worst


def mmr_select(
    scores: npt.ArrayLike,
    vectors: npt.ArrayLike,
    k: int,
    lam: float,
) -> IArray:
    """Indices of the ``k`` candidates chosen by MMR, in selection order.

    ``vectors`` are the candidates' unit-norm embeddings. Ties break towards the lower index, so
    the selection is deterministic.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if k < 0:
        raise ValueError("k must be non-negative")
    relevance = normalize_scores(scores)
    n = relevance.shape[0]
    take = min(k, n)
    if take == 0:
        return np.empty(0, dtype=np.int64)
    if lam >= 1.0:
        return np.asarray(np.argsort(-relevance, kind="stable")[:take], dtype=np.int64)
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != n:
        raise ValueError("vectors must be a 2-D array with one row per score")
    # Cosine of unit-norm rows, mapped from [-1, 1] to [0, 1] to match the relevance scale.
    similarity = np.clip(0.5 * (matrix @ matrix.T + 1.0), 0.0, 1.0)
    first = int(np.argmax(relevance))
    selected: list[int] = [first]
    available = np.ones(n, dtype=bool)
    available[first] = False
    # ``worst`` is the diversity term of :func:`mmr_objective`, carried forward instead of
    # recomputed: max over a set grows by one element at a time, so np.maximum is enough.
    worst = similarity[:, first].copy()
    while len(selected) < take:
        marginal = lam * relevance - (1.0 - lam) * worst
        marginal[~available] = -np.inf
        choice = int(np.argmax(marginal))
        selected.append(choice)
        available[choice] = False
        np.maximum(worst, similarity[:, choice], out=worst)
    return np.asarray(selected, dtype=np.int64)


def cap_per_group(
    order: npt.ArrayLike, groups: npt.ArrayLike, max_per_group: int, k: int | None = None
) -> IArray:
    """Keep at most ``max_per_group`` items per group in rank order, then the top ``k``.

    ``order`` holds candidate indices in ranked order; ``groups[i]`` is the group of candidate
    index ``i``.
    """
    if max_per_group < 1:
        raise ValueError("max_per_group must be >= 1")
    ranked = np.asarray(order, dtype=np.int64).ravel()
    labels = np.asarray(groups, dtype=np.int64).ravel()
    counts: dict[int, int] = {}
    kept: list[int] = []
    limit = len(ranked) if k is None else k
    for index in ranked:
        group = int(labels[index])
        if counts.get(group, 0) >= max_per_group:
            continue
        counts[group] = counts.get(group, 0) + 1
        kept.append(int(index))
        if len(kept) >= limit:
            break
    return np.asarray(kept, dtype=np.int64)


def intra_list_similarity(vectors: npt.ArrayLike) -> float:
    """Mean pairwise cosine similarity of a recommendation list (0.0 for fewer than 2 items)."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0
    gram = matrix @ matrix.T
    n = matrix.shape[0]
    total = float(gram.sum() - np.trace(gram))
    return float(total / (n * (n - 1)))
