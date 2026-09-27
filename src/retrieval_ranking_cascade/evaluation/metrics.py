"""Ranking metrics: DCG/NDCG, MRR, recall, catalog coverage and intra-list diversity.

All of them take *already ranked* gain vectors, so the same code evaluates the cascade, the
baselines and the IPS-weighted variants: a weighted metric is just a metric over weighted gains.

NDCG uses the standard exponential-gain form (Jarvelin and Kekalainen 2002; Burges et al. 2005)
``DCG = sum_i (2 ** g_i - 1) / log2(i + 2)`` and divides by the DCG of the same gains sorted
descending, which is the ideal ranking of the *available* items. With continuous gains (IPS
weights, ground-truth relevance) the exponential form would blow up, so ``gain_mode="linear"``
uses ``g_i`` directly; that is the convention used throughout the benches for weighted metrics.
A query whose ideal DCG is 0 (no relevant item) contributes NDCG 0 and is reported separately,
never silently dropped.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "GainMode",
    "catalog_coverage",
    "dcg",
    "discounts",
    "mean_reciprocal_rank",
    "ndcg",
    "ndcg_at_k",
    "recall_at_k_sets",
]

GainMode = Literal["linear", "exponential"]
FArray = npt.NDArray[np.float64]


def discounts(n: int) -> FArray:
    """``1 / log2(i + 2)`` for ranks ``i = 0 .. n - 1``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return 1.0 / np.log2(np.arange(n, dtype=np.float64) + 2.0)


def _gains(values: npt.ArrayLike, mode: GainMode) -> FArray:
    gains = np.asarray(values, dtype=np.float64)
    if gains.size and not np.all(np.isfinite(gains)):
        raise ValueError("gains must be finite")
    if np.any(gains < 0.0):
        raise ValueError("gains must be non-negative")
    if mode == "exponential":
        return np.exp2(gains) - 1.0
    return gains


def dcg(ranked_gains: npt.ArrayLike, k: int | None = None, *, mode: GainMode = "linear") -> float:
    """Discounted cumulative gain of a ranked gain vector, truncated at ``k``."""
    gains = _gains(ranked_gains, mode)
    cut = gains.shape[0] if k is None else min(k, gains.shape[0])
    return float(np.dot(gains[:cut], discounts(cut)))


def ndcg(
    ranked_gains: npt.ArrayLike,
    k: int | None = None,
    *,
    mode: GainMode = "linear",
    ideal_gains: npt.ArrayLike | None = None,
) -> float:
    """NDCG of a ranked gain vector. ``ideal_gains`` defaults to the same gains, sorted."""
    gains = np.asarray(ranked_gains, dtype=np.float64)
    pool = gains if ideal_gains is None else np.asarray(ideal_gains, dtype=np.float64)
    best = dcg(np.sort(pool)[::-1], k, mode=mode)
    if best <= 0.0:
        return 0.0
    return dcg(gains, k, mode=mode) / best


def ndcg_at_k(ranked_gains: npt.ArrayLike, k: int, *, mode: GainMode = "linear") -> float:
    return ndcg(ranked_gains, k, mode=mode)


def mean_reciprocal_rank(ranked_relevant: npt.ArrayLike, k: int | None = None) -> float:
    """1 / rank of the first relevant item (0 if none within ``k``), for one ranked list."""
    flags = np.asarray(ranked_relevant, dtype=bool).ravel()
    cut = flags.shape[0] if k is None else min(k, flags.shape[0])
    hits = np.flatnonzero(flags[:cut])
    return float(1.0 / (hits[0] + 1)) if hits.size else 0.0


def recall_at_k_sets(retrieved: npt.ArrayLike, relevant: npt.ArrayLike) -> float:
    """``|retrieved and relevant| / |relevant|``; an empty relevant set counts as 1.0."""
    got = np.unique(np.asarray(retrieved, dtype=np.int64).ravel())
    want = np.unique(np.asarray(relevant, dtype=np.int64).ravel())
    if want.size == 0:
        return 1.0
    return float(np.intersect1d(got, want, assume_unique=True).size / want.size)


def catalog_coverage(recommendations: npt.ArrayLike, n_items: int) -> float:
    """Share of the catalog that appears in at least one recommendation list."""
    if n_items < 1:
        raise ValueError("n_items must be >= 1")
    items = np.asarray(recommendations, dtype=np.int64).ravel()
    items = items[items >= 0]
    return float(np.unique(items).size / n_items)
