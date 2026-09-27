"""Recall canary: periodically compare ANN results with exact brute force on sampled users.

HNSW recall degrades silently: a bad rebuild, a wrong efSearch, stale or partially overwritten
vectors all still return plausible-looking neighbours. The canary samples real user queries,
runs both the serving index and an exact search over the bundle's item embeddings (the source of
truth), and raises an alert when mean recall@k falls below a threshold. One cycle is enough to
detect a corrupted index, which :func:`corrupt_index` reproduces as a drill.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt

from retrieval_ranking_cascade.retrieval.ann import HnswIndex, exact_top_k, recall_at_k

__all__ = ["CanaryResult", "RecallCanary", "corrupt_index"]

QuerySampler = Callable[[np.random.Generator, int], npt.NDArray[np.float32]]


@dataclass(frozen=True, slots=True)
class CanaryResult:
    cycle: int
    recall_mean: float
    recall_min: float
    queries: int
    k: int
    ef_search: int
    threshold: float
    alert: bool
    duration_seconds: float


class RecallCanary:
    """Runs recall cycles against an index; thread-safe, keeps the latest result."""

    def __init__(
        self,
        *,
        index_getter: Callable[[], HnswIndex],
        vectors_getter: Callable[[], npt.NDArray[np.float32]],
        sampler: QuerySampler,
        k: int = 100,
        sample_size: int = 64,
        threshold: float = 0.9,
        ef_search: int | None = None,
        seed: int = 0,
        on_result: Callable[[CanaryResult], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if k < 1 or sample_size < 1:
            raise ValueError("k and sample_size must be >= 1")
        self._index_getter = index_getter
        self._vectors_getter = vectors_getter
        self._sampler = sampler
        self.k = k
        self.sample_size = sample_size
        self.threshold = threshold
        self.ef_search = ef_search
        self._rng = np.random.default_rng(seed)
        self._on_result = on_result
        self._clock = clock
        self._lock = threading.Lock()
        self._cycle = 0
        self.last: CanaryResult | None = None

    def run_cycle(self) -> CanaryResult:
        with self._lock:
            start = self._clock()
            index = self._index_getter()
            vectors = self._vectors_getter()
            queries = self._sampler(self._rng, self.sample_size)
            k = min(self.k, vectors.shape[0])
            ef = self.ef_search if self.ef_search is not None else index.params.ef_search
            _, approx = index.search(queries, k, ef_search=ef)
            _, exact = exact_top_k(queries, vectors, k)
            recalls = recall_at_k(approx, exact)
            self._cycle += 1
            result = CanaryResult(
                cycle=self._cycle,
                recall_mean=float(recalls.mean()),
                recall_min=float(recalls.min()),
                queries=int(queries.shape[0]),
                k=k,
                ef_search=int(ef),
                threshold=self.threshold,
                alert=bool(recalls.mean() < self.threshold),
                duration_seconds=self._clock() - start,
            )
            self.last = result
        if self._on_result is not None:
            self._on_result(result)
        return result


def corrupt_index(index: HnswIndex, fraction: float, *, seed: int = 0) -> int:
    """Overwrite ``fraction`` of the stored vectors with random unit vectors, in place.

    This simulates partial storage corruption or an id-mapping bug after a rebuild: the graph
    still returns neighbours, but many of them are wrong. Returns the number of rows overwritten.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    # faiss ships no usable type information, so the flat storage view is untyped here.
    storage: Any = faiss.downcast_index(index.raw.storage)
    codes = faiss.vector_to_array(storage.codes)
    vectors = codes.view(np.float32).reshape(index.size, index.dim).copy()
    rng = np.random.default_rng(seed)
    count = max(1, round(fraction * index.size))
    rows = rng.choice(index.size, size=count, replace=False)
    noise = rng.normal(size=(rows.shape[0], index.dim)).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    vectors[rows] = noise
    faiss.copy_array_to_vector(vectors.reshape(-1).view(np.uint8), storage.codes)
    return int(rows.shape[0])
