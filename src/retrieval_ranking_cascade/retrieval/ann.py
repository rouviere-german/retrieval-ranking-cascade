"""HNSW approximate nearest-neighbour search (faiss) and the exact brute-force reference.

The index stores unit-norm item embeddings under inner-product metric, so scores are cosine
similarities. ``efSearch`` is passed per query through ``faiss.SearchParametersHNSW`` rather than
by mutating the index, which keeps concurrent searches with different budgets thread-safe.

Serialization uses ``faiss.serialize_index`` (the in-memory form of ``write_index``). Bytes are
only handed to ``faiss.deserialize_index`` after their SHA-256 matches the expected digest: the
faiss reader is a C++ parser, and a tampered file must never reach it.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt

__all__ = [
    "AnnIntegrityError",
    "HnswIndex",
    "HnswParams",
    "exact_top_k",
    "recall_at_k",
    "set_faiss_threads",
]

FArray = npt.NDArray[np.float32]
IArray = npt.NDArray[np.int64]


class AnnIntegrityError(ValueError):
    """Index bytes do not match their expected hash, or the index does not match its metadata."""


def set_faiss_threads(threads: int) -> None:
    """Set faiss' OpenMP thread count (1 for per-request serving, >1 for batch jobs)."""
    faiss.omp_set_num_threads(max(1, int(threads)))


@dataclass(frozen=True, slots=True)
class HnswParams:
    m: int = 32
    ef_construction: int = 200
    ef_search: int = 768

    def __post_init__(self) -> None:
        if not 4 <= self.m <= 128:
            raise ValueError("HNSW M must be between 4 and 128")
        if not 16 <= self.ef_construction <= 2048:
            raise ValueError("efConstruction must be between 16 and 2048")
        if not 1 <= self.ef_search <= 4096:
            raise ValueError("efSearch must be between 1 and 4096")


def _as_queries(vectors: npt.ArrayLike, dim: int) -> FArray:
    array = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"expected vectors of dimension {dim}")
    if not np.all(np.isfinite(array)):
        raise ValueError("vectors contain NaN or infinite values")
    return array


class HnswIndex:
    """A faiss ``IndexHNSWFlat`` (inner product) with its build parameters."""

    def __init__(self, index: Any, params: HnswParams) -> None:
        self._index = index
        self.params = params

    @property
    def dim(self) -> int:
        return int(self._index.d)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    @property
    def raw(self) -> Any:
        """The underlying faiss index (used by the corruption drill in tests and benches)."""
        return self._index

    @classmethod
    def build(cls, vectors: npt.ArrayLike, params: HnswParams, *, threads: int = 1) -> HnswIndex:
        """Build from unit-norm vectors. ``threads=1`` gives a deterministic graph."""
        array = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("vectors must be a non-empty 2-D array")
        index = faiss.IndexHNSWFlat(int(array.shape[1]), params.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = params.ef_construction
        index.hnsw.efSearch = params.ef_search
        set_faiss_threads(threads)
        try:
            index.add(_as_queries(array, int(array.shape[1])))
        finally:
            set_faiss_threads(1)
        return cls(index, params)

    def search(
        self, queries: npt.ArrayLike, k: int, *, ef_search: int | None = None
    ) -> tuple[FArray, IArray]:
        """Top-``k`` (scores, ids) per query; ids of -1 (not found) are possible."""
        if k < 1:
            raise ValueError("k must be >= 1")
        array = _as_queries(queries, self.dim)
        ef = self.params.ef_search if ef_search is None else int(ef_search)
        if ef < 1:
            raise ValueError("ef_search must be >= 1")
        # faiss' bundled type information omits the per-search parameter classes.
        params = faiss.SearchParametersHNSW()  # type: ignore[attr-defined]
        params.efSearch = ef
        scores, ids = self._index.search(array, min(k, self.size), params=params)
        return np.asarray(scores, dtype=np.float32), np.asarray(ids, dtype=np.int64)

    def to_bytes(self) -> bytes:
        return bytes(faiss.serialize_index(self._index).tobytes())

    def copy(self) -> HnswIndex:
        """An independent deep copy, for drills that deliberately damage an index.

        ``corrupt_index`` writes through faiss' storage, which would otherwise damage the bundle's
        own index and silently contaminate every measurement taken afterwards.
        """
        data = self.to_bytes()
        index = faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8).copy())
        return HnswIndex(index, self.params)

    @classmethod
    def from_bytes(
        cls, data: bytes, *, expected_sha256: str, params: HnswParams, dim: int, size: int
    ) -> HnswIndex:
        """Verify the hash, then deserialize and check dimension, size and index type."""
        actual = hashlib.sha256(data).hexdigest()
        if not hmac.compare_digest(actual, expected_sha256):
            raise AnnIntegrityError("index bytes do not match the expected sha256")
        index = faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8).copy())
        if not isinstance(index, faiss.IndexHNSWFlat):
            raise AnnIntegrityError("index is not an IndexHNSWFlat")
        if int(index.d) != dim or int(index.ntotal) != size:
            raise AnnIntegrityError("index dimension or size does not match the bundle")
        if int(index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
            raise AnnIntegrityError("index metric is not inner product")
        return cls(index, params)


def exact_top_k(
    queries: npt.ArrayLike, vectors: npt.ArrayLike, k: int, *, chunk: int = 256
) -> tuple[FArray, IArray]:
    """Brute-force top-``k`` by inner product, sorted by descending score (ties by id)."""
    base = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    q = _as_queries(queries, int(base.shape[1]))
    k = min(k, base.shape[0])
    all_scores = np.empty((q.shape[0], k), dtype=np.float32)
    all_ids = np.empty((q.shape[0], k), dtype=np.int64)
    for start in range(0, q.shape[0], chunk):
        block = q[start : start + chunk] @ base.T
        part = np.argpartition(-block, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(block, part, axis=1)
        order = np.lexsort((part, -part_scores), axis=1)
        all_ids[start : start + chunk] = np.take_along_axis(part, order, axis=1)
        all_scores[start : start + chunk] = np.take_along_axis(part_scores, order, axis=1)
    return all_scores, all_ids


def recall_at_k(approx_ids: npt.ArrayLike, exact_ids: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Per-query ``|approx ∩ exact| / |exact|``; -1 placeholders never count as hits."""
    approx = np.asarray(approx_ids, dtype=np.int64)
    exact = np.asarray(exact_ids, dtype=np.int64)
    if approx.ndim != 2 or exact.ndim != 2 or approx.shape[0] != exact.shape[0]:
        raise ValueError("approx and exact ids must be 2-D with the same number of rows")
    recalls = np.empty(approx.shape[0], dtype=np.float64)
    for row in range(approx.shape[0]):
        truth = exact[row][exact[row] >= 0]
        found = approx[row][approx[row] >= 0]
        recalls[row] = (
            np.intersect1d(truth, found, assume_unique=False).shape[0] / truth.shape[0]
            if truth.shape[0]
            else 1.0
        )
    return recalls
