"""HNSW search, the exact brute-force reference, recall and the index integrity gate."""

from __future__ import annotations

import hashlib
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval_ranking_cascade.retrieval.ann import (
    AnnIntegrityError,
    HnswIndex,
    HnswParams,
    exact_top_k,
    recall_at_k,
    set_faiss_threads,
)

pytestmark = pytest.mark.unit


def _unit_vectors(n: int, dim: int, seed: int = 0) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, dim)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.ascontiguousarray(matrix)


@pytest.fixture(scope="module")
def small_index() -> tuple[HnswIndex, npt.NDArray[np.float32]]:
    vectors = _unit_vectors(400, 16, seed=1)
    return HnswIndex.build(vectors, HnswParams(m=16, ef_construction=120, ef_search=200)), vectors


# --- params ---


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"m": 2}, "HNSW M"),
        ({"m": 200}, "HNSW M"),
        ({"ef_construction": 4}, "efConstruction"),
        ({"ef_search": 0}, "efSearch"),
        ({"ef_search": 9999}, "efSearch"),
    ],
)
def test_hnsw_params_are_bounded(overrides: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HnswParams(**overrides)


# --- exact search ---


def test_exact_top_k_matches_a_full_argsort() -> None:
    vectors = _unit_vectors(120, 8, seed=2)
    queries = _unit_vectors(5, 8, seed=3)
    scores, ids = exact_top_k(queries, vectors, 10)
    reference = queries @ vectors.T
    for row in range(queries.shape[0]):
        expected = np.argsort(-reference[row], kind="stable")[:10]
        assert ids[row].tolist() == expected.tolist()
        np.testing.assert_allclose(scores[row], reference[row][expected], rtol=1e-5)


def test_exact_top_k_breaks_ties_towards_the_lower_id() -> None:
    """Duplicated vectors must produce a stable, id-ordered answer, not an arbitrary one."""
    vectors = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (5, 1))
    _, ids = exact_top_k(np.asarray([[1.0, 0.0]], dtype=np.float32), vectors, 3)
    assert ids[0].tolist() == [0, 1, 2]


def test_exact_top_k_clamps_k_and_chunks_consistently() -> None:
    vectors = _unit_vectors(30, 4, seed=4)
    queries = _unit_vectors(9, 4, seed=5)
    _, ids = exact_top_k(queries, vectors, 99)
    assert ids.shape == (9, 30)
    _, chunked = exact_top_k(queries, vectors, 5, chunk=2)
    _, whole = exact_top_k(queries, vectors, 5, chunk=1024)
    assert chunked.tolist() == whole.tolist()


def test_exact_top_k_accepts_a_single_query_vector() -> None:
    vectors = _unit_vectors(20, 4, seed=6)
    _, ids = exact_top_k(vectors[3], vectors, 1)
    assert ids.shape == (1, 1)
    assert int(ids[0, 0]) == 3


@pytest.mark.parametrize(
    "bad", [np.full((1, 4), np.nan, dtype=np.float32), np.full((1, 4), np.inf, dtype=np.float32)]
)
def test_exact_top_k_refuses_non_finite_queries(bad: npt.NDArray[np.float32]) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        exact_top_k(bad, _unit_vectors(5, 4), 2)


def test_exact_top_k_refuses_a_query_of_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="dimension"):
        exact_top_k(np.zeros((2, 3), dtype=np.float32), _unit_vectors(5, 4), 2)


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=40),
    dim=st.integers(min_value=2, max_value=8),
    k=st.integers(min_value=1, max_value=12),
    seed=st.integers(min_value=0, max_value=500),
)
def test_exact_top_k_scores_are_sorted_and_ids_distinct(
    n: int, dim: int, k: int, seed: int
) -> None:
    vectors = _unit_vectors(n, dim, seed=seed)
    queries = _unit_vectors(3, dim, seed=seed + 1)
    scores, ids = exact_top_k(queries, vectors, k)
    assert scores.shape == ids.shape == (3, min(k, n))
    for row in range(3):
        assert np.all(np.diff(scores[row]) <= 1e-6)
        assert len(set(ids[row].tolist())) == ids.shape[1]


# --- recall ---


def test_recall_at_k_counts_the_intersection_and_ignores_padding() -> None:
    exact = np.asarray([[1, 2, 3, 4]], dtype=np.int64)
    approx = np.asarray([[1, 2, -1, 9]], dtype=np.int64)
    assert recall_at_k(approx, exact)[0] == pytest.approx(0.5)
    # Padding in the truth row shrinks the denominator rather than counting as a miss.
    assert recall_at_k(np.asarray([[1, 2]]), np.asarray([[1, -1]]))[0] == pytest.approx(1.0)


def test_recall_at_k_of_an_empty_truth_row_is_one() -> None:
    assert recall_at_k(np.asarray([[5]]), np.asarray([[-1]]))[0] == pytest.approx(1.0)


def test_recall_at_k_refuses_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        recall_at_k(np.zeros((2, 3), dtype=np.int64), np.zeros((3, 3), dtype=np.int64))
    with pytest.raises(ValueError, match="2-D"):
        recall_at_k(np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64))


# --- HNSW ---


def test_hnsw_search_agrees_with_exact_search_at_a_generous_ef(
    small_index: tuple[HnswIndex, npt.NDArray[np.float32]],
) -> None:
    index, vectors = small_index
    queries = _unit_vectors(20, 16, seed=7)
    _, approx = index.search(queries, 20, ef_search=1024)
    _, exact = exact_top_k(queries, vectors, 20)
    assert float(recall_at_k(approx, exact).mean()) > 0.95


def test_recall_increases_with_ef_search(
    small_index: tuple[HnswIndex, npt.NDArray[np.float32]],
) -> None:
    """The recall/latency knob has to actually be a knob: this is the curve the bench measures."""
    index, vectors = small_index
    queries = _unit_vectors(40, 16, seed=8)
    _, exact = exact_top_k(queries, vectors, 50)
    recalls = []
    for ef in (8, 16, 64, 512):
        _, approx = index.search(queries, 50, ef_search=ef)
        recalls.append(float(recall_at_k(approx, exact).mean()))
    assert recalls == sorted(recalls)
    assert recalls[0] < recalls[-1]


def test_hnsw_reports_its_dimension_and_size(
    small_index: tuple[HnswIndex, npt.NDArray[np.float32]],
) -> None:
    index, vectors = small_index
    assert index.dim == vectors.shape[1]
    assert index.size == vectors.shape[0]
    assert isinstance(index.raw, faiss.IndexHNSWFlat)


def test_hnsw_search_clamps_k_to_the_index_size(
    small_index: tuple[HnswIndex, npt.NDArray[np.float32]],
) -> None:
    index, _ = small_index
    _, ids = index.search(_unit_vectors(2, 16, seed=9), 10_000)
    assert ids.shape == (2, index.size)


def test_hnsw_search_uses_the_params_ef_by_default(
    small_index: tuple[HnswIndex, npt.NDArray[np.float32]],
) -> None:
    index, _ = small_index
    queries = _unit_vectors(5, 16, seed=10)
    default = index.search(queries, 10)[1]
    explicit = index.search(queries, 10, ef_search=index.params.ef_search)[1]
    assert default.tolist() == explicit.tolist()


def test_hnsw_refuses_bad_search_arguments(
    small_index: tuple[HnswIndex, npt.NDArray[np.float32]],
) -> None:
    index, _ = small_index
    with pytest.raises(ValueError, match="k must be"):
        index.search(_unit_vectors(1, 16), 0)
    with pytest.raises(ValueError, match="ef_search"):
        index.search(_unit_vectors(1, 16), 5, ef_search=0)
    with pytest.raises(ValueError, match="dimension"):
        index.search(_unit_vectors(1, 8), 5)


def test_building_from_an_empty_or_one_dimensional_array_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty 2-D"):
        HnswIndex.build(np.zeros((0, 4), dtype=np.float32), HnswParams())
    with pytest.raises(ValueError, match="non-empty 2-D"):
        HnswIndex.build(np.zeros(4, dtype=np.float32), HnswParams())


def test_building_twice_with_one_thread_gives_the_same_bytes() -> None:
    """The bundle's index digest is part of its identity, so the build has to be reproducible."""
    vectors = _unit_vectors(200, 8, seed=11)
    params = HnswParams(m=8, ef_construction=64, ef_search=64)
    first = HnswIndex.build(vectors, params, threads=1).to_bytes()
    second = HnswIndex.build(vectors, params, threads=1).to_bytes()
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_index_round_trips_through_a_hash_verified_load(
    small_index: tuple[HnswIndex, npt.NDArray[np.float32]],
) -> None:
    index, vectors = small_index
    data = index.to_bytes()
    restored = HnswIndex.from_bytes(
        data,
        expected_sha256=hashlib.sha256(data).hexdigest(),
        params=index.params,
        dim=index.dim,
        size=index.size,
    )
    queries = _unit_vectors(6, 16, seed=12)
    assert restored.search(queries, 5)[1].tolist() == index.search(queries, 5)[1].tolist()
    del vectors


def test_a_tampered_index_never_reaches_the_faiss_parser(
    small_index: tuple[HnswIndex, npt.NDArray[np.float32]],
) -> None:
    index, _ = small_index
    data = index.to_bytes()
    digest = hashlib.sha256(data).hexdigest()
    tampered = bytearray(data)
    tampered[len(tampered) // 2] ^= 0xFF
    with pytest.raises(AnnIntegrityError, match="sha256"):
        HnswIndex.from_bytes(
            bytes(tampered),
            expected_sha256=digest,
            params=index.params,
            dim=index.dim,
            size=index.size,
        )


@pytest.mark.parametrize(("dim_delta", "size_delta"), [(1, 0), (0, 1)])
def test_an_index_that_does_not_match_the_bundle_metadata_is_refused(
    small_index: tuple[HnswIndex, npt.NDArray[np.float32]], dim_delta: int, size_delta: int
) -> None:
    index, _ = small_index
    data = index.to_bytes()
    with pytest.raises(AnnIntegrityError, match="dimension or size"):
        HnswIndex.from_bytes(
            data,
            expected_sha256=hashlib.sha256(data).hexdigest(),
            params=index.params,
            dim=index.dim + dim_delta,
            size=index.size + size_delta,
        )


def test_an_index_of_the_wrong_type_or_metric_is_refused() -> None:
    vectors = _unit_vectors(50, 4, seed=13)
    flat = faiss.IndexFlatIP(4)
    flat.add(vectors)
    data = bytes(faiss.serialize_index(flat).tobytes())
    with pytest.raises(AnnIntegrityError, match="not an IndexHNSWFlat"):
        HnswIndex.from_bytes(
            data,
            expected_sha256=hashlib.sha256(data).hexdigest(),
            params=HnswParams(),
            dim=4,
            size=50,
        )
    l2 = faiss.IndexHNSWFlat(4, 8)
    l2.add(vectors)
    l2_data = bytes(faiss.serialize_index(l2).tobytes())
    with pytest.raises(AnnIntegrityError, match="metric"):
        HnswIndex.from_bytes(
            l2_data,
            expected_sha256=hashlib.sha256(l2_data).hexdigest(),
            params=HnswParams(),
            dim=4,
            size=50,
        )


def test_copy_is_independent_of_the_original(
    small_index: tuple[HnswIndex, npt.NDArray[np.float32]],
) -> None:
    """The corruption drill writes through faiss storage; a copy keeps the bundle's index intact."""
    index, _ = small_index
    clone = index.copy()
    # faiss ships no type information for the flat storage view, hence the untyped handle.
    storage: Any = faiss.downcast_index(clone.raw.storage)
    codes = faiss.vector_to_array(storage.codes)
    zeros = np.zeros_like(codes)
    faiss.copy_array_to_vector(zeros, storage.codes)
    queries = _unit_vectors(3, 16, seed=14)
    assert clone.search(queries, 4)[1].tolist() != index.search(queries, 4)[1].tolist()


def test_set_faiss_threads_accepts_zero_and_negative_values() -> None:
    set_faiss_threads(0)
    assert faiss.omp_get_max_threads() == 1
    set_faiss_threads(2)
    assert faiss.omp_get_max_threads() == 2
    set_faiss_threads(1)
