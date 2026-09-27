"""The recall canary and the index-corruption drill it is supposed to catch."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from retrieval_ranking_cascade.retrieval.ann import HnswIndex, HnswParams
from retrieval_ranking_cascade.retrieval.canary import CanaryResult, RecallCanary, corrupt_index

pytestmark = pytest.mark.unit


def _unit_vectors(n: int, dim: int, seed: int = 0) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, dim)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.ascontiguousarray(matrix)


@pytest.fixture(scope="module")
def vectors() -> npt.NDArray[np.float32]:
    return _unit_vectors(600, 16, seed=1)


@pytest.fixture
def index(vectors: npt.NDArray[np.float32]) -> HnswIndex:
    return HnswIndex.build(vectors, HnswParams(m=16, ef_construction=200, ef_search=256))


def _canary(
    index: HnswIndex,
    vectors: npt.NDArray[np.float32],
    *,
    threshold: float = 0.9,
    k: int = 20,
    sample_size: int = 24,
    on_result: Any = None,
    clock: Any = None,
) -> RecallCanary:
    holder = {"index": index}

    def sampler(rng: np.random.Generator, size: int) -> npt.NDArray[np.float32]:
        rows = rng.integers(0, vectors.shape[0], size=size)
        return np.ascontiguousarray(vectors[rows])

    canary = RecallCanary(
        index_getter=lambda: holder["index"],
        vectors_getter=lambda: vectors,
        sampler=sampler,
        k=k,
        sample_size=sample_size,
        threshold=threshold,
        on_result=on_result,
        seed=3,
        **({"clock": clock} if clock is not None else {}),
    )
    canary.swap = holder  # type: ignore[attr-defined]
    return canary


# --- configuration ---


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"threshold": 0.0}, "threshold"),
        ({"threshold": 1.5}, "threshold"),
        ({"k": 0}, "k and sample_size"),
        ({"sample_size": 0}, "k and sample_size"),
    ],
)
def test_the_canary_validates_its_configuration(
    index: HnswIndex, vectors: npt.NDArray[np.float32], overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        RecallCanary(
            index_getter=lambda: index,
            vectors_getter=lambda: vectors,
            sampler=lambda rng, size: vectors[:size],
            **overrides,
        )


# --- healthy cycles ---


def test_a_healthy_index_passes_and_the_result_records_the_cycle(
    index: HnswIndex, vectors: npt.NDArray[np.float32]
) -> None:
    canary = _canary(index, vectors)
    result = canary.run_cycle()
    assert isinstance(result, CanaryResult)
    assert result.cycle == 1
    assert result.queries == 24
    assert result.k == 20
    assert result.ef_search == index.params.ef_search
    assert result.threshold == 0.9
    assert result.recall_mean > 0.95
    assert result.recall_min <= result.recall_mean
    assert not result.alert
    assert result.duration_seconds >= 0.0
    assert canary.last == result


def test_cycles_increment_and_the_latest_result_is_kept(
    index: HnswIndex, vectors: npt.NDArray[np.float32]
) -> None:
    canary = _canary(index, vectors)
    for expected in (1, 2, 3):
        assert canary.run_cycle().cycle == expected
    latest = canary.last
    assert latest is not None
    assert latest.cycle == 3


def test_no_result_is_reported_before_the_first_cycle(
    index: HnswIndex, vectors: npt.NDArray[np.float32]
) -> None:
    assert _canary(index, vectors).last is None


def test_k_is_clamped_to_the_corpus_size(vectors: npt.NDArray[np.float32]) -> None:
    small = _unit_vectors(30, 16, seed=2)
    index = HnswIndex.build(small, HnswParams(m=8, ef_construction=64, ef_search=64))
    result = _canary(index, small, k=1_000).run_cycle()
    assert result.k == 30
    del vectors


def test_an_explicit_ef_search_overrides_the_index_default(
    index: HnswIndex, vectors: npt.NDArray[np.float32]
) -> None:
    canary = _canary(index, vectors)
    canary.ef_search = 4
    result = canary.run_cycle()
    assert result.ef_search == 4
    # Almost no graph exploration, so recall collapses and the canary says so.
    assert result.recall_mean < 0.9
    assert result.alert


def test_the_on_result_callback_sees_every_cycle(
    index: HnswIndex, vectors: npt.NDArray[np.float32]
) -> None:
    seen: list[CanaryResult] = []
    canary = _canary(index, vectors, on_result=seen.append)
    canary.run_cycle()
    canary.run_cycle()
    assert [result.cycle for result in seen] == [1, 2]


def test_a_custom_clock_measures_the_cycle_duration(
    index: HnswIndex, vectors: npt.NDArray[np.float32]
) -> None:
    ticks = iter([100.0, 100.25])
    canary = _canary(index, vectors, clock=lambda: next(ticks))
    assert canary.run_cycle().duration_seconds == pytest.approx(0.25)


# --- corruption drill ---


def test_a_corrupted_index_alerts_within_one_cycle(
    index: HnswIndex, vectors: npt.NDArray[np.float32]
) -> None:
    """The headline claim: one cycle is enough, because exact search is the source of truth."""
    canary = _canary(index, vectors)
    assert not canary.run_cycle().alert
    damaged = index.copy()
    overwritten = corrupt_index(damaged, 0.5, seed=4)
    canary.swap["index"] = damaged  # type: ignore[attr-defined]
    result = canary.run_cycle()
    assert overwritten == pytest.approx(0.5 * index.size, abs=1)
    assert result.alert
    assert result.recall_mean < 0.9
    assert result.cycle == 2


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.6, 1.0])
def test_recall_falls_monotonically_with_the_corrupted_fraction(
    index: HnswIndex, vectors: npt.NDArray[np.float32], fraction: float
) -> None:
    damaged = index.copy()
    corrupt_index(damaged, fraction, seed=5)
    canary = _canary(damaged, vectors, sample_size=32)
    clean = _canary(index, vectors, sample_size=32).run_cycle().recall_mean
    assert canary.run_cycle().recall_mean < clean


def test_corrupting_a_copy_leaves_the_original_index_untouched(
    index: HnswIndex, vectors: npt.NDArray[np.float32]
) -> None:
    before = _canary(index, vectors).run_cycle().recall_mean
    corrupt_index(index.copy(), 1.0, seed=6)
    after = _canary(index, vectors).run_cycle().recall_mean
    assert after == pytest.approx(before)


def test_corruption_overwrites_the_requested_number_of_rows(index: HnswIndex) -> None:
    damaged = index.copy()
    assert corrupt_index(damaged, 0.25, seed=7) == round(0.25 * index.size)
    # Even a tiny fraction touches at least one row, so the drill is never a no-op.
    tiny = index.copy()
    assert corrupt_index(tiny, 1e-9, seed=8) == 1


def test_corrupted_rows_are_still_unit_vectors(index: HnswIndex) -> None:
    """The graph keeps returning plausible neighbours; that is exactly why this is silent."""
    import faiss

    damaged = index.copy()
    corrupt_index(damaged, 1.0, seed=9)
    storage: Any = faiss.downcast_index(damaged.raw.storage)
    codes = faiss.vector_to_array(storage.codes)
    rows = codes.view(np.float32).reshape(damaged.size, damaged.dim)
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_an_out_of_range_corruption_fraction_is_refused(index: HnswIndex, fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        corrupt_index(index, fraction)


# --- concurrency ---


def test_concurrent_cycles_produce_consecutive_cycle_numbers(
    index: HnswIndex, vectors: npt.NDArray[np.float32]
) -> None:
    """Cycles are serialized by the canary's own lock, so the counter cannot race."""
    canary = _canary(index, vectors, sample_size=8, k=5)
    results: list[CanaryResult] = []
    lock = threading.Lock()

    def work() -> None:
        result = canary.run_cycle()
        with lock:
            results.append(result)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(result.cycle for result in results) == [1, 2, 3, 4, 5, 6]
