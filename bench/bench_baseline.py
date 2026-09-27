"""Micro-benchmarks for the request-path components, with CI regression thresholds.

Usage:
  uv run python bench/bench_baseline.py           # run, write bench/results/baseline.json
  uv run python bench/bench_baseline.py --check   # also exit 1 if bench/thresholds.json is exceeded

These cover the pieces a recommendation request touches per call, plus the redaction and rate
limiting every response goes through. Nothing here trains a model: the quality numbers live in
``bench_quality.py`` and the end-to-end latency SLO in ``bench_load.py``.

Thresholds are deliberately loose (several times the numbers measured on a laptop) because shared
CI runners are noisy; they catch order-of-magnitude regressions, not small drifts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

from retrieval_ranking_cascade.core.ratelimit import RateLimiter
from retrieval_ranking_cascade.core.redaction import redact_text
from retrieval_ranking_cascade.data.generator import (
    PERIOD_HISTORY,
    generate,
    world_config,
)
from retrieval_ranking_cascade.evaluation.propensity import (
    estimate_propensities,
)
from retrieval_ranking_cascade.ml.drift import DriftMonitor
from retrieval_ranking_cascade.ranking.features import (
    FEATURE_NAMES,
    assemble_features,
    build_snapshot,
)
from retrieval_ranking_cascade.ranking.ranker import Ranker, RankerConfig, RankerDataset
from retrieval_ranking_cascade.ranking.rules import filter_seen, mmr_select
from retrieval_ranking_cascade.retrieval.ann import HnswIndex, HnswParams
from retrieval_ranking_cascade.serving.audit import AuditLog, verify_file
from retrieval_ranking_cascade.serving.budget import BudgetController

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "baseline.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"


def _timed(function: Callable[[], object], runs: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    micros = np.asarray(samples, dtype=np.float64) / 1_000.0
    return {
        "runs": runs,
        "p50_us": round(float(np.percentile(micros, 50)), 1),
        "p95_us": round(float(np.percentile(micros, 95)), 1),
    }


def bench_redaction() -> dict[str, float]:
    line = (
        "user jane.doe@example.com paid with 4111 1111 1111 1111 from +49 30 1234 5678; "
        + "lorem ipsum dolor sit amet " * 35
    )
    return _timed(lambda: redact_text(line), runs=2_000)


def bench_rate_limiter() -> dict[str, float]:
    limiter = RateLimiter(rate=1e9, burst=1_000_000_000)
    return _timed(lambda: limiter.check("client"), runs=20_000)


def bench_drift() -> dict[str, float]:
    rng = np.random.default_rng(0)
    monitor = DriftMonitor({"x": rng.normal(size=10_000)})
    current = {"x": rng.normal(loc=0.1, size=10_000)}
    return _timed(lambda: monitor.check(current), runs=30, warmup=2)


def bench_ann_search(dim: int = 32, items: int = 50_000) -> dict[str, float]:
    """One ANN top-500 query at the serving default ``efSearch`` over a 50k-item index."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(items, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = HnswIndex.build(vectors, HnswParams(m=48, ef_construction=300), threads=4)
    query = vectors[:1].copy()
    return _timed(lambda: index.search(query, 500, ef_search=768), runs=200, warmup=5)


def bench_feature_assembly() -> dict[str, float]:
    world, catalog, interactions = generate(world_config("tiny"))
    snapshot = build_snapshot(
        interactions.periods(PERIOD_HISTORY), catalog, n_users=world.config.n_users
    )
    rng = np.random.default_rng(0)
    items = rng.choice(snapshot.n_items, size=500, replace=False).astype(np.int64)
    scores = rng.normal(size=500).astype(np.float32)
    return _timed(lambda: assemble_features(snapshot, 3, items, scores), runs=500)


def bench_mmr(candidates: int = 200, k: int = 60, dim: int = 32) -> dict[str, float]:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(candidates, dim))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = rng.normal(size=candidates)
    return _timed(lambda: mmr_select(scores, vectors, k, 0.7), runs=200)


def bench_filter_seen() -> dict[str, float]:
    rng = np.random.default_rng(0)
    candidates = rng.choice(100_000, size=500, replace=False)
    seen = rng.choice(100_000, size=200, replace=False)
    return _timed(lambda: filter_seen(candidates, seen), runs=2_000)


def bench_budget_plan() -> dict[str, float]:
    controller = BudgetController(total_seconds=0.05, retrieval_seconds=0.02, ranking_seconds=0.02)

    def plan() -> object:
        budget = controller.start()
        budget.plan_retrieval(500, 768)
        return budget.plan_ranking(500)

    return _timed(plan, runs=5_000)


def bench_audit_append(workdir: Path) -> dict[str, float]:
    log = AuditLog(workdir / "append.jsonl")
    counter = iter(range(10**9))
    return _timed(
        lambda: log.append("bench", "step", {"index": next(counter), "note": "x" * 100}), runs=100
    )


def bench_audit_verify(workdir: Path) -> dict[str, float]:
    path = workdir / "verify.jsonl"
    log = AuditLog(path)
    for index in range(500):
        log.append("bench", "step", {"index": index, "note": "x" * 100})
    return _timed(lambda: verify_file(path), runs=10, warmup=1)


def _synthetic_ranker(rows: int = 4_000, queries: int = 200) -> Ranker:
    """A 200-tree lambdarank model over random features: shaped like the real one, quick to fit.

    This bench measures ``predict`` under concurrency, which depends on the tree count and the row
    count and not on what the model learned, so the pipeline's world is not needed here.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(size=(rows, len(FEATURE_NAMES))).astype(np.float32)
    group = np.full(queries, rows // queries, dtype=np.int32)
    labels = (rng.random(rows) < 0.2).astype(np.int32)
    dataset = RankerDataset(
        features=features,
        labels=labels,
        groups=group,
        positions=np.tile(np.arange(1, rows // queries + 1, dtype=np.int32), queries),
        weights=np.ones(rows, dtype=np.float64),
        users=np.repeat(np.arange(queries, dtype=np.int32), rows // queries),
        items=rng.integers(0, 10_000, size=rows).astype(np.int32),
    )
    return Ranker.train(dataset, RankerConfig(ips_weights=False, threads=4))


def _concurrent_predict(
    call: Callable[[npt.NDArray[np.float32]], object],
    features: npt.NDArray[np.float32],
    *,
    threads: int,
    calls: int,
) -> dict[str, float]:
    """Latency and aggregate throughput of ``threads`` callers hammering ``call`` at once.

    Both are needed: a lock leaves the median caller's own latency almost untouched while it caps
    what the pool as a whole can do, so throughput is where serialization shows up first.
    """
    barrier = threading.Barrier(threads)
    samples: list[list[float]] = [[] for _ in range(threads)]
    spans: list[tuple[float, float]] = [(0.0, 0.0)] * threads

    def worker(slot: int) -> None:
        for _ in range(3):  # warm up this thread's own state (a parsed booster, page faults)
            call(features)
        barrier.wait()  # every thread measures while every other thread is also working
        started = time.perf_counter()
        for _ in range(calls):
            began = time.perf_counter()
            call(features)
            samples[slot].append((time.perf_counter() - began) * 1e6)
        spans[slot] = (started, time.perf_counter())

    workers = [threading.Thread(target=worker, args=(slot,)) for slot in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    micros = np.concatenate([np.asarray(slot, dtype=np.float64) for slot in samples])
    elapsed = max(end for _, end in spans) - min(start for start, _ in spans)
    return {
        "runs": int(micros.size),
        "p50_us": round(float(np.percentile(micros, 50)), 1),
        "p95_us": round(float(np.percentile(micros, 95)), 1),
        "calls_per_second": round(float(micros.size) / elapsed, 1) if elapsed > 0 else 0.0,
    }


def bench_ranker_predict_threads(
    rows: int = 100, threads: int = 8, calls: int = 60
) -> dict[str, dict[str, float]]:
    """``Ranker.predict`` under concurrency: a booster per thread against one behind a mutex.

    Serving runs stage 2 on a pool of ``server_threads`` (8 by default) worker threads. ``Ranker``
    gives each thread its own parsed booster built from the shared model text (a
    ``threading.local``); the alternative is one booster guarded by a lock. Three variants are
    measured over the same feature matrix and the same model, so ``ranking/ranker.py`` can quote a
    measurement instead of an intuition: one caller alone (the reference), eight callers with a
    booster each, and eight callers sharing one booster behind a ``threading.Lock``.

    ``rows`` is one request's worth of feature rows at the default ``rerank_depth``. It is a literal
    rather than a read of ``Settings``, so the benchmark's name - and with it its threshold in
    ``bench/thresholds.json`` - stays stable if that default is ever retuned.

    Only the single-caller variant is gated by a threshold. The other two measure contention, and
    what a shared CI runner reports for that depends on how many cores it has and who else is on
    them; they are recorded for reading, not for failing a build.
    """
    ranker = _synthetic_ranker()
    rng = np.random.default_rng(1)
    features = rng.normal(size=(rows, len(FEATURE_NAMES))).astype(np.float32)
    # The control: one booster, parsed once from the same model text, shared behind a lock.
    shared = lgb.Booster(model_str=ranker.to_text())
    lock = threading.Lock()

    def under_mutex(batch: npt.NDArray[np.float32]) -> object:
        with lock:
            return shared.predict(batch, num_threads=1)

    prefix = f"ranker_predict_{rows}_rows"
    return {
        f"{prefix}_1_thread": _concurrent_predict(ranker.predict, features, threads=1, calls=calls),
        f"{prefix}_{threads}_threads_thread_local": _concurrent_predict(
            ranker.predict, features, threads=threads, calls=calls
        ),
        f"{prefix}_{threads}_threads_shared_mutex": _concurrent_predict(
            under_mutex, features, threads=threads, calls=calls
        ),
    }


def bench_propensity() -> dict[str, float]:
    config = world_config("small")
    _, _, interactions = generate(config)
    return _timed(
        lambda: estimate_propensities(interactions, slots=config.slots), runs=20, warmup=2
    )


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def run() -> dict[str, dict[str, float]]:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        return {
            "redact_text_1kb": bench_redaction(),
            "rate_limiter_check": bench_rate_limiter(),
            "drift_check_10k_vs_10k": bench_drift(),
            "ann_search_top500_50k_items": bench_ann_search(),
            "assemble_features_500_rows": bench_feature_assembly(),
            "mmr_select_200_candidates": bench_mmr(),
            "filter_seen_500_candidates": bench_filter_seen(),
            "budget_plan_request": bench_budget_plan(),
            **bench_ranker_predict_threads(),
            "audit_append_fsync": bench_audit_append(workdir),
            "audit_verify_500_entries": bench_audit_verify(workdir),
            "estimate_propensities_200k_rows": bench_propensity(),
        }


def check_thresholds(results: dict[str, dict[str, float]]) -> list[str]:
    limits: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))[
        "baseline"
    ]
    problems = []
    for name, bounds in limits.items():
        if name not in results:
            problems.append(f"{name}: benchmark missing")
            continue
        for metric, limit in bounds.items():
            value = results[name][metric]
            if value > limit:
                problems.append(f"{name}.{metric} = {value} exceeds threshold {limit}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="request-path micro-benchmarks")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"machine": machine(), "results": results}, indent=2) + "\n", encoding="utf-8"
    )
    print("| benchmark | runs | p50 (us) | p95 (us) | calls/s |")
    print("|---|---:|---:|---:|---:|")
    for name, stats in results.items():
        # Throughput is only meaningful where several callers were measured at once.
        throughput = stats.get("calls_per_second")
        print(
            f"| {name} | {stats['runs']:.0f} | {stats['p50_us']} | {stats['p95_us']} | "
            f"{'-' if throughput is None else f'{throughput:.1f}'} |"
        )
    if args.check:
        problems = check_thresholds(results)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all micro-benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
