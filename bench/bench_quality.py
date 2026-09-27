"""Quality benchmarks: recall, ranking, bias correction, robustness and the canary drill.

Usage:
  uv run python bench/bench_quality.py                  # train once, measure everything
  uv run python bench/bench_quality.py --check          # also enforce bench/thresholds.json
  uv run python bench/bench_quality.py --scale smoke    # the 1% slice CI trains on

What it reports, and why each number is here:

* **retrieval** - recall@500 against exact search across ``efSearch``. The cascade cannot recommend
  what it does not retrieve, so this is the ceiling on everything below it.
* **ranking** - NDCG@10 and MRR@10 on held-out slates for the logged policy, a popularity baseline,
  the tower alone and the full cascade, each scored naively, with IPS weights and against the
  world's true relevance.
* **bias** - the gap between the click-based estimates and ground truth, naive versus IPS, plus the
  examination exponent recovered from the randomized slice against the one the world used.
* **corpus** - full-catalogue top-10 quality for popularity, tower, cascade and an exact
  full-corpus ranking upper bound, with coverage, intra-list similarity and cold-start slices.
* **ablations** - rerank depth (the cost/quality knob), the ranker's label treatment,
  logQ correction on/off, ``efSearch`` and MMR lambda.
* **robustness** - rank shift of a shilled item under robust versus naive item statistics.
* **canary** - recall before and after half the index vectors are corrupted.

One training run at ``small`` scale (5k users, 10k items, ~200k impressions) feeds all of it and
takes well under a minute on six CPU cores.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from retrieval_ranking_cascade.data.generator import PERIOD_LABEL, PERIOD_TEST
from retrieval_ranking_cascade.evaluation.offline import (
    corpus_evaluation,
    make_user_sample,
    position_bias_report,
    slate_evaluation,
)
from retrieval_ranking_cascade.evaluation.propensity import (
    estimate_eta,
    estimate_propensities,
)
from retrieval_ranking_cascade.evaluation.shilling import (
    AttackConfig,
    run_shilling_attack,
)
from retrieval_ranking_cascade.evaluation.simulator import FeedbackSimulator
from retrieval_ranking_cascade.pipeline import (
    TrainingConfig,
    TrainingResult,
    train_bundle,
)
from retrieval_ranking_cascade.ranking.features import (
    item_side_features,
    user_side_features,
)
from retrieval_ranking_cascade.ranking.ranker import (
    Ranker,
    RankerConfig,
    build_ranker_dataset,
)
from retrieval_ranking_cascade.retrieval.ann import (
    HnswIndex,
    exact_top_k,
    recall_at_k,
)
from retrieval_ranking_cascade.retrieval.canary import corrupt_index
from retrieval_ranking_cascade.retrieval.two_tower import TowerConfig
from retrieval_ranking_cascade.serving.budget import BudgetController
from retrieval_ranking_cascade.serving.cascade import Cascade, CascadeConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "quality.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"
EF_SWEEP = (64, 128, 256, 512, 768, 1024)
MMR_SWEEP = (1.0, 0.7, 0.4)
# Every corpus evaluation draws the same number of users with the same seed, so a policy's number
# means the same thing in the main table and in the ablations.
CORPUS_USERS = 150
DEPTH_SWEEP = (50, 100, 150, 250, 500)
# The ablation's baseline equals the served default, so its rows compare with the main tables.
DEFAULT_RERANK_DEPTH = 150
BLOCKS = ("retrieval", "ranking", "corpus", "robustness", "canary")


def _cascade(
    result: TrainingResult, *, ef_search: int | None = None, rerank_depth: int | None = None
) -> Cascade:
    """A cascade with a generous budget, so quality numbers are never degraded by the clock."""
    controller = BudgetController(total_seconds=60.0, retrieval_seconds=30.0, ranking_seconds=30.0)
    config = CascadeConfig(
        candidate_k=result.config.candidate_k,
        rerank_depth=min(rerank_depth or DEFAULT_RERANK_DEPTH, result.config.candidate_k),
        default_k=20,
        ef_search=ef_search or result.config.hnsw.ef_search,
    )
    return Cascade(result.bundle, controller, config)


def bench_rerank_depth(result: TrainingResult) -> dict[str, dict[str, Any]]:
    """NDCG and per-request cost against the rerank depth: the cascade's central trade-off.

    Latency here is the in-process cascade cost, not an HTTP number - ``bench/bench_load.py`` owns
    the served percentiles. The point is the shape: cost is linear in depth while quality saturates.
    """
    users = make_user_sample(
        result.interactions.periods(PERIOD_TEST),
        result.bundle.n_users,
        CORPUS_USERS,
        seed=result.config.seed,
    )
    rows: dict[str, dict[str, Any]] = {}
    for depth in DEPTH_SWEEP:
        if depth > result.config.candidate_k:
            continue
        cascade = _cascade(result, rerank_depth=depth)
        report = corpus_evaluation(
            cascade=cascade, world=result.world, users=users, k=10, upper_bound_users=0
        )
        for user in users[:20]:
            cascade.recommend(int(user), k=20)
        started = time.perf_counter()
        for user in users:
            cascade.recommend(int(user), k=20)
        per_request_ms = (time.perf_counter() - started) / len(users) * 1000.0
        rows[str(depth)] = {
            "corpus_ndcg_at_10": report["cascade_ndcg_at_k"],
            "served_ndcg_at_10": report["cascade_served_ndcg_at_k"],
            "coverage": report["cascade"]["catalog_coverage"],  # type: ignore[index]
            "ms_per_request": round(per_request_ms, 3),
        }
    return rows


def bench_retrieval(result: TrainingResult, sample: int = 200) -> dict[str, Any]:
    bundle = result.bundle
    rng = np.random.default_rng(result.config.seed)
    users = rng.choice(bundle.n_users, size=min(sample, bundle.n_users), replace=False)
    queries = bundle.user_vectors(users)
    k = min(result.config.candidate_k, bundle.n_items)
    started = time.perf_counter()
    _, exact = exact_top_k(queries, bundle.item_vectors, k)
    exact_seconds = (time.perf_counter() - started) / queries.shape[0]
    curve: dict[str, dict[str, float]] = {}
    for ef in EF_SWEEP:
        started = time.perf_counter()
        _, approx = bundle.index.search(queries, k, ef_search=ef)
        elapsed = (time.perf_counter() - started) / queries.shape[0]
        recalls = recall_at_k(approx, exact)
        curve[str(ef)] = {
            "recall_mean": round(float(recalls.mean()), 6),
            "recall_p05": round(float(np.percentile(recalls, 5)), 6),
            "search_ms": round(1000.0 * elapsed, 4),
        }
    default = str(result.config.hnsw.ef_search)
    return {
        "k": k,
        "queries": int(queries.shape[0]),
        "default_ef_search": result.config.hnsw.ef_search,
        "recall_at_default": curve[default]["recall_mean"] if default in curve else None,
        "exact_search_ms": round(1000.0 * exact_seconds, 4),
        "speedup_vs_exact": (
            round(exact_seconds / (curve[default]["search_ms"] / 1000.0), 2)
            if default in curve and curve[default]["search_ms"] > 0
            else None
        ),
        "curve": curve,
        "scaling": bench_index_scaling(result, queries),
    }


def bench_index_scaling(
    result: TrainingResult, queries: np.ndarray, sizes: tuple[int, ...] = (2_500, 10_000, 50_000)
) -> dict[str, dict[str, float]]:
    """ANN versus exact latency as the catalogue grows, over prefixes of the real item embeddings.

    The crossover is a fact about sizes, not an article of faith. A brute-force inner product is one
    dense matrix multiply, which BLAS runs with near-perfect locality, so at a few thousand
    32-dimensional vectors it simply beats a graph walk; its cost then grows linearly with the
    catalogue while HNSW's stays nearly flat. Fitting those two lines gives
    ``crossover_items_estimate``: the catalogue size at which the graph starts to pay, on this
    machine, at this dimension and this ``k``.
    """
    bundle = result.bundle
    rows: dict[str, dict[str, float]] = {}
    for size in sizes:
        if size > bundle.n_items:
            continue
        vectors = np.ascontiguousarray(bundle.item_vectors[:size])
        index = HnswIndex.build(vectors, result.config.hnsw, threads=result.config.threads)
        k = min(result.config.candidate_k, size)
        started = time.perf_counter()
        _, approx = index.search(queries, k, ef_search=result.config.hnsw.ef_search)
        ann_seconds = (time.perf_counter() - started) / queries.shape[0]
        started = time.perf_counter()
        _, exact = exact_top_k(queries, vectors, k)
        exact_seconds = (time.perf_counter() - started) / queries.shape[0]
        rows[str(size)] = {
            "ann_ms": round(1000.0 * ann_seconds, 4),
            "exact_ms": round(1000.0 * exact_seconds, 4),
            "speedup": round(exact_seconds / ann_seconds, 2) if ann_seconds > 0 else 0.0,
            "recall": round(float(recall_at_k(approx, exact).mean()), 6),
        }
    rows.update(_crossover_estimate(rows))
    return rows


def _crossover_estimate(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Least-squares fit of ``exact = a * n`` and ``ann = c + d * n``, and where they meet."""
    sizes = sorted(int(key) for key in rows if key.isdigit())
    if len(sizes) < 2:
        return {}
    n = np.asarray(sizes, dtype=np.float64)
    exact = np.asarray([rows[str(size)]["exact_ms"] for size in sizes])
    ann = np.asarray([rows[str(size)]["ann_ms"] for size in sizes])
    design = np.stack([np.ones_like(n), n], axis=1)
    exact_c, exact_d = np.linalg.lstsq(design, exact, rcond=None)[0]
    ann_c, ann_d = np.linalg.lstsq(design, ann, rcond=None)[0]
    slope_gap = exact_d - ann_d
    estimate = float((ann_c - exact_c) / slope_gap) if slope_gap > 0 else float("inf")
    return {
        "fit": {
            "exact_ms_per_1k_items": round(1000.0 * float(exact_d), 6),
            "ann_ms_per_1k_items": round(1000.0 * float(ann_d), 6),
            "ann_fixed_ms": round(float(ann_c), 4),
            "crossover_items_estimate": (
                round(estimate) if np.isfinite(estimate) and estimate > 0 else -1.0
            ),
        }
    }


def bench_ranking_and_bias(result: TrainingResult) -> dict[str, Any]:
    cascade = _cascade(result)
    sessions = result.interactions.periods(PERIOD_TEST)
    propensity = estimate_propensities(
        result.interactions, slots=result.world_config.slots, min_impressions=50
    )
    report = slate_evaluation(
        sessions=sessions,
        cascade=cascade,
        world=result.world,
        propensity=propensity,
        k=10,
        max_sessions=1_500,
        seed=result.config.seed,
    )
    positions = position_bias_report(
        result.interactions, result.world, propensity, slots=result.world_config.slots
    )
    bias: dict[str, Any] = {
        "estimated_eta": round(estimate_eta(propensity), 4),
        "true_eta": result.world_config.examination_eta,
        "naive_advantage_bias": round(report.naive_bias, 6),
        "ips_advantage_bias": round(report.ips_bias, 6),
        "advantage_bias_reduction": (
            round(1.0 - report.ips_bias / report.naive_bias, 4) if report.naive_bias > 0 else None
        ),
        "naive_position_bias": round(positions.naive_bias, 6),
        "ips_position_bias": round(positions.ips_bias, 6),
        "position_bias_reduction": (
            round(1.0 - positions.ips_bias / positions.naive_bias, 4)
            if positions.naive_bias > 0
            else None
        ),
        "estimated_propensities": [round(value, 4) for value in propensity.propensities],
        "true_propensities": [
            round(float(value), 4)
            for value in result.world.examination(np.arange(1, result.world_config.slots + 1))
        ],
    }
    return {"slate": report.as_dict(), "bias": bias, "position_bias": positions.as_dict()}


def bench_corpus(
    result: TrainingResult, users: int = CORPUS_USERS, upper_bound_users: int = 40
) -> dict[str, Any]:
    cascade = _cascade(result)
    sample = make_user_sample(
        result.interactions.periods(PERIOD_TEST),
        result.bundle.n_users,
        users,
        seed=result.config.seed,
    )
    report = corpus_evaluation(
        cascade=cascade,
        world=result.world,
        users=sample,
        k=10,
        upper_bound_users=upper_bound_users,
    )
    simulator = FeedbackSimulator(result.world, seed=result.config.seed)
    served = [(int(user), cascade.recommend(int(user), k=10).item_ids) for user in sample[:100]]
    report["simulated"] = simulator.simulate(served).as_dict()
    return report


def bench_ablations(result: TrainingResult) -> dict[str, Any]:
    """logQ correction on/off (a retrain), plus efSearch and MMR lambda (no retrain)."""
    cascade = _cascade(result)
    sample = make_user_sample(
        result.interactions.periods(PERIOD_TEST),
        result.bundle.n_users,
        CORPUS_USERS,
        seed=result.config.seed,
    )
    ef_rows: dict[str, dict[str, Any]] = {}
    for ef in (128, 256, result.config.hnsw.ef_search):
        report = corpus_evaluation(
            cascade=cascade,
            world=result.world,
            users=sample,
            k=10,
            upper_bound_users=0,
            ef_search=ef,
        )
        ef_rows[str(ef)] = {
            "cascade_ndcg_at_k": report["cascade_ndcg_at_k"],
            "coverage": report["cascade"]["catalog_coverage"],  # type: ignore[index]
        }
    mmr_rows = {}
    for lam in MMR_SWEEP:
        lists = [cascade.recommend(int(user), k=10, mmr_lambda=lam) for user in sample]
        similarity = float(np.mean([cascade.list_similarity(item.item_ids) for item in lists]))
        gains = []
        for user, served in zip(sample, lists, strict=True):
            relevance = result.world.relevance_rows(np.asarray([int(user)]))[0]
            gains.append(float(np.mean(relevance[np.asarray(served.item_ids, dtype=np.int64)])))
        mmr_rows[str(lam)] = {
            "intra_list_similarity": round(similarity, 6),
            "mean_true_relevance": round(float(np.mean(gains)), 6),
            "distinct_categories": round(
                float(np.mean([len({item.category for item in served.items}) for served in lists])),
                3,
            ),
        }

    depth_rows = bench_rerank_depth(result)
    ranker_rows = _ranker_label_ablation(result)
    no_logq = train_bundle(
        TrainingConfig(
            scale=result.config.scale,
            seed=result.config.seed,
            candidate_k=result.config.candidate_k,
            threads=result.config.threads,
            tower=TowerConfig(
                **{
                    **{
                        name: getattr(result.config.tower, name)
                        for name in TowerConfig.__dataclass_fields__
                    },
                    "logq_correction": False,
                }
            ),
            hnsw=result.config.hnsw,
            ranker=result.config.ranker,
            snapshot=result.config.snapshot,
        )
    )
    logq_rows = {
        "on": {
            "recall_at_default": round(result.metrics[f"recall_at_{result.config.candidate_k}"], 6),
            "tower_final_loss": round(result.metrics["tower_final_loss"], 4),
            "corpus_ndcg_at_10": bench_corpus(result, upper_bound_users=0)["cascade_ndcg_at_k"],
        },
        "off": {
            "recall_at_default": round(
                no_logq.metrics[f"recall_at_{no_logq.config.candidate_k}"], 6
            ),
            "tower_final_loss": round(no_logq.metrics["tower_final_loss"], 4),
            "corpus_ndcg_at_10": bench_corpus(no_logq, upper_bound_users=0)["cascade_ndcg_at_k"],
        },
    }
    return {
        "ef_search": ef_rows,
        "mmr_lambda": mmr_rows,
        "logq_correction": logq_rows,
        "ranker_labels": ranker_rows,
        "rerank_depth": depth_rows,
    }


def _ranker_label_ablation(result: TrainingResult) -> dict[str, dict[str, Any]]:
    """Retrain only the ranker under three label treatments and score both protocols.

    The towers, the index and the feature snapshot are reused, so the only thing that changes is how
    the lambdarank objective treats position bias. This is the ablation that decided the default.
    """
    sessions = result.interactions.periods(PERIOD_TEST)
    users = make_user_sample(sessions, result.bundle.n_users, CORPUS_USERS, seed=result.config.seed)
    variants = {
        "naive_clicks": {"ips_weights": False, "position_debias": False},
        "ips_weights": {"ips_weights": True, "position_debias": False},
        "lgb_unbiased": {"ips_weights": False, "position_debias": True},
    }
    base = {name: getattr(result.config.ranker, name) for name in RankerConfig.__dataclass_fields__}
    user_vectors, item_vectors, retrieved = _training_inputs(result)
    rows: dict[str, dict[str, Any]] = {}
    for name, overrides in variants.items():
        config = RankerConfig(**{**base, **overrides})
        dataset = build_ranker_dataset(
            snapshot=result.train_snapshot,
            label_sessions=result.interactions.periods(PERIOD_LABEL),
            user_vectors=user_vectors,
            item_vectors=item_vectors,
            retrieved=retrieved,
            config=config,
            slots=result.world_config.slots,
            propensity=result.propensity,
        )
        ranker = Ranker.train(dataset, config)
        cascade = _cascade(_with_ranker(result, ranker))
        slate = slate_evaluation(
            sessions=sessions,
            cascade=cascade,
            world=result.world,
            propensity=result.propensity,
            k=10,
            max_sessions=600,
            seed=result.config.seed,
        )
        corpus = corpus_evaluation(
            cascade=cascade, world=result.world, users=users, k=10, upper_bound_users=0
        )
        importance = ranker.feature_importance()
        total = sum(importance.values()) or 1.0
        rows[name] = {
            "slate_ndcg_truth": round(slate.metrics["cascade"]["ndcg_truth"], 6),
            "slate_ndcg_naive": round(slate.metrics["cascade"]["ndcg_naive"], 6),
            "tower_slate_ndcg_truth": round(slate.metrics["tower"]["ndcg_truth"], 6),
            "corpus_ndcg_at_10": corpus["cascade_ndcg_at_k"],
            "tower_corpus_ndcg_at_10": corpus["tower"]["ndcg_at_k"],  # type: ignore[index]
            "tower_score_importance": round(importance["tower_score"] / total, 4),
        }
    return rows


def _training_inputs(
    result: TrainingResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild exactly what ``pipeline.train_bundle`` fed the ranker: user/item vectors and
    ANN candidates, all from the *training* snapshot, so an ablation is comparable with the run it
    is ablating.
    """
    snapshot = result.train_snapshot
    model = result.bundle.model
    user_side = user_side_features(snapshot)
    item_side = item_side_features(snapshot)
    users = np.arange(snapshot.n_users)
    items = np.arange(snapshot.n_items)
    user_vectors, _ = model.user.embed(users, user_side)
    item_vectors, _ = model.item.embed(items, item_side)
    user_vectors = np.asarray(user_vectors, dtype=np.float32)
    item_vectors = np.asarray(item_vectors, dtype=np.float32)
    index = HnswIndex.build(item_vectors, result.config.hnsw, threads=result.config.threads)
    candidate_k = min(result.config.candidate_k, snapshot.n_items)
    retrieved = np.empty((snapshot.n_users, candidate_k), dtype=np.int64)
    for start in range(0, snapshot.n_users, 2_048):
        _, ids = index.search(
            user_vectors[start : start + 2_048],
            candidate_k,
            ef_search=result.config.hnsw.ef_search,
        )
        retrieved[start : start + 2_048] = ids
    return user_vectors, item_vectors, retrieved


def _with_ranker(result: TrainingResult, ranker: Ranker) -> TrainingResult:
    """A copy of the result whose bundle carries a different ranker (nothing else changes)."""
    bundle = replace(result.bundle, ranker=ranker)
    return replace(result, bundle=bundle)


def bench_robustness(result: TrainingResult) -> dict[str, Any]:
    """Shill a median-popularity item and measure where it lands, across attack sizes and shapes.

    Three scenarios, because the defence is not uniform:

    * ``fresh_50`` - 50 single-session accounts, 1% of the user base. The cheap attack the tenure
      weight is aimed at.
    * ``fresh_200`` - 200 single-session accounts, 4% of the user base. A very large attack.
    * ``aged_200`` - 200 accounts with four sessions each. Four times the traffic, full tenure
      weight, so only the CTR prior is left. This is where the defence stops, and the number is
      reported rather than omitted.
    """
    snapshot = result.bundle.snapshot
    target = int(np.argsort(snapshot.item_clickers)[len(snapshot.item_clickers) // 2])
    scenarios = {"fresh_50": (50, 1), "fresh_200": (200, 1), "aged_200": (200, 4)}
    out: dict[str, Any] = {
        "target_item": target,
        "n_users": result.world_config.n_users,
        "n_items": result.world_config.n_items,
    }
    for name, (attackers, sessions) in scenarios.items():
        report = run_shilling_attack(
            interactions=result.interactions,
            catalog=result.catalog,
            model=result.bundle.model,
            ranker=result.bundle.ranker,
            n_users=result.world_config.n_users,
            slots=result.world_config.slots,
            config=AttackConfig(
                target_item=target,
                attackers=attackers,
                filler_items=8,
                sessions_per_attacker=sessions,
                sample_users=12,
                seed=5,
            ),
            snapshot_config=result.config.snapshot,
        )
        out[name] = report.as_dict()
    return out


def bench_canary(result: TrainingResult, fraction: float = 0.5) -> dict[str, Any]:
    from prometheus_client import CollectorRegistry  # noqa: PLC0415

    from retrieval_ranking_cascade.serving.metrics import create_cascade_metrics  # noqa: PLC0415
    from retrieval_ranking_cascade.serving.monitoring import CascadeMonitor  # noqa: PLC0415

    cascade = _cascade(result)
    # Corrupt a *copy*: the bundle's index is shared with every other block in this run, and
    # damaging it here would quietly invalidate everything measured afterwards.
    cascade.swap_index(cascade.index.copy())
    monitor = CascadeMonitor(
        cascade,
        create_cascade_metrics(CollectorRegistry()),
        canary_k=100,
        canary_sample_size=64,
        canary_threshold=0.9,
        seed=result.config.seed,
    )
    healthy = monitor.canary.run_cycle()
    corrupted_rows = corrupt_index(cascade.index, fraction, seed=11)
    broken = monitor.canary.run_cycle()
    return {
        "healthy_recall": round(healthy.recall_mean, 6),
        "healthy_alert": healthy.alert,
        "corrupted_rows": corrupted_rows,
        "corrupted_recall": round(broken.recall_mean, 6),
        "corrupted_alert": broken.alert,
        "cycles_to_detect": broken.cycle - healthy.cycle,
        "cycle_seconds": round(broken.duration_seconds, 4),
    }


def machine() -> dict[str, object]:
    import os  # noqa: PLC0415

    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def run(
    scale: str, *, ablations: bool, threads: int, blocks: tuple[str, ...] = BLOCKS
) -> dict[str, Any]:
    started = time.perf_counter()
    result = train_bundle(TrainingConfig(scale=scale, threads=threads), repo_root=ROOT)
    report: dict[str, Any] = {
        "scale": scale,
        "blocks": list(blocks),
        "train_seconds": round(sum(result.durations.values()), 2),
        "train_metrics": {k: round(float(v), 6) for k, v in sorted(result.metrics.items())},
        "metric_hash": result.metric_hash,
        "bundle_id": result.manifest.bundle_id,
    }
    if "retrieval" in blocks:
        report["retrieval"] = bench_retrieval(result)
    if "ranking" in blocks:
        report.update(bench_ranking_and_bias(result))
    if "corpus" in blocks:
        report["corpus"] = bench_corpus(result)
    if "robustness" in blocks:
        report["robustness"] = bench_robustness(result)
    if "canary" in blocks:
        report["canary"] = bench_canary(result)
    if ablations:
        report["ablations"] = bench_ablations(result)
    report["total_seconds"] = round(time.perf_counter() - started, 2)
    return report


def _slate(report: dict[str, Any], policy: str, metric: str) -> float:
    return float(report["slate"]["metrics"][policy][metric])


def check_thresholds(report: dict[str, Any]) -> list[str]:
    """Enforce the thresholds that apply to the blocks this run actually produced."""
    limits = json.loads(THRESHOLDS.read_text(encoding="utf-8"))["quality"]
    problems: list[str] = []
    if "retrieval" in report:
        recall = report["retrieval"]["recall_at_default"]
        if recall is None or recall < limits["min_recall_at_default"]:
            problems.append(f"recall_at_default = {recall} below {limits['min_recall_at_default']}")
    if "slate" not in report:
        return problems

    cascade_ndcg = _slate(report, "cascade", "ndcg_truth")
    for baseline, floor in limits["min_ndcg_margin_over"].items():
        margin = cascade_ndcg - _slate(report, baseline, "ndcg_truth")
        if margin < floor:
            problems.append(
                f"cascade ndcg_truth margin over {baseline} = {margin:.4f} below {floor}"
            )
    if cascade_ndcg < limits["min_cascade_slate_ndcg_truth"]:
        problems.append(
            f"cascade slate ndcg_truth = {cascade_ndcg:.4f} below "
            f"{limits['min_cascade_slate_ndcg_truth']}"
        )

    corpus = report["corpus"]
    if corpus["cascade_ndcg_at_k"] < limits["min_corpus_ndcg"]:
        problems.append(
            f"corpus cascade ndcg@10 = {corpus['cascade_ndcg_at_k']} below "
            f"{limits['min_corpus_ndcg']}"
        )
    for baseline, floor in limits["min_corpus_ndcg_margin_over"].items():
        margin = corpus["cascade_ndcg_at_k"] - corpus[baseline]["ndcg_at_k"]
        if margin < floor:
            problems.append(
                f"corpus cascade ndcg margin over {baseline} = {margin:.4f} below {floor}"
            )
    retained = corpus.get("upper_bound_retained")
    if retained is not None and retained < limits["min_upper_bound_retained"]:
        problems.append(
            f"upper_bound_retained = {retained} below {limits['min_upper_bound_retained']}"
        )

    bias = report["bias"]
    for kind in ("position", "advantage"):
        if bias[f"ips_{kind}_bias"] > bias[f"naive_{kind}_bias"]:
            problems.append(f"IPS did not shrink the {kind} bias against ground truth")
        reduction = bias.get(f"{kind}_bias_reduction")
        floor = limits[f"min_{kind}_bias_reduction"]
        if reduction is not None and reduction < floor:
            problems.append(f"{kind}_bias_reduction = {reduction} below {floor}")
    eta_error = abs(bias["estimated_eta"] - bias["true_eta"])
    if eta_error > limits["max_eta_error"]:
        problems.append(f"estimated eta off by {eta_error:.3f} (max {limits['max_eta_error']})")

    for scenario, gate in limits["shilling"].items():
        block = report["robustness"][scenario]
        after = block["robust"]["rank_after"]
        if after < gate["min_robust_rank_after"]:
            problems.append(
                f"{scenario}: robust rank_after = {after} above (better than) the floor "
                f"{gate['min_robust_rank_after']}"
            )
        damping = block.get("damping")
        if damping is not None and damping < gate["min_damping"]:
            problems.append(f"{scenario}: shilling damping = {damping} below {gate['min_damping']}")

    canary = report["canary"]
    if canary["healthy_alert"]:
        problems.append("canary alerted on a healthy index")
    if not canary["corrupted_alert"]:
        problems.append("canary did not alert on a corrupted index")
    if canary["cycles_to_detect"] > limits["max_canary_cycles_to_detect"]:
        problems.append(f"canary needed {canary['cycles_to_detect']} cycles to detect corruption")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="cascade quality benchmarks")
    parser.add_argument("--scale", default="small", choices=["tiny", "smoke", "small", "full"])
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    parser.add_argument("--ablations", action="store_true", help="also run the ablation sweeps")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json-only", action="store_true", help="print only the JSON report")
    parser.add_argument(
        "--only",
        default="",
        help=f"comma-separated subset of blocks to run ({', '.join(BLOCKS)}); default all",
    )
    args = parser.parse_args(argv)

    blocks = BLOCKS
    if args.only:
        blocks = tuple(name.strip() for name in args.only.split(",") if name.strip())
        unknown = sorted(set(blocks) - set(BLOCKS))
        if unknown:
            parser.error(f"unknown block(s): {', '.join(unknown)}")

    report = run(args.scale, ablations=args.ablations, threads=args.threads, blocks=blocks)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"machine": machine(), "report": report}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.json_only:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_tables(report)
    if args.check:
        problems = check_thresholds(report)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all quality thresholds met")
    return 0


def _print_tables(report: dict[str, Any]) -> None:
    print(
        f"\nscale={report['scale']}  train={report['train_seconds']}s  "
        f"bundle={report['bundle_id'][:12]}"
    )
    if "retrieval" in report:
        _print_retrieval(report["retrieval"])
    if "slate" in report:
        _print_ranking(report)
    if "corpus" in report:
        _print_corpus(report["corpus"])
    if "robustness" in report:
        _print_robustness(report["robustness"])
    if "canary" in report:
        _print_canary(report["canary"])
    if "ablations" in report:
        _print_ablations(report["ablations"])


def _print_retrieval(retrieval: dict[str, Any]) -> None:
    print(f"\n### Retrieval (recall@{retrieval['k']} vs exact, {retrieval['queries']} queries)\n")
    print("| efSearch | recall@k | recall p05 | search ms/query |")
    print("|---:|---:|---:|---:|")
    for ef, row in retrieval["curve"].items():
        print(
            f"| {ef} | {row['recall_mean']:.4f} | {row['recall_p05']:.4f} | "
            f"{row['search_ms']:.3f} |"
        )
    print(
        f"\nexact brute force over the same {retrieval['queries']} queries: "
        f"{retrieval['exact_search_ms']:.3f} ms/query (ANN at the default efSearch is "
        f"{retrieval['speedup_vs_exact']}x faster)"
    )
    if retrieval["scaling"]:
        print("\n| catalogue size | ANN ms/query | exact ms/query | ANN speedup | recall |")
        print("|---:|---:|---:|---:|---:|")
        for size, row in retrieval["scaling"].items():
            if not size.isdigit():
                continue
            print(
                f"| {size} | {row['ann_ms']:.3f} | {row['exact_ms']:.3f} | {row['speedup']}x | "
                f"{row['recall']:.4f} |"
            )
        fit = retrieval["scaling"].get("fit")
        if fit:
            print(
                f"\nfitted cost: exact {fit['exact_ms_per_1k_items']:.4f} ms per 1k items "
                f"(linear), ANN {fit['ann_fixed_ms']:.2f} ms + "
                f"{fit['ann_ms_per_1k_items']:.4f} ms per 1k items; the two lines meet near "
                f"{fit['crossover_items_estimate']:.0f} items"
            )


def _print_ranking(report: dict[str, Any]) -> None:
    print(f"\n### Slate reranking ({report['slate']['sessions']} held-out sessions, k=10)\n")
    print("| policy | NDCG naive | NDCG IPS | NDCG truth | MRR naive | MRR truth |")
    print("|---|---:|---:|---:|---:|---:|")
    for policy, values in report["slate"]["metrics"].items():
        print(
            f"| {policy} | {values['ndcg_naive']:.4f} | {values['ndcg_ips']:.4f} | "
            f"{values['ndcg_truth']:.4f} | {values['mrr_naive']:.4f} | {values['mrr_truth']:.4f} |"
        )
    bias = report["bias"]
    positions = report["position_bias"]
    print("\n### Position bias (mean relevance per rank, non-randomized impressions)\n")
    print("| rank | naive CTR | IPS estimate | true relevance |")
    print("|---:|---:|---:|---:|")
    for rank, naive, ips, truth in zip(
        positions["ranks"],
        positions["naive_relevance"],
        positions["ips_relevance"],
        positions["true_relevance"],
        strict=True,
    ):
        print(f"| {rank} | {naive:.4f} | {ips:.4f} | {truth:.4f} |")
    print(
        f"\nmean absolute error vs truth: naive {bias['naive_position_bias']:.4f}, "
        f"IPS {bias['ips_position_bias']:.4f} "
        f"(reduction {bias['position_bias_reduction']})"
    )
    print(
        f"policy-advantage bias: naive {bias['naive_advantage_bias']:.4f}, "
        f"IPS {bias['ips_advantage_bias']:.4f} "
        f"(reduction {bias['advantage_bias_reduction']}); "
        f"eta estimated {bias['estimated_eta']} vs true {bias['true_eta']}"
    )


def _print_corpus(corpus: dict[str, Any]) -> None:
    print(f"\n### Full corpus top-{corpus['k']} ({corpus['users']} users)\n")
    print("| policy | NDCG@10 | intra-list sim | coverage | cold-item share | cold-user NDCG |")
    print("|---|---:|---:|---:|---:|---:|")
    for policy in ("popularity", "tower", "cascade_relevance", "cascade", "exact_rank"):
        row = corpus.get(policy)
        if not isinstance(row, dict):
            continue
        cold = row["cold_user_ndcg_at_k"]
        print(
            f"| {policy} | {row['ndcg_at_k']:.4f} | {row['intra_list_similarity']:.4f} | "
            f"{row['catalog_coverage']:.4f} | {row['cold_item_share']:.4f} | "
            f"{'n/a' if cold is None else f'{cold:.4f}'} |"
        )
    print(
        f"\ncascade_relevance retains {corpus.get('upper_bound_retained')} of the exact-ranking "
        f"upper bound (gap {corpus.get('upper_bound_gap')}); MMR then costs "
        f"{corpus['diversity_cost']} NDCG; simulated CTR {corpus['simulated']['ctr']}"
    )


def _print_robustness(robustness: dict[str, Any]) -> None:
    print("\n### Shilling robustness\n")
    print(
        "| attack | accounts | sessions each | item statistics | rank before | rank after | "
        "lift | damping |"
    )
    print("|---|---:|---:|---|---:|---:|---:|---:|")
    for scenario in ("fresh_50", "fresh_200", "aged_200"):
        block = robustness.get(scenario)
        if block is None:
            continue
        for name in ("robust", "naive"):
            row = block[name]
            damping = block["damping"] if name == "robust" else ""
            print(
                f"| {scenario} | {block['attackers']} | "
                f"{block['sessions_per_attacker']} | {name} | {row['rank_before']} | "
                f"{row['rank_after']} | {row['lift']}x | {damping} |"
            )
    print(
        f"\ntarget item {robustness['target_item']} of {robustness['n_items']}, "
        f"{robustness['n_users']} legitimate users; a lower rank_after is a more "
        f"successful attack."
    )
    print(
        "The two aggregations rank the untouched catalogue differently, so compare rank_after "
        "first; damping is the ratio of lifts and is only meaningful while the robust build still "
        "keeps the item out of the visible ranks."
    )


def _print_canary(canary: dict[str, Any]) -> None:
    print(
        f"\n### Recall canary\n\nhealthy recall {canary['healthy_recall']:.4f} "
        f"(alert {canary['healthy_alert']}) -> after corrupting {canary['corrupted_rows']} vectors "
        f"{canary['corrupted_recall']:.4f} (alert {canary['corrupted_alert']}) in "
        f"{canary['cycles_to_detect']} cycle(s) of {canary['cycle_seconds']}s"
    )


def _print_ablations(ablations: dict[str, Any]) -> None:
    print("\n### Ablations\n")
    print("| logQ correction | recall@default | tower loss | corpus NDCG@10 |")
    print("|---|---:|---:|---:|")
    for name, row in ablations["logq_correction"].items():
        print(
            f"| {name} | {row['recall_at_default']:.4f} | {row['tower_final_loss']:.4f} | "
            f"{row['corpus_ndcg_at_10']:.4f} |"
        )
    print(
        "\n| ranker labels | slate NDCG truth | slate NDCG naive | tower slate NDCG | "
        "corpus NDCG | tower corpus NDCG | tower_score gain share |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, row in ablations["ranker_labels"].items():
        print(
            f"| {name} | {row['slate_ndcg_truth']:.4f} | {row['slate_ndcg_naive']:.4f} | "
            f"{row['tower_slate_ndcg_truth']:.4f} | {row['corpus_ndcg_at_10']:.4f} | "
            f"{row['tower_corpus_ndcg_at_10']:.4f} | {row['tower_score_importance']:.3f} |"
        )
    print("\n| MMR lambda | intra-list sim | mean true relevance | distinct categories |")
    print("|---:|---:|---:|---:|")
    for lam, row in ablations["mmr_lambda"].items():
        print(
            f"| {lam} | {row['intra_list_similarity']:.4f} | {row['mean_true_relevance']:.4f} | "
            f"{row['distinct_categories']} |"
        )
    print("\n| rerank depth | corpus NDCG@10 | served NDCG@10 | coverage | ms/request |")
    print("|---:|---:|---:|---:|---:|")
    for depth, row in ablations["rerank_depth"].items():
        print(
            f"| {depth} | {row['corpus_ndcg_at_10']:.4f} | {row['served_ndcg_at_10']:.4f} | "
            f"{row['coverage']:.4f} | {row['ms_per_request']:.2f} |"
        )
    print("\n| efSearch | cascade NDCG@10 | coverage |")
    print("|---:|---:|---:|")
    for ef, row in ablations["ef_search"].items():
        print(f"| {ef} | {row['cascade_ndcg_at_k']:.4f} | {row['coverage']:.4f} |")


if __name__ == "__main__":
    raise SystemExit(main())
