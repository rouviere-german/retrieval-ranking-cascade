"""Training pipeline: synthetic world -> contracts -> towers -> index -> ranker -> bundle.

The pipeline is a pure function of its :class:`TrainingConfig` (and therefore of its seeds): the
same config produces the same impression log, the same tower weights, the same ranker text and the
same metrics. ``metric_hash`` in the result is the hash of the rounded training metrics, which is
what ``tests/integration/test_pipeline.py`` asserts against to catch accidental non-determinism.

Two feature snapshots exist on purpose:

* the **training snapshot** aggregates the history period only. It supplies the tower's side
  features and the ranker's features, so no feature a ranker row uses can have been computed from
  the click it is trained to predict.
* the **serving snapshot** adds the label period. It ships in the bundle, so serving features are
  fresher than training features - which is a real distribution shift, not a bug, and it is exactly
  what the ``feature_psi`` monitor is there to bound.

Because the item tower's side features come from the snapshot, item embeddings differ between the
two snapshots. The candidate-mining index at training time is built from training-snapshot vectors,
and the bundle's index from serving-snapshot vectors, so each index always matches the vectors its
consumer recomputes.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from retrieval_ranking_cascade.core.canonical import canonical_json, sha256_hex
from retrieval_ranking_cascade.core.logging import get_logger
from retrieval_ranking_cascade.data.generator import (
    PERIOD_HISTORY,
    PERIOD_LABEL,
    PERIOD_TEST,
    Catalog,
    Interactions,
    World,
    WorldConfig,
    generate,
    world_config,
)
from retrieval_ranking_cascade.evaluation.propensity import (
    PropensityModel,
    estimate_propensities,
)
from retrieval_ranking_cascade.ml.contracts import (
    catalog_schema,
    impressions_schema,
    validate,
)
from retrieval_ranking_cascade.ml.manifest import RunManifest, create_manifest, seed_everything
from retrieval_ranking_cascade.ml.serialization import dumps_npz
from retrieval_ranking_cascade.ranking.features import (
    FEATURE_NAMES,
    FeatureSnapshot,
    SnapshotConfig,
    build_snapshot,
    item_side_features,
    user_side_features,
)
from retrieval_ranking_cascade.ranking.ranker import (
    Ranker,
    RankerConfig,
    RankerDataset,
    build_ranker_dataset,
)
from retrieval_ranking_cascade.retrieval.ann import (
    HnswIndex,
    HnswParams,
    exact_top_k,
    recall_at_k,
    set_faiss_threads,
)
from retrieval_ranking_cascade.retrieval.two_tower import (
    TowerConfig,
    TrainingHistory,
    train_two_tower,
)
from retrieval_ranking_cascade.serving.bundle import (
    Bundle,
    BundleManifest,
    build_manifest,
    write_bundle,
)

__all__ = [
    "REFERENCE_SAMPLE",
    "TrainingConfig",
    "TrainingResult",
    "held_out_sessions",
    "train_bundle",
]

REFERENCE_SAMPLE = 20_000
_PACKAGES = ("numpy", "pandas", "pandera", "pydantic", "faiss-cpu", "lightgbm", "duckdb")
_log = get_logger("retrieval_ranking_cascade.pipeline")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Everything the pipeline needs. ``scale`` selects a preset world from ``data.generator``."""

    scale: str = "small"
    seed: int = 7
    candidate_k: int = 500
    recall_sample: int = 200
    threads: int = 4
    # Ranks with fewer randomized impressions than this inherit the nearest reliable rank.
    propensity_min_impressions: int = 20
    tower: TowerConfig = field(default_factory=lambda: TowerConfig(epochs=8))
    hnsw: HnswParams = field(default_factory=lambda: HnswParams(m=48, ef_construction=300))
    ranker: RankerConfig = field(default_factory=RankerConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)

    def world(self) -> WorldConfig:
        return world_config(self.scale, seed=self.seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "seed": self.seed,
            "candidate_k": self.candidate_k,
            "tower": asdict(self.tower),
            "hnsw": asdict(self.hnsw),
            "ranker": asdict(self.ranker),
            "snapshot": asdict(self.snapshot),
        }


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The trained bundle plus the data and splits an evaluation run needs."""

    config: TrainingConfig
    world_config: WorldConfig
    world: World
    catalog: Catalog
    interactions: Interactions
    bundle: Bundle
    artifacts: dict[str, bytes]
    train_snapshot: FeatureSnapshot
    propensity: PropensityModel
    ranker_dataset: RankerDataset
    tower_history: TrainingHistory
    metrics: dict[str, float]
    durations: dict[str, float]

    @property
    def manifest(self) -> BundleManifest:
        return self.bundle.manifest

    def period(self, *periods: int) -> Interactions:
        return self.interactions.periods(*periods)

    @property
    def metric_hash(self) -> str:
        """Hash of the rounded metrics: a single value a determinism test can compare."""
        rounded = {name: round(float(value), 6) for name, value in sorted(self.metrics.items())}
        return sha256_hex(canonical_json(rounded))

    def write(self, directory: Path) -> Path:
        return write_bundle(directory, self.bundle.manifest, self.artifacts)


def _lockfile_hash(root: Path | None) -> str | None:
    if root is None:
        return None
    path = root / "uv.lock"
    return sha256_hex(path.read_bytes()) if path.is_file() else None


def _retrieve_all(
    index: HnswIndex,
    user_vectors: npt.NDArray[np.float32],
    k: int,
    ef_search: int,
    chunk: int = 2_048,
) -> npt.NDArray[np.int64]:
    """ANN top-``k`` for every user, in chunks so memory stays bounded."""
    out = np.empty((user_vectors.shape[0], min(k, index.size)), dtype=np.int64)
    for start in range(0, user_vectors.shape[0], chunk):
        _, ids = index.search(user_vectors[start : start + chunk], k, ef_search=ef_search)
        out[start : start + chunk] = ids
    return out


def _recall_at_default(
    *,
    index: HnswIndex,
    user_vectors: npt.NDArray[np.float32],
    item_vectors: npt.NDArray[np.float32],
    k: int,
    ef_search: int,
    sample: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    size = min(sample, user_vectors.shape[0])
    rows = rng.choice(user_vectors.shape[0], size=size, replace=False)
    _, approx = index.search(user_vectors[rows], k, ef_search=ef_search)
    _, exact = exact_top_k(user_vectors[rows], item_vectors, k)
    return float(recall_at_k(approx, exact).mean())


def _reference_arrays(
    dataset: RankerDataset, norms: npt.NDArray[np.floating[Any]], seed: int
) -> dict[str, npt.NDArray[np.float64]]:
    """Subsampled training distributions the drift monitors compare live traffic against."""
    rng = np.random.default_rng(seed)
    rows = dataset.features.shape[0]
    take = min(REFERENCE_SAMPLE, rows)
    index = rng.choice(rows, size=take, replace=False) if take < rows else np.arange(rows)
    arrays = {
        f"feature_{name}": dataset.features[index, column].astype(np.float64)
        for column, name in enumerate(FEATURE_NAMES)
    }
    arrays["embedding_norms"] = np.asarray(norms, dtype=np.float64)
    return arrays


def train_bundle(  # noqa: PLR0915 - the pipeline is a linear sequence of stages
    config: TrainingConfig | None = None, *, repo_root: Path | None = None
) -> TrainingResult:
    """Run the whole pipeline in memory and return a loadable, hash-consistent bundle."""
    cfg = config or TrainingConfig()
    seed_everything(cfg.seed)
    set_faiss_threads(cfg.threads)
    durations: dict[str, float] = {}
    world_cfg = cfg.world()

    step = time.perf_counter()
    world, catalog, interactions = generate(world_cfg)
    durations["generate_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    validate(
        interactions.to_frame(),
        impressions_schema(
            n_users=world_cfg.n_users, n_items=world_cfg.n_items, slots=world_cfg.slots
        ),
    )
    validate(
        catalog.to_frame(),
        catalog_schema(n_items=world_cfg.n_items, n_categories=world_cfg.n_categories),
    )
    durations["validate_seconds"] = time.perf_counter() - step

    history = interactions.periods(PERIOD_HISTORY)
    label = interactions.periods(PERIOD_LABEL)
    # Propensities come from the randomized slice of the data available at training time; the test
    # period is never touched here.
    propensity = estimate_propensities(
        interactions.periods(PERIOD_HISTORY, PERIOD_LABEL),
        slots=world_cfg.slots,
        min_impressions=cfg.propensity_min_impressions,
    )

    step = time.perf_counter()
    train_snapshot = build_snapshot(
        history, catalog, n_users=world_cfg.n_users, config=cfg.snapshot
    )
    serving_snapshot = build_snapshot(
        interactions.periods(PERIOD_HISTORY, PERIOD_LABEL),
        catalog,
        n_users=world_cfg.n_users,
        config=cfg.snapshot,
    )
    durations["snapshot_seconds"] = time.perf_counter() - step

    train_user_side = user_side_features(train_snapshot)
    train_item_side = item_side_features(train_snapshot)
    positives = history.select(history.clicked == 1)
    step = time.perf_counter()
    model, tower_history = train_two_tower(
        users=positives.user_id,
        items=positives.item_id,
        user_side=train_user_side,
        item_side=train_item_side,
        known_items=np.unique(history.item_id),
        config=cfg.tower,
        on_epoch=lambda epoch, loss: _log.info("tower_epoch", epoch=epoch, loss=round(loss, 5)),
    )
    durations["tower_seconds"] = time.perf_counter() - step

    all_users = np.arange(world_cfg.n_users)
    all_items = np.arange(world_cfg.n_items)
    train_user_vectors, _ = model.user.embed(all_users, train_user_side)
    train_item_vectors, _ = model.item.embed(all_items, train_item_side)

    step = time.perf_counter()
    mining_index = HnswIndex.build(train_item_vectors, cfg.hnsw, threads=cfg.threads)
    retrieved = _retrieve_all(
        mining_index,
        np.asarray(train_user_vectors, dtype=np.float32),
        cfg.candidate_k,
        cfg.hnsw.ef_search,
    )
    durations["mining_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    dataset = build_ranker_dataset(
        snapshot=train_snapshot,
        label_sessions=label,
        user_vectors=np.asarray(train_user_vectors, dtype=np.float32),
        item_vectors=np.asarray(train_item_vectors, dtype=np.float32),
        retrieved=retrieved,
        config=cfg.ranker,
        slots=world_cfg.slots,
        propensity=propensity,
    )
    ranker = Ranker.train(dataset, cfg.ranker)
    durations["ranker_seconds"] = time.perf_counter() - step

    serving_user_side = user_side_features(serving_snapshot)
    serving_item_side = item_side_features(serving_snapshot)
    serving_user_vectors, _ = model.user.embed(all_users, serving_user_side)
    serving_item_vectors, item_norms = model.item.embed(all_items, serving_item_side)
    step = time.perf_counter()
    index = HnswIndex.build(serving_item_vectors, cfg.hnsw, threads=cfg.threads)
    durations["index_seconds"] = time.perf_counter() - step
    set_faiss_threads(1)

    recall = _recall_at_default(
        index=index,
        user_vectors=np.asarray(serving_user_vectors, dtype=np.float32),
        item_vectors=np.asarray(serving_item_vectors, dtype=np.float32),
        k=cfg.candidate_k,
        ef_search=cfg.hnsw.ef_search,
        sample=cfg.recall_sample,
        seed=cfg.seed,
    )
    metrics = {
        "train_pairs": float(len(positives)),
        "train_click_rate": float(history.click_rate),
        "ranker_queries": float(dataset.n_queries),
        "ranker_rows": float(dataset.labels.shape[0]),
        "ranker_trees": float(ranker.num_trees),
        "tower_final_loss": float(tower_history.epoch_loss[-1]),
        f"recall_at_{cfg.candidate_k}": recall,
        "embedding_norm_mean": float(np.mean(item_norms)),
        "impressions": float(len(interactions)),
        "ips_weight_mean": float(dataset.weights.mean()),
    }

    reference = _reference_arrays(dataset, item_norms, cfg.seed)
    artifacts = {
        "tower.npz": dumps_npz(model.to_arrays()),
        "index.faiss": index.to_bytes(),
        "ranker.txt": ranker.to_text().encode("utf-8"),
        "snapshot.npz": dumps_npz(serving_snapshot.arrays()),
        "reference.npz": dumps_npz(reference),
    }
    run: RunManifest = create_manifest(
        seed=cfg.seed,
        datasets=[],
        params=dict(cfg.to_dict()),
        metrics=metrics,
        packages=_PACKAGES,
    )
    manifest = build_manifest(
        scale=cfg.scale,
        seed=cfg.seed,
        data_sha256=interactions.sha256(),
        snapshot=serving_snapshot,
        candidate_k=cfg.candidate_k,
        embedding_norms=item_norms,
        embedding_dim=model.output_dim,
        world=asdict(world_cfg),
        tower=asdict(cfg.tower),
        hnsw=asdict(cfg.hnsw),
        ranker=asdict(cfg.ranker),
        snapshot_params=asdict(cfg.snapshot),
        artifacts=artifacts,
        metrics=metrics,
        run=run,
        slots=world_cfg.slots,
        lockfile_sha256=_lockfile_hash(repo_root),
        propensities=[float(value) for value in propensity.propensities],
    )
    bundle = Bundle(
        manifest=manifest,
        model=model,
        index=index,
        ranker=ranker,
        snapshot=serving_snapshot,
        item_vectors=np.asarray(serving_item_vectors, dtype=np.float32),
        user_side=serving_user_side,
        item_side=serving_item_side,
        reference=reference,
    )
    _log.info(
        "pipeline_complete",
        bundle=manifest.short_id,
        scale=cfg.scale,
        recall=round(recall, 4),
        seconds=round(sum(durations.values()), 2),
    )
    return TrainingResult(
        config=cfg,
        world_config=world_cfg,
        world=world,
        catalog=catalog,
        interactions=interactions,
        bundle=bundle,
        artifacts=artifacts,
        train_snapshot=train_snapshot,
        propensity=propensity,
        ranker_dataset=dataset,
        tower_history=tower_history,
        metrics=metrics,
        durations=durations,
    )


def held_out_sessions(result: TrainingResult) -> Interactions:
    """The held-out test-period sessions offline evaluation scores against."""
    return result.interactions.periods(PERIOD_TEST)
