"""The training pipeline end to end: contracts, reproducibility, and what it hands to serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from retrieval_ranking_cascade.data.generator import PERIOD_HISTORY, PERIOD_LABEL, PERIOD_TEST
from retrieval_ranking_cascade.pipeline import (
    REFERENCE_SAMPLE,
    TrainingConfig,
    held_out_sessions,
    train_bundle,
)
from retrieval_ranking_cascade.ranking.features import FEATURE_NAMES
from retrieval_ranking_cascade.retrieval.ann import exact_top_k, recall_at_k
from retrieval_ranking_cascade.retrieval.two_tower import TowerConfig
from retrieval_ranking_cascade.serving.bundle import load_bundle

pytestmark = pytest.mark.integration


def _tiny_config(**overrides: Any) -> TrainingConfig:
    values: dict[str, Any] = {
        "scale": "tiny",
        "threads": 2,
        "candidate_k": 200,
        "recall_sample": 40,
        "tower": TowerConfig(epochs=3, batch_size=256, uniform_negatives=256),
    }
    values.update(overrides)
    return TrainingConfig(**values)


# --- what the pipeline produces ---


def test_the_pipeline_returns_a_loadable_bundle_and_its_inputs(tiny_training: Any) -> None:
    result = tiny_training
    assert result.bundle.bundle_id == result.manifest.bundle_id
    assert result.bundle.n_items == result.world_config.n_items
    assert result.bundle.n_users == result.world_config.n_users
    assert result.bundle.index.size == result.world_config.n_items
    assert result.bundle.ranker.num_trees > 0
    assert len(result.interactions) > 0
    assert set(result.artifacts) == {
        "tower.npz",
        "index.faiss",
        "ranker.txt",
        "snapshot.npz",
        "reference.npz",
    }


def test_the_reported_metrics_cover_every_stage(tiny_training: Any) -> None:
    metrics = tiny_training.metrics
    for name in (
        "train_pairs",
        "train_click_rate",
        "ranker_queries",
        "ranker_rows",
        "ranker_trees",
        "tower_final_loss",
        "embedding_norm_mean",
        "impressions",
        "ips_weight_mean",
    ):
        assert name in metrics
        assert np.isfinite(metrics[name])
    assert metrics[f"recall_at_{tiny_training.config.candidate_k}"] > 0.9
    assert 0.0 < metrics["train_click_rate"] < 1.0
    assert metrics["ips_weight_mean"] >= 1.0


def test_every_stage_is_timed(tiny_training: Any) -> None:
    durations = tiny_training.durations
    for name in (
        "generate_seconds",
        "validate_seconds",
        "snapshot_seconds",
        "tower_seconds",
        "mining_seconds",
        "ranker_seconds",
        "index_seconds",
    ):
        assert durations[name] >= 0.0
    assert sum(durations.values()) > 0.0


def test_the_tower_loss_falls_over_the_epochs(tiny_training: Any) -> None:
    losses = tiny_training.tower_history.epoch_loss
    assert len(losses) == tiny_training.config.tower.epochs
    assert losses[-1] < losses[0]
    assert tiny_training.tower_history.steps > 0


# --- the two snapshots ---


def test_training_features_never_see_the_label_period(tiny_training: Any) -> None:
    """The ranker's features come from history only; the bundle's serving snapshot adds labels."""
    result = tiny_training
    train = result.train_snapshot
    serving = result.bundle.snapshot
    history = result.period(PERIOD_HISTORY)
    assert train.day <= serving.day
    assert float(train.user_impressions.sum()) == pytest.approx(float(len(history)))
    assert float(serving.user_impressions.sum()) > float(train.user_impressions.sum())
    assert train.sha256() != serving.sha256()


def test_the_serving_index_matches_the_serving_snapshot_s_vectors(tiny_training: Any) -> None:
    """Each index is built from the vectors its consumer recomputes, or recall is meaningless."""
    bundle = tiny_training.bundle
    queries = bundle.user_vectors(np.arange(min(40, bundle.n_users)))
    _, approx = bundle.index.search(queries, 50, ef_search=bundle.index.params.ef_search)
    _, exact = exact_top_k(queries, bundle.item_vectors, 50)
    assert float(recall_at_k(approx, exact).mean()) > 0.9


def test_the_periods_partition_the_log(tiny_training: Any) -> None:
    result = tiny_training
    total = len(result.interactions)
    parts = sum(
        len(result.period(period)) for period in (PERIOD_HISTORY, PERIOD_LABEL, PERIOD_TEST)
    )
    assert parts == total
    assert len(held_out_sessions(result)) == len(result.period(PERIOD_TEST))
    assert len(held_out_sessions(result)) > 0


# --- propensities ---


def test_propensities_are_estimated_from_the_training_periods_only(tiny_training: Any) -> None:
    model = tiny_training.propensity
    assert model.propensities[0] == pytest.approx(1.0)
    assert model.slots == tiny_training.world_config.slots
    # These are empirical per-rank ratios, so they are noisy rather than strictly decreasing; what
    # has to hold is the level: the bottom of the page is examined far less than the top.
    assert model.propensities.max() <= 1.0
    assert float(model.propensities[-3:].mean()) < 0.5
    assert tiny_training.manifest.propensities == [
        pytest.approx(float(value)) for value in model.propensities
    ]


# --- drift reference ---


def test_the_reference_holds_one_subsampled_column_per_feature(tiny_training: Any) -> None:
    reference = tiny_training.bundle.reference
    for name in FEATURE_NAMES:
        column = reference[f"feature_{name}"]
        assert column.size <= REFERENCE_SAMPLE
        assert column.size > 0
        assert np.all(np.isfinite(column))
    norms = reference["embedding_norms"]
    assert norms.size == tiny_training.world_config.n_items
    assert float(norms.mean()) == pytest.approx(
        tiny_training.metrics["embedding_norm_mean"], rel=1e-5
    )


# --- reproducibility ---


def test_the_same_config_reproduces_the_data_hash_and_the_metric_hash(tmp_path: Path) -> None:
    """The determinism gate: values a test can compare across two whole training runs.

    Every artifact comes out byte-identical, so the artifact digests in the two manifests match.
    The ``bundle_id`` does not, and deliberately: it also commits to ``created_at``, so two runs of
    the same config are two distinguishable deployments of identical content.
    """
    config = _tiny_config(seed=5)
    first = train_bundle(config)
    second = train_bundle(config)
    assert first.interactions.sha256() == second.interactions.sha256()
    assert first.metric_hash == second.metric_hash
    assert first.bundle.ranker.to_text() == second.bundle.ranker.to_text()
    assert first.bundle.snapshot.sha256() == second.bundle.snapshot.sha256()
    assert first.manifest.artifacts == second.manifest.artifacts
    assert first.manifest.bundle_id != second.manifest.bundle_id
    np.testing.assert_allclose(first.bundle.item_vectors, second.bundle.item_vectors, rtol=0)
    # And the bundle written to disk is byte-identical too.
    first.write(tmp_path / "a")
    second.write(tmp_path / "b")
    for name in first.artifacts:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_a_different_seed_changes_the_data_and_the_metrics(tiny_training: Any) -> None:
    other = train_bundle(_tiny_config(seed=tiny_training.config.seed + 1))
    assert other.interactions.sha256() != tiny_training.interactions.sha256()
    assert other.metric_hash != tiny_training.metric_hash
    assert other.manifest.artifacts != tiny_training.manifest.artifacts


def test_the_metric_hash_is_a_hash_of_the_rounded_metrics(tiny_training: Any) -> None:
    assert len(tiny_training.metric_hash) == 64
    assert int(tiny_training.metric_hash, 16) >= 0


def test_the_run_manifest_records_the_seed_params_and_package_versions(
    tiny_training: Any,
) -> None:
    run = tiny_training.manifest.run
    assert run.seed == tiny_training.config.seed
    assert run.params["scale"] == "tiny"
    assert "tower" in run.params
    packages = dict(run.packages)
    for name in ("numpy", "faiss-cpu", "lightgbm", "duckdb", "pandera"):
        assert name in packages
        assert packages[name]


def test_the_lockfile_hash_is_recorded_when_a_repo_root_is_given(
    tiny_training: Any, tmp_path: Path
) -> None:
    assert tiny_training.manifest.lockfile_sha256 is not None
    without = train_bundle(_tiny_config(seed=9), repo_root=None)
    assert without.manifest.lockfile_sha256 is None
    del tmp_path


def test_the_config_round_trips_to_a_dict_for_the_manifest(tiny_training: Any) -> None:
    payload = tiny_training.config.to_dict()
    assert payload["scale"] == "tiny"
    assert payload["tower"]["epochs"] == 3
    assert payload["hnsw"]["m"] >= 4
    assert payload["ranker"]["num_boost_round"] > 0
    assert payload["snapshot"]["robust"] is True


# --- writing and loading ---


def test_the_written_bundle_is_the_one_the_pipeline_built(
    tiny_training: Any, tiny_bundle_dir: Path
) -> None:
    loaded = load_bundle(tiny_bundle_dir)
    assert loaded.bundle_id == tiny_training.bundle.bundle_id
    assert loaded.snapshot.sha256() == tiny_training.bundle.snapshot.sha256()
    assert loaded.ranker.to_text() == tiny_training.bundle.ranker.to_text()
    assert loaded.manifest.metrics == tiny_training.manifest.metrics


def test_a_smoke_scale_run_finishes_and_meets_a_recall_threshold() -> None:
    """The 1% training smoke CI runs: cheap, but it has to assert on a real metric."""
    result = train_bundle(
        _tiny_config(
            scale="smoke",
            candidate_k=100,
            recall_sample=30,
            tower=TowerConfig(epochs=2, batch_size=256, uniform_negatives=256),
        )
    )
    assert result.metrics["recall_at_100"] > 0.9
    assert result.metrics["ranker_queries"] > 0
    assert result.bundle.index.size == result.world_config.n_items
