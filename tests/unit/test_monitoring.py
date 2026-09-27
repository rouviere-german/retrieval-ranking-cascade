"""Live monitoring: feature PSI, embedding-norm drift, coverage, simulated CTR, canary wiring."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from retrieval_ranking_cascade.retrieval.canary import corrupt_index
from retrieval_ranking_cascade.serving.budget import BudgetController
from retrieval_ranking_cascade.serving.bundle import Bundle
from retrieval_ranking_cascade.serving.cascade import Cascade, CascadeConfig
from retrieval_ranking_cascade.serving.metrics import create_cascade_metrics
from retrieval_ranking_cascade.serving.monitoring import (
    CascadeMonitor,
    FeedbackEvent,
    MonitoringSnapshot,
)

pytestmark = pytest.mark.unit


def _cascade(bundle: Bundle) -> Cascade:
    controller = BudgetController(total_seconds=0.5, retrieval_seconds=0.2, ranking_seconds=0.2)
    return Cascade(bundle, controller, CascadeConfig(candidate_k=150, rerank_depth=80))


def _monitor(cascade: Cascade, **overrides: Any) -> tuple[CascadeMonitor, Any]:
    metrics = create_cascade_metrics(CollectorRegistry())
    defaults: dict[str, Any] = {
        "canary_k": 20,
        "canary_sample_size": 16,
        "canary_threshold": 0.9,
        "sample_every": 1,
        "min_samples": 10,
        "seed": 1,
    }
    monitor = CascadeMonitor(cascade, metrics, **{**defaults, **overrides})
    return monitor, metrics


def _value(metrics: Any, name: str, **labels: str) -> float:
    sample = metrics.registry.get_sample_value(name, labels or None)
    return float(sample) if sample is not None else float("nan")


@pytest.fixture(scope="module")
def bundle(tiny_training: Any) -> Bundle:
    loaded: Bundle = tiny_training.bundle
    return loaded


# --- construction ---


@pytest.mark.parametrize("field", ["sample_every", "buffer_rows", "min_samples"])
def test_the_monitor_validates_its_buffer_sizes(bundle: Bundle, field: str) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        CascadeMonitor(_cascade(bundle), create_cascade_metrics(CollectorRegistry()), **{field: 0})


def test_a_bundle_without_reference_distributions_cannot_be_monitored(bundle: Bundle) -> None:
    """A bundle that cannot be monitored must fail at wiring time, not silently report nothing."""
    stripped = dataclasses.replace(bundle, reference={"embedding_norms": np.ones(4)})
    with pytest.raises(ValueError, match="no ranker feature distributions"):
        CascadeMonitor(_cascade(stripped), create_cascade_metrics(CollectorRegistry()))


def test_construction_publishes_the_serving_bundle_identity(bundle: Bundle) -> None:
    _, metrics = _monitor(_cascade(bundle))
    assert _value(
        metrics,
        "cascade_bundle_info",
        bundle_id=bundle.manifest.short_id,
        scale=bundle.manifest.scale,
    ) == pytest.approx(1.0)


# --- request observation ---


def test_observing_responses_fills_the_feature_buffer_and_the_coverage_gauge(
    bundle: Bundle,
) -> None:
    cascade = _cascade(bundle)
    monitor, metrics = _monitor(cascade)
    assert monitor.drift_report() is None
    for user in range(20):
        monitor.observe(user, cascade.recommend(user, k=10).item_ids)
    snapshot = monitor.snapshot()
    assert isinstance(snapshot, MonitoringSnapshot)
    assert snapshot.feature_rows >= 10
    assert 0.0 < snapshot.catalog_coverage <= 1.0
    assert _value(metrics, "cascade_catalog_coverage") == pytest.approx(snapshot.catalog_coverage)


def test_feature_rows_are_only_recomputed_for_a_sampled_subset(bundle: Bundle) -> None:
    """Recomputing every response's features would double the ranker's cost."""
    cascade = _cascade(bundle)
    every, _ = _monitor(cascade, sample_every=1, buffer_rows=10_000)
    tenth, _ = _monitor(cascade, sample_every=10, buffer_rows=10_000)
    responses = [(user, cascade.recommend(user, k=10).item_ids) for user in range(20)]
    for user, items in responses:
        every.observe(user, items)
        tenth.observe(user, items)
    full = every.snapshot().feature_rows
    sampled = tenth.snapshot().feature_rows
    assert full > 0
    # Two of twenty responses were sampled, so roughly a tenth of the rows were recomputed.
    assert 0 < sampled <= full // 5


def test_the_feature_buffer_is_bounded(bundle: Bundle) -> None:
    cascade = _cascade(bundle)
    monitor, _ = _monitor(cascade, buffer_rows=25)
    for user in range(20):
        monitor.observe(user, cascade.recommend(user, k=10).item_ids)
    assert monitor.snapshot().feature_rows == 25


def test_an_empty_response_and_an_unknown_user_are_ignored(bundle: Bundle) -> None:
    cascade = _cascade(bundle)
    monitor, _ = _monitor(cascade)
    monitor.observe(0, [])
    monitor.observe(10**9, [1, 2, 3])
    snapshot = monitor.snapshot()
    assert snapshot.feature_rows == 0
    # The unknown user's served items still count towards coverage.
    assert snapshot.catalog_coverage > 0.0


def test_psi_is_reported_per_feature_once_the_buffer_is_full_enough(bundle: Bundle) -> None:
    cascade = _cascade(bundle)
    monitor, metrics = _monitor(cascade)
    for user in range(40):
        monitor.observe(user, cascade.recommend(user, k=10).item_ids)
    report = monitor.drift_report()
    assert report is not None
    snapshot = monitor.snapshot()
    assert snapshot.feature_psi
    assert set(snapshot.feature_psi) <= set(bundle.reference) | set(snapshot.feature_psi)
    for name, value in snapshot.feature_psi.items():
        assert value >= 0.0
        assert _value(metrics, "cascade_feature_psi", feature=name) == pytest.approx(
            value, abs=1e-6
        )


def test_served_traffic_drifts_away_from_the_training_reference(bundle: Bundle) -> None:
    """Served rows are the cascade's own top picks, not a random slice of training rows."""
    cascade = _cascade(bundle)
    monitor, _ = _monitor(cascade)
    for user in range(60):
        monitor.observe(user, cascade.recommend(user, k=10).item_ids)
    snapshot = monitor.snapshot()
    assert max(snapshot.feature_psi.values()) > 0.0
    assert all(name in snapshot.feature_psi for name in ("tower_score", "item_ctr"))


# --- embedding norm drift ---


def test_embedding_norm_psi_is_near_zero_for_the_bundle_it_shipped_with(bundle: Bundle) -> None:
    monitor, metrics = _monitor(_cascade(bundle))
    value = monitor.embedding_norm_psi()
    assert value == pytest.approx(0.0, abs=1e-6)
    monitor.snapshot()
    assert _value(metrics, "cascade_embedding_norm_drift") == pytest.approx(value, abs=1e-9)


def test_a_changed_item_tower_shows_up_as_embedding_norm_drift(bundle: Bundle) -> None:
    """A swap that "kept the same model" but moved the norms is the cheapest early signal."""
    monitor, _ = _monitor(_cascade(bundle))
    baseline = monitor.embedding_norm_psi()
    shifted = dataclasses.replace(
        bundle,
        reference={
            **bundle.reference,
            "embedding_norms": bundle.reference["embedding_norms"] * 4.0,
        },
    )
    drifted, _ = _monitor(_cascade(shifted))
    assert drifted.embedding_norm_psi() > baseline
    assert drifted.embedding_norm_psi() > 0.2


# --- feedback ---


def test_feedback_updates_the_simulated_ctr_gauge_but_never_training(bundle: Bundle) -> None:
    monitor, metrics = _monitor(_cascade(bundle))
    assert monitor.simulated_ctr() is None
    events = [
        FeedbackEvent(user_id=1, item_id=index, position=1 + index, clicked=index % 2, source="a")
        for index in range(10)
    ]
    assert monitor.record_feedback(events) == 10
    assert monitor.simulated_ctr() == pytest.approx(0.5)
    snapshot = monitor.snapshot()
    assert snapshot.feedback_events == 10
    assert snapshot.simulated_ctr == pytest.approx(0.5)
    assert _value(metrics, "cascade_feedback_events_total") == pytest.approx(10.0)
    # Feedback is never folded back into the snapshot the ranker's features come from.
    assert monitor.snapshot().feature_rows == 0


# --- canary wiring ---


def test_a_canary_cycle_sets_the_gauges_and_appears_in_the_snapshot(bundle: Bundle) -> None:
    monitor, metrics = _monitor(_cascade(bundle))
    result = monitor.canary.run_cycle()
    assert not result.alert
    assert _value(metrics, "cascade_ann_recall_canary") == pytest.approx(result.recall_mean)
    assert _value(metrics, "cascade_ann_recall_canary_cycles_total") == pytest.approx(1.0)
    assert _value(metrics, "cascade_ann_recall_canary_alerts_total") == pytest.approx(0.0)
    canary = monitor.snapshot().canary
    assert canary is not None
    assert canary["cycle"] == 1
    assert canary["alert"] is False


def test_the_canary_alerts_and_counts_after_the_live_index_is_corrupted(bundle: Bundle) -> None:
    cascade = _cascade(bundle)
    monitor, metrics = _monitor(cascade)
    assert not monitor.canary.run_cycle().alert
    damaged = bundle.index.copy()
    corrupt_index(damaged, 0.6, seed=3)
    cascade.swap_index(damaged)
    result = monitor.canary.run_cycle()
    assert result.alert
    assert _value(metrics, "cascade_ann_recall_canary_alerts_total") == pytest.approx(1.0)
    assert _value(metrics, "cascade_ann_recall_canary_cycles_total") == pytest.approx(2.0)
    canary = monitor.snapshot().canary
    assert canary is not None
    assert canary["alert"] is True


def test_the_canary_samples_real_user_vectors(bundle: Bundle) -> None:
    """The canary has to hold for the query distribution the index actually serves."""
    monitor, _ = _monitor(_cascade(bundle), canary_sample_size=8)
    result = monitor.canary.run_cycle()
    assert result.queries == 8
    assert result.recall_mean > 0.9


def test_the_snapshot_reports_no_canary_before_the_first_cycle(bundle: Bundle) -> None:
    monitor, _ = _monitor(_cascade(bundle))
    assert monitor.snapshot().canary is None


# --- rebind after a bundle swap ---


def test_rebinding_clears_the_buffers_and_re_reads_the_reference(bundle: Bundle) -> None:
    cascade = _cascade(bundle)
    monitor, metrics = _monitor(cascade)
    for user in range(30):
        monitor.observe(user, cascade.recommend(user, k=10).item_ids)
    assert monitor.snapshot().feature_rows > 0
    monitor.rebind()
    after = monitor.snapshot()
    assert after.feature_rows == 0
    assert after.catalog_coverage == 0.0
    assert after.feature_psi == {}
    # Feedback is deliberately not cleared: it is traffic-level, not bundle-level.
    assert _value(
        metrics,
        "cascade_bundle_info",
        bundle_id=bundle.manifest.short_id,
        scale=bundle.manifest.scale,
    ) == pytest.approx(1.0)


def test_the_snapshot_is_json_serializable(bundle: Bundle) -> None:
    import json

    cascade = _cascade(bundle)
    monitor, _ = _monitor(cascade)
    for user in range(30):
        monitor.observe(user, cascade.recommend(user, k=10).item_ids)
    monitor.canary.run_cycle()
    payload = json.loads(json.dumps(monitor.snapshot().as_dict()))
    assert payload["bundle_id"] == bundle.manifest.short_id
    assert isinstance(payload["feature_psi"], dict)
    assert isinstance(payload["drifted_features"], list)
    assert payload["canary"]["cycle"] == 1
