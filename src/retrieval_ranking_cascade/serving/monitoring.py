"""Live monitoring: feature drift, embedding-norm drift, the recall canary and coverage.

Three things silently break a served cascade, and none of them raises an exception:

* **feature drift** - the snapshot ages, activity levels move, and features the ranker was fitted
  on no longer look the way they did. PSI over the bundle's stored training reference bounds this.
* **embedding-norm drift** - a bundle swap that changed the item tower moves the pre-normalization
  norms even though the unit-norm outputs still look fine. It is the cheapest early signal that
  "the same" model is no longer the same.
* **ANN recall decay** - a bad rebuild, a wrong ``efSearch`` or partially corrupted vectors still
  return plausible neighbours. Only a comparison against exact search finds it, which is what the
  canary does on a sample of live user vectors.

Sampling. Recomputing feature rows for every response would double the ranker's cost, so the
monitor keeps every ``sample_every``-th response's rows in a bounded ring buffer and reports PSI
once the buffer holds at least ``min_samples`` rows. Buffers are bounded and never grow with
traffic, which matters more here than statistical purity.

Feedback ingestion is deliberately *not* wired into training. Accepted events update the
simulated-CTR gauge and the coverage gauge only; retraining from live feedback without a fresh
randomized slice would bake the current policy's position bias into the next model.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from retrieval_ranking_cascade.core.logging import get_logger
from retrieval_ranking_cascade.ml.drift import DriftMonitor, DriftReport, psi
from retrieval_ranking_cascade.ranking.features import FEATURE_NAMES, assemble_features
from retrieval_ranking_cascade.retrieval.canary import CanaryResult, RecallCanary
from retrieval_ranking_cascade.serving.bundle import Bundle
from retrieval_ranking_cascade.serving.cascade import Cascade
from retrieval_ranking_cascade.serving.metrics import CascadeMetrics

__all__ = ["CascadeMonitor", "FeedbackEvent", "MonitoringSnapshot"]

_log = get_logger("retrieval_ranking_cascade.monitoring")


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    """One validated impression outcome reported by a client."""

    user_id: int
    item_id: int
    position: int
    clicked: int
    source: str


@dataclass(frozen=True, slots=True)
class MonitoringSnapshot:
    """What ``/v1/monitoring`` returns and what the metrics gauges are set from."""

    bundle_id: str
    feature_psi: dict[str, float]
    drifted_features: tuple[str, ...]
    embedding_norm_psi: float
    feature_rows: int
    catalog_coverage: float
    simulated_ctr: float | None
    feedback_events: int
    canary: dict[str, float | bool | int] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "feature_psi": {k: round(v, 6) for k, v in self.feature_psi.items()},
            "drifted_features": list(self.drifted_features),
            "embedding_norm_psi": round(self.embedding_norm_psi, 6),
            "feature_rows": self.feature_rows,
            "catalog_coverage": round(self.catalog_coverage, 6),
            "simulated_ctr": None if self.simulated_ctr is None else round(self.simulated_ctr, 6),
            "feedback_events": self.feedback_events,
            "canary": self.canary,
        }


class CascadeMonitor:
    """Owns the drift monitors, the recall canary and the recent-traffic buffers."""

    def __init__(
        self,
        cascade: Cascade,
        metrics: CascadeMetrics,
        *,
        canary_k: int = 100,
        canary_sample_size: int = 32,
        canary_threshold: float = 0.9,
        psi_threshold: float = 0.2,
        sample_every: int = 20,
        buffer_rows: int = 4_000,
        min_samples: int = 200,
        seed: int = 0,
    ) -> None:
        if sample_every < 1 or buffer_rows < 1 or min_samples < 1:
            raise ValueError("sample_every, buffer_rows and min_samples must be >= 1")
        self._cascade = cascade
        self._metrics = metrics
        self._psi_threshold = psi_threshold
        self._sample_every = sample_every
        self._min_samples = min_samples
        self._requests = 0
        self._lock = threading.Lock()
        self._features: deque[npt.NDArray[np.float64]] = deque(maxlen=buffer_rows)
        self._served_items: deque[int] = deque(maxlen=buffer_rows)
        self._feedback: deque[FeedbackEvent] = deque(maxlen=buffer_rows)
        self._rng = np.random.default_rng(seed)
        bundle = cascade.bundle
        self._drift = self._build_drift_monitor(bundle, psi_threshold)
        self._norm_reference = np.asarray(
            bundle.reference.get("embedding_norms", np.ones(1)), dtype=np.float64
        )
        self.canary = RecallCanary(
            index_getter=lambda: cascade.index,
            vectors_getter=cascade.item_vectors,
            sampler=self._sample_user_vectors,
            k=canary_k,
            sample_size=canary_sample_size,
            threshold=canary_threshold,
            on_result=self._record_canary,
            seed=seed,
        )
        metrics.bundle_info.labels(
            bundle_id=bundle.manifest.short_id, scale=bundle.manifest.scale
        ).set(1.0)

    @staticmethod
    def _build_drift_monitor(bundle: Bundle, psi_threshold: float) -> DriftMonitor:
        reference = {
            name: bundle.reference[f"feature_{name}"]
            for name in FEATURE_NAMES
            if f"feature_{name}" in bundle.reference
        }
        if not reference:
            raise ValueError("bundle reference has no ranker feature distributions")
        return DriftMonitor(reference, psi_threshold=psi_threshold)

    def _sample_user_vectors(self, rng: np.random.Generator, size: int) -> npt.NDArray[np.float32]:
        """Sample real user vectors for the canary (the query distribution it must hold for)."""
        bundle = self._cascade.bundle
        users = rng.integers(0, bundle.n_users, size=min(size, bundle.n_users))
        return bundle.user_vectors(users)

    def _record_canary(self, result: CanaryResult) -> None:
        self._metrics.ann_recall_canary.set(result.recall_mean)
        self._metrics.ann_recall_canary_cycles_total.inc()
        if result.alert:
            self._metrics.ann_recall_canary_alerts_total.inc()
            _log.warning(
                "ann_recall_canary_alert",
                recall_mean=round(result.recall_mean, 4),
                threshold=result.threshold,
                cycle=result.cycle,
                k=result.k,
            )

    # --- request observation ---

    def observe(self, user_id: int, item_ids: Sequence[int]) -> None:
        """Record one response. Feature rows are recomputed for a sampled subset only."""
        if not item_ids:
            return
        with self._lock:
            self._requests += 1
            self._served_items.extend(int(item) for item in item_ids)
            sample = self._requests % self._sample_every == 0
        if not sample:
            return
        bundle = self._cascade.bundle
        if not 0 <= user_id < bundle.n_users:
            return
        items = np.asarray(item_ids, dtype=np.int64)
        user_vector = bundle.user_vectors(np.asarray([user_id]))
        tower_scores = (bundle.item_vectors[items] @ user_vector[0]).astype(np.float32)
        rows = assemble_features(bundle.snapshot, user_id, items, tower_scores)
        with self._lock:
            self._features.extend(np.asarray(row, dtype=np.float64) for row in rows)

    def record_feedback(self, events: Sequence[FeedbackEvent]) -> int:
        with self._lock:
            self._feedback.extend(events)
        self._metrics.feedback_events_total.inc(len(events))
        return len(events)

    # --- reports ---

    def drift_report(self) -> DriftReport | None:
        """PSI/KS report over the sampled feature rows, or None below ``min_samples``."""
        with self._lock:
            rows = list(self._features)
        if len(rows) < self._min_samples:
            return None
        matrix = np.vstack(rows)
        tracked = set(self._drift.features)
        current = {
            name: matrix[:, column] for column, name in enumerate(FEATURE_NAMES) if name in tracked
        }
        return self._drift.check(current)

    def embedding_norm_psi(self) -> float:
        """PSI of the live bundle's item-embedding norms against the training reference."""
        bundle = self._cascade.bundle
        _, norms = bundle.model.item.embed(np.arange(bundle.n_items), bundle.item_side)
        return psi(self._norm_reference, np.asarray(norms, dtype=np.float64))

    def simulated_ctr(self) -> float | None:
        with self._lock:
            events = list(self._feedback)
        if not events:
            return None
        return float(np.mean([event.clicked for event in events]))

    def snapshot(self) -> MonitoringSnapshot:
        """Compute every monitor, update the gauges and return the values."""
        bundle = self._cascade.bundle
        report = self.drift_report()
        feature_psi: dict[str, float] = {}
        drifted: tuple[str, ...] = ()
        if report is not None:
            feature_psi = {item.feature: item.psi for item in report.features}
            drifted = report.drifted_features
            for name, value in feature_psi.items():
                self._metrics.feature_psi.labels(feature=name).set(value)
            if drifted:
                _log.warning("feature_drift_detected", features=list(drifted))
        norm_psi = self.embedding_norm_psi()
        self._metrics.embedding_norm_drift.set(norm_psi)
        with self._lock:
            served = list(self._served_items)
            rows = len(self._features)
            feedback = len(self._feedback)
        coverage = float(len(set(served)) / bundle.n_items) if served else 0.0
        self._metrics.catalog_coverage.set(coverage)
        ctr = self.simulated_ctr()
        if ctr is not None:
            self._metrics.simulated_ctr.set(ctr)
        last = self.canary.last
        canary: dict[str, float | bool | int] | None = None
        if last is not None:
            canary = {
                "cycle": last.cycle,
                "recall_mean": round(last.recall_mean, 6),
                "recall_min": round(last.recall_min, 6),
                "queries": last.queries,
                "k": last.k,
                "ef_search": last.ef_search,
                "threshold": last.threshold,
                "alert": last.alert,
            }
        return MonitoringSnapshot(
            bundle_id=bundle.manifest.short_id,
            feature_psi=feature_psi,
            drifted_features=drifted,
            embedding_norm_psi=norm_psi,
            feature_rows=rows,
            catalog_coverage=coverage,
            simulated_ctr=ctr,
            feedback_events=feedback,
            canary=canary,
        )

    def rebind(self) -> None:
        """Re-read the reference distributions after a bundle swap."""
        bundle = self._cascade.bundle
        self._drift = self._build_drift_monitor(bundle, self._psi_threshold)
        self._norm_reference = np.asarray(
            bundle.reference.get("embedding_norms", np.ones(1)), dtype=np.float64
        )
        with self._lock:
            self._features.clear()
            self._served_items.clear()
        self._metrics.bundle_info.labels(
            bundle_id=bundle.manifest.short_id, scale=bundle.manifest.scale
        ).set(1.0)
