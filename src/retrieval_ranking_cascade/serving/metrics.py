"""Prometheus metrics for the cascade, on the app's own registry (never the global default)."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

__all__ = ["CascadeMetrics", "create_cascade_metrics"]

# Stage latencies span microseconds (rules) to tens of milliseconds (a wide ANN query), so the
# buckets are logarithmic and the top one sits above the 50 ms request budget.
_STAGE_BUCKETS = (
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.25,
)


@dataclass(frozen=True, slots=True)
class CascadeMetrics:
    registry: CollectorRegistry
    stage_latency_seconds: Histogram
    requests_total: Counter
    degraded_responses_total: Counter
    in_flight_requests: Gauge
    candidates_returned: Histogram
    ann_recall_canary: Gauge
    ann_recall_canary_cycles_total: Counter
    ann_recall_canary_alerts_total: Counter
    feature_psi: Gauge
    embedding_norm_drift: Gauge
    catalog_coverage: Gauge
    simulated_ctr: Gauge
    bundle_info: Gauge
    feedback_events_total: Counter
    feedback_rejected_total: Counter
    admin_actions_total: Counter


def create_cascade_metrics(
    registry: CollectorRegistry, namespace: str = "cascade"
) -> CascadeMetrics:
    """Create the cascade metric families on ``registry``."""
    return CascadeMetrics(
        registry=registry,
        stage_latency_seconds=Histogram(
            "stage_latency_seconds",
            "Wall-clock seconds spent in each cascade stage.",
            ["stage"],
            namespace=namespace,
            registry=registry,
            buckets=_STAGE_BUCKETS,
        ),
        requests_total=Counter(
            "requests_total",
            "Recommendation requests served, by degradation state.",
            ["degraded"],
            namespace=namespace,
            registry=registry,
        ),
        degraded_responses_total=Counter(
            "degraded_responses_total",
            "Responses that dropped quality to stay inside the latency budget.",
            ["reason"],
            namespace=namespace,
            registry=registry,
        ),
        in_flight_requests=Gauge(
            "in_flight_requests",
            "Accepted recommendation requests currently being served.",
            namespace=namespace,
            registry=registry,
        ),
        candidates_returned=Histogram(
            "candidates_returned",
            "Candidates the ANN stage returned before ranking.",
            namespace=namespace,
            registry=registry,
            buckets=(1, 16, 64, 128, 256, 500, 1000, 2000),
        ),
        ann_recall_canary=Gauge(
            "ann_recall_canary",
            "Mean recall of the ANN index against exact search in the last canary cycle.",
            namespace=namespace,
            registry=registry,
        ),
        ann_recall_canary_cycles_total=Counter(
            "ann_recall_canary_cycles_total",
            "Completed recall canary cycles.",
            namespace=namespace,
            registry=registry,
        ),
        ann_recall_canary_alerts_total=Counter(
            "ann_recall_canary_alerts_total",
            "Canary cycles whose mean recall fell below the threshold.",
            namespace=namespace,
            registry=registry,
        ),
        feature_psi=Gauge(
            "feature_psi",
            "Population Stability Index of a ranker feature against its training reference.",
            ["feature"],
            namespace=namespace,
            registry=registry,
        ),
        embedding_norm_drift=Gauge(
            "embedding_norm_drift",
            "PSI of item embedding pre-normalization norms against the training reference.",
            namespace=namespace,
            registry=registry,
        ),
        catalog_coverage=Gauge(
            "catalog_coverage",
            "Share of the catalog seen across recent recommendation lists.",
            namespace=namespace,
            registry=registry,
        ),
        simulated_ctr=Gauge(
            "simulated_ctr",
            "Click-through rate reported by the offline feedback simulator.",
            namespace=namespace,
            registry=registry,
        ),
        bundle_info=Gauge(
            "bundle_info",
            "Always 1; the labels carry the serving bundle's identity.",
            ["bundle_id", "scale"],
            namespace=namespace,
            registry=registry,
        ),
        feedback_events_total=Counter(
            "feedback_events_total",
            "Accepted feedback events.",
            namespace=namespace,
            registry=registry,
        ),
        feedback_rejected_total=Counter(
            "feedback_rejected_total",
            "Feedback events rejected by the contract or the per-source rate limit.",
            ["reason"],
            namespace=namespace,
            registry=registry,
        ),
        admin_actions_total=Counter(
            "admin_actions_total",
            "Privileged actions recorded in the audit chain.",
            ["action"],
            namespace=namespace,
            registry=registry,
        ),
    )
