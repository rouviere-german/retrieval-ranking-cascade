"""Offline demo: train a bundle, serve it over HTTP in-process, then break it on purpose.

``uv run python tasks.py demo`` runs this. It needs no network, no API key and no container: it
trains on the ``small`` world (about 15 s on 6 CPU cores), writes a real bundle to disk, loads it
back through the hash verification path, and drives the FastAPI app with ``TestClient``.

The interesting part is the second half. A demo that only shows the happy path proves nothing, so
this one also:

* fires an over-budget request and shows the response coming back flagged ``degraded``;
* asks for a deliberately expensive ``candidate_k``/``efSearch`` pair and shows the cost cap
  refusing it;
* calls an admin route with a non-admin key and shows the 403;
* corrupts half the vectors inside the live index and shows the recall canary catching it in one
  cycle;
* flips one byte in a bundle artifact and shows the swap being refused and audited;
* verifies the admin audit chain, then tampers with it and shows the verification failing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from retrieval_ranking_cascade.core.config import Settings
from retrieval_ranking_cascade.core.security import hash_api_key
from retrieval_ranking_cascade.evaluation.propensity import estimate_eta, estimate_propensities
from retrieval_ranking_cascade.evaluation.simulator import FeedbackSimulator
from retrieval_ranking_cascade.pipeline import TrainingConfig, TrainingResult, train_bundle
from retrieval_ranking_cascade.retrieval.canary import corrupt_index
from retrieval_ranking_cascade.retrieval.two_tower import TowerConfig
from retrieval_ranking_cascade.serving.api import (
    build_service,
    create_cascade_app,
    feedback_sources,
)
from retrieval_ranking_cascade.serving.audit import AuditError, AuditLog
from retrieval_ranking_cascade.serving.budget import BudgetController
from retrieval_ranking_cascade.serving.bundle import load_bundle
from retrieval_ranking_cascade.serving.metrics import create_cascade_metrics

__all__ = ["run_demo"]

_SERVING_KEY = "demo-serving-key-" + "s" * 24
_ADMIN_KEY = "demo-admin-key-" + "a" * 24
_SERVING_AUTH = {"X-API-Key": _SERVING_KEY}
_ADMIN_AUTH = {"X-API-Key": _ADMIN_KEY}


def _settings(workdir: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "api_key_sha256": [hash_api_key(_SERVING_KEY), hash_api_key(_ADMIN_KEY)],
        "admin_api_key_sha256": [hash_api_key(_ADMIN_KEY)],
        "bundle_dir": workdir / "bundle",
        "audit_log_path": workdir / "audit" / "admin.jsonl",
        "rate_limit_per_second": 10_000.0,
        "rate_limit_burst": 10_000,
        "feedback_rate_limit_per_second": 10_000.0,
        "feedback_rate_limit_burst": 10_000,
        "canary_sample_size": 32,
        "canary_k": 100,
        "metrics_require_auth": True,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _train(workdir: Path, scale: str) -> TrainingResult:
    config = TrainingConfig(
        scale=scale,
        threads=4,
        tower=TowerConfig(epochs=6, batch_size=512, uniform_negatives=512),
    )
    result = train_bundle(config, repo_root=Path(__file__).resolve().parents[2])
    result.write(workdir / "bundle")
    return result


def run_demo(  # noqa: PLR0915 - a linear script of scenarios; splitting it hides the order
    workdir: Path, *, scale: str = "small"
) -> dict[str, Any]:
    """Run the whole demo and return a JSON-serializable summary."""
    workdir.mkdir(parents=True, exist_ok=True)
    result = _train(workdir, scale)
    settings = _settings(workdir)
    bundle = load_bundle(settings.bundle_dir)
    metrics = create_cascade_metrics(CollectorRegistry())
    service = build_service(bundle, settings, metrics)
    app = create_cascade_app(service)
    summary: dict[str, Any] = {
        "scale": scale,
        "bundle_id": bundle.manifest.short_id,
        "data_sha256": bundle.manifest.data_sha256[:12],
        "impressions": int(result.metrics["impressions"]),
        "train_seconds": round(sum(result.durations.values()), 1),
        f"recall_at_{result.config.candidate_k}": round(
            result.metrics[f"recall_at_{result.config.candidate_k}"], 4
        ),
    }

    with TestClient(app, raise_server_exceptions=False) as client:
        summary["readyz"] = client.get("/readyz").json()["status"]

        healthy = client.post(
            "/v1/recommend", json={"user_id": 3, "k": 10}, headers=_SERVING_AUTH
        ).json()
        summary["items"] = len(healthy["items"])
        summary["reranked"] = healthy["reranked"]
        summary["degraded"] = healthy["degraded"]
        summary["stage_ms"] = healthy["stage_ms"]
        summary["distinct_categories"] = len({item["category"] for item in healthy["items"]})

        cold = client.post(
            "/v1/recommend", json={"user_id": 10**9, "k": 10}, headers=_SERVING_AUTH
        ).json()
        summary["cold_start_items"] = len(cold["items"])
        summary["cold_start_flagged"] = cold["cold_start"]

        summary["unauthenticated_status"] = client.post(
            "/v1/recommend", json={"user_id": 3}
        ).status_code
        expensive = client.post(
            "/v1/recommend",
            json={"user_id": 3, "candidate_k": 2_000, "ef_search": 2_048},
            headers=_SERVING_AUTH,
        )
        summary["cost_cap_status"] = expensive.status_code
        summary["cost_cap_code"] = expensive.json()["error"]["code"]

        summary["admin_without_scope_status"] = client.post(
            "/v1/admin/reindex", json={}, headers=_SERVING_AUTH
        ).status_code

        # Feedback: the contract rejects an out-of-range item, a good batch is accepted.
        source = feedback_sources(1, seed=1)[0]
        bad = client.post(
            "/v1/feedback",
            json={
                "source": source,
                "events": [
                    {
                        "user_id": 3,
                        "item_id": bundle.n_items + 5,
                        "position": 1,
                        "clicked": 1,
                    }
                ],
            },
            headers=_SERVING_AUTH,
        )
        summary["feedback_contract_status"] = bad.status_code

        simulator = FeedbackSimulator(result.world, seed=7)
        served: list[tuple[int, list[int]]] = []
        for user_id in range(40):
            response = client.post(
                "/v1/recommend", json={"user_id": user_id, "k": 10}, headers=_SERVING_AUTH
            ).json()
            served.append((user_id, [item["item_id"] for item in response["items"]]))
        simulated = simulator.simulate(served)
        accepted = 0
        for user_id, items in served:
            outcome = simulator.simulate_list(user_id, items)
            events = [
                {
                    "user_id": user_id,
                    "item_id": int(item),
                    "position": position,
                    "clicked": int(click),
                }
                for position, (item, click) in enumerate(zip(items, outcome, strict=True), start=1)
            ]
            accepted += client.post(
                "/v1/feedback",
                json={"source": source, "events": events},
                headers=_SERVING_AUTH,
            ).json()["accepted"]
        summary["feedback_accepted"] = accepted
        summary["simulated_ctr"] = round(simulated.ctr, 4)

        # Position-bias propensities, recovered from the randomized slice only.
        propensity = estimate_propensities(
            result.interactions, slots=result.world_config.slots, min_impressions=50
        )
        summary["estimated_eta"] = round(estimate_eta(propensity), 3)
        summary["true_eta"] = result.world_config.examination_eta

        healthy_canary = client.post("/v1/admin/canary", headers=_ADMIN_AUTH).json()
        summary["canary_recall_healthy"] = round(healthy_canary["recall_mean"], 4)
        summary["canary_alert_healthy"] = healthy_canary["alert"]

        # Budget pressure: a 1 ms budget cannot fit the reranker, so the response degrades.
        tight = build_service(
            bundle,
            settings,
            create_cascade_metrics(CollectorRegistry()),
            controller=BudgetController(
                total_seconds=0.001, retrieval_seconds=0.0005, ranking_seconds=0.0005
            ),
        )
        tight_result = tight.cascade.recommend(3, k=10)
        summary["tight_budget_degraded"] = tight_result.degraded
        summary["tight_budget_reasons"] = list(tight_result.degraded_reasons)
        summary["tight_budget_items"] = len(tight_result.items)

        # Corrupt the live index: the canary must alert on the very next cycle.
        corrupted = corrupt_index(service.cascade.index, 0.5, seed=3)
        broken = client.post("/v1/admin/canary", headers=_ADMIN_AUTH).json()
        summary["corrupted_vectors"] = corrupted
        summary["canary_recall_corrupted"] = round(broken["recall_mean"], 4)
        summary["canary_alert_corrupted"] = broken["alert"]

        # A reindex from the (uncorrupted) item vectors restores recall.
        client.post("/v1/admin/reindex", json={}, headers=_ADMIN_AUTH)
        restored = client.post("/v1/admin/canary", headers=_ADMIN_AUTH).json()
        summary["canary_recall_after_reindex"] = round(restored["recall_mean"], 4)
        summary["canary_alert_after_reindex"] = restored["alert"]

        # Tamper with a bundle artifact: the swap is refused and the refusal is audited.
        tampered_dir = workdir / "tampered"
        tampered_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(p.name for p in settings.bundle_dir.iterdir()):
            (tampered_dir / name).write_bytes((settings.bundle_dir / name).read_bytes())
        ranker_path = tampered_dir / "ranker.txt"
        ranker_path.write_bytes(ranker_path.read_bytes().replace(b"leaf_value", b"leaf_valve", 1))
        rejected = client.post(
            "/v1/admin/bundle", json={"directory": str(tampered_dir)}, headers=_ADMIN_AUTH
        )
        summary["tampered_bundle_status"] = rejected.status_code
        summary["tampered_bundle_code"] = rejected.json()["error"]["code"]

        good_swap = client.post(
            "/v1/admin/bundle", json={"directory": str(settings.bundle_dir)}, headers=_ADMIN_AUTH
        )
        summary["bundle_swap_status"] = good_swap.status_code

        # A swap resets the monitoring buffers, so replay some traffic before reading them.
        for user_id in range(60):
            client.post("/v1/recommend", json={"user_id": user_id, "k": 10}, headers=_SERVING_AUTH)
        monitoring = client.get("/v1/monitoring", headers=_SERVING_AUTH).json()
        summary["catalog_coverage"] = monitoring["catalog_coverage"]
        summary["embedding_norm_psi"] = monitoring["embedding_norm_psi"]
        summary["drifted_features"] = monitoring["drifted_features"]

        exposition = client.get("/metrics", headers=_SERVING_AUTH).text
        summary["metrics_exposed"] = sorted(
            name
            for name in (
                "cascade_stage_latency_seconds",
                "cascade_degraded_responses_total",
                "cascade_ann_recall_canary",
                "cascade_feature_psi",
                "cascade_embedding_norm_drift",
                "cascade_catalog_coverage",
            )
            if name in exposition
        )

    audit = AuditLog(settings.audit_log_path)
    summary["audit_entries"] = audit.verify()
    summary["audit_actions"] = [entry.action for entry in audit.read()]
    summary["audit_head"] = audit.head[:12]
    # Rewrite one entry's actor but keep its recorded hash: the chain must reject it.
    tampered_audit = workdir / "audit" / "tampered.jsonl"
    lines = settings.audit_log_path.read_text(encoding="utf-8").splitlines()
    forged = json.loads(lines[-1])
    forged["actor"] = "attacker0000"
    tampered_audit.write_text(
        "\n".join([*lines[:-1], json.dumps(forged, sort_keys=True, separators=(",", ":"))]) + "\n",
        encoding="utf-8",
    )
    try:
        AuditLog(tampered_audit).verify()
    except AuditError:
        summary["tampered_audit_detected"] = True
    else:  # pragma: no cover - only reachable if the chain stops working
        summary["tampered_audit_detected"] = False
    return summary
