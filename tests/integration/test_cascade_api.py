"""The HTTP surface: authentication, input bounds, cost caps, admin scope, feedback, monitoring."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from retrieval_ranking_cascade.core.config import Settings
from retrieval_ranking_cascade.core.security import hash_api_key
from retrieval_ranking_cascade.retrieval.canary import corrupt_index
from retrieval_ranking_cascade.serving.api import (
    CascadeService,
    InFlightTracker,
    build_service,
    create_cascade_app,
    feedback_sources,
    query_cost,
)
from retrieval_ranking_cascade.serving.audit import AuditLog
from retrieval_ranking_cascade.serving.budget import BudgetController, CostEstimator
from retrieval_ranking_cascade.serving.bundle import Bundle, load_bundle
from retrieval_ranking_cascade.serving.metrics import create_cascade_metrics

pytestmark = pytest.mark.integration

SERVING_KEY = "test-serving-key-" + "s" * 24
ADMIN_KEY = "test-admin-key-" + "a" * 26
OTHER_KEY = "test-other-key-" + "o" * 26
SERVING = {"X-API-Key": SERVING_KEY}
ADMIN = {"X-API-Key": ADMIN_KEY}


def _settings(tmp_path: Path, bundle_dir: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "api_key_sha256": [
            hash_api_key(SERVING_KEY),
            hash_api_key(ADMIN_KEY),
            hash_api_key(OTHER_KEY),
        ],
        "admin_api_key_sha256": [hash_api_key(ADMIN_KEY)],
        "bundle_dir": bundle_dir,
        # The shared pytest basetemp holds both the session bundle and every per-test copy of it,
        # so this is the smallest root that covers what these tests legitimately load.
        "bundle_root": tmp_path.parent,
        "audit_log_path": tmp_path / "audit" / "admin.jsonl",
        "rate_limit_per_second": 10_000.0,
        "rate_limit_burst": 10_000,
        "feedback_rate_limit_per_second": 10_000.0,
        "feedback_rate_limit_burst": 10_000,
        "candidate_k": 150,
        "rerank_depth": 80,
        "default_ef_search": 256,
        "max_ef_search": 1_024,
        "max_candidate_k": 500,
        "max_query_cost": 8_000.0,
        "request_budget_ms": 500.0,
        "retrieval_budget_ms": 200.0,
        "ranking_budget_ms": 200.0,
        "canary_k": 20,
        "canary_sample_size": 16,
        "metrics_require_auth": True,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def service(tiny_bundle_dir: Path, tmp_path: Path) -> CascadeService:
    settings = _settings(tmp_path, tiny_bundle_dir)
    bundle = load_bundle(tiny_bundle_dir)
    return build_service(bundle, settings, create_cascade_metrics(CollectorRegistry()))


@pytest.fixture
def client(service: CascadeService) -> Iterator[TestClient]:
    with TestClient(create_cascade_app(service), raise_server_exceptions=False) as test_client:
        yield test_client


# --- the cost model ---


def test_the_query_cost_grows_with_both_knobs_and_validates_them() -> None:
    assert query_cost(500, 768) > query_cost(500, 256)
    assert query_cost(1_000, 256) > query_cost(500, 256)
    # A small efSearch is priced at the candidate count, because the graph visits max(ef, k) nodes.
    assert query_cost(500, 100) == pytest.approx(query_cost(500, 500))
    for bad in ((0, 10), (10, 0)):
        with pytest.raises(ValueError, match="must be >= 1"):
            query_cost(*bad)


# --- health and readiness ---


def test_healthz_and_readyz_need_no_key(client: TestClient) -> None:
    assert client.get("/healthz").status_code == 200
    ready = client.get("/readyz")
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready"
    assert body["checks"]["index"] == "ok"


def test_readyz_fails_while_the_index_does_not_match_the_bundle(
    service: CascadeService, client: TestClient
) -> None:
    from retrieval_ranking_cascade.retrieval.ann import HnswIndex, HnswParams

    short = HnswIndex.build(
        service.cascade.item_vectors()[:10], HnswParams(m=8, ef_construction=32, ef_search=32)
    )
    service.cascade.swap_index(short)
    assert not service.ready()
    assert client.get("/readyz").status_code == 503


# --- authentication ---


@pytest.mark.parametrize(
    "path", ["/v1/recommend", "/v1/feedback", "/v1/admin/reindex", "/v1/admin/bundle"]
)
def test_every_post_route_requires_a_key(client: TestClient, path: str) -> None:
    assert client.post(path, json={}).status_code == 401


@pytest.mark.parametrize("path", ["/v1/bundle", "/v1/monitoring", "/metrics"])
def test_every_get_route_requires_a_key(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401


def test_an_unknown_key_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/recommend", json={"user_id": 1}, headers={"X-API-Key": "not-a-real-key"}
    )
    assert response.status_code == 401


# --- recommendations ---


def test_a_recommendation_carries_its_provenance_and_stage_timings(client: TestClient) -> None:
    response = client.post("/v1/recommend", json={"user_id": 3, "k": 10}, headers=SERVING)
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 10
    assert [item["rank"] for item in body["items"]] == list(range(1, 11))
    assert len(body["bundle_id"]) == 64
    assert body["reranked"] is True
    assert body["degraded"] is False
    assert body["cold_start"] is False
    assert body["ef_search"] == 256
    assert body["in_flight"] >= 1
    assert body["contention"] >= 1.0
    assert body["queued_ms"] >= 0.0
    assert set(body["stage_ms"]) == {"retrieval", "ranking", "rules"}


def test_an_unknown_user_is_answered_from_popularity_and_flagged(client: TestClient) -> None:
    body = client.post("/v1/recommend", json={"user_id": 10**9}, headers=SERVING).json()
    assert body["cold_start"] is True
    assert body["items"]
    assert body["ef_search"] == 0


def test_defaults_are_applied_when_the_body_omits_them(client: TestClient) -> None:
    body = client.post("/v1/recommend", json={"user_id": 2}, headers=SERVING).json()
    assert len(body["items"]) <= 20
    assert body["ef_search"] == 256


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"user_id": 1, "k": 1_000}, "k_too_large"),
        ({"user_id": 1, "candidate_k": 10_000}, "candidate_k_too_large"),
        ({"user_id": 1, "ef_search": 4_096}, "ef_search_too_large"),
        ({"user_id": 1, "k": 60, "candidate_k": 20}, "k_above_candidates"),
        ({"user_id": 1, "candidate_k": 500, "ef_search": 1_024}, "query_too_expensive"),
    ],
)
def test_out_of_bounds_parameters_are_refused_with_a_specific_code(
    client: TestClient, payload: dict[str, int], code: str
) -> None:
    response = client.post("/v1/recommend", json=payload, headers=SERVING)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": -1},
        {"user_id": 1, "k": 0},
        {"user_id": 1, "diversity_lambda": 1.5},
        {"user_id": 1, "surprise": 1},
        {},
    ],
)
def test_the_request_schema_rejects_malformed_bodies(
    client: TestClient, payload: dict[str, Any]
) -> None:
    assert client.post("/v1/recommend", json=payload, headers=SERVING).status_code == 422


def test_the_cost_cap_refuses_a_pair_that_is_individually_legal(
    service: CascadeService, client: TestClient
) -> None:
    """A caller with a valid key must not be able to buy a hundred requests' worth of ANN work."""
    bounds = service.bounds
    # Both pairs use a candidate_k and an efSearch inside their own maxima; only the product bites.
    assert bounds.max_candidate_k >= 300
    assert bounds.max_ef_search == 1_024
    assert query_cost(300, 256) <= bounds.max_query_cost
    assert query_cost(300, bounds.max_ef_search) > bounds.max_query_cost
    legal = client.post(
        "/v1/recommend", json={"user_id": 1, "candidate_k": 300, "ef_search": 256}, headers=SERVING
    )
    assert legal.status_code == 200
    expensive = client.post(
        "/v1/recommend",
        json={"user_id": 1, "candidate_k": 300, "ef_search": 1_024},
        headers=SERVING,
    )
    assert expensive.status_code == 422
    assert expensive.json()["error"]["code"] == "query_too_expensive"


def test_a_degraded_response_is_labelled_and_counted(tiny_bundle_dir: Path, tmp_path: Path) -> None:
    settings = _settings(tmp_path, tiny_bundle_dir)
    metrics = create_cascade_metrics(CollectorRegistry())
    service = build_service(
        load_bundle(tiny_bundle_dir),
        settings,
        metrics,
        controller=BudgetController(
            total_seconds=0.002,
            retrieval_seconds=0.001,
            ranking_seconds=0.001,
            estimators={
                "retrieval": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1e-9),
                "ranking": CostEstimator(fixed_seconds=0.0, seconds_per_unit=1.0),
            },
        ),
    )
    with TestClient(create_cascade_app(service), raise_server_exceptions=False) as client:
        body = client.post("/v1/recommend", json={"user_id": 3, "k": 10}, headers=SERVING).json()
    assert body["degraded"] is True
    assert "skip_rerank" in body["degraded_reasons"]
    assert body["reranked"] is False
    assert body["items"]
    assert metrics.registry.get_sample_value(
        "cascade_degraded_responses_total", {"reason": "skip_rerank"}
    ) == pytest.approx(1.0)
    assert metrics.registry.get_sample_value(
        "cascade_requests_total", {"degraded": "true"}
    ) == pytest.approx(1.0)


def test_user_ids_never_reach_the_logs_in_the_clear(service: CascadeService) -> None:
    pseudonym = service.pseudonym(123_456)
    assert pseudonym != "123456"
    assert len(pseudonym) == 12
    assert service.pseudonym(123_456) == pseudonym


# --- feedback ---


def test_a_valid_feedback_batch_is_accepted_and_counted(client: TestClient) -> None:
    source = feedback_sources(1, seed=2)[0]
    response = client.post(
        "/v1/feedback",
        json={
            "source": source,
            "events": [
                {"user_id": 1, "item_id": 2, "position": 1, "clicked": 1},
                {"user_id": 1, "item_id": 3, "position": 2, "clicked": 0},
            ],
        },
        headers=SERVING,
    )
    assert response.status_code == 200
    assert response.json() == {"accepted": 2, "total_events": 2}


def test_feedback_that_violates_the_contract_is_refused(
    service: CascadeService, client: TestClient
) -> None:
    source = feedback_sources(1, seed=3)[0]
    response = client.post(
        "/v1/feedback",
        json={
            "source": source,
            "events": [
                {
                    "user_id": 1,
                    "item_id": service.cascade.bundle.n_items + 10,
                    "position": 1,
                    "clicked": 1,
                }
            ],
        },
        headers=SERVING,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "feedback_contract_violation"
    assert service.metrics.registry.get_sample_value(
        "cascade_feedback_rejected_total", {"reason": "contract"}
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "source": "not-hex",
            "events": [{"user_id": 1, "item_id": 1, "position": 1, "clicked": 1}],
        },
        {"source": "0123456789ab", "events": []},
        {"source": "0123456789ab", "events": [{"user_id": 1, "item_id": 1, "position": 0}]},
    ],
)
def test_the_feedback_schema_rejects_malformed_bodies(
    client: TestClient, payload: dict[str, Any]
) -> None:
    assert client.post("/v1/feedback", json=payload, headers=SERVING).status_code == 422


def test_one_noisy_source_cannot_flood_the_statistics(
    tiny_bundle_dir: Path, tmp_path: Path
) -> None:
    """A noisy ``source`` is throttled without taking a quiet one's budget with it."""
    settings = _settings(
        tmp_path,
        tiny_bundle_dir,
        feedback_rate_limit_per_second=1.0,
        feedback_rate_limit_burst=2,
    )
    metrics = create_cascade_metrics(CollectorRegistry())
    service = build_service(load_bundle(tiny_bundle_dir), settings, metrics)
    noisy, quiet = feedback_sources(2, seed=4)
    body = {"events": [{"user_id": 1, "item_id": 2, "position": 1, "clicked": 1}]}
    with TestClient(create_cascade_app(service), raise_server_exceptions=False) as client:
        codes = [
            client.post("/v1/feedback", json={**body, "source": noisy}, headers=SERVING).status_code
            for _ in range(6)
        ]
        # A different source has its own bucket and is unaffected.
        assert (
            client.post("/v1/feedback", json={**body, "source": quiet}, headers=SERVING).status_code
            == 200
        )
    assert codes[:2] == [200, 200]
    assert 429 in codes
    rejected = metrics.registry.get_sample_value(
        "cascade_feedback_rejected_total", {"reason": "rate_limited"}
    )
    assert rejected is not None
    assert rejected >= 1.0


def test_a_rate_limited_source_is_told_when_to_retry(tiny_bundle_dir: Path, tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        tiny_bundle_dir,
        feedback_rate_limit_per_second=1.0,
        feedback_rate_limit_burst=1,
    )
    service = build_service(
        load_bundle(tiny_bundle_dir), settings, create_cascade_metrics(CollectorRegistry())
    )
    source = feedback_sources(1, seed=5)[0]
    body = {
        "source": source,
        "events": [{"user_id": 1, "item_id": 2, "position": 1, "clicked": 1}],
    }
    with TestClient(create_cascade_app(service), raise_server_exceptions=False) as client:
        client.post("/v1/feedback", json=body, headers=SERVING)
        limited = client.post("/v1/feedback", json=body, headers=SERVING)
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


def test_rotating_the_source_does_not_buy_a_client_more_feedback_budget(
    tiny_bundle_dir: Path, tmp_path: Path
) -> None:
    """``source`` is chosen by the caller, so the bound on a client is keyed on its API key."""
    settings = _settings(
        tmp_path,
        tiny_bundle_dir,
        feedback_rate_limit_per_second=1.0,
        feedback_rate_limit_burst=1,
        feedback_client_rate_multiplier=4.0,
    )
    metrics = create_cascade_metrics(CollectorRegistry())
    service = build_service(load_bundle(tiny_bundle_dir), settings, metrics)
    events = [{"user_id": 1, "item_id": 2, "position": 1, "clicked": 1}]
    with TestClient(create_cascade_app(service), raise_server_exceptions=False) as client:
        codes = [
            client.post(
                "/v1/feedback", json={"source": source, "events": events}, headers=SERVING
            ).status_code
            for source in feedback_sources(20, seed=6)
        ]
    assert codes[0] == 200
    # Four tokens of per-key burst at 4/s: a fresh source cannot mint a fresh allowance.
    assert codes.count(200) <= 8
    assert codes.count(429) >= 10
    assert metrics.registry.get_sample_value(
        "cascade_feedback_rejected_total", {"reason": "client_rate_limited"}
    ) == pytest.approx(float(codes.count(429)))


def test_two_api_keys_do_not_share_one_feedback_budget(
    tiny_bundle_dir: Path, tmp_path: Path
) -> None:
    """The per-key bucket must bound a client without letting one client starve another."""
    settings = _settings(
        tmp_path,
        tiny_bundle_dir,
        feedback_rate_limit_per_second=1.0,
        feedback_rate_limit_burst=1,
        feedback_client_rate_multiplier=1.0,
    )
    service = build_service(
        load_bundle(tiny_bundle_dir), settings, create_cascade_metrics(CollectorRegistry())
    )
    source = feedback_sources(1, seed=7)[0]
    body = {"source": source, "events": [{"user_id": 1, "item_id": 2, "position": 1, "clicked": 1}]}
    with TestClient(create_cascade_app(service), raise_server_exceptions=False) as client:
        first = client.post("/v1/feedback", json=body, headers=SERVING).status_code
        repeat = client.post("/v1/feedback", json=body, headers=SERVING).status_code
        other = client.post("/v1/feedback", json=body, headers={"X-API-Key": OTHER_KEY}).status_code
    assert (first, repeat, other) == (200, 429, 200)


# --- introspection ---


def test_the_bundle_route_exposes_the_manifest_and_the_live_index_digest(
    client: TestClient,
) -> None:
    body = client.get("/v1/bundle", headers=SERVING).json()
    assert len(body["bundle_id"]) == 64
    assert len(body["live_index_sha256"]) == 64
    assert body["scale"] == "tiny"
    assert body["lockfile_sha256"]
    assert body["metrics"]
    assert body["audit_head"] == "0" * 64


def test_the_monitoring_route_reports_drift_coverage_and_the_canary(client: TestClient) -> None:
    for user in range(40):
        client.post("/v1/recommend", json={"user_id": user, "k": 10}, headers=SERVING)
    client.post("/v1/admin/canary", headers=ADMIN)
    body = client.get("/v1/monitoring", headers=SERVING).json()
    assert body["feature_rows"] > 0
    assert body["catalog_coverage"] > 0.0
    assert body["embedding_norm_psi"] == pytest.approx(0.0, abs=1e-6)
    assert body["canary"]["cycle"] == 1
    assert body["canary"]["alert"] is False


def test_the_metrics_endpoint_carries_both_http_and_cascade_families(client: TestClient) -> None:
    client.post("/v1/recommend", json={"user_id": 1, "k": 5}, headers=SERVING)
    text = client.get("/metrics", headers=SERVING).text
    for family in (
        "cascade_stage_latency_seconds",
        "cascade_requests_total",
        "cascade_candidates_returned",
        "cascade_in_flight_requests",
        "cascade_bundle_info",
        "http_requests_total",
    ):
        assert family in text


# --- admin scope ---


@pytest.mark.parametrize("path", ["/v1/admin/reindex", "/v1/admin/bundle", "/v1/admin/canary"])
def test_a_serving_key_cannot_reach_an_admin_route(client: TestClient, path: str) -> None:
    response = client.post(path, json={"directory": "."}, headers=SERVING)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_reindex_swaps_the_index_and_records_the_new_digest_in_the_chain(
    service: CascadeService, client: TestClient
) -> None:
    before = client.get("/v1/bundle", headers=SERVING).json()["live_index_sha256"]
    response = client.post(
        "/v1/admin/reindex", json={"m": 16, "ef_construction": 64, "ef_search": 128}, headers=ADMIN
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "reindex"
    assert body["index_sha256"] != before
    assert body["audit_seq"] == 0
    after = client.get("/v1/bundle", headers=SERVING).json()
    assert after["live_index_sha256"] == body["index_sha256"]
    # The bundle on disk is untouched: its manifest still describes the index it shipped with.
    assert after["bundle_id"] == body["bundle_id"]
    entries = AuditLog(service.settings.audit_log_path).read()
    assert [entry.action for entry in entries] == ["reindex"]
    assert entries[0].details["index_sha256"] == body["index_sha256"]


def test_a_reindexed_index_still_passes_the_recall_canary(client: TestClient) -> None:
    client.post(
        "/v1/admin/reindex", json={"m": 32, "ef_construction": 200, "ef_search": 512}, headers=ADMIN
    )
    canary = client.post("/v1/admin/canary", headers=ADMIN).json()
    assert canary["alert"] is False
    assert canary["recall_mean"] > 0.9


def test_a_corrupted_live_index_makes_the_canary_alert_on_the_next_cycle(
    service: CascadeService, client: TestClient
) -> None:
    assert client.post("/v1/admin/canary", headers=ADMIN).json()["alert"] is False
    damaged = service.cascade.index.copy()
    corrupt_index(damaged, 0.6, seed=7)
    service.cascade.swap_index(damaged)
    alerted = client.post("/v1/admin/canary", headers=ADMIN).json()
    assert alerted["alert"] is True
    assert alerted["cycle"] == 2
    assert alerted["recall_mean"] < alerted["threshold"]


@pytest.mark.parametrize(
    "payload", [{"m": 2}, {"m": 200}, {"ef_construction": 1}, {"ef_search": 99_999}]
)
def test_reindex_parameters_are_bounded(client: TestClient, payload: dict[str, int]) -> None:
    assert client.post("/v1/admin/reindex", json=payload, headers=ADMIN).status_code == 422


def test_swapping_in_a_verified_bundle_is_accepted_and_audited(
    service: CascadeService, client: TestClient, tiny_bundle_dir: Path
) -> None:
    response = client.post(
        "/v1/admin/bundle", json={"directory": str(tiny_bundle_dir)}, headers=ADMIN
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "bundle_swap"
    assert body["bundle_id"] == service.cascade.bundle.bundle_id
    entries = AuditLog(service.settings.audit_log_path).read()
    assert entries[-1].action == "bundle_swap"
    assert entries[-1].details["previous_bundle_id"] == body["bundle_id"]


def test_swapping_in_a_tampered_bundle_is_refused_and_the_refusal_is_audited(
    service: CascadeService, client: TestClient, tiny_bundle_dir: Path, tmp_path: Path
) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(tiny_bundle_dir, tampered)
    path = tampered / "ranker.txt"
    path.write_bytes(path.read_bytes().replace(b"leaf_value", b"leaf_valve", 1))
    before = service.cascade.bundle.bundle_id
    response = client.post("/v1/admin/bundle", json={"directory": str(tampered)}, headers=ADMIN)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "bundle_rejected"
    # The live bundle is unchanged, and the attempt is on the record.
    assert service.cascade.bundle.bundle_id == before
    entries = AuditLog(service.settings.audit_log_path).read()
    assert entries[-1].action == "bundle_swap_rejected"
    assert entries[-1].details["error"] == "BundleIntegrityError"


def test_a_bundle_directory_outside_the_root_is_refused_before_anything_is_loaded(
    tiny_bundle_dir: Path, tmp_path: Path
) -> None:
    """A leaked admin key must not turn the loader into an arbitrary-path read probe."""

    def never_called(directory: Path) -> Bundle:
        raise AssertionError(f"load_bundle must not be reached for {directory}")

    settings = _settings(tmp_path, tiny_bundle_dir, bundle_root=tmp_path / "allowed")
    metrics = create_cascade_metrics(CollectorRegistry())
    service = build_service(
        load_bundle(tiny_bundle_dir), settings, metrics, bundle_loader=never_called
    )
    before = service.cascade.bundle.bundle_id
    with TestClient(create_cascade_app(service), raise_server_exceptions=False) as client:
        for outside in ("../../../../etc/passwd", str(tmp_path / "elsewhere"), "C:/Windows"):
            response = client.post("/v1/admin/bundle", json={"directory": outside}, headers=ADMIN)
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "bundle_rejected"
    assert service.cascade.bundle.bundle_id == before
    entries = AuditLog(settings.audit_log_path).read()
    assert [entry.action for entry in entries] == ["bundle_swap_rejected"] * 3
    assert {entry.details["error"] for entry in entries} == {"outside_root"}
    assert AuditLog(settings.audit_log_path).verify() == 3
    assert metrics.registry.get_sample_value(
        "cascade_admin_actions_total", {"action": "bundle_swap_rejected"}
    ) == pytest.approx(3.0)


def test_the_live_bundle_directory_stays_loadable_from_outside_the_root(
    tiny_bundle_dir: Path, tmp_path: Path
) -> None:
    """Containment must not lock an operator out of reloading the bundle the process is serving."""
    settings = _settings(tmp_path, tiny_bundle_dir, bundle_root=tmp_path / "allowed")
    service = build_service(
        load_bundle(tiny_bundle_dir), settings, create_cascade_metrics(CollectorRegistry())
    )
    with TestClient(create_cascade_app(service), raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/admin/bundle", json={"directory": str(tiny_bundle_dir)}, headers=ADMIN
        )
    assert response.status_code == 200
    assert response.json()["action"] == "bundle_swap"


def test_swapping_from_a_directory_that_does_not_exist_is_refused_and_audited(
    service: CascadeService, client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/v1/admin/bundle", json={"directory": str(tmp_path / "nowhere")}, headers=ADMIN
    )
    assert response.status_code == 422
    entries = AuditLog(service.settings.audit_log_path).read()
    assert entries[-1].action == "bundle_swap_rejected"


def test_the_admin_audit_chain_verifies_after_a_run_of_actions(
    service: CascadeService, client: TestClient, tiny_bundle_dir: Path
) -> None:
    client.post("/v1/admin/reindex", json={}, headers=ADMIN)
    client.post("/v1/admin/bundle", json={"directory": str(tiny_bundle_dir)}, headers=ADMIN)
    client.post("/v1/admin/bundle", json={"directory": "/does/not/exist"}, headers=ADMIN)
    log = AuditLog(service.settings.audit_log_path)
    assert log.verify() == 3
    assert client.get("/v1/bundle", headers=SERVING).json()["audit_head"] == log.head


def test_admin_actions_are_counted_per_action(service: CascadeService, client: TestClient) -> None:
    client.post("/v1/admin/reindex", json={}, headers=ADMIN)
    assert service.metrics.registry.get_sample_value(
        "cascade_admin_actions_total", {"action": "reindex"}
    ) == pytest.approx(1.0)


# --- in-flight tracking ---


def test_the_in_flight_tracker_counts_up_and_down_and_remembers_its_peak() -> None:
    tracker = InFlightTracker()
    assert tracker.depth == 0
    assert tracker.enter() == 1
    assert tracker.enter() == 2
    assert tracker.peak == 2
    tracker.leave()
    tracker.leave()
    tracker.leave()
    assert tracker.depth == 0
    assert tracker.peak == 2


def test_the_in_flight_gauge_is_published(service: CascadeService, client: TestClient) -> None:
    client.post("/v1/recommend", json={"user_id": 1, "k": 5}, headers=SERVING)
    sample = service.metrics.registry.get_sample_value("cascade_in_flight_requests")
    assert sample is not None


# --- deterministic feedback sources ---


def test_feedback_source_ids_are_deterministic_and_well_formed() -> None:
    first = feedback_sources(4, seed=1)
    assert first == feedback_sources(4, seed=1)
    assert first != feedback_sources(4, seed=2)
    assert all(len(source) == 12 and int(source, 16) >= 0 for source in first)
