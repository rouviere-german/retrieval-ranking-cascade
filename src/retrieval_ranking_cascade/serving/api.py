"""HTTP surface for the cascade: recommendations, feedback, monitoring and admin.

Security posture, in the order a request meets it:

1. the baseline middleware stack (request id, security headers, body-size cap, per-client token
   bucket) from ``core.app``;
2. API-key authentication on every route except ``/healthz`` and ``/readyz``;
3. **input bounds** - ``user_id``, ``k``, ``candidate_k``, ``ef_search`` and ``lambda`` are
   validated against the configured maxima, and a *query cost cap* rejects combinations that are
   individually legal but expensive together, so a caller with a valid key cannot turn one request
   into a hundred requests' worth of ANN work;
4. **admin scope** - ``/v1/admin/*`` additionally requires the key digest to appear in
   ``APP_ADMIN_API_KEY_SHA256``, and every privileged action is appended to a hash-chained audit
   log before it returns;
5. **feedback limits** - feedback is validated by its pandera contract and metered by two token
   buckets. ``source`` is a caller-chosen label, not an identity: it gets its own bucket so one
   noisy source does not spend a quiet one's budget, but a client that rotates it would mint a
   fresh bucket per request, so what bounds a *client* is a second bucket keyed on its
   authenticated key id (``feedback_client_rate_multiplier`` times the per-source rate). Accepted
   feedback only moves monitoring gauges - it never reaches the feature snapshot, the item
   statistics the ranker uses or training (see ``serving.monitoring.record_feedback``).

User ids never reach the logs or the audit chain in the clear: they are replaced by keyed,
truncated digests (:class:`serving.audit.UserPseudonymizer`).

Every route is a **synchronous** ``def``. A recommendation is several milliseconds of numpy, faiss
and LightGBM work, and an ``async def`` handler would run it on the event loop and block every other
in-flight request for its whole duration - which is what turned a 10 ms cascade into a 27 ms p50 at
50 RPS before this was fixed. FastAPI runs sync handlers in a worker thread instead, and the heavy
libraries release the GIL, so requests overlap.

That makes the worker pool the queue, and the queue is what the latency budget has to know about.
Two dependencies run on the event loop, before a request is handed to a thread: one stamps
``perf_counter()`` so the handler can subtract the wait from the budget, and one counts accepted
requests in flight so the budget can scale its predictions by the contention they imply. Measured
queue time alone is not enough, because under real overload most of the waiting happens in the
accept backlog and in clients' connection pools, which are invisible from inside the process; the
in-flight count is the part that can be seen. The pool is sized at ``server_threads`` rather than
left at the framework default, because dozens of threads all holding the GIL for their Python
portions is slower than a handful.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, NoReturn

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Request
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ConfigDict, Field

from retrieval_ranking_cascade.core.app import create_app
from retrieval_ranking_cascade.core.canonical import sha256_hex
from retrieval_ranking_cascade.core.config import Settings
from retrieval_ranking_cascade.core.errors import AppError
from retrieval_ranking_cascade.core.logging import get_logger
from retrieval_ranking_cascade.core.ratelimit import RateLimiter
from retrieval_ranking_cascade.core.security import (
    ApiKeyVerifier,
    Principal,
    extract_api_key,
    require_api_key,
)
from retrieval_ranking_cascade.ml.contracts import (
    DataContractError,
    feedback_schema,
    validate,
)
from retrieval_ranking_cascade.retrieval.ann import HnswIndex, HnswParams, set_faiss_threads
from retrieval_ranking_cascade.serving.audit import AuditLog, UserPseudonymizer
from retrieval_ranking_cascade.serving.budget import BudgetController
from retrieval_ranking_cascade.serving.bundle import (
    Bundle,
    BundleIntegrityError,
    load_bundle,
)
from retrieval_ranking_cascade.serving.cascade import Cascade, CascadeConfig, CascadeResult
from retrieval_ranking_cascade.serving.metrics import CascadeMetrics, create_cascade_metrics
from retrieval_ranking_cascade.serving.monitoring import CascadeMonitor, FeedbackEvent

__all__ = [
    "AdminResponse",
    "BundleSwapRequest",
    "CascadeService",
    "FeedbackRequest",
    "RecommendRequest",
    "RecommendResponse",
    "ReindexRequest",
    "build_service",
    "create_app_from_settings",
    "create_cascade_app",
    "feedback_sources",
    "query_cost",
    "require_admin",
]

_log = get_logger("retrieval_ranking_cascade.api")
_MAX_FEEDBACK_EVENTS = 200


def query_cost(candidate_k: int, ef_search: int) -> float:
    """Cost proxy for one ANN query: ``candidate_k * sqrt(max(ef_search, candidate_k))``.

    HNSW work grows roughly linearly in the number of visited nodes (``max(ef, k)``) and the
    per-request cost of everything downstream grows linearly in ``candidate_k``. The square root
    keeps a large ``efSearch`` from dominating the cap while still pricing it, so the limit reads
    as "how many candidates may this request pull, at how careful a search".
    """
    if candidate_k < 1 or ef_search < 1:
        raise ValueError("candidate_k and ef_search must be >= 1")
    return float(candidate_k) * math.sqrt(float(max(ef_search, candidate_k)))


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int = Field(ge=0, le=2**31 - 1)
    k: int | None = Field(default=None, ge=1)
    candidate_k: int | None = Field(default=None, ge=1)
    ef_search: int | None = Field(default=None, ge=1)
    diversity_lambda: float | None = Field(default=None, ge=0.0, le=1.0)


class RecommendedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: int
    rank: int
    score: float
    category: int


class RecommendResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RecommendedItem]
    bundle_id: str
    degraded: bool
    degraded_reasons: list[str]
    reranked: bool
    cold_start: bool
    candidates: int
    reranked_candidates: int
    ef_search: int
    queued_ms: float
    contention: float
    in_flight: int
    stage_ms: dict[str, float]


class FeedbackItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int = Field(ge=0, le=2**31 - 1)
    item_id: int = Field(ge=0)
    position: int = Field(ge=1)
    clicked: int = Field(ge=0, le=1)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(pattern=r"^[0-9a-f]{12}$")
    events: list[FeedbackItem] = Field(min_length=1, max_length=_MAX_FEEDBACK_EVENTS)


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: int
    total_events: int


class ReindexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    m: int = Field(default=48, ge=4, le=128)
    ef_construction: int = Field(default=300, ge=16, le=2048)
    ef_search: int = Field(default=768, ge=1, le=4096)


class BundleSwapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str = Field(min_length=1, max_length=512)


class AdminResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    bundle_id: str
    index_sha256: str
    audit_seq: int
    audit_hash: str


@dataclass(frozen=True, slots=True)
class _Bounds:
    """Per-request limits read from Settings, grouped so they are all checked in one place."""

    max_k: int
    max_candidate_k: int
    max_ef_search: int
    max_query_cost: float
    default_candidate_k: int
    default_ef_search: int


class CascadeService:
    """Everything the routes need: the cascade, the monitor, the audit log and the limits."""

    def __init__(
        self,
        cascade: Cascade,
        monitor: CascadeMonitor,
        metrics: CascadeMetrics,
        settings: Settings,
        *,
        audit: AuditLog | None = None,
        bundle_loader: Any = load_bundle,
        pseudonymizer: UserPseudonymizer | None = None,
    ) -> None:
        self.cascade = cascade
        self.monitor = monitor
        self.metrics = metrics
        self.settings = settings
        self.audit = audit or AuditLog(settings.audit_log_path)
        self.in_flight = InFlightTracker()
        self._load_bundle = bundle_loader
        self._pseudonym = pseudonymizer or UserPseudonymizer()
        self._feedback_limiter = RateLimiter(
            settings.feedback_rate_limit_per_second,
            settings.feedback_rate_limit_burst,
            max_keys=settings.rate_limit_max_clients,
        )
        # A second limiter, not a second key in the first one: rotating ``source`` inserts entries
        # into the per-source limiter, and an eviction there readmits an evicted key with a full
        # bucket (see core.ratelimit). Keeping the per-key buckets in their own map means that
        # churn can never hand a client a fresh client-level allowance.
        multiplier = settings.feedback_client_rate_multiplier
        self._feedback_client_limiter = RateLimiter(
            settings.feedback_rate_limit_per_second * multiplier,
            max(1, math.ceil(settings.feedback_rate_limit_burst * multiplier)),
            max_keys=settings.rate_limit_max_clients,
        )
        self._admin_lock = threading.Lock()
        self.bounds = _Bounds(
            max_k=settings.max_k,
            max_candidate_k=settings.max_candidate_k,
            max_ef_search=settings.max_ef_search,
            max_query_cost=settings.max_query_cost,
            default_candidate_k=settings.candidate_k,
            default_ef_search=settings.default_ef_search,
        )

    # --- request validation ---

    def resolve(self, body: RecommendRequest) -> tuple[int, int, int]:
        """Apply defaults, the per-parameter maxima and the query cost cap."""
        bounds = self.bounds
        k = self.cascade.config.default_k if body.k is None else body.k
        candidate_k = bounds.default_candidate_k if body.candidate_k is None else body.candidate_k
        ef_search = bounds.default_ef_search if body.ef_search is None else body.ef_search
        if k > bounds.max_k:
            raise AppError(422, "k_too_large", f"k must be at most {bounds.max_k}.")
        if candidate_k > bounds.max_candidate_k:
            raise AppError(
                422,
                "candidate_k_too_large",
                f"candidate_k must be at most {bounds.max_candidate_k}.",
            )
        if ef_search > bounds.max_ef_search:
            raise AppError(
                422, "ef_search_too_large", f"ef_search must be at most {bounds.max_ef_search}."
            )
        if k > candidate_k:
            raise AppError(422, "k_above_candidates", "k must not exceed candidate_k.")
        if query_cost(candidate_k, ef_search) > bounds.max_query_cost:
            raise AppError(
                422,
                "query_too_expensive",
                "candidate_k and ef_search together exceed the configured query cost cap.",
            )
        return k, candidate_k, ef_search

    # --- serving ---

    def recommend(
        self, body: RecommendRequest, queued_seconds: float = 0.0, in_flight: int = 1
    ) -> RecommendResponse:
        k, candidate_k, ef_search = self.resolve(body)
        contention = BudgetController.contention(in_flight, self.settings.server_threads)
        result = self.cascade.recommend(
            body.user_id,
            k=k,
            candidate_k=candidate_k,
            ef_search=ef_search,
            mmr_lambda=body.diversity_lambda,
            queued_seconds=queued_seconds,
            contention=contention,
        )
        self._record(result)
        self.monitor.observe(body.user_id, result.item_ids)
        return RecommendResponse(
            items=[
                RecommendedItem(
                    item_id=item.item_id, rank=item.rank, score=item.score, category=item.category
                )
                for item in result.items
            ],
            bundle_id=result.bundle_id,
            degraded=result.degraded,
            degraded_reasons=list(result.degraded_reasons),
            reranked=result.reranked,
            cold_start=result.cold_start,
            candidates=result.candidates,
            reranked_candidates=result.reranked_candidates,
            ef_search=result.ef_search,
            queued_ms=round(1000.0 * result.queued_seconds, 3),
            contention=round(result.contention, 3),
            in_flight=in_flight,
            stage_ms={
                name: round(1000.0 * value, 3) for name, value in result.stage_seconds.items()
            },
        )

    def _record(self, result: CascadeResult) -> None:
        for stage, seconds in result.stage_seconds.items():
            self.metrics.stage_latency_seconds.labels(stage=stage).observe(seconds)
        self.metrics.stage_latency_seconds.labels(stage="queue").observe(result.queued_seconds)
        self.metrics.in_flight_requests.set(float(self.in_flight.depth))
        self.metrics.requests_total.labels(degraded=str(result.degraded).lower()).inc()
        self.metrics.candidates_returned.observe(result.candidates)
        for reason in result.degraded_reasons:
            self.metrics.degraded_responses_total.labels(reason=reason).inc()

    # --- feedback ---

    def ingest(self, principal: Principal, body: FeedbackRequest) -> FeedbackResponse:
        """Validate and record one feedback batch, under two token buckets.

        ``source`` is picked by the caller and is not an identity, so it cannot be the thing that
        bounds a client: a client that rotates it would mint a fresh bucket per request. It still
        earns its own bucket, because that is what keeps one noisy source from spending a quiet
        one's budget under the same key. The bound on the *client* is the second bucket, keyed on
        the authenticated key id. The per-source bucket is checked first, so a request that is
        already refused does not spend a client token.
        """
        self._check_feedback_limits(principal, body.source)
        bundle = self.cascade.bundle
        frame = pd.DataFrame(
            [
                {
                    "user_id": event.user_id,
                    "item_id": event.item_id,
                    "position": event.position,
                    "clicked": event.clicked,
                    "source": body.source,
                }
                for event in body.events
            ]
        ).astype({"user_id": "int64", "item_id": "int64", "position": "int64", "clicked": "int64"})
        schema = feedback_schema(n_items=bundle.n_items, slots=bundle.manifest.slots)
        try:
            validate(frame, schema)
        except DataContractError as exc:
            self.metrics.feedback_rejected_total.labels(reason="contract").inc()
            _log.warning("feedback_contract_violation", failures=exc.failures, source=body.source)
            raise AppError(
                422, "feedback_contract_violation", "Feedback events violate the data contract."
            ) from None
        events = [
            FeedbackEvent(
                user_id=event.user_id,
                item_id=event.item_id,
                position=event.position,
                clicked=event.clicked,
                source=body.source,
            )
            for event in body.events
        ]
        accepted = self.monitor.record_feedback(events)
        return FeedbackResponse(accepted=accepted, total_events=len(body.events))

    def _check_feedback_limits(self, principal: Principal, source: str) -> None:
        """Per-source bucket first, then the per-key bucket that actually bounds the client."""
        for reason, limiter, key in (
            ("rate_limited", self._feedback_limiter, f"{principal.key_id}:{source}"),
            ("client_rate_limited", self._feedback_client_limiter, principal.key_id),
        ):
            decision = limiter.check(key)
            if not decision.allowed:
                self.metrics.feedback_rejected_total.labels(reason=reason).inc()
                raise AppError(
                    429,
                    "feedback_rate_limited",
                    "This client is sending feedback too quickly.",
                    headers={"Retry-After": str(decision.retry_after_seconds)},
                )

    # --- admin ---

    def reindex(self, principal: Principal, body: ReindexRequest) -> AdminResponse:
        """Rebuild the ANN index from the live bundle's item vectors, then swap it in.

        The bundle on disk is untouched: its manifest still describes the index it shipped with, so
        the new index's digest is recorded in the audit chain and exposed by ``/v1/bundle`` instead
        of silently invalidating the bundle's own hashes.
        """
        with self._admin_lock:
            params = HnswParams(
                m=body.m, ef_construction=body.ef_construction, ef_search=body.ef_search
            )
            index = HnswIndex.build(self.cascade.item_vectors(), params, threads=1)
            self.cascade.swap_index(index)
            digest = self._index_digest(index)
            entry = self.audit.append(
                principal.key_id,
                "reindex",
                {
                    "bundle_id": self.cascade.bundle.bundle_id,
                    "index_sha256": digest,
                    "m": body.m,
                    "ef_construction": body.ef_construction,
                    "ef_search": body.ef_search,
                },
            )
        self.metrics.admin_actions_total.labels(action="reindex").inc()
        _log.info("reindex_complete", index_sha256=digest[:12], audit_seq=entry.seq)
        return AdminResponse(
            action="reindex",
            bundle_id=self.cascade.bundle.bundle_id,
            index_sha256=digest,
            audit_seq=entry.seq,
            audit_hash=entry.hash,
        )

    def swap_bundle(self, principal: Principal, body: BundleSwapRequest) -> AdminResponse:
        """Load a bundle from disk, hash-verified, and swap it in; a refusal is audited too.

        The directory is checked for containment in ``bundle_root`` (default ``data_dir``) *before*
        the loader is called, so a leaked admin key cannot use the loader as an existence oracle for
        arbitrary paths on the host. The live ``bundle_dir`` is always allowed, so reloading the
        bundle the process is already serving works wherever it lives.
        """
        directory = Path(body.directory)
        with self._admin_lock:
            outside = self._outside_bundle_root(directory)
            if outside is not None:
                self._audit_swap_refusal(principal, directory, outside)
            try:
                bundle: Bundle = self._load_bundle(directory)
            except (BundleIntegrityError, OSError, ValueError) as exc:
                self._audit_swap_refusal(principal, directory, type(exc).__name__)
            previous = self.cascade.bundle.bundle_id
            self.cascade.swap_bundle(bundle)
            self.monitor.rebind()
            digest = self._index_digest(bundle.index)
            entry = self.audit.append(
                principal.key_id,
                "bundle_swap",
                {
                    "previous_bundle_id": previous,
                    "bundle_id": bundle.bundle_id,
                    "index_sha256": digest,
                    "scale": bundle.manifest.scale,
                },
            )
        self.metrics.admin_actions_total.labels(action="bundle_swap").inc()
        _log.info("bundle_swapped", bundle=bundle.manifest.short_id, audit_seq=entry.seq)
        return AdminResponse(
            action="bundle_swap",
            bundle_id=bundle.bundle_id,
            index_sha256=digest,
            audit_seq=entry.seq,
            audit_hash=entry.hash,
        )

    def _outside_bundle_root(self, directory: Path) -> str | None:
        """``None`` when the directory may be loaded, else the reason to audit and refuse with.

        Resolution happens before the check, so ``..`` segments, a symlink or a short 8.3 path
        cannot walk out of the root, and a path that cannot be resolved at all is refused rather
        than passed on.
        """
        try:
            target = directory.resolve()
            allowed = (self.settings.resolved_bundle_root, self.settings.bundle_dir.resolve())
        except OSError:
            return "unresolvable_path"
        if target in allowed or any(target.is_relative_to(root) for root in allowed):
            return None
        return "outside_root"

    def _audit_swap_refusal(self, principal: Principal, directory: Path, reason: str) -> NoReturn:
        self.audit.append(
            principal.key_id,
            "bundle_swap_rejected",
            {"directory": str(directory), "error": reason},
        )
        self.metrics.admin_actions_total.labels(action="bundle_swap_rejected").inc()
        raise AppError(
            422, "bundle_rejected", "The bundle failed verification and was not loaded."
        ) from None

    @staticmethod
    def _index_digest(index: Any) -> str:
        return sha256_hex(index.to_bytes())

    # --- introspection ---

    def bundle_info(self) -> dict[str, Any]:
        manifest = self.cascade.bundle.manifest
        return {
            "bundle_id": manifest.bundle_id,
            "scale": manifest.scale,
            "created_at": manifest.created_at,
            "seed": manifest.seed,
            "data_sha256": manifest.data_sha256,
            "lockfile_sha256": manifest.lockfile_sha256,
            "n_items": manifest.n_items,
            "n_users": manifest.n_users,
            "candidate_k": manifest.candidate_k,
            "embedding_dim": manifest.embedding_dim,
            "hnsw": manifest.hnsw,
            "live_index_sha256": self._index_digest(self.cascade.index),
            "metrics": manifest.metrics,
            "audit_head": self.audit.head,
        }

    def pseudonym(self, user_id: int) -> str:
        return self._pseudonym(user_id)

    def ready(self) -> bool:
        index = self.cascade.index
        return bool(index.size > 0 and index.size == self.cascade.bundle.n_items)


class InFlightTracker:
    """Counts accepted requests currently being served, for the contention estimate."""

    def __init__(self) -> None:
        self._depth = 0
        self._peak = 0
        self._lock = threading.Lock()

    @property
    def depth(self) -> int:
        with self._lock:
            return self._depth

    @property
    def peak(self) -> int:
        with self._lock:
            return self._peak

    def enter(self) -> int:
        with self._lock:
            self._depth += 1
            self._peak = max(self._peak, self._depth)
            return self._depth

    def leave(self) -> None:
        with self._lock:
            self._depth = max(0, self._depth - 1)


async def request_arrival() -> float:
    """Stamp the moment the request reached the event loop, before any worker thread picks it up.

    Async dependencies resolve on the loop; a synchronous handler is dispatched to the thread pool
    afterwards. The difference between this stamp and the start of the handler is the queue wait.
    """
    return time.perf_counter()


async def request_slot(request: Request) -> AsyncIterator[int]:
    """Occupy an in-flight slot for the whole request and yield how many are occupied."""
    tracker: InFlightTracker = request.app.state.in_flight
    depth = tracker.enter()
    try:
        yield depth
    finally:
        tracker.leave()


ArrivalTime = Annotated[float, Depends(request_arrival)]
InFlightDepth = Annotated[int, Depends(request_slot)]


def require_admin(request: Request) -> Principal:
    """FastAPI dependency: a valid API key that is also listed as an admin key."""
    principal = require_api_key(request)
    admin: ApiKeyVerifier = request.app.state.admin_verifier
    if admin.verify(extract_api_key(request.headers)) is None:
        _log.warning("admin_scope_denied", key_id=principal.key_id)
        raise AppError(403, "forbidden", "This key does not hold the admin scope.")
    return principal


AdminPrincipal = Annotated[Principal, Depends(require_admin)]
ServingPrincipal = Annotated[Principal, Depends(require_api_key)]


def build_service(
    bundle: Bundle,
    settings: Settings,
    metrics: CascadeMetrics,
    *,
    controller: BudgetController | None = None,
    audit: AuditLog | None = None,
    bundle_loader: Any = load_bundle,
) -> CascadeService:
    """Wire a bundle into a cascade, a monitor and a service using ``settings``."""
    budget = controller or BudgetController(
        total_seconds=settings.request_budget_ms / 1000.0,
        retrieval_seconds=settings.retrieval_budget_ms / 1000.0,
        ranking_seconds=settings.ranking_budget_ms / 1000.0,
    )
    cascade = Cascade(
        bundle,
        budget,
        CascadeConfig(
            candidate_k=settings.candidate_k,
            rerank_depth=settings.rerank_depth,
            default_k=settings.default_k,
            ef_search=settings.default_ef_search,
            mmr_lambda=settings.mmr_lambda,
            max_per_category=settings.max_per_category,
        ),
    )
    monitor = CascadeMonitor(
        cascade,
        metrics,
        canary_k=settings.canary_k,
        canary_sample_size=settings.canary_sample_size,
        canary_threshold=settings.canary_threshold,
        seed=bundle.manifest.seed,
    )
    return CascadeService(
        cascade, monitor, metrics, settings, audit=audit, bundle_loader=bundle_loader
    )


def _worker_pool_lifespan(threads: int) -> Any:
    """Size anyio's thread pool, which is where FastAPI runs the synchronous handlers."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        import anyio.to_thread  # noqa: PLC0415 - only needed when an app is actually served

        limiter = anyio.to_thread.current_default_thread_limiter()
        previous = limiter.total_tokens
        limiter.total_tokens = threads
        _log.info("worker_pool_sized", threads=threads, previous=previous)
        try:
            yield
        finally:
            limiter.total_tokens = previous

    return lifespan


def create_cascade_app(
    service: CascadeService,
    *,
    readiness_checks: dict[str, Any] | None = None,
) -> FastAPI:
    """The hardened baseline app plus the cascade's routes.

    HTTP and cascade metrics share the service's registry, so one ``/metrics`` scrape carries both.
    """
    settings = service.settings
    checks: dict[str, Any] = {"index": service.ready, **(readiness_checks or {})}
    app = create_app(
        settings,
        readiness_checks=checks,
        registry=service.metrics.registry,
        lifespan=_worker_pool_lifespan(settings.server_threads),
    )
    app.state.service = service
    app.state.cascade_metrics = service.metrics
    app.state.in_flight = service.in_flight
    app.state.admin_verifier = ApiKeyVerifier(settings.admin_api_key_sha256)

    @app.post("/v1/recommend", response_model=RecommendResponse)
    def recommend(
        body: RecommendRequest,
        principal: ServingPrincipal,
        arrived_at: ArrivalTime,
        in_flight: InFlightDepth,
    ) -> RecommendResponse:
        response = service.recommend(body, max(time.perf_counter() - arrived_at, 0.0), in_flight)
        _log.info(
            "recommend",
            key_id=principal.key_id,
            user=service.pseudonym(body.user_id),
            k=len(response.items),
            degraded=response.degraded,
            queued_ms=response.queued_ms,
            in_flight=in_flight,
            bundle=response.bundle_id[:12],
        )
        return response

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    def feedback(body: FeedbackRequest, principal: ServingPrincipal) -> FeedbackResponse:
        response = service.ingest(principal, body)
        _log.info(
            "feedback_accepted",
            key_id=principal.key_id,
            source=body.source,
            events=response.accepted,
        )
        return response

    @app.get("/v1/bundle")
    def bundle_info(principal: ServingPrincipal) -> dict[str, Any]:
        del principal
        return service.bundle_info()

    @app.get("/v1/monitoring")
    def monitoring(principal: ServingPrincipal) -> dict[str, Any]:
        del principal
        return service.monitor.snapshot().as_dict()

    @app.post("/v1/admin/reindex", response_model=AdminResponse)
    def reindex(body: ReindexRequest, principal: AdminPrincipal) -> AdminResponse:
        return service.reindex(principal, body)

    @app.post("/v1/admin/bundle", response_model=AdminResponse)
    def swap_bundle(body: BundleSwapRequest, principal: AdminPrincipal) -> AdminResponse:
        return service.swap_bundle(principal, body)

    @app.post("/v1/admin/canary")
    def canary(principal: AdminPrincipal) -> dict[str, Any]:
        del principal
        result = service.monitor.canary.run_cycle()
        return {
            "cycle": result.cycle,
            "recall_mean": round(result.recall_mean, 6),
            "recall_min": round(result.recall_min, 6),
            "queries": result.queries,
            "k": result.k,
            "alert": result.alert,
            "threshold": result.threshold,
        }

    return app


def create_app_from_settings(settings: Settings) -> FastAPI:
    """Load the bundle from ``settings.bundle_dir`` and build the app (used by ``serve``)."""
    # One faiss thread per search by default: the server already has many request threads, and
    # letting each search fan out would oversubscribe the CPU and hurt tail latency.
    set_faiss_threads(settings.faiss_threads)
    bundle = load_bundle(settings.bundle_dir)
    service = build_service(bundle, settings, create_cascade_metrics(CollectorRegistry()))
    return create_cascade_app(service)


def feedback_sources(count: int, *, seed: int = 0) -> Sequence[str]:
    """Deterministic 12-hex-character feedback source ids, for the demo and the load test."""
    rng = np.random.default_rng(seed)
    return [f"{int(value):012x}" for value in rng.integers(0, 16**12, size=count)]
