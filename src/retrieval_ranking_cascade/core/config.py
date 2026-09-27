"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. API keys (serving and admin scope) are configured as
SHA-256 digests, so the process never holds a raw key; nothing secret is stored in plain text.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(
        default="retrieval-ranking-cascade", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$"
    )
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security ---
    api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=1_048_576, ge=1)
    metrics_require_auth: bool = True

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"

    # --- cascade serving ---
    bundle_dir: Path = Path("var/bundle")
    faiss_threads: int = Field(default=1, ge=1, le=64)
    # Worker threads FastAPI runs the synchronous handlers in. A handful beats the framework
    # default of 40: the Python parts of a request hold the GIL, so extra threads only add
    # contention and context switches.
    server_threads: int = Field(default=8, ge=1, le=256)
    default_k: int = Field(default=20, ge=1)
    max_k: int = Field(default=100, ge=1)
    candidate_k: int = Field(default=500, ge=1)
    max_candidate_k: int = Field(default=2_000, ge=1)
    # Candidates stage 2 scores, out of candidate_k retrieved (see serving.cascade).
    rerank_depth: int = Field(default=150, ge=1)
    default_ef_search: int = Field(default=768, ge=1)
    max_ef_search: int = Field(default=2_048, ge=1)
    # Query cost cap: candidate_k * sqrt(ef_search) must stay below this (see serving.cost).
    max_query_cost: float = Field(default=40_000.0, gt=0)
    request_budget_ms: float = Field(default=50.0, gt=0)
    retrieval_budget_ms: float = Field(default=20.0, gt=0)
    ranking_budget_ms: float = Field(default=20.0, gt=0)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    max_per_category: int = Field(default=5, ge=1)
    canary_interval_seconds: float = Field(default=30.0, gt=0)
    canary_sample_size: int = Field(default=32, ge=1)
    canary_k: int = Field(default=100, ge=1)
    canary_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    admin_api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    feedback_rate_limit_per_second: float = Field(default=2.0, gt=0)
    feedback_rate_limit_burst: int = Field(default=20, ge=1)
    # A caller picks its own ``source``, so the per-source bucket cannot bound a client on its own.
    # The per-key bucket runs at this multiple of the per-source rate and burst: how many distinct
    # sources one API key may keep busy at once.
    feedback_client_rate_multiplier: float = Field(default=4.0, ge=1.0, le=1_000.0)
    audit_log_path: Path = Path("var/audit/admin.jsonl")

    # --- storage ---
    data_dir: Path = Path("var")
    # Directories ``/v1/admin/bundle`` may load from. Defaults to ``data_dir``; the live
    # ``bundle_dir`` is always allowed so a reload in place works wherever it lives.
    bundle_root: Path | None = None

    @property
    def resolved_bundle_root(self) -> Path:
        return (self.bundle_root or self.data_dir).resolve()

    @field_validator("api_key_sha256", "admin_api_key_sha256", mode="before")
    @classmethod
    def _parse_key_hashes(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("api_key_sha256", "admin_api_key_sha256")
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        for digest in value:
            if not _SHA256_HEX.fullmatch(digest):
                raise ValueError("api_key_sha256 entries must be lowercase hex SHA-256 digests")
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.default_k > self.max_k:
            raise ValueError("default_k must not exceed max_k")
        if self.candidate_k > self.max_candidate_k:
            raise ValueError("candidate_k must not exceed max_candidate_k")
        if self.rerank_depth > self.candidate_k:
            raise ValueError("rerank_depth must not exceed candidate_k")
        if self.default_ef_search > self.max_ef_search:
            raise ValueError("default_ef_search must not exceed max_ef_search")
        if self.retrieval_budget_ms + self.ranking_budget_ms > self.request_budget_ms:
            raise ValueError("stage budgets must fit inside request_budget_ms")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
