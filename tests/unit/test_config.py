from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from retrieval_ranking_cascade.core.config import Settings, get_settings

pytestmark = pytest.mark.unit


def test_defaults_fail_closed() -> None:
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == []
    assert settings.cors_allow_origins == []
    assert settings.enable_docs is False
    assert settings.metrics_require_auth is True
    assert settings.host == "127.0.0.1"
    assert settings.admin_api_key_sha256 == []
    assert settings.faiss_threads == 1


def test_environment_variables_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_API_KEY_SHA256", f"{'a' * 64}, {'b' * 64}")
    monkeypatch.setenv("APP_CORS_ALLOW_ORIGINS", "https://app.example.com,http://localhost:3000")
    monkeypatch.setenv("APP_ADMIN_API_KEY_SHA256", "c" * 64)
    monkeypatch.setenv("APP_DEFAULT_EF_SEARCH", "128")
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == ["a" * 64, "b" * 64]
    assert settings.cors_allow_origins == ["https://app.example.com", "http://localhost:3000"]
    assert settings.admin_api_key_sha256 == ["c" * 64]
    assert settings.default_ef_search == 128


@pytest.mark.parametrize("origins", [["*"], ["https://example.com/"], ["example.com"]])
def test_wildcard_or_malformed_origins_are_rejected(origins: list[str]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_allow_origins=origins)


@pytest.mark.parametrize("digest", ["A" * 64, "abc", "g" * 64])
@pytest.mark.parametrize("field", ["api_key_sha256", "admin_api_key_sha256"])
def test_non_sha256_key_hashes_are_rejected(field: str, digest: str) -> None:
    values: dict[str, Any] = {field: [digest]}
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"default_k": 10, "max_k": 5},
        {"candidate_k": 100, "max_candidate_k": 50},
        {"default_ef_search": 100, "max_ef_search": 50},
        {"request_budget_ms": 10.0, "retrieval_budget_ms": 8.0, "ranking_budget_ms": 8.0},
    ],
)
def test_inconsistent_bounds_and_budgets_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_env_example_documents_every_setting(repo_root: Path) -> None:
    lines = (repo_root / ".env.example").read_text(encoding="utf-8").splitlines()
    documented = {
        line.split("=", 1)[0].strip() for line in lines if line.strip() and not line.startswith("#")
    }
    expected = {
        "ANTHROPIC_API_KEY" if name == "anthropic_api_key" else f"APP_{name.upper()}"
        for name in Settings.model_fields
    }
    assert documented == expected


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
