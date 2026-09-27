"""Shared fixtures. Every test runs offline: non-loopback sockets are blocked."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog
from hypothesis import HealthCheck, settings

settings.register_profile("dev", max_examples=60, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


@pytest.fixture(autouse=True)
def _offline_and_clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name, raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            raise RuntimeError(f"tests must not use the network (attempted {host})")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    structlog.reset_defaults()
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tiny_training() -> Any:
    """One trained bundle at ``tiny`` scale, shared by every test that needs a real model.

    Training takes a couple of seconds, which is cheap once and not cheap per test. Tests must treat
    the result as read-only; anything that mutates an index takes a ``copy()`` first.
    """
    from retrieval_ranking_cascade.pipeline import TrainingConfig, train_bundle
    from retrieval_ranking_cascade.retrieval.two_tower import TowerConfig

    config = TrainingConfig(
        scale="tiny",
        threads=2,
        candidate_k=200,
        recall_sample=40,
        tower=TowerConfig(epochs=3, batch_size=256, uniform_negatives=256),
    )
    return train_bundle(config, repo_root=REPO_ROOT)


@pytest.fixture(scope="session")
def tiny_bundle_dir(tiny_training: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The session bundle written to disk, for tests that exercise the load path."""
    directory = tmp_path_factory.mktemp("bundle") / "tiny"
    tiny_training.write(directory)
    return directory
