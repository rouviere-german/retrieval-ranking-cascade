"""The serving bundle: tower, HNSW index, ranker and feature snapshot as one verified unit.

A cascade is only correct when its four artifacts agree. A tower trained yesterday with today's
feature snapshot silently changes the meaning of ``tower_score``; an index built from other
embeddings returns neighbours from a different space. So they ship together, addressed by one
``bundle_id``, and the manifest records what produced them: the impression-log hash, every seed,
every hyper-parameter, the lockfile hash and a SHA-256 per artifact file.

Load order matters for safety. The manifest is read first, every artifact's digest is checked
against it, and only then is a parser allowed near the bytes: faiss' index reader and LightGBM's
model reader are C++ parsers, and ``.npz`` is loaded with ``allow_pickle=False``. Nothing in the
bundle is pickled. ``bundle_id`` is the SHA-256 of the canonical manifest with the id removed, so
it covers the artifact digests transitively and a tampered manifest fails its own id check.

Item embeddings are not stored: they are recomputed from the item tower and the snapshot at load
time, which makes the recall canary an end-to-end check (index versus the vectors the towers
actually produce now) instead of a comparison against a second stored copy. ``reference.npz``
carries the subsampled training distributions the drift monitors compare live traffic against, so
a bundle is self-contained for monitoring as well as for serving.
"""

from __future__ import annotations

import hmac
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from retrieval_ranking_cascade.core.canonical import canonical_json, sha256_hex
from retrieval_ranking_cascade.core.fileio import atomic_write_bytes, atomic_write_text
from retrieval_ranking_cascade.ml.manifest import RunManifest
from retrieval_ranking_cascade.ml.serialization import dumps_npz, loads_npz
from retrieval_ranking_cascade.ranking.features import (
    FeatureSnapshot,
    item_side_features,
    user_side_features,
)
from retrieval_ranking_cascade.ranking.ranker import Ranker
from retrieval_ranking_cascade.retrieval.ann import HnswIndex, HnswParams
from retrieval_ranking_cascade.retrieval.two_tower import TowerConfig, TwoTowerModel

__all__ = [
    "MANIFEST_NAME",
    "ArtifactRef",
    "Bundle",
    "BundleIntegrityError",
    "BundleManifest",
    "load_bundle",
    "write_bundle",
]

MANIFEST_NAME: Final = "manifest.json"
_TOWER_FILE: Final = "tower.npz"
_INDEX_FILE: Final = "index.faiss"
_RANKER_FILE: Final = "ranker.txt"
_SNAPSHOT_FILE: Final = "snapshot.npz"
_REFERENCE_FILE: Final = "reference.npz"
_ARTIFACT_FILES: Final = (
    _TOWER_FILE,
    _INDEX_FILE,
    _RANKER_FILE,
    _SNAPSHOT_FILE,
    _REFERENCE_FILE,
)


class BundleIntegrityError(ValueError):
    """A bundle artifact, manifest or id does not match its recorded hash."""


class ArtifactRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class BundleManifest(BaseModel):
    """Everything needed to say what this bundle is and to reproduce it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    bundle_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str
    scale: str
    seed: int
    data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lockfile_sha256: str | None = None
    n_users: int = Field(ge=1)
    n_items: int = Field(ge=1)
    n_categories: int = Field(ge=1)
    slots: int = Field(ge=1)
    candidate_k: int = Field(ge=1)
    embedding_dim: int = Field(ge=1)
    embedding_norm_mean: float
    embedding_norm_std: float
    world: dict[str, JsonValue]
    tower: dict[str, JsonValue]
    hnsw: dict[str, JsonValue]
    ranker: dict[str, JsonValue]
    snapshot: dict[str, JsonValue]
    # Examination propensities the ranker's IPS weights were built from, for provenance.
    propensities: list[float] = Field(default_factory=list)
    artifacts: dict[str, ArtifactRef]
    metrics: dict[str, float]
    run: RunManifest

    def identity_payload(self) -> dict[str, JsonValue]:
        """The manifest without ``bundle_id``: what the id is the hash of."""
        body: dict[str, JsonValue] = self.model_dump(mode="json")
        body.pop("bundle_id")
        return body

    def expected_id(self) -> str:
        return sha256_hex(canonical_json(self.identity_payload()))

    @property
    def short_id(self) -> str:
        return self.bundle_id[:12]


def _digest(data: bytes) -> ArtifactRef:
    return ArtifactRef(sha256=sha256_hex(data), size_bytes=len(data))


def build_manifest(
    *,
    scale: str,
    seed: int,
    data_sha256: str,
    snapshot: FeatureSnapshot,
    candidate_k: int,
    embedding_norms: npt.NDArray[np.floating[Any]],
    embedding_dim: int,
    world: Mapping[str, JsonValue],
    tower: Mapping[str, JsonValue],
    hnsw: Mapping[str, JsonValue],
    ranker: Mapping[str, JsonValue],
    snapshot_params: Mapping[str, JsonValue],
    artifacts: Mapping[str, bytes],
    metrics: Mapping[str, float],
    run: RunManifest,
    slots: int,
    propensities: list[float] | None = None,
    lockfile_sha256: str | None = None,
    created_at: str | None = None,
) -> BundleManifest:
    """Assemble a manifest and derive its ``bundle_id`` from its own contents."""
    missing = set(_ARTIFACT_FILES) - set(artifacts)
    if missing:
        raise ValueError(f"bundle artifacts missing: {sorted(missing)}")
    fields: dict[str, Any] = {
        "bundle_id": "0" * 64,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "scale": scale,
        "seed": seed,
        "data_sha256": data_sha256,
        "snapshot_sha256": snapshot.sha256(),
        "lockfile_sha256": lockfile_sha256,
        "n_users": snapshot.n_users,
        "n_items": snapshot.n_items,
        "n_categories": snapshot.n_categories,
        "slots": slots,
        "candidate_k": candidate_k,
        "embedding_dim": embedding_dim,
        "embedding_norm_mean": float(np.mean(embedding_norms)),
        "embedding_norm_std": float(np.std(embedding_norms)),
        "world": dict(world),
        "tower": dict(tower),
        "hnsw": dict(hnsw),
        "ranker": dict(ranker),
        "snapshot": dict(snapshot_params),
        "propensities": list(propensities or []),
        "artifacts": {name: _digest(data) for name, data in sorted(artifacts.items())},
        "metrics": {key: float(value) for key, value in sorted(metrics.items())},
        "run": run,
    }
    draft = BundleManifest.model_validate(fields)
    return draft.model_copy(update={"bundle_id": draft.expected_id()})


@dataclass(frozen=True, slots=True)
class Bundle:
    """A loaded, hash-verified bundle plus the arrays the cascade serves from."""

    manifest: BundleManifest
    model: TwoTowerModel
    index: HnswIndex
    ranker: Ranker
    snapshot: FeatureSnapshot
    item_vectors: npt.NDArray[np.float32]
    user_side: npt.NDArray[np.float32]
    item_side: npt.NDArray[np.float32]
    reference: dict[str, npt.NDArray[np.float64]]

    @property
    def bundle_id(self) -> str:
        return self.manifest.bundle_id

    @property
    def n_items(self) -> int:
        return self.snapshot.n_items

    @property
    def n_users(self) -> int:
        return self.snapshot.n_users

    def user_vectors(self, users: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Embed users through the user tower using the snapshot's side features."""
        index = np.asarray(users, dtype=np.int64)
        if index.size and (index.min() < 0 or index.max() >= self.n_users):
            raise ValueError("user ids must be within the bundle's user range")
        vectors, _ = self.model.user.embed(index, self.user_side[index])
        return np.asarray(vectors, dtype=np.float32)

    def to_files(self) -> dict[str, bytes]:
        """The artifact bytes, regenerated from the in-memory objects (used by the registry)."""
        return {
            _TOWER_FILE: dumps_npz(self.model.to_arrays()),
            _INDEX_FILE: self.index.to_bytes(),
            _RANKER_FILE: self.ranker.to_text().encode("utf-8"),
            _SNAPSHOT_FILE: dumps_npz(self.snapshot.arrays()),
            _REFERENCE_FILE: dumps_npz(self.reference),
        }


def _known_fields(values: Mapping[str, JsonValue], names: Iterable[str]) -> dict[str, Any]:
    """Manifest values narrowed to a dataclass' own fields, so a newer manifest still loads."""
    wanted = set(names)
    return {key: value for key, value in values.items() if key in wanted}


def _tower_config(manifest: BundleManifest) -> TowerConfig:
    return TowerConfig(**_known_fields(manifest.tower, TowerConfig.__dataclass_fields__))


def _hnsw_params(manifest: BundleManifest) -> HnswParams:
    return HnswParams(**_known_fields(manifest.hnsw, HnswParams.__dataclass_fields__))


def write_bundle(directory: Path, manifest: BundleManifest, artifacts: Mapping[str, bytes]) -> Path:
    """Write the artifacts, then the manifest, so a half-written bundle never looks complete."""
    for name, reference in manifest.artifacts.items():
        data = artifacts.get(name)
        if data is None:
            raise BundleIntegrityError(f"manifest lists {name} but no bytes were supplied")
        if not hmac.compare_digest(sha256_hex(data), reference.sha256):
            raise BundleIntegrityError(f"{name} does not match the digest in the manifest")
    if not hmac.compare_digest(manifest.bundle_id, manifest.expected_id()):
        raise BundleIntegrityError("bundle_id does not match the manifest contents")
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in sorted(artifacts.items()):
        atomic_write_bytes(directory / name, data)
    path = directory / MANIFEST_NAME
    atomic_write_text(path, manifest.model_dump_json(indent=2) + "\n")
    return path


def load_bundle(directory: Path) -> Bundle:
    """Load and verify a bundle written by :func:`write_bundle`."""
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleIntegrityError(f"no {MANIFEST_NAME} in {directory}")
    try:
        manifest = BundleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BundleIntegrityError(f"{MANIFEST_NAME} is not a valid bundle manifest") from exc
    if not hmac.compare_digest(manifest.bundle_id, manifest.expected_id()):
        raise BundleIntegrityError("bundle_id does not match the manifest contents")

    payloads: dict[str, bytes] = {}
    for name in _ARTIFACT_FILES:
        expected = manifest.artifacts.get(name)
        if expected is None:
            raise BundleIntegrityError(f"manifest does not list the required artifact {name}")
        path = directory / name
        if not path.is_file():
            raise BundleIntegrityError(f"missing bundle artifact {name}")
        data = path.read_bytes()
        if len(data) != expected.size_bytes or not hmac.compare_digest(
            sha256_hex(data), expected.sha256
        ):
            raise BundleIntegrityError(f"bundle artifact {name} does not match its digest")
        payloads[name] = data

    snapshot = FeatureSnapshot.from_arrays(loads_npz(payloads[_SNAPSHOT_FILE]))
    if not hmac.compare_digest(snapshot.sha256(), manifest.snapshot_sha256):
        raise BundleIntegrityError("feature snapshot content hash does not match the manifest")
    model = TwoTowerModel.from_arrays(loads_npz(payloads[_TOWER_FILE]), _tower_config(manifest))
    if model.output_dim != manifest.embedding_dim:
        raise BundleIntegrityError("tower output dimension does not match the manifest")
    ranker = Ranker.from_text(
        payloads[_RANKER_FILE].decode("utf-8"),
        expected_sha256=manifest.artifacts[_RANKER_FILE].sha256,
    )
    index = HnswIndex.from_bytes(
        payloads[_INDEX_FILE],
        expected_sha256=manifest.artifacts[_INDEX_FILE].sha256,
        params=_hnsw_params(manifest),
        dim=manifest.embedding_dim,
        size=snapshot.n_items,
    )
    reference_arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in loads_npz(payloads[_REFERENCE_FILE]).items()
    }
    user_side = user_side_features(snapshot)
    item_side = item_side_features(snapshot)
    item_vectors, _ = model.item.embed(np.arange(snapshot.n_items), item_side)
    return Bundle(
        manifest=manifest,
        model=model,
        index=index,
        ranker=ranker,
        snapshot=snapshot,
        item_vectors=np.asarray(item_vectors, dtype=np.float32),
        user_side=user_side,
        item_side=item_side,
        reference=reference_arrays,
    )
