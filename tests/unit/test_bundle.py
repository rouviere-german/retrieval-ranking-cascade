"""The serving bundle: manifest identity, hash-verified load, tamper detection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from retrieval_ranking_cascade.core.canonical import canonical_json, sha256_hex
from retrieval_ranking_cascade.serving.bundle import (
    MANIFEST_NAME,
    ArtifactRef,
    Bundle,
    BundleIntegrityError,
    BundleManifest,
    load_bundle,
    write_bundle,
)

pytestmark = pytest.mark.unit

_ARTIFACTS = ("tower.npz", "index.faiss", "ranker.txt", "snapshot.npz", "reference.npz")


@pytest.fixture
def bundle_copy(tiny_bundle_dir: Path, tmp_path: Path) -> Path:
    """A writable copy of the session bundle, so a test can damage it freely."""
    target = tmp_path / "bundle"
    shutil.copytree(tiny_bundle_dir, target)
    return target


def _manifest(directory: Path) -> BundleManifest:
    return BundleManifest.model_validate_json(
        (directory / MANIFEST_NAME).read_text(encoding="utf-8")
    )


def _write_manifest(directory: Path, manifest: BundleManifest) -> None:
    (directory / MANIFEST_NAME).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


# --- manifest identity ---


def test_the_bundle_id_is_the_hash_of_the_rest_of_the_manifest(tiny_bundle_dir: Path) -> None:
    manifest = _manifest(tiny_bundle_dir)
    assert manifest.bundle_id == manifest.expected_id()
    assert manifest.bundle_id == sha256_hex(canonical_json(manifest.identity_payload()))
    assert "bundle_id" not in manifest.identity_payload()
    assert manifest.short_id == manifest.bundle_id[:12]


def test_the_id_covers_the_artifact_digests_transitively(tiny_bundle_dir: Path) -> None:
    manifest = _manifest(tiny_bundle_dir)
    nudged = dict(manifest.artifacts)
    nudged["ranker.txt"] = ArtifactRef(sha256="f" * 64, size_bytes=1)
    assert manifest.model_copy(update={"artifacts": nudged}).expected_id() != manifest.bundle_id


def test_every_required_artifact_is_listed_with_a_digest_and_a_size(
    tiny_bundle_dir: Path,
) -> None:
    manifest = _manifest(tiny_bundle_dir)
    assert set(manifest.artifacts) == set(_ARTIFACTS)
    for name in _ARTIFACTS:
        reference = manifest.artifacts[name]
        actual = (tiny_bundle_dir / name).read_bytes()
        assert reference.sha256 == sha256_hex(actual)
        assert reference.size_bytes == len(actual)


def test_the_manifest_records_what_produced_the_bundle(tiny_bundle_dir: Path) -> None:
    manifest = _manifest(tiny_bundle_dir)
    assert len(manifest.data_sha256) == 64
    assert len(manifest.snapshot_sha256) == 64
    assert manifest.lockfile_sha256 is not None
    assert manifest.seed >= 0
    assert manifest.scale == "tiny"
    assert manifest.propensities
    assert manifest.propensities[0] == pytest.approx(1.0)
    assert manifest.run.seed == manifest.seed
    assert "tower" in manifest.run.params
    assert manifest.embedding_dim >= 1
    assert manifest.metrics


def test_the_manifest_forbids_unknown_fields_and_bad_digests(tiny_bundle_dir: Path) -> None:
    body = json.loads((tiny_bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="extra_forbidden"):
        BundleManifest.model_validate({**body, "surprise": 1})
    with pytest.raises(ValueError, match="string_pattern_mismatch"):
        BundleManifest.model_validate({**body, "data_sha256": "nope"})


# --- writing ---


def test_writing_refuses_bytes_that_do_not_match_the_manifest(
    tiny_bundle_dir: Path, tmp_path: Path
) -> None:
    manifest = _manifest(tiny_bundle_dir)
    artifacts = {name: (tiny_bundle_dir / name).read_bytes() for name in _ARTIFACTS}
    with pytest.raises(BundleIntegrityError, match="does not match the digest"):
        write_bundle(
            tmp_path / "out", manifest, {**artifacts, "ranker.txt": b"something else entirely"}
        )


def test_writing_refuses_a_missing_artifact(tiny_bundle_dir: Path, tmp_path: Path) -> None:
    manifest = _manifest(tiny_bundle_dir)
    artifacts = {name: (tiny_bundle_dir / name).read_bytes() for name in _ARTIFACTS}
    del artifacts["index.faiss"]
    with pytest.raises(BundleIntegrityError, match="but no bytes were supplied"):
        write_bundle(tmp_path / "out", manifest, artifacts)


def test_writing_refuses_a_manifest_whose_id_does_not_match_itself(
    tiny_bundle_dir: Path, tmp_path: Path
) -> None:
    manifest = _manifest(tiny_bundle_dir).model_copy(update={"scale": "small"})
    artifacts = {name: (tiny_bundle_dir / name).read_bytes() for name in _ARTIFACTS}
    with pytest.raises(BundleIntegrityError, match="bundle_id does not match"):
        write_bundle(tmp_path / "out", manifest, artifacts)


def test_a_written_bundle_loads_back_and_serves_the_same_neighbours(
    tiny_training: Any, tmp_path: Path
) -> None:
    directory = tmp_path / "roundtrip"
    tiny_training.write(directory)
    loaded = load_bundle(directory)
    original = tiny_training.bundle
    assert loaded.bundle_id == original.bundle_id
    assert loaded.n_items == original.n_items
    assert loaded.n_users == original.n_users
    np.testing.assert_allclose(loaded.item_vectors, original.item_vectors, atol=1e-6)
    query = loaded.user_vectors(np.asarray([0, 1, 2]))
    assert loaded.index.search(query, 5)[1].tolist() == original.index.search(query, 5)[1].tolist()


def test_to_files_regenerates_bytes_that_still_match_the_manifest(tiny_training: Any) -> None:
    bundle: Bundle = tiny_training.bundle
    files = bundle.to_files()
    assert set(files) == set(_ARTIFACTS)
    for name, data in files.items():
        assert sha256_hex(data) == bundle.manifest.artifacts[name].sha256


# --- loading and verification ---


def test_a_missing_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(BundleIntegrityError, match=f"no {MANIFEST_NAME}"):
        load_bundle(tmp_path / "nowhere")


def test_an_unparseable_manifest_is_reported(bundle_copy: Path) -> None:
    (bundle_copy / MANIFEST_NAME).write_text("{ not json", encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="not a valid bundle manifest"):
        load_bundle(bundle_copy)


def test_editing_the_manifest_breaks_its_own_id(bundle_copy: Path) -> None:
    manifest = _manifest(bundle_copy)
    _write_manifest(bundle_copy, manifest.model_copy(update={"seed": manifest.seed + 1}))
    with pytest.raises(BundleIntegrityError, match="bundle_id does not match"):
        load_bundle(bundle_copy)


@pytest.mark.parametrize("name", _ARTIFACTS)
def test_flipping_one_byte_in_any_artifact_is_caught_before_it_is_parsed(
    bundle_copy: Path, name: str
) -> None:
    path = bundle_copy / name
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(BundleIntegrityError, match=f"{name} does not match its digest"):
        load_bundle(bundle_copy)


@pytest.mark.parametrize("name", _ARTIFACTS)
def test_a_deleted_artifact_is_reported_by_name(bundle_copy: Path, name: str) -> None:
    (bundle_copy / name).unlink()
    with pytest.raises(BundleIntegrityError, match=f"missing bundle artifact {name}"):
        load_bundle(bundle_copy)


def test_truncating_an_artifact_is_caught_by_its_recorded_size(bundle_copy: Path) -> None:
    path = bundle_copy / "ranker.txt"
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(BundleIntegrityError, match="does not match its digest"):
        load_bundle(bundle_copy)


def test_a_manifest_that_omits_a_required_artifact_is_refused(bundle_copy: Path) -> None:
    """Dropping an entry and re-deriving the id must not buy a bundle that loads."""
    manifest = _manifest(bundle_copy)
    artifacts = {k: v for k, v in manifest.artifacts.items() if k != "reference.npz"}
    draft = manifest.model_copy(update={"artifacts": artifacts})
    _write_manifest(bundle_copy, draft.model_copy(update={"bundle_id": draft.expected_id()}))
    with pytest.raises(BundleIntegrityError, match=r"required artifact reference\.npz"):
        load_bundle(bundle_copy)


def test_a_manifest_that_lies_about_the_snapshot_hash_is_refused(bundle_copy: Path) -> None:
    manifest = _manifest(bundle_copy)
    draft = manifest.model_copy(update={"snapshot_sha256": "0" * 64})
    _write_manifest(bundle_copy, draft.model_copy(update={"bundle_id": draft.expected_id()}))
    with pytest.raises(BundleIntegrityError, match="feature snapshot content hash"):
        load_bundle(bundle_copy)


def test_a_manifest_that_lies_about_the_embedding_dimension_is_refused(bundle_copy: Path) -> None:
    manifest = _manifest(bundle_copy)
    draft = manifest.model_copy(update={"embedding_dim": manifest.embedding_dim + 1})
    _write_manifest(bundle_copy, draft.model_copy(update={"bundle_id": draft.expected_id()}))
    with pytest.raises(BundleIntegrityError, match="tower output dimension"):
        load_bundle(bundle_copy)


def test_nothing_in_a_bundle_is_pickled(tiny_bundle_dir: Path) -> None:
    """``.npz`` is loaded with ``allow_pickle=False``; a pickled array must not even be written."""
    for name in ("tower.npz", "snapshot.npz", "reference.npz"):
        data = (tiny_bundle_dir / name).read_bytes()
        assert b"numpy.core.multiarray._reconstruct" not in data
        assert b"cnumpy" not in data


# --- loaded bundle behaviour ---


def test_user_vectors_are_unit_norm_and_indexed_by_user_id(tiny_training: Any) -> None:
    bundle: Bundle = tiny_training.bundle
    vectors = bundle.user_vectors(np.asarray([0, 5, bundle.n_users - 1]))
    assert vectors.shape == (3, bundle.manifest.embedding_dim)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_user_vectors_refuse_ids_outside_the_bundle(tiny_training: Any) -> None:
    bundle: Bundle = tiny_training.bundle
    with pytest.raises(ValueError, match="within the bundle's user range"):
        bundle.user_vectors(np.asarray([bundle.n_users]))
    with pytest.raises(ValueError, match="within the bundle's user range"):
        bundle.user_vectors(np.asarray([-1]))
    assert bundle.user_vectors(np.empty(0, dtype=np.int64)).shape[0] == 0


def test_item_vectors_are_recomputed_from_the_tower_not_stored(tiny_training: Any) -> None:
    """That is what makes the recall canary an end-to-end check of the live index."""
    bundle: Bundle = tiny_training.bundle
    recomputed, _ = bundle.model.item.embed(np.arange(bundle.n_items), bundle.item_side)
    np.testing.assert_allclose(bundle.item_vectors, recomputed, rtol=1e-6)
    assert "item_vectors.npz" not in bundle.to_files()


def test_the_reference_arrays_cover_every_ranker_feature(tiny_training: Any) -> None:
    bundle: Bundle = tiny_training.bundle
    assert "embedding_norms" in bundle.reference
    assert any(name.startswith("feature_") for name in bundle.reference)
    for values in bundle.reference.values():
        assert values.dtype == np.float64
        assert values.size > 0
