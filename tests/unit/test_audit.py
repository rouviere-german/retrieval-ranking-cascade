"""The hash-chained admin audit log and the user pseudonymizer."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval_ranking_cascade.serving.audit import (
    GENESIS,
    AuditEntry,
    AuditError,
    AuditLog,
    UserPseudonymizer,
    pseudonymous_user_id,
    verify_file,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def log(tmp_path: Path) -> AuditLog:
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    return AuditLog(tmp_path / "audit" / "admin.jsonl", clock=lambda: stamp)


def _rewrite(path: Path, transform: object) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert callable(transform)
    path.write_text("\n".join(transform(lines)) + "\n", encoding="utf-8")


# --- basics ---


def test_an_empty_log_reads_as_nothing_with_a_genesis_head(log: AuditLog) -> None:
    assert log.read() == []
    assert log.head == GENESIS
    assert log.verify() == 0


def test_appending_builds_a_chain_from_genesis(log: AuditLog) -> None:
    first = log.append("key-1", "reindex", {"m": 32})
    second = log.append("key-1", "bundle_swap", {"scale": "small"})
    assert first.seq == 0
    assert first.prev == GENESIS
    assert second.seq == 1
    assert second.prev == first.hash
    assert log.head == second.hash
    assert log.verify() == 2
    assert verify_file(log.path) == 2


def test_the_parent_directory_is_created_on_first_append(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "admin.jsonl"
    AuditLog(path).append("key", "reindex")
    assert path.is_file()


def test_details_default_to_an_empty_mapping(log: AuditLog) -> None:
    assert log.append("key", "canary").details == {}


def test_each_entry_hash_covers_its_own_payload(log: AuditLog) -> None:
    entry = log.append("key-1", "reindex", {"index_sha256": "a" * 64})
    assert entry.hash == entry.expected_hash()
    assert "hash" not in entry.payload()
    assert entry.payload()["actor"] == "key-1"


def test_secrets_in_details_are_redacted_before_they_are_written(log: AuditLog) -> None:
    """The audit trail must not become a second copy of a payload it was handed."""
    entry = log.append("key-1", "bundle_swap", {"api_key": "super-secret-value", "scale": "tiny"})
    assert "super-secret-value" not in json.dumps(entry.details)
    assert "super-secret-value" not in log.path.read_text(encoding="utf-8")
    assert entry.details["scale"] == "tiny"


def test_blank_lines_are_skipped_when_reading(log: AuditLog) -> None:
    log.append("key", "reindex")
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write("\n   \n")
    assert len(log.read()) == 1
    assert log.verify() == 1


# --- tamper detection ---


def test_editing_an_entry_without_rehashing_breaks_its_own_hash(log: AuditLog) -> None:
    log.append("key-1", "reindex")
    log.append("key-1", "bundle_swap")

    def forge(lines: list[str]) -> list[str]:
        body = json.loads(lines[-1])
        body["actor"] = "attacker"
        return [*lines[:-1], json.dumps(body, sort_keys=True, separators=(",", ":"))]

    _rewrite(log.path, forge)
    with pytest.raises(AuditError, match="entry 1: contents do not match"):
        log.verify()


def test_rehashing_an_edited_entry_still_breaks_the_link_to_the_next_one(log: AuditLog) -> None:
    """Recomputing the hash is not enough: the following entry commits to the old one."""
    log.append("key-1", "reindex")
    log.append("key-1", "bundle_swap")
    log.append("key-1", "reindex")
    entries = log.read()
    forged = entries[1].model_copy(update={"action": "innocent"})
    rehashed = AuditEntry.model_validate(
        {
            **forged.payload(),
            "hash": AuditEntry.model_validate(
                {**forged.payload(), "hash": GENESIS}
            ).expected_hash(),
        }
    )

    def replace(lines: list[str]) -> list[str]:
        return [
            lines[0],
            json.dumps(rehashed.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
            lines[2],
        ]

    _rewrite(log.path, replace)
    with pytest.raises(AuditError, match="entry 2: prev does not match"):
        log.verify()


def test_deleting_an_entry_is_caught_as_a_sequence_gap(log: AuditLog) -> None:
    for _ in range(3):
        log.append("key-1", "reindex")
    _rewrite(log.path, lambda lines: [lines[0], lines[2]])
    with pytest.raises(AuditError, match="sequence number"):
        log.verify()


def test_reordering_two_entries_is_caught(log: AuditLog) -> None:
    log.append("key-1", "reindex")
    log.append("key-2", "bundle_swap")
    _rewrite(log.path, lambda lines: [lines[1], lines[0]])
    with pytest.raises(AuditError, match="sequence number"):
        log.verify()


def test_a_line_that_is_not_a_valid_entry_is_rejected_with_its_line_number(log: AuditLog) -> None:
    log.append("key-1", "reindex")
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    with pytest.raises(AuditError, match="line 2"):
        log.read()


def test_an_entry_with_a_malformed_hash_field_is_rejected(log: AuditLog) -> None:
    log.append("key-1", "reindex")

    def forge(lines: list[str]) -> list[str]:
        body = json.loads(lines[0])
        body["hash"] = "not-a-digest"
        return [json.dumps(body, sort_keys=True, separators=(",", ":"))]

    _rewrite(log.path, forge)
    with pytest.raises(AuditError, match="line 1"):
        log.read()


@settings(max_examples=25, deadline=None)
@given(
    actions=st.lists(
        st.sampled_from(["reindex", "bundle_swap", "bundle_swap_rejected"]),
        min_size=1,
        max_size=8,
    )
)
def test_any_chain_verifies_and_every_prefix_hash_links_forward(
    tmp_path_factory: pytest.TempPathFactory, actions: list[str]
) -> None:
    path = tmp_path_factory.mktemp("chain") / "admin.jsonl"
    chain = AuditLog(path)
    entries = [chain.append(f"key-{index}", action) for index, action in enumerate(actions)]
    assert chain.verify() == len(actions)
    previous = GENESIS
    for index, entry in enumerate(entries):
        assert entry.seq == index
        assert entry.prev == previous
        assert entry.hash == entry.expected_hash()
        previous = entry.hash
    assert chain.head == entries[-1].hash


@settings(max_examples=20, deadline=None)
@given(
    count=st.integers(min_value=2, max_value=6),
    target=st.integers(min_value=0, max_value=5),
)
def test_flipping_any_single_entry_is_always_detected(
    tmp_path_factory: pytest.TempPathFactory, count: int, target: int
) -> None:
    path = tmp_path_factory.mktemp("tamper") / "admin.jsonl"
    chain = AuditLog(path)
    for index in range(count):
        chain.append(f"key-{index}", "reindex", {"n": index})
    index = target % count
    lines = path.read_text(encoding="utf-8").splitlines()
    body = json.loads(lines[index])
    body["details"] = {"n": 9_999}
    lines[index] = json.dumps(body, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(AuditError):
        chain.verify()


# --- pseudonymizer ---


def test_the_pseudonymizer_is_stable_for_a_key_and_distinct_across_users() -> None:
    pseudonym = UserPseudonymizer(key=b"k" * 32)
    assert pseudonym(17) == pseudonym(17)
    assert pseudonym(17) != pseudonym(18)
    assert len(pseudonym(17)) == 12
    assert pseudonymous_user_id(17, b"k" * 32) == pseudonym(17)


def test_a_different_key_gives_a_different_digest() -> None:
    assert UserPseudonymizer(key=b"a" * 32)(5) != UserPseudonymizer(key=b"b" * 32)(5)


def test_the_default_key_is_random_per_instance() -> None:
    assert UserPseudonymizer()(5) != UserPseudonymizer()(5)


def test_the_digest_is_keyed_so_a_small_id_space_cannot_be_enumerated() -> None:
    """An unkeyed hash of a user id is reversible by brute force; an HMAC is not."""
    unkeyed = hashlib.sha256(b"42").hexdigest()[:12]
    assert UserPseudonymizer(key=b"k" * 32)(42) != unkeyed


@pytest.mark.parametrize("length", [7, 65])
def test_the_pseudonym_length_is_bounded(length: int) -> None:
    with pytest.raises(ValueError, match="length must be"):
        UserPseudonymizer(length=length)
