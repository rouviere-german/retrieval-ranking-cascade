"""Hash-chained audit log for privileged operations (reindex, bundle swap).

Every entry commits to its predecessor: ``hash = sha256(canonical_json(entry without hash))``
and the entry carries ``prev``, the previous entry's hash. Deleting, reordering or editing any
entry breaks the chain at that point, so :meth:`AuditLog.verify` localizes tampering instead of
only detecting it. The chain starts from ``GENESIS`` (64 zeros).

The log is append-only JSONL. Appends take the sidecar file lock from ``core.fileio``, so
concurrent workers cannot interleave a read-head/append pair, and each line is fsynced before the
lock is released. Details are passed through ``redact_value`` and user ids through
:func:`pseudonymous_user_id`, so an audit trail never becomes a second copy of the payload.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from retrieval_ranking_cascade.core.canonical import canonical_json, sha256_hex
from retrieval_ranking_cascade.core.fileio import file_lock
from retrieval_ranking_cascade.core.redaction import redact_value

__all__ = [
    "GENESIS",
    "AuditEntry",
    "AuditError",
    "AuditLog",
    "UserPseudonymizer",
    "verify_file",
]

GENESIS = "0" * 64
_HEX64 = r"^[0-9a-f]{64}$"


class AuditError(ValueError):
    """The audit chain is broken: bad JSON, wrong link, wrong hash or a sequence gap."""


class AuditEntry(BaseModel):
    """One audit record. ``hash`` covers every other field, including ``prev``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=0)
    prev: str = Field(pattern=_HEX64)
    recorded_at: str
    actor: str = Field(min_length=1, max_length=64)
    action: str = Field(min_length=1, max_length=64)
    details: dict[str, JsonValue]
    hash: str = Field(pattern=_HEX64)

    def payload(self) -> dict[str, JsonValue]:
        """The hashed part of the entry (everything but ``hash``)."""
        body: dict[str, JsonValue] = self.model_dump(mode="json")
        body.pop("hash")
        return body

    def expected_hash(self) -> str:
        return sha256_hex(canonical_json(self.payload()))


class UserPseudonymizer:
    """Keyed, truncated digests of user ids for logs and audit details.

    The key is random per process, so digests correlate requests inside one process lifetime but
    cannot be reversed by dictionary attack over a small id space, and they do not correlate
    across restarts or across services. A shared, persisted key would allow cross-restart
    correlation; that is a deliberate trade against re-identification risk.
    """

    def __init__(self, key: bytes | None = None, *, length: int = 12) -> None:
        if not 8 <= length <= 64:
            raise ValueError("length must be between 8 and 64 hex characters")
        self._key = key if key is not None else secrets.token_bytes(32)
        self._length = length

    def __call__(self, user_id: int) -> str:
        message = str(int(user_id)).encode("ascii")
        return hmac.new(self._key, message, "sha256").hexdigest()[: self._length]


def pseudonymous_user_id(user_id: int, key: bytes) -> str:
    """Module-level form of :class:`UserPseudonymizer` for callers that hold their own key."""
    return UserPseudonymizer(key)(user_id)


class AuditLog:
    """Append-only hash-chained log stored as JSONL at ``path``."""

    def __init__(self, path: Path, *, clock: Any = None) -> None:
        self.path = Path(path)
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))

    # --- reading ---

    def read(self) -> list[AuditEntry]:
        """Parse every entry without checking the chain (see :meth:`verify`)."""
        if not self.path.exists():
            return []
        entries: list[AuditEntry] = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entries.append(AuditEntry.model_validate_json(line))
            except ValueError as exc:
                raise AuditError(f"line {number}: not a valid audit entry") from exc
        return entries

    def verify(self) -> int:
        """Verify the whole chain and return the number of entries. Raises on any break."""
        expected_prev = GENESIS
        count = 0
        for index, entry in enumerate(self.read()):
            if entry.seq != index:
                raise AuditError(f"entry {index}: sequence number is {entry.seq}")
            if not hmac.compare_digest(entry.prev, expected_prev):
                raise AuditError(f"entry {index}: prev does not match the previous entry's hash")
            if not hmac.compare_digest(entry.hash, entry.expected_hash()):
                raise AuditError(f"entry {index}: contents do not match the recorded hash")
            expected_prev = entry.hash
            count = index + 1
        return count

    @property
    def head(self) -> str:
        """Hash of the last entry, or ``GENESIS`` for an empty log."""
        entries = self.read()
        return entries[-1].hash if entries else GENESIS

    # --- writing ---

    def append(
        self, actor: str, action: str, details: Mapping[str, JsonValue] | None = None
    ) -> AuditEntry:
        """Append one entry under the file lock and fsync it before returning."""
        safe_details: dict[str, JsonValue] = redact_value(dict(details or {}))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.path):
            existing = self.read()
            payload: dict[str, JsonValue] = {
                "seq": len(existing),
                "prev": existing[-1].hash if existing else GENESIS,
                "recorded_at": self._clock().isoformat(),
                "actor": actor,
                "action": action,
                "details": safe_details,
            }
            entry = AuditEntry.model_validate(
                {**payload, "hash": sha256_hex(canonical_json(payload))}
            )
            line = json.dumps(entry.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return entry


def verify_file(path: Path) -> int:
    """Verify the chain in ``path`` and return the entry count."""
    return AuditLog(path).verify()
