"""Every ``path.py::symbol`` citation in the shipped code and prose must resolve.

A docstring that cites a benchmark or a test is a claim about this repository, and a citation that
points at nothing is the cheapest kind of dishonesty to ship: it reads like evidence, and nothing
checks it. ``scripts/check_baseline.py`` resolves the references in CLAIMS.md and
docs/THREAT_MODEL.md; this test covers everything else - the package's own docstrings, the benches,
the scripts and the prose files - so a renamed test or a deleted benchmark cannot leave a dangling
"measured in ..." behind it.

``tests/`` is deliberately not scanned: ``tests/integration/test_check_baseline.py`` builds broken
references on purpose to prove the baseline checker catches them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# `path.py` or ``path.py::symbol``, in Markdown single backticks or reStructuredText doubles.
CITATION = re.compile(
    r"`{1,2}((?:src|tests|bench|scripts)/[A-Za-z0-9_./-]+\.py)(?:::([A-Za-z_][A-Za-z0-9_]*))?`{1,2}"
)
# A citation must point at something a reader can run or read: a function, or a class.
DEFINITION = "^\\s*(?:async\\s+)?(?:def|class)\\s+{symbol}\\b"
# `tests/...py` in scripts/check_baseline.py's own docstring is the shape of a reference, not one.
PLACEHOLDER = "..."
PROSE = ("README.md", "CLAIMS.md", "MODEL_CARD.md", "CONTRIBUTING.md", "SECURITY.md")
# Enough of the repo's citations to make a silently non-matching pattern fail this test too.
MINIMUM_CITATIONS = 150


def _scanned_files(root: Path) -> list[Path]:
    files = [root / name for name in PROSE]
    files += sorted((root / "docs").rglob("*.md"))
    files += sorted((root / "src").rglob("*.py"))
    files += sorted((root / "bench").glob("*.py"))
    files += sorted((root / "scripts").glob("*.py"))
    files.append(root / "tasks.py")
    return [path for path in files if path.is_file()]


def _citations(root: Path) -> list[tuple[Path, str, str | None]]:
    found: list[tuple[Path, str, str | None]] = []
    for path in _scanned_files(root):
        text = path.read_text(encoding="utf-8")
        for match in CITATION.finditer(text):
            if PLACEHOLDER in match.group(1):
                continue
            found.append((path.relative_to(root), match.group(1), match.group(2)))
    return found


@pytest.mark.unit
def test_the_citation_scan_covers_the_repository(repo_root: Path) -> None:
    citations = _citations(repo_root)
    assert len(citations) >= MINIMUM_CITATIONS, f"only {len(citations)} citations matched"
    assert any(symbol for _, _, symbol in citations), "no ::symbol citation matched"
    assert {path.parts[0] for path, _, _ in citations} >= {"docs", "src"}


@pytest.mark.unit
def test_every_cited_file_and_symbol_exists(repo_root: Path) -> None:
    dangling: list[str] = []
    for source, reference, symbol in _citations(repo_root):
        target = repo_root / reference
        if not target.is_file():
            dangling.append(f"{source}: {reference} does not exist")
            continue
        if symbol is None:
            continue
        pattern = DEFINITION.format(symbol=re.escape(symbol))
        if not re.search(pattern, target.read_text(encoding="utf-8"), re.M):
            dangling.append(f"{source}: {reference}::{symbol} is not defined there")
    assert not dangling, "dangling citations:\n" + "\n".join(dangling)
