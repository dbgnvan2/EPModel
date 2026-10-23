"""The review checklist names every document, and every name in it resolves.

Purpose: keep docs/SPEC_REVIEW_CHECKLIST.md a complete index of docs/, config/bowen/ and config/sweep/, so a
         reviewer reading from it never misses a document and never follows a dead link.
Spec:    CLAUDE.md — "Documentation stays current with the build"
Tests:   this file

These assert on documents only; nothing here imports `src/`.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
CHECKLIST = DOCS / "SPEC_REVIEW_CHECKLIST.md"
COVERED_ROOTS = (DOCS, REPO / "config" / "bowen", REPO / "config" / "sweep")  # config/sweep since stage 2 (D12)

# Markdown link targets: `](<path with spaces>)` or `](path)`.
_LINK = re.compile(r"\]\(<([^>]+)>\)|\]\(([^)\s]+)\)")


def _link_targets() -> list[Path]:
    text = CHECKLIST.read_text(encoding="utf-8")
    targets = []
    for angle, plain in _LINK.findall(text):
        target = angle or plain
        if "://" in target:
            continue
        targets.append((DOCS / target.split("#")[0]).resolve())
    return targets


def _documents() -> list[Path]:
    found = []
    for root in COVERED_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path == CHECKLIST:
                continue
            if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(REPO).parts):
                continue
            found.append(path.resolve())
    return found


def test_every_checklist_link_exists():
    missing = [str(t.relative_to(REPO)) for t in _link_targets() if not t.exists()]
    assert not missing, f"checklist links to files that do not exist: {missing}"


def test_every_document_is_in_the_checklist():
    targets = _link_targets()
    files = {t for t in targets if t.is_file()}
    dirs = [t for t in targets if t.is_dir()]
    unlisted = [
        str(doc.relative_to(REPO))
        for doc in _documents()
        if doc not in files and not any(d in doc.parents for d in dirs)
    ]
    assert not unlisted, (
        "documents missing from docs/SPEC_REVIEW_CHECKLIST.md — add each to a review section "
        f"or to 'Not for review': {sorted(unlisted)}"
    )
