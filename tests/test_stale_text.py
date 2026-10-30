"""No item the owner has decided is still described as pending owner review (version 2 stage 1, S1.9 rider (i)).

Purpose: version 1 left ten comments and docstrings in `src/`, `config/` and `tools/` reading "Claude's recommendation
         pending owner review" after the owner had decided each item (spec revision 32's review, 2026-10-17;
         `docs/version_1_dispositions.md`; `docs/version_2_plan.md`). A status line is a claim (P6), so this test fails
         when the phrase appears there again. An item genuinely awaiting the owner is listed in ``PENDING``, with where
         its decision will be recorded; `tools/version_1_status.py` is exempt because its matches are its rules.
         D5a adds the documents that state the current status: README.md, CLAUDE.md, the checklist, `_STATUS.md`'s
         `▶ STATE` line and every `docs/` file's own status line. Dated logs (CHANGELOG, TODO, the rest of
         `_STATUS.md`, plans and records) are history and are not scanned.
Spec:    docs/implementation_plan_version_2.md#S1.9; docs/plan_doc_claim_tests.md#D5a
Tests:   this file
"""

from __future__ import annotations

import re
from pathlib import Path

from doc_text import own_status, state_line

REPO = Path(__file__).resolve().parents[1]
ROOTS = ("src", "config", "tools")
EXEMPT = {"tools/version_1_status.py"}  # its rules match the phrase in the dispositions it reads
SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".txt"}
# The phrase, across a line break and a comment marker as a wrapped comment or docstring splits it.
PHRASE = re.compile(r"pending(?:\s|#)+owner(?:\s|#)+review", re.IGNORECASE)
# Path -> how many matches it may hold, for items the owner has not decided yet. Empty at stage 1.
# Version 2's dispositions (2026-10-29) are Claude's recommendations awaiting the owner; each decision is recorded in
# that file's `decided` cell, and both entries go when every row there is the owner's.
PENDING: dict[str, int] = {"docs/SPEC_REVIEW_CHECKLIST.md": 1, "docs/version_2_dispositions.md (status line)": 1}
# D5a: documents read whole, and the one whose state line alone is current; every other `docs/` file is read only for
# its own status line.
CURRENT_STATE_DOCS = ("README.md", "CLAUDE.md", "docs/SPEC_REVIEW_CHECKLIST.md")
STATE_LINE_DOC = "docs/theory/_STATUS.md"


def stale_text(repo: Path = REPO) -> dict[str, int]:
    found = {}
    for root in ROOTS:
        for path in sorted((repo / root).rglob("*")):
            rel = path.relative_to(repo).as_posix()
            if not path.is_file() or path.suffix not in SUFFIXES or "__pycache__" in path.parts or rel in EXEMPT:
                continue
            count = len(PHRASE.findall(path.read_text(encoding="utf-8", errors="replace")))
            if count:
                found[rel] = count
    return found


def current_state_text(repo: Path = REPO) -> dict[str, int]:
    """D5a: matches in the documents that state the current status, keyed as `PENDING` is."""
    texts = {doc: (repo / doc).read_text(encoding="utf-8") for doc in CURRENT_STATE_DOCS if (repo / doc).is_file()}
    if (repo / STATE_LINE_DOC).is_file():
        texts[f"{STATE_LINE_DOC} (▶ STATE line)"] = state_line((repo / STATE_LINE_DOC).read_text(encoding="utf-8")) or ""
    for path in sorted((repo / "docs").rglob("*.md")):
        rel = path.relative_to(repo).as_posix()
        if rel not in CURRENT_STATE_DOCS and rel != STATE_LINE_DOC:
            texts[f"{rel} (status line)"] = own_status(path.read_text(encoding="utf-8")) or ""
    return {key: n for key, text in texts.items() if (n := len(PHRASE.findall(text)))}


def test_s1_9_no_pending_review_text_for_decided_items():
    found = stale_text()
    assert {p: n for p, n in found.items() if n > PENDING.get(p, 0)} == {}, found


def test_s1_9_the_scan_finds_a_wrapped_phrase(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("# Claude's recommendation, pending\n# owner review: x\n")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "b.md").write_text("pending owner review\n")
    assert stale_text(tmp_path) == {"src/a.py": 1, "config/b.md": 1}


def test_d5a_current_state_documents_carry_no_stale_pending_review():
    """D5a: the real repo. An item truly awaiting the owner is listed in `PENDING`."""
    found = current_state_text()
    assert {p: n for p, n in found.items() if n > PENDING.get(p, 0)} == {}, found


def test_d5a_the_scan_reads_status_lines_and_the_state_line(tmp_path):
    """D5a: one match in each new place is found; the same phrase in a dated log or below a status line is not."""
    (tmp_path / "docs" / "theory").mkdir(parents=True)
    (tmp_path / "README.md").write_text("Spec revisions 24–27 are Claude's recommendations, pending owner\nreview.\n")
    (tmp_path / "CLAUDE.md").write_text("- pending owner review\n")
    (tmp_path / "docs" / "SPEC_REVIEW_CHECKLIST.md").write_text("**For review:** [a.md](a.md) — pending owner review\n")
    (tmp_path / "docs" / "theory" / "_STATUS.md").write_text(
        "# Run status\n\n> ## ▶ STATE — stage 2, pending owner review.\n>\n> **2026-10-01** — pending owner review.\n")
    (tmp_path / "docs" / "plan.md").write_text("# Plan\n\n**Status: DRAFT, pending owner review.**\n")
    (tmp_path / "docs" / "record.md").write_text("# Record\n\n**Status: built.**\n\n" + "x\n" * 20
                                                 + "Item 4, pending owner review.\n")
    (tmp_path / "CHANGELOG.md").write_text("- 2026-10-01: pending owner review\n")
    assert current_state_text(tmp_path) == {
        "README.md": 1, "CLAUDE.md": 1, "docs/SPEC_REVIEW_CHECKLIST.md": 1,
        "docs/theory/_STATUS.md (▶ STATE line)": 1, "docs/plan.md (status line)": 1,
    }
