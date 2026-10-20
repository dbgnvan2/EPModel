"""No item the owner has decided is still described as pending owner review (version 2 stage 1, S1.9 rider (i)).

Purpose: version 1 left ten comments and docstrings in `src/`, `config/` and `tools/` reading "Claude's recommendation
         pending owner review" after the owner had decided each item (spec revision 32's review, 2026-10-17;
         `docs/version_1_dispositions.md`; `docs/version_2_plan.md`). A status line is a claim (P6), so this test fails
         when the phrase appears there again. An item genuinely awaiting the owner is listed in ``PENDING``, with where
         its decision will be recorded; `tools/version_1_status.py` is exempt because its matches are its rules.
Spec:    docs/implementation_plan_version_2.md#S1.9
Tests:   this file
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = ("src", "config", "tools")
EXEMPT = {"tools/version_1_status.py"}  # its rules match the phrase in the dispositions it reads
SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".txt"}
# The phrase, across a line break and a comment marker as a wrapped comment or docstring splits it.
PHRASE = re.compile(r"pending(?:\s|#)+owner(?:\s|#)+review", re.IGNORECASE)
# Path -> how many matches it may hold, for items the owner has not decided yet. Empty at stage 1.
PENDING: dict[str, int] = {}


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


def test_s1_9_no_pending_review_text_for_decided_items():
    found = stale_text()
    assert {p: n for p, n in found.items() if n > PENDING.get(p, 0)} == {}, found


def test_s1_9_the_scan_finds_a_wrapped_phrase(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("# Claude's recommendation, pending\n# owner review: x\n")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "b.md").write_text("pending owner review\n")
    assert stale_text(tmp_path) == {"src/a.py": 1, "config/b.md": 1}
