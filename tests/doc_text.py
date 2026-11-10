"""Reading helpers shared by the document-claim tests.

Purpose: one definition of a document's sentences, its blocks and the tracked files, so the tests that read
         documents cannot disagree about what they read. The status-line and `▶ STATE` readers went with D5a and
         D5b, retired 2026-11-10 under the effort budget's rule 5 (no test asserts on prose).
Spec:    docs/plan_doc_claim_tests.md#D2
Tests:   tests/test_doc_claims.py, tests/test_asr_quotations.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Units a sentence never crosses: blank lines, table rows and list items.
_BLOCK = re.compile(r"\n\s*\n|\n(?=\s*(?:\||[-*] |\d+\. ))")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z*`(\[])")


def claims_data(repo: Path = REPO) -> dict:
    return json.loads((repo / "tests" / "doc_claims.json").read_text(encoding="utf-8"))


def phrase_pattern(phrases: list[str]) -> re.Pattern:
    """Any of the phrases, case-insensitive, each bounded as a whole word where its edge is a word character."""
    def bounded(phrase: str) -> str:
        return ((r"\b" if phrase[0].isalnum() else "") + re.escape(phrase).replace(r"\ ", r"\s+")
                + (r"\b" if phrase[-1].isalnum() else ""))
    return re.compile("|".join(bounded(p) for p in phrases), re.IGNORECASE)


def sentences(text: str) -> list[str]:
    """Sentences, each kept inside its own paragraph, list item or table row."""
    out = []
    for block in _BLOCK.split(text):
        out += [s for s in _SENTENCE_END.split(re.sub(r"\s*\n\s*", " ", block).strip()) if s]
    return out


def blocks(text: str) -> list[tuple[int, int]]:
    """The (start, end) of each paragraph, list item and table row, in order."""
    out, head = [], 0
    for match in _BLOCK.finditer(text):
        out.append((head, match.start()))
        head = match.end()
    return out + [(head, len(text))]


def tracked_files(repo: Path = REPO) -> list[str]:
    """Every path git tracks. A stray untracked copy of a removed file must not count as present."""
    result = subprocess.run(["git", "ls-files", "-z"], cwd=repo, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"git ls-files failed; these tests need a git checkout:\n{result.stderr}"
    return [p for p in result.stdout.split("\0") if p]
