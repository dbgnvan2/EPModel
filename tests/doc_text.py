"""Reading helpers shared by the document-claim tests.

Purpose: one definition of a document's own status line, its `▶ STATE` line, its sentences and the tracked files, so
         the tests in `docs/plan_doc_claim_tests.md` cannot disagree about what they read.
Spec:    docs/plan_doc_claim_tests.md#D2, #D5a, #D5b
Tests:   tests/test_doc_claims.py, tests/test_stale_text.py, tests/test_doc_status.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# A status line is the front matter's `status:` or the first bold `**Status: …**`, within the opening lines.
_STATUS_HEAD_LINES = 12
_FRONT_STATUS = re.compile(r"^status:\s*(.+)$", re.M)
_BOLD_STATUS = re.compile(r"\*\*Status:\s*(.+?)\*\*", re.S)
_STATE_LINE = re.compile(r"^> ## ▶ STATE.*$", re.M)
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


def own_status(text: str) -> str | None:
    """The document's own status text, with blockquote markers and line breaks flattened, or None."""
    head = "\n".join(text.splitlines()[:_STATUS_HEAD_LINES])
    match = _FRONT_STATUS.search(head) or _BOLD_STATUS.search(head)
    return re.sub(r"\s*\n>?\s*", " ", match.group(1)).strip() if match else None


def state_line(text: str) -> str | None:
    """`_STATUS.md`'s `▶ STATE` line: the run status as it stands, above the dated log."""
    match = _STATE_LINE.search(text)
    return match.group(0) if match else None


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
