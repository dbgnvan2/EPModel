"""docs/STATUS.md is what tools/status.py writes from the current files (T-225).

Purpose: the one place a session reads the spec's revision, the requirement count and the engine hash; a hand edit or
         a source that moved without a rerun fails here.
Spec:    docs/todo/2026-11-11_whole_repo_review.md (T-225)
Tests:   this file
"""

from __future__ import annotations

from tools import status


def test_t225_status_page_is_current():
    assert status.OUT.read_text(encoding="utf-8") == status.render(status.facts()), \
        "stale: rerun python3 tools/status.py"
