"""One helper parses a generated record's JSON block (version 2 stage 1, S1.5d).

Purpose: before stage 1 the block's pattern was pasted into twelve places across ``tools/`` and ``tests/``, so a
         change to how records end would have had to find every copy. This test fails on a second copy.
Spec:    docs/implementation_plan_version_2.md#S1.5d
Tests:   this file
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.record_json import record_json

REPO = Path(__file__).resolve().parents[2]
FENCE = "`" * 3  # built, so this file does not hold the text it searches for
NEEDLES = (FENCE + "json\\n(.*?)\\n" + FENCE,  # the pattern, as written in a raw string
           'split("' + FENCE + 'json"')     # the same parse by splitting


def test_s1_5d_one_record_parser():
    copies = [(path.relative_to(REPO).as_posix(), needle)
              for path in sorted((REPO / "tools").rglob("*.py")) + sorted((REPO / "tests").rglob("*.py"))
              for needle in NEEDLES for _ in range(path.read_text(encoding="utf-8").count(needle))]
    assert copies == [("tools/record_json.py", NEEDLES[0])], copies
    record = f"# A record\n\n{FENCE}json\n[{{\"criterion\": \"M11.C.1\"}}]\n{FENCE}\n"
    assert record_json(record) == [{"criterion": "M11.C.1"}]
    with pytest.raises(ValueError, match="no json block"):
        record_json("# A record with no block\n")
