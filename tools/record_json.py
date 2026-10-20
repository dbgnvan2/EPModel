"""Read the machine-readable JSON block a generated record ends with.

Purpose: one parser for the ``json`` block every generated record carries, so the block's pattern is written once
         (version 2 stage 1, S1.5d; before it, twelve copies across ``tools/`` and ``tests/``).
Spec:    docs/implementation_plan_version_2.md#S1.5d
Tests:   tests/bowen/test_record_json.py::test_s1_5d_one_record_parser
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BLOCK = re.compile(r"```json\n(.*?)\n```", re.S)


def record_json(source: str | Path):
    """The first JSON block in a record, given its text or its path; a record with none raises ValueError."""
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    found = BLOCK.search(text)
    if found is None:
        raise ValueError("the record has no json block")
    return json.loads(found.group(1))
