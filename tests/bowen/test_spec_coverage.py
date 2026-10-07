"""The spec coverage report is current.

Purpose: fail when docs/spec_coverage.md no longer matches what the generator
         produces from the spec, the tests and the overrides — a stale coverage
         report is a status claim that does not reconcile with its artifact (P6).
Spec:    docs/bowen_agent_model_spec_v2.md#M14.1, #M14.2
Tests:   this file
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("spec_coverage_tool", REPO / "tools" / "spec_coverage.py")
TOOL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TOOL)
OVERRIDES = REPO / "tools" / "spec_coverage_phase_b.json"


def test_m141_spec_coverage_is_current():
    expected = TOOL.report(json.loads(OVERRIDES.read_text(encoding="utf-8")), OVERRIDES.name)
    actual = (REPO / "docs" / "spec_coverage.md").read_text(encoding="utf-8")
    assert actual == expected, "regenerate: python3 tools/spec_coverage.py tools/spec_coverage_phase_b.json docs/spec_coverage.md"


def test_m141_every_done_names_an_existing_test():
    rows = TOOL.coverage(json.loads(OVERRIDES.read_text(encoding="utf-8")))
    for row in rows:
        if row["status"] != "done":
            continue
        assert row["evidence"], row["id"]
        for item in row["evidence"]:
            path, _, name = item.partition("::")
            source = (REPO / path).read_text(encoding="utf-8")
            assert not name or f"def {name}(" in source, f"{row['id']}: {item} does not exist"
