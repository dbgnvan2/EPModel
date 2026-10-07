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


def test_m141_no_id_resolves_to_an_unassigned_phase():
    """A new ID that no MODULE_PHASES pattern matches was reported "Phase unassigned"."""
    rows = TOOL.coverage(json.loads(OVERRIDES.read_text(encoding="utf-8")))
    unassigned = [r["id"] for r in rows if r["note"] == "Phase unassigned"]
    assert unassigned == [], f"IDs with no building phase: {unassigned}"


def test_m141_excluded_evidence_is_not_shown():
    """A test pinning a superseded revision must not be merged back in as evidence."""
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    named = TOOL.tests_for("M4.C.1", TOOL.named_tests())
    assert named, "no test is named for M4.C.1, so this guard checks nothing"
    excluded = named[0]
    overrides["M4.C.1"] = {"status": "partial", "note": "guard fixture", "exclude_evidence": [excluded]}
    rows = {r["id"]: r for r in TOOL.coverage(overrides)}
    assert excluded not in rows["M4.C.1"]["evidence"]
    assert set(named[1:]) <= set(rows["M4.C.1"]["evidence"])  # only the excluded one is dropped
