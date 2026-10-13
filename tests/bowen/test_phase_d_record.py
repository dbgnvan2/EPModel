"""The Phase D ensemble record is current and complete (Phase D plan step 12).

Purpose: fail when the committed record's hash no longer matches the code and config, or when it misses a built
         criterion or a not-built one.
Spec:    docs/bowen_agent_model_spec_v2.md#M13.4, #M17.A.1
Tests:   this file
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("tools.phase_d_record", REPO / "tools" / "phase_d_record.py")
_tool = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _tool
_spec.loader.exec_module(_tool)


def record_text() -> str:
    return (REPO / "docs" / "phase_d_ensemble_record.md").read_text(encoding="utf-8")


def test_phase_d_record_is_current():
    match = re.search(r"^code_hash: ([0-9a-f]{64})$", record_text(), re.M)
    assert match, "the record carries no code hash"
    assert match.group(1) == _tool._record.code_hash(*_tool.HASHED_TOOLS), "stale: rerun python3 tools/phase_d_record.py"


def test_phase_d_record_covers_every_criterion_and_names_what_is_not_built():
    from src.bowen.ensemble.criteria_d import CRITERIA_D

    text = record_text()
    for cid in CRITERIA_D:
        assert f"| `{cid}` |" in text, cid
    for cid in _tool.NOT_BUILT:
        assert f"- `{cid}` —" in text, cid
        assert cid not in CRITERIA_D


def test_phase_d_record_carries_no_phase_c_wording_and_its_table_matches_its_json():
    """Review 2026-10-13: the page is the Phase C renderer's, reworded; none of it may be left, and the verdict table and
    the machine-readable block must agree."""
    import json

    text = record_text()
    assert "Phase C" not in text
    table = dict(re.findall(r"^\| `(M11\.C\.\d+)` \| \w+ \| \*\*(\w+)\*\* \|", text, re.M))
    block = json.loads(text.split("```json", 1)[1].split("```", 1)[0])
    assert table == {v["criterion"]: v["outcome"] for v in block} and table
