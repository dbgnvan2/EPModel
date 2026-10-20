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
    from tools.record_json import record_json

    text = record_text()
    assert "Phase C" not in text
    # Variants carry a bracketed name (M11.C.14[firmness], revision 31).
    table = dict(re.findall(r"^\| `(M11\.C\.\d+(?:\[[^\]]+\])?)` \| \w+ \| \*\*(\w+)\*\* \|", text, re.M))
    block = record_json(text)
    assert table == {v["criterion"]: v["outcome"] for v in block} and table


# --- Part II, the life course (plan §4.4; spec revision 28) ---


def record_text_2() -> str:
    return (REPO / "docs" / "phase_d2_ensemble_record.md").read_text(encoding="utf-8")


def test_phase_d2_record_is_current():
    match = re.search(r"^code_hash: ([0-9a-f]{64})$", record_text_2(), re.M)
    assert match, "the record carries no code hash"
    assert match.group(1) == _tool._record.code_hash(*_tool.HASHED_TOOLS), \
        "stale: rerun python3 tools/phase_d_record.py --part 2"


def test_phase_d2_record_covers_every_criterion_and_names_what_is_not_run():
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2

    text = record_text_2()
    for criterion in CRITERIA_D2.values():
        assert f"| `{criterion.id}` |" in text, criterion.id
    for cid in _tool.NOT_RUN_2:
        assert f"- `{cid}` —" in text, cid
    assert "Phase C" not in text
