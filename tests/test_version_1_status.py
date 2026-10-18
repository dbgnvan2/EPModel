"""The version 1 status page is current, covers every criterion, and applies V1-V3 as the definition states.

Purpose: fail when docs/version_1_status.md is not what tools/version_1_status.py renders from the committed records
         and dispositions, when a spec criterion has no row, or when the tool would accept a failing premise as a
         finding, count a pending disposition as final, or pass a composite with no recorded mutation.
Spec:    docs/version_1_definition.md#1 (V1, V2, V3)
Tests:   this file
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("tools.version_1_status", REPO / "tools" / "version_1_status.py")
tool = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tool
_spec.loader.exec_module(tool)


def _by_key(rows):
    return {s.key: s for s in rows}


def test_v1_status_page_is_current():
    assert tool.render(tool.statuses()) == tool.PAGE.read_text(encoding="utf-8"), \
        "stale or edited: rerun python3 tools/version_1_status.py"


def test_v1_every_spec_criterion_has_a_status_row():
    rows = tool.statuses()
    bases = {s.key.split("[", 1)[0] for s in rows}
    assert set(tool.spec_criteria()) <= bases
    assert all(s.status for s in rows)


def _with_row(key, **changes):
    """The committed record rows with one row's outcome, or one readout's ``holds``, changed (a synthetic input)."""
    rows = []
    for k, source, row in tool.record_rows():
        if k == key:
            row = copy.deepcopy(row)
            for readout in changes.pop("fail_readouts", ()):
                next(r for r in row["readouts"] if r["readout"] == readout)["holds"] = False
            row.update(changes)
        rows.append((k, source, row))
    return rows


def _a_passing(cls):
    return next(k for k, _, r in tool.record_rows() if r["class"] == cls and r["outcome"] == "PASS")


def test_v1_a_failing_premise_cannot_be_accepted_as_a_finding():
    """V2: a failing premise is a defect. Adversarial: the disposition that would make it look settled is refused.
    The failing row is synthetic, so the test does not depend on which premises fail today."""
    key = _a_passing("premise")
    fake = {**tool.dispositions(), key: {"kind": "accepted finding", "owner": True, "decided": "owner 2026-10-17",
                                         "reason": "x"}}
    with mock.patch.object(tool, "record_rows", return_value=_with_row(key, outcome="FAIL")), \
            mock.patch.object(tool, "dispositions", return_value=fake), pytest.raises(SystemExit, match="V2"):
        tool.statuses()


def test_v1_a_mixed_criterions_failing_premise_limb_blocks_v2_and_cannot_be_accepted():
    """A mixed criterion whose declared premise limb does not hold is a failing premise: with `M11.C.22`'s severe
    limb failing, its proposed acceptance is refused; with no disposition, the row blocks V2."""
    rows = _with_row("M11.C.22", outcome="FAIL", fail_readouts=("severe_removal_effect",))
    with mock.patch.object(tool, "record_rows", return_value=rows), pytest.raises(SystemExit, match="premise limb"):
        tool.statuses()
    bare = {k: v for k, v in tool.dispositions().items() if k != "M11.C.22"}
    with mock.patch.object(tool, "record_rows", return_value=rows), mock.patch.object(tool, "dispositions",
                                                                                      return_value=bare):
        row = _by_key(tool.statuses())["M11.C.22"]
    assert "V2" in row.blocks and not row.final


def test_v1_a_disposition_that_cannot_apply_is_refused():
    """P2: a disposition is never skipped silently. Out of scope on a built criterion, and a base id whose rows are
    all variants, both raise."""
    real = tool.dispositions()
    built = {**real, "M11.C.5": {**real["M11.C.5"], "kind": "out of scope"}}
    with mock.patch.object(tool, "dispositions", return_value=built), pytest.raises(SystemExit, match="out of scope"):
        tool.statuses()
    base = {**real, "M11.C.27": {"kind": "accepted finding", "owner": True, "decided": "owner 2026-10-17",
                                 "reason": "x"}}
    with mock.patch.object(tool, "dispositions", return_value=base), pytest.raises(SystemExit, match="variants"):
        tool.statuses()


def test_v1_a_row_after_a_break_in_the_dispositions_table_raises(tmp_path):
    text = tool.DISPOSITIONS.read_text(encoding="utf-8")
    first = text.index("| `M11.C.13` |")
    broken = tmp_path / "dispositions.md"
    broken.write_text(text[:first] + "\n" + text[first:], encoding="utf-8")
    with mock.patch.object(tool, "DISPOSITIONS", broken), pytest.raises(SystemExit, match="break"):
        tool.dispositions()


def test_v1_a_registered_criterion_with_no_record_row_means_stale_records():
    with mock.patch.object(tool, "registered", return_value=tool.registered() | {"M11.C.1[not run]"}), \
            pytest.raises(SystemExit, match="stale"):
        tool.statuses()


def test_v1_a_pending_acceptance_is_not_final_and_an_owner_acceptance_is():
    real = tool.dispositions()
    assert not real["M11.C.5"]["owner"], "this test reads M11.C.5 as pending; pick another pending row"
    pending = _by_key(tool.statuses())["M11.C.5"]
    assert not pending.final and pending.blocks == {"V1"}
    owned = {**real, "M11.C.5": {**real["M11.C.5"], "owner": True, "decided": "owner 2026-10-17"}}
    with mock.patch.object(tool, "dispositions", return_value=owned):
        decided = _by_key(tool.statuses())["M11.C.5"]
    assert decided.final and not decided.blocks


def test_v1_a_passing_composite_with_no_red_mutant_blocks_v3():
    """Adversarial: with every recorded mutant removed, no passing composite or mixed row may count as final."""
    with mock.patch.object(tool, "red_mutants", return_value={}):
        rows = tool.statuses()
    passing = [s for s in rows if s.verdict == "PASS" and s.cls in tool.DIRECTIONAL]
    assert passing and all("V3" in s.blocks and not s.final for s in passing)


def test_v1_a_disposition_on_a_passing_criterion_is_refused():
    """P6: a disposition that no longer applies must be removed, not left standing beside a pass."""
    fake = {**tool.dispositions(), "M11.C.16": {"kind": "accepted finding", "owner": True,
                                                 "decided": "owner 2026-10-17", "reason": "x"}}
    with mock.patch.object(tool, "dispositions", return_value=fake), pytest.raises(SystemExit, match="passes"):
        tool.statuses()


def test_v1_a_spec_criterion_no_record_lists_is_flagged_not_built():
    """A criterion that is in the spec and in no record or not-built list (as `M11.C.46` was) gets a row that says
    so, and a premise among them blocks V2."""
    extra = {**tool.spec_criteria(), "M11.C.99": "premise"}
    with mock.patch.object(tool, "spec_criteria", return_value=extra):
        row = _by_key(tool.statuses())["M11.C.99"]
    assert row.status == "not built" and "no record lists it" in row.evidence and "V2" in row.blocks
