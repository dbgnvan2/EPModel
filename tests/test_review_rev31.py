"""Spec revision 31, part A: the owner's review of revisions 24-30 applied to the documents.

Purpose: pin each documents-only change of revision 31 so a later edit cannot quietly undo it.
Spec:    docs/implementation_plan_phase_d.md#4.11 (R31.A1-R31.A12); docs/REVIEW_spec_revisions_24-30.md
Tests:   this file
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = (REPO / "docs" / "bowen_agent_model_spec_v2.md").read_text(encoding="utf-8")
EXPLAINER = (REPO / "docs" / "model_explainer.md").read_text(encoding="utf-8")
CHANGES = (REPO / "config" / "bowen" / "constants_changes.md").read_text(encoding="utf-8")
PLAN = (REPO / "docs" / "implementation_plan_phase_d.md").read_text(encoding="utf-8")


def register_row(field: str) -> str:
    """The `M14.A` register row for ``field``."""
    rows = [line for line in SPEC.splitlines() if line.startswith(f"| `{field}` |")]
    assert len(rows) == 1, (field, len(rows))
    return rows[0]


def criterion_row(cid: str) -> str:
    rows = [line for line in SPEC.splitlines() if line.startswith(f"| **{cid}** |")]
    assert len(rows) == 1, (cid, len(rows))
    return rows[0]


def requirement(rid: str) -> str:
    """The requirement that opens with ``**rid**``, up to the next requirement or heading (its bullets and closing
    paragraph included)."""
    match = re.search(rf"^\*\*{re.escape(rid)}\*\*.*?(?=\n\n\*\*M|\n\n#|\Z)", SPEC, flags=re.M | re.S)
    assert match, rid
    return match.group(0)


def test_r31a1_channel_focus_row_not_from_belief():
    row = register_row("channel_focus")
    assert "from belief" not in row
    assert "moves delivered" in row


def test_r31a2_tolerance_row_per_member():
    row = register_row("tolerance")
    assert "| Family |" in row
    assert "one declared value per member" in row


def test_r31a3_c6_limit_a_restated():
    row = criterion_row("M11.C.6")
    assert "**MUST NOT** complete in one or two generations" in row
    assert "**MUST** be achievable within roughly ten" not in row
    assert "open for the owner" not in row
    assert "Open conflict, for the owner" not in requirement("M7.E.1f")


def test_r31a3_explainer_marks_traversal_contested():
    lines = [line for line in EXPLAINER.splitlines() if line.startswith("| Full traversal |")]
    assert len(lines) == 1 and "`[X]`" in lines[0]


def test_r31a4_c14_row_restated():
    row = criterion_row("M11.C.14")
    assert "no-technique arm" in row
    assert "lowers" in row
    exception = [line for line in SPEC.splitlines() if line.startswith("| `M11.C.14` |")]
    assert len(exception) == 1 and "one-sided" in exception[0]


def test_r31a5_m4d6f_gap_mentions_felt_danger():
    text = requirement("M4.D.6f")
    assert "principled only when declared" not in text
    assert "M1.B.3e" in text


def test_r31a6_m7e1c_not_open():
    text = requirement("M7.E.1c")
    assert "open owner decision" not in text
    assert "not applied at `t0`" in text


def test_r31a7_c10_superseded_clauses_struck():
    row = criterion_row("M11.C.10")
    # Revision 11's and revision 29's mutation clauses, superseded at revision 30, are struck through.
    assert "~~`M1.D.8`'s family-of-origin input and `M4.D.6`'s learning from witnessed cutoff" in row
    assert "~~**Mutations:** `M4.D.6f` removed, and `M1.B.3d`'s reactive rise removed." in row


def test_r31a8_c2_settings_not_to_be_added():
    assert "are to be added to `config/bowen/criteria.md`" not in criterion_row("M11.C.2")


REVIEWED = re.compile(r"⟦rev(2[4-9]|30)\b[^⟧]*⟧")


def test_r31a9_no_pending_review_tags_for_rev24_30():
    tags = REVIEWED.findall(SPEC)
    assert tags, "no revision 24-30 tags found: the pattern is wrong"
    pending = [m.group(0) for m in REVIEWED.finditer(SPEC) if "pending owner review" in m.group(0)]
    assert pending == []
    assert "⟦rev28⟧" not in SPEC  # inconsistency 7: every revision 28 tag names its source
    assert "pending owner review" not in EXPLAINER
    assert "pending owner review" not in CHANGES


def test_r31a9_tag_check_catches_a_pending_tag():
    """Adversarial: the pattern must see a pending tag of the kind the review cleared."""
    sample = "text ⟦rev27 · Claude's recommendation 2026-10-12, pending owner review⟧ more"
    assert [m.group(0) for m in REVIEWED.finditer(sample) if "pending owner review" in m.group(0)]


def test_r31a10_launch_age_approved():
    text = requirement("M7.C.1g")
    assert "The owner has not stated a launch age" not in text
    assert "2026-10-16" in text


def test_r31a11_plan_closes_the_two_owner_questions():
    assert "| 9 | Focus steering's size (`focus_steer_gain` 1.0 has no measurable effect in Phase C) | owner decision |" not in PLAN
    assert "- **For the owner:** whether \"pain is necessary\" should read an active symptom as it does" not in PLAN


def test_r31a12_spec_status_names_revision_31():
    assert re.search(r"^version: 2\.0, revision 31$", SPEC, flags=re.M)
    assert "revision 31, the owner's review of revisions 24 to 30 applied" in SPEC.splitlines()[3]
