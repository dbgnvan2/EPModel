"""Consistency guards over the specification and the documents that quote it.

Purpose: turn a class of silent documentation drift into a failing test.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.10, #M11.D.11, #M11.D.13
Tests:   this file

These assert on *documents*, not on model code, so they do not anticipate the
implementation plan — nothing here imports `src/`. They exist because a
failure-pattern sweep on 2026-08-28 found fifteen defects in the document set,
of which the ones below were all mechanically detectable and none had a guard:

  * `M11.2` bounded ensemble testing at `M11.C.16` while the table grew to 33
    (a range bound over a growing set — P29).
  * Seven `M11.C` criteria and one `M16.T` criterion carried a phase in their
    table row and appeared in no phase's exit condition, so the phase could be
    declared done with the test unwritten (P21/P25).
  * Nine prohibitions were written `No X MUST Y`, which under the document's own
    normative vocabulary (§0.3) states a permission (P19).
  * The published proposal carried revision 4's requirement count through
    revisions 5 and 6, while a status line asserted the document set was
    consistent (P6).

Each test below is written so that reverting the corresponding fix turns it red.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "docs" / "bowen_agent_model_spec_v2.md"

# Documents that cite spec IDs and must not cite one that does not exist.
CITING_DOCS = [
    REPO / "docs" / "model_explainer.md",
    REPO / "docs" / "agent_model_proposal.html",
    REPO / "docs" / "theory" / "_STATUS.md",
    REPO / "CLAUDE.md",
    REPO / "README.md",
    REPO / "CHANGELOG.md",
    REPO / "TODO.md",
]

# A requirement definition: a bolded bare ID starting a line, or heading a table
# row. The optional list marker matters — §0's own requirements (M0.1–M0.4) are
# written as bullet items, and an extractor that missed them reported every
# reference to M0.4 as dangling.
_DEF_LINE = re.compile(r"^(?:[-*]\s+)?\*\*(M\d+(?:\.[A-Za-z0-9]+)*)\*\*", re.M)
_DEF_ROW = re.compile(r"^\| \*\*(M\d+(?:\.[A-Za-z0-9]+)*)\*\* \|", re.M)

# A reference: any ID whose last segment is numeric (with an optional lowercase
# suffix). That admits the two-level IDs §0.4 insists are IDs in their own right
# — M0.3, M11.2, M13.1 — while excluding module and part headings, whose last
# segment is a letter (M11, M11.C, M1.A).
#
# An earlier version required three levels or more. It passed its own mutation
# because renaming a two-level ID left every reference to it invisible to the
# scan — the exact shape of defect this file exists to catch.
_REF = re.compile(r"\b(M\d+(?:\.[A-Za-z0-9]+)*\.\d+[a-z]?)\b")


def _spec_text() -> str:
    return SPEC.read_text(encoding="utf-8")


def defined_ids(text: str) -> list[str]:
    return _DEF_LINE.findall(text) + _DEF_ROW.findall(text)


# --------------------------------------------------------------------------
# M11.D.10 — every ID machine-extractable by one pattern, and each used once
# --------------------------------------------------------------------------

def test_m11d10_spec_ids_are_unique():
    ids = defined_ids(_spec_text())
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert duplicates == [], f"requirement IDs defined more than once: {duplicates}"


def test_m11d10_spec_ids_match_the_single_pattern():
    """Extract loosely, then check strictly.

    Extracting with the strict pattern and then testing the results against it
    is a test that cannot fail — it was written that way first. The loose
    pattern catches anything bolded that starts like an ID, so a malformed one
    (`M1.A.3!`, `M1..4`, `M-1.A`) is collected and then rejected here.
    """
    loose = re.compile(r"^\*\*(M[0-9][^\s*]*)\*\*", re.M)
    strict = re.compile(r"^M\d+(\.[A-Za-z0-9]+)*$")
    bad = sorted({i for i in loose.findall(_spec_text()) if not strict.match(i)})
    assert bad == [], f"IDs not matching the §0.4 scheme: {bad}"


# --------------------------------------------------------------------------
# Cross-reference resolution — spec and every document that quotes it
# --------------------------------------------------------------------------

def test_every_spec_reference_resolves_within_the_spec():
    text = _spec_text()
    known = set(defined_ids(text))
    dangling = sorted({r for r in _REF.findall(text) if r not in known})
    assert dangling == [], f"spec references IDs it never defines: {dangling}"


def test_every_citing_document_resolves_against_the_spec():
    known = set(defined_ids(_spec_text()))
    failures = {}
    for doc in CITING_DOCS:
        dangling = sorted(
            {r for r in _REF.findall(doc.read_text(encoding="utf-8")) if r not in known}
        )
        if dangling:
            failures[doc.relative_to(REPO).as_posix()] = dangling
    assert failures == {}, f"documents cite spec IDs that do not exist: {failures}"


# --------------------------------------------------------------------------
# M11.D.13 — the normative vocabulary is used as §0.3 defines it
# --------------------------------------------------------------------------

def test_m11d13_no_inverted_prohibitions():
    """`No X MUST Y` states a permission under §0.3, not a prohibition."""
    inverted = re.findall(
        r"\*?\*?No\b[^.\n]{0,60}?\*\*MUST\*\*(?! NOT)", _spec_text()
    )
    assert inverted == [], (
        "prohibitions written in the inverted form 'No X MUST Y', which §0.3 "
        f"reads as a permission: {inverted}"
    )


# --------------------------------------------------------------------------
# M11.D.11 — every criterion gates the phase its own row names (M13.2a)
# --------------------------------------------------------------------------

def _criterion_phases(text: str) -> dict[str, str]:
    """Criterion ID -> the phase letter its table row assigns it."""
    phases = {}
    for cid, rest in re.findall(r"^\| \*\*(M11\.C\.\d+)\*\* \|(.*)$", text, re.M):
        cells = [c.strip() for c in rest.split(" | ")]
        if len(cells) >= 3 and cells[2] in {"B", "C", "D", "E"}:
            phases[cid] = cells[2]
    return phases


def _phase_exit_conditions(text: str) -> dict[str, str]:
    exits = {}
    for letter in "BCDEF":
        row = re.search(r"^\| \*\*%s\*\* \|(.*)$" % letter, text, re.M)
        if row:
            exits[letter] = row.group(1)
    return exits


def test_m11d11_every_criterion_gates_its_phase():
    text = _spec_text()
    exits = _phase_exit_conditions(text)
    ungated = sorted(
        (cid, phase)
        for cid, phase in _criterion_phases(text).items()
        if phase in exits and cid not in exits[phase]
    )
    assert ungated == [], (
        "criteria carry a phase in the M11.C table but appear in no phase exit "
        f"condition, so the phase can be declared done with them unwritten: {ungated}"
    )


def test_m11d11_log_criteria_gate_a_phase():
    """M16's tests must gate somewhere; M16.T.4 gated nothing when written."""
    text = _spec_text()
    exits = " ".join(_phase_exit_conditions(text).values())
    log_tests = sorted(set(re.findall(r"\bM16\.T\.\d+\b", text)))
    assert log_tests, "no M16.T criteria found — the table moved or was renamed"
    ungated = [t for t in log_tests if t not in exits]
    assert ungated == [], f"M16 log criteria gating no phase: {ungated}"


# --------------------------------------------------------------------------
# M11.2 — ensemble testing is stated over the whole set, not a stale range
# --------------------------------------------------------------------------

def test_m112_is_not_bounded_by_a_stale_numeric_range():
    """A range bound over a growing set silently stops covering it (P29)."""
    m112 = re.search(r"\*\*M11\.2\*\*[^\n]*", _spec_text())
    assert m112, "M11.2 not found"
    assert not re.search(r"M11\.C\.\d+\s*[–-]\s*M11\.C\.\d+", m112.group()), (
        "M11.2 bounds ensemble testing by a numeric range over M11.C; the table "
        "grows, so the bound stops covering it. State it over the whole set."
    )


# --------------------------------------------------------------------------
# P6 — status claims reconcile to the artifact they describe
# --------------------------------------------------------------------------

def test_requirement_counts_agree_with_the_spec():
    actual = len(set(defined_ids(_spec_text())))
    claimed = {}
    for doc in CITING_DOCS:
        for n in re.findall(
            r"(\d{3,4})\s+(?:numbered requirements|unique IDs)",
            doc.read_text(encoding="utf-8"),
        ):
            claimed.setdefault(doc.relative_to(REPO).as_posix(), set()).add(int(n))
    # Documents legitimately cite earlier, smaller counts as history ("that left
    # the spec at 321"). The *largest* count a document states is its claim about
    # the spec as it stands, and that is the one that must be true.
    wrong = {d: sorted(v) for d, v in claimed.items() if max(v) != actual}
    assert wrong == {}, (
        f"documents' current requirement-count claim is not the spec's {actual}: {wrong}"
    )


def test_claimed_test_count_matches_the_suite():
    """CLAUDE.md's green-suite gate is the count a future session checks against."""
    total = 0
    for path in sorted((REPO / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        total += sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
    claude = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    claimed = re.findall(r"all (\d+) tests pass", claude)
    assert claimed, "CLAUDE.md no longer states a suite size; the gate is gone"
    assert [int(c) for c in claimed] == [total] * len(claimed), (
        f"CLAUDE.md claims {claimed} tests; the suite defines {total}"
    )
