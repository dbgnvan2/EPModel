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

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

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
    REPO / "docs" / "milestone_review_brief.md",
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
    # Case-insensitive, and tolerant of MUST being inside an enclosing bold span
    # rather than bolded on its own. The first version required capital "No" and a
    # separately-bolded **MUST**; it reported the class closed while four
    # instances survived — one of them M11.D.12, the guard written for this very
    # finding, in the form the finding forbids.
    inverted = re.findall(
        # Words only between "no" and MUST — a clause boundary (comma-plus-clause,
        # semicolon, quote, backtick) means the "no" is not the subject of the
        # MUST, and matching across one produced four false alarms. A guard that
        # cries wolf gets switched off, which is its own failure mode.
        r"(?i)\bno\s+(?:[a-z][a-z-]*,?\s+){0,8}(?:\*\*)?MUST(?:\*\*)?(?! NOT)\b",
        _spec_text(),
    )
    assert inverted == [], (
        "prohibitions written in the inverted form 'No X MUST Y', which §0.3 "
        f"reads as a permission: {inverted}"
    )


# --------------------------------------------------------------------------
# M11.D.11 — every criterion gates the phase its own row names (M13.2a)
# --------------------------------------------------------------------------

def _criterion_rows(text: str) -> list[tuple[str, list[str]]]:
    return [
        (cid, [c.strip() for c in rest.split(" | ")])
        for cid, rest in re.findall(r"^\| \*\*(M11\.C\.\d+)\*\* \|(.*)$", text, re.M)
    ]


def _phase_of(cell: str) -> str | None:
    """The phase letter in a table cell, however it is decorated.

    The first version required the cell to be exactly `B`/`C`/`D`/`E`. One row
    already used ``**`→E`**`` and was silently dropped, and bolding every phase
    cell — a purely cosmetic edit — made the guard check *zero* criteria and stay
    green. Strip decoration and read the letter.
    """
    bare = re.sub(r"[^A-Za-z]", "", cell)
    return bare if bare in {"B", "C", "D", "E", "F"} else None


def _phase_exit_conditions(text: str) -> dict[str, str]:
    """Phase letter -> its *Done when* cell only.

    The first version returned the whole M13 row, so a criterion named in the
    "Builds" column satisfied a guard that M13.2a scopes to the exit condition —
    and the Builds column is written in exactly that style.
    """
    exits = {}
    for letter in "BCDEF":
        row = re.search(r"^\| \*\*%s\*\* \|(.*)$" % letter, text, re.M)
        if row:
            cells = [c.strip() for c in row.group(1).split(" | ")]
            exits[letter] = cells[-1] if cells else ""
    return exits


def test_m11d11_every_criterion_row_declares_a_readable_phase():
    """No row may be silently dropped — that is how the guard went vacuum-green."""
    rows = _criterion_rows(_spec_text())
    assert rows, "no M11.C rows parsed — the table moved or was renamed"
    unreadable = sorted(
        cid for cid, cells in rows if len(cells) < 3 or _phase_of(cells[2]) is None
    )
    assert unreadable == [], (
        f"M11.C rows whose phase cell this guard cannot read, so it checks nothing "
        f"for them: {unreadable}"
    )


def test_m11d11_every_criterion_gates_its_phase():
    text = _spec_text()
    exits = _phase_exit_conditions(text)
    rows = _criterion_rows(text)
    checked = 0
    ungated = []
    for cid, cells in rows:
        phase = _phase_of(cells[2]) if len(cells) >= 3 else None
        if phase is None or phase not in exits:
            continue
        checked += 1
        if cid not in exits[phase]:
            ungated.append((cid, phase))
    # A criterion assigned to a phase M13 does not table (E, F — out of scope per
    # §0.1) cannot be gated there, and is not a defect. Everything else must be
    # checked, so a narrowed scan cannot report clean by looking at nothing.
    gateable = [
        cid
        for cid, cells in rows
        if len(cells) >= 3 and _phase_of(cells[2]) in exits
    ]
    assert checked == len(gateable), (
        f"guard checked {checked} of {len(gateable)} gateable criteria — a "
        "silently narrowed scan reports clean because it looked at nothing"
    )
    assert gateable, "no criteria are gateable — M13's phase table moved or was renamed"
    ungated = sorted(ungated)
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
# M11.D.14 — §0.3's own coverage figures are measured, not stated once
# --------------------------------------------------------------------------

def test_m11d14_section_0_3_counts_are_current():
    """The figures in §0.3 were written once and were stale on arrival.

    They were measured at the parent commit, so the sixteen MUSTs the same
    commit added were already missing from them — the identical drift the
    requirement-count guard exists to catch, one paragraph away from it.
    """
    text = _spec_text()
    must = len(re.findall(r"\*\*MUST(?: NOT)?\*\*", text))
    tests = len(set(re.findall(r"`(test_[a-z0-9_]+)`", text)))
    stated = re.search(
        r"roughly (\d+) bolded MUST/MUST NOT tokens against (\d+) named tests", text
    )
    assert stated, "§0.3 no longer states its coverage figures; the honesty note is gone"
    assert (int(stated.group(1)), int(stated.group(2))) == (must, tests), (
        f"§0.3 states {stated.group(1)} MUSTs / {stated.group(2)} tests; "
        f"the document has {must} / {tests}"
    )


# --------------------------------------------------------------------------
# P6 — status claims reconcile to the artifact they describe
# --------------------------------------------------------------------------

def test_requirement_counts_agree_with_the_spec():
    actual = len(set(defined_ids(_spec_text())))
    claimed = {}
    for doc in CITING_DOCS:
        # "405 requirements over 16 modules" was invisible to the first version,
        # which required the literal "numbered requirements" or "unique IDs".
        for n in re.findall(
            r"(\d{3,4})\s+(?:numbered\s+)?requirements|(?:\b)(\d{3,4})\s+unique IDs",
            doc.read_text(encoding="utf-8"),
        ):
            n = n[0] or n[1] if isinstance(n, tuple) else n
            claimed.setdefault(doc.relative_to(REPO).as_posix(), set()).add(int(n))
    # Documents legitimately cite earlier, smaller counts as history ("that left
    # the spec at 321"). The *largest* count a document states is its claim about
    # the spec as it stands, and that is the one that must be true.
    wrong = {d: sorted(v) for d, v in claimed.items() if max(v) != actual}
    assert wrong == {}, (
        f"documents' current requirement-count claim is not the spec's {actual}: {wrong}"
    )


def _collected() -> tuple[int, list[str]]:
    """The number of tests pytest collects — what "all N tests pass" claims — and their node IDs.

    Counting `def test_*` in the AST was the earlier method. It undercounts a
    parametrized test, misses files outside the glob, and agreed with the real
    count only by coincidence until Phase B added a parametrized test (TODO.md).
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO / "tests"), "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=120, cwd=REPO,
    )
    match = re.search(r"^(\d+)(?:/\d+)? tests? collected", result.stdout, re.M)
    assert match, f"could not read pytest's collection count:\n{result.stdout[-2000:]}{result.stderr[-2000:]}"
    return int(match.group(1)), [line for line in result.stdout.splitlines() if "::" in line]


def _collected_test_count() -> int:
    return _collected()[0]


def _collected_expected_failures(node_ids: list[str]) -> int:
    """How many collected tests are expected failures, as `tools/spec_coverage.py` reads the marks (MR-20)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("spec_coverage_for_count", REPO / "tools" / "spec_coverage.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    xfails = tool.expected_failures()
    return sum(1 for node in node_ids
               if f"{node.split('::', 1)[0]}::{node.rsplit('::', 1)[1].split('[', 1)[0]}" in xfails)


# D4.1: "N tests pass", "N tests passing", "N tests green", "N tests, all passing", and one word between, as in
# "N tests stay green". The word boundary after the verb keeps "passed" (a past-tense run report, D4.2) out; the
# pattern this replaced read "850 tests passed" as a claim and missed "37 tests, all passing".
_SUITE_SIZE = re.compile(r"\b(\d+)\s+tests?,?\s+(?:all\s+|\w+\s+)?(?:pass|passes|passing|green)\b", re.I)


def suite_size_claims(text: str) -> list[int]:
    """D4.1, D4.2: every present-tense suite-size claim in the text."""
    return [int(c) for c in _SUITE_SIZE.findall(text)]


def _suite_size_history() -> list[dict]:
    # Read directly, not through tests/doc_text.py: tools/spec_coverage.py loads this file by path, outside pytest.
    claims = json.loads((REPO / "tests" / "doc_claims.json").read_text(encoding="utf-8"))
    return claims["suite_size_history"]["entries"]


def _without_history(doc: str, text: str) -> str:
    """D4.3: drop the dated suite-size phrases the data file lists for this document."""
    for entry in _suite_size_history():
        if entry["doc"] == doc:
            text = text.replace(entry["phrase"], "")
    return text


def test_claimed_test_count_matches_the_suite():
    """CLAUDE.md's green-suite gate is the count a future session checks against.

    Spec:  docs/plan_doc_claim_tests.md#D4 (the widened pattern, D4.1–D4.4)
    """
    total, node_ids = _collected()
    # Every document, not CLAUDE.md alone: "37 tests green" sat stale in
    # _STATUS.md and CHANGELOG.md while CLAUDE.md's own figure was correct.
    # "37 tests, all passing" sat stale in README.md's Layout block while the
    # pattern read only "N tests pass" and "N tests green" (D4).
    wrong = {}
    for doc in CITING_DOCS:
        rel = doc.relative_to(REPO).as_posix()
        claims = suite_size_claims(_without_history(rel, doc.read_text(encoding="utf-8")))
        bad = sorted({c for c in claims if c != total})
        if bad:
            wrong[rel] = bad
    claude = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    # Stage 2 milestone review MR-20: "all N tests pass" was untrue while one test is a strict expected failure, so the
    # gate reads "N tests: P pass and X is a strict expected failure" as well, and checks its sum.
    gate = re.search(r"all (\d+) tests pass|(\d+) tests: (\d+) pass and (\d+) (?:is a|are) strict expected failures?", claude)
    assert gate, "CLAUDE.md no longer states a suite size; the green-suite gate is gone"
    # The expected-failure figure is checked against the marks, not only the sum (learning-qa 2026-10-24): "all N
    # pass" is refused while any collected test is an expected failure, and X must be how many are.
    xfails = _collected_expected_failures(node_ids)
    if gate.group(1):
        assert xfails == 0, f"CLAUDE.md says {gate.group(0)!r}, but {xfails} collected tests are expected failures"
    else:
        assert int(gate.group(2)) == total == int(gate.group(3)) + int(gate.group(4)), (
            f"CLAUDE.md's suite size {gate.group(0)!r} does not add up to the {total} tests collected"
        )
        assert int(gate.group(4)) == xfails, (
            f"CLAUDE.md's gate names {gate.group(4)} expected failures; {xfails} collected tests are marked xfail"
        )
    assert wrong == {}, f"documents claim a suite size that is not {total}: {wrong}"


@pytest.mark.parametrize("text, claims", [
    # D4.1: present-tense claims, each read.
    ("all 924 tests pass", [924]),
    ("924 tests pass", [924]),
    ("924 tests passing", [924]),
    ("924 tests green", [924]),
    ("tests/               37 tests, all passing", [37]),
    ("37 tests, all green", [37]),
    ("the old engine and its 36 tests stay green", [36]),
    ("37 tests,\nall passing", [37]),
    # D4.2: past-tense run reports are history by their tense, and are not read.
    ("850 passed in CI run 18", []),
    ("Suite: 850 tests.", []),
    ("850 tests passed on 3.11", []),
])
def test_d4_suite_size_phrasings_are_recognised(text, claims):
    """D4.1, D4.2: the phrasings the documents use, including the one that sat stale in README.md."""
    assert suite_size_claims(text) == claims


def test_d4_history_entries_still_exist():
    """D4.3: each dated phrase the data file excuses is still in its document, so the list cannot outlive it."""
    gone = [(e["doc"], e["phrase"]) for e in _suite_size_history()
            if e["phrase"] not in (REPO / e["doc"]).read_text(encoding="utf-8")]
    assert gone == []
    assert all(e["why"] for e in _suite_size_history())


# --------------------------------------------------------------------------
# M11.D.17 — the state-and-mechanism register (M14.A) has no orphans
# --------------------------------------------------------------------------

_PHASES = {"B", "C", "D", "E"}
_STEP = re.compile(r"\bstep ([1-9])\b")


def _register_tables(text: str) -> tuple[list[list[str]], list[list[str]]]:
    """Return the M14.A variables table and mechanisms table as rows of cells."""
    start = text.index("### M14.A State and mechanism register")
    end = text.index("\n---\n", start)
    section = text[start:end]

    def table_after(label: str, header_first_cell: str) -> list[list[str]]:
        at = section.index(label)
        rows = []
        for line in section[at:].splitlines()[1:]:
            if not line.startswith("|"):
                if rows:
                    break
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells[0] in (header_first_cell,) or set(cells[0]) <= {"-"}:
                continue
            rows.append(cells)
        return rows

    return (
        table_after("**State variables**", "Variable"),
        table_after("**Mechanisms**", "Mechanism"),
    )


def register_orphans(text: str) -> list[str]:
    """Every M11.D.17 violation in the register, as readable strings."""
    variables, mechanisms = _register_tables(text)
    problems = []
    if not variables or not mechanisms:
        problems.append("register tables are empty or missing")
    for row in variables:
        name, _owner, _cls, written_by, phase = row
        phases = {p.strip() for p in phase.split(";")}
        if not written_by or written_by in {"—", "-"}:
            problems.append(f"variable {name} has no writer")
        if not phases or not phases <= _PHASES:
            problems.append(f"variable {name} has no valid phase: {phase!r}")
    placed_names = {row[0] for row in mechanisms}
    for row in mechanisms:
        name, owner, mode = row[0], row[1], row[2]
        phase = row[-1]
        placed = (
            _STEP.search(mode)
            or "slow tick" in mode
            or "documented composite" in mode
            or mode.startswith("called from")
        )
        if not placed:
            problems.append(f"mechanism {name} is not placed in M3.D.1's order")
        if phase not in _PHASES:
            problems.append(f"mechanism {name} has no valid phase: {phase!r}")
        # The static form of M4.B.2: an agent-owned mechanism may read only its
        # owner's own state, beliefs and delivered events (D6).
        if owner != "engine":
            reads = row[4].lower()
            allowed = ("own", "belief", "inbox", "delivered")
            if not any(word in reads for word in allowed) or "true" in reads:
                problems.append(f"mechanism {name} owned by {owner} reads unobservable state")
    assert len(placed_names) == len(mechanisms), "duplicate mechanism rows"
    return problems


def test_m11d17_register_has_no_orphans():
    """M14.A's register is checked, not trusted.

    Spec:  docs/bowen_agent_model_spec_v2.md#M11.D.17, #M14.A.4
    """
    assert register_orphans(_spec_text()) == []


def test_m11d17_check_catches_a_missing_writer():
    """The check is proved able to fail: blank one writer and it must say so."""
    spec = _spec_text()
    row = next(line for line in spec.splitlines() if line.startswith("| `conductance` | Relationship |"))
    cells = row.split(" | ")
    blanked = " | ".join(cells[:3] + ["—"] + cells[4:])  # the writer column, whatever it says today
    assert blanked != row
    assert "variable `conductance` has no writer" in register_orphans(spec.replace(row, blanked))


def test_m11d17_check_catches_an_unplaced_mechanism():
    text = _spec_text().replace(
        "| engine | synchronous batch, step 1 |", "| engine | whenever convenient |", 1
    )
    assert any("not placed" in p for p in register_orphans(text))


def test_m11d17_check_catches_an_agent_reading_true_state():
    text = _spec_text().replace(
        "| Perceive (`M4.B.1`) | engine | synchronous batch, step 3 | every tick | inbox |",
        "| Perceive (`M4.B.1`) | Person | synchronous batch, step 3 | every tick | another person's true acute_anxiety |",
    )
    assert any("unobservable" in p for p in register_orphans(text))
