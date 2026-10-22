"""The spec index and the ledger index are current, complete, and read from what they claim.

Purpose: docs/spec_index.jsonl and docs/theory/ledger_index.jsonl are generated from the spec, the ledger, the code
         and the tests; a stale line is a claim that no longer reconciles with its source (P6). The fixtures prove
         each field's reader on text it cannot misread by luck.
Spec:    docs/plan_model_ergonomics.md#B1, #B2, #B3, #B4
Tests:   this file
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("spec_index_tool", REPO / "tools" / "spec_index.py")
TOOL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TOOL)

# B3 (2026-10-22): the `Spec:` lines in src/ that name a section heading of the spec (`M1.A`, `M13`), not a numbered
# requirement. Listed in TODO.md for the owner (the plan's §8 Q4); this test is xfail until they are decided.
B3_SECTION_ANCHORS = {
    "src/bowen/engine/events.py": ["M1.F"],
    "src/bowen/engine/invariants.py": ["M6"],
    "src/bowen/engine/log_records.py": ["M16.A"],
    "src/bowen/engine/objects.py": ["M1", "M14.A"],
    "src/bowen/engine/objects.py::Person": ["M1.A"],
    "src/bowen/engine/objects.py::Relationship": ["M1.B"],
    "src/bowen/engine/objects.py::Triangle": ["M1.C"],
    "src/bowen/engine/objects.py::Family": ["M1.D"],
    "src/bowen/io/load.py::load_script": ["M13"],
    "src/bowen/render/trace.py": ["M11.F"],
    "src/bowen/scenario/family.py": ["M2"],
    "src/bowen/scenario/family.py::FamilyInstance": ["M2"],
    "src/bowen/scenario/policy_source.py": ["M13"],
    "src/bowen/scenario/scripted_source.py": ["M13"],
    "src/bowen/scenario/scripted_source.py::ScriptedSource": ["M13"],
}


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_b2a_spec_index_is_current():
    expected = TOOL.render(TOOL.spec_index())
    actual = TOOL.SPEC_INDEX.read_text(encoding="utf-8")
    assert actual == expected, "regenerate: python3 tools/spec_index.py"


def test_b2a_ledger_index_is_current():
    expected = TOOL.render(TOOL.ledger_index())
    actual = TOOL.LEDGER_INDEX.read_text(encoding="utf-8")
    assert actual == expected, "regenerate: python3 tools/spec_index.py"


def test_b2b_every_spec_id_appears_once_in_document_order():
    ids = [r["id"] for r in _records(TOOL.SPEC_INDEX)]
    assert ids == TOOL.coverage_tool.spec_ids(TOOL.SPEC.read_text(encoding="utf-8"))
    assert len(ids) == len(set(ids))


@pytest.mark.xfail(strict=True, reason="B3: fifteen `Spec:` lines name a section heading, not a requirement; "
                                        "TODO.md, 'Model ergonomics', for the owner's Q4")
def test_b2c_every_id_cited_in_src_is_defined_by_the_spec():
    _, unknown = TOOL.code_citations()
    assert unknown == {}, f"src/ cites IDs the spec does not define: {unknown}"


def test_b3_the_undefined_citations_are_exactly_the_listed_section_anchors():
    """The xfail above is for B3's list and nothing else: a new undefined citation, or one of these fixed, fails here
    so the list in TODO.md and the xfail are brought up to date."""
    _, unknown = TOOL.code_citations()
    assert unknown == B3_SECTION_ANCHORS
    headings = set(re.findall(r"^#{2,3} (M\d+(?:\.[A-Z])?)\b", TOOL.SPEC.read_text(encoding="utf-8"), re.M))
    cited = {c for cites in unknown.values() for c in cites}
    assert cited <= headings, f"not a section heading of the spec either: {sorted(cited - headings)}"


def test_b1_lines_start_at_the_defining_line_and_do_not_overlap():
    lines = TOOL.SPEC.read_text(encoding="utf-8").split("\n")
    previous_end = 0
    for record in _records(TOOL.SPEC_INDEX):
        start, end = record["lines"]
        defining = TOOL.DEF_ROW.match(lines[start - 1]) or TOOL.DEF_LINE.match(lines[start - 1])
        assert defining and defining.group(1) == record["id"], record["id"]
        assert previous_end < start <= end, record["id"]
        assert lines[end - 1].strip(), f"{record['id']}: the block ends on a blank line"
        previous_end = end


FIXTURE_SPEC = """---
version: 2.0, revision 99
---

# A spec

## M1 — Objects

### M1.A `Person`

**M1.A.1** — *the first rule, in **bold**.* The second sentence cites `kerr_book/ks23.md` · KS23.3 and L01.1 `[I]`
and the KB12 interview (K12.6), with FE03.21 and Tape 6 (T6.1) and DL §2.6 and `[I]` again, `[X]`.

> "a block quote that mentions KS99.9"

- **M1.A.2** WITHDRAWN at revision 11. It read: something.

## M11 — Criteria

| ID | Statement | Test |
|---|---|---|
| **M11.C.1** | Differentiation protects: a cell. Second sentence. | `test_m11c1_x` |

**M11.C.9** — e.g. an abbreviation does not end it; a long one """ + "x" * 400 + """ ends here. Next.
"""


def test_b1_fields_are_read_from_a_fixture():
    """Mutation for each reader: drop the heading reset, the row branch, the emphasis cut, the 300-char cut, the tag
    dedupe, the WITHDRAWN test or a source pattern."""
    blocks = TOOL.spec_blocks(FIXTURE_SPEC)
    by_id = {b["id"]: b for b in blocks}
    assert [b["id"] for b in blocks] == ["M1.A.1", "M1.A.2", "M11.C.1", "M11.C.9"]
    one = by_id["M1.A.1"]
    assert (one["module"], one["heading"]) == ("M1 — Objects", "M1.A `Person`")
    assert one["lines"] == [11, 14], "the block runs to the quote and stops before the next ID"
    assert TOOL.first_sentence(one) == "the first rule, in bold."
    assert TOOL.grade_tags(one["text"]) == ["I", "X"]
    assert TOOL.sources(one["text"], TOOL.source_pattern()) == [
        "KS23.3", "L01.1", "KB12", "K12.6", "FE03.21", "Tape 6", "T6.1", "DL §2.6", "KS99.9"]
    two = by_id["M1.A.2"]
    assert two["lines"] == [16, 16] and "WITHDRAWN" in two["text"]
    assert TOOL.first_sentence(two) == "WITHDRAWN at revision 11."
    row = by_id["M11.C.1"]
    assert row["row"] and row["lines"] == [22, 22] and (row["module"], row["heading"]) == ("M11 — Criteria", "")
    assert TOOL.first_sentence(row) == "Differentiation protects: a cell."
    long = TOOL.first_sentence(by_id["M11.C.9"])
    assert len(long) == TOOL.FIRST_SENTENCE_CHARS and long.startswith("e.g. an abbreviation does not end it")


def test_b1_spec_citations_follow_the_document_named():
    """A bare `#ID` continues the document the previous citation named; an anchor in another document is not a spec
    citation; a trailing comma or full stop is not part of the ID. Mutation: `document` not carried over."""
    doc = """Purpose: x.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1, #M4.C.1a,
             #M1.F.2; docs/implementation_plan_version_2.md#S1.5a, #S1.5b; docs/bowen_agent_model_spec_v2.md#M16.B.
    Tests:   tests/x.py::test_m4c1_y
    """
    assert TOOL.spec_citations(doc) == ["M4.C.1", "M4.C.1a", "M1.F.2", "M16.B"]
    assert TOOL.spec_citations("Spec: docs/implementation_plan_version_2.md#S1.5a\n") == []
    assert TOOL.spec_citations("No spec line, #M1.A.1 in prose.\n") == []


def test_b1_cited_spec_ids_respect_id_boundaries():
    defined = {"M4.C.3a", "M4.C.3", "M4.C", "M1"}
    text = "M4.C.3a, then M4.C.3. Then M4.C.30 and M1 and xM1 and M1.Z."
    assert TOOL.cited_spec_ids(text, defined) == ["M4.C.3a", "M4.C.3", "M1"]


def test_b1_code_names_an_existing_definition():
    """Every `path::qualname` in the index resolves to a module, class or function in src/."""
    trees: dict[str, ast.Module] = {}
    for record in _records(TOOL.SPEC_INDEX):
        for where in record["code"]:
            path, _, qualname = where.partition("::")
            tree = trees.setdefault(path, ast.parse((REPO / path).read_text(encoding="utf-8")))
            node: ast.AST = tree
            for name in [n for n in qualname.split(".") if n]:
                node = next((c for c in getattr(node, "body", [])
                             if isinstance(c, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and c.name == name),
                            None)
                assert node is not None, f"{record['id']}: {where} does not exist"


def test_b1_status_fields_are_coverage_verbatim():
    rows = {r["id"]: r for r in TOOL.coverage_tool.coverage(json.loads(TOOL.OVERRIDES.read_text(encoding="utf-8")))}
    for record in _records(TOOL.SPEC_INDEX):
        row = rows[record["id"]]
        assert (record["status"], record["status_note"], record["tests"]) == (row["status"], row["note"], row["evidence"])


def test_b1_ledger_mentions_are_the_inverse_of_the_ledger_index():
    spec = _records(TOOL.SPEC_INDEX)
    ledger = _records(TOOL.LEDGER_INDEX)
    inverse: dict[str, list[str]] = {}
    for entry in ledger:
        for spec_id in entry["spec_ids"]:
            inverse.setdefault(spec_id, []).append(entry["id"])
    assert {r["id"]: r["ledger_mentions"] for r in spec if r["ledger_mentions"]} == inverse
    # An independent read: every defined ID the ledger text names, found without the entry parser, is in the
    # inverse unless each line naming it lies outside every entry's span (the ledger's preamble and closing
    # sections are not entries).
    defined = {r["id"] for r in spec}
    lines = TOOL.LEDGER.read_text(encoding="utf-8").split("\n")
    spans = [tuple(e["lines"]) for e in ledger]
    for number, line in enumerate(lines, start=1):
        inside = any(start <= number <= end for start, end in spans)
        for spec_id in TOOL.CITED_SPEC_ID.findall(line):
            if spec_id in defined and inside:
                assert spec_id in inverse, f"line {number} names {spec_id} inside an entry"
    assert len(inverse) >= 50


def test_b4_every_entry_heading_is_indexed_once_at_its_line():
    text = TOOL.LEDGER.read_text(encoding="utf-8")
    lines = text.split("\n")
    records = _records(TOOL.LEDGER_INDEX)
    headings = [m.group(2) for m in (TOOL.LEDGER_ENTRY.match(line) for line in lines) if m]
    assert [r["id"] for r in records] == headings
    assert len(headings) == len(set(headings))
    for record in records:
        assert TOOL.LEDGER_ENTRY.match(lines[record["lines"][0] - 1]).group(2) == record["id"]
        assert record["lines"][0] <= record["lines"][1]
    assert "L01.1" in headings and "KS-A.1" in headings and "U1" in headings and "FE-B" in headings
    assert "Ch01" not in headings


def test_b4_withdrawn_and_merged_entries_stay_with_their_status():
    by_id = {r["id"]: r for r in _records(TOOL.LEDGER_INDEX)}
    assert by_id["L01.7"]["status"] == "withdrawn"
    assert by_id["L03.2"]["status"] == "withdrawn as a separate finding"
    assert "merged into L03.1" in by_id["L03.2"]["title"]
    assert by_id["L01.1"]["status"] == "stands" and by_id["L01.1"]["chapter"].startswith("Ch01")
    assert by_id["KS-E.2"]["labels"] == ["new", "K-ext"]


def test_b4_fields_are_read_from_a_fixture():
    """Mutation: the chapter not reset at a `##`, a `###` subsection without an ID closing the entry, or a bracketed
    label left in the title."""
    text = """# Ledger

## Ch01 — First paper  *(1957)*

### L01.1 — A finding  `[stands]`

Cites `M1.A.1` and M1.A.1a; M1.A.10 is not defined.

### L01.2 — *(merged into L01.1)*  `[withdrawn as a separate finding]`

## KS-A · Six requirements

### KS-A.1 — One  `[new]`  `[K-ext]`

### Not an entry

Still KS-A.1's text, citing M2.

## U1 — A resolution  `[user]` → `M1.A.1`
"""
    entries = TOOL.ledger_entries(text)
    assert [e["id"] for e in entries] == ["L01.1", "L01.2", "KS-A", "KS-A.1", "U1"]
    by_id = {e["id"]: e for e in entries}
    assert by_id["L01.1"]["chapter"] == "Ch01 — First paper  *(1957)*" and by_id["L01.1"]["lines"] == [5, 7]
    assert TOOL.cited_spec_ids(by_id["L01.1"]["text"], {"M1.A.1", "M1.A.1a", "M2"}) == ["M1.A.1", "M1.A.1a"]
    assert by_id["L01.2"]["title"] == "*(merged into L01.1)*"
    assert by_id["KS-A"]["chapter"] == "" and by_id["KS-A"]["status"] == ""
    assert by_id["KS-A.1"]["chapter"] == "KS-A · Six requirements" and by_id["KS-A.1"]["labels"] == ["new", "K-ext"]
    assert by_id["KS-A.1"]["lines"] == [13, 17], "the ID-less `###` stays inside the entry"
    assert TOOL.cited_spec_ids(by_id["KS-A.1"]["text"], {"M2"}) == ["M2"]
    assert by_id["U1"]["title"] == "A resolution → `M1.A.1`" and by_id["U1"]["status"] == "user"
