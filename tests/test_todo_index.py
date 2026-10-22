"""TODO.md, the archive's closed-items region and the changelog index are what the records and entries say now.

Purpose: `tools/todo_index.py` generates the three; a hand edit to any of them, a record ticked without a run of the
         tool, or a new changelog entry left out of the index, fails here. The records must stay well formed (every
         open item has an ID and exactly one of ⟨now⟩ ⟨later⟩; IDs are unique), and the tool must never write a
         record. Each rule is tried on a fixture that breaks it as well as on the real files.
Spec:    CLAUDE.md — "Token budget — how to read this repo" (owner rule, 2026-10-22)
Tests:   this file
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import todo_index as ti  # noqa: E402

RECORD = "docs/todo/2026-10-01_fixture.md"
GOOD = ("# A batch\n\nA note the index does not read.\n\n"
        "- [ ] T-001 ⟨now⟩ **Fix the thing** (found 2026-10-03). Why it matters,\n  on a second line.\n"
        "- [x] T-002 ⟨later⟩ ~~**Decide the other thing.**~~ *Decided 2026-10-05.* Raised 2026-10-02.\n"
        "- [ ] T-003 ⟨later⟩ no bold span, a plain first line long enough to be cut to the title's word limit here\n")


# ------------------------------------------------------------------------------------------------ the real files

def test_records_are_well_formed():
    _, problems = ti.records()
    assert problems == []


def test_todo_md_is_current():
    assert (REPO / ti.TODO).read_text(encoding="utf-8") == ti.generate()[ti.TODO], \
        "regenerate: python3 tools/todo_index.py"


def test_archive_closed_region_is_current():
    assert (REPO / ti.ARCHIVE).read_text(encoding="utf-8") == ti.generate()[ti.ARCHIVE], \
        "regenerate: python3 tools/todo_index.py"


def test_changelog_index_is_current():
    assert (REPO / ti.CHANGELOG).read_text(encoding="utf-8") == ti.generate()[ti.CHANGELOG], \
        "regenerate: python3 tools/todo_index.py"


def test_every_open_item_is_indexed_once():
    items, _ = ti.records()
    todo = (REPO / ti.TODO).read_text(encoding="utf-8")
    for item in items:
        assert todo.count(f"- {item.id} ·") == (0 if item.closed else 1), item.id


# ------------------------------------------------------------------------------------------------ the records' rules

def test_a_well_formed_record_parses():
    items, problems = ti.parse_record(RECORD, GOOD)
    assert problems == []
    assert [(i.id, i.tag, i.closed, i.line) for i in items] == [("T-001", "now", False, 5), ("T-002", "later", True, 7),
                                                                 ("T-003", "later", False, 8)]
    assert items[0].text == "**Fix the thing** (found 2026-10-03). Why it matters,\n  on a second line."


@pytest.mark.parametrize("line, problem", [
    ("- [ ] no id here", "item without an ID"),
    ("- [ ] T-009 no tag", "exactly one of"),
    ("- [ ] T-009 ⟨now⟩ ⟨later⟩ two tags", "exactly one of"),
    ("- [ ] T-009 ⟨soon⟩ a tag that is not one of the two", "exactly one of"),
])
def test_an_item_without_an_id_or_with_other_than_one_tag_is_reported(line, problem):
    items, problems = ti.parse_record(RECORD, GOOD + line + "\n")
    assert len(items) == 3 and len(problems) == 1 and problem in problems[0], problems


def test_a_duplicate_id_across_records_is_reported(tmp_path):
    (tmp_path / "docs" / "todo").mkdir(parents=True)
    (tmp_path / "docs" / "todo" / "2026-10-01_a.md").write_text(GOOD, encoding="utf-8")
    (tmp_path / "docs" / "todo" / "2026-10-02_b.md").write_text("# B\n\n- [ ] T-001 ⟨later⟩ again\n", encoding="utf-8")
    _, problems = ti.records(tmp_path)
    assert problems == ["docs/todo/2026-10-02_b.md:3: T-001 is also at docs/todo/2026-10-01_a.md:5"]


def test_a_record_name_without_a_date_is_reported():
    _, problems = ti.parse_record("docs/todo/fixture.md", GOOD)
    assert problems == ["docs/todo/fixture.md: the file name is not <YYYY-MM-DD>_<slug>.md"]


# ------------------------------------------------------------------------------------------------ the rendering

def test_title_and_dates():
    items, _ = ti.parse_record(RECORD, GOOD)
    assert [ti.title(i.text) for i in items] == [
        "Fix the thing", "Decide the other thing", "no bold span, a plain first line long …"]
    assert [ti.item_date(i) for i in items] == ["2026-10-03", "2026-10-05", "2026-10-01"]  # first date, else the record's
    assert ti.closed_date(items[1]) == "2026-10-05"                                        # latest date


def test_a_title_cut_inside_backticks_closes_them():
    assert ti.title("one two three four five six seven `eight nine` ten").endswith("`eight` …")


def test_open_items_go_to_todo_by_tag_and_ticked_ones_to_the_archive():
    items, _ = ti.parse_record(RECORD, GOOD)
    todo = ti.render_todo(items)
    assert "## Now (1)\n\n- T-001 · 2026-10-03 · Fix the thing · [record](docs/todo/2026-10-01_fixture.md)\n" in todo
    assert "## Later (1)\n\n- T-003 ·" in todo and "T-002" not in todo
    assert ti.render_closed(items) == "- T-002 · closed 2026-10-05 · Decide the other thing · [record](todo/2026-10-01_fixture.md)"
    assert ti.render_closed(items[:1]) == "(none yet)"


def test_changelog_index_takes_each_entry_in_order_with_its_date():
    text = ("# Changelog\n\nintro\n\n## [Unreleased] — 2026-10-11\n\n### First (2026-10-22)\n\nbody\n\n"
            "### Undated, takes the section's\n\n## 2026-08-22 — old\n\n### Third (2026-08-22)\n")
    assert ti.render_changelog_index(text) == ("- 2026-10-22 — First (2026-10-22)\n"
                                               "- 2026-10-11 — Undated, takes the section's\n"
                                               "- 2026-08-22 — Third (2026-08-22)")


def test_a_region_is_replaced_in_place_or_put_before_the_first_section():
    text = "# T\n\nintro\n\n## A\n\nbody\n"
    first = ti.with_region(text, "<!-- b -->", "<!-- e -->", "x\ny")
    assert first == "# T\n\nintro\n\n<!-- b -->\nx\ny\n<!-- e -->\n\n## A\n\nbody\n"
    assert ti.with_region(first, "<!-- b -->", "<!-- e -->", "z") == "# T\n\nintro\n\n<!-- b -->\nz\n<!-- e -->\n\n## A\n\nbody\n"


# ------------------------------------------------------------------------------------------------ the tool

def _copy(tmp_path: Path) -> Path:
    for rel in (ti.TODO, ti.ARCHIVE, ti.CHANGELOG):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    shutil.copytree(REPO / ti.RECORDS, tmp_path / ti.RECORDS)
    return tmp_path


def test_the_tool_rewrites_the_three_files_and_never_a_record(tmp_path):
    repo = _copy(tmp_path)
    before = {p.name: p.read_bytes() for p in (repo / ti.RECORDS).glob("*.md")}
    (repo / ti.TODO).write_text("hand-edited\n", encoding="utf-8")
    assert (repo / ti.TODO).read_text(encoding="utf-8") != ti.generate(repo)[ti.TODO]  # the current check sees it
    assert ti.main([], repo) == 0
    assert (repo / ti.TODO).read_text(encoding="utf-8") == (REPO / ti.TODO).read_text(encoding="utf-8")
    assert {p.name: p.read_bytes() for p in (repo / ti.RECORDS).glob("*.md")} == before


def test_the_tool_refuses_a_malformed_record_and_writes_nothing(tmp_path, capsys):
    repo = _copy(tmp_path)
    record = next(iter(sorted((repo / ti.RECORDS).glob("*.md"))))
    record.write_text(record.read_text(encoding="utf-8") + "- [ ] T-999 no tag\n", encoding="utf-8")
    before = (repo / ti.TODO).read_bytes()
    assert ti.main([], repo) == 1
    assert "T-999 must carry exactly one of" in capsys.readouterr().err
    assert (repo / ti.TODO).read_bytes() == before
