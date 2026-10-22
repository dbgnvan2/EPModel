"""`TODO.md` and `CHANGELOG.md` stay short: their closed and old parts leave for the archives under the suite's eye.

Purpose: both files are read in part by most sessions and grew without bound (130 KB and 148 KB on 2026-10-22, when
         the ticked items and the entries dated before version 2 moved to `docs/TODO_archive.md` and
         `docs/CHANGELOG_archive.md`). This test keeps the archives current by the suite and not by habit: it fails
         when `TODO.md` holds a ticked item closed more than ``TICKED_ITEM_MAX_AGE_DAYS`` ago or ticked without a
         date, and when either live file is larger than its limit. A ticked item's close date is the latest ISO date
         it carries, the rule the archive groups by. The limits are the named constants below.
Spec:    docs/plan_model_ergonomics.md (the two files the plan left growing); CLAUDE.md — "Documentation stays current
         with the build"
Tests:   this file
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TODO = "TODO.md"
CHANGELOG = "CHANGELOG.md"
TODO_ARCHIVE = "docs/TODO_archive.md"
CHANGELOG_ARCHIVE = "docs/CHANGELOG_archive.md"

# The limits. 60 KB each was the suggested start (2026-10-22); that day the open items alone were 82 KB and the
# version 2 entries 62 KB, so each limit starts above what its file held. Lower a limit when its file shrinks; never
# raise one to pass — move ticked items, or the oldest entries, to the archive instead.
TODO_LIMIT_BYTES = 100_000
CHANGELOG_LIMIT_BYTES = 80_000
TICKED_ITEM_MAX_AGE_DAYS = 30

_ITEM = re.compile(r"^(\s*)- \[([ xX])\] ")
_ISO = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")


@dataclass(frozen=True)
class TickedItem:
    line: int          # 1-based line of its `- [x]`
    text: str          # the item and its continuation lines, nested items included
    closed: dt.date | None  # the latest ISO date it carries; None when it carries none


def _extent(lines: list[str], i: int) -> int:
    """End (exclusive) of the list item at line i: its continuation lines and the items nested under it."""
    indent = len(_ITEM.match(lines[i]).group(1))
    j = i + 1
    while j < len(lines):
        line = lines[j]
        if line.strip() == "" or line.startswith("#") or not line.startswith(" "):
            break
        nested = _ITEM.match(line) or re.match(r"^(\s*)- ", line)
        if nested and len(nested.group(1)) <= indent:
            break
        j += 1
    return j


def ticked_items(text: str) -> list[TickedItem]:
    """Every `- [x]` item at any indentation, with its close date."""
    lines = text.split("\n")
    found = []
    for i, line in enumerate(lines):
        match = _ITEM.match(line)
        if match and match.group(2).lower() == "x":
            block = "\n".join(lines[i:_extent(lines, i)])
            dates = sorted(_ISO.findall(block))
            closed = dt.date.fromisoformat(dates[-1]) if dates else None
            found.append(TickedItem(i + 1, block, closed))
    return found


def stale_items(text: str, today: dt.date, max_age_days: int = TICKED_ITEM_MAX_AGE_DAYS) -> list[str]:
    """The ticked items that should have left for the archive: closed over `max_age_days` ago, or undated."""
    out = []
    for item in ticked_items(text):
        head = item.text.split("\n", 1)[0].strip()[:80]
        if item.closed is None:
            out.append(f"line {item.line}: ticked item carries no date; add the day it was closed: {head!r}")
        elif (today - item.closed).days > max_age_days:
            out.append(f"line {item.line}: ticked {(today - item.closed).days} days ago ({item.closed}); "
                       f"move it to {TODO_ARCHIVE}: {head!r}")
    return out


def size_findings(sizes: dict[str, int], limits: dict[str, int]) -> list[str]:
    return [f"{name} is {size:,} bytes, over its {limits[name]:,}-byte limit; move its oldest part to the archive"
            for name, size in sizes.items() if size > limits[name]]


# ------------------------------------------------------------------------------------------------ the real files

def test_todo_holds_no_stale_ticked_item():
    text = (REPO / TODO).read_text(encoding="utf-8")
    assert stale_items(text, dt.date.today()) == []


def test_live_files_are_within_their_limits():
    sizes = {name: (REPO / name).stat().st_size for name in (TODO, CHANGELOG)}
    limits = {TODO: TODO_LIMIT_BYTES, CHANGELOG: CHANGELOG_LIMIT_BYTES}
    assert size_findings(sizes, limits) == []


def test_each_live_file_names_its_archive():
    """The pointer each live file carries resolves, so a reader of the short file finds the rest."""
    for live, archive in ((TODO, TODO_ARCHIVE), (CHANGELOG, CHANGELOG_ARCHIVE)):
        assert (REPO / archive).is_file(), archive
        assert archive in (REPO / live).read_text(encoding="utf-8"), f"{live} does not name {archive}"


# ------------------------------------------------------------------------------------------------ the scan itself

_TODAY = dt.date(2026, 10, 22)


def _item(days_ago: int, ticked: bool = True, indent: str = "") -> str:
    day = _TODAY - dt.timedelta(days=days_ago)
    return f"{indent}- [{'x' if ticked else ' '}] **An item.** *Done {day}.*\n"


def test_a_ticked_item_older_than_the_limit_is_reported():
    text = "# TODO\n\n## A\n\n" + _item(TICKED_ITEM_MAX_AGE_DAYS + 1)
    found = stale_items(text, _TODAY)
    assert len(found) == 1 and "move it to docs/TODO_archive.md" in found[0]


def test_a_recent_ticked_item_and_an_old_open_item_pass():
    text = "# TODO\n\n## A\n\n" + _item(TICKED_ITEM_MAX_AGE_DAYS) + _item(400, ticked=False)
    assert stale_items(text, _TODAY) == []


def test_an_undated_ticked_item_is_reported():
    assert stale_items("- [x] **Done, no date.**\n", _TODAY) == [
        "line 1: ticked item carries no date; add the day it was closed: '- [x] **Done, no date.**'"]


def test_the_latest_date_in_the_item_is_its_close_date():
    """An item that recalls an old date and was closed recently is recent; the date may sit on a continuation line."""
    old, recent = _TODAY - dt.timedelta(days=200), _TODAY - dt.timedelta(days=3)
    text = f"- [x] ~~**Decide X** (raised {old}).~~\n  *Decided {recent}.*\n"
    assert [i.closed for i in ticked_items(text)] == [recent]
    assert stale_items(text, _TODAY) == []


def test_a_nested_ticked_item_under_an_open_item_is_read():
    old = _TODAY - dt.timedelta(days=90)
    text = (f"- [ ] **Open parent** ({old}).\n"
            f"  - a plain sub-bullet.\n"
            f"  - [x] a closed sub-item, done {old}.\n"
            f"    its continuation line.\n")
    items = ticked_items(text)
    assert [(i.line, i.closed) for i in items] == [(3, old)]
    assert items[0].text.endswith("its continuation line.")
    assert len(stale_items(text, _TODAY)) == 1


def test_an_items_block_stops_at_the_next_item_heading_or_blank_line():
    text = ("- [x] first, done 2026-01-01.\n"
            "- [x] second, done 2026-10-20.\n"
            "  continued 2026-10-21.\n"
            "\n"
            "## next\n"
            "- [ ] open, 2020-01-01.\n")
    assert [(i.line, str(i.closed)) for i in ticked_items(text)] == [(1, "2026-01-01"), (2, "2026-10-21")]


@pytest.mark.parametrize("size, over", [(TODO_LIMIT_BYTES, False), (TODO_LIMIT_BYTES + 1, True)])
def test_a_file_over_its_limit_is_reported(size, over):
    found = size_findings({TODO: size}, {TODO: TODO_LIMIT_BYTES})
    assert (len(found) == 1) is over
