"""The page of what version 2 stage 1 changed in the records is current, and lists what it says it lists (S1.10b).

Purpose: `docs/version_2_stage_1_changes.md` is generated from the records at the tag `version-1` and as committed
         now. A hand edit, or a record regenerated without the page, fails the first test. The others check the
         comparison on rows built here: a changed verdict is listed, an effect that moved past the tag's half-width
         is listed and one that moved less is not (P7), the D2 record's variants are matched to the tag's rows by
         position, a value record lists a changed value, every field of a row is compared, and a record that was not
         regenerated, or that compares no row, is refused (P35).
Spec:    docs/implementation_plan_version_2.md#S1.10b
Tests:   this file
"""

from __future__ import annotations

from unittest import mock

import pytest

from tools import record_changes as rc

CRITERIA = rc.Record("x", "criterion", ("criterion",), "outcome")
D2 = rc.Record("d2", "criterion", ("criterion",), "outcome", positional=True)


def test_s1_10b_changes_page_is_current():
    assert rc.OUT.read_text(encoding="utf-8") == rc.build(), "stale: rerun python3 tools/record_changes.py"


def _row(criterion, outcome, *readouts):
    return {"criterion": criterion, "outcome": outcome,
            "readouts": [{"readout": n, "mean_difference": m, "half_width": h, "p": 0.01} for n, m, h in readouts]}


def test_s1_10b_a_changed_verdict_and_a_moved_effect_are_listed():
    then = [_row("A", "PASS", ("r", 1.0, 0.5)), _row("B", "PASS", ("r", 1.0, 0.5)), _row("C", "PASS", ("r", 1.0, 0.5)),
            _row("D", "FAIL", ("r", 1.0, 0.5))]
    now = [_row("A", "FAIL", ("r", 0.2, 0.5)), _row("B", "PASS", ("r", 1.6, 0.5)), _row("C", "PASS", ("r", 1.4, 0.5)),
           _row("E", "PASS", ("r", 1.0, 0.5))]
    c = rc.compare(then, now, CRITERIA)
    assert c.compared == 3 and [k for k, _, _ in c.changed] == ["A"]
    assert c.differ == 4  # A's outcome and mean difference, B's and C's mean difference
    assert [(k, name) for k, name, _, _ in c.moved] == [("B", "r")]  # C moved 0.4, inside the tag's 0.5
    assert [k for k, _ in c.gone] == ["D"] and [k for k, _ in c.new] == ["E"]
    page = rc.render([(CRITERIA, "a" * 64, "b" * 64, c)])
    assert "| A | PASS | FAIL |" in page and "| B | r |" in page and "| C | r |" not in page
    assert "Rows only at the tag: 1" in page and "| D | FAIL |" in page and "| E | PASS |" in page


def test_s1_10b_a_readout_on_one_side_only_is_listed():
    """A readout the tag's row has and the current row does not, or the reverse, is a moved effect and a difference,
    whatever its size (fix re-sweep, 2026-10-19)."""
    then = [_row("A", "PASS", ("r", 1.0, 0.5), ("gone", 0.1, 0.5))]
    now = [_row("A", "PASS", ("r", 1.0, 0.5), ("added", 0.1, 0.5))]
    c = rc.compare(then, now, CRITERIA)
    assert [(k, name) for k, name, _, _ in c.moved] == [("A", "added"), ("A", "gone")]
    assert c.differ == 8  # each one-sided readout's four fields


def test_s1_10b_every_field_is_compared():
    """A row whose verdict and mean differences are the tag's but whose p, half-width, seed count or flags are not is
    counted and listed, and the record is not called unchanged; a NaN on both sides is no difference, and a mean
    difference that became NaN is a moved effect (narrow re-sweep, 2026-10-19: the tool compared only verdicts and
    mean differences while the page said every value)."""
    then = [{**_row("A", "PASS", ("r", 1.0, 0.5)), "seeds": 100, "flags": []}, _row("B", "PASS", ("r", 1.0, 0.5))]
    now = [{**_row("A", "PASS", ("r", 1.0, 0.5)), "seeds": 50, "flags": ["low"]}, _row("B", "PASS", ("r", 1.0, 0.5))]
    now[0]["readouts"][0].update(p=0.2, half_width=0.6)
    c = rc.compare(then, now, CRITERIA)
    assert not c.changed and not c.moved
    assert [(k, name) for k, name, _, _ in c.values] == [
        ("A", "flags"), ("A", "readouts.r.half_width"), ("A", "readouts.r.p"), ("A", "seeds")]
    page = rc.render([(CRITERIA, "a" * 64, "b" * 64, c)])
    assert "No change:" not in page and "Fields not exactly the tag's, in rows on both sides: 4." in page
    assert "| A | readouts.r.p | 0.01 | 0.2 |" in page and "| A | seeds | 100 | 50 |" in page
    nan = float("nan")
    assert not rc.compare([_row("A", "PASS", ("r", nan, 0.5))], [_row("A", "PASS", ("r", nan, 0.5))], CRITERIA).values
    became = rc.compare(then[1:], [_row("B", "PASS", ("r", nan, 0.5))], CRITERIA)
    assert [(k, name) for k, name, _, _ in became.moved] == [("B", "r")]
    unchanged = rc.render([(CRITERIA, "a" * 64, "b" * 64, rc.compare(then, then, CRITERIA))])
    assert "No change: 2 of 2 rows are exactly the tag's, every field." in unchanged


def test_s1_10b_the_tags_d2_rows_are_matched_by_position():
    """At the tag the D2 record printed `M11.C.10` for its family-of-origin variant, and `M11.C.2` for the
    twelve-person family (S1.7b): both are matched to their current keys, so a change in either is listed."""
    then = [_row("M11.C.10", "FAIL"), _row("M11.C.10", "PASS"), _row("M11.C.2", "FAIL")]
    now = [_row("M11.C.10", "FAIL"), _row("M11.C.10[family of origin]", "FAIL"),
           _row("M11.C.2[twelve-person family]", "PASS")]
    c = rc.compare(then, now, D2)
    assert c.compared == 3 and not c.gone and not c.new
    assert [k for k, _, _ in c.changed] == ["M11.C.10[family of origin]", "M11.C.2[twelve-person family]"]
    with pytest.raises(ValueError, match="row 2"):
        rc.compare(then, now[:2] + [_row("M11.C.6", "FAIL")], D2)


def test_s1_10b_a_value_record_lists_a_changed_value():
    record = rc.Record("occ", "variant", ("variant", "arm"), None)
    then = [{"variant": "v", "arm": "a", "whole_run": {"x": 0.5, "y": 1.0}}]
    now = [{"variant": "v", "arm": "a", "whole_run": {"x": 0.5, "y": 1.25}}]
    c = rc.compare(then, now, record)
    assert [k for k, _, _ in c.changed] == ["v · a"] and c.differ == 1
    assert "| v · a | whole_run.y | 1 | 1.25 |" in rc.render([(record, "a" * 64, "b" * 64, c)])
    assert not rc.compare(then, then, record).changed


def test_s1_10b_a_record_not_regenerated_is_refused(tmp_path):
    text = "# r\n\ncode_hash: " + "a" * 64 + "\n\n```json\n[]\n```\n"
    (tmp_path / "x").write_text(text, encoding="utf-8")
    with mock.patch.object(rc, "RECORDS", (CRITERIA,)), mock.patch.object(rc, "tag_text", return_value=text), \
            mock.patch.object(rc, "REPO", tmp_path):
        with pytest.raises(SystemExit, match="not regenerated"):
            rc.build()
        (tmp_path / "x").write_text(text.replace("code_hash: ", "hash: "), encoding="utf-8")
        with pytest.raises(SystemExit, match="no code_hash or engine_hash line"):
            rc.build()
        regenerated = text.replace("a" * 64, "b" * 64)
        (tmp_path / "x").write_text(regenerated, encoding="utf-8")
        with mock.patch.object(rc, "tag_text", return_value=text.replace("code_hash: ", "hash: ")), \
                pytest.raises(SystemExit, match="no code_hash or engine_hash line"):
            rc.build()  # the tag's side without a hash line is refused too
        with pytest.raises(SystemExit, match="nothing was compared"):
            rc.build()  # regenerated, but no row on both sides: refused, not "no change" (narrow re-sweep)
        rows = '[{"criterion": "A", "outcome": "PASS", "readouts": []}]'
        with mock.patch.object(rc, "tag_text", return_value=text.replace("[]", rows)):
            (tmp_path / "x").write_text(regenerated.replace("[]", rows), encoding="utf-8")
            assert "No change: 1 of 1 rows" in rc.build()  # regenerated with rows: the page is written
