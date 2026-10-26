"""The open batch's mark cannot hide a record that is stale for any reason but D31, nor outlive the batch.

Purpose: check `tests/open_batch.py`: a marked test that fails is an expected failure only while the batch is open,
         its record is as the batch found it, and it passes with the hash computed on the tree the batch found; the
         listed records are all as found or all regenerated, never some of each; every listed record is read by a
         marked test, and no test carries the mark once the batch is closed.
Spec:    docs/implementation_plan_version_2_stage_3.md#3.6 (D31)
Tests:   this file
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

import open_batch  # tests/ is on the path: pytest imports its conftest by basename
from open_batch import AT_OPEN, stale_in_open_batch

REPO = Path(__file__).resolve().parents[1]
MARK = re.compile(r'^@stale_in_open_batch\("([^"]+)"\)$', re.M)
RECORD = "docs/phase_c_ensemble_record.md"
# The outcome an expected failure raises, named here so the test below is not read as a test marked xfail.
EXPECTED_FAILURE = pytest.xfail.Exception
STAMPED = "docs/phase_d_ensemble_record.md"  # prints the hash of the code alone (``code_hash()``, no tool) it was built on
TOOLED = "docs/phase_c_ensemble_record.md"  # prints ``code_hash(*HASHED_TOOLS)``, with its tool


def _marked() -> dict[str, list[str]]:
    """Each record named by a mark in the suite, with the files that mark a test for it."""
    found: dict[str, list[str]] = {}
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        for record in MARK.findall(path.read_text(encoding="utf-8")):
            found.setdefault(record, []).append(path.name)
    return found


def _stamped(record: str = STAMPED) -> str:
    return re.search(r"^code_hash: ([0-9a-f]{64})$", (REPO / record).read_text(encoding="utf-8"), re.M).group(1)


def test_d31_the_records_are_all_as_found_or_all_regenerated():
    """While `OPEN` is set every listed record is the file the batch found (D31 regenerates none before step 10); once
    all are regenerated the batch is over: set `OPEN` to None and take the marks off. Some of each is an error: a
    record regenerated alone would leave the others' tests excused while that one is checked."""
    if open_batch.OPEN is None:
        assert _marked() == {}, "the batch is closed: take the marks off the record-current tests"
    else:
        found = {r: open_batch.as_found(r) for r in AT_OPEN}
        assert all(found.values()), (
            f"regenerated while the batch is open: {sorted(r for r, f in found.items() if not f)}; D31 regenerates "
            "every record once, at step 10, and then the batch closes (OPEN = None, the marks off)")


def test_d31_every_listed_record_is_marked_and_no_other():
    if open_batch.OPEN is not None:
        assert sorted(_marked()) == sorted(AT_OPEN)


def test_d31_the_hash_as_found_is_the_hash_the_records_were_built_on():
    """The tree `hash_as_found` reads gives the hash a committed record prints, and the working tree's differs, so
    the rerun under it is a test of the batch's one cause."""
    from tools import ensemble_record

    if open_batch.OPEN is None:
        return
    with open_batch.hash_as_found():
        assert ensemble_record.code_hash() == _stamped()
        # with a tool (the path remapped into the tree as found): the Phase C ensemble record's stamp
        assert ensemble_record.code_hash(*ensemble_record.HASHED_TOOLS) == _stamped(TOOLED)
    assert ensemble_record.code_hash() != _stamped(), "the hash has not moved: no record is stale by the batch"


def test_d31_the_mark_excuses_only_a_failure_the_hash_explains(monkeypatch):
    """A failure is an expected failure while the batch is open, the record is unchanged, and the test passes with the
    hash the batch found; a failure that hash does not explain, a record changed since, or a closed batch lets it
    fail; a pass passes."""
    from tools import ensemble_record

    actual = hashlib.sha256((REPO / RECORD).read_bytes()).hexdigest()

    @stale_in_open_batch(RECORD)
    def stale():
        assert ensemble_record.code_hash() == _stamped(), "stale: rerun"

    @stale_in_open_batch(RECORD)
    def broken():
        raise AssertionError("a fault the hash does not explain")

    @stale_in_open_batch(RECORD)
    def current():
        return "ran"

    monkeypatch.setattr(open_batch, "OPEN", "a test batch")
    monkeypatch.setitem(AT_OPEN, RECORD, actual)
    with pytest.raises(EXPECTED_FAILURE, match="stale by decision: a test batch"):
        stale()
    original = ensemble_record.code_hash
    with pytest.raises(AssertionError, match="fails with the hash the batch found too"):
        broken()
    assert ensemble_record.code_hash is original, "the pinned hash outlived a failed rerun"
    assert current() == "ran"
    monkeypatch.setitem(AT_OPEN, RECORD, "0" * 64)  # the record changed after the batch opened
    with pytest.raises(AssertionError, match="stale: rerun"):
        stale()
    monkeypatch.setitem(AT_OPEN, RECORD, actual)
    monkeypatch.setattr(open_batch, "OPEN", None)  # the batch is closed
    with pytest.raises(AssertionError, match="stale: rerun"):
        stale()


def test_d31_a_record_the_batch_does_not_list_cannot_be_marked():
    with pytest.raises(KeyError, match="not a record the batch lists"):
        stale_in_open_batch("docs/no_such_record.md")
