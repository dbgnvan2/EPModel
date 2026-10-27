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
import subprocess
from pathlib import Path

import pytest

import open_batch  # tests/ is on the path: pytest imports its conftest by basename
from open_batch import AT_OPEN, hash_as_found, stale_in_open_batch

REPO = Path(__file__).resolve().parents[1]
MARK = re.compile(r'^@stale_in_open_batch\("([^"]+)"\)$', re.M)
RECORD = "docs/phase_c_ensemble_record.md"
# The outcome an expected failure raises, named here so the test below is not read as a test marked xfail.
EXPECTED_FAILURE = pytest.xfail.Exception
STAMPED = "docs/phase_d_ensemble_record.md"  # prints the hash of the code alone (``code_hash()``, no tool) it was built on


def _marked() -> dict[str, list[str]]:
    """Each record named by a mark in the suite, with the files that mark a test for it."""
    found: dict[str, list[str]] = {}
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        for record in MARK.findall(path.read_text(encoding="utf-8")):
            found.setdefault(record, []).append(path.name)
    return found


def _stamped(record: str = STAMPED) -> str:
    """The code hash ``record`` printed at ``OPENED_AT``, the commit whose records the batch kept (read from git, so the
    checks below still test the machinery once the records are regenerated and the batch is closed)."""
    text = subprocess.run(["git", "-C", str(REPO), "show", f"{open_batch.OPENED_AT}:{record}"], capture_output=True,
                          text=True, check=True, timeout=60).stdout
    return re.search(r"^code_hash: ([0-9a-f]{64})$", text, re.M).group(1)


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

    with open_batch.hash_as_found():
        assert ensemble_record.code_hash() == _stamped()
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


def test_d31_the_rerun_leaves_out_a_mutant_added_since_the_batch_opened():
    """Step 7 adds mutants to the Phase D mutation record's tool. A mutant whose criterion does not pass is listed in
    the record's "Not run" section, so it changes the render as well as the hash; the rerun reads the tool's mutants
    as the batch found them, and restores the current list after."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("open_batch_check.phase_d_mutation_record",
                                                  REPO / "tools" / "phase_d_mutation_record.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    now = {m.id for m in tool.MUTANTS}
    then = subprocess.run(["git", "-C", str(REPO), "show", f"{open_batch.OPENED_AT}:tools/phase_d_mutation_record.py"],
                          capture_output=True, text=True, check=True, timeout=60).stdout
    assert "displaced-coping-removed" in now and 'Mutant("displaced-coping-removed"' not in then
    with hash_as_found({"tool": tool}):
        kept = {m.id for m in tool.MUTANTS}
        witness = next(m for m in tool.MUTANTS if m.id == "witness-path-removed")
    assert kept == {i for i in now if f'Mutant("{i}"' in then} and "development-fixed-at-anchor" in kept
    assert "displaced-coping-removed" not in kept
    assert witness.criteria == ("M11.C.2[twelve-person family]",)              # step 7 added M11.C.2; cut here
    assert {m.id for m in tool.MUTANTS} == now, "the found mutants outlived the rerun"
    assert "M11.C.2" in next(m for m in tool.MUTANTS if m.id == "witness-path-removed").criteria



def test_d31_a_criterion_added_to_a_mutant_is_cut_by_its_own_definition():
    """Step 9 adds ``M11.C.45`` to the Phase C ``one-sided-deviation``. Other mutants quoted ``"M11.C.45"`` in the file
    the batch found, so a cut by the whole file kept it and the record rendered a "Not run" line it did not have
    (the step 9 suite run, 2026-10-26); the cut reads the mutant's own definition, and the list is restored after."""
    from tools import mutation_record

    then = subprocess.run(["git", "-C", str(REPO), "show", f"{open_batch.OPENED_AT}:tools/mutation_record.py"],
                          capture_output=True, text=True, check=True, timeout=60).stdout
    now = next(m for m in mutation_record.MUTANTS if m.id == "one-sided-deviation").criteria
    assert "M11.C.45" in now and '"M11.C.45"' in then
    with hash_as_found({"tool": mutation_record}):
        cut = next(m for m in mutation_record.MUTANTS if m.id == "one-sided-deviation").criteria
    assert cut == ("M11.C.27[", "M11.C.44")
    assert next(m for m in mutation_record.MUTANTS if m.id == "one-sided-deviation").criteria == now

def test_d31_a_tool_that_imports_mutants_gets_the_defining_files_cut():
    """The diagnostic record's tool imports the mutation record's ``MUTANTS``; its own file defines none. Cut by its own
    file, every mutant would go and its lookup would fail (the step 7 suite run, 2026-10-26: StopIteration). It gets
    the cut of the file that defines the tuple, as the same object, and both are restored after."""
    from tools import diagnostic_record, mutation_record

    before = diagnostic_record.MUTANTS
    assert before is mutation_record.MUTANTS and before
    with hash_as_found({"tool": diagnostic_record}):
        assert diagnostic_record.MUTANTS == mutation_record.MUTANTS and diagnostic_record.MUTANTS
        assert {m.id for m in diagnostic_record.MUTANTS} >= {d.mutant for d in diagnostic_record.DIAGNOSTICS if d.mutant}
    assert diagnostic_record.MUTANTS is before and mutation_record.MUTANTS is before


def test_d31_a_record_the_batch_does_not_list_cannot_be_marked():
    with pytest.raises(KeyError, match="not a record the batch lists"):
        stale_in_open_batch("docs/no_such_record.md")


def test_d31_the_rerun_leaves_out_a_criterion_added_since_the_batch_opened():
    """Step 8 adds `M11.C.49` to the Phase C criteria; the ensemble record the batch keeps has no row for it. The rerun
    reads the table as the batch found it, in the module and in a namespace that imported it, and restores it after;
    a keyed variant is kept by its base id."""
    from src.bowen.ensemble import criteria, criteria_d

    now = criteria.CRITERIA
    namespace = {"CRITERIA": now, "tool": criteria}
    with hash_as_found(namespace):
        assert "M11.C.49" not in criteria.CRITERIA and "M11.C.49" not in namespace["CRITERIA"]
        assert set(criteria.CRITERIA) == set(now) - {"M11.C.49"}
        assert namespace["CRITERIA"] is criteria.CRITERIA
    assert criteria.CRITERIA is now and namespace["CRITERIA"] is now, "the found table outlived the rerun"
    keyed = [k for k in criteria_d.CRITERIA_D if "[" in k]
    assert keyed
    with hash_as_found({"tool": criteria_d}):
        assert set(keyed) <= set(criteria_d.CRITERIA_D)


def test_d31_a_table_is_cut_by_the_file_that_defines_it():
    """A module of `src/bowen/ensemble` that holds another's table by name (here a stand-in loaded from
    `criteria_d.py`'s path holding `criteria.CRITERIA`) gets the defining file's cut, the same object, not one made
    from its own file, which quotes none of `CRITERIA`'s ids (the review of 2026-10-26)."""
    import types

    from src.bowen.ensemble import criteria, criteria_d

    holder = types.ModuleType("stand_in")
    holder.__file__ = criteria_d.__file__
    holder.CRITERIA = criteria.CRITERIA
    with hash_as_found({"holder": holder}):
        assert holder.CRITERIA is criteria.CRITERIA
        assert set(holder.CRITERIA) == set(criteria.CRITERIA) and "M11.C.1" in holder.CRITERIA
        assert "M11.C.49" not in holder.CRITERIA
    assert holder.CRITERIA is criteria.CRITERIA and "M11.C.49" in holder.CRITERIA


def test_d31_the_status_tool_reads_the_criteria_the_batch_found():
    """`registry_as_found` leaves a criterion added since the batch opened out of the registry and the spec's criteria
    the status tool reads, keeps every other, and restores both on exit."""
    import tools.version_1_status as v1

    before_registered, before_spec = v1.registered(), v1.spec_criteria()
    assert "M11.C.49" in before_registered and "M11.C.49" in before_spec
    with open_batch.registry_as_found() as added:
        assert added == {"M11.C.49"}
        assert v1.registered() == before_registered - {"M11.C.49"}
        assert set(v1.spec_criteria()) == set(before_spec) - {"M11.C.49"}
    assert v1.registered() == before_registered and v1.spec_criteria() == before_spec
