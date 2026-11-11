"""Smoke test of T-176's trace of `M11.C.48` (`tools/t176_negative_run_trace.py`), on one seed.

Purpose: the trace sets other baselines beside `M11.C.48`'s, so its reading holds only if it reads the run the
         criterion reads. This test runs the tool's seed on the Phase B family and checks that its arms are
         `negative_run_values`'s, that its first cutoffs are the readout's, and that the page renders and keeps its
         hand-written "Reading" section.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.48
Tests:   this file
"""

from __future__ import annotations

import math

import tools.t176_negative_run_trace as t176
from src.bowen.ensemble.criteria_d2 import negative_run_values


def test_m11c48_t176_trace_reproduces_the_criterions_arms(tmp_path):
    """Seed 0 of the Phase B family: the trace's treatment and baseline are the criterion's arms, its cutoff count
    and the cutoffs it leaves out are the criterion's, and the criterion's contrast row reads each cutoff's own run less its
    uncut baseline. Since spec revision 47 (T-179) the criterion's row reads only cut ties with a contact by their
    week, the readout's condition, and revision 46's row reads the no-contact ones at a run of 0 beside it. Mutations:
    running the seed without the declared spell (``spells=()``) changes the arms (seen on seed 0 while the tool was
    written); restoring revision 46's condition in the readout (a cut tie always in contact) stops the trace."""
    row = t176.one_seed(("phase_c", 0))
    (run, uncut, firsts, unmatched, silent, lopsided, _, _, _,
     _) = negative_run_values("phase_c", 0, t176.weeks_of("phase_c"))
    assert math.isclose(row["treatment"], run) and math.isclose(row["baseline"], uncut)
    assert len(row["cutoffs"]) == firsts
    # T-185: a cutoff with no uncut tie of its own sides is left out of the arms as an unmatched one is, and the
    # trace counts it apart, as the readout does (learning-qa 2026-11-10: the summary pooled it into "unmatched")
    assert sum(not c["matched"] for c in row["cutoffs"]) == unmatched + silent + lopsided
    assert silent == sum(not c["own_contact"] for c in row["cutoffs"]) > 0
    one = t176.summarise([row])["phase_c"]
    assert (one["unmatched"], one["unsided"], one["no_contact"], one["matched"]) == (
        unmatched, lopsided, silent, firsts - unmatched - lopsided - silent)
    rev46 = dict(t176.CONTRASTS)["revision 46's readout: a cut tie with no contact by its cutoff's week read at a run of 0"]
    old = [c["run"] - c["uncut"] for c in row["cutoffs"] if c["matched_rev46"]]
    assert math.isclose(t176.seed_mean(row["cutoffs"], rev46), math.fsum(old) / len(old))
    assert len(old) > sum(c["matched"] for c in row["cutoffs"])  # revision 46 read the no-contact cutoffs too
    criterion = dict(t176.CONTRASTS)[t176.CONTRASTS[0][0]]
    assert math.isclose(t176.seed_mean(row["cutoffs"], criterion), run - uncut)
    page = tmp_path / "trace.md"
    page.write_text("old\n\n## Reading\n\nhand-written\n", encoding="utf-8")
    keep = page.read_text(encoding="utf-8")
    text = t176.render(t176.summarise([row, row]), "e" * 12, "2026-11-02", keep[keep.index(t176.READING):])
    assert "Not a real family (`M11.F.9`)." in text and text.endswith("hand-written\n")


def test_m11c48_t176_rows_the_reading_rests_on_read_what_they_say():
    """Seed 0 of the Phase B family, each cutoff checked against the readout's own functions called afresh: the contact
    flag, the split of the uncut ties into those cut off later and never, and the run k weeks before on the cut tie and
    on the uncut ties. A cut tie with no contact stands on a run of 0 and is out of the contact-only rows (learning-qa,
    2026-11-02). Every uncut set read here is the readout's own since T-185: the uncut ties in contact with as many
    sides as the cut tie, and at least one cutoff on this seed has fewer of those than ties in contact, so a trace that
    skipped the sides filter would fail. Mutations: swapping ``uncut_never`` and ``uncut_later``, reading the lag at
    ``week - k`` in place of ``week + 1 - k``, or dropping the sides filter, fails here."""
    from src.bowen.readouts.negative_runs import (acted_across, contacts, first_cutoffs, run_at, runs_by_week,
                                                  sides)

    records, kinds, members, declared, relations = t176.run_seed("phase_c", 0)
    row = t176.trace(records, kinds, members, declared, relations)
    per_tie = {}
    for c in contacts(records, kinds, members):
        if c.tie not in declared:
            per_tie.setdefault(c.tie, []).append(c)
    runs = {t: runs_by_week(cs) for t, cs in per_tie.items()}
    cut = {t: w for t, w in first_cutoffs(records, members).items() if t not in declared}
    acted = acted_across(records)
    assert declared
    contact_only = t176.CONTRASTS[0][1]  # the criterion's row: cut ties with a contact only (T-179)
    assert {c["own_contact"] for c in row["cutoffs"]} == {True, False}
    assert any(c["lags"] for c in row["cutoffs"])
    assert any(c["uncut_later"] is not None for c in row["cutoffs"])  # the split is read, not None == None
    narrowed: list[bool] = []
    for f in row["cutoffs"]:
        tie, week = f["tie"], f["week"]
        assert cut[tie] == week
        assert f["own_contact"] == any(c.week <= week for c in per_tie.get(tie, []))
        if not f["own_contact"]:
            assert f["run"] == 0 and contact_only(f) is None
        in_contact = [t for t in runs if t != tie and cut.get(t, week + 1) > week and runs[t][0][0] <= week]
        uncut = [t for t in in_contact if sides(acted, t, week) == sides(acted, tie, week)]
        narrowed.append(len(uncut) < len(in_contact))
        assert f["matched_rev46"] == bool(in_contact)
        later = [run_at(runs[t], week + 1) for t in uncut if t in cut]
        assert f["uncut_later"] == (math.fsum(later) / len(later) if later else None)
        assert f["uncut_never"] is None  # every member tie on the Phase B family is cut off on seed 0
        for k, (own_before, uncut_before) in f["lags"].items():
            assert own_before == run_at(runs.get(tie, []), week + 1 - k)
            if not uncut:
                assert uncut_before is None
                continue
            assert math.isclose(uncut_before, math.fsum(run_at(runs[t], week + 1 - k) for t in uncut) / len(uncut))
    assert any(narrowed)  # the sides filter bites on this seed: the rows above are not the in-contact set
    assert row["never_cut"] == 0 and row["ties"] == len(cut)


def test_m11c48_t185_the_trace_counts_unsided_cutoffs_apart_from_unmatched_ones():
    """Seed 1 of the Phase B family, where one first cutoff is in contact and has uncut ties in contact but none of
    its own sides. The readout counts it under `unsided_cutoffs`, so the trace's summary must too: pooled into
    "unmatched", as the summary read it until this fix, the page would report one unmatched cutoff too many beside a
    criterion that reports it apart (learning-qa 2026-11-10, P2).
    Mutation: counting ``own_contact and not matched`` as unmatched again, the pre-fix formula, fails here."""
    row = t176.one_seed(("phase_c", 1))
    run, uncut, firsts, unmatched, silent, lopsided, _, _, _, _ = negative_run_values(
        "phase_c", 1, t176.weeks_of("phase_c"))
    one = t176.summarise([row])["phase_c"]
    assert lopsided > 0, "seed 1 no longer has an unsided first cutoff; pick a seed that does"
    assert (one["unmatched"], one["unsided"], one["no_contact"]) == (unmatched, lopsided, silent)
    assert one["unmatched"] + one["unsided"] == sum(c["own_contact"] and not c["matched"] for c in row["cutoffs"])

