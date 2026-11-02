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
    and unmatched count are the criterion's, and the criterion's contrast row reads each cutoff's own run less its
    uncut baseline. Mutation: running the seed without the declared spell (``spells=()``) changes the arms (seen on
    seed 0 while the tool was written)."""
    row = t176.one_seed(("phase_c", 0))
    run, uncut, firsts, unmatched, _ = negative_run_values("phase_c", 0, t176.weeks_of("phase_c"))
    assert math.isclose(row["treatment"], run) and math.isclose(row["baseline"], uncut)
    assert len(row["cutoffs"]) == firsts and sum(not c["matched"] for c in row["cutoffs"]) == unmatched
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
    2026-11-02). Mutations: swapping ``uncut_never`` and ``uncut_later``, or reading the lag at ``week - k`` in place of
    ``week + 1 - k``, fails here."""
    from src.bowen.readouts.negative_runs import contacts, first_cutoffs, run_at, runs_by_week

    records, kinds, members, declared, relations = t176.run_seed("phase_c", 0)
    row = t176.trace(records, kinds, members, declared, relations)
    per_tie = {}
    for c in contacts(records, kinds, members):
        if c.tie not in declared:
            per_tie.setdefault(c.tie, []).append(c)
    runs = {t: runs_by_week(cs) for t, cs in per_tie.items()}
    cut = {t: w for t, w in first_cutoffs(records, members).items() if t not in declared}
    assert declared
    contact_only = dict(t176.CONTRASTS)["criterion, cut ties with a contact by their cutoff's week only"]
    assert {c["own_contact"] for c in row["cutoffs"]} == {True, False}
    assert any(c["lags"] for c in row["cutoffs"])
    assert any(c["uncut_later"] is not None for c in row["cutoffs"])  # the split is read, not None == None
    for f in row["cutoffs"]:
        tie, week = f["tie"], f["week"]
        assert cut[tie] == week
        assert f["own_contact"] == any(c.week <= week for c in per_tie.get(tie, []))
        if not f["own_contact"]:
            assert f["run"] == 0 and contact_only(f) is None
        uncut = [t for t in runs if t != tie and cut.get(t, week + 1) > week and runs[t][0][0] <= week]
        later = [run_at(runs[t], week + 1) for t in uncut if t in cut]
        assert f["uncut_later"] == (math.fsum(later) / len(later) if later else None)
        assert f["uncut_never"] is None  # every member tie on the Phase B family is cut off on seed 0
        for k, (own_before, uncut_before) in f["lags"].items():
            assert own_before == run_at(runs.get(tie, []), week + 1 - k)
            if not uncut:
                assert uncut_before is None
                continue
            assert math.isclose(uncut_before, math.fsum(run_at(runs[t], week + 1 - k) for t in uncut) / len(uncut))
    assert row["never_cut"] == 0 and row["ties"] == len(cut)
