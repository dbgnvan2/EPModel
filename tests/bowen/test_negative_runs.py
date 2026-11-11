"""`M11.C.48`'s readout on hand-built records (version 2 group G1, spec revision 36).
Purpose: the run arithmetic of `src/bowen/readouts/negative_runs.py` on records written here, so each definition is
         checked on its own: the run before a tie's first cutoff, the matched-week baseline on the uncut ties, a week
         holding both contacts, refusals, a declared cut-off tie, and a tie with no contact in the window. Unit tests
         of a readout with invented definitions; none is a finding (`M11.5`).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.48
Tests:   this file
"""
from __future__ import annotations

import pytest

from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord, EmittedRecord
from src.bowen.io.load import load_event_kinds
from src.bowen.readouts.negative_runs import (FirstCutoff, acted_across, arm_values, contacts, first_cutoff_runs,
                                               first_cutoffs, runs_by_week, sides)

KINDS = load_event_kinds()
A, B, C, D, E, F = (PersonId(x) for x in "abcdef")
AB, CD, EF = TieId.of(A, B), TieId.of(C, D), TieId.of(E, F)


def act(kind, sender, target, week, index=0):
    return EmittedRecord(Event(
        id=EventId(week, str(sender.value), index), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
        targets=(target,), intensity=10.0, timestamp=week, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED))


def refused(record):
    return EffectRecord(record.event.timestamp, "refusal", record.event.id, acute_anxiety=((record.event.sender, 1.0),))


def test_m11c48_run_before_the_first_cutoff_against_uncut_ties():
    """Tie a~b takes three conflicts and is cut at week 4; c~d took one, a positive contact, then one more. The run
    before the cutoff is 3; the baseline is c~d's run standing in week 4, which is 1 (the positive contact ended the
    first run); the longest run c~d reached, also 1, is reported beside it."""
    records = [act("CONFLICT", A, B, 1), act("CONFLICT", B, A, 2), act("CONFLICT", A, B, 3), act("CUTOFF", A, B, 4),
               act("CONFLICT", C, D, 1), act("STAY-IN-CONTACT", C, D, 2), act("CONFLICT", D, C, 3)]
    assert first_cutoffs(records) == {AB: 4}
    firsts = first_cutoff_runs(records, KINDS)
    assert [(f.tie, f.week, f.run, f.uncut, f.longest_uncut) for f in firsts] == [(AB, 4, 3, 1.0, 1)]
    assert arm_values(firsts) == (3.0, 1.0)
    # the direction the criterion reads: the cut tie's run against the uncut tie's, at the same week
    assert firsts[0].run > firsts[0].uncut


def test_m11c48_a_week_with_both_contacts_keeps_the_run():
    """Week 1 holds a conflict and a positive contact on a~b: the negative applies and the run is 1, not 0; the week 3
    cutoff's own week counts its conflict, and the cutoff itself is not a contact."""
    records = [act("CONFLICT", A, B, 1, 0), act("STAY-IN-CONTACT", B, A, 1, 1), act("CONFLICT", A, B, 2),
               act("CONFLICT", B, A, 3, 0), act("CUTOFF", A, B, 3, 1)]
    tie = [c for c in contacts(records, KINDS) if c.tie == AB]
    assert runs_by_week(tie) == [(1, 1), (2, 2), (3, 3)]
    assert first_cutoff_runs(records, KINDS)[0].run == 3


def test_m11c48_a_refusal_is_a_negative_contact_counted_once():
    """A refused approach is a negative contact whatever was refused; a refused conflict is one contact, not two."""
    approach, conflict = act("TRIANGLE", B, A, 1), act("CONFLICT", B, A, 2)
    records = [approach, refused(approach), conflict, refused(conflict), act("CUTOFF", A, B, 3)]
    assert [c.negative for c in contacts(records, KINDS)] == [True, True]
    assert first_cutoff_runs(records, KINDS)[0].run == 2


def test_m11c48_a_declared_cut_off_tie_is_left_out_on_both_sides():
    """e~f is declared cut off and takes five conflicts with no CUTOFF emitted: it is no first cutoff and no uncut tie.
    Without the exclusion it would be the baseline's longest run; with it, c~d's standing run is the baseline. And a
    tie whose first contact comes after the cutoff's week (c~d's second contact pattern, here on a~b's week 4 window)
    is not an uncut tie in that window."""
    records = ([act("CONFLICT", E, F, w) for w in range(1, 6)] +
               [act("CONFLICT", A, B, 1), act("CONFLICT", A, B, 2), act("CUTOFF", A, B, 4), act("CONFLICT", C, D, 3)])
    plain = first_cutoff_runs(records, KINDS)
    assert [(f.tie, f.uncut, f.longest_uncut) for f in plain] == [(AB, 2.5, 4)]  # e~f (4 by week 4) and c~d (1), their mean
    declared = first_cutoff_runs(records, KINDS, declared_cut=frozenset({EF}))
    assert [(f.tie, f.run, f.uncut, f.longest_uncut) for f in declared] == [(AB, 2, 1.0, 1)]
    late = [r for r in records if r.event.targets != (D,)] + [act("CONFLICT", C, D, 9)]  # c~d's first contact is week 9
    assert [(f.uncut, f.longest_uncut) for f in first_cutoff_runs(late, KINDS)] == [(4.0, 4)]  # e~f alone; c~d not counted


def test_m11c48_two_first_cutoffs_on_one_seed():
    """Two ties cut off on one seed: a tie cut later is an uncut tie at the earlier cutoff's week and not at its own;
    a tie's first cutoff is its earliest `CUTOFF`, not its last; and the arms average over every first cutoff."""
    records = ([act("CONFLICT", A, B, 1), act("CONFLICT", A, B, 2), act("CUTOFF", A, B, 2)] +
               [act("CONFLICT", C, D, w) for w in range(1, 5)] + [act("CUTOFF", C, D, 4), act("CUTOFF", C, D, 8)] +
               [act("CONFLICT", E, F, 4)])
    firsts = first_cutoff_runs(records, KINDS)
    assert [(f.tie, f.week, f.run, f.uncut, f.longest_uncut) for f in firsts] == [(AB, 2, 2, 2.0, 2), (CD, 4, 4, 1.0, 1)]
    assert arm_values(firsts) == (3.0, 1.5)


def test_m11c48_members_filter_and_no_cutoff_seed():
    """Contacts with a person outside ``members`` are not read; a seed with no first cutoff reads 0 in both arms."""
    records = [act("CONFLICT", A, B, 1), act("CONFLICT", A, E, 1), act("CUTOFF", A, B, 2)]
    assert {c.tie for c in contacts(records, KINDS, frozenset({A, B}))} == {AB}
    assert first_cutoffs(records, frozenset({A, E})) == {}
    assert arm_values(first_cutoff_runs([act("CONFLICT", A, B, 1)], KINDS)) == (0.0, 0.0)


def test_m11c48_t123_an_unmatched_first_cutoff_is_left_out_of_both_arms_and_counted():
    """T-123 (learnings P7; Claude's recommendation 2026-10-29, pending owner review): a first cutoff with no uncut tie
    in its week has no baseline. Counting it as 0 there and keeping its run in the treatment biases the comparison
    toward a pass, so it is left out of both arms and counted. a~b is cut at week 2 with no other tie in contact by then
    (unmatched); c~d is cut at week 6 with e~f uncut (matched)."""
    from src.bowen.readouts.negative_runs import unmatched

    records = ([act("CONFLICT", A, B, 1), act("CONFLICT", A, B, 2), act("CUTOFF", A, B, 2)] +
               [act("CONFLICT", C, D, w) for w in range(3, 7)] + [act("CUTOFF", C, D, 6)] +
               [act("CONFLICT", E, F, 4)])
    firsts = first_cutoff_runs(records, KINDS)
    assert [(f.tie, f.matched) for f in firsts] == [(AB, False), (CD, True)]
    assert arm_values(firsts) == (4.0, 1.0)  # c~d's run of 4 against e~f's 1; a~b's 2 against nothing is not read
    assert unmatched(firsts) == 1
    alone = first_cutoff_runs(records[:3], KINDS)  # a seed whose only first cutoff is unmatched reads 0 in both
    assert arm_values(alone) == (0.0, 0.0) and unmatched(alone) == 1
    # An uncut tie in contact whose standing run is 0 (its last contact positive) is a baseline of 0, not no baseline:
    # matched means a tie was uncut and in contact by the cutoff's week, not that its run was above 0 (learning-qa).
    zero = records[:3] + [act("CONFLICT", E, F, 1), act("STAY-IN-CONTACT", E, F, 2)]
    firsts = first_cutoff_runs(zero, KINDS)
    assert [(f.tie, f.matched, f.uncut) for f in firsts] == [(AB, True, 0.0)]
    assert arm_values(firsts) == (2.0, 0.0) and unmatched(firsts) == 0


def test_m11c48_t185_a_first_cutoff_cannot_disagree_with_its_own_exit():
    """Each first cutoff is matched, no-contact, unmatched or unsided, and only one of them (spec revision 57), so the
    flags are checked where the record is built: a record that says it is unmatched while a same-sided uncut tie stood
    that week would be counted under no exit at all, and the four counts would not sum to the cutoffs (learnings P2).
    Mutation: dropping the whole check lets such a record through, and `unmatched` + `no_contact` + `unsided` + the
    matched ones no longer sum (measured 2026-11-10). It must be the whole check: either clause alone catches all four
    of the records below, so removing one of the two leaves this test green, and neither clause is proved on its own."""
    FirstCutoff(TieId.of(A, B), 3, 2, 1.0, 2)  # the matched default
    FirstCutoff(TieId.of(A, B), 3, 0, 0.0, 0, matched=False, contact=False, any_uncut=False, sided=False)
    FirstCutoff(TieId.of(A, B), 3, 2, 0.0, 0, matched=False, contact=True, any_uncut=False, sided=False)
    FirstCutoff(TieId.of(A, B), 3, 2, 0.0, 0, matched=False, contact=True, any_uncut=True, sided=False)
    for bad in ({"matched": False, "contact": True, "any_uncut": True, "sided": True},
                {"matched": True, "contact": True, "any_uncut": True, "sided": False},
                {"matched": True, "contact": False, "any_uncut": True, "sided": True},
                {"matched": False, "contact": True, "any_uncut": False, "sided": True}):
        with pytest.raises(ValueError):
            FirstCutoff(TieId.of(A, B), 3, 2, 1.0, 2, **bad)


def test_m11c48_t185_each_exits_left_out_runs_are_reported():
    """A count says how many cutoffs an exit left out, not which way the drop moved an arm: on the Phase B family the
    unsided drop leaves out runs shorter than the matched ones, which raises the treatment arm, and the unmatched drop
    leaves out longer ones (learning-qa 2026-11-10, P2/P9). So the mean run of each exit's left-out cutoffs is
    reported beside the arms. Here one cutoff is unmatched at a run of 5 and one unsided at a run of 1, and an exit
    that left nothing out reads 0.0.
    Mutation: reporting the matched cutoffs' mean, or pooling the two exits into one figure, reads 3.0 for both."""
    from src.bowen.readouts.negative_runs import left_out_mean_run

    firsts = [FirstCutoff(TieId.of(A, B), 3, 5, 0.0, 0, matched=False, contact=True, any_uncut=False, sided=False),
              FirstCutoff(TieId.of(C, D), 4, 1, 0.0, 0, matched=False, contact=True, any_uncut=True, sided=False),
              FirstCutoff(TieId.of(E, F), 5, 9, 2.0, 3)]
    assert left_out_mean_run(firsts, "unmatched") == 5.0
    assert left_out_mean_run(firsts, "unsided") == 1.0
    assert left_out_mean_run(firsts[2:], "unsided") == 0.0 == left_out_mean_run(firsts[2:], "unmatched")


def test_m11c48_t179_a_cut_tie_with_no_contact_is_left_out_of_both_arms_and_counted():
    """T-179 (the owner's option (a), 2026-11-02; learnings P7): the baseline reads an uncut tie only if it had a
    contact by the cutoff's week, so a first cutoff on a tie with no contact by its week is left out of both arms and
    counted, not read at a run of 0. a~b is cut at week 2 with no contact; c~d takes four conflicts in weeks 3–6 and is
    cut at week 6; e~f is uncut, with a conflict in weeks 1 and 4. Read at 0, a~b's cutoff would make the arms
    (2.0, 1.5)."""
    from src.bowen.readouts.negative_runs import no_contact, unmatched

    records = ([act("CUTOFF", A, B, 2)] + [act("CONFLICT", C, D, w) for w in range(3, 7)] +
               [act("CUTOFF", C, D, 6), act("CONFLICT", E, F, 1), act("CONFLICT", E, F, 4)])
    firsts = first_cutoff_runs(records, KINDS)
    assert [(f.tie, f.contact, f.matched) for f in firsts] == [(AB, False, False), (CD, True, True)]
    assert arm_values(firsts) == (4.0, 2.0)  # c~d's run of 4 against e~f's 2 at week 6
    assert no_contact(firsts) == 1 and unmatched(firsts) == 0


def test_m11c48_t179_each_first_cutoff_is_counted_once():
    """T-179: every first cutoff is matched, unmatched or no-contact, and only one of them. a~b is cut at week 1 with no
    contact and no tie in contact either: it counts as no-contact, not also as unmatched. c~d has a contact and no
    uncut tie in contact by week 2: unmatched. e~f, two conflicts, is cut at week 5 against the uncut tie a~d, in
    contact from week 3: matched."""
    from src.bowen.readouts.negative_runs import no_contact, unmatched, unsided

    AD = TieId.of(A, D)
    records = [act("CUTOFF", A, B, 1), act("CONFLICT", C, D, 2), act("CUTOFF", C, D, 2), act("CONFLICT", A, D, 3),
               act("CONFLICT", E, F, 4), act("CONFLICT", E, F, 5), act("CUTOFF", E, F, 5)]
    firsts = first_cutoff_runs(records, KINDS)
    assert [(f.tie, f.contact, f.matched) for f in firsts] == [(AB, False, False), (CD, True, False), (EF, True, True)]
    assert (no_contact(firsts), unmatched(firsts), sum(f.matched for f in firsts)) == (1, 1, 1)
    assert unsided(firsts) == 0  # T-185: a fourth way out, counted apart from the other two
    assert (no_contact(firsts) + unmatched(firsts) + unsided(firsts)
            + sum(f.matched for f in firsts) == len(firsts))
    assert arm_values(firsts) == (2.0, 1.0) and AD not in {f.tie for f in firsts}


def test_m11c48_t185_a_ties_sides_are_the_members_who_acted_on_it_by_a_week():
    """T-185: a tie's sides at a week are how many of its members had sent a move other than a `CUTOFF` across it by
    then, the week included. a~b: A acts in week 1 and B in week 3, so the tie is one-sided in weeks 1 and 2 and
    two-sided from week 3. A `CUTOFF` is not an act on the tie: c~d, where only C's move is a cutoff, stays one-sided."""
    records = [act("CONFLICT", A, B, 1), act("STAY-IN-CONTACT", B, A, 3),
               act("CONFLICT", D, C, 1), act("CUTOFF", C, D, 2)]
    acted = acted_across(records)
    assert [sides(acted, AB, w) for w in (0, 1, 2, 3, 9)] == [0, 1, 1, 2, 2]
    assert [sides(acted, CD, w) for w in (1, 2, 9)] == [1, 1, 1]


def test_m11c48_t185_the_baseline_reads_the_uncut_ties_with_as_many_sides():
    """T-185 (the owner's Q3 (a) of 2026-11-09; learnings P7): the baseline reads only the uncut ties with as many
    sides as the cut tie, since a cut tie is more often one both members act on and such ties hold longer runs. a~b
    is two-sided (both A and B send conflicts) and is cut at week 5 on a run of 4. c~d is two-sided with a standing run
    of 1; e~f is one-sided with a standing run of 3. The baseline is c~d's 1 alone, and the longest run reported beside
    it is c~d's too. Read over both uncut ties it would be 2.0, which is what the readout gave before this change."""
    from src.bowen.readouts.negative_runs import unsided

    records = ([act("CONFLICT", A, B, w) for w in (1, 2, 3)] + [act("CONFLICT", B, A, 4), act("CUTOFF", A, B, 5)] +
               [act("CONFLICT", C, D, 1), act("CONFLICT", D, C, 2), act("STAY-IN-CONTACT", C, D, 3),
                act("CONFLICT", D, C, 4)] +
               [act("CONFLICT", E, F, w) for w in (1, 2, 3)])
    firsts = first_cutoff_runs(records, KINDS)
    assert [(f.tie, f.run, f.uncut, f.longest_uncut, f.matched, f.sided) for f in firsts] == [(AB, 4, 1.0, 2, True, True)]
    assert arm_values(firsts) == (4.0, 1.0) and unsided(firsts) == 0


def test_m11c48_t185_a_cutoff_with_no_uncut_tie_of_its_sides_is_left_out_and_counted():
    """T-185: a first cutoff with an uncut tie in its week but none with as many sides has no matched baseline. It is
    left out of both arms and counted (``unsided``), as T-123's and T-179's cutoffs are, since reading it against ties
    of other sides is the unmatched comparison the change removes. a~b is two-sided and cut at week 5; the only uncut
    tie, e~f, is one-sided. c~d, two-sided and cut at week 9, is matched against e~f only if sides are ignored."""
    from src.bowen.readouts.negative_runs import no_contact, unmatched, unsided

    records = ([act("CONFLICT", A, B, w) for w in (1, 2, 3)] + [act("CONFLICT", B, A, 4), act("CUTOFF", A, B, 5)] +
               [act("CONFLICT", C, D, 6), act("CONFLICT", D, C, 7), act("CUTOFF", C, D, 9)] +
               [act("CONFLICT", E, F, w) for w in (1, 2, 3)])
    firsts = first_cutoff_runs(records, KINDS)
    assert [(f.tie, f.contact, f.any_uncut, f.sided, f.matched) for f in firsts] == [
        (AB, True, True, False, False), (CD, True, True, False, False)]
    assert arm_values(firsts) == (0.0, 0.0)  # no matched first cutoff on the seed: 0 in both arms
    assert (unsided(firsts), unmatched(firsts), no_contact(firsts)) == (2, 0, 0)
    assert unsided(firsts) + unmatched(firsts) + no_contact(firsts) + sum(f.matched for f in firsts) == len(firsts)


def test_m11c48_t179_a_contact_in_the_cutoffs_week_counts_on_both_sides():
    """T-179: one contact condition on both arms. a~b's only contact is a conflict in its cutoff's week, and e~f's only
    contact is in that week too: both are in contact by week 2, so the cutoff is matched, run 1 against 1."""
    from src.bowen.readouts.negative_runs import no_contact

    records = [act("CONFLICT", A, B, 2, 0), act("CUTOFF", A, B, 2, 1), act("CONFLICT", E, F, 2)]
    firsts = first_cutoff_runs(records, KINDS)
    assert [(f.tie, f.contact, f.matched) for f in firsts] == [(AB, True, True)]
    assert arm_values(firsts) == (1.0, 1.0) and no_contact(firsts) == 0
