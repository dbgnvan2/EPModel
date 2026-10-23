"""`M11.C.48`'s readout on hand-built records (version 2 group G1, spec revision 36).
Purpose: the run arithmetic of `src/bowen/readouts/negative_runs.py` on records written here, so each definition is
         checked on its own: the run before a tie's first cutoff, the matched-week baseline on the uncut ties, a week
         holding both contacts, refusals, a declared cut-off tie, and a tie with no contact in the window. Unit tests
         of a readout with invented definitions; none is a finding (`M11.5`).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.48
Tests:   this file
"""
from __future__ import annotations

from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord, EmittedRecord
from src.bowen.io.load import load_event_kinds
from src.bowen.readouts.negative_runs import arm_values, contacts, first_cutoff_runs, first_cutoffs, runs_by_week

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
    late = first_cutoff_runs(records + [act("CONFLICT", E, F, 9)], KINDS, declared_cut=frozenset({EF, CD}))
    assert [(f.uncut, f.longest_uncut) for f in late] == [(0.0, 0)]  # no uncut tie has a contact in the window


def test_m11c48_members_filter_and_no_cutoff_seed():
    """Contacts with a person outside ``members`` are not read; a seed with no first cutoff reads 0 in both arms."""
    records = [act("CONFLICT", A, B, 1), act("CONFLICT", A, E, 1), act("CUTOFF", A, B, 2)]
    assert {c.tie for c in contacts(records, KINDS, frozenset({A, B}))} == {AB}
    assert first_cutoffs(records, frozenset({A, E})) == {}
    assert arm_values(first_cutoff_runs([act("CONFLICT", A, B, 1)], KINDS)) == (0.0, 0.0)
