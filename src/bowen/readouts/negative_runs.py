"""Negative-contact runs on a tie — what a person took before cutting off, read off a run's records.

Purpose: count, per tie, the run of negative contacts (conflict, a rebuff, a refusal) received across it with no
         positive contact between, and compare the run that ends in a tie's first cutoff with the run standing in that
         same week on the same seed's ties that were not cut off by then. The readout `M11.C.48` reads; a readout only,
         never selected: the engine does not import it (`M11.D.22`).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.48, #M1.D.2b
Tests:   tests/bowen/test_negative_runs.py::test_m11c48_run_before_the_first_cutoff_against_uncut_ties,
         tests/bowen/test_negative_runs.py::test_m11c48_a_week_with_both_contacts_keeps_the_run,
         tests/bowen/test_negative_runs.py::test_m11c48_a_declared_cut_off_tie_is_left_out_on_both_sides,
         tests/bowen/test_negative_runs.py::test_m11c48_a_refusal_is_a_negative_contact_counted_once,
         tests/bowen/test_negative_runs.py::test_m11c48_two_first_cutoffs_on_one_seed,
         tests/bowen/test_negative_runs.py::test_m11c48_members_filter_and_no_cutoff_seed

The definitions, `[I]` (`docs/readouts_phase_c.md`, "Negative runs"; the proposal
`docs/PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md` §4, the owner's D12):

* a **negative contact** on a tie is an act in `config/bowen/contact_classes.md`'s `negative` or `rebuff` set emitted
  by one member toward the other, or any move across the tie that was refused (a ``refusal`` record), each counted
  once: a refused negative act is one contact, not two;
* a **positive contact** is an act in the `positive` set delivered (emitted and not refused); it ends the run. A week
  that holds both a negative and a positive contact counts its negatives and keeps the run (the owner's Q6 for
  bridging: the negative applies);
* a **run** is the number of negative contacts since the last positive contact, in either direction. Weeks with no
  contact do not end it;
* a tie's **first cutoff** is the first ``CUTOFF`` emitted across it. Its run is the run at that week, the cutoff
  itself not counted. An ``uncut`` tie over a window is one with no ``CUTOFF`` emitted in it and a contact in it;
  the baseline for a cutoff is the mean of the runs standing on the uncut ties in the cutoff's week — the same point
  in time, read the same way, so neither arm has more weeks to build a run in. The longest run an uncut tie reached
  within the window is reported beside it, not compared: a maximum over ties and weeks against one point is biased
  against the criterion (learnings P7; the revision 36 report's decisions).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import math

from src.bowen.engine.counting import count
from src.bowen.engine.events import EventKinds, Mechanism
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord, EmittedRecord

CUTOFF = "CUTOFF"


@dataclass(frozen=True)
class Contact:
    week: int
    tie: TieId
    negative: bool


@dataclass(frozen=True)
class FirstCutoff:
    tie: TieId
    week: int
    run: int  # negative contacts received across the tie in a row, up to the cutoff's week
    uncut: float  # the mean run standing in the cutoff's week on the ties with no CUTOFF in weeks [0, week]
    longest_uncut: int  # reported only: the longest run those ties reached within the window


def contacts(records, kinds: EventKinds, members: frozenset[PersonId] | None = None) -> list[Contact]:
    """Every contact across a tie between two members, as the definitions above read it, in week order."""
    refused = {r.cause for r in records if isinstance(r, EffectRecord) and r.mechanism == "refusal"}
    negative = kinds.negative_contacts | kinds.rebuffs
    out: list[Contact] = []
    for r in records:
        if not isinstance(r, EmittedRecord) or r.event.mechanism is not Mechanism.MOVE or r.event.sender is None:
            continue
        e = r.event
        if e.kind == CUTOFF:
            continue  # the cutoff is what the run precedes, not part of it
        for target in e.targets:
            if target == e.sender or (members is not None and not {e.sender, target} <= members):
                continue
            tie = TieId.of(e.sender, target)
            if e.kind in negative or e.id in refused:
                out.append(Contact(e.timestamp, tie, True))
            elif e.kind in kinds.positive_contacts:
                out.append(Contact(e.timestamp, tie, False))
    return sorted(out, key=lambda c: (c.week, str(c.tie), not c.negative))


def first_cutoffs(records, members: frozenset[PersonId] | None = None) -> dict[TieId, int]:
    """The week of the first CUTOFF emitted across each tie."""
    weeks: dict[TieId, int] = {}
    for r in records:
        if isinstance(r, EmittedRecord) and r.event.kind == CUTOFF and r.event.sender is not None:
            for target in r.event.targets:
                if members is not None and not {r.event.sender, target} <= members:
                    continue
                tie = TieId.of(r.event.sender, target)
                if tie not in weeks or r.event.timestamp < weeks[tie]:
                    weeks[tie] = r.event.timestamp
    return weeks


def runs_by_week(tie_contacts: list[Contact]) -> list[tuple[int, int]]:
    """A tie's run after each of its weeks with contact, as (week, run). Negatives in a week count before its positive
    ends the run, and a week with both keeps the run."""
    out, run = [], 0
    by_week: dict[int, list[Contact]] = defaultdict(list)
    for c in tie_contacts:
        by_week[c.week].append(c)
    for week in sorted(by_week):
        week_contacts = by_week[week]
        negatives = count(c.negative for c in week_contacts)
        if negatives:
            run += negatives
        elif any(not c.negative for c in week_contacts):
            run = 0
        out.append((week, run))
    return out


def run_at(runs: list[tuple[int, int]], week: int) -> int:
    """The run standing at the start of ``week``: after the last contact week before it."""
    standing = 0
    for w, run in runs:
        if w >= week:
            break
        standing = run
    return standing


def longest_run_within(runs: list[tuple[int, int]], last_week: int) -> int:
    """Reported beside the baseline, not the baseline (the module docstring says why)."""
    return max((run for w, run in runs if w <= last_week), default=0)


def first_cutoff_runs(records, kinds: EventKinds, members: frozenset[PersonId] | None = None,
                      declared_cut: frozenset[TieId] = frozenset()) -> list[FirstCutoff]:
    """Purpose: each tie's first cutoff with the negative run before it, the mean run standing in that week on the
    seed's ties not cut off by then (the baseline), and the longest run those ties reached in the window (reported).
    A tie the family declares cut off (``declared_cut``) has no first cutoff in the run and is not an uncut tie
    either: it is left out on both sides. A tie with no contact in the window is not an uncut tie in it.
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.48
    Tests:   tests/bowen/test_negative_runs.py::test_m11c48_run_before_the_first_cutoff_against_uncut_ties,
             tests/bowen/test_negative_runs.py::test_m11c48_a_declared_cut_off_tie_is_left_out_on_both_sides
    """
    per_tie: dict[TieId, list[Contact]] = defaultdict(list)
    for c in contacts(records, kinds, members):
        if c.tie not in declared_cut:
            per_tie[c.tie].append(c)
    runs = {tie: runs_by_week(cs) for tie, cs in per_tie.items()}
    cutoffs = {t: w for t, w in first_cutoffs(records, members).items() if t not in declared_cut}
    out = []
    for tie, week in sorted(cutoffs.items(), key=lambda kv: (kv[1], str(kv[0]))):
        own = run_at(runs.get(tie, []), week + 1)  # contacts in the cutoff's week count (same-week batch, M1.F.8)
        uncut = [t for t in sorted(runs, key=str) if t != tie and (t not in cutoffs or cutoffs[t] > week)
                 and any(w <= week for w, _ in runs[t])]
        standing = [run_at(runs[t], week + 1) for t in uncut]
        mean = math.fsum(standing) / len(standing) if standing else 0.0
        longest = max((longest_run_within(runs[t], week) for t in uncut), default=0)
        out.append(FirstCutoff(tie, week, own, mean, longest))
    return out


def arm_values(firsts: list[FirstCutoff]) -> tuple[float, float]:
    """`M11.C.48`'s two arms for one seed: the mean run before its first cutoffs (treatment) and the mean of the matched
    uncut ties' standing runs in those weeks (baseline). A seed with no cutoff gives 0 in both arms, and is counted."""
    if not firsts:
        return 0.0, 0.0
    n = len(firsts)
    return math.fsum(f.run for f in firsts) / n, math.fsum(f.uncut for f in firsts) / n
