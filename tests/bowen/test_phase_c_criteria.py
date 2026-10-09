"""The arms of two Phase C criteria, checked in the default suite (decided 2026-10-08).

Purpose: show that `M11.C.42`'s declared absence removes the third member from the pair's choices at that
         one tick and nowhere else, that `M11.C.45`'s readout is a rate per person-week, that `M11.C.29`'s
         held-open tie is not cut while it is held, and that `M11.C.41`'s restated readout divides by what was offered.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.29, #M11.C.41, #M11.C.42, #M11.C.45, #M17.D.3
Tests:   this file

The criteria themselves run over ensembles outside the default suite (`pytest -m ensemble`).
"""

from __future__ import annotations

import pytest

from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import DeliveredRecord, EmittedRecord, SelectionRecord
import src.bowen.ensemble.criteria as criteria
from src.bowen.engine.events import EventId
from src.bowen.ensemble.criteria import (CRITERIA, SETTINGS, act, arm_c29, arm_c42, arm_spell_triad,
                                         reactive_per_offered, scenario, spell)

P = PersonId
PAIR, THIRD = (P("f"), P("m")), P("c")
T0 = 10  # a week with history before it: these tests check the mechanism, not the criterion's declared week (0)
SEEDS = range(12)


def toward_third_at(records, tick):
    """The pair's acts at ``tick`` that name the third member: emitted toward it, or withheld toward it."""
    emitted = [r.event for r in records if isinstance(r, EmittedRecord) and r.event.timestamp == tick
               and r.event.sender in PAIR and THIRD in r.event.targets]
    withheld = [r for r in records if isinstance(r, SelectionRecord) and r.tick == tick and r.actor in PAIR
                and r.withheld_toward == THIRD]
    return emitted + withheld


def run(seed, unavailable=None):
    weeks = T0 + 2
    return scenario("triad", seed, weeks, spells=spell("triad", weeks), unavailable=unavailable)[1]


def test_m11c42_absence_removes_the_third_from_the_pairs_choices_that_tick():
    absent = {T0: (PAIR, THIRD)}
    open_, closed = [], []
    for seed in SEEDS:
        open_ += toward_third_at(run(seed), T0)
        closed += toward_third_at(run(seed, absent), T0)
    assert open_, "no seed acted toward the third member at t0: the check would be vacuous"
    assert closed == []


def tick_of(record) -> int:
    """Every record's tick: emitted records by their event's timestamp, delivered ones by delivery tick."""
    if isinstance(record, EmittedRecord):
        return record.event.timestamp
    if isinstance(record, DeliveredRecord):
        return record.delivery.delivered_tick
    return record.tick


def test_m11c42_absence_changes_nothing_before_its_tick():
    for seed in SEEDS:
        open_, closed = run(seed), run(seed, {T0: (PAIR, THIRD)})
        assert {type(r) for r in open_ if tick_of(r) < T0} >= {EmittedRecord, DeliveredRecord, SelectionRecord}
        assert [r for r in open_ if tick_of(r) < T0] == [r for r in closed if tick_of(r) < T0]


def test_m11c42_absence_forms_no_triangle_holding_the_absent_member():
    """At that tick neither pair member can TRIANGLE at all: every triad they could form holds the third member."""
    for seed in SEEDS:
        at_t0 = [r for r in run(seed, {T0: (PAIR, THIRD)})
                 if isinstance(r, SelectionRecord) and r.tick == T0 and r.actor in PAIR]
        assert at_t0, "no selection recorded for the pair at t0"
        assert not [label for r in at_t0 for label in r.legal_set if label.startswith("TRIANGLE>")]


def test_m11c42_readout_counts_the_whole_pairs_triangles():
    """The readout is the pair's TRIANGLE acts toward the third member after t0, not the sender's alone."""
    settings, weeks, t0 = SETTINGS["M11.C.42"], SETTINGS["M11.C.42"]["weeks"], SETTINGS["M11.C.42"]["t0"]
    by_partner = 0
    for seed in SEEDS:
        reuse = arm_c42("treatment", seed, settings).readouts["triangle_reuse"]
        _, records = scenario("triad", seed, weeks, spells=spell("triad", weeks), forced=act("f", "TRIANGLE", "c", t0),
                              held_open=criteria.held(t0, "f-c"))
        later = [r.event for r in records if isinstance(r, EmittedRecord) and r.event.kind == "TRIANGLE"
                 and r.event.targets == (THIRD,) and r.event.timestamp > t0]
        assert reuse == sum(1 for e in later if e.sender in PAIR)
        by_partner += sum(1 for e in later if e.sender == P("m"))
    assert by_partner, "the partner never triangled the third member: the pair-wide count is untested"


def test_m11c45_triangle_rate_is_per_person_week():
    weeks = SETTINGS["M11.C.45"]["weeks"]
    for arm in ("baseline", "treatment"):
        rate = arm_spell_triad(arm, 3, {"weeks": weeks}).readouts["triangle_rate"]
        spells = spell("triad", weeks) if arm == "treatment" else ()
        _, records = scenario("triad", 3, weeks, spells=spells)
        triangles = sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.mechanism is Mechanism.MOVE
                        and r.event.kind == "TRIANGLE" and r.event.sender in {*PAIR, THIRD})
        assert triangles > 0 or arm == "baseline"
        assert rate * 3 * weeks == pytest.approx(triangles)


# --- Step S: each scripted act's ties held open until it is made, the same in both arms ------------

SCRIPTED = [cid for cid in CRITERIA if cid in ("M11.C.3", "M11.C.4", "M11.C.5", "M11.C.29", "M11.C.35", "M11.C.42")
            or cid.startswith("M11.C.27[")]  # M11.C.32 scripts its act at week 0: there is nothing to hold


def test_s_every_scripted_criterion_holds_the_same_ties_in_both_arms(monkeypatch):
    seen = {}

    def capture(family, seed, weeks, **kwargs):
        seen.setdefault(cid, []).append(kwargs.get("held_open"))
        raise StopIteration

    monkeypatch.setattr(criteria, "scenario", capture)
    for cid in SCRIPTED:
        for arm in CRITERIA[cid].arms:
            with pytest.raises(StopIteration):
                CRITERIA[cid].arm(arm, 0, CRITERIA[cid].settings)
    assert set(seen) == set(SCRIPTED) and len(SCRIPTED) == 10
    for cid, holds in seen.items():
        assert holds[0] is not None and holds[0][0] and holds[0] == holds[1], cid
        assert holds[0][1].start == 0 and holds[0][1].stop >= CRITERIA[cid].settings["t0"], cid


@pytest.mark.parametrize("cid", SCRIPTED)
def test_s_every_scripted_act_is_made_with_the_hold(cid):
    """Without the hold these skipped in 25-60% of seeds (docs/DECISIONS — PHASE C FAILING.md, step S)."""
    criterion = CRITERIA[cid]
    for arm in criterion.arms:
        for seed in range(8):
            criteria.scenario.last_source = None
            criterion.arm(arm, seed, criterion.settings)
            assert criteria.scenario.last_source.skipped == 0, (cid, arm, seed)


# --- M11.C.29: the f-m tie held open through the disguise span ----------------------------------

C29 = SETTINGS["M11.C.29"]
HELD = range(0, C29["t0"] + C29["disguise_span"])


def cutoffs_between_f_and_m(records):
    return [r.event for r in records if isinstance(r, EmittedRecord) and r.event.kind == "CUTOFF"
            and {r.event.sender, *r.event.targets} == {P("f"), P("m")}]


def test_m11c29_held_tie_is_not_cut_while_held():
    held, free = [], []
    for seed in range(20):
        run_ = lambda hold: scenario("triad", seed, HELD.stop, spells=spell("triad", HELD.stop),  # noqa: E731
                                     held_open=hold)[1]
        held += cutoffs_between_f_and_m(run_((((P("f"), P("m")),), HELD)))
        free += cutoffs_between_f_and_m(run_(None))
    assert free, "no seed cut the f-m tie unheld: the check would be vacuous"
    assert held == []


def test_m11c29_hold_lets_every_withdrawal_be_made():
    """With the hold, the disguise's four withdrawals are never illegal; without it, some are (measured: 162 of
    500 seeds skipped one, docs/DECISIONS — PHASE C FAILING.md step S)."""
    settings = CRITERIA["M11.C.29"].settings
    unheld = 0
    original = criteria.scenario
    for seed in range(20):
        arm_c29("baseline", seed, settings)
        assert criteria.scenario.last_source.skipped == 0
        try:
            criteria.scenario = lambda *a, held_open=None, **k: original(*a, **k)  # noqa: E731
            criteria.scenario.last_source = None
            arm_c29("baseline", seed, settings)
            unheld += criteria.scenario.last_source.skipped  # scenario sets it on the module's current name
        finally:
            criteria.scenario = original
    assert unheld, "no seed skipped a withdrawal without the hold: the check would be vacuous"


def test_m11c29_hold_is_the_same_in_both_arms(monkeypatch):
    seen = {}

    def capture(family, seed, weeks, **kwargs):
        seen[kwargs.get("forced") and next(iter(kwargs["forced"].values())).kind] = kwargs["held_open"]
        raise StopIteration

    monkeypatch.setattr(criteria, "scenario", capture)
    for arm in ("baseline", "treatment"):
        with pytest.raises(StopIteration):
            arm_c29(arm, 0, CRITERIA["M11.C.29"].settings)
    assert set(seen) == {"DISTANCE", "I-POSITION"} and seen["DISTANCE"] == seen["I-POSITION"]
    assert seen["DISTANCE"] == (((P("f"), P("m")),), HELD)


# --- M11.C.41: reactive acts selected over reactive acts offered (Q2) ------------------------------


def _selection(tick, actor, event_id, legal):
    from src.bowen.engine.log_records import DecidedBy

    return SelectionRecord(tick, P(actor), DecidedBy.POLICY, event_id, legal_set=legal)


def _emitted(kind, tick, actor):
    from types import SimpleNamespace

    return EmittedRecord(SimpleNamespace(id=EventId(tick, actor, 0), kind=kind))


def test_m11c41_reactive_per_offered_divides_by_what_the_legal_set_offered():
    wide = ("CONFLICT>m", "PURSUE>m", "TRIANGLE>c", "DISTANCE>m", "CUTOFF>m", "I-POSITION>m")
    narrow = ("DISTANCE>m", "CUTOFF>m", "I-POSITION>m")  # a lower level: layers 1 and 2 closed (M4.D.3a)
    # Both runs choose one reactive act out of two weeks; only what was offered differs.
    a = [_selection(0, "f", EventId(0, "f", 0), wide), _emitted("CONFLICT", 0, "f"),
         _selection(1, "f", EventId(1, "f", 0), wide), _emitted("I-POSITION", 1, "f")]
    b = [_selection(0, "f", EventId(0, "f", 0), narrow), _emitted("DISTANCE", 0, "f"),
         _selection(1, "f", EventId(1, "f", 0), narrow), _emitted("I-POSITION", 1, "f")]
    assert reactive_per_offered(a) == pytest.approx(1 / 10)
    assert reactive_per_offered(b) == pytest.approx(1 / 4)
    # The share of all moves, the declared readout, cannot tell them apart; this one rises where fewer were offered.
    assert reactive_per_offered(b) > reactive_per_offered(a)
    withheld = [_selection(0, "f", None, narrow)]  # a WITHHOLD emits nothing: offered, not selected
    assert reactive_per_offered(withheld) == 0.0


def test_m11c29_readout_counts_the_weeks_the_third_persons_symptom_is_active(monkeypatch):
    """The restated readout (X1): weeks from onset to re-arm, not weeks with any accumulation, which is positive
    whenever acute anxiety is above the floor (the old count, saturated in both arms)."""
    original = criteria.scenario
    seen = []

    def wrapped(*args, watch=None, **kwargs):
        def both(state):
            watch(state)
            c = state.people[P("c")]
            seen.append((c.symptom_active[c.channel_prior], c.symptom_load[c.channel_prior]))
        return original(*args, watch=both, **kwargs)

    monkeypatch.setattr(criteria, "scenario", wrapped)
    accumulating = active = 0
    for seed in range(4):
        seen.clear()
        readout = arm_c29("baseline", seed, CRITERIA["M11.C.29"].settings).readouts["third_symptom_weeks"]
        assert len(seen) == C29["weeks"]  # the watcher ran once per week
        assert readout == sum(a for a, _ in seen)
        accumulating += sum(load > 0 for _, load in seen)
        active += readout
    assert active < accumulating, "the symptom was active in every week with any load: the check would be vacuous"
