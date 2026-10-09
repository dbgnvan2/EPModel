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
                                         reactive_over_chance, scenario, spell)

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

def _scripting():
    """Every criterion whose arms script an act, found by running each arm's setup, not from a typed list."""
    found = set()
    original = criteria.scenario

    def capture(family, seed, weeks, **kwargs):
        if kwargs.get("forced"):
            found.add(cid)
        raise StopIteration

    criteria.scenario = capture
    try:
        for cid, criterion in CRITERIA.items():
            for arm in criterion.arms:
                try:
                    criterion.arm(arm, 0, criterion.settings)
                except StopIteration:
                    pass
    finally:
        criteria.scenario = original
    return found


SCRIPTING = _scripting()
SCRIPTED = sorted(cid for cid in SCRIPTING if CRITERIA[cid].settings["t0"] > 0)  # at week 0 there is nothing to hold


def test_s_scripted_criteria_are_found_by_running_them():
    """The probes' list and this file's both come from running the arms (learning-qa, 2026-10-09)."""
    assert SCRIPTING == set(_diag_probes().scripted_criteria())
    assert set(SCRIPTING) - set(SCRIPTED) == {"M11.C.32"} and len(SCRIPTED) == 10


def _diag_probes():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "tools" / "probes.py"
    spec = importlib.util.spec_from_file_location("tools.probes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert set(seen) == set(SCRIPTED)
    for cid, holds in seen.items():
        assert holds[0] is not None and holds[0][0] and holds[0] == holds[1], cid
        assert holds[0][1].start == 0 and holds[0][1].stop >= CRITERIA[cid].settings["t0"], cid


# M11.C.29's actor has full systems perspective and may be inside their own I-POSITION sequence with the same
# person: on a step week they select nothing, and an act toward them is rewritten to STAY-IN-CONTACT (M5.D.9).
OWN_SEQUENCE = {"M11.C.29": {"owed a step", "rewritten"}}


@pytest.mark.parametrize("cid", SCRIPTED)
def test_s_every_scripted_act_is_made_with_the_hold(cid):
    """Made means emitted as scripted (or an I-POSITION sequence begun), not merely placed. Without the hold, acts
    were skipped in 25-60% of seeds (docs/DECISIONS — PHASE C FAILING.md, step S)."""
    criterion = CRITERIA[cid]
    for arm in criterion.arms:
        for seed in range(8):
            moves = criterion.arm(arm, seed, criterion.settings).moves
            scripted = moves["(scripted acts)"]
            missed = {k[len("(scripted act: "):-1]: n for k, n in moves.items()
                      if k.startswith("(scripted act: ") and k != "(scripted act: made)" and n}
            assert set(missed) <= OWN_SEQUENCE.get(cid, set()), (cid, arm, seed, missed)
            assert moves["(scripted act: made)"] + sum(missed.values()) == scripted, (cid, arm, seed)


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
    """With the hold, the disguise's four withdrawals are never illegal; without it, some are (s-scripted-weeks in
    docs/phase_c_diagnostic_record.md: no week is legal in every seed for this arm)."""
    settings = CRITERIA["M11.C.29"].settings
    unheld = 0
    original = criteria.scenario
    for seed in range(20):
        arm_c29("baseline", seed, settings)
        assert criteria.scenario.last_source.skipped == 0  # not legal: the case the hold removes
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


def test_m11c41_reactive_over_chance_is_zero_for_a_chooser_at_random_at_any_set_size():
    """Adversarial (P7): the first restatement rose whenever the legal set shrank. A chooser that picks uniformly
    must score 0 here whether layers 1 and 2 are open or closed; one that prefers reactive acts must score above."""
    import itertools

    wide = ("CONFLICT>m", "PURSUE>m", "TRIANGLE>c", "DISTANCE>m", "CUTOFF>m", "I-POSITION>m")
    narrow = ("DISTANCE>m", "CUTOFF>m", "I-POSITION>m")  # a lower level: layers 1 and 2 closed (M4.D.3a)

    def uniform(legal):  # every outcome chosen once: the expectation of a uniform chooser
        records = []
        for week, label in enumerate(legal):
            records += [_selection(week, "f", EventId(week, "f", 0), legal), _emitted(label.split(">")[0], week, "f")]
        return records

    assert reactive_over_chance(uniform(wide)) == pytest.approx(0.0)
    assert reactive_over_chance(uniform(narrow)) == pytest.approx(0.0)
    always = list(itertools.chain.from_iterable(
        [_selection(w, "f", EventId(w, "f", 0), narrow), _emitted("DISTANCE", w, "f")] for w in range(3)))
    assert reactive_over_chance(always) == pytest.approx(1 - 2 / 3)  # always reactive, where chance is 2 of 3
    withheld = [_selection(0, "f", None, narrow)]  # a WITHHOLD emits nothing: offered, not selected
    assert reactive_over_chance(withheld) == pytest.approx(-2 / 3)
    assert reactive_over_chance([_selection(0, "f", None, ())]) == 0.0  # a fallback with no legal set is left out


def test_s_absence_and_hold_in_the_same_week_both_apply():
    """One redraw carries every constraint: before, the hold's redraw replaced the absence's (correctness review,
    2026-10-09; latent, since no criterion yet sets both for one week)."""
    week = 5
    for seed in range(6):
        _, records = scenario("triad", seed, week + 1, unavailable={week: (PAIR, THIRD)},
                              held_open=(((P("f"), P("m")),), range(0, week + 1)))
        legal = [label for r in records if isinstance(r, SelectionRecord) and r.tick == week and r.actor in PAIR
                 for label in r.legal_set]
        assert legal, "no selection for the pair that week"
        assert not [x for x in legal if x.endswith(">c") or x.startswith("TRIANGLE>")]  # the absence
        assert not [x for x in legal if x in ("CUTOFF>m", "CUTOFF>f")]  # the hold
