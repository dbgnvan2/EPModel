"""The arms of two Phase C criteria, checked in the default suite (decided 2026-10-08).

Purpose: show that `M11.C.42`'s declared absence removes the third member from the pair's choices at that
         one tick and nowhere else, and that `M11.C.45`'s readout is a rate per person-week.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.42, #M11.C.45, #M17.D.3
Tests:   this file

The criteria themselves run over ensembles outside the default suite (`pytest -m ensemble`).
"""

from __future__ import annotations

import pytest

from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EmittedRecord, SelectionRecord
from src.bowen.ensemble.criteria import SETTINGS, arm_spell_triad, scenario, spell

P = PersonId
PAIR, THIRD = (P("f"), P("m")), P("c")
T0 = SETTINGS["M11.C.42"]["t0"]
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


def test_m11c42_absence_changes_nothing_before_its_tick():
    for seed in SEEDS:
        before = [r for r in run(seed) if getattr(r, "tick", None) is not None and r.tick < T0]
        absent = [r for r in run(seed, {T0: (PAIR, THIRD)}) if getattr(r, "tick", None) is not None and r.tick < T0]
        assert before == absent


def test_m11c45_triangle_rate_is_per_person_week():
    weeks = SETTINGS["M11.C.45"]["weeks"]
    for arm in ("baseline", "treatment"):
        rate = arm_spell_triad(arm, 3, {"weeks": weeks}).readouts["triangle_rate"]
        spells = spell("triad", weeks) if arm == "treatment" else ()
        _, records = scenario("triad", 3, weeks, spells=spells)
        triangles = sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.kind == "TRIANGLE"
                        and r.event.sender in {*PAIR, THIRD})
        assert triangles > 0 or arm == "baseline"
        assert rate * 3 * weeks == pytest.approx(triangles)
