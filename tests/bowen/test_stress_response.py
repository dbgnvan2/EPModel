"""Stress moves functioning; a stressor lasts as long as declared (spec revision 18).

Purpose: prove that a long stressor weighs for its duration and a short one does not; that coping under load raises
         functioning and soothing lowers it, each scaled by basic level; that a calm week settles the stress-moved
         swing back slowly; and that the soothing acts come from config. Comparisons assert directions (M0.4).
Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.5e, #M1.F.6, #M7.A.1
Tests:   this file
"""

from __future__ import annotations

import pytest

from src.bowen.engine.act import Selection
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.state import new_run_state
from src.bowen.engine.stress_response import register_stressors, respond, stressor_load
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA = P("ravi"), P("marta")
PARAMS = engine_params(load_constants())
LOADED = 10.0  # above the usual level of 0 that fresh() gives everyone (revision 22)


def fresh():
    """The Phase B family, each person with a usual level of 0: a load in these tests is a load above usual
    (revision 22)."""
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    state = initialise_run(state, PARAMS)
    for person in state.people.values():
        person.usual_excess = 0.0
    return state


def stressor(duration, tick=0):
    return Event(id=EventId(tick, "script:0", 0), kind="JOB_LOSS", mechanism=Mechanism.EXOGENOUS_STRESSOR,
                 sender=None, targets=(RAVI,), intensity=50.0, timestamp=tick, duration=duration, exogenous=True,
                 source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS)


def test_m1f6_a_long_stressor_weighs_for_its_duration():
    """No job for 15 weeks weighs for 15 weeks; a one-week stressor only on arrival."""
    long_, short = fresh(), fresh()
    register_stressors(long_, (stressor(15),))
    register_stressors(short, (stressor(1),))
    weighed = {}
    for name, state in (("long", long_), ("short", short)):
        weeks = []
        for tick in range(1, 20):
            state.tick = tick
            weeks.append(sum(v for r in stressor_load(state, PARAMS) for _, v in r.acute_anxiety))
        weighed[name] = weeks
    assert all(w > 0 for w in weighed["long"][:14]) and all(w == 0 for w in weighed["long"][14:])
    assert all(w == 0 for w in weighed["short"])
    assert long_.stressors == []                                      # spent stressors are dropped


def _respond(kind, basic):
    state = fresh()
    state.people[RAVI].basic_level = state.people[RAVI].functional_level = basic
    before = state.people[RAVI].functional_level
    respond(state, (Selection(RAVI, kind, (MARTA,), 100.0),), {RAVI: LOADED}, PARAMS)
    return state.people[RAVI].functional_level - before, state.people[RAVI].stress_swing


def test_m1a5e_coping_raises_functioning_and_soothing_lowers_it():
    coped, swing = _respond("STAY-IN-CONTACT", 40.0)
    soothed, _ = _respond("DISTANCE", 40.0)
    other, _ = _respond("CONFLICT", 40.0)
    assert coped > 0 == other > soothed
    assert swing == coped                                            # kept apart from borrowed pseudo-self
    assert _respond("WITHHOLD", 40.0)[0] > 0                         # holding back is coping


def test_m1a5e_basic_level_scales_both_directions():
    """A higher basic self rises more when it copes and falls less when it soothes."""
    assert _respond("STAY-IN-CONTACT", 60.0)[0] > _respond("STAY-IN-CONTACT", 30.0)[0] > 0
    assert _respond("DISTANCE", 30.0)[0] < _respond("DISTANCE", 60.0)[0] < 0


def test_m1a5e_no_load_no_response_and_calm_settles_back_slowly():
    state = fresh()
    state.people[RAVI].functional_level += 4.0
    state.people[RAVI].stress_swing = 4.0
    level = state.people[RAVI].functional_level
    respond(state, (Selection(RAVI, "STAY-IN-CONTACT", (MARTA,), 100.0),), {RAVI: 0.0}, PARAMS)
    shed = level - state.people[RAVI].functional_level
    assert shed == pytest.approx(PARAMS.response_relaxation_rate * 4.0) and 0 < shed < 4.0  # slow, not a snap back


def test_m1a5e_soothing_acts_come_from_config(tmp_path):
    from src.bowen.io.load import load_soothing
    from src.bowen.scenario.config_parse import ConfigError

    kinds = load_event_kinds()
    assert kinds.soothing == {"DISTANCE", "CUTOFF", "TRIANGLE", "UNDERFUNCTION"}
    bad = tmp_path / "s.md"
    bad.write_text("| kind | why it avoids or soothes |\n|---|---|\n| `DRINKING` | not a move |\n")
    with pytest.raises(ConfigError, match="DRINKING"):
        load_soothing(kinds, bad)


def test_m1a5e_stress_pushes_functioning_down_to_a_floor_below_basic_and_no_further():
    """Revision 19: without the floor, soothing spiralled (11 of 20 four-year runs ran away)."""
    state = fresh()
    person = state.people[RAVI]
    floor = person.basic_level - PARAMS.stress_floor_below_basic
    person.functional_level = floor + 0.01
    for _ in range(50):
        respond(state, (Selection(RAVI, "DISTANCE", (MARTA,), 100.0),), {RAVI: 500.0}, PARAMS)
    assert person.functional_level == pytest.approx(floor)
    person.functional_level = floor - 3.0                            # already below by other means: not lowered more
    respond(state, (Selection(RAVI, "DISTANCE", (MARTA,), 100.0),), {RAVI: 500.0}, PARAMS)
    assert person.functional_level == floor - 3.0


def test_m1a5e_a_fallback_hold_is_not_coping():
    """Review 2026-10-11: a hold forced by having nothing legal is not a choice, so it does not raise functioning."""
    from src.bowen.engine.log_records import DecidedBy

    state = fresh()
    before = state.people[RAVI].functional_level
    fallback = Selection(RAVI, "WITHHOLD", (), 0.0, decided_by=DecidedBy.FALLBACK)
    respond(state, (fallback,), {RAVI: LOADED}, PARAMS)
    assert state.people[RAVI].functional_level == before


def test_m1a5e_a_clamped_swing_still_decays():
    """Review 2026-10-11: at the floor a calm week cannot lower functioning, but the swing still decays, so nothing is
    left over to take back later from an unrelated gain."""
    state = fresh()
    person = state.people[RAVI]
    person.functional_level = person.basic_level - PARAMS.stress_floor_below_basic
    person.stress_swing = 5.0
    respond(state, (), {RAVI: 0.0}, PARAMS)
    assert person.functional_level == person.basic_level - PARAMS.stress_floor_below_basic
    assert person.stress_swing == pytest.approx(5.0 * (1 - PARAMS.response_relaxation_rate))


def test_m1e7_the_systems_adjustment_settles_back_when_calm_and_holds_under_load():
    """Revision 20 (owner decision 2026-10-11): the functioning gained when someone else's held stance worked settles
    back when calm, at the rate the stress-moved swing does; it does not settle while the person is loaded."""
    state = fresh()
    person = state.people[RAVI]
    person.functional_level += 1.0
    person.adjust_swing = 1.0
    level = person.functional_level
    respond(state, (), {RAVI: LOADED}, PARAMS)
    assert person.functional_level == level and person.adjust_swing == 1.0
    respond(state, (), {RAVI: 0.0}, PARAMS)
    assert level - person.functional_level == pytest.approx(PARAMS.response_relaxation_rate)
    assert person.adjust_swing == pytest.approx(1.0 - PARAMS.response_relaxation_rate)
    for _ in range(500):
        respond(state, (), {RAVI: 0.0}, PARAMS)
    assert person.functional_level == pytest.approx(level - 1.0, abs=1e-3)


def test_m1e7_a_calm_week_sheds_both_swings_together_even_of_opposite_sign():
    """Review 2026-10-11: the calm change is the rate times the sum of both swings, and each decays by its own share."""
    state = fresh()
    person = state.people[RAVI]
    person.stress_swing, person.adjust_swing = -3.0, 1.0
    level = person.functional_level
    respond(state, (), {RAVI: 0.0}, PARAMS)
    assert person.functional_level - level == pytest.approx(PARAMS.response_relaxation_rate * 2.0)
    assert person.stress_swing == pytest.approx(-3.0 * (1 - PARAMS.response_relaxation_rate))
    assert person.adjust_swing == pytest.approx(1.0 - PARAMS.response_relaxation_rate)



def test_m1e7_every_change_lands_in_one_ledger_so_calm_returns_functioning_to_its_start():
    """Review 2026-10-12: a fall recorded in two ledgers (the stress swing, and taken off the adjustment too) left
    functioning above its start once calm, and a renewed stance refilled the adjustment, so the stack came back through
    the other ledger. Soothe under load while adjusted, then stay calm: functioning returns to where the ledgers say."""
    state = fresh()
    person = state.people[RAVI]
    start = person.functional_level
    person.functional_level, person.adjust_swing = start + 1.0, 1.0
    respond(state, (Selection(RAVI, "DISTANCE", (MARTA,), 100.0),), {RAVI: LOADED}, PARAMS)
    assert person.functional_level < start + 1.0 and person.adjust_swing == 1.0  # the fall is the stress swing's
    for _ in range(1500):
        respond(state, (), {RAVI: 0.0}, PARAMS)
    assert person.functional_level == pytest.approx(start, abs=1e-6)


def test_m1a5e_calm_is_relative_to_the_persons_usual_level():
    """Revision 22 (owner decision 2026-10-12): the load that drives the response is the excess above the person's own
    usual level. A person whose usual level is 30 is calm at 30 and settles back; at 40 they respond as a person with a
    usual level of 0 does at 10. Before revision 22 a fixed band of 5 left a family running at 30-35 never calm."""
    high, low = fresh(), fresh()
    high.people[RAVI].usual_excess = 30.0
    for state, load in ((high, 40.0), (low, 10.0)):
        state.people[RAVI].basic_level = state.people[RAVI].functional_level = 40.0
        respond(state, (Selection(RAVI, "STAY-IN-CONTACT", (MARTA,), 100.0),), {RAVI: load}, PARAMS)
    assert low.people[RAVI].stress_swing > 0
    assert high.people[RAVI].stress_swing == pytest.approx(low.people[RAVI].stress_swing)
    calm = fresh()
    person = calm.people[RAVI]
    person.usual_excess, person.stress_swing = 30.0, -4.0
    person.functional_level -= 4.0
    level = person.functional_level
    respond(calm, (Selection(RAVI, "DISTANCE", (MARTA,), 100.0),), {RAVI: 30.0}, PARAMS)
    assert person.functional_level > level                            # at usual: calm, so the swing settles back
    assert person.stress_swing == pytest.approx(-4.0 * (1 - PARAMS.response_relaxation_rate))


def test_m1a5e_the_usual_level_follows_slowly():
    """Revision 22: for the first 1 / usual_level_rate weeks the usual level is the plain mean of the weeks so far, so it
    catches up with the family's level; after that it follows at usual_level_rate a week, so a lasting stressor shows
    as above usual for months. The dead keep no usual level."""
    from src.bowen.engine.stress_response import update_usual

    state = fresh()
    person = state.people[RAVI]
    person.usual_excess, person.usual_weeks = None, 0
    loads = [10.0, 30.0, 20.0, 40.0]
    for value in loads:
        update_usual(state, {RAVI: value}, PARAMS)
    assert person.usual_excess == pytest.approx(sum(loads) / len(loads))   # the plain mean, early on
    steady = round(1 / PARAMS.usual_level_rate)
    for _ in range(steady):
        update_usual(state, {RAVI: 25.0}, PARAMS)
    before = person.usual_excess
    update_usual(state, {RAVI: 125.0}, PARAMS)
    assert person.usual_excess == pytest.approx(before + PARAMS.usual_level_rate * (125.0 - before))
    for _ in range(12):                                                # three months of a stressor
        update_usual(state, {RAVI: 125.0}, PARAMS)
    assert person.usual_excess < 75.0                                  # still well below: the stressor still counts
    person.alive = False
    before = person.usual_excess
    update_usual(state, {RAVI: 999.0}, PARAMS)
    assert person.usual_excess == before

def test_m1a5e_soothing_also_scales_with_the_load_above_usual_and_week_0_is_calm():
    """Review 2026-10-12: soothing at usual 30 and load 40 moves functioning as at usual 0 and load 10; and before a
    person has a usual level (week 0) they are calm, so a heavy load neither moves them nor opens practice."""
    from src.bowen.engine.external import self_practice

    high, low = fresh(), fresh()
    high.people[RAVI].usual_excess = 30.0
    for state, load in ((high, 40.0), (low, 10.0)):
        state.people[RAVI].basic_level = state.people[RAVI].functional_level = 40.0
        respond(state, (Selection(RAVI, "DISTANCE", (MARTA,), 100.0),), {RAVI: load}, PARAMS)
    assert low.people[RAVI].stress_swing < 0
    assert high.people[RAVI].stress_swing == pytest.approx(low.people[RAVI].stress_swing)
    new = fresh()
    new.people[RAVI].usual_excess = None
    before = new.people[RAVI].functional_level
    respond(new, (Selection(RAVI, "DISTANCE", (MARTA,), 100.0),), {RAVI: 40.0}, PARAMS)
    self_practice(new, (Selection(RAVI, "STAY-IN-CONTACT", (MARTA,), 100.0),), {RAVI: 40.0}, PARAMS)
    assert new.people[RAVI].functional_level == before and new.practice == []
