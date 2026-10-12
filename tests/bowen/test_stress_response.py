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
LOADED = PARAMS.anxiety_band_low + 10.0


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, PARAMS)


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
