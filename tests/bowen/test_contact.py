"""The two-sided appraisal function of spec revision 11 (`M4.C.1`–`M4.C.1c`, Phase C plan D2).

Purpose: test that anxiety follows deviation from a felt contact optimum on either
         side, that level sets steepness and width, that anxiety moves the optimum
         toward closeness, and that a severed tie's cost builds over time.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1, #M4.C.1a, #M4.C.1b, #M4.C.1c
Tests:   this file

These are unit tests of a set mechanism: each checks a direction the spec states,
on the Phase B family, with the invented constants of ``config/bowen/constants.md``.
None is a finding about families (`M11.5`).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.bowen.engine.appraise_base import appraisal_delta
from src.bowen.engine.contact import (
    ContactNotInitialised, band, deviation, deviation_at, initialise_contact, optimum, relax_contact,
    resting_contact, steepness, too_little, too_much,
)
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.state import new_run_state
from src.bowen.engine.standing_load import tie_term
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, ANA, BRUNO = P("ravi"), P("marta"), P("ana"), P("bruno")
RAVI_MARTA, ANA_BRUNO = TieId.of(RAVI, MARTA), TieId.of(ANA, BRUNO)
PARAMS = engine_params(load_constants())


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    initialise_contact(state.people, state.ties, PARAMS)
    return state


def move(kind, intensity, sender=RAVI, target=MARTA):
    return Event(
        id=EventId(0, str(sender.value), 0), kind=kind, mechanism=Mechanism.MOVE, sender=sender, targets=(target,),
        intensity=intensity, timestamp=0, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
    )


def delta(state, event, recipient=MARTA):
    state.store.record_event(event)
    return appraisal_delta(state, Delivery(1, event.id, recipient, Role.TARGET, emitted_tick=0, latency=1), event, PARAMS)


def test_m4c1_deviation_counts_on_both_sides():
    """Too little contact and too much impingement both raise the deviation (KS03.2)."""
    state = fresh()
    marta, tie = state.people[MARTA], state.ties[RAVI_MARTA]
    opt = optimum(marta, tie, PARAMS)
    at_optimum = deviation_at(marta, tie, PARAMS, opt, 0.0)
    assert at_optimum == 0.0
    assert deviation_at(marta, tie, PARAMS, 0.0, 0.0) > at_optimum  # too little
    assert deviation_at(marta, tie, PARAMS, opt, 1.0) > at_optimum  # too much
    assert too_little(marta, tie, PARAMS) >= 0 and too_much(marta, tie, PARAMS) == 0.0


def test_m4c1_delivered_event_is_appraised_by_its_change_in_deviation():
    """Contact toward someone below their optimum relieves; a push past their band is a load."""
    approach = delta(fresh(), move("STAY-IN-CONTACT", 30.0))
    push = delta(fresh(), move("CONFLICT", 250.0))
    assert approach < 0 < push


def test_m4c1_the_drives_are_given_and_the_same_for_every_person():
    """No per-person drive parameter exists: the optimum is derived from bond energy and anxiety only."""
    state = fresh()
    tie = state.ties[RAVI_MARTA]
    ravi, marta = state.people[RAVI], state.people[MARTA]
    calm_twin = dataclasses.replace(marta, acute_anxiety=ravi.acute_anxiety, chronic_anxiety=ravi.chronic_anxiety)
    assert optimum(ravi, tie, PARAMS) == optimum(calm_twin, tie, PARAMS)


def test_m4c1a_steepness_falls_and_band_widens_with_level():
    marta = fresh().people[MARTA]
    higher = dataclasses.replace(marta, functional_level=marta.functional_level + 20)
    assert steepness(higher, PARAMS) < steepness(marta, PARAMS)
    assert band(higher, PARAMS) > band(marta, PARAMS)


def test_m4c1a_the_same_event_weighs_more_at_lower_level():
    low, high = fresh(), fresh()
    high.people[MARTA].functional_level += 20
    assert delta(low, move("CONFLICT", 250.0)) > delta(high, move("CONFLICT", 250.0)) > 0


def test_m4c1b_anxiety_moves_the_optimum_toward_closeness():
    state = fresh()
    marta, tie = state.people[MARTA], state.ties[RAVI_MARTA]
    anxious = dataclasses.replace(marta, acute_anxiety=marta.chronic_anxiety + 30)
    assert optimum(anxious, tie, PARAMS) > optimum(marta, tie, PARAMS)


def test_m4c1c_initial_contact_is_resting_and_a_cut_tie_starts_empty():
    state = fresh()
    tie = state.ties[RAVI_MARTA]
    assert tie.felt_contact[MARTA] == pytest.approx(resting_contact(state.people[MARTA], tie, PARAMS))
    assert state.ties[ANA_BRUNO].felt_contact == {ANA: 0.0, BRUNO: 0.0}
    assert all(v == 0.0 for t in state.ties.values() for v in t.felt_impingement.values())


def test_m4c1c_standing_load_is_the_too_little_side():
    """One function serves both: the standing term is zero exactly when the too-little side is."""
    state = fresh()
    marta, tie = state.people[MARTA], state.ties[RAVI_MARTA]
    tie.felt_contact[MARTA] = optimum(marta, tie, PARAMS)
    assert tie_term(marta, tie, 0.0, PARAMS) == 0.0
    tie.felt_contact[MARTA] = 0.0
    assert tie_term(marta, tie, 0.0, PARAMS) > 0.0


def test_m4c1c_too_little_builds_over_time_on_a_severed_tie():
    """A cut does not bring its whole cost at once: the gap grows as contact relaxes toward none."""
    state = fresh()
    tie = state.ties[RAVI_MARTA]
    tie.interactive = False
    marta = state.people[MARTA]
    gaps = []
    for _ in range(6):
        gaps.append(too_little(marta, tie, PARAMS))
        relax_contact(state.people, state.ties, PARAMS)
    assert all(b > a for a, b in zip(gaps, gaps[1:]))


def test_m4c1c_impingement_relaxes_toward_none():
    state = fresh()
    tie = state.ties[RAVI_MARTA]
    tie.felt_impingement[MARTA] = 0.9
    relax_contact(state.people, state.ties, PARAMS)
    assert 0.0 < tie.felt_impingement[MARTA] < 0.9


def test_m4c1c_uninitialised_contact_raises():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    with pytest.raises(ContactNotInitialised):
        deviation(state.people[MARTA], state.ties[RAVI_MARTA], PARAMS)
