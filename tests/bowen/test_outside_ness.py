"""Outside-ness: the two axes, how they move, act identity, and the counterfeit detector (Phase C step 3).

Purpose: test each step 3 requirement as a direction on the Phase B family.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.9, #M1.A.9a, #M4.C.5, #M5.F.1, #M5.F.2, #M5.F.2b, #M5.F.3
Tests:   this file

Unit tests of set mechanisms with invented constants; none is a finding (`M11.5`).
"""

from __future__ import annotations

import dataclasses

from src.bowen.engine.appraise import appraisal_delta, inward_reading
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.outside_ness import efficacy, update_outside_ness
from src.bowen.engine.state import new_run_state
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.readouts.counterfeit import INWARD, OUTWARD, read_counterfeit
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA = P("ravi"), P("marta")
PARAMS = engine_params(load_constants())


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, PARAMS)


def move(kind, intensity, sender=RAVI, target=MARTA, tick=0, index=0):
    return Event(
        id=EventId(tick, str(sender.value), index), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
        targets=(target,), intensity=intensity, timestamp=tick, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
    )


def received(state, event, recipient=MARTA):
    state.store.record_event(event)
    return Delivery(event.timestamp + 1, event.id, recipient, Role.TARGET, emitted_tick=event.timestamp, latency=1)


def set_axes(state, person, outward, inward):
    state.people[person].outside_ness_outward = outward
    state.people[person].outside_ness_inward = inward


def test_m1a9a_efficacy_needs_both_axes_low():
    """A conjunction of two negatives: either axis failing fails the position."""
    marta = fresh().people[MARTA]
    both_low = dataclasses.replace(marta, outside_ness_outward=0.1, outside_ness_inward=0.1)
    forceful = dataclasses.replace(marta, outside_ness_outward=0.9, outside_ness_inward=0.1)
    compliant = dataclasses.replace(marta, outside_ness_outward=0.1, outside_ness_inward=0.9)
    assert efficacy(both_low) > efficacy(forceful) == efficacy(compliant)
    # One axis failing badly is worse than both moderate: a conjunction, not an average.
    moderate = dataclasses.replace(marta, outside_ness_outward=0.6, outside_ness_inward=0.6)
    assert efficacy(forceful) < efficacy(moderate)


def test_m1a9_axes_start_from_basic_level():
    state = fresh()
    ravi, pia = state.people[RAVI], state.people[P("pia")]
    assert ravi.basic_level < pia.basic_level
    assert ravi.outside_ness_outward > pia.outside_ness_outward and ravi.outside_ness_inward > pia.outside_ness_inward


def test_m1a9_forceful_acts_raise_outward_impingement():
    quiet, pushing = fresh(), fresh()
    pushing.store.record_event(move("CONFLICT", 150.0))
    for state in (quiet, pushing):
        update_outside_ness(state, {}, PARAMS)
    assert pushing.people[RAVI].outside_ness_outward > quiet.people[RAVI].outside_ness_outward


def test_m1a9_accommodating_acts_raise_inward_impingement():
    quiet, giving_way = fresh(), fresh()
    giving_way.store.record_event(move("UNDERFUNCTION", 150.0))
    for state in (quiet, giving_way):
        update_outside_ness(state, {}, PARAMS)
    assert giving_way.people[RAVI].outside_ness_inward > quiet.people[RAVI].outside_ness_inward
    assert giving_way.people[RAVI].outside_ness_outward == quiet.people[RAVI].outside_ness_outward


def test_m4c5_hearing_criticism_raises_inward_impingement():
    """Reading the other's event as critical is itself the evidence (M4.C.5)."""
    state = fresh()
    push = move("CONFLICT", 250.0)
    reading = inward_reading(state, received(state, push), push, PARAMS)
    assert reading > 0
    heard, calm = fresh(), fresh()
    update_outside_ness(heard, {MARTA: reading}, PARAMS)
    update_outside_ness(calm, {}, PARAMS)
    assert heard.people[MARTA].outside_ness_inward > calm.people[MARTA].outside_ness_inward


def test_m5f1_sender_state_scales_what_the_move_delivers():
    """The same move delivers more impingement from a forceful sender — read through the event."""
    calm, forceful = fresh(), fresh()
    set_axes(calm, RAVI, 0.0, 0.0)
    set_axes(forceful, RAVI, 1.0, 0.0)
    push = move("CONFLICT", 150.0)
    assert appraisal_delta(forceful, received(forceful, push), push, PARAMS) > appraisal_delta(
        calm, received(calm, push), push, PARAMS
    )


def test_m5f2_same_move_can_land_with_the_opposite_sign():
    """An approach is relief from a calm sender and a load from a forceful one."""
    calm, forceful = fresh(), fresh()
    set_axes(calm, RAVI, 0.0, 0.0)
    set_axes(forceful, RAVI, 1.0, 0.0)
    approach = move("PURSUE", 60.0)
    assert appraisal_delta(calm, received(calm, approach), approach, PARAMS) < 0 < appraisal_delta(
        forceful, received(forceful, approach), approach, PARAMS
    )


def test_m5f1_a_compliant_senders_contact_lands_hollow():
    full, hollow = fresh(), fresh()
    set_axes(full, RAVI, 0.0, 0.0)
    set_axes(hollow, RAVI, 0.0, 1.0)
    approach = move("STAY-IN-CONTACT", 60.0)
    relief_full = appraisal_delta(full, received(full, approach), approach, PARAMS)
    relief_hollow = appraisal_delta(hollow, received(hollow, approach), approach, PARAMS)
    assert relief_full < relief_hollow < 0  # still relief, less of it


def test_m5f2b_detector_reports_which_axis_failed():
    """Equal counterfeit magnitude, opposite directions, told apart by axis (FE03.1)."""
    state = fresh()
    declarer = dataclasses.replace(state.people[RAVI], outside_ness_outward=0.9, outside_ness_inward=0.1)
    accommodator = dataclasses.replace(state.people[RAVI], outside_ness_outward=0.1, outside_ness_inward=0.9)
    assert read_counterfeit(declarer, PARAMS).failed == {OUTWARD}
    assert read_counterfeit(accommodator, PARAMS).failed == {INWARD}
    clear = dataclasses.replace(state.people[RAVI], outside_ness_outward=0.1, outside_ness_inward=0.1)
    assert not read_counterfeit(clear, PARAMS).counterfeit


def test_m5f3_concession_is_graded_continuously_by_the_receiver():
    """The value of a concession, as the receiver appraises it, varies continuously, not as a binary."""
    values = []
    for inward in (0.0, 0.25, 0.5, 0.75, 1.0):
        state = fresh()
        set_axes(state, RAVI, 0.0, inward)
        concession = move("UNDERFUNCTION", 60.0)
        values.append(appraisal_delta(state, received(state, concession), concession, PARAMS))
    assert len(set(round(v, 12) for v in values)) == len(values)
    assert values == sorted(values)  # monotone in how hollow the concession is
