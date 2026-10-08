"""Investment and the reactive detectors (Phase C step 2).

Purpose: test that investment is directed and valence-blind, and that every person
         carries three drifting detectors whose third is two-sided.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.8, #M1.A.19, #M1.A.18a
Tests:   this file
"""

from __future__ import annotations

from src.bowen.engine.contact import initialise_contact
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.objects import REACTIVE_DETECTORS
from src.bowen.engine.reactive import investment_share, update_attention_state, update_investment, update_reactive_state
from src.bowen.engine.state import new_run_state
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, NADIA = P("ravi"), P("marta"), P("nadia")
RAVI_MARTA, RAVI_NADIA = TieId.of(RAVI, MARTA), TieId.of(RAVI, NADIA)
PARAMS = engine_params(load_constants())


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    initialise_contact(state.people, state.ties, PARAMS)
    return state


def emitted(state, sender, target, valence, index):
    state.store.record_event(Event(
        id=EventId(state.tick, str(sender.value), index), kind="CONFLICT", mechanism=Mechanism.MOVE, sender=sender,
        targets=(target,), intensity=10.0, timestamp=state.tick, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED, valence=valence,
    ))


def test_m1b8_conflict_registers_as_high_investment():
    """What a person appraised on a tie raises their investment in it, whatever its sign."""
    state = fresh()
    update_investment(state, {(RAVI_MARTA, RAVI): 30.0}, PARAMS)  # a conflict-laden week with Marta
    assert investment_share(state, RAVI, RAVI_MARTA) > investment_share(state, RAVI, RAVI_NADIA)
    relief = fresh()
    update_investment(relief, {(RAVI_MARTA, RAVI): abs(-30.0)}, PARAMS)  # the caller passes |Δ|: valence-blind
    assert investment_share(relief, RAVI, RAVI_MARTA) == investment_share(state, RAVI, RAVI_MARTA)


def test_m1b8_investment_is_directed():
    state = fresh()
    update_investment(state, {(RAVI_MARTA, RAVI): 30.0}, PARAMS)
    tie = state.ties[RAVI_MARTA]
    assert tie.investment[RAVI] > tie.investment[MARTA]


def test_m1a19_every_person_carries_three_drifting_detectors():
    state = fresh()
    assert all(set(p.reactive_state) == set(REACTIVE_DETECTORS) for p in state.people.values())
    state.ties[RAVI_MARTA].felt_impingement[MARTA] = 1.0
    update_attention_state(state, {}, PARAMS)
    first = state.people[MARTA].reactive_state["critical_urge"]
    update_attention_state(state, {}, PARAMS)
    assert 0 < first < state.people[MARTA].reactive_state["critical_urge"]  # it drifts, it does not jump


def test_m1a19_evaluating_detector_is_two_sided():
    """Praise registers as much as blame (K07.1)."""
    blame, praise, neither = fresh(), fresh(), fresh()
    emitted(blame, MARTA, RAVI, -1, 0)
    emitted(praise, MARTA, RAVI, +1, 0)
    emitted(neither, MARTA, RAVI, 0, 0)
    for state in (blame, praise, neither):
        update_reactive_state(state, PARAMS)
    value = {name: s.people[MARTA].reactive_state["evaluating"] for name, s in
             (("blame", blame), ("praise", praise), ("neither", neither))}
    assert value["blame"] == value["praise"] > value["neither"] == 0.0
