"""Repair of a distant or conflicted tie, and conflict's binding (version 2 group G1, spec revision 36).
Purpose: test `M1.D.2c` as directions on the Phase B family: a run of conflicts hardens a tie into open conflict as a
         run of withdrawals hardens it distant; the stock at entry is the reference level; a positive contact removes
         a third of it and a negative adds a third back; at zero the tie is ordinary; conflict binds a smaller share
         than distance. Unit tests of set mechanisms with invented constants; none is a finding (`M11.5`).
Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2c, #M4.G.1, #M6.4
Tests:   this file
"""

from __future__ import annotations

import pytest

from src.bowen.engine.act import Selection, act
from src.bowen.engine.consolidate import consolidate
from src.bowen.engine.cutoff_stock import bridge_ties
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.moves import apply_move_effects
from src.bowen.engine.objects import TieState
from src.bowen.engine.state import new_run_state
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA = P("ravi"), P("marta")
MARITAL = TieId.of(RAVI, MARTA)
PARAMS = engine_params(load_constants())
VIS = HouseholdConductanceVisibility(PARAMS.per_hop_fidelity)


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, PARAMS)


def move(kind, sender, target, intensity=100.0, tick=0, index=0):
    return Event(
        id=EventId(tick, str(sender.value), index), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
        targets=(target,), intensity=intensity, timestamp=tick, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
    )


def batch_of(state, *events):
    batch = []
    for event in events:
        state.store.record_event(event)
        batch.append(Delivery(event.timestamp + 1, event.id, event.targets[0], Role.TARGET,
                              emitted_tick=event.timestamp, latency=1))
    return tuple(batch)


def positive(state, tick=0):
    return bridge_ties(state, batch_of(state, move("STAY-IN-CONTACT", RAVI, MARTA, tick=tick)), (), PARAMS)


def negative(state, tick=0):
    return bridge_ties(state, batch_of(state, move("CONFLICT", RAVI, MARTA, tick=tick)), (), PARAMS)


def hardened(state, kind, run=None):
    for tick in range(run or PARAMS.hardening_run_length):
        state.tick = tick
        act(state, Selection(MARTA if tick % 2 else RAVI, kind, (RAVI if tick % 2 else MARTA,), 2.0), VIS)
    return consolidate(state, PARAMS)


# --- entry (M4.G.1, M1.D.2c) -----------------------------------------------------------------------


@pytest.mark.parametrize("kind, state_after", [("DISTANCE", TieState.DISTANT), ("CONFLICT", TieState.OPEN_CONFLICT)])
def test_m1d2c_a_run_hardens_the_tie_and_records_the_reference_level(kind, state_after):
    state = fresh()
    tie = state.ties[MARITAL]
    tie.distance_bound_anxiety = 30.0
    records = hardened(state, kind)
    assert tie.tie_state is state_after and tie.reference_level == 30.0
    assert (MARITAL, f"tie_state_{state_after.value}", 1.0) in [row for r in records for row in r.ties]


def test_m1d2c_a_mixed_run_hardens_nothing():
    state = fresh()
    for tick, kind in ((0, "CONFLICT"), (1, "DISTANCE"), (2, "CONFLICT")):
        state.tick = tick
        act(state, Selection(MARTA, kind, (RAVI,), 2.0), VIS)
    consolidate(state, PARAMS)
    assert state.ties[MARITAL].tie_state is TieState.ORDINARY and state.ties[MARITAL].reference_level == 0.0


# --- repair by thirds (A8, A9) ---------------------------------------------------------------------


@pytest.mark.parametrize("entry", [TieState.DISTANT, TieState.OPEN_CONFLICT])
def test_m1d2c_three_positive_contacts_repair(entry):
    """Each positive contact removes a third of the reference level; after the third the tie is ordinary again."""
    state = fresh()
    tie = state.ties[MARITAL]
    tie.tie_state, tie.reference_level, tie.distance_bound_anxiety = entry, 30.0, 30.0
    steps = PARAMS.tie_repair_steps
    for n in range(1, steps):
        records = positive(state, tick=n)
        assert [r.mechanism for r in records] == ["tie_repair_dissipation"]
        assert tie.distance_bound_anxiety == pytest.approx(30.0 * (steps - n) / steps)
        assert tie.tie_state is entry
    records = positive(state, tick=steps)
    assert tie.distance_bound_anxiety == 0.0 and tie.tie_state is TieState.ORDINARY and tie.reference_level == 0.0
    assert (MARITAL, "tie_state_ordinary", 1.0) in records[-1].ties


def test_m1d2c_a_negative_contact_adds_a_third_back_never_above_the_reference():
    state = fresh()
    tie = state.ties[MARITAL]
    tie.tie_state, tie.reference_level, tie.distance_bound_anxiety = TieState.DISTANT, 30.0, 30.0
    assert negative(state, tick=0) == [] and tie.distance_bound_anxiety == 30.0  # already at the reference
    positive(state, tick=1)
    positive(state, tick=2)
    assert tie.distance_bound_anxiety == pytest.approx(10.0)
    records = negative(state, tick=3)
    assert [r.mechanism for r in records] == ["tie_repair_reversed"]
    assert tie.distance_bound_anxiety == pytest.approx(20.0) and tie.tie_state is TieState.DISTANT
    negative(state, tick=4)
    assert tie.distance_bound_anxiety == pytest.approx(30.0)
    negative(state, tick=5)
    assert tie.distance_bound_anxiety == pytest.approx(30.0)


def test_m1d2c_the_negative_wins_a_week_with_both():
    state = fresh()
    tie = state.ties[MARITAL]
    tie.tie_state, tie.reference_level, tie.distance_bound_anxiety = TieState.DISTANT, 30.0, 10.0
    batch = batch_of(state, move("CONFLICT", RAVI, MARTA, index=0), move("STAY-IN-CONTACT", RAVI, MARTA, index=1))
    bridge_ties(state, batch, (), PARAMS)
    assert tie.distance_bound_anxiety == pytest.approx(20.0)


def test_m1d2c_binding_past_the_reference_raises_it():
    """A withdrawal into a distant tie that holds more than its reference level raises the level to the stock: the
    tie now holds that much to repair."""
    state = fresh()
    tie = state.ties[MARITAL]
    tie.tie_state, tie.reference_level, tie.distance_bound_anxiety = TieState.DISTANT, 30.0, 30.0
    ravi = state.people[RAVI]
    ravi.acute_anxiety = ravi.chronic_anxiety + 100.0
    apply_move_effects(state, batch_of(state, move("DISTANCE", RAVI, MARTA)), PARAMS)
    assert tie.distance_bound_anxiety > 30.0 and tie.reference_level == tie.distance_bound_anxiety


def test_m1d2c_an_ordinary_tie_sheds_a_third_of_its_stock_on_a_positive_contact():
    state = fresh()
    tie = state.ties[MARITAL]
    tie.distance_bound_anxiety = 30.0
    positive(state)
    assert tie.distance_bound_anxiety == pytest.approx(20.0) and tie.tie_state is TieState.ORDINARY
    assert negative(state, tick=1) == [] and tie.distance_bound_anxiety == pytest.approx(20.0)


# --- conflict binds (A15) ----------------------------------------------------------------------------


def test_m1d2c_conflict_binds_a_smaller_share_than_distance():
    assert 0.0 < PARAMS.conflict_binding_rate < PARAMS.distance_binding_rate
    bound = {}
    for kind in ("DISTANCE", "CONFLICT"):
        state = fresh()
        ravi = state.people[RAVI]
        ravi.acute_anxiety = ravi.chronic_anxiety + 20.0
        before = ravi.acute_anxiety
        records = apply_move_effects(state, batch_of(state, move(kind, RAVI, MARTA)), PARAMS)
        bound[kind] = state.ties[MARITAL].distance_bound_anxiety
        assert ravi.acute_anxiety == pytest.approx(before - bound[kind])  # moved, not destroyed (M1.D.2a)
        assert [r.mechanism for r in records] == [f"{kind.lower()}_binding"]
    assert 0.0 < bound["CONFLICT"] < bound["DISTANCE"]
