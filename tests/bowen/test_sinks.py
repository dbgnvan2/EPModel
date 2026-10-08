"""Sink allocation and the budget (Phase C step 10).

Purpose: test that each sink draws the budget in proportion to the acts that express it,
         that the overflow goes to conflict with families of origin, that allocation
         conserves the budget, and that only a completed exchange lowers it.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.1, #M1.D.3, #M6.I.1
Tests:   this file
"""

from __future__ import annotations

from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.objects import Sink
from src.bowen.engine.sinks import OVERFLOW, allocate_sinks, reduce_budget, sink_weights
from src.bowen.engine.state import new_run_state
from src.bowen.io.load import CONFIG_DIR, load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, NADIA, ANA = P("ravi"), P("marta"), P("nadia"), P("ana")
PARAMS = engine_params(load_constants())
KINDS = load_event_kinds()


def fresh():
    family = load_family(CONFIG_DIR / "family_phase_c.md")
    state = new_run_state(dict(family.people), dict(family.ties), family.family, KINDS)
    return initialise_run(state, PARAMS)


def acts(state, kind, sender, target, n):
    for i in range(n):
        state.store.record_event(Event(
            id=EventId(state.tick, str(sender.value), 100 + i + 10 * len(state.store.events())), kind=kind,
            mechanism=Mechanism.MOVE, sender=sender, targets=(target,), intensity=50.0, timestamp=state.tick,
            duration=1, exogenous=False, source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
        ))


def allocated_after(kind, sender, target, weeks=40):
    state = fresh()
    acts(state, kind, sender, target, 3)
    for _ in range(weeks):
        allocate_sinks(state, PARAMS)
    return state


def test_m1d1_marital_conflict_draws_the_budget_to_its_sink():
    state = allocated_after("CONFLICT", RAVI, MARTA)
    sinks = state.family.sink_allocations
    assert sinks[Sink.MARITAL_CONFLICT] > max(sinks[Sink.SPOUSE_DYSFUNCTION], sinks[Sink.CHILD_PROJECTION])


def test_m1d1_each_sink_has_its_own_acts():
    assert sink_weights(allocated_after("OVERFUNCTION", RAVI, MARTA, 0), PARAMS)[Sink.SPOUSE_DYSFUNCTION.value] > 0
    projection = allocated_after("OVERFUNCTION", MARTA, NADIA)
    assert projection.family.sink_allocations[Sink.CHILD_PROJECTION] > 0
    assert projection.family.sink_allocations[Sink.MARITAL_CONFLICT] == 0


def test_m1d3_overflow_goes_to_conflict_with_the_families_of_origin():
    state = allocated_after("CONFLICT", MARTA, ANA)
    assert state.family.overflow > 0 and all(v == 0 for v in state.family.sink_allocations.values())


def test_m6i1_allocation_conserves_the_budget():
    state = fresh()
    acts(state, "CONFLICT", RAVI, MARTA, 2)
    acts(state, "OVERFUNCTION", MARTA, NADIA, 2)
    budget = state.family.undifferentiation_budget
    for _ in range(200):
        allocate_sinks(state, PARAMS)
        total = sum(state.family.sink_allocations.values()) + state.family.overflow
        assert total <= budget + 1e-9 and state.family.undifferentiation_budget == budget
    assert abs(total - budget) < 1e-6  # the whole budget ends up allocated


def test_m6i1_only_a_completed_exchange_reduces_the_budget():
    state = allocated_after("CONFLICT", RAVI, MARTA)
    budget = state.family.undifferentiation_budget
    record = reduce_budget(state, PARAMS, None)
    assert state.family.undifferentiation_budget == budget - PARAMS.exchange_budget_reduction
    assert record.mechanism == "differentiating_exchange"
    assert sum(state.family.sink_allocations.values()) + state.family.overflow <= state.family.undifferentiation_budget + 1e-9


def test_m1d1_no_acts_no_movement():
    state = fresh()
    assert allocate_sinks(state, PARAMS) == [] and sink_weights(state, PARAMS)[OVERFLOW] == 0
