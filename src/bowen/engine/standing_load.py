"""The standing load — tick step 1, before any event is delivered.

Purpose: load every person from every tie each tick, whether or not anything
         happened, plus a self-generated term from their own basic level.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.A.1, #M4.A.2, #M4.A.3, #M4.A.4, #M4.A.5, #M6.I.8, #M3.D.2
Tests:   tests/bowen/test_mechanisms.py

The standing-load function is invented (M10.C.1). The form used here, graded [I]:

    tie term  = standing_load_gain × bond_energy × m / max(functional_level, floor)
    m         = interactive_standing_fraction   for an interactive tie
              = 1                               for a non-interactive tie (cut off, worry edge)
              + the intensity of any TRIGGER active on the tie this tick (M4.A.2)
    self term = standing_load_gain × (100 − basic_level) / 100            (M4.A.5)

Why ``m`` differs: M4.A.3 says RECONCILIATION converts standing load back into
interaction-driven load, so an interactive tie carries part of its load through
events and less of it as standing load. A cut-off tie carries all of it.

The self term is a function of ``basic_level`` only and has no parameter of its
own (M4.A.5, M10.A.1): it borrows the tie term's scale, and it is not divided by
``functional_level`` because it is intra-person.

"Must not swamp M4.A.1" is read, by the owner's decision of 2026-10-06, at the
level of the family and of each nuclear-household member. On the Phase B family
both hold. One peripheral agent there has a single tie and carries a self term
above that tie's load (0.118 against 0.098 a week); the test records which, and
it is not treated as a defect.
"""

from __future__ import annotations

from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.params import EngineParams
from src.bowen.engine.objects import Person, Relationship
from src.bowen.engine.state import RunState


def trigger_intensity(state: RunState, tie: TieId, tick: int) -> float:
    return sum(t.intensity for t in state.triggers if t.tie == tie and t.first_tick <= tick <= t.last_tick)


def tie_term(person: Person, tie: Relationship, spike: float, params: EngineParams) -> float:
    """Purpose: one tie's standing load on one of its members this tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.A.1, #M4.A.2, #M4.A.3
    Tests:   tests/bowen/test_mechanisms.py::test_m4a1_every_tie_loads_every_tick_without_events
    """
    multiplier = (params.interactive_standing_fraction if tie.interactive else 1.0) + spike
    divisor = max(person.functional_level, params.functional_level_floor)
    return params.standing_load_gain * tie.bond_energy * multiplier / divisor


def self_term(person: Person, params: EngineParams) -> float:
    """Purpose: the self-generated load, derived from basic_level only.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.A.5
    Tests:   tests/bowen/test_mechanisms.py::test_m4a5_self_generated_load_derives_from_basic_level
    """
    return params.standing_load_gain * (100.0 - person.basic_level) / 100.0


def apply_standing_load(state: RunState, params: EngineParams) -> tuple[list[EffectRecord], frozenset[TieId]]:
    """Purpose: run tick step 1 on every person and every tie.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.A.1, #M6.I.8
    Tests:   tests/bowen/test_mechanisms.py::test_m4a2_trigger_spikes_standing_load_without_contact

    Returns the effect records and the set of ties that loaded, which the
    invariant check compares with the whole tie set (M6.I.8). All loads are
    computed from the state before any is applied.
    """
    loads: dict[PersonId, float] = {}
    loaded: set[TieId] = set()
    for tie_id in sorted(state.ties):
        tie = state.ties[tie_id]
        spike = trigger_intensity(state, tie_id, state.tick)
        for member in tie.id.members():
            person = state.people[member]
            if not person.alive:
                continue
            loads[member] = loads.get(member, 0.0) + tie_term(person, tie, spike, params)
        loaded.add(tie_id)
    for pid in sorted(state.people):
        person = state.people[pid]
        if person.alive:
            loads[pid] = loads.get(pid, 0.0) + self_term(person, params)
    for pid in sorted(loads):
        state.people[pid].acute_anxiety += loads[pid]
    record = EffectRecord(
        tick=state.tick,
        mechanism="standing_load",
        cause=None,
        acute_anxiety=tuple((pid, loads[pid]) for pid in sorted(loads)),
    )
    return [record], frozenset(loaded)
