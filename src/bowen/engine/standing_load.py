"""The standing load — tick step 1, before any event is delivered.

Purpose: load every person from every tie each tick, whether or not anything
         happened, as the "too little" side of the two-sided appraisal function,
         plus a self-generated term from their own basic level.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1c, #M4.A.1, #M4.A.2, #M4.A.3, #M4.A.4, #M4.A.5, #M6.I.8, #M3.D.2
Tests:   tests/bowen/test_mechanisms.py; tests/bowen/test_contact.py

Spec revision 11 (`M4.C.1c`, owner decision A2.1) made the standing load the "too
little" side of `M4.C.1`'s deviation function, so one function serves both the
standing term and delivered events. The forms are in ``contact.py``; here, [I]:

    tie term  = standing_load_gain × steepness(fl) × too_little(person, tie, spike)
    self term = standing_load_gain × (100 − basic_level) / 100            (M4.A.5)

What used to be separate rules now follows from the function:

* a cut-off or non-interactive tie relaxes toward no contact, so its term is the
  largest a tie of that bond energy can carry (M4.A.4: worry edges need no new
  machinery);
* an interactive tie rests at part of its optimum, so it carries a smaller term —
  `RECONCILIATION` restores that resting contact at once (M4.A.3, M1.B.4);
* a `TRIGGER` adds to the "too little" side on its tie, with no contact (M4.A.2).

The self term is unchanged from Phase B: a function of ``basic_level`` only, with
no parameter of its own (M4.A.5, M10.A.1), not divided by ``functional_level``
because it is intra-person.
"""

from __future__ import annotations

from src.bowen.engine.contact import steepness, too_little
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import SCALE_MAX, Person, Relationship
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState


def trigger_intensity(state: RunState, tie: TieId, tick: int) -> float:
    return sum(t.intensity for t in state.triggers if t.tie == tie and t.first_tick <= tick <= t.last_tick)


def tie_term(person: Person, tie: Relationship, spike: float, params: EngineParams) -> float:
    """Purpose: one tie's standing load on one of its members this tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1c, #M4.A.1, #M4.A.2
    Tests:   tests/bowen/test_mechanisms.py::test_m4a1_every_tie_loads_every_tick_without_events
    """
    return params.standing_load_gain * steepness(person, params) * too_little(person, tie, params, spike)


def self_term(person: Person, params: EngineParams) -> float:
    """Purpose: the self-generated load, derived from basic_level only.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.A.5
    Tests:   tests/bowen/test_mechanisms.py::test_m4a5_self_generated_load_derives_from_basic_level
    """
    return params.standing_load_gain * (SCALE_MAX - person.basic_level) / SCALE_MAX


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
