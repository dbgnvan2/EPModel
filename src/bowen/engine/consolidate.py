"""Consolidation — tick step 9, before the invariants assert.

Purpose: decay acute anxiety toward its chronic floor, decay bond energy (at or
         near zero), and harden ties that repeated moves have made distant.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.8, #M1.A.7a, #M1.B.4, #M4.G.1
Tests:   tests/bowen/test_mechanisms.py

* Acute anxiety sheds ``acute_decay_rate`` of its excess over the chronic floor
  each tick, and never falls below the floor (M1.A.8, M1.A.7a).
* Bond energy decays at ``bond_energy_decay_rate``, which M1.B.4 requires to be
  at or near zero (0 in Phase B).
* M4.G.1: when the last ``hardening_run_length`` moves on a tie, in either
  direction, are all ``DISTANCE``, the tie registers as distant — "three
  withdrawals in a row … not three independent events". A cut-off tie stays cut off.
* Revision 11 (`M4.C.1c`): felt contact relaxes toward its resting value and felt
  impingement toward zero (``contact.relax_contact``), so a severed tie's "too little"
  side builds over the ticks after the cut.
"""

from __future__ import annotations

from src.bowen.engine.contact import relax_contact
from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import TieId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import TieState
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState

# M5.A.1 fixes the repertoire; the name is the spec's.
DISTANCE_KIND = "DISTANCE"


def _recent_moves_on(state: RunState, tie: TieId, count: int) -> list[str]:
    a, b = tie.members()
    kinds = [
        e.kind
        for e in state.store.events()
        if e.mechanism is Mechanism.MOVE
        and e.timestamp <= state.tick
        and e.sender in (a, b)
        # A move counts on this tie when it goes from one member to the other,
        # whoever else it is also addressed to (review, 2026-10-06).
        and ({a, b} - {e.sender}) <= set(e.targets)
    ]
    return kinds[-count:]


def consolidate(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: run tick step 9's state changes.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.8, #M1.B.4, #M4.G.1, #M4.C.1c
    Tests:   tests/bowen/test_mechanisms.py::test_m4g1_three_withdrawals_register_as_distant_tie
    """
    decay = []
    for pid in sorted(state.people):
        person = state.people[pid]
        before = person.acute_anxiety
        excess = max(0.0, before - person.chronic_anxiety)
        person.acute_anxiety = person.chronic_anxiety + excess * (1.0 - params.acute_decay_rate)
        if person.acute_anxiety != before:
            decay.append((pid, person.acute_anxiety - before))

    tie_changes = []
    for tie_id in sorted(state.ties):
        tie = state.ties[tie_id]
        if params.bond_energy_decay_rate:
            lost = tie.bond_energy * params.bond_energy_decay_rate
            tie.bond_energy -= lost
            tie_changes.append((tie_id, "bond_energy", -lost))
        if tie.tie_state in (TieState.CUT_OFF, TieState.DISTANT):
            continue
        recent = _recent_moves_on(state, tie_id, params.hardening_run_length)
        if len(recent) == params.hardening_run_length and all(k == DISTANCE_KIND for k in recent):
            tie.tie_state = TieState.DISTANT
            tie_changes.append((tie_id, "tie_state_distant", 1.0))

    contact_changes = relax_contact(state.people, state.ties, params)

    records = []
    if decay:
        records.append(EffectRecord(state.tick, "acute_decay", None, acute_anxiety=tuple(decay)))
    if tie_changes:
        records.append(EffectRecord(state.tick, "consolidation", None, ties=tuple(tie_changes)))
    if contact_changes:
        rows = tuple((tie, f"{field}:{member}", value) for tie, field, member, value in contact_changes)
        records.append(EffectRecord(state.tick, "contact_relaxation", None, ties=rows))
    return records
