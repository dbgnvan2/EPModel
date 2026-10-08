"""Symptom accumulation and onset (Phase C step 2).

Purpose: test the leaky chronicity integrator and the endogenous event at threshold.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.3, #M4.C.3a, #M7.D.1, #M1.A.6, #M1.A.11b, #M1.F.7
Tests:   this file
"""

from __future__ import annotations

import dataclasses

from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.objects import SymptomChannel
from src.bowen.engine.state import new_run_state
from src.bowen.engine.symptoms import accumulate_symptoms, integrate, threshold
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

MARTA, NADIA = PersonId("marta"), PersonId("nadia")
PARAMS = engine_params(load_constants())
VIS = HouseholdConductanceVisibility(PARAMS.per_hop_fidelity)


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, PARAMS)


def run_excursions(person, pattern):
    """Feed ``pattern`` (excess per tick) through the integrator; return the peak load."""
    peak = 0.0
    for excess in pattern:
        person.acute_anxiety = person.chronic_anxiety + excess
        integrate(person, PARAMS)
        peak = max(peak, person.symptom_load[person.channel_prior])
    return peak


def test_m4c3a_one_sustained_excursion_builds_more_than_several_resolved():
    """Same total time above the floor: one unresolved excursion outbuilds four resolved ones."""
    state = fresh()
    sustained = dataclasses.replace(state.people[MARTA], symptom_load={c: 0.0 for c in SymptomChannel})
    resolved = dataclasses.replace(state.people[MARTA], symptom_load={c: 0.0 for c in SymptomChannel})
    one = [8.0] * 12 + [0.0] * 12
    four = ([8.0] * 3 + [0.0] * 6) * 4
    assert sum(one) == sum(four)
    assert run_excursions(sustained, one) > run_excursions(resolved, four)


def test_m1a6_threshold_reads_functional_level():
    marta = fresh().people[MARTA]
    higher = dataclasses.replace(marta, functional_level=marta.functional_level + 10)
    assert threshold(higher, PARAMS) > threshold(marta, PARAMS)
    assert threshold(dataclasses.replace(marta, basic_level=marta.basic_level + 10), PARAMS) == threshold(marta, PARAMS)


def test_m1a11b_load_accumulates_in_the_constitutional_channel():
    nadia = fresh().people[NADIA]
    nadia.acute_anxiety = nadia.chronic_anxiety + 5
    integrate(nadia, PARAMS)
    loaded = {c for c, v in nadia.symptom_load.items() if v > 0}
    assert loaded == {nadia.channel_prior}


def test_m7d1_threshold_crossing_emits_endogenous_event():
    """Once over threshold: one endogenous event to the bearer's interactive ties; re-armed only well below."""
    state = fresh()
    marta = state.people[MARTA]
    marta.acute_anxiety = marta.chronic_anxiety + 2 * threshold(marta, PARAMS)
    records = accumulate_symptoms(state, PARAMS, VIS)
    [event] = [e for e in state.store.events() if e.mechanism is Mechanism.ENDOGENOUS_SYMPTOM]
    assert event.sender == MARTA and not event.exogenous and event.kind == "SYMPTOM_ONSET"
    partners = {o for t in state.ties_of(MARTA) if t.interactive for o in t.id.members() if o != MARTA}
    assert set(event.targets) == partners
    assert any(r.mechanism == "symptom_onset" for r in records if hasattr(r, "mechanism"))
    state.tick += 1
    accumulate_symptoms(state, PARAMS, VIS)  # still above: no second event
    assert len([e for e in state.store.events() if e.mechanism is Mechanism.ENDOGENOUS_SYMPTOM]) == 1
    marta.acute_anxiety = marta.chronic_anxiety
    marta.symptom_load[marta.channel_prior] = 0.0
    state.tick += 1
    accumulate_symptoms(state, PARAMS, VIS)
    assert not marta.symptom_active[marta.channel_prior]
