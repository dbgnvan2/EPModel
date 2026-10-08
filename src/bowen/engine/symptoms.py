"""Symptom accumulation and onset — part of tick step 9.

Purpose: integrate each person's time above their chronic floor into symptom load
         in their channel, and emit an endogenous event when it crosses threshold.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.3, #M4.C.3a, #M7.D.1, #M1.A.6, #M1.A.11b, #M1.F.7
Tests:   tests/bowen/test_symptoms.py

Phase C builds `M7.D.1` (spec revision 12, P3). The forms are the project's, [I]:

    load  ← load × (1 − symptom_leak_rate) + max(0, acute − chronic)      (M4.C.3a: time above the floor)
    onset when load ≥ symptom_threshold_gain × functional_level            (M1.A.6: against functional level)
    re-arm when load < symptom_rearm_fraction × threshold

The integrand is **time above the floor, not peak**, and the integrator leaks, so
many resolved excursions and one unresolved excursion are distinguishable: a
sustained excursion builds load the separated ones lose between them.

Load accumulates in the person's constitutional channel (`channel_prior`,
`M1.A.11b`). The family-focus term that can move it (`M1.A.11c`) and the channels'
substitution (`M7.D.2`) are Phase D.

On onset the bearer emits one ``SYMPTOM_ONSET`` event (`M1.F.7`: endogenous,
never drawn from incidence data), addressed to everyone on an interactive tie with
them, at ``symptom_event_intensity``. Recipients appraise it through their tie to the
bearer. *Not yet built:* `M4.D.5d`'s gate of onset on emotional reserve (the policy,
step 6).
"""

from __future__ import annotations

from src.bowen.engine.act import inject
from src.bowen.engine.contact import excess
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import Person, SymptomChannel
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState
from src.bowen.engine.visibility import HouseholdConductanceVisibility

SYMPTOM_KIND = "SYMPTOM_ONSET"


def threshold(person: Person, params: EngineParams) -> float:
    """Purpose: the onset threshold, evaluated against functional level.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.6, #M7.D.1
    Tests:   tests/bowen/test_symptoms.py::test_m1a6_threshold_reads_functional_level
    """
    return params.symptom_threshold_gain * person.functional_level


def integrate(person: Person, params: EngineParams) -> SymptomChannel | None:
    """Purpose: one tick of the leaky chronicity integrator; return the channel if onset fires now.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.3, #M4.C.3a, #M7.D.1
    Tests:   tests/bowen/test_symptoms.py::test_m4c3a_one_sustained_excursion_builds_more_than_several_resolved
    """
    channel = person.channel_prior
    if channel is None or not person.alive:
        return None
    load = person.symptom_load[channel] * (1.0 - params.symptom_leak_rate) + excess(person)
    person.symptom_load[channel] = load
    limit = threshold(person, params)
    if not person.symptom_active[channel] and load >= limit:
        person.symptom_active[channel] = True
        return channel
    if person.symptom_active[channel] and load < params.symptom_rearm_fraction * limit:
        person.symptom_active[channel] = False
    return None


def accumulate_symptoms(
    state: RunState, params: EngineParams, visibility: HouseholdConductanceVisibility
) -> list:
    """Purpose: run the integrator for everyone and emit an endogenous event at each onset.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.D.1, #M1.F.7
    Tests:   tests/bowen/test_symptoms.py::test_m7d1_threshold_crossing_emits_endogenous_event
    """
    records: list = []
    loads = []
    for pid in sorted(state.people):
        person = state.people[pid]
        channel = integrate(person, params)
        if person.channel_prior is not None and person.alive:
            loads.append((pid, person.symptom_load[person.channel_prior]))
        if channel is None:
            continue
        partners = tuple(
            sorted(other for t in state.ties_of(pid) if t.interactive
                   for other in t.id.members() if other != pid and state.people[other].alive)
        )
        event = Event(
            id=EventId(state.tick, f"symptom:{pid}", list(SymptomChannel).index(channel)),
            kind=SYMPTOM_KIND, mechanism=Mechanism.ENDOGENOUS_SYMPTOM, sender=pid, targets=partners,
            intensity=params.symptom_event_intensity, timestamp=state.tick, duration=1, exogenous=False,
            source_position=SourcePosition.NONE, channel=Channel.ENDOGENOUS,
        )
        records += inject(state, event, visibility)
        records.append(EffectRecord(state.tick, "symptom_onset", event.id, sinks=((f"symptom_load:{channel.value}", 1.0),)))
    if loads:
        records.append(EffectRecord(state.tick, "symptom_accumulation", None,
                                    people=tuple((pid, "symptom_load", load) for pid, load in loads)))
    return records
