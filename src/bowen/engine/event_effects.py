"""Events that change structure rather than anxiety.

Purpose: apply TRIGGER, RECONCILIATION, INSTITUTIONALIZE and binder-unavailable
         events when they fall due, and a delivered CUTOFF to its tie.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.A.2, #M4.A.3, #M4.A.4, #M1.F.9, #M1.B.4, #M6.I.7
Tests:   tests/bowen/test_mechanisms.py

Tie events (TRIGGER, RECONCILIATION) and INSTITUTIONALIZE are applied at step 2
of the tick they are scheduled for. They have no recipients — a TRIGGER is "no
contact at all" (M4.A.2) — so they never pass through delivery. A CUTOFF is a
move: it is delivered like any other and, on delivery, makes its tie
non-interactive. No tie is ever removed (M6.I.7: there is no exit from the field).

Revision 11 (Phase C step 1, plan D2): a delivered CUTOFF also removes felt
impingement on the tie at once, for both members, and that change is appraised —
the relief `M11.C.4` needs now. Felt contact is not touched: it relaxes toward
zero over the following ticks, which is the cost that comes later (`M4.C.1c`).
RECONCILIATION restores resting contact at once (`M1.B.4`: zero re-activation latency),
and returns any anxiety `DISTANCE` bound into the tie to its members, split equally
(`M1.D.2a` (c); Phase C step 5).
"""

from __future__ import annotations

from src.bowen.engine.contact import deviation, resting_contact, steepness
from src.bowen.engine.events import BinderKind, Delivery, Event, Mechanism, Role
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.moves import release_bound_distance
from src.bowen.engine.objects import SymptomChannel, TieState
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import ActiveTrigger, RunState

# M5.A.1 fixes the move repertoire, so this name is the spec's, not editorial content.
CUTOFF_KIND = "CUTOFF"

STRUCTURAL = frozenset(
    {Mechanism.TRIGGER, Mechanism.RECONCILIATION, Mechanism.INSTITUTIONALIZE, Mechanism.BINDER_UNAVAILABLE}
)


def apply_structural_event(state: RunState, event: Event, params: EngineParams) -> list[EffectRecord]:
    """Purpose: apply one structural event at its tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.A.2, #M4.A.3, #M4.A.4, #M1.F.9
    Tests:   tests/bowen/test_mechanisms.py::test_m4a3_reconciliation_converts_standing_to_interaction_load
    """
    if event.mechanism is Mechanism.TRIGGER:
        # Applied at step 2 of tick t; the standing load ran at step 1, so the
        # spike lands from step 1 of t + 1, for `duration` ticks (M3.D.2, M4.A.2).
        state.triggers.append(
            ActiveTrigger(event.on_tie, event.intensity, event.timestamp + 1, event.timestamp + event.duration)
        )
        return [EffectRecord(state.tick, "trigger", event.id, ties=((event.on_tie, "standing_spike", event.intensity),))]
    if event.mechanism is Mechanism.RECONCILIATION:
        tie = state.ties[event.on_tie]
        # M1.B.4: reunion restores full coupling with zero re-activation latency — bond energy is untouched.
        tie.interactive = True
        tie.tie_state = TieState.ORDINARY
        for member in tie.id.members():
            tie.felt_contact[member] = resting_contact(state.people[member], tie, params)
        # M1.D.2a (c), M6.4: distancing prevented, the anxiety it bound returns to the tie's members.
        returned = release_bound_distance(
            state, tie, tuple(m for m in tie.id.members() if state.people[m].alive)
        )
        ties = [(tie.id, "interactive", 1.0)]
        if returned:
            ties.append((tie.id, "distance_bound_anxiety", -sum(v for _, v in returned)))
        return [EffectRecord(state.tick, "reconciliation", event.id, acute_anxiety=returned, ties=tuple(ties))]
    if event.mechanism is Mechanism.INSTITUTIONALIZE:
        changed = []
        for target in event.targets:
            for tie in state.ties_of(target):
                # M4.A.4: worry edges — non-interactive, bond energy retained, no new machinery.
                tie.interactive = False
                changed.append((tie.id, "interactive", 0.0))
        return [EffectRecord(state.tick, "institutionalize", event.id, ties=tuple(sorted(changed)))]
    if event.mechanism is Mechanism.BINDER_UNAVAILABLE:
        return [_release_binder(state, event)]
    raise ValueError(f"{event.mechanism} is not a structural event")


def _release_binder(state: RunState, event: Event) -> EffectRecord:
    """M1.F.9: the binder's held anxiety returns to the family budget; none is discarded."""
    binder = event.binder
    if binder.kind is BinderKind.TIE_DISTANCE:
        a, b = binder.target.split("~")
        tie = state.ties[TieId.of(PersonId(a), PersonId(b))]
        held, tie.distance_bound_anxiety = tie.distance_bound_anxiety, 0.0
        detail = ((tie.id, "distance_bound_anxiety", -held),)
        triangles = ()
    elif binder.kind is BinderKind.TRIANGLE_POSITION:
        tri_id = _triangle_id(binder.target)
        triangle = state.triangles[tri_id]
        held, triangle.bound_anxiety = triangle.bound_anxiety, 0.0
        detail = ()
        triangles = ((tri_id, "bound_anxiety", f"-{held}"),)
    elif binder.kind is BinderKind.SYMPTOM_CHANNEL:
        person_id, channel = binder.target.split(":")
        person = state.people[PersonId(person_id)]
        held = person.symptom_load[SymptomChannel(channel)]
        person.symptom_load[SymptomChannel(channel)] = 0.0
        detail = ()
        triangles = ()
    else:
        raise NotImplementedError(
            "external-agent responsibility (M1.E.3) is built in Phase C; Phase B cannot release it"
        )
    state.family.undifferentiation_budget += held
    return EffectRecord(
        state.tick, "binder_unavailable", event.id, ties=detail, triangles=triangles,
        sinks=(("undifferentiation_budget", held),),
    )


def _triangle_id(text: str) -> TriangleId:
    a, b, c = text.split("/")
    return TriangleId.of(PersonId(a), PersonId(b), PersonId(c))


def apply_delivered_cutoffs(state: RunState, batch: tuple[Delivery, ...], params: EngineParams) -> list[EffectRecord]:
    """Purpose: a delivered CUTOFF makes its tie non-interactive and cut off, and removes impingement at once.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3, #M6.I.7, #M4.C.1c
    Tests:   tests/bowen/test_mechanisms.py::test_m1b3_four_tie_states_are_distinct
    """
    records = []
    for delivery in sorted(batch):
        if delivery.role is not Role.TARGET:
            continue
        event = state.store.event(delivery.event_id)
        if event.kind != CUTOFF_KIND or event.sender is None:
            continue
        tie = state.tie_between(event.sender, delivery.recipient)
        tie.interactive = False
        tie.tie_state = TieState.CUT_OFF
        relief = []
        for member in tie.id.members():
            person = state.people[member]
            before = deviation(person, tie, params)
            tie.felt_impingement[member] = 0.0
            delta = params.appraisal_gain * steepness(person, params) * (deviation(person, tie, params) - before)
            if delta:
                person.acute_anxiety += delta
                relief.append((member, delta))
        records.append(EffectRecord(
            state.tick, "cutoff", event.id, acute_anxiety=tuple(relief), ties=((tie.id, "interactive", 0.0),),
        ))
    return records
