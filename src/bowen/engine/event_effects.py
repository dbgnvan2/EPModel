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
"""

from __future__ import annotations

from src.bowen.engine.events import BinderKind, Delivery, Event, Mechanism, Role
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import SymptomChannel, TieState
from src.bowen.engine.state import ActiveTrigger, RunState

# M5.A.1 fixes the move repertoire, so this name is the spec's, not editorial content.
CUTOFF_KIND = "CUTOFF"

STRUCTURAL = frozenset(
    {Mechanism.TRIGGER, Mechanism.RECONCILIATION, Mechanism.INSTITUTIONALIZE, Mechanism.BINDER_UNAVAILABLE}
)


def apply_structural_event(state: RunState, event: Event) -> list[EffectRecord]:
    """Purpose: apply one structural event at its tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.A.2, #M4.A.3, #M4.A.4, #M1.F.9
    Tests:   tests/bowen/test_mechanisms.py::test_m4a3_reconciliation_converts_standing_to_interaction_load
    """
    if event.mechanism is Mechanism.TRIGGER:
        state.triggers.append(
            ActiveTrigger(event.on_tie, event.intensity, event.timestamp, event.timestamp + event.duration - 1)
        )
        return [EffectRecord(state.tick, "trigger", event.id, ties=((event.on_tie, "standing_spike", event.intensity),))]
    if event.mechanism is Mechanism.RECONCILIATION:
        tie = state.ties[event.on_tie]
        # M1.B.4: reunion restores full coupling with zero re-activation latency — bond energy is untouched.
        tie.interactive = True
        tie.tie_state = TieState.ORDINARY
        return [EffectRecord(state.tick, "reconciliation", event.id, ties=((tie.id, "interactive", 1.0),))]
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


def apply_delivered_cutoffs(state: RunState, batch: tuple[Delivery, ...]) -> list[EffectRecord]:
    """Purpose: a delivered CUTOFF makes its tie non-interactive and cut off; bond energy stays.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3, #M6.I.7
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
        records.append(EffectRecord(state.tick, "cutoff", event.id, ties=((tie.id, "interactive", 0.0),)))
    return records
