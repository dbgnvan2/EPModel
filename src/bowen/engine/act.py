"""Act — tick step 8 — and the injection of scheduled inputs at step 2.

Purpose: turn a selection into a full event, fill its witnesses and deliveries
         through visibility, record it, and queue its deliveries.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.E.1, #M4.E.1a, #M1.F.1, #M1.F.4, #M16.A.2, #M16.A.3
Tests:   tests/bowen/test_mechanisms.py::test_m4e1_scripted_move_becomes_full_event
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import DecidedBy, DeliveredRecord, EmittedRecord, SelectionRecord
from src.bowen.engine.state import RunState
from src.bowen.engine.visibility import HouseholdConductanceVisibility


@dataclass(frozen=True)
class Selection:
    """One person's chosen move this tick. In Phase B the script supplies it."""

    actor: PersonId
    kind: str
    targets: tuple[PersonId, ...]
    intensity: float
    route: tuple[PersonId, ...] = ()
    source_position: SourcePosition = SourcePosition.NONE
    index: int = 0
    decided_by: DecidedBy = DecidedBy.SCRIPTED


def act(state: RunState, selection: Selection, visibility: HouseholdConductanceVisibility) -> list:
    """Purpose: make a selection into an event and queue its deliveries.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.E.1, #M4.E.1a
    Tests:   tests/bowen/test_mechanisms.py::test_m4e1_scripted_move_becomes_full_event
    """
    if state.kinds.mechanism_of(selection.kind) is not Mechanism.MOVE:
        raise ValueError(f"{selection.kind} is not a move")
    event = Event(
        id=EventId(state.tick, str(selection.actor), selection.index),
        kind=selection.kind,
        mechanism=Mechanism.MOVE,
        sender=selection.actor,
        targets=tuple(sorted(selection.targets)),
        intensity=selection.intensity,
        timestamp=state.tick,
        duration=1,
        exogenous=False,
        source_position=selection.source_position,
        channel=Channel.SCRIPTED if selection.decided_by is DecidedBy.SCRIPTED else Channel.AUTOMATIC,
        route=selection.route,
        fidelity=visibility.fidelity_for(selection.route),
    )
    resolved = visibility.resolve(event, state.people, state.ties)
    state.store.record_event(resolved.event)
    for delivery in resolved.deliveries:
        state.queue.schedule(delivery)
    return [
        SelectionRecord(tick=state.tick, actor=selection.actor, decided_by=selection.decided_by, event_id=event.id),
        EmittedRecord(resolved.event),
    ]


def inject(state: RunState, event: Event, visibility: HouseholdConductanceVisibility) -> list:
    """Purpose: bring a scheduled exogenous or structural event, or an endogenous one, into the run at its tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.6, #M1.F.7, #M3.C.2
    Tests:   tests/bowen/test_mechanisms.py::test_m4a2_trigger_spikes_standing_load_without_contact

    An exogenous event has no edge and arrives in its own tick (latency 0), so
    it must be injected before that tick's deliveries are released.
    """
    if event.timestamp != state.tick:
        raise ValueError(f"event {event.id} injected at tick {state.tick}")
    resolved = visibility.resolve(event, state.people, state.ties)
    state.store.record_event(resolved.event)
    for delivery in resolved.deliveries:
        state.queue.schedule(delivery)
    return [EmittedRecord(resolved.event)]


def record_deliveries(state: RunState, batch) -> list:
    for delivery in batch:
        state.store.record_delivery(delivery)
    return [DeliveredRecord(d) for d in batch]
