"""The in-run event store — always present, and causally load-bearing.

Purpose: keep every emitted event and every delivery for the run, queryable
         per person from inside the run.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.B.3, #M16.C.4, #M16.D.1, #M1.F.7
Tests:   tests/bowen/test_events.py::test_m16b3_event_store_answers_per_person_queries

M16.B.3 separates two things called logging. This is the **event store**: it
has no off switch, because `M1.E.7c`'s first and fourth forms read from it
(Phase C), so disabling it would change behaviour. The optional **persistence
sink**, which must be a pure observer, is ``src/bowen/io/sinks.py``.
The delayed, scoped view of `M16.D` is built on these queries in Phase C.
"""

from __future__ import annotations

from src.bowen.engine.events import Delivery, Event, EventId, Role
from src.bowen.engine.identifiers import PersonId


class EventStore:
    """Purpose: append-only record of the run's events and deliveries.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.B.3
    Tests:   tests/bowen/test_events.py::test_m16b3_event_store_answers_per_person_queries
    """

    def __init__(self) -> None:
        self._events: dict[EventId, Event] = {}
        self._deliveries: list[Delivery] = []
        # Indexes and a sorted cache, kept so per-tick queries do not re-sort the whole run.
        # They hold nothing the two primary stores do not; every answer is the same.
        self._sorted: tuple[Event, ...] | None = None
        self._by_sender: dict[PersonId, list[Event]] = {}
        self._by_recipient: dict[PersonId, list[Delivery]] = {}

    def record_event(self, event: Event) -> None:
        if event.id in self._events:
            raise ValueError(f"event {event.id} recorded twice")
        self._events[event.id] = event
        self._sorted = None
        if event.sender is not None:
            self._by_sender.setdefault(event.sender, []).append(event)

    def record_delivery(self, delivery: Delivery) -> None:
        if delivery.event_id not in self._events:
            raise ValueError(f"delivery for unrecorded event {delivery.event_id}")
        self._deliveries.append(delivery)
        self._by_recipient.setdefault(delivery.recipient, []).append(delivery)

    def event(self, event_id: EventId) -> Event:
        return self._events[event_id]

    def events(self) -> tuple[Event, ...]:
        if self._sorted is None:
            self._sorted = tuple(self._events[k] for k in sorted(self._events))
        return self._sorted

    def sent_by(self, person: PersonId) -> tuple[Event, ...]:
        return tuple(sorted(self._by_sender.get(person, ()), key=lambda e: e.id))

    def delivered_to(self, person: PersonId, role: Role | None = None) -> tuple[Delivery, ...]:
        return tuple(
            d
            for d in sorted(self._by_recipient.get(person, ()))
            if role is None or d.role is role
        )

    def view_of(self, person: PersonId) -> tuple[Event, ...]:
        """Every event the person sent, received as target, or witnessed (M16.C.4)."""
        ids = {e.id for e in self.sent_by(person)} | {d.event_id for d in self.delivered_to(person)}
        return tuple(self._events[k] for k in sorted(ids))

    def count_by_origin(self) -> dict[str, int]:
        """M1.F.7 — exogenous and endogenous events stay countable separately."""
        exogenous = sum(1 for e in self._events.values() if e.exogenous)
        return {"exogenous": exogenous, "endogenous": len(self._events) - exogenous}
