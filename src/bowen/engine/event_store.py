"""The in-run event store — always present, and causally load-bearing.

Purpose: keep every emitted event and every delivery for the run, queryable
         per person from inside the run.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.B.3, #M16.C.4, #M16.D.1, #M1.F.7
Tests:   tests/bowen/test_events.py::test_m16b3_event_store_answers_per_person_queries

M16.B.3 separates two things called logging. This is the **event store**: it
has no off switch, because `M1.E.7c`'s first and fourth forms read from it,
so disabling it would change behaviour. The optional **persistence
sink**, which must be a pure observer, is ``src/bowen/io/sinks.py``.
The delayed, scoped view of `M16.D` is built on these queries (``delayed_view``).
"""

from __future__ import annotations

from src.bowen.engine.events import Delivery, Event, EventId, Mechanism, Role
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.counting import count


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
        # Phase D (plan B1): windowed indexes, so a mechanism reads only the moves it needs instead of the whole
        # run every tick. Each holds a subset of ``_events`` and is sorted by id when read, so every query returns
        # exactly what filtering ``events()`` would (tests/bowen/test_events.py::test_b1_indexes_match_a_full_scan).
        self._moves_at: dict[int, list[Event]] = {}
        self._moves_of_kind: dict[str, list[Event]] = {}
        self._moves_on_pair: dict[frozenset, list[Event]] = {}
        self._dirty: set = set()
        self._recipient_sorted: dict[PersonId, tuple[Delivery, ...]] = {}
        self._by_recipient_tick: dict[tuple[PersonId, int], list[Delivery]] = {}

    def record_event(self, event: Event) -> None:
        if event.id in self._events:
            raise ValueError(f"event {event.id} recorded twice")
        self._events[event.id] = event
        self._sorted = None
        if event.sender is not None:
            self._by_sender.setdefault(event.sender, []).append(event)
            if event.mechanism is Mechanism.MOVE:
                self._index_move(event)

    def _index_move(self, event: Event) -> None:
        for key, index in ((event.timestamp, self._moves_at), (event.kind, self._moves_of_kind)):
            index.setdefault(key, []).append(event)
            self._dirty.add((id(index), key))
        for target in dict.fromkeys(event.targets):
            if target == event.sender:
                continue
            pair = frozenset((event.sender, target))
            self._moves_on_pair.setdefault(pair, []).append(event)
            self._dirty.add((id(self._moves_on_pair), pair))

    def _read(self, index: dict, key) -> list[Event]:
        found = index.get(key)
        if found is None:
            return []
        if (id(index), key) in self._dirty:
            found.sort(key=lambda e: e.id)
            self._dirty.discard((id(index), key))
        return found

    def moves_in(self, after: int, upto: int) -> list[Event]:
        """Moves (with a sender) whose timestamp is in (after, upto], in id order."""
        out: list[Event] = []
        for tick in range(after + 1, upto + 1):
            out.extend(self._read(self._moves_at, tick))
        out.sort(key=lambda e: e.id)
        return out

    def moves_of_kind(self, kind: str) -> list[Event]:
        """Every move (with a sender) of one kind, in id order."""
        return list(self._read(self._moves_of_kind, kind))

    def moves_on_pair(self, a: PersonId, b: PersonId) -> list[Event]:
        """Every move (with a sender) from one of the two to the other, whoever else it also targets, in id order."""
        return list(self._read(self._moves_on_pair, frozenset((a, b))))

    def record_delivery(self, delivery: Delivery) -> None:
        if delivery.event_id not in self._events:
            raise ValueError(f"delivery for unrecorded event {delivery.event_id}")
        self._deliveries.append(delivery)
        self._by_recipient.setdefault(delivery.recipient, []).append(delivery)
        self._recipient_sorted.pop(delivery.recipient, None)
        self._by_recipient_tick.setdefault((delivery.recipient, delivery.delivered_tick), []).append(delivery)

    def event(self, event_id: EventId) -> Event:
        return self._events[event_id]

    def events(self) -> tuple[Event, ...]:
        if self._sorted is None:
            self._sorted = tuple(self._events[k] for k in sorted(self._events))
        return self._sorted

    def sent_by(self, person: PersonId) -> tuple[Event, ...]:
        return tuple(sorted(self._by_sender.get(person, ()), key=lambda e: e.id))

    def delivered_to(self, person: PersonId, role: Role | None = None) -> tuple[Delivery, ...]:
        ordered = self._recipient_sorted.get(person)
        if ordered is None:
            ordered = self._recipient_sorted[person] = tuple(sorted(self._by_recipient.get(person, ())))
        return ordered if role is None else tuple(d for d in ordered if d.role is role)

    def delivered_at(self, person: PersonId, tick: int) -> tuple[Delivery, ...]:
        """The deliveries that reached the person at one tick, in the order ``delivered_to`` gives them."""
        return tuple(sorted(self._by_recipient_tick.get((person, tick), ())))

    def delayed_view(self, person: PersonId, now: int, delay: int) -> tuple[Event, ...]:
        """Purpose: one agent's own events, older than the delay, and nothing else (M16.D.1).
        Spec:    docs/bowen_agent_model_spec_v2.md#M16.D.1, #M16.D.2
        Tests:   tests/bowen/test_external.py::test_m16t4_delayed_view_is_scoped_and_lagged

        Own: sent by the person, or delivered to it as target or witness. A routed event
        (a private hop, ``M1.F.4``) the person did not send and was not addressed by is
        excluded. Only events emitted at or before ``now - delay``.
        """
        if delay < 0:
            raise ValueError("the delay is non-negative")
        cutoff = now - delay
        ids = {e.id for e in self._by_sender.get(person, ()) if e.timestamp <= cutoff}
        for d in self._by_recipient.get(person, ()):
            event = self._events[d.event_id]
            if event.timestamp > cutoff:
                continue
            if event.route and d.role is not Role.TARGET:
                continue  # another agent's private hop
            ids.add(event.id)
        return tuple(self._events[k] for k in sorted(ids))

    def view_of(self, person: PersonId) -> tuple[Event, ...]:
        """Every event the person sent, received as target, or witnessed (M16.C.4)."""
        ids = {e.id for e in self.sent_by(person)} | {d.event_id for d in self.delivered_to(person)}
        return tuple(self._events[k] for k in sorted(ids))

    def count_by_origin(self) -> dict[str, int]:
        """M1.F.7 — exogenous and endogenous events stay countable separately."""
        exogenous = count(1 for e in self._events.values() if e.exogenous)
        return {"exogenous": exogenous, "endogenous": len(self._events) - exogenous}
