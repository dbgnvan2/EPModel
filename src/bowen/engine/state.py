"""The state of one run: the objects, the queue, the store, and the clock.

Purpose: hold everything a tick reads and writes, in one place the mechanisms share.
Spec:    docs/bowen_agent_model_spec_v2.md#M3.D.1, #M16.B.3, #M1.C.3
Tests:   tests/bowen/test_mechanisms.py

``RunState`` is created fresh for every run (M11.D.6: a second run sees no
state from the first) and is never stored at module level.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import combinations

from src.bowen.engine.event_store import EventStore
from src.bowen.engine.events import EventKinds, EventQueue
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.objects import Family, Person, Relationship, Triangle


@dataclass(frozen=True)
class ActiveTrigger:
    """A TRIGGER's spike on one tie's standing term, for the ticks it lasts (M4.A.2)."""

    tie: TieId
    intensity: float
    first_tick: int
    last_tick: int


@dataclass
class RunState:
    people: dict[PersonId, Person]
    ties: dict[TieId, Relationship]
    family: Family
    kinds: EventKinds
    triangles: dict[TriangleId, Triangle] = field(default_factory=dict)
    queue: EventQueue = field(default_factory=EventQueue)
    store: EventStore = field(default_factory=EventStore)
    triggers: list[ActiveTrigger] = field(default_factory=list)
    tick: int = 0

    def ties_of(self, person: PersonId) -> tuple[Relationship, ...]:
        return tuple(self.ties[t] for t in sorted(self.ties) if person in t.members())

    def tie_between(self, a: PersonId, b: PersonId) -> Relationship | None:
        return self.ties.get(TieId.of(a, b))


def closed_triads(ties: dict[TieId, Relationship]) -> tuple[TriangleId, ...]:
    """Purpose: the persistent triangle topology — every trio whose three pairs all have ties.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.C.3
    Tests:   tests/bowen/test_mechanisms.py::test_m1c3_topology_is_the_closed_triads

    The spec stores topology apart from activity but does not say which trios
    are triangles. Requiring all three ties is the project's rule, graded [I].
    """
    people = sorted({p for t in ties for p in t.members()})
    return tuple(
        TriangleId.of(a, b, c)
        for a, b, c in combinations(people, 3)
        if all(TieId.of(x, y) in ties for x, y in ((a, b), (a, c), (b, c)))
    )


def new_run_state(
    people: dict[PersonId, Person],
    ties: dict[TieId, Relationship],
    family: Family,
    kinds: EventKinds,
) -> RunState:
    """Copy the inputs, so a run never mutates the declaration it started from (M11.D.6)."""
    people = {k: copy.deepcopy(v) for k, v in people.items()}
    ties = {k: copy.deepcopy(v) for k, v in ties.items()}
    triangles = {t: Triangle(t) for t in closed_triads(ties)}
    return RunState(people=people, ties=ties, family=copy.deepcopy(family), kinds=kinds, triangles=triangles)
