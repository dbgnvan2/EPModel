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

import json

from src.bowen.engine.draws import DrawService
from src.bowen.engine.event_store import EventStore
from src.bowen.engine.events import EventKinds, EventQueue
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.objects import Family, Person, Relationship, Triangle


@dataclass(frozen=True)
class ActiveStressor:
    """An exogenous stressor that lasts: it keeps loading its target each week after its arrival (revision 18)."""

    event_id: object
    target: PersonId
    intensity: float
    first_tick: int
    last_tick: int | None  # None: no end (spec revision 28, M6.3h: a lost income lasts until someone finds work)


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
    stressors: list[ActiveStressor] = field(default_factory=list)  # revision 18: stressors that last
    # Revision 19 (M1.E.7): self-directed acts under load awaiting judgement (external.Practice).
    practice: list = field(default_factory=list)
    adjusted: dict = field(default_factory=dict)  # revision 19: (actor, other) -> the week the other last adjusted
    # Revision 21 (M1.B.5): (actor, other) -> the week of the actor's last self-directed act or hold toward the other.
    stances: dict = field(default_factory=dict)
    tick: int = 0
    draws: DrawService = field(default_factory=lambda: DrawService(0))
    # M1.A.4k (Phase D plan B6): each person's last estimator_window slow ticks of weeks (slow_tick.Week).
    estimator_history: dict = field(default_factory=dict)
    # Spec revision 28 (Phase D part II): the families other than the reference one, keyed by household (M7.C.1h), and
    # the life course's own state (``life_course.LifeCourse``); None when the run has no life course, as every run
    # before part II.
    other_families: dict = field(default_factory=dict)
    life: object | None = None

    def families(self) -> tuple[Family, ...]:
        """The reference family, then the others in household order."""
        return (self.family,) + tuple(self.other_families[h] for h in sorted(self.other_families))

    def family_of(self, person: PersonId) -> Family:
        """The family whose household the person lives in; the reference family when their household has none."""
        return self.other_families.get(self.people[person].household_id, self.family)

    def add_person(self, person: Person) -> None:
        """Purpose: a person enters the field during a run, at birth or partnering (M7.C.1h, M7.C.1i).
        Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1h, #M7.C.1i, #M1.A.20
        Tests:   tests/bowen/test_life_course.py::test_m1a20_a_birth_identifier_is_stable_across_arms
        """
        if person.id in self.people:
            raise ValueError(f"{person.id} is already in the field")
        self.people[person.id] = person

    def add_tie(self, tie: Relationship) -> tuple[TriangleId, ...]:
        """Purpose: a tie enters the field; the closed triads it completes become triangles (M1.C.3).
        Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1a, #M1.C.3
        Tests:   tests/bowen/test_life_course.py::test_m7c1h_a_new_tie_completes_its_triangles
        """
        if tie.id in self.ties:
            raise ValueError(f"{tie.id} is already in the field")
        for end in tie.id.members():
            if end not in self.people:
                raise ValueError(f"{tie.id} names {end}, who is not in the field")
        self.ties[tie.id] = tie
        a, b = tie.id.members()
        added = []
        for third in sorted(self.people):
            if third in (a, b):
                continue
            if TieId.of(a, third) in self.ties and TieId.of(b, third) in self.ties:
                trio = TriangleId.of(a, b, third)
                if trio not in self.triangles:
                    self.triangles[trio] = Triangle(trio)
                    added.append(trio)
        return tuple(added)

    def ties_of(self, person: PersonId) -> tuple[Relationship, ...]:
        # Cached per person while the set of ties is unchanged (plan B1); the tie objects are the live ones.
        key = tuple(self.ties)
        cache = self.__dict__.setdefault("_ties_of_cache", {})
        if cache.get("key") != key:
            cache.clear()
            cache["key"] = key
        if person not in cache:
            cache[person] = tuple(t for t in sorted(self.ties) if person in t.members())
        return tuple(self.ties[t] for t in cache[person])

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
    seed: int = 0,
) -> RunState:
    """Copy the inputs, so a run never mutates the declaration it started from (M11.D.6)."""
    people = {k: copy.deepcopy(v) for k, v in people.items()}
    ties = {k: copy.deepcopy(v) for k, v in ties.items()}
    triangles = {t: Triangle(t) for t in closed_triads(ties)}
    return RunState(
        people=people, ties=ties, family=copy.deepcopy(family), kinds=kinds, triangles=triangles,
        draws=DrawService(seed),
    )


def canonical_state(state: RunState) -> str:
    """Purpose: the run's object state as one canonical string, for comparing two runs.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.T.3, #M11.D.6
    Tests:   tests/bowen/test_log.py::test_m16t3_sink_does_not_change_results
    """
    from src.bowen.engine.log_records import plain

    body = {
        "tick": state.tick,
        "people": [plain(state.people[k]) for k in sorted(state.people)],
        "ties": [plain(state.ties[k]) for k in sorted(state.ties)],
        "triangles": [plain(state.triangles[k]) for k in sorted(state.triangles)],
        "family": plain(state.family),
        **({"other_families": [plain(state.other_families[h]) for h in sorted(state.other_families)]}
           if state.other_families else {}),
        **({"life": plain(state.life)} if state.life is not None else {}),  # the life course, revision 28
        "events": [plain(e) for e in state.store.events()],
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
