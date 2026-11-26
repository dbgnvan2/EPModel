"""A scripted divorce — a spouse leaves the household, and the event's load lands on the couple and the children.

Purpose: build the events of one divorce at its week, from the run's state as it is that week: a ``SEPARATION``
         (structural) and a ``DIVORCE`` stressor for each spouse and for each child through their tie to each parent,
         sized from the Holmes–Rahe scale (``config/bowen/event_sizes.md``). ``DivorceSource`` schedules them.
Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1m, #M1.F.6a
Tests:   tests/bowen/test_reactive_amount.py

Divorce is not drawn by the life course (revision 66); a scenario schedules it. ``scale`` lets a scenario size it up for
a family under more strain — by level or chronic anxiety, never by observed conflict, which would make conflict cause
child focus by construction (the owner, 2026-11-19; `M7.C.1m`). [I]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.bowen.engine.event_sizes import sized_intensity
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import Emitter, ScheduledDropRecord
from src.bowen.engine.objects import SCALE_MAX
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState
from src.bowen.engine.tick import Source

SEPARATION, DIVORCE = "SEPARATION", "DIVORCE"


def divorce_events(state: RunState, params: EngineParams, week: int, leaver: PersonId,
                   scale: float = 1.0) -> tuple[Event, ...]:
    """Purpose: the events of a divorce in the reference family at ``week``, ``leaver`` the spouse who moves out.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1m
    Tests:   tests/bowen/test_reactive_amount.py::test_m7c1m_divorce_moves_the_leaver_and_keeps_the_tie

    The loads are sized from ``state`` as it is when this is called: who is alive, which children exist and each
    child tie's bond energy. So it builds only in ``week`` itself (revision 69, T-269): events built earlier would
    miss a child born before ``week`` and could address a spouse who has died. It raises if either spouse is dead, or
    if a parent–child tie of the family does not join one spouse to a child (revision 68, T-261 (3), (5)).
    """
    if week != state.tick:  # revision 69, T-269: built at its week, from that week's state
        raise ValueError(f"a divorce at week {week} is built in that week, not at tick {state.tick}; use DivorceSource")
    tie = state.family.marital_tie
    if tie is None or leaver not in tie.members():
        raise ValueError(f"{leaver} is not a spouse in the reference family")
    if scale < 0:
        raise ValueError("scale is non-negative")
    spouses = tie.members()
    dead = [str(p) for p in spouses if not state.people[p].alive]
    if dead:  # revision 68, T-261 (3): a widowed spouse cannot divorce
        raise ValueError(f"{', '.join(dead)} is not alive; a divorce needs both spouses living")
    events = [Event(id=EventId(week, "separation", 0), kind=SEPARATION, mechanism=Mechanism.SEPARATION, sender=None,
                    targets=(leaver,), intensity=0.0, timestamp=week, duration=1, exogenous=True,
                    source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS)]

    def load(n: int, target: PersonId, who: str, reach: float) -> Event:
        intensity, weeks = sized_intensity(state.kinds, params, DIVORCE, who, reach=reach)
        return Event(id=EventId(week, "divorce", n), kind=DIVORCE, mechanism=Mechanism.EXOGENOUS_STRESSOR,
                     sender=None, targets=(target,), intensity=intensity * scale, timestamp=week, duration=weeks,
                     exogenous=True, source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS)

    n = 0
    for spouse in spouses:
        events.append(load(n, spouse, "self", 1.0))
        n += 1
    for child_tie in sorted(state.family.parent_child_ties):
        parents = set(child_tie.members()) & set(spouses)
        # Revision 68, T-261 (5): a nuclear parent–child tie has exactly one spouse in it and is no spouse's tie to
        # their own parent (a family-of-origin tie), so a grandparent living in the household is not read as a child.
        if len(parents) != 1 or child_tie in state.family.origin_ties:
            raise ValueError(f"{child_tie} is not a tie between one spouse and a child of the nuclear family")
        child = next(p for p in child_tie.members() if p not in spouses)
        if not state.people[child].alive:
            continue
        events.append(load(n, child, "tie", state.ties[child_tie].bond_energy / SCALE_MAX))
        n += 1
    return tuple(events)


@dataclass
class DivorceSource:
    """Purpose: a Source that adds scripted divorces to ``inner``'s events, each built in its own week from the run's
             live ``state``, so it sees who has died and who has been born since the run began (revision 69, T-269).
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1m
    Tests:   tests/bowen/test_reactive_amount.py::test_m7c1m_t269_a_divorce_whose_spouse_has_died_is_dropped_and_recorded

    ``divorces`` holds ``(week, leaver, scale)``. A divorce whose spouse has died by its week is not a divorce: it is
    dropped and recorded in ``dropped`` as ``(week, leaver, the dead)``; ``built`` holds the weeks built, and ``summary``
    gives "N of M" with those not yet reached. ``state`` is the one object the run is given, read here, never written. [I]
    A drop is also emitted to ``emitter``, when one is given, as a ``ScheduledDropRecord`` (revision 70, T-295 (1)).
    A week asked for twice returns the same events and records nothing twice (T-295 (3)); ``rules`` and ``ticks`` are
    ``inner``'s, so a tool that reads them from its source reads them through this one (T-295 (2)).
    """

    inner: Source
    state: RunState
    params: EngineParams
    divorces: tuple[tuple[int, PersonId, float], ...]
    dropped: list[tuple[int, PersonId, tuple[PersonId, ...]]] = field(default_factory=list)
    built: list[int] = field(default_factory=list)
    emitter: Emitter | None = None
    _weeks: dict[int, tuple[Event, ...]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        weeks = [week for week, _, _ in self.divorces]
        if len(set(weeks)) != len(weeks):
            raise ValueError("one divorce per week: two would share their events' ids")
        if any(week < self.state.tick for week in weeks):
            raise ValueError(f"a divorce before the run's tick {self.state.tick} would never be built")
        tie = self.state.family.marital_tie
        strangers = [str(leaver) for _, leaver, _ in self.divorces if tie is None or leaver not in tie.members()]
        if strangers:
            raise ValueError(f"{', '.join(strangers)} is not a spouse in the reference family")

    def scheduled(self, tick: int) -> tuple[Event, ...]:
        if tick != self.state.tick:
            raise RuntimeError(f"DivorceSource asked for tick {tick} while its state is at tick {self.state.tick}")
        if tick not in self._weeks:  # T-295 (3): each week's drops and builds are recorded once
            self._weeks[tick] = self._build(tick)
        return tuple(self.inner.scheduled(tick)) + self._weeks[tick]

    def _build(self, tick: int) -> tuple[Event, ...]:
        events: list[Event] = []
        for week, leaver, scale in self.divorces:
            if week != tick:
                continue
            tie = self.state.family.marital_tie
            dead = tuple(p for p in tie.members() if not self.state.people[p].alive) if tie is not None else ()
            if dead:
                self.dropped.append((week, leaver, dead))
                if self.emitter is not None:
                    self.emitter.emit(ScheduledDropRecord(tick, DIVORCE, dead, "a spouse died before the week"))
                continue
            events += divorce_events(self.state, self.params, week, leaver, scale)
            self.built.append(week)
        return tuple(events)

    @property
    def rules(self) -> Any:
        return self.inner.rules  # T-295 (2): the wrapped source's policy rules

    @property
    def ticks(self) -> int:
        return self.inner.ticks

    def selections(self, tick: int, active: tuple[PersonId, ...], state: RunState):
        if state is not self.state:
            raise RuntimeError("DivorceSource is run on a state other than its own")
        return self.inner.selections(tick, active, state)

    def summary(self) -> str:
        waiting = len(self.divorces) - len(self.dropped) - len(self.built)
        return (f"{len(self.dropped)} of {len(self.divorces)} divorces dropped: a spouse had died before the week; "
                f"{len(self.built)} built, {waiting} not yet reached")
