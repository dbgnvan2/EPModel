"""A scripted divorce — a spouse leaves the household, and the event's load lands on the couple and the children.

Purpose: build the events of one divorce at one week, for a scenario to schedule: a ``SEPARATION`` (structural) and a
         ``DIVORCE`` stressor for each spouse and for each child through their tie to each parent, sized from the
         Holmes–Rahe scale (``config/bowen/event_sizes.md``).
Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1m, #M1.F.6a
Tests:   tests/bowen/test_reactive_amount.py

Divorce is not drawn by the life course (revision 66); a scenario schedules it. ``scale`` lets a scenario size it up for
a family under more strain — by level or chronic anxiety, never by observed conflict, which would make conflict cause
child focus by construction (the owner, 2026-11-19; `M7.C.1m`). [I]
"""

from __future__ import annotations

from src.bowen.engine.event_sizes import sized_intensity
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.objects import SCALE_MAX
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState

SEPARATION, DIVORCE = "SEPARATION", "DIVORCE"


def divorce_events(state: RunState, params: EngineParams, week: int, leaver: PersonId,
                   scale: float = 1.0) -> tuple[Event, ...]:
    """Purpose: the events of a divorce in the reference family at ``week``, ``leaver`` the spouse who moves out.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1m
    Tests:   tests/bowen/test_reactive_amount.py::test_m7c1m_divorce_moves_the_leaver_and_keeps_the_tie

    The loads are sized from ``state`` as it is when this is called: who is alive, which children exist and each
    child tie's bond energy. A scenario that builds the events before the run sizes them from the starting state,
    so a child born before ``week`` gets none (review 2026-11-19).
    """
    tie = state.family.marital_tie
    if tie is None or leaver not in tie.members():
        raise ValueError(f"{leaver} is not a spouse in the reference family")
    if scale < 0:
        raise ValueError("scale is non-negative")
    spouses = tie.members()
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
        if not set(child_tie.members()) & set(spouses):
            continue
        child = next(p for p in child_tie.members() if p not in spouses)
        if not state.people[child].alive:
            continue
        events.append(load(n, child, "tie", state.ties[child_tie].bond_energy / SCALE_MAX))
        n += 1
    return tuple(events)
