"""What a person may read when it selects — the boundary `M4.B.2` draws, built by the engine.

Purpose: hand the policy one person's own state, the person's own ties as the person
         has them, its beliefs about ties it is not party to, the triangle topology
         it belongs to, and the events delivered to it this tick — and nothing else.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.B.2, #M4.B.3, #M4.D.1e, #M9.8
Tests:   tests/bowen/test_policy.py::test_m4b2_policy_imports_only_the_observation

The policy package imports this module and never ``RunState``, so a policy that
reached past the boundary would have to import something the static guard forbids.
What counts as the person's own:

* its own fields — acute and chronic anxiety, functional level, the two outside-ness
  axes, systems perspective, financial dependence, learned values;
* for each tie it is party to: who is on the other end, whether the tie is live
  (interactive, not cut off), and its **own** deviation on it (its own felt contact and
  impingement against its own optimum);
* public facts about the other end: alive, and whether it is an external agent;
* the closed triads it belongs to — topology, not state — and, for each person it
  could turn to, the triad a `TRIANGLE` act would form. That is read from its own
  deviation on its own ties (``moves.triangle_outsider``), as the act's physics reads it;
* its beliefs (`M9.8`) and this tick's deliveries to it.

Nothing here reads another person's anxiety, outside-ness or felt state, or the state
of a tie the person is not party to, or any future quantity (`M4.B.3`).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.bowen.engine.contact import deviation, excess
from src.bowen.engine.events import Role as DeliveryRole
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.moves import triangle_outsider
from src.bowen.engine.objects import Role, TieState
from src.bowen.engine.outside_ness import axes
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState


@dataclass(frozen=True)
class TieView:
    """One of the person's own ties, as the person has it."""

    other: PersonId
    live: bool                # interactive and not cut off
    cut_off: bool             # not live: cut off, or a worry edge
    severed: bool             # cut off (M1.B.3), the one non-live state REDUCE_CUTOFF can act on
    deviation: float          # the person's own deviation on the tie (M4.C.1)
    other_alive: bool
    other_external: bool


@dataclass(frozen=True)
class Delivered:
    """One event delivered to the person this tick: what it was, from whom, in which role."""

    kind: str
    sender: PersonId | None
    witnessed: bool


@dataclass(frozen=True)
class Observation:
    tick: int
    person: PersonId
    acute_excess: float
    functional_level: float
    outside_ness_outward: float
    outside_ness_inward: float
    systems_perspective: float
    financially_dependent: bool
    external: bool                                # the person is an external agent (M1.E.1)
    ties: tuple[TieView, ...]
    triads: tuple[TriangleId, ...]                # every closed triad the person belongs to (topology)
    triangle_for: Mapping[PersonId, TriangleId]   # target → the triad a TRIANGLE act to them forms
    beliefs: Mapping[TieId, tuple[float, float]]  # tie → (believed tension, believed contact)
    learned_values: Mapping[str, float]
    inbox: tuple[Delivered, ...]

    def tie_to(self, other: PersonId) -> TieView | None:
        return next((t for t in self.ties if t.other == other), None)


def observe(state: RunState, person: PersonId, params: EngineParams) -> Observation:
    """Purpose: build the one person's observation for selection.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.B.2
    Tests:   tests/bowen/test_policy.py::test_m4b2_observation_holds_no_other_persons_state
    """
    me = state.people[person]
    outward, inward = axes(me)
    ties = []
    for tie in sorted(state.ties_of(person), key=lambda t: t.id):
        other = next(m for m in tie.id.members() if m != person)
        cut = tie.tie_state is TieState.CUT_OFF
        ties.append(TieView(
            other=other, live=tie.interactive and not cut, cut_off=cut or not tie.interactive, severed=cut,
            deviation=deviation(me, tie, params),
            other_alive=state.people[other].alive, other_external=state.people[other].role is Role.EXTERNAL,
        ))
    triangle_for = {}
    for view in ties:
        found = triangle_outsider(state, person, view.other, params) if view.live and view.other_alive else None
        if found is not None:
            triangle_for[view.other] = found[0]
    inbox = tuple(
        Delivered(state.store.event(d.event_id).kind, state.store.event(d.event_id).sender,
                  d.role is DeliveryRole.WITNESS)
        for d in state.store.delivered_to(person)
        if d.delivered_tick == state.tick
    )
    return Observation(
        tick=state.tick, person=person, acute_excess=excess(me), functional_level=me.functional_level,
        outside_ness_outward=outward, outside_ness_inward=inward,
        systems_perspective=me.systems_perspective or 0.0, financially_dependent=me.financially_dependent,
        external=me.role is Role.EXTERNAL,
        ties=tuple(ties), triads=tuple(t for t in sorted(state.triangles) if person in t.members),
        triangle_for=MappingProxyType(triangle_for),
        beliefs=MappingProxyType({t: (b.tension, b.contact) for t, b in (me.tie_beliefs or {}).items()}),
        learned_values=MappingProxyType(dict(me.learned_values)), inbox=inbox,
    )
