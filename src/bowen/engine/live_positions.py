"""The live-position predicate — one rule, used in three places.

Purpose: count the occupants of a group who hold a live position.
Spec:    docs/bowen_agent_model_spec_v2.md#M8.1, #M8.2, #M8.3, #M8.4, #M8.5, #M13.1
Tests:   tests/bowen/test_live_positions.py

M8.1, as written in the spec::

    positions_live(g) = |{ p in g : present(p) AND NOT fused_into(p, other ∈ g) }|

M8.5 requires this to be implemented once and called from every site that
needs it: whether a third party is a witness or a new peripheral triangle
(visibility), whether the coach is in the outside position (M11.C.7), and
whether a triangle is active (M1.C.3). Callers describe each occupant; this
module never reads model state itself. An avoidance gate for M5.C is not built:
its uncalled helper, ``avoidance_available``, was removed at spec revision 39
(D27). No caller passes a fusion map, so the fused-occupant branch is not reached
in a run (M8.1 partial, D28).

The roles fall out of the rule rather than being special-cased: an external agent
in the outside position counts (M8.2), one who has taken a side does not (M8.3),
and a displaced member is not *present* in this sense (M8.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.counting import count


@dataclass(frozen=True)
class Occupant:
    """One member of a group, as the caller sees them this tick.

    ``present`` is physical availability; ``emotionally_active`` is false for a
    displaced member (M8.4). ``fused_into`` names the group member this
    occupant is fused into, or None.
    """

    person: PersonId
    present: bool = True
    emotionally_active: bool = True
    fused_into: PersonId | None = None

    def __post_init__(self) -> None:
        if self.fused_into == self.person:
            raise ValueError(f"{self.person} cannot be fused into themselves")


def _validated(group: Iterable[Occupant]) -> tuple[Occupant, ...]:
    occupants = tuple(group)
    people = [o.person for o in occupants]
    if len(set(people)) != len(people):
        raise ValueError(f"an occupant appears twice: {sorted(map(str, people))}")
    return occupants


def positions_live(group: Iterable[Occupant]) -> int:
    """Purpose: count occupants who are present, active, and not fused into another group member.
    Spec:    docs/bowen_agent_model_spec_v2.md#M8.1, #M8.2, #M8.3, #M8.4
    Tests:   tests/bowen/test_live_positions.py::test_m81_positions_live_counts_unfused_present_occupants
    """
    occupants = _validated(group)
    members = {o.person for o in occupants}
    return count(
        1
        for o in occupants
        if o.present
        and o.emotionally_active
        # Fusion into someone outside the group does not remove a position here:
        # the rule reads "fused_into(p, other ∈ g)".
        and not (o.fused_into is not None and o.fused_into in members)
    )
