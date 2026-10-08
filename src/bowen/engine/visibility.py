"""Visibility — who receives or witnesses an event, when, and with what fidelity.

Purpose: compute an event's witness set and every recipient's delivery from tie
         and household state, so no sender ever chooses its audience.
Spec:    docs/bowen_agent_model_spec_v2.md#M3.E.1, #M1.F.1b, #M4.E.1a, #M1.F.4, #M3.C.1, #M3.C.2, #M8.5
Tests:   tests/bowen/test_activation_visibility.py

The spec names the inputs — co-residence, tie conductance and route — and not
the rule. The rule below is the project's, graded [I]:

* **Witness.** A person other than the sender and the targets witnesses an
  event when the event was not routed privately (an empty ``route``), they are
  alive, they share a household with the sender or with a target, and they hold
  an interactive tie of positive conductance to the sender or to a target.
* **M8.5.** A candidate is then checked through the live-position predicate
  with the sender and each target. If the candidate is fused into one of them,
  the group has fewer than three live positions and the candidate is not a
  neutral witness; it is returned as a peripheral-triangle candidate instead,
  for the Phase C mechanism that acts on alignment (M8.6). Phase B carries no
  fusion, so no candidate is excluded there.
* **Timing.** A target receives the event after its tie's latency (M3.C.2). A
  witness receives it in the same tick as the earliest target delivery it
  overheard. An event with no sender (exogenous) has no edge; it arrives in the
  tick it is scheduled, latency 0.
* **Fidelity.** Fidelity degrades per private hop (M1.F.4):
  ``per_hop_fidelity ** len(route)``. Phase B uses a declared constant where
  M3.D.4b has a draw class (plan decision D7).

A TRIGGER or RECONCILIATION names a tie and has no targets; it produces no
deliveries and no witnesses (M4.A.2: no contact). A move addressed across a
non-interactive tie — cut off, or a worry edge — is refused (M1.B.3: a cut-off
tie carries no events), loudly, rather than dropped. The exception is `REDUCE_CUTOFF`
across a cut-off tie, the act that reopens it (M5.B.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.bowen.engine.events import Delivery, Event, Role
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.live_positions import Occupant, positions_live
from src.bowen.engine.objects import Person, Relationship, TieState

# M5.B.3 names the move; the name is the spec's.
REDUCE_CUTOFF_KIND = "REDUCE_CUTOFF"


class MissingTie(ValueError):
    """A sender addressed a target it has no tie to."""


class InactiveTie(ValueError):
    """A move was addressed across a cut-off or worry-edge tie, which carries no events (M1.B.3).

    `REDUCE_CUTOFF` (M5.B.3) is the move meant to act on a severed tie, and it alone
    crosses a cut-off tie (not a worry edge).
    """


@dataclass(frozen=True)
class Visible:
    """The visibility component's answer for one event."""

    event: Event
    deliveries: tuple[Delivery, ...]
    peripheral_candidates: tuple[PersonId, ...]


class HouseholdConductanceVisibility:
    """Purpose: the Phase B visibility component.
    Spec:    docs/bowen_agent_model_spec_v2.md#M3.E.1, #M1.F.1b
    Tests:   tests/bowen/test_activation_visibility.py::test_m1f1b_witnesses_computed_from_household_and_conductance
    """

    component_id = "household_conductance_visibility"
    version = "1"

    def __init__(self, per_hop_fidelity: float) -> None:
        if not 0.0 < per_hop_fidelity <= 1.0:
            raise ValueError("per_hop_fidelity is in (0, 1]")
        self.per_hop_fidelity = per_hop_fidelity

    def fidelity_for(self, route: tuple[PersonId, ...]) -> float:
        """Purpose: fidelity after the route's private hops.
        Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.4
        Tests:   tests/bowen/test_activation_visibility.py::test_m1f4_fidelity_degrades_per_private_hop
        """
        return self.per_hop_fidelity ** len(route)

    @staticmethod
    def _holds_live_position(
        candidate: PersonId,
        sender: PersonId | None,
        target: PersonId,
        fusion: Mapping[PersonId, PersonId],
    ) -> bool:
        """Whether the candidate's own position is live in the group {sender, target, candidate}.

        The group is counted twice through the M8 predicate — with the candidate's
        fusion and without it — so a fusion between sender and target cannot be
        mistaken for the candidate's.
        """
        anchors = [a for a in (sender, target) if a is not None]
        others = [Occupant(a, fused_into=fusion.get(a)) for a in anchors]
        as_is = positions_live(others + [Occupant(candidate, fused_into=fusion.get(candidate))])
        unfused = positions_live(others + [Occupant(candidate)])
        return as_is == unfused

    def resolve(
        self,
        event: Event,
        people: Mapping[PersonId, Person],
        ties: Mapping[TieId, Relationship],
        fusion: Mapping[PersonId, PersonId] | None = None,
    ) -> Visible:
        """Purpose: fill the witness set and schedule every delivery for one event.
        Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.1b, #M4.E.1a, #M3.C.2, #M8.5
        Tests:   tests/bowen/test_activation_visibility.py::test_m4e1a_witnesses_filled_by_visibility_not_script
        """
        fusion = fusion or {}
        target_deliveries = []
        for target in event.targets:
            if event.sender is None:
                latency = 0
            else:
                tie = ties.get(TieId.of(event.sender, target))
                if tie is None:
                    raise MissingTie(f"{event.sender} has no tie to {target}")
                # M5.B.3: the one move meant to act on a severed tie crosses it; a worry edge
                # (M4.A.4) and every other move are still refused (Phase C step 7).
                crosses = event.kind == REDUCE_CUTOFF_KIND and tie.tie_state is TieState.CUT_OFF
                if not tie.interactive and not crosses:
                    raise InactiveTie(f"{event.kind} from {event.sender} to {target}: the tie carries no events")
                latency = tie.latency
            target_deliveries.append(
                Delivery(
                    delivered_tick=event.timestamp + latency,
                    event_id=event.id,
                    recipient=target,
                    role=Role.TARGET,
                    emitted_tick=event.timestamp,
                    latency=latency,
                )
            )

        witnesses: list[PersonId] = []
        peripheral: list[PersonId] = []
        if target_deliveries and not event.route:
            anchors = set(event.targets) | ({event.sender} if event.sender else set())
            households = {people[p].household_id for p in anchors}
            for candidate in sorted(people):
                person = people[candidate]
                if candidate in anchors or not person.alive or person.household_id not in households:
                    continue
                linked = any(
                    (tie := ties.get(TieId.of(candidate, anchor))) is not None
                    and tie.interactive
                    and tie.conductance > 0
                    for anchor in anchors
                )
                if not linked:
                    continue
                neutral = all(
                    self._holds_live_position(candidate, event.sender, target, fusion)
                    for target in event.targets
                )
                (witnesses if neutral else peripheral).append(candidate)

        resolved = event.with_witnesses(tuple(witnesses))
        first = min(d.delivered_tick for d in target_deliveries) if target_deliveries else None
        witness_deliveries = [
            Delivery(
                delivered_tick=first,
                event_id=event.id,
                recipient=w,
                role=Role.WITNESS,
                emitted_tick=event.timestamp,
                latency=first - event.timestamp,
            )
            for w in resolved.witnesses
        ]
        return Visible(
            event=resolved,
            deliveries=tuple(sorted(target_deliveries + witness_deliveries)),
            peripheral_candidates=tuple(peripheral),
        )
