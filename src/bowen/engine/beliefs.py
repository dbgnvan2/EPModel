"""The belief store about ties a person is not party to — `M9.8`, Phase C's minimal form (P2a).

Purpose: hold, per person, a belief about the state of each tie it is not party to,
         written only from moves delivered to that person as target or witness,
         after per-hop fidelity, so that it can differ from the tie's true state.
Spec:    docs/bowen_agent_model_spec_v2.md#M9.8, #M9.1, #M4.B.2, #M16.A.5a
Tests:   tests/bowen/test_beliefs.py

**What a belief holds.** Two numbers in [0, 1] per tie, the believed counterparts
of the tie's true contact and impingement (the mean of its members' felt contact
and felt impingement, ``true_counterpart``), so the signed discrepancy `M16.A.5a`
asks for can be computed per tick. ``tension`` is believed impingement, the
quantity `M4.D.2`'s triangle position and `M5.C`'s "marital distance high" read
through belief (`M4.B.2`); ``contact`` is believed closeness.

**At t0** every person holds a belief about every tie it is not party to, at an
uninformed prior: tension 0 and contact ``interactive_resting_contact``, [I] —
a person assumes a tie it has seen nothing of is at rest. The prior reads no
true state.

**Each tick** (step 3, after perceive), for every move delivered to the person,
each leg sender→target the person is not party to is an observation of that tie:

    observed tension = max(0, impingement component) × intensity / intensity_scale, clamped to [0, 1]
    observed contact = max(0, contact component) × intensity / intensity_scale, clamped to [0, 1]

Only what was delivered is observable: the kind, its intensity and the delivery's
fidelity. The sender's hidden axes (`M5.F.1`'s assault and hollow terms) are not
read. Several observations of one tie in a tick are combined as their
fidelity-weighted mean, and the belief moves toward it by
``belief_rate × mean fidelity`` — order-independent, so the batch rule (`M1.F.8`)
holds. A belief with nothing delivered does not move: there is no drift toward
truth, which is what lets a misperceived alliance persist (`M9.8`). The smoothing
rule is [I] (revision 12, P2a).

The rest of `M9` — the family-level store, attribution (`M9.6`), belief as a
channel into appraisal (`M9.7`) — is Phase D.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bowen.engine.contact import clamp_unit
from src.bowen.engine.events import Delivery, Event, Mechanism
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import Person, Relationship
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState


@dataclass
class TieBelief:
    """One person's belief about one tie it is not party to (`M9.8`)."""

    tension: float
    contact: float


class BeliefsNotInitialised(RuntimeError):
    """A person has no belief store: ``initialise_beliefs`` was not called."""


def beliefs_of(person: Person) -> dict[TieId, TieBelief]:
    if person.tie_beliefs is None:
        raise BeliefsNotInitialised(f"{person.id} has no belief store")
    return person.tie_beliefs


def initialise_beliefs(people: dict[PersonId, Person], ties: dict[TieId, Relationship], params: EngineParams) -> None:
    """Purpose: give every person an uninformed belief about every tie it is not party to.
    Spec:    docs/bowen_agent_model_spec_v2.md#M9.8
    Tests:   tests/bowen/test_beliefs.py::test_m98_every_person_believes_about_each_tie_it_is_not_party_to
    """
    for pid in sorted(people):
        people[pid].tie_beliefs = {
            tid: TieBelief(tension=0.0, contact=params.interactive_resting_contact)
            for tid in sorted(ties)
            if pid not in tid.members()
        }


def update_beliefs(
    state: RunState,
    perceived: dict[PersonId, tuple[tuple[Delivery, Event], ...]],
    params: EngineParams,
) -> list[EffectRecord]:
    """Purpose: move each person's beliefs toward what was delivered to them this tick, and nothing else.
    Spec:    docs/bowen_agent_model_spec_v2.md#M9.8, #M4.B.2, #M1.F.4, #M1.F.8
    Tests:   tests/bowen/test_beliefs.py::test_m98_belief_written_only_from_delivered_events
    """
    seen: dict[tuple[PersonId, TieId], list[tuple[float, float, float]]] = {}
    for pid in sorted(perceived):
        for delivery, event in perceived[pid]:
            if event.mechanism is not Mechanism.MOVE or event.sender is None:
                continue
            contact, impingement = state.kinds.components_of(event.kind)
            strength = event.intensity / params.intensity_scale
            for target in event.targets:
                if target == event.sender:
                    continue
                tie = TieId.of(event.sender, target)
                seen.setdefault((pid, tie), []).append((
                    event.fidelity,
                    clamp_unit(max(0.0, impingement) * strength),
                    clamp_unit(max(0.0, contact) * strength),
                ))
    changes = []
    for (pid, tie), obs in sorted(seen.items()):
        person = state.people[pid]
        if not person.alive:
            continue
        store = beliefs_of(person)
        if tie not in store:  # the person's own tie, or one the family does not hold
            continue
        weight = sum(f for f, _, _ in obs)
        if weight <= 0:
            continue
        tension = sum(f * t for f, t, _ in obs) / weight
        contact = sum(f * c for f, _, c in obs) / weight
        step = params.belief_rate * weight / len(obs)
        belief = store[tie]
        d_tension = step * (tension - belief.tension)
        d_contact = step * (contact - belief.contact)
        belief.tension = clamp_unit(belief.tension + d_tension)
        belief.contact = clamp_unit(belief.contact + d_contact)
        if d_tension:
            changes.append((pid, f"tie_beliefs[{tie}].tension", d_tension))
        if d_contact:
            changes.append((pid, f"tie_beliefs[{tie}].contact", d_contact))
    return [EffectRecord(state.tick, "belief", None, people=tuple(changes))] if changes else []


def true_counterpart(tie: Relationship) -> TieBelief:
    """Purpose: the true-state counterpart of a belief, for `M16.A.5a`'s signed discrepancy.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.A.5a
    Tests:   tests/bowen/test_beliefs.py::test_m98_belief_can_differ_from_the_true_state

    Not called by the belief update: it reads the tie's true state, which the
    update must never read (`M4.B.2`).
    """
    members = tie.id.members()
    return TieBelief(
        tension=sum(tie.felt_impingement[m] for m in members) / len(members),
        contact=sum(tie.felt_contact[m] for m in members) / len(members),
    )
