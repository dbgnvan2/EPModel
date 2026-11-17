"""The belief store about ties a person is not party to (`M9.8`).

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
rule is [I].
"""

from __future__ import annotations

import dataclasses

import math
from dataclasses import dataclass

from src.bowen.engine.contact import clamp_unit
from src.bowen.engine.events import Delivery, Event, Mechanism
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.invariants import InvariantViolation
from src.bowen.engine.log_records import BeliefWriteRecord
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


def starting_beliefs(state: RunState) -> list[BeliefWriteRecord]:
    """Purpose: the record of every person's starting belief about each tie they are not party to, written by
             ``initialise_beliefs`` before the first tick, so the founders' priors are in the log as an entrant's are
             when they enter (``life_course``; T-251). The first tick emits them; they change no state.
    Spec:    docs/bowen_agent_model_spec_v2.md#M9.8, #M16.A.5
    Tests:   tests/bowen/test_tick.py::test_t251_the_founders_starting_tie_beliefs_are_logged_at_week_0
    """
    records = []
    for pid in sorted(state.people):
        for tid, belief in sorted((state.people[pid].tie_beliefs or {}).items()):
            records += [BeliefWriteRecord(state.tick, pid, f"tie_beliefs[{tid}].tension", belief.tension, None),
                        BeliefWriteRecord(state.tick, pid, f"tie_beliefs[{tid}].contact", belief.contact, None)]
    return with_truth(state, records)


def update_beliefs(
    state: RunState,
    perceived: dict[PersonId, tuple[tuple[Delivery, Event], ...]],
    params: EngineParams,
) -> list[BeliefWriteRecord]:
    """Purpose: move each person's beliefs toward what was delivered to them this tick, and nothing else.
    Spec:    docs/bowen_agent_model_spec_v2.md#M9.8, #M4.B.2, #M1.F.4, #M1.F.8, #M16.A.5, #M16.A.5a
    Tests:   tests/bowen/test_beliefs.py::test_m98_belief_written_only_from_delivered_events,
             tests/bowen/test_beliefs.py::test_m16a5_a_tie_belief_write_is_its_own_record_with_its_truth

    Each write is logged as a ``BeliefWriteRecord`` (`M16.A.5`), apart from ground-truth effects.
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
        weight = math.fsum(f for f, _, _ in obs)
        if weight <= 0:
            continue
        tension = math.fsum(f * t for f, t, _ in obs) / weight
        contact = math.fsum(f * c for f, _, c in obs) / weight
        step = params.belief_rate * weight / len(obs)
        belief = store[tie]
        d_tension = step * (tension - belief.tension)
        d_contact = step * (contact - belief.contact)
        belief.tension = clamp_unit(belief.tension + d_tension)
        belief.contact = clamp_unit(belief.contact + d_contact)
        # The true counterpart is added to the record afterwards by ``with_truth`` (M16.A.5a); the update never reads it.
        if d_tension:
            changes.append(BeliefWriteRecord(state.tick, pid, f"tie_beliefs[{tie}].tension", belief.tension, None))
        if d_contact:
            changes.append(BeliefWriteRecord(state.tick, pid, f"tie_beliefs[{tie}].contact", belief.contact, None))
    return changes


def with_truth(state: RunState, records: list[BeliefWriteRecord]) -> list[BeliefWriteRecord]:
    """Purpose: put each tie belief's true counterpart on its log record, so the signed discrepancy can be read per
             tick (`M16.A.5a`). A pure observer of the log: it reads the tie's true state into the record only, after
             the belief is written, and writes no state. In the week, the truth is read at step 3, when the belief is
             written, before step 4 appraises the same deliveries (T-248, revision 62). A ``tie_beliefs[...]`` subject
             that names no tie raises ``InvariantViolation``.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.A.5, #M16.A.5a
    Tests:   tests/bowen/test_beliefs.py::test_m16a5_a_tie_belief_write_is_its_own_record_with_its_truth
    """
    if not records:
        return records
    ties = {f"tie_beliefs[{tie_id}]": tie for tie_id, tie in state.ties.items()}
    out = []
    for record in records:
        prefix, _, field_name = record.subject.rpartition(".")
        truth = None
        if prefix.startswith("tie_beliefs["):
            tie = ties.get(prefix)
            if tie is None:  # a tie belief with no tie is a defect, not a record with no truth; an invariant
                # violation, so run_tick logs the failing tick before the raise (T-239 (d), T-251)
                raise InvariantViolation(f"belief record {record.subject!r} names no tie in the field")
            truth = getattr(true_counterpart(tie), field_name)
        out.append(dataclasses.replace(record, true_value=truth))
    return out


def true_counterpart(tie: Relationship) -> TieBelief:
    """Purpose: the true-state counterpart of a belief, for `M16.A.5a`'s signed discrepancy.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.A.5a
    Tests:   tests/bowen/test_beliefs.py::test_m98_belief_can_differ_from_the_true_state

    Not called by the belief update: it reads the tie's true state, which the
    update must never read (`M4.B.2`).
    """
    members = tie.id.members()
    return TieBelief(
        tension=math.fsum(tie.felt_impingement[m] for m in members) / len(members),
        contact=math.fsum(tie.felt_contact[m] for m in members) / len(members),
    )
