"""Cutoff as rejection.

Purpose: hold a cutoff as the cutter's rejection of the other, on a tie that stays open; refuse what the rejected
         person sends, at the sender's cost; and lower rejection when the cutter accepts contact.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3c, #M5.B.3, #M4.C.1c, #M6.4
Tests:   tests/bowen/test_rejection.py

Absence of interaction is distance, not cutoff (`M1.B.3c`). So:

* each member of a tie holds a ``rejection`` in [0, 1]; the tie is cut off while either holds any, and its
  ``tie_state`` says so. It stays interactive;
* a move delivered to a person whose rejection of its sender is above ``rejection_refusal_threshold`` is **refused**,
  except ``CUTOFF`` and ``REDUCE_CUTOFF``. It does not reach that person at all. Their rejection rises, and the sender
  bears the refusal as acute anxiety. Below the threshold the move is received as usual (graded rejection);
* rejection fades by ``rejection_fade`` every week (``fade_rejection``), so a single cutoff wears off unless refusals
  or further cutoffs renew it;
* ``REDUCE_CUTOFF`` is the contact the cutter accepts. It lowers both members' rejection (``moves.reduce_cutoff``);
* the pull back toward contact is not written here. While a tie is cut, neither member rests at any contact
  (``contact.resting_contact``), so the "too little" side grows for both until the tie reopens (`M4.C.1c`).

Every rate is [I] (``config/bowen/constants.md``).
"""

from __future__ import annotations

from src.bowen.engine.contact import loss_weight, steepness
from src.bowen.engine.events import Delivery, Mechanism, Role
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import SCALE_MAX, Relationship, TieState
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState

# M5.A.1 fixes the move repertoire, so these names are the spec's, not editorial content.
NEVER_REFUSED = frozenset({"CUTOFF", "REDUCE_CUTOFF"})
CEILING = 1.0  # rejection is a share, [0, 1]


def raise_rejection(tie: Relationship, member: PersonId, amount: float) -> float:
    """Purpose: raise one member's rejection of the other, to the ceiling; the tie is then cut off.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3c
    Tests:   tests/bowen/test_rejection.py::test_m1b3c_repeated_refusals_raise_rejection_to_its_ceiling

    Returns the change.
    """
    before = tie.rejection.get(member, 0.0)
    tie.rejection[member] = min(CEILING, before + amount)
    if tie.rejected() and tie.tie_state is not TieState.CUT_OFF:
        tie.uncut_state = tie.tie_state  # a hardened distant tie stays distant once the cutoff ends
        tie.tie_state = TieState.CUT_OFF
    return tie.rejection[member] - before


def lower_rejection(tie: Relationship, member: PersonId, amount: float) -> float:
    """Purpose: lower one member's rejection; the tie returns to its state before the cut when neither holds any.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3c, #M5.B.3
    Tests:   tests/bowen/test_rejection.py::test_m5b3_accepted_contact_lowers_both_rejections_and_reopens_the_tie_at_zero

    Returns the change (zero or negative).
    """
    before = tie.rejection.get(member, 0.0)
    if before == 0:
        return 0.0
    tie.rejection[member] = max(0.0, before - amount)
    if tie.rejection[member] == 0.0:
        end_principled(tie, member)
    if tie.tie_state is TieState.CUT_OFF and not tie.rejected():
        tie.tie_state = tie.uncut_state
    return tie.rejection[member] - before


def end_principled(tie: Relationship, member: PersonId) -> None:
    """Purpose: a principled cutoff is the declared rejection only; once it is gone, or the holder's own policy cuts off
    on the tie, what they hold there is reactive.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3d
    Tests:   tests/bowen/test_cutoff_and_focus.py::test_m1b3d_a_policy_cutoff_after_a_principled_one_is_reactive,
             tests/bowen/test_cutoff_and_focus.py::test_m1b3d_a_faded_principled_cutoff_does_not_return_as_principled
    """
    if member in tie.principled:
        tie.principled = tuple(m for m in tie.principled if m != member)


def strength(intensity: float, params: EngineParams) -> float:
    return intensity / params.intensity_scale


def refuse(state: RunState, batch: tuple[Delivery, ...], params: EngineParams) -> tuple[tuple[Delivery, ...], list]:
    """Purpose: take out of a tick's batch the moves a rejecting person refuses, and apply each refusal.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3c, #M6.4
    Tests:   tests/bowen/test_rejection.py::test_m1b3c_a_refused_approach_never_reaches_the_cutter_and_costs_the_sender

    Returns the deliveries that go on to be recorded and perceived, and one ``refusal`` record per refused delivery
    (the log's record of it). A refused delivery is never recorded as delivered. Only the target's delivery is
    refused; a witness still sees the move.
    """
    accepted: list[Delivery] = []
    records: list = []
    for delivery in sorted(batch):
        event = state.store.event(delivery.event_id)
        tie = (state.tie_between(delivery.recipient, event.sender)
               if event.sender is not None and delivery.role is Role.TARGET else None)
        refused = (
            tie is not None and event.mechanism is Mechanism.MOVE and event.kind not in NEVER_REFUSED
            and tie.rejection.get(delivery.recipient, 0.0) > params.rejection_refusal_threshold
            and state.people[event.sender].alive and state.people[delivery.recipient].alive
        )
        if not refused:
            accepted.append(delivery)
            continue
        size = strength(event.intensity, params)
        rose = raise_rejection(tie, delivery.recipient, params.rejection_refusal_gain * size)
        sender = state.people[event.sender]
        cost = params.refusal_cost_gain * size * tie.conductance * steepness(sender, params)
        sender.acute_anxiety += cost
        records.append(EffectRecord(
            state.tick, "refusal", event.id,
            acute_anxiety=((sender.id, cost),) if cost else (),
            ties=((tie.id, f"rejection:{delivery.recipient}", rose),),
        ))
    return tuple(accepted), records


def fade_rejection(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: every held rejection fades by a fixed amount each week; a tie stops being cut off when none is left.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3c
    Tests:   tests/bowen/test_rejection.py::test_m1b3c_unrenewed_rejection_fades_and_the_tie_reopens

    Step 9. Refusals and cutoffs in the same week have already renewed it (step 2 and step 4).
    """
    changes = []
    for tie_id in sorted(state.ties):
        tie = state.ties[tie_id]
        if not tie.rejected():
            continue
        for member in sorted(tie.rejection):
            change = lower_rejection(tie, member, params.rejection_fade)
            if change:
                changes.append((tie_id, f"rejection:{member}", change))
        if not tie.rejected():
            changes.append((tie_id, f"tie_state_{tie.tie_state.value}", 1.0))
    return [EffectRecord(state.tick, "rejection_fade", None, ties=tuple(changes))] if changes else []


def seed_declared_rejection(ties: dict, params: EngineParams) -> None:
    """Purpose: a tie the family declares cut off starts with both members rejecting the other.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3c, #M2.A.1
    Tests:   tests/bowen/test_rejection.py::test_m1b3c_a_declared_cutoff_starts_open_with_both_members_rejecting

    [I]: neither family says which member cut the other off, so both hold ``declared_rejection``. A principled cutoff
    names its holder (`M1.B.3d`), so only that member holds it.
    """
    for tie_id in sorted(ties):
        tie = ties[tie_id]
        if tie.tie_state is TieState.CUT_OFF and not tie.rejected():
            for member in tie.principled or tie_id.members():
                raise_rejection(tie, member, params.declared_rejection)


def reactive_cutoff_cost(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 1 — each person holding a reactive cutoff pays for it each week, more the less differentiated.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3d, #M6.4
    Tests:   tests/bowen/test_cutoff_and_focus.py::test_m1b3d_reactive_cutoff_raises_the_cutters_anxiety

    cost = reactive_cutoff_gain × rejection held × (100 − basic_level) ÷ 100, [I] (a placeholder the owner
    accepted). A principled cutoff (``tie.principled``) costs nothing; one toward a person felt as
    dangerous costs in proportion to the danger not felt (`M1.B.3e`).
    """
    paid: dict[PersonId, float] = {}
    for tie_id in sorted(state.ties):
        tie = state.ties[tie_id]
        for member, held in sorted(tie.rejection.items()):
            person = state.people[member]
            weight = loss_weight(person, tie)  # principled: 0; toward danger: 1 − the danger felt (revision 30)
            if held <= 0.0 or not person.alive or weight == 0.0:
                continue
            paid[member] = paid.get(member, 0.0) + weight * (
                params.reactive_cutoff_gain * held * (SCALE_MAX - person.basic_level) / SCALE_MAX)
    for member, amount in paid.items():
        state.people[member].acute_anxiety += amount
    if not paid:
        return []
    return [EffectRecord(state.tick, "reactive_cutoff", None, acute_anxiety=tuple(sorted(paid.items())))]
