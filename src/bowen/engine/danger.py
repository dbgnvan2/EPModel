"""Danger: felt from what a person receives, or declared, and what it does (spec revision 30, `M1.B.3e`).

Purpose: a person who keeps intruding on another comes to be felt as dangerous by them. Contact with a dangerous person
         costs the one who receives it, and a cutoff toward them is principled in proportion to the danger felt.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3e
Tests:   tests/bowen/test_origin_danger_focus.py

The owner (2026-10-14): a self-chosen principled cutoff is "refusing contact with a person that is abusive or is a
serious threat (criminal, drug addict) where contact is very risky for one's safety", and "Agents, like people, can
become demanding to the point of being abusive, either because of tremendous stress or low level of differentiation."
So danger is read from behaviour: the impingement a person receives from another (`M4.C.1`), averaged over
``danger_window`` weeks, above ``danger_threshold``. A low-level or heavily stressed person intrudes more by the
existing mechanisms, so danger emerges from level and stress; nothing names an abuser. A declared ``danger`` (family
config) is a floor, for a threat the run does not simulate (a criminal, an addiction). [I] throughout.
"""

from __future__ import annotations

from src.bowen.engine.appraise import tie_change
from src.bowen.engine.contact import clamp_unit
from src.bowen.engine.events import Delivery
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState


def felt_from(intrusion: float, declared: float, params: EngineParams) -> float:
    """The danger felt at a running average of received intrusion, with the other's declared danger as a floor."""
    read = clamp_unit((intrusion - params.danger_threshold) / params.danger_threshold)
    return max(read, declared)


def update_danger(state: RunState, batch: tuple[Delivery, ...], params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 4 — accepted contact from someone felt as dangerous costs the recipient; then this week's
    received intrusion updates what each person feels from each other.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3e, #M6.4
    Tests:   tests/bowen/test_origin_danger_focus.py::test_m1b3e_contact_with_a_dangerous_person_costs_the_recipient,
             tests/bowen/test_origin_danger_focus.py::test_m1b3e_felt_danger_rises_with_sustained_intrusion,
             tests/bowen/test_origin_danger_focus.py::test_m1b3e_felt_danger_is_held_while_rejection_is_held

    Received intrusion is an exponential running average with rate 1 ÷ ``danger_window``, read from each delivered
    move after the week's move effects and landed contacts (`tie_change` on that state). The cost reads last week's felt
    danger: ``danger_gain`` × felt danger × the move's intensity ÷ ``intensity_scale``,
    a source in `M6.4`'s table. Refused moves never reach here (`M1.B.3c`), so a cutoff ends the cost at once.
    While a person holds rejection toward the other, the danger they feel from them does not fall (the owner,
    2026-10-14: "hold felt danger while rejection is held"): a refused move is never received, so without this the
    danger would fade once the cutoff worked and the cutoff would turn reactive by itself. It can still rise. A dead
    person's declared danger stops being a floor.
    """
    week: dict[tuple[TieId, PersonId], float] = {}
    cost: dict[PersonId, float] = {}
    for delivery in sorted(batch):
        event = state.store.event(delivery.event_id)
        change = tie_change(state, delivery, event, params)
        if change is None or not state.people[delivery.recipient].alive:
            continue
        tie, _, intrusion = change
        week[tie.id, delivery.recipient] = week.get((tie.id, delivery.recipient), 0.0) + intrusion
        felt = tie.felt_danger.get(delivery.recipient, 0.0)
        if felt > 0.0:
            paid = params.danger_gain * felt * event.intensity / params.intensity_scale
            cost[delivery.recipient] = cost.get(delivery.recipient, 0.0) + paid
    for pid, paid in cost.items():
        state.people[pid].acute_anxiety += paid
    rows = []
    for tie_id in sorted(state.ties):
        tie = state.ties[tie_id]
        for member in tie_id.members():
            if not state.people[member].alive:
                continue
            before = tie.received_intrusion.get(member, 0.0)
            now = before + (week.get((tie_id, member), 0.0) - before) / params.danger_window
            if now or before:
                tie.received_intrusion[member] = now
            other = state.people[tie_id.other(member)]
            felt = felt_from(now, other.danger if other.alive else 0.0, params)
            if tie.rejection.get(member, 0.0) > 0.0:  # the owner, 2026-10-14: held while rejection is held
                felt = max(felt, tie.felt_danger.get(member, 0.0))
            if felt != tie.felt_danger.get(member, 0.0):
                rows.append((tie_id, f"felt_danger:{member}", felt - tie.felt_danger.get(member, 0.0)))
                tie.felt_danger[member] = felt
    records = []
    if cost:
        records.append(EffectRecord(state.tick, "danger_contact", None, acute_anxiety=tuple(sorted(cost.items()))))
    if rows:
        records.append(EffectRecord(state.tick, "felt_danger", None, ties=tuple(rows)))
    return records
