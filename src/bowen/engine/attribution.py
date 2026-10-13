"""Attribution and the family's focus — step 3, with the tie beliefs (Phase D B8, spec revision 25).

Purpose: write, for each person, where they believe the family's difficulty lies, from the moves they sent or were
         delivered; and write, for each person, what the family's anxious attention to them focuses.
Spec:    docs/bowen_agent_model_spec_v2.md#M9.6, #M9.1, #M1.A.11c, #M4.B.2
Tests:   tests/bowen/test_attribution.py

Revision 25 is Claude's recommendation, pending owner review (the owner asked for the work to proceed overnight).

**Attribution** (`M9.6`). Each move the person sent, or that was delivered to them as target or witness, names a
locus by its kind (``EventKinds.attribution``, from ``config/bowen/attribution.md``):

* ``other`` — for its sender, the targets; for a target, the sender; for a witness, half to each side;
* ``target`` — its targets, shared equally; ``sender`` — its sender.

A person's observations in a tick are combined as their fidelity-weighted mean (the sender's own act has fidelity 1),
and the weights move toward it by ``attribution_rate × summed fidelity``, capped at 1:

    w ← (1 − r) × w + r × observed share,     r = min(1, attribution_rate × Σ fidelity)

so the order of a tick's moves changes nothing (`M1.F.8`), and the weights sum to at most 1. Nothing undelivered
writes it, and nothing reads the true state (`M4.B.2`). No rule writes `M9.6`'s three patterns.

**The family's focus** (`M1.A.11c` term 2). Every person's ``channel_focus`` decays by ``focus_rate`` a week. A move
delivered to a person as its target, from a sender above their usual level (`M1.A.5e`), adds ``focus_rate ×
fidelity`` to the target's focus on that kind's channel (``EventKinds.focus``). What the family attends to under
anxiety then moves the channel a symptom shows in (`symptoms.expressed_channel`); constitution still overrides it.
"""

from __future__ import annotations

from src.bowen.engine.contact import excess
from src.bowen.engine.events import Delivery, Event, Mechanism, Role
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import SymptomChannel
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState
from src.bowen.engine.stress_response import above_usual


def loci(event: Event, role: str, locus: str) -> dict[PersonId, float]:
    """Purpose: whom one move places the family's difficulty with, as one observer sees it, as shares summing to 1.
    Spec:    docs/bowen_agent_model_spec_v2.md#M9.6
    Tests:   tests/bowen/test_attribution.py::test_m96_each_side_of_a_conflict_says_the_other
    """
    sender, targets = event.sender, tuple(event.targets)
    if locus == "sender":
        return {sender: 1.0}
    if locus == "target":
        return {t: 1.0 / len(targets) for t in targets}
    # "other": each side says the other; a witness splits it between the two sides
    if role == "sender":
        return {t: 1.0 / len(targets) for t in targets}
    if role == "target":
        return {sender: 1.0}
    sides = (sender, targets)  # a witness splits it equally between the two sides
    shares = {sender: 1.0 / len(sides)}
    for t in targets:
        shares[t] = shares.get(t, 0.0) + 1.0 / len(sides) / len(targets)
    return shares


def update_attribution(state: RunState, perceived, batch: tuple[Delivery, ...], params: EngineParams) -> list:
    """Purpose: move each person's attribution toward what the moves they sent or were delivered this tick place.
    Spec:    docs/bowen_agent_model_spec_v2.md#M9.6, #M9.1, #M4.B.2, #M1.F.8
    Tests:   tests/bowen/test_attribution.py::test_m96_attribution_is_written_only_from_moves_sent_or_delivered
    """
    seen: dict[PersonId, list[tuple[float, dict[PersonId, float]]]] = {}
    rules = state.kinds.attribution

    def observe(person, fidelity, shares):
        seen.setdefault(person, []).append((fidelity, shares))

    for pid in sorted(perceived):
        for delivery, event in perceived[pid]:
            locus = rules.get(event.kind)
            if event.mechanism is not Mechanism.MOVE or event.sender is None or locus is None:
                continue
            role = "target" if delivery.role is Role.TARGET else "witness"
            observe(pid, event.fidelity, loci(event, role, locus))
    sent = {}
    for delivery in batch:  # the sender knows what it did, once per move delivered this tick
        event = state.store.event(delivery.event_id)
        if event.mechanism is Mechanism.MOVE and event.sender is not None and event.kind in rules:
            sent[event.id] = event
    for event in sorted(sent.values(), key=lambda e: e.id):
        observe(event.sender, 1.0, loci(event, "sender", rules[event.kind]))
    changes = []
    for pid in sorted(seen):
        person = state.people[pid]
        if not person.alive:
            continue
        obs = seen[pid]
        weight = sum(f for f, _ in obs)
        if weight <= 0:
            continue
        observed: dict[PersonId, float] = {}
        for f, shares in obs:
            for who, share in shares.items():
                observed[who] = observed.get(who, 0.0) + f * share / weight
        rate = min(1.0, params.attribution_rate * weight)
        before = dict(person.attribution)
        for who in sorted(set(before) | set(observed)):
            person.attribution[who] = (1.0 - rate) * before.get(who, 0.0) + rate * observed.get(who, 0.0)
        changes += [(pid, f"attribution[{who}]", person.attribution[who] - before.get(who, 0.0))
                    for who in sorted(person.attribution) if person.attribution[who] != before.get(who, 0.0)]
    return [EffectRecord(state.tick, "attribution", None, people=tuple(changes))] if changes else []


def update_focus(state: RunState, batch: tuple[Delivery, ...], params: EngineParams) -> list:
    """Purpose: what the family's anxious attention to each person focuses decays, and moves delivered by an anxious
    sender add to it on that act's channel.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.11c
    Tests:   tests/bowen/test_attribution.py::test_m1a11c_an_anxious_familys_attention_moves_the_focus
    """
    for pid in sorted(state.people):
        person = state.people[pid]
        for channel in sorted(person.channel_focus, key=lambda c: c.value):
            person.channel_focus[channel] *= 1.0 - params.focus_rate
    changes = []
    for delivery in sorted(batch):
        if delivery.role is not Role.TARGET:
            continue
        event = state.store.event(delivery.event_id)
        channel = state.kinds.focus.get(event.kind)
        if event.mechanism is not Mechanism.MOVE or event.sender is None or channel is None:
            continue
        sender, target = state.people[event.sender], state.people[delivery.recipient]
        if not target.alive or above_usual(sender, excess(sender)) <= 0:
            continue
        add = params.focus_rate * event.fidelity
        target.channel_focus[SymptomChannel(channel)] = target.channel_focus.get(SymptomChannel(channel), 0.0) + add
        changes.append((target.id, f"channel_focus[{channel}]", add))
    return [EffectRecord(state.tick, "family_focus", None, people=tuple(changes))] if changes else []
