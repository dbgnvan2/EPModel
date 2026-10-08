"""Perception and appraisal — tick steps 3 and 4.

Purpose: give each person the events delivered to them this tick, and appraise
         each one: as the change it makes to the receiver's two-sided deviation,
         through a gain that defends against content under anxiety, with the
         witness's, the speaker's and the calmer sender's terms.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.B.1, #M4.C.1, #M4.C.2, #M4.C.4, #M4.C.5, #M4.C.6, #M5.F.1, #M5.F.2, #M4.C.7, #M4.C.8, #M4.C.8a, #M4.C.9, #M4.C.10, #M1.F.2, #M1.F.3, #M1.F.4, #M1.F.5, #M1.F.8
Tests:   tests/bowen/test_appraise.py; tests/bowen/test_contact.py; tests/bowen/test_mechanisms.py

Every form below is the project's, graded [I]; the constants are in
``config/bowen/constants.md``. Phase C steps 1 and 2.

**A move addressed to a person on a tie** (`M4.C.1`) moves the receiver's felt
contact and impingement by the kind's components, and is appraised by the change
in deviation:

    scale     = intensity / intensity_scale × conductance × route_damping ** len(route) × fidelity
    Δ contact = contact component × scale × content_gain          (M4.C.2)
                × (1 − hollow_gain × sender's inward impingement)    (M5.F.1: empty words)
    Δ imp     = (impingement component + assault_gain × sender's outward impingement
                 + assertion_gain if the event is an assertion-form I-POSITION) × scale    (M5.F.1, M5.F.4)
    Δ acute   = appraisal_gain × steepness × (deviation after − before) × sign
                × (1 − effective_perspective × own_share)            (M4.C.6)
                × attention factor                                    (M4.C.8, M4.C.8a)

* **Gain** (`M4.C.2`): above ``defence_threshold`` points of excess anxiety the
  content of an event is not heard, only defended against. ``content_gain`` falls
  linearly from 1 to 0 between the threshold and twice the threshold and scales
  the contact component only — so an anxious person takes in no relief from an
  approach while its impingement still lands. Not a plain product.
* **Self-attribution** (`M4.C.6`): a person with systems perspective attributes
  part of what arrives to their own recent output on the tie — the share of the
  last ``reappraisal_window`` ticks' moves on it that they sent. Immediate; no
  new information arrives.
* **Act identity** (`M5.F.1`, `M5.F.2`): the sender's two outside-ness axes act
  **through the event** (`M4.B.2a`) — see ``outside_ness.py``. A forceful sender's
  move can land as a load where a calm sender's same move is relief: the opposite sign.
* **Attention** (`M4.C.8`, `M4.C.8a`): attention at the feeling channel amplifies
  by ``1 + attention_gain × (1 − objectivity)``; at the intellect it orders, by
  ``1 − attention_gain``. Objectivity is the receiver's outside-ness efficacy,
  ``1 − max(outward, inward)`` (`M1.A.9a`). *Changed at step 3: step 2 used 1 − the mean.*
* **The inward reading** (`M4.C.5`): the rise in the receiver's "too much" side from
  each delivered move — reading the event as critical. Returned for step 9's
  outside-ness drift; computed before any move is emitted.

**A witness** (`M4.C.9`) appraises from its own position — its own steepness,
its ties to **both** the sender and each target, and the exchange's intensity:

    Δ acute = witness_weight × intensity / intensity_scale × mean(cond(w, sender), cond(w, target))
              × route_damping ** len(route) × fidelity × steepness(w) × sign

It is not a copy of a target's appraisal. Being one step removed is also what
makes the witnessed form less reactive for the listener (`M4.C.7`).

**The speaker** (`M4.C.7`) takes an echo of what it addressed:
``speaker_echo_gain × scale × steepness(speaker)``, using the conductance of its tie
to the person addressed. Addressed to a neutral third on a thin tie, the echo is
smaller than addressed to the other directly.

**A calmer sender** (`M4.C.10`): when a delivered move's sender is less anxious
than its target, ``calm_transfer_rate × conductance × gap`` moves from the target to
the sender — a transfer, logged as one (`M6.4`). Seeking it is never a move; the
learner finds it.

**An exogenous stressor** has no tie: ``intensity / intensity_scale × steepness ×
fidelity``. **An endogenous symptom event** (`M7.D.1`) is appraised the same way,
through the receiver's tie to the bearer.

The batch is applied as a whole (`M1.F.8`): every change is computed from the
state before the batch, then applied in canonical order.
"""

from __future__ import annotations

from src.bowen.engine.contact import clamp_unit, deviation, deviation_at, excess, felt_contact, felt_impingement, steepness
from src.bowen.engine.events import Attention, Delivery, Event, Mechanism, Role
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import Person, Relationship
from src.bowen.engine.outside_ness import axes, efficacy
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState

APPRAISED = frozenset({Mechanism.MOVE, Mechanism.EXOGENOUS_STRESSOR, Mechanism.ENDOGENOUS_SYMPTOM})


def perceive(state: RunState, batch: tuple[Delivery, ...]) -> dict[PersonId, tuple[tuple[Delivery, Event], ...]]:
    """Purpose: what each person reads this tick — events addressed to them and events they witnessed.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.B.1
    Tests:   tests/bowen/test_mechanisms.py::test_m4b1_person_reads_addressed_and_witnessed_events
    """
    perceived: dict[PersonId, list[tuple[Delivery, Event]]] = {}
    for delivery in sorted(batch):
        perceived.setdefault(delivery.recipient, []).append((delivery, state.store.event(delivery.event_id)))
    return {p: tuple(v) for p, v in sorted(perceived.items())}


# --- the person-side terms ---------------------------------------------------------------


def content_gain(person: Person, params: EngineParams) -> float:
    """Purpose: the share of an event's content a person can take in at their anxiety.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.2
    Tests:   tests/bowen/test_appraise.py::test_m4c2_content_is_defended_against_above_threshold
    """
    over = max(0.0, excess(person) - params.defence_threshold)
    return clamp_unit(1.0 - over / params.defence_threshold)


def effective_perspective(person: Person, tie: Relationship | None, params: EngineParams) -> float:
    """Purpose: systems perspective as it holds on one tie, lost under anxiety and on a loaded tie.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.4, #M1.A.18d
    Tests:   tests/bowen/test_appraise.py::test_m4c4_perspective_falls_with_anxiety_and_on_a_loaded_tie

    One per-person value (``systems_perspective``), attenuated by acute anxiety and by
    the tie's emotional intensity, read as the person's deviation on it:

        effective = systems_perspective / (1 + excess / perspective_anxiety_scale + deviation)
    """
    held = person.systems_perspective or 0.0
    load = deviation(person, tie, params) if tie is not None else 0.0
    return held / (1.0 + excess(person) / params.perspective_anxiety_scale + load)


def objectivity(person: Person) -> float:
    """The receiver's outside-ness efficacy: both impingement axes low (M1.A.9a, M4.C.8a)."""
    return efficacy(person)


def attention_factor(person: Person, event: Event, params: EngineParams) -> float:
    """Purpose: attending to a channel amplifies it unless the stance is objective; attending to the intellect orders.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.8, #M4.C.8a
    Tests:   tests/bowen/test_appraise.py::test_m4c8_attention_at_feeling_amplifies_and_objectivity_gates_it
    """
    if event.attention is Attention.FEELING:
        return 1.0 + params.attention_gain * (1.0 - objectivity(person))
    if event.attention is Attention.INTELLECT:
        return 1.0 - params.attention_gain
    return 1.0


def own_share(state: RunState, person: PersonId, other: PersonId, window: int) -> float:
    """Share of the moves between two people in the last ``window`` ticks that ``person`` sent (M4.C.6)."""
    sent = total = 0
    for event in state.store.events():
        if event.mechanism is not Mechanism.MOVE or event.sender is None:
            continue
        if not state.tick - window < event.timestamp <= state.tick:
            continue
        if {event.sender, *event.targets} >= {person, other} and event.sender in (person, other):
            total += 1
            sent += event.sender == person
    return sent / total if total else 0.0


# --- one delivery ------------------------------------------------------------------------


def _scale(event: Event, conductance: float, params: EngineParams) -> float:
    return event.intensity / params.intensity_scale * conductance * params.route_damping ** len(event.route) * event.fidelity


def _conductance(state: RunState, a: PersonId, b: PersonId | None) -> float:
    if b is None:
        return 1.0
    tie = state.tie_between(a, b)
    return tie.conductance if tie is not None else 0.0


def _tie_change(state: RunState, delivery: Delivery, event: Event, params: EngineParams):
    """The tie and the (Δ contact, Δ impingement) a move delivered to its target makes, or None."""
    if event.mechanism is not Mechanism.MOVE or delivery.role is not Role.TARGET or event.sender is None:
        return None
    tie = state.tie_between(delivery.recipient, event.sender)
    if tie is None:
        return None
    contact, impingement = state.kinds.components_of(event.kind)
    scale = _scale(event, tie.conductance, params)
    receiver = state.people[delivery.recipient]
    outward, inward = axes(state.people[event.sender])
    d_contact = contact * scale * content_gain(receiver, params) * (1.0 - params.hollow_gain * inward)
    d_imp = (impingement + params.assault_gain * outward + (params.assertion_gain if event.assertion else 0.0)) * scale
    return tie, d_contact, d_imp


def inward_reading(state: RunState, delivery: Delivery, event: Event, params: EngineParams) -> float:
    """Purpose: how much more impinged on a person feels from one delivered move — reading it as critical.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.5
    Tests:   tests/bowen/test_outside_ness.py::test_m4c5_hearing_criticism_raises_inward_impingement
    """
    change = _tie_change(state, delivery, event, params)
    if change is None:
        return 0.0
    tie, _, d_imp = change
    person = state.people[delivery.recipient]
    imp = felt_impingement(person, tie)
    band_now = deviation_at(person, tie, params, 1.0, imp) - deviation_at(person, tie, params, 1.0, 0.0)
    band_after = deviation_at(person, tie, params, 1.0, clamp_unit(imp + d_imp)) - deviation_at(person, tie, params, 1.0, 0.0)
    return max(0.0, band_after - band_now)


def appraisal_delta(state: RunState, delivery: Delivery, event: Event, params: EngineParams) -> float:
    """Purpose: the change in the receiver's acute anxiety from one delivery, from the pre-batch state.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1, #M4.C.2, #M4.C.6, #M4.C.8, #M4.C.9, #M1.F.2, #M1.F.3, #M1.F.4
    Tests:   tests/bowen/test_contact.py::test_m4c1_delivered_event_is_appraised_by_its_change_in_deviation
    """
    person = state.people[delivery.recipient]
    sign = state.kinds.sign(event.kind, event.source_position)
    change = _tie_change(state, delivery, event, params)
    if change is not None:
        tie, d_contact, d_imp = change
        contact, imp = felt_contact(person, tie), felt_impingement(person, tie)
        before = deviation_at(person, tie, params, contact, imp)
        after = deviation_at(person, tie, params, clamp_unit(contact + d_contact), clamp_unit(imp + d_imp))
        reattributed = effective_perspective(person, tie, params) * own_share(
            state, person.id, event.sender, params.reappraisal_window
        )
        return (
            params.appraisal_gain * steepness(person, params) * (after - before) * sign
            * (1.0 - reattributed) * attention_factor(person, event, params)
        )
    if delivery.role is Role.WITNESS:
        return witness_delta(state, person, event, params) * sign
    conductance = _conductance(state, person.id, event.sender)
    return _scale(event, conductance, params) * steepness(person, params) * sign * attention_factor(person, event, params)


def witness_delta(state: RunState, witness: Person, event: Event, params: EngineParams) -> float:
    """Purpose: a witness appraises from its own position, reading its ties to both parties.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.9, #M4.C.7, #M1.F.5
    Tests:   tests/bowen/test_appraise.py::test_m4c9_witness_appraisal_reads_both_ties
    """
    to_sender = _conductance(state, witness.id, event.sender) if event.sender is not None else 1.0
    to_targets = [_conductance(state, witness.id, t) for t in event.targets] or [0.0]
    reach = (to_sender + max(to_targets)) / 2.0
    return params.witness_weight * _scale(event, reach, params) * steepness(witness, params)


def speaker_echo(state: RunState, delivery: Delivery, event: Event, params: EngineParams) -> float:
    """Purpose: the speaker's own reaction to what it addressed, smaller on a thinner tie.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.7
    Tests:   tests/bowen/test_appraise.py::test_m4c7_addressing_a_neutral_third_is_less_reactive_for_the_speaker
    """
    if event.mechanism is not Mechanism.MOVE or delivery.role is not Role.TARGET or event.sender is None:
        return 0.0
    speaker = state.people[event.sender]
    if not speaker.alive:
        return 0.0
    conductance = _conductance(state, event.sender, delivery.recipient)
    return params.speaker_echo_gain * _scale(event, conductance, params) * steepness(speaker, params)


def calm_transfer(state: RunState, delivery: Delivery, event: Event, params: EngineParams) -> float:
    """Purpose: anxiety a calmer sender takes from its target, from the pre-batch state; 0 otherwise.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.10, #M6.4
    Tests:   tests/bowen/test_appraise.py::test_m4c10_a_calmer_sender_lowers_the_receivers_anxiety
    """
    if event.mechanism is not Mechanism.MOVE or delivery.role is not Role.TARGET or event.sender is None:
        return 0.0
    sender, receiver = state.people[event.sender], state.people[delivery.recipient]
    if not sender.alive:
        return 0.0
    gap = receiver.acute_anxiety - sender.acute_anxiety
    if gap <= 0:
        return 0.0
    return params.calm_transfer_rate * _conductance(state, receiver.id, sender.id) * gap


# --- the batch -----------------------------------------------------------------------------


def apply_appraisal(
    state: RunState,
    perceived: dict[PersonId, tuple[tuple[Delivery, Event], ...]],
    params: EngineParams,
) -> tuple[list[EffectRecord], dict[tuple[TieId, PersonId], float], dict[PersonId, float]]:
    """Purpose: apply one tick's appraisals, contact changes, echoes and calm transfers as a batch (M1.F.8).
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1, #M4.C.7, #M4.C.10, #M1.F.5, #M1.F.8
    Tests:   tests/bowen/test_mechanisms.py::test_m1f8_same_tick_batch_order_does_not_change_state

    Returns the records; for `M1.B.8`'s investment, the size of what each person
    appraised on each tie this tick, valence-blind (absolute); and `M4.C.5`'s inward
    reading per person.
    """
    items = [
        (delivery, event)
        for pid in sorted(perceived)
        for delivery, event in perceived[pid]
        if event.mechanism in APPRAISED and state.people[pid].alive
    ]
    items.sort(key=lambda x: x[0])
    anxiety: dict = {}
    per_event: dict = {}
    contact_moves: dict = {}
    transfers: dict = {}
    attended: dict[tuple[TieId, PersonId], float] = {}
    readings: dict[PersonId, float] = {}

    def add(event_id, person, value):
        if value:
            anxiety[person] = anxiety.get(person, 0.0) + value
            per_event.setdefault(event_id, {}).setdefault(person, 0.0)
            per_event[event_id][person] += value

    for delivery, event in items:
        delta = appraisal_delta(state, delivery, event, params)
        add(event.id, delivery.recipient, delta)
        add(event.id, event.sender, speaker_echo(state, delivery, event, params))
        change = _tie_change(state, delivery, event, params)
        if change is not None:
            tie, d_contact, d_imp = change
            key = (tie.id, delivery.recipient)
            c, i = contact_moves.get(key, (0.0, 0.0))
            contact_moves[key] = (c + d_contact, i + d_imp)
            attended[key] = attended.get(key, 0.0) + abs(delta)
            readings[delivery.recipient] = readings.get(delivery.recipient, 0.0) + inward_reading(
                state, delivery, event, params
            )
        moved = calm_transfer(state, delivery, event, params)
        if moved:
            transfers.setdefault(event.id, []).append((delivery.recipient, event.sender, moved))

    for person in sorted(anxiety):
        state.people[person].acute_anxiety += anxiety[person]
    for (tie_id, member), (d_contact, d_imp) in sorted(contact_moves.items()):
        tie = state.ties[tie_id]
        tie.felt_contact[member] = clamp_unit(tie.felt_contact[member] + d_contact)
        tie.felt_impingement[member] = clamp_unit(tie.felt_impingement[member] + d_imp)
    for event_id in sorted(transfers):
        for receiver, sender, moved in transfers[event_id]:
            state.people[receiver].acute_anxiety -= moved
            state.people[sender].acute_anxiety += moved

    records = [
        EffectRecord(tick=state.tick, mechanism="appraisal", cause=event_id,
                     acute_anxiety=tuple(sorted(per_event[event_id].items())))
        for event_id in sorted(per_event)
    ]
    for event_id in sorted(transfers):
        pairs: dict = {}
        for receiver, sender, moved in transfers[event_id]:
            pairs[receiver] = pairs.get(receiver, 0.0) - moved
            pairs[sender] = pairs.get(sender, 0.0) + moved
        records.append(EffectRecord(tick=state.tick, mechanism="calm_contact", cause=event_id,
                                    acute_anxiety=tuple(sorted(pairs.items()))))
    return records, attended, readings
