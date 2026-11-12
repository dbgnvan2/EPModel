"""The social cues: what a move's signal channels carry, and how a receiver reads them (T-202).

Purpose: while ``social_cues_enabled`` is on (`M4.C.1f`), stamp each move with its four signal channels from the
         sender's state (`M1.F.1c`, `M4.C.13`, `M4.C.15`, `M9.9`), read them at the receiver as a perceived cue vector
         (`M4.C.11`, `M4.C.12`, `M4.C.14`), turn that vector into felt contact and felt impingement (`M4.C.1e`), and
         keep each person's believed cue vector per partner (`M9.9`). Off, nothing here runs and nothing is drawn.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.1c, #M4.C.1e, #M4.C.1f, #M4.C.11, #M4.C.12, #M4.C.13, #M4.C.14,
         #M4.C.15, #M9.9, #M4.C.6, #M4.C.9, #M1.F.4
Tests:   tests/bowen/test_social_cues.py

Every form below is the project's and `[I]`; the constants are in ``config/bowen/constants.md``. The cues are
readings of a received signal: they move `M4.C.1`'s two drives and are not drives (`M4.C.1e`).

**The signals a sender sends** (``sender_signals``, at act). The words carry no cue (the deviation form, plan D1).
On each nonverbal channel *c* with share ``s_c``:

    anxious  = min(1, excess / defence_threshold)                                  (M4.C.13)
    believed = mean over the targets of max(0, −belief about the target's distress)  (M9.9)
    leak     = (cue_leak_gain × anxious + belief_leak_gain × believed) × s_c
    expectations = −leak × cue_leak_expectations_share;  distress = −leak × cue_leak_distress_share
    attention    = −attention_load_gain × min(1, stressor load / attention_load_scale) × s_c        (M4.C.15)

each clamped to [−1, 1]. The stressor load is the summed intensity ÷ ``intensity_scale`` of the lasting exogenous
stressors on the sender this week (`M1.F.6`), a proxy for `M1.A.10`'s unbuilt ``life_energy``.

**The reading** (``reading``, once per delivery, cached on the run state). A channel is present when the move is
delivered in presence (no private hop); after a hop only the words are (`M1.F.4`). Words are always detected. A
present nonverbal channel is detected with probability ``p = s + (1 − s) × m``, where ``m`` is the channel's largest
|cue| and ``s = cue_sensitivity_floor + cue_sensitivity_level_gain × (1 − level/100)`` (`M4.C.14`); an undetected
channel reads as nothing. On a detected channel each cue is read with the meaning error (`M4.C.12`):

    read = sent − bias + spread × (2u − 1),   clamped to [−1, 1]
    bias   = cue_bias_floor + cue_bias_level_gain × (1 − level/100) + cue_bias_anxiety_gain × anxious
             + cue_bias_belief_gain × max(0, −belief about the sender's cue)
    spread = cue_spread_floor + cue_spread_level_gain × (1 − level/100)

The error changes what a present signal is read to mean; it never creates one where none was detected. The
perceived vector weighs the words against the share-weighted mean of the detected nonverbal channels by
``w = min(cap, nonverbal_weight_base + nonverbal_weight_anxiety_gain × anxious)`` (`M4.C.11`); with no nonverbal
channel detected the words take all the weight. Every draw is of the ``cue_perception`` class, keyed by the event's
tick and index, the receiver, the sender and a label naming the channel and cue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.bowen.engine.contact import clamp_unit, excess
from src.bowen.engine.draws import DrawKey
from src.bowen.engine.events import Delivery, Event, Mechanism, Role
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import SCALE_MAX, Person
from src.bowen.engine.params import EngineParams
from src.bowen.engine.signals import CUES, NEUTRAL, NONVERBAL, Cue, SignalChannel, Signals
from src.bowen.engine.state import RunState

__all__ = ["Cue", "SignalChannel", "Signals", "CUES", "NONVERBAL", "NEUTRAL_VECTOR", "Reading", "on", "sender_signals", "reading",
           "cue_components", "witness_factor", "own_leak_share", "update_cue_beliefs", "belief"]

DRAW_CLASS = "cue_perception"
NEUTRAL_VECTOR = NEUTRAL
_ATTENTION, _APPROVAL, _EXPECTATIONS, _DISTRESS = (CUES.index(c) for c in CUES)


def on(params: EngineParams | None) -> bool:
    """Whether the social cues are on (`M4.C.1f`)."""
    return params is not None and params.social_cues_enabled == 1


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def anxious_share(person: Person, params: EngineParams) -> float:
    """Excess anxiety as a share of ``defence_threshold``, saturating at one (`M4.C.13`, `M4.C.11`, `M4.C.12`)."""
    return clamp_unit(excess(person) / params.defence_threshold)


def low_level_share(person: Person) -> float:
    """How far a person's functional level is below the top of the scale, 0 at 100 and 1 at 0 (`M4.C.14`)."""
    return clamp_unit((SCALE_MAX - person.functional_level) / SCALE_MAX)


def belief(person: Person, partner: PersonId | None) -> tuple[float, ...]:
    """The cue vector ``person`` believes ``partner`` sends it (`M9.9`); neutral when none is held."""
    if partner is None or not person.cue_beliefs:
        return NEUTRAL
    return person.cue_beliefs.get(partner.value, NEUTRAL)


def stressor_load(state: RunState, person: PersonId, params: EngineParams) -> float:
    """The lasting exogenous stressors on ``person`` this week, as summed intensity ÷ ``intensity_scale`` (`M4.C.15`)."""
    return math.fsum(
        s.intensity / params.intensity_scale for s in state.stressors
        if s.target == person and s.first_tick <= state.tick and (s.last_tick is None or s.last_tick >= state.tick)
    )


def _shares(params: EngineParams) -> dict[SignalChannel, float]:
    return {SignalChannel.TONE: params.cue_tone_share, SignalChannel.FACE: params.cue_face_share,
            SignalChannel.POSTURE: params.cue_posture_share}


# --- the sender --------------------------------------------------------------------------------------------------


def sender_signals(state: RunState, sender: PersonId, targets: tuple[PersonId, ...], params: EngineParams) -> Signals:
    """Purpose: the four signal channels a move carries, from the sender's own state; the sender does not choose them.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.1c, #M4.C.13, #M4.C.15, #M9.9
    Tests:   tests/bowen/test_social_cues.py::test_m4c13_sc4_leak_grows_with_the_senders_excess,
             tests/bowen/test_social_cues.py::test_m4c15_sc9_stressor_load_lowers_the_attention_cue,
             tests/bowen/test_social_cues.py::test_m99_sc10_believed_distress_is_sent_as_leak
    """
    person = state.people[sender]
    believed = [max(0.0, -belief(person, t)[_DISTRESS]) for t in targets if t != sender]
    believed_mean = math.fsum(believed) / len(believed) if believed else 0.0
    leak = params.cue_leak_gain * anxious_share(person, params) + params.belief_leak_gain * believed_mean
    load = min(1.0, stressor_load(state, sender, params) / params.attention_load_scale)
    channels = {}
    for channel, share in _shares(params).items():
        vector = [0.0] * len(CUES)
        vector[_ATTENTION] = _clamp(-params.attention_load_gain * load * share)
        vector[_EXPECTATIONS] = _clamp(-leak * share * params.cue_leak_expectations_share)
        vector[_DISTRESS] = _clamp(-leak * share * params.cue_leak_distress_share)
        channels[channel.value] = tuple(vector)
    return Signals(**channels)


# --- the receiver ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """One delivery's perceived cue vector, split into the part the words carry and the part tone, face and posture
    carry (`M4.C.6` reattributes the second only), each already weighted (`M4.C.11`)."""

    words: tuple[float, ...]
    nonverbal: tuple[float, ...]

    @property
    def total(self) -> tuple[float, ...]:
        return tuple(w + n for w, n in zip(self.words, self.nonverbal))


def nonverbal_weight(person: Person, params: EngineParams) -> float:
    """Purpose: the weight a receiver gives tone, face and posture against the words; it rises with acute anxiety.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.11
    Tests:   tests/bowen/test_social_cues.py::test_m4c11_sc5_nonverbal_share_rises_with_receiver_anxiety
    """
    return min(params.nonverbal_weight_cap,
               params.nonverbal_weight_base + params.nonverbal_weight_anxiety_gain * anxious_share(person, params))


def sensitivity(person: Person, params: EngineParams) -> float:
    """Purpose: how readily a neutral nonverbal signal is detected; it rises as level falls and never reaches zero.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.14
    Tests:   tests/bowen/test_social_cues.py::test_m4c14_sc7_lower_level_detects_a_small_leak_more_often
    """
    return params.cue_sensitivity_floor + params.cue_sensitivity_level_gain * low_level_share(person)


def meaning_bias(person: Person, believed: tuple[float, ...], params: EngineParams) -> tuple[float, ...]:
    """Purpose: the error toward each cue's threat pole: anxiety, a believed threat and a lower level raise it.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.12, #M4.C.14
    Tests:   tests/bowen/test_social_cues.py::test_m4c12_sc6_neutral_cue_read_as_threat_more_at_low_level
    """
    base = (params.cue_bias_floor + params.cue_bias_level_gain * low_level_share(person)
            + params.cue_bias_anxiety_gain * anxious_share(person, params))
    return tuple(base + params.cue_bias_belief_gain * max(0.0, -b) for b in believed)


def meaning_spread(person: Person, params: EngineParams) -> float:
    """The half-width of the meaning error's spread; it rises as level falls (`M4.C.14`)."""
    return params.cue_spread_floor + params.cue_spread_level_gain * low_level_share(person)


def _draw(state: RunState, event: Event, receiver: PersonId, purpose: str) -> float:
    key = DrawKey.make(DRAW_CLASS, tick=event.timestamp, actor=receiver, partner=event.sender, purpose=purpose,
                       index=event.id.index)
    return state.draws.uniform(key)


def _read(state: RunState, event: Event, person: Person, channel: SignalChannel, sent: tuple[float, ...],
          bias: tuple[float, ...], spread: float) -> tuple[float, ...]:
    return tuple(
        _clamp(s - b + spread * (2.0 * _draw(state, event, person.id, f"{channel.value}:{cue.value}") - 1.0))
        for cue, s, b in zip(CUES, sent, bias)
    )


def reading(state: RunState, delivery: Delivery, event: Event, params: EngineParams) -> Reading:
    """Purpose: what a receiver reads in a delivered move's signal channels — once per delivery, then cached.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1e, #M4.C.11, #M4.C.12, #M4.C.14, #M1.F.4, #M4.C.9
    Tests:   tests/bowen/test_social_cues.py::test_m4c12_sc6_presence_is_never_imagined,
             tests/bowen/test_social_cues.py::test_m4c9_sc3_private_hop_carries_words_only

    A move without signals is refused while the cues are on (`M1.F.1c`).
    """
    key = (event.id, delivery.recipient)
    cached = state.cue_readings.get(key)
    if cached is not None:
        return cached
    if event.signals is None:
        raise ValueError(f"move {event.id} carries no signals while the social cues are on (M1.F.1c)")
    person = state.people[delivery.recipient]
    believed = belief(person, event.sender)
    bias = meaning_bias(person, believed, params)
    spread = meaning_spread(person, params)
    words = _read(state, event, person, SignalChannel.WORDS, event.signals.words, bias, spread)
    detected: list[tuple[float, tuple[float, ...]]] = []
    if not event.route:  # in presence; a private hop carries the words only (M1.F.4)
        s = sensitivity(person, params)
        for channel, share in _shares(params).items():
            sent = event.signals.channel(channel)
            strongest = max(abs(v) for v in sent)
            if share > 0 and _draw(state, event, person.id, f"detect:{channel.value}") < s + (1.0 - s) * strongest:
                detected.append((share, _read(state, event, person, channel, sent, bias, spread)))
    if detected:
        w = nonverbal_weight(person, params)
        total_share = math.fsum(share for share, _ in detected)
        mean = tuple(math.fsum(share * v[i] for share, v in detected) / total_share for i in range(len(CUES)))
        result = Reading(tuple((1.0 - w) * v for v in words), tuple(w * v for v in mean))
    else:
        result = Reading(words, NEUTRAL)
    state.cue_readings[key] = result
    return result


def _weights(params: EngineParams) -> tuple[tuple[float, float], ...]:
    """Each cue's (felt contact, felt impingement) weight, in the order of ``CUES`` (`M4.C.1e`)."""
    return (
        (params.cue_attention_contact_weight, params.cue_attention_impingement_weight),
        (params.cue_approval_contact_weight, params.cue_approval_impingement_weight),
        (params.cue_expectations_contact_weight, params.cue_expectations_impingement_weight),
        (params.cue_distress_contact_weight, params.cue_distress_impingement_weight),
    )


def cue_components(vector: tuple[float, ...], params: EngineParams) -> tuple[float, float]:
    """Purpose: the felt contact and felt impingement a perceived cue vector adds, before the move's scale.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1e
    Tests:   tests/bowen/test_social_cues.py::test_m4c1e_sc1_threat_pole_lowers_contact_and_raises_impingement

    A cue at its threat pole (negative) lowers felt contact by its contact weight and raises felt impingement by its
    impingement weight; at its other pole it does the reverse.
    """
    weights = _weights(params)
    contact = math.fsum(wc * p for (wc, _), p in zip(weights, vector))
    impingement = math.fsum(-wi * p for (_, wi), p in zip(weights, vector))
    return contact, impingement


def unmet_expectation(person: Person, partner: PersonId | None, vector: tuple[float, ...],
                      params: EngineParams) -> float:
    """Purpose: the felt contact lost when an attention or approval cue falls short of what is believed (`M9.9`).
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1e, #M9.9
    Tests:   tests/bowen/test_social_cues.py::test_m4c1e_sc1_an_unmet_expectation_reads_as_too_little
    """
    believed = belief(person, partner)
    shortfall = math.fsum(max(0.0, believed[i] - vector[i]) for i in (_ATTENTION, _APPROVAL))
    return -params.unmet_expectation_weight * shortfall


def witness_factor(vector: tuple[float, ...], params: EngineParams) -> float:
    """Purpose: a witness's appraisal is larger for the threat it reads in the exchange's channels (`M4.C.9`).
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.9
    Tests:   tests/bowen/test_social_cues.py::test_m4c9_sc3_witness_reads_all_four
    """
    return 1.0 + math.fsum((wc + wi) * max(0.0, -p) for (wc, wi), p in zip(_weights(params), vector))


def _leak(event: Event) -> float:
    if event.signals is None:
        return 0.0
    return math.fsum(abs(v) for channel in NONVERBAL for v in event.signals.channel(channel))


def own_leak_share(state: RunState, person: PersonId, other: PersonId, window: int) -> float:
    """Purpose: the share of the nonverbal leak on a tie in the last ``window`` ticks that ``person`` sent (`M4.C.6`).
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.6
    Tests:   tests/bowen/test_social_cues.py::test_m4c6_sc8_own_leak_share_reads_the_nonverbal_channels
    """
    own = theirs = 0.0
    for event in state.store.moves_in(state.tick - window, state.tick):
        if {event.sender, *event.targets} >= {person, other} and event.sender in (person, other):
            if event.sender == person:
                own += _leak(event)
            else:
                theirs += _leak(event)
    total = own + theirs
    return own / total if total > 0 else 0.0


def update_cue_beliefs(state: RunState, deliveries, params: EngineParams) -> list[EffectRecord]:
    """Purpose: each target's belief about the cues its sender sends moves toward what it perceived (`M9.9`).
    Spec:    docs/bowen_agent_model_spec_v2.md#M9.9
    Tests:   tests/bowen/test_social_cues.py::test_m99_sc10_belief_moves_toward_the_perceived_cues

    Applied after the batch, so every reading in it was made against the belief as the week found it. Written only
    from deliveries to the person as target; a witness's reading does not write it. Returns one ``belief`` effect
    record carrying every change, as ``beliefs.update_beliefs`` does (`M16.A.5`).
    """
    changes = []
    for delivery, event in sorted(deliveries, key=lambda pair: (str(pair[1].id), pair[0].recipient.value)):
        if delivery.role is not Role.TARGET or event.mechanism is not Mechanism.MOVE or event.sender is None:
            continue
        found = state.cue_readings.get((event.id, delivery.recipient))
        if found is None:
            continue
        person = state.people[delivery.recipient]
        held = belief(person, event.sender)
        moved = tuple(_clamp(b + params.belief_cue_rate * (p - b)) for b, p in zip(held, found.total))
        if person.cue_beliefs is None:
            person.cue_beliefs = {}
        person.cue_beliefs[event.sender.value] = moved
        changes += [(person.id, f"cue_beliefs[{event.sender.value}].{cue.value}", m - b)
                    for cue, b, m in zip(CUES, held, moved) if m != b]
    return [EffectRecord(state.tick, "belief", None, people=tuple(changes))] if changes else []
