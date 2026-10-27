"""Appraisal beyond the base form: gain, perspective, reappraisal, attention, witness, speaker, calm contact.

Purpose: test each Phase C step 2 appraisal requirement as a direction on the Phase B family.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.2, #M4.C.4, #M4.C.6, #M4.C.7, #M4.C.8, #M4.C.8a, #M4.C.9, #M4.C.10
Tests:   this file

Unit tests of set mechanisms, with invented constants; none is a finding (`M11.5`).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.bowen.engine.appraise import (
    appraisal_delta, apply_appraisal, calm_transfer, content_gain, effective_perspective, speaker_echo, witness_delta,
)
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.events import Attention, Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.state import new_run_state
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, NADIA, PIA = P("ravi"), P("marta"), P("nadia"), P("pia")
RAVI_MARTA = TieId.of(RAVI, MARTA)
PARAMS = engine_params(load_constants())


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, PARAMS)


def move(kind="CONFLICT", intensity=250.0, sender=RAVI, target=MARTA, tick=0, index=0, **extra):
    return Event(
        id=EventId(tick, str(sender.value), index), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
        targets=(target,), intensity=intensity, timestamp=tick, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED, **extra,
    )


def as_target(state, event, recipient=MARTA):
    state.store.record_event(event)
    return Delivery(event.timestamp + 1, event.id, recipient, Role.TARGET, emitted_tick=event.timestamp, latency=1)


def test_m4c2_content_is_defended_against_above_threshold():
    """Above threshold an approach brings no relief: its content is not heard, only its impingement lands."""
    calm, anxious = fresh(), fresh()
    for state in (calm, anxious):  # a neutral sender, so act identity (M5.F.1) does not mask the gain
        state.people[RAVI].outside_ness_outward = state.people[RAVI].outside_ness_inward = 0.0
    anxious.people[MARTA].acute_anxiety += 3 * PARAMS.defence_threshold
    assert content_gain(calm.people[MARTA], PARAMS) == 1.0
    assert content_gain(anxious.people[MARTA], PARAMS) == 0.0
    approach = move("PURSUE", 60.0)
    relief_calm = appraisal_delta(calm, as_target(calm, approach), approach, PARAMS)
    relief_anxious = appraisal_delta(anxious, as_target(anxious, approach), approach, PARAMS)
    assert relief_calm < 0 <= relief_anxious


def test_m4c4_perspective_falls_with_anxiety_and_on_a_loaded_tie():
    state = fresh()
    marta = dataclasses.replace(state.people[MARTA], systems_perspective=0.8)
    tie = state.ties[RAVI_MARTA]
    anxious = dataclasses.replace(marta, acute_anxiety=marta.chronic_anxiety + 20)
    assert effective_perspective(anxious, tie, PARAMS) < effective_perspective(marta, tie, PARAMS)
    tie.felt_impingement[MARTA] = 1.0  # the most loaded tie
    assert effective_perspective(marta, tie, PARAMS) < effective_perspective(marta, None, PARAMS)


def test_m4c6_reappraisal_reattributes_own_output():
    """With perspective, part of what arrives is read as one's own output returning — at once."""
    blind, seeing = fresh(), fresh()
    seeing.people[MARTA].systems_perspective = 0.9
    for state in (blind, seeing):
        state.store.record_event(move("CONFLICT", 250.0, sender=MARTA, target=RAVI, index=1))  # Marta's own push
    reply = move("CONFLICT", 250.0, index=2)
    assert 0 < appraisal_delta(seeing, as_target(seeing, reply), reply, PARAMS) < appraisal_delta(
        blind, as_target(blind, reply), reply, PARAMS
    )


def test_m4c7_witnessed_form_is_less_reactive_for_the_listener():
    """The same statement lands less on Marta overheard than addressed to her (KS18.2)."""
    state = fresh()
    direct = move("CONFLICT", 250.0, target=MARTA, index=1)
    aside = move("CONFLICT", 250.0, target=PIA, index=2)
    state.store.record_event(aside)
    addressed = appraisal_delta(state, as_target(state, direct), direct, PARAMS)
    overheard = witness_delta(state, state.people[MARTA], aside, PARAMS)
    assert 0 < overheard < addressed


def test_m4c7_addressing_a_neutral_third_is_less_reactive_for_the_speaker():
    state = fresh()
    statement = move("CONFLICT", 250.0, target=MARTA)
    delivery = as_target(state, statement)
    direct = speaker_echo(state, delivery, statement, PARAMS)
    state.ties[RAVI_MARTA].conductance = 0.1  # a thin tie, as to a neutral professional
    thin = speaker_echo(state, delivery, statement, PARAMS)
    assert 0 < thin < direct


def test_m4c8_attention_at_feeling_amplifies_and_objectivity_gates_it():
    state = fresh()
    plain_event, feeling, intellect = (move(attention=a, index=i) for i, a in enumerate(Attention))
    base = appraisal_delta(state, as_target(state, plain_event), plain_event, PARAMS)
    amplified = appraisal_delta(state, as_target(state, feeling), feeling, PARAMS)
    ordered = appraisal_delta(state, as_target(state, intellect), intellect, PARAMS)
    assert ordered < base < amplified
    objective = fresh()
    objective.people[MARTA].outside_ness_outward = objective.people[MARTA].outside_ness_inward = 0.0
    gated = appraisal_delta(objective, as_target(objective, feeling), feeling, PARAMS)
    assert gated == base  # M4.C.8a: attention from an objective stance does not amplify


def test_m4c9_witness_appraisal_reads_both_ties():
    """C's appraisal of A→B rises with C's tie to A, B fixed; a copy of B's appraisal would not move."""
    low, high = fresh(), fresh()
    low.ties[TieId.of(NADIA, RAVI)].conductance = high.ties[TieId.of(NADIA, RAVI)].conductance / 2
    exchange = move("CONFLICT", 250.0)
    assert witness_delta(high, high.people[NADIA], exchange, PARAMS) > witness_delta(low, low.people[NADIA], exchange, PARAMS)
    target_low = appraisal_delta(low, as_target(low, exchange), exchange, PARAMS)
    target_high = appraisal_delta(high, as_target(high, exchange), exchange, PARAMS)
    assert target_low == target_high


def test_m4c10_a_calmer_sender_lowers_the_receivers_anxiety():
    state = fresh()
    state.people[MARTA].acute_anxiety += 20
    contact = move("STAY-IN-CONTACT", 30.0)
    delivery = as_target(state, contact)
    assert calm_transfer(state, delivery, contact, PARAMS) > 0
    state.people[RAVI].acute_anxiety += 40  # now the sender is the more anxious
    assert calm_transfer(state, delivery, contact, PARAMS) == 0.0


def test_m4c10_calm_transfer_balances():
    """A transfer, not a sink: what the receiver loses the sender takes (M6.4)."""
    state = fresh()
    state.people[MARTA].acute_anxiety += 20
    contact = move("STAY-IN-CONTACT", 30.0)
    delivery = as_target(state, contact)
    perceived = {MARTA: ((delivery, contact),)}
    records, _, _ = apply_appraisal(state, perceived, PARAMS)
    [calm] = [r for r in records if r.mechanism == "calm_contact"]
    assert abs(sum(v for _, v in calm.acute_anxiety)) < 1e-12 and dict(calm.acute_anxiety)[MARTA] < 0


# --- Version 2 stage 3, group A (spec revision 39, approved 2026-10-25): written first, built at step 9 -------------


def test_m4c1d_s3_10_relieved_receiver_returns_contact():
    """S3.10 (`M4.C.1d`, D22): a target relieved by a delivered move with positive contact (its appraisal lowers its
    anxiety, the whole delivery read, Q11) returns `reciprocal_contact_share` of the contact it took in to the sender's
    felt contact on the same tie, through the sender's own `M4.C.2` gain, so a sender above `defence_threshold` takes in
    none of it; a target at its optimum, whose deviation
    does not fall, returns none. The rule names no kind: the same for `STAY-IN-CONTACT`, `PURSUE` and `TRIANGLE`."""
    from src.bowen.engine.appraise import tie_change

    share = getattr(PARAMS, "reciprocal_contact_share", None)
    assert share is not None, "reciprocal_contact_share is not in the register (M4.C.1d is not built)"
    cases = [(kind, target_contact, relieved, anxious) for kind in ("STAY-IN-CONTACT", "PURSUE", "TRIANGLE")
             for target_contact, relieved in ((0.0, True), (1.0, False)) for anxious in (False, True)]
    for kind, target_contact, relieved, anxious in cases:
        state = fresh()
        tie = state.ties[RAVI_MARTA]
        tie.felt_contact[MARTA], tie.felt_impingement[MARTA] = target_contact, 0.0
        tie.felt_contact[RAVI] = 0.2
        sender = state.people[RAVI]
        sender.acute_anxiety = sender.chronic_anxiety + (3 * PARAMS.defence_threshold if anxious else 0.0)
        sender.outside_ness_outward = sender.outside_ness_inward = 0.0  # a neutral sender (M5.F.1)
        event = move(kind, 60.0)
        delivery = as_target(state, event)
        assert (appraisal_delta(state, delivery, event, PARAMS) < 0) is relieved, (kind, target_contact)
        _tie, d_contact, _imp = tie_change(state, delivery, event, PARAMS)
        assert d_contact > 0, kind
        gain = content_gain(sender, PARAMS)
        assert gain == (0.0 if anxious else 1.0), (kind, anxious, gain)
        expected = share * d_contact * gain if relieved else 0.0
        before = tie.felt_contact[RAVI]
        apply_appraisal(state, {MARTA: ((delivery, event),)}, PARAMS)
        assert tie.felt_contact[RAVI] - before == pytest.approx(expected, abs=1e-12), (kind, target_contact, anxious)
        if relieved and not anxious:
            assert expected > 0, kind


def test_m4c1d_sender_appraises_the_returned_contact_by_their_own_deviation():
    """`M4.C.1d`: the sender appraises the returned contact by the change in their own two-sided deviation — relief on
    the "too little" side of their optimum, nothing within its band or past it (the band is not felt, and the "too
    much" side is impingement, which the returned contact does not carry) — and it is felt contact, not `M1.B.8`'s investment. Each case is set against the
    same delivery with `reciprocal_contact_share` 0, so only the returned contact differs."""
    from src.bowen.engine.contact import optimum

    def run(params, sender_contact, check=None):
        state = fresh()
        tie = state.ties[RAVI_MARTA]
        tie.felt_contact[MARTA], tie.felt_impingement[MARTA] = 0.0, 0.0  # the target is relieved
        sender = state.people[RAVI]
        sender.acute_anxiety = sender.chronic_anxiety
        sender.outside_ness_outward = sender.outside_ness_inward = 0.0
        opt = optimum(sender, tie, PARAMS)
        tie.felt_contact[RAVI], tie.felt_impingement[RAVI] = sender_contact(opt), 0.0
        if check is not None:
            assert check(tie.felt_contact[RAVI], opt), "fixture: the sender's contact is not where the case needs it"
        event = move("STAY-IN-CONTACT", 60.0)
        _, attended, _ = apply_appraisal(state, {MARTA: ((as_target(state, event), event),)}, params)
        return sender.acute_anxiety, attended

    none = dataclasses.replace(PARAMS, reciprocal_contact_share=0.0)
    wanting = lambda opt: 0.25 * opt  # noqa: E731 — well below the optimum: closeness is wanted
    past = lambda opt: min(1.0, opt + 0.3)  # noqa: E731 — past it: more contact is not wanted
    with_share, attended = run(PARAMS, wanting)
    without, attended_none = run(none, wanting)
    assert with_share < without, "returned contact on the 'too little' side must relieve the sender"
    beyond = lambda contact, opt: contact > opt  # noqa: E731
    assert run(PARAMS, past, beyond)[0] == run(none, past, beyond)[0], "returned contact past the optimum must not be felt"
    assert attended == attended_none and (RAVI_MARTA, RAVI) not in attended, "returned contact counted as investment"


def test_m4c1d_relief_reads_the_whole_delivery():
    """`M4.C.1d` (Q11): relief is the target's two-sided deviation, contact and impingement together. A `CONFLICT` that
    eases the target's "too little" side a little while its impingement raises the "too much" side more is not relief,
    and returns nothing, though its contact side alone would read as relief."""
    from src.bowen.engine.appraise import tie_change
    from src.bowen.engine.contact import band, deviation_at, optimum

    state = fresh()
    tie = state.ties[RAVI_MARTA]
    target = state.people[MARTA]
    edge = optimum(target, tie, PARAMS) - band(target, PARAMS)
    tie.felt_contact[MARTA], tie.felt_impingement[MARTA] = edge - 0.02, band(target, PARAMS)
    tie.felt_contact[RAVI] = 0.2
    sender = state.people[RAVI]
    sender.acute_anxiety = sender.chronic_anxiety
    event = move("CONFLICT", 60.0)
    delivery = as_target(state, event)
    _tie, d_contact, d_imp = tie_change(state, delivery, event, PARAMS)
    contact, imp = tie.felt_contact[MARTA], tie.felt_impingement[MARTA]
    contact_only = deviation_at(target, tie, PARAMS, min(1.0, contact + d_contact), imp)
    assert d_contact > 0 and contact_only < deviation_at(target, tie, PARAMS, contact, imp), "fixture: contact side eases"
    assert appraisal_delta(state, delivery, event, PARAMS) > 0, "fixture: the delivery as a whole is not relief"
    before = tie.felt_contact[RAVI]
    apply_appraisal(state, {MARTA: ((delivery, event),)}, PARAMS)
    assert tie.felt_contact[RAVI] == before, "a delivery that raised the target's deviation returned contact"
