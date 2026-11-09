"""Unit tests for the social cues and signal channels (spec revision 55, T-202).

Purpose: each `SC-` requirement of `docs/proposal_social_cues.md`, as `docs/spec_revision_55_draft.md` states it,
         tested at the unit level; the ensemble criteria `M11.C.50`–`M11.C.55` are in
         `tests/bowen/test_criteria_sc.py`.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.1c, #M4.C.1e, #M4.C.1f, #M4.C.11, #M4.C.12, #M4.C.13, #M4.C.14,
         #M4.C.15, #M9.9, #M4.C.6, #M4.C.9, #M1.F.4
Tests:   this file
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

from src.bowen.engine import cues
from src.bowen.engine.act import Selection, act
from src.bowen.engine.appraise import apply_appraisal, tie_change
from src.bowen.engine.cues import Cue, SignalChannel, Signals
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.log_records import EmittedRecord, serialize
from src.bowen.engine.state import ActiveStressor, new_run_state
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, NADIA = P("ravi"), P("marta"), P("nadia")
RAVI_MARTA = TieId.of(RAVI, MARTA)
OFF = engine_params(load_constants())
ON = dataclasses.replace(OFF, social_cues_enabled=1)
# The reduction's conditions (M4.C.1f): nothing on the nonverbal channels, no meaning error, full detection.
NEUTRAL = dataclasses.replace(
    ON, cue_leak_gain=0.0, belief_leak_gain=0.0, attention_load_gain=0.0,
    cue_bias_floor=0.0, cue_bias_level_gain=0.0, cue_bias_anxiety_gain=0.0, cue_bias_belief_gain=0.0,
    cue_spread_floor=0.0, cue_spread_level_gain=0.0, cue_sensitivity_floor=1.0, cue_sensitivity_level_gain=0.0,
)
# The cues on, with the reading exact (no meaning error, full detection) but the sender's leak left as configured.
NEUTRAL_READ = dataclasses.replace(
    ON, cue_bias_floor=0.0, cue_bias_level_gain=0.0, cue_bias_anxiety_gain=0.0, cue_bias_belief_gain=0.0,
    cue_spread_floor=0.0, cue_spread_level_gain=0.0, cue_sensitivity_floor=1.0, cue_sensitivity_level_gain=0.0,
)
VIS = HouseholdConductanceVisibility(OFF.per_hop_fidelity)


def fresh(params=OFF):
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, params)


def emitted(records):
    return [r.event for r in records if isinstance(r, EmittedRecord)]


def deliver(state, event, recipient=MARTA, role=Role.TARGET):
    return Delivery(event.timestamp + 1, event.id, recipient, role, emitted_tick=event.timestamp, latency=1)


def send(state, params, kind="CONFLICT", sender=RAVI, target=MARTA, intensity=250.0, index=0):
    events = emitted(act(state, Selection(sender, kind, (target,), intensity, index=index), VIS, params))
    assert len(events) == 1
    return events[0]


# --- SC-3: the four signal channels (M1.F.1c) --------------------------------------------------------------------


def test_m1f1_sc3_signals_absent_and_unlogged_when_off():
    state = fresh()
    event = send(state, OFF)
    assert event.signals is None
    assert '"signals"' not in serialize(EmittedRecord(event))


def test_m1f1c_sc3_move_carries_four_channels_when_on():
    state = fresh(ON)
    event = send(state, ON)
    assert isinstance(event.signals, Signals)
    for channel in SignalChannel:
        vector = event.signals.channel(channel)
        assert len(vector) == len(Cue) and all(-1.0 <= v <= 1.0 for v in vector)
    assert event.signals.channel(SignalChannel.WORDS) == (0.0,) * len(Cue)  # words carry no cue (plan D1)
    assert '"signals"' in serialize(EmittedRecord(event))


def test_m1f1_sc3_move_without_signals_is_refused_when_on():
    state = fresh(ON)
    event = send(state, OFF)  # stamped by an engine with the cues off: no signals
    batch = ((deliver(state, event), event),)
    with pytest.raises(ValueError, match="signals"):
        apply_appraisal(state, {MARTA: batch}, ON)


def test_m1f1c_sc3_signals_reject_values_outside_the_unit_interval():
    with pytest.raises(ValueError):
        Signals(tone=(0.0, 0.0, 1.5, 0.0))
    with pytest.raises(ValueError):
        Signals(face=(0.0, 0.0))


# --- SC-2: the reduction, and the switch (M4.C.1f) ----------------------------------------------------------------


def _felt(state):
    return {(t, m): (tie.felt_contact[m], tie.felt_impingement[m])
            for t, tie in sorted(state.ties.items()) for m in t.members()}


@pytest.mark.parametrize("kind", sorted(k for k in load_event_kinds().mechanisms
                                        if load_event_kinds().mechanisms[k] is Mechanism.MOVE
                                        and k not in ("I-POSITION", "WITHHOLD")))
def test_m4c1f_sc2_reduction_with_equal_channels(kind):
    """With the cues on, nothing on tone, face and posture, no meaning error and full detection, every move kind's
    felt change on the tie equals its change with the cues off."""
    results = []
    for params in (OFF, NEUTRAL):
        state = fresh(params)
        event = send(state, params, kind=kind)
        before = _felt(state)
        apply_appraisal(state, {MARTA: ((deliver(state, event), event),)}, params)
        after = _felt(state)
        results.append({k: (after[k][0] - before[k][0], after[k][1] - before[k][1]) for k in after})
    assert results[0] == results[1]


def test_m4c1f_sc2_no_cue_draw_when_off():
    state = fresh(OFF)
    event = send(state, OFF)
    drawn = []
    original = state.draws.uniform

    def spy(key):
        drawn.append(key.draw_class)
        return original(key)

    state.draws.uniform = spy
    try:
        apply_appraisal(state, {MARTA: ((deliver(state, event), event),)}, OFF)
    finally:
        del state.draws.uniform
    assert "cue_perception" not in drawn


# --- helpers for the reading ------------------------------------------------------------------------------------


def signed(index=0, *, words=None, tone=None, face=None, posture=None, kind="CONFLICT", sender=RAVI, target=MARTA,
           route=(), tick=0, intensity=250.0, witnesses=()):
    """A move with the given signal channels, built directly, as ``test_appraise.move`` builds one."""
    zero = (0.0,) * len(Cue)
    signals = Signals(words=words or zero, tone=tone or zero, face=face or zero, posture=posture or zero)
    return Event(
        id=EventId(tick, str(sender.value), index), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
        targets=(target,), intensity=intensity, timestamp=tick, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED, route=route, signals=signals,
        witnesses=witnesses, witnesses_computed=bool(witnesses),
    )


def cue(**values):
    """A cue vector from named cues, the rest neutral."""
    return tuple(values.get(c.value, 0.0) for c in Cue)


def read(state, event, params, recipient=MARTA, role=Role.TARGET):
    return cues.reading(state, deliver(state, event, recipient, role), event, params)


def at(state, person, *, level=None, excess=None):
    p = state.people[person]
    if level is not None:
        p.functional_level = level
    if excess is not None:
        p.acute_anxiety = p.chronic_anxiety + excess
    return p


# --- SC-1: the cue vector moves the two drives (M4.C.1e) ----------------------------------------------------------


def test_m4c1e_sc1_disapproval_raises_inward_impingement():
    """One delivery, the perceived approval cue at its threat pole against neutral, no perception error: the
    receiver's felt impingement rises more and its felt contact falls."""
    state = fresh(NEUTRAL)
    _, c_neutral, i_neutral = tie_change(state, deliver(state, signed(0)), signed(0), NEUTRAL)
    threat = cue(approval=-1.0)
    disapproving = signed(1, tone=threat, face=threat, posture=threat)
    _, c_threat, i_threat = tie_change(state, deliver(state, disapproving), disapproving, NEUTRAL)
    assert i_threat > i_neutral
    assert c_threat < c_neutral


def test_m4c1e_sc1_threat_pole_lowers_contact_and_raises_impingement():
    for name in ("attention", "approval", "expectations", "distress"):
        contact, impingement = cues.cue_components(cue(**{name: -1.0}), ON)
        back_c, back_i = cues.cue_components(cue(**{name: 1.0}), ON)
        assert contact <= 0.0 <= impingement and (contact, impingement) != (0.0, 0.0), name
        assert (back_c, back_i) == (-contact, -impingement), name


def test_m4c1e_sc1_cues_only_move_contact_and_impingement():
    """The cue reading writes no person field but acute anxiety and the M9.9 belief, and no policy module reads the
    belief: the cues are readings, not drives (M4.C.1e)."""
    import pathlib

    state = fresh(ON)
    event = send(state, ON)
    snapshot = {pid: copy.deepcopy(vars(p)) for pid, p in state.people.items()}
    apply_appraisal(state, {MARTA: ((deliver(state, event), event),)}, ON)
    changed = {name for pid, p in state.people.items() for name, value in vars(p).items()
               if snapshot[pid][name] != value}
    assert changed <= {"acute_anxiety", "cue_beliefs"}
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "bowen"
    readers = [f for f in sorted((root / "policy").rglob("*.py")) if "cue_beliefs" in f.read_text()]
    assert readers == []


def test_m4c1e_sc1_an_unmet_expectation_reads_as_too_little():
    """A received attention cue below what the receiver believes the partner sends lowers felt contact."""
    state = fresh(NEUTRAL)
    event = signed(0)
    _, c_none, _ = tie_change(state, deliver(state, event), event, NEUTRAL)
    state = fresh(NEUTRAL)
    state.people[MARTA].cue_beliefs = {RAVI.value: cue(attention=0.6)}
    _, c_held, _ = tie_change(state, deliver(state, event), event, NEUTRAL)
    assert c_held < c_none


# --- SC-3, Q9: a hop carries the words only; a witness reads all four (M1.F.4, M4.C.9) ----------------------------


def test_m4c9_sc3_private_hop_carries_words_only():
    threat = cue(expectations=-0.8, distress=-0.8)
    state = fresh(NEUTRAL)
    direct = read(state, signed(0, tone=threat, face=threat, posture=threat), NEUTRAL)
    hopped = read(state, signed(1, tone=threat, face=threat, posture=threat, route=(NADIA,)), NEUTRAL)
    assert direct.nonverbal != cues.NEUTRAL_VECTOR
    assert hopped.nonverbal == cues.NEUTRAL_VECTOR


def test_m4c9_sc3_witness_reads_all_four():
    threat = cue(expectations=-0.8, distress=-0.8)
    state = fresh(NEUTRAL)
    event = signed(0, tone=threat, face=threat, posture=threat, witnesses=(NADIA,))
    seen = read(state, event, NEUTRAL, recipient=NADIA, role=Role.WITNESS)
    assert seen.nonverbal != cues.NEUTRAL_VECTOR
    assert cues.witness_factor(seen.total, NEUTRAL) > 1.0


# --- SC-4 (M4.C.13): tone, face and posture leak --------------------------------------------------------------------


def test_m4c13_sc4_anxious_sender_leak_raises_receiver_impingement():
    """The same move with the same words, the sender's acute anxiety high against low: the receiver feels more
    impinged on by the anxious sender's move."""
    results = []
    for excess in (0.0, 2 * ON.defence_threshold):
        state = fresh(NEUTRAL_READ)
        at(state, RAVI, excess=excess)
        event = send(state, NEUTRAL_READ)
        _, _, d_imp = tie_change(state, deliver(state, event), event, NEUTRAL_READ)
        results.append(d_imp)
    calm, anxious = results
    assert anxious > calm


def test_m4c13_sc4_leak_grows_with_the_senders_excess():
    state = fresh(ON)
    leaks = []
    for excess in (0.0, 0.5 * ON.defence_threshold, ON.defence_threshold, 3 * ON.defence_threshold):
        at(state, RAVI, excess=excess)
        signals = cues.sender_signals(state, RAVI, (MARTA,), ON)
        leaks.append(-signals.tone[2] - signals.tone[3])
        assert signals.words == cues.NEUTRAL_VECTOR
    assert leaks[0] == 0.0 < leaks[1] < leaks[2] == leaks[3]  # saturates at defence_threshold


# --- SC-5 (M4.C.11): anxiety raises the weight on the nonverbal channels -------------------------------------------


def test_m4c11_sc5_nonverbal_share_rises_with_receiver_anxiety():
    state = fresh(ON)
    weights = [cues.nonverbal_weight(at(state, MARTA, excess=e), ON)
               for e in (0.0, 0.5 * ON.defence_threshold, 5 * ON.defence_threshold)]
    assert weights[0] < weights[1] < weights[2] <= ON.nonverbal_weight_cap < 1.0
    threat = cue(distress=-0.8)
    shares = []
    for excess in (0.0, ON.defence_threshold):
        state = fresh(NEUTRAL)
        at(state, MARTA, excess=excess)
        r = read(state, signed(0, tone=threat, face=threat, posture=threat), NEUTRAL)
        shares.append(-r.nonverbal[3])
    assert shares[1] > shares[0]


# --- SC-6 (M4.C.12): real or imagined, on meaning only -------------------------------------------------------------

N_DRAWS = 2000


def _mean_read(level, params=ON, n=N_DRAWS, belief=None, which=1):
    state = fresh(params)
    at(state, MARTA, level=level, excess=0.0)
    if belief is not None:
        state.people[MARTA].cue_beliefs = {RAVI.value: belief}
    total = 0.0
    for i in range(n):
        total += read(state, signed(i), params).total[which]
    return total / n


def test_m4c12_sc6_neutral_cue_read_as_threat_more_at_low_level():
    """Every sender neutral: the approval cue is read toward disapproval, and more so at a low level."""
    high, low = _mean_read(80.0), _mean_read(20.0)
    assert high < 0.0
    assert low < high


def test_m4c12_sc6_a_believed_threat_is_read_into_neutral_moves():
    assert _mean_read(50.0, belief=cue(approval=-0.8)) < _mean_read(50.0)


def test_m4c12_sc6_presence_is_never_imagined():
    """A channel that is not present (after a hop) or not detected reads as nothing, at every level, anxiety and
    belief: the error is on meaning, never on presence."""
    blind = dataclasses.replace(ON, cue_sensitivity_floor=0.0, cue_sensitivity_level_gain=0.0)
    for level in (0.0, 30.0, 70.0, 100.0):
        for params, route in ((ON, (NADIA,)), (blind, ())):
            state = fresh(params)
            at(state, MARTA, level=level, excess=3 * ON.defence_threshold)
            state.people[MARTA].cue_beliefs = {RAVI.value: cue(approval=-1.0, distress=-1.0)}
            for i in range(50):
                assert read(state, signed(i, route=route), params).nonverbal == cues.NEUTRAL_VECTOR


# --- SC-7 (M4.C.14, M4.C.1a): level changes sensitivity and meaning error by degree --------------------------------


def test_m4c14_sc7_no_term_reaches_zero_at_the_top_level():
    from src.bowen.engine.contact import steepness

    state = fresh(ON)
    top = at(state, MARTA, level=100.0, excess=0.0)
    assert cues.sensitivity(top, ON) > 0.0
    assert min(cues.meaning_bias(top, cues.NEUTRAL_VECTOR, ON)) > 0.0
    assert cues.meaning_spread(top, ON) > 0.0
    assert steepness(top, ON) > 0.0


def _detected(level, n=N_DRAWS):
    tone_only = dataclasses.replace(ON, cue_face_share=0.0, cue_posture_share=0.0)
    state = fresh(tone_only)
    at(state, MARTA, level=level, excess=0.0)
    small = cue(distress=-0.05)
    return sum(read(state, signed(i, tone=small), tone_only).nonverbal != cues.NEUTRAL_VECTOR for i in range(n))


def test_m4c14_sc7_lower_level_detects_a_small_leak_more_often():
    assert _detected(20.0) > _detected(80.0)


def test_m4c14_sc7_lower_level_reads_threat_into_neutral_more_often():
    def threatened(level):
        state = fresh(ON)
        at(state, MARTA, level=level, excess=0.0)
        # reported words only (a hop), so detection, which level also moves, plays no part (M4.C.12 alone)
        return sum(read(state, signed(i, route=(NADIA,)), ON).total[3] < -0.05 for i in range(N_DRAWS))

    assert threatened(20.0) > threatened(80.0)


# --- SC-8 (M4.C.6): self-attribution by channel ----------------------------------------------------------------------


def test_m4c6_sc8_own_leak_share_reads_the_nonverbal_channels():
    state = fresh(ON)
    leak = cue(expectations=-0.5)
    mine = signed(0, sender=MARTA, target=RAVI, tone=leak)
    theirs = signed(1, sender=RAVI, target=MARTA)  # words only: no leak
    state.store.record_event(mine)
    state.store.record_event(theirs)
    state.tick = 1
    assert cues.own_leak_share(state, MARTA, RAVI, ON.reappraisal_window) == 1.0
    assert cues.own_leak_share(state, RAVI, MARTA, ON.reappraisal_window) == 0.0


def test_m4c6_sc8_attribution_reads_the_nonverbal_part_only():
    """A reappraiser whose own leak made the tie's recent leak takes less from the other's nonverbal threat; the
    words part of the same move lands in full."""
    from src.bowen.engine.appraise import appraisal_delta

    def delta(perspective, nonverbal):
        state = fresh(NEUTRAL)
        state.people[MARTA].systems_perspective = perspective
        state.store.record_event(signed(0, sender=MARTA, target=RAVI, tone=cue(expectations=-0.5)))
        state.tick = 1
        threat = cue(expectations=-0.8, distress=-0.8) if nonverbal else None
        event = signed(5, tick=1, tone=threat, face=threat, posture=threat, intensity=20.0)  # below the clamp
        return appraisal_delta(state, deliver(state, event), event, NEUTRAL)

    assert delta(1.0, nonverbal=True) < delta(0.0, nonverbal=True)
    assert delta(1.0, nonverbal=False) == delta(0.0, nonverbal=False)


# --- SC-9 (M4.C.15, a proxy): attention is finite ---------------------------------------------------------------------


def test_m4c15_sc9_stressor_load_lowers_the_attention_cue():
    state = fresh(ON)
    at(state, RAVI, excess=0.0)
    free = cues.sender_signals(state, RAVI, (MARTA,), ON)
    state.stressors.append(ActiveStressor(EventId(0, "script", 99), RAVI, 47.0, 0, 20))
    loaded = cues.sender_signals(state, RAVI, (MARTA,), ON)
    assert free.tone[0] == 0.0 > loaded.tone[0]
    assert loaded.words == cues.NEUTRAL_VECTOR


# --- SC-10 (M9.9): expectations are beliefs ----------------------------------------------------------------------------


def test_m99_sc10_belief_moves_toward_the_perceived_cues():
    state = fresh(ON)
    threat = cue(approval=-0.9)
    event = signed(0, tone=threat, face=threat, posture=threat)
    apply_appraisal(state, {MARTA: ((deliver(state, event), event),)}, ON)
    held = state.people[MARTA].cue_beliefs[RAVI.value]
    assert -1.0 < held[1] < 0.0
    assert state.people[RAVI].cue_beliefs is None  # written only at the receiver


def test_m99_sc10_believed_distress_is_sent_as_leak():
    state = fresh(ON)
    at(state, RAVI, excess=0.0)
    neutral = cues.sender_signals(state, RAVI, (MARTA,), ON)
    state.people[RAVI].cue_beliefs = {MARTA.value: cue(distress=-0.8)}
    believing = cues.sender_signals(state, RAVI, (MARTA,), ON)
    assert believing.tone[2] < neutral.tone[2] and believing.tone[3] < neutral.tone[3]
    assert cues.sender_signals(state, RAVI, (NADIA,), ON).tone == neutral.tone  # the belief is about Marta only
