"""Attribution, the family's focus, appraisal through belief, and the marital-distance gate (Phase D B8).

Purpose: test spec revision 25 (Claude's recommendation, pending owner review): where each person believes the
         family's difficulty lies, what an anxious family's attention focuses, a witness appraising through belief,
         and the marital-distance gate; and the readout that recovers the sink from belief.
Spec:    docs/bowen_agent_model_spec_v2.md#M9.6, #M9.7, #M1.A.11c, #M5.C.1, #M4.B.2
Tests:   this file
"""

from __future__ import annotations

import dataclasses

import pytest

from src.bowen.engine.appraise import apply_appraisal, believed_tension, marital_distance_high, witness_delta
from src.bowen.engine.attribution import loci, update_attribution, update_focus
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.objects import Sink, SymptomChannel
from src.bowen.engine.state import new_run_state
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.readouts.attribution import sink_from_allocations, sink_from_beliefs
from src.bowen.scenario.params import engine_params

PARAMS = engine_params(load_constants())
RAVI, MARTA, NADIA, PIA, ANA = (PersonId(p) for p in ("ravi", "marta", "nadia", "pia", "ana"))
CHILDREN = frozenset({NADIA, PIA})


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    state = initialise_run(state, PARAMS)
    for person in state.people.values():
        person.acute_anxiety, person.usual_excess = person.chronic_anxiety, 0.0
    return state


def move(kind, sender, *targets, t=0, n=0):
    return Event(id=EventId(t, f"{sender.value}-{kind}", n), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
                 targets=tuple(sorted(targets)), intensity=100.0, timestamp=t, duration=1, exogenous=False,
                 source_position=SourcePosition.NONE, channel=Channel.AUTOMATIC)


def deliver(state, event, witnesses=()):
    """Record ``event`` and return (perceived, batch) as step 3 sees them: its targets and the named witnesses."""
    state.store.record_event(event)
    batch = tuple(sorted(Delivery(event.timestamp, event.id, r, role, event.timestamp, 0)
                         for r, role in [(t, Role.TARGET) for t in event.targets]
                         + [(w, Role.WITNESS) for w in witnesses]))
    perceived = {}
    for d in batch:
        perceived.setdefault(d.recipient, []).append((d, event))
    return {k: tuple(v) for k, v in perceived.items()}, batch


def test_m96_attribution_rules_come_from_config():
    kinds = load_event_kinds()
    assert kinds.attribution["CONFLICT"] == "other" and kinds.attribution["UNDERFUNCTION"] == "sender"
    assert kinds.attribution["OVERFUNCTION"] == "target" and "I-POSITION" not in kinds.attribution
    assert kinds.focus["OVERFUNCTION"] == "physical" and kinds.focus["CONFLICT"] == "social"
    with pytest.raises(ValueError, match="attribution"):
        dataclasses.replace(kinds, attribution={"JOB_LOSS": "other"})


def test_m96_each_side_of_a_conflict_says_the_other():
    conflict = move("CONFLICT", RAVI, MARTA)
    assert loci(conflict, "sender", "other") == {MARTA: 1.0}
    assert loci(conflict, "target", "other") == {RAVI: 1.0}
    assert loci(conflict, "witness", "other") == {RAVI: 0.5, MARTA: 0.5}
    state = fresh()
    perceived, batch = deliver(state, conflict, witnesses=(NADIA,))
    update_attribution(state, perceived, batch, PARAMS)
    assert max(state.people[RAVI].attribution, key=state.people[RAVI].attribution.get) == MARTA
    assert max(state.people[MARTA].attribution, key=state.people[MARTA].attribution.get) == RAVI


def test_m96_giving_way_and_being_taken_care_of_place_it_with_the_same_one_for_both():
    """UNDERFUNCTION places it with its sender, OVERFUNCTION with its target, for every observer: both say the same one."""
    state = fresh()
    for n, event in enumerate((move("UNDERFUNCTION", MARTA, RAVI, n=0), move("OVERFUNCTION", RAVI, MARTA, n=1))):
        perceived, batch = deliver(state, event)
        update_attribution(state, perceived, batch, PARAMS)
    for spouse in (RAVI, MARTA):
        assert state.people[spouse].attribution.get(MARTA, 0) > state.people[spouse].attribution.get(RAVI, 0)


def test_m96_attribution_is_written_only_from_moves_sent_or_delivered():
    """A person who neither sent nor received nor witnessed a move is not moved by it; the weights sum to at most 1;
    and the order of a tick's moves does not matter (M1.F.8)."""
    state = fresh()
    a, b = move("CONFLICT", RAVI, MARTA, n=0), move("OVERFUNCTION", MARTA, NADIA, n=1)
    pa, ba = deliver(state, a)
    pb, bb = deliver(state, b)
    perceived = {k: pa.get(k, ()) + pb.get(k, ()) for k in set(pa) | set(pb)}
    reversed_ = {k: tuple(reversed(v)) for k, v in perceived.items()}
    other = fresh()
    deliver(other, a), deliver(other, b)
    update_attribution(state, perceived, ba + bb, PARAMS)
    update_attribution(other, reversed_, tuple(reversed(ba + bb)), PARAMS)
    assert state.people[PIA].attribution == {} and state.people[ANA].attribution == {}
    for pid in (RAVI, MARTA, NADIA):
        assert state.people[pid].attribution == pytest.approx(other.people[pid].attribution)
        assert sum(state.people[pid].attribution.values()) <= 1.0 + 1e-12


def test_m1a11c_an_anxious_familys_attention_moves_the_focus():
    """A move delivered by a sender above their usual level adds to the target's focus on that act's channel; a calm
    sender's adds nothing; focus decays every week."""
    state = fresh()
    state.people[RAVI].acute_anxiety += 10  # above usual
    perceived, batch = deliver(state, move("OVERFUNCTION", RAVI, NADIA))
    update_focus(state, batch, PARAMS)
    assert state.people[NADIA].channel_focus[SymptomChannel.PHYSICAL] == pytest.approx(PARAMS.focus_rate)
    calm = fresh()
    perceived, batch = deliver(calm, move("OVERFUNCTION", RAVI, NADIA))
    update_focus(calm, batch, PARAMS)
    assert calm.people[NADIA].channel_focus == {}
    update_focus(state, (), PARAMS)
    assert state.people[NADIA].channel_focus[SymptomChannel.PHYSICAL] == pytest.approx(
        PARAMS.focus_rate * (1 - PARAMS.focus_rate))


def test_m97_a_witness_appraises_through_its_belief_about_the_tie():
    """M9.7: a witness who believes the tie tense appraises the same exchange more; the tie's true state is not read."""
    state = fresh()
    event = move("CONFLICT", RAVI, MARTA)
    nadia = state.people[NADIA]
    base = witness_delta(state, nadia, event, PARAMS)
    nadia.tie_beliefs[TieId.of(RAVI, MARTA)].tension = 0.8
    assert believed_tension(nadia, event) == 0.8
    assert witness_delta(state, nadia, event, PARAMS) == pytest.approx(base * (1 + PARAMS.belief_appraisal_gain * 0.8))
    tie = state.ties[TieId.of(RAVI, MARTA)]
    tie.felt_impingement = {m: 1.0 for m in tie.felt_impingement}  # the true state, which must not be read
    assert witness_delta(state, nadia, event, PARAMS) == pytest.approx(base * (1 + PARAMS.belief_appraisal_gain * 0.8))


def test_m5c1_under_marital_distance_no_move_improves_the_symptom_bearer():
    """While both spouses sit below their contact optimum on the marital tie, a move to a member with an active
    symptom cannot lower their anxiety; a calm sender's transfer does not reach them; without the symptom it can."""
    def run(distant, symptomatic):
        state = fresh()
        tie = state.ties[TieId.of(RAVI, MARTA)]
        for m in tie.id.members():
            tie.felt_contact[m] = 0.0 if distant else 1.0
        nadia = state.people[NADIA]
        nadia.acute_anxiety += 20  # anxious, so a calmer sender's contact would relieve her
        nadia.symptom_active[nadia.channel_prior] = symptomatic
        perceived, batch = deliver(state, move("STAY-IN-CONTACT", MARTA, NADIA))
        was = nadia.acute_anxiety
        apply_appraisal(state, perceived, PARAMS)
        return marital_distance_high(state, PARAMS), nadia.acute_anxiety - was

    gated, change = run(distant=True, symptomatic=True)
    assert gated and change >= 0
    # Re-sweep 2026-10-12: a symptomatic spouse is not gated; only the couple's children are.
    state = fresh()
    tie = state.ties[TieId.of(RAVI, MARTA)]
    for m in tie.id.members():
        tie.felt_contact[m] = 0.0
    marta = state.people[MARTA]
    marta.acute_anxiety += 20
    marta.symptom_active[marta.channel_prior] = True
    perceived, _ = deliver(state, move("STAY-IN-CONTACT", NADIA, MARTA))
    was = marta.acute_anxiety
    apply_appraisal(state, perceived, PARAMS)
    assert marta.acute_anxiety < was
    assert run(distant=True, symptomatic=False)[1] < 0
    assert run(distant=False, symptomatic=True)[1] < 0


def test_m96_the_readout_recovers_each_pattern():
    each_other = {RAVI: {MARTA: 0.6, NADIA: 0.1}, MARTA: {RAVI: 0.5}}
    same_one = {RAVI: {MARTA: 0.6}, MARTA: {MARTA: 0.4, RAVI: 0.1}}
    children = {RAVI: {NADIA: 0.4}, MARTA: {PIA: 0.3, RAVI: 0.1}}
    spouses = (RAVI, MARTA)
    assert sink_from_beliefs(each_other, spouses, CHILDREN) is Sink.MARITAL_CONFLICT
    assert sink_from_beliefs(same_one, spouses, CHILDREN) is Sink.SPOUSE_DYSFUNCTION
    assert sink_from_beliefs(children, spouses, CHILDREN) is Sink.CHILD_PROJECTION
    assert sink_from_beliefs({RAVI: {MARTA: 0.5}}, spouses, CHILDREN) is None  # one spouse blames no one yet
    assert sink_from_beliefs({RAVI: {ANA: 0.5}, MARTA: {ANA: 0.5}}, spouses, CHILDREN) is None
    assert sink_from_allocations({Sink.MARITAL_CONFLICT: 3.0, Sink.SPOUSE_DYSFUNCTION: 1.0,
                                  Sink.CHILD_PROJECTION: 0.0}) is Sink.MARITAL_CONFLICT
    assert sink_from_allocations({s: 0.0 for s in Sink}) is None


# --- review of revisions 25-27 (2026-10-12): mutants that survived the tests above ---


def test_m96_a_witness_splits_a_conflict_between_the_two_sides_through_the_update():
    state = fresh()
    perceived, batch = deliver(state, move("CONFLICT", RAVI, MARTA), witnesses=(NADIA,))
    update_attribution(state, perceived, batch, PARAMS)
    nadia = state.people[NADIA].attribution
    assert nadia[RAVI] == pytest.approx(nadia[MARTA]) and nadia[RAVI] > 0


def test_m96_the_step_grows_with_summed_fidelity_and_a_faint_delivery_counts_less():
    """Two observations in a tick move the weights further than one; a delivery at lower fidelity moves them less."""
    def weight(events, fidelity=1.0):
        state = fresh()
        perceived, batch = {}, ()
        for n, e in enumerate(events):
            e = dataclasses.replace(e, fidelity=fidelity)
            p, b = deliver(state, e)
            for k, v in p.items():
                perceived[k] = perceived.get(k, ()) + v
            batch += b
        update_attribution(state, perceived, batch, PARAMS)
        return state.people[MARTA].attribution.get(RAVI, 0.0)

    one = weight([move("CONFLICT", RAVI, MARTA)])
    two = weight([move("CONFLICT", RAVI, MARTA, n=0), move("CONFLICT", RAVI, MARTA, n=1)])
    assert one == pytest.approx(PARAMS.attribution_rate) and two == pytest.approx(2 * PARAMS.attribution_rate)
    assert weight([move("CONFLICT", RAVI, MARTA)], fidelity=0.5) < one


def test_m96_the_dead_hold_no_new_attribution():
    state = fresh()
    state.people[MARTA].alive = False
    perceived, batch = deliver(state, move("CONFLICT", RAVI, MARTA))
    update_attribution(state, perceived, batch, PARAMS)
    assert state.people[MARTA].attribution == {}


def test_m1a11c_only_the_target_of_an_anxious_act_is_focused_on():
    """A witness of an anxious act gains no focus; only its target."""
    state = fresh()
    state.people[RAVI].acute_anxiety += 10
    perceived, batch = deliver(state, move("OVERFUNCTION", RAVI, NADIA), witnesses=(PIA,))
    update_focus(state, batch, PARAMS)
    assert state.people[PIA].channel_focus == {} and state.people[NADIA].channel_focus


def test_m97_belief_makes_a_witnessed_load_worse_and_leaves_a_relief_alone():
    """Review 2026-10-12: believed tension scales a witnessed load, not a witnessed relief."""
    state = fresh()
    event = move("CONFLICT", RAVI, MARTA)
    nadia = state.people[NADIA]
    base_load, base_relief = witness_delta(state, nadia, event, PARAMS, 1), witness_delta(state, nadia, event, PARAMS, -1)
    nadia.tie_beliefs[TieId.of(RAVI, MARTA)].tension = 0.8
    assert witness_delta(state, nadia, event, PARAMS, 1) > base_load
    assert witness_delta(state, nadia, event, PARAMS, -1) == pytest.approx(base_relief)


def test_m96_a_tie_at_the_top_reads_as_no_pattern():
    assert sink_from_beliefs({RAVI: {MARTA: 0.4, NADIA: 0.4}, MARTA: {RAVI: 0.5}}, (RAVI, MARTA), CHILDREN) is None
