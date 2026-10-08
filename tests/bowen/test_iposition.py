"""The I-POSITION state machine, the assertion form and the success state (Phase C step 8).

Purpose: test each M5.D, M5.E and M5.F.4 requirement as a direction, driving a sequence
         by hand where a unit test can, and over an ensemble where the requirement is
         about what usually happens.
Spec:    docs/bowen_agent_model_spec_v2.md#M5.D.1, #M5.D.2, #M5.D.2a, #M5.D.3, #M5.D.4, #M5.D.4a, #M5.D.5, #M5.D.6, #M5.D.7, #M5.D.7a, #M5.D.8, #M5.D.9, #M5.E.3, #M5.E.7, #M5.E.8, #M5.F.2a, #M5.F.4, #M1.C.5
Tests:   this file

Unit tests of set mechanisms with invented constants; none is a finding (`M11.5`).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from functools import lru_cache
from pathlib import Path

from src.bowen.engine import iposition as ip
from src.bowen.engine.act import Selection, act
from src.bowen.engine.appraise import appraisal_delta
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.log_records import EffectRecord, EmittedRecord
from src.bowen.engine.moves import DEFAULT_AREA, apply_move_effects
from src.bowen.engine.objects import TieState
from src.bowen.engine.outside_ness import axes
from src.bowen.engine.state import new_run_state
from src.bowen.engine.tick import run_tick
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_policy_rules, load_script
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.params import engine_params
from src.bowen.scenario.policy_source import PolicySource
from src.bowen.scenario.scripted_source import ScriptedSource

P = PersonId
RAVI, MARTA, NADIA = P("ravi"), P("marta"), P("nadia")
FAMILY_TRIAD = TriangleId.of(MARTA, NADIA, RAVI)
PARAMS = engine_params(load_constants())
KINDS = load_event_kinds()
VIS = HouseholdConductanceVisibility(PARAMS.per_hop_fidelity)
REPO = Path(__file__).resolve().parents[2]


def fresh(perspective=1.0):
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, KINDS)
    initialise_run(state, PARAMS)
    state.people[RAVI].systems_perspective = perspective
    return state


def calm(state, *people):
    for pid in people:
        state.people[pid].acute_anxiety = state.people[pid].chronic_anxiety


def choose_iposition(state, target=MARTA):
    return act(state, Selection(actor=RAVI, kind="I-POSITION", targets=(target,), intensity=100.0), VIS, PARAMS)


def deliver_to(state, recipient, sender, kind="CONFLICT", index=0):
    """A move from ``sender`` delivered to ``recipient`` this tick."""
    event = Event(
        id=EventId(state.tick - 1, str(sender.value), 50 + index), kind=kind, mechanism=Mechanism.MOVE,
        sender=sender, targets=(recipient,), intensity=100.0, timestamp=state.tick - 1, duration=1,
        exogenous=False, source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
    )
    state.store.record_event(event)
    state.store.record_delivery(Delivery(state.tick, event.id, recipient, Role.TARGET, emitted_tick=state.tick - 1, latency=1))


def week(state):
    state.tick += 1
    return ip.advance_sequences(state, PARAMS)


def take_owed_step(state):
    owed, _ = ip.sequence_selections(state, PARAMS)
    [step] = owed
    return act(state, step, VIS, PARAMS)


def seq(state):
    return state.people[RAVI].iposition_state


def prepared_to_opposition(state):
    """Run PREPARE to its end, calmly, and make the DEFINE step."""
    choose_iposition(state)
    for _ in range(PARAMS.prepare_ticks):
        calm(state, RAVI)
        week(state)
    assert seq(state).due == ip.DEFINE
    take_owed_step(state)
    assert seq(state).stage == ip.OPPOSITION


def opposition_held(state):
    prepared_to_opposition(state)
    calm(state, RAVI)
    week(state)
    deliver_to(state, RAVI, MARTA)
    ip.advance_sequences(state, PARAMS)
    assert seq(state).stage == ip.HOLD


# --- starting: a sequence, or the assertion form -------------------------------------------


def test_m5d1_iposition_is_not_a_single_tick_move():
    state = fresh()
    records = choose_iposition(state)
    assert not any(isinstance(r, EmittedRecord) for r in records)
    assert seq(state).stage == ip.PREPARE


def test_m5f4_low_perspective_executes_the_assertion_form():
    """Without perspective the move is a claim: it raises reactivity and counts against the claimant."""
    state = fresh(perspective=0.0)
    [event] = [r.event for r in choose_iposition(state) if isinstance(r, EmittedRecord)]
    assert event.assertion and seq(state) is None
    genuine = dataclasses.replace(event, assertion=False)
    delivery = Delivery(1, event.id, MARTA, Role.TARGET, emitted_tick=0, latency=1)
    assert appraisal_delta(state, delivery, event, PARAMS) > appraisal_delta(state, delivery, genuine, PARAMS)
    before = state.people[RAVI].outside_ness_outward
    apply_move_effects(state, (delivery,), PARAMS)
    assert state.people[RAVI].outside_ness_outward > before  # M5.F.2a: claiming is negative evidence


def test_m5d4a_an_angry_mover_executes_the_assertion_form():
    state = fresh(perspective=1.0)
    state.ties[TieId.of(RAVI, MARTA)].felt_impingement[RAVI] = 1.0
    assert ip.angry(state, RAVI, MARTA, PARAMS)
    [event] = [r.event for r in choose_iposition(state) if isinstance(r, EmittedRecord)]
    assert event.assertion


# --- PREPARE -------------------------------------------------------------------------------


def test_m5d2a_prepare_occupies_several_weeks_and_can_fail():
    state = fresh()
    choose_iposition(state)
    for n in range(PARAMS.prepare_ticks - 1):
        calm(state, RAVI)
        week(state)
        assert seq(state).due is None  # still preparing
    calm(state, RAVI)
    week(state)
    assert seq(state).due == ip.DEFINE and PARAMS.prepare_ticks > 1
    failing = fresh()
    choose_iposition(failing)
    failing.people[RAVI].acute_anxiety = failing.people[RAVI].chronic_anxiety + 2 * PARAMS.defence_threshold
    week(failing)
    assert seq(failing) is None


def test_m5d2a_unprepared_sequence_holds_with_less_capacity():
    prepared, rushed = fresh(), fresh()
    for state in (prepared, rushed):
        choose_iposition(state)
    for _ in range(PARAMS.prepare_ticks):
        calm(prepared, RAVI)
        week(prepared)
    calm(rushed, RAVI)
    rushed.tick += 1
    deliver_to(rushed, RAVI, MARTA)  # the issue comes up in the first week
    ip.advance_sequences(rushed, PARAMS)
    assert seq(rushed).due == ip.DEFINE and seq(rushed).prepared == 0
    assert ip.capacity(rushed.people[RAVI], PARAMS) < ip.capacity(prepared.people[RAVI], PARAMS)


# --- the opposition ------------------------------------------------------------------------


def test_m5e7_genuine_iposition_withdraws_contact_from_the_other():
    state = fresh()
    choose_iposition(state)
    for _ in range(PARAMS.prepare_ticks):
        calm(state, RAVI)
        week(state)
    [event] = [r.event for r in take_owed_step(state) if isinstance(r, EmittedRecord)]
    assert not event.assertion
    tie = state.ties[TieId.of(RAVI, MARTA)]
    before = tie.felt_contact[MARTA]
    apply_move_effects(state, (Delivery(state.tick + 1, event.id, MARTA, Role.TARGET,
                                        emitted_tick=state.tick, latency=1),), PARAMS)
    assert tie.felt_contact[MARTA] < before


def test_m5e3_no_reaction_means_the_move_did_not_land():
    state = fresh()
    prepared_to_opposition(state)
    level = state.people[RAVI].functional_level
    for _ in range(PARAMS.opposition_window):
        calm(state, RAVI)
        records = week(state)
    assert seq(state) is None and any("did_not_land" in f for r in records for _, f, _ in r.people)
    assert state.people[RAVI].functional_level == level  # not a success


def test_m5d3_abort_returns_the_mover_to_the_prior_balance():
    state = fresh()
    choose_iposition(state)
    start_balance, start_axes = seq(state).start_balance, seq(state).start_axes
    for _ in range(PARAMS.prepare_ticks):
        calm(state, RAVI)
        week(state)
    take_owed_step(state)
    assert axes(state.people[RAVI]) != start_axes  # rehearsal moved them
    state.tick += 1
    deliver_to(state, RAVI, MARTA)
    state.people[RAVI].acute_anxiety = state.people[RAVI].chronic_anxiety + 99.0
    ip.advance_sequences(state, PARAMS)
    assert seq(state).due == ip.ABORT
    [event] = [r.event for r in take_owed_step(state) if isinstance(r, EmittedRecord)]
    assert event.kind in ip.ABORT_ACTS and seq(state) is None
    assert axes(state.people[RAVI]) == start_axes
    assert state.ties[TieId.of(RAVI, MARTA)].functioning_balance[DEFAULT_AREA] == start_balance


def test_m5d4_an_angry_mover_stalls_rather_than_aborts():
    state = fresh()
    opposition_held(state)
    state.ties[TieId.of(RAVI, MARTA)].felt_impingement[RAVI] = 1.0  # angry on the tie
    ends = []
    for n in range(PARAMS.stall_limit):
        records = week(state)
        deliver_to(state, RAVI, MARTA, index=n + 1)
        ends += [f for r in records for _, f, _ in r.people if "end" in f]
        if seq(state) is not None:
            assert seq(state).stage == ip.HOLD and seq(state).due is None  # no peak, no abort
    assert seq(state) is None and ends == ["iposition:end:stalled"]


def drive_to_resolve(state):
    opposition_held(state)
    calm(state, RAVI)
    state.ties[TieId.of(RAVI, MARTA)].felt_impingement[RAVI] = 0.0
    week(state)
    deliver_to(state, RAVI, MARTA, index=1)  # the final attack, met calmly
    ip.advance_sequences(state, PARAMS)
    assert seq(state).stage == ip.RESOLVE


def test_m5d2_states_run_in_order():
    state = fresh()
    state.people[RAVI].functional_level = 60.0
    drive_to_resolve(state)
    history = list(seq(state).history)
    assert history == [s for s in ip.STATES if s in history]
    assert history == [ip.PREPARE, ip.OPPOSITION, ip.HOLD, ip.PEAK, ip.RESOLVE]


def test_m5d5_opposition_pulls_up_to_the_movers_level():
    state = fresh()
    state.people[RAVI].functional_level = 60.0
    marta_before = state.people[MARTA].functional_level
    drive_to_resolve(state)
    gap = 60.0 - marta_before
    assert state.people[MARTA].functional_level == marta_before + PARAMS.pull_up_rate * gap
    mean = sum(p.functional_level for p in state.people.values()) / len(state.people)
    assert state.people[RAVI].functional_level > mean  # pulled toward the mover, not a mean


def test_m5d6_follow_up_is_due_the_week_after_resolve_and_skipping_it_reverts():
    state = fresh()
    state.people[RAVI].functional_level = 60.0
    drive_to_resolve(state)
    resolved_at, marta_raised = state.tick, state.people[MARTA].functional_level
    week(state)
    assert seq(state).due == ip.FOLLOW_UP and state.tick == resolved_at + 1
    tie = state.ties[TieId.of(RAVI, MARTA)]
    tie.interactive, tie.tie_state = False, TieState.CUT_OFF  # the follow-up cannot be made
    _, records = ip.sequence_selections(state, PARAMS)
    assert seq(state) is None and state.people[MARTA].functional_level < marta_raised


def complete_one(state):
    state.people[RAVI].functional_level = 60.0
    drive_to_resolve(state)
    week(state)
    return take_owed_step(state)


def test_m5d7_completed_exchange_raises_level_and_lowers_the_triangle():
    state = fresh()
    basic = state.people[RAVI].basic_level
    before_floor = state.triangles[FAMILY_TRIAD].intensity_floor
    complete_one(state)
    assert seq(state) is None
    assert state.people[RAVI].functional_level == 60.0 + PARAMS.exchange_gain
    assert state.people[RAVI].basic_level == basic  # M5.D.7: never written
    assert state.triangles[FAMILY_TRIAD].intensity_floor > before_floor


def test_m1c5_the_triangle_decrement_is_permanent_and_lowers_routing():
    from src.bowen.engine.moves import triangle_transfer

    def routed(state):
        for pid in (RAVI, MARTA):  # the tense pair; Nadia is recruited (decided 2026-10-08)
            state.people[pid].acute_anxiety = state.people[pid].chronic_anxiety + 20
        nadia = state.people[NADIA].acute_anxiety
        event = Event(id=EventId(state.tick, "ravi", 9), kind="TRIANGLE", mechanism=Mechanism.MOVE, sender=RAVI,
                      targets=(NADIA,), intensity=100.0, timestamp=state.tick, duration=1, exogenous=False,
                      source_position=SourcePosition.NONE, channel=Channel.SCRIPTED)
        triangle_transfer(state, event, NADIA, PARAMS)
        return state.people[NADIA].acute_anxiety - nadia

    plain, decremented = fresh(), fresh()
    complete_one(decremented)
    floor = decremented.triangles[FAMILY_TRIAD].intensity_floor
    for pid in (RAVI, MARTA, NADIA):
        decremented.people[pid].functional_level = plain.people[pid].functional_level
    assert routed(decremented) < routed(plain)
    for _ in range(50):
        week(decremented)
    assert decremented.triangles[FAMILY_TRIAD].intensity_floor == floor  # nothing reverts it


def test_m5e8_completion_leaves_the_tie_more_solid():
    state = fresh()
    start = {p: axes(state.people[p]) for p in (RAVI, MARTA)}
    state.ties[TieId.of(RAVI, MARTA)].functioning_habit[DEFAULT_AREA] = -0.6
    complete_one(state)
    for p in (RAVI, MARTA):
        now = axes(state.people[p])
        assert now[0] < start[p][0] and now[1] < start[p][1]  # not a return to baseline
    assert state.ties[TieId.of(RAVI, MARTA)].functioning_habit[DEFAULT_AREA] == 0.0


def test_m5d7a_config_says_why_the_increment_is_small():
    text = (REPO / "config/bowen/constants.md").read_text()
    assert "`exchange_gain` is small on purpose (`M5.D.7a`)" in text


# --- one outcome a week ----------------------------------------------------------------------


def test_m5d9_a_due_step_is_the_weeks_outcome():
    state = fresh()
    choose_iposition(state)
    for _ in range(PARAMS.prepare_ticks):
        calm(state, RAVI)
        week(state)
    owed, _ = ip.sequence_selections(state, PARAMS)
    assert [(s.actor, s.kind, s.sequence_step) for s in owed] == [(RAVI, "I-POSITION", ip.DEFINE)]


def test_m5d9_acts_on_the_sequence_tie_count_as_stay_in_contact():
    state = fresh()
    choose_iposition(state)
    toward_marta = Selection(actor=RAVI, kind="CONFLICT", targets=(MARTA,), intensity=50.0, value_key="k")
    toward_nadia = dataclasses.replace(toward_marta, targets=(NADIA,))
    assert ip.redirect_to_sequence_tie(state, toward_marta).kind == "STAY-IN-CONTACT"
    assert ip.redirect_to_sequence_tie(state, toward_nadia) == toward_nadia


def test_m5d9_a_step_that_cannot_be_made_is_deferred_and_logged():
    state = fresh()
    choose_iposition(state)
    for _ in range(PARAMS.prepare_ticks):
        calm(state, RAVI)
        week(state)
    tie = state.ties[TieId.of(RAVI, MARTA)]
    tie.interactive, tie.tie_state = False, TieState.CUT_OFF
    owed, records = ip.sequence_selections(state, PARAMS)
    assert owed == [] and seq(state).due == ip.DEFINE and any("deferred" in f for r in records for _, f, _ in r.people)


# --- what usually happens: an ensemble --------------------------------------------------------


@lru_cache(maxsize=1)
def attempts_over_seeds(seeds=24, weeks=100):
    """Ravi, given perspective, keeps taking positions with Marta while the family runs on the policy."""
    constants, family = load_constants(), load_family()
    script = load_script(kinds=KINDS, family=family)
    events = ScriptedSource("events", weeks, script.events, ())
    outcomes = []
    for seed in range(seeds):
        parts = assemble(constants, KINDS, family, events, seed=seed)
        parts.state.people[RAVI].systems_perspective = 1.0
        policy = PolicySource(events, parts.params, load_policy_rules())

        class Source:
            def scheduled(self, tick):
                return policy.scheduled(tick)

            def selections(self, tick, active, state):
                chosen = [s for s in policy.selections(tick, active, state) if s.actor != RAVI or RAVI not in active]
                if RAVI in active and state.people[RAVI].iposition_state is None:
                    chosen.append(Selection(actor=RAVI, kind="I-POSITION", targets=(MARTA,), intensity=100.0))
                elif RAVI in active:
                    chosen += [s for s in policy.selections(tick, (RAVI,), state)]
                return tuple(chosen)

        out = []
        sink = type("Sink", (), {"emit": lambda self, r: out.append(r)})()
        for _ in range(weeks):
            run_tick(parts.state, Source(), parts.params, parts.visibility, parts.activation, sink)
        marks = [f for r in out if isinstance(r, EffectRecord) and r.mechanism == "iposition"
                 for p, f, _ in r.people if p == RAVI and f.startswith("iposition:")]
        outcomes.append(tuple(marks))
    return tuple(outcomes)


def test_m5d3_abort_is_the_usual_first_outcome():
    """At the first opposition, aborting is commoner than holding."""
    held = aborted = 0
    for marks in attempts_over_seeds():
        at_opposition = False
        for mark in marks:
            if mark == "iposition:enter:OPPOSITION":
                at_opposition = True
            elif at_opposition and mark == "iposition:enter:HOLD":
                held, at_opposition = held + 1, False
            elif at_opposition and mark.startswith("iposition:end:"):
                aborted += mark == "iposition:end:aborted"
                at_opposition = False
    assert aborted > held > 0


def test_m5d8_success_usually_follows_several_failures():
    firsts = []
    for marks in attempts_over_seeds():
        ends = [m.split(":")[2] for m in marks if m.startswith("iposition:end:")]
        if "completed" in ends:
            firsts.append(ends.index("completed") + 1)
    assert firsts, "no sequence completed in the ensemble"
    assert sum(1 for n in firsts if n == 1) < len(firsts) / 2  # the first attempt rarely succeeds
    assert sorted(firsts)[len(firsts) // 2] > 1
    assert Counter(firsts).most_common(1)[0][0] > 1
