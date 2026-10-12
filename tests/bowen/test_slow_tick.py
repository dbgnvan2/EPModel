"""The slow tick and the basic-level estimator (Phase D plan B5, B6).

Purpose: prove that the slow tick fires at the end of week 51, not at week 0; that the estimator raises
         ``basic_level`` only on a loaded, stable, broad window, never on a benign one, never for a dependent person,
         and only as far as the person acted self-directed under load; that chronic anxiety falls only with a rise;
         and that ties wear by their members' chronic anxiety, offset by contact. Comparisons assert directions (M0.4).
Spec:    docs/bowen_agent_model_spec_v2.md#M3.B.1, #M2.A.0b, #M1.A.4a, #M1.A.4b, #M1.A.4d, #M1.A.4h, #M1.A.4k,
         #M7.A.1a, #M7.B.1, #M7.E.4
Tests:   this file
"""

from __future__ import annotations

import dataclasses

import pytest

from src.bowen.engine.act import Selection, act
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.invariants import InvariantViolation
from src.bowen.engine.log_records import CollectingEmitter, DecidedBy, EffectRecord
from src.bowen.engine.slow_tick import Week, assert_slow_invariants, estimate, history_length, run_slow_tick
from src.bowen.engine.state import new_run_state
from src.bowen.engine.tick import run_tick
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.params import engine_params
from src.bowen.scenario.scripted_source import ScriptedSource

P = PersonId
RAVI, MARTA, NADIA, ANA = P("ravi"), P("marta"), P("nadia"), P("ana")
MARITAL = TieId.of(RAVI, MARTA)
PARAMS = engine_params(load_constants())
VIS = HouseholdConductanceVisibility(PARAMS.per_hop_fidelity)
N = history_length(PARAMS)


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, PARAMS)


def window(functional=50.0, load=5.0, self_directed=True, nodal_every=20, wobble=0.0):
    """A full window of weeks: steady functioning under steady load, with a distinct nodal event every so often."""
    return [Week(t, functional + (wobble if t % 2 else -wobble), load, self_directed,
                 (f"KIND_{t // nodal_every % 4}",) if nodal_every and t % nodal_every == 0 else ()) for t in range(N)]


class _Quiet:
    def scheduled(self, tick):
        return ()

    def selections(self, tick, active, state):
        return ()


# --- B5: the slow tick -----------------------------------------------------------------------------------------


def test_m3b1_slow_tick_fires_at_the_end_of_week_51_not_at_week_0():
    parts = assemble(load_constants(), load_event_kinds(), load_family(), ScriptedSource("q", 104, (), ()), seed=0)
    emitter = CollectingEmitter()
    for _ in range(104):
        run_tick(parts.state, _Quiet(), parts.params, parts.visibility, parts.activation, emitter)
    fired = [r.tick for r in emitter.records if isinstance(r, EffectRecord) and r.mechanism == "slow_tick"]
    assert fired == [51, 103]


def test_m1a4k_history_records_load_channel_and_nodal_events():
    """Each week: the load at selection, the channel the person acted in, the nodal kinds that reached them, and the
    week's own functional_level."""
    from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition

    stressor = Event(id=EventId(1, "script:0", 0), kind="JOB_LOSS", mechanism=Mechanism.EXOGENOUS_STRESSOR,
                     sender=None, targets=(RAVI,), intensity=50.0, timestamp=1, duration=1, exogenous=True,
                     source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS)

    class Script:
        def scheduled(self, tick):
            return (stressor,) if tick == 1 else ()

        def selections(self, tick, active, state):
            if tick == 0:
                return (Selection(RAVI, "STAY-IN-CONTACT", (MARTA,), 100.0),)
            if tick == 1:
                return (Selection(RAVI, "OVERFUNCTION", (NADIA,), 100.0),)
            return ()

    parts = assemble(load_constants(), load_event_kinds(), load_family(), ScriptedSource("q", 3, (), ()), seed=0)
    state = parts.state
    state.people[RAVI].acute_anxiety = state.people[RAVI].chronic_anxiety + 7.0
    levels, own = [], []
    for _ in range(3):
        run_tick(state, Script(), parts.params, parts.visibility, parts.activation, CollectingEmitter())
        levels.append(state.people[RAVI].functional_level)
        own.append(state.people[RAVI].functional_level - max(0.0, state.people[RAVI].pseudo_self or 0.0))
    weeks = state.estimator_history[RAVI]
    assert [w.tick for w in weeks] == [0, 1, 2]
    # the excess at selection: the 7 set here, plus week 0's standing load and appraisals before selection
    assert weeks[0].load > 7.0
    assert [w.self_directed for w in weeks] == [True, False, False]       # STAY-IN-CONTACT, OVERFUNCTION, none
    assert [w.nodal for w in weeks] == [(), ("JOB_LOSS",), ()]
    # Revision 19: the estimator reads functioning less the borrowed part. OVERFUNCTION (week 1, landing week 2) raised
    # Ravi's functional level by borrowing, and the reading does not count it.
    assert [w.functional_level for w in weeks] == pytest.approx(own)
    assert levels[2] > levels[1] and own[2] == pytest.approx(own[1])


def test_m7e4_ties_wear_by_their_members_chronic_anxiety():
    calm, anxious = fresh(), fresh()
    for m in MARITAL.members():
        anxious.people[m].chronic_anxiety = calm.people[m].chronic_anxiety + 20
    calm.tick = anxious.tick = PARAMS.slow_tick_fast_ticks - 1
    bond = calm.ties[MARITAL].bond_energy
    start = calm.ties[MARITAL].conductance
    run_slow_tick(calm, PARAMS)
    run_slow_tick(anxious, PARAMS)
    assert anxious.ties[MARITAL].conductance < calm.ties[MARITAL].conductance < start
    assert calm.ties[MARITAL].bond_energy == bond                               # M1.B.4
    assert calm.ties[MARITAL].dyad_age == 1


def _contact(state, times, refused=False):
    """Send STAY-IN-CONTACT from Ravi to Marta ``times`` times, and deliver it (or refuse it, M1.B.3c)."""
    from src.bowen.engine.act import record_deliveries
    from src.bowen.engine.rejection import raise_rejection, refuse

    if refused:
        raise_rejection(state.ties[MARITAL], MARTA, 0.9)
    for index in range(times):
        act(state, Selection(RAVI, "STAY-IN-CONTACT", (MARTA,), 100.0, index=index, decided_by=DecidedBy.SCRIPTED),
            VIS)
    batch = state.queue.release(state.tick + 1)
    accepted, _ = refuse(state, batch, PARAMS)
    record_deliveries(state, accepted)


def test_m7e4_contact_offsets_tie_wear_only_in_its_year():
    worn, kept, stale = fresh(), fresh(), fresh()
    year = PARAMS.slow_tick_fast_ticks
    enough = int(1 / PARAMS.wear_offset_per_act)
    _contact(kept, enough)
    _contact(stale, enough)
    start = worn.ties[MARITAL].conductance
    worn.tick = kept.tick = year - 1
    stale.tick = 2 * year - 1                                  # the contact was in the year before this slow tick
    for state in (worn, kept, stale):
        run_slow_tick(state, PARAMS)
    assert worn.ties[MARITAL].conductance < start == kept.ties[MARITAL].conductance
    assert stale.ties[MARITAL].conductance == worn.ties[MARITAL].conductance


def test_m7e4_refused_contact_offsets_nothing():
    """Review finding: contact a cutter refused (M1.B.3c) never reached them, so it does not keep the tie."""
    kept, refused = fresh(), fresh()
    enough = int(1 / PARAMS.wear_offset_per_act)
    _contact(kept, enough)
    _contact(refused, enough, refused=True)
    for state in (kept, refused):
        for m in MARITAL.members():                            # the same chronic anxiety in both arms
            state.people[m].chronic_anxiety = kept.people[m].chronic_anxiety
        state.tick = PARAMS.slow_tick_fast_ticks - 1
        run_slow_tick(state, PARAMS)
    assert refused.ties[MARITAL].conductance < kept.ties[MARITAL].conductance


def test_m7b1_chronic_anxiety_falls_only_with_a_rise():
    risen, still = fresh(), fresh()
    risen.estimator_history[RAVI] = window(functional=risen.people[RAVI].basic_level + 10)
    still.estimator_history[RAVI] = window(functional=still.people[RAVI].basic_level + 10, self_directed=False)
    before = risen.people[RAVI].chronic_anxiety
    run_slow_tick(risen, PARAMS)
    run_slow_tick(still, PARAMS)
    assert risen.people[RAVI].basic_level > still.people[RAVI].basic_level
    assert risen.people[RAVI].chronic_anxiety < before == still.people[RAVI].chronic_anxiety


def test_m7b1_slow_invariants_catch_a_rise_in_chronic_anxiety():
    state = fresh()
    before = {pid: (p.basic_level, p.chronic_anxiety) for pid, p in state.people.items()}
    state.people[RAVI].chronic_anxiety += 1
    with pytest.raises(InvariantViolation, match="M7.B.1"):
        assert_slow_invariants(state, before)


# --- B6: the estimator -----------------------------------------------------------------------------------------


def test_m1a4b_a_benign_window_does_not_raise_basic_level():
    """An easy life reads as nothing: elevated, stable functioning with no load is no evidence (M1.A.4b)."""
    basic = 40.0
    assert estimate(window(functional=60, load=0.0), basic, False, PARAMS) is None
    assert estimate(window(functional=60, load=5.0, nodal_every=0), basic, False, PARAMS).rise == 0


def test_m1a4k_a_loaded_stable_broad_window_raises_basic_level():
    reading = estimate(window(functional=50), 40.0, False, PARAMS)
    assert reading.breadth >= PARAMS.estimator_breadth and reading.rise > 0


def test_m1a4k_an_unstable_window_raises_less():
    steady = estimate(window(functional=50), 40.0, False, PARAMS)
    shaky = estimate(window(functional=50, wobble=4.0), 40.0, False, PARAMS)
    assert 0 <= shaky.rise < steady.rise


def test_m7a1a_a_dependent_twin_does_not_rise():
    independent = estimate(window(functional=50), 40.0, False, PARAMS)
    dependent = estimate(window(functional=50), 40.0, True, PARAMS)
    assert independent.rise > 0 == dependent.rise


def test_m1a4h_carrying_load_counts_and_discharging_it_does_not():
    held = estimate(window(functional=50, self_directed=True), 40.0, False, PARAMS)
    discharged = estimate(window(functional=50, self_directed=False), 40.0, False, PARAMS)
    assert held.rise > 0 == discharged.rise


def test_m1a4d_the_capacity_to_rise_falls_off_toward_the_bottom():
    low = estimate(window(functional=30), 20.0, False, PARAMS)
    high = estimate(window(functional=70), 60.0, False, PARAMS)
    assert 0 < low.rise < high.rise                                             # same evidence, 10 points


def test_m1a4k_the_estimator_waits_for_a_full_window():
    assert estimate(window()[1:], 40.0, False, PARAMS) is None


def test_m7a1_basic_level_never_falls_on_the_slow_tick():
    state = fresh()
    state.estimator_history[RAVI] = window(functional=state.people[RAVI].basic_level - 15)
    before = state.people[RAVI].basic_level
    run_slow_tick(state, PARAMS)
    assert state.people[RAVI].basic_level == before


def test_m3b1_a_long_run_keeps_its_slow_invariants():
    """Four years of the Phase C family on the policy: the slow tick runs four times and never raises."""
    from src.bowen.io.load import load_policy_rules
    from src.bowen.scenario.policy_source import PolicySource

    weeks = 4 * PARAMS.slow_tick_fast_ticks
    script = ScriptedSource("q", weeks, (), ())
    parts = assemble(load_constants(), load_event_kinds(), load_family(), script, seed=3)
    source = PolicySource(script, parts.params, load_policy_rules())
    emitter = CollectingEmitter()
    for _ in range(weeks):
        run_tick(parts.state, source, parts.params, parts.visibility, parts.activation, emitter)
    assert sum(1 for r in emitter.records if isinstance(r, EffectRecord) and r.mechanism == "slow_tick") == 4
    assert dataclasses.is_dataclass(parts.state.estimator_history[RAVI][0])


def test_m1a4k_a_recurring_stressor_is_one_situation():
    """Breadth counts kinds of nodal event: the same stressor every week is one situation, however often it recurs."""
    recurring = [dataclasses.replace(w, nodal=("JOB_LOSS",) if w.nodal else ()) for w in window(functional=50)]
    assert estimate(recurring, 40.0, False, PARAMS).breadth == 1 < PARAMS.estimator_breadth
    assert estimate(recurring, 40.0, False, PARAMS).rise == 0


def test_m7e4_contact_counts_in_the_year_it_lands():
    """Review finding: a STAY-IN-CONTACT sent in a year's last week lands in the next, and counts there."""
    late, none = fresh(), fresh()
    year = PARAMS.slow_tick_fast_ticks
    late.tick = year - 1
    _contact(late, int(1 / PARAMS.wear_offset_per_act))        # sent at week 51, delivered at week 52
    late.tick = none.tick = 2 * year - 1
    run_slow_tick(late, PARAMS)
    run_slow_tick(none, PARAMS)
    assert late.ties[MARITAL].conductance > none.ties[MARITAL].conductance


def test_m1a4k_nodal_kinds_come_from_config(tmp_path):
    from src.bowen.engine.events import EventKinds
    from src.bowen.io.load import load_nodal_reach
    from src.bowen.scenario.config_parse import ConfigError

    kinds = load_event_kinds()
    assert {"JOB_LOSS", "MOVE", "SERIOUS_ILLNESS", "SYMPTOM_ONSET", "INSTITUTIONALIZE"} <= kinds.nodal
    assert "PURSUE" not in kinds.nodal and "TRIGGER" not in kinds.nodal
    assert kinds.nodal_reach["INSTITUTIONALIZE"] == "ties" and kinds.nodal_reach["MOVE"] == "reached"
    head = "| kind | reach | what it is |\n|---|---|---|\n"
    for body, match in (("| `JOB_LOSSS` | reached | a typo |\n", "JOB_LOSSS"),
                        ("| `MOVE` | reached | a |\n| `MOVE` | reached | b |\n", "once"),
                        ("| `MOVE` | far | a |\n", "reach")):
        bad = tmp_path / "nodal.md"
        bad.write_text(head + body)
        with pytest.raises(ConfigError, match=match):
            load_nodal_reach(kinds, bad)
    with pytest.raises(ValueError, match="never a move"):
        EventKinds(mechanisms=kinds.mechanisms, nodal=frozenset({"PURSUE"}))


def test_m1a4k_a_nodal_event_counts_for_everyone_it_reaches():
    """A symptom happens to its bearer and reaches those it is addressed to; a member moving out reaches everyone
    tied to them; a stressor reaches its target and its witnesses."""
    from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
    from src.bowen.engine.slow_tick import reached_by

    state = fresh()
    def ev(kind, mechanism, sender=None, targets=(), witnesses=()):
        e = Event(id=EventId(0, "t", 0), kind=kind, mechanism=mechanism, sender=sender, targets=targets,
                  intensity=1.0, timestamp=0, duration=1, exogenous=sender is None,
                  source_position=SourcePosition.NONE,
                  channel=Channel.EXOGENOUS if sender is None else Channel.ENDOGENOUS)
        return e.with_witnesses(witnesses) if witnesses else e
    onset = ev("SYMPTOM_ONSET", Mechanism.ENDOGENOUS_SYMPTOM, sender=NADIA, targets=(MARTA,))
    assert reached_by(state, onset) == {NADIA, MARTA}
    leaving = ev("INSTITUTIONALIZE", Mechanism.INSTITUTIONALIZE, targets=(NADIA,))
    assert reached_by(state, leaving) == {NADIA} | {m for t in state.ties_of(NADIA) for m in t.id.members()}
    stressor = ev("MOVE", Mechanism.EXOGENOUS_STRESSOR, targets=(RAVI,), witnesses=(MARTA,))
    assert reached_by(state, stressor) == {RAVI, MARTA}
    with pytest.raises(ValueError, match="reaches no one"):
        reached_by(state, ev("MOVE", Mechanism.EXOGENOUS_STRESSOR))


def test_m1a4k_the_estimator_leaves_out_the_systems_adjustment():
    """Revision 20 (owner decision 2026-10-11): the gain taken when someone else's held stance worked (adjust_swing,
    M1.E.7) is not the person's own functioning, so the history reads functioning without it — as without borrowed
    pseudo-self."""
    from src.bowen.engine.slow_tick import record_week

    state = fresh()
    person = state.people[RAVI]
    person.pseudo_self = 0.0
    person.functional_level, person.adjust_swing = 55.0, 4.0
    record_week(state, PARAMS, {RAVI: 10.0}, (), ())
    assert [w.functional_level for w in state.estimator_history[RAVI]] == [51.0]
