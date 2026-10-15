"""Spec revision 30 (plan §4.10): origin cutoff as an input, danger and the cutoff chosen for safety, focus that locks in.

Purpose: unit tests for `M1.D.8a` (S1), `M1.B.3e` (S4) and `M1.B.8b` (S6); each names the spec ID it verifies.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.8a, #M1.B.3e, #M1.B.8b
Tests:   this file
"""

from __future__ import annotations

import dataclasses

import pytest

from src.bowen.engine.contact import loss_weight, optimum, too_little
from src.bowen.engine.danger import felt_from, update_danger
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.invariants import LEDGER, SOURCE
from src.bowen.engine.life_course import _set_origin_cutoff, note_origin_cutoff
from src.bowen.engine.reactive import own_acts_attention
from src.bowen.engine.rejection import raise_rejection, reactive_cutoff_cost
from src.bowen.engine.state import new_run_state
from src.bowen.ensemble.criteria import scenario
from src.bowen.io.load import CONFIG_DIR, load_constants, load_event_kinds, load_family
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.params import engine_params
from src.bowen.scenario.scripted_source import ScriptedSource

P = PersonId
PARAMS = engine_params(load_constants())
RAVI, MARTA, NADIA, LEO = P("ravi"), P("marta"), P("nadia"), P("leo")
MARITAL = TieId.of(RAVI, MARTA)


def fresh(params=PARAMS):
    family = load_family(CONFIG_DIR / "family_phase_c.md")
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, params)


def life_state():
    state, _ = scenario("phase_d", 0, 1)
    return state


# --- S1: origin cutoff (M1.D.8a) -----------------------------------------------------------------------------------


def test_m1d8a_a_child_of_a_cutting_off_parent_carries_more_chronic_anxiety():
    """Adversarial: the same child at 20, grown up with a parent holding rejection or with none."""
    grown_cut, grown_open = life_state(), life_state()
    grown_cut.life.origin_exposure[LEO] = [0.6 * 52, 52]
    grown_open.life.origin_exposure[LEO] = [0.0, 52]
    before = grown_cut.people[LEO].chronic_anxiety
    _set_origin_cutoff(grown_cut, LEO, PARAMS)
    _set_origin_cutoff(grown_open, LEO, PARAMS)
    assert grown_cut.people[LEO].origin_cutoff == pytest.approx(0.6)
    assert grown_cut.people[LEO].chronic_anxiety - before == pytest.approx(PARAMS.origin_cutoff_anxiety_gain * 0.6)
    assert grown_open.people[LEO].chronic_anxiety == before and grown_open.people[LEO].origin_cutoff == 0.0


def test_m1d8a_and_seeks_more_closeness():
    """The same tie: a person with an origin cutoff wants more contact on it (fusion), so its "too little" gap is larger;
    with none the optimum is unchanged."""
    state = fresh()
    tie = state.ties[MARITAL]
    tie.felt_contact = {RAVI: 0.0, MARTA: 0.0}
    plain = optimum(state.people[RAVI], tie, PARAMS)
    gap = too_little(state.people[RAVI], tie, PARAMS)
    state.people[RAVI].origin_cutoff = 0.8
    assert optimum(state.people[RAVI], tie, PARAMS) == pytest.approx(min(1.0, plain * (1 + PARAMS.origin_fusion_gain * 0.8)))
    assert optimum(state.people[RAVI], tie, PARAMS) > plain and too_little(state.people[RAVI], tie, PARAMS) > gap > 0


def test_m1d8a_read_from_the_parents_held_rejection_not_from_a_label():
    """Adversarial: a tie labelled cut off with no rejection held adds nothing; rejection a parent holds does, at its
    size; and the largest a parent holds that week is what counts."""
    state = life_state()
    state.life.origin_exposure.clear()
    for tie in state.ties.values():
        tie.rejection = {}
    note_origin_cutoff(state, PARAMS)
    assert state.life.origin_exposure[LEO] == [0.0, 1]
    tie = state.tie_between(RAVI, P("dario"))
    tie.rejection = {RAVI: 0.6, P("dario"): 1.0}  # Dario's own rejection is not Leo's parent's
    state.ties[MARITAL].rejection = {MARTA: 0.3}  # a second parent's smaller rejection: the largest counts, not the sum
    note_origin_cutoff(state, PARAMS)
    assert state.life.origin_exposure[LEO] == [pytest.approx(0.6), 2]


def test_m1d8a_set_once_at_twenty():
    """Only children under 20 accumulate, and the value is set once: the exposure is consumed."""
    state = life_state()
    state.life.origin_exposure[LEO] = [10.0, 20]
    _set_origin_cutoff(state, LEO, PARAMS)
    assert LEO not in state.life.origin_exposure
    state.people[LEO].age_years = PARAMS.development_end_age_years
    note_origin_cutoff(state, PARAMS)
    assert LEO not in state.life.origin_exposure  # at 20 and after, nothing more is added


# --- S4: danger (M1.B.3e) ------------------------------------------------------------------------------------------


def move(state, sender, target, kind="CONFLICT", index=0):
    event = Event(id=EventId(state.tick, sender.value, index), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
                  targets=(target,), intensity=100.0, timestamp=state.tick, duration=1, exogenous=False,
                  source_position=SourcePosition.NONE, channel=Channel.AUTOMATIC)
    state.store.record_event(event)
    return Delivery(state.tick, event.id, target, Role.TARGET, state.tick, 0)


def test_m1b3e_felt_danger_rises_with_sustained_intrusion():
    """Adversarial: weeks of intrusion from Ravi raise Marta's felt danger from him above zero; the same weeks without
    it leave none; and it is per direction (Ravi feels nothing from Marta)."""
    intruded, quiet = fresh(), fresh()
    for week in range(3 * PARAMS.danger_window):
        for state in (intruded, quiet):
            state.tick = week
        batch = tuple(move(intruded, RAVI, MARTA, index=i) for i in range(4))
        update_danger(intruded, batch, PARAMS)
        update_danger(quiet, (), PARAMS)
    tie = intruded.ties[MARITAL]
    assert tie.received_intrusion[MARTA] > PARAMS.danger_threshold
    assert tie.felt_danger[MARTA] > 0 and tie.felt_danger.get(RAVI, 0.0) == 0
    assert quiet.ties[MARITAL].felt_danger.get(MARTA, 0.0) == 0


def test_m1b3e_felt_danger_is_scaled_and_floored_by_a_declared_danger():
    t = PARAMS.danger_threshold
    assert felt_from(t, 0.0, PARAMS) == 0.0 and felt_from(1.5 * t, 0.0, PARAMS) == pytest.approx(0.5)
    assert felt_from(10 * t, 0.0, PARAMS) == 1.0 and felt_from(0.0, 0.7, PARAMS) == 0.7


def test_m1b3e_contact_with_a_dangerous_person_costs_the_recipient():
    """Adversarial: the same move costs Marta only when she feels Ravi as dangerous; the cost is logged as a source."""
    felt, safe = fresh(), fresh()
    felt.ties[MARITAL].felt_danger[MARTA] = 1.0
    costs = {}
    for name, state in (("felt", felt), ("safe", safe)):
        before = state.people[MARTA].acute_anxiety
        records = update_danger(state, (move(state, RAVI, MARTA, kind="PURSUE"),), PARAMS)
        costs[name] = state.people[MARTA].acute_anxiety - before
        if name == "felt":
            assert [r.mechanism for r in records][0] == "danger_contact"
    assert costs["felt"] == pytest.approx(PARAMS.danger_gain) and costs["safe"] == 0.0
    assert LEDGER["danger_contact"] == SOURCE


def test_m1b3e_a_cutoff_toward_danger_is_principled_in_proportion():
    """Adversarial: the same cutoff, toward a person felt as half dangerous and toward a safe one: the reactive cost and
    the "too little" term are halved in the first; a declared principled cutoff pays nothing."""
    toward_danger, toward_safe = fresh(), fresh()
    for state in (toward_danger, toward_safe):
        tie = state.ties[MARITAL]
        raise_rejection(tie, MARTA, 1.0)
        tie.felt_contact = {RAVI: 0.0, MARTA: 0.0}
    toward_danger.ties[MARITAL].felt_danger[MARTA] = 0.5
    arms = (("danger", toward_danger), ("safe", toward_safe))
    gap = {n: too_little(s.people[MARTA], s.ties[MARITAL], PARAMS) for n, s in arms}  # before the cost moves anxiety
    cost = {n: {p: v for r in reactive_cutoff_cost(s, PARAMS) for p, v in r.acute_anxiety}[MARTA] for n, s in arms}
    assert cost["danger"] == pytest.approx(0.5 * cost["safe"]) and cost["safe"] > 0
    assert gap["danger"] == pytest.approx(0.5 * gap["safe"]) and gap["safe"] > 0
    assert loss_weight(toward_safe.people[RAVI], toward_safe.ties[MARITAL]) == 1.0  # holding no rejection: all felt
    declared = fresh()
    raise_rejection(declared.ties[MARITAL], MARTA, 1.0)
    declared.ties[MARITAL].felt_contact = {RAVI: 0.0, MARTA: 0.0}
    declared.ties[MARITAL] = dataclasses.replace(declared.ties[MARITAL], principled=(MARTA,))
    assert too_little(declared.people[MARTA], declared.ties[MARITAL], PARAMS) == 0.0
    assert MARTA not in dict(p for r in reactive_cutoff_cost(declared, PARAMS) for p in r.acute_anxiety)


def test_m1b3e_danger_is_never_named():
    """No reference family declares a danger; the engine reads it from behaviour only."""
    for name in ("family_phase_c.md", "family_phase_d.md"):
        assert all(p.danger == 0.0 for p in load_family(CONFIG_DIR / name).people.values())


# --- S6: focus that locks in (M1.B.8b) -----------------------------------------------------------------------------


def test_m1b8b_a_projection_act_adds_to_the_senders_attention():
    """Adversarial: an OVERFUNCTION from Ravi to Nadia adds to Ravi's attention on that tie; a DISTANCE adds nothing;
    nothing goes to Nadia's side."""
    params = dataclasses.replace(PARAMS, focus_reinforce_gain=2.0)
    state = fresh(params)
    tie = TieId.of(RAVI, NADIA)

    def sent(kind):
        return Event(id=EventId(0, "ravi", 0), kind=kind, mechanism=Mechanism.MOVE, sender=RAVI, targets=(NADIA,),
                     intensity=100.0, timestamp=0, duration=1, exogenous=False, source_position=SourcePosition.NONE,
                     channel=Channel.AUTOMATIC)
    assert own_acts_attention(state, {}, [sent("OVERFUNCTION")], params) == {(tie, RAVI): pytest.approx(2.0)}
    assert own_acts_attention(state, {}, [sent("DISTANCE")], params) == {}


# --- S5: the danger premise (M11.C.47) and the force of an anxious act (M4.D.1h) -----------------------------------


def test_m4d1h_an_anxious_act_has_more_force():
    """Adversarial: the same automatic outcome, chosen calm and chosen anxious; the anxious one lands harder, bounded by
    (1 + force_anxiety_gain); a self-directed outcome keeps its force."""
    from src.bowen.engine.observe import observe
    from src.bowen.policy.policy import force, legal_outcomes
    state = fresh()
    outcomes = legal_outcomes(observe(state, RAVI, PARAMS), load_event_kinds(), PARAMS)
    auto = next(o for o in outcomes if o.channel == "automatic")
    own = next(o for o in outcomes if o.channel != "automatic")
    state.people[RAVI].acute_anxiety = state.people[RAVI].chronic_anxiety
    calm = observe(state, RAVI, PARAMS)
    state.people[RAVI].acute_anxiety = state.people[RAVI].chronic_anxiety + 100.0
    anxious = observe(state, RAVI, PARAMS)
    assert force(auto, calm, PARAMS) == PARAMS.policy_intensity
    assert force(auto, anxious, PARAMS) == pytest.approx(PARAMS.policy_intensity * (1 + PARAMS.force_anxiety_gain / 2))
    assert force(own, anxious, PARAMS) == PARAMS.policy_intensity
    state.people[RAVI].acute_anxiety += 1e6
    assert force(auto, observe(state, RAVI, PARAMS), PARAMS) < PARAMS.policy_intensity * (1 + PARAMS.force_anxiety_gain)


def test_m11c47_arms_differ_only_in_one_persons_level_and_stress():
    """Adversarial: each arm's setup and stressors, captured from the scenario call. Only the first spouse's level
    differs, and only the treatment has stressors, all on that spouse; both read the same two readouts."""
    from unittest import mock

    from src.bowen.ensemble import criteria_d2
    from src.bowen.ensemble.criteria import FAMILIES, scenario
    settings = {**criteria_d2.CRITERIA_D2["M11.C.47"].settings, "weeks": 26}
    seen = {}
    for arm in ("treatment", "baseline"):
        def capture(family, seed, weeks, **kw):
            seen[arm] = kw
            return scenario(family, seed, weeks, **kw)
        with mock.patch.object(criteria_d2, "scenario", capture):
            readouts = criteria_d2.arm_c47(arm, 0, settings).readouts
        assert set(readouts) == {"felt_danger", "principled_cutoff", "held_weeks"}
    levels = {}
    for arm, kw in seen.items():
        state = assemble(load_constants(), load_event_kinds(), load_family(FAMILIES["danger"]),
                         ScriptedSource("t", 1, (), ()), seed=0).state
        kw["setup"](state)
        levels[arm] = {pid: p.basic_level for pid, p in state.people.items()}
    person = criteria_d2.PARENTS_DANGER[0]
    assert levels["treatment"][person] == settings["level_low"] and levels["baseline"][person] == settings["level_high"]
    assert {k: v for k, v in levels["treatment"].items() if k != person} == \
        {k: v for k, v in levels["baseline"].items() if k != person}
    assert seen["baseline"]["spells"] == ()
    spells = seen["treatment"]["spells"]
    assert spells and all(e.targets == (person,) and e.kind == settings["stress_kind"] for e in spells)
    assert [(r.name, r.report_only) for r in criteria_d2.CRITERIA_D2["M11.C.47"].readouts] == [
        ("felt_danger", False), ("principled_cutoff", False), ("held_weeks", True)]


def test_m11c47_principled_cutoff_reads_zero_without_rejection():
    """The baseline, in which no one comes to feel danger from the person, reads 0 principled cutoff: a value, which
    the runner compares, not a missing share (correctness review, 2026-10-14)."""
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2, arm_c47
    settings = {**CRITERIA_D2["M11.C.47"].settings, "weeks": 26}
    assert arm_c47("baseline", 0, settings).readouts["principled_cutoff"] == 0.0


@pytest.mark.ensemble
def test_m11c47_low_level_and_stress_breed_danger():
    import os

    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2
    from src.bowen.ensemble.runner import PASS, run_criterion
    constants = load_constants()
    rules = {k: constants[k] for k in ("ensemble_block", "ensemble_cap", "ensemble_precision", "ensemble_margin",
                                       "ensemble_alpha", "fallback_flag_rate", "equivalence_margin")}
    verdict = run_criterion(CRITERIA_D2["M11.C.47"], rules, workers=os.cpu_count() or 1)
    assert verdict.outcome == PASS, f"M11.C.47: {verdict.outcome} {verdict.readouts}"


# --- S6: anxious focus compounds (M1.B.8b) -------------------------------------------------------------------------


def test_m1b8b_anxious_focus_compounds_where_it_already_is():
    """Adversarial: two children functioning alike; with the parent's attention split 6 to 4 the larger share gains what
    the smaller loses, so the split widens and the total is unchanged (the owner, 2026-10-14: "saturate the investment",
    then "Shift, gain 10"); split equally it stays equal; a calm parent moves nothing; investment never goes below 0."""
    from src.bowen.engine.reactive import weakness_focus
    ravi_nadia, ravi_pia = TieId.of(RAVI, NADIA), TieId.of(RAVI, P("pia"))

    def gained(split, above):
        state = fresh()
        state.people[NADIA].functional_level = state.people[P("pia")].functional_level = 35.0
        for pid in (RAVI, MARTA):
            state.people[pid].usual_excess = 0.0
        state.ties[ravi_nadia].investment[RAVI], state.ties[ravi_pia].investment[RAVI] = split
        before = {t: state.ties[t].investment[RAVI] for t in (ravi_nadia, ravi_pia)}
        weakness_focus(state, {RAVI: above, MARTA: above}, PARAMS)
        return {t: state.ties[t].investment[RAVI] - before[t] for t in before}

    uneven = gained((6.0, 4.0), 10.0)
    spread = 0.6 ** 2 + 0.4 ** 2
    assert uneven[ravi_nadia] == pytest.approx(PARAMS.focus_compound_gain * 0.1 * 10.0 * 0.6 * (0.6 - spread))
    assert uneven[ravi_nadia] > 0 > uneven[ravi_pia]  # the split widens
    assert uneven[ravi_nadia] + uneven[ravi_pia] == pytest.approx(0.0)  # and adds no investment
    even = gained((5.0, 5.0), 10.0)
    assert even[ravi_nadia] == pytest.approx(even[ravi_pia]) == pytest.approx(0.0)
    assert gained((6.0, 4.0), 0.0) == {ravi_nadia: 0.0, ravi_pia: 0.0}
    tiny = gained((99.0, 1e-9), 1000.0)  # the smaller share cannot be driven below zero
    assert tiny[ravi_pia] >= -1e-9


def _two_alike(state) -> None:
    """The four-person fixture's children made alike, so nothing but the run can tell them apart."""
    c1, c2 = state.people[P("c1")], state.people[P("c2")]
    for name in ("basic_level", "functional_level", "chronic_anxiety", "acute_anxiety", "channel_prior"):
        setattr(c2, name, getattr(c1, name))


def _focus_run(seed, overrides=None, weeks=156):
    from src.bowen.ensemble.criteria_d import nodal
    shares, chosen = [], []
    parents, kids = (P("f"), P("m")), (P("c1"), P("c2"))

    def watch(state):
        if state.tick % 26 == 25 or not shares:
            for par in parents:
                held = [state.tie_between(par, k).investment.get(par, 0.0) for k in kids]
                shares.append((state.tick, max(held) / sum(held) if sum(held) else 0.5))
                if state.tick == weeks - 1:
                    chosen.append(kids[held.index(max(held))])
    scenario("four", seed, weeks, spells=nodal(weeks, parents), setup=_two_alike, watch=watch, overrides=overrides)
    start = min(t for t, _ in shares)
    first = sum(s for t, s in shares if t == start) / 2
    last = sum(s for t, s in shares if t == weeks - 1) / 2
    return first, last, chosen


@pytest.mark.ensemble
def test_m1b8b_focus_concentrates_and_grows():
    """Two children alike, 3 years, 20 seeds: with compounding the larger share of each parent's attention grows from
    the first week to the end, and ends larger than with compounding off. (Read from the first week: with the shift
    form, the share is already 0.84 by week 26 and holds, 0.835 at 3 years, ad hoc 2026-10-14.)"""
    on = [_focus_run(s) for s in range(20)]
    off = [_focus_run(s, {"focus_compound_gain": 0.0}) for s in range(20)]
    mean = lambda xs: sum(xs) / len(xs)
    assert mean([last for _, last, _ in on]) > mean([first for first, _, _ in on])
    assert mean([last for _, last, _ in on]) > mean([last for _, last, _ in off])


@pytest.mark.ensemble
def test_m1b8b_no_child_is_favoured_across_seeds():
    """Adversarial: with two children alike, the focus settles on each of them in some seeds; identity decides nothing."""
    from collections import Counter
    focus = Counter(k for s in range(20) for k in _focus_run(s)[2])
    assert focus[P("c1")] >= 8 and focus[P("c2")] >= 8  # of 40 parent-ends; each at least a fifth


# --- the wiring: each mechanism is called where the spec puts it (correctness review, 2026-10-14) ------------------


def test_m1d8a_the_yearly_childhood_step_sets_origin_cutoff_at_twenty():
    """The yearly childhood step, not only the helper, sets the origin cutoff at 20 and consumes the exposure."""
    from src.bowen.engine.life_course import _childhood
    state = life_state()
    state.people[LEO].age_years = PARAMS.development_end_age_years
    state.life.origin_exposure[LEO] = [0.5 * 52, 52]
    _childhood(state, PARAMS)
    assert state.people[LEO].origin_cutoff == pytest.approx(0.5) and LEO not in state.life.origin_exposure


def test_m1b8b_the_tick_adds_own_acts_to_attention():
    """The tick passes the week's emitted acts to `own_acts_attention`."""
    from unittest import mock

    from src.bowen.engine import tick
    calls = []
    real = tick.own_acts_attention

    def spy(state, attended, sent, params):
        calls.append(len(sent))
        return real(state, attended, sent, params)
    with mock.patch.object(tick, "own_acts_attention", spy):
        scenario("phase_c", 0, 4)
    assert len(calls) == 4 and sum(calls) > 0


def test_m4d1h_decide_sets_intensity_by_force():
    """Every policy selection takes its intensity from `force`."""
    from unittest import mock

    from src.bowen.policy import policy
    marker = 77.0
    with mock.patch.object(policy, "force", lambda outcome, obs, params: marker):
        _, records = scenario("phase_c", 0, 4)
    from src.bowen.engine.log_records import EmittedRecord
    moves = [r.event for r in records if isinstance(r, EmittedRecord) and r.event.sender is not None
             and r.event.mechanism == Mechanism.MOVE]
    assert moves and all(e.intensity == marker for e in moves)


def test_m1b3e_a_declared_danger_is_read_from_the_family_config():
    """`dangers: person=value` sets the floor; a value outside 0 to 1, an unknown person or no value is refused; and
    an origin cutoff cannot be declared (`M1.D.8a`: 0 for a person whose childhood is not simulated)."""
    from src.bowen.io.load import read_text
    from src.bowen.scenario.family import ConfigError, build_family
    text = read_text(CONFIG_DIR / "family_phase_c.md")
    with_line = lambda line: text.replace("undifferentiation_budget: 90\n", f"undifferentiation_budget: 90\n{line}\n")
    family = build_family(with_line("dangers: ravi=0.4"), load_constants())
    assert family.people[RAVI].danger == 0.4 and family.people[MARTA].danger == 0.0
    for bad in ("ravi=1.5", "nobody=0.2", "ravi"):
        with pytest.raises(ConfigError, match="danger"):
            build_family(with_line(f"dangers: {bad}"), load_constants())
    with pytest.raises(ConfigError):
        build_family(with_line("origin_cutoffs: ravi=0.4"), load_constants())


def test_m11c2_twelve_arms_do_not_share_their_books():
    """Learning-qa 2026-10-14: the cached run's books are copied, so two arms read from it count the same."""
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2, arm_c2_twelve
    settings = {**CRITERIA_D2["M11.C.2[twelve-person family]"].settings, "years": 3}
    first, second = arm_c2_twelve("baseline", 0, settings), arm_c2_twelve("treatment", 0, settings)
    assert first.moves is not second.moves and first.moves == second.moves


def test_m1b3e_a_dead_persons_declared_danger_is_no_longer_a_floor():
    """Adversarial (re-sweep, 2026-10-14): a declared danger floors what the other feels while the declarer lives; once
    they have died it does not, and felt danger falls to what received intrusion alone gives (here none)."""
    state = fresh()
    state.people[RAVI].danger = 0.6
    update_danger(state, (), PARAMS)
    assert state.ties[MARITAL].felt_danger[MARTA] == pytest.approx(0.6)
    state.people[RAVI].alive = False
    update_danger(state, (), PARAMS)
    assert state.ties[MARITAL].felt_danger.get(MARTA, 0.0) == 0.0


def test_m1b3e_felt_danger_is_held_while_rejection_is_held():
    """Adversarial (the owner, 2026-10-14: "hold felt danger while rejection is held"): two years with nothing received
    after felt danger 0.8; holding rejection keeps it at 0.8, so the cutoff stays principled; without rejection it falls
    to 0; and once the rejection is released it falls."""
    holding, open_ = fresh(), fresh()
    for state in (holding, open_):
        state.ties[MARITAL].felt_danger[MARTA] = 0.8
        state.ties[MARITAL].received_intrusion[MARTA] = 1.8 * PARAMS.danger_threshold  # reads as 0.8
    raise_rejection(holding.ties[MARITAL], MARTA, 1.0)
    for _ in range(2 * 52):
        update_danger(holding, (), PARAMS)
        update_danger(open_, (), PARAMS)
    assert holding.ties[MARITAL].felt_danger[MARTA] == pytest.approx(0.8)
    assert loss_weight(holding.people[MARTA], holding.ties[MARITAL]) == pytest.approx(0.2)
    assert open_.ties[MARITAL].felt_danger.get(MARTA, 0.0) == 0.0
    holding.ties[MARITAL].rejection = {}
    update_danger(holding, (), PARAMS)
    assert holding.ties[MARITAL].felt_danger.get(MARTA, 0.0) == 0.0
