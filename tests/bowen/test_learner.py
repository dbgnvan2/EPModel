"""The learner (Phase C step 7, plan D4) and the triangle position the values are kept by.

Purpose: test that relief reinforces and cost discourages, only inside the horizon and
         discounted by age; that another person's calming reinforces; that repeated
         relief habituates; that the self-directed channel is never reinforced; that
         values are kept per anxiety band and per believed triangle position.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.6, #M4.D.6a, #M4.D.6b, #M4.D.6d, #M4.D.6e, #M4.G.3, #M4.D.2, #M4.D.3
Tests:   this file

Unit tests of a set mechanism with invented constants; none is a finding (`M11.5`).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.bowen.engine.act import Selection, act
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.learner import EligibleAct, learn, register_acts
from src.bowen.engine.log_records import DecidedBy, SelectionRecord
from src.bowen.engine.observe import observe
from src.bowen.engine.state import new_run_state
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.policy.policy import band, triangle_position, value_key
from src.bowen.engine.tick import run
from src.bowen.io.load import load_policy_rules, load_script
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.params import engine_params
from src.bowen.scenario.policy_source import PolicySource
from src.bowen.scenario.scripted_source import ScriptedSource

P = PersonId
RAVI, MARTA, NADIA, PIA = P("ravi"), P("marta"), P("nadia"), P("pia")
PARAMS = engine_params(load_constants())
KINDS = load_event_kinds()
H = PARAMS.credit_horizon
KEY = "mid|inside|PURSUE|marta"


class Collect:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)


def policy_run(ticks, seed=3):
    # Copied from test_policy.py: the name ``tests`` is shadowed by an installed package (see test_register.py).
    family = load_family()
    script = load_script(kinds=KINDS, family=family)
    events_only = ScriptedSource(script.script_id + "-events", script.ticks, script.events, ())
    parts = assemble(load_constants(), KINDS, family, events_only, seed=seed)
    out = Collect()
    run(parts.state, PolicySource(events_only, parts.params, load_policy_rules()), parts.params,
        parts.visibility, parts.activation, out, ticks=ticks)
    return parts.state, out.records


def fresh(params=PARAMS):
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, KINDS)
    return initialise_run(state, params)


def eligible(state, key=KEY, others=(MARTA,), repetitions=0, at=0):
    state.people[RAVI].eligible_acts.append(EligibleAct(tick=at, key=key, others=others, repetitions=repetitions))


def tick_with(state, changes, params=PARAMS):
    """End one tick in which each person's acute anxiety changed by ``changes``; run the learner."""
    start = {p: q.acute_anxiety for p, q in state.people.items()}
    for pid, delta in changes.items():
        state.people[pid].acute_anxiety += delta
    records = learn(state, start, params)
    state.tick += 1
    return records


def value(state, key=KEY):
    return state.people[RAVI].learned_values.get(key, 0.0)


def run_horizon(state, per_tick, params=PARAMS):
    for _ in range(params.credit_horizon):
        tick_with(state, per_tick, params)


def test_m4d6_an_act_followed_by_relief_is_reinforced():
    relieved, worsened = fresh(), fresh()
    eligible(relieved)
    eligible(worsened)
    run_horizon(relieved, {RAVI: -2.0})
    run_horizon(worsened, {RAVI: +2.0})
    assert value(relieved) > 0 > value(worsened)


def test_m4d6_update_is_the_declared_form():
    state = fresh()
    state.people[RAVI].learned_values[KEY] = 1.0
    eligible(state, others=())
    run_horizon(state, {RAVI: -1.0})
    signal = sum(PARAMS.credit_discount ** k * 1.0 for k in range(H))
    assert value(state) == pytest.approx(1.0 + PARAMS.learning_rate * (signal - 1.0))


def test_m4d6b_nothing_outside_the_horizon_is_credited():
    """A cost that arrives after the horizon does not reach the act that caused it (M4.B.3)."""
    early, late = fresh(), fresh()
    for state in (early, late):
        eligible(state, others=())
    run_horizon(early, {})
    run_horizon(late, {})
    tick_with(late, {RAVI: +50.0})  # a deferred cost, one tick too late
    assert value(early) == value(late) == 0.0
    assert not late.people[RAVI].eligible_acts


def test_m4d6_credit_is_discounted_by_age():
    soon, later = fresh(), fresh()
    for state in (soon, later):
        eligible(state, others=())
    tick_with(soon, {RAVI: -3.0}); tick_with(soon, {}); tick_with(soon, {})
    tick_with(later, {}); tick_with(later, {}); tick_with(later, {RAVI: -3.0})
    assert value(soon) > value(later) > 0


def test_m4d6e_another_persons_calming_reinforces():
    """The child learns that what it did calmed the parent (FE07.4): the target's relief is a signal."""
    calmed, unweighted = fresh(), fresh(dataclasses.replace(PARAMS, cross_person_weight=0.0))
    eligible(calmed)
    eligible(unweighted)
    run_horizon(calmed, {MARTA: -4.0})
    run_horizon(unweighted, {MARTA: -4.0}, dataclasses.replace(PARAMS, cross_person_weight=0.0))
    assert value(calmed) > 0 == value(unweighted)


def test_m4g3_repeated_relief_habituates():
    first, fourth = fresh(), fresh()
    eligible(first, repetitions=0)
    eligible(fourth, repetitions=3)
    run_horizon(first, {RAVI: -2.0})
    run_horizon(fourth, {RAVI: -2.0})
    assert 0 < value(fourth) < value(first)
    costly_first, costly_fourth = fresh(), fresh()
    eligible(costly_first, repetitions=0)
    eligible(costly_fourth, repetitions=3)
    run_horizon(costly_first, {RAVI: +2.0})
    run_horizon(costly_fourth, {RAVI: +2.0})
    assert value(costly_first) == value(costly_fourth) < 0  # a cost is not habituated


def test_m4g3_repetitions_count_identical_recent_acts():
    state = fresh()
    visibility = HouseholdConductanceVisibility(PARAMS.per_hop_fidelity)
    for week in range(3):
        state.tick = week
        selection = Selection(actor=RAVI, kind="PURSUE", targets=(MARTA,), intensity=50.0,
                              decided_by=DecidedBy.POLICY, value_key=KEY)
        act(state, selection, visibility, PARAMS)
        register_acts(state, (selection,), PARAMS)
    assert [a.repetitions for a in state.people[RAVI].eligible_acts] == [0, 1, 2]


def test_m4d6d_self_directed_channel_is_never_reinforced():
    state, records = policy_run(ticks=12)
    self_kinds = set(KINDS.in_channel("self"))
    keys = {k for p in state.people.values() for k in p.learned_values}
    assert keys and not any(k.split("|")[2] in self_kinds for k in keys)
    selections = [r for r in records if isinstance(r, SelectionRecord)]
    assert any(r.legal_set and any(l.split(">")[0] in self_kinds for l in r.legal_set) for r in selections)
    # And a self-directed selection, emitted, opens no account.
    fresh_state = fresh()
    iposition = Selection(actor=RAVI, kind="I-POSITION", targets=(MARTA,), intensity=50.0, decided_by=DecidedBy.POLICY)
    act(fresh_state, iposition, HouseholdConductanceVisibility(PARAMS.per_hop_fidelity), PARAMS)
    register_acts(fresh_state, (iposition,), PARAMS)
    assert fresh_state.people[RAVI].eligible_acts == []


def test_m4d3_values_are_kept_per_anxiety_band():
    calm, anxious = fresh(), fresh()
    anxious.people[RAVI].acute_anxiety = anxious.people[RAVI].chronic_anxiety + 40.0
    keys = {value_key(observe(s, RAVI, PARAMS), "PURSUE", MARTA, PARAMS) for s in (calm, anxious)}
    assert len(keys) == 2 and {k.split("|")[0] for k in keys} == {
        band(observe(calm, RAVI, PARAMS), PARAMS), "high"}


def test_m4d2_triangle_position_is_read_through_belief():
    """Nadia's position in the family triad moves with what she believes of her parents' tie, not with the tie."""
    believed_close = fresh()
    believed_close.people[NADIA].tie_beliefs[TieId.of(RAVI, MARTA)].contact = 1.0
    truly_close = fresh()
    for m in (RAVI, MARTA):
        truly_close.ties[TieId.of(RAVI, MARTA)].felt_contact[m] = 1.0
    position = {name: triangle_position(observe(s, NADIA, PARAMS), PARAMS) for name, s in
                (("base", fresh()), ("believed", believed_close), ("true", truly_close))}
    assert position["believed"][0] == "outside"
    assert position["base"][0] == position["true"][0] == "inside"
    assert ("marta~ravi.contact", 1.0) in position["believed"][1]  # the belief read is recorded
