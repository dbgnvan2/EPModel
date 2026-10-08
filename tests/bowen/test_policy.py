"""The policy skeleton (Phase C step 6, plan D3).

Purpose: test the legal set, the two channels and their mixing, the capacity and
         loaded-tie gates, WITHHOLD, competing urges, the declared fallback, keyed
         selection, the rationale record, and the M4.B.2 boundary.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1, #M4.D.1a, #M4.D.1b, #M4.D.1d, #M4.D.1e, #M4.D.1f, #M4.D.3, #M4.D.3a, #M4.D.3b, #M4.D.4, #M5.C.1, #M5.F.5, #M3.D.4b, #M4.B.2, #M4.B.3, #M16.A.3b, #M16.A.3c
Tests:   this file

Unit tests of set mechanisms with invented forms and constants; none is a finding (`M11.5`).
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from src.bowen.engine.act import WITHHOLD, act
from src.bowen.engine.draws import DrawService
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.log_records import DecidedBy, EmittedRecord, SelectionRecord
from src.bowen.engine.objects import TieState
from src.bowen.engine.observe import observe
from src.bowen.engine.state import new_run_state
from src.bowen.engine.tick import run
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_policy_rules, load_script
from src.bowen.policy.policy import (
    AUTOMATIC, SELF, competing_urges, decide, legal_outcomes, propensities, self_channel_weight, self_score,
)
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.config_parse import ConfigError
from src.bowen.scenario.params import engine_params
from src.bowen.scenario.policy_rules import parse_policy_rules
from src.bowen.scenario.policy_source import PolicySource
from src.bowen.scenario.scripted_source import ScriptedSource

P = PersonId
RAVI, MARTA, NADIA, ANA = P("ravi"), P("marta"), P("nadia"), P("ana")
PARAMS = engine_params(load_constants())
KINDS = load_event_kinds()
RULES = load_policy_rules()
REPO = Path(__file__).resolve().parents[2]


def fresh(**changes):
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, KINDS)
    for pid, fields in changes.items():
        for name, value in fields.items():
            setattr(state.people[P(pid)], name, value)
    return initialise_run(state, PARAMS)


def probs_of(state, pid=RAVI):
    outcomes, probs = propensities(observe(state, pid, PARAMS), KINDS, PARAMS)
    return dict(zip((o.label for o in outcomes), probs)), outcomes


def channel_mass(state, channel, pid=RAVI):
    outcomes, probs = propensities(observe(state, pid, PARAMS), KINDS, PARAMS)
    return sum(p for o, p in zip(outcomes, probs) if o.channel == channel)


class Collect:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)


def policy_run(ticks=6, seed=3):
    constants, family = load_constants(), load_family()
    script = load_script(kinds=KINDS, family=family)
    events_only = ScriptedSource(script.script_id + "-events", script.ticks, script.events, ())
    parts = assemble(constants, KINDS, family, events_only, seed=seed)
    source = PolicySource(events_only, parts.params, RULES)
    out = Collect()
    run(parts.state, source, parts.params, parts.visibility, parts.activation, out, ticks=ticks)
    return parts.state, out.records


# --- one outcome, the legal set, the record ------------------------------------------------


def test_m4d1_exactly_one_outcome_per_person_per_tick():
    state, records = policy_run(ticks=6)
    selections = [r for r in records if isinstance(r, SelectionRecord)]
    for tick in range(6):
        actors = [r.actor for r in selections if r.tick == tick]
        assert sorted(actors) == sorted(p for p in state.people if state.people[p].alive)


def test_m4d1e_legal_set_excludes_impossible_and_gated_acts():
    state = fresh()
    state.ties[TieId.of(RAVI, MARTA)].interactive = False
    state.ties[TieId.of(RAVI, MARTA)].tie_state = TieState.CUT_OFF
    ravi = observe(state, RAVI, PARAMS)
    labels = {o.label for o in legal_outcomes(ravi, KINDS, PARAMS)}
    assert not any(label.endswith(">marta") for label in labels)  # nothing crosses a cut-off tie
    assert {o.kind for o in legal_outcomes(ravi, KINDS, PARAMS)} <= set(KINDS.channels) | {WITHHOLD}
    triangles = {o.target for o in legal_outcomes(ravi, KINDS, PARAMS) if o.kind == "TRIANGLE"}
    assert triangles == set(ravi.triangle_for) and triangles  # only toward someone a closed triad holds
    ana = observe(fresh(), ANA, PARAMS)  # Ana's ties close no triad
    assert not any(o.kind == "TRIANGLE" for o in legal_outcomes(ana, KINDS, PARAMS))
    # M5.C.1: I-POSITION fails outright for a financially dependent person — removed, not degraded.
    assert state.people[NADIA].financially_dependent and not state.people[RAVI].financially_dependent
    assert not any(o.kind == "I-POSITION" for o in legal_outcomes(observe(fresh(), NADIA, PARAMS), KINDS, PARAMS))
    assert any(o.kind == "I-POSITION" for o in legal_outcomes(observe(fresh(), RAVI, PARAMS), KINDS, PARAMS))


def test_m4d1e_selection_is_renormalised_over_the_legal_set():
    probs, _ = probs_of(fresh())
    # A degrading gate (M4.D.3b's loaded tie) can take a legal act to zero; it is not part of the mask.
    assert abs(sum(probs.values()) - 1.0) < 1e-12 and all(p >= 0 for p in probs.values())


def test_m16a3b_selection_record_carries_the_legal_set_and_the_decider():
    _, records = policy_run(ticks=1)
    selections = [r for r in records if isinstance(r, SelectionRecord)]
    for record in selections:
        if record.decided_by is DecidedBy.FALLBACK:  # Bruno: his one tie starts cut off
            assert record.legal_set == () and record.fallback_rule == "hold"
            continue
        assert record.decided_by is DecidedBy.POLICY and record.legal_set and record.draw is not None
        assert {label for label, _ in record.propensities} == set(record.legal_set)
    assert {r.decided_by for r in selections} == {DecidedBy.POLICY, DecidedBy.FALLBACK}


# --- the two channels ----------------------------------------------------------------------


def test_m4d1a_mixing_weight_reads_functional_level_only():
    base = fresh()
    anxious = fresh(ravi={"acute_anxiety": 90.0})
    tilted = fresh()
    tilted.people[RAVI].outside_ness_outward = 0.9
    mass = channel_mass(base, SELF)
    assert mass == pytest.approx(self_channel_weight(base.people[RAVI].functional_level, PARAMS))
    assert channel_mass(anxious, SELF) == pytest.approx(mass) == pytest.approx(channel_mass(tilted, SELF))
    higher = fresh(ravi={"functional_level": 70.0})
    assert channel_mass(higher, SELF) > mass


def test_m4d1a_low_level_is_nearly_all_automatic():
    low = fresh(ravi={"functional_level": 15.0, "basic_level": 15.0})
    assert channel_mass(low, AUTOMATIC) > 0.95


def test_m4d3_no_term_raises_reactive_acts_with_anxiety():
    """Before anything is learned, anxiety changes nothing in the automatic channel (amended M4.D.3)."""
    calm, anxious = probs_of(fresh())[0], probs_of(fresh(ravi={"acute_anxiety": 90.0}))[0]
    auto = [label for label in calm if label.split(">")[0] in KINDS.in_channel(AUTOMATIC)]
    assert all(calm[label] == pytest.approx(anxious[label]) for label in auto)


def test_m4d3a_lower_level_slides_selection_to_older_acts():
    def oldest_share(level):
        probs, outcomes = probs_of(fresh(ravi={"functional_level": level, "basic_level": level}))
        auto = [o for o in outcomes if o.channel == AUTOMATIC]
        total = sum(probs[o.label] for o in auto)
        return sum(probs[o.label] for o in auto if KINDS.layers[o.kind] == 0) / total
    assert oldest_share(10.0) > oldest_share(60.0)


def test_m4d3b_loaded_tie_engagement_gated_by_systems_perspective():
    def self_toward_marta(perspective):
        state = fresh(ravi={"systems_perspective": perspective})
        state.ties[TieId.of(RAVI, MARTA)].felt_impingement[RAVI] = 1.0  # loaded
        probs, outcomes = probs_of(state)
        return sum(probs[o.label] for o in outcomes if o.channel == SELF and o.target == MARTA)
    assert self_toward_marta(0.0) == 0.0 < self_toward_marta(1.0)


def test_m4d4_iposition_not_monotone_in_level():
    """Rising level opens the self-directed channel and closes the gap to the position."""
    def iposition(level):
        probs, outcomes = probs_of(fresh(ravi={"functional_level": level, "basic_level": level}))
        return sum(probs[o.label] for o in outcomes if o.kind == "I-POSITION")
    curve = [iposition(float(level)) for level in range(10, 100, 10)]
    peak = curve.index(max(curve))
    assert 0 < peak < len(curve) - 1 and curve[-1] < max(curve)


def test_m5f5_self_directed_scores_read_position_not_relief():
    _, outcomes = probs_of(fresh())
    iposition = next(o for o in outcomes if o.kind == "I-POSITION")
    base = observe(fresh(), RAVI, PARAMS)
    anxious = observe(fresh(ravi={"acute_anxiety": 90.0}), RAVI, PARAMS)
    tilted = dataclasses.replace(base, outside_ness_inward=0.9)
    assert self_score(iposition, base) == self_score(iposition, anxious)
    assert self_score(iposition, tilted) > self_score(iposition, base)


# --- WITHHOLD, urges, fallback -------------------------------------------------------------


def withheld_decision():
    obs = observe(fresh(), RAVI, PARAMS)
    for seed in range(2000):
        decision = decide(obs, KINDS, PARAMS, RULES, DrawService(seed))
        if decision.selection.kind == WITHHOLD and decision.selection.withheld:
            return decision
    raise AssertionError("no WITHHOLD in 2000 seeds")


def test_m4d1b_withhold_computes_the_automatic_move_and_emits_nothing():
    selection = withheld_decision().selection
    assert selection.withheld in KINDS.in_channel(AUTOMATIC)
    state = fresh()
    records = act(state, selection, HouseholdConductanceVisibility(PARAMS.per_hop_fidelity), PARAMS)
    assert not any(isinstance(r, EmittedRecord) for r in records) and not state.store.events()
    [record] = [r for r in records if isinstance(r, SelectionRecord)]
    assert record.event_id is None and record.withheld == selection.withheld


def test_m4d1b_withheld_move_still_changes_tie_state():
    selection = withheld_decision().selection
    state = fresh()
    tie = state.tie_between(RAVI, selection.targets[0])
    before = tie.investment.get(RAVI, 0.0)
    act(state, selection, HouseholdConductanceVisibility(PARAMS.per_hop_fidelity), PARAMS)
    assert tie.investment[RAVI] > before


def test_m4d1d_competing_urges_raise_anxiety_by_entropy():
    undecided = observe(fresh(ravi={"functional_level": 80.0}), RAVI, PARAMS)  # every layer fully available
    outcomes = legal_outcomes(undecided, KINDS, PARAMS)
    first = next(o for o in outcomes if o.channel == AUTOMATIC)
    settled = dataclasses.replace(undecided, learned_values={first.value_key: 8.0})
    assert competing_urges(outcomes, undecided, KINDS, PARAMS) == pytest.approx(PARAMS.competing_urge_gain)
    assert 0 < competing_urges(outcomes, settled, KINDS, PARAMS) < competing_urges(outcomes, undecided, KINDS, PARAMS)
    _, records = policy_run(ticks=1)
    assert any(getattr(r, "mechanism", None) == "competing_urges" for r in records)


def test_m4d1f_fallback_on_empty_legal_set_is_flagged():
    state = fresh()
    for tie in state.ties_of(ANA):
        tie.interactive, tie.tie_state = False, TieState.CUT_OFF
    decision = decide(observe(state, ANA, PARAMS), KINDS, PARAMS, RULES, DrawService(0))
    assert decision.selection.decided_by is DecidedBy.FALLBACK and decision.selection.fallback_rule == "hold"
    assert decision.selection.withheld is None and not decision.selection.targets


def test_m4d1f_non_finite_score_falls_back():
    obs = observe(fresh(), RAVI, PARAMS)
    key = next(o for o in legal_outcomes(obs, KINDS, PARAMS) if o.channel == AUTOMATIC).value_key
    broken = dataclasses.replace(obs, learned_values={key: math.inf})
    assert decide(broken, KINDS, PARAMS, RULES, DrawService(0)).selection.decided_by is DecidedBy.FALLBACK


def test_m4d1f_rules_come_from_config():
    assert (RULES.tie_break, RULES.fallback) == ("equal_probability", "hold")
    text = (REPO / "config/bowen/policy.md").read_text().replace("| `fallback` | hold |", "| `fallback` | coin_flip |")
    with pytest.raises(ConfigError, match="not implemented"):
        parse_policy_rules(text)


# --- keyed draws ---------------------------------------------------------------------------


def test_m3d4b_selection_draw_is_slot_keyed():
    """The same person in the same tick draws the same number whatever anyone else's state is."""
    a, b = fresh(), fresh(marta={"acute_anxiety": 95.0})
    da = decide(observe(a, RAVI, PARAMS), KINDS, PARAMS, RULES, DrawService(11)).selection
    db = decide(observe(b, RAVI, PARAMS), KINDS, PARAMS, RULES, DrawService(11)).selection
    assert da.draw == db.draw and da == db


# --- M4.B.2 and M4.B.3, statically -----------------------------------------------------------

POLICY_DIR = REPO / "src" / "bowen" / "policy"
ALLOWED = {
    "src.bowen.engine.act", "src.bowen.engine.draws", "src.bowen.engine.events", "src.bowen.engine.identifiers",
    "src.bowen.engine.log_records", "src.bowen.engine.observe", "src.bowen.engine.params",
}
FUTURE = {"queue", "store", "scheduled", "release", "events"}


def disallowed_imports(source: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(source)):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
        for name in names:
            if name.startswith("src.") and name not in ALLOWED and not name.startswith("src.bowen.policy"):
                found.add(name)
    return found


def future_reads(source: str) -> set[str]:
    return {n.attr for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute)} & FUTURE


def test_m4b2_policy_imports_only_the_observation():
    for path in sorted(POLICY_DIR.glob("*.py")):
        assert disallowed_imports(path.read_text()) == set(), path


def test_m4b2_check_catches_a_policy_importing_run_state():
    mutant = "from src.bowen.engine.state import RunState\nfrom src.bowen.engine.observe import Observation\n"
    assert disallowed_imports(mutant) == {"src.bowen.engine.state"}


def test_m4b3_policy_reads_no_queue_or_store():
    """No look-ahead: the policy cannot reach scheduled or future events (M4.B.3)."""
    for path in sorted(POLICY_DIR.glob("*.py")):
        assert future_reads(path.read_text()) == set(), path
    assert future_reads("x = state.queue.release(t)") == {"queue", "release"}


def test_m4b2_observation_holds_no_other_persons_state():
    base, changed = fresh(), fresh()
    marta = changed.people[MARTA]
    marta.acute_anxiety, marta.outside_ness_outward = 99.0, 1.0
    changed.ties[TieId.of(MARTA, NADIA)].felt_impingement[MARTA] = 1.0  # a tie Ravi is not party to
    assert observe(base, RAVI, PARAMS) == observe(changed, RAVI, PARAMS)
