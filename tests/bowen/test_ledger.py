"""M6.4's stock-and-flow ledger — M6.I.6 as restated at revision 12 (Phase C step 10).

Purpose: test that every anxiety stock change is logged by a named mechanism, that
         transfers balance every tick of a real run, and that an unlogged, unnamed or
         unbalanced change is caught.
Spec:    docs/bowen_agent_model_spec_v2.md#M6.I.6, #M6.4, #M6.I.2, #M6.1
Tests:   this file
"""

from __future__ import annotations

import dataclasses

from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.invariants import LEDGER, ledger_problems, snapshot
from src.bowen.engine.log_records import EffectRecord, InvariantRecord, InvariantStatus
from src.bowen.engine.state import new_run_state
from src.bowen.engine.tick import run
from src.bowen.io.load import CONFIG_DIR, load_constants, load_event_kinds, load_family, load_policy_rules, load_script
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.params import engine_params
from src.bowen.scenario.policy_source import PolicySource
from src.bowen.scenario.scripted_source import ScriptedSource

P = PersonId
RAVI, MARTA = P("ravi"), P("marta")
PARAMS = engine_params(load_constants())
KINDS = load_event_kinds()
TOL = PARAMS.invariant_tolerance


def fresh():
    family = load_family(CONFIG_DIR / "family_phase_c.md")
    state = new_run_state(dict(family.people), dict(family.ties), family.family, KINDS)
    return initialise_run(state, PARAMS)


def policy_run(weeks=60, seed=4):
    family = load_family(CONFIG_DIR / "family_phase_c.md")
    script = load_script(kinds=KINDS, family=load_family())
    events = ScriptedSource("events", weeks, script.events, ())
    parts = assemble(load_constants(), KINDS, family, events, seed=seed)
    out = []
    sink = type("Sink", (), {"emit": lambda self, r: out.append(r)})()
    run(parts.state, PolicySource(events, parts.params, load_policy_rules()), parts.params, parts.visibility,
        parts.activation, sink, ticks=weeks)
    return out


def test_m6i6_transfers_balance_every_tick():
    records = policy_run()
    invariants = [r for r in records if isinstance(r, InvariantRecord)]
    assert len(invariants) == 60 and all(dict(r.results)["M6.I.6"] is InvariantStatus.PASSED for r in invariants)
    used = {r.mechanism for r in records if isinstance(r, EffectRecord) and r.acute_anxiety}
    assert {"distance_binding", "calm_contact", "appraisal", "acute_decay"} <= used  # transfers really ran
    assert used <= set(LEDGER)


def test_m64_every_stock_change_is_logged_by_name():
    state = fresh()
    before = snapshot(state)
    state.people[RAVI].acute_anxiety += 3.0
    assert any("logged" in p for p in ledger_problems(state, before, (), TOL))  # unlogged
    unnamed = EffectRecord(0, "mystery", None, acute_anxiety=((RAVI, 3.0),))
    assert any("not in M6.4" in p for p in ledger_problems(state, before, (unnamed,), TOL))
    named = EffectRecord(0, "appraisal", None, acute_anxiety=((RAVI, 3.0),))
    assert ledger_problems(state, before, (named,), TOL) == []


def test_m6i6_an_unbalanced_transfer_is_caught():
    state = fresh()
    before = snapshot(state)
    state.people[RAVI].acute_anxiety -= 2.0
    state.ties[TieId.of(RAVI, MARTA)].distance_bound_anxiety += 1.5  # half a point vanished
    leaky = EffectRecord(0, "distance_binding", None, acute_anxiety=((RAVI, -2.0),),
                         ties=((TieId.of(RAVI, MARTA), "distance_bound_anxiety", 1.5),))
    assert any("does not balance" in p for p in ledger_problems(state, before, (leaky,), TOL))


def test_m6i2_distance_runs_outside_the_budget():
    records = policy_run(weeks=30)
    for r in records:
        if isinstance(r, EffectRecord) and r.mechanism == "distance_binding":
            assert not any(name == "undifferentiation_budget" or name.startswith("sink:") for name, _ in r.sinks)
