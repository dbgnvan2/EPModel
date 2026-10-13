"""Golden run fingerprints for behaviour-preserving engine changes (Phase D plan B1).

Purpose: a refactor that is meant to change nothing (the store's windowed indexes) must leave a seeded run's
         whole log byte-identical. Each fingerprint is the SHA-256 of the run's records' repr.
Spec:    docs/implementation_plan_phase_d.md B1; docs/bowen_agent_model_spec_v2.md#M3.D.4a (seeded draws)
Tests:   this file

**Updating a fingerprint is a decision, not a fix.** Change one only in the commit of an engine change meant to
alter behaviour (plan B4 onward), and say in that commit which change moved it. Fingerprints first taken
2026-10-10 at the engine of commit 713458a (spec revision 14, before B1). Moved 2026-10-11 by B3 (channel stamping:
the log only), B4 (cutoff as rejection, spec revisions 14 and 16: behaviour) and B5/B6 (the slow tick: tie wear
and the estimator), spec revisions 17-19 (nodal events, stress response, perspective, soothing review), and revision 20 (the
system's adjustment settles back; then, 2026-10-12, a level the estimator leaves out), and revision 21
(holding a position stops the functioning exchange), and revision 22 (calm relative to the person).
Revision 23 (the estimator reads pseudo-self with its sign) did not move them, and that says nothing about it: the
phase_c run is 104 weeks and never fills the estimator's 156-week window, and the phase_d run has no nodal events, so
no one's breadth reaches `estimator_breadth` and no rise is possible under any reading (review 2026-10-12). M1.A.4k's
reading is pinned by tests/bowen/test_slow_tick.py instead. Moved 2026-10-12 by revision 24 (B7: the symptom integrand
read above the usual level, the channels, lock-in and removal; behaviour), and by revisions 25-27 (attribution and the
family's focus, belief in the witness's appraisal, the marital-distance gate, the tie-weighted cross-person term, and
borrowed pseudo-self settling back; behaviour).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import src.bowen.ensemble.criteria as criteria
from src.bowen.engine.log_records import CollectingEmitter
from src.bowen.engine.tick import run_tick
from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_policy_rules
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.policy_source import PolicySource
from src.bowen.scenario.scripted_source import ScriptedSource

REPO = Path(__file__).resolve().parents[2]
GOLDEN = {
    # (family, seed, weeks): fingerprint
    ("phase_c", 7, 104): "61787089648f79bab229711b614af1cb8d89af801ce98fab4152179f4854af2d",
    ("phase_d", 11, 208): "d860880a9b5b45e92e0d65cbcbab1b74e297f9000982149792b2ec8d9b563cf9",
}


def _records(family: str, seed: int, weeks: int):
    constants, kinds = load_constants(), load_event_kinds()
    path = criteria.FAMILIES["phase_c"] if family == "phase_c" else REPO / "config" / "bowen" / "family_phase_d.md"
    spells = tuple(criteria.spell("phase_c", weeks)) if family == "phase_c" else ()
    script = ScriptedSource("golden", weeks, spells, ())
    parts = assemble(constants, kinds, load_family(path), script, seed=seed)
    source = PolicySource(script, parts.params, load_policy_rules())
    emitter = CollectingEmitter()
    for _ in range(weeks):
        run_tick(parts.state, source, parts.params, parts.visibility, parts.activation, emitter)
    return emitter.records


@pytest.mark.parametrize("family,seed,weeks", sorted(GOLDEN))
def test_b1_seeded_run_log_is_unchanged(family, seed, weeks):
    fingerprint = hashlib.sha256(repr(_records(family, seed, weeks)).encode()).hexdigest()
    assert fingerprint == GOLDEN[(family, seed, weeks)], (
        "the run's log changed; if an engine change was meant to change behaviour, update the fingerprint in the "
        "same commit and say why (see the module docstring)")
