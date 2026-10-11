"""Golden run fingerprints for behaviour-preserving engine changes (Phase D plan B1).

Purpose: a refactor that is meant to change nothing (the store's windowed indexes) must leave a seeded run's
         whole log byte-identical. Each fingerprint is the SHA-256 of the run's records' repr.
Spec:    docs/implementation_plan_phase_d.md B1; docs/bowen_agent_model_spec_v2.md#M3.D.4a (seeded draws)
Tests:   this file

**Updating a fingerprint is a decision, not a fix.** Change one only in the commit of an engine change meant to
alter behaviour (plan B4 onward), and say in that commit which change moved it. Fingerprints first taken
2026-10-10 at the engine of commit 713458a (spec revision 14, before B1). Moved 2026-10-11 by B3 (channel stamping:
the log only) and B4 (cutoff as rejection, spec revisions 14 and 16: behaviour).
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
    ("phase_c", 7, 104): "1552c22a3a522517e787eb259deb5a55fa5f940bb623c8d90085b6dba51b2225",
    ("phase_d", 11, 208): "f8f3c1902060b40c6dd306e52d2b9377a40d38c9f109485ed1023d5eaf9c47e2",
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
