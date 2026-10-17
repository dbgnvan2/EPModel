"""Golden run fingerprints for behaviour-preserving engine changes (Phase D plan B1).

Purpose: a refactor that is meant to change nothing (the store's windowed indexes) must leave a seeded run's
         whole log unchanged. Each fingerprint is the SHA-256 of the run's records in canonical JSON, with every float
         rounded to `FINGERPRINT_DIGITS` significant digits.
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
borrowed pseudo-self settling back; behaviour). Moved 2026-10-13 by spec revision 28 (Phase D part II, plan step L1:
the phase_d family now carries its life course — the owner's starting ages, the children's derived values, the coach
tied to the coachee only — so its run changed; behaviour). The phase_c fingerprint did not move: a run with no life
course is unchanged by part II. Moved 2026-10-14 by spec revision 29 (plan §4.8: a reactive cutoff's cost, learning
from what follows a witnessed cutoff, focus on a child's weakness and focus steering projection-type acts, and, in
phase_d, chronic anxiety built over childhood; behaviour, both runs), and again the same day by the review fixes
(focus steering made a redistribution among a kind's targets that keeps the kind's total weight; behaviour, both
runs). Moved 2026-10-14 by spec revision 30 (plan §4.10: an anxious act lands with more force, received intrusion
read as danger and its contact cost, focus that compounds and a projection act that adds to attention, and, in phase_d,
the origin cutoff set at 20; behaviour, both runs), and in phase_d again the same day by felt danger held while
rejection is held (the owner's decision; behaviour), and in both again by focus compounding made a shift among the
children that adds no investment (the owner: "Shift, gain 10"; behaviour). Moved 2026-10-16 by spec revision 31 part B:
an over-functioner takes in proportion to their room above the stress floor (`M11.C.21`, the owner's form; behaviour,
both runs) and, in phase_d, a child's development band opened by the parents' acts (R31.B7). The phase_b run is
scripted and did not move.

**Why the floats are rounded (2026-10-15).** The fingerprints were taken from the exact repr, and CI on Linux failed
them while the Mac passed: the last bits of a float differ between platforms and Python versions (Python 3.12 changed
how `sum` adds floats; the C maths library differs between macOS and Linux). Measured on this Mac, Python 3.11 against
3.12 on the same seeds: no record, field or count differs; 75,496 of 197,685 floats in the phase_d log differ, by at
most 1.7e-11 relative, and none differs once rounded to 10 significant digits. NumPy 2.5.4 against 1.26.4 changed
nothing. Rounding to 8 digits leaves a margin of 100 over that. That a change meant to alter behaviour moves some
value by more than 1e-8 is an assumption, not a measurement; the test below proves only that 1e-6 is seen. Linux was not
measured here. A rounding boundary can still split two platforms by chance; if one does, fewer digits is the fix. It is
not the fix for a small value that is the difference of two near-equal numbers (a different maths library can change
most of its digits; check the floats below 1e-8 first), nor for a comparison that goes the other way on another
platform (the log then differs in structure). Rounding moved both fingerprints and nothing else (the run is the same).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import src.bowen.ensemble.criteria as criteria
from src.bowen.engine.log_records import CollectingEmitter, to_dict
from src.bowen.engine.tick import run_tick
from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_policy_rules
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.policy_source import PolicySource
from src.bowen.scenario.scripted_source import ScriptedSource

REPO = Path(__file__).resolve().parents[2]
FINGERPRINT_DIGITS = 8  # significant digits kept per float; see the module docstring
ZERO_BELOW = 1e-12  # a float this close to zero is written as 0, so 1e-17 and -1e-17 agree
GOLDEN = {
    # (family, seed, weeks): fingerprint
    ("phase_c", 7, 104): "824f90a3025fb2ebf10d6f739112d091056e208b73468fd1eaad16e2dab869d2",
    ("phase_d", 11, 208): "1117c4fd7a0020032b9375234dd3fd7fda666990b56f335e33d2bc821f831230",
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


def _rounded(value):
    if isinstance(value, float):
        return 0.0 if abs(value) < ZERO_BELOW else float(f"{value:.{FINGERPRINT_DIGITS}g}")
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def _fingerprint(bodies) -> str:
    """The SHA-256 of record bodies (`to_dict` forms) in canonical JSON, floats rounded."""
    text = json.dumps([_rounded(b) for b in bodies], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _scaled(value, factor):
    if isinstance(value, float):
        return value * factor
    if isinstance(value, dict):
        return {k: _scaled(v, factor) for k, v in value.items()}
    if isinstance(value, list):
        return [_scaled(v, factor) for v in value]
    return value


def _float_count(value) -> int:
    if isinstance(value, float):
        return 1
    if isinstance(value, dict):
        return sum(_float_count(v) for v in value.values())
    if isinstance(value, list):
        return sum(_float_count(v) for v in value)
    return 0


def test_b1_fingerprint_ignores_last_bits_and_sees_real_changes():
    """Every float scaled by 1 + 1e-12 leaves the fingerprint alone; scaled by 1 + 1e-6, it moves."""
    bodies = [to_dict(r) for r in _records("phase_c", 7, 4)]
    assert _float_count(bodies) > 100, "the short run must log enough floats to test"
    base = _fingerprint(bodies)
    assert _fingerprint([_scaled(b, 1 + 1e-12) for b in bodies]) == base
    assert _fingerprint([_scaled(b, 1 + 1e-6) for b in bodies]) != base


@pytest.mark.parametrize("family,seed,weeks", sorted(GOLDEN))
def test_b1_seeded_run_log_is_unchanged(family, seed, weeks):
    fingerprint = _fingerprint([to_dict(r) for r in _records(family, seed, weeks)])
    assert fingerprint == GOLDEN[(family, seed, weeks)], (
        "the run's log changed; if an engine change was meant to change behaviour, update the fingerprint in the "
        "same commit and say why (see the module docstring)")
