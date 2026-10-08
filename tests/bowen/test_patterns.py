"""Pattern readouts (Phase C step 11, plan D6).

Purpose: test each declared readout on a case it must read and a contrast it must not, the
         refusal of an undeclared pattern, and that the code and the declaration document agree.
Spec:    docs/bowen_agent_model_spec_v2.md#M5.A.1a
Tests:   this file
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.moves import DEFAULT_AREA
from src.bowen.engine.state import new_run_state
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.readouts.patterns import PATTERNS, ReadoutParams, UndeclaredPattern, read_pattern
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, NADIA = P("ravi"), P("marta"), P("nadia")
MARITAL = TieId.of(RAVI, MARTA)
TRIAD = TriangleId.of(MARTA, NADIA, RAVI)
CONSTANTS = load_constants()
PARAMS = engine_params(CONSTANTS)
RP = ReadoutParams.from_constants(CONSTANTS)
REPO = Path(__file__).resolve().parents[2]


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, PARAMS)


def acts(state, *spec):
    for i, (kind, sender, target) in enumerate(spec):
        state.store.record_event(Event(
            id=EventId(state.tick, str(sender.value), i + 10 * len(state.store.events())), kind=kind,
            mechanism=Mechanism.MOVE, sender=sender, targets=(target,), intensity=50.0, timestamp=state.tick,
            duration=1, exogenous=False, source_position=SourcePosition.NONE, channel=Channel.SCRIPTED))


def read(name, state, subject=MARITAL, records=()):
    return read_pattern(name, state, subject, PARAMS, RP, records)


def test_m5a1a_conflict_needs_both_sides():
    both, one = fresh(), fresh()
    acts(both, ("CONFLICT", RAVI, MARTA), ("CONFLICT", RAVI, MARTA), ("CONFLICT", MARTA, RAVI), ("CONFLICT", MARTA, RAVI))
    acts(one, *[("CONFLICT", RAVI, MARTA)] * 4)
    assert read("conflict", both) and not read("conflict", one)


def test_m5a1a_over_underfunctioning_needs_a_hardened_pole_and_matching_acts():
    held, flat = fresh(), fresh()
    for state in (held, flat):
        acts(state, ("OVERFUNCTION", MARTA, RAVI))
    held.ties[MARITAL].functioning_habit[DEFAULT_AREA] = 0.8  # marta (the tie's first member) over
    flat.ties[MARITAL].functioning_habit[DEFAULT_AREA] = 0.1
    assert read("over_underfunctioning", held) and not read("over_underfunctioning", flat)
    mismatched = fresh()
    mismatched.ties[MARITAL].functioning_habit[DEFAULT_AREA] = 0.8
    acts(mismatched, ("OVERFUNCTION", RAVI, MARTA))  # the under side taking charge does not match
    assert not read("over_underfunctioning", mismatched)


def test_m5a1a_distance_needs_withdrawal_and_too_little_contact():
    withdrawn, approached = fresh(), fresh()
    for state in (withdrawn, approached):
        for m in MARITAL.members():
            state.ties[MARITAL].felt_contact[m] = 0.0
    acts(withdrawn, ("DISTANCE", MARTA, RAVI), ("DISTANCE", MARTA, RAVI))
    acts(approached, ("DISTANCE", MARTA, RAVI), ("PURSUE", RAVI, MARTA), ("PURSUE", RAVI, MARTA))
    assert read("distance", withdrawn) and not read("distance", approached)
    still_close = fresh()
    for m in MARITAL.members():
        still_close.ties[MARITAL].felt_contact[m] = 1.0  # withdrawing, but not yet short of contact
    acts(still_close, ("DISTANCE", MARTA, RAVI), ("DISTANCE", MARTA, RAVI))
    assert not read("distance", still_close)


def test_m5a1a_cutoff_needs_a_severed_tie_held_for_the_window():
    state = fresh()
    ana_bruno = TieId.of(P("ana"), P("bruno"))
    assert read("cutoff", state, ana_bruno) and not read("cutoff", state, MARITAL)
    just_cut = fresh()
    just_cut.ties[MARITAL].interactive = False
    acts(just_cut, ("CUTOFF", RAVI, MARTA))
    assert not read("cutoff", just_cut)  # not yet held for W


def recompute(tick, active, inside):
    return EffectRecord(tick, "triangle_recompute", None, triangles=(
        (TRIAD, "active", "true" if active else "false"), (TRIAD, "inside_pair", inside if active else "none")))


def test_m5a1a_fixed_triangle_is_the_same_outsider_across_activations():
    same = [recompute(t, t % 2 == 0, "nadia~ravi") for t in range(8)]
    varied = [recompute(t, t % 2 == 0, "nadia~ravi" if t < 4 else "marta~nadia") for t in range(8)]
    assert read("fixed_triangle", fresh(), TRIAD, same) and not read("fixed_triangle", fresh(), TRIAD, varied)


def test_m5a1a_an_undeclared_pattern_is_refused():
    with pytest.raises(UndeclaredPattern):
        read("projection", fresh())


def test_m5a1a_readouts_doc_and_code_agree():
    doc = (REPO / "docs/readouts_phase_c.md").read_text()
    declared = set(re.findall(r"^\| `([a-z_]+)` \| a (?:tie|triangle) \|", doc, re.M))
    assert declared == set(PATTERNS)
