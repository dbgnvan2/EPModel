"""Pattern readouts (Phase C step 11, plan D6).

Purpose: test each declared readout on a case it must read and a contrast it must not, the
         refusal of an undeclared pattern, that the code and the declaration document agree, and
         that a pattern holds until a new one emerges (owner decision D2, 2026-10-10).
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
from src.bowen.engine.state import new_run_state
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.readouts.patterns import (
    PATTERNS, Reading, ReadoutParams, UndeclaredPattern, current_patterns, cutoff, distance, fixed_triangle,
    last_projection_act, over_underfunctioning, read_pattern, tie_readings,
)
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, NADIA = P("ravi"), P("marta"), P("nadia")
MARITAL = TieId.of(RAVI, MARTA)
TRIAD = TriangleId.of(MARTA, NADIA, RAVI)
MARTA_NADIA, RAVI_NADIA = TieId.of(MARTA, NADIA), TieId.of(RAVI, NADIA)
PARENTS = {MARTA_NADIA: MARTA, RAVI_NADIA: RAVI}
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


def read(name, state, subject=MARITAL, records=(), parents=None):
    return read_pattern(name, state, subject, PARAMS, RP, records, parents)


def at(state, tick, *spec):
    state.tick = tick
    acts(state, *spec)
    return state


def test_m5a1a_window_and_threshold_are_the_owners():
    assert (RP.window, RP.min_acts) == (4, 3)  # 3 or more times in a month (owner, 2026-10-10)


def test_m5a1a_conflict_needs_three_acts_in_four_weeks():
    two = at(fresh(), 2, ("CONFLICT", RAVI, MARTA), ("CONFLICT", MARTA, RAVI))
    three = at(fresh(), 2, ("CONFLICT", RAVI, MARTA), ("CONFLICT", MARTA, RAVI), ("CONFLICT", RAVI, MARTA))
    one_sided = at(fresh(), 2, *[("CONFLICT", RAVI, MARTA)] * 4)
    assert read("conflict", three) and not read("conflict", two) and not read("conflict", one_sided)
    old = at(fresh(), 1, ("CONFLICT", RAVI, MARTA), ("CONFLICT", MARTA, RAVI), ("CONFLICT", RAVI, MARTA))
    old.tick = 5  # the acts are now outside the 4-week window
    assert not read("conflict", old)


def test_m5a1a_distance_needs_withdrawal_and_too_little_contact():
    withdrawn, approached, still_close = fresh(), fresh(), fresh()
    for state, contact in ((withdrawn, 0.0), (approached, 0.0), (still_close, 1.0)):
        for m in MARITAL.members():
            state.ties[MARITAL].felt_contact[m] = contact
    at(withdrawn, 2, *[("DISTANCE", MARTA, RAVI)] * 3)
    at(approached, 2, *[("DISTANCE", MARTA, RAVI)] * 3, *[("PURSUE", RAVI, MARTA)] * 3)
    at(still_close, 2, *[("DISTANCE", MARTA, RAVI)] * 3)
    assert read("distance", withdrawn) and not read("distance", approached) and not read("distance", still_close)


def test_m5a1a_no_interaction_reads_distance_not_cutoff():
    """The owner's correction: a severed, silent tie is distance; cutoff is rejection, not absence."""
    state = fresh()
    state.ties[MARITAL].interactive = False
    state.tick = 8
    assert read("distance", state) and not read("cutoff", state)


def test_m5a1a_distance_level_rises_with_weeks_without_interaction():
    early, late = fresh(), fresh()
    early.tick, late.tick = 5, 20
    a, b = (distance(s, MARITAL, PARAMS, RP) for s in (early, late))
    assert a.silent and b.silent and b.count > a.count >= RP.window


def test_m5a1a_cutoff_needs_refusals_of_approaches():
    refused = fresh()
    for week in (1, 2, 3):
        at(refused, week, ("PURSUE", RAVI, MARTA), ("CUTOFF", MARTA, RAVI))
    unprompted = at(fresh(), 3, *[("CUTOFF", MARTA, RAVI)] * 3)
    reading = cutoff(refused, MARITAL, RP)
    assert reading is not None and reading.toward == MARTA
    assert not read("cutoff", unprompted)


def test_m5a1a_over_underfunctioning_needs_three_acts_one_way():
    one_way = at(fresh(), 2, ("OVERFUNCTION", MARTA, RAVI), ("OVERFUNCTION", MARTA, RAVI), ("UNDERFUNCTION", RAVI, MARTA))
    reading = over_underfunctioning(one_way, MARITAL, RP)
    assert reading is not None and reading.toward == MARTA
    even = at(fresh(), 2, *[("OVERFUNCTION", MARTA, RAVI)] * 3, *[("OVERFUNCTION", RAVI, MARTA)] * 3)
    assert not read("over_underfunctioning", even)


def test_m5a1a_pursuit_reads_as_fusion_over_under_or_projection():
    one_way = at(fresh(), 2, *[("PURSUE", RAVI, MARTA)] * 3)
    found = {r.pattern: r for r in tie_readings(one_way, MARITAL, PARAMS, RP)}
    assert set(found) == {"over_underfunctioning"} and found["over_underfunctioning"].toward == MARTA  # ravi goes along
    both = at(fresh(), 2, ("PURSUE", RAVI, MARTA), ("PURSUE", MARTA, RAVI), ("PURSUE", RAVI, MARTA))
    assert read("fusion", both)
    parent = at(fresh(), 2, *[("PURSUE", MARTA, NADIA)] * 3)
    found = {r.pattern for r in tie_readings(parent, MARTA_NADIA, PARAMS, RP, MARTA)}
    assert found == {"projection"}


def test_m5a1a_projection_reads_parent_to_child_acts():
    state = at(fresh(), 2, ("OVERFUNCTION", RAVI, NADIA), ("TRIANGLE", RAVI, NADIA), ("PURSUE", RAVI, NADIA))
    assert read("projection", state, RAVI_NADIA, parents=PARENTS)
    child = at(fresh(), 2, *[("PURSUE", NADIA, RAVI)] * 3)  # the child's acts are not projection
    assert not read("projection", child, RAVI_NADIA, parents=PARENTS)
    assert not read("projection", state, RAVI_NADIA)  # no parent known, no projection


def recompute(tick, active, inside):
    return EffectRecord(tick, "triangle_recompute", None, triangles=(
        (TRIAD, "active", "true" if active else "false"), (TRIAD, "inside_pair", inside if active else "none")))


def test_m5a1a_fixed_triangle_is_the_same_outsider_across_activations():
    same = [recompute(t, t % 2 == 0, "nadia~ravi") for t in range(8)]
    varied = [recompute(t, t % 2 == 0, "nadia~ravi" if t < 4 else "marta~nadia") for t in range(8)]
    assert read("fixed_triangle", fresh(), TRIAD, same) and not read("fixed_triangle", fresh(), TRIAD, varied)
    assert fixed_triangle(same, TRIAD, RP).toward == MARTA


def test_m5a1a_an_undeclared_pattern_is_refused():
    with pytest.raises(UndeclaredPattern):
        read("pursuit", fresh())


def test_m5a1a_readouts_doc_and_code_agree():
    doc = (REPO / "docs/readouts_phase_c.md").read_text()
    declared = set(re.findall(r"^\| `([a-z_]+)` \| a (?:tie|triangle) \|", doc, re.M))
    assert declared == set(PATTERNS)


def R(pattern, count=3, first=0, toward=None, silent=False, subject=MARITAL):
    return Reading(pattern, subject, count, first, toward, silent)


def test_m5a1a_pattern_holds_until_a_new_one_emerges():
    held = current_patterns({}, [R("conflict")])
    held = current_patterns(held, [])  # a quiet read: conflict still holds
    assert held[MARITAL].pattern == "conflict"
    held = current_patterns(held, [R("over_underfunctioning", toward=MARTA)])
    assert held[MARITAL].pattern == "over_underfunctioning"
    held = current_patterns(held, [R("over_underfunctioning", toward=RAVI)])  # reversed: a new pattern
    assert held[MARITAL].toward == RAVI


def test_m5a1a_quiet_block_keeps_the_current_pattern():
    held = current_patterns({}, [R("conflict", first=1)])
    for _ in range(5):
        held = current_patterns(held, [])
    assert held[MARITAL] == R("conflict", first=1)


def test_m5a1a_continuing_pattern_keeps_its_start():
    held = current_patterns({}, [R("distance", count=4, first=0, silent=True)])
    held = current_patterns(held, [R("distance", count=8, first=4, silent=True)])
    assert (held[MARITAL].count, held[MARITAL].first) == (8, 0)


def test_m5a1a_silence_does_not_replace_a_cutoff():
    held = current_patterns({}, [R("cutoff", toward=MARTA)])
    held = current_patterns(held, [R("distance", count=6, silent=True)])
    assert held[MARITAL].pattern == "cutoff"
    held = current_patterns(held, [R("distance", count=3)])  # distance by withdrawal acts is new
    assert held[MARITAL].pattern == "distance"


def test_m5a1a_tie_break_is_more_acts_then_earlier_start():
    assert current_patterns({}, [R("conflict", 3), R("distance", 4)])[MARITAL].pattern == "distance"
    assert current_patterns({}, [R("conflict", 3, first=2), R("fusion", 3, first=1)])[MARITAL].pattern == "fusion"


def test_m5a1a_lopsided_pursuit_is_over_underfunctioning_not_fusion():
    """The owner, 2026-10-10: A pursues B five times, B pursues A once — over/underfunctioning, B over."""
    lopsided = at(fresh(), 2, *[("PURSUE", RAVI, MARTA)] * 5, ("PURSUE", MARTA, RAVI))
    found = {r.pattern: r for r in tie_readings(lopsided, MARITAL, PARAMS, RP)}
    assert "fusion" not in found and found["over_underfunctioning"].toward == MARTA
    balanced = at(fresh(), 2, *[("PURSUE", RAVI, MARTA)] * 2, *[("PURSUE", MARTA, RAVI)] * 2)
    assert {r.pattern for r in tie_readings(balanced, MARITAL, PARAMS, RP)} == {"fusion"}


def test_m5a1a_last_projection_act_counts_single_acts():
    state = at(fresh(), 5, ("OVERFUNCTION", RAVI, NADIA))
    assert last_projection_act(state, RAVI_NADIA, RAVI) == 5
    assert last_projection_act(state, RAVI_NADIA, None) is None
    assert last_projection_act(at(fresh(), 5, ("OVERFUNCTION", NADIA, RAVI)), RAVI_NADIA, RAVI) is None


def test_m5a1a_projection_holds_until_twelve_weeks_without_evidence():
    """The owner, 2026-10-10: projection stays until 12 turns show it has changed or stopped."""
    hold = RP.projection_hold
    assert hold == 12
    proj = Reading("projection", RAVI_NADIA, 3, 0, toward=RAVI)
    louder = Reading("conflict", RAVI_NADIA, 5, 8)
    held = current_patterns({}, [proj])
    # a louder pattern, 8 weeks after the last projection act: the projection holds
    kept = current_patterns(held, [louder], now=10, projection_last={RAVI_NADIA: 2}, projection_hold=hold)
    assert kept[RAVI_NADIA].pattern == "projection"
    # the same, 12 weeks after the last projection act: the new pattern replaces it
    changed = current_patterns(held, [louder], now=14, projection_last={RAVI_NADIA: 2}, projection_hold=hold)
    assert changed[RAVI_NADIA].pattern == "conflict"
    # 12 weeks without a projection act and nothing new read: the projection has stopped
    stopped = current_patterns(held, [], now=14, projection_last={RAVI_NADIA: 2}, projection_hold=hold)
    assert RAVI_NADIA not in stopped
    # read again inside the hold alongside a louder pattern: it continues
    again = current_patterns(held, [louder, Reading("projection", RAVI_NADIA, 3, 9, toward=RAVI)], now=10,
                             projection_last={RAVI_NADIA: 9}, projection_hold=hold)
    assert again[RAVI_NADIA].pattern == "projection" and again[RAVI_NADIA].first == 0

