"""Event sizes from the Holmes–Rahe scale (version 2, spec revision 36).
Purpose: test `M1.F.6a`, `M6.3g`, `M7.C.1l` and `M7.D.1a` as directions: the sizes come from config and keep the
         scale's order; a lasting event delivers its whole size over its weeks; grief is sized by the scale and the
         tie; an addition carries a load for its weeks; a symptom's onset loads the bearer and, by bond energy, the
         family. Unit tests of set mechanisms with invented or owner-given constants; none is a finding (`M11.5`).
Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.6a, #M6.3g, #M7.C.1l, #M7.D.1a, #M1.F.6
Tests:   this file
"""

from __future__ import annotations

import dataclasses

import pytest

from src.bowen.engine.event_sizes import sized_intensity
from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.state import new_run_state
from src.bowen.engine.stress_response import register_stressors, stressor_load
from src.bowen.engine.symptoms import accumulate_symptoms, threshold
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

MARTA, RAVI, NADIA = PersonId("marta"), PersonId("ravi"), PersonId("nadia")
PARAMS = engine_params(load_constants())
KINDS = load_event_kinds()
VIS = HouseholdConductanceVisibility(PARAMS.per_hop_fidelity)


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, KINDS)
    state = initialise_run(state, PARAMS)
    for person in state.people.values():
        person.usual_excess = 0.0  # the integrator reads the excess above the usual level
    return state


def life_course_tests():
    """The life-course test module's fixtures (``tests`` is not a package)."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).with_name("test_life_course.py")
    spec = importlib.util.spec_from_file_location("test_life_course_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- M1.F.6a: the table ---------------------------------------------------------------------------


def test_m1f6a_sizes_come_from_config_and_keep_the_scales_ratios():
    sizes = KINDS.sizes
    assert sizes[("GRIEF", "spouse")].lcu == 100 and sizes[("GRIEF", "relative")].lcu == 63
    assert ("INCOME_LOSS", "self") not in sizes  # outside the scale by decision: keeps income_loss_intensity
    order = [sizes[k].lcu for k in (("GRIEF", "spouse"), ("GRIEF", "relative"), ("OWN_ILLNESS", "self"),
                                    ("PARTNERING_LOAD", "self"), ("JOB_LOSS", "self"), ("RETIREMENT_LOAD", "tie"),
                                    ("FAMILY_ILLNESS", "tie"), ("BIRTH_LOAD", "self"), ("JOB_CHANGE", "self"),
                                    ("LAUNCH_LOAD", "self"), ("FIND_WORK_LOAD", "self"), ("MOVE", "self"))]
    assert order == sorted(order, reverse=True)
    spouse, _ = sized_intensity(KINDS, PARAMS, "GRIEF", "spouse", weeks=1)
    relative, _ = sized_intensity(KINDS, PARAMS, "GRIEF", "relative", weeks=1)
    job, _ = sized_intensity(KINDS, PARAMS, "JOB_LOSS", "self", weeks=1)
    assert spouse / relative == pytest.approx(100 / 63) and spouse / job == pytest.approx(100 / 47)
    assert job == pytest.approx(PARAMS.lcu_scale * 47)  # the owner's anchor: 324 units at 6.9 per LCU
    assert job == pytest.approx(324.0, abs=1.0)


def test_m1f6a_a_lasting_event_delivers_its_whole_size_over_its_weeks():
    """Arrival plus `M1.F.6`'s persistence over the following weeks sums to the table's size, so a one-week and a
    twelve-week event of the same LCU weigh the same in all."""
    state = fresh()
    one, _ = sized_intensity(KINDS, PARAMS, "SERIOUS_ILLNESS", "self", weeks=1)
    arrival, weeks = sized_intensity(KINDS, PARAMS, "SERIOUS_ILLNESS", "self")
    assert weeks == 12 and arrival < one
    assert arrival * (1 + PARAMS.stressor_persistence * (weeks - 1)) == pytest.approx(one)
    assert sized_intensity(KINDS, PARAMS, "GRIEF", "spouse", reach=0.5, weeks=1)[0] == pytest.approx(
        sized_intensity(KINDS, PARAMS, "GRIEF", "spouse", weeks=1)[0] / 2)
    with pytest.raises(ValueError):
        sized_intensity(KINDS, PARAMS, "INCOME_LOSS", "self")
    with pytest.raises(ValueError):
        sized_intensity(dataclasses.replace(KINDS, sizes=None), PARAMS, "GRIEF", "spouse")


def test_m1f6a_the_table_is_validated_on_load():
    from src.bowen.io.load import load_event_sizes
    base = load_event_kinds()
    assert set(base.sizes) >= {("OWN_ILLNESS", "self"), ("FAMILY_ILLNESS", "tie"), ("LAUNCH_LOAD", "self")}
    for kind, _ in base.sizes:
        assert kind in base.mechanisms, kind  # every sized kind is a declared event kind
    assert all(s.weeks is None or s.weeks >= 1 for s in base.sizes.values())
    assert callable(load_event_sizes)


# --- M7.D.1a: a symptom's onset is an illness ---------------------------------------------------------


def test_m7d1a_a_symptom_onset_loads_the_bearer_and_the_family():
    """Two arms: an onset in week 0 and none. In the onset arm the bearer carries the own-illness load and each partner
    the family-illness load at the tie's bond energy share, from the next week, for the table's weeks; in the other arm
    nobody does. The partner on the stronger tie carries more."""
    state = fresh()
    marta = state.people[MARTA]
    marta.acute_anxiety = marta.chronic_anxiety + 2 * threshold(marta, PARAMS)
    accumulate_symptoms(state, PARAMS, VIS)
    assert state.arising and all(e.mechanism is Mechanism.EXOGENOUS_STRESSOR and e.timestamp == 1 for e in state.arising)
    kinds = {e.kind: e for e in state.arising if e.kind == "OWN_ILLNESS"}
    assert kinds["OWN_ILLNESS"].targets == (MARTA,) and kinds["OWN_ILLNESS"].duration == 12
    family = {e.targets[0]: e for e in state.arising if e.kind == "FAMILY_ILLNESS"}
    bonds = {o: state.tie_between(MARTA, o).bond_energy for o in family}
    assert family[RAVI].intensity > family[NADIA].intensity > 0 and bonds[RAVI] > bonds[NADIA]
    assert family[RAVI].intensity / family[NADIA].intensity == pytest.approx(bonds[RAVI] / bonds[NADIA])
    own_total = kinds["OWN_ILLNESS"].intensity * (1 + PARAMS.stressor_persistence * 11)
    assert own_total == pytest.approx(PARAMS.lcu_scale * 53)
    state.tick = 1
    due = state.arising_due()
    assert len(due) == 1 + len(family) and state.arising == []
    register_stressors(state, due)
    state.tick = 2
    loaded = {pid for r in stressor_load(state, PARAMS) for pid, _ in r.acute_anxiety}
    assert MARTA in loaded and RAVI in loaded
    quiet = fresh()
    accumulate_symptoms(quiet, PARAMS, VIS)
    assert quiet.arising == []


def test_m7d1a_the_family_illness_load_passes_an_external_agent_by():
    """Two arms on one onset: with Nadia a family member she carries `FAMILY_ILLNESS`; with Nadia an external agent on
    the same interactive tie she carries none (G2 passes the coach by at the release; the load does the same)."""
    from src.bowen.engine.objects import Role

    def family_loads(role):
        state = fresh()
        state.people[NADIA].role = role
        marta = state.people[MARTA]
        marta.acute_anxiety = marta.chronic_anxiety + 2 * threshold(marta, PARAMS)
        accumulate_symptoms(state, PARAMS, VIS)
        return {e.targets[0] for e in state.arising if e.kind == "FAMILY_ILLNESS"}

    assert NADIA in family_loads(Role.MEMBER)
    assert NADIA not in family_loads(Role.EXTERNAL) and RAVI in family_loads(Role.EXTERNAL)


def test_m7d1a_an_arising_stressor_is_never_dropped_unseen():
    state = fresh()
    marta = state.people[MARTA]
    marta.acute_anxiety = marta.chronic_anxiety + 2 * threshold(marta, PARAMS)
    accumulate_symptoms(state, PARAMS, VIS)
    state.tick = 2  # week 1 was skipped
    with pytest.raises(RuntimeError, match="missed their week"):
        state.arising_due()


# --- M6.3g and M7.C.1l: the life course --------------------------------------------------------------


def test_m63g_grief_is_the_scales_size_times_the_tie():
    """A spouse's grief is sized at 100 LCU and a relative's at 63, each times the tie's bond energy share, delivered
    over grief's weeks; grief_weight 0 switches it off (M10.B.5's override)."""
    from src.bowen.engine.life_course import GRIEF, die, due_events

    _ready_to_die = life_course_tests()._ready_to_die
    built = _ready_to_die()
    state, params = built.state, built.params
    bonds = {t.id.other(MARTA): t.bond_energy for t in state.ties_of(MARTA) if t.interactive}
    die(state, MARTA, params)
    state.tick += 1
    grief = {e.targets[0]: e for e in due_events(state) if e.kind == GRIEF}
    spouse = next(f.marital_tie.other(MARTA) for f in state.families() if f.marital_tie and MARTA in f.marital_tie.members())
    for pid, event in grief.items():
        who = "spouse" if pid == spouse else "relative"
        total = params.lcu_scale * state.kinds.sizes[("GRIEF", who)].lcu * bonds[pid] / 100
        assert event.intensity * (1 + params.stressor_persistence * (event.duration - 1)) == pytest.approx(total)
    off = _ready_to_die()
    off.params = dataclasses.replace(off.params, grief_weight=0.0)
    die(off.state, MARTA, off.params)
    off.state.tick += 1
    assert all(e.intensity == 0.0 for e in due_events(off.state) if e.kind == GRIEF)


def test_m7c1l_an_addition_carries_a_load_for_its_weeks():
    """Beside each structural addition a lasting stressor of the table's size and weeks reaches those it lands on:
    a partnering loads the couple, and a launch the child and the parents still in the household the child leaves
    (both checked here); the birth's load is checked by its kind only (its targets are the parents and siblings, as
    `life_course._birth` passes them to the same `_addition_load`)."""
    from src.bowen.engine.life_course import BIRTH_LOAD, LAUNCH_LOAD, PARTNERING_LOAD

    built = life_course_tests().pair_now(1, {})
    pending = {e.kind: e for e in built.state.life.pending}
    assert PARTNERING_LOAD in pending
    load = pending[PARTNERING_LOAD]
    assert load.mechanism is Mechanism.EXOGENOUS_STRESSOR and load.duration == 12 and len(load.targets) == 2
    assert load.intensity * (1 + built.params.stressor_persistence * 11) == pytest.approx(built.params.lcu_scale * 50)
    assert pending["PARTNERING"].targets == load.targets and pending["PARTNERING"].timestamp == load.timestamp
    assert BIRTH_LOAD in built.state.kinds.mechanisms
    lc = life_course_tests()
    launched, _ = lc.parts()
    state, params = launched.state, launched.params
    parents = tuple(state.life.parents[lc.LEO])
    assert all(state.people[p].household_id == state.people[lc.LEO].household_id for p in parents)
    state.people[lc.LEO].age_years = params.launch_age_years - 1
    lc.yearly(state, params)
    launch = [e for e in state.life.pending if e.kind == LAUNCH_LOAD]
    assert len(launch) == 1 and launch[0].targets == tuple(sorted((lc.LEO, *parents)))  # the parents he left
    assert launch[0].intensity * (1 + params.stressor_persistence * 11) == pytest.approx(params.lcu_scale * 29)
