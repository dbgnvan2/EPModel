"""Spec revision 66: anxiety raises the amount of reactive behaviour, conflict consumes capacity, and divorce.

Purpose: unit tests for `M4.D.3c`, `M4.D.3d`, `M4.D.3a` (conflict's layer) and `M7.C.1m`; each names the spec ID it
         verifies.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.3c, #M4.D.3d, #M4.D.3a, #M7.C.1m
Tests:   this file
"""

from __future__ import annotations

import dataclasses

import pytest

from src.bowen.engine.act import Selection
from src.bowen.engine.draws import DrawService
from src.bowen.engine.event_effects import apply_structural_event
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.log_records import DecidedBy
from src.bowen.engine.moves import stress_floor
from src.bowen.engine.observe import observe
from src.bowen.engine.state import new_run_state
from src.bowen.engine.stress_response import consume_conflict
from src.bowen.io.load import CONFIG_DIR, load_constants, load_event_kinds, load_family
from src.bowen.policy.policy import AUTOMATIC, availability, reactive_extras
from src.bowen.scenario.divorce import divorce_events
from src.bowen.scenario.params import engine_params

P = PersonId
PARAMS = engine_params(load_constants())
KINDS = load_event_kinds()
RAVI, MARTA, NADIA, PIA = P("ravi"), P("marta"), P("nadia"), P("pia")


def fresh(params=PARAMS):
    family = load_family(CONFIG_DIR / "family_phase_c.md")
    state = new_run_state(dict(family.people), dict(family.ties), family.family, KINDS)
    return initialise_run(state, params)


def at_excess(state, pid, excess):
    state.people[pid].acute_anxiety = state.people[pid].chronic_anxiety + excess
    return observe(state, pid, PARAMS)


def mean_extras(excess, seeds=200, params=PARAMS):
    total = 0
    for seed in range(seeds):
        state = fresh(params)
        total += len(reactive_extras(at_excess(state, RAVI, excess), KINDS, params, DrawService(seed), frozenset()))
    return total / seeds


# --- M4.D.3c: anxiety raises the amount of reactive behaviour -------------------------------------------------------


def test_m4d3c_a_calm_person_makes_no_extra_acts_and_a_switched_off_register_makes_none():
    """Adversarial: no excess anxiety, or `reactive_extra_max` 0, gives no extra act and takes no draw."""
    state = fresh()
    assert reactive_extras(at_excess(state, RAVI, 0.0), KINDS, PARAMS, DrawService(1), frozenset()) == ()
    off = dataclasses.replace(PARAMS, reactive_extra_max=0.0)
    assert reactive_extras(at_excess(state, RAVI, 500.0), KINDS, off, DrawService(1), frozenset()) == ()


def test_m4d3c_more_anxiety_makes_more_reactive_acts_on_average():
    """Adversarial: the same person at three levels of excess anxiety; the mean count rises and stays at or below the
    register's expected number."""
    low, mid, high = mean_extras(20.0), mean_extras(100.0), mean_extras(400.0)
    assert low < mid < high
    expected = lambda a: PARAMS.reactive_extra_max * a / (a + PARAMS.reactive_half_anxiety)  # noqa: E731
    for got, a in ((low, 20.0), (mid, 100.0), (high, 400.0)):
        assert got <= expected(a) + 0.2


def test_m4d3c_extra_acts_are_automatic_toward_distinct_people_and_not_the_weeks_own_target():
    """Each extra act is an automatic move, in its own slot, toward someone not already acted toward this week."""
    for seed in range(50):
        state = fresh()
        obs = at_excess(state, RAVI, 400.0)
        extras = reactive_extras(obs, KINDS, PARAMS, DrawService(seed), frozenset({MARTA}))
        targets = [s.targets[0] for s in extras]
        assert MARTA not in targets
        assert len(targets) == len(set(targets))
        assert [s.index for s in extras] == list(range(1, len(extras) + 1))
        assert all(s.decided_by is DecidedBy.POLICY for s in extras)
        assert all(s.kind in KINDS.in_channel(AUTOMATIC) for s in extras)


# --- M4.D.3a, M4.D.3d: conflict does not need capacity; it consumes it ----------------------------------------------


def test_m4d3a_conflict_is_in_the_oldest_layer_and_available_at_any_level():
    """Revision 66: CONFLICT sits at layer 0, so a person at functional level 5 has it in full."""
    state = fresh()
    state.people[RAVI].functional_level = 5.0
    assert availability("CONFLICT", observe(state, RAVI, PARAMS), KINDS, PARAMS) == 1.0


def conflict(actor=RAVI, target=MARTA, decided_by=DecidedBy.POLICY):
    return Selection(actor=actor, kind="CONFLICT", targets=(target,), intensity=100.0, decided_by=decided_by)


def test_m4d3d_a_conflict_takes_functioning_from_its_sender_on_the_stress_ledger():
    """Adversarial: a CONFLICT and a DISTANCE from the same person; only the conflict costs, by the register's
    amount, and the stress swing carries exactly what moved."""
    state = fresh()
    person = state.people[RAVI]
    level, swing = person.functional_level, person.stress_swing
    records = consume_conflict(state, (conflict(),), PARAMS)
    assert person.functional_level == pytest.approx(level - PARAMS.conflict_capacity_cost)
    assert person.stress_swing - swing == pytest.approx(-PARAMS.conflict_capacity_cost)
    assert records and records[0].mechanism == "conflict_consumption"
    other = fresh()
    before = other.people[RAVI].functional_level
    distance = dataclasses.replace(conflict(), kind="DISTANCE")
    assert consume_conflict(other, (distance,), PARAMS) == []
    assert other.people[RAVI].functional_level == before


def test_m4d3d_conflict_stops_at_the_stress_floor_and_a_fallback_costs_nothing():
    state = fresh()
    person = state.people[RAVI]
    person.functional_level = stress_floor(person, PARAMS)
    consume_conflict(state, (conflict(),), PARAMS)
    assert person.functional_level == pytest.approx(stress_floor(person, PARAMS))
    other = fresh()
    before = other.people[RAVI].functional_level
    consume_conflict(other, (conflict(decided_by=DecidedBy.FALLBACK),), PARAMS)
    assert other.people[RAVI].functional_level == before


# --- M7.C.1m: divorce ------------------------------------------------------------------------------------------------


def test_m7c1m_divorce_moves_the_leaver_and_keeps_the_tie():
    """The separation gives the leaver a household of their own and leaves every tie, its bond energy and its
    interactivity as it was; the family's marital tie still names the pair."""
    state = fresh()
    events = divorce_events(state, PARAMS, 0, RAVI)
    tie = state.ties[TieId.of(RAVI, MARTA)]
    bond, interactive = tie.bond_energy, tie.interactive
    separation = next(e for e in events if e.kind == "SEPARATION")
    apply_structural_event(state, separation, PARAMS)
    assert state.people[RAVI].household_id == "ravi_apart"
    assert state.people[MARTA].household_id != "ravi_apart"
    assert tie.bond_energy == bond and tie.interactive == interactive
    assert state.family.marital_tie == TieId.of(RAVI, MARTA)


def test_m7c1m_divorce_is_sized_from_holmes_rahe_and_reaches_the_children_by_tie():
    """Each spouse carries 73 life change units over the event's weeks; each child carries the same at their tie's
    bond energy share, once per parent."""
    state = fresh()
    events = [e for e in divorce_events(state, PARAMS, 0, RAVI) if e.kind == "DIVORCE"]

    def total(e):
        return e.intensity * (1.0 + PARAMS.stressor_persistence * (e.duration - 1))

    spouses = [e for e in events if e.targets[0] in (RAVI, MARTA)]
    assert len(spouses) == 2
    for e in spouses:
        assert total(e) == pytest.approx(PARAMS.lcu_scale * 73)
    for child in (NADIA, PIA):
        loads = [e for e in events if e.targets[0] == child]
        assert len(loads) == 2
        bonds = sorted(state.ties[TieId.of(parent, child)].bond_energy for parent in (RAVI, MARTA))
        assert sorted(total(e) for e in loads) == pytest.approx([PARAMS.lcu_scale * 73 * b / 100 for b in bonds])


def test_m7c1m_divorce_refuses_a_non_spouse_and_scales_only_by_its_argument():
    state = fresh()
    with pytest.raises(ValueError):
        divorce_events(state, PARAMS, 0, NADIA)
    plain = divorce_events(state, PARAMS, 0, RAVI)
    double = divorce_events(state, PARAMS, 0, RAVI, scale=2.0)
    for a, b in zip(plain, double):
        assert b.intensity == pytest.approx(2.0 * a.intensity)


# --- Review guards, 2026-11-19 ----------------------------------------------------------------------------------------


def test_m4d3c_an_excluded_outcome_is_never_an_extra_act():
    """Adversarial: a criterion's withheld or held-open outcomes (``excluded``) are never drawn as an extra act, so a
    redrawn actor keeps their extras without the act the arm withholds (review M2)."""
    for seed in range(50):
        state = fresh()
        obs = at_excess(state, RAVI, 400.0)
        everyone = reactive_extras(obs, KINDS, PARAMS, DrawService(seed), frozenset())
        banned = frozenset(f"{s.kind}>{s.targets[0]}" for s in everyone)
        again = reactive_extras(obs, KINDS, PARAMS, DrawService(seed), frozenset(), banned)
        assert not {f"{s.kind}>{s.targets[0]}" for s in again} & banned


def test_m4d3c_a_scripted_actor_gets_no_extra_acts():
    """A scripted selection is the actor's whole week: the policy adds no extra act to it (`M4.D.3c`)."""
    from src.bowen.io.load import load_policy_rules
    from src.bowen.scenario.policy_source import PolicySource
    from src.bowen.scenario.scripted_source import ScriptedSource

    scripted = Selection(actor=RAVI, kind="CONFLICT", targets=(MARTA,), intensity=1.0, decided_by=DecidedBy.SCRIPTED)
    source = PolicySource(ScriptedSource("guard", 2, (), ((0, scripted),)), PARAMS, load_policy_rules())
    for seed in range(20):
        state = fresh()
        state.draws = DrawService(seed)
        state.people[RAVI].acute_anxiety = state.people[RAVI].chronic_anxiety + 400.0
        chosen = source.selections(0, tuple(sorted(state.people)), state)
        assert [s for s in chosen if s.actor == RAVI] == [scripted]


def test_m11_readouts_count_only_the_weeks_own_choice():
    """Adversarial: an extra reactive act (slot 1) is not a selection in the fallback rate's denominator, nor in
    `M11.C.41`'s reactive-over-chance mean (review M1, L1)."""
    from src.bowen.engine.events import EventId
    from src.bowen.engine.log_records import SelectionRecord
    from src.bowen.ensemble.criteria import bookkeeping, reactive_over_chance

    own = SelectionRecord(tick=0, actor=RAVI, decided_by=DecidedBy.POLICY, event_id=EventId(0, "ravi", 0),
                          legal_set=("STAY-IN-CONTACT>marta", "CONFLICT>marta"))
    extra = SelectionRecord(tick=0, actor=RAVI, decided_by=DecidedBy.POLICY, event_id=EventId(0, "ravi", 1),
                            legal_set=("CONFLICT>nadia",))
    _, _, selections = bookkeeping([own, extra])
    assert selections["ravi"] == 1
    assert reactive_over_chance([own, extra]) == reactive_over_chance([own])


def test_m11g_the_household_readout_does_not_move_with_which_spouse_leaves():
    """Adversarial: after a separation the evaluation's nuclear household is the declared one, whichever spouse left
    (review L2)."""
    from src.bowen.readouts.evaluation import _nuclear

    for leaver in (MARTA, RAVI):
        state = fresh()
        declared = state.family.household
        for event in divorce_events(state, PARAMS, 0, leaver)[:1]:
            apply_structural_event(state, event, PARAMS)
        assert state.people[leaver].household_id != declared
        assert _nuclear(state, tuple(sorted(state.family.marital_tie.members()))) == declared
