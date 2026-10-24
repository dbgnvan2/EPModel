"""The life course — spec revision 28, Phase D part II (plan §4.3, steps L0–L8).

Purpose: pin each part of the life course to its requirement: ages and acting, a child's chronic anxiety and
         development, launch, partners and births from settings, death and its disposition, lost income and finding
         work, and the coach's career.
Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1, #M6.3, #M7.B.1a, #M7.E.1f, #M2.A.0j, #M2.A.0k
Tests:   this file
"""

from __future__ import annotations

import dataclasses
from collections import deque
from pathlib import Path

import pytest

from src.bowen.engine.moves import DEFAULT_AREA, _pole_of, relax_pseudo_self

from src.bowen.engine.events import Mechanism, Role as DeliveryRole
from src.bowen.engine.event_sizes import sized_intensity
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId, child_id
from src.bowen.engine.life_course import (
    GRIEF, age_at_death, die, due_events, find_work, total_anxiety, yearly, _birth, _childhood,
)
from src.bowen.engine.log_records import CollectingEmitter, Tee, serialize
from src.bowen.engine.objects import Relationship, Role, Sink
from src.bowen.engine.slow_tick import Week
from src.bowen.engine.state import canonical_state
from src.bowen.engine.tick import run_tick
from src.bowen.io.sinks import JsonlFileSink
from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_policy_rules, read_text
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.config_parse import ConfigError
from src.bowen.scenario.family import build_family
from src.bowen.scenario.policy_source import PolicySource
from src.bowen.scenario.scripted_source import ScriptedSource

REPO = Path(__file__).resolve().parents[2]
PHASE_D = REPO / "config" / "bowen" / "family_phase_d.md"
PARTNERS = REPO / "config" / "bowen" / "partners_phase_d.md"
P = PersonId
RAVI, MARTA, LEO, NADIA, PIA, HALIM, ANA, TEODOR = (P(x) for x in (
    "ravi", "marta", "leo", "nadia", "pia", "halim", "ana", "teodor"))


def parts(seed: int = 3, weeks: int = 52 * 40, family=None):
    constants, kinds = load_constants(), load_event_kinds()
    script = ScriptedSource("life", weeks, (), ())
    built = assemble(constants, kinds, family or load_family(PHASE_D), script, seed=seed)
    return built, PolicySource(script, built.params, load_policy_rules())


def run_weeks(built, source, weeks: int) -> CollectingEmitter:
    emitter = CollectingEmitter()
    for _ in range(weeks):
        run_tick(built.state, source, built.params, built.visibility, built.activation, emitter)
    return emitter


@pytest.fixture(scope="module")
def paired():
    """Seed 1 to the week before Leo and Sara's first child: Leo pairs at 28, in year 19 (shared, read-only)."""
    built, source = parts(seed=1)
    emitter = run_weeks(built, source, 52 * 21 - 1)
    return built, emitter


def pair_now(seed: int, params_override: dict | None = None):
    """A fresh run with Leo at the year before his pairing age, then the slow tick that pairs him."""
    built, _ = parts(seed=seed)
    if params_override:
        built.params = dataclasses.replace(built.params, **params_override)
    built.state.people[LEO].age_years = built.state.life.settings.partners[0].pair_age - 1
    yearly(built.state, built.params)
    return built


def history(state, pid, functional: float, load: float = 0.0, weeks: int = 52):
    state.estimator_history[pid] = deque(Week(t, functional, load, False, ()) for t in range(weeks))




# --- L0: topology that can grow --------------------------------------------------------------------------------


def test_m7c1f_age_advances_on_the_slow_tick():
    built, source = parts()
    run_weeks(built, source, 52)
    people = built.state.people
    assert (people[PIA].age_years, people[RAVI].age_years, people[HALIM].age_years) == (2, 31, 36)


def test_m7c1h_a_new_tie_completes_its_triangles():
    built, _ = parts()
    state = built.state
    tie = Relationship(id=TieId.of(LEO, P("iris")), conductance=0.5, bond_energy=10, latency=1)
    before = set(state.triangles)
    added = state.add_tie(tie)
    assert set(added) == set(state.triangles) - before
    assert all({LEO, P("iris")} <= set(t.members) for t in added)
    assert TriangleId.of(LEO, P("iris"), MARTA) in added  # Marta is tied to both
    with pytest.raises(ValueError, match="already in the field"):
        state.add_tie(tie)


def test_m1a20_a_birth_identifier_is_stable_across_arms():
    """Two arms that differ in their parameters and seed give the couple's children the same identifiers."""
    ids = []
    for seed, override in ((1, {}), (2, {"grief_weight": 0.0, "new_tie_investment_share": 0.5})):
        built = pair_now(seed, override)
        for _ in range(5):  # the children are born two and five years after pairing
            yearly(built.state, built.params)
        ids.append(sorted(p for p in built.state.people if "#" in p.value))
    assert ids[0] == ids[1] == [child_id(LEO, P("sara"), 1), child_id(LEO, P("sara"), 2)]


def test_m1d1_each_household_couple_has_its_own_family(paired):
    built, _ = paired
    state = built.state
    family = state.other_families["leo_sara"]
    assert family.marital_tie == TieId.of(LEO, P("sara")) and family.household == "leo_sara"
    assert state.family_of(LEO) is family and state.family_of(RAVI) is state.family
    assert [f.household for f in state.families()] == ["ravi_marta", "leo_sara"]


# --- L1: the family data ------------------------------------------------------------------------------------------


def test_m2a0j_ages_come_from_the_three_settings():
    family = load_family(PHASE_D)
    ages = {p.value: family.ages_years[p] for p in family.people}
    assert ages == {"teodor": 55, "ana": 52, "viktor": 52, "sofia": 50, "ravi": 30, "marta": 28, "dario": 27,
                    "iris": 25, "leo": 9, "nadia": 4, "pia": 1, "halim": 35}


def test_m2a0j_changing_a_setting_changes_only_ages():
    text, partners = read_text(PHASE_D), read_text(PARTNERS)
    base = build_family(text, load_constants(), partners_text=partners)
    moved = build_family(text.replace("start_age_g2: 30", "start_age_g2: 33"), load_constants(), partners_text=partners)
    for pid in base.people:
        shift = 3 if base.generations[pid] == 2 else 0
        assert moved.ages_years[pid] == base.ages_years[pid] + shift
        assert dataclasses.replace(moved.people[pid], age_years=None) == dataclasses.replace(base.people[pid], age_years=None)
    assert moved.ties == base.ties


def test_m2a0j_a_gap_must_put_the_anchor_at_zero():
    text = read_text(PHASE_D).replace("| `ravi` | Ravi | 2 | 0 |", "| `ravi` | Ravi | 2 | -1 |")
    with pytest.raises(ConfigError, match="anchor at 0"):
        build_family(text, load_constants(), partners_text=read_text(PARTNERS))


def test_m2b2_the_coach_is_tied_to_the_coachee_only():
    family = load_family(PHASE_D)
    assert [t for t in family.ties if HALIM in t.members()] == [TieId.of(HALIM, RAVI)]
    text = read_text(PHASE_D).replace(
        "| `halim` | `ravi` | professional |", "| `halim` | `marta` | professional | 0.3 | 10 | 1 | ordinary | yes |\n"
        "| `halim` | `ravi` | professional |")
    with pytest.raises(ConfigError, match="tied to the coachee only"):
        build_family(text, load_constants(), partners_text=read_text(PARTNERS))


def test_m2a0a_a_child_carries_no_supplied_chronic_anxiety():
    text = read_text(PHASE_D).replace("| `pia` | Pia | 3 | 0 | female | ravi_marta | — | — | — |",
                                      "| `pia` | Pia | 3 | 0 | female | ravi_marta | — | — | 40 |")
    with pytest.raises(ConfigError, match="derived below the fixation age"):
        build_family(text, load_constants(), partners_text=read_text(PARTNERS))


# --- L2: who acts -------------------------------------------------------------------------------------------------


def test_m7c1f_a_child_under_ten_never_selects():
    built, source = parts()
    active = built.activation.active(0, built.state.people.values())
    assert NADIA not in active and PIA not in active and LEO not in active and RAVI in active
    emitter = run_weeks(built, source, 52)
    actors = {r.actor for r in emitter.records if type(r).__name__ == "SelectionRecord"}
    assert not actors & {LEO, NADIA, PIA}
    run_weeks(built, source, 1)  # Leo turned 10 at the slow tick
    assert LEO in built.activation.active(built.state.tick, built.state.people.values())


def test_m7c1f_a_child_under_ten_still_witnesses():
    built, source = parts()
    run_weeks(built, source, 26)
    for child in (NADIA, PIA):
        assert built.state.store.delivered_to(child, DeliveryRole.WITNESS)


# --- L3: chronic anxiety before 12 ------------------------------------------------------------------------------


def test_m7b1a_child_chronic_reads_parents_and_witnessed_history():
    built, _ = parts()
    state, params = built.state, built.params
    for pid in (RAVI, MARTA):
        history(state, pid, 40.0)
    history(state, PIA, 39.5, load=10.0)
    _childhood(state, params)
    parents = (state.people[RAVI].chronic_anxiety + state.people[MARTA].chronic_anxiety) / 2
    assert state.people[PIA].chronic_anxiety == pytest.approx(parents + params.witnessed_gain * 10.0)


def test_m7b1a_chronic_is_fixed_at_the_fixation_age():
    built, _ = parts()
    state, params = built.state, built.params
    state.people[LEO].age_years = params.chronic_anxiety_fixation_age_years + 1
    before = state.people[LEO].chronic_anxiety
    history(state, LEO, 39.5, load=30.0)
    _childhood(state, params)
    assert state.people[LEO].chronic_anxiety == before


def test_m7b1a_chronic_integrates_every_childhood_year():
    """Revision 29 (the owner: "chronic Anxiety builds overchildhood"). Adversarial: a hard early childhood followed by a
    calm year at 11 must end higher than a childhood calm throughout; the year at 11 alone would read them the same."""
    def chronic(loads):
        built, _ = parts()
        state, params = built.state, built.params
        for pid in (RAVI, MARTA):
            history(state, pid, 40.0)
        for load in loads:
            history(state, PIA, 39.5, load=load)
            _childhood(state, params)
        return state.people[PIA].chronic_anxiety, state, params

    hard, state, params = chronic([20.0, 20.0, 0.0])
    calm, _, _ = chronic([0.0, 0.0, 0.0])
    parents = (state.people[RAVI].chronic_anxiety + state.people[MARTA].chronic_anxiety) / 2
    assert hard == pytest.approx(parents + params.witnessed_gain * 40.0 / 3)
    assert hard > calm == pytest.approx(parents)


def test_m7b1a_a_child_present_at_t0_counts_from_t0():
    """Leo is 9 at t0: only the years the run observed are averaged, three of them by his twelfth birthday."""
    built, source = parts()
    run_weeks(built, source, 52 * 3)
    assert built.state.people[LEO].age_years == built.params.chronic_anxiety_fixation_age_years
    assert len(built.state.life.childhood_loads[LEO]) == 3


# --- L4: a child's development ----------------------------------------------------------------------------------


def _developing(age: int):
    built, _ = parts()
    state, params = built.state, built.params
    state.people[NADIA].age_years = age
    state.life.anchors[NADIA] = 39.5
    return state, params


def test_m7e1f_level_held_at_parents_mean_before_five():
    state, params = _developing(3)
    state.people[RAVI].basic_level = 41.0
    _childhood(state, params)
    assert state.people[NADIA].basic_level == pytest.approx((41.0 + 40.0) / 2)


def test_m7e1f_parents_functioning_up_raises_the_level():
    up, params = _developing(8)
    down, _ = _developing(8)
    for state, f in ((up, 45.0), (down, 34.0)):
        for pid in (RAVI, MARTA):
            history(state, pid, f)
        history(state, NADIA, 39.5)
        _childhood(state, params)
    assert up.people[NADIA].basic_level > 39.5 > down.people[NADIA].basic_level


def test_m7e1f_level_never_leaves_two_percent_band():
    """Adversarial: years of parents functioning far above their level must stop at the band, not run on."""
    state, params = _developing(6)
    for pid in (RAVI, MARTA, NADIA):
        history(state, pid, 90.0)
    for _ in range(30):
        state.people[NADIA].age_years = 10
        _childhood(state, params)
    assert state.people[NADIA].basic_level == pytest.approx(39.5 * (1 + params.development_band))


def _band_after(acts: list | None, functioning: float, parents_functioning: float | None = None):
    """Nadia's level after 30 development years at 10 with the given parental-act counts ([all, projection,
    supportive]; None for none received) and every year's functioning fixed."""
    state, params = _developing(6)
    if acts is not None:
        state.life.parental_acts[NADIA] = list(acts)
    for pid in (RAVI, MARTA):
        history(state, pid, functioning if parents_functioning is None else parents_functioning)
    history(state, NADIA, functioning)
    for _ in range(30):
        state.people[NADIA].age_years = 10
        _childhood(state, params)
    return state.people[NADIA].basic_level, params


def test_m7e1f_no_parental_acts_keeps_two_percent_band():
    """R31.B7: with no parental acts to the child, the band stays ±2% (`development_band`), both ways."""
    high, params = _band_after(None, 90.0)
    low, _ = _band_after(None, 5.0)
    assert high == pytest.approx(39.5 * (1 + params.development_band))
    assert low == pytest.approx(39.5 * (1 - params.development_band))


def test_m7e1f_projection_opens_room_to_fall_to_ten_percent():
    """R31.B7: when every parental act to the child was projection-type, the floor is −10%
    (`development_band_projection`); half projection opens half the extra room."""
    full, params = _band_after([20, 20, 0], 5.0)
    half, _ = _band_after([20, 10, 0], 5.0)
    assert full == pytest.approx(39.5 * (1 - params.development_band_projection))
    extra = params.development_band_projection - params.development_band
    assert half == pytest.approx(39.5 * (1 - params.development_band - extra / 2))


def test_m7e1f_support_opens_room_to_rise_to_five_percent():
    """R31.B7: when every parental act to the child was supportive, the ceiling is +5% (`development_band_support`)."""
    full, params = _band_after([20, 0, 20], 90.0)
    none_of_them, _ = _band_after([20, 0, 0], 90.0)
    assert full == pytest.approx(39.5 * (1 + params.development_band_support))
    assert none_of_them == pytest.approx(39.5 * (1 + params.development_band))


def test_m7e1f_acts_open_room_but_do_not_push():
    """Adversarial (R31.B7): a child who receives only projection acts, while everyone functions above basic, does not
    fall. The acts set how far the level may move; the development rule alone moves it."""
    level, params = _band_after([20, 20, 0], 60.0)
    assert level == pytest.approx(39.5 * (1 + params.development_band))
    # Everyone functioning at their own basic level: supportive acts alone do not raise the child's level.
    state, params = _developing(10)
    state.life.parental_acts[NADIA] = [20, 0, 20]
    for pid in (RAVI, MARTA):
        state.people[pid].basic_level = 40.0
        history(state, pid, 40.0)
    state.people[NADIA].basic_level = 39.5
    history(state, NADIA, 39.5)
    _childhood(state, params)
    assert state.people[NADIA].basic_level == pytest.approx(39.5)


def test_m7e1f_a_narrowing_band_does_not_move_the_level():
    """Adversarial (learning-qa 2026-10-16): the shares change over childhood, so the band can narrow under a level
    already at its old bound. The level must stay where it is unless the development rule moves it; the band only
    limits a step. Projection's share falls while everyone functions below basic: the floor rises past the level, and
    the level must not be pulled up toward it. The mirror for the ceiling."""
    state, params = _developing(10)
    anchor = 39.5
    for pid in (RAVI, MARTA, NADIA):
        history(state, pid, 5.0)                    # d < 0: the rule pulls down
    state.people[NADIA].basic_level = anchor * (1 - params.development_band_projection)  # at the old floor
    state.life.parental_acts[NADIA] = [40, 20, 0]  # projection's share halved: the floor is now higher
    before = state.people[NADIA].basic_level
    _childhood(state, params)
    assert state.people[NADIA].basic_level == pytest.approx(before)
    state, params = _developing(10)
    for pid in (RAVI, MARTA, NADIA):
        history(state, pid, 90.0)                   # d > 0: the rule pushes up
    state.people[NADIA].basic_level = anchor * (1 + params.development_band_support)  # at the old ceiling
    state.life.parental_acts[NADIA] = [40, 0, 20]  # support's share halved: the ceiling is now lower
    before = state.people[NADIA].basic_level
    _childhood(state, params)
    assert state.people[NADIA].basic_level == pytest.approx(before)


def test_m7e1f_parental_acts_counted_from_the_acts_sent():
    """R31.B7: each week, the acts a living parent sends a child aged 5 to 19 are counted, by projection-type and
    supportive kind, from what the week emitted."""
    from src.bowen.engine.act import SELF_CHANNEL
    from src.bowen.engine.log_records import EmittedRecord
    from src.bowen.engine.sinks import PROJECTION_KINDS

    built, source = parts()
    state, params = built.state, built.params
    ages = {pid: state.people[pid].age_years for pid in state.life.parents}
    emitter = run_weeks(built, source, 52)  # ages advance at the end of week 51, after the week is counted
    expected = {}
    for record in emitter.records:
        if not isinstance(record, EmittedRecord) or record.event.mechanism is not Mechanism.MOVE:
            continue
        event = record.event
        for child, parents in state.life.parents.items():
            if (event.sender in parents and child in event.targets
                    and params.development_start_age_years <= ages[child] < params.development_end_age_years):
                row = expected.setdefault(child, [0, 0, 0])
                row[0] += 1
                row[1] += event.kind in PROJECTION_KINDS
                row[2] += state.kinds.channels.get(event.kind) == SELF_CHANNEL
    assert expected, "no parental act reached a developing child in a year: the check is empty"
    assert state.life.parental_acts == expected


def test_m7e1f_development_stops_at_twenty():
    state, params = _developing(params_age := 21)
    for pid in (RAVI, MARTA, NADIA):
        history(state, pid, 60.0)
    before = state.people[NADIA].basic_level
    _childhood(state, params)
    assert params_age > params.development_end_age_years and state.people[NADIA].basic_level == before


def test_m7e1f_the_estimator_does_not_run_on_a_developing_child(monkeypatch):
    """Review 2026-10-13: read the estimator's calls, not the level, which the band clips anyway."""
    import src.bowen.engine.slow_tick as slow_tick

    built, source = parts()
    read = []
    real = slow_tick.estimate
    monkeypatch.setattr(slow_tick, "estimate", lambda weeks, *a: read.append(weeks) or real(weeks, *a))
    run_weeks(built, source, 52 * 2)
    histories = {id(built.state.estimator_history[p]) for p in (LEO, NADIA, PIA)}
    assert read and not any(id(w) in histories for w in read)


# --- L5: mortality and the disposition at death --------------------------------------------------------------------


def test_m7c1j_hazard_rises_with_age():
    params = parts()[0].params
    us = [(i + 0.5) / 200 for i in range(200)]
    median = sorted(age_at_death(u, 0, params) for u in us)[100]
    assert 75 <= median <= 85
    assert all(age_at_death(u, 70, params) >= 70 for u in us)
    # Remaining years shrink with age: the hazard rises.
    assert sorted(age_at_death(u, 70, params) - 70 for u in us)[100] < sorted(age_at_death(u, 40, params) - 40 for u in us)[100]


def test_m7c1j_age_at_death_drawn_once_at_entry():
    a, _ = parts(seed=4)
    b, _ = parts(seed=4)
    c, _ = parts(seed=5)
    assert a.state.life.death_age == b.state.life.death_age != c.state.life.death_age
    assert HALIM not in a.state.life.death_age  # an external agent retires; it does not die in the run
    assert set(a.state.life.death_age) == {p for p, q in a.state.people.items() if q.role is Role.MEMBER}


def _ready_to_die(seed=3):
    built, source = parts(seed=seed)
    run_weeks(built, source, 30)
    return built


def test_m63b_anxiety_split_by_bond_energy():
    built = _ready_to_die()
    state = built.state
    stock = state.people[ANA].acute_anxiety
    before = {p: q.acute_anxiety for p, q in state.people.items()}
    ties = {t.id.other(ANA): t.bond_energy for t in state.ties_of(ANA)}
    for tie in state.ties_of(ANA):
        tie.distance_bound_anxiety = 0.0
    for tri in state.triangles.values():
        tri.bound_anxiety = 0.0
    state.people[ANA].symptom_bound = 0.0
    die(state, ANA, built.params)
    total = sum(ties.values())
    for other, bond in ties.items():
        assert state.people[other].acute_anxiety - before[other] == pytest.approx(stock * bond / total)
    assert state.people[ANA].acute_anxiety == 0.0 and not state.people[ANA].alive


def test_m63b_no_share_to_an_external_agent():
    """M6.3b's share only: bound stocks are emptied first, because M6.3c returns a symptom's bound stock to its
    partners on interactive ties and discharges a triangle to its living members, an external agent among them (found
    2026-10-14, when a changed run left Ravi's symptom holding anxiety)."""
    built = _ready_to_die()
    state = built.state
    state.people[RAVI].symptom_bound = 0.0
    for tie in state.ties_of(RAVI):
        tie.distance_bound_anxiety = 0.0
    for tri in state.triangles.values():
        tri.bound_anxiety = 0.0
    before = state.people[HALIM].acute_anxiety
    die(state, RAVI, built.params)
    assert state.people[HALIM].acute_anxiety == before


def test_m63b_falls_back_to_the_household():
    built = _ready_to_die()
    state = built.state
    for tie in state.ties_of(P("dario")):
        tie.bond_energy = 0.0
    state.people[P("iris")].household_id = "dario"
    stock = state.people[P("dario")].acute_anxiety
    before = state.people[P("iris")].acute_anxiety
    die(state, P("dario"), built.params)
    assert state.people[P("iris")].acute_anxiety - before == pytest.approx(stock)


def test_m63a_a_death_with_no_living_mourner_queues_no_nodal_event():
    """Found by the part II ensemble (2026-10-13): Ana died after everyone tied to her, and her DEATH event reached no
    one, which the estimator's nodal count raises on. The death is still disposed of and logged."""
    built = _ready_to_die()
    state = built.state
    for tie in state.ties_of(ANA):
        other = tie.id.other(ANA)
        if other != MARTA:
            tie.interactive = False
    state.people[MARTA].household_id = "elsewhere"
    tie = state.tie_between(ANA, MARTA)
    tie.interactive = False
    record = die(state, ANA, built.params)
    assert record.mechanism == "death"
    state.tick += 1
    assert not [e for e in due_events(state) if e.kind == "DEATH"]


def test_m63c_tie_and_triangle_holdings_go_to_the_living():
    built = _ready_to_die()
    state = built.state
    tie = state.tie_between(RAVI, MARTA)
    tie.distance_bound_anxiety = 6.0
    tri = state.triangles[TriangleId.of(RAVI, MARTA, NADIA)]
    tri.bound_anxiety = 4.0
    state.people[RAVI].symptom_bound = 0.0
    record = die(state, RAVI, built.params)
    assert tie.distance_bound_anxiety == 0.0 and tri.bound_anxiety == 0.0 and not tri.active
    moved = dict(record.acute_anxiety)
    assert moved[NADIA] >= 2.0 - 1e-9  # half the triangle's 4
    assert sum(v for p, v in record.acute_anxiety if p == MARTA) >= 6.0 + 2.0 - 1e-9


def test_m63d_investment_redistributed_over_other_ties():
    built = _ready_to_die()
    state = built.state
    for tie in state.ties_of(MARTA):
        tie.investment[MARTA] = 1.0
    total = sum(t.investment[MARTA] for t in state.ties_of(MARTA))
    die(state, RAVI, built.params)
    assert state.tie_between(RAVI, MARTA).investment[MARTA] == 0.0
    assert sum(t.investment.get(MARTA, 0.0) for t in state.ties_of(MARTA)) == pytest.approx(total)


def test_m63e_balances_settle_back_after_a_death():
    built, source = parts()
    state = built.state
    tie = state.tie_between(RAVI, MARTA)
    # Ravi over-functioned for Marta (the balance on his pole); empty at t0, so it is set, not copied from the keys.
    tie.functioning_balance = {DEFAULT_AREA: _pole_of(tie, RAVI) * 1.0}
    state.people[RAVI].pseudo_self, state.people[MARTA].pseudo_self = 4.0, -4.0
    state.people[RAVI].functional_level += 4.0
    state.people[MARTA].functional_level -= 4.0
    die(state, MARTA, built.params)
    total = sum(p.pseudo_self or 0.0 for p in state.people.values())
    # Read on the tie itself (2026-10-14): over a run, Ravi's other exchanges move his pseudo-self as much as this does.
    state.tick += built.params.exchange_window + 1  # no exchange with Marta since she died
    relax_pseudo_self(state, built.params)
    assert state.people[RAVI].pseudo_self < 4.0  # the borrowed self does not survive the lender
    assert state.people[MARTA].pseudo_self > -4.0
    assert sum(p.pseudo_self or 0.0 for p in state.people.values()) == pytest.approx(total)


def test_m63f_sinks_naming_the_dead_return_to_the_family():
    built = _ready_to_die()
    family = built.state.family
    family.sink_allocations.update({Sink.MARITAL_CONFLICT: 5.0, Sink.SPOUSE_DYSFUNCTION: 6.0, Sink.CHILD_PROJECTION: 7.0})
    die(built.state, MARTA, built.params)
    assert family.sink_allocations == {Sink.MARITAL_CONFLICT: 0.0, Sink.SPOUSE_DYSFUNCTION: 0.0, Sink.CHILD_PROJECTION: 7.0}
    assert not family.dissolved
    die(built.state, RAVI, built.params)
    assert family.dissolved


def test_m63g_grief_lasts_one_to_three_years_by_bond_energy():
    built = _ready_to_die()
    state, params = built.state, built.params
    bonds = {t.id.other(MARTA): t.bond_energy for t in state.ties_of(MARTA) if t.interactive}
    die(state, MARTA, params)
    state.tick += 1
    grief = {e.targets[0]: e for e in due_events(state) if e.kind == GRIEF}
    assert set(grief) == {p for p in bonds if state.people[p].role is Role.MEMBER}
    for pid, event in grief.items():
        assert event.mechanism is Mechanism.EXOGENOUS_STRESSOR
        who = "spouse" if TieId.of(pid, MARTA) in {f.marital_tie for f in state.families()} else "relative"
        arrival, _ = sized_intensity(state.kinds, params, GRIEF, who, reach=bonds[pid] / 100, weeks=event.duration)
        assert event.intensity == pytest.approx(params.grief_weight * arrival)  # revision 36: sized from the scale
        assert params.grief_min_weeks <= event.duration <= params.grief_min_weeks + params.grief_extra_weeks
    strongest, weakest = max(grief, key=bonds.get), min(grief, key=bonds.get)
    assert grief[strongest].duration > grief[weakest].duration


def test_m11c37_anxiety_conserved_across_death():
    """M11.C.37 (an M11.4 exception, like C.31): the field's total anxiety is the same just after M6.3b–c."""
    built, source = parts(seed=2)
    run_weeks(built, source, 40)
    state = built.state
    state.tie_between(RAVI, MARTA).distance_bound_anxiety = 3.0
    state.triangles[TriangleId.of(RAVI, MARTA, NADIA)].bound_anxiety = 2.0
    before = total_anxiety(state)
    die(state, RAVI, built.params)
    assert total_anxiety(state) == pytest.approx(before, abs=built.params.invariant_tolerance)


def test_m11c37_mutation_discarding_the_dead_persons_anxiety_turns_it_red():
    """The mutation C.37 names, through the real die(): the dead person's anxiety discarded instead of passed on
    (death_transfer_share 0). Review 2026-10-13: the first version stubbed die() and so tested nothing."""
    built, source = parts(seed=2)
    run_weeks(built, source, 40)
    state = built.state
    before = total_anxiety(state)
    held = state.people[RAVI].acute_anxiety
    die(state, RAVI, dataclasses.replace(built.params, death_transfer_share=0.0))
    assert held > 0 and total_anxiety(state) == pytest.approx(before - held)


def test_m11c37_a_run_raises_when_a_death_loses_anxiety(monkeypatch):
    """The run-time check in yearly(): a disposition that loses anxiety the rule does not discard raises."""
    import src.bowen.engine.life_course as life_course
    from src.bowen.engine.invariants import InvariantViolation

    built, source = parts(seed=2)
    run_weeks(built, source, 40)
    state = built.state
    real = life_course.die

    def leaky(state, pid, params):
        record = real(state, pid, params)
        state.people[MARTA].acute_anxiety -= 1.0
        return record

    monkeypatch.setattr(life_course, "die", leaky)
    state.life.death_age[RAVI] = state.people[RAVI].age_years
    with pytest.raises(InvariantViolation, match="M6.I.6"):
        life_course.yearly(state, built.params)


# --- L6: lost income and finding work --------------------------------------------------------------------------------


def _widowed(seed=3):
    built, source = parts(seed=seed)
    run_weeks(built, source, 30)
    die(built.state, RAVI, built.params)
    return built, source


def test_m63h_income_loss_lasts_until_someone_finds_work():
    built, source = _widowed()
    state = built.state
    income = [s for s in state.stressors if s.last_tick is None]
    assert {s.target for s in income} == {MARTA, LEO, NADIA, PIA}
    built.params = dataclasses.replace(built.params, find_work_rate=0.0)  # nobody finds work: the loss must last
    run_weeks(built, PolicySource(source.script, built.params, source.rules), 52 * 3)
    assert {s.target for s in state.stressors if s.last_tick is None and state.people[s.target].alive} \
        >= {MARTA, NADIA, PIA}


def test_m7c1k_find_work_leaves_twenty_percent_and_sets_earner():
    built, _ = _widowed()
    state, params = built.state, dataclasses.replace(built.params, find_work_rate=1.0)
    state.people[MARTA].usual_excess = 0.0
    state.people[MARTA].earner = False
    records = find_work(state, {MARTA: params.find_work_pressure_scale}, params)
    assert records and state.people[MARTA].earner and not state.people[MARTA].financially_dependent
    income = [s for s in state.stressors if s.last_tick is None]
    assert all(s.intensity == pytest.approx(params.income_loss_intensity * params.income_shortfall_share)
               for s in income)
    assert find_work(state, {MARTA: params.find_work_pressure_scale}, params) == []  # resolved: no second search


def test_m7c1k_find_work_unavailable_without_a_loss():
    built, _ = parts()
    params = dataclasses.replace(built.params, find_work_rate=1.0)
    assert find_work(built.state, {RAVI: 100.0}, params) == []


def test_m7c1k_find_work_follows_pressure_not_a_schedule():
    """Who finds work and when vary by seed; with no pressure no one does (M7.C.1k: not scheduled)."""
    weeks = set()
    for seed in range(6):
        built, source = _widowed(seed)
        found = None
        for _ in range(156):
            emitter = run_weeks(built, source, 1)
            if any(getattr(r, "mechanism", None) == "find_work" for r in emitter.records):
                found = built.state.tick
                break
        weeks.add(found)
    assert len(weeks) > 1
    built, source = _widowed()
    for pid in built.state.people:
        built.state.people[pid].usual_excess = 1e9  # never above their usual level: no pressure
    assert find_work(built.state, {MARTA: 50.0}, dataclasses.replace(built.params, find_work_rate=1.0)) == []


# --- L7: launch, partners, births ------------------------------------------------------------------------------------


def test_m7c1g_launch_moves_household_and_ends_dependence():
    built, _ = parts()
    state, params = built.state, built.params
    state.people[LEO].age_years = params.launch_age_years - 1
    yearly(state, params)
    leo = state.people[LEO]
    assert leo.household_id == "leo" and not leo.financially_dependent and leo.earner
    assert state.tie_between(RAVI, LEO) is not None  # the ties stay


def test_m7c1h_partner_level_within_pairing_tolerance():
    gaps = []
    for seed in range(20):
        built = pair_now(seed)
        gaps.append(built.state.people[P("sara")].basic_level - built.state.people[LEO].basic_level)
    tolerance = built.params.spouse_basic_level_tolerance
    assert all(abs(g) <= tolerance + 1e-9 for g in gaps)
    assert min(gaps) < 0 < max(gaps)  # drawn across the tolerance, not pinned to one side


def test_m7c1h_partner_comes_only_from_settings(paired):
    built, _ = paired
    declared = {P(x) for x in ("sara", "omar", "theo", "sara_parent", "omar_parent", "theo_parent")}
    founders = set(load_family(PHASE_D).people)
    entered = {p for p in built.state.people if p not in founders and "#" not in p.value}
    assert entered <= declared | {P("halim_successor")}
    bad = read_text(PARTNERS).replace("| `sara` | Sara | `leo` |", "| `ravi` | Sara | `leo` |")
    with pytest.raises(ConfigError, match="already declared"):
        build_family(read_text(PHASE_D), load_constants(), partners_text=bad)


def test_m7c1h_new_couple_gets_a_family_with_derived_budget(paired):
    built, _ = paired
    state, params = built.state, built.params
    family = state.other_families["leo_sara"]
    mean = (state.people[LEO].basic_level + state.people[P("sara")].basic_level) / 2
    assert family.undifferentiation_budget == pytest.approx(params.budget_per_undifferentiation * (100 - mean), rel=1e-3)
    assert state.people[LEO].household_id == state.people[P("sara")].household_id == "leo_sara"
    assert state.people[P("sara_parent")].inactive


def _before_first_birth():
    built = pair_now(1)
    state = built.state
    for tie in state.ties_of(LEO):
        tie.investment[LEO] = 1.0
    return state, built.params, sum(t.investment[LEO] for t in state.ties_of(LEO))


def test_m7c1i_birth_ties_redistribute_investment():
    """M7.C.1a: the newborn's tie claims a share of the parent's existing investment, taken from the other ties."""
    state, params, total = _before_first_birth()
    row = state.life.settings.partners[0]
    _birth(state, row, 2, params)  # the second-born, also the last-born (M7.E.1c)
    child = child_id(LEO, P("sara"), 2)
    others = sum(t.investment.get(LEO, 0.0) for t in state.ties_of(LEO) if t.id != TieId.of(LEO, child))
    assert others == pytest.approx(total * (1 - params.new_tie_investment_share))
    assert state.tie_between(LEO, child).investment[LEO] >= params.new_tie_investment_share * total - 1e-9
    assert state.life.parents[child] == (LEO, P("sara")) and state.people[child].age_years == 0


def test_m7e1c_birth_situations_add_focus_not_a_target():
    """A firstborn gets more of a parent's focus than the plain share; nothing names a projection target."""
    state, params, total = _before_first_birth()
    _birth(state, state.life.settings.partners[0], 1, params)
    first = child_id(LEO, P("sara"), 1)
    assert state.tie_between(LEO, first).investment[LEO] >= params.new_tie_investment_share * total * (
        1 + params.birth_focus_gain) - 1e-9
    assert not any("target" in f.name for f in dataclasses.fields(state.life))


def test_m64_entry_and_new_budget_are_logged_sources(paired):
    _, emitter = paired
    assert {"partnering", "life_course"} <= {getattr(r, "mechanism", None) for r in emitter.records}
    built = pair_now(1)
    assert _birth(built.state, built.state.life.settings.partners[0], 1, built.params).mechanism == "birth"


# --- L8: the coach's career --------------------------------------------------------------------------------------


def test_m2a0k_successor_enters_at_retirement_and_learning_is_kept():
    built, source = parts()
    state, params = built.state, built.params
    state.people[HALIM].age_years = params.coach_retirement_age_years - 1
    state.people[RAVI].systems_perspective = 0.4
    records = yearly(state, params)
    successor = P("halim_successor")
    assert any(r.mechanism == "coach_retirement" for r in records)
    assert state.people[HALIM].inactive and not state.tie_between(HALIM, RAVI).interactive
    new = state.people[successor]
    assert new.role is Role.EXTERNAL and new.age_years == params.coach_start_age_years and new.sex.value == "female"
    assert [t.id for t in state.ties_of(successor)] == [TieId.of(RAVI, successor)]
    assert state.people[RAVI].systems_perspective == 0.4
    assert successor in built.activation.active(state.tick, state.people.values())
    assert HALIM not in built.activation.active(state.tick, state.people.values())


# --- L9: the 40-year run -----------------------------------------------------------------------------------------


@pytest.mark.ensemble
def test_m3b1_forty_year_run_completes():
    """One seed of the twelve-person family over 40 years: every invariant holds every week (they raise)."""
    built, source = parts(seed=11)
    emitter = run_weeks(built, source, 52 * 40)
    assert built.state.tick == 52 * 40
    assert {"death", "partnering", "birth"} <= {getattr(r, "mechanism", None) for r in emitter.records}


def test_m16t3_sink_does_not_change_a_life_course_run(tmp_path):
    """M16.T.3's re-run for part II: the twelve-person family with its life course, the slow clock, the coach and the
    belief channel, sink attached and detached, reaches the same state, life course included, through the first
    pairing and birth (seed 1: Leo pairs in year 19)."""
    weeks = 52 * 22
    detached, source = parts(seed=1)
    kept = run_weeks(detached, source, weeks)
    attached, source = parts(seed=1)
    collected, path = CollectingEmitter(), tmp_path / "run.jsonl"
    with JsonlFileSink(path) as sink:
        tee = Tee(collected, sink)
        for _ in range(weeks):
            run_tick(attached.state, source, attached.params, attached.visibility, attached.activation, tee)
    assert {"partnering", "birth"} <= {getattr(r, "mechanism", None) for r in kept.records}
    assert canonical_state(attached.state) == canonical_state(detached.state)
    assert path.read_text(encoding="utf-8").splitlines() == [serialize(r) for r in collected.records]



def test_m16t3_canonical_state_sees_the_life_course():
    """The comparison above would miss a sink that changed the life course if canonical_state left it out: a change to
    state.life alone must change it."""
    built, _ = parts()
    before = canonical_state(built.state)
    built.state.life.pending.append(None)
    assert canonical_state(built.state) != before

# --- review 2026-10-13 ---------------------------------------------------------------------------------------------


def test_m7e1f_an_orphan_reads_no_dead_parent():
    """A dead parent's frozen values are not read: the child's chronic anxiety and level stay where they are."""
    built, _ = parts()
    state, params = built.state, built.params
    for pid in (RAVI, MARTA):
        state.people[pid].alive = False
    history(state, PIA, 39.5, load=10.0)
    before = (state.people[PIA].chronic_anxiety, state.people[PIA].basic_level)
    _childhood(state, params)
    assert (state.people[PIA].chronic_anxiety, state.people[PIA].basic_level) == before


def test_m7c1h_a_partner_row_that_could_never_fire_is_refused():
    bad = read_text(PARTNERS).replace("| `sara` | Sara | `leo` | 28 |", "| `sara` | Sara | `leo` | 9 |")
    with pytest.raises(ConfigError, match="would never happen"):
        build_family(read_text(PHASE_D), load_constants(), partners_text=bad)
    twice = read_text(PARTNERS).replace("| `omar` | Omar | `nadia` |", "| `omar` | Omar | `leo` |")
    with pytest.raises(ConfigError, match="one per member"):
        build_family(read_text(PHASE_D), load_constants(), partners_text=twice)


def test_m7c1a_an_older_sibling_and_the_coachee_redistribute_too():
    state, params, _ = _before_first_birth()
    _birth(state, state.life.settings.partners[0], 1, params)
    first = child_id(LEO, P("sara"), 1)
    tie = state.tie_between(first, LEO)
    for t in state.ties_of(first):
        t.investment[first] = 1.0
    total = sum(t.investment[first] for t in state.ties_of(first))
    _birth(state, state.life.settings.partners[0], 2, params)
    second = child_id(LEO, P("sara"), 2)
    assert state.tie_between(first, second).investment[first] == pytest.approx(params.new_tie_investment_share * total)
    assert tie.investment[first] == pytest.approx(1 - params.new_tie_investment_share)


def test_m2a0k_the_coachees_investment_leaves_the_retired_coach():
    built, _ = parts()
    state, params = built.state, built.params
    for t in state.ties_of(RAVI):
        t.investment[RAVI] = 1.0
    state.people[HALIM].age_years = params.coach_retirement_age_years - 1
    yearly(state, params)
    assert state.tie_between(HALIM, RAVI).investment[RAVI] == 0.0
    assert state.tie_between(RAVI, P("halim_successor")).investment[RAVI] > 0


def test_m7c1_a_missed_life_event_raises():
    built = _ready_to_die()
    state = built.state
    die(state, MARTA, built.params)
    state.tick += 2
    with pytest.raises(RuntimeError, match="missed their week"):
        due_events(state)


def test_m7c1k_finding_work_reaches_a_household_with_no_ties_between_its_members():
    """Found by the part II ensemble (2026-10-13): Leo found work and the event named Nadia, to whom he has no tie."""
    built, source = _widowed()
    state = built.state
    params = dataclasses.replace(built.params, find_work_rate=1.0)
    state.people[LEO].age_years = params.working_age_years
    for pid in (MARTA, NADIA, PIA):
        state.people[pid].usual_excess = 1e9  # only Leo is pressed
    state.people[LEO].usual_excess = 0.0
    assert find_work(state, {LEO: params.find_work_pressure_scale}, params)
    state.tick += 1
    work = [e for e in due_events(state) if e.kind == "FIND_WORK"]
    assert work and LEO in work[0].targets and NADIA in work[0].targets and work[0].sender is None
    visible = built.visibility.resolve(work[0], state.people, state.ties)  # must not raise MissingTie
    reached = {d.recipient for d in visible.deliveries if d.role is DeliveryRole.TARGET}
    assert {LEO, NADIA} <= reached  # and it is delivered, at once, to everyone it names
    assert all(d.latency == 0 for d in visible.deliveries if d.role is DeliveryRole.TARGET)


@pytest.mark.parametrize("name, value", [("development_band", 1.5), ("development_rate", -0.1), ("find_work_rate", 2.0),
                                         ("income_shortfall_share", -0.1), ("death_transfer_share", 1.1),
                                         ("grief_weight", -1.0), ("mortality_b", -0.1), ("find_work_pressure_scale", 0.0),
                                         ("development_start_age_years", 30), ("working_age_years", 40),
                                         ("coach_retirement_age_years", 20)])
def test_m7c1_life_course_constants_out_of_range_are_refused(name, value):
    """Review 2026-10-14 (adjacent finding): revision 28's constants had no range checks."""
    import dataclasses

    from src.bowen.io.load import load_constants
    from src.bowen.scenario.params import engine_params
    with pytest.raises(ValueError):
        dataclasses.replace(engine_params(load_constants()), **{name: value})
