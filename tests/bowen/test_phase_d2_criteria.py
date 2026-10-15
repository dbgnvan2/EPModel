"""The Phase D part II criteria: the life course (plan §4.4; spec revisions 28 and 29, plan §4.8).

Purpose: one gate test per criterion, running its arms through the ensemble runner and requiring PASS
         (``ensemble``-marked, so excluded from the default run), and fast checks that each arm reads what it says.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.15, #M11.C.6, #M11.C.10, #M11.C.2, #M10.B.5
Tests:   this file
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from src.bowen.ensemble.criteria import ROLES, scenario
from src.bowen.ensemble.criteria_d import nodal
from src.bowen.ensemble.criteria_d2 import (
    CRITERIA_D2, NO_GRIEF_OR_INCOME, _declared_pair, _origin_pairs, _renewals, _uncut, arm_c10, arm_c10_origin,
    arm_c15, cut_off_tie_weeks, life_run, partner_of,
)
from src.bowen.ensemble.runner import PASS, run_criterion
from src.bowen.io.load import load_constants

C = load_constants()
RULES = {k: C[k] for k in ("ensemble_block", "ensemble_cap", "ensemble_precision", "ensemble_margin",
                           "ensemble_alpha", "fallback_flag_rate", "equivalence_margin")}
WORKERS = os.cpu_count() or 1


def gate(cid: str) -> None:
    verdict = run_criterion(CRITERIA_D2[cid], RULES, workers=WORKERS)
    assert verdict.outcome == PASS, f"{cid}: {verdict.outcome} {verdict.readouts}"


@pytest.mark.ensemble
def test_m11c15_death_destabilises_like_recovery():
    gate("M11.C.15")


@pytest.mark.ensemble
def test_m11c6_multigenerational_decline_in_projection_line():
    gate("M11.C.6")


@pytest.mark.ensemble
def test_m11c10_cutoff_begets_cutoff():
    gate("M11.C.10")


@pytest.mark.ensemble
def test_m11c10_cutoff_begets_cutoff_from_the_family_of_origin():
    gate("M11.C.10[family of origin]")


@pytest.mark.ensemble
def test_m11c2_symptom_concentrates_on_projection_target_in_the_twelve_person_family():
    gate("M11.C.2[twelve-person family]")


def test_m11c15_grief_and_income_off_and_the_transfer_kept_in_every_arm():
    """M10.B.5: grief and lost income are off in every arm; M6.3b's transfer stays at 1 (owner decision 2026-10-13).
    The control reads the same in both readouts."""
    assert NO_GRIEF_OR_INCOME == {"grief_scale": 0.0, "income_loss_intensity": 0.0}
    settings = CRITERIA_D2["M11.C.15"].settings
    control = arm_c15("baseline", 0, settings).readouts
    assert control["death_rise"] == control["recovery_rise"]
    treated = arm_c15("treatment", 0, settings).readouts
    assert treated["death_rise"] != control["death_rise"] and treated["recovery_rise"] != control["recovery_rise"]


@pytest.mark.ensemble
def test_m11c6_the_children_are_read_at_twenty():
    """One 40-year run: each third-generation child is read when they turn 20, within the 2% band of their anchor."""
    summary = life_run(0, CRITERIA_D2["M11.C.6"].settings["years"])
    band = 0.02 * 40 + 1e-9
    assert set(summary["development"]) == set(summary["children"])
    assert all(abs(v) <= band for v in summary["development"].values())


# --- spec revision 29 (plan §4.8, step R6) -------------------------------------------------------------------------


def test_m11c10_the_treatment_holds_the_cutoff_and_the_baseline_holds_it_open():
    """Two years of each arm: in the treatment the cutter holds rejection every week, renewed on schedule; in the
    baseline the tie is never cut."""
    weeks, every = 104, CRITERIA_D2["M11.C.10"].settings["renew_every"]
    a, b = ROLES["phase_d", "cut_off_a"], ROLES["phase_d", "cut_off_b"]
    parents = (ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"])
    held, opened = [], []
    scenario("phase_d", 0, weeks, spells=nodal(weeks, parents), forced=_renewals(weeks, every),
             watch=lambda st: held.append(st.tie_between(a, b).rejection.get(a, 0.0)))
    renewals = scenario.last_source
    scenario("phase_d", 0, weeks, spells=nodal(weeks, parents), setup=_uncut(_declared_pair()), held_open=(((a, b),), range(weeks)),
             watch=lambda st: opened.append(max(st.tie_between(a, b).rejection.values(), default=0.0)))
    assert all(r > 0 for r in held)
    assert renewals.made == len(range(0, weeks, every)) and renewals.skipped == 0
    assert all(r == 0 for r in opened)


HELD_WEEKS = "(cutter-weeks holding rejection on the cutoffs the arm names)"


def test_m11c10_each_arm_runs_what_it_names():
    """Through arm_c10 and life_run, two years each: the treatment's cutter holds rejection every week and every renewal
    is made and counted; the baseline's never does, and nothing is scripted."""
    settings = {"years": 2, "renew_every": CRITERIA_D2["M11.C.10"].settings["renew_every"]}
    treated, baseline = arm_c10("treatment", 0, settings), arm_c10("baseline", 0, settings)
    renewals = len(range(0, 2 * 52, settings["renew_every"]))
    assert treated.moves[HELD_WEEKS] == 2 * 52 and baseline.moves[HELD_WEEKS] == 0
    assert treated.moves["(scripted acts)"] == renewals == treated.moves["(scripted act: made)"]
    assert baseline.moves["(scripted acts)"] == 0


def test_m11c10_family_of_origin_arm_holds_both_parents_cutoffs():
    """The second variant (the owner: "b as second variant"): each parent cuts off each of their own parents, renewed
    in successive weeks, and holds all four; its baseline holds those ties open."""
    pairs = _origin_pairs()
    parents = {ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"]}
    assert len(pairs) == 4 and {a for a, _ in pairs} == parents
    settings = {"years": 2, "renew_every": CRITERIA_D2["M11.C.10"].settings["renew_every"]}
    treated, baseline = arm_c10_origin("treatment", 0, settings), arm_c10_origin("baseline", 0, settings)
    assert treated.moves[HELD_WEEKS] == 4 * 2 * 52 and baseline.moves[HELD_WEEKS] == 0
    assert treated.moves["(scripted act: made)"] == treated.moves["(scripted acts)"] > 0
    forced = _renewals(52, 13, pairs)
    assert len({w for w, a in forced if a == ROLES["phase_d", "parent_1"]}) == 8  # two ties, alternate weeks


def test_m11c10_reads_rejection_toward_partners_and_parents():
    """Adversarial (the owner: "partner and own parents", "less to one's children"): rejection held by anyone else
    than generation 3, the declared pair and a parent rejecting the child among them, is not counted; toward a parent or partner it is; toward their
    own child it is reported only."""
    state, _ = scenario("phase_d", 0, 1)
    parents = (ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"])
    children = tuple(sorted(p for p in state.people if set(state.life.parents.get(p, ())) == set(parents)))
    for tie in state.ties.values():
        tie.rejection = {}
    a, b = ROLES["phase_d", "cut_off_a"], ROLES["phase_d", "cut_off_b"]
    state.tie_between(a, b).rejection = {a: 1.0, b: 1.0}
    state.tie_between(*parents).rejection = {parents[0]: 1.0}
    state.tie_between(parents[0], children[0]).rejection = {parents[0]: 1.0}  # the parent rejects, not the child
    assert cut_off_tie_weeks(state, children) == (0, 0)
    state.tie_between(parents[0], children[0]).rejection = {children[0]: 0.5}
    assert cut_off_tie_weeks(state, children) == (1, 0)
    # Generation 3 have no partner or child at t0, so one is given to the first child: a partner (the second child,
    # by stand-in) and an own child (the third), each on a new tie the first child rejects on.
    from unittest import mock

    from src.bowen.ensemble import criteria_d2
    from src.bowen.engine.identifiers import TieId
    template = state.tie_between(parents[0], children[0])
    for other in children[1:]:
        state.add_tie(dataclasses.replace(template, id=TieId.of(children[0], other), rejection={children[0]: 1.0}))
    state.life.parents[children[2]] = (children[0],)
    with mock.patch.object(criteria_d2, "partner_of", lambda s, pid: children[1] if pid == children[0] else None):
        assert cut_off_tie_weeks(state, children[:1]) == (2, 1)  # the parent and the partner; the child reported


def test_m11c10_partner_of_reads_the_marital_tie():
    state, _ = scenario("phase_d", 0, 1)
    p1, p2 = ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"]
    assert partner_of(state, p1) == p2 and partner_of(state, p2) == p1


def test_m11c10_runs_200_seeds():
    """The owner: "seeds are expensive - 200 instead": a fixed count, not the adaptive stop."""
    for cid in ("M11.C.10", "M11.C.10[family of origin]"):
        assert CRITERIA_D2[cid].seeds == 200


def test_m11c10_reads_cut_off_tie_weeks():
    """The readout is tie-weeks of rejection, not cut-off acts; rejection of one's own children is reported only."""
    summary = life_run(0, 2, "held", CRITERIA_D2["M11.C.10"].settings["renew_every"])
    assert summary["cut_weeks"] > 0 and summary["child_cut_weeks"] >= 0  # no gen-3 child has a child within 2 years
    assert [(r.name, r.report_only) for r in CRITERIA_D2["M11.C.10"].readouts] == [
        ("cut_off_tie_weeks", False), ("child_cut_weeks", True)]


def test_m11c2_twelve_reads_load_by_age():
    """Read by age: over three years, ages 9 to 11 are seen only for the child who was 9 at t0, and acts are a rate per
    year under the age given."""
    summary = life_run(0, 3, "declared", 0, (10, 9, 11))
    assert len(summary["aged_load"]) == 1  # only one child is 9 to 11 within the first three years
    assert set(summary["acts_per_year"]) == set(summary["children"])


def test_m11c10_a_renewal_toward_a_dead_target_is_counted_as_target_dead():
    """Review 2026-10-14: a scripted act whose target has died was counted "not legal"; it is "target dead" now."""
    from types import SimpleNamespace

    from src.bowen.engine.act import Selection
    from src.bowen.ensemble import criteria

    from src.bowen.engine.identifiers import PersonId

    state, _ = scenario("triad", 0, 1)
    f, m = PersonId("f"), PersonId("m")
    state.people[m].alive = False
    policy = SimpleNamespace(selections=lambda tick, active, st: (), scheduled=lambda tick: (), params=None)
    source = criteria.Forced(policy, {(1, f): Selection(actor=f, kind="CUTOFF", targets=(m,), intensity=100.0)})
    source.selections(1, (f,), state)
    assert source.status == {(1, f): "target dead"} and source.made == 0
