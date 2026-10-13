"""The Phase D part II criteria: the life course (plan §4.4; spec revision 28).

Purpose: one gate test per criterion, running its arms through the ensemble runner and requiring PASS
         (``ensemble``-marked, so excluded from the default run), and fast checks that each arm reads what it says.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.15, #M11.C.6, #M11.C.10, #M11.C.2, #M10.B.5
Tests:   this file
"""

from __future__ import annotations

import os

import pytest

from src.bowen.ensemble.criteria_d2 import CRITERIA_D2, RELEASE_ONLY, arm_c15, life_run
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
def test_m11c2_symptom_concentrates_on_projection_target_in_the_twelve_person_family():
    gate("M11.C.2[twelve-person family]")


def test_m11c15_only_the_release_acts_in_every_arm():
    """M10.B.5: grief, lost income and M6.3b's transfer are off in every arm, and the control reads the same in both
    readouts."""
    assert RELEASE_ONLY == {"grief_scale": 0.0, "income_loss_intensity": 0.0, "death_transfer_share": 0.0}
    settings = CRITERIA_D2["M11.C.15"].settings
    control = arm_c15("baseline", 0, settings).readouts
    assert control["death_rise"] == control["recovery_rise"]
    treated = arm_c15("treatment", 0, settings).readouts
    assert treated["death_rise"] != control["death_rise"] and treated["recovery_rise"] != control["recovery_rise"]


@pytest.mark.ensemble
def test_m11c6_the_children_are_read_at_twenty():
    """One 40-year run: each third-generation child is read when they turn 20, within the 2% band of their anchor."""
    summary = life_run(0, CRITERIA_D2["M11.C.6"].settings["years"], False)
    band = 0.02 * 40 + 1e-9
    assert set(summary["development"]) == set(summary["children"])
    assert all(abs(v) <= band for v in summary["development"].values())
