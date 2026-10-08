"""The ensemble statistic and runner (Phase C step 12, plan D7, D8).

Purpose: test the signed-rank statistic against published values, Holm's correction, and the
         runner's verdicts — direction, margin, adaptive stopping, nulls, assay limits, missing
         moves and the fallback flag — on toy arms whose truth is known.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.4e, #M17.A.1, #M17.A.3, #M17.A.4, #M11.4a, #M11.4d, #M11.1f, #M11.D.18, #M13.4
Tests:   this file
"""

from __future__ import annotations

import pytest

from src.bowen.ensemble.runner import FAIL, PASS, UNDETERMINED, Criterion, Readout, run_criterion
from src.bowen.ensemble.stats import holm, signed_rank
from src.bowen.io.load import load_constants

from ensemble_toys import noisy  # noqa: E402 — tests/bowen is on the path under pytest's rootdir

C = load_constants()
RULES = {k: C[k] for k in ("ensemble_block", "ensemble_cap", "ensemble_precision", "ensemble_margin",
                           "ensemble_alpha", "fallback_flag_rate", "equivalence_margin")}


def crit(readouts=(Readout("x", +1),), requires=(), **settings):
    return Criterion("toy", "check", ("baseline", "treatment"), tuple(readouts), noisy, requires, settings)


def test_m114e_signed_rank_matches_table():
    """Exact one-sided p against published Wilcoxon values."""
    assert signed_rank([1, 2, 3, 4, 5])[2] == pytest.approx(1 / 32)
    # n = 10, W- = 10 (ranks 1..4 negative): P(T <= 10) = 0.0420 in the published table.
    assert signed_rank([-1, -2, -3, -4, 5, 6, 7, 8, 9, 10])[2] == pytest.approx(0.0420, abs=5e-4)
    # n = 8, W- = 5: P(T <= 5) = 0.0391.
    assert signed_rank([-1, -4, 2, 3, 5, 6, 7, 8])[2] == pytest.approx(0.0391, abs=5e-4)


def test_m114e_defined_for_degenerate_arms():
    w, n, p = signed_rank([0.0, 0.0, 0.0])
    assert n == 0 and p == 1.0
    assert signed_rank([2.0] * 30)[2] < 1e-6  # a constant shift, zero variance, by normal approximation


def test_m17a4_differences_below_the_margin_do_not_count():
    assert signed_rank([0.01] * 20, margin=0.05)[2] == 1.0


def test_m114e_holm_steps_down():
    assert holm([0.01, 0.04, 0.03], 0.05) == [True, False, False]
    assert holm([0.01, 0.02, 0.03], 0.05) == [True, True, True]


def test_m17a1_a_real_effect_passes_and_records_the_seeds():
    verdict = run_criterion(crit(effect=1.0), RULES)
    assert verdict.outcome == PASS and verdict.seeds >= RULES["ensemble_block"]


def test_m17a1_undetermined_at_cap_does_not_pass():
    rules = dict(RULES, ensemble_cap=RULES["ensemble_block"])
    verdict = run_criterion(crit(effect=0.3, extra_noise=20.0), rules)
    assert verdict.outcome == UNDETERMINED


def test_m04_a_reversed_effect_fails():
    assert run_criterion(crit(effect=-1.0), RULES).outcome == FAIL


def test_m114a_a_null_passes_only_inside_its_equivalence_bound():
    assert run_criterion(crit((Readout("x", 0),), effect=0.0), RULES).outcome == PASS
    assert run_criterion(crit((Readout("x", 0),), effect=1.0), RULES).outcome == FAIL


def test_m114d_a_null_at_a_bound_is_flagged_as_an_assay_limit():
    verdict = run_criterion(crit((Readout("bounded", 0, (0.0, 1.0)),), bounded=0.0), RULES)
    assert any("assay limit" in f for f in verdict.flagged)


def test_m111f_a_criterion_fails_when_its_move_never_occurs():
    verdict = run_criterion(crit(requires=("CUTOFF",), effect=1.0, cutoffs=0), RULES)
    assert verdict.outcome == FAIL and verdict.missing_moves == ["CUTOFF"]


def test_m11d18_fallback_rate_is_reported_and_flagged():
    verdict = run_criterion(crit(effect=1.0, fallbacks=5), RULES)
    assert verdict.fallback_rate == pytest.approx(0.5) and verdict.fallback_by_person == {"ravi": 0.5}
    assert verdict.outcome == PASS and any("fallback" in f for f in verdict.flagged)


def test_m17a1_precision_is_measured_against_both_arms_spread():
    """A treatment arm much noisier than the baseline converges (decided 2026-10-08, report §10).

    Against the baseline arm's sd alone (0.2 here) the half-width would need ~6,000 seeds; against the pooled sd
    it converges well inside the cap. This is M11.C.16's shape: lower level makes the treatment arm more variable.
    """
    verdict = run_criterion(crit(effect=1.0, noise=0.2, extra_noise=2.0), RULES)
    assert verdict.outcome == PASS and verdict.seeds < RULES["ensemble_cap"]


def test_m17a1_stopping_does_not_depend_on_which_arm_is_called_baseline():
    noisy_treatment = run_criterion(crit(effect=1.0, noise=0.2, extra_noise=2.0), RULES)
    noisy_baseline = run_criterion(crit(effect=1.0, noise=0.2, baseline_extra_noise=2.0), RULES)
    assert noisy_treatment.seeds == noisy_baseline.seeds
