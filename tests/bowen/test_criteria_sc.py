"""The social cues' acceptance criteria, `M11.C.50`–`M11.C.55` (spec revision 55, T-202).

Purpose: one gate test per criterion through the ensemble runner (``ensemble``-marked, so excluded from the default
         run), and fast checks that each arm runs with the cues on and reads what it says.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.50 to #M11.C.55, #M4.C.1f
Tests:   this file
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from src.bowen.ensemble import criteria_d2, criteria_sc
from src.bowen.ensemble.criteria import CRITERIA
from src.bowen.ensemble.criteria_d2 import CRITERIA_D2, supplier_run
from src.bowen.ensemble.runner import PASS, run_criterion
from src.bowen.io.load import load_constants

C = load_constants()
RULES = {k: C[k] for k in ("ensemble_block", "ensemble_cap", "ensemble_precision", "ensemble_margin",
                           "ensemble_alpha", "fallback_flag_rate", "equivalence_margin")}
WORKERS = os.cpu_count() or 1
SC = ("M11.C.50", "M11.C.51", "M11.C.52", "M11.C.53", "M11.C.54")


def gate(criterion) -> None:
    verdict = run_criterion(criterion, RULES, workers=WORKERS)
    assert verdict.outcome == PASS, f"{criterion.id}: {verdict.outcome} {verdict.readouts}"


@pytest.mark.ensemble
def test_m11c50_calmer_delivery_met_with_more_contact():
    gate(CRITERIA["M11.C.50"])


@pytest.mark.ensemble
def test_m11c51_believed_disapproval_read_into_neutral_moves():
    gate(CRITERIA["M11.C.51"])


@pytest.mark.ensemble
def test_m11c52_self_attribution_damps_escalation():
    gate(CRITERIA["M11.C.52"])


@pytest.mark.ensemble
def test_m11c53_preoccupied_spouse_felt_as_less_attentive():
    gate(CRITERIA["M11.C.53"])


@pytest.mark.ensemble
def test_m11c54_false_belief_lowers_the_childs_functioning():
    gate(CRITERIA["M11.C.54"])


@pytest.mark.ensemble
def test_m11c55_main_supplier_learned_not_by_birth_order():
    gate(CRITERIA_D2["M11.C.55"])


@pytest.mark.parametrize("cid", SC)
def test_m4c1f_both_arms_of_every_cue_criterion_run_with_the_cues_on(cid):
    """Each arm passes ``social_cues_enabled`` = 1 to its run, so the criteria test the cues while the switch's
    default stays off (`M4.C.1f`)."""
    seen = []
    real = criteria_sc.scenario

    def spy(*args, **kwargs):
        seen.append(kwargs.get("overrides"))
        family, seed, weeks = args[:3]
        return real(family, seed, min(weeks, 6), *args[3:], **kwargs)

    criterion = CRITERIA[cid]
    with mock.patch.object(criteria_sc, "scenario", spy):
        for arm in criterion.arms:
            criterion.arm(arm, 0, criterion.settings)
    assert len(seen) == 2 and all(o == {"social_cues_enabled": 1} for o in seen)


def test_m11c55_predictors_read_before_the_window():
    """The relief predictor is read at the window's first week, from the same run, before any move the window counts;
    each predictor hits at most once per parent; the run has the cues on."""
    seen = []
    real = criteria_d2.scenario

    def spy(*args, **kwargs):
        seen.append(kwargs.get("overrides"))
        return real(*args, **kwargs)

    supplier_run.cache_clear()
    try:
        with mock.patch.object(criteria_d2, "scenario", spy):
            summary = supplier_run(0, 2, 1)
    finally:
        supplier_run.cache_clear()
    assert seen == [{"social_cues_enabled": 1}]
    assert set(summary["hits"]) == {"relief", "birth_order"}
    assert all(0 <= v <= summary["parents_supplied"] <= 2 for v in summary["hits"].values())
    assert len(summary["relief"]) == 3 * 2  # every (child, parent) pair read at the window's start


def test_m11c55_relief_not_read_is_refused():
    """A run too short to reach the window's start leaves the predictor unread, and the run is refused rather than
    read as a miss in both arms."""
    supplier_run.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="not read"):
            supplier_run(0, 1, 2)
    finally:
        supplier_run.cache_clear()


@pytest.mark.parametrize("first", ["src.bowen.ensemble.criteria_sc", "src.bowen.ensemble.criteria"])
def test_m11c50_criteria_table_is_whole_whichever_module_is_imported_first(first):
    """A pool worker that unpickles one of ``criteria_sc``'s arms imports ``criteria_sc`` before ``criteria``; the
    first regeneration of revision 55 failed there on a circular import (2026-11-08). In a fresh interpreter, either
    order gives the whole table, once each, in the same order."""
    import subprocess
    import sys
    from pathlib import Path

    code = (f"import {first}\nfrom src.bowen.ensemble.criteria import CRITERIA\n"
            "print(','.join(k for k in CRITERIA))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         cwd=Path(__file__).resolve().parents[2], check=True).stdout.strip()
    assert out == ",".join(CRITERIA)
    assert all(cid in out.split(",") for cid in SC)
