"""The Phase C acceptance gate: each criterion over an adaptive ensemble (plan §3, step 14).

Purpose: one test per Phase C `M11.C` criterion, each running its paired arms through the
         ensemble runner and requiring PASS. UNDETERMINED at the cap does not pass (`M13.4`).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.2, #M13.4, #M11.5
Tests:   this file

Marked ``ensemble``: excluded from the default run (pytest.ini) and run with
``python3 -m pytest -m ensemble``. ``tools/ensemble_record.py`` runs the same criteria and writes
the committed record the default suite checks. A criterion not built in Phase C fails here with
its reason; it is not skipped.
"""

from __future__ import annotations

import os

import pytest

from src.bowen.ensemble.criteria import CRITERIA, NOT_BUILT
from src.bowen.ensemble.runner import PASS, run_criterion
from src.bowen.io.load import load_constants

pytestmark = pytest.mark.ensemble
C = load_constants()
RULES = {k: C[k] for k in ("ensemble_block", "ensemble_cap", "ensemble_precision", "ensemble_margin",
                           "ensemble_alpha", "fallback_flag_rate", "equivalence_margin")}
WORKERS = os.cpu_count() or 1


def gate(prefix: str) -> None:
    if prefix in NOT_BUILT:
        pytest.fail(f"{prefix} is not built in Phase C: {NOT_BUILT[prefix]}")
    ids = [cid for cid in CRITERIA if cid == prefix or cid.startswith(prefix + "[")]
    assert ids, f"no criterion defined for {prefix}"
    failed = []
    for cid in ids:
        verdict = run_criterion(CRITERIA[cid], RULES, workers=WORKERS)
        if verdict.outcome != PASS:
            failed.append(f"{cid}: {verdict.outcome}")
    assert not failed, "; ".join(failed)


def test_m11c1_lower_c_reaches_threshold_sooner():
    gate("M11.C.1")

def test_m11c3_triangle_relieves_seeker_costs_third():
    gate("M11.C.3")

def test_m11c4_cutoff_trades_now_against_later():
    gate("M11.C.4")

def test_m11c5_change_back_reaction_shape():
    gate("M11.C.5")

def test_m11c16_repertoire_concentration_depends_on_level():
    gate("M11.C.16")

def test_m11c19_counterfeit_axis_is_identified():
    gate("M11.C.19")

def test_m11c25_dominant_pole_independent_of_sex():
    gate("M11.C.25")

def test_m11c27_twosome_two_by_two_all_cells():
    gate("M11.C.27")

def test_m11c29_relief_and_differentiation_differ_in_time_course():
    gate("M11.C.29")

def test_m11c32_mover_anger_stalls_and_degrades():
    gate("M11.C.32")

def test_m11c35_witness_appraisal_depends_on_both_ties():
    gate("M11.C.35")

def test_m11c38_graded_parameter_orders_primary_readout():
    gate("M11.C.38")

def test_m11c41_level_and_stress_each_exacerbate_the_pattern():
    gate("M11.C.41")

def test_m11c42_relieving_triangle_is_reused():
    gate("M11.C.42")

def test_m11c44_position_value_inverts_with_load():
    gate("M11.C.44")

def test_m11c45_triangles_quiet_when_calm():
    gate("M11.C.45")

def test_m11c7_topology_not_coach_skill():
    gate("M11.C.7")

def test_m11c13_help_relocates_not_reduces():
    gate("M11.C.13")

def test_m11c14_technique_null_under_marital_distance():
    gate("M11.C.14")
