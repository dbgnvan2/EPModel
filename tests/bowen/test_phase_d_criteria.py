"""The Phase D estimator criteria (Phase D plan step 6; spec revision 17).

Purpose: one gate test per criterion, each running its paired arms through the ensemble runner and requiring PASS
         (``ensemble``-marked, so excluded from the default run), and fast checks that each arm runs and differs only
         in its declared channel.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.17, #M11.C.18, #M11.C.20, #M11.C.33, #M11.2, #M13.4
Tests:   this file
"""

from __future__ import annotations

import os

import pytest

from src.bowen.ensemble.criteria_d import CRITERIA_D, MOVER, c33_inputs, nodal
from src.bowen.ensemble.runner import PASS, run_criterion
from src.bowen.io.load import load_constants, load_event_kinds

C = load_constants()
RULES = {k: C[k] for k in ("ensemble_block", "ensemble_cap", "ensemble_precision", "ensemble_margin",
                           "ensemble_alpha", "fallback_flag_rate", "equivalence_margin")}
WORKERS = os.cpu_count() or 1


def gate(cid: str) -> None:
    verdict = run_criterion(CRITERIA_D[cid], RULES, workers=WORKERS)
    assert verdict.outcome == PASS, f"{cid}: {verdict.outcome} {verdict.readouts}"


@pytest.mark.ensemble
def test_m11c18_estimator_separates_cutoff_from_iposition():
    gate("M11.C.18")


@pytest.mark.ensemble
def test_m11c20_estimator_separates_borrowed_from_basic():
    gate("M11.C.20")


@pytest.mark.ensemble
def test_m11c33_dependence_gate_blocks_basic_level_rise():
    gate("M11.C.33")


@pytest.mark.ensemble
def test_m11c17_coach_helps_and_is_not_required():
    gate("M11.C.17")


def test_m11c18_the_nodal_load_fills_every_window_with_three_kinds():
    """Every 156-week estimator window of the declared load holds the three kinds (estimator_breadth)."""
    weeks, window = CRITERIA_D["M11.C.18"].settings["weeks"], C["estimator_window"] * C["slow_tick_fast_ticks"]
    events = nodal(weeks, (MOVER,))
    assert {e.kind for e in events} <= load_event_kinds().nodal
    for end in range(window - 1, weeks, C["slow_tick_fast_ticks"]):
        kinds = {e.kind for e in events if end - window < e.timestamp <= end}
        assert len(kinds) >= C["estimator_breadth"], end


def test_m11c33_arms_differ_only_at_the_slow_tick():
    """Review 2026-10-11: up to the first slow tick that can raise basic_level (end of week 155), the mover makes
    the same choices in both arms, week by week — the dependence is visible to nothing but the slow tick."""
    from src.bowen.engine.log_records import SelectionRecord
    from src.bowen.ensemble.criteria import scenario

    weeks = C["estimator_window"] * C["slow_tick_fast_ticks"]
    settings = CRITERIA_D["M11.C.33"].settings
    for seed in (0, 3, 5):
        picks = {}
        for arm in ("baseline", "treatment"):
            _, records = scenario("triad", seed, weeks, **c33_inputs(arm, weeks, settings["act_every"]))
            # what the mover could choose, and what it chose, each week
            picks[arm] = [(r.tick, tuple(label for label, _ in r.propensities), r.event_id) for r in records
                          if isinstance(r, SelectionRecord) and r.actor == MOVER]
        assert picks["baseline"] == picks["treatment"], seed
        assert any(e is not None and "I-POSITION" in str(labels) for _, labels, e in picks["baseline"])
