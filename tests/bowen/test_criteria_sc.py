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

from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.log_records import EmittedRecord
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


def _supply(sender, target, week, index=0) -> EmittedRecord:
    """One supplying move (`M11.G.5`'s first kind) across a run's records."""
    from src.bowen.readouts.roles import load_supply_kinds

    return EmittedRecord(Event(
        id=EventId(week, sender.value, index), kind=sorted(load_supply_kinds())[0], mechanism=Mechanism.MOVE,
        sender=sender, targets=(target,), intensity=1.0, timestamp=week, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.AUTOMATIC))


def gate(criterion) -> None:
    verdict = run_criterion(criterion, RULES, workers=WORKERS)
    assert verdict.outcome == PASS, f"{criterion.id}: {verdict.outcome} {verdict.readouts}"


def _c55_people() -> tuple:
    """`M11.C.55`'s two parents and their children, as ``supplier_run`` reads them from the family."""
    from src.bowen.ensemble.criteria import FAMILIES, ROLES
    from src.bowen.io.load import load_family

    parents = (ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"])
    family = load_family(FAMILIES["phase_d"])
    return parents, tuple(sorted(p for p in family.people if set(family.parents_of(p)) == set(parents)))


def _c55_run(*, relief_at, records=(), seed=0, years=2, from_year=1, lead_weeks=4):
    """``supplier_run`` over a driven watch instead of a run: ``relief_at(tick, child, parent)`` is the learned value
    the child holds on the supplying kinds toward the parent after week ``tick − 1``, and ``records`` are the run's.
    The watch is called for every week of the run, as ``criteria.scenario`` calls it, so which week it keeps is the
    code's choice, not the fixture's."""
    from types import SimpleNamespace

    from src.bowen.engine.value_keys import learned_value_key
    from src.bowen.readouts.roles import load_supply_kinds

    parents, children = _c55_people()
    kind = sorted(load_supply_kinds())[0]

    def people(tick):
        return {c: SimpleNamespace(
            learned_values={learned_value_key("low", "inside", kind, parent.value): relief_at(tick, c, parent)
                            for parent in parents},
            sibling_position=SimpleNamespace(rank=1 + children.index(c))) for c in children}

    def fake_scenario(_family, _seed, weeks, **kwargs):
        for tick in range(1, weeks + 1):
            kwargs["watch"](SimpleNamespace(tick=tick, people=people(tick)))
        return SimpleNamespace(tick=weeks, people=people(weeks)), list(records)

    supplier_run.cache_clear()
    try:
        with mock.patch.object(criteria_d2, "scenario", fake_scenario):
            return supplier_run(seed, years, from_year, lead_weeks)
    finally:
        supplier_run.cache_clear()


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
    """Every predictor is read from the same run, before any move the window counts; each hits at most once per
    parent; the run has the cues on."""
    seen = []
    real = criteria_d2.scenario

    def spy(*args, **kwargs):
        seen.append(kwargs.get("overrides"))
        return real(*args, **kwargs)

    supplier_run.cache_clear()
    try:
        with mock.patch.object(criteria_d2, "scenario", spy):
            summary = supplier_run(0, 2, 1, 4)
    finally:
        supplier_run.cache_clear()
    assert seen == [{"social_cues_enabled": 1}]
    assert set(summary["hits"]) == {"relief", "prior_supply", "birth_order"}
    assert all(0 <= v <= summary["parents_supplied"] <= 2 for v in summary["hits"].values())
    assert len(summary["relief"]) == 3 * 2  # every (child, parent) pair read at the lead's week


def test_m11c55_relief_not_read_is_refused():
    """A run too short to reach the week the relief is read at leaves the predictor unread, and the run is refused
    rather than read as a miss in both arms. So is a lead longer than the window's start (read week 0, before the
    first tick)."""
    supplier_run.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="not read"):
            supplier_run(0, 1, 2, 4)
        with pytest.raises(RuntimeError, match="not read"):
            supplier_run(0, 2, 1, 52)
    finally:
        supplier_run.cache_clear()


def test_m11c55_t206_the_relief_is_read_a_lead_before_the_window():
    """T-206 (the owner's decision of 2026-11-09): the relief is read ``relief_lead_weeks`` before the window's first
    week, not at the window's own start, so the value kept is not the one the policy selects by over the window. The
    value here is the week it was read in, so the summary names the week: the lead's, 48, not the window's start, 52.
    Mutation: reading at the window's start (``state.tick == start``) reads 52 here."""
    start = 1 * criteria_d2.YEAR
    summary = _c55_run(relief_at=lambda tick, child, parent: float(tick), from_year=1, lead_weeks=4)
    assert summary["read_week"] == start - 4 < start
    assert set(summary["relief"].values()) == {float(start - 4)}
    assert len(summary["relief"]) == 3 * 2  # every (child, parent) pair read at that week


def test_m11c55_t206_the_relief_is_compared_with_prior_supply_and_with_chance():
    """T-206: the relief predictor's hits are counted against the supply before the window and against chance, not
    against birth order (which is reported only). Here the second child holds the highest relief toward both parents
    and supplies both of them most over the window, so the relief names both parents' main supplier; the first child
    supplied both most before the window, so prior supply names neither; and the first-born (the first child) names
    neither either. Chance is the parents supplied over the three children."""
    parents, children = _c55_people()
    window = 1 * criteria_d2.YEAR
    records = ([_supply(children[0], parent, week) for parent in parents for week in (2, 3)] +
               [_supply(children[1], parent, window + week) for parent in parents for week in (1, 2)] +
               [_supply(children[0], parent, window + 3) for parent in parents])
    summary = _c55_run(relief_at=lambda tick, child, parent: 1.0 if child == children[1] else 0.0,
                       records=records, from_year=1, lead_weeks=4)
    assert summary["hits"] == {"relief": 2, "prior_supply": 0, "birth_order": 0}
    assert summary["parents_supplied"] == 2 and summary["chance"] == 2 / 3


def test_m11c55_t206_the_arms_read_the_relief_against_prior_supply_and_chance():
    """T-206: both compared readouts read the relief predictor in the treatment; the baseline reads the supply before
    the window in one and chance in the other, so a pass needs the relief above both. Birth order, the baseline until
    T-206, is reported beside them and gates nothing."""
    summary = {"hits": {"relief": 2, "prior_supply": 1, "birth_order": 0}, "parents_supplied": 2, "chance": 2 / 3,
               "read_week": 988, "relief": {}, "records": []}
    with mock.patch.object(criteria_d2, "supplier_run", lambda *args: summary):
        base = criteria_d2.arm_c55("baseline", 0, CRITERIA_D2["M11.C.55"].settings)
        treatment = criteria_d2.arm_c55("treatment", 0, CRITERIA_D2["M11.C.55"].settings)
    assert base.readouts == {"predicted_main_supplier": 1.0, "predicted_over_chance": 2 / 3,
                             "parents_supplied": 2.0, "birth_order_hits": 0.0}
    assert treatment.readouts == {"predicted_main_supplier": 2.0, "predicted_over_chance": 2.0,
                                  "parents_supplied": 2.0, "birth_order_hits": 0.0}
    readouts = CRITERIA_D2["M11.C.55"].readouts
    assert [(r.name, r.direction) for r in readouts if not r.report_only] == [
        ("predicted_main_supplier", +1), ("predicted_over_chance", +1)]
    assert [r.name for r in readouts if r.report_only] == ["parents_supplied", "birth_order_hits"]


def test_m11c55_t206_prior_supply_is_counted_before_the_window_only():
    """T-206: the prior-supply predictor is the main supplier over ``[0, window start)``. The window's own supply must
    not be counted into it, or the predictor is reading the thing it predicts and the baseline arm rises to the
    treatment's. Here the second child supplies both parents five times inside the window and the first child twice
    before it: prior supply names the first child, so it names neither parent's main supplier and reads 0 hits.
    Mutation: counting the supply over the whole run (dropping ``until=start``) makes prior supply name the second
    child, the window's own main supplier, and reads 2 hits (learning-qa and the correctness pass, 2026-11-10)."""
    parents, children = _c55_people()
    window = 1 * criteria_d2.YEAR
    records = ([_supply(children[0], parent, week) for parent in parents for week in (2, 3)] +
               [_supply(children[1], parent, window + week) for parent in parents for week in (1, 2, 3, 4, 5)])
    summary = _c55_run(relief_at=lambda tick, child, parent: 0.0, records=records, from_year=1, lead_weeks=4)
    assert summary["hits"]["prior_supply"] == 0
    assert summary["parents_supplied"] == 2


def test_m11c55_t206_a_parent_no_child_supplied_before_the_window_is_no_prior_hit():
    """T-206: with no supply at all before the window the prior-supply predictor names no one, and
    ``main_supplier`` returning None must not count as naming the main supplier. The second child supplies both
    parents inside the window and no one supplies before it, so prior supply reads 0 hits while both parents are
    supplied. Mutation: treating a predictor that names no one as a hit reads 2 here."""
    parents, children = _c55_people()
    window = 1 * criteria_d2.YEAR
    records = [_supply(children[1], parent, window + week) for parent in parents for week in (1, 2)]
    summary = _c55_run(relief_at=lambda tick, child, parent: 0.0, records=records, from_year=1, lead_weeks=4)
    assert summary["hits"]["prior_supply"] == 0 and summary["parents_supplied"] == 2


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
