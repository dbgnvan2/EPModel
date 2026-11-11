"""Smoke tests for ``tools/sc_pass_probe.py``, T-206's and T-208's probe (a diagnostic in no record's hash).

Purpose: the probe's ``equal`` condition is `M4.C.1f`'s reduction as the unit tests state it, its ``recorded`` condition
         reads what the criterion's own arm reads, and its summary parses its own rows.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.52, #M4.C.1f
Tests:   this file
"""

import dataclasses
import importlib.util
import json
from pathlib import Path

from tools import sc_pass_probe as probe

_spec = importlib.util.spec_from_file_location(
    "_social_cues_tests", Path(__file__).parent / "bowen" / "test_social_cues.py")
_cues_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cues_tests)
NEUTRAL, ON = _cues_tests.NEUTRAL, _cues_tests.ON


def test_m4c1f_probe_equal_condition_is_the_unit_tests_reduction():
    assert dataclasses.replace(ON, **probe.EQUAL) == NEUTRAL
    assert probe.C52["equal"]["social_cues_enabled"] == 1


def test_m11c52_probe_recorded_condition_reads_the_criterions_arm(tmp_path):
    from src.bowen.ensemble.criteria import SETTINGS
    from src.bowen.ensemble.criteria_sc import arm_c52

    row = probe.c52_seed((3, "recorded"))
    for arm in ("baseline", "treatment"):
        assert row[arm] == arm_c52(arm, 3, SETTINGS["M11.C.52"]).readouts["escalation"]
    path = tmp_path / "rows.jsonl"
    rows = [row, probe.c52_seed((4, "recorded"))]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = probe.summarise(probe._rows([path]))
    assert len(out) == 1 and out[0].startswith("c52 recorded (seeds 3–4, n 2): escalation")
    assert probe.paired(rows, rows).startswith("effect(b) − effect(a) +0.00")


def test_m11c55_probe_calls_supplier_run_as_the_criterion_declares_it():
    """T-206's probe reads `M11.C.55`'s own run through ``criteria_d2.supplier_run``, so a change to that function's
    arguments breaks the probe silently — spec revision 57 added ``lead_weeks`` and left the probe calling it with
    three arguments, which raised `TypeError` on the first seed (learning-qa 2026-11-10). The engine is not run here:
    ``supplier_run`` is replaced by a stub that records what it was called with, so the test checks the call itself.
    Mutation: dropping any argument from the call, or passing the lead where the window goes, fails here."""
    import collections
    import inspect
    from unittest import mock

    from src.bowen.ensemble import criteria_d2
    from src.bowen.ensemble.criteria import SETTINGS

    settings = SETTINGS["M11.C.55"]
    # the relief is read for the real children, so a plain dict would need their ids; 0.0 for every pair ties the
    # highest, which names no one, and the stub's empty records leave every supply count empty
    summary = {"records": [], "relief": collections.defaultdict(float),
               "hits": {"relief": 0, "prior_supply": 0, "birth_order": 0}, "parents_supplied": 0}
    stub = mock.Mock(return_value=summary)
    with mock.patch.object(criteria_d2, "supplier_run", stub):
        row = probe.c55_seed((3, "recorded"))
    bound = inspect.signature(criteria_d2.supplier_run.__wrapped__).bind(*stub.call_args.args,
                                                                        **stub.call_args.kwargs)
    bound.apply_defaults()
    assert bound.arguments == {"seed": 3, "years": settings["years"], "from_year": settings["window_from_year"],
                               "lead_weeks": settings["relief_lead_weeks"]}
    assert row["probe"] == "c55" and row["seed"] == 3
