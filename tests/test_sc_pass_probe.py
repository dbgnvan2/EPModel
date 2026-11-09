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
