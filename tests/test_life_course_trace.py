"""Smoke test for ``tools/life_course_trace.py`` (T-238), on a four-week run.

Purpose: the tool runs the twelve-person family with its life course under the policy and renders the run.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.C.1
Tests:   this file
"""

from src.bowen.engine.log_records import LogHeader, SelectionRecord
from tools import life_course_trace as tool


def test_m16c1_life_course_trace_renders():
    """Mutation: the policy source replaced by the bare script (no selections are made)."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.io.load import load_family
    from src.bowen.render.trace import render

    family = load_family(criteria.FAMILIES["phase_d"])
    assert family.life is not None and len(family.people) == 12
    run = tool.records(7, 4, family=family)
    assert isinstance(run[0], LogHeader) and run[0].seed == 7
    assert any(isinstance(r, SelectionRecord) for r in run)
    text = render(run, family.display_names)
    assert "| family instance | phase_d |" in text
    assert "| 3 |" in text and "| 4 |" not in text  # weeks 0-3
