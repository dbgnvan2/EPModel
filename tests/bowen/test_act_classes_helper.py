"""T-063: "self-directed or a hold" is one check, ``act.holds_or_self_directed``, used by every site that asks it.

Purpose: keep the four sites that judged it (``external.record_stances``, ``external.self_practice``,
         ``stress_response``, ``slow_tick``) from drifting apart.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.5, #M1.E.7, #M1.A.5e
"""

from __future__ import annotations

import inspect

from src.bowen.engine import external, slow_tick, stress_response
from src.bowen.engine.act import WITHHOLD, holds_or_self_directed
from src.bowen.io.load import load_event_kinds

KINDS = load_event_kinds()


def test_t063_a_hold_or_a_self_directed_act_and_nothing_else():
    assert holds_or_self_directed(KINDS, WITHHOLD)
    assert holds_or_self_directed(KINDS, "STAY-IN-CONTACT")      # the self channel
    assert not holds_or_self_directed(KINDS, "CONFLICT")         # automatic
    assert not holds_or_self_directed(KINDS, None)               # no act


def test_t063_no_site_writes_the_check_out_again():
    for module in (external, slow_tick, stress_response):
        source = inspect.getsource(module)
        assert "== WITHHOLD or" not in source, module.__name__
