"""ScriptedSource and the Phase B script.

Purpose: prove the script parses strictly and is validated against the event
         kinds and the family.
Spec:    docs/bowen_agent_model_spec_v2.md#M13, #M1.F.6, #M10.B.1, #M10.B.2
Tests:   this file
"""

from __future__ import annotations

import pytest

from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.io.load import CONFIG_DIR, load_event_kinds, load_family, load_script
from src.bowen.scenario.config_parse import ConfigError
from src.bowen.scenario.scripted_source import build_script

TEXT = (CONFIG_DIR / "script_phase_b.md").read_text(encoding="utf-8")


def build(text):
    return build_script(text, load_event_kinds(), load_family())


def edit(old, new):
    assert TEXT.count(old) == 1, old
    return TEXT.replace(old, new)


def test_m13_script_parses_and_validates():
    script = load_script()
    assert script.ticks == 40
    trigger = [e for e in script.events if e.kind == "TRIGGER"]
    assert len(trigger) == 1 and trigger[0].on_tie == TieId.of(PersonId("ana"), PersonId("bruno"))
    assert trigger[0].timestamp == 10 and trigger[0].mechanism is Mechanism.TRIGGER
    assert all(s.kind in load_event_kinds().moves() for _, s in script.moves)


def test_m1f6_scripted_stressors_are_spells():
    job_loss = [e for e in load_script().events if e.kind == "JOB_LOSS"][0]
    assert (job_loss.timestamp, job_loss.duration, job_loss.exogenous) == (0, 34, True)


def test_m13_without_removes_one_kind_only():
    script = load_script()
    other = script.without("TRIGGER")
    assert [e.kind for e in other.events] == ["JOB_LOSS"] and other.moves == script.moves


@pytest.mark.parametrize(
    "old, new, message",
    [
        ("| 1 | `ravi` | CONFLICT |", "| 1 | `teodor` | CONFLICT |", "not in phase_b_reduced"),
        ("| 1 | `ravi` | CONFLICT |", "| 1 | `ravi` | SULK |", "unknown event kind"),
        ("| 0 | JOB_LOSS |", "| 0 | CONFLICT |", "is a move"),
        ("| 1 | `ravi` | CONFLICT |", "| 1 | `ravi` | TRIGGER |", "not a move"),
        ("| 36 | `ravi` | PURSUE | `sofia` |", "| 36 | `ravi` | PURSUE | `bruno` |", "no tie"),
        ("| 36 | `ravi` | PURSUE |", "| 40 | `ravi` | PURSUE |", "outside the script"),
        ("| 3 | `marta` | DISTANCE |", "| 2 | `marta` | DISTANCE |", "already moves"),
        ("| `ana`~`bruno` |", "| `ana`~`sofia` |", "no tie"),
        ("| 0 | JOB_LOSS | `ravi` | — | 400 | 34 |", "| 0 | JOB_LOSS | `ravi` | — | 400 | 0 |", "duration"),
        ("grade: [I]", "grade: [T]", "grade"),
        ("instance_id: phase_b_reduced", "instance_id: reference_family", "written for"),
        ("| 1 | `ravi` | CONFLICT | `marta` | 200 | — | none |", "| 1 | `ravi` | CONFLICT | `marta` | 200 | — | sideways |", "source_position"),
    ],
)
def test_m10b2_script_parses_strictly(old, new, message):
    with pytest.raises(ConfigError, match=message):
        build(edit(old, new))
