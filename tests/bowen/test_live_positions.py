"""The live-position predicate.

Purpose: prove M8's counting rule, its three role consequences, and that it is
         implemented exactly once.
Spec:    docs/bowen_agent_model_spec_v2.md#M8.1, #M8.2, #M8.3, #M8.4, #M8.5
Tests:   this file
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.live_positions import Occupant, avoidance_available, positions_live

REPO = Path(__file__).resolve().parents[2]
RAVI, MARTA, NADIA, COACH = (PersonId(n) for n in ("ravi", "marta", "nadia", "coach"))


def trio(**third) -> list[Occupant]:
    return [Occupant(RAVI), Occupant(MARTA), Occupant(NADIA, **third)]


def test_m81_positions_live_counts_unfused_present_occupants():
    assert positions_live(trio()) == 3
    assert positions_live(trio(present=False)) == 2
    assert positions_live(trio(fused_into=MARTA)) == 2


def test_m81_fusion_outside_the_group_does_not_remove_a_position():
    """The rule counts fusion into another member *of the group* only."""
    outsider = PersonId("ana")
    assert positions_live(trio(fused_into=outsider)) == 3


def test_m81_avoidance_is_a_step_at_three():
    assert avoidance_available(trio()) is False
    assert avoidance_available(trio(fused_into=MARTA)) is True
    assert avoidance_available([Occupant(RAVI), Occupant(MARTA)]) is True
    four = trio() + [Occupant(COACH)]
    assert avoidance_available(four) is False


def test_m82_neutral_external_in_outside_position_counts():
    group = [Occupant(RAVI), Occupant(MARTA), Occupant(COACH)]
    assert positions_live(group) == 3 and not avoidance_available(group)


def test_m83_external_who_has_taken_a_side_does_not_count():
    group = [Occupant(RAVI), Occupant(MARTA), Occupant(COACH, fused_into=MARTA)]
    assert positions_live(group) == 2 and avoidance_available(group)


def test_m84_displaced_inactive_member_does_not_count():
    assert positions_live(trio(emotionally_active=False)) == 2


def test_m81_input_validation():
    with pytest.raises(ValueError, match="twice"):
        positions_live([Occupant(RAVI), Occupant(RAVI)])
    with pytest.raises(ValueError, match="themselves"):
        Occupant(RAVI, fused_into=RAVI)


def test_m85_single_implementation():
    """Static: one definition of each predicate in src/bowen, and no second copy of its arithmetic."""
    definitions = []
    copies = []
    for path in sorted((REPO / "src" / "bowen").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        definitions += [
            f"{path.name}:{node.name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in {"positions_live", "avoidance_available"}
        ]
        if path.name != "live_positions.py" and re.search(r"fused_into", source):
            copies.append(path.name)
    assert sorted(definitions) == ["live_positions.py:avoidance_available", "live_positions.py:positions_live"]
    assert copies == [], f"fusion is read outside live_positions.py: {copies}"
