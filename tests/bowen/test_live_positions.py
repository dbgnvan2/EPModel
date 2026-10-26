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
from src.bowen.engine.live_positions import Occupant, positions_live

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


def test_m82_neutral_external_in_outside_position_counts():
    group = [Occupant(RAVI), Occupant(MARTA), Occupant(COACH)]
    assert positions_live(group) == 3


def test_m83_external_who_has_taken_a_side_does_not_count():
    group = [Occupant(RAVI), Occupant(MARTA), Occupant(COACH, fused_into=MARTA)]
    assert positions_live(group) == 2


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
            if isinstance(node, ast.FunctionDef) and node.name == "positions_live"
        ]
        # A caller may describe occupants (and so mention fused_into), but only
        # through the predicate: it must import from live_positions.
        if (
            path.name != "live_positions.py"
            and re.search(r"fused_into", source)
            and "from src.bowen.engine.live_positions import" not in source
        ):
            copies.append(path.name)
    assert definitions == ["live_positions.py:positions_live"]
    assert copies == [], f"fusion is read without the predicate: {copies}"


# --- Version 2 stage 3, T-014 (spec revision 39, D27): written first, red at plan step 5, green at step 6 -----------


def _calls_in(path: Path) -> set[str]:
    """The names of the functions called anywhere in ``path`` (a bare name or an attribute's last part)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))}


def test_m8_s3_20_predicate_has_three_callers():
    """S3.20 (`M8`, D27): `positions_live` is called from exactly three modules — the witness test (`visibility.py`),
    the coach's position (`external.py`) and triangle activity (`recompute.py`) — and no function beside it in
    `live_positions.py` goes uncalled in `src/`; `M8`'s preamble says the rule is written for three places."""
    sources = sorted((REPO / "src").rglob("*.py"))
    callers = {path.name for path in sources if "positions_live" in _calls_in(path)}
    assert callers == {"visibility.py", "external.py", "recompute.py"}, sorted(callers)
    module = REPO / "src" / "bowen" / "engine" / "live_positions.py"
    defined = {node.name for node in ast.parse(module.read_text(encoding="utf-8")).body
               if isinstance(node, ast.FunctionDef)}
    called = set().union(*(_calls_in(path) for path in sources))
    assert sorted(defined - called - {"positions_live"}) == [], "an uncalled function in live_positions.py"
    preamble = (REPO / "docs" / "spec_rules" / "M08.md").read_text(encoding="utf-8")
    assert "written for three places" in preamble and "written for four places" not in preamble
