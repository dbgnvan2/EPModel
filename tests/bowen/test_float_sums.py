"""Every float total is exact on every Python version (version 2 stage 1, S1.1; plan D2 (b)).

Purpose: Python 3.12 changed the builtin ``sum()`` over floats to compensated summation, so a run could differ in its
         last bits between 3.11 and 3.12 wherever it summed floats. Stage 1 moved every float total in ``src/bowen`` to
         ``math.fsum`` and every integer total to ``src.bowen.engine.counting.count``, which refuses a float. These
         tests fail if a builtin ``sum`` comes back anywhere in ``src/bowen``, or if ``count`` sums a float. The scan
         left out ``render/`` (which no verdict reads, S1.8b) until the stage 1 review (2026-10-19) moved its two sums
         too, so a total printed in a trace is the same on every version.
Spec:    docs/implementation_plan_version_2.md#S1.1
Tests:   this file
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.bowen.engine.counting import count

BOWEN = Path(__file__).resolve().parents[2] / "src" / "bowen"


def builtin_sums(root: Path = BOWEN) -> list[str]:
    """Each place the name ``sum`` is read in ``root``: a call, or the builtin passed as a value."""
    found = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Name) and node.id == "sum" and isinstance(node.ctx, ast.Load):
                found.append(f"{path.relative_to(root).as_posix()}:{node.lineno}")
    return found


def test_s1_1a_no_builtin_sum_in_bowen():
    assert builtin_sums() == []


def test_s1_1a_the_scan_finds_a_sum(tmp_path):
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "a.py").write_text("x = sum([0.1, 0.2])\ny = max(map(sum, [[1]]))\n")
    (tmp_path / "render").mkdir()
    (tmp_path / "render" / "b.py").write_text("z = sum([1])\n")
    assert builtin_sums(tmp_path) == ["engine/a.py:1", "engine/a.py:2", "render/b.py:1"]


def test_s1_1b_count_refuses_a_float():
    assert count([1, 2, True, False]) == 4 and count([]) == 0 and count(np.int64(3) for _ in range(2)) == 6
    assert count(np.array([0.5, 2.0, 3.0]) > 1.0) == 2  # numpy booleans, a counted condition over an array
    for floats in ([1, 0.5], [2.0], [np.float64(1.0)]):
        with pytest.raises(TypeError, match="math.fsum"):
            count(floats)
