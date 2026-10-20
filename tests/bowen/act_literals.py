"""Find literal sets of act names in a module's source: a classification of acts written in code (rule 9).

Purpose: let a test assert that a module reads its act classifications from config instead of holding them as
         literals (version 2 stage 1, S1.0 and S1.3a).
Spec:    docs/implementation_plan_version_2.md#S1.0, #S1.3a
Tests:   tests/bowen/test_patterns.py::test_s1_0_act_sets_come_from_config,
         tests/bowen/test_config.py::test_s1_3a_act_classes_come_from_config

A set display (``{"PURSUE", "CONFLICT"}``), or a ``set``/``frozenset`` call over a list or tuple display, whose items
are all string constants naming declared moves. A tuple assigned to names (``CUTOFF, TRIANGLE = "CUTOFF", "TRIANGLE"``)
is naming the spec's acts, not classifying them, and is not reported.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.bowen.io.load import load_event_kinds


def _all_moves(nodes, moves) -> bool:
    return bool(nodes) and all(isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in moves
                               for n in nodes)


def literal_act_sets(path: Path) -> list[tuple[int, list[str]]]:
    """Each literal set of declared act names in ``path``, as (line, names)."""
    moves = load_event_kinds().moves()
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        items = None
        if isinstance(node, ast.Set):
            items = node.elts
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("set", "frozenset")
              and len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple))):
            items = node.args[0].elts
        if items is not None and _all_moves(items, moves):
            found.append((node.lineno, [n.value for n in items]))
    return found
