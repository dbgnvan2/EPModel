"""Find literal sets of act names in a module's source: a classification of acts written in code (rule 9).

Purpose: let a test assert that a module reads its act classifications from config instead of holding them as
         literals (version 2 stage 1, S1.0 and S1.3a).
Spec:    docs/implementation_plan_version_2.md#S1.0, #S1.3a
Tests:   tests/bowen/test_patterns.py::test_s1_0_act_sets_come_from_config,
         tests/bowen/test_config.py::test_s1_3a_act_classes_come_from_config

A set, list or tuple display (``{"PURSUE", "CONFLICT"}``, ``("PURSUE", "CONFLICT")``) whose items are all string
constants naming declared moves, so a ``set``/``frozenset`` call over one is found too; and a comparison with a string
constant naming a declared move (``kind == "SPLIT"``), a class of one act (stage 1 review, 2026-10-19). A tuple
assigned to names (``CUTOFF, TRIANGLE = "CUTOFF", "TRIANGLE"``) is naming the spec's acts, not classifying them, and is
not reported.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.bowen.io.load import load_event_kinds


def _all_moves(nodes, moves) -> bool:
    return bool(nodes) and all(isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in moves
                               for n in nodes)


def literal_act_sets(path: Path) -> list[tuple[int, list[str]]]:
    """Each literal set of declared act names in ``path``, and each comparison with one, as (line, names)."""
    moves = load_event_kinds().moves()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    naming = {id(node.value) for node in ast.walk(tree)  # NAME_A, NAME_B = "A", "B": names, not a class
              if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple)
              and all(isinstance(target, ast.Tuple) for target in node.targets)}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)) and id(node) not in naming:
            items = node.elts
        elif isinstance(node, ast.Compare):
            items = [n for n in (node.left, *node.comparators) if _all_moves([n], moves)]
        else:
            continue
        if _all_moves(items, moves):
            found.append((node.lineno, [n.value for n in items]))
    return sorted(found)
