"""The grid engine is retired (version 2 stage 1, S1.6; plan D4).

Purpose: the frozen grid engine (`src/engine.py`, its entry point, its Monte Carlo script, its 37 tests and its
         configuration document `docs/model_config.md`) is removed from the tree; it stays in git history and at the tag
         `version-1`. This test fails if any of the five files comes back or any Python file imports the engine (the
         document was covered only through the documentation index until the stage 1 review, 2026-10-19).
Spec:    docs/implementation_plan_version_2.md#S1.6; docs/bowen_agent_model_spec_v2.md#M13.2
Tests:   this file
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RETIRED = ("src/engine.py", "src/main.py", "monte_carlo.py", "tests/test_simulator.py", "docs/model_config.md")
RETIRED_MODULES = ("src.engine", "src.main", "monte_carlo")


def _python_files():
    yield from sorted(REPO.glob("*.py"))
    for part in ("src", "tools", "tests"):
        yield from sorted(p for p in (REPO / part).rglob("*.py") if "__pycache__" not in p.parts)


def _imports_retired(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        found += [n for n in names if any(n == m or n.startswith(m + ".") for m in RETIRED_MODULES)]
    return found


def test_s1_6_the_grid_engine_is_retired():
    assert [path for path in RETIRED if (REPO / path).exists()] == []
    importers = {path.relative_to(REPO).as_posix(): names for path in _python_files()
                 if (names := _imports_retired(ast.parse(path.read_text(encoding="utf-8"))))}
    assert importers == {}
