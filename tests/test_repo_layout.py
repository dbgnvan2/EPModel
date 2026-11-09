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


# --- SC-11 (spec revision 55): no role in a config key or a Person field (M1.A.14b) -------------------------------


def _role_words() -> set[str]:
    import re

    text = (REPO / "config" / "lint" / "role_labels.md").read_text(encoding="utf-8")
    return {m.group(1).lower() for m in re.finditer(r"^\| `([^`]+)` \|", text, re.M)}


def _words(name: str) -> set[str]:
    import re

    return {w.lower() for w in re.split(r"[_\W]+", name) if w}


def test_m1a14b_sc11_no_role_field():
    """No config key under ``config/bowen/`` and no ``Person`` field names a role; who supplies whom is a readout
    computed after the run (`M11.G.5`), and no engine or policy module imports it."""
    import re

    roles = _role_words()
    assert {"caretaker", "scapegoat", "helper", "parentified"} <= roles
    keys = set()
    for path in sorted((REPO / "config" / "bowen").rglob("*.md")):
        keys |= {m.group(1) for m in re.finditer(r"^\| `([A-Za-z0-9_.\-]+)` \|", path.read_text(encoding="utf-8"), re.M)}
    tree = ast.parse((REPO / "src" / "bowen" / "engine" / "objects.py").read_text(encoding="utf-8"))
    person = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Person")
    fields = {s.target.id for s in person.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
    named = sorted(name for name in keys | fields if _words(name) & roles)
    assert named == [], f"a config key or Person field names a role: {named}"
    for folder in ("engine", "policy"):
        for path in sorted((REPO / "src" / "bowen" / folder).rglob("*.py")):
            assert "readouts.roles" not in path.read_text(encoding="utf-8"), path
