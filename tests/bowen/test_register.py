"""The M14.A register describes the code, not a plan of it.

Purpose: fail when an object field has no register row, or a row names no field.
Spec:    docs/bowen_agent_model_spec_v2.md#M14.A.4, #M11.D.17
Tests:   this file

`test_m11d17_register_has_no_orphans` checks the register's own structure. It
cannot see the code, so on its own a register could stay internally consistent
while describing objects that no longer exist (P6). This test ties the two.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

from src.bowen.engine.objects import Family, Person, Relationship, Triangle

# Loaded by path: the name `tests` is shadowed by an installed package's own
# top-level `tests`, so a package import would find the wrong module.
_CONSISTENCY = Path(__file__).resolve().parents[1] / "test_spec_consistency.py"
_spec = importlib.util.spec_from_file_location("spec_consistency_for_register", _CONSISTENCY)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_register_tables, _spec_text = _module._register_tables, _module._spec_text

CLASSES = {"Person": Person, "Relationship": Relationship, "Triangle": Triangle, "Family": Family}


def register_fields(text: str) -> dict[str, set[str]]:
    variables, _ = _register_tables(text)
    by_owner: dict[str, set[str]] = {name: set() for name in CLASSES}
    for name, owner, *_ in variables:
        by_owner.setdefault(owner, set()).add(name.strip("`"))
    return by_owner


def mismatches(text: str) -> dict[str, dict[str, list[str]]]:
    registered = register_fields(text)
    problems = {}
    for owner, cls in CLASSES.items():
        code = {f.name for f in dataclasses.fields(cls)}
        rows = registered.get(owner, set())
        missing_row = sorted(code - rows)
        missing_field = sorted(rows - code)
        if missing_row or missing_field:
            problems[owner] = {"no register row": missing_row, "no code field": missing_field}
    unknown_owners = sorted(set(registered) - set(CLASSES))
    if unknown_owners:
        problems["unknown owners"] = {"owners": unknown_owners}
    return problems


def test_m14a_register_matches_object_fields():
    assert mismatches(_spec_text()) == {}


def test_m14a_check_catches_a_field_with_no_row():
    """Mutation: a register that lost a row must be reported."""
    text = _spec_text().replace("| `taboo_set` | Relationship |", "| `taboo_set_renamed` | Relationship |")
    problems = mismatches(text)
    assert problems["Relationship"]["no register row"] == ["taboo_set"]
    assert problems["Relationship"]["no code field"] == ["taboo_set_renamed"]
