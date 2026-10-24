"""Every test a `Tests:` line or the spec names exists (milestone review MR-9, systemic pattern SP-B).

Purpose: a spec row or a docstring's `Tests:` line can name a test that does not exist, and the suite stays green,
         because coverage takes its evidence from test names, not from these references. This gate resolves every
         name: (a) in the `Tests:` blocks of the docstrings under src/, tools/ and tests/ (the `Tests:` line and its
         continuation lines, indented deeper); (b) in the spec, outside its fenced examples. A ``path::name`` must
         name a test function defined in that file; a bare ``test_...`` name must be defined somewhere under tests/;
         a ``tests/...py`` path in a `Tests:` block must exist. A template such as ``test_m4c2_…`` (a name ending in
         ``_``) is not a name.
Spec:    docs/bowen_agent_model_spec_v2.md#M14.1 (the coverage report's evidence), §0.4 (the docstring convention)
Tests:   this file

The allowlists below hold every name that does not resolve today, each with its reason; a name on a list that
resolves, or that is no longer named, fails the gate too, so the lists only shrink.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "docs" / "bowen_agent_model_spec_v2.md"
SOURCES = ("src", "tools", "tests")

T129 = "fixed with the next change that moves the engine hash (T-129)"
NOT_WRITTEN = "names a test of a built requirement that is not written yet (T-004)"
NOT_BUILT = "names the test of a requirement not built yet; written when it is built"

# `Tests:` references in src/bowen/ docstrings: an edit there moves the engine hash, so they wait.
SOURCE_ALLOWED = {
    "src/bowen/engine/rejection.py: tests/bowen/test_rejection.py::test_m1b3c_faded_cutoff_waits_for_a_positive_contact":
        T129,
}

# Names in the spec that do not resolve: tests T-004 lists as named by the spec and not written (M1.F.2's needs a
# test of its own clause; M11.D.12 is not built), and tests the spec names for requirements that are not built
# (spec_coverage.md: not done), which are written when the requirement is.
SPEC_ALLOWED = {
    "test_m1f2_same_kind_moves_receiver_on_opposite_sides": NOT_WRITTEN,  # M1.F.2, T-004
    "test_m11d12_corpus_magnitudes_are_not_config_keys": NOT_WRITTEN,  # M11.D.12, T-004
    "test_m11c9_position_effect_peaks_midscale": NOT_BUILT,  # M11.C.9, out of version 1
    "test_m11c28_sink_mobility_is_protective": NOT_BUILT,  # M11.C.28
    "test_m17b4_time_to_event_report_does_not_collapse_a_bimodal_fixture": NOT_BUILT,  # Phase E
    "test_m17d2_rival_arm_is_run_and_reported": NOT_BUILT,
    "test_m17d2_criterion_asserts_which_member_not_that_members_differ": NOT_BUILT,
    "test_m17d2_non_discriminating_criterion_is_reported_as_such": NOT_BUILT,
    "test_m17d4_null_arm_observable_stays_below_declared_floor": NOT_BUILT,
    "test_m17e2_null_value_removes_effect": NOT_BUILT,
    "test_m17e2_reports_monotonicity": NOT_BUILT,
    "test_m17e2_flags_rising_edge_operating_point": NOT_BUILT,
    "test_m17f1_reports_trajectory_through_event": NOT_BUILT,
    "test_m17f1_time_to_event_is_a_distribution": NOT_BUILT,
    "test_m17f1_reports_non_occurrence_fraction": NOT_BUILT,
    "test_m17f2_readout_refuses_to_report_before_settling_condition_is_met": NOT_BUILT,
}

NAME = re.compile(r"\b(test_\w+)\b(?!\.py)")
PATH_NAME = re.compile(r"((?:[\w.-]+/)*test_\w+\.py)::(test_\w+)")
TEST_PATH = re.compile(r"\b(tests/(?:[\w-]+/)*[\w-]+\.py)\b")


def defined_tests() -> dict[str, set[str]]:
    """Test function name -> the test files (repo-relative) that define it, methods included."""
    found: dict[str, set[str]] = {}
    for path in sorted((REPO / "tests").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                found.setdefault(node.name, set()).add(path.relative_to(REPO).as_posix())
    return found


def _tests_blocks(text: str) -> list[str]:
    """Each `Tests:` line with its continuation lines (indented deeper than it), as one string."""
    lines, blocks = text.splitlines(), []
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)Tests:", line)
        if not m:
            continue
        indent, block = len(m.group(1)), [line]
        for nxt in lines[i + 1:]:
            if not nxt.strip() or len(nxt) - len(nxt.lstrip()) <= indent:
                break
            block.append(nxt)
        blocks.append("\n".join(block))
    return blocks


def unresolved(text: str, defined: dict[str, set[str]], tests_block: bool) -> list[str]:
    """The names, ``path::name`` pairs and (in a `Tests:` block) test file paths in ``text`` that do not resolve."""
    bad = []
    for path, name in PATH_NAME.findall(text):
        files = defined.get(name, set())
        if not any(f == path or f.endswith("/" + path) for f in files):
            bad.append(f"{path}::{name}")
    for name in NAME.findall(PATH_NAME.sub("", text)):
        if not name.endswith("_") and name not in defined:
            bad.append(name)
    if tests_block:
        bad += [p for p in TEST_PATH.findall(text) if not (REPO / p).exists()]
    return bad


def spec_text_outside_fences(text: str) -> str:
    return re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)


def source_faults(defined: dict[str, set[str]]) -> list[str]:
    faults = []
    for root in SOURCES:
        for path in sorted((REPO / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO).as_posix()
            for doc in docstrings(path.read_text(encoding="utf-8")):
                for block in _tests_blocks(doc):
                    faults += [f"{rel}: {ref}" for ref in unresolved(block, defined, tests_block=True)]
    return faults


def docstrings(source: str) -> list[str]:
    """The module's, each class's and each function's docstring; a `Tests:` line inside a string a test builds as a
    fixture is not a reference."""
    kinds = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    return [doc for node in ast.walk(ast.parse(source)) if isinstance(node, kinds)
            for doc in [ast.get_docstring(node)] if doc]


def spec_faults(defined: dict[str, set[str]]) -> list[str]:
    return unresolved(spec_text_outside_fences(SPEC.read_text(encoding="utf-8")), defined, tests_block=False)


def test_m141_every_test_a_tests_line_names_exists():
    """(a): every `Tests:` reference under src/, tools/ and tests/ resolves, but those allowed above."""
    faults = source_faults(defined_tests())
    assert sorted(set(faults) - set(SOURCE_ALLOWED)) == []
    assert sorted(set(SOURCE_ALLOWED) - set(faults)) == [], "resolved or no longer named: take it off SOURCE_ALLOWED"


def test_m141_every_test_the_spec_names_exists():
    """(b): every test name and ``path::name`` in the spec resolves, but those allowed above."""
    faults = spec_faults(defined_tests())
    assert sorted(set(faults) - set(SPEC_ALLOWED)) == []
    assert sorted(set(SPEC_ALLOWED) - set(faults)) == [], "resolved or no longer named: take it off SPEC_ALLOWED"


def test_m141_the_gate_reads_each_form():
    """The gate is not vacuous: a fabricated name fails in each form it reads, a real one passes, a template and a
    fenced example are not read, and a `Tests:` block's continuation line is read."""
    defined = {"test_real": {"tests/bowen/test_x.py"}}
    assert unresolved("`test_real` and tests/bowen/test_x.py::test_real", defined, False) == []
    assert unresolved("`test_fake_name`", defined, False) == ["test_fake_name"]
    assert unresolved("tests/bowen/test_y.py::test_real", defined, False) == ["tests/bowen/test_y.py::test_real"]
    assert unresolved("test_x.py::test_fake", defined, False) == ["test_x.py::test_fake"]
    assert unresolved("`test_m4c2_…`", defined, False) == []
    assert unresolved("tests/test_no_such_file.py", defined, True) == ["tests/test_no_such_file.py"]
    assert spec_text_outside_fences("a\n```\nTests: test_fake\n```\nb\n") == "a\n\nb\n"
    block = _tests_blocks('"""\nTests:   tests/a.py::test_one,\n         tests/b.py::test_two\n\nTests: x\n"""')
    assert block[0].endswith("tests/b.py::test_two") and len(block) == 2
