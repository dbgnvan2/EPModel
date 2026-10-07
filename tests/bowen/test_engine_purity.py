"""Engine purity.

Purpose: the engine writes nothing to disk (G9), contains no file I/O or UI
         (G4), and imports nothing from tests, readouts, rendering,
         configuration or I/O (G13).
Spec:    docs/bowen_agent_model_spec_v2.md#M16.T.2, #M11.D.1, #M11.D.22, #M16.B.1, #M16.B.2, #M3.D.6
Tests:   this file
"""

from __future__ import annotations

import ast
import builtins
import io
import os
from pathlib import Path

import pytest

from src.bowen.engine.log_records import CollectingEmitter
from src.bowen.engine.tick import run
from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_script
from src.bowen.scenario.assemble import assemble


def test_m16t2_engine_writes_nothing(tmp_path, monkeypatch):
    """G9: run the engine in an empty working directory with every write path trapped."""
    constants, kinds, family = load_constants(), load_event_kinds(), load_family()
    parts = assemble(constants, kinds, family, load_script(kinds=kinds, family=family), seed=7)
    monkeypatch.chdir(tmp_path)
    writes = []
    real_open, real_io_open, real_os_open = builtins.open, io.open, os.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            writes.append((file, mode))
        return real_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND):
            writes.append((path, flags))
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    run(parts.state, parts.source, parts.params, parts.visibility, parts.activation, CollectingEmitter(), ticks=40)
    assert writes == []
    assert list(tmp_path.iterdir()) == []


ENGINE = Path(__file__).resolve().parents[2] / "src" / "bowen" / "engine"

# M11.D.1: file I/O, process and network access, and UI toolkits.
FORBIDDEN_MODULES = {
    "os", "sys", "io", "shutil", "pathlib", "subprocess", "socket", "tempfile", "glob", "csv", "pickle",
    "sqlite3", "urllib", "requests", "logging", "tkinter", "pygame", "matplotlib", "builtins", "importlib",
}
FORBIDDEN_CALLS = {"open", "print", "input", "exec", "eval", "__import__", "breakpoint"}
FORBIDDEN_METHODS = {"write", "write_text", "write_bytes", "mkdir", "makedirs", "unlink", "touch",
                     "dump", "savetxt", "save", "tofile", "remove", "rmdir",
                     # reads too (review, 2026-10-06): the engine takes everything from its caller
                     "load", "loadtxt", "genfromtxt", "fromfile", "read_text", "read_bytes", "import_module"}
# M11.D.22 and the plan's layout: the engine depends on nothing outside itself.
FORBIDDEN_IMPORT_PREFIXES = ("tests", "src.bowen.render", "src.bowen.readouts", "src.bowen.scenario",
                             "src.bowen.io", "src.bowen.run", "src.engine", "src.main")
# M3.D.6: no language model in the decision path.
LLM_MODULES = {"anthropic", "openai", "transformers", "langchain", "llama_index", "ollama"}


def _imports(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def engine_io_findings(root: Path = ENGINE) -> list[str]:
    findings = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _imports(tree):
            top = module.split(".")[0]
            if top in FORBIDDEN_MODULES or top in LLM_MODULES:
                findings.append(f"{path.name}: imports {module}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                    findings.append(f"{path.name}:{node.lineno}: calls {node.func.id}()")
                if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_METHODS:
                    findings.append(f"{path.name}:{node.lineno}: calls .{node.func.attr}()")
    return findings


def engine_import_findings(root: Path = ENGINE) -> list[str]:
    return [
        f"{path.name}: imports {module}"
        for path in sorted(root.rglob("*.py"))
        for module in _imports(ast.parse(path.read_text(encoding="utf-8")))
        if module.startswith(FORBIDDEN_IMPORT_PREFIXES)
    ]


def test_m11d1_engine_has_no_io():
    """G4: no file I/O, no process or network access, no UI and no LLM anywhere in the engine package."""
    assert engine_io_findings() == []


def test_m11d22_engine_modules_do_not_import_tests_or_readouts():
    """G13: the engine imports only itself, the standard library and NumPy."""
    assert engine_import_findings() == []


def test_m11d1_the_scans_find_what_they_look_for(tmp_path):
    """The scanners are proved able to fail on a planted module."""
    (tmp_path / "bad.py").write_text(
        "import os\nfrom src.bowen.render.trace import render\nopen('x', 'w')\nprint(1)\n"
        "import json\njson.dump({}, None)\nimport numpy as np\nnp.load('x.npy')\n"
    )
    io_found = engine_io_findings(tmp_path)
    assert any("imports os" in f for f in io_found)
    assert any("calls open()" in f for f in io_found) and any("calls print()" in f for f in io_found)
    assert any(".dump()" in f for f in io_found) and any(".load()" in f for f in io_found)
    assert engine_import_findings(tmp_path) == ["bad.py: imports src.bowen.render.trace"]


SOURCED_WORDS = ("bowen says", "sourced", "theoretically grounded", "derived from bowen", "from the corpus",
                 "calibrated to", "empirically")


def invented_constant_findings(root: Path = ENGINE.parent) -> list[str]:
    from src.bowen.scenario.constants import SCHEMA

    invented = [k for k, spec in SCHEMA.items() if spec.grade == "[I]"]
    findings = []
    for path in sorted(root.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lowered = line.lower()
            if any(k in line for k in invented) and any(w in lowered for w in SOURCED_WORDS):
                findings.append(f"{path.name}:{number}: {line.strip()}")
    return findings


def test_m11d4_invented_constants_labelled():
    """No line naming an [I] constant describes it as sourced (M0.2). Phase C gate; built at Phase B."""
    from src.bowen.io.load import load_constants

    constants = load_constants()
    assert all(c.grade == "[I]" for c in constants.values.values())
    assert invented_constant_findings() == []


def test_m11d4_scan_finds_a_constant_described_as_sourced(tmp_path):
    (tmp_path / "bad.py").write_text("standing_load_gain = 0.2  # calibrated to Bowen's Ch02\n")
    assert len(invented_constant_findings(tmp_path)) == 1
