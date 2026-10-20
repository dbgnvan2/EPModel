"""`requirements.txt` is the set of third-party packages the code imports.

Purpose: D3. CI installs only `requirements.txt`, so a missing package fails there and an unused one installs for
         nothing. At `7aa9fc5` the file listed five packages nothing imported, and numpy twice.
Spec:    docs/plan_doc_claim_tests.md#D3
Tests:   this file
"""

from __future__ import annotations

import ast
import functools
import importlib.metadata
import re
import sys
from pathlib import Path, PurePosixPath

from doc_text import claims_data, tracked_files

REPO = Path(__file__).resolve().parents[1]

# requirements.txt exactly as it was at `7aa9fc5`, before stage 1 step 5.
REQUIREMENTS_2026_10_19 = """\
# --- System Modeling & Math ---
numpy                   # Numerical processing for emotional states
pandas                  # Data structures for process logging
networkx                # Modeling emotional triangles and systems

numpy
pygame-ce
matplotlib



# --- Engineering & Auditing ---
pytest>=8.4,<9           # Test runner; 8.x is what pytest-asyncio 1.x requires if installed
ruff                    # Linter
python-dotenv           # Environment management"""


@functools.cache
def _installed() -> dict[str, list[str]]:
    return importlib.metadata.packages_distributions()


def normalise(name: str) -> str:
    """D3.3: a distribution name as PEP 503 compares it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def local_modules(files: list[str]) -> set[str]:
    """D3.2: the repo's own top-level packages and its test-helper and root modules, derived from the tree."""
    py = [PurePosixPath(p) for p in files if p.endswith(".py")]
    return ({p.parts[0] for p in py if len(p.parts) > 1}
            | {p.stem for p in py if p.parts[0] == "tests" or len(p.parts) == 1})


def imported_names(source: str, filename: str = "<fixture>") -> set[str]:
    """D3.1: the top-level names a module imports: `import x`, absolute `from x import y`, and a literal argument to
    `importlib.import_module` or `__import__`."""
    names = set()
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
              and getattr(node.func, "attr", getattr(node.func, "id", None)) in ("import_module", "__import__")):
            names.add(node.args[0].value)
    return {name.split(".")[0] for name in names}


def imported_distributions(sources: dict[str, str], files: list[str]) -> dict[str, set[str]]:
    """D3.2, D3.3: each third-party distribution imported, with the files that import it."""
    skip = set(sys.stdlib_module_names) | {"__future__"} | local_modules(files)
    installed = _installed()
    found: dict[str, set[str]] = {}
    for path, source in sources.items():
        for name in imported_names(source, path) - skip:
            # An import that is not installed keeps its own name, so it is still reported (D3.3).
            found.setdefault(normalise(sorted(installed.get(name, [name]))[0]), set()).add(path)
    return found


def requirement_names(text: str) -> list[str]:
    """D3.4: the distribution names `requirements.txt` lists, in order, duplicates kept."""
    lines = (line.split("#", 1)[0].strip() for line in text.splitlines())
    return [normalise(re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line).group(0)) for line in lines if line]


def requirement_findings(sources: dict[str, str], files: list[str], requirements: str,
                         not_imported: dict[str, str]) -> dict[str, list[str]]:
    """D3.4–D3.6: every way the requirements and the imports disagree. Empty when they agree."""
    listed = requirement_names(requirements)
    allowed = {normalise(name) for name in not_imported}
    imported = imported_distributions(sources, files)
    findings = {
        "listed twice in requirements.txt": sorted({n for n in listed if listed.count(n) > 1}),
        "allowlisted in tests/doc_claims.json, not in requirements.txt": sorted(allowed - set(listed)),
        "imported, not in requirements.txt": [f"{n} ({', '.join(sorted(imported[n]))})"
                                              for n in sorted(set(imported) - set(listed))],
        "in requirements.txt, not imported": sorted(set(listed) - allowed - set(imported)),
    }
    return {kind: names for kind, names in findings.items() if names}


def _repo_sources(files: list[str]) -> dict[str, str]:
    return {p: (REPO / p).read_text(encoding="utf-8") for p in files if p.endswith(".py") and (REPO / p).is_file()}


def _tools_not_imported() -> dict[str, str]:
    return {k: v for k, v in claims_data()["tools_not_imported"].items() if not k.startswith("_")}


def test_d3_requirements_match_the_imports():
    """D3.6: the real repo."""
    files = tracked_files()
    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8")
    findings = requirement_findings(_repo_sources(files), files, requirements, _tools_not_imported())
    assert findings == {}, f"{findings}; a new local module is read from git, so stage it (git add) first"


def test_d3_the_requirements_file_of_2026_10_19_is_reported():
    """The file as it was at `7aa9fc5`: five packages nothing imports, and numpy listed twice. The sources are fixed
    here, so a later import of one of the five does not change what this test expects."""
    sources = {"src/a.py": "import numpy\n", "tests/test_a.py": "import pytest\n"}
    findings = requirement_findings(sources, sorted(sources), REQUIREMENTS_2026_10_19, {"ruff": "the linter"})
    assert findings == {
        "listed twice in requirements.txt": ["numpy"],
        "in requirements.txt, not imported": ["matplotlib", "networkx", "pandas", "pygame-ce", "python-dotenv"],
    }


def test_d3_an_unlisted_import_is_reported():
    """D3.3, D3.6: a module that imports packages the requirements do not list. `_pytest` is installed and is reported
    by its distribution's name, `pytest`; a package that is not installed is reported by its own name."""
    files = ["tools/new_tool.py"]
    sources = {"tools/new_tool.py": "import numpy\nimport _pytest\nimport not_installed_pkg\n"}
    findings = requirement_findings(sources, files, "numpy\nruff\n", {"ruff": "the linter"})
    assert findings == {"imported, not in requirements.txt": ["not-installed-pkg (tools/new_tool.py)",
                                                              "pytest (tools/new_tool.py)"]}


def test_d3_local_stdlib_relative_and_helper_imports_are_not_third_party():
    """D3.2 (adversarial): imports that look third-party to a naive scan and are not."""
    files = ["src/bowen/engine.py", "tools/spec_coverage.py", "tests/bowen/act_literals.py", "tests/test_x.py"]
    source = ("from __future__ import annotations\nfrom . import policy\nimport src.bowen\nimport act_literals\n"
              "import tomllib\nimport importlib\nimportlib.import_module('tools.spec_coverage')\n")
    assert imported_distributions({"tests/test_x.py": source}, files) == {}
    assert imported_names(source) >= {"src", "act_literals", "tomllib", "tools"}


def test_d3_dynamic_imports_with_a_literal_are_read():
    """D3.1: `importlib.import_module` and `__import__` with a literal argument count as imports."""
    source = "import importlib\nimportlib.import_module('numpy.random')\n__import__('pytest')\n"
    assert imported_names(source) == {"importlib", "numpy", "pytest"}


def test_d3_an_allowlisted_tool_must_be_listed():
    """D3.5: an installed-not-imported entry whose package is not in requirements.txt fails."""
    files = ["tools/a.py"]
    findings = requirement_findings({"tools/a.py": "import numpy\n"}, files, "numpy\n", {"ruff": "the linter"})
    assert findings == {"allowlisted in tests/doc_claims.json, not in requirements.txt": ["ruff"]}


def test_d3_requirement_lines_are_parsed_to_names():
    """D3.4: comments, versions, extras and markers are not part of the name."""
    text = "# c\n\nnumpy>=1.26  # x\nPyYAML[extra]==6.0 ; python_version>'3.10'\npytest-cov\n"
    assert requirement_names(text) == ["numpy", "pyyaml", "pytest-cov"]
