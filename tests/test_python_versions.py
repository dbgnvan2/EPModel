"""Every tracked Python file compiles on the oldest Python the suite claims to support.

Purpose: CI runs the suite on each version in `.github/workflows/tests.yml`'s matrix, but this Mac's `python3` is newer,
         so syntax that only a newer Python accepts passes here and fails in CI. On 2026-10-20 an f-string expression
         holding a backslash (accepted from 3.12, PEP 701) in `tests/bowen/test_spec_coverage.py` failed CI's 3.11 job at
         collection while the local suite passed. `ast.parse(..., feature_version=...)` does not catch that, so this
         test compiles every tracked file with the oldest listed interpreter when it is installed, and skips, naming
         it, when it is not (`pytest.ini`'s `-rfEsx` prints the reason under `-q`).
Spec:    none — repo engineering (global testing rules: every check lives in the suite; a syntax-check test for this class)
Tests:   this file
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from doc_text import tracked_files

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"
SUBPROCESS_TIMEOUT_S = 120

# Run in the child interpreter: compile each path given on the command line and print each syntax error.
_COMPILE_EACH = """
import sys
for path in sys.argv[1:]:
    try:
        compile(open(path, encoding="utf-8").read(), path, "exec", dont_inherit=True)
    except SyntaxError as e:
        print(f"{path}:{e.lineno}: {e.msg}")
"""


def _oldest_supported() -> str:
    match = re.search(r"python-version:\s*\[([^\]]*)\]", WORKFLOW.read_text(encoding="utf-8"))
    assert match, f"{WORKFLOW} has no python-version matrix"
    versions = re.findall(r'"(\d+\.\d+)"', match.group(1))
    assert versions, f"{WORKFLOW}'s python-version matrix lists no versions"
    return min(versions, key=lambda v: tuple(int(n) for n in v.split(".")))


def test_every_tracked_python_file_compiles_on_the_oldest_supported_python():
    version = _oldest_supported()
    if sys.version_info[:2] == tuple(int(n) for n in version.split(".")):
        interpreter = sys.executable
    else:
        interpreter = shutil.which(f"python{version}")
    if interpreter is None:
        pytest.skip(f"python{version} is not installed here; CI's {version} job is the check")
    files = [path for path in tracked_files(REPO) if path.endswith(".py")]
    assert "tests/test_python_versions.py" in files, "git ls-files did not list this file; the listing is broken"
    result = subprocess.run([interpreter, "-I", "-c", _COMPILE_EACH, *files], cwd=REPO, capture_output=True, text=True,
                            timeout=SUBPROCESS_TIMEOUT_S)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == []
