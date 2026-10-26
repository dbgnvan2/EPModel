"""The source is clean under ruff's pyflakes rules (version 2 stage 3, T-129).

Purpose: keep `src/`, `tools/` and `tests/` free of unused imports, undefined names and the rest of ruff's `F` rules,
         and SP-B's allowlist of unresolved `Tests:` lines in `src/bowen/` empty, from the commit that fixes them on.
Spec:    docs/implementation_plan_version_2_stage_3.md#3.7 (S3.22, T-129)
Tests:   this file

Written red at plan step 5; plan step 6, the first commit of the batch to move the engine hash, turned it green.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

from test_named_tests_resolve import SOURCE_ALLOWED  # tests/ is on the path: pytest imports test modules by basename

REPO = Path(__file__).resolve().parents[1]


def test_s3_22_ruff_f_clean():
    """S3.22: `ruff check --select F src tools tests` reports nothing, and `SOURCE_ALLOWED` is empty."""
    ruff = [sys.executable, "-m", "ruff"] if importlib.util.find_spec("ruff") else [shutil.which("ruff")]
    assert ruff[-1] is not None, "ruff is not installed (requirements.txt lists it)"
    found = subprocess.run([*ruff, "check", "--select", "F", "--output-format", "concise", "src", "tools", "tests"],
                           cwd=REPO, capture_output=True, text=True, timeout=120)
    assert found.returncode == 0, found.stdout[-4000:] + found.stderr[-1000:]
    assert SOURCE_ALLOWED == {}, sorted(SOURCE_ALLOWED)
