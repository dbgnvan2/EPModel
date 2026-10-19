"""Version 1's status page is frozen at the tag `version-1`, and the checks that made it final still pass there.

Purpose: version 1 ends at the tag `version-1` (d62cfdc, owner 2026-10-19). Version 2 regenerates the records the
         page is rendered from, so the page is no longer checked against the working tree. Instead the tag must mark
         that commit, the committed page and the dispositions it was rendered from must be byte-identical to the tag's,
         and the tag's own status tests (the page is what the tool renders, every criterion has a row, V1-V3's
         refusals) must pass when run in a copy of the tagged tree. A missing tag fails; it is never skipped.
Spec:    docs/version_1_definition.md#1 (V1, V2, V3); docs/version_2_plan.md#stage-0 (0.1)
Tests:   this file
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAG = "version-1"
TAG_COMMIT = "d62cfdc0d4918a5da38d4321e1fd7be1485fafc7"
FROZEN = ("docs/version_1_status.md", "docs/version_1_dispositions.md")
TAGGED_TESTS = "tests/test_version_1_status.py"
# What the tag's status tests read: the tool and the record tools it loads, the records, dispositions and spec, the
# code's criterion registries and the config they load, the test files the tool checks by name, and pytest.ini.
ARCHIVED = ("pytest.ini", "src", "config", "tools", "tests", "docs")


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, timeout=120)
    assert result.returncode == 0, (
        f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}\n"
        f"The tag `{TAG}` must be present locally (git fetch --tags); CI checks out with fetch-depth 0.")
    return result.stdout


def test_v1_the_tag_marks_the_commit_the_owner_chose():
    assert _git("rev-parse", f"{TAG}^{{commit}}").decode().strip() == TAG_COMMIT


def test_v1_the_committed_page_and_dispositions_are_the_tags():
    for path in FROZEN:
        assert (REPO / path).read_bytes() == _git("show", f"{TAG}:{path}"), (
            f"{path} differs from the tag {TAG}: version 1's record is frozen; "
            f"restore it with git checkout {TAG} -- {path}")


def test_v1_the_tags_status_tests_pass_in_a_copy_of_the_tagged_tree(tmp_path):
    with tarfile.open(fileobj=io.BytesIO(_git("archive", "--format=tar", TAG, *ARCHIVED))) as tar:
        tar.extractall(tmp_path, filter="data")
    expected = set(re.findall(r"^def (test_\w+)\(", (tmp_path / TAGGED_TESTS).read_text(encoding="utf-8"), re.M))
    assert expected, f"{TAGGED_TESTS} at {TAG} defines no tests"
    # The parent's pytest options (PYTEST_ADDOPTS, PYTEST_PLUGINS) would change what runs at the tag.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("PYTEST_")}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", TAGGED_TESTS, "-q", "-rA", "-p", "no:cacheprovider"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    tail = "\n".join(result.stdout.splitlines()[-30:])
    assert result.returncode == 0, f"the status tests fail at {TAG}:\n{tail}\n{result.stderr[-2000:]}"
    passed = set(re.findall(rf"^PASSED {re.escape(TAGGED_TESTS)}::(\w+)", result.stdout, re.M))
    assert passed == expected, f"at {TAG}, not passed: {sorted(expected - passed)}\n{tail}"
