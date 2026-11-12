"""Version 1's status page is frozen at the tag `version-1`, and the checks that made it final still pass there.

Purpose: version 1 ends at the tag `version-1` (d62cfdc, owner 2026-10-19). Version 2 regenerates the records the
         page is rendered from, so the page is no longer checked against the working tree. Instead the tag must mark
         that commit, and the committed page and the dispositions it was rendered from must be byte-identical to the
         tag's. A missing tag fails; it is never skipped. (The re-run of the tag's own status tests in a copy of the
         tagged tree was removed 2026-11-11, T-222: it re-checked a tree that cannot change.)
Spec:    docs/version_1_definition.md#1 (V1, V2, V3); docs/version_2_plan.md#stage-0 (0.1)
Tests:   this file
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAG = "version-1"
TAG_COMMIT = "d62cfdc0d4918a5da38d4321e1fd7be1485fafc7"
FROZEN = ("docs/version_1_status.md", "docs/version_1_dispositions.md")


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
