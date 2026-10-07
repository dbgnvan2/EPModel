"""M11.D.7 — tests never write to real artifact paths.

Purpose: prove the conftest guard stops writes, moves and deletes in the
         repository while allowing them under the test's temporary directory,
         and that the suite-wide fingerprint detects a changed artifact.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.7
Tests:   this file
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from src.bowen.io.sinks import JsonlFileSink

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / ".m11d7_probe"
# Captured at import, before the guard patches os: used only to clean up a probe
# the guard failed to stop, so a broken guard cannot leave a file behind.
_REAL_UNLINK = os.unlink

_root_spec = importlib.util.spec_from_file_location("root_conftest", REPO / "tests" / "conftest.py")
ROOT_CONFTEST = importlib.util.module_from_spec(_root_spec)
_root_spec.loader.exec_module(ROOT_CONFTEST)


def test_m11d7_no_production_paths_in_tests(tmp_path):
    """G8: writes, moves and deletes in the repository raise; the same under tmp_path succeed."""
    try:
        with pytest.raises(AssertionError, match="outside"):
            PROBE.write_text("must not exist")
        with pytest.raises(AssertionError, match="outside"):
            JsonlFileSink(REPO / "runs" / "probe.jsonl")
        with pytest.raises(AssertionError, match="outside"):
            os.replace(str(REPO / "README.md"), str(REPO / "README.moved"))
        with pytest.raises(AssertionError, match="outside"):
            os.remove(str(REPO / "README.md"))
    finally:
        created = PROBE.exists()
        if created:
            _REAL_UNLINK(PROBE)
    assert not created, "the guard let a test write into the repository"
    assert (REPO / "README.md").exists()
    (tmp_path / "fine.txt").write_text("ok")
    os.replace(tmp_path / "fine.txt", tmp_path / "moved.txt")
    os.remove(tmp_path / "moved.txt")


def test_m11d7_reading_committed_config_is_allowed():
    assert (REPO / "config" / "bowen" / "constants.md").read_text(encoding="utf-8")


def test_m11d7_fingerprint_detects_a_changed_artifact(tmp_path):
    for entry in ROOT_CONFTEST.WATCHED:
        target = tmp_path / entry
        if entry.endswith(".md"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("spec")
        else:
            target.mkdir(parents=True, exist_ok=True)
            (target / "a.txt").write_text("one")
    before = ROOT_CONFTEST.fingerprint(tmp_path)
    (tmp_path / "config" / "bowen" / "a.txt").write_text("two")
    after = ROOT_CONFTEST.fingerprint(tmp_path)
    assert [k for k in before if before[k] != after[k]] == [str(Path("config/bowen/a.txt"))]
