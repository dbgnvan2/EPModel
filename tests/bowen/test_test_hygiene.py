"""M11.D.7 — tests never write to real artifact paths.

Purpose: prove the conftest guard stops a write to the repository and allows one
         under the test's temporary directory.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.7
Tests:   this file
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.bowen.io.sinks import JsonlFileSink

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / ".m11d7_probe"


def test_m11d7_no_production_paths_in_tests(tmp_path):
    """G8: a write into the repository raises; the same write under tmp_path succeeds."""
    try:
        with pytest.raises(AssertionError, match="outside"):
            PROBE.write_text("must not exist")
        with pytest.raises(AssertionError, match="outside"):
            JsonlFileSink(REPO / "runs" / "probe.jsonl")
    finally:
        created = PROBE.exists()
        PROBE.unlink(missing_ok=True)
    assert not created, "the guard let a test write into the repository"
    (tmp_path / "fine.txt").write_text("ok")
    assert (tmp_path / "fine.txt").read_text() == "ok"


def test_m11d7_reading_committed_config_is_allowed():
    assert (REPO / "config" / "bowen" / "constants.md").read_text(encoding="utf-8")
