"""T-299's short rerun test: its report is the render of its JSON, and its scaler never edits this repository.

Spec:    docs/review/2026-11-26_short_rerun_test.md
Tests:   this file
"""

from __future__ import annotations

import json

import pytest

from tools import short_rerun as sr

JSON = sr.REPO / "docs" / "review" / "2026-11-26_short_rerun_test.json"


def test_t299_the_report_is_its_json_rendered():
    data = json.loads(JSON.read_text(encoding="utf-8"))
    assert sr.render(data) == JSON.with_suffix(".md").read_text(encoding="utf-8"), \
        "stale or edited: rerun python3 tools/short_rerun.py render " + data["json_name"]


def test_t299_the_scaler_refuses_this_repository():
    with pytest.raises(SystemExit):
        sr.scale(sr.REPO, 0.2)
