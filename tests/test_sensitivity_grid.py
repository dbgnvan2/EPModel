"""The T-236 sensitivity grid's record is current.

Purpose: ``docs/sensitivity_distance_triangle.md`` is what ``tools/sensitivity_grid.py`` renders from the committed
         cache for the current engine, without running anything (P6): a stale cache, a changed grid or an engine
         change raises here.
Spec:    docs/bowen_agent_model_spec_v2.md#M17.E.1
Tests:   this file
"""

from unittest import mock

import pytest

import tools.record_cache as record_cache
import tools.sensitivity_grid as tool
from tools.record_cache import Cache


@pytest.mark.xfail(strict=True, reason="T-266: spec revision 67 (2026-11-21, M5.D.3b and M5.D.9b) moved the engine hash; this record is regenerated on Mac Mini1 and is stale on the new engine until then")
def test_t236_sensitivity_record_is_current():
    with mock.patch.object(record_cache, "child_rows", side_effect=AssertionError("ran a child in a test")):
        text = tool.build(Cache(tool.CACHE_NAME))
    assert tool.RECORD.read_text(encoding="utf-8") == text + "\n"
