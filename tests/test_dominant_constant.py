"""The dominant-constant proposals page is current.

Purpose: ``docs/dominant_constant_proposals.md`` is what ``tools/dominant_constant.py`` renders from the committed
         sweep record and the T-236 grid's cache, without running anything (P6).
Spec:    docs/bowen_agent_model_spec_v2.md#M17.E.2
Tests:   this file
"""

from unittest import mock

import tools.dominant_constant as tool
import tools.record_cache as record_cache


def test_m17e2_t077_dominant_constant_page_is_current():
    with mock.patch.object(record_cache, "child_rows", side_effect=AssertionError("ran a child in a test")):
        text = tool.render()
    assert tool.PAGE.read_text(encoding="utf-8") == text, "stale or edited: rerun python3 tools/dominant_constant.py"
