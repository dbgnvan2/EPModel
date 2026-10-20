"""The Phase D ensemble record is current and complete (Phase D plan step 12).

Purpose: fail when the committed record is not what its cache renders for the current code and config, or when it
         misses a built criterion or a not-built one. Since version 2 stage 1 (S1.8c) the record renders from
         docs/records_cache/, so the tests here run no ensemble.
Spec:    docs/bowen_agent_model_spec_v2.md#M13.4, #M17.A.1; docs/implementation_plan_version_2.md#S1.7b, #S1.8c
Tests:   this file
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("tools.phase_d_record", REPO / "tools" / "phase_d_record.py")
_tool = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _tool
_spec.loader.exec_module(_tool)


def record_text() -> str:
    return (REPO / "docs" / "phase_d_ensemble_record.md").read_text(encoding="utf-8")


def _rendered(part: int) -> str:
    from tools.record_cache import Cache

    try:
        return _tool.build(part, Cache(_tool.CACHES[part]))
    except KeyError as missing:
        flag = "" if part == 1 else " --part 2"
        raise AssertionError(f"stale: rerun python3 tools/phase_d_record.py{flag} ({missing})") from None


def test_phase_d_record_is_current():
    """The record is exactly what its cache renders for the current code (S1.8c): a changed engine has no cached
    result (stale), and a hand edit does not match."""
    assert _rendered(1) == record_text(), "stale or edited: rerun python3 tools/phase_d_record.py"


def test_phase_d_record_covers_every_criterion_and_names_what_is_not_built():
    from src.bowen.ensemble.criteria_d import CRITERIA_D

    text = record_text()
    for cid in CRITERIA_D:
        assert f"| `{cid}` |" in text, cid
    for cid in _tool.NOT_BUILT:
        assert f"- `{cid}` —" in text, cid
        assert cid not in CRITERIA_D


def test_phase_d_record_carries_no_phase_c_wording_and_its_table_matches_its_json():
    """Review 2026-10-13: the page is the Phase C renderer's, reworded; none of it may be left, and the verdict table and
    the machine-readable block must agree."""
    from tools.record_json import record_json

    text = record_text()
    assert "Phase C" not in text
    # Variants carry a bracketed name (M11.C.14[firmness], revision 31).
    table = dict(re.findall(r"^\| `(M11\.C\.\d+(?:\[[^\]]+\])?)` \| \w+ \| \*\*(\w+)\*\* \|", text, re.M))
    block = record_json(text)
    assert table == {v["criterion"]: v["outcome"] for v in block} and table


# --- Part II, the life course (plan §4.4; spec revision 28) ---


def record_text_2() -> str:
    return (REPO / "docs" / "phase_d2_ensemble_record.md").read_text(encoding="utf-8")


def test_phase_d2_record_is_current():
    assert _rendered(2) == record_text_2(), "stale or edited: rerun python3 tools/phase_d_record.py --part 2"


def test_phase_d2_record_covers_every_criterion_and_names_what_is_not_run():
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2

    text = record_text_2()
    for key in CRITERIA_D2:  # each under its own key (S1.7b)
        assert f"| `{key}` |" in text, key
    for cid in _tool.NOT_RUN_2:
        assert f"- `{cid}` —" in text, cid
    assert "Phase C" not in text


# --- Version 2 stage 1: the Phase D records render from a cache (S1.8c), each row under its key (S1.7b) ---


def _stub_verdict(criterion, rules, workers=1):
    from src.bowen.ensemble.runner import Verdict

    return Verdict(criterion.id, criterion.cls, "PASS", 3, [], 0.0, {}, [], [], {})


def _in_tmp_cache(tmp_path):
    import tools.record_cache as record_cache

    return mock.patch.object(record_cache, "CACHE_DIR", tmp_path / "cache")


def test_s1_8c_records_render_from_the_cache(tmp_path):
    """A build with every result cached renders the record without running a criterion, and prints the code's hash."""
    from tools.record_cache import Cache

    with _in_tmp_cache(tmp_path), mock.patch("src.bowen.ensemble.runner.run_criterion", side_effect=_stub_verdict):
        first = _tool.build(2, Cache("d2"), workers=1)
    with _in_tmp_cache(tmp_path), mock.patch("src.bowen.ensemble.runner.run_criterion",
                                             side_effect=AssertionError("ran a criterion with its result cached")):
        assert _tool.build(2, Cache("d2")) == first
    assert f"code_hash: {_tool.result_hash()}" in first


def test_s1_8c_a_stale_entry_is_rerun(tmp_path):
    """P8 dirty state: a cache holding results for older code is not reused. Every criterion is run again on the
    current code, and saving the cache drops the old entries."""
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2
    from tools.record_cache import Cache

    with _in_tmp_cache(tmp_path), mock.patch("src.bowen.ensemble.runner.run_criterion", side_effect=_stub_verdict):
        with mock.patch.object(_tool, "result_hash", return_value="0" * 64):
            _tool.build(2, Cache("d2"), workers=1)  # results for older code
        cache = Cache("d2")
        old = set(cache.entries)
        with mock.patch("src.bowen.ensemble.runner.run_criterion", side_effect=_stub_verdict) as ran:
            _tool.build(2, cache, workers=1)
        assert ran.call_count == len(CRITERIA_D2)
        cache.save()
        assert old and not old & set(Cache("d2").entries)
    with pytest.raises(KeyError), _in_tmp_cache(tmp_path), mock.patch.object(_tool, "result_hash", return_value="1" * 64):
        _tool.build(2, Cache("d2"))  # without workers, a missing result is never run


def test_s1_7b_variants_print_under_their_own_keys(tmp_path):
    """`M11.C.10` and `M11.C.10[family of origin]` share a criterion id, and `M11.C.2`'s twelve-person variant has
    part I's `M11.C.2` as its id; each is its own row, under its ``CRITERIA_D2`` key, in the table and the
    machine-readable block."""
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2
    from tools.record_cache import Cache
    from tools.record_json import record_json

    with _in_tmp_cache(tmp_path), mock.patch("src.bowen.ensemble.runner.run_criterion", side_effect=_stub_verdict):
        text = _tool.build(2, Cache("d2"), workers=1)
    for key in ("M11.C.10", "M11.C.10[family of origin]", "M11.C.2[twelve-person family]"):
        assert key in CRITERIA_D2 and text.count(f"| `{key}` |") == 1, key
    assert "| `M11.C.2` |" not in text  # part I's criterion, not run in part II
    assert [r["criterion"] for r in record_json(text)] == list(CRITERIA_D2)
