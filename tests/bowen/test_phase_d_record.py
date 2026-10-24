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


def test_m17a1_a_known_defect_is_printed_beside_its_verdict():
    """Milestone review MR-3: `M11.C.48`'s PASS was printed unqualified while its readout's baseline is known wrong
    (T-123) and its named mutant survived (T-119). Each key of ``VERDICT_NOTES`` that this record holds is printed
    with its note in the verdict cell, once; the machine-readable block keeps the bare outcome; no other row carries a
    note. Mutation: the renderer does not read ``VERDICT_NOTES``."""
    from tools.record_json import record_json

    notes = _tool.VERDICT_NOTES
    assert set(notes) == {"M11.C.48", "M11.C.48[phase_d]"}
    text = _rendered(2)
    outcomes = {r["criterion"]: r["outcome"] for r in record_json(text)}
    for key, note in notes.items():
        row = next(line for line in text.splitlines() if line.startswith(f"| `{key}` |"))
        assert f"| **{outcomes[key]}** ({note}) |" in row, row
        assert text.count(note) == 2  # the two notes are the same text, one per row
    assert all(outcomes[k] == "PASS" for k in notes)
    assert text.count("(T-123)") == len(notes)


def test_m17a1_each_verdict_note_is_true_of_the_records_and_the_todo_list():
    """Learning-qa, 2026-10-24: a ``VERDICT_NOTES`` note is a claim (P6) and nothing checked it. Each key must be a
    criterion an ensemble record holds (a typo or a key left behind is refused, not skipped); each T-ID it cites must
    be open in `TODO.md`; "named mutant survived" must be what the mutation records say. When T-119 or T-123 closes,
    or a mutant turns the criterion red, this fails until the note is changed. Mutation: a note citing a closed item."""
    from tools.version_2_status import survived_mutants
    from tools.version_2_sweep import central_verdicts

    centrals, survivors = central_verdicts(), survived_mutants()
    open_items = set(re.findall(r"^- (T-\d+) ", (REPO / "TODO.md").read_text(encoding="utf-8"), re.M))
    for key, note in _tool.VERDICT_NOTES.items():
        assert key in centrals, f"VERDICT_NOTES names {key}, which no ensemble record holds"
        cited = set(re.findall(r"T-\d+", note))
        assert cited and cited <= open_items, f"{key}'s note cites {sorted(cited - open_items)}, not open in TODO.md"
        if "named mutant survived" in note:
            assert any(kind == "named" for kind, _ in survivors.get(key, [])), (
                f"{key}'s note says its named mutant survived; no mutation record says so")
