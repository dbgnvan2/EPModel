"""The Phase D mutation record is current, and its mutants apply (`M11.1a`, `M11.1d`).

Purpose: fail when the committed record is not what its cache renders for the current engine, config, child and
         mutant edits, or when a mutant's replacement does not occur exactly once.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.1a, #M11.1d, #M11.C.10
Tests:   this file
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("tools.phase_d_mutation_record",
                                               REPO / "tools" / "phase_d_mutation_record.py")
_tool = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _tool
_spec.loader.exec_module(_tool)
from tools.record_cache import Cache  # noqa: E402
from tools.record_json import record_json  # noqa: E402


def test_m111a_every_phase_d_mutant_applies_exactly_once(tmp_path):
    for mutant in _tool.MUTANTS:
        root = tmp_path / mutant.id
        for file in {f for f, _, _ in ((mutant.file, mutant.old, mutant.new), *mutant.also)}:
            (root / file).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / file, root / file)
        _tool.apply(root, mutant)  # raises SystemExit unless every replacement occurs exactly once
        for file, _, new in ((mutant.file, mutant.old, mutant.new), *mutant.also):
            assert new in (root / file).read_text(encoding="utf-8"), mutant.id


def test_m111d_phase_d_mutation_record_is_current():
    try:
        rendered = _tool.build(Cache("phase_d_mutation"))
    except KeyError as missing:
        raise AssertionError(f"stale: rerun python3 tools/phase_d_mutation_record.py ({missing})") from None
    assert rendered == _tool.RECORD.read_text(encoding="utf-8"), \
        "stale or edited: rerun python3 tools/phase_d_mutation_record.py"


def test_m11c10_family_of_origin_variant_is_proved_red_by_removing_m4d6f():
    """The spec's mutation clause for `M11.C.10[family of origin]` (restated 2026-10-15): `M4.D.6f` removed MUST turn
    it red. Read from the committed record."""
    rows = record_json(_tool.RECORD)
    row = next((r for r in rows if r["mutant"] == "witnessed-cutoff-learning-removed"
                and r["criterion"] == "M11.C.10[family of origin]"), None)
    assert row is not None, "the variant was not run under M4.D.6f removed: does it still pass in the part II record?"
    assert row["result"] in ("red", "reversed"), row


def test_m11c14_gate_removed_is_run_on_both_variants_and_its_result_matches_the_spec():
    """The spec's mutation clause for `M11.C.14` (revision 31, R31.B5): the marital-distance gate removed MUST turn at
    least one technique variant red. Measured 2026-10-16, it survives both (the gate blocks about a point of relief to
    the child per run in this fixture); the spec marks the clause not met, for the owner. This test pins that the
    mutant was run on both passing variants, and that the spec's account agrees with the record: if a later change makes
    the gate bite, the spec line must change with it."""
    rows = record_json(_tool.RECORD)
    run = {r["criterion"]: r["result"] for r in rows if r["mutant"] == "marital-distance-gate-removed"}
    assert set(run) == {"M11.C.14[firmness]", "M11.C.14[talking it out]"}, run
    spec = (REPO / "docs" / "bowen_agent_model_spec_v2.md").read_text(encoding="utf-8")
    bites = any(result in ("red", "reversed") for result in run.values())
    assert ("gate-removed mutant survives both variants" in spec) is (not bites), run


def test_m111d_every_passing_phase_d2_criterion_has_a_mutant_or_names_where_it_is_reported():
    """Phase C's `test_m111d_every_passing_criterion_has_a_mutant_run`, for part II: a newly passing criterion with no
    mutant here and no entry in `AD_HOC` fails, so the record cannot read as full coverage."""
    rows = record_json(_tool.RECORD)
    recorded = {r["criterion"] for r in rows}
    missing = sorted(_tool.passing() - recorded - set(_tool.AD_HOC))
    assert not missing, f"passing with no mutant and not in AD_HOC: {missing}"
    assert not set(_tool.AD_HOC) & recorded, "a criterion in AD_HOC now has a mutant here; remove it from AD_HOC"


def test_s1_9_engine_hash_ignores_comments(monkeypatch):
    """Rider (iv) (version 2 plan D7): the record's own hash reads ``apply`` and ``run_in_copy`` through
    ``source_text``, so a comment added to either leaves it unchanged, and a code change moves it."""
    real = _tool.inspect.getsource
    before = _tool.engine_hash()
    for extra, moves in (("    # a comment added to the function\n", False), ("S1_9_EDIT = 1\n", True)):
        monkeypatch.setattr(_tool.inspect, "getsource", lambda f, extra=extra: real(f) + extra)
        assert (_tool.engine_hash() != before) is moves, extra
