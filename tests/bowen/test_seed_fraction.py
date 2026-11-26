"""T-299's run mix: the two slow Phase D records run at a fraction of their seeds (`tools/seed_fraction.py`).

Purpose: a short run scales the seed settings and nothing else, is cached apart from the full run, states its
         fraction in the record, and a rebuild from the cache reads that fraction back; at 1 nothing changes.
Spec:    docs/review/2026-11-26_short_rerun_test.md#recommended-mix-of-runs
Tests:   this file
"""

from __future__ import annotations

import shutil
from unittest import mock

import pytest

from tools import phase_d_record as _tool
from tools import seed_fraction as sf
from tools.mutant_runner import Mutant, apply

from test_phase_d_record import _in_tmp_cache, _stub_verdict  # tests/bowen/ is on the path (pytest imports by basename)

REPO = sf.REPO


def test_t299_a_fraction_scales_the_seed_settings_and_one_is_the_full_run(tmp_path):
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2

    rules = {"ensemble_block": 50, "ensemble_cap": 500, "ensemble_precision": 0.25, "other": 3}
    assert sf.scaled_rules(rules, 1) is rules and sf.definition(None, 1) is None
    assert sf.scaled_rules(rules, 0.2) == {"ensemble_block": 10, "ensemble_cap": 100, "ensemble_precision": 0.559,
                                           "other": 3}
    assert sf.scaled_criterion(CRITERIA_D2["M11.C.10"], 0.2).seeds == CRITERIA_D2["M11.C.10"].seeds // 5
    mutant = Mutant("m", "k", ("M11.C.6",), "config/bowen/criteria.md", "| M11.C.48 | seeds | ",
                    "| M11.C.48 | seeds | ", "no-op")
    assert sf.scaled_mutant(mutant, 1) is mutant
    short = sf.scaled_mutant(mutant, 0.2)
    assert short.definition() != mutant.definition()
    for part in ("config",):
        shutil.copytree(REPO / part, tmp_path / part)
    apply(tmp_path, short)  # each scaled line occurs exactly once in the repository's config
    assert "| `ensemble_cap` | 100 |" in (tmp_path / "config/bowen/constants.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        sf.check(0)


def test_t299_a_short_record_states_its_fraction_and_rebuilds_from_it(tmp_path, monkeypatch):
    """Mutation: drop the fraction from the cache key, the note or ``declared``, and a check here fails."""
    from tools.record_cache import Cache

    record = tmp_path / "phase_d2_ensemble_record.md"
    monkeypatch.setattr(_tool, "RECORDS", {**_tool.RECORDS, 2: record})
    seen = []

    def stub(criterion, rules, workers=1):
        seen.append(rules["ensemble_block"])
        return _stub_verdict(criterion, rules, workers)

    def child(part, cid, workers, fixed=None, fraction=1.0):  # in process, not a real child
        return _tool.verdict_row(part, cid, workers, fixed, fraction)

    with _in_tmp_cache(tmp_path), mock.patch("src.bowen.ensemble.runner.run_criterion", side_effect=stub), \
            mock.patch.object(_tool, "child_verdict", side_effect=child):
        text = _tool.build(2, Cache("d2"), workers=1, fraction=0.2)
        assert set(seen) == {10} and "`--seed-fraction 0.2`" in text
        with pytest.raises(KeyError):
            _tool.build(2, Cache("d2"), fraction=1.0)  # the full run's keys hold nothing
        record.write_text(text, encoding="utf-8")
        assert sf.declared(record) == 0.2
        assert _tool.build(2, Cache("d2")) == text  # the rebuild reads the record's fraction
