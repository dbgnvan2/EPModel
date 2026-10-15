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
    rows = _tool.json.loads(_tool.re.search(r"```json\n(.*?)\n```", _tool.RECORD.read_text(encoding="utf-8"),
                                            _tool.re.S).group(1))
    row = next(r for r in rows if r["mutant"] == "witnessed-cutoff-learning-removed"
               and r["criterion"] == "M11.C.10[family of origin]")
    assert row["result"] in ("red", "reversed"), row
