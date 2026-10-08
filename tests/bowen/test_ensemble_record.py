"""The committed ensemble record is current (plan D7).

Purpose: fail the default suite when docs/phase_c_ensemble_record.md was produced against code
         or config that has since changed, or omits a criterion — so a stale record cannot stand
         in for a run.
Spec:    docs/bowen_agent_model_spec_v2.md#M13.4, #M17.A.1, #M11.D.18
Tests:   this file
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

from src.bowen.ensemble.criteria import CRITERIA, NOT_BUILT

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ensemble_record_tool", REPO / "tools" / "ensemble_record.py")
_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tool)


def record_text() -> str:
    return (REPO / "docs" / "phase_c_ensemble_record.md").read_text(encoding="utf-8")


def test_m134_ensemble_record_is_current():
    match = re.search(r"^code_hash: ([0-9a-f]{64})$", record_text(), re.M)
    assert match, "the record carries no code hash"
    assert match.group(1) == _tool.code_hash(), "stale: rerun python3 tools/ensemble_record.py"


def test_m134_record_covers_every_criterion_and_names_what_is_not_built():
    text = record_text()
    for cid in CRITERIA:
        assert f"| `{cid}` |" in text, cid
    for cid in NOT_BUILT:
        assert f"- `{cid}` —" in text, cid


def test_m11d18_record_reports_the_fallback_rate():
    assert "## Fallback rate by person (`M11.D.18`)" in record_text()


_mspec = importlib.util.spec_from_file_location("mutation_record_tool", REPO / "tools" / "mutation_record.py")
_mutants = importlib.util.module_from_spec(_mspec)
sys.modules[_mspec.name] = _mutants  # its dataclass resolves its own module
_mspec.loader.exec_module(_mutants)


def mutation_text() -> str:
    return (REPO / "docs" / "phase_c_mutation_record.md").read_text(encoding="utf-8")


def test_m111a_every_mutant_applies_exactly_once():
    for mutant in _mutants.MUTANTS:
        for file, old, _ in ((mutant.file, mutant.old, mutant.new), *mutant.also):
            text = (REPO / file).read_text(encoding="utf-8")
            assert text.count(old) == 1, f"{mutant.id}: a mutant that does not apply proves nothing (M11.1a)"


def test_m111d_mutation_record_is_current():
    match = re.search(r"^code_hash: ([0-9a-f]{64})$", mutation_text(), re.M)
    assert match, "the mutation record carries no code hash"
    assert match.group(1) == _tool.code_hash(), "stale: rerun python3 tools/mutation_record.py"


def test_m111d_every_passing_criterion_has_a_mutant_run():
    """Every criterion passing in the ensemble record was run under at least one non-representation mutant."""
    rows = _mutants.json.loads(re.search(r"```json\n(.*?)\n```", mutation_text(), re.S).group(1))
    proved = {r["criterion"] for r in rows if r["kind"] != _mutants.REPRESENTATION}
    missing = sorted(_mutants.passing() - proved)
    assert not missing, f"passing with no mutant run: {missing}"


def test_m111c_representation_mutants_ran_on_every_passing_criterion():
    rows = _mutants.json.loads(re.search(r"```json\n(.*?)\n```", mutation_text(), re.S).group(1))
    for mutant in (m for m in _mutants.MUTANTS if m.kind == _mutants.REPRESENTATION):
        ran = {r["criterion"] for r in rows if r["mutant"] == mutant.id}
        assert _mutants.passing() <= ran, f"{mutant.id} did not run on {sorted(_mutants.passing() - ran)}"


def test_d9_sweep_record_is_current():
    """Plan D9: every composite criterion was swept at low and high α, H and temperature on the current code."""
    text = (REPO / "docs" / "phase_c_sweep_record.md").read_text(encoding="utf-8")
    match = re.search(r"^code_hash: ([0-9a-f]{64})$", text, re.M)
    assert match and match.group(1) == _tool.code_hash(), "stale: rerun python3 tools/sweep_record.py"
    rows = _mutants.json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
    swept = {(r["criterion"], r["setting"]) for r in rows}
    for cid, criterion in CRITERIA.items():
        if criterion.cls == "composite":
            for name in ("learning_rate", "credit_horizon", "policy_temperature"):
                assert sum(1 for c, s in swept if c == cid and s.startswith(name)) == 2, (cid, name)
