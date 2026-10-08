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
    assert match.group(1) == _tool.code_hash(*_tool.HASHED_TOOLS), "stale: rerun python3 tools/ensemble_record.py"


def test_m134_record_covers_every_criterion_and_names_what_is_not_built():
    text = record_text()
    for cid in CRITERIA:
        assert f"| `{cid}` |" in text, cid
    for cid in NOT_BUILT:
        assert f"- `{cid}` —" in text, cid


def test_m11d18_record_reports_the_fallback_rate():
    assert "## Fallback rate by person (`M11.D.18`)" in record_text()


_sspec = importlib.util.spec_from_file_location("sweep_record_tool", REPO / "tools" / "sweep_record.py")
_sweep = importlib.util.module_from_spec(_sspec)
sys.modules[_sspec.name] = _sweep

# Loaded under its real module name, so the sweep tool's `from tools.mutation_record import ...` reuses this
# copy (one load), and the module's __name__ matches the name it is registered under.
_mspec = importlib.util.spec_from_file_location("tools.mutation_record", REPO / "tools" / "mutation_record.py")
_mutants = importlib.util.module_from_spec(_mspec)
sys.modules[_mspec.name] = _mutants  # its dataclass resolves its own module
_mspec.loader.exec_module(_mutants)
_sspec.loader.exec_module(_sweep)  # after the mutation tool: it imports from it


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
    assert match.group(1) == _tool.code_hash(*_mutants.HASHED_TOOLS), "stale: rerun python3 tools/mutation_record.py"


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
    assert match and match.group(1) == _tool.code_hash(*_sweep.HASHED_TOOLS), "stale: rerun python3 tools/sweep_record.py"
    rows = _mutants.json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
    swept = {(r["criterion"], r["setting"]) for r in rows}
    for cid, criterion in CRITERIA.items():
        if criterion.cls == "composite":
            for name in ("learning_rate", "credit_horizon", "policy_temperature"):
                assert sum(1 for c, s in swept if c == cid and s.startswith(name)) == 2, (cid, name)


def test_m111d_mutation_hash_covers_the_mutant_list():
    """Changing a mutant must make the mutation record stale (gate finding 2026-10-08, P6)."""
    tool = REPO / "tools" / "mutation_record.py"
    assert _tool.code_hash(tool) != _tool.code_hash()
    assert tool.resolve() in _mutants.HASHED_TOOLS


def test_m134_every_record_hashes_the_ensemble_tool():
    """RULE_KEYS in tools/ensemble_record.py feeds every record, so every record's hash covers it (re-gate A)."""
    ensemble_tool = (REPO / "tools" / "ensemble_record.py").resolve()
    for tools in (_tool.HASHED_TOOLS, _mutants.HASHED_TOOLS, _sweep.HASHED_TOOLS):
        assert ensemble_tool in {Path(t).resolve() for t in tools}
    assert _tool.code_hash(ensemble_tool) != _tool.code_hash()


def test_criteria_settings_are_parsed_strictly(tmp_path):
    """config/bowen/criteria.md: a setting the criteria do not read, or one missing, raises (M11.D.3's rule)."""
    import pytest

    from src.bowen.ensemble.criteria import load_settings
    from src.bowen.scenario.config_parse import ConfigError

    text = (REPO / "config" / "bowen" / "criteria.md").read_text(encoding="utf-8")
    extra = tmp_path / "extra.md"
    extra.write_text(text + "| M11.C.1 | horizon | 5 |\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no setting 'horizon'"):
        load_settings(extra)
    missing = tmp_path / "missing.md"
    missing.write_text(text.replace("| M11.C.42 | weeks | 40 |\n", ""), encoding="utf-8")
    with pytest.raises(ConfigError, match="M11.C.42 weeks"):
        load_settings(missing)


def test_sweep_tool_shares_the_loaded_mutation_tool():
    """One copy of the mutation tool, not two (third gate finding 3), registered under its own name (fourth gate)."""
    assert _sweep.Mutant is _mutants.Mutant and _sweep.run_mutant is _mutants.run_mutant
    assert _mutants.__name__ == "tools.mutation_record" and sys.modules["tools.mutation_record"] is _mutants


def _setting_reads():
    """Each top-level function's literal ``settings["k"]`` reads, and every literal ``SETTINGS["s"]["k"]`` / ``SPELL["k"]``."""
    import ast

    tree = ast.parse((REPO / "src" / "bowen" / "ensemble" / "criteria.py").read_text(encoding="utf-8"))

    def key(node):
        return node.slice.value if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) else None

    by_function, module = {}, set()
    for top in tree.body:
        if isinstance(top, ast.FunctionDef):
            by_function[top.name] = {key(n) for n in ast.walk(top) if isinstance(n, ast.Subscript)
                                     and isinstance(n.value, ast.Name) and n.value.id == "settings" and key(n)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and key(n):
            if isinstance(n.value, ast.Name) and n.value.id == "SPELL":
                module.add(("spell", key(n)))
            elif isinstance(n.value, ast.Subscript) and isinstance(n.value.value, ast.Name) \
                    and n.value.value.id == "SETTINGS" and key(n.value):
                module.add((key(n.value), key(n)))
    return by_function, module


def test_criteria_required_matches_what_the_arms_read():
    """REQUIRED, the config's schema, and the arms' reads agree in both directions (third gate finding 1)."""
    from src.bowen.ensemble.criteria import INJECTED, REQUIRED

    by_function, module = _setting_reads()
    # Forward: every key an arm reads is in the settings that criterion is given (else KeyError at ensemble time).
    for cid, criterion in CRITERIA.items():
        missing = by_function[criterion.arm.__name__] - set(criterion.settings)
        assert not missing, f"{cid}: its arm reads {sorted(missing)}, which its settings do not hold"
    # Reverse: every required setting is read — by an arm of its section, or by name at module level.
    read = set(module)
    for cid, criterion in CRITERIA.items():
        read |= {(cid.split("[")[0], k) for k in by_function[criterion.arm.__name__] - INJECTED}
    level_rows = {("M11.C.38", k) for k in REQUIRED["M11.C.38"] if k.startswith("level_")}  # read by prefix
    unread = {(s, k) for s, names in REQUIRED.items() for k in names} - read - level_rows
    assert not unread, f"declared in REQUIRED but read nowhere: {sorted(unread)}"
    assert not (read - {(s, k) for s, names in REQUIRED.items() for k in names}), "read but not declared"
