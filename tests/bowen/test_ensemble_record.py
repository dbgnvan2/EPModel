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
from src.bowen.io.load import load_constants

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


_sspec = importlib.util.spec_from_file_location("tools.sweep_record", REPO / "tools" / "sweep_record.py")
_sweep = importlib.util.module_from_spec(_sspec)
sys.modules[_sspec.name] = _sweep

# Loaded under its real module name, so the sweep tool's `from tools.mutation_record import ...` reuses this
# copy (one load), and the module's __name__ matches the name it is registered under.
_mspec = importlib.util.spec_from_file_location("tools.mutation_record", REPO / "tools" / "mutation_record.py")
_mutants = importlib.util.module_from_spec(_mspec)
sys.modules[_mspec.name] = _mutants  # its dataclass resolves its own module
_mspec.loader.exec_module(_mutants)
from tools.record_cache import Cache, engine_hash, result_key  # noqa: E402  (the cache both tools share)
import tools.mutant_runner as _runner  # noqa: E402
import tools.record_cache as _cache  # noqa: E402
_sspec.loader.exec_module(_sweep)  # after the mutation tool: it imports from it


def mutation_text() -> str:
    return (REPO / "docs" / "phase_c_mutation_record.md").read_text(encoding="utf-8")


def test_m111a_every_mutant_applies_exactly_once(tmp_path):
    """Through apply() itself, on a copy: joint mutants replace in sequence, so each later `old` is counted in the
    text the earlier replacements left (correctness review)."""
    import shutil

    for mutant in _mutants.MUTANTS:
        root = tmp_path / mutant.id
        for file in {f for f, _, _ in ((mutant.file, mutant.old, mutant.new), *mutant.also)}:
            (root / file).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / file, root / file)
        _mutants.apply(root, mutant)  # raises SystemExit unless every replacement occurs exactly once
        for file, _, new in ((mutant.file, mutant.old, mutant.new), *mutant.also):
            assert new in (root / file).read_text(encoding="utf-8"), mutant.id


def test_m111d_mutation_record_is_current():
    """The record is exactly what its cache renders for the current engine and every mutant's current edits: a changed
    engine or mutant has no cached result (KeyError: rerun the tool), and a hand edit does not match."""
    try:
        rendered = _mutants.build(Cache("mutation"))
    except KeyError as missing:
        raise AssertionError(f"stale: rerun python3 tools/mutation_record.py ({missing})") from None
    assert rendered == mutation_text(), "stale or edited: rerun python3 tools/mutation_record.py"


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
    try:
        rendered = _sweep.build(Cache("sweep"))
    except KeyError as missing:
        raise AssertionError(f"stale: rerun python3 tools/sweep_record.py ({missing})") from None
    assert rendered == text, "stale or edited: rerun python3 tools/sweep_record.py"
    rows = _mutants.json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
    swept = {(r["criterion"], r["setting"]) for r in rows}
    for cid, criterion in CRITERIA.items():
        if criterion.cls == "composite":
            for name in ("learning_rate", "credit_horizon", "policy_temperature"):
                assert sum(1 for c, s in swept if c == cid and s.startswith(name)) == 2, (cid, name)


def test_m111d_a_result_is_keyed_on_the_engine_and_the_mutants_edits():
    """A changed edit or engine makes a cached result unusable; a changed label does not (2026-10-09: relabelling a
    mutant no longer reruns anything)."""
    import dataclasses

    mutant = next(m for m in _mutants.MUTANTS if m.id == "level-blind")
    engine = engine_hash()
    key = result_key(engine, mutant.definition(), "M11.C.16")
    relabelled = dataclasses.replace(mutant, what="another description", criteria=("M11.C.1",))
    assert result_key(engine, relabelled.definition(), "M11.C.16") == key
    edited = dataclasses.replace(mutant, new=mutant.new + " ")
    assert result_key(engine, edited.definition(), "M11.C.16") != key
    assert result_key("0" * 64, mutant.definition(), "M11.C.16") != key
    assert result_key(engine, mutant.definition(), "M11.C.1") != key


def test_m134_every_record_hashes_the_ensemble_tool():
    """RULE_KEYS in tools/ensemble_record.py feeds every record, and the runner's child produces the cached results,
    so the engine hash covers both, as does the occupancy record's (re-gate A; 2026-10-09)."""
    ensemble_tool = (REPO / "tools" / "ensemble_record.py").resolve()
    runner = (REPO / "tools" / "mutant_runner.py").resolve()
    for tools in (_tool.HASHED_TOOLS, _occupancy.HASHED_TOOLS):
        assert ensemble_tool in {Path(t).resolve() for t in tools}
    assert runner in {Path(t).resolve() for t in _occupancy.HASHED_TOOLS}
    assert engine_hash() == _tool.code_hash(ensemble_tool, runner)
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
    """One copy of the runner, not two (third gate finding 3); both tools registered under their own names (fourth and fifth gates)."""
    assert _sweep.Mutant is _mutants.Mutant is _runner.Mutant and _sweep.run_cached is _mutants.run_cached
    assert _mutants.__name__ == "tools.mutation_record" and sys.modules["tools.mutation_record"] is _mutants
    assert _sweep.__name__ == "tools.sweep_record" and sys.modules["tools.sweep_record"] is _sweep


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


def _availability(source: str, level: float, layer: int) -> float:
    """Evaluate an availability return line (the rule or a mutant of it) at a functional level and layer."""
    from types import SimpleNamespace

    from src.bowen.engine.objects import SCALE_MAX

    params = SimpleNamespace(capacity_level_per_layer=load_constants()["capacity_level_per_layer"])
    expression = source.strip().removeprefix("return ")
    return eval(expression, {"min": min, "max": max, "SCALE_MAX": SCALE_MAX},
                {"obs": SimpleNamespace(functional_level=level), "layer": layer, "params": params})


def _c16_levels() -> tuple[float, float]:
    """C.16's two arms' mean initial member functional level, read from the family file and the criterion's settings.

    Levels move during a run (measured 2026-10-08 over 20 seeds: the middle 90% of member-weeks is about 8-41 in the
    lowered arm and 25-54 in the baseline arm), so these are where the arms start, not the whole range they occupy.
    """
    from src.bowen.ensemble.criteria import FAMILIES, SETTINGS
    from src.bowen.io.load import load_family

    settings = SETTINGS["M11.C.16"]
    members = [p.functional_level for p in load_family(FAMILIES["phase_c"]).people.values()
               if p.role.value == "member"]
    mean = sum(members) / len(members)
    return max(0.0, mean - settings["level_treatment"]), max(0.0, mean - settings["level_baseline"])


def _layers() -> list[int]:
    from src.bowen.io.load import load_event_kinds

    return sorted({layer for layer in load_event_kinds().layers.values() if layer})


def _falls(source: str, low: float, high: float) -> list[int]:
    """The layers on which an availability line is lower at ``high`` than at ``low``."""
    return [layer for layer in _layers() if _availability(source, low, layer) > _availability(source, high, layer)]


def test_m111a_availability_inversion_inverts_at_c16s_levels():
    """An inversion must change behaviour where the criterion runs, not saturate into a deletion (csdp sweep finding).

    Between C.16's two arms' starting levels the rule never falls; its inverted mutant never rises, and falls on at
    least one layer the clamp does not hide.
    """
    inverted = next(m for m in _mutants.MUTANTS if m.id == "availability-level-inverted")
    low, high = _c16_levels()
    assert low < high
    assert _falls(_mutants.AVAILABILITY, low, high) == []
    assert all(_availability(inverted.new, low, layer) >= _availability(inverted.new, high, layer)
               for layer in _layers())
    assert _falls(inverted.new, low, high), "the inversion is flat at C.16's levels: it is a deletion, not an inversion"


def test_m111a_the_saturated_inversion_would_be_caught():
    """The first inversion, (100 - level), and the deletion both fail the check above at C.16's levels."""
    saturated = "return min(1.0, max(0.0, (100.0 - obs.functional_level) / (layer * params.capacity_level_per_layer)))"
    deleted = next(m for m in _mutants.MUTANTS if m.id == "availability-level-independent").new
    low, high = _c16_levels()
    assert _falls(saturated, low, high) == []
    assert _falls(deleted, low, high) == []


def test_m111d_a_mutant_is_reversed_only_when_every_interval_flips():
    """`reversed` is a flip of the result; a result that merely vanishes stays `red` (csdp re-sweep finding 1)."""
    named = next(m for m in _mutants.MUTANTS if m.kind == _mutants.SIGN)

    def readout(mean, half, direction=-1):
        return {"readout": "r", "direction": direction, "mean_difference": mean, "half_width": half}

    assert _mutants.judge(named, "FAIL", [readout(+0.05, 0.02)]) == "reversed"
    assert _mutants.judge(named, "FAIL", [readout(-0.008, 0.02)]) == "red"  # C.16 under the availability inversion
    assert _mutants.judge(named, "FAIL", [readout(+0.01, 0.02)]) == "red"  # opposite sign, interval spans zero
    assert _mutants.judge(named, "FAIL", [readout(+0.05, 0.02), readout(+0.01, 0.02, +1)]) == "red"
    assert _mutants.judge(named, "PASS", [readout(-0.05, 0.02)]) == "survived"
    assert _mutants.judge(named, "FAIL", [readout(+0.02, 0.02)]) == "red"  # the interval's edge touches zero


def test_m111d_reversed_ignores_readouts_that_do_not_gate():
    """A report-only readout (C.16's top-move share) or an equivalence readout cannot block or make `reversed`."""
    named = next(m for m in _mutants.MUTANTS if m.kind == _mutants.SIGN)
    entropy_flipped = {"readout": "e", "direction": -1, "mean_difference": +0.05, "half_width": 0.02}
    top_share_kept = {"readout": "t", "direction": 1, "mean_difference": +0.03, "half_width": 0.01, "report_only": True}
    equivalence = {"readout": "q", "direction": 0, "mean_difference": +0.0, "half_width": 0.01}
    assert _mutants.judge(named, "FAIL", [entropy_flipped, top_share_kept]) == "reversed"
    assert _mutants.judge(named, "FAIL", [entropy_flipped, equivalence]) == "reversed"
    assert _mutants.judge(named, "FAIL", [{**top_share_kept, "mean_difference": -0.05}]) == "red"
    assert _mutants.judge(named, "FAIL", [equivalence]) == "red"


def test_m111d_record_results_match_judge_and_the_declared_readouts():
    """Each row's result is judge() recomputed from its readouts, and each readout carries the report_only flag its
    criterion declares, so a renamed runner key cannot let a report-only readout gate again (csdp sweep finding)."""
    rows = _mutants.json.loads(re.search(r"```json\n(.*?)\n```", mutation_text(), re.S).group(1))
    by_id = {m.id: m for m in _mutants.MUTANTS}
    assert any(x.get("report_only") for r in rows for x in r["readouts"]), "no report-only readout in the record"
    for r in rows:
        assert r["result"] == _mutants.judge(by_id[r["mutant"]], r["outcome"], r["readouts"]), r["mutant"]
        declared = {x.name: x.report_only for x in CRITERIA[r["criterion"]].readouts}
        for x in r["readouts"]:
            assert x["report_only"] is declared[x["readout"]], (r["mutant"], r["criterion"], x["readout"])


_ospec = importlib.util.spec_from_file_location("tools.level_occupancy", REPO / "tools" / "level_occupancy.py")
_occupancy = importlib.util.module_from_spec(_ospec)
sys.modules[_ospec.name] = _occupancy
_ospec.loader.exec_module(_occupancy)


def occupancy_rows() -> list[dict]:
    text = (REPO / "docs" / "phase_c_level_occupancy.md").read_text(encoding="utf-8")
    return _mutants.json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))


def test_m111a_level_occupancy_record_is_current():
    text = (REPO / "docs" / "phase_c_level_occupancy.md").read_text(encoding="utf-8")
    match = re.search(r"^code_hash: ([0-9a-f]{64})$", text, re.M)
    assert match and match.group(1) == _occupancy.record_hash(), "stale: rerun python3 tools/level_occupancy.py"
    assert {(r["variant"], r["arm"]) for r in occupancy_rows()} == {
        (v, a) for v in ("unmodified", "availability-level-inverted") for a in ("baseline", "treatment")}


def test_m111a_occupancy_bands_are_where_the_inversion_clamps():
    """The record's band edges are read from the constants and the shared pivot, and are where the inverted rule
    meets the deletion (1 on a layer) or makes a layer unavailable (0), so a changed constant cannot leave them stale."""
    edges = _occupancy.bands()
    inverted, layers = _mutants.AVAILABILITY_INVERTED, _layers()
    every, layer_1, none = (edges[k][1] for k in ("deletion_on_every_layer", "deletion_on_layer_1",
                                                    "layers_above_0_unavailable"))
    assert all(_availability(inverted, every, k) == 1.0 for k in layers)
    assert _availability(inverted, every + 1, max(layers)) < 1.0
    assert _availability(inverted, layer_1, 1) == 1.0 > _availability(inverted, layer_1 + 1, 1)
    assert all(_availability(inverted, none, k) == 0.0 for k in layers)
    assert _availability(inverted, none - 1, 1) > 0.0
    assert [edges[k][0] for k in ("deletion_on_every_layer", "deletion_on_layer_1", "layers_above_0_unavailable")] \
        == ["le", "le", "ge"]


def test_m111a_availability_deletion_is_full_on_every_layer():
    """`availability-level-independent` is described as every layer fully available at every level."""
    assert all(_availability(_mutants.AVAILABILITY_DELETED, level, k) == 1.0
               for level in (0.0, 50.0, 100.0) for k in _layers())


def test_m111a_availability_inversion_acts_over_c16s_run():
    """In each arm, over the whole run and over C.16's window, at least a quarter of member-weeks lie in the partial
    band (above the every-layer deletion edge, below the pivot), where the inversion falls with level, so it is not a
    deletion in disguise (csdp sweep finding: a bare > 0 could not tell). The quarter is [I], declared here: well
    below today's shares (about 0.54 to 0.97) and far above a saturated inversion's (near 0)."""
    for r in occupancy_rows():
        if r["variant"] == "availability-level-inverted":
            for span in ("whole_run", "window"):
                s = r[span]
                assert 1.0 - s["deletion_on_every_layer"] - s["layers_above_0_unavailable"] >= 0.25, (r["arm"], span)


def test_m111d_a_broken_mutant_is_not_proof():
    """A mutant whose replacement text is broken (a name its file does not import) proves nothing, whatever its kind;
    an engine exception under a mutant is a red (correctness reviews)."""
    named = next(m for m in _mutants.MUTANTS if m.kind == _mutants.SIGN)
    representation = next(m for m in _mutants.MUTANTS if m.kind == _mutants.REPRESENTATION)
    assert _mutants.judge(named, "BROKEN") == "broken"
    assert _mutants.judge(representation, "BROKEN") == "broken"
    assert _mutants.judge(named, "RAISED") == "red"
    assert NameError in _mutants.BROKEN_ERRORS and AssertionError not in _mutants.BROKEN_ERRORS


def test_m111d_cache_save_keeps_only_results_in_use(tmp_path, monkeypatch):
    """A rerun drops cached results no current mutant uses, so the committed cache does not grow stale entries."""
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    cache = Cache("probe")
    cache.put("a", {"x": 1})
    cache.put("b", {"x": 2})
    cache.save()
    again = Cache("probe")
    assert again.get("a") == {"x": 1}
    again.save()
    assert _mutants.json.loads((tmp_path / "probe.json").read_text()) == {"a": {"x": 1}}


def test_d9_sweep_marks_opposite_sign_and_report_only():
    row = {"criterion": "M11.C.16", "outcome": "PASS", "seeds": 50, "readouts": [
        {"readout": "repertoire_entropy", "direction": -1, "mean_difference": 0.01, "half_width": 0.05},
        {"readout": "top_move_share", "direction": 1, "mean_difference": 0.02, "half_width": 0.01, "report_only": True}]}
    text = _sweep.render([("learning_rate=0.1", row)], {"M11.C.16": {"outcome": "PASS"}}, "0" * 64)
    assert "`repertoire_entropy` +0.01 **opposite sign**" in text
    assert "`top_move_share` +0.02 *(report only)*" in text
    assert "**reversed**" not in text.split("## Machine-readable")[0].split("engine_hash")[1]


def test_d9_sweep_renders_a_row_without_readouts():
    """A RAISED or BROKEN child row carries no readouts; the sweep record must still render (correctness review)."""
    row = {"criterion": "M11.C.16", "outcome": "RAISED", "seeds": 0, "error": "AssertionError: invariant"}
    text = _sweep.render([("learning_rate=0.1", row)], {"M11.C.16": {"outcome": "PASS"}}, "0" * 64)
    assert "| RAISED | 0 | *AssertionError: invariant* |" in text


def test_m115_report_section_11_is_generated():
    """Every number in report §11 comes from the records: the section equals what tools/c16_report.py renders."""
    import importlib

    spec = importlib.util.spec_from_file_location("tools.c16_report", REPO / "tools" / "c16_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    text = report.REPORT.read_text(encoding="utf-8")
    assert text[text.index(report.HEADING):] == report.SECTION, "stale: rerun python3 tools/c16_report.py"


def test_m111d_run_cached_never_caches_an_errored_row(tmp_path, monkeypatch):
    """An engine exception is reported (red) but rerun next time; a failure of the machine stops the run instead of
    becoming a cached red; finished results are flushed before a later failure (learning-qa, 2026-10-09)."""
    import pytest

    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    mutant = next(m for m in _mutants.MUTANTS if m.id == "level-blind")
    rows = {"A": {"criterion": "A", "outcome": "PASS", "seeds": 50, "readouts": []},
            "B": {"criterion": "B", "outcome": "RAISED", "seeds": 0, "error": "AssertionError: invariant"}}
    monkeypatch.setattr(_cache, "run_in_copy", lambda m, tool, *args, **kw: [rows[c] for c in args[1:-2]])
    cache = Cache("probe")
    assert [r["outcome"] for r in _cache.run_cached(cache, mutant, ["A", "B"], 1)] == ["PASS", "RAISED"]
    stored = _mutants.json.loads((tmp_path / "probe.json").read_text())
    assert [r["outcome"] for r in stored.values()] == ["PASS"]
    (tmp_path / "probe.json").unlink()
    rows["B"] = {"criterion": "B", "outcome": "RAISED", "seeds": 0, "error": "BrokenProcessPool: a worker died"}
    with pytest.raises(SystemExit, match="the run failed, not the model"):
        _cache.run_cached(Cache("probe"), mutant, ["A", "B"], 1)
    stored = _mutants.json.loads((tmp_path / "probe.json").read_text())
    assert [r["outcome"] for r in stored.values()] == ["PASS"]  # A, finished before B failed, was kept


_dspec = importlib.util.spec_from_file_location("tools.diagnostic_record", REPO / "tools" / "diagnostic_record.py")
_diagnostic = importlib.util.module_from_spec(_dspec)
sys.modules[_dspec.name] = _diagnostic
_dspec.loader.exec_module(_diagnostic)


def test_d0_diagnostic_record_is_current():
    """The diagnostic record is what its cache renders for the current engine, variants and probes."""
    try:
        rendered = _diagnostic.build(Cache("diagnostic"))
    except KeyError as missing:
        raise AssertionError(f"stale: rerun python3 tools/diagnostic_record.py ({missing})") from None
    assert rendered == _diagnostic.RECORD.read_text(encoding="utf-8"), "stale or edited: rerun the tool"


def test_d0_diagnostics_never_count_as_proof():
    """Coverage reads mutation proof from the mutation record alone; a diagnostic, even a red one, is not proof."""
    import importlib

    coverage = importlib.import_module("tools.spec_coverage")
    assert coverage.MUTATION_RECORD.resolve() == (REPO / "docs" / "phase_c_mutation_record.md").resolve()
    assert _diagnostic.RECORD.resolve() != coverage.MUTATION_RECORD.resolve()
    assert not {d.mutant for d in _diagnostic.DIAGNOSTICS if d.mutant} - {m.id for m in _mutants.MUTANTS}


def test_d0_a_probe_key_holds_its_own_source():
    """Editing one probe reruns that probe only: its key holds its source, its seeds and its variant's edits."""
    import dataclasses

    d = next(x for x in _diagnostic.DIAGNOSTICS if x.probe)
    key = _diagnostic.probe_key("0" * 64, d)
    assert _diagnostic.probe_key("0" * 64, dataclasses.replace(d, seeds=d.seeds + 1)) != key
    assert _diagnostic.probe_key("0" * 64, dataclasses.replace(d, what="relabelled")) == key
