"""The committed ensemble record is current (plan D7).

Purpose: fail the default suite when docs/phase_c_ensemble_record.md was produced against code
         or config that has since changed, or omits a criterion — so a stale record cannot stand
         in for a run.
Spec:    docs/bowen_agent_model_spec_v2.md#M13.4, #M17.A.1, #M11.D.18
Tests:   this file
"""

from __future__ import annotations

import pytest

import importlib.util
import re
import sys
from pathlib import Path
from unittest import mock

from src.bowen.ensemble.criteria import CRITERIA, NOT_BUILT
from src.bowen.io.load import load_constants

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ensemble_record_tool", REPO / "tools" / "ensemble_record.py")
_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tool)


def record_text() -> str:
    return (REPO / "docs" / "phase_c_ensemble_record.md").read_text(encoding="utf-8")


def test_m134_ensemble_record_is_current():
    """MR-23: the record is exactly what its cache renders for the current code, without running anything: a changed
    engine has no cached result (stale), and a hand edit or a renderer change does not match."""
    from tools.record_cache import Cache

    with mock.patch("src.bowen.ensemble.runner.run_criterion", side_effect=AssertionError("ran a criterion in a test")):
        assert _tool.build(Cache(_tool.CACHE_NAME)) == record_text(), "stale or edited: rerun python3 tools/ensemble_record.py"
    match = re.search(r"^code_hash: ([0-9a-f]{64})$", record_text(), re.M)
    assert match and match.group(1) == _tool.result_hash(), "the record does not print the code's hash"


def _stub_verdict(criterion, rules, workers=1):
    from src.bowen.ensemble.runner import Verdict

    return Verdict(criterion.id, criterion.cls, "PASS", 3, [], 0.0, {}, [], [], {})


def _in_tmp_cache(tmp_path):
    import tools.record_cache as record_cache

    return mock.patch.object(record_cache, "CACHE_DIR", tmp_path / "cache")


def test_mr23_record_renders_from_the_cache(tmp_path):
    """MR-23: a build with every result cached renders the record without running a criterion and prints the code's
    hash, which does not cover this tool: a change to the tool alone moves no cached result."""
    from tools.record_cache import Cache

    with _in_tmp_cache(tmp_path), mock.patch("src.bowen.ensemble.runner.run_criterion", side_effect=_stub_verdict):
        first = _tool.build(Cache("c"), workers=1)  # each result flushed to the file as it is made
    with _in_tmp_cache(tmp_path), mock.patch("src.bowen.ensemble.runner.run_criterion",
                                             side_effect=AssertionError("ran a criterion with its result cached")):
        assert _tool.build(Cache("c")) == first
    assert f"code_hash: {_tool.result_hash()}" in first
    assert _tool.result_hash() == _tool.code_hash()
    for cid in CRITERIA:
        assert f"| `{cid}` |" in first, cid


def test_mr23_a_stale_entry_is_rerun(tmp_path):
    """P8 dirty state: a cache holding results for older code is not reused. Every criterion is run again on the
    current code, and saving the cache drops the old entries."""
    from tools.record_cache import Cache

    with _in_tmp_cache(tmp_path), mock.patch("src.bowen.ensemble.runner.run_criterion", side_effect=_stub_verdict) as ran:
        with mock.patch.object(_tool, "result_hash", return_value="0" * 64):
            _tool.build(Cache("c"), workers=1)  # results for older code
        cache = Cache("c")
        old = set(cache.entries)
        ran.reset_mock()
        _tool.build(cache, workers=1)
        assert ran.call_count == len(CRITERIA)
        cache.save()
        assert not old & set(Cache("c").entries) and len(Cache("c").entries) == len(CRITERIA)


def test_m17a1_t018_a_near_limit_criterion_is_rerun_at_the_fixed_count(tmp_path):
    """Revision 70, T-018 (`M17.A.1`): a first verdict within the declared multiple of half-widths of zero, on fewer
    seeds than the declared count, is run again in one block of that count, and the record prints both counts and the
    first verdict; one clear of the limit is not run again."""
    from src.bowen.ensemble.runner import Verdict
    from tools import detection_limit
    from tools.record_cache import Cache

    multiple, fixed = detection_limit.limits()
    near = "M11.C.52"
    calls = []

    def stub(criterion, rules, workers=1):
        calls.append((criterion.id, rules["ensemble_block"], rules["ensemble_cap"]))
        seeds = rules["ensemble_cap"] if rules["ensemble_block"] == fixed else 3
        width = 1.0 if criterion.id == near and seeds == 3 else 0.1
        readout = {"readout": "x", "direction": 1, "mean_difference": 1.0, "half_width": width, "p": 0.01,
                   "holds": True}
        return Verdict(criterion.id, criterion.cls, "PASS", seeds, [readout], 0.0, {}, [], [], {})

    with _in_tmp_cache(tmp_path), mock.patch("src.bowen.ensemble.runner.run_criterion", side_effect=stub):
        text = _tool.build(Cache("c"), workers=1)
    assert [c for c in calls if c[0] == near] == [(near, mock.ANY, mock.ANY), (near, fixed, fixed)]
    assert sum(1 for c in calls if c[1] == fixed) == 1
    assert f"| {fixed} (first run 3 seeds, PASS) |" in text


def test_s1_8a_render_is_not_hashed(tmp_path, monkeypatch):
    """S1.8a: a code edit under ``src/bowen/render/`` on a copy leaves ``code_hash`` unchanged, so a renderer change
    reruns no record; a code edit to the engine on the same copy changes it, so the test can fail."""
    import shutil

    copy = tmp_path / "repo"
    for part in ("src/bowen", "config/bowen"):
        shutil.copytree(REPO / part, copy / part, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(_tool, "REPO", copy)
    before = _tool.code_hash()
    for name, changes in (("render/trace.py", False), ("engine/tick.py", True)):
        path = copy / "src" / "bowen" / name
        path.write_text(path.read_text(encoding="utf-8") + "\nS1_8A_EDIT = 1\n", encoding="utf-8")
        assert (_tool.code_hash() != before) is changes, name


def test_t031_a_config_files_prose_does_not_move_the_hash(tmp_path, monkeypatch):
    """T-031: a note, a blank line or a heading added to a config file leaves ``code_hash`` unchanged, since no loader
    reads them; a table cell or a metadata value changed moves it."""
    import shutil

    copy = tmp_path / "repo"
    for part in ("src/bowen", "config/bowen"):
        shutil.copytree(REPO / part, copy / part, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(_tool, "REPO", copy)
    before = _tool.code_hash()
    acts, constants = copy / "config" / "bowen" / "act_classes.md", copy / "config" / "bowen" / "constants.md"
    acts_text, constants_text = acts.read_text(encoding="utf-8"), constants.read_text(encoding="utf-8")
    acts.write_text("# A new heading\n\n> A new note.\n" + acts_text.replace("> Read by", ">  Read, as before, by")
                    + "\n\n> A closing note.\n", encoding="utf-8")
    assert _tool.code_hash() == before, "prose moved the hash"
    row = next(line for line in acts_text.splitlines() if line.startswith("| ") and "---" not in line)
    for path, old, new in ((acts, acts_text, acts_text.replace(row, row[:-1] + " x|", 1)),
                           (constants, constants_text, constants_text.replace("frozen_at: 2026-10-07", "frozen_at: 2026-10-08"))):
        path.write_text(new, encoding="utf-8")
        assert _tool.code_hash() != before, path.name
        path.write_text(old, encoding="utf-8")


def test_t031_config_text_keeps_a_break_between_table_rows():
    """T-031: a blank line or note between two table rows ends the table, which the strict parser then refuses, so
    the hash keeps the break; a ``## `` section line is kept, since the sectioned parser reads it."""
    from tools.ensemble_record import config_text

    assert config_text("| a |\n\n> n\n| b |\n") != config_text("| a |\n| b |\n")
    assert config_text("> n\n| a |\n\n| b |\n") == config_text("| a |\n\n\n| b |\n")
    assert "## People" in config_text("# Title\n## People\n### sub\n")
    assert "sub" not in config_text("# Title\n## People\n### sub\n")


def test_s1_5b_code_text_keeps_a_line_separator_inside_a_string(tmp_path):
    """S1.5b: ``code_text`` numbers lines as ``tokenize`` does. ``str.splitlines`` also breaks at U+2028, so two of
    them inside a string shifted every later line by two; a docstring after a line holding a two-byte character was
    then looked for at the wrong column and kept in the hash."""
    source = ('X = "a\u2028b\u2028c"\n'
              "class C:\n"
              "    \u00e9\u00e9 = 1\n"
              "    def f(self):\n"
              '        """doc"""\n'
              "        return 1\n")
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    text = _tool.code_text(path)
    assert '"""doc"""' not in text
    assert '"a\u2028b\u2028c"' in text.split("\x00")


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
from tools.record_json import record_json  # noqa: E402
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



def test_m17a1_t297_a_near_limit_mutant_row_is_rerun_at_the_fixed_count(tmp_path):
    """Revision 71, T-297 (4): a mutant's result near the detection limit on fewer seeds than the declared count is run
    again under the same mutant with the runner's block and cap set to that count, and the record prints both counts;
    a result clear of the limit is not. T-297 (5): the rerun's cache definition moves with the code of
    ``rerun_rules``, so a change to how a rerun's rules are built does not reuse cached reruns."""
    from tools import detection_limit

    multiple, fixed = detection_limit.limits()
    near, clear = "M11.C.1", "M11.C.3"
    mutant = _mutants.Mutant("t297-probe", _mutants.NAMED, (near, clear), "config/bowen/constants.md", "a", "b", "probe")
    calls = []

    def stub(cache, run, ids, workers):
        calls.append((run.definition(), tuple(ids)))
        rerun = len(run.also) == len(mutant.also) + 2
        width = 0.1 if rerun or ids[0] == clear else 1.0
        return [{"criterion": cid, "outcome": "PASS", "seeds": fixed if rerun else 3, "readouts": [
            {"readout": "x", "direction": 1, "mean_difference": 1.0, "half_width": 0.1 if cid == clear else width}]}
            for cid in ids]

    with mock.patch.object(_mutants, "MUTANTS", (mutant,)), mock.patch.object(_mutants, "passing",
                                                                                return_value={near, clear}), \
            mock.patch.object(_mutants, "run_cached", side_effect=stub):
        text = _mutants.build(Cache("unused"), workers=1)
    assert [ids for _, ids in calls] == [(near, clear), (near,)]
    assert [new.split(" | ")[:2] for _, _, new in calls[1][0]["edits"][1:]] == [
        ["| `ensemble_block`", str(fixed)], ["| `ensemble_cap`", str(fixed)]]
    assert f"| {fixed} (first run 3 seeds, PASS) |" in text and f"| `{clear}` | PASS | 3 |" in text
    before = detection_limit.rerun_definition(multiple, fixed)
    with mock.patch.object(detection_limit.inspect, "getsource", return_value="def rerun_rules(rules, fixed):\n    0\n"):
        assert detection_limit.rerun_definition(multiple, fixed) != before

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
    rows = record_json(mutation_text())
    proved = {r["criterion"] for r in rows if r["kind"] != _mutants.REPRESENTATION}
    missing = sorted(_mutants.passing() - proved)
    assert not missing, f"passing with no mutant run: {missing}"


def test_m111c_representation_mutants_ran_on_every_passing_criterion():
    rows = record_json(mutation_text())
    for mutant in (m for m in _mutants.MUTANTS if m.kind == _mutants.REPRESENTATION):
        ran = {r["criterion"] for r in rows if r["mutant"] == mutant.id}
        assert _mutants.passing() <= ran, f"{mutant.id} did not run on {sorted(_mutants.passing() - ran)}"


def test_m111b_t077_each_severing_mutant_ran_on_its_passing_criteria():
    """`M11.1b` (revision 70, T-077): every severing mutant has a result on each passing criterion it targets, and the
    record judges it red or survived (a survivor is a finding, reported, not tuned away)."""
    rows = record_json(mutation_text())
    severing = [m for m in _mutants.MUTANTS if m.kind == _mutants.SEVERING]
    assert severing
    for mutant in severing:
        chosen, _ = _mutants.targets(mutant, CRITERIA, _mutants.passing())
        results = {r["criterion"]: r["result"] for r in rows if r["mutant"] == mutant.id}
        assert chosen and set(chosen) <= set(results), f"{mutant.id} did not run on {sorted(set(chosen) - set(results))}"
        assert {results[c] for c in chosen} <= {"red", "reversed", "survived"}, mutant.id


def test_d9_sweep_record_is_current():
    """Plan D9: every composite criterion was swept at low and high α, H and temperature on the current code."""
    text = (REPO / "docs" / "phase_c_sweep_record.md").read_text(encoding="utf-8")
    try:
        rendered = _sweep.build(Cache("sweep"))
    except KeyError as missing:
        raise AssertionError(f"stale: rerun python3 tools/sweep_record.py ({missing})") from None
    assert rendered == text, "stale or edited: rerun python3 tools/sweep_record.py"
    rows = record_json(text)
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


def test_m134_every_record_hashes_the_rule_keys():
    """RULE_KEYS feeds every record, so every hash covers its value; the runner's child produces the cached results, so
    the engine hash and the occupancy record's cover it too (re-gate A, 2026-10-09; Phase D plan B2: the value, not the
    whole file, so the hashing code can change without changing the hash)."""
    runner = (REPO / "tools" / "mutant_runner.py").resolve()
    assert runner in {Path(t).resolve() for t in _occupancy.HASHED_TOOLS}
    assert engine_hash() == _tool.code_hash(runner)
    base = _tool.code_hash()
    original = _tool.RULE_KEYS
    try:
        _tool.RULE_KEYS = original + ("a_new_rule",)
        assert _tool.code_hash() != base
    finally:
        _tool.RULE_KEYS = original
    assert _tool.code_hash() == base


def test_b2_code_text_is_the_same_on_every_python_version():
    """3.12 splits an f-string into several tokens where 3.11 gives one; the hash text must not differ by version
    (review 2026-10-10). Run on this interpreter and, where present, on python3.11 and python3.13."""
    import shutil
    import subprocess
    import sys

    script = ("import sys; sys.path.insert(0, {repo!r}); from pathlib import Path; import hashlib; "
              "from tools.ensemble_record import code_text; "
              "print(hashlib.sha256(''.join(code_text(p) for p in sorted(Path({repo!r}, 'src', 'bowen').rglob('*.py'))).encode()).hexdigest())"
              ).format(repo=str(REPO))
    results = set()
    for exe in {sys.executable, shutil.which("python3.11"), shutil.which("python3.13")} - {None}:
        out = subprocess.run([exe, "-c", script], capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        results.add(out.stdout.strip())
    assert len(results) == 1, results


def test_b2_code_text_keeps_fstring_text_exactly(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text('x = 1\ns = f"{x=}"\n')
    b.write_text('x = 1\ns = f"{x =}"\n')
    assert _tool.code_text(a) != _tool.code_text(b)  # they print different text, so they must hash differently


def test_b2_code_text_drops_only_docstrings_and_comments(tmp_path):
    """The hash ignores prose, keeps code on a docstring's line, and keeps the inside of strings exactly."""
    a = tmp_path / "a.py"
    a.write_text('"""Module doc."""\n\n\ndef f():\n    """Doc."""; x = 1  # note\n    s = """one\n\n  two  """\n    return x\n')
    b = tmp_path / "b.py"
    b.write_text('"""Other doc."""\ndef f():\n    """Changed."""; x = 1\n    s = """one\n\n  two  """\n    return x\n')
    assert _tool.code_text(a) == _tool.code_text(b)
    text = _tool.code_text(a)
    assert "x\x00=\x001" in text and "one\n\n  two  " in text and "note" not in text and "Doc." not in text
    c = tmp_path / "c.py"
    c.write_text('def f():\n    """Doc."""; x = 2\n    s = """one\n\n  two  """\n    return x\n')
    assert _tool.code_text(c) != _tool.code_text(b)  # code on a docstring's line is hashed
    d = tmp_path / "d.py"
    d.write_text('def f():\n    """Doc."""; x = 1\n    s = """one\n  two  """\n    return x\n')
    assert _tool.code_text(d) != _tool.code_text(b)  # a blank line inside a string is content

def test_criteria_settings_are_parsed_strictly(tmp_path):
    """config/bowen/criteria.md: a setting the criteria do not read, or one missing, raises (M11.D.3's rule)."""
    import pytest

    from src.bowen.ensemble.criteria import load_settings
    from src.bowen.scenario.config_parse import ConfigError

    text = (REPO / "config" / "bowen" / "criteria.md").read_text(encoding="utf-8")
    extra = tmp_path / "extra.md"
    extra.write_text(text + "| M11.C.1 | horizon | 5 | [I] |\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no setting 'horizon'"):
        load_settings(extra)
    missing = tmp_path / "missing.md"
    missing.write_text(text.replace("| M11.C.42 | weeks | 40 | [I] |\n", ""), encoding="utf-8")
    with pytest.raises(ConfigError, match="M11.C.42 weeks"):
        load_settings(missing)
    graded = tmp_path / "graded.md"
    graded.write_text(text.replace("| M11.C.42 | weeks | 40 | [I] |", "| M11.C.42 | weeks | 40 | [T] |"), encoding="utf-8")
    with pytest.raises(ConfigError, match="every setting is"):
        load_settings(graded)  # R31.B2: a setting presented as sourced is refused


def test_sweep_tool_shares_the_loaded_mutation_tool():
    """One copy of the runner, not two (third gate finding 3); both tools registered under their own names (fourth and fifth gates)."""
    assert _sweep.Mutant is _mutants.Mutant is _runner.Mutant and _sweep.run_cached is _mutants.run_cached
    assert _mutants.__name__ == "tools.mutation_record" and sys.modules["tools.mutation_record"] is _mutants
    assert _sweep.__name__ == "tools.sweep_record" and sys.modules["tools.sweep_record"] is _sweep


def _setting_reads():
    """Each top-level function's literal ``settings["k"]`` reads, and every literal ``SETTINGS["s"]["k"]`` /
    ``SPELL["k"]`` / ``NODAL["k"]``, over every criteria module (Phase C, Phase D and the social cues)."""
    import ast

    tree = ast.Module(body=[node for name in ("criteria.py", "criteria_d.py", "criteria_d2.py",
                                         "criteria_sc.py")
                            for node in ast.parse((REPO / "src" / "bowen" / "ensemble" / name)
                                                  .read_text(encoding="utf-8")).body], type_ignores=[])

    def key(node):
        return node.slice.value if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) else None

    by_function, module = {}, set()
    for top in tree.body:
        if isinstance(top, ast.FunctionDef):
            by_function[top.name] = {key(n) for n in ast.walk(top) if isinstance(n, ast.Subscript)
                                     and isinstance(n.value, ast.Name) and n.value.id == "settings" and key(n)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and key(n):
            if isinstance(n.value, ast.Name) and n.value.id in ("SPELL", "NODAL"):
                module.add((n.value.id.lower(), key(n)))
            elif isinstance(n.value, ast.Subscript) and isinstance(n.value.value, ast.Name) \
                    and n.value.value.id == "SETTINGS" and key(n.value):
                module.add((key(n.value), key(n)))
    return by_function, module


def test_criteria_required_matches_what_the_arms_read():
    """REQUIRED, the config's schema, and the arms' reads agree in both directions (third gate finding 1)."""
    from src.bowen.ensemble.criteria import INJECTED, REQUIRED

    from src.bowen.ensemble.criteria_d import CRITERIA_D
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2

    by_function, module = _setting_reads()
    criteria = {**CRITERIA, **CRITERIA_D, **CRITERIA_D2}
    # Forward: every key an arm reads is in the settings that criterion is given (else KeyError at ensemble time).
    for cid, criterion in criteria.items():
        missing = by_function[criterion.arm.__name__] - set(criterion.settings)
        assert not missing, f"{cid}: its arm reads {sorted(missing)}, which its settings do not hold"
    # Reverse: every required setting is read — by an arm of its section, or by name at module level.
    read = set(module)
    for cid, criterion in criteria.items():
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
    rows = record_json(mutation_text())
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
    return record_json(text)


def test_m111a_level_occupancy_record_is_current():
    """T-068: the record is what its cache renders for the current engine, without running a child, so a stale record
    or a hand-edited table fails."""
    from tools.record_cache import Cache

    with mock.patch.object(_occupancy, "run_in_copy", side_effect=AssertionError("ran a child in a test")):
        built = _occupancy.build(Cache(_occupancy.CACHE_NAME), run=False)
    assert built == (REPO / "docs" / "phase_c_level_occupancy.md").read_text(encoding="utf-8"), \
        "stale or edited: rerun python3 tools/level_occupancy.py"
    assert {(r["variant"], r["arm"]) for r in occupancy_rows()} == {
        (v, a) for v in ("unmodified", "availability-level-inverted") for a in ("baseline", "treatment")}


def _occupancy_child(tmp_path, monkeypatch):
    """Point the tool at a scratch record and cache; return a row maker for a child's output."""
    monkeypatch.setattr(_occupancy, "RECORD", tmp_path / "phase_c_level_occupancy.md")
    monkeypatch.setattr(_occupancy, "REPO", tmp_path)  # the path it prints is relative to the repository
    monkeypatch.setattr(sys, "argv", ["level_occupancy.py"])
    import tools.record_cache as record_cache

    monkeypatch.setattr(record_cache, "CACHE_DIR", tmp_path / "cache")
    shares = {name: 0.5 for name in _occupancy.bands()}
    return lambda arm: {"arm": arm, "member_weeks": {"whole_run": 10, "window": 5}, "whole_run": shares,
                        "window": shares}


def test_m111a_level_occupancy_refuses_a_short_child(tmp_path, monkeypatch):
    """A child that exits 0 having printed a row for only some of C.16's arms is a failed variant, and no record is
    written; a row for each arm writes it (cold sweep, 2026-10-20: the rows were never counted, so a short record
    could be written with no warning)."""
    row = _occupancy_child(tmp_path, monkeypatch)
    record = _occupancy.RECORD
    monkeypatch.setattr(_occupancy, "run_in_copy", lambda *args, **kw: [row("baseline")])
    assert _occupancy.main() == 1 and not record.exists()
    monkeypatch.setattr(_occupancy, "run_in_copy", lambda *args, **kw: [row(a) for a in CRITERIA["M11.C.16"].arms])
    assert _occupancy.main() == 0 and len(occupancy_rows_of(record)) == 4  # two variants, two arms each


def test_t068_the_occupancy_key_covers_the_band_pivot(monkeypatch):
    """T-068 (review 2026-11-17): the band edges come from ``INVERSION_PIVOT``, so a changed pivot is a different key
    and the cached rows are not rendered against new edges."""
    before = _occupancy.variant_key(None)
    monkeypatch.setattr(_occupancy, "INVERSION_PIVOT", _occupancy.INVERSION_PIVOT + 0.1)
    assert _occupancy.variant_key(None) != before


def test_t026_a_failed_occupancy_child_keeps_the_other_variants_rows(tmp_path, monkeypatch):
    """T-026, T-068: a variant whose child fails does not lose the other's rows, which are cached; the run exits 1
    without writing the record, and the next run runs only the variant with no cached rows."""
    from tools.child_process import ChildFailed

    row = _occupancy_child(tmp_path, monkeypatch)
    arms = CRITERIA["M11.C.16"].arms
    calls = []

    def failing(mutant, *args, **kw):
        calls.append(mutant.id if mutant else None)
        if mutant is not None:
            raise ChildFailed("toy child failed", [])
        return [row(a) for a in arms]

    monkeypatch.setattr(_occupancy, "run_in_copy", failing)
    assert _occupancy.main() == 1 and not _occupancy.RECORD.exists()
    assert calls == [None, "availability-level-inverted"]
    calls.clear()
    monkeypatch.setattr(_occupancy, "run_in_copy", lambda mutant, *a, **kw: calls.append(mutant.id) or [row(a) for a in arms])
    assert _occupancy.main() == 0 and calls == ["availability-level-inverted"]
    assert len(occupancy_rows_of(_occupancy.RECORD)) == 4


def occupancy_rows_of(path) -> list[dict]:
    return record_json(path.read_text(encoding="utf-8"))


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


@pytest.mark.xfail(strict=True, reason="T-264: at revision 66 the availability-level-inverted treatment arm is 84% of member-weeks in the every-layer deletion band over the whole run (98.5% in the window), so the partial band is under the quarter; for the owner")
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
    start, end = report.bounds(text)
    assert text[start:end].rstrip("\n") == report.SECTION.rstrip("\n"), "stale: rerun python3 tools/c16_report.py"
    assert "## 12." in text[end:], "the section after §11 was lost"
    # T-148: the report's two `M11.C.16` rows and the spec's `M11.5` row restate §11, so they are generated with it
    for key, (path, prefix, after) in report.ROW_PREFIXES.items():
        assert report.row_line(path.read_text(encoding="utf-8"), prefix, after) == report.RENDERED[key], (
            f"stale {key}: rerun python3 tools/c16_report.py")


def _c16_render(edits: dict[str, dict]):
    """``tools/c16_report.render`` on the committed records with some of `M11.C.16`'s mutant rows edited."""
    import copy

    import tools.c16_report as report

    mutants, occupancy, ensemble, sweep = copy.deepcopy(report._records())
    for row in mutants:
        if row["criterion"] == "M11.C.16" and row["mutant"] in edits:
            row.update(edits[row["mutant"]])
    return report.render(mutants, occupancy, ensemble, sweep)


def _gating(mean: float, half: float) -> list[dict]:
    return [{"readout": "repertoire_entropy", "mean_difference": mean, "half_width": half},
            {"readout": "top_move_share", "mean_difference": -mean, "half_width": half}]


def _c16_audit(edits: dict[str, dict]) -> dict:
    """The committed emergence audit record's JSON with some of `M11.C.16`'s carrier candidates edited."""
    import copy

    import tools.c16_report as report
    from tools.record_json import record_json

    audit = copy.deepcopy(record_json(report.AUDIT_RECORD))
    for entry in audit["carriers"]:
        if entry["criterion"] == "M11.C.16":
            for cand in entry["candidates"]:
                if cand["ablation"] in edits:
                    cand.update(edits[cand["ablation"]])
    return audit


def test_t189_c16_row_l0_sentence_follows_the_audit_record():
    """T-189 (owner 2026-11-05, spec revision 49): the `M11.5` row's L0 sentence is computed from the emergence audit
    record. Adversarial (P7): each edit makes the committed sentence false, and the rendered row follows it; a missing
    candidate stops the tool. Mutation: the sentence written as literal text fails the edited-verdict checks."""
    import copy

    import pytest

    import tools.c16_report as report

    committed = report.RENDERED["spec_row"]
    # At revision 67 (T-266) the audit's L0 inversion turns it red and the removal as well is red (100 seeds).
    assert "inverting availability turns it red (+0.000542 ± 0.0006, 100 seeds)" in committed
    assert "with availability removed as well it is red, not still passing (+0.000285 ± 0.00055, 100 seeds)" in committed
    records = copy.deepcopy(report._records())
    same = report.render(*records, audit=_c16_audit(
        {"C7:availability-and-learner-removed": {"verdict": "reversed"},
         "C7:availability-level-inverted": {"verdict": "survived"}}))["spec_row"]
    assert "inverting availability leaves it passing" in same
    assert "it is reversed, not still passing (+0.000285" in same
    other = report.render(*records, audit=_c16_audit(
        {"C7:availability-and-learner-removed": {"verdict": "survived"},
         "C7:availability-level-inverted": {"verdict": "red"}}))["spec_row"]
    assert "inverting availability turns it red" in other
    assert "it is still passing (+0.000285" in other
    mutants = copy.deepcopy(records[0])
    for row in mutants:
        if row["criterion"] == "M11.C.16" and row["mutant"] == "learner-disabled":
            row["seeds"] = 120
    # The runner stops a run when it converges, so the two learner-disabled runs may differ: each count is stated. Since
    # revision 71 (T-297 (4)) the removal's row is rerun at the detection limit, 200 seeds.
    assert "(120 seeds alone, 200 with availability removed)" in report.render(mutants, *records[1:])["spec_row"]
    gone = _c16_audit({})
    for entry in gone["carriers"]:
        if entry["criterion"] == "M11.C.16":
            entry["candidates"] = [c for c in entry["candidates"] if c["ablation"] != "C7:availability-level-inverted"]
    with pytest.raises(SystemExit, match="C7:availability-level-inverted"):
        report.render(*records, audit=gone)
    with pytest.raises(SystemExit, match="has no single worded"):
        report.render(*records, audit=_c16_audit({"C7:availability-and-learner-removed": {"verdict": "broken"}}))


def test_t148_c16_sentences_follow_the_mutant_results():
    """T-148 (stage 3 milestone review MR-1): the sentences §11 wrote as literal text (the learning bullet, "no other
    single rule removes the result", "what does not hold") and the rows that restate §11 are computed from the mutant
    results. Adversarial (P7): each edit below makes the old literal sentence false, and the rendered text follows it;
    a result the generator has no wording for stops it. Mutation: any of those sentences put back as literal text
    fails one of these checks."""
    import pytest

    red_band = _c16_render({"band-level-independent": {"result": "red", "readouts": _gating(-0.001, 0.003)}})
    assert "Removing `band-level-independent` alone also turns it red" in red_band["section"]
    assert "No other single rule's removal turns it red" not in red_band["section"]
    assert "`band-level-independent`" in red_band["row_class"] and "but for `band-level-independent`" in red_band["section"]
    calm = _c16_render({"steepness-inverted": {"result": "survived", "readouts": _gating(-0.03, 0.006)},
                        "band-level-independent": {"result": "survived", "readouts": _gating(-0.036, 0.007)},
                        # the record's own results for these two moved at revision 70 (T-300); calm sets them here
                        "learner-inverted": {"result": "survived", "readouts": _gating(-0.006, 0.007)},
                        "level-blind": {"result": "survived", "readouts": _gating(-0.003, 0.003)}})
    assert "Of the other inversions" not in calm["section"] and "No other single rule's removal" in calm["section"]
    assert "`steepness-inverted`" not in calm["row_class"]
    kept = _c16_render({"availability-and-learner-removed": {"result": "survived", "readouts": _gating(-0.02, 0.005)}})
    assert "**Learning changes the result's size, not its sign.**" in kept["section"]
    assert "with availability and the learner removed together." in kept["section"]
    assert "but not with availability removed as well" not in kept["row_status"]
    gone = _c16_render({"availability-and-learner-removed": {"result": "reversed", "readouts": _gating(0.01, 0.002)}})
    assert "**Learning alone does not carry the result" in gone["section"]
    assert "so learning carries the result only jointly with availability, not alone." in gone["section"]
    assert "but not with availability removed as well" in gone["row_status"]
    assert "with availability removed as well it is reversed" in gone["spec_row"]
    with pytest.raises(SystemExit, match="learner-disabled"):
        _c16_render({"learner-disabled": {"result": "red"}})
    with pytest.raises(SystemExit, match="has no wording"):
        _c16_render({"routing-level-inverted": {"result": "broken"}})
    # the batch's correctness pass: another inversion reversing C.16 is not "no inversion run reverses C.16"
    other = _c16_render({"availability-level-inverted": {"result": "survived", "readouts": _gating(-0.02, 0.005)},
                         "steepness-inverted": {"result": "reversed", "readouts": _gating(0.02, 0.005)}})
    assert "No inversion run" not in other["section"]
    assert "does not reverse C.16; `steepness-inverted` does (below).** It leaves it passing" in other["section"]
    assert "`steepness-inverted` flips C.16, which by that test would make it one" in other["section"]
    # and availability alone turning it red is not "either set alone keeps a passing result"
    alone = _c16_render({"availability-level-independent": {"result": "red", "readouts": _gating(-0.001, 0.003)},
                         "c16-cited-grounds-removed": {"result": "survived", "readouts": _gating(-0.03, 0.006)}})
    assert "Either set alone keeps a passing result" not in alone["section"]
    assert "either set alone leaves it passing" not in alone["section"]
    assert "each single deletion survives" not in alone["section"]
    assert "Availability alone carries it" in alone["section"]
    assert "but for `availability-level-independent`" in alone["section"]
    # the table's "representation mutants: unchanged" row is checked, not written (the batch's learning-qa pass)
    with pytest.raises(SystemExit, match="representation mutants are not all unchanged"):
        _c16_render({"clamp-within-tolerance": {"result": "encoding artefact"}})
    import tools.c16_report as report
    mutants, occupancy, ensemble, sweep = report._records()
    with pytest.raises(SystemExit, match="no M11.C.16 row"):
        report.render(mutants, occupancy, ensemble, [r for r in sweep if r["criterion"] != "M11.C.16"])
    # §10's table has a row that starts like §13's status row: the status row is looked for from §13's heading on
    text = report.REPORT.read_text(encoding="utf-8")
    path, prefix, after = report.ROW_PREFIXES["row_status"]
    assert after and text.count("\n" + prefix) >= 2 and text.index("\n" + prefix) < text.index(after)
    with pytest.raises(SystemExit, match="lines start with"):
        report.row_line(text, prefix)
    assert report.row_line(text, prefix, after).startswith(prefix)
    with pytest.raises(SystemExit, match="name a mutant twice"):
        report.render(mutants + [r for r in mutants if r["criterion"] == "M11.C.16"][:1], occupancy, ensemble, sweep)


def test_m111d_run_cached_never_caches_an_errored_row(tmp_path, monkeypatch):
    """An engine exception is reported (red) but rerun next time; a failure of the machine is a FAILED row, never a
    cached red, and the run reports it (version 2 stage 1, S1.5a: before it, the run stopped); finished results are
    flushed before a later failure (learning-qa, 2026-10-09)."""
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_cache, "_uncached", [])
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
    assert [r["outcome"] for r in _cache.run_cached(Cache("probe"), mutant, ["A", "B"], 1)] == ["PASS", _cache.FAILED]
    stored = _mutants.json.loads((tmp_path / "probe.json").read_text())
    assert [r["outcome"] for r in stored.values()] == ["PASS"]  # A, finished before B failed, was kept
    assert _cache.exit_status() == 1


def test_m111d_a_raised_or_broken_row_exits_nonzero(tmp_path, monkeypatch, capsys):
    """A run whose only errored row is RAISED, or BROKEN, exits 1 and names the row: it is in the record and not in
    the cache, so the record's test fails until it gives a result (cold sweep, 2026-10-20, C1: only FAILED rows were
    counted, and the tool exited 0). A run with no errored row exits 0."""
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    mutant = next(m for m in _mutants.MUTANTS if m.id == "level-blind")
    for outcome, error in (("RAISED", "AssertionError: invariant"), ("BROKEN", "NameError: name 'x' is not defined")):
        monkeypatch.setattr(_cache, "_uncached", [])
        rows = {"A": {"criterion": "A", "outcome": "PASS", "seeds": 50, "readouts": []},
                "B": {"criterion": "B", "outcome": outcome, "seeds": 0, "error": error}}
        monkeypatch.setattr(_cache, "run_in_copy", lambda m, tool, *args, rows=rows, **kw: [rows[c] for c in args[1:-2]])
        assert [r["outcome"] for r in _cache.run_cached(Cache(outcome), mutant, ["A", "B"], 1)] == ["PASS", outcome]
        capsys.readouterr()
        assert _cache.exit_status() == 1
        assert f"level-blind → B ({outcome})" in capsys.readouterr().out
    monkeypatch.setattr(_cache, "_uncached", [])
    assert _cache.run_cached(Cache("clean"), mutant, ["A"], 1)[0]["outcome"] == "PASS"
    assert _cache.exit_status() == 0


def test_m111d_run_cached_returns_a_fresh_row_as_a_reload_would(tmp_path, monkeypatch):
    """A row run now and the same row read back from the cache render alike (key order included); otherwise a
    record written in the run that produced it fails its own currency test (found 2026-10-09)."""
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    mutant = next(m for m in _mutants.MUTANTS if m.id == "level-blind")
    row = {"outcome": "PASS", "criterion": "A", "seeds": 50, "readouts": [{"readout": "x", "direction": 1}]}
    monkeypatch.setattr(_cache, "run_in_copy", lambda m, tool, *args, **kw: [row])
    fresh = _cache.run_cached(Cache("probe"), mutant, ["A"], 1)
    reloaded = _cache.run_cached(Cache("probe"), mutant, ["A"], 1)
    dump = lambda rows: _mutants.json.dumps(rows)  # noqa: E731 — order-sensitive, as rendering is
    assert dump(fresh) == dump(reloaded)
    assert list(fresh[0]) == sorted(row)


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
    """Coverage reads mutation proof from the mutation records alone; a diagnostic, even a red one, is not proof."""
    import importlib

    coverage = importlib.import_module("tools.spec_coverage")
    records = {p.resolve() for p in coverage.MUTATION_RECORDS}
    assert (REPO / "docs" / "phase_c_mutation_record.md").resolve() in records
    assert _diagnostic.RECORD.resolve() not in records
    assert not {d.mutant for d in _diagnostic.DIAGNOSTICS if d.mutant} - {m.id for m in _mutants.MUTANTS}


def test_d0_a_probe_key_holds_its_own_source():
    """Editing one probe reruns that probe only: its key holds its source, its seeds and its variant's edits."""
    import dataclasses

    d = next(x for x in _diagnostic.DIAGNOSTICS if x.probe)
    key = _diagnostic.probe_key("0" * 64, d)
    assert _diagnostic.probe_key("0" * 64, dataclasses.replace(d, seeds=d.seeds + 1)) != key
    assert _diagnostic.probe_key("0" * 64, dataclasses.replace(d, what="relabelled")) == key


def test_d0_a_probe_key_covers_the_whole_probe_module():
    """A probe reads helpers and module constants (`horizons` reads C4_NODAL_WEEKS); its key holds them all."""
    import inspect

    for d in (x for x in _diagnostic.DIAGNOSTICS if x.probe):
        assert inspect.getsource(_diagnostic.probes) in _diagnostic.probe_source(d.probe)


def _scripted(statuses: dict[str, int]):
    """An arm result whose scripted-act bookkeeping is ``statuses``, as ``criteria.scripted_bookkeeping`` books it."""
    from collections import Counter

    from src.bowen.ensemble.runner import ArmResult

    moves = Counter({f"(scripted act: {s})": n for s, n in statuses.items()})
    moves["(scripted acts)"] = sum(statuses.values())
    return ArmResult({"x": 0.0}, moves)


def test_t073_arm_made_classifies_every_case():
    """A scripted act that was neither made nor skipped is not reported as made (narrow re-check, 2026-10-09); a
    placed act rewritten afterwards is not made (T-073)."""
    arm_made = _diagnostic.probes.arm_made
    assert arm_made(_scripted({"made": 1})) == "made"
    assert arm_made(_scripted({"not legal": 1})) == "skipped"
    assert arm_made(_scripted({"rewritten": 1})) == "skipped"
    assert arm_made(_scripted({"made": 1, "owed a step": 1})) == "partly"
    assert arm_made(_scripted({"not reached": 1})) == "not reached"
    assert arm_made(_scripted({})) == "none"


def test_t073_horizons_judge_made_from_the_arm_result(monkeypatch):
    """T-073: `horizons` reads whether an act was made from the arm's own result, not from the acts placed."""
    import types

    import src.bowen.ensemble.criteria as criteria

    def arm(name, seed, settings):
        return _scripted({"rewritten": 1} if name == "treatment" else {"made": 1})

    fake = {cid: types.SimpleNamespace(settings={}, arm=arm) for cid in ("M11.C.4", "M11.C.5")}
    monkeypatch.setattr(criteria, "CRITERIA", fake)
    rows = _diagnostic.probes.horizons(1)
    assert rows and all(r["acts"] == "treatment skipped, baseline made" for r in rows)


def test_t025_probes_take_no_builtin_sum():
    """T-025: a float mean in the probes is an fsum and an integer total a count, never the builtin sum."""
    import ast

    tree = ast.parse((REPO / "tools" / "probes.py").read_text(encoding="utf-8"))
    assert [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "sum"] == []


def test_d0_summarise_leaves_out_seeds_with_nothing_to_measure():
    rows = [{"arm": "a", "seed": 0, "x": None}, {"arm": "a", "seed": 1, "x": 2.0}, {"arm": "a", "seed": 2, "x": 4.0}]
    line = _diagnostic.summarise(rows)[2]
    assert line == "| a | 3 | 3 / 3 / 2 / 4 / 100% |"


def test_m134_code_hash_ignores_prose_but_not_code(tmp_path):
    """Editing a comment or docstring in the engine must not rerun every record; editing code must (learnings P38).
    Phase D plan B2: a docstring on its def's own line is prose too, and only its token is dropped, so the code that
    shares the line is still hashed."""
    from tools.ensemble_record import code_text

    base = 'def f(x):\n    """Doc."""\n    # a comment\n    return x + 1  # trailing\n\ndef g(): "inline"; return 2\n'
    variants = {
        "prose": base.replace("Doc.", "Changed doc.").replace("a comment", "another comment").replace("trailing", "t"),
        "code": base.replace("x + 1", "x + 2"),
        "inline docstring": base.replace('"inline"', '"other"'),
        "code on a docstring's line": base.replace("return 2", "return 3"),
    }
    path = tmp_path / "m.py"
    path.write_text(base)
    original = code_text(path)
    seen = {}
    for name, text in variants.items():
        path.write_text(text)
        seen[name] = code_text(path) == original
    assert seen == {"prose": True, "code": False, "inline docstring": True, "code on a docstring's line": False}


def test_criteria_modules_bind_no_top_level_name_twice():
    """A criteria module that binds one top-level name twice changes every arm that read the first binding: on
    2026-10-17 `M11.C.46`'s `PARTNER` (the triad's M) replaced `M11.C.22`'s and `M11.C.11`'s (the nuclear four's c2),
    caught only by comparing seeds after a record had been regenerated. Loop variables named with a leading `_` may
    repeat."""
    import ast

    for name in ("criteria.py", "criteria_d.py", "criteria_d2.py"):
        tree = ast.parse((REPO / "src" / "bowen" / "ensemble" / name).read_text(encoding="utf-8"))
        bound: dict[str, list[int]] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.For)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name) and not (isinstance(node, ast.For) and n.id.startswith("_")):
                        bound.setdefault(n.id, []).append(node.lineno)
        twice = {k: v for k, v in bound.items() if len(v) > 1}
        assert twice == {}, f"{name}: {twice}"


def test_t069_infrastructure_errors_match_by_class_not_by_name(monkeypatch):
    """A subclass of OSError (here FileNotFoundError) is an error of the machine, so its row is FAILED and never
    cached; the user's own stop (KeyboardInterrupt) and a model error are not."""
    monkeypatch.setattr(_cache, "_uncached", [])
    rows = {"A": {"criterion": "A", "outcome": "RAISED", "seeds": 0, "error": "FileNotFoundError: no such file"},
            "B": {"criterion": "B", "outcome": "RAISED", "seeds": 0, "error": "KeyboardInterrupt: "},
            "C": {"criterion": "C", "outcome": "RAISED", "seeds": 0, "error": "AssertionError: invariant"}}
    monkeypatch.setattr(_cache, "run_in_copy", lambda m, tool, *args, **kw: list(rows.values()))
    out = _cache.child_rows(None, "mutant_runner.py", ["A", "B", "C"], 1, "t")
    assert [r["outcome"] for r in out] == [_cache.FAILED, "RAISED", "RAISED"]


def test_t070_a_probe_with_no_rows_and_a_diagnostic_with_no_criterion_stop_the_record(tmp_path, monkeypatch):
    import dataclasses

    import pytest

    probe = next(x for x in _diagnostic.DIAGNOSTICS if x.probe)
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_diagnostic, "run_in_copy", lambda *a, **k: [])
    with pytest.raises(SystemExit, match="returned no rows"):
        _diagnostic.results(Cache("probe"), probe, 1)
    bare = dataclasses.replace(next(x for x in _diagnostic.DIAGNOSTICS if not x.probe), criteria=())
    with pytest.raises(SystemExit, match="names no criterion"):
        _diagnostic.results(Cache("probe"), bare, 1)


def test_t026_a_probe_that_fails_is_reported_and_the_record_goes_on(tmp_path, monkeypatch):
    from tools.child_process import ChildFailed

    probe = next(x for x in _diagnostic.DIAGNOSTICS if x.probe)
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_cache, "_uncached", [])

    def fail(*a, **k):
        raise ChildFailed("probe timed out after 7200 s", [])

    monkeypatch.setattr(_diagnostic, "run_in_copy", fail)
    cache = Cache("probe")
    assert _diagnostic.results(cache, probe, 1) == []
    assert not cache.entries and _cache.exit_status() == 1
    assert "failed" in _diagnostic.summarise([])[0]


def test_t030_a_record_write_is_whole_or_absent(tmp_path, monkeypatch):
    import pytest

    from tools.atomic_write import write_text_atomic

    target = tmp_path / "r.md"
    write_text_atomic(target, "old\n")
    monkeypatch.setattr("os.replace", lambda *a: (_ for _ in ()).throw(OSError("killed")))
    with pytest.raises(OSError):
        write_text_atomic(target, "new\n")
    assert target.read_text() == "old\n" and [p.name for p in tmp_path.iterdir()] == ["r.md"]


def test_t029_a_flush_keeps_a_result_another_run_wrote_meanwhile(tmp_path, monkeypatch):
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    first, second = Cache("two"), Cache("two")
    first.put("a" * 64, {"outcome": "PASS"})
    first.flush()
    second.put("b" * 64, {"outcome": "PASS"})
    second.flush()
    assert set(Cache("two").entries) == {"a" * 64, "b" * 64}
    second.save()  # the end-of-record prune keeps the other run's result too
    assert set(Cache("two").entries) == {"a" * 64, "b" * 64}
