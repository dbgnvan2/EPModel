"""The constant sweep of stage 2 (`M17.E.1`) and its detection-limit rule (`M17.A.1`, revision 37).

Purpose: prove the sweep driver's arithmetic (the fraction, the rerun rule, the range draw), that a sample edits the
         constants file's rows and nothing else, that the config is read strictly, that a cached result on a stale
         engine hash is not reused (P8), and that the record renders from the cache.
Spec:    docs/bowen_agent_model_spec_v2.md#M17.E.1, #M17.A.1; docs/implementation_plan_version_2.md#2.3 (S2.1–S2.7)
Tests:   this file

No test here runs the engine: ``tools.record_cache.child_rows`` is replaced by a stub that answers from the mutant's
id, and every patched module attribute is restored by ``mock.patch``.
"""

from __future__ import annotations

import io
import re
import shutil
import tokenize
from pathlib import Path
from unittest import mock

import pytest

import tools.record_cache as record_cache
import tools.sweep_config as sweep_config
import tools.version_2_sweep as tool
from tools.mutant_runner import apply
from tools.record_cache import Cache
from src.bowen.io.load import load_constants

REPO = Path(__file__).resolve().parents[1]
CONFIG_TEXT = REPO.joinpath("config", "sweep", "sweep.md").read_text(encoding="utf-8")

# A small config for the driver tests: two real criteria (one passing, one an accepted finding), two constants each.
TEST_CONFIG = """# test config

## Rules

| constant | value | grade | unit | why |
|---|---|---|---|---|
| `sweep_samples` | 8 | [I] | samples | test |
| `sweep_seed` | 2 | [I] | seed | test |
| `detection_limit_multiple` | 2.0 | [I] | half-widths | test |
| `detection_limit_seeds` | 200 | [I] | seeds | test |

## Ranges

| criterion | constant | low | high | scale | bound | grade | why it is read |
|---|---|---|---|---|---|---|---|
| `M11.C.16` | `learning_rate` | 0.1 | 0.4 | log | — | [I] | test |
| `M11.C.16` | `credit_horizon` | 2 | 6 | integer | — | [I] | test |
| `M11.C.42` | `learning_rate` | 0.1 | 0.4 | log | — | [I] | test |
| `M11.C.42` | `triangle_activity_window` | 2 | 8 | integer | — | [I] | test |
"""

# The stub's answer per sample index: (outcome, mean_difference, half_width) of the first run; a rerun (a mutant
# whose edits set the stopping rows) answers with the second tuple. Samples 4, 5 and 7 are within two half-widths
# (an UNDETERMINED first run always is: it stopped at the cap without converging).
FIRST = {0: ("PASS", 3.0, 0.5), 1: ("PASS", 2.0, 0.4), 2: ("PASS", 1.5, 0.3), 3: ("FAIL", -2.0, 0.5),
         4: ("PASS", 0.5, 0.4), 5: ("UNDETERMINED", 0.6, 0.9), 6: ("RAISED", 0.0, 0.0), 7: ("FAIL", -0.3, 0.4)}
RERUN = {4: ("PASS", 0.9, 0.2), 5: ("UNDETERMINED", 0.6, 0.6), 7: ("FAIL", -0.5, 0.2)}
RERUNS, CLEAR = (4, 5, 7), (0, 1, 2, 3)
FIRST_SEEDS, RERUN_SEEDS = 100, 200


def _index(mutant) -> int:
    return int(re.search(r"sample (\d+)", mutant.id).group(1))


def _is_rerun(mutant) -> bool:
    return any("ensemble_block" in old for _f, old, _n in mutant.also)


def _row(cid: str, outcome: str, mean: float, half: float, seeds: int) -> dict:
    row = {"criterion": cid, "outcome": outcome, "seeds": seeds, "readouts": [
        {"readout": "effect", "direction": 1, "mean_difference": mean, "half_width": half},
        {"readout": "shown", "direction": 1, "mean_difference": 0.01, "half_width": 5.0, "report_only": True}]}
    if outcome == "RAISED":
        row = {"criterion": cid, "outcome": outcome, "seeds": 0, "error": "ValueError: test"}
    return row


def _stub_child_rows(mutant, _tool, ids, _workers, _label):
    i = _index(mutant)
    answer, seeds = (RERUN[i], RERUN_SEEDS) if _is_rerun(mutant) else (FIRST[i], FIRST_SEEDS)
    return [_row(cid, *answer, seeds) for cid in ids]


def _patched(tmp_path, engine="a" * 64, child=_stub_child_rows):
    """The driver on a temporary cache, a fixed engine hash and a stubbed child; the test config in place."""
    config = tmp_path / "sweep.md"
    if not config.exists():
        config.write_text(TEST_CONFIG, encoding="utf-8")
    return (mock.patch.object(record_cache, "CACHE_DIR", tmp_path / "cache"),
            mock.patch.object(record_cache, "engine_hash", return_value=engine),
            mock.patch.object(tool, "engine_hash", return_value=engine),
            mock.patch.object(record_cache, "child_rows", side_effect=child),
            mock.patch.multiple(tool, load_config=lambda: sweep_config.load_config(config),
                                not_run_keys=lambda: sweep_config.not_run_keys(config)))


def _build(tmp_path, workers=1, **kw):
    patches = _patched(tmp_path, **kw)
    with patches[0], patches[1], patches[2], patches[3] as ran, patches[4]:
        cache = Cache("sweep_test")
        text = tool.build(cache, workers)
        cache.save()
    return text, ran


def _results(text: str) -> list[dict]:
    import json

    return json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])


# --- S2.1 the detection-limit rule ------------------------------------------------------------------------------

def test_m17a1_s2_1_a_near_limit_sample_is_rerun_at_the_fixed_count(tmp_path):
    """A sample whose first run leaves the gating effect within two half-widths of zero is run again with the
    stopping rows set to the fixed count, the verdict is the second run's, and both seed counts are recorded."""
    text, ran = _build(tmp_path)
    for result in _results(text):
        rows = {r["index"]: r for r in result["rows"]}
        for i in RERUNS:
            assert rows[i]["rerun"] and rows[i]["first_outcome"] == FIRST[i][0] and rows[i]["outcome"] == RERUN[i][0]
            assert rows[i]["seeds"] == FIRST_SEEDS and rows[i]["rerun_seeds"] == RERUN_SEEDS
        assert result["reruns"] == len(RERUNS)
    reruns = [c.args[0] for c in ran.call_args_list if _is_rerun(c.args[0])]
    assert len(reruns) == 2 * len(RERUNS)  # two criteria
    for mutant in reruns:
        stopping = {old.split("`")[1]: new for _f, old, new in mutant.also if "ensemble_" in old}
        assert stopping == {"ensemble_block": "| `ensemble_block` | 200 | ", "ensemble_cap": "| `ensemble_cap` | 200 | "}
    assert f"{RERUN_SEEDS}" in text and "yes (first PASS)" in text


def test_m17a1_s2_1_a_clear_effect_is_not_rerun(tmp_path):
    """A first run whose every gating effect is at least two half-widths from zero stands; a report-only readout
    within the limit does not trigger a rerun."""
    text, ran = _build(tmp_path)
    for result in _results(text):
        for r in result["rows"]:
            if r["index"] in CLEAR:
                assert not r["rerun"] and r["outcome"] == FIRST[r["index"]][0] and r["seeds"] == FIRST_SEEDS
    assert not any(_is_rerun(c.args[0]) and _index(c.args[0]) in CLEAR for c in ran.call_args_list)
    assert tool.near_limit({"readouts": [{"direction": 1, "mean_difference": 1.0, "half_width": 0.49}]}, 2.0) is False
    assert tool.near_limit({"readouts": [{"direction": 1, "mean_difference": 1.0, "half_width": 0.51}]}, 2.0) is True
    assert tool.near_limit({"readouts": [{"direction": 0, "mean_difference": 0.0, "half_width": 1.0}]}, 2.0) is False


def test_m17a1_s2_1_a_fixed_seed_criterion_is_not_rerun(tmp_path):
    """A first run that already used the fixed count or more is not run again, however close to the limit."""
    def child(mutant, _tool, ids, _workers, _label):
        assert not _is_rerun(mutant), "rerun a sample that had already run at the fixed count"
        return [_row(cid, "PASS", 0.1, 0.4, RERUN_SEEDS) for cid in ids]

    text, _ran = _build(tmp_path, child=child)
    for result in _results(text):
        assert result["reruns"] == 0 and all(r["seeds"] == RERUN_SEEDS for r in result["rows"])


def test_m17a1_s2_1_a_criterion_fixing_its_own_seeds_below_the_count_is_refused(tmp_path):
    """A criterion with its own `seeds` below the fixed count ignores the sweep's block and cap, so its rerun would run
    the same seeds again; the sweep refuses rather than record a rerun that is not one (`M17.A.1`)."""
    def child(mutant, _tool, ids, _workers, _label):
        seeds = 100  # the criterion's own count, whatever the stopping rows say
        answer = RERUN[_index(mutant)] if _is_rerun(mutant) else FIRST[_index(mutant)]
        return [_row(cid, *answer, seeds) for cid in ids]

    patches = _patched(tmp_path, child=child)
    with patches[0], patches[1], patches[2], patches[3], patches[4], pytest.raises(SystemExit, match="fixes its own seed count"):
        tool.build(Cache("sweep_test"), 1)


def test_m17a1_a_criterion_fixing_its_own_seeds_is_refused_before_its_rerun_runs(tmp_path):
    """Review MR-12 (owner 2026-10-24): the refusal comes before the rerun is run, so the criterion's own seeds are
    never run a second time. Mutation: the check before the rerun removed (the rerun runs, then the backstop refuses)."""
    reruns = []

    def child(mutant, _tool, ids, _workers, _label):
        if _is_rerun(mutant):
            reruns.append(mutant.id)
        answer = RERUN[_index(mutant)] if _is_rerun(mutant) else FIRST[_index(mutant)]
        return [_row(cid, *answer, 100) for cid in ids]

    patches = _patched(tmp_path, child=child)
    with patches[0], patches[1], patches[2], patches[3], patches[4], \
            mock.patch.object(tool, "own_seeds", return_value=100), \
            pytest.raises(SystemExit, match="fixes its own seed count at 100"):
        tool.build(Cache("sweep_test"), 1)
    assert reruns == []


def test_m17a1_a_criterion_fixing_its_own_seeds_is_swept_while_no_sample_needs_a_rerun(tmp_path):
    """Review MR-12 (owner 2026-10-24): a criterion that fixes fewer seeds than the rerun count is swept, not refused,
    when every sample clears the detection limit, as `M11.C.48`'s eight did. Mutation: the refusal moved before any
    run (the build raises)."""
    def child(mutant, _tool, ids, _workers, _label):
        return [_row(cid, "PASS", 3.0, 0.3, FIRST_SEEDS) for cid in ids]

    patches = _patched(tmp_path, child=child)
    with patches[0], patches[1], patches[2], patches[3] as ran, patches[4], \
            mock.patch.object(tool, "own_seeds", return_value=100):
        text = tool.build(Cache("sweep_test"), 1)
    assert not any(_is_rerun(call.args[0]) for call in ran.call_args_list)
    for result in _results(text):
        assert result["reruns"] == 0 and result["held"] == result["counted"] == result["samples"] == 8


def test_m17a1_own_seeds_reads_the_criterions_section():
    """`M11.C.48` fixes 100 seeds, and its Phase D variant reads the same section; a criterion with no count has none."""
    assert tool.own_seeds("M11.C.48") == 100 and tool.own_seeds("M11.C.48[phase_d]") == 100
    assert tool.own_seeds("M11.C.16") is None


# --- S2.2 the fraction ------------------------------------------------------------------------------------------

def test_m17e1_s2_2_fraction_counts_only_passing_samples_and_reports_errors(tmp_path):
    """Held = PASS verdicts of the samples that gave a verdict; UNDETERMINED is counted apart and does not hold; an
    errored sample is counted, shown, and left out of the denominator (P2), and is not cached (P1)."""
    text, _ran = _build(tmp_path)
    for result in _results(text):
        assert (result["samples"], result["counted"], result["held"]) == (8, 7, 4)
        assert result["undetermined"] == 1 and result["errored"] == 1
        assert f"| `{result['criterion']}` |" in text and "| 4 of 7 |" in text
    assert "*ValueError: test*" in text
    assert tool.summarise([]) == {"samples": 0, "counted": 0, "held": 0, "undetermined": 0, "errored": 0, "reruns": 0}
    with mock.patch.object(record_cache, "CACHE_DIR", tmp_path / "cache"):
        cache = Cache("sweep_test")
    assert len(cache.entries) == 2 * (7 + len(RERUNS))  # per criterion: seven first runs gave a result, three reruns; no RAISED


# --- S2.3 the draw ----------------------------------------------------------------------------------------------

def test_s2_3_samples_stay_in_range_are_whole_and_seeded():
    """Every drawn value is inside its range, an integer constant is whole, a log range is spread on a log scale,
    and the draw is a function of the seed and the criterion key alone."""
    _rules, ranges = sweep_config.load_config()
    for key, declared in ranges.items():
        samples = tool.draw(declared, 200, 2, key)
        assert len(samples) == 200
        for sample in samples:
            for r in declared:
                v = sample[r.constant]
                assert r.low <= v <= r.high
                if r.scale == "integer":
                    assert isinstance(v, int)
        for log in (r for r in declared if r.scale == "log"):
            values = [s[log.constant] for s in samples]
            import math
            mid = math.sqrt(log.low * log.high)
            assert 60 <= sum(1 for v in values if v < mid) <= 140  # about half below the geometric midpoint
        assert tool.draw(declared, 3, 2, key) == samples[:3]
        assert tool.draw(declared, 3, 3, key) != samples[:3]
    a, b = ranges["M11.C.16"], ranges["M11.C.42"]
    assert tool.draw(a, 1, 2, "M11.C.16")[0]["learning_rate"] != tool.draw(b, 1, 2, "M11.C.42")[0]["learning_rate"]


def test_s2_3_a_longer_draw_begins_with_the_shorter_one():
    """Raising `sweep_samples` adds samples after the ones already run, so their cached results are reused."""
    _rules, ranges = sweep_config.load_config()
    declared = ranges["M11.C.16"]
    assert tool.draw(declared, 32, 2, "M11.C.16")[:8] == tool.draw(declared, 8, 2, "M11.C.16")
    linear = [sweep_config.Range("k", "learning_rate", 0.1, 0.4, "linear", None, "[I]", "t")]
    assert all(0.1 <= s["learning_rate"] <= 0.4 for s in tool.draw(linear, 50, 2, "k"))
    assert tool.format_value(6) == "6" and tool.format_value(0.23615940923795128) == "0.236159"


# --- S2.4 dirty state -------------------------------------------------------------------------------------------

def test_s2_4_a_cached_result_on_a_stale_hash_is_not_reused(tmp_path):
    """P8: results cached under an older engine hash are run again, and saving drops them; without workers a
    missing result raises instead of running."""
    _text, ran = _build(tmp_path, engine="0" * 64)
    first_calls = ran.call_count
    assert first_calls == 2 * (8 + len(RERUNS))
    with mock.patch.object(record_cache, "CACHE_DIR", tmp_path / "cache"):
        old = set(Cache("sweep_test").entries)
    _text, ran = _build(tmp_path, engine="1" * 64)
    assert ran.call_count == first_calls
    with mock.patch.object(record_cache, "CACHE_DIR", tmp_path / "cache"):
        assert old and not old & set(Cache("sweep_test").entries)
    patches = _patched(tmp_path, engine="2" * 64)
    with patches[0], patches[1], patches[2], patches[3], patches[4], pytest.raises(KeyError):
        tool.build(Cache("sweep_test"))


# --- S2.5 the renderer ------------------------------------------------------------------------------------------

def _clean_child_rows(mutant, _tool, ids, _workers, _label):
    """The stub with sample 6 giving a result, so every result is cached and a cache-only build can render."""
    if _index(mutant) == 6:
        return [_row(cid, "PASS", 2.0, 0.3, FIRST_SEEDS) for cid in ids]
    return _stub_child_rows(mutant, _tool, ids, _workers, _label)


def test_s2_5_record_renders_from_the_cache_and_a_missing_result_raises(tmp_path):
    """A second build with every result cached renders the same text without a child; the text says a fraction is
    not a verdict, prints the engine hash and the central verdict; a changed sample value is a missing result."""
    first, _ran = _build(tmp_path, child=_clean_child_rows)

    def never(*_a, **_k):
        raise AssertionError("ran a child with its result cached")

    patches = _patched(tmp_path, child=never)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        assert tool.build(Cache("sweep_test")) == first
        with mock.patch.object(tool, "format_value", lambda v: "9"), pytest.raises(KeyError):
            tool.build(Cache("sweep_test"))
    assert "measurement of a range, not a" in first and f"engine_hash: {'a' * 64}" in first
    assert "| `M11.C.16` | composite | PASS |" in first and "| `M11.C.42` | composite | FAIL |" in first
    assert "## `M11.C.16` — 5 of 8 samples held" in first and "| 5 of 8 | 1 | 3 | 0 |" in first
    verdicts = re.findall(r"\| (?:yes \(first \w+\)|no) \| (\w+) \|", first)
    assert len(verdicts) == 16 and set(verdicts) <= {"PASS", "FAIL", "UNDETERMINED"}


def test_s2_5_a_premise_is_never_swept(tmp_path):
    """`M11.5`: a config naming a premise or a check is refused before anything runs."""
    config = tmp_path / "sweep.md"
    config.write_text(TEST_CONFIG.replace("| `M11.C.42` | `learning_rate`", "| `M11.C.1` | `learning_rate`")
                      .replace("| `M11.C.42` | `triangle_activity_window`", "| `M11.C.1` | `triangle_activity_window`"),
                      encoding="utf-8")
    patches = _patched(tmp_path)
    with patches[0], patches[1], patches[2], patches[3] as ran, patches[4], pytest.raises(SystemExit, match="M11.5"):
        tool.build(Cache("sweep_test"), 1)
    assert ran.call_count == 0  # refused before any sample of any criterion ran


# --- S2.6 config ------------------------------------------------------------------------------------------------

def test_s2_6_sweep_config_is_loaded_strictly_with_grades():
    """The committed config loads; each malformed form raises: a range that does not bracket the central value, a
    high above its bound, an unknown constant, a bad scale, a non-whole integer range, a missing grade, a duplicate
    row, a wrong header, a missing rule, an unknown rule."""
    rules, ranges = sweep_config.load_config()
    assert set(rules) == set(sweep_config.RULE_NAMES) and {"M11.C.16", "M11.C.42"} <= set(ranges)
    centrals = tool.central_verdicts()
    assert all(centrals[key]["class"] in tool.DIRECTIONAL for key in ranges)  # never a premise or a check (M11.5)
    assert all(r.grade.startswith("[I]") for declared in ranges.values() for r in declared)
    assert all(r.low <= float(sweep_config.central_values()[r.constant]) <= r.high
               for declared in ranges.values() for r in declared)
    bad = [
        ("| 0.1 | 0.4 | log | — | [I] | test |\n| `M11.C.16` | `credit_horizon`", "| 0.3 | 0.4 | log | — | [I] | test |\n| `M11.C.16` | `credit_horizon`"),
        ("| 0.1 | 0.4 | log | — | [I] | test |\n| `M11.C.16` | `credit_horizon`", "| 0.1 | 0.4 | log | 0.3 | [I] | test |\n| `M11.C.16` | `credit_horizon`"),
        ("| `M11.C.16` | `learning_rate`", "| `M11.C.16` | `no_such_constant`"),
        ("| 0.1 | 0.4 | log |", "| 0.1 | 0.4 | cubic |"),
        ("| 2 | 6 | integer |", "| 2.5 | 6 | integer |"),
        ("| 2 | 6 | integer |", "| 2 | 6 | log |"),  # a log range on an int constant
        ("| 0.1 | 0.4 | log |", "| 1 | 4 | integer |"),  # an integer range on a float constant
        ("| 2 | 6 | integer | — | [I] | test |", "| 2 | 6 | integer | — | invented | test |"),
        ("| `M11.C.42` | `triangle_activity_window`", "| `M11.C.42` | `learning_rate`"),
        ("| criterion | constant |", "| key | constant |"),
        ("| `sweep_seed` | 2 | [I] | seed | test |\n", ""),
        ("| `sweep_seed` |", "| `sweep_seeds` |"),
        ("| `detection_limit_multiple` | 2.0 |", "| `detection_limit_multiple` | 0 |"),
        ("| 2 | 6 | integer | — | [I] | test |", "| 2 | 6 | integer | — | [I] | |"),
    ]
    for old, new in bad:
        assert TEST_CONFIG.count(old) >= 1, old
        broken = TEST_CONFIG.replace(old, new, 1)
        with pytest.raises(sweep_config.SweepConfigError):
            sweep_config.load_rules(broken)
            sweep_config.load_ranges(broken, sweep_config.central_values())
    central = sweep_config.central_values()
    assert central["ensemble_block"] == "50" and central["learning_rate"] == "0.2"


def test_s2_6_no_range_or_rule_literal_in_the_tool():
    """No range bound, rule value or sample count is a literal in the driver: its only number tokens are 0 and 1
    (indices and the exit status), and the config's numbers appear in neither tool's source as numbers."""
    def numbers(path: Path) -> list[str]:
        text = path.read_text(encoding="utf-8")
        return [t.string for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type == tokenize.NUMBER]

    assert set(numbers(REPO / "tools" / "version_2_sweep.py")) <= {"0", "1"}
    rules, ranges = sweep_config.load_config()
    declared = {str(v) for v in rules.values()} | {f"{x:g}" for d in ranges.values() for r in d for x in (r.low, r.high)}
    declared -= {"0", "1"}
    for path in (REPO / "tools" / "version_2_sweep.py", REPO / "tools" / "sweep_config.py"):
        assert not declared & set(numbers(path)), path.name


def test_s2_6_explainer_quotes_the_config_values():
    """`model_explainer.md` §18.7 quotes the detection-limit multiple and seed count; they must be the config's."""
    rules, _ranges = sweep_config.load_config()
    text = (REPO / "docs" / "model_explainer.md").read_text(encoding="utf-8")
    section = text.split("### 18.7", 1)[1].split("\n## ", 1)[0]
    assert f"({rules['detection_limit_multiple']})" in section and f"({rules['detection_limit_seeds']})" in section


# --- S2.7 a sample's edits --------------------------------------------------------------------------------------

def test_m17e1_s2_7_a_sample_edits_each_constant_row_once(tmp_path):
    """Applied to a copy of the config, a sample mutant changes exactly its constants' rows in ``constants.md`` and
    no other line; the stopping rows of a rerun likewise; a value is written as the strict parser reads it back."""
    _rules, ranges = sweep_config.load_config()
    central = sweep_config.central_values()
    root = tmp_path / "copy"
    shutil.copytree(REPO / "config", root / "config", ignore=shutil.ignore_patterns("__pycache__"))
    before = (REPO / tool.CONSTANTS).read_text(encoding="utf-8").splitlines()
    for key, declared in ranges.items():
        sample = tool.draw(declared, 1, 2, key)[0]
        mutant = tool.sample_mutant(key, 0, sample, central)
        assert len(mutant.definition()["edits"]) == len(declared)
        assert all(file == tool.CONSTANTS for file, _o, _n in mutant.definition()["edits"])
        apply(root, tool.rerun_mutant(mutant, 200, central))
        after = (root / tool.CONSTANTS).read_text(encoding="utf-8").splitlines()
        changed = {i for i, (a, b) in enumerate(zip(before, after)) if a != b}
        real = sum(1 for _f, old, new in mutant.definition()["edits"] if old != new)  # a draw equal to the central value edits nothing
        assert len(after) == len(before) and len(changed) == real + len(tool.STOPPING_ROWS)
        written = sweep_config.central_values(root / tool.CONSTANTS)
        for r in declared:
            assert written[r.constant] == tool.format_value(sample[r.constant])
            if r.scale == "integer":
                assert written[r.constant].isdigit()
        assert written["ensemble_block"] == "200" and written["ensemble_cap"] == "200"
        loaded = load_constants(root / tool.CONSTANTS)  # the engine's strict parser reads the sample back
        for r in declared:
            assert float(loaded[r.constant]) == float(written[r.constant])
            assert isinstance(loaded[r.constant], int) == (r.scale == "integer")
        shutil.copyfile(REPO / tool.CONSTANTS, root / tool.CONSTANTS)


# --- the child that runs a criterion ----------------------------------------------------------------------------

PHASE_D_CONFIG = TEST_CONFIG.replace("| `M11.C.42` | `learning_rate`", "| `M11.C.2` | `learning_rate`").replace(
    "| `M11.C.42` | `triangle_activity_window` | 2 | 8 | integer", "| `M11.C.2` | `credit_horizon` | 2 | 6 | integer")


def test_m17e1_phase_d_criterion_is_run_by_the_phase_d_child_under_its_hash(tmp_path):
    """A criterion of the Phase D part I record is run by ``tools/phase_d_mutant_child.py`` and cached under that
    record's hash, a Phase C criterion by ``tools/mutant_runner.py`` under the engine hash (the part I sweep's first
    run raised ``KeyError`` on every sample: Phase C's child knows only Phase C's criteria). A change to the Phase D
    hash reruns the Phase D samples and none of Phase C's (P8)."""
    (tmp_path / "sweep.md").write_text(PHASE_D_CONFIG, encoding="utf-8")
    ran: list[tuple[str, str]] = []

    def child(mutant, child_tool, ids, workers, label):
        ran.extend((child_tool, cid) for cid in ids)
        return _clean_child_rows(mutant, child_tool, ids, workers, label)

    def build(phase_d_engine):
        with mock.patch.object(tool, "phase_d_engine_hash", return_value=phase_d_engine):
            _build(tmp_path, child=child)

    build("d" * 64)
    assert {c for t, c in ran if t == "phase_d_mutant_child.py"} == {"M11.C.2"}
    assert {c for t, c in ran if t == "mutant_runner.py"} == {"M11.C.16"}
    ran.clear()
    build("e" * 64)
    assert ran and {c for _t, c in ran} == {"M11.C.2"}


def test_m17e1_another_child_without_its_hash_is_refused(tmp_path):
    """``run_cached`` keys a result by the hash it is given; a child other than Phase C's with no hash of its own
    would be keyed by ``engine_hash``, which does not cover that child, so a change to it would reuse stale results
    (P8). The call is refused before anything runs (review, 2026-10-23)."""
    with mock.patch.object(record_cache, "CACHE_DIR", tmp_path / "cache"), \
            mock.patch.object(record_cache, "child_rows", side_effect=AssertionError("ran a child")):
        with pytest.raises(ValueError, match="hash"):
            record_cache.run_cached(Cache("sweep_test"), None, ["M11.C.2"], 1, child="phase_d_mutant_child.py")


# --- S2.5 and S2.6 a criterion declared and not run ----------------------------------------------------------------

NOT_RUN = """
## Declared, not run

| criterion | why |
|---|---|
| `M11.C.42` | test: declared only |
"""


def test_m17e1_s2_5_a_declared_not_run_key_is_shown_and_never_run(tmp_path):
    """A key listed under 'Declared, not run' is in the record as that, with its ranges and no number, and none of
    its samples runs; the other key runs as before."""
    (tmp_path / "sweep.md").write_text(TEST_CONFIG + NOT_RUN, encoding="utf-8")
    text, ran = _build(tmp_path)
    assert ran.call_count and all(c.args[2] == ["M11.C.16"] for c in ran.call_args_list)
    assert "## `M11.C.42` — declared, not run" in text and "Not run: test: declared only" in text
    assert "; `triangle_activity_window` 2–8 | declared, not run | — | — | — |" in text
    shown = {r["criterion"]: r for r in _results(text)}
    assert shown["M11.C.42"]["rows"] == [] and "held" not in shown["M11.C.42"] and shown["M11.C.16"]["counted"]


def test_m17e1_s2_5_a_listed_key_with_a_cached_result_is_refused(tmp_path):
    """P8: once a listed key has a result in the cache (a share run with ``only``), the record is refused until its
    row is taken off the list, so a key that has run is never shown as not run."""
    (tmp_path / "sweep.md").write_text(TEST_CONFIG + NOT_RUN, encoding="utf-8")
    patches = _patched(tmp_path)
    with patches[0], patches[1], patches[2], patches[3] as ran, patches[4]:
        cache = Cache("sweep_test")
        assert [r["criterion"] for r in tool.sweep(cache, 1, ["M11.C.42"])] == ["M11.C.42"]  # a share runs it
        assert ran.call_count
        with pytest.raises(SystemExit, match="Declared, not run"):
            tool.build(cache, 1)
        with pytest.raises(SystemExit, match="Declared, not run"):
            tool.build(cache)


def test_s2_6_not_run_list_is_loaded_strictly():
    """The committed list names declared keys only; a listed key without ranges, a repeat, an empty reason or a
    wrong header or a misspelt heading raises; a config without the section lists nothing."""
    _rules, ranges = sweep_config.load_config()
    assert set(sweep_config.not_run_keys()) <= set(ranges)
    test_ranges = sweep_config.load_ranges(TEST_CONFIG, sweep_config.central_values())
    assert sweep_config.load_not_run(TEST_CONFIG, test_ranges) == {}
    assert sweep_config.load_not_run(TEST_CONFIG + NOT_RUN, test_ranges) == {"M11.C.42": "test: declared only"}
    for broken in (NOT_RUN.replace("M11.C.42", "M11.C.10"), NOT_RUN + "| `M11.C.42` | again |\n",
                   NOT_RUN.replace("test: declared only", ""), NOT_RUN.replace("| criterion | why |", "| key | why |"),
                   NOT_RUN.replace("## Declared, not run", "## Declared, not yet run")):
        with pytest.raises(sweep_config.SweepConfigError):
            sweep_config.load_not_run(TEST_CONFIG + broken, test_ranges)


# --- S2.11 the committed record ---------------------------------------------------------------------------------

def test_s2_11_sweep_record_is_current():
    """The committed sweep record is what the tool renders from the committed cache for the current engine and
    config, without running anything (P6); a stale cache or a changed range would raise here."""
    with mock.patch.object(record_cache, "child_rows", side_effect=AssertionError("ran a child in a test")):
        assert tool.RECORD.read_text(encoding="utf-8") == tool.build(Cache(tool.CACHE_NAME))


def test_m17e1_a_known_defect_is_printed_beside_the_central_verdict():
    """Milestone review MR-3: the sweep record showed `M11.C.48`'s central PASS unqualified. A key of the Phase D
    record tool's ``VERDICT_NOTES`` is printed with its note in the summary table's central-verdict cell and under its
    own heading; the machine-readable block is unchanged. Mutation: the renderer does not read the notes."""
    from tools.phase_d_record import VERDICT_NOTES

    with mock.patch.object(record_cache, "child_rows", side_effect=AssertionError("ran a child in a test")):
        text = tool.build(Cache(tool.CACHE_NAME))
    for key, note in VERDICT_NOTES.items():
        row = next(line for line in text.splitlines() if line.startswith(f"| `{key}` |"))
        assert f"| PASS ({note}) |" in row, row
        heading = text.split(f"## `{key}` — ", 1)[1].split("\n## ", 1)[0]
        assert f"Central verdict PASS: {note}." in heading
    assert text.count("(T-119)") == 2 * len(VERDICT_NOTES)
