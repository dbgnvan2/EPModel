"""The emergence audit (`M11.1g`; `docs/plan_v3_emergence_audit.md` §6, EA-1 to EA-16).

Purpose: the audit's scope, edits, verdicts, controls, carriers, cache key and record are what the plan says. EA-3 to
         EA-7 compare two arms, an ablation against the unmodified model, in a temporary copy, and assert a direction;
         the rest are tool checks on fixtures, like the other record tests. EA-16 is `tests/test_doc_checklist.py`.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.1g; docs/plan_v3_emergence_audit.md#6
Tests:   this file
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import emergence_audit as ea  # noqa: E402
from tools import record_cache  # noqa: E402
from tools.record_cache import Cache, result_key  # noqa: E402
from tools.record_json import record_json  # noqa: E402

ENGINE = "fixture-engine"


def readout(name="r", diff=1.0, half=0.1, direction=1, p=1e-6, **extra):
    return {"readout": name, "direction": direction, "mean_difference": diff, "half_width": half,
            "report_only": False, "p": p, **extra}


def result(key, outcome="PASS", readouts=None, seeds=50, **extra):
    return {"criterion": key, "outcome": outcome, "seeds": seeds,
            "readouts": [readout()] if readouts is None else readouts, "missing_moves": [], **extra}


def record_row(key, cls, outcome="PASS", seeds=50):
    return {"criterion": key, "class": cls, "outcome": outcome, "seeds": seeds, "readouts": []}


def config(stage="full", pilot=None, exclusions=None, controls=None):
    controls = (("positive", "X.39", "L0", "red", "learning only"),
                ("negative", "X.3", "L0", "survived", "physics")) if controls is None else controls
    return ea.Config({"X.39": "learning only"} if exclusions is None else exclusions, controls, stage, pilot or {})


def records(extra_c=(), extra_d=()):
    return {"phase_c": [record_row("X.3", "premise"), record_row("X.19", "check"), record_row("X.5", "composite", "FAIL"),
                        record_row("X.16", "composite"), *extra_c],
            "phase_d": [record_row("X.39", "composite"), record_row("X.41", "mixed"), *extra_d],
            "phase_d2": []}


class FakeCache:
    """A cache whose results are given by (mutant id, criterion); every key asked for is kept."""

    def __init__(self, given: dict, mutants):
        self.entries = {result_key(ENGINE, m.definition(), c): r for m in mutants for (mid, c), r in given.items()
                        if mid == m.id}
        self.asked = []

    def get(self, key):
        self.asked.append(key)
        return self.entries.get(key)


def joints(key, phase="phase_c"):
    return [m for _, m in ea.carrier_candidates(ea.Row(key, phase, "composite", 50))[0]]


def full_cache(overrides=None, rows=("X.16", "X.41"), controls=True):
    """Every ablation on every fixture row passes unless ``overrides`` says otherwise; L0's controls hold."""
    given = {}
    for a in ea.ABLATIONS:
        for k in rows:
            given[(a.id, k)] = result(k)
    if controls:
        given[("L0", "X.39")] = result("X.39", "FAIL", [readout(diff=0.0, half=0.1, p=0.9)])
        given[("L0", "X.3")] = result("X.3")
    mutants = list(ea.ABLATIONS)
    for k in rows:
        for m in joints(k):
            given[(m.id, k)] = result(k)
            mutants.append(m)
    given.update(overrides or {})
    return FakeCache(given, mutants)


def build(cache, **kw):
    kw = {"records": records(), "config": config(), "limbs": {}, "engine": ENGINE, "alpha": 0.05, "agreement": False,
          **kw}
    return ea.build(cache, **kw)


# --- EA-1 ------------------------------------------------------------------------------------------------------------
def test_m111g_ea1_scope_is_passing_composite_and_mixed():
    audited, _, by_class, excluded = ea.in_scope(records(), config())
    assert [r.key for r in audited] == ["X.16", "X.41"]  # not the premise, the check, the failing composite
    assert sorted(k for k, _ in by_class) == ["X.19", "X.3"]
    assert excluded == [("X.39", "learning only")]


def test_m111g_ea1_the_real_scope_is_the_plans():
    """On the committed records the scope is plan §3's thirteen rows less `M11.C.41`'s two level cells (T-192), with
    `M11.C.39` excluded as the control, and the two cue rows that pass since spec revision 55, `M11.C.52` and
    `M11.C.55` (Claude's recommendation 2026-11-11, pending the owner's review: `M11.1g` audits every passing composite
    row, so they enter by its rule)."""
    audited, rows, by_class, excluded = ea.in_scope(ea.read_records(), ea.read_config())
    ea.check_registries(rows)
    keys = {r.key for r in audited}
    assert len(audited) == 13 and {"M11.C.48[phase_d]", "M11.C.52", "M11.C.55"} <= keys
    assert [k for k, _ in excluded] == ["M11.C.39"]
    # T-192 (owner 2026-11-05): M11.C.41's two level cells have only premise limbs, so they leave the scope.
    level_cells = ["M11.C.41[light: lower level]", "M11.C.41[heavy: lower level]"]
    assert [(k, c) for k, c in by_class if k in level_cells] == [(k, ea.NO_COMPOSITE) for k in level_cells]


def test_t192_a_mixed_row_with_only_premise_limbs_leaves_scope():
    """A passing mixed row each of whose gating readouts is a declared premise limb is left out by class; one with a
    composite limb left, with no declared limb, or with no readouts in its record stays in scope."""
    def mixed(key, *names, report_only=()):
        row = record_row(key, "mixed")
        row["readouts"] = [{"readout": n} for n in names] + [{"readout": n, "report_only": True} for n in report_only]
        return row

    recs = records(extra_d=[mixed("X.50", "a", "b", report_only=("c",)), mixed("X.51[v]", "a", "b"),
                            mixed("X.52", "a")])
    limbs = {"X.50": frozenset({"a", "b"}), "X.51": frozenset({"a"}), "X.41": frozenset({"a"})}
    audited, _, by_class, _ = ea.in_scope(recs, config(), limbs)
    assert ("X.50", ea.NO_COMPOSITE) in by_class
    assert {"X.41", "X.51[v]", "X.52"} <= {r.key for r in audited} and "X.50" not in {r.key for r in audited}
    assert ("X.50", ea.NO_COMPOSITE) not in ea.in_scope(recs, config(), {})[2]


# --- EA-2 ------------------------------------------------------------------------------------------------------------
def test_m111g_ea2_every_ablation_applies_exactly_once(tmp_path):
    audited, _, _, _ = ea.in_scope(ea.read_records(), ea.read_config())
    every = [*ea.ABLATIONS, *ea.CARRIERS]
    for row in audited:
        every += [m for _, m in ea.carrier_candidates(row)[0]]
    for i, mutant in enumerate(every):
        root = tmp_path / str(i)
        for file in {f for f, _, _ in ea.edits(mutant)}:
            (root / file).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / file, root / file)
        ea.apply(root, mutant)  # raises SystemExit unless every replacement occurs exactly once
        for file, _, new in ea.edits(mutant):
            assert new in (root / file).read_text(encoding="utf-8"), mutant.id


# --- EA-3 to EA-7: an ablation against the unmodified model, in a copy -----------------------------------------------
@pytest.fixture(scope="module")
def unmodified():
    return {name: ea.run_probe(None, name) for name in ("learned", "own", "cross", "witnessed", "soothing")}


def test_m111g_ea3_l0_leaves_fewer_learned_values(unmodified):
    assert unmodified["learned"] > 0
    assert ea.run_probe(ea.L0, "learned") < unmodified["learned"]


def test_m111g_ea4_l1_credits_less_own_relief(unmodified):
    assert abs(ea.run_probe(ea.L1, "own")) < abs(unmodified["own"])


def test_m111g_ea5_l2_credits_less_cross_person_relief(unmodified):
    assert abs(ea.run_probe(ea.L2, "cross")) < abs(unmodified["cross"])


def test_m111g_ea6_l3_fewer_witnessed_cutoff_updates(unmodified):
    assert ea.run_probe(ea.L3, "witnessed") < unmodified["witnessed"]


def test_m111g_ea7_l4_fewer_soothing_reviews(unmodified):
    assert ea.run_probe(ea.L4, "soothing") < unmodified["soothing"]


# --- EA-8 ------------------------------------------------------------------------------------------------------------
def test_m111g_ea8_controls_hold_in_the_record():
    data = record_json(ea.RECORD)
    by = {c["control"]: c for c in data["controls"]}
    assert by["positive"]["criterion"] == "M11.C.39" and by["positive"]["verdict"] in ea.RED
    assert by["negative"]["criterion"] == "M11.C.3" and by["negative"]["verdict"] == "survived"
    assert all(c["held"] for c in data["controls"])


def test_m111g_ea8_build_refuses_a_failed_control():
    with pytest.raises(SystemExit, match="positive X.39"):
        build(full_cache({("L0", "X.39"): result("X.39")}))  # learning off in both arms and still passing
    with pytest.raises(SystemExit, match="negative X.3"):
        build(full_cache({("L0", "X.3"): result("X.3", "FAIL", [readout(diff=0.0, p=0.9)])}))


def test_m111g_ea8_an_errored_control_is_not_blamed_on_the_ablation():
    with pytest.raises(SystemExit, match="was not run"):
        build(full_cache({("L0", "X.39"): result("X.39", record_cache.FAILED, [])}))
    with pytest.raises(SystemExit, match="positive X.39"):  # an engine exception is red but is not the positive control's FAIL
        build(full_cache({("L0", "X.39"): result("X.39", "RAISED", [])}))


def test_m111g_ea8_agreement_disagreement_raises(tmp_path):
    from tools import phase_d_mutation_record as d_tool

    record = tmp_path / "phase_d_mutation_record.md"
    record.write_text("engine_hash: fixture-d\n", encoding="utf-8")
    entries = [{"criterion": "M11.C.39", "ablation": "L0", "outcome": "FAIL", "seeds": 50}]
    twin = {"mutant": "learning-off-in-both-arms", "criterion": "M11.C.39", "outcome": "FAIL", "seeds": 50}
    with mock.patch.object(ea, "engine_hash", return_value=ENGINE), \
            mock.patch.object(d_tool, "engine_hash", return_value="fixture-d"), \
            mock.patch.object(d_tool, "RECORD", record):
        with mock.patch.object(ea, "record_json", return_value=[twin]):
            assert any("agrees" in line for line in ea.check_agreement(entries, ENGINE))
        with mock.patch.object(ea, "record_json", return_value=[{**twin, "outcome": "PASS", "seeds": 200}]):
            with pytest.raises(SystemExit, match="the same edits"):
                ea.check_agreement(entries, ENGINE)
        drifted = [dataclasses.replace(m, new="| `learning_rate` | 0.1 |") if m.id == "learner-off-share-kept" else m
                   for m in d_tool.MUTANTS]
        with mock.patch.object(d_tool, "MUTANTS", drifted):
            with pytest.raises(SystemExit, match="L0 twins"):
                ea.check_agreement(entries, ENGINE)


# --- EA-9 ------------------------------------------------------------------------------------------------------------
def test_m111g_ea9_verdicts_follow_judge():
    row = ea.Row("X.16", "phase_c", "composite", 50)

    def v(res):
        return ea.verdict(row, ea.L0, res, {}, 0.05)[0]

    assert v(result("X.16")) == "survived"
    assert v(result("X.16", "FAIL", [readout(diff=-1.0, half=0.1, p=0.9)])) == "reversed"
    assert v(result("X.16", "FAIL", [readout(diff=-0.05, half=0.1, p=0.9)])) == "red"
    assert v(result("X.16", "UNDETERMINED", [readout(diff=0.05, half=0.2, p=0.3)])) == "red"
    assert v(result("X.16", "RAISED", [])) == "red"
    assert v(result("X.16", "BROKEN", [])) == "broken"
    assert v(result("X.16", record_cache.FAILED, [])) == "not run (child failed)"


def test_m111g_ea9_broken_and_not_run_are_never_cached(tmp_path):
    cache = Cache("emergence_audit_fixture")
    cache.path, cache.entries = tmp_path / "c.json", {}
    rows = [result("X.16", "BROKEN", []), result("X.17", record_cache.FAILED, []), result("X.18")]
    with mock.patch.object(record_cache, "child_rows", return_value=rows):
        record_cache.run_cached(cache, ea.L0, ["X.16", "X.17", "X.18"], 1, child="phase_d_mutant_child.py",
                                engine=ENGINE)
    assert [r["criterion"] for r in cache.entries.values()] == ["X.18"]
    record_cache._uncached.clear()  # the errored rows this test made are not this process's run's


# --- EA-10 -----------------------------------------------------------------------------------------------------------
def test_m111g_ea10_premise_limb_does_not_decide_a_mixed_row():
    row = ea.Row("X.41", "phase_d", "mixed", 50)
    limbs = {"X.41": frozenset({"premise_limb"})}
    premise_fails = result("X.41", "FAIL", [readout("composite_limb", p=1e-8), readout("premise_limb", diff=0.0, p=0.8)])
    got, note = ea.verdict(row, ea.L0, premise_fails, limbs, 0.05)
    assert got == "survived" and "premise_limb" in note
    composite_fails = result("X.41", "FAIL", [readout("composite_limb", diff=0.0, p=0.8), readout("premise_limb", p=1e-8)])
    assert ea.verdict(row, ea.L0, composite_fails, limbs, 0.05)[0] == "red"
    missing_move = {**premise_fails, "missing_moves": ["CUTOFF"]}
    assert ea.verdict(row, ea.L0, missing_move, limbs, 0.05)[0] == "red"


def test_m111g_ea10_a_mixed_row_without_declared_premise_limbs_says_so():
    data = record_json(build(full_cache()))
    x41 = [v for v in data["verdicts"] if v["criterion"] == "X.41"]
    assert x41 and all("premise limbs not declared" in v["note"] for v in x41)
    assert all("premise limbs not declared" not in v["note"] for v in data["verdicts"] if v["criterion"] == "X.16")


def test_m111g_ea10_a_phase_c_row_with_premise_limbs_is_refused():
    with pytest.raises(SystemExit, match="Phase C mixed row"):
        build(full_cache(rows=("X.16", "X.41", "X.42")), records=records(extra_c=[record_row("X.42", "mixed")]),
              limbs={"X.42": frozenset({"premise_limb"})})


def test_m111g_ea10_an_equivalence_limb_without_within_bound_raises():
    row = ea.Row("X.41", "phase_d", "mixed", 50)
    res = result("X.41", "FAIL", [readout("eq", direction=0), readout("premise_limb", diff=0.0, p=0.8)])
    with pytest.raises(SystemExit, match="within_bound"):
        ea.verdict(row, ea.L0, res, {"X.41": frozenset({"premise_limb"})}, 0.05)


# --- EA-11 -----------------------------------------------------------------------------------------------------------
def test_m111g_ea11_carriers_named_only_for_survivors():
    red_l0 = {("L0", "X.41"): result("X.41", "FAIL", [readout(diff=0.0, p=0.9)])}
    c1 = next(m for m in joints("X.16") if m.id == "L0+C1")
    cache = full_cache({**red_l0, (c1.id, "X.16"): result("X.16", "FAIL", [readout(diff=0.0, p=0.9)])})
    data = record_json(build(cache))
    assert [c["criterion"] for c in data["carriers"]] == ["X.16"]  # X.41 is red under L0: no carriers for it
    assert data["carriers"][0]["carriers"] == ["C1"]
    text = build(full_cache(red_l0))
    assert f"no carrier among {len(joints('X.16'))} candidates" in text


def test_m111g_ea11_an_exception_is_not_named_a_carrier():
    c1 = next(m for m in joints("X.16") if m.id == "L0+C1")
    red_l0 = {("L0", "X.41"): result("X.41", "FAIL", [readout(diff=0.0, p=0.9)])}
    data = record_json(text := build(full_cache({**red_l0, (c1.id, "X.16"): result("X.16", "RAISED", [])})))
    assert data["carriers"][0]["carriers"] == [] and data["carriers"][0]["red_without_fail"] == ["C1 (RAISED)"]
    assert "not named a carrier: C1 (RAISED)" in text


# --- EA-12 -----------------------------------------------------------------------------------------------------------
def test_m111g_ea12_no_row_dropped_silently():
    data = record_json(text := build(full_cache()))
    pairs = {(v["criterion"], v["ablation"]) for v in data["verdicts"]}
    assert pairs == {(k, a.id) for k in ("X.16", "X.41") for a in ea.ABLATIONS}
    assert "2 of 5 passing rows audited; 1 excluded; 2 out by class" in text
    assert "- `X.39`: learning only" in text
    pilot = build(full_cache(), config=config("pilot", {"X.16": "short"}))
    assert "- `X.41`: not in the pilot; runs at the full stage" in pilot
    with pytest.raises(SystemExit, match="not in scope"):
        build(full_cache(), config=config("pilot", {"X.5": "failing"}))
    with pytest.raises(SystemExit, match="not passing composite or mixed"):
        build(full_cache(), config=config(exclusions={"X.3": "a premise"}))


def test_m111g_ea12_the_config_is_parsed_strictly():
    text = ea.CONFIG.read_text(encoding="utf-8")
    assert ea.read_config(text).stage in ea.STAGES
    for stage in ea.STAGES:
        assert ea.read_config(text.replace("| stage | pilot |", f"| stage | {stage} |")
                              .replace("| stage | l0 |", f"| stage | {stage} |")
                              .replace("| stage | carriers |", f"| stage | {stage} |")
                              .replace("| stage | full |", f"| stage | {stage} |")).stage == stage
    with pytest.raises(SystemExit, match="unknown ablation"):
        ea.read_config(text.replace("| positive | `M11.C.39` | L0 |", "| positive | `M11.C.39` | L9 |"))
    with pytest.raises(SystemExit, match="not one of pilot, l0, carriers, full"):
        ea.read_config(text.replace("| stage | pilot |", "| stage | some |").replace("| stage | l0 |", "| stage | some |")
                       .replace("| stage | carriers |", "| stage | some |").replace("| stage | full |", "| stage | some |"))
    with pytest.raises(SystemExit, match="cells"):
        ea.read_config(text.replace("| `M11.C.39` | its treatment arm", "| `M11.C.39` | its | treatment arm"))
    with pytest.raises(SystemExit, match="twice"):
        ea.read_config(text.replace("| `M11.C.48[phase_d]` |", "| `M11.C.27[unstable,remove_one]` |"))


def test_m111g_ea12_l0_stage_runs_l0_on_every_row_and_nothing_else():
    """Plan §7 step 4 (owner 2026-11-03): the l0 stage needs L0's results on every in-scope row and no others; L1 to
    L4 and the carriers are not asked for, and the record says so."""
    cache = full_cache()
    l0_keys = {result_key(ENGINE, ea.L0.definition(), k) for k in ("X.16", "X.41", "X.39", "X.3")}
    cache.entries = {k: v for k, v in cache.entries.items() if k in l0_keys}
    data = record_json(text := build(cache, config=config("l0")))
    assert {(v["criterion"], v["ablation"]) for v in data["verdicts"]} == {("X.16", "L0"), ("X.41", "L0")}
    assert data["carriers"] == [] and data["stage"] == "l0"
    assert set(cache.asked) <= l0_keys
    assert "- L1 to L4 on every row: run at the full stage (plan §7 step 5)" in text
    assert "2 of 2 rows survived L0: `X.16`, `X.41`." in text
    red = {("L0", "X.41"): result("X.41", "FAIL", [readout(diff=0.0, p=0.9)])}
    assert "1 of 2 rows survived L0: `X.16`." in build(full_cache(red), config=config("l0"))


# --- EA-13 -----------------------------------------------------------------------------------------------------------
def test_m111g_ea13_key_is_engine_and_edits():
    row = ea.Row("X.16", "phase_c", "composite", 50)
    cache = FakeCache({("L0", "X.16"): result("X.16")}, [ea.L0])
    relabelled = dataclasses.replace(ea.L0, what="a new label", id="L0")
    assert ea._results(cache, relabelled, [row], None, ENGINE)["X.16"]["outcome"] == "PASS"  # a label reruns nothing
    edited = dataclasses.replace(ea.L0, new="| `learning_rate` | 0.1 |")
    with pytest.raises(KeyError):
        ea._results(cache, edited, [row], None, ENGINE)  # an edit does
    with pytest.raises(KeyError):
        ea._results(cache, ea.L0, [row], None, "another-engine")  # and so does the engine


# --- EA-14 -----------------------------------------------------------------------------------------------------------
def test_m111g_ea14_emergence_audit_record_is_current():
    try:
        rendered = ea.build(Cache(ea.CACHE))
    except KeyError as missing:
        raise AssertionError(f"stale: rerun python3 tools/emergence_audit.py ({missing})") from None
    assert rendered == ea.RECORD.read_text(encoding="utf-8"), "stale or edited: rerun python3 tools/emergence_audit.py"


def test_m111g_ea14_a_missing_result_raises_keyerror():
    cache = full_cache()
    cache.entries.pop(result_key(ENGINE, ea.L2.definition(), "X.41"))
    with pytest.raises(KeyError):
        build(cache)


# --- EA-15 -----------------------------------------------------------------------------------------------------------
def test_m111g_ea15_class_shown_premise_never_audited():
    data = record_json(text := build(full_cache()))
    assert {v["class"] for v in data["verdicts"]} <= set(ea.AUDITED)
    assert "X.3" not in {v["criterion"] for v in data["verdicts"]}
    assert "| `X.41` | mixed | L0 |" in text and "| `X.16` | composite | L0 |" in text
    committed = record_json(ea.RECORD)
    assert {v["class"] for v in committed["verdicts"]} <= set(ea.AUDITED)


def test_m111g_ea12_carriers_stage_runs_l0_and_carriers_and_no_l1_to_l4():
    """Owner 2026-11-04: the carriers stage needs L0 on every in-scope row and the controls, and the carriers of each
    L0 survivor; L1 to L4 are not asked for, and the record says so."""
    allowed = {result_key(ENGINE, ea.L0.definition(), k) for k in ("X.16", "X.41", "X.39", "X.3")}
    allowed |= {result_key(ENGINE, m.definition(), k) for k in ("X.16", "X.41") for m in joints(k)}
    red = {("L0", "X.41"): result("X.41", "FAIL", [readout(diff=0.0, p=0.9)])}
    for overrides, survivors in (({}, ["X.16", "X.41"]), (red, ["X.16"])):
        cache = full_cache(overrides)
        cache.entries = {k: v for k, v in cache.entries.items() if k in allowed}
        data = record_json(text := build(cache, config=config("carriers")))
        assert {(v["criterion"], v["ablation"]) for v in data["verdicts"]} == {("X.16", "L0"), ("X.41", "L0")}
        assert [c["criterion"] for c in data["carriers"]] == survivors and data["stage"] == "carriers"
        assert all(len(c["candidates"]) == len(joints(c["criterion"])) for c in data["carriers"])
        assert set(cache.asked) <= allowed
        assert "- L1 to L4 on every row: run at the full stage (plan §7 step 5)" in text
        assert ea.STAGE_TEXT["carriers"] in text and "### `X.16`" in text


# --- T-184: the cache's NumPy version, the merge's running check, one machine's share -------------------------------
class _Share:
    """``main`` on the fixture records at the carriers stage, with a real cache file under ``tmp_path`` and a stub child
    that passes every row; ``ran`` is each (mutant id, criteria) the stub was asked to run."""

    def __init__(self, tmp_path, stage="carriers", pin=None, given=None):
        self.tmp_path, self.stage, self.pin, self.ran, self.given = tmp_path, stage, pin, [], given or {}

    def fake_run_cached(self, cache, mutant, ids, workers, label, child, engine):
        self.ran.append((mutant.id, tuple(ids)))
        out = [self.given.get((mutant.id, cid), result(cid)) for cid in ids]
        for r in out:
            cache.put(result_key(engine, mutant.definition(), r["criterion"]), r)
        cache.flush()
        return out

    def __enter__(self):
        import numpy as np

        from tools import version_2_sweep

        self.patches = [mock.patch.object(record_cache, "CACHE_DIR", self.tmp_path),
                        mock.patch.object(ea, "read_records", return_value=records()),
                        mock.patch.object(ea, "read_config", return_value=config(self.stage)),
                        mock.patch("tools.version_2_status.premise_limbs", return_value={}),
                        mock.patch.object(ea, "engine_hash", return_value=ENGINE),
                        mock.patch.object(ea, "_alpha", return_value=0.05),
                        mock.patch.object(ea, "check_agreement", return_value=[]),
                        mock.patch.object(ea, "check_registries"),
                        mock.patch.object(ea, "RECORD", self.tmp_path / "record.md"),
                        mock.patch.object(ea, "run_cached", side_effect=self.fake_run_cached),
                        mock.patch.object(version_2_sweep, "sweep_numpy_pin", return_value=self.pin or np.__version__)]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()

    def file(self):
        return record_cache.read_cache_file(self.tmp_path / f"{ea.CACHE}.json")


def _seed_cache(tmp_path, meta):
    other = {"a" * 64: result("Y.1"), "b" * 64: result("Y.2")}
    record_cache.write_cache_file(tmp_path / f"{ea.CACHE}.json", other, meta)
    return other


def test_t184_a_share_runs_only_its_rows_and_keeps_the_others(tmp_path):
    """A share runs L0 and the carriers of its own rows only, and flushes: every other cached result is kept (step 4's
    subset run kept only the keys it read, and dropped the pilot's five)."""
    import numpy as np

    other = _seed_cache(tmp_path, {"numpy": np.__version__})
    with _Share(tmp_path) as s:
        assert ea.main(["--criteria", "X.16", "--workers", "1"]) == 0
        entries, meta = s.file()
    assert s.ran[0] == ("L0", ("X.16",)) and {ids for _, ids in s.ran} == {("X.16",)}
    assert {mid for mid, _ in s.ran} == {"L0", *(m.id for m in joints("X.16"))}  # no L1 to L4 at this stage
    assert all(entries[k] == v for k, v in other.items())  # nothing pruned
    assert len(entries) == len(other) + 1 + len(joints("X.16")) and meta == {"numpy": np.__version__}
    with _Share(tmp_path) as s:
        ea.main(["--criteria", "X.41", "--candidates", "C1", "C3", "--workers", "1"])
        entries, _ = s.file()
    assert [mid for mid, _ in s.ran] == ["L0", "L0+C1", "L0+C3"]  # one row's carriers split by label
    assert all(entries[k] == v for k, v in other.items())


def test_t184_a_share_refuses_an_unknown_row_or_label_before_running(tmp_path):
    import numpy as np

    _seed_cache(tmp_path, {"numpy": np.__version__})
    for argv, match in ((["--criteria", "X.16", "X.99"], "X.99"), (["--criteria", "X.39"], "X.39"),
                        (["--criteria", "X.16", "--candidates", "C9"], "C9")):
        with _Share(tmp_path) as s, pytest.raises(SystemExit, match=match):
            ea.main([*argv, "--workers", "1"])
        assert s.ran == []
    with _Share(tmp_path, stage="l0") as s, pytest.raises(SystemExit, match="runs none"):
        ea.main(["--criteria", "X.16", "--candidates", "C1", "--workers", "1"])
    assert s.ran == []


def test_t184_a_run_records_numpy_and_refuses_another(tmp_path):
    """The audit's cache records the NumPy version as the sweep's does; a run under another one, or on a cache of
    unknown origin, is refused before anything runs. The committed cache records the sweep machines' pin."""
    import numpy as np

    from tools import version_2_sweep

    with _Share(tmp_path) as s:
        ea.main(["--criteria", "X.16", "--workers", "1"])
        assert s.file()[1] == {"numpy": np.__version__}
    for meta, pin, match in (({"numpy": "0.0.2"}, None, "0.0.2"), ({}, None, "records no NumPy"),
                             ({"numpy": np.__version__}, "0.0.1", "0.0.1")):
        _seed_cache(tmp_path, meta)
        for argv in (["--criteria", "X.16"], []):
            with _Share(tmp_path, pin=pin) as s, pytest.raises(SystemExit, match=match):
                ea.main([*argv, "--workers", "1"])
            assert s.ran == []
    assert Cache(ea.CACHE).meta == {"numpy": version_2_sweep.sweep_numpy_pin()}


def test_t184_merge_sees_a_running_audit():
    """``local_sweep_running`` looks for the audit as well as the sweep, and its pattern never matches pgrep itself."""
    import re
    import subprocess

    from tools import merge_cache

    with mock.patch.object(merge_cache.subprocess, "run",
                           return_value=subprocess.CompletedProcess([], 1, b"", b"")) as run:
        merge_cache.local_sweep_running()
    pattern = run.call_args.args[0][-1]
    for line in ("python3 tools/emergence_audit.py --criteria M11.C.16 --workers 8",
                 ".venv/bin/python tools/version_2_sweep.py --criteria M11.C.42"):
        assert re.search(pattern, line), line
    for line in ("python3 tools/merge_cache.py other.json", f"pgrep -f {pattern}",
                 "python3 -m pytest tests/test_emergence_audit.py -q", "python3 -m pytest tests/test_version_2_sweep.py"):
        assert not re.search(pattern, line), line


def test_t184_a_share_runs_no_carriers_for_a_row_red_under_l0(tmp_path, capsys):
    """A share runs a row's carriers only when it survives L0, as a whole run does; a key named twice runs once."""
    import numpy as np

    _seed_cache(tmp_path, {"numpy": np.__version__})
    red = {("L0", "X.41"): result("X.41", "FAIL", [readout(diff=0.0, p=0.9)])}
    with _Share(tmp_path, given=red) as s:
        ea.main(["--criteria", "X.41", "X.16", "X.41", "--workers", "1"])
    l0 = [ids for mid, ids in s.ran if mid == "L0"]
    assert sorted(l0) == [("X.16",), ("X.41",)]  # one child each; X.41, named twice, asked once
    assert {ids for mid, ids in s.ran if mid != "L0"} == {("X.16",)}  # no carrier of X.41
    assert "X.41: red under L0, so no carriers run" in capsys.readouterr().out


def test_t184_render_only_runs_nothing_and_refuses_a_missing_result(tmp_path):
    """``--render-only`` renders the whole record from the cache with no run and no NumPy check, writes no cache, and
    stops with a message, writing no record, when a result is missing."""
    cache = full_cache()
    l1 = {result_key(ENGINE, m.definition(), k) for m in ea.ABLATIONS[1:] for k in ("X.16", "X.41")}
    entries = {k: v for k, v in cache.entries.items() if k not in l1}  # the carriers stage asks for no L1-L4
    record_cache.write_cache_file(tmp_path / f"{ea.CACHE}.json", entries, {})  # no NumPy recorded: not checked
    before = (tmp_path / f"{ea.CACHE}.json").read_bytes()
    with _Share(tmp_path, pin="0.0.1") as s:
        assert ea.main(["--render-only"]) == 0
        rendered = (tmp_path / "record.md").read_text(encoding="utf-8")
        fake = FakeCache({}, [])
        fake.entries = record_cache.read_cache_file(tmp_path / f"{ea.CACHE}.json")[0]  # as reloaded: sorted keys
        expected = build(fake, config=config("carriers"))
    assert s.ran == [] and rendered == expected and (tmp_path / f"{ea.CACHE}.json").read_bytes() == before
    (tmp_path / "record.md").unlink()
    record_cache.write_cache_file(tmp_path / f"{ea.CACHE}.json",
                                  {k: v for k, v in entries.items()
                                   if k != result_key(ENGINE, ea.L0.definition(), "X.16")}, {})
    before = (tmp_path / f"{ea.CACHE}.json").read_bytes()
    with _Share(tmp_path) as s, pytest.raises(SystemExit, match="the record is not written"):
        ea.main(["--render-only"])
    assert s.ran == [] and not (tmp_path / "record.md").exists()
    assert (tmp_path / f"{ea.CACHE}.json").read_bytes() == before


def test_t184_argument_refusals(tmp_path):
    for argv in (["--candidates", "C1"], ["--criteria", "X.16", "--render-only"]):
        with _Share(tmp_path) as s, pytest.raises(SystemExit):
            ea.main(argv)
        assert s.ran == []
