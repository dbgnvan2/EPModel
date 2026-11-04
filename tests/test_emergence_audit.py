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
    """On the committed records the scope is plan §3's thirteen rows, with `M11.C.39` excluded as the control."""
    audited, rows, _, excluded = ea.in_scope(ea.read_records(), ea.read_config())
    ea.check_registries(rows)
    assert len(audited) == 13 and "M11.C.48[phase_d]" in {r.key for r in audited}
    assert [k for k, _ in excluded] == ["M11.C.39"]


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
    assert ea.read_config(text).stage in ("pilot", "full")
    with pytest.raises(SystemExit, match="unknown ablation"):
        ea.read_config(text.replace("| positive | `M11.C.39` | L0 |", "| positive | `M11.C.39` | L9 |"))
    with pytest.raises(SystemExit, match="not pilot or full"):
        ea.read_config(text.replace("| stage | pilot |", "| stage | some |").replace("| stage | full |", "| stage | some |"))
    with pytest.raises(SystemExit, match="cells"):
        ea.read_config(text.replace("| `M11.C.39` | its treatment arm", "| `M11.C.39` | its | treatment arm"))
    with pytest.raises(SystemExit, match="twice"):
        ea.read_config(text.replace("| `M11.C.48[phase_d]` |", "| `M11.C.27[unstable,remove_one]` |"))


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
