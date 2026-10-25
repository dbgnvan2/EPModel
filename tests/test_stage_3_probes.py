"""Smoke tests of stage 3's measurement tools (steps 2 and 3), on a few seeds.

Purpose: `tools/stage_3_projection_trace.py`, `tools/stage_3_iposition_probe.py` and
         `tools/stage_3_displaced_coping_probe.py` write the review pages the owner's decisions D18 and D20 wait on
         (`docs/implementation_plan_version_2_stage_3.md` §3.3.1, §3.3.2). These
         tests run each tool's own arm functions on a few seeds and check that what they read is what the engine did:
         the trace's terms reproduce the development step, and the probe's arms differ only in the mover's level and
         in whether the act is scripted. Each tool's page keeps its hand-written "Reading" section when rewritten.
Spec:    docs/bowen_agent_model_spec_v2.md#M7.E.1f, #M11.C.6, #M5.F.4, #M11.C.29, #M1.A.5e
Tests:   this file
"""

from __future__ import annotations

import math

import tools.stage_3_iposition_probe as probe
import tools.stage_3_projection_trace as trace
from src.bowen.ensemble.criteria_d import ENGINE


def test_stage_3_trace_reproduces_the_development_step(tmp_path):
    """On seed 0, in each arm: the readout `development` is the sum of the traced steps, and every step the band did
    not bound is ``development_rate`` × the mean of the two traced terms, so the trace reads what the rule read.
    Mutation (reasoned, not run): the own term read from the raw functional level, with pseudo-self, would break the
    second check on any step where the child holds pseudo-self."""
    row = trace.one_seed(0)
    for arm in trace.ARMS:
        steps = row[arm]["steps"]
        assert sorted(steps) == list(range(ENGINE.development_start_age_years + 1,
                                           ENGINE.development_end_age_years + 1)), arm
        assert math.isclose(sum(s["step"] for s in steps.values()), row[arm]["development"], abs_tol=1e-9), arm
        free = [s for s in steps.values() if not s["clamped"]]
        assert free, arm
        for s in free:
            terms = [s["own_term"]] + ([] if math.isnan(s["parents_term"]) else [s["parents_term"]])
            assert math.isclose(s["step"], ENGINE.development_rate * sum(terms) / len(terms), abs_tol=1e-9)
    assert row["baseline"]["acts"] == 0 < row["treatment"]["acts"]
    page = tmp_path / "trace.md"
    page.write_text("old\n\n## Reading\n\nhand-written\n", encoding="utf-8")
    keep = page.read_text(encoding="utf-8")
    text = trace.render(trace.summarise([row, row], ENGINE.development_rate), "e" * 12, "2026-10-24", 2,
                        keep[keep.index(trace.READING):])
    assert "## The answer: which term carries the difference" in text and text.endswith("hand-written\n")


def test_stage_3_probe_arms_differ_only_in_the_movers_level(tmp_path):
    """On seeds 0–2: each act arm makes one `I-POSITION` sequence from F and each control none; the mover's level is
    the arm's; and the page renders and keeps its "Reading" section. Mutation: dropping the control arms' withhold
    lets the policy's own `I-POSITION` from F through (seen on seed 0 while the tool was written)."""
    rows = [probe.one_seed(s) for s in range(3)]
    for row in rows:
        for level in probe.LEVELS:
            acting, control = row[f"act_{level:g}"], row[f"none_{level:g}"]
            assert acting["steps"].get("iposition:begin:PREPARE", 0) + acting["forms"].get("assertion", 0) == 1
            assert not control["forms"] and not control["steps"]
            assert acting["mover_basic"] == control["mover_basic"] == level
    page = tmp_path / "probe.md"
    probe.main(["--seeds", "3", "--workers", "1", "--out", str(page)])
    page.write_text(page.read_text(encoding="utf-8").replace("(to be written)", "hand-written"), encoding="utf-8")
    probe.main(["--seeds", "3", "--workers", "1", "--out", str(page)])
    text = page.read_text(encoding="utf-8")
    assert "## Contrasts, every seed" in text and "hand-written" in text


def test_stage_3_displaced_coping_probe_moves_conserved_stress_functioning(tmp_path):
    """On seed 0, as built and under P2 with `OVERFUNCTION`: the child's own term is the sum of its parts at every step
    (so the probe's split reads what the step read); as built, P2's wrapper moves nothing; under P2 it moves
    functioning to and from the child in the treatment arm; and the page renders and keeps its "Reading" section.
    Mutation (reasoned, not run): the weekly split read against the basic level after the step would break the
    first check by the step ÷ 52 at every step."""
    import tools.stage_3_displaced_coping_probe as coping

    row = coping.one_seed(0, ("as_built", "p2_overfunction"))
    for variant in ("as_built", "p2_overfunction"):
        for arm in coping.ARMS:
            for s in row[variant][arm]["steps"].values():
                parts = s["stress"] + s["adjust_neg"] + s["unledgered"] - s["developed"]
                assert math.isclose(s["own_term"], parts, abs_tol=1e-9), (variant, arm)
    assert all(row["as_built"][arm][c] == 0 for arm in coping.ARMS for c in ("displaced_to_child",
                                                                          "displaced_from_child"))
    assert row["p2_overfunction"]["treatment"]["displaced_to_child"] > 0
    assert row["p2_overfunction"]["baseline"]["displaced_to_child"] == 0
    page = tmp_path / "coping.md"
    page.write_text("old\n\n## Reading\n\nhand-written\n", encoding="utf-8")
    full = {v: row[v] if v in row else row["p2_overfunction"] for v in coping.VARIANTS}
    text = coping.render(coping.summarise([{"seed": 0, **full}, {"seed": 1, **full}], ENGINE.development_rate),
                         "e" * 12, "2026-10-25", 2, page.read_text(encoding="utf-8").split("\n\n", 1)[1])
    assert "## The direction under each variant" in text and text.endswith("hand-written\n")


def test_stage_3_displaced_coping_probe_f1_ledger_holds_at_the_floor_and_sheds_off_it():
    """F1 as the probe puts it in (`docs/PROPOSAL — DISPLACED COPING (P2).md` §4.2): a week in which the floor held
    functional level while the swing decayed puts the held amount on the ledger, and the bound stops it being shed;
    off the floor a calm week sheds ``response_relaxation_rate`` of it, moving functional level with it; "kept" never
    sheds. Mutation (run 2026-10-25): shedding without the floor's bound (``move = -step``) takes the first week's
    person below ``basic − stress_floor_below_basic``, and the test fails."""
    from types import SimpleNamespace

    import tools.stage_3_displaced_coping_probe as coping

    floor = 50.0 - ENGINE.stress_floor_below_basic
    person = SimpleNamespace(alive=True, basic_level=50.0, functional_level=floor, stress_swing=9.8, adjust_swing=0.0,
                             usual_excess=None)
    state = SimpleNamespace(people={"a": person})
    ledger = {}
    # The week the floor clamped: the swing decayed from 10 to 9.8 and functional level did not move.
    assert coping.floor_held(state, {}, ENGINE, ledger, {"a": (floor, 10.0, 0.0)}, "shed") == {}
    assert math.isclose(ledger["a"], 0.2) and person.functional_level == floor
    # Off the floor and calm: a share of the ledger is shed, functional level with it.
    person.functional_level = floor + 5.0
    shed = coping.floor_held(state, {}, ENGINE, ledger, {"a": (floor + 5.0, 9.8, 0.0)}, "shed")
    step = ENGINE.response_relaxation_rate * 0.2
    assert math.isclose(shed["a"], -step) and math.isclose(person.functional_level, floor + 5.0 - step)
    assert math.isclose(ledger["a"], 0.2 - step)
    # "kept" records the same amount and never sheds it.
    kept = {}
    person.functional_level = floor + 5.0
    assert coping.floor_held(state, {}, ENGINE, kept, {"a": (floor + 5.2, 9.8, 0.0)}, "kept") == {}
    assert math.isclose(kept["a"], -0.2) and person.functional_level == floor + 5.0


def test_stage_3_displaced_coping_probe_joins_split_runs_by_seed():
    """Rows from runs of different variants (``--variants``) join by seed; a variant in two files is refused."""
    import pytest

    import tools.stage_3_displaced_coping_probe as coping

    joined = coping.merge([[{"seed": 1, "as_built": 1}, {"seed": 0, "as_built": 0}], [{"seed": 0, "as_built_f1": 2},
                                                                                       {"seed": 1, "as_built_f1": 3}]])
    assert joined == [{"seed": 0, "as_built": 0, "as_built_f1": 2}, {"seed": 1, "as_built": 1, "as_built_f1": 3}]
    assert coping.present(joined) == ["as_built", "as_built_f1"]
    # A split run without "as_built" (the F1 share alone) still summarises; it has nothing to compare against.
    steps = {"6": dict.fromkeys(coping.PARTS, 0.0)}
    alone = {"seed": 0, "as_built_f1": {arm: {"development": 0.0, "acts": 0, "steps": steps,
                                               **dict.fromkeys(coping.COUNTS, 0.0)} for arm in coping.ARMS}}
    assert coping.summarise([alone, {**alone, "seed": 1}], ENGINE.development_rate)["against_as_built"] == {}
    with pytest.raises(SystemExit):
        coping.merge([[{"seed": 0, "as_built": 0}], [{"seed": 0, "as_built": 1}]])
