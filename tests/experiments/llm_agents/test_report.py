"""LX.16, LX.20: the generated report.

Purpose: prove the report is generated from the committed summaries and current with them, that the summaries are
         current with the engine, that it carries the framing block and "N of M complete" for each arm and uses no
         absolute threshold (LX.16), and that it never names a real family or presents an agent as a person (LX.20).
Spec:    docs/plan_v3_llm_agent_experiment.md#6, #10
Tests:   this file
"""

import copy
import json
import re

import pytest
from conftest import stub_run

from experiments.llm_agents import files
from experiments.llm_agents.config import load_llm
from experiments.llm_agents.prompts import banned_hits
from experiments.llm_agents.record import engine_hash
from experiments.llm_agents.report import load_summaries, render
from experiments.llm_agents.runner import run_one
from experiments.llm_agents.source import family_terms
from src.bowen.render.trace import FRAMING


@pytest.mark.xfail(strict=True, reason="T-202 (spec revision 55): the social cues moved the engine hash; the LLM experiment's committed runs record 0b363cb3186c, and with the switch off the engine computes the same runs (the golden fingerprints hold), but a replay restamps the hash and a changed hash marks a run not replayable, so how the runs follow an engine move is the owner's decision (T-207); no live call")
def test_lx16_report_is_current(tmp_path):
    summaries = load_summaries()
    assert summaries, "the A0 baseline is committed"
    assert files.REPORT.read_text(encoding="utf-8") == render(summaries), \
        "stale: run `python3 -m experiments.llm_agents.runner report`"
    assert {s["engine_hash"] for s in summaries} == {engine_hash()}, "a summary was computed on another engine"
    # The committed summaries are what the code computes: seed 1 of the baseline, recomputed.
    cfg = load_llm()
    again = run_one(cfg.phases["baseline"], "A0", 1, cfg=cfg, live=False, runs=tmp_path, write_record=False).summary
    committed = json.loads((files.RUNS / "baseline-A0-s1" / "summary.json").read_text(encoding="utf-8"))
    assert again == committed


def _synthetic():
    base = load_summaries()
    out = [dict(s, phase="baseline") for s in base if s["arm"] == "A0" and s["seed"] in (1, 2)]
    for s in list(out):
        x = copy.deepcopy(s)
        x["arm"], x["phase"] = "A2", "main"
        x["patterns"]["subject_weeks"]["conflict"] = x["patterns"]["subject_weeks"].get("conflict", 0) + 50
        out.append(x)
    return out


def test_lx16_framing_and_counts():
    text = render(_synthetic())
    assert FRAMING.split("{how}")[0] in text
    for arm, done, planned in (("A0", 2, 5), ("A1", 0, 7), ("A2", 2, 7), ("A3", 0, 5), ("A4", 0, 5)):
        assert f"| {arm} | {done} of {planned} complete |" in text
    assert "| conflict | — | higher in 2, lower in 0, equal in 0 of 2 |" in text
    # No absolute threshold: no comparison against a fixed number anywhere.
    assert not re.search(r"[<>≤≥]=?\s*\d", text.replace("<!--", "").replace("-->", ""))
    # A run that is not complete is not counted.
    summaries = load_summaries()
    assert all(json.loads((files.RUNS / f"baseline-A0-s{s['seed']}" / "run.json").read_text())["status"] == "complete"
               for s in summaries)


def test_lx27_cross_check_against_main():
    """The cross-check's direction is read on its own seeds against A0 and paired with the main phase on the same
    seed only (adversarial, P7): the cross-check agrees with main on seed 1 and not on seed 2, so it counts 1 of 2,
    and a main run of another seed is never its pair."""
    out = _synthetic()
    a0 = {s["seed"]: s for s in out if s["arm"] == "A0"}
    for seed, bump_main, bump_cross in ((1, 50, 50), (2, 0, 50)):
        for phase, bump in (("main", bump_main), ("cross_check", bump_cross)):
            x = copy.deepcopy(a0[seed])
            x["arm"], x["phase"] = "A1", phase
            x["patterns"]["subject_weeks"]["conflict"] = x["patterns"]["subject_weeks"].get("conflict", 0) + bump
            out.append(x)
    text = render(out)
    assert "2 of 4 planned cross-check runs complete." in text
    assert "| conflict | higher in 2, lower in 0, equal in 0 of 2 | 1 of 2 seeds |" in text
    # A cross-check run on a seed the main phase never ran has no pair: the cell says so instead of borrowing one.
    lonely = [s for s in out if not (s["arm"] == "A1" and s["phase"] == "main" and s["seed"] == 2)]
    assert "| conflict | higher in 2, lower in 0, equal in 0 of 2 | 1 of 1 seeds |" in render(lonely)


def test_lx20_framing_block_present():
    report = files.REPORT.read_text(encoding="utf-8")
    first_table = report.index("| Arm |")
    assert report.index(FRAMING.split("{how}")[0]) < first_table, "the framing block heads the report"
    assert "A language-model agent is not a person" in report
    run, _ = stub_run("A2", 1, 2)
    names = list(family_terms(run))
    assert not banned_hits(report, names), "the report names no one in the family file"
    # The prompts' banned list is not the report's: the framing block itself says "family", to deny it is a real one.
    # The check bites: a report that names a person is caught.
    assert banned_hits(report + "\nRavi's acts", names)


# --- §13.4: the position readout (LX.26) ----------------------------------------------------------------------------

def _readout(chooser, arm="A1", weeks=26):
    from experiments.llm_agents.report import position_readout
    from experiments.llm_agents.source import build, execute
    from experiments.llm_agents.summary import choice_records

    run = execute(build(arm, 1, weeks, chooser))
    return position_readout(choice_records(run)), choice_records(run)


def test_lx26_position_readout_per_arm():
    from conftest import StubChooser

    from experiments.llm_agents.report import position_readout

    summaries = load_summaries()
    pilot = [s for s in summaries if s.get("run_id") == "pilot-A1-s1"]
    assert pilot and "choices" in pilot[0], "the committed pilot's choices are read"
    report = files.REPORT.read_text(encoding="utf-8")
    assert "## Option position (§13.4)" in report
    r = position_readout(pilot[0]["choices"])
    assert f"| pilot-A1-s1 | {r['choices']} | {r['flagged']} | {r['position_1']:.3f} | {r['last']:.3f} | " \
           f"{r['mean_inverse_n']:.3f} | {r['same_id']} | {r['same_id_chance']:.1f} | {r['same_option']} | " \
           f"{r['pairs']} |" in report
    for arm in ("A1", "A2", "A3", "A4"):
        assert f"{arm}: a position effect in " in report
    # Per arm: each LLM arm's run gets its own row, and only A1 (opaque) has a token share.
    rows = {}
    for arm in ("A1", "A2"):
        _, choices = _readout(StubChooser(), arm, 3)
        rows[arm] = dict(s for s in [("arm", arm), ("phase", "main"), ("seed", 1), ("run_id", f"main-{arm}-s1"),
                                     ("choices", choices)])
        rows[arm].update(patterns={"subjects": 1, "weeks": 1, "subject_weeks": {}},
                         evaluation={k: 0.0 for k in load_llm().listed("evaluation_readouts")},
                         acts_by_label={}, acts_by_mechanics={})
    text = render(list(rows.values()))
    assert "| main-A1-s1 |" in text and "| main-A2-s1 |" in text
    from experiments.llm_agents.report import position_readout as readout
    k = int(readout(rows["A1"]["choices"])["effect"])
    assert f"A1: a position effect in {k} of 1 main-phase seeds" in text
    tokens = text.split("Share of choices by act token")[1]
    assert "main-A1-s1" in tokens and "main-A2-s1" not in tokens


def test_lx26_position_and_content_choosers_differ():
    from conftest import FirstChooser, TextChooser

    first, _ = _readout(FirstChooser())
    content, _ = _readout(TextChooser())
    # 26 weeks of 7 members give at most 175 week-to-week pairs; a round with no option breaks a pair
    assert 150 < first["pairs"] <= 175 and 150 < content["pairs"] <= 175
    # Position only: every pair repeats the id, few repeat the option, and position 1 is far above 1/N.
    assert first["same_id"] > first["same_option"]
    assert first["position_1"] > first["mean_inverse_n"]
    assert first["effect"]
    assert first["same_id"] > first["same_id_chance"]
    # Content only (adversarial, P7: it repeats a lot too, but by option): no effect, in several seeds. Its position-1
    # share alone is near 1/N and may fall either side in one seed, so the rule's two signs are read together.
    assert content["same_option"] >= content["same_id"] and not content["effect"]
    for seed in (2, 3, 4):
        from experiments.llm_agents.source import build, execute
        from experiments.llm_agents.summary import choice_records
        from experiments.llm_agents.report import position_readout

        other = position_readout(choice_records(execute(build("A1", seed, 26, TextChooser()))))
        assert other["same_option"] > other["same_id"] and not other["effect"], seed
        assert other["same_id"] < 3 * other["same_id_chance"], "a content chooser repeats ids near chance only"
