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

from conftest import stub_run

from experiments.llm_agents import files
from experiments.llm_agents.config import load_llm
from experiments.llm_agents.prompts import banned_hits
from experiments.llm_agents.record import engine_hash
from experiments.llm_agents.report import load_summaries, render
from experiments.llm_agents.runner import run_one
from experiments.llm_agents.source import family_terms
from src.bowen.render.trace import FRAMING


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
    out = [dict(s, phase="baseline") for s in base if s["seed"] in (1, 2)]
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
