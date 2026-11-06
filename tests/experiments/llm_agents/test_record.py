"""LX.10: the run record has every field of §8.

Purpose: prove ``run.json`` carries the models requested and returned, the API and SDK versions, effort, thinking,
         ``max_tokens``, sampling, the time, the engine, code, template and label hashes, the arm, seed and
         permutations, the status and the counts, each filled in.
Spec:    docs/plan_v3_llm_agent_experiment.md#8, #10
Tests:   this file
"""

import json

import pytest
from conftest import StubTransport
from helpers import config, phase

from experiments.llm_agents import record
from experiments.llm_agents.runner import run_one


def test_lx10_run_record_fields(tmp_path):
    cfg = config()
    run_one(phase(3), "A3", 2, cfg=cfg, live=True, transport=StubTransport(), runs=tmp_path)
    rec = json.loads((tmp_path / "main-A3-s2" / "run.json").read_text())
    assert tuple(rec) == record.FIELDS
    model = cfg.get("model_main")
    assert rec["model_requested"] == model and rec["models_returned"] == [model]
    assert rec["api_version"] == "stub" and rec["sdk_version"] == "stub"
    assert (rec["effort"], rec["thinking"], rec["max_tokens"]) == (cfg.effort(model), cfg.thinking(model),
                                                                  cfg.integer("max_tokens"))
    assert rec["sampling"].startswith("default; not settable")
    assert rec["time"].endswith("Z")
    assert rec["engine_hash"] == "0b363cb3186c"
    for h in ("code_hash", "template_hash", "label_hash", "config_hash"):
        assert len(rec[h]) == 12, h
    assert (rec["arm"], rec["seed"], rec["status"]) == ("A3", 2, "complete")
    assert set(rec["permutations"]) == {"person_order", "act_token", "label_of"}
    assert set(rec["counts"]) >= {"calls", "reasks", "fallbacks", "retries", "cost"}
    assert rec["counts"]["calls"] == rec["selections"] == 3 * 7 and rec["counts"]["cost"] > 0


def test_lx10_status_is_one_of_the_declared():
    with pytest.raises(ValueError):
        record.make(run_id="x", phase="main", pool="main", arm="A1", seed=1, family="phase_c", weeks_planned=1,
                    weeks_completed=1, model="m", models_returned=[], api_version=None, sdk_version=None,
                    effort=None, thinking=None, max_tokens=None, permutations=None, status="done",
                    status_detail=None, counts={}, selections=0, fallback_share=0.0)
