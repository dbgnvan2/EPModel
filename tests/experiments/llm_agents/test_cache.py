"""LX.8, LX.9: the response cache.

Purpose: prove a replay never calls the client and a miss marks the run not replayable with nothing filled in (LX.8),
         and that two runs with an identical prompt keep separate entries (LX.9, P8 dirty state).
Spec:    docs/plan_v3_llm_agent_experiment.md#8, #10
Tests:   this file
"""

import gzip
import json

import pytest
from conftest import StubTransport
from helpers import config, phase

from experiments.llm_agents.cache import ResponseCache, cache_key
from experiments.llm_agents.client import CachedChooser, NotReplayable, request_body
from experiments.llm_agents.config import load_text
from experiments.llm_agents.prompts import Prompt
from experiments.llm_agents.runner import replay_all, run_one


def _never(cfg):
    raise AssertionError("a replay built a transport")


def test_lx8_replay_never_calls(tmp_path):
    cfg = config()
    live = run_one(phase(5), "A1", 1, cfg=cfg, live=True, transport=StubTransport(), runs=tmp_path)
    assert live.record["status"] == "complete" and live.record["counts"]["calls"] > 0
    replay = run_one(phase(5), "A1", 1, cfg=cfg, live=False, transport_factory=_never, runs=tmp_path,
                     write_record=False)
    assert replay.record["status"] == "complete"
    assert replay.record["counts"]["calls"] == 0
    assert replay.record["counts"]["cached"] == live.record["counts"]["calls"]
    assert replay.summary == live.summary


def test_lx8_miss_is_not_replayable(tmp_path):
    cfg = config()
    run_one(phase(5), "A1", 1, cfg=cfg, live=True, transport=StubTransport(), runs=tmp_path)
    path = tmp_path / "main-A1-s1" / "responses.jsonl.gz"
    lines = gzip.decompress(path.read_bytes()).decode().splitlines()
    week_3 = [ln for ln in lines if json.loads(ln)["week"] == 3]
    path.write_bytes(gzip.compress(("\n".join(ln for ln in lines if ln not in week_3[:1]) + "\n").encode()))
    out = run_one(phase(5), "A1", 1, cfg=cfg, live=False, transport_factory=_never, runs=tmp_path, write_record=False)
    assert out.record["status"] == "not_replayable"
    assert out.summary is None, "nothing is filled in"
    assert out.record["weeks_completed"] == 3


def test_lx9_run_key_separates_entries(tmp_path):
    cfg, text = config(), load_text()
    model = cfg.get("model_main")
    prompt = Prompt("system", "<round>same</round>", (("1", "a"), ("2", "b")))
    body = request_body(cfg, model, prompt)
    keys = {cache_key(rid, model, "A1", seed, "P1", 0, 0, body) for rid, seed in (("r1", 1), ("r2", 1), ("r1", 2))}
    assert len(keys) == 3
    cache = ResponseCache(tmp_path / "x.jsonl.gz", runs=tmp_path)
    first = CachedChooser(run_id="r1", model=model, arm="A1", seed=1, cfg=cfg, cache=cache, reask_text=text["reask"],
                          transport=StubTransport())
    first.choose(0, "P1", prompt)
    # The same prompt, in another run on the same shared cache (dirty state): not answered from the first run's entry.
    second = CachedChooser(run_id="r1", model=model, arm="A1", seed=2, cfg=cfg, cache=cache, reask_text=text["reask"])
    with pytest.raises(NotReplayable):
        second.choose(0, "P1", prompt)
    # Two seeds' runs written to one runs directory keep disjoint entries.
    for seed in (1, 2):
        run_one(phase(3), "A1", seed, cfg=cfg, live=True, transport=StubTransport(), runs=tmp_path)
    a, b = (ResponseCache.load(tmp_path / f"main-A1-s{s}" / "responses.jsonl.gz").entries for s in (1, 2))
    assert a and b and set(a).isdisjoint(b)


def test_lx8_replay_leaves_unfinished_runs_resumable(tmp_path):
    # A run stopped at the budget or by retries never fetched its later weeks. Replay must not run into that gap and
    # relabel it not_replayable ("the prompt has changed"); its status stays resumable, and replay skips it.
    # The smoke record is one call, not a run: it is skipped too.
    for phase_name, status in (("main", "incomplete_transient"), ("main", "stopped_at_budget"), ("smoke", "complete")):
        d = tmp_path / f"{phase_name}-A1-s{len(status)}"
        d.mkdir()
        rec = {"run_id": d.name, "status": status, "phase": phase_name, "arm": "A1", "seed": 1, "weeks_planned": 208}
        (d / "run.json").write_text(json.dumps(rec))
    assert replay_all(tmp_path) == []
    for d in tmp_path.iterdir():
        assert json.loads((d / "run.json").read_text())["status"] in ("incomplete_transient", "stopped_at_budget",
                                                                      "complete")


def test_lx8_replay_that_differs_is_not_replayable(tmp_path):
    # A change that leaves every prompt the same but moves what follows a choice still hits the cache. Replay catches
    # it by its summary: the same run replays to the committed summary, and a different one is not replayable.
    cfg = config()
    run_one(phase(5), "A1", 1, cfg=cfg, live=True, transport=StubTransport(), runs=tmp_path)
    record = tmp_path / "main-A1-s1" / "run.json"
    assert [o.record["status"] for o in replay_all(tmp_path, cfg)] == ["complete"]
    assert json.loads(record.read_text())["status"] == "complete"
    summary = tmp_path / "main-A1-s1" / "summary.json"
    data = json.loads(summary.read_text())
    data["llm_selections"] += 1
    data["summary_hash"] = _hashed(data)
    summary.write_text(json.dumps(data))
    replay_all(tmp_path, cfg)
    rec = json.loads(record.read_text())
    assert rec["status"] == "not_replayable" and f"differs from the committed {data['summary_hash']}" in rec["status_detail"]


def _hashed(summary: dict) -> str:
    from experiments.llm_agents.cache import canonical_json, text_hash
    return text_hash(canonical_json({k: v for k, v in summary.items() if k != "summary_hash"}))[:12]


def test_lx8_t207_replay_reads_behavior_not_the_engine_hash(tmp_path, monkeypatch):
    # T-207 (the owner, 2026-11-09, "Replay by behavior"): the engine hash is provenance, not the test. A replay on a
    # moved engine that reaches the same behavior is complete and restamps the summary; a replay on a moved engine
    # whose behavior differs is still not replayable, though the hash rule no longer decides it.
    from collections import Counter

    from experiments.llm_agents import record as run_record
    from experiments.llm_agents import summary as run_summary

    cfg = config()
    run_one(phase(5), "A1", 1, cfg=cfg, live=True, transport=StubTransport(), runs=tmp_path)
    d = tmp_path / "main-A1-s1"
    live_record = json.loads((d / "run.json").read_text())
    committed = json.loads((d / "summary.json").read_text())
    choices = (d / "choices.jsonl.gz").read_bytes()

    # The engine moved and the behavior did not: complete, the summary restamped, the run record left as it was.
    monkeypatch.setattr(run_record, "engine_hash", lambda: "feedfacecafe")
    [out] = replay_all(tmp_path, cfg)
    assert out.record["status"] == "complete"
    restamped = json.loads((d / "summary.json").read_text())
    assert restamped["engine_hash"] == "feedfacecafe" and restamped["summary_hash"] == _hashed(restamped)
    assert run_summary.behavior(restamped) == run_summary.behavior(committed)
    assert json.loads((d / "run.json").read_text()) == live_record, "the live run's engine hash stays as provenance"
    assert (d / "choices.jsonl.gz").read_bytes() == choices

    # The engine moved again and its behavior differs, with every prompt the same: not replayable.
    monkeypatch.setattr(run_record, "engine_hash", lambda: "0123456789ab")
    acts = run_summary.act_counts
    monkeypatch.setattr(run_summary, "act_counts", lambda run: acts(run) + Counter({"CONFLICT": 1}))
    [out] = replay_all(tmp_path, cfg)
    rec = json.loads((d / "run.json").read_text())
    assert rec["status"] == "not_replayable" and "differs from the committed" in rec["status_detail"]
    assert "engine_hash" in rec["status_detail"]
    assert json.loads((d / "summary.json").read_text()) == restamped, "a refused replay writes no summary"


def test_lx8_t207_replay_refuses_changed_choices_or_a_summary_off_its_hash(tmp_path):
    # Same acts: a replay whose choices differ from the committed choices is refused even when the summary matches.
    # A committed summary whose hash is not the hash of its content was edited by hand, and is refused (P6).
    from experiments.llm_agents import summary as run_summary

    cfg = config()
    run_one(phase(5), "A1", 1, cfg=cfg, live=True, transport=StubTransport(), runs=tmp_path)
    d = tmp_path / "main-A1-s1"
    good = (d / "run.json").read_text()
    choices = run_summary.read_choices(d)
    choices[0] = dict(choices[0], week=choices[0]["week"] + 1)
    run_summary.write_choices(choices, d, tmp_path)
    replay_all(tmp_path, cfg)
    rec = json.loads((d / "run.json").read_text())
    assert rec["status"] == "not_replayable" and "choices differ" in rec["status_detail"]

    (d / "run.json").write_text(good)
    (d / "choices.jsonl.gz").unlink()
    data = json.loads((d / "summary.json").read_text())
    data["summary_hash"] = "000000000000"
    (d / "summary.json").write_text(json.dumps(data))
    replay_all(tmp_path, cfg)
    rec = json.loads((d / "run.json").read_text())
    assert rec["status"] == "not_replayable" and "not the hash of its content" in rec["status_detail"]


@pytest.mark.xfail(strict=True, reason="spec revision 57 (T-185, T-206, 2026-11-10): the two changes the owner held for this batch moved the engine hash from 927550dfcb8a to d3779684dd53, and every record is regenerated once on it at the version 2 definition plan's step 4 (docs/implementation_plan_version_2_definition.md §4.1), not here, so this record is stale by decision until then")
def test_lx25_committed_pilot_replays_fixed(tmp_path):
    import shutil

    from experiments.llm_agents import files
    from experiments.llm_agents.summary import CHOICES

    committed = files.RUNS / "pilot-A1-s1"
    rec = json.loads((committed / "run.json").read_text())
    assert "option_order" not in rec, "the first pilot's record predates the field: it is read as fixed"
    shutil.copytree(committed, tmp_path / "pilot-A1-s1")
    (tmp_path / "pilot-A1-s1" / CHOICES).unlink()
    [out] = replay_all(tmp_path, config())
    assert out.record["status"] == "complete" and out.record["option_order"] == "fixed"
    assert out.summary["summary_hash"] == json.loads((committed / "summary.json").read_text())["summary_hash"]
    assert json.loads((tmp_path / "pilot-A1-s1" / "run.json").read_text()) == rec, "the record is left as it was"
    # The matched replay writes the run's choices with their positions, the same as the committed file.
    assert (tmp_path / "pilot-A1-s1" / CHOICES).read_bytes() == (committed / CHOICES).read_bytes()


def test_lx8_t207_replay_refuses_a_changed_status_or_a_complete_run_without_its_summary(tmp_path):
    # The behavior check needs both summaries; a replay that ends in another status than the committed run's, or a
    # complete run with no committed summary to compare, is not the committed run, and is refused rather than passed.
    cfg = config()
    run_one(phase(5), "A1", 1, cfg=cfg, live=True, transport=StubTransport(), runs=tmp_path)
    d = tmp_path / "main-A1-s1"
    good = json.loads((d / "run.json").read_text())
    (d / "run.json").write_text(json.dumps(dict(good, status="invalid")))
    replay_all(tmp_path, cfg)
    rec = json.loads((d / "run.json").read_text())
    assert rec["status"] == "not_replayable"
    assert "status complete differs from the committed invalid" in rec["status_detail"]

    (d / "run.json").write_text(json.dumps(good))
    (d / "summary.json").unlink()
    replay_all(tmp_path, cfg)
    rec = json.loads((d / "run.json").read_text())
    assert rec["status"] == "not_replayable" and "has no summary" in rec["status_detail"]
    assert not (d / "summary.json").exists(), "a refused replay writes no summary"
