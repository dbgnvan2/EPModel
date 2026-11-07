"""LX.11: the budget cap.

Purpose: prove the estimate refuses a phase over its cap before anything is built, and a run stops before the round
         whose worst case would cross the cap, announcing what it dropped, over a real-length run where the cap bites
         (P9).
Spec:    docs/plan_v3_llm_agent_experiment.md#5, #10
Tests:   this file
"""

import json
import logging

import pytest
from conftest import StubTransport
from helpers import config

from experiments.llm_agents.budget import OverCap, estimate, spent
from experiments.llm_agents.cache import ResponseCache
from experiments.llm_agents.runner import members, run_phase


def _paid(directory, cost, record_cost=None):
    """A run directory whose cache holds one paid response, and optionally a record that claims a different cost."""
    cache = ResponseCache(directory / "responses.jsonl.gz", runs=directory.parent)
    cache.put({"key": "k", "cost": cost})
    cache.save()
    if record_cost is not None:
        (directory / "run.json").write_text(json.dumps({"pool": "main", "counts": {"cost": record_cost}}))


def _never(cfg):
    raise AssertionError("a transport was built for a phase over its cap")


def test_lx11_estimate_over_cap_refuses(tmp_path):
    cfg = config()
    est = estimate(cfg.phases["main"], cfg, members(cfg))
    assert est["calls"] == 4 * 5 * 208 * 7 and not est["over"], "the decided phase fits its cap"
    pin, pout = cfg.prices(cfg.get("model_main"))
    per = (cfg.number("estimate_input_tokens") * pin + cfg.number("estimate_output_tokens") * pout) / 1e6
    assert est["cost"] == pytest.approx(est["calls"] * per)
    low = config(cap_main_usd=est["cost"] * 0.5)
    assert estimate(low.phases["main"], low, members(low))["over"]
    with pytest.raises(OverCap):
        run_phase("main", live=True, cfg=low, transport_factory=_never, runs=tmp_path)
    # What the pool already spent counts against the cap (the committed caches, across sessions).
    _paid(tmp_path / "pilot-A1-s1", cfg.cap("main") - est["cost"] * 0.5)
    with pytest.raises(OverCap):
        run_phase("main", live=True, cfg=cfg, transport_factory=_never, runs=tmp_path)


def test_lx11_stops_before_crossing(tmp_path, caplog):
    # A1 over the decided horizon, two seeds. The estimate ($0.66) passes the $1 cap, but each response uses five
    # times the estimated input, as a real pilot might: the cap bites part-way through the first run.
    cap = 1.0
    cfg = config(phase__main="arms=A1; seeds=1,2; weeks=208; model=claude-haiku-5-5; pool=main", cap_main_usd=cap)
    transport = StubTransport(usage={"input_tokens": 10000, "output_tokens": 50})
    with caplog.at_level(logging.WARNING, logger="experiments.llm_agents.runner"):
        outcomes = run_phase("main", live=True, cfg=cfg, transport_factory=lambda c: transport, runs=tmp_path)
    assert len(outcomes) == 1
    rec = outcomes[0].record
    assert rec["status"] == "stopped_at_budget"
    assert 0 < rec["weeks_completed"] < 208
    assert cap - 0.05 < rec["counts"]["cost"] <= cap, "stopped near the cap, and did not cross it"
    assert f"weeks {rec['weeks_completed']} of 208 completed, the rest dropped" in rec["status_detail"]
    assert any("seed 2 not started" in r.getMessage() for r in caplog.records), "the run not started is announced"
    assert outcomes[0].summary is None, "a stopped run is not a result"


def test_lx11_phase_record_lists_runs_not_started(tmp_path):
    cfg = config(phase__main="arms=A1; seeds=1,2; weeks=208; model=claude-haiku-5-5; pool=main", cap_main_usd=1.0)
    transport = StubTransport(usage={"input_tokens": 10000, "output_tokens": 50})
    run_phase("main", live=True, cfg=cfg, transport_factory=lambda c: transport, runs=tmp_path)
    data = json.loads((tmp_path / "phase-main.json").read_text())
    assert data == {"phase": "main", "runs_planned": 2, "runs": {"main-A1-s1": "stopped_at_budget"},
                    "not_started": ["main-A1-s2"]}


def test_lx11_spent_counts_every_cached_response(tmp_path):
    # A crash leaves a cache and no record; a rerun's record can count fewer entries than its cache holds. The money
    # paid is what the caches hold, so both count in full.
    cfg = config()
    _paid(tmp_path / "main-A1-s1", 0.30)
    _paid(tmp_path / "main-A1-s2", 0.25, record_cost=0.05)
    _paid(tmp_path / "pilot-A1-s1", 0.10)          # the pilot draws on the main pool (llm.md)
    _paid(tmp_path / "cross_check-A1-s1", 0.40)
    assert spent(tmp_path, cfg) == pytest.approx({"main": 0.65, "cross_check": 0.40})
    assert spent(tmp_path, cfg, by="phase") == pytest.approx({"main": 0.55, "pilot": 0.10, "cross_check": 0.40})


def test_lx11_cached_rounds_are_not_charged(tmp_path):
    # A rerun of a phase answered wholly from its cache calls nothing, so neither the estimate nor the worst case may
    # stop it for money it does not spend.
    cfg = config(phase__main="arms=A1; seeds=1; weeks=3; model=claude-haiku-5-5; pool=main")
    first = StubTransport()
    run_phase("main", live=True, cfg=cfg, transport_factory=lambda c: first, runs=tmp_path)
    paid = spent(tmp_path, cfg)["main"]
    est = estimate(cfg.phases["main"], cfg, members(cfg))["cost"]
    tight = config(phase__main="arms=A1; seeds=1; weeks=3; model=claude-haiku-5-5; pool=main",
                   cap_main_usd=max(paid, est) * 1.001)

    class Refuses(StubTransport):
        def send(self, body):
            raise AssertionError("a cached round called the API")

    outcomes = run_phase("main", live=True, cfg=tight, transport_factory=lambda c: Refuses(), runs=tmp_path)
    assert outcomes[0].record["status"] == "complete"
    assert outcomes[0].record["counts"]["calls"] == 0


def test_lx11_total_cap_refuses_before_starting(tmp_path):
    # The cross-check's own pool has room, but the main pool's spend leaves the total cap too little: the phase is
    # refused before anything is built, and the stop names the total cap.
    cfg = config()
    est = estimate(cfg.phases["cross_check"], cfg, members(cfg))
    assert est["cost"] < cfg.cap("cross_check")
    _paid(tmp_path / "main-A1-s1", cfg.number("cap_total_usd") - est["cost"] * 0.5)
    with pytest.raises(OverCap, match="total cap"):
        run_phase("cross_check", live=True, cfg=cfg, transport_factory=_never, runs=tmp_path)
