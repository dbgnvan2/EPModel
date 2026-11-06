"""The runner: phases, arms and seeds; the estimate; ``--live``; replay; the summaries and the report.

Purpose: run a phase's runs (A0 on the policy; A1–A4 through the cache, and through the API only under ``--live``),
         refuse a phase whose estimate is over its cap, stop before a round that would cross it, mark every run with
         an honest status, and write its cache, audit prompts, record and summary under ``runs/<run_id>/``.
Spec:    docs/plan_v3_llm_agent_experiment.md#5, #7, #8, #11
Tests:   tests/experiments/llm_agents/test_isolation.py::test_lx15_live_client_only_when_asked,
         tests/experiments/llm_agents/test_budget.py::test_lx11_stops_before_crossing,
         tests/experiments/llm_agents/test_client.py::test_lx12_resumes_from_cache

Usage (from the repo root):

    python3 -m experiments.llm_agents.runner estimate main
    python3 -m experiments.llm_agents.runner run baseline           # A0: no call, no cost
    python3 -m experiments.llm_agents.runner run pilot --live       # only on the owner's go-ahead
    python3 -m experiments.llm_agents.runner replay                 # every run, from its cache; never calls
    python3 -m experiments.llm_agents.runner report
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from experiments.llm_agents import files
from experiments.llm_agents import record as run_record
from experiments.llm_agents import summary as run_summary
from experiments.llm_agents.arms import SPECS
from experiments.llm_agents.budget import Budget, BudgetStop, estimate, refuse_over_cap, spent, worst_case
from experiments.llm_agents.cache import ResponseCache, canonical_json
from experiments.llm_agents.client import (
    AnthropicTransport, CachedChooser, NotReplayable, PermanentError, TransientExhausted,
)
from experiments.llm_agents.config import LLMConfig, Phase, load_llm, load_text
from experiments.llm_agents.source import build, step

log = logging.getLogger("experiments.llm_agents.runner")


def run_id(phase: str, arm: str, seed: int) -> str:
    return f"{phase}-{arm}-s{seed}"


def audit_weeks(cfg: LLMConfig, weeks: int) -> set[int]:
    out = set()
    for w in cfg.listed("audit_prompt_weeks"):
        out.add(weeks - 1 if w == "last" else int(w))
    return out


def members(cfg: LLMConfig) -> int:
    """The family members who choose each week (the external agent stays on the policy)."""
    run = build("A0", 1, 1)
    return sum(1 for p in run.state.people.values() if p.role.name == "MEMBER")


@dataclass
class Outcome:
    record: dict
    summary: dict | None


def run_one(phase: Phase, arm: str, seed: int, *, cfg: LLMConfig, live: bool,
            transport_factory: Callable[[LLMConfig], object] | None = None, transport=None,
            budget: Budget | None = None, runs: Path = files.RUNS, write_record: bool = True,
            sleep: Callable[[float], None] = time.sleep) -> Outcome:
    """Purpose: one run of a phase, to an honest status, with what it got written under ``runs/<run_id>/``.
    Spec:    docs/plan_v3_llm_agent_experiment.md#5, #7, #8
    Tests:   tests/experiments/llm_agents/test_budget.py::test_lx11_stops_before_crossing,
             tests/experiments/llm_agents/test_client.py::test_lx12_resumes_from_cache
    """
    rid = run_id(phase.name, arm, seed)
    directory = runs / rid
    spec = SPECS[arm]
    model = phase.model if spec.llm else None
    text = load_text()
    chooser = None
    audit: dict = {}
    if spec.llm:
        if model is None:
            raise ValueError(f"phase {phase.name} runs {arm} but names no model")
        if live and transport is None:
            transport = (transport_factory or AnthropicTransport)(cfg)
        cache = ResponseCache.load(directory / "responses.jsonl.gz", runs)
        chooser = CachedChooser(run_id=rid, model=model, arm=arm, seed=seed, cfg=cfg, cache=cache,
                                reask_text=text["reask"], transport=transport if live else None, budget=budget,
                                sleep=sleep)
    weeks_audit = audit_weeks(cfg, phase.weeks)

    def before_round(week, prompts):
        if live and budget is not None:
            calling = [prompt for token, prompt in prompts if chooser.needs_call(week, token, prompt)]
            if calling:
                budget.check(week, worst_case(calling, cfg, model, text["reask"], budget.largest_call))

    def on_prompt(week, token, prompt):
        if week in weeks_audit:
            audit[f"{week}:{token}"] = {"week": week, "person": token, "system": prompt.system, "user": prompt.user}

    run = build(arm, seed, phase.weeks, chooser, before_round=before_round, on_prompt=on_prompt)
    watch = run_summary.PatternWatch(run)
    status, detail, done = "complete", None, 0
    try:
        for _ in range(phase.weeks):
            step(run)
            watch.after_week(run)
            done += 1
    except BudgetStop as e:
        status, detail = "stopped_at_budget", f"{e}; weeks {done} of {phase.weeks} completed, the rest dropped"
    except TransientExhausted as e:
        status, detail = "incomplete_transient", f"{e}; weeks {done} of {phase.weeks}; a rerun resumes from the cache"
    except NotReplayable as e:
        status, detail = "not_replayable", str(e)
    except PermanentError as e:
        status, detail = "stopped_permanent", str(e)
    finally:
        if chooser is not None and chooser.cache.new:
            chooser.cache.save()
    selections = len(getattr(run.source, "choices", []))
    counts = chooser.counts.as_record() if chooser else {"calls": 0, "cached": 0, "reasks": 0, "fallbacks": 0,
                                                         "retries": 0, "cost": 0.0, "failures": {}}
    share = counts["fallbacks"] / selections if selections else 0.0
    if status == "complete" and share > cfg.number("max_fallback_share"):
        status, detail = "invalid", f"{counts['fallbacks']} of {selections} selections were flagged holds"
    models_returned = {e.get("model") for e in chooser.cache.entries.values() if e.get("model")} if chooser else ()
    rec = run_record.make(
        run_id=rid, phase=phase.name, pool=phase.pool if spec.llm else None, arm=arm, seed=seed,
        family=cfg.get("family"), weeks_planned=phase.weeks, weeks_completed=done, model=model,
        models_returned=models_returned, api_version=getattr(transport, "api_version", None) if live else None,
        sdk_version=getattr(transport, "sdk_version", None) if live else None,
        effort=cfg.effort(model) if model else None, thinking=cfg.thinking(model) if model else None,
        max_tokens=cfg.integer("max_tokens") if model else None,
        permutations=run.perms.as_record() if run.perms else None, status=status, status_detail=detail,
        counts=counts, selections=selections, fallback_share=round(share, 6))
    summary = run_summary.summarize(run, watch, cfg) if status == "complete" else None
    if write_record:
        run_record.write(rec, directory, runs)
        if audit and chooser is not None and chooser.cache.new:
            lines = "".join(canonical_json(audit[k]) + "\n" for k in sorted(audit))
            files.write_bytes(directory / "prompts_audit.jsonl.gz", gzip.compress(lines.encode(), mtime=0), runs=runs)
    if summary is not None and write_record:
        run_summary.write(summary, directory, runs)
    if status == "stopped_permanent":
        raise PermanentError(detail)
    return Outcome(rec, summary)


def run_phase(name: str, *, live: bool, cfg: LLMConfig | None = None, transport_factory=None,
              runs: Path = files.RUNS) -> list[Outcome]:
    """Purpose: a whole phase: the estimate against the cap first, then each run, stopping all at the cap (§5).
    Spec:    docs/plan_v3_llm_agent_experiment.md#5
    Tests:   tests/experiments/llm_agents/test_budget.py::test_lx11_estimate_over_cap_refuses
    """
    cfg = cfg or load_llm()
    phase = cfg.phases[name]
    if phase.model is not None and not live:
        raise SystemExit(f"phase {name} calls {phase.model}: it runs only with --live, on the owner's go-ahead; "
                         "use `replay` to rerun a committed phase from its cache")
    if name == "smoke":
        raise SystemExit("the smoke call is `smoke --live`, not `run smoke`")
    budget = None
    transport = None
    if phase.model is not None:
        by_pool = spent(runs, cfg)
        est = estimate(phase, cfg, members(cfg))
        # the estimate covers the whole phase, so what this phase already paid (a resumed run's cached weeks) is in it
        already = spent(runs, cfg, by="phase").get(name, 0.0)
        refuse_over_cap(est, by_pool.get(phase.pool, 0.0) - already, sum(by_pool.values()) - already)
        budget = Budget(cfg.cap(phase.pool), cfg.number("cap_total_usd"), by_pool.get(phase.pool, 0.0),
                        sum(by_pool.values()))
        transport = (transport_factory or AnthropicTransport)(cfg)
        if hasattr(transport, "check_models"):
            transport.check_models((phase.model,))
    outcomes = []
    stopped, why = False, ""
    not_started: list[str] = []
    for arm in phase.arms:
        for seed in phase.seeds:
            if stopped:
                log.warning("phase %s: %s seed %d not started (%s)", name, arm, seed, why)
                not_started.append(run_id(name, arm, seed))
                continue
            out = run_one(phase, arm, seed, cfg=cfg, live=live, transport=transport, budget=budget, runs=runs)
            outcomes.append(out)
            if out.record["status"] in ("stopped_at_budget", "incomplete_transient"):
                stopped, why = True, f"{out.record['run_id']} ended {out.record['status']}"
    write_phase_record(name, len(phase.arms) * len(phase.seeds), outcomes, not_started, runs)
    return outcomes


def write_phase_record(name: str, planned: int, outcomes: list[Outcome], not_started: list[str], runs: Path) -> None:
    """Purpose: the phase's own record, so the runs not started are an artifact, not only a log line (§5, P2).
    Spec:    docs/plan_v3_llm_agent_experiment.md#5
    Tests:   tests/experiments/llm_agents/test_budget.py::test_lx11_phase_record_lists_runs_not_started
    """
    data = {"phase": name, "runs_planned": planned, "runs": {o.record["run_id"]: o.record["status"] for o in outcomes},
            "not_started": not_started}
    files.write_text(runs / f"phase-{name}.json", json.dumps(data, indent=2, sort_keys=True) + "\n", runs=runs)


REPLAYED = ("complete", "invalid")


def replay_all(runs: Path = files.RUNS, cfg: LLMConfig | None = None) -> list[Outcome]:
    """Purpose: every committed complete or invalid run, from its cache; never calls the client (LX.8). A moved prompt
             (a cache miss) or a replay whose summary differs from the committed one marks it not replayable; a
             stopped or incomplete run, and the one-call smoke record, are skipped, their status left as it is (P1).
    Spec:    docs/plan_v3_llm_agent_experiment.md#6
    Tests:   tests/experiments/llm_agents/test_cache.py::test_lx8_replay_leaves_unfinished_runs_resumable,
             tests/experiments/llm_agents/test_cache.py::test_lx8_replay_that_differs_is_not_replayable
    """
    cfg = cfg or load_llm()
    outcomes = []
    for path in sorted(runs.glob("*/run.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["status"] not in REPLAYED or rec["phase"] == "smoke":
            # a stopped or incomplete run never fetched its later weeks; replaying it would miss there and call a
            # resumable status "not_replayable" (P1). A live rerun resumes it instead.
            log.info("replay: %s skipped, status %s", rec["run_id"], rec["status"])
            continue
        phase = cfg.phases[rec["phase"]]
        phase = Phase(phase.name, phase.arms, phase.seeds, rec["weeks_planned"], phase.model, phase.pool, phase.calls)
        out = run_one(phase, rec["arm"], rec["seed"], cfg=cfg, live=False, runs=runs, write_record=False)
        detail = out.record["status_detail"] if out.record["status"] == "not_replayable" else None
        committed = path.parent / "summary.json"
        if detail is None and out.summary is not None and committed.exists():
            was = json.loads(committed.read_text(encoding="utf-8"))["summary_hash"]
            if out.summary["summary_hash"] != was:
                moved = [k for k in ("engine_hash", "code_hash", "template_hash", "label_hash", "config_hash")
                         if rec.get(k) != out.record[k]]
                detail = (f"the replay's summary {out.summary['summary_hash']} differs from the committed {was}; "
                          f"changed since the run: {', '.join(moved) or 'no recorded hash'}")
        if detail is not None:
            log.warning("replay: %s not replayable: %s", rec["run_id"], detail)
            rec["status"], rec["status_detail"] = "not_replayable", detail
            run_record.write(rec, path.parent, runs)
        outcomes.append(out)
    return outcomes


def smoke(cfg: LLMConfig, transport_factory=None, runs: Path = files.RUNS) -> dict:
    """LX.L1: one call, the first family member's first round of A1 seed 1, cached and recorded (owner-approved)."""
    phase = cfg.phases["smoke"]
    by_pool = spent(runs, cfg)
    already = spent(runs, cfg, by="phase").get("smoke", 0.0)
    refuse_over_cap(estimate(phase, cfg, members(cfg)), by_pool.get(phase.pool, 0.0) - already,
                    sum(by_pool.values()) - already)
    transport = (transport_factory or AnthropicTransport)(cfg)
    transport.check_models(cfg.models())
    first = {}

    class Once:
        def choose(self, week, token, prompt):
            first.setdefault("call", (week, token, prompt))
            from experiments.llm_agents.source import Answer
            return Answer(prompt.ids[0])

    step(build(phase.arms[0], phase.seeds[0], phase.weeks, Once()))
    week, token, prompt = first["call"]
    rid = run_id("smoke", phase.arms[0], phase.seeds[0])
    cache = ResponseCache.load(runs / rid / "responses.jsonl.gz", runs)
    chooser = CachedChooser(run_id=rid, model=phase.model, arm=phase.arms[0], seed=phase.seeds[0], cfg=cfg,
                            cache=cache, reask_text=load_text()["reask"], transport=transport,
                            budget=Budget(cfg.cap(phase.pool), cfg.number("cap_total_usd"),
                                          by_pool.get(phase.pool, 0.0), sum(by_pool.values())))
    answer = chooser.choose(week, token, prompt)
    cache.save()
    entry = cache.get(chooser.keys[(week, token)])
    rec = run_record.make(
        run_id=rid, phase="smoke", pool=phase.pool, arm=phase.arms[0], seed=phase.seeds[0], family=cfg.get("family"),
        weeks_planned=phase.weeks, weeks_completed=0, model=phase.model, models_returned=[entry.get("model")],
        api_version=transport.api_version, sdk_version=transport.sdk_version, effort=cfg.effort(phase.model),
        thinking=cfg.thinking(phase.model), max_tokens=cfg.integer("max_tokens"), permutations=None,
        status="complete" if answer.choice else "invalid", status_detail=f"one call; parsed choice {answer.choice!r}",
        counts=chooser.counts.as_record(), selections=1, fallback_share=0.0 if answer.choice else 1.0)
    run_record.write(rec, runs / rid, runs)
    return rec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m experiments.llm_agents.runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("estimate").add_argument("phase")
    r = sub.add_parser("run")
    r.add_argument("phase")
    r.add_argument("--live", action="store_true", help="call the API (owner's go-ahead only); otherwise refused")
    s = sub.add_parser("smoke")
    s.add_argument("--live", action="store_true")
    sub.add_parser("replay")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_llm()
    if args.command == "estimate":
        est = estimate(cfg.phases[args.phase], cfg, members(cfg))
        print(json.dumps(est, indent=2))
        return 1 if est["over"] else 0
    if args.command == "run":
        for out in run_phase(args.phase, live=args.live, cfg=cfg):
            print(out.record["run_id"], out.record["status"], out.record["counts"]["cost"])
        return 0
    if args.command == "smoke":
        if not args.live:
            raise SystemExit("the smoke call is live: pass --live, on the owner's go-ahead")
        print(json.dumps(smoke(cfg), indent=2))
        return 0
    if args.command == "replay":
        for out in replay_all(cfg=cfg):
            print(out.record["run_id"], out.record["status"])
        return 0
    if args.command == "report":
        from experiments.llm_agents.report import write_report

        print(write_report())
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
