"""The hard budget cap: the estimate before a phase and the worst case before each round.

Purpose: refuse a phase whose estimate is over its cap, and stop a run before the round whose worst case would cross
         it, saying what was dropped (P9); cost comes from each response's ``usage`` and the price table.
Spec:    docs/plan_v3_llm_agent_experiment.md#5
Tests:   tests/experiments/llm_agents/test_budget.py::test_lx11_estimate_over_cap_refuses,
         tests/experiments/llm_agents/test_budget.py::test_lx11_stops_before_crossing
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from experiments.llm_agents import files
from experiments.llm_agents.cache import ResponseCache
from experiments.llm_agents.config import LLMConfig, Phase, load_llm
from experiments.llm_agents.prompts import schema


class BudgetStop(RuntimeError):
    """The next round's worst case would cross the cap; the run stops before it."""

    def __init__(self, week: int, spent: float, worst: float, limit: float, which: str = "pool"):
        super().__init__(f"week {week}: spent ${spent:.4f} + worst case ${worst:.4f} would cross the {which} cap "
                         f"${limit:.2f}")
        self.week, self.spent, self.worst, self.limit, self.which = week, spent, worst, limit, which


class OverCap(RuntimeError):
    """A phase's estimate is over its cap; it does not start."""


def call_cost(usage: dict, prices: tuple[float, float]) -> float:
    pin, pout = prices
    inputs = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) \
        + usage.get("cache_read_input_tokens", 0)
    return (inputs * pin + usage.get("output_tokens", 0) * pout) / 1_000_000


def call_bounds(prompts: Sequence, cfg: LLMConfig, model: str, reask_text: str) -> list[float]:
    """Each call's bound from its text and its output schema (input too): input at ``worst_case_chars_per_token``,
    output at ``max_tokens`` (§5)."""
    pin, pout = cfg.prices(model)
    cpt, out = cfg.number("worst_case_chars_per_token"), cfg.integer("max_tokens")
    return [((len(p.system) + len(p.user) + len(reask_text) + len(json.dumps(schema(p.ids)))) / cpt * pin
             + out * pout) / 1_000_000 for p in prompts]


def worst_case(prompts: Sequence, cfg: LLMConfig, model: str, reask_text: str, largest_seen: float = 0.0) -> float:
    """The round's worst case: every call at the larger of its text bound and the costliest call seen so far (a
    tokenizer can beat the characters-per-token bound), and every call counted twice, for a possible re-ask (§5)."""
    return sum(2 * max(b, largest_seen) for b in call_bounds(prompts, cfg, model, reask_text))


def phase_calls(phase: Phase, members: int) -> int:
    return len(phase.arms) * len(phase.seeds) * phase.weeks * members if phase.calls is None else phase.calls


def estimate(phase: Phase, cfg: LLMConfig, members: int) -> dict:
    """Purpose: calls × tokens × price for a phase, against its cap (§5)."""
    if phase.model is None:
        return {"phase": phase.name, "calls": 0, "cost": 0.0, "cap": None, "over": False}
    pin, pout = cfg.prices(phase.model)
    calls = phase_calls(phase, members)
    per = (cfg.number("estimate_input_tokens") * pin + cfg.number("estimate_output_tokens") * pout) / 1_000_000
    cost = calls * per
    cap = min(cfg.cap(phase.pool), cfg.number("cap_total_usd"))
    return {"phase": phase.name, "calls": calls, "per_call": per, "cost": cost, "cap": cap,
            "pool_cap": cfg.cap(phase.pool), "total_cap": cfg.number("cap_total_usd"), "over": cost > cap}


def refuse_over_cap(est: dict, spent: float = 0.0, total_spent: float = 0.0) -> None:
    """Purpose: refuse a phase whose estimate, on top of what was spent, crosses its pool's cap or the total cap.
    Spec:    docs/plan_v3_llm_agent_experiment.md#5
    Tests:   tests/experiments/llm_agents/test_budget.py::test_lx11_estimate_over_cap_refuses,
             tests/experiments/llm_agents/test_budget.py::test_lx11_total_cap_refuses_before_starting
    """
    if est["cap"] is None:
        return
    for name, paid, cap in (("pool", spent, est["pool_cap"]), ("total", total_spent, est["total_cap"])):
        if paid + est["cost"] > cap:
            raise OverCap(f"phase {est['phase']}: spent ${paid:.2f} + estimate ${est['cost']:.2f} "
                          f"({est['calls']} calls) is over the {name} cap ${cap:.2f}; not started")


def spent(runs: Path = files.RUNS, cfg: LLMConfig | None = None, by: str = "pool") -> dict[str, float]:
    """Purpose: what has been paid, by pool (or by phase), so caps hold across sessions (§5).
    Spec:    docs/plan_v3_llm_agent_experiment.md#5
    Tests:   tests/experiments/llm_agents/test_budget.py::test_lx11_spent_counts_every_cached_response

    It is summed from the response caches, every entry ever paid for, not from the run records: a run that crashed
    before its record was written, or a rerun whose changed prompt left older entries untouched, still counts.
    """
    cfg = cfg or load_llm()
    out: dict[str, float] = {}
    for path in sorted(runs.glob("*/responses.jsonl.gz")):
        phase_name = path.parent.name.split("-", 1)[0]
        record = path.parent / "run.json"
        pool = json.loads(record.read_text(encoding="utf-8")).get("pool") if record.exists() else None
        if pool is None:
            phase = cfg.phases.get(phase_name)
            pool = phase.pool if phase is not None else None
        if pool is None:
            raise ValueError(f"{path}: no pool for its phase {phase_name!r}; its spend cannot be counted")
        cost = sum(float(e.get("cost", 0.0)) for e in ResponseCache.load(path, runs).entries.values())
        name = pool if by == "pool" else phase_name
        out[name] = out.get(name, 0.0) + cost
    return out


@dataclass
class Budget:
    """The money one phase may still spend: its pool's cap less what the pool has spent, within the total cap."""

    pool_cap: float
    total_cap: float
    pool_spent: float = 0.0
    total_spent: float = 0.0
    run_spent: float = 0.0
    largest_call: float = 0.0

    def __post_init__(self):
        self._lock = threading.Lock()

    @property
    def limit(self) -> float:
        return min(self.pool_cap - self.pool_spent, self.total_cap - self.total_spent)

    def add(self, cost: float) -> None:
        with self._lock:
            self.pool_spent += cost
            self.total_spent += cost
            self.run_spent += cost
            self.largest_call = max(self.largest_call, cost)

    def check(self, week: int, worst: float) -> None:
        with self._lock:
            if worst > self.limit:
                if self.pool_cap - self.pool_spent <= self.total_cap - self.total_spent:
                    raise BudgetStop(week, self.pool_spent, worst, self.pool_cap, "pool")
                raise BudgetStop(week, self.total_spent, worst, self.total_cap, "total")
