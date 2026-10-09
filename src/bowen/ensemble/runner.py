"""The ensemble runner — paired arms on keyed seeds, stopped adaptively (plan D7, D8).

Purpose: run a criterion's two arms on the same seeds in parallel processes, add seeds in
         blocks until the per-seed difference's interval is tight or the cap is reached, and
         return a verdict — PASS, FAIL or UNDETERMINED — with everything the record needs.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.2, #M11.1f, #M11.4a, #M11.4d, #M11.4e, #M11.D.18, #M13.4, #M17.A.1, #M17.A.3, #M17.A.4
Tests:   tests/bowen/test_ensemble.py

Arms are coupled through keyed draws (`M3.D.4a`): the same seed in both arms draws the same
number for the same event. A criterion is a ``Criterion``: an ``arm`` function, importable at
module level so worker processes can run it, returning an ``ArmResult`` per (arm, seed).

* **Direction.** For each readout the per-seed difference ``treatment − baseline`` is tested
  one-sided in the declared direction by ``stats.signed_rank``, differences below the margin
  counting as zero (`M17.A.4`), Holm-corrected across the readouts (`M11.4e`).
* **Stopping** (`M17.A.1`, `M13.4`). Seeds come in blocks of ``ensemble_block``. After each
  block, every readout's interval half-width, ``1.96 × sd(d) / √n``, is compared with
  ``ensemble_precision`` times the **pooled** seed-to-seed sd of the two arms,
  ``√((sd_baseline² + sd_treatment²) / 2)``. All within: stop. At ``ensemble_cap`` without that:
  **UNDETERMINED**, which does not pass. *Decided 2026-10-08* (``docs/phase_c_completion_report.md``
  §10): the baseline arm alone is the wrong ruler when the arms differ in spread, and a ruler must
  not depend on which arm is called baseline; the differences' own sd would make the rule a fixed
  seed count, since half-width over its own sd is 1.96/√n.
* **Margin** (and a null's equivalence bound) is relative to the baseline arm's seed-to-seed sd
  (plan D8), so it is unit-free. A scale with no spread falls back to the differences' sd.
* **A null** (`M11.4a`) passes only if the whole interval lies within ``equivalence_margin``
  baseline sds of zero; it reports the interval and the n reached.
* **A required move that never occurs** fails the criterion (`M11.1f`).
* **Assay limit** (`M11.4d`): the baseline's mean position in the readout's attainable range
  is reported; a null whose baseline sits at a bound is reported as an assay limit.
* **Fallback rate** (`M11.D.18`) per person and per kind is reported; a PASS whose overall
  rate exceeds ``fallback_flag_rate`` is flagged.
"""

from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.bowen.ensemble.stats import holm, signed_rank

PASS, FAIL, UNDETERMINED = "PASS", "FAIL", "UNDETERMINED"


@dataclass(frozen=True)
class Readout:
    name: str
    direction: int                      # +1: treatment > baseline; -1: treatment < baseline; 0: a null
    bounds: tuple[float, float] = (-math.inf, math.inf)
    report_only: bool = False           # reported beside the verdict, never tested (e.g. M11.C.16's top share)


@dataclass
class ArmResult:
    readouts: dict[str, float]
    moves: Counter = field(default_factory=Counter)
    fallbacks: Counter = field(default_factory=Counter)    # person -> fallback selections
    selections: Counter = field(default_factory=Counter)   # person -> all selections


@dataclass(frozen=True)
class Criterion:
    id: str
    cls: str                            # M11.5: premise, composite, check, mixed
    arms: tuple[str, str]               # (baseline, treatment)
    readouts: tuple[Readout, ...]
    arm: Callable[[str, int, dict], ArmResult]
    requires_moves: tuple[str, ...] = ()
    settings: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class Verdict:
    criterion: str
    cls: str
    outcome: str
    seeds: int
    readouts: list[dict]
    fallback_rate: float
    fallback_by_person: dict
    flagged: list[str]
    missing_moves: list[str]
    scripted: dict = field(default_factory=dict)  # arm -> {"made", "not legal", "not selecting"}, over all seeds


def _run_seed(args):
    criterion, seed = args
    base = criterion.arm(criterion.arms[0], seed, criterion.settings)
    treat = criterion.arm(criterion.arms[1], seed, criterion.settings)
    return seed, base, treat


def run_criterion(criterion: Criterion, rules: dict, workers: int = 1, first_seed: int = 0) -> Verdict:
    """Purpose: run one criterion adaptively and return its verdict.
    Spec:    docs/bowen_agent_model_spec_v2.md#M17.A.1, #M13.4, #M11.4e, #M17.A.4, #M11.1f, #M11.D.18
    Tests:   tests/bowen/test_ensemble.py::test_m17a1_undetermined_at_cap_does_not_pass
    """
    results = []
    seed = first_seed
    converged = False
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        while len(results) < rules["ensemble_cap"]:
            block = [(criterion, s) for s in range(seed, seed + rules["ensemble_block"])]
            seed += rules["ensemble_block"]
            results += list(pool.map(_run_seed, block)) if pool else [_run_seed(b) for b in block]
            converged = _converged(criterion, results, rules)
            if converged:
                break
    finally:
        if pool:
            pool.shutdown()
    return _verdict(criterion, results, converged, rules)


def _arrays(criterion, results, name):
    base = np.array([b.readouts[name] for _, b, _ in results], dtype=float)
    treat = np.array([t.readouts[name] for _, _, t in results], dtype=float)
    return base, treat


def _pooled_scale(base, treat, diff) -> float:
    """The precision ruler: the pooled seed-to-seed sd of the two arms (decided 2026-10-08)."""
    if len(base) < 2:
        return 0.0
    sd = math.sqrt((float(np.var(base, ddof=1)) + float(np.var(treat, ddof=1))) / 2)
    if sd == 0.0:
        sd = float(np.std(diff, ddof=1))
    return sd


def _scale(base, diff) -> float:
    sd = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
    if sd == 0.0:
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return sd


def _half_width(diff) -> float:
    return 1.96 * float(np.std(diff, ddof=1)) / math.sqrt(len(diff)) if len(diff) > 1 else math.inf


def _converged(criterion, results, rules) -> bool:
    for readout in criterion.readouts:
        if readout.report_only:
            continue
        base, treat = _arrays(criterion, results, readout.name)
        diff = treat - base
        scale = _pooled_scale(base, treat, diff)
        if scale == 0.0:
            continue  # no spread anywhere: nothing more to learn
        if _half_width(diff) >= rules["ensemble_precision"] * scale:
            return False
    return True


def _verdict(criterion, results, converged, rules) -> Verdict:
    rows, p_values, directional = [], [], []
    for readout in criterion.readouts:
        base, treat = _arrays(criterion, results, readout.name)
        diff = treat - base
        scale = _scale(base, diff)
        lo, hi = readout.bounds
        position = None
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            position = float((np.mean(base) - lo) / (hi - lo))
        row = {"readout": readout.name, "direction": readout.direction, "mean_difference": float(np.mean(diff)),
               "half_width": _half_width(diff), "baseline_sd": scale, "pooled_sd": _pooled_scale(base, treat, diff),
               "baseline_position": position}
        if readout.report_only:
            row["report_only"] = True
        elif readout.direction == 0:
            row["equivalence_bound"] = rules["equivalence_margin"] * scale
            row["within_bound"] = abs(row["mean_difference"]) + row["half_width"] <= row["equivalence_bound"]
            row["assay_limit"] = position is not None and (position <= 0.01 or position >= 0.99)
        else:
            w, n, p = signed_rank(readout.direction * diff, margin=rules["ensemble_margin"] * scale)
            row.update({"w_plus": w, "n_nonzero": n, "p": p})
            p_values.append(p)
            directional.append(row)
        rows.append(row)
    rejected = holm(p_values, rules["ensemble_alpha"]) if p_values else []
    for row, rej in zip(directional, rejected):
        row["holds"] = rej
    moves = Counter()
    fallbacks, selections = Counter(), Counter()
    for _, b, t in results:
        moves.update(t.moves)
        fallbacks.update(b.fallbacks); fallbacks.update(t.fallbacks)
        selections.update(b.selections); selections.update(t.selections)
    missing = [m for m in criterion.requires_moves if moves[m] == 0]
    rate = sum(fallbacks.values()) / max(1, sum(selections.values()))
    by_person = {str(p): fallbacks[p] / selections[p] for p in sorted(selections, key=str) if selections[p]}
    if missing:
        outcome = FAIL  # M11.1f
    elif not converged:
        outcome = UNDETERMINED
    else:
        held = all(r.get("holds", False) for r in directional) and all(
            r.get("within_bound", True) for r in rows if r["direction"] == 0 and not r.get("report_only"))
        outcome = PASS if held else FAIL
    flagged = []
    if outcome == PASS and rate > rules["fallback_flag_rate"]:
        flagged.append(f"fallback rate {rate:.2f} above {rules['fallback_flag_rate']}")
    for r in rows:
        if r["direction"] == 0 and r.get("assay_limit"):
            flagged.append(f"{r['readout']}: baseline at a bound — an assay limit, not evidence (M11.4d)")
    return Verdict(criterion.id, criterion.cls, outcome, len(results), rows, rate, by_person, flagged, missing,
                   scripted_counts(criterion, results))


SCRIPTED = {"made": "(scripted act made)", "not legal": "(scripted act not legal, skipped)",
            "not selecting": "(scripted act not made, actor not selecting)"}


def scripted_counts(criterion: Criterion, results) -> dict:
    """Purpose: each arm's scripted acts over all seeds, made or not, so a skip rate is reported beside the verdict
             (a skipped seed adds a difference of exactly 0; step S of docs/DECISIONS — PHASE C FAILING.md).
    Spec:    docs/bowen_agent_model_spec_v2.md#M17.D.3
    Tests:   tests/bowen/test_ensemble.py::test_m17d3_verdict_reports_scripted_acts_not_made
    """
    counts = {}
    for name, index in zip(criterion.arms, (1, 2)):
        totals = {k: sum(r[index].moves[label] for r in results) for k, label in SCRIPTED.items()}
        if any(totals.values()):
            counts[name] = totals
    return counts
