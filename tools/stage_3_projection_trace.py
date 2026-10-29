"""Trace each term of `M7.E.1f`'s development step for `M11.C.6`'s child, in both arms: a measurement, never a verdict.

Purpose: stage 3 step 2a (`docs/implementation_plan_version_2_stage_3.md` §3.3.1, owner decision D18). For the
         declared child (Nadia) in both of `M11.C.6`'s arms, ages 5–20, at each yearly development step: the child's
         own term and the living parents' term of `d` (`src/bowen/engine/life_course.py::_childhood`), what each
         term's year-mean is made of (raw functional level, pseudo-self, the system's adjustment), each parent's
         borrowed pseudo-self (``max(0, pseudo_self)``), the child's load, and the weeks of coping and soothing
         (`M1.A.5e`). It answers one question: is the parents' term what lifts the child under projection? It writes
         `docs/review/stage_3_projection_trace.md` and keeps that page's hand-written "Reading" section.
Spec:    docs/bowen_agent_model_spec_v2.md#M7.E.1f, #M11.C.6, #M1.A.5e
Tests:   tests/test_stage_3_probes.py::test_stage_3_trace_reproduces_the_development_step

    python3 tools/stage_3_projection_trace.py [--seeds 150] [--workers 10]

The tool is in no ``HASHED_TOOLS`` list and no record reads it, so it moves no record. It runs `M11.C.6`'s own arm
function (``criteria_d2.arm_c6``) and chains a read-only ``watch`` onto the arm's own by patching
``criteria_d2.scenario`` for the call, restored after. The watch runs after each tick, when the week's history has been
appended and the slow tick and the life course have run, so the history it reads is the one the step read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "docs" / "review" / "stage_3_projection_trace.md"
ARMS = ("baseline", "treatment")
READING = "## Reading"
# Each step's fields: the two terms of d, and what each year-mean is made of.
STEP_FIELDS = ("own_term", "parents_term", "parents_term_p1", "child_fl", "child_pseudo", "child_adjust", "child_stress",
               "parents_fl", "parents_pseudo", "parents_adjust", "parents_borrowed", "child_load", "coping_weeks",
               "soothing_weeks", "step", "clamped")


def one_seed(seed: int) -> dict:
    """Both arms of one seed: ``{arm: {"development": float, "steps": {age: {field: value}}}}``."""
    from src.bowen.engine import life_course
    from src.bowen.engine.identifiers import PersonId
    from src.bowen.ensemble import criteria_d2
    from src.bowen.ensemble.criteria import SETTINGS
    from src.bowen.ensemble.criteria_d import ENGINE

    settings = SETTINGS["M11.C.6"]
    child = PersonId(settings["child"])
    out = {}
    for arm in ARMS:
        steps, weekly, last_basic = {}, {}, {}

        def watch(state):
            person = state.people[child]
            parents = life_course._living_parents(state, child)
            for pid in (child, *parents):
                p = state.people[pid]
                weekly.setdefault(pid, []).append((p.functional_level, p.pseudo_self or 0.0,
                                                   max(0.0, p.adjust_swing), p.stress_swing))
                del weekly[pid][:-ENGINE.slow_tick_fast_ticks]
            age = person.age_years
            before = last_basic.get("basic", person.basic_level)
            last_basic["basic"] = person.basic_level
            # The step ran this tick only at the slow tick, for an age the rule develops after the anchor year.
            if state.tick % ENGINE.slow_tick_fast_ticks or age is None:
                return
            if not (ENGINE.development_start_age_years < age <= ENGINE.development_end_age_years):
                return
            if not person.alive:
                return
            history = list(state.estimator_history.get(child, ()))[-ENGINE.slow_tick_fast_ticks:]
            own = math.fsum(w.functional_level for w in history) / len(history)
            theirs = [m for p in parents
                      if (m := life_course._year_mean(state, p, ENGINE,  # revision 41 (T-131): net of displaced coping
                                                     lambda w: w.functional_level - w.displaced)) is not None]
            parents_basic = life_course._parents_mean(state, child, "basic_level")
            row = {"own_term": own - before,
                   "parents_term": (math.fsum(theirs) / len(theirs) - parents_basic) if theirs else math.nan,
                   "child_load": statistics.fmean(w.load for w in history),
                   "coping_weeks": sum(1 for w in history if w.self_directed),
                   "soothing_weeks": sum(1 for w in history if w.soothed),
                   "step": person.basic_level - before}
            terms = [row["own_term"]] + ([row["parents_term"]] if theirs else [])
            predicted = ENGINE.development_rate * math.fsum(terms) / len(terms)
            row["clamped"] = float(abs(row["step"] - predicted) > 1e-9)
            mean = lambda pid, i: statistics.fmean(w[i] for w in weekly[pid])  # noqa: E731
            row["child_fl"], row["child_pseudo"], row["child_adjust"], row["child_stress"] = (mean(child, i)
                                                                                         for i in range(4))
            row["parents_fl"] = statistics.fmean(mean(p, 0) for p in parents) if parents else math.nan
            row["parents_pseudo"] = statistics.fmean(mean(p, 1) for p in parents) if parents else math.nan
            row["parents_adjust"] = statistics.fmean(mean(p, 2) for p in parents) if parents else math.nan
            row["parents_borrowed"] = (statistics.fmean(statistics.fmean(max(0.0, w[1]) for w in weekly[p])
                                                        for p in parents) if parents else math.nan)
            # P1 as §3.3.1 words it, read on this run's weeks: each parent's functional level less the pseudo-self
            # they hold borrowed, against their basic level. A first-order estimate: the run is not re-run under it.
            row["parents_term_p1"] = ((row["parents_fl"] - row["parents_borrowed"] - parents_basic) if parents
                                      else math.nan)
            steps[age] = row

        original = criteria_d2.scenario

        def chained(*args, watch=None, **kwargs):
            def both(state):
                if watch:
                    watch(state)
                trace(state)
            return original(*args, watch=both, **kwargs)

        trace = watch
        criteria_d2.scenario = chained
        try:
            res = criteria_d2.arm_c6(arm, seed, settings)
        finally:
            criteria_d2.scenario = original
        out[arm] = {"development": res.readouts["development"], "acts": res.readouts["projection_acts"],
                    "steps": steps}
    return {"seed": seed, **out}


def paired(values: list[float]) -> tuple[float, float, int]:
    """Mean of paired differences and its 95% half-width, ``1.96 × sd / √n`` (the runner's, `M17.A.1`)."""
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return (vals[0] if vals else math.nan), math.inf, len(vals)
    return statistics.fmean(vals), 1.96 * statistics.stdev(vals) / math.sqrt(len(vals)), len(vals)


def summarise(rows: list[dict], rate: float) -> dict:
    """The paired treatment − baseline difference of each term's contribution to development, summed over the steps,
    and per age. A term's contribution at one step is ``rate × term ÷ (number of terms)``, so the two contributions
    add to the unclamped step (``M7.E.1f``)."""
    ages = sorted({a for r in rows for arm in ARMS for a in r[arm]["steps"]})

    def contribution(steps, name):
        total = 0.0
        for row in steps.values():
            n = 1 if math.isnan(row["parents_term"]) else 2
            total += 0.0 if math.isnan(row[name]) else rate * row[name] / n
        return total

    totals = {}
    for name in ("own_term", "parents_term", "parents_term_p1"):
        totals[name] = paired([contribution(r["treatment"]["steps"], name) - contribution(r["baseline"]["steps"], name)
                               for r in rows])
    totals["development"] = paired([r["treatment"]["development"] - r["baseline"]["development"] for r in rows])
    totals["steps_sum"] = paired([sum(s["step"] for s in r["treatment"]["steps"].values())
                                  - sum(s["step"] for s in r["baseline"]["steps"].values()) for r in rows])
    totals["acts"] = paired([r["treatment"]["acts"] - r["baseline"]["acts"] for r in rows])
    clamps = {arm: sum(s["clamped"] for r in rows for s in r[arm]["steps"].values()) for arm in ARMS}
    n_steps = {arm: sum(len(r[arm]["steps"]) for r in rows) for arm in ARMS}
    by_age = {}
    for age in ages:
        both = [r for r in rows if age in r["baseline"]["steps"] and age in r["treatment"]["steps"]]
        by_age[age] = {f: paired([r["treatment"]["steps"][age][f] - r["baseline"]["steps"][age][f] for r in both])
                       for f in STEP_FIELDS if f != "clamped"}
        by_age[age]["baseline"] = {f: statistics.fmean(r["baseline"]["steps"][age][f] for r in both)
                                   for f in ("own_term", "parents_term", "child_load", "parents_borrowed")}
    means = {arm: {f: statistics.fmean(s[f] for r in rows for s in r[arm]["steps"].values()
                                       if not math.isnan(s[f]))
                   for f in STEP_FIELDS if f != "clamped"} for arm in ARMS}
    return {"totals": totals, "clamps": clamps, "n_steps": n_steps, "by_age": by_age, "means": means,
            "seeds": len(rows)}


def _fmt(t) -> str:
    mean, half, n = t
    return f"{mean:+.4f} ± {half:.4f}"


def render(summary: dict, engine: str, when: str, seeds: int, keep: str) -> str:
    t = summary["totals"]
    own, par = t["own_term"][0], t["parents_term"][0]
    total = own + par
    share = par / total if total else math.nan
    lines = [
        "# Stage 3 step 2a — the projection trace (`M11.C.6`, `M7.E.1f`)",
        "",
        "Generated by `tools/stage_3_projection_trace.py`; do not edit above the \"Reading\" section, which the tool",
        f"keeps. Engine hash `{engine}`, {when}, seeds 0–{seeds - 1} ({seeds}, `M11.C.6`'s seeds in the Phase D part II",
        "record). Both arms of `M11.C.6` (`src/bowen/ensemble/criteria_d2.py::arm_c6`): treatment, the parents act as the",
        "policy chooses; baseline, their projection-type acts toward Nadia withheld. Each step is the yearly development",
        "step of `src/bowen/engine/life_course.py::_childhood` at ages 6–20 (age 5 sets the anchor). Every difference is",
        "treatment − baseline on the same seed, mean ± 1.96 × sd ÷ √n. Not a real family (`M11.F.9`).",
        "",
        "**What the step reads.** `d` is the mean of the child's own term (the year-mean of the estimator's functional",
        "level less the child's basic level) and the living parents' term (their mean year-mean less their mean basic",
        "level). The estimator's functional level is `functional_level − pseudo_self − max(0, adjust_swing)`",
        "(`src/bowen/engine/slow_tick.py::record_week`, revision 23): both terms already leave out pseudo-self, borrowed",
        "and lent, with its sign. The columns below split each year-mean into those three parts.",
        "",
        "## The answer: which term carries the difference",
        "",
        "| Quantity, summed over the steps | Treatment − baseline |",
        "|---|---|",
        f"| `development` (basic level at 20 less the anchor, the criterion's readout) | {_fmt(t['development'])} |",
        f"| the steps' sum (equals `development` unless a child died) | {_fmt(t['steps_sum'])} |",
        f"| the child's own term's contribution, `rate × own ÷ 2` | {_fmt(t['own_term'])} |",
        f"| the parents' term's contribution, `rate × parents ÷ 2` | {_fmt(t['parents_term'])} |",
        f"| P1's parents' term on the same weeks, first-order (no re-run): `rate × P1 ÷ 2` | {_fmt(t['parents_term_p1'])} |",
        f"| projection-type acts toward the child (report only) | {_fmt(t['acts'])} |",
        "",
        f"The parents' term carries {share:.0%} of the summed difference of the two contributions "
        f"({par:+.4f} of {total:+.4f}). Steps the band bound: {summary['clamps']['baseline']:.0f} of "
        f"{summary['n_steps']['baseline']} in the baseline, {summary['clamps']['treatment']:.0f} of "
        f"{summary['n_steps']['treatment']} in the treatment.",
        "",
        "## Each arm's mean over every step",
        "",
        "| Field | Baseline | Treatment |",
        "|---|---|---|",
    ]
    labels = {"own_term": "child's own term", "parents_term": "parents' term",
              "parents_term_p1": "P1's parents' term (first-order estimate)",
              "child_fl": "child's raw functional level", "child_pseudo": "child's pseudo-self (signed)",
              "child_adjust": "child's adjustment, `max(0, adjust_swing)`",
              "child_stress": "child's stress swing (`M1.A.5e`'s coping and soothing, in raw functional level)",
              "parents_fl": "parents' raw functional level", "parents_pseudo": "parents' pseudo-self (signed)",
              "parents_adjust": "parents' adjustment", "parents_borrowed": "parents' borrowed, `max(0, pseudo_self)`",
              "child_load": "child's load (acute above chronic, at selection)",
              "coping_weeks": "child's weeks coping (self-directed choice, `M1.A.5e`)",
              "soothing_weeks": "child's weeks soothing", "step": "the step"}
    for f, label in labels.items():
        m = summary["means"]
        lines.append(f"| {label} | {m['baseline'][f]:.4f} | {m['treatment'][f]:.4f} |")
    lines += ["", "## By age: treatment − baseline", "",
              "| Age | own term | parents' term | child's raw FL | child's pseudo-self | child's stress swing "
              "| parents' borrowed | child's load | coping weeks | soothing weeks |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for age, row in summary["by_age"].items():
        cells = [_fmt(row[f]) for f in ("own_term", "parents_term", "child_fl", "child_pseudo", "child_stress",
                                        "parents_borrowed", "child_load", "coping_weeks", "soothing_weeks")]
        lines.append(f"| {age} | " + " | ".join(cells) + " |")
    lines += ["", keep.rstrip() if keep else f"{READING}\n\n(to be written)", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=150)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    from src.bowen.ensemble.criteria_d import ENGINE
    from tools.record_cache import engine_hash

    seeds = range(args.seeds)
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(one_seed, seeds))
    else:
        rows = [one_seed(s) for s in seeds]
    summary = summarise(rows, ENGINE.development_rate)
    old = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
    keep = old[old.index(READING):] if READING in old else ""
    args.out.write_text(render(summary, engine_hash()[:12], dt.date.today().isoformat(), args.seeds, keep),
                        encoding="utf-8")
    print(f"wrote {args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
