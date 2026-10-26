"""Probe one genuine `I-POSITION` from a level-35 and a level-60 mover: a measurement, never a verdict.

Purpose: stage 3 step 2b (`docs/implementation_plan_version_2_stage_3.md` §3.3.2, owner decision D20). On the triad
         fixture, mover F, target M, third C, the same seeds: F's basic and functional level set to 35 or 60 (what
         initialisation derives from level derived again, as `M11.C.32`'s arm does) and F's systems perspective 1.0,
         so the act starts as the genuine sequence unless F is angry (`M5.F.4`, `M5.D.4a`). F's `I-POSITION` toward M
         is scripted at ``T0``; the declared spell (`config/bowen/criteria.md`, the `spell` rows) runs on both parents,
         with no spell event from ``T0`` to the end of the PREPARE window (``T0 + prepare_ticks``). Each level also
         runs a control arm on the same seed with nothing scripted, so the act's own effect at each level is
         separated from what level does to the family anyway. Read over the six weeks after the act's week: M's
         acute anxiety, M's felt impingement on the F–M tie, the family's summed excess, the `CONFLICT` share of the
         moves on the F–M tie; over the run: completions. It writes `docs/review/stage_3_iposition_probe.md` and keeps
         that page's hand-written "Reading" section.
Spec:    docs/bowen_agent_model_spec_v2.md#M5.F.4, #M5.F.2c, #M11.C.29
Tests:   tests/test_stage_3_probes.py::test_stage_3_probe_arms_differ_only_in_the_movers_level

    python3 tools/stage_3_iposition_probe.py [--seeds 200] [--workers 10]

The tool is in no ``HASHED_TOOLS`` list and no record reads it, so it moves no record. The arm design is `[I]`, this
probe's, declared here before it ran.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "docs" / "review" / "stage_3_iposition_probe.md"
READING = "## Reading"
LEVELS = (35.0, 60.0)
# The act's week. Not `M11.C.29`'s 4: measured on seeds 0–39 with no spell, F was angry on the F–M tie at week 4 in 14
# (level 35) and 13 (level 60) seeds, so the act would run as the assertion form, and over `defence_threshold` in 6 at
# level 35, so PREPARE would fail; at week 1, in none at either level. Week 0 is not used (config/bowen/criteria.md:
# every member starts on their chronic floor).
T0 = 1
WINDOW = 6       # weeks read after the act's week
AFTER = 24       # weeks run after T0, so a sequence that defines on time can finish its hold window
READOUTS = ("target_acute", "target_impingement", "family_excess", "conflict_share", "completions")


def one_arm(level: float, scripted: bool, seed: int) -> dict:
    from src.bowen.engine.contact import excess
    from src.bowen.engine.events import Mechanism
    from src.bowen.engine.identifiers import PersonId as P, TieId
    from src.bowen.engine.log_records import EffectRecord, EmittedRecord
    from src.bowen.ensemble.criteria import _reinitialise, act, held, scenario, spell
    from src.bowen.ensemble.criteria_d import ENGINE

    f, m = P("f"), P("m")
    prepare_end = T0 + ENGINE.prepare_ticks
    weeks = T0 + AFTER
    window = range(T0 + 1, T0 + 1 + WINDOW)
    spells = tuple(e for e in spell("triad", weeks) if not T0 <= e.timestamp <= prepare_end)

    def setup(state):
        mover = state.people[f]
        mover.basic_level = mover.functional_level = level
        mover.pseudo_self = mover.swing
        _reinitialise(state)
        mover.systems_perspective = 1.0

    tie = TieId.of(f, m)
    per_week = []

    def watch(state):  # after week state.tick − 1
        t = state.tick - 1
        if t in window:
            per_week.append((state.people[m].acute_anxiety, state.ties[tie].felt_impingement[m],
                             math.fsum(excess(p) for p in state.people.values() if p.alive)))

    # F's own I-POSITION, toward anyone, is withheld every week but the scripted one, in every arm, so the control has
    # none and the act arm has one.
    withheld = {t: {f: frozenset({"I-POSITION"})} for t in range(weeks) if not (scripted and t == T0)}
    state, records = scenario("triad", seed, weeks, spells=spells, setup=setup, watch=watch, withheld=withheld,
                              forced=act("f", "I-POSITION", "m", T0) if scripted else None,
                              held_open=held(prepare_end + 1, "f-m"))
    on_tie = [r.event.kind for r in records if isinstance(r, EmittedRecord) and r.event.mechanism is Mechanism.MOVE
              and r.event.timestamp in window and {r.event.sender, *r.event.targets} >= {f, m}]
    steps = Counter(field for r in records if isinstance(r, EffectRecord) and r.mechanism == "iposition"
                    for pid, field, _ in r.people if pid == f)
    forms = Counter(("assertion" if r.event.demand_share >= 1.0 else "genuine")  # revision 39: share 1 is the form
                    for r in records if isinstance(r, EmittedRecord) and r.event.kind == "I-POSITION"
                    and r.event.sender == f)
    return {
        "target_acute": statistics.fmean(w[0] for w in per_week),
        "target_impingement": statistics.fmean(w[1] for w in per_week),
        "family_excess": statistics.fmean(w[2] for w in per_week),
        "conflict_share": (sum(k == "CONFLICT" for k in on_tie) / len(on_tie)) if on_tie else math.nan,
        "completions": float(steps["iposition:end:completed"]),
        "moves_on_tie": len(on_tie), "mover_basic": state.people[f].basic_level,
        "steps": dict(steps), "forms": dict(forms),
    }


def one_seed(seed: int) -> dict:
    return {"seed": seed, **{f"{'act' if s else 'none'}_{level:g}": one_arm(level, s, seed)
                             for level in LEVELS for s in (True, False)}}


def paired(values) -> tuple[float, float, int]:
    """Mean of paired differences and its 95% half-width, ``1.96 × sd / √n`` (the runner's, `M17.A.1`); a seed whose
    readout is undefined in either arm (no move on the tie) is left out and counted in ``n``."""
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return (vals[0] if vals else math.nan), math.inf, len(vals)
    return statistics.fmean(vals), 1.96 * statistics.stdev(vals) / math.sqrt(len(vals)), len(vals)


def contrasts(rows: list[dict]) -> dict:
    low, high = (f"{level:g}" for level in LEVELS)
    out = {}
    for name in READOUTS:
        r = lambda row, arm: row[arm][name]  # noqa: E731
        out[name] = {
            "levels_with_act": paired([r(x, f"act_{low}") - r(x, f"act_{high}") for x in rows]),
            "act_effect_low": paired([r(x, f"act_{low}") - r(x, f"none_{low}") for x in rows]),
            "act_effect_high": paired([r(x, f"act_{high}") - r(x, f"none_{high}") for x in rows]),
            "difference_in_differences": paired([(r(x, f"act_{low}") - r(x, f"none_{low}"))
                                                 - (r(x, f"act_{high}") - r(x, f"none_{high}")) for x in rows]),
        }
    return out


def defined(row: dict, arm: str) -> bool:
    """The scripted act reached DEFINE as the genuine `I-POSITION` (not the assertion form, not a failed PREPARE)."""
    return row[arm]["forms"].get("genuine", 0) >= 1


def summarise(rows: list[dict]) -> dict:
    low, high = (f"{level:g}" for level in LEVELS)
    both = [x for x in rows if defined(x, f"act_{low}") and defined(x, f"act_{high}")]
    out = {"seeds": len(rows), "contrasts": contrasts(rows), "defined_seeds": len(both),
           "contrasts_defined": contrasts(both) if len(both) > 1 else None, "means": {}, "steps": {}, "forms": {}}
    for arm in rows[0]:
        if arm == "seed":
            continue
        out["means"][arm] = {n: statistics.fmean(v for x in rows if not math.isnan(v := x[arm][n]))
                             if any(not math.isnan(x[arm][n]) for x in rows) else math.nan for n in READOUTS}
        out["means"][arm]["moves_on_tie"] = statistics.fmean(x[arm]["moves_on_tie"] for x in rows)
        out["steps"][arm] = sum((Counter(x[arm]["steps"]) for x in rows), Counter())
        out["forms"][arm] = sum((Counter(x[arm]["forms"]) for x in rows), Counter())
    return out


def _fmt(t) -> str:
    mean, half, n = t
    return f"{mean:+.4f} ± {half:.4f} (n {n})"


def render(summary: dict, engine: str, when: str, keep: str, prepare_ticks: int) -> str:
    low, high = (f"{level:g}" for level in LEVELS)
    n = summary["seeds"]
    prepare_end = T0 + prepare_ticks
    lines = [
        "# Stage 3 step 2b — the I-position probe (`M5.F.4`, `M11.C.29`)",
        "",
        "Generated by `tools/stage_3_iposition_probe.py`; do not edit above the \"Reading\" section, which the tool keeps.",
        f"Engine hash `{engine}`, {when}, seeds 0–{n - 1} ({n}). The triad fixture (`config/bowen/fixtures/triad.md`):",
        f"mover F at basic and functional level {low} or {high}, systems perspective 1.0, `I-POSITION` toward M",
        f"scripted at week {T0} (F's own `I-POSITION` toward anyone withheld every other week, in every arm); the declared spell on both parents with no spell event in weeks {T0}–{prepare_end}",
        f"(the act week and its PREPARE window, `prepare_ticks` {prepare_ticks}); the F–M tie held open until week",
        f"{prepare_end}. Each level also runs a control on the same seed with nothing scripted. Readouts over weeks",
        f"{T0 + 1}–{T0 + WINDOW} (the six after the act's week; DEFINE comes early when M's impinging move arrives",
        "during PREPARE, `M5.D.2a`): M's acute anxiety, M's felt",
        "impingement on the F–M tie, the family's summed excess (acute above chronic, all three), the `CONFLICT` share of",
        f"the moves either of F and M sent the other; completions (`iposition:end:completed`) over weeks 0–{T0 + AFTER - 1}.",
        "Every difference is paired on the seed, mean ± 1.96 × sd ÷ √n; a seed with no move on the tie in either arm has",
        "no `CONFLICT` share and is left out of that row, as `n` shows. Not a real family (`M11.F.9`).",
        "",
        "## Contrasts, every seed",
        "",
        f"| Readout | level {low} − level {high}, both acting | act − control at {low} | act − control at {high} "
        f"| difference in differences |",
        "|---|---|---|---|---|",
    ]
    for name in READOUTS:
        c = summary["contrasts"][name]
        lines.append(f"| {name} | {_fmt(c['levels_with_act'])} | {_fmt(c['act_effect_low'])} | "
                     f"{_fmt(c['act_effect_high'])} | {_fmt(c['difference_in_differences'])} |")
    lines += ["", f"### Only the seeds where both levels' act reached DEFINE as the genuine form "
              f"({summary['defined_seeds']} of {n})", ""]
    if summary["contrasts_defined"]:
        lines += [f"| Readout | level {low} − level {high}, both acting | act − control at {low} "
                  f"| act − control at {high} | difference in differences |", "|---|---|---|---|---|"]
        for name in READOUTS:
            c = summary["contrasts_defined"][name]
            lines.append(f"| {name} | {_fmt(c['levels_with_act'])} | {_fmt(c['act_effect_low'])} | "
                         f"{_fmt(c['act_effect_high'])} | {_fmt(c['difference_in_differences'])} |")
    else:
        lines.append("Fewer than two such seeds; no contrast.")
    lines += ["", "## Each arm's mean", "", "| Arm | " + " | ".join(READOUTS) + " | moves on the tie |",
              "|---|" + "---|" * (len(READOUTS) + 1)]
    for arm, means in summary["means"].items():
        lines.append(f"| {arm} | " + " | ".join(f"{means[k]:.4f}" for k in READOUTS)
                     + f" | {means['moves_on_tie']:.2f} |")
    lines += ["", "## What became of the act (counts over all seeds)", "",
              "| Arm | `I-POSITION` events by form | the sequence's steps (F's `iposition` records) |", "|---|---|---|"]
    for arm in summary["steps"]:
        forms = ", ".join(f"{k} {v}" for k, v in sorted(summary["forms"][arm].items())) or "none"
        steps = ", ".join(f"`{k}` {v}" for k, v in sorted(summary["steps"][arm].items())) or "none"
        lines.append(f"| {arm} | {forms} | {steps} |")
    lines += ["", keep.rstrip() if keep else f"{READING}\n\n(to be written)", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    from src.bowen.ensemble.criteria_d import ENGINE
    from tools.record_cache import engine_hash

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(one_seed, range(args.seeds)))
    else:
        rows = [one_seed(s) for s in range(args.seeds)]
    old = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
    keep = old[old.index(READING):] if READING in old else ""
    args.out.write_text(render(summarise(rows), engine_hash()[:12], dt.date.today().isoformat(), keep,
                               ENGINE.prepare_ticks), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
