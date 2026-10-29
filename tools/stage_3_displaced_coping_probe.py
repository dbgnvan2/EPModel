"""Probe P2, displaced coping, on `M11.C.6`'s arms with no engine change: does the direction move? A measurement.

Purpose: stage 3 step 3 (`docs/implementation_plan_version_2_stage_3.md` §3.3.1, owner decision D18 revisited). For
         the declared child (Nadia) in both of `M11.C.6`'s arms, nine variants of the engine: as built, and P2 as
         `docs/PROPOSAL — DISPLACED COPING (P2).md` words it with each of two kind lists, and the same two with the
         receiver's side only (the sender gains no functioning); and F1 (the proposal's §4.2, the floor-held
         functioning on a ledger the estimator leaves out), alone and with P2, shed or kept. P2 and F1 are put in by
         wrapping ``tick.respond`` and ``tick.record_week`` in this process for the call, restored after; nothing
         under ``src/bowen/`` is edited. For
         each variant it reports the treatment − baseline difference in `development` and in the two terms of
         `M7.E.1f`'s step, and splits the child's own term into the parts its year-mean is made of, so the half of the
         own-term difference the stress swing does not carry (`docs/review/stage_3_projection_trace.md`) is located.
         It writes `docs/review/stage_3_displaced_coping_probe.md` and keeps that page's hand-written "Reading".
Spec:    docs/bowen_agent_model_spec_v2.md#M7.E.1f, #M11.C.6, #M1.A.5e
Tests:   tests/test_stage_3_probes.py::test_stage_3_displaced_coping_probe_moves_conserved_stress_functioning

    python3 tools/stage_3_displaced_coping_probe.py [--seeds 150] [--workers 10] [--variants V ...]
                                                     [--rows-out F | --rows-in F [F ...]]

The tool is in no ``HASHED_TOOLS`` list and no record reads it, so it moves no record. The P2 wrapper is the
proposal's rule, not the engine's, and its result is a probe's, not a criterion's (`M11.5`).

**Since plan step 7 (2026-10-26) the engine builds P2 and F1** (`M1.A.5f`, ``floor_held``), so "as built" includes
them and the P2 and F1 variants apply them a second time. Its page records the engine of step 3 (hash
``66dcbb1e8ba5``) and is not rerun; the own term's parts below count the engine's ``floor_held`` as a ledger, as they
count F1's, so the decomposition still adds up on the current engine.

**The own term's parts.** The step's own term is the year-mean of ``functional_level − pseudo_self −
max(0, adjust_swing)`` less the basic level before the step. Write ``functional_level`` as ``basic_level +
stress_swing + pseudo_self + adjust_swing + U``; ``U`` is functioning on no ledger (an `I-POSITION` completion's
``exchange_gain``, a bound that stopped a swing's relaxation, and the start of the development years). Then the own
term is ``stress_swing + min(0, adjust_swing) + (U + developed) − developed``, with ``developed`` the basic level
before the step less the anchor: the step's own restoring part, since the development step moves the basic level and
not the functional level.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "docs" / "review" / "stage_3_displaced_coping_probe.md"
ARMS = ("baseline", "treatment")
READING = "## Reading"
# The variants: the kinds whose sender's load P2 meets through the receiver. None is the engine as built.
VARIANTS = {"as_built": None, "p2_overfunction": ("OVERFUNCTION",), "p2_overfunction_pursue": ("OVERFUNCTION", "PURSUE"),
            "receiver_only_overfunction": ("OVERFUNCTION",),
            "receiver_only_overfunction_pursue": ("OVERFUNCTION", "PURSUE"),
            "as_built_f1": None, "p2_overfunction_f1": ("OVERFUNCTION",),
            "as_built_f1_kept": None, "p2_overfunction_f1_kept": ("OVERFUNCTION",)}
# F1 (the proposal's §4.2): what the stress response's bounds held goes on a `floor_held` ledger that the estimator's
# reading leaves out. "shed": a calm week sheds it by `response_relaxation_rate`, moving functional level with it, as
# far as the bound lets (the recommended answer to §9's Q2); "kept": it is never shed (Q2's alternative).
F1 = {"as_built_f1": "shed", "p2_overfunction_f1": "shed", "as_built_f1_kept": "kept", "p2_overfunction_f1_kept": "kept"}
# The receiver-only variants: the receiver loses what P2 moves and the sender gains no functioning (the "calmer" sender
# of `KS04.7` read as anxiety, not functioning); not conserved.
RECEIVER_ONLY = frozenset({"receiver_only_overfunction", "receiver_only_overfunction_pursue"})
# Each step's parts of the child's own term (see the module docstring), and what P2 moved.
PARTS = ("own_term", "parents_term", "stress", "adjust_neg", "unledgered", "developed", "at_anchor", "held",
         "f1_ledger")
COUNTS = ("displaced_to_child", "displaced_from_child", "completions", "held_weeks", "floor_weeks")


def displaced(state, selections, load, params, kinds, tally, child, sender_gains: bool = True) -> None:
    """P2 as the proposal words it: a loaded sender's act of a listed kind meets their load through its receiver.

    For each selection the policy or a script made (not a fallback) by a living person whose load is above their usual
    level (`M1.A.5e`), of a kind in ``kinds``: the coping amount ``response_up_gain × above / 100 × basic / 100`` is
    split evenly over the act's living family targets who do not hold a position toward the sender (`M1.B.5`), and
    each share moves from the target's functional level to the sender's, on both stress swings, conserved. A target is
    not taken below ``basic − stress_floor_below_basic`` and the sender not above 100; what a bound stops moves to
    no one. With ``sender_gains`` false the sender's side is left out (the receiver-only variants)."""
    from src.bowen.engine.act import WITHHOLD
    from src.bowen.engine.log_records import DecidedBy
    from src.bowen.engine.moves import holds_position
    from src.bowen.engine.objects import SCALE_MAX
    from src.bowen.engine.stress_response import above_usual

    for sel in sorted(selections, key=lambda s: (s.actor, s.index)):
        if sel.decided_by is DecidedBy.FALLBACK or sel.kind == WITHHOLD or sel.kind not in kinds:
            continue
        sender = state.people.get(sel.actor)
        above = above_usual(sender, load.get(sel.actor, 0.0)) if sender is not None else 0.0
        if sender is None or not sender.alive or above <= 0:
            continue
        targets = [t for t in sel.targets if t in state.people and state.people[t].alive and t != sel.actor
                   and not holds_position(state, t, sel.actor, params)]
        if not targets:
            continue
        share = params.response_up_gain * above / SCALE_MAX * sender.basic_level / SCALE_MAX / len(targets)
        for t in targets:
            receiver = state.people[t]
            floor = max(0.0, receiver.basic_level - params.stress_floor_below_basic)
            amount = max(0.0, min(share, receiver.functional_level - floor, SCALE_MAX - sender.functional_level))
            if not amount:
                continue
            receiver.functional_level -= amount
            receiver.stress_swing -= amount
            if sender_gains:
                sender.functional_level += amount
                sender.stress_swing += amount
            if t == child:
                tally["displaced_to_child"] += amount
            if sel.actor == child:
                tally["displaced_from_child"] += amount


def floor_held(state, load, params, ledger, before, mode) -> dict:
    """F1 as the proposal words it, after the stress response has run: each living person's ``(change in
    functional_level) − (change in stress_swing + adjust_swing)`` this week is added to their ``floor_held`` ledger.
    With ``mode`` "shed", a calm person (load not above their usual level) then sheds ``response_relaxation_rate`` of
    the ledger, moving functional level with it, but not below ``basic − stress_floor_below_basic`` nor outside 0–100;
    what the bound stops stays on the ledger. Returns each person's change in functional level from the shedding."""
    from src.bowen.engine.objects import SCALE_MAX
    from src.bowen.engine.stress_response import above_usual

    shed = {}
    for pid, (fl, st, adj) in before.items():
        person = state.people[pid]
        if not person.alive:
            continue
        ledger[pid] = ledger.get(pid, 0.0) + ((person.functional_level - fl) - (person.stress_swing - st)
                                              - (person.adjust_swing - adj))
        if mode != "shed" or not ledger[pid] or above_usual(person, load.get(pid, 0.0)) > 0:
            continue
        floor = max(0.0, person.basic_level - params.stress_floor_below_basic)
        step = params.response_relaxation_rate * ledger[pid]
        if step > 0:
            move = -max(0.0, min(step, person.functional_level - floor))
        else:
            move = max(0.0, min(-step, SCALE_MAX - person.functional_level))
        if move:
            person.functional_level += move
            ledger[pid] += move
            shed[pid] = move
    return shed


def one_seed(seed: int, variants: tuple[str, ...] = tuple(VARIANTS)) -> dict:
    """The variants' two arms on one seed: ``{variant: {arm: {"development": float, "steps": {...}, ...}}}``."""
    import dataclasses

    from src.bowen.engine import iposition, life_course, tick
    from src.bowen.engine.identifiers import PersonId
    from src.bowen.ensemble import criteria_d2
    from src.bowen.ensemble.criteria import SETTINGS
    from src.bowen.ensemble.criteria_d import ENGINE

    settings = SETTINGS["M11.C.6"]
    child = PersonId(settings["child"])
    out = {}
    for variant in variants:
        kinds, f1 = VARIANTS[variant], F1.get(variant)
        out[variant] = {}
        for arm in ARMS:
            steps, weekly, last_basic, anchor = {}, [], {}, {}
            tally = dict.fromkeys(COUNTS, 0.0)
            held = {"stock": 0.0}  # functioning the stress response's bounds kept from relaxing, since week 0
            ledger = {}  # F1's floor_held, per person (empty without F1)

            def watch(state):
                person = state.people[child]
                fl, ps, adj, st = (person.functional_level, person.pseudo_self or 0.0, person.adjust_swing,
                                   person.stress_swing)
                age = person.age_years
                # The basic level the week was lived at: a development step this tick moved it after the week.
                before = last_basic.get("basic", person.basic_level)
                last_basic["basic"] = person.basic_level
                # Under F1 its ledger is a ledger: what it holds is neither unledgered nor held by a bound off-ledger.
                led = ledger.get(child, 0.0) + getattr(person, "floor_held", 0.0)  # F1's, and the engine's (step 7)
                if "u" not in anchor and child in state.life.anchors:
                    anchor["u"], anchor["held"] = fl - before - st - ps - adj - led, held["stock"] - led
                weekly.append((st, min(0.0, adj), fl - before - st - ps - adj - led, held["stock"] - led, led))
                del weekly[:-ENGINE.slow_tick_fast_ticks]
                if state.tick % ENGINE.slow_tick_fast_ticks or age is None or not person.alive:
                    return
                if not (ENGINE.development_start_age_years < age <= ENGINE.development_end_age_years):
                    return
                parents = life_course._living_parents(state, child)
                history = list(state.estimator_history.get(child, ()))[-ENGINE.slow_tick_fast_ticks:]
                own = math.fsum(w.functional_level for w in history) / len(history)
                # the parents' term as at revision 39; from revision 41 (T-131) the engine leaves `w.displaced` out
                # of it, and this historical tool is not rerun
                theirs = [m for p in parents
                          if (m := life_course._year_mean(state, p, ENGINE, lambda w: w.functional_level)) is not None]
                parents_basic = life_course._parents_mean(state, child, "basic_level")
                developed = before - state.life.anchors[child]
                steps[age] = {"own_term": own - before,
                              "parents_term": (math.fsum(theirs) / len(theirs) - parents_basic) if theirs else math.nan,
                              "stress": statistics.fmean(w[0] for w in weekly),
                              "adjust_neg": statistics.fmean(w[1] for w in weekly),
                              "unledgered": statistics.fmean(w[2] for w in weekly) + developed,
                              "developed": developed,
                              "at_anchor": anchor["u"],
                              "held": statistics.fmean(w[3] for w in weekly) - anchor["held"],
                              "f1_ledger": statistics.fmean(w[4] for w in weekly)}

            original_scenario, original_respond, original_complete, original_record = (
                criteria_d2.scenario, tick.respond, iposition.complete, tick.record_week)

            def chained(*args, watch=None, **kwargs):
                def both(state):
                    if watch:
                        watch(state)
                    trace(state)
                return original_scenario(*args, watch=both, **kwargs)

            def respond(state, selections, load, params):
                if kinds:
                    displaced(state, selections, load, params, kinds, tally, child, variant not in RECEIVER_ONLY)
                person = state.people[child]
                before = (person.functional_level, person.stress_swing, person.adjust_swing)
                everyone = {pid: (p.functional_level, p.stress_swing, p.adjust_swing)
                            for pid, p in state.people.items() if p.alive} if f1 else {}
                records = original_respond(state, selections, load, params)
                # What the response left off its ledgers: functioning's change less the swings' (0 unless a bound
                # stopped functioning from moving while the swings decayed, `stress_response.respond`).
                gap = ((person.functional_level - before[0]) - (person.stress_swing - before[1])
                       - (person.adjust_swing - before[2]))
                shed = floor_held(state, load, params, ledger, everyone, f1) if f1 else {}
                held["stock"] += gap + shed.get(child, 0.0)
                if "u" in anchor:
                    tally["held_weeks"] += abs(gap) > 1e-12
                    floor = max(0.0, person.basic_level - params.stress_floor_below_basic)
                    tally["floor_weeks"] += person.functional_level <= floor + 1e-9
                return records

            def record_week(state, params, load, selections, events):
                original_record(state, params, load, selections, events)
                # F1: the estimator's reading leaves the floor_held ledger out.
                for pid, amount in ledger.items():
                    weeks = state.estimator_history.get(pid)
                    if amount and weeks and weeks[-1].tick == state.tick:
                        weeks[-1] = dataclasses.replace(weeks[-1], functional_level=weeks[-1].functional_level - amount)

            def complete(state, mover, params):
                if mover == child:
                    tally["completions"] += 1
                return original_complete(state, mover, params)

            trace = watch
            criteria_d2.scenario, tick.respond, iposition.complete = chained, respond, complete
            if f1:
                tick.record_week = record_week
            try:
                res = criteria_d2.arm_c6(arm, seed, settings)
            finally:
                criteria_d2.scenario, tick.respond, iposition.complete, tick.record_week = (
                    original_scenario, original_respond, original_complete, original_record)
            out[variant][arm] = {"development": res.readouts["development"], "acts": res.readouts["projection_acts"],
                                 "steps": steps, **tally}
    return {"seed": seed, **out}


def paired(values: list[float]) -> tuple[float, float, int]:
    """Mean of paired differences and its 95% half-width, ``1.96 × sd / √n`` (the runner's, `M17.A.1`)."""
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return (vals[0] if vals else math.nan), math.inf, len(vals)
    return statistics.fmean(vals), 1.96 * statistics.stdev(vals) / math.sqrt(len(vals)), len(vals)


def contribution(steps: dict, name: str, rate: float) -> float:
    """A part's contribution to development summed over the steps: ``rate × part ÷ (number of terms)``. The parts of
    the own term share its divisor, so they add to the own term's contribution."""
    total = 0.0
    for row in steps.values():
        n = 1 if math.isnan(row["parents_term"]) else 2
        if name == "restoring":
            value = -row["developed"]
        elif name == "other":  # the unledgered part neither held by a bound since the anchor nor there at it
            value = row["unledgered"] - row["at_anchor"] - row["held"]
        else:
            value = row.get(name, 0.0)  # rows from before F1 carry no "f1_ledger"
        total += 0.0 if math.isnan(value) else rate * value / n
    return total


def present(rows: list[dict]) -> list[str]:
    """The variants every row carries, in `VARIANTS` order."""
    return [v for v in VARIANTS if all(v in r for r in rows)]


def merge(groups: list[list[dict]]) -> list[dict]:
    """Rows from several runs of the same seeds (``--variants`` splits), joined by seed; a seed's variants must not
    repeat across runs."""
    by_seed: dict[int, dict] = {}
    for rows in groups:
        for row in rows:
            joined = by_seed.setdefault(row["seed"], {"seed": row["seed"]})
            clash = (set(row) & set(joined)) - {"seed"}
            if clash:
                raise SystemExit(f"seed {row['seed']}: variants {sorted(clash)} in more than one rows file")
            joined.update(row)
    return [by_seed[s] for s in sorted(by_seed)]


def summarise(rows: list[dict], rate: float) -> dict:
    """Per variant: treatment − baseline on `development`, each term and each part; each arm's mean of P2's counts."""
    names = ("own_term", "parents_term", "stress", "adjust_neg", "unledgered", "restoring", "at_anchor", "held",
             "other", "f1_ledger")
    summary = {"variants": present(rows)}
    for variant in summary["variants"]:
        diffs = {"development": paired([r[variant]["treatment"]["development"] - r[variant]["baseline"]["development"]
                                        for r in rows])}
        for name in names:
            diffs[name] = paired([contribution(r[variant]["treatment"]["steps"], name, rate)
                                  - contribution(r[variant]["baseline"]["steps"], name, rate) for r in rows])
        diffs["acts"] = paired([r[variant]["treatment"]["acts"] - r[variant]["baseline"]["acts"] for r in rows])
        arms = {arm: {c: statistics.fmean(r[variant][arm][c] for r in rows) for c in COUNTS} for arm in ARMS}
        for arm in ARMS:
            arms[arm]["development"] = statistics.fmean(r[variant][arm]["development"] for r in rows)
        summary[variant] = {"diffs": diffs, "arms": arms}
    summary["against_as_built"] = {
        variant: {arm: paired([r[variant][arm]["development"] - r["as_built"][arm]["development"] for r in rows])
                  for arm in ARMS} for variant in summary["variants"] if variant != "as_built"
        and "as_built" in summary["variants"]}  # a split run without "as_built" has nothing to compare against
    summary["seeds"] = len(rows)
    return summary


def _fmt(t) -> str:
    mean, half, _n = t
    return f"{mean:+.4f} ± {half:.4f}"


def render(summary: dict, engine: str, when: str, seeds: int, keep: str) -> str:
    """The page: the direction under each variant, the own term's parts, and P2's amounts."""
    labels = {"as_built": "as built", "p2_overfunction": "P2, `OVERFUNCTION`",
              "p2_overfunction_pursue": "P2, `OVERFUNCTION` and `PURSUE`",
              "receiver_only_overfunction": "receiver only, `OVERFUNCTION`",
              "receiver_only_overfunction_pursue": "receiver only, `OVERFUNCTION` and `PURSUE`",
              "as_built_f1": "F1, shed", "p2_overfunction_f1": "P2, `OVERFUNCTION`, with F1, shed",
              "as_built_f1_kept": "F1, kept", "p2_overfunction_f1_kept": "P2, `OVERFUNCTION`, with F1, kept"}
    lines = [
        "# Stage 3 step 3 — the displaced coping probe (`M11.C.6`, `M7.E.1f`, P2 and F1)",
        "",
        "Generated by `tools/stage_3_displaced_coping_probe.py`; do not edit above the \"Reading\" section, which the",
        f"tool keeps. Engine hash `{engine}`, {when}, seeds 0–{seeds - 1} ({seeds}). Both arms of `M11.C.6`",
        "(`src/bowen/ensemble/criteria_d2.py::arm_c6`): treatment, the parents act as the policy chooses; baseline, their",
        "projection-type acts toward Nadia withheld. P2 and F1 are put in by wrapping `tick.respond` (and, for F1,",
        "`tick.record_week`) in the tool's process, as `docs/PROPOSAL — DISPLACED COPING (P2).md` words them; nothing",
        "under `src/bowen/` is edited and no record moves. F1 \"shed\" sheds the `floor_held` ledger in a calm week;",
        "F1 \"kept\" never sheds it (the proposal's §9, Q2).",
        "Every difference is treatment − baseline on the same seed, mean ± 1.96 × sd ÷ √n. A probe's result, not a",
        "criterion's (`M11.5`). Not a real family (`M11.F.9`).",
        "",
        "## The direction under each variant",
        "",
        "| Variant | `development` | own term | parents' term | projection-type acts |",
        "|---|---|---|---|---|",
    ]
    for variant in summary["variants"]:
        d = summary[variant]["diffs"]
        lines.append(f"| {labels[variant]} | {_fmt(d['development'])} | {_fmt(d['own_term'])} | "
                     f"{_fmt(d['parents_term'])} | {_fmt(d['acts'])} |")
    lines += [
        "",
        "## The child's own term, by part (each part's contribution, summed over the steps)",
        "",
        "`stress` is the stress swing (`M1.A.5e`); `adjust` is `min(0, adjust_swing)`; `unledgered` is functioning on",
        "no ledger, measured from the anchor; `restoring` is minus the development so far, the step's own pull back.",
        "`unledgered` splits into what was there at the anchor (age 5), what the stress response's bounds held since",
        "(the swings decayed while a bound kept functioning from moving, `stress_response.respond`), and the rest.",
        "Under F1 what a bound held is on the `floor_held` ledger, which the reading leaves out: it is shown in the last",
        "column and is not part of the own term.",
        "",
        "| Variant | own term | stress | adjust | unledgered | restoring | unledgered: at the anchor | "
        "unledgered: held by a bound | unledgered: the rest | F1's ledger (not read) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant in summary["variants"]:
        d = summary[variant]["diffs"]
        lines.append(f"| {labels[variant]} | {_fmt(d['own_term'])} | {_fmt(d['stress'])} | {_fmt(d['adjust_neg'])} | "
                     f"{_fmt(d['unledgered'])} | {_fmt(d['restoring'])} | {_fmt(d['at_anchor'])} | "
                     f"{_fmt(d['held'])} | {_fmt(d['other'])} | {_fmt(d['f1_ledger'])} |")
    lines += [
        "",
        "## Each arm's means (per seed, over ages 5–20)",
        "",
        "| Variant | Arm | `development` | moved to the child by P2 | moved from the child by P2 | "
        "the child's `I-POSITION` completions | weeks a bound held the child's functioning, from age 5 | "
        "weeks the child was at the stress floor, from age 5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variant in summary["variants"]:
        for arm in ARMS:
            a = summary[variant]["arms"][arm]
            lines.append(f"| {labels[variant]} | {arm} | {a['development']:+.4f} | {a['displaced_to_child']:.4f} | "
                         f"{a['displaced_from_child']:.4f} | {a['completions']:.2f} | {a['held_weeks']:.1f} | "
                         f"{a['floor_weeks']:.1f} |")
    lines += [
        "",
        "## Each variant against the engine as built, same arm and seed (`development`)",
        "",
        "| Variant | Baseline | Treatment |",
        "|---|---|---|",
    ]
    for variant, arms in summary["against_as_built"].items():
        lines.append(f"| {labels[variant]} | {_fmt(arms['baseline'])} | {_fmt(arms['treatment'])} |")
    lines += ["", keep.rstrip() if keep else f"{READING}\n\n(to be written)", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=150)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--rows-out", type=Path, help="also write the per-seed rows here, as JSON")
    parser.add_argument("--rows-in", type=Path, nargs="+",
                        help="render from rows written by --rows-out (several are joined by seed); run nothing")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS),
                        help="run only these variants (rows from other runs can be joined with --rows-in)")
    args = parser.parse_args(argv)
    from src.bowen.ensemble.criteria_d import ENGINE
    from tools.record_cache import engine_hash

    if args.rows_in:
        rows = merge([json.loads(f.read_text(encoding="utf-8")) for f in args.rows_in])
        args.seeds = len(rows)
    elif args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(partial(one_seed, variants=tuple(args.variants)), range(args.seeds)))
    else:
        rows = [one_seed(s, tuple(args.variants)) for s in range(args.seeds)]
    if args.rows_out:
        # JSON keys are strings: the ages in each arm's steps come back as "6".."20", which nothing below reads.
        args.rows_out.write_text(json.dumps(rows), encoding="utf-8")
    old = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
    keep = old[old.index(READING):] if READING in old else ""
    args.out.write_text(render(summarise(rows, ENGINE.development_rate), engine_hash()[:12],
                               dt.date.today().isoformat(), args.seeds, keep), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
