"""Trace why `M11.C.48`'s direction is reversed: a measurement, never a verdict (T-176).

Purpose: T-176 (`docs/todo/2026-11-01_pre_tag_milestone_review.md`; the pre-tag review's MR-6 and Unverified 3). On
         both families, each seed run exactly as `M11.C.48`'s arm runs it (`negative_run_values`: the declared spell
         on the Phase B family over ``weeks``, the nodal load on the twelve-person family over ``years``; the seeds
         0–199 the record reads), it takes the run's records and reads every matched first cutoff the readout reads
         (`src/bowen/readouts/negative_runs.py`), then sets the cut tie's run against other baselines than the
         criterion's: the uncut ties split by whether they are cut off later in the run or never; the uncut ties of
         the same declared relation (spouse, parent and child, sibling); the cut tie's own run some weeks before its
         cutoff (does the run lengthen on the tie that is cut off?); and the cut ties that had a contact by their
         cutoff's week only, the condition the readout puts on the uncut ties and not on the cut one. It also counts
         contacts and negative contacts on each tie in the 52 weeks before the cutoff. It writes
         `docs/review/t176_negative_run_trace.md` and keeps that page's hand-written "Reading" section.
         Since spec revision 47 (T-179) the readout leaves out a cut tie with no contact by its week, so the criterion's
         rows read the cut ties with a contact; one row keeps revision 46's reading, a cut tie with no contact at a run
         of 0, beside it. Since T-185 (the owner's Q3 (a) of 2026-11-09) the readout's baseline reads only the uncut
         ties with as many sides as the cut tie, so every row here reads those ties; revision 46's row still reads
         whether any uncut tie was in contact.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.48
Tests:   tests/test_t176_negative_run_trace.py::test_m11c48_t176_trace_reproduces_the_criterions_arms,
         tests/test_t176_negative_run_trace.py::test_m11c48_t176_rows_the_reading_rests_on_read_what_they_say

    nice -n 15 python3 tools/t176_negative_run_trace.py [--seeds 200] [--workers 8]

The tool is in no ``HASHED_TOOLS`` list and no record reads it, so it moves no record. The baselines are `[I]`, this
trace's, declared here before it ran; none of them is the criterion's, which stays as the spec states it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "docs" / "review" / "t176_negative_run_trace.md"
READING = "## Reading"
FAMILIES = ("phase_c", "phase_d")
LABELS = {"phase_c": "the Phase B family (`M11.C.48`)", "phase_d": "the twelve-person family (`M11.C.48[phase_d]`)"}
LAGS = (4, 13, 52)   # weeks before the cutoff's week at which the cut tie's own run is read again
WINDOW = 52          # weeks before the cutoff over which contacts are counted
LATE = 52            # a cutoff after this week had a year for a run to build on its tie
UNDECLARED = "not declared"


def weeks_of(family: str) -> int:
    from src.bowen.ensemble.criteria import SETTINGS
    from src.bowen.ensemble.criteria_d2 import YEAR

    settings = SETTINGS["M11.C.48"]
    return settings["weeks"] if family == "phase_c" else settings["years"] * YEAR


def run_seed(family: str, seed: int):
    """The seed's run as `negative_run_values` makes it: (records, kinds, members, declared ties, relations)."""
    from src.bowen.engine.objects import Role, TieState
    from src.bowen.ensemble.criteria import FAMILIES as PATHS, ROLES, scenario, spell
    from src.bowen.ensemble.criteria_d import nodal
    from src.bowen.io.load import load_family

    weeks = weeks_of(family)
    declared: set = set()

    def setup(state) -> None:
        declared.update(t for t, tie in state.ties.items() if tie.tie_state is TieState.CUT_OFF)

    if family == "phase_d":
        spells = nodal(weeks, (ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"]))
    else:
        spells = spell(family, weeks)
    state, records = scenario(family, seed, weeks, spells=spells, setup=setup)
    members = frozenset(p for p, q in state.people.items() if q.role is Role.MEMBER)
    relations = {t: r.value for t, r in load_family(PATHS[family]).relations.items()}
    return records, state.kinds, members, frozenset(declared), relations


def _mean(values) -> float | None:
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def trace(records, kinds, members, declared, relations) -> dict:
    """Every first cutoff the readout reads, with the trace's baselines beside the criterion's."""
    from src.bowen.readouts.negative_runs import (acted_across, arm_values, contacts, first_cutoff_runs, first_cutoffs,
                                                  run_at, runs_by_week, sides)

    per_tie = defaultdict(list)
    for c in contacts(records, kinds, members):
        if c.tie not in declared:
            per_tie[c.tie].append(c)
    runs = {tie: runs_by_week(cs) for tie, cs in per_tie.items()}
    cutoffs = {t: w for t, w in first_cutoffs(records, members).items() if t not in declared}
    firsts = first_cutoff_runs(records, kinds, members, declared)
    acted = acted_across(records)

    def window(tie, week):
        cs = [c for c in per_tie.get(tie, []) if week - WINDOW < c.week <= week]
        return len(cs), sum(c.negative for c in cs)

    rows = []
    for f in firsts:
        tie, week = f.tie, f.week
        in_contact = [t for t in sorted(runs, key=str) if t != tie and (t not in cutoffs or cutoffs[t] > week)
                      and any(w <= week for w, _ in runs[t])]
        # the readout's own baseline ties since T-185: those with as many sides as the cut tie. ``in_contact`` is kept
        # for revision 46's contrast row, which matched a cutoff whenever any uncut tie was in contact.
        uncut = [t for t in in_contact if sides(acted, t, week) == sides(acted, tie, week)]
        standing = {t: run_at(runs[t], week + 1) for t in uncut}
        relation = relations.get(tie, UNDECLARED)
        own = runs.get(tie, [])
        lags = {}
        for k in LAGS:
            if k <= week:
                lags[k] = (run_at(own, week + 1 - k), _mean(run_at(runs[t], week + 1 - k) for t in uncut))
        own_window = window(tie, week)
        uncut_windows = [window(t, week) for t in uncut]
        rows.append({
            "tie": tie, "week": week, "relation": relation, "run": f.run, "uncut": f.uncut, "matched": f.matched,
            "own_contact": any(w <= week for w, _ in own),
            "matched_rev46": bool(in_contact),  # revision 46's readout matched a cutoff whenever an uncut tie was in contact
            # the readout's exits since T-185: a cutoff in contact with uncut ties but none of its own sides is
            # "unsided", not "unmatched" (learning-qa 2026-11-10: the summary pooled the two)
            "any_uncut": bool(in_contact), "sided": bool(uncut),
            "uncut_never": _mean(standing[t] for t in uncut if t not in cutoffs),
            "uncut_later": _mean(standing[t] for t in uncut if t in cutoffs),
            "uncut_same": _mean(standing[t] for t in uncut if relations.get(t, UNDECLARED) == relation),
            "uncut_by_relation": [(relations.get(t, UNDECLARED), standing[t]) for t in uncut],
            "lags": lags,
            "contacts": own_window[0], "negatives": own_window[1],
            "uncut_contacts": _mean(c for c, _ in uncut_windows), "uncut_negatives": _mean(n for _, n in uncut_windows),
        })
        if (f.contact != rows[-1]["own_contact"] or f.matched != (bool(uncut) and f.contact)
                or (uncut and not math.isclose(f.uncut, _mean(standing.values())))):
            raise RuntimeError(f"the trace's uncut ties are not the readout's at {tie}, week {week}")
    treatment, baseline = arm_values(firsts)
    known = {t for t in relations if set(t.members()) <= members and t not in declared} | set(runs) | set(cutoffs)
    never = known - set(cutoffs)
    return {"cutoffs": rows, "treatment": treatment, "baseline": baseline, "ties": len(known),
            "never_cut": len(never), "never_cut_with_contact": len(never & set(runs))}


def one_seed(args) -> dict:
    family, seed = args
    return {"family": family, "seed": seed, **trace(*run_seed(family, seed))}


def interval(values) -> tuple[float, float, int]:
    """Mean ± 1.96 × sd ÷ √n over seeds."""
    values = [v for v in values if v is not None]
    n = len(values)
    if n == 0:
        return math.nan, math.nan, 0
    mean = math.fsum(values) / n
    half = 1.96 * statistics.stdev(values) / math.sqrt(n) if n > 1 else math.nan
    return mean, half, n


def seed_mean(cutoffs, value) -> float | None:
    """A seed's mean of ``value`` over its first cutoffs where it is defined (None where it is nowhere). Each contrast
    says which cutoffs it reads: ``_matched`` wraps the ones that read the readout's matched cutoffs."""
    return _mean(v for v in (value(c) for c in cutoffs) if v is not None)


def _matched(value):
    """A contrast over the readout's matched first cutoffs: a tie in contact by its cutoff's week, with an uncut tie in
    contact that week (T-123, T-179)."""
    return lambda c: value(c) if c["matched"] else None


def _diff(a, b):
    return None if a is None or b is None else a - b


CONTRASTS = tuple((name, _matched(value)) for name, value in (
    ("criterion: cut tie's run − mean run on every uncut tie that week (a contact by the cutoff's week on both)",
     lambda c: c["run"] - c["uncut"]),
    ("cut tie's run − uncut ties never cut off in the run",
     lambda c: _diff(c["run"], c["uncut_never"])),
    ("cut tie's run − uncut ties cut off later in the run",
     lambda c: _diff(c["run"], c["uncut_later"])),
    ("cut tie's run − uncut ties of the same declared relation",
     lambda c: _diff(c["run"], c["uncut_same"])),
    (f"criterion, cutoffs after week {LATE} only",
     lambda c: c["run"] - c["uncut"] if c["week"] > LATE else None),
)) + (
    # Revision 46's readout (T-176): a cut tie with no contact by its cutoff's week read at a run of 0 against the uncut
    # ties in contact. Kept beside the criterion so the page shows what T-179 changed.
    ("revision 46's readout: a cut tie with no contact by its cutoff's week read at a run of 0",
     lambda c: c["run"] - c["uncut"] if c["matched_rev46"] else None),
) + tuple((name, _matched(value)) for name, value in tuple(
    # Each lag row reads the cutoffs in week k or later, so the lag rows read different cutoffs.
    (f"cut tie's run at its cutoff − its own run {k} weeks before",
     (lambda k: lambda c: c["run"] - c["lags"][k][0] if k in c["lags"] else None)(k))
    for k in LAGS
) + (
    (f"the 4-week row on the cutoffs the {LAGS[1]}-week row reads",
     lambda c: c["run"] - c["lags"][4][0] if LAGS[1] in c["lags"] else None),
) + tuple(
    (f"the same change over {k} weeks, cut tie − mean of the uncut ties",
     (lambda k: lambda c: (c["run"] - c["lags"][k][0]) - (c["uncut"] - c["lags"][k][1])
      if k in c["lags"] and c["lags"][k][1] is not None else None)(k)) for k in LAGS
) + (
    (f"contacts in the {WINDOW} weeks up to the cutoff's, cut tie − mean of the uncut ties",
     lambda c: _diff(c["contacts"], c["uncut_contacts"])),
    (f"negative contacts in the {WINDOW} weeks up to the cutoff's, cut tie − mean of the uncut ties",
     lambda c: _diff(c["negatives"], c["uncut_negatives"])),
))


def summarise(rows: list[dict]) -> dict:
    out = {}
    for family in FAMILIES:
        seeds = [r for r in rows if r["family"] == family]
        if not seeds:
            continue
        cut = [c for r in seeds for c in r["cutoffs"]]
        matched = [c for c in cut if c["matched"]]
        headline = interval([r["treatment"] - r["baseline"] for r in seeds])
        contrasts = [(name, interval([seed_mean(r["cutoffs"], f) for r in seeds])) for name, f in CONTRASTS]
        relations = sorted({c["relation"] for c in matched} | {rel for c in matched for rel, _ in c["uncut_by_relation"]})
        by_relation = []
        for rel in relations:
            mine = [c["run"] for c in matched if c["relation"] == rel]
            theirs = [run for c in matched for r2, run in c["uncut_by_relation"] if r2 == rel]
            by_relation.append((rel, len(mine), _mean(mine), len(theirs), _mean(theirs)))
        out[family] = {
            "seeds": len(seeds), "headline": headline,
            "with_matched": sum(1 for r in seeds if any(c["matched"] for c in r["cutoffs"])),
            "firsts": len(cut), "matched": len(matched),
            "zero_run": sum(1 for c in matched if c["run"] == 0),
            "no_contact": sum(1 for c in cut if not c["own_contact"]),
            "unmatched": sum(1 for c in cut if c["own_contact"] and not c["any_uncut"]),
            "unsided": sum(1 for c in cut if c["own_contact"] and c["any_uncut"] and not c["sided"]),
            "no_contact_rev46": sum(1 for c in cut if not c["own_contact"] and c["matched_rev46"]),
            "median_week": statistics.median(c["week"] for c in matched) if matched else None,
            "early": sum(1 for c in matched if c["week"] < 13),
            "contrasts": contrasts, "by_relation": by_relation,
            "ties": (min(r["ties"] for r in seeds), max(r["ties"] for r in seeds)),
            "never_cut": sum(r["never_cut"] for r in seeds),
            "never_cut_with_contact": sum(r["never_cut_with_contact"] for r in seeds),
            "pooled": (_mean(c["run"] for c in matched), _mean(c["uncut"] for c in matched)),
        }
    return out


def _fmt(t) -> str:
    mean, half, n = t
    if n == 0:
        return "— (n 0)"
    return f"{mean:+.3f} ± {half:.3f} (n {n})" if not math.isnan(half) else f"{mean:+.3f} (n {n})"


def _num(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def render(summary: dict, engine: str, when: str, keep: str) -> str:
    lines = [
        "# T-176 — why `M11.C.48`'s direction is reversed (a trace)",
        "",
        "Generated by `tools/t176_negative_run_trace.py` (`nice -n 15 python3 tools/t176_negative_run_trace.py "
        "--seeds 200 --workers 8`); do not edit above the \"Reading\" section, which the tool keeps.",
        f"Engine hash `{engine}`, {when}. Each seed is run as `M11.C.48`'s arm runs it (`negative_run_values`, "
        "`src/bowen/ensemble/criteria_d2.py`), and every first cutoff is read by the criterion's own readout "
        "(`src/bowen/readouts/negative_runs.py`); on every cutoff the tool checks that the readout matched it exactly when "
        "the cut tie had a contact by its week and the trace's uncut set is not empty, and that the set's mean run is "
        "the readout's baseline. Each row but revision 46's is a per-seed mean over the seed's matched first cutoffs "
        "where the row is defined (revision 46's row reads every cutoff with an uncut tie in contact), then mean ± "
        "1.96 × sd ÷ √n over the seeds where it is defined (`n`). The first row of each table is the criterion's own "
        "difference and is read over every seed, a seed with no matched cutoff entering as 0, as the record reads it "
        "(`docs/phase_d2_ensemble_record.md`). A negative contact is the readout's: a rebuff, a refusal or an act in "
        "the negative set, received in either direction. Not a real family (`M11.F.9`).",
    ]
    for family, s in summary.items():
        lines += ["", f"## {LABELS[family]}", "",
                  f"{s['seeds']} seeds; {s['firsts']} first cutoffs: {s['matched']} matched, {s['unmatched']} "
                  f"unmatched, {s['unsided']} with no uncut tie of their own sides (T-185) "
                  f"and {s['no_contact']} on a tie with no contact by the cutoff's week, the last three left "
                  f"out as the readout leaves them out ({s['no_contact_rev46']} of the no-contact cutoffs had an uncut "
                  f"tie in contact, so revision 46's readout matched them at a run of 0); "
                  f"{s['with_matched']} seeds with a matched cutoff. Of the matched cutoffs, {s['zero_run']} stand on "
                  f"a run of 0; the median "
                  f"matched cutoff falls in week {s['median_week']}, and {s['early']} fall in weeks 0–12. Each seed has "
                  f"{s['ties'][0]}–{s['ties'][1]} member ties not declared cut off (declared, or with a contact or a "
                  f"cutoff in the run); {s['never_cut']} of them over all seeds are never cut off, "
                  f"{s['never_cut_with_contact']} of those with a contact. Pooled over "
                  f"the matched cutoffs, the cut tie's run is {_num(s['pooled'][0])} and its uncut baseline "
                  f"{_num(s['pooled'][1])}.",
                  "", "| Contrast | mean ± 1.96 sd/√n (n seeds) |", "|---|---|",
                  f"| criterion's difference, every seed (the record's `negative_run`) | {_fmt(s['headline'])} |"]
        lines += [f"| {name} | {_fmt(t)} |" for name, t in s["contrasts"]]
        lines += ["", "### By declared relation, pooled over the matched cutoffs", "",
                  "| Relation | cut ties | their mean run at the cutoff | uncut ties standing that week "
                  "(tie-weeks) | their mean run |", "|---|---|---|---|---|"]
        lines += [f"| {rel} | {n1} | {_num(m1)} | {n2} | {_num(m2)} |" for rel, n1, m1, n2, m2 in s["by_relation"]]
    lines += ["", keep.rstrip() if keep else f"{READING}\n\n(to be written)", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--family", choices=FAMILIES, action="append")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    from tools.record_cache import engine_hash

    if args.family and set(args.family) != set(FAMILIES) and args.out == OUT:
        parser.error("a run of one family would rewrite the page whose Reading quotes both; pass --out")
    jobs = [(f, s) for f in (args.family or FAMILIES) for s in range(args.seeds)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(one_seed, jobs))
    else:
        rows = [one_seed(j) for j in jobs]
    old = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
    keep = old[old.index(READING):] if READING in old else ""
    args.out.write_text(render(summarise(rows), engine_hash()[:12], dt.date.today().isoformat(), keep),
                        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
