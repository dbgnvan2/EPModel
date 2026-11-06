"""T-197's probe of `M11.C.2[twelve-person family]`: does its pass hold when the arms are matched. A measurement, never a verdict.

Purpose: run the row's own life run (``criteria_d2.life_run``, the family as declared, its ages) per seed, with one more
         observer, and read for each child what the row's two arms differ in besides projection: birth order, and the
         child's state when its basic level is fixed at 5 (``M7.E.1f``; basic level, chronic anxiety, functional level)
         and its mean symptom load before 5. "At 5" is the first week the child is seen at 5 or older: Leo, 8 at week
         0, is read at week 0 and has no load before 5, so the last regression leaves him out. Then three readings of the row's direction: the row as the record reads
         it; matched on birth order (each rank's load when it is the most-projected child against when it is the
         least, across seeds); and within seed, every child's load at the ages read regressed on its acts per year with
         birth rank and its state at 5 held, the seed's mean removed. The family's three children are always Leo,
         Nadia and Pia, so holding birth rank holds which child it is (and with it age at week 0 and the length of the
         window under 20); rank enters as a linear term, and as two indicators in the "which child" line. A file of
         some children only (e.g. ranks 2 and 3) still carries the whole family's row in ``row_treatment``.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.2, #M11.5; learnings P7
Tests:   none; a diagnostic for T-197, reported in docs/todo/2026-11-03_version_3_emergence_audit.md (T-197). Outside the
         engine hash: it changes nothing a run computes. The added observer only reads state.

    python3 tools/c2_matched_probe.py --seeds N [--start S] [--workers W] > rows.jsonl
    python3 tools/c2_matched_probe.py --summarise rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def declared_ranks() -> dict[str, int]:
    """Each child's sibling rank as ``family_phase_d.md`` declares it (the People table's ``sibling_rank`` column)."""
    from src.bowen.ensemble.criteria import FAMILIES

    lines = [line for line in Path(FAMILIES["phase_d"]).read_text(encoding="utf-8").splitlines() if line.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    ranks = {}
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(head) or cells[0] == "a":
            break
        rank = cells[head.index("sibling_rank")]
        if rank.isdigit():
            ranks[cells[0].strip("`")] = int(rank)
    return ranks


def rows_for_seed(seed: int) -> list[dict]:
    from src.bowen.ensemble import criteria_d2
    from src.bowen.ensemble.criteria import SETTINGS
    settings = SETTINGS["M11.C.2"]  # the variant's own settings (criteria_d2.CRITERIA)
    ages = settings["projection_until_age"], settings["load_from_age"], settings["load_to_age"]
    ranks = declared_ranks()
    at_five: dict = {}
    early = defaultdict(lambda: [0.0, 0])
    original = criteria_d2.scenario

    def scenario(*args, watch=None, **kwargs):
        def both(state):
            if watch is not None:
                watch(state)
            for pid, person in state.people.items():
                if person.age_years is None or not person.alive:
                    continue
                if person.age_years < 5:
                    early[pid][0] += math.fsum(person.symptom_load.values())
                    early[pid][1] += 1
                elif pid not in at_five:
                    at_five[pid] = (person.basic_level, person.chronic_anxiety, person.functional_level)
        return original(*args, watch=both, **kwargs)

    criteria_d2.life_run.cache_clear()  # a cached summary would skip the observer
    try:
        with mock.patch.object(criteria_d2, "scenario", scenario):
            summary = criteria_d2.life_run(seed, settings["years"], "declared", 0, ages)
    finally:
        criteria_d2.life_run.cache_clear()  # no later caller reuses a summary built under the patch
    treatment = criteria_d2._pick("treatment", summary["acts_per_year"], summary["aged_load"])
    baseline = criteria_d2._pick("baseline", summary["acts_per_year"], summary["aged_load"])
    out = []
    for c in summary["children"]:
        five = at_five.get(c)
        e = early.get(c)
        out.append({
            "seed": seed, "child": str(c), "rank": ranks[str(c)],
            "acts_per_year": summary["acts_per_year"][c], "aged_load": summary["aged_load"].get(c),
            "basic_at_5": five[0] if five else None, "anxiety_at_5": five[1] if five else None,
            "functional_at_5": five[2] if five else None,
            "load_before_5": e[0] / e[1] if e and e[1] else None,
            "row_treatment": treatment, "row_baseline": baseline,
        })
    return out


def _ci(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n < 2:
        return f"{n}: too few"
    m = math.fsum(vals) / n
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in vals) / (n - 1))
    return f"{m:+.3f} ± {1.96 * sd / math.sqrt(n):.3f} (n {n})"


def _diff_means(a, b):
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return f"too few ({len(a)}, {len(b)})"

    def ms(v):
        m = math.fsum(v) / len(v)
        return m, math.fsum((x - m) ** 2 for x in v) / (len(v) - 1)
    (ma, va), (mb, vb) = ms(a), ms(b)
    return f"{ma - mb:+.3f} ± {1.96 * math.sqrt(va / len(a) + vb / len(b)):.3f} (n {len(a)} most, {len(b)} least)"


def summarise(rows: list[dict]) -> list[str]:
    import numpy as np

    by_seed = defaultdict(list)
    for r in rows:
        r["is_rank_2"], r["is_rank_3"] = float(r["rank"] == 2), float(r["rank"] == 3)
        by_seed[r["seed"]].append(r)
    lines = [f"seeds: {len(by_seed)}; children: {len(rows)}; "
             f"never seen at the ages read: {sum(1 for r in rows if r['aged_load'] is None)}"]
    row = [rs[0]["row_treatment"] - rs[0]["row_baseline"] for rs in by_seed.values()]
    lines.append(f"the row (most-projected load - least-projected load, 0 on a tie): {_ci(row)}")
    most, least = {}, {}
    for s, rs in by_seed.items():
        read = sorted((r for r in rs if r["aged_load"] is not None), key=lambda r: r["acts_per_year"])
        if read and read[0]["acts_per_year"] != read[-1]["acts_per_year"]:
            most[s], least[s] = read[-1], read[0]
    lines.append(f"seeds with a pick: {len(most)}")
    for field in ("rank", "basic_at_5", "anxiety_at_5", "functional_at_5", "load_before_5"):
        d = [most[s][field] - least[s][field] for s in most
             if most[s][field] is not None and least[s][field] is not None]
        lines.append(f"most - least, {field}: {_ci(d)}")
    for rank in sorted({r["rank"] for r in rows}):
        lines.append(f"rank {rank}: most-projected in {sum(1 for s in most if most[s]['rank'] == rank)} seeds, "
                     f"least in {sum(1 for s in least if least[s]['rank'] == rank)}; its load as most - as least: "
                     + _diff_means([most[s]["aged_load"] for s in most if most[s]["rank"] == rank],
                                   [least[s]["aged_load"] for s in least if least[s]["rank"] == rank]))
    # within seed: load on acts per year, birth rank and state at 5 held, the seed's mean removed from each column
    for label, cov in (("acts only", ()), ("with rank", ("rank",)), ("with which child (indicators)", ("is_rank_2", "is_rank_3")),
                       ("with rank and state at 5", ("rank", "basic_at_5", "anxiety_at_5", "functional_at_5")),
                       ("with rank, state at 5 and load before 5",
                        ("rank", "basic_at_5", "anxiety_at_5", "functional_at_5", "load_before_5"))):
        cols = ("acts_per_year",) + cov
        data = [(s, r) for s, rs in by_seed.items() for r in rs
                if r["aged_load"] is not None and all(r[c] is not None for c in cols)]
        seeds = sorted({s for s, _ in data})
        y, x, g = [], [], []
        for s in seeds:
            group = [r for t, r in data if t == s]
            if len(group) < 2:
                continue
            my = sum(r["aged_load"] for r in group) / len(group)
            mx = [sum(r[c] for r in group) / len(group) for c in cols]
            for r in group:
                y.append(r["aged_load"] - my)
                x.append([r[c] - m for c, m in zip(cols, mx)])
                g.append(s)
        X, Y = np.array(x, float), np.array(y, float)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        # seed-clustered standard errors
        bread = np.linalg.pinv(X.T @ X)
        meat = np.zeros((len(cols), len(cols)))
        for s in set(g):
            idx = [i for i, t in enumerate(g) if t == s]
            u = X[idx].T @ resid[idx]
            meat += np.outer(u, u)
        se = np.sqrt(np.diag(bread @ meat @ bread))
        lines.append(f"within seed, {label}: load per act per year {beta[0]:+.3f} ± {1.96 * se[0]:.3f} "
                     f"({len(set(g))} seeds, {len(Y)} children)")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summarise", nargs="+", metavar="JSONL")
    args = parser.parse_args()
    if args.summarise:
        rows = [json.loads(line) for p in args.summarise for line in Path(p).read_text().splitlines() if line]
        print("\n".join(summarise(rows)))
        return 0
    if args.seeds is None:
        parser.error("--seeds is required unless --summarise is given")
    seeds = range(args.start, args.start + args.seeds)
    with Pool(args.workers) as pool:
        for rows in pool.imap(rows_for_seed, seeds):
            for row in rows:
                print(json.dumps(row, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
