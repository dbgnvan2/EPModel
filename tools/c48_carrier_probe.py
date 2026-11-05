"""T-185's per-act probe of `M11.C.48[phase_d]`: what carries the cut tie's longer negative run. A measurement, never a verdict.

Purpose: run the row's own scenario (the twelve-person family, its nodal load, 40 years; ``criteria_d2.negative_run_values``)
         and, for each matched first cutoff, split the comparison the row makes. The row's baseline is the mean run
         standing that week on every uncut tie in contact. Here it is also read on three subsets of those ties: the
         cutter's own other ties, the other member's own other ties, and the ties touching neither. If the cut tie's run
         is no longer than the cutter's other ties' runs, the difference comes from who cuts (a person-level term), not
         from the tie's own history. The probe also reports, for the cut tie and the baseline ties, the share of each
         person's moves that are negative contacts, and each person's degree (ties in contact by the week).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.48, #M11.1d, #M11.1g; docs/plan_v3_emergence_audit.md#Q3
Tests:   none; a diagnostic for T-185, reported in docs/todo/2026-11-03_version_3_emergence_audit.md (T-185). Outside the
         engine hash: it changes nothing a run computes.

    python3 tools/c48_carrier_probe.py --seeds N [--start S] [--l0]

``--l0`` runs it in a temporary copy of the repository with the emergence audit's L0 applied (``tools/emergence_audit.L0``,
``tools/mutant_runner.apply``), never in the working tree. One JSON row per matched first cutoff.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import shutil
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _mean(xs):
    xs = list(xs)
    return math.fsum(xs) / len(xs) if xs else None


def rows_for_seed(seed: int) -> list[dict]:
    from src.bowen.engine.events import Mechanism
    from src.bowen.engine.identifiers import TieId
    from src.bowen.engine.log_records import EmittedRecord
    from src.bowen.engine.objects import Role, TieState
    from src.bowen.ensemble.criteria import ROLES, SETTINGS, scenario
    from src.bowen.ensemble.criteria_d import nodal
    from src.bowen.ensemble.criteria_d2 import YEAR
    from src.bowen.readouts import negative_runs as nr

    weeks = SETTINGS["M11.C.48"]["years"] * YEAR
    declared: set = set()

    def setup(state) -> None:
        declared.update(t for t, tie in state.ties.items() if tie.tie_state is TieState.CUT_OFF)

    parents = (ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"])
    state, records = scenario("phase_d", seed, weeks, spells=nodal(weeks, parents), setup=setup)
    members = frozenset(p for p, q in state.people.items() if q.role is Role.MEMBER)
    kinds = state.kinds
    firsts = nr.first_cutoff_runs(records, kinds, members, frozenset(declared))

    # the same contacts and runs the readout reads
    per_tie = defaultdict(list)
    for c in nr.contacts(records, kinds, members):
        if c.tie not in declared:
            per_tie[c.tie].append(c)
    runs = {t: nr.runs_by_week(cs) for t, cs in per_tie.items()}
    cut_weeks = {t: w for t, w in nr.first_cutoffs(records, members).items() if t not in declared}

    # who sent each first cutoff, and every person's moves by week (for shares up to the cutoff's week)
    cutter = {}
    moves = []  # (week, sender, kind)
    negative = kinds.negative_contacts | kinds.rebuffs
    for r in records:
        if not isinstance(r, EmittedRecord) or r.event.mechanism is not Mechanism.MOVE or r.event.sender is None:
            continue
        e = r.event
        if e.sender not in members:
            continue
        moves.append((e.timestamp, e.sender, e.kind))
        if e.kind == nr.CUTOFF:
            for t in e.targets:
                tie = TieId.of(e.sender, t)
                if cut_weeks.get(tie) == e.timestamp and tie not in cutter:
                    cutter[tie] = e.sender

    def share_negative(person, week):
        own = [k for w, s, k in moves if s == person and w <= week and k != nr.CUTOFF]
        return count_share(own, negative)

    def count_share(ks, subset):
        return (sum(1 for k in ks if k in subset) / len(ks)) if ks else None

    # a tie's sides at a week: how many of its members had sent any move but CUTOFF across it by then (week included,
    # as the run's own window is)
    first_sent: dict = {}
    for r in records:
        if not isinstance(r, EmittedRecord) or r.event.mechanism is not Mechanism.MOVE or r.event.sender is None:
            continue
        e = r.event
        if e.kind == nr.CUTOFF:
            continue
        for t in e.targets:
            if t == e.sender:
                continue
            key = (TieId.of(e.sender, t), e.sender)
            if key not in first_sent or e.timestamp < first_sent[key]:
                first_sent[key] = e.timestamp

    def sides(tie, week):
        return sum(1 for m in tie.members() if first_sent.get((tie, m), week + 1) <= week)

    out = []
    for f in firsts:
        if not f.matched:
            continue
        a = cutter.get(f.tie)
        b = f.tie.other(a) if a is not None else None
        uncut = [t for t in runs if t != f.tie and (t not in cut_weeks or cut_weeks[t] > f.week)
                 and nr.in_contact(runs[t], f.week)]
        standing = {t: nr.run_at(runs[t], f.week + 1) for t in uncut}
        own_sides = sides(f.tie, f.week)
        same = [standing[t] for t in uncut if sides(t, f.week) == own_sides]
        mine_same = [standing[t] for t in uncut if a in t.members() and sides(t, f.week) == own_sides]
        two = [t for t in uncut if sides(t, f.week) == 2]
        degree = Counter(m for t in uncut + [f.tie] for m in t.members())
        mine = [standing[t] for t in uncut if a in t.members()]
        theirs = [standing[t] for t in uncut if b in t.members()]
        neither = [standing[t] for t in uncut if a not in t.members() and b not in t.members()]
        base_people = sorted({m for t in uncut for m in t.members()}, key=str)
        out.append({
            "seed": seed, "week": f.week, "tie": str(f.tie), "cutter": str(a),
            "run": f.run, "uncut_mean": f.uncut, "check": abs(_mean(standing.values()) - f.uncut) < 1e-9,
            "cutter_other_mean": _mean(mine), "cutter_other_n": len(mine),
            "other_other_mean": _mean(theirs), "other_other_n": len(theirs),
            "neither_mean": _mean(neither), "neither_n": len(neither),
            "cutter_negative_share": share_negative(a, f.week),
            "other_negative_share": share_negative(b, f.week),
            "family_negative_share": _mean(x for x in (share_negative(p, f.week) for p in base_people) if x is not None),
            "cutter_degree": degree[a], "other_degree": degree[b],
            "family_degree": _mean(degree[p] for p in base_people),
            "sides": own_sides, "same_sides_mean": _mean(same), "same_sides_n": len(same),
            "cutter_other_same_sides_mean": _mean(mine_same),
            "baseline_two_sided_share": len(two) / len(uncut),
            "two_sided_mean": _mean(standing[t] for t in two),
            "one_sided_mean": _mean(standing[t] for t in uncut if sides(t, f.week) < 2),
        })
    return out


def _in_l0_copy(args: list[str]) -> int:
    from tools.emergence_audit import L0
    from tools.mutant_runner import apply

    with tempfile.TemporaryDirectory(prefix="c48-l0-") as name:
        tmp = Path(name)
        for part in ("src", "config", "tools"):
            shutil.copytree(REPO / part, tmp / part, ignore=shutil.ignore_patterns("__pycache__"))
        apply(tmp, L0)
        return subprocess.call([sys.executable, str(tmp / "tools" / "c48_carrier_probe.py"), *args], cwd=tmp)


def _seed_mean(rows: list[dict], f) -> str:
    """Per seed, the mean of ``f(row)`` over that seed's cutoffs where it is defined; then the mean over seeds, its 95%
    interval (1.96 standard errors), and how many seeds and cutoffs it reads."""
    per = defaultdict(list)
    for r in rows:
        v = f(r)
        if v is not None:
            per[r["seed"]].append(v)
    vals = [math.fsum(v) / len(v) for v in per.values()]
    n = len(vals)
    if n < 2:
        return f"{n} seeds: too few"
    m = math.fsum(vals) / n
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in vals) / (n - 1))
    return f"{m:+.3f} ± {1.96 * sd / math.sqrt(n):.3f} ({n} seeds, {sum(map(len, per.values()))} cutoffs)"


def summarise(rows: list[dict]) -> list[str]:
    """The lines T-185's note quotes. Each subset line reads only the cutoffs whose subset is not empty, so beside it is
    the row's own difference on those same cutoffs (learning-qa, 2026-11-05, P7): the drop between the two is what the
    subset's matching removes, not a smaller sample. Per-seed means, except the two-sided shares, which are pooled."""
    if not rows:
        raise SystemExit("no rows")
    if not all(r["check"] for r in rows):
        raise SystemExit("a row's baseline does not equal the readout's: the probe no longer reads what the row reads")

    def diff(a, b):
        return lambda r: r[a] - r[b] if r[a] is not None and r[b] is not None else None

    def paired(label, rs, field):
        on = [r for r in rs if r[field] is not None]
        return (f"{label}: {_seed_mean(on, diff('run', field))}; the row on the same cutoffs "
                f"{_seed_mean(on, diff('run', 'uncut_mean'))}")

    two = [r for r in rows if r["sides"] == 2]
    one = [r for r in rows if r["sides"] == 1]
    return [
        f"matched first cutoffs: {len(rows)} on {len({r['seed'] for r in rows})} seeds; "
        f"cut ties with both members acting: {len(two)}, with one: {len(one)}, with none: {len(rows) - len(two) - len(one)}",
        f"the row (run - mean of all uncut ties): {_seed_mean(rows, diff('run', 'uncut_mean'))}",
        paired("run - uncut ties with as many sides", rows, "same_sides_mean"),
        paired("  two-sided cut ties, against two-sided uncut ties", two, "same_sides_mean"),
        paired("  one-sided cut ties, against one-sided uncut ties", one, "same_sides_mean"),
        paired("run - the cutter's other uncut ties", rows, "cutter_other_mean"),
        paired("run - the cutter's other uncut ties with as many sides", rows, "cutter_other_same_sides_mean"),
        paired("run - the other member's other uncut ties", rows, "other_other_mean"),
        paired("run - uncut ties touching neither member", rows, "neither_mean"),
        f"the cutter's other uncut ties - all uncut ties: {_seed_mean(rows, diff('cutter_other_mean', 'uncut_mean'))}",
        f"share of ties both members act on (pooled): cut {len(two) / len(rows):.3f}, baseline "
        f"{math.fsum(r['baseline_two_sided_share'] for r in rows) / len(rows):.3f}",
        f"uncut two-sided ties - uncut one-sided ties: {_seed_mean(rows, diff('two_sided_mean', 'one_sided_mean'))}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--l0", action="store_true")
    parser.add_argument("--summarise", nargs="+", metavar="JSONL", help="summarise rows this probe printed")
    args = parser.parse_args()
    if args.summarise:
        rows = [json.loads(line) for path in args.summarise for line in Path(path).read_text().splitlines() if line]
        print("\n".join(summarise(rows)))
        return 0
    if args.seeds is None:
        parser.error("--seeds is required unless --summarise is given")
    if args.l0:
        return _in_l0_copy(["--seeds", str(args.seeds), "--start", str(args.start)])
    for seed in range(args.start, args.start + args.seeds):
        for row in rows_for_seed(seed):
            print(json.dumps(row, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
