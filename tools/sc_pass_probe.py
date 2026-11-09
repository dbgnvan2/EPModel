"""T-206's and T-208's probes of the two social-cue passes, `M11.C.52` and `M11.C.55`. A measurement, never a verdict.

Purpose: rerun each criterion's own arms (``criteria_sc.arm_c52``, ``criteria_d2.supplier_run``) on more seeds and
         under declared conditions that the records do not run, and read what carries each pass.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.52, #M11.C.55, #M4.C.1f, #M4.C.6, #M10.B.5, #M11.5; learnings P7
Tests:   none; a diagnostic for T-206 and T-208, reported in docs/review/2026-11-09_sc_pass_probes.md. Outside the engine
         hash: it changes nothing a run computes. Every condition is an override passed to the criterion's own
         ``scenario`` call for that run only, or the named mutant applied to a temporary copy of the repository.

**T-208 (`M11.C.52`).** One JSON row per seed with both arms' ``escalation``. Conditions:

* ``recorded``: the criterion as the record runs it (``social_cues_enabled`` on in both arms).
* ``equal``: the cues on with `M4.C.1f`'s reduction (nothing on tone, face and posture, no meaning error, full
  detection; the overrides of ``tests/bowen/test_social_cues.py::NEUTRAL``). The nonverbal part is then zero, so the
  nonverbal discount (`M4.C.6` by channel) has nothing to act on; what is left of the difference is carried by the
  other paths ``systems_perspective`` takes.
* ``off``: the cues off in both arms (`M4.C.6`'s whole-event rule, as before revision 55).

``--mutant`` runs the condition in a temporary copy with ``cue-attribution-discount-removed`` applied
(``tools/mutation_record.MUTANTS``, ``tools/mutant_runner.apply``), never in the working tree.

**T-206 (`M11.C.55`).** One JSON row per seed: each predictor's hits (the learned relief, birth order, and the child
who supplied the parent most before the window, which is not the policy's input), the parents supplied, and how many
parents the relief predictor names at all. Conditions: ``recorded``, and ``learning_off`` (`M10.B.5`'s switch,
``criteria_d.LEARNING_OFF``, in the same run as the cues).

    python3 tools/sc_pass_probe.py c52 --condition recorded --seeds N [--start S] [--workers W] [--mutant] > rows.jsonl
    python3 tools/sc_pass_probe.py c55 --condition learning_off --seeds N [--start S] [--workers W] > rows.jsonl
    python3 tools/sc_pass_probe.py summarise rows.jsonl [more.jsonl ...]
    python3 tools/sc_pass_probe.py paired a.jsonl b.jsonl     # per-seed difference of two c52 files' effects
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MUTANT = "cue-attribution-discount-removed"
# `M4.C.1f`'s reduction, as ``tests/bowen/test_social_cues.py`` states it (``NEUTRAL``).
EQUAL = {"cue_leak_gain": 0.0, "belief_leak_gain": 0.0, "attention_load_gain": 0.0,
         "cue_bias_floor": 0.0, "cue_bias_level_gain": 0.0, "cue_bias_anxiety_gain": 0.0, "cue_bias_belief_gain": 0.0,
         "cue_spread_floor": 0.0, "cue_spread_level_gain": 0.0, "cue_sensitivity_floor": 1.0,
         "cue_sensitivity_level_gain": 0.0}
C52 = {"recorded": {"social_cues_enabled": 1}, "equal": {"social_cues_enabled": 1, **EQUAL},
       "off": {"social_cues_enabled": 0}}
C55 = ("recorded", "learning_off")


def c52_seed(job) -> dict:
    seed, condition = job
    from src.bowen.ensemble import criteria_sc
    from src.bowen.ensemble.criteria import SETTINGS

    with mock.patch.object(criteria_sc, "CUES_ON", C52[condition]):
        base = criteria_sc.arm_c52("baseline", seed, SETTINGS["M11.C.52"]).readouts["escalation"]
        treat = criteria_sc.arm_c52("treatment", seed, SETTINGS["M11.C.52"]).readouts["escalation"]
    return {"probe": "c52", "condition": condition, "seed": seed, "baseline": base, "treatment": treat}


def c55_seed(job) -> dict:
    seed, condition = job
    from src.bowen.ensemble import criteria_d2
    from src.bowen.ensemble.criteria import ROLES, SETTINGS
    from src.bowen.ensemble.criteria_d import LEARNING_OFF
    from src.bowen.readouts.roles import load_supply_kinds, main_supplier, supply_counts
    from src.bowen.io.load import load_family

    settings = SETTINGS["M11.C.55"]
    original = criteria_d2.scenario

    def scenario(*args, overrides=None, **kwargs):
        merged = dict(overrides or {})
        if condition == "learning_off":
            merged.update(LEARNING_OFF)
        return original(*args, overrides=merged, **kwargs)

    criteria_d2.supplier_run.cache_clear()  # a cached summary would skip the override
    try:
        with mock.patch.object(criteria_d2, "scenario", scenario):
            summary = criteria_d2.supplier_run(seed, settings["years"], settings["window_from_year"])
    finally:
        criteria_d2.supplier_run.cache_clear()
    kinds = load_supply_kinds()
    start = settings["window_from_year"] * criteria_d2.YEAR
    parents = (ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"])
    family = load_family(criteria_d2.FAMILIES["phase_d"])
    children = tuple(sorted(p for p in family.people if set(family.parents_of(p)) == set(parents)))
    records = summary["records"]
    window = supply_counts(records, kinds, since=start)
    before = supply_counts(records, kinds, until=start)
    relief = summary["relief"]
    prior_hits = named = 0
    for parent in parents:
        main = main_supplier(window, parent, children)
        values = sorted(((relief[c, parent], c) for c in children), reverse=True)
        named += values[0][0] != values[1][0]
        if main is not None:
            prior_hits += main_supplier(before, parent, children) == main
    return {"probe": "c55", "condition": condition, "seed": seed, "relief": summary["hits"]["relief"],
            "birth_order": summary["hits"]["birth_order"], "prior_supply": prior_hits,
            "parents_supplied": summary["parents_supplied"], "relief_names": named, "children": len(children),
            "learned_nonzero": sum(1 for v in relief.values() if v != 0.0)}


def run(args) -> int:
    jobs = [(s, args.condition) for s in range(args.start, args.start + args.seeds)]
    fn = c52_seed if args.probe == "c52" else c55_seed
    with Pool(args.workers) as pool:
        for row in pool.imap(fn, jobs):
            if args.mutant:
                row["mutant"] = MUTANT
            print(json.dumps(row), flush=True)
    return 0


def in_mutant_copy(argv: list[str]) -> int:
    from tools.mutant_runner import apply
    from tools.mutation_record import MUTANTS

    mutant = next(m for m in MUTANTS if m.id == MUTANT)
    with tempfile.TemporaryDirectory(prefix="sc-probe-mutant-") as name:
        tmp = Path(name)
        for part in ("src", "config", "tools"):
            shutil.copytree(REPO / part, tmp / part, ignore=shutil.ignore_patterns("__pycache__"))
        apply(tmp, mutant)
        # --mutant stays on the copy's command line, so its rows carry the mutant's id
        return subprocess.call([sys.executable, str(tmp / "tools" / "sc_pass_probe.py"), *argv, "--in-copy"], cwd=tmp)


# --- summaries ------------------------------------------------------------------------------------------------------


def _rows(paths):
    return [json.loads(line) for p in paths for line in Path(p).read_text(encoding="utf-8").splitlines() if line]


def _test(diff, base, direction):
    """The runner's reading of one directional readout (``runner._verdict``): mean, half-width, p after the margin."""
    import numpy as np
    from src.bowen.ensemble.runner import _half_width, _scale
    from src.bowen.ensemble.stats import signed_rank
    from src.bowen.io.load import load_constants

    diff, base = np.array(diff, dtype=float), np.array(base, dtype=float)
    margin = load_constants()["ensemble_margin"] * _scale(base, diff)
    _, n, p = signed_rank(direction * diff, margin=margin)
    return float(np.mean(diff)), _half_width(diff), p, n


def summarise(rows) -> list[str]:
    out = []
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["probe"], r["condition"], r.get("mutant", "")), []).append(r)
    for (probe, condition, mutant), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["seed"])
        label = f"{probe} {condition}{' ' + mutant if mutant else ''} (seeds {rs[0]['seed']}–{rs[-1]['seed']}, n {len(rs)})"
        if probe == "c52":
            m, h, p, n = _test([r["treatment"] - r["baseline"] for r in rs], [r["baseline"] for r in rs], -1)
            out.append(f"{label}: escalation {m:+.1f} ± {h:.1f}, p {p:.3g} (n nonzero after margin {n}); "
                       f"baseline mean {math.fsum(r['baseline'] for r in rs) / len(rs):.1f}")
        else:
            base = [r["birth_order"] for r in rs]
            for name in ("relief", "prior_supply"):
                m, h, p, _ = _test([r[name] - r["birth_order"] for r in rs], base, +1)
                out.append(f"{label}: {name} − birth_order {m:+.3f} ± {h:.3f}, p {p:.3g}")
            n = len(rs)
            supplied = math.fsum(r["parents_supplied"] for r in rs)
            chance = math.fsum(r["parents_supplied"] / r["children"] for r in rs) / n
            out.append(f"{label}: per seed, hits relief {math.fsum(r['relief'] for r in rs) / n:.3f}, birth order "
                       f"{math.fsum(r['birth_order'] for r in rs) / n:.3f}, prior supply "
                       f"{math.fsum(r['prior_supply'] for r in rs) / n:.3f}, chance {chance:.3f}; parents supplied "
                       f"{supplied / n:.3f}, named by relief {math.fsum(r['relief_names'] for r in rs) / n:.3f}, "
                       f"nonzero learned sums {math.fsum(r['learned_nonzero'] for r in rs) / n:.2f}")
    return out


def paired(a_rows, b_rows) -> str:
    """Per seed, (b's treatment − baseline) − (a's): whether b's effect differs from a's on the same seeds."""
    a = {r["seed"]: r["treatment"] - r["baseline"] for r in a_rows}
    b = {r["seed"]: r["treatment"] - r["baseline"] for r in b_rows}
    seeds = sorted(set(a) & set(b))
    d = [b[s] - a[s] for s in seeds]
    m = math.fsum(d) / len(d)
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in d) / (len(d) - 1))
    return f"effect(b) − effect(a) {m:+.2f} ± {1.96 * sd / math.sqrt(len(d)):.2f} (n {len(d)})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("probe", choices=("c52", "c55", "summarise", "paired"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--condition", default="recorded")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mutant", action="store_true")
    ap.add_argument("--in-copy", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.probe == "summarise":
        print("\n".join(summarise(_rows(args.files))))
        return 0
    if args.probe == "paired":
        print(paired(_rows(args.files[:1]), _rows(args.files[1:2])))
        return 0
    if args.condition not in (C52 if args.probe == "c52" else C55):
        raise SystemExit(f"{args.probe}: no condition {args.condition!r}")
    if args.mutant and args.probe != "c52":
        raise SystemExit("--mutant is c52's only")
    if args.mutant and not args.in_copy:
        return in_mutant_copy(sys.argv[1:])
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
