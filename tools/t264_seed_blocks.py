"""T-264's seed blocks: do revision 66's borderline verdict moves hold as effects on fresh, disjoint seeds?

Purpose: T-253's seed-block probe (``tools/t253_seed_blocks.py``) widened to every criteria dict and to the Phase D
         mutants. ``mutate`` copies an engine checkout's ``src``, ``config`` and ``tools`` and applies one Phase D
         mutant there, as ``tools/mutant_runner.apply`` does; ``run`` runs one criterion on one fixed block of keyed
         seeds of a checkout or mutant copy; ``extract`` keeps each block's tested readouts per seed; ``summarise``
         renders from that file alone. A measurement, never a verdict.
Spec:    docs/bowen_agent_model_spec_v2.md#M17.A.1, #M11.4e; learnings P23
Tests:   none; a diagnostic for T-264, reported in docs/review/2026-11-21_t264_seed_blocks.md. Outside the engine hash:
         it changes nothing a run computes.

    python3 tools/t264_seed_blocks.py mutate REPO MUTANT_ID DEST
    python3 tools/t264_seed_blocks.py run REPO CRITERION FIRST_SEED N WORKERS OUT.json   # OUT: <tag>__<engine>__<first>.json
    python3 tools/t264_seed_blocks.py extract OUT_DIR > docs/review/t264_seed_blocks.json
    python3 tools/t264_seed_blocks.py summarise docs/review/t264_seed_blocks.json > docs/review/2026-11-21_t264_seed_blocks.md
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARGIN = 0.1  # ensemble_margin (config/bowen), as the runner applies it: the margin is this times the baseline sd
ALPHA = 0.05  # ensemble_alpha
# tag -> what the block runs: the criterion, and the Phase D mutant applied (None for the unmodified engine)
ITEMS = {"c10": ("M11.C.10", None), "c12": ("M11.C.12", None), "c17": ("M11.C.17", None),
         "c44": ("M11.C.44", None), "c49": ("M11.C.49", None), "c51": ("M11.C.51", None),
         "m46": ("M11.C.46", "triangle-help-ignored"), "m7": ("M11.C.7[one spouse over sided]", "coach-position-blind"),
         "m55": ("M11.C.55", "supply-by-birth-order")}


def mutate(repo: str, mutant_id: str, dest: str) -> None:
    sys.path.insert(0, repo)
    os.chdir(repo)
    from tools.mutant_runner import apply
    from tools.phase_d_mutation_record import MUTANTS
    mutant = next(m for m in MUTANTS if m.id == mutant_id)
    Path(dest).mkdir(parents=True)
    for part in ("src", "config", "tools"):
        shutil.copytree(Path(repo) / part, Path(dest) / part, ignore=shutil.ignore_patterns("__pycache__"))
    apply(Path(dest), mutant)


def run(repo: str, cid: str, first: int, n: int, workers: int, out: str) -> None:
    sys.path.insert(0, repo)
    os.chdir(repo)
    from src.bowen.ensemble import runner
    from src.bowen.ensemble.criteria import CRITERIA
    from src.bowen.ensemble.criteria_d import CRITERIA_D
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2
    from src.bowen.io.load import load_constants
    crit = dataclasses.replace({**CRITERIA, **CRITERIA_D, **CRITERIA_D2}[cid], seeds=n)
    c = load_constants()
    rules = {k: c[k] for k in ("ensemble_block", "ensemble_cap", "ensemble_precision", "ensemble_margin",
                               "ensemble_alpha", "fallback_flag_rate", "equivalence_margin")}
    captured = {}
    verdict_of = runner._verdict

    def keep(criterion, results, converged, rules):
        captured["rows"] = [(s, b.readouts, t.readouts) for s, b, t in results]
        return verdict_of(criterion, results, converged, rules)

    runner._verdict = keep
    started = time.time()
    v = runner.run_criterion(crit, rules, workers=workers, first_seed=first)
    Path(out).write_text(json.dumps({"repo": repo, "criterion": cid, "first_seed": first, "n": n,
                                     "seconds": time.time() - started, "outcome": v.outcome, "readouts": v.readouts,
                                     "per_seed": captured["rows"]}, default=float))


def extract(out_dir: str) -> dict:
    data: dict = {}
    for path in sorted(Path(out_dir).glob("*__*__*.json")):
        tag, engine, first = path.stem.split("__")
        raw = json.loads(path.read_text())
        assert raw["criterion"] == ITEMS[tag][0], path
        tested = [r for r in raw["readouts"] if "p" in r and not r.get("report_only") and r["direction"]]
        data.setdefault(tag, {}).setdefault(engine, {})[first] = {
            "outcome": raw["outcome"], "seconds": round(raw["seconds"]), "seeds": [s for s, _, _ in raw["per_seed"]],
            "tested": {r["readout"]: {"direction": r["direction"], "p": r["p"],
                                      "base": [b[r["readout"]] for _, b, _ in raw["per_seed"]],
                                      "treat": [t[r["readout"]] for _, _, t in raw["per_seed"]]} for r in tested}}
    return data


def _stats(base, treat, direction):
    import numpy as np
    sys.path.insert(0, str(REPO))
    from src.bowen.ensemble.stats import signed_rank
    base, treat = np.array(base, float), np.array(treat, float)
    diff = treat - base
    sd = float(np.std(base, ddof=1)) or float(np.std(diff, ddof=1))
    _, _, p = signed_rank(direction * diff, margin=MARGIN * sd)
    return float(np.mean(diff)), 1.96 * float(np.std(diff, ddof=1)) / math.sqrt(len(diff)), p


def summarise(data: dict, header: list[str] | None = None) -> str:
    out = header or ["# T-264: seed blocks for revision 66's borderline verdict moves", "",
           "Generated by `tools/t264_seed_blocks.py summarise docs/review/t264_seed_blocks.json`; do not edit. Run on "
           "Mac Mini1 on 2026-11-21. Engines: `r64` is revision 64 (`429afd4`, engine hash `d9e26aca1b99`) and `r66` "
           "revision 66 (`6abe75a`, `4aaa9029f63a`); a mutant row runs that Phase D mutant applied to the engine. "
           "Every block is fresh and disjoint from the records' seeds (first seed 40000, 41000 or 42000), with the "
           "larger of the two records' seed counts, fixed. A block's outcome and p are the runner's own verdict, Holm "
           "across the tested readouts included; the pooled p is each readout's one-sided signed-rank p with the "
           "runner's margin over all blocks of the engine, unadjusted. A fixed block that ends short of the runner's "
           "precision rule is UNDETERMINED where the record, free to add seeds, would go on. Under a mutant, a block "
           "that does not pass catches it. A measurement, not a verdict; its reading is "
           "in T-264's record.", ""]
    for tag, engines in data.items():
        cid, mutant = ITEMS[tag]
        names = list(next(iter(next(iter(engines.values())).values()))["tested"])
        out += [f"## `{cid}`" + (f" under `{mutant}`" if mutant else ""), "",
                "| Block | Engine | Seeds | Outcome | " + " | ".join(f"`{n}`: difference ± half-width, p" for n in names)
                + " |", "|---" * (4 + len(names)) + "|"]
        for first in sorted({f for e in engines.values() for f in e}, key=int):
            for engine in sorted(engines):
                b = engines[engine].get(first)
                if b:
                    cells = []
                    for n in names:
                        t = b["tested"][n]
                        m, h, _ = _stats(t["base"], t["treat"], t["direction"])
                        cells.append(f"{m:+.4g} ± {h:.2g}, {t['p']:.3g}")
                    out.append(f"| {first} | {engine} | {len(b['seeds'])} | {b['outcome']} | " + " | ".join(cells) + " |")
        out += ["", "| Engine | Seeds pooled | Blocks passing | "
                + " | ".join(f"`{n}`: pooled difference ± half-width, p" for n in names) + " |",
                "|---" * (3 + len(names)) + "|"]
        for engine in sorted(engines):
            blocks = list(engines[engine].values())
            cells = []
            for n in names:
                t = [b["tested"][n] for b in blocks]
                m, h, p = _stats(sum((x["base"] for x in t), []), sum((x["treat"] for x in t), []), t[0]["direction"])
                cells.append(f"{m:+.4g} ± {h:.2g}, {p:.3g}")
            out.append(f"| {engine} | {sum(len(b['seeds']) for b in blocks)} | "
                       f"{sum(b['outcome'] == 'PASS' for b in blocks)} of {len(blocks)} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "mutate":
        mutate(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
    elif mode == "extract":
        print(json.dumps(extract(sys.argv[2])))
    elif mode == "summarise":
        print(summarise(json.loads(Path(sys.argv[2]).read_text())))
