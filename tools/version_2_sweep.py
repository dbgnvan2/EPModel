"""The constant sweep of version 2 stage 2 (`M17.E.1`): draw joint samples of the `[I]` constants a criterion's
mechanism reads, run each as a cached mutant of ``config/bowen/constants.md``, apply the detection-limit rule, and
report the fraction of samples in which the asserted direction held.

Purpose: for each criterion declared in ``config/sweep/sweep.md``, report *n of S* samples over the declared ranges
         in which the direction held — a measurement beside the gating verdict, never a verdict. A sample is a
         ``Mutant`` whose edits write the sampled values into the constants file's rows in a temporary copy of the
         repository (``tools/mutant_runner.py``), so nothing in ``src/bowen/`` or the working tree changes and the
         engine hash does not move; results are cached by the engine hash and the edits (``tools/record_cache.py``),
         so this file can change without rerunning anything. A criterion of a Phase D record is run by
         ``tools/phase_d_mutant_child.py`` and cached under ``tools/phase_d_mutation_record.py``'s hash, which covers
         that child; any other by ``tools/mutant_runner.py``'s child under the engine hash.
Spec:    docs/bowen_agent_model_spec_v2.md#M17.E.1, #M17.A.1 (the detection-limit rule, revision 37);
         docs/implementation_plan_version_2.md#2.2 (D8, D10, D11)
Tests:   tests/test_version_2_sweep.py

    python3 tools/version_2_sweep.py [--workers N]

The detection-limit rule (D8): a sample whose first run, under the runner's adaptive stop, leaves any gating
readout's mean difference within ``detection_limit_multiple`` half-widths of zero is run again with the stopping
rows (``ensemble_block``, ``ensemble_cap``) set to ``detection_limit_seeds``, so the runner takes exactly that many
seeds; the sample's verdict is the second run's and both seed counts are recorded. A first run that already used
that many seeds or more is not rerun. The fraction counts a sample as holding only when its verdict is PASS;
UNDETERMINED is counted apart; an errored sample (RAISED, BROKEN or FAILED) is counted apart and left out of the
denominator, never dropped (P2), and a FAILED one is not cached, so it runs again next time (P1).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.mutant_runner import Mutant  # noqa: E402
from tools.phase_d_mutation_record import CHILD as PHASE_D_CHILD  # noqa: E402
from tools.phase_d_mutation_record import engine_hash as phase_d_engine_hash  # noqa: E402
from tools.record_cache import CHILD, ERRORED, Cache, engine_hash, exit_status, result_key, run_cached  # noqa: E402
from tools.record_json import record_json  # noqa: E402
from tools.sweep_config import Range, central_values, load_config  # noqa: E402

RECORD = REPO / "docs" / "version_2_sweep_record.md"
CACHE_NAME = "sweep_v2"
CONSTANTS = "config/bowen/constants.md"
STOPPING_ROWS = ("ensemble_block", "ensemble_cap")  # the runner's block and cap, both set to the fixed count (D8)
ENSEMBLE_RECORDS = (REPO / "docs" / "phase_c_ensemble_record.md", REPO / "docs" / "phase_d_ensemble_record.md",
                    REPO / "docs" / "phase_d2_ensemble_record.md")
PASS, UNDETERMINED = "PASS", "UNDETERMINED"
DIRECTIONAL = ("composite", "mixed")  # the classes a sweep reads (M11.5); never a premise or a check
# A criterion of these records is run by the Phase D child, which knows CRITERIA_D and CRITERIA_D2; any other by
# Phase C's, which knows only CRITERIA (the part I sweep's first run raised KeyError on all 104 samples).
PHASE_D_RECORDS = ("phase_d_ensemble_record", "phase_d2_ensemble_record")


# --- sampling ---------------------------------------------------------------------------------------------------

def draw(ranges: list[Range], n: int, seed: int, key: str) -> list[dict[str, int | float]]:
    """``n`` joint samples of ``ranges``, seeded by ``seed`` and ``key``, drawn one sample at a time so the first
    samples of a longer draw are the shorter draw's. A log range is log-uniform, an integer range uniform over its
    whole values, a linear range uniform. A value outside its range raises; nothing is clipped."""
    rng = np.random.default_rng([int(seed), *key.encode("utf-8")])
    out = []
    for _ in range(n):
        sample: dict[str, int | float] = {}
        for r in ranges:
            if r.scale == "integer":
                value: int | float = int(rng.integers(int(r.low), int(r.high) + 1))
            elif r.scale == "log":
                value = float(math.exp(rng.uniform(math.log(r.low), math.log(r.high))))
            else:
                value = float(rng.uniform(r.low, r.high))
            if not r.low <= value <= r.high:
                raise ValueError(f"{key} {r.constant}: drew {value} outside [{r.low}, {r.high}]")
            sample[r.constant] = value
        out.append(sample)
    return out


def format_value(value: int | float) -> str:
    """The value as the constants file's strict parser reads it back: a whole number for an integer constant, else a
    float with six significant digits (``float()`` reads every form ``g`` prints)."""
    return str(value) if isinstance(value, int) else f"{value:.6g}"


# --- mutants ----------------------------------------------------------------------------------------------------

def _row_edit(name: str, central: dict[str, str], value: str) -> tuple[str, str, str]:
    return CONSTANTS, f"| `{name}` | {central[name]} | ", f"| `{name}` | {value} | "


def sample_mutant(key: str, index: int, sample: dict[str, int | float], central: dict[str, str]) -> Mutant:
    """One sample as a mutant: every sampled constant's row replaced, the first as the mutant's own edit and the rest
    in ``also`` (``M11.1a``: every write named; ``apply`` refuses a row text that is not found exactly once)."""
    edits = [_row_edit(name, central, format_value(v)) for name, v in sample.items()]
    what = ", ".join(f"{name}={format_value(v)}" for name, v in sample.items())
    return Mutant(f"{key} sample {index}", "sweep", (key,), *edits[0], f"{key} sample {index}: {what}",
                  tuple(edits[1:]))


def rerun_mutant(first: Mutant, seeds: int, central: dict[str, str]) -> Mutant:
    """The same sample with the runner's block and cap both set to ``seeds``: ``run_criterion`` then takes one block of
    exactly that many seeds (D8)."""
    extra = tuple(_row_edit(name, central, str(seeds)) for name in STOPPING_ROWS)
    return Mutant(f"{first.id} at {seeds} seeds", first.kind, first.criteria, first.file, first.old, first.new,
                  f"{first.what}; {STOPPING_ROWS[0]}={seeds}, {STOPPING_ROWS[1]}={seeds}", first.also + extra)


# --- the rule and the fraction ----------------------------------------------------------------------------------

def near_limit(row: dict, multiple: float) -> bool:
    """D8: any gating directional readout whose mean difference is within ``multiple`` half-widths of zero. An
    equivalence limb (direction 0) has no detection limit of this kind and is not read (`M17.A.1`)."""
    gating = [r for r in row.get("readouts", ()) if not r.get("report_only") and r.get("direction")]
    return any(abs(r["mean_difference"]) < multiple * r["half_width"] for r in gating)


def summarise(rows: list[dict]) -> dict:
    """The fraction from a criterion's sample rows: ``held`` of ``counted`` (errored samples left out of the
    denominator and counted apart), with the undetermined and rerun counts."""
    errored = [r for r in rows if r["outcome"] in ERRORED]
    counted = [r for r in rows if r["outcome"] not in ERRORED]
    return {"samples": len(rows), "counted": len(counted), "held": sum(1 for r in counted if r["outcome"] == PASS),
            "undetermined": sum(1 for r in counted if r["outcome"] == UNDETERMINED), "errored": len(errored),
            "reruns": sum(1 for r in rows if r["rerun"])}


# --- running ----------------------------------------------------------------------------------------------------

def _result(cache: Cache, mutant: Mutant, key: str, workers: int | None, engine: str, child: str) -> dict:
    if workers is not None:
        return run_cached(cache, mutant, [key], workers, child=child, engine=engine)[0]
    cached = cache.get(result_key(engine, mutant.definition(), key))
    if cached is None:
        raise KeyError(f"{mutant.id}: no cached result for the current engine and sample")
    return cached


def _readouts(row: dict) -> list[dict]:
    return [{"readout": r["readout"], "mean_difference": r["mean_difference"], "half_width": r["half_width"],
             "direction": r["direction"]} for r in row.get("readouts", ()) if not r.get("report_only")]


def sweep_criterion(cache: Cache, key: str, ranges: list[Range], rules: dict, central: dict[str, str],
                    workers: int | None, engine: str, child: str = CHILD) -> dict:
    multiple, fixed = rules["detection_limit_multiple"], int(rules["detection_limit_seeds"])
    rows = []
    for index, sample in enumerate(draw(ranges, int(rules["sweep_samples"]), int(rules["sweep_seed"]), key)):
        mutant = sample_mutant(key, index, sample, central)
        first = _result(cache, mutant, key, workers, engine, child)
        row = {"index": index, "values": {k: format_value(v) for k, v in sample.items()}, "outcome": first["outcome"],
               "seeds": first["seeds"], "readouts": _readouts(first), "rerun": False, "error": first.get("error", "")}
        if first["outcome"] not in ERRORED and near_limit(first, multiple) and first["seeds"] < fixed:
            second = _result(cache, rerun_mutant(mutant, fixed, central), key, workers, engine, child)
            if second["outcome"] not in ERRORED and second["seeds"] != fixed:
                raise SystemExit(f"{key} sample {index}: the rerun took {second['seeds']} seeds, not {fixed}; the criterion "
                                 "fixes its own seed count (config/bowen/criteria.md), which the runner reads before the "
                                 "block and cap, so the sweep cannot rerun it at the declared count (M17.A.1)")
            row.update({"rerun": True, "first_outcome": first["outcome"], "outcome": second["outcome"],
                        "rerun_seeds": second["seeds"], "rerun_readouts": _readouts(second),
                        "error": second.get("error", "")})
        rows.append(row)
    return {"criterion": key, **summarise(rows), "constants": [
        {"constant": r.constant, "low": r.low, "high": r.high, "scale": r.scale, "grade": r.grade} for r in ranges],
        "rows": rows}


def central_verdicts() -> dict[str, dict]:
    out = {}
    for path in ENSEMBLE_RECORDS:
        for row in record_json(path):
            out[row["criterion"]] = {"outcome": row["outcome"], "class": row["class"], "record": path.stem}
    return out


# --- rendering --------------------------------------------------------------------------------------------------

def _fraction(s: dict) -> str:
    return f"{s['held']} of {s['counted']}" if s["counted"] else "no counted sample"


def render(results: list[dict], centrals: dict[str, dict], rules: dict, engine: str, phase_d_engine: str) -> str:
    lines = [
        "# Version 2 sweep record (`M17.E.1`, stage 2)",
        "",
        "Generated by `tools/version_2_sweep.py`; do not edit. **Every number here is a measurement of a range, not a",
        "verdict.** For each criterion below, the `[I]` constants its mechanism reads were drawn jointly over the ranges",
        "declared in `config/sweep/sweep.md` and the criterion was run once per sample; *held* is the number of samples",
        "whose verdict was PASS, of the samples that gave a verdict. The gating verdict is the one in the criterion's",
        "ensemble record (*central verdict*); a fraction here changes no verdict and is never written as a pass or a",
        "fail. A premise passing is never a finding (`M11.5`), and no premise is swept.",
        "",
        f"The sweep's rules (`config/sweep/sweep.md`, each `[I]`): `sweep_samples` {rules['sweep_samples']} per criterion,",
        f"`sweep_seed` {rules['sweep_seed']}; the detection-limit rule (`M17.A.1`, plan D8): a sample whose first run left a",
        f"gating effect within {rules['detection_limit_multiple']} half-widths of zero was run again at a fixed",
        f"{rules['detection_limit_seeds']} seeds and its verdict is the second run's (*reruns*). *Undetermined* samples did",
        "not converge and do not count as holding; *errored* samples (the run raised, did not import or failed) are",
        "counted here and left out of the denominator, never dropped.",
        "",
        f"At `sweep_samples` {rules['sweep_samples']}, these fractions time the sweep (`docs/implementation_plan_version_2.md` §2.2, D10)",
        "and do not measure a criterion; the plan's §2.10 says what the pilot was for.",
        "",
        f"engine_hash: {engine}",
        f"phase_d_engine_hash: {phase_d_engine}",
        "",
        "A Phase C criterion is run by `tools/mutant_runner.py` and its results cached under `engine_hash`; a Phase D",
        "criterion by `tools/phase_d_mutant_child.py` under `phase_d_engine_hash`, which adds that child and the source of",
        "`apply` and `run_in_copy`. Both are kept in `docs/records_cache/sweep_v2.json`; this file is rendered from that",
        "cache.",
        "",
        "| Criterion | Class | Central verdict | Constants swept (low–high) | Held | Undetermined | Reruns at fixed count | Errored |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in results:
        c = centrals[s["criterion"]]
        swept = "; ".join(f"`{x['constant']}` {x['low']:g}–{x['high']:g}" for x in s["constants"])
        lines.append(f"| `{s['criterion']}` | {c['class']} | {c['outcome']} | {swept} | {_fraction(s)} | {s['undetermined']} | "
                     f"{s['reruns']} | {s['errored']} |")
    for s in results:
        lines += ["", f"## `{s['criterion']}` — {_fraction(s)} samples held", "",
                  "| Sample | Values | Seeds | Rerun | Verdict | Gating readouts (mean difference ± half-width) |",
                  "|---|---|---|---|---|---|"]
        for r in s["rows"]:
            values = ", ".join(f"`{k}` {v}" for k, v in r["values"].items())
            seeds = f"{r['seeds']} then {r['rerun_seeds']}" if r["rerun"] else str(r["seeds"])
            rerun = f"yes (first {r['first_outcome']})" if r["rerun"] else "no"
            shown = r["rerun_readouts"] if r["rerun"] else r["readouts"]
            readouts = "; ".join(f"`{x['readout']}` {x['mean_difference']:+.3g} ± {abs(x['half_width']):.2g}" for x in shown)
            if r["error"]:
                readouts = f"*{r['error']}*"
            lines.append(f"| {r['index']} | {values} | {seeds} | {rerun} | {r['outcome']} | {readouts} |")
    lines += ["", "## Machine-readable", "", "```json", json.dumps(results, indent=1, default=float), "```", ""]
    return "\n".join(lines)


def build(cache: Cache, workers: int | None = None) -> str:
    """The record, from the cache. With ``workers``, missing results are run first; without, a missing result raises
    ``KeyError``, so a test can check the committed record without running anything."""
    rules, ranges = load_config()
    central, centrals, engine, phase_d_engine = central_values(), central_verdicts(), engine_hash(), phase_d_engine_hash()
    for key in ranges:  # every key checked before any sample runs
        if key not in centrals:
            raise SystemExit(f"{key}: swept but in no ensemble record")
        if centrals[key]["class"] not in DIRECTIONAL:
            raise SystemExit(f"{key}: a {centrals[key]['class']} is not swept (M11.5; plan D10)")
    results = [sweep_criterion(cache, key, declared, rules, central, workers, phase_d_engine, PHASE_D_CHILD.name)
               if centrals[key]["record"] in PHASE_D_RECORDS
               else sweep_criterion(cache, key, declared, rules, central, workers, engine)
               for key, declared in ranges.items()]
    return render(results, centrals, rules, engine, phase_d_engine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    cache = Cache(CACHE_NAME)
    text = build(cache, args.workers)
    cache.save()
    RECORD.write_text(text, encoding="utf-8")
    print(f"wrote {RECORD.relative_to(REPO)}")
    return exit_status()


if __name__ == "__main__":
    raise SystemExit(main())
