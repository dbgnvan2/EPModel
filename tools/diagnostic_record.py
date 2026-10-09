"""Diagnose the failing Phase C criteria: criterion runs under named variants, and probes. Never a verdict.

Purpose: the evidence for each diagnosis D1-D6 of docs/plan_phase_c_failing_criteria.md, in one generated record
         (docs/phase_c_diagnostic_record.md) that never gates and never proves (D0).
Spec:    docs/plan_phase_c_failing_criteria.md#D0
Tests:   tests/bowen/test_ensemble_record.py::test_d0_diagnostic_record_is_current,
         tests/bowen/test_ensemble_record.py::test_d0_diagnostics_never_count_as_proof

    python3 tools/diagnostic_record.py [--workers N]

A diagnostic either runs criteria (failing ones included) under the unmodified model or a mutant from
tools/mutation_record.py, or runs a probe from tools/probes.py. Results are cached like the mutation record's
(tools/record_cache.py): a probe's key holds the whole of tools/probes.py, so any edit to it reruns the probes.

The record's freshness test is in the default suite on purpose: the decision memo cites this record, and a stale one
would mislead. An engine edit therefore needs a rerun of this tool (minutes) as well as of the other records.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.mutant_runner import Mutant, run_in_copy  # noqa: E402
from tools.mutation_record import MUTANTS  # noqa: E402
from tools.record_cache import Cache, engine_hash, result_key, run_cached  # noqa: E402
import tools.probes as probes  # noqa: E402

RECORD = REPO / "docs" / "phase_c_diagnostic_record.md"
CONSTANTS = "config/bowen/constants.md"


@dataclass(frozen=True)
class Diagnostic:
    id: str
    step: str                      # the plan's step: D1-D6
    what: str
    mutant: str | None = None      # a mutant id from tools/mutation_record.py, or None for the unmodified model
    criteria: tuple[str, ...] = () # criterion ids, or prefixes ending in "["
    probe: str | None = None       # a probe name from tools/probes.py
    seeds: int = 0                 # a probe's seed count ([I], declared here)
    constant: tuple[str, str] | None = None  # (name, value): one row of config/bowen/constants.md changed instead


def constant_mutant(name: str, value: str) -> Mutant:
    text = (REPO / CONSTANTS).read_text(encoding="utf-8")
    line = next(row for row in text.splitlines() if row.startswith(f"| `{name}` | "))
    old = "| ".join(line.split("| ")[:3])
    return Mutant(f"{name}={value}", "diagnostic", ("*",), CONSTANTS, old, f"| `{name}` | {value} ",
                  f"{name} set to {value}")


def mutant(d: Diagnostic) -> Mutant | None:
    if d.constant:
        return constant_mutant(*d.constant)
    return next(m for m in MUTANTS if m.id == d.mutant) if d.mutant else None


DIAGNOSTICS = (
    Diagnostic("d1-c29-third-person", "D1", "M11.C.29's three members: weeks above the chronic floor, peak excess, "
               "and peak symptom load against the onset threshold, per arm", probe="c29_third_person", seeds=20),
    Diagnostic("d2-c41-unmodified", "D2", "M11.C.41's four cells, unmodified, for comparison",
               criteria=("M11.C.41[",)),
    Diagnostic("d2-c41-availability-fixed", "D2", "M11.C.41's four cells with M4.D.3a's availability made "
               "level-independent (every layer fully available): does the reactive share then rise at a lower level?",
               mutant="availability-level-independent", criteria=("M11.C.41[",)),
    Diagnostic("d3-triangle-relief", "D3", "What the learner credits a TRIANGLE with, against every other automatic "
               "act, in M11.C.42's and M11.C.45's arms: closed signal, share relieved, learned value at the end",
               probe="triangle_relief", seeds=20),
    Diagnostic("d3-triangle-relief-old-roles", "D3", "The same under step 5's alliance reading of TRIANGLE (the other "
               "parent loaded), for comparison", mutant="triangle-roles-swapped", probe="triangle_relief", seeds=20),
    Diagnostic("d4-c27-deviation-terms", "D4", "M11.C.27's pair deviation at the end of each arm, by side and term, "
               "per member", probe="c27_deviation_terms", seeds=20),
    Diagnostic("d5-c44-act-counts", "D5", "The act counts behind M11.C.44's outside/inside ratio, per arm",
               probe="c44_act_counts", seeds=50),
    Diagnostic("d6-horizons", "D6", "M11.C.4 with its nodal event later and M11.C.5 with a longer run: treatment "
               "minus baseline per readout, every other setting the criterion's own", probe="horizons", seeds=20),
    Diagnostic("d1-c29-third-person-calm", "D1", "The same as d1-c29-third-person with the declared spell removed "
               "from both arms: does the third person then stay below threshold?", probe="c29_third_person_calm",
               seeds=20),
    Diagnostic("d3-triangle-relief-own-relief-only", "D3", "The same as d3-triangle-relief with M4.D.6e's cross-person "
               "weight set to 0, so an act is credited with the actor's own relief only: is the loaded third's "
               "distress what makes TRIANGLE a cost?", constant=("cross_person_weight", "0.0"),
               probe="triangle_relief", seeds=20),
    Diagnostic("d0-scripted-acts", "D0", "Every criterion whose arms script an act: in how many seeds it was made, or "
               "skipped because it was not legal that week", probe="scripted_acts", seeds=20),
    Diagnostic("d5-spell-effect", "D5", "Is M11.C.44/.45's calm arm calm? Each triad member's mean acute anxiety and "
               "share of weeks above the chronic floor, per arm", probe="spell_effect", seeds=20),
)


def ids_for(d: Diagnostic) -> list[str]:
    from src.bowen.ensemble.criteria import CRITERIA

    return [cid for cid in CRITERIA if any(cid == c or (c.endswith("[") and cid.startswith(c)) for c in d.criteria)]


def probe_source(name: str) -> str:
    """The whole of tools/probes.py, and the probe's name: a probe reads helpers, other probes and module constants
    (a narrower key missed `horizons`' constants), so any edit to the file reruns the probes, which take minutes."""
    return f"{name}\n" + inspect.getsource(probes)


def probe_key(engine: str, d: Diagnostic) -> str:
    m = mutant(d)
    definition = {"edits": m.definition() if m else None, "probe": probe_source(d.probe), "seeds": d.seeds}
    return result_key(engine, definition, f"probe:{d.probe}")


def results(cache: Cache, d: Diagnostic, workers: int | None) -> list[dict]:
    """A diagnostic's rows, from the cache; with ``workers``, missing ones are run first."""
    engine = engine_hash()
    if d.probe:
        key = probe_key(engine, d)
        if cache.get(key) is None:
            if workers is None:
                raise KeyError(f"{d.id}: no cached result for the current engine, edits and probe")
            rows = run_in_copy(mutant(d), "probes.py", "--probe", d.probe, "--seeds", str(d.seeds))
            if not rows:
                raise SystemExit(f"{d.id}: probe {d.probe} returned no rows")
            cache.put(key, {"rows": rows})
            cache.flush()
            print(f"{d.id}: ran", flush=True)
        return cache.get(key)["rows"]
    ids = ids_for(d)
    if not ids:
        raise SystemExit(f"{d.id}: names no criterion ({d.criteria})")
    if workers is not None:
        return run_cached(cache, mutant(d), ids, workers, label=d.id)
    m = mutant(d)
    rows = [cache.get(result_key(engine, m.definition() if m else None, cid)) for cid in ids]
    if None in rows:
        raise KeyError(f"{d.id}: no cached result for the current engine and edits")
    return rows


def summarise(rows: list[dict]) -> list[str]:
    """A probe's rows grouped by every text field except the seed: per number, the mean, the median, the maximum
    and the share of seeds above zero (a heavy tail moves the mean; the median and the share do not)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple((k, v) for k, v in r.items() if isinstance(v, str)), []).append(r)
    numbers = [k for k, v in rows[0].items() if not isinstance(v, str) and k != "seed"]
    names = [k for k, _ in next(iter(groups))]
    lines = ["| " + " | ".join(names + ["seeds"] + [f"{n} (mean / median / max / >0)" for n in numbers]) + " |",
             "|" + "---|" * (len(names) + 1 + len(numbers))]
    for group, members in groups.items():
        cells = [v for _, v in group] + [str(len(members))]
        for n in numbers:
            values = sorted(float(m[n]) for m in members if m[n] is not None)  # None: nothing to measure that seed
            if not values:
                cells.append("—")
                continue
            median = (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2
            above = sum(v > 0 for v in values) / len(values)
            cells.append(f"{sum(values) / len(values):.3g} / {median:.3g} / {values[-1]:.3g} / {above:.0%}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render(sections: list[tuple[Diagnostic, list[dict]]], engine: str) -> str:
    lines = [
        "# Phase C diagnostic record",
        "",
        "Generated by `tools/diagnostic_record.py`; do not edit. Evidence for the diagnoses in",
        "`docs/plan_phase_c_failing_criteria.md`. **Nothing here is a verdict**: a criterion run under a variant is",
        "reported, never gating and never counted as mutation proof (`tools/spec_coverage.py` reads only the",
        "mutation record). Results are cached (`docs/records_cache/diagnostic.json`) under the engine hash, each",
        "variant's edits and each probe's own source.",
        "",
        f"engine_hash: {engine}",
    ]
    for d, rows in sections:
        variant = (f"with `{d.constant[0]}` = {d.constant[1]}" if d.constant else
                   f"under `{d.mutant}`" if d.mutant else "unmodified")
        lines += ["", f"## {d.step} · `{d.id}`", "", f"{d.what}. Model: {variant}."]
        if d.probe:
            lines += ["", f"Probe `{d.probe}` over {d.seeds} seeds; per group over seeds: mean / median / maximum / "
                      "share above zero (a seed with nothing to measure is left out).", ""]
            lines += summarise(rows)
        else:
            lines += ["", "| Criterion | Outcome | Seeds | Readouts (mean difference ± half-width) |", "|---|---|---|---|"]
            for r in rows:
                cells = "; ".join(f"`{x['readout']}` {x['mean_difference']:+.3g} ± {x['half_width']:.2g}"
                                  for x in r.get("readouts", ())) or r.get("error", "—")
                lines.append(f"| `{r['criterion']}` | {r['outcome']} | {r['seeds']} | {cells} |")
    return "\n".join(lines) + "\n"


def build(cache: Cache, workers: int | None = None) -> str:
    return render([(d, results(cache, d, workers)) for d in DIAGNOSTICS], engine_hash())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    cache = Cache("diagnostic")
    text = build(cache, args.workers)
    cache.save()
    RECORD.write_text(text, encoding="utf-8")
    print(f"wrote {RECORD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
