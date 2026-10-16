"""Run Phase D part II criteria under declared mutants and write the committed Phase D mutation record.

Purpose: prove a passing Phase D criterion fails when the mechanism its spec row names is removed (`M11.1d`), as
         ``tools/mutation_record.py`` does for Phase C. Phase D's other mutation runs are reported in the plan, ad hoc.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.1a, #M11.1d, #M11.C.10
Tests:   tests/bowen/test_phase_d_mutation_record.py::test_m111d_phase_d_mutation_record_is_current

    python3 tools/phase_d_mutation_record.py [--workers N]

A mutant is a literal replacement applied to a temporary copy of the repository, never to the working tree, and must
match exactly once (`M11.1a`). Only criteria that pass in ``docs/phase_d2_ensemble_record.md`` are run. Results are
cached in ``docs/records_cache/phase_d_mutation.json`` under the hash of the engine, the config and
``tools/phase_d_mutant_child.py`` and of ``apply`` and ``run_in_copy`` (which copy the repository and apply the edits),
with each mutant's edits; the record is rendered from the cache. Outcomes are judged as in the Phase C record
(``tools/mutation_record.py``'s ``judge``): an engine exception under a mutant (RAISED) is a red, reported and never
cached, as is a broken replacement (BROKEN, no evidence); a failure of the machine running the child stops the build
after keeping what finished. A passing criterion with no mutant here must be named in ``AD_HOC`` with where its
mutation is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.mutant_runner import Mutant, apply, run_in_copy  # noqa: E402,F401
from tools.mutation_record import NAMED, judge  # noqa: E402
from tools.record_cache import ERRORED, INFRASTRUCTURE_ERRORS, Cache, result_key  # noqa: E402

RECORD = REPO / "docs" / "phase_d_mutation_record.md"
ENSEMBLE_RECORD = REPO / "docs" / "phase_d2_ensemble_record.md"
CHILD = REPO / "tools" / "phase_d_mutant_child.py"
MUTANTS = (
    Mutant("witnessed-cutoff-learning-removed", NAMED, ("M11.C.10[family of origin]",), "config/bowen/constants.md",
           "| `witnessed_cutoff_gain` | 0.05 |", "| `witnessed_cutoff_gain` | 0.0 |",
           "M4.D.6f removed: a witness's learned CUTOFF value takes no step toward what followed a witnessed cutoff"),
)
# Passing part II criteria with no mutant in this record, and where their mutation is reported.
AD_HOC = {
    "M11.C.15": "run ad hoc, `docs/implementation_plan_phase_d.md` §4.4 (Mutations)",
    "M11.C.47": "a premise; run ad hoc, `docs/implementation_plan_phase_d.md` §4.10 (Results, Mutations)",
}


def engine_hash() -> str:
    """The engine, config and rules (``code_hash``), the child, and the source of ``apply`` and ``run_in_copy``, which
    decide what is copied and how an edit lands. Not the rest of ``tools/mutant_runner.py``: that is Phase C's child."""
    from tools.ensemble_record import code_hash

    digest = hashlib.sha256(code_hash(CHILD).encode())
    for function in (apply, run_in_copy):
        digest.update(inspect.getsource(function).encode())
    return digest.hexdigest()


def passing() -> set[str]:
    """The ``CRITERIA_D2`` keys that pass in the part II record. The record names two criteria by one id, so its rows
    are matched to the keys in order, each row's id checked against its key."""
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2

    text = ENSEMBLE_RECORD.read_text(encoding="utf-8")
    rows = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
    keys = list(CRITERIA_D2)
    if len(rows) != len(keys) or any(CRITERIA_D2[k].id != r["criterion"] for k, r in zip(keys, rows)):
        raise SystemExit(f"{ENSEMBLE_RECORD.name}'s rows no longer match CRITERIA_D2 in order; rerun its tool")
    return {k for k, r in zip(keys, rows) if r["outcome"] == "PASS"}


def build(cache: Cache, workers: int | None = None) -> str:
    """The record, from the cache. With ``workers``, missing results are run first; without, a missing one raises
    KeyError, so a test can check the committed record without running anything."""
    passed, engine = passing(), engine_hash()
    rows, skipped = [], []
    for mutant in MUTANTS:
        run_ids = [c for c in mutant.criteria if c in passed]
        skipped += [(mutant, c) for c in mutant.criteria if c not in passed]
        keys = {c: result_key(engine, mutant.definition(), c) for c in run_ids}
        missing = [c for c in run_ids if cache.get(keys[c]) is None]
        if missing and workers is None:
            raise KeyError(f"{mutant.id}: no cached result for the current engine and edits")
        fresh = {}
        if missing:
            for result in run_in_copy(mutant, CHILD.name, "--child", *missing, "--workers", str(workers)):
                if result.get("error", "").split(":")[0] in INFRASTRUCTURE_ERRORS:
                    cache.flush()  # keep the rows this child finished before it failed
                    raise SystemExit(f"{mutant.id} → {result['criterion']}: the run failed, not the model "
                                     f"({result['error']}); nothing was cached for it")
                fresh[result["criterion"]] = result
                if result["outcome"] not in ERRORED:  # an errored row is reported but rerun next time, never cached
                    cache.put(keys[result["criterion"]], result)
            cache.flush()
        for c in run_ids:
            result = cache.get(keys[c]) or fresh.get(c)
            if result is None:
                raise SystemExit(f"{mutant.id} → {c}: the child printed no result")
            rows.append((mutant, result))
    recorded = {r["criterion"] for _, r in rows}
    unproved = sorted(passed - recorded - set(AD_HOC))
    if unproved:
        raise SystemExit(f"passing in part II with no mutant here and none named in AD_HOC: {unproved}")
    return render(rows, skipped, engine)


def render(rows, skipped, engine: str) -> str:
    lines = [
        "# Phase D mutation record",
        "",
        "Generated by `tools/phase_d_mutation_record.py`; do not edit. Each mutant is a literal replacement applied to a",
        "temporary copy of the repository and run against a criterion that passes in `docs/phase_d2_ensemble_record.md`,",
        "with the same adaptive rules. A mutant is **red** when the criterion stops passing and **survived** when it still",
        "passes; a red mutant is **reversed** when every gating readout's interval lies wholly on the side opposite its",
        "direction (`M11.1d`). Phase D's other mutation runs are ad hoc and reported in",
        "`docs/implementation_plan_phase_d.md`.",
        "",
        f"engine_hash: {engine}",
        "",
        "| Mutant | Kind | What it changes | Criterion | Verdict under mutant | Seeds | Difference under mutant | Result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mutant, result in rows:
        note = f" ({result['error']})" if result.get("error") else ""
        diffs = "; ".join(f"`{x['readout']}` {x['mean_difference']:+.3g} ± {x['half_width']:.2g}"
                          + (f", p {x['p']:.2g}" if "p" in x else "") + (" (reported)" if x.get("report_only") else "")
                          for x in result.get("readouts", ()))
        lines.append(f"| `{mutant.id}` | {mutant.kind} | {mutant.what} | `{result['criterion']}` | {result['outcome']}{note} "
                     f"| {result['seeds']} | {diffs or '—'} | "
                     f"**{judge(mutant, result['outcome'], result.get('readouts', ()))}** |")
    lines += ["", "## Not run", "", "Criteria a mutant targets that do not pass in the part II record.", ""]
    lines += [f"- `{m.id}` → `{cid}`" for m, cid in skipped] or ["- none"]
    lines += ["", "## Passing with no mutant in this record", "",
              "Part II criteria that pass but are not proved here; where their mutation is reported instead.", ""]
    lines += [f"- `{cid}`: {where}" for cid, where in sorted(AD_HOC.items())] or ["- none"]
    lines += ["", "## Machine-readable", "", "```json",
              json.dumps([{"mutant": m.id, "kind": m.kind, "criterion": r["criterion"], "outcome": r["outcome"],
                           "seeds": r["seeds"], "result": judge(m, r["outcome"], r.get("readouts", ())),
                           "readouts": r.get("readouts", []), **({"error": r["error"]} if r.get("error") else {})}
                          for m, r in rows], indent=1, default=float),
              "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    cache = Cache("phase_d_mutation")
    text = build(cache, args.workers)
    cache.save()
    RECORD.write_text(text, encoding="utf-8")
    print(f"wrote {RECORD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
