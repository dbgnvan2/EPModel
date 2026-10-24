"""Run the Phase D acceptance criteria built so far over adaptive ensembles and write the committed records.

Purpose: produce docs/phase_d_ensemble_record.md — each Phase D criterion's verdict, seeds, per-readout difference,
         interval and p, fallback rate and flags — stamped with a hash of the code and config it ran against (Phase D
         plan step 12). The mutation runs are in docs/phase_d_mutation_record.md (tools/phase_d_mutation_record.py).
Spec:    docs/bowen_agent_model_spec_v2.md#M17.A.1, #M11.D.18, #M11.5, #M13.4;
         docs/implementation_plan_version_2.md#S1.7b, #S1.8c
Tests:   tests/bowen/test_phase_d_record.py::test_phase_d_record_is_current,
         tests/bowen/test_phase_d_record.py::test_s1_8c_records_render_from_the_cache,
         tests/bowen/test_phase_d_record.py::test_s1_8c_a_stale_entry_is_rerun,
         tests/bowen/test_phase_d_record.py::test_s1_7b_variants_print_under_their_own_keys

    python3 tools/phase_d_record.py [--part 1|2] [--workers N] [--only ID ...]

Part 1 writes docs/phase_d_ensemble_record.md (part I); part 2 writes docs/phase_d2_ensemble_record.md, the life
course's criteria (part II, spec revision 28, plan §4.4).

Each verdict is cached in docs/records_cache/ (version 2 stage 1, S1.8c), keyed as the Phase C records' results are:
by the hash of the code that computes a verdict (``result_hash``: ``src/bowen`` outside ``render/``,
``config/bowen`` and the rules), and the criterion's key. This tool and the renderer are not in the key, so a change
to either re-renders the record from the cache in seconds; a change to the engine reruns every criterion. ``--only``
runs only the named criteria that have no result; the record is written when every criterion has one. Each row is
printed under its ``CRITERIA_D`` or ``CRITERIA_D2`` key, so a criterion's variants are separate rows (S1.7b).

Generated, never hand-edited; a default-suite test re-renders the record from its cache and fails when it differs.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.record_cache import Cache, result_key  # noqa: E402

RECORD = REPO / "docs" / "phase_d_ensemble_record.md"
RECORDS = {1: RECORD, 2: REPO / "docs" / "phase_d2_ensemble_record.md"}
CACHES = {1: "phase_d_ensemble", 2: "phase_d2_ensemble"}  # docs/records_cache/<name>.json

_spec = importlib.util.spec_from_file_location("tools.ensemble_record", REPO / "tools" / "ensemble_record.py")
_record = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_record)

# Phase D part I criteria with no arm yet, and why (docs/implementation_plan_phase_d.md, step 10).
NOT_BUILT = {
    "M11.C.9": "no rule reads sibling position, so building it means a new mechanism, not a gate (`M1.A.14c`); "
               "out of version 1, Claude's recommendation 2026-10-17, decided by the owner 2026-10-17 "
               "(`docs/version_1_dispositions.md`)",
    "M11.C.28": "the sink allocation feeds no behaviour, so pinning its mobility can change no symptom",
}


# Part II criteria not run as an ensemble, and why (plan §4.4).
NOT_RUN_2 = {
    "M11.C.37": "a conservation check with no second arm (an `M11.4` exception); proved by "
                "`tests/bowen/test_life_course.py::test_m11c37_anxiety_conserved_across_death` and its mutant",
}


def criteria_for(part: int) -> dict:
    if part == 1:
        from src.bowen.ensemble.criteria_d import CRITERIA_D
        return CRITERIA_D
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2
    return CRITERIA_D2


def result_hash() -> str:
    """The hash of the code that computes a verdict: ``src/bowen`` outside ``render/``, ``config/bowen`` and the rules
    (``code_hash`` with no tool). The record prints it; each cached result is keyed by it (S1.8c)."""
    return _record.code_hash()


def build(part: int, cache: Cache, workers: int | None = None, only: list[str] | None = None) -> str:
    """Purpose: the record for ``part``, from the cache. With ``workers``, a criterion with no result for the current
             code is run first and cached (with ``only``, only those it names); a criterion still without a result
             raises KeyError, so a test can check the committed record without running anything.
    Spec:    docs/implementation_plan_version_2.md#S1.8c, #S1.7b
    Tests:   tests/bowen/test_phase_d_record.py::test_s1_8c_records_render_from_the_cache,
             tests/bowen/test_phase_d_record.py::test_s1_8c_a_stale_entry_is_rerun
    """
    from src.bowen.ensemble.runner import Verdict, run_criterion
    from src.bowen.io.load import load_constants

    constants = load_constants()
    rules = {k: constants[k] for k in _record.RULE_KEYS}
    engine = result_hash()
    verdicts = []
    for cid, criterion in criteria_for(part).items():
        key = result_key(engine, None, cid)
        entry = cache.get(key)
        if entry is None and workers is not None and (not only or any(cid.startswith(o) for o in only)):
            started = time.time()
            # S1.7b: the row carries the criterion's key, so two variants that share an id are two rows.
            verdict = dataclasses.replace(run_criterion(criterion, rules, workers=workers), criterion=cid)
            cache.put(key, dataclasses.asdict(verdict))
            cache.flush()  # keep every finished result, whatever happens to the next criterion
            entry = cache.get(key)
            print(f"{cid}: {verdict.outcome} ({verdict.seeds} seeds, {time.time() - started:.0f} s)", flush=True)
        if entry is None:
            raise KeyError(cid)
        verdicts.append((Verdict(**entry), None))
    return render(verdicts, rules, engine, part)


REWORDED = (
    ("# Phase C ensemble record", "# Phase D ensemble record (part I)"),
    ("Generated by `tools/ensemble_record.py`", "Generated by `tools/phase_d_record.py`"),
    ("The constants are the Phase C freeze of 2026-10-07.",
     "Spec revisions 24–27 are Claude's recommendations, approved by the owner 2026-10-16 (revision 31)."),
    ("## Not built in Phase C", "## Not built in Phase D part I"),
)


REWORDED_2 = (
    ("# Phase C ensemble record", "# Phase D ensemble record (part II, the life course)"),
    ("Generated by `tools/ensemble_record.py`", "Generated by `tools/phase_d_record.py --part 2`"),
    ("The constants are the Phase C freeze of 2026-10-07.",
     "Spec revision 28 applies the owner's life-course decisions of 2026-10-13; its forms are Claude's recommendation, "
     "approved by the owner 2026-10-16."),
    ("## Not built in Phase C", "## Not run as an ensemble in part II"),
)


# A known defect printed beside a verdict, in this record's verdict cell and in the sweep record's central-verdict cell
# (tools/version_2_sweep.py), until the defect is fixed and the records regenerated (milestone review MR-3). A note
# changes no verdict and nothing a run computes reads it; neither tool is in a cache key, so editing this re-renders
# the records without a rerun. Remove a key's entry when the items it names close.
VERDICT_NOTES = {
    "M11.C.48": "not evidence (T-123); named mutant survived (T-119)",
    "M11.C.48[phase_d]": "not evidence (T-123); named mutant survived (T-119)",
}


def noted(key: str, verdict: str) -> str:
    """``verdict`` with ``key``'s note from ``VERDICT_NOTES`` beside it, or ``verdict`` alone."""
    return f"{verdict} ({VERDICT_NOTES[key]})" if key in VERDICT_NOTES else verdict


def render(verdicts, rules, hash_: str, part: int = 1) -> str:
    """The Phase C renderer's page, with its four Phase C phrases replaced; each must be found exactly once, so a
    reworded renderer fails here rather than leaving Phase C text in this record (review 2026-10-13). A verdict with a
    known defect carries its ``VERDICT_NOTES`` note in its cell."""
    text = _record.render(verdicts, NOT_BUILT if part == 1 else NOT_RUN_2, rules, hash_)
    for old, new in (REWORDED if part == 1 else REWORDED_2):
        if text.count(old) != 1:
            raise SystemExit(f"tools/ensemble_record.py's page no longer contains {old!r} once; update REWORDED")
        text = text.replace(old, new)
    lines = text.split("\n")
    for v, _ in verdicts:
        if v.criterion not in VERDICT_NOTES:
            continue
        rows = [i for i, line in enumerate(lines) if line.startswith(f"| `{v.criterion}` | {v.cls} | **{v.outcome}** |")]
        if len(rows) != 1:
            raise SystemExit(f"{v.criterion}: {len(rows)} verdict rows to note, not one")
        old = f"| **{v.outcome}** |"
        lines[rows[0]] = lines[rows[0]].replace(old, f"| {noted(v.criterion, f'**{v.outcome}**')} |", 1)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--part", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    cache = Cache(CACHES[args.part])
    try:
        text = build(args.part, cache, workers=args.workers, only=args.only)
    except KeyError as missing:
        raise SystemExit(f"no result for {missing} on the current code: run without --only, or name it") from None
    cache.save()  # every criterion's entry was used; entries for older code go
    record = RECORDS[args.part]
    record.write_text(text, encoding="utf-8")
    print(f"wrote {record.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
