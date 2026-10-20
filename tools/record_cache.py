"""Cache the generated Phase C records' results, so a regeneration reruns only what changed.

Purpose: a full regeneration took 35-45 minutes and any edit to a tool forced one (2026-10-09, learnings P38). A
         cached result is keyed by the engine hash, the variant's own edits and the criterion, and a record is
         re-rendered from the cache in under a second.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.1d, #M13.4; docs/implementation_plan_version_2.md#S1.5a
Tests:   tests/bowen/test_ensemble_record.py::test_m111d_a_result_is_keyed_on_the_engine_and_the_mutants_edits,
         tests/bowen/test_mutant_runner.py::test_s1_5a_a_failed_child_does_not_stop_the_record,
         tests/bowen/test_mutant_runner.py::test_s1_5a_a_cache_write_cut_short_leaves_the_last_whole_file

A child that fails or times out does not stop the record (S1.5a): each id it printed no row for becomes a FAILED row,
never cached, the build goes on, and the tool exits 1 (``exit_status``).

The engine hash (``engine_hash``) covers everything under ``src/bowen`` (``render/`` aside, S1.8a) and ``config/bowen``,
``tools/ensemble_record.py`` (RULE_KEYS) and ``tools/mutant_runner.py`` (the child that produces each result). This
file, the mutant list and the renderers are not in it: changing them reruns nothing unless a mutant's edits change.
A change to the engine reruns everything. The caches are committed in ``docs/records_cache/``; a record's test
re-renders it from its cache and compares.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.child_process import ChildFailed  # noqa: E402
from tools.mutant_runner import Mutant, run_in_copy  # noqa: E402

CACHE_DIR = REPO / "docs" / "records_cache"
# The child failed, timed out or hit an error of the machine running it: no evidence about the model, so the row is
# reported, never cached, and rerun next time; the record run goes on to the next item (version 2 stage 1, S1.5a).
FAILED = "FAILED"
ERRORED = ("RAISED", "BROKEN", FAILED)
# Errors of the machine running the child, not of the mutated model: never a result (learning-qa, 2026-10-09).
INFRASTRUCTURE_ERRORS = ("BrokenProcessPool", "MemoryError", "OSError", "TimeoutError", "KeyboardInterrupt")
ENGINE_TOOLS = (REPO / "tools" / "mutant_runner.py",)  # RULE_KEYS is in every hash (code_hash), not by file
CHILD = "mutant_runner.py"
_failed: list[str] = []  # this process's FAILED rows, for ``exit_status``


def engine_hash() -> str:
    from tools.ensemble_record import code_hash

    return code_hash(*ENGINE_TOOLS)


def result_key(engine: str, definition: dict | None, item: str) -> str:
    payload = json.dumps({"engine": engine, "definition": definition, "item": item}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class Cache:
    """A JSON file of results by key. ``save`` keeps only the keys used since loading, so stale entries go."""

    def __init__(self, name: str):
        self.path = CACHE_DIR / f"{name}.json"
        self.entries = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.used: set[str] = set()

    def get(self, key: str) -> dict | None:
        self.used.add(key)
        return self.entries.get(key)

    def put(self, key: str, result: dict) -> None:
        self.used.add(key)  # stored as it will be reloaded (sorted keys), so a fresh and a cached result render alike
        self.entries[key] = json.loads(json.dumps(result, sort_keys=True, default=float))

    def flush(self) -> None:
        """Write every entry, pruning nothing: called after each run, so a later failure loses no finished result."""
        self._write(self.entries)

    def save(self) -> None:
        """Write the entries used since loading: called once a whole record has been built."""
        self._write({k: self.entries[k] for k in self.used if k in self.entries})

    def _write(self, entries: dict) -> None:
        """Write a whole file beside the cache and move it into place, so a kill mid-write leaves the last whole
        file, not a cut one the next run cannot read (security pass, 2026-10-19)."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(f"{self.path.name}.{os.getpid()}.partial")  # one per run (git ignores it)
        partial.write_text(json.dumps(dict(sorted(entries.items())), indent=1, sort_keys=True, default=float) + "\n",
                           encoding="utf-8")
        os.replace(partial, self.path)


def _is_row(r: dict) -> bool:
    """A printed line is a row when it names its criterion and outcome as text, and any error is text: anything
    else would stop the record with a ``KeyError`` or ``AttributeError`` below (fix re-sweep, 2026-10-20)."""
    return (isinstance(r.get("criterion"), str) and isinstance(r.get("outcome"), str)
            and isinstance(r.get("error", ""), str))


def child_rows(mutant: Mutant | None, tool: str, ids: list[str], workers: int, label: str) -> list[dict]:
    """One child's row for each of ``ids``, in order. An id it printed no row for, because it failed or timed out
    first, gets a ``FAILED`` row, as does a row reporting an error of the machine; the caller goes on (S1.5a)."""
    try:
        printed, failure = run_in_copy(mutant, tool, "--child", *ids, "--workers", str(workers)), None
    except ChildFailed as failed:
        printed, failure = failed.rows, str(failed.code)
        print(f"{label}: {failure}", flush=True)  # in full, with the child's error output; each row keeps its first line
    rows = {r["criterion"]: r for r in printed if _is_row(r)}
    stray = sum(1 for r in printed if not _is_row(r))
    if stray:  # a line that decodes but is not a row stops nothing; it is counted (security pass, 2026-10-19)
        print(f"{label}: {stray} printed line(s) that are not rows ignored", flush=True)
    out = []
    for cid in ids:
        row = rows.get(cid)
        if row is None:
            row = {"criterion": cid, "outcome": FAILED, "seeds": 0,
                   "error": f"ChildFailed: {(failure or 'the child printed no row').splitlines()[0]}"}
        elif row.get("error", "").split(":")[0] in INFRASTRUCTURE_ERRORS:
            row = {**row, "outcome": FAILED}
        if row["outcome"] == FAILED:
            _failed.append(f"{label} → {cid}")
            print(f"{label} → {cid}: the run failed, not the model ({row['error']}); not cached, rerun next time",
                  flush=True)
        out.append(row)
    return out


def exit_status() -> int:
    """A record tool's exit status: 1 when any row this run FAILED (the record is written, and is stale until those
    rows are rerun), else 0."""
    if _failed:
        print(f"{len(_failed)} row(s) failed and were not cached; rerun the tool to complete the record:",
              *_failed, sep="\n  ", flush=True)
    return 1 if _failed else 0


def run_cached(cache: Cache, mutant: Mutant | None, ids: list[str], workers: int, label: str | None = None) -> list[dict]:
    """Each criterion's result under ``mutant`` (``None``: the unmodified model): from the cache where its key is
    present, otherwise run (all the missing ones in one child) and cached. Results come back in ``ids`` order."""
    engine = engine_hash()
    label = label or (mutant.id if mutant else "unmodified")
    keys = {cid: result_key(engine, mutant.definition() if mutant else None, cid) for cid in ids}
    missing = [cid for cid in ids if cache.get(keys[cid]) is None]
    fresh = {}
    if missing:
        for result in child_rows(mutant, CHILD, missing, workers, label):
            fresh[result["criterion"]] = result
            if result["outcome"] not in ERRORED:  # an errored row is reported but rerun next time, never cached
                cache.put(keys[result["criterion"]], result)
        cache.flush()  # keep every finished row, whatever happens to the next item
        print(f"{label}: ran {len(missing)} of {len(ids)}", flush=True)
    # The cached copy first: it is normalised as a reload gives it, so a record built in the run that produced its
    # results renders the same as its later re-render from the cache. An errored row is only in ``fresh``.
    return [cache.get(keys[cid]) or fresh.get(cid) for cid in ids]
