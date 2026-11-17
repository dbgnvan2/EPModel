"""Cache the generated Phase C records' results, so a regeneration reruns only what changed.

Purpose: a full regeneration took 35-45 minutes and any edit to a tool forced one (2026-10-09, learnings P38). A
         cached result is keyed by the engine hash, the variant's own edits and the criterion, and a record is
         re-rendered from the cache in under a second.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.1d, #M13.4; docs/implementation_plan_version_2.md#S1.5a
Tests:   tests/bowen/test_ensemble_record.py::test_m111d_a_result_is_keyed_on_the_engine_and_the_mutants_edits,
         tests/bowen/test_ensemble_record.py::test_m111d_a_raised_or_broken_row_exits_nonzero,
         tests/bowen/test_mutant_runner.py::test_s1_5a_a_failed_child_does_not_stop_the_record,
         tests/bowen/test_mutant_runner.py::test_s1_5a_a_cache_write_cut_short_leaves_the_last_whole_file

A child that fails or times out does not stop the record (S1.5a): each id it printed no row for becomes a FAILED row,
never cached, the build goes on, and the tool exits 1 (``exit_status``). A RAISED or BROKEN row is written into the
record and never cached either, so it exits 1 too (cold sweep, 2026-10-20).

The engine hash (``engine_hash``) covers everything under ``src/bowen`` (``render/`` aside, S1.8a) and ``config/bowen``,
``tools/ensemble_record.py`` (RULE_KEYS) and ``tools/mutant_runner.py`` (the child that produces each result). This
file, the mutant list and the renderers are not in it: changing them reruns nothing unless a mutant's edits change.
A change to the engine reruns everything. The caches are committed in ``docs/records_cache/``; a record's test
re-renders it from its cache and compares.
"""

from __future__ import annotations

import builtins
import contextlib
import concurrent.futures
import concurrent.futures.process
import concurrent.futures.thread
import fcntl
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
# A row names its error class as text ("FileNotFoundError: ..."); the class is looked up and tested with issubclass, so
# a subclass of one of these counts too (T-069). KeyboardInterrupt is not one: it is the user's stop, not a fault.
INFRASTRUCTURE_ERRORS = (OSError, MemoryError, TimeoutError, concurrent.futures.BrokenExecutor)
ENGINE_TOOLS = (REPO / "tools" / "mutant_runner.py",)  # RULE_KEYS is in every hash (code_hash), not by file
CHILD = "mutant_runner.py"
META = "_meta"  # a cache file's own facts beside its results; a result key is a sha256 hex digest, so never this
_uncached: list[tuple[str, str]] = []  # this process's errored rows, (outcome, "label → id"), for ``exit_status``


def is_infrastructure_error(name: str) -> bool:
    """Whether the error class called ``name`` (a builtin or a ``concurrent.futures`` class) is an error of the machine."""
    cls = getattr(builtins, name, None) or getattr(concurrent.futures.process, name, None) \
        or getattr(concurrent.futures.thread, name, None) or getattr(concurrent.futures, name, None)
    return isinstance(cls, type) and issubclass(cls, INFRASTRUCTURE_ERRORS)


def engine_hash() -> str:
    from tools.ensemble_record import code_hash

    return code_hash(*ENGINE_TOOLS)


def result_key(engine: str, definition: dict | None, item: str) -> str:
    payload = json.dumps({"engine": engine, "definition": definition, "item": item}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


@contextlib.contextmanager
def _locked(path: Path):
    """An exclusive lock on the directory holding the cache file ``path``: it makes no file, so nothing is left in the
    repository, and one directory is one lock for every cache in it (a flush is short)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing releases the lock


class Cache:
    """A JSON file of results by key. ``save`` keeps only the keys used since loading, so stale entries go. ``meta`` is
    the file's own facts, kept under ``META`` beside the results and never a result key (the sweep's NumPy version)."""

    def __init__(self, name: str):
        self.path = CACHE_DIR / f"{name}.json"
        self.entries, self.meta = read_cache_file(self.path) if self.path.exists() else ({}, {})
        self.used: set[str] = set()
        self.loaded = set(self.entries)  # the keys this process read; a key it dropped later stays dropped

    def get(self, key: str) -> dict | None:
        self.used.add(key)
        return self.entries.get(key)

    def put(self, key: str, result: dict) -> None:
        self.used.add(key)  # stored as it will be reloaded (sorted keys), so a fresh and a cached result render alike
        self.entries[key] = json.loads(json.dumps(result, sort_keys=True, default=float))

    def flush(self) -> None:
        """Write every entry, pruning nothing: called after each run, so a later failure loses no finished result.
        Under a lock, results another run wrote since this one loaded are kept (a key this run dropped stays dropped), so two runs of one
        tool at once keep each other's results (T-029)."""
        with _locked(self.path):
            if self.path.exists():
                try:
                    on_disk = read_cache_file(self.path)[0]
                    self.entries = {**{k: v for k, v in on_disk.items() if k not in self.loaded}, **self.entries}
                except (OSError, ValueError):
                    pass  # an unreadable file is overwritten whole, as before
            write_cache_file(self.path, self.entries, self.meta)

    def save(self) -> None:
        """Write the entries used since loading: called once a whole record has been built. Under the flush's lock,
        a result another run wrote since this one loaded is kept, so a concurrent run's results are not pruned (T-029)."""
        with _locked(self.path):
            kept = {k: self.entries[k] for k in self.used if k in self.entries}
            if self.path.exists():
                try:
                    on_disk = read_cache_file(self.path)[0]
                    kept = {**{k: v for k, v in on_disk.items() if k not in self.loaded}, **kept}
                except (OSError, ValueError):
                    pass
            write_cache_file(self.path, kept, self.meta)


def read_cache_file(path: Path) -> tuple[dict, dict]:
    """A cache file's results and its ``META`` (empty when the file has none)."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    return entries, entries.pop(META, {})


def write_cache_file(path: Path, entries: dict, meta: dict) -> None:
    """Write a whole file beside the cache and move it into place, so a kill mid-write leaves the last whole file, not
    a cut one the next run cannot read (security pass, 2026-10-19). An empty ``meta`` writes no ``META`` key."""
    if META in entries:
        raise ValueError(f"{path.name}: {META} is the file's own facts, never a result key")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")  # one per run (git ignores it)
    payload = {**({META: meta} if meta else {}), **entries}
    partial.write_text(json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _is_row(r: dict) -> bool:
    """A printed line is a row when it names its criterion and outcome as text, and any error is text: anything
    else would stop the record with a ``KeyError`` or ``AttributeError`` below (fix re-sweep, 2026-10-20)."""
    return (isinstance(r.get("criterion"), str) and isinstance(r.get("outcome"), str)
            and isinstance(r.get("error", ""), str))


# A child's limit in seconds (T-254, 2026-11-16), set from times measured on Mac Mini1 at 8 workers at spec revision 63.
# A Phase C child keeps ``run_in_copy``'s 7200 s. A Phase D child running one or two criteria gets three times the
# longest row, `M11.C.10[family of origin]`'s 7157 s; one running all of part II gets about twice that part's
# measured 17,025 s (the record's part II step, 07:54-12:38 on 2026-11-16). A child that hits its limit fails with
# "timed out after N s" and its rows are rerun next time. Outside every hash: a limit decides nothing a row says.
DEFAULT_TIMEOUT = 7200
PHASE_D_CHILD = "phase_d_mutant_child.py"
PHASE_D_ROW_TIMEOUT = 21600
PHASE_D_PART_TIMEOUT = 36000


def timeout_for(child: str, whole_part: bool) -> int:
    """The limit for ``child``: Phase C's default, or Phase D's for a few rows or for the whole of part II."""
    if child != PHASE_D_CHILD:
        return DEFAULT_TIMEOUT
    return PHASE_D_PART_TIMEOUT if whole_part else PHASE_D_ROW_TIMEOUT


def child_rows(mutant: Mutant | None, tool: str, ids: list[str], workers: int, label: str,
               timeout: int = DEFAULT_TIMEOUT) -> list[dict]:
    """One child's row for each of ``ids``, in order. An id it printed no row for, because it failed or timed out
    first, gets a ``FAILED`` row, as does a row reporting an error of the machine; the caller goes on (S1.5a).
    ``timeout`` is the child's limit in seconds, ``run_in_copy``'s default unless the record sets its own."""
    try:
        printed, failure = run_in_copy(mutant, tool, "--child", *ids, "--workers", str(workers), timeout=timeout), None
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
        elif is_infrastructure_error(row.get("error", "").split(":")[0]):
            row = {**row, "outcome": FAILED}
        if row["outcome"] in ERRORED:
            _uncached.append((row["outcome"], f"{label} → {cid}"))
        if row["outcome"] == FAILED:
            print(f"{label} → {cid}: the run failed, not the model ({row['error']}); not cached, rerun next time",
                  flush=True)
        out.append(row)
    return out


def exit_status() -> int:
    """A record tool's exit status: 1 when any row this run FAILED, RAISED or was BROKEN, else 0. Each is written into
    the record and not cached, so the record's test fails until the row gives a result (cold sweep, 2026-10-20: a
    RAISED or BROKEN row exited 0, and a row that raises on every run left a record that could never be committed)."""
    failed = [row for outcome, row in _uncached if outcome == FAILED]
    errored = [f"{row} ({outcome})" for outcome, row in _uncached if outcome != FAILED]
    if failed:
        print(f"{len(failed)} row(s) failed and were not cached; rerun the tool to complete the record:",
              *failed, sep="\n  ", flush=True)
    if errored:
        print(f"{len(errored)} row(s) raised or did not import and were not cached; the record is not current until "
              "each gives a result, and rerunning will not change a row that raises every time:",
              *errored, sep="\n  ", flush=True)
    return 1 if _uncached else 0


def run_cached(cache: Cache, mutant: Mutant | None, ids: list[str], workers: int, label: str | None = None,
               child: str = CHILD, engine: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> list[dict]:
    """Each criterion's result under ``mutant`` (``None``: the unmodified model): from the cache where its key is
    present, otherwise run (all the missing ones in one child) and cached. Results come back in ``ids`` order.
    ``child`` is the tool that runs them and ``engine`` the hash their keys carry, which must cover that child;
    by default Phase C's child and ``engine_hash``. ``timeout`` is the child's limit (``timeout_for``)."""
    if engine is None and child != CHILD:
        raise ValueError(f"{child}: pass the hash its results are keyed by; engine_hash does not cover that child")
    engine = engine or engine_hash()
    label = label or (mutant.id if mutant else "unmodified")
    keys = {cid: result_key(engine, mutant.definition() if mutant else None, cid) for cid in ids}
    missing = [cid for cid in ids if cache.get(keys[cid]) is None]
    fresh = {}
    if missing:
        for result in child_rows(mutant, child, missing, workers, label, timeout=timeout):
            fresh[result["criterion"]] = result
            if result["outcome"] not in ERRORED:  # an errored row is reported but rerun next time, never cached
                cache.put(keys[result["criterion"]], result)
        cache.flush()  # keep every finished row, whatever happens to the next item
        print(f"{label}: ran {len(missing)} of {len(ids)}", flush=True)
    # The cached copy first: it is normalised as a reload gives it, so a record built in the run that produced its
    # results renders the same as its later re-render from the cache. An errored row is only in ``fresh``.
    return [cache.get(keys[cid]) or fresh.get(cid) for cid in ids]
