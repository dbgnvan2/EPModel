"""Cache the generated Phase C records' results, so a regeneration reruns only what changed.

Purpose: a full regeneration took 35-45 minutes and any edit to a tool forced one (2026-10-09, learnings P38). A
         cached result is keyed by the engine hash, the variant's own edits and the criterion, and a record is
         re-rendered from the cache in under a second.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.1d, #M13.4
Tests:   tests/bowen/test_ensemble_record.py::test_m111d_a_result_is_keyed_on_the_engine_and_the_mutants_edits

The engine hash (``engine_hash``) covers everything under ``src/bowen`` and ``config/bowen``,
``tools/ensemble_record.py`` (RULE_KEYS) and ``tools/mutant_runner.py`` (the child that produces each result). This
file, the mutant list and the renderers are not in it: changing them reruns nothing unless a mutant's edits change.
A change to the engine reruns everything. The caches are committed in ``docs/records_cache/``; a record's test
re-renders it from its cache and compares.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.mutant_runner import Mutant, run_in_copy  # noqa: E402

CACHE_DIR = REPO / "docs" / "records_cache"
ERRORED = ("RAISED", "BROKEN")
# Errors of the machine running the child, not of the mutated model: never a result (learning-qa, 2026-10-09).
INFRASTRUCTURE_ERRORS = ("BrokenProcessPool", "MemoryError", "OSError", "TimeoutError", "KeyboardInterrupt")
ENGINE_TOOLS = (REPO / "tools" / "ensemble_record.py", REPO / "tools" / "mutant_runner.py")


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
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(sorted(entries.items())), indent=1, sort_keys=True, default=float) + "\n",
                             encoding="utf-8")


def run_cached(cache: Cache, mutant: Mutant, ids: list[str], workers: int, label: str | None = None) -> list[dict]:
    """Each criterion's result under ``mutant``: from the cache where its key is present, otherwise run (all the
    missing ones in one child) and cached. Results come back in ``ids`` order."""
    engine = engine_hash()
    keys = {cid: result_key(engine, mutant.definition(), cid) for cid in ids}
    missing = [cid for cid in ids if cache.get(keys[cid]) is None]
    fresh = {}
    if missing:
        for result in run_in_copy(mutant, "mutant_runner.py", "--child", *missing, "--workers", str(workers)):
            error = result.get("error", "")
            if error.split(":")[0] in INFRASTRUCTURE_ERRORS:
                cache.flush()  # keep the rows this child finished before it failed
                raise SystemExit(f"{label or mutant.id} → {result['criterion']}: the run failed, not the model "
                                 f"({error}); nothing was cached for it")
            fresh[result["criterion"]] = result
            if result["outcome"] not in ERRORED:  # an errored row is reported but rerun next time, never cached
                cache.put(keys[result["criterion"]], result)
        cache.flush()
        print(f"{label or mutant.id}: ran {len(missing)} of {len(ids)}", flush=True)
    return [fresh.get(cid) or cache.get(keys[cid]) for cid in ids]
