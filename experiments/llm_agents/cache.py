"""The response cache: one gzipped JSON-lines file per run, keyed by the run and the request.

Purpose: store every response under ``sha256(run_id, model, arm, seed, person token, week, attempt, request body)``,
         so two runs never share an answer (LX.9) and a replay finds each call or knows the prompt moved (LX.8).
Spec:    docs/plan_v3_llm_agent_experiment.md#8
Tests:   tests/experiments/llm_agents/test_cache.py::test_lx9_run_key_separates_entries,
         tests/experiments/llm_agents/test_cache.py::test_lx8_miss_is_not_replayable

The plan's key is ``(model, arm, seed, person token, week, attempt, body)``; the run id is added, so a phase rerun
under another run id (a pilot of 26 weeks against the main phase's 208) keeps its own entries too.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
from pathlib import Path

from experiments.llm_agents import files


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(run_id: str, model: str, arm: str, seed: int, person_token: str, week: int, attempt: int,
              body: dict) -> str:
    parts = [run_id, model, arm, str(seed), person_token, str(week), str(attempt), canonical_json(body)]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResponseCache:
    """The entries of one run. ``save`` rewrites the file whole, atomically, in key order (a stable diff)."""

    def __init__(self, path: Path | None, entries: dict | None = None, runs: Path = files.RUNS):
        self.path, self.runs = path, runs
        self.entries: dict[str, dict] = dict(entries or {})
        self.new = 0
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: Path, runs: Path = files.RUNS) -> "ResponseCache":
        entries = {}
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        entry = json.loads(line)
                        entries[entry["key"]] = entry
        return cls(path, entries, runs)

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self.entries.get(key)

    def put(self, entry: dict) -> None:
        with self._lock:
            if entry["key"] in self.entries:
                raise KeyError(f"cache key {entry['key']} written twice")
            self.entries[entry["key"]] = entry
            self.new += 1

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            lines = "".join(canonical_json(self.entries[k]) + "\n" for k in sorted(self.entries))
        # mtime=0: the same entries give the same bytes, so an unchanged cache is an unchanged file.
        files.write_bytes(self.path, gzip.compress(lines.encode("utf-8"), mtime=0), runs=self.runs)
