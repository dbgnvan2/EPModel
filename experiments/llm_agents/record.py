"""The run record: what a run was asked, what it got and how it ended.

Purpose: write ``runs/<run_id>/run.json`` with every field of §8: the models requested and returned, the API and SDK
         versions, effort, thinking, ``max_tokens``, sampling, the time, the engine, code, template and label hashes,
         the arm, seed and permutations, the order the options were numbered in, the status and the counts (LX.10,
         LX.25). A record without ``option_order`` was written before it existed and is read as ``fixed`` (§13.2).
Spec:    docs/plan_v3_llm_agent_experiment.md#8
Tests:   tests/experiments/llm_agents/test_record.py::test_lx10_run_record_fields,
         tests/experiments/llm_agents/test_record.py::test_lx25_record_carries_option_order
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from experiments.llm_agents import files
from experiments.llm_agents.config import CONFIG, HERE

STATUSES = ("complete", "stopped_at_budget", "incomplete_transient", "invalid", "not_replayable", "stopped_permanent")
SAMPLING = ("default; not settable, not sent, not seedable (on API version 2023-06-01, the SDK's, claude-haiku-5-5 and "
            "claude-sonnet-5-5 return a 400 for any non-default temperature, top_p or top_k; on 2026-10-21 any value is a 400)")
FIELDS = ("run_id", "phase", "pool", "arm", "seed", "family", "weeks_planned", "weeks_completed", "model_requested",
          "models_returned", "api_version", "sdk_version", "effort", "thinking", "max_tokens", "sampling", "time",
          "engine_hash", "code_hash", "template_hash", "label_hash", "config_hash", "permutations", "option_order", "status",
          "status_detail", "counts", "selections", "fallback_share")


def file_hash(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def code_hash() -> str:
    """The experiment's own code: every module of the package, in name order."""
    return file_hash(*sorted(HERE.glob("*.py")))


def engine_hash() -> str:
    from tools.record_cache import engine_hash as of_engine

    return of_engine()[:12]


def make(*, run_id: str, phase: str, pool: str | None, arm: str, seed: int, family: str, weeks_planned: int,
         weeks_completed: int, model: str | None, models_returned, api_version: str | None, sdk_version: str | None,
         effort: str | None, thinking: str | None, max_tokens: int | None, permutations: dict | None,
         option_order: str | None, status: str,
         status_detail: str | None, counts: dict, selections: int, fallback_share: float) -> dict:
    """Purpose: the run record, with every field of §8 and nothing left implicit.
    Spec:    docs/plan_v3_llm_agent_experiment.md#8
    Tests:   tests/experiments/llm_agents/test_record.py::test_lx10_run_record_fields
    """
    if status not in STATUSES:
        raise ValueError(f"status {status!r} is not one of {STATUSES}")
    record = {
        "run_id": run_id, "phase": phase, "pool": pool, "arm": arm, "seed": seed, "family": family,
        "weeks_planned": weeks_planned, "weeks_completed": weeks_completed, "model_requested": model,
        "models_returned": sorted(set(models_returned)), "api_version": api_version, "sdk_version": sdk_version,
        "effort": effort, "thinking": thinking, "max_tokens": max_tokens,
        "sampling": SAMPLING if model else None, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_hash": engine_hash(), "code_hash": code_hash(),
        "template_hash": file_hash(CONFIG / "prompt_text.md"), "label_hash": file_hash(CONFIG / "act_labels.md"),
        "config_hash": file_hash(CONFIG / "llm.md", CONFIG / "banned_terms.md"),
        "permutations": permutations, "option_order": option_order, "status": status, "status_detail": status_detail, "counts": counts,
        "selections": selections, "fallback_share": fallback_share,
    }
    assert tuple(record) == FIELDS
    return record


def option_order(record: dict) -> str:
    """Purpose: the order a recorded run's options were numbered in; a record from before the field is ``fixed``.
    Spec:    docs/plan_v3_llm_agent_experiment.md#13.2
    Tests:   tests/experiments/llm_agents/test_cache.py::test_lx25_committed_pilot_replays_fixed
    """
    return record.get("option_order") or "fixed"


def write(record: dict, directory: Path, runs: Path = files.RUNS) -> Path:
    return files.write_text(directory / "run.json", json.dumps(record, indent=2, sort_keys=False) + "\n", runs=runs)
