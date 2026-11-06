"""Where the experiment may write: its runs directory and its report, nothing else.

Purpose: refuse any write outside ``experiments/llm_agents/runs/`` and ``docs/llm_agent_experiment_report.md`` (LX.1),
         so the experiment can never touch the engine, its config or the model's records.
Spec:    docs/plan_v3_llm_agent_experiment.md#10 (LX.1)
Tests:   tests/experiments/llm_agents/test_isolation.py::test_lx1_writes_only_under_runs

Every write in the package goes through ``write_bytes`` or ``write_text`` here; a test greps the package for any other.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "experiments" / "llm_agents" / "runs"
REPORT = REPO / "docs" / "llm_agent_experiment_report.md"


class OutsideRuns(PermissionError):
    """A write was asked for outside the experiment's runs directory and its report."""


def allowed(path: Path, runs: Path = RUNS, report: Path = REPORT) -> Path:
    target = Path(os.path.abspath(path))
    runs, report = Path(os.path.abspath(runs)), Path(os.path.abspath(report))
    if target == report or runs in target.parents:
        return target
    raise OutsideRuns(f"the experiment writes only under {runs} and to {report}, not {target}")


def write_bytes(path: Path, data: bytes, runs: Path = RUNS, report: Path = REPORT) -> Path:
    """Write atomically: to a sibling temporary file, then replace, so a crash leaves the old file whole."""
    target = allowed(path, runs, report)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    return target


def write_text(path: Path, text: str, runs: Path = RUNS, report: Path = REPORT) -> Path:
    return write_bytes(path, text.encode("utf-8"), runs, report)
