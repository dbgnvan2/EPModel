"""Apply a mutant to a temporary copy of the repository and run criteria there: the code that produces results.

Purpose: the one place that applies a mutant and runs criteria in a child process. Its content is part of the engine
         hash (``tools/record_cache.py``), so only code that changes what a run computes belongs here; caching,
         keys, rendering and how a child process is run (``tools/child_process.py``) live elsewhere and can
         change without a rerun.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.1a, #M11.1d, #M13.4; docs/implementation_plan_version_2.md#S1.5a
Tests:   tests/bowen/test_ensemble_record.py::test_m111a_every_mutant_applies_exactly_once
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BROKEN_ERRORS = (NameError, ImportError, SyntaxError)


@dataclass(frozen=True)
class Mutant:
    id: str
    kind: str
    criteria: tuple[str, ...]
    file: str
    old: str
    new: str
    what: str
    also: tuple[tuple[str, str, str], ...] = ()  # further (file, old, new) for a joint mutant (M11.1a: name every write)

    def definition(self) -> dict:
        """What the mutant changes: its key in the cache. ``what`` and ``criteria`` are labels, not changes."""
        return {"edits": [[self.file, self.old, self.new], *map(list, self.also)]}


def apply(root: Path, mutant: Mutant) -> None:
    for file, old, new in ((mutant.file, mutant.old, mutant.new), *mutant.also):
        path = root / file
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"mutant {mutant.id}: the replaced text occurs {count} times in {file}, not once")
        path.write_text(text.replace(old, new), encoding="utf-8")


def child(ids: list[str], workers: int) -> None:
    """Run in the mutated copy: print one JSON line per criterion."""
    from src.bowen.ensemble.criteria import CRITERIA
    from src.bowen.ensemble.runner import run_criterion
    from src.bowen.io.load import load_constants
    from tools.ensemble_record import RULE_KEYS

    constants = load_constants()
    rules = {k: constants[k] for k in RULE_KEYS}
    for cid in ids:
        try:
            v = run_criterion(CRITERIA[cid], rules, workers=workers)
            readouts = [{"readout": r["readout"], "direction": r["direction"], "mean_difference": r["mean_difference"],
                         "half_width": r["half_width"], "report_only": r.get("report_only", False)} for r in v.readouts]
            print(json.dumps({"criterion": cid, "outcome": v.outcome, "seeds": v.seeds, "readouts": readouts,
                              "missing_moves": v.missing_moves},
                             default=float), flush=True)
        except BROKEN_ERRORS as error:  # the replacement text itself is broken (an unimported name): no evidence
            print(json.dumps({"criterion": cid, "outcome": "BROKEN", "seeds": 0,
                              "error": f"{type(error).__name__}: {str(error)[:160]}"}), flush=True)
        except Exception as error:  # the engine raised (an invariant fired) on the mutated model: a red, with cause
            print(json.dumps({"criterion": cid, "outcome": "RAISED", "seeds": 0,
                              "error": f"{type(error).__name__}: {str(error)[:160]}"}), flush=True)


def run_in_copy(mutant: Mutant | None, tool: str, *args: str, timeout: int = 7200) -> list[dict]:
    """Run ``tools/<tool>`` with ``args`` in a temporary copy of the repository, ``mutant`` applied there (never to
    the working tree); return the JSON lines it prints. How the child is run, and what a failed or timed-out child
    raises, is ``tools/child_process.py`` (outside the engine hash, S1.5a)."""
    from tools.child_process import run_child

    label = mutant.id if mutant else "no mutant"
    with tempfile.TemporaryDirectory(prefix="bowen-mutant-") as tmp:
        root = Path(tmp)
        for part in ("src", "config", "tools"):
            shutil.copytree(REPO / part, root / part, ignore=shutil.ignore_patterns("__pycache__"))
        if mutant is not None:
            apply(root, mutant)
        return run_child([sys.executable, str(root / "tools" / tool), *args], root, f"{tool} under {label}", timeout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", nargs="*", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    child(args.child, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
