"""Run Phase D part II criteria in a mutated copy of the repository: the code that produces the Phase D mutation record's
results.

Purpose: the child ``tools/phase_d_mutation_record.py`` runs in each mutated copy, as ``tools/mutant_runner.py`` is for
         Phase C. It is a separate file so the Phase C records' engine hash, which covers ``mutant_runner.py``, does
         not move. Its content is in this record's hash, so only code that changes what a run computes belongs here.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.1a, #M11.1d, #M11.C.10
Tests:   tests/bowen/test_phase_d_mutation_record.py::test_m111d_phase_d_mutation_record_is_current
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def child(keys: list[str], workers: int) -> None:
    """Print one JSON line per criterion, keyed by its ``CRITERIA_D2`` key (two criteria there share an id)."""
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2
    from src.bowen.ensemble.runner import run_criterion
    from src.bowen.io.load import load_constants
    from tools.ensemble_record import RULE_KEYS
    from tools.mutant_runner import BROKEN_ERRORS

    constants = load_constants()
    rules = {k: constants[k] for k in RULE_KEYS}
    for key in keys:
        try:
            v = run_criterion(CRITERIA_D2[key], rules, workers=workers)
            readouts = [{"readout": r["readout"], "direction": r["direction"], "mean_difference": r["mean_difference"],
                         "half_width": r["half_width"], "report_only": r.get("report_only", False),
                         **({"p": r["p"]} if "p" in r else {})} for r in v.readouts]
            print(json.dumps({"criterion": key, "outcome": v.outcome, "seeds": v.seeds, "readouts": readouts},
                             default=float), flush=True)
        except BROKEN_ERRORS as error:  # the replacement text itself is broken: no evidence
            print(json.dumps({"criterion": key, "outcome": "BROKEN", "seeds": 0,
                              "error": f"{type(error).__name__}: {str(error)[:160]}"}), flush=True)
        except Exception as error:  # the engine raised on the mutated model: a red, with cause
            print(json.dumps({"criterion": key, "outcome": "RAISED", "seeds": 0,
                              "error": f"{type(error).__name__}: {str(error)[:160]}"}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", nargs="*", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    child(args.child, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
