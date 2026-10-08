"""Run the Phase C criteria under declared source mutants and write the committed mutation record.

Purpose: prove each passing criterion fails when the rule behind it is removed or inverted
         (`M11.1d`, plan §3's mutation column), and holds when the code is re-encoded
         without changing what it computes (`M11.1c`).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.1a, #M11.1c, #M11.1d, #M11.5, #M10.C.5
Tests:   tests/bowen/test_ensemble_record.py::test_m111d_every_passing_criterion_has_a_mutant_run

    python3 tools/mutation_record.py [--workers N] [--only MUTANT ...]

Each mutant is a literal source replacement applied to a temporary copy of the repository —
never to the working tree — and must match exactly once (`M11.1a`: a mutation that does not
apply proves nothing). Only criteria that pass in `docs/phase_c_ensemble_record.md` are run;
a mutant whose criteria all fail is listed as not run. The record is generated, never edited.

Outcomes:

* deletion, named or sign-inverted mutant — **red** if the criterion no longer passes
  (FAIL or UNDETERMINED), **survived** if it still passes. A survivor means the criterion is
  not proved by that mutant and is reported as such.
* representation mutant — **unchanged** if the criterion still passes, otherwise an
  **encoding artefact** (`M11.1c`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RECORD = REPO / "docs" / "phase_c_mutation_record.md"
ENSEMBLE_RECORD = REPO / "docs" / "phase_c_ensemble_record.md"
# The mutant list here, and RULE_KEYS in tools/ensemble_record.py, are inputs to the record.
HASHED_TOOLS = (Path(__file__).resolve(), REPO / "tools" / "ensemble_record.py")
DELETION, NAMED, SIGN, REPRESENTATION = "deletion", "named", "sign-inverted", "representation"
LEVEL = ("M11.C.1", "M11.C.38", "M11.C.41")


@dataclass(frozen=True)
class Mutant:
    id: str
    kind: str
    criteria: tuple[str, ...]  # criterion ids, or a prefix ending in "[" for expanded families
    file: str
    old: str
    new: str
    what: str
    also: tuple[tuple[str, str, str], ...] = ()  # further (file, old, new) for a joint mutant (M11.1a: name every write)


LEVEL_BLIND = (  # every remaining read of level on the path to onset, made level-independent at level 50
    ("src/bowen/engine/contact.py", "return params.contact_band_max * person.functional_level / SCALE_MAX",
     "return params.contact_band_max * 50.0 / SCALE_MAX"),
    ("src/bowen/engine/symptoms.py", "return params.symptom_threshold_gain * person.functional_level",
     "return params.symptom_threshold_gain * 50.0"),
    ("src/bowen/policy/policy.py",
     "return min(1.0, max(0.0, functional_level / SCALE_MAX)) ** params.self_channel_exponent",
     "return 0.5 ** params.self_channel_exponent"),
    ("src/bowen/policy/policy.py",
     "return min(1.0, max(0.0, obs.functional_level / (layer * params.capacity_level_per_layer)))",
     "return min(1.0, max(0.0, 50.0 / (layer * params.capacity_level_per_layer)))"),
    ("src/bowen/engine/standing_load.py",
     "return params.standing_load_gain * (SCALE_MAX - person.basic_level) / SCALE_MAX",
     "return params.standing_load_gain * 0.5"),
    ("src/bowen/engine/outside_ness.py",
     "start = params.initial_impingement_scale * (1.0 - person.basic_level / SCALE_MAX)",
     "start = params.initial_impingement_scale * 0.5"),
    ("src/bowen/engine/moves.py",
     "return max(0.0, 1.0 - sum(p.functional_level for p in members) / (len(members) * SCALE_MAX))",
     "return 0.5"),
    ("src/bowen/engine/iposition.py",
     "return params.hold_gain * person.functional_level * efficacy(person)",
     "return params.hold_gain * 50.0 * efficacy(person)"),
)


MUTANTS = (
    # --- plan §3's named mutations --------------------------------------------------------------
    Mutant("steepness-level-independent", NAMED, LEVEL, "src/bowen/engine/contact.py",
           "return SCALE_MAX / max(person.functional_level, params.functional_level_floor)",
           "return SCALE_MAX / max(SCALE_MAX / 2, params.functional_level_floor)",
           "M4.C.1a's steepness made independent of functional_level (fixed at its value for level 50)"),
    Mutant("mixing-weight-level-independent", NAMED, ("M11.C.41",), "src/bowen/policy/policy.py",
           "return min(1.0, max(0.0, functional_level / SCALE_MAX)) ** params.self_channel_exponent",
           "return 0.5 ** params.self_channel_exponent",
           "M4.D.1a's mixing weight made independent of functional_level"),
    Mutant("triangle-transfer-removed", NAMED, ("M11.C.3",), "src/bowen/engine/moves.py",
           "    if total <= 0:\n        return None\n    for pid, amount in moved:",
           "    if True:\n        return None\n    for pid, amount in moved:",
           "M1.C.1's transfer removed"),
    Mutant("learner-disabled", NAMED, ("M11.C.16", "M11.C.42"), "src/bowen/engine/learner.py",
           "delta = params.learning_rate * (signal - value)", "delta = 0.0 * (signal - value)",
           "M4.D.6 disabled in both arms: no learned value ever moves"),
    Mutant("axes-collapsed", NAMED, ("M11.C.19",), "src/bowen/readouts/counterfeit.py",
           "    outward, inward = axes(person)\n    failed = set()",
           "    outward = inward = sum(axes(person)) / 2\n    failed = set()",
           "M5.F.2b's two axes collapsed to their mean"),
    Mutant("sex-term-in-pole", NAMED, ("M11.C.25",), "src/bowen/engine/moves.py",
           "    push = params.balance_push_gain * _strength(event, params)\n",
           "    push = params.balance_push_gain * _strength(event, params)\n"
           "    push *= 2.0 if state.people[over].sex.value == \"female\" else 0.5\n",
           "a sex term added to pole assignment: a female over-functioner pushes four times as hard"),
    Mutant("one-sided-deviation", NAMED, ("M11.C.27[", "M11.C.44"), "src/bowen/engine/contact.py",
           "    return little + much\n", "    return little\n",
           "M4.C.1 made one-sided: only too little contact is felt"),
    Mutant("anger-gate-inverted", NAMED, ("M11.C.32",), "src/bowen/engine/iposition.py",
           "return too_much(state.people[mover], tie, params) > params.anger_threshold",
           "return too_much(state.people[mover], tie, params) <= params.anger_threshold",
           "M5.D.4's anger gate inverted"),
    Mutant("witness-position-blind", NAMED, ("M11.C.35",), "src/bowen/engine/appraise.py",
           "    reach = sum(ends) / len(ends)  # the mean of its tie to each party\n",
           "    reach = 1.0\n",
           "witness appraisal made a copy that reads neither of the witness's ties"),
    Mutant("relief-tension-independent", NAMED, ("M11.C.45",), "src/bowen/engine/moves.py",
           "moved = [(p.id, min(excess(p), rate * excess(p))) for p in insiders]",
           "moved = [(p.id, min(excess(p), rate * SCALE_MAX / 10)) for p in insiders]",
           "M1.C.1's relief made independent of the pair's tension (a fixed amount, capped by excess)"),
    # --- added after steepness-level-independent survived on M11.C.1/.38/.41: the other rule that
    # --- reads level on the path to onset (M1.A.6's threshold)
    Mutant("threshold-level-independent", DELETION, LEVEL, "src/bowen/engine/symptoms.py",
           "return params.symptom_threshold_gain * person.functional_level",
           "return params.symptom_threshold_gain * 50.0",
           "M1.A.6's symptom threshold made independent of functional_level (fixed at its value for level 50)"),
    Mutant("threshold-level-inverted", SIGN, LEVEL, "src/bowen/engine/symptoms.py",
           "return params.symptom_threshold_gain * person.functional_level",
           "return params.symptom_threshold_gain * (100.0 - person.functional_level)",
           "M1.A.6's symptom threshold falls as functional_level rises"),
    # --- and M4.A.5's self term, the standing load a lower basic level carries every tick
    Mutant("standing-load-level-independent", DELETION, LEVEL, "src/bowen/engine/standing_load.py",
           "return params.standing_load_gain * (SCALE_MAX - person.basic_level) / SCALE_MAX",
           "return params.standing_load_gain * 0.5",
           "M4.A.5's self term made independent of basic_level (fixed at its value for level 50)"),
    Mutant("standing-load-level-inverted", SIGN, LEVEL, "src/bowen/engine/standing_load.py",
           "return params.standing_load_gain * (SCALE_MAX - person.basic_level) / SCALE_MAX",
           "return params.standing_load_gain * person.basic_level / SCALE_MAX",
           "M4.A.5's self term rises with basic_level instead of falling"),
    Mutant("level-blind", DELETION, LEVEL, "src/bowen/engine/contact.py",
           "return SCALE_MAX / max(person.functional_level, params.functional_level_floor)",
           "return SCALE_MAX / max(SCALE_MAX / 2, params.functional_level_floor)",
           "every rule that reads level made level-independent at once: steepness, band, threshold, mixing "
           "weight, layer availability, M4.A.5's self term, initial outside-ness, triangle routing capacity "
           "and I-POSITION hold capacity",
           also=LEVEL_BLIND),
    # --- M11.1d: sign-inverted mutants of each passing criterion's core rule --------------------
    Mutant("steepness-inverted", SIGN, LEVEL, "src/bowen/engine/contact.py",
           "return SCALE_MAX / max(person.functional_level, params.functional_level_floor)",
           "return max(person.functional_level, params.functional_level_floor) / (SCALE_MAX / 4)",
           "M4.C.1a's steepness rises with functional_level instead of falling"),
    Mutant("mixing-weight-inverted", SIGN, ("M11.C.41",), "src/bowen/policy/policy.py",
           "return min(1.0, max(0.0, functional_level / SCALE_MAX)) ** params.self_channel_exponent",
           "return min(1.0, max(0.0, 1.0 - functional_level / SCALE_MAX)) ** params.self_channel_exponent",
           "M4.D.1a's mixing weight falls with functional_level instead of rising"),
    Mutant("learner-inverted", SIGN, ("M11.C.16", "M11.C.42"), "src/bowen/engine/learner.py",
           "delta = params.learning_rate * (signal - value)", "delta = params.learning_rate * (-signal - value)",
           "M4.D.6 inverted: relief lowers an act's learned value, distress raises it"),
    Mutant("axes-swapped", SIGN, ("M11.C.19",), "src/bowen/readouts/counterfeit.py",
           "    outward, inward = axes(person)\n    failed = set()",
           "    inward, outward = axes(person)\n    failed = set()",
           "M5.F.2b's axes read the wrong way round"),
    Mutant("deviation-inverted", SIGN, ("M11.C.27[", "M11.C.44"), "src/bowen/engine/contact.py",
           "    return little + much\n", "    return -(little + much)\n",
           "M4.C.1's deviation inverted: moving away from the optimum relieves"),
    Mutant("anger-gate-removed", DELETION, ("M11.C.32",), "src/bowen/engine/iposition.py",
           "return too_much(state.people[mover], tie, params) > params.anger_threshold",
           "return False",
           "M5.D.4's anger gate removed: the mover is never angry"),
    Mutant("witness-reach-inverted", SIGN, ("M11.C.35",), "src/bowen/engine/appraise.py",
           "    reach = sum(ends) / len(ends)  # the mean of its tie to each party\n",
           "    reach = 1.0 - sum(ends) / len(ends)\n",
           "the witness feels more through weaker ties"),
    Mutant("relief-tension-inverted", SIGN, ("M11.C.45",), "src/bowen/engine/moves.py",
           "moved = [(p.id, min(excess(p), rate * excess(p))) for p in insiders]",
           "moved = [(p.id, min(excess(p), rate * max(0.0, SCALE_MAX / 10 - excess(p)))) for p in insiders]",
           "M1.C.1's relief falls as the pair's tension rises (still conserved, so the ledger holds)"),
    # --- M11.1c: representation mutants, numerically equivalent by construction ------------------
    Mutant("appraisal-sum-order-reversed", REPRESENTATION, ("*",), "src/bowen/engine/appraise.py",
           "    items.sort(key=lambda x: x[0])\n", "    items.sort(key=lambda x: x[0], reverse=True)\n",
           "same-tick appraisal summed in reverse order (M1.F.8: order-free up to float rounding)"),
    Mutant("clamp-within-tolerance", REPRESENTATION, ("*",), "src/bowen/engine/contact.py",
           "    return min(1.0, max(0.0, value))\n", "    return min(1.0 - 1e-12, max(1e-12, value))\n",
           "felt contact and impingement clamped 1e-12 inside [0, 1]"),
)


def passing() -> set[str]:
    text = ENSEMBLE_RECORD.read_text(encoding="utf-8")
    data = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
    return {row["criterion"] for row in data if row["outcome"] == "PASS"}


def targets(mutant: Mutant, all_ids, passed: set[str]) -> tuple[list[str], list[str]]:
    def matches(cid):
        return any(c == "*" or cid == c or (c.endswith("[") and cid.startswith(c)) or cid.startswith(c + "[")
                   for c in mutant.criteria)

    chosen = [cid for cid in all_ids if matches(cid)]
    return [c for c in chosen if c in passed], [c for c in chosen if c not in passed]


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
                         "half_width": r["half_width"]} for r in v.readouts]
            print(json.dumps({"criterion": cid, "outcome": v.outcome, "seeds": v.seeds, "readouts": readouts},
                             default=float), flush=True)
        except Exception as error:  # an invariant raising under a mutant is a red, reported with its cause
            print(json.dumps({"criterion": cid, "outcome": "RAISED", "seeds": 0,
                              "error": f"{type(error).__name__}: {str(error)[:160]}"}), flush=True)


def run_mutant(mutant: Mutant, ids: list[str], workers: int) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="bowen-mutant-") as tmp:
        root = Path(tmp)
        for part in ("src", "config", "tools"):
            shutil.copytree(REPO / part, root / part, ignore=shutil.ignore_patterns("__pycache__"))
        apply(root, mutant)
        out = subprocess.run([sys.executable, str(root / "tools" / "mutation_record.py"), "--child", *ids,
                              "--workers", str(workers)], cwd=root, capture_output=True, text=True, timeout=7200)
        if out.returncode != 0:
            raise SystemExit(f"mutant {mutant.id} child failed:\n{out.stderr[-2000:]}")
        return [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]


def judge(mutant: Mutant, outcome: str) -> str:
    if mutant.kind == REPRESENTATION:
        return "unchanged" if outcome == "PASS" else "encoding artefact"
    return "survived" if outcome == "PASS" else "red"


def render(rows, skipped, hash_: str) -> str:
    lines = [
        "# Phase C mutation record",
        "",
        "Generated by `tools/mutation_record.py`; do not edit. Each mutant is a literal source replacement applied",
        "to a temporary copy of the repository, run against the criteria that pass in",
        "`docs/phase_c_ensemble_record.md`, with the same adaptive rules. A deletion, named or sign-inverted mutant",
        "is **red** when the criterion stops passing and **survived** when it still passes; a survivor means that",
        "mutant does not prove the criterion. A representation mutant (`M11.1c`) should leave every verdict",
        "unchanged; a change is reported as an **encoding artefact**.",
        "",
        f"code_hash: {hash_}",
        "",
        "| Mutant | Kind | What it changes | Criterion | Verdict under mutant | Seeds | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for mutant, result in rows:
        note = f" ({result['error']})" if result.get("error") else ""
        lines.append(f"| `{mutant.id}` | {mutant.kind} | {mutant.what} | `{result['criterion']}` | "
                     f"{result['outcome']}{note} | {result['seeds']} | **{judge(mutant, result['outcome'])}** |")
    lines += ["", "## Not run", "",
              "Criteria a mutant targets that do not pass at the central setting — a mutant cannot prove a failing",
              "criterion.", ""]
    lines += [f"- `{m.id}` → `{cid}`" for m, cid in skipped] or ["- none"]
    lines += ["", "## Machine-readable", "", "```json",
              json.dumps([{"mutant": m.id, "kind": m.kind, "criterion": r["criterion"], "outcome": r["outcome"],
                           "result": judge(m, r["outcome"])} for m, r in rows], indent=1),
              "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--child", nargs="*")
    args = parser.parse_args()
    if args.child is not None:
        child(args.child, args.workers)
        return 0
    from src.bowen.ensemble.criteria import CRITERIA
    from tools.ensemble_record import code_hash

    passed = passing()
    rows, skipped = [], []
    for mutant in MUTANTS:
        if args.only and mutant.id not in args.only:
            continue
        run_ids, not_run = targets(mutant, list(CRITERIA), passed)
        skipped += [(mutant, cid) for cid in not_run if mutant.criteria != ("*",)]
        if not run_ids:
            continue
        for result in run_mutant(mutant, run_ids, args.workers):
            rows.append((mutant, result))
            print(f"{mutant.id} → {result['criterion']}: {result['outcome']} ({judge(mutant, result['outcome'])})",
                  flush=True)
    RECORD.write_text(render(rows, skipped, code_hash(*HASHED_TOOLS)), encoding="utf-8")
    print(f"wrote {RECORD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
