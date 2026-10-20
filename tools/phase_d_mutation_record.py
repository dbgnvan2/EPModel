"""Run Phase D criteria under declared mutants and write the committed Phase D mutation record.

Purpose: prove a passing Phase D criterion fails when the mechanism its spec row names is removed (`M11.1d`), as
         ``tools/mutation_record.py`` does for Phase C. Phase D's other mutation runs are reported in the plan, ad hoc.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.1a, #M11.1d, #M11.C.10, #M11.C.14
Tests:   tests/bowen/test_phase_d_mutation_record.py::test_m111d_phase_d_mutation_record_is_current

    python3 tools/phase_d_mutation_record.py [--workers N]

A mutant is a literal replacement applied to a temporary copy of the repository, never to the working tree, and must
match exactly once (`M11.1a`). Only criteria that pass in ``docs/phase_d2_ensemble_record.md`` (part II) or, for a
part I criterion a mutant names (`M11.C.14`, revision 31), in ``docs/phase_d_ensemble_record.md`` are run. Results are
cached in ``docs/records_cache/phase_d_mutation.json`` under the hash of the engine, the config and
``tools/phase_d_mutant_child.py`` and of ``apply`` and ``run_in_copy`` (which copy the repository and apply the edits),
with each mutant's edits; the record is rendered from the cache. Outcomes are judged as in the Phase C record
(``tools/mutation_record.py``'s ``judge``): an engine exception under a mutant (RAISED) is a red, reported and never
cached, as is a broken replacement (BROKEN, no evidence); a child that fails or times out, or a failure of the machine
running it, is a FAILED row (no evidence, never cached) and the build goes on (version 2 stage 1, S1.5a). A
passing part II criterion with no mutant here must be named in ``AD_HOC`` with where its mutation is reported;
part I's other mutations are reported in the plan, ad hoc, and are not listed here.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.mutant_runner import Mutant, apply, run_in_copy  # noqa: E402,F401
from tools.mutation_record import NAMED, judge  # noqa: E402
from tools.record_cache import ERRORED, Cache, child_rows, exit_status, result_key  # noqa: E402
from tools.record_json import record_json  # noqa: E402

RECORD = REPO / "docs" / "phase_d_mutation_record.md"
ENSEMBLE_RECORD = REPO / "docs" / "phase_d2_ensemble_record.md"
PART_1_RECORD = REPO / "docs" / "phase_d_ensemble_record.md"
CHILD = REPO / "tools" / "phase_d_mutant_child.py"
MUTANTS = (
    Mutant("witnessed-cutoff-learning-removed", NAMED, ("M11.C.10[family of origin]",), "config/bowen/constants.md",
           "| `witnessed_cutoff_gain` | 0.05 |", "| `witnessed_cutoff_gain` | 0.0 |",
           "M4.D.6f removed: a witness's learned CUTOFF value takes no step toward what followed a witnessed cutoff"),
    # M11.C.2 on the twelve-person family passes at revision 31 (it failed at revision 30); its spec row names the
    # witness path (M1.F.5). witness_weight is read only by witness appraisal (M4.C.9), so 0 removes the path.
    Mutant("witness-path-removed", NAMED, ("M11.C.2[twelve-person family]",), "config/bowen/constants.md",
           "| `witness_weight` | 0.5 |", "| `witness_weight` | 0.0 |",
           "M1.F.5 removed: a witnessed exchange moves the witness's anxiety by nothing"),
    # The second mutation the plan names for it (§4.8, revision 29): focus on a child's weakness with focus steering.
    Mutant("weakness-focus-and-steering-removed", NAMED, ("M11.C.2[twelve-person family]",), "config/bowen/constants.md",
           "| `perceived_weakness_gain` | 0.01 |", "| `perceived_weakness_gain` | 0.0 |",
           "M1.B.8a and M4.D.1g removed: no focus on a child's weakness, and focus steers no projection-type act",
           also=(("config/bowen/constants.md", "| `focus_steer_gain` | 1.0 |", "| `focus_steer_gain` | 0.0 |"),)),
    # Revision 31 (R31.B5): the gate removed MUST turn at least one technique variant red.
    Mutant("marital-distance-gate-removed", NAMED, ("M11.C.14[firmness]", "M11.C.14[talking it out]"),
           "src/bowen/engine/appraise.py",
           "    if not distant or not any(state.people[recipient].symptom_active.values()):\n        return False\n",
           "    return False\n",
           "M5.C.1 removed: a move addressed to a symptomatic child under marital distance improves them as any other"),
    # Version 1 (2026-10-17, V3 and the triangle change): mutants for passing part I and part II criteria whose
    # mutation was ad hoc or missing.
    Mutant("triangle-help-ignored", NAMED, ("M11.C.46",), "src/bowen/engine/moves.py",
           "    help_ = third_help(state, seeker.id, partner, outsider)\n",
           "    help_ = 1.0\n",
           "M1.C.1's dependence on the third's help removed (owner decision 2026-10-09): every third helps fully"),
    # M11.C.39's row names this mutation. Its baseline already has learning off, so the mutant makes the two arms the
    # same by construction: it confirms the arms differ only in learning, and adds nothing beyond that (review
    # 2026-10-17).
    Mutant("learning-off-in-both-arms", NAMED, ("M11.C.39",), "config/bowen/constants.md",
           "| `learning_rate` | 0.2 |", "| `learning_rate` | 0.0 |",
           "M4.D.6 disabled in both arms: no learned value moves, as in the criterion's baseline (LEARNING_OFF)",
           also=(("config/bowen/constants.md", "| `witnessed_cutoff_gain` | 0.05 |",
                  "| `witnessed_cutoff_gain` | 0.0 |"),)),
    Mutant("death-bound-anxiety-discarded", NAMED, ("M11.C.15",), "src/bowen/engine/life_course.py",
           "    moved = release_bound(state, pid)  # as when a symptom ends (M7.D.2a); nothing if no partner is left\n",
           "    moved = ()  # mutant: the symptom-bound anxiety is discarded at the death, not released\n"
           "    if person.symptom_bound:\n"
           "        people_rows.append((pid, \"symptom_bound\", -person.symptom_bound))\n"
           "        person.symptom_bound = 0.0\n",
           "M6.3c's release of the dead person's symptom-bound anxiety (M6.I.1) replaced by discarding it"),
    # (M11.C.15's arm calls die() directly, not through yearly(), so the death check there is not on its path.)
    # Revision 32's regeneration (with the nodal spike) turned M11.C.17, M11.C.22 and two M11.C.7 pairs green. M11.C.17's
    # row: the external agent's effect (M1.E.7's landed contact) removed MUST turn it red; the same edit is what the two
    # M11.C.7 pairs' deviation limb rests on (the coach's position only matters if landing does something).
    Mutant("landed-contact-changes-nothing", NAMED,
           ("M11.C.17", "M11.C.7[joint over one spouse]", "M11.C.7[joint over sided]"), "config/bowen/constants.md",
           "| `perspective_gain` | 0.1 |", "| `perspective_gain` | 0.0 |",
           "M1.E.7's landed contact removed: a coach's contact that lands raises no systems perspective"),
    # M11.C.22's row: M7.D.2a's stabilising term flattened MUST turn it red (the lock-in that holds the partners' excess).
    Mutant("lock-in-removed", NAMED, ("M11.C.22",), "config/bowen/constants.md",
           "| `lock_in_gain` | 0.2 |", "| `lock_in_gain` | 0.0 |",
           "M7.D.2a's stabilising term removed: an active symptom holds none of its partners' excess"),
    # M11.C.6's row (restated at revision 32): fixing each child's basic level at its anchor MUST turn it red.
    Mutant("development-fixed-at-anchor", NAMED, ("M11.C.6",), "config/bowen/constants.md",
           "| `development_rate` | 0.1 |", "| `development_rate` | 0.0 |",
           "M7.E.1f's development removed: each child's basic level stays at its anchor from 5 to 20"),
    # M11.C.7's row: M8.2/M8.3 made blind to position, so a sided coach counts as live and its contact can land.
    # Not run on M11.C.7[joint over one spouse]: neither arm has a SPLIT toward the coach, so it would change nothing.
    Mutant("coach-position-blind", NAMED, ("M11.C.7[one spouse over sided]", "M11.C.7[joint over sided]"),
           "src/bowen/engine/external.py",
           "        occupant = Occupant(coach, fused_into=coach_side(state, coach, members, params))\n",
           "        occupant = Occupant(coach, fused_into=None)\n",
           "M8.3 made position-blind: a coach drawn onto one spouse's side still counts as a live position"),
    # M11.C.11's return limb (composite): a removed member's symptom hold is frozen while they are away (no
    # interactive partner to release to) and comes back on return. Ad hoc 2026-10-17: lock_in +12.1 of the +2.3, and
    # this mutant reverses it (-1.77 +/- 1.61, 150 seeds).
    Mutant("removed-member-hold-released-while-away", NAMED, ("M11.C.11",), "src/bowen/engine/symptoms.py",
           "    partners = _partners(state, pid)\n    amount = share * person.symptom_bound\n",
           "    partners = tuple(sorted(o for t in state.ties_of(pid) for o in t.id.members()\n"
           "                            if o != pid and state.people[o].alive))\n"
           "    amount = share * person.symptom_bound\n",
           "M7.D.2a's hold released over every tie, worry edges and non-interactive ties included, at every release "
           "(a symptom's end, a death, past the turn, a binder lost): so a removed member's symptom gives back what it "
           "held while they are away, not on return"),
)
# Passing part II criteria with no mutant in this record, and where their mutation is reported.
AD_HOC = {
    "M11.C.47": "a premise; run ad hoc, `docs/implementation_plan_phase_d.md` §4.10 (Results, Mutations)",
}


def engine_hash() -> str:
    """The engine, config and rules (``code_hash``), the child, and the source of ``apply`` and ``run_in_copy``, which
    decide what is copied and how an edit lands. Not the rest of ``tools/mutant_runner.py``: that is Phase C's child.
    The two functions are read through ``source_text``, as ``code_hash`` reads a file, so a comment or a docstring
    there reruns nothing (version 2 stage 1, rider (iv)).
    Tests: tests/bowen/test_phase_d_mutation_record.py::test_s1_9_engine_hash_ignores_comments
    """
    from tools.ensemble_record import code_hash, source_text

    digest = hashlib.sha256(code_hash(CHILD).encode())
    for function in (apply, run_in_copy):
        digest.update(source_text(inspect.getsource(function)).encode())
    return digest.hexdigest()


def passing() -> set[str]:
    """The ``CRITERIA_D2`` keys that pass in the part II record. Each row is printed under its key (version 2 stage 1,
    S1.7b), so the rows must be the keys, in order."""
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2

    text = ENSEMBLE_RECORD.read_text(encoding="utf-8")
    rows = record_json(text)
    if [r["criterion"] for r in rows] != list(CRITERIA_D2):
        raise SystemExit(f"{ENSEMBLE_RECORD.name}'s rows no longer match CRITERIA_D2 in order; rerun its tool")
    return {r["criterion"] for r in rows if r["outcome"] == "PASS"}


def passing_part_1() -> set[str]:
    """The ``CRITERIA_D`` keys a mutant here names that pass in the part I record. Their ids are their keys."""
    text = PART_1_RECORD.read_text(encoding="utf-8")
    rows = record_json(text)
    named = {c for m in MUTANTS for c in m.criteria} - set(_part_2_keys())  # part I criteria a mutant names
    absent = sorted(named - {r["criterion"] for r in rows})
    if absent:  # absent is not failing (correctness review 2026-10-16): the part I record is stale
        raise SystemExit(f"{PART_1_RECORD.name} has no row for {absent}; rerun python3 tools/phase_d_record.py")
    return {r["criterion"] for r in rows if r["outcome"] == "PASS" and r["criterion"] in named}


def _part_2_keys():
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2

    return CRITERIA_D2


def build(cache: Cache, workers: int | None = None) -> str:
    """The record, from the cache. With ``workers``, missing results are run first; without, a missing one raises
    KeyError, so a test can check the committed record without running anything."""
    part_2, engine = passing(), engine_hash()
    passed = part_2 | passing_part_1()
    rows, skipped = [], []
    for mutant in MUTANTS:
        run_ids = [c for c in mutant.criteria if c in passed]
        skipped += [(mutant, c) for c in mutant.criteria if c not in passed]
        keys = {c: result_key(engine, mutant.definition(), c) for c in run_ids}
        missing = [c for c in run_ids if cache.get(keys[c]) is None]
        if missing and workers is None:
            raise KeyError(f"{mutant.id}: no cached result for the current engine and edits")
        fresh = {}
        if missing:
            for result in child_rows(mutant, CHILD.name, missing, workers, mutant.id):
                fresh[result["criterion"]] = result
                if result["outcome"] not in ERRORED:  # an errored row is reported but rerun next time, never cached
                    cache.put(keys[result["criterion"]], result)
            cache.flush()
        for c in run_ids:
            result = cache.get(keys[c]) or fresh.get(c)
            if result is None:
                raise SystemExit(f"{mutant.id} → {c}: the child printed no result")
            rows.append((mutant, result))
    recorded = {r["criterion"] for _, r in rows}
    unproved = sorted(part_2 - recorded - set(AD_HOC))
    if unproved:
        raise SystemExit(f"passing in part II with no mutant here and none named in AD_HOC: {unproved}")
    return render(rows, skipped, engine)


def render(rows, skipped, engine: str) -> str:
    lines = [
        "# Phase D mutation record",
        "",
        "Generated by `tools/phase_d_mutation_record.py`; do not edit. Each mutant is a literal replacement applied to a",
        "temporary copy of the repository and run against a criterion that passes in `docs/phase_d2_ensemble_record.md`",
        "(part II) or `docs/phase_d_ensemble_record.md` (part I), with the same adaptive rules. A mutant is **red** when the criterion stops passing and **survived** when it still",
        "passes; a red mutant is **reversed** when every gating readout's interval lies wholly on the side opposite its",
        "direction (`M11.1d`). Phase D's other mutation runs are ad hoc and reported in",
        "`docs/implementation_plan_phase_d.md`.",
        "",
        f"engine_hash: {engine}",
        "",
        "| Mutant | Kind | What it changes | Criterion | Verdict under mutant | Seeds | Difference under mutant | Result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mutant, result in rows:
        note = f" ({result['error']})" if result.get("error") else ""
        diffs = "; ".join(f"`{x['readout']}` {x['mean_difference']:+.3g} ± {x['half_width']:.2g}"
                          + (f", p {x['p']:.2g}" if "p" in x else "") + (" (reported)" if x.get("report_only") else "")
                          for x in result.get("readouts", ()))
        lines.append(f"| `{mutant.id}` | {mutant.kind} | {mutant.what} | `{result['criterion']}` | {result['outcome']}{note} "
                     f"| {result['seeds']} | {diffs or '—'} | "
                     f"**{judge(mutant, result['outcome'], result.get('readouts', ()))}** |")
    lines += ["", "## Not run", "", "Criteria a mutant targets that do not pass in their ensemble record.", ""]
    lines += [f"- `{m.id}` → `{cid}`" for m, cid in skipped] or ["- none"]
    lines += ["", "## Passing with no mutant in this record", "",
              "Part II criteria that pass but are not proved here; where their mutation is reported instead.", ""]
    lines += [f"- `{cid}`: {where}" for cid, where in sorted(AD_HOC.items())] or ["- none"]
    lines += ["", "## Machine-readable", "", "```json",
              json.dumps([{"mutant": m.id, "kind": m.kind, "criterion": r["criterion"], "outcome": r["outcome"],
                           "seeds": r["seeds"], "result": judge(m, r["outcome"], r.get("readouts", ())),
                           "readouts": r.get("readouts", []), **({"error": r["error"]} if r.get("error") else {})}
                          for m, r in rows], indent=1, default=float),
              "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    cache = Cache("phase_d_mutation")
    text = build(cache, args.workers)
    cache.save()
    RECORD.write_text(text, encoding="utf-8")
    print(f"wrote {RECORD.relative_to(REPO)}")
    return exit_status()


if __name__ == "__main__":
    raise SystemExit(main())
