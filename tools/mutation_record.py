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
  not proved by that mutant and is reported as such. A red mutant is **reversed** when every
  gating readout's interval lies wholly on the side opposite its direction: the result flipped
  rather than vanished (`M11.1d`). Report-only and equivalence readouts are not gating. A mutant
  whose replacement text is broken (NameError, ImportError, SyntaxError) is **broken**: it proves
  nothing and is not counted as red. An engine exception under a mutant is a red.
* representation mutant — **unchanged** if the criterion still passes, otherwise an
  **encoding artefact** (`M11.1c`).
"""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.mutant_runner import BROKEN_ERRORS, Mutant, apply  # noqa: E402,F401
from tools.record_cache import Cache, engine_hash, result_key, run_cached  # noqa: E402

RECORD = REPO / "docs" / "phase_c_mutation_record.md"
ENSEMBLE_RECORD = REPO / "docs" / "phase_c_ensemble_record.md"
# The mutant list here, and RULE_KEYS in tools/ensemble_record.py, are inputs to the record.
DELETION, NAMED, SIGN, REPRESENTATION = "deletion", "named", "sign-inverted", "representation"
LEVEL = ("M11.C.1", "M11.C.38", "M11.C.41", "M11.C.16")



AVAILABILITY = "return min(1.0, max(0.0, obs.functional_level / (layer * params.capacity_level_per_layer)))"
# Every layer fully available at every level: the rule at level 50, which clamps to 1 on every layer while
# capacity_level_per_layer x the highest layer <= 50 (tested by test_m111a_availability_deletion_is_full_on_every_layer).
AVAILABILITY_DELETED = "return min(1.0, max(0.0, (SCALE_MAX / 2) / (layer * params.capacity_level_per_layer)))"
INVERSION_PIVOT = 0.6  # [I] the inversion is (INVERSION_PIVOT x SCALE_MAX - level) / (layer x capacity): reflected about 0.3

LEVEL_BLIND = (  # every remaining read of level on the path to onset, made level-independent at level 50
    ("src/bowen/engine/contact.py", "return params.contact_band_max * person.functional_level / SCALE_MAX",
     "return params.contact_band_max * (SCALE_MAX / 2) / SCALE_MAX"),
    ("src/bowen/engine/symptoms.py", "return params.symptom_threshold_gain * person.functional_level",
     "return params.symptom_threshold_gain * 50.0"),  # SCALE_MAX / 2; symptoms.py does not import SCALE_MAX
    ("src/bowen/policy/policy.py",
     "return min(1.0, max(0.0, functional_level / SCALE_MAX)) ** params.self_channel_exponent",
     "return 0.5 ** params.self_channel_exponent"),
    ("src/bowen/policy/policy.py",
     "return min(1.0, max(0.0, obs.functional_level / (layer * params.capacity_level_per_layer)))",
     AVAILABILITY_DELETED),
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
     "return params.hold_gain * (SCALE_MAX / 2) * efficacy(person)"),
)


# Reflected about level 30, between C.16's two arms' starting levels (about 24 and 39), so the inversion falls with
# level there instead of saturating at 1 in both arms. It equals the deletion wherever it clamps to 1, and makes a
# layer unavailable above the pivot; docs/phase_c_level_occupancy.md (tools/level_occupancy.py) gives how much of each
# C.16 arm lies in those ranges. Tested by test_m111a_availability_inversion_inverts_at_c16s_levels.
AVAILABILITY_INVERTED = (f"return min(1.0, max(0.0, (SCALE_MAX * {INVERSION_PIVOT} - obs.functional_level) "
                         "/ (layer * params.capacity_level_per_layer)))")
BAND = "return params.contact_band_max * person.functional_level / SCALE_MAX"
OUTSIDE_NESS = "start = params.initial_impingement_scale * (1.0 - person.basic_level / SCALE_MAX)"
ROUTING = "return max(0.0, 1.0 - sum(p.functional_level for p in members) / (len(members) * SCALE_MAX))"
HOLD = "return params.hold_gain * person.functional_level * efficacy(person)"

MUTANTS = (
    # --- plan §3's named mutations --------------------------------------------------------------
    Mutant("steepness-level-independent", NAMED, LEVEL, "src/bowen/engine/contact.py",
           "return SCALE_MAX / max(person.functional_level, params.functional_level_floor)",
           "return SCALE_MAX / max(SCALE_MAX / 2, params.functional_level_floor)",
           "M4.C.1a's steepness made independent of functional_level (fixed at its value for level 50)"),
    Mutant("mixing-weight-level-independent", NAMED, ("M11.C.41", "M11.C.16"), "src/bowen/policy/policy.py",
           "return min(1.0, max(0.0, functional_level / SCALE_MAX)) ** params.self_channel_exponent",
           "return 0.5 ** params.self_channel_exponent",
           "M4.D.1a's mixing weight made independent of functional_level"),
    Mutant("triangle-transfer-removed", NAMED, ("M11.C.3",), "src/bowen/engine/moves.py",
           "    if moved <= 0:\n        return None\n    seeker.acute_anxiety -= moved",
           "    if True:\n        return None\n    seeker.acute_anxiety -= moved",
           "M1.C.1's transfer removed"),
    Mutant("triangle-help-ignored", NAMED, ("M11.C.3", "M11.C.42", "M11.C.45"), "src/bowen/engine/moves.py",
           "    help_ = third_help(state, seeker.id, partner, outsider)\n",
           "    help_ = 1.0\n",
           "M1.C.1's dependence on the third's help removed (owner decision 2026-10-09): every third helps fully"),
    Mutant("triangle-credit-cross-person", NAMED, ("M11.C.42", "M11.C.45"), "src/bowen/engine/learner.py",
           "        recruited = set(event.targets) if event.kind == TRIANGLE_KIND else set()\n",
           "        recruited = set()\n",
           "M4.D.6e's exception removed (owner decision 2026-10-09): a TRIANGLE's recruited third back in its credit"),
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
           "moved = min(excess(seeker), rate * excess(seeker))",
           "moved = min(excess(seeker), rate * SCALE_MAX / 10)",
           "M1.C.1's relief made independent of the seeker's tension (a fixed amount, capped by excess)"),
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
    # --- M11.C.16 (2026-10-08): its named mutant, the learner disabled, survives, so which rule carries it?
    # --- Each level-reading rule is run on it alone, deleted and inverted (M11.1d). The steepness, threshold and
    # --- standing-load mutants above reach C.16 through LEVEL. An inversion must change behaviour between C.16's two
    # --- arms' levels, or it is a deletion in disguise (M11.1a; csdp sweep finding). See AVAILABILITY_INVERTED.
    Mutant("availability-level-independent", DELETION, ("M11.C.16",), "src/bowen/policy/policy.py",
           AVAILABILITY, AVAILABILITY_DELETED,
           "M4.D.3a's layer availability removed: every layer fully available at every level"),
    Mutant("availability-level-inverted", SIGN, ("M11.C.16",), "src/bowen/policy/policy.py",
           AVAILABILITY, AVAILABILITY_INVERTED,
           "M4.D.3a's layer availability reflected about level 30, so it falls as level rises over C.16's levels"),
    Mutant("availability-and-learner-removed", DELETION, ("M11.C.16",), "src/bowen/policy/policy.py",
           AVAILABILITY, AVAILABILITY_DELETED,
           "M4.D.3a's availability removed and M4.D.6 disabled, together",
           also=(("src/bowen/engine/learner.py", "delta = params.learning_rate * (signal - value)",
                  "delta = 0.0 * (signal - value)"),)),
    Mutant("band-level-independent", DELETION, ("M11.C.16",), "src/bowen/engine/contact.py",
           BAND, "return params.contact_band_max * (SCALE_MAX / 2) / SCALE_MAX",
           "M4.C.1a's band made independent of functional_level (fixed at level 50)"),
    Mutant("band-level-inverted", SIGN, ("M11.C.16",), "src/bowen/engine/contact.py",
           BAND, "return params.contact_band_max * (SCALE_MAX - person.functional_level) / SCALE_MAX",
           "M4.C.1a's band narrows as functional_level rises"),
    Mutant("outside-ness-level-independent", DELETION, ("M11.C.16",), "src/bowen/engine/outside_ness.py",
           OUTSIDE_NESS, "start = params.initial_impingement_scale * 0.5",
           "M1.A.9's initial outside-ness made independent of basic_level (fixed at level 50)"),
    Mutant("outside-ness-level-inverted", SIGN, ("M11.C.16",), "src/bowen/engine/outside_ness.py",
           OUTSIDE_NESS, "start = params.initial_impingement_scale * (person.basic_level / SCALE_MAX)",
           "M1.A.9's initial outside-ness rises with basic_level"),
    Mutant("routing-level-independent", DELETION, ("M11.C.16",), "src/bowen/engine/moves.py",
           ROUTING, "return 0.5", "M1.C.3a's routing capacity made independent of functional_level"),
    Mutant("routing-level-inverted", SIGN, ("M11.C.16",), "src/bowen/engine/moves.py",
           ROUTING, "return max(0.0, sum(p.functional_level for p in members) / (len(members) * SCALE_MAX))",
           "M1.C.3a's routing capacity rises with the members' functional_level"),
    Mutant("hold-level-independent", DELETION, ("M11.C.16",), "src/bowen/engine/iposition.py",
           HOLD, "return params.hold_gain * (SCALE_MAX / 2) * efficacy(person)",
           "M5.D.3's hold capacity made independent of functional_level (fixed at level 50)"),
    Mutant("hold-level-inverted", SIGN, ("M11.C.16",), "src/bowen/engine/iposition.py",
           HOLD, "return params.hold_gain * (SCALE_MAX - person.functional_level) * efficacy(person)",
           "M5.D.3's hold capacity falls as functional_level rises"),
    # --- M11.C.16's grounds (csdp sweeps, 2026-10-08). Its criterion row cites M4.C.1a (KS03.2: steepness and band)
    # --- and M1.C.3a (routing). M11.5's redundancy clause proves a premise by removing the whole set, so the cited
    # --- grounds are removed together, alone and with M4.D.3a's availability, the one rule that names the repertoire.
    Mutant("c16-cited-grounds-removed", DELETION, ("M11.C.16",), "src/bowen/engine/contact.py",
           "return SCALE_MAX / max(person.functional_level, params.functional_level_floor)",
           "return SCALE_MAX / max(SCALE_MAX / 2, params.functional_level_floor)",
           "M4.C.1a's steepness and band and M1.C.3a's routing capacity, the rules C.16's criterion row cites, "
           "made level-independent together; availability kept",
           also=(("src/bowen/engine/contact.py", BAND, "return params.contact_band_max * (SCALE_MAX / 2) / SCALE_MAX"),
                 ("src/bowen/engine/moves.py", ROUTING, "return 0.5"))),
    Mutant("c16-grounds-removed", DELETION, ("M11.C.16",), "src/bowen/policy/policy.py",
           AVAILABILITY, AVAILABILITY_DELETED,
           "M4.C.1a's steepness and band and M1.C.3a's routing capacity, which C.16's criterion row cites, made "
           "level-independent together with M4.D.3a's availability",
           also=(("src/bowen/engine/contact.py",
                  "return SCALE_MAX / max(person.functional_level, params.functional_level_floor)",
                  "return SCALE_MAX / max(SCALE_MAX / 2, params.functional_level_floor)"),
                 ("src/bowen/engine/contact.py", BAND, "return params.contact_band_max * (SCALE_MAX / 2) / SCALE_MAX"),
                 ("src/bowen/engine/moves.py", ROUTING, "return 0.5"))),
    # --- M11.1d: sign-inverted mutants of each passing criterion's core rule --------------------
    Mutant("triangle-roles-swapped", SIGN, ("M11.C.3",), "src/bowen/engine/moves.py",
           "    outsider = target\n    seeker = state.people[event.sender]\n",
           "    outsider, partner = partner, target\n    seeker = state.people[event.sender]\n",
           "M1.C.1's roles swapped back to step 5's alliance reading: sender and target inside, the partner outside"),
    Mutant("steepness-inverted", SIGN, LEVEL, "src/bowen/engine/contact.py",
           "return SCALE_MAX / max(person.functional_level, params.functional_level_floor)",
           "return max(person.functional_level, params.functional_level_floor) / (SCALE_MAX / 4)",
           "M4.C.1a's steepness rises with functional_level instead of falling"),
    Mutant("mixing-weight-inverted", SIGN, ("M11.C.41", "M11.C.16"), "src/bowen/policy/policy.py",
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
           "moved = min(excess(seeker), rate * excess(seeker))",
           "moved = min(excess(seeker), rate * max(0.0, SCALE_MAX / 10 - excess(seeker)))",
           "M1.C.1's relief falls as the seeker's tension rises (still conserved, so the ledger holds)"),
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


def judge(mutant: Mutant, outcome: str, readouts=()) -> str:
    """A red mutant is **reversed** when every gating readout's interval lies wholly on the side opposite its
    direction. Report-only readouts (never tested) and equivalence readouts (direction 0) are not gating. A mutant
    whose replacement text is broken is **broken** for every kind and never counts as proof; an engine exception
    under a mutant (RAISED) is a red, as the criterion can no longer pass."""
    if outcome == "BROKEN":
        return "broken"
    if mutant.kind == REPRESENTATION:
        return "unchanged" if outcome == "PASS" else "encoding artefact"
    if outcome == "PASS":
        return "survived"
    gating = [x for x in readouts if x["direction"] and not x.get("report_only")]
    if gating and all(x["mean_difference"] * x["direction"] < 0 and abs(x["mean_difference"]) > x["half_width"]
                      for x in gating):
        return "reversed"
    return "red"


def render(rows, skipped, engine: str) -> str:
    lines = [
        "# Phase C mutation record",
        "",
        "Generated by `tools/mutation_record.py`; do not edit. Each mutant is a literal source replacement applied",
        "to a temporary copy of the repository, run against the criteria that pass in",
        "`docs/phase_c_ensemble_record.md`, with the same adaptive rules. A deletion, named or sign-inverted mutant",
        "is **red** when the criterion stops passing and **survived** when it still passes; a survivor means that",
        "mutant does not prove the criterion. A red mutant is marked **reversed** when every gating readout's interval",
        "lies wholly on the side opposite its declared direction: the result flipped, rather than vanished (`M11.1d`).",
        "Report-only and equivalence readouts are not gating. A mutant whose replacement text is broken (an",
        "unimported name) is **broken** and proves nothing; an engine exception under a mutant is a red. A",
        "representation mutant (`M11.1c`) should leave every verdict",
        "unchanged; a change is reported as an **encoding artefact**.",
        "",
        f"engine_hash: {engine}",
        "",
        "Results are cached by `tools/mutant_runner.py` under the engine hash and each mutant's own edits",
        "(`docs/records_cache/mutation.json`); this file is rendered from that cache.",
        "",
        "| Mutant | Kind | What it changes | Criterion | Verdict under mutant | Seeds | Difference under mutant | Result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mutant, result in rows:
        note = f" ({result['error']})" if result.get("error") else ""
        diffs = "; ".join(f"`{x['readout']}` {x['mean_difference']:+.3g} ± {x['half_width']:.2g}"
                          for x in result.get("readouts", ()))
        lines.append(f"| `{mutant.id}` | {mutant.kind} | {mutant.what} | `{result['criterion']}` | "
                     f"{result['outcome']}{note} | {result['seeds']} | {diffs or '—'} | "
                     f"**{judge(mutant, result['outcome'], result.get('readouts', ()))}** |")
    lines += ["", "## Not run", "",
              "Criteria a mutant targets that do not pass at the central setting — a mutant cannot prove a failing",
              "criterion.", ""]
    lines += [f"- `{m.id}` → `{cid}`" for m, cid in skipped] or ["- none"]
    lines += ["", "## Machine-readable", "", "```json",
              json.dumps([{"mutant": m.id, "kind": m.kind, "criterion": r["criterion"], "outcome": r["outcome"],
                           "seeds": r["seeds"], "result": judge(m, r["outcome"], r.get("readouts", ())),
                           "readouts": r.get("readouts", []), **({"error": r["error"]} if r.get("error") else {})}
                          for m, r in rows],
                         indent=1, default=float),
              "```", ""]
    return "\n".join(lines)


def build(cache: Cache, workers: int | None = None) -> str:
    """The record, from the cache. With ``workers``, missing results are run first; without, a missing result raises
    KeyError, so a test can check the committed record without running anything."""
    from src.bowen.ensemble.criteria import CRITERIA

    passed, engine = passing(), engine_hash()
    rows, skipped = [], []
    for mutant in MUTANTS:
        run_ids, not_run = targets(mutant, list(CRITERIA), passed)
        skipped += [(mutant, cid) for cid in not_run if mutant.criteria != ("*",)]
        if not run_ids:
            continue
        if workers is not None:
            results = run_cached(cache, mutant, run_ids, workers)
        else:
            results = [cache.get(result_key(engine, mutant.definition(), cid)) for cid in run_ids]
            if None in results:
                raise KeyError(f"{mutant.id}: no cached result for the current engine and edits")
        rows += [(mutant, result) for result in results]
    return render(rows, skipped, engine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    cache = Cache("mutation")
    text = build(cache, args.workers)
    cache.save()
    RECORD.write_text(text, encoding="utf-8")
    print(f"wrote {RECORD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
