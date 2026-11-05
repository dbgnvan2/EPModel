"""Rerun every passing composite criterion with learning removed, and write the committed emergence audit record.

Purpose: ask of the whole model at once whether a passing result needs learning (`M11.1g`): each passing composite or
         mixed row is rerun under ablations of the four learning paths (L0 all, L1-L4 one each), and each row that
         survives L0 is rerun under L0 joined with each non-learning policy term and the row's own mutants (the
         carrier candidates C1-C7), to name what carries it. A survivor is evidence for triage, not a defect by itself;
         the defect test is `M11.1d`'s, decided by the owner.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.1g, #M11.1d, #M11.5, #M11.1a; docs/plan_v3_emergence_audit.md
Tests:   tests/test_emergence_audit.py

    python3 tools/emergence_audit.py [--workers N]
    python3 tools/emergence_audit.py --criteria KEY... [--candidates LABEL...] [--workers N]   (one machine's share)
    python3 tools/emergence_audit.py --render-only
    python3 tools/emergence_audit.py --probe NAME      (in a mutated copy only: EA-3 to EA-7's short runs)

Each ablation and carrier is a ``tools.mutant_runner.Mutant``: a literal replacement applied to a temporary copy of the
repository, never to the working tree, that must match exactly once (`M11.1a`). The rows and their classes are read
from the three ensemble records' JSON blocks; which rows are excluded, the controls and the stage are read from
``config/audit/emergence_audit.md``. Phase C rows run through Phase C's child (``tools/mutant_runner.py``), Phase D
rows through ``tools/phase_d_mutant_child.py``. Verdicts are ``tools/mutation_record.py``'s ``judge``: survived, red,
reversed; a broken edit, a failed child and an engine exception are reported and never cached (``record_cache``).

The results are cached in ``docs/records_cache/emergence_audit.json`` under ``engine_hash`` (both children's hashes,
which cover ``src/bowen``, ``config/bowen``, the rules, both children and ``apply`` and ``run_in_copy``) and each
mutant's edits. This file, the config file and the record are not in it, so a label, a reason or the renderer changes
without a rerun (learnings P38). Nothing here changes what a run computes. The cache records the NumPy version its
results were made under (``_meta``, as the sweep's does, ``tools/version_2_sweep.check_numpy``), so two machines'
shares can be merged by ``tools/merge_cache.py`` (T-184; ``docs/two_mac_sweep.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.mutant_runner import Mutant, apply, run_in_copy  # noqa: E402,F401
from tools.record_cache import Cache, exit_status, result_key, run_cached  # noqa: E402
from tools.record_json import record_json  # noqa: E402

CONFIG = REPO / "config" / "audit" / "emergence_audit.md"
RECORD = REPO / "docs" / "emergence_audit_record.md"
CACHE = "emergence_audit"
RULES_VIEW = REPO / "docs" / "spec_rules" / "M11.md"
RECORDS = {
    "phase_c": REPO / "docs" / "phase_c_ensemble_record.md",
    "phase_d": REPO / "docs" / "phase_d_ensemble_record.md",
    "phase_d2": REPO / "docs" / "phase_d2_ensemble_record.md",
}
CHILDREN = {"phase_c": "mutant_runner.py", "phase_d": "phase_d_mutant_child.py", "phase_d2": "phase_d_mutant_child.py"}
AUDITED = ("composite", "mixed")  # M11.5: a premise or a check passing is never a finding
CLASSES = ("premise", "composite", "check", "mixed")
ABLATION, CARRIER = "ablation", "carrier"
RED = ("red", "reversed")
# plan §7: pilot = step 3 (L0 on the pilot rows); l0 = step 4 (L0 on every in-scope row, no carriers, owner
# 2026-11-03); carriers = steps 4 and 6 (L0 on every row and the carriers of each L0 survivor, no L1 to L4, owner
# 2026-11-04); full = steps 4 to 6 (L0 to L4 on every row, and the carriers of each L0 survivor)
STAGES = ("pilot", "l0", "carriers", "full")
CARRIER_STAGES = ("carriers", "full")
STAGE_TEXT = {
    "pilot": " The pilot (plan §7 step 3): L0 only, on the pilot rows and the two controls; no carriers.",
    "l0": " The L0 stage (plan §7 step 4): L0 only, on every in-scope row and the two controls; L1 to L4 and the "
          "carriers are not run.",
    "carriers": " The carriers stage (plan §7 steps 4 and 6, owner 2026-11-04): L0 on every in-scope row and the two "
                "controls, and the carriers of each L0 survivor; L1 to L4 are not run.",
    "full": "",
}
NOT_EVIDENCE = ("broken", "not run (child failed)")  # judge's: no evidence either way
# plan §2: the mutation records' mutants whose edits must be L0's, so the audit's L0 rows can be checked against them
L0_TWINS = ("learning-off-in-both-arms", "learner-off-share-kept")
CONSTANTS = "config/bowen/constants.md"
LEARNER = "src/bowen/engine/learner.py"
POLICY = "src/bowen/policy/policy.py"


def _constant(name: str, old: str, new: str) -> tuple[str, str, str]:
    return CONSTANTS, f"| `{name}` | {old} |", f"| `{name}` | {new} |"


def _mutant(mid: str, kind: str, edits: tuple[tuple[str, str, str], ...], what: str) -> Mutant:
    (file, old, new), *also = edits
    return Mutant(mid, kind, (), file, old, new, what, also=tuple(also))


# --- the ablations (plan §2) -------------------------------------------------------------------------------------
L0 = _mutant("L0", ABLATION, (_constant("learning_rate", "0.2", "0.0"), _constant("witnessed_cutoff_gain", "0.05", "0.0")),
             "all learning off: no learned value moves from 0 (the same edit as `LEARNING_OFF`)")
L1 = _mutant("L1", ABLATION, ((LEARNER,
              "            felt = relief.get(pid, 0.0) + params.cross_person_weight * cross_person_relief(state, pid, act.others, relief)\n",
              "            felt = params.cross_person_weight * cross_person_relief(state, pid, act.others, relief)\n"),),
             "own relief not credited (`M4.D.6a`): the actor's own felt change is left out of the signal")
L2 = _mutant("L2", ABLATION, (_constant("cross_person_weight", "0.5", "0.0"),),
             "cross-person term off (`M4.D.6e`): the others' change in anxiety is left out of the signal")
L3 = _mutant("L3", ABLATION, (_constant("witnessed_cutoff_gain", "0.05", "0.0"),),
             "witnessed-cutoff learning off (`M4.D.6f`)")
L4 = _mutant("L4", ABLATION, (_constant("soothing_review_penalty", "0.5", "0.0"),),
             "soothing review off (`M4.D.6`): a review that finds soothing did not help cuts nothing")
ABLATIONS = (L0, L1, L2, L3, L4)


def _phase_d_mutant(mid: str) -> Mutant:
    from tools.phase_d_mutation_record import MUTANTS

    return next(m for m in MUTANTS if m.id == mid)


# --- the carrier candidates C1-C6 (plan §4); C7 is each row's own mutants, built per row ---------------------------
CARRIERS = (
    _mutant("C1", CARRIER, ((POLICY,
             "    return min(1.0, max(0.0, obs.functional_level / (layer * params.capacity_level_per_layer)))\n",
             "    return 1.0\n"),),
            "every capacity layer available (`M4.D.3a`)"),
    _mutant("C2", CARRIER, ((lambda m: ((m.file, m.old, m.new), *m.also))(
                _phase_d_mutant("mixing-weight-blind-to-level-in-policy-only"))),
            "the mixing weight blind to level, in the policy only (`M4.D.1a`)"),
    _mutant("C3", CARRIER, (_constant("focus_steer_gain", "1.0", "0.0"),), "focus steering off (`M4.D.1g`)"),
    _mutant("C4", CARRIER, (_constant("perspective_steer_gain", "0.5", "0.0"),), "perspective steering off (`M4.D.1a`)"),
    _mutant("C5", CARRIER, ((POLICY,
             "    return obs.systems_perspective if view.deviation > params.loaded_tie_threshold else 1.0\n",
             "    return 1.0\n"),),
            "loaded-tie gate off (`M4.D.3b`)"),
    _mutant("C6", CARRIER, (_constant("competing_urge_gain", "0.5", "0.0"),), "competing urges off (`M4.D.1d`)"),
)
NOT_TESTED = ("the legal set's structural preconditions (a tie to a living person; a closed triad for `TRIANGLE`): "
              "they define what an act is and are not mutated (plan §4)")


def edits(mutant: Mutant) -> list[tuple[str, str, str]]:
    return [(mutant.file, mutant.old, mutant.new), *mutant.also]


def joint(base: Mutant, candidate: Mutant, mid: str) -> Mutant | None:
    """``base`` with ``candidate``'s edits added: None when the candidate adds no edit (its edits are all ``base``'s).
    An edit of the same text to something else cannot be joined and raises."""
    mine = {old: (file, new) for file, old, new in edits(base)}
    extra = []
    for file, old, new in edits(candidate):
        if old in mine:
            if mine[old] != (file, new):
                raise SystemExit(f"{mid}: {candidate.id} edits the text {base.id} edits, to something else")
            continue
        extra.append((file, old, new))
    if not extra:
        return None
    return Mutant(mid, CARRIER, (), base.file, base.old, base.new, f"{base.what}; with {candidate.what}",
                  also=(*base.also, *extra))


# --- the config file ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    exclusions: dict[str, str]
    controls: tuple[tuple[str, str, str, str, str], ...]  # (control, criterion, ablation, expected, reason)
    stage: str
    pilot: dict[str, str]


def _table(text: str, heading: str, header: str) -> list[list[str]]:
    part = text.split(f"## {heading}\n", 1)
    if len(part) != 2 or text.count(f"## {heading}\n") != 1:
        raise SystemExit(f"{CONFIG.name}: the section '## {heading}' is missing or repeated")
    lines = part[1].split("\n## ", 1)[0].splitlines()
    try:
        start = lines.index(header)
    except ValueError:
        raise SystemExit(f"{CONFIG.name}: '## {heading}' has no table headed {header!r}") from None
    if not lines[start + 1].startswith("|---"):
        raise SystemExit(f"{CONFIG.name}: '## {heading}''s table has no separator line")
    width = header.count("|") - 1
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != width:
            raise SystemExit(f"{CONFIG.name}: a row of '## {heading}' has {len(cells)} cells, not {width} (a `|` in a "
                             f"reason?): {line!r}")
        rows.append(cells)
    return rows


def _unique(pairs: list[list[str]], heading: str) -> dict[str, str]:
    out = {}
    for cell, reason in pairs:
        key = _id(cell)
        if key in out:
            raise SystemExit(f"{CONFIG.name}: '## {heading}' names {key} twice")
        out[key] = reason
    return out


def _id(cell: str) -> str:
    if not (cell.startswith("`") and cell.endswith("`")):
        raise SystemExit(f"{CONFIG.name}: a criterion is written in backticks, not {cell!r}")
    return cell.strip("`")


def read_config(text: str | None = None) -> Config:
    """Purpose: the exclusions, controls, stage and pilot rows, parsed strictly.
    Spec:    docs/plan_v3_emergence_audit.md#3, #5
    Tests:   tests/test_emergence_audit.py::test_m111g_ea12_no_row_dropped_silently,
             tests/test_emergence_audit.py::test_m111g_ea12_the_config_is_parsed_strictly
    """
    text = CONFIG.read_text(encoding="utf-8") if text is None else text
    exclusions = _unique(_table(text, "Exclusions", "| criterion | reason |"), "Exclusions")
    controls = []
    for control, cid, ablation, expected, reason in _table(text, "Controls",
                                                            "| control | criterion | ablation | expected | reason |"):
        if ablation not in {a.id for a in ABLATIONS}:
            raise SystemExit(f"{CONFIG.name}: control {control!r} names an unknown ablation {ablation!r}")
        if expected not in ("red", "survived"):
            raise SystemExit(f"{CONFIG.name}: control {control!r} expects {expected!r}, not red or survived")
        controls.append((control, _id(cid), ablation, expected, reason))
    stage = dict(_table(text, "Stage", "| setting | value |")).get("stage")
    if stage not in STAGES:
        raise SystemExit(f"{CONFIG.name}: the stage is {stage!r}, not one of {', '.join(STAGES)}")
    pilot = _unique(_table(text, "Pilot rows", "| criterion | reason |"), "Pilot rows")
    return Config(exclusions, tuple(controls), stage, pilot)


# --- scope (plan §3) -----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Row:
    key: str
    phase: str
    cls: str
    seeds: int


def read_records() -> dict[str, list[dict]]:
    return {phase: record_json(path) for phase, path in RECORDS.items()}


def in_scope(records: dict[str, list[dict]], config: Config, limbs: dict[str, frozenset[str]] | None = None):
    """Purpose: the passing composite and mixed rows of the records, less the exclusions; and what was left out.
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.1g, #M11.5; docs/plan_v3_emergence_audit.md#3
    Tests:   tests/test_emergence_audit.py::test_m111g_ea1_scope_is_passing_composite_and_mixed,
             tests/test_emergence_audit.py::test_t192_a_mixed_row_with_only_premise_limbs_leaves_scope

    Returns (audited rows, every row by key, the passing rows left out by class as (key, class), the excluded rows as
    (key, reason)). A mixed row each of whose gating readouts is a declared premise limb (``limbs``; by default
    ``tools/version_2_status.premise_limbs``, both dispositions files) has no composite limb to judge, so it is left
    out by class, as a premise is (T-192, owner 2026-11-05). A key held by two records, a class outside `M11.5`'s
    four, a stale exclusion and a control on a row that does not pass all raise."""
    if limbs is None:
        from tools.version_2_status import premise_limbs

        limbs = premise_limbs()
    every: dict[str, tuple[str, dict]] = {}
    for phase, rows in records.items():
        for r in rows:
            if r["criterion"] in every:
                raise SystemExit(f"{r['criterion']}: in two records ({every[r['criterion']][0]} and {phase})")
            if r["class"] not in CLASSES:
                raise SystemExit(f"{r['criterion']}: class {r['class']!r} is not one of M11.5's {CLASSES}")
            every[r["criterion"]] = (phase, r)
    passing = [(k, p, r) for k, (p, r) in every.items() if r["outcome"] == "PASS"]
    def all_premise(k, r):
        declared = limbs.get(k, limbs.get(k.split("[", 1)[0], frozenset()))
        gating = {o["readout"] for o in r.get("readouts", []) if not o.get("report_only")}
        return r["class"] == "mixed" and bool(gating) and gating <= declared

    no_composite = [(k, p, r) for k, p, r in passing if all_premise(k, r)]
    composite = [(k, p, r) for k, p, r in passing if r["class"] in AUDITED and not all_premise(k, r)]
    stale = sorted(set(config.exclusions) - {k for k, _, _ in composite})
    if stale:
        raise SystemExit(f"{CONFIG.name}: excludes {stale}, which are not passing composite or mixed rows")
    for control, cid, *_ in config.controls:
        if cid not in every or every[cid][1]["outcome"] != "PASS":
            raise SystemExit(f"{CONFIG.name}: the {control} control {cid} does not pass in its record")
    audited = [Row(k, p, r["class"], r["seeds"]) for k, p, r in composite if k not in config.exclusions]
    by_class = [(k, r["class"]) for k, _, r in passing if r["class"] not in AUDITED]
    by_class += [(k, NO_COMPOSITE) for k, _, _ in no_composite]
    excluded = [(k, config.exclusions[k]) for k, _, _ in composite if k in config.exclusions]
    rows = {k: Row(k, p, r["class"], r["seeds"]) for k, (p, r) in every.items()}
    return audited, rows, by_class, excluded


NO_COMPOSITE = "mixed, every gating readout a declared premise limb"


def check_registries(rows: dict[str, Row]) -> None:
    """Each row's child is chosen by the registry that holds its key: Phase C's for ``CRITERIA``, Phase D's for
    ``CRITERIA_D`` and ``CRITERIA_D2``. The record a row came from must agree."""
    from src.bowen.ensemble.criteria import CRITERIA
    from src.bowen.ensemble.criteria_d import CRITERIA_D
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2

    for key, row in rows.items():
        held = [n for n, reg in (("phase_c", CRITERIA), ("phase_d", CRITERIA_D), ("phase_d2", CRITERIA_D2)) if key in reg]
        if held != [row.phase]:
            raise SystemExit(f"{key}: in {row.phase}'s record but registered in {held or 'no registry'}")


# --- each row's own mutants (C7) ---------------------------------------------------------------------------------
def own_mutants(row: Row) -> list[Mutant]:
    """Every deletion, named or sign-inverted mutant the row's mutation record names for it, as that record targets."""
    if row.phase == "phase_c":
        from tools.mutation_record import MUTANTS, REPRESENTATION, targets

        return [m for m in MUTANTS if m.kind != REPRESENTATION and m.criteria != ("*",)
                and targets(m, [row.key], {row.key})[0]]
    from tools.phase_d_mutation_record import MUTANTS

    return [m for m in MUTANTS if row.key in m.criteria]


def carrier_candidates(row: Row) -> tuple[list[tuple[str, Mutant]], list[str]]:
    """(label, joint mutant) for C1-C6 and each of the row's own mutants (C7), each joined with L0; and the labels
    of candidates not run, each with why: it adds nothing to L0, or it has an earlier candidate's edits."""
    out, same = [], []
    for label, cand in [*((c.id, c) for c in CARRIERS), *((f"C7:{m.id}", m) for m in own_mutants(row))]:
        mutant = joint(L0, cand, f"L0+{label}")
        twin = next((lab for lab, m in out if mutant is not None and m.definition() == mutant.definition()), None)
        if mutant is None:
            same.append(f"{label} (the same edits as L0)")
        elif twin is not None:
            same.append(f"{label} (the same edits as {twin})")
        else:
            out.append((label, mutant))
    return out, same


# --- verdicts (plan §3) --------------------------------------------------------------------------------------------
def verdict(row: Row, mutant: Mutant, result: dict, limbs: dict[str, frozenset[str]], alpha: float) -> tuple[str, str]:
    """Purpose: the row's verdict under ``mutant`` and a note: ``judge``'s, on a mixed row's composite limbs only.
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.1g, #M11.5; docs/plan_v3_emergence_audit.md#3
    Tests:   tests/test_emergence_audit.py::test_m111g_ea9_verdicts_follow_judge,
             tests/test_emergence_audit.py::test_m111g_ea10_premise_limb_does_not_decide_a_mixed_row

    A mixed row's premise limbs (``tools/version_2_status.premise_limbs``) are run and shown but do not decide it, as on
    the status page: a FAIL whose composite limbs all hold (each limb's Holm decision recomputed from the row's p values
    over every directional limb, premise limbs included, as the runner makes it, so a premise limb with the smallest p
    can stop the step-down before a composite limb: the same family as the row's own verdict) survives. A required move that never occurred, a run that did
    not converge and an engine exception are red whatever the limbs."""
    from tools.mutation_record import judge

    outcome, readouts = result["outcome"], result.get("readouts", [])
    premise = limbs.get(row.key, limbs.get(row.key.split("[", 1)[0], frozenset())) if row.cls == "mixed" else frozenset()
    if not premise or outcome != "FAIL" or result.get("missing_moves"):
        return judge(mutant, outcome, readouts), ""
    composite = [r for r in readouts if r["readout"] not in premise]
    directional = [r for r in readouts if r["direction"] and not r.get("report_only")]
    if any("p" not in r for r in directional):
        raise SystemExit(f"{row.key}: a mixed row with premise limbs needs each limb's p; its child printed none")
    from src.bowen.ensemble.stats import holm

    holds = dict(zip((r["readout"] for r in directional), holm([r["p"] for r in directional], alpha)))
    held = all(holds[r["readout"]] for r in composite if r["direction"] and not r.get("report_only")) and all(
        _within(row, r) for r in composite if r["direction"] == 0 and not r.get("report_only"))
    failed = sorted(r["readout"] for r in directional if r["readout"] in premise and not holds[r["readout"]])
    if held:
        return "survived", f"premise limb {', '.join(failed)} failed; the composite limbs hold"
    return judge(mutant, outcome, composite), ""


def _within(row: Row, r: dict) -> bool:
    if "within_bound" not in r:
        raise SystemExit(f"{row.key}: equivalence limb {r['readout']} has no within_bound; its child printed none")
    return bool(r["within_bound"])


def _alpha() -> float:
    from src.bowen.io.load import load_constants

    return float(load_constants()["ensemble_alpha"])


# --- the engine hash and the cache ---------------------------------------------------------------------------------
def engine_hash() -> str:
    """Both children's hashes: Phase C's (``code_hash`` with ``tools/mutant_runner.py``) and Phase D's (the Phase D
    mutation record's: its child and the source of ``apply`` and ``run_in_copy``). Both cover ``src/bowen``,
    ``config/bowen`` and the rules; nothing of this tool is in it."""
    from tools import record_cache
    from tools.phase_d_mutation_record import engine_hash as phase_d_hash

    return hashlib.sha256(f"{record_cache.engine_hash()}|{phase_d_hash()}".encode()).hexdigest()


def _results(cache: Cache, mutant: Mutant, rows: list[Row], workers: int | None, engine: str) -> dict[str, dict]:
    """Each row's result under ``mutant``, from the cache or (with ``workers``) run in that row's child."""
    out = {}
    for child in sorted({CHILDREN[r.phase] for r in rows}):
        ids = [r.key for r in rows if CHILDREN[r.phase] == child]
        if workers is None:
            for key in ids:
                found = cache.get(result_key(engine, mutant.definition(), key))
                if found is None:
                    raise KeyError(f"{mutant.id} → {key}: no cached result for the current engine and edits")
                out[key] = found
        else:
            for result in run_cached(cache, mutant, ids, workers, f"{mutant.id} ({child})", child=child, engine=engine):
                out[result["criterion"]] = result
    return out


# --- building the record -------------------------------------------------------------------------------------------
def m115(key: str, text: str | None = None) -> tuple[str, str]:
    """The row's `M11.5` entry, as the rules view gives it: (the rule it rests on, the note)."""
    text = RULES_VIEW.read_text(encoding="utf-8") if text is None else text
    base = key.split("[", 1)[0]
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith(f"| {base} |") and len(cells) == 4:
            return cells[2], cells[3]
    return "(no `M11.5` row)", ""


def build(cache: Cache, workers: int | None = None, *, records: dict | None = None, config: Config | None = None,
          limbs: dict | None = None, engine: str | None = None, alpha: float | None = None,
          agreement: bool = True) -> str:
    """Purpose: the record, from the cache; with ``workers``, missing results are run first, and without, a missing
             one raises KeyError, so a test can check the committed record without running anything.
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.1g; docs/plan_v3_emergence_audit.md#3, #4, #7
    Tests:   tests/test_emergence_audit.py::test_m111g_ea14_emergence_audit_record_is_current,
             tests/test_emergence_audit.py::test_m111g_ea8_build_refuses_a_failed_control,
             tests/test_emergence_audit.py::test_m111g_ea11_carriers_named_only_for_survivors,
             tests/test_emergence_audit.py::test_m111g_ea12_l0_stage_runs_l0_on_every_row_and_nothing_else,
             tests/test_emergence_audit.py::test_m111g_ea12_carriers_stage_runs_l0_and_carriers_and_no_l1_to_l4
    """
    from_files = records is None
    records = read_records() if from_files else records
    config = read_config() if config is None else config
    if limbs is None:
        from tools.version_2_status import premise_limbs

        limbs = premise_limbs()
    engine = engine_hash() if engine is None else engine
    alpha = _alpha() if alpha is None else alpha
    audited, rows, by_class, excluded = in_scope(records, config, limbs)
    if from_files:
        check_registries(rows)
    def premise_of(row):
        return limbs.get(row.key, limbs.get(row.key.split("[", 1)[0], frozenset())) if row.cls == "mixed" else frozenset()

    phase_c_premise = sorted(r.key for r in audited if r.phase == "phase_c" and premise_of(r))
    if phase_c_premise:
        raise SystemExit(f"{phase_c_premise}: a Phase C mixed row with premise limbs cannot be judged; the Phase C "
                         "child (tools/mutant_runner.py) prints no p or within_bound for its limbs")
    undeclared = ("premise limbs not declared (docs/version_1_dispositions.md, docs/version_2_dispositions.md), "
                  "so judged on every limb")
    keys = {r.key for r in audited}
    stray = sorted(set(config.pilot) - keys)
    if stray:
        raise SystemExit(f"{CONFIG.name}: pilot rows {stray} are not in scope")
    pilot = config.stage == "pilot"
    run_rows = [r for r in audited if r.key in config.pilot] if pilot else audited
    full = config.stage == "full"
    ablations = ABLATIONS if full else (L0,)

    def entry(row, mutant, result, label):
        v, note = verdict(row, mutant, result, limbs, alpha)
        note = note or result.get("error", "")
        if row.cls == "mixed" and not premise_of(row):
            note = f"{note}; {undeclared}" if note else undeclared
        return {"criterion": row.key, "class": row.cls, "ablation": label, "outcome": result["outcome"],
                "seeds": result["seeds"], "verdict": v, "note": note,
                "readouts": result.get("readouts", []), "premise_limbs": sorted(premise_of(row))}

    controls = []
    for control, cid, aid, expected, reason in config.controls:
        mutant = next(a for a in ABLATIONS if a.id == aid)
        row = rows[cid]
        e = entry(row, mutant, _results(cache, mutant, [row], workers, engine)[cid], aid)
        if e["verdict"] in NOT_EVIDENCE:
            raise SystemExit(f"control {control} {cid} under {aid} was not run ({e['verdict']}): rerun it; this says "
                             "nothing about the ablation")
        held = (e["verdict"] in RED and e["outcome"] == "FAIL") if expected == "red" else e["verdict"] == "survived"
        controls.append({**e, "control": control, "expected": expected, "reason": reason, "held": held})
    failed = [c for c in controls if not c["held"]]
    if failed:
        raise SystemExit("a control did not hold, so the ablation is not doing what it says; the record is not "
                         "rendered: " + "; ".join(f"{c['control']} {c['criterion']} under {c['ablation']}: "
                                                  f"{c['verdict']}, expected {c['expected']}" for c in failed))

    verdicts = []
    for mutant in ablations:
        results = _results(cache, mutant, run_rows, workers, engine)
        verdicts += [entry(r, mutant, results[r.key], mutant.id) for r in run_rows]

    carriers = []
    if config.stage in CARRIER_STAGES:
        survivors = [r for r in run_rows if any(v["criterion"] == r.key and v["ablation"] == "L0"
                                                 and v["verdict"] == "survived" for v in verdicts)]
        for row in survivors:
            candidates, same = carrier_candidates(row)
            tried = []
            for label, mutant in candidates:
                tried.append(entry(row, mutant, _results(cache, mutant, [row], workers, engine)[row.key], label))
            rests, note = m115(row.key)
            carriers.append({"criterion": row.key, "candidates": tried, "not_run": same,
                             "carriers": [t["ablation"] for t in tried if t["verdict"] in RED and t["outcome"] == "FAIL"],
                             "red_without_fail": [f"{t['ablation']} ({t['outcome']})" for t in tried
                                                  if t["verdict"] in RED and t["outcome"] != "FAIL"],
                             "m115_rests_on": rests, "m115_note": note})

    agreed = check_agreement(controls + verdicts, engine) if agreement else []
    not_yet = [] if not pilot else [r.key for r in audited if r.key not in config.pilot]
    return render(engine, config, audited, by_class, excluded, controls, verdicts, carriers, not_yet, agreed)


def share(cache: Cache, keys: list[str], workers: int, labels: list[str] | None = None, *,
          records: dict | None = None, config: Config | None = None, limbs: dict | None = None,
          engine: str | None = None, alpha: float | None = None) -> int:
    """Purpose: one machine's share of a split run (T-184): the stage's ablations on the named rows and, at the carriers
             and full stages, the carriers of each that survives L0; with ``labels``, L0 and only those carriers, so
             one row's carriers can be split across machines. Every key and label is checked before anything runs.
             The caller flushes the cache, pruning nothing, and writes no record, so every other cached result is kept.
    Spec:    docs/plan_v3_emergence_audit.md#7; docs/two_mac_sweep.md
    Tests:   tests/test_emergence_audit.py::test_t184_a_share_runs_only_its_rows_and_keeps_the_others,
             tests/test_emergence_audit.py::test_t184_a_share_refuses_an_unknown_row_or_label_before_running

    Returns the number of results asked for (cached or run)."""
    records = read_records() if records is None else records
    config = read_config() if config is None else config
    if limbs is None:
        from tools.version_2_status import premise_limbs

        limbs = premise_limbs()
    engine = engine_hash() if engine is None else engine
    alpha = _alpha() if alpha is None else alpha
    audited = {r.key: r for r in in_scope(records, config, limbs)[0]}
    unknown = sorted(set(keys) - set(audited))
    if unknown:
        raise SystemExit(f"{unknown}: not rows the audit judges (the controls run only in a whole run); nothing ran")
    chosen = [audited[k] for k in dict.fromkeys(keys)]
    carry = config.stage in CARRIER_STAGES
    if labels and not carry:
        raise SystemExit(f"--candidates names carriers, and the {config.stage} stage runs none; nothing ran")
    for row in chosen if labels else ():
        bad = sorted(set(labels) - {label for label, _ in carrier_candidates(row)[0]})
        if bad:
            raise SystemExit(f"{row.key}: {bad} are not among its carrier candidates; nothing ran")
    ablations = (L0,) if labels or config.stage != "full" else ABLATIONS
    asked, l0 = 0, {}
    for mutant in ablations:
        results = _results(cache, mutant, chosen, workers, engine)
        asked += len(chosen)
        l0 = results if mutant is L0 else l0
    for row in chosen if carry else ():
        v, _ = verdict(row, L0, l0[row.key], limbs, alpha)
        if v != "survived":
            print(f"{row.key}: {v} under L0, so no carriers run", flush=True)
            continue
        for label, mutant in carrier_candidates(row)[0]:
            if not labels or label in labels:
                _results(cache, mutant, [row], workers, engine)
                asked += 1
    return asked


def check_numpy(cache: Cache) -> None:
    """Purpose: refuse a run under a NumPy other than the sweep machines' pin or other than the one the cache's results
             were made under, and record it in the cache's ``_meta`` (T-184): the sweep's own check, so
             ``tools/merge_cache.py`` can show two machines' results match.
    Spec:    docs/two_mac_sweep.md
    Tests:   tests/test_emergence_audit.py::test_t184_a_run_records_numpy_and_refuses_another"""
    import numpy as np

    from tools import version_2_sweep

    version_2_sweep.check_numpy(cache, np.__version__, version_2_sweep.sweep_numpy_pin())


def check_agreement(entries: list[dict], engine: str) -> list[str]:
    """Purpose: the audit's L0 rows must agree with the mutation records' rows for any mutant whose edits are L0's, on
             the same engine (plan §2: ``learning-off-in-both-arms`` on `M11.C.39`, ``learner-off-share-kept`` on
             `M11.C.34`). Returns one line per comparison made or not made; a disagreement raises.
    Spec:    docs/plan_v3_emergence_audit.md#2
    Tests:   tests/test_emergence_audit.py::test_m111g_ea8_controls_hold_in_the_record,
             tests/test_emergence_audit.py::test_m111g_ea8_agreement_disagreement_raises
    """
    from tools import mutation_record as c_tool
    from tools import phase_d_mutation_record as d_tool
    from tools import record_cache

    if engine != engine_hash():
        return ["not checked: the audit's results are not on the current engine"]
    by_id = {m.id: m for m in (*c_tool.MUTANTS, *d_tool.MUTANTS)}
    drifted = [t for t in L0_TWINS if t not in by_id or by_id[t].definition() != L0.definition()]
    if drifted:
        raise SystemExit(f"{drifted}: plan §2's L0 twins are missing or their edits are no longer L0's")
    lines = []
    for tool, current in ((c_tool, record_cache.engine_hash()), (d_tool, d_tool.engine_hash())):
        same = {m.id for m in tool.MUTANTS if m.definition() == L0.definition()}
        if not same:
            continue
        text = tool.RECORD.read_text(encoding="utf-8")
        found = re.search(r"^engine_hash: (\S+)$", text, re.M)
        if found is None or found.group(1) != current:
            lines.append(f"not checked against `{tool.RECORD.name}`: it is not on the current engine")
            continue
        for row in record_json(text):
            if row["mutant"] not in same:
                continue
            mine = next((e for e in entries if e["criterion"] == row["criterion"] and e["ablation"] == "L0"), None)
            if mine is None:
                lines.append(f"`{row['mutant']}` → `{row['criterion']}`: not run here yet")
                continue
            if (mine["outcome"], mine["seeds"]) != (row["outcome"], row["seeds"]):
                raise SystemExit(f"L0 on {row['criterion']} gives {mine['outcome']} at {mine['seeds']} seeds; "
                                 f"{row['mutant']}, the same edits, gives {row['outcome']} at {row['seeds']} in "
                                 f"{tool.RECORD.name}")
            lines.append(f"`{row['mutant']}` → `{row['criterion']}`: agrees ({row['outcome']}, {row['seeds']} seeds)")
    return lines


# --- rendering -----------------------------------------------------------------------------------------------------
def _diffs(e: dict) -> str:
    premise = set(e.get("premise_limbs", ()))
    return "; ".join(f"`{x['readout']}` {x['mean_difference']:+.3g} ± {x['half_width']:.2g}"
                     + (f", p {x['p']:.2g}" if "p" in x else "") + (" (reported)" if x.get("report_only") else "")
                     + (" (premise limb)" if x["readout"] in premise else "")
                     for x in e["readouts"]) or "—"


def _line(e: dict) -> str:
    note = f" ({e['note']})" if e["note"] else ""
    return (f"| `{e['criterion']}` | {e['class']} | {e['ablation']} | {e['outcome']} | {e['seeds']} | {_diffs(e)} "
            f"| **{e['verdict']}**{note} |")


def render(engine, config, audited, by_class, excluded, controls, verdicts, carriers, not_yet, agreed) -> str:
    """Purpose: the record's text, from the built entries; it ends in a JSON block (``tools/record_json.py``).
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.1g, #M11.5; docs/plan_v3_emergence_audit.md#3
    Tests:   tests/test_emergence_audit.py::test_m111g_ea15_class_shown_premise_never_audited
    """
    passing = len(audited) + len(by_class) + len(excluded)
    head = "| Criterion | Class | Ablation | Outcome | Seeds | Difference under ablation | Verdict |"
    lines = [
        "# Emergence audit record",
        "",
        "Generated by `tools/emergence_audit.py`; do not edit. Every passing composite or mixed row of the three ensemble",
        "records is rerun with learning removed, in a temporary copy of the repository, under the records' adaptive rules",
        "(`M11.1g`; `docs/plan_v3_emergence_audit.md`). A row **survived** when it still passes, **red** when it stops",
        "passing, and **reversed** when it is red and every gating readout's interval lies wholly on the side opposite its",
        "direction (`tools/mutation_record.py`'s `judge`). A mixed row is judged on its composite limbs only. A survivor",
        "is evidence for triage, not a defect by itself: the defect test is `M11.1d`'s, whether a rule that carries it",
        "names its outcome, and the owner decides it for each survivor.",
        "",
        f"engine_hash: {engine}",
        "",
        f"Stage: **{config.stage}**." + STAGE_TEXT[config.stage],
        "",
        "## Ablations",
        "",
        "| ID | What it removes | Edits |",
        "|---|---|---|",
        *(f"| {a.id} | {a.what} | {len(edits(a))} |" for a in ABLATIONS),
        "",
        "## Scope",
        "",
        f"{len(audited)} of {passing} passing rows audited; {len(excluded)} excluded; "
        f"{len(by_class)} out by class (a premise or a check passing, or a mixed row with no composite limb, is "
        "never a finding, `M11.5`).",
        "",
        "| Criterion | Class | Seeds at pass |",
        "|---|---|---|",
        *(f"| `{r.key}` | {r.cls} | {r.seeds} |" for r in audited),
        "",
        "Excluded:",
        "",
        *([f"- `{k}`: {reason}" for k, reason in excluded] or ["- none"]),
        "",
        "Out by class:",
        "",
        *([f"- `{k}` ({cls})" for k, cls in by_class] or ["- none"]),
        "",
        "## Controls",
        "",
        "Never findings. The build refuses to render the record if either does not hold.",
        "",
        "| Control | Expected | " + head[2:],
        "|---|---|---|---|---|---|---|---|---|",
        *(f"| {c['control']} | {c['expected']} " + _line(c) for c in controls),
        "",
        *(f"- {c['control']}: {c['reason']}" for c in controls),
        "",
        "## Verdicts",
        "",
        head,
        "|---|---|---|---|---|---|---|",
        *(_line(e) for e in verdicts),
        "",
        "Not run yet:",
        "",
        *([f"- `{k}`: not in the pilot; runs at the full stage" for k in not_yet]
          + (["- L1 to L4 on every row: run at the full stage (plan §7 step 5)"]
             if config.stage in ("l0", "carriers") else [])
          or ["- none"]),
        "",
        "## Carriers",
        "",
    ]
    if config.stage == "pilot":
        lines += ["Not run in the pilot: carriers are run for each L0 survivor at the full stage (plan §7 step 6).", ""]
    elif config.stage == "l0":
        survivors = [e["criterion"] for e in verdicts if e["verdict"] == "survived"]
        lines += [f"Not run at the L0 stage: carriers are run for each L0 survivor at the full stage (plan §7 step 6). "
                  f"{len(survivors)} of {len(verdicts)} rows survived L0"
                  + (": " + ", ".join(f"`{k}`" for k in survivors) if survivors else "") + ".", ""]
    elif not carriers:
        lines += ["No row survived L0, so no carrier was run.", ""]
    for c in carriers:
        found = ", ".join(c["carriers"]) or f"no carrier among {len(c['candidates'])} candidates"
        lines += [f"### `{c['criterion']}`", "", f"Carriers: **{found}**.", "",
                  f"`M11.5` says it rests on: {c['m115_rests_on']}. Note: {c['m115_note'] or '—'}", "", head,
                  "|---|---|---|---|---|---|---|", *(_line(e) for e in c["candidates"]), ""]
        if c["not_run"]:
            lines += [f"Not run: {'; '.join(c['not_run'])}.", ""]
        if c.get("red_without_fail"):
            lines += [f"Red by an exception or an undetermined row, not a FAIL, so not named a carrier: "
                      f"{'; '.join(c['red_without_fail'])}.", ""]
    lines += [f"Not tested: {NOT_TESTED}.", "", "## Agreement with the mutation records", "",
              *([f"- {a}" for a in agreed] or ["- no mutant in either mutation record has L0's edits"]), "",
              "## Machine-readable", "", "```json",
              json.dumps({"engine_hash": engine, "stage": config.stage,
                          "scope": {"audited": len(audited), "passing": passing, "excluded": [k for k, _ in excluded],
                                    "out_by_class": [k for k, _ in by_class]},
                          "controls": controls, "verdicts": verdicts, "carriers": carriers, "not_run_yet": not_yet},
                         indent=1, default=float),
              "```", ""]
    return "\n".join(lines)


# --- EA-3 to EA-7: short runs in a mutated copy ----------------------------------------------------------------------
PROBE_TICKS = 40  # [I] plan §6: a short run, long enough for each path to write a learned value at seed 0
PROBE_SEED = 0


def probe(name: str) -> dict:
    """Purpose: one short measurement of a learning path, run in a copy with an ablation applied (or none).
    Spec:    docs/plan_v3_emergence_audit.md#6 (EA-3 to EA-7)
    Tests:   tests/test_emergence_audit.py::test_m111g_ea3_l0_leaves_fewer_learned_values

    ``learned``: nonzero learned values after a policy run of the Phase B family. ``own`` and ``cross``: one act's
    learned value after a credit horizon in which only the actor's (``own``) or only the target's (``cross``) acute
    anxiety fell by 2 each tick. ``witnessed`` and ``soothing``: the learned values a witnessed-cutoff update or a
    soothing review moved (rows with a nonzero change; a path turned off still writes its rows, with no change)."""
    from src.bowen.engine.identifiers import PersonId
    from src.bowen.engine.initialise import initialise_run
    from src.bowen.engine.learner import EligibleAct, learn
    from src.bowen.engine.state import new_run_state
    from src.bowen.engine.tick import run
    from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_policy_rules, load_script
    from src.bowen.scenario.assemble import assemble
    from src.bowen.scenario.params import engine_params
    from src.bowen.scenario.policy_source import PolicySource
    from src.bowen.scenario.scripted_source import ScriptedSource

    kinds, family = load_event_kinds(), load_family()
    if name in ("own", "cross"):
        params = engine_params(load_constants())
        actor, target = PersonId("ravi"), PersonId("marta")
        state = initialise_run(new_run_state(dict(family.people), dict(family.ties), family.family, kinds), params)
        key = "mid|inside|PURSUE|marta"
        state.people[actor].eligible_acts.append(EligibleAct(tick=0, key=key, others=(target,), repetitions=0))
        felt = actor if name == "own" else target
        for _ in range(params.credit_horizon):
            start = {p: q.acute_anxiety for p, q in state.people.items()}
            state.people[felt].acute_anxiety -= 2.0
            learn(state, start, params)
            state.tick += 1
        return {"probe": name, "value": state.people[actor].learned_values.get(key, 0.0)}

    class Collect:
        def __init__(self):
            self.records = []

        def emit(self, record):
            self.records.append(record)

    script = load_script(kinds=kinds, family=family)
    events_only = ScriptedSource(script.script_id + "-events", script.ticks, script.events, ())
    parts = assemble(load_constants(), kinds, family, events_only, seed=PROBE_SEED)
    out = Collect()
    run(parts.state, PolicySource(events_only, parts.params, load_policy_rules()), parts.params, parts.visibility,
        parts.activation, out, ticks=PROBE_TICKS)
    if name == "learned":
        return {"probe": name, "value": sum(1 for p in parts.state.people.values()
                                            for v in p.learned_values.values() if v != 0.0)}
    kind = {"witnessed": "witnessed_cutoff", "soothing": "soothing_review"}[name]
    moved = sum(1 for r in out.records if getattr(r, "mechanism", None) == kind
                for _, _, change in getattr(r, "people", ()) if change)
    return {"probe": name, "value": moved}


def run_probe(mutant: Mutant | None, name: str) -> float:
    """``probe(name)`` in a temporary copy with ``mutant`` applied (None: unmodified)."""
    rows = [r for r in run_in_copy(mutant, "emergence_audit.py", "--probe", name, timeout=600) if r.get("probe") == name]
    if len(rows) != 1:
        raise SystemExit(f"probe {name} under {mutant.id if mutant else 'no mutant'}: {len(rows)} rows, not one")
    return rows[0]["value"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--probe", choices=("learned", "own", "cross", "witnessed", "soothing"))
    parser.add_argument("--criteria", nargs="+", metavar="KEY",
                        help="run only these rows (one machine's share, docs/two_mac_sweep.md): write the cache, "
                             "pruning nothing, and no record")
    parser.add_argument("--candidates", nargs="+", metavar="LABEL",
                        help="with --criteria: L0 and only these carrier candidates (C1 to C6, C7:<mutant>)")
    parser.add_argument("--render-only", action="store_true",
                        help="run nothing: render the record from the cache, or stop if a result is missing")
    args = parser.parse_args(argv)
    if args.probe:
        print(json.dumps(probe(args.probe), default=float), flush=True)
        return 0
    if args.candidates and not args.criteria:
        parser.error("--candidates splits the named rows' carriers, and needs --criteria")
    if args.criteria and args.render_only:
        parser.error("--criteria runs a share and writes no record; --render-only runs nothing")
    cache = Cache(CACHE)
    if args.render_only:  # runs nothing, so no NumPy check; the cache is never written
        try:
            text = build(cache)
        except KeyError as missing:
            raise SystemExit(f"{missing.args[0]}; the record is not written. Run the missing share (the controls run "
                             "only in a whole run), or merge the other machine's cache first.") from None
        RECORD.write_text(text, encoding="utf-8")
        print(f"wrote {RECORD.relative_to(REPO) if RECORD.is_relative_to(REPO) else RECORD}")
        return 0
    check_numpy(cache)
    if args.criteria:
        try:
            asked = share(cache, args.criteria, args.workers, args.candidates)
        finally:
            cache.flush()  # every key kept: a share never prunes another machine's or another row's results
        print(f"share: {asked} result(s) for {len(set(args.criteria))} row(s); cache written, nothing pruned; no record "
              "written", flush=True)
        return exit_status()
    text = build(cache, args.workers)
    cache.save()
    RECORD.write_text(text, encoding="utf-8")
    print(f"wrote {RECORD.relative_to(REPO) if RECORD.is_relative_to(REPO) else RECORD}")
    return exit_status()


if __name__ == "__main__":
    raise SystemExit(main())
