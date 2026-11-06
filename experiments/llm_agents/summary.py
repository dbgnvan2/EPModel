"""A run's summary: the readouts the report compares, computed by the model's own readout functions.

Purpose: from a finished run, take the `M11.G.1` evaluation's numeric readouts over its window, the share of
         subject-weeks holding each declared pattern (read every week by ``tie_readings``, ``fixed_triangle`` and
         ``current_patterns``), and the family members' act frequencies by mechanics and by the label shown (§6).
Spec:    docs/plan_v3_llm_agent_experiment.md#6
Tests:   tests/experiments/llm_agents/test_report.py::test_lx16_report_is_current

The same functions read every arm, A0 included. A summary carries the engine hash it was computed on; the report
refuses a summary computed on another.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from experiments.llm_agents import files
from experiments.llm_agents.cache import canonical_json, text_hash
from src.bowen.engine.act import WITHHOLD
from src.bowen.engine.events import Mechanism
from src.bowen.engine.log_records import EmittedRecord, SelectionRecord
from src.bowen.engine.objects import Role, TieState
from src.bowen.io.load import load_constants
from src.bowen.readouts.evaluation import evaluate
from src.bowen.readouts.patterns import (
    Reading, ReadoutParams, current_patterns, fixed_triangle, last_projection_act, tie_readings,
)
from src.bowen.render.trace import framing


class PatternWatch:
    """Reads the declared patterns after every week, as ``tools/famdiag_export.py`` does at each block's end."""

    def __init__(self, run):
        self.rp = ReadoutParams.from_constants(load_constants())
        fam = run.family
        self.family = fam
        # A tie declared cut off holds that pattern from the start, until a new one emerges (owner D2).
        self.held = {t: Reading("cutoff", t, 0, 0) for t, tie in fam.ties.items() if tie.tie_state is TieState.CUT_OFF}
        self.subject_weeks = Counter()
        self.weeks = 0

    def after_week(self, run) -> None:
        state, week = run.state, run.state.tick - 1
        readings = [r for tie in state.ties
                    for r in tie_readings(state, tie, run.params, self.rp, self.family.parent_on_tie.get(tie), now=week)]
        readings += [r for tri in state.triangles if (r := fixed_triangle(run.records, tri, self.rp)) is not None]
        evidence = {tie: last for tie, parent in self.family.parent_on_tie.items()
                    if (last := last_projection_act(state, tie, parent, now=week)) is not None}
        self.held = current_patterns(self.held, readings, now=week, projection_last=evidence,
                                     projection_hold=self.rp.projection_hold)
        self.subject_weeks.update(r.pattern for r in self.held.values())
        self.weeks += 1


def _dig(tree: dict, dotted: str):
    for part in dotted.split("."):
        tree = tree[part]
    return tree


def act_counts(run) -> Counter:
    """The family members' acts, from the run's records: each move emitted and each act held back, by kind."""
    members = {p for p, person in run.state.people.items() if person.role is Role.MEMBER}
    counts = Counter()
    for r in run.records:
        if isinstance(r, EmittedRecord) and r.event.mechanism is Mechanism.MOVE and r.event.sender in members:
            counts[r.event.kind] += 1
        elif isinstance(r, SelectionRecord) and r.actor in members and r.withheld:
            counts[f"{WITHHOLD}:{r.withheld}"] += 1
    return counts


def summarize(run, watch: PatternWatch, cfg) -> dict:
    """Purpose: the run's summary for the report; every number in it is read from the run by a model function.
    Spec:    docs/plan_v3_llm_agent_experiment.md#6
    Tests:   tests/experiments/llm_agents/test_report.py::test_lx16_report_is_current
    """
    from experiments.llm_agents.record import engine_hash

    window = min(cfg.integer("evaluation_window_weeks"), run.state.tick)
    ev = evaluate(run.state, run.records, run.params, framing(run.records), window, run.family.generations)
    acts = act_counts(run)
    label_of = run.perms.label_of if run.perms is not None else {}

    def shown(key: str) -> str:
        if key.startswith(f"{WITHHOLD}:"):
            return f"{WITHHOLD}:{label_of.get(key.split(':', 1)[1], key.split(':', 1)[1])}"
        return label_of.get(key, key)

    by_label = Counter()
    for k, n in acts.items():
        by_label[shown(k)] += n
    subjects = len(run.state.ties) + len(run.state.triangles)
    choices = getattr(run.source, "choices", [])
    out = {
        "arm": run.arm, "seed": run.seed, "weeks": run.state.tick, "engine_hash": engine_hash(),
        "evaluation": {k: _dig(ev, k) for k in cfg.listed("evaluation_readouts")},
        "patterns": {"subjects": subjects, "weeks": watch.weeks,
                     "subject_weeks": dict(sorted(watch.subject_weeks.items()))},
        "acts_by_mechanics": dict(sorted(acts.items())),
        "acts_by_label": dict(sorted(by_label.items())),
        "llm_selections": len(choices),
        "llm_fallbacks": sum(1 for c in choices if c.flag == "llm_failed_to_parse"),
        "llm_no_options": sum(1 for c in choices if c.flag == "llm_no_options"),
    }
    out["summary_hash"] = text_hash(canonical_json(out))[:12]
    return out


def write(summary: dict, directory: Path, runs: Path = files.RUNS) -> Path:
    return files.write_text(directory / "summary.json", json.dumps(summary, indent=2) + "\n", runs=runs)
