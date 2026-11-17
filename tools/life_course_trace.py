"""Render one run of the twelve-person family with the life course, under the policy, for a person to read (T-238).

Purpose: the trace no other command gives: the Phase D family (`config/bowen/family_phase_d.md`, life course on), under
         the policy and the learner, with the declared nodal load on the two parents that `M11.C.6`, `M11.C.10` and
         `M11.C.2` run (`criteria_d.nodal`), rendered by the same trace renderer `tools/phase_c_trace.py` uses. It is the
         declared arm with nothing forced: no cutoff renewed, no tie held open. A run to read, not a finding.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.C.1, #M2.3
Tests:   tests/test_life_course_trace.py::test_m16c1_life_course_trace_renders

    python3 tools/life_course_trace.py [--seed 7] [--years 40] [--view nadia]

Writes ``docs/review/life_course_trace_seed<seed>[_<view>].md``. The default 40 years is `M11.C.2`'s run length.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def records(seed: int, weeks: int, family=None) -> list:
    """The run's records, opening with its header (`M16.A.1`)."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.engine.log_records import CollectingEmitter
    from src.bowen.engine.tick import run_tick
    from src.bowen.ensemble.criteria_d import nodal
    from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_frozen_constants, load_policy_rules, load_spec_revision
    from src.bowen.scenario.assemble import assemble
    from src.bowen.scenario.header import build_header
    from src.bowen.scenario.policy_source import PolicySource
    from src.bowen.scenario.scripted_source import ScriptedSource

    constants, kinds = load_constants(), load_event_kinds()
    family = family if family is not None else load_family(criteria.FAMILIES["phase_d"])
    parents = (criteria.ROLES["phase_d", "parent_1"], criteria.ROLES["phase_d", "parent_2"])
    script = ScriptedSource("phase_d-life-trace", weeks, tuple(nodal(weeks, parents)), ())
    parts = assemble(constants, kinds, family, script, seed=seed)
    source = PolicySource(script, parts.params, load_policy_rules())
    collector = CollectingEmitter()
    collector.emit(build_header(
        seed=seed, spec_revision=load_spec_revision(), constants=constants, frozen=load_frozen_constants(),
        kinds=kinds, family=family, script=script, activation=parts.activation, visibility=parts.visibility,
        policy=source.rules,
    ))
    for _ in range(weeks):
        run_tick(parts.state, source, parts.params, parts.visibility, parts.activation, collector)
    return collector.records


def trace_text(seed: int, weeks: int, view=None) -> str:
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.io.load import load_family
    from src.bowen.render.trace import render

    family = load_family(criteria.FAMILIES["phase_d"])
    if view is not None and view not in family.display_names:
        raise SystemExit(f"--view {view.value!r} is not in the family: {sorted(p.value for p in family.display_names)}")
    return render(records(seed, weeks, family=family), family.display_names, view=view)


def main() -> int:
    from src.bowen.engine.identifiers import PersonId
    from src.bowen.ensemble.criteria_d2 import YEAR

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--years", type=int, default=40)
    parser.add_argument("--view", help="one person's view (a person id)")
    args = parser.parse_args()
    view = PersonId(args.view) if args.view else None
    out = REPO / "docs" / "review" / (f"life_course_trace_seed{args.seed}" + (f"_{args.view}" if view else "") + ".md")
    out.write_text(trace_text(args.seed, args.years * YEAR, view), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
