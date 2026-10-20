"""Render one 104-week run of the Phase C family for the owner's review (plan §9).

Purpose: the trace plan §9 asks a person to read — whether learned behaviour reads as Bowen's account, which no
         criterion can test. The run is `M11.C.1`'s baseline arm: the Phase C family at its own levels, under the
         policy and the learner, with the declared spell (`M11.3`), for `M11.C.1`'s 104 weeks.
Spec:    docs/implementation_plan_phase_c.md#9, docs/bowen_agent_model_spec_v2.md#M16.C.1
Tests:   tests/bowen/test_render.py::test_m16c1_phase_c_trace_renders,
         tests/bowen/test_render.py::test_m16c1_phase_c_chart_renders

    python3 tools/phase_c_trace.py [--seed 7] [--view nadia]

Writes ``docs/review/phase_c_trace_seed<seed>[_<view>].md`` and, for the whole family (no ``--view``), the same run's
week-by-week chart of each person's anxiety and functioning, ``docs/review/phase_c_chart_seed<seed>.html``
(``tools/run_chart.py``). Seed 7 matches the Phase B review traces; it is not chosen for what it shows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def records(seed: int, after_tick=None, family=None) -> list:
    """The run's records, opening with its header (`M16.A.1`). ``after_tick(state, records, params)``, when given,
    is called after every week; `tools/famdiag_export.py` reads its patterns there. It observes and changes nothing.
    ``family``, when given, is the Phase C family already loaded, so a caller loads it once (stage 1 rider (v))."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.engine.log_records import CollectingEmitter
    from src.bowen.engine.tick import run_tick
    from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_frozen_constants, load_policy_rules, load_spec_revision
    from src.bowen.scenario.assemble import assemble
    from src.bowen.scenario.header import build_header
    from src.bowen.scenario.policy_source import PolicySource
    from src.bowen.scenario.scripted_source import ScriptedSource

    weeks = criteria.SETTINGS["M11.C.1"]["weeks"]
    constants, kinds = load_constants(), load_event_kinds()
    family = family if family is not None else load_family(criteria.FAMILIES["phase_c"])
    script = ScriptedSource("phase_c-trace", weeks, tuple(criteria.spell("phase_c", weeks)), ())
    parts = assemble(constants, kinds, family, script, seed=seed)
    criteria.lowered(criteria.SETTINGS["M11.C.1"]["level_baseline"])(parts.state)  # arm_c1's baseline setup, exactly
    source = PolicySource(script, parts.params, load_policy_rules())
    collector = CollectingEmitter()
    collector.emit(build_header(
        seed=seed, spec_revision=load_spec_revision(), constants=constants, frozen=load_frozen_constants(),
        kinds=kinds, family=family, script=script, activation=parts.activation, visibility=parts.visibility,
    ))
    for _ in range(weeks):
        run_tick(parts.state, source, parts.params, parts.visibility, parts.activation, collector)
        if after_tick is not None:
            after_tick(parts.state, collector.records, parts.params)
    return collector.records


def trace_text(seed: int, view=None) -> str:
    """The rendered trace. The renderer reads from the records that the policy chose the acts (Phase D plan B2)."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.io.load import load_family
    from src.bowen.render.trace import render

    family = load_family(criteria.FAMILIES["phase_c"])
    names = family.display_names
    if view is not None and view not in names:
        raise SystemExit(f"--view {view.value!r} is not in the Phase C family: {sorted(p.value for p in names)}")
    return render(records(seed, family=family), names, view=view)


def outputs(seed: int) -> tuple[str, str]:
    """The trace and the chart, rendered from one run so they cannot describe different runs."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.io.load import load_family
    from tools.run_chart import WeeklySeries, render_chart
    from src.bowen.render.trace import render

    family = load_family(criteria.FAMILIES["phase_c"])
    series = WeeklySeries()
    run = records(seed, after_tick=series.observe, family=family)
    return render(run, family.display_names), render_chart(run, series, family)


def main() -> int:
    from src.bowen.engine.identifiers import PersonId

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--view", help="one person's view (a person id)")
    args = parser.parse_args()
    view = PersonId(args.view) if args.view else None
    review = REPO / "docs" / "review"
    if view is not None:
        written = {review / f"phase_c_trace_seed{args.seed}_{args.view}.md": trace_text(args.seed, view)}
    else:
        trace, chart = outputs(args.seed)
        written = {review / f"phase_c_trace_seed{args.seed}.md": trace,
                   review / f"phase_c_chart_seed{args.seed}.html": chart}
    for out, text in written.items():
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
