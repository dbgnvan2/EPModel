"""Render one 104-week run of the Phase C family for the owner's review (plan §9).

Purpose: the trace plan §9 asks a person to read — whether learned behaviour reads as Bowen's account, which no
         criterion can test. The run is `M11.C.1`'s baseline arm: the Phase C family at its own levels, under the
         policy and the learner, with the declared spell (`M11.3`), for `M11.C.1`'s 104 weeks.
Spec:    docs/implementation_plan_phase_c.md#9, docs/bowen_agent_model_spec_v2.md#M16.C.1
Tests:   tests/bowen/test_render.py::test_m16c1_phase_c_trace_renders

    python3 tools/phase_c_trace.py [--seed 7] [--view nadia]

Writes ``docs/review/phase_c_trace_seed<seed>[_<view>].md``. Seed 7 matches the Phase B review traces; it is not
chosen for what it shows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def records(seed: int) -> list:
    """The run's records, opening with its header (`M16.A.1`)."""
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
    family = load_family(criteria.FAMILIES["phase_c"])
    script = ScriptedSource("phase_c-trace", weeks, tuple(criteria.spell("phase_c", weeks)), ())
    parts = assemble(constants, kinds, family, script, seed=seed)
    source = PolicySource(script, parts.params, load_policy_rules())
    collector = CollectingEmitter()
    collector.emit(build_header(
        seed=seed, spec_revision=load_spec_revision(), constants=constants, frozen=load_frozen_constants(),
        kinds=kinds, family=family, script=script, activation=parts.activation, visibility=parts.visibility,
    ))
    for _ in range(weeks):
        run_tick(parts.state, source, parts.params, parts.visibility, parts.activation, collector)
    return collector.records


PHASE_B_FRAMING = "This run is scripted: no one in it\n> chose anything (Phase B)."
PHASE_C_FRAMING = ("This run is not scripted: each person's outcome is\n> selected by the policy, and the automatic "
                   "channel learns from felt relief (Phase C).")


def trace_text(seed: int, view=None) -> str:
    """The rendered trace, with the renderer's Phase B framing sentence replaced (it says no one chose anything,
    which is false here). The renderer is in the engine hash, so its framing is fixed with the next engine change
    (``TODO.md``); until then this replacement fails loudly if the sentence moves."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.io.load import load_family
    from src.bowen.render.trace import render

    names = load_family(criteria.FAMILIES["phase_c"]).display_names
    text = render(records(seed), names, view=view)
    if PHASE_B_FRAMING not in text:
        raise SystemExit("the renderer's framing changed: update PHASE_B_FRAMING in tools/phase_c_trace.py")
    return text.replace(PHASE_B_FRAMING, PHASE_C_FRAMING)


def main() -> int:
    from src.bowen.engine.identifiers import PersonId

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--view", help="one person's view (a person id)")
    args = parser.parse_args()
    view = PersonId(args.view) if args.view else None
    out = REPO / "docs" / "review" / f"phase_c_trace_seed{args.seed}{'_' + args.view if args.view else ''}.md"
    out.write_text(trace_text(args.seed, view), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
