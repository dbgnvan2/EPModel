"""T-259: revision 67's registered predictions P1–P5 (`docs/proposal_conviction_and_coach_sessions.md` §3.4).

Purpose: run each prediction's two arms once, before any constant is tuned, and print each as met or not met. The
         fixture is T-259's (`tests/bowen/test_iposition.py::attempts_over_seeds`): the triad, F given perspective
         and taking positions with M, 192 seeds of 100 weeks. P2–P4 are premises (`M11.5`): a pass confirms the code
         and is never a finding. P1 and P5 are measurements.
Spec:    docs/bowen_agent_model_spec_v2.md#M5.D.3b, #M5.D.3
Tests:   none; a probe, run on Mac Mini1 (`python3 tools/t259_conviction_predictions.py --workers 8`)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LOAD = 10.0  # P4's scripted load: acute anxiety added to the mover each week, [I]

ARMS = {
    "level60": dict(level=60.0),
    "level30": dict(level=30.0),
    "bond80": dict(level=60.0, bond=80.0),
    "bond20": dict(level=60.0, bond=20.0),
    "load": dict(level=60.0, load=LOAD),
    "level30_no_lines": dict(level=30.0, line_tendency_max=0.0),
}


def run_seed(arm: str, seed: int, weeks: int) -> tuple[str, ...]:
    from src.bowen.engine.act import Selection
    from src.bowen.engine.identifiers import PersonId, TieId
    from src.bowen.engine.log_records import EffectRecord
    from src.bowen.engine.tick import run_tick
    from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_policy_rules
    from src.bowen.scenario.assemble import assemble
    from src.bowen.scenario.policy_source import PolicySource
    from src.bowen.scenario.scripted_source import ScriptedSource

    spec = ARMS[arm]
    constants, family = load_constants(), load_family(REPO / "config" / "bowen" / "fixtures" / "triad.md")
    mover, target = PersonId("f"), PersonId("m")
    events = ScriptedSource("events", weeks, (), ())
    parts = assemble(constants, load_event_kinds(), family, events, seed=seed)
    params = parts.params
    if "line_tendency_max" in spec:
        params = dataclasses.replace(params, line_tendency_max=spec["line_tendency_max"])
    person = parts.state.people[mover]
    person.systems_perspective = 1.0
    person.basic_level = person.functional_level = spec["level"]
    if "bond" in spec:
        parts.state.ties[TieId.of(mover, target)].bond_energy = spec["bond"]
    policy = PolicySource(events, params, load_policy_rules())

    class Source:
        def scheduled(self, tick):
            return policy.scheduled(tick)

        def selections(self, tick, active, state):
            chosen = [s for s in policy.selections(tick, active, state) if s.actor != mover or mover not in active]
            if mover in active and state.people[mover].iposition_state is None:
                chosen.append(Selection(actor=mover, kind="I-POSITION", targets=(target,), intensity=100.0))
            elif mover in active:
                chosen += list(policy.selections(tick, (mover,), state))
            return tuple(chosen)

    out = []
    sink = type("Sink", (), {"emit": lambda self, r: out.append(r)})()
    for _ in range(weeks):
        if spec.get("load"):
            parts.state.people[mover].acute_anxiety += spec["load"]
        run_tick(parts.state, Source(), params, parts.visibility, parts.activation, sink)
    return tuple(f for r in out if isinstance(r, EffectRecord) and r.mechanism == "iposition"
                 for p, f, _ in r.people if p == mover and f.startswith("iposition:"))


def tally(runs: list[tuple[str, ...]]) -> dict:
    held = aborted = completed = 0
    for marks in runs:
        at_opposition = False
        for mark in marks:
            if mark == "iposition:enter:OPPOSITION":
                at_opposition = True
            elif at_opposition and mark == "iposition:enter:HOLD":
                held, at_opposition = held + 1, False
            elif at_opposition and mark.startswith("iposition:end:"):
                aborted += mark == "iposition:end:aborted"
                at_opposition = False
            if mark == "iposition:end:completed":
                completed += 1
    share = held / (held + aborted) if held + aborted else None
    return {"held": held, "aborted": aborted, "hold_share": share, "completed": completed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=192)
    parser.add_argument("--weeks", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    jobs = [(arm, seed) for arm in ARMS for seed in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        marks = list(pool.map(run_seed, [a for a, _ in jobs], [s for _, s in jobs], [args.weeks] * len(jobs)))
    by_arm = {arm: tally([m for (a, _), m in zip(jobs, marks) if a == arm]) for arm in ARMS}

    def more(a, b, key):
        """True or False, or None when an arm has no value (no sequence reached the first opposition)."""
        x, y = by_arm[a][key], by_arm[b][key]
        return None if x is None or y is None else x > y

    verdicts = {
        "P1": ("measurement", by_arm["level60"]["completed"] > 0,
               f"revision 67 completes {by_arm['level60']['completed']} (revision 66: 0)"),
        "P2": ("premise", more("level60", "level30", "hold_share"), "hold share, level 60 against 30"),
        "P3": ("premise", more("bond20", "bond80", "hold_share"), "hold share, bond 20 against 80"),
        "P4": ("premise", more("level60", "load", "hold_share"), "hold share, no load against load"),
        "P5": ("measurement", by_arm["level30"]["completed"] > 0 and bool(more("level30", "level30_no_lines", "completed")),
               "completions at level 30, the declared line tendency against 0"),
    }
    result = {"seeds": args.seeds, "weeks": args.weeks, "arms": by_arm,
              "verdicts": {k: {"class": c, "met": "not measurable" if m is None else m, "what": w}
                           for k, (c, m, w) in verdicts.items()}}
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
