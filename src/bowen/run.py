"""Run the Phase B model from the repository's configuration.

Purpose: the composition root — load config, assemble, emit the header, run,
         and (if asked) persist the log. The engine itself never touches files.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.A.1, #M16.B.1, #M3.D.4
Tests:   tests/bowen/test_log.py, tests/bowen/test_determinism.py

    python3 -m src.bowen.run --seed 7 --log runs/phase_b.jsonl --trace runs/phase_b_trace.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import CollectingEmitter, Emitter, Record, Tee
from src.bowen.engine.state import RunState
from src.bowen.engine.tick import run
from src.bowen.io.load import (
    load_constants, load_event_kinds, load_family, load_frozen_constants, load_script, load_spec_revision,
)
from src.bowen.io.sinks import JsonlFileSink
from src.bowen.render.trace import render
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.header import build_header
from src.bowen.scenario.scripted_source import ScriptedSource


@dataclass
class RunResult:
    state: RunState
    records: list[Record]


def run_phase_b(seed: int, *, script: ScriptedSource | None = None, extra: Emitter | None = None) -> RunResult:
    """Purpose: one full Phase B run; records are collected and also sent to ``extra`` if given.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.A.1, #M16.B.1
    Tests:   tests/bowen/test_determinism.py::test_m11d5_same_seed_same_log
    """
    constants, kinds, family = load_constants(), load_event_kinds(), load_family()
    script = script or load_script(kinds=kinds, family=family)
    parts = assemble(constants, kinds, family, script, seed=seed)
    collector = CollectingEmitter()
    emitter = Tee(collector, extra) if extra is not None else collector
    emitter.emit(
        build_header(
            seed=seed, spec_revision=load_spec_revision(), constants=constants, frozen=load_frozen_constants(),
            kinds=kinds, family=family, script=script, activation=parts.activation, visibility=parts.visibility,
        )
    )
    run(parts.state, parts.source, parts.params, parts.visibility, parts.activation, emitter, ticks=script.ticks)
    return RunResult(parts.state, collector.records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase B model and write its log.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", type=Path, required=True, help="where to write the JSONL log")
    parser.add_argument("--trace", type=Path, help="where to write the rendered markdown trace")
    parser.add_argument("--view", help="render one person's view (a person id)")
    args = parser.parse_args(argv)
    names = load_family().display_names
    if args.view and PersonId(args.view) not in names:
        parser.error(f"--view {args.view!r} is not in the family: {sorted(p.value for p in names)}")
    with JsonlFileSink(args.log) as sink:
        result = run_phase_b(args.seed, extra=sink)
    print(f"{len(result.records)} records written to {args.log}")
    if args.trace:
        view = PersonId(args.view) if args.view else None
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(render(result.records, names, view=view), encoding="utf-8")
        print(f"trace written to {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
