"""PolicySource — Phase C's Source: scheduled events from a script, selections from the policy.

Purpose: give every active person one outcome a tick from the policy, unless the
         script names a move for them that tick, and pass the script's exogenous and
         structural events through unchanged.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1, #M4.B.2, #M13
Tests:   tests/bowen/test_policy.py::test_m4d1_exactly_one_outcome_per_person_per_tick

The policy sees each person only through ``observe`` (`M4.B.2`). Selections are made
from the state as it stands at step 7, person by person in identifier order; no
person's selection changes another's observation within the tick, because nothing is
emitted until step 8.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from src.bowen.engine.act import Selection
from src.bowen.engine.events import Event
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.observe import observe
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState
from src.bowen.policy.policy import decide
from src.bowen.policy.rules import PolicyRules
from src.bowen.scenario.scripted_source import ScriptedSource


@dataclass(frozen=True)
class PolicySource:
    script: ScriptedSource
    params: EngineParams
    rules: PolicyRules

    @property
    def ticks(self) -> int:
        return self.script.ticks

    def scheduled(self, tick: int) -> tuple[Event, ...]:
        return self.script.scheduled(tick)

    def selections(self, tick: int, active: tuple[PersonId, ...], state: RunState) -> tuple[Selection, ...]:
        scripted = {s.actor: s for s in self.script.selections(tick, active, state)}
        chosen = []
        for pid in sorted(active):
            if pid in scripted:
                chosen.append(scripted[pid])
                continue
            if not state.people[pid].alive:
                continue
            decision = decide(observe(state, pid, self.params), state.kinds, self.params, self.rules, state.draws)
            chosen.append(dataclasses.replace(decision.selection, urge=decision.urge))
        return tuple(chosen)
