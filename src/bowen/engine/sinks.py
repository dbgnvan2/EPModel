"""Sink allocation — where the family's undifferentiation is absorbed (Phase C step 10).

Purpose: reallocate the undifferentiation budget among the three sinks and the overflow
         each week from what the family did, conserving the budget, and lower the
         budget only on a completed differentiating exchange.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.1, #M1.D.2, #M1.D.3, #M6.I.1, #M6.I.2, #M5.D.7
Tests:   tests/bowen/test_sinks.py

Every form is the project's, graded [I]. Each sink is weighted by the acts that express it
over the last ``sink_window`` weeks, counted from the event store:

* **marital conflict** — `CONFLICT` between the spouses;
* **spouse dysfunction** — `OVERFUNCTION` and `UNDERFUNCTION` between the spouses (the
  functioning reciprocity, `M1.B.5`);
* **child projection** — `OVERFUNCTION`, `PURSUE` and `TRIANGLE` from a spouse to a child
  of the nuclear household;
* **overflow** (`M1.D.3`) — `CONFLICT` on a nuclear adult's family-of-origin tie: its
  destination is conflict with the families of origin, not distance.

Distance is not a sink (`M1.D.2`, `M6.I.2`): it binds anxiety into ties (`M1.D.2a`) outside the
budget. Each week every allocation moves ``sink_rate`` of the way to its share of the whole
budget, in proportion to the weights; with no weight anywhere nothing moves. The shares sum
to the budget, so the allocations never exceed it and their total moves only toward it
(`M6.I.1`). A completed exchange (`M5.D.7`) lowers the budget by
``exchange_budget_reduction`` — the one sink of the budget, logged — and every allocation is
scaled down so their total stays within it.
"""

from __future__ import annotations

from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import Sink
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState

OVERFLOW = "overflow"
CONFLICT, OVERFUNCTION, UNDERFUNCTION, PURSUE, TRIANGLE = (
    "CONFLICT", "OVERFUNCTION", "UNDERFUNCTION", "PURSUE", "TRIANGLE",
)


def _on(event, tie) -> bool:
    a, b = tie.members()
    return event.sender in (a, b) and ({a, b} - {event.sender}) <= set(event.targets)


def sink_weights(state: RunState, params: EngineParams) -> dict[str, float]:
    """Purpose: how strongly the family's recent acts express each sink and the overflow.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.1, #M1.D.3
    Tests:   tests/bowen/test_sinks.py::test_m1d1_marital_conflict_draws_the_budget_to_its_sink
    """
    family = state.family
    weights = {Sink.MARITAL_CONFLICT.value: 0.0, Sink.SPOUSE_DYSFUNCTION.value: 0.0,
               Sink.CHILD_PROJECTION.value: 0.0, OVERFLOW: 0.0}
    spouses: set[PersonId] = set(family.marital_tie.members()) if family.marital_tie else set()
    for event in state.store.events():
        if event.mechanism is not Mechanism.MOVE or event.sender is None:
            continue
        if not state.tick - params.sink_window < event.timestamp <= state.tick:
            continue
        if family.marital_tie and _on(event, family.marital_tie):
            if event.kind == CONFLICT:
                weights[Sink.MARITAL_CONFLICT.value] += 1
            elif event.kind in (OVERFUNCTION, UNDERFUNCTION):
                weights[Sink.SPOUSE_DYSFUNCTION.value] += 1
        if event.sender in spouses and event.kind in (OVERFUNCTION, PURSUE, TRIANGLE):
            if any(_on(event, t) for t in family.parent_child_ties):
                weights[Sink.CHILD_PROJECTION.value] += 1
        if event.kind == CONFLICT and any(_on(event, t) for t in family.origin_ties):
            weights[OVERFLOW] += 1
    return weights


def allocate_sinks(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 9 — move each allocation toward its share of the budget; the budget is conserved.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.1, #M1.D.3, #M6.I.1
    Tests:   tests/bowen/test_sinks.py::test_m6i1_allocation_conserves_the_budget
    """
    family = state.family
    weights = sink_weights(state, params)
    total = sum(weights.values())
    if total <= 0 or family.undifferentiation_budget <= 0:
        return []
    changes = []
    for sink in Sink:
        target = family.undifferentiation_budget * weights[sink.value] / total
        delta = params.sink_rate * (target - family.sink_allocations[sink])
        family.sink_allocations[sink] += delta
        if delta:
            changes.append((f"sink:{sink.value}", delta))
    target = family.undifferentiation_budget * weights[OVERFLOW] / total
    delta = params.sink_rate * (target - family.overflow)
    family.overflow += delta
    if delta:
        changes.append((f"sink:{OVERFLOW}", delta))
    return [EffectRecord(state.tick, "sink_allocation", None, sinks=tuple(changes))] if changes else []


def reduce_budget(state: RunState, params: EngineParams, cause) -> EffectRecord | None:
    """Purpose: a completed differentiating exchange lowers the budget — its one logged sink.
    Spec:    docs/bowen_agent_model_spec_v2.md#M6.I.1, #M5.D.7
    Tests:   tests/bowen/test_sinks.py::test_m6i1_only_a_completed_exchange_reduces_the_budget
    """
    family = state.family
    reduction = min(params.exchange_budget_reduction, family.undifferentiation_budget)
    if reduction <= 0:
        return None
    family.undifferentiation_budget -= reduction
    allocated = sum(family.sink_allocations.values()) + family.overflow
    if allocated > family.undifferentiation_budget > 0 or family.undifferentiation_budget == 0:
        scale = family.undifferentiation_budget / allocated if allocated else 0.0
        for sink in Sink:
            family.sink_allocations[sink] *= scale
        family.overflow *= scale
    return EffectRecord(state.tick, "differentiating_exchange", cause, sinks=(("undifferentiation_budget", -reduction),))
