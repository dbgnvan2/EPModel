"""The fast tick, in M3.D.1's order, and the slow-tick hook.

Purpose: run one week of the model as M3.D.1's nine steps, assert the
         invariants, fire the slow tick every 52 fast ticks, and emit the week's
         log records through the caller's emitter.
Spec:    docs/bowen_agent_model_spec_v2.md#M3.A.1, #M3.B.1, #M3.D.1, #M3.D.2, #M3.D.3, #M4.G.2, #M16.B.1
Tests:   tests/bowen/test_tick.py

The nine steps, and where each lives:

1. standing load ............ ``standing_load.apply_standing_load``
2. deliver .................. scheduled inputs enter (``act.inject``), structural
                              events apply (``event_effects``), the tick's batch is
                              released and recorded
3. perceive ................. ``appraise.perceive``
4. appraise ................. ``appraise.apply_appraisal``; delivered cutoffs
5. involvement .............. ``recompute.recompute_involvement``
6. triangles ................ ``recompute.recompute_triangles``
7. select ................... the activation component names who selects; the
                              source supplies their selections (a script, in Phase B)
8. act ...................... ``act.act``
9. consolidate .............. ``symptoms.accumulate_symptoms`` (the integrator and onset,
                              M4.C.3, M7.D.1); ``reactive.update_attention_state``
                              (investment, M1.B.8; the detectors, M1.A.19);
                              ``consolidate.consolidate``; then the invariants assert

The slow tick (M3.B.1) fires after the 52nd, 104th, … fast tick. In Phase B it
does nothing but record that it fired; its contents are Phase D.
"""

from __future__ import annotations

from typing import Protocol

from src.bowen.engine.act import Selection, act, inject, record_deliveries
from src.bowen.engine.activation import SynchronousActivation
from src.bowen.engine.appraise import apply_appraisal, perceive
from src.bowen.engine.consolidate import consolidate
from src.bowen.engine.event_effects import STRUCTURAL, apply_delivered_cutoffs, apply_structural_event
from src.bowen.engine.events import Event
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.invariants import assert_invariants, snapshot
from src.bowen.engine.log_records import EffectRecord, Emitter, TickRecord
from src.bowen.engine.params import EngineParams
from src.bowen.engine.reactive import update_attention_state
from src.bowen.engine.recompute import recompute_involvement, recompute_triangles
from src.bowen.engine.standing_load import apply_standing_load
from src.bowen.engine.state import RunState
from src.bowen.engine.symptoms import accumulate_symptoms
from src.bowen.engine.visibility import HouseholdConductanceVisibility

STEPS = (
    "standing_load", "deliver", "perceive", "appraise", "involvement",
    "triangles", "select", "act", "consolidate",
)


class Source(Protocol):
    """Where a tick's inputs come from. Phase B's is ``ScriptedSource``; Phase C adds the policy."""

    def scheduled(self, tick: int) -> tuple[Event, ...]:
        """Exogenous and structural events whose tick this is."""

    def selections(self, tick: int, active: tuple[PersonId, ...], state: RunState) -> tuple[Selection, ...]:
        """One selection at most per active person."""


class SelectionFromInactivePerson(RuntimeError):
    """The source selected for someone activation did not name."""


def run_tick(
    state: RunState,
    source: Source,
    params: EngineParams,
    visibility: HouseholdConductanceVisibility,
    activation: SynchronousActivation,
    emitter: Emitter,
) -> tuple[str, ...]:
    """Purpose: run one fast tick in M3.D.1's order and return the steps run.
    Spec:    docs/bowen_agent_model_spec_v2.md#M3.D.1, #M3.D.2, #M3.D.3, #M4.G.2
    Tests:   tests/bowen/test_tick.py::test_m3d1_steps_run_in_spec_order
    """
    steps: list[str] = []
    records: list = [TickRecord(state.tick)]
    before = snapshot(state)

    # 1 — standing load, before anything is delivered (M3.D.2, M6.I.8).
    effects, loaded = apply_standing_load(state, params)
    records += effects
    steps.append("standing_load")

    # 2 — deliver.
    for event in sorted(source.scheduled(state.tick), key=lambda e: e.id):
        if event.timestamp != state.tick:
            raise ValueError(f"source scheduled {event.id} for tick {state.tick}")
        records += inject(state, event, visibility)
        if event.mechanism in STRUCTURAL:
            records += apply_structural_event(state, state.store.event(event.id), params)
    batch = state.queue.release(state.tick)
    records += record_deliveries(state, batch)
    steps.append("deliver")

    # 3 — perceive.
    perceived = perceive(state, batch)
    steps.append("perceive")

    # 4 — appraise.
    effects, attended = apply_appraisal(state, perceived, params)
    records += effects
    records += apply_delivered_cutoffs(state, batch, params)
    steps.append("appraise")

    # 5, 6 — before selection, because gates and propensities read them (M3.D.3).
    recompute_involvement(state)
    steps.append("involvement")
    records += recompute_triangles(state, params)
    steps.append("triangles")

    # 7 — select.
    active = activation.active(state.tick, state.people.values())
    selections = source.selections(state.tick, active, state)
    actors = [s.actor for s in selections]
    if not set(actors) <= set(active):
        raise SelectionFromInactivePerson(f"selections for {sorted(set(actors) - set(active))}")
    if len(actors) != len(set(actors)):
        raise ValueError("a person selected more than once in one tick")
    steps.append("select")

    # 8 — act.
    for selection in sorted(selections, key=lambda s: (s.actor, s.index)):
        records += act(state, selection, visibility)
    steps.append("act")

    # 9 — consolidate, then the invariants assert (M4.G.2). The integrator reads this
    # tick's time above the floor before decay (M4.C.3a).
    records += accumulate_symptoms(state, params, visibility)
    records += update_attention_state(state, attended, params)
    records += consolidate(state, params)
    steps.append("consolidate")
    records.append(assert_invariants(state, before, loaded, tuple(steps), params.invariant_tolerance))

    if (state.tick + 1) % params.slow_tick_fast_ticks == 0:
        records.append(EffectRecord(state.tick, "slow_tick", None))

    for record in records:
        emitter.emit(record)
    state.tick += 1
    return tuple(steps)


def run(
    state: RunState,
    source: Source,
    params: EngineParams,
    visibility: HouseholdConductanceVisibility,
    activation: SynchronousActivation,
    emitter: Emitter,
    ticks: int,
) -> RunState:
    """Purpose: run ``ticks`` fast ticks from the state's current tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M3.A.1
    Tests:   tests/bowen/test_tick.py::test_m3a1_one_record_per_fast_tick
    """
    if ticks < 0:
        raise ValueError("ticks is non-negative")
    for _ in range(ticks):
        run_tick(state, source, params, visibility, activation, emitter)
    return state
