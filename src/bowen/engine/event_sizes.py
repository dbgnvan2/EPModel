"""Event sizes from the Holmes–Rahe scale (version 2, spec revision 36).

Purpose: size the life course's stressors from one table (``config/bowen/event_sizes.md``) instead of a constant per
         event. The scale's life change units (LCU) order the events; ``lcu_scale`` (6.9 intensity units per LCU,
         the owner's anchor: a job loss of 47 LCU = Phase B's 324-unit spell) turns them into intensity units; the
         tie's bond energy scales a loss and an illness in the family (``reach``); and an event that lasts arrives
         at an intensity that delivers the whole size over its weeks under `M1.F.6`'s persistence
         (``total = arrival × (1 + stressor_persistence × (weeks − 1))``).
Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.6a, #M6.3g, #M7.C.1l, #M7.D.1a
Tests:   tests/bowen/test_event_sizes.py

Every number here is ``[I]`` or the owner's; the scale orders events and the corpus supports no magnitude
(`docs/PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md`). The sizing changes what an event weighs, never which
act a person chooses: no pattern is written (`model_explainer.md` §19).
"""

from __future__ import annotations

from src.bowen.engine.events import EventKinds
from src.bowen.engine.params import EngineParams


def sized_intensity(kinds: EventKinds, params: EngineParams, kind: str, who: str, *, reach: float = 1.0,
                    weeks: int | None = None) -> tuple[float, int]:
    """Purpose: the arrival intensity and duration of a stressor of ``kind`` for ``who`` (self, spouse, relative or
    tie), at ``reach`` (1, or the tie's bond energy share), over ``weeks`` (the table's unless given).
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.6a
    Tests:   tests/bowen/test_event_sizes.py::test_m1f6a_a_lasting_event_delivers_its_whole_size_over_its_weeks
    """
    if kinds.sizes is None:
        raise ValueError("event sizes are not loaded (config/bowen/event_sizes.md)")
    size = kinds.sizes.get((kind, who))
    if size is None:  # not in the table, or there with no size (lost income, `M6.3h`): it keeps its own constant
        raise ValueError(f"{kind} for {who} has no size in event_sizes.md")
    weeks = weeks if weeks is not None else (size.weeks or 1)
    if weeks < 1 or reach < 0:
        raise ValueError("weeks is at least 1 and reach non-negative")
    total = params.lcu_scale * size.lcu * reach
    return total / (1.0 + params.stressor_persistence * (weeks - 1)), weeks
