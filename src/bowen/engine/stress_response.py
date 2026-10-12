"""Stress moves functioning, up or down; and a stressor lasts as long as it is declared to (spec revision 18).

Purpose: keep an exogenous stressor loading its target for its declared duration; move each person's functioning by
         what they do under load — up when they cope (the self-directed channel, or holding back), down when they
         avoid or soothe — scaled by their basic level; and let that movement settle back slowly when they are calm.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.5e, #M1.F.6, #M7.A.1
Tests:   tests/bowen/test_stress_response.py

The owner, 2026-10-11: "stressor induced anxiety 'forces' one to function up. E.g. job loss leads to responsible
spending and serious job search OR it can lead to avoidance behaviour and 'soothing', e.g. drinking. This would be
proportional to level of basic self and nature of stressor. Some stressors are short lived — partner mad for 2 turns
vs. no job for 15 turns." (``docs/PROPOSAL — STRESS ON FUNCTIONING, COACHING AND SELF-WORK.md`` §1, approved.)

* **A lasting stressor** (``register_stressors``, ``stressor_load``). An exogenous stressor of duration *d* arrives
  as before; for its next *d − 1* weeks it adds ``stressor_persistence`` × its arrival's size (intensity ÷
  ``intensity_scale`` × the target's steepness) to the target's acute anxiety, at step 1. Logged by name
  (``stressor_load``, an `M6.4` source).
* **The response** (``respond``, after the week's acts). With load = excess acute anxiety at selection:
  - coping — a self-directed act or ``WITHHOLD`` — raises ``functional_level`` by
    ``response_up_gain × load / 100 × basic_level / 100``;
  - avoiding or soothing — a kind in ``config/bowen/stress_response.md`` — lowers it by
    ``response_down_gain × load / 100 × (1 − basic_level / 100)``;
  - any other act leaves it.
  Nothing writes "a higher level copes": which channel a person chooses already rises with level (`M4.D.1a`), and the
  sizes scale with it.
* **Settling** — a person whose load is below ``anxiety_band_low`` sheds ``response_relaxation_rate`` of their
  ``stress_swing`` each week: slowly, and not as a symmetric restoring force (`M7.A.1`). What lasts is what the
  person kept doing.

``functional_level`` stays in [0, 100], and stress never pushes it below ``basic_level − stress_floor_below_basic``
(revision 19, owner decision 2026-10-11): without that floor, soothing lowered functioning, which steepened
reactivity, which brought more soothing, and anxiety ran away in 11 of 20 four-year runs of the Phase C family.
Every form and constant is [I].
"""

from __future__ import annotations

from src.bowen.engine.act import SELF_CHANNEL, WITHHOLD
from src.bowen.engine.contact import steepness
from src.bowen.engine.events import Event, Mechanism
from src.bowen.engine.log_records import DecidedBy, EffectRecord
from src.bowen.engine.objects import SCALE_MAX
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import ActiveStressor, RunState


def register_stressors(state: RunState, events) -> None:
    """Purpose: a stressor declared to last more than one week keeps weighing on its targets after it arrives.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.6 (revision 18)
    Tests:   tests/bowen/test_stress_response.py::test_m1f6_a_long_stressor_weighs_for_its_duration
    """
    for event in events:
        if event.mechanism is Mechanism.EXOGENOUS_STRESSOR and event.duration > 1:
            for target in event.targets:
                state.stressors.append(ActiveStressor(event.id, target, event.intensity, event.timestamp + 1,
                                                      event.timestamp + event.duration - 1))


def stressor_load(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 1 — each lasting stressor loads its target this week; spent ones are dropped.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.6 (revision 18), #M6.4
    Tests:   tests/bowen/test_stress_response.py::test_m1f6_a_long_stressor_weighs_for_its_duration
    """
    state.stressors = [s for s in state.stressors if s.last_tick >= state.tick]
    records = []
    for stressor in sorted((s for s in state.stressors if s.first_tick <= state.tick), key=lambda s: (str(s.event_id), s.target.value)):
        person = state.people[stressor.target]
        if not person.alive:
            continue
        load = params.stressor_persistence * stressor.intensity / params.intensity_scale * steepness(person, params)
        if load:
            person.acute_anxiety += load
            records.append(EffectRecord(state.tick, "stressor_load", stressor.event_id,
                                        acute_anxiety=((person.id, load),)))
    return records


def respond(state: RunState, selections, load: dict, params: EngineParams) -> list[EffectRecord]:
    """Purpose: what a person does under load moves their functioning; calm, it settles back slowly.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.5e (revision 18), #M7.A.1
    Tests:   tests/bowen/test_stress_response.py::test_m1a5e_coping_raises_functioning_and_soothing_lowers_it
    """
    # A fallback hold (nothing legal, or no finite score) is not a choice, so it is not coping (review 2026-10-11).
    chose = {s.actor: s.kind for s in selections if s.decided_by is not DecidedBy.FALLBACK}
    rows = []
    for pid in sorted(state.people):
        person = state.people[pid]
        if not person.alive:
            continue
        carried, kind = load.get(pid, 0.0), chose.get(pid)
        level = person.basic_level / SCALE_MAX
        if carried > params.anxiety_band_low:
            coped = kind == WITHHOLD or (kind is not None and state.kinds.channels.get(kind) == SELF_CHANNEL)
            if coped:
                change = params.response_up_gain * carried / SCALE_MAX * level
            elif kind in state.kinds.soothing:
                change = -params.response_down_gain * carried / SCALE_MAX * (1.0 - level)
            else:
                change = 0.0
        else:
            change = -params.response_relaxation_rate * person.stress_swing
        relaxing = carried <= params.anxiety_band_low
        before = person.functional_level
        if change < 0:  # revision 19: stress pushes functioning down to a floor below basic level, and no further
            floor = max(0.0, person.basic_level - params.stress_floor_below_basic)
            change = max(change, min(0.0, floor - before))
        person.functional_level = min(SCALE_MAX, max(0.0, before + change))
        moved = person.functional_level - before
        # Calm, the swing decays by its own share whether or not a bound let functioning move: a clamped relaxation
        # must not leave an amount behind to take back later from an unrelated gain (review 2026-10-11).
        person.stress_swing = person.stress_swing * (1.0 - params.response_relaxation_rate) if relaxing \
            else person.stress_swing + moved
        if moved:
            rows.append((pid, "functional_level", moved))
    return [EffectRecord(state.tick, "stress_response", None, people=tuple(rows))] if rows else []
