"""Stress moves functioning, up or down; and a stressor lasts as long as it is declared to.

Purpose: keep an exogenous stressor loading its target for its declared duration; move each person's functioning by
         what they do under load — up when they cope (the self-directed channel, or holding back), down when they
         avoid or soothe — scaled by their basic level; and let that movement settle back slowly when they are calm.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.5e, #M1.A.5f, #M1.F.6, #M7.A.1
Tests:   tests/bowen/test_stress_response.py

* **A lasting stressor** (``register_stressors``, ``stressor_load``). An exogenous stressor of duration *d* arrives
  as before; for its next *d − 1* weeks it adds ``stressor_persistence`` × its arrival's size (intensity ÷
  ``intensity_scale`` × the target's steepness) to the target's acute anxiety, at step 1. Logged by name
  (``stressor_load``, an `M6.4` source).
* **The response** (``respond``, after the week's acts). With load = excess acute anxiety at selection above the
  person's usual level:
  - coping — a self-directed act or ``WITHHOLD`` — raises ``functional_level`` by
    ``response_up_gain × load / 100 × basic_level / 100``;
  - avoiding or soothing — a kind in ``config/bowen/stress_response.md`` — lowers it by
    ``response_down_gain × load / 100 × (1 − basic_level / 100)``;
  - any other act leaves it, except a displacing act (``displace``, below).
  Nothing writes "a higher level copes": which channel a person chooses already rises with level (`M4.D.1a`), and the
  sizes scale with it.
* **Displaced coping** (``displace``, revision 39, `M1.A.5f`). A loaded person's act of a kind listed as *displacing*
  in ``config/bowen/stress_response.md`` (``OVERFUNCTION``) meets their load through its receivers: the coping amount
  moves from the receivers' ``functional_level`` to the sender's, split evenly, on both people's ``stress_swing``,
  conserved. A receiver holding a position toward the sender gives nothing; none is taken below their floor, and the
  sender not above 100.
* **Calm is relative to the person.** Each person keeps a usual level: the
  plain mean of their excess anxiety over the weeks so far for the first ``1 / usual_level_rate`` weeks, then a running
  average over about a year (``update_usual``). The load that drives the response is the excess above it, and a person
  at or below it is calm.
* **Settling** — a calm person sheds ``response_relaxation_rate`` of their ``stress_swing`` each week: slowly, and not
  as a symmetric restoring force (`M7.A.1`). What lasts is what the
  person kept doing. The same calm week sheds the same share of ``adjust_swing``, the functioning the person gained
  when someone else's held stance worked (`M1.E.7`).
* **What a bound holds is on a ledger** (revision 39, F1, `M1.A.5e`). When the floor, or the scale's 0–100, stops
  ``functional_level`` from moving while the swings decay, the amount it stopped goes to ``floor_held``; a calm week
  then sheds ``response_relaxation_rate`` of ``floor_held``, moving ``functional_level`` with it as far as the bound
  lets, and what the bound stops of that stays on the ledger. So ``functional_level − stress_swing − adjust_swing −
  floor_held`` never moves by a bound, and the estimator leaves ``floor_held`` out (`M1.A.4k`).

``functional_level`` stays in [0, 100], and stress never pushes it below ``basic_level − stress_floor_below_basic``.
Every form and constant is [I].
"""

from __future__ import annotations

from src.bowen.engine.act import SELF_CHANNEL, WITHHOLD
from src.bowen.engine.contact import steepness
from src.bowen.engine.events import Mechanism
from src.bowen.engine.log_records import DecidedBy, EffectRecord
from src.bowen.engine.moves import holds_position, stress_floor
from src.bowen.engine.objects import SCALE_MAX, Role
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import ActiveStressor, RunState


def register_stressors(state: RunState, events) -> None:
    """Purpose: a stressor declared to last more than one week keeps weighing on its targets after it arrives.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.6
    Tests:   tests/bowen/test_stress_response.py::test_m1f6_a_long_stressor_weighs_for_its_duration
    """
    for event in events:
        if event.mechanism is Mechanism.EXOGENOUS_STRESSOR and event.duration > 1:
            for target in event.targets:
                state.stressors.append(ActiveStressor(event.id, target, event.intensity, event.timestamp + 1,
                                                      event.timestamp + event.duration - 1))


def stressor_load(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 1 — each lasting stressor loads its target this week; spent ones are dropped.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.6, #M6.4
    Tests:   tests/bowen/test_stress_response.py::test_m1f6_a_long_stressor_weighs_for_its_duration
    """
    state.stressors = [s for s in state.stressors if s.last_tick is None or s.last_tick >= state.tick]
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


def above_usual(person, carried: float) -> float:
    """Purpose: how far this week's excess is above the person's usual level; 0 before they have one.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.5e
    Tests:   tests/bowen/test_stress_response.py::test_m1a5e_calm_is_relative_to_the_persons_usual_level
    """
    return 0.0 if person.usual_excess is None else carried - person.usual_excess


def update_usual(state: RunState, load: dict, params: EngineParams) -> None:
    """Purpose: each living person's usual level follows their excess anxiety, slowly.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.5e
    Tests:   tests/bowen/test_stress_response.py::test_m1a5e_the_usual_level_follows_slowly

    For the first ``1 / usual_level_rate`` weeks it is the plain mean of the weeks so far, so it catches up with the
    family's own level within weeks; after that, a running average weighted ``usual_level_rate`` to the newest week.
    """
    for pid in sorted(state.people):
        person = state.people[pid]
        if not person.alive:
            continue
        person.usual_weeks += 1
        rate = max(params.usual_level_rate, 1.0 / person.usual_weeks)
        usual = person.usual_excess if person.usual_excess is not None else 0.0
        person.usual_excess = usual + rate * (load.get(pid, 0.0) - usual)


def displace(state: RunState, selections, load: dict, params: EngineParams) -> list[tuple]:
    """Purpose: a loaded person's displacing act meets their load through its receivers, conserved.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.5f
    Tests:   tests/bowen/test_stress_response.py::test_m1a5f_s3_1a_displaced_coping_moves_functioning_from_receiver_to_sender,
             tests/bowen/test_stress_response.py::test_m1a5f_s3_1b_no_displacement_when_calm_held_or_at_the_floor

    For each act the policy or a script chose (a fallback hold is not a choice), of a kind in ``kinds.displacing``,
    sent by a living person whose load is above their usual level: the coping amount ``response_up_gain × above / 100
    × basic_level / 100`` is split evenly over the act's living family receivers. From each, in id order, the smallest
    of its share, its room above its floor (``functional_level − floor``) and the sender's room below 100 moves from the
    receiver's ``functional_level`` and ``stress_swing`` to the sender's. A receiver holding a position toward the
    sender (`M1.B.5`) gives nothing. Nothing reads a receiver's age, role in the family, focus or readout. Returns
    each person's change in functional level, as ``(person, "functional_level", change)`` rows.
    """
    moved: dict = {}
    for sel in sorted(selections, key=lambda s: (s.actor, s.index)):
        if sel.decided_by is DecidedBy.FALLBACK or sel.kind not in state.kinds.displacing:
            continue
        sender = state.people.get(sel.actor)
        if sender is None or not sender.alive:
            continue
        above = above_usual(sender, load.get(sel.actor, 0.0))
        if above <= 0:  # only a loaded sender displaces; calm, the act does nothing here (M1.A.5e)
            continue
        receivers = sorted({t for t in sel.targets if t != sel.actor and t in state.people
                            and state.people[t].alive and state.people[t].role is Role.MEMBER})
        if not receivers:
            continue
        share = params.response_up_gain * above / SCALE_MAX * sender.basic_level / SCALE_MAX / len(receivers)
        for pid in receivers:
            receiver = state.people[pid]
            if holds_position(state, pid, sel.actor, params):  # M1.B.5: a held position gives nothing
                continue
            room = receiver.functional_level - stress_floor(receiver, params)
            amount = max(0.0, min(share, room, SCALE_MAX - sender.functional_level))
            if not amount:
                continue
            receiver.functional_level -= amount
            receiver.stress_swing -= amount
            sender.functional_level += amount
            sender.stress_swing += amount
            moved[pid] = moved.get(pid, 0.0) - amount
            moved[sel.actor] = moved.get(sel.actor, 0.0) + amount
    return [(pid, "functional_level", change) for pid, change in sorted(moved.items()) if change]


def bounded(person, level: float, params: EngineParams) -> float:
    """``level`` within the scale, and a fall stopped at the stress floor (revision 19): a level already below the
    floor may rise but not fall further."""
    lowest = min(person.functional_level, stress_floor(person, params))
    return min(SCALE_MAX, max(0.0, lowest, level)) if level < person.functional_level else min(SCALE_MAX, max(0.0, level))


def respond(state: RunState, selections, load: dict, params: EngineParams) -> list[EffectRecord]:
    """Purpose: what a person does under load moves their functioning; calm, it settles back slowly.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.5e, #M1.A.5f, #M7.A.1, #M1.E.7
    Tests:   tests/bowen/test_stress_response.py::test_m1a5e_coping_raises_functioning_and_soothing_lowers_it,
             tests/bowen/test_stress_response.py::test_m1e7_the_systems_adjustment_settles_back_when_calm_and_holds_under_load,
             tests/bowen/test_stress_response.py::test_m1a5e_s3_1d_floor_held_functioning_is_on_a_ledger

    Displaced coping (``displace``) runs first, so each person's own response below reads the level it left.
    """
    # A fallback hold (nothing legal, or no finite score) is not a choice, so it is not coping (review 2026-10-11).
    chose = {s.actor: s.kind for s in selections if s.decided_by is not DecidedBy.FALLBACK}
    rows = displace(state, selections, load, params)  # M1.A.5f, revision 39
    for pid in sorted(state.people):
        person = state.people[pid]
        if not person.alive:
            continue
        above, kind = above_usual(person, load.get(pid, 0.0)), chose.get(pid)
        level = person.basic_level / SCALE_MAX
        if above > 0:  # revision 22: loaded means above their own usual level
            coped = kind == WITHHOLD or (kind is not None and state.kinds.channels.get(kind) == SELF_CHANNEL)
            if coped:
                change = params.response_up_gain * above / SCALE_MAX * level
            elif kind in state.kinds.soothing:
                change = -params.response_down_gain * above / SCALE_MAX * (1.0 - level)
            else:
                change = 0.0
        else:
            change = -params.response_relaxation_rate * (person.stress_swing + person.adjust_swing)
        relaxing = above <= 0
        before = person.functional_level
        person.functional_level = bounded(person, before + change, params)
        moved = person.functional_level - before
        # Calm, the swing decays by its own share whether or not a bound let functioning move: a clamped relaxation
        # must not leave an amount behind to take back later from an unrelated gain (review 2026-10-11).
        if relaxing:
            swings = person.stress_swing + person.adjust_swing
            person.stress_swing *= 1.0 - params.response_relaxation_rate
            person.adjust_swing *= 1.0 - params.response_relaxation_rate  # revision 20: the system's adjustment too
            # Revision 39 (F1): what the bound stopped of that decay goes on the floor_held ledger, and the ledger is
            # then shed by the swings' share, as far as the bound lets; what it stops stays on the ledger. In a week the
            # bound did not stop, nothing is added: the two sides differ there only by rounding (review 2026-10-26).
            if person.functional_level != before + change:
                person.floor_held += moved - (person.stress_swing + person.adjust_swing - swings)
            if person.floor_held:
                level = person.functional_level
                person.functional_level = bounded(
                    person, level - params.response_relaxation_rate * person.floor_held, params)
                person.floor_held += person.functional_level - level
                moved = person.functional_level - before
        else:
            person.stress_swing += moved
        if moved:
            rows.append((pid, "functional_level", moved))
    return [EffectRecord(state.tick, "stress_response", None, people=tuple(rows))] if rows else []
