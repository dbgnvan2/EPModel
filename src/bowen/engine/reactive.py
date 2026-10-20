"""The reactive state's three detectors, and investment — part of tick step 9.

Purpose: update, for every person, the drifting reactive state of `M1.A.19` and the
         directed, valence-blind investment of `M1.B.8`.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.19, #M1.A.18a, #M1.B.8
Tests:   tests/bowen/test_reactive.py

The forms are the project's, [I]. Each detector drifts toward a target at
``reactive_rate`` per tick (``d ← d + rate × (target − d)``):

* ``critical_urge`` — the urge to become critical (`L20.2`): the person's "too much"
  side summed over their ties. Being impinged on is what the urge answers.
* ``inside_other_problem`` — attention sitting inside the other's problem (`K04.11`):
  the person's investment share on a tie times that partner's excess anxiety, at the
  most-invested tie, on a 0–1 scale (excess / 100).
* ``evaluating`` — diagnosing, criticising or praising (`K07.1`): the number of the
  person's own emitted moves this tick that carry a valence of **either** sign.
  Praise counts as much as blame; a negative-only detector misses half the cases.

**Investment** (`M1.B.8`) is the share of thought a tie occupies. Each tick a person's
attention on a tie fades by ``investment_leak_rate`` and gains the absolute size of
what they appraised on it plus their deviation on it. The absolute value makes it
**valence-blind**: conflict-laden preoccupation registers as high investment, and
nothing reads warmth, agreement or tie quality. It is stored per person, so it is
directed. A tie with an external agent accumulates none (`M1.E.7e`: the coach tie stays thin). ``investment_share`` normalises it across the person's ties.
"""

from __future__ import annotations

import math

from src.bowen.engine.contact import deviation, excess, too_much
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import REACTIVE_DETECTORS, SCALE_MAX, Role
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState
from src.bowen.engine.counting import count


def investment_share(state: RunState, person: PersonId, tie: TieId) -> float:
    """Purpose: the share of a person's attention one of their ties occupies.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.8
    Tests:   tests/bowen/test_reactive.py::test_m1b8_conflict_registers_as_high_investment
    """
    total = math.fsum(t.investment.get(person, 0.0) for t in state.ties_of(person))
    return state.ties[tie].investment.get(person, 0.0) / total if total else 0.0


def update_investment(
    state: RunState, attended: dict[tuple[TieId, PersonId], float], params: EngineParams
) -> list[tuple]:
    """Purpose: fade and add each person's attention on each of their ties.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.8
    Tests:   tests/bowen/test_reactive.py::test_m1b8_investment_is_directed
    """
    changes = []
    for tie_id in sorted(state.ties):
        tie = state.ties[tie_id]
        if any(state.people[m].role is Role.EXTERNAL for m in tie_id.members()):
            continue  # M1.E.7e: the coach tie stays thin and accumulates no investment
        for member in tie_id.members():
            person = state.people[member]
            if not person.alive:
                continue
            before = tie.investment.get(member, 0.0)
            gained = attended.get((tie_id, member), 0.0) + deviation(person, tie, params)
            tie.investment[member] = before * (1.0 - params.investment_leak_rate) + gained
            if tie.investment[member] != before:
                changes.append((tie_id, f"investment:{member}", tie.investment[member] - before))
    return changes


def _evaluations(state: RunState, person: PersonId) -> int:
    return count(1 for e in state.store.moves_in(state.tick - 1, state.tick) if e.sender == person and e.valence != 0)


def detector_targets(state: RunState, person: PersonId, params: EngineParams) -> dict[str, float]:
    """Purpose: where each of the three detectors is drifting toward this tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.19, #M1.A.18a
    Tests:   tests/bowen/test_reactive.py::test_m1a19_evaluating_detector_is_two_sided
    """
    me = state.people[person]
    ties = state.ties_of(person)
    critical = math.fsum(too_much(me, t, params) for t in ties)
    inside = 0.0
    for t in ties:
        other = next(p for p in t.id.members() if p != person)
        inside = max(inside, investment_share(state, person, t.id) * excess(state.people[other]) / SCALE_MAX)
    return {"critical_urge": critical, "inside_other_problem": inside, "evaluating": float(_evaluations(state, person))}


def update_reactive_state(state: RunState, params: EngineParams) -> list[tuple]:
    """Purpose: drift every person's three detectors toward their targets.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.19
    Tests:   tests/bowen/test_reactive.py::test_m1a19_every_person_carries_three_drifting_detectors
    """
    changes = []
    for pid in sorted(state.people):
        person = state.people[pid]
        if not person.alive or person.reactive_state is None:
            continue
        targets = detector_targets(state, pid, params)
        for name in REACTIVE_DETECTORS:
            moved = params.reactive_rate * (targets[name] - person.reactive_state[name])
            person.reactive_state[name] += moved
            if moved:
                changes.append((pid, name, moved))
    return changes


def own_acts_attention(state: RunState, attended: dict, events, params: EngineParams) -> dict:
    """Purpose: a projection-type act a person sends keeps their attention on that tie (revision 30).
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.8b
    Tests:   tests/bowen/test_origin_danger_focus.py::test_m1b8b_a_projection_act_adds_to_the_senders_attention

    Returns ``attended`` with ``focus_reinforce_gain`` × intensity ÷ ``intensity_scale`` added on the sender's side of
    each tie a projection-type move (`M4.D.1g`'s set) crossed this week. With `M4.D.1g`'s steering it closes the loop:
    attention brings acts, and acts bring attention, so focus can settle on one child and grow. No child is named.
    """
    from src.bowen.engine.sinks import PROJECTION_KINDS  # sinks is a leaf; imported here to keep this module's head

    added = dict(attended)
    for event in events:
        if event.kind not in PROJECTION_KINDS or event.sender is None or not state.people[event.sender].alive:
            continue
        for target in event.targets:
            tie = state.tie_between(event.sender, target)
            if tie is not None:
                key = (tie.id, event.sender)
                added[key] = added.get(key, 0.0) + params.focus_reinforce_gain * event.intensity / params.intensity_scale
    return added


def update_attention_state(
    state: RunState, attended: dict[tuple[TieId, PersonId], float], params: EngineParams
) -> list[EffectRecord]:
    """Purpose: run investment, then the detectors that read it, and log both.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.8, #M1.A.19
    Tests:   tests/bowen/test_reactive.py::test_m1a19_every_person_carries_three_drifting_detectors
    """
    records = []
    investment = update_investment(state, attended, params)
    detectors = update_reactive_state(state, params)
    if investment:
        records.append(EffectRecord(state.tick, "investment", None, ties=tuple(investment)))
    if detectors:
        records.append(EffectRecord(state.tick, "reactive_state", None, people=tuple(detectors)))
    return records


def weakness_focus(state: RunState, load: dict, params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 9 — an anxious parent's attention grows toward a dependent child who shows a weakness.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.8a, #M1.B.8b
    Tests:   tests/bowen/test_cutoff_and_focus.py::test_m1b8a_perceived_weakness_adds_focus_only_while_anxious,
             tests/bowen/test_origin_danger_focus.py::test_m1b8b_anxious_focus_compounds_where_it_already_is

    For each family, each living parent whose load at selection is above their usual level adds to their investment on
    the tie to each dependent child: ``perceived_weakness_gain × load above usual ÷ 100 × shortfall``, [I]. The
    shortfall is how far the child's functional level is below their dependent siblings' mean (none for an only child),
    plus the child's declared ``weakness`` × 100. No child is named: it is what the children show (revision 29).
    Revision 30 (`M1.B.8b`): ``focus_compound_gain × load above usual ÷ 100 × total × share × (share − Σ share²)`` is
    added, where share is the child's part of the parent's attention among the children and total that attention: a
    child above the share-weighted mean share gains what the others lose, so anxious focus moves to where it already is
    and, once it starts, tends to grow (the owner, 2026-10-14). It adds no investment, so the total stays bounded (the
    owner, 2026-10-14, "saturate the investment"; the earlier form, in proportion to investment, grew without limit).
    Equal children stay equal.
    """
    from src.bowen.engine.stress_response import above_usual  # stress_response imports this module's neighbours

    rows = []
    for family in state.families():
        if family.marital_tie is None:
            continue
        parents = [p for p in family.marital_tie.members() if state.people[p].alive]
        children = sorted({m for t in family.parent_child_ties for m in t.members()} - set(family.marital_tie.members()))
        children = [c for c in children if state.people[c].alive and state.people[c].financially_dependent]
        for parent in parents:
            pressure = max(0.0, above_usual(state.people[parent], load.get(parent, 0.0)))
            if pressure <= 0.0:
                continue
            held = {c: t.investment.get(parent, 0.0) for c in children if (t := state.ties.get(TieId.of(parent, c)))}
            total = math.fsum(held.values())
            spread = math.fsum((h / total) * (h / total) for h in held.values()) if total > 0 else 0.0  # Σ share²
            for child in children:
                tie = state.ties.get(TieId.of(parent, child))
                if tie is None:
                    continue
                others = [state.people[c].functional_level for c in children if c != child]
                shortfall = max(0.0, math.fsum(others) / len(others) - state.people[child].functional_level) if others else 0.0
                shortfall += state.people[child].weakness * SCALE_MAX
                gain = params.perceived_weakness_gain * pressure / SCALE_MAX * shortfall
                # M1.B.8b, revision 30: anxious focus moves to where it already is — a child above the mean share gains
                # what a child below it loses, so the larger share grows and the total is unchanged (zero-sum over the
                # children, since Σ share × (share − Σ share²) = 0). Never below zero investment.
                if total > 0.0:
                    share = held[child] / total
                    shift = params.focus_compound_gain * pressure / SCALE_MAX * total * share * (share - spread)
                    gain += max(shift, -held[child])
                if gain != 0.0:
                    tie.investment[parent] = tie.investment.get(parent, 0.0) + gain
                    rows.append((tie.id, f"investment:{parent}", gain))
    return [EffectRecord(state.tick, "weakness_focus", None, ties=tuple(rows))] if rows else []
