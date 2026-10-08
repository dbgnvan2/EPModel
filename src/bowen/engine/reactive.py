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

from src.bowen.engine.contact import deviation, excess, too_much
from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import REACTIVE_DETECTORS, SCALE_MAX, Role
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState


def investment_share(state: RunState, person: PersonId, tie: TieId) -> float:
    """Purpose: the share of a person's attention one of their ties occupies.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.8
    Tests:   tests/bowen/test_reactive.py::test_m1b8_conflict_registers_as_high_investment
    """
    total = sum(t.investment.get(person, 0.0) for t in state.ties_of(person))
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
    return sum(
        1 for e in state.store.events()
        if e.timestamp == state.tick and e.sender == person and e.mechanism is Mechanism.MOVE and e.valence != 0
    )


def detector_targets(state: RunState, person: PersonId, params: EngineParams) -> dict[str, float]:
    """Purpose: where each of the three detectors is drifting toward this tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.19, #M1.A.18a
    Tests:   tests/bowen/test_reactive.py::test_m1a19_evaluating_detector_is_two_sided
    """
    me = state.people[person]
    ties = state.ties_of(person)
    critical = sum(too_much(me, t, params) for t in ties)
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
