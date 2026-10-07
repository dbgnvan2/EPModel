"""Perception and the base appraisal — tick steps 3 and 4.

Purpose: give each person the events delivered to them this tick, and raise
         their acute anxiety by M4.C.1's base product.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.B.1, #M4.C.1, #M1.F.2, #M1.F.3, #M1.F.4, #M1.F.5, #M1.F.8
Tests:   tests/bowen/test_mechanisms.py

**Phase B builds `M4.C.1` only** (plan decision D1, `M13`). It knowingly fails
`M4.C.2` — "a plain product is a failing implementation" — until Phase C adds
the gain function, and no Phase B test asserts an appraisal magnitude.

    Δ acute = intensity × conductance / max(functional_level, floor)
              × route_damping ** len(route)          (M1.F.3: a neutral third damps)
              × fidelity                             (M1.F.4)
              × sign(kind, source_position)          (M1.F.2)

``conductance`` is the recipient's tie to the sender. A witness may have no tie
to the sender (M1.F.5 still requires it to appraise); it then uses its tie to
the target it overheard. An exogenous event has no edge and uses 1. Phase C
replaces the witness rule with M4.C.9's, which reads both ties.

The batch is applied as a whole: every delta is computed from the state before
the batch, then summed in canonical order, so the order deliveries arrived in
cannot decide the outcome (M1.F.8).
"""

from __future__ import annotations

from src.bowen.engine.events import Delivery, Event, Mechanism, Role
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState

APPRAISED = frozenset({Mechanism.MOVE, Mechanism.EXOGENOUS_STRESSOR})


def perceive(state: RunState, batch: tuple[Delivery, ...]) -> dict[PersonId, tuple[tuple[Delivery, Event], ...]]:
    """Purpose: what each person reads this tick — events addressed to them and events they witnessed.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.B.1
    Tests:   tests/bowen/test_mechanisms.py::test_m4b1_person_reads_addressed_and_witnessed_events
    """
    perceived: dict[PersonId, list[tuple[Delivery, Event]]] = {}
    for delivery in sorted(batch):
        perceived.setdefault(delivery.recipient, []).append((delivery, state.store.event(delivery.event_id)))
    return {p: tuple(v) for p, v in sorted(perceived.items())}


def _conductance(state: RunState, recipient: PersonId, event: Event, role: Role) -> float:
    if event.sender is None:
        return 1.0
    direct = state.tie_between(recipient, event.sender)
    if direct is not None:
        return direct.conductance
    if role is Role.WITNESS:
        candidates = [state.tie_between(recipient, t) for t in event.targets]
        return max((t.conductance for t in candidates if t is not None), default=0.0)
    return 0.0


def appraisal_delta(state: RunState, delivery: Delivery, event: Event, params: EngineParams) -> float:
    """Purpose: M4.C.1's base product for one delivery.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1
    Tests:   tests/bowen/test_mechanisms.py::test_m4c1_delivered_event_raises_receiver_anxiety
    """
    person = state.people[delivery.recipient]
    divisor = max(person.functional_level, params.functional_level_floor)
    route_gain = params.route_damping ** len(event.route)
    sign = state.kinds.sign(event.kind, event.source_position)
    return (
        event.intensity
        * _conductance(state, delivery.recipient, event, delivery.role)
        / divisor
        * route_gain
        * event.fidelity
        * sign
    )


def apply_base_appraisal(
    state: RunState,
    perceived: dict[PersonId, tuple[tuple[Delivery, Event], ...]],
    params: EngineParams,
) -> list[EffectRecord]:
    """Purpose: apply one tick's appraisals as a batch (M1.F.8).
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1, #M1.F.5, #M1.F.8
    Tests:   tests/bowen/test_mechanisms.py::test_m1f8_same_tick_batch_order_does_not_change_state
    """
    deltas: list[tuple[Delivery, Event, float]] = []
    for pid in sorted(perceived):
        for delivery, event in perceived[pid]:
            if event.mechanism in APPRAISED and state.people[pid].alive:
                deltas.append((delivery, event, appraisal_delta(state, delivery, event, params)))
    records: list[EffectRecord] = []
    by_event: dict = {}
    for delivery, event, delta in sorted(deltas, key=lambda x: x[0]):
        by_event.setdefault(event.id, []).append((delivery.recipient, delta))
    for delivery, event, delta in sorted(deltas, key=lambda x: x[0]):
        state.people[delivery.recipient].acute_anxiety += delta
    for event_id in sorted(by_event):
        records.append(
            EffectRecord(
                tick=state.tick,
                mechanism="base_appraisal",
                cause=event_id,
                acute_anxiety=tuple(sorted(by_event[event_id])),
            )
        )
    return records
