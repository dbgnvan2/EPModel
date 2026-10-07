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

``conductance`` for a target is its tie to the sender; an exogenous event has
no edge and uses 1. A **witness** (M1.F.5 requires it to appraise) takes the
event's own edge — the sender's tie to the nearest target — times its best tie
to the sender or a target. So a witness never takes more of an event than the
edge it travelled on. *Changed at Phase B step 12*: the first rule used the
witness's tie to the target alone, and the first rendered trace showed
witnesses hit harder than the person addressed (a daughter overhearing her
grandmother's call to her mother took +2.9 against her mother's +1.8). Phase C
replaces this with M4.C.9's rule, which reads both ties.

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
    if role is Role.TARGET:
        if event.sender is None:
            return 1.0
        direct = state.tie_between(recipient, event.sender)
        return direct.conductance if direct is not None else 0.0
    anchors = [p for p in (event.sender, *event.targets) if p is not None]
    own = max((t.conductance for a in anchors if (t := state.tie_between(recipient, a)) is not None), default=0.0)
    if event.sender is None:
        edge = 1.0
    else:
        edges = [t.conductance for x in event.targets if (t := state.tie_between(event.sender, x)) is not None]
        edge = max(edges, default=0.0)
    return edge * own


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
