"""Perception and appraisal — tick steps 3 and 4.

Purpose: give each person the events delivered to them this tick, and appraise
         each one as the change it makes to the receiver's two-sided deviation.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.B.1, #M4.C.1, #M4.C.1a, #M1.F.2, #M1.F.3, #M1.F.4, #M1.F.5, #M1.F.8
Tests:   tests/bowen/test_mechanisms.py; tests/bowen/test_contact.py

Spec revision 11 re-derived `M4.C.1` (Phase C step 1, plan decision D2). Three cases:

**An event addressed to a person on a tie** moves that person's felt contact and
felt impingement on the tie by the kind's two components (``event_kinds.md``):

    scale        = intensity / intensity_scale × conductance × route_damping ** len(route) × fidelity
    Δ contact    = contact component × scale           Δ impingement = impingement component × scale
    Δ acute      = appraisal_gain × steepness(fl) × (deviation after − deviation before) × sign

So an approach toward someone below their optimum is **relief** (Δ acute < 0), and
a push past their tolerated band is a load. `sign` is M1.F.2's source-position sign.

**An exogenous stressor** has no tie, so it is appraised by its intensity:
``intensity / intensity_scale × steepness(fl) × fidelity`` (the Phase B magnitude).

**A witness** keeps Phase B's rule for now: the event's own edge times the witness's
best tie to the sender or a target, scaled as for a stressor. *Interim.* Phase C
step 2 replaces it with `M4.C.9`'s rule, which reads the witness's ties to both.

The batch is applied as a whole: every change is computed from the state before
the batch, then applied in canonical order, so arrival order cannot decide the
outcome (M1.F.8). Contact changes on one tie in one batch are summed, then clamped.

The gain function (`M4.C.2`) is not built yet; Phase C step 2 adds it.
"""

from __future__ import annotations

from src.bowen.engine.contact import clamp_unit, deviation_at, felt_contact, felt_impingement, steepness
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


def _scale(event: Event, conductance: float, params: EngineParams) -> float:
    return event.intensity / params.intensity_scale * conductance * params.route_damping ** len(event.route) * event.fidelity


def _tie_change(state: RunState, delivery: Delivery, event: Event, params: EngineParams):
    """The tie and the (Δ contact, Δ impingement) a target-on-a-tie delivery makes, or None."""
    if delivery.role is not Role.TARGET or event.sender is None:
        return None
    tie = state.tie_between(delivery.recipient, event.sender)
    if tie is None:
        return None
    contact, impingement = state.kinds.components_of(event.kind)
    scale = _scale(event, tie.conductance, params)
    return tie, contact * scale, impingement * scale


def appraisal_delta(state: RunState, delivery: Delivery, event: Event, params: EngineParams) -> float:
    """Purpose: the change in the receiver's acute anxiety from one delivery, from the pre-batch state.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1, #M4.C.1a, #M1.F.2, #M1.F.3, #M1.F.4
    Tests:   tests/bowen/test_contact.py::test_m4c1_delivered_event_is_appraised_by_its_change_in_deviation
    """
    person = state.people[delivery.recipient]
    sign = state.kinds.sign(event.kind, event.source_position)
    change = _tie_change(state, delivery, event, params)
    if change is not None:
        tie, d_contact, d_imp = change
        contact, imp = felt_contact(person, tie), felt_impingement(person, tie)
        before = deviation_at(person, tie, params, contact, imp)
        after = deviation_at(person, tie, params, clamp_unit(contact + d_contact), clamp_unit(imp + d_imp))
        return params.appraisal_gain * steepness(person, params) * (after - before) * sign
    conductance = _conductance(state, delivery.recipient, event, delivery.role)
    return _scale(event, conductance, params) * steepness(person, params) * sign


def apply_base_appraisal(
    state: RunState,
    perceived: dict[PersonId, tuple[tuple[Delivery, Event], ...]],
    params: EngineParams,
) -> list[EffectRecord]:
    """Purpose: apply one tick's appraisals and contact changes as a batch (M1.F.8).
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
    contact_moves: dict = {}
    for delivery, event, _ in sorted(deltas, key=lambda x: x[0]):
        change = _tie_change(state, delivery, event, params)
        if change is not None:
            tie, d_contact, d_imp = change
            key = (tie.id, delivery.recipient)
            c, i = contact_moves.get(key, (0.0, 0.0))
            contact_moves[key] = (c + d_contact, i + d_imp)
    for delivery, event, delta in sorted(deltas, key=lambda x: x[0]):
        state.people[delivery.recipient].acute_anxiety += delta
    for (tie_id, member), (d_contact, d_imp) in sorted(contact_moves.items()):
        tie = state.ties[tie_id]
        tie.felt_contact[member] = clamp_unit(tie.felt_contact[member] + d_contact)
        tie.felt_impingement[member] = clamp_unit(tie.felt_impingement[member] + d_imp)
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
