"""The external agent's contact — when it lands, and what a landing changes (Phase C step 9).

Purpose: decide, for each move an external agent delivers to a family member, whether
         it lands, and raise the recipient's systems perspective only then.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.E.1, #M1.E.6, #M1.E.7, #M1.E.7c, #M1.E.7d, #M1.E.7e, #M1.E.8, #M16.D.1
Tests:   tests/bowen/test_external.py

Every form is the project's, graded [I]; the constants are in ``config/bowen/constants.md``.

A move from an external agent, delivered to a family member as target, **lands** with
probability

    landing_rate × coach quality × binders failing × frequency × (1 + delayed_view_bonus × own log)

* **coach quality** — the agent's efficacy, ``1 − max(outward, inward)``: an agent caught in
  the family's anxiety transmits nothing. This is `M1.E.7c`'s third form, non-participation
  — the agent's failure to be recruited.
* **binders failing** — 1 if the recipient has an active symptom, or its symptom load has
  reached ``binder_failure_fraction`` of its threshold; 0 otherwise (`M1.E.7`: pain is
  necessary). Without it no contact lands.
* **frequency** (`M1.E.8`) — with ``n`` contacts from the agent to the recipient in the last
  ``contact_window`` weeks, 1 up to ``contact_optimum`` and ``(optimum / n) ** 2`` above it,
  so more coaching past a low rate is worse, not better.
* **own log** — 1 if the recipient's delayed view (`M16.D`: its own events older than
  ``delayed_view_weeks``) holds any move it sent, else 0: `M1.E.7c`'s fourth form, delayed
  self-observation, makes landing likelier. With the store emptied it never fires (`M16.T.6`).

The occasion (`M1.E.7d`) is one keyed uniform per contact (``landed_contact``, dyad-keyed).
A landed contact raises the recipient's ``systems_perspective`` by ``perspective_gain`` of
its distance from 1. Nothing else writes it: it never rises spontaneously and is never a
reinforcement target (`M1.E.7`, `M4.D.6d`). The other two forms of `M1.E.7c` (instance and
category supply) are not built.
"""

from __future__ import annotations

from src.bowen.engine.draws import DrawKey
from src.bowen.engine.events import Delivery, Mechanism, Role
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import Role as PersonRole
from src.bowen.engine.outside_ness import efficacy
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState
from src.bowen.engine.symptoms import threshold


def binders_failing(person, params: EngineParams) -> bool:
    """Purpose: the recipient's binders are failing — a symptom active, or the load near threshold.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.E.7
    Tests:   tests/bowen/test_external.py::test_m1e7_contact_lands_only_when_binders_fail
    """
    if any(person.symptom_active.values()):
        return True
    return max(person.symptom_load.values()) >= params.binder_failure_fraction * threshold(person, params)


def landing_probability(state: RunState, coach, recipient, params: EngineParams) -> float:
    """Purpose: the chance one contact lands.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.E.7, #M1.E.7c, #M1.E.7d, #M1.E.8
    Tests:   tests/bowen/test_external.py::test_m1e8_more_contact_past_a_low_rate_lands_less
    """
    if not binders_failing(recipient, params):
        return 0.0
    recent = sum(
        1 for e in state.store.sent_by(coach.id)
        if e.mechanism is Mechanism.MOVE and recipient.id in e.targets
        and state.tick - params.contact_window < e.timestamp <= state.tick
    )
    frequency = 1.0 if recent <= params.contact_optimum else (params.contact_optimum / recent) ** 2
    own_log = any(e.sender == recipient.id and e.mechanism is Mechanism.MOVE
                  for e in state.store.delayed_view(recipient.id, state.tick, params.delayed_view_weeks))
    p = params.landing_rate * efficacy(coach) * frequency * (1.0 + params.delayed_view_bonus * own_log)
    return min(1.0, p)


def apply_landed_contacts(state: RunState, batch: tuple[Delivery, ...], params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 4 — each external agent's move delivered to a family member lands or does not.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.E.7, #M1.E.7d
    Tests:   tests/bowen/test_external.py::test_m1e7_perspective_rises_only_on_a_landed_contact
    """
    records = []
    for delivery in sorted(batch):
        if delivery.role is not Role.TARGET:
            continue
        event = state.store.event(delivery.event_id)
        if event.mechanism is not Mechanism.MOVE or event.sender is None:
            continue
        coach, recipient = state.people[event.sender], state.people[delivery.recipient]
        if coach.role is not PersonRole.EXTERNAL or recipient.role is PersonRole.EXTERNAL or not recipient.alive:
            continue
        p = landing_probability(state, coach, recipient, params)
        if p <= 0:
            continue
        key = DrawKey.make("landed_contact", tick=state.tick, actor=coach.id, partner=recipient.id,
                           purpose="land", index=0)
        if state.draws.uniform(key) < p:
            before = recipient.systems_perspective or 0.0
            recipient.systems_perspective = before + params.perspective_gain * (1.0 - before)
            records.append(EffectRecord(state.tick, "landed_contact", event.id, people=(
                (recipient.id, "systems_perspective", recipient.systems_perspective - before),)))
    return records
