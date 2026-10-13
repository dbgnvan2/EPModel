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
  ``contact_window`` weeks, 1 up to ``contact_optimum`` and ``(optimum / n) ** contact_excess_exponent`` above it,
  so more coaching past a low rate is worse, not better.
* **own log** — 1 if the recipient's delayed view (`M16.D`: its own events older than
  ``delayed_view_weeks``) holds any move it sent, else 0: `M1.E.7c`'s fourth form, delayed
  self-observation, makes landing likelier. With the store emptied it never fires (`M16.T.6`).

The occasion (`M1.E.7d`) is one keyed uniform per contact (``landed_contact``, dyad-keyed).
A landed contact raises the recipient's ``systems_perspective`` by ``perspective_gain`` of
its distance from 1. *Spec revisions 18 and 19 (owner, 2026-10-11):* a person's own practice also raises it — a
self-directed act (or ``WITHHOLD``) made under load that worked (the person held their position and the tie calmed
over ``practice_window`` weeks) adds ``self_practice_gain`` of its distance from 1 (``self_practice``): perspective is
self-discovered. Coaching gives larger, regular boosts to the same loop. It is never a reinforcement target
(`M4.D.6d`). The other two forms of `M1.E.7c` (instance and
category supply) are not built.
"""

from __future__ import annotations

from typing import NamedTuple

from src.bowen.engine.act import SELF_CHANNEL, WITHHOLD
from src.bowen.engine.contact import deviation
from src.bowen.engine.draws import DrawKey
from src.bowen.engine.events import Delivery, Mechanism, Role
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import DecidedBy, EffectRecord
from src.bowen.engine.moves import TRIANGLE, triangle_partner
from src.bowen.engine.objects import SCALE_MAX, Role as PersonRole
from src.bowen.engine.outside_ness import efficacy
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState
from src.bowen.engine.stress_response import above_usual
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
    frequency = 1.0 if recent <= params.contact_optimum else (params.contact_optimum / recent) ** params.contact_excess_exponent
    own_log = any(e.sender == recipient.id and e.mechanism is Mechanism.MOVE
                  for e in state.store.delayed_view(recipient.id, state.tick, params.delayed_view_weeks))
    p = params.landing_rate * efficacy(coach) * frequency * (1.0 + params.delayed_view_bonus * own_log)
    return min(1.0, p)


def apply_landed_contacts(state: RunState, batch: tuple[Delivery, ...], params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 4 — each external agent's move delivered to a family member lands or does not.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.E.7, #M1.E.7d
    Tests:   tests/bowen/test_external.py::test_m1e7_perspective_rises_only_on_a_landed_contact_or_own_practice
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


def record_stances(state: RunState, selections, params: EngineParams) -> None:
    """Purpose: remember, per pair, the week a person took a stance toward the other, and end it on a reactive act.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.5 (revision 21)
    Tests:   tests/bowen/test_moves.py::test_m1b5_holding_a_position_stops_the_exchange_both_ways,
             tests/bowen/test_moves.py::test_m1b5_a_stance_lapses_with_time_or_a_reactive_act

    A self-directed act or ``WITHHOLD`` toward someone takes a stance toward them. Any automatic act toward them ends
    it — over-functioning, pursuing, conflict, distancing, cutting off, giving way — and so does triangling a third
    person in about them, the inside partner the ``TRIANGLE`` would form (owner decision 2026-10-12: a stance is
    broken by behaving reactively toward that person). A fallback hold has no target and is not a stance.
    """
    for selection in selections:
        if not selection.targets or selection.decided_by is DecidedBy.FALLBACK:
            continue
        if selection.kind == WITHHOLD or state.kinds.channels.get(selection.kind) == SELF_CHANNEL:
            for other in selection.targets:
                state.stances[(selection.actor, other)] = state.tick
            continue
        about = set(selection.targets)
        if selection.kind == TRIANGLE:
            for target in selection.targets:
                found = triangle_partner(state, selection.actor, target, params)
                if found is not None:
                    about.add(found[1])
        for other in about:
            state.stances.pop((selection.actor, other), None)


class Practice(NamedTuple):
    """A self-directed act under load awaiting judgement (revision 19)."""

    actor: PersonId
    other: PersonId
    tick: int
    deviation: float


def self_practice(state: RunState, selections, load: dict, params: EngineParams) -> list[EffectRecord]:
    """Purpose: a self-directed act that worked builds the person's systems perspective, slowly — self-discovered.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.E.7 (revisions 18, 19)
    Tests:   tests/bowen/test_external.py::test_m1e7_a_self_directed_act_that_worked_builds_perspective

    The owner, 2026-10-11: perspective "should be 'self discovered' and emerge from learned behaviour. If an agent
    happens to do a 'systems' type behaviour and it works, the agent should be able to 'remember' that." It works when
    the person held their position and the tie calmed (decided the same day): a self-directed act, or ``WITHHOLD``,
    made toward someone under load is judged ``practice_window`` weeks later. It counts if, over those weeks, the
    person made no avoiding or soothing act toward that person, and their own deviation on the tie is lower than when
    they acted. Then perspective rises by ``self_practice_gain`` of its distance from 1. Immediate relief does not
    count: a systems move often feels worse first (the change-back reaction), and the self-directed channel is never
    reinforced by relief (`M4.D.6d`).

    When a stance works, the other functions up by ``system_adjust_gain`` (revision 19): they are not going to win.
    That gain is kept in the other's ``adjust_swing`` as a level up to ``system_adjust_gain``, renewed when the stance
    works again, never stacked (revision 20). It settles back when they are calm (``stress_response.respond``), and the
    estimator does not read it (`M1.A.4k`).
    """
    rows = []
    due = [p for p in state.practice if p.tick + params.practice_window <= state.tick]
    state.practice = [p for p in state.practice if p.tick + params.practice_window > state.tick]
    for actor, other, tick, before in due:
        person, tie = state.people[actor], state.tie_between(actor, other)
        if not person.alive or tie is None:
            continue
        soothed = any(e.sender == actor and e.kind in state.kinds.soothing and tick < e.timestamp <= state.tick
                      for e in state.store.moves_on_pair(actor, other))
        if soothed or deviation(person, tie, params) >= before:
            continue
        gained = person.systems_perspective or 0.0
        person.systems_perspective = gained + params.self_practice_gain * (1.0 - gained)
        rows.append((actor, "systems_perspective", person.systems_perspective - gained))
        # Revision 19, the system adjusts (owner decision 2026-10-11): the other, finding they are not going to win,
        # functions up. "If the person doesn't change back, the system adjusts — the other agents realize that they
        # aren't going to win and improve their functioning."
        partner = state.people[other]
        # Once per stance, not once per act (review 2026-10-11): weekly acts toward the same person open overlapping
        # windows, and each would otherwise add the gain again.
        last = state.adjusted.get((actor, other))
        if last is not None and state.tick - last < params.practice_window:
            continue
        if partner.alive:
            before_level = partner.functional_level
            # Revision 20 (owner decision 2026-10-12): a level, not a stack. The other is up by system_adjust_gain
            # while the stance keeps working; a renewed stance tops it back up, and it never accumulates past the gain.
            top_up = max(0.0, params.system_adjust_gain - partner.adjust_swing)
            partner.functional_level = min(SCALE_MAX, before_level + top_up)
            if partner.functional_level != before_level:
                state.adjusted[(actor, other)] = state.tick  # only an adjustment that happened uses up the window
                partner.adjust_swing += partner.functional_level - before_level
                rows.append((other, "functional_level", partner.functional_level - before_level))
    for selection in sorted(selections, key=lambda s: s.actor):
        person = state.people[selection.actor]
        self_directed = selection.kind == WITHHOLD or state.kinds.channels.get(selection.kind) == SELF_CHANNEL
        # Under load means above the person's usual level, as for the stress response (M1.A.5e, revision 22).
        if not (self_directed and person.alive and selection.targets
                and above_usual(person, load.get(selection.actor, 0.0)) > 0):
            continue
        other = selection.targets[0]
        tie = state.tie_between(selection.actor, other)
        if tie is not None:
            state.practice.append(Practice(selection.actor, other, state.tick, deviation(person, tie, params)))
    return [EffectRecord(state.tick, "self_practice", None, people=tuple(rows))] if rows else []
