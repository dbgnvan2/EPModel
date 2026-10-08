"""Act — tick step 8 — and the injection of scheduled inputs at step 2.

Purpose: turn a selection into a full event, fill its witnesses and deliveries
         through visibility, record it, and queue its deliveries.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.E.1, #M4.E.1a, #M1.F.1, #M1.F.4, #M16.A.2, #M16.A.3, #M4.D.1b
Tests:   tests/bowen/test_mechanisms.py::test_m4e1_scripted_move_becomes_full_event
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from src.bowen.engine import iposition

from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.objects import Role
from src.bowen.engine.log_records import DecidedBy, DeliveredRecord, EffectRecord, EmittedRecord, SelectionRecord
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState
from src.bowen.engine.visibility import HouseholdConductanceVisibility


# M4.D.1b: the outcome that is not a move. The spec's name.
WITHHOLD = "WITHHOLD"
I_POSITION = "I-POSITION"


@dataclass(frozen=True)
class Selection:
    """One person's outcome this tick: a move, or ``WITHHOLD``. A script or the policy supplies it.

    The rationale fields are the policy's (`M16.A.3`–`M16.A.3c`); a scripted selection
    leaves them empty. A ``WITHHOLD`` names the automatic act it held back in
    ``withheld`` and its target in ``targets``; a fallback hold names neither.
    """

    actor: PersonId
    kind: str
    targets: tuple[PersonId, ...]
    intensity: float
    route: tuple[PersonId, ...] = ()
    source_position: SourcePosition = SourcePosition.NONE
    index: int = 0
    decided_by: DecidedBy = DecidedBy.SCRIPTED
    propensities: tuple[tuple[str, float], ...] = ()
    draw: float | None = None
    legal_set: tuple[str, ...] = ()
    withheld: str | None = None
    fallback_rule: str | None = None
    # M4.D.1d: acute anxiety the unresolved competition adds this tick; the engine applies it.
    urge: float = 0.0
    # M4.D.6: the learned-value key of an automatic act, which the learner credits; "" otherwise.
    value_key: str = ""
    # M16.A.3a: the beliefs the policy read for this selection.
    beliefs_used: tuple[tuple[str, float], ...] = ()
    # M5.D.9: the I-POSITION sequence step this outcome is, if it is one (DEFINE, ABORT, FOLLOW_UP).
    sequence_step: str | None = None


def _record(state: RunState, selection: Selection, event_id) -> SelectionRecord:
    return SelectionRecord(
        tick=state.tick, actor=selection.actor, decided_by=selection.decided_by, event_id=event_id,
        propensities=selection.propensities, draw=selection.draw, legal_set=selection.legal_set,
        beliefs_used=selection.beliefs_used,
        withheld=selection.withheld,
        withheld_toward=selection.targets[0] if selection.withheld and selection.targets else None,
        fallback_rule=selection.fallback_rule, sequence_step=selection.sequence_step,
    )


def withhold(state: RunState, selection: Selection, params: EngineParams) -> list:
    """Purpose: record a withheld outcome; the act held back still changes its tie.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1b, #M4.D.1f, #M1.B.8
    Tests:   tests/bowen/test_policy.py::test_m4d1b_withheld_move_still_changes_tie_state

    Nothing is emitted. The urge was computed and detected, so it occupies thought: the
    actor's investment in the tie toward the person it would have acted on rises by
    ``withhold_investment_gain × intensity / intensity_scale`` (`M1.B.8`, [I]). "I did not
    take any obvious I-position with Mother; I just did not anxiously hover over her"
    (`KS08.1`): what the other no longer receives is the absence of the act.
    """
    records: list = [_record(state, selection, None)]
    if selection.withheld is None:
        return records
    tie = state.tie_between(selection.actor, selection.targets[0])
    if any(state.people[m].role is Role.EXTERNAL for m in tie.id.members()):
        return records  # M1.E.7e: the coach tie accumulates no investment
    gain = params.withhold_investment_gain * selection.intensity / params.intensity_scale
    tie.investment[selection.actor] = tie.investment.get(selection.actor, 0.0) + gain
    records.append(EffectRecord(state.tick, "withhold", None, ties=((tie.id, f"investment:{selection.actor}", gain),)))
    return records


def act(state: RunState, selection: Selection, visibility: HouseholdConductanceVisibility,
        params: EngineParams | None = None) -> list:
    """Purpose: make a selection into an event and queue its deliveries, or record a withheld outcome.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.E.1, #M4.E.1a, #M4.D.1b
    Tests:   tests/bowen/test_mechanisms.py::test_m4e1_scripted_move_becomes_full_event
    """
    if selection.kind == WITHHOLD:
        if params is None:
            raise ValueError("a withheld outcome needs the engine parameters")
        return withhold(state, selection, params)
    assertion = False
    if selection.kind == I_POSITION and selection.sequence_step is None and params is not None:
        # M5.D.1, M5.D.2a, M5.F.4: a freshly selected I-POSITION either starts a sequence,
        # emitting nothing this week, or executes at once as the assertion form.
        target = selection.targets[0]
        if not iposition.assertion_form(state, selection.actor, target, params):
            started = iposition.begin(state, selection.actor, target)
            return [_record(state, dataclasses.replace(selection, sequence_step=iposition.PREPARE), None), started]
        assertion = True
    if state.kinds.mechanism_of(selection.kind) is not Mechanism.MOVE:
        raise ValueError(f"{selection.kind} is not a move")
    event = Event(
        id=EventId(state.tick, str(selection.actor), selection.index),
        kind=selection.kind,
        mechanism=Mechanism.MOVE,
        sender=selection.actor,
        targets=tuple(sorted(selection.targets)),
        intensity=selection.intensity,
        timestamp=state.tick,
        duration=1,
        exogenous=False,
        source_position=selection.source_position,
        channel=Channel.SCRIPTED if selection.decided_by is DecidedBy.SCRIPTED else Channel.AUTOMATIC,
        route=selection.route,
        fidelity=visibility.fidelity_for(selection.route),
        assertion=assertion,
    )
    resolved = visibility.resolve(event, state.people, state.ties)
    state.store.record_event(resolved.event)
    for delivery in resolved.deliveries:
        state.queue.schedule(delivery)
    records = [_record(state, selection, event.id), EmittedRecord(resolved.event)]
    if selection.sequence_step in iposition.OUTCOME_STEPS and params is not None:
        records += iposition.step_done(state, selection.actor, params)
    return records


def inject(state: RunState, event: Event, visibility: HouseholdConductanceVisibility) -> list:
    """Purpose: bring a scheduled exogenous or structural event, or an endogenous one, into the run at its tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.6, #M1.F.7, #M3.C.2
    Tests:   tests/bowen/test_mechanisms.py::test_m4a2_trigger_spikes_standing_load_without_contact

    An exogenous event has no edge and arrives in its own tick (latency 0), so
    it must be injected before that tick's deliveries are released.
    """
    if event.timestamp != state.tick:
        raise ValueError(f"event {event.id} injected at tick {state.tick}")
    resolved = visibility.resolve(event, state.people, state.ties)
    state.store.record_event(resolved.event)
    for delivery in resolved.deliveries:
        state.queue.schedule(delivery)
    return [EmittedRecord(resolved.event)]


def record_deliveries(state: RunState, batch) -> list:
    for delivery in batch:
        state.store.record_delivery(delivery)
    return [DeliveredRecord(d) for d in batch]
