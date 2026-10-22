"""Events that change structure rather than anxiety.

Purpose: apply TRIGGER, RECONCILIATION, INSTITUTIONALIZE and binder-unavailable
         events when they fall due, and a delivered CUTOFF to its tie.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.A.2, #M4.A.3, #M4.A.4, #M1.F.9, #M1.B.4, #M6.I.7
Tests:   tests/bowen/test_mechanisms.py

Tie events (TRIGGER, RECONCILIATION) and INSTITUTIONALIZE are applied at step 2
of the tick they are scheduled for. They have no recipients — a TRIGGER is "no
contact at all" (M4.A.2) — so they never pass through delivery. A CUTOFF is a
move: it is delivered like any other and, on delivery, raises its sender's rejection
of the target (M1.B.3c); the tie stays interactive. No tie is ever
removed (M6.I.7: there is no exit from the field).

A delivered CUTOFF also removes the cutter's felt impingement on the tie at once, and that
change is appraised for the cutter only, since the target is the one rejected (M1.B.3c):
the relief `M11.C.4` needs. Felt contact is not touched: it relaxes toward
zero over the following ticks, which is the cost that comes later (`M4.C.1c`).
RECONCILIATION restores resting contact at once (`M1.B.4`: zero re-activation latency),
and returns any anxiety `DISTANCE` bound into the tie to its members, split equally
(`M1.D.2a` (c)).
"""

from __future__ import annotations

import math

from src.bowen.engine.contact import deviation, resting_contact, steepness
from src.bowen.engine.events import BinderKind, Delivery, Event, Mechanism, Role
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.moves import release_bound_distance
from src.bowen.engine.objects import SymptomChannel, TieState
from src.bowen.engine.params import EngineParams
from src.bowen.engine.rejection import end_principled, raise_rejection
from src.bowen.engine.state import ActiveTrigger, RunState

# M5.A.1 fixes the move repertoire, so this name is the spec's, not editorial content.
CUTOFF_KIND = "CUTOFF"

STRUCTURAL = frozenset(
    {Mechanism.TRIGGER, Mechanism.RECONCILIATION, Mechanism.INSTITUTIONALIZE, Mechanism.BINDER_UNAVAILABLE}
)


def nodal_spike(state: RunState, event: Event, params: EngineParams) -> list[EffectRecord]:
    """Purpose: a nodal event re-opens the absence of the cut ties of everyone it reaches (`M4.C.1c`).
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1c, #M4.A.2
    Tests:   tests/bowen/test_mechanisms.py::test_m4c1c_a_nodal_event_spikes_the_cut_ties_of_those_it_reaches,
             tests/bowen/test_mechanisms.py::test_m4c1c_a_nodal_event_leaves_uncut_ties_and_ties_to_the_dead_alone

    Each tie of a person the event reaches
    (``slow_tick.reached_by``, the estimator's reading of reach) that is cut off, with rejection held either way
    (`M1.B.3c`: cutoff is rejection), and whose two members are alive, takes a spike of ``nodal_spike_intensity`` on its
    standing term for the event's week, the next tick (this tick's standing load has run), as a ``TRIGGER`` on that tie
    would. Like a ``TRIGGER``'s, the spike is on the tie, so both its members feel it. A tie reached through both
    members is spiked once per event; spikes from separate events in one week add up, as ``TRIGGER``s do. Ties to the
    dead are left out (a death's own disposition, `M6.3`, covers them), and so are ties that are only not interactive:
    a retired coach's, a removed member's while away, a criterion's arm.
    """
    if event.kind not in state.kinds.nodal or params.nodal_spike_intensity <= 0:
        return []
    from src.bowen.engine.slow_tick import reached_by  # slow_tick reads the estimator's history; imported late

    stored = state.store.event(event.id)  # with the witnesses visibility resolved
    cut = sorted({tie.id for pid in reached_by(state, stored) for tie in state.ties_of(pid)
                  if tie.rejected() and all(state.people[m].alive for m in tie.id.members())})
    for tie_id in cut:  # one week, whatever the event's own duration (the owner's form: "for the event's week")
        state.triggers.append(ActiveTrigger(tie_id, params.nodal_spike_intensity, event.timestamp + 1, event.timestamp + 1))
    if not cut:
        return []
    return [EffectRecord(state.tick, "nodal_spike", event.id,
                         ties=tuple((tie_id, "standing_spike", params.nodal_spike_intensity) for tie_id in cut))]


def apply_structural_event(state: RunState, event: Event, params: EngineParams) -> list[EffectRecord]:
    """Purpose: apply one structural event at its tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.A.2, #M4.A.3, #M4.A.4, #M1.F.9
    Tests:   tests/bowen/test_mechanisms.py::test_m4a3_reconciliation_converts_standing_to_interaction_load
    """
    if event.mechanism is Mechanism.TRIGGER:
        # Applied at step 2 of tick t; the standing load ran at step 1, so the
        # spike lands from step 1 of t + 1, for `duration` ticks (M3.D.2, M4.A.2).
        state.triggers.append(
            ActiveTrigger(event.on_tie, event.intensity, event.timestamp + 1, event.timestamp + event.duration)
        )
        return [EffectRecord(state.tick, "trigger", event.id, ties=((event.on_tie, "standing_spike", event.intensity),))]
    if event.mechanism is Mechanism.RECONCILIATION:
        tie = state.ties[event.on_tie]
        # M1.B.4: reunion restores full coupling with zero re-activation latency — bond energy is untouched.
        tie.interactive = True
        tie.tie_state = TieState.ORDINARY
        cleared = [(tie.id, f"rejection:{m}", -tie.rejection[m]) for m in sorted(tie.rejection) if tie.rejection[m]]
        tie.rejection.clear()  # M1.B.3c: reunion ends the rejection with the separation
        for member in tie.id.members():
            end_principled(tie, member)  # M1.B.3d: the declared rejection is gone, so is its kind
        for member in tie.id.members():
            tie.felt_contact[member] = resting_contact(state.people[member], tie, params)
        # M1.D.2a (c), M6.4: distancing prevented, the anxiety it bound returns to the tie's members.
        returned = release_bound_distance(
            state, tie, tuple(m for m in tie.id.members() if state.people[m].alive)
        )
        ties = [(tie.id, "interactive", 1.0)] + cleared
        if returned:
            ties.append((tie.id, "distance_bound_anxiety", -math.fsum(v for _, v in returned)))
        return [EffectRecord(state.tick, "reconciliation", event.id, acute_anxiety=returned, ties=tuple(ties))]
    if event.mechanism is Mechanism.INSTITUTIONALIZE:
        changed = []
        for target in event.targets:
            for tie in state.ties_of(target):
                # M4.A.4: worry edges — non-interactive, bond energy retained, no new machinery.
                tie.interactive = False
                changed.append((tie.id, "interactive", 0.0))
        return [EffectRecord(state.tick, "institutionalize", event.id, ties=tuple(sorted(changed)))]
    if event.mechanism is Mechanism.BINDER_UNAVAILABLE:
        records = [_release_binder(state, event)]
        if event.binder.kind is BinderKind.SYMPTOM_CHANNEL:
            # Revision 24 (M7.D.2a): what the family's focus on the symptom held goes back to the partners.
            from src.bowen.engine.symptoms import release_bound

            bearer = PersonId(event.binder.target.split(":")[0])
            moved = release_bound(state, bearer)
            if moved:
                records.append(EffectRecord(state.tick, "lock_in", event.id, acute_anxiety=moved,
                                            people=((bearer, "symptom_bound", -math.fsum(v for _, v in moved)),)))
        return records
    raise ValueError(f"{event.mechanism} is not a structural event")


def _release_binder(state: RunState, event: Event) -> EffectRecord:
    """M1.F.9: the binder's held anxiety returns to the family budget; none is discarded. A symptom channel's hold
    returns to the partners instead."""
    binder = event.binder
    if binder.kind is BinderKind.TIE_DISTANCE:
        a, b = binder.target.split("~")
        tie = state.ties[TieId.of(PersonId(a), PersonId(b))]
        held, tie.distance_bound_anxiety = tie.distance_bound_anxiety, 0.0
        detail = ((tie.id, "distance_bound_anxiety", -held),)
        triangles = ()
    elif binder.kind is BinderKind.TRIANGLE_POSITION:
        tri_id = _triangle_id(binder.target)
        triangle = state.triangles[tri_id]
        held, triangle.bound_anxiety = triangle.bound_anxiety, 0.0
        detail = ()
        triangles = ((tri_id, "bound_anxiety", f"-{held}"),)
    elif binder.kind is BinderKind.SYMPTOM_CHANNEL:
        person_id, channel = binder.target.split(":")
        person = state.people[PersonId(person_id)]
        # Revision 24 (review 2026-10-12): the symptom ends, and its load — an integral of weeks above the usual level,
        # not an anxiety stock — is not credited to the budget. What the symptom held of the family's anxiety is its
        # hold (symptom_bound), which apply_structural_event returns to the partners.
        held = 0.0
        person.symptom_load[SymptomChannel(channel)] = 0.0
        person.symptom_active[SymptomChannel(channel)] = False
        detail = ()
        triangles = ()
    else:
        raise NotImplementedError(
            "external-agent responsibility (M1.E.3) is built in Phase C; Phase B cannot release it"
        )
    state.family.undifferentiation_budget += held
    return EffectRecord(
        state.tick, "binder_unavailable", event.id, ties=detail, triangles=triangles,
        sinks=(("undifferentiation_budget", held),),
    )


def _triangle_id(text: str) -> TriangleId:
    a, b, c = text.split("/")
    return TriangleId.of(PersonId(a), PersonId(b), PersonId(c))


def apply_delivered_cutoffs(state: RunState, batch: tuple[Delivery, ...], params: EngineParams) -> list[EffectRecord]:
    """Purpose: a delivered CUTOFF raises its sender's rejection of the target, and removes the sender's impingement at once.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3c, #M1.B.3, #M6.I.7, #M4.C.1c
    Tests:   tests/bowen/test_rejection.py::test_m1b3c_a_delivered_cutoff_leaves_the_tie_open_as_the_cutters_rejection,
             tests/bowen/test_mechanisms.py::test_m1b3_four_tie_states_are_distinct
    """
    records = []
    for delivery in sorted(batch):
        if delivery.role is not Role.TARGET:
            continue
        event = state.store.event(delivery.event_id)
        if event.kind != CUTOFF_KIND or event.sender is None:
            continue
        tie = state.tie_between(event.sender, delivery.recipient)
        end_principled(tie, event.sender)  # M1.B.3d: a cutoff the policy selects is reactive
        rose = raise_rejection(tie, event.sender, params.rejection_cutoff_gain * event.intensity / params.intensity_scale)
        relief = []
        person = state.people[event.sender]
        before = deviation(person, tie, params)
        tie.felt_impingement[event.sender] = 0.0
        delta = params.appraisal_gain * steepness(person, params) * (deviation(person, tie, params) - before)
        if delta:
            person.acute_anxiety += delta
            relief.append((event.sender, delta))
        records.append(EffectRecord(
            state.tick, "cutoff", event.id, acute_anxiety=tuple(relief),
            ties=((tie.id, f"rejection:{event.sender}", rose),),
        ))
    return records
