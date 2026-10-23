"""Symptom accumulation, onset, lock-in and removal — part of tick step 9.

Purpose: integrate each person's time above their usual level into symptom load in the channel they express, emit an
         endogenous event when it crosses threshold, let the family's focus on an active symptom bind and release
         anxiety, and remove a bearer when the others' tolerance gives out.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.3, #M4.C.3a, #M7.D.1, #M7.D.2, #M7.D.2a, #M7.D.2c, #M7.D.2d,
         #M7.D.4, #M1.A.6, #M1.A.11b, #M1.A.11c, #M1.F.7
Tests:   tests/bowen/test_symptoms.py

The forms are the project's, [I]:

    integrand  = max(0, excess − usual_excess)                     (M4.C.3a: above the usual level)
    into c     = integrand / (1 + channel_protection × Σ_{c'≠c} load[c'] / threshold)      (M7.D.2c)
    load[c]    ← load[c] × (1 − symptom_leak_rate) + into c   (every channel leaks; only the expressed one is fed)
    onset when load[c] ≥ threshold = symptom_threshold_gain × functional_level           (M1.A.6)
    re-arm when load[c] < symptom_rearm_fraction × threshold

**The integrand is time above the person's usual level.** It is still time above a floor, not peak, so
many resolved excursions and one unresolved excursion stay distinguishable.

**The expressed channel** (`M7.D.2`, `M1.A.11b`, `M1.A.11c`) is the one with the largest
``constitution_strength × [c = channel_prior] + (1 − constitution_strength) × channel_focus[c]``, ties to the prior.
With no focus the prior is expressed; a family focus can move it; a constitution strength of 1 overrides any focus.
``channel_focus`` is written from the acts delivered (`M1.A.11c`): engine-side attention, not a belief
any policy reads.

**Lock-in** (`M7.D.2a`, `M7.D.2d`). For a bearer with an active symptom of severity ``x = load / threshold``:

    s(x) = lock_in_gain × x × (1 − x / lock_in_turn)

Below the turn (s > 0) each partner on an interactive tie moves ``s × max(0, partner's excess above usual)`` of their
acute anxiety into the bearer's ``symptom_bound``: the family calms around the symptom. Past the turn (s < 0)
``min(1, |s|)`` of the bound stock goes back to those partners each week, and the same share of the bearer's own
excess above usual spills to them, split equally: the symptom destabilises what it was holding, and holds the bearer's
anxiety no longer. When the symptom ends (re-arm) or a ``binder_unavailable`` names it, the whole bound stock goes back. Every
movement is a transfer in `M6.4`'s table (``lock_in``). Removal (`INSTITUTIONALIZE`) makes the ties non-interactive,
so both directions stop; the bound stock stays with the bearer. No rule writes resistance to removal or relief on it.

**Removal fired by tolerance** (`M7.D.4`). If a living member has an active symptom and some other member on an
interactive tie to them has excess above usual greater than their ``Family.tolerance``, the member carrying the most
active symptom load is removed by an endogenous ``INSTITUTIONALIZE`` sent by that member. It reads the others'
anxiety, not the bearer's severity. It is suppressed while decision ownership is held outside the family (`M5.C.1`).
A member with no tolerance listed never fires it.

On onset the bearer emits one ``SYMPTOM_ONSET`` event (`M1.F.7`: endogenous, never drawn from incidence data),
addressed to everyone on an interactive tie with them, at ``symptom_event_intensity``.
"""

from __future__ import annotations

import math

from src.bowen.engine.act import inject
from src.bowen.engine.contact import excess
from src.bowen.engine.event_sizes import sized_intensity
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import SCALE_MAX, Person, SymptomChannel, Role
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState
from src.bowen.engine.stress_response import above_usual
from src.bowen.engine.visibility import HouseholdConductanceVisibility

SYMPTOM_KIND = "SYMPTOM_ONSET"
OWN_ILLNESS_KIND, FAMILY_ILLNESS_KIND = "OWN_ILLNESS", "FAMILY_ILLNESS"  # M7.D.1a, revision 36
REMOVAL_KIND = "INSTITUTIONALIZE"
CHANNELS = tuple(SymptomChannel)


def threshold(person: Person, params: EngineParams) -> float:
    """Purpose: the onset threshold, evaluated against functional level.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.6, #M7.D.1
    Tests:   tests/bowen/test_symptoms.py::test_m1a6_threshold_reads_functional_level
    """
    # Floored as steepness is (M4.C.1a), so a person at functional level 0 has a small threshold, not none
    # (review 2026-10-12: severity and protection divide by it).
    return params.symptom_threshold_gain * max(person.functional_level, params.functional_level_floor)


def expressed_channel(person: Person, params: EngineParams) -> SymptomChannel | None:
    """Purpose: the channel the person's load goes to: the prior, unless the family's focus outweighs it.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.D.2, #M1.A.11b, #M1.A.11c
    Tests:   tests/bowen/test_symptoms.py::test_m1a11c_a_family_focus_moves_the_channel_and_constitution_overrides_it
    """
    prior = person.channel_prior
    if prior is None:
        return None
    strength = params.constitution_strength if person.constitution_strength is None else person.constitution_strength

    def weight(c: SymptomChannel) -> float:
        return strength * (c is prior) + (1.0 - strength) * person.channel_focus.get(c, 0.0)

    best = max(weight(c) for c in CHANNELS)
    return prior if weight(prior) >= best else next(c for c in CHANNELS if weight(c) == best)


def integrate(person: Person, params: EngineParams) -> tuple[SymptomChannel, ...]:
    """Purpose: one tick of the leaky chronicity integrator; return every channel whose onset fires now.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.3, #M4.C.3a, #M7.D.1, #M7.D.2c
    Tests:   tests/bowen/test_symptoms.py::test_m4c3a_one_sustained_excursion_builds_more_than_several_resolved,
             tests/bowen/test_symptoms.py::test_m7d2c_load_in_one_channel_slows_onset_in_another
    """
    channel = expressed_channel(person, params)
    if channel is None or not person.alive:
        return ()
    limit = threshold(person, params)
    integrand = max(0.0, above_usual(person, excess(person)))
    others = math.fsum(person.symptom_load[c] for c in CHANNELS if c is not channel)
    fed = person.symptom_suppression * integrand / (1.0 + params.channel_protection * others / limit)
    fired = []
    for c in CHANNELS:
        load = person.symptom_load[c] * (1.0 - params.symptom_leak_rate) + (fed if c is channel else 0.0)
        person.symptom_load[c] = load
        # Every channel is checked: one that is not fed can still cross when functional level falls (review
        # 2026-10-12), and each crossing is its own onset.
        if not person.symptom_active[c] and load >= limit:
            person.symptom_active[c] = True
            fired.append(c)
        elif person.symptom_active[c] and load < params.symptom_rearm_fraction * limit:
            person.symptom_active[c] = False
    return tuple(fired)


def lock_in_coefficient(x: float, params: EngineParams) -> float:
    """Purpose: how strongly an active symptom of severity x holds the family's anxiety; negative past the turn.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.D.2a, #M7.D.2d
    Tests:   tests/bowen/test_symptoms.py::test_m7d2d_lock_in_rises_to_a_peak_and_turns_over
    """
    return params.lock_in_gain * x * (1.0 - x / params.lock_in_turn)


def _partners(state: RunState, pid: PersonId) -> tuple[PersonId, ...]:
    return tuple(sorted(other for t in state.ties_of(pid) if t.interactive
                        for other in t.id.members() if other != pid and state.people[other].alive))


def _member_partners(state: RunState, pid: PersonId) -> tuple[PersonId, ...]:
    """The living family members on an interactive tie to ``pid``: who a symptom's load and its release reach. An
    external agent (the coach) is passed by, `M1.E.7e`."""
    return tuple(o for o in _partners(state, pid) if state.people[o].role is Role.MEMBER)


def _release_shares(state: RunState, pid: PersonId) -> tuple[tuple[PersonId, float], ...]:
    """Purpose: how what a bearer's symptom held is split when it is released — by the bond energy of the bearer's
    interactive ties to living family members (G2, spec revision 36): a parent takes more than a sibling, an outsider
    none. Ties all at zero bond energy split equally. Returns ``(partner, share)`` with shares summing to 1, or
    nothing when no living member is on an interactive tie, in which case the hold stays with the bearer.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.D.2a, #M7.D.2d, #M1.F.9, #M6.4
    Tests:   tests/bowen/test_symptoms.py::test_m7d2a_release_is_split_by_bond_energy,
             tests/bowen/test_symptoms.py::test_m7d2a_release_passes_an_external_agent_by
    """
    weights = tuple(sorted(
        (other, tie.bond_energy) for tie in state.ties_of(pid) if tie.interactive
        for other in tie.id.members()
        if other != pid and state.people[other].alive and state.people[other].role is Role.MEMBER
    ))
    if not weights:
        return ()
    total = math.fsum(w for _, w in weights)
    if total <= 0:
        return tuple((other, 1.0 / len(weights)) for other, _ in weights)
    return tuple((other, w / total) for other, w in weights if w > 0)  # a tie at zero beside live ones takes nothing


def release_bound(state: RunState, pid: PersonId, share: float = 1.0) -> tuple[tuple[PersonId, float], ...]:
    """Purpose: return ``share`` of what a bearer's symptom holds to the family members on interactive ties, each
    taking the bond energy share of the bearer's tie to them (G2, spec revision 36; equal split before it).
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.D.2a, #M7.D.2d, #M1.F.9, #M6.4
    Tests:   tests/bowen/test_symptoms.py::test_m7d2a_a_symptom_ending_returns_what_it_held,
             tests/bowen/test_symptoms.py::test_m7d2a_release_is_split_by_bond_energy
    """
    person = state.people[pid]
    shares = _release_shares(state, pid)
    amount = share * person.symptom_bound
    if not shares or amount <= 0:
        return ()
    person.symptom_bound -= amount
    for other, weight in shares:
        state.people[other].acute_anxiety += amount * weight
    return tuple((other, amount * weight) for other, weight in shares)


def _lock_in_record(state: RunState, pid: PersonId, moved, bound_change: float, cause=None) -> EffectRecord:
    return EffectRecord(state.tick, "lock_in", cause, acute_anxiety=tuple(moved),
                        people=((pid, "symptom_bound", bound_change),))


def lock_in(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: the family's focus on each active symptom binds its anxiety below the turn and releases it past it.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.D.2a, #M7.D.2d, #M6.4
    Tests:   tests/bowen/test_symptoms.py::test_m7d2a_the_family_calms_around_a_mild_symptom,
             tests/bowen/test_symptoms.py::test_m7d2d_past_the_turn_the_symptom_gives_back_what_it_held
    """
    records = []
    for pid in sorted(state.people):
        person = state.people[pid]
        if not person.alive:
            continue
        active = [c for c in CHANNELS if person.symptom_active[c]]
        if not active:
            if person.symptom_bound > 0:  # the symptom has ended: what it held goes back (M1.F.9's logic)
                moved = release_bound(state, pid)
                if moved:
                    records.append(_lock_in_record(state, pid, moved, -math.fsum(v for _, v in moved)))
            continue
        x = max(person.symptom_load[c] for c in active) / threshold(person, params)
        s = lock_in_coefficient(x, params)
        if s > 0:
            moved = []
            for other in _partners(state, pid):
                partner = state.people[other]
                take = min(s * max(0.0, above_usual(partner, excess(partner))), excess(partner))
                if take > 0:
                    partner.acute_anxiety -= take
                    person.symptom_bound += take
                    moved.append((other, -take))
            if moved:
                records.append(_lock_in_record(state, pid, moved, -math.fsum(v for _, v in moved)))
        elif s < 0:
            share = min(1.0, -s)
            moved = list(release_bound(state, pid, share))
            if moved:
                records.append(_lock_in_record(state, pid, moved, -math.fsum(v for _, v in moved)))
            shares = _release_shares(state, pid)  # G2: the spill is split as the release is, by bond energy
            spill = share * max(0.0, above_usual(person, excess(person)))
            if shares and spill > 0:  # the severe symptom no longer holds the bearer's own anxiety either
                person.acute_anxiety -= spill
                for other, weight in shares:
                    state.people[other].acute_anxiety += spill * weight
                records.append(EffectRecord(state.tick, "lock_in", None,
                                            acute_anxiety=((pid, -spill),) + tuple((o, spill * w) for o, w in shares),
                                            people=((pid, "symptom_bound", 0.0),)))
    return records


def tolerance_removal(state: RunState, params: EngineParams, visibility: HouseholdConductanceVisibility) -> list:
    """Purpose: remove the member carrying the most active symptom load when another member's tolerance gives out.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.D.4, #M1.D.5, #M5.C.1
    Tests:   tests/bowen/test_symptoms.py::test_m7d4_removal_fires_off_the_others_tolerance_not_the_bearers_severity
    """
    family = state.family
    if family.decision_ownership_external or not family.tolerance:
        return []
    bearers = [(math.fsum(p.symptom_load[c] for c in CHANNELS if p.symptom_active[c]), pid)
               for pid, p in sorted(state.people.items()) if p.alive and any(p.symptom_active.values())
               and any(t.interactive for t in state.ties_of(pid))]
    if not bearers:
        return []
    _, bearer = max(bearers)
    for other in _partners(state, bearer):
        limit = family.tolerance.get(other)
        partner = state.people[other]
        if limit is not None and above_usual(partner, excess(partner)) > limit:
            event = Event(id=EventId(state.tick, f"removal:{bearer}", 0), kind=REMOVAL_KIND,
                          mechanism=Mechanism.INSTITUTIONALIZE, sender=other, targets=(bearer,), intensity=0.0,
                          timestamp=state.tick, duration=1, exogenous=False, source_position=SourcePosition.NONE,
                          channel=Channel.ENDOGENOUS)
            from src.bowen.engine.event_effects import apply_structural_event, nodal_spike

            records = inject(state, event, visibility) + nodal_spike(state, event, params)  # M4.C.1c, revision 32
            return records + apply_structural_event(state, state.store.event(event.id), params)
    return []


def _onset(state: RunState, pid: PersonId, channel: SymptomChannel, params: EngineParams,
           visibility: HouseholdConductanceVisibility) -> list:
    """One endogenous SYMPTOM_ONSET from the bearer to everyone on an interactive tie with them (M7.D.1, M1.F.7)."""
    event = Event(
        id=EventId(state.tick, f"symptom:{pid}", list(SymptomChannel).index(channel)),
        kind=SYMPTOM_KIND, mechanism=Mechanism.ENDOGENOUS_SYMPTOM, sender=pid, targets=_partners(state, pid),
        intensity=params.symptom_event_intensity, timestamp=state.tick, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.ENDOGENOUS,
    )
    from src.bowen.engine.event_effects import nodal_spike  # event_effects imports from this module's neighbours

    records = inject(state, event, visibility) + nodal_spike(state, event, params) + [  # M4.C.1c, revision 32
        EffectRecord(state.tick, "symptom_onset", event.id, sinks=((f"symptom_load:{channel.value}", 1.0),))]
    return records + _illness_loads(state, pid, channel, params, visibility)


def _illness_loads(state: RunState, pid: PersonId, channel: SymptomChannel, params: EngineParams,
                   visibility: HouseholdConductanceVisibility) -> list:
    """Purpose: a symptom's onset is also an illness: the bearer carries the scale's own-illness load and each family
    member on an interactive tie the family-illness load at the tie's bond energy share, for the table's weeks
    (M7.D.1a); an external agent on an interactive tie, the coach, takes none, as G2's release passes them by. They arise after this week's step 2, so they arrive at the next week's, as the life course's
    events do (``RunState.arising``); nothing is injected here.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.D.1a, #M1.F.6a
    Tests:   tests/bowen/test_event_sizes.py::test_m7d1a_a_symptom_onset_loads_the_bearer_and_the_family,
             tests/bowen/test_event_sizes.py::test_m7d1a_the_family_illness_load_passes_an_external_agent_by
    """
    origin = f"illness:{pid}:{channel.value}"
    own, weeks = sized_intensity(state.kinds, params, OWN_ILLNESS_KIND, "self")
    events = [_stressor(state, origin, 0, OWN_ILLNESS_KIND, (pid,), own, weeks)]
    for index, other in enumerate(_member_partners(state, pid), start=1):
        tie = state.tie_between(pid, other)
        reach = tie.bond_energy / SCALE_MAX if tie is not None else 0.0
        intensity, weeks = sized_intensity(state.kinds, params, FAMILY_ILLNESS_KIND, "tie", reach=reach)
        if intensity > 0:
            events.append(_stressor(state, origin, index, FAMILY_ILLNESS_KIND, (other,), intensity, weeks))
    state.arising += events
    return []


def _stressor(state: RunState, origin: str, index: int, kind: str, targets, intensity: float, weeks: int) -> Event:
    """A load the engine itself raises at an onset (`M7.D.1a`): the stressor mechanism (`exogenous=True`, as `M1.F.7`
    requires of it) on the `ENDOGENOUS` channel, since its cause is an event of the run and not the world outside
    it. The evaluation readout's stress term (`M11.G.2`) reads the channel and leaves these loads out."""
    tick = state.tick + 1  # due at the next week's step 2
    return Event(id=EventId(tick, origin, index), kind=kind, mechanism=Mechanism.EXOGENOUS_STRESSOR, sender=None,
                 targets=tuple(sorted(targets)), intensity=intensity, timestamp=tick, duration=weeks,
                 exogenous=True, source_position=SourcePosition.NONE, channel=Channel.ENDOGENOUS)


def accumulate_symptoms(
    state: RunState, params: EngineParams, visibility: HouseholdConductanceVisibility
) -> list:
    """Purpose: run the integrator for everyone, emit an endogenous event at each onset, then lock-in and removal.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.D.1, #M1.F.7, #M7.D.2a, #M7.D.4
    Tests:   tests/bowen/test_symptoms.py::test_m7d1_threshold_crossing_emits_endogenous_event
    """
    records: list = []
    loads = []
    for pid in sorted(state.people):
        person = state.people[pid]
        fired = integrate(person, params)
        if person.channel_prior is not None and person.alive:
            loads.append((pid, math.fsum(person.symptom_load.values())))
        for channel in fired:
            records += _onset(state, pid, channel, params, visibility)
    if loads:
        records.append(EffectRecord(state.tick, "symptom_accumulation", None,
                                    people=tuple((pid, "symptom_load", load) for pid, load in loads)))
    records += lock_in(state, params)
    records += tolerance_removal(state, params, visibility)
    return records
