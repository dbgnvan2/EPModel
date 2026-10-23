"""Symptom accumulation and onset (Phase C step 2).

Purpose: test the leaky chronicity integrator and the endogenous event at threshold.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.3, #M4.C.3a, #M7.D.1, #M1.A.6, #M1.A.11b, #M1.F.7
Tests:   this file
"""

from __future__ import annotations

import dataclasses

from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.objects import SymptomChannel
from src.bowen.engine.state import new_run_state
from src.bowen.engine.symptoms import accumulate_symptoms, integrate, threshold
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

MARTA, NADIA = PersonId("marta"), PersonId("nadia")
PARAMS = engine_params(load_constants())
VIS = HouseholdConductanceVisibility(PARAMS.per_hop_fidelity)


def fresh():
    """The reduced family, everyone with a usual level of 0 excess (revision 24 reads the integrand above it)."""
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    state = initialise_run(state, PARAMS)
    for person in state.people.values():
        person.usual_excess = 0.0
    return state


def run_excursions(person, pattern):
    """Feed ``pattern`` (excess per tick) through the integrator; return the peak load."""
    peak = 0.0
    for excess in pattern:
        person.acute_anxiety = person.chronic_anxiety + excess
        integrate(person, PARAMS)
        peak = max(peak, person.symptom_load[person.channel_prior])
    return peak


def test_m4c3a_one_sustained_excursion_builds_more_than_several_resolved():
    """Same total time above the floor: one unresolved excursion outbuilds four resolved ones."""
    state = fresh()
    sustained = dataclasses.replace(state.people[MARTA], symptom_load={c: 0.0 for c in SymptomChannel})
    resolved = dataclasses.replace(state.people[MARTA], symptom_load={c: 0.0 for c in SymptomChannel})
    one = [8.0] * 12 + [0.0] * 12
    four = ([8.0] * 3 + [0.0] * 6) * 4
    assert sum(one) == sum(four)
    assert run_excursions(sustained, one) > run_excursions(resolved, four)


def test_m1a6_threshold_reads_functional_level():
    marta = fresh().people[MARTA]
    higher = dataclasses.replace(marta, functional_level=marta.functional_level + 10)
    assert threshold(higher, PARAMS) > threshold(marta, PARAMS)
    assert threshold(dataclasses.replace(marta, basic_level=marta.basic_level + 10), PARAMS) == threshold(marta, PARAMS)


def test_m1a11b_load_accumulates_in_the_constitutional_channel():
    nadia = fresh().people[NADIA]
    nadia.acute_anxiety = nadia.chronic_anxiety + 5
    integrate(nadia, PARAMS)
    loaded = {c for c, v in nadia.symptom_load.items() if v > 0}
    assert loaded == {nadia.channel_prior}


def test_m7d1_threshold_crossing_emits_endogenous_event():
    """Once over threshold: one endogenous event to the bearer's interactive ties; re-armed only well below."""
    state = fresh()
    marta = state.people[MARTA]
    marta.acute_anxiety = marta.chronic_anxiety + 2 * threshold(marta, PARAMS)
    records = accumulate_symptoms(state, PARAMS, VIS)
    [event] = [e for e in state.store.events() if e.mechanism is Mechanism.ENDOGENOUS_SYMPTOM]
    assert event.sender == MARTA and not event.exogenous and event.kind == "SYMPTOM_ONSET"
    partners = {o for t in state.ties_of(MARTA) if t.interactive for o in t.id.members() if o != MARTA}
    assert set(event.targets) == partners
    assert any(r.mechanism == "symptom_onset" for r in records if hasattr(r, "mechanism"))
    state.tick += 1
    accumulate_symptoms(state, PARAMS, VIS)  # still above: no second event
    assert len([e for e in state.store.events() if e.mechanism is Mechanism.ENDOGENOUS_SYMPTOM]) == 1
    marta.acute_anxiety = marta.chronic_anxiety
    marta.symptom_load[marta.channel_prior] = 0.0
    state.tick += 1
    accumulate_symptoms(state, PARAMS, VIS)
    assert not marta.symptom_active[marta.channel_prior]


# --- Phase D B7, spec revision 24 (Claude's recommendation, pending owner review) ---

import pytest

from src.bowen.engine.event_effects import apply_structural_event
from src.bowen.engine.events import BinderKind, BinderRef, Channel, Event, EventId, SourcePosition
from src.bowen.engine.invariants import ledger_problems, snapshot
from src.bowen.engine.symptoms import expressed_channel, lock_in, lock_in_coefficient, release_bound, tolerance_removal

RAVI, PIA = PersonId("ravi"), PersonId("pia")

def calm(state):
    """Everyone at their floor and their usual level, so only what a test sets moves anything."""
    for person in state.people.values():
        person.acute_anxiety = person.chronic_anxiety
        person.usual_excess = 0.0
    return state


def bearer(state, pid, severity):
    """``pid`` carries an active symptom of the given severity (load ÷ threshold) in their prior channel."""
    person = state.people[pid]
    person.symptom_load[person.channel_prior] = severity * threshold(person, PARAMS)
    person.symptom_active[person.channel_prior] = True
    return person


def test_m4c3a_the_integrand_reads_above_the_usual_level():
    """Revision 24: a steady excess at the person's usual level builds nothing; the same excess above it does."""
    state = fresh()
    marta = state.people[MARTA]
    marta.acute_anxiety = marta.chronic_anxiety + 20
    marta.usual_excess = 20.0
    integrate(marta, PARAMS)
    assert marta.symptom_load[marta.channel_prior] == 0.0
    marta.usual_excess = 10.0
    integrate(marta, PARAMS)
    assert marta.symptom_load[marta.channel_prior] == pytest.approx(10.0)


def test_m1a11c_a_family_focus_moves_the_channel_and_constitution_overrides_it():
    """M1.A.11b/c: the prior with no focus; a focus moves it; a constitution strength of 1 overrides any focus."""
    nadia = fresh().people[NADIA]
    assert expressed_channel(nadia, PARAMS) is nadia.channel_prior is SymptomChannel.MENTAL
    nadia.channel_focus = {SymptomChannel.SOCIAL: 1.0}  # at the default strength 0.5, a focus of 1 only ties
    assert expressed_channel(nadia, PARAMS) is SymptomChannel.MENTAL
    nadia.channel_focus = {SymptomChannel.SOCIAL: 2.0}
    assert expressed_channel(nadia, PARAMS) is SymptomChannel.SOCIAL
    nadia.constitution_strength = 1.0
    assert expressed_channel(nadia, PARAMS) is SymptomChannel.MENTAL
    nadia.constitution_strength = 0.0
    nadia.channel_focus = {}
    assert expressed_channel(nadia, PARAMS) is SymptomChannel.MENTAL  # a tie goes to the prior
    # The load follows the expressed channel (M7.D.2: the same deficit can present as any of the three).
    nadia.constitution_strength, nadia.channel_focus = 0.2, {SymptomChannel.PHYSICAL: 1.0}
    nadia.acute_anxiety, nadia.usual_excess = nadia.chronic_anxiety + 5, 0.0
    integrate(nadia, PARAMS)
    assert {c for c, v in nadia.symptom_load.items() if v > 0} == {SymptomChannel.PHYSICAL}


def weeks_to_onset(person, channel, params=PARAMS):
    for week in range(1, 200):
        if channel in integrate(person, params):
            return week
    return None


def test_m7d2c_load_in_one_channel_slows_onset_in_another():
    """M7.D.2c: at equal anxiety, load carried in one channel delays onset in a second; with no protection it does not."""
    import dataclasses

    def person_with(load_elsewhere, params):
        nadia = fresh().people[NADIA]
        nadia.constitution_strength, nadia.channel_focus = 0.0, {SymptomChannel.SOCIAL: 1.0}
        nadia.acute_anxiety, nadia.usual_excess = nadia.chronic_anxiety + 12, 0.0
        nadia.symptom_load[SymptomChannel.MENTAL] = load_elsewhere * threshold(nadia, params)
        return nadia

    clear = weeks_to_onset(person_with(0.0, PARAMS), SymptomChannel.SOCIAL)
    loaded = weeks_to_onset(person_with(0.4, PARAMS), SymptomChannel.SOCIAL)
    assert clear is not None and (loaded is None or loaded > clear)
    unprotected = dataclasses.replace(PARAMS, channel_protection=0.0)
    assert weeks_to_onset(person_with(0.4, unprotected), SymptomChannel.SOCIAL, unprotected) == clear


def test_m7d2d_lock_in_rises_to_a_peak_and_turns_over():
    """M7.D.2d: the stabilising term rises with severity to a peak at half the turn, is zero at the turn, negative past it."""
    turn = PARAMS.lock_in_turn
    peak = lock_in_coefficient(turn / 2, PARAMS)
    assert lock_in_coefficient(0.0, PARAMS) == 0.0
    assert 0 < lock_in_coefficient(turn / 4, PARAMS) < peak and 0 < lock_in_coefficient(3 * turn / 4, PARAMS) < peak
    assert lock_in_coefficient(turn, PARAMS) == pytest.approx(0.0)
    assert lock_in_coefficient(1.5 * turn, PARAMS) < 0


def test_m7d2a_the_family_calms_around_a_mild_symptom():
    """M7.D.2a: a partner above their usual level gives part of it to a mild symptom's hold; the ledger balances."""
    state = calm(fresh())
    marta = bearer(state, MARTA, PARAMS.lock_in_turn / 2)
    ravi = state.people[RAVI]
    ravi.acute_anxiety += 10
    before, was = snapshot(state), ravi.acute_anxiety
    records = lock_in(state, PARAMS)
    assert ravi.acute_anxiety < was and marta.symptom_bound == pytest.approx(was - ravi.acute_anxiety)
    assert ledger_problems(state, before, records, 1e-9) == []
    # A partner at their usual level gives nothing: the hold reads the excess above it.
    state = calm(fresh())
    bearer(state, MARTA, PARAMS.lock_in_turn / 2)
    lock_in(state, PARAMS)
    assert state.people[MARTA].symptom_bound == 0.0


def test_m7d2d_past_the_turn_the_symptom_gives_back_what_it_held():
    """M7.D.2d: past the turn the hold is released to the partners, and the ledger balances."""
    state = calm(fresh())
    marta = bearer(state, MARTA, 1.5 * PARAMS.lock_in_turn)
    marta.symptom_bound = 8.0
    marta.acute_anxiety += 10  # and the severe symptom no longer holds the bearer's own excess
    before, was = snapshot(state), marta.acute_anxiety
    records = lock_in(state, PARAMS)
    assert marta.symptom_bound < 8.0 and marta.acute_anxiety < was
    partners = {o for t in state.ties_of(MARTA) if t.interactive for o in t.id.members() if o != MARTA}
    assert all(state.people[p].acute_anxiety > state.people[p].chronic_anxiety for p in partners)
    assert ledger_problems(state, before, records, 1e-9) == []


def test_m7d2a_a_symptom_ending_returns_what_it_held():
    """When the symptom ends, or a binder_unavailable names it (M1.F.9), the whole hold goes back to the partners."""
    state = calm(fresh())
    marta = state.people[MARTA]
    marta.symptom_bound = 6.0  # no active symptom: it has ended
    records = lock_in(state, PARAMS)
    assert marta.symptom_bound == 0.0
    assert sum(v for r in records for _, v in r.acute_anxiety) == pytest.approx(6.0)

    state = calm(fresh())
    marta = bearer(state, MARTA, 1.0)
    marta.symptom_bound = 6.0
    event = Event(id=EventId(0, "remit", 0), kind="BINDER_UNAVAILABLE", mechanism=Mechanism.BINDER_UNAVAILABLE,
                  sender=None, targets=(), intensity=0.0, timestamp=0, duration=1, exogenous=True,
                  source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS,
                  binder=BinderRef(BinderKind.SYMPTOM_CHANNEL, f"marta:{marta.channel_prior.value}"))
    before, budget = snapshot(state), state.family.undifferentiation_budget
    records = apply_structural_event(state, event, PARAMS)
    assert marta.symptom_bound == 0.0 and not marta.symptom_active[marta.channel_prior]
    # Review 2026-10-12: the load is an integral, not anxiety, and is not credited to the budget.
    assert state.family.undifferentiation_budget == budget and marta.symptom_load[marta.channel_prior] == 0.0
    assert [r.mechanism for r in records] == ["binder_unavailable", "lock_in"]
    assert ledger_problems(state, before, [r for r in records if r.mechanism == "lock_in"], 1e-9) == []


def test_m7d2a_removal_stops_the_hold():
    """INSTITUTIONALIZE makes the bearer's ties non-interactive, so lock-in neither takes nor gives back."""
    state = calm(fresh())
    marta = bearer(state, MARTA, PARAMS.lock_in_turn / 2)
    state.people[RAVI].acute_anxiety += 10
    for tie in state.ties_of(MARTA):
        tie.interactive = False
    assert lock_in(state, PARAMS) == [] and marta.symptom_bound == 0.0


def test_m7d4_removal_fires_off_the_others_tolerance_not_the_bearers_severity():
    """M7.D.4: removal fires when a remaining member's excess passes their tolerance, at any severity; decision
    ownership held outside the family suppresses it; no tolerance listed, never."""
    def run(severity, ravi_excess, external=False, tolerance=True):
        state = calm(fresh())
        bearer(state, NADIA, severity)
        state.people[RAVI].acute_anxiety += ravi_excess
        state.family.tolerance = {RAVI: 5.0} if tolerance else {}
        state.family.decision_ownership_external = external
        records = tolerance_removal(state, PARAMS, VIS)
        removed = any(getattr(r, "mechanism", None) == "institutionalize" for r in records)
        return removed, state

    assert run(1.0, 10)[0] and run(5.0, 10)[0]          # severity does not decide it
    assert not run(5.0, 2)[0]                            # the others' tolerance does
    assert not run(1.0, 10, external=True)[0]            # M5.C.1: decision ownership held outside
    assert not run(1.0, 10, tolerance=False)[0]
    removed, state = run(1.0, 10)
    assert not any(t.interactive for t in state.ties_of(NADIA))
    [event] = [e for e in state.store.events() if e.mechanism is Mechanism.INSTITUTIONALIZE]
    assert event.sender == RAVI and event.targets == (NADIA,) and not event.exogenous


# --- review of revision 24 (2026-10-12): mutants that survived the tests above ---


def test_m7d2a_a_partner_gives_only_what_is_above_their_usual_level():
    """The hold takes s × the partner's excess above usual, not × their excess."""
    state = calm(fresh())
    marta = bearer(state, MARTA, PARAMS.lock_in_turn / 2)
    ravi = state.people[RAVI]
    ravi.acute_anxiety, ravi.usual_excess = ravi.chronic_anxiety + 10, 6.0
    lock_in(state, PARAMS)
    s = lock_in_coefficient(marta.symptom_load[marta.channel_prior] / threshold(marta, PARAMS), PARAMS)
    assert marta.symptom_bound == pytest.approx(s * 4.0)


def test_m7d4_removal_reads_the_excess_above_usual():
    """A partner whose excess is high but is their usual level does not fire removal."""
    state = calm(fresh())
    bearer(state, NADIA, 1.0)
    ravi = state.people[RAVI]
    ravi.acute_anxiety, ravi.usual_excess = ravi.chronic_anxiety + 10, 8.0
    state.family.tolerance = {RAVI: 5.0}
    assert tolerance_removal(state, PARAMS, VIS) == []


def test_m7d4_a_bearer_already_removed_does_not_block_another():
    """A removed bearer (no interactive tie) is not chosen again, so another bearer can still be removed."""
    state = calm(fresh())
    bearer(state, NADIA, 1.0)
    bearer(state, PIA, 5.0)  # the larger load, but already removed
    for tie in state.ties_of(PIA):
        tie.interactive = False
    state.people[RAVI].acute_anxiety += 10
    state.family.tolerance = {RAVI: 5.0}
    tolerance_removal(state, PARAMS, VIS)
    [event] = [e for e in state.store.events() if e.mechanism is Mechanism.INSTITUTIONALIZE]
    assert event.targets == (NADIA,)


def test_m7d2_every_channel_leaks_and_re_arms_on_its_own():
    """Load in a channel that is not fed still leaks, and its symptom ends when it falls below re-arm."""
    nadia = fresh().people[NADIA]
    other = SymptomChannel.SOCIAL
    assert expressed_channel(nadia, PARAMS) is not other
    nadia.acute_anxiety, nadia.usual_excess = nadia.chronic_anxiety, 0.0
    nadia.symptom_load[other] = 10.0
    integrate(nadia, PARAMS)
    assert nadia.symptom_load[other] == pytest.approx(10.0 * (1 - PARAMS.symptom_leak_rate))
    nadia.symptom_active[other] = True
    nadia.symptom_load[other] = 0.1 * PARAMS.symptom_rearm_fraction * threshold(nadia, PARAMS)
    integrate(nadia, PARAMS)
    assert not nadia.symptom_active[other]


def test_m7d2_a_channel_that_is_not_fed_can_still_cross_and_each_crossing_is_reported():
    """Review 2026-10-12: when functional level falls, load left in an old channel can cross; both onsets fire."""
    nadia = fresh().people[NADIA]
    nadia.acute_anxiety, nadia.usual_excess = nadia.chronic_anxiety + 50, 0.0
    for c in (SymptomChannel.SOCIAL, nadia.channel_prior):
        nadia.symptom_load[c] = 0.99 * threshold(nadia, PARAMS) / (1 - PARAMS.symptom_leak_rate)
    nadia.functional_level -= 5  # lowers the threshold under both
    assert set(integrate(nadia, PARAMS)) == {SymptomChannel.SOCIAL, nadia.channel_prior}


def test_m7d2d_past_the_turn_the_release_is_a_share_not_all():
    """Past the turn min(1, |s|) of the hold goes back each week: at s = −0.45 a hold of 8 keeps 4.4."""
    state = calm(fresh())
    x = 1.5 * PARAMS.lock_in_turn
    marta = bearer(state, MARTA, x)
    marta.symptom_bound = 8.0
    lock_in(state, PARAMS)
    share = min(1.0, -lock_in_coefficient(x, PARAMS))
    assert 0 < share < 1 and marta.symptom_bound == pytest.approx(8.0 * (1 - share))


def test_m64_an_unlogged_change_to_a_symptoms_hold_is_caught():
    """The ledger's per-person check on symptom_bound: a change no record logs is reported."""
    state = calm(fresh())
    before = snapshot(state)
    state.people[MARTA].symptom_bound += 3.0
    assert any("symptom-bound" in p for p in ledger_problems(state, before, [], 1e-9))


def test_m1a6_a_threshold_is_never_zero():
    """Review 2026-10-12: floored like steepness, so severity and protection never divide by zero."""
    marta = fresh().people[MARTA]
    marta.functional_level = 0.0
    assert threshold(marta, PARAMS) > 0
    marta.acute_anxiety, marta.usual_excess = marta.chronic_anxiety + 5, 0.0
    assert integrate(marta, PARAMS) == (marta.channel_prior,)  # before the floor this divided by zero


def test_m4c1c_a_removal_spikes_the_cut_ties_of_those_it_reaches():
    """Revision 32: removal by tolerance is a nodal event (INSTITUTIONALIZE, reach "ties"), and its own call site
    spikes the cut ties of those it reaches (review 2026-10-17: that call site had no test)."""
    from src.bowen.engine.rejection import raise_rejection

    state = calm(fresh())
    bearer(state, NADIA, 1.0)
    state.people[RAVI].acute_anxiety += 10
    state.family.tolerance = {RAVI: 5.0}
    raise_rejection(state.ties[TieId.of(RAVI, MARTA)], MARTA, 0.5)
    records = tolerance_removal(state, PARAMS, VIS)
    spiked = {t for r in records if getattr(r, "mechanism", None) == "nodal_spike" for t, _, _ in r.ties}
    assert TieId.of(RAVI, MARTA) in spiked


# --- G2: release by tie (spec revision 36) -----------------------------------------------------------


def test_m7d2a_release_is_split_by_bond_energy():
    """B1–B3: what a symptom held goes back in proportion to the bond energy of the bearer's ties to living members:
    Marta's hold goes to Ana (40), Nadia (50), Pia (50) and Ravi (60) in those proportions, not a quarter each."""
    state = calm(fresh())
    marta = state.people[MARTA]
    marta.symptom_bound = 200.0
    records = lock_in(state, PARAMS)
    got = {pid: v for r in records for pid, v in r.acute_anxiety}
    bonds = {other: state.tie_between(MARTA, other).bond_energy for other in got}
    total = sum(bonds.values())
    assert marta.symptom_bound == 0.0 and sum(got.values()) == pytest.approx(200.0)
    assert got == {other: pytest.approx(200.0 * bonds[other] / total) for other in got}
    assert got[RAVI] > got[NADIA] == got[PIA] > got[PersonId("ana")]


def test_m7d2a_release_ignores_dead_and_non_interactive_partners_and_zero_bonds_split_equally():
    """B4, B5 and the dirty-state case: a dead partner and a non-interactive tie take nothing; with every live tie at
    zero bond energy the split is equal; a run that releases twice splits by the bond energy of each week."""
    state = calm(fresh())
    marta = state.people[MARTA]
    state.people[RAVI].alive = False
    state.tie_between(MARTA, PIA).interactive = False
    marta.symptom_bound = 90.0
    got = {pid: v for r in lock_in(state, PARAMS) for pid, v in r.acute_anxiety}
    assert set(got) == {NADIA, PersonId("ana")} and got[NADIA] == pytest.approx(50.0) and got[PersonId("ana")] == pytest.approx(40.0)
    zero = calm(fresh())
    for tie in zero.ties_of(MARTA):
        tie.bond_energy = 0.0
    zero.people[MARTA].symptom_bound = 40.0
    got = {pid: v for r in lock_in(zero, PARAMS) for pid, v in r.acute_anxiety}
    assert len(got) == 4 and all(v == pytest.approx(10.0) for v in got.values())
    state = calm(fresh())
    state.people[MARTA].symptom_bound = 100.0
    first = {pid: v for r in lock_in(state, PARAMS) for pid, v in r.acute_anxiety}
    state.tie_between(MARTA, RAVI).bond_energy = 0.0  # next week the marital tie holds nothing
    state.people[MARTA].symptom_bound = 100.0
    second = {pid: v for r in lock_in(state, PARAMS) for pid, v in r.acute_anxiety}
    assert first[RAVI] > 0 and RAVI not in second and second[NADIA] > first[NADIA]


def test_m7d2a_release_passes_an_external_agent_by():
    """B6: an external agent (M1.A.17) on an interactive tie receives none of the release."""
    from src.bowen.engine.objects import Role
    state = calm(fresh())
    sofia = state.people[PersonId("sofia")]
    sofia.role = Role.EXTERNAL
    state.people[RAVI].symptom_bound = 100.0
    got = {pid: v for r in lock_in(state, PARAMS) for pid, v in r.acute_anxiety}
    assert PersonId("sofia") not in got and sum(got.values()) == pytest.approx(100.0)
    kept = calm(fresh())
    for pid in kept.people:
        if pid != RAVI:
            kept.people[pid].role = Role.EXTERNAL
    kept.people[RAVI].symptom_bound = 100.0
    assert lock_in(kept, PARAMS) == [] and kept.people[RAVI].symptom_bound == 100.0  # no receiver: the hold stays


def test_m7d2a_the_spill_past_the_turn_is_split_by_bond_energy():
    """B7: the bearer's own spill past the turn follows the same split as the release."""
    state = calm(fresh())
    marta = bearer(state, MARTA, PARAMS.lock_in_turn * 1.5)
    marta.acute_anxiety = marta.chronic_anxiety + 40.0
    records = [r for r in lock_in(state, PARAMS) if any(pid == MARTA and v < 0 for pid, v in r.acute_anxiety)]
    assert records
    got = {pid: v for pid, v in records[-1].acute_anxiety if pid != MARTA}
    assert got[RAVI] > got[NADIA] == got[PIA] > got[PersonId("ana")] > 0


def test_m7d2a_intake_is_unchanged_by_the_split():
    """G2 changes the release only: what partners give into the hold is still each partner's own excess share."""
    state = calm(fresh())
    marta = bearer(state, MARTA, 1.0)
    for pid in (RAVI, NADIA):
        state.people[pid].acute_anxiety = state.people[pid].chronic_anxiety + 10.0
    got = {pid: -v for r in lock_in(state, PARAMS) for pid, v in r.acute_anxiety}
    assert got[RAVI] == pytest.approx(got[NADIA]) and marta.symptom_bound == pytest.approx(got[RAVI] + got[NADIA])


def test_m7d2a_binder_unavailable_release_uses_the_same_split():
    """B3: a `binder_unavailable` naming the channel (M1.F.9) returns the whole hold by the same bond-energy split
    as a release at re-arm; the coach takes nothing and the family's gains sum to what left the hold."""
    state = calm(fresh())
    marta = state.people[MARTA]
    marta.symptom_bound = 90.0
    event = Event(id=EventId(0, "script", 0), kind="BINDER_UNAVAILABLE", mechanism=Mechanism.BINDER_UNAVAILABLE,
                  sender=None, targets=(MARTA,), intensity=0.0, timestamp=0, duration=1, exogenous=True,
                  source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
                  binder=BinderRef(BinderKind.SYMPTOM_CHANNEL, f"marta:{marta.channel_prior.value}"))
    records = apply_structural_event(state, event, PARAMS)
    [release] = [r for r in records if r.mechanism == "lock_in"]
    got = dict(release.acute_anxiety)
    weights = {t.id.other(MARTA): t.bond_energy for t in state.ties_of(MARTA)
               if t.interactive and state.people[t.id.other(MARTA)].alive}
    assert marta.symptom_bound == 0.0 and sum(got.values()) == pytest.approx(90.0)
    assert PersonId("halim") not in got and len(got) == len(weights)
    for pid, value in got.items():
        assert value == pytest.approx(90.0 * weights[pid] / sum(weights.values()))
