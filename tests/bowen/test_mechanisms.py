"""The Phase B mechanisms, one at a time, on the Phase B family.

Purpose: prove each step-8 mechanism's direction of effect and its constraints.
         Comparisons are between two arms and assert a direction, not a magnitude (M0.4).
Spec:    docs/bowen_agent_model_spec_v2.md#M4.A, #M4.B.1, #M4.C.1, #M4.E.1, #M4.G, #M1.F.2-#M1.F.9, #M1.A.8, #M1.A.12, #M1.B.3, #M1.B.4, #M1.C.3, #M6
Tests:   this file
"""

from __future__ import annotations

import dataclasses

import pytest

from src.bowen.engine.act import Selection, act, inject, record_deliveries
from src.bowen.engine.appraise import appraisal_delta, apply_appraisal, perceive
from src.bowen.engine.consolidate import consolidate
from src.bowen.engine.event_effects import apply_delivered_cutoffs, apply_structural_event
from src.bowen.engine.events import (
    BinderKind, BinderRef, Channel, Delivery, Event, EventId, EventKinds, Mechanism, Role, SourcePosition,
)
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.invariants import InvariantViolation, assert_invariants, snapshot
from src.bowen.engine.log_records import InvariantStatus
from src.bowen.engine.objects import TieState
from src.bowen.engine.recompute import is_member, recompute_involvement, recompute_triangles
from src.bowen.engine.standing_load import apply_standing_load, self_term, tie_term
from src.bowen.engine.contact import relax_contact
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.state import new_run_state
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, NADIA, PIA, ANA, SOFIA, BRUNO = (P(n) for n in ("ravi", "marta", "nadia", "pia", "ana", "sofia", "bruno"))
ANA_BRUNO = TieId.of(ANA, BRUNO)
PARAMS = engine_params(load_constants())
VIS = HouseholdConductanceVisibility(PARAMS.per_hop_fidelity)


def fresh(kinds: EventKinds | None = None):
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, kinds or load_event_kinds())
    return initialise_run(state, PARAMS)


def scripted(kind, tick, targets=(), *, on_tie=None, intensity=1.0, duration=1, binder=None, index=0):
    mechanism = load_event_kinds().mechanism_of(kind)
    return Event(
        id=EventId(tick, f"script:{index}", 0), kind=kind, mechanism=mechanism, sender=None,
        targets=tuple(sorted(targets)), intensity=intensity, timestamp=tick, duration=duration, exogenous=True,
        source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS, on_tie=on_tie, binder=binder,
    )


def deliver_now(state, event):
    """Record an event and hand its deliveries to perception, as tick steps 2–3 do."""
    resolved = VIS.resolve(event, state.people, state.ties)
    state.store.record_event(resolved.event)
    batch = tuple(d for d in resolved.deliveries)
    for d in batch:
        state.store.record_delivery(d)
    return resolved.event, batch


def acute(state, person):
    return state.people[person].acute_anxiety


# --- M0.3 ------------------------------------------------------------------------------


def test_m03_engine_params_come_from_the_register():
    constants = load_constants()
    for field in dataclasses.fields(PARAMS):
        assert getattr(PARAMS, field.name) == constants[field.name]
        assert constants.values[field.name].grade == "[I]"


# --- M4.A: the standing load ---------------------------------------------------------


def test_m4a1_every_tie_loads_every_tick_without_events():
    state = fresh()
    before = {p: acute(state, p) for p in state.people}
    _, loaded = apply_standing_load(state, PARAMS)
    assert loaded == frozenset(state.ties)
    assert all(acute(state, p) > before[p] for p in state.people)
    assert state.store.events() == ()


def test_m4a1_load_rises_with_bond_energy_and_falls_with_functional_level():
    person, tie = fresh().people[ANA], fresh().ties[ANA_BRUNO]
    assert tie_term(person, dataclasses.replace(tie, bond_energy=80), 0, PARAMS) > tie_term(person, tie, 0, PARAMS)
    assert tie_term(dataclasses.replace(person, functional_level=60), tie, 0, PARAMS) < tie_term(person, tie, 0, PARAMS)


def test_m4a2_trigger_spikes_standing_load_without_contact():
    """Unit form of gate G3: the spike reaches Ana through the standing term, with no delivery."""
    quiet, triggered = fresh(), fresh()
    for state in (quiet, triggered):
        state.tick = 10
    trigger = scripted("TRIGGER", 10, on_tie=ANA_BRUNO, intensity=1.0)
    inject(triggered, trigger, VIS)
    apply_structural_event(triggered, triggered.store.event(trigger.id), PARAMS)
    for state in (quiet, triggered):
        state.tick = 11  # applied at step 2 of week 10, the spike lands at step 1 of week 11
    apply_standing_load(quiet, PARAMS)
    apply_standing_load(triggered, PARAMS)
    assert acute(triggered, ANA) > acute(quiet, ANA)
    assert acute(triggered, BRUNO) > acute(quiet, BRUNO)
    assert triggered.queue.pending() == 0 and triggered.store.delivered_to(ANA) == ()


def test_m4a3_reconciliation_converts_standing_to_interaction_load():
    state = fresh()
    state.tick = 3
    cut_off_load = tie_term(state.people[ANA], state.ties[ANA_BRUNO], 0, PARAMS)
    apply_structural_event(state, scripted("RECONCILIATION", 3, on_tie=ANA_BRUNO), PARAMS)
    assert state.ties[ANA_BRUNO].interactive
    assert tie_term(state.people[ANA], state.ties[ANA_BRUNO], 0, PARAMS) < cut_off_load


def test_m4a4_institutionalize_makes_worry_edges():
    state = fresh()
    bonds = {t.id: t.bond_energy for t in state.ties_of(NADIA)}
    loads = {t.id: tie_term(state.people[RAVI], t, 0, PARAMS) for t in state.ties_of(NADIA) if RAVI in t.id.members()}
    apply_structural_event(state, scripted("INSTITUTIONALIZE", 0, targets=(NADIA,)), PARAMS)
    for tie in state.ties_of(NADIA):
        assert not tie.interactive and tie.bond_energy == bonds[tie.id]
    # Revision 11 (M4.C.1c): a worry edge's load builds as its contact relaxes toward none.
    for _ in range(10):
        relax_contact(state.people, state.ties, PARAMS)
    for tie_id, load in loads.items():
        assert tie_term(state.people[RAVI], state.ties[tie_id], 0, PARAMS) > load
    assert len(state.ties) == 8  # no tie removed (M6.I.7)


def test_m4a5_self_generated_load_derives_from_basic_level():
    state = fresh()
    ravi = state.people[RAVI]
    lower = dataclasses.replace(ravi, basic_level=20.0, functional_level=20.0)
    assert self_term(lower, PARAMS) > self_term(ravi, PARAMS) > 0
    # A person with no ties still loads.
    alone = new_run_state({RAVI: ravi}, {}, state.family, state.kinds)
    start = acute(alone, RAVI)
    apply_standing_load(alone, PARAMS)
    assert acute(alone, RAVI) > start


def test_m4a5_self_generated_load_does_not_swamp_tie_load():
    """Owner's reading of "must not swamp M4.A.1" (2026-10-06): the family as a whole, and
    each member of the nuclear household, whose dynamics M11.C.1 tests.

    A peripheral agent with a single tie can carry a self term above that one tie's
    load — Sofia does on the Phase B family — and that is recorded, not hidden.
    """
    family, state = load_family(), fresh()

    def tie_load(pid):
        return sum(tie_term(state.people[pid], t, 0, PARAMS) for t in state.ties_of(pid))

    total_self = sum(self_term(p, PARAMS) for p in state.people.values())
    assert total_self < sum(tie_load(p) for p in state.people)
    nuclear = [p for p in state.people if state.people[p].household_id == family.nuclear_household]
    assert len(nuclear) == 4
    for pid in nuclear:
        assert self_term(state.people[pid], PARAMS) < tie_load(pid), pid
    exceeding = sorted(p.value for p in state.people if self_term(state.people[p], PARAMS) >= tie_load(p))
    assert exceeding == ["sofia"]  # the recorded exception; a change here should be looked at


# --- M4.B / M4.C.1 / M1.F.2-.5, .8 ---------------------------------------------------


def _conflict(state, sender=RAVI, target=MARTA, intensity=5.0, **kwargs):
    act(state, Selection(sender, "CONFLICT", (target,), intensity, **kwargs), VIS)
    return state.queue.release(state.tick + 1)


def test_m4b1_person_reads_addressed_and_witnessed_events():
    state = fresh()
    batch = _conflict(state)
    state.tick = 1
    record_deliveries(state, batch)
    perceived = perceive(state, batch)
    assert [d.role for d, _ in perceived[MARTA]] == [Role.TARGET]
    assert [d.role for d, _ in perceived[NADIA]] == [Role.WITNESS]


def test_m4c1_delivered_event_raises_receiver_anxiety():
    """A conflict strong enough to push impingement past the band is a load (revision 11 form)."""
    quiet, hit = fresh(), fresh()
    batch = _conflict(hit, intensity=200.0)
    hit.tick = 1
    record_deliveries(hit, batch)
    apply_appraisal(hit, perceive(hit, batch), PARAMS)
    assert acute(hit, MARTA) > acute(quiet, MARTA)
    assert acute(hit, NADIA) > acute(quiet, NADIA)  # M1.F.5: witnesses appraise too


def _delta(event, kinds=None):
    state = fresh(kinds)
    state.store.record_event(event)
    d = Delivery(1, event.id, MARTA, Role.TARGET, emitted_tick=0, latency=1)
    return appraisal_delta(state, d, event, PARAMS)


def _move(**overrides):
    values = dict(
        id=EventId(0, "ravi", 0), kind="CONFLICT", mechanism=Mechanism.MOVE, sender=RAVI, targets=(MARTA,),
        intensity=5.0, timestamp=0, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
    )
    values.update(overrides)
    return Event(**values)


def test_m1f2_source_position_can_flip_sign():
    kinds = load_event_kinds()
    flipped = EventKinds(kinds.mechanisms, {("CONFLICT", SourcePosition.OUTSIDE): -1}, kinds.components)
    strong = dict(intensity=200.0)
    assert _delta(_move(source_position=SourcePosition.OUTSIDE, **strong), flipped) < 0 < _delta(_move(**strong), flipped)


def test_m1f3_route_through_neutral_third_damps():
    """A neutral third on the route shrinks the appraisal, whichever side of the optimum it lands."""
    for intensity in (5.0, 200.0):  # relief at low intensity, a load at high
        assert abs(_delta(_move(route=(ANA,), intensity=intensity))) < abs(_delta(_move(intensity=intensity)))


def test_m1f4_appraisal_attenuated_by_fidelity():
    for intensity in (5.0, 200.0):
        assert abs(_delta(_move(fidelity=0.5, intensity=intensity))) < abs(_delta(_move(fidelity=1.0, intensity=intensity)))


def test_m1f8_same_tick_batch_order_does_not_change_state():
    def run(order):
        state = fresh()
        act(state, Selection(RAVI, "CONFLICT", (MARTA,), 5.0), VIS)
        act(state, Selection(NADIA, "PURSUE", (MARTA,), 3.0), VIS)
        act(state, Selection(PIA, "CONFLICT", (RAVI,), 2.0), VIS)
        batch = state.queue.release(1)
        state.tick = 1
        record_deliveries(state, batch)
        apply_appraisal(state, perceive(state, tuple(order(batch))), PARAMS)
        return {p: acute(state, p) for p in sorted(state.people)}

    assert run(lambda b: b) == run(lambda b: tuple(reversed(b)))


def test_m1f9_binder_unavailable_returns_held_anxiety_to_budget():
    state = fresh()
    state.ties[TieId.of(RAVI, MARTA)].distance_bound_anxiety = 7.0
    budget = state.family.undifferentiation_budget
    event = scripted("BINDER_UNAVAILABLE", 0, targets=(MARTA, RAVI),
                     binder=BinderRef(BinderKind.TIE_DISTANCE, str(TieId.of(RAVI, MARTA))))
    [record] = apply_structural_event(state, event, PARAMS)
    assert state.family.undifferentiation_budget == budget + 7.0
    assert state.ties[TieId.of(RAVI, MARTA)].distance_bound_anxiety == 0.0
    assert record.sinks == (("undifferentiation_budget", 7.0),)


# --- M1.B.3, M1.B.4, M4.G.1 ------------------------------------------------------------


def test_m1b3_four_tie_states_are_distinct():
    """Cut off and resolved low-contact both carry no events; only energy separates them."""
    state = fresh()
    ana = state.people[ANA]
    cut_off = state.ties[ANA_BRUNO]
    resolved = dataclasses.replace(cut_off, tie_state=TieState.RESOLVED_LOW_CONTACT, interactive=True, bond_energy=10.0)
    assert tie_term(ana, cut_off, 0, PARAMS) > tie_term(ana, resolved, 0, PARAMS)
    # A delivered CUTOFF makes a tie cut off and non-interactive, and keeps it.
    act(state, Selection(RAVI, "CUTOFF", (SOFIA,), 4.0), VIS)
    batch = state.queue.release(1) + state.queue.release(2)
    state.tick = 2
    record_deliveries(state, batch)
    apply_delivered_cutoffs(state, batch, PARAMS)
    tie = state.ties[TieId.of(RAVI, SOFIA)]
    assert tie.tie_state is TieState.CUT_OFF and not tie.interactive and tie.bond_energy == 40


def test_m1b4_reunion_restores_coupling_immediately():
    state = fresh()
    bond = state.ties[ANA_BRUNO].bond_energy
    apply_structural_event(state, scripted("RECONCILIATION", 0, on_tie=ANA_BRUNO), PARAMS)
    tie = state.ties[ANA_BRUNO]
    assert tie.interactive and tie.bond_energy == bond and tie.tie_state is TieState.ORDINARY


def test_m4g1_three_withdrawals_register_as_distant_tie():
    state = fresh()
    for tick, sender, target in ((0, MARTA, RAVI), (1, RAVI, MARTA), (2, MARTA, RAVI)):
        state.tick = tick
        act(state, Selection(sender, "DISTANCE", (target,), 2.0), VIS)
    consolidate(state, PARAMS)
    assert state.ties[TieId.of(RAVI, MARTA)].tie_state is TieState.DISTANT


def test_m4g1_withdrawals_separated_by_other_traffic_do_not_harden():
    state = fresh()
    for tick, kind in ((0, "DISTANCE"), (1, "CONFLICT"), (2, "DISTANCE"), (3, "DISTANCE")):
        state.tick = tick
        act(state, Selection(MARTA, kind, (RAVI,), 2.0), VIS)
    consolidate(state, PARAMS)
    assert state.ties[TieId.of(RAVI, MARTA)].tie_state is TieState.ORDINARY


# --- M1.A.8 ---------------------------------------------------------------------------


def test_m1a8_acute_decays_toward_the_chronic_floor_and_not_below():
    state = fresh()
    ravi = state.people[RAVI]
    ravi.acute_anxiety = ravi.chronic_anxiety + 10.0
    consolidate(state, PARAMS)
    assert ravi.chronic_anxiety < ravi.acute_anxiety < ravi.chronic_anxiety + 10.0
    ravi.acute_anxiety = ravi.chronic_anxiety - 3.0  # e.g. after a calming event
    consolidate(state, PARAMS)
    assert ravi.acute_anxiety == ravi.chronic_anxiety


# --- M1.A.12, M1.C.3 ------------------------------------------------------------------


def test_m1a12_membership_is_derived_from_involvement():
    state = fresh()
    recompute_involvement(state)
    assert state.people[RAVI].involvement_weight > state.people[BRUNO].involvement_weight
    assert is_member(state, RAVI, PARAMS)
    state.ties[ANA_BRUNO].bond_energy = 0.0
    recompute_involvement(state)
    assert not is_member(state, BRUNO, PARAMS)


def test_m1c3_topology_is_the_closed_triads():
    assert set(fresh().triangles) == {TriangleId.of(RAVI, MARTA, NADIA), TriangleId.of(RAVI, MARTA, PIA)}


def test_m1c3_activity_is_a_readout_of_recent_triangle_acts():
    """Amended M1.C.3 (revision 11): active means a TRIANGLE act within the window — tension alone does nothing."""
    state = fresh()
    for p in (RAVI, MARTA):
        state.people[p].acute_anxiety += 20.0  # high tension, no TRIANGLE act
    recompute_triangles(state, PARAMS)
    assert not any(t.active for t in state.triangles.values())
    act(state, Selection(RAVI, "TRIANGLE", (NADIA,), 3.0), VIS)
    recompute_triangles(state, PARAMS)
    assert state.triangles[TriangleId.of(RAVI, MARTA, NADIA)].active
    assert not state.triangles[TriangleId.of(RAVI, MARTA, PIA)].active
    state.tick = PARAMS.triangle_activity_window  # the act now falls outside the window
    recompute_triangles(state, PARAMS)
    assert not state.triangles[TriangleId.of(RAVI, MARTA, NADIA)].active


def test_m1c3_calm_and_tense_systems_get_no_threshold():
    """No calm-system threshold in step 6: the same TRIANGLE act activates its triangle at any tension."""
    calm, tense = fresh(), fresh()
    for p in (RAVI, MARTA):
        tense.people[p].acute_anxiety += 40.0
    for state in (calm, tense):
        act(state, Selection(RAVI, "TRIANGLE", (NADIA,), 3.0), VIS)
        recompute_triangles(state, PARAMS)
    key = TriangleId.of(RAVI, MARTA, NADIA)
    assert calm.triangles[key].active and tense.triangles[key].active


def test_m85_a_triangle_with_an_absent_member_is_not_active():
    state = fresh()
    act(state, Selection(RAVI, "TRIANGLE", (NADIA,), 3.0), VIS)
    act(state, Selection(RAVI, "TRIANGLE", (PIA,), 3.0, index=1), VIS)
    state.people[NADIA].alive = False
    recompute_triangles(state, PARAMS)
    assert not state.triangles[TriangleId.of(RAVI, MARTA, NADIA)].active
    assert state.triangles[TriangleId.of(RAVI, MARTA, PIA)].active


def test_m1c3_triangle_move_sets_the_inside_pair():
    state = fresh()
    act(state, Selection(RAVI, "TRIANGLE", (NADIA,), 3.0), VIS)
    recompute_triangles(state, PARAMS)
    triangle = state.triangles[TriangleId.of(RAVI, MARTA, NADIA)]
    # The target is the recruited third, outside; the sender and the remaining member are inside (2026-10-08).
    assert triangle.inside_pair == TieId.of(RAVI, MARTA) and triangle.outside == NADIA
    assert triangle.activation_memory == 1


# --- M4.E.1 -----------------------------------------------------------------------------


def test_m4e1_scripted_move_becomes_full_event():
    state = fresh()
    records = act(state, Selection(RAVI, "CONFLICT", (MARTA,), 5.0), VIS)
    [event] = state.store.events()
    assert event.witnesses_computed and event.witnesses == (NADIA, PIA)
    assert event.channel is Channel.SCRIPTED and event.sender == RAVI
    assert state.queue.pending() == 3
    assert [type(r).__name__ for r in records] == ["SelectionRecord", "EmittedRecord"]
    with pytest.raises(ValueError, match="not a move"):
        act(state, Selection(RAVI, "TRIGGER", (MARTA,), 1.0, index=1), VIS)


# --- M4.G.2 / M6 ------------------------------------------------------------------------


def _checked(state, mutate=None, steps=("standing_load", "deliver")):
    before = snapshot(state)
    effects, loaded = apply_standing_load(state, PARAMS)
    if mutate:
        loaded = mutate(state, loaded)
    return assert_invariants(state, before, loaded, steps, PARAMS.invariant_tolerance, tuple(effects))


def test_m4g2_invariants_asserted_every_tick():
    record = _checked(fresh())
    statuses = dict(record.results)
    assert set(statuses) == {f"M6.I.{i}" for i in range(1, 9)}
    assert statuses["M6.I.6"].value == "passed"  # restated at revision 12, asserted from Phase C step 10
    assert all(s.value == "passed" for k, s in statuses.items() if k != "M6.I.6")


@pytest.mark.parametrize(
    "invariant, mutate, steps",
    [
        ("M6.I.5", lambda s, l: (setattr(s.people[RAVI], "basic_level", 45.0), l)[1], ("standing_load", "deliver")),
        ("M6.I.8", lambda s, l: l - {ANA_BRUNO}, ("standing_load", "deliver")),
        ("M6.I.8", None, ("deliver", "standing_load")),
        ("M6.I.7", lambda s, l: (s.ties.pop(ANA_BRUNO), l - {ANA_BRUNO})[1], ("standing_load", "deliver")),
        ("M6.I.4", lambda s, l: (setattr(s.people[RAVI], "pseudo_self", 3.0), l)[1], ("standing_load", "deliver")),
    ],
)
def test_m4g2_a_violation_raises(invariant, mutate, steps):
    with pytest.raises(InvariantViolation, match=invariant):
        _checked(fresh(), mutate, steps)


def test_m4g2a_m6i6_is_asserted_once_restated():
    """M4.G.2a held M6.I.6 back until revision 12 restated it; from Phase C step 10 it is asserted."""
    record = _checked(fresh(), None, ("standing_load", "deliver"))
    assert dict(record.results)["M6.I.6"] is InvariantStatus.PASSED
    with pytest.raises(InvariantViolation, match="M6.I.6"):
        _checked(fresh(), lambda s, l: (setattr(s.people[RAVI], "acute_anxiety", 99.0), l)[1],
                 ("standing_load", "deliver"))


def test_m1f5_a_witness_takes_no_more_than_the_edge_it_overheard():
    """Step 12 finding: a witness's conductance is bounded by the event's own edge."""
    state = fresh()
    act(state, Selection(ANA, "TRIANGLE", (MARTA,), 120.0), VIS)
    batch = state.queue.release(1)
    state.tick = 1
    record_deliveries(state, batch)
    edge = state.tie_between(ANA, MARTA).conductance
    for delivery, event in [(d, state.store.event(d.event_id)) for d in batch]:
        delta = appraisal_delta(state, delivery, event, PARAMS)
        divisor = max(state.people[delivery.recipient].functional_level, PARAMS.functional_level_floor)
        assert delta * divisor / event.intensity <= edge + 1e-12, delivery.recipient


def test_m4g1_a_withdrawal_addressed_to_several_still_counts_on_each_tie():
    """Review suspicion (2026-10-06): multi-target withdrawals were left out of the run."""
    state = fresh()
    for tick in range(3):
        state.tick = tick
        act(state, Selection(MARTA, "DISTANCE", (RAVI, NADIA), 2.0), VIS)
    consolidate(state, PARAMS)
    assert state.ties[TieId.of(RAVI, MARTA)].tie_state is TieState.DISTANT
    assert state.ties[TieId.of(MARTA, NADIA)].tie_state is TieState.DISTANT
