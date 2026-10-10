"""Move physics (Phase C step 5, plan D5).

Purpose: test each move's own effect as a direction on the Phase B family.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2a, #M1.C.1, #M1.C.3a, #M1.B.5, #M1.B.6, #M1.B.7, #M6.I.4, #M5.B.1, #M5.B.2, #M5.B.3, #M5.B.6
Tests:   this file

Unit tests of set mechanisms with invented constants; none is a finding (`M11.5`).
"""

from __future__ import annotations

import copy

from src.bowen.engine.appraise import _tie_change, appraisal_delta
from src.bowen.engine.event_effects import apply_structural_event
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.moves import DEFAULT_AREA, apply_move_effects, settle_functioning
from src.bowen.engine.objects import TieState
from src.bowen.engine.recompute import recompute_triangles
from src.bowen.engine.state import new_run_state
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, NADIA, PIA = P("ravi"), P("marta"), P("nadia"), P("pia")
MARITAL = TieId.of(RAVI, MARTA)
RAVI_NADIA = TieId.of(RAVI, NADIA)
FAMILY_TRIAD = TriangleId.of(MARTA, NADIA, RAVI)
PARAMS = engine_params(load_constants())


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, PARAMS)


def move(kind, sender, target, intensity=100.0, tick=0, index=0):
    return Event(
        id=EventId(tick, str(sender.value), index), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
        targets=(target,), intensity=intensity, timestamp=tick, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
    )


def deliver(state, *events):
    """Record the events and apply their effects as step 4 would, on delivery to each target."""
    batch = []
    for event in events:
        state.store.record_event(event)
        batch.append(Delivery(event.timestamp + 1, event.id, event.targets[0], Role.TARGET,
                              emitted_tick=event.timestamp, latency=1))
    return apply_move_effects(state, tuple(batch), PARAMS)


def anxious(state, *people, by=20.0):
    for pid in people:
        state.people[pid].acute_anxiety = state.people[pid].chronic_anxiety + by


def total_anxiety(state):
    return sum(p.acute_anxiety for p in state.people.values()) + sum(t.distance_bound_anxiety for t in state.ties.values())


# --- DISTANCE ------------------------------------------------------------------------------


def test_m1d2a_distance_binds_anxiety_into_the_tie_without_destroying_it():
    state = fresh()
    anxious(state, MARTA)
    before_marta, before_total = state.people[MARTA].acute_anxiety, total_anxiety(state)
    deliver(state, move("DISTANCE", MARTA, RAVI))
    bound = state.ties[MARITAL].distance_bound_anxiety
    assert bound > 0
    assert state.people[MARTA].acute_anxiety == before_marta - bound
    assert abs(total_anxiety(state) - before_total) < 1e-12  # moved, not destroyed
    settle_functioning(state, PARAMS)
    assert state.ties[MARITAL].distance_bound_anxiety == bound  # persists


def test_m1d2a_bound_anxiety_returns_when_distancing_is_prevented():
    state = fresh()
    anxious(state, MARTA)
    deliver(state, move("DISTANCE", MARTA, RAVI))
    bound, before_total = state.ties[MARITAL].distance_bound_anxiety, total_anxiety(state)
    before = {p: state.people[p].acute_anxiety for p in (RAVI, MARTA)}
    reconcile = Event(
        id=EventId(1, "script", 0), kind="RECONCILIATION", mechanism=Mechanism.RECONCILIATION, sender=None,
        targets=(), intensity=0.0, timestamp=1, duration=1, exogenous=True, source_position=SourcePosition.NONE,
        channel=Channel.SCRIPTED, on_tie=MARITAL,
    )
    apply_structural_event(state, reconcile, PARAMS)
    assert state.ties[MARITAL].distance_bound_anxiety == 0.0
    for p in (RAVI, MARTA):
        assert abs(state.people[p].acute_anxiety - before[p] - bound / 2) < 1e-12
    assert abs(total_anxiety(state) - before_total) < 1e-12


# --- TRIANGLE ------------------------------------------------------------------------------


def triangle_relief(state, seeker=RAVI, third=MARTA):
    """Deliver the seeker's TRIANGLE toward the third; return each person's change and the transfer record."""
    before = {p: q.acute_anxiety for p, q in state.people.items()}
    records = deliver(state, move("TRIANGLE", seeker, third))
    change = {p: state.people[p].acute_anxiety - before[p] for p in before}
    return change, [r for r in records if r.mechanism == "triangle_transfer"]


def strained_with_nadia():
    """Ravi is anxious and most strained with Nadia, so a TRIANGLE toward Marta recruits her into that twosome."""
    state = fresh()
    anxious(state, RAVI, NADIA)
    state.ties[RAVI_NADIA].felt_impingement[RAVI] = 1.0
    return state


def test_m1c1_triangle_relieves_the_seeker_and_loads_the_third():
    """Owner decision 2026-10-09: only the seeker is relieved; the partner's anxiety eases only through their own
    act. The third absorbs what the seeker passes on, and generates more (KS03.1)."""
    change, records = triangle_relief(strained_with_nadia())
    assert change[RAVI] < 0 and records
    assert change[NADIA] == 0.0  # the partner, the other in the tense twosome, is not relieved
    given = -change[RAVI]
    assert change[MARTA] - given > 0.1 * given  # not conservative: the outsider's position generates anxiety


def test_m1c1_partner_not_relieved_by_seekers_triangle():
    for strained in (NADIA, PIA):
        state = fresh()
        anxious(state, RAVI, NADIA, PIA)
        state.ties[TieId.of(RAVI, strained)].felt_impingement[RAVI] = 1.0
        change, records = triangle_relief(state)
        assert change[RAVI] < 0 and change[NADIA] == change[PIA] == 0.0
        # The partner still defines the triad: the one Ravi is most strained with.
        assert strained in next(tri for tri, _, _ in records[0].triangles).members


def test_m1c1_anxious_third_helps_less():
    calm, tense = strained_with_nadia(), strained_with_nadia()
    anxious(tense, MARTA, by=60.0)
    assert triangle_relief(tense)[0][RAVI] > triangle_relief(calm)[0][RAVI] < 0  # less relief from a tense third


def test_m1c1_third_aligned_with_partner_helps_less():
    """A third closer to the partner than to the seeker is on the partner's side, and helps the seeker less."""
    with_seeker, with_partner = strained_with_nadia(), strained_with_nadia()
    with_seeker.ties[TieId.of(NADIA, MARTA)].bond_energy = 10.0
    with_partner.ties[TieId.of(NADIA, MARTA)].bond_energy = 90.0
    assert triangle_relief(with_partner)[0][RAVI] > triangle_relief(with_seeker)[0][RAVI] < 0


def test_m1c3a_better_differentiated_triangle_routes_less():
    routed = {}
    for lift in (0.0, 30.0):
        state = fresh()
        for p in (RAVI, NADIA, MARTA):
            state.people[p].functional_level += lift
        anxious(state, RAVI, MARTA)
        before = state.people[NADIA].acute_anxiety
        deliver(state, move("TRIANGLE", RAVI, NADIA))  # Nadia is recruited into Ravi and Marta's twosome
        routed[lift] = state.people[NADIA].acute_anxiety - before
    assert routed[30.0] < routed[0.0]


# --- OVERFUNCTION / UNDERFUNCTION --------------------------------------------------------


def set_balance(state, tie, balance, habit):
    state.ties[tie].functioning_balance[DEFAULT_AREA] = balance
    state.ties[tie].functioning_habit[DEFAULT_AREA] = habit


def balance(state, tie=RAVI_NADIA):
    return state.ties[tie].functioning_balance[DEFAULT_AREA]


# On ravi~nadia the first member is nadia: a positive balance is Nadia over-functioning.


def test_m1b6_flip_is_relative_and_immediate():
    """The same assertion flips a weak domination and not a strong one — a comparison, not a threshold."""
    weak, strong = fresh(), fresh()
    set_balance(weak, RAVI_NADIA, -0.1, 0.0)   # Ravi over, mildly
    set_balance(strong, RAVI_NADIA, -0.9, 0.0)  # Ravi over, markedly
    for state in (weak, strong):
        deliver(state, move("OVERFUNCTION", NADIA, RAVI, intensity=100.0))
    assert balance(weak) > 0 > balance(strong)  # flipped at once in the first, not in the second


def test_m1b6_a_flip_reverts_unless_sustained():
    once, sustained = fresh(), fresh()
    for state in (once, sustained):
        set_balance(state, RAVI_NADIA, -0.2, -0.8)  # a hardened configuration: Ravi over
        deliver(state, move("OVERFUNCTION", NADIA, RAVI, intensity=200.0))
        assert balance(state) > 0  # flipped
    for week in range(1, 120):
        if week % 2 == 0:
            deliver(sustained, move("OVERFUNCTION", NADIA, RAVI, intensity=200.0, tick=week))
        settle_functioning(once, PARAMS)
        settle_functioning(sustained, PARAMS)
    assert balance(once) < 0 < balance(sustained)


def test_m1b5_balance_has_no_stable_midpoint():
    for start in (0.05, -0.05):
        state = fresh()
        set_balance(state, RAVI_NADIA, start, 0.0)
        for _ in range(200):
            settle_functioning(state, PARAMS)
        assert abs(balance(state)) > 0.9 and balance(state) * start > 0


def test_m1b7_raising_a_marked_underfunctioner_costs_more_than_reducing_the_overfunctioner():
    step_back, assert_up = fresh(), fresh()
    for state in (step_back, assert_up):
        set_balance(state, RAVI_NADIA, -0.8, -0.8)  # Ravi a marked over-functioner
    deliver(step_back, move("UNDERFUNCTION", RAVI, NADIA, intensity=100.0))  # the over-functioner yields
    deliver(assert_up, move("OVERFUNCTION", NADIA, RAVI, intensity=100.0))   # the under-functioner asserts
    assert balance(step_back) > balance(assert_up) > -0.8


def test_m6i4_pseudo_self_moves_with_functional_level_and_is_conserved():
    state = fresh()
    assert all(p.pseudo_self == p.swing for p in state.people.values())
    total = sum(p.pseudo_self for p in state.people.values())
    fl = {p: state.people[p].functional_level for p in (RAVI, NADIA)}
    deliver(state, move("OVERFUNCTION", RAVI, NADIA))
    gain = state.people[RAVI].functional_level - fl[RAVI]
    assert gain > 0 and state.people[NADIA].functional_level == fl[NADIA] - gain
    assert abs(sum(p.pseudo_self for p in state.people.values()) - total) < 1e-12
    assert all(p.basic_level == fresh().people[p.id].basic_level for p in state.people.values())


# --- M5.B ----------------------------------------------------------------------------------


def cut_with_bound(state, held=6.0):
    tie = state.ties[MARITAL]
    tie.interactive, tie.tie_state = False, TieState.CUT_OFF
    tie.distance_bound_anxiety = held


def test_m5b3_reduce_cutoff_cost_lands_on_the_third_party():
    """The bound anxiety goes to the third parties whose distancing is stripped, not to the mover (L22.6)."""
    state = fresh()
    cut_with_bound(state)
    before = {p: state.people[p].acute_anxiety for p in (RAVI, MARTA, NADIA, PIA)}
    deliver(state, move("REDUCE_CUTOFF", RAVI, MARTA))
    assert state.ties[MARITAL].interactive and state.ties[MARITAL].tie_state is TieState.ORDINARY
    assert state.people[RAVI].acute_anxiety == before[RAVI] and state.people[MARTA].acute_anxiety == before[MARTA]
    assert state.people[NADIA].acute_anxiety - before[NADIA] == state.people[PIA].acute_anxiety - before[PIA] == 3.0


def test_m5b3_reduce_cutoff_lowers_anxiety_and_never_raises_basic_level():
    state = fresh()
    cut_with_bound(state, held=0.0)
    for m in MARITAL.members():
        state.ties[MARITAL].felt_contact[m] = 0.0  # long severed: far below the optimum
    event = move("REDUCE_CUTOFF", RAVI, MARTA, intensity=60.0)
    state.store.record_event(event)
    delivery = Delivery(1, event.id, MARTA, Role.TARGET, emitted_tick=0, latency=1)
    assert appraisal_delta(state, delivery, event, PARAMS) < 0
    basic = {p: q.basic_level for p, q in state.people.items()}
    deliver(state, move("REDUCE_CUTOFF", RAVI, MARTA, index=1))
    assert {p: q.basic_level for p, q in state.people.items()} == basic


def test_m5b6_provoke_intensity_is_a_dial_separate_from_content():
    state = fresh()
    changes = []
    for i, intensity in enumerate((40.0, 80.0)):
        event = move("PROVOKE", RAVI, MARTA, intensity=intensity, index=i)
        state.store.record_event(event)
        _, d_contact, d_imp = _tie_change(state, Delivery(1, event.id, MARTA, Role.TARGET, emitted_tick=0, latency=1), event, PARAMS)
        changes.append((d_contact, d_imp))
    (c1, i1), (c2, i2) = changes
    assert abs(c1 / i1 - c2 / i2) < 1e-12  # the same content
    assert abs(c2 - 2 * c1) < 1e-12 and abs(i2 - 2 * i1) < 1e-12  # at twice the dial


def triangle_after(state, *events):
    for event in events:
        state.store.record_event(event)
    state.tick = max(e.timestamp for e in events)
    recompute_triangles(state, PARAMS)
    return state.triangles[FAMILY_TRIAD]


def test_m5b1_detriangle_returns_the_third_to_neutral_with_knowledge_intact():
    aligned = triangle_after(fresh(), move("TRIANGLE", RAVI, NADIA))
    assert aligned.active and aligned.outside == NADIA  # the recruited third (decided 2026-10-08)
    state = fresh()
    triangle_after(state, move("TRIANGLE", RAVI, NADIA))
    knowledge = copy.deepcopy(state.people[NADIA].tie_beliefs)
    triangle = triangle_after(state, move("DETRIANGLE", MARTA, NADIA, tick=1))
    assert not triangle.active and triangle.outside is None
    assert state.people[NADIA].tie_beliefs == knowledge


def test_m5b2_prevent_alignment_acts_before_any_alignment_forms():
    guarded = fresh()
    assert not guarded.triangles[FAMILY_TRIAD].active  # nothing to respond to yet
    triangle = triangle_after(guarded, move("PREVENT_ALIGNMENT", MARTA, NADIA), move("TRIANGLE", RAVI, NADIA, tick=1))
    assert not triangle.active
    open_ = triangle_after(fresh(), move("TRIANGLE", RAVI, NADIA, tick=1))
    assert open_.active
