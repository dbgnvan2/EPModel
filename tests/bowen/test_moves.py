"""Move physics (Phase C step 5, plan D5).

Purpose: test each move's own effect as a direction on the Phase B family.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2a, #M1.C.1, #M1.C.3a, #M1.B.5, #M1.B.6, #M1.B.7, #M6.I.4, #M5.B.1, #M5.B.2, #M5.B.3, #M5.B.6
Tests:   this file

Unit tests of set mechanisms with invented constants; none is a finding (`M11.5`).
"""

from __future__ import annotations

import pytest

import copy

from src.bowen.engine.appraise import tie_change, appraisal_delta
from src.bowen.engine.event_effects import apply_structural_event
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.log_records import DecidedBy
from src.bowen.engine.moves import DEFAULT_AREA, apply_move_effects, settle_functioning
from src.bowen.engine.objects import TieState
from src.bowen.engine.recompute import recompute_triangles
from src.bowen.engine.rejection import raise_rejection
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
    raise_rejection(tie, MARTA, PARAMS.rejection_relief)  # one accepted REDUCE_CUTOFF reopens it
    tie.distance_bound_anxiety = held


def test_m5b3_reduce_cutoff_moves_no_stock_to_the_third_party():
    """Revision 36 (G1, D7): the act lowers rejection and moves nothing; the tie's stock stays in the tie for the
    bridging pass (`M1.D.2b`), and no third party receives a transfer. The third party's cost on the reopening is the
    outlet they lose, which `M11.C`'s readouts measure, not a sum written here (L22.6, still cited, not yet shown)."""
    state = fresh()
    cut_with_bound(state)
    before = {p: state.people[p].acute_anxiety for p in (RAVI, MARTA, NADIA, PIA)}
    records = deliver(state, move("REDUCE_CUTOFF", RAVI, MARTA))
    assert state.ties[MARITAL].interactive and state.ties[MARITAL].tie_state is TieState.CUT_OFF  # until step 4 bridges
    assert not state.ties[MARITAL].rejected() and state.ties[MARITAL].distance_bound_anxiety == 6.0
    assert all(state.people[p].acute_anxiety == before[p] for p in before)
    assert [r.mechanism for r in records] == ["reduce_cutoff"] and records[0].acute_anxiety == ()


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
        _, d_contact, d_imp = tie_change(state, Delivery(1, event.id, MARTA, Role.TARGET, emitted_tick=0, latency=1), event, PARAMS)
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


def test_m1a5e_giving_way_cannot_take_the_giver_below_the_floor():
    """Revision 19: the stress floor bounds the pseudo-self exchange too; without it 2 of 20 runs still ran away."""
    state = fresh()
    giver = state.people[MARTA]
    floor = giver.basic_level - PARAMS.stress_floor_below_basic
    giver.functional_level = floor + 0.5
    deliver(state, move("UNDERFUNCTION", MARTA, RAVI))
    assert giver.functional_level == floor
    deliver(state, move("UNDERFUNCTION", MARTA, RAVI, tick=1))
    assert giver.functional_level == floor                         # nothing more to give



def _stance(state, holder, other, kind="STAY-IN-CONTACT", tick=0):
    from src.bowen.engine.act import Selection
    from src.bowen.engine.external import record_stances

    state.tick = tick
    record_stances(state, (Selection(holder, kind, (other,), 100.0),), PARAMS)


def _exchange(state, kind, sender, target, tick):
    """Deliver one exchange at ``tick`` and return what moved: (sender's, target's) functional level, the balance."""
    state.tick = tick
    before = (state.people[sender].functional_level, state.people[target].functional_level,
              state.ties[TieId.of(sender, target)].functioning_balance[DEFAULT_AREA])
    records = deliver(state, move(kind, sender, target, tick=tick - 1, index=tick))
    after = (state.people[sender].functional_level, state.people[target].functional_level,
             state.ties[TieId.of(sender, target)].functioning_balance[DEFAULT_AREA])
    return tuple(a - b for a, b in zip(after, before)), records


def test_m1b5_holding_a_position_stops_the_exchange_both_ways():
    """Revision 21 (owner decision 2026-10-12): the exchange takes two. A person holding a position toward the sender
    is not taken over by an OVERFUNCTION and does not take over for an UNDERFUNCTION; without a stance both move."""
    for kind in ("OVERFUNCTION", "UNDERFUNCTION"):
        held, plain = fresh(), fresh()
        _stance(held, RAVI, NADIA)
        moved, records = _exchange(held, kind, NADIA, RAVI, tick=2)
        assert moved == (0.0, 0.0, 0.0), kind
        assert [r.mechanism for r in records] == ["position_held"]
        moved, _ = _exchange(plain, kind, NADIA, RAVI, tick=2)
        assert moved[0] != 0 and moved[1] != 0 and moved[2] != 0, kind
    withheld = fresh()                                               # holding back counts as a stance too
    _stance(withheld, RAVI, NADIA, kind="WITHHOLD")
    assert _exchange(withheld, "OVERFUNCTION", NADIA, RAVI, tick=2)[0] == (0.0, 0.0, 0.0)


def test_m1b5_a_stance_lapses_with_time_or_a_reactive_act():
    """Held means a self-directed act or hold toward that person in the last practice_window weeks, and no reactive
    act toward them since: any automatic act toward them, or triangling someone in about them (owner decision
    2026-10-12). Toward someone else, or by the other person, it changes nothing."""
    from src.bowen.engine.act import Selection
    from src.bowen.engine.external import record_stances

    edge = fresh()                                                   # still held at the window's last week
    _stance(edge, RAVI, NADIA)
    assert _exchange(edge, "OVERFUNCTION", NADIA, RAVI, tick=PARAMS.practice_window)[0] == (0.0, 0.0, 0.0)
    late = fresh()
    _stance(late, RAVI, NADIA)
    assert _exchange(late, "OVERFUNCTION", NADIA, RAVI, tick=PARAMS.practice_window + 1)[0][1] != 0
    for kind in ("DISTANCE", "PURSUE", "OVERFUNCTION", "CONFLICT"):  # a reactive act toward them ends it
        broken = fresh()
        _stance(broken, RAVI, NADIA)
        _stance(broken, RAVI, NADIA, kind=kind, tick=1)
        assert _exchange(broken, "OVERFUNCTION", NADIA, RAVI, tick=2)[0][1] != 0, kind
    earlier = fresh()                                                # a reactive act before the stance does not undo it
    _stance(earlier, RAVI, NADIA, kind="DISTANCE", tick=0)
    _stance(earlier, RAVI, NADIA, tick=1)
    assert _exchange(earlier, "OVERFUNCTION", NADIA, RAVI, tick=3)[0] == (0.0, 0.0, 0.0)
    theirs = fresh()                                                 # the other person's reactive act does not end it
    _stance(theirs, RAVI, NADIA)
    _stance(theirs, NADIA, RAVI, kind="DISTANCE", tick=1)
    assert _exchange(theirs, "OVERFUNCTION", NADIA, RAVI, tick=2)[0] == (0.0, 0.0, 0.0)
    elsewhere = fresh()                                              # a stance toward someone else gives no cover
    _stance(elsewhere, RAVI, MARTA)
    assert _exchange(elsewhere, "OVERFUNCTION", NADIA, RAVI, tick=2)[0][1] != 0
    reactive_elsewhere = fresh()                                     # nor does a reactive act toward someone else end it
    _stance(reactive_elsewhere, RAVI, NADIA)
    _stance(reactive_elsewhere, RAVI, PIA, kind="CONFLICT", tick=1)
    assert _exchange(reactive_elsewhere, "OVERFUNCTION", NADIA, RAVI, tick=2)[0] == (0.0, 0.0, 0.0)
    automatic = fresh()                                              # an automatic act is not a stance
    _stance(automatic, RAVI, NADIA, kind="OVERFUNCTION")
    assert automatic.stances == {}
    forced = fresh()                                                 # a fallback hold is not a stance
    forced.tick = 0
    record_stances(forced, (Selection(RAVI, "WITHHOLD", (NADIA,), 0.0, decided_by=DecidedBy.FALLBACK),), PARAMS)
    assert forced.stances == {}


def test_m1b5_triangling_someone_in_about_them_ends_the_stance():
    """Triangling Marta in to get away from Nadia ends Ravi's stance toward Nadia, the inside partner the TRIANGLE
    forms, though the act is aimed at Marta (owner decision 2026-10-12)."""
    from src.bowen.engine.moves import triangle_partner

    state = strained_with_nadia()
    assert triangle_partner(state, RAVI, MARTA, PARAMS)[1] == NADIA
    _stance(state, RAVI, NADIA)
    _stance(state, RAVI, MARTA, kind="TRIANGLE", tick=1)
    assert (RAVI, NADIA) not in state.stances


def test_m6i4_pseudo_self_settles_back_when_the_exchange_stops():
    """Revision 27: with no over/under-functioning between a pair for exchange_window weeks, the borrower's pseudo-self
    moves back to the lender, conserved, with functional level; a recent exchange, or same-signed pseudo-self, holds it."""
    from src.bowen.engine.moves import _pole_of, relax_pseudo_self

    def set_up(recent, balance=True, at=None):
        state = fresh()
        for pid, ps in ((RAVI, 6.0), (MARTA, -4.0)):
            person = state.people[pid]
            person.functional_level += ps - (person.pseudo_self or 0.0)
            person.pseudo_self = ps
        tie = state.ties[MARITAL]
        tie.functioning_balance[DEFAULT_AREA] = 0.5 * _pole_of(tie, RAVI) if balance else 0.0  # Ravi took charge
        state.tick = PARAMS.exchange_window + 2
        if recent:
            state.store.record_event(move("OVERFUNCTION", RAVI, MARTA, tick=state.tick - 1 if at is None else at))
        return state

    state = set_up(recent=False)
    total, levels = sum(p.pseudo_self or 0.0 for p in state.people.values()), dict(
        (p, state.people[p].functional_level) for p in (RAVI, MARTA))
    relax_pseudo_self(state, PARAMS)
    moved = PARAMS.pseudo_self_relaxation_rate * 4.0
    assert state.people[RAVI].pseudo_self == 6.0 - moved and state.people[MARTA].pseudo_self == -4.0 + moved
    assert state.people[RAVI].functional_level == levels[RAVI] - moved
    assert state.people[MARTA].functional_level == levels[MARTA] + moved
    assert sum(p.pseudo_self or 0.0 for p in state.people.values()) == total
    state = set_up(recent=True)
    relax_pseudo_self(state, PARAMS)
    assert state.people[RAVI].pseudo_self == 6.0
    # Review 2026-10-12: a pair that never exchanged (no balance on their tie) does not settle across it.
    state = set_up(recent=False, balance=False)
    relax_pseudo_self(state, PARAMS)
    assert state.people[RAVI].pseudo_self == 6.0


def test_m6i4_the_exchange_window_counts_the_last_exchange_window_weeks():
    """An exchange exactly exchange_window weeks back is outside the window and lets the pair settle; one week later
    it holds them."""
    from src.bowen.engine.moves import _pole_of, relax_pseudo_self

    def settles(age):
        state = fresh()
        for pid, ps in ((RAVI, 6.0), (MARTA, -4.0)):
            state.people[pid].pseudo_self = ps
        tie = state.ties[MARITAL]
        tie.functioning_balance[DEFAULT_AREA] = 0.5 * _pole_of(tie, RAVI)
        state.tick = PARAMS.exchange_window + 2
        state.store.record_event(move("OVERFUNCTION", RAVI, MARTA, tick=state.tick - age))
        relax_pseudo_self(state, PARAMS)
        return state.people[RAVI].pseudo_self < 6.0

    assert settles(PARAMS.exchange_window) and not settles(PARAMS.exchange_window - 1)


def test_m6i4_a_borrower_still_exchanging_with_its_source_keeps_what_it_borrowed():
    """Review 2026-10-12: pseudo-self settles back only on the tie it was exchanged on. Nadia borrowed from Marta and
    is still exchanging with her; Ravi lent to someone else; nothing moves between Nadia and Ravi, whose tie carries no
    balance, nor between Nadia and Marta, who are still exchanging."""
    from src.bowen.engine.moves import _pole_of, relax_pseudo_self

    state = fresh()
    nadia, marta = P("nadia"), P("marta")
    for pid, ps in ((nadia, 5.0), (marta, -5.0), (RAVI, -3.0)):
        state.people[pid].pseudo_self = ps
    tie = state.ties[TieId.of(marta, nadia)]
    tie.functioning_balance[DEFAULT_AREA] = 0.5 * _pole_of(tie, nadia)
    state.tick = PARAMS.exchange_window + 2
    state.store.record_event(move("OVERFUNCTION", nadia, marta, tick=state.tick - 1))
    relax_pseudo_self(state, PARAMS)
    assert (state.people[nadia].pseudo_self, state.people[marta].pseudo_self, state.people[RAVI].pseudo_self) == (5.0, -5.0, -3.0)


def _taken_at(below_basic: float, kind: str = "OVERFUNCTION") -> float:
    """What Ravi, the over-functioner, takes from Nadia in one exchange when his functional level is ``below_basic``
    under his basic level (stress's part, as `M11.C.21`'s collapse writes it)."""
    state = fresh()
    ravi = state.people[RAVI]
    ravi.functional_level = ravi.basic_level - below_basic
    ravi.stress_swing -= below_basic
    sender, target = (RAVI, NADIA) if kind == "OVERFUNCTION" else (NADIA, RAVI)
    moved, _ = _exchange(state, kind, sender, target, tick=2)
    return moved[0] if kind == "OVERFUNCTION" else moved[1]


def test_m6i4_collapsed_taker_takes_in_proportion_to_room():
    """Revision 31 (`M11.C.21`, the owner's form 2026-10-16): what an over-functioner takes is scaled by their room
    above the stress floor, s = clip((f − (b − floor)) / floor, 0, 1): full at basic, a third at 10 below (floor 15),
    none at the floor. The same whichever of the two sends the act."""
    floor = PARAMS.stress_floor_below_basic
    full = _taken_at(0.0)
    assert full > 0
    assert _taken_at(10.0) == pytest.approx(full * (floor - 10.0) / floor)
    assert _taken_at(floor) == 0.0
    assert _taken_at(10.0, kind="UNDERFUNCTION") == pytest.approx(_taken_at(0.0, kind="UNDERFUNCTION") * (floor - 10.0) / floor)


def test_m6i4_a_low_basic_taker_at_zero_takes_nothing():
    """learning-qa 2026-10-16: the floor is clamped at 0 as the giver's and stress's are. A taker with basic level 5 at
    functional level 0 is at the floor and takes nothing; halfway between the floor and basic takes half."""
    state = fresh()
    ravi = state.people[RAVI]
    ravi.basic_level, ravi.functional_level, ravi.pseudo_self, ravi.stress_swing = 5.0, 0.0, 0.0, -5.0
    assert _exchange(state, "OVERFUNCTION", RAVI, NADIA, tick=2)[0][0] == 0.0
    half = fresh()
    ravi = half.people[RAVI]
    ravi.basic_level, ravi.functional_level, ravi.pseudo_self, ravi.stress_swing = 5.0, 2.5, 0.0, -2.5
    assert _exchange(half, "OVERFUNCTION", RAVI, NADIA, tick=2)[0][0] == pytest.approx(_taken_at(0.0) / 2)


def test_m6i4_a_taker_at_basic_zero_takes_in_full():
    """Fix-loop re-sweep 2026-10-16: with basic level 0 the floor and basic coincide (no span), and the rule still reads
    "full at or above basic". A taker at basic 0 takes the full amount; below basic is not reachable (levels clamp at 0)."""
    state = fresh()
    ravi = state.people[RAVI]
    ravi.basic_level, ravi.functional_level, ravi.pseudo_self, ravi.stress_swing = 0.0, 0.0, 0.0, 0.0
    assert _exchange(state, "OVERFUNCTION", RAVI, NADIA, tick=2)[0][0] == pytest.approx(_taken_at(0.0))


def test_m6i4_with_no_floor_below_basic_a_taker_below_basic_takes_nothing():
    """Narrow review 2026-10-17: with ``stress_floor_below_basic`` 0 the floor is basic itself, so there is no span. A
    taker at basic takes in full; one point below basic is below the floor and takes nothing. The giver's floor is
    basic too, so Nadia starts 10 above hers to have something to give."""
    import dataclasses

    no_floor = dataclasses.replace(PARAMS, stress_floor_below_basic=0.0)
    taken = []
    for below in (0.0, 1.0):
        state = fresh()
        nadia = state.people[NADIA]
        nadia.functional_level = nadia.basic_level + 10.0
        nadia.pseudo_self = (nadia.pseudo_self or 0.0) + 10.0
        ravi = state.people[RAVI]
        ravi.functional_level -= below
        ravi.stress_swing -= below
        event = move("OVERFUNCTION", RAVI, NADIA, tick=1, index=2)
        state.tick = 2
        state.store.record_event(event)
        apply_move_effects(state, (Delivery(2, event.id, NADIA, Role.TARGET, emitted_tick=1, latency=1),), no_floor)
        taken.append(state.people[RAVI].functional_level - (ravi.basic_level - below))
    assert taken[0] == pytest.approx(_taken_at(0.0))
    assert taken[1] == 0.0


def test_m6i4_capacity_does_not_limit_a_taker_above_basic():
    """Adversarial: a taker functioning above their basic level (borrowed pseudo-self) takes the full amount; the rule
    limits a collapsed taker, it does not reward or tax ordinary over-functioning."""
    state = fresh()
    ravi = state.people[RAVI]
    ravi.functional_level = ravi.basic_level + 5.0
    ravi.pseudo_self = (ravi.pseudo_self or 0.0) + 5.0
    above, _ = _exchange(state, "OVERFUNCTION", RAVI, NADIA, tick=2)
    assert above[0] == pytest.approx(_taken_at(0.0))


# --- Version 2 stage 3, group C (spec revision 39, approved 2026-10-25): written first, built at step 7 -------------


def test_m6i4a_s3_4_lent_functioning_returns_with_lost_capacity():
    """S3.4 (`M6.I.4a`, D19): when a taker's capacity falls over the week by a cause other than the exchange, the
    borrowed pseudo-self returns to its lenders in the share of capacity lost, `(s_before − s_after) ÷ s_before`, with
    the same of functional level, split over the lenders in proportion to what each holds lent; the family's total
    pseudo-self and functional level are unchanged. A taker with no capacity at the week's start, or whose fall stays
    at or above basic level (where `s` is 1), returns nothing; no lender is given back more than they lent.

    The call is the contract plan step 7 builds to: ``moves.return_lent_functioning(state, params, week_start)``, with
    ``week_start`` each person's functional level at the week's start; this week carries no exchange, so nothing
    `M6.I.4` moved needs leaving out of `f`."""
    from src.bowen.engine import moves
    from src.bowen.engine.moves import _pole_of

    def set_up(lent):
        """Ravi, at basic level 50, borrowed 6 and took charge on both his ties; each partner holds what they lent."""
        state = fresh()
        ravi = state.people[RAVI]
        ravi.basic_level, ravi.functional_level, ravi.pseudo_self = 50.0, 50.0, 6.0
        for pid, held in lent.items():
            person = state.people[pid]
            person.functional_level += held - (person.pseudo_self or 0.0)
            person.pseudo_self = held
            tie = state.ties[TieId.of(RAVI, pid)]
            tie.functioning_balance[DEFAULT_AREA] = 0.5 * _pole_of(tie, RAVI)
        state.tick = PARAMS.exchange_window + 2
        return state

    def week(state, ravi_level_after):
        start = {pid: p.functional_level for pid, p in state.people.items()}
        state.people[RAVI].functional_level = ravi_level_after          # a fall by a cause outside the exchange
        totals = (sum(p.pseudo_self or 0.0 for p in state.people.values()),
                  sum(p.functional_level for p in state.people.values()))
        before = {pid: (p.functional_level, p.pseudo_self) for pid, p in state.people.items()}
        returned = getattr(moves, "return_lent_functioning", None)
        assert returned is not None, "M6.I.4a's return is not built (moves.return_lent_functioning)"
        returned(state, PARAMS, start)
        after = (sum(p.pseudo_self or 0.0 for p in state.people.values()),
                 sum(p.functional_level for p in state.people.values()))
        assert after == pytest.approx(totals, abs=1e-12)                  # conserved
        return {pid: (state.people[pid].functional_level - before[pid][0],
                      (state.people[pid].pseudo_self or 0.0) - (before[pid][1] or 0.0)) for pid in state.people}

    floor = 50.0 - PARAMS.stress_floor_below_basic
    half = floor + 0.5 * (50.0 - floor)                                   # s falls from 1 to 0.5: half the capacity
    moved = week(set_up({MARTA: -4.0, NADIA: -2.0}), half)
    back = 0.5 * 6.0
    assert moved[RAVI][1] == pytest.approx(-back) and moved[RAVI][0] == pytest.approx(-back)
    assert moved[MARTA][1] == pytest.approx(back * 4 / 6) and moved[MARTA][0] == pytest.approx(back * 4 / 6)
    assert moved[NADIA][1] == pytest.approx(back * 2 / 6) and moved[NADIA][0] == pytest.approx(back * 2 / 6)

    capped = week(set_up({MARTA: -1.0}), half)                            # a lender gets back no more than they lent
    assert capped[MARTA][1] == pytest.approx(1.0) and capped[RAVI][1] == pytest.approx(-1.0)

    above = set_up({MARTA: -4.0})
    above.people[RAVI].functional_level = 55.0
    assert all(v == (0.0, 0.0) for v in week(above, 51.0).values())       # still at or above basic: s stays 1

    empty = set_up({MARTA: -4.0})
    empty.people[RAVI].functional_level = floor
    assert all(v == (0.0, 0.0) for v in week(empty, floor - 1.0).values())  # no capacity at the week's start


def test_m6i4a_the_weeks_exchange_is_left_out_of_the_fall():
    """`M6.I.4a`: the fall that returns borrowed pseudo-self is a fall by a cause other than the exchange. Ravi, below
    basic level and holding pseudo-self borrowed from Marta, gives way to Nadia this week (`M6.I.4`); what he gave is
    read from the week's ``functioning_shift`` records (``exchanged``) and left out, so nothing returns to Marta. Read
    without it, the same week would return some (review 2026-10-26: the exclusion was untested)."""
    from src.bowen.engine.moves import _pole_of, exchanged, return_lent_functioning, stress_floor

    def week(leave_out_exchange):
        state = fresh()
        ravi, marta = state.people[RAVI], state.people[MARTA]
        ravi.basic_level, ravi.pseudo_self = 50.0, 6.0
        ravi.functional_level = 0.5 * (50.0 + stress_floor(ravi, PARAMS))  # half his capacity: s is sensitive here
        marta.functional_level += -4.0 - (marta.pseudo_self or 0.0)
        marta.pseudo_self = -4.0
        tie = state.ties[MARITAL]
        tie.functioning_balance[DEFAULT_AREA] = 0.5 * _pole_of(tie, RAVI)  # Ravi took charge with Marta
        start = {pid: p.functional_level for pid, p in state.people.items()}
        records = deliver(state, move("OVERFUNCTION", NADIA, RAVI, intensity=100.0))  # Nadia takes charge of Ravi
        gave = exchanged(records)
        assert gave.get(RAVI, 0.0) < 0 < gave.get(NADIA, 0.0), gave        # Ravi gave way, and the record says so
        assert ravi.functional_level < start[RAVI]
        marta_before = marta.pseudo_self
        return_lent_functioning(state, PARAMS, start, gave if leave_out_exchange else None)
        return marta.pseudo_self - marta_before

    assert week(leave_out_exchange=True) == 0.0
    assert week(leave_out_exchange=False) > 0.0                            # the exclusion is what stopped it


def test_m6i4a_exchanged_reads_only_the_exchange():
    """``exchanged`` sums the pseudo-self each person moved in ``functioning_shift`` effects and nothing else: the
    settling back and the return itself are not the exchange."""
    from src.bowen.engine.log_records import EffectRecord
    from src.bowen.engine.moves import exchanged

    records = [EffectRecord(1, "functioning_shift", None, people=((RAVI, "pseudo_self", -2.0), (NADIA, "pseudo_self", 2.0))),
               EffectRecord(1, "functioning_shift", None, people=((RAVI, "pseudo_self", -1.0), (MARTA, "pseudo_self", 1.0))),
               EffectRecord(1, "pseudo_self_relaxation", None, people=((RAVI, "pseudo_self", 5.0),)),
               EffectRecord(1, "pseudo_self_return", None, people=((RAVI, "pseudo_self", 7.0),))]
    assert exchanged(records) == {RAVI: -3.0, NADIA: 2.0, MARTA: 1.0}
