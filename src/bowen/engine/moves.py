"""Move physics — what each act does beyond its contact and impingement components (Phase C plan D5).

Purpose: apply, on delivery, each move's specific effect on state — binding by
         `DISTANCE`, the triangle's transfer to its outsider, the functioning
         balance and pseudo-self exchange of over- and underfunctioning, and the
         release of a severed tie by `REDUCE_CUTOFF` — and settle the
         functioning balance once a tick.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2a, #M1.C.1, #M1.C.3a, #M1.B.5, #M1.B.6, #M1.B.7, #M6.I.4, #M5.B.3, #M5.B.3a, #M5.B.6, #M6.4
Tests:   tests/bowen/test_moves.py

Every form below is the project's, graded [I]; the constants are in
``config/bowen/constants.md``. Each effect is applied at step 4 when the move is
delivered to its target, after the batch's appraisal, in canonical delivery order.
Every change to an anxiety stock here is a **transfer** named in `M6.4`'s table,
except the outsider's own positional anxiety, which is a **source** the table names
(`KS03.1`). ``s`` below is a delivery's ``intensity / intensity_scale``.

**`DISTANCE` binds** (`M1.D.2a`, `KS04.1`). The distancer moves
``distance_binding_rate × s`` of their excess acute anxiety (never more than the
excess) out of themself and into the tie's ``distance_bound_anxiety``. It persists
(nothing here decays it), it is readable on the tie, and it returns to the tie's
members, split equally, on `RECONCILIATION` (``event_effects``). Total anxiety is
unchanged by the act: distance is not a way of destroying it.

**`TRIANGLE`** (`M1.C.1`, `KS03.1`). The sender turns to the target. The triangle is
the closed triad holding both whose third member stands furthest from the sender's
optimum on the sender's tie to them (ties broken by identifier) — the tie the sender
is turning away from. The sender and the target are the inside pair; the third is
outside, as step 6's readout has it. Each insider moves
``triangle_transfer_rate × capacity × s`` of its excess to the outsider, and the
outsider generates ``outsider_positional_gain`` times the total on top: "outsiders
both generate their own anxiety and absorb anxiety from insiders". ``capacity`` is
``1 − mean functional_level / 100`` of the three, so the same topology routes less
among better-differentiated members (`M1.C.3a`). With no closed triad there is no
transfer; the move's components still land.

**`OVERFUNCTION` and `UNDERFUNCTION`** (`M1.B.5`–`M1.B.7`, `M6.I.4`).
``functioning_balance[area]`` ∈ [−1, 1] is positive when the tie's first member (by
identifier) is over-functioning. Taking charge pushes it toward the sender's pole by
``balance_push_gain × s``; giving way pushes it toward the target's. A push past zero
flips the pole at once — the flip is the comparison of the under-functioner's
assertion with the other's current domination, not a threshold (`M1.B.6`). A push
toward the middle by the under-functioner asserting itself is scaled by
``1 − reversal_asymmetry × |habit|``; the over-functioner stepping back is not, so
reducing a marked over-functioner costs less than raising a marked under-functioner
(`M1.B.7`). The same act moves ``pseudo_self_transfer_gain × s`` points of
pseudo-self, and the same of ``functional_level``, from the one giving way to the one
taking charge, conserved (`M6.I.4`), never carrying either outside [0, 100].
*One area, ``DEFAULT_AREA``, until the family declares areas of joint activity
(`M1.B.9`).*

**Settling, step 9.** Each area's balance moves ``balance_settle_rate`` of the way to
a pole — the pole of its ``functioning_habit`` when the habit is set, else its own —
and the habit moves ``balance_harden_rate`` of the way to the balance. So the balance
has no stable midpoint (`M1.B.5`); the configuration hardens with time (`L05.3`); and a
flip against a hardened habit reverts unless the mover sustains it long enough for the
habit to cross (`M1.B.6`).

**`REDUCE_CUTOFF`** (`M5.B.3`, `M5.B.3a`, `L22.6`). On delivery the tie becomes
interactive and ordinary. More contact is never penalised as contact; its
impingement component can still carry the receiver past its optimum (`M4.C.1`). The
tie's bound anxiety is released to the third members of the closed triads holding the
tie, split equally — the transient cost lands on the third party whose distancing is
being stripped, not on the person increasing contact. With no third it stays bound.

**`PROVOKE`** (`M5.B.6`) has no effect beyond its components: its content is its kind's
two components and its intensity a separate dial.
"""

from __future__ import annotations

import math

from src.bowen.engine.contact import deviation, excess
from src.bowen.engine.events import Delivery, Event, Mechanism, Role
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import SCALE_MAX, Person, Relationship, TieState
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState

# M5.A.1 and M5.B fix the repertoire; these names are the spec's.
DISTANCE, TRIANGLE, OVERFUNCTION, UNDERFUNCTION, REDUCE_CUTOFF = (
    "DISTANCE", "TRIANGLE", "OVERFUNCTION", "UNDERFUNCTION", "REDUCE_CUTOFF",
)
# The single area every tie holds until the family declares its areas of joint activity (M1.B.9).
DEFAULT_AREA = "joint"


def _strength(event: Event, params: EngineParams) -> float:
    return event.intensity / params.intensity_scale


# --- DISTANCE ------------------------------------------------------------------------------


def bind_distance(state: RunState, event: Event, tie: Relationship, params: EngineParams) -> EffectRecord | None:
    """Purpose: move the distancer's excess anxiety into the tie, where it persists.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2a, #M6.4
    Tests:   tests/bowen/test_moves.py::test_m1d2a_distance_binds_anxiety_into_the_tie_without_destroying_it
    """
    sender = state.people[event.sender]
    bound = min(excess(sender), params.distance_binding_rate * _strength(event, params) * excess(sender))
    if bound <= 0:
        return None
    sender.acute_anxiety -= bound
    tie.distance_bound_anxiety += bound
    return EffectRecord(
        state.tick, "distance_binding", event.id,
        acute_anxiety=((sender.id, -bound),), ties=((tie.id, "distance_bound_anxiety", bound),),
    )


def release_bound_distance(state: RunState, tie: Relationship, to: tuple[PersonId, ...]) -> tuple:
    """Give a tie's bound anxiety to ``to``, split equally; returns the (person, amount) moved."""
    held = tie.distance_bound_anxiety
    if held <= 0 or not to:
        return ()
    tie.distance_bound_anxiety = 0.0
    share = held / len(to)
    for pid in to:
        state.people[pid].acute_anxiety += share
    return tuple((pid, share) for pid in to)


# --- TRIANGLE ------------------------------------------------------------------------------


def _triads_holding(state: RunState, a: PersonId, b: PersonId) -> list[tuple[TriangleId, PersonId]]:
    return [
        (tri_id, next(m for m in tri_id.members if m not in (a, b)))
        for tri_id in sorted(state.triangles)
        if a in tri_id.members and b in tri_id.members
    ]


def triangle_outsider(state: RunState, sender: PersonId, target: PersonId, params: EngineParams):
    """The triad and outsider a TRIANGLE act from ``sender`` to ``target`` forms, or None.

    The outsider is the third member furthest from the sender's optimum on the
    sender's tie to them — the tie being turned away from.
    """
    candidates = [
        (tri_id, third) for tri_id, third in _triads_holding(state, sender, target)
        if state.people[third].alive
    ]
    if not candidates:
        return None
    person = state.people[sender]
    return max(candidates, key=lambda c: (deviation(person, state.tie_between(sender, c[1]), params), _desc(c[1])))


def _desc(pid: PersonId) -> tuple:
    # max() with ties broken toward the smallest identifier.
    return tuple(-ord(ch) for ch in pid.value)


def routing_capacity(members: tuple[Person, ...]) -> float:
    """Purpose: how much a triangle can route — a function of its members' functional level.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.C.3a
    Tests:   tests/bowen/test_moves.py::test_m1c3a_better_differentiated_triangle_routes_less
    """
    return max(0.0, 1.0 - sum(p.functional_level for p in members) / (len(members) * SCALE_MAX))


def triangle_transfer(state: RunState, event: Event, target: PersonId, params: EngineParams) -> EffectRecord | None:
    """Purpose: the insiders pass anxiety to the outsider, who generates more of their own.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.C.1, #M1.C.3a, #M6.4
    Tests:   tests/bowen/test_moves.py::test_m1c1_triangle_relieves_the_insiders_and_loads_the_outsider
    """
    found = triangle_outsider(state, event.sender, target, params)
    if found is None:
        return None
    tri_id, outsider = found
    insiders = (state.people[event.sender], state.people[target])
    capacity = routing_capacity((*insiders, state.people[outsider]))
    rate = params.triangle_transfer_rate * capacity * _strength(event, params)
    moved = [(p.id, min(excess(p), rate * excess(p))) for p in insiders]
    total = sum(m for _, m in moved)
    if total <= 0:
        return None
    for pid, amount in moved:
        state.people[pid].acute_anxiety -= amount
    generated = params.outsider_positional_gain * total
    state.people[outsider].acute_anxiety += total + generated
    return EffectRecord(
        state.tick, "triangle_transfer", event.id,
        acute_anxiety=tuple(sorted([(pid, -m) for pid, m in moved if m] + [(outsider, total + generated)])),
        triangles=((tri_id, "outsider", outsider.value),),
        # M6.4: of the outsider's rise, ``total`` is a transfer and this is a source, logged by name.
        people=((outsider, "positional_anxiety", generated),) if generated else (),
    )


# --- OVERFUNCTION / UNDERFUNCTION --------------------------------------------------------


def _pole_of(tie: Relationship, person: PersonId) -> float:
    """+1 if ``person`` over-functioning is a positive balance on this tie, else −1."""
    return 1.0 if person == tie.id.a else -1.0


def initialise_functioning(people: dict[PersonId, Person], ties: dict[TieId, Relationship]) -> None:
    """Purpose: pseudo-self starts as each person's swing; every tie starts balanced in its one area.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.5a, #M1.B.5, #M6.I.4
    Tests:   tests/bowen/test_moves.py::test_m6i4_pseudo_self_moves_with_functional_level_and_is_conserved
    """
    for pid in sorted(people):
        people[pid].pseudo_self = people[pid].swing
    for tid in sorted(ties):
        ties[tid].functioning_balance = {DEFAULT_AREA: 0.0}
        ties[tid].functioning_habit = {DEFAULT_AREA: 0.0}


def functioning_shift(state: RunState, event: Event, tie: Relationship, target: PersonId, params: EngineParams) -> EffectRecord:
    """Purpose: push the tie's balance toward a pole and move pseudo-self with it, conserved.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.5, #M1.B.6, #M1.B.7, #M6.I.4
    Tests:   tests/bowen/test_moves.py::test_m1b6_flip_is_relative_and_immediate
    """
    taking_charge = event.kind == OVERFUNCTION
    over, under = (event.sender, target) if taking_charge else (target, event.sender)
    direction = _pole_of(tie, over)
    balance = tie.functioning_balance[DEFAULT_AREA]
    habit = tie.functioning_habit[DEFAULT_AREA]
    push = params.balance_push_gain * _strength(event, params)
    # M1.B.7: the under-functioner asserting itself against the current pole is the costly direction.
    asserting_from_below = taking_charge and balance * direction < 0
    if asserting_from_below:
        push *= max(0.0, 1.0 - params.reversal_asymmetry * abs(habit))
    new_balance = max(-1.0, min(1.0, balance + direction * push))
    tie.functioning_balance[DEFAULT_AREA] = new_balance

    giver, taker = state.people[under], state.people[over]
    amount = min(
        params.pseudo_self_transfer_gain * _strength(event, params),
        giver.functional_level, SCALE_MAX - taker.functional_level,
    )
    amount = max(0.0, amount)
    for person, sign in ((taker, 1.0), (giver, -1.0)):
        person.functional_level += sign * amount
        person.pseudo_self = (person.pseudo_self or 0.0) + sign * amount
    flipped = balance != 0 and new_balance != 0 and math.copysign(1, balance) != math.copysign(1, new_balance)
    ties = [(tie.id, f"functioning_balance:{DEFAULT_AREA}", new_balance - balance)]
    if flipped:
        ties.append((tie.id, f"functioning_flip:{DEFAULT_AREA}", new_balance))
    return EffectRecord(
        state.tick, "functioning_shift", event.id, ties=tuple(ties),
        people=((taker.id, "pseudo_self", amount), (giver.id, "pseudo_self", -amount)) if amount else (),
    )


def settle_functioning(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 9 — each balance runs toward a pole, and the habit hardens toward the balance.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.5, #M1.B.6
    Tests:   tests/bowen/test_moves.py::test_m1b5_balance_has_no_stable_midpoint
    """
    changes = []
    for tid in sorted(state.ties):
        tie = state.ties[tid]
        for area in sorted(tie.functioning_balance):
            balance, habit = tie.functioning_balance[area], tie.functioning_habit[area]
            anchor = habit if habit != 0 else balance
            pole = 0.0 if anchor == 0 else math.copysign(1.0, anchor)
            d_balance = params.balance_settle_rate * (pole - balance)
            tie.functioning_balance[area] = balance + d_balance
            d_habit = params.balance_harden_rate * (tie.functioning_balance[area] - habit)
            tie.functioning_habit[area] = habit + d_habit
            if d_balance:
                changes.append((tid, f"functioning_balance:{area}", d_balance))
            if d_habit:
                changes.append((tid, f"functioning_habit:{area}", d_habit))
    return [EffectRecord(state.tick, "functioning_settle", None, ties=tuple(changes))] if changes else []


# --- REDUCE_CUTOFF -------------------------------------------------------------------------


def reduce_cutoff(state: RunState, event: Event, tie: Relationship) -> EffectRecord:
    """Purpose: reopen a severed tie; its bound anxiety goes to the third party whose distancing is stripped.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.B.3, #M5.B.3a, #M6.4
    Tests:   tests/bowen/test_moves.py::test_m5b3_reduce_cutoff_cost_lands_on_the_third_party
    """
    was_cut = not tie.interactive or tie.tie_state is TieState.CUT_OFF
    tie.interactive = True
    tie.tie_state = TieState.ORDINARY
    a, b = tie.id.members()
    thirds = tuple(sorted({third for _, third in _triads_holding(state, a, b) if state.people[third].alive}))
    released = release_bound_distance(state, tie, thirds)
    ties = [(tie.id, "interactive", 1.0)] if was_cut else []
    if released:
        ties.append((tie.id, "distance_bound_anxiety", -sum(v for _, v in released)))
    return EffectRecord(state.tick, "reduce_cutoff", event.id, acute_anxiety=released, ties=tuple(ties))


# --- the step-4 pass -----------------------------------------------------------------------


def apply_move_effects(state: RunState, batch: tuple[Delivery, ...], params: EngineParams) -> list[EffectRecord]:
    """Purpose: apply each delivered move's own effect, in canonical delivery order.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2a, #M1.C.1, #M1.B.5, #M5.B.3, #M1.F.8
    Tests:   tests/bowen/test_moves.py
    """
    records: list[EffectRecord] = []
    for delivery in sorted(batch):
        if delivery.role is not Role.TARGET:
            continue
        event = state.store.event(delivery.event_id)
        if event.mechanism is not Mechanism.MOVE or event.sender is None:
            continue
        if not (state.people[event.sender].alive and state.people[delivery.recipient].alive):
            continue
        tie = state.tie_between(event.sender, delivery.recipient)
        if tie is None:
            continue
        if event.kind == DISTANCE:
            record = bind_distance(state, event, tie, params)
        elif event.kind == TRIANGLE:
            record = triangle_transfer(state, event, delivery.recipient, params)
        elif event.kind in (OVERFUNCTION, UNDERFUNCTION):
            record = functioning_shift(state, event, tie, delivery.recipient, params)
        elif event.kind == REDUCE_CUTOFF:
            record = reduce_cutoff(state, event, tie)
        else:
            record = None
        if record is not None:
            records.append(record)
    return records
