"""The life course — ages, acting, a child's development, launch, partners, births, deaths and the coach's career.

Purpose: carry a run through a family's life course on the slow tick, and the weekly search for work after a lost
         income, as spec revision 28 states them.
Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1, #M7.C.1a, #M7.C.1f, #M7.C.1g, #M7.C.1h, #M7.C.1i, #M7.C.1j,
         #M7.C.1k, #M7.B.1a, #M7.E.1f, #M7.E.1c, #M6.3, #M6.3a, #M6.3b, #M6.3c, #M6.3d, #M6.3e, #M6.3f, #M6.3g,
         #M6.3h, #M2.A.0k
Tests:   tests/bowen/test_life_course.py

Revision 28 applies the owner's decisions of 2026-10-13 (P13–P15); every form here is Claude's recommendation, pending
owner review, and every constant is `[I]` unless the owner stated it (acting from 10, development from 5 to 20 within
2% of the parents' level, the coach's ages, 20% less income).

A run has a life course only when its family declares one (``state.life`` is not None). Every run before Phase D part
II has none and is unchanged.

**Each slow tick** (``yearly``), after the estimator and its invariants, in this order:

1. everyone whose age the run tracks is a year older (`M7.C.1f`);
2. a child under the fixation age has its chronic anxiety derived: the parents' mean, plus ``witnessed_gain`` × the
   child's mean load over the year (`M7.B.1a`);
3. a child develops (`M7.E.1f`): held at the parents' mean before 5; anchored at 5; from 6 to 20 moved by
   ``development_rate × d`` within ``anchor × (1 ± development_band)``, where ``d`` is half the parents' mean
   functioning above basic over the year and half the child's own;
4. a child of ``launch_age_years`` leaves home, an earner and not dependent (`M7.C.1g`);
5. whoever has reached the age at death drawn for them at entry dies (`M7.C.1j`), with the disposition of `M6.3`;
6. a coach of ``coach_retirement_age_years`` retires and a successor enters (`M2.A.0k`);
7. a member reaching the pairing age of a partner row pairs (`M7.C.1h`);
8. a couple's children are born at the years their row declares (`M7.C.1i`).

The life events this produces are injected at the next week's delivery (``due_events``), so the estimator counts them
as nodal events, and grief and lost income arrive as stressors (`M1.F.6`).

**Each week** (``find_work``), after the acts: a member of a household with an unresolved loss of income finds work
with a chance set by the pressure on them (`M7.C.1k`).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

from src.bowen.engine.beliefs import TieBelief
from src.bowen.engine.contact import excess, initialise_contact
from src.bowen.engine.draws import DrawKey
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId, child_id
from src.bowen.engine.invariants import InvariantViolation
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.moves import initialise_functioning
from src.bowen.engine.objects import (
    REACTIVE_DETECTORS, SCALE_MAX, Family, Person, Relationship, Role, Sex, SiblingPosition, Sink, SymptomChannel,
)
from src.bowen.engine.outside_ness import initialise_outside_ness
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import ActiveStressor, RunState
from src.bowen.engine.stress_response import above_usual
from src.bowen.engine.symptoms import release_bound

DEATH, BIRTH, PARTNERING, LAUNCH, FIND_WORK, RETIREMENT = (
    "DEATH", "BIRTH", "PARTNERING", "LAUNCH", "FIND_WORK", "RETIREMENT")
GRIEF, INCOME_LOSS = "GRIEF", "INCOME_LOSS"


@dataclass(frozen=True)
class TieValues:
    """The starting values of a tie the life course adds, declared in the partner settings."""

    conductance: float
    bond_energy: float
    latency: int


@dataclass(frozen=True)
class PartnerRow:
    """One partner the settings declare (`M7.C.1h`), with the couple's children (`M7.C.1i`)."""

    id: PersonId
    partner_of: PersonId
    pair_age: int
    age_gap: int
    sex: Sex
    chronic_anxiety: float
    sibling_position: SiblingPosition
    channel_prior: SymptomChannel
    origin_id: PersonId
    origin_sex: Sex
    children_at_years: tuple[int, ...]
    children_sexes: tuple[Sex | None, ...]
    children_channels: tuple[SymptomChannel | None, ...]


@dataclass(frozen=True)
class LifeSettings:
    """What a family declares about its life course: partner rows, tie values, the coach and the coachee."""

    partners: tuple[PartnerRow, ...]
    spouse_tie: TieValues
    parent_tie: TieValues
    sibling_tie: TieValues
    origin_tie: TieValues
    coach: PersonId | None
    coachee: PersonId | None
    successor_sex: Sex


@dataclass
class LifeCourse:
    """Purpose: the life course's own run state — parents, anchors, ages at death, what is pending.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1
    Tests:   tests/bowen/test_life_course.py
    """

    settings: LifeSettings
    parents: dict[PersonId, tuple[PersonId, ...]]
    anchors: dict[PersonId, float] = field(default_factory=dict)
    death_age: dict[PersonId, int] = field(default_factory=dict)
    paired_at: dict[PersonId, int] = field(default_factory=dict)  # partner -> the slow tick they paired
    pending: list = field(default_factory=list)
    income: dict[str, list] = field(default_factory=dict)  # household -> [[event id, resolved], ...]
    coach: PersonId | None = None
    coach_template: tuple | None = None  # the original coach's (basic, chronic, channel prior, tie values)
    slow_ticks: int = 0


# --- entry -----------------------------------------------------------------------------------------------------


def age_at_death(u: float, age_now: int, params: EngineParams) -> int:
    """Purpose: the age at which a person alive at ``age_now`` dies, from the Gompertz hazard and one uniform.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1j
    Tests:   tests/bowen/test_life_course.py::test_m7c1j_hazard_rises_with_age

    With ``hazard(a) = a0 × exp(b × a)``, survival from ``age_now`` to ``x`` is
    ``exp(−(a0 / b) × (exp(b x) − exp(b age_now)))``; setting it to ``u`` gives ``x``.
    """
    a0, b = params.mortality_a, params.mortality_b
    x = math.log(math.exp(b * age_now) - (b / a0) * math.log(max(u, math.ulp(0.0)))) / b
    return max(age_now, math.floor(x))


def _draw_death(state: RunState, pid: PersonId, slow_tick: int, params: EngineParams) -> None:
    person = state.people[pid]
    if person.role is not Role.MEMBER or person.age_years is None or person.inactive:
        return
    u = state.draws.uniform(DrawKey.make("mortality", slow_tick=slow_tick, person=pid, purpose="age_at_death"))
    state.life.death_age[pid] = age_at_death(u, person.age_years, params)


def new_life_course(state: RunState, settings: LifeSettings, parents: dict, params: EngineParams) -> LifeCourse:
    """Purpose: start a run's life course: ages at death drawn for everyone at ``t0``, anchors for children past 5.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1j, #M7.E.1f, #M2.A.0k
    Tests:   tests/bowen/test_life_course.py::test_m7c1j_age_at_death_drawn_once_at_entry
    """
    life = LifeCourse(settings=settings, parents={k: tuple(v) for k, v in parents.items()}, coach=settings.coach)
    state.life = life
    for pid in sorted(state.people):
        _draw_death(state, pid, 0, params)
        person = state.people[pid]
        if pid in life.parents and person.age_years is not None \
                and params.development_start_age_years <= person.age_years <= params.development_end_age_years:
            life.anchors[pid] = _parents_mean(state, pid, "basic_level")
    if settings.coach is not None:
        coach = state.people[settings.coach]
        tie = state.tie_between(settings.coach, settings.coachee)
        life.coach_template = (coach.basic_level, coach.chronic_anxiety, coach.channel_prior,
                               TieValues(tie.conductance, tie.bond_energy, tie.latency))
    return life


def _living_parents(state: RunState, pid: PersonId) -> tuple[PersonId, ...]:
    return tuple(p for p in state.life.parents[pid] if state.people[p].alive)


def _parents_mean(state: RunState, pid: PersonId, name: str) -> float | None:
    """The living parents' mean; None when both have died (review 2026-10-13: a dead parent's frozen values and the
    history that stopped at their death are not this year's)."""
    parents = _living_parents(state, pid)
    return math.fsum(getattr(state.people[p], name) for p in parents) / len(parents) if parents else None


def _member(person: Person) -> bool:
    """A family member who takes part: not an external agent, not an inactive holder (review 2026-10-13)."""
    return person.alive and person.role is Role.MEMBER and not person.inactive


def total_anxiety(state: RunState) -> float:
    """Purpose: every anxiety stock M6.4 names, summed over the field: acute, tie-bound, triangle-bound, symptom-bound.
    Spec:    docs/bowen_agent_model_spec_v2.md#M6.I.6, #M11.C.37
    Tests:   tests/bowen/test_life_course.py::test_m11c37_anxiety_conserved_across_death
    """
    return (math.fsum(p.acute_anxiety + p.symptom_bound for p in state.people.values())
            + math.fsum(t.distance_bound_anxiety for t in state.ties.values())
            + math.fsum(t.bound_anxiety for t in state.triangles.values()))


def _introduce(state: RunState, person: Person, ties: list[Relationship], params: EngineParams) -> None:
    """A person and their ties enter the field with the starting state ``initialise_run`` gives founders."""
    state.add_person(person)
    initialise_outside_ness({person.id: person}, params)
    initialise_functioning({person.id: person}, {})
    for tie in ties:
        state.add_tie(tie)
    added = {t.id: t for t in ties}
    initialise_contact(state.people, added, params)
    initialise_functioning({}, added)
    resting = TieBelief(tension=0.0, contact=params.interactive_resting_contact)
    person.tie_beliefs = {tid: resting for tid in sorted(state.ties) if person.id not in tid.members()}
    for pid in sorted(state.people):
        other = state.people[pid]
        if pid != person.id and other.tie_beliefs is not None:
            for tid in added:
                if pid not in tid.members():
                    other.tie_beliefs[tid] = resting


def _new_person(pid: PersonId, role: Role, sex: Sex, household: str, basic: float, chronic: float,
                channel: SymptomChannel | None, sibling: SiblingPosition | None, age: int | None, *,
                dependent: bool, earner: bool, inactive: bool = False) -> Person:
    return Person(
        id=pid, role=role, sex=sex, household_id=household, basic_level=basic, functional_level=basic,
        chronic_anxiety=chronic, acute_anxiety=chronic, sibling_position=sibling, financially_dependent=dependent,
        channel_prior=channel, systems_perspective=0.0, reactive_state={name: 0.0 for name in REACTIVE_DETECTORS},
        age_years=age, earner=earner, inactive=inactive,
    )


def _tie(a: PersonId, b: PersonId, values: TieValues) -> Relationship:
    return Relationship(id=TieId.of(a, b), conductance=values.conductance, bond_energy=values.bond_energy,
                        latency=values.latency)


def _release_investment(state: RunState, tie: Relationship, holder: PersonId) -> None:
    """M6.3d: what ``holder`` invested in a tie that goes quiet moves to their other live ties, in proportion."""
    held = tie.investment.get(holder, 0.0)
    if not state.people[holder].alive or not held:
        return
    rest = [t for t in state.ties_of(holder) if t.id != tie.id and t.interactive
            and state.people[t.id.other(holder)].alive]
    tie.investment[holder] = 0.0
    if not rest:
        tie.investment[holder] = held  # nowhere else to put it: it stays where it was
        return
    base = math.fsum(t.investment.get(holder, 0.0) for t in rest)
    for t in rest:
        share = t.investment.get(holder, 0.0) / base if base > 0 else 1.0 / len(rest)
        t.investment[holder] = t.investment.get(holder, 0.0) + held * share


def _redistribute_investment(state: RunState, person: PersonId, new_tie: TieId, params: EngineParams) -> None:
    """M7.C.1a: a new claimant takes ``new_tie_investment_share`` of the person's existing investment, by construction."""
    others = [t for t in state.ties_of(person) if t.id != new_tie]
    total = math.fsum(t.investment.get(person, 0.0) for t in others)
    for t in others:
        if person in t.investment:
            t.investment[person] *= 1.0 - params.new_tie_investment_share
    state.ties[new_tie].investment[person] = params.new_tie_investment_share * total


def _life_event(state: RunState, kind: str, origin: str, targets, *, exogenous: bool = True,
                intensity: float = 0.0, duration: int = 1, mechanism: Mechanism = Mechanism.LIFE_EVENT) -> Event:
    """A life event has no sender: it reaches its targets directly, so it needs no tie to them (the 2026-10-13
    ensemble crashed when FIND_WORK named its taker as sender and a sibling with no tie to them as a target)."""
    tick = state.tick + 1  # injected at the next week's delivery
    index = sum(1 for e in state.life.pending if e.id.tick == tick and e.id.origin == origin)
    return Event(
        id=EventId(tick, origin, index), kind=kind, mechanism=mechanism, sender=None,
        targets=tuple(sorted(set(targets))), intensity=intensity, timestamp=tick,
        duration=duration, exogenous=exogenous, source_position=SourcePosition.NONE,
        channel=Channel.EXOGENOUS if exogenous else Channel.SELF_DIRECTED,
    )


def due_events(state: RunState) -> tuple[Event, ...]:
    """Purpose: the life events whose week this is, injected at step 2 with the source's own.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1, #M1.F.6
    Tests:   tests/bowen/test_life_course.py::test_m63g_grief_lasts_one_to_three_years_by_bond_energy
    """
    if state.life is None:
        return ()
    stale = [e.id for e in state.life.pending if e.timestamp < state.tick]
    if stale:  # review 2026-10-13: a queued life event must never be dropped unseen
        raise RuntimeError(f"life events missed their week: {stale}")
    due = tuple(e for e in state.life.pending if e.timestamp == state.tick)
    state.life.pending = [e for e in state.life.pending if e.timestamp > state.tick]
    return due


# --- the slow tick ---------------------------------------------------------------------------------------------


def yearly(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: the life course's part of the slow tick, in the module docstring's order.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1, #M3.B.1
    Tests:   tests/bowen/test_life_course.py
    """
    life = state.life
    if life is None:
        return []
    life.slow_ticks += 1
    records: list[EffectRecord] = []
    rows: list[tuple] = []
    for pid in sorted(state.people):
        person = state.people[pid]
        if person.alive and person.age_years is not None:
            person.age_years += 1
            rows.append((pid, "age_years", 1.0))
    rows += _childhood(state, params)
    rows += _launch(state, params)
    if rows:
        records.append(EffectRecord(state.tick, "life_course", None, people=tuple(rows)))
    for pid in sorted(life.death_age):
        person = state.people[pid]
        if person.alive and person.age_years is not None and person.age_years >= life.death_age[pid]:
            before, held = total_anxiety(state), person.acute_anxiety
            records.append(die(state, pid, params))
            # M6.I.6 across the death, checked at run time (review 2026-10-13): only what death_transfer_share
            # discards may leave the field.
            lost = before - total_anxiety(state)
            if abs(lost - (1.0 - params.death_transfer_share) * held) > params.invariant_tolerance:
                raise InvariantViolation(f"tick {state.tick}: {pid}'s death changed total anxiety by {-lost:+.6g} "
                                         "(M6.I.6, M11.C.37)")
    records += _retire_coach(state, params)
    records += _pair(state, params)
    records += _births(state, params)
    return records


def _year_mean(state: RunState, pid: PersonId, params: EngineParams, read) -> float | None:
    weeks = list(state.estimator_history.get(pid, ()))[-params.slow_tick_fast_ticks:]
    return math.fsum(read(w) for w in weeks) / len(weeks) if weeks else None


def _childhood(state: RunState, params: EngineParams) -> list[tuple]:
    """M7.B.1a and M7.E.1f: a child's chronic anxiety before 12, and its basic level from 5 to 20. Only living parents
    are read; an orphan keeps the values they have and develops by their own functioning alone (review 2026-10-13)."""
    life, rows = state.life, []
    for pid in sorted(life.parents):
        person = state.people[pid]
        if not person.alive or person.age_years is None:
            continue
        age = person.age_years
        parents_chronic, parents_basic = (_parents_mean(state, pid, "chronic_anxiety"),
                                          _parents_mean(state, pid, "basic_level"))
        # The age was just advanced, so the year lived at fixation − 1 is read here, then the value is fixed.
        if age <= params.chronic_anxiety_fixation_age_years and parents_chronic is not None:
            load = _year_mean(state, pid, params, lambda w: w.load) or 0.0
            chronic = parents_chronic + params.witnessed_gain * load
            if chronic != person.chronic_anxiety:
                rows.append((pid, "chronic_anxiety", chronic - person.chronic_anxiety))
                person.chronic_anxiety = chronic
        if age > params.development_end_age_years:
            continue
        before = person.basic_level
        if age < params.development_start_age_years:
            if parents_basic is not None:
                person.basic_level = parents_basic
        elif pid not in life.anchors:
            life.anchors[pid] = parents_basic if parents_basic is not None else person.basic_level
            person.basic_level = life.anchors[pid]
        else:
            anchor = life.anchors[pid]
            own = _year_mean(state, pid, params, lambda w: w.functional_level)
            theirs = [m for p in _living_parents(state, pid)
                      if (m := _year_mean(state, p, params, lambda w: w.functional_level)) is not None]
            if own is None:
                continue
            terms = [own - person.basic_level]  # the child's own functioning above basic
            if theirs and parents_basic is not None:
                terms.append(math.fsum(theirs) / len(theirs) - parents_basic)  # and the living parents'
            d = math.fsum(terms) / len(terms)
            low, high = anchor * (1.0 - params.development_band), anchor * (1.0 + params.development_band)
            person.basic_level = min(high, max(low, person.basic_level + params.development_rate * d))
        if person.basic_level != before:
            rows.append((pid, "basic_level", person.basic_level - before))
    return rows


def _launch(state: RunState, params: EngineParams) -> list[tuple]:
    """M7.C.1g: at the launch age a child leaves home, an earner and no longer dependent."""
    rows = []
    for pid in sorted(state.life.parents):
        person = state.people[pid]
        parents = state.life.parents[pid]
        at_home = any(state.people[p].household_id == person.household_id for p in parents)
        if person.alive and person.age_years == params.launch_age_years and at_home:
            person.household_id = pid.value
            person.financially_dependent = False
            person.earner = True
            rows.append((pid, "launched", 1.0))
            state.life.pending.append(_life_event(state, LAUNCH, f"life:launch:{pid}", (pid,)))
    return rows


# --- death -------------------------------------------------------------------------------------------------------


def die(state: RunState, pid: PersonId, params: EngineParams) -> EffectRecord:
    """Purpose: a death and its disposition, `M6.3a`–`M6.3h`, in order.
    Spec:    docs/bowen_agent_model_spec_v2.md#M6.3a, #M6.3b, #M6.3c, #M6.3d, #M6.3f, #M6.3g, #M6.3h
    Tests:   tests/bowen/test_life_course.py::test_m11c37_anxiety_conserved_across_death
    """
    person = state.people[pid]
    ties = state.ties_of(pid)
    acute, tie_rows, tri_rows, people_rows = [], [], [], []

    # M6.3b — the dead person's acute anxiety to the living family members, by bond energy.
    weights = {}
    for tie in ties:
        other = tie.id.other(pid)
        if _member(state.people[other]):
            weights[other] = weights.get(other, 0.0) + tie.bond_energy
    if math.fsum(weights.values()) <= 0:
        household = [p for p in sorted(state.people) if p != pid and _member(state.people[p])
                     and state.people[p].household_id == person.household_id]
        anyone = [p for p in sorted(state.people) if p != pid and _member(state.people[p])]
        weights = {p: 1.0 for p in (household or anyone)}
    if not weights:
        raise RuntimeError(f"{pid}'s death leaves their anxiety nowhere to go: no living family member (M6.3b)")
    # death_transfer_share is 1 by the rule; only M11.C.37's mutant sets it lower. Below 1 the rest is discarded and
    # M11.C.37 fails.
    held_acute, total = person.acute_anxiety, math.fsum(weights.values())
    stock = held_acute * params.death_transfer_share
    person.acute_anxiety = 0.0
    acute.append((pid, -held_acute))
    for other in sorted(weights):
        share = stock * weights[other] / total
        state.people[other].acute_anxiety += share
        acute.append((other, share))

    # M6.3c — what the dead person's ties, triangles and symptoms held.
    for tie in ties:
        if tie.distance_bound_anxiety:
            other = tie.id.other(pid)
            held, tie.distance_bound_anxiety = tie.distance_bound_anxiety, 0.0
            state.people[other].acute_anxiety += held
            acute.append((other, held))
            tie_rows.append((tie.id, "distance_bound_anxiety", -held))
    for tri_id in sorted(state.triangles):
        if pid not in tri_id.members:
            continue
        triangle = state.triangles[tri_id]
        living = [m for m in tri_id.members if m != pid and state.people[m].alive]
        if triangle.bound_anxiety and living:
            held, triangle.bound_anxiety = triangle.bound_anxiety, 0.0
            for m in living:
                state.people[m].acute_anxiety += held / len(living)
                acute.append((m, held / len(living)))
            tri_rows.append((tri_id, "bound_anxiety", f"-{held}"))
        triangle.active = False
    moved = release_bound(state, pid)  # as when a symptom ends (M7.D.2a); nothing if no partner is left
    if moved:
        acute += list(moved)
        people_rows.append((pid, "symptom_bound", -math.fsum(v for _, v in moved)))

    # M6.3d — each living person's investment in the tie to the dead goes to their other ties.
    for tie in ties:
        _release_investment(state, tie, tie.id.other(pid))

    # M6.3f — sink allocations that named the dead return to their family, logged. The allocation is per sink, not per
    # child, so a dead child returns the whole child-projection allocation (an approximation the spec states).
    sink_rows = []
    for family in state.families():
        prefix = "" if family is state.family else f"{family.household}/"
        if family.marital_tie is not None and pid in family.marital_tie.members():
            named = (Sink.MARITAL_CONFLICT, Sink.SPOUSE_DYSFUNCTION)
            if not any(state.people[m].alive for m in family.marital_tie.members() if m != pid):
                family.dissolved = True
        elif any(pid in t.members() for t in family.parent_child_ties):
            named = (Sink.CHILD_PROJECTION,)
        else:
            continue
        for sink in named:
            if family.sink_allocations[sink]:
                sink_rows.append((f"{prefix}sink:{sink.value}", -family.sink_allocations[sink]))
                family.sink_allocations[sink] = 0.0

    mourners = sorted(t.id.other(pid) for t in ties if t.interactive and _member(state.people[t.id.other(pid)]))
    bond = {t.id.other(pid): t.bond_energy for t in ties}
    person.alive = False
    for tie in ties:  # M6.3a: the ties stay in the field, carrying nothing
        tie.interactive = False
    life = state.life
    if mourners:  # a nodal event must reach someone (slow_tick.reached_by); a death no one survives is logged below only
        life.pending.append(_life_event(state, DEATH, f"life:death:{pid}", mourners))
    for other in mourners:  # M6.3g — grief, in proportion to the tie
        share = bond[other] / SCALE_MAX
        weeks = params.grief_min_weeks + round(params.grief_extra_weeks * share)
        life.pending.append(_life_event(state, GRIEF, f"life:grief:{pid}", (other,), intensity=params.grief_scale * share,
                                        duration=weeks, mechanism=Mechanism.EXOGENOUS_STRESSOR))
    if person.earner:  # M6.3h — a lost income: a stressor with no end until someone finds work
        household = sorted(p for p, q in state.people.items() if q.alive and q.role is Role.MEMBER
                           and q.household_id == person.household_id)
        if household:
            # It arrives as any stressor does, and then weighs with no end (M6.3h amends M1.F.6 for it), so the life
            # course registers its later weeks itself.
            event = _life_event(state, INCOME_LOSS, f"life:income:{pid}", household,
                                intensity=params.income_loss_intensity, mechanism=Mechanism.EXOGENOUS_STRESSOR)
            life.pending.append(event)
            life.income.setdefault(person.household_id, []).append([event.id, False])
            state.stressors += [ActiveStressor(event.id, member, event.intensity, event.timestamp + 1, None)
                                for member in household]
    people_rows.append((pid, "alive", -1.0))
    return EffectRecord(state.tick, "death", None, acute_anxiety=tuple(acute), ties=tuple(tie_rows),
                        triangles=tuple(tri_rows), people=tuple(people_rows), sinks=tuple(sink_rows))


# --- the weekly search for work ----------------------------------------------------------------------------------


def find_work(state: RunState, load: dict, params: EngineParams) -> list[EffectRecord]:
    """Purpose: a member pressed by a lost income finds work, at 20% less income (`M7.C.1k`, `M6.3h`).
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1k, #M6.3h
    Tests:   tests/bowen/test_life_course.py::test_m7c1k_find_work_leaves_twenty_percent_and_sets_earner
    """
    life = state.life
    if life is None:
        return []
    records = []
    for household in sorted(life.income):
        open_events = [entry for entry in life.income[household] if not entry[1]]
        if not open_events:
            continue
        for pid in sorted(state.people):
            person = state.people[pid]
            if (not person.alive or person.inactive or person.role is not Role.MEMBER
                    or person.household_id != household
                    or (person.age_years is not None and person.age_years < max(params.working_age_years,
                                                                                 params.acting_age_years))):
                continue
            pressure = min(1.0, max(0.0, above_usual(person, load.get(pid, 0.0))) / params.find_work_pressure_scale)
            u = state.draws.uniform(DrawKey.make("life_course", tick=state.tick, person=pid, purpose="find_work"))
            if u >= params.find_work_rate * pressure:
                continue
            ids = {entry[0] for entry in open_events}
            state.stressors = [dataclasses.replace(s, intensity=s.intensity * params.income_shortfall_share)
                               if s.event_id in ids else s for s in state.stressors]
            for entry in open_events:
                entry[1] = True
            person.earner = True
            person.financially_dependent = False
            household_now = [p for p, q in state.people.items() if q.alive and q.household_id == household]
            life.pending.append(_life_event(state, FIND_WORK, f"life:work:{pid}", household_now, exogenous=False))
            records.append(EffectRecord(state.tick, "find_work", None, people=((pid, "earner", 1.0),)))
            break
    return records


# --- the coach's career --------------------------------------------------------------------------------------------


def _retire_coach(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """M2.A.0k: at the retirement age the coach's tie goes quiet and a successor starts over with the coachee."""
    life = state.life
    if life.coach is None:
        return []
    coach = state.people[life.coach]
    if coach.inactive or coach.age_years is None or coach.age_years < params.coach_retirement_age_years:
        return []
    coachee = life.settings.coachee
    coach.inactive = True
    if not state.people[coachee].alive:  # nobody left to coach: the coach retires and no successor enters
        for tie in state.ties_of(coach.id):
            tie.interactive = False
        return [EffectRecord(state.tick, "coach_retirement", None, people=((coach.id, "inactive", 1.0),))]
    for tie in state.ties_of(coach.id):
        _release_investment(state, tie, tie.id.other(coach.id))  # as at a death (M6.3d): the coachee's goes elsewhere
        tie.interactive = False
    basic, chronic, channel, values = life.coach_template
    successor = _new_person(PersonId(f"{coach.id.value}_successor"), Role.EXTERNAL, life.settings.successor_sex,
                            f"{coach.id.value}_successor", basic, chronic, channel, None,
                            params.coach_start_age_years, dependent=False, earner=True)
    _introduce(state, successor, [_tie(successor.id, coachee, values)], params)
    _redistribute_investment(state, coachee, TieId.of(successor.id, coachee), params)  # M7.C.1a
    life.coach = successor.id
    life.pending.append(_life_event(state, RETIREMENT, f"life:retire:{coach.id}", (coachee,)))
    return [EffectRecord(state.tick, "coach_retirement", None,
                         people=((coach.id, "inactive", 1.0), (successor.id, "entered", 1.0)))]


# --- partners and births -------------------------------------------------------------------------------------------


def _family_budget(state: RunState, a: PersonId, b: PersonId, params: EngineParams) -> float:
    mean = math.fsum(state.people[p].basic_level for p in (a, b)) / len((a, b))
    return params.budget_per_undifferentiation * (SCALE_MAX - mean)


def _origin_ties(state: RunState, person: PersonId) -> tuple[TieId, ...]:
    parents = state.life.parents.get(person, ())
    siblings = {c for c, ps in state.life.parents.items() if c != person and ps and set(ps) == set(parents)}
    return tuple(sorted(t.id for t in state.ties_of(person) if t.id.other(person) in set(parents) | siblings))


def _pair(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """M7.C.1h: a member reaching a partner row's pairing age pairs with that partner, who enters from the settings."""
    life, settings, records = state.life, state.life.settings, []
    for row in settings.partners:
        member = state.people.get(row.partner_of)
        if row.id in state.people or member is None or not member.alive or member.age_years != row.pair_age:
            continue
        u = state.draws.uniform(DrawKey.make("life_course", tick=state.tick, person=row.id, purpose="partner_level"))
        tolerance = params.spouse_basic_level_tolerance
        basic = min(SCALE_MAX, max(0.0, member.basic_level + (u - (1.0 - u)) * tolerance))  # uniform in ±tolerance
        household = f"{row.partner_of.value}_{row.id.value}"
        partner = _new_person(row.id, Role.MEMBER, row.sex, household, basic, row.chronic_anxiety, row.channel_prior,
                              row.sibling_position, member.age_years + row.age_gap, dependent=False, earner=True)
        holder = _new_person(row.origin_id, Role.MEMBER, row.origin_sex, row.origin_id.value, basic,
                             row.chronic_anxiety, row.channel_prior, None, None, dependent=False, earner=True,
                             inactive=True)
        member.household_id = household
        _introduce(state, holder, [], params)
        spouse = _tie(row.partner_of, row.id, settings.spouse_tie)
        _introduce(state, partner, [spouse, _tie(row.id, row.origin_id, settings.origin_tie)], params)
        _redistribute_investment(state, row.partner_of, spouse.id, params)
        state.other_families[household] = Family(
            undifferentiation_budget=_family_budget(state, row.partner_of, row.id, params),
            marital_tie=spouse.id, origin_ties=_origin_ties(state, row.partner_of) + (TieId.of(row.id, row.origin_id),),
            household=household,
        )
        life.paired_at[row.id] = life.slow_ticks
        _draw_death(state, row.id, life.slow_ticks, params)
        life.pending.append(_life_event(state, PARTNERING, f"life:pair:{row.id}", (row.partner_of, row.id)))
        records.append(EffectRecord(state.tick, "partnering", None,
                                    people=((row.id, "entered", 1.0), (row.origin_id, "entered", 1.0))))
    return records


def _births(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """M7.C.1i: a couple's children are born at the years after pairing that their row declares."""
    life, records = state.life, []
    for row in life.settings.partners:
        if row.id not in life.paired_at:
            continue
        a, b = row.partner_of, row.id
        if not (state.people[a].alive and state.people[b].alive):
            continue
        years = life.slow_ticks - life.paired_at[row.id]
        for order, at in enumerate(row.children_at_years, start=1):
            if at != years:
                continue
            records.append(_birth(state, row, order, params))
    return records


def _birth(state: RunState, row: PartnerRow, order: int, params: EngineParams) -> EffectRecord:
    life = state.life
    a, b = row.partner_of, row.id
    pid = child_id(a, b, order)
    sex = row.children_sexes[order - 1]
    if sex is None:
        u = state.draws.uniform(DrawKey.make("life_course", tick=state.tick, person=pid, purpose="sex"))
        sexes = sorted(Sex, key=lambda x: x.value)
        sex = sexes[min(len(sexes) - 1, math.floor(u * len(sexes)))]
    channel = row.children_channels[order - 1]
    if channel is None:
        u = state.draws.uniform(DrawKey.make("life_course", tick=state.tick, person=pid, purpose="channel"))
        channels = list(SymptomChannel)
        channel = channels[min(len(channels) - 1, math.floor(u * len(channels)))]
    life.parents[pid] = (a, b)
    household = state.people[a].household_id
    basic = _parents_mean(state, pid, "basic_level")
    child = _new_person(pid, Role.MEMBER, sex, household, basic, _parents_mean(state, pid, "chronic_anxiety"), channel,
                        SiblingPosition(order, len(row.children_at_years)), 0, dependent=True, earner=False)
    siblings = sorted(c for c, ps in life.parents.items() if c != pid and set(ps) == {a, b} and state.people[c].alive)
    settings = life.settings
    ties = [_tie(p, pid, settings.parent_tie) for p in (a, b)] + [_tie(s, pid, settings.sibling_tie) for s in siblings]
    _introduce(state, child, ties, params)
    family = state.other_families.get(household)
    # M7.E.1c: the firstborn; the firstborn of its sex; the last-born; a birth at a time of high stress (a parent above
    # their usual level). A reality defect is not modelled.
    earlier = [child_id(a, b, k) for k in range(1, order)]
    earlier_sexes = {state.people[c].sex for c in earlier if c in state.people}  # living or dead (review 2026-10-13)
    situations = (order == 1) + (order > 1 and sex not in earlier_sexes) + (order == len(row.children_at_years))
    for sibling in siblings:  # M7.C.1a for the older siblings too (review 2026-10-13)
        _redistribute_investment(state, sibling, TieId.of(sibling, pid), params)
    for parent in (a, b):
        tid = TieId.of(parent, pid)
        _redistribute_investment(state, parent, tid, params)
        stressed = above_usual(state.people[parent], excess(state.people[parent])) > 0
        state.ties[tid].investment[parent] *= 1.0 + params.birth_focus_gain * (situations + stressed)
    if family is not None:
        family.parent_child_ties = tuple(sorted(family.parent_child_ties + tuple(TieId.of(p, pid) for p in (a, b))))
    _draw_death(state, pid, life.slow_ticks, params)
    life.pending.append(_life_event(state, BIRTH, f"life:birth:{pid}", (a, b, *siblings)))
    return EffectRecord(state.tick, "birth", None, people=((pid, "entered", 1.0),))
