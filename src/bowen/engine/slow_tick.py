"""The slow tick, and the basic-level estimator (Phase D plan B5, B6).

Purpose: once a year (every ``slow_tick_fast_ticks`` fast ticks, the first at the end of week 51), estimate each
         person's ``basic_level`` from their functioning history, lower chronic anxiety only with a rise, wear each
         tie by its members' chronic anxiety, and age each dyad; then check the slow-clock invariants.
Spec:    docs/bowen_agent_model_spec_v2.md#M3.B.1, #M1.A.4a, #M1.A.4b, #M1.A.4d, #M1.A.4h, #M1.A.4k, #M7.A.1,
         #M7.A.1a, #M7.B.1, #M7.E.4, #M2.A.0b, #M6.I.5
Tests:   tests/bowen/test_slow_tick.py

**The history** (``record_week``, at the end of every fast tick): per person, the week's ``functional_level``, the
load they carried when they chose (excess acute anxiety over the chronic floor), whether they acted in the
self-directed channel or withheld, and the kinds of nodal event — exogenous stressor — delivered to them. It keeps the last
``estimator_window`` slow ticks of weeks.

**The estimate** (``estimate``; owner decision P4, spec ``M1.A.4k``). Once the window is full:

    m, sd = the load-weighted mean and standard deviation of functional_level        (M1.A.4b)
    b     = the number of distinct kinds of nodal event in the window                (breadth)
    q     = the load-weighted share of weeks acted self-directed or withheld          (M1.A.4h)
    e     = m − estimator_sd_weight × sd − basic_level

``basic_level`` rises by ``estimator_rate × (basic_level / 100) × q × e`` when ``e > 0``, ``b ≥ estimator_breadth``
and the person is not financially dependent (``M7.A.1a``). ``basic_level / 100`` is ``M1.A.4d``'s falloff toward the
bottom of the scale, with no hard gate. A benign window carries no load, so it carries no evidence. ``basic_level``
does not fall in D-I (plan B6). Every form and constant is [I].

**Chronic anxiety** (``M7.B.1``, owner decision P5) changes only with a rise in ``basic_level``, keeping its share of
the distance to the top of the scale: ``chronic × (100 − new) / (100 − old)``. It never drifts and never rises.

**Tie wear** (``M7.E.4``, P5, P6): each slow tick a tie's conductance falls by ``tie_wear_rate × mean chronic / 100``
of itself, less ``wear_offset_per_act`` of that for each ``STAY-IN-CONTACT`` sent on it over the year, to none. Bond
energy is untouched (``M1.B.4``). Only a ``STAY-IN-CONTACT`` that reached the other counts: one refused under
``M1.B.3c`` offsets nothing.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from src.bowen.engine.act import SELF_CHANNEL, WITHHOLD
from src.bowen.engine.events import Mechanism, Role
from src.bowen.engine.iposition import STAY_IN_CONTACT
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.invariants import InvariantViolation
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import SCALE_MAX
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState



@dataclass(frozen=True)
class Week:
    """One fast tick of one person's history, as the estimator reads it."""

    tick: int
    functional_level: float
    load: float
    self_directed: bool
    nodal: tuple[str, ...]  # the kinds of nodal event delivered that week: a recurring stressor is one situation


def history_length(params: EngineParams) -> int:
    return params.estimator_window * params.slow_tick_fast_ticks


def record_week(state: RunState, params: EngineParams, load: dict, selections, batch) -> None:
    """Purpose: add this week to each living person's estimator history.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.4k, #M1.A.4b, #M1.A.4h
    Tests:   tests/bowen/test_slow_tick.py::test_m1a4k_history_records_load_channel_and_nodal_events

    ``load`` is each person's excess acute anxiety at selection; ``selections`` are the week's outcomes; ``batch``
    is the week's accepted deliveries. The channel is the kind's, whoever decided it: a scripted act counts by what
    it is (a criterion arm that scripts ``I-POSITION`` or exchanges is declaring that behaviour, M11.C.18, M11.C.33),
    although the log stamps it ``scripted``.
    """
    chose = {s.actor: s.kind for s in selections}
    nodal: dict[PersonId, set[str]] = {}
    for delivery in batch:
        event = state.store.event(delivery.event_id)
        if event.mechanism is Mechanism.EXOGENOUS_STRESSOR:
            nodal.setdefault(delivery.recipient, set()).add(event.kind)
    size = history_length(params)
    for pid in sorted(state.people):
        person = state.people[pid]
        if not person.alive:
            continue
        kind = chose.get(pid)
        self_directed = kind == WITHHOLD or (kind is not None and state.kinds.channels.get(kind) == SELF_CHANNEL)
        weeks = state.estimator_history.setdefault(pid, deque(maxlen=size))
        weeks.append(Week(state.tick, person.functional_level, load.get(pid, 0.0), self_directed,
                          tuple(sorted(nodal.get(pid, ())))))


@dataclass(frozen=True)
class Estimate:
    mean: float
    sd: float
    breadth: int
    self_share: float
    evidence: float
    rise: float


def estimate(weeks, basic_level: float, dependent: bool, params: EngineParams) -> Estimate | None:
    """Purpose: the estimator's reading of one person's window, and the rise it supports.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.4a, #M1.A.4b, #M1.A.4d, #M1.A.4h, #M1.A.4k, #M7.A.1a
    Tests:   tests/bowen/test_slow_tick.py::test_m1a4b_a_benign_window_does_not_raise_basic_level,
             tests/bowen/test_slow_tick.py::test_m1a4k_a_loaded_stable_broad_window_raises_basic_level

    ``None`` while the window is not full, or when it carried no load: no evidence either way.
    """
    if len(weeks) < history_length(params):
        return None
    total = sum(w.load for w in weeks)
    if total <= 0:
        return None
    mean = sum(w.load * w.functional_level for w in weeks) / total
    sd = math.sqrt(sum(w.load * (w.functional_level - mean) * (w.functional_level - mean) for w in weeks) / total)
    breadth = len({event for w in weeks for event in w.nodal})
    self_share = sum(w.load for w in weeks if w.self_directed) / total
    evidence = mean - params.estimator_sd_weight * sd - basic_level
    supported = evidence > 0 and breadth >= params.estimator_breadth and not dependent
    rise = params.estimator_rate * (basic_level / SCALE_MAX) * self_share * evidence if supported else 0.0
    return Estimate(mean, sd, breadth, self_share, evidence, min(rise, SCALE_MAX - basic_level))


def run_slow_tick(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: the slow tick's contents, in order: the estimator, chronic anxiety, tie wear, dyad age.
    Spec:    docs/bowen_agent_model_spec_v2.md#M3.B.1, #M7.A.1, #M7.B.1, #M7.E.4
    Tests:   tests/bowen/test_slow_tick.py::test_m3b1_slow_tick_fires_at_the_end_of_week_51_not_at_week_0

    The life-stage update (``M7.C.1``) is a D-II hook and does nothing here.
    """
    before = {pid: (p.basic_level, p.chronic_anxiety) for pid, p in state.people.items()}
    people = []
    for pid in sorted(state.people):
        person = state.people[pid]
        if not person.alive:
            continue
        reading = estimate(state.estimator_history.get(pid, ()), person.basic_level, person.financially_dependent,
                           params)
        if reading is None or reading.rise <= 0:
            continue
        old = person.basic_level
        person.basic_level = old + reading.rise
        person.chronic_anxiety *= (SCALE_MAX - person.basic_level) / (SCALE_MAX - old)  # M7.B.1: falls with the rise
        people += [(pid, "basic_level", person.basic_level - old),
                   (pid, "chronic_anxiety", person.chronic_anxiety - before[pid][1])]
    ties = []
    year = state.tick - params.slow_tick_fast_ticks
    for tid in sorted(state.ties):
        tie = state.ties[tid]
        a, b = tid.members()
        chronic = math.fsum(state.people[m].chronic_anxiety for m in (a, b)) / len((a, b))
        # Only contact that reached the other this year counts, by when it arrived: a STAY-IN-CONTACT refused under
        # M1.B.3c offsets nothing, and one sent in the year's last week counts in the year it lands.
        contact = 0
        for member, other in ((a, b), (b, a)):
            for d in state.store.delivered_to(member, Role.TARGET):
                if d.delivered_tick > year:
                    event = state.store.event(d.event_id)
                    contact += event.kind == STAY_IN_CONTACT and event.sender == other
        share = max(0.0, 1.0 - params.wear_offset_per_act * contact)
        worn = tie.conductance * params.tie_wear_rate * chronic / SCALE_MAX * share
        if worn:
            tie.conductance -= worn
            ties.append((tid, "conductance", -worn))
        tie.dyad_age += 1
        ties.append((tid, "dyad_age", 1.0))
    assert_slow_invariants(state, before)
    return [EffectRecord(state.tick, "slow_tick", None, ties=tuple(ties), people=tuple(people))]


def assert_slow_invariants(state: RunState, before: dict) -> None:
    """Purpose: raise if the slow tick lowered a basic level, raised a dependent person's, or raised chronic anxiety.
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.A.1a, #M7.B.1, #M1.A.4a
    Tests:   tests/bowen/test_slow_tick.py::test_m7b1_slow_invariants_catch_a_rise_in_chronic_anxiety
    """
    for pid, (basic, chronic) in before.items():
        person = state.people[pid]
        if person.basic_level < basic:
            raise InvariantViolation(f"tick {state.tick}: {pid} basic_level fell on the slow tick (plan B6)")
        if person.basic_level > basic and person.financially_dependent:
            raise InvariantViolation(f"tick {state.tick}: M7.A.1a violated — {pid} rose while dependent")
        if person.chronic_anxiety > chronic:
            raise InvariantViolation(f"tick {state.tick}: M7.B.1 violated — {pid} chronic anxiety rose")
