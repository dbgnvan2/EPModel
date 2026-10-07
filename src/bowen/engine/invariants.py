"""The M6 invariants, asserted at the end of every fast tick.

Purpose: raise — never warn — when a tick leaves the model in a state the
         invariants forbid.
Spec:    docs/bowen_agent_model_spec_v2.md#M6, #M4.G.2, #M4.G.2a, #M6.1
Tests:   tests/bowen/test_mechanisms.py::test_m4g2_invariants_asserted_every_tick

What each check means **in Phase B**, where little moves the quantities several
invariants govern (the report must say they hold trivially there):

* M6.I.1 — exactly three sinks, none negative, their total no larger than the
  budget, and their total unchanged by the tick (nothing reallocates in Phase B).
* M6.I.2 — the sinks are exactly the three, so distance is not one of them.
* M6.I.3 — ``life_energy_ratio`` is unset or within [0, 1].
* M6.I.4 — total pseudo-self is unchanged by the tick.
* M6.I.5 — ``basic_level`` is unchanged by a fast tick: solid self does not
  move in exchange (only the slow-tick estimator writes it).
* M6.I.6 — **disabled** (M4.G.2a). Kept as a named check that raises if enabled.
* M6.I.7 — no person and no tie has left the field.
* M6.I.8 — the standing load ran before delivery this tick, on every tie.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import InvariantRecord, InvariantStatus
from src.bowen.engine.objects import Sink
from src.bowen.engine.state import RunState


class InvariantViolation(AssertionError):
    """M6: a violation raises."""


class M6I6NotRestated(NotImplementedError):
    """M4.G.2a: M6.I.6 is unsatisfiable as worded and may not be enabled until restated."""


@dataclass(frozen=True)
class Snapshot:
    people: frozenset[PersonId]
    ties: frozenset[TieId]
    basic_levels: tuple[tuple[PersonId, float], ...]
    pseudo_self_total: float
    sink_total: float


def snapshot(state: RunState) -> Snapshot:
    return Snapshot(
        people=frozenset(state.people),
        ties=frozenset(state.ties),
        basic_levels=tuple((p, state.people[p].basic_level) for p in sorted(state.people)),
        pseudo_self_total=sum(p.pseudo_self or 0.0 for p in state.people.values()),
        sink_total=sum(state.family.sink_allocations.values()),
    )


def check_m6i6(enabled: bool = False) -> InvariantStatus:
    if enabled:
        raise M6I6NotRestated("M6.I.6 must be restated before it is asserted (M4.G.2a, external review A3)")
    return InvariantStatus.DISABLED


def assert_invariants(
    state: RunState,
    before: Snapshot,
    loaded_ties: frozenset[TieId],
    steps: tuple[str, ...],
    tolerance: float,
) -> InvariantRecord:
    """Purpose: assert every M6 invariant except M6.I.6, and record the results.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.G.2, #M4.G.2a, #M6.1
    Tests:   tests/bowen/test_mechanisms.py::test_m4g2_invariants_asserted_every_tick
    """
    sinks = state.family.sink_allocations

    def require(condition: bool, invariant: str, detail: str) -> tuple[str, InvariantStatus]:
        if not condition:
            raise InvariantViolation(f"tick {state.tick}: {invariant} violated — {detail}")
        return (invariant, InvariantStatus.PASSED)

    total = sum(sinks.values())
    results = [
        require(
            set(sinks) == set(Sink)
            and all(v >= -tolerance for v in sinks.values())
            and total <= state.family.undifferentiation_budget + tolerance
            and abs(total - before.sink_total) <= tolerance,
            "M6.I.1",
            f"sinks {dict(sinks)} against budget {state.family.undifferentiation_budget}",
        ),
        require(len(sinks) == 3 and not any("distance" in s.value for s in sinks), "M6.I.2", "a fourth sink"),
        require(
            all(p.life_energy_ratio is None or 0.0 <= p.life_energy_ratio <= 1.0 for p in state.people.values()),
            "M6.I.3",
            "life_energy_ratio outside [0, 1]",
        ),
        require(
            abs(sum(p.pseudo_self or 0.0 for p in state.people.values()) - before.pseudo_self_total) <= tolerance,
            "M6.I.4",
            "total pseudo-self changed",
        ),
        require(
            tuple((p, state.people[p].basic_level) for p in sorted(state.people)) == before.basic_levels,
            "M6.I.5",
            "basic_level moved within a fast tick",
        ),
        ("M6.I.6", check_m6i6()),
        require(
            frozenset(state.people) >= before.people and frozenset(state.ties) >= before.ties,
            "M6.I.7",
            "a person or tie left the field",
        ),
        require(
            loaded_ties == frozenset(state.ties)
            and "standing_load" in steps
            and "deliver" in steps
            and steps.index("standing_load") < steps.index("deliver"),
            "M6.I.8",
            f"standing load covered {len(loaded_ties)} of {len(state.ties)} ties; steps {steps}",
        ),
    ]
    return InvariantRecord(tick=state.tick, results=tuple(results))
