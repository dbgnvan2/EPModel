"""The M6 invariants, asserted at the end of every fast tick, with M6.4's stock-and-flow ledger.

Purpose: raise — never warn — when a tick leaves the model in a state the
         invariants forbid, including any change to an anxiety stock that no named
         source, sink or transfer accounts for.
Spec:    docs/bowen_agent_model_spec_v2.md#M6, #M6.4, #M4.G.2, #M4.G.2a, #M6.1
Tests:   tests/bowen/test_mechanisms.py::test_m4g2_invariants_asserted_every_tick; tests/bowen/test_ledger.py

What each check means from Phase C step 10:

* M6.I.1 — exactly three sinks; every allocation, and the overflow, non-negative; their
  total within the budget; and the budget changed only by records that name it — a
  completed differentiating exchange (its one sink) and ``binder_unavailable``'s return.
* M6.I.2 — the sinks are exactly the three, so distance is not one of them.
* M6.I.3 — ``life_energy_ratio`` is unset or within [0, 1] (life energy is not built in C).
* M6.I.4 — total pseudo-self is unchanged by the tick.
* M6.I.5 — ``basic_level`` is unchanged by a fast tick.
* **M6.I.6, restated at revision 12 — the ledger of `M6.4`.** Every person's change in acute
  anxiety over the tick equals the sum of the changes the tick's records log for them, every
  tie's bound anxiety likewise, and every record that logs one is of a mechanism `M6.4`
  names: a source, a sink or a transfer (``LEDGER``). Each transfer record balances: what it
  takes from acute anxiety and tie-bound anxiety sums to zero, apart from a source it logs by
  name (the outsider's positional anxiety). An unnamed or unlogged change raises.
* M6.I.7 — no person and no tie has left the field.
* M6.I.8 — the standing load ran before delivery this tick, on every tie.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord, InvariantRecord, InvariantStatus
from src.bowen.engine.objects import Sink
from src.bowen.engine.state import RunState

SOURCE, SINK, TRANSFER = "source", "sink", "transfer"
# M6.4's table, as the mechanisms that carry it. A record that changes an anxiety stock and is
# not here raises.
LEDGER = {
    "standing_load": SOURCE,        # M4.C.1c, the "too little" side
    "appraisal": SOURCE,            # M4.C.1, including the speaker's echo (M4.C.7)
    "cutoff": SOURCE,               # M4.C.1: the relief a delivered CUTOFF is appraised as
    "competing_urges": SOURCE,      # M4.D.1d
    "acute_decay": SINK,            # M1.A.8, toward the chronic floor
    "calm_contact": TRANSFER,       # M4.C.10
    "distance_binding": TRANSFER,   # M1.D.2a, person to tie
    "triangle_transfer": TRANSFER,  # M1.C.1, seeker to outsider, plus KS03.1's source
    "reconciliation": TRANSFER,     # M4.A.3, tie back to its members
    "reduce_cutoff": TRANSFER,      # M5.B.3, tie to the third party (L22.6)
    "binder_unavailable": TRANSFER, # M1.F.9, a binder's hold back to the family budget
}
# Named sources a transfer record may carry on top of its balance (logged in ``people``).
NAMED_SOURCES = {"positional_anxiety"}
BUDGET = "undifferentiation_budget"


class InvariantViolation(AssertionError):
    """M6: a violation raises."""


@dataclass(frozen=True)
class Snapshot:
    people: frozenset[PersonId]
    ties: frozenset[TieId]
    basic_levels: tuple[tuple[PersonId, float], ...]
    pseudo_self_total: float
    acute: tuple[tuple[PersonId, float], ...] = ()
    tie_bound: tuple[tuple[TieId, float], ...] = ()
    budget: float = 0.0


def snapshot(state: RunState) -> Snapshot:
    return Snapshot(
        people=frozenset(state.people),
        ties=frozenset(state.ties),
        basic_levels=tuple((p, state.people[p].basic_level) for p in sorted(state.people)),
        pseudo_self_total=sum(p.pseudo_self or 0.0 for p in state.people.values()),
        acute=tuple((p, state.people[p].acute_anxiety) for p in sorted(state.people)),
        tie_bound=tuple((t, state.ties[t].distance_bound_anxiety) for t in sorted(state.ties)),
        budget=state.family.undifferentiation_budget,
    )


def ledger_problems(state: RunState, before: Snapshot, records, tolerance: float) -> list[str]:
    """Purpose: M6.I.6 as restated — every stock change logged by a named mechanism; transfers balance.
    Spec:    docs/bowen_agent_model_spec_v2.md#M6.I.6, #M6.4, #M6.1
    Tests:   tests/bowen/test_ledger.py::test_m64_every_stock_change_is_logged_by_name
    """
    problems = []
    logged_acute: dict[PersonId, float] = {}
    logged_bound: dict[TieId, float] = {}
    for record in records:
        if not isinstance(record, EffectRecord):
            continue
        acute = sum(v for _, v in record.acute_anxiety)
        bound = sum(v for _, f, v in record.ties if f == "distance_bound_anxiety")
        budget = sum(v for name, v in record.sinks if name == BUDGET)
        named = sum(v for _, f, v in record.people if f in NAMED_SOURCES)
        if not (record.acute_anxiety or bound):
            continue
        kind = LEDGER.get(record.mechanism)
        if kind is None:
            problems.append(f"{record.mechanism} changed an anxiety stock and is not in M6.4's table")
            continue
        if kind == TRANSFER and abs(acute + bound + budget - named) > tolerance:
            problems.append(f"{record.mechanism} transfer does not balance: {acute + bound + budget - named:+.6g}")
        for pid, v in record.acute_anxiety:
            logged_acute[pid] = logged_acute.get(pid, 0.0) + v
        for tid, f, v in record.ties:
            if f == "distance_bound_anxiety":
                logged_bound[tid] = logged_bound.get(tid, 0.0) + v
    for pid, was in before.acute:
        if pid not in state.people:
            continue
        moved = state.people[pid].acute_anxiety - was
        if abs(moved - logged_acute.get(pid, 0.0)) > tolerance:
            problems.append(f"{pid}'s acute anxiety moved {moved:+.6g}, logged {logged_acute.get(pid, 0.0):+.6g}")
    for tid, was in before.tie_bound:
        if tid not in state.ties:
            continue  # a tie that left the field is M6.I.7's to report
        moved = state.ties[tid].distance_bound_anxiety - was
        if abs(moved - logged_bound.get(tid, 0.0)) > tolerance:
            problems.append(f"{tid}'s bound anxiety moved {moved:+.6g}, logged {logged_bound.get(tid, 0.0):+.6g}")
    return problems


def budget_problems(state: RunState, before: Snapshot, records, tolerance: float) -> list[str]:
    """Purpose: M6.I.1 — the budget conserved across its sinks, reduced only by named records.
    Spec:    docs/bowen_agent_model_spec_v2.md#M6.I.1, #M1.D.1, #M1.D.3
    Tests:   tests/bowen/test_sinks.py::test_m6i1_allocation_conserves_the_budget
    """
    family = state.family
    problems = []
    logged = sum(v for r in records if isinstance(r, EffectRecord) for name, v in r.sinks if name == BUDGET)
    if abs(family.undifferentiation_budget - before.budget - logged) > tolerance:
        problems.append(f"budget moved {family.undifferentiation_budget - before.budget:+.6g}, logged {logged:+.6g}")
    allocations = [*family.sink_allocations.values(), family.overflow]
    if any(v < -tolerance for v in allocations):
        problems.append(f"a negative allocation: {dict(family.sink_allocations)}, overflow {family.overflow}")
    if sum(allocations) > family.undifferentiation_budget + tolerance:
        problems.append(f"allocations {sum(allocations)} exceed the budget {family.undifferentiation_budget}")
    return problems


def assert_invariants(
    state: RunState,
    before: Snapshot,
    loaded_ties: frozenset[TieId],
    steps: tuple[str, ...],
    tolerance: float,
    records: tuple = (),
) -> InvariantRecord:
    """Purpose: assert every M6 invariant, M6.I.6 as restated, and record the results.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.G.2, #M4.G.2a, #M6.1
    Tests:   tests/bowen/test_mechanisms.py::test_m4g2_invariants_asserted_every_tick
    """
    sinks = state.family.sink_allocations

    def require(condition: bool, invariant: str, detail: str) -> tuple[str, InvariantStatus]:
        if not condition:
            raise InvariantViolation(f"tick {state.tick}: {invariant} violated — {detail}")
        return (invariant, InvariantStatus.PASSED)

    budget = budget_problems(state, before, records, tolerance)
    ledger = ledger_problems(state, before, records, tolerance)
    results = [
        require(set(sinks) == set(Sink) and not budget, "M6.I.1", "; ".join(budget) or "not three sinks"),
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
        require(not ledger, "M6.I.6", "; ".join(ledger)),
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
