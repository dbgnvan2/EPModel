"""Pattern readouts — the patterns Bowen named, read off a run, never selected (Phase C step 11).

Purpose: report whether a tie or a triangle shows conflict, over/underfunctioning,
         distance, cutoff or a fixed triangle, each by the definition declared in
         ``docs/readouts_phase_c.md`` before any criterion reads it.
Spec:    docs/bowen_agent_model_spec_v2.md#M5.A.1a
Tests:   tests/bowen/test_patterns.py

`M5.A.1a`: each core move is one act; the pattern of the same name is a readout over a tie's
or a triangle's history of reciprocal acts, and each readout's definition is `[I]` and
declared before a criterion that reads it runs. ``read_pattern`` refuses a name that is not
declared. The parameters are in ``config/bowen/constants.md`` (``pattern_*``). Projection is
Phase D. A readout. The engine never imports it (`M11.D.22`).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bowen.engine.contact import too_little
from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import TieId, TriangleId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.moves import DEFAULT_AREA
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState

APPROACH = frozenset({"PURSUE", "STAY-IN-CONTACT", "REDUCE_CUTOFF"})


class UndeclaredPattern(KeyError):
    """A criterion asked for a pattern docs/readouts_phase_c.md does not declare (M5.A.1a)."""


@dataclass(frozen=True)
class ReadoutParams:
    window: int
    min_acts: int
    pole: float
    fixed_activations: int

    @classmethod
    def from_constants(cls, constants) -> "ReadoutParams":
        return cls(constants["pattern_window"], constants["pattern_min_acts"], constants["pattern_pole"],
                   constants["pattern_fixed_activations"])


def _moves_on(state: RunState, tie: TieId, window: int):
    a, b = tie.members()
    return [
        e for e in state.store.events()
        if e.mechanism is Mechanism.MOVE and e.sender in (a, b) and ({a, b} - {e.sender}) <= set(e.targets)
        and state.tick - window < e.timestamp <= state.tick
    ]


def conflict(state: RunState, tie: TieId, rp: ReadoutParams) -> bool:
    """Both members emit CONFLICT toward each other, each at least ``min_acts`` times in W."""
    moves = _moves_on(state, tie, rp.window)
    return all(sum(1 for e in moves if e.sender == m and e.kind == "CONFLICT") >= rp.min_acts for m in tie.members())


def over_underfunctioning(state: RunState, tie: TieId, rp: ReadoutParams) -> bool:
    """The hardened habit is at a pole, and in W the over side took charge or the under side gave way."""
    relationship = state.ties[tie]
    habit = relationship.functioning_habit.get(DEFAULT_AREA, 0.0)
    if abs(habit) < rp.pole:
        return False
    over = tie.a if habit > 0 else tie.b
    under = tie.b if habit > 0 else tie.a
    moves = _moves_on(state, tie, rp.window)
    return any((e.sender == over and e.kind == "OVERFUNCTION") or (e.sender == under and e.kind == "UNDERFUNCTION")
               for e in moves)


def distance(state: RunState, tie: TieId, params: EngineParams, rp: ReadoutParams) -> bool:
    """DISTANCE acts outnumber approaches in W, and both members are below their optimum on the tie."""
    moves = _moves_on(state, tie, rp.window)
    withdrawals = sum(1 for e in moves if e.kind == "DISTANCE")
    approaches = sum(1 for e in moves if e.kind in APPROACH)
    relationship = state.ties[tie]
    below = all(too_little(state.people[m], relationship, params) > 0 for m in tie.members())
    return withdrawals > approaches and below


def cutoff(state: RunState, tie: TieId, rp: ReadoutParams) -> bool:
    """The tie is non-interactive, and has been for at least W (no CUTOFF on it inside W)."""
    if state.ties[tie].interactive:
        return False
    recent_cut = [e for e in _moves_on(state, tie, rp.window) if e.kind == "CUTOFF"]
    return not recent_cut


def fixed_triangle(records, triangle: TriangleId, rp: ReadoutParams) -> bool:
    """The same outsider across the last ``fixed_activations`` activations of the triangle."""
    outsiders, active = [], False
    for r in records:
        if not isinstance(r, EffectRecord) or r.mechanism != "triangle_recompute":
            continue
        for tri, field, value in r.triangles:
            if tri != triangle:
                continue
            if field == "active":
                active = value == "true"
            elif field == "inside_pair" and value != "none" and active:
                inside = set(value.split("~"))
                outsiders.append(next(m.value for m in triangle.members if m.value not in inside))
    last = outsiders[-rp.fixed_activations:]
    return len(last) == rp.fixed_activations and len(set(last)) == 1


# The declared readouts, by the name docs/readouts_phase_c.md gives each.
PATTERNS = {
    "conflict": "tie",
    "over_underfunctioning": "tie",
    "distance": "tie",
    "cutoff": "tie",
    "fixed_triangle": "triangle",
}


def read_pattern(name: str, state: RunState, subject, params: EngineParams, rp: ReadoutParams, records=()) -> bool:
    """Purpose: read one declared pattern; refuse an undeclared one.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.A.1a
    Tests:   tests/bowen/test_patterns.py::test_m5a1a_an_undeclared_pattern_is_refused
    """
    if name not in PATTERNS:
        raise UndeclaredPattern(f"pattern {name!r} is not declared in docs/readouts_phase_c.md (M5.A.1a)")
    if name == "conflict":
        return conflict(state, subject, rp)
    if name == "over_underfunctioning":
        return over_underfunctioning(state, subject, rp)
    if name == "distance":
        return distance(state, subject, params, rp)
    if name == "cutoff":
        return cutoff(state, subject, rp)
    return fixed_triangle(records, subject, rp)
