"""Pattern readouts — the patterns Bowen named, read off a run, never selected (Phase C step 11).

Purpose: report whether a tie or a triangle shows conflict, distance, over/underfunctioning,
         fusion, projection, cutoff or a fixed triangle, each by the definition declared in
         ``docs/readouts_phase_c.md``, and which pattern a tie holds over time.
Spec:    docs/bowen_agent_model_spec_v2.md#M5.A.1a
Tests:   tests/bowen/test_patterns.py

`M5.A.1a`: each core move is one act; the pattern of the same name is a readout over a tie's
or a triangle's history of reciprocal acts, and each readout's definition is `[I]` and
declared before a criterion that reads it runs. ``read_pattern`` refuses a name that is not
declared. The parameters are in ``config/bowen/constants.md`` (``pattern_*``). A readout. The
engine never imports it (`M11.D.22`).

The definitions are the owner's (2026-10-10, ``docs/PROPOSAL — RUN VISUALIZATION MAPPING.md`` §0,
§4): an emotional pattern is a **repeating** pattern of activity, ``pattern_min_acts`` (3) or more
acts in ``pattern_window`` (4) weeks; no interaction is an emerging level of distance; cutoff is
rejection, not absence; pursuit is fusion, over/underfunctioning or projection; and a pattern ends
only when a new pattern emerges (``current_patterns``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from src.bowen.engine.contact import too_little
from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState

from src.bowen.io.load import load_pattern_acts

# Which acts each readout counts: config/bowen/pattern_acts.md (Phase D plan B2; rule 9).
ACTS = load_pattern_acts()
APPROACH = ACTS["approach"]
# The act set the child-projection sink counts (src/bowen/engine/sinks.py), from a parent to a child.
PROJECTION_ACTS = ACTS["projection"]


class UndeclaredPattern(KeyError):
    """A criterion asked for a pattern docs/readouts_phase_c.md does not declare (M5.A.1a)."""


@dataclass(frozen=True)
class ReadoutParams:
    window: int
    min_acts: int
    fixed_activations: int
    projection_hold: int = 0

    @classmethod
    def from_constants(cls, constants) -> "ReadoutParams":
        return cls(constants["pattern_window"], constants["pattern_min_acts"], constants["pattern_fixed_activations"],
                   constants["pattern_projection_hold"])


@dataclass(frozen=True)
class Reading:
    """One pattern whose start condition holds on a subject at the tick read.

    ``count`` is the acts in the window that make it (for silent distance, the weeks without
    interaction); ``first`` the tick of the first of them; ``toward`` the person it points from or
    to: the over side, the projecting parent, the one refusing, a fixed triangle's outsider.
    """

    pattern: str
    subject: TieId | TriangleId
    count: int
    first: int
    toward: PersonId | None = None
    silent: bool = False


class _At:
    """The run's state read as of week ``tick``: every attribute is the state's except ``tick``. Lets a caller read the
    patterns at a week other than ``state.tick`` without changing the state (Phase D plan B2)."""

    def __init__(self, state: RunState, tick: int) -> None:
        self._state, self.tick = state, tick

    def __getattr__(self, name):
        if name == "_state":  # not yet set (a copy or unpickle): no recursion
            raise AttributeError(name)
        return getattr(self._state, name)

    def __setattr__(self, name, value):
        if name not in ("_state", "tick"):
            raise AttributeError(f"a read-only view of the state; {name!r} not writable")
        object.__setattr__(self, name, value)


def _as_of(state: RunState, now: int | None):
    return state if now is None else _At(state, now)


def _moves_on(state: RunState, tie: TieId, window: int):
    a, b = tie.members()
    return [
        e for e in state.store.events()
        if e.mechanism is Mechanism.MOVE and e.sender in (a, b) and ({a, b} - {e.sender}) <= set(e.targets)
        and state.tick - window < e.timestamp <= state.tick
    ]


def _first(events) -> int:
    return min(e.timestamp for e in events)


def _projection_acts(moves, parent: PersonId | None):
    return [e for e in moves if parent is not None and e.sender == parent and e.kind in PROJECTION_ACTS]


def conflict(state: RunState, tie: TieId, rp: ReadoutParams) -> Reading | None:
    """At least ``min_acts`` CONFLICT acts on the tie in W, with both members contributing."""
    hits = [e for e in _moves_on(state, tie, rp.window) if e.kind in ACTS["conflict"]]
    if len(hits) < rp.min_acts or {e.sender for e in hits} != set(tie.members()):
        return None
    return Reading("conflict", tie, len(hits), _first(hits))


def distance(state: RunState, tie: TieId, params: EngineParams, rp: ReadoutParams) -> Reading | None:
    """Either at least ``min_acts`` DISTANCE acts in W, more than the approaches, with both members below
    their contact optimum; or no interaction on the tie for at least W, at a level that is the weeks
    without interaction (the owner, 2026-10-10: no interaction is an emerging level of distance)."""
    moves = _moves_on(state, tie, rp.window)
    if not moves:
        silent = silent_weeks(state, tie)
        if silent >= rp.window:
            return Reading("distance", tie, silent, state.tick - silent + 1, silent=True)
        return None
    withdrawals = [e for e in moves if e.kind in ACTS["withdrawal"]]
    approaches = sum(1 for e in moves if e.kind in APPROACH)
    relationship = state.ties[tie]
    below = all(too_little(state.people[m], relationship, params) > 0 for m in tie.members())
    if len(withdrawals) >= rp.min_acts and len(withdrawals) > approaches and below:
        return Reading("distance", tie, len(withdrawals), _first(withdrawals))
    return None


def silent_weeks(state: RunState, tie: TieId) -> int:
    """Weeks since the last act on the tie in either direction, counting from the run's start."""
    a, b = tie.members()
    last = max((e.timestamp for e in state.store.events()
                if e.mechanism is Mechanism.MOVE and e.sender in (a, b) and ({a, b} - {e.sender}) <= set(e.targets)),
               default=-1)
    return state.tick - last


def over_underfunctioning(state: RunState, tie: TieId, rp: ReadoutParams,
                          parent: PersonId | None = None) -> Reading | None:
    """At least ``min_acts`` acts in W in one direction, more than in the other. An act counts for X as the
    over side when X takes charge of Y (OVERFUNCTION), or Y gives way to X (UNDERFUNCTION) or goes along
    with X (PURSUE). A parent's acts that read as projection do not count here."""
    projected = set(map(id, _projection_acts(_moves_on(state, tie, rp.window), parent)))
    moves = [e for e in _moves_on(state, tie, rp.window) if id(e) not in projected]
    sides = {}
    for over in tie.members():
        under = tie.other(over)
        sides[over] = [e for e in moves if (e.sender == over and e.kind in ACTS["over"])
                       or (e.sender == under and e.kind in ACTS["under"])]
    over = max(tie.members(), key=lambda m: len(sides[m]))
    hits, reverse = sides[over], sides[tie.other(over)]
    if len(hits) < rp.min_acts or len(hits) <= len(reverse):
        return None
    return Reading("over_underfunctioning", tie, len(hits), _first(hits), toward=over)


def fusion(state: RunState, tie: TieId, rp: ReadoutParams, parent: PersonId | None = None) -> Reading | None:
    """At least ``min_acts`` PURSUE acts in W, both members pursuing: each going along with the other."""
    projected = set(map(id, _projection_acts(_moves_on(state, tie, rp.window), parent)))
    hits = [e for e in _moves_on(state, tie, rp.window) if e.kind in ACTS["fusion"] and id(e) not in projected]
    if len(hits) < rp.min_acts or {e.sender for e in hits} != set(tie.members()):
        return None
    return Reading("fusion", tie, len(hits), _first(hits))


def last_projection_act(state: RunState, tie: TieId, parent: PersonId | None, now: int | None = None) -> int | None:
    """The week of the parent's latest projection act toward the child, of any count; None if never. Every such act
    is evidence the projection continues (the owner, 2026-10-10). ``now`` reads as of that week."""
    state = _as_of(state, now)
    if parent is None:
        return None
    child = tie.other(parent)
    return max((e.timestamp for e in state.store.events()
                if e.mechanism is Mechanism.MOVE and e.sender == parent and child in e.targets
                and e.kind in PROJECTION_ACTS and e.timestamp <= state.tick), default=None)


def projection(state: RunState, tie: TieId, rp: ReadoutParams, parent: PersonId | None) -> Reading | None:
    """At least ``min_acts`` acts in W from a parent focused on the child: OVERFUNCTION or PURSUE toward the
    child, or a TRIANGLE that recruits the child. The act set is the child-projection sink's."""
    hits = _projection_acts(_moves_on(state, tie, rp.window), parent)
    if len(hits) < rp.min_acts:
        return None
    return Reading("projection", tie, len(hits), _first(hits), toward=parent)


def cutoff(state: RunState, tie: TieId, rp: ReadoutParams) -> Reading | None:
    """At least ``min_acts`` refusals in W by one member: a CUTOFF sent to the other in the same or the next
    week after an approach from them. Rejection, not absence (the owner, 2026-10-10)."""
    moves = _moves_on(state, tie, rp.window + 1)
    for refuser in tie.members():
        other = tie.other(refuser)
        approaches = [e.timestamp for e in moves if e.sender == other and e.kind in APPROACH]
        refusals = [e for e in moves if e.sender == refuser and e.kind in ACTS["refusal"]
                    and e.timestamp > state.tick - rp.window
                    and any(e.timestamp - 1 <= t <= e.timestamp for t in approaches)]
        if len(refusals) >= rp.min_acts:
            return Reading("cutoff", tie, len(refusals), _first(refusals), toward=refuser)
    return None


def fixed_triangle(records, triangle: TriangleId, rp: ReadoutParams) -> Reading | None:
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
                outsiders.append((r.tick, next(m for m in triangle.members if m.value not in inside)))
    last = outsiders[-rp.fixed_activations:]
    if len(last) < rp.fixed_activations or len({o for _, o in last}) != 1:
        return None
    return Reading("fixed_triangle", triangle, len(last), last[0][0], toward=last[0][1])


# The declared readouts, by the name docs/readouts_phase_c.md gives each.
PATTERNS = {
    "conflict": "tie",
    "distance": "tie",
    "over_underfunctioning": "tie",
    "fusion": "tie",
    "projection": "tie",
    "cutoff": "tie",
    "fixed_triangle": "triangle",
}


def tie_readings(state: RunState, tie: TieId, params: EngineParams, rp: ReadoutParams,
                 parent: PersonId | None = None, now: int | None = None) -> list[Reading]:
    """Purpose: every declared tie pattern whose start condition holds on ``tie`` at ``state.tick``.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.A.1a
    Tests:   tests/bowen/test_patterns.py::test_m5a1a_pursuit_reads_as_fusion_over_under_or_projection
    ``parent`` is the tie's parent when it is a parent–child tie, else None. ``now`` reads the window ending at that
    week instead of at ``state.tick``.
    """
    state = _as_of(state, now)
    over_under = over_underfunctioning(state, tie, rp, parent)
    # Lopsided pursuit is over/underfunctioning, not fusion (the owner, 2026-10-10): fusion is read only when
    # neither side's going-along outweighs the other's.
    found = (conflict(state, tie, rp), distance(state, tie, params, rp), over_under,
             None if over_under is not None else fusion(state, tie, rp, parent), projection(state, tie, rp, parent),
             cutoff(state, tie, rp))
    return [r for r in found if r is not None]


def read_pattern(name: str, state: RunState, subject, params: EngineParams, rp: ReadoutParams, records=(),
                 parent_on_tie: Mapping[TieId, PersonId] | None = None) -> bool:
    """Purpose: read one declared pattern; refuse an undeclared one.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.A.1a
    Tests:   tests/bowen/test_patterns.py::test_m5a1a_an_undeclared_pattern_is_refused
    """
    if name not in PATTERNS:
        raise UndeclaredPattern(f"pattern {name!r} is not declared in docs/readouts_phase_c.md (M5.A.1a)")
    if name == "fixed_triangle":
        return fixed_triangle(records, subject, rp) is not None
    parent = (parent_on_tie or {}).get(subject)
    return any(r.pattern == name for r in tie_readings(state, subject, params, rp, parent))


def _winner(readings: list[Reading]) -> Reading:
    """More acts first, then the earlier first act, then the name (`[I]`, the visualization plan's V2)."""
    return min(readings, key=lambda r: (-r.count, r.first, r.pattern))


def current_patterns(previous: Mapping, readings: list[Reading], now: int | None = None,
                     projection_last: Mapping | None = None, projection_hold: int = 0) -> dict:
    """Purpose: the pattern each subject holds after one read — a pattern ends only when a new one emerges.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.A.1a (owner decision D2, 2026-10-10)
    Tests:   tests/bowen/test_patterns.py::test_m5a1a_pattern_holds_until_a_new_one_emerges

    ``previous`` maps a subject to the Reading it held. A subject with no reading keeps what it held. A
    reading of the same pattern in the same direction continues the held one: its count is the new
    one and its start stays where the pattern began. A different pattern, or the same pattern
    pointing the other way, replaces it. One exception, `[I]`: silence does not replace a cutoff,
    because the silence is the cutoff continuing, not a new pattern.

    **Projection holds longer** (the owner, 2026-10-10): once read, it stays until ``projection_hold`` (12) weeks
    pass with no projection act from that parent to that child (``projection_last``, acts of any count). Until then
    no other reading replaces it. After that, the current reading replaces it, or, with none, the projection ends.
    With ``now`` absent the hold does not apply.
    """
    held = dict(previous)
    by_subject: dict = {}
    for r in readings:
        by_subject.setdefault(r.subject, []).append(r)
    if now is not None and projection_hold:
        for subject, old in list(held.items()):
            if old.pattern != "projection":
                continue
            last = (projection_last or {}).get(subject)
            if last is not None and now - last < projection_hold:
                again = [r for r in by_subject.get(subject, ()) if r.pattern == "projection" and r.toward == old.toward]
                if again:
                    by_subject[subject] = again  # read again: it continues, whatever else was read
                else:
                    by_subject.pop(subject, None)  # evidence within the hold: nothing replaces it
            elif subject not in by_subject:
                del held[subject]  # stopped: 12 weeks without a projection act, and nothing new to take its place
    for subject, found in by_subject.items():
        new = _winner(found)
        old = held.get(subject)
        if old is not None and (old.pattern, old.toward) == (new.pattern, new.toward):
            held[subject] = replace(new, first=old.first)  # the same pattern continuing: its start stays
            continue
        if old is not None and old.pattern == "cutoff" and new.silent:
            continue
        held[subject] = new
    return held
