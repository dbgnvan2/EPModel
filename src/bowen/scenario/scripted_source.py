"""ScriptedSource — Phase B's fixed script, read from markdown.

Purpose: supply each tick's scheduled events and moves from a hand-written
         script, in place of the policy Phase C will build.
Spec:    docs/bowen_agent_model_spec_v2.md#M13 (Phase B: "No policy — a fixed script drives it"), #M1.F.6, #M1.F.7, #M10.B.1, #M10.B.2
Tests:   tests/bowen/test_script.py

A script has two sections. **Events** are exogenous stressors and structural
events (TRIGGER, RECONCILIATION, INSTITUTIONALIZE, BINDER_UNAVAILABLE), each
with a start tick and a duration — spells, never per-tick probabilities
(M1.F.6). **Moves** are one person's move in one tick. Every value in a script
is invented and the script says so (``grade: [I]``).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bowen.engine.act import Selection
from src.bowen.engine.events import (
    BinderKind, BinderRef, Channel, Event, EventId, EventKinds, Mechanism, SourcePosition,
)
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.state import RunState
from src.bowen.scenario.config_parse import ConfigError, parse_sectioned_document
from src.bowen.scenario.family import FamilyInstance

EVENT_COLUMNS = ("tick", "kind", "targets", "tie", "intensity", "duration", "binder")
MOVE_COLUMNS = ("tick", "actor", "kind", "targets", "intensity", "route", "source_position")
METADATA = frozenset({"script_id", "instance_id", "ticks", "grade"})
_NONE = {"—", "-", ""}


def _ids(cell: str) -> tuple[PersonId, ...]:
    if cell in _NONE:
        return ()
    return tuple(PersonId(part.strip().strip("`")) for part in cell.split(","))


def _int(raw: str, where: str, name: str) -> int:
    if not raw.isdigit():
        raise ConfigError(f"{where}: {name} {raw!r} is not a non-negative integer")
    return int(raw)


def _float(raw: str, where: str, name: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{where}: {name} {raw!r} is not a number") from None


@dataclass(frozen=True)
class ScriptedSource:
    """Purpose: the Phase B Source — scheduled events and moves by tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M13
    Tests:   tests/bowen/test_script.py::test_m13_script_parses_and_validates
    """

    script_id: str
    ticks: int
    events: tuple[Event, ...]
    moves: tuple[tuple[int, Selection], ...]

    def scheduled(self, tick: int) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.timestamp == tick)

    def selections(self, tick: int, active: tuple[PersonId, ...], state: RunState) -> tuple[Selection, ...]:
        chosen = tuple(s for t, s in self.moves if t == tick)
        missing = [str(s.actor) for s in chosen if s.actor not in active]
        if missing:
            raise ConfigError(f"script {self.script_id}: tick {tick} moves for inactive {missing}")
        return chosen

    def without(self, kind: str) -> "ScriptedSource":
        """The same script with every event of one kind removed — the other arm of a comparison."""
        return ScriptedSource(self.script_id + f"-without-{kind}", self.ticks,
                              tuple(e for e in self.events if e.kind != kind), self.moves)


def build_script(text: str, kinds: EventKinds, family: FamilyInstance, *, source: str = "<script>") -> ScriptedSource:
    """Purpose: parse and validate a script against the event kinds and the family.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.6, #M10.B.2
    Tests:   tests/bowen/test_script.py::test_m10b2_script_parses_strictly
    """
    document = parse_sectioned_document(
        text, sections={"Events": EVENT_COLUMNS, "Moves": MOVE_COLUMNS}, metadata_keys=METADATA, source=source
    )
    meta = document.metadata
    if meta["grade"] != "[I]":
        raise ConfigError(f"{source}: a script is invented and must say grade: [I]")
    if meta["instance_id"] != family.instance_id:
        raise ConfigError(f"{source}: written for {meta['instance_id']!r}, not {family.instance_id!r}")
    ticks = _int(meta["ticks"], source, "ticks")
    positions = {p.value: p for p in SourcePosition}
    binders = {b.value: b for b in BinderKind}

    def check_people(ids, where):
        for pid in ids:
            if pid not in family.people:
                raise ConfigError(f"{where}: {pid} is not in {family.instance_id}")

    def check_tick(tick, where):
        if tick >= ticks:
            raise ConfigError(f"{where}: tick {tick} is outside the script's {ticks} ticks")

    events: list[Event] = []
    table = document.sections["Events"]
    for number, (row, line) in enumerate(zip(table.rows, table.row_lines), start=1):
        where = f"{source}:{line}"
        tick = _int(row["tick"], where, "tick")
        check_tick(tick, where)
        try:
            mechanism = kinds.mechanism_of(row["kind"])
        except ValueError as error:
            raise ConfigError(f"{where}: {error}") from None
        if mechanism is Mechanism.MOVE:
            raise ConfigError(f"{where}: {row['kind']} is a move; put it under Moves")
        targets = tuple(sorted(_ids(row["targets"])))
        check_people(targets, where)
        tie = None
        if row["tie"] not in _NONE:
            a, b = _ids(row["tie"].replace("~", ","))
            check_people((a, b), where)
            tie = TieId.of(a, b)
            if tie not in family.ties:
                raise ConfigError(f"{where}: no tie {tie} in {family.instance_id}")
        binder = None
        if row["binder"] not in _NONE:
            kind_text, _, target = row["binder"].partition(" ")
            if kind_text not in binders or not target:
                raise ConfigError(f"{where}: binder must be '<kind> <target>', kinds {sorted(binders)}")
            binder = BinderRef(binders[kind_text], target.strip("`"))
        try:
            events.append(
                Event(
                    id=EventId(tick, f"script:{number}", 0), kind=row["kind"], mechanism=mechanism,
                    sender=None, targets=targets, intensity=_float(row["intensity"], where, "intensity"),
                    timestamp=tick, duration=_int(row["duration"], where, "duration"), exogenous=True,
                    source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS, on_tie=tie, binder=binder,
                )
            )
        except ValueError as error:
            if isinstance(error, ConfigError):
                raise
            raise ConfigError(f"{where}: {error}") from None

    moves: list[tuple[int, Selection]] = []
    seen: set[tuple[int, PersonId]] = set()
    table = document.sections["Moves"]
    for row, line in zip(table.rows, table.row_lines):
        where = f"{source}:{line}"
        tick = _int(row["tick"], where, "tick")
        check_tick(tick, where)
        actor = PersonId(row["actor"].strip("`"))
        targets = _ids(row["targets"])
        route = _ids(row["route"])
        check_people((actor, *targets, *route), where)
        try:
            mechanism = kinds.mechanism_of(row["kind"])
        except ValueError as error:
            raise ConfigError(f"{where}: {error}") from None
        if mechanism is not Mechanism.MOVE:
            raise ConfigError(f"{where}: {row['kind']} is not a move; put it under Events")
        if (tick, actor) in seen:
            raise ConfigError(f"{where}: {actor} already moves in tick {tick} (one move per tick, M4.D)")
        seen.add((tick, actor))
        for target in targets:
            if TieId.of(actor, target) not in family.ties:
                raise ConfigError(f"{where}: {actor} has no tie to {target}")
        if row["source_position"] not in positions:
            raise ConfigError(f"{where}: source_position must be one of {sorted(positions)}")
        moves.append((tick, Selection(
            actor=actor, kind=row["kind"], targets=tuple(sorted(targets)),
            intensity=_float(row["intensity"], where, "intensity"), route=route,
            source_position=positions[row["source_position"]],
        )))
    return ScriptedSource(meta["script_id"], ticks, tuple(events), tuple(moves))
