"""Run-log record types, their canonical serialisation, and the emitter interface.

Purpose: define what a run records (M16.A) and how records leave the engine
         without the engine writing anything (M16.B.1).
Spec:    docs/bowen_agent_model_spec_v2.md#M16.A, #M16.B.1, #M3.D.5
Tests:   tests/bowen/test_log_records.py

Every record serialises to one line of canonical JSON — sorted keys, no
whitespace, no NaN — so two runs at one seed produce byte-identical logs
(M3.D.5, M16.A.6). Serialisation produces a string; writing it anywhere is the
caller's business.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
from dataclasses import dataclass
from typing import Any, Protocol

from src.bowen.engine.events import Delivery, Event, EventId
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId


class DecidedBy(enum.Enum):
    """M16.A.3c — what chose the outcome."""

    SCRIPTED = "scripted"
    POLICY = "policy"
    TIE_BREAK = "tie_break"
    FALLBACK = "fallback"


class InvariantStatus(enum.Enum):
    PASSED = "passed"
    DISABLED = "disabled"   # M4.G.2a — M6.I.6 until it is restated


@dataclass(frozen=True)
class LogHeader:
    """M16.A.1, M16.A.1a, M16.A.7 — enough to replay and attribute the run."""

    seed: int
    config_hash: str
    spec_revision: str
    instance_id: str
    activation_component: str
    activation_version: str
    visibility_component: str
    visibility_version: str
    activation_regime: str
    constants_frozen_at: dt.date | None
    constant_changed_after_freeze: tuple[tuple[str, bool], ...]

    record_type = "header"


@dataclass(frozen=True)
class TickRecord:
    tick: int
    record_type = "tick"


@dataclass(frozen=True)
class EmittedRecord:
    """M16.A.2 — the full M1.F.1 field set at emission."""

    event: Event
    record_type = "emitted"


@dataclass(frozen=True)
class DeliveredRecord:
    """M16.A.2 — emitted and delivered times and the latency between them."""

    delivery: Delivery
    record_type = "delivered"


@dataclass(frozen=True)
class SelectionRecord:
    """M16.A.3, .3a, .3b, .3c — the rationale, not only the outcome.

    In Phase B every selection is ``SCRIPTED`` with empty propensities, no
    draw, no beliefs and no legal set; the fields exist so Phase C fills them
    without a format change.
    """

    tick: int
    actor: PersonId
    decided_by: DecidedBy
    event_id: EventId | None
    propensities: tuple[tuple[str, float], ...] = ()
    draw: float | None = None
    beliefs_used: tuple[tuple[str, float], ...] = ()
    legal_set: tuple[str, ...] = ()
    record_type = "selection"


@dataclass(frozen=True)
class EffectRecord:
    """M16.A.4 — what a cause did, recorded beside it.

    ``cause`` is the delivered event, or ``None`` for a per-tick mechanism such
    as the standing load, which ``mechanism`` then names.
    """

    tick: int
    mechanism: str
    cause: EventId | None
    acute_anxiety: tuple[tuple[PersonId, float], ...] = ()
    ties: tuple[tuple[TieId, str, float], ...] = ()
    triangles: tuple[tuple[TriangleId, str, str], ...] = ()
    sinks: tuple[tuple[str, float], ...] = ()
    # Per-person state other than acute anxiety: (person, field, change or value).
    people: tuple[tuple[PersonId, str, float], ...] = ()
    record_type = "effect"


@dataclass(frozen=True)
class InvariantRecord:
    """M16.A.4 — the tick's invariant results, beside the tick's effects."""

    tick: int
    results: tuple[tuple[str, InvariantStatus], ...]
    record_type = "invariants"


@dataclass(frozen=True)
class BeliefWriteRecord:
    """M16.A.5, M16.A.5a — tagged apart from ground truth; discrepancy computable."""

    tick: int
    holder: PersonId
    subject: str
    value: float
    true_value: float | None
    record_type = "belief_write"

    def discrepancy(self) -> float | None:
        return None if self.true_value is None else self.value - self.true_value


Record = (
    LogHeader
    | TickRecord
    | EmittedRecord
    | DeliveredRecord
    | SelectionRecord
    | EffectRecord
    | InvariantRecord
    | BeliefWriteRecord
)


class Emitter(Protocol):
    """M16.B.1 — the caller supplies this; the engine only calls ``emit``."""

    def emit(self, record: Record) -> None: ...


class Tee:
    """Send each record to several emitters, in order. Writes nothing itself."""

    def __init__(self, *emitters: Emitter) -> None:
        self._emitters = emitters

    def emit(self, record: Record) -> None:
        for emitter in self._emitters:
            emitter.emit(record)


class CollectingEmitter:
    """Keeps records in memory. Tests and the renderer use it; it writes nothing."""

    def __init__(self) -> None:
        self.records: list[Record] = []

    def emit(self, record: Record) -> None:
        self.records.append(record)


def plain(value: Any) -> Any:
    """A JSON-ready form of any record value: identifiers as text, enums as values."""
    if isinstance(value, PersonId):
        return value.value
    if isinstance(value, (TieId, TriangleId, EventId)):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dt.date):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        items = [(json.dumps(plain(k), sort_keys=True) if not isinstance(plain(k), str) else plain(k), plain(v))
                 for k, v in value.items()]
        return dict(sorted(items))
    if isinstance(value, (set, frozenset)):
        return sorted((plain(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, float) and value != value:
        raise ValueError("NaN cannot be logged")
    return value


def to_dict(record: Record) -> dict[str, Any]:
    body = plain(record)
    body["record_type"] = record.record_type
    return body


def serialize(record: Record) -> str:
    """Purpose: one canonical JSON line per record.
    Spec:    docs/bowen_agent_model_spec_v2.md#M3.D.5, #M16.A.6
    Tests:   tests/bowen/test_log_records.py::test_m16a6_serialisation_is_canonical
    """
    return json.dumps(to_dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
