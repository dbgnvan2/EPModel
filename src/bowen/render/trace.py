"""The readable trace — a template rendering of a run log.

Purpose: turn a run's records into a markdown trace a person can read end to
         end, with every event's effects beside it.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.C.1, #M16.C.2, #M16.C.4, #M16.C.5, #M11.F
Tests:   tests/bowen/test_render.py

**A rendering, not an interpretation** (M16.C.1). The text comes from fixed
templates over the records; the same records always give the same bytes. No
language model is involved, and this module imports none.

Each event line carries what M16.C.2 lists — the week, the actor, the move, the
target and witnesses, and what it did — in the shape of the worked trace in
``docs/agent_model_proposal.html`` §4.2. "What it did" is in reader units:
anxiety changes in points, tie and triangle changes in words.

The standing load and the decay toward each person's floor run every week for
everyone; the trace says so once rather than printing them line by line. A
TRIGGER is the exception, because its whole effect runs through the standing load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from src.bowen.engine.events import Event, EventId, Role
from src.bowen.engine.log_records import (
    DeliveredRecord, EffectRecord, EmittedRecord, InvariantRecord, InvariantStatus, LogHeader, Record, TickRecord,
)
from src.bowen.engine.identifiers import PersonId

FRAMING = """\
> **What this is.** Output of a simulation that runs one theory's stated mechanisms
> over an invented family. It is a consistency engine for that theory — *if Bowen's
> account is right, what follows for a family shaped like this?* — not a measurement,
> not a prediction, and not a "virtual family" in the predictive sense (spec `M11.F.10`).
> It does not concern any real family or person, and nothing here is advice (`M11.F.9`).
> Every number in it comes from invented constants (`[I]`); only differences between
> two runs that share them carry meaning (`M0.4`). This run is scripted: no one in it
> chose anything (Phase B)."""

TABLE_HEAD = "| Week | Who | Move | Toward | Witnesses | What it did |\n|---|---|---|---|---|---|"


@dataclass(frozen=True)
class _Index:
    events: dict[EventId, Event]
    deliveries: dict[EventId, list]
    effects: dict[EventId, list[EffectRecord]]


def _index(records: Iterable[Record]) -> _Index:
    events, deliveries, effects = {}, {}, {}
    for record in records:
        if isinstance(record, EmittedRecord):
            events[record.event.id] = record.event
        elif isinstance(record, DeliveredRecord):
            deliveries.setdefault(record.delivery.event_id, []).append(record.delivery)
        elif isinstance(record, EffectRecord) and record.cause is not None:
            effects.setdefault(record.cause, []).append(record)
    return _Index(events, deliveries, effects)


class _Names:
    def __init__(self, names: Mapping[PersonId, str] | None) -> None:
        self._names = dict(names or {})

    def __call__(self, pid: PersonId | None) -> str:
        if pid is None:
            return "—"
        return self._names.get(pid, pid.value)

    def tie(self, text: str) -> str:
        return "–".join(self(PersonId(p)) for p in str(text).split("~"))

    def triangle(self, text: str) -> str:
        return "–".join(self(PersonId(p)) for p in str(text).split("/"))


def _signed(value: float) -> str:
    return f"{value:+.1f}"


def _what_it_did(event: Event, index: _Index, names: _Names, view: PersonId | None) -> str:
    parts = []
    delivered = sorted(index.deliveries.get(event.id, []))
    arrival = {d.delivered_tick for d in delivered if d.role is Role.TARGET}
    if arrival and arrival != {event.timestamp}:
        parts.append(f"arrives week {', '.join(str(t) for t in sorted(arrival))}")
    roles = {d.recipient: d.role for d in delivered}
    for effect in index.effects.get(event.id, []):
        if effect.mechanism == "base_appraisal":
            changes = [
                f"{names(p)} {_signed(v)}{' (witness)' if roles.get(p) is Role.WITNESS else ''}"
                for p, v in effect.acute_anxiety
                if view is None or p == view
            ]
            if changes:
                parts.append("anxiety " + ", ".join(changes))
        elif effect.mechanism == "trigger":
            for tie, _, intensity in effect.ties:
                parts.append(
                    f"standing load on {names.tie(tie)} spikes next week (multiplier +{intensity:g}) "
                    f"for {event.duration} week{'s' if event.duration != 1 else ''}; no event crosses the tie"
                )
        elif effect.mechanism == "cutoff":
            parts += [f"{names.tie(t)} now cut off (no events; bond energy kept)" for t, _, _ in effect.ties]
        elif effect.mechanism == "reconciliation":
            parts += [f"{names.tie(t)} reconnected" for t, _, _ in effect.ties]
        elif effect.mechanism == "institutionalize":
            parts.append(f"{len(effect.ties)} ties become worry edges (no events; bond energy kept)")
        elif effect.mechanism == "binder_unavailable":
            parts += [f"{v:.1f} returned to the family budget" for _, v in effect.sinks]
    return "; ".join(parts) or "—"


def _event_row(event: Event, index: _Index, names: _Names, view: PersonId | None) -> str:
    toward = ", ".join(names(t) for t in event.targets) or (names.tie(event.on_tie) if event.on_tie else "—")
    witnesses = ", ".join(names(w) for w in event.witnesses) or "—"
    who = names(event.sender) if event.sender else "(from outside)"
    return f"| {event.timestamp} | {who} | {event.kind} | {toward} | {witnesses} | {_what_it_did(event, index, names, view)} |"


def _system_rows(effect: EffectRecord, names: _Names) -> list[str]:
    rows = []
    if effect.mechanism == "triangle_recompute":
        for tri, field, value in effect.triangles:
            if field == "active":
                rows.append(f"{names.triangle(tri)} triangle {'active' if value == 'true' else 'inactive'}")
            elif field == "inside_pair" and value != "none":
                rows.append(f"{names.triangle(tri)}: inside pair {names.tie(value)}")
    elif effect.mechanism == "consolidation":
        rows += [f"{names.tie(t)} tie now distant" for t, field, _ in effect.ties if field == "tie_state_distant"]
    elif effect.mechanism == "slow_tick":
        rows.append("slow tick (yearly) fired — nothing runs on it in Phase B")
    return [f"| {effect.tick} | — | (system) | — | — | {text} |" for text in rows]


def _concerns(event: Event, view: PersonId) -> bool:
    return view == event.sender or view in event.targets or view in event.witnesses


def render(
    records: Iterable[Record],
    names: Mapping[PersonId, str] | None = None,
    view: PersonId | None = None,
) -> str:
    """Purpose: render a run log as a markdown trace; ``view`` restricts it to one person's events.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.C.1, #M16.C.2, #M16.C.4, #M16.C.5
    Tests:   tests/bowen/test_render.py::test_m16t1_trace_renders_scripted_run
    """
    records = list(records)
    if not records or not isinstance(records[0], LogHeader):
        raise ValueError("a log opens with its header (M16.A.1); this one does not")
    header = records[0]
    label = _Names(names)
    index = _index(records)

    lines = [
        "# Run trace" + (f" — {label(view)}'s view" if view else ""),
        "",
        FRAMING,
        "",
        "| Header | |",
        "|---|---|",
        f"| seed | {header.seed} |",
        f"| config hash | `{header.config_hash}` |",
        f"| spec revision | {header.spec_revision} |",
        f"| family instance | {header.instance_id} |",
        f"| activation | {header.activation_component} v{header.activation_version} ({header.activation_regime}) |",
        f"| visibility | {header.visibility_component} v{header.visibility_version} |",
        f"| constants frozen | {header.constants_frozen_at or 'not frozen'} |",
        f"| constants changed since freeze | "
        f"{', '.join(k for k, changed in header.constant_changed_after_freeze if changed) or 'none'} |",
        "",
        "The standing load and the decay toward each person's floor run every week for everyone "
        "and are not listed line by line. Effects are shown beside the event that caused them.",
        "",
        TABLE_HEAD,
    ]
    weeks = 0
    disabled: set[str] = set()
    for record in records[1:]:
        if isinstance(record, TickRecord):
            weeks += 1
        elif isinstance(record, EmittedRecord):
            if view is None or _concerns(record.event, view):
                lines.append(_event_row(record.event, index, label, view))
        elif isinstance(record, EffectRecord) and record.cause is None and view is None:
            lines += _system_rows(record, label)
        elif isinstance(record, InvariantRecord):
            disabled |= {k for k, s in record.results if s is InvariantStatus.DISABLED}
    lines += [
        "",
        f"{weeks} weeks. The invariants were asserted at the end of every week"
        + (f"; {', '.join(sorted(disabled))} is disabled until it is restated (`M4.G.2a`)." if disabled else "."),
        "",
    ]
    return "\n".join(lines)
