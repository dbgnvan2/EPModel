"""Parse the event-kind vocabulary from markdown.

Purpose: load ``config/bowen/event_kinds.md`` into the engine's ``EventKinds``.
Spec:    docs/bowen_agent_model_spec_v2.md#M10.B.1, #M10.B.2, #M5.A.1
Tests:   tests/bowen/test_events.py::test_m10b1_event_kinds_come_from_config
"""

from __future__ import annotations

import re

from src.bowen.engine.events import EventKinds, Mechanism, SourcePosition
from src.bowen.scenario.config_parse import ConfigError, parse_table_document

COLUMNS = ("kind", "mechanism", "inside_sign", "outside_sign", "contact", "impingement", "spec")
_SIGNS = {"+1": 1, "-1": -1}
_KIND = re.compile(r"^[A-Z][A-Z_-]*$")


def parse_event_kinds(text: str, *, source: str = "<event kinds>") -> EventKinds:
    document = parse_table_document(text, columns=COLUMNS, metadata_keys=frozenset(), source=source)
    by_value = {m.value: m for m in Mechanism}
    mechanisms: dict[str, Mechanism] = {}
    signs: dict[tuple[str, SourcePosition], int] = {}
    components: dict[str, tuple[float, float]] = {}
    for row, line in zip(document.rows, document.row_lines):
        where = f"{source}:{line}"
        kind = row["kind"].strip("`")
        if not _KIND.match(kind):
            raise ConfigError(f"{where}: kind {kind!r} must be upper case")
        if kind in mechanisms:
            raise ConfigError(f"{where}: duplicate kind {kind!r}")
        if row["mechanism"] not in by_value:
            raise ConfigError(f"{where}: unknown mechanism {row['mechanism']!r}")
        mechanisms[kind] = by_value[row["mechanism"]]
        for column, position in (("inside_sign", SourcePosition.INSIDE), ("outside_sign", SourcePosition.OUTSIDE)):
            if row[column] not in _SIGNS:
                raise ConfigError(f"{where}: {column} must be +1 or -1, got {row[column]!r}")
            signs[(kind, position)] = _SIGNS[row[column]]
        parts = []
        for column in ("contact", "impingement"):
            try:
                value = float(row[column])
            except ValueError:
                raise ConfigError(f"{where}: {column} must be a number, got {row[column]!r}") from None
            if not -1.0 <= value <= 1.0:
                raise ConfigError(f"{where}: {column} must be in [-1, 1], got {value}")
            parts.append(value)
        components[kind] = (parts[0], parts[1])
    if not mechanisms:
        raise ConfigError(f"{source}: no event kinds declared")
    return EventKinds(mechanisms, signs, components)
