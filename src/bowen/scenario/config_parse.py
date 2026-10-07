"""Strict parsing of markdown configuration documents.

Purpose: turn a markdown config document into metadata and table rows, raising
         on anything it does not recognise.
Spec:    docs/bowen_agent_model_spec_v2.md#M10.B.2
Tests:   tests/bowen/test_config.py::test_m11d3_config_rejects_malformed_line

A config document may contain only these kinds of line:

* blank lines;
* headings (``#`` …);
* notes (``>`` …);
* ``key: value`` metadata lines, for keys the caller declares;
* exactly one markdown table, whose header must match the caller's columns.

Anything else raises ``ConfigError`` with its line number. The frozen engine's
``_apply_config`` skipped what it could not parse and fell back to defaults;
``M10.B.2`` forbids that, so there is no fallback path here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_METADATA = re.compile(r"^([a-z][a-z0-9_]*):\s*(.*?)\s*$")
_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")


class ConfigError(ValueError):
    """A config document is malformed, incomplete, or names something unknown."""


@dataclass(frozen=True)
class TableDocument:
    """A parsed config document: declared metadata and table rows keyed by column."""

    metadata: dict[str, str]
    rows: tuple[dict[str, str], ...]
    row_lines: tuple[int, ...]


def _cells(line: str) -> list[str]:
    inner = line.strip()
    if not (inner.startswith("|") and inner.endswith("|")) or len(inner) < 2:
        raise ValueError("not a table row")
    return [cell.strip() for cell in inner[1:-1].split("|")]


def parse_table_document(
    text: str,
    *,
    columns: tuple[str, ...],
    metadata_keys: frozenset[str],
    source: str,
) -> TableDocument:
    """Purpose: parse one strict markdown config document.
    Spec:    docs/bowen_agent_model_spec_v2.md#M10.B.2
    Tests:   tests/bowen/test_config.py::test_m11d3_config_rejects_malformed_line
    """
    metadata: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    row_lines: list[int] = []
    state = "before_table"  # -> "expect_separator" -> "in_table" -> "after_table"

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        where = f"{source}:{number}"
        stripped = line.strip()

        if stripped.startswith("|"):
            try:
                cells = _cells(stripped)
            except ValueError:
                raise ConfigError(f"{where}: malformed table row: {line!r}") from None
            if state == "before_table":
                if tuple(cells) != columns:
                    raise ConfigError(
                        f"{where}: table header {cells} does not match {list(columns)}"
                    )
                state = "expect_separator"
            elif state == "expect_separator":
                if len(cells) != len(columns) or not all(
                    _SEPARATOR_CELL.match(c) for c in cells
                ):
                    raise ConfigError(f"{where}: expected a table separator row")
                state = "in_table"
            elif state == "in_table":
                if len(cells) != len(columns):
                    raise ConfigError(
                        f"{where}: row has {len(cells)} cells, expected {len(columns)}"
                    )
                rows.append(dict(zip(columns, cells)))
                row_lines.append(number)
            else:
                raise ConfigError(f"{where}: a second table is not allowed")
            continue

        if state in ("expect_separator",):
            raise ConfigError(f"{where}: expected a table separator row")
        if state == "in_table":
            state = "after_table"

        if not stripped or stripped.startswith("#") or stripped.startswith(">"):
            continue

        match = _METADATA.match(stripped)
        if match is None:
            raise ConfigError(f"{where}: unrecognised line: {line!r}")
        key, value = match.groups()
        if key not in metadata_keys:
            raise ConfigError(f"{where}: unknown metadata key {key!r}")
        if key in metadata:
            raise ConfigError(f"{where}: duplicate metadata key {key!r}")
        metadata[key] = value

    if state in ("before_table", "expect_separator"):
        raise ConfigError(f"{source}: no table found")
    missing = metadata_keys - metadata.keys()
    if missing:
        raise ConfigError(f"{source}: missing metadata keys {sorted(missing)}")
    return TableDocument(metadata=metadata, rows=tuple(rows), row_lines=tuple(row_lines))
