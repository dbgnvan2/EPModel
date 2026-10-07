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
    line_offset: int = 0,
) -> TableDocument:
    """Purpose: parse one strict markdown config document.
    Spec:    docs/bowen_agent_model_spec_v2.md#M10.B.2
    Tests:   tests/bowen/test_config.py::test_m11d3_config_rejects_malformed_line
    """
    metadata: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    row_lines: list[int] = []
    state = "before_table"  # -> "expect_separator" -> "in_table" -> "after_table"

    for number, raw in enumerate(text.splitlines(), start=1 + line_offset):
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


@dataclass(frozen=True)
class SectionedDocument:
    """A config document with metadata up front and one table per ``## section``."""

    metadata: dict[str, str]
    sections: dict[str, TableDocument]


def parse_sectioned_document(
    text: str,
    *,
    sections: dict[str, tuple[str, ...]],
    metadata_keys: frozenset[str],
    source: str,
) -> SectionedDocument:
    """Purpose: parse a document whose ``## name`` sections each hold exactly one table.
    Spec:    docs/bowen_agent_model_spec_v2.md#M10.B.2
    Tests:   tests/bowen/test_family.py::test_m10b2_family_sections_parse_strictly

    The preamble before the first section may hold headings, notes and the
    declared metadata, but no table. Every declared section must appear once;
    an undeclared section raises.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("## ")]
    preamble_end = starts[0] if starts else len(lines)
    metadata: dict[str, str] = {}
    for number, raw in enumerate(lines[:preamble_end], start=1):
        stripped = raw.strip()
        where = f"{source}:{number}"
        if not stripped or stripped.startswith("#") or stripped.startswith(">"):
            continue
        if stripped.startswith("|"):
            raise ConfigError(f"{where}: a table must sit inside a ## section")
        match = _METADATA.match(stripped)
        if match is None:
            raise ConfigError(f"{where}: unrecognised line: {raw!r}")
        key, value = match.groups()
        if key not in metadata_keys:
            raise ConfigError(f"{where}: unknown metadata key {key!r}")
        if key in metadata:
            raise ConfigError(f"{where}: duplicate metadata key {key!r}")
        metadata[key] = value
    missing = metadata_keys - metadata.keys()
    if missing:
        raise ConfigError(f"{source}: missing metadata keys {sorted(missing)}")

    parsed: dict[str, TableDocument] = {}
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        name = lines[start][3:].strip()
        if name not in sections:
            raise ConfigError(f"{source}:{start + 1}: unknown section {name!r}")
        if name in parsed:
            raise ConfigError(f"{source}:{start + 1}: duplicate section {name!r}")
        body = "\n".join(lines[start + 1 : end])
        parsed[name] = parse_table_document(
            body, columns=sections[name], metadata_keys=frozenset(), source=source, line_offset=start + 1
        )
    absent = sections.keys() - parsed.keys()
    if absent:
        raise ConfigError(f"{source}: missing sections {sorted(absent)}")
    return SectionedDocument(metadata=metadata, sections=parsed)
