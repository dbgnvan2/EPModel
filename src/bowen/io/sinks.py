"""The persistence sink — writes log records outward, and changes nothing.

Purpose: write each record as one canonical JSON line to the file the caller
         names, and only publish that file if the run finished.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.B.1, #M16.B.3, #M16.T.3
Tests:   tests/bowen/test_log.py::test_m16t3_sink_does_not_change_results

M16.B.3: the sink must be a pure observer — a run with it attached and one
without reach the same final state at the same seed. It only serialises; it
never holds, alters or returns a record.

Records go to ``<name>.partial`` and the file is renamed to its real name only
when the ``with`` block exits cleanly. A run that raises leaves the ``.partial``
file behind, so a truncated log can never be mistaken for a shorter complete one.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.bowen.engine.log_records import Record, serialize


class JsonlFileSink:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + ".partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.partial.open("w", encoding="utf-8", newline="\n")

    def emit(self, record: Record) -> None:
        self._handle.write(serialize(record) + "\n")

    def close(self, *, completed: bool = True) -> None:
        self._handle.close()
        if completed:
            os.replace(self.partial, self.path)

    def __enter__(self) -> "JsonlFileSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(completed=exc_type is None)
