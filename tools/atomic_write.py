"""Write a generated record whole or not at all: the one write the record tools share (T-030).

Purpose: a record written in place with ``write_text`` is cut short by a kill mid-write, and the next tool that reads
         it fails. This writes a temporary file beside it and moves it into place, as ``record_cache`` does for its
         caches. Outside the engine hash: how a file is written decides nothing a run computes.
Tests:   tests/bowen/test_ensemble_record.py::test_t030_a_record_write_is_whole_or_absent
"""

from __future__ import annotations

import os
from pathlib import Path


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` by writing ``path.<pid>.partial`` and renaming it over ``path``."""
    path = Path(path)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    try:
        partial.write_text(text, encoding=encoding)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
