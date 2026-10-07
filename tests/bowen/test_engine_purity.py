"""Engine purity.

Purpose: the engine writes nothing to disk (G9). The static scans for file I/O
         and forbidden imports (G4, G13) land at step 13.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.T.2, #M16.B.1, #M16.B.2
Tests:   this file
"""

from __future__ import annotations

import builtins
import io
import os

import pytest

from src.bowen.engine.log_records import CollectingEmitter
from src.bowen.engine.tick import run
from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_script
from src.bowen.scenario.assemble import assemble


def test_m16t2_engine_writes_nothing(tmp_path, monkeypatch):
    """G9: run the engine in an empty working directory with every write path trapped."""
    constants, kinds, family = load_constants(), load_event_kinds(), load_family()
    parts = assemble(constants, kinds, family, load_script(kinds=kinds, family=family), seed=7)
    monkeypatch.chdir(tmp_path)
    writes = []
    real_open, real_io_open, real_os_open = builtins.open, io.open, os.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            writes.append((file, mode))
        return real_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND):
            writes.append((path, flags))
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    run(parts.state, parts.source, parts.params, parts.visibility, parts.activation, CollectingEmitter(), ticks=40)
    assert writes == []
    assert list(tmp_path.iterdir()) == []
