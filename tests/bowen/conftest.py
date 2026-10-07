"""Shared setup for the v2 agent-model tests, and the M11.D.7 write guard.

Purpose: put the repo on the import path, and fail any test that writes outside
         pytest's temporary directory.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.7
Tests:   tests/bowen/test_test_hygiene.py::test_m11d7_no_production_paths_in_tests

The guard wraps ``open``, ``io.open``, ``os.open``, ``os.mkdir``, and the calls
that move, delete or shrink files — ``os.replace``, ``os.rename``, ``os.remove``,
``os.unlink``, ``os.rmdir``, ``os.truncate`` and ``shutil.rmtree``. Any of them
touching a path outside pytest's base temporary directory raises
``ProductionPathWrite``. Reading the committed config is allowed. The suite-wide
fingerprint in ``tests/conftest.py`` catches anything this misses.
"""

import builtins
import io
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_WRITE_MODES = set("wax+")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


class ProductionPathWrite(AssertionError):
    """M11.D.7: a test resolved a real artifact path for writing."""


def _inside(path, root: Path) -> bool:
    if isinstance(path, int):  # an already-open file descriptor
        return True
    resolved = Path(os.fsdecode(path)).expanduser().resolve()
    return resolved == root or root in resolved.parents


@pytest.fixture(autouse=True)
def m11d7_writes_stay_in_tmp(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.getbasetemp().resolve()
    real_open, real_os_open, real_mkdir = builtins.open, os.open, os.mkdir

    def check(path, what):
        if not _inside(path, root):
            raise ProductionPathWrite(f"test tried to {what} {path!r} outside {root}")

    def guarded_open(file, mode="r", *args, **kwargs):
        if _WRITE_MODES & set(mode):
            check(file, f"open ({mode})")
        return real_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & _WRITE_FLAGS:
            check(path, "os.open for writing")
        return real_os_open(path, flags, *args, **kwargs)

    def guarded_mkdir(path, *args, **kwargs):
        check(path, "mkdir")
        return real_mkdir(path, *args, **kwargs)

    def guarded(name, real, arity):
        def wrapper(*args, **kwargs):
            for path in args[:arity]:
                check(path, name)
            return real(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(os, "mkdir", guarded_mkdir)
    for name, arity in (("replace", 2), ("rename", 2), ("remove", 1), ("unlink", 1), ("rmdir", 1), ("truncate", 1)):
        monkeypatch.setattr(os, name, guarded(f"os.{name}", getattr(os, name), arity))
    monkeypatch.setattr(shutil, "rmtree", guarded("shutil.rmtree", shutil.rmtree, 1))
    yield
