"""Run a record tool's child process and collect the JSON lines it prints: the plumbing, outside the engine hash.

Purpose: a child runs in its own session, so a timeout, or a signal that stops the parent, kills its whole process
         group, its worker pool included; a child that fails or times out raises ``ChildFailed`` with the rows it
         printed first, and its error output (version 2 stage 1, S1.5a). This code was in ``tools/mutant_runner.py``
         and moved here at the stage 1 review (2026-10-19): that file is in the engine hash, and how a child is run
         does not change what it computes, so a fix here reruns no result (`CLAUDE.md`, learnings P38).
Spec:    docs/implementation_plan_version_2.md#S1.5a
Tests:   tests/bowen/test_mutant_runner.py::test_s1_5a_a_failed_child_does_not_stop_the_record,
         tests/bowen/test_mutant_runner.py::test_s1_5a_a_timed_out_child_leaves_no_process,
         tests/bowen/test_mutant_runner.py::test_s1_5a_a_stopped_parent_leaves_no_process,
         tests/bowen/test_mutant_runner.py::test_s1_5a_a_line_cut_short_is_reported,
         tests/bowen/test_mutant_runner.py::test_s1_5a_an_ignored_stop_signal_stays_ignored,
         tests/bowen/test_mutant_runner.py::test_s1_5a_a_second_stop_signal_cannot_interrupt_the_kill
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path

STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)
STDERR_TAIL = 2000  # characters of a failed child's error output kept in the failure


class ChildFailed(SystemExit):
    """The child exited non-zero, ran past its timeout, or printed a line that does not decode. ``rows`` are the JSON
    lines it printed before that, so a caller can keep them; a caller that does not catch this stops, as a
    ``SystemExit``."""

    def __init__(self, message: str, rows: list[dict]):
        super().__init__(message)
        self.rows = rows


def decode(stdout: str) -> tuple[list[dict], int]:
    """The JSON rows a child printed, and how many of its lines that start as a row do not decode (a line the kill
    cut short)."""
    rows, cut = [], 0
    for line in stdout.splitlines():
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                cut += 1
    return rows, cut


@contextmanager
def _stop_signals_raise():
    """While a child runs, SIGTERM and SIGHUP raise ``SystemExit`` in the parent, so its ``finally`` kills the child's
    group: a child in its own session gets no signal sent to the parent. A signal the parent ignores stays ignored, so
    a run under ``nohup`` outlives its terminal; the handler acts once, so a second signal cannot interrupt the kill
    (fix re-sweep, 2026-10-19). Signal handlers can only be set in the main thread; elsewhere the parent's own handling
    stands."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(signum, _frame):
        for s in previous:
            signal.signal(s, signal.SIG_IGN)
        raise SystemExit(f"stopped by signal {signal.Signals(signum).name}")

    previous = {s: signal.getsignal(s) for s in STOP_SIGNALS if signal.getsignal(s) is not signal.SIG_IGN}
    for s in previous:
        signal.signal(s, stop)
    try:
        yield
    finally:
        for s, handler in previous.items():  # None: a handler not set from Python, restored as the default
            signal.signal(s, signal.SIG_DFL if handler is None else handler)


def run_child(argv: list[str], cwd: Path, label: str, timeout: int) -> list[dict]:
    """Run ``argv`` in ``cwd`` in its own process group and return the JSON rows it prints. The group is killed
    however the call ends."""
    with _stop_signals_raise():
        child = None
        try:
            child = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     start_new_session=True)
            try:
                stdout, stderr = child.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                kill_group(child)
                stdout, stderr = child.communicate()
                rows, cut = decode(stdout)
                raise ChildFailed(f"{label} timed out after {timeout} s{_cut(cut)}:\n{stderr[-STDERR_TAIL:]}",
                                  rows) from None
        finally:
            if child is not None:
                kill_group(child)  # the child's own workers, whether it ended, failed, timed out or was stopped
    rows, cut = decode(stdout)
    if child.returncode != 0:
        raise ChildFailed(f"{label} failed (exit {child.returncode}){_cut(cut)}:\n{stderr[-STDERR_TAIL:]}", rows)
    if cut:
        raise ChildFailed(f"{label} exited 0 but printed {cut} line(s) that do not decode:\n{stderr[-STDERR_TAIL:]}",
                          rows)
    return rows


def _cut(cut: int) -> str:
    return f"; {cut} line(s) cut short were dropped" if cut else ""


def kill_group(child: subprocess.Popen) -> None:
    """Kill every process left in the child's process group; nothing to do once the group is empty."""
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
