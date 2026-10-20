"""A failed or timed-out child does not stop a record run, and leaves no process behind (version 2 stage 1, S1.5a).

Purpose: before stage 1, ``run_in_copy`` raised ``SystemExit`` when its child failed, so one failed mutant stopped the
         whole record, and a timed-out child's worker pool was left running (seen 2026-10-19: the workers were alive,
         orphaned, after the child was killed). These tests run stub children as real processes, in a stub repository.
Spec:    docs/implementation_plan_version_2.md#S1.5a
Tests:   this file
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

import tools.child_process as child_process
import tools.mutant_runner as runner
import tools.record_cache as record_cache

# A child that prints a row for each id in turn and exits 1 at an id starting "fail", as a child that crashes does.
FAILING_CHILD = """\
import json, sys
ids = sys.argv[sys.argv.index("--child") + 1:sys.argv.index("--workers")]
for cid in ids:
    if cid.startswith("fail"):
        sys.stderr.write("Traceback: the crash marker\\n")
        sys.exit(1)
    if cid.startswith("stray"):
        print(json.dumps({"note": cid}), flush=True)
        continue
    error = {"error": "BrokenProcessPool: a worker died"} if cid.startswith("machine") else {}
    print(json.dumps({"criterion": cid, "outcome": "PASS", "seeds": 3, "readouts": [], **error}), flush=True)
"""

# A child that starts a worker pool, writes its workers' process ids (to the file named by its argument, so the test can
# clean up whatever happens) and prints them, then waits on them past any timeout.
POOL_CHILD = """\
import json, sys, time
from concurrent.futures import ProcessPoolExecutor

def nap(seconds):
    time.sleep(seconds)

if __name__ == "__main__":
    pool = ProcessPoolExecutor(2)
    futures = [pool.submit(nap, 120) for _ in range(2)]
    while len(pool._processes) < 2:
        time.sleep(0.05)
    pids = sorted(pool._processes)
    open(sys.argv[1], "w").write(json.dumps(pids))
    print(json.dumps({"criterion": "pool", "pids": pids}), flush=True)
    for f in futures:
        f.result()
"""


def stub_repo(root: Path, **tools: str) -> Path:
    for part in ("src", "config", "tools"):
        (root / part).mkdir(parents=True, exist_ok=True)
    for name, text in tools.items():
        (root / "tools" / f"{name}.py").write_text(text, encoding="utf-8")
    return root


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists, owned by someone else: the pid was reused
        return False
    return True


def in_tmp(tmp_path, monkeypatch):
    """The copy goes under tmp_path, and its cleanup's relative unlinks (``rmtree`` with ``dir_fd``) resolve there for
    the write guard (``tests/bowen/conftest.py``)."""
    monkeypatch.chdir(tmp_path)
    return mock.patch.object(tempfile, "tempdir", str(tmp_path))


def test_s1_5a_a_failed_child_does_not_stop_the_record(tmp_path, monkeypatch, capsys):
    """The rows the child printed before it failed are kept and cached; each id after them is a FAILED row, not
    cached; a row reporting an error of the machine is FAILED too; a line that decodes but names no criterion is
    counted, and its id is FAILED, instead of stopping the record (security pass, 2026-10-19); the failure is printed
    with the child's error output; and the next item runs."""
    repo = stub_repo(tmp_path / "repo", mutant_runner=FAILING_CHILD)
    with mock.patch.object(runner, "REPO", repo), in_tmp(tmp_path, monkeypatch), \
            mock.patch.object(record_cache, "CACHE_DIR", tmp_path / "cache"), mock.patch.object(record_cache, "_failed", []):
        cache = record_cache.Cache("stub")
        first = record_cache.run_cached(cache, None, ["a", "machine", "stray", "fail", "after"], workers=1)
        assert [(r["criterion"], r["outcome"]) for r in first] == [
            ("a", "PASS"), ("machine", "FAILED"), ("stray", "FAILED"), ("fail", "FAILED"), ("after", "FAILED")]
        assert "exit 1" in first[3]["error"]
        out = capsys.readouterr().out
        assert "the crash marker" in out  # the cause reaches the log, not only "failed"
        assert "1 printed line(s) with no criterion ignored" in out
        stored = json.loads((tmp_path / "cache" / "stub.json").read_text())
        assert [r["criterion"] for r in stored.values()] == ["a"]  # only the finished, valid row
        second = record_cache.run_cached(cache, None, ["a", "b"], workers=1)  # the record goes on
        assert [(r["criterion"], r["outcome"]) for r in second] == [("a", "PASS"), ("b", "PASS")]
        assert record_cache.exit_status() == 1  # the run reports that rows failed


def test_s1_5a_a_timed_out_child_leaves_no_process(tmp_path, monkeypatch):
    """The child is killed with its whole process group: its pool's workers are gone, and the rows it printed come
    back with the failure. Run in a thread with a deadline, because a call that kills only the child waits forever on
    the pipe its orphaned workers still hold; the test kills any worker left, so a failure leaves nothing running."""
    import signal
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as Deadline

    repo = stub_repo(tmp_path / "repo", pool_child=POOL_CHILD)
    pidfile = tmp_path / "pids.json"
    with mock.patch.object(runner, "REPO", repo), in_tmp(tmp_path, monkeypatch), ThreadPoolExecutor(1) as thread:
        call = thread.submit(runner.run_in_copy, None, "pool_child.py", str(pidfile), timeout=6)
        try:
            try:
                call.result(timeout=40)
            except Deadline:
                raise AssertionError("run_in_copy did not return after the child's timeout") from None
            except child_process.ChildFailed as failed:
                (row,) = failed.rows
                assert "timed out" in str(failed.code) and len(row["pids"]) == 2
            else:
                raise AssertionError("a child past its timeout did not raise ChildFailed")
            deadline = time.monotonic() + 5
            pids = json.loads(pidfile.read_text())
            while any(alive(p) for p in pids) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not any(alive(p) for p in pids), "the timed-out child's workers are still running"
        finally:
            for pid in json.loads(pidfile.read_text()) if pidfile.exists() else ():
                if alive(pid):
                    os.kill(pid, signal.SIGKILL)


# A parent that runs POOL_CHILD through ``run_child`` and is then stopped by a signal, as a background record run is.
PARENT = """\
import sys
sys.path.insert(0, sys.argv[1])
from tools.child_process import run_child
run_child([sys.executable, sys.argv[2], sys.argv[3]], sys.argv[4], "pool", 600)
"""


def test_s1_5a_a_stopped_parent_leaves_no_process(tmp_path):
    """SIGTERM to the parent kills the child's group: a child in its own session gets no signal sent to the parent,
    so without the parent's handler its pool would run on after the parent is gone (review 2026-10-19)."""
    import signal
    import subprocess
    import sys

    (tmp_path / "pool_child.py").write_text(POOL_CHILD, encoding="utf-8")
    (tmp_path / "parent.py").write_text(PARENT, encoding="utf-8")
    pidfile = tmp_path / "pids.json"
    parent = subprocess.Popen([sys.executable, str(tmp_path / "parent.py"), str(runner.REPO),
                               str(tmp_path / "pool_child.py"), str(pidfile), str(tmp_path)],
                              stdout=(tmp_path / "out").open("w"), stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 30
        while not (pidfile.exists() and pidfile.read_text()) and time.monotonic() < deadline:
            time.sleep(0.1)
        pids = json.loads(pidfile.read_text())
        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=20)
        deadline = time.monotonic() + 5
        while any(alive(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not any(alive(p) for p in pids), "the stopped parent's child workers are still running"
    finally:
        if parent.poll() is None:
            parent.kill()
        for pid in json.loads(pidfile.read_text()) if pidfile.exists() and pidfile.read_text() else ():
            if alive(pid):
                os.kill(pid, signal.SIGKILL)


def test_s1_5a_a_line_cut_short_is_reported(tmp_path):
    """A line that does not decode is never a row: a child that exits 0 having printed one fails, keeping the rows
    before it, instead of the record stopping on a JSONDecodeError (review 2026-10-19)."""
    import sys

    script = 'print(\'{"criterion": "a"}\'); print(\'{"criterion": "b", "outc\')'
    with pytest.raises(child_process.ChildFailed, match="do not decode") as failed:
        child_process.run_child([sys.executable, "-c", script], tmp_path, "cut", 30)
    assert failed.value.rows == [{"criterion": "a"}]
    assert child_process.decode('{"a": 1}\n{"b"\nnot a row\n') == ([{"a": 1}], 1)


def test_s1_5a_an_ignored_stop_signal_stays_ignored(tmp_path):
    """A parent run under ``nohup`` ignores SIGHUP; ``run_child`` keeps it ignored, so a closed terminal does not stop
    an unattended record run. The child sends the parent SIGHUP and then prints its row (fix re-sweep, 2026-10-19)."""
    import signal
    import sys
    import threading

    assert threading.current_thread() is threading.main_thread()
    script = ("import json, os, signal; os.kill(os.getppid(), signal.SIGHUP); "
              "print(json.dumps({'criterion': 'a'}))")
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        assert child_process.run_child([sys.executable, "-c", script], tmp_path, "nohup", 30) == [{"criterion": "a"}]
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)


def test_s1_5a_a_second_stop_signal_cannot_interrupt_the_kill():
    """The first SIGTERM raises ``SystemExit``; from then until the handlers are restored the stop signals are ignored,
    so a second cannot interrupt the ``finally`` that kills the child's group (fix re-sweep, 2026-10-19)."""
    import signal
    import threading

    assert threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM)
    with child_process._stop_signals_raise():
        assert signal.getsignal(signal.SIGTERM) not in (signal.SIG_DFL, signal.SIG_IGN)  # the handler is set
        with pytest.raises(SystemExit, match="SIGTERM"):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)  # the handler runs at the next bytecode; this is never reached
        # every stop signal, not only the one that came: SIGTERM then SIGHUP, as a closing terminal sends, cannot
        # interrupt the kill either (narrow re-sweep, 2026-10-19)
        assert all(signal.getsignal(s) is signal.SIG_IGN for s in child_process.STOP_SIGNALS)
    assert signal.getsignal(signal.SIGTERM) is previous


def test_s1_5a_a_stop_signal_during_the_spawn_still_kills_the_child(tmp_path):
    """A SIGTERM that arrives as ``Popen`` returns, before ``run_child`` holds the child, raises once it does, so the
    child's group is killed. Before, the handler raised from inside ``Popen`` and the child ran on (correctness and
    security passes, 2026-10-19)."""
    import signal
    import subprocess
    import sys
    import threading

    assert threading.current_thread() is threading.main_thread()
    real, spawned = subprocess.Popen, []

    def popen(*args, **kwargs):
        child = real(*args, **kwargs)
        spawned.append(child)
        os.kill(os.getpid(), signal.SIGTERM)  # the stop arrives before the caller has the child
        return child

    try:
        with mock.patch.object(subprocess, "Popen", popen), pytest.raises(SystemExit, match="SIGTERM"):
            child_process.run_child([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, "spawn", 30)
        (child,) = spawned
        assert child.wait(timeout=10) == -signal.SIGKILL, "the child started during the stop is still running"
    finally:
        for child in spawned:
            if child.poll() is None:
                child.kill()
                child.wait()


def test_s1_5a_a_stop_signal_while_restoring_reaches_the_restored_handler():
    """A SIGHUP that arrives while the handlers are restored, after SIGTERM's is back and before SIGHUP's, goes to the
    restored SIGHUP handler. Before, the stop handler took it, set both signals ignored and raised from the ``finally``,
    leaving SIGTERM ignored (narrow re-sweep, 2026-10-19)."""
    import signal
    import threading

    assert threading.current_thread() is threading.main_thread()
    got, real = [], signal.signal

    def mine(signum, _frame):
        got.append(signum)

    def injecting(signum, handler):
        old = real(signum, handler)
        if signum == signal.SIGTERM and handler is mine:  # SIGTERM restored, SIGHUP not yet
            os.kill(os.getpid(), signal.SIGHUP)
        return old

    before = {s: signal.getsignal(s) for s in child_process.STOP_SIGNALS}
    try:
        for s in child_process.STOP_SIGNALS:
            real(s, mine)
        with mock.patch.object(signal, "signal", injecting):
            with child_process._stop_signals_raise():
                pass
        assert got == [signal.SIGHUP]
        assert all(signal.getsignal(s) is mine for s in child_process.STOP_SIGNALS)
    finally:
        for s, handler in before.items():
            real(s, signal.SIG_DFL if handler is None else handler)


def test_s1_5a_a_handler_not_set_from_python_is_restored_as_the_default():
    """``signal.getsignal`` returns ``None`` for a handler not set from Python; it is restored as ``SIG_DFL``, since
    ``signal.signal(s, None)`` raises (fix re-sweep, 2026-10-19; tested at the narrow re-sweep)."""
    import signal
    import threading

    assert threading.current_thread() is threading.main_thread()
    real = signal.getsignal
    before = real(signal.SIGHUP)
    try:
        signal.signal(signal.SIGHUP, lambda *_: None)
        with mock.patch.object(signal, "getsignal", lambda s: None if s == signal.SIGHUP else real(s)):
            with child_process._stop_signals_raise():
                pass
        assert real(signal.SIGHUP) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGHUP, signal.SIG_DFL if before is None else before)


def test_s1_5a_a_child_that_cannot_start_raises_its_own_error(tmp_path):
    """A ``Popen`` that fails (here, a missing working directory) raises its own error: no child was started, so
    there is no group to kill (fix re-sweep, 2026-10-19; tested at the narrow re-sweep)."""
    import sys

    with pytest.raises(FileNotFoundError):
        child_process.run_child([sys.executable, "-c", "pass"], tmp_path / "missing", "no start", 30)


def test_s1_5a_a_cache_write_cut_short_leaves_the_last_whole_file(tmp_path):
    """A kill while the cache is written leaves the last whole file, which the next run reads; before, the file was
    written in place and a cut one stopped the next run with a JSONDecodeError (security pass, 2026-10-19)."""
    with mock.patch.object(record_cache, "CACHE_DIR", tmp_path):
        cache = record_cache.Cache("c")
        cache.put("k", {"a": 1})
        cache.flush()
        whole = (tmp_path / "c.json").read_text(encoding="utf-8")
        real = Path.write_text

        def cut(self, text, *args, **kwargs):
            real(self, text[: len(text) // 2], *args, **kwargs)
            raise KeyboardInterrupt  # killed mid-write

        cache.put("k2", {"b": 2})
        with mock.patch.object(Path, "write_text", cut), pytest.raises(KeyboardInterrupt):
            cache.flush()
        assert (tmp_path / "c.json").read_text(encoding="utf-8") == whole
        assert record_cache.Cache("c").entries == {"k": {"a": 1}}


def test_s1_5a_a_stop_signal_while_setting_the_handlers_restores_both():
    """A SIGTERM right after its handler is set, before SIGHUP's, raises with both handlers restored. Before, the
    handlers were set outside the ``try``, so it raised with both left ignored (narrow re-sweep, 2026-10-19)."""
    import signal
    import threading

    assert threading.current_thread() is threading.main_thread()
    real = signal.signal

    def mine(_signum, _frame):
        pass

    def injecting(signum, handler):
        old = real(signum, handler)
        if signum == signal.SIGTERM and handler not in (mine, signal.SIG_IGN, signal.SIG_DFL):  # the stop handler set
            os.kill(os.getpid(), signal.SIGTERM)
        return old

    before = {s: signal.getsignal(s) for s in child_process.STOP_SIGNALS}
    try:
        for s in child_process.STOP_SIGNALS:
            real(s, mine)
        with mock.patch.object(signal, "signal", injecting), pytest.raises(SystemExit, match="SIGTERM"):
            with child_process._stop_signals_raise():
                pass
        assert all(signal.getsignal(s) is mine for s in child_process.STOP_SIGNALS)
    finally:
        for s, handler in before.items():
            real(s, signal.SIG_DFL if handler is None else handler)
