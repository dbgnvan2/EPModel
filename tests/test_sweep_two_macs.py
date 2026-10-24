"""The stage 2 sweep split across two Macs (`M17.E.1`; plan §2.10, 2026-10-23).

Purpose: prove that a run filtered to some criteria checks every key first, runs only its share and writes only the
         cache, never a partial record, and loses no cached result (P8); that two caches merge into their union and
         the merge refuses a key holding two different results or files made under different NumPy versions; that
         the NumPy version is recorded in the cache outside the result keys and a run under another version is
         refused; and that the SSH driver builds its commands right and refuses to start on a mismatched Mac.
Spec:    docs/bowen_agent_model_spec_v2.md#M17.E.1; docs/implementation_plan_version_2.md#2.10; docs/two_mac_sweep.md
Tests:   this file

No test here runs the engine or opens an SSH connection: ``tools.record_cache.child_rows`` and the driver's
``subprocess.run`` are stubs, and every patched module attribute is restored by ``mock.patch``.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import tools.merge_cache as merge_cache
import tools.record_cache as record_cache
import tools.sweep_config as sweep_config
import tools.two_mac_sweep as driver
import tools.version_2_sweep as tool
from tools.record_cache import Cache

REPO = Path(__file__).resolve().parents[1]
LOCAL_SWEEP_RUNNING = merge_cache.local_sweep_running  # the real one; the autouse fixture below stubs it

# Two real directional criteria, one constant each, two samples; the stub's effect is clear, so nothing reruns.
CONFIG = """# test config

## Rules

| constant | value | grade | unit | why |
|---|---|---|---|---|
| `sweep_samples` | 2 | [I] | samples | test |
| `sweep_seed` | 2 | [I] | seed | test |
| `detection_limit_multiple` | 2.0 | [I] | half-widths | test |
| `detection_limit_seeds` | 200 | [I] | seeds | test |

## Ranges

| criterion | constant | low | high | scale | bound | grade | why it is read |
|---|---|---|---|---|---|---|---|
| `M11.C.16` | `learning_rate` | 0.1 | 0.4 | log | — | [I] | test |
| `M11.C.42` | `learning_rate` | 0.1 | 0.4 | log | — | [I] | test |
"""
PIN = "1.26.4"


def _child(mutant, tool_name, ids, _workers, _label):
    return [{"criterion": cid, "outcome": "PASS", "seeds": 100, "readouts": [
        {"readout": "effect", "direction": 1, "mean_difference": 3.0, "half_width": 0.5}]} for cid in ids]


def _patches(tmp_path, config=CONFIG, child=_child):
    path = tmp_path / "sweep.md"
    path.write_text(config, encoding="utf-8")
    record = tmp_path / "record.md"
    return [mock.patch.object(record_cache, "CACHE_DIR", tmp_path / "cache"),
            mock.patch.object(record_cache, "engine_hash", return_value="a" * 64),
            mock.patch.object(tool, "engine_hash", return_value="a" * 64),
            mock.patch.object(record_cache, "child_rows", side_effect=child),
            mock.patch.multiple(tool, load_config=lambda: sweep_config.load_config(path),
                                not_run_keys=lambda: sweep_config.not_run_keys(path)),
            mock.patch.object(tool, "RECORD", record),
            mock.patch.object(tool, "sweep_numpy_pin", return_value=np.__version__),
            mock.patch.object(record_cache, "_uncached", [])]


class _Patched:
    """Every patch of ``_patches`` entered at once; ``ran`` is the stubbed ``child_rows``."""

    def __init__(self, tmp_path, **kw):
        self.patches = _patches(tmp_path, **kw)

    def __enter__(self):
        entered = [p.__enter__() for p in self.patches]
        self.ran = entered[3]
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.__exit__(*exc)


def _criteria_run(ran) -> set[str]:
    return {c for call in ran.call_args_list for c in call.args[2]}


def _write_cache(path: Path, entries: dict, meta: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record_cache.write_cache_file(path, entries, meta or {})


# --- the criterion filter ---------------------------------------------------------------------------------------

def test_m17e1_filtered_run_runs_only_the_named_criteria(tmp_path):
    with _Patched(tmp_path) as p:
        assert tool.main(["--criteria", "M11.C.42", "--workers", "1"]) == 0
        assert _criteria_run(p.ran) == {"M11.C.42"}


def test_m17e1_filter_checks_every_key_before_anything_runs(tmp_path):
    """A premise elsewhere in the config is refused even when the filter names only a directional key, and a key the
    config does not declare is refused; in neither case does any sample run."""
    premise = CONFIG.replace("| `M11.C.16` | `learning_rate`", "| `M11.C.1` | `learning_rate`")
    with _Patched(tmp_path, config=premise) as p, pytest.raises(SystemExit, match="M11.5"):
        tool.main(["--criteria", "M11.C.42", "--workers", "1"])
    assert p.ran.call_count == 0
    with _Patched(tmp_path) as p, pytest.raises(SystemExit, match="M11.C.99"):
        tool.main(["--criteria", "M11.C.42", "M11.C.99", "--workers", "1"])
    assert p.ran.call_count == 0


def test_m17e1_filtered_run_writes_the_cache_and_never_the_record(tmp_path):
    with _Patched(tmp_path):
        tool.RECORD.write_text("the full record\n", encoding="utf-8")
        tool.main(["--criteria", "M11.C.42", "--workers", "1"])
        assert tool.RECORD.read_text(encoding="utf-8") == "the full record\n"
        cache = Cache(tool.CACHE_NAME)
        assert {r["criterion"] for r in cache.entries.values()} == {"M11.C.42"}


def test_m17e1_filtered_run_keeps_every_earlier_result(tmp_path):
    """P8: a filtered run starts from a cache holding the other share's results and a result no current key reads; its
    write keeps all of them (``Cache.save`` would keep only the keys this run read)."""
    path = tmp_path / "cache" / f"{tool.CACHE_NAME}.json"
    earlier = {"f" * 64: {"criterion": "M11.C.16", "outcome": "PASS"},
               "e" * 64: {"criterion": "M11.C.7", "outcome": "FAIL"}}
    _write_cache(path, earlier, {"numpy": np.__version__})
    with _Patched(tmp_path):
        tool.main(["--criteria", "M11.C.42", "--workers", "1"])
    entries, meta = record_cache.read_cache_file(path)
    assert earlier.items() <= entries.items() and len(entries) == 4  # the two earlier and M11.C.42's two samples
    assert meta == {"numpy": np.__version__}


def test_m17e1_render_only_runs_nothing_and_a_missing_result_writes_nothing(tmp_path):
    with _Patched(tmp_path) as p:
        tool.main(["--workers", "1"])
        full = tool.RECORD.read_text(encoding="utf-8")
        tool.RECORD.write_text("stale\n", encoding="utf-8")
        p.ran.reset_mock()
        assert tool.main(["--render-only"]) == 0
        assert p.ran.call_count == 0 and tool.RECORD.read_text(encoding="utf-8") == full
    cold = tmp_path / "cold"
    cold.mkdir()
    with _Patched(cold) as p:
        tool.RECORD.write_text("stale\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="no cached result"):
            tool.main(["--render-only"])
        assert p.ran.call_count == 0 and tool.RECORD.read_text(encoding="utf-8") == "stale\n"


def test_m17e1_filter_and_render_only_together_are_refused(tmp_path):
    with _Patched(tmp_path), pytest.raises(SystemExit):
        tool.main(["--render-only", "--criteria", "M11.C.42"])


# --- the NumPy version ------------------------------------------------------------------------------------------

def test_m17e1_a_run_records_the_numpy_version_outside_the_result_keys(tmp_path):
    with _Patched(tmp_path):
        tool.main(["--criteria", "M11.C.42", "--workers", "1"])
        cache = Cache(tool.CACHE_NAME)
    raw = json.loads(cache.path.read_text(encoding="utf-8"))
    assert raw[record_cache.META] == {"numpy": np.__version__}
    assert record_cache.META not in cache.entries and all(re.fullmatch("[0-9a-f]{64}", k) for k in cache.entries)


def test_m17e1_a_run_under_another_numpy_is_refused_before_anything_runs(tmp_path):
    """Against the pin, and against the version a cache already records."""
    with _Patched(tmp_path) as p, mock.patch.object(tool, "sweep_numpy_pin", return_value="0.0.1"), \
            pytest.raises(SystemExit, match="0.0.1"):
        tool.main(["--criteria", "M11.C.42", "--workers", "1"])
    assert p.ran.call_count == 0
    _write_cache(tmp_path / "cache" / f"{tool.CACHE_NAME}.json", {}, {"numpy": "0.0.2"})
    with _Patched(tmp_path) as p, pytest.raises(SystemExit, match="0.0.2"):
        tool.main(["--criteria", "M11.C.42", "--workers", "1"])
    assert p.ran.call_count == 0


def test_m17e1_a_cache_without_meta_is_written_without_it(tmp_path):
    """The other caches carry no version: loading and writing one leaves no ``_meta`` key in it."""
    with mock.patch.object(record_cache, "CACHE_DIR", tmp_path):
        (tmp_path / "plain.json").write_text(json.dumps({"a" * 64: {"outcome": "PASS"}}), encoding="utf-8")
        cache = Cache("plain")
        cache.flush()
        assert record_cache.META not in json.loads((tmp_path / "plain.json").read_text(encoding="utf-8"))


def test_m17e1_sweep_machines_pin_numpy_and_ci_does_not():
    """``requirements-sweep.txt`` pins the version the committed cache records; ``requirements.txt``, which CI installs
    on Python 3.11–3.13 (1.26.4 has no 3.13 wheel), leaves NumPy unpinned."""
    assert tool.sweep_numpy_pin() == PIN
    assert Cache(tool.CACHE_NAME).meta == {"numpy": PIN}
    lines = [ln.split("#")[0].strip() for ln in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    assert "numpy" in lines


# --- the merge --------------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_local_sweep():
    """The merge checks for a sweep running on this machine; the tests must not depend on one running here."""
    with mock.patch.object(merge_cache, "local_sweep_running", return_value=False) as found:
        yield found


def _two(tmp_path, ours: dict, theirs: dict, our_meta=None, their_meta=None):
    a, b = tmp_path / "ours.json", tmp_path / "theirs.json"
    _write_cache(a, ours, {"numpy": PIN} if our_meta is None else our_meta)
    _write_cache(b, theirs, {"numpy": PIN} if their_meta is None else their_meta)
    return a, b


def test_m17e1_merge_is_the_union(tmp_path):
    shared = {"criterion": "M11.C.16", "outcome": "PASS", "seeds": 100}
    a, b = _two(tmp_path, {"1" * 64: shared, "2" * 64: {"criterion": "M11.C.16", "outcome": "FAIL"}},
                {"1" * 64: dict(shared), "3" * 64: {"criterion": "M11.C.42", "outcome": "PASS"}})
    assert merge_cache.merge(a, b) == (1, 1)
    entries, meta = record_cache.read_cache_file(a)
    assert set(entries) == {"1" * 64, "2" * 64, "3" * 64} and meta == {"numpy": PIN}


def test_m17e1_merge_refuses_a_key_with_two_results_and_writes_nothing(tmp_path):
    a, b = _two(tmp_path, {"1" * 64: {"criterion": "M11.C.16", "outcome": "PASS", "mean_difference": 0.5}},
                {"1" * 64: {"criterion": "M11.C.16", "outcome": "PASS", "mean_difference": 0.5000001},
                 "3" * 64: {"criterion": "M11.C.42", "outcome": "PASS"}})
    before = a.read_bytes()
    with pytest.raises(SystemExit, match="1 key"):
        merge_cache.merge(a, b)
    assert a.read_bytes() == before


@pytest.mark.parametrize("our_meta, their_meta", [({"numpy": PIN}, {"numpy": "2.1.0"}), ({"numpy": PIN}, {}),
                                                  ({}, {"numpy": PIN})])
def test_m17e1_merge_refuses_another_or_an_unknown_numpy(tmp_path, our_meta, their_meta):
    a, b = _two(tmp_path, {"1" * 64: {"outcome": "PASS"}}, {"3" * 64: {"outcome": "PASS"}}, our_meta, their_meta)
    before = a.read_bytes()
    with pytest.raises(SystemExit, match="NumPy"):
        merge_cache.merge(a, b)
    assert a.read_bytes() == before


def test_m17e1_merged_cache_renders_the_whole_record_without_a_run(tmp_path):
    """Each Mac sweeps its share into its own cache; the union renders the record a single run renders."""
    single, mac_a, mac_b = (tmp_path / d for d in ("single", "a", "b"))
    for d in (single, mac_a, mac_b):
        d.mkdir()
    with _Patched(single):
        tool.main(["--workers", "1"])
        whole = tool.RECORD.read_text(encoding="utf-8")
    with _Patched(mac_a):
        tool.main(["--criteria", "M11.C.16", "--workers", "1"])
    with _Patched(mac_b):
        tool.main(["--criteria", "M11.C.42", "--workers", "1"])
    name = f"{tool.CACHE_NAME}.json"
    merge_cache.merge(mac_a / "cache" / name, mac_b / "cache" / name)
    with _Patched(mac_a) as p:
        tool.main(["--render-only"])
        assert p.ran.call_count == 0 and tool.RECORD.read_text(encoding="utf-8") == whole


def test_m17e1_merge_refuses_while_a_sweep_runs_here(tmp_path, _no_local_sweep):
    """A running sweep rewrites the whole cache file it loaded after each run, so a merge made meanwhile would be lost."""
    a, b = _two(tmp_path, {"1" * 64: {"outcome": "PASS"}}, {"3" * 64: {"outcome": "PASS"}})
    before = a.read_bytes()
    _no_local_sweep.return_value = True
    with pytest.raises(SystemExit, match="running on this machine"):
        merge_cache.merge(a, b)
    assert a.read_bytes() == before


def test_m17e1_local_sweep_check_fails_closed():
    """pgrep's exit 1 is "none running"; anything else, an error included, counts as running."""
    for code, expected in ((0, True), (1, False), (2, True)):
        with mock.patch.object(merge_cache.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], code, b"", b"")):
            assert LOCAL_SWEEP_RUNNING() is expected


# --- the cross-check and the NumPy refusals ------------------------------------------------------------------------

def test_m17e1_recompute_runs_the_named_criteria_again(tmp_path, capsys):
    with _Patched(tmp_path) as p:
        tool.main(["--workers", "1"])
        p.ran.reset_mock()
        capsys.readouterr()
        tool.main(["--criteria", "M11.C.42", "--recompute", "--workers", "1"])
        assert "dropped 2 cached result(s) of M11.C.42" in capsys.readouterr().out
        assert _criteria_run(p.ran) == {"M11.C.42"} and p.ran.call_count == 2  # both samples, M11.C.16 untouched
        with pytest.raises(SystemExit):
            tool.main(["--recompute", "--workers", "1"])


def test_m17e1_recompute_keeps_a_result_whose_rerun_failed(tmp_path):
    def failing(mutant, tool_name, ids, _workers, _label):
        return [{"criterion": cid, "outcome": "FAILED", "seeds": 0, "error": "ChildFailed: test"} for cid in ids]

    with _Patched(tmp_path):
        tool.main(["--workers", "1"])
        before = Cache(tool.CACHE_NAME).entries
    with _Patched(tmp_path, child=failing):
        tool.main(["--criteria", "M11.C.42", "--recompute", "--workers", "1"])
        assert Cache(tool.CACHE_NAME).entries == before


def test_m17e1_recompute_interrupted_keeps_the_dropped_results_on_disk(tmp_path):
    """Milestone review MR-25: `--recompute` drops the share's results and the cache is flushed after every sample, so
    an interrupt after the first sample left the file without the others. The put-back and flush run in a `finally`.
    Mutation: the put-back only on a normal finish (the second sample's earlier result is lost)."""
    calls = []

    def interrupted(mutant, tool_name, ids, workers, label):
        calls.append(mutant.id)
        if len(calls) > 1:
            raise KeyboardInterrupt
        return _child(mutant, tool_name, ids, workers, label)

    with _Patched(tmp_path):
        tool.main(["--workers", "1"])
        before = Cache(tool.CACHE_NAME).entries
    with _Patched(tmp_path, child=interrupted), pytest.raises(KeyboardInterrupt):
        tool.main(["--criteria", "M11.C.42", "--recompute", "--workers", "1"])
    assert len(calls) == 2, "the interrupt did not fall after the first sample; the test reads nothing"
    with _Patched(tmp_path):
        assert Cache(tool.CACHE_NAME).entries == before


def test_m17e1_render_only_does_not_check_numpy(tmp_path):
    """CI renders the committed record under NumPy 2.x; only a run is held to the pin."""
    with _Patched(tmp_path):
        tool.main(["--workers", "1"])
    with _Patched(tmp_path), mock.patch.object(tool, "sweep_numpy_pin", return_value="0.0.1"):
        assert tool.main(["--render-only"]) == 0


def test_m17e1_a_cache_with_results_and_no_version_is_refused(tmp_path):
    _write_cache(tmp_path / "cache" / f"{tool.CACHE_NAME}.json", {"1" * 64: {"criterion": "M11.C.42"}}, {})
    with _Patched(tmp_path) as p, pytest.raises(SystemExit, match="records no NumPy version"):
        tool.main(["--criteria", "M11.C.42", "--workers", "1"])
    assert p.ran.call_count == 0


# --- the SSH driver ---------------------------------------------------------------------------------------------

def _facts(**kw) -> dict:
    return {"head": "c" * 40, "clean": True, "engine": "a" * 64, "phase_d_engine": "d" * 64, "python": "3.12.3",
            "machine": "arm64", "macos": "15.7.9", "numpy": PIN, "pin": PIN, **kw}


class _Remote:
    """A stub of ``subprocess.run``: answers the remote facts and ``pgrep`` (one state per call, the last repeated)
    and records every command."""

    def __init__(self, facts: dict, states=("stopped", "running")):
        self.facts, self.states, self.commands = facts, list(states), []

    def __call__(self, argv, **kw):
        self.commands.append(argv)
        assert kw.get("timeout"), "every call carries a timeout (P5)"
        command = argv[-1]
        if "two_mac_sweep.py facts" in command:
            out = json.dumps(self.facts)
        elif "pgrep" in command:
            out = (self.states.pop(0) if len(self.states) > 1 else self.states[0]) + "\n"
        else:
            out = ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    def launches(self) -> list[list[str]]:
        return [c for c in self.commands if "nohup" in c[-1]]


def _drive(remote, *argv, here=None):
    with mock.patch.object(driver, "facts", return_value=here or _facts()), \
            mock.patch.object(driver.subprocess, "run", remote), mock.patch.object(driver.time, "sleep"):
        driver.main(list(argv))


def test_m17e1_driver_starts_the_share_with_the_venv_python_detached():
    remote = _Remote(_facts())
    _drive(remote, "start", "--criteria", "M11.C.2", "M11.C.7[joint over sided]", "--workers", "10")
    [start] = remote.launches()
    assert start[:-2] == ["ssh", *driver.SSH_OPTIONS] and start[-2] == driver.HOST
    words = shlex.split(start[-1])
    assert words[:6] == ["cd", driver.REMOTE_REPO, "&&", "nohup", "sh", "-c"]
    assert words[7:] == [">>", driver.REMOTE_LOG, "2>&1", "<", "/dev/null", "&"]
    inner = words[6]
    assert (f"caffeinate -i {driver.REMOTE_PYTHON} tools/version_2_sweep.py --criteria M11.C.2 "
            "'M11.C.7[joint over sided]' --workers 10;") in inner
    assert inner.startswith('echo "START $(date) ') and inner.endswith('echo "EXIT $? $(date)"')
    assert "python3 " not in start[-1]


def test_m17e1_driver_start_command_runs_under_sh(tmp_path):
    """The nested quoting survives a real shell: the inner command, with the Python replaced by ``echo``, prints the
    arguments as one word each and the EXIT line."""
    remote = _Remote(_facts())
    _drive(remote, "start", "--criteria", "M11.C.7[joint over sided]", "--recompute")
    inner = shlex.split(remote.launches()[0][-1])[6].replace(f"caffeinate -i {driver.REMOTE_PYTHON}", "printf '<%s>'")
    out = subprocess.run(["sh", "-c", inner], capture_output=True, text=True, timeout=30, cwd=tmp_path).stdout
    assert "<tools/version_2_sweep.py><--criteria><M11.C.7[joint over sided]><--recompute>" in out
    assert re.search(r"EXIT 0 \w", out)


@pytest.mark.parametrize("mismatch", [{"head": "b" * 40}, {"clean": False}, {"engine": "b" * 64},
                                      {"phase_d_engine": "b" * 64}, {"python": "3.12.4"}, {"machine": "x86_64"},
                                      {"numpy": "2.1.0"}])
def test_m17e1_driver_refuses_to_start_on_a_mismatched_mac(mismatch):
    remote = _Remote(_facts(**mismatch))
    with pytest.raises(SystemExit, match=next(iter(mismatch))):
        _drive(remote, "start", "--criteria", "M11.C.2")
    assert not remote.launches()


def test_m17e1_driver_reports_a_different_macos_and_goes_on(capsys):
    remote = _Remote(_facts(macos="15.7.8"))
    _drive(remote, "start", "--criteria", "M11.C.2")
    assert remote.launches() and "macOS differs" in capsys.readouterr().out


def test_m17e1_driver_refuses_an_undeclared_key_before_any_ssh():
    remote = _Remote(_facts())
    with pytest.raises(SystemExit, match="M11.C.99"):
        _drive(remote, "start", "--criteria", "M11.C.99")
    assert remote.commands == []


@pytest.mark.parametrize("code, expected", [(0, "running"), (1, "stopped"), (2, "error"), (3, "error")])
def test_m17e1_driver_state_maps_pgreps_exit_under_sh(tmp_path, code, expected):
    """The remote ``state`` command, run by a real shell with a ``pgrep`` that exits ``code``: an error is never
    read as stopped, so ``start`` fails closed."""
    sent = []

    def capture(argv, **kw):
        sent.append(argv[-1])
        return subprocess.CompletedProcess(argv, 0, "running\n", "")

    with mock.patch.object(driver.subprocess, "run", capture):
        driver.state()
    fake = tmp_path / "pgrep"
    fake.write_text(f"#!/bin/sh\nexit {code}\n", encoding="utf-8")
    fake.chmod(0o755)
    out = subprocess.run(["sh", "-c", sent[0]], capture_output=True, text=True, timeout=30,
                         env={"PATH": f"{tmp_path}:/usr/bin:/bin"}).stdout
    assert out.strip() == expected


@pytest.mark.parametrize("now", ["running", "error"])
def test_m17e1_driver_refuses_to_start_unless_stopped(now):
    remote = _Remote(_facts(), states=(now,))
    with pytest.raises(SystemExit, match="nothing started"):
        _drive(remote, "start", "--criteria", "M11.C.2")
    assert not remote.launches()


def test_m17e1_driver_reports_a_run_that_stopped_at_once():
    remote = _Remote(_facts(), states=("stopped", "stopped"))
    with pytest.raises(SystemExit, match="stopped within"):
        _drive(remote, "start", "--criteria", "M11.C.2")
    assert len(remote.launches()) == 1


def test_m17e1_driver_fetches_to_a_new_path_only(tmp_path):
    remote = _Remote(_facts(), states=("stopped",))
    out = tmp_path / "mini1.json"
    _drive(remote, "fetch", str(out))
    assert [c for c in remote.commands if c[0] == "scp"] == [
        ["scp", *driver.SSH_OPTIONS, f"{driver.HOST}:{driver.REMOTE_REPO_SCP}/docs/records_cache/sweep_v2.json", str(out)]]
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="exists"):
        _drive(remote, "fetch", str(out))


def test_m17e1_driver_retries_a_read_but_never_a_start():
    """A dropped connection (255) or a timeout is retried, with backoff, on a read; the start's launch is tried once."""
    calls = []

    def dropped(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 255, "", "connection refused")

    with mock.patch.object(driver.subprocess, "run", dropped), mock.patch.object(driver.time, "sleep") as slept, \
            pytest.raises(SystemExit, match="no connection"):
        driver.ssh("true")
    assert len(calls) == driver.RETRIES and [c.args[0] for c in slept.call_args_list] == [
        driver.BACKOFF_S * 2 ** i for i in range(driver.RETRIES - 1)]

    def timed_out(argv, **kw):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    calls.clear()
    with mock.patch.object(driver.subprocess, "run", timed_out), mock.patch.object(driver.time, "sleep"), \
            pytest.raises(SystemExit, match="no connection"):
        driver.ssh("true", retry=False)
    assert len(calls) == 1


def test_m17e1_driver_facts_are_this_machines():
    f = driver.facts()
    assert f["engine"] == record_cache.engine_hash() and f["numpy"] == np.__version__ and f["pin"] == PIN
    assert re.fullmatch("[0-9a-f]{40}", f["head"]) and isinstance(f["clean"], bool) and f["python"] and f["machine"]
