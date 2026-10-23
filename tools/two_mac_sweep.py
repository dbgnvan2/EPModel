"""Drive Mac Mini1's share of the stage 2 sweep over SSH (`M17.E.1`; docs/two_mac_sweep.md).

Purpose: start ``tools/version_2_sweep.py --criteria ...`` on Mac Mini1 only when it is on this checkout's commit,
         clean, with the same two engine hashes and the pinned NumPy; report whether it is still running; and copy
         its cache back to a new file here for ``tools/merge_cache.py``. It never commits, pushes or merges anything.
Spec:    docs/bowen_agent_model_spec_v2.md#M17.E.1; docs/implementation_plan_version_2.md#2.10; docs/two_mac_sweep.md
Tests:   tests/test_sweep_two_macs.py (the test_m17e1_driver_* tests; ``subprocess.run`` stubbed, no SSH opened)

    python3 tools/two_mac_sweep.py check
    python3 tools/two_mac_sweep.py start --criteria KEY ... [--recompute] [--workers N]
    python3 tools/two_mac_sweep.py status
    python3 tools/two_mac_sweep.py fetch OUT
    python3 tools/two_mac_sweep.py facts          # this machine's facts as JSON; what ``check`` runs on Mac Mini1

Mac Mini1's facts (verified 2026-10-23): its checkout and its Python are below; plain ``python3`` there is Homebrew's
3.14 with no NumPy and is never used. A read-only call is retried on a dropped connection (ssh exit 255) or a timeout,
with backoff; ``start`` is not, so a start whose answer was lost is never repeated (the next ``start`` finds the run
with ``pgrep`` and refuses). The remote log, appended to, gets a ``START`` line with the arguments and an ``EXIT``
line with the run's exit status, which ``status`` and ``fetch`` print.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HOST = "davemini@mini1.local"
REMOTE_REPO = "~/ProjectsMini1/EPModel"  # in a remote shell command, where ``~`` expands
REMOTE_REPO_SCP = "ProjectsMini1/EPModel"  # the same, as scp takes it (relative to the remote home)
REMOTE_PYTHON = ".venv/bin/python"  # the uv venv: Python 3.12.3, NumPy 1.26.4
REMOTE_LOG = "~/epmodel_sweep.log"  # outside the checkout, so the run leaves it clean; appended to
CACHE_FILE = "docs/records_cache/sweep_v2.json"
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
TIMEOUT_S = 120  # one SSH or git call; the facts call hashes the engine on the remote side
RETRIES, BACKOFF_S = 3, 5  # read-only calls: attempts and the first wait, doubled each time
SAME = ("head", "engine", "phase_d_engine", "python", "machine")  # must be equal on both machines
STARTUP_WAIT_S = 15  # after a start, how long before the driver checks the run is still alive


def _run(argv: list[str], retry: bool) -> str:
    for attempt in range(RETRIES if retry else 1):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            done = None
        if done is not None and done.returncode != 255:  # 255: ssh or scp could not connect
            if done.returncode:
                raise SystemExit(f"{argv[0]} exited {done.returncode}: {done.stderr.strip()}")
            return done.stdout
        if retry and attempt + 1 < RETRIES:
            time.sleep(BACKOFF_S * 2 ** attempt)
    raise SystemExit(f"{argv[0]} {HOST}: no connection or timed out ({RETRIES if retry else 1} attempt(s))")


def ssh(command: str, retry: bool = True) -> str:
    return _run(["ssh", *SSH_OPTIONS, HOST, command], retry)


def facts() -> dict:
    """This machine's commit, whether its tracked files are clean (the sweep cache aside, which a share's run
    changes), its two engine hashes, its Python, CPU and macOS, its NumPy and the sweep machines' pin."""
    import platform

    import numpy as np

    from tools.phase_d_mutation_record import engine_hash as phase_d_engine_hash
    from tools.record_cache import engine_hash
    from tools.version_2_sweep import sweep_numpy_pin

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=TIMEOUT_S,
                              check=True).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".", f":!{CACHE_FILE}")
    return {"head": git("rev-parse", "HEAD"), "clean": not dirty, "engine": engine_hash(),
            "phase_d_engine": phase_d_engine_hash(), "python": platform.python_version(),
            "machine": platform.machine(), "macos": platform.mac_ver()[0], "numpy": np.__version__,
            "pin": sweep_numpy_pin()}


def check() -> dict:
    """Mac Mini1's facts, once they are found to match this machine's; any mismatch stops here. A different macOS is
    reported, not refused: bit-identity across the two is what the cross-check (``--recompute``) tests."""
    here = facts()
    there = json.loads(ssh(f"cd {REMOTE_REPO} && {REMOTE_PYTHON} tools/two_mac_sweep.py facts"))
    problems = [f"{k}: here {here[k]}, Mac Mini1 {there.get(k)}" for k in SAME if here[k] != there.get(k)]
    problems += [f"clean: {name} has uncommitted changes to tracked files" for name, f in (("here", here),
                 ("Mac Mini1", there)) if not f.get("clean")]
    problems += [f"numpy: {name} runs {f.get('numpy')}, the pin is {here['pin']}" for name, f in (("here", here),
                 ("Mac Mini1", there)) if f.get("numpy") != here["pin"]]
    if problems:
        raise SystemExit("Mac Mini1 does not match this checkout; nothing started:\n  " + "\n  ".join(problems))
    if here["macos"] != there.get("macos"):
        print(f"macOS differs: here {here['macos']}, Mac Mini1 {there.get('macos')}; run the cross-check "
              "(docs/two_mac_sweep.md) before trusting a merge")
    return there


def state() -> str:
    """``running``, ``stopped`` or ``error`` (pgrep itself failed). ``[v]`` keeps pgrep from matching the remote shell
    running this very command line."""
    return ssh("pgrep -f '[v]ersion_2_sweep.py' > /dev/null; case $? in 0) echo running;; 1) echo stopped;; "
               "*) echo error;; esac").strip()


def _last_lines() -> str:
    """The log's last START and EXIT lines, then its last lines: which run the tail belongs to and how it ended."""
    return ssh(f"grep -E '^(START|EXIT) ' {REMOTE_LOG} | tail -n 2; echo ---; tail -n 5 {REMOTE_LOG}")


def start(criteria: list[str], workers: int | None, recompute: bool = False) -> None:
    from tools.version_2_sweep import _checked

    _checked(criteria)  # a mistyped key stops here, not on Mac Mini1 after "started" (check: same commit and config)
    check()
    now = state()
    if now != "stopped":
        raise SystemExit(f"Mac Mini1: {'a sweep is already running' if now == 'running' else 'pgrep failed'}; "
                         "nothing started")
    args = ["--criteria", *criteria, *(["--recompute"] if recompute else []),
            *(["--workers", str(workers)] if workers else [])]
    quoted = " ".join(shlex.quote(a) for a in args)
    inner = (f'echo "START $(date) {quoted}"; caffeinate -i {REMOTE_PYTHON} tools/version_2_sweep.py {quoted}; '
             'echo "EXIT $? $(date)"')
    ssh(f"cd {REMOTE_REPO} && nohup sh -c {shlex.quote(inner)} >> {REMOTE_LOG} 2>&1 < /dev/null &", retry=False)
    time.sleep(STARTUP_WAIT_S)
    if state() != "running":
        raise SystemExit(f"Mac Mini1's run stopped within {STARTUP_WAIT_S} s:\n{_last_lines()}")
    print(f"started on Mac Mini1: {len(criteria)} criteria, running after {STARTUP_WAIT_S} s; log {REMOTE_LOG}")


def status() -> None:
    print(state())
    print(_last_lines(), end="")


def fetch(out: Path) -> None:
    if out.exists():
        raise SystemExit(f"{out} exists; give a new path, so nothing is overwritten")
    now = state()
    if now != "stopped":
        print(f"Mac Mini1: {now}; this copy holds only the results finished so far")
    _run(["scp", *SSH_OPTIONS, f"{HOST}:{REMOTE_REPO_SCP}/{CACHE_FILE}", str(out)], retry=True)
    print(_last_lines(), end="")
    print(f"copied Mac Mini1's cache to {out}; merge it with python3 tools/merge_cache.py {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("facts")
    sub.add_parser("check")
    begin = sub.add_parser("start")
    begin.add_argument("--criteria", nargs="+", required=True, metavar="KEY")
    begin.add_argument("--workers", type=int)
    begin.add_argument("--recompute", action="store_true", help="the cross-check: run the criteria again")
    sub.add_parser("status")
    copy = sub.add_parser("fetch")
    copy.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "facts":
        print(json.dumps(facts()))
    elif args.command == "check":
        print(f"Mac Mini1 matches: {json.dumps(check())}")
    elif args.command == "start":
        start(args.criteria, args.workers, args.recompute)
    elif args.command == "status":
        status()
    else:
        fetch(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
