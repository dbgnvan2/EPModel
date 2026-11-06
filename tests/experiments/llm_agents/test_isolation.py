"""LX.1, LX.15: the experiment stays outside the engine, and the live client is built only when asked.

Purpose: prove nothing under ``src/bowen/`` or ``config/bowen/`` names the experiment and the experiment writes only
         under its runs directory and its report (LX.1); that no key is in the repo and the live client is built only
         under ``--live`` (LX.15).
Spec:    docs/plan_v3_llm_agent_experiment.md#7, #10
Tests:   this file
"""

import re
import subprocess
import sys

import pytest
from conftest import REPO_ROOT, StubTransport
from helpers import config, phase

from experiments.llm_agents import files
from experiments.llm_agents.runner import main, run_one, run_phase

PACKAGE = REPO_ROOT / "experiments" / "llm_agents"


def test_lx1_engine_never_imports_the_experiment():
    for root in ("src/bowen", "config/bowen"):
        for path in (REPO_ROOT / root).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".md"):
                assert not re.search(r"\bexperiments\b", path.read_text(encoding="utf-8")), path


def test_lx1_writes_only_under_runs(tmp_path):
    for bad in (REPO_ROOT / "config" / "bowen" / "constants.md", REPO_ROOT / "src" / "bowen" / "x.py",
                files.RUNS / ".." / "config" / "llm.md", files.RUNS, REPO_ROOT / "docs" / "other.md"):
        with pytest.raises(files.OutsideRuns):
            files.allowed(bad)
    assert files.allowed(files.RUNS / "main-A1-s1" / "run.json")
    assert files.allowed(files.REPORT) == files.REPORT
    # Every write in the package goes through files.py.
    write = re.compile(r"\.write_(text|bytes)\(|open\([^)]*['\"][wax]b?['\"]|os\.(replace|rename|remove)\(")
    for path in PACKAGE.glob("*.py"):
        if path.name == "files.py":
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if write.search(line) and "files.write_" not in line:
                pytest.fail(f"{path.name}:{n} writes outside files.py: {line.strip()}")
    # A run writes into its runs directory and nowhere else.
    run_one(phase(2), "A1", 1, cfg=config(), live=True, transport=StubTransport(), runs=tmp_path)
    assert {p.name for p in (tmp_path / "main-A1-s1").iterdir()} == {"responses.jsonl.gz", "run.json", "summary.json",
                                                                       "prompts_audit.jsonl.gz"}


def test_lx15_no_key_in_repo():
    pattern = "sk-" + "ant-" + "[A-Za-z0-9_-]{20,}"
    found = subprocess.run(["git", "-C", str(REPO_ROOT), "grep", "-nIE", pattern], capture_output=True, text=True)
    assert found.returncode == 1, f"a key-shaped string is tracked:\n{found.stdout[:500]}{found.stderr[:200]}"
    assert "ANTHROPIC_API_KEY=" not in (REPO_ROOT / "experiments" / "llm_agents" / "config" / "llm.md").read_text()


def test_lx15_live_client_only_when_asked(tmp_path):
    # Importing the whole package imports no SDK.
    code = ("import sys; sys.path.insert(0, %r); import experiments.llm_agents.runner, experiments.llm_agents.report; "
            "print('anthropic' in sys.modules)" % str(REPO_ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"

    built = []

    def factory(cfg):
        built.append(cfg)
        return StubTransport()

    # Without --live: an LLM phase is refused, and a run replays only (here an empty cache: not replayable).
    with pytest.raises(SystemExit):
        run_phase("main", live=False, transport_factory=factory, runs=tmp_path)
    with pytest.raises(SystemExit):
        main(["run", "pilot"])
    with pytest.raises(SystemExit):
        main(["smoke"])
    out = run_one(phase(2), "A1", 1, cfg=config(), live=False, transport_factory=factory, runs=tmp_path)
    assert built == [] and out.record["status"] == "not_replayable"
    # With --live, the client is built.
    run_one(phase(2), "A1", 1, cfg=config(), live=True, transport_factory=factory, runs=tmp_path / "live")
    assert len(built) == 1
