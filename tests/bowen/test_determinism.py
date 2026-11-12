"""Determinism and clean runs.

Purpose: two runs at one seed produce byte-identical logs, in one process and
         across processes; and a second run in a process sees nothing of the first.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.5, #M11.D.6, #M3.D.5, #M16.A.6
Tests:   this file
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from src.bowen.engine.log_records import serialize
from src.bowen.io.load import load_script
from src.bowen.run import run_phase_b

REPO = Path(__file__).resolve().parents[2]


def log_text(records) -> str:
    return "".join(serialize(r) + "\n" for r in records)


def fresh_process_log_digest(seed: int, hash_salt: str, script_without: str | None = None) -> str:
    program = (
        f"import sys, hashlib; sys.path.insert(0, {str(REPO)!r})\n"
        "from src.bowen.engine.log_records import serialize\n"
        "from src.bowen.io.load import load_script\n"
        "from src.bowen.run import run_phase_b\n"
        f"script = load_script(){'.without(%r)' % script_without if script_without else ''}\n"
        f"records = run_phase_b({seed}, script=script).records\n"
        "print(hashlib.sha256(''.join(serialize(r) + '\\n' for r in records).encode()).hexdigest())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=120,
        env=dict(os.environ, PYTHONHASHSEED=hash_salt),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def digest(records) -> str:
    return hashlib.sha256(log_text(records).encode()).hexdigest()


def test_m11d5_same_seed_same_log():
    """G6: byte-identical logs, twice in one process and in two processes with different hash salts."""
    first, second = run_phase_b(7).records, run_phase_b(7).records
    assert log_text(first) == log_text(second)
    assert fresh_process_log_digest(7, "1") == fresh_process_log_digest(7, "2") == digest(first)


def test_m11d5_a_different_seed_is_recorded_in_the_header():
    """Phase B draws nothing, so only the header's seed differs between seeds."""
    a, b = run_phase_b(7).records, run_phase_b(8).records
    assert serialize(a[0]) != serialize(b[0])
    assert log_text(a[1:]) == log_text(b[1:])


def test_m11d6_second_run_is_clean():
    """G7: run another script first; the second run's log equals the same run in a fresh process."""
    run_phase_b(7, script=load_script().without("JOB_LOSS"))
    after_other = run_phase_b(7).records
    assert digest(after_other) == fresh_process_log_digest(7, "3")


_POLICY_RUN = (
    "import sys, hashlib; sys.path.insert(0, {repo!r})\n"
    "from src.bowen.engine.log_records import serialize\n"
    "from src.bowen.engine.tick import run\n"
    "from src.bowen.io.load import CONFIG_DIR, load_constants, load_event_kinds, load_family, load_policy_rules, load_script\n"
    "from src.bowen.scenario.assemble import assemble\n"
    "from src.bowen.scenario.policy_source import PolicySource\n"
    "from src.bowen.scenario.scripted_source import ScriptedSource\n"
    "class Collect:\n"
    "    def __init__(self): self.records = []\n"
    "    def emit(self, record): self.records.append(record)\n"
    "kinds = load_event_kinds()\n"
    "script = load_script(kinds=kinds, family=load_family())\n"
    "events = ScriptedSource('events', {weeks}, script.events, ())\n"
    "parts = assemble(load_constants(), kinds, load_family(CONFIG_DIR / 'family_phase_c.md'), events, seed={seed})\n"
    "sink = Collect()\n"
    "run(parts.state, PolicySource(events, parts.params, load_policy_rules()), parts.params, parts.visibility,\n"
    "    parts.activation, sink, ticks={weeks})\n"
    "print(len(sink.records), hashlib.sha256(''.join(serialize(r) + '\\n' for r in sink.records).encode()).hexdigest())\n"
)


def test_m11d5_policy_run_same_log_across_hash_salts():
    """T-223: a policy run, which draws, gives a byte-identical log in two processes with different hash salts (the
    scripted run above draws nothing, so it cannot show this)."""
    program = _POLICY_RUN.format(repo=str(REPO), seed=5, weeks=40)
    outputs = set()
    for salt in ("1", "2"):
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=120,
                                env=dict(os.environ, PYTHONHASHSEED=salt))
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1
    count, _ = outputs.pop().split()
    assert int(count) > 40  # more than one record per week: the run acted
