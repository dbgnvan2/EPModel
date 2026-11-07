"""Shared setup for the LLM-agent experiment's tests.

Purpose: put the repo on the import path, and give the tests one stub-chooser run per arm, shared by the module that
         reads it, so a real-length run is paid for once.
Spec:    docs/plan_v3_llm_agent_experiment.md#10
Tests:   tests/experiments/llm_agents/test_prompts.py::test_lx4_no_banned_term_in_any_arm

No test here calls the API: every chooser is a stub or a replayed cache (§7).
"""

import functools
import hashlib
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.llm_agents.source import Answer, build, execute  # noqa: E402


class StubChooser:
    """A deterministic stand-in for the model: a hash of (salt, week, person, prompt) picks an option id.

    It also keeps every prompt it was shown, keyed by (week, person token)."""

    def __init__(self, salt: str = "stub"):
        self.salt, self.prompts = salt, {}

    def choose(self, week, person_token, prompt):
        self.prompts[(week, person_token)] = prompt
        digest = hashlib.sha256(f"{self.salt}|{week}|{person_token}|{prompt.user}".encode()).digest()
        return Answer(prompt.ids[int.from_bytes(digest[:4], "big") % len(prompt.ids)])


@functools.cache
def stub_run(arm: str, seed: int, weeks: int, salt: str = "stub"):
    """One run of ``arm`` under the stub chooser, with every prompt it rendered."""
    chooser = StubChooser(salt)
    run = execute(build(arm, seed, weeks, chooser))
    return run, chooser


class TextChooser:
    """A chooser that picks by what an option says, never where it is listed: the option whose text sorts first
    (LX.22, LX.26). Under any order it picks the same option from the same options."""

    def __init__(self):
        self.prompts = {}

    def choose(self, week, person_token, prompt):
        self.prompts[(week, person_token)] = prompt
        return Answer(min(prompt.options, key=lambda o: o[1])[0])


class FirstChooser:
    """A chooser that always answers the first option listed: position only (LX.26)."""

    def choose(self, week, person_token, prompt):
        return Answer(prompt.ids[0])


class DrawRecorder:
    """Stands in for ``state.draws``: passes every call through and keeps every draw key asked of it (LX.22)."""

    def __init__(self, inner):
        self.inner, self.keys = inner, set()

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            self.keys.update(a for a in args if type(a).__name__ == "DrawKey")
            return attr(*args, **kwargs)
        return call


@functools.cache
def captured_run(arm: str, seed: int, weeks: int, option_order: str = "shuffled"):
    """One stub-chooser run that also keeps each round's ``RoundInput``, keyed by (week, person token) (LX.23)."""
    chooser = StubChooser()
    run = build(arm, seed, weeks, chooser, option_order=option_order)
    rounds, inner = {}, run.source._round

    def _round(state, pid):
        out = inner(state, pid)
        rounds[(out[0].round - 1, out[0].person)] = out[0]
        return out
    run.source._round = _round
    return execute(run), chooser, rounds


class StubTransport:
    """A stand-in for the API: answers each request with an option id picked by a hash of the request, and counts
    the calls. ``fail`` maps a call number (1-based) to an exception to raise instead; ``fail_if(body)`` returns one
    for every request it matches (calls in a week run concurrently, so a request is matched by its content)."""

    api_version = "stub"
    sdk_version = "stub"

    def __init__(self, fail=None, usage=None, text=None, fail_if=None):
        self.calls, self.fail, self.text, self.fail_if = 0, dict(fail or {}), text, fail_if
        self.usage = usage or {"input_tokens": 2000, "output_tokens": 50}
        self.bodies = []
        self.lock = threading.Lock()

    def check_models(self, models):
        return {m: m for m in models}

    def send(self, body):
        import json

        from experiments.llm_agents.client import Response

        with self.lock:
            self.calls += 1
            self.bodies.append(body)
        if self.calls in self.fail:
            raise self.fail[self.calls]
        if self.fail_if is not None and (error := self.fail_if(body)) is not None:
            raise error
        ids = body["output_config"]["format"]["schema"]["properties"]["choice"]["enum"]
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).digest()
        text = self.text(body) if self.text else json.dumps({"choice": ids[digest[0] % len(ids)]})
        return Response(text=text, usage=dict(self.usage), stop_reason="end_turn", stop_category=None,
                        model=body["model"], request_id=f"req_{self.calls}")


def final_state(state):
    """What two runs must agree on to have reached the same final state: the invariants' snapshot, every stock of
    every person and tie, and the whole event store."""
    from src.bowen.engine.invariants import snapshot

    people = tuple((p.value, repr(sorted(vars(state.people[p]).items(), key=lambda kv: kv[0])))
                   for p in sorted(state.people))
    ties = tuple((str(t), repr(sorted(vars(state.ties[t]).items(), key=lambda kv: kv[0]))) for t in sorted(state.ties))
    return repr(snapshot(state)), people, ties, tuple(repr(e) for e in state.store.events())
