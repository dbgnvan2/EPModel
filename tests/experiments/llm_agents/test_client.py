"""LX.12, LX.13, LX.14: retries, permanent failures and the failed-parse path.

Purpose: prove transient failures are retried with logged backoff and a run out of retries is incomplete and resumes
         from the cache (LX.12), permanent ones are not retried and stop the run (LX.13), and a refusal, max_tokens or
         unreadable answer gets one re-ask and then a flagged hold, with an over-share run invalid (LX.14). The SDK's
         errors are mapped through a stand-in ``anthropic`` module; the real API is never called.
Spec:    docs/plan_v3_llm_agent_experiment.md#7, #10
Tests:   this file
"""

import json
import logging
import random
import sys
import types

import pytest
from conftest import StubTransport
from helpers import config, phase

from experiments.llm_agents.cache import ResponseCache
from experiments.llm_agents.client import (
    AnthropicTransport, CachedChooser, PermanentError, Response, TransientError, call_with_retries, send_with_retries,
)
from experiments.llm_agents.config import load_text
from experiments.llm_agents.prompts import Prompt
from experiments.llm_agents.runner import run_one

FAKE_KEY = "sk-" + "ant-api03-" + "SECRETSECRETSECRETSECRET"  # built, so no key-shaped literal is tracked
NO_SLEEP = lambda seconds: None  # noqa: E731
PROMPT = Prompt("system", "<round>r</round>", (("1", "a"), ("2", "b")))


def _chooser(tmp_path, transport, sleeps=None, run_id="t"):
    cfg = config()
    return CachedChooser(run_id=run_id, model=cfg.get("model_main"), arm="A1", seed=1, cfg=cfg,
                         cache=ResponseCache(tmp_path / "c.jsonl.gz", runs=tmp_path), reask_text=load_text()["reask"],
                         transport=transport, sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_lx12_transient_is_retried(tmp_path, caplog):
    sleeps = []
    t = StubTransport(fail={1: TransientError("429", retry_after=7.0), 2: TransientError("529"),
                            3: TransientError("timeout")})
    with caplog.at_level(logging.WARNING, logger="experiments.llm_agents.client"):
        c = _chooser(tmp_path, t, sleeps)
        answer = c.choose(0, "P1", PROMPT)
    assert answer.choice in PROMPT.ids and t.calls == 4
    assert c.counts.retries == 3 and len(sleeps) == 3
    assert sleeps[0] >= 7.0, "retry-after is honoured"
    cap = config().number("backoff_cap_seconds")
    assert all(0 < s <= cap for s in sleeps)
    assert sum("retrying" in r.message for r in caplog.records) == 3


def test_lx12_backoff_is_capped_and_grows():
    cfg = config(max_attempts=8)
    sleeps = []
    t2 = StubTransport(fail={n: TransientError("500") for n in range(1, 8)})
    body = {"model": "m", "output_config": {"format": {"schema": {"properties": {"choice": {"enum": ["1"]}}}}}}
    send_with_retries(t2, body, cfg, sleeps.append, random.Random(0))
    assert len(sleeps) == 7 and max(sleeps) <= cfg.number("backoff_cap_seconds")
    assert sleeps[5] > sleeps[0]


def test_lx12_exhausted_is_not_a_result(tmp_path):
    cfg = config()
    attempts = cfg.integer("max_attempts")
    t = StubTransport(fail_if=_in_round(4, TransientError("503")))
    out = run_one(phase(6), "A1", 1, cfg=cfg, live=True, transport=t, runs=tmp_path, sleep=NO_SLEEP)
    assert out.record["status"] == "incomplete_transient"
    assert out.summary is None and not (tmp_path / "main-A1-s1" / "summary.json").exists()
    assert out.record["weeks_completed"] == 3
    assert t.calls >= attempts, "each failing request was tried max_attempts times"
    assert out.record["selections"] == 3 * 7, "no choice recorded in the week that failed"


def test_lx12_resumes_from_cache(tmp_path):
    cfg = config()
    failing = StubTransport(fail_if=_in_round(4, TransientError("connection")))
    first = run_one(phase(6), "A1", 1, cfg=cfg, live=True, transport=failing, runs=tmp_path, sleep=NO_SLEEP)
    assert first.record["status"] == "incomplete_transient"
    cached = len(ResponseCache.load(tmp_path / "main-A1-s1" / "responses.jsonl.gz").entries)
    assert cached >= 3 * 7
    again = StubTransport()
    resumed = run_one(phase(6), "A1", 1, cfg=cfg, live=True, transport=again, runs=tmp_path)
    assert resumed.record["status"] == "complete"
    assert resumed.record["counts"]["cached"] == cached and again.calls == resumed.record["counts"]["calls"]
    whole = run_one(phase(6), "A1", 1, cfg=cfg, live=True, transport=StubTransport(), runs=tmp_path / "whole")
    assert resumed.summary == whole.summary, "a resumed run is the run made in one go"


def _in_round(n, error):
    return lambda body: error if f"Round {n}." in body["messages"][0]["content"] else None


def test_lx13_permanent_is_not_retried(tmp_path, caplog):
    sleeps = []
    t = StubTransport(fail={1: PermanentError("401: invalid x-api-key " + FAKE_KEY)})
    with caplog.at_level(logging.ERROR, logger="experiments.llm_agents.client"):
        with pytest.raises(PermanentError):
            _chooser(tmp_path, t, sleeps).choose(0, "P1", PROMPT)
    assert t.calls == 1 and sleeps == []
    assert caplog.records and all("sk-ant-api03" not in r.getMessage() for r in caplog.records), "the key is redacted"
    # A permanent failure stops the run, and its record says so.
    t = StubTransport(fail_if=_in_round(2, PermanentError("400: bad request")))
    with pytest.raises(PermanentError):
        run_one(phase(4), "A1", 1, cfg=config(), live=True, transport=t, runs=tmp_path)
    rec = json.loads((tmp_path / "main-A1-s1" / "run.json").read_text())
    assert rec["status"] == "stopped_permanent" and rec["weeks_completed"] == 1
    sent = [json.dumps(b, sort_keys=True) for b in t.bodies]
    assert len(sent) == len(set(sent)), "no request was sent twice"


def _fake_sdk(raise_):
    """A stand-in ``anthropic`` module whose client raises ``raise_(sdk)`` on every create."""
    sdk = types.ModuleType("anthropic")
    sdk.__version__ = "fake"

    class APIError(Exception):
        def __init__(self, message="", response=None, status_code=None):
            super().__init__(message)
            self.message, self.response, self.status_code = message, response, status_code

    class APIConnectionError(APIError):
        pass

    class APITimeoutError(APIConnectionError):
        pass

    class APIStatusError(APIError):
        pass

    class RateLimitError(APIStatusError):
        pass

    class NotFoundError(APIStatusError):
        pass

    for cls in (APIError, APIConnectionError, APITimeoutError, APIStatusError, RateLimitError, NotFoundError):
        setattr(sdk, cls.__name__, cls)

    class Messages:
        @property
        def with_raw_response(self):
            return self

        def create(self, **body):
            raise raise_(sdk)

    class Models:
        def __init__(self):
            self.calls = 0

        def retrieve(self, model):
            self.calls += 1
            if self.calls == 1:
                raise raise_(sdk)
            return types.SimpleNamespace(id=model)

    class Anthropic:
        def __init__(self, timeout, max_retries):
            assert max_retries == 0, "the SDK's own retries are off"
            self.timeout, self.messages, self.default_headers = timeout, Messages(), {}
            self.models = Models()

    sdk.Anthropic = Anthropic
    return sdk


@pytest.mark.parametrize("status, kind", [(400, PermanentError), (401, PermanentError), (403, PermanentError),
                                          (404, PermanentError), (408, TransientError), (500, TransientError),
                                          (529, TransientError)])
def test_lx13_status_codes_map_to_permanent_or_transient(monkeypatch, status, kind):
    sdk = _fake_sdk(lambda m: m.APIStatusError("boom " + FAKE_KEY, status_code=status))
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    with pytest.raises(kind) as e:
        AnthropicTransport(config()).send({})
    assert "SECRETSECRET" not in str(e.value)


@pytest.mark.parametrize("make", [
    lambda m: m.RateLimitError("slow", response=types.SimpleNamespace(headers={"retry-after": "3"}), status_code=429),
    lambda m: m.APITimeoutError("t"), lambda m: m.APIConnectionError("c")])
def test_lx12_rate_limit_timeout_connection_are_transient(monkeypatch, make):
    monkeypatch.setitem(sys.modules, "anthropic", _fake_sdk(make))
    with pytest.raises(TransientError) as e:
        AnthropicTransport(config()).send({})
    if e.value.what == "429":
        assert e.value.retry_after == 3.0


def _reply(text, stop="end_turn", category=None):
    return Response(text=text, usage={"input_tokens": 10, "output_tokens": 5}, stop_reason=stop,
                    stop_category=category, model="m", request_id="r")


class Scripted:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def send(self, body):
        self.bodies.append(body)
        return self.replies.pop(0)


def test_lx14_failed_parse_path(tmp_path):
    ok = _reply(json.dumps({"choice": "2"}))
    reask = load_text()["reask"]
    cases = [
        ([_reply("", "refusal", "cyber"), ok], "2", 1),
        ([_reply('{"choice": "', "max_tokens"), ok], "2", 1),
        ([_reply(json.dumps({"choice": "9"})), ok], "2", 1),     # outside the legal set
        ([_reply("not json"), _reply("{}")], None, 1),
        ([_reply("", "refusal", None), _reply("", "refusal", None)], None, 1),
        ([ok], "2", 0),
    ]
    for i, (replies, choice, reasks) in enumerate(cases):
        t = Scripted(replies)
        c = _chooser(tmp_path / str(i), t)
        answer = c.choose(0, "P1", PROMPT)
        assert (answer.choice, answer.reasks) == (choice, reasks), i
        assert c.counts.fallbacks == (choice is None)
        if len(t.bodies) == 2:
            assert t.bodies[1]["messages"][0]["content"].endswith(reask), "the re-ask repeats the round"
            assert len(c.cache.entries) == 2, "the re-ask is cached separately"
        enum = t.bodies[0]["output_config"]["format"]["schema"]["properties"]["choice"]["enum"]
        assert enum == ["1", "2"]
    # A run whose every answer fails is invalid, and its counts say so.
    bad = StubTransport(text=lambda body: "no")
    out = run_one(phase(3), "A1", 1, cfg=config(), live=True, transport=bad, runs=tmp_path / "run")
    assert out.record["status"] == "invalid" and out.summary is None
    assert out.record["counts"]["fallbacks"] == out.record["selections"] > 0
    assert out.record["counts"]["failures"] == {"missing": 2 * out.record["selections"]}


def test_lx12_retry_after_beyond_the_cap_is_honoured():
    # A retry-after longer than the backoff cap is waited out (up to its own maximum), not cut to the cap, which
    # would spend the attempts early and end a run incomplete sooner than it needs to.
    cfg = config()
    assert cfg.number("backoff_cap_seconds") < 120 <= cfg.number("retry_after_max_seconds")
    fails = [TransientError("429", 120.0), TransientError("429", 10_000.0)]
    sleeps = []

    def call():
        if fails:
            raise fails.pop(0)
        return "ok"

    assert call_with_retries(call, cfg, sleeps.append, random.Random(0)) == "ok"
    assert sleeps == [120.0, cfg.number("retry_after_max_seconds")]


def test_lx12_model_check_is_retried(monkeypatch):
    # The Models check goes through the one retry loop: a 529 on the first retrieve is retried, not raised raw.
    monkeypatch.setitem(sys.modules, "anthropic", _fake_sdk(lambda m: m.APIStatusError("busy", status_code=529)))
    transport = AnthropicTransport(config())
    assert transport.check_models(("claude-haiku-5-5",), sleep=NO_SLEEP) == {"claude-haiku-5-5": "claude-haiku-5-5"}
    assert transport.client.models.calls == 2
    monkeypatch.setitem(sys.modules, "anthropic", _fake_sdk(lambda m: m.NotFoundError("gone", status_code=404)))
    with pytest.raises(PermanentError):
        AnthropicTransport(config()).check_models(("claude-haiku-5-5",), sleep=NO_SLEEP)
