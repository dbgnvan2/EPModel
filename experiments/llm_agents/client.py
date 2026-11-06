"""The client layer: the request, the cache, the retry loop, the parse and the one re-ask.

Purpose: answer one person's round from the response cache, or, only under ``--live``, from the API through one
         retry loop that logs every retry, retries transient failures and stops on permanent ones; parse the answer
         against the round's legal options, re-ask once, and report a failed parse rather than guess (L4, P1, P5).
Spec:    docs/plan_v3_llm_agent_experiment.md#4, #7, #8
Tests:   tests/experiments/llm_agents/test_client.py, tests/experiments/llm_agents/test_cache.py

This is the only module that imports ``anthropic``, and it does so inside ``AnthropicTransport`` only, so the
package and every test import without it (§7). The key is the SDK's own: ``ANTHROPIC_API_KEY`` in the environment,
never a value in the repo, the config or a log (S1).
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from experiments.llm_agents.budget import Budget, call_cost
from experiments.llm_agents.cache import ResponseCache, cache_key, text_hash
from experiments.llm_agents.config import LLMConfig
from experiments.llm_agents.prompts import Prompt, schema
from experiments.llm_agents.source import Answer

log = logging.getLogger("experiments.llm_agents.client")

KEY_SHAPE = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


def redact(text: str) -> str:
    return KEY_SHAPE.sub("[redacted]", text)


class TransientError(Exception):
    """A failure worth retrying: 429, 408, 5xx, a timeout or a lost connection."""

    def __init__(self, what: str, retry_after: float | None = None):
        super().__init__(what)
        self.what, self.retry_after = what, retry_after


class PermanentError(Exception):
    """A failure no retry mends (400, 401, 403, 404, another 4xx): the run stops with it (LX.13)."""


class TransientExhausted(Exception):
    """Retries ran out: the run is ``incomplete_transient``, no choice is recorded, and a rerun resumes (P1)."""


class NotReplayable(Exception):
    """A replay found no cached response: the rendered prompt has moved, and nothing is filled in (LX.8)."""


@dataclass(frozen=True)
class Response:
    text: str
    usage: dict
    stop_reason: str | None
    stop_category: str | None
    model: str | None
    request_id: str | None


class Transport(Protocol):
    def send(self, body: dict) -> Response: ...


def request_body(cfg: LLMConfig, model: str, prompt: Prompt, reask: str | None = None) -> dict:
    """Purpose: the Messages API request for one round; the re-ask repeats the round with ``reask`` appended.
    Spec:    docs/plan_v3_llm_agent_experiment.md#4
    Tests:   tests/experiments/llm_agents/test_client.py::test_lx14_failed_parse_path
    """
    user = prompt.user if reask is None else f"{prompt.user}\n\n{reask}"
    return {
        "model": model,
        "max_tokens": cfg.integer("max_tokens"),
        "system": prompt.system,
        "messages": [{"role": "user", "content": user}],
        "thinking": {"type": cfg.thinking(model)},
        "output_config": {"effort": cfg.effort(model), "format": {"type": "json_schema", "schema": schema(prompt.ids)}},
    }


def parse(entry: dict, ids: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Purpose: the choice in a response, checked against the legal ids, or why there is none (L4).
    Spec:    docs/plan_v3_llm_agent_experiment.md#7
    Tests:   tests/experiments/llm_agents/test_client.py::test_lx14_failed_parse_path
    """
    if entry.get("stop_reason") == "refusal":
        return None, f"refusal:{entry.get('stop_category') or 'none'}"
    if entry.get("stop_reason") == "max_tokens":
        return None, "max_tokens"
    try:
        choice = json.loads(entry.get("text") or "").get("choice")
    except (ValueError, AttributeError):
        return None, "missing"
    if choice is None:
        return None, "missing"
    if str(choice) not in ids:
        return None, "not_legal"
    return str(choice), None


@dataclass
class Counts:
    calls: int = 0        # responses taken from the API this session
    cached: int = 0       # responses taken from the cache
    reasks: int = 0
    fallbacks: int = 0
    retries: int = 0
    cost: float = 0.0     # the cost of every response the run holds, cached or new
    failures: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        return {"calls": self.calls, "cached": self.cached, "reasks": self.reasks, "fallbacks": self.fallbacks,
                "retries": self.retries, "cost": round(self.cost, 6), "failures": dict(sorted(self.failures.items()))}


class CachedChooser:
    """Purpose: the Chooser for arms A1–A4: cache first, then the transport if one is given (``--live``).
    Spec:    docs/plan_v3_llm_agent_experiment.md#7, #8
    Tests:   tests/experiments/llm_agents/test_cache.py::test_lx8_replay_never_calls,
             tests/experiments/llm_agents/test_client.py::test_lx12_transient_is_retried,
             tests/experiments/llm_agents/test_client.py::test_lx13_permanent_is_not_retried
    """

    def __init__(self, *, run_id: str, model: str, arm: str, seed: int, cfg: LLMConfig, cache: ResponseCache,
                 reask_text: str, transport: Transport | None = None, budget: Budget | None = None,
                 sleep: Callable[[float], None] = time.sleep, jitter: random.Random | None = None):
        self.run_id, self.model, self.arm, self.seed, self.cfg = run_id, model, arm, seed, cfg
        self.cache, self.reask_text, self.transport, self.budget = cache, reask_text, transport, budget
        self.sleep, self.jitter = sleep, jitter or random.Random(0)
        self.counts = Counts()
        self.prompt_hashes: dict[str, str] = {}
        self.keys: dict[tuple[int, str], str] = {}   # the key each (week, person) was last answered from
        self._lock = threading.Lock()

    def _key(self, week: int, token: str, prompt: Prompt, attempt: int) -> str:
        body = request_body(self.cfg, self.model, prompt, None if attempt == 0 else self.reask_text)
        return cache_key(self.run_id, self.model, self.arm, self.seed, token, week, attempt, body)

    def needs_call(self, week: int, token: str, prompt: Prompt) -> bool:
        """Whether answering this prompt could call the API: its first attempt is not cached, or it is cached, does
        not parse, and the re-ask is not cached. A round wholly answered from the cache costs nothing (§5)."""
        entry = self.cache.get(self._key(week, token, prompt, 0))
        if entry is None:
            return True
        return parse(entry, prompt.ids)[0] is None and self.cache.get(self._key(week, token, prompt, 1)) is None

    def choose(self, week: int, person_token: str, prompt: Prompt) -> Answer:
        reason = None
        for attempt in (0, 1):
            body = request_body(self.cfg, self.model, prompt, None if attempt == 0 else self.reask_text)
            key = cache_key(self.run_id, self.model, self.arm, self.seed, person_token, week, attempt, body)
            entry = self.cache.get(key)
            if entry is None:
                if self.transport is None:
                    raise NotReplayable(f"{self.run_id}: no cached response for {person_token} week {week} "
                                        f"attempt {attempt}; the rendered prompt has changed")
                entry = self._fetch(key, body, week, person_token, attempt)
            else:
                with self._lock:
                    self.counts.cached += 1
                    self.counts.cost += entry.get("cost", 0.0)
            with self._lock:
                self.prompt_hashes[key] = entry["prompt_hash"]
                self.keys[(week, person_token)] = key
            choice, reason = parse(entry, prompt.ids)
            if choice is not None:
                if attempt:
                    with self._lock:
                        self.counts.reasks += 1
                return Answer(choice, reasks=attempt)
            with self._lock:
                self.counts.failures[reason.split(":")[0]] = self.counts.failures.get(reason.split(":")[0], 0) + 1
        with self._lock:
            self.counts.reasks += 1
            self.counts.fallbacks += 1
        log.warning("%s week %d %s: failed to parse after a re-ask (%s); a flagged hold", self.run_id, week,
                    person_token, reason)
        return Answer(None, reasks=1, reason=reason)

    def _fetch(self, key: str, body: dict, week: int, token: str, attempt: int) -> dict:
        response = send_with_retries(self.transport, body, self.cfg, self.sleep, self.jitter,
                                     on_retry=self._count_retry, what=f"{self.run_id} week {week} {token}")
        cost = call_cost(response.usage, self.cfg.prices(self.model))
        entry = {"key": key, "run_id": self.run_id, "week": week, "person": token, "attempt": attempt,
                 "prompt_hash": text_hash(json_body(body)), "text": response.text, "usage": response.usage,
                 "stop_reason": response.stop_reason, "stop_category": response.stop_category,
                 "model": response.model, "request_id": response.request_id, "cost": cost,
                 "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.cache.put(entry)
        with self._lock:
            self.counts.calls += 1
            self.counts.cost += cost
        if self.budget is not None:
            self.budget.add(cost)
        return entry

    def _count_retry(self) -> None:
        with self._lock:
            self.counts.retries += 1


def json_body(body: dict) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def send_with_retries(transport: Transport, body: dict, cfg: LLMConfig, sleep: Callable[[float], None],
                      jitter: random.Random, on_retry: Callable[[], None] = lambda: None, what: str = "") -> Response:
    """Purpose: one loop for every retry (E1, E5, P5): transient failures up to ``max_attempts`` with exponential
             backoff and jitter capped at ``backoff_cap_seconds``, honouring ``retry-after``; permanent ones at once.
    Spec:    docs/plan_v3_llm_agent_experiment.md#7
    Tests:   tests/experiments/llm_agents/test_client.py::test_lx12_transient_is_retried,
             tests/experiments/llm_agents/test_client.py::test_lx12_exhausted_is_not_a_result,
             tests/experiments/llm_agents/test_client.py::test_lx13_permanent_is_not_retried
    """
    return call_with_retries(lambda: transport.send(body), cfg, sleep, jitter, on_retry, what)


def call_with_retries(call: Callable[[], object], cfg: LLMConfig, sleep: Callable[[float], None],
                      jitter: random.Random, on_retry: Callable[[], None] = lambda: None, what: str = "",
                      endpoint: str = "POST /v1/messages"):
    """Purpose: the retry loop itself, for the Messages call and the Models check alike (P5: one loop, every call).
             A ``retry-after`` longer than the backoff cap is honoured up to ``retry_after_max_seconds``.
    Spec:    docs/plan_v3_llm_agent_experiment.md#7
    Tests:   tests/experiments/llm_agents/test_client.py::test_lx12_retry_after_beyond_the_cap_is_honoured,
             tests/experiments/llm_agents/test_client.py::test_lx12_model_check_is_retried
    """
    attempts, base, cap = cfg.integer("max_attempts"), cfg.number("backoff_base_seconds"), cfg.number(
        "backoff_cap_seconds")
    after_max = cfg.number("retry_after_max_seconds")
    for n in range(1, attempts + 1):
        try:
            return call()
        except PermanentError as e:
            log.error("%s %s: permanent failure, not retried: %s", endpoint, what, redact(str(e)))
            raise
        except TransientError as e:
            if n == attempts:
                log.error("%s %s: %s on attempt %d of %d; retries exhausted", endpoint, what, e.what, n,
                          attempts)
                raise TransientExhausted(f"{what}: {e.what} after {attempts} attempts") from e
            delay = min(cap, base * 2 ** (n - 1)) * (0.5 + 0.5 * jitter.random())
            if e.retry_after is not None:
                delay = min(after_max, max(delay, e.retry_after))
            log.warning("%s %s: %s on attempt %d of %d; retrying in %.1f s", endpoint, what, e.what, n,
                        attempts, delay)
            on_retry()
            sleep(delay)
    raise AssertionError("unreachable")


class AnthropicTransport:
    """Purpose: the live transport, built only under ``--live`` (LX.15): the SDK with its own retries off.
    Spec:    docs/plan_v3_llm_agent_experiment.md#7
    Tests:   tests/experiments/llm_agents/test_isolation.py::test_lx15_live_client_only_when_asked

    Its request shape against the real API is checked by the owner-approved smoke call (LX.L1), not by the suite.
    """

    def __init__(self, cfg: LLMConfig):
        import anthropic  # the one import of the SDK, here only (§7)

        self.sdk = anthropic
        self.cfg = cfg
        self.client = anthropic.Anthropic(timeout=cfg.number("timeout_seconds"), max_retries=0)

    @property
    def sdk_version(self) -> str:
        return getattr(self.sdk, "__version__", "unknown")

    @property
    def api_version(self) -> str:
        headers = getattr(self.client, "default_headers", {}) or {}
        return str(headers.get("anthropic-version", "sdk default"))

    def check_models(self, models: tuple[str, ...], sleep: Callable[[float], None] = time.sleep) -> dict:
        """Each model id against the Models API, through the one retry loop; a missing one stops the phase (§4)."""
        found = {}
        for m in models:
            found[m] = call_with_retries(lambda m=m: self._mapped(lambda: self.client.models.retrieve(m).id),
                                         self.cfg, sleep, random.Random(0), what=m, endpoint="GET /v1/models")
        return found

    def send(self, body: dict) -> Response:
        raw, message = self._mapped(lambda: self._create(body))
        return self._response(raw, message)

    def _create(self, body: dict):
        raw = self.client.messages.with_raw_response.create(**body)
        return raw, raw.parse()

    def _mapped(self, call: Callable[[], object]):
        """The SDK's errors as this layer's: 408, 429, 5xx, timeouts and connection errors transient, other 4xx
        permanent (§7)."""
        sdk = self.sdk
        try:
            return call()
        except sdk.RateLimitError as e:
            after = e.response.headers.get("retry-after") if e.response is not None else None
            raise TransientError("429", float(after) if after and after.replace(".", "", 1).isdigit() else None) from e
        except sdk.APITimeoutError as e:
            raise TransientError("timeout") from e
        except sdk.APIConnectionError as e:
            raise TransientError("connection") from e
        except sdk.APIStatusError as e:
            snippet = redact(str(getattr(e, "message", e)))[:200]
            if e.status_code == 408 or e.status_code >= 500:
                raise TransientError(str(e.status_code)) from e
            raise PermanentError(f"{e.status_code}: {snippet}") from e

    @staticmethod
    def _response(raw, message) -> Response:
        text = "".join(getattr(b, "text", "") for b in message.content if getattr(b, "type", "") == "text")
        details = getattr(message, "stop_details", None)
        usage = message.usage.model_dump() if hasattr(message.usage, "model_dump") else dict(message.usage)
        limits = {k: v for k, v in raw.headers.items() if k.lower().startswith("anthropic-ratelimit-")}
        if limits:
            log.info("rate limits: %s", limits)
        return Response(text=text, usage={k: v for k, v in usage.items() if isinstance(v, (int, float))},
                        stop_reason=message.stop_reason,
                        stop_category=getattr(details, "category", None) if details is not None else None,
                        model=message.model, request_id=getattr(message, "_request_id", None)
                        or raw.headers.get("request-id"))
