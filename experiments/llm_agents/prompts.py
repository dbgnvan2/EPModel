"""The prompt renderer — pure: values in, text out.

Purpose: render one person's round as the system prompt and a ``<round>`` message, from engine-generated values only,
         in the arm's labels, with the options numbered; and lint any text against the banned terms.
Spec:    docs/plan_v3_llm_agent_experiment.md#3.1, #3.2, #3.3, #3.4
Tests:   tests/experiments/llm_agents/test_prompts.py

No network, no file I/O, no engine state (L9): the source (``source.py``) turns an observation into a ``RoundInput``
of tokens and numbers, and this module only formats it. Every fixed word comes from ``config/prompt_text.md`` and
every act label from ``config/act_labels.md``.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from experiments.llm_agents.arms import ArmSpec, Permutations
from experiments.llm_agents.config import Label


@dataclass(frozen=True)
class OptionSpec:
    kind: str
    target: str               # the target's token
    third: str | None = None  # a triad act's third member's token
    hold: bool = False        # hold back from this automatic act (WITHHOLD)


@dataclass(frozen=True)
class Seen:
    """One event that reached the person this round, as tokens. ``kind`` is None for an event from outside."""

    kind: str | None
    sender: str | None
    target: str | None
    witnessed: bool


@dataclass(frozen=True)
class RoundInput:
    round: int
    tension: float
    change: float | None                         # None in the person's first round
    ties: tuple[tuple[str, float, str], ...]     # (other token, own strain, contact state: open/broken/stopped)
    beliefs: tuple[tuple[str, str, float, float], ...]  # (a, b, believed tension, believed contact)
    seen: tuple[Seen, ...]
    history: tuple[tuple[int, str, float | None], ...]  # (round, option text shown, change after it)
    options: tuple[OptionSpec, ...]


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    options: tuple[tuple[str, str], ...]         # (id, text), ids "1".."N"

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(i for i, _ in self.options)


@dataclass(frozen=True)
class Labeller:
    """How one run names acts: opaque tokens (A1) or plain labels, possibly shuffled (A3)."""

    spec: ArmSpec
    perms: Permutations
    labels: Mapping[str, Label]
    text: Mapping[str, str]

    def option(self, o: OptionSpec) -> str:
        if self.spec.opaque:
            key = "option_opaque_triad" if o.third is not None else "option_opaque"
            base = self.text[key].format(token=self.perms.act_token[o.kind], target=o.target, third=o.third)
        else:
            base = self.labels[self.perms.label_of[o.kind]].option.format(target=o.target, third=o.third)
        return self.text["option_hold"].format(text=base) if o.hold else base

    def seen(self, s: Seen) -> str:
        t = self.text
        if s.kind is None or s.kind not in self.labels:
            return t["seen_outside"]
        if self.spec.opaque:
            token = self.perms.act_token[s.kind]
            if s.witnessed:
                return t["seen_opaque_witness"].format(sender=s.sender, act=token, target=s.target)
            return t["seen_opaque_target"].format(sender=s.sender, act=token)
        phrase = self.labels[self.perms.label_of[s.kind]].seen.format(target=s.target if s.witnessed else "you")
        return t["seen_witness" if s.witnessed else "seen_target"].format(sender=s.sender, act=phrase)


def _num(x: float) -> str:
    return str(int(round(x)))


def _signed(x: float) -> str:
    n = int(round(x))
    return f"+{n}" if n > 0 else str(n)


def render(labeller: Labeller, r: RoundInput) -> Prompt:
    """Purpose: the system prompt and the round message, with the options numbered in a fixed order."""
    t, feedback = labeller.text, labeller.spec.feedback
    lines = ["<round>", t["round_header"].format(round=r.round)]
    if feedback:
        lines.append(t["tension_first"].format(tension=_num(r.tension)) if r.change is None
                     else t["tension_line"].format(tension=_num(r.tension), change=_signed(r.change)))
    lines.append(t["tie_header"])
    contact = {"open": t["contact_open"], "broken": t["contact_broken"], "stopped": t["contact_stopped"]}
    # A4 shows no tension number at all: the strain on a tie and a pair's believed tension go with the person's own (§2)
    tie_key, belief_key = ("tie_line", "belief_line") if feedback else ("tie_line_no_feedback", "belief_line_no_feedback")
    lines += [t[tie_key].format(other=o, strain=_num(s), contact=contact[c]) for o, s, c in r.ties]
    lines.append(t["belief_header"])
    lines += [t[belief_key].format(a=a, b=b, tension=_num(x), contact=f"{c:.2f}") for a, b, x, c in r.beliefs] \
        or [t["belief_none"]]
    lines.append(t["seen_header"])
    lines += [labeller.seen(s) for s in r.seen] or [t["seen_none"]]
    lines.append(t["history_header"])
    if r.history:
        for rnd, text, change in r.history:
            if feedback and change is not None:
                lines.append(t["history_line"].format(round=rnd, option=text, change=_signed(change)))
            else:
                lines.append(t["history_line_no_feedback"].format(round=rnd, option=text))
    else:
        lines.append(t["history_none"])
    options = tuple((str(i), labeller.option(o)) for i, o in enumerate(ordered_options(labeller, r.options), start=1))
    lines.append(t["option_header"])
    lines += [t["option_line"].format(id=i, text=text) for i, text in options]
    lines += ["</round>", t["answer"]]
    return Prompt(t["system"], "\n".join(lines), options)


def ordered_options(labeller: Labeller, options: Iterable[OptionSpec]) -> tuple[OptionSpec, ...]:
    """The options in the order ``render`` numbers them: id ``n`` is element ``n - 1``."""
    options = list(options)
    acts = sorted((o for o in options if not o.hold), key=lambda o: (o.target, labeller.option(o)))
    holds = sorted((o for o in options if o.hold), key=lambda o: (o.target, labeller.option(o)))
    return tuple(acts + holds)


@functools.lru_cache(maxsize=512)
def _pattern(terms: tuple[str, ...]) -> re.Pattern:
    alternatives = "|".join(re.escape(t.lower()) for t in sorted(set(terms), key=len, reverse=True))
    return re.compile(rf"(?<![a-z])(?:{alternatives})(?![a-z])")


def banned_hits(text: str, terms: Iterable[str]) -> list[str]:
    """Purpose: every banned term the text contains, case-insensitive and whole-word (LX.4).
    Spec:    docs/plan_v3_llm_agent_experiment.md#3.4
    Tests:   tests/experiments/llm_agents/test_prompts.py::test_lx4_a_banned_term_is_caught
    """
    terms = tuple(terms)
    if not terms:
        return []
    low = text.lower()
    if not _pattern(terms).search(low):  # one pass over the text for the common case: no hit
        return []
    return sorted({t for t in terms if _pattern((t,)).search(low)})


def schema(ids: tuple[str, ...]) -> dict:
    """The structured-output schema of one round: the choice is one of the round's option ids (§4)."""
    return {"type": "object", "properties": {"choice": {"type": "string", "enum": list(ids)}},
            "required": ["choice"], "additionalProperties": False}
