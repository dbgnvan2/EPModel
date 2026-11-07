"""LLMSource — an engine Source whose family members choose through a chooser, and the run's assembly.

Purpose: give every active living family member one selection a week, chosen by a chooser (an LLM, a stub or a
         replayed cache) from the policy's own legal set narrowed by the offer gate, and give the external agent the
         model's policy; assemble a run exactly as the ensemble does, with A0 on the policy.
Spec:    docs/plan_v3_llm_agent_experiment.md#3.1, #4, #7, #8
Tests:   tests/experiments/llm_agents/test_source.py

The engine is used, never changed: the legal set is ``policy.legal_outcomes``, the capacity gate is
``policy.availability``, an act's force is ``policy.force``, and an LLM selection is stamped ``DecidedBy.SCRIPTED``
with no value key, so the learner registers nothing for it (§4). Selections are formed from the state at step 7 and
nothing is emitted before step 8, so the week's calls are independent and run concurrently.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from experiments.llm_agents.arms import SPECS, Permutations, permutations
from experiments.llm_agents.config import load_labels, load_llm, load_text
from experiments.llm_agents.prompts import Labeller, OptionSpec, Prompt, RoundInput, Seen, render
from src.bowen.engine.act import WITHHOLD, Selection
from src.bowen.engine.draws import DrawKey
from src.bowen.engine.events import Role as DeliveryRole
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import DecidedBy
from src.bowen.engine.objects import Role
from src.bowen.engine.observe import Observation, observe
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState
from src.bowen.engine.tick import run_tick
from src.bowen.ensemble.criteria import FAMILIES, Forced, spell
from src.bowen.io.load import load_constants, load_event_kinds, load_family, load_policy_rules
from src.bowen.policy.policy import AUTOMATIC, NEEDS_TRIAD, Outcome, availability, force, legal_outcomes
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.policy_source import PolicySource
from src.bowen.scenario.scripted_source import ScriptedSource

OFFER_PURPOSE = "llm_offer"
FAILED = "llm_failed_to_parse"
NO_OPTIONS = "llm_no_options"


@dataclass(frozen=True)
class Answer:
    """A chooser's answer: an option id, or None when the response could not be read even after the re-ask."""

    choice: str | None
    reasks: int = 0
    reason: str | None = None   # why it failed to parse: refusal, max_tokens, missing, not_legal


class Chooser(Protocol):
    def choose(self, week: int, person_token: str, prompt: Prompt) -> Answer: ...


@dataclass(frozen=True)
class Choice:
    """One family member's week, as the experiment records it (the summary and the history read it)."""

    week: int
    person: str
    token: str
    option_id: str | None
    option: str | None          # the text shown for the chosen option
    kind: str | None            # the act kind the mechanics ran (WITHHOLD for a hold)
    target: str | None
    flag: str | None            # None, llm_failed_to_parse or llm_no_options
    reasks: int
    tension: float
    options: int = 0                     # how many options were shown (the list's length, N)
    spec: OptionSpec | None = None      # the chosen option's mechanics, as shown (§13.4)


def offered(obs: Observation, outcomes: Sequence[Outcome], kinds, params: EngineParams, state: RunState) -> list[Outcome]:
    """Purpose: the offer gate (§3.1): an automatic act is offered with probability ``availability``, one keyed draw
             per kind per person per week under its own purpose, so no other draw moves (LX.19).
    Spec:    docs/plan_v3_llm_agent_experiment.md#3.1
    Tests:   tests/experiments/llm_agents/test_source.py::test_lx19_offer_draw_is_isolated
    """
    order = sorted(kinds.layers)
    open_kinds = set()
    for kind in sorted({o.kind for o in outcomes if o.channel == AUTOMATIC}):
        share = availability(kind, obs, kinds, params)
        if share >= 1.0:
            open_kinds.add(kind)
            continue
        key = DrawKey.make("move_selection", tick=obs.tick, actor=obs.person, purpose=OFFER_PURPOSE,
                           index=order.index(kind))
        if state.draws.uniform(key) < share:
            open_kinds.add(kind)
    return [o for o in outcomes if o.kind != WITHHOLD and (o.channel != AUTOMATIC or o.kind in open_kinds)]


@dataclass
class _Memory:
    tension: float | None = None
    history: list = field(default_factory=list)   # [round, option text, tension when chosen, tension next round]


class LLMSource:
    """Purpose: the engine Source for arms A1–A4.
    Spec:    docs/plan_v3_llm_agent_experiment.md#2, #3.1, #7
    Tests:   tests/experiments/llm_agents/test_source.py::test_lx2_one_selection_per_active_member,
             tests/experiments/llm_agents/test_source.py::test_lx2_external_agent_keeps_the_policy,
             tests/experiments/llm_agents/test_source.py::test_lx3_choice_is_always_legal
    """

    def __init__(self, script: ScriptedSource, params: EngineParams, labeller: Labeller, chooser: Chooser, *,
                 history_rounds: int, max_concurrency: int = 1,
                 before_round: Callable[[int, Sequence[tuple[str, Prompt]]], None] | None = None,
                 on_prompt: Callable[[int, str, Prompt], None] | None = None):
        self.script, self.params, self.labeller, self.chooser = script, params, labeller, chooser
        self.policy = PolicySource(script, params, load_policy_rules())
        self.history_rounds, self.max_concurrency = history_rounds, max_concurrency
        self.before_round, self.on_prompt = before_round, on_prompt
        self.tokens = {PersonId(p): f"P{i}" for i, p in enumerate(labeller.perms.person_order, start=1)}
        self.memory: dict[PersonId, _Memory] = {}
        self.choices: list[Choice] = []

    @property
    def ticks(self) -> int:
        return self.script.ticks

    def scheduled(self, tick: int):
        return self.script.scheduled(tick)

    def _round(self, state: RunState, pid: PersonId) -> tuple:
        obs = observe(state, pid, self.params)
        tok = self.tokens
        legal = legal_outcomes(obs, state.kinds, self.params)
        acts = offered(obs, legal, state.kinds, self.params, state)
        specs, by_spec = [], {}
        for o in acts:
            third = None
            if o.kind in NEEDS_TRIAD:
                third = tok[next(m for m in obs.triangle_for[o.target].members if m not in (pid, o.target))]
            spec = OptionSpec(o.kind, tok[o.target], third)
            specs.append(spec)
            by_spec[spec] = o
            if o.channel == AUTOMATIC:
                hold = OptionSpec(o.kind, tok[o.target], third, hold=True)
                specs.append(hold)
                by_spec[hold] = o
        mem = self.memory.setdefault(pid, _Memory())
        tension = obs.acute_excess
        change = None if mem.tension is None else tension - mem.tension
        if mem.history and mem.history[-1][3] is None:
            # the change after an act is to this person's next round, whatever happened in that round
            mem.history[-1][3] = tension
        history = [(rnd, text, after - then) for rnd, text, then, after in mem.history[-self.history_rounds:]]
        ties = tuple(sorted((tok[v.other], v.deviation,
                             "broken" if v.severed else ("stopped" if not v.live or v.cut_off else "open"))
                            for v in obs.ties if v.other_alive))
        beliefs = tuple(sorted((*sorted((tok[a], tok[b])), x, c) for (a, b), (x, c) in
                               ((t.members(), v) for t, v in obs.beliefs.items())))
        seen = []
        for d in state.store.delivered_at(pid, state.tick):
            ev = state.store.event(d.event_id)
            seen.append(Seen(ev.kind if ev.kind in self.labeller.labels else None,
                             tok.get(ev.sender) if ev.sender is not None else None,
                             tok.get(ev.targets[0]) if ev.targets else None, d.role is DeliveryRole.WITNESS))
        r = RoundInput(obs.tick + 1, tension, change, ties, beliefs, tuple(seen), tuple(history), tuple(specs),
                       tok[pid])
        return r, specs, by_spec, tuple(o.label for o in legal), obs

    def selections(self, tick: int, active: tuple[PersonId, ...], state: RunState) -> tuple[Selection, ...]:
        scripted = {s.actor: s for s in self.script.selections(tick, active, state)}
        external = tuple(p for p in active if p not in scripted and state.people[p].role is Role.EXTERNAL)
        chosen = list(scripted.values()) + list(self.policy.selections(tick, external, state))
        family = [p for p in sorted(active) if p not in scripted and state.people[p].alive
                  and state.people[p].role is not Role.EXTERNAL]
        rounds = {p: self._round(state, p) for p in family}
        prompts = {p: render(self.labeller, rounds[p][0]) for p in family if rounds[p][1]}
        if self.before_round is not None:
            self.before_round(tick, [(self.tokens[p], prompts[p]) for p in sorted(prompts)])
        for p in sorted(prompts):
            if self.on_prompt is not None:
                self.on_prompt(tick, self.tokens[p], prompts[p])
        with ThreadPoolExecutor(max_workers=max(1, self.max_concurrency)) as pool:
            futures = {p: pool.submit(self.chooser.choose, tick, self.tokens[p], prompts[p]) for p in sorted(prompts)}
            answers = {p: f.result() for p, f in futures.items()}
        for p in family:
            r, specs, by_spec, legal, obs = rounds[p]
            chosen.append(self._select(p, r, specs, by_spec, legal, obs, prompts.get(p), answers.get(p)))
        return tuple(sorted(chosen, key=lambda s: s.actor))

    def _select(self, pid, r, specs, by_spec, legal, obs, prompt, answer) -> Selection:
        mem = self.memory[pid]
        mem.tension = r.tension
        token = self.tokens[pid]
        if prompt is None or answer is None or answer.choice not in prompt.ids:
            flag = NO_OPTIONS if prompt is None else FAILED
            self.choices.append(Choice(r.round - 1, pid.value, token, None, None, WITHHOLD, None, flag,
                                       answer.reasks if answer else 0, r.tension,
                                       len(prompt.ids) if prompt is not None else 0))
            return Selection(actor=pid, kind=WITHHOLD, targets=(), intensity=0.0, decided_by=DecidedBy.FALLBACK,
                             legal_set=legal, fallback_rule=flag)
        # the id is a position in the order this prompt was numbered in, never the fixed order (§13.1, LX.24)
        spec = prompt.specs[int(answer.choice) - 1]
        outcome = by_spec[spec]
        text = dict(prompt.options)[answer.choice]
        mem.history.append([r.round, text, r.tension, None])
        if spec.hold:
            sel = Selection(actor=pid, kind=WITHHOLD, targets=(outcome.target,), intensity=self.params.policy_intensity,
                            decided_by=DecidedBy.SCRIPTED, legal_set=legal, withheld=outcome.kind)
        else:
            sel = Selection(actor=pid, kind=outcome.kind, targets=(outcome.target,),
                            intensity=force(outcome, obs, self.params), decided_by=DecidedBy.SCRIPTED,
                            legal_set=legal)
        self.choices.append(Choice(r.round - 1, pid.value, token, answer.choice, text, sel.kind,
                                   outcome.target.value, None, answer.reasks, r.tension, len(prompt.ids), spec))
        return sel


@dataclass
class Run:
    arm: str
    seed: int
    weeks: int
    state: RunState
    params: EngineParams
    source: object
    perms: Permutations | None
    family: object
    parts: object
    records: list = field(default_factory=list)


def build(arm: str, seed: int, weeks: int, chooser: Chooser | None = None, *,
          before_round=None, on_prompt=None, option_order: str | None = None) -> Run:
    """Purpose: assemble one run as the ensemble's ``criteria.scenario`` does, with A0 on the policy (LX.17).
             ``option_order`` is the order its options are numbered in; None takes ``llm.md``'s (§13.1).
    Spec:    docs/plan_v3_llm_agent_experiment.md#2, #5
    Tests:   tests/experiments/llm_agents/test_source.py::test_lx17_a0_matches_the_ensemble
    """
    cfg = load_llm()
    name = cfg.get("family")
    if cfg.get("spells") != "standard":
        raise ValueError(f"llm.md spells {cfg.get('spells')!r}: only 'standard' is built")
    fam = load_family(FAMILIES[name])
    events = ScriptedSource(f"{name}-arm", weeks, tuple(spell(name, weeks)), ())
    parts = assemble(load_constants(), load_event_kinds(), fam, events, seed=seed)
    spec = SPECS[arm]
    if not spec.llm:
        source, perms = Forced(PolicySource(events, parts.params, load_policy_rules()), {}), None
    else:
        if chooser is None:
            raise ValueError(f"arm {arm} needs a chooser")
        labels = load_labels()
        perms = permutations(arm, seed, sorted(p.value for p in parts.state.people), labels)
        labeller = Labeller(spec, perms, labels, load_text(), option_order or cfg.get("option_order"), seed)
        source = LLMSource(events, parts.params, labeller, chooser, history_rounds=cfg.integer("history_rounds"),
                           max_concurrency=cfg.integer("max_concurrency"), before_round=before_round,
                           on_prompt=on_prompt)
    return Run(arm, seed, weeks, parts.state, parts.params, source, perms, fam, parts)


def step(run: Run, sink=None, keep: bool = True) -> None:
    """One week of the run. ``keep`` collects its records in ``run.records``; ``sink``, if given, gets them too.
    With neither, the week is run with logging off (LX.18)."""
    records = run.records

    class _Sink:
        def emit(self, r):
            if keep:
                records.append(r)
            if sink is not None:
                sink.emit(r)

    run_tick(run.state, run.source, run.params, run.parts.visibility, run.parts.activation, _Sink())


def execute(run: Run, sink=None, until: int | None = None, after_week=None, keep: bool = True) -> Run:
    for _ in range(run.weeks if until is None else until):
        step(run, sink, keep)
        if after_week is not None:
            after_week(run)
    return run


def family_terms(run: Run) -> tuple[str, ...]:
    """The family file's ids and display names, each word of them, for the lint (§3.4): banned too."""
    words = set()
    for pid, name in run.family.display_names.items():
        words.add(getattr(pid, "value", pid))
        words.update(name.split())
    return tuple(sorted(words))
