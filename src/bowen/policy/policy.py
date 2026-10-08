"""The policy skeleton — one outcome per person per tick (Phase C step 6, plan D3).

Purpose: from one person's observation, form the legal set, score the two channels,
         mix them by functional level, draw one outcome with a keyed draw, and report
         the rationale — the legal set, the propensities, the draw and what decided.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1, #M4.D.1a, #M4.D.1b, #M4.D.1d, #M4.D.1e, #M4.D.1f, #M4.D.2, #M4.D.3, #M4.D.3a, #M4.D.3b, #M4.D.4, #M5.C.1, #M5.F.5, #M3.D.4b, #M4.B.2, #M4.B.3
Tests:   tests/bowen/test_policy.py

Every form is the project's, graded [I]; numbers are in ``config/bowen/constants.md``,
the channel and capacity layer of each act in ``config/bowen/event_kinds.md``, the
tie-break and fallback rules in ``config/bowen/policy.md``.

**Outcomes** are an act toward one person, or ``WITHHOLD``. The legal set (`M4.D.1e`)
is formed first, from structural preconditions — a tie to a living person who is not an
external agent; across a severed tie only `REDUCE_CUTOFF`, and `REDUCE_CUTOFF` only there;
`TRIANGLE`, `DETRIANGLE` and `PREVENT_ALIGNMENT` only toward someone a closed triad holds —
and the `M5.C` gates that remove an act: `I-POSITION` is removed
for a financially dependent person (`M5.C.1`: it "MUST fail outright"). ``WITHHOLD`` is
legal when an automatic act is.

**Two channels** (`M4.D.1a`). The self-directed channel's weight is
``(functional_level / 100) ** self_channel_exponent`` — functional level only — and the
automatic channel has the rest. If one channel has nothing legal the other has all of it.

* **Automatic:** ``exp(value / temperature) × availability``. ``value`` is the learned
  value for (anxiety band, act, target) — for `TRIANGLE`, for the triad the act would
  form (`M11.C.42`) — and is 0 until learned, equal for every act (plan D3). The band
  (`M4.D.3`(b)) is the person's excess acute anxiety against two edges. ``availability``
  is `M4.D.3a`'s capacity gate: an act in layer ``k`` is available at
  ``min(1, functional_level / (k × capacity_level_per_layer))``; layer 0 always. No term
  reads anxiety and raises a reactive act (`M4.D.3`).
* **Self-directed:** scored by `M5.F.5`'s position, never by relief. With efficacy
  ``e = 1 − max(outward, inward)``, `I-POSITION`, ``WITHHOLD`` and the four `M5.B` family
  moves score the gap ``1 − e`` —
  the position is not held, so take or hold one — and `STAY-IN-CONTACT` scores ``e`` —
  held, so stay in contact. The scores are in [0, 1] and enter as a softmax over their
  logarithms, ``score ** (1 / temperature)``, so a position fully held gives `I-POSITION`
  no weight at all rather than ``exp(0)``. An act toward a **loaded**
  tie (own deviation above ``loaded_tie_threshold``) is scaled by
  ``systems_perspective`` (`M4.D.3b`): without mindware the loaded tie is avoided.
  Weight rises with level, the gap falls with it, so `I-POSITION`'s propensity is not
  monotone in level (`M4.D.4`).

**Selection** is one categorical draw over the mixed distribution, keyed
``move_selection(tick, actor, "select", 0)`` (`M3.D.4b`, slot). A ``WITHHOLD`` draws the
automatic act it holds back from the automatic distribution with purpose ``"withheld"``;
it is recorded and not emitted (`M4.D.1b`). **Competing urges** (`M4.D.1d`): the
automatic distribution's normalised entropy times ``competing_urge_gain`` is returned
for the engine to add to the person's acute anxiety, whatever the outcome.

**Tie-break and fallback** (`M4.D.1f`), as ``policy.md`` declares: equal scores take
equal probability in the softmax, so no tie-break draw is made; an empty legal set, or a
non-finite weight, gives the fallback — the person holds, nothing is emitted and no
automatic act is computed — flagged ``FALLBACK``.

The policy reads no belief yet: its one belief reader, `M4.D.2`'s triangle position, is
left to the learner's conditioning (step 7). ``beliefs_used`` is therefore empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.bowen.engine.act import WITHHOLD, Selection
from src.bowen.engine.draws import DrawKey, DrawService
from src.bowen.engine.events import EventKinds
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import DecidedBy
from src.bowen.engine.observe import Observation
from src.bowen.engine.params import EngineParams
from src.bowen.policy.rules import PolicyRules

AUTOMATIC, SELF = "automatic", "self"
# M5.A.1, M5.B and M5.C.1 name these acts; the names are the spec's.
CUTOFF, TRIANGLE, I_POSITION, STAY_IN_CONTACT = "CUTOFF", "TRIANGLE", "I-POSITION", "STAY-IN-CONTACT"
REDUCE_CUTOFF, DETRIANGLE, PREVENT_ALIGNMENT = "REDUCE_CUTOFF", "DETRIANGLE", "PREVENT_ALIGNMENT"
NEEDS_TRIAD = frozenset({TRIANGLE, DETRIANGLE, PREVENT_ALIGNMENT})


@dataclass(frozen=True)
class Outcome:
    kind: str
    target: PersonId | None
    channel: str
    value_key: str = ""

    @property
    def label(self) -> str:
        return self.kind if self.target is None else f"{self.kind}>{self.target}"


@dataclass(frozen=True)
class Decision:
    selection: Selection
    urge: float


def band(obs: Observation, params: EngineParams) -> str:
    if obs.acute_excess < params.anxiety_band_low:
        return "low"
    return "mid" if obs.acute_excess < params.anxiety_band_high else "high"


def value_key(obs: Observation, kind: str, target: PersonId, params: EngineParams) -> str:
    about = obs.triangle_for.get(target) if kind == TRIANGLE else None
    where = "/".join(m.value for m in about.members) if about is not None else target.value
    return f"{band(obs, params)}|{kind}|{where}"


def legal_outcomes(obs: Observation, kinds: EventKinds, params: EngineParams) -> list[Outcome]:
    """Purpose: the legal set, formed before selection from structure and the removing gates.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1e, #M5.C.1
    Tests:   tests/bowen/test_policy.py::test_m4d1e_legal_set_excludes_impossible_and_gated_acts
    """
    legal = []
    for view in obs.ties:
        if not view.other_alive or view.other_external:
            continue
        for kind, channel in sorted(kinds.channels.items()):
            # Across a severed tie the one legal act is to reduce the cutoff; on a live tie it is not legal.
            if kind == REDUCE_CUTOFF:
                if not view.severed:
                    continue
            elif not view.live:
                continue
            if kind in NEEDS_TRIAD and view.other not in obs.triangle_for:
                continue
            if kind == I_POSITION and obs.financially_dependent:
                continue  # M5.C.1: fails outright, so it is removed, not degraded
            key = value_key(obs, kind, view.other, params) if channel == AUTOMATIC else ""
            legal.append(Outcome(kind, view.other, channel, key))
    if any(o.channel == AUTOMATIC for o in legal):
        legal.append(Outcome(WITHHOLD, None, SELF))
    return legal


def self_channel_weight(functional_level: float, params: EngineParams) -> float:
    """Purpose: the mixing weight — a function of functional level only.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1a
    Tests:   tests/bowen/test_policy.py::test_m4d1a_mixing_weight_reads_functional_level_only
    """
    return min(1.0, max(0.0, functional_level / 100.0)) ** params.self_channel_exponent


def availability(kind: str, obs: Observation, kinds: EventKinds, params: EngineParams) -> float:
    """Purpose: the capacity gate on newer acts — older layers stay available as level falls.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.3a
    Tests:   tests/bowen/test_policy.py::test_m4d3a_lower_level_slides_selection_to_older_acts
    """
    layer = kinds.layers[kind]
    if layer == 0:
        return 1.0
    return min(1.0, max(0.0, obs.functional_level / (layer * params.capacity_level_per_layer)))


def efficacy(obs: Observation) -> float:
    return 1.0 - max(obs.outside_ness_outward, obs.outside_ness_inward)


def self_score(outcome: Outcome, obs: Observation) -> float:
    """Purpose: the self-directed channel's score — the two-axis position, never relief.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.F.5, #M4.D.6d
    Tests:   tests/bowen/test_policy.py::test_m5f5_self_directed_scores_read_position_not_relief
    """
    held = efficacy(obs)
    return held if outcome.kind == STAY_IN_CONTACT else 1.0 - held


def _loaded_gate(outcome: Outcome, obs: Observation, params: EngineParams) -> float:
    """M4.D.3b: engagement with a loaded tie is gated by systems perspective."""
    if outcome.target is None:
        return 1.0
    view = obs.tie_to(outcome.target)
    return obs.systems_perspective if view.deviation > params.loaded_tie_threshold else 1.0


def channel_weights(outcomes: list[Outcome], obs: Observation, kinds: EventKinds, params: EngineParams) -> list[float]:
    """Unnormalised within-channel weights, before mixing."""
    weights = []
    for o in outcomes:
        if o.channel == AUTOMATIC:
            value = obs.learned_values.get(o.value_key, 0.0)
            weights.append(math.exp(value / params.policy_temperature) * availability(o.kind, obs, kinds, params))
        else:
            weights.append(self_score(o, obs) ** (1.0 / params.policy_temperature) * _loaded_gate(o, obs, params))
    return weights


def propensities(obs: Observation, kinds: EventKinds, params: EngineParams) -> tuple[list[Outcome], list[float]]:
    """Purpose: the legal set and its mixed selection probabilities.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1, #M4.D.1a, #M4.D.2
    Tests:   tests/bowen/test_policy.py::test_m4d4_iposition_not_monotone_in_level
    """
    outcomes = legal_outcomes(obs, kinds, params)
    weights = channel_weights(outcomes, obs, kinds, params)
    if any(not math.isfinite(w) for w in weights):
        return outcomes, [math.nan] * len(outcomes)
    totals = {c: sum(w for o, w in zip(outcomes, weights) if o.channel == c) for c in (AUTOMATIC, SELF)}
    mix = {SELF: self_channel_weight(obs.functional_level, params)}
    mix[AUTOMATIC] = 1.0 - mix[SELF]
    live = {c for c in totals if totals[c] > 0}
    if live and sum(mix[c] for c in live) <= 0:
        mix = {c: 1.0 for c in live}  # the only channel with anything legal takes all of it
    norm = sum(mix[c] for c in live)
    probs = [mix[o.channel] / norm * w / totals[o.channel] if o.channel in live else 0.0
             for o, w in zip(outcomes, weights)]
    return outcomes, probs


def competing_urges(outcomes: list[Outcome], obs: Observation, kinds: EventKinds, params: EngineParams) -> float:
    """Purpose: anxiety from unresolved competition — the automatic channel's normalised entropy.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1d, #M6.4
    Tests:   tests/bowen/test_policy.py::test_m4d1d_competing_urges_raise_anxiety_by_entropy
    """
    auto = [w for o, w in zip(outcomes, channel_weights(outcomes, obs, kinds, params)) if o.channel == AUTOMATIC]
    total = sum(auto)
    if len(auto) < 2 or total <= 0 or not math.isfinite(total):
        return 0.0
    entropy = -sum(w / total * math.log(w / total) for w in auto if w > 0)
    return params.competing_urge_gain * entropy / math.log(len(auto))


def _key(obs: Observation, purpose: str) -> DrawKey:
    return DrawKey.make("move_selection", tick=obs.tick, actor=obs.person, purpose=purpose, index=0)


def decide(obs: Observation, kinds: EventKinds, params: EngineParams, rules: PolicyRules,
           draws: DrawService) -> Decision:
    """Purpose: resolve exactly one outcome for one person this tick, with its rationale.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.1, #M4.D.1b, #M4.D.1f, #M16.A.3, #M16.A.3b, #M16.A.3c
    Tests:   tests/bowen/test_policy.py::test_m4d1_exactly_one_outcome_per_person_per_tick
    """
    outcomes, probs = propensities(obs, kinds, params)
    legal = tuple(o.label for o in outcomes)
    if not outcomes or any(not math.isfinite(p) for p in probs) or sum(probs) <= 0:
        return Decision(Selection(
            actor=obs.person, kind=WITHHOLD, targets=(), intensity=0.0, decided_by=DecidedBy.FALLBACK,
            legal_set=legal, fallback_rule=rules.fallback,
        ), 0.0)
    urge = competing_urges(outcomes, obs, kinds, params)
    u = draws.uniform(_key(obs, "select"))
    chosen = outcomes[_invert(u, probs)]
    rationale = dict(
        decided_by=DecidedBy.POLICY, legal_set=legal, draw=u,
        propensities=tuple((o.label, p) for o, p in zip(outcomes, probs)),
    )
    if chosen.kind == WITHHOLD:
        auto = [(o, p) for o, p in zip(outcomes, probs) if o.channel == AUTOMATIC]
        held = auto[_invert(draws.uniform(_key(obs, "withheld")), [p for _, p in auto])][0]
        return Decision(Selection(
            actor=obs.person, kind=WITHHOLD, targets=(held.target,), intensity=params.policy_intensity,
            withheld=held.kind, **rationale,
        ), urge)
    return Decision(Selection(
        actor=obs.person, kind=chosen.kind, targets=(chosen.target,), intensity=params.policy_intensity,
        **rationale,
    ), urge)


def _invert(u: float, weights: list[float]) -> int:
    """Inverse transform: one uniform, one index (M3.D.4a)."""
    target = u * sum(weights)
    running = 0.0
    for index, weight in enumerate(weights):
        running += weight
        if target < running:
            return index
    return max(i for i, w in enumerate(weights) if w > 0)
