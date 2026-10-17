"""Reading the active sink from the belief configuration, and the configuration from the sink (`M9.6`).

Purpose: classify the spouses' attributions into `M9.6`'s three patterns, which name the three sinks of `M1.D.1`;
         and predict the pattern from the family's sink allocation.
Spec:    docs/bowen_agent_model_spec_v2.md#M9.6, #M11.C.26, #M1.D.1, #M11.D.22
Tests:   tests/bowen/test_attribution.py::test_m96_the_readout_recovers_each_pattern

Outside the engine (`M11.D.22`): it reads the run's state after the fact and changes nothing. Spec revision 25, Phase
D B8; Claude's recommendation, approved by the owner 2026-10-16.

* **Marital conflict — each says the other:** each spouse's largest attribution is the other spouse.
* **Spouse dysfunction — both say the same one:** both spouses' largest is the same spouse.
* **Child projection — both say the children:** both spouses' largest is a child of the nuclear household.
* Anything else, including a spouse with no attribution yet, is no pattern (``None``).
"""

from __future__ import annotations

from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.objects import Sink


def top(attribution: dict[PersonId, float]) -> PersonId | None:
    """The person a weight vector places the difficulty with most; None if empty or tied at the top."""
    positive = {p: w for p, w in attribution.items() if w > 0}
    if not positive:
        return None
    best = max(positive.values())
    leaders = [p for p, w in positive.items() if w == best]
    return leaders[0] if len(leaders) == 1 else None


def sink_from_beliefs(attributions: dict[PersonId, dict[PersonId, float]], spouses: tuple[PersonId, PersonId],
                      children: frozenset[PersonId]) -> Sink | None:
    """Purpose: the sink `M9.6`'s pattern names, read from the two spouses' attributions only.
    Spec:    docs/bowen_agent_model_spec_v2.md#M9.6
    Tests:   tests/bowen/test_attribution.py::test_m96_the_readout_recovers_each_pattern
    """
    a, b = spouses
    top_a, top_b = top(attributions.get(a, {})), top(attributions.get(b, {}))
    if top_a is None or top_b is None:
        return None
    if top_a == b and top_b == a:
        return Sink.MARITAL_CONFLICT
    if top_a == top_b and top_a in spouses:
        return Sink.SPOUSE_DYSFUNCTION
    if top_a in children and top_b in children:
        return Sink.CHILD_PROJECTION
    return None


def sink_from_allocations(allocations: dict[Sink, float]) -> Sink | None:
    """Purpose: the sink holding the largest share of the budget; None if none holds any or the top is tied.
    Spec:    docs/bowen_agent_model_spec_v2.md#M9.6, #M1.D.1
    Tests:   tests/bowen/test_attribution.py::test_m96_the_readout_recovers_each_pattern
    """
    positive = {s: v for s, v in allocations.items() if v > 0}
    if not positive:
        return None
    best = max(positive.values())
    leaders = [s for s, v in positive.items() if v == best]
    return leaders[0] if len(leaders) == 1 else None
