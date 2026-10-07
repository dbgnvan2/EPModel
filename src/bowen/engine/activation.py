"""Activation — which persons select a move in a given fast tick.

Purpose: decide, each tick, who selects; kept separate from visibility (M3.E.1).
Spec:    docs/bowen_agent_model_spec_v2.md#M3.E.1, #M3.E.2, #M16.A.1a
Tests:   tests/bowen/test_activation_visibility.py

The default regime — every person selects every fast tick — is the synchronous
end of a range of activation schemes. It is a modelling assumption, graded
[I] (M3.E.2), and the log header records which regime ran (M16.A.1a).
"""

from __future__ import annotations

from typing import Iterable

from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.objects import Person


class SynchronousActivation:
    """Purpose: every living person selects, every fast tick.
    Spec:    docs/bowen_agent_model_spec_v2.md#M3.E.2
    Tests:   tests/bowen/test_activation_visibility.py::test_m3e2_synchronous_activation_selects_every_living_person
    """

    component_id = "synchronous_activation"
    version = "1"
    regime = "synchronous"
    grade = "[I]"

    def active(self, tick: int, people: Iterable[Person]) -> tuple[PersonId, ...]:
        if tick < 0:
            raise ValueError("tick is non-negative")
        return tuple(sorted(p.id for p in people if p.alive))


ACTIVATION_COMPONENTS = {SynchronousActivation.regime: SynchronousActivation}


def activation_for(regime: str) -> SynchronousActivation:
    try:
        return ACTIVATION_COMPONENTS[regime]()
    except KeyError:
        raise ValueError(f"no activation component for regime {regime!r}") from None
