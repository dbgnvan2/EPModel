"""Activation — which persons select a move in a given fast tick.

Purpose: decide, each tick, who selects; kept separate from visibility (M3.E.1).
Spec:    docs/bowen_agent_model_spec_v2.md#M3.E.1, #M3.E.2, #M16.A.1a
Tests:   tests/bowen/test_activation_visibility.py

The default regime — every person selects every fast tick — is the synchronous
end of a range of activation schemes. It is a modelling assumption, graded
[I] (M3.E.2), and the log header records which regime ran (M16.A.1a).

A person whose age the run tracks selects only from ``acting_age`` (10, the owner's) until death (M7.C.1f), and an
inactive person (a retired coach, a family-of-origin holder) never selects (M2.A.0k). A run that tracks no ages is
unchanged.
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

    def __init__(self, acting_age: int = 0) -> None:
        self.acting_age = acting_age

    def active(self, tick: int, people: Iterable[Person]) -> tuple[PersonId, ...]:
        """Spec: docs/bowen_agent_model_spec_v2.md#M3.E.2, #M7.C.1f
        Tests: tests/bowen/test_life_course.py::test_m7c1f_a_child_under_ten_never_selects
        """
        if tick < 0:
            raise ValueError("tick is non-negative")
        return tuple(sorted(p.id for p in people if p.alive and not p.inactive
                            and (p.age_years is None or p.age_years >= self.acting_age)))


ACTIVATION_COMPONENTS = {SynchronousActivation.regime: SynchronousActivation}


def activation_for(regime: str, acting_age: int = 0) -> SynchronousActivation:
    try:
        return ACTIVATION_COMPONENTS[regime](acting_age)
    except KeyError:
        raise ValueError(f"no activation component for regime {regime!r}") from None
