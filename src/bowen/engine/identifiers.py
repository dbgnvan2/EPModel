"""Stable identifiers for persons, ties and triangles.

Purpose: give every object an identifier that is the same in every counterfactual arm.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.20, #M1.B.13, #M1.C.7, #M2.A.0h
Tests:   tests/bowen/test_objects.py::test_m1a20_identifiers_are_declared_not_counted

Identifiers are never allocated from a run-time counter. A death in one arm
would shift every later counter value in the other, and keyed draws (M3.D.4a)
would then pair different people across arms (M1.A.20).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PERSON_ID = re.compile(r"^[a-z][a-z0-9_]*(\+[a-z][a-z0-9_]*#\d+)*$")


@dataclass(frozen=True, order=True)
class PersonId:
    """A declared person identifier. Display names are not identifiers (M2.A.0h)."""

    value: str

    def __post_init__(self) -> None:
        if not _PERSON_ID.match(self.value):
            raise ValueError(f"invalid person identifier {self.value!r}")

    def __str__(self) -> str:
        return self.value


def child_id(parent_a: PersonId, parent_b: PersonId, birth_order: int) -> PersonId:
    """Purpose: derive a newborn's identifier from its parents and birth order.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.20
    Tests:   tests/bowen/test_objects.py::test_m1a20_child_id_depends_only_on_parents_and_order
    """
    if parent_a == parent_b:
        raise ValueError("a child needs two distinct parents")
    if birth_order < 1:
        raise ValueError("birth order starts at 1")
    first, second = sorted((parent_a, parent_b))
    # "+" and "#" cannot occur in a declared founder id, so a derived id
    # cannot collide with one.
    return PersonId(f"{first.value}+{second.value}#{birth_order}")


@dataclass(frozen=True, order=True)
class TieId:
    """The unordered pair of a tie's members (M1.B.13), stored sorted."""

    a: PersonId
    b: PersonId

    @classmethod
    def of(cls, x: PersonId, y: PersonId) -> "TieId":
        if x == y:
            raise ValueError(f"a tie needs two distinct persons, got {x} twice")
        first, second = sorted((x, y))
        return cls(first, second)

    def __post_init__(self) -> None:
        if not self.a < self.b:
            raise ValueError("TieId members must be distinct and sorted; use TieId.of")

    def members(self) -> tuple[PersonId, PersonId]:
        return (self.a, self.b)

    def other(self, person: PersonId) -> PersonId:
        if person == self.a:
            return self.b
        if person == self.b:
            return self.a
        raise ValueError(f"{person} is not on tie {self}")

    def __str__(self) -> str:
        return f"{self.a}~{self.b}"


@dataclass(frozen=True, order=True)
class TriangleId:
    """The sorted triple of a triangle's members (M1.C.7)."""

    members: tuple[PersonId, PersonId, PersonId]

    @classmethod
    def of(cls, x: PersonId, y: PersonId, z: PersonId) -> "TriangleId":
        trio = tuple(sorted((x, y, z)))
        if len(set(trio)) != 3:
            raise ValueError(f"a triangle needs three distinct persons, got {trio}")
        return cls(trio)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if len(set(self.members)) != 3 or tuple(sorted(self.members)) != self.members:
            raise ValueError("TriangleId members must be distinct and sorted; use TriangleId.of")

    def ties(self) -> tuple[TieId, TieId, TieId]:
        x, y, z = self.members
        return (TieId.of(x, y), TieId.of(x, z), TieId.of(y, z))

    def __str__(self) -> str:
        return "/".join(str(m) for m in self.members)
