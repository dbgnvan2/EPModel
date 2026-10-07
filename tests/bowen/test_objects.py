"""Identifiers and the structural constraints on the four core objects.

Purpose: prove the M1 constraints that can be enforced at construction.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.20, #M1.B.13, #M1.C.7, #M1.A, #M1.B, #M1.C, #M1.D
Tests:   this file
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from src.bowen.engine import identifiers
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId, child_id
from src.bowen.engine.objects import (
    Family,
    Person,
    Relationship,
    Role,
    Sex,
    SiblingPosition,
    Sink,
    StructuralTier,
    SymptomChannel,
    TieState,
    Triangle,
)

RAVI, MARTA, NADIA = PersonId("ravi"), PersonId("marta"), PersonId("nadia")


def person(pid: PersonId = RAVI, **overrides) -> Person:
    values = dict(
        id=pid,
        role=Role.MEMBER,
        sex=Sex.MALE,
        household_id="h1",
        basic_level=39.0,
        functional_level=39.0,
        chronic_anxiety=44.0,
        sibling_position=SiblingPosition(1, 3),
        financially_dependent=False,
    )
    values.update(overrides)
    return Person(**values)


def tie(x: PersonId = RAVI, y: PersonId = MARTA, **overrides) -> Relationship:
    values = dict(id=TieId.of(x, y), conductance=1.0, bond_energy=10.0, latency=1)
    values.update(overrides)
    return Relationship(**values)


# --- identifiers -----------------------------------------------------------


def test_m1a20_identifiers_are_declared_not_counted():
    """No counter, no global state: identifiers come from declarations or parents (M1.A.20)."""
    source = Path(identifiers.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "count" not in names | attrs, "itertools.count or a counter attribute in identifiers"
    assert not any(isinstance(n, ast.Global) for n in ast.walk(tree))
    with pytest.raises(ValueError):
        PersonId("Ravi Smith")  # a display name is not an identifier (M2.A.0h)


def test_m1a20_child_id_depends_only_on_parents_and_order():
    first = child_id(RAVI, MARTA, 1)
    assert first == child_id(MARTA, RAVI, 1)  # parent order is irrelevant
    assert first != child_id(RAVI, MARTA, 2)
    # Two arms, one of which has an extra death: the child's id cannot move.
    assert child_id(RAVI, MARTA, 1) == PersonId("marta+ravi#1")


def test_m1a20_derived_id_cannot_collide_with_a_founder():
    derived = child_id(RAVI, MARTA, 1)
    assert "+" in derived.value and "#" in derived.value
    with pytest.raises(ValueError):
        PersonId("marta#1")  # "#" without a parent pair is not a founder id


def test_m1b13_tie_id_is_unordered_pair():
    assert TieId.of(RAVI, MARTA) == TieId.of(MARTA, RAVI)
    assert TieId.of(RAVI, MARTA).members() == (MARTA, RAVI)
    with pytest.raises(ValueError):
        TieId.of(RAVI, RAVI)
    with pytest.raises(ValueError):
        TieId(RAVI, MARTA)  # unsorted direct construction is refused


def test_m1c7_triangle_id_is_sorted_triple():
    assert TriangleId.of(RAVI, NADIA, MARTA) == TriangleId.of(NADIA, MARTA, RAVI)
    assert TriangleId.of(RAVI, NADIA, MARTA).members == (MARTA, NADIA, RAVI)
    with pytest.raises(ValueError):
        TriangleId.of(RAVI, RAVI, MARTA)


# --- Person -----------------------------------------------------------------


@pytest.mark.parametrize("field_name", ["basic_level", "functional_level"])
@pytest.mark.parametrize("value", [-0.1, 100.1])
def test_m1a2_basic_level_out_of_range_raises_not_clips(field_name, value):
    with pytest.raises(ValueError, match="not clipped"):
        person(**{field_name: value})


def test_m1a2_full_scale_is_accepted():
    """M1.A.2 forbids clipping to [10, 80]; the ends of 0–100 must construct."""
    assert person(basic_level=0.0, functional_level=0.0).basic_level == 0.0
    assert person(basic_level=100.0, functional_level=100.0).basic_level == 100.0


def test_m1a1_reactivity_is_not_stored():
    fields = {f.name for f in dataclasses.fields(Person)}
    assert not any("reactivity" in name and name != "programmed_reactivity" for name in fields)


def test_m1a5a_functional_level_decomposes_into_basic_plus_swing():
    p = person(basic_level=39.0, functional_level=45.5)
    assert p.swing == pytest.approx(6.5)
    assert p.basic_level + p.swing == p.functional_level


def test_m1a9a_outside_ness_is_two_dimensional():
    fields = {f.name for f in dataclasses.fields(Person)}
    assert {"outside_ness_outward", "outside_ness_inward"} <= fields
    assert "outside_ness" not in fields


def test_m1a11_exactly_three_symptom_channels():
    assert [c.value for c in SymptomChannel] == ["physical", "mental", "social"]
    assert set(person().symptom_load) == set(SymptomChannel)
    with pytest.raises(ValueError, match="three"):
        person(symptom_load={SymptomChannel.PHYSICAL: 0.0})


def test_m1a12_membership_is_not_a_stored_set():
    fields = {f.name for f in dataclasses.fields(Family)}
    assert not fields & {"members", "membership", "member_ids"}


def test_m1a13_exactly_three_structural_tiers():
    assert len(StructuralTier) == 3


def test_m1a15_no_material_stock():
    every_field = {
        f.name for cls in (Person, Relationship, Triangle, Family) for f in dataclasses.fields(cls)
    }
    assert not any(word in name for name in every_field for word in ("resource", "wealth", "money", "stock"))


def test_m1a17_external_agent_is_a_person_with_a_role():
    coach = person(PersonId("halim"), role=Role.EXTERNAL)
    assert isinstance(coach, Person) and coach.role is Role.EXTERNAL


def test_m1a22_household_is_required():
    with pytest.raises(ValueError, match="household_id"):
        person(household_id="")


# --- Relationship -------------------------------------------------------------


def test_m1b1_tie_connects_exactly_two_persons():
    assert len(tie().id.members()) == 2


def test_m1b2_conductance_has_no_distance_or_contact_input():
    fields = {f.name for f in dataclasses.fields(Relationship)}
    banned = ("distance_miles", "physical_distance", "contact_frequency", "last_interaction", "elapsed")
    assert not fields & set(banned)


def test_m1b3_four_tie_states_are_representable():
    """Construction half of M1.B.3; that they are distinguishable in behaviour is a step 8 test."""
    assert {TieState.CUT_OFF, TieState.DISTANT, TieState.RESOLVED_LOW_CONTACT, TieState.OPEN_CONFLICT} <= set(TieState)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_m3c2_latency_is_whole_ticks_at_least_one(bad):
    with pytest.raises(ValueError, match="latency"):
        tie(latency=bad)


# --- Triangle -----------------------------------------------------------------


def test_m1c1_triangle_holds_members_inside_pair_outside_and_bound_anxiety():
    tri = Triangle(TriangleId.of(RAVI, MARTA, NADIA), inside_pair=TieId.of(RAVI, NADIA), outside=MARTA)
    assert tri.outside == MARTA and tri.bound_anxiety == 0.0
    with pytest.raises(ValueError):
        Triangle(TriangleId.of(RAVI, MARTA, NADIA), inside_pair=TieId.of(RAVI, NADIA), outside=RAVI)


def test_m1c3_topology_is_stored_apart_from_activity():
    """A persistent triangle exists while inactive (M1.C.3)."""
    tri = Triangle(TriangleId.of(RAVI, MARTA, NADIA))
    assert tri.active is False and tri.id == TriangleId.of(RAVI, MARTA, NADIA)


# --- Family -------------------------------------------------------------------


def test_m1d1_exactly_three_sinks():
    assert len(Sink) == 3
    with pytest.raises(ValueError, match="three"):
        Family(undifferentiation_budget=1.0, sink_allocations={Sink.MARITAL_CONFLICT: 1.0})


def test_m1d2_distance_is_not_a_sink():
    assert not any("distance" in s.value for s in Sink)
