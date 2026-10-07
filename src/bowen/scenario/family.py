"""Build a family instance from its markdown declaration.

Purpose: turn a family declaration into Person, Relationship and Family objects,
         enforcing the reference-family rules that can be checked at construction.
Spec:    docs/bowen_agent_model_spec_v2.md#M2, #M2.3, #M2.3a, #M2.A.0a, #M2.A.0c, #M2.A.0e, #M2.A.0h, #M1.D.8
Tests:   tests/bowen/test_family.py

Every value in a family declaration is invented (`M2`: "every value here is
[I]"); the declaration says so in its ``grade`` metadata, and this module
refuses one that does not.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.objects import Family, Person, Relationship, Role, Sex, SiblingPosition, TieState
from src.bowen.scenario.config_parse import ConfigError, parse_sectioned_document
from src.bowen.scenario.constants import Constants

PEOPLE_COLUMNS = (
    "id", "name", "generation", "age", "sex", "household", "basic_level", "functional_level",
    "chronic_anxiety", "sibling_rank", "sibship_size", "financially_dependent", "role",
)
TIE_COLUMNS = ("a", "b", "relation", "conductance", "bond_energy", "latency", "tie_state", "interactive")
METADATA = frozenset({"instance_id", "grade", "nuclear_household", "undifferentiation_budget"})
_YES_NO = {"yes": True, "no": False}
_NONE = {"—", "-", ""}


class Relation(enum.Enum):
    SPOUSE = "spouse"
    PARENT_OF = "parent_of"  # column a is the parent of column b
    SIBLING = "sibling"


@dataclass(frozen=True)
class FamilyInstance:
    """Purpose: a constructed family — its objects plus the declaration facts that are not object state.
    Spec:    docs/bowen_agent_model_spec_v2.md#M2
    Tests:   tests/bowen/test_family.py::test_m23_reduced_instance_is_not_closed
    """

    instance_id: str
    people: Mapping[PersonId, Person]
    ties: Mapping[TieId, Relationship]
    family: Family
    display_names: Mapping[PersonId, str]
    ages_years: Mapping[PersonId, int]
    generations: Mapping[PersonId, int | None]
    relations: Mapping[TieId, Relation]
    parent_on_tie: Mapping[TieId, PersonId]
    nuclear_household: str

    def parents_of(self, person: PersonId) -> frozenset[PersonId]:
        return frozenset(
            parent
            for tie, parent in self.parent_on_tie.items()
            if person in tie.members() and parent != person
        )

    def family_of_origin_ties(self, person: PersonId) -> tuple[TieId, ...]:
        """Ties from ``person`` to a parent or a sibling (M1.D.8)."""
        found = []
        for tie, relation in self.relations.items():
            if person not in tie.members():
                continue
            if relation is Relation.SIBLING or (
                relation is Relation.PARENT_OF and self.parent_on_tie[tie] != person
            ):
                found.append(tie)
        return tuple(sorted(found))

    def nuclear_adults(self) -> tuple[PersonId, ...]:
        return tuple(
            sorted(
                p.id
                for p in self.people.values()
                if p.household_id == self.nuclear_household and not p.financially_dependent
            )
        )


def _number(raw: str, kind: type, where: str, name: str):
    try:
        if kind is int:
            if not raw.isdigit():
                raise ValueError
            return int(raw)
        return float(raw)
    except ValueError:
        raise ConfigError(f"{where}: {name} {raw!r} is not a valid {kind.__name__}") from None


def _choice(raw: str, options: Mapping[str, object], where: str, name: str):
    if raw not in options:
        raise ConfigError(f"{where}: {name} {raw!r} is not one of {sorted(options)}")
    return options[raw]


def build_family(text: str, constants: Constants, *, source: str = "<family>") -> FamilyInstance:
    """Purpose: parse and validate a family declaration into a FamilyInstance.
    Spec:    docs/bowen_agent_model_spec_v2.md#M2.1, #M2.3, #M2.A.0a, #M2.A.0e
    Tests:   tests/bowen/test_family.py::test_m21_family_declared_in_markdown
    """
    document = parse_sectioned_document(
        text,
        sections={"People": PEOPLE_COLUMNS, "Ties": TIE_COLUMNS},
        metadata_keys=METADATA,
        source=source,
    )
    meta = document.metadata
    if meta["grade"] != "[I]":
        raise ConfigError(f"{source}: a family declaration is invented and must say grade: [I] (M2)")

    fixation_age = constants["chronic_anxiety_fixation_age_years"]
    sexes = {s.value: s for s in Sex}
    roles = {r.value: r for r in Role}

    people: dict[PersonId, Person] = {}
    names: dict[PersonId, str] = {}
    ages: dict[PersonId, int] = {}
    generations: dict[PersonId, int | None] = {}
    people_table = document.sections["People"]
    for row, line in zip(people_table.rows, people_table.row_lines):
        where = f"{source}:{line}"
        try:
            pid = PersonId(row["id"].strip("`"))
        except ValueError as error:
            raise ConfigError(f"{where}: {error}") from None
        if pid in people:
            raise ConfigError(f"{where}: duplicate person {pid}")
        if not row["name"]:
            raise ConfigError(f"{where}: name is empty")
        age = _number(row["age"], int, where, "age")
        # M2.A.0a: the test is life stage, not generation.
        if age < fixation_age:
            raise ConfigError(
                f"{where}: {pid} is {age}, below the fixation age {fixation_age}; a supplied "
                "chronic_anxiety is an error for an agent not yet past it (M2.A.0a)"
            )
        if row["sex"] not in sexes:
            raise ConfigError(f"{where}: sex {row['sex']!r} must be declared as one of {sorted(sexes)} (M1.A.21)")
        sibling = None
        if row["sibling_rank"] not in _NONE:
            sibling = SiblingPosition(
                _number(row["sibling_rank"], int, where, "sibling_rank"),
                _number(row["sibship_size"], int, where, "sibship_size"),
            )
        try:
            person = Person(
                id=pid,
                role=_choice(row["role"], roles, where, "role"),
                sex=sexes[row["sex"]],
                household_id=row["household"],
                basic_level=_number(row["basic_level"], float, where, "basic_level"),
                functional_level=_number(row["functional_level"], float, where, "functional_level"),
                chronic_anxiety=_number(row["chronic_anxiety"], float, where, "chronic_anxiety"),
                # Acute anxiety starts on its floor (M1.A.7a: chronic is the floor it decays toward).
                acute_anxiety=_number(row["chronic_anxiety"], float, where, "chronic_anxiety"),
                sibling_position=sibling,
                financially_dependent=_choice(row["financially_dependent"], _YES_NO, where, "financially_dependent"),
            )
        except ValueError as error:
            if isinstance(error, ConfigError):
                raise
            raise ConfigError(f"{where}: {error}") from None
        people[pid] = person
        names[pid] = row["name"]
        ages[pid] = age
        generations[pid] = None if row["generation"] in _NONE else _number(row["generation"], int, where, "generation")

    states = {s.value: s for s in TieState}
    relations_by_value = {r.value: r for r in Relation}
    ties: dict[TieId, Relationship] = {}
    relations: dict[TieId, Relation] = {}
    parent_on_tie: dict[TieId, PersonId] = {}
    ties_table = document.sections["Ties"]
    for row, line in zip(ties_table.rows, ties_table.row_lines):
        where = f"{source}:{line}"
        a, b = PersonId(row["a"].strip("`")), PersonId(row["b"].strip("`"))
        for end in (a, b):
            if end not in people:
                raise ConfigError(f"{where}: tie names undeclared person {end}")
        tie_id = TieId.of(a, b)
        if tie_id in ties:
            raise ConfigError(f"{where}: duplicate tie {tie_id}")
        relation = _choice(row["relation"], relations_by_value, where, "relation")
        state = _choice(row["tie_state"], states, where, "tie_state")
        interactive = _choice(row["interactive"], _YES_NO, where, "interactive")
        if state is TieState.CUT_OFF and interactive:
            raise ConfigError(f"{where}: a cut-off tie carries no events, so it cannot be interactive (M1.B.3)")
        try:
            ties[tie_id] = Relationship(
                id=tie_id,
                conductance=_number(row["conductance"], float, where, "conductance"),
                bond_energy=_number(row["bond_energy"], float, where, "bond_energy"),
                latency=_number(row["latency"], int, where, "latency"),
                tie_state=state,
                interactive=interactive,
            )
        except ValueError as error:
            if isinstance(error, ConfigError):
                raise
            raise ConfigError(f"{where}: {error}") from None
        relations[tie_id] = relation
        if relation is Relation.PARENT_OF:
            parent_on_tie[tie_id] = a
        if relation is Relation.SPOUSE:
            # M2.A.0c: compare basic_level, never functional_level. M2.A.0e: within the tolerance.
            gap = abs(people[a].basic_level - people[b].basic_level)
            tolerance = constants["spouse_basic_level_tolerance"]
            if gap > tolerance:
                raise ConfigError(
                    f"{where}: spouses {a} and {b} differ by {gap} in basic_level, beyond the "
                    f"tolerance {tolerance} (M2.A.0e)"
                )

    instance = FamilyInstance(
        instance_id=meta["instance_id"],
        people=MappingProxyType(people),
        ties=MappingProxyType(ties),
        family=Family(undifferentiation_budget=_number(meta["undifferentiation_budget"], float, source, "undifferentiation_budget")),
        display_names=MappingProxyType(names),
        ages_years=MappingProxyType(ages),
        generations=MappingProxyType(generations),
        relations=MappingProxyType(relations),
        parent_on_tie=MappingProxyType(parent_on_tie),
        nuclear_household=meta["nuclear_household"],
    )
    # M2.3 / M1.D.8: the nuclear family is not closed — each of its adults has a
    # family-of-origin tie.
    adults = instance.nuclear_adults()
    if not adults:
        raise ConfigError(f"{source}: no adult in nuclear household {meta['nuclear_household']!r}")
    for adult in adults:
        if not instance.family_of_origin_ties(adult):
            raise ConfigError(
                f"{source}: {adult} has no family-of-origin tie; a closed nuclear family cannot "
                "compute its own driving term (M2.3, M1.D.8)"
            )
    return instance
