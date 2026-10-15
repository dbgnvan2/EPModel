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
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.life_course import LifeSettings, PartnerRow, TieValues
from src.bowen.engine.objects import (
    REACTIVE_DETECTORS, Family, Person, Relationship, Role, Sex, SiblingPosition, SymptomChannel, TieState,
)
from src.bowen.scenario.config_parse import ConfigError, parse_sectioned_document
from src.bowen.scenario.constants import Constants

PEOPLE_COLUMNS = (
    "id", "name", "generation", "age", "sex", "household", "basic_level", "functional_level",
    "chronic_anxiety", "sibling_rank", "sibship_size", "financially_dependent", "role", "channel_prior",
)
TIE_COLUMNS = ("a", "b", "relation", "conductance", "bond_energy", "latency", "tie_state", "interactive")
METADATA = frozenset({"instance_id", "grade", "nuclear_household", "undifferentiation_budget"})
# Spec revision 29: principled cutoffs (M1.B.3d), as ``holder>other`` pairs, and a child's real weakness (M1.B.8a), as
# ``person=value`` pairs, each comma-separated, are declared only. Spec revision 30: a person's declared danger to others
# (M1.B.3e), and the origin cutoff of a person whose childhood is not simulated (M1.D.8a), as ``person=value`` pairs.
OPTIONAL_METADATA = frozenset({"principled_cutoffs", "weaknesses", "dangers"})
# Spec revision 28 (M2.A.0j, M2.A.0k, M7.C.1h): a family with a life course declares its starting ages as three
# settings, each person's gap to their generation's anchor, who earns, the coachee, the successor coach's sex, and the
# file of partner settings. Children below the fixation age carry no chronic anxiety, and children below the end of
# development no basic level: both are derived from their parents (M7.B.1a, M7.E.1f).
LIFE_PEOPLE_COLUMNS = (
    "id", "name", "generation", "age_gap", "sex", "household", "basic_level", "functional_level",
    "chronic_anxiety", "sibling_rank", "sibship_size", "financially_dependent", "earner", "role", "channel_prior",
)
LIFE_METADATA = METADATA | frozenset({"start_age_g1", "start_age_g2", "start_age_youngest_g3", "partners", "coachee",
                                      "successor_coach_sex"})
PARTNER_COLUMNS = (
    "id", "name", "partner_of", "pair_age", "age_gap", "sex", "chronic_anxiety", "sibling_rank", "sibship_size",
    "channel_prior", "origin_id", "origin_sex", "children_at_years", "children_sexes", "children_channels",
)
PARTNER_METADATA = frozenset({"instance_id", "grade", "spouse_conductance", "spouse_bond_energy", "parent_conductance",
                              "parent_bond_energy", "sibling_conductance", "sibling_bond_energy", "origin_conductance",
                              "origin_bond_energy", "tie_latency"})
_DRAW = "draw"
_YES_NO = {"yes": True, "no": False}
_NONE = {"—", "-", ""}


class Relation(enum.Enum):
    SPOUSE = "spouse"
    PARENT_OF = "parent_of"  # column a is the parent of column b
    SIBLING = "sibling"
    # M1.E.1: an external agent's real ties to the family; not a family-of-origin tie. Phase C step 9.
    PROFESSIONAL = "professional"


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
    life: LifeSettings | None = None  # spec revision 28: None for a family with no life course

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


def declares_life_course(text: str) -> bool:
    """A family with a life course declares its starting ages (spec revision 28, M2.A.0j)."""
    return re.search(r"^start_age_g1:", text, re.M) is not None


def partners_file(text: str) -> str | None:
    """The partner settings file a family with a life course names, relative to its own (M7.C.1h)."""
    match = re.search(r"^partners:\s*(\S+)\s*$", text, re.M)
    return match.group(1) if match else None


def _ages(rows, meta, generations, source: str) -> dict:
    """M2.A.0j: each member's age is their generation's anchor plus their declared gap."""
    anchors = {1: _number(meta["start_age_g1"], int, source, "start_age_g1"),
               2: _number(meta["start_age_g2"], int, source, "start_age_g2"),
               3: _number(meta["start_age_youngest_g3"], int, source, "start_age_youngest_g3")}
    gaps: dict[int, list[int]] = {}
    ages = {}
    for pid, raw, where in rows:
        generation = generations[pid]
        if generation is None:
            continue
        if generation not in anchors:
            raise ConfigError(f"{where}: {pid} is in generation {generation}; only 1, 2 and 3 have a starting age")
        try:
            gap = int(raw)
        except ValueError:
            raise ConfigError(f"{where}: age_gap {raw!r} is not a whole number") from None
        gaps.setdefault(generation, []).append(gap)
        ages[pid] = anchors[generation] + gap
    for generation, values in gaps.items():
        # Generation 3's anchor is its youngest member; generations 1 and 2 are anchored on their eldest.
        if (generation == len(anchors) and min(values) != 0) or (generation < len(anchors) and max(values) != 0):
            raise ConfigError(f"{source}: generation {generation}'s gaps {sorted(values)} do not put its anchor at 0 "
                              "(the youngest of generation 3, the eldest of 1 and 2; M2.A.0j)")
    return ages


def build_partners(text: str, constants: Constants, family_ids, *, source: str = "<partners>",
                   coach=None, coachee=None, successor_sex: Sex | None = None, ages=None) -> LifeSettings:
    """Purpose: parse the partner settings a family with a life course declares (M7.C.1h, M7.C.1i).
    Spec:    docs/bowen_agent_model_spec_v2.md#M7.C.1h, #M7.C.1i
    Tests:   tests/bowen/test_life_course.py::test_m7c1h_partner_comes_only_from_settings
    """
    document = parse_sectioned_document(text, sections={"Partners": PARTNER_COLUMNS}, metadata_keys=PARTNER_METADATA,
                                        source=source)
    meta = document.metadata
    if meta["grade"] != "[I]":
        raise ConfigError(f"{source}: partner settings are invented and must say grade: [I] (M2)")
    latency = _number(meta["tie_latency"], int, source, "tie_latency")

    def values(name: str) -> TieValues:
        return TieValues(_number(meta[f"{name}_conductance"], float, source, f"{name}_conductance"),
                         _number(meta[f"{name}_bond_energy"], float, source, f"{name}_bond_energy"), latency)

    sexes = {x.value: x for x in Sex}
    channels = {c.value: c for c in SymptomChannel}
    rows, seen = [], set(family_ids)
    table = document.sections["Partners"]
    for row, line in zip(table.rows, table.row_lines):
        where = f"{source}:{line}"
        pid, origin = PersonId(row["id"].strip("`")), PersonId(row["origin_id"].strip("`"))
        member = PersonId(row["partner_of"].strip("`"))
        for new in (pid, origin):
            if new in seen:
                raise ConfigError(f"{where}: {new} is already declared; a partner and their origin are new people")
            seen.add(new)
        if member not in family_ids:
            raise ConfigError(f"{where}: partner_of {member} is not in the family")
        if member in {r.partner_of for r in rows}:
            raise ConfigError(f"{where}: {member} already has a partner row; one per member")
        pair_age = _number(row["pair_age"], int, where, "pair_age")
        if ages is not None and pair_age <= ages[member]:
            raise ConfigError(f"{where}: {member} is {ages[member]} at t0, so a pairing at {pair_age} would never "
                              "happen (M7.C.1h)")
        try:
            age_gap = int(row["age_gap"])
        except ValueError:
            raise ConfigError(f"{where}: age_gap {row['age_gap']!r} is not a whole number") from None
        years = () if row["children_at_years"] in _NONE else tuple(
            _number(y.strip(), int, where, "children_at_years") for y in row["children_at_years"].split(";"))
        def per_child(raw: str, options, name: str):
            if not years:
                return ()
            items = [x.strip() for x in raw.split(";")]
            if len(items) != len(years):
                raise ConfigError(f"{where}: {name} names {len(items)} children; children_at_years names {len(years)}")
            return tuple(None if x == _DRAW else _choice(x, options, where, name) for x in items)
        if list(years) != sorted(set(years)):
            raise ConfigError(f"{where}: children_at_years must rise: {years}")
        rows.append(PartnerRow(
            id=pid, partner_of=member, pair_age=pair_age,
            age_gap=age_gap, sex=_choice(row["sex"], sexes, where, "sex"),
            chronic_anxiety=_number(row["chronic_anxiety"], float, where, "chronic_anxiety"),
            sibling_position=SiblingPosition(_number(row["sibling_rank"], int, where, "sibling_rank"),
                                             _number(row["sibship_size"], int, where, "sibship_size")),
            channel_prior=_choice(row["channel_prior"], channels, where, "channel_prior"),
            origin_id=origin, origin_sex=_choice(row["origin_sex"], sexes, where, "origin_sex"),
            children_at_years=years, children_sexes=per_child(row["children_sexes"], sexes, "children_sexes"),
            children_channels=per_child(row["children_channels"], channels, "children_channels"),
        ))
    return LifeSettings(partners=tuple(rows), spouse_tie=values("spouse"), parent_tie=values("parent"),
                        sibling_tie=values("sibling"), origin_tie=values("origin"), coach=coach, coachee=coachee,
                        successor_sex=successor_sex or Sex.FEMALE)


def _principled(declared: str, ties: dict, source: str) -> None:
    """M1.B.3d, revision 29: mark each declared principled cutoff on its tie; the tie must be declared cut off."""
    for entry in (e.strip() for e in declared.split(",") if e.strip() not in _NONE):
        holder, _, other = entry.partition(">")
        tie = ties.get(TieId.of(PersonId(holder.strip()), PersonId(other.strip()))) if other else None
        if tie is None or tie.tie_state is not TieState.CUT_OFF:
            raise ConfigError(f"{source}: principled cutoff {entry!r} names no cut-off tie (M1.B.3d)")
        tie.principled = tuple(sorted(set(tie.principled) | {PersonId(holder.strip())}))


def _per_person(declared: str, people: dict, source: str, field: str, spec: str) -> None:
    """A ``person=value`` list setting one 0-to-1 field on each person it names (M1.B.8a, M1.B.3e)."""
    for entry in (e.strip() for e in declared.split(",") if e.strip() not in _NONE):
        name, _, value = entry.partition("=")
        pid = PersonId(name.strip())
        try:
            number = float(value)
        except ValueError:
            raise ConfigError(f"{source}: {field} {entry!r} is not person=value ({spec})") from None
        if pid not in people or not 0.0 <= number <= 1.0:
            raise ConfigError(f"{source}: {field} {entry!r} names no person or is outside 0 to 1 ({spec})")
        setattr(people[pid], field, number)


def build_family(text: str, constants: Constants, *, source: str = "<family>",
                 partners_text: str | None = None, partners_source: str = "<partners>") -> FamilyInstance:
    """Purpose: parse and validate a family declaration into a FamilyInstance.
    Spec:    docs/bowen_agent_model_spec_v2.md#M2.1, #M2.3, #M2.A.0a, #M2.A.0e, #M2.A.0j, #M2.A.0k
    Tests:   tests/bowen/test_family.py::test_m21_family_declared_in_markdown
    """
    life = declares_life_course(text)
    document = parse_sectioned_document(
        text,
        sections={"People": LIFE_PEOPLE_COLUMNS if life else PEOPLE_COLUMNS, "Ties": TIE_COLUMNS},
        metadata_keys=LIFE_METADATA if life else METADATA,
        source=source,
        optional_keys=OPTIONAL_METADATA,
    )
    meta = document.metadata
    if meta["grade"] != "[I]":
        raise ConfigError(f"{source}: a family declaration is invented and must say grade: [I] (M2)")

    fixation_age = constants["chronic_anxiety_fixation_age_years"]
    sexes = {s.value: s for s in Sex}
    channels = {c.value: c for c in SymptomChannel}
    roles = {r.value: r for r in Role}

    people: dict[PersonId, Person] = {}
    gap_rows: list = []
    pending_children: dict = {}
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
        generations[pid] = None if row["generation"] in _NONE else _number(row["generation"], int, where, "generation")
        if life:
            age = None  # M2.A.0j: computed from the anchors below
            gap_rows.append((pid, row["age_gap"], where))
        else:
            age = _number(row["age"], int, where, "age")
        # M2.A.0a: the test is life stage, not generation.
        if age is not None and age < fixation_age:
            raise ConfigError(
                f"{where}: {pid} is {age}, below the fixation age {fixation_age}; a supplied "
                "chronic_anxiety is an error for an agent not yet past it (M2.A.0a)"
            )
        derived = row["chronic_anxiety"] in _NONE or row["basic_level"] in _NONE
        if derived and not life:
            raise ConfigError(f"{where}: only a family with a life course derives a child's values (M7.B.1a, M7.E.1f)")
        if derived:
            pending_children[pid] = (row["basic_level"] in _NONE, row["chronic_anxiety"] in _NONE, where)
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
                # A derived value starts at 0 and is set from the parents once the ties are read (revision 28).
                basic_level=0.0 if row["basic_level"] in _NONE else _number(row["basic_level"], float, where, "basic_level"),
                functional_level=0.0 if row["functional_level"] in _NONE else _number(
                    row["functional_level"], float, where, "functional_level"),
                chronic_anxiety=0.0 if row["chronic_anxiety"] in _NONE else _number(
                    row["chronic_anxiety"], float, where, "chronic_anxiety"),
                # Acute anxiety starts on its floor (M1.A.7a: chronic is the floor it decays toward).
                acute_anxiety=0.0 if row["chronic_anxiety"] in _NONE else _number(
                    row["chronic_anxiety"], float, where, "chronic_anxiety"),
                earner=life and _choice(row["earner"], _YES_NO, where, "earner"),
                sibling_position=sibling,
                financially_dependent=_choice(row["financially_dependent"], _YES_NO, where, "financially_dependent"),
                # M1.A.11b: the constitutional channel prior is declared data, [I].
                channel_prior=_choice(row["channel_prior"], channels, where, "channel_prior"),
                # Nobody starts with a systems perspective: M1.E.7 makes it rise only on a landed
                # external-agent contact. The reactive state's three detectors start at rest (M1.A.19).
                systems_perspective=0.0,
                reactive_state={name: 0.0 for name in REACTIVE_DETECTORS},
            )
        except ValueError as error:
            if isinstance(error, ConfigError):
                raise
            raise ConfigError(f"{where}: {error}") from None
        people[pid] = person
        names[pid] = row["name"]
        if age is not None:
            ages[pid] = age
    if life:
        ages.update(_ages(gap_rows, meta, generations, source))
        for pid in people:
            if generations[pid] is None:  # an external agent: the coach enters at the coach's starting age (M2.A.0k)
                if people[pid].role is not Role.EXTERNAL:
                    raise ConfigError(f"{source}: {pid} is a member with no generation, so no starting age (M2.A.0j)")
                ages[pid] = constants["coach_start_age_years"]
            people[pid].age_years = ages[pid]

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
        if state is TieState.CUT_OFF and not interactive:
            raise ConfigError(f"{where}: a cut-off tie is rejection, not severing, so it stays interactive (M1.B.3c)")
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

    _principled(meta.get("principled_cutoffs", ""), ties, source)
    _per_person(meta.get("weaknesses", ""), people, source, "weakness", "M1.B.8a")
    _per_person(meta.get("dangers", ""), people, source, "danger", "M1.B.3e")
    life_settings = None
    if life:
        life_settings = _life(meta, people, ties, ages, parent_on_tie, pending_children, constants, source,
                              partners_text, partners_source)
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
        life=life_settings,
    )
    instance.family.household = meta["nuclear_household"]
    # M2.3 / M1.D.8: the nuclear family is not closed — each of its adults has a
    # family-of-origin tie.
    adults = instance.nuclear_adults()
    nuclear = {p for p, person in people.items() if person.household_id == meta["nuclear_household"]}
    instance.family.marital_tie = next(
        (t for t, r in relations.items() if r is Relation.SPOUSE and set(t.members()) <= nuclear), None)
    instance.family.parent_child_ties = tuple(sorted(
        t for t, r in relations.items() if r is Relation.PARENT_OF and set(t.members()) <= nuclear))
    instance.family.origin_ties = tuple(sorted({t for a in adults for t in instance.family_of_origin_ties(a)}))
    if not adults:
        raise ConfigError(f"{source}: no adult in nuclear household {meta['nuclear_household']!r}")
    for adult in adults:
        if not instance.family_of_origin_ties(adult):
            raise ConfigError(
                f"{source}: {adult} has no family-of-origin tie; a closed nuclear family cannot "
                "compute its own driving term (M2.3, M1.D.8)"
            )
    return instance


def _life(meta, people, ties, ages, parent_on_tie, pending_children, constants, source, partners_text,
          partners_source) -> LifeSettings:
    """Spec revision 28: children's derived values (M2.A.0a, M7.B.1a, M7.E.1f), the coach (M2.A.0k), the partners."""
    fixation, development_end = constants["chronic_anxiety_fixation_age_years"], constants["development_end_age_years"]
    for pid in sorted(people):
        person, age = people[pid], ages[pid]
        derived_basic, derived_chronic, where = pending_children.get(pid, (False, False, source))
        if (age < fixation) != derived_chronic and person.role is Role.MEMBER:
            raise ConfigError(f"{where}: {pid} is {age}; chronic_anxiety is derived below the fixation age "
                              f"{fixation} and supplied from it (M2.A.0a, M7.B.1a)")
        if derived_basic and age >= development_end:
            raise ConfigError(f"{where}: {pid} is {age}; only a child under {development_end} derives its basic level "
                              "(M7.E.1f)")
        if not (derived_basic or derived_chronic):
            continue
        parents = sorted(p for t, p in parent_on_tie.items() if pid in t.members() and p != pid)
        if len(parents) != 2 or any(p in pending_children for p in parents):
            raise ConfigError(f"{where}: {pid} derives values from two declared parents; found {parents}")
        if derived_chronic:
            person.chronic_anxiety = person.acute_anxiety = math.fsum(people[p].chronic_anxiety for p in parents) / len(parents)
        if derived_basic:
            person.basic_level = person.functional_level = math.fsum(people[p].basic_level for p in parents) / len(parents)
    externals = sorted(p for p, person in people.items() if person.role is Role.EXTERNAL)
    coachee = PersonId(meta["coachee"])
    if len(externals) > 1:
        raise ConfigError(f"{source}: a family with a life course has one coach at a time (M2.A.0k): {externals}")
    for coach in externals:
        coach_ties = [t for t in ties if coach in t.members()]
        if coach_ties != [TieId.of(coach, coachee)]:
            raise ConfigError(f"{source}: the coach is tied to the coachee only (M2.B.2, owner 2026-10-13): {coach_ties}")
    if partners_text is None:
        raise ConfigError(f"{source}: a family with a life course names its partner settings (M7.C.1h)")
    sexes = {x.value: x for x in Sex}
    return build_partners(partners_text, constants, set(people), source=partners_source,
                          coach=externals[0] if externals else None, coachee=coachee,
                          successor_sex=_choice(meta["successor_coach_sex"], sexes, source, "successor_coach_sex"),
                          ages=ages)
