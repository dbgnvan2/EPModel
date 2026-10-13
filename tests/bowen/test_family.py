"""The Phase B family, built from markdown.

Purpose: prove the reduced instance is declared in config, matches the spec,
         is not closed, and enforces the pairing and life-stage rules.
Spec:    docs/bowen_agent_model_spec_v2.md#M2.1, #M2.3, #M2.3a, #M2.A, #M2.A.0a, #M2.A.0c, #M2.A.0e, #M2.B.1, #M1.D.8
Tests:   this file
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.objects import Role, Sex, TieState
from src.bowen.io.load import CONFIG_DIR, load_constants, load_family
from src.bowen.scenario.config_parse import ConfigError
from src.bowen.scenario.family import build_family

REPO = Path(__file__).resolve().parents[2]
SPEC = (REPO / "docs" / "bowen_agent_model_spec_v2.md").read_text(encoding="utf-8")
FAMILY_TEXT = (CONFIG_DIR / "family_reduced.md").read_text(encoding="utf-8")
P = PersonId


def build(text: str):
    return build_family(text, load_constants())


def edit(old: str, new: str) -> str:
    assert FAMILY_TEXT.count(old) == 1, old
    return FAMILY_TEXT.replace(old, new)


# --- M2.1 / M2.3a -------------------------------------------------------------------


def test_m21_family_declared_in_markdown():
    """The family loads from config, and no person's name or id appears in Python source."""
    instance = load_family()
    assert instance.instance_id == "phase_b_reduced"
    names = {n.lower() for n in instance.display_names.values()} | {p.value for p in instance.people}
    offenders = []
    for path in sorted((REPO / "src" / "bowen").rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        offenders += [f"{path.name}: {n}" for n in names if re.search(rf"\b{re.escape(n)}\b", text)]
    assert offenders == []


def test_m23a_instance_matches_the_spec():
    """Members and ties are exactly M2.3a's."""
    m23a = re.search(r"^\*\*M2\.3a\*\*.*$", SPEC, re.M).group(0)
    members_text = re.search(r"Members: (.*?)\. Ties:", m23a).group(1)
    expected_people = {P(n.lower()) for n in re.findall(r"\b([A-Z][a-z]+)\b", members_text)}
    ties_text = re.search(r"Ties: (.*?)\. Ana–Bruno is the dormant", m23a).group(1)
    expected_ties = {TieId.of(P(a.lower()), P(b.lower())) for a, b in re.findall(r"([A-Z][a-z]+)–([A-Z][a-z]+)", ties_text)}
    instance = load_family()
    assert len(expected_people) == 7 and set(instance.people) == expected_people
    assert len(expected_ties) == 8 and set(instance.ties) == expected_ties


# Bruno left M2.A at revision 13 (owner decision D5, 2026-10-10); the closed phases' instances keep him.
LEFT_M2A_AT_REVISION_13 = {"bruno"}


def spec_rows():
    rows = {}
    for line in SPEC.splitlines():
        m = re.match(r"^\| \d+a? \| (\w+) \| (\d) \| (\d+) \| (\w+) \| ([\d.]+) \| ([\d.]+) \| ([^|]+) \| \**(yes|no)\** \|",
                     line)
        if m:
            rows[m.group(1).lower()] = m.groups()
    return rows


def assert_matches_spec_table(instance, exempt=frozenset(), *, values=True):
    """``values=False`` checks generation and sex only: the closed phases keep the values M2.A had before revision 28."""
    rows = spec_rows()
    assert {p.value for p in instance.people if p.value not in rows} == set(exempt)
    for pid, person in instance.people.items():
        if pid.value in exempt:
            continue
        _, gen, age, sex, basic, chronic, _sib, dependent = rows[pid.value]
        assert instance.generations[pid] == int(gen)
        assert person.sex is Sex(sex)
        if not values:
            continue
        assert instance.ages_years[pid] == int(age)
        assert person.sex is Sex(sex)
        assert person.basic_level == float(basic)
        assert person.chronic_anxiety == float(chronic)
        assert person.financially_dependent is (dependent == "yes")


def test_m2a_values_match_the_spec_table():
    """The Phase B people are M2.A's by generation and sex; Bruno is history. Their ages and values are the closed
    phase's record: M2.A's moved with the life course at revision 28 (M2.A.0j), and the Phase B instance did not."""
    assert_matches_spec_table(load_family(), LEFT_M2A_AT_REVISION_13, values=False)


PHASE_D = REPO / "config" / "bowen" / "family_phase_d.md"


def test_m2a_phase_d_family_matches_spec_table():
    """Revision 13: the owner's family — 4 grandparents, 2 parents, 2 parents' siblings, 3 children, the coach."""
    instance = load_family(PHASE_D)
    halim = {p for p, person in instance.people.items() if person.role is Role.EXTERNAL}
    assert_matches_spec_table(instance, {p.value for p in halim})
    by_generation = {g: sorted(p.value for p in instance.people if instance.generations[p] == g and p not in halim)
                     for g in (1, 2, 3)}
    assert by_generation == {1: ["ana", "sofia", "teodor", "viktor"], 2: ["dario", "iris", "marta", "ravi"],
                             3: ["leo", "nadia", "pia"]}
    assert len(halim) == 1


def test_m2a_ravi_sibling_starts_cut_off():
    instance = load_family(PHASE_D)
    tie = TieId.of(P("ravi"), P("dario"))
    assert instance.ties[tie].tie_state is TieState.CUT_OFF and instance.ties[tie].interactive  # M1.B.3c
    assert tie in instance.family_of_origin_ties(P("ravi"))


def test_m2a0i_leo_declared_male():
    assert load_family(PHASE_D).people[P("leo")].sex is Sex.MALE


def test_m23_reduced_instance_is_not_closed():
    instance = load_family()
    assert instance.nuclear_adults() == (P("marta"), P("ravi"))
    for adult in instance.nuclear_adults():
        assert instance.family_of_origin_ties(adult)
    closed = edit("| `ana` | `marta` | parent_of | 0.6 | 40 | 1 | ordinary | yes |\n", "")
    with pytest.raises(ConfigError, match="marta has no family-of-origin tie"):
        build(closed)


def test_m23a_dormant_family_of_origin_tie_is_ana_bruno():
    instance = load_family()
    tie = instance.ties[TieId.of(P("ana"), P("bruno"))]
    assert tie.tie_state is TieState.CUT_OFF and tie.interactive  # M1.B.3c: a cutoff is rejection, not severing
    assert TieId.of(P("ana"), P("bruno")) in instance.family_of_origin_ties(P("ana"))


def test_m2b1_only_declared_ties_are_instantiated():
    instance = load_family()
    assert len(instance.ties) == 8 < 7 * 6 // 2


# --- M2.A.0a, M2.A.0c, M2.A.0e --------------------------------------------------------


def test_m2a0e_spouses_within_tolerance():
    load_family()  # Ravi 39 / Marta 40 passes
    with pytest.raises(ConfigError, match="beyond the tolerance"):
        build(edit("| `ravi` | Ravi | 2 | 52 | male | ravi_marta | 39 | 39 |", "| `ravi` | Ravi | 2 | 52 | male | ravi_marta | 38 | 38 |"))


def test_m2a0c_pairing_uses_basic_not_functional_level():
    # Functional levels far apart, basic levels within tolerance: accepted.
    build(edit("| `ravi` | Ravi | 2 | 52 | male | ravi_marta | 39 | 39 |", "| `ravi` | Ravi | 2 | 52 | male | ravi_marta | 39 | 55 |"))
    # Functional levels equal, basic levels apart: refused.
    with pytest.raises(ConfigError, match="basic_level"):
        build(edit("| `ravi` | Ravi | 2 | 52 | male | ravi_marta | 39 | 39 |", "| `ravi` | Ravi | 2 | 52 | male | ravi_marta | 37 | 40 |"))


def test_m2a0a_supplied_chronic_anxiety_only_past_fixation_age():
    with pytest.raises(ConfigError, match="below the fixation age"):
        build(edit("| `pia` | Pia | 3 | 14 |", "| `pia` | Pia | 3 | 11 |"))


def test_m1a21_sex_must_be_declared():
    with pytest.raises(ConfigError, match="M1.A.21"):
        build(edit("| `pia` | Pia | 3 | 14 | female |", "| `pia` | Pia | 3 | 14 | not stated |"))


def test_m1b3c_cut_off_tie_must_stay_interactive():
    with pytest.raises(ConfigError, match="cut-off"):
        build(edit("| 0.8 | 55 | 1 | cut_off | yes |", "| 0.8 | 55 | 1 | cut_off | no |"))


def test_m2_family_declaration_must_be_graded_invented():
    with pytest.raises(ConfigError, match="grade"):
        build(edit("grade: [I]", "grade: [T]"))


# --- M10.B.2 ----------------------------------------------------------------------------


def test_m10b2_family_sections_parse_strictly():
    with pytest.raises(ConfigError, match="unknown section"):
        build(FAMILY_TEXT + "\n## Pets\n\n| a |\n|---|\n")
    with pytest.raises(ConfigError, match="missing sections"):
        build(FAMILY_TEXT.split("## Ties")[0])
    with pytest.raises(ConfigError, match="duplicate tie"):
        build(FAMILY_TEXT.rstrip("\n") + "\n| `marta` | `ravi` | spouse | 1.0 | 60 | 1 | ordinary | yes |\n")
    with pytest.raises(ConfigError, match="undeclared person"):
        build(FAMILY_TEXT.rstrip("\n") + "\n| `ravi` | `teodor` | sibling | 1.0 | 60 | 1 | ordinary | yes |\n")
    with pytest.raises(ConfigError, match="inside a ## section"):
        build("| x |\n" + FAMILY_TEXT)
