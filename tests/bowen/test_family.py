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
from src.bowen.engine.objects import Sex, TieState
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


def test_m2a_values_match_the_spec_table():
    """Every value M2.A states for these seven people is the value the config declares."""
    rows = {}
    for line in SPEC.splitlines():
        m = re.match(r"^\| \d+ \| (\w+) \| (\d) \| (\d+) \| (\w+) \| (\d+) \| (\d+) \| ([^|]+) \| \**(yes|no)\** \|", line)
        if m:
            rows[m.group(1).lower()] = m.groups()
    instance = load_family()
    for pid, person in instance.people.items():
        _, gen, age, sex, basic, chronic, _sib, dependent = rows[pid.value]
        assert instance.generations[pid] == int(gen)
        assert instance.ages_years[pid] == int(age)
        assert person.sex is Sex(sex)
        assert person.basic_level == float(basic)
        assert person.chronic_anxiety == float(chronic)
        assert person.financially_dependent is (dependent == "yes")


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
    assert tie.tie_state is TieState.CUT_OFF and not tie.interactive
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


def test_m1b3_cut_off_tie_cannot_be_interactive():
    with pytest.raises(ConfigError, match="cut-off"):
        build(edit("| 0.8 | 55 | 1 | cut_off | no |", "| 0.8 | 55 | 1 | cut_off | yes |"))


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
