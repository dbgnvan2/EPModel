"""Config strictness and the constants register.

Purpose: prove the markdown config parser fails loudly, and the repository's
         constants register loads with every constant graded.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.3, #M10.B.2, #M10.1, #M0.1, #M10.B.4
Tests:   this file
"""

from __future__ import annotations

import datetime as dt

import pytest

from src.bowen.io.load import load_constants
from src.bowen.scenario.config_parse import ConfigError
from src.bowen.scenario.constants import GRADES, SCHEMA, parse_constants

HEADER = "| key | value | grade | unit | spec |\n|---|---|---|---|---|\n"
GOOD_ROWS = (
    "| `fast_tick_weeks` | 1 | [I] | week | `M3.A.1` |\n"
    "| `slow_tick_fast_ticks` | 52 | [I] | fast tick | `M3.B.1` |\n"
    "| `invariant_tolerance` | 1e-9 | [I] | anxiety unit | `M6.1` |\n"
    "| `spouse_basic_level_tolerance` | 1.0 | [I] | point | `M2.A.0e` |\n"
    "| `chronic_anxiety_fixation_age_years` | 12 | [I] | year | `M2.A.0a` |\n"
    "| `per_hop_fidelity` | 0.8 | [I] | fraction | `M1.F.4` |\n"
    "| `standing_load_gain` | 0.2 | [I] | u | `M4.A.1` |\n"
    "| `appraisal_gain` | 3.0 | [I] | u | `M4.C.1` |\n"
    "| `intensity_scale` | 100.0 | [I] | u | `M4.C.1` |\n"
    "| `contact_band_max` | 0.2 | [I] | u | `M4.C.1a` |\n"
    "| `anxiety_togetherness_gain` | 0.5 | [I] | u | `M4.C.1b` |\n"
    "| `interactive_resting_contact` | 0.5 | [I] | u | `M4.C.1c` |\n"
    "| `contact_relaxation_rate` | 0.1 | [I] | u | `M4.C.1c` |\n"
    "| `impingement_relaxation_rate` | 0.3 | [I] | u | `M4.C.1` |\n"
    "| `functional_level_floor` | 1.0 | [I] | u | `M4.C.1` |\n"
    "| `acute_decay_rate` | 0.2 | [I] | u | `M1.A.8` |\n"
    "| `route_damping` | 0.5 | [I] | u | `M1.F.3` |\n"
    "| `hardening_run_length` | 3 | [I] | u | `M4.G.1` |\n"
    "| `bond_energy_decay_rate` | 0.0 | [I] | u | `M1.B.4` |\n"
    "| `triangle_activity_window` | 4 | [I] | u | `M1.C.3` |\n"
    "| `involvement_membership_threshold` | 0.5 | [I] | u | `M1.A.12` |\n"
)


def doc(rows: str = GOOD_ROWS, *, before: str = "frozen_at: unset\nactivation_regime: synchronous\n\n", after: str = "") -> str:
    return "# constants\n\n> a note\n\n" + before + HEADER + rows + after


def test_m11d3_good_document_parses():
    constants = parse_constants(doc())
    assert constants["slow_tick_fast_ticks"] == 52
    assert isinstance(constants["fast_tick_weeks"], int)
    assert constants.frozen_at is None


def test_m11d3_config_rejects_unknown_key():
    rows = GOOD_ROWS + "| `softmax_temperature` | 1.0 | [I] | — | `M4.D.1` |\n"
    with pytest.raises(ConfigError, match="unknown key 'softmax_temperature'"):
        parse_constants(doc(rows))


@pytest.mark.parametrize(
    "stray",
    [
        "SLOW_TICK = 52\n",          # an assignment, not a metadata line
        "just some prose\n",
        "| broken row with no end\n",
    ],
)
def test_m11d3_config_rejects_malformed_line(stray):
    with pytest.raises(ConfigError):
        parse_constants(doc(after=stray))


def test_m11d3_config_rejects_missing_required_key():
    rows = "".join(GOOD_ROWS.splitlines(keepends=True)[:2])
    with pytest.raises(ConfigError, match="missing keys \\[.*'invariant_tolerance'"):
        parse_constants(doc(rows))


def test_m11d3_config_rejects_duplicate_key():
    rows = GOOD_ROWS + "| `fast_tick_weeks` | 1 | [I] | week | `M3.A.1` |\n"
    with pytest.raises(ConfigError, match="duplicate key"):
        parse_constants(doc(rows))


def test_m11d3_config_rejects_wrong_cell_count():
    rows = GOOD_ROWS + "| `fast_tick_weeks` | 1 | [I] |\n"
    with pytest.raises(ConfigError, match="cells"):
        parse_constants(doc(rows))


def test_m11d3_config_rejects_non_numeric_value():
    rows = GOOD_ROWS.replace("| 52 |", "| fifty-two |")
    with pytest.raises(ConfigError, match="not a valid int"):
        parse_constants(doc(rows))


def test_m11d3_config_rejects_float_for_integer_key():
    rows = GOOD_ROWS.replace("| 52 |", "| 52.5 |")
    with pytest.raises(ConfigError, match="not a valid int"):
        parse_constants(doc(rows))


def test_m11d3_config_rejects_wrong_table_header():
    bad = doc().replace("| key | value | grade | unit | spec |", "| key | value |")
    with pytest.raises(ConfigError, match="header"):
        parse_constants(bad)


def test_m11d3_config_rejects_unknown_metadata_key():
    with pytest.raises(ConfigError, match="unknown metadata key 'seed'"):
        parse_constants(doc(before="frozen_at: unset\nactivation_regime: synchronous\nseed: 7\n\n"))


def test_m11d3_config_rejects_missing_frozen_at():
    with pytest.raises(ConfigError, match="missing metadata keys"):
        parse_constants(doc(before=""))


def test_m0_1_grade_must_match_the_spec():
    """An invented constant relabelled as textual in config must not load (M0.1, M0.2)."""
    rows = GOOD_ROWS.replace("| 1 | [I] | week |", "| 1 | [T] | week |")
    with pytest.raises(ConfigError, match="does not match the spec's \\[I\\]"):
        parse_constants(doc(rows))


def test_m0_1_unknown_grade_rejected():
    rows = GOOD_ROWS.replace("| 1 | [I] | week |", "| 1 | invented | week |")
    with pytest.raises(ConfigError, match="is not one of"):
        parse_constants(doc(rows))


def test_m10b4_frozen_at_parses_a_date_and_rejects_anything_else():
    assert parse_constants(doc(before="frozen_at: 2026-10-06\nactivation_regime: synchronous\n\n")).frozen_at == dt.date(2026, 10, 6)
    with pytest.raises(ConfigError, match="frozen_at"):
        parse_constants(doc(before="frozen_at: soon\nactivation_regime: synchronous\n\n"))


def test_m101_repository_constants_load_and_are_graded():
    """The committed register parses, carries every schema key, and every grade is legal."""
    constants = load_constants()
    assert set(constants.values) == set(SCHEMA)
    assert all(c.grade in GRADES for c in constants.values.values())


def test_m3a1_fast_tick_is_one_week_and_graded_invented():
    constants = load_constants()
    assert constants["fast_tick_weeks"] == 1
    assert constants.values["fast_tick_weeks"].grade == "[I]"


def test_m3b1_slow_tick_is_52_fast_ticks():
    assert load_constants()["slow_tick_fast_ticks"] == 52


def test_m3e2_activation_regime_is_declared_and_checked():
    assert load_constants().activation_regime == "synchronous"
    with pytest.raises(ConfigError, match="activation_regime"):
        parse_constants(doc(before="frozen_at: unset\nactivation_regime: random_sequential\n\n"))


def test_m10b4_a_retired_key_is_rejected_in_the_live_register():
    """A key retired since the freeze may appear only in a frozen snapshot (M10.B.4)."""
    with pytest.raises(ConfigError, match="unknown key"):
        parse_constants(doc(GOOD_ROWS + "| `tension_activation_threshold` | 5.0 | [I] | u | `M1.C.3` |\n"))
