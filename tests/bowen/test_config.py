"""Config strictness and the constants register.

Purpose: prove the markdown config parser fails loudly, and the repository's
         constants register loads with every constant graded.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.3, #M10.B.2, #M10.1, #M0.1, #M10.B.4;
         docs/implementation_plan_version_2.md#S1.3a, #S1.3b
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
    "| `defence_threshold` | 15.0 | [I] | u | `M4.C.2` |\n"
    "| `witness_weight` | 0.5 | [I] | u | `M4.C.9` |\n"
    "| `speaker_echo_gain` | 0.2 | [I] | u | `M4.C.7` |\n"
    "| `reappraisal_window` | 4 | [I] | u | `M4.C.6` |\n"
    "| `attention_gain` | 0.5 | [I] | u | `M4.C.8` |\n"
    "| `perspective_anxiety_scale` | 10.0 | [I] | u | `M4.C.4` |\n"
    "| `calm_transfer_rate` | 0.05 | [I] | u | `M4.C.10` |\n"
    "| `symptom_leak_rate` | 0.1 | [I] | u | `M4.C.3a` |\n"
    "| `symptom_threshold_gain` | 1.5 | [I] | u | `M7.D.1` |\n"
    "| `symptom_rearm_fraction` | 0.5 | [I] | u | `M7.D.1` |\n"
    "| `symptom_event_intensity` | 100.0 | [I] | u | `M7.D.1` |\n"
    "| `reactive_rate` | 0.2 | [I] | u | `M1.A.19` |\n"
    "| `investment_leak_rate` | 0.1 | [I] | u | `M1.B.8` |\n"
    "| `initial_impingement_scale` | 0.8 | [I] | u | `M1.A.9` |\n"
    "| `outside_ness_rate` | 0.1 | [I] | u | `M1.A.9` |\n"
    "| `hollow_gain` | 0.5 | [I] | u | `M5.F.1` |\n"
    "| `assault_gain` | 0.5 | [I] | u | `M5.F.1` |\n"
    "| `outside_ness_threshold_outward` | 0.5 | [I] | u | `M5.C.1` |\n"
    "| `outside_ness_threshold_inward` | 0.5 | [I] | u | `M5.C.1` |\n"
    "| `involvement_membership_threshold` | 0.5 | [I] | u | `M1.A.12` |\n"
    "| `belief_rate` | 0.3 | [I] | u | `M9.8` |\n"
    "| `rejection_cutoff_gain` | 0.5 | [I] | u | `M1.B.3c` |\n"
    "| `rejection_refusal_gain` | 0.1 | [I] | u | `M1.B.3c` |\n"
    "| `rejection_relief` | 0.25 | [I] | u | `M1.B.3c` |\n"
    "| `refusal_cost_gain` | 1.0 | [I] | u | `M1.B.3c` |\n"
    "| `declared_rejection` | 1.0 | [I] | u | `M1.B.3c` |\n"
    "| `rejection_refusal_threshold` | 0.3 | [I] | u | `M1.B.3c` |\n"
    "| `rejection_fade` | 0.05 | [I] | u | `M1.B.3c` |\n"
    "| `estimator_window` | 3 | [I] | u | `M1.A.4k` |\n"
    "| `estimator_breadth` | 3 | [I] | u | `M1.A.4k` |\n"
    "| `estimator_sd_weight` | 1.0 | [I] | u | `M1.A.4k` |\n"
    "| `estimator_rate` | 0.5 | [I] | u | `M1.A.4k` |\n"
    "| `tie_wear_rate` | 0.05 | [I] | u | `M7.E.4` |\n"
    "| `wear_offset_per_act` | 0.1 | [I] | u | `M7.E.4` |\n"
    "| `perspective_channel_gain` | 0.5 | [I] | u | `M4.D.1a` |\n"
    "| `self_practice_gain` | 0.001 | [I] | u | `M1.E.7` |\n"
    "| `stressor_persistence` | 0.5 | [I] | u | `M1.F.6` |\n"
    "| `response_up_gain` | 0.5 | [I] | u | `M1.A.5e` |\n"
    "| `response_down_gain` | 0.5 | [I] | u | `M1.A.5e` |\n"
    "| `response_relaxation_rate` | 0.02 | [I] | u | `M1.A.5e` |\n"
    "| `stress_floor_below_basic` | 15.0 | [I] | u | `M1.A.5e` |\n"
    "| `practice_window` | 4 | [I] | u | `M1.E.7` |\n"
    "| `perspective_steer_gain` | 0.5 | [I] | u | `M4.D.1a` |\n"
    "| `system_adjust_gain` | 1.0 | [I] | u | `M1.E.7` |\n"
    "| `soothing_review_every` | 12 | [I] | u | `M4.D.6` |\n"
    "| `soothing_review_penalty` | 0.5 | [I] | u | `M4.D.6` |\n"
    "| `usual_level_rate` | 0.02 | [I] | u | `M1.A.5e` |\n"
    "| `constitution_strength` | 0.5 | [I] | u | `M1.A.11c` |\n"
    "| `channel_protection` | 1.0 | [I] | u | `M7.D.2c` |\n"
    "| `lock_in_gain` | 0.2 | [I] | u | `M7.D.2a` |\n"
    "| `lock_in_turn` | 3.0 | [I] | u | `M7.D.2d` |\n"
    "| `attribution_rate` | 0.05 | [I] | u | `M9.6` |\n"
    "| `focus_rate` | 0.05 | [I] | u | `M1.A.11c` |\n"
    "| `belief_appraisal_gain` | 1.0 | [I] | u | `M9.7` |\n"
    "| `marital_distance_fraction` | 0.25 | [I] | u | `M5.C.1` |\n"
    "| `pseudo_self_relaxation_rate` | 0.05 | [I] | u | `M6.I.4` |\n"
    "| `exchange_window` | 8 | [I] | u | `M6.I.4` |\n"
    "| `acting_age_years` | 10 | [I] | u | `M7.C.1f` |\n"
    "| `development_start_age_years` | 5 | [I] | u | `M7.E.1f` |\n"
    "| `development_end_age_years` | 20 | [I] | u | `M7.E.1f` |\n"
    "| `development_band` | 0.02 | [I] | u | `M7.E.1f` |\n"
    "| `development_rate` | 0.1 | [I] | u | `M7.E.1f` |\n"
    "| `development_band_projection` | 0.10 | [I] | u | `M7.E.1f` |\n"
    "| `development_band_support` | 0.05 | [I] | u | `M7.E.1f` |\n"
    "| `witnessed_gain` | 0.5 | [I] | u | `M7.B.1a` |\n"
    "| `launch_age_years` | 20 | [I] | u | `M7.C.1g` |\n"
    "| `working_age_years` | 16 | [I] | u | `M7.C.1k` |\n"
    "| `find_work_rate` | 0.1 | [I] | u | `M7.C.1k` |\n"
    "| `find_work_pressure_scale` | 10.0 | [I] | u | `M7.C.1k` |\n"
    "| `grief_scale` | 20.0 | [I] | u | `M6.3g` |\n"
    "| `grief_min_weeks` | 52 | [I] | u | `M6.3g` |\n"
    "| `grief_extra_weeks` | 104 | [I] | u | `M6.3g` |\n"
    "| `income_loss_intensity` | 15.0 | [I] | u | `M6.3h` |\n"
    "| `income_shortfall_share` | 0.2 | [I] | u | `M6.3h` |\n"
    "| `mortality_a` | 0.0000657 | [I] | u | `M7.C.1j` |\n"
    "| `mortality_b` | 0.085 | [I] | u | `M7.C.1j` |\n"
    "| `budget_per_undifferentiation` | 1.49 | [I] | u | `M7.C.1h` |\n"
    "| `coach_start_age_years` | 35 | [I] | u | `M2.A.0k` |\n"
    "| `coach_retirement_age_years` | 68 | [I] | u | `M2.A.0k` |\n"
    "| `new_tie_investment_share` | 0.2 | [I] | u | `M7.C.1a` |\n"
    "| `birth_focus_gain` | 0.1 | [I] | u | `M7.E.1c` |\n"
    "| `death_transfer_share` | 1.0 | [I] | u | `M6.3b` |\n"
    "| `reactive_cutoff_gain` | 0.1 | [I] | u | `M1.B.3d` |\n"
    "| `witnessed_cutoff_gain` | 0.05 | [I] | u | `M4.D.6f` |\n"
    "| `witness_window` | 13 | [I] | u | `M4.D.6f` |\n"
    "| `perceived_weakness_gain` | 0.01 | [I] | u | `M1.B.8a` |\n"
    "| `focus_steer_gain` | 1.0 | [I] | u | `M4.D.1g` |\n"
    "| `origin_cutoff_anxiety_gain` | 10.0 | [I] | u | `M1.D.8a` |\n"
    "| `origin_fusion_gain` | 0.5 | [I] | u | `M1.D.8a` |\n"
    "| `nodal_spike_intensity` | 1.0 | [I] | u | `M4.C.1c` |\n"
    "| `danger_window` | 26 | [I] | u | `M1.B.3e` |\n"
    "| `danger_threshold` | 0.25 | [I] | u | `M1.B.3e` |\n"
    "| `danger_gain` | 2.0 | [I] | u | `M1.B.3e` |\n"
    "| `focus_reinforce_gain` | 1.0 | [I] | u | `M1.B.8b` |\n"
    "| `force_anxiety_gain` | 2.0 | [I] | u | `M4.D.1h` |\n"
    "| `focus_compound_gain` | 2.0 | [I] | u | `M1.B.8b` |\n"
    "| `self_channel_exponent` | 2.0 | [I] | u | `M4.D.1a` |\n"
    "| `policy_temperature` | 1.0 | [I] | u | `M4.D.1` |\n"
    "| `anxiety_band_low` | 5.0 | [I] | u | `M4.D.3` |\n"
    "| `anxiety_band_high` | 15.0 | [I] | u | `M4.D.3` |\n"
    "| `capacity_level_per_layer` | 20.0 | [I] | u | `M4.D.3a` |\n"
    "| `competing_urge_gain` | 0.5 | [I] | u | `M4.D.1d` |\n"
    "| `withhold_investment_gain` | 0.2 | [I] | u | `M4.D.1b` |\n"
    "| `loaded_tie_threshold` | 0.2 | [I] | u | `M4.D.3b` |\n"
    "| `policy_intensity` | 100.0 | [I] | u | `M4.D.1` |\n"
    "| `learning_rate` | 0.2 | [I] | u | `M4.D.6` |\n"
    "| `credit_horizon` | 3 | [I] | u | `M4.D.6b` |\n"
    "| `credit_discount` | 0.7 | [I] | u | `M4.D.6` |\n"
    "| `cross_person_weight` | 0.5 | [I] | u | `M4.D.6e` |\n"
    "| `habituation_rate` | 0.7 | [I] | u | `M4.G.3` |\n"
    "| `habituation_window` | 8 | [I] | u | `M4.G.3` |\n"
    "| `prepare_ticks` | 4 | [I] | u | `M5.D.2a` |\n"
    "| `rehearsal_rate` | 0.05 | [I] | u | `M1.A.9` |\n"
    "| `assertion_perspective_threshold` | 0.3 | [I] | u | `M5.F.4` |\n"
    "| `anger_threshold` | 0.3 | [I] | u | `M5.D.4` |\n"
    "| `assertion_gain` | 0.5 | [I] | u | `M5.F.4` |\n"
    "| `assertion_evidence_gain` | 0.05 | [I] | u | `M5.F.2a` |\n"
    "| `opposition_window` | 3 | [I] | u | `M5.E.3` |\n"
    "| `hold_gain` | 0.5 | [I] | u | `M5.D.3` |\n"
    "| `stall_limit` | 4 | [I] | u | `M5.D.4` |\n"
    "| `hold_window` | 6 | [I] | u | `M5.D.2` |\n"
    "| `pull_up_rate` | 0.2 | [I] | u | `M5.D.5` |\n"
    "| `exchange_gain` | 0.5 | [I] | u | `M5.D.7a` |\n"
    "| `triangle_floor_decrement` | 0.1 | [I] | u | `M1.C.5` |\n"
    "| `respect_gain` | 0.2 | [I] | u | `M5.E.8` |\n"
    "| `debit_gain` | 0.3 | [I] | u | `M5.E.7` |\n"
    "| `session_interval_weeks` | 8 | [I] | u | `M1.E.8` |\n"
    "| `landing_rate` | 0.15 | [I] | u | `M1.E.7` |\n"
    "| `delayed_view_bonus` | 1.0 | [I] | u | `M1.E.7c` |\n"
    "| `binder_failure_fraction` | 0.8 | [I] | u | `M1.E.7` |\n"
    "| `contact_optimum` | 4 | [I] | u | `M1.E.8` |\n"
    "| `contact_window` | 26 | [I] | u | `M1.E.8` |\n"
    "| `perspective_gain` | 0.1 | [I] | u | `M1.E.7` |\n"
    "| `delayed_view_weeks` | 26 | [I] | u | `M16.D.2` |\n"
    "| `contact_excess_exponent` | 2.0 | [I] | u | `M1.E.8` |\n"
    "| `sink_window` | 8 | [I] | u | `M1.D.1` |\n"
    "| `sink_rate` | 0.1 | [I] | u | `M1.D.1` |\n"
    "| `exchange_budget_reduction` | 2.0 | [I] | u | `M6.I.1` |\n"
    "| `pattern_window` | 4 | [I] | u | `M5.A.1a` |\n"
    "| `pattern_min_acts` | 3 | [I] | u | `M5.A.1a` |\n"
    "| `pattern_fixed_activations` | 3 | [I] | u | `M5.A.1a` |\n"
    "| `pattern_projection_hold` | 12 | [I] | u | `M5.A.1a` |\n"
    "| `ensemble_block` | 50 | [I] | u | `M17.A.1` |\n"
    "| `ensemble_cap` | 500 | [I] | u | `M17.A.1` |\n"
    "| `ensemble_precision` | 0.25 | [I] | u | `M17.A.1` |\n"
    "| `ensemble_margin` | 0.1 | [I] | u | `M17.A.4` |\n"
    "| `ensemble_alpha` | 0.05 | [I] | u | `M11.4e` |\n"
    "| `fallback_flag_rate` | 0.2 | [I] | u | `M11.D.18` |\n"
    "| `equivalence_margin` | 0.5 | [I] | u | `M11.4a` |\n"
    "| `distance_binding_rate` | 0.3 | [I] | u | `M1.D.2a` |\n"
    "| `triangle_transfer_rate` | 0.3 | [I] | u | `M1.C.1` |\n"
    "| `outsider_positional_gain` | 0.5 | [I] | u | `M1.C.1` |\n"
    "| `balance_push_gain` | 0.2 | [I] | u | `M1.B.5` |\n"
    "| `balance_settle_rate` | 0.05 | [I] | u | `M1.B.5` |\n"
    "| `balance_harden_rate` | 0.02 | [I] | u | `M1.B.6` |\n"
    "| `reversal_asymmetry` | 0.5 | [I] | u | `M1.B.7` |\n"
    "| `pseudo_self_transfer_gain` | 2.0 | [I] | u | `M6.I.4` |\n"
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


def test_s1_3a_act_classes_come_from_config():
    """S1.3a (rule 9): the engine's and the policy's act classes are the sets ``config/bowen/act_classes.md`` declares,
    read through the kinds (the engine reads no file, `M11.D.22`), and neither module holds a literal set of act names.
    The two policy classes are also the lists the spec's `M5.B.4` and `M5.B.5` name, so the file cannot drift from
    the spec."""
    import re
    from pathlib import Path

    from act_literals import literal_act_sets  # tests/bowen is on the path under pytest's rootdir
    from src.bowen.engine import external
    from src.bowen.io.load import load_act_classes, load_event_kinds
    from src.bowen.policy import policy

    kinds = load_event_kinds()
    classes = load_act_classes(kinds)
    assert {c: getattr(kinds, c) for c in classes} == classes and all(classes.values())
    for module in (external, policy):
        assert literal_act_sets(Path(module.__file__)) == [], module.__name__
    spec = Path(__file__).resolve().parents[2].joinpath("docs", "bowen_agent_model_spec_v2.md").read_text(encoding="utf-8")
    for requirement, name in (("M5.B.4", "family_to_external"), ("M5.B.5", "external_repertoire")):
        line = next(l for l in spec.splitlines() if l.startswith(f"**{requirement}**"))
        assert set(re.findall(r"`([A-Z][A-Z_-]+)`", line)) & kinds.moves() == classes[name], requirement


def test_s1_3b_an_undeclared_act_fails_to_load(tmp_path):
    """S1.3b (`M10.B.2`): a misspelt act in the act classes fails at load, as does a class missing or listed twice, a
    class with no acts, and a side-taking act that is not a family-to-external move (review 2026-10-19)."""
    from src.bowen.io.load import CONFIG_DIR, load_act_classes, load_event_kinds

    kinds = load_event_kinds()
    text = (CONFIG_DIR / "act_classes.md").read_text(encoding="utf-8")
    broken = {
        "misspelt": (text.replace("| side_taking | SPLIT |", "| side_taking | SPILT |"), "not a declared move"),
        "missing": ("\n".join(l for l in text.splitlines() if not l.startswith("| side_taking |")), "exactly"),
        "twice": (text + "| side_taking | SPLIT | again |\n", "exactly"),
        "empty": (text.replace("| side_taking | SPLIT |", "| side_taking |  |"), "no acts"),
        "off the side": (text.replace("| side_taking | SPLIT |", "| side_taking | PURSUE |"), "family-to-external"),
    }
    assert all(b != text for b, _ in broken.values())
    for name, (body, message) in broken.items():
        path = tmp_path / f"{name}.md"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ConfigError, match=message):
            load_act_classes(kinds, path)


def test_s1_3a_kinds_without_act_classes_fail_where_a_class_is_read():
    """Kinds built without the act classes carry None, not an empty set, so the policy fails where it reads a class
    instead of treating every act as in none (review 2026-10-19)."""
    from src.bowen.engine.events import ACT_CLASSES, EventKinds
    from src.bowen.io.load import load_event_kinds

    kinds = load_event_kinds()
    assert all(getattr(EventKinds(kinds.mechanisms), c) is None for c in ACT_CLASSES)
    assert all(getattr(kinds, c) for c in ACT_CLASSES)  # loaded, each class has acts

