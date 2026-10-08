"""The constants register: every numeric constant, its value and its grade.

Purpose: load the markdown constants register into validated, graded values.
Spec:    docs/bowen_agent_model_spec_v2.md#M10.1, #M10.B.1, #M10.B.4, #M0.1
Tests:   tests/bowen/test_config.py

Values live only in ``config/bowen/constants.md`` (M10.B.1). This module holds
the *schema* — which keys exist, their type, and the grade the spec assigns —
so that a key the engine does not know raises (M10.B.2) and a constant cannot
be relabelled as sourced in config while the spec calls it invented (M0.1,
M0.2). Keys are added here only together with the mechanism that reads them.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.bowen.scenario.config_parse import ConfigError, parse_table_document

COLUMNS = ("key", "value", "grade", "unit", "spec")
GRADES = frozenset({"[T]", "[M]", "[D]", "[#]", "[I]", "[X]"})
UNFROZEN = "unset"
# M3.E.2: the activation regime is a modelling choice, graded [I] by the spec.
ACTIVATION_REGIMES = frozenset({"synchronous"})
ACTIVATION_REGIME_GRADE = "[I]"


@dataclass(frozen=True)
class KeySpec:
    """What the schema requires of one constant."""

    kind: type
    grade: str
    spec: str


# Every key the engine reads. The grade is the spec's, not a choice made here:
# M10.C.1 lists the fast tick length and the chronic-anxiety fixation age as
# invented, M6.1 requires the invariant tolerance to be [I], and M2.A.0e's
# spouse tolerance is a project decision (user amendment to D3, 2026-08-24).
SCHEMA: Mapping[str, KeySpec] = MappingProxyType(
    {
        "fast_tick_weeks": KeySpec(int, "[I]", "M3.A.1"),
        "slow_tick_fast_ticks": KeySpec(int, "[I]", "M3.B.1"),
        "invariant_tolerance": KeySpec(float, "[I]", "M6.1"),
        "spouse_basic_level_tolerance": KeySpec(float, "[I]", "M2.A.0e"),
        "chronic_anxiety_fixation_age_years": KeySpec(int, "[I]", "M2.A.0a"),
        "per_hop_fidelity": KeySpec(float, "[I]", "M1.F.4"),
        "standing_load_gain": KeySpec(float, "[I]", "M4.A.1"),
        "appraisal_gain": KeySpec(float, "[I]", "M4.C.1"),
        "intensity_scale": KeySpec(float, "[I]", "M4.C.1"),
        "contact_band_max": KeySpec(float, "[I]", "M4.C.1a"),
        "anxiety_togetherness_gain": KeySpec(float, "[I]", "M4.C.1b"),
        "interactive_resting_contact": KeySpec(float, "[I]", "M4.C.1c"),
        "contact_relaxation_rate": KeySpec(float, "[I]", "M4.C.1c"),
        "impingement_relaxation_rate": KeySpec(float, "[I]", "M4.C.1"),
        "functional_level_floor": KeySpec(float, "[I]", "M4.C.1"),
        "acute_decay_rate": KeySpec(float, "[I]", "M1.A.8"),
        "route_damping": KeySpec(float, "[I]", "M1.F.3"),
        "hardening_run_length": KeySpec(int, "[I]", "M4.G.1"),
        "bond_energy_decay_rate": KeySpec(float, "[I]", "M1.B.4"),
        "triangle_activity_window": KeySpec(int, "[I]", "M1.C.3"),
        "defence_threshold": KeySpec(float, "[I]", "M4.C.2"),
        "witness_weight": KeySpec(float, "[I]", "M4.C.9"),
        "speaker_echo_gain": KeySpec(float, "[I]", "M4.C.7"),
        "reappraisal_window": KeySpec(int, "[I]", "M4.C.6"),
        "attention_gain": KeySpec(float, "[I]", "M4.C.8"),
        "perspective_anxiety_scale": KeySpec(float, "[I]", "M4.C.4"),
        "calm_transfer_rate": KeySpec(float, "[I]", "M4.C.10"),
        "symptom_leak_rate": KeySpec(float, "[I]", "M4.C.3a"),
        "symptom_threshold_gain": KeySpec(float, "[I]", "M7.D.1"),
        "symptom_rearm_fraction": KeySpec(float, "[I]", "M7.D.1"),
        "symptom_event_intensity": KeySpec(float, "[I]", "M7.D.1"),
        "reactive_rate": KeySpec(float, "[I]", "M1.A.19"),
        "investment_leak_rate": KeySpec(float, "[I]", "M1.B.8"),
        "initial_impingement_scale": KeySpec(float, "[I]", "M1.A.9"),
        "outside_ness_rate": KeySpec(float, "[I]", "M1.A.9"),
        "hollow_gain": KeySpec(float, "[I]", "M5.F.1"),
        "assault_gain": KeySpec(float, "[I]", "M5.F.1"),
        "outside_ness_threshold_outward": KeySpec(float, "[I]", "M5.C.1"),
        "outside_ness_threshold_inward": KeySpec(float, "[I]", "M5.C.1"),
        "involvement_membership_threshold": KeySpec(float, "[I]", "M1.A.12"),
        "belief_rate": KeySpec(float, "[I]", "M9.8"),
        "contact_excess_exponent": KeySpec(float, "[I]", "M1.E.8"),
        "ensemble_block": KeySpec(int, "[I]", "M17.A.1"),
        "ensemble_cap": KeySpec(int, "[I]", "M17.A.1"),
        "ensemble_precision": KeySpec(float, "[I]", "M17.A.1"),
        "ensemble_margin": KeySpec(float, "[I]", "M17.A.4"),
        "ensemble_alpha": KeySpec(float, "[I]", "M11.4e"),
        "fallback_flag_rate": KeySpec(float, "[I]", "M11.D.18"),
        "equivalence_margin": KeySpec(float, "[I]", "M11.4a"),
        "pattern_window": KeySpec(int, "[I]", "M5.A.1a"),
        "pattern_min_acts": KeySpec(int, "[I]", "M5.A.1a"),
        "pattern_pole": KeySpec(float, "[I]", "M5.A.1a"),
        "pattern_fixed_activations": KeySpec(int, "[I]", "M5.A.1a"),
        "sink_window": KeySpec(int, "[I]", "M1.D.1"),
        "sink_rate": KeySpec(float, "[I]", "M1.D.1"),
        "exchange_budget_reduction": KeySpec(float, "[I]", "M6.I.1"),
        "session_interval_weeks": KeySpec(int, "[I]", "M1.E.8"),
        "landing_rate": KeySpec(float, "[I]", "M1.E.7"),
        "delayed_view_bonus": KeySpec(float, "[I]", "M1.E.7c"),
        "binder_failure_fraction": KeySpec(float, "[I]", "M1.E.7"),
        "contact_optimum": KeySpec(int, "[I]", "M1.E.8"),
        "contact_window": KeySpec(int, "[I]", "M1.E.8"),
        "perspective_gain": KeySpec(float, "[I]", "M1.E.7"),
        "delayed_view_weeks": KeySpec(int, "[I]", "M16.D.2"),
        "prepare_ticks": KeySpec(int, "[I]", "M5.D.2a"),
        "rehearsal_rate": KeySpec(float, "[I]", "M1.A.9"),
        "assertion_perspective_threshold": KeySpec(float, "[I]", "M5.F.4"),
        "anger_threshold": KeySpec(float, "[I]", "M5.D.4"),
        "assertion_gain": KeySpec(float, "[I]", "M5.F.4"),
        "assertion_evidence_gain": KeySpec(float, "[I]", "M5.F.2a"),
        "opposition_window": KeySpec(int, "[I]", "M5.E.3"),
        "hold_gain": KeySpec(float, "[I]", "M5.D.3"),
        "stall_limit": KeySpec(int, "[I]", "M5.D.4"),
        "hold_window": KeySpec(int, "[I]", "M5.D.2"),
        "pull_up_rate": KeySpec(float, "[I]", "M5.D.5"),
        "exchange_gain": KeySpec(float, "[I]", "M5.D.7a"),
        "triangle_floor_decrement": KeySpec(float, "[I]", "M1.C.5"),
        "respect_gain": KeySpec(float, "[I]", "M5.E.8"),
        "debit_gain": KeySpec(float, "[I]", "M5.E.7"),
        "learning_rate": KeySpec(float, "[I]", "M4.D.6"),
        "credit_horizon": KeySpec(int, "[I]", "M4.D.6b"),
        "credit_discount": KeySpec(float, "[I]", "M4.D.6"),
        "cross_person_weight": KeySpec(float, "[I]", "M4.D.6e"),
        "habituation_rate": KeySpec(float, "[I]", "M4.G.3"),
        "habituation_window": KeySpec(int, "[I]", "M4.G.3"),
        "policy_intensity": KeySpec(float, "[I]", "M4.D.1"),
        "self_channel_exponent": KeySpec(float, "[I]", "M4.D.1a"),
        "policy_temperature": KeySpec(float, "[I]", "M4.D.1"),
        "anxiety_band_low": KeySpec(float, "[I]", "M4.D.3"),
        "anxiety_band_high": KeySpec(float, "[I]", "M4.D.3"),
        "capacity_level_per_layer": KeySpec(float, "[I]", "M4.D.3a"),
        "competing_urge_gain": KeySpec(float, "[I]", "M4.D.1d"),
        "withhold_investment_gain": KeySpec(float, "[I]", "M4.D.1b"),
        "loaded_tie_threshold": KeySpec(float, "[I]", "M4.D.3b"),
        "distance_binding_rate": KeySpec(float, "[I]", "M1.D.2a"),
        "triangle_transfer_rate": KeySpec(float, "[I]", "M1.C.1"),
        "outsider_positional_gain": KeySpec(float, "[I]", "M1.C.1"),
        "balance_push_gain": KeySpec(float, "[I]", "M1.B.5"),
        "balance_settle_rate": KeySpec(float, "[I]", "M1.B.5"),
        "balance_harden_rate": KeySpec(float, "[I]", "M1.B.6"),
        "reversal_asymmetry": KeySpec(float, "[I]", "M1.B.7"),
        "pseudo_self_transfer_gain": KeySpec(float, "[I]", "M6.I.4"),
    }
)

# Keys a frozen snapshot may still hold after the mechanism that read them was
# replaced. A retired key is accepted only when parsing a snapshot, never in the
# live register, and its removal is logged in constants_changes.md (M10.B.4).
# Retired at Phase C step 1 by spec revision 11 (M4.C.1c replaced the standing
# load's interactive fraction; amended M1.C.3 forbade the tension threshold).
RETIRED: Mapping[str, KeySpec] = MappingProxyType(
    {
        "interactive_standing_fraction": KeySpec(float, "[I]", "M4.A.3"),
        "tension_activation_threshold": KeySpec(float, "[I]", "M1.C.3"),
    }
)


@dataclass(frozen=True)
class Constant:
    value: int | float
    grade: str
    unit: str
    spec: str


@dataclass(frozen=True)
class Constants:
    """The resolved register. ``frozen_at`` is None until the values are frozen (M10.B.4)."""

    values: Mapping[str, Constant]
    frozen_at: dt.date | None
    activation_regime: str = "synchronous"

    def __getitem__(self, key: str) -> int | float:
        return self.values[key].value


def _parse_value(raw: str, kind: type, where: str) -> int | float:
    try:
        if kind is int:
            if not raw.lstrip("-").isdigit():
                raise ValueError
            return int(raw)
        return float(raw)
    except ValueError:
        raise ConfigError(f"{where}: value {raw!r} is not a valid {kind.__name__}") from None


def _parse_frozen_at(raw: str, source: str) -> dt.date | None:
    if raw == UNFROZEN:
        return None
    try:
        return dt.date.fromisoformat(raw)
    except ValueError:
        raise ConfigError(
            f"{source}: frozen_at must be an ISO date or {UNFROZEN!r}, got {raw!r}"
        ) from None


def parse_constants(
    text: str,
    *,
    source: str = "<constants>",
    schema: Mapping[str, KeySpec] = SCHEMA,
    require_all: bool = True,
) -> Constants:
    """Purpose: parse the constants register, rejecting unknown, missing or mislabelled keys.
    Spec:    docs/bowen_agent_model_spec_v2.md#M10.B.2, #M10.1, #M0.1, #M10.B.4
    Tests:   tests/bowen/test_config.py::test_m11d3_config_rejects_unknown_key
    """
    document = parse_table_document(
        text, columns=COLUMNS, metadata_keys=frozenset({"frozen_at", "activation_regime"}), source=source
    )
    values: dict[str, Constant] = {}
    for row, line in zip(document.rows, document.row_lines):
        where = f"{source}:{line}"
        key = row["key"].strip("`")
        if key not in schema:
            raise ConfigError(f"{where}: unknown key {key!r}")
        if key in values:
            raise ConfigError(f"{where}: duplicate key {key!r}")
        expected = schema[key]
        grade = row["grade"]
        if grade not in GRADES:
            raise ConfigError(f"{where}: {key}: grade {grade!r} is not one of {sorted(GRADES)}")
        if grade != expected.grade:
            raise ConfigError(
                f"{where}: {key}: grade {grade} does not match the spec's {expected.grade}"
            )
        spec = row["spec"].strip("`")
        if spec != expected.spec:
            raise ConfigError(f"{where}: {key}: spec {spec!r} does not match {expected.spec!r}")
        if not row["unit"]:
            raise ConfigError(f"{where}: {key}: unit is empty")
        values[key] = Constant(
            value=_parse_value(row["value"], expected.kind, where),
            grade=grade,
            unit=row["unit"],
            spec=spec,
        )
    missing = schema.keys() - values.keys()
    if missing and require_all:
        raise ConfigError(f"{source}: missing keys {sorted(missing)}")
    regime = document.metadata["activation_regime"]
    if regime not in ACTIVATION_REGIMES:
        raise ConfigError(f"{source}: activation_regime {regime!r} is not one of {sorted(ACTIVATION_REGIMES)}")
    return Constants(
        values=MappingProxyType(values),
        frozen_at=_parse_frozen_at(document.metadata["frozen_at"], source),
        activation_regime=regime,
    )
