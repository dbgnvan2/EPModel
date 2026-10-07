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
        "interactive_standing_fraction": KeySpec(float, "[I]", "M4.A.3"),
        "functional_level_floor": KeySpec(float, "[I]", "M4.C.1"),
        "acute_decay_rate": KeySpec(float, "[I]", "M1.A.8"),
        "route_damping": KeySpec(float, "[I]", "M1.F.3"),
        "hardening_run_length": KeySpec(int, "[I]", "M4.G.1"),
        "bond_energy_decay_rate": KeySpec(float, "[I]", "M1.B.4"),
        "tension_activation_threshold": KeySpec(float, "[I]", "M1.C.3"),
        "involvement_membership_threshold": KeySpec(float, "[I]", "M1.A.12"),
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
    if missing:
        raise ConfigError(f"{source}: missing keys {sorted(missing)}")
    regime = document.metadata["activation_regime"]
    if regime not in ACTIVATION_REGIMES:
        raise ConfigError(f"{source}: activation_regime {regime!r} is not one of {sorted(ACTIVATION_REGIMES)}")
    return Constants(
        values=MappingProxyType(values),
        frozen_at=_parse_frozen_at(document.metadata["frozen_at"], source),
        activation_regime=regime,
    )
