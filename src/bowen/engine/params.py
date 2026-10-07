"""The numeric parameters the engine's mechanisms read.

Purpose: hand the engine its constants as plain values, so the engine never
         imports configuration code (M11.D.1, the package layout of the plan).
Spec:    docs/bowen_agent_model_spec_v2.md#M0.3, #M10.1
Tests:   tests/bowen/test_mechanisms.py

Every field is graded in ``config/bowen/constants.md``; ``src/bowen/scenario/params.py``
builds this object from the parsed register. All are invented ([I]).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineParams:
    slow_tick_fast_ticks: int
    invariant_tolerance: float
    per_hop_fidelity: float
    standing_load_gain: float              # M4.A.1 — the standing-load function's scale
    interactive_standing_fraction: float   # M4.A.3 — share of standing load an interactive tie keeps
    functional_level_floor: float          # divisor floor for M4.A.1 and M4.C.1 at functional_level 0
    acute_decay_rate: float                # M1.A.8 — fraction of excess over the chronic floor shed per tick
    route_damping: float                   # M1.F.3 — gain per neutral third on the route
    hardening_run_length: int              # M4.G.1 — consecutive withdrawals that make a tie distant
    bond_energy_decay_rate: float          # M1.B.4 — at or near zero
    tension_activation_threshold: float    # M1.C.3 — tie tension at which a triangle can activate
    involvement_membership_threshold: float  # M1.A.12 — membership is a threshold over involvement

    def __post_init__(self) -> None:
        for name in ("per_hop_fidelity", "interactive_standing_fraction", "route_damping"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0.0 < self.acute_decay_rate <= 1.0:
            raise ValueError("acute_decay_rate must be in (0, 1]")
        if not 0.0 <= self.bond_energy_decay_rate < 1.0:
            raise ValueError("bond_energy_decay_rate must be in [0, 1)")
        if self.functional_level_floor <= 0:
            raise ValueError("functional_level_floor must be positive")
        if self.hardening_run_length < 2:
            raise ValueError("hardening_run_length must be at least 2")
