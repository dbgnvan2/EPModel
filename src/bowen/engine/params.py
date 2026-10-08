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
    standing_load_gain: float              # M4.C.1c — scale of the "too little" side, every tick
    appraisal_gain: float                  # M4.C.1 — scale of a delivered event's change in deviation
    intensity_scale: float                 # M4.C.1 — event intensity that moves felt contact by one component
    contact_band_max: float                # M4.C.1a — tolerated deviation at functional level 100
    anxiety_togetherness_gain: float       # M4.C.1b — how far anxiety moves the optimum toward closeness
    interactive_resting_contact: float     # M4.C.1c — resting contact on an interactive tie, as a share of the optimum
    contact_relaxation_rate: float         # M4.C.1c — share of the gap to resting contact closed per tick
    impingement_relaxation_rate: float     # M4.C.1 — share of felt impingement shed per tick
    functional_level_floor: float          # M4.C.1a — steepness floor at functional_level 0
    acute_decay_rate: float                # M1.A.8 — fraction of excess over the chronic floor shed per tick
    route_damping: float                   # M1.F.3 — gain per neutral third on the route
    hardening_run_length: int              # M4.G.1 — consecutive withdrawals that make a tie distant
    bond_energy_decay_rate: float          # M1.B.4 — at or near zero
    triangle_activity_window: int          # M1.C.3 amended — ticks a TRIANGLE act keeps its triangle active
    defence_threshold: float               # M4.C.2 — excess anxiety above which content is defended against
    witness_weight: float                  # M4.C.9 — a witness's share of an exchange it overhears
    speaker_echo_gain: float               # M4.C.7 — the speaker's own reaction to what it addressed
    reappraisal_window: int                # M4.C.6 — ticks of own emissions a reappraiser counts on a tie
    attention_gain: float                  # M4.C.8 — amplification of attended feeling; ordering of attended intellect
    perspective_anxiety_scale: float       # M4.C.4, M1.A.18d — excess anxiety that halves the effective perspective
    calm_transfer_rate: float              # M4.C.10 — share of the anxiety gap a calmer sender takes per unit conductance
    symptom_leak_rate: float               # M4.C.3a — share of the chronicity integral that leaks per tick
    symptom_threshold_gain: float          # M1.A.6, M7.D.1 — onset threshold per point of functional level
    symptom_rearm_fraction: float          # M7.D.1 — share of threshold below which a channel can fire again
    symptom_event_intensity: float         # M7.D.1 — intensity of the endogenous event a symptom emits
    reactive_rate: float                   # M1.A.19 — share of the gap each detector closes per tick
    investment_leak_rate: float            # M1.B.8 — share of attention on a tie that fades per tick
    initial_impingement_scale: float       # M1.A.9 — both axes at t0 = this × (1 − basic_level / 100)
    outside_ness_rate: float               # M1.A.9 — share of the gap to this tick's behaviour closed per tick
    hollow_gain: float                     # M5.F.1 — how far inward impingement empties a move's contact
    assault_gain: float                    # M5.F.1 — impingement a move gains per unit of outward impingement
    outside_ness_threshold_outward: float  # M5.C.1, revision 12 P9 — the gate fails above this on the outward axis
    outside_ness_threshold_inward: float   # M5.C.1, revision 12 P9 — and above this on the inward axis
    involvement_membership_threshold: float  # M1.A.12 — membership is a threshold over involvement
    belief_rate: float                     # M9.8 — share of the gap to a witnessed observation closed per tick, at full fidelity

    def __post_init__(self) -> None:
        for name in (
            "per_hop_fidelity", "route_damping", "interactive_resting_contact",
            "contact_relaxation_rate", "impingement_relaxation_rate",
        ):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0.0 < self.acute_decay_rate <= 1.0:
            raise ValueError("acute_decay_rate must be in (0, 1]")
        if not 0.0 <= self.bond_energy_decay_rate < 1.0:
            raise ValueError("bond_energy_decay_rate must be in [0, 1)")
        if self.functional_level_floor <= 0:
            raise ValueError("functional_level_floor must be positive")
        if not 0.0 <= self.contact_band_max < 1.0:
            raise ValueError("contact_band_max must be in [0, 1)")
        if self.intensity_scale <= 0 or self.appraisal_gain < 0 or self.anxiety_togetherness_gain < 0:
            raise ValueError("intensity_scale must be positive; appraisal_gain and anxiety_togetherness_gain non-negative")
        for name in ("initial_impingement_scale", "outside_ness_rate", "hollow_gain",
                     "outside_ness_threshold_outward", "outside_ness_threshold_inward"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.assault_gain < 0:
            raise ValueError("assault_gain must be non-negative")
        for name in ("witness_weight", "calm_transfer_rate", "symptom_leak_rate", "symptom_rearm_fraction",
                     "reactive_rate", "investment_leak_rate", "attention_gain", "belief_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if min(self.defence_threshold, self.perspective_anxiety_scale, self.symptom_threshold_gain) <= 0:
            raise ValueError("defence_threshold, perspective_anxiety_scale and symptom_threshold_gain must be positive")
        if self.reappraisal_window < 1 or self.speaker_echo_gain < 0 or self.symptom_event_intensity < 0:
            raise ValueError("reappraisal_window >= 1; speaker_echo_gain and symptom_event_intensity non-negative")
        if self.triangle_activity_window < 1:
            raise ValueError("triangle_activity_window must be at least 1")
        if self.hardening_run_length < 2:
            raise ValueError("hardening_run_length must be at least 2")
