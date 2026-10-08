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
    distance_binding_rate: float  # M1.D.2a — share of the distancer's excess bound into the tie per unit scaled intensity
    triangle_transfer_rate: float # M1.C.1 — share of each insider's excess passed to the outsider per unit scaled intensity, before capacity
    outsider_positional_gain: float # M1.C.1, KS03.1 — the outsider's own positional anxiety per unit absorbed
    balance_push_gain: float      # M1.B.5 — how far one act pushes the functioning balance per unit scaled intensity
    balance_settle_rate: float    # M1.B.5 — share of the gap to its pole a balance closes per tick
    balance_harden_rate: float    # M1.B.6, L05.3 — share of the gap to the balance the habit closes per tick
    reversal_asymmetry: float     # M1.B.7 — how much harder it is to raise a marked under-functioner
    pseudo_self_transfer_gain: float # M6.I.4 — pseudo-self points one over/underfunctioning act moves per unit scaled intensity
    self_channel_exponent: float  # M4.D.1a — mixing weight of the self-directed channel = (functional_level / 100) ** this
    policy_temperature: float     # M4.D.1 — the softmax temperature over each channel's scores
    anxiety_band_low: float       # M4.D.3(b) — excess acute anxiety at the top of the low band
    anxiety_band_high: float      # M4.D.3(b) — excess acute anxiety at the bottom of the high band
    capacity_level_per_layer: float # M4.D.3a — functional level a newer layer needs, per layer, for full availability
    competing_urge_gain: float    # M4.D.1d — acute anxiety added per tick at a fully undecided automatic channel
    withhold_investment_gain: float # M4.D.1b — attention a withheld move still puts into its tie
    loaded_tie_threshold: float   # M4.D.3b — a tie whose deviation exceeds this is loaded
    policy_intensity: float                # M4.D.1 — the intensity of every act the policy emits, Phase C step 6
    learning_rate: float       # M4.D.6, plan D4 — value ← value + learning_rate × (signal − value)
    credit_horizon: int        # M4.D.6b — weeks after an act over which its felt effect is credited to it
    credit_discount: float     # M4.D.6, plan D4 — the signal an act receives at age k is weighted credit_discount ** k
    cross_person_weight: float # M4.D.6e — weight on the target's and witnesses' change in anxiety
    habituation_rate: float    # M4.G.3 — relief credited to the n-th recent repetition is scaled by habituation_rate ** n
    habituation_window: int    # M4.G.3 — how far back an identical act counts as a repetition
    prepare_ticks: int         # M5.D.2a — weeks of private preparation before DEFINE
    rehearsal_rate: float      # M1.A.9, M5.D.2a — private rehearsal lowers each impingement axis by this share per PREPARE tick
    assertion_perspective_threshold: float # M5.F.4 — below this, I-POSITION executes as the assertion form
    anger_threshold: float     # M5.D.4, M5.D.4a — the mover's too-much side on the tie above which the mover is angry
    assertion_gain: float      # M5.F.4 — extra impingement an assertion-form I-POSITION delivers
    assertion_evidence_gain: float # M5.F.2a — claiming a position raises the claimant's outward axis
    opposition_window: int     # M5.E.3 — weeks after DEFINE in which no opposition means the move did not land
    hold_gain: float           # M5.D.3 — capacity to hold = hold_gain × functional_level × efficacy
    stall_limit: int           # M5.D.4 — angry weeks in HOLD before the sequence lapses
    hold_window: int           # M5.D.2 — weeks in HOLD without a final attack before the sequence settles without a peak
    pull_up_rate: float        # M5.D.5 — on RESOLVE the opposition's functional level closes this share of the gap to the mover's
    exchange_gain: float       # M5.D.7, M5.D.7a — a completed exchange's functional-level increment; small by design
    triangle_floor_decrement: float # M1.C.5 — a completed exchange's permanent decrement in each triangle holding the pair
    respect_gain: float        # M5.E.8 — a completed exchange lowers both parties' impingement axes by this share
    debit_gain: float          # M5.E.7 — contact the mover withdraws from the other with a genuine I-POSITION
    session_interval_weeks: int # M1.E.8, K07.4 — an external agent's session falls every this many weeks (six to eight a year)
    landing_rate: float        # M1.E.7 — the base chance a contact lands; low by source (K07.4: about five in two years)
    delayed_view_bonus: float  # M1.E.7c — a contact that can draw on the recipient's own delayed log lands this much more often
    binder_failure_fraction: float # M1.E.7 — binders count as failing when a symptom is active or the load reaches this share of threshold
    contact_optimum: int       # M1.E.8 — above this many contacts in the window, the chance each lands falls with the square of the excess
    contact_window: int        # M1.E.8 — the window over which contacts are counted
    perspective_gain: float    # M1.E.7 — a landed contact raises systems_perspective by this share of its distance from 1
    delayed_view_weeks: int    # M16.D.2 — the delay of the delayed view: six months, as Bowen suggested
    sink_window: int           # M1.D.1 — weeks of acts that weight each sink
    sink_rate: float           # M1.D.1 — share of the gap to its target share each allocation closes per tick
    exchange_budget_reduction: float # M6.I.1, M5.D.7 — what a completed differentiating exchange removes from the budget
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
        if min(self.self_channel_exponent, self.policy_temperature, self.capacity_level_per_layer, self.policy_intensity) <= 0:
            raise ValueError("self_channel_exponent, policy_temperature, capacity_level_per_layer and policy_intensity must be positive")
        if not 0 <= self.anxiety_band_low <= self.anxiety_band_high:
            raise ValueError("anxiety bands need 0 <= low <= high")
        if min(self.competing_urge_gain, self.withhold_investment_gain, self.loaded_tie_threshold) < 0:
            raise ValueError("competing_urge_gain, withhold_investment_gain and loaded_tie_threshold are non-negative")
        for name in ("learning_rate", "credit_discount", "cross_person_weight", "habituation_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.credit_horizon < 1 or self.habituation_window < 1:
            raise ValueError("credit_horizon and habituation_window must be at least 1")
        for name in ("rehearsal_rate", "assertion_perspective_threshold", "pull_up_rate", "respect_gain",
                     "triangle_floor_decrement"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if min(self.prepare_ticks, self.opposition_window, self.stall_limit, self.hold_window) < 1:
            raise ValueError("prepare_ticks, opposition_window, stall_limit and hold_window must be at least 1")
        if min(self.anger_threshold, self.assertion_gain, self.assertion_evidence_gain, self.hold_gain,
               self.exchange_gain, self.debit_gain) < 0:
            raise ValueError("the I-POSITION gains and thresholds are non-negative")
        for name in ("landing_rate", "binder_failure_fraction", "perspective_gain"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if min(self.session_interval_weeks, self.contact_optimum, self.contact_window) < 1 or self.delayed_view_weeks < 0 or self.delayed_view_bonus < 0:
            raise ValueError("the external-agent windows must be at least 1; delayed_view_weeks and delayed_view_bonus non-negative")
        if self.sink_window < 1 or not 0.0 <= self.sink_rate <= 1.0 or self.exchange_budget_reduction < 0:
            raise ValueError("sink_window >= 1, sink_rate in [0, 1], exchange_budget_reduction non-negative")
        if self.outsider_positional_gain < 0 or self.pseudo_self_transfer_gain < 0:
            raise ValueError("outsider_positional_gain and pseudo_self_transfer_gain must be non-negative")
        for name in ("witness_weight", "calm_transfer_rate", "symptom_leak_rate", "symptom_rearm_fraction",
                     "reactive_rate", "investment_leak_rate", "attention_gain", "belief_rate",
                     "distance_binding_rate", "triangle_transfer_rate", "balance_push_gain", "balance_settle_rate",
                     "balance_harden_rate", "reversal_asymmetry"):
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
