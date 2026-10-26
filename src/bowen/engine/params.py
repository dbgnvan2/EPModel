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
    conflict_binding_rate: float  # M1.D.2c, revision 36 — the same for a conflict, a smaller share
    cutoff_stock_cap: float       # M1.D.2b, revision 36 — the most stock a cut-off tie holds (the largest measured at a cut)
    cutoff_threat_weight: float   # M1.D.2b, revision 36 — intensity per unit of stock per week at bond energy 100
    cutoff_decay_weeks: int       # M1.D.2b, revision 36 — weeks for a cut-off tie's stock to fall to its floor (the owner's five years)
    cutoff_floor_share: float     # M1.D.2b, revision 36 — the floor as a share of the peak (the owner's 10%)
    positive_bridge_dissipation: float  # M1.D.2b, revision 36 — share of the stock a positive contact across a cut tie lets go (the owner's 80%)
    tie_repair_steps: int         # M1.D.2c, revision 36 — positive contacts that repair a distant or conflicted tie (the owner's thirds)
    triangle_transfer_rate: float # M1.C.1 — share of the seeker's excess passed to the outsider per unit scaled intensity, before capacity and the third's help
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
    anger_threshold: float     # M5.D.4, M5.D.4a — the mover's too-much side on the tie above which the mover is angry
    assertion_gain: float      # M5.F.4 — extra impingement an I-POSITION delivers, times its demand share
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
    contact_excess_exponent: float         # M1.E.8 — past the optimum, landing falls as (optimum / n) ** this
    belief_rate: float                     # M9.8 — share of the gap to a witnessed observation closed per tick, at full fidelity
    rejection_cutoff_gain: float           # M1.B.3c — the cutter's rejection a CUTOFF adds per unit strength
    rejection_refusal_gain: float          # M1.B.3c — rejection each refused move adds per unit strength
    rejection_relief: float                # M1.B.3c, M5.B.3 — rejection an accepted REDUCE_CUTOFF removes per unit strength
    refusal_cost_gain: float               # M1.B.3c — the refused sender's acute anxiety per unit strength × conductance × steepness
    declared_rejection: float              # M1.B.3c — each member's rejection on a tie the family declares cut off
    rejection_refusal_threshold: float     # M1.B.3c, revision 16 — rejection above which a move is refused
    rejection_fade: float                  # M1.B.3c, revision 16 — rejection lost per week
    estimator_window: int                  # M1.A.4k — slow ticks of functioning history the estimator reads
    estimator_breadth: int                 # M1.A.4k — distinct kinds of nodal event the window must include
    estimator_sd_weight: float             # M1.A.4k — how far the load-weighted spread discounts the mean
    estimator_rate: float                  # M1.A.4k — share of the evidence a slow tick turns into a rise
    tie_wear_rate: float                   # M7.E.4 — conductance lost per slow tick per unit mean chronic anxiety / 100
    wear_offset_per_act: float             # M7.E.4 — share of a year's wear each STAY-IN-CONTACT on the tie offsets
    perspective_channel_gain: float        # M4.D.1a, revision 18 — how far perspective opens the self-directed channel
    self_practice_gain: float              # M1.E.7, revision 18 — perspective a self-directed act under load adds, of its distance from 1
    stressor_persistence: float            # M1.F.6, revision 18 — each later week of a lasting stressor, as a share of its arrival
    response_up_gain: float                # M1.A.5e, revision 18 — functioning gained per unit load by coping, at basic level 100
    response_down_gain: float              # M1.A.5e, revision 18 — functioning lost per unit load by soothing, at basic level 0
    response_relaxation_rate: float        # M1.A.5e, M7.A.1 — share of the stress-moved swing shed per calm week
    stress_floor_below_basic: float        # M1.A.5e, revision 19 — how far below basic_level stress can push functioning
    practice_window: int                   # M1.E.7, revision 19 — weeks after a self-directed act before it is judged to have worked; M1.B.5, revision 21 — how long a stance holds
    perspective_steer_gain: float          # M4.D.1a, revision 19 — how far full perspective scales a steered act's weight
    system_adjust_gain: float              # M1.E.7, revision 19 — functioning the other gains when a held stance works
    soothing_review_every: int             # M4.D.6, revision 19 — weeks between a person's chances to ask whether soothing helped
    soothing_review_penalty: float         # M4.D.6, revision 19 — share of a soothing act's learned value lost when it did not help
    usual_level_rate: float                # M1.A.5e, revision 22 — share per week by which a person's usual level follows their excess
    constitution_strength: float           # M1.A.11c, revision 24 — how strongly the constitutional channel prior holds against a family focus
    channel_protection: float              # M7.D.2c, revision 24 — how much load in other channels slows a channel's feed
    lock_in_gain: float                    # M7.D.2a, revision 24 — scale of the share of a partner's excess an active symptom holds
    attribution_rate: float                # M9.6, revision 25 — share per unit of fidelity by which attribution moves toward what a move places
    focus_rate: float                      # M1.A.11c, revision 25 — weekly decay of the family's focus, and its gain per anxious act
    marital_distance_fraction: float       # M5.C.1, revision 25 — felt contact below this share of optimum, both spouses, is marital distance
    belief_appraisal_gain: float           # M9.7, revision 25 — how much a witness's believed tension of the tie scales its appraisal
    pseudo_self_relaxation_rate: float     # M6.I.4, revision 27 — weekly share of a pair's opposite pseudo-self that settles back
    exchange_window: int                   # M6.I.4, revision 27 — weeks without over/under-functioning before it settles
    acting_age_years: int                  # M7.C.1f, revision 28 — a person selects acts from this age until death
    development_start_age_years: int       # M7.E.1f, revision 28 — a child's basic level is fixed to its anchor and develops from this age
    development_end_age_years: int         # M7.E.1f, revision 28 — development levels off; the estimator governs from here
    development_band: float                # M7.E.1f, revision 28 — how far a child's level may end from the anchor
    development_rate: float                # M7.E.1f, revision 28 — how far a year's functioning moves a developing child's level
    development_band_projection: float     # M7.E.1f, revision 31 — the floor's room when every parental act to the child was projection
    development_band_support: float        # M7.E.1f, revision 31 — the ceiling's room when every parental act to the child was supportive
    witnessed_gain: float                  # M7.B.1a, revision 28 — a child's own load raises its chronic anxiety above the parents' before the fixation age
    chronic_anxiety_fixation_age_years: int  # M2.A.0a, M7.B.1a — revision 28 reads it: a child's chronic anxiety is derived until this age
    spouse_basic_level_tolerance: float    # M2.A.0e — revision 28 reads it: a partner's level is drawn within it (M7.C.1h)
    launch_age_years: int                  # M7.C.1g, revision 28 — a child leaves the household and supports themselves
    working_age_years: int                 # M7.C.1k, revision 28 — the youngest who can take FIND_WORK
    find_work_rate: float                  # M7.C.1k, revision 28 — how readily a pressed member finds work
    find_work_pressure_scale: float        # M7.C.1k, revision 28 — the load above usual at which the pressure to find work is full
    grief_weight: float                    # M6.3g, revision 36 — grief's size as a multiple of its Holmes–Rahe size (M1.F.6a); 0 switches it off
    lcu_scale: float                       # M1.F.6a, revision 36 — intensity units per life change unit (the owner's anchor: 324 ÷ 47)
    grief_min_weeks: int                   # M6.3g, revision 28 — grief on the weakest tie lasts a year (the owner's 1-3 years)
    grief_extra_weeks: int                 # M6.3g, revision 28 — grief on the strongest tie lasts two more years
    income_loss_intensity: float           # M6.3h, revision 28 — the size of the stress of a lost income
    income_shortfall_share: float          # M6.3h, revision 28 — what remains of the stress once someone finds work
    mortality_a: float                     # M7.C.1j, revision 28 — the Gompertz curve's level: with mortality_b, a median age at death near 80
    mortality_b: float                     # M7.C.1j, revision 28 — the Gompertz curve's rise with age
    budget_per_undifferentiation: float    # M7.C.1h, revision 28 — a new family's budget; 1.49 gives about 90 at the ravi_marta family's 39.5
    coach_start_age_years: int             # M2.A.0k, revision 28 — a new coach's age at entry
    coach_retirement_age_years: int        # M2.A.0k, revision 28 — the coach retires and a new one starts over
    new_tie_investment_share: float        # M7.C.1a, revision 28 — what a new tie claims of a person's investment at partnering or birth
    birth_focus_gain: float                # M7.E.1c, revision 28 — each of the five situations at a birth adds anxious focus
    death_transfer_share: float            # M6.3b, revision 28 — 1 is the rule; only M11.C.37's mutant sets 0
    reactive_cutoff_gain: float            # M1.B.3d, revision 29 — a held reactive cutoff's weekly cost to its holder
    witnessed_cutoff_gain: float           # M4.D.6f, revision 29 — learning per point of functional change seen
    witness_window: int                    # M4.D.6f, revision 29 — weeks between a witnessed cutoff and its reading
    perceived_weakness_gain: float         # M1.B.8a, revision 29 — focus per point of a child's shortfall
    focus_steer_gain: float                # M4.D.1g, revision 29 — how strongly focus steers projection-type acts
    origin_cutoff_anxiety_gain: float        # M1.D.8a, revision 30 — intensity a child carries from the cutoff they grew up with
    origin_fusion_gain: float                # M1.D.8a, revision 30 — the pull toward fusion a child carries from it
    nodal_spike_intensity: float             # M4.C.1c, revision 32 — a nodal event's spike on the cut ties of those it reaches
    danger_window: int                       # M1.B.3e, revision 30 — weeks of received intrusion that make felt danger
    danger_threshold: float                  # M1.B.3e, revision 30 — intrusion beyond ordinary family life
    danger_gain: float                       # M1.B.3e, revision 30 — what contact with a dangerous person costs
    focus_reinforce_gain: float              # M1.B.8b, revision 30 — attention follows one's own anxious acts
    force_anxiety_gain: float                # M4.D.1h, revision 30 — an anxious person acts with more force
    focus_compound_gain: float               # M1.B.8b, revision 30 — anxious focus grows where it already is
    lock_in_turn: float                    # M7.D.2d, revision 24 — the severity (load over threshold) past which lock-in turns over

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
        for name in ("rehearsal_rate", "pull_up_rate", "respect_gain", "triangle_floor_decrement"):
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
                     "balance_harden_rate", "reversal_asymmetry", "rejection_cutoff_gain",
                     "rejection_refusal_gain", "rejection_relief", "rejection_fade",
                     "conflict_binding_rate", "cutoff_floor_share", "positive_bridge_dissipation"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        # Revision 36 (M1.D.2b, c): the cap and weight are non-negative; the decay needs a positive span and a floor
        # share above zero (a zero share makes the weekly factor 0); repair needs at least one step.
        if self.cutoff_stock_cap < 0 or self.cutoff_threat_weight < 0:
            raise ValueError("cutoff_stock_cap and cutoff_threat_weight must be non-negative")
        if self.cutoff_decay_weeks < 1 or self.cutoff_floor_share <= 0.0 or self.tie_repair_steps < 1:
            raise ValueError("cutoff_decay_weeks and tie_repair_steps must be at least 1 and cutoff_floor_share positive")
        if min(self.defence_threshold, self.perspective_anxiety_scale, self.symptom_threshold_gain) <= 0:
            raise ValueError("defence_threshold, perspective_anxiety_scale and symptom_threshold_gain must be positive")
        if self.reappraisal_window < 1 or self.speaker_echo_gain < 0 or self.symptom_event_intensity < 0:
            raise ValueError("reappraisal_window >= 1; speaker_echo_gain and symptom_event_intensity non-negative")
        if self.refusal_cost_gain < 0 or not 0.0 < self.declared_rejection <= 1.0:
            raise ValueError("refusal_cost_gain must be non-negative and declared_rejection in (0, 1]")
        if self.estimator_window < 1 or self.estimator_breadth < 0 or self.estimator_sd_weight < 0:
            raise ValueError("estimator_window >= 1; estimator_breadth and estimator_sd_weight non-negative")
        for name in ("estimator_rate", "tie_wear_rate", "wear_offset_per_act", "perspective_channel_gain",
                     "self_practice_gain", "stressor_persistence", "response_up_gain", "response_down_gain",
                     "response_relaxation_rate", "perspective_steer_gain"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.soothing_review_every < 1 or not 0.0 <= self.soothing_review_penalty <= 1.0:
            raise ValueError("soothing_review_every >= 1; soothing_review_penalty in [0, 1]")
        if self.system_adjust_gain < 0:
            raise ValueError("system_adjust_gain must be non-negative")
        if not 0.0 < self.usual_level_rate <= 1.0:
            raise ValueError("usual_level_rate must be in (0, 1]")
        if not 0.0 <= self.constitution_strength <= 1.0:
            raise ValueError("constitution_strength must be in [0, 1]")
        if not (0.0 <= self.attribution_rate <= 1.0 and 0.0 <= self.focus_rate <= 1.0) or self.belief_appraisal_gain < 0:
            raise ValueError("attribution_rate and focus_rate in [0, 1]; belief_appraisal_gain non-negative")
        if not 0.0 < self.marital_distance_fraction <= 1.0:
            raise ValueError("marital_distance_fraction must be in (0, 1]")
        if not 0.0 <= self.pseudo_self_relaxation_rate <= 1.0 or self.exchange_window < 1:
            raise ValueError("pseudo_self_relaxation_rate in [0, 1]; exchange_window at least 1")
        if self.channel_protection < 0 or self.lock_in_gain < 0 or self.lock_in_turn <= 0:
            raise ValueError("channel_protection and lock_in_gain non-negative; lock_in_turn positive")
        if self.stress_floor_below_basic < 0 or self.practice_window < 1:
            raise ValueError("stress_floor_below_basic must be non-negative and practice_window at least 1")
        if not 0.0 <= self.rejection_refusal_threshold < 1.0:
            raise ValueError("rejection_refusal_threshold must be in [0, 1)")
        if self.triangle_activity_window < 1:
            raise ValueError("triangle_activity_window must be at least 1")
        if self.hardening_run_length < 2:
            raise ValueError("hardening_run_length must be at least 2")
        # Revision 28's life course (review 2026-10-14, adjacent finding): ages ordered, shares in [0, 1], sizes non-negative.
        if not (0 <= self.acting_age_years and 0 <= self.development_start_age_years <= self.development_end_age_years
                and self.working_age_years <= self.launch_age_years
                and 0 <= self.coach_start_age_years < self.coach_retirement_age_years):
            raise ValueError("life-course ages out of order: development start <= end, working age <= launch age, "
                             "coach start < retirement, all non-negative")
        if not (self.development_band <= self.development_band_projection <= 1.0
                and self.development_band <= self.development_band_support <= 1.0):
            raise ValueError("development_band_projection and development_band_support must lie in [development_band, 1]")
        for name in ("development_band", "development_rate", "find_work_rate", "income_shortfall_share",
                     "new_tie_investment_share", "death_transfer_share"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if min(self.witnessed_gain, self.grief_weight, self.lcu_scale, self.income_loss_intensity, self.mortality_a,
               self.mortality_b, self.budget_per_undifferentiation, self.birth_focus_gain) < 0:
            raise ValueError("witnessed_gain, grief_weight, lcu_scale, income_loss_intensity, mortality_a, mortality_b, "
                             "budget_per_undifferentiation and birth_focus_gain must be non-negative")
        if self.find_work_pressure_scale <= 0 or self.grief_min_weeks < 0 or self.grief_extra_weeks < 0:
            raise ValueError("find_work_pressure_scale must be positive and the grief weeks non-negative")
        # Revision 29 (learning-qa review 2026-10-14): the witnessed-cutoff step stays bounded only for a gain in [0, 1].
        if self.nodal_spike_intensity < 0:
            raise ValueError("nodal_spike_intensity must be non-negative")
        if not 0.0 <= self.witnessed_cutoff_gain <= 1.0 or self.witness_window < 1:
            raise ValueError("witnessed_cutoff_gain must be in [0, 1] and witness_window at least 1")
        if min(self.reactive_cutoff_gain, self.perceived_weakness_gain, self.focus_steer_gain) < 0:
            raise ValueError("reactive_cutoff_gain, perceived_weakness_gain and focus_steer_gain must be non-negative")
        if self.danger_window < 1 or self.danger_threshold <= 0:
            raise ValueError("danger_window must be at least 1 and danger_threshold positive")
        if min(self.origin_cutoff_anxiety_gain, self.origin_fusion_gain, self.danger_gain, self.focus_reinforce_gain,
               self.force_anxiety_gain, self.focus_compound_gain) < 0:
            raise ValueError("origin_cutoff_anxiety_gain, origin_fusion_gain, danger_gain, focus_reinforce_gain, "
                             "force_anxiety_gain and focus_compound_gain must be non-negative")
