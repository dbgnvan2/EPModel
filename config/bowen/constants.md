# EPModel v2 — constants register

> Every numeric constant the engine reads, with its grade (spec `M10.1`, `M0.1`).
> `[I]` means invented: a modelling decision with no source. An `[I]` value must
> never be described as sourced, derived from Bowen, or theoretically grounded (`M0.2`).
> Parsed strictly by `src/bowen/scenario/constants.py`: an unknown key, a missing
> key, a grade that differs from the spec's, or any unrecognised line raises (`M10.B.2`).
> `frozen_at` is set before the acceptance suite first runs against these values;
> any later change that alters an acceptance outcome is logged (`M10.B.4`).
>
> `chronic_anxiety_fixation_age_years` has no source. `M2.A.0a` states that every
> agent in the reference family is past it, and the youngest is 14, so it must be
> at most 14. 12 was chosen at Phase B step 5.
>
> `activation_regime` is `[I]` (`M3.E.2`): every person selects every fast tick.
> `per_hop_fidelity` stands in for the per-hop fidelity draw of `M3.D.4b` in
> Phase B, which makes no draws (plan decision D7). 0.8 was chosen at step 7.
>
> The step 8 constants (from `standing_load_gain` down) were chosen together so that,
> on the Phase B family, the standing load alone holds each person a few points above
> their chronic floor: steady excess = standing load ÷ `acute_decay_rate`. `M4.G.1`'s
> "three withdrawals" is the spec's own number, still `[I]`. `bond_energy_decay_rate`
> is 0 because `M1.B.4` requires decay at or near zero. `functional_level_floor` only
> stops a division by zero at `functional_level` 0, which `M1.A.2` allows.
>
> **Phase C step 1, 2026-10-07 — spec revision 11's two-sided appraisal (`M4.C.1`–`M4.C.1c`, plan D2).**
> `standing_load_gain` keeps its value and now scales the "too little" side of the appraisal function.
> The constants from `appraisal_gain` to `impingement_relaxation_rate` and `triangle_activity_window` are new;
> `interactive_standing_fraction` and `tension_activation_threshold` are retired. Every change is logged in
> `constants_changes.md`. The new values were chosen so that, on the Phase B family, magnitudes stay on the
> Phase B scale: `intensity_scale` 100 makes a stressor's appraisal equal Phase B's (intensity ÷ functional
> level); an interactive tie resting at half its optimum (`interactive_resting_contact` 0.5) carries roughly
> Phase B's interactive standing load. `triangle_activity_window` 4 keeps a triangle active for four weeks
> after a `TRIANGLE` act. All are `[I]`; Phase C re-freezes at step 14.
>
> **Phase C step 2, 2026-10-07 — the rest of the appraisal** (`M4.C.2`–`M4.C.10`, `M4.C.3`, `M7.D.1`,
> `M1.A.19`, `M1.B.8`). The constants from `defence_threshold` to `investment_leak_rate` are new and logged.
> Chosen so that a witness takes about half of what Phase B gave it, below the person addressed; content stops
> getting through at 15–30 points of excess anxiety; and the symptom threshold (1.5 × functional level, about 60
> for the Phase B adults) sits above the integral a steady few points of excess reach (excess ÷ `symptom_leak_rate`),
> so onset needs a sustained excursion. All `[I]`.
>
> **Phase C step 3, 2026-10-07 — outside-ness** (`M1.A.9`, `M1.A.9a`, `M4.C.5`, `M5.F`). The six constants from
> `initial_impingement_scale` down are new and logged. At 0.8, the Phase B adults (basic level about 40) start
> near 0.5 on both axes, at the gate's thresholds: the gate passes or fails on behaviour, not on the start.
> `hollow_gain` and `assault_gain` at 0.5 keep act identity smaller than the move's own components. All `[I]`.
>
> **Phase C steps 4–7, 2026-10-07** — beliefs (`belief_rate`), move physics (`distance_binding_rate` …
> `pseudo_self_transfer_gain`), the policy (`self_channel_exponent` … `policy_intensity`) and the learner
> (`learning_rate` … `habituation_window`). Each is new and logged in `constants_changes.md`. All `[I]`.
>
> **Phase C step 8, 2026-10-07 — the `I-POSITION` state machine** (`M5.D`, `M5.E`, `M5.F.4`, `M1.C.5`), from
> `prepare_ticks` to `debit_gain`. All `[I]`. **`exchange_gain` is small on purpose (`M5.D.7a`):** half a point
> of functional level per completed exchange, against basic levels around 40, and `basic_level` is never
> written by an exchange. No realistic number of exchanges in a run should read as differentiation: the KB
> interviews call that reading "grotesque". **`hold_gain` was chosen against `M5.D.3`**, so that aborts are the
> usual outcome at first opposition; `constants_changes.md` records the measurement.
>
> **Phase D B4, 2026-10-11 — cutoff as rejection** (`M1.B.3c`, owner decisions P1 and P2), from
> `rejection_cutoff_gain` to `declared_rejection`. All `[I]`. One `CUTOFF` at the policy's intensity takes rejection
> halfway to its ceiling; four accepted `REDUCE_CUTOFF` acts clear it; a refusal adds a tenth. A refusal costs its
> sender about as much as being met with conflict. Neither family states who cut off whom, so both members of a
> declared cut-off tie start fully rejecting. **Revision 16 (owner, 2026-10-11, graded rejection):** a move is refused
> only above `rejection_refusal_threshold`, and rejection fades by `rejection_fade` a week. One cutoff (0.5) refuses for
> about four weeks and each refusal renews it by about two; unrenewed, it is gone in ten weeks, a declared cutoff in
> twenty. The threshold is 0.3, not the 0.5 first proposed: at 0.5 a single cutoff (exactly 0.5) refused nothing
> (owner decision 2026-10-11). Measured in `docs/implementation_plan_phase_d.md` B4.
>
> **Phase D B5 and B6, 2026-10-11 — the slow tick and the estimator** (`M1.A.4k`, `M7.E.4`; owner decisions P4–P6),
> from `estimator_window` to `wear_offset_per_act`. All `[I]`.
> - **The estimator** reads three years of history and wants three distinct kinds of nodal event in it. It turns half of the
>   evidence a year into a rise, scaled by the falloff and the self-directed share.
> - **Tie wear.** At mean chronic anxiety 40, a tie loses 2% of its conductance a year; ten `STAY-IN-CONTACT` acts on it
>   offset that year's wear entirely.
> - **The sweep.** `M1.A.4c`'s sensitivity sweep over window and breadth is added when `M11.C.18` first reads the
>   estimator.

frozen_at: 2026-10-07
activation_regime: synchronous

| key | value | grade | unit | spec |
|---|---|---|---|---|
| `fast_tick_weeks` | 1 | [I] | week | `M3.A.1` |
| `slow_tick_fast_ticks` | 52 | [I] | fast tick | `M3.B.1` |
| `invariant_tolerance` | 1e-9 | [I] | anxiety unit | `M6.1` |
| `spouse_basic_level_tolerance` | 1.0 | [I] | basic-level point | `M2.A.0e` |
| `chronic_anxiety_fixation_age_years` | 12 | [I] | year of age | `M2.A.0a` |
| `per_hop_fidelity` | 0.8 | [I] | fraction kept per private hop | `M1.F.4` |
| `standing_load_gain` | 0.2 | [I] | anxiety per tick per unit of "too little" deviation per unit of steepness | `M4.A.1` |
| `appraisal_gain` | 3.0 | [I] | anxiety per unit change in deviation per unit of steepness | `M4.C.1` |
| `intensity_scale` | 100.0 | [I] | event intensity per unit component | `M4.C.1` |
| `contact_band_max` | 0.2 | [I] | deviation tolerated at functional level 100 | `M4.C.1a` |
| `anxiety_togetherness_gain` | 0.5 | [I] | relative rise of the optimum per 100 points of excess anxiety | `M4.C.1b` |
| `interactive_resting_contact` | 0.5 | [I] | share of the optimum | `M4.C.1c` |
| `contact_relaxation_rate` | 0.1 | [I] | share of the gap to resting contact closed per tick | `M4.C.1c` |
| `impingement_relaxation_rate` | 0.3 | [I] | share of felt impingement shed per tick | `M4.C.1` |
| `functional_level_floor` | 1.0 | [I] | functional-level point; also floors the symptom threshold (`M1.A.6`, revision 24) | `M4.C.1` |
| `acute_decay_rate` | 0.2 | [I] | fraction of excess per tick | `M1.A.8` |
| `route_damping` | 0.5 | [I] | gain per neutral third | `M1.F.3` |
| `hardening_run_length` | 3 | [I] | consecutive moves | `M4.G.1` |
| `bond_energy_decay_rate` | 0.0 | [I] | fraction per tick | `M1.B.4` |
| `triangle_activity_window` | 4 | [I] | fast ticks | `M1.C.3` |
| `defence_threshold` | 15.0 | [I] | points of excess anxiety | `M4.C.2` |
| `witness_weight` | 0.5 | [I] | share of an overheard exchange | `M4.C.9` |
| `speaker_echo_gain` | 0.2 | [I] | share of what was addressed | `M4.C.7` |
| `reappraisal_window` | 4 | [I] | fast ticks | `M4.C.6` |
| `attention_gain` | 0.5 | [I] | relative change in appraisal | `M4.C.8` |
| `perspective_anxiety_scale` | 10.0 | [I] | points of excess anxiety | `M4.C.4` |
| `calm_transfer_rate` | 0.05 | [I] | share of the anxiety gap per unit conductance | `M4.C.10` |
| `symptom_leak_rate` | 0.1 | [I] | share of the integral per tick | `M4.C.3a` |
| `symptom_threshold_gain` | 1.5 | [I] | integral units per functional-level point | `M7.D.1` |
| `symptom_rearm_fraction` | 0.5 | [I] | share of the threshold | `M7.D.1` |
| `symptom_event_intensity` | 100.0 | [I] | event intensity | `M7.D.1` |
| `reactive_rate` | 0.2 | [I] | share of the gap per tick | `M1.A.19` |
| `investment_leak_rate` | 0.1 | [I] | share per tick | `M1.B.8` |
| `initial_impingement_scale` | 0.8 | [I] | impingement at basic level 0 | `M1.A.9` |
| `outside_ness_rate` | 0.1 | [I] | share of the gap per tick | `M1.A.9` |
| `hollow_gain` | 0.5 | [I] | share of contact hollowed at full inward impingement | `M5.F.1` |
| `assault_gain` | 0.5 | [I] | impingement added per unit outward impingement | `M5.F.1` |
| `outside_ness_threshold_outward` | 0.5 | [I] | outward impingement | `M5.C.1` |
| `outside_ness_threshold_inward` | 0.5 | [I] | inward impingement | `M5.C.1` |
| `belief_rate` | 0.3 | [I] | share of the gap to an observation per tick, at full fidelity | `M9.8` |
| `rejection_cutoff_gain` | 0.5 | [I] | rejection per unit strength of a delivered CUTOFF | `M1.B.3c` |
| `rejection_refusal_gain` | 0.1 | [I] | rejection per unit strength of a refused move | `M1.B.3c` |
| `rejection_relief` | 0.25 | [I] | rejection removed per unit strength of an accepted REDUCE_CUTOFF | `M1.B.3c` |
| `refusal_cost_gain` | 1.0 | [I] | anxiety per unit strength × conductance × steepness | `M1.B.3c` |
| `declared_rejection` | 1.0 | [I] | rejection, each member of a declared cut-off tie | `M1.B.3c` |
| `rejection_refusal_threshold` | 0.3 | [I] | rejection above which a move is refused | `M1.B.3c` |
| `rejection_fade` | 0.05 | [I] | rejection lost per fast tick | `M1.B.3c` |
| `estimator_window` | 3 | [I] | slow ticks | `M1.A.4k` |
| `estimator_breadth` | 3 | [I] | distinct kinds of nodal event | `M1.A.4k` |
| `estimator_sd_weight` | 1.0 | [I] | standard deviations | `M1.A.4k` |
| `estimator_rate` | 0.5 | [I] | share of the evidence per slow tick | `M1.A.4k` |
| `tie_wear_rate` | 0.05 | [I] | share of conductance per slow tick at mean chronic anxiety 100 | `M7.E.4` |
| `wear_offset_per_act` | 0.1 | [I] | share of a year's wear per STAY-IN-CONTACT | `M7.E.4` |
| `perspective_channel_gain` | 0.5 | [I] | share of the automatic channel's weight full perspective moves to the self-directed | `M4.D.1a` |
| `self_practice_gain` | 0.001 | [I] | share of the distance to 1, per self-directed act under load that worked (revision 19) | `M1.E.7` |
| `stressor_persistence` | 0.5 | [I] | share of the arrival's load, per later week | `M1.F.6` |
| `response_up_gain` | 0.5 | [I] | functioning per unit load (load / 100), at basic level 100 | `M1.A.5e` |
| `response_down_gain` | 0.5 | [I] | functioning per unit load (load / 100), at basic level 0 | `M1.A.5e` |
| `response_relaxation_rate` | 0.02 | [I] | share of the stress-moved swing, and of the system's adjustment (`M1.E.7`, revision 20), shed per calm week | `M1.A.5e` |
| `stress_floor_below_basic` | 15.0 | [I] | points of functional level below basic_level | `M1.A.5e` |
| `practice_window` | 4 | [I] | fast ticks; also how long a stance holds (`M1.B.5`, revision 21) | `M1.E.7` |
| `perspective_steer_gain` | 0.5 | [I] | share of a steered act's weight at full perspective | `M4.D.1a` |
| `system_adjust_gain` | 1.0 | [I] | points of functional level the other is up by while a held stance works: a level, renewed, not stacked (revision 20) | `M1.E.7` |
| `soothing_review_every` | 12 | [I] | fast ticks | `M4.D.6` |
| `soothing_review_penalty` | 0.5 | [I] | share of the learned value | `M4.D.6` |
| `usual_level_rate` | 0.02 | [I] | share per week: a person's usual level follows their excess anxiety over about a year | `M1.A.5e` |
| `constitution_strength` | 0.5 | [I] | weight of the constitutional channel prior against a family focus; a person's own value overrides it (revision 24) | `M1.A.11c` |
| `channel_protection` | 1.0 | [I] | divisor per threshold of load carried in the other channels (revision 24) | `M7.D.2c` |
| `lock_in_gain` | 0.2 | [I] | scale of the share of a partner's excess above usual an active symptom holds each week (revision 24) | `M7.D.2a` |
| `attribution_rate` | 0.05 | [I] | share per unit of fidelity by which a person's attribution moves toward what a move places (revision 25) | `M9.6` |
| `focus_rate` | 0.05 | [I] | weekly decay of the family's focus on a person's channel, and what one anxious act adds (revision 25) | `M1.A.11c` |
| `marital_distance_fraction` | 0.25 | [I] | share of the felt contact optimum below which, for both spouses, marital distance is high (revision 25) | `M5.C.1` |
| `belief_appraisal_gain` | 1.0 | [I] | a witness's appraisal is scaled by 1 + this × its believed tension of the tie (revision 25) | `M9.7` |
| `pseudo_self_relaxation_rate` | 0.05 | [I] | weekly share of a pair's opposite pseudo-self that settles back once their exchange stops (revision 27) | `M6.I.4` |
| `exchange_window` | 8 | [I] | weeks without over/under-functioning between a pair before their pseudo-self settles (revision 27) | `M6.I.4` |
| `acting_age_years` | 10 | [I] | year of age (the owner's, 2026-10-13) | `M7.C.1f` |
| `development_start_age_years` | 5 | [I] | year of age (the owner's, 2026-10-13) | `M7.E.1f` |
| `development_end_age_years` | 20 | [I] | year of age (the owner's, 2026-10-13) | `M7.E.1f` |
| `development_band` | 0.02 | [I] | share of the parents' level (the owner's: level x .02) | `M7.E.1f` |
| `development_rate` | 0.1 | [I] | basic-level points per point of mean functioning above basic, per year | `M7.E.1f` |
| `witnessed_gain` | 0.5 | [I] | chronic-anxiety points per point of the year's mean load | `M7.B.1a` |
| `launch_age_years` | 20 | [I] | year of age | `M7.C.1g` |
| `working_age_years` | 16 | [I] | year of age | `M5.B.7` |
| `find_work_rate` | 0.1 | [I] | probability per week at full pressure | `M5.B.7` |
| `find_work_pressure_scale` | 10.0 | [I] | anxiety points above the usual level for full pressure | `M5.B.7` |
| `grief_scale` | 20.0 | [I] | stressor intensity at bond energy 100 | `M6.3g` |
| `grief_min_weeks` | 52 | [I] | fast ticks | `M6.3g` |
| `grief_extra_weeks` | 104 | [I] | fast ticks at bond energy 100 | `M6.3g` |
| `income_loss_intensity` | 15.0 | [I] | stressor intensity | `M6.3h` |
| `income_shortfall_share` | 0.2 | [I] | share of the loss that stays (the owner's: 20% less income) | `M6.3h` |
| `mortality_a` | 0.0000657 | [I] | annual hazard at age 0 | `M7.C.1j` |
| `mortality_b` | 0.085 | [I] | per year of age | `M7.C.1j` |
| `budget_per_undifferentiation` | 1.49 | [I] | budget units per point of the couple's distance below 100 | `M7.C.1h` |
| `coach_start_age_years` | 35 | [I] | year of age (the owner's, 2026-10-13) | `M2.A.0k` |
| `coach_retirement_age_years` | 68 | [I] | year of age (the owner's, 2026-10-13) | `M2.A.0k` |
| `new_tie_investment_share` | 0.2 | [I] | share of the person's existing investment | `M7.C.1a` |
| `birth_focus_gain` | 0.1 | [I] | share added to the parent's investment in the child per situation present | `M7.E.1c` |
| `death_transfer_share` | 1.0 | [I] | share of the dead person's acute anxiety that goes to the living; 1 is the rule; only `M11.C.37`'s mutant sets it lower | `M6.3b` |
| `lock_in_turn` | 3.0 | [I] | severity (load ÷ threshold) past which lock-in releases instead of holding; peak at half of it (revision 24) | `M7.D.2d` |
| `self_channel_exponent` | 2.0 | [I] | exponent on functional_level / 100 | `M4.D.1a` |
| `policy_temperature` | 1.0 | [I] | learned-value units | `M4.D.1` |
| `anxiety_band_low` | 5.0 | [I] | points of excess anxiety | `M4.D.3` |
| `anxiety_band_high` | 15.0 | [I] | points of excess anxiety | `M4.D.3` |
| `capacity_level_per_layer` | 20.0 | [I] | functional-level points per layer | `M4.D.3a` |
| `competing_urge_gain` | 0.5 | [I] | points of acute anxiety at maximal entropy | `M4.D.1d` |
| `withhold_investment_gain` | 0.2 | [I] | share of attention per unit scaled intensity | `M4.D.1b` |
| `loaded_tie_threshold` | 0.2 | [I] | deviation units | `M4.D.3b` |
| `policy_intensity` | 100.0 | [I] | intensity units | `M4.D.1` |
| `learning_rate` | 0.2 | [I] | share of the gap to the signal per update | `M4.D.6` |
| `credit_horizon` | 3 | [I] | fast ticks | `M4.D.6b` |
| `credit_discount` | 0.7 | [I] | weight per tick of age | `M4.D.6` |
| `cross_person_weight` | 0.5 | [I] | weight on the others' mean relief | `M4.D.6e` |
| `habituation_rate` | 0.7 | [I] | relief kept per earlier repetition | `M4.G.3` |
| `habituation_window` | 8 | [I] | fast ticks | `M4.G.3` |
| `prepare_ticks` | 4 | [I] | fast ticks | `M5.D.2a` |
| `rehearsal_rate` | 0.05 | [I] | share of each axis per tick | `M1.A.9` |
| `assertion_perspective_threshold` | 0.3 | [I] | systems perspective | `M5.F.4` |
| `anger_threshold` | 0.3 | [I] | too-much deviation | `M5.D.4` |
| `assertion_gain` | 0.5 | [I] | impingement per unit scaled intensity | `M5.F.4` |
| `assertion_evidence_gain` | 0.05 | [I] | outward impingement per assertion | `M5.F.2a` |
| `opposition_window` | 3 | [I] | fast ticks | `M5.E.3` |
| `hold_gain` | 0.3 | [I] | excess anxiety per point of functional level at full efficacy | `M5.D.3` |
| `stall_limit` | 4 | [I] | fast ticks | `M5.D.4` |
| `hold_window` | 6 | [I] | fast ticks | `M5.D.2` |
| `pull_up_rate` | 0.2 | [I] | share of the level gap | `M5.D.5` |
| `exchange_gain` | 0.5 | [I] | functional-level points | `M5.D.7a` |
| `triangle_floor_decrement` | 0.1 | [I] | share of routing capacity | `M1.C.5` |
| `respect_gain` | 0.2 | [I] | share of each axis | `M5.E.8` |
| `debit_gain` | 0.3 | [I] | felt contact per unit scaled intensity | `M5.E.7` |
| `session_interval_weeks` | 8 | [I] | fast ticks | `M1.E.8` |
| `landing_rate` | 0.15 | [I] | probability per contact at full coach quality | `M1.E.7` |
| `delayed_view_bonus` | 1.0 | [I] | added share of the landing rate | `M1.E.7c` |
| `binder_failure_fraction` | 0.8 | [I] | share of the symptom threshold | `M1.E.7` |
| `contact_optimum` | 4 | [I] | contacts in the window | `M1.E.8` |
| `contact_window` | 26 | [I] | fast ticks | `M1.E.8` |
| `perspective_gain` | 0.1 | [I] | share of the gap to 1 | `M1.E.7` |
| `delayed_view_weeks` | 26 | [I] | fast ticks | `M16.D.2` |
| `contact_excess_exponent` | 2.0 | [I] | exponent | `M1.E.8` |
| `sink_window` | 8 | [I] | fast ticks | `M1.D.1` |
| `sink_rate` | 0.1 | [I] | share of the gap per tick | `M1.D.1` |
| `exchange_budget_reduction` | 2.0 | [I] | budget units per completed exchange | `M6.I.1` |
| `pattern_window` | 4 | [I] | fast ticks | `M5.A.1a` |
| `pattern_min_acts` | 3 | [I] | acts on the tie in the window | `M5.A.1a` |
| `pattern_fixed_activations` | 3 | [I] | consecutive activations | `M5.A.1a` |
| `pattern_projection_hold` | 12 | [I] | fast ticks without a projection act | `M5.A.1a` |
| `ensemble_block` | 50 | [I] | seeds per block | `M17.A.1` |
| `ensemble_cap` | 500 | [I] | seeds | `M17.A.1` |
| `ensemble_precision` | 0.25 | [I] | pooled seed-to-seed sd of the two arms | `M17.A.1` |
| `ensemble_margin` | 0.1 | [I] | baseline seed-to-seed sd | `M17.A.4` |
| `ensemble_alpha` | 0.05 | [I] | family-wise error | `M11.4e` |
| `fallback_flag_rate` | 0.2 | [I] | share of selections | `M11.D.18` |
| `equivalence_margin` | 0.5 | [I] | baseline seed-to-seed sd | `M11.4a` |
| `distance_binding_rate` | 0.3 | [I] | share of excess bound per unit scaled intensity | `M1.D.2a` |
| `triangle_transfer_rate` | 0.3 | [I] | share of each insider's excess passed per unit scaled intensity | `M1.C.1` |
| `outsider_positional_gain` | 0.5 | [I] | positional anxiety per unit transferred | `M1.C.1` |
| `balance_push_gain` | 0.2 | [I] | balance per unit scaled intensity | `M1.B.5` |
| `balance_settle_rate` | 0.05 | [I] | share of the gap to the pole per tick | `M1.B.5` |
| `balance_harden_rate` | 0.02 | [I] | share of the gap per tick | `M1.B.6` |
| `reversal_asymmetry` | 0.5 | [I] | share of an upward push lost at a fully hardened habit | `M1.B.7` |
| `pseudo_self_transfer_gain` | 2.0 | [I] | functional-level points per unit scaled intensity | `M6.I.4` |
| `involvement_membership_threshold` | 0.5 | [I] | involvement units | `M1.A.12` |
