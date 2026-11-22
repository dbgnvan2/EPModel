# Run trace

> **What this is.** Output of a simulation that runs one theory's stated mechanisms
> over an invented family. It is a consistency engine for that theory — *if Bowen's
> account is right, what follows for a family shaped like this?* — not a measurement,
> not a prediction, and not a "virtual family" in the predictive sense (spec `M11.F.10`).
> It does not concern any real family or person, and nothing here is advice (`M11.F.9`).
> Every number in it comes from invented constants (`[I]`); only differences between
> two runs that share them carry meaning (`M0.4`). This run is not scripted: each person's outcome is
> selected by the policy, and the automatic channel learns from felt relief (Phase C).

| Header | |
|---|---|
| seed | 7 |
| config hash | `fd662a6911abc860412c247a88c1a438021caceacf07e746fe193f826ab66c83` |
| spec revision | 2.0, revision 67 |
| family instance | phase_c |
| activation | synchronous_activation v1 (synchronous) |
| visibility | household_conductance_visibility v1 |
| constants frozen | 2026-10-07 |
| constants changed since freeze | acting_age_years, attention_load_gain, attention_load_scale, attribution_rate, belief_anxiety_scale, belief_appraisal_gain, belief_cue_rate, belief_leak_gain, birth_focus_gain, budget_per_undifferentiation, channel_protection, coach_retirement_age_years, coach_side_window, coach_start_age_years, conflict_binding_rate, conflict_capacity_cost, constitution_strength, cue_approval_contact_weight, cue_approval_impingement_weight, cue_attention_contact_weight, cue_attention_impingement_weight, cue_bias_anxiety_gain, cue_bias_belief_gain, cue_bias_floor, cue_bias_level_gain, cue_distress_contact_weight, cue_distress_impingement_weight, cue_expectations_contact_weight, cue_expectations_impingement_weight, cue_face_share, cue_leak_distress_share, cue_leak_expectations_share, cue_leak_gain, cue_posture_share, cue_sensitivity_floor, cue_sensitivity_level_gain, cue_spread_floor, cue_spread_level_gain, cue_tone_share, cue_uncertainty_gain, cue_uncertainty_window, cutoff_decay_weeks, cutoff_floor_share, cutoff_stock_cap, cutoff_threat_weight, danger_gain, danger_threshold, danger_window, death_transfer_share, declared_rejection, development_band, development_band_projection, development_band_support, development_end_age_years, development_rate, development_start_age_years, estimator_breadth, estimator_rate, estimator_sd_weight, estimator_window, exchange_window, find_work_pressure_scale, find_work_rate, focus_compound_gain, focus_rate, focus_reinforce_gain, focus_steer_gain, force_anxiety_gain, grief_extra_weeks, grief_min_weeks, grief_weight, importance_max, income_loss_intensity, income_shortfall_share, launch_age_years, lcu_scale, line_importance, line_tendency_max, lock_in_gain, lock_in_turn, marital_distance_fraction, mortality_a, mortality_b, new_tie_investment_share, nodal_spike_intensity, nonverbal_weight_anxiety_gain, nonverbal_weight_base, nonverbal_weight_cap, origin_cutoff_anxiety_gain, origin_fusion_gain, parent_pull_rate, pattern_min_acts, pattern_projection_hold, pattern_window, perceived_weakness_gain, perspective_channel_gain, perspective_steer_gain, positive_bridge_dissipation, practice_window, pseudo_self_relaxation_rate, reactive_cutoff_gain, reactive_extra_max, reactive_half_anxiety, reciprocal_contact_share, refusal_cost_gain, rejection_cutoff_gain, rejection_fade, rejection_refusal_gain, rejection_refusal_threshold, rejection_relief, response_down_gain, response_relaxation_rate, response_up_gain, self_practice_gain, social_cues_enabled, soothing_review_every, soothing_review_penalty, stress_floor_below_basic, stressor_persistence, system_adjust_gain, tie_discount, tie_repair_steps, tie_wear_rate, unmet_expectation_weight, usual_level_rate, wear_offset_per_act, witness_window, witnessed_cutoff_gain, witnessed_gain, working_age_years |

The standing load, the decay toward each person's floor, the relaxation of felt contact, symptom accumulation, a symptom's weekly hold on the family's anxiety (lock-in; what it gives back is shown), investment, the reactive detectors, outside-ness, belief writes (logged apart from ground truth), attribution, the family's focus and borrowed pseudo-self settling back run every week for everyone and are not listed line by line. Effects are shown beside the event that caused them.

| Week | Who | Move | Toward | Witnesses | What it did |
|---|---|---|---|---|---|
| 0 | Ana | WITHHOLD | Bruno | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 0 | Bruno | CUTOFF | Ana | — | arrives week 1; anxiety Ana +1.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 0 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 1; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.3, Marta -0.3 |
| 0 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | arrives week 1; anxiety Marta -0.1, Nadia +0.6 (witness), Pia -0.1, Ravi +1.3 (witness); Pia takes 2.0 of functioning from the other |
| 0 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 1; anxiety Marta -0.3, Nadia -0.4, Pia +0.6 (witness), Ravi +1.3 (witness); Marta left outside: anxiety Marta +0.5, Nadia -0.3 (of which 0.2 the outsider's own) |
| 0 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 1; anxiety Marta +1.4 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +3.8; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia takes 2.0 of functioning from the other |
| 0 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 1; anxiety Marta -0.6, Nadia +1.2 (witness), Pia +1.1 (witness), Ravi -0.4; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1; Marta left outside: anxiety Marta +1.1, Ravi -0.7 (of which 0.4 the outsider's own) |
| 0 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 2; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +2.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Sofia takes 1.9 of functioning from the other |
| 1 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 1 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 1 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 1 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 1 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 1 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 1 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.3 |
| 1 | Bruno | CUTOFF | Ana | — | arrives week 2; anxiety Ana +1.5, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 1 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 2; anxiety Marta +0.6, Nadia +0.9, Pia +0.6 (witness), Ravi +1.5 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3; Marta rejects the other on Marta–Nadia (rejection +0.53; the tie stays open); impingement removed at once, anxiety Marta -0.6 |
| 1 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 2; anxiety Marta +1.4 (witness), Nadia -0.4, Pia +0.6 (witness), Ravi -0.5; Ravi left outside: anxiety Nadia -0.4, Ravi +0.6 (of which 0.2 the outsider's own) |
| 1 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 2; anxiety Marta +1.8 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +3.6; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; 1.0 of the sender's anxiety bound into Pia–Ravi |
| 1 | Ravi | WITHHOLD | Nadia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 1 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 3; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +3.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; 0.4 of the sender's anxiety bound into Ravi–Sofia |
| 2 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 2 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 3; anxiety Ana -0.6, Marta -0.7, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.3, Marta -0.3 |
| 2 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.11); the sender bears the refusal: anxiety Bruno +2.9 |
| 2 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 3; anxiety Dr Halim +0.1, Marta +0.2, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.2 (witness) |
| 2 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | arrives week 3; anxiety Ana +2.1, Marta +0.4, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); 4.0 of the sender's anxiety bound into Ana–Marta |
| 2 | Nadia | REDUCE_CUTOFF | Marta | Pia, Ravi | arrives week 3; anxiety Marta +1.4, Nadia +0.5, Pia +0.6 (witness), Ravi +1.6 (witness); Marta's rejection -0.25 |
| 2 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 3; anxiety Marta +2.0 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +2.6; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia takes 2.2 of functioning from the other |
| 2 | Ravi | STAY-IN-CONTACT | Nadia | Marta, Pia | arrives week 3; anxiety Marta +1.4 (witness), Nadia +0.3, Pia +0.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.1, Ravi +0.1 |
| 2 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 4; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +2.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 1.1 of the sender's anxiety bound into Ravi–Sofia |
| 3 | — | (system) | — | — | Ana–Marta: a positive contact repairs the tie; held anxiety -1.3 |
| 3 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 4; anxiety Ana +0.4, Marta +2.6, Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Ana +0.2, Marta -0.2; 2.7 of the sender's anxiety bound into Ana–Marta |
| 3 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 3 | Marta | STAY-IN-CONTACT | Ravi | Nadia, Pia | arrives week 4; anxiety Marta -0.0, Nadia +1.3 (witness), Pia +1.1 (witness), Ravi -0.9; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4 |
| 3 | Nadia | CONFLICT | Marta | Pia, Ravi | arrives week 4; anxiety Marta +6.3, Nadia +0.6, Pia +0.7 (witness), Ravi +2.4 (witness); 3.2 of the sender's anxiety bound into Marta–Nadia |
| 3 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 4; anxiety Marta +1.7 (witness), Nadia +0.6, Pia +0.6 (witness), Ravi +2.1; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Nadia rejects the other on Nadia–Ravi (rejection +0.58; the tie stays open); impingement removed at once, anxiety Nadia -0.3 |
| 3 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 4; anxiety Marta +3.3, Nadia +0.8 (witness), Pia +0.5, Ravi +2.1 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4 |
| 3 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 4; anxiety Marta +2.7 (witness), Nadia +1.3 (witness), Pia +4.5, Ravi +0.8; 6.7 of the sender's anxiety bound into Pia–Ravi |
| 3 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 4; anxiety Marta +1.6, Nadia +1.8 (witness), Pia +1.5 (witness), Ravi +0.8; Marta left outside: anxiety Marta +3.7, Ravi -2.5 (of which 1.2 the outsider's own) |
| 3 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 5; anxiety Marta +1.0 (witness), Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +3.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia takes 2.1 of functioning from the other |
| 4 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.6 (witness), Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +3.7 |
| 4 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +3.1, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +1.9 (witness) |
| 4 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 4 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.11); the sender bears the refusal: anxiety Ana +2.4 |
| 4 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.11); the sender bears the refusal: anxiety Bruno +3.1 |
| 4 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 5; anxiety Marta +0.8, Nadia +1.6, Pia +1.0 (witness), Ravi +3.2 (witness); Marta rejects the other on Marta–Nadia (rejection +0.75; the tie stays open); impingement removed at once, anxiety Marta -4.9 |
| 4 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 5; anxiety Marta +0.8, Nadia +1.1 (witness), Pia +1.1, Ravi +2.9 (witness); Marta rejects the other on Marta–Pia (rejection +0.75; the tie stays open); impingement removed at once, anxiety Marta -2.9 |
| 4 | Nadia | DETRIANGLE | Ravi | Marta, Pia | arrives week 5; anxiety Marta +1.6 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +1.0 |
| 4 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 5; anxiety Marta +2.7 (witness), Nadia +1.2 (witness), Pia +0.6, Ravi +2.7; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; Ravi left outside: anxiety Pia -1.8, Ravi +2.7 (of which 0.9 the outsider's own) |
| 4 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 5; anxiety Marta +2.8 (witness), Nadia +1.3 (witness), Pia +1.8, Ravi +0.9 |
| 4 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 6; anxiety Marta +1.3 (witness), Nadia +1.2 (witness), Pia +1.0 (witness), Ravi +0.6, Sofia +0.4; Ravi rejects the other on Ravi–Sofia (rejection +0.69; the tie stays open); impingement removed at once, anxiety Ravi -7.1 |
| 4 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 6; anxiety Marta +1.0 (witness), Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +1.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; 1.7 of the sender's anxiety bound into Ravi–Sofia |
| 5 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 5 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 5 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 5 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +0.4, Marta +3.5, Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana takes 2.1 of functioning from the other |
| 5 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.08); the sender bears the refusal: anxiety Bruno +3.2 |
| 5 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +3.9, Marta +0.5, Nadia +1.2 (witness), Pia +1.0 (witness), Ravi +1.7 (witness); 9.3 of the sender's anxiety bound into Ana–Marta |
| 5 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | arrives week 6; anxiety Marta +0.8, Nadia +0.5, Pia +1.0 (witness), Ravi +3.5 (witness); Nadia takes 2.7 of functioning from the other |
| 5 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 6; anxiety Marta +1.8 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +1.6; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 5 | Pia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 5 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 6; anxiety Marta +1.5, Nadia +1.1 (witness), Pia +0.6, Ravi +3.1 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; Pia rejects the other on Marta–Pia (rejection +0.68; the tie stays open); impingement removed at once, anxiety Pia -0.6 |
| 5 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Ravi +5.1; anxiety Marta +2.7 (witness), Pia +1.0 (witness) |
| 5 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | arrives week 6; anxiety Marta +2.2, Nadia +2.2 (witness), Pia +1.8 (witness), Ravi +1.0; Marta holds a position: no functioning changes hands |
| 5 | Sofia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 5 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 6, 7; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.7, Nadia +2.5, Pia +2.0; anxiety Sofia +1.4 |
| 6 | (from outside) | OWN_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +1.9; still weighing this week: anxiety Ravi +0.9; still weighing this week: anxiety Ravi +1.2; still weighing this week: anxiety Ravi +1.2; still weighing this week: anxiety Ravi +1.2; still weighing this week: anxiety Ravi +1.2; still weighing this week: anxiety Ravi +1.2; still weighing this week: anxiety Ravi +1.1; still weighing this week: anxiety Ravi +1.1; still weighing this week: anxiety Ravi +1.1; still weighing this week: anxiety Ravi +1.1; still weighing this week: anxiety Ravi +1.1 |
| 6 | (from outside) | FAMILY_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +0.8, Nadia +0.3 (witness), Pia +0.3 (witness), Ravi +0.5 (witness); still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.6; still weighing this week: anxiety Marta +0.6; still weighing this week: anxiety Marta +0.6; still weighing this week: anxiety Marta +0.6 |
| 6 | (from outside) | FAMILY_ILLNESS | Nadia | Marta, Ravi | a 12-week spell; anxiety Marta +0.3 (witness), Nadia +0.6, Ravi +0.4 (witness); still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3 |
| 6 | (from outside) | FAMILY_ILLNESS | Pia | Marta, Ravi | a 12-week spell; anxiety Marta +0.3 (witness), Pia +0.5, Ravi +0.4 (witness); still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2 |
| 6 | (from outside) | FAMILY_ILLNESS | Sofia | — | a 12-week spell; anxiety Sofia +0.4; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2 |
| 6 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 6 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 6 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.12); the sender bears the refusal: anxiety Ana +2.5 |
| 6 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.3 |
| 6 | Marta | DETRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.1; anxiety Nadia +0.8 (witness), Ravi +2.4 (witness) |
| 6 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 7; anxiety Marta +0.9, Nadia +4.2, Pia +1.1 (witness), Ravi +3.8 (witness) |
| 6 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 7; anxiety Marta +3.2 (witness), Nadia +0.7, Pia +1.0 (witness), Ravi +7.0; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Nadia takes 2.9 of functioning from the other |
| 6 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 7; anxiety Marta +3.5 (witness), Nadia +1.2 (witness), Pia +0.6, Ravi +2.5; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Ravi takes 0.6 of functioning from the other |
| 6 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Pia +3.0; anxiety Nadia +1.1 (witness), Ravi +3.5 (witness) |
| 6 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 7; anxiety Dr Halim +0.3, Marta +1.9 (witness), Nadia +1.1 (witness), Pia +1.0 (witness), Ravi +0.3 |
| 6 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | arrives week 7; anxiety Marta +3.8 (witness), Nadia +1.4 (witness), Pia +1.8, Ravi +1.1; Pia takes 3.2 of functioning from the other |
| 6 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.5; anxiety Marta +1.2 (witness), Nadia +0.8 (witness), Pia +0.8 (witness) |
| 6 | Marta | SYMPTOM_ONSET | Ana, Dr Halim, Nadia, Pia, Ravi | — | arrives week 7; the nodal event re-opens the cut tie Ana–Bruno: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, physical channel; anxiety Ana +1.6, Dr Halim +0.5, Nadia +2.3, Pia +2.0, Ravi +3.3 |
| 7 | (from outside) | OWN_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +1.7, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); still weighing this week: anxiety Marta +0.9; still weighing this week: anxiety Marta +0.9; still weighing this week: anxiety Marta +1.1; still weighing this week: anxiety Marta +1.1; still weighing this week: anxiety Marta +1.1; still weighing this week: anxiety Marta +1.1; still weighing this week: anxiety Marta +1.1; still weighing this week: anxiety Marta +1.1; still weighing this week: anxiety Marta +1.1; still weighing this week: anxiety Marta +1.1; still weighing this week: anxiety Marta +1.1 |
| 7 | (from outside) | FAMILY_ILLNESS | Ana | — | a 12-week spell; anxiety Ana +0.5; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2 |
| 7 | (from outside) | FAMILY_ILLNESS | Nadia | Marta, Ravi | a 12-week spell; anxiety Marta +0.4 (witness), Nadia +0.5, Ravi +0.4 (witness); still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3 |
| 7 | (from outside) | FAMILY_ILLNESS | Pia | Marta, Ravi | a 12-week spell; anxiety Marta +0.4 (witness), Pia +0.5, Ravi +0.4 (witness); still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2 |
| 7 | (from outside) | FAMILY_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +0.4 (witness), Nadia +0.3 (witness), Pia +0.3 (witness), Ravi +0.9; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6 |
| 7 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 7 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 7 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 8; anxiety Ana +0.4, Marta +5.0, Nadia +1.0 (witness), Pia +0.9 (witness), Ravi +1.9 (witness); calmer sender takes some anxiety: Ana +1.0, Marta -1.0; Ana takes 2.5 of functioning from the other |
| 7 | Bruno | CUTOFF | Ana | — | arrives week 8; anxiety Ana +0.9, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.07; the tie stays open) |
| 7 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 8; anxiety Marta +1.0, Nadia +2.0, Pia +1.2 (witness), Ravi +5.3 (witness) |
| 7 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 8; anxiety Marta +2.3 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +2.5; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 7 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.3; anxiety Pia +1.1 (witness), Ravi +5.0 (witness) |
| 7 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 8; anxiety Marta +3.7 (witness), Nadia +1.2 (witness), Pia +0.6, Ravi +7.0; calmer sender takes some anxiety: Pia +1.3, Ravi -1.3; Pia takes 0.2 of functioning from the other |
| 7 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Pia +2.9; anxiety Nadia +1.1 (witness), Ravi +4.7 (witness) |
| 7 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 8; anxiety Marta +4.3 (witness), Nadia +1.5 (witness), Pia +1.9, Ravi +1.4; 41.6 of the sender's anxiety bound into Pia–Ravi |
| 7 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | arrives week 9; anxiety Marta +2.3 (witness), Nadia +1.4 (witness), Pia +1.3 (witness), Ravi +0.9, Sofia +2.1 |
| 7 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.5; anxiety Marta +1.5 (witness), Nadia +0.9 (witness), Pia +0.8 (witness) |
| 8 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.9 (witness), Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +5.0 |
| 8 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +3.7, Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +2.5 (witness) |
| 8 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 8 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 8 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.4, Marta +3.9, Nadia +1.0 (witness), Pia +0.9 (witness), Ravi +2.0 (witness); calmer sender takes some anxiety: Ana +1.5, Marta -1.5 |
| 8 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.5 |
| 8 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Dr Halim +0.1, Marta +1.5 (witness), Nadia +0.7 (witness), Pia +0.6 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Ravi -0.7 |
| 8 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.8, Marta +0.7, Nadia +1.4 (witness), Pia +1.2 (witness), Ravi +2.8 (witness); Ana takes 3.5 of functioning from the other |
| 8 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.17); the sender bears the refusal: anxiety Marta +5.9; anxiety Nadia +1.4 (witness), Ravi +5.6 (witness) |
| 8 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 9; anxiety Marta +1.2, Nadia +2.0, Pia +1.3 (witness), Ravi +5.9 (witness); Nadia left outside: anxiety Marta -7.1, Nadia +10.6 (of which 3.5 the outsider's own) |
| 8 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 9; anxiety Marta +4.1 (witness), Nadia +0.7, Pia +1.1 (witness), Ravi +4.3; 28.9 of the sender's anxiety bound into Nadia–Ravi |
| 8 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.16); the sender bears the refusal: anxiety Nadia +3.4; anxiety Pia +1.2 (witness), Ravi +5.3 (witness) |
| 8 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Pia +3.0; anxiety Nadia +1.2 (witness), Ravi +5.0 (witness) |
| 8 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 9; anxiety Marta +4.3 (witness), Nadia +1.3 (witness), Pia +0.6, Ravi +3.7; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; 21.4 of the sender's anxiety bound into Pia–Ravi |
| 8 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 9; anxiety Marta +4.3 (witness), Nadia +1.3 (witness), Pia +2.3, Ravi +1.3; Ravi takes 0.0 of functioning from the other |
| 8 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | arrives week 10; anxiety Marta +2.4 (witness), Nadia +1.3 (witness), Pia +1.1 (witness), Ravi +0.8, Sofia +1.2 |
| 8 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.5; anxiety Marta +1.7 (witness), Nadia +0.9 (witness), Pia +0.8 (witness) |
| 8 | Nadia | SYMPTOM_ONSET | Marta, Ravi | Pia | arrives week 9; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, mental channel; anxiety Marta +3.4, Pia +0.8 (witness), Ravi +4.2 |
| 9 | (from outside) | OWN_ILLNESS | Nadia | Marta, Ravi | a 12-week spell; anxiety Marta +0.9 (witness), Nadia +1.2, Ravi +1.2 (witness); still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.7; still weighing this week: anxiety Nadia +0.7; still weighing this week: anxiety Nadia +0.7; still weighing this week: anxiety Nadia +0.7; still weighing this week: anxiety Nadia +0.8; still weighing this week: anxiety Nadia +0.8 |
| 9 | (from outside) | FAMILY_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +0.8, Nadia +0.3 (witness), Pia +0.2 (witness), Ravi +0.5 (witness); still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5 |
| 9 | (from outside) | FAMILY_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +0.4 (witness), Nadia +0.3 (witness), Pia +0.2 (witness), Ravi +1.0; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5 |
| 9 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 9 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 9 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 10; anxiety Ana +0.3, Marta +3.7, Nadia +1.1 (witness), Pia +0.9 (witness), Ravi +2.1 (witness); calmer sender takes some anxiety: Ana +1.4, Marta -1.4 |
| 9 | Bruno | CUTOFF | Ana | — | arrives week 10; anxiety Ana +1.1, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 9 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 10; anxiety Marta +1.4, Nadia +2.0, Pia +1.4 (witness), Ravi +6.4 (witness); Marta takes 0.3 of functioning from the other |
| 9 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 10; anxiety Marta +1.4, Nadia +2.6 (witness), Pia +2.3 (witness), Ravi +4.3; 48.9 of the sender's anxiety bound into Marta–Ravi |
| 9 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 10; anxiety Marta +4.6 (witness), Nadia +0.7, Pia +1.1 (witness), Ravi +4.0; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6 |
| 9 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.06); the sender bears the refusal: anxiety Nadia +3.3; anxiety Pia +1.1 (witness), Ravi +5.2 (witness) |
| 9 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 10; anxiety Marta +4.4 (witness), Nadia +1.2 (witness), Pia +0.5, Ravi +3.8; calmer sender takes some anxiety: Pia +1.9, Ravi -1.9 |
| 9 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | arrives week 10; anxiety Marta +5.9 (witness), Nadia +1.6 (witness), Pia +1.7, Ravi +1.6 |
| 9 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 10; anxiety Marta +5.7 (witness), Nadia +1.7, Pia +1.4 (witness), Ravi +1.6; Ravi rejects the other on Nadia–Ravi (rejection +0.93; the tie stays open); impingement removed at once, anxiety Ravi -11.9 |
| 9 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 10; anxiety Marta +4.0, Nadia +2.6 (witness), Pia +2.3 (witness), Ravi +1.6; calmer sender takes some anxiety: Marta -0.0, Ravi +0.0; 40.6 of the sender's anxiety bound into Marta–Ravi |
| 9 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.9 (witness), Nadia +1.0 (witness), Pia +0.9 (witness) |
| 9 | — | (system) | — | — | Marta's symptom gives back 4.0 (Ana +0.8, Nadia +1.0, Pia +1.0, Ravi +1.2) |
| 9 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -6.5, Ana +1.3, Nadia +1.6, Pia +1.6, Ravi +2.0) |
| 9 | — | (system) | — | — | Ravi's symptom gives back 3.6 (Marta +1.1, Nadia +0.9, Pia +0.9, Sofia +0.7) |
| 9 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -6.7, Marta +2.0, Nadia +1.7, Pia +1.7, Sofia +1.3) |
| 10 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 10 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 10 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 10 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 10 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.3 |
| 10 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.5 |
| 10 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.16); the sender bears the refusal: anxiety Marta +6.1; anxiety Nadia +1.4 (witness), Ravi +5.4 (witness) |
| 10 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | arrives week 11; anxiety Ana +3.4, Marta +0.7, Nadia +1.4 (witness), Pia +1.2 (witness), Ravi +2.7 (witness) |
| 10 | Nadia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 10 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Nadia +3.7; anxiety Marta +5.3 (witness), Pia +1.3 (witness) |
| 10 | Pia | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 10 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 11; anxiety Marta +5.1 (witness), Nadia +1.4 (witness), Pia +0.6, Ravi +3.7; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1; 20.6 of the sender's anxiety bound into Pia–Ravi |
| 10 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 11; anxiety Marta +7.9, Nadia +2.3 (witness), Pia +2.0 (witness), Ravi +1.2; calmer sender takes some anxiety: Marta -0.5, Ravi +0.5 |
| 10 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | arrives week 12; anxiety Marta +2.4 (witness), Nadia +1.4 (witness), Pia +1.2 (witness), Ravi +0.7, Sofia +4.6 |
| 10 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +2.0 (witness), Nadia +1.1 (witness), Pia +1.0 (witness) |
| 10 | Pia | SYMPTOM_ONSET | Marta, Ravi | Nadia | arrives week 11; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, social channel; anxiety Marta +3.9, Nadia +0.9 (witness), Ravi +4.2 |
| 10 | — | (system) | — | — | Marta's symptom gives back 3.2 (Ana +0.6, Nadia +0.8, Pia +0.8, Ravi +0.9) |
| 10 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -1.3, Ana +0.3, Nadia +0.3, Pia +0.3, Ravi +0.4) |
| 10 | — | (system) | — | — | Ravi's symptom gives back 1.3 (Marta +0.4, Nadia +0.3, Pia +0.3, Sofia +0.3) |
| 10 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -0.1, Marta +0.0, Nadia +0.0, Pia +0.0, Sofia +0.0) |
| 11 | (from outside) | OWN_ILLNESS | Pia | Marta, Ravi | a 12-week spell; anxiety Marta +1.1 (witness), Pia +1.1, Ravi +1.2 (witness); still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6 |
| 11 | (from outside) | FAMILY_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +0.9, Nadia +0.3 (witness), Pia +0.2 (witness), Ravi +0.5 (witness); still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5 |
| 11 | (from outside) | FAMILY_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +0.5 (witness), Nadia +0.3 (witness), Pia +0.2 (witness), Ravi +1.0; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.5 |
| 11 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 11 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 11 | Ana | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 11 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.4 |
| 11 | Bruno | CUTOFF | Ana | — | arrives week 12; anxiety Ana +1.0, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 11 | Marta | PREVENT_ALIGNMENT | Nadia | Pia, Ravi | arrives week 12; anxiety Marta +0.8, Nadia +1.4, Pia +0.8 (witness), Ravi +3.5 (witness); calmer sender takes some anxiety: Marta +0.7, Nadia -0.7 |
| 11 | Marta | TRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.18); the sender bears the refusal: anxiety Marta +6.9; anxiety Nadia +1.6 (witness), Ravi +6.2 (witness) |
| 11 | Nadia | PURSUE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +3.8; anxiety Marta +5.6 (witness), Pia +1.3 (witness) |
| 11 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.8; anxiety Pia +1.3 (witness), Ravi +6.0 (witness) |
| 11 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 11 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 12; anxiety Marta +3.5, Nadia +3.0 (witness), Pia +2.6 (witness), Ravi +1.4; calmer sender takes some anxiety: Marta -0.6, Ravi +0.6; 17.0 of the sender's anxiety bound into Marta–Ravi |
| 11 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | arrives week 13; anxiety Marta +2.7 (witness), Nadia +1.6 (witness), Pia +1.3 (witness), Ravi +0.8, Sofia +3.0; 41.1 of the sender's anxiety bound into Ravi–Sofia |
| 11 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 13; anxiety Marta +2.0 (witness), Nadia +1.2 (witness), Pia +1.0 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; Sofia rejects the other on Ravi–Sofia (rejection +0.62; the tie stays open); impingement removed at once, anxiety Sofia -6.0 |
| 11 | — | (system) | — | — | Marta's symptom gives back 5.7 (Ana +1.1, Nadia +1.4, Pia +1.4, Ravi +1.7) |
| 11 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -9.3, Ana +1.9, Nadia +2.3, Pia +2.3, Ravi +2.8) |
| 11 | — | (system) | — | — | Ravi's symptom gives back 1.9 (Marta +0.6, Nadia +0.5, Pia +0.5, Sofia +0.4) |
| 11 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -1.8, Marta +0.6, Nadia +0.5, Pia +0.5, Sofia +0.4) |
| 12 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.3 (witness), Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +5.0 |
| 12 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.6, Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +2.5 (witness) |
| 12 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 12 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 12 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 12 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 12 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 12 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +0.4, Marta +7.2, Nadia +1.3 (witness), Pia +1.1 (witness), Ravi +2.3 (witness); calmer sender takes some anxiety: Ana +1.1, Marta -1.1; Ana takes 1.0 of functioning from the other |
| 12 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.6 |
| 12 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.7 |
| 12 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +4.5, Marta +0.9, Nadia +1.7 (witness), Pia +1.5 (witness), Ravi +3.1 (witness) |
| 12 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 13; anxiety Marta +1.4, Nadia +3.4 (witness), Pia +2.9 (witness), Ravi +5.1; Marta rejects the other on Marta–Ravi (rejection +0.92; the tie stays open); impingement removed at once, anxiety Marta -7.5 |
| 12 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.06); the sender bears the refusal: anxiety Marta +7.1; anxiety Nadia +1.7 (witness), Ravi +6.1 (witness) |
| 12 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 13; anxiety Marta +5.9 (witness), Nadia +0.8, Pia +1.4 (witness), Ravi +3.4; Nadia rejects the other on Nadia–Ravi (rejection +0.72; the tie stays open); impingement removed at once, anxiety Nadia -6.0 |
| 12 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +4.0; anxiety Pia +1.4 (witness), Ravi +6.1 (witness) |
| 12 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 12 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Pia +2.9; anxiety Nadia +1.4 (witness), Ravi +5.0 (witness) |
| 12 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 13; anxiety Marta +5.5 (witness), Nadia +1.6 (witness), Pia +3.8, Ravi +1.3; Pia holds a position: no functioning changes hands |
| 12 | Ravi | PURSUE | Nadia | Marta, Pia | arrives week 13; anxiety Marta +5.5 (witness), Nadia +5.8, Pia +1.3 (witness), Ravi +1.3; calmer sender takes some anxiety: Nadia -1.5, Ravi +1.5 |
| 12 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.13); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +2.2 (witness), Nadia +1.3 (witness), Pia +1.1 (witness) |
| 12 | — | (system) | — | — | Marta's symptom gives back 6.9 (Ana +1.4, Nadia +1.7, Pia +1.7, Ravi +2.1) |
| 12 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -17.9, Ana +3.6, Nadia +4.5, Pia +4.5, Ravi +5.4) |
| 12 | — | (system) | — | — | Ravi's symptom gives back 1.6 (Marta +0.5, Nadia +0.4, Pia +0.4, Sofia +0.3) |
| 12 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -1.9, Marta +0.6, Nadia +0.5, Pia +0.5, Sofia +0.4) |
| 13 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 13 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 13 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 13 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 14; anxiety Ana +0.4, Marta +3.9, Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +2.6 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; 8.5 of the sender's anxiety bound into Ana–Marta |
| 13 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.8 |
| 13 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.8 |
| 13 | Marta | REDUCE_CUTOFF | Ravi | Nadia, Pia | arrives week 14; anxiety Marta +0.8, Nadia +1.9 (witness), Pia +1.7 (witness), Ravi +4.0; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; Marta's rejection -0.25 |
| 13 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 14; anxiety Dr Halim +0.6, Marta +0.4, Nadia +1.4 (witness), Pia +1.2 (witness), Ravi +3.2 (witness) |
| 13 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +4.3; anxiety Pia +1.6 (witness), Ravi +6.3 (witness) |
| 13 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +4.3; anxiety Marta +6.7 (witness), Pia +1.7 (witness) |
| 13 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +3.4; anxiety Nadia +1.7 (witness), Ravi +6.0 (witness) |
| 13 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 14; anxiety Marta +6.1 (witness), Nadia +1.7 (witness), Pia +0.7, Ravi +7.9; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1; 17.3 of the sender's anxiety bound into Pia–Ravi |
| 13 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ravi +6.3; anxiety Nadia +3.1 (witness), Pia +2.6 (witness) |
| 13 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.16); the sender bears the refusal: anxiety Ravi +3.8; anxiety Marta +2.7 (witness), Nadia +1.6 (witness), Pia +1.3 (witness) |
| 13 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.06); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +2.1 (witness), Nadia +1.3 (witness), Pia +1.1 (witness) |
| 13 | Ana | SYMPTOM_ONSET | Bruno, Marta | Nadia, Pia, Ravi | arrives week 14; the nodal event re-opens the cut tie Ana–Bruno: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, physical channel; anxiety Bruno +3.0, Marta +2.4, Nadia +1.0 (witness), Pia +0.9 (witness), Ravi +1.7 (witness) |
| 13 | Bruno | SYMPTOM_ONSET | Ana | — | arrives week 14; the nodal event re-opens the cut tie Ana–Bruno: its standing load spikes next week (too-little side +1 × optimum); symptom onset, social channel; anxiety Ana +1.9 |
| 13 | — | (system) | — | — | Marta's symptom gives back 6.0 (Ana +1.2, Nadia +1.5, Pia +1.5, Ravi +1.8) |
| 13 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -26.1, Ana +5.2, Nadia +6.5, Pia +6.5, Ravi +7.8) |
| 13 | — | (system) | — | — | Ravi's symptom gives back 0.8 (Marta +0.2, Nadia +0.2, Pia +0.2, Sofia +0.2) |
| 13 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -0.8, Marta +0.2, Nadia +0.2, Pia +0.2, Sofia +0.2) |
| 14 | (from outside) | OWN_ILLNESS | Ana | — | a 12-week spell; anxiety Ana +1.3; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7 |
| 14 | (from outside) | FAMILY_ILLNESS | Bruno | — | a 12-week spell; anxiety Bruno +1.0; still weighing this week: anxiety Bruno +0.5; still weighing this week: anxiety Bruno +0.5; still weighing this week: anxiety Bruno +0.5; still weighing this week: anxiety Bruno +0.5; still weighing this week: anxiety Bruno +0.5; still weighing this week: anxiety Bruno +0.5; still weighing this week: anxiety Bruno +0.5; still weighing this week: anxiety Bruno +0.5; still weighing this week: anxiety Bruno +0.5; still weighing this week: anxiety Bruno +0.5; still weighing this week: anxiety Bruno +0.5 |
| 14 | (from outside) | FAMILY_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +0.7, Nadia +0.2 (witness), Pia +0.2 (witness), Ravi +0.4 (witness); still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4 |
| 14 | (from outside) | OWN_ILLNESS | Bruno | — | a 12-week spell; anxiety Bruno +2.1; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.1; still weighing this week: anxiety Bruno +1.1; still weighing this week: anxiety Bruno +1.1; still weighing this week: anxiety Bruno +1.1; still weighing this week: anxiety Bruno +1.1; still weighing this week: anxiety Bruno +1.1 |
| 14 | (from outside) | FAMILY_ILLNESS | Ana | — | a 12-week spell; anxiety Ana +0.6; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3 |
| 14 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 14 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.9 |
| 14 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 14 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 14 | Ana | WITHHOLD | Bruno | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 14 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +4.1 |
| 14 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +7.1; anxiety Nadia +1.9 (witness), Ravi +6.3 (witness) |
| 14 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | arrives week 15; anxiety Ana +3.6, Marta +0.9, Nadia +1.9 (witness), Pia +1.6 (witness), Ravi +3.2 (witness) |
| 14 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 15; anxiety Marta +5.2, Nadia +0.9, Pia +1.7 (witness), Ravi +6.7 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.99; the tie stays open); impingement removed at once, anxiety Nadia -1.1 |
| 14 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +4.6; anxiety Marta +7.3 (witness), Pia +1.8 (witness) |
| 14 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 15; anxiety Marta +5.9 (witness), Nadia +1.7 (witness), Pia +0.7, Ravi +3.4; calmer sender takes some anxiety: Pia +1.4, Ravi -1.4; 32.9 of the sender's anxiety bound into Pia–Ravi |
| 14 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +3.4; anxiety Nadia +1.8 (witness), Ravi +5.9 (witness) |
| 14 | Ravi | WITHHOLD | Sofia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 14 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.19); the sender bears the refusal: anxiety Ravi +7.5; anxiety Nadia +3.9 (witness), Pia +3.3 (witness) |
| 14 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +2.2 (witness), Nadia +1.4 (witness), Pia +1.1 (witness) |
| 14 | — | (system) | — | — | Marta's symptom gives back 1.3 (Ana +0.3, Nadia +0.3, Pia +0.3, Ravi +0.4) |
| 14 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -17.6, Ana +3.5, Nadia +4.4, Pia +4.4, Ravi +5.3) |
| 14 | — | (system) | — | — | Nadia's symptom gives back 5.2 (Marta +2.6, Ravi +2.6) |
| 14 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -11.0, Marta +5.5, Ravi +5.5) |
| 14 | — | (system) | — | — | Ravi's symptom gives back 3.8 (Marta +1.1, Nadia +0.9, Pia +0.9, Sofia +0.8) |
| 14 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -14.2, Marta +4.3, Nadia +3.5, Pia +3.5, Sofia +2.8) |
| 15 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 15 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.9 |
| 15 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 15 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 15 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 15 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 16; anxiety Ana +0.4, Marta +6.1, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +2.6 (witness); calmer sender takes some anxiety: Ana +1.5, Marta -1.5; 7.5 of the sender's anxiety bound into Ana–Marta |
| 15 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.8 |
| 15 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +4.1 |
| 15 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 16; anxiety Marta +1.5, Nadia +4.1 (witness), Pia +3.4 (witness), Ravi +6.3; 59.8 of the sender's anxiety bound into Marta–Ravi |
| 15 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +7.5; anxiety Nadia +2.1 (witness), Ravi +6.8 (witness) |
| 15 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +4.9; anxiety Pia +1.8 (witness), Ravi +7.1 (witness) |
| 15 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +4.9; anxiety Marta +7.5 (witness), Pia +1.9 (witness) |
| 15 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 16; anxiety Marta +5.4 (witness), Nadia +1.7 (witness), Pia +0.6, Ravi +3.7; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9 |
| 15 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +3.1; anxiety Nadia +1.7 (witness), Ravi +5.4 (witness) |
| 15 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.20); the sender bears the refusal: anxiety Ravi +4.8; anxiety Marta +3.6 (witness), Nadia +2.2 (witness), Pia +1.8 (witness) |
| 15 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.16); the sender bears the refusal: anxiety Ravi +7.9; anxiety Nadia +4.4 (witness), Pia +3.6 (witness) |
| 15 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 16; anxiety Dr Halim +1.0, Marta +3.8 (witness), Nadia +1.8 (witness), Pia +1.5 (witness), Ravi +0.5 |
| 15 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +2.3 (witness), Nadia +1.5 (witness), Pia +1.2 (witness) |
| 15 | — | (system) | — | — | Marta's symptom gives back 0.2 (Ana +0.0, Nadia +0.1, Pia +0.1, Ravi +0.1) |
| 15 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -31.4, Ana +6.3, Nadia +7.9, Pia +7.9, Ravi +9.4) |
| 15 | — | (system) | — | — | Nadia's symptom gives back 10.7 (Marta +5.3, Ravi +5.3) |
| 15 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -26.5, Marta +13.3, Ravi +13.3) |
| 15 | — | (system) | — | — | Ravi's symptom gives back 6.7 (Marta +2.0, Nadia +1.7, Pia +1.7, Sofia +1.3) |
| 15 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -52.6, Marta +15.8, Nadia +13.1, Pia +13.1, Sofia +10.5) |
| 16 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +4.8 |
| 16 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +2.4 (witness) |
| 16 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 16 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 16 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 16 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.7 |
| 16 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 17; anxiety Ana +0.4, Marta +3.6, Nadia +1.6 (witness), Pia +1.3 (witness), Ravi +2.6 (witness); calmer sender takes some anxiety: Ana +2.1, Marta -2.1 |
| 16 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +4.1 |
| 16 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 17; anxiety Dr Halim +0.1, Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +1.9 (witness); calmer sender takes some anxiety: Dr Halim +1.5, Marta -1.5 |
| 16 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 17; anxiety Marta +1.3, Nadia +3.7 (witness), Pia +3.1 (witness), Ravi +4.5; 28.6 of the sender's anxiety bound into Marta–Ravi |
| 16 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.11); the sender bears the refusal: anxiety Marta +6.5; anxiety Pia +1.5 (witness), Ravi +5.9 (witness) |
| 16 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.0; anxiety Marta +7.7 (witness), Pia +2.0 (witness) |
| 16 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.0; anxiety Pia +1.9 (witness), Ravi +7.3 (witness) |
| 16 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 17; anxiety Marta +6.5 (witness), Nadia +2.0 (witness), Pia +0.7, Ravi +3.6; 51.9 of the sender's anxiety bound into Pia–Ravi |
| 16 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +3.7; anxiety Nadia +2.1 (witness), Ravi +6.7 (witness) |
| 16 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.20); the sender bears the refusal: anxiety Ravi +4.7; anxiety Marta +3.5 (witness), Nadia +2.3 (witness), Pia +1.8 (witness) |
| 16 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.20); the sender bears the refusal: anxiety Ravi +8.0; anxiety Marta +7.5 (witness), Pia +1.9 (witness) |
| 16 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +8.0; anxiety Nadia +4.6 (witness), Pia +3.7 (witness) |
| 16 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +2.1; anxiety Marta +2.5 (witness), Nadia +1.6 (witness), Pia +1.3 (witness) |
| 16 | Sofia | SYMPTOM_ONSET | Ravi | Marta, Nadia, Pia | arrives week 18; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, physical channel; anxiety Marta +1.8 (witness), Nadia +1.1 (witness), Pia +0.9 (witness), Ravi +2.4 |
| 16 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -2.0, Ana +0.4, Nadia +0.5, Pia +0.5, Ravi +0.6) |
| 16 | — | (system) | — | — | Nadia's symptom gives back 10.4 (Marta +5.2, Ravi +5.2) |
| 16 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -43.8, Marta +21.9, Ravi +21.9) |
| 16 | — | (system) | — | — | Ravi's symptom gives back 2.4 (Marta +0.7, Nadia +0.6, Pia +0.6, Sofia +0.5) |
| 16 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -71.2, Marta +21.4, Nadia +17.8, Pia +17.8, Sofia +14.2) |
| 17 | (from outside) | OWN_ILLNESS | Sofia | — | a 12-week spell; anxiety Sofia +1.4; still weighing this week: anxiety Sofia +0.7; still weighing this week: anxiety Sofia +0.7; still weighing this week: anxiety Sofia +0.7; still weighing this week: anxiety Sofia +0.7; still weighing this week: anxiety Sofia +0.7; still weighing this week: anxiety Sofia +0.7; still weighing this week: anxiety Sofia +0.7; still weighing this week: anxiety Sofia +0.7; still weighing this week: anxiety Sofia +0.7; still weighing this week: anxiety Sofia +0.7; still weighing this week: anxiety Sofia +0.7 |
| 17 | (from outside) | FAMILY_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +0.4 (witness), Nadia +0.2 (witness), Pia +0.2 (witness), Ravi +0.7; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4 |
| 17 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 17 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 17 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 17 | Ana | WITHHOLD | Bruno | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 17 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 18; anxiety Ana +0.4, Marta +2.4, Nadia +1.6 (witness), Pia +1.3 (witness), Ravi +2.5 (witness); calmer sender takes some anxiety: Ana +1.8, Marta -1.8; Marta takes 0.0 of functioning from the other |
| 17 | Bruno | CUTOFF | Ana | — | arrives week 18; anxiety Ana +1.0, Bruno +0.8; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 17 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 18; anxiety Dr Halim +1.1, Marta +0.5, Nadia +1.9 (witness), Pia +1.5 (witness), Ravi +3.8 (witness) |
| 17 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 18; anxiety Marta +1.5, Nadia +4.5 (witness), Pia +3.6 (witness), Ravi +3.6; 66.7 of the sender's anxiety bound into Marta–Ravi |
| 17 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.1; anxiety Pia +1.9 (witness), Ravi +7.4 (witness) |
| 17 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.1; anxiety Marta +7.3 (witness), Pia +1.9 (witness) |
| 17 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 18; anxiety Marta +5.8 (witness), Nadia +1.9 (witness), Pia +0.7, Ravi +3.6; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1 |
| 17 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 18; anxiety Marta +3.8, Nadia +1.9 (witness), Pia +0.7, Ravi +5.8 (witness); calmer sender takes some anxiety: Marta -1.5, Pia +1.5; Pia rejects the other on Marta–Pia (rejection +0.10; the tie stays open) |
| 17 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.13); the sender bears the refusal: anxiety Ravi +4.6; anxiety Marta +3.6 (witness), Nadia +2.5 (witness), Pia +1.8 (witness) |
| 17 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +7.9; anxiety Marta +7.3 (witness), Pia +1.9 (witness) |
| 17 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +2.3; anxiety Marta +2.8 (witness), Nadia +1.9 (witness), Pia +1.4 (witness) |
| 17 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -36.4, Ana +7.3, Nadia +9.1, Pia +9.1, Ravi +10.9) |
| 17 | — | (system) | — | — | Nadia's symptom gives back 4.5 (Marta +2.3, Ravi +2.3) |
| 17 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -57.1, Marta +28.6, Ravi +28.6) |
| 17 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -80.2, Marta +24.1, Nadia +20.1, Pia +20.1, Sofia +16.0) |
| 18 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 18 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 18 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 18 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 19; anxiety Ana +0.4, Marta +2.2, Nadia +1.8 (witness), Pia +1.4 (witness), Ravi +2.5 (witness); calmer sender takes some anxiety: Ana +1.6, Marta -1.6; Ana rejects the other on Ana–Marta (rejection +0.74; the tie stays open); impingement removed at once, anxiety Ana -1.1 |
| 18 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +4.0 |
| 18 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 19; anxiety Marta +1.3, Nadia +4.2 (witness), Pia +3.2 (witness), Ravi +3.5; 24.2 of the sender's anxiety bound into Marta–Ravi |
| 18 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +6.5; anxiety Nadia +2.1 (witness), Ravi +5.8 (witness) |
| 18 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.3; anxiety Pia +1.9 (witness), Ravi +7.1 (witness) |
| 18 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.3; anxiety Marta +7.2 (witness), Pia +1.9 (witness) |
| 18 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +3.9; anxiety Nadia +2.3 (witness), Ravi +6.6 (witness) |
| 18 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 19; anxiety Marta +6.7 (witness), Nadia +2.3 (witness), Pia +0.8, Ravi +3.5; Pia rejects the other on Pia–Ravi (rejection +0.94; the tie stays open); impingement removed at once, anxiety Pia -0.1 |
| 18 | Ravi | WITHHOLD | Sofia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 18 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 19; anxiety Dr Halim +1.5, Marta +3.9 (witness), Nadia +2.0 (witness), Pia +1.5 (witness), Ravi +0.4 |
| 18 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 18 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -1.7, Ana +0.3, Nadia +0.4, Pia +0.4, Ravi +0.5) |
| 18 | — | (system) | — | — | Nadia's symptom gives back 0.3 (Marta +0.2, Ravi +0.2) |
| 18 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -37.2, Marta +18.6, Ravi +18.6) |
| 18 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -48.6, Marta +14.6, Nadia +12.2, Pia +12.2, Sofia +9.7) |
| 19 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 19 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 19 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 19 | Ana | REDUCE_CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 20; anxiety Ana +0.3, Marta +2.1, Nadia +1.2 (witness), Pia +0.9 (witness), Ravi +1.7 (witness); calmer sender takes some anxiety: Ana +1.8, Marta -1.8; Ana's rejection -0.25 |
| 19 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.7 |
| 19 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +4.3 |
| 19 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Marta +7.4; anxiety Pia +1.8 (witness), Ravi +6.8 (witness) |
| 19 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.18); the sender bears the refusal: anxiety Marta +4.4; anxiety Nadia +2.3 (witness), Pia +1.7 (witness), Ravi +3.1 (witness) |
| 19 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.2; anxiety Marta +7.1 (witness), Pia +1.8 (witness) |
| 19 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.2; anxiety Pia +1.8 (witness), Ravi +7.0 (witness) |
| 19 | Pia | STAY-IN-CONTACT | Ravi | Marta, Nadia | arrives week 20; anxiety Marta +3.6 (witness), Nadia +1.2 (witness), Pia +0.4, Ravi +3.0 |
| 19 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.0; anxiety Nadia +2.5 (witness), Ravi +6.9 (witness) |
| 19 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +4.3; anxiety Marta +3.4 (witness), Nadia +2.3 (witness), Pia +1.8 (witness) |
| 19 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +7.3; anxiety Marta +7.0 (witness), Pia +1.8 (witness) |
| 19 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.11); the sender bears the refusal: anxiety Ravi +7.3; anxiety Marta +6.7 (witness), Nadia +2.3 (witness) |
| 19 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 19 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -21.7, Ana +4.3, Nadia +5.4, Pia +5.4, Ravi +6.5) |
| 19 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -35.0, Marta +17.5, Ravi +17.5) |
| 19 | — | (system) | — | — | Pia's symptom gives back 1.6 (Marta +0.8, Ravi +0.8) |
| 19 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -1.3, Marta +0.6, Ravi +0.6) |
| 19 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -51.4, Marta +15.4, Nadia +12.9, Pia +12.9, Sofia +10.3) |
| 20 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +1.7 (witness), Pia +1.2 (witness), Ravi +4.7 |
| 20 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +1.7 (witness), Pia +1.2 (witness), Ravi +2.3 (witness) |
| 20 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 20 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 20 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 20 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.7 |
| 20 | Ana | CUTOFF | Bruno | — | arrives week 21; anxiety Ana +0.6, Bruno +1.8; calmer sender takes some anxiety: Ana +0.3, Bruno -0.3; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 20 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +4.3 |
| 20 | Marta | TRIANGLE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +8.2; anxiety Pia +2.0 (witness), Ravi +7.3 (witness) |
| 20 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 21; anxiety Dr Halim +1.6, Marta +0.5, Nadia +2.3 (witness), Pia +1.8 (witness), Ravi +4.2 (witness) |
| 20 | Nadia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 20 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.1; anxiety Marta +6.9 (witness), Pia +1.8 (witness) |
| 20 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 21; anxiety Marta +7.4 (witness), Nadia +2.5 (witness), Pia +0.8, Ravi +3.8; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 20 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.2; anxiety Nadia +2.5 (witness), Ravi +7.0 (witness) |
| 20 | Ravi | PURSUE | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +7.6; anxiety Marta +7.1 (witness), Nadia +2.4 (witness) |
| 20 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 21; anxiety Dr Halim +1.5, Marta +4.3 (witness), Nadia +2.2 (witness), Pia +1.7 (witness), Ravi +0.5 |
| 20 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Sofia +2.4; anxiety Marta +3.0 (witness), Nadia +2.1 (witness), Pia +1.6 (witness) |
| 20 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -53.4, Ana +10.7, Nadia +13.4, Pia +13.4, Ravi +16.0) |
| 20 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -37.1, Marta +18.5, Ravi +18.5) |
| 20 | — | (system) | — | — | Pia's symptom gives back 13.5 (Marta +6.8, Ravi +6.8) |
| 20 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -14.7, Marta +7.3, Ravi +7.3) |
| 20 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -89.7, Marta +26.9, Nadia +22.4, Pia +22.4, Sofia +17.9) |
| 21 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 21 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 21 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.7 |
| 21 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 21 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 21 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 21 | Ana | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 21 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.9 |
| 21 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +4.4 |
| 21 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 22; anxiety Marta +1.7, Nadia +5.7 (witness), Pia +4.2 (witness), Ravi +7.9 |
| 21 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.21); the sender bears the refusal: anxiety Marta +5.1; anxiety Nadia +2.8 (witness), Pia +2.1 (witness), Ravi +3.9 (witness) |
| 21 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +8.5; anxiety Pia +2.1 (witness), Ravi +7.8 (witness) |
| 21 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +5.4; anxiety Pia +1.9 (witness), Ravi +6.9 (witness) |
| 21 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.4; anxiety Marta +7.0 (witness), Pia +1.8 (witness) |
| 21 | Pia | TRIANGLE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.4; anxiety Nadia +2.7 (witness), Ravi +7.4 (witness) |
| 21 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 22; anxiety Marta +7.7 (witness), Nadia +2.7 (witness), Pia +0.9, Ravi +3.6; Pia takes 1.1 of functioning from the other |
| 21 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +4.8; anxiety Marta +3.5 (witness), Nadia +2.5 (witness), Pia +1.8 (witness) |
| 21 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +7.6; anxiety Marta +7.1 (witness), Pia +1.8 (witness) |
| 21 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +2.5; anxiety Marta +3.2 (witness), Nadia +2.2 (witness), Pia +1.7 (witness) |
| 21 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -67.9, Ana +13.6, Nadia +17.0, Pia +17.0, Ravi +20.4) |
| 21 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -44.9, Marta +22.5, Ravi +22.5) |
| 21 | — | (system) | — | — | Pia's symptom gives back 26.1 (Marta +13.0, Ravi +13.0) |
| 21 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -40.0, Marta +20.0, Ravi +20.0) |
| 21 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -105.3, Marta +31.6, Nadia +26.3, Pia +26.3, Sofia +21.1) |
| 22 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.5 |
| 22 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 22 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 22 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 22 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 22 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 22 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | arrives week 23; anxiety Ana +0.3, Marta +4.0, Nadia +1.3 (witness), Pia +1.0 (witness), Ravi +1.9 (witness); calmer sender takes some anxiety: Ana +3.2, Marta -3.2 |
| 22 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +3.0 |
| 22 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +4.4 |
| 22 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.20); the sender bears the refusal: anxiety Marta +8.8; anxiety Nadia +2.9 (witness), Ravi +8.2 (witness) |
| 22 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 23; anxiety Marta +1.8, Nadia +5.9 (witness), Pia +4.3 (witness), Ravi +3.7 |
| 22 | Nadia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 22 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 23; anxiety Marta +7.2 (witness), Nadia +1.1, Pia +1.9 (witness), Ravi +6.2; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 22 | Pia | TRIANGLE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.4; anxiety Nadia +2.7 (witness), Ravi +7.8 (witness) |
| 22 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 23; anxiety Marta +7.9 (witness), Nadia +2.8 (witness), Pia +0.9, Ravi +3.7; 34.2 of the sender's anxiety bound into Pia–Ravi |
| 22 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 23; anxiety Dr Halim +1.5, Marta +4.4 (witness), Nadia +2.4 (witness), Pia +1.8 (witness), Ravi +0.5 |
| 22 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 23; anxiety Marta +6.9, Nadia +5.2 (witness), Pia +3.8 (witness), Ravi +1.6; calmer sender takes some anxiety: Marta -4.7, Ravi +4.7; Ravi rejects the other on Marta–Ravi (rejection +0.98; the tie stays open); impingement removed at once, anxiety Ravi -11.9 |
| 22 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 24; anxiety Marta +3.3 (witness), Nadia +2.3 (witness), Pia +1.7 (witness), Ravi +1.0, Sofia +0.5; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 22 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -84.8, Ana +17.0, Nadia +21.2, Pia +21.2, Ravi +25.5) |
| 22 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -58.2, Marta +29.1, Ravi +29.1) |
| 22 | — | (system) | — | — | Pia's symptom gives back 23.9 (Marta +12.0, Ravi +12.0) |
| 22 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -66.0, Marta +33.0, Ravi +33.0) |
| 22 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -135.4, Marta +40.6, Nadia +33.9, Pia +33.9, Sofia +27.1) |
| 22 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 22 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -7.1, Ravi +7.1) |
| 23 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +9.7 |
| 23 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 23 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 23 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 24; anxiety Ana +0.3, Marta +1.0, Nadia +1.3 (witness), Pia +1.0 (witness), Ravi +1.9 (witness); calmer sender takes some anxiety: Ana +2.7, Marta -2.7 |
| 23 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +3.2 |
| 23 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +4.5 |
| 23 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +9.1; anxiety Nadia +3.0 (witness), Ravi +8.6 (witness) |
| 23 | Marta | CONFLICT | Ravi | Nadia, Pia | refused by Ravi (rejection +0.07); the sender bears the refusal: anxiety Marta +9.1; anxiety Nadia +6.2 (witness), Pia +4.6 (witness) |
| 23 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +5.7; anxiety Pia +1.9 (witness), Ravi +7.5 (witness) |
| 23 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +5.7; anxiety Marta +7.5 (witness), Pia +1.9 (witness) |
| 23 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 24; anxiety Marta +7.0 (witness), Nadia +2.5 (witness), Pia +0.8, Ravi +3.7; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2 |
| 23 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +3.9; anxiety Nadia +2.4 (witness), Ravi +7.0 (witness) |
| 23 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.19); the sender bears the refusal: anxiety Ravi +8.0; anxiety Nadia +5.2 (witness), Pia +3.8 (witness) |
| 23 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +8.0; anxiety Marta +7.2 (witness), Pia +1.9 (witness) |
| 23 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 25; anxiety Marta +3.4 (witness), Nadia +2.5 (witness), Pia +1.8 (witness), Ravi +2.1, Sofia +0.6; calmer sender takes some anxiety: Ravi -1.4, Sofia +1.4; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 23 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -105.7, Ana +21.1, Nadia +26.4, Pia +26.4, Ravi +31.7) |
| 23 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -70.8, Marta +35.4, Ravi +35.4) |
| 23 | — | (system) | — | — | Pia's symptom gives back 6.9 (Marta +3.4, Ravi +3.4) |
| 23 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -35.6, Marta +17.8, Ravi +17.8) |
| 23 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -116.8, Marta +35.0, Nadia +29.2, Pia +29.2, Sofia +23.4) |
| 23 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 23 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -17.7, Ravi +17.7) |
| 24 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +1.7 (witness), Pia +1.3 (witness), Ravi +5.0 |
| 24 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +1.7 (witness), Pia +1.3 (witness), Ravi +2.5 (witness) |
| 24 | — | (system) | — | — | Ana–Marta: an accepted contact bridges the cutoff; held anxiety -24.2 |
| 24 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 24 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 24 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 24 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 24 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 24 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 24 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 25; anxiety Ana +0.5, Marta +2.3, Nadia +2.5 (witness), Pia +1.7 (witness), Ravi +3.3 (witness); calmer sender takes some anxiety: Ana +3.2, Marta -3.2; Ana rejects the other on Ana–Marta (rejection +0.18; the tie stays open) |
| 24 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +3.5 |
| 24 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +4.5 |
| 24 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 25; anxiety Dr Halim +0.1, Nadia +1.3 (witness), Pia +0.9 (witness), Ravi +2.2 (witness); calmer sender takes some anxiety: Dr Halim +2.7, Marta -2.7 |
| 24 | Marta | CONFLICT | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +9.0; anxiety Nadia +6.4 (witness), Pia +4.5 (witness) |
| 24 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 25; anxiety Dr Halim +1.9, Marta +0.5, Nadia +2.9 (witness), Pia +2.0 (witness), Ravi +5.1 (witness) |
| 24 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.23); the sender bears the refusal: anxiety Marta +5.4; anxiety Nadia +3.1 (witness), Pia +2.2 (witness), Ravi +4.2 (witness) |
| 24 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.1; anxiety Pia +2.0 (witness), Ravi +7.5 (witness) |
| 24 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.1; anxiety Marta +7.7 (witness), Pia +2.0 (witness) |
| 24 | Pia | TRIANGLE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.2; anxiety Nadia +2.8 (witness), Ravi +7.4 (witness) |
| 24 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 25; anxiety Marta +7.6 (witness), Nadia +2.8 (witness), Pia +0.8, Ravi +3.7; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; Ravi takes 0.1 of functioning from the other |
| 24 | Ravi | DISTANCE | Marta | Nadia, Pia | refused by Marta (rejection +0.21); the sender bears the refusal: anxiety Ravi +8.5; anxiety Nadia +5.9 (witness), Pia +4.1 (witness) |
| 24 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 25; anxiety Dr Halim +1.7, Marta +4.8 (witness), Nadia +2.7 (witness), Pia +1.9 (witness), Ravi +0.5 |
| 24 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.20); the sender bears the refusal: anxiety Ravi +8.5; anxiety Marta +7.9 (witness), Nadia +2.9 (witness) |
| 24 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Sofia +2.8; anxiety Marta +3.3 (witness), Nadia +2.5 (witness), Pia +1.8 (witness) |
| 24 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -105.8, Ana +21.2, Nadia +26.4, Pia +26.4, Ravi +31.7) |
| 24 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -75.6, Marta +37.8, Ravi +37.8) |
| 24 | — | (system) | — | — | Pia's symptom gives back 2.9 (Marta +1.4, Ravi +1.4) |
| 24 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -61.0, Marta +30.5, Ravi +30.5) |
| 24 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -158.7, Marta +47.6, Nadia +39.7, Pia +39.7, Sofia +31.7) |
| 24 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 24 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -27.5, Ravi +27.5) |
| 25 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +24.8 |
| 25 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 25 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 25 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 25 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +3.9 |
| 25 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 25 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 25 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 26; anxiety Ana +0.5, Marta +6.8, Nadia +2.6 (witness), Pia +1.8 (witness), Ravi +3.5 (witness); calmer sender takes some anxiety: Ana +4.1, Marta -4.1 |
| 25 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +3.6 |
| 25 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 25 | Marta | WITHHOLD | Ana | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 25 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +9.2; anxiety Nadia +6.5 (witness), Pia +4.6 (witness) |
| 25 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.20); the sender bears the refusal: anxiety Marta +9.2; anxiety Pia +2.3 (witness), Ravi +8.7 (witness) |
| 25 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.4; anxiety Marta +8.1 (witness), Pia +2.2 (witness) |
| 25 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.4; anxiety Pia +2.1 (witness), Ravi +7.9 (witness) |
| 25 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 26; anxiety Marta +7.8 (witness), Nadia +2.9 (witness), Pia +0.9, Ravi +3.7; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia takes 0.6 of functioning from the other |
| 25 | Pia | TRIANGLE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.4; anxiety Nadia +2.8 (witness), Ravi +7.6 (witness) |
| 25 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +8.7; anxiety Marta +8.0 (witness), Nadia +3.0 (witness) |
| 25 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +5.3; anxiety Marta +4.0 (witness), Nadia +3.1 (witness), Pia +2.1 (witness) |
| 25 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 26; anxiety Dr Halim +1.7, Marta +5.0 (witness), Nadia +2.8 (witness), Pia +2.0 (witness), Ravi +0.5 |
| 25 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +2.8; anxiety Marta +3.3 (witness), Nadia +2.6 (witness), Pia +1.7 (witness) |
| 25 | — | (system) | — | — | Ana's symptom gives back 9.3 (Bruno +5.4, Marta +3.9) |
| 25 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -5.3, Bruno +3.1, Marta +2.2) |
| 25 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -132.7, Ana +26.5, Nadia +33.2, Pia +33.2, Ravi +39.8) |
| 25 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -101.3, Marta +50.7, Ravi +50.7) |
| 25 | — | (system) | — | — | Pia's symptom gives back 0.8 (Marta +0.4, Ravi +0.4) |
| 25 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -95.2, Marta +47.6, Ravi +47.6) |
| 25 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -206.5, Marta +62.0, Nadia +51.6, Pia +51.6, Sofia +41.3) |
| 25 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 25 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -37.2, Ravi +37.2) |
| 26 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 26 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 26 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 26 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 26 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 26 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 26 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 27; anxiety Ana +0.6, Marta +3.8, Nadia +2.8 (witness), Pia +1.8 (witness), Ravi +3.6 (witness); calmer sender takes some anxiety: Ana +4.0, Marta -4.0 |
| 26 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +3.8 |
| 26 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +4.6 |
| 26 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 27; anxiety Dr Halim +2.1, Marta +0.6, Nadia +3.3 (witness), Pia +2.2 (witness), Ravi +5.6 (witness) |
| 26 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 27; anxiety Ana +3.5, Marta +1.1, Nadia +3.5 (witness), Pia +2.3 (witness), Ravi +4.6 (witness); Marta rejects the other on Ana–Marta (rejection +1.00; the tie stays open); impingement removed at once, anxiety Marta -11.1 |
| 26 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +9.4; anxiety Nadia +7.1 (witness), Pia +4.6 (witness) |
| 26 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.7; anxiety Pia +2.1 (witness), Ravi +8.2 (witness) |
| 26 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.7; anxiety Marta +8.1 (witness), Pia +2.1 (witness) |
| 26 | Pia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 26 | Pia | TRIANGLE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.2; anxiety Nadia +2.9 (witness), Ravi +7.6 (witness) |
| 26 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 27; anxiety Dr Halim +1.7, Marta +5.0 (witness), Nadia +3.0 (witness), Pia +2.0 (witness), Ravi +0.5 |
| 26 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 28; anxiety Marta +3.9 (witness), Nadia +3.2 (witness), Pia +2.0 (witness), Ravi +1.0, Sofia +2.4; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -2.5 |
| 26 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 28; anxiety Marta +3.1 (witness), Nadia +2.6 (witness), Pia +1.6 (witness), Ravi +2.0, Sofia +0.5; calmer sender takes some anxiety: Ravi -1.6, Sofia +1.6; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Sofia -2.4 |
| 26 | — | (system) | — | — | Ana's symptom gives back 25.7 (Bruno +14.9, Marta +10.8) |
| 26 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -18.3, Bruno +10.6, Marta +7.7) |
| 26 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -182.4, Ana +36.5, Nadia +45.6, Pia +45.6, Ravi +54.7) |
| 26 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -109.4, Marta +54.7, Ravi +54.7) |
| 26 | — | (system) | — | — | Pia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 26 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -99.9, Marta +49.9, Ravi +49.9) |
| 26 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -220.0, Marta +66.0, Nadia +55.0, Pia +55.0, Sofia +44.0) |
| 26 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 26 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -38.5, Ravi +38.5) |
| 27 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.5 |
| 27 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 27 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 27 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 27 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 27 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 27 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 27 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +3.0; anxiety Nadia +3.0 (witness), Pia +1.9 (witness), Ravi +3.7 (witness) |
| 27 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +4.0 |
| 27 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +5.2 |
| 27 | Marta | WITHHOLD | Nadia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 27 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.20); the sender bears the refusal: anxiety Marta +9.1; anxiety Nadia +3.6 (witness), Ravi +8.9 (witness) |
| 27 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +9.1; anxiety Nadia +7.3 (witness), Pia +4.6 (witness) |
| 27 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +7.1; anxiety Pia +2.1 (witness), Ravi +8.3 (witness) |
| 27 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +7.1; anxiety Marta +8.1 (witness), Pia +2.2 (witness) |
| 27 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 28; anxiety Marta +7.5 (witness), Nadia +3.2 (witness), Pia +0.9, Ravi +6.1; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Ravi takes 0.1 of functioning from the other |
| 27 | Pia | TRIANGLE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.3; anxiety Nadia +3.1 (witness), Ravi +7.7 (witness) |
| 27 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.1; anxiety Marta +4.0 (witness), Nadia +3.4 (witness), Pia +2.0 (witness) |
| 27 | Ravi | TRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Ravi +8.6; anxiety Nadia +6.4 (witness), Pia +4.0 (witness) |
| 27 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 27 | — | (system) | — | — | Ana's symptom gives back 32.8 (Bruno +19.0, Marta +13.8) |
| 27 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -38.0, Bruno +22.0, Marta +16.0) |
| 27 | — | (system) | — | — | Bruno's symptom gives back 2.2 (Ana +2.2) |
| 27 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -5.1, Ana +5.1) |
| 27 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -168.6, Ana +33.7, Nadia +42.1, Pia +42.1, Ravi +50.6) |
| 27 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -113.6, Marta +56.8, Ravi +56.8) |
| 27 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -99.4, Marta +49.7, Ravi +49.7) |
| 27 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -211.4, Marta +63.4, Nadia +52.8, Pia +52.8, Sofia +42.3) |
| 27 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 27 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -35.2, Ravi +35.2) |
| 28 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.3 (witness), Nadia +2.0 (witness), Pia +1.3 (witness), Ravi +4.9 |
| 28 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.6, Nadia +2.0 (witness), Pia +1.3 (witness), Ravi +2.5 (witness) |
| 28 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 28 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 28 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 28 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 28 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 28 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 28 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 28 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 28 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 28 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +2.9; anxiety Nadia +3.1 (witness), Pia +1.9 (witness), Ravi +3.6 (witness) |
| 28 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +3.9 |
| 28 | Bruno | CUTOFF | Ana | — | arrives week 29; anxiety Ana +1.3, Bruno +1.2; calmer sender takes some anxiety: Ana -1.5, Bruno +1.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 28 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 29; anxiety Marta +1.9, Nadia +7.9 (witness), Pia +4.7 (witness), Ravi +11.4; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Marta -0.3 |
| 28 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +9.7; anxiety Nadia +3.9 (witness), Ravi +9.1 (witness) |
| 28 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.20); the sender bears the refusal: anxiety Marta +5.8; anxiety Nadia +3.9 (witness), Pia +2.4 (witness), Ravi +4.6 (witness) |
| 28 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +7.6; anxiety Pia +2.2 (witness), Ravi +8.6 (witness) |
| 28 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +7.6; anxiety Marta +8.5 (witness), Pia +2.2 (witness) |
| 28 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 29; anxiety Marta +7.9 (witness), Nadia +3.3 (witness), Pia +0.9, Ravi +3.8; calmer sender takes some anxiety: Pia +1.4, Ravi -1.4; Ravi takes 0.1 of functioning from the other |
| 28 | Pia | TRIANGLE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.3; anxiety Nadia +3.3 (witness), Ravi +7.7 (witness) |
| 28 | Ravi | CONFLICT | Pia | Marta, Nadia | refused by Pia (rejection +0.15); the sender bears the refusal: anxiety Ravi +8.8; anxiety Marta +8.2 (witness), Nadia +3.5 (witness) |
| 28 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.22); the sender bears the refusal: anxiety Ravi +8.8; anxiety Marta +8.4 (witness), Pia +2.2 (witness) |
| 28 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +8.8; anxiety Nadia +6.9 (witness), Pia +4.2 (witness) |
| 28 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 28 | — | (system) | — | — | Ana's symptom gives back 18.3 (Bruno +10.6, Marta +7.7) |
| 28 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -43.3, Bruno +25.0, Marta +18.2) |
| 28 | — | (system) | — | — | Bruno's symptom gives back 12.6 (Ana +12.6) |
| 28 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -39.3, Ana +39.3) |
| 28 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -190.2, Ana +38.0, Nadia +47.5, Pia +47.5, Ravi +57.0) |
| 28 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -125.7, Marta +62.8, Ravi +62.8) |
| 28 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -104.3, Marta +52.1, Ravi +52.1) |
| 28 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -239.6, Marta +71.9, Nadia +59.9, Pia +59.9, Sofia +47.9) |
| 28 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 28 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -35.3, Ravi +35.3) |
| 29 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 29 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 29 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 29 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 29 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +5.9 |
| 29 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 29 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +3.2; anxiety Nadia +3.4 (witness), Pia +2.0 (witness), Ravi +3.9 (witness) |
| 29 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +4.2 |
| 29 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +5.8 |
| 29 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Marta +9.8; anxiety Nadia +8.2 (witness), Pia +4.8 (witness) |
| 29 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 30; anxiety Dr Halim +2.3, Marta +0.6, Nadia +3.9 (witness), Pia +2.3 (witness), Ravi +5.9 (witness) |
| 29 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.20); the sender bears the refusal: anxiety Marta +9.8; anxiety Pia +2.5 (witness), Ravi +9.7 (witness) |
| 29 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 30; anxiety Marta +7.8, Nadia +1.6, Pia +2.2 (witness), Ravi +8.7 (witness); calmer sender takes some anxiety: Marta -7.3, Nadia +7.3; Nadia rejects the other on Marta–Nadia (rejection +0.00; the tie stays open) |
| 29 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +7.9; anxiety Marta +8.6 (witness), Pia +2.2 (witness) |
| 29 | Pia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 29 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 30; anxiety Marta +4.3, Nadia +3.4 (witness), Pia +0.9, Ravi +8.0 (witness); calmer sender takes some anxiety: Marta -8.9, Pia +8.9; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open) |
| 29 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.4; anxiety Nadia +7.5 (witness), Pia +4.4 (witness) |
| 29 | Ravi | CONFLICT | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.4; anxiety Marta +8.7 (witness), Nadia +3.8 (witness) |
| 29 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.08); the sender bears the refusal: anxiety Ravi +9.4; anxiety Marta +8.8 (witness), Pia +2.3 (witness) |
| 29 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 29 | — | (system) | — | — | Ana's symptom gives back 4.6 (Bruno +2.6, Marta +1.9) |
| 29 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -79.7, Bruno +46.1, Marta +33.6) |
| 29 | — | (system) | — | — | Bruno's symptom gives back 13.0 (Ana +13.0) |
| 29 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -69.7, Ana +69.7) |
| 29 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -223.8, Ana +44.8, Nadia +56.0, Pia +56.0, Ravi +67.1) |
| 29 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -144.8, Marta +72.4, Ravi +72.4) |
| 29 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -120.8, Marta +60.4, Ravi +60.4) |
| 29 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -300.4, Marta +90.1, Nadia +75.1, Pia +75.1, Sofia +60.1) |
| 29 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 29 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -40.5, Ravi +40.5) |
| 30 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 30 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 30 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 30 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 30 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 30 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +3.3; anxiety Nadia +3.6 (witness), Pia +2.1 (witness), Ravi +4.2 (witness) |
| 30 | Ana | CUTOFF | Bruno | — | arrives week 31; anxiety Ana +0.9, Bruno +4.3; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -0.4 |
| 30 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 30 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +5.9; anxiety Nadia +4.2 (witness), Pia +2.4 (witness), Ravi +4.8 (witness) |
| 30 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +9.9; anxiety Pia +2.5 (witness), Ravi +9.8 (witness) |
| 30 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +9.9; anxiety Nadia +8.4 (witness), Pia +4.9 (witness) |
| 30 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +8.0; anxiety Marta +8.7 (witness), Pia +2.2 (witness) |
| 30 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 31; anxiety Marta +6.0, Nadia +1.6, Pia +2.2 (witness), Ravi +8.8 (witness); calmer sender takes some anxiety: Marta -8.2, Nadia +8.2; Nadia rejects the other on Marta–Nadia (rejection +0.00; the tie stays open) |
| 30 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 31; anxiety Marta +8.3 (witness), Nadia +3.7 (witness), Pia +0.9, Ravi +6.3; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7 |
| 30 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.15); the sender bears the refusal: anxiety Ravi +5.5; anxiety Marta +4.2 (witness), Nadia +3.7 (witness), Pia +2.2 (witness) |
| 30 | Ravi | CONFLICT | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.2; anxiety Nadia +7.5 (witness), Pia +4.3 (witness) |
| 30 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.2; anxiety Marta +8.6 (witness), Pia +2.2 (witness) |
| 30 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 32; anxiety Marta +3.3 (witness), Nadia +2.9 (witness), Pia +1.7 (witness), Ravi +1.4, Sofia +0.6; calmer sender takes some anxiety: Ravi -2.3, Sofia +2.3; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 30 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -100.1, Bruno +58.0, Marta +42.2) |
| 30 | — | (system) | — | — | Bruno's symptom gives back 2.6 (Ana +2.6) |
| 30 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -56.2, Ana +56.2) |
| 30 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -240.3, Ana +48.1, Nadia +60.1, Pia +60.1, Ravi +72.1) |
| 30 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -151.0, Marta +75.5, Ravi +75.5) |
| 30 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -139.2, Marta +69.6, Ravi +69.6) |
| 30 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -299.2, Marta +89.8, Nadia +74.8, Pia +74.8, Sofia +59.8) |
| 30 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 30 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -45.7, Ravi +45.7) |
| 31 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 31 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 31 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 31 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 31 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 31 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 31 | Ana | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 31 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +4.3 |
| 31 | Bruno | CUTOFF | Ana | — | arrives week 32; anxiety Ana +1.0, Bruno +1.0; calmer sender takes some anxiety: Ana -2.5, Bruno +2.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open); impingement removed at once, anxiety Bruno -2.8 |
| 31 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +8.6 (witness), Pia +5.0 (witness) |
| 31 | Marta | TRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +4.1 (witness), Ravi +9.6 (witness) |
| 31 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Pia +2.5 (witness), Ravi +9.8 (witness) |
| 31 | Nadia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 31 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 32; anxiety Marta +3.5, Nadia +1.6, Pia +2.2 (witness), Ravi +8.7 (witness); calmer sender takes some anxiety: Marta -8.8, Nadia +8.8; Nadia rejects the other on Marta–Nadia (rejection +0.00; the tie stays open) |
| 31 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 32; anxiety Marta +8.3 (witness), Nadia +3.6 (witness), Pia +0.9, Ravi +3.7; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia takes 0.3 of functioning from the other |
| 31 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Pia +4.5; anxiety Nadia +3.5 (witness), Ravi +8.1 (witness) |
| 31 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.1; anxiety Marta +8.5 (witness), Pia +2.2 (witness) |
| 31 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 32; anxiety Marta +8.5 (witness), Nadia +3.7 (witness), Pia +4.5, Ravi +1.8; Ravi rejects the other on Pia–Ravi (rejection +1.00; the tie stays open); impingement removed at once, anxiety Ravi -11.8 |
| 31 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.5; anxiety Marta +4.2 (witness), Nadia +3.8 (witness), Pia +2.2 (witness) |
| 31 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.17); the sender bears the refusal: anxiety Sofia +3.0; anxiety Marta +3.3 (witness), Nadia +3.0 (witness), Pia +1.7 (witness) |
| 31 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -80.2, Bruno +46.4, Marta +33.8) |
| 31 | — | (system) | — | — | Bruno's symptom gives back 0.6 (Ana +0.6) |
| 31 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -43.3, Ana +43.3) |
| 31 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -246.6, Ana +49.3, Nadia +61.7, Pia +61.7, Ravi +74.0) |
| 31 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -150.3, Marta +75.1, Ravi +75.1) |
| 31 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -132.2, Marta +66.1, Ravi +66.1) |
| 31 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -291.7, Marta +87.5, Nadia +72.9, Pia +72.9, Sofia +58.3) |
| 31 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 31 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -51.0, Ravi +51.0) |
| 32 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.1 (witness), Pia +1.2 (witness), Ravi +4.9 |
| 32 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.7, Nadia +2.1 (witness), Pia +1.2 (witness), Ravi +2.5 (witness) |
| 32 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 32 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 32 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 32 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +3.9 |
| 32 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +1.2 |
| 32 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 32 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +3.1; anxiety Nadia +3.5 (witness), Pia +2.0 (witness), Ravi +4.0 (witness) |
| 32 | Ana | CUTOFF | Bruno | — | arrives week 33; anxiety Ana +0.8, Bruno +5.3; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ana -0.5 |
| 32 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +5.2 |
| 32 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 33; anxiety Dr Halim +0.1, Nadia +1.8 (witness), Pia +1.0 (witness), Ravi +2.7 (witness); calmer sender takes some anxiety: Dr Halim +4.8, Marta -4.8; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.55; the tie stays open); impingement removed at once, anxiety Dr Halim -2.9 |
| 32 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 33; anxiety Dr Halim +2.3, Marta +0.6, Nadia +4.2 (witness), Pia +2.4 (witness), Ravi +6.2 (witness) |
| 32 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Pia +2.5 (witness), Ravi +9.8 (witness) |
| 32 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +4.3 (witness), Ravi +9.6 (witness) |
| 32 | Nadia | TRIANGLE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +8.1; anxiety Marta +8.6 (witness), Pia +2.2 (witness) |
| 32 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.20); the sender bears the refusal: anxiety Nadia +8.1; anxiety Pia +2.2 (witness), Ravi +8.7 (witness) |
| 32 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +4.5; anxiety Marta +8.4 (witness), Nadia +3.8 (witness) |
| 32 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.5; anxiety Nadia +3.6 (witness), Ravi +8.2 (witness) |
| 32 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 33; anxiety Dr Halim +1.9, Marta +5.4 (witness), Nadia +3.7 (witness), Pia +2.1 (witness), Ravi +0.5 |
| 32 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.2; anxiety Nadia +7.8 (witness), Pia +4.5 (witness) |
| 32 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.4; anxiety Marta +4.4 (witness), Nadia +4.0 (witness), Pia +2.3 (witness) |
| 32 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.13); the sender bears the refusal: anxiety Sofia +3.1; anxiety Marta +3.4 (witness), Nadia +3.1 (witness), Pia +1.8 (witness) |
| 32 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -68.1, Bruno +39.4, Marta +28.7) |
| 32 | — | (system) | — | — | Bruno's symptom gives back 0.2 (Ana +0.2) |
| 32 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -35.6, Ana +35.6) |
| 32 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -250.7, Ana +50.1, Nadia +62.7, Pia +62.7, Ravi +75.2) |
| 32 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -149.1, Marta +74.5, Ravi +74.5) |
| 32 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -135.4, Marta +67.7, Ravi +67.7) |
| 32 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -296.9, Marta +89.1, Nadia +74.2, Pia +74.2, Sofia +59.4) |
| 32 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 32 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -55.6, Ravi +55.6) |
| 33 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.5 |
| 33 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 33 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 33 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 33 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 33 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 33 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 33 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 33 | Ana | CUTOFF | Bruno | — | arrives week 34; anxiety Ana +0.8, Bruno +6.6; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 33 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +3.1; anxiety Nadia +3.6 (witness), Pia +2.0 (witness), Ravi +3.9 (witness) |
| 33 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +5.7 |
| 33 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.15); the sender bears the refusal: anxiety Marta +6.1; anxiety Nadia +4.4 (witness), Pia +2.5 (witness), Ravi +4.9 (witness) |
| 33 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.1; anxiety Pia +2.5 (witness), Ravi +9.6 (witness) |
| 33 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.25); the sender bears the refusal: anxiety Marta +3.0; anxiety Nadia +4.3 (witness), Pia +2.5 (witness), Ravi +6.1 (witness) |
| 33 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.1; anxiety Nadia +4.4 (witness), Ravi +9.6 (witness) |
| 33 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +8.4; anxiety Pia +2.3 (witness), Ravi +8.7 (witness) |
| 33 | Nadia | TRIANGLE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +8.4; anxiety Marta +8.9 (witness), Pia +2.3 (witness) |
| 33 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +4.6; anxiety Marta +8.6 (witness), Nadia +3.9 (witness) |
| 33 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.6; anxiety Nadia +3.8 (witness), Ravi +8.2 (witness) |
| 33 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 34; anxiety Dr Halim +1.9, Marta +5.6 (witness), Nadia +3.9 (witness), Pia +2.2 (witness), Ravi +0.6 |
| 33 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.2; anxiety Marta +8.8 (witness), Pia +2.3 (witness) |
| 33 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.6; anxiety Marta +4.5 (witness), Nadia +4.1 (witness), Pia +2.4 (witness) |
| 33 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +3.1; anxiety Marta +3.3 (witness), Nadia +3.0 (witness), Pia +1.8 (witness) |
| 33 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -63.3, Bruno +36.6, Marta +26.6) |
| 33 | — | (system) | — | — | Bruno's symptom gives back 0.1 (Ana +0.1) |
| 33 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -44.8, Ana +44.8) |
| 33 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -237.4, Ana +47.5, Nadia +59.3, Pia +59.3, Ravi +71.2) |
| 33 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -159.1, Marta +79.6, Ravi +79.6) |
| 33 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -137.0, Marta +68.5, Ravi +68.5) |
| 33 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -315.2, Marta +94.6, Nadia +78.8, Pia +78.8, Sofia +63.0) |
| 33 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 33 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -57.9, Ravi +57.9) |
| 34 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 34 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 34 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 34 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 34 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 34 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 34 | Ana | CUTOFF | Bruno | — | arrives week 35; anxiety Ana +0.9, Bruno +6.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 34 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +3.2; anxiety Nadia +3.7 (witness), Pia +2.2 (witness), Ravi +4.0 (witness) |
| 34 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +5.9 |
| 34 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +6.1; anxiety Nadia +4.6 (witness), Pia +2.7 (witness), Ravi +5.0 (witness) |
| 34 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.2; anxiety Pia +2.7 (witness), Ravi +9.9 (witness) |
| 34 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.26); the sender bears the refusal: anxiety Marta +3.1; anxiety Nadia +4.5 (witness), Pia +2.6 (witness), Ravi +6.4 (witness) |
| 34 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 35; anxiety Marta +2.0, Nadia +4.5 (witness), Pia +6.0, Ravi +9.9 (witness); Marta rejects the other on Marta–Pia (rejection +0.00; the tie stays open); impingement removed at once, anxiety Marta -0.2 |
| 34 | Nadia | TRIANGLE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +8.6; anxiety Marta +8.9 (witness), Pia +2.4 (witness) |
| 34 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +8.6; anxiety Pia +2.4 (witness), Ravi +8.8 (witness) |
| 34 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +4.9; anxiety Nadia +3.9 (witness), Ravi +8.5 (witness) |
| 34 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +4.9; anxiety Marta +8.7 (witness), Nadia +4.0 (witness) |
| 34 | Ravi | CONFLICT | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.4; anxiety Marta +8.9 (witness), Pia +2.4 (witness) |
| 34 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 35; anxiety Dr Halim +2.0, Marta +5.7 (witness), Nadia +4.0 (witness), Pia +2.4 (witness), Ravi +0.6 |
| 34 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.6; anxiety Marta +4.5 (witness), Nadia +4.2 (witness), Pia +2.5 (witness) |
| 34 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +3.2; anxiety Marta +3.2 (witness), Nadia +3.1 (witness), Pia +1.8 (witness) |
| 34 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -67.9, Bruno +39.3, Marta +28.6) |
| 34 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 34 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -50.5, Ana +50.5) |
| 34 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -263.8, Ana +52.8, Nadia +65.9, Pia +65.9, Ravi +79.1) |
| 34 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -163.0, Marta +81.5, Ravi +81.5) |
| 34 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -145.7, Marta +72.8, Ravi +72.8) |
| 34 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -328.9, Marta +98.7, Nadia +82.2, Pia +82.2, Sofia +65.8) |
| 34 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 34 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -58.9, Ravi +58.9) |
| 35 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 35 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 35 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 35 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 35 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 35 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 35 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 35 | Ana | CUTOFF | Bruno | — | arrives week 36; anxiety Ana +0.9, Bruno +4.4; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 35 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +3.3; anxiety Nadia +3.9 (witness), Pia +2.3 (witness), Ravi +4.1 (witness) |
| 35 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +6.0 |
| 35 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +10.2; anxiety Nadia +4.7 (witness), Ravi +10.0 (witness) |
| 35 | Marta | TRIANGLE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.20); the sender bears the refusal: anxiety Marta +10.2; anxiety Nadia +9.5 (witness), Pia +5.6 (witness) |
| 35 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +6.1; anxiety Nadia +4.7 (witness), Pia +2.8 (witness), Ravi +5.0 (witness) |
| 35 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 36; anxiety Marta +8.9 (witness), Nadia +1.8, Pia +2.5 (witness), Ravi +5.6; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 35 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +8.9; anxiety Pia +2.5 (witness), Ravi +9.0 (witness) |
| 35 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +5.1; anxiety Marta +8.8 (witness), Nadia +4.2 (witness) |
| 35 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +5.1; anxiety Nadia +4.1 (witness), Ravi +8.7 (witness) |
| 35 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.6; anxiety Marta +4.5 (witness), Nadia +4.2 (witness), Pia +2.6 (witness) |
| 35 | Ravi | PURSUE | Pia | Marta, Nadia | refused by Pia (rejection +0.23); the sender bears the refusal: anxiety Ravi +9.4; anxiety Marta +9.0 (witness), Nadia +4.3 (witness) |
| 35 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Ravi +9.4; anxiety Nadia +8.5 (witness), Pia +5.0 (witness) |
| 35 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +3.2; anxiety Marta +3.2 (witness), Nadia +3.1 (witness), Pia +1.9 (witness) |
| 35 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -76.2, Bruno +44.1, Marta +32.1) |
| 35 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 35 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -52.5, Ana +52.5) |
| 35 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -266.6, Ana +53.3, Nadia +66.6, Pia +66.6, Ravi +80.0) |
| 35 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -167.4, Marta +83.7, Ravi +83.7) |
| 35 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -156.2, Marta +78.1, Ravi +78.1) |
| 35 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -339.2, Marta +101.8, Nadia +84.8, Pia +84.8, Sofia +67.8) |
| 35 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 35 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -60.7, Ravi +60.7) |
| 36 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +1.4 (witness), Ravi +4.9 |
| 36 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +1.4 (witness), Ravi +2.4 (witness) |
| 36 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 36 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 36 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +7.3 |
| 36 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 36 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 36 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 36 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 36 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 36 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 36 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 36 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 36 | Ana | CUTOFF | Bruno | — | arrives week 37; anxiety Ana +0.9, Bruno +4.6; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 36 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +3.4; anxiety Nadia +3.9 (witness), Pia +2.4 (witness), Ravi +4.2 (witness) |
| 36 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +6.2 |
| 36 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +10.3; anxiety Pia +2.9 (witness), Ravi +10.1 (witness) |
| 36 | Marta | TRIANGLE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.3; anxiety Nadia +9.6 (witness), Pia +5.9 (witness) |
| 36 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.14); the sender bears the refusal: anxiety Marta +3.1; anxiety Nadia +4.7 (witness), Pia +2.9 (witness), Ravi +6.4 (witness) |
| 36 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.0; anxiety Pia +2.7 (witness), Ravi +9.2 (witness) |
| 36 | Nadia | PURSUE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +9.0; anxiety Marta +9.1 (witness), Pia +2.6 (witness) |
| 36 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +5.3; anxiety Nadia +4.1 (witness), Ravi +8.7 (witness) |
| 36 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +5.3; anxiety Marta +8.8 (witness), Nadia +4.2 (witness) |
| 36 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.5; anxiety Nadia +8.6 (witness), Pia +5.3 (witness) |
| 36 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 37; anxiety Dr Halim +2.1, Marta +5.8 (witness), Nadia +4.2 (witness), Pia +2.6 (witness), Ravi +0.6 |
| 36 | Ravi | CONFLICT | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.5; anxiety Marta +9.0 (witness), Pia +2.6 (witness) |
| 36 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 38; anxiety Marta +3.1 (witness), Nadia +3.0 (witness), Pia +2.0 (witness), Ravi +2.2, Sofia +0.6; calmer sender takes some anxiety: Ravi -2.7, Sofia +2.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 36 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -78.0, Bruno +45.2, Marta +32.8) |
| 36 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 36 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -50.7, Ana +50.7) |
| 36 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -281.3, Ana +56.3, Nadia +70.3, Pia +70.3, Ravi +84.4) |
| 36 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -185.3, Marta +92.7, Ravi +92.7) |
| 36 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -159.8, Marta +79.9, Ravi +79.9) |
| 36 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -361.3, Marta +108.4, Nadia +90.3, Pia +90.3, Sofia +72.3) |
| 36 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 36 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -58.4, Ravi +58.4) |
| 37 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 37 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 37 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 37 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 37 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 37 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 37 | Ana | CUTOFF | Bruno | — | arrives week 38; anxiety Ana +1.0, Bruno +4.6; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -2.0 |
| 37 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +3.6; anxiety Nadia +4.0 (witness), Pia +2.6 (witness), Ravi +4.1 (witness) |
| 37 | Bruno | CUTOFF | Ana | — | arrives week 38; anxiety Ana +2.0, Bruno +1.2; calmer sender takes some anxiety: Ana -3.4, Bruno +3.4; Bruno rejects the other on Ana–Bruno (rejection +0.30; the tie stays open); impingement removed at once, anxiety Bruno -14.9 |
| 37 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.05); the sender bears the refusal: anxiety Marta +3.0; anxiety Nadia +4.7 (witness), Pia +3.1 (witness), Ravi +6.3 (witness) |
| 37 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +10.1; anxiety Nadia +4.8 (witness), Ravi +9.8 (witness) |
| 37 | Marta | PURSUE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.1; anxiety Nadia +9.7 (witness), Pia +6.2 (witness) |
| 37 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.1; anxiety Pia +2.9 (witness), Ravi +9.1 (witness) |
| 37 | Nadia | PURSUE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.1; anxiety Marta +9.0 (witness), Pia +2.8 (witness) |
| 37 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +5.7; anxiety Nadia +4.2 (witness), Ravi +8.6 (witness) |
| 37 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +5.7; anxiety Marta +8.8 (witness), Nadia +4.3 (witness) |
| 37 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.4; anxiety Marta +4.4 (witness), Nadia +4.3 (witness), Pia +2.8 (witness) |
| 37 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.1; anxiety Marta +8.8 (witness), Pia +2.7 (witness) |
| 37 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 38; anxiety Marta +8.8 (witness), Nadia +4.3 (witness), Pia +6.8, Ravi +1.8; Ravi rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 37 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 37 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -79.5, Bruno +46.0, Marta +33.5) |
| 37 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 37 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -45.1, Ana +45.1) |
| 37 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -297.0, Ana +59.4, Nadia +74.2, Pia +74.2, Ravi +89.1) |
| 37 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -193.8, Marta +96.9, Ravi +96.9) |
| 37 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -176.2, Marta +88.1, Ravi +88.1) |
| 37 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -370.1, Marta +111.0, Nadia +92.5, Pia +92.5, Sofia +74.0) |
| 37 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 37 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -61.0, Ravi +61.0) |
| 38 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 38 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 38 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 38 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 38 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 38 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 38 | Ana | CUTOFF | Bruno | — | arrives week 39; anxiety Ana +1.0, Bruno +9.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 38 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +3.8; anxiety Nadia +4.0 (witness), Pia +2.6 (witness), Ravi +4.0 (witness) |
| 38 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +5.9 |
| 38 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +10.1; anxiety Pia +3.2 (witness), Ravi +10.0 (witness) |
| 38 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.05); the sender bears the refusal: anxiety Marta +3.0; anxiety Nadia +4.9 (witness), Pia +3.1 (witness), Ravi +6.3 (witness) |
| 38 | Marta | PURSUE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.1; anxiety Nadia +9.9 (witness), Pia +6.3 (witness) |
| 38 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 39; anxiety Ana +5.7, Marta +1.2, Nadia +5.0 (witness), Pia +3.2 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 38 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.2; anxiety Pia +2.9 (witness), Ravi +8.9 (witness) |
| 38 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.2; anxiety Marta +8.9 (witness), Pia +2.8 (witness) |
| 38 | Pia | TRIANGLE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +5.8; anxiety Marta +8.6 (witness), Nadia +4.3 (witness) |
| 38 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +5.8; anxiety Nadia +4.3 (witness), Ravi +8.5 (witness) |
| 38 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +8.9; anxiety Marta +8.6 (witness), Pia +2.7 (witness) |
| 38 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.22); the sender bears the refusal: anxiety Ravi +8.9; anxiety Marta +8.6 (witness), Nadia +4.3 (witness) |
| 38 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +8.9; anxiety Nadia +8.6 (witness), Pia +5.5 (witness) |
| 38 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 40; anxiety Marta +3.3 (witness), Nadia +3.2 (witness), Pia +2.1 (witness), Ravi +2.2, Sofia +0.7; calmer sender takes some anxiety: Ravi -2.7, Sofia +2.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 38 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -73.5, Bruno +42.6, Marta +31.0) |
| 38 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 38 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -27.1, Ana +27.1) |
| 38 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -296.0, Ana +59.2, Nadia +74.0, Pia +74.0, Ravi +88.8) |
| 38 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -184.9, Marta +92.5, Ravi +92.5) |
| 38 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -171.4, Marta +85.7, Ravi +85.7) |
| 38 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -345.7, Marta +103.7, Nadia +86.4, Pia +86.4, Sofia +69.1) |
| 38 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 38 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -61.2, Ravi +61.2) |
| 39 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 39 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 39 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 39 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 39 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 39 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 39 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +5.3 |
| 39 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +4.0; anxiety Nadia +4.0 (witness), Pia +2.6 (witness), Ravi +4.0 (witness) |
| 39 | Bruno | CUTOFF | Ana | — | arrives week 40; anxiety Ana +2.5, Bruno +1.4; calmer sender takes some anxiety: Ana -3.8, Bruno +3.8; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open); impingement removed at once, anxiety Bruno -6.8 |
| 39 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.05); the sender bears the refusal: anxiety Marta +3.1; anxiety Nadia +4.9 (witness), Pia +3.1 (witness), Ravi +6.3 (witness) |
| 39 | Marta | CONFLICT | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.2; anxiety Nadia +10.0 (witness), Pia +6.4 (witness) |
| 39 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +10.2; anxiety Nadia +4.9 (witness), Ravi +9.7 (witness) |
| 39 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.3; anxiety Pia +2.9 (witness), Ravi +9.0 (witness) |
| 39 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.3; anxiety Marta +9.0 (witness), Pia +2.8 (witness) |
| 39 | Pia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 39 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +5.8; anxiety Marta +8.8 (witness), Nadia +4.4 (witness) |
| 39 | Ravi | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 39 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.1; anxiety Marta +8.9 (witness), Pia +2.8 (witness) |
| 39 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.6; anxiety Marta +4.4 (witness), Nadia +4.4 (witness), Pia +2.9 (witness) |
| 39 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.17); the sender bears the refusal: anxiety Sofia +3.3; anxiety Marta +3.2 (witness), Nadia +3.2 (witness), Pia +2.1 (witness) |
| 39 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -68.0, Bruno +39.4, Marta +28.6) |
| 39 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 39 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -50.6, Ana +50.6) |
| 39 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -282.9, Ana +56.6, Nadia +70.7, Pia +70.7, Ravi +84.9) |
| 39 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -186.9, Marta +93.5, Ravi +93.5) |
| 39 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -169.0, Marta +84.5, Ravi +84.5) |
| 39 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -367.3, Marta +110.2, Nadia +91.8, Pia +91.8, Sofia +73.5) |
| 39 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 39 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -57.7, Ravi +57.7) |
| 40 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.3 (witness), Nadia +2.3 (witness), Pia +1.5 (witness), Ravi +4.7 |
| 40 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.7, Nadia +2.3 (witness), Pia +1.5 (witness), Ravi +2.3 (witness) |
| 40 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 40 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 40 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 40 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 40 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 40 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 40 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 40 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 40 | Ana | CUTOFF | Bruno | — | arrives week 41; anxiety Ana +1.2, Bruno +10.8; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ana -4.3 |
| 40 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +4.4; anxiety Nadia +4.1 (witness), Pia +2.8 (witness), Ravi +4.2 (witness) |
| 40 | Bruno | CUTOFF | Ana | — | arrives week 41; anxiety Ana +2.6, Bruno +1.2; calmer sender takes some anxiety: Ana -2.8, Bruno +2.8; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Bruno -10.8 |
| 40 | Dr Halim | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 41; anxiety Dr Halim +0.1, Marta +2.7 (witness), Nadia +2.0 (witness), Pia +1.3 (witness); calmer sender takes some anxiety: Dr Halim +2.6, Ravi -2.6; Dr Halim rejects the other on Dr Halim–Ravi (rejection +0.55; the tie stays open); impingement removed at once, anxiety Dr Halim -0.4 |
| 40 | Marta | CONFLICT | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.1; anxiety Nadia +10.1 (witness), Pia +6.7 (witness) |
| 40 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.1; anxiety Nadia +4.9 (witness), Ravi +10.0 (witness) |
| 40 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.25); the sender bears the refusal: anxiety Marta +6.0; anxiety Nadia +5.0 (witness), Pia +3.3 (witness), Ravi +5.1 (witness) |
| 40 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.2; anxiety Pia +3.0 (witness), Ravi +9.3 (witness) |
| 40 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.2; anxiety Marta +8.9 (witness), Pia +2.9 (witness) |
| 40 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +5.9; anxiety Nadia +4.3 (witness), Ravi +8.7 (witness) |
| 40 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +5.9; anxiety Marta +8.6 (witness), Nadia +4.3 (witness) |
| 40 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.0; anxiety Marta +8.5 (witness), Pia +2.8 (witness) |
| 40 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.0; anxiety Marta +8.6 (witness), Nadia +4.3 (witness) |
| 40 | Ravi | CONFLICT | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.0; anxiety Nadia +8.7 (witness), Pia +5.8 (witness) |
| 40 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 40 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -81.7, Bruno +47.3, Marta +34.4) |
| 40 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 40 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -31.3, Ana +31.3) |
| 40 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -289.8, Ana +58.0, Nadia +72.4, Pia +72.4, Ravi +86.9) |
| 40 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -179.4, Marta +89.7, Ravi +89.7) |
| 40 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -159.6, Marta +79.8, Ravi +79.8) |
| 40 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -328.0, Marta +98.4, Nadia +82.0, Pia +82.0, Sofia +65.6) |
| 40 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 40 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -57.7, Ravi +57.7) |
| 41 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 41 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 41 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 41 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 41 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 41 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 41 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 41 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 41 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +4.5; anxiety Nadia +4.0 (witness), Pia +2.8 (witness), Ravi +4.0 (witness) |
| 41 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +6.0 |
| 41 | Bruno | CUTOFF | Ana | — | arrives week 42; anxiety Ana +2.1, Bruno +1.4; calmer sender takes some anxiety: Ana -3.1, Bruno +3.1; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 41 | Marta | TRIANGLE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +10.0 (witness), Pia +7.1 (witness) |
| 41 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 42; anxiety Ana +7.7, Marta +1.2, Nadia +5.0 (witness), Pia +3.5 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 41 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +3.0; anxiety Nadia +4.9 (witness), Pia +3.5 (witness), Ravi +6.3 (witness) |
| 41 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +5.0 (witness), Ravi +9.8 (witness) |
| 41 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.3; anxiety Marta +8.9 (witness), Pia +3.2 (witness) |
| 41 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.3; anxiety Pia +3.2 (witness), Ravi +9.1 (witness) |
| 41 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +6.4; anxiety Marta +8.6 (witness), Nadia +4.3 (witness) |
| 41 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +6.4; anxiety Nadia +4.3 (witness), Ravi +8.6 (witness) |
| 41 | Ravi | TRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.1; anxiety Nadia +8.9 (witness), Pia +6.3 (witness) |
| 41 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 42; anxiety Marta +8.7 (witness), Nadia +10.6, Pia +3.1 (witness), Ravi +1.8; Ravi rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ravi -0.1 |
| 41 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +5.7; anxiety Marta +4.2 (witness), Nadia +4.4 (witness), Pia +3.1 (witness) |
| 41 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 43; anxiety Marta +3.1 (witness), Nadia +3.3 (witness), Pia +2.3 (witness), Ravi +1.7, Sofia +0.7; calmer sender takes some anxiety: Ravi -3.2, Sofia +3.2; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 41 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -61.4, Bruno +35.6, Marta +25.9) |
| 41 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 41 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -32.9, Ana +32.9) |
| 41 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -281.0, Ana +56.2, Nadia +70.2, Pia +70.2, Ravi +84.3) |
| 41 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -190.3, Marta +95.2, Ravi +95.2) |
| 41 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -165.1, Marta +82.5, Ravi +82.5) |
| 41 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -359.3, Marta +107.8, Nadia +89.8, Pia +89.8, Sofia +71.9) |
| 41 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 41 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -61.6, Ravi +61.6) |
| 42 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 42 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 42 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 42 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 42 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 42 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 42 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 42 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 42 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 42 | Ana | CUTOFF | Bruno | — | arrives week 43; anxiety Ana +1.4, Bruno +11.4; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ana -4.2 |
| 42 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +5.1; anxiety Nadia +4.1 (witness), Pia +2.9 (witness), Ravi +4.3 (witness) |
| 42 | Bruno | CUTOFF | Ana | — | arrives week 43; anxiety Ana +2.7, Bruno +1.4; calmer sender takes some anxiety: Ana -3.1, Bruno +3.1; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Bruno -11.4 |
| 42 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +5.8; anxiety Nadia +5.0 (witness), Pia +3.5 (witness), Ravi +5.2 (witness) |
| 42 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +9.6; anxiety Nadia +4.9 (witness), Ravi +10.2 (witness) |
| 42 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +9.6; anxiety Nadia +9.9 (witness), Pia +7.0 (witness) |
| 42 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 43; anxiety Marta +1.9, Nadia +11.1, Pia +3.5 (witness), Ravi +10.2 (witness); Marta rejects the other on Marta–Nadia (rejection +0.00; the tie stays open) |
| 42 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.4; anxiety Pia +3.2 (witness), Ravi +9.4 (witness) |
| 42 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.4; anxiety Marta +8.6 (witness), Pia +3.2 (witness) |
| 42 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 42 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 43; anxiety Marta +8.3 (witness), Nadia +4.3 (witness), Pia +1.3, Ravi +8.8; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Pia -0.8 |
| 42 | Ravi | CONFLICT | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.4; anxiety Marta +8.3 (witness), Pia +3.0 (witness) |
| 42 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.4; anxiety Marta +8.3 (witness), Nadia +4.3 (witness) |
| 42 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.5; anxiety Marta +3.8 (witness), Nadia +4.3 (witness), Pia +3.2 (witness) |
| 42 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.17); the sender bears the refusal: anxiety Sofia +3.4; anxiety Marta +2.8 (witness), Nadia +3.2 (witness), Pia +2.4 (witness) |
| 42 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -76.1, Bruno +44.1, Marta +32.1) |
| 42 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 42 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -38.9, Ana +38.9) |
| 42 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -273.4, Ana +54.7, Nadia +68.3, Pia +68.3, Ravi +82.0) |
| 42 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -193.2, Marta +96.6, Ravi +96.6) |
| 42 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -167.0, Marta +83.5, Ravi +83.5) |
| 42 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -344.8, Marta +103.4, Nadia +86.2, Pia +86.2, Sofia +69.0) |
| 42 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 42 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -55.6, Ravi +55.6) |
| 43 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 43 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 43 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 43 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 43 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 43 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 43 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +7.3 |
| 43 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +5.5; anxiety Nadia +4.0 (witness), Pia +3.1 (witness), Ravi +4.1 (witness) |
| 43 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +7.3 |
| 43 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +5.3; anxiety Nadia +5.0 (witness), Pia +3.8 (witness), Ravi +5.1 (witness) |
| 43 | Marta | TRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +8.9; anxiety Nadia +4.9 (witness), Ravi +10.0 (witness) |
| 43 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 44; anxiety Marta +1.8, Nadia +3.5, Pia +3.7 (witness), Ravi +10.0 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 43 | Nadia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 43 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.3; anxiety Marta +8.0 (witness), Pia +3.4 (witness) |
| 43 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +6.7; anxiety Marta +7.5 (witness), Nadia +4.2 (witness) |
| 43 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +6.7; anxiety Nadia +4.2 (witness), Ravi +8.6 (witness) |
| 43 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 44; anxiety Marta +8.9, Nadia +8.7 (witness), Pia +6.6 (witness), Ravi +1.9; calmer sender takes some anxiety: Marta -12.3, Ravi +12.3; Ravi rejects the other on Marta–Ravi (rejection +0.05; the tie stays open) |
| 43 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 44; anxiety Marta +7.7 (witness), Nadia +4.3 (witness), Pia +7.7, Ravi +1.9; Ravi rejects the other on Pia–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ravi -5.8 |
| 43 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | refused by Dr Halim (rejection +0.23); the sender bears the refusal: anxiety Ravi +2.8; anxiety Marta +5.0 (witness), Nadia +4.3 (witness), Pia +3.3 (witness) |
| 43 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 45; anxiety Marta +2.7 (witness), Nadia +3.2 (witness), Pia +2.4 (witness), Ravi +2.0, Sofia +0.7; calmer sender takes some anxiety: Ravi -4.0, Sofia +4.0; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 43 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -64.4, Bruno +37.3, Marta +27.1) |
| 43 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 43 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -44.4, Ana +44.4) |
| 43 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -282.5, Ana +56.5, Nadia +70.6, Pia +70.6, Ravi +84.7) |
| 43 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -186.7, Marta +93.4, Ravi +93.4) |
| 43 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -153.9, Marta +76.9, Ravi +76.9) |
| 43 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -353.2, Marta +106.0, Nadia +88.3, Pia +88.3, Sofia +70.6) |
| 43 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 43 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -60.5, Ravi +60.5) |
| 44 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.0 (witness), Nadia +2.3 (witness), Pia +1.8 (witness), Ravi +4.8 |
| 44 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.0, Nadia +2.3 (witness), Pia +1.8 (witness), Ravi +2.4 (witness) |
| 44 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 44 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 44 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 44 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 44 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 44 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 44 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 44 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.1 |
| 44 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.9; anxiety Nadia +4.1 (witness), Pia +3.1 (witness), Ravi +4.4 (witness) |
| 44 | Bruno | CUTOFF | Ana | — | arrives week 45; anxiety Ana +3.0, Bruno +1.4; calmer sender takes some anxiety: Ana -3.9, Bruno +3.9; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 44 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +8.3; anxiety Nadia +4.9 (witness), Ravi +10.3 (witness) |
| 44 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +8.3; anxiety Nadia +9.7 (witness), Pia +7.3 (witness) |
| 44 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +5.0; anxiety Nadia +5.0 (witness), Pia +3.7 (witness), Ravi +5.3 (witness) |
| 44 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 45; anxiety Marta +7.4 (witness), Nadia +1.9, Pia +3.4 (witness), Ravi +7.2; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Nadia -3.2 |
| 44 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 45; anxiety Marta +5.3, Nadia +1.9, Pia +3.3 (witness), Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -11.8, Nadia +11.8; Nadia rejects the other on Marta–Nadia (rejection +0.30; the tie stays open); impingement removed at once, anxiety Nadia -7.5 |
| 44 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 44 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 45; anxiety Marta +7.2 (witness), Nadia +4.3 (witness), Pia +1.4, Ravi +8.5; calmer sender takes some anxiety: Pia +1.8, Ravi -1.8; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Pia -5.2 |
| 44 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.6; anxiety Marta +7.2 (witness), Nadia +4.3 (witness) |
| 44 | Ravi | CONFLICT | Marta | Nadia, Pia | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Ravi +9.6; anxiety Nadia +8.7 (witness), Pia +6.5 (witness) |
| 44 | Ravi | CONFLICT | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.6; anxiety Marta +7.2 (witness), Pia +3.3 (witness) |
| 44 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +3.6; anxiety Marta +2.7 (witness), Nadia +3.3 (witness), Pia +2.5 (witness) |
| 44 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -80.6, Bruno +46.7, Marta +33.9) |
| 44 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 44 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -43.0, Ana +43.0) |
| 44 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -268.5, Ana +53.7, Nadia +67.1, Pia +67.1, Ravi +80.5) |
| 44 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -179.0, Marta +89.5, Ravi +89.5) |
| 44 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -172.3, Marta +86.1, Ravi +86.1) |
| 44 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -346.9, Marta +104.1, Nadia +86.7, Pia +86.7, Sofia +69.4) |
| 44 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 44 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -67.6, Ravi +67.6) |
| 45 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 45 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 45 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 45 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 45 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 45 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 45 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 45 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 45 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.2 |
| 45 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.9; anxiety Nadia +4.2 (witness), Pia +3.2 (witness), Ravi +4.5 (witness) |
| 45 | Bruno | CUTOFF | Ana | — | arrives week 46; anxiety Ana +3.3, Bruno +1.4; calmer sender takes some anxiety: Ana -3.9, Bruno +3.9; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 45 | Marta | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 45 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +4.9; anxiety Nadia +5.0 (witness), Pia +3.8 (witness), Ravi +5.3 (witness) |
| 45 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +8.1; anxiety Nadia +4.8 (witness), Ravi +10.3 (witness) |
| 45 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 46; anxiety Marta +7.2 (witness), Nadia +1.8, Pia +3.4 (witness), Ravi +6.9; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 45 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +9.2; anxiety Pia +3.4 (witness), Ravi +9.3 (witness) |
| 45 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 46; anxiety Marta +5.4, Nadia +4.2 (witness), Pia +1.4, Ravi +9.0 (witness); calmer sender takes some anxiety: Marta -12.3, Pia +12.3; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 45 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 46; anxiety Marta +6.9 (witness), Nadia +4.2 (witness), Pia +1.4, Ravi +6.0; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 45 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +7.2 (witness), Pia +3.4 (witness) |
| 45 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +5.7; anxiety Marta +3.8 (witness), Nadia +4.5 (witness), Pia +3.6 (witness) |
| 45 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +7.2 (witness), Nadia +4.4 (witness) |
| 45 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +3.5; anxiety Marta +2.7 (witness), Nadia +3.2 (witness), Pia +2.6 (witness) |
| 45 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -80.9, Bruno +46.8, Marta +34.0) |
| 45 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 45 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -42.8, Ana +42.8) |
| 45 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -268.8, Ana +53.8, Nadia +67.2, Pia +67.2, Ravi +80.6) |
| 45 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -165.1, Marta +82.5, Ravi +82.5) |
| 45 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -150.1, Marta +75.1, Ravi +75.1) |
| 45 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -349.5, Marta +104.8, Nadia +87.4, Pia +87.4, Sofia +69.9) |
| 45 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 45 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -62.6, Ravi +62.6) |
| 46 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 46 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 46 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 46 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 46 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 46 | Ana | CUTOFF | Bruno | — | arrives week 47; anxiety Ana +1.8, Bruno +13.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ana -3.5 |
| 46 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.9; anxiety Nadia +4.2 (witness), Pia +3.4 (witness), Ravi +4.4 (witness) |
| 46 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.15); the sender bears the refusal: anxiety Bruno +7.7 |
| 46 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Marta +8.5; anxiety Nadia +9.6 (witness), Pia +7.9 (witness) |
| 46 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +8.5; anxiety Nadia +4.8 (witness), Ravi +10.0 (witness) |
| 46 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +5.1; anxiety Nadia +5.0 (witness), Pia +4.1 (witness), Ravi +5.2 (witness) |
| 46 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 47; anxiety Marta +1.7, Nadia +11.0, Pia +3.9 (witness), Ravi +10.0 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -8.0 |
| 46 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 47; anxiety Marta +5.6, Nadia +1.8, Pia +3.5 (witness), Ravi +9.0 (witness); calmer sender takes some anxiety: Marta -10.7, Nadia +10.7; Nadia rejects the other on Marta–Nadia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Nadia -11.0 |
| 46 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 47; anxiety Marta +7.5 (witness), Nadia +1.8, Pia +3.5 (witness), Ravi +3.6; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -9.8 |
| 46 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Pia +7.3; anxiety Marta +7.3 (witness), Nadia +4.3 (witness) |
| 46 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 47; anxiety Marta +5.3, Nadia +4.2 (witness), Pia +1.5, Ravi +8.8 (witness); calmer sender takes some anxiety: Marta -12.0, Pia +12.0; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 46 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.5; anxiety Nadia +8.8 (witness), Pia +7.1 (witness) |
| 46 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.5; anxiety Marta +7.5 (witness), Nadia +4.4 (witness) |
| 46 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 47; anxiety Marta +7.5 (witness), Nadia +9.8, Pia +3.6 (witness), Ravi +1.9; Ravi rejects the other on Nadia–Ravi (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ravi -11.5 |
| 46 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 48; anxiety Marta +2.8 (witness), Nadia +3.2 (witness), Pia +2.7 (witness), Ravi +1.8, Sofia +0.7; calmer sender takes some anxiety: Ravi -2.7, Sofia +2.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 46 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -79.5, Bruno +46.0, Marta +33.5) |
| 46 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 46 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -52.8, Ana +52.8) |
| 46 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -239.5, Ana +47.9, Nadia +59.9, Pia +59.9, Ravi +71.9) |
| 46 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -147.3, Marta +73.6, Ravi +73.6) |
| 46 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -145.4, Marta +72.7, Ravi +72.7) |
| 46 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -320.9, Marta +96.3, Nadia +80.2, Pia +80.2, Sofia +64.2) |
| 46 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 46 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -53.4, Ravi +53.4) |
| 47 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 47 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 47 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +4.9 |
| 47 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 47 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 47 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 47 | Ana | CUTOFF | Bruno | — | arrives week 48; anxiety Ana +1.8, Bruno +8.8; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 47 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.8; anxiety Nadia +4.2 (witness), Pia +3.5 (witness), Ravi +4.2 (witness) |
| 47 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.7 |
| 47 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +8.6; anxiety Nadia +4.8 (witness), Ravi +9.7 (witness) |
| 47 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +5.1; anxiety Nadia +5.0 (witness), Pia +4.2 (witness), Ravi +5.1 (witness) |
| 47 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +8.6; anxiety Nadia +9.6 (witness), Pia +8.0 (witness) |
| 47 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 48; anxiety Marta +4.6, Nadia +1.9, Pia +3.7 (witness), Ravi +8.9 (witness); calmer sender takes some anxiety: Marta -11.5, Nadia +11.5; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 47 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 48; anxiety Marta +7.7 (witness), Nadia +1.9, Pia +3.7 (witness), Ravi +5.6; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 47 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 48; anxiety Marta +7.6 (witness), Nadia +4.3 (witness), Pia +1.5, Ravi +6.0; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 47 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 48; anxiety Marta +3.0, Nadia +4.3 (witness), Pia +1.5, Ravi +8.8 (witness); calmer sender takes some anxiety: Marta -13.1, Pia +13.1; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 47 | Ravi | WITHHOLD | Pia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 47 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 48; anxiety Marta +7.7, Nadia +8.6 (witness), Pia +7.2 (witness), Ravi +1.8; calmer sender takes some anxiety: Marta -13.1, Ravi +13.1; Ravi rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 47 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | refused by Dr Halim (rejection +0.23); the sender bears the refusal: anxiety Ravi +2.8; anxiety Marta +4.9 (witness), Nadia +4.3 (witness), Pia +3.6 (witness) |
| 47 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 47 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -72.5, Bruno +42.0, Marta +30.5) |
| 47 | — | (system) | — | — | Bruno's symptom gives back 0.0 (Ana +0.0) |
| 47 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -64.1, Ana +64.1) |
| 47 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -223.1, Ana +44.6, Nadia +55.8, Pia +55.8, Ravi +66.9) |
| 47 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -162.8, Marta +81.4, Ravi +81.4) |
| 47 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -163.2, Marta +81.6, Ravi +81.6) |
| 47 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -316.7, Marta +95.0, Nadia +79.2, Pia +79.2, Sofia +63.3) |
| 47 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 47 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -51.0, Ravi +51.0) |
| 48 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.0 (witness), Nadia +2.3 (witness), Pia +1.9 (witness), Ravi +4.7 |
| 48 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.0, Nadia +2.3 (witness), Pia +1.9 (witness), Ravi +2.4 (witness) |
| 48 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 48 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 48 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 48 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 48 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 48 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 48 | Ana | CUTOFF | Bruno | — | arrives week 49; anxiety Ana +1.8, Bruno +6.3; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 48 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.9; anxiety Nadia +4.2 (witness), Pia +3.7 (witness), Ravi +4.4 (witness) |
| 48 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.8 |
| 48 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 49; anxiety Dr Halim +0.1, Nadia +1.8 (witness), Pia +1.6 (witness), Ravi +2.5 (witness); calmer sender takes some anxiety: Dr Halim +6.4, Marta -6.4 |
| 48 | Marta | TRIANGLE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +8.7; anxiety Nadia +9.5 (witness), Pia +8.2 (witness) |
| 48 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +5.2; anxiety Nadia +5.0 (witness), Pia +4.3 (witness), Ravi +5.2 (witness) |
| 48 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +8.7; anxiety Nadia +4.8 (witness), Ravi +9.8 (witness) |
| 48 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +8.7; anxiety Pia +4.1 (witness), Ravi +9.9 (witness) |
| 48 | Nadia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 48 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +9.2; anxiety Pia +3.8 (witness), Ravi +9.1 (witness) |
| 48 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 49; anxiety Marta +3.0, Nadia +4.3 (witness), Pia +1.6, Ravi +8.8 (witness); calmer sender takes some anxiety: Marta -13.6, Pia +13.6; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 48 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 49; anxiety Marta +7.7 (witness), Nadia +4.3 (witness), Pia +1.6, Ravi +3.6; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 48 | Ravi | WITHHOLD | Nadia | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 48 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.4; anxiety Marta +7.8 (witness), Nadia +4.3 (witness) |
| 48 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +5.8; anxiety Marta +4.0 (witness), Nadia +4.4 (witness), Pia +3.7 (witness) |
| 48 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Sofia +3.7; anxiety Marta +2.9 (witness), Nadia +3.2 (witness), Pia +2.8 (witness) |
| 48 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -81.9, Bruno +47.4, Marta +34.5) |
| 48 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -60.2, Ana +60.2) |
| 48 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -227.2, Ana +45.4, Nadia +56.8, Pia +56.8, Ravi +68.1) |
| 48 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -163.4, Marta +81.7, Ravi +81.7) |
| 48 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -155.6, Marta +77.8, Ravi +77.8) |
| 48 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -322.5, Marta +96.7, Nadia +80.6, Pia +80.6, Sofia +64.5) |
| 48 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 48 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -53.0, Ravi +53.0) |
| 49 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 49 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 49 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 49 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 49 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 49 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 49 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 49 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 49 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 49 | Ana | CUTOFF | Bruno | — | arrives week 50; anxiety Ana +1.8, Bruno +6.3; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 49 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.9; anxiety Nadia +4.2 (witness), Pia +3.7 (witness), Ravi +4.5 (witness) |
| 49 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.7 |
| 49 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +8.9; anxiety Nadia +4.8 (witness), Ravi +10.1 (witness) |
| 49 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +5.4; anxiety Nadia +5.0 (witness), Pia +4.3 (witness), Ravi +5.3 (witness) |
| 49 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +8.9; anxiety Nadia +9.5 (witness), Pia +8.2 (witness) |
| 49 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 50; anxiety Marta +5.1, Nadia +1.8, Pia +3.7 (witness), Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -11.0, Nadia +11.0; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 49 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 50; anxiety Marta +7.9 (witness), Nadia +1.8, Pia +3.7 (witness), Ravi +6.1; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 49 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 49 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 50; anxiety Marta +3.1, Nadia +4.2 (witness), Pia +1.5, Ravi +9.0 (witness); calmer sender takes some anxiety: Marta -12.6, Pia +12.6; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 49 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.6; anxiety Marta +7.9 (witness), Pia +3.7 (witness) |
| 49 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.8; anxiety Marta +4.1 (witness), Nadia +4.4 (witness), Pia +3.9 (witness) |
| 49 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Ravi +9.6; anxiety Nadia +8.6 (witness), Pia +7.4 (witness) |
| 49 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +3.7; anxiety Marta +3.0 (witness), Nadia +3.2 (witness), Pia +2.9 (witness) |
| 49 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -80.7, Bruno +46.7, Marta +34.0) |
| 49 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -57.3, Ana +57.3) |
| 49 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -231.3, Ana +46.3, Nadia +57.8, Pia +57.8, Ravi +69.4) |
| 49 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -147.5, Marta +73.7, Ravi +73.7) |
| 49 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -146.4, Marta +73.2, Ravi +73.2) |
| 49 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -306.8, Marta +92.0, Nadia +76.7, Pia +76.7, Sofia +61.4) |
| 49 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 49 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -54.7, Ravi +54.7) |
| 50 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 50 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 50 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 50 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.5 |
| 50 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 50 | Ana | CUTOFF | Bruno | — | arrives week 51; anxiety Ana +1.8, Bruno +6.4; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 50 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.9; anxiety Nadia +4.1 (witness), Pia +3.7 (witness), Ravi +4.4 (witness) |
| 50 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.9 |
| 50 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +9.2; anxiety Nadia +4.7 (witness), Ravi +10.1 (witness) |
| 50 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 51; anxiety Ana +11.5, Marta +1.1, Nadia +4.9 (witness), Pia +4.4 (witness), Ravi +5.2 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 50 | Marta | TRIANGLE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +9.2; anxiety Nadia +9.4 (witness), Pia +8.4 (witness) |
| 50 | Nadia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 50 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 51; anxiety Marta +4.8, Nadia +1.8, Pia +3.8 (witness), Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -10.9, Nadia +10.9; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 50 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 51; anxiety Marta +8.1 (witness), Nadia +4.3 (witness), Pia +1.6, Ravi +5.7; calmer sender takes some anxiety: Pia +1.4, Ravi -1.4; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 50 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 51; anxiety Marta +3.2, Nadia +4.2 (witness), Pia +1.6, Ravi +9.0 (witness); calmer sender takes some anxiety: Marta -12.5, Pia +12.5; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 50 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | refused by Dr Halim (rejection +0.24); the sender bears the refusal: anxiety Ravi +2.9; anxiety Marta +5.3 (witness), Nadia +4.3 (witness), Pia +3.8 (witness) |
| 50 | Ravi | CONFLICT | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.6; anxiety Marta +8.2 (witness), Nadia +4.3 (witness) |
| 50 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.6; anxiety Marta +8.2 (witness), Pia +3.8 (witness) |
| 50 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +3.6; anxiety Marta +3.0 (witness), Nadia +3.2 (witness), Pia +3.0 (witness) |
| 50 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -77.3, Bruno +44.8, Marta +32.6) |
| 50 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -55.1, Ana +55.1) |
| 50 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -210.7, Ana +42.1, Nadia +52.7, Pia +52.7, Ravi +63.2) |
| 50 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -147.2, Marta +73.6, Ravi +73.6) |
| 50 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -147.7, Marta +73.9, Ravi +73.9) |
| 50 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -300.6, Marta +90.2, Nadia +75.2, Pia +75.2, Sofia +60.1) |
| 50 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 50 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -51.0, Ravi +51.0) |
| 51 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 51 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 51 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 51 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 51 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +3.9 |
| 51 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 51 | Ana | CUTOFF | Bruno | — | arrives week 52; anxiety Ana +1.8, Bruno +6.4; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 51 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.9; anxiety Nadia +4.1 (witness), Pia +4.0 (witness), Ravi +4.2 (witness) |
| 51 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.7 |
| 51 | Marta | TRIANGLE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +9.1; anxiety Nadia +9.3 (witness), Pia +8.9 (witness) |
| 51 | Marta | TRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +9.1; anxiety Nadia +4.6 (witness), Ravi +9.4 (witness) |
| 51 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +5.5; anxiety Nadia +4.8 (witness), Pia +4.6 (witness), Ravi +4.9 (witness) |
| 51 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 52; anxiety Marta +1.8, Nadia +10.9, Pia +4.5 (witness), Ravi +9.4 (witness); Marta rejects the other on Marta–Nadia (rejection +0.00; the tie stays open); impingement removed at once, anxiety Marta -6.8 |
| 51 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 52; anxiety Marta +8.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +5.6; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 51 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +8.9; anxiety Pia +4.1 (witness), Ravi +8.6 (witness) |
| 51 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 52; anxiety Marta +8.0 (witness), Nadia +4.2 (witness), Pia +1.7, Ravi +3.8; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 51 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 52; anxiety Marta +3.2, Nadia +4.2 (witness), Pia +1.7, Ravi +8.4 (witness); calmer sender takes some anxiety: Marta -12.6, Pia +12.6; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 51 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.0; anxiety Marta +8.1 (witness), Pia +4.1 (witness) |
| 51 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +5.6; anxiety Marta +4.0 (witness), Nadia +4.3 (witness), Pia +4.2 (witness) |
| 51 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.0; anxiety Marta +8.1 (witness), Nadia +4.3 (witness) |
| 51 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +3.6; anxiety Marta +2.9 (witness), Nadia +3.2 (witness), Pia +3.0 (witness) |
| 51 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -83.3, Bruno +48.2, Marta +35.1) |
| 51 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -59.0, Ana +59.0) |
| 51 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -221.8, Ana +44.4, Nadia +55.4, Pia +55.4, Ravi +66.5) |
| 51 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -150.3, Marta +75.1, Ravi +75.1) |
| 51 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -145.5, Marta +72.7, Ravi +72.7) |
| 51 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -306.4, Marta +91.9, Nadia +76.6, Pia +76.6, Sofia +61.3) |
| 51 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 51 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -49.1, Ravi +49.1) |
| 51 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |
| 52 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.1 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +4.7 |
| 52 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.3, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.3 (witness) |
| 52 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 52 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 52 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 52 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 52 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 52 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 52 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 52 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 52 | Ana | CUTOFF | Bruno | — | arrives week 53; anxiety Ana +1.8, Bruno +6.4; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 52 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.8; anxiety Nadia +4.1 (witness), Pia +4.0 (witness), Ravi +4.3 (witness) |
| 52 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.7 |
| 52 | Marta | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 52 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +9.0; anxiety Nadia +4.7 (witness), Ravi +9.8 (witness) |
| 52 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +5.4; anxiety Nadia +4.8 (witness), Pia +4.7 (witness), Ravi +5.1 (witness) |
| 52 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 53; anxiety Marta +4.0, Nadia +1.8, Pia +4.1 (witness), Ravi +9.0 (witness); calmer sender takes some anxiety: Marta -11.1, Nadia +11.1; Nadia rejects the other on Marta–Nadia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Nadia -7.5 |
| 52 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 53; anxiety Marta +8.1 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +4.8; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -10.7 |
| 52 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.24); the sender bears the refusal: anxiety Pia +8.5; anxiety Nadia +4.2 (witness), Ravi +8.8 (witness) |
| 52 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.24); the sender bears the refusal: anxiety Pia +8.5; anxiety Marta +8.0 (witness), Nadia +4.3 (witness) |
| 52 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.4; anxiety Marta +8.1 (witness), Nadia +4.3 (witness) |
| 52 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 53; anxiety Marta +8.0 (witness), Nadia +10.7, Pia +4.1 (witness), Ravi +1.9; Ravi rejects the other on Nadia–Ravi (rejection +0.30; the tie stays open); impingement removed at once, anxiety Ravi -11.5 |
| 52 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.6; anxiety Marta +4.1 (witness), Nadia +4.3 (witness), Pia +4.2 (witness) |
| 52 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +3.9; anxiety Marta +3.0 (witness), Nadia +3.2 (witness), Pia +3.1 (witness) |
| 52 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -76.4, Bruno +44.2, Marta +32.2) |
| 52 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -55.0, Ana +55.0) |
| 52 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -229.9, Ana +46.0, Nadia +57.5, Pia +57.5, Ravi +69.0) |
| 52 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -157.4, Marta +78.7, Ravi +78.7) |
| 52 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -155.0, Marta +77.5, Ravi +77.5) |
| 52 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -323.5, Marta +97.0, Nadia +80.9, Pia +80.9, Sofia +64.7) |
| 52 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 52 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -52.4, Ravi +52.4) |
| 53 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 53 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 53 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 53 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 53 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 53 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.8; anxiety Nadia +4.1 (witness), Pia +4.0 (witness), Ravi +4.3 (witness) |
| 53 | Ana | CUTOFF | Bruno | — | arrives week 54; anxiety Ana +1.8, Bruno +6.4; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 53 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.8 |
| 53 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +9.3; anxiety Nadia +4.7 (witness), Ravi +9.8 (witness) |
| 53 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +5.6; anxiety Nadia +4.9 (witness), Pia +4.7 (witness), Ravi +5.1 (witness) |
| 53 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 54; anxiety Marta +1.9, Nadia +9.4 (witness), Pia +9.0 (witness), Ravi +11.4; Marta rejects the other on Marta–Ravi (rejection +0.20; the tie stays open); impingement removed at once, anxiety Marta -0.7 |
| 53 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 54; anxiety Marta +8.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +4.1; calmer sender takes some anxiety: Nadia +0.0, Ravi -0.0; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 53 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 54; anxiety Marta +4.2, Nadia +1.8, Pia +4.1 (witness), Ravi +8.8 (witness); calmer sender takes some anxiety: Marta -10.6, Nadia +10.6; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 53 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.23); the sender bears the refusal: anxiety Pia +8.5; anxiety Nadia +4.2 (witness), Ravi +8.7 (witness) |
| 53 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Pia +8.5; anxiety Marta +8.1 (witness), Nadia +4.3 (witness) |
| 53 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 55; anxiety Marta +4.2 (witness), Nadia +4.2 (witness), Pia +4.1 (witness), Ravi +1.1, Sofia +6.2; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 53 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.2; anxiety Marta +8.0 (witness), Pia +4.0 (witness) |
| 53 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.1; anxiety Marta +8.1 (witness), Nadia +4.2 (witness) |
| 53 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.0; anxiety Marta +3.2 (witness), Nadia +3.2 (witness), Pia +3.1 (witness) |
| 53 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -75.6, Bruno +43.8, Marta +31.8) |
| 53 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -54.9, Ana +54.9) |
| 53 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -240.5, Ana +48.1, Nadia +60.1, Pia +60.1, Ravi +72.2) |
| 53 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -142.6, Marta +71.3, Ravi +71.3) |
| 53 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -150.4, Marta +75.2, Ravi +75.2) |
| 53 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -288.4, Marta +86.5, Nadia +72.1, Pia +72.1, Sofia +57.7) |
| 53 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 53 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -58.3, Ravi +58.3) |
| 54 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 54 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 54 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +4.9 |
| 54 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 54 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 54 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 54 | Ana | CUTOFF | Bruno | — | arrives week 55; anxiety Ana +1.8, Bruno +6.4; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -5.1 |
| 54 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.8; anxiety Nadia +4.1 (witness), Pia +4.0 (witness), Ravi +4.3 (witness) |
| 54 | Bruno | CUTOFF | Ana | — | arrives week 55; anxiety Ana +5.1, Bruno +1.8; calmer sender takes some anxiety: Ana -4.4, Bruno +4.4; Bruno rejects the other on Ana–Bruno (rejection +0.45; the tie stays open); impingement removed at once, anxiety Bruno -20.8 |
| 54 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +5.8; anxiety Nadia +4.8 (witness), Pia +4.7 (witness), Ravi +5.1 (witness) |
| 54 | Marta | PURSUE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Marta +9.6; anxiety Nadia +9.4 (witness), Pia +9.0 (witness) |
| 54 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +9.5; anxiety Pia +4.5 (witness), Ravi +9.9 (witness) |
| 54 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +9.0; anxiety Marta +8.6 (witness), Pia +4.2 (witness) |
| 54 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 55; anxiety Marta +5.2, Nadia +1.8, Pia +4.1 (witness), Ravi +9.0 (witness); calmer sender takes some anxiety: Marta -10.7, Nadia +10.7; Nadia rejects the other on Marta–Nadia (rejection +0.00; the tie stays open) |
| 54 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 55; anxiety Marta +5.2, Nadia +4.2 (witness), Pia +1.7, Ravi +8.8 (witness); calmer sender takes some anxiety: Marta -12.7, Pia +12.7; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open) |
| 54 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 55; anxiety Marta +8.4 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +5.7; calmer sender takes some anxiety: Pia +1.3, Ravi -1.3; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 54 | Ravi | DISTANCE | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.5; anxiety Nadia +8.5 (witness), Pia +8.2 (witness) |
| 54 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.5; anxiety Marta +8.6 (witness), Nadia +4.3 (witness) |
| 54 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +5.7; anxiety Marta +4.5 (witness), Nadia +4.3 (witness), Pia +4.2 (witness) |
| 54 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.5; anxiety Marta +8.6 (witness), Pia +4.2 (witness) |
| 54 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 56; anxiety Marta +3.2 (witness), Nadia +3.1 (witness), Pia +3.0 (witness), Ravi +1.3, Sofia +0.8; calmer sender takes some anxiety: Ravi -4.3, Sofia +4.3; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open); impingement removed at once, anxiety Sofia -4.1 |
| 54 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -75.5, Bruno +43.7, Marta +31.8) |
| 54 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -55.1, Ana +55.1) |
| 54 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -220.5, Ana +44.1, Nadia +55.1, Pia +55.1, Ravi +66.2) |
| 54 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -143.5, Marta +71.8, Ravi +71.8) |
| 54 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -144.1, Marta +72.0, Ravi +72.0) |
| 54 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -308.4, Marta +92.5, Nadia +77.1, Pia +77.1, Sofia +61.7) |
| 54 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 54 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -51.5, Ravi +51.5) |
| 55 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 55 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 55 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 55 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 55 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 55 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 55 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 55 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 55 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 55 | Ana | CUTOFF | Bruno | — | arrives week 56; anxiety Ana +1.8, Bruno +17.1; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 55 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.8; anxiety Nadia +4.1 (witness), Pia +3.9 (witness), Ravi +4.3 (witness) |
| 55 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +7.3 |
| 55 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +6.0; anxiety Nadia +4.9 (witness), Pia +4.7 (witness), Ravi +5.1 (witness) |
| 55 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +9.4 (witness), Pia +9.0 (witness) |
| 55 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +4.7 (witness), Ravi +9.9 (witness) |
| 55 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +9.9; anxiety Pia +4.5 (witness), Ravi +9.9 (witness) |
| 55 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 56; anxiety Marta +9.0 (witness), Nadia +1.8, Pia +4.2 (witness), Ravi +5.2; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 55 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 56; anxiety Marta +3.9, Nadia +1.8, Pia +4.2 (witness), Ravi +9.1 (witness); calmer sender takes some anxiety: Marta -12.4, Nadia +12.4; Nadia rejects the other on Marta–Nadia (rejection +0.00; the tie stays open) |
| 55 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 56; anxiety Marta +5.0, Nadia +4.2 (witness), Pia +1.7, Ravi +9.0 (witness); calmer sender takes some anxiety: Marta -14.0, Pia +14.0; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 55 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 56; anxiety Marta +8.9 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +5.2; calmer sender takes some anxiety: Pia +1.8, Ravi -1.8; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 55 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.4; anxiety Marta +8.9 (witness), Pia +4.2 (witness) |
| 55 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.4; anxiety Marta +8.9 (witness), Nadia +4.3 (witness) |
| 55 | Ravi | DISTANCE | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.4; anxiety Nadia +8.5 (witness), Pia +8.2 (witness) |
| 55 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +4.0; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness) |
| 55 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -68.0, Bruno +39.4, Marta +28.6) |
| 55 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -24.1, Ana +24.1) |
| 55 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -223.6, Ana +44.7, Nadia +55.9, Pia +55.9, Ravi +67.1) |
| 55 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -158.8, Marta +79.4, Ravi +79.4) |
| 55 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -161.5, Marta +80.7, Ravi +80.7) |
| 55 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -319.1, Marta +95.7, Nadia +79.8, Pia +79.8, Sofia +63.8) |
| 55 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 55 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -60.3, Ravi +60.3) |
| 56 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.3 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +4.8 |
| 56 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.7, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.4 (witness) |
| 56 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 56 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 56 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 56 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 56 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 56 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 56 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 56 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 56 | Ana | CUTOFF | Bruno | — | arrives week 57; anxiety Ana +1.7, Bruno +9.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 56 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.6; anxiety Nadia +3.9 (witness), Pia +3.8 (witness), Ravi +4.2 (witness) |
| 56 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.5 |
| 56 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 57; anxiety Dr Halim +0.1, Nadia +1.9 (witness), Pia +1.8 (witness), Ravi +2.6 (witness); calmer sender takes some anxiety: Dr Halim +7.0, Marta -7.0 |
| 56 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.4; anxiety Nadia +9.5 (witness), Pia +9.1 (witness) |
| 56 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.4; anxiety Nadia +4.7 (witness), Ravi +10.0 (witness) |
| 56 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +6.2; anxiety Nadia +4.9 (witness), Pia +4.7 (witness), Ravi +5.2 (witness) |
| 56 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 57; anxiety Marta +2.1, Nadia +10.9, Pia +4.6 (witness), Ravi +10.1 (witness); Marta rejects the other on Marta–Nadia (rejection +0.25; the tie stays open); impingement removed at once, anxiety Marta -11.4 |
| 56 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 57; anxiety Marta +3.6, Nadia +1.8, Pia +4.2 (witness), Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -12.8, Nadia +12.8; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -10.9 |
| 56 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 57; anxiety Marta +9.3 (witness), Nadia +1.8, Pia +4.2 (witness), Ravi +4.7; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 56 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 57; anxiety Marta +3.6, Nadia +4.3 (witness), Pia +1.7, Ravi +9.1 (witness); calmer sender takes some anxiety: Marta -14.4, Pia +14.4; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 56 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 57; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +3.7; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 56 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 57; anxiety Marta +11.1, Nadia +8.7 (witness), Pia +8.4 (witness), Ravi +1.9; calmer sender takes some anxiety: Marta -14.2, Ravi +14.2; Ravi rejects the other on Marta–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ravi -3.6 |
| 56 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | refused by Dr Halim (rejection +0.24); the sender bears the refusal: anxiety Ravi +2.9; anxiety Marta +6.1 (witness), Nadia +4.4 (witness), Pia +4.2 (witness) |
| 56 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.7; anxiety Marta +9.4 (witness), Nadia +4.4 (witness) |
| 56 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +3.9; anxiety Marta +3.3 (witness), Nadia +3.1 (witness), Pia +3.0 (witness) |
| 56 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -47.9, Bruno +27.7, Marta +20.2) |
| 56 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -51.8, Ana +51.8) |
| 56 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -244.7, Ana +48.9, Nadia +61.2, Pia +61.2, Ravi +73.4) |
| 56 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -169.8, Marta +84.9, Ravi +84.9) |
| 56 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -174.6, Marta +87.3, Ravi +87.3) |
| 56 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -366.5, Marta +109.9, Nadia +91.6, Pia +91.6, Sofia +73.3) |
| 56 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 56 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -56.0, Ravi +56.0) |
| 57 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 57 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 57 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 57 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 57 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 57 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 57 | Ana | CUTOFF | Bruno | — | arrives week 58; anxiety Ana +1.8, Bruno +6.4; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 57 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.9; anxiety Nadia +4.1 (witness), Pia +4.0 (witness), Ravi +4.5 (witness) |
| 57 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.0 |
| 57 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.3; anxiety Nadia +9.5 (witness), Pia +9.1 (witness) |
| 57 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.3; anxiety Pia +4.6 (witness), Ravi +10.3 (witness) |
| 57 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +6.2; anxiety Nadia +4.9 (witness), Pia +4.7 (witness), Ravi +5.3 (witness) |
| 57 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.3; anxiety Nadia +4.7 (witness), Ravi +10.3 (witness) |
| 57 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 58; anxiety Marta +9.4 (witness), Nadia +1.8, Pia +4.2 (witness), Ravi +5.3; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 57 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.2; anxiety Pia +4.2 (witness), Ravi +9.6 (witness) |
| 57 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.20); the sender bears the refusal: anxiety Pia +8.8; anxiety Marta +9.3 (witness), Nadia +4.4 (witness) |
| 57 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 58; anxiety Marta +3.6, Nadia +4.3 (witness), Pia +1.7, Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -14.6, Pia +14.6; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 57 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.3 (witness), Nadia +4.4 (witness) |
| 57 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | refused by Dr Halim (rejection +0.24); the sender bears the refusal: anxiety Ravi +3.0; anxiety Marta +6.1 (witness), Nadia +4.4 (witness), Pia +4.2 (witness) |
| 57 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.15); the sender bears the refusal: anxiety Ravi +5.9; anxiety Marta +4.7 (witness), Nadia +4.4 (witness), Pia +4.3 (witness) |
| 57 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.0; anxiety Marta +3.4 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 57 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -74.0, Bruno +42.8, Marta +31.2) |
| 57 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -58.1, Ana +58.1) |
| 57 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -239.4, Ana +47.9, Nadia +59.8, Pia +59.8, Ravi +71.8) |
| 57 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -178.2, Marta +89.1, Ravi +89.1) |
| 57 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -177.5, Marta +88.8, Ravi +88.8) |
| 57 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -361.0, Marta +108.3, Nadia +90.3, Pia +90.3, Sofia +72.2) |
| 57 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 57 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -64.5, Ravi +64.5) |
| 58 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 58 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 58 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 58 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 58 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 58 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 58 | Ana | CUTOFF | Bruno | — | arrives week 59; anxiety Ana +1.8, Bruno +6.4; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 58 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.9; anxiety Nadia +4.2 (witness), Pia +4.0 (witness), Ravi +4.5 (witness) |
| 58 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 58 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.4; anxiety Nadia +4.8 (witness), Ravi +10.4 (witness) |
| 58 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 59; anxiety Marta +2.1, Nadia +9.5 (witness), Pia +9.2 (witness), Ravi +11.9; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Marta -5.1 |
| 58 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.4; anxiety Pia +4.6 (witness), Ravi +10.5 (witness) |
| 58 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.1; anxiety Pia +4.2 (witness), Ravi +9.5 (witness) |
| 58 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 59; anxiety Marta +9.3 (witness), Nadia +1.8, Pia +4.2 (witness), Ravi +3.8; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 58 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +8.7; anxiety Marta +9.2 (witness), Nadia +4.4 (witness) |
| 58 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.24); the sender bears the refusal: anxiety Pia +8.7; anxiety Nadia +4.3 (witness), Ravi +9.4 (witness) |
| 58 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Ravi +9.8; anxiety Nadia +8.6 (witness), Pia +8.2 (witness) |
| 58 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.2 (witness), Pia +4.1 (witness) |
| 58 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.2 (witness), Nadia +4.3 (witness) |
| 58 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 58 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -76.8, Bruno +44.4, Marta +32.3) |
| 58 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -57.0, Ana +57.0) |
| 58 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -274.1, Ana +54.8, Nadia +68.5, Pia +68.5, Ravi +82.2) |
| 58 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -166.9, Marta +83.4, Ravi +83.4) |
| 58 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -178.9, Marta +89.5, Ravi +89.5) |
| 58 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -348.1, Marta +104.4, Nadia +87.0, Pia +87.0, Sofia +69.6) |
| 58 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 58 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -60.0, Ravi +60.0) |
| 59 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 59 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 59 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 59 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 59 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 59 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 59 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 59 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 59 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.23); the sender bears the refusal: anxiety Ana +9.2 |
| 59 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 60; anxiety Ana +1.4, Marta +7.2, Nadia +4.2 (witness), Pia +4.0 (witness), Ravi +4.5 (witness); calmer sender takes some anxiety: Ana +8.7, Marta -8.7; Ana rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 59 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.4 |
| 59 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 60; anxiety Marta +2.1, Nadia +9.6 (witness), Pia +9.2 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 59 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.4; anxiety Pia +4.6 (witness), Ravi +10.5 (witness) |
| 59 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 60; anxiety Marta +2.1, Nadia +4.8 (witness), Pia +10.5, Ravi +10.4 (witness); Marta rejects the other on Marta–Pia (rejection +0.00; the tie stays open); impingement removed at once, anxiety Marta -5.3 |
| 59 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +6.3; anxiety Nadia +4.9 (witness), Pia +4.7 (witness), Ravi +5.3 (witness) |
| 59 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.1; anxiety Pia +4.2 (witness), Ravi +9.5 (witness) |
| 59 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.24); the sender bears the refusal: anxiety Nadia +9.2; anxiety Marta +9.3 (witness), Pia +4.2 (witness) |
| 59 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +8.7; anxiety Marta +9.3 (witness), Nadia +4.4 (witness) |
| 59 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.09); the sender bears the refusal: anxiety Pia +8.7; anxiety Nadia +4.3 (witness), Ravi +9.4 (witness) |
| 59 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +6.0; anxiety Marta +4.7 (witness), Nadia +4.5 (witness), Pia +4.3 (witness) |
| 59 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Marta +9.4 (witness), Pia +4.2 (witness) |
| 59 | Ravi | CONFLICT | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Nadia +8.8 (witness), Pia +8.4 (witness) |
| 59 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +4.0; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness) |
| 59 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -81.3, Bruno +47.1, Marta +34.2) |
| 59 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -50.9, Ana +50.9) |
| 59 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -267.2, Ana +53.4, Nadia +66.8, Pia +66.8, Ravi +80.2) |
| 59 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -165.9, Marta +83.0, Ravi +83.0) |
| 59 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -174.5, Marta +87.3, Ravi +87.3) |
| 59 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -366.9, Marta +110.1, Nadia +91.7, Pia +91.7, Sofia +73.4) |
| 59 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 59 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -61.7, Ravi +61.7) |
| 60 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +5.0 |
| 60 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.5 (witness) |
| 60 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 60 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 60 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 60 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 60 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 60 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 60 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 60 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.07); the sender bears the refusal: anxiety Ana +9.3 |
| 60 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 61; anxiety Ana +1.4, Marta +6.5, Nadia +4.2 (witness), Pia +4.0 (witness), Ravi +4.5 (witness); calmer sender takes some anxiety: Ana +9.2, Marta -9.2; Ana rejects the other on Ana–Marta (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -11.3 |
| 60 | Bruno | CUTOFF | Ana | — | arrives week 61; anxiety Ana +4.7, Bruno +1.7; calmer sender takes some anxiety: Ana -4.2, Bruno +4.2; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open); impingement removed at once, anxiety Bruno -9.9 |
| 60 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 61; anxiety Marta +2.1, Nadia +10.9, Pia +4.6 (witness), Ravi +10.5 (witness); Marta rejects the other on Marta–Nadia (rejection +0.00; the tie stays open) |
| 60 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Marta +10.4; anxiety Nadia +9.6 (witness), Pia +9.2 (witness) |
| 60 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 61; anxiety Ana +11.3, Marta +1.3, Nadia +4.9 (witness), Pia +4.7 (witness), Ravi +5.3 (witness); Marta rejects the other on Ana–Marta (rejection +0.10; the tie stays open); impingement removed at once, anxiety Marta -11.4 |
| 60 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 61; anxiety Marta +2.1, Nadia +4.8 (witness), Pia +3.3, Ravi +10.3 (witness); Marta rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 60 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.1; anxiety Pia +4.3 (witness), Ravi +9.6 (witness) |
| 60 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 61; anxiety Marta +9.4 (witness), Nadia +1.8, Pia +4.2 (witness), Ravi +6.4; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 60 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +8.8; anxiety Marta +9.4 (witness), Nadia +4.5 (witness) |
| 60 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +8.8; anxiety Nadia +4.4 (witness), Ravi +9.5 (witness) |
| 60 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Nadia +8.8 (witness), Pia +8.4 (witness) |
| 60 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +5.9; anxiety Marta +4.7 (witness), Nadia +4.5 (witness), Pia +4.3 (witness) |
| 60 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.3 (witness), Pia +4.2 (witness) |
| 60 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.0; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness) |
| 60 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -85.4, Bruno +49.5, Marta +36.0) |
| 60 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -41.3, Ana +41.3) |
| 60 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -257.2, Ana +51.4, Nadia +64.3, Pia +64.3, Ravi +77.1) |
| 60 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -171.8, Marta +85.9, Ravi +85.9) |
| 60 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -187.7, Marta +93.9, Ravi +93.9) |
| 60 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -360.1, Marta +108.0, Nadia +90.0, Pia +90.0, Sofia +72.0) |
| 60 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 60 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -55.8, Ravi +55.8) |
| 61 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 61 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 61 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 61 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 61 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 61 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 61 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.2 |
| 61 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +7.0; anxiety Nadia +4.1 (witness), Pia +4.0 (witness), Ravi +4.5 (witness) |
| 61 | Bruno | CUTOFF | Ana | — | arrives week 62; anxiety Ana +4.6, Bruno +1.6; calmer sender takes some anxiety: Ana -3.5, Bruno +3.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 61 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +10.4; anxiety Pia +4.6 (witness), Ravi +10.5 (witness) |
| 61 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.4; anxiety Nadia +9.6 (witness), Pia +9.3 (witness) |
| 61 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 62; anxiety Ana +10.8, Marta +1.3, Nadia +4.8 (witness), Pia +4.7 (witness), Ravi +5.3 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 61 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 62; anxiety Dr Halim +2.0, Marta +0.6, Nadia +4.8 (witness), Pia +4.6 (witness), Ravi +6.7 (witness) |
| 61 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.2; anxiety Pia +4.3 (witness), Ravi +9.7 (witness) |
| 61 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 62; anxiety Marta +9.4 (witness), Nadia +1.9, Pia +4.3 (witness), Ravi +3.8; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 61 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +8.8; anxiety Nadia +4.4 (witness), Ravi +9.5 (witness) |
| 61 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +8.8; anxiety Marta +9.4 (witness), Nadia +4.5 (witness) |
| 61 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.15); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.4 (witness), Nadia +4.5 (witness) |
| 61 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Nadia +8.8 (witness), Pia +8.4 (witness) |
| 61 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.3 (witness), Pia +4.2 (witness) |
| 61 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.1; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.2 (witness) |
| 61 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -78.4, Bruno +45.4, Marta +33.0) |
| 61 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -22.5, Ana +22.5) |
| 61 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -255.0, Ana +51.0, Nadia +63.7, Pia +63.7, Ravi +76.5) |
| 61 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -179.2, Marta +89.6, Ravi +89.6) |
| 61 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -184.2, Marta +92.1, Ravi +92.1) |
| 61 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -361.6, Marta +108.5, Nadia +90.4, Pia +90.4, Sofia +72.3) |
| 61 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 61 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -63.9, Ravi +63.9) |
| 62 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 62 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 62 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 62 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 62 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 62 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 62 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.1 |
| 62 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 63; anxiety Ana +1.4, Marta +6.5, Nadia +4.1 (witness), Pia +3.9 (witness), Ravi +4.4 (witness); calmer sender takes some anxiety: Ana +9.6, Marta -9.6; Ana rejects the other on Ana–Marta (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ana -15.1 |
| 62 | Bruno | CUTOFF | Ana | — | arrives week 63; anxiety Ana +4.7, Bruno +1.8; calmer sender takes some anxiety: Ana -4.0, Bruno +4.0; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 62 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 63; anxiety Ana +7.7, Marta +1.3, Nadia +4.8 (witness), Pia +4.7 (witness), Ravi +5.3 (witness); Marta rejects the other on Ana–Marta (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -6.5 |
| 62 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.4; anxiety Pia +4.7 (witness), Ravi +10.7 (witness) |
| 62 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.5; anxiety Nadia +9.7 (witness), Pia +9.3 (witness) |
| 62 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.16); the sender bears the refusal: anxiety Nadia +9.2; anxiety Marta +9.4 (witness), Pia +4.3 (witness) |
| 62 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.2; anxiety Pia +4.3 (witness), Ravi +9.7 (witness) |
| 62 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +8.8; anxiety Nadia +4.4 (witness), Ravi +9.5 (witness) |
| 62 | Pia | TRIANGLE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +8.8; anxiety Marta +9.4 (witness), Nadia +4.5 (witness) |
| 62 | Ravi | CONFLICT | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Marta +9.4 (witness), Pia +4.3 (witness) |
| 62 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Nadia +8.8 (witness), Pia +8.5 (witness) |
| 62 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +6.0; anxiety Marta +4.7 (witness), Nadia +4.5 (witness), Pia +4.3 (witness) |
| 62 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Marta +9.4 (witness), Nadia +4.5 (witness) |
| 62 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 64; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness), Ravi +2.5, Sofia +0.8; calmer sender takes some anxiety: Ravi -4.4, Sofia +4.4; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 62 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -69.0, Bruno +39.9, Marta +29.0) |
| 62 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -37.8, Ana +37.8) |
| 62 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -274.9, Ana +55.0, Nadia +68.7, Pia +68.7, Ravi +82.5) |
| 62 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -175.0, Marta +87.5, Ravi +87.5) |
| 62 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -187.7, Marta +93.8, Ravi +93.8) |
| 62 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -372.3, Marta +111.7, Nadia +93.1, Pia +93.1, Sofia +74.5) |
| 62 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 62 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -65.1, Ravi +65.1) |
| 63 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 63 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 63 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 63 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 63 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 63 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 63 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 63 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 63 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.2 |
| 63 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 64; anxiety Ana +1.4, Marta +6.6, Nadia +4.1 (witness), Pia +3.9 (witness), Ravi +4.4 (witness); calmer sender takes some anxiety: Ana +9.8, Marta -9.8; Ana rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 63 | Bruno | STAY-IN-CONTACT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +5.5 |
| 63 | Marta | CONFLICT | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.5; anxiety Nadia +9.8 (witness), Pia +9.4 (witness) |
| 63 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.4; anxiety Pia +4.8 (witness), Ravi +10.7 (witness) |
| 63 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +6.3; anxiety Nadia +4.8 (witness), Pia +4.7 (witness), Ravi +5.3 (witness) |
| 63 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.25); the sender bears the refusal: anxiety Marta +10.5; anxiety Nadia +4.8 (witness), Ravi +10.4 (witness) |
| 63 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.2; anxiety Marta +9.5 (witness), Pia +4.3 (witness) |
| 63 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.2; anxiety Pia +4.4 (witness), Ravi +9.8 (witness) |
| 63 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 64; anxiety Marta +7.8, Nadia +4.4 (witness), Pia +1.8, Ravi +9.5 (witness); calmer sender takes some anxiety: Marta -15.0, Pia +15.0; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open); impingement removed at once, anxiety Pia -3.2 |
| 63 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +8.8; anxiety Marta +9.4 (witness), Nadia +4.4 (witness) |
| 63 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.3 (witness), Nadia +4.4 (witness) |
| 63 | Ravi | CONFLICT | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.3 (witness), Pia +4.3 (witness) |
| 63 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | refused by Dr Halim (rejection +0.24); the sender bears the refusal: anxiety Ravi +3.0; anxiety Marta +6.1 (witness), Nadia +4.4 (witness), Pia +4.2 (witness) |
| 63 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +4.1; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 63 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -69.6, Bruno +40.3, Marta +29.3) |
| 63 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -43.0, Ana +43.0) |
| 63 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -262.1, Ana +52.4, Nadia +65.5, Pia +65.5, Ravi +78.6) |
| 63 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -171.7, Marta +85.9, Ravi +85.9) |
| 63 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -177.7, Marta +88.9, Ravi +88.9) |
| 63 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -346.6, Marta +104.0, Nadia +86.7, Pia +86.7, Sofia +69.3) |
| 63 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 63 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -61.1, Ravi +61.1) |
| 64 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +5.0 |
| 64 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.5 (witness) |
| 64 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 64 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 64 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 64 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 64 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 64 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 64 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.3 |
| 64 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 65; anxiety Ana +1.4, Marta +6.9, Nadia +4.1 (witness), Pia +3.9 (witness), Ravi +4.4 (witness); calmer sender takes some anxiety: Ana +10.3, Marta -10.3; Ana rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 64 | Bruno | CUTOFF | Ana | — | arrives week 65; anxiety Ana +3.7, Bruno +1.7; calmer sender takes some anxiety: Ana -4.5, Bruno +4.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 64 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 65; anxiety Dr Halim +0.1, Marta +2.5 (witness), Nadia +1.8 (witness), Pia +1.7 (witness); calmer sender takes some anxiety: Dr Halim +2.9, Ravi -2.9; the contact lands with Ravi: systems perspective +0.10 |
| 64 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Marta +10.5; anxiety Nadia +9.8 (witness), Pia +9.3 (witness) |
| 64 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +6.3; anxiety Nadia +4.9 (witness), Pia +4.6 (witness), Ravi +5.2 (witness) |
| 64 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.5; anxiety Nadia +4.9 (witness), Ravi +10.5 (witness) |
| 64 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.5; anxiety Pia +4.7 (witness), Ravi +10.6 (witness) |
| 64 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 65; anxiety Marta +9.5 (witness), Nadia +1.9, Pia +4.3 (witness), Ravi +8.1; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 64 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.2; anxiety Pia +4.3 (witness), Ravi +9.7 (witness) |
| 64 | Pia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 64 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 65; anxiety Marta +9.4 (witness), Nadia +4.4 (witness), Pia +1.8, Ravi +7.9; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 64 | Ravi | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 64 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.4 (witness), Pia +4.3 (witness) |
| 64 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +6.0; anxiety Marta +4.7 (witness), Nadia +4.5 (witness), Pia +4.3 (witness) |
| 64 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.1; anxiety Marta +3.4 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 64 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -78.1, Bruno +45.2, Marta +32.9) |
| 64 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -41.7, Ana +41.7) |
| 64 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -285.5, Ana +57.1, Nadia +71.4, Pia +71.4, Ravi +85.6) |
| 64 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -181.4, Marta +90.7, Ravi +90.7) |
| 64 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -187.2, Marta +93.6, Ravi +93.6) |
| 64 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -374.4, Marta +112.3, Nadia +93.6, Pia +93.6, Sofia +74.9) |
| 64 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 64 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -64.0, Ravi +64.0) |
| 65 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 65 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 65 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 65 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 65 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 65 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 65 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 65 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.3 |
| 65 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 66; anxiety Ana +1.4, Marta +3.6, Nadia +4.1 (witness), Pia +3.9 (witness), Ravi +4.5 (witness); calmer sender takes some anxiety: Ana +8.4, Marta -8.4; Ana rejects the other on Ana–Marta (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -10.0 |
| 65 | Bruno | CUTOFF | Ana | — | arrives week 66; anxiety Ana +3.3, Bruno +1.8; calmer sender takes some anxiety: Ana -4.6, Bruno +4.6; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 65 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 66; anxiety Ana +10.0, Marta +1.3, Nadia +4.9 (witness), Pia +4.6 (witness), Ravi +5.3 (witness); Marta rejects the other on Ana–Marta (rejection +0.15; the tie stays open); impingement removed at once, anxiety Marta -11.4 |
| 65 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 66; anxiety Marta +2.1, Nadia +9.6, Pia +4.7 (witness), Ravi +10.7 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -7.6 |
| 65 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 66; anxiety Marta +2.1, Nadia +4.9 (witness), Pia +10.4, Ravi +10.6 (witness); Marta rejects the other on Marta–Pia (rejection +0.00; the tie stays open); impingement removed at once, anxiety Marta -3.5 |
| 65 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 66; anxiety Marta +2.1, Nadia +9.8 (witness), Pia +9.3 (witness), Ravi +11.0; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 65 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 66; anxiety Marta +9.4 (witness), Nadia +1.8, Pia +4.2 (witness), Ravi +3.7; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 65 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 66; anxiety Marta +7.6, Nadia +1.8, Pia +4.2 (witness), Ravi +9.6 (witness); calmer sender takes some anxiety: Marta -11.9, Nadia +11.9; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -10.9 |
| 65 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Pia +8.6; anxiety Nadia +4.4 (witness), Ravi +9.5 (witness) |
| 65 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +8.6; anxiety Marta +9.2 (witness), Nadia +4.4 (witness) |
| 65 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 67; anxiety Marta +4.7 (witness), Nadia +4.4 (witness), Pia +4.2 (witness), Ravi +1.2, Sofia +5.8; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -0.5 |
| 65 | Ravi | CONFLICT | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.3 (witness), Pia +4.2 (witness) |
| 65 | Ravi | CONFLICT | Marta | Nadia, Pia | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Ravi +9.9; anxiety Nadia +8.8 (witness), Pia +8.4 (witness) |
| 65 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 66; anxiety Marta +9.3 (witness), Nadia +4.4 (witness), Pia +9.5, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ravi -5.5 |
| 65 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 65 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -81.8, Bruno +47.4, Marta +34.5) |
| 65 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -45.3, Ana +45.3) |
| 65 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -289.6, Ana +57.9, Nadia +72.4, Pia +72.4, Ravi +86.9) |
| 65 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -163.3, Marta +81.7, Ravi +81.7) |
| 65 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -164.8, Marta +82.4, Ravi +82.4) |
| 65 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -345.7, Marta +103.7, Nadia +86.4, Pia +86.4, Sofia +69.1) |
| 65 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 65 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -61.5, Ravi +61.5) |
| 66 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 66 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 66 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 66 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 66 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 66 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 66 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 66 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.3 |
| 66 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +7.1; anxiety Nadia +4.1 (witness), Pia +3.9 (witness), Ravi +4.5 (witness) |
| 66 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.16); the sender bears the refusal: anxiety Bruno +8.9 |
| 66 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 67; anxiety Ana +8.8, Marta +1.3, Nadia +4.8 (witness), Pia +4.6 (witness), Ravi +5.2 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 66 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 67; anxiety Marta +2.1, Nadia +4.8 (witness), Pia +3.3, Ravi +10.4 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -8.4 |
| 66 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 67; anxiety Marta +2.1, Nadia +10.8, Pia +4.6 (witness), Ravi +10.5 (witness); Marta rejects the other on Marta–Nadia (rejection +0.00; the tie stays open) |
| 66 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 67; anxiety Marta +2.1, Nadia +9.6 (witness), Pia +9.2 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 66 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 67; anxiety Marta +9.5 (witness), Nadia +1.9, Pia +4.3 (witness), Ravi +3.7; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 66 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.2; anxiety Pia +4.3 (witness), Ravi +9.7 (witness) |
| 66 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 67; anxiety Marta +8.4, Nadia +4.5 (witness), Pia +1.8, Ravi +9.7 (witness); calmer sender takes some anxiety: Marta -13.2, Pia +13.2; Pia rejects the other on Marta–Pia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Pia -10.4 |
| 66 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +8.9; anxiety Marta +9.5 (witness), Nadia +4.5 (witness) |
| 66 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Nadia +8.9 (witness), Pia +8.5 (witness) |
| 66 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Marta +9.4 (witness), Pia +4.3 (witness) |
| 66 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +10.0; anxiety Marta +9.4 (witness), Nadia +4.5 (witness) |
| 66 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.0; anxiety Marta +3.4 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 66 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -81.2, Bruno +47.0, Marta +34.2) |
| 66 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -47.6, Ana +47.6) |
| 66 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -212.6, Ana +42.5, Nadia +53.1, Pia +53.1, Ravi +63.8) |
| 66 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -157.9, Marta +78.9, Ravi +78.9) |
| 66 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -181.7, Marta +90.9, Ravi +90.9) |
| 66 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -340.6, Marta +102.2, Nadia +85.2, Pia +85.2, Sofia +68.1) |
| 66 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 66 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -57.3, Ravi +57.3) |
| 67 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 67 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 67 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 67 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 67 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 67 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 67 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.3 |
| 67 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +7.0; anxiety Nadia +4.1 (witness), Pia +3.9 (witness), Ravi +4.5 (witness) |
| 67 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.14); the sender bears the refusal: anxiety Bruno +9.0 |
| 67 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.4; anxiety Nadia +4.8 (witness), Ravi +10.5 (witness) |
| 67 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 68; anxiety Ana +9.1, Marta +1.3, Nadia +4.8 (witness), Pia +4.6 (witness), Ravi +5.2 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 67 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 68; anxiety Marta +2.1, Nadia +3.5, Pia +4.6 (witness), Ravi +10.5 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -7.1 |
| 67 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 68; anxiety Marta +2.1, Nadia +9.6 (witness), Pia +9.2 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open) |
| 67 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 68; anxiety Marta +9.5 (witness), Nadia +1.9, Pia +4.3 (witness), Ravi +3.7; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 67 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 68; anxiety Marta +7.1, Nadia +1.8, Pia +4.2 (witness), Ravi +9.7 (witness); calmer sender takes some anxiety: Marta -12.5, Nadia +12.5; Nadia rejects the other on Marta–Nadia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Nadia -10.9 |
| 67 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 68; anxiety Marta +7.4, Nadia +4.4 (witness), Pia +1.7, Ravi +9.6 (witness); calmer sender takes some anxiety: Marta -14.0, Pia +14.0; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 67 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 68; anxiety Marta +9.3 (witness), Nadia +4.4 (witness), Pia +1.7, Ravi +7.7; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Pia -4.4 |
| 67 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Marta +9.4 (witness), Pia +4.2 (witness) |
| 67 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Marta +9.4 (witness), Nadia +4.4 (witness) |
| 67 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 69; anxiety Marta +4.7 (witness), Nadia +4.5 (witness), Pia +4.3 (witness), Ravi +1.2, Sofia +6.8; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 67 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.1; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 67 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -77.8, Bruno +45.1, Marta +32.8) |
| 67 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -47.3, Ana +47.3) |
| 67 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -228.7, Ana +45.7, Nadia +57.2, Pia +57.2, Ravi +68.6) |
| 67 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -164.7, Marta +82.4, Ravi +82.4) |
| 67 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -160.9, Marta +80.5, Ravi +80.5) |
| 67 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -335.7, Marta +100.7, Nadia +83.9, Pia +83.9, Sofia +67.1) |
| 67 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 67 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -58.7, Ravi +58.7) |
| 68 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +5.0 |
| 68 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.5 (witness) |
| 68 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 68 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 68 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 68 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 68 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 68 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 68 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.4 |
| 68 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 69; anxiety Ana +1.4, Marta +5.7, Nadia +4.1 (witness), Pia +3.9 (witness), Ravi +4.5 (witness); calmer sender takes some anxiety: Ana +8.5, Marta -8.5; Ana rejects the other on Ana–Marta (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ana -15.1 |
| 68 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.9 |
| 68 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 69; anxiety Marta +2.1, Nadia +9.6 (witness), Pia +9.2 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 68 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 69; anxiety Marta +2.1, Nadia +9.8, Pia +4.6 (witness), Ravi +10.5 (witness); Marta rejects the other on Marta–Nadia (rejection +0.00; the tie stays open) |
| 68 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 69; anxiety Ana +4.7, Marta +1.3, Nadia +4.8 (witness), Pia +4.6 (witness), Ravi +5.2 (witness); Marta rejects the other on Ana–Marta (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -5.7 |
| 68 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 69; anxiety Marta +2.1, Nadia +4.8 (witness), Pia +9.4, Ravi +10.5 (witness); Marta rejects the other on Marta–Pia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Marta -11.4 |
| 68 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 69; anxiety Marta +9.4 (witness), Nadia +1.8, Pia +4.2 (witness), Ravi +3.7; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 68 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +9.1; anxiety Pia +4.2 (witness), Ravi +9.6 (witness) |
| 68 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 69; anxiety Marta +6.4, Nadia +4.4 (witness), Pia +1.7, Ravi +9.5 (witness); calmer sender takes some anxiety: Marta -13.2, Pia +13.2; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -9.4 |
| 68 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 69; anxiety Marta +9.3 (witness), Nadia +4.4 (witness), Pia +1.7, Ravi +6.7; calmer sender takes some anxiety: Pia +1.3, Ravi -1.3; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 68 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.1; anxiety Nadia +8.9 (witness), Pia +8.5 (witness) |
| 68 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.1; anxiety Marta +9.5 (witness), Pia +4.3 (witness) |
| 68 | Ravi | CONFLICT | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.1; anxiety Marta +9.5 (witness), Nadia +4.5 (witness) |
| 68 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 70; anxiety Marta +3.4 (witness), Nadia +3.3 (witness), Pia +3.1 (witness), Ravi +3.0, Sofia +0.8; calmer sender takes some anxiety: Ravi -4.6, Sofia +4.6; Sofia rejects the other on Ravi–Sofia (rejection +0.20; the tie stays open); impingement removed at once, anxiety Sofia -6.3 |
| 68 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -80.1, Bruno +46.4, Marta +33.7) |
| 68 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -47.0, Ana +47.0) |
| 68 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -230.4, Ana +46.1, Nadia +57.6, Pia +57.6, Ravi +69.1) |
| 68 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -145.3, Marta +72.6, Ravi +72.6) |
| 68 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -152.1, Marta +76.0, Ravi +76.0) |
| 68 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -331.0, Marta +99.3, Nadia +82.8, Pia +82.8, Sofia +66.2) |
| 68 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 68 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -55.4, Ravi +55.4) |
| 69 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 69 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 69 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 69 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 69 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 69 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 69 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.2 |
| 69 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +7.0; anxiety Nadia +4.0 (witness), Pia +3.9 (witness), Ravi +4.4 (witness) |
| 69 | Bruno | CUTOFF | Ana | — | arrives week 70; anxiety Ana +3.7, Bruno +1.8; calmer sender takes some anxiety: Ana -4.5, Bruno +4.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 69 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 70; anxiety Marta +2.1, Nadia +4.8 (witness), Pia +8.3, Ravi +10.3 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -5.9 |
| 69 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 70; anxiety Ana +7.1, Marta +1.3, Nadia +4.8 (witness), Pia +4.5 (witness), Ravi +5.1 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 69 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 70; anxiety Marta +2.1, Nadia +4.3, Pia +4.6 (witness), Ravi +10.3 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -6.9 |
| 69 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 70; anxiety Marta +6.9, Nadia +1.9, Pia +4.3 (witness), Ravi +9.6 (witness); calmer sender takes some anxiety: Marta -11.9, Nadia +11.9; Nadia rejects the other on Marta–Nadia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Nadia -10.9 |
| 69 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 70; anxiety Marta +9.5 (witness), Nadia +1.9, Pia +4.3 (witness), Ravi +3.7; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 69 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 70; anxiety Marta +5.9, Nadia +4.4 (witness), Pia +1.8, Ravi +9.5 (witness); calmer sender takes some anxiety: Marta -13.4, Pia +13.4; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -8.3 |
| 69 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 70; anxiety Marta +9.4 (witness), Nadia +4.4 (witness), Pia +1.8, Ravi +3.7; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 69 | Ravi | WITHHOLD | Pia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 69 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Marta +9.5 (witness), Pia +4.3 (witness) |
| 69 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Nadia +8.9 (witness), Pia +8.5 (witness) |
| 69 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +4.2; anxiety Marta +3.6 (witness), Nadia +3.4 (witness), Pia +3.2 (witness) |
| 69 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -65.0, Bruno +37.6, Marta +27.3) |
| 69 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -40.1, Ana +40.1) |
| 69 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -199.8, Ana +40.0, Nadia +50.0, Pia +50.0, Ravi +59.9) |
| 69 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -160.1, Marta +80.1, Ravi +80.1) |
| 69 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -158.0, Marta +79.0, Ravi +79.0) |
| 69 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -336.0, Marta +100.8, Nadia +84.0, Pia +84.0, Sofia +67.2) |
| 69 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 69 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -68.6, Ravi +68.6) |
| 70 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 70 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 70 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 70 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 70 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 70 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 70 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.2 |
| 70 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +7.0; anxiety Nadia +4.0 (witness), Pia +3.9 (witness), Ravi +4.4 (witness) |
| 70 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +8.7 |
| 70 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 71; anxiety Marta +2.1, Nadia +9.5 (witness), Pia +9.1 (witness), Ravi +6.4; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 70 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 71; anxiety Marta +2.1, Nadia +8.9, Pia +4.5 (witness), Ravi +10.4 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -6.0 |
| 70 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.4; anxiety Nadia +4.7 (witness), Ravi +10.4 (witness) |
| 70 | Nadia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 70 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 71; anxiety Marta +6.0, Nadia +1.8, Pia +4.2 (witness), Ravi +9.5 (witness); calmer sender takes some anxiety: Marta -11.4, Nadia +11.4; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -8.9 |
| 70 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.20); the sender bears the refusal: anxiety Pia +8.7; anxiety Marta +9.3 (witness), Nadia +4.4 (witness) |
| 70 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 71; anxiety Marta +6.6, Nadia +4.3 (witness), Pia +1.7, Ravi +9.5 (witness); calmer sender takes some anxiety: Marta -12.5, Pia +12.5; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 70 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Nadia +8.8 (witness), Pia +8.4 (witness) |
| 70 | Ravi | CONFLICT | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Marta +9.5 (witness), Pia +4.3 (witness) |
| 70 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 71; anxiety Marta +9.4 (witness), Nadia +4.4 (witness), Pia +9.4, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ravi -8.1 |
| 70 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +3.9; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness) |
| 70 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -66.3, Bruno +38.4, Marta +27.9) |
| 70 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -33.6, Ana +33.6) |
| 70 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -187.7, Ana +37.5, Nadia +46.9, Pia +46.9, Ravi +56.3) |
| 70 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -125.3, Marta +62.7, Ravi +62.7) |
| 70 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -139.7, Marta +69.9, Ravi +69.9) |
| 70 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -291.4, Marta +87.4, Nadia +72.8, Pia +72.8, Sofia +58.3) |
| 70 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 70 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -42.0, Ravi +42.0) |
| 71 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 71 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 71 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 71 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 71 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 71 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 71 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.1 |
| 71 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 72; anxiety Ana +1.4, Marta +3.9, Nadia +4.0 (witness), Pia +3.8 (witness), Ravi +4.3 (witness); calmer sender takes some anxiety: Ana +8.1, Marta -8.1; Ana rejects the other on Ana–Marta (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ana -10.2 |
| 71 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.0 |
| 71 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 72; anxiety Marta +2.1, Nadia +8.0, Pia +4.5 (witness), Ravi +10.3 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -5.4 |
| 71 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 72; anxiety Ana +7.0, Marta +1.2, Nadia +4.7 (witness), Pia +4.5 (witness), Ravi +5.1 (witness); Marta rejects the other on Ana–Marta (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -3.9 |
| 71 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 72; anxiety Marta +2.1, Nadia +9.5 (witness), Pia +9.0 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 71 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 72; anxiety Marta +2.1, Nadia +4.7 (witness), Pia +7.6, Ravi +10.3 (witness); Marta rejects the other on Marta–Pia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Marta -10.9 |
| 71 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 72; anxiety Marta +5.4, Nadia +1.8, Pia +4.2 (witness), Ravi +9.5 (witness); calmer sender takes some anxiety: Marta -10.8, Nadia +10.8; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -8.0 |
| 71 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 72; anxiety Marta +9.4 (witness), Nadia +1.8, Pia +4.2 (witness), Ravi +6.3; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 71 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 72; anxiety Marta +6.5, Nadia +4.4 (witness), Pia +1.7, Ravi +9.5 (witness); calmer sender takes some anxiety: Marta -12.3, Pia +12.3; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -7.6 |
| 71 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 72; anxiety Marta +9.3 (witness), Nadia +4.4 (witness), Pia +1.7, Ravi +6.2; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Pia rejects the other on Pia–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Pia -6.4 |
| 71 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Nadia +8.6 (witness), Pia +8.2 (witness) |
| 71 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.2 (witness), Pia +4.2 (witness) |
| 71 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 73; anxiety Marta +4.6 (witness), Nadia +4.4 (witness), Pia +4.2 (witness), Ravi +1.2, Sofia +4.4; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ravi -0.6 |
| 71 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.0; anxiety Marta +3.4 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 71 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -52.9, Bruno +30.6, Marta +22.3) |
| 71 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -33.4, Ana +33.4) |
| 71 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -166.9, Ana +33.4, Nadia +41.7, Pia +41.7, Ravi +50.1) |
| 71 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -124.3, Marta +62.2, Ravi +62.2) |
| 71 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -139.5, Marta +69.7, Ravi +69.7) |
| 71 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -249.3, Marta +74.8, Nadia +62.3, Pia +62.3, Sofia +49.9) |
| 71 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 71 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -38.7, Ravi +38.7) |
| 72 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +5.0 |
| 72 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.5 (witness) |
| 72 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 72 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 72 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 72 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 72 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 72 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 72 | Ana | WITHHOLD | Bruno | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 72 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.8; anxiety Nadia +4.0 (witness), Pia +3.8 (witness), Ravi +4.3 (witness) |
| 72 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.0 |
| 72 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 73; anxiety Dr Halim +0.1, Nadia +1.8 (witness), Pia +1.7 (witness), Ravi +2.5 (witness); calmer sender takes some anxiety: Dr Halim +6.3, Marta -6.3 |
| 72 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 73; anxiety Marta +2.1, Nadia +9.4 (witness), Pia +9.0 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 72 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 73; anxiety Marta +2.1, Nadia +4.7 (witness), Pia +6.7, Ravi +10.2 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -5.2 |
| 72 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 73; anxiety Ana +5.7, Marta +1.2, Nadia +4.7 (witness), Pia +4.5 (witness), Ravi +5.1 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 72 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 73; anxiety Marta +2.0, Nadia +7.1, Pia +4.5 (witness), Ravi +10.2 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -4.8 |
| 72 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 73; anxiety Marta +9.3 (witness), Nadia +1.8, Pia +4.2 (witness), Ravi +3.8; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 72 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 73; anxiety Marta +4.8, Nadia +1.8, Pia +4.2 (witness), Ravi +9.5 (witness); calmer sender takes some anxiety: Marta -10.4, Nadia +10.4; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -7.1 |
| 72 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 73; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +6.0; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -7.5 |
| 72 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 73; anxiety Marta +5.2, Nadia +4.3 (witness), Pia +1.7, Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -11.8, Pia +11.8; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -6.7 |
| 72 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Marta +9.4 (witness), Pia +4.2 (witness) |
| 72 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +10.0; anxiety Nadia +8.8 (witness), Pia +8.4 (witness) |
| 72 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 73; anxiety Marta +9.4 (witness), Nadia +4.4 (witness), Pia +7.5, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ravi -10.2 |
| 72 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 72 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -48.4, Bruno +28.0, Marta +20.4) |
| 72 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -30.0, Ana +30.0) |
| 72 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -135.6, Ana +27.1, Nadia +33.9, Pia +33.9, Ravi +40.7) |
| 72 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -114.0, Marta +57.0, Ravi +57.0) |
| 72 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -113.5, Marta +56.7, Ravi +56.7) |
| 72 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -251.4, Marta +75.4, Nadia +62.8, Pia +62.8, Sofia +50.3) |
| 72 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 72 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -37.7, Ravi +37.7) |
| 73 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 73 | — | (system) | — | — | Dr Halim–Marta tie now ordinary |
| 73 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 73 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 73 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 73 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 73 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 73 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +8.9 |
| 73 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.7; anxiety Nadia +3.9 (witness), Pia +3.7 (witness), Ravi +4.3 (witness) |
| 73 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.9 |
| 73 | Marta | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 73 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 74; anxiety Marta +2.0, Nadia +7.5, Pia +4.5 (witness), Ravi +10.2 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -4.2 |
| 73 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.2; anxiety Nadia +4.7 (witness), Ravi +10.3 (witness) |
| 73 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 74; anxiety Marta +4.2, Nadia +1.8, Pia +4.2 (witness), Ravi +9.5 (witness); calmer sender takes some anxiety: Marta -11.1, Nadia +11.1; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -7.5 |
| 73 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 74; anxiety Marta +9.3 (witness), Nadia +1.9, Pia +4.2 (witness), Ravi +3.7; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -7.0 |
| 73 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 74; anxiety Marta +4.6, Nadia +4.4 (witness), Pia +1.7, Ravi +9.5 (witness); calmer sender takes some anxiety: Marta -12.5, Pia +12.5; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 73 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 74; anxiety Marta +9.2 (witness), Nadia +4.4 (witness), Pia +1.7, Ravi +4.9; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -6.7 |
| 73 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 74; anxiety Marta +9.2 (witness), Nadia +4.4 (witness), Pia +6.7, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -4.9 |
| 73 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Nadia +8.7 (witness), Pia +8.3 (witness) |
| 73 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 74; anxiety Marta +9.2 (witness), Nadia +7.0, Pia +4.2 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -11.9 |
| 73 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +4.2; anxiety Marta +3.5 (witness), Nadia +3.4 (witness), Pia +3.3 (witness) |
| 73 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -39.2, Bruno +22.7, Marta +16.5) |
| 73 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -24.6, Ana +24.6) |
| 73 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -138.1, Ana +27.6, Nadia +34.5, Pia +34.5, Ravi +41.4) |
| 73 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -122.7, Marta +61.3, Ravi +61.3) |
| 73 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -124.3, Marta +62.2, Ravi +62.2) |
| 73 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -249.8, Marta +74.9, Nadia +62.4, Pia +62.4, Sofia +50.0) |
| 73 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 73 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -45.6, Ravi +45.6) |
| 74 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 74 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 74 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 74 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 74 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 74 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.7; anxiety Nadia +3.9 (witness), Pia +3.7 (witness), Ravi +4.2 (witness) |
| 74 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.9 |
| 74 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.9 |
| 74 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.2; anxiety Nadia +4.7 (witness), Ravi +10.3 (witness) |
| 74 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 75; anxiety Marta +2.0, Nadia +9.4 (witness), Pia +9.0 (witness), Ravi +6.4; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 74 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 75; anxiety Ana +6.9, Marta +1.2, Nadia +4.7 (witness), Pia +4.5 (witness), Ravi +5.1 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 74 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 75; anxiety Marta +3.7, Nadia +1.8, Pia +4.1 (witness), Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -10.1, Nadia +10.1; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 74 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 75; anxiety Marta +9.1 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +3.9; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 74 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 75; anxiety Marta +9.0 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +4.2; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -5.8 |
| 74 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 75; anxiety Marta +4.6, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -11.5, Pia +11.5; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 74 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Nadia +8.6 (witness), Pia +8.1 (witness) |
| 74 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 75; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +5.8, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -4.3 |
| 74 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.1 (witness), Pia +4.1 (witness) |
| 74 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 76; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness), Ravi +3.3, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.5, Sofia +3.5; Sofia rejects the other on Ravi–Sofia (rejection +0.30; the tie stays open); impingement removed at once, anxiety Sofia -1.1 |
| 74 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -36.9, Bruno +21.3, Marta +15.5) |
| 74 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -22.0, Ana +22.0) |
| 74 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -143.1, Ana +28.6, Nadia +35.8, Pia +35.8, Ravi +42.9) |
| 74 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -98.7, Marta +49.3, Ravi +49.3) |
| 74 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -100.6, Marta +50.3, Ravi +50.3) |
| 74 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -201.1, Marta +60.3, Nadia +50.3, Pia +50.3, Sofia +40.2) |
| 74 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 74 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -29.6, Ravi +29.6) |
| 75 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 75 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 75 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 75 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 75 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 75 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 75 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.8; anxiety Nadia +3.9 (witness), Pia +3.8 (witness), Ravi +4.3 (witness) |
| 75 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.0 |
| 75 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.9 |
| 75 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 76; anxiety Marta +2.1, Nadia +4.7 (witness), Pia +6.6, Ravi +10.2 (witness); Marta rejects the other on Marta–Pia (rejection +0.15; the tie stays open); impingement removed at once, anxiety Marta -9.4 |
| 75 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 76; anxiety Marta +2.1, Nadia +9.3 (witness), Pia +8.9 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 75 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 76; anxiety Ana +6.2, Marta +1.2, Nadia +4.6 (witness), Pia +4.5 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 75 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 75 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 76; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +4.1; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -5.6 |
| 75 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 76; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +3.7; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -3.0 |
| 75 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 76; anxiety Marta +4.2, Nadia +4.3 (witness), Pia +1.7, Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -11.0, Pia +11.0; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -6.6 |
| 75 | Ravi | REDUCE_CUTOFF | Pia | Marta, Nadia | arrives week 76; anxiety Marta +3.8 (witness), Nadia +1.8 (witness), Pia +3.0, Ravi +0.8; Pia's rejection -0.25; Ravi's rejection -0.25 |
| 75 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 76; anxiety Marta +9.2 (witness), Nadia +5.6, Pia +4.1 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ravi -6.7 |
| 75 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Nadia +8.6 (witness), Pia +8.2 (witness) |
| 75 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +4.0; anxiety Marta +3.4 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 75 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -42.2, Bruno +24.4, Marta +17.8) |
| 75 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -23.6, Ana +23.6) |
| 75 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -124.3, Ana +24.9, Nadia +31.1, Pia +31.1, Ravi +37.3) |
| 75 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -97.5, Marta +48.8, Ravi +48.8) |
| 75 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -96.9, Marta +48.5, Ravi +48.5) |
| 75 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -201.9, Marta +60.6, Nadia +50.5, Pia +50.5, Sofia +40.4) |
| 75 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 75 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -27.6, Ravi +27.6) |
| 76 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +5.0 |
| 76 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.5 (witness) |
| 76 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 76 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 76 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 76 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 76 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 76 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.8; anxiety Nadia +3.9 (witness), Pia +3.7 (witness), Ravi +4.3 (witness) |
| 76 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.9 |
| 76 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.0 |
| 76 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.2; anxiety Nadia +4.7 (witness), Ravi +10.2 (witness) |
| 76 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 77; anxiety Ana +6.4, Marta +1.2, Nadia +4.7 (witness), Pia +4.4 (witness), Ravi +5.1 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 76 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 77; anxiety Marta +2.0, Nadia +9.4 (witness), Pia +8.9 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 76 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 77; anxiety Marta +2.0, Nadia +7.3, Pia +4.4 (witness), Ravi +10.2 (witness); Marta rejects the other on Marta–Nadia (rejection +0.15; the tie stays open); impingement removed at once, anxiety Marta -5.0 |
| 76 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 77; anxiety Marta +3.5, Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -9.9, Nadia +9.9; Nadia rejects the other on Marta–Nadia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Nadia -7.3 |
| 76 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 77; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +3.1; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 76 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 77; anxiety Marta +4.6, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -10.9, Pia +10.9; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 76 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.24); the sender bears the refusal: anxiety Pia +8.6; anxiety Marta +9.1 (witness), Nadia +4.3 (witness) |
| 76 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 77; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +4.7, Ravi +1.9; Ravi rejects the other on Pia–Ravi (rejection +0.11; the tie stays open); impingement removed at once, anxiety Ravi -2.5 |
| 76 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.7; anxiety Nadia +8.5 (witness), Pia +8.1 (witness) |
| 76 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.7; anxiety Marta +9.1 (witness), Pia +4.1 (witness) |
| 76 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.1; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.2 (witness) |
| 76 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -39.8, Bruno +23.1, Marta +16.8) |
| 76 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -24.0, Ana +24.0) |
| 76 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -112.6, Ana +22.5, Nadia +28.1, Pia +28.1, Ravi +33.8) |
| 76 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -84.4, Marta +42.2, Ravi +42.2) |
| 76 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -92.9, Marta +46.4, Ravi +46.4) |
| 76 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -172.1, Marta +51.6, Nadia +43.0, Pia +43.0, Sofia +34.4) |
| 76 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 76 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -23.8, Ravi +23.8) |
| 77 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 77 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 77 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 77 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 77 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 77 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 77 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.8; anxiety Nadia +3.9 (witness), Pia +3.7 (witness), Ravi +4.2 (witness) |
| 77 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +9.0 |
| 77 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.0 |
| 77 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.2; anxiety Nadia +4.7 (witness), Ravi +10.2 (witness) |
| 77 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 78; anxiety Marta +2.0, Nadia +7.7, Pia +4.4 (witness), Ravi +10.1 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -2.6 |
| 77 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 78; anxiety Marta +2.0, Nadia +9.4 (witness), Pia +8.9 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 77 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 78; anxiety Ana +4.7, Marta +1.2, Nadia +4.7 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 77 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 78; anxiety Marta +2.6, Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -10.1, Nadia +10.1; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -7.7 |
| 77 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 78; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +3.3; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 77 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 78; anxiety Marta +4.7, Nadia +4.3 (witness), Pia +1.7, Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -11.5, Pia +11.5; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 77 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 78; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +4.3; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Pia rejects the other on Pia–Ravi (rejection +0.25; the tie stays open); impingement removed at once, anxiety Pia -3.1 |
| 77 | Ravi | DETRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +4.1; anxiety Marta +3.8 (witness), Nadia +1.8 (witness) |
| 77 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Nadia +8.6 (witness), Pia +8.2 (witness) |
| 77 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.2 (witness), Pia +4.1 (witness) |
| 77 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.1; anxiety Marta +3.5 (witness), Nadia +3.4 (witness), Pia +3.2 (witness) |
| 77 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -39.6, Bruno +22.9, Marta +16.7) |
| 77 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -23.8, Ana +23.8) |
| 77 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -110.9, Ana +22.2, Nadia +27.7, Pia +27.7, Ravi +33.3) |
| 77 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -92.2, Marta +46.1, Ravi +46.1) |
| 77 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -103.7, Marta +51.8, Ravi +51.8) |
| 77 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -197.8, Marta +59.3, Nadia +49.4, Pia +49.4, Sofia +39.6) |
| 77 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 77 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -25.8, Ravi +25.8) |
| 78 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 78 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 78 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 78 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 78 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 78 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 78 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.9 |
| 78 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.7; anxiety Nadia +3.9 (witness), Pia +3.7 (witness), Ravi +4.3 (witness) |
| 78 | Bruno | CUTOFF | Ana | — | arrives week 79; anxiety Ana +4.4, Bruno +1.8; calmer sender takes some anxiety: Ana -3.5, Bruno +3.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 78 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.2; anxiety Nadia +4.7 (witness), Ravi +10.2 (witness) |
| 78 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 79; anxiety Marta +2.1, Nadia +9.4 (witness), Pia +8.9 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 78 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 79; anxiety Marta +2.0, Nadia +8.0, Pia +4.5 (witness), Ravi +10.2 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -2.3 |
| 78 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 79; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +3.0; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -4.0 |
| 78 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 79; anxiety Marta +2.3, Nadia +1.8, Pia +4.1 (witness), Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -10.0, Nadia +10.0; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -8.0 |
| 78 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +8.6; anxiety Marta +9.2 (witness), Nadia +4.3 (witness) |
| 78 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 79; anxiety Marta +4.2, Nadia +4.3 (witness), Pia +1.7, Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -11.0, Pia +11.0; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 78 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 79; anxiety Marta +9.3 (witness), Nadia +4.4 (witness), Pia +3.8, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ravi -2.8 |
| 78 | Ravi | TRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Nadia +8.7 (witness), Pia +8.3 (witness) |
| 78 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 79; anxiety Marta +9.3 (witness), Nadia +4.0, Pia +4.2 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ravi -6.5 |
| 78 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 80; anxiety Marta +3.5 (witness), Nadia +3.4 (witness), Pia +3.2 (witness), Ravi +3.6, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.5, Sofia +3.5; Sofia rejects the other on Ravi–Sofia (rejection +0.20; the tie stays open) |
| 78 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -36.1, Bruno +20.9, Marta +15.2) |
| 78 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -22.2, Ana +22.2) |
| 78 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -112.3, Ana +22.5, Nadia +28.1, Pia +28.1, Ravi +33.7) |
| 78 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -93.3, Marta +46.6, Ravi +46.6) |
| 78 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -93.8, Marta +46.9, Ravi +46.9) |
| 78 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -204.2, Marta +61.3, Nadia +51.1, Pia +51.1, Sofia +40.8) |
| 78 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 78 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -28.1, Ravi +28.1) |
| 79 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 79 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 79 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 79 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 79 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 79 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 79 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 79 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 79 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 79 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 79 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 79 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 79 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.7; anxiety Nadia +3.9 (witness), Pia +3.7 (witness), Ravi +4.2 (witness) |
| 79 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.9 |
| 79 | Bruno | CUTOFF | Ana | — | arrives week 80; anxiety Ana +4.1, Bruno +1.8; calmer sender takes some anxiety: Ana -3.3, Bruno +3.3; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 79 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.3; anxiety Nadia +4.7 (witness), Ravi +10.2 (witness) |
| 79 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 80; anxiety Marta +2.0, Nadia +8.3, Pia +4.5 (witness), Ravi +10.2 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -2.0 |
| 79 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 80; anxiety Ana +7.3, Marta +1.2, Nadia +4.7 (witness), Pia +4.5 (witness), Ravi +5.1 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 79 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 80; anxiety Marta +2.1, Nadia +9.4 (witness), Pia +8.9 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 79 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 80; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +2.1; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -3.5 |
| 79 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 80; anxiety Marta +2.0, Nadia +1.8, Pia +4.1 (witness), Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -10.1, Nadia +10.1; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -8.3 |
| 79 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 80; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +3.4; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia rejects the other on Pia–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Pia -6.3 |
| 79 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 80; anxiety Marta +3.8, Nadia +4.3 (witness), Pia +1.7, Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -11.5, Pia +11.5; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 79 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 80; anxiety Marta +9.1 (witness), Nadia +3.5, Pia +4.1 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -2.1 |
| 79 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 80; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +3.9, Ravi +1.9; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -3.4 |
| 79 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.7; anxiety Nadia +8.5 (witness), Pia +8.1 (witness) |
| 79 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 81; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.2 (witness), Ravi +3.9, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.6, Sofia +3.6; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 79 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -31.0, Bruno +17.9, Marta +13.0) |
| 79 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -15.0, Ana +15.0) |
| 79 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -112.3, Ana +22.5, Nadia +28.1, Pia +28.1, Ravi +33.7) |
| 79 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -92.2, Marta +46.1, Ravi +46.1) |
| 79 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -102.5, Marta +51.3, Ravi +51.3) |
| 79 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -178.3, Marta +53.5, Nadia +44.6, Pia +44.6, Sofia +35.7) |
| 79 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 79 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -27.2, Ravi +27.2) |
| 80 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +5.0 |
| 80 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.5 (witness) |
| 80 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 80 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 80 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 80 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 80 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 80 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 80 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.9 |
| 80 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.7; anxiety Nadia +3.9 (witness), Pia +3.7 (witness), Ravi +4.2 (witness) |
| 80 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.15); the sender bears the refusal: anxiety Bruno +8.8 |
| 80 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 81; anxiety Dr Halim +0.1, Nadia +1.8 (witness), Pia +1.7 (witness), Ravi +2.5 (witness); calmer sender takes some anxiety: Dr Halim +6.1, Marta -6.1; the contact lands with Marta: systems perspective +0.10 |
| 80 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 81; anxiety Marta +2.1, Nadia +4.7 (witness), Pia +7.1, Ravi +10.2 (witness); Marta rejects the other on Marta–Pia (rejection +0.25; the tie stays open); impingement removed at once, anxiety Marta -10.4 |
| 80 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 81; anxiety Ana +5.1, Marta +1.2, Nadia +4.7 (witness), Pia +4.5 (witness), Ravi +5.1 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 80 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 81; anxiety Marta +2.0, Nadia +7.5, Pia +4.5 (witness), Ravi +10.2 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.7 |
| 80 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 81; anxiety Marta +1.7, Nadia +1.8, Pia +4.1 (witness), Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -9.8, Nadia +9.8; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -7.5 |
| 80 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 81; anxiety Marta +9.3 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +1.8; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 80 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 81; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +3.0; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -2.9 |
| 80 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 81; anxiety Marta +3.4, Nadia +4.3 (witness), Pia +1.7, Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -11.2, Pia +11.2; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -7.1 |
| 80 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 81; anxiety Marta +9.3 (witness), Nadia +4.4 (witness), Pia +2.9, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -3.0 |
| 80 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Nadia +8.7 (witness), Pia +8.2 (witness) |
| 80 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 82; anxiety Marta +4.7 (witness), Nadia +4.4 (witness), Pia +4.2 (witness), Ravi +1.2, Sofia +1.4; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ravi -4.4 |
| 80 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Sofia +4.1; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 80 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -32.2, Bruno +18.7, Marta +13.6) |
| 80 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -13.3, Ana +13.3) |
| 80 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -116.3, Ana +23.3, Nadia +29.1, Pia +29.1, Ravi +34.9) |
| 80 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -97.1, Marta +48.5, Ravi +48.5) |
| 80 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -94.8, Marta +47.4, Ravi +47.4) |
| 80 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -199.8, Marta +59.9, Nadia +50.0, Pia +50.0, Sofia +40.0) |
| 80 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 80 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -27.6, Ravi +27.6) |
| 81 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 81 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 81 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 81 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 81 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 81 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 81 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.9 |
| 81 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.7; anxiety Nadia +3.9 (witness), Pia +3.7 (witness), Ravi +4.2 (witness) |
| 81 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.1 |
| 81 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 82; anxiety Marta +2.0, Nadia +4.7 (witness), Pia +6.3, Ravi +10.1 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -2.5 |
| 81 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 82; anxiety Ana +4.7, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 81 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 82; anxiety Marta +2.0, Nadia +6.6, Pia +4.4 (witness), Ravi +10.1 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.5 |
| 81 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 82; anxiety Marta +2.0, Nadia +9.3 (witness), Pia +8.9 (witness), Ravi +6.4; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -2.8 |
| 81 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 82; anxiety Marta +1.5, Nadia +1.8, Pia +4.1 (witness), Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -9.0, Nadia +9.0; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -6.6 |
| 81 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 82; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +2.2; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 81 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 82; anxiety Marta +2.5, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -10.4, Pia +10.4; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -6.3 |
| 81 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 82; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +2.6; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -2.6 |
| 81 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.2 (witness), Pia +4.1 (witness) |
| 81 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 82; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +2.6, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -2.6 |
| 81 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 82; anxiety Marta +2.8, Nadia +8.6 (witness), Pia +8.2 (witness), Ravi +2.0; calmer sender takes some anxiety: Marta -9.8, Ravi +9.8; Ravi rejects the other on Marta–Ravi (rejection +0.85; the tie stays open); impingement removed at once, anxiety Ravi -11.9 |
| 81 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 83; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness), Ravi +2.6, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.9, Sofia +3.9; Sofia rejects the other on Ravi–Sofia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Sofia -0.8 |
| 81 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -29.8, Bruno +17.3, Marta +12.6) |
| 81 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -16.5, Ana +16.5) |
| 81 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -83.0, Ana +16.6, Nadia +20.8, Pia +20.8, Ravi +24.9) |
| 81 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -83.0, Marta +41.5, Ravi +41.5) |
| 81 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -79.9, Marta +39.9, Ravi +39.9) |
| 81 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -167.7, Marta +50.3, Nadia +41.9, Pia +41.9, Sofia +33.5) |
| 81 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 81 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -26.2, Ravi +26.2) |
| 82 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 82 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 82 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 82 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 82 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 82 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 82 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 82 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 82 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 82 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.8 |
| 82 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.6; anxiety Nadia +3.9 (witness), Pia +3.7 (witness), Ravi +4.2 (witness) |
| 82 | Bruno | CUTOFF | Ana | — | arrives week 83; anxiety Ana +4.4, Bruno +1.8; calmer sender takes some anxiety: Ana -3.2, Bruno +3.2; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 82 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 83; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +5.6, Ravi +10.1 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -2.2 |
| 82 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 83; anxiety Marta +2.0, Nadia +5.9, Pia +4.4 (witness), Ravi +10.1 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.3 |
| 82 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 83; anxiety Ana +4.7, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 82 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 83; anxiety Marta +9.3 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +2.0; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 82 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 83; anxiety Marta +1.3, Nadia +1.8, Pia +4.1 (witness), Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -9.1, Nadia +9.1; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -5.9 |
| 82 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 83; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +2.3; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -2.3 |
| 82 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 83; anxiety Marta +2.2, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -10.5, Pia +10.5; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -5.6 |
| 82 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.2 (witness), Pia +4.1 (witness) |
| 82 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 83; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +2.3, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -2.3 |
| 82 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Nadia +8.6 (witness), Pia +8.2 (witness) |
| 82 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 84; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness), Ravi +2.8, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.5, Sofia +3.5; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 82 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -25.3, Bruno +14.6, Marta +10.6) |
| 82 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -17.0, Ana +17.0) |
| 82 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -76.1, Ana +15.2, Nadia +19.0, Pia +19.0, Ravi +22.8) |
| 82 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -83.5, Marta +41.7, Ravi +41.7) |
| 82 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -83.3, Marta +41.7, Ravi +41.7) |
| 82 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -166.1, Marta +49.8, Nadia +41.5, Pia +41.5, Sofia +33.2) |
| 82 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 82 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -24.4, Ravi +24.4) |
| 83 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 83 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 83 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 83 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 83 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 83 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 83 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.8 |
| 83 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.7; anxiety Nadia +3.9 (witness), Pia +3.7 (witness), Ravi +4.2 (witness) |
| 83 | Bruno | CUTOFF | Ana | — | arrives week 84; anxiety Ana +3.8, Bruno +1.8; calmer sender takes some anxiety: Ana -3.1, Bruno +3.1; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 83 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 84; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +5.0, Ravi +10.1 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.9 |
| 83 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 84; anxiety Ana +4.7, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 83 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 84; anxiety Marta +2.0, Nadia +9.3 (witness), Pia +8.9 (witness), Ravi +5.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 83 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 84; anxiety Marta +2.0, Nadia +5.2, Pia +4.4 (witness), Ravi +10.1 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.1 |
| 83 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 84; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +1.8; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -2.1 |
| 83 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 84; anxiety Marta +1.1, Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -8.6, Nadia +8.6; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -5.2 |
| 83 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 84; anxiety Marta +1.9, Nadia +4.2 (witness), Pia +1.7, Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -10.0, Pia +10.0; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -5.0 |
| 83 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 84; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +2.0; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -2.0 |
| 83 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 84; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +2.0, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -2.0 |
| 83 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Nadia +8.6 (witness), Pia +8.2 (witness) |
| 83 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 84; anxiety Marta +9.2 (witness), Nadia +2.1, Pia +4.1 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.20; the tie stays open); impingement removed at once, anxiety Ravi -4.5 |
| 83 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 85; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness), Ravi +3.2, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.3, Sofia +3.3; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 83 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -25.6, Bruno +14.8, Marta +10.8) |
| 83 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -11.9, Ana +11.9) |
| 83 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -80.1, Ana +16.0, Nadia +20.0, Pia +20.0, Ravi +24.0) |
| 83 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -70.6, Marta +35.3, Ravi +35.3) |
| 83 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -71.4, Marta +35.7, Ravi +35.7) |
| 83 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -154.9, Marta +46.5, Nadia +38.7, Pia +38.7, Sofia +31.0) |
| 83 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 83 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -20.7, Ravi +20.7) |
| 84 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +5.0 |
| 84 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.5 (witness) |
| 84 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 84 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 84 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 84 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 84 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 84 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 84 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.8 |
| 84 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.6; anxiety Nadia +3.8 (witness), Pia +3.7 (witness), Ravi +4.2 (witness) |
| 84 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.15); the sender bears the refusal: anxiety Bruno +8.8 |
| 84 | Marta | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 84 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 85; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +4.4, Ravi +10.1 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.6 |
| 84 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 85; anxiety Marta +2.0, Nadia +4.6, Pia +4.4 (witness), Ravi +10.1 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.9 |
| 84 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 85; anxiety Ana +4.3, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 84 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 85; anxiety Marta +0.9, Nadia +1.8, Pia +4.1 (witness), Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -9.2, Nadia +9.2; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -4.6 |
| 84 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 85; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +1.0; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -1.8 |
| 84 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 85; anxiety Marta +1.6, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -10.5, Pia +10.5; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -4.4 |
| 84 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 85; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +1.7; calmer sender takes some anxiety: Pia +0.0, Ravi -0.0; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -1.7 |
| 84 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 85; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -1.7 |
| 84 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 85; anxiety Marta +1.9, Nadia +8.6 (witness), Pia +8.2 (witness), Ravi +2.0; calmer sender takes some anxiety: Marta -10.5, Ravi +10.5; Ravi rejects the other on Marta–Ravi (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ravi -3.8 |
| 84 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 86; anxiety Marta +4.6 (witness), Nadia +4.4 (witness), Pia +4.2 (witness), Ravi +1.2, Sofia +0.7; Ravi rejects the other on Ravi–Sofia (rejection +0.02; the tie stays open); impingement removed at once, anxiety Ravi -4.1 |
| 84 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 85; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -1.0 |
| 84 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.18); the sender bears the refusal: anxiety Sofia +4.1; anxiety Marta +3.4 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 84 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -21.7, Bruno +12.5, Marta +9.1) |
| 84 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -8.9, Ana +8.9) |
| 84 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -75.5, Ana +15.1, Nadia +18.9, Pia +18.9, Ravi +22.7) |
| 84 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -80.2, Marta +40.1, Ravi +40.1) |
| 84 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -79.8, Marta +39.9, Ravi +39.9) |
| 84 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -164.0, Marta +49.2, Nadia +41.0, Pia +41.0, Sofia +32.8) |
| 84 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 84 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -19.9, Ravi +19.9) |
| 85 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 85 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 85 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 85 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 85 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 85 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 85 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 86; anxiety Ana +1.3, Marta +0.4, Nadia +3.8 (witness), Pia +3.6 (witness), Ravi +4.2 (witness); calmer sender takes some anxiety: Ana +7.3, Marta -7.3; Ana rejects the other on Ana–Marta (rejection +0.70; the tie stays open); impingement removed at once, anxiety Ana -10.0 |
| 85 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.7 |
| 85 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.1 |
| 85 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 86; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +3.9, Ravi +10.1 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.4 |
| 85 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 86; anxiety Marta +2.0, Nadia +4.1, Pia +4.4 (witness), Ravi +10.1 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.8 |
| 85 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 86; anxiety Marta +2.0, Nadia +9.3 (witness), Pia +8.8 (witness), Ravi +4.5; Marta rejects the other on Marta–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Marta -3.4 |
| 85 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 86; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.8; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 85 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 86; anxiety Marta +0.8, Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -8.5, Nadia +8.5; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -4.1 |
| 85 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 86; anxiety Marta +1.4, Nadia +4.2 (witness), Pia +1.7, Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -9.9, Pia +9.9; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -3.9 |
| 85 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 86; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +1.5; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 85 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 87; anxiety Marta +4.6 (witness), Nadia +4.3 (witness), Pia +4.1 (witness), Ravi +1.2, Sofia +1.2; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -2.0 |
| 85 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 86; anxiety Marta +2.2, Nadia +8.6 (witness), Pia +8.2 (witness), Ravi +2.0; calmer sender takes some anxiety: Marta -9.3, Ravi +9.3; Ravi rejects the other on Marta–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -4.5 |
| 85 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.1 (witness), Pia +4.1 (witness) |
| 85 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 87; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness), Ravi +2.0, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.2, Sofia +3.2; Sofia rejects the other on Ravi–Sofia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Sofia -1.5 |
| 85 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -18.6, Bruno +10.8, Marta +7.8) |
| 85 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -10.2, Ana +10.2) |
| 85 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -70.5, Ana +14.1, Nadia +17.6, Pia +17.6, Ravi +21.1) |
| 85 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -69.4, Marta +34.7, Ravi +34.7) |
| 85 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -69.9, Marta +35.0, Ravi +35.0) |
| 85 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -141.2, Marta +42.4, Nadia +35.3, Pia +35.3, Sofia +28.2) |
| 85 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 85 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -18.8, Ravi +18.8) |
| 86 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 86 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 86 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 86 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 86 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 86 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 86 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +6.4; anxiety Nadia +3.7 (witness), Pia +3.6 (witness), Ravi +4.0 (witness) |
| 86 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.5 |
| 86 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.1 |
| 86 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 87; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +3.4, Ravi +10.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.2 |
| 86 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 87; anxiety Marta +2.0, Nadia +3.6, Pia +4.4 (witness), Ravi +10.0 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.6 |
| 86 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 87; anxiety Ana +2.8, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open); impingement removed at once, anxiety Marta -0.1 |
| 86 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 87; anxiety Marta +2.0, Nadia +9.2 (witness), Pia +8.7 (witness), Ravi +4.0; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open) |
| 86 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 87; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +1.3; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -1.4 |
| 86 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 87; anxiety Marta +0.6, Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -8.1, Nadia +8.1; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -3.6 |
| 86 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 87; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +1.9; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -1.3 |
| 86 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 87; anxiety Marta +1.2, Nadia +4.3 (witness), Pia +1.7, Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -9.5, Pia +9.5; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -3.4 |
| 86 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 88; anxiety Marta +4.6 (witness), Nadia +4.3 (witness), Pia +4.1 (witness), Ravi +1.2, Sofia +0.5; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -1.7 |
| 86 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 87; anxiety Marta +9.2 (witness), Nadia +1.4, Pia +4.1 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ravi -1.7 |
| 86 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 87; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +1.3, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ravi -2.7 |
| 86 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 88; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness), Ravi +1.7, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.5, Sofia +3.5; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Sofia -0.5 |
| 86 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -5.2, Bruno +3.0, Marta +2.2) |
| 86 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -7.3, Ana +7.3) |
| 86 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -34.2, Ana +6.8, Nadia +8.5, Pia +8.5, Ravi +10.3) |
| 86 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -57.6, Marta +28.8, Ravi +28.8) |
| 86 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -62.0, Marta +31.0, Ravi +31.0) |
| 86 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -121.3, Marta +36.4, Nadia +30.3, Pia +30.3, Sofia +24.3) |
| 86 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 86 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -16.1, Ravi +16.1) |
| 87 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 87 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 87 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 87 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 87 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 87 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 87 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.6 |
| 87 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.5; anxiety Nadia +3.7 (witness), Pia +3.6 (witness), Ravi +4.1 (witness) |
| 87 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.0 |
| 87 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 88; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +3.0, Ravi +10.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.0 |
| 87 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 88; anxiety Marta +2.0, Nadia +3.2, Pia +4.4 (witness), Ravi +10.0 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.5 |
| 87 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 88; anxiety Ana +3.1, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 87 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 88; anxiety Marta +0.5, Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -8.0, Nadia +8.0; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -3.2 |
| 87 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 88; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.6; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -1.1 |
| 87 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 88; anxiety Marta +9.0 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +1.1; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 87 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 88; anxiety Marta +1.0, Nadia +4.2 (witness), Pia +1.7, Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -9.4, Pia +9.4; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -3.0 |
| 87 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 88; anxiety Marta +9.1 (witness), Nadia +1.1, Pia +4.1 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -0.6 |
| 87 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 89; anxiety Marta +4.5 (witness), Nadia +4.3 (witness), Pia +4.1 (witness), Ravi +1.2, Sofia +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -1.5 |
| 87 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.7; anxiety Nadia +8.5 (witness), Pia +8.1 (witness) |
| 87 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 89; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness), Ravi +1.5, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.5, Sofia +3.5; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Sofia -0.3 |
| 87 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -7.7, Bruno +4.4, Marta +3.2) |
| 87 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -6.0, Ana +6.0) |
| 87 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -50.6, Ana +10.1, Nadia +12.7, Pia +12.7, Ravi +15.2) |
| 87 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -57.2, Marta +28.6, Ravi +28.6) |
| 87 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -57.2, Marta +28.6, Ravi +28.6) |
| 87 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -110.7, Marta +33.2, Nadia +27.7, Pia +27.7, Sofia +22.1) |
| 87 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 87 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -12.6, Ravi +12.6) |
| 88 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +5.0 |
| 88 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.5 (witness) |
| 88 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 88 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 88 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 88 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 88 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 88 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 88 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 89; anxiety Ana +1.3, Marta +0.1, Nadia +3.8 (witness), Pia +3.6 (witness), Ravi +4.1 (witness); calmer sender takes some anxiety: Ana +7.1, Marta -7.1; Ana rejects the other on Ana–Marta (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ana -6.0 |
| 88 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.6 |
| 88 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.9 |
| 88 | Dr Halim | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 89; anxiety Dr Halim +0.1, Marta +2.6 (witness), Nadia +1.9 (witness), Pia +1.8 (witness), Ravi +1.4; calmer sender takes some anxiety: Dr Halim +2.7, Ravi -2.7; Dr Halim takes 0.0 of functioning from the other |
| 88 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 89; anxiety Marta +2.0, Nadia +9.2 (witness), Pia +8.8 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.0 |
| 88 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 89; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +2.7, Ravi +10.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.9 |
| 88 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 89; anxiety Ana +2.8, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.1 |
| 88 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 89; anxiety Marta +2.0, Nadia +2.8, Pia +4.4 (witness), Ravi +10.0 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.4 |
| 88 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 89; anxiety Marta +9.1 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.4; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 88 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 89; anxiety Marta +0.4, Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -8.1, Nadia +8.1; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -2.8 |
| 88 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 89; anxiety Marta +0.9, Nadia +4.2 (witness), Pia +1.7, Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -9.5, Pia +9.5; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -2.7 |
| 88 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 89; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +1.5; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 88 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.1 (witness), Pia +4.1 (witness) |
| 88 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 90; anxiety Marta +4.6 (witness), Nadia +4.3 (witness), Pia +4.1 (witness), Ravi +1.2, Sofia +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -1.3 |
| 88 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 89; anxiety Marta +1.0, Nadia +8.6 (witness), Pia +8.1 (witness), Ravi +2.0; calmer sender takes some anxiety: Marta -8.8, Ravi +8.8; Ravi rejects the other on Marta–Ravi (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ravi -5.4 |
| 88 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 90; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.0 (witness), Ravi +1.3, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.4, Sofia +3.4; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Sofia -0.3 |
| 88 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -9.6, Bruno +5.6, Marta +4.1) |
| 88 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -5.8, Ana +5.8) |
| 88 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -43.0, Ana +8.6, Nadia +10.7, Pia +10.7, Ravi +12.9) |
| 88 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -52.5, Marta +26.2, Ravi +26.2) |
| 88 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -57.0, Marta +28.5, Ravi +28.5) |
| 88 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -112.6, Marta +33.8, Nadia +28.2, Pia +28.2, Sofia +22.5) |
| 88 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 88 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -11.0, Ravi +11.0) |
| 89 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 89 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 89 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 89 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 89 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 89 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 89 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.5 |
| 89 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 90; anxiety Ana +1.3, Marta +0.0, Nadia +3.7 (witness), Pia +3.6 (witness), Ravi +4.1 (witness); calmer sender takes some anxiety: Ana +7.3, Marta -7.3; Ana rejects the other on Ana–Marta (rejection +0.05; the tie stays open) |
| 89 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.0 |
| 89 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 90; anxiety Marta +2.0, Nadia +2.5, Pia +4.4 (witness), Ravi +9.9 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.3 |
| 89 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 90; anxiety Marta +2.0, Nadia +9.2 (witness), Pia +8.7 (witness), Ravi +2.7; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.9 |
| 89 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 90; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +2.3, Ravi +9.9 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.7 |
| 89 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 90; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.9; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 89 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 90; anxiety Marta +0.3, Nadia +1.8, Pia +4.1 (witness), Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -8.0, Nadia +8.0; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -2.5 |
| 89 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 90; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +1.4; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 89 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 90; anxiety Marta +0.7, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -9.4, Pia +9.4; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -2.3 |
| 89 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 90; anxiety Marta +0.9, Nadia +8.6 (witness), Pia +8.2 (witness), Ravi +2.0; calmer sender takes some anxiety: Marta -8.9, Ravi +8.9; Ravi rejects the other on Marta–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -2.7 |
| 89 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.1 (witness), Pia +4.1 (witness) |
| 89 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 91; anxiety Marta +4.6 (witness), Nadia +4.3 (witness), Pia +4.1 (witness), Ravi +1.2, Sofia +0.2; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -1.1 |
| 89 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 91; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness), Ravi +1.1, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.4, Sofia +3.4; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Sofia -0.2 |
| 89 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -5.2, Bruno +3.0, Marta +2.2) |
| 89 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -4.7, Ana +4.7) |
| 89 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -27.1, Ana +5.4, Nadia +6.8, Pia +6.8, Ravi +8.1) |
| 89 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -57.0, Marta +28.5, Ravi +28.5) |
| 89 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -61.8, Marta +30.9, Ravi +30.9) |
| 89 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -116.7, Marta +35.0, Nadia +29.2, Pia +29.2, Sofia +23.3) |
| 89 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 89 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -11.0, Ravi +11.0) |
| 90 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 90 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 90 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 90 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 90 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 90 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 90 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 91; anxiety Ana +1.3, Marta +0.4, Nadia +3.7 (witness), Pia +3.5 (witness), Ravi +4.0 (witness); calmer sender takes some anxiety: Ana +7.0, Marta -7.0; Ana rejects the other on Ana–Marta (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -1.6 |
| 90 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.5 |
| 90 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.0 |
| 90 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 91; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +2.0, Ravi +9.9 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.6 |
| 90 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 91; anxiety Marta +2.0, Nadia +2.1, Pia +4.3 (witness), Ravi +9.9 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.2 |
| 90 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 91; anxiety Ana +1.6, Marta +1.2, Nadia +4.6 (witness), Pia +4.3 (witness), Ravi +4.9 (witness); Marta rejects the other on Ana–Marta (rejection +0.10; the tie stays open); impingement removed at once, anxiety Marta -0.4 |
| 90 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 91; anxiety Marta +2.0, Nadia +9.2 (witness), Pia +8.7 (witness), Ravi +2.4; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open) |
| 90 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 91; anxiety Marta +0.2, Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -7.6, Nadia +7.6; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -2.1 |
| 90 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 91; anxiety Marta +9.1 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.8; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 90 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 91; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +1.2; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -0.6 |
| 90 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 91; anxiety Marta +0.6, Nadia +4.2 (witness), Pia +1.7, Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -9.0, Pia +9.0; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -2.0 |
| 90 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 92; anxiety Marta +4.6 (witness), Nadia +4.3 (witness), Pia +4.1 (witness), Ravi +1.2, Sofia +0.1; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -0.9 |
| 90 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 91; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +0.6, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.20; the tie stays open); impingement removed at once, anxiety Ravi -2.9 |
| 90 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.1 (witness), Pia +4.1 (witness) |
| 90 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 92; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness), Ravi +0.9, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.7, Sofia +3.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Sofia -0.1 |
| 90 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -3.6, Bruno +2.1, Marta +1.5) |
| 90 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -3.7, Ana +3.7) |
| 90 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -21.3, Ana +4.3, Nadia +5.3, Pia +5.3, Ravi +6.4) |
| 90 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -48.0, Marta +24.0, Ravi +24.0) |
| 90 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -52.4, Marta +26.2, Ravi +26.2) |
| 90 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -101.1, Marta +30.3, Nadia +25.3, Pia +25.3, Sofia +20.2) |
| 90 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 90 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -10.1, Ravi +10.1) |
| 91 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 91 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 91 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 91 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 91 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 91 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 91 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 92; anxiety Ana +1.3, Nadia +3.7 (witness), Pia +3.5 (witness), Ravi +4.0 (witness); calmer sender takes some anxiety: Ana +6.9, Marta -6.9; Ana rejects the other on Ana–Marta (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -1.4 |
| 91 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.5 |
| 91 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.9 |
| 91 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 92; anxiety Marta +2.0, Nadia +9.2 (witness), Pia +8.7 (witness), Ravi +2.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 91 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 92; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +1.8, Ravi +10.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.5 |
| 91 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 92; anxiety Ana +1.4, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.05; the tie stays open) |
| 91 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 92; anxiety Marta +2.0, Nadia +1.9, Pia +4.4 (witness), Ravi +10.0 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.1 |
| 91 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 92; anxiety Marta +9.1 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.8; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 91 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 92; anxiety Marta +0.1, Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -7.5, Nadia +7.5; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -1.9 |
| 91 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 92; anxiety Marta +9.0 (witness), Nadia +4.2 (witness), Pia +1.7, Ravi +0.5; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -0.5 |
| 91 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 92; anxiety Marta +0.5, Nadia +4.2 (witness), Pia +1.7, Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -8.9, Pia +8.9; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -1.8 |
| 91 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.1 (witness), Pia +4.1 (witness) |
| 91 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 92; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +0.5, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -0.5 |
| 91 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.7; anxiety Nadia +8.5 (witness), Pia +8.1 (witness) |
| 91 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 93; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness), Ravi +0.8, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.4, Sofia +3.4; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 91 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -1.8, Bruno +1.0, Marta +0.7) |
| 91 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -2.8, Ana +2.8) |
| 91 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -29.3, Ana +5.9, Nadia +7.3, Pia +7.3, Ravi +8.8) |
| 91 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -46.6, Marta +23.3, Ravi +23.3) |
| 91 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -47.1, Marta +23.5, Ravi +23.5) |
| 91 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -94.2, Marta +28.3, Nadia +23.6, Pia +23.6, Sofia +18.8) |
| 91 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 91 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -8.9, Ravi +8.9) |
| 92 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +5.0 |
| 92 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.5 (witness) |
| 92 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 92 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 92 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 92 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 92 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 92 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 92 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 93; anxiety Ana +1.3, Nadia +3.7 (witness), Pia +3.5 (witness), Ravi +4.0 (witness); calmer sender takes some anxiety: Ana +7.5, Marta -7.5; Ana rejects the other on Ana–Marta (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -1.2 |
| 92 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.5 |
| 92 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.9 |
| 92 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 93; anxiety Marta +2.0, Nadia +9.2 (witness), Pia +8.8 (witness), Ravi +2.4; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.9 |
| 92 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 93; anxiety Ana +1.2, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.05; the tie stays open) |
| 92 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 93; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +1.5, Ravi +10.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.4 |
| 92 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 93; anxiety Marta +2.0, Nadia +1.6, Pia +4.4 (witness), Ravi +10.0 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.1 |
| 92 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 93; anxiety Marta +0.1, Nadia +1.8, Pia +4.1 (witness), Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -8.2, Nadia +8.2; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -1.6 |
| 92 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 93; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.7; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 92 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 93; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +0.4; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; Pia rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 92 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 93; anxiety Marta +0.4, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -9.6, Pia +9.6; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -1.5 |
| 92 | Ravi | CONFLICT | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.2 (witness), Nadia +4.4 (witness) |
| 92 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 94; anxiety Marta +4.6 (witness), Nadia +4.3 (witness), Pia +4.1 (witness), Ravi +1.2, Sofia +0.7; Ravi rejects the other on Ravi–Sofia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ravi -1.6 |
| 92 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 93; anxiety Marta +1.9, Nadia +8.6 (witness), Pia +8.2 (witness), Ravi +2.0; calmer sender takes some anxiety: Marta -9.1, Ravi +9.1; Ravi rejects the other on Marta–Ravi (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ravi -5.2 |
| 92 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 94; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness), Ravi +1.2, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.7, Sofia +3.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Sofia -0.7 |
| 92 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -2.1, Bruno +1.2, Marta +0.9) |
| 92 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -2.3, Ana +2.3) |
| 92 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -32.4, Ana +6.5, Nadia +8.1, Pia +8.1, Ravi +9.7) |
| 92 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -58.5, Marta +29.3, Ravi +29.3) |
| 92 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -58.9, Marta +29.4, Ravi +29.4) |
| 92 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -123.9, Marta +37.2, Nadia +31.0, Pia +31.0, Sofia +24.8) |
| 92 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 92 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -9.6, Ravi +9.6) |
| 93 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 93 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 93 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 93 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 93 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 93 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 93 | Ana | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 93 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.5 |
| 93 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.9 |
| 93 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 94; anxiety Marta +2.0, Nadia +2.5, Pia +4.4 (witness), Ravi +10.0 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 93 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 94; anxiety Ana +1.9, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.05; the tie stays open) |
| 93 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.1; anxiety Nadia +4.6 (witness), Ravi +10.0 (witness) |
| 93 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 94; anxiety Nadia +1.8, Pia +4.1 (witness), Ravi +9.4 (witness); calmer sender takes some anxiety: Marta -8.3, Nadia +8.3; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -2.5 |
| 93 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 94; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.6; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 93 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 94; anxiety Marta +1.7, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -9.2, Pia +9.2; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 93 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +8.6; anxiety Marta +9.1 (witness), Nadia +4.3 (witness) |
| 93 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.9; anxiety Marta +9.2 (witness), Pia +4.1 (witness) |
| 93 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 94; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +1.5, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ravi -0.1 |
| 93 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Nadia +8.6 (witness), Pia +8.2 (witness) |
| 93 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 95; anxiety Marta +4.6 (witness), Nadia +4.3 (witness), Pia +4.1 (witness), Ravi +1.2, Sofia +0.6; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -0.5 |
| 93 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 95; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness), Ravi +0.5, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.7, Sofia +3.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Sofia -0.6 |
| 93 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -4.1, Bruno +2.4, Marta +1.7) |
| 93 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -2.4, Ana +2.4) |
| 93 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -24.5, Ana +4.9, Nadia +6.1, Pia +6.1, Ravi +7.3) |
| 93 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -56.3, Marta +28.2, Ravi +28.2) |
| 93 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -53.0, Marta +26.5, Ravi +26.5) |
| 93 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -109.1, Marta +32.7, Nadia +27.3, Pia +27.3, Sofia +21.8) |
| 93 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 93 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -10.6, Ravi +10.6) |
| 94 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 94 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 94 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 94 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 94 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 94 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 94 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.2; anxiety Nadia +3.7 (witness), Pia +3.5 (witness), Ravi +4.0 (witness) |
| 94 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.2 |
| 94 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +9.0 |
| 94 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 95; anxiety Marta +2.0, Nadia +9.3 (witness), Pia +8.8 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 94 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 95; anxiety Marta +2.0, Nadia +3.4, Pia +4.4 (witness), Ravi +10.0 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 94 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.1; anxiety Nadia +4.6 (witness), Ravi +10.1 (witness) |
| 94 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 95; anxiety Ana +3.2, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 94 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 95; anxiety Marta +9.1 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.6; Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 94 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 95; anxiety Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -8.2, Nadia +8.2; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -3.4 |
| 94 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 95; anxiety Marta +2.0, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -9.6, Pia +9.6; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 94 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 95; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +1.5; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Pia rejects the other on Pia–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Pia -0.8 |
| 94 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Nadia +8.5 (witness), Pia +8.1 (witness) |
| 94 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.1 (witness), Pia +4.1 (witness) |
| 94 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 96; anxiety Marta +4.6 (witness), Nadia +4.3 (witness), Pia +4.1 (witness), Ravi +1.2, Sofia +0.8; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 94 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.2; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 94 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -1.9, Ana +1.9) |
| 94 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -47.5, Ana +9.5, Nadia +11.9, Pia +11.9, Ravi +14.2) |
| 94 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -48.7, Marta +24.4, Ravi +24.4) |
| 94 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -55.9, Marta +28.0, Ravi +28.0) |
| 94 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -102.5, Marta +30.8, Nadia +25.6, Pia +25.6, Sofia +20.5) |
| 94 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 94 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -10.6, Ravi +10.6) |
| 95 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 95 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 95 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 95 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 95 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 95 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 95 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.3 |
| 95 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.3; anxiety Nadia +3.7 (witness), Pia +3.5 (witness), Ravi +4.0 (witness) |
| 95 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +8.9 |
| 95 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.1; anxiety Nadia +4.6 (witness), Ravi +10.1 (witness) |
| 95 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 96; anxiety Ana +3.8, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 95 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 96; anxiety Marta +2.0, Nadia +9.2 (witness), Pia +8.8 (witness), Ravi +5.1; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open) |
| 95 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 95 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 96; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.5; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -1.8 |
| 95 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 96; anxiety Marta +1.8, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -9.7, Pia +9.7; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 95 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 96; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +1.9; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -1.7 |
| 95 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 96; anxiety Marta +9.2 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.40; the tie stays open); impingement removed at once, anxiety Ravi -1.3 |
| 95 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 96; anxiety Marta +9.2 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ravi -2.8 |
| 95 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 97; anxiety Marta +4.6 (witness), Nadia +4.3 (witness), Pia +4.1 (witness), Ravi +1.2, Sofia +1.3; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 95 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 96; anxiety Dr Halim +0.4, Marta +6.0 (witness), Nadia +4.4 (witness), Pia +4.2 (witness), Ravi +0.6 |
| 95 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 95 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -1.2, Bruno +0.7, Marta +0.5) |
| 95 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -1.7, Ana +1.7) |
| 95 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -37.4, Ana +7.5, Nadia +9.3, Pia +9.3, Ravi +11.2) |
| 95 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -53.2, Marta +26.6, Ravi +26.6) |
| 95 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -56.4, Marta +28.2, Ravi +28.2) |
| 95 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -115.8, Marta +34.7, Nadia +28.9, Pia +28.9, Sofia +23.2) |
| 95 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 95 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -10.9, Ravi +10.9) |
| 96 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +5.0 |
| 96 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.5 (witness) |
| 96 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 96 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 96 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 96 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 96 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 96 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.4 |
| 96 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.3; anxiety Nadia +3.7 (witness), Pia +3.5 (witness), Ravi +4.1 (witness) |
| 96 | Bruno | CUTOFF | Ana | — | arrives week 97; anxiety Ana +5.7, Bruno +1.8; calmer sender takes some anxiety: Ana -2.8, Bruno +2.8; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 96 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 97; anxiety Dr Halim +0.1, Nadia +1.9 (witness), Pia +1.8 (witness), Ravi +2.6 (witness); calmer sender takes some anxiety: Dr Halim +5.5, Marta -5.5; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.52; the tie stays open) |
| 96 | Marta | WITHHOLD | Pia | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 96 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 97; anxiety Marta +2.0, Nadia +3.7, Pia +4.4 (witness), Ravi +10.1 (witness); Marta rejects the other on Marta–Nadia (rejection +0.10; the tie stays open) |
| 96 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 97; anxiety Marta +2.0, Nadia +9.3 (witness), Pia +8.8 (witness), Ravi +4.6; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open) |
| 96 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 97; anxiety Marta +9.1 (witness), Nadia +1.8, Pia +4.1 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 96 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 97; anxiety Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -8.1, Nadia +8.1; Nadia rejects the other on Marta–Nadia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Nadia -3.7 |
| 96 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 96 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 97; anxiety Marta +1.6, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -9.5, Pia +9.5; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open) |
| 96 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 97; anxiety Marta +9.0 (witness), Nadia +4.3 (witness), Pia +1.4, Ravi +1.9; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 96 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 98; anxiety Marta +4.5 (witness), Nadia +4.3 (witness), Pia +4.0 (witness), Ravi +1.2, Sofia +1.2; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 96 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.7; anxiety Marta +9.0 (witness), Pia +4.0 (witness) |
| 96 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.2; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 96 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -3.0, Bruno +1.7, Marta +1.3) |
| 96 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -1.7, Ana +1.7) |
| 96 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -70.8, Ana +14.2, Nadia +17.7, Pia +17.7, Ravi +21.2) |
| 96 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -54.9, Marta +27.5, Ravi +27.5) |
| 96 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -59.2, Marta +29.6, Ravi +29.6) |
| 96 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -104.1, Marta +31.2, Nadia +26.0, Pia +26.0, Sofia +20.8) |
| 96 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 96 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -11.0, Ravi +11.0) |
| 97 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 97 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 97 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 97 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 97 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 97 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 97 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.4; anxiety Nadia +3.8 (witness), Pia +3.6 (witness), Ravi +4.1 (witness) |
| 97 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.5 |
| 97 | Bruno | CUTOFF | Ana | — | arrives week 98; anxiety Ana +5.4, Bruno +1.8; calmer sender takes some anxiety: Ana -2.6, Bruno +2.6; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 97 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +4.6 (witness), Ravi +10.0 (witness) |
| 97 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 98; anxiety Ana +3.9, Marta +1.2, Nadia +4.6 (witness), Pia +4.4 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 97 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 98; anxiety Marta +2.0, Nadia +9.2 (witness), Pia +8.7 (witness), Ravi +5.3; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 97 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 98; anxiety Marta +9.0 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.1; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -1.3 |
| 97 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 98; anxiety Nadia +1.8, Pia +4.0 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -7.6, Nadia +7.6; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 97 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 98; anxiety Marta +9.0 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +0.9; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; Pia rejects the other on Pia–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Pia -2.7 |
| 97 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 98; anxiety Marta +1.5, Nadia +4.2 (witness), Pia +1.7, Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -9.0, Pia +9.0; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 97 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.7; anxiety Nadia +8.5 (witness), Pia +8.1 (witness) |
| 97 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 98; anxiety Marta +9.0 (witness), Nadia +1.3, Pia +4.0 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ravi -0.1 |
| 97 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 98; anxiety Marta +9.0 (witness), Nadia +4.3 (witness), Pia +1.8, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -0.9 |
| 97 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 99; anxiety Marta +3.4 (witness), Nadia +3.3 (witness), Pia +3.1 (witness), Ravi +0.6, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.5, Sofia +3.5; Sofia rejects the other on Ravi–Sofia (rejection +0.20; the tie stays open); impingement removed at once, anxiety Sofia -1.3 |
| 97 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -6.7, Bruno +3.9, Marta +2.8) |
| 97 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -1.0, Ana +1.0) |
| 97 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -29.0, Ana +5.8, Nadia +7.2, Pia +7.2, Ravi +8.7) |
| 97 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -34.1, Marta +17.0, Ravi +17.0) |
| 97 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -42.4, Marta +21.2, Ravi +21.2) |
| 97 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -79.3, Marta +23.8, Nadia +19.8, Pia +19.8, Sofia +15.9) |
| 97 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 97 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -7.8, Ravi +7.8) |
| 98 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 98 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 98 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 98 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 98 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 98 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 98 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 99; anxiety Ana +1.3, Nadia +3.8 (witness), Pia +3.6 (witness), Ravi +4.1 (witness); calmer sender takes some anxiety: Ana +7.1, Marta -7.1; Ana rejects the other on Ana–Marta (rejection +0.30; the tie stays open); impingement removed at once, anxiety Ana -9.1 |
| 98 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.4 |
| 98 | Bruno | CUTOFF | Ana | — | arrives week 99; anxiety Ana +4.9, Bruno +1.7; calmer sender takes some anxiety: Ana -2.4, Bruno +2.4; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 98 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +4.6 (witness), Ravi +10.1 (witness) |
| 98 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 99; anxiety Ana +4.4, Marta +1.2, Nadia +4.6 (witness), Pia +4.3 (witness), Ravi +5.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.05; the tie stays open) |
| 98 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 99; anxiety Marta +2.0, Nadia +9.2 (witness), Pia +8.7 (witness), Ravi +3.7; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.2 |
| 98 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 99; anxiety Marta +2.0, Nadia +4.9, Pia +4.3 (witness), Ravi +10.0 (witness); Marta rejects the other on Marta–Nadia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Marta -0.0 |
| 98 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 99; anxiety Marta +9.1 (witness), Nadia +1.8, Pia +4.1 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.00; the tie stays open) |
| 98 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 99; anxiety Marta +0.0, Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -7.8, Nadia +7.8; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -4.9 |
| 98 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 98 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Pia +8.5; anxiety Marta +9.0 (witness), Nadia +4.3 (witness) |
| 98 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 99; anxiety Marta +9.1 (witness), Nadia +4.3 (witness), Pia +1.1, Ravi +2.0; Ravi rejects the other on Pia–Ravi (rejection +0.00; the tie stays open) |
| 98 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +9.8; anxiety Marta +9.0 (witness), Pia +4.0 (witness) |
| 98 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 99; anxiety Marta +1.2, Nadia +8.5 (witness), Pia +8.1 (witness), Ravi +2.0; calmer sender takes some anxiety: Marta -8.6, Ravi +8.6; Ravi rejects the other on Marta–Ravi (rejection +0.30; the tie stays open); impingement removed at once, anxiety Ravi -11.9 |
| 98 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 100; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness), Ravi +1.7, Sofia +0.8; calmer sender takes some anxiety: Ravi -3.2, Sofia +3.2; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 98 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -4.7, Bruno +2.7, Marta +2.0) |
| 98 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -0.4, Ana +0.4) |
| 98 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -28.8, Ana +5.8, Nadia +7.2, Pia +7.2, Ravi +8.6) |
| 98 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -45.8, Marta +22.9, Ravi +22.9) |
| 98 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -43.7, Marta +21.8, Ravi +21.8) |
| 98 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -88.2, Marta +26.5, Nadia +22.1, Pia +22.1, Sofia +17.6) |
| 98 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 98 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -7.0, Ravi +7.0) |
| 99 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 99 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 99 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 99 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 99 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 99 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 99 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 100; anxiety Ana +1.3, Nadia +3.7 (witness), Pia +3.5 (witness), Ravi +4.0 (witness); calmer sender takes some anxiety: Ana +7.1, Marta -7.1; Ana rejects the other on Ana–Marta (rejection +0.05; the tie stays open) |
| 99 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.3 |
| 99 | Bruno | CUTOFF | Ana | — | arrives week 100; anxiety Ana +4.4, Bruno +1.7; calmer sender takes some anxiety: Ana -2.3, Bruno +2.3; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 99 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +4.6 (witness), Ravi +10.1 (witness) |
| 99 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 100; anxiety Marta +2.0, Nadia +9.2 (witness), Pia +8.7 (witness), Ravi +6.0; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -1.0 |
| 99 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 100; anxiety Marta +2.0, Nadia +5.5, Pia +4.3 (witness), Ravi +10.0 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 99 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 100; anxiety Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -7.7, Nadia +7.7; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -5.5 |
| 99 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 100; anxiety Marta +9.1 (witness), Nadia +1.8, Pia +4.1 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 99 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 100; anxiety Marta +2.4, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -9.1, Pia +9.1; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 99 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 100; anxiety Marta +9.0 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +2.0; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3; Pia rejects the other on Pia–Ravi (rejection +0.10; the tie stays open); impingement removed at once, anxiety Pia -2.0 |
| 99 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 100; anxiety Marta +1.0, Nadia +8.5 (witness), Pia +8.0 (witness), Ravi +1.9; calmer sender takes some anxiety: Marta -8.8, Ravi +8.8; Ravi rejects the other on Marta–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -6.0 |
| 99 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 100; anxiety Marta +9.0 (witness), Nadia +4.3 (witness), Pia +1.5, Ravi +1.9; Ravi rejects the other on Pia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -2.0 |
| 99 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 101; anxiety Marta +4.5 (witness), Nadia +4.3 (witness), Pia +4.0 (witness), Ravi +1.2, Sofia +1.0; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ravi -1.2 |
| 99 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Sofia +4.1; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness) |
| 99 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -0.5, Bruno +0.3, Marta +0.2) |
| 99 | — | (system) | — | — | Bruno's symptom is past its turn and holds their anxiety no longer (Bruno -0.0, Ana +0.0) |
| 99 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -22.8, Ana +4.6, Nadia +5.7, Pia +5.7, Ravi +6.8) |
| 99 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -41.6, Marta +20.8, Ravi +20.8) |
| 99 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -41.3, Marta +20.7, Ravi +20.7) |
| 99 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -70.9, Marta +21.3, Nadia +17.7, Pia +17.7, Sofia +14.2) |
| 99 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 99 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -4.9, Ravi +4.9) |
| 100 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +5.0 |
| 100 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.3 (witness), Pia +2.2 (witness), Ravi +2.5 (witness) |
| 100 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 100 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 100 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 100 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 100 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 100 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 100 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +6.3; anxiety Nadia +3.7 (witness), Pia +3.5 (witness), Ravi +4.0 (witness) |
| 100 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.3 |
| 100 | Bruno | CUTOFF | Ana | — | arrives week 101; anxiety Ana +3.9, Bruno +1.7; calmer sender takes some anxiety: Ana -2.6, Bruno +2.6; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 100 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 101; anxiety Ana +4.7, Marta +1.2, Nadia +4.5 (witness), Pia +4.3 (witness), Ravi +4.9 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 100 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +4.6 (witness), Ravi +10.0 (witness) |
| 100 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 101; anxiety Marta +2.0, Nadia +9.1 (witness), Pia +8.6 (witness), Ravi +6.5; Marta rejects the other on Marta–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -2.3 |
| 100 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 101; anxiety Marta +9.1 (witness), Nadia +1.8, Pia +4.1 (witness), Ravi +0.1; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -2.2 |
| 100 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 101; anxiety Nadia +1.8, Pia +4.1 (witness), Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -7.6, Nadia +7.6; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 100 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 101; anxiety Marta +9.0 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +1.7; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 100 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 101; anxiety Marta +2.2, Nadia +4.3 (witness), Pia +1.7, Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -9.0, Pia +9.0; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 100 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 101; anxiety Marta +2.3, Nadia +8.6 (witness), Pia +8.1 (witness), Ravi +2.0; calmer sender takes some anxiety: Marta -8.8, Ravi +8.8; Ravi rejects the other on Marta–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ravi -6.5 |
| 100 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 101; anxiety Marta +9.1 (witness), Nadia +2.2, Pia +4.1 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ravi -0.1 |
| 100 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +5.9; anxiety Marta +4.6 (witness), Nadia +4.3 (witness), Pia +4.1 (witness) |
| 100 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.1; anxiety Marta +3.4 (witness), Nadia +3.2 (witness), Pia +3.1 (witness) |
| 100 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -0.3, Bruno +0.2, Marta +0.1) |
| 100 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -7.0, Ana +1.4, Nadia +1.7, Pia +1.7, Ravi +2.1) |
| 100 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -37.8, Marta +18.9, Ravi +18.9) |
| 100 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -32.2, Marta +16.1, Ravi +16.1) |
| 100 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -73.5, Marta +22.0, Nadia +18.4, Pia +18.4, Sofia +14.7) |
| 100 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 100 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -3.8, Ravi +3.8) |
| 101 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 101 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 101 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 101 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 101 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 101 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 101 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +6.3; anxiety Nadia +3.7 (witness), Pia +3.5 (witness), Ravi +4.0 (witness) |
| 101 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.3 |
| 101 | Bruno | CUTOFF | Ana | — | arrives week 102; anxiety Ana +3.4, Bruno +1.6; calmer sender takes some anxiety: Ana -2.7, Bruno +2.7; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 101 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 102; anxiety Marta +2.0, Nadia +4.6 (witness), Pia +5.0, Ravi +10.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.45; the tie stays open); impingement removed at once, anxiety Marta -5.3 |
| 101 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 102; anxiety Marta +2.0, Nadia +5.3, Pia +4.3 (witness), Ravi +9.9 (witness); Marta rejects the other on Marta–Nadia (rejection +0.10; the tie stays open) |
| 101 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 102; anxiety Ana +4.7, Marta +1.2, Nadia +4.5 (witness), Pia +4.3 (witness), Ravi +4.9 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 101 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 101 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 102; anxiety Marta +9.1 (witness), Nadia +1.8, Pia +4.1 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -1.9 |
| 101 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 102; anxiety Marta +9.0 (witness), Nadia +4.3 (witness), Pia +1.7, Ravi +2.1; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 101 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 102; anxiety Marta +2.0, Nadia +4.3 (witness), Pia +1.7, Ravi +9.3 (witness); calmer sender takes some anxiety: Marta -8.4, Pia +8.4; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open); impingement removed at once, anxiety Pia -5.0 |
| 101 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 102; anxiety Marta +9.0 (witness), Nadia +1.9, Pia +4.0 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 101 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 102; anxiety Marta +2.0, Nadia +8.5 (witness), Pia +8.1 (witness), Ravi +1.9; calmer sender takes some anxiety: Marta -8.0, Ravi +8.0; Ravi rejects the other on Marta–Ravi (rejection +0.05; the tie stays open) |
| 101 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 103; anxiety Marta +4.6 (witness), Nadia +4.3 (witness), Pia +4.1 (witness), Ravi +1.2, Sofia +1.3; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 101 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +4.1; anxiety Marta +3.5 (witness), Nadia +3.3 (witness), Pia +3.1 (witness) |
| 101 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -0.3, Bruno +0.2, Marta +0.1) |
| 101 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -12.4, Ana +2.5, Nadia +3.1, Pia +3.1, Ravi +3.7) |
| 101 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -37.6, Marta +18.8, Ravi +18.8) |
| 101 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -40.5, Marta +20.2, Ravi +20.2) |
| 101 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -72.1, Marta +21.6, Nadia +18.0, Pia +18.0, Sofia +14.4) |
| 101 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 101 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -3.6, Ravi +3.6) |
| 102 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 102 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 102 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 102 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 102 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 102 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 102 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 103; anxiety Ana +1.3, Marta +0.2, Nadia +3.7 (witness), Pia +3.5 (witness), Ravi +4.0 (witness); calmer sender takes some anxiety: Ana +6.7, Marta -6.7; Ana rejects the other on Ana–Marta (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ana -10.4 |
| 102 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +8.3 |
| 102 | Bruno | CUTOFF | Ana | — | arrives week 103; anxiety Ana +3.0, Bruno +1.6; calmer sender takes some anxiety: Ana -2.5, Bruno +2.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 102 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 103; anxiety Ana +5.1, Marta +1.2, Nadia +4.5 (witness), Pia +4.3 (witness), Ravi +4.9 (witness); Marta rejects the other on Ana–Marta (rejection +0.05; the tie stays open); impingement removed at once, anxiety Marta -0.2 |
| 102 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +10.0; anxiety Nadia +4.6 (witness), Ravi +10.0 (witness) |
| 102 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 103; anxiety Marta +2.0, Nadia +9.1 (witness), Pia +8.6 (witness), Ravi +6.4; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open); impingement removed at once, anxiety Marta -1.2 |
| 102 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 103; anxiety Marta +2.0, Nadia +6.4, Pia +4.3 (witness), Ravi +9.9 (witness); Marta rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 102 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 103; anxiety Marta +9.0 (witness), Nadia +1.8, Pia +4.0 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open); impingement removed at once, anxiety Nadia -1.6 |
| 102 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 103; anxiety Nadia +1.8, Pia +4.0 (witness), Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -7.0, Nadia +7.0; Nadia rejects the other on Marta–Nadia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Nadia -9.9 |
| 102 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 103; anxiety Marta +9.0 (witness), Nadia +4.2 (witness), Pia +1.7, Ravi +1.9; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 102 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 103; anxiety Marta +1.2, Nadia +4.2 (witness), Pia +1.7, Ravi +9.2 (witness); calmer sender takes some anxiety: Marta -8.4, Pia +8.4; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 102 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | — |
| 102 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 103; anxiety Marta +9.0 (witness), Nadia +1.6, Pia +4.0 (witness), Ravi +2.0; Ravi rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 102 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +9.7; anxiety Nadia +8.5 (witness), Pia +8.1 (witness) |
| 102 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 102 | — | (system) | — | — | Ana's symptom is past its turn and holds their anxiety no longer (Ana -0.1, Bruno +0.1, Marta +0.0) |
| 102 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -6.4, Ana +1.3, Nadia +1.6, Pia +1.6, Ravi +1.9) |
| 102 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -24.5, Marta +12.3, Ravi +12.3) |
| 102 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -28.3, Marta +14.2, Ravi +14.2) |
| 102 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -56.9, Marta +17.1, Nadia +14.2, Pia +14.2, Sofia +11.4) |
| 102 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 102 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -2.7, Ravi +2.7) |
| 103 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 103 | — | (system) | — | — | Marta–Nadia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 103 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.5 |
| 103 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 103 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 103 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 103 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | — |
| 103 | Ana | UNDERFUNCTION | Bruno | — | — |
| 103 | Bruno | CONFLICT | Ana | — | — |
| 103 | Marta | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 103 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | — |
| 103 | Marta | PURSUE | Pia | Nadia, Ravi | — |
| 103 | Nadia | CUTOFF | Marta | Pia, Ravi | — |
| 103 | Nadia | CUTOFF | Ravi | Marta, Pia | — |
| 103 | Pia | CUTOFF | Ravi | Marta, Nadia | — |
| 103 | Pia | CUTOFF | Marta | Nadia, Ravi | — |
| 103 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | — |
| 103 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | — |
| 103 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | — |
| 103 | Ravi | CUTOFF | Pia | Marta, Nadia | — |
| 103 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | — |
| 103 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -28.8, Marta +14.4, Ravi +14.4) |
| 103 | — | (system) | — | — | Pia's symptom is past its turn and holds their anxiety no longer (Pia -35.5, Marta +17.8, Ravi +17.8) |
| 103 | — | (system) | — | — | Ravi's symptom is past its turn and holds their anxiety no longer (Ravi -72.8, Marta +21.8, Nadia +18.2, Pia +18.2, Sofia +14.6) |
| 103 | — | (system) | — | — | Sofia's symptom gives back 0.0 (Ravi +0.0) |
| 103 | — | (system) | — | — | Sofia's symptom is past its turn and holds their anxiety no longer (Sofia -2.5, Ravi +2.5) |
| 103 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |

104 weeks. The invariants were asserted at the end of every week.
