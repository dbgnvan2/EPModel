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
| config hash | `c9cc593d9066535be8fefc848fa35c00fb84fce2ef98929c2814688af32f9065` |
| spec revision | 2.0, revision 39 |
| family instance | phase_c |
| activation | synchronous_activation v1 (synchronous) |
| visibility | household_conductance_visibility v1 |
| constants frozen | 2026-10-07 |
| constants changed since freeze | acting_age_years, attribution_rate, belief_appraisal_gain, birth_focus_gain, budget_per_undifferentiation, channel_protection, coach_retirement_age_years, coach_start_age_years, conflict_binding_rate, constitution_strength, cutoff_decay_weeks, cutoff_floor_share, cutoff_stock_cap, cutoff_threat_weight, danger_gain, danger_threshold, danger_window, death_transfer_share, declared_rejection, development_band, development_band_projection, development_band_support, development_end_age_years, development_rate, development_start_age_years, estimator_breadth, estimator_rate, estimator_sd_weight, estimator_window, exchange_window, find_work_pressure_scale, find_work_rate, focus_compound_gain, focus_rate, focus_reinforce_gain, focus_steer_gain, force_anxiety_gain, grief_extra_weeks, grief_min_weeks, grief_weight, income_loss_intensity, income_shortfall_share, launch_age_years, lcu_scale, lock_in_gain, lock_in_turn, marital_distance_fraction, mortality_a, mortality_b, new_tie_investment_share, nodal_spike_intensity, origin_cutoff_anxiety_gain, origin_fusion_gain, pattern_min_acts, pattern_projection_hold, pattern_window, perceived_weakness_gain, perspective_channel_gain, perspective_steer_gain, positive_bridge_dissipation, practice_window, pseudo_self_relaxation_rate, reactive_cutoff_gain, refusal_cost_gain, rejection_cutoff_gain, rejection_fade, rejection_refusal_gain, rejection_refusal_threshold, rejection_relief, response_down_gain, response_relaxation_rate, response_up_gain, self_practice_gain, soothing_review_every, soothing_review_penalty, stress_floor_below_basic, stressor_persistence, system_adjust_gain, tie_repair_steps, tie_wear_rate, usual_level_rate, wear_offset_per_act, witness_window, witnessed_cutoff_gain, witnessed_gain, working_age_years |

The standing load, the decay toward each person's floor, the relaxation of felt contact, symptom accumulation, a symptom's weekly hold on the family's anxiety (lock-in; what it gives back is shown), investment, the reactive detectors, outside-ness, attribution, the family's focus and borrowed pseudo-self settling back run every week for everyone and are not listed line by line. Effects are shown beside the event that caused them.

| Week | Who | Move | Toward | Witnesses | What it did |
|---|---|---|---|---|---|
| 0 | Ana | WITHHOLD | Bruno | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 0 | Bruno | CUTOFF | Ana | — | arrives week 1; anxiety Ana +1.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 0 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 1; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.3, Marta -0.3 |
| 0 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | arrives week 1; anxiety Marta +0.5, Nadia +0.6 (witness), Pia -0.1, Ravi +1.3 (witness); Pia takes 2.0 of functioning from the other |
| 0 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 1; anxiety Marta -0.3, Nadia +0.5, Pia +0.6 (witness), Ravi +1.3 (witness); Marta left outside: anxiety Marta +0.6, Nadia -0.4 (of which 0.2 the outsider's own) |
| 0 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 1; anxiety Marta +1.4 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +3.8; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia takes 2.0 of functioning from the other |
| 0 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 1; anxiety Marta -0.6, Nadia +1.2 (witness), Pia +1.1 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1; Marta left outside: anxiety Marta +1.2, Ravi -0.8 (of which 0.4 the outsider's own) |
| 0 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 2; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +2.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Sofia takes 2.0 of functioning from the other |
| 1 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 1 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 1 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 1 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 1 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 1 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 1 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.3 |
| 1 | Bruno | CUTOFF | Ana | — | arrives week 2; anxiety Ana +1.5, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 1 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 2; anxiety Marta +0.6, Nadia +0.9, Pia +0.6 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3; Marta rejects the other on Marta–Nadia (rejection +0.54; the tie stays open); impingement removed at once, anxiety Marta -0.6 |
| 1 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 2; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi -0.5; Ravi left outside: anxiety Nadia -0.5, Ravi +0.8 (of which 0.3 the outsider's own) |
| 1 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 2; anxiety Marta +1.8 (witness), Nadia +0.9 (witness), Pia +0.4, Ravi +3.6; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; 1.0 of the sender's anxiety bound into Pia–Ravi |
| 1 | Ravi | WITHHOLD | Nadia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 1 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 3; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +3.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6; 0.4 of the sender's anxiety bound into Ravi–Sofia |
| 2 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 2 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 3; anxiety Ana +0.3, Marta -0.7, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.3, Marta -0.3 |
| 2 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.11); the sender bears the refusal: anxiety Bruno +2.9 |
| 2 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 3; anxiety Dr Halim +0.1, Marta +0.2, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +1.1 (witness) |
| 2 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.11); the sender bears the refusal: anxiety Nadia +2.7; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 2 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 3; anxiety Marta +2.0 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +2.6; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia takes 2.2 of functioning from the other |
| 2 | Ravi | STAY-IN-CONTACT | Nadia | Marta, Pia | arrives week 3; anxiety Marta +1.3 (witness), Nadia -0.7, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.2, Ravi +0.2 |
| 2 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 4; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +2.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 1.1 of the sender's anxiety bound into Ravi–Sofia |
| 3 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 3 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 4; anxiety Ana +0.4, Marta +2.6, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.3, Marta -0.3; 2.4 of the sender's anxiety bound into Ana–Marta |
| 3 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 3 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 4; anxiety Marta +0.6, Nadia +1.6 (witness), Pia +1.3 (witness), Ravi +2.1; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3 |
| 3 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.7 (witness), Ravi +2.2 (witness) |
| 3 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 4; anxiety Marta +2.8, Nadia +0.8 (witness), Pia +0.5, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5 |
| 3 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 4; anxiety Marta +2.7 (witness), Nadia +1.2 (witness), Pia +4.4, Ravi +0.8; 6.6 of the sender's anxiety bound into Pia–Ravi |
| 3 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 5; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +3.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia takes 2.1 of functioning from the other |
| 4 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.6 (witness), Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +3.7 |
| 4 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +3.2, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +1.8 (witness) |
| 4 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 4 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 4 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.11); the sender bears the refusal: anxiety Ana +2.4 |
| 4 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.11); the sender bears the refusal: anxiety Bruno +3.1 |
| 4 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 5; anxiety Marta +0.7, Nadia +1.6, Pia +0.8 (witness), Ravi +2.4 (witness); calmer sender takes some anxiety: Marta +0.2, Nadia -0.2; Marta rejects the other on Marta–Nadia (rejection +0.38; the tie stays open) |
| 4 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 5; anxiety Marta +1.8 (witness), Nadia +0.7, Pia +0.7 (witness), Ravi +5.0; 9.0 of the sender's anxiety bound into Nadia–Ravi |
| 4 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 5; anxiety Marta +2.5 (witness), Nadia +1.1 (witness), Pia +0.5, Ravi +2.5; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Ravi left outside: anxiety Pia -1.5, Ravi +2.2 (of which 0.7 the outsider's own) |
| 4 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 5; anxiety Marta +2.6 (witness), Nadia +1.2 (witness), Pia +1.8, Ravi +0.9 |
| 4 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 6; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +1.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; 1.6 of the sender's anxiety bound into Ravi–Sofia |
| 5 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 5 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 5 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +0.4, Marta +3.3, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana takes 2.3 of functioning from the other |
| 5 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.08); the sender bears the refusal: anxiety Bruno +3.2 |
| 5 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +3.9, Marta +0.5, Nadia +1.1 (witness), Pia +0.9 (witness), Ravi +1.4 (witness); 8.1 of the sender's anxiety bound into Ana–Marta |
| 5 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 6; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +1.5; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 5 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 6; anxiety Marta +2.5 (witness), Nadia +1.2 (witness), Pia +0.5, Ravi +2.4; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Ravi takes 1.2 of functioning from the other |
| 5 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 6; anxiety Marta +2.9 (witness), Nadia +1.3 (witness), Pia +1.8, Ravi +1.0; 9.0 of the sender's anxiety bound into Pia–Ravi |
| 5 | Sofia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 5 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 6, 7; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.6, Nadia +2.4, Pia +2.0; anxiety Sofia +1.3 |
| 6 | (from outside) | OWN_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +1.8; still weighing this week: anxiety Ravi +0.9; still weighing this week: anxiety Ravi +0.9; still weighing this week: anxiety Ravi +0.9; still weighing this week: anxiety Ravi +1.0; still weighing this week: anxiety Ravi +1.0; still weighing this week: anxiety Ravi +1.0; still weighing this week: anxiety Ravi +1.0; still weighing this week: anxiety Ravi +1.0; still weighing this week: anxiety Ravi +1.0; still weighing this week: anxiety Ravi +1.0; still weighing this week: anxiety Ravi +0.9 |
| 6 | (from outside) | FAMILY_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +0.7, Nadia +0.3 (witness), Pia +0.3 (witness), Ravi +0.5 (witness); still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4 |
| 6 | (from outside) | FAMILY_ILLNESS | Nadia | Marta, Ravi | a 12-week spell; anxiety Marta +0.3 (witness), Nadia +0.6, Ravi +0.4 (witness); still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2 |
| 6 | (from outside) | FAMILY_ILLNESS | Pia | Marta, Ravi | a 12-week spell; anxiety Marta +0.3 (witness), Pia +0.5, Ravi +0.4 (witness); still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3 |
| 6 | (from outside) | FAMILY_ILLNESS | Sofia | — | a 12-week spell; anxiety Sofia +0.4; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2 |
| 6 | — | (system) | — | — | Nadia–Ravi: a positive contact repairs the tie; held anxiety -3.0 |
| 6 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 6 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.12); the sender bears the refusal: anxiety Ana +2.5 |
| 6 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.3 |
| 6 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | arrives week 7; anxiety Marta +0.8, Nadia +1.1 (witness), Pia +4.8, Ravi +2.9 (witness); Marta takes 2.1 of functioning from the other |
| 6 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 7; anxiety Marta +2.3 (witness), Nadia +0.7, Pia +0.8 (witness), Ravi +6.4; calmer sender takes some anxiety: Nadia +0.0, Ravi -0.0; Nadia takes 2.7 of functioning from the other |
| 6 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 7; anxiety Marta +2.5 (witness), Nadia +1.1 (witness), Pia +0.6, Ravi +2.2; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Ravi takes 1.0 of functioning from the other |
| 6 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 7; anxiety Dr Halim +0.3, Marta +1.4 (witness), Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +0.3 |
| 6 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 8; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +2.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 7 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 8; anxiety Ana +0.4, Marta +3.8, Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana takes 2.4 of functioning from the other |
| 7 | Bruno | CUTOFF | Ana | — | arrives week 8; anxiety Ana +0.9, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.07; the tie stays open) |
| 7 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | arrives week 8; anxiety Marta +0.8, Nadia +0.4, Pia +0.9 (witness), Ravi +2.7 (witness); Nadia takes 2.9 of functioning from the other |
| 7 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 8; anxiety Marta +1.5 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +1.4; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 7 | Pia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 7 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 8; anxiety Marta +2.9 (witness), Nadia +1.3 (witness), Pia +3.3, Ravi +1.0; Pia holds a position: no functioning changes hands |
| 7 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +3.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3 |
| 7 | Marta | SYMPTOM_ONSET | Ana, Dr Halim, Nadia, Pia, Ravi | — | arrives week 8; the nodal event re-opens the cut tie Ana–Bruno: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, physical channel; anxiety Ana +1.5, Dr Halim +0.5, Nadia +2.3, Pia +2.2, Ravi +3.3 |
| 8 | (from outside) | OWN_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +1.5, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); still weighing this week: anxiety Marta +0.9; still weighing this week: anxiety Marta +1.0; still weighing this week: anxiety Marta +0.9; still weighing this week: anxiety Marta +0.9; still weighing this week: anxiety Marta +0.9; still weighing this week: anxiety Marta +0.9; still weighing this week: anxiety Marta +0.9; still weighing this week: anxiety Marta +0.9; still weighing this week: anxiety Marta +0.9; still weighing this week: anxiety Marta +0.9; still weighing this week: anxiety Marta +0.9 |
| 8 | (from outside) | FAMILY_ILLNESS | Ana | — | a 12-week spell; anxiety Ana +0.5; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2; still weighing this week: anxiety Ana +0.2 |
| 8 | (from outside) | FAMILY_ILLNESS | Nadia | Marta, Ravi | a 12-week spell; anxiety Marta +0.3 (witness), Nadia +0.5, Ravi +0.4 (witness); still weighing this week: anxiety Nadia +0.3; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2; still weighing this week: anxiety Nadia +0.2 |
| 8 | (from outside) | FAMILY_ILLNESS | Pia | Marta, Ravi | a 12-week spell; anxiety Marta +0.3 (witness), Pia +0.5, Ravi +0.4 (witness); still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3; still weighing this week: anxiety Pia +0.3 |
| 8 | (from outside) | FAMILY_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +0.4 (witness), Nadia +0.3 (witness), Pia +0.3 (witness), Ravi +0.9; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5; still weighing this week: anxiety Ravi +0.5 |
| 8 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.6 (witness), Nadia +1.4 (witness), Pia +1.3 (witness), Ravi +4.0 |
| 8 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +3.2, Nadia +1.4 (witness), Pia +1.3 (witness), Ravi +2.0 (witness) |
| 8 | — | (system) | — | — | Nadia–Ravi: a positive contact repairs the tie; held anxiety -2.0 |
| 8 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 8 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.4, Marta +3.6, Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9 |
| 8 | Bruno | CUTOFF | Ana | — | arrives week 9; anxiety Ana +1.0, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 8 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Dr Halim +0.1, Marta +1.1 (witness), Nadia +0.6 (witness), Pia +0.6 (witness); calmer sender takes some anxiety: Dr Halim +0.8, Ravi -0.8 |
| 8 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +1.2, Marta +0.6, Nadia +1.1 (witness), Pia +1.1 (witness), Ravi +1.7 (witness); Ana takes 3.2 of functioning from the other |
| 8 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 9; anxiety Marta +2.9 (witness), Nadia +0.6, Pia +1.0 (witness), Ravi +4.0; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5; Nadia takes 2.8 of functioning from the other |
| 8 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 9; anxiety Marta +1.9, Nadia +1.0 (witness), Pia +0.6, Ravi +3.0 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; 13.6 of the sender's anxiety bound into Marta–Pia |
| 8 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 9; anxiety Marta +4.0 (witness), Nadia +1.4 (witness), Pia +2.0, Ravi +1.1; Pia holds a position: no functioning changes hands |
| 8 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 10; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3; 2.4 of the sender's anxiety bound into Ravi–Sofia |
| 9 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 10; anxiety Ana +0.3, Marta +3.4, Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +1.5 (witness); calmer sender takes some anxiety: Ana +1.0, Marta -1.0 |
| 9 | Bruno | CUTOFF | Ana | — | arrives week 10; anxiety Ana +0.8, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 9 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 10; anxiety Marta +1.1, Nadia +4.0, Pia +1.2 (witness), Ravi +3.9 (witness); Marta takes 0.9 of functioning from the other |
| 9 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 10; anxiety Marta +2.1 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.7; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6 |
| 9 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 10; anxiety Marta +3.1 (witness), Nadia +0.9 (witness), Pia +0.6, Ravi +1.8; calmer sender takes some anxiety: Pia +1.7, Ravi -1.7; 8.5 of the sender's anxiety bound into Pia–Ravi |
| 9 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 10; anxiety Marta +4.0 (witness), Nadia +1.2 (witness), Pia +2.0, Ravi +1.2; Pia left outside: anxiety Pia +10.3, Ravi -6.8 (of which 3.4 the outsider's own) |
| 9 | Sofia | PROVOKE | Ravi | Marta, Nadia, Pia | arrives week 11; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +3.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 10 | — | (system) | — | — | Nadia–Ravi: a positive contact repairs the tie; held anxiety -1.3 |
| 10 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 10 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 10 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.2 |
| 10 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.3 |
| 10 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 11; anxiety Marta +1.1, Nadia +1.0 (witness), Pia +4.7, Ravi +3.5 (witness); 25.7 of the sender's anxiety bound into Marta–Pia |
| 10 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 11; anxiety Marta +2.8 (witness), Nadia +0.6, Pia +0.9 (witness), Ravi +2.3; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Ravi left outside: anxiety Nadia -1.8, Ravi +2.6 (of which 0.9 the outsider's own) |
| 10 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 11; anxiety Marta +3.2 (witness), Nadia +1.0 (witness), Pia +0.6, Ravi +6.6; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1 |
| 10 | Ravi | STAY-IN-CONTACT | Marta | Nadia, Pia | arrives week 11; anxiety Marta +1.5, Nadia +1.1 (witness), Pia +1.2 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.5, Ravi +0.5 |
| 10 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 12; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.4, Sofia +1.4; Sofia rejects the other on Ravi–Sofia (rejection +0.55; the tie stays open) |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 11 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 12; anxiety Ana +0.3, Marta -0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 11 | Bruno | CUTOFF | Ana | — | arrives week 12; anxiety Ana +0.8, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 11 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 12; anxiety Marta +0.9, Nadia +1.0 (witness), Pia +3.0, Ravi +3.5 (witness); 8.3 of the sender's anxiety bound into Marta–Pia |
| 11 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 12; anxiety Marta +2.6 (witness), Nadia +0.6, Pia +0.8 (witness), Ravi +2.0; calmer sender takes some anxiety: Nadia +0.7, Ravi -0.7; 12.0 of the sender's anxiety bound into Nadia–Ravi |
| 11 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 11 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 12; anxiety Marta +6.8, Nadia +2.3 (witness), Pia +2.5 (witness), Ravi +1.2; 13.9 of the sender's anxiety bound into Marta–Ravi |
| 11 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 13; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 12 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.0 (witness), Nadia +1.2 (witness), Pia +1.3 (witness), Ravi +4.4 |
| 12 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.0, Nadia +1.2 (witness), Pia +1.3 (witness), Ravi +2.2 (witness) |
| 12 | — | (system) | — | — | Ana–Marta: a positive contact repairs the tie; held anxiety -3.5 |
| 12 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +0.3, Marta +1.3, Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana rejects the other on Ana–Marta (rejection +0.62; the tie stays open); impingement removed at once, anxiety Ana -4.3 |
| 12 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.4 |
| 12 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +4.2, Marta +0.6, Nadia +1.0 (witness), Pia +1.0 (witness), Ravi +1.6 (witness); Marta takes 1.0 of functioning from the other |
| 12 | Nadia | PREVENT_ALIGNMENT | Ravi | Marta, Pia | arrives week 13; anxiety Marta +1.8 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.2; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 12 | Pia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 12 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 13; anxiety Marta +3.4 (witness), Nadia +1.0 (witness), Pia +2.2, Ravi +1.1; Ravi rejects the other on Pia–Ravi (rejection +0.79; the tie stays open); impingement removed at once, anxiety Ravi -4.0 |
| 12 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 14; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; 2.8 of the sender's anxiety bound into Ravi–Sofia |
| 12 | Pia | SYMPTOM_ONSET | Marta, Ravi | Nadia | arrives week 13; the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, social channel; anxiety Marta +3.3, Nadia +0.7 (witness), Ravi +3.5 |
| 13 | (from outside) | OWN_ILLNESS | Pia | Marta, Ravi | a 12-week spell; anxiety Marta +0.9 (witness), Pia +1.2, Ravi +1.0 (witness); still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6; still weighing this week: anxiety Pia +0.6 |
| 13 | (from outside) | FAMILY_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +0.8, Nadia +0.2 (witness), Pia +0.3 (witness), Ravi +0.4 (witness); still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4 |
| 13 | (from outside) | FAMILY_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +0.4 (witness), Nadia +0.2 (witness), Pia +0.3 (witness), Ravi +0.8; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.3; still weighing this week: anxiety Ravi +0.3 |
| 13 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 13 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 13 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 13 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.11); the sender bears the refusal: anxiety Bruno +3.5 |
| 13 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 14; anxiety Marta +1.0, Nadia +2.0 (witness), Pia +2.1 (witness), Ravi +2.8; Ravi left outside: anxiety Marta -4.3, Ravi +6.4 (of which 2.1 the outsider's own) |
| 13 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 14; anxiety Marta +2.8 (witness), Nadia +0.6, Pia +0.9 (witness), Ravi +5.8; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; 4.6 of the sender's anxiety bound into Nadia–Ravi |
| 13 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 14; anxiety Marta +9.1, Nadia +1.2 (witness), Pia +0.6, Ravi +3.9 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; 7.1 of the sender's anxiety bound into Marta–Pia |
| 13 | Ravi | PREVENT_ALIGNMENT | Marta | Nadia, Pia | arrives week 14; anxiety Marta +1.8, Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1 |
| 13 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 15; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3; Ravi takes 0.7 of functioning from the other |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 14 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 14 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 14 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 14 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 15; anxiety Ana +0.3, Marta +1.0, Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Ana +1.0, Marta -1.0; Marta takes 1.0 of functioning from the other |
| 14 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 14 | Marta | PURSUE | Pia | Nadia, Ravi | arrives week 15; anxiety Marta +1.0, Nadia +1.2 (witness), Pia +2.8, Ravi +4.2 (witness) |
| 14 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Nadia +2.7; anxiety Pia +1.0 (witness), Ravi +3.3 (witness) |
| 14 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.14); the sender bears the refusal: anxiety Pia +3.1; anxiety Marta +2.7 (witness), Nadia +0.9 (witness) |
| 14 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.15); the sender bears the refusal: anxiety Ravi +3.5; anxiety Marta +1.5 (witness), Nadia +1.0 (witness), Pia +1.1 (witness) |
| 14 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 16; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +5.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.4, Sofia +1.4; 1.3 of the sender's anxiety bound into Ravi–Sofia |
| 15 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 15 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.3 |
| 15 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 15 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 16; anxiety Marta +1.0, Nadia +2.3 (witness), Pia +2.4 (witness), Ravi +8.6; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4; Ravi holds a position: no functioning changes hands |
| 15 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +2.8; anxiety Pia +1.1 (witness), Ravi +3.5 (witness) |
| 15 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Pia +3.2; anxiety Marta +2.6 (witness), Nadia +0.9 (witness) |
| 15 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 16; anxiety Marta +2.9 (witness), Nadia +1.0 (witness), Pia +1.7, Ravi +1.1; 32.7 of the sender's anxiety bound into Pia–Ravi |
| 15 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 17; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +4.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; 1.3 of the sender's anxiety bound into Ravi–Sofia |
| 16 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.9 (witness), Nadia +1.2 (witness), Pia +1.3 (witness), Ravi +4.1 |
| 16 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +3.7, Nadia +1.2 (witness), Pia +1.3 (witness), Ravi +2.1 (witness) |
| 16 | Ana | CUTOFF | Bruno | — | arrives week 17; anxiety Ana +0.5, Bruno +0.8; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 16 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 16 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 17; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Dr Halim +0.8, Marta -0.8 |
| 16 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 17; anxiety Marta +1.0, Nadia +2.7 (witness), Pia +2.9 (witness), Ravi +3.0; 11.5 of the sender's anxiety bound into Marta–Ravi |
| 16 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 17; anxiety Marta +2.7 (witness), Nadia +0.6, Pia +1.0 (witness), Ravi +2.4; Ravi left outside: anxiety Nadia -2.1, Ravi +3.2 (of which 1.1 the outsider's own) |
| 16 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Pia +3.4; anxiety Marta +3.3 (witness), Nadia +1.1 (witness) |
| 16 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.0; anxiety Marta +1.5 (witness), Nadia +1.0 (witness), Pia +1.1 (witness) |
| 16 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 18; anxiety Marta +1.0 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3 |
| 16 | Bruno | SYMPTOM_ONSET | Ana | — | arrives week 17; the nodal event re-opens the cut tie Ana–Bruno: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); symptom onset, social channel; anxiety Ana +1.9 |
| 17 | (from outside) | OWN_ILLNESS | Bruno | — | a 12-week spell; anxiety Bruno +1.9; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0 |
| 17 | (from outside) | FAMILY_ILLNESS | Ana | — | a 12-week spell; anxiety Ana +0.6; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3 |
| 17 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.5 |
| 17 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 17 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 17 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 17 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 17 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 18; anxiety Ana +0.3, Marta +0.5, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 17 | Bruno | CUTOFF | Ana | — | arrives week 18; anxiety Ana +0.5, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Bruno -0.4 |
| 17 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 18; anxiety Marta +0.9, Nadia +3.4, Pia +1.2 (witness), Ravi +3.6 (witness); calmer sender takes some anxiety: Marta +0.0, Nadia -0.0 |
| 17 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.0; anxiety Pia +1.1 (witness), Ravi +3.3 (witness) |
| 17 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 17 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 18; anxiety Marta +2.9 (witness), Nadia +2.4, Pia +1.0 (witness), Ravi +1.1; 22.6 of the sender's anxiety bound into Nadia–Ravi |
| 17 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 19; anxiety Marta +1.2 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +1.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 18 | — | (system) | — | — | Ana–Marta: an accepted contact bridges the cutoff; held anxiety -5.3 |
| 18 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 18 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 18 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.3 |
| 18 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.7 |
| 18 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 19; anxiety Marta +0.9, Nadia +1.2 (witness), Pia +3.0, Ravi +3.7 (witness); 17.6 of the sender's anxiety bound into Marta–Pia |
| 18 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.1; anxiety Pia +1.1 (witness), Ravi +3.1 (witness) |
| 18 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 19; anxiety Marta +7.7, Nadia +1.2 (witness), Pia +0.7, Ravi +3.7 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Pia takes 3.0 of functioning from the other |
| 18 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.7; anxiety Marta +1.6 (witness), Nadia +1.0 (witness), Pia +1.0 (witness) |
| 18 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 19 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 20; anxiety Ana +0.4, Marta +1.2, Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Ana +0.4, Marta -0.4 |
| 19 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.7 |
| 19 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 20; anxiety Marta +0.9, Nadia +1.1 (witness), Pia +3.2, Ravi +3.4 (witness); 6.9 of the sender's anxiety bound into Marta–Pia |
| 19 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 19 | Pia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 19 | Ravi | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 19 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 19 | Nadia | SYMPTOM_ONSET | Marta, Ravi | Pia | arrives week 20; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, mental channel; anxiety Marta +3.4, Pia +0.7 (witness), Ravi +3.2 |
| 20 | (from outside) | OWN_ILLNESS | Nadia | Marta, Ravi | a 12-week spell; anxiety Marta +1.0 (witness), Nadia +1.2, Ravi +0.9 (witness); still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6; still weighing this week: anxiety Nadia +0.6 |
| 20 | (from outside) | FAMILY_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +0.8, Nadia +0.2 (witness), Pia +0.2 (witness), Ravi +0.4 (witness); still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.5 |
| 20 | (from outside) | FAMILY_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +0.4 (witness), Nadia +0.2 (witness), Pia +0.2 (witness), Ravi +0.8; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.4; still weighing this week: anxiety Ravi +0.3; still weighing this week: anxiety Ravi +0.3; still weighing this week: anxiety Ravi +0.3; still weighing this week: anxiety Ravi +0.3; still weighing this week: anxiety Ravi +0.3; still weighing this week: anxiety Ravi +0.3; still weighing this week: anxiety Ravi +0.3; still weighing this week: anxiety Ravi +0.3; still weighing this week: anxiety Ravi +0.3 |
| 20 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.1 (witness), Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +3.9 |
| 20 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.1, Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +1.9 (witness) |
| 20 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.2 |
| 20 | Ana | CUTOFF | Bruno | — | arrives week 21; anxiety Ana +0.5, Bruno +0.5; calmer sender takes some anxiety: Ana +1.2, Bruno -1.2; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 20 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.8 |
| 20 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 21; anxiety Marta +1.0, Nadia +1.2 (witness), Pia +1.5, Ravi +3.5 (witness); 5.7 of the sender's anxiety bound into Marta–Pia |
| 20 | Nadia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 20 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 21; anxiety Marta +1.2, Nadia +1.2 (witness), Pia +0.6, Ravi +3.7 (witness); calmer sender takes some anxiety: Marta -0.3, Pia +0.3; Marta takes 0.8 of functioning from the other |
| 20 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.0; anxiety Marta +1.9 (witness), Nadia +1.2 (witness), Pia +1.2 (witness) |
| 20 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 22; anxiety Marta +1.3 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; Ravi takes 1.2 of functioning from the other |
| 21 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 21 | Ana | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 21 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.8 |
| 21 | Marta | I-POSITION | — | — | starts a sequence in PREPARE: nothing said this week |
| 21 | — | (system) | — | — | Marta's I-POSITION sequence: begin prepare |
| 21 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.1; anxiety Pia +1.1 (witness), Ravi +3.2 (witness) |
| 21 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 22; anxiety Marta +1.2, Nadia +1.1 (witness), Pia +0.6, Ravi +3.1 (witness); calmer sender takes some anxiety: Marta -0.2, Pia +0.2; Marta takes 1.0 of functioning from the other |
| 21 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +2.9; anxiety Marta +1.8 (witness), Nadia +1.2 (witness), Pia +1.2 (witness) |
| 21 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 23; anxiety Marta +1.2 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; 2.4 of the sender's anxiety bound into Ravi–Sofia |
| 22 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.2 |
| 22 | — | (system) | — | — | Marta's I-POSITION sequence: end prepare failed |
| 22 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 23; anxiety Ana +0.3, Marta +2.7, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; 3.9 of the sender's anxiety bound into Ana–Marta |
| 22 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.9 |
| 22 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 23; anxiety Marta +0.9, Nadia +1.1 (witness), Pia +2.9, Ravi +3.0 (witness); 6.7 of the sender's anxiety bound into Marta–Pia |
| 22 | Nadia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 22 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 23; anxiety Marta +5.2, Nadia +1.2 (witness), Pia +0.6, Ravi +3.2 (witness); calmer sender takes some anxiety: Marta -0.3, Pia +0.3; Pia takes 2.9 of functioning from the other |
| 22 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 23; anxiety Dr Halim +0.3, Marta +1.9 (witness), Nadia +1.0 (witness), Pia +1.0 (witness), Ravi +0.3 |
| 22 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 24; anxiety Marta +1.4 (witness), Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +4.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; 1.0 of the sender's anxiety bound into Ravi–Sofia |
| 23 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 24; anxiety Ana +0.3, Marta -0.9, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 23 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.9 |
| 23 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 24; anxiety Marta +1.0, Nadia +1.0 (witness), Pia +2.2, Ravi +2.8 (witness); 14.1 of the sender's anxiety bound into Marta–Pia |
| 23 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.1; anxiety Pia +1.0 (witness), Ravi +2.8 (witness) |
| 23 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Pia +3.0; anxiety Marta +3.2 (witness), Nadia +1.0 (witness) |
| 23 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 24; anxiety Marta +0.5, Nadia +2.5 (witness), Pia +2.3 (witness), Ravi +0.9; Marta left outside: anxiety Marta +10.0, Ravi -6.7 (of which 3.3 the outsider's own) |
| 23 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 25; anxiety Marta +1.3 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +2.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 24 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.1 (witness), Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +3.5 |
| 24 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.1, Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +1.8 (witness) |
| 24 | — | (system) | — | — | Ana–Marta: an accepted contact bridges the cutoff; held anxiety -4.1 |
| 24 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +3.5 |
| 24 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 24 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 24 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 24 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 24 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 24 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 24 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.11); the sender bears the refusal: anxiety Ana +2.2 |
| 24 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.9 |
| 24 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 25; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 24 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 25; anxiety Marta +1.0, Nadia +1.1 (witness), Pia +2.4, Ravi +2.8 (witness); 16.0 of the sender's anxiety bound into Marta–Pia |
| 24 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.3; anxiety Pia +1.0 (witness), Ravi +2.7 (witness) |
| 24 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 25; anxiety Marta +5.2, Nadia +1.1 (witness), Pia +0.6, Ravi +3.0 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4; Pia takes 3.1 of functioning from the other |
| 24 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 25; anxiety Marta +4.4, Nadia +2.3 (witness), Pia +2.1 (witness), Ravi +0.9; 22.5 of the sender's anxiety bound into Marta–Ravi |
| 24 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 26; anxiety Marta +1.5 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +2.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 1.1 of the sender's anxiety bound into Ravi–Sofia |
| 25 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 26; anxiety Ana +0.3, Marta +0.9, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 25 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 25 | Marta | WITHHOLD | Dr Halim | — | held back SPLIT: computed, not emitted; attention on the tie rises |
| 25 | Nadia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 25 | Pia | PROVOKE | Marta | Nadia, Ravi | arrives week 26; anxiety Marta +3.4, Nadia +0.8 (witness), Pia +0.4, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -0.0, Pia +0.0 |
| 25 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.14); the sender bears the refusal: anxiety Ravi +2.4; anxiety Marta +1.9 (witness), Nadia +1.1 (witness), Pia +1.0 (witness) |
| 25 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 27; anxiety Marta +1.5 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +1.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; 1.0 of the sender's anxiety bound into Ravi–Sofia |
| 26 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 27; anxiety Ana +0.3, Marta +3.3, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 26 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.8 |
| 26 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 27; anxiety Marta +1.0, Nadia +3.2, Pia +0.9 (witness), Ravi +2.6 (witness); calmer sender takes some anxiety: Marta +0.6, Nadia -0.6 |
| 26 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.11); the sender bears the refusal: anxiety Nadia +3.2; anxiety Pia +1.0 (witness), Ravi +2.8 (witness) |
| 26 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 26 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 27; anxiety Dr Halim +0.3, Marta +2.0 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +0.2 |
| 26 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 28; anxiety Marta +1.6 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +2.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; 1.0 of the sender's anxiety bound into Ravi–Sofia |
| 27 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 27 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 27 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 28; anxiety Ana +0.3, Marta +3.3, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 27 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.9 |
| 27 | Marta | WITHHOLD | Nadia | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 27 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 28; anxiety Marta +3.1 (witness), Nadia +0.7, Pia +0.8 (witness), Ravi +0.9; Nadia rejects the other on Nadia–Ravi (rejection +0.75; the tie stays open) |
| 27 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.14); the sender bears the refusal: anxiety Pia +2.8; anxiety Marta +3.5 (witness), Nadia +1.1 (witness) |
| 27 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.15); the sender bears the refusal: anxiety Ravi +2.5; anxiety Marta +2.0 (witness), Nadia +1.2 (witness), Pia +1.1 (witness) |
| 27 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 28 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.2 (witness), Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +3.3 |
| 28 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.5, Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +1.7 (witness) |
| 28 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 28 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 28 | Ana | CUTOFF | Bruno | — | arrives week 29; anxiety Ana +0.5, Bruno +0.5; calmer sender takes some anxiety: Ana +1.3, Bruno -1.3; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 28 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.9 |
| 28 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 29; anxiety Marta +1.2, Nadia +2.1 (witness), Pia +1.9 (witness), Ravi +1.5; Marta rejects the other on Marta–Ravi (rejection +0.80; the tie stays open); impingement removed at once, anxiety Marta -0.6 |
| 28 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.3; anxiety Pia +1.1 (witness), Ravi +3.1 (witness) |
| 28 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Pia +2.9; anxiety Marta +3.9 (witness), Nadia +1.2 (witness) |
| 28 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 30; anxiety Marta +1.8 (witness), Nadia +1.2 (witness), Pia +1.0 (witness), Ravi +0.5, Sofia +0.2; Ravi rejects the other on Ravi–Sofia (rejection +0.76; the tie stays open); impingement removed at once, anxiety Ravi -3.4 |
| 28 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 29 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.5 |
| 29 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 29 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.12); the sender bears the refusal: anxiety Ana +2.4 |
| 29 | Bruno | CUTOFF | Ana | — | arrives week 30; anxiety Ana +0.3, Bruno +0.8; Bruno rejects the other on Ana–Bruno (rejection +0.32; the tie stays open); impingement removed at once, anxiety Bruno -0.2 |
| 29 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 30; anxiety Marta +1.1, Nadia +2.0 (witness), Pia +1.8 (witness), Ravi +1.7; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2; Ravi takes 2.6 of functioning from the other |
| 29 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 30; anxiety Marta +1.5, Nadia +0.7, Pia +1.0 (witness), Ravi +2.8 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.75; the tie stays open); impingement removed at once, anxiety Nadia -1.0 |
| 29 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 29 | Ravi | TRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Ravi +4.3; anxiety Nadia +2.0 (witness), Pia +1.7 (witness) |
| 29 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 30 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 30 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 30 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 30 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 30 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 30 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 30 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 30 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 30 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.4 |
| 30 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | arrives week 31; anxiety Ana +0.4, Marta +0.7, Nadia +1.1 (witness), Pia +1.0 (witness), Ravi +1.3 (witness) |
| 30 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 31; anxiety Marta +3.0 (witness), Nadia +0.7, Pia +0.8 (witness), Ravi +1.2; Ravi takes 2.9 of functioning from the other |
| 30 | Pia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 30 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.15); the sender bears the refusal: anxiety Ravi +2.2; anxiety Marta +2.1 (witness), Nadia +1.3 (witness), Pia +1.1 (witness) |
| 30 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.5; anxiety Marta +1.5 (witness), Nadia +0.9 (witness), Pia +0.8 (witness) |
| 31 | Ana | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 31 | Bruno | CUTOFF | Ana | — | arrives week 32; anxiety Ana +0.6, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 31 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 32; anxiety Marta +1.1, Nadia +2.4 (witness), Pia +2.0 (witness), Ravi +5.8; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; 9.1 of the sender's anxiety bound into Marta–Ravi |
| 31 | Nadia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 31 | Pia | STAY-IN-CONTACT | Marta | Nadia, Ravi | arrives week 32; anxiety Marta +1.4, Nadia +0.8 (witness), Pia +0.4, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7 |
| 31 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Ravi +3.6; anxiety Marta +3.0 (witness), Pia +0.8 (witness) |
| 31 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.5; anxiety Marta +1.5 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 32 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.3 (witness), Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +2.9 |
| 32 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.6, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +1.4 (witness) |
| 32 | — | (system) | — | — | Marta–Pia: a positive contact repairs the tie; held anxiety -40.6 |
| 32 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.5 |
| 32 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 32 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 32 | Ana | CUTOFF | Bruno | — | arrives week 33; anxiety Ana +0.5, Bruno +0.3; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.20; the tie stays open); impingement removed at once, anxiety Ana -0.9 |
| 32 | Bruno | CUTOFF | Ana | — | arrives week 33; anxiety Ana +0.6, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Bruno -0.3 |
| 32 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 33; anxiety Dr Halim +0.1, Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.51; the tie stays open); impingement removed at once, anxiety Dr Halim -0.2 |
| 32 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 33; anxiety Dr Halim +0.2, Marta +0.3, Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +1.2 (witness) |
| 32 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 33; anxiety Marta +3.2 (witness), Nadia +0.7, Pia +0.8 (witness), Ravi +4.4; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2 |
| 32 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 33; anxiety Marta +7.1, Nadia +1.2 (witness), Pia +0.6, Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7 |
| 32 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 33; anxiety Marta +1.6, Nadia +2.5 (witness), Pia +2.0 (witness), Ravi +0.8; Ravi rejects the other on Marta–Ravi (rejection +0.81; the tie stays open); impingement removed at once, anxiety Ravi -4.4 |
| 32 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.4 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 33 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.5 |
| 33 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 33 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 33 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 33 | Ana | CUTOFF | Bruno | — | arrives week 34; anxiety Ana +0.5, Bruno +0.2; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 33 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 33 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | arrives week 34; anxiety Ana +3.2, Marta +0.7, Nadia +1.3 (witness), Pia +1.1 (witness), Ravi +1.3 (witness) |
| 33 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.6; anxiety Pia +0.9 (witness), Ravi +2.2 (witness) |
| 33 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 34; anxiety Marta +2.0, Nadia +1.1 (witness), Pia +0.6, Ravi +2.1 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; Marta takes 0.3 of functioning from the other |
| 33 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 34; anxiety Dr Halim +0.2, Marta +2.3 (witness), Nadia +1.2 (witness), Pia +1.0 (witness), Ravi +0.2 |
| 33 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.3 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 34 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 34 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 35; anxiety Ana +0.4, Marta +0.6, Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Ana rejects the other on Ana–Marta (rejection +0.62; the tie stays open); impingement removed at once, anxiety Ana -3.6 |
| 34 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 34 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 35; anxiety Ana +1.1, Marta +0.7, Nadia +1.2 (witness), Pia +1.0 (witness), Ravi +1.2 (witness); Marta rejects the other on Ana–Marta (rejection +0.77; the tie stays open); impingement removed at once, anxiety Marta -0.6 |
| 34 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 34 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 35; anxiety Marta +5.3, Nadia +1.2 (witness), Pia +0.6, Ravi +2.5 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; 6.1 of the sender's anxiety bound into Marta–Pia |
| 34 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Ravi +3.7; anxiety Marta +3.2 (witness), Pia +0.9 (witness) |
| 34 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.4 (witness), Nadia +0.9 (witness), Pia +0.8 (witness) |
| 35 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +6.1 |
| 35 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 35 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 35 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 35 | Ana | REDUCE_CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 36; anxiety Ana +0.3, Marta +0.4, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9; Ana's rejection -0.25; Marta's rejection -0.25 |
| 35 | Bruno | CUTOFF | Ana | — | arrives week 36; anxiety Ana +0.2, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.15; the tie stays open) |
| 35 | Marta | TRIANGLE | Pia | Nadia, Ravi | arrives week 36; anxiety Marta +1.1, Nadia +1.2 (witness), Pia -0.1, Ravi +2.4 (witness); Pia left outside: anxiety Marta -4.4, Pia +6.7 (of which 2.2 the outsider's own) |
| 35 | Nadia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 35 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.14); the sender bears the refusal: anxiety Pia +2.7; anxiety Marta +3.8 (witness), Nadia +1.3 (witness) |
| 35 | Ravi | WITHHOLD | Sofia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 35 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.5 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 36 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.2 (witness), Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +2.9 |
| 36 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.4, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +1.4 (witness) |
| 36 | — | (system) | — | — | Ana–Marta: an accepted contact bridges the cutoff; held anxiety -5.5 |
| 36 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +3.5 |
| 36 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 36 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 36 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 36 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.4 |
| 36 | Bruno | CUTOFF | Ana | — | arrives week 37; anxiety Ana +0.7, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 36 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Marta +5.6; anxiety Pia +0.9 (witness), Ravi +2.2 (witness) |
| 36 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.6; anxiety Pia +0.9 (witness), Ravi +2.1 (witness) |
| 36 | Pia | DETRIANGLE | Marta | Nadia, Ravi | arrives week 37; anxiety Marta +1.7, Nadia +0.7 (witness), Pia +0.4, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -1.1, Pia +1.1 |
| 36 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Ravi +3.7; anxiety Marta +3.1 (witness), Pia +0.8 (witness) |
| 36 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.4 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 37 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.7 |
| 37 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 37 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 37 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 37 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.4 |
| 37 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 38; anxiety Dr Halim +0.2, Marta +0.3, Nadia +1.1 (witness), Pia +0.9 (witness), Ravi +1.4 (witness) |
| 37 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.6; anxiety Pia +0.9 (witness), Ravi +2.2 (witness) |
| 37 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 38; anxiety Marta +2.3, Nadia +1.0 (witness), Pia +0.6, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9; Pia rejects the other on Marta–Pia (rejection +0.73; the tie stays open); impingement removed at once, anxiety Pia -0.1 |
| 37 | Ravi | WITHHOLD | Pia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 37 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 38 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 38 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 38 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.08); the sender bears the refusal: anxiety Bruno +3.5 |
| 38 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.15); the sender bears the refusal: anxiety Marta +5.5; anxiety Nadia +1.0 (witness), Ravi +2.0 (witness) |
| 38 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.06); the sender bears the refusal: anxiety Nadia +3.6; anxiety Pia +1.0 (witness), Ravi +2.3 (witness) |
| 38 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 39; anxiety Marta +3.6 (witness), Nadia +1.2 (witness), Pia +0.6, Ravi +0.7; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; Pia rejects the other on Pia–Ravi (rejection +0.72; the tie stays open); impingement removed at once, anxiety Pia -0.1 |
| 38 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 39; anxiety Marta +3.9 (witness), Nadia +1.3 (witness), Pia +0.3, Ravi +0.8; 17.4 of the sender's anxiety bound into Pia–Ravi |
| 38 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.4 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 39 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 39 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 39 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 39 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Marta +5.6; anxiety Pia +1.1 (witness), Ravi +2.6 (witness) |
| 39 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +3.6; anxiety Pia +1.0 (witness), Ravi +2.4 (witness) |
| 39 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.14); the sender bears the refusal: anxiety Pia +2.8; anxiety Marta +3.7 (witness), Nadia +1.3 (witness) |
| 39 | Ravi | WITHHOLD | Dr Halim | — | held back DISPLACE: computed, not emitted; attention on the tie rises |
| 39 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 40 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.1 (witness), Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +2.9 |
| 40 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.3, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +1.4 (witness) |
| 40 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.6 |
| 40 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 40 | Ana | CUTOFF | Bruno | — | arrives week 41; anxiety Ana +0.5, Bruno +0.1; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 40 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 40 | Dr Halim | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 41; anxiety Dr Halim +0.1, Marta +1.3 (witness), Nadia +0.7 (witness), Pia +0.6 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Ravi -0.6; Dr Halim rejects the other on Dr Halim–Ravi (rejection +0.51; the tie stays open) |
| 40 | Marta | PURSUE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.17); the sender bears the refusal: anxiety Marta +5.9; anxiety Nadia +2.7 (witness), Pia +2.2 (witness) |
| 40 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 41; anxiety Marta +1.4, Nadia +0.7, Pia +0.9 (witness), Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -0.6, Nadia +0.6; Nadia rejects the other on Marta–Nadia (rejection +0.49; the tie stays open) |
| 40 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 41; anxiety Marta +8.2, Nadia +1.2 (witness), Pia +0.6, Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -1.5, Pia +1.5; 6.6 of the sender's anxiety bound into Marta–Pia |
| 40 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.14); the sender bears the refusal: anxiety Ravi +3.4; anxiety Marta +3.3 (witness), Nadia +1.1 (witness) |
| 40 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 41 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +3.8 |
| 41 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.6 |
| 41 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 41 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 41 | Ana | REDUCE_CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 42; anxiety Ana +0.3, Marta +0.6, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +1.6, Marta -1.6; Ana's rejection -0.02; Marta's rejection -0.17 |
| 41 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 41 | Marta | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 41 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 42; anxiety Marta +3.3 (witness), Nadia +0.7, Pia +1.0 (witness), Ravi +4.4; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6 |
| 41 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 42; anxiety Marta +2.3, Nadia +1.0 (witness), Pia +0.6, Ravi +2.1 (witness); calmer sender takes some anxiety: Marta -2.4, Pia +2.4; Marta takes 0.7 of functioning from the other |
| 41 | Ravi | CONFLICT | Nadia | Marta, Pia | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Ravi +3.6; anxiety Marta +3.3 (witness), Pia +0.9 (witness) |
| 41 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 43; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -1.5, Sofia +1.5; Sofia rejects the other on Ravi–Sofia (rejection +0.56; the tie stays open) |
| 41 | — | (system) | — | — | Pia's symptom gives back 38.9 (Marta +19.5, Ravi +19.5) |
| 42 | — | (system) | — | — | Ana–Marta: an accepted contact bridges the cutoff; held anxiety -1.0 |
| 42 | — | (system) | — | — | Ana–Marta tie now ordinary |
| 42 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.8 |
| 42 | Ana | CUTOFF | Bruno | — | arrives week 43; anxiety Ana +0.5, Bruno +0.4; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 42 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.7 |
| 42 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 43; anxiety Dr Halim +0.4, Marta +0.4, Nadia +1.3 (witness), Pia +1.1 (witness), Ravi +1.7 (witness) |
| 42 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.5; anxiety Pia +1.0 (witness), Ravi +2.3 (witness) |
| 42 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 42 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.17); the sender bears the refusal: anxiety Ravi +4.1; anxiety Marta +3.5 (witness), Nadia +1.3 (witness) |
| 42 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 43 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.3 |
| 43 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 43 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 44; anxiety Ana +0.3, Marta +0.3, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +1.1, Marta -1.1 |
| 43 | Bruno | CUTOFF | Ana | — | arrives week 44; anxiety Ana +0.1, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open); impingement removed at once, anxiety Bruno -0.1 |
| 43 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 44; anxiety Ana +0.2, Marta +0.7, Nadia +1.1 (witness), Pia +1.0 (witness), Ravi +1.2 (witness); Marta rejects the other on Ana–Marta (rejection +0.84; the tie stays open); impingement removed at once, anxiety Marta -0.3 |
| 43 | Nadia | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 43 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 44; anxiety Marta +1.8, Nadia +1.0 (witness), Pia +0.6, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -1.4, Pia +1.4; Marta takes 0.8 of functioning from the other |
| 43 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 44; anxiety Marta +6.2, Nadia +2.7 (witness), Pia +2.3 (witness), Ravi +0.8; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1 |
| 43 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.3 (witness), Nadia +0.9 (witness), Pia +0.8 (witness) |
| 44 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.0 (witness), Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +2.9 |
| 44 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.1, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +1.5 (witness) |
| 44 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 44 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 44 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.6 |
| 44 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.17); the sender bears the refusal: anxiety Marta +5.5; anxiety Pia +1.3 (witness), Ravi +3.0 (witness) |
| 44 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 45; anxiety Marta +2.9 (witness), Nadia +0.7, Pia +0.9 (witness), Ravi +1.3; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Ravi left outside: anxiety Nadia -2.2, Ravi +3.3 (of which 1.1 the outsider's own) |
| 44 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 44 | Ravi | WITHHOLD | Pia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 44 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.3 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 45 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 45 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 45 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 45 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 45 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.6 (witness) |
| 45 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 45 | Marta | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 45 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 46; anxiety Marta +2.7 (witness), Nadia +0.7, Pia +0.8 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6; Ravi takes 2.9 of functioning from the other |
| 45 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 46; anxiety Marta +1.5, Nadia +1.0 (witness), Pia +0.6, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -1.3, Pia +1.3; 11.5 of the sender's anxiety bound into Marta–Pia |
| 45 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.17); the sender bears the refusal: anxiety Ravi +4.0; anxiety Marta +3.1 (witness), Nadia +1.2 (witness) |
| 45 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.4 (witness), Nadia +1.1 (witness), Pia +0.9 (witness) |
| 45 | — | (system) | — | — | Marta's symptom gives back 0.3 (Ana +0.1, Nadia +0.1, Pia +0.1, Ravi +0.1) |
| 45 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -0.0, Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 46 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 46 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.9 |
| 46 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 46 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.12); the sender bears the refusal: anxiety Ana +2.5 |
| 46 | Bruno | CUTOFF | Ana | — | arrives week 47; anxiety Ana +0.4, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.03; the tie stays open) |
| 46 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.16); the sender bears the refusal: anxiety Marta +5.3; anxiety Nadia +1.2 (witness), Ravi +2.0 (witness) |
| 46 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.8; anxiety Pia +1.2 (witness), Ravi +2.6 (witness) |
| 46 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 47; anxiety Marta +2.8 (witness), Nadia +1.1 (witness), Pia +0.5, Ravi +5.9; calmer sender takes some anxiety: Pia +1.7, Ravi -1.7; 3.5 of the sender's anxiety bound into Pia–Ravi |
| 46 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Ravi +3.7; anxiety Marta +3.5 (witness), Pia +1.1 (witness) |
| 46 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.06); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.2 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 47 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +0.9 |
| 47 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.8 |
| 47 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 47 | Ana | CUTOFF | Bruno | — | arrives week 48; anxiety Ana +0.5, Bruno +0.1; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ana -0.1 |
| 47 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.6 |
| 47 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 48; anxiety Dr Halim +0.3, Marta +0.3, Nadia +1.3 (witness), Pia +1.1 (witness), Ravi +1.5 (witness) |
| 47 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.11); the sender bears the refusal: anxiety Nadia +3.8; anxiety Pia +1.1 (witness), Ravi +2.4 (witness) |
| 47 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 48; anxiety Marta +2.5 (witness), Nadia +1.0 (witness), Pia +0.5, Ravi +1.0; calmer sender takes some anxiety: Pia +1.7, Ravi -1.7; Ravi holds a position: no functioning changes hands |
| 47 | Ravi | WITHHOLD | Pia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 47 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 48 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.9 (witness), Nadia +1.6 (witness), Pia +1.2 (witness), Ravi +2.8 |
| 48 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +3.9, Nadia +1.6 (witness), Pia +1.2 (witness), Ravi +1.4 (witness) |
| 48 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 48 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 48 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +0.9 (witness) |
| 48 | Bruno | CUTOFF | Ana | — | arrives week 49; anxiety Ana +0.2, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 48 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 49; anxiety Dr Halim +0.1, Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.8, Marta -0.8 |
| 48 | Marta | WITHHOLD | Pia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 48 | Nadia | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 48 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 49; anxiety Marta +1.4, Nadia +0.9 (witness), Pia +0.6, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -1.6, Pia +1.6; 8.7 of the sender's anxiety bound into Marta–Pia |
| 48 | Ravi | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 48 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 49 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 49 | — | (system) | — | — | Dr Halim–Marta tie now ordinary |
| 49 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.6 |
| 49 | Bruno | WITHHOLD | Ana | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 49 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.16); the sender bears the refusal: anxiety Marta +5.0; anxiety Nadia +1.1 (witness), Ravi +1.9 (witness) |
| 49 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.9; anxiety Pia +1.2 (witness), Ravi +2.6 (witness) |
| 49 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 49 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 50; anxiety Marta +6.3, Nadia +3.0 (witness), Pia +2.3 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.2, Ravi +0.2 |
| 49 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.3 (witness), Nadia +1.1 (witness), Pia +0.9 (witness) |
| 50 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +0.9 |
| 50 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 51; anxiety Ana +0.4, Marta +0.2, Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9; Ana rejects the other on Ana–Marta (rejection +0.62; the tie stays open) |
| 50 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.4 |
| 50 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Marta +5.1; anxiety Pia +1.4 (witness), Ravi +3.1 (witness) |
| 50 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 50 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 50 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 51; anxiety Dr Halim +0.1, Marta +2.0 (witness), Nadia +1.2 (witness), Pia +1.0 (witness), Ravi +0.2 |
| 50 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.09); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.2 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 50 | — | (system) | — | — | Bruno's symptom gives back 7.9 (Ana +7.9) |
| 51 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 51 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.4 |
| 51 | Ana | CUTOFF | Bruno | — | arrives week 52; anxiety Ana +0.5, Bruno +0.5; calmer sender takes some anxiety: Ana +0.7, Bruno -0.7; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 51 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.08); the sender bears the refusal: anxiety Bruno +3.4 |
| 51 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 52; anxiety Marta +1.0, Nadia +2.6 (witness), Pia +2.1 (witness), Ravi +0.7; Ravi takes 3.2 of functioning from the other |
| 51 | Nadia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 51 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 52; anxiety Marta +2.4 (witness), Nadia +1.0 (witness), Pia +0.5, Ravi +3.5; calmer sender takes some anxiety: Pia +1.4, Ravi -1.4 |
| 51 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.15); the sender bears the refusal: anxiety Ravi +3.5; anxiety Marta +2.9 (witness), Nadia +1.2 (witness) |
| 51 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 51 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |
| 52 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.9 (witness), Nadia +1.6 (witness), Pia +1.2 (witness), Ravi +2.8 |
| 52 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +3.8, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +1.4 (witness) |
| 52 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +3.3 |
| 52 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 52 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ana +2.0; anxiety Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +0.9 (witness) |
| 52 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 52 | Marta | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 52 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 53; anxiety Marta +1.0, Nadia +0.8, Pia +1.1 (witness), Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -0.3, Nadia +0.3; Nadia rejects the other on Marta–Nadia (rejection +0.27; the tie stays open) |
| 52 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 53; anxiety Marta +1.2, Nadia +0.9 (witness), Pia +0.6, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -1.5, Pia +1.5; Marta takes 0.7 of functioning from the other |
| 52 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.16); the sender bears the refusal: anxiety Ravi +2.0; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.1 (witness) |
| 52 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.2 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 53 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 53 | Ana | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 53 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 53 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.16); the sender bears the refusal: anxiety Marta +5.3; anxiety Nadia +1.3 (witness), Ravi +2.1 (witness) |
| 53 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 54; anxiety Marta +3.3 (witness), Nadia +0.8, Pia +1.0 (witness), Ravi +3.5; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 53 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 54; anxiety Marta +2.6 (witness), Nadia +1.0 (witness), Pia +0.6, Ravi +0.7; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; 9.8 of the sender's anxiety bound into Pia–Ravi |
| 53 | Ravi | WITHHOLD | Sofia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 53 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.3 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 54 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +3.7 |
| 54 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 54 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +1.1 (witness), Pia +0.9 (witness), Ravi +0.9 (witness) |
| 54 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 54 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 55; anxiety Dr Halim +0.3, Marta +0.3, Nadia +1.3 (witness), Pia +1.0 (witness), Ravi +1.4 (witness) |
| 54 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.7; anxiety Pia +1.0 (witness), Ravi +2.1 (witness) |
| 54 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 55; anxiety Marta +2.8 (witness), Nadia +1.1 (witness), Pia +0.5, Ravi +4.5; calmer sender takes some anxiety: Pia +1.4, Ravi -1.4; 3.7 of the sender's anxiety bound into Pia–Ravi |
| 54 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 55; anxiety Marta +1.2, Nadia +2.4 (witness), Pia +2.0 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3; 19.7 of the sender's anxiety bound into Marta–Ravi |
| 54 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.3 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 55 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 55 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 55 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 55 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 55 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +0.8 (witness) |
| 55 | Bruno | CUTOFF | Ana | — | arrives week 56; anxiety Ana +0.4, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.30; the tie stays open) |
| 55 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 56; anxiety Ana +0.3, Marta +0.6, Nadia +1.3 (witness), Pia +1.1 (witness), Ravi +1.1 (witness); Marta rejects the other on Ana–Marta (rejection +0.04; the tie stays open) |
| 55 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 55 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 56; anxiety Marta +1.2, Nadia +1.0 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -1.5, Pia +1.5; Pia rejects the other on Marta–Pia (rejection +0.54; the tie stays open) |
| 55 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.14); the sender bears the refusal: anxiety Ravi +2.9; anxiety Marta +2.8 (witness), Nadia +1.1 (witness) |
| 55 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 56 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.0 (witness), Nadia +1.6 (witness), Pia +1.2 (witness), Ravi +2.6 |
| 56 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.1, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +1.3 (witness) |
| 56 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 56 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +1.9 |
| 56 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 56 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 56 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +1.1 (witness), Pia +0.9 (witness), Ravi +0.9 (witness) |
| 56 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.5 |
| 56 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 57; anxiety Dr Halim +0.1, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.9, Marta -0.9; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 56 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +5.4; anxiety Nadia +1.2 (witness), Ravi +1.9 (witness) |
| 56 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.9; anxiety Pia +1.1 (witness), Ravi +2.2 (witness) |
| 56 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 57; anxiety Marta +1.2, Nadia +1.0 (witness), Pia +0.6, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -1.7, Pia +1.7; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 56 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 57; anxiety Marta +8.9, Nadia +2.6 (witness), Pia +2.1 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -1.0, Ravi +1.0; 7.8 of the sender's anxiety bound into Marta–Ravi |
| 56 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.4 (witness), Nadia +1.1 (witness), Pia +0.9 (witness) |
| 57 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 57 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +0.9 |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 57 | Ana | CUTOFF | Bruno | — | arrives week 58; anxiety Ana +0.5, Bruno +0.7; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -1.0 |
| 57 | Bruno | CUTOFF | Ana | — | arrives week 58; anxiety Ana +1.0, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Bruno -0.7 |
| 57 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 58; anxiety Marta +1.1, Nadia +3.3 (witness), Pia +2.7 (witness), Ravi +5.1; Ravi holds a position: no functioning changes hands |
| 57 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 58; anxiety Marta +3.3 (witness), Nadia +0.8, Pia +1.0 (witness), Ravi +1.0; Nadia rejects the other on Nadia–Ravi (rejection +0.79; the tie stays open) |
| 57 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 58; anxiety Marta +2.7 (witness), Nadia +1.1 (witness), Pia +0.6, Ravi +0.7; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; Pia rejects the other on Pia–Ravi (rejection +0.47; the tie stays open) |
| 57 | Ravi | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 57 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.5; anxiety Marta +1.3 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 58 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +3.9 |
| 58 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.5 |
| 58 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 58 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 58 | Ana | CUTOFF | Bruno | — | arrives week 59; anxiety Ana +0.5, Bruno +0.6; calmer sender takes some anxiety: Ana +0.7, Bruno -0.7; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 58 | Bruno | WITHHOLD | Ana | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 58 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +5.5; anxiety Nadia +1.1 (witness), Ravi +1.9 (witness) |
| 58 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +4.0; anxiety Pia +1.0 (witness), Ravi +2.1 (witness) |
| 58 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 59; anxiety Marta +1.0, Nadia +1.0 (witness), Pia +0.6, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -1.5, Pia +1.5; Marta takes 0.9 of functioning from the other |
| 58 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +3.1; anxiety Marta +2.7 (witness), Nadia +1.0 (witness) |
| 58 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 59 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +1.9 |
| 59 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 59 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 59 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.5 |
| 59 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.4 |
| 59 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Marta +5.2; anxiety Pia +1.1 (witness), Ravi +2.3 (witness) |
| 59 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +4.0; anxiety Pia +1.1 (witness), Ravi +2.2 (witness) |
| 59 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 60; anxiety Marta +2.4 (witness), Nadia +1.0 (witness), Pia +0.6, Ravi +0.6; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 59 | Ravi | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 59 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.2 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 60 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.9 (witness), Nadia +1.6 (witness), Pia +1.3 (witness), Ravi +2.6 |
| 60 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +3.9, Nadia +1.6 (witness), Pia +1.3 (witness), Ravi +1.3 (witness) |
| 60 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 60 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 60 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +1.2 (witness), Pia +1.0 (witness), Ravi +1.0 (witness) |
| 60 | Bruno | CUTOFF | Ana | — | arrives week 61; anxiety Ana +0.4, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 60 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | arrives week 61; anxiety Ana +0.8, Marta +0.6, Nadia +1.6 (witness), Pia +1.3 (witness), Ravi +1.3 (witness) |
| 60 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Nadia +4.0; anxiety Pia +1.0 (witness), Ravi +2.0 (witness) |
| 60 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 61; anxiety Marta +2.9 (witness), Nadia +1.2 (witness), Pia +0.6, Ravi +5.1; Pia takes 2.9 of functioning from the other |
| 60 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 61; anxiety Marta +1.0, Nadia +2.7 (witness), Pia +2.2 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -2.4, Ravi +2.4; 20.8 of the sender's anxiety bound into Marta–Ravi |
| 60 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.5; anxiety Marta +1.3 (witness), Nadia +1.1 (witness), Pia +0.8 (witness) |
| 60 | — | (system) | — | — | Ravi's symptom gives back 146.3 (Marta +43.9, Nadia +36.6, Pia +36.6, Sofia +29.3) |
| 61 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 61 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 61 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 61 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 61 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.6 |
| 61 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.4 |
| 61 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 62; anxiety Dr Halim +0.5, Marta +0.4, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +1.7 (witness) |
| 61 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 62; anxiety Marta +0.5, Nadia +0.9, Pia +1.0 (witness), Ravi +2.4 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.29; the tie stays open) |
| 61 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 62; anxiety Marta +9.0, Nadia +1.5 (witness), Pia +0.7, Ravi +2.6 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; Pia takes 3.5 of functioning from the other |
| 61 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | arrives week 63; anxiety Marta +1.6 (witness), Nadia +1.2 (witness), Pia +0.8 (witness), Ravi +0.4, Sofia +0.0; 19.5 of the sender's anxiety bound into Ravi–Sofia |
| 61 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 61 | — | (system) | — | — | Marta's symptom gives back 7.5 (Ana +1.5, Nadia +1.9, Pia +1.9, Ravi +2.3) |
| 61 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -2.7, Ana +0.5, Nadia +0.7, Pia +0.7, Ravi +0.8) |
| 62 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 62 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ana +2.0; anxiety Nadia +1.1 (witness), Pia +0.8 (witness), Ravi +1.0 (witness) |
| 62 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.4 |
| 62 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 63; anxiety Ana +1.3, Marta +0.8, Nadia +1.6 (witness), Pia +1.1 (witness), Ravi +1.4 (witness); Marta rejects the other on Ana–Marta (rejection +0.05; the tie stays open) |
| 62 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 63; anxiety Marta +4.0 (witness), Nadia +0.9, Pia +1.0 (witness), Ravi +0.4; Ravi takes 3.6 of functioning from the other |
| 62 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 63; anxiety Marta +4.6 (witness), Nadia +1.7 (witness), Pia +0.6, Ravi +3.5; 15.6 of the sender's anxiety bound into Pia–Ravi |
| 62 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +3.3 (witness), Pia +0.8 (witness) |
| 62 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +1.6 (witness), Nadia +1.3 (witness), Pia +0.8 (witness) |
| 62 | — | (system) | — | — | Marta's symptom gives back 38.9 (Ana +7.8, Nadia +9.7, Pia +9.7, Ravi +11.7) |
| 62 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -11.0, Ana +2.2, Nadia +2.7, Pia +2.7, Ravi +3.3) |
| 63 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 63 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.8 |
| 63 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +2.2; anxiety Nadia +1.5 (witness), Pia +1.0 (witness), Ravi +1.1 (witness) |
| 63 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 63 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +6.4; anxiety Pia +1.0 (witness), Ravi +2.4 (witness) |
| 63 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 64; anxiety Marta +3.9 (witness), Nadia +1.1, Pia +1.0 (witness), Ravi +0.6; Ravi takes 3.7 of functioning from the other |
| 63 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 64; anxiety Marta +5.6, Nadia +1.8 (witness), Pia +0.6, Ravi +2.6 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; 13.5 of the sender's anxiety bound into Marta–Pia |
| 63 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | arrives week 65; anxiety Marta +1.6 (witness), Nadia +1.5 (witness), Pia +0.8 (witness), Ravi +0.3; 30.7 of the sender's anxiety bound into Ravi–Sofia |
| 63 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +1.5 (witness), Nadia +1.4 (witness), Pia +0.8 (witness) |
| 63 | Pia | SYMPTOM_ONSET | Marta, Ravi | Nadia | arrives week 64; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, social channel; anxiety Marta +3.6, Nadia +1.1 (witness), Ravi +2.1 |
| 63 | Sofia | SYMPTOM_ONSET | Ravi | Marta, Nadia, Pia | arrives week 65; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, physical channel; anxiety Marta +1.1 (witness), Nadia +1.0 (witness), Pia +0.6 (witness), Ravi +1.2 |
| 63 | — | (system) | — | — | Marta's symptom gives back 32.6 (Ana +6.5, Nadia +8.1, Pia +8.1, Ravi +9.8) |
| 63 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -10.0, Ana +2.0, Nadia +2.5, Pia +2.5, Ravi +3.0) |
| 63 | — | (system) | — | — | Nadia's symptom gives back 6.2 (Marta +3.1, Ravi +3.1) |
| 63 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -3.5, Marta +1.8, Ravi +1.8) |
| 64 | (from outside) | OWN_ILLNESS | Pia | Marta, Ravi | a 12-week spell; anxiety Marta +1.0 (witness), Pia +1.1, Ravi +0.6 (witness); still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.5; still weighing this week: anxiety Pia +0.5 |
| 64 | (from outside) | FAMILY_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +0.9, Nadia +0.3 (witness), Pia +0.2 (witness), Ravi +0.2 (witness); still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4; still weighing this week: anxiety Marta +0.4 |
| 64 | (from outside) | FAMILY_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +0.4 (witness), Nadia +0.3 (witness), Pia +0.2 (witness), Ravi +0.5; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2 |
| 64 | (from outside) | OWN_ILLNESS | Sofia | — | a 12-week spell; anxiety Sofia +1.3; still weighing this week: anxiety Sofia +0.6; still weighing this week: anxiety Sofia +0.6; still weighing this week: anxiety Sofia +0.6; still weighing this week: anxiety Sofia +0.6; still weighing this week: anxiety Sofia +0.6; still weighing this week: anxiety Sofia +0.6; still weighing this week: anxiety Sofia +0.6; still weighing this week: anxiety Sofia +0.6; still weighing this week: anxiety Sofia +0.6; still weighing this week: anxiety Sofia +0.6; still weighing this week: anxiety Sofia +0.6 |
| 64 | (from outside) | FAMILY_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +0.3 (witness), Nadia +0.3 (witness), Pia +0.2 (witness), Ravi +0.4; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2; still weighing this week: anxiety Ravi +0.2 |
| 64 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.2 (witness), Nadia +1.7 (witness), Pia +1.1 (witness), Ravi +2.6 |
| 64 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.4, Nadia +1.7 (witness), Pia +1.1 (witness), Ravi +1.3 (witness) |
| 64 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 64 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.3 |
| 64 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 64 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.15); the sender bears the refusal: anxiety Ana +3.0 |
| 64 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.4 |
| 64 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 65; anxiety Dr Halim +0.1, Marta +1.3 (witness), Nadia +0.9 (witness), Pia +0.5 (witness); calmer sender takes some anxiety: Dr Halim +1.2, Ravi -1.2 |
| 64 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 65; anxiety Dr Halim +0.8, Marta +0.4, Nadia +1.9 (witness), Pia +1.1 (witness), Ravi +1.5 (witness) |
| 64 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 65; anxiety Marta +3.8 (witness), Nadia +1.2, Pia +1.0 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; 37.5 of the sender's anxiety bound into Nadia–Ravi |
| 64 | Pia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 64 | Ravi | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 64 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.4 (witness), Nadia +1.3 (witness), Pia +0.7 (witness) |
| 64 | Ana | SYMPTOM_ONSET | Bruno, Marta | Nadia, Pia, Ravi | arrives week 65; the nodal event re-opens the cut tie Ana–Bruno: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, physical channel; anxiety Bruno +2.7, Marta +2.2, Nadia +1.2 (witness), Pia +0.7 (witness), Ravi +0.7 (witness) |
| 64 | — | (system) | — | — | Marta's symptom gives back 30.9 (Ana +6.2, Nadia +7.7, Pia +7.7, Ravi +9.3) |
| 64 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -21.6, Ana +4.3, Nadia +5.4, Pia +5.4, Ravi +6.5) |
| 64 | — | (system) | — | — | Nadia's symptom gives back 32.4 (Marta +16.2, Ravi +16.2) |
| 64 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -22.5, Marta +11.3, Ravi +11.3) |
| 65 | (from outside) | OWN_ILLNESS | Ana | — | a 12-week spell; anxiety Ana +1.5; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7; still weighing this week: anxiety Ana +0.7 |
| 65 | (from outside) | FAMILY_ILLNESS | Bruno | — | a 12-week spell; anxiety Bruno +0.9; still weighing this week: anxiety Bruno +0.4; still weighing this week: anxiety Bruno +0.4; still weighing this week: anxiety Bruno +0.4; still weighing this week: anxiety Bruno +0.4; still weighing this week: anxiety Bruno +0.4; still weighing this week: anxiety Bruno +0.4; still weighing this week: anxiety Bruno +0.4; still weighing this week: anxiety Bruno +0.4; still weighing this week: anxiety Bruno +0.4; still weighing this week: anxiety Bruno +0.4; still weighing this week: anxiety Bruno +0.4 |
| 65 | (from outside) | FAMILY_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +0.7, Nadia +0.3 (witness), Pia +0.2 (witness), Ravi +0.2 (witness); still weighing this week: anxiety Marta +0.3; still weighing this week: anxiety Marta +0.3; still weighing this week: anxiety Marta +0.3; still weighing this week: anxiety Marta +0.3; still weighing this week: anxiety Marta +0.3; still weighing this week: anxiety Marta +0.3; still weighing this week: anxiety Marta +0.3; still weighing this week: anxiety Marta +0.3; still weighing this week: anxiety Marta +0.3; still weighing this week: anxiety Marta +0.3; still weighing this week: anxiety Marta +0.3 |
| 65 | — | (system) | — | — | Dr Halim–Ravi tie now ordinary |
| 65 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +3.2 |
| 65 | Bruno | CUTOFF | Ana | — | arrives week 66; anxiety Ana +0.8, Bruno +0.7; calmer sender takes some anxiety: Ana -0.4, Bruno +0.4; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 65 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 66; anxiety Dr Halim +1.0, Marta +0.4, Nadia +2.0 (witness), Pia +1.1 (witness), Ravi +1.6 (witness) |
| 65 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 66; anxiety Marta +2.8 (witness), Nadia +0.9, Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; 16.9 of the sender's anxiety bound into Nadia–Ravi |
| 65 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 66; anxiety Marta +5.5, Nadia +2.2 (witness), Pia +0.6, Ravi +2.8 (witness); calmer sender takes some anxiety: Marta -0.0, Pia +0.0; 14.1 of the sender's anxiety bound into Marta–Pia |
| 65 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | arrives week 67; anxiety Marta +1.5 (witness), Nadia +1.4 (witness), Pia +0.8 (witness), Ravi +0.4, Sofia +0.2; 22.3 of the sender's anxiety bound into Ravi–Sofia |
| 65 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 65 | — | (system) | — | — | Marta's symptom gives back 16.0 (Ana +3.2, Nadia +4.0, Pia +4.0, Ravi +4.8) |
| 65 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -29.1, Ana +5.8, Nadia +7.3, Pia +7.3, Ravi +8.7) |
| 65 | — | (system) | — | — | Nadia's symptom gives back 12.0 (Marta +6.0, Ravi +6.0) |
| 65 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -2.2, Marta +1.1, Ravi +1.1) |
| 66 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.6 |
| 66 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Ana +2.5; anxiety Nadia +1.8 (witness), Pia +1.1 (witness), Ravi +1.1 (witness) |
| 66 | Bruno | CUTOFF | Ana | — | arrives week 67; anxiety Ana +1.1, Bruno +0.7; calmer sender takes some anxiety: Ana -0.4, Bruno +0.4; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 66 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 67; anxiety Ana +1.2, Marta +0.7, Nadia +1.9 (witness), Pia +1.1 (witness), Ravi +1.2 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 66 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 67; anxiety Marta +2.5 (witness), Nadia +0.9, Pia +0.7 (witness), Ravi +0.3; calmer sender takes some anxiety: Nadia +0.7, Ravi -0.7; Nadia–Ravi functioning balance flips; Ravi takes 2.7 of functioning from the other |
| 66 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 67; anxiety Marta +3.1, Nadia +1.9 (witness), Pia +0.6, Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; 20.6 of the sender's anxiety bound into Marta–Pia |
| 66 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 67; anxiety Marta +6.3, Nadia +3.8 (witness), Pia +2.2 (witness), Ravi +0.7 |
| 66 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.5 (witness), Nadia +1.6 (witness), Pia +0.8 (witness) |
| 66 | — | (system) | — | — | Marta's symptom gives back 1.8 (Ana +0.4, Nadia +0.5, Pia +0.5, Ravi +0.6) |
| 66 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -7.3, Ana +1.5, Nadia +1.8, Pia +1.8, Ravi +2.2) |
| 66 | — | (system) | — | — | Nadia's symptom gives back 4.7 (Marta +2.3, Ravi +2.3) |
| 67 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 67 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +3.3 |
| 67 | Bruno | CUTOFF | Ana | — | arrives week 68; anxiety Ana +1.0, Bruno +0.7; calmer sender takes some anxiety: Ana -0.4, Bruno +0.4; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 67 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 68; anxiety Dr Halim +0.9, Marta +0.3, Nadia +2.0 (witness), Pia +1.0 (witness), Ravi +1.4 (witness) |
| 67 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 68; anxiety Marta +2.6 (witness), Nadia +1.1, Pia +0.7 (witness), Ravi +0.3; 18.1 of the sender's anxiety bound into Nadia–Ravi |
| 67 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 68; anxiety Marta +3.2, Nadia +1.8 (witness), Pia +0.5, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -1.4, Pia +1.4; Pia takes 2.7 of functioning from the other |
| 67 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.14); the sender bears the refusal: anxiety Ravi +2.7; anxiety Marta +3.2 (witness), Nadia +1.7 (witness) |
| 67 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.5 (witness), Nadia +1.4 (witness), Pia +0.7 (witness) |
| 67 | — | (system) | — | — | Marta's symptom gives back 0.4 (Ana +0.1, Nadia +0.1, Pia +0.1, Ravi +0.1) |
| 67 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -5.5, Ana +1.1, Nadia +1.4, Pia +1.4, Ravi +1.7) |
| 67 | — | (system) | — | — | Nadia's symptom gives back 4.0 (Marta +2.0, Ravi +2.0) |
| 67 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -0.3, Marta +0.2, Ravi +0.2) |
| 68 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.1 (witness), Nadia +2.1 (witness), Pia +1.1 (witness), Ravi +2.3 |
| 68 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.2, Nadia +2.1 (witness), Pia +1.1 (witness), Ravi +1.2 (witness) |
| 68 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +4.4 |
| 68 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 68 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +3.3 |
| 68 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.13); the sender bears the refusal: anxiety Bruno +3.5 |
| 68 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 69; anxiety Dr Halim +0.9, Marta +0.4, Nadia +2.0 (witness), Pia +1.0 (witness), Ravi +1.4 (witness) |
| 68 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 69; anxiety Marta +2.6 (witness), Nadia +1.0, Pia +0.6 (witness), Ravi +0.2; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; 11.9 of the sender's anxiety bound into Nadia–Ravi |
| 68 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 69; anxiety Marta +3.4, Nadia +2.0 (witness), Pia +0.5, Ravi +2.2 (witness); calmer sender takes some anxiety: Marta -1.3, Pia +1.3; 6.5 of the sender's anxiety bound into Marta–Pia |
| 68 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 69; anxiety Marta +2.8 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.6; Ravi rejects the other on Nadia–Ravi (rejection +0.76; the tie stays open); impingement removed at once, anxiety Ravi -0.4 |
| 68 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.6 (witness), Nadia +1.6 (witness), Pia +0.8 (witness) |
| 68 | — | (system) | — | — | Marta's symptom gives back 0.2 (Ana +0.0, Nadia +0.1, Pia +0.1, Ravi +0.1) |
| 68 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -8.3, Ana +1.7, Nadia +2.1, Pia +2.1, Ravi +2.5) |
| 68 | — | (system) | — | — | Nadia's symptom gives back 0.9 (Marta +0.5, Ravi +0.5) |
| 69 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 69 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +3.2 |
| 69 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.5 |
| 69 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 70; anxiety Dr Halim +0.9, Marta +0.4, Nadia +2.0 (witness), Pia +1.0 (witness), Ravi +1.4 (witness) |
| 69 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.13); the sender bears the refusal: anxiety Nadia +4.6; anxiety Marta +2.4 (witness), Pia +0.6 (witness) |
| 69 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 70; anxiety Marta +3.0, Nadia +1.7 (witness), Pia +0.5, Ravi +1.9 (witness); calmer sender takes some anxiety: Marta -1.3, Pia +1.3; 10.5 of the sender's anxiety bound into Marta–Pia |
| 69 | Ravi | WITHHOLD | Sofia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 69 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.7 (witness), Nadia +1.7 (witness), Pia +0.8 (witness) |
| 69 | — | (system) | — | — | Marta's symptom gives back 0.1 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 69 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -4.9, Ana +1.0, Nadia +1.2, Pia +1.2, Ravi +1.5) |
| 70 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 70 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 70 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Ana +2.4; anxiety Nadia +1.9 (witness), Pia +1.0 (witness), Ravi +1.1 (witness) |
| 70 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 70 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 71; anxiety Dr Halim +0.9, Marta +0.3, Nadia +1.9 (witness), Pia +0.9 (witness), Ravi +1.4 (witness) |
| 70 | Nadia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 70 | Pia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 70 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 71; anxiety Marta +6.0, Nadia +3.8 (witness), Pia +1.9 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.5, Ravi +0.5 |
| 70 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.7 (witness), Nadia +1.8 (witness), Pia +0.9 (witness) |
| 70 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 70 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -1.2, Ana +0.2, Nadia +0.3, Pia +0.3, Ravi +0.4) |
| 71 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 71 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 71 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 72; anxiety Ana +0.5, Marta +0.9, Nadia +1.7 (witness), Pia +0.9 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana rejects the other on Ana–Marta (rejection +0.76; the tie stays open) |
| 71 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 71 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 72; anxiety Dr Halim +0.9, Marta +0.3, Nadia +1.9 (witness), Pia +0.9 (witness), Ravi +1.4 (witness) |
| 71 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +5.2; anxiety Pia +0.9 (witness), Ravi +2.1 (witness) |
| 71 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 72; anxiety Marta +2.2, Nadia +1.5 (witness), Pia +0.5, Ravi +1.7 (witness); calmer sender takes some anxiety: Marta -1.5, Pia +1.5; 10.0 of the sender's anxiety bound into Marta–Pia |
| 71 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 72; anxiety Marta +5.3, Nadia +3.8 (witness), Pia +1.9 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.5, Ravi +0.5 |
| 71 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 73; anxiety Marta +1.5 (witness), Nadia +1.6 (witness), Pia +0.8 (witness), Ravi +0.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia rejects the other on Ravi–Sofia (rejection +0.64; the tie stays open) |
| 71 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 71 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -0.6, Ana +0.1, Nadia +0.1, Pia +0.1, Ravi +0.2) |
| 72 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.1 (witness), Nadia +2.1 (witness), Pia +1.1 (witness), Ravi +2.4 |
| 72 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.2, Nadia +2.1 (witness), Pia +1.1 (witness), Ravi +1.2 (witness) |
| 72 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 72 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 72 | Ana | WITHHOLD | Bruno | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 72 | Bruno | CUTOFF | Ana | — | arrives week 73; anxiety Ana +0.8, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.15; the tie stays open) |
| 72 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 73; anxiety Dr Halim +0.1, Nadia +1.2 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.9, Marta -0.9; Marta left outside: anxiety Dr Halim -0.1, Marta +0.1 (of which 0.0 the outsider's own) |
| 72 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.17); the sender bears the refusal: anxiety Marta +5.7; anxiety Pia +1.1 (witness), Ravi +2.5 (witness) |
| 72 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 73; anxiety Marta +2.8 (witness), Nadia +1.2, Pia +0.7 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.80; the tie stays open) |
| 72 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 72 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.16); the sender bears the refusal: anxiety Ravi +1.9; anxiety Marta +1.7 (witness), Nadia +1.8 (witness), Pia +0.9 (witness) |
| 72 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 72 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 72 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -1.3, Ana +0.3, Nadia +0.3, Pia +0.3, Ravi +0.4) |
| 73 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +3.2 |
| 73 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 73 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 73 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 73 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +3.0 |
| 73 | Bruno | CUTOFF | Ana | — | arrives week 74; anxiety Ana +0.9, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 73 | Marta | WITHHOLD | Pia | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 73 | Nadia | TRIANGLE | Ravi | Marta, Pia | refused by Ravi (rejection +0.16); the sender bears the refusal: anxiety Nadia +5.7; anxiety Marta +2.7 (witness), Pia +0.7 (witness) |
| 73 | Pia | TRIANGLE | Marta | Nadia, Ravi | arrives week 74; anxiety Marta +1.7, Nadia +1.4 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -1.8, Pia +1.8; Marta left outside: anxiety Marta +1.5, Pia -1.0 (of which 0.5 the outsider's own) |
| 73 | Ravi | WITHHOLD | Nadia | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 73 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.13); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.5 (witness), Nadia +1.6 (witness), Pia +0.8 (witness) |
| 73 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 73 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -1.1, Ana +0.2, Nadia +0.3, Pia +0.3, Ravi +0.3) |
| 74 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 74 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 74 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 74 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 74 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 74 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 74 | Ana | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 74 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.13); the sender bears the refusal: anxiety Bruno +3.5 |
| 74 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 75; anxiety Dr Halim +0.9, Marta +0.3, Nadia +1.8 (witness), Pia +0.9 (witness), Ravi +1.4 (witness) |
| 74 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.16); the sender bears the refusal: anxiety Nadia +5.8; anxiety Marta +3.5 (witness), Pia +0.9 (witness) |
| 74 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 74 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 75; anxiety Marta +5.8, Nadia +3.9 (witness), Pia +1.9 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.6, Ravi +0.6 |
| 74 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.07); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.5 (witness), Nadia +1.6 (witness), Pia +0.8 (witness) |
| 74 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 74 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -0.2, Ana +0.0, Nadia +0.1, Pia +0.1, Ravi +0.1) |
| 75 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 75 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 75 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Ana +2.2; anxiety Nadia +1.6 (witness), Pia +0.8 (witness), Ravi +0.9 (witness) |
| 75 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.07); the sender bears the refusal: anxiety Bruno +3.6 |
| 75 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 76; anxiety Ana +0.5, Marta +0.6, Nadia +1.9 (witness), Pia +1.0 (witness), Ravi +1.1 (witness); Marta rejects the other on Ana–Marta (rejection +0.16; the tie stays open) |
| 75 | Nadia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 75 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 75 | Ravi | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 75 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.5 (witness), Nadia +1.7 (witness), Pia +0.8 (witness) |
| 75 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 75 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -0.0, Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 76 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.9 (witness), Nadia +2.1 (witness), Pia +1.1 (witness), Ravi +2.5 |
| 76 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +3.9, Nadia +2.1 (witness), Pia +1.1 (witness), Ravi +1.3 (witness) |
| 76 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 76 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 76 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 76 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +2.2; anxiety Nadia +1.6 (witness), Pia +0.8 (witness), Ravi +1.0 (witness) |
| 76 | Bruno | WITHHOLD | Ana | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 76 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 77; anxiety Dr Halim +0.8, Marta +0.3, Nadia +1.9 (witness), Pia +0.9 (witness), Ravi +1.4 (witness) |
| 76 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.17); the sender bears the refusal: anxiety Nadia +6.0; anxiety Marta +4.0 (witness), Pia +1.1 (witness) |
| 76 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 77; anxiety Marta +7.9, Nadia +1.6 (witness), Pia +0.5, Ravi +1.9 (witness); calmer sender takes some anxiety: Marta -2.3, Pia +2.3; Marta holds a position: no functioning changes hands |
| 76 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 77; anxiety Marta +3.2 (witness), Nadia +1.8 (witness), Pia +0.2, Ravi +0.6 |
| 76 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.5 (witness), Nadia +1.7 (witness), Pia +0.9 (witness) |
| 76 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 76 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -0.0, Marta +0.0, Ravi +0.0) |
| 76 | — | (system) | — | — | Pia's symptom gives back 33.7 (Marta +16.9, Ravi +16.9) |
| 77 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 77 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.1 |
| 77 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 77 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 77 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 77 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +2.2; anxiety Nadia +1.6 (witness), Pia +0.8 (witness), Ravi +0.9 (witness) |
| 77 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.5 |
| 77 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.18); the sender bears the refusal: anxiety Marta +3.5; anxiety Nadia +2.1 (witness), Pia +1.1 (witness), Ravi +1.3 (witness) |
| 77 | Nadia | PURSUE | Marta | Pia, Ravi | arrives week 78; anxiety Marta +6.5, Nadia +1.2, Pia +1.1 (witness), Ravi +2.7 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4 |
| 77 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 78; anxiety Marta +1.6, Nadia +1.6 (witness), Pia +0.5, Ravi +1.8 (witness); calmer sender takes some anxiety: Marta -2.4, Pia +2.4; 12.1 of the sender's anxiety bound into Marta–Pia |
| 77 | Ravi | WITHHOLD | Pia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 77 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.5 (witness), Nadia +1.8 (witness), Pia +0.9 (witness) |
| 77 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 77 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -0.5, Ana +0.1, Nadia +0.1, Pia +0.1, Ravi +0.2) |
| 77 | — | (system) | — | — | Nadia's symptom gives back 2.4 (Marta +1.2, Ravi +1.2) |
| 77 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -1.9, Marta +0.9, Ravi +0.9) |
| 78 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 78 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 78 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.13); the sender bears the refusal: anxiety Ana +2.8 |
| 78 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 78 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 79; anxiety Marta +1.1, Nadia +4.2 (witness), Pia +2.1 (witness), Ravi +0.7; Marta rejects the other on Marta–Ravi (rejection +0.91; the tie stays open); impingement removed at once, anxiety Marta -1.5 |
| 78 | Nadia | TRIANGLE | Ravi | Marta, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Nadia +6.1; anxiety Marta +3.6 (witness), Pia +1.0 (witness) |
| 78 | Pia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 78 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 79; anxiety Marta +3.5 (witness), Nadia +2.0 (witness), Pia +3.3, Ravi +0.7 |
| 78 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.3 (witness), Nadia +1.6 (witness), Pia +0.8 (witness) |
| 78 | — | (system) | — | — | Marta's symptom gives back 0.1 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 78 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -1.4, Ana +0.3, Nadia +0.4, Pia +0.4, Ravi +0.4) |
| 78 | — | (system) | — | — | Nadia's symptom gives back 3.9 (Marta +1.9, Ravi +1.9) |
| 78 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -2.9, Marta +1.4, Ravi +1.4) |
| 79 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +13.9 |
| 79 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.1 |
| 79 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 79 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 79 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 79 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 80; anxiety Ana +0.4, Marta +0.8, Nadia +1.4 (witness), Pia +0.7 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +1.3, Marta -1.3; Ana rejects the other on Ana–Marta (rejection +0.46; the tie stays open) |
| 79 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 79 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 80; anxiety Marta +1.1, Nadia +3.8 (witness), Pia +1.9 (witness), Ravi +1.2; Ravi left outside: anxiety Marta -5.9, Ravi +8.8 (of which 2.9 the outsider's own) |
| 79 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.2; anxiety Marta +3.7 (witness), Pia +1.1 (witness) |
| 79 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 80; anxiety Marta +5.4, Nadia +1.6 (witness), Pia +0.5, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -2.2, Pia +2.2; Pia takes 2.7 of functioning from the other |
| 79 | Ravi | CONFLICT | Nadia | Marta, Pia | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Ravi +3.4; anxiety Marta +3.4 (witness), Pia +1.0 (witness) |
| 79 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.3 (witness), Nadia +1.4 (witness), Pia +0.7 (witness) |
| 79 | — | (system) | — | — | Marta's symptom gives back 0.1 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 79 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -1.4, Ana +0.3, Nadia +0.4, Pia +0.4, Ravi +0.4) |
| 79 | — | (system) | — | — | Nadia's symptom gives back 5.1 (Marta +2.6, Ravi +2.6) |
| 79 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -4.6, Marta +2.3, Ravi +2.3) |
| 80 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +1.8 (witness), Nadia +2.2 (witness), Pia +1.1 (witness), Ravi +2.6 |
| 80 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +3.7, Nadia +2.1 (witness), Pia +1.1 (witness), Ravi +1.3 (witness) |
| 80 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 80 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 80 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 80 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 80 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 80 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 80 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 80 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.13); the sender bears the refusal: anxiety Ana +2.8 |
| 80 | Bruno | CUTOFF | Ana | — | arrives week 81; anxiety Ana +0.4, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.08; the tie stays open) |
| 80 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 81; anxiety Dr Halim +0.1, Nadia +1.1 (witness), Pia +0.5 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +1.1, Marta -1.1; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own); the contact lands with Marta: systems perspective +0.10 |
| 80 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | arrives week 81; anxiety Marta +1.2, Nadia +2.2 (witness), Pia -0.3, Ravi +2.6 (witness); Pia takes 3.6 of functioning from the other |
| 80 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.3; anxiety Marta +3.7 (witness), Pia +1.0 (witness) |
| 80 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 81; anxiety Marta +3.0, Nadia +1.7 (witness), Pia +0.5, Ravi +2.1 (witness); calmer sender takes some anxiety: Marta -2.1, Pia +2.1; Pia takes 1.1 of functioning from the other |
| 80 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 81; anxiety Marta +4.1 (witness), Nadia +2.2 (witness), Pia +2.8, Ravi +0.7 |
| 80 | Sofia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 80 | — | (system) | — | — | Marta's symptom gives back 0.2 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.1) |
| 80 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -3.5, Ana +0.7, Nadia +0.9, Pia +0.9, Ravi +1.0) |
| 80 | — | (system) | — | — | Nadia's symptom gives back 6.4 (Marta +3.2, Ravi +3.2) |
| 80 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -8.6, Marta +4.3, Ravi +4.3) |
| 81 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 81 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 81 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 81 | Ana | CUTOFF | Bruno | — | arrives week 82; anxiety Ana +0.6, Bruno +1.0; calmer sender takes some anxiety: Ana +0.4, Bruno -0.4; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ana -0.1 |
| 81 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 81 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 82; anxiety Dr Halim +0.5, Marta +0.4, Nadia +2.0 (witness), Pia +0.9 (witness), Ravi +1.6 (witness) |
| 81 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.3; anxiety Marta +4.1 (witness), Pia +0.8 (witness) |
| 81 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 82; anxiety Marta +2.4, Nadia +1.7 (witness), Pia +0.5, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -1.7, Pia +1.7; 15.4 of the sender's anxiety bound into Marta–Pia |
| 81 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.18); the sender bears the refusal: anxiety Ravi +3.8; anxiety Marta +4.0 (witness), Pia +0.8 (witness) |
| 81 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.5 (witness), Nadia +1.5 (witness), Pia +0.6 (witness) |
| 81 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 82, 83; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +3.9, Nadia +3.6, Pia +1.6; anxiety Sofia +1.3 |
| 81 | — | (system) | — | — | Marta's symptom gives back 0.4 (Ana +0.1, Nadia +0.1, Pia +0.1, Ravi +0.1) |
| 81 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -10.5, Ana +2.1, Nadia +2.6, Pia +2.6, Ravi +3.1) |
| 81 | — | (system) | — | — | Nadia's symptom gives back 5.9 (Marta +2.9, Ravi +2.9) |
| 81 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -11.5, Marta +5.7, Ravi +5.7) |
| 82 | (from outside) | OWN_ILLNESS | Ravi | Marta, Nadia, Pia | a 12-week spell; anxiety Marta +1.1 (witness), Nadia +1.0 (witness), Pia +0.4 (witness), Ravi +1.2; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.7; still weighing this week: anxiety Ravi +0.7; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6; still weighing this week: anxiety Ravi +0.6 |
| 82 | (from outside) | FAMILY_ILLNESS | Marta | Nadia, Pia, Ravi | a 12-week spell; anxiety Marta +1.1, Nadia +0.5 (witness), Pia +0.2 (witness), Ravi +0.3 (witness); still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.6; still weighing this week: anxiety Marta +0.6; still weighing this week: anxiety Marta +0.6; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5; still weighing this week: anxiety Marta +0.5 |
| 82 | (from outside) | FAMILY_ILLNESS | Nadia | Marta, Ravi | a 12-week spell; anxiety Marta +0.5 (witness), Nadia +0.9, Ravi +0.3 (witness); still weighing this week: anxiety Nadia +0.4; still weighing this week: anxiety Nadia +0.4; still weighing this week: anxiety Nadia +0.4; still weighing this week: anxiety Nadia +0.4; still weighing this week: anxiety Nadia +0.4; still weighing this week: anxiety Nadia +0.4; still weighing this week: anxiety Nadia +0.4; still weighing this week: anxiety Nadia +0.4; still weighing this week: anxiety Nadia +0.4; still weighing this week: anxiety Nadia +0.4; still weighing this week: anxiety Nadia +0.4 |
| 82 | (from outside) | FAMILY_ILLNESS | Pia | Marta, Ravi | a 12-week spell; anxiety Marta +0.5 (witness), Pia +0.4, Ravi +0.3 (witness); still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2; still weighing this week: anxiety Pia +0.2 |
| 82 | (from outside) | FAMILY_ILLNESS | Sofia | — | a 12-week spell; anxiety Sofia +0.4; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2; still weighing this week: anxiety Sofia +0.2 |
| 82 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 82 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.8 |
| 82 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.4 |
| 82 | Marta | PREVENT_ALIGNMENT | Nadia | Pia, Ravi | arrives week 83; anxiety Marta +0.8, Nadia +0.4, Pia +0.5 (witness), Ravi +1.5 (witness) |
| 82 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.3; anxiety Marta +3.8 (witness), Pia +0.8 (witness) |
| 82 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 83; anxiety Marta +1.9, Nadia +1.4 (witness), Pia +0.4, Ravi +1.7 (witness); calmer sender takes some anxiety: Marta -2.0, Pia +2.0; 9.4 of the sender's anxiety bound into Marta–Pia |
| 82 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 83; anxiety Marta +4.5 (witness), Nadia +2.1 (witness), Pia +1.0, Ravi +0.8; Ravi rejects the other on Pia–Ravi (rejection +0.90; the tie stays open) |
| 82 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.4 (witness), Nadia +1.3 (witness), Pia +0.6 (witness) |
| 82 | — | (system) | — | — | Marta's symptom gives back 0.2 (Ana +0.0, Nadia +0.1, Pia +0.1, Ravi +0.1) |
| 82 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -11.8, Ana +2.4, Nadia +3.0, Pia +3.0, Ravi +3.6) |
| 82 | — | (system) | — | — | Nadia's symptom gives back 4.1 (Marta +2.0, Ravi +2.0) |
| 82 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -11.4, Marta +5.7, Ravi +5.7) |
| 83 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 83 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +12.6 |
| 83 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +1.5 |
| 83 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 83 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 83 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.8 |
| 83 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 83 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 84; anxiety Marta +1.3, Nadia +3.6 (witness), Pia +1.6 (witness), Ravi +0.9; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4; Marta rejects the other on Marta–Ravi (rejection +0.34; the tie stays open) |
| 83 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.0; anxiety Marta +3.5 (witness), Pia +0.7 (witness) |
| 83 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.13); the sender bears the refusal: anxiety Pia +2.0; anxiety Marta +3.1 (witness), Nadia +1.5 (witness) |
| 83 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | arrives week 84; anxiety Marta +4.5 (witness), Nadia +2.1 (witness), Pia +0.9, Ravi +0.8; Pia takes 3.6 of functioning from the other |
| 83 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.3 (witness), Nadia +1.3 (witness), Pia +0.5 (witness) |
| 83 | — | (system) | — | — | Marta's symptom gives back 0.1 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 83 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -7.1, Ana +1.4, Nadia +1.8, Pia +1.8, Ravi +2.1) |
| 83 | — | (system) | — | — | Nadia's symptom gives back 2.2 (Marta +1.1, Ravi +1.1) |
| 83 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -7.2, Marta +3.6, Ravi +3.6) |
| 84 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.3 (witness), Nadia +2.2 (witness), Pia +0.9 (witness), Ravi +2.7 |
| 84 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.6, Nadia +2.2 (witness), Pia +0.9 (witness), Ravi +1.3 (witness) |
| 84 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +4.1 |
| 84 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 84 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.9 |
| 84 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 84 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 84 | Ana | CUTOFF | Bruno | — | arrives week 85; anxiety Ana +0.6, Bruno +1.7; calmer sender takes some anxiety: Ana +0.4, Bruno -0.4; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 84 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 84 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 85; anxiety Marta +1.4, Nadia +3.6 (witness), Pia +1.5 (witness), Ravi +0.9; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1; Ravi takes 0.9 of functioning from the other |
| 84 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.2; anxiety Marta +3.5 (witness), Pia +0.7 (witness) |
| 84 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 85; anxiety Marta +2.1, Nadia +1.4 (witness), Pia +0.4, Ravi +1.8 (witness); calmer sender takes some anxiety: Marta -2.1, Pia +2.1; 10.3 of the sender's anxiety bound into Marta–Pia |
| 84 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | arrives week 86; anxiety Marta +2.3 (witness), Nadia +2.1 (witness), Pia +0.9 (witness), Ravi +0.5, Sofia +2.6; Ravi takes 3.5 of functioning from the other |
| 84 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +1.6 (witness), Nadia +1.5 (witness), Pia +0.6 (witness) |
| 84 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 84 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -10.2, Ana +2.0, Nadia +2.5, Pia +2.5, Ravi +3.1) |
| 84 | — | (system) | — | — | Nadia's symptom gives back 1.3 (Marta +0.7, Ravi +0.7) |
| 84 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -9.9, Marta +4.9, Ravi +4.9) |
| 85 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 85 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 85 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Ana +2.2; anxiety Nadia +1.7 (witness), Pia +0.7 (witness), Ravi +1.1 (witness) |
| 85 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 85 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 86; anxiety Marta +1.4, Nadia +4.3 (witness), Pia +1.8 (witness), Ravi +5.7; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; 13.7 of the sender's anxiety bound into Marta–Ravi |
| 85 | Nadia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 85 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 86; anxiety Marta +1.8, Nadia +1.2 (witness), Pia +0.4, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -2.1, Pia +2.1; 7.7 of the sender's anxiety bound into Marta–Pia |
| 85 | Ravi | WITHHOLD | Dr Halim | — | held back DISPLACE: computed, not emitted; attention on the tie rises |
| 85 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +0.6 (witness) |
| 85 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 85 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -6.5, Ana +1.3, Nadia +1.6, Pia +1.6, Ravi +1.9) |
| 85 | — | (system) | — | — | Nadia's symptom gives back 0.7 (Marta +0.4, Ravi +0.4) |
| 85 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -7.9, Marta +4.0, Ravi +4.0) |
| 86 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 86 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 86 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 87; anxiety Ana +0.4, Marta +1.1, Nadia +1.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana rejects the other on Ana–Marta (rejection +0.35; the tie stays open) |
| 86 | Bruno | CUTOFF | Ana | — | arrives week 87; anxiety Ana +0.2, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.15; the tie stays open); impingement removed at once, anxiety Bruno -0.5 |
| 86 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 87; anxiety Marta +1.2, Nadia +1.4 (witness), Pia +0.2, Ravi +1.8 (witness); Marta rejects the other on Marta–Pia (rejection +0.78; the tie stays open); impingement removed at once, anxiety Marta -4.0 |
| 86 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +5.8; anxiety Marta +3.4 (witness), Pia +0.6 (witness) |
| 86 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Pia +1.8; anxiety Marta +2.9 (witness), Nadia +1.3 (witness) |
| 86 | Ravi | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 86 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +1.4 (witness), Nadia +1.3 (witness), Pia +0.5 (witness) |
| 86 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 86 | — | (system) | — | — | Nadia's symptom gives back 0.3 (Marta +0.2, Ravi +0.2) |
| 86 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -4.0, Marta +2.0, Ravi +2.0) |
| 87 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 87 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +1.8 |
| 87 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.1 |
| 87 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.6 |
| 87 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 87 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.8 |
| 87 | Bruno | CUTOFF | Ana | — | arrives week 88; anxiety Ana +0.7, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 87 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 88; anxiety Marta +1.2, Nadia +3.5 (witness), Pia +1.5 (witness), Ravi +1.0; calmer sender takes some anxiety: Marta +0.5, Ravi -0.5; Marta rejects the other on Marta–Ravi (rejection +0.20; the tie stays open) |
| 87 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.8; anxiety Marta +3.7 (witness), Pia +0.7 (witness) |
| 87 | Pia | PREVENT_ALIGNMENT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +1.5; anxiety Marta +2.3 (witness), Nadia +1.0 (witness) |
| 87 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.17); the sender bears the refusal: anxiety Ravi +3.7; anxiety Marta +3.8 (witness), Pia +0.7 (witness) |
| 87 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.3 (witness), Nadia +1.2 (witness), Pia +0.5 (witness) |
| 87 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 87 | — | (system) | — | — | Nadia's symptom gives back 0.2 (Marta +0.1, Ravi +0.1) |
| 87 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -3.1, Marta +1.5, Ravi +1.5) |
| 88 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.4 (witness), Nadia +2.1 (witness), Pia +0.9 (witness), Ravi +2.6 |
| 88 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.8, Nadia +2.1 (witness), Pia +0.9 (witness), Ravi +1.3 (witness) |
| 88 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.9 |
| 88 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 88 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.9 |
| 88 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 88 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ana +2.1; anxiety Nadia +1.7 (witness), Pia +0.7 (witness), Ravi +1.0 (witness) |
| 88 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.4 |
| 88 | Dr Halim | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 89; anxiety Dr Halim +0.1, Marta +1.6 (witness), Nadia +1.1 (witness), Pia +0.5 (witness), Ravi +0.4; calmer sender takes some anxiety: Dr Halim +0.9, Ravi -0.9; Ravi holds a position: no functioning changes hands |
| 88 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 89; anxiety Marta +1.3, Nadia +3.6 (witness), Pia +1.5 (witness), Ravi +0.9; Ravi holds a position: no functioning changes hands |
| 88 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.9; anxiety Marta +3.6 (witness), Pia +0.7 (witness) |
| 88 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Pia +2.0; anxiety Nadia +1.5 (witness), Ravi +1.8 (witness) |
| 88 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 89; anxiety Marta +3.6 (witness), Nadia +1.7 (witness), Pia +2.6, Ravi +0.7; 26.5 of the sender's anxiety bound into Pia–Ravi |
| 88 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.2 (witness), Nadia +1.2 (witness), Pia +0.5 (witness) |
| 88 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 88 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -1.7, Ana +0.3, Nadia +0.4, Pia +0.4, Ravi +0.5) |
| 88 | — | (system) | — | — | Nadia's symptom gives back 0.1 (Marta +0.1, Ravi +0.1) |
| 88 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -4.1, Marta +2.1, Ravi +2.1) |
| 89 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 89 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +3.5 |
| 89 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 89 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 89 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.8 |
| 89 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.08); the sender bears the refusal: anxiety Bruno +3.4 |
| 89 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 90; anxiety Marta +1.3, Nadia +2.3 (witness), Pia +2.9, Ravi +2.8 (witness); 14.7 of the sender's anxiety bound into Marta–Pia |
| 89 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.1; anxiety Marta +3.4 (witness), Pia +0.7 (witness) |
| 89 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 89 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 90; anxiety Marta +1.2, Nadia +2.8 (witness), Pia +1.2 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -1.3, Ravi +1.3; Ravi rejects the other on Marta–Ravi (rejection +0.70; the tie stays open); impingement removed at once, anxiety Ravi -2.1 |
| 89 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.2 (witness), Nadia +1.1 (witness), Pia +0.5 (witness) |
| 89 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 89 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -2.0, Ana +0.4, Nadia +0.5, Pia +0.5, Ravi +0.6) |
| 89 | — | (system) | — | — | Nadia's symptom gives back 0.1 (Marta +0.1, Ravi +0.1) |
| 89 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -6.0, Marta +3.0, Ravi +3.0) |
| 90 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.9 |
| 90 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 90 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 90 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.8 |
| 90 | Bruno | CUTOFF | Ana | — | arrives week 91; anxiety Ana +0.6, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 90 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | arrives week 91; anxiety Marta +1.2, Nadia +1.9 (witness), Pia +0.2, Ravi +2.3 (witness); Pia holds a position: no functioning changes hands |
| 90 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.0; anxiety Marta +3.3 (witness), Pia +0.7 (witness) |
| 90 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 90 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | arrives week 92; anxiety Marta +1.5 (witness), Nadia +1.4 (witness), Pia +0.6 (witness), Ravi +0.4, Sofia +0.8 |
| 90 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.3 (witness), Nadia +1.2 (witness), Pia +0.5 (witness) |
| 90 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 90 | — | (system) | — | — | Nadia's symptom gives back 0.1 (Marta +0.0, Ravi +0.0) |
| 90 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -4.6, Marta +2.3, Ravi +2.3) |
| 91 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 91 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 91 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ana +2.1; anxiety Nadia +1.8 (witness), Pia +0.8 (witness), Ravi +1.1 (witness) |
| 91 | Bruno | CUTOFF | Ana | — | arrives week 92; anxiety Ana +0.7, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 91 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.16); the sender bears the refusal: anxiety Marta +5.9; anxiety Nadia +3.7 (witness), Pia +1.6 (witness) |
| 91 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.8; anxiety Marta +3.2 (witness), Pia +0.7 (witness) |
| 91 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 91 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 92; anxiety Marta +2.9 (witness), Nadia +1.4 (witness), Pia +0.9, Ravi +0.6; 13.9 of the sender's anxiety bound into Pia–Ravi |
| 91 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.2 (witness), Nadia +1.2 (witness), Pia +0.5 (witness) |
| 91 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 91 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -2.5, Marta +1.2, Ravi +1.2) |
| 92 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.2 (witness), Nadia +2.1 (witness), Pia +0.9 (witness), Ravi +2.6 |
| 92 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.5, Nadia +2.1 (witness), Pia +0.9 (witness), Ravi +1.3 (witness) |
| 92 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.0 |
| 92 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.9 |
| 92 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 92 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 92 | Ana | CUTOFF | Bruno | — | arrives week 93; anxiety Ana +0.6, Bruno +1.8; calmer sender takes some anxiety: Ana +0.6, Bruno -0.6; Ana rejects the other on Ana–Bruno (rejection +0.03; the tie stays open); impingement removed at once, anxiety Ana -0.5 |
| 92 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.4 |
| 92 | Marta | CONFLICT | Ravi | Nadia, Pia | refused by Ravi (rejection +0.17); the sender bears the refusal: anxiety Marta +6.4; anxiety Nadia +4.7 (witness), Pia +2.0 (witness) |
| 92 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 93; anxiety Marta +3.3 (witness), Nadia +1.2, Pia +0.7 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 92 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Pia +2.1; anxiety Nadia +1.5 (witness), Ravi +1.9 (witness) |
| 92 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 93; anxiety Marta +2.5 (witness), Nadia +1.2 (witness), Pia +0.6, Ravi +0.6; 8.9 of the sender's anxiety bound into Pia–Ravi |
| 92 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.4 (witness), Nadia +1.3 (witness), Pia +0.6 (witness) |
| 92 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 92 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -4.6, Marta +2.3, Ravi +2.3) |
| 93 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +5.6 |
| 93 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.9 |
| 93 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 93 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 93 | Ana | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 93 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 93 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 94; anxiety Marta +1.3, Nadia +2.0 (witness), Pia +1.3, Ravi +2.5 (witness); Marta rejects the other on Marta–Pia (rejection +0.15; the tie stays open) |
| 93 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 94; anxiety Nadia +1.2, Pia +0.8 (witness), Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -0.3, Nadia +0.3; Nadia rejects the other on Marta–Nadia (rejection +0.85; the tie stays open) |
| 93 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Pia +2.2; anxiety Nadia +1.6 (witness), Ravi +2.0 (witness) |
| 93 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 94; anxiety Marta +2.9 (witness), Nadia +1.4 (witness), Pia +3.0, Ravi +0.5; calmer sender takes some anxiety: Pia -0.2, Ravi +0.2; 4.0 of the sender's anxiety bound into Pia–Ravi |
| 93 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.3 (witness), Nadia +1.3 (witness), Pia +0.5 (witness) |
| 93 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 93 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -0.3, Ana +0.1, Nadia +0.1, Pia +0.1, Ravi +0.1) |
| 93 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 93 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -3.5, Marta +1.7, Ravi +1.7) |
| 94 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +1.9 |
| 94 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 94 | Ana | WITHHOLD | Bruno | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 94 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 94 | Marta | CONFLICT | Ravi | Nadia, Pia | refused by Ravi (rejection +0.17); the sender bears the refusal: anxiety Marta +6.4; anxiety Nadia +4.8 (witness), Pia +2.1 (witness) |
| 94 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Nadia +5.8; anxiety Marta +3.1 (witness), Pia +0.6 (witness) |
| 94 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +2.2; anxiety Nadia +1.5 (witness), Ravi +1.9 (witness) |
| 94 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ravi +2.8; anxiety Nadia +3.6 (witness), Pia +1.6 (witness) |
| 94 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.3 (witness), Nadia +1.2 (witness), Pia +0.5 (witness) |
| 94 | — | (system) | — | — | Marta's symptom gives back 0.0 (Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 94 | — | (system) | — | — | Marta's symptom is past its turn and holds their anxiety no longer (Marta -0.1, Ana +0.0, Nadia +0.0, Pia +0.0, Ravi +0.0) |
| 94 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 94 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -1.6, Marta +0.8, Ravi +0.8) |
| 95 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +1.9 |
| 95 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.9 |
| 95 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.1 |
| 95 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 95 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.12); the sender bears the refusal: anxiety Ana +2.6 |
| 95 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 95 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 96; anxiety Marta +1.3, Nadia +1.9 (witness), Pia +1.4, Ravi +2.4 (witness); Marta rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 95 | Nadia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 95 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +2.2; anxiety Nadia +1.6 (witness), Ravi +2.0 (witness) |
| 95 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.13); the sender bears the refusal: anxiety Ravi +2.9; anxiety Marta +2.5 (witness), Pia +0.5 (witness) |
| 95 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 95 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 95 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -2.7, Marta +1.4, Ravi +1.4) |
| 96 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.2 (witness), Nadia +2.1 (witness), Pia +0.9 (witness), Ravi +2.6 |
| 96 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.5, Nadia +2.1 (witness), Pia +0.9 (witness), Ravi +1.3 (witness) |
| 96 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +1.9 |
| 96 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 96 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +0.8 |
| 96 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.13); the sender bears the refusal: anxiety Ana +2.6 |
| 96 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 96 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 97; anxiety Dr Halim +0.1, Nadia +1.0 (witness), Pia +0.5 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +1.0, Marta -1.0; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.51; the tie stays open) |
| 96 | Marta | WITHHOLD | Pia | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 96 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +5.9; anxiety Marta +3.1 (witness), Pia +0.6 (witness) |
| 96 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 96 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | arrives week 98; anxiety Marta +1.5 (witness), Nadia +1.4 (witness), Pia +0.6 (witness), Ravi +0.4, Sofia +2.6; 30.8 of the sender's anxiety bound into Ravi–Sofia |
| 96 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.2 (witness), Nadia +1.2 (witness), Pia +0.5 (witness) |
| 96 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 96 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -1.6, Marta +0.8, Ravi +0.8) |
| 97 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 97 | Ana | WITHHOLD | Bruno | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 97 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 97 | Marta | CONFLICT | Ravi | Nadia, Pia | refused by Ravi (rejection +0.17); the sender bears the refusal: anxiety Marta +6.3; anxiety Nadia +5.1 (witness), Pia +2.2 (witness) |
| 97 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +5.8; anxiety Marta +3.4 (witness), Pia +0.7 (witness) |
| 97 | Pia | STAY-IN-CONTACT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +1.5; anxiety Marta +2.2 (witness), Nadia +1.0 (witness) |
| 97 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.12); the sender bears the refusal: anxiety Ravi +3.1; anxiety Marta +3.0 (witness), Pia +0.6 (witness) |
| 97 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.1 (witness), Nadia +1.1 (witness), Pia +0.5 (witness) |
| 97 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 97 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -0.9, Marta +0.4, Ravi +0.4) |
| 97 | — | (system) | — | — | Sofia's symptom gives back 51.5 (Ravi +51.5) |
| 98 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.8 |
| 98 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 98 | — | (system) | — | — | Pia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +5.1 |
| 98 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.6 |
| 98 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 98 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 99; anxiety Marta +1.3, Nadia +2.3 (witness), Pia +2.6, Ravi +2.8 (witness); 20.2 of the sender's anxiety bound into Marta–Pia |
| 98 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +6.0; anxiety Marta +3.3 (witness), Pia +0.7 (witness) |
| 98 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 98 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | arrives week 100; anxiety Marta +1.4 (witness), Nadia +1.4 (witness), Pia +0.6 (witness), Ravi +0.4, Sofia +1.0; 21.1 of the sender's anxiety bound into Ravi–Sofia |
| 98 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.1 (witness), Nadia +1.1 (witness), Pia +0.5 (witness) |
| 98 | — | (system) | — | — | Ana's symptom gives back 63.8 (Bruno +37.0, Marta +26.9) |
| 98 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 98 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -1.2, Marta +0.6, Ravi +0.6) |
| 99 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.1 |
| 99 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +1.0 |
| 99 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.6 |
| 99 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +4.6 |
| 99 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 100; anxiety Marta +1.2, Nadia +2.0 (witness), Pia +1.4, Ravi +2.5 (witness); Marta rejects the other on Marta–Pia (rejection +0.06; the tie stays open) |
| 99 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 100; anxiety Nadia +1.2, Pia +0.8 (witness), Ravi +2.1 (witness); calmer sender takes some anxiety: Marta -0.0, Nadia +0.0; Nadia rejects the other on Marta–Nadia (rejection +0.45; the tie stays open) |
| 99 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Pia +2.2; anxiety Nadia +1.7 (witness), Ravi +2.0 (witness) |
| 99 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Ravi +3.3; anxiety Nadia +4.6 (witness), Pia +2.0 (witness) |
| 99 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.1 (witness), Nadia +1.1 (witness), Pia +0.5 (witness) |
| 99 | Bruno | SYMPTOM_ONSET | Ana | — | arrives week 100; the nodal event re-opens the cut tie Ana–Bruno: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); symptom onset, social channel; anxiety Ana +2.1 |
| 99 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 99 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -0.8, Marta +0.4, Ravi +0.4) |
| 100 | (from outside) | OWN_ILLNESS | Bruno | — | a 12-week spell; anxiety Bruno +2.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0; still weighing this week: anxiety Bruno +1.0 |
| 100 | (from outside) | FAMILY_ILLNESS | Ana | — | a 12-week spell; anxiety Ana +0.7; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3; still weighing this week: anxiety Ana +0.3 |
| 100 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +2.1 (witness), Nadia +2.1 (witness), Pia +0.9 (witness), Ravi +2.6 |
| 100 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; the nodal event re-opens the cut tie Ana–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Dr Halim–Marta: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Nadia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Pia: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Marta–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Nadia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Pia–Ravi: its standing load spikes next week (too-little side +1 × optimum); the nodal event re-opens the cut tie Ravi–Sofia: its standing load spikes next week (too-little side +1 × optimum); anxiety Marta +4.3, Nadia +2.1 (witness), Pia +0.9 (witness), Ravi +1.3 (witness) |
| 100 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +2.0 |
| 100 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +1.9 |
| 100 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 101; anxiety Ana +0.4, Marta +0.8, Nadia +1.7 (witness), Pia +0.8 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Ana +1.3, Marta -1.3; Ana rejects the other on Ana–Marta (rejection +0.66; the tie stays open); impingement removed at once, anxiety Ana -6.3 |
| 100 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +4.5 |
| 100 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 101; anxiety Ana +4.2, Marta +0.7, Nadia +2.3 (witness), Pia +1.0 (witness), Ravi +1.4 (witness); 14.4 of the sender's anxiety bound into Ana–Marta |
| 100 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +6.1; anxiety Marta +3.1 (witness), Pia +0.7 (witness) |
| 100 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 100 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Ravi +3.0; anxiety Nadia +4.2 (witness), Pia +1.9 (witness) |
| 100 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.1 (witness), Nadia +1.1 (witness), Pia +0.5 (witness) |
| 100 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 100 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -1.3, Marta +0.7, Ravi +0.7) |
| 101 | — | (system) | — | — | Marta–Ravi: a negative contact renews the cutoff's held anxiety to its peak +0.9 |
| 101 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.1 |
| 101 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +1.2 |
| 101 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 102; anxiety Ana +0.4, Marta +1.2, Nadia +1.5 (witness), Pia +0.7 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9; Ana rejects the other on Ana–Marta (rejection +0.09; the tie stays open) |
| 101 | Bruno | CUTOFF | Ana | — | arrives week 102; anxiety Ana +0.8, Bruno +0.9; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 101 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 102; anxiety Marta +1.1, Nadia +2.1 (witness), Pia +1.8, Ravi +2.6 (witness); 9.1 of the sender's anxiety bound into Marta–Pia |
| 101 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 101 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 101 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | arrives week 103; anxiety Marta +1.3 (witness), Nadia +1.4 (witness), Pia +0.6 (witness), Ravi +0.4, Sofia +0.8; 12.1 of the sender's anxiety bound into Ravi–Sofia |
| 101 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.1 (witness), Nadia +1.2 (witness), Pia +0.5 (witness) |
| 101 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 101 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -2.2, Marta +1.1, Ravi +1.1) |
| 102 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 102 | — | (system) | — | — | Ravi–Sofia: a negative contact renews the cutoff's held anxiety to its peak +1.2 |
| 102 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.7 |
| 102 | Bruno | CUTOFF | Ana | — | arrives week 103; anxiety Ana +1.2, Bruno +0.9; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 102 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Marta +2.9; anxiety Nadia +1.6 (witness), Pia +0.7 (witness), Ravi +1.0 (witness) |
| 102 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +5.9; anxiety Marta +2.9 (witness), Pia +0.7 (witness) |
| 102 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Pia +2.3; anxiety Nadia +1.7 (witness), Ravi +2.1 (witness) |
| 102 | Ravi | WITHHOLD | Sofia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 102 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 102 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 102 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -0.7, Marta +0.4, Ravi +0.4) |
| 102 | — | (system) | — | — | Ravi's symptom gives back 73.3 (Marta +22.0, Nadia +18.3, Pia +18.3, Sofia +14.7) |
| 103 | — | (system) | — | — | Ana–Marta: a negative contact renews the cutoff's held anxiety to its peak +0.1 |
| 103 | — | (system) | — | — | Marta–Pia: a negative contact renews the cutoff's held anxiety to its peak +2.1 |
| 103 | — | (system) | — | — | Nadia–Ravi: a negative contact renews the cutoff's held anxiety to its peak +2.1 |
| 103 | Ana | OVERFUNCTION | Bruno | — | — |
| 103 | Bruno | CONFLICT | Ana | — | — |
| 103 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | — |
| 103 | Nadia | CUTOFF | Marta | Pia, Ravi | — |
| 103 | Pia | TRIANGLE | Ravi | Marta, Nadia | — |
| 103 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | — |
| 103 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | — |
| 103 | — | (system) | — | — | Nadia's symptom gives back 0.0 (Marta +0.0, Ravi +0.0) |
| 103 | — | (system) | — | — | Nadia's symptom is past its turn and holds their anxiety no longer (Nadia -3.4, Marta +1.7, Ravi +1.7) |
| 103 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |

104 weeks. The invariants were asserted at the end of every week.
