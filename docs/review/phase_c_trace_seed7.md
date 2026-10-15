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
| config hash | `ee730a7ac8010083c25d0986fc299fbbb1ca70a9bdbeb607c0a352d77d604034` |
| spec revision | 2.0, revision 29 |
| family instance | phase_c |
| activation | synchronous_activation v1 (synchronous) |
| visibility | household_conductance_visibility v1 |
| constants frozen | 2026-10-07 |
| constants changed since freeze | acting_age_years, attribution_rate, belief_appraisal_gain, birth_focus_gain, budget_per_undifferentiation, channel_protection, coach_retirement_age_years, coach_start_age_years, constitution_strength, danger_gain, danger_threshold, danger_window, death_transfer_share, declared_rejection, development_band, development_end_age_years, development_rate, development_start_age_years, estimator_breadth, estimator_rate, estimator_sd_weight, estimator_window, exchange_window, find_work_pressure_scale, find_work_rate, focus_compound_gain, focus_rate, focus_reinforce_gain, focus_steer_gain, force_anxiety_gain, grief_extra_weeks, grief_min_weeks, grief_scale, income_loss_intensity, income_shortfall_share, launch_age_years, lock_in_gain, lock_in_turn, marital_distance_fraction, mortality_a, mortality_b, new_tie_investment_share, origin_cutoff_anxiety_gain, origin_fusion_gain, pattern_min_acts, pattern_projection_hold, pattern_window, perceived_weakness_gain, perspective_channel_gain, perspective_steer_gain, practice_window, pseudo_self_relaxation_rate, reactive_cutoff_gain, refusal_cost_gain, rejection_cutoff_gain, rejection_fade, rejection_refusal_gain, rejection_refusal_threshold, rejection_relief, response_down_gain, response_relaxation_rate, response_up_gain, self_practice_gain, soothing_review_every, soothing_review_penalty, stress_floor_below_basic, stressor_persistence, system_adjust_gain, tie_wear_rate, usual_level_rate, wear_offset_per_act, witness_window, witnessed_cutoff_gain, witnessed_gain, working_age_years |

The standing load, the decay toward each person's floor, the relaxation of felt contact, symptom accumulation, a symptom's hold on the family's anxiety (lock-in), investment, the reactive detectors, outside-ness, attribution, the family's focus and borrowed pseudo-self settling back run every week for everyone and are not listed line by line. Effects are shown beside the event that caused them.

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
| 1 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 2; anxiety Marta +1.8 (witness), Nadia +0.9 (witness), Pia +0.4, Ravi +3.6; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 1 | Ravi | WITHHOLD | Nadia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 1 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 3; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +3.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6 |
| 2 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 2 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 3; anxiety Ana +0.3, Marta -0.7, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.3, Marta -0.3 |
| 2 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.11); the sender bears the refusal: anxiety Bruno +2.9 |
| 2 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 3; anxiety Dr Halim +0.1, Marta +0.2, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +1.1 (witness) |
| 2 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.11); the sender bears the refusal: anxiety Nadia +2.7; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 2 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 3; anxiety Marta +2.0 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +2.6; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia takes 2.2 of functioning from the other |
| 2 | Ravi | STAY-IN-CONTACT | Nadia | Marta, Pia | arrives week 3; anxiety Marta +1.3 (witness), Nadia -0.7, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.2, Ravi +0.2 |
| 2 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 4; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +2.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 1.2 of the sender's anxiety bound into Ravi–Sofia |
| 3 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 3 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 4; anxiety Ana +0.4, Marta +2.6, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.3, Marta -0.3; 2.4 of the sender's anxiety bound into Ana–Marta |
| 3 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 3 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 4; anxiety Marta +0.6, Nadia +1.6 (witness), Pia +1.3 (witness), Ravi +2.1; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3 |
| 3 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.7 (witness), Ravi +2.2 (witness) |
| 3 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 4; anxiety Marta +2.8, Nadia +0.8 (witness), Pia +0.5, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5 |
| 3 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 4; anxiety Marta +2.7 (witness), Nadia +1.2 (witness), Pia +4.4, Ravi +0.8 |
| 3 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 5; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +3.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia takes 2.1 of functioning from the other |
| 4 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +3.7 |
| 4 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.2, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +1.8 (witness) |
| 4 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 4 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 4 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.11); the sender bears the refusal: anxiety Ana +2.4 |
| 4 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.11); the sender bears the refusal: anxiety Bruno +3.1 |
| 4 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 5; anxiety Marta +0.7, Nadia +1.6, Pia +0.8 (witness), Ravi +2.4 (witness); calmer sender takes some anxiety: Marta +0.2, Nadia -0.2; Marta rejects the other on Marta–Nadia (rejection +0.38; the tie stays open) |
| 4 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 5; anxiety Marta +1.8 (witness), Nadia +0.7, Pia +0.7 (witness), Ravi +3.2; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; 8.9 of the sender's anxiety bound into Nadia–Ravi |
| 4 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 5; anxiety Marta +2.5 (witness), Nadia +1.1 (witness), Pia +0.5, Ravi +2.6; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Ravi left outside: anxiety Pia -1.4, Ravi +2.2 (of which 0.7 the outsider's own) |
| 4 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 5; anxiety Marta +2.8 (witness), Nadia +1.3 (witness), Pia +1.8, Ravi +0.9 |
| 4 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 6; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +1.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; 1.5 of the sender's anxiety bound into Ravi–Sofia |
| 5 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 5 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 5 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +0.4, Marta +3.3, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana takes 2.3 of functioning from the other |
| 5 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.08); the sender bears the refusal: anxiety Bruno +3.2 |
| 5 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +3.8, Marta +0.5, Nadia +1.1 (witness), Pia +0.9 (witness), Ravi +1.4 (witness) |
| 5 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 6; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +1.5; calmer sender takes some anxiety: Nadia +0.7, Ravi -0.7 |
| 5 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 6; anxiety Marta +2.5 (witness), Nadia +1.2 (witness), Pia +0.5, Ravi +2.5; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Ravi takes 2.7 of functioning from the other |
| 5 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 6; anxiety Marta +3.0 (witness), Nadia +1.4 (witness), Pia +1.8, Ravi +1.0 |
| 5 | Sofia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 5 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 6, 7; symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.6, Nadia +2.4, Pia +2.0; anxiety Sofia +1.3 |
| 6 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 6 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.12); the sender bears the refusal: anxiety Ana +2.5 |
| 6 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.3 |
| 6 | Marta | TRIANGLE | Pia | Nadia, Ravi | arrives week 7; anxiety Marta +0.9, Nadia +1.0 (witness), Pia +0.5, Ravi +2.4 (witness); Pia left outside: anxiety Marta -3.6, Pia +5.4 (of which 1.8 the outsider's own) |
| 6 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 7; anxiety Marta +2.2 (witness), Nadia +0.6, Pia +0.8 (witness), Ravi +6.7; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Nadia takes 2.7 of functioning from the other |
| 6 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 7; anxiety Marta +2.5 (witness), Nadia +1.1 (witness), Pia +0.6, Ravi +2.1; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Ravi takes 2.8 of functioning from the other |
| 6 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 7; anxiety Dr Halim +0.4, Marta +1.6 (witness), Nadia +1.0 (witness), Pia +0.9 (witness), Ravi +0.3 |
| 6 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 8; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +2.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1 |
| 6 | Marta | SYMPTOM_ONSET | Ana, Dr Halim, Nadia, Pia, Ravi | — | arrives week 7; symptom onset, physical channel; anxiety Ana +1.5, Dr Halim +0.5, Nadia +2.4, Pia +2.1, Ravi +3.0 |
| 7 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 7 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 8; anxiety Ana +0.4, Marta +4.0, Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana takes 2.5 of functioning from the other |
| 7 | Bruno | CUTOFF | Ana | — | arrives week 8; anxiety Ana +0.9, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.07; the tie stays open) |
| 7 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | arrives week 8; anxiety Marta +0.8, Nadia +0.1, Pia +0.9 (witness), Ravi +2.4 (witness); Nadia takes 2.9 of functioning from the other |
| 7 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 8; anxiety Marta +1.6 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +1.4; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 7 | Pia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 7 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 8; anxiety Marta +3.3 (witness), Nadia +1.3 (witness), Pia +3.4, Ravi +1.0; Pia holds a position: no functioning changes hands |
| 7 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +2.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 8 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.4 (witness), Pia +1.3 (witness), Ravi +3.6 |
| 8 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.4, Nadia +1.4 (witness), Pia +1.3 (witness), Ravi +1.8 (witness) |
| 8 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.4, Marta +3.6, Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 8 | Bruno | CUTOFF | Ana | — | arrives week 9; anxiety Ana +1.0, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 8 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Dr Halim +0.1, Marta +1.1 (witness), Nadia +0.6 (witness), Pia +0.6 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Ravi -0.7 |
| 8 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.9, Marta +0.6, Nadia +1.1 (witness), Pia +1.1 (witness), Ravi +1.5 (witness); Ana takes 3.1 of functioning from the other |
| 8 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 9; anxiety Marta +2.9 (witness), Nadia +0.6, Pia +1.0 (witness), Ravi +3.6; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5; Nadia takes 2.7 of functioning from the other |
| 8 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 9; anxiety Marta +2.0, Nadia +0.8 (witness), Pia +0.6, Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9; 12.5 of the sender's anxiety bound into Marta–Pia |
| 8 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 9; anxiety Marta +4.2 (witness), Nadia +1.4 (witness), Pia +2.0, Ravi +1.0; Pia holds a position: no functioning changes hands |
| 8 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 10; anxiety Marta +1.2 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; 2.1 of the sender's anxiety bound into Ravi–Sofia |
| 9 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 10; anxiety Ana +0.3, Marta +3.7, Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9 |
| 9 | Bruno | CUTOFF | Ana | — | arrives week 10; anxiety Ana +0.8, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 9 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 10; anxiety Marta +1.2, Nadia +4.1, Pia +1.2 (witness), Ravi +3.3 (witness); Marta takes 3.2 of functioning from the other |
| 9 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 10; anxiety Marta +2.2 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.5; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 9 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 10; anxiety Marta +3.6 (witness), Nadia +1.0 (witness), Pia +0.6, Ravi +6.9; calmer sender takes some anxiety: Pia +1.4, Ravi -1.4 |
| 9 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 10; anxiety Marta +4.5 (witness), Nadia +1.2 (witness), Pia +2.0, Ravi +1.0; Pia left outside: anxiety Pia +8.7, Ravi -5.8 (of which 2.9 the outsider's own) |
| 9 | Sofia | PROVOKE | Ravi | Marta, Nadia, Pia | arrives week 11; anxiety Marta +1.0 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +3.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1 |
| 10 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.2 |
| 10 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.3 |
| 10 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 11; anxiety Marta +1.1, Nadia +0.9 (witness), Pia +3.4, Ravi +2.7 (witness); 26.5 of the sender's anxiety bound into Marta–Pia |
| 10 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 11; anxiety Marta +2.7 (witness), Nadia +0.6, Pia +0.9 (witness), Ravi +1.9; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Ravi left outside: anxiety Nadia -1.8, Ravi +2.7 (of which 0.9 the outsider's own) |
| 10 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 11; anxiety Marta +3.3 (witness), Nadia +1.1 (witness), Pia +0.7, Ravi +2.0; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; 13.3 of the sender's anxiety bound into Pia–Ravi |
| 10 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 11; anxiety Marta +7.8, Nadia +2.1 (witness), Pia +2.2 (witness), Ravi +1.0; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3 |
| 10 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 12; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; Sofia rejects the other on Ravi–Sofia (rejection +0.54; the tie stays open) |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 11 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 12; anxiety Ana +0.3, Marta +0.1, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 11 | Bruno | CUTOFF | Ana | — | arrives week 12; anxiety Ana +0.8, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 11 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 12; anxiety Marta +0.9, Nadia +1.0 (witness), Pia +3.5, Ravi +2.9 (witness) |
| 11 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 12; anxiety Marta +3.0 (witness), Nadia +0.6, Pia +1.0 (witness), Ravi +4.2; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5; Nadia takes 2.7 of functioning from the other |
| 11 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 11 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 12; anxiety Marta +2.4, Nadia +2.5 (witness), Pia +2.6 (witness), Ravi +1.0 |
| 11 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 13; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3; Sofia rejects the other on Ravi–Sofia (rejection +0.51; the tie stays open) |
| 12 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.3 (witness), Pia +1.3 (witness), Ravi +3.8 |
| 12 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.0, Nadia +1.3 (witness), Pia +1.3 (witness), Ravi +1.9 (witness) |
| 12 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +0.3, Marta +1.3, Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Ana rejects the other on Ana–Marta (rejection +0.61; the tie stays open); impingement removed at once, anxiety Ana -4.2 |
| 12 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.4 |
| 12 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +4.2, Marta +0.6, Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +1.6 (witness); Marta takes 3.0 of functioning from the other |
| 12 | Nadia | PREVENT_ALIGNMENT | Ravi | Marta, Pia | arrives week 13; anxiety Marta +2.0 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.5; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 12 | Pia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 12 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 13; anxiety Marta +3.4 (witness), Nadia +1.1 (witness), Pia +2.6, Ravi +1.1; Ravi rejects the other on Pia–Ravi (rejection +0.83; the tie stays open); impingement removed at once, anxiety Ravi -4.0 |
| 12 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 14; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; 2.4 of the sender's anxiety bound into Ravi–Sofia |
| 13 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 13 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.3 |
| 13 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.11); the sender bears the refusal: anxiety Bruno +3.4 |
| 13 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 14; anxiety Marta +0.9, Nadia +2.0 (witness), Pia +2.2 (witness), Ravi +2.4; calmer sender takes some anxiety: Marta +0.0, Ravi -0.0; Ravi left outside: anxiety Marta -3.4, Ravi +5.1 (of which 1.7 the outsider's own) |
| 13 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 14; anxiety Marta +2.4 (witness), Nadia +0.6, Pia +0.9 (witness), Ravi +1.9; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Nadia rejects the other on Nadia–Ravi (rejection +0.70; the tie stays open) |
| 13 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 14; anxiety Marta +1.2, Nadia +0.9 (witness), Pia +0.6, Ravi +2.9 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9; Pia rejects the other on Marta–Pia (rejection +0.70; the tie stays open); impingement removed at once, anxiety Pia -2.7 |
| 13 | Ravi | PREVENT_ALIGNMENT | Marta | Nadia, Pia | arrives week 14; anxiety Marta +1.8, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +0.7 |
| 13 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 15; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; Ravi takes 2.3 of functioning from the other |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 14 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 14 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 14 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 14 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 15; anxiety Ana +0.3, Marta +0.9, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Marta takes 2.4 of functioning from the other |
| 14 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 14 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.15); the sender bears the refusal: anxiety Marta +4.4; anxiety Nadia +0.9 (witness), Ravi +2.9 (witness) |
| 14 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +1.0 (witness), Ravi +2.9 (witness) |
| 14 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.14); the sender bears the refusal: anxiety Pia +3.1; anxiety Marta +3.0 (witness), Nadia +1.0 (witness) |
| 14 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.15); the sender bears the refusal: anxiety Ravi +3.1; anxiety Marta +1.3 (witness), Nadia +0.9 (witness), Pia +1.0 (witness) |
| 14 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 16; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +4.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 15 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.4 |
| 15 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 15 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 16; anxiety Marta +0.8, Nadia +2.3 (witness), Pia +2.5 (witness), Ravi +8.2; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3 |
| 15 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.9 (witness), Ravi +2.6 (witness) |
| 15 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.14); the sender bears the refusal: anxiety Pia +3.1; anxiety Marta +3.0 (witness), Nadia +1.1 (witness) |
| 15 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 16; anxiety Marta +3.4 (witness), Nadia +1.3 (witness), Pia +5.1, Ravi +1.0; Ravi takes 3.2 of functioning from the other |
| 15 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 17; anxiety Marta +1.0 (witness), Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +4.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.4, Sofia +1.4 |
| 16 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.2 (witness), Pia +1.3 (witness), Ravi +3.8 |
| 16 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.4, Nadia +1.2 (witness), Pia +1.3 (witness), Ravi +1.9 (witness) |
| 16 | Ana | CUTOFF | Bruno | — | arrives week 17; anxiety Ana +0.5, Bruno +0.7; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 16 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 16 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 17; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7 |
| 16 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 17; anxiety Marta +0.9, Nadia +2.2 (witness), Pia +2.6 (witness), Ravi +2.4; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4; 18.8 of the sender's anxiety bound into Marta–Ravi |
| 16 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 17; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6 |
| 16 | Pia | REDUCE_CUTOFF | Marta | Nadia, Ravi | arrives week 17; anxiety Marta +2.1, Nadia +0.6 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; Pia's rejection -0.25; bound anxiety released to the third party: Ravi +38.9 |
| 16 | Ravi | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 16 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 18; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -2.0, Sofia +2.0; Ravi takes 2.2 of functioning from the other |
| 16 | Bruno | SYMPTOM_ONSET | Ana | — | arrives week 17; symptom onset, social channel; anxiety Ana +2.0 |
| 16 | Pia | SYMPTOM_ONSET | Marta, Ravi | Nadia | arrives week 17; symptom onset, social channel; anxiety Marta +2.8, Nadia +0.8 (witness), Ravi +2.9 |
| 17 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 17 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 17 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 18; anxiety Ana +0.3, Marta -0.5, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.4, Marta -0.4 |
| 17 | Bruno | CUTOFF | Ana | — | arrives week 18; anxiety Ana +0.6, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Bruno -0.3 |
| 17 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 18; anxiety Marta +0.8, Nadia +5.0, Pia +1.1 (witness), Ravi +2.6 (witness); calmer sender takes some anxiety: Marta +0.7, Nadia -0.7; Marta takes 2.7 of functioning from the other |
| 17 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.0; anxiety Pia +1.2 (witness), Ravi +2.8 (witness) |
| 17 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Pia +3.6; anxiety Marta +3.3 (witness), Nadia +1.2 (witness) |
| 17 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.20); the sender bears the refusal: anxiety Ravi +5.7; anxiety Marta +2.9 (witness), Pia +1.3 (witness) |
| 17 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 19; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.5, Sofia +1.5; Ravi takes 2.3 of functioning from the other |
| 17 | Nadia | SYMPTOM_ONSET | Marta, Ravi | Pia | arrives week 18; symptom onset, mental channel; anxiety Marta +2.8, Pia +0.8 (witness), Ravi +2.9 |
| 18 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 18 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.6 |
| 18 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 19; anxiety Marta +0.7, Nadia +0.8 (witness), Pia +1.2, Ravi +2.1 (witness); Marta rejects the other on Marta–Pia (rejection +0.72; the tie stays open); impingement removed at once, anxiety Marta -2.5 |
| 18 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 19; anxiety Marta +0.7, Nadia +0.6, Pia +1.1 (witness), Ravi +2.5 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.74; the tie stays open); impingement removed at once, anxiety Nadia -3.9 |
| 18 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 19; anxiety Marta +3.9, Nadia +0.9 (witness), Pia +0.7, Ravi +2.2 (witness); calmer sender takes some anxiety: Marta -0.3, Pia +0.3; 13.7 of the sender's anxiety bound into Marta–Pia |
| 18 | Ravi | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 18 | Sofia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 19 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 20; anxiety Ana +0.4, Marta +2.3, Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 19 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 19 | Marta | TRIANGLE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.14); the sender bears the refusal: anxiety Marta +3.7; anxiety Pia +1.0 (witness), Ravi +2.1 (witness) |
| 19 | Nadia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 19 | Pia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 19 | Ravi | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 19 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 20 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.1 |
| 20 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.1, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.5 (witness) |
| 20 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 20 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 20 | Ana | CUTOFF | Bruno | — | arrives week 21; anxiety Ana +0.5, Bruno +0.4; calmer sender takes some anxiety: Ana +1.1, Bruno -1.1; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 20 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.7 |
| 20 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 21; anxiety Marta +0.8, Nadia +1.1 (witness), Pia +4.6, Ravi +2.5 (witness) |
| 20 | Nadia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 20 | Pia | DETRIANGLE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.4; anxiety Marta +1.8 (witness), Nadia +0.8 (witness) |
| 20 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | arrives week 21; anxiety Marta +3.0 (witness), Nadia +1.3 (witness), Pia +1.2, Ravi +0.8; Pia holds a position: no functioning changes hands |
| 20 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 22; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +2.2, Sofia +0.4; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1 |
| 21 | Ana | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 21 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.7 |
| 21 | Marta | PREVENT_ALIGNMENT | Ravi | Nadia, Pia | arrives week 22; anxiety Marta +0.5, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.2; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3 |
| 21 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.0; anxiety Pia +1.1 (witness), Ravi +2.3 (witness) |
| 21 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 22; anxiety Marta +2.3 (witness), Nadia +1.0 (witness), Pia +0.7, Ravi +0.6; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Pia rejects the other on Pia–Ravi (rejection +0.69; the tie stays open); impingement removed at once, anxiety Pia -1.3 |
| 21 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.16); the sender bears the refusal: anxiety Ravi +2.4; anxiety Marta +1.4 (witness), Nadia +1.2 (witness), Pia +1.3 (witness) |
| 21 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 23; anxiety Marta +1.0 (witness), Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +3.2, Sofia +0.4; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia takes 2.3 of functioning from the other |
| 22 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 23; anxiety Ana +0.4, Marta +0.4, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana rejects the other on Ana–Marta (rejection +0.59; the tie stays open) |
| 22 | Bruno | CUTOFF | Ana | — | arrives week 23; anxiety Ana +0.5, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open) |
| 22 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 23; anxiety Marta +0.7, Nadia +1.2 (witness), Pia +3.3, Ravi +2.7 (witness) |
| 22 | Nadia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 22 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Pia +3.3; anxiety Nadia +1.1 (witness), Ravi +2.6 (witness) |
| 22 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 23; anxiety Dr Halim +0.2, Marta +1.5 (witness), Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +0.2 |
| 22 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 24; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia rejects the other on Ravi–Sofia (rejection +0.24; the tie stays open) |
| 23 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 23 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 24; anxiety Ana +0.3, Marta -0.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 23 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.6 |
| 23 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 24; anxiety Marta +0.7, Nadia +1.2 (witness), Pia +2.2, Ravi +3.1 (witness) |
| 23 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.1; anxiety Pia +1.2 (witness), Ravi +2.6 (witness) |
| 23 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Pia +3.5; anxiety Marta +2.6 (witness), Nadia +1.1 (witness) |
| 23 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 24; anxiety Marta +6.2, Nadia +2.4 (witness), Pia +2.7 (witness), Ravi +0.9; Marta holds a position: no functioning changes hands |
| 23 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 25; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +2.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 24 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.2 |
| 24 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 24 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.4 |
| 24 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.7 |
| 24 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 25; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 24 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 25; anxiety Marta +0.8, Nadia +1.4 (witness), Pia +2.2, Ravi +3.5 (witness) |
| 24 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.3; anxiety Pia +1.1 (witness), Ravi +2.5 (witness) |
| 24 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Pia +3.7; anxiety Nadia +1.4 (witness), Ravi +3.4 (witness) |
| 24 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 25; anxiety Marta +2.4, Nadia +2.3 (witness), Pia +2.5 (witness), Ravi +0.9; 24.1 of the sender's anxiety bound into Marta–Ravi |
| 24 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 26; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.10; the tie stays open) |
| 25 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 25 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 25 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | arrives week 26; anxiety Ana +0.3, Marta +1.9, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 25 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 25 | Marta | WITHHOLD | Dr Halim | — | held back DISPLACE: computed, not emitted; attention on the tie rises |
| 25 | Nadia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 25 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Pia +3.8; anxiety Nadia +1.2 (witness), Ravi +3.0 (witness) |
| 25 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 27; anxiety Marta +1.0 (witness), Nadia +0.9 (witness), Pia +1.0 (witness), Ravi +0.4, Sofia +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.70; the tie stays open); impingement removed at once, anxiety Ravi -3.0 |
| 25 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 27; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 2.8 of the sender's anxiety bound into Ravi–Sofia |
| 26 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 27; anxiety Ana +0.4, Marta +3.1, Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Ana takes 2.4 of functioning from the other |
| 26 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.6 |
| 26 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.14); the sender bears the refusal: anxiety Marta +3.6; anxiety Pia +1.2 (witness), Ravi +2.7 (witness) |
| 26 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.2; anxiety Pia +1.2 (witness), Ravi +2.7 (witness) |
| 26 | Pia | STAY-IN-CONTACT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.4; anxiety Marta +1.6 (witness), Nadia +0.7 (witness) |
| 26 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 27; anxiety Dr Halim +0.3, Marta +1.4 (witness), Nadia +0.9 (witness), Pia +1.0 (witness), Ravi +0.2 |
| 26 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.8 (witness) |
| 27 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 28; anxiety Ana +0.3, Marta +0.3, Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Marta takes 2.4 of functioning from the other |
| 27 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 27 | Marta | WITHHOLD | Nadia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 27 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 28; anxiety Marta +2.6 (witness), Nadia +0.6, Pia +1.2 (witness), Ravi +5.6 |
| 27 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Pia +3.6; anxiety Marta +2.4 (witness), Nadia +1.0 (witness) |
| 27 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.14); the sender bears the refusal: anxiety Ravi +2.2; anxiety Marta +1.1 (witness), Nadia +1.0 (witness), Pia +1.1 (witness) |
| 27 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 28 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.2 |
| 28 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.2, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 28 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 28 | Ana | CUTOFF | Bruno | — | arrives week 29; anxiety Ana +0.5, Bruno +0.2; calmer sender takes some anxiety: Ana +1.1, Bruno -1.1; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 28 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.7 |
| 28 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 29; anxiety Marta +0.8, Nadia +2.3 (witness), Pia +2.5 (witness), Ravi +5.8 |
| 28 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.2; anxiety Pia +1.4 (witness), Ravi +3.0 (witness) |
| 28 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 29; anxiety Marta +2.2 (witness), Nadia +1.0 (witness), Pia +0.7, Ravi +0.9; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia rejects the other on Pia–Ravi (rejection +0.66; the tie stays open) |
| 28 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 30; anxiety Marta +1.1 (witness), Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +0.5, Sofia +0.6; Ravi rejects the other on Ravi–Sofia (rejection +0.33; the tie stays open) |
| 28 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 29 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.12); the sender bears the refusal: anxiety Ana +2.4 |
| 29 | Bruno | CUTOFF | Ana | — | arrives week 30; anxiety Ana +0.4, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.13; the tie stays open) |
| 29 | Marta | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 29 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 30; anxiety Marta +1.3, Nadia +0.6, Pia +1.2 (witness), Ravi +2.7 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.52; the tie stays open) |
| 29 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 29 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | arrives week 30; anxiety Marta +1.1, Nadia +2.1 (witness), Pia +2.3 (witness), Ravi +0.8; Marta takes 3.1 of functioning from the other |
| 29 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 30 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 31; anxiety Dr Halim +0.2, Marta +0.2, Nadia +0.9 (witness), Pia +1.0 (witness), Ravi +1.5 (witness) |
| 30 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 31; anxiety Marta +2.0 (witness), Nadia +0.6, Pia +1.1 (witness), Ravi +1.9; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Ravi left outside: anxiety Nadia -1.7, Ravi +2.6 (of which 0.9 the outsider's own) |
| 30 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 32; anxiety Marta +1.1 (witness), Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +0.5, Sofia +1.0; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 30 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.8 (witness) |
| 31 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 31 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 31 | Ana | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 31 | Bruno | CUTOFF | Ana | — | arrives week 32; anxiety Ana +0.6, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 31 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 32; anxiety Marta +0.7, Nadia +1.9 (witness), Pia +2.0 (witness), Ravi +2.0; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; Ravi takes 2.8 of functioning from the other |
| 31 | Nadia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 31 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Pia +3.4; anxiety Nadia +1.2 (witness), Ravi +3.1 (witness) |
| 31 | Ravi | CONFLICT | Nadia | Marta, Pia | arrives week 32; anxiety Marta +2.6 (witness), Nadia +4.1, Pia +1.3 (witness), Ravi +0.9 |
| 31 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 33; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia rejects the other on Ravi–Sofia (rejection +0.21; the tie stays open); impingement removed at once, anxiety Sofia -0.5 |
| 31 | — | (system) | — | — | Nadia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 32 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.4 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.4 |
| 32 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.8, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.7 (witness) |
| 32 | Ana | CUTOFF | Bruno | — | arrives week 33; anxiety Ana +0.5, Bruno +0.2; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.08; the tie stays open); impingement removed at once, anxiety Ana -0.3 |
| 32 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.4 |
| 32 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 33; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.51; the tie stays open) |
| 32 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Marta +3.7; anxiety Pia +1.1 (witness), Ravi +2.5 (witness) |
| 32 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 33; anxiety Marta +2.4 (witness), Nadia +0.7, Pia +1.1 (witness), Ravi +1.4; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Nadia rejects the other on Nadia–Ravi (rejection +0.73; the tie stays open); impingement removed at once, anxiety Nadia -4.1 |
| 32 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Pia +3.6; anxiety Nadia +1.1 (witness), Ravi +2.7 (witness) |
| 32 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 33; anxiety Marta +6.5, Nadia +2.5 (witness), Pia +2.7 (witness), Ravi +0.8 |
| 32 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.8 (witness) |
| 33 | Ana | CUTOFF | Bruno | — | arrives week 34; anxiety Ana +0.5, Bruno +0.6; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 33 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 33 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.16); the sender bears the refusal: anxiety Marta +1.2; anxiety Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +1.5 (witness) |
| 33 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.1; anxiety Pia +1.0 (witness), Ravi +2.2 (witness) |
| 33 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 34; anxiety Marta +2.1 (witness), Nadia +0.9 (witness), Pia +0.7, Ravi +0.8; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia rejects the other on Pia–Ravi (rejection +0.25; the tie stays open) |
| 33 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 34; anxiety Dr Halim +0.4, Marta +1.6 (witness), Nadia +1.1 (witness), Pia +1.2 (witness), Ravi +0.2 |
| 33 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.7 (witness) |
| 34 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 34 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 35; anxiety Ana +0.4, Marta +0.1, Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Ana rejects the other on Ana–Marta (rejection +0.60; the tie stays open); impingement removed at once, anxiety Ana -0.3 |
| 34 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 34 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 35; anxiety Ana +0.3, Marta +0.5, Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +1.2 (witness); Marta rejects the other on Ana–Marta (rejection +0.76; the tie stays open); impingement removed at once, anxiety Marta -0.1 |
| 34 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 35; anxiety Marta +1.5 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 34 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Pia +3.6; anxiety Nadia +1.3 (witness), Ravi +3.0 (witness) |
| 34 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Ravi +4.1; anxiety Marta +2.3 (witness), Pia +1.1 (witness) |
| 34 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.9 (witness) |
| 35 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.8; anxiety Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.9 (witness) |
| 35 | Bruno | CUTOFF | Ana | — | arrives week 36; anxiety Ana +0.6, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open); impingement removed at once, anxiety Bruno -0.0 |
| 35 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 36; anxiety Marta +0.7, Nadia +2.6 (witness), Pia +2.8 (witness), Ravi +6.9; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3 |
| 35 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 36; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2 |
| 35 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Pia +3.6; anxiety Marta +2.4 (witness), Nadia +1.1 (witness) |
| 35 | Ravi | WITHHOLD | Sofia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 35 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.0 (witness), Nadia +0.9 (witness), Pia +0.9 (witness) |
| 36 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +3.1 |
| 36 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 36 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 36 | Bruno | CUTOFF | Ana | — | arrives week 37; anxiety Ana +1.1, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 36 | Marta | TRIANGLE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Marta +3.8; anxiety Pia +1.1 (witness), Ravi +2.3 (witness) |
| 36 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.3; anxiety Pia +1.0 (witness), Ravi +2.2 (witness) |
| 36 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 37; anxiety Marta +1.1, Nadia +1.2 (witness), Pia +0.8, Ravi +2.8 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Pia rejects the other on Marta–Pia (rejection +0.77; the tie stays open) |
| 36 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Ravi +4.3; anxiety Marta +2.3 (witness), Pia +1.1 (witness) |
| 36 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.0 (witness), Nadia +0.9 (witness), Pia +0.9 (witness) |
| 37 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 37 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.5 |
| 37 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.8; anxiety Pia +1.2 (witness), Ravi +2.7 (witness) |
| 37 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.3; anxiety Pia +1.2 (witness), Ravi +2.5 (witness) |
| 37 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 38; anxiety Marta +1.5, Nadia +1.1 (witness), Pia +0.7, Ravi +2.5 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Pia rejects the other on Marta–Pia (rejection +0.28; the tie stays open) |
| 37 | Ravi | WITHHOLD | Nadia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 37 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 38 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 38 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.08); the sender bears the refusal: anxiety Bruno +3.5 |
| 38 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +3.8; anxiety Nadia +1.0 (witness), Ravi +2.3 (witness) |
| 38 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 39; anxiety Marta +0.5, Nadia +0.7, Pia +1.1 (witness), Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -0.1, Nadia +0.1; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 38 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 38 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Ravi +4.2; anxiety Marta +2.2 (witness), Pia +1.1 (witness) |
| 38 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +1.0 (witness), Nadia +0.9 (witness), Pia +1.0 (witness) |
| 39 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 39 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 39 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 39 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Marta +3.8; anxiety Pia +1.2 (witness), Ravi +2.5 (witness) |
| 39 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 40; anxiety Marta +0.9, Nadia +0.6, Pia +1.1 (witness), Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -0.2, Nadia +0.2; Nadia rejects the other on Marta–Nadia (rejection +0.00; the tie stays open) |
| 39 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 39 | Ravi | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 39 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.9 (witness) |
| 40 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +3.1 |
| 40 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 40 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.6 |
| 40 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 40 | Dr Halim | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 41; anxiety Dr Halim +0.1, Marta +1.0 (witness), Nadia +0.7 (witness), Pia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.8, Ravi -0.8; Dr Halim rejects the other on Dr Halim–Ravi (rejection +0.51; the tie stays open) |
| 40 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 41; anxiety Marta +0.8, Nadia +2.4 (witness), Pia +2.6 (witness), Ravi +2.0; calmer sender takes some anxiety: Marta +0.5, Ravi -0.5; Ravi left outside: anxiety Marta -2.8, Ravi +4.2 (of which 1.4 the outsider's own) |
| 40 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 41; anxiety Marta +0.8, Nadia +0.6, Pia +1.0 (witness), Ravi +2.2 (witness); calmer sender takes some anxiety: Marta -0.1, Nadia +0.1; Nadia rejects the other on Marta–Nadia (rejection +0.05; the tie stays open) |
| 40 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Pia +3.5; anxiety Nadia +1.1 (witness), Ravi +2.6 (witness) |
| 40 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Ravi +4.2; anxiety Marta +2.1 (witness), Pia +1.0 (witness) |
| 40 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 41 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 41 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 41 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 41 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.9 (witness) |
| 41 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.7 |
| 41 | Marta | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 41 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 42; anxiety Marta +1.8 (witness), Nadia +0.6, Pia +0.9 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5; Nadia rejects the other on Nadia–Ravi (rejection +0.08; the tie stays open) |
| 41 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Pia +3.6; anxiety Marta +2.8 (witness), Nadia +1.3 (witness) |
| 41 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 42; anxiety Marta +1.9, Nadia +2.4 (witness), Pia +2.6 (witness), Ravi +0.9; 20.3 of the sender's anxiety bound into Marta–Ravi |
| 41 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.8 (witness) |
| 42 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.5 |
| 42 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.7 |
| 42 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 43; anxiety Dr Halim +0.2, Marta +0.2, Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +1.5 (witness) |
| 42 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 43; anxiety Marta +0.7, Nadia +0.6, Pia +1.0 (witness), Ravi +2.1 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.10; the tie stays open) |
| 42 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 42 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +3.5; anxiety Marta +1.7 (witness), Pia +0.8 (witness) |
| 42 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 43 | Ana | WITHHOLD | Bruno | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 43 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.7 |
| 43 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 44; anxiety Ana +0.3, Marta +0.4, Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +1.0 (witness); Marta rejects the other on Ana–Marta (rejection +0.45; the tie stays open) |
| 43 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 43 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 44; anxiety Marta +2.4 (witness), Nadia +1.1 (witness), Pia +0.7, Ravi +1.0; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 43 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 44; anxiety Marta +2.0, Nadia +1.9 (witness), Pia +2.0 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3; 12.0 of the sender's anxiety bound into Marta–Ravi |
| 43 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.9 (witness) |
| 44 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +3.1 |
| 44 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 44 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 44 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 44 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 44 | Ana | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 44 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.8 |
| 44 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.8; anxiety Pia +1.2 (witness), Ravi +2.6 (witness) |
| 44 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 45; anxiety Marta +1.8 (witness), Nadia +0.6, Pia +0.9 (witness), Ravi -1.7; Ravi holds a position: no functioning changes hands |
| 44 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 44 | Ravi | WITHHOLD | Nadia | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 44 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +1.0 (witness), Nadia +0.9 (witness), Pia +1.0 (witness) |
| 45 | Ana | WITHHOLD | Bruno | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 45 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.8 |
| 45 | Marta | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 45 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 45 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 46; anxiety Marta +0.7, Nadia +1.0 (witness), Pia +0.7, Ravi +2.4 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Pia rejects the other on Marta–Pia (rejection +0.35; the tie stays open) |
| 45 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 46; anxiety Marta +0.2, Nadia +1.6 (witness), Pia +1.7 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.8, Ravi +0.8; Marta left outside: anxiety Marta +2.3, Ravi -1.5 (of which 0.8 the outsider's own) |
| 45 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.9 (witness) |
| 46 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 46 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 46 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 46 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 46 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 46 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 46 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.12); the sender bears the refusal: anxiety Ana +2.4 |
| 46 | Bruno | CUTOFF | Ana | — | arrives week 47; anxiety Ana +0.6, Bruno +0.8; Bruno rejects the other on Ana–Bruno (rejection +0.08; the tie stays open) |
| 46 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 47; anxiety Marta +0.7, Nadia +2.1 (witness), Pia +2.2 (witness), Ravi +6.5 |
| 46 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.1; anxiety Pia +1.2 (witness), Ravi +2.6 (witness) |
| 46 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Pia +3.5; anxiety Nadia +1.1 (witness), Ravi +2.5 (witness) |
| 46 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 47; anxiety Marta +0.4, Nadia +1.8 (witness), Pia +1.9 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.6, Ravi +0.6; 15.1 of the sender's anxiety bound into Marta–Ravi |
| 46 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.8 (witness) |
| 46 | — | (system) | — | — | Dr Halim–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 47 | Ana | CUTOFF | Bruno | — | arrives week 48; anxiety Ana +0.5; calmer sender takes some anxiety: Ana +1.1, Bruno -1.1; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ana -0.2 |
| 47 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.7 |
| 47 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Marta +3.9; anxiety Pia +1.4 (witness), Ravi +2.9 (witness) |
| 47 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.3; anxiety Pia +1.3 (witness), Ravi +2.7 (witness) |
| 47 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 48; anxiety Marta +2.5 (witness), Nadia +1.2 (witness), Pia +0.7, Ravi +1.2; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1 |
| 47 | Ravi | WITHHOLD | Nadia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 47 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 48 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +3.1 |
| 48 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 48 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 49; anxiety Ana +0.4, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +1.5, Marta -1.5; Ana rejects the other on Ana–Marta (rejection +0.60; the tie stays open) |
| 48 | Bruno | CUTOFF | Ana | — | arrives week 49; anxiety Ana +0.6, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 48 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 49; anxiety Dr Halim +0.1, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7; Marta left outside: anxiety Dr Halim -0.2, Marta +0.3 (of which 0.1 the outsider's own) |
| 48 | Marta | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 48 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 48 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 49; anxiety Marta +1.2, Nadia +1.0 (witness), Pia +0.7, Ravi +2.4 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; Pia rejects the other on Marta–Pia (rejection +0.15; the tie stays open) |
| 48 | Ravi | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 48 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 49 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 49 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 49 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 49 | Bruno | WITHHOLD | Ana | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 49 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 50; anxiety Marta +0.8, Nadia +2.2 (witness), Pia +2.3 (witness), Ravi +1.6; Marta rejects the other on Marta–Ravi (rejection +0.83; the tie stays open); impingement removed at once, anxiety Marta -0.2 |
| 49 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.16); the sender bears the refusal: anxiety Nadia +3.7; anxiety Pia +1.5 (witness), Ravi +3.3 (witness) |
| 49 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 49 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | arrives week 50; anxiety Marta +0.2, Nadia +1.9 (witness), Pia +2.1 (witness), Ravi +0.8; calmer sender takes some anxiety: Marta -0.7, Ravi +0.7; Marta takes 2.9 of functioning from the other |
| 49 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.14); the sender bears the refusal: anxiety Sofia +2.1; anxiety Marta +1.1 (witness), Nadia +1.1 (witness), Pia +1.2 (witness) |
| 50 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 51; anxiety Ana +0.4, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +1.0, Marta -1.0; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 50 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.13); the sender bears the refusal: anxiety Bruno +3.5 |
| 50 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.6; anxiety Nadia +1.3 (witness), Ravi +3.2 (witness) |
| 50 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 50 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 51; anxiety Marta +2.3 (witness), Nadia +1.1 (witness), Pia +0.8, Ravi +0.5; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1; Ravi left outside: anxiety Pia -3.3, Ravi +4.9 (of which 1.6 the outsider's own) |
| 50 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 51; anxiety Marta +1.4 (witness), Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +0.3 |
| 50 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.06); the sender bears the refusal: anxiety Sofia +2.0; anxiety Marta +1.1 (witness), Nadia +1.1 (witness), Pia +1.1 (witness) |
| 51 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 51 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 51 | Ana | CUTOFF | Bruno | — | arrives week 52; anxiety Ana +0.5, Bruno +0.1; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 51 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.07); the sender bears the refusal: anxiety Bruno +3.5 |
| 51 | Marta | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 51 | Nadia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 51 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 52; anxiety Marta +2.0 (witness), Nadia +1.0 (witness), Pia +0.7, Ravi +1.1; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia rejects the other on Pia–Ravi (rejection +0.40; the tie stays open) |
| 51 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Ravi +4.3; anxiety Marta +2.2 (witness), Pia +1.1 (witness) |
| 51 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 51 | Nadia | SYMPTOM_ONSET | Marta, Ravi | Pia | arrives week 52; symptom onset, mental channel; anxiety Marta +2.3, Pia +0.9 (witness), Ravi +2.8 |
| 51 | — | (system) | — | — | Dr Halim–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 51 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |
| 52 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.4 (witness), Nadia +1.4 (witness), Pia +1.4 (witness), Ravi +3.4 |
| 52 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.8, Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +1.7 (witness) |
| 52 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 52 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.8; anxiety Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.9 (witness) |
| 52 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 52 | Marta | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 52 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 53; anxiety Marta +0.4, Nadia +0.7, Pia +1.2 (witness), Ravi +2.9 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.25; the tie stays open) |
| 52 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 53; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +0.7, Ravi +2.7 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Marta takes 3.0 of functioning from the other |
| 52 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +4.4; anxiety Marta +2.1 (witness), Nadia +1.0 (witness) |
| 52 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +0.9 (witness), Nadia +0.9 (witness), Pia +1.1 (witness) |
| 52 | Sofia | SYMPTOM_ONSET | Ravi | Marta, Nadia, Pia | arrives week 54; symptom onset, physical channel; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +1.7 |
| 53 | Ana | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 53 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 53 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 54; anxiety Marta +0.6, Nadia +1.9 (witness), Pia +2.2 (witness), Ravi +1.5; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2; Marta rejects the other on Marta–Ravi (rejection +0.37; the tie stays open) |
| 53 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 54; anxiety Marta +1.8 (witness), Nadia +0.7, Pia +1.1 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Ravi left outside: anxiety Nadia -2.2, Ravi +3.3 (of which 1.1 the outsider's own) |
| 53 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 54; anxiety Marta +1.7 (witness), Nadia +0.9 (witness), Pia +0.8, Ravi +0.9; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 53 | Ravi | WITHHOLD | Pia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 53 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 55; anxiety Marta +0.8 (witness), Nadia +0.9 (witness), Pia +1.0 (witness), Ravi +0.0, Sofia +0.4; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia rejects the other on Ravi–Sofia (rejection +0.63; the tie stays open) |
| 54 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 54 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 54 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 54 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.8; anxiety Nadia +0.9 (witness), Pia +1.0 (witness), Ravi +1.1 (witness) |
| 54 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.6 |
| 54 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 55; anxiety Marta +0.6, Nadia +1.8 (witness), Pia +2.1 (witness), Ravi +1.3; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 54 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.14); the sender bears the refusal: anxiety Nadia +3.2; anxiety Pia +1.2 (witness), Ravi +2.5 (witness) |
| 54 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 55; anxiety Marta +1.7 (witness), Nadia +0.9 (witness), Pia +0.8, Ravi +0.8; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Pia rejects the other on Pia–Ravi (rejection +0.05; the tie stays open) |
| 54 | Ravi | DISTANCE | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +4.4; anxiety Nadia +1.9 (witness), Pia +2.2 (witness) |
| 54 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.8 (witness), Nadia +0.9 (witness), Pia +1.0 (witness) |
| 55 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 55 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.8; anxiety Nadia +0.9 (witness), Pia +1.1 (witness), Ravi +1.2 (witness) |
| 55 | Bruno | CUTOFF | Ana | — | arrives week 56; anxiety Ana +0.8, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.30; the tie stays open) |
| 55 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 56; anxiety Dr Halim +0.1, Marta +0.2, Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +1.6 (witness) |
| 55 | Nadia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 55 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 56; anxiety Marta +5.2, Nadia +1.2 (witness), Pia +0.8, Ravi +3.0 (witness); calmer sender takes some anxiety: Marta -0.3, Pia +0.3; Pia takes 3.0 of functioning from the other |
| 55 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Ravi +4.4; anxiety Marta +1.9 (witness), Pia +1.1 (witness) |
| 55 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 56 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +1.4 (witness), Pia +1.6 (witness), Ravi +3.4 |
| 56 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.6, Nadia +1.4 (witness), Pia +1.6 (witness), Ravi +1.7 (witness) |
| 56 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 57; anxiety Ana +0.4, Marta +0.0, Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Ana rejects the other on Ana–Marta (rejection +0.30; the tie stays open) |
| 56 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.5 |
| 56 | Dr Halim | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 57; anxiety Dr Halim +0.1, Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Ravi -0.7; Dr Halim rejects the other on Dr Halim–Ravi (rejection +0.52; the tie stays open); impingement removed at once, anxiety Dr Halim -0.1 |
| 56 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 57; anxiety Marta +0.7, Nadia +1.8 (witness), Pia +2.0 (witness), Ravi +1.3; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2; Marta rejects the other on Marta–Ravi (rejection +0.10; the tie stays open) |
| 56 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 57; anxiety Marta +0.3, Nadia +0.7, Pia +1.1 (witness), Ravi +2.5 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.20; the tie stays open) |
| 56 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 57; anxiety Marta +2.4, Nadia +1.4 (witness), Pia +0.7, Ravi +3.4 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5 |
| 56 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 57; anxiety Dr Halim +0.1, Marta +1.5 (witness), Nadia +1.1 (witness), Pia +1.2 (witness), Ravi +0.3 |
| 56 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 58; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.2, Sofia +0.4; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia rejects the other on Ravi–Sofia (rejection +0.52; the tie stays open) |
| 57 | Ana | CUTOFF | Bruno | — | arrives week 58; anxiety Ana +0.5, Bruno +0.4; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -1.4 |
| 57 | Bruno | CUTOFF | Ana | — | arrives week 58; anxiety Ana +1.4, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Bruno -0.4 |
| 57 | Marta | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 57 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 58; anxiety Marta +1.9 (witness), Nadia +0.7, Pia +0.9 (witness), Ravi +0.4; Nadia rejects the other on Nadia–Ravi (rejection +0.44; the tie stays open) |
| 57 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 58; anxiety Marta +1.8 (witness), Nadia +0.9 (witness), Pia +0.7, Ravi +1.5; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia rejects the other on Pia–Ravi (rejection +0.15; the tie stays open) |
| 57 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +2.7; anxiety Marta +1.2 (witness), Nadia +1.2 (witness), Pia +1.2 (witness) |
| 57 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.9 (witness), Nadia +0.9 (witness), Pia +0.9 (witness) |
| 58 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 58 | Ana | CUTOFF | Bruno | — | arrives week 59; anxiety Ana +0.5, Bruno +0.3; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 58 | Bruno | WITHHOLD | Ana | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 58 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 59; anxiety Marta +0.7, Nadia +1.7 (witness), Pia +1.8 (witness), Ravi +1.0; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; Marta rejects the other on Marta–Ravi (rejection +0.10; the tie stays open) |
| 58 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 59; anxiety Marta +0.3, Nadia +0.6, Pia +1.0 (witness), Ravi +2.3 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.10; the tie stays open) |
| 58 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 59; anxiety Marta +0.8, Nadia +1.2 (witness), Pia +0.7, Ravi +2.9 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5; Marta takes 3.0 of functioning from the other |
| 58 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +4.2; anxiety Marta +2.1 (witness), Pia +1.1 (witness) |
| 58 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 59 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.4 |
| 59 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.4 |
| 59 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.15); the sender bears the refusal: anxiety Marta +3.2; anxiety Nadia +1.3 (witness), Ravi +3.1 (witness) |
| 59 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.2; anxiety Pia +1.0 (witness), Ravi +2.2 (witness) |
| 59 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 60; anxiety Marta +1.7 (witness), Nadia +0.9 (witness), Pia +0.8, Ravi +1.2; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia rejects the other on Pia–Ravi (rejection +0.10; the tie stays open) |
| 59 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.6; anxiety Marta +1.1 (witness), Nadia +1.1 (witness), Pia +1.3 (witness) |
| 59 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.8 (witness), Nadia +0.9 (witness), Pia +1.0 (witness) |
| 60 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +3.4 |
| 60 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.6, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +1.7 (witness) |
| 60 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +1.2 (witness) |
| 60 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.4 |
| 60 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 61; anxiety Dr Halim +0.2, Marta +0.2, Nadia +1.0 (witness), Pia +1.2 (witness), Ravi +1.7 (witness) |
| 60 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.3; anxiety Pia +1.0 (witness), Ravi +2.2 (witness) |
| 60 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 61; anxiety Marta +2.0 (witness), Nadia +1.1 (witness), Pia +0.8, Ravi +6.8; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Pia takes 3.0 of functioning from the other |
| 60 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +4.4; anxiety Nadia +1.9 (witness), Pia +2.1 (witness) |
| 60 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 62; anxiety Marta +0.7 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 61 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 61 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 61 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.5 |
| 61 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 61 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.15); the sender bears the refusal: anxiety Marta +3.2; anxiety Nadia +1.3 (witness), Ravi +3.4 (witness) |
| 61 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 62; anxiety Marta +0.1, Nadia +0.7, Pia +0.9 (witness), Ravi +2.4 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.15; the tie stays open) |
| 61 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 62; anxiety Marta +1.1, Nadia +1.3 (witness), Pia +0.7, Ravi +3.4 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; 15.7 of the sender's anxiety bound into Marta–Pia |
| 61 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +5.1; anxiety Marta +2.1 (witness), Nadia +1.1 (witness) |
| 61 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.11); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.7 (witness), Nadia +0.8 (witness), Pia +0.8 (witness) |
| 61 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 62, 63; symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.2, Nadia +2.3, Pia +2.3; anxiety Sofia +1.5 |
| 62 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +1.2 (witness) |
| 62 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 62 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 63; anxiety Ana +0.3, Marta +0.4, Nadia +1.1 (witness), Pia +1.2 (witness), Ravi +1.5 (witness); Marta rejects the other on Ana–Marta (rejection +0.34; the tie stays open) |
| 62 | Nadia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 62 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 63; anxiety Marta +1.9 (witness), Nadia +1.0 (witness), Pia +0.6, Ravi +4.7; calmer sender takes some anxiety: Pia +1.7, Ravi -1.7 |
| 62 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.17); the sender bears the refusal: anxiety Ravi +5.1; anxiety Marta +2.1 (witness), Pia +1.1 (witness) |
| 62 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.8 (witness) |
| 63 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.5 |
| 63 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 63 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.15); the sender bears the refusal: anxiety Marta +3.3; anxiety Nadia +1.3 (witness), Ravi +3.6 (witness) |
| 63 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 64; anxiety Marta +2.0 (witness), Nadia +0.7, Pia +1.1 (witness), Ravi +4.3; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 63 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 64; anxiety Marta +5.4, Nadia +1.2 (witness), Pia +0.6, Ravi +3.2 (witness); calmer sender takes some anxiety: Marta -1.1, Pia +1.1 |
| 63 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +5.0; anxiety Marta +2.2 (witness), Nadia +1.2 (witness) |
| 63 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.9 (witness), Nadia +0.9 (witness), Pia +0.9 (witness) |
| 64 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +1.4 (witness), Pia +1.4 (witness), Ravi +3.7 |
| 64 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.6, Nadia +1.4 (witness), Pia +1.4 (witness), Ravi +1.8 (witness) |
| 64 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 64 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 64 | Bruno | CUTOFF | Ana | — | arrives week 65; anxiety Ana +0.6, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 64 | Dr Halim | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 65; anxiety Dr Halim +0.1, Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.9; calmer sender takes some anxiety: Dr Halim +0.8, Ravi -0.8; Dr Halim takes 2.1 of functioning from the other; the contact lands with Ravi: systems perspective +0.10 |
| 64 | Marta | WITHHOLD | Nadia | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 64 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 65; anxiety Marta +2.1 (witness), Nadia +0.7, Pia +1.1 (witness), Ravi +5.1; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4 |
| 64 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 64 | Ravi | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 64 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.9 (witness), Nadia +0.9 (witness), Pia +0.9 (witness) |
| 65 | Ana | CUTOFF | Bruno | — | arrives week 66; anxiety Ana +0.5, Bruno +0.5; calmer sender takes some anxiety: Ana +0.4, Bruno -0.4; Ana rejects the other on Ana–Bruno (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ana -1.3 |
| 65 | Bruno | CUTOFF | Ana | — | arrives week 66; anxiety Ana +1.0, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Bruno -0.5 |
| 65 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Marta +3.3; anxiety Pia +1.1 (witness), Ravi +3.2 (witness) |
| 65 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 66; anxiety Marta +1.9 (witness), Nadia +0.7, Pia +1.0 (witness), Ravi +0.9; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Ravi left outside: anxiety Nadia -1.9, Ravi +2.9 (of which 1.0 the outsider's own) |
| 65 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 66; anxiety Marta +3.4, Nadia +1.3 (witness), Pia +0.6, Ravi +3.7 (witness); calmer sender takes some anxiety: Marta -1.1, Pia +1.1 |
| 65 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 66; anxiety Marta +2.1 (witness), Nadia +1.1 (witness), Pia +0.0, Ravi +1.1; Ravi rejects the other on Pia–Ravi (rejection +0.82; the tie stays open); impingement removed at once, anxiety Ravi -2.8 |
| 65 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 66 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 66 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 66 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ana +2.0; anxiety Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +1.4 (witness) |
| 66 | Bruno | CUTOFF | Ana | — | arrives week 67; anxiety Ana +0.3, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 66 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.16); the sender bears the refusal: anxiety Marta +2.1; anxiety Nadia +1.2 (witness), Pia +1.2 (witness), Ravi +1.7 (witness) |
| 66 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 67; anxiety Marta +1.9 (witness), Nadia +0.6, Pia +1.0 (witness), Ravi +4.8; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6 |
| 66 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 67; anxiety Marta +1.4, Nadia +1.1 (witness), Pia +0.7, Ravi +3.3 (witness); calmer sender takes some anxiety: Marta -1.1, Pia +1.1; 11.9 of the sender's anxiety bound into Marta–Pia |
| 66 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.16); the sender bears the refusal: anxiety Ravi +5.2; anxiety Nadia +2.3 (witness), Pia +2.4 (witness) |
| 66 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.9 (witness), Nadia +1.0 (witness), Pia +1.0 (witness) |
| 67 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.6 |
| 67 | Bruno | CUTOFF | Ana | — | arrives week 68; anxiety Ana +0.8, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 67 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 68; anxiety Marta +0.7, Nadia +2.3 (witness), Pia +2.4 (witness), Ravi +6.8; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4 |
| 67 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 68; anxiety Marta +2.0 (witness), Nadia +0.7, Pia +1.1 (witness), Ravi +2.9; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4 |
| 67 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 68; anxiety Marta +1.3, Nadia +0.9 (witness), Pia +0.6, Ravi +2.7 (witness); calmer sender takes some anxiety: Marta -1.2, Pia +1.2; 9.2 of the sender's anxiety bound into Marta–Pia |
| 67 | Ravi | REDUCE_CUTOFF | Nadia | Marta, Pia | arrives week 68; anxiety Marta +1.4 (witness), Pia +0.7 (witness), Ravi +0.6; Nadia's rejection -0.25; bound anxiety released to the third party: Marta +8.9 |
| 67 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 69; anxiety Marta +0.8 (witness), Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3; Sofia rejects the other on Ravi–Sofia (rejection +0.35; the tie stays open) |
| 67 | — | (system) | — | — | Dr Halim–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 68 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +1.4 (witness), Pia +1.4 (witness), Ravi +3.9 |
| 68 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.6, Nadia +1.4 (witness), Pia +1.4 (witness), Ravi +1.9 (witness) |
| 68 | Ana | CUTOFF | Bruno | — | arrives week 69; anxiety Ana +0.5, Bruno +0.5; calmer sender takes some anxiety: Ana +0.6, Bruno -0.6; Ana rejects the other on Ana–Bruno (rejection +0.03; the tie stays open); impingement removed at once, anxiety Ana -0.4 |
| 68 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.3 |
| 68 | Marta | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 68 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 69; anxiety Marta +1.9 (witness), Nadia +0.7, Pia +1.0 (witness), Ravi +1.2; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5; Ravi left outside: anxiety Nadia -2.1, Ravi +3.2 (of which 1.1 the outsider's own) |
| 68 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 69; anxiety Marta +2.8, Nadia +1.0 (witness), Pia +0.6, Ravi +2.9 (witness); calmer sender takes some anxiety: Marta -1.6, Pia +1.6 |
| 68 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 69; anxiety Marta +0.2, Nadia +2.4 (witness), Pia +2.5 (witness), Ravi +1.1; Ravi rejects the other on Marta–Ravi (rejection +0.87; the tie stays open); impingement removed at once, anxiety Ravi -4.6 |
| 68 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.8 (witness), Nadia +0.9 (witness), Pia +0.9 (witness) |
| 69 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.6 |
| 69 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.3 |
| 69 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.16); the sender bears the refusal: anxiety Marta +3.4; anxiety Nadia +2.0 (witness), Pia +2.1 (witness) |
| 69 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 70; anxiety Marta +2.2 (witness), Nadia +0.7, Pia +1.2 (witness), Ravi +4.2; calmer sender takes some anxiety: Nadia +1.6, Ravi -1.6 |
| 69 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 70; anxiety Marta +1.3, Nadia +1.0 (witness), Pia +0.6, Ravi +2.8 (witness); calmer sender takes some anxiety: Marta -2.6, Pia +2.6; 9.4 of the sender's anxiety bound into Marta–Pia |
| 69 | Ravi | WITHHOLD | Sofia | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 69 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.9 (witness), Nadia +1.0 (witness), Pia +1.0 (witness) |
| 70 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +1.0 (witness), Pia +1.0 (witness), Ravi +1.4 (witness) |
| 70 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.4 |
| 70 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.08); the sender bears the refusal: anxiety Marta +3.8; anxiety Nadia +2.5 (witness), Pia +2.6 (witness) |
| 70 | Nadia | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 70 | Pia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 70 | Ravi | TRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.18); the sender bears the refusal: anxiety Ravi +5.8; anxiety Nadia +2.5 (witness), Pia +2.6 (witness) |
| 70 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 72; anxiety Marta +0.8 (witness), Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +0.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.4, Sofia +1.4; Sofia rejects the other on Ravi–Sofia (rejection +0.15; the tie stays open) |
| 71 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 71 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 71 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 71 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 71 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 71 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.5 |
| 71 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 71 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 72; anxiety Dr Halim +0.3, Marta +0.2, Nadia +1.2 (witness), Pia +1.2 (witness), Ravi +2.2 (witness) |
| 71 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 72; anxiety Marta +0.3, Nadia +0.6, Pia +1.0 (witness), Ravi +2.8 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4; Nadia rejects the other on Marta–Nadia (rejection +0.35; the tie stays open) |
| 71 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 72; anxiety Marta +1.0, Nadia +0.9 (witness), Pia +0.6, Ravi +2.6 (witness); calmer sender takes some anxiety: Marta -1.6, Pia +1.6; 10.9 of the sender's anxiety bound into Marta–Pia |
| 71 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.17); the sender bears the refusal: anxiety Ravi +5.5; anxiety Nadia +2.5 (witness), Pia +2.7 (witness) |
| 71 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 73; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -2.0, Sofia +2.0; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 72 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +3.9 |
| 72 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.6, Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +1.9 (witness) |
| 72 | Ana | WITHHOLD | Bruno | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 72 | Bruno | CUTOFF | Ana | — | arrives week 73; anxiety Ana +0.3, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 72 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 73; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7 |
| 72 | Marta | WITHHOLD | Ana | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 72 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 73; anxiety Marta +2.1 (witness), Nadia +0.7, Pia +1.1 (witness), Ravi +1.2; calmer sender takes some anxiety: Nadia +1.6, Ravi -1.6; Nadia rejects the other on Nadia–Ravi (rejection +0.75; the tie stays open) |
| 72 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 72 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.18); the sender bears the refusal: anxiety Ravi +5.6; anxiety Marta +2.1 (witness), Nadia +1.1 (witness) |
| 72 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 73 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 73 | Bruno | CUTOFF | Ana | — | arrives week 74; anxiety Ana +0.8, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 73 | Marta | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 73 | Nadia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 73 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 74; anxiety Marta +0.8, Nadia +0.8 (witness), Pia +0.6, Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -1.3, Pia +1.3; Marta takes 2.6 of functioning from the other |
| 73 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.4; anxiety Marta +1.2 (witness), Nadia +1.3 (witness), Pia +1.4 (witness) |
| 73 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 75; anxiety Marta +0.8 (witness), Nadia +0.9 (witness), Pia +1.0 (witness), Ravi +0.8, Sofia +0.4; calmer sender takes some anxiety: Ravi -1.5, Sofia +1.5; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 74 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 74 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 74 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 74 | Ana | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 74 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.4 |
| 74 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 75; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +4.4, Ravi +3.2 (witness) |
| 74 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 75; anxiety Marta -0.0, Nadia +0.6, Pia +1.0 (witness), Ravi +2.5 (witness); Marta takes 2.8 of functioning from the other |
| 74 | Pia | TRIANGLE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.13); the sender bears the refusal: anxiety Pia +3.2; anxiety Marta +1.4 (witness), Nadia +0.8 (witness) |
| 74 | Ravi | TRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.17); the sender bears the refusal: anxiety Ravi +5.3; anxiety Nadia +2.3 (witness), Pia +2.6 (witness) |
| 74 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.8 (witness), Nadia +1.0 (witness), Pia +1.0 (witness) |
| 75 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 75 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 75 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 75 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 75 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 75 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 75 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +1.1 (witness), Pia +1.1 (witness), Ravi +1.4 (witness) |
| 75 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.08); the sender bears the refusal: anxiety Bruno +3.5 |
| 75 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.15); the sender bears the refusal: anxiety Marta +2.8; anxiety Pia +1.2 (witness), Ravi +3.1 (witness) |
| 75 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 75 | Pia | TRIANGLE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.14); the sender bears the refusal: anxiety Pia +3.6; anxiety Marta +1.5 (witness), Nadia +0.9 (witness) |
| 75 | Ravi | STAY-IN-CONTACT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.9; anxiety Marta +0.6 (witness), Nadia +0.8 (witness), Pia +0.8 (witness) |
| 75 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +0.9 (witness) |
| 76 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.2 (witness), Nadia +1.5 (witness), Pia +1.5 (witness), Ravi +3.8 |
| 76 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.4, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +1.9 (witness) |
| 76 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 76 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.8; anxiety Nadia +1.0 (witness), Pia +1.1 (witness), Ravi +1.4 (witness) |
| 76 | Bruno | WITHHOLD | Ana | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 76 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 77; anxiety Marta +0.6, Nadia +1.3 (witness), Pia +3.1, Ravi +3.3 (witness) |
| 76 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 77; anxiety Marta +2.0 (witness), Nadia +0.7, Pia +1.3 (witness), Ravi +8.4; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6 |
| 76 | Pia | TRIANGLE | Marta | Nadia, Ravi | arrives week 77; anxiety Marta +0.4, Nadia +1.2 (witness), Pia +0.7, Ravi +3.2 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Marta left outside: anxiety Marta +4.2, Pia -2.8 (of which 1.4 the outsider's own) |
| 76 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +3.1; anxiety Marta +1.1 (witness), Nadia +1.3 (witness), Pia +1.4 (witness) |
| 76 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 78; anxiety Marta +0.8 (witness), Nadia +0.9 (witness), Pia +1.0 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.5, Sofia +1.5; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 77 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 77 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.8; anxiety Nadia +1.0 (witness), Pia +1.0 (witness), Ravi +1.2 (witness) |
| 77 | Bruno | CUTOFF | Ana | — | arrives week 78; anxiety Ana +0.4, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open) |
| 77 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 78; anxiety Marta +0.6, Nadia +1.3 (witness), Pia +1.4, Ravi +3.3 (witness) |
| 77 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 78; anxiety Marta +0.5, Nadia +0.7, Pia +1.1 (witness), Ravi +2.8 (witness); Marta takes 2.9 of functioning from the other |
| 77 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 78; anxiety Marta +0.6, Nadia +1.2 (witness), Pia +0.7, Ravi +3.2 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; 14.7 of the sender's anxiety bound into Marta–Pia |
| 77 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +3.2; anxiety Marta +1.2 (witness), Nadia +1.7 (witness), Pia +1.6 (witness) |
| 77 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.8 (witness), Nadia +1.1 (witness), Pia +1.1 (witness) |
| 78 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 78 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 78 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.4 |
| 78 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.4 |
| 78 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +2.7 (witness), Pia +2.6 (witness) |
| 78 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 79; anxiety Marta +1.8 (witness), Nadia +0.8, Pia +1.2 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.8, Ravi -0.8; Ravi left outside: anxiety Nadia -1.9, Ravi +2.9 (of which 1.0 the outsider's own) |
| 78 | Pia | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 78 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +3.2; anxiety Marta +1.1 (witness), Nadia +1.5 (witness), Pia +1.4 (witness) |
| 78 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 80; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +1.0 (witness), Ravi +0.8, Sofia +0.4; calmer sender takes some anxiety: Ravi -1.5, Sofia +1.5; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 79 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 79 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 79 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.8; anxiety Nadia +1.1 (witness), Pia +1.1 (witness), Ravi +1.4 (witness) |
| 79 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.08); the sender bears the refusal: anxiety Bruno +3.4 |
| 79 | Marta | WITHHOLD | Dr Halim | — | held back DISPLACE: computed, not emitted; attention on the tie rises |
| 79 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 79 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.14); the sender bears the refusal: anxiety Pia +3.4; anxiety Marta +1.6 (witness), Nadia +1.1 (witness) |
| 79 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.17); the sender bears the refusal: anxiety Ravi +5.2; anxiety Nadia +2.8 (witness), Pia +2.7 (witness) |
| 79 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.8 (witness), Nadia +1.1 (witness), Pia +1.1 (witness) |
| 80 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.1 (witness), Nadia +1.6 (witness), Pia +1.5 (witness), Ravi +3.7 |
| 80 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.2, Nadia +1.5 (witness), Pia +1.5 (witness), Ravi +1.9 (witness) |
| 80 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 80 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.4 |
| 80 | Bruno | CUTOFF | Ana | — | arrives week 81; anxiety Ana +0.5, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 80 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 81; anxiety Dr Halim +0.1, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7; the contact lands with Marta: systems perspective +0.10 |
| 80 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 81; anxiety Marta +0.6, Nadia +2.3 (witness), Pia +2.3 (witness), Ravi +2.4; calmer sender takes some anxiety: Marta +1.0, Ravi -1.0; Marta rejects the other on Marta–Ravi (rejection +0.14; the tie stays open) |
| 80 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 81; anxiety Marta +4.7, Nadia +0.8, Pia +1.3 (witness), Ravi +3.3 (witness); Nadia takes 3.1 of functioning from the other |
| 80 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Pia +3.7; anxiety Marta +2.0 (witness), Nadia +1.4 (witness) |
| 80 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.17); the sender bears the refusal: anxiety Ravi +5.4; anxiety Marta +1.9 (witness), Pia +1.3 (witness) |
| 80 | Sofia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 80 | — | (system) | — | — | Marta–Pia: rejection has faded; the tie is no longer cut off (ordinary) |
| 81 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 81 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 81 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 81 | Marta | DETRIANGLE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +1.9; anxiety Pia +0.8 (witness), Ravi +2.0 (witness) |
| 81 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 82; anxiety Marta +0.8, Nadia +0.7, Pia +1.2 (witness), Ravi +3.0 (witness); Marta left outside: anxiety Marta +3.5, Nadia -2.3 (of which 1.2 the outsider's own) |
| 81 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 82; anxiety Marta +0.8, Nadia +1.2 (witness), Pia +0.8, Ravi +3.0 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; 15.1 of the sender's anxiety bound into Marta–Pia |
| 81 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.17); the sender bears the refusal: anxiety Ravi +5.3; anxiety Marta +2.2 (witness), Pia +1.4 (witness) |
| 81 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.9 (witness), Nadia +1.1 (witness), Pia +1.1 (witness) |
| 82 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 82 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +1.0 (witness), Pia +1.0 (witness), Ravi +1.3 (witness) |
| 82 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.3 |
| 82 | Marta | TRIANGLE | Pia | Nadia, Ravi | arrives week 83; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.1, Ravi +2.9 (witness); Pia left outside: anxiety Marta -4.1, Pia +6.2 (of which 2.1 the outsider's own) |
| 82 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 83; anxiety Marta +1.7 (witness), Nadia +0.7, Pia +1.1 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +1.0, Ravi -1.0; Nadia rejects the other on Nadia–Ravi (rejection +0.01; the tie stays open) |
| 82 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 83; anxiety Marta +0.8, Nadia +0.9 (witness), Pia +0.7, Ravi +2.4 (witness); calmer sender takes some anxiety: Marta -1.3, Pia +1.3; Marta takes 2.6 of functioning from the other |
| 82 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.17); the sender bears the refusal: anxiety Ravi +5.4; anxiety Marta +2.0 (witness), Pia +1.3 (witness) |
| 82 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.8 (witness), Nadia +1.0 (witness), Pia +1.1 (witness) |
| 83 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 83 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 83 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 83 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 83 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.5 |
| 83 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.08); the sender bears the refusal: anxiety Bruno +3.4 |
| 83 | Marta | TRIANGLE | Pia | Nadia, Ravi | arrives week 84; anxiety Marta +0.6, Nadia +1.0 (witness), Pia +1.9, Ravi +2.6 (witness); Pia left outside: anxiety Marta -3.2, Pia +4.9 (of which 1.6 the outsider's own) |
| 83 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 84; anxiety Marta +1.5 (witness), Nadia +0.7, Pia +1.1 (witness), Ravi +1.0; calmer sender takes some anxiety: Nadia +0.7, Ravi -0.7; Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 83 | Pia | TRIANGLE | Marta | Nadia, Ravi | arrives week 84; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +0.8, Ravi +2.5 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Marta left outside: anxiety Marta +4.5, Pia -3.0 (of which 1.5 the outsider's own) |
| 83 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 84; anxiety Marta +2.1 (witness), Nadia +1.4 (witness), Pia +0.8, Ravi +1.1; Ravi rejects the other on Pia–Ravi (rejection +0.52; the tie stays open) |
| 83 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 85; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +1.0 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.4, Sofia +1.4; Sofia rejects the other on Ravi–Sofia (rejection +0.25; the tie stays open) |
| 84 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.1 (witness), Nadia +1.4 (witness), Pia +1.6 (witness), Ravi +3.7 |
| 84 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.3, Nadia +1.4 (witness), Pia +1.6 (witness), Ravi +1.8 (witness) |
| 84 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 84 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 84 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 84 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Marta +2.8; anxiety Nadia +2.1 (witness), Pia +2.3 (witness) |
| 84 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 85; anxiety Marta +0.6, Nadia +0.7, Pia +1.2 (witness), Ravi +2.7 (witness); calmer sender takes some anxiety: Marta -0.0, Nadia +0.0; Marta holds a position: no functioning changes hands |
| 84 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 85; anxiety Marta +0.7, Nadia +1.0 (witness), Pia +0.8, Ravi +2.5 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; 15.6 of the sender's anxiety bound into Marta–Pia |
| 84 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 86; anxiety Marta +1.1 (witness), Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +0.6, Sofia +0.4; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open); impingement removed at once, anxiety Ravi -0.2 |
| 84 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +1.0 (witness) |
| 85 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +0.9 (witness), Pia +1.0 (witness), Ravi +1.2 (witness) |
| 85 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +3.4 |
| 85 | Marta | PURSUE | Pia | Nadia, Ravi | arrives week 86; anxiety Marta +0.6, Nadia +1.0 (witness), Pia +5.5, Ravi +2.6 (witness); calmer sender takes some anxiety: Marta +0.7, Pia -0.7 |
| 85 | Nadia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 85 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 86; anxiety Marta +0.5, Nadia +0.9 (witness), Pia +0.7, Ravi +2.3 (witness); Pia rejects the other on Marta–Pia (rejection +0.66; the tie stays open); impingement removed at once, anxiety Pia -7.1 |
| 85 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.1; anxiety Marta +1.1 (witness), Nadia +1.5 (witness), Pia +1.6 (witness) |
| 85 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.7; anxiety Marta +0.8 (witness), Nadia +1.0 (witness), Pia +1.1 (witness) |
| 86 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 86 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.17); the sender bears the refusal: anxiety Ana +2.5; anxiety Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.7 (witness) |
| 86 | Bruno | CUTOFF | Ana | — | arrives week 87; anxiety Ana +0.3, Bruno +0.7; calmer sender takes some anxiety: Ana -0.7, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 86 | Marta | STAY-IN-CONTACT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +1.8; anxiety Pia +0.8 (witness), Ravi +1.7 (witness) |
| 86 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 87; anxiety Marta +0.5, Nadia +0.9, Pia +1.4 (witness), Ravi +3.1 (witness); Marta holds a position: no functioning changes hands |
| 86 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 87; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +0.9, Ravi +2.8 (witness); Marta takes 3.4 of functioning from the other |
| 86 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +3.5; anxiety Marta +1.2 (witness), Nadia +1.7 (witness), Pia +2.1 (witness) |
| 86 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.8 (witness), Nadia +1.1 (witness), Pia +1.3 (witness) |
| 86 | Ana | SYMPTOM_ONSET | Bruno, Marta | Nadia, Pia, Ravi | arrives week 87; symptom onset, physical channel; anxiety Bruno +2.7, Marta +1.1, Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +1.0 (witness) |
| 87 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 87 | Ana | CUTOFF | Bruno | — | arrives week 88; anxiety Ana +0.6, Bruno +1.0; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ana -0.8 |
| 87 | Bruno | CUTOFF | Ana | — | arrives week 88; anxiety Ana +0.7, Bruno +0.7; calmer sender takes some anxiety: Ana -0.3, Bruno +0.3; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Bruno -1.0 |
| 87 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.15); the sender bears the refusal: anxiety Marta +2.6; anxiety Nadia +1.0 (witness), Ravi +2.6 (witness) |
| 87 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 88; anxiety Marta +4.2, Nadia +0.8, Pia +1.8 (witness), Ravi +3.7 (witness); Marta holds a position: no functioning changes hands |
| 87 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 88; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.0, Ravi +2.9 (witness); Marta–Pia functioning balance flips; Marta takes 3.4 of functioning from the other |
| 87 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 88; anxiety Marta +1.8 (witness), Nadia +1.4, Pia +1.5 (witness), Ravi +1.2; Ravi rejects the other on Nadia–Ravi (rejection +0.93; the tie stays open); impingement removed at once, anxiety Ravi -0.1 |
| 87 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.7 (witness), Nadia +1.1 (witness), Pia +1.5 (witness) |
| 87 | Pia | SYMPTOM_ONSET | Marta, Ravi | Nadia | arrives week 88; symptom onset, social channel; anxiety Marta +1.7, Nadia +0.8 (witness), Ravi +3.0 |
| 88 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.0 (witness), Nadia +1.5 (witness), Pia +1.8 (witness), Ravi +3.7 |
| 88 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.1, Nadia +1.4 (witness), Pia +1.8 (witness), Ravi +1.8 (witness) |
| 88 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +2.3; anxiety Nadia +1.2 (witness), Pia +1.7 (witness), Ravi +1.6 (witness) |
| 88 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.4 |
| 88 | Dr Halim | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 89; anxiety Dr Halim +0.1, Marta +1.1 (witness), Nadia +1.2 (witness), Pia +1.6 (witness), Ravi +1.5; calmer sender takes some anxiety: Dr Halim +0.8, Ravi -0.8; Dr Halim takes 2.8 of functioning from the other |
| 88 | Marta | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 88 | Nadia | TRIANGLE | Ravi | Marta, Pia | refused by Ravi (rejection +0.12); the sender bears the refusal: anxiety Nadia +4.3; anxiety Marta +1.6 (witness), Pia +1.6 (witness) |
| 88 | Pia | DETRIANGLE | Marta | Nadia, Ravi | arrives week 89; anxiety Marta +0.3, Nadia +0.7 (witness), Pia +0.7, Ravi +1.7 (witness) |
| 88 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.19); the sender bears the refusal: anxiety Ravi +5.8; anxiety Marta +1.6 (witness), Pia +1.7 (witness) |
| 88 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.7 (witness), Nadia +1.1 (witness), Pia +1.5 (witness) |
| 89 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 89 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 89 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 90; anxiety Ana +0.4, Nadia +1.1 (witness), Pia +1.6 (witness), Ravi +1.6 (witness); calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana rejects the other on Ana–Marta (rejection +0.75; the tie stays open) |
| 89 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +3.5 |
| 89 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 90; anxiety Dr Halim +0.1, Marta +0.2, Nadia +1.1 (witness), Pia +1.6 (witness), Ravi +2.0 (witness) |
| 89 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 90; anxiety Marta +0.7, Nadia +0.8, Pia +1.8 (witness), Ravi +3.7 (witness); Marta left outside: anxiety Marta +7.8, Nadia -5.2 (of which 2.6 the outsider's own) |
| 89 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 90; anxiety Marta +0.4, Nadia +1.1 (witness), Pia +1.1, Ravi +3.1 (witness); Marta–Pia functioning balance flips; Marta takes 0.6 of functioning from the other |
| 89 | Ravi | DISTANCE | Marta | Nadia, Pia | refused by Marta (rejection +0.18); the sender bears the refusal: anxiety Ravi +6.0; anxiety Nadia +2.3 (witness), Pia +3.2 (witness) |
| 89 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.8; anxiety Marta +0.8 (witness), Nadia +1.1 (witness), Pia +1.6 (witness) |
| 90 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 90 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2; anxiety Nadia +1.1 (witness), Pia +1.6 (witness), Ravi +1.6 (witness) |
| 90 | Bruno | CUTOFF | Ana | — | arrives week 91; anxiety Ana +0.2, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.15; the tie stays open) |
| 90 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.16); the sender bears the refusal: anxiety Marta +2.6; anxiety Nadia +1.3 (witness), Ravi +3.6 (witness) |
| 90 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 91; anxiety Marta +2.8, Nadia +0.8, Pia +2.1 (witness), Ravi +4.1 (witness); Nadia takes 3.4 of functioning from the other |
| 90 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 91; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.2, Ravi +0.2; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia rejects the other on Pia–Ravi (rejection +0.87; the tie stays open) |
| 90 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.16); the sender bears the refusal: anxiety Ravi +6.1; anxiety Marta +1.5 (witness), Pia +1.6 (witness) |
| 90 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +1.5 (witness) |
| 90 | Dr Halim | SYMPTOM_ONSET | Marta, Ravi | Nadia, Pia | arrives week 91; symptom onset, physical channel; anxiety Marta +0.5, Nadia +0.9 (witness), Pia +1.2 (witness), Ravi +1.0 |
| 91 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +2.1; anxiety Nadia +1.0 (witness), Pia +1.6 (witness), Ravi +1.6 (witness) |
| 91 | Bruno | CUTOFF | Ana | — | arrives week 92; anxiety Ana +0.7, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 91 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 92; anxiety Marta +0.6, Nadia +1.9 (witness), Pia +2.9 (witness), Ravi +1.9; calmer sender takes some anxiety: Marta +0.7, Ravi -0.7; Ravi left outside: anxiety Marta -2.3, Ravi +3.4 (of which 1.1 the outsider's own) |
| 91 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.8; anxiety Marta +1.5 (witness), Pia +1.5 (witness) |
| 91 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 92; anxiety Marta +0.4, Nadia +1.2 (witness), Pia +1.2, Ravi +3.6 (witness) |
| 91 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +6.1; anxiety Marta +1.6 (witness), Pia +1.6 (witness) |
| 91 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +1.5 (witness) |
| 92 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.0 (witness), Nadia +1.3 (witness), Pia +2.0 (witness), Ravi +4.1 |
| 92 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.1, Nadia +1.3 (witness), Pia +2.0 (witness), Ravi +2.0 (witness) |
| 92 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 92 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 92 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 92 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 92 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 92 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 93; anxiety Ana +0.4, Marta +0.1, Nadia +1.0 (witness), Pia +1.5 (witness), Ravi +1.5 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Ana rejects the other on Ana–Marta (rejection +0.40; the tie stays open) |
| 92 | Bruno | CUTOFF | Ana | — | arrives week 93; anxiety Ana +0.6, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 92 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 93; anxiety Marta +0.5, Nadia +2.3 (witness), Pia +3.5 (witness), Ravi +8.5; calmer sender takes some anxiety: Marta +0.9, Ravi -0.9; Marta takes 3.2 of functioning from the other |
| 92 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 93; anxiety Marta +0.8, Nadia +0.8, Pia +1.9 (witness), Ravi +3.8 (witness); Marta left outside: anxiety Marta +6.1, Nadia -4.1 (of which 2.0 the outsider's own) |
| 92 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 93; anxiety Marta +0.3, Nadia +1.1 (witness), Pia +1.2, Ravi +3.5 (witness); calmer sender takes some anxiety: Marta -0.0, Pia +0.0 |
| 92 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +6.2; anxiety Marta +1.6 (witness), Pia +1.6 (witness) |
| 92 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +0.8 (witness), Nadia +1.1 (witness), Pia +1.6 (witness) |
| 93 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 93 | Ana | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 93 | Bruno | CUTOFF | Ana | — | arrives week 94; anxiety Ana +0.6, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 93 | Marta | PREVENT_ALIGNMENT | Nadia | Pia, Ravi | arrives week 94; anxiety Marta +0.3, Nadia +0.4, Pia +1.0 (witness), Ravi +2.3 (witness) |
| 93 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 94; anxiety Marta +0.6, Nadia +0.7, Pia +1.7 (witness), Ravi +3.9 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.75; the tie stays open); impingement removed at once, anxiety Nadia -0.4 |
| 93 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 94; anxiety Marta +4.2, Nadia +1.4 (witness), Pia +1.2, Ravi +4.7 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4 |
| 93 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 94; anxiety Marta +1.5 (witness), Nadia +0.6, Pia +1.6 (witness), Ravi +1.4; Ravi rejects the other on Nadia–Ravi (rejection +0.10; the tie stays open) |
| 93 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +0.8 (witness), Nadia +1.1 (witness), Pia +1.7 (witness) |
| 94 | Ana | WITHHOLD | Bruno | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 94 | Bruno | CUTOFF | Ana | — | arrives week 95; anxiety Ana +0.5, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 94 | Marta | REDUCE_CUTOFF | Ravi | Nadia, Pia | arrives week 95; anxiety Marta +0.3, Nadia +1.4 (witness), Pia +2.1 (witness), Ravi +2.8; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4; Marta's rejection -0.25; Ravi's rejection -0.10; bound anxiety released to the third party: Dr Halim +30.1, Nadia +30.1, Pia +30.1 |
| 94 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 94 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 95; anxiety Marta +0.6, Nadia +1.2 (witness), Pia +1.1, Ravi +4.1 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6 |
| 94 | Ravi | TRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.18); the sender bears the refusal: anxiety Ravi +6.7; anxiety Nadia +2.4 (witness), Pia +3.6 (witness) |
| 94 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 95 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 95 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 95 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 96; anxiety Ana +0.4, Nadia +0.8 (witness), Pia +1.3 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Ana +1.3, Marta -1.3; Ana rejects the other on Ana–Marta (rejection +0.15; the tie stays open) |
| 95 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.2 |
| 95 | Marta | DETRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +1.6; anxiety Nadia +0.7 (witness), Ravi +2.6 (witness) |
| 95 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 95 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 96; anxiety Marta +2.1, Nadia +1.4 (witness), Pia +1.3, Ravi +4.8 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Marta holds a position: no functioning changes hands |
| 95 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 96; anxiety Marta +1.4 (witness), Nadia +0.7, Pia +1.5 (witness), Ravi +1.4; Ravi rejects the other on Nadia–Ravi (rejection +0.10; the tie stays open) |
| 95 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 96 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.0 (witness), Nadia +1.3 (witness), Pia +2.0 (witness), Ravi +4.7 |
| 96 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.0, Nadia +1.3 (witness), Pia +2.0 (witness), Ravi +2.3 (witness) |
| 96 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 97; anxiety Ana +0.4, Marta +0.1, Nadia +0.8 (witness), Pia +1.2 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Ana +1.3, Marta -1.3; Ana rejects the other on Ana–Marta (rejection +0.05; the tie stays open) |
| 96 | Bruno | CUTOFF | Ana | — | arrives week 97; anxiety Ana +0.4, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 96 | Dr Halim | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 97; anxiety Dr Halim +0.1, Marta +0.6, Nadia +1.1 (witness), Pia +1.8 (witness), Ravi +2.6 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Marta -0.5; Dr Halim takes 3.1 of functioning from the other |
| 96 | Marta | WITHHOLD | Nadia | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 96 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 97; anxiety Marta +1.6 (witness), Nadia +0.8, Pia +1.7 (witness), Ravi +0.6; Nadia rejects the other on Nadia–Ravi (rejection +0.02; the tie stays open); impingement removed at once, anxiety Nadia -0.3 |
| 96 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 96 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.18); the sender bears the refusal: anxiety Ravi +7.0; anxiety Marta +1.6 (witness), Pia +1.7 (witness) |
| 96 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.13); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +0.8 (witness), Nadia +1.1 (witness), Pia +1.7 (witness) |
| 97 | Ana | WITHHOLD | Bruno | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 97 | Bruno | CUTOFF | Ana | — | arrives week 98; anxiety Ana +0.4, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 97 | Marta | PROVOKE | Ravi | Nadia, Pia | arrives week 98; anxiety Marta +0.3, Nadia +1.4 (witness), Pia +2.1 (witness), Ravi +5.9; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2 |
| 97 | Nadia | PROVOKE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Marta +1.0 (witness), Pia +1.0 (witness) |
| 97 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 98; anxiety Marta +1.6 (witness), Nadia +1.0 (witness), Pia +1.1, Ravi +1.1; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Pia rejects the other on Pia–Ravi (rejection +0.48; the tie stays open) |
| 97 | Ravi | REDUCE_CUTOFF | Marta | Nadia, Pia | arrives week 98; anxiety Marta +0.3, Nadia +1.4 (witness), Pia +2.1 (witness), Ravi +0.7; Marta's rejection -0.25 |
| 97 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 99; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +1.5 (witness), Ravi +0.6, Sofia +0.4; calmer sender takes some anxiety: Ravi -1.4, Sofia +1.4; Sofia rejects the other on Ravi–Sofia (rejection +0.55; the tie stays open) |
| 97 | Marta | SYMPTOM_ONSET | Ana, Dr Halim, Nadia, Pia, Ravi | — | arrives week 98; symptom onset, physical channel; anxiety Ana +1.5, Dr Halim +0.4, Nadia +2.2, Pia +3.4, Ravi +3.7 |
| 98 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 98 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 98 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 98 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +0.8 (witness), Pia +1.3 (witness), Ravi +1.4 (witness) |
| 98 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.1 |
| 98 | Marta | PREVENT_ALIGNMENT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +1.7; anxiety Pia +1.0 (witness), Ravi +2.1 (witness) |
| 98 | Nadia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 98 | Pia | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 98 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Ravi +6.4; anxiety Marta +1.6 (witness), Nadia +1.1 (witness) |
| 98 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 100; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +1.4 (witness), Ravi +1.2, Sofia +0.4; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 98 | Sofia | SYMPTOM_ONSET | Ravi | Marta, Nadia, Pia | arrives week 100; symptom onset, physical channel; anxiety Marta +0.5 (witness), Nadia +0.7 (witness), Pia +1.0 (witness), Ravi +2.2 |
| 98 | — | (system) | — | — | Marta–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 99 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +0.9 (witness), Pia +1.3 (witness), Ravi +1.4 (witness) |
| 99 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.2 |
| 99 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.16); the sender bears the refusal: anxiety Marta +2.8; anxiety Nadia +1.3 (witness), Ravi +4.3 (witness) |
| 99 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 100; anxiety Marta +4.6, Nadia +0.7, Pia +1.8 (witness), Ravi +4.0 (witness); Marta holds a position: no functioning changes hands |
| 99 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 100; anxiety Marta +3.1, Nadia +1.2 (witness), Pia +1.0, Ravi +4.1 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9 |
| 99 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 100; anxiety Marta +4.6, Nadia +2.8 (witness), Pia +4.2 (witness), Ravi +1.2 |
| 99 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.13); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +1.4 (witness) |
| 100 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.0 (witness), Nadia +1.4 (witness), Pia +2.0 (witness), Ravi +4.4 |
| 100 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.1, Nadia +1.3 (witness), Pia +2.0 (witness), Ravi +2.2 (witness) |
| 100 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.13); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +1.0 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 100 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.3 |
| 100 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.17); the sender bears the refusal: anxiety Marta +1.8; anxiety Nadia +1.4 (witness), Pia +2.1 (witness), Ravi +2.1 (witness) |
| 100 | Nadia | TRIANGLE | Ravi | Marta, Pia | refused by Ravi (rejection +0.15); the sender bears the refusal: anxiety Nadia +3.7; anxiety Marta +1.6 (witness), Pia +1.6 (witness) |
| 100 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 101; anxiety Marta +1.5 (witness), Nadia +1.0 (witness), Pia +1.1, Ravi +1.4; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Pia rejects the other on Pia–Ravi (rejection +0.10; the tie stays open) |
| 100 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 101; anxiety Marta +1.6, Nadia +3.3 (witness), Pia +4.9 (witness), Ravi +1.3 |
| 100 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.09); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +1.3 (witness) |
| 101 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.12); the sender bears the refusal: anxiety Ana +1.9; anxiety Nadia +0.9 (witness), Pia +1.4 (witness), Ravi +1.4 (witness) |
| 101 | Bruno | CUTOFF | Ana | — | arrives week 102; anxiety Ana +0.2, Bruno +0.7; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open) |
| 101 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.17); the sender bears the refusal: anxiety Marta +2.9; anxiety Pia +2.2 (witness), Ravi +4.4 (witness) |
| 101 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 101 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 102; anxiety Marta +1.5 (witness), Nadia +1.0 (witness), Pia +1.1, Ravi +1.3; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; 22.3 of the sender's anxiety bound into Pia–Ravi |
| 101 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.17); the sender bears the refusal: anxiety Ravi +5.9; anxiety Marta +1.6 (witness), Pia +1.6 (witness) |
| 101 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.9; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +1.3 (witness) |
| 102 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.5 |
| 102 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.12); the sender bears the refusal: anxiety Bruno +3.3 |
| 102 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.13); the sender bears the refusal: anxiety Marta +1.7; anxiety Nadia +1.1 (witness), Pia +1.7 (witness), Ravi +1.6 (witness) |
| 102 | Nadia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 102 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 103; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +0.9, Ravi +3.0 (witness); calmer sender takes some anxiety: Marta -1.4, Pia +1.4; Marta takes 0.3 of functioning from the other |
| 102 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | — |
| 102 | Sofia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 103 | Ana | CUTOFF | Bruno | — | — |
| 103 | Bruno | CUTOFF | Ana | — | — |
| 103 | Marta | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 103 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | — |
| 103 | Pia | CUTOFF | Ravi | Marta, Nadia | — |
| 103 | Ravi | PURSUE | Nadia | Marta, Pia | — |
| 103 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | — |
| 103 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |

104 weeks. The invariants were asserted at the end of every week.
