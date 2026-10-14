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
| config hash | `99bbb5c691e57c9144931f15ada9c2bb0f0379ce30f4e17b0e23e2aeaf6ebe5b` |
| spec revision | 2.0, revision 29 |
| family instance | phase_c |
| activation | synchronous_activation v1 (synchronous) |
| visibility | household_conductance_visibility v1 |
| constants frozen | 2026-10-07 |
| constants changed since freeze | acting_age_years, attribution_rate, belief_appraisal_gain, birth_focus_gain, budget_per_undifferentiation, channel_protection, coach_retirement_age_years, coach_start_age_years, constitution_strength, death_transfer_share, declared_rejection, development_band, development_end_age_years, development_rate, development_start_age_years, estimator_breadth, estimator_rate, estimator_sd_weight, estimator_window, exchange_window, find_work_pressure_scale, find_work_rate, focus_rate, focus_steer_gain, grief_extra_weeks, grief_min_weeks, grief_scale, income_loss_intensity, income_shortfall_share, launch_age_years, lock_in_gain, lock_in_turn, marital_distance_fraction, mortality_a, mortality_b, new_tie_investment_share, pattern_min_acts, pattern_projection_hold, pattern_window, perceived_weakness_gain, perspective_channel_gain, perspective_steer_gain, practice_window, pseudo_self_relaxation_rate, reactive_cutoff_gain, refusal_cost_gain, rejection_cutoff_gain, rejection_fade, rejection_refusal_gain, rejection_refusal_threshold, rejection_relief, response_down_gain, response_relaxation_rate, response_up_gain, self_practice_gain, soothing_review_every, soothing_review_penalty, stress_floor_below_basic, stressor_persistence, system_adjust_gain, tie_wear_rate, usual_level_rate, wear_offset_per_act, witness_window, witnessed_cutoff_gain, witnessed_gain, working_age_years |

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
| 0 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 2; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +2.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Sofia takes 2.0 of functioning from the other |
| 1 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 1 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 1 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 1 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 1 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 1 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 1 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.2 |
| 1 | Bruno | CUTOFF | Ana | — | arrives week 2; anxiety Ana +1.5, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 1 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 2; anxiety Marta +0.5, Nadia +0.8, Pia +0.5 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3; Marta rejects the other on Marta–Nadia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -0.6 |
| 1 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 2; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi -0.6; Ravi left outside: anxiety Nadia -0.5, Ravi +0.7 (of which 0.2 the outsider's own) |
| 1 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 2; anxiety Marta -0.2, Nadia +0.6 (witness), Pia +0.4, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4; Marta takes 2.0 of functioning from the other |
| 1 | Ravi | WITHHOLD | Nadia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 1 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 3; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +3.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5 |
| 2 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 2 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 3; anxiety Ana +0.3, Marta -0.6, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.2, Marta -0.2 |
| 2 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 2 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 3; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +1.0 (witness) |
| 2 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Pia +0.5 (witness), Ravi +1.4 (witness) |
| 2 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 3; anxiety Marta +1.4 (witness), Nadia +0.7 (witness), Pia +0.4, Ravi +4.8; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 2.8 of the sender's anxiety bound into Pia–Ravi |
| 2 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 3; anxiety Marta +1.3 (witness), Nadia -0.7, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Nadia left outside: anxiety Nadia +2.7, Ravi -1.8 (of which 0.9 the outsider's own) |
| 2 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 4; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +2.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 1.1 of the sender's anxiety bound into Ravi–Sofia |
| 3 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 3 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 4; anxiety Ana +0.3, Marta +2.1, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.3, Marta -0.3; 2.0 of the sender's anxiety bound into Ana–Marta |
| 3 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 3 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 4; anxiety Marta +0.5, Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +1.2; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4 |
| 3 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Pia +0.6 (witness), Ravi +1.7 (witness) |
| 3 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 4; anxiety Marta +3.1, Nadia +0.7 (witness), Pia +0.4, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5 |
| 3 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | arrives week 4; anxiety Marta +1.3 (witness), Nadia +1.2, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.2, Ravi +0.2; Nadia takes 2.0 of functioning from the other |
| 3 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 5; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +3.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia takes 2.0 of functioning from the other |
| 4 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +3.4 |
| 4 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +1.7 (witness) |
| 4 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 4 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 4 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 4 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 4 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 5; anxiety Marta +0.5, Nadia +1.0, Pia +0.6 (witness), Ravi +1.7 (witness); calmer sender takes some anxiety: Marta +0.4, Nadia -0.4; Marta rejects the other on Marta–Nadia (rejection +0.45; the tie stays open) |
| 4 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 5; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +5.7; Nadia takes 2.0 of functioning from the other |
| 4 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 5; anxiety Marta +1.4 (witness), Nadia +0.7 (witness), Pia +0.4, Ravi +0.6; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Ravi left outside: anxiety Pia -0.9, Ravi +1.3 (of which 0.4 the outsider's own) |
| 4 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 5; anxiety Marta +1.4 (witness), Nadia +0.7 (witness), Pia +1.2, Ravi +0.6 |
| 4 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 6; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; 1.4 of the sender's anxiety bound into Ravi–Sofia |
| 5 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 5 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 5 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +0.3, Marta +2.5, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana takes 2.0 of functioning from the other |
| 5 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 5 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +2.1, Marta +0.3, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +1.0 (witness) |
| 5 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 6; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.7; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Ravi left outside: anxiety Nadia -1.2, Ravi +1.8 (of which 0.6 the outsider's own) |
| 5 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 6; anxiety Marta +1.5 (witness), Nadia +0.7 (witness), Pia +0.4, Ravi +1.5; calmer sender takes some anxiety: Pia +1.3, Ravi -1.3; Ravi takes 2.0 of functioning from the other |
| 5 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 6; anxiety Marta +1.5 (witness), Nadia +0.7 (witness), Pia +4.4, Ravi +0.7 |
| 5 | Sofia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 5 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 6, 7; symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.5, Nadia +2.2, Pia +2.2; anxiety Sofia +1.3 |
| 6 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 6 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.1 |
| 6 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 6 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | arrives week 7; anxiety Marta +0.5, Nadia +0.7 (witness), Pia +3.4, Ravi +2.0 (witness); Marta takes 2.0 of functioning from the other |
| 6 | Nadia | PREVENT_ALIGNMENT | Ravi | Marta, Pia | arrives week 7; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.5; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 6 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 7; anxiety Marta +1.5 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.2; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Ravi takes 2.0 of functioning from the other |
| 6 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 7; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 6 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 8; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +2.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 7 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 8; anxiety Ana +0.3, Marta +2.6, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana takes 2.0 of functioning from the other |
| 7 | Bruno | CUTOFF | Ana | — | arrives week 8; anxiety Ana +0.6, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 7 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | arrives week 8; anxiety Marta +0.5, Nadia +0.0, Pia +0.7 (witness), Ravi +1.7 (witness); calmer sender takes some anxiety: Marta +0.1, Nadia -0.1; Nadia takes 2.0 of functioning from the other |
| 7 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 8; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2 |
| 7 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 8; anxiety Marta +1.5 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Pia–Ravi functioning balance flips; Ravi takes 2.0 of functioning from the other |
| 7 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 8; anxiety Marta +1.5 (witness), Nadia +0.6 (witness), Pia +3.8, Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 7 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +2.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 8 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.6 |
| 8 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.8 (witness) |
| 8 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 8 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.3, Marta +3.2, Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana takes 2.0 of functioning from the other |
| 8 | Bruno | CUTOFF | Ana | — | arrives week 9; anxiety Ana +0.8, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 8 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Dr Halim +0.1, Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Ravi -0.6 |
| 8 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.5, Marta +0.3, Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +0.9 (witness); Ana takes 2.0 of functioning from the other |
| 8 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 9; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.2; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; 6.7 of the sender's anxiety bound into Nadia–Ravi |
| 8 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 9; anxiety Marta +5.3, Nadia +0.7 (witness), Pia +0.6, Ravi +1.8 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Pia takes 2.0 of functioning from the other |
| 8 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 9; anxiety Marta +1.8 (witness), Nadia +0.7 (witness), Pia +2.5, Ravi +0.5; Ravi takes 2.0 of functioning from the other |
| 8 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 10; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +3.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia takes 2.0 of functioning from the other |
| 8 | Marta | SYMPTOM_ONSET | Ana, Dr Halim, Nadia, Pia, Ravi | — | arrives week 9; symptom onset, physical channel; anxiety Ana +1.5, Dr Halim +0.5, Nadia +2.1, Pia +2.8, Ravi +2.7 |
| 9 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 9 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | arrives week 10; anxiety Ana +0.3, Marta +3.4, Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 9 | Bruno | CUTOFF | Ana | — | arrives week 10; anxiety Ana +0.7, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 9 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 10; anxiety Marta +0.7, Nadia +3.3, Pia +0.8 (witness), Ravi +1.6 (witness); Marta takes 2.0 of functioning from the other |
| 9 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 10; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4 |
| 9 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 10; anxiety Marta +2.0 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +1.3; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; 8.2 of the sender's anxiety bound into Pia–Ravi |
| 9 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 10; anxiety Marta +2.0 (witness), Nadia +0.6 (witness), Pia +1.4, Ravi +0.5; Pia left outside: anxiety Pia +4.3, Ravi -2.9 (of which 1.4 the outsider's own) |
| 9 | Sofia | PROVOKE | Ravi | Marta, Nadia, Pia | arrives week 11; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +2.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 9 | Pia | SYMPTOM_ONSET | Marta, Ravi | Nadia | arrives week 10; symptom onset, social channel; anxiety Marta +3.3, Nadia +0.7 (witness), Ravi +2.6 |
| 10 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 10 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 10 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.8 |
| 10 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 10 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 11; anxiety Marta +0.6, Nadia +0.7 (witness), Pia +3.7, Ravi +1.6 (witness); 10.9 of the sender's anxiety bound into Marta–Pia |
| 10 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 11; anxiety Marta +1.6 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Ravi left outside: anxiety Nadia -1.1, Ravi +1.6 (of which 0.5 the outsider's own) |
| 10 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 11; anxiety Marta +1.9 (witness), Nadia +0.7 (witness), Pia +0.6, Ravi +2.6; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 10 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 11; anxiety Marta +3.3, Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.2, Ravi +0.2 |
| 10 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 12; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia +1.0 (witness), Ravi +2.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 11 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 12; anxiety Ana +0.3, Marta +0.9, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 11 | Bruno | CUTOFF | Ana | — | arrives week 12; anxiety Ana +0.6, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 11 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | arrives week 12; anxiety Marta +0.6, Nadia -0.1, Pia +0.8 (witness), Ravi +1.6 (witness); calmer sender takes some anxiety: Marta +0.0, Nadia -0.0; Nadia takes 2.0 of functioning from the other |
| 11 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 12; anxiety Marta +1.6 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.3; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; 6.1 of the sender's anxiety bound into Nadia–Ravi |
| 11 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 11 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 12; anxiety Marta +5.0, Nadia +1.4 (witness), Pia +1.8 (witness), Ravi +0.5 |
| 11 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 13; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 12 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.9 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +3.3 |
| 12 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.8, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.6 (witness) |
| 12 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +0.3, Marta +1.0, Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; 3.7 of the sender's anxiety bound into Ana–Marta |
| 12 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 12 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +2.4, Marta +0.4, Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +0.8 (witness); Marta takes 2.0 of functioning from the other |
| 12 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 13; anxiety Marta +1.9 (witness), Nadia +0.4, Pia +0.8 (witness), Ravi +4.7; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Nadia takes 2.0 of functioning from the other |
| 12 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 12 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 13; anxiety Marta +1.9 (witness), Nadia +0.6 (witness), Pia +1.3, Ravi +0.5; 10.2 of the sender's anxiety bound into Pia–Ravi |
| 12 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 14; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +2.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 1.9 of the sender's anxiety bound into Ravi–Sofia |
| 13 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 13 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 13 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 13 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 14; anxiety Marta +0.6, Nadia +1.3 (witness), Pia +1.8 (witness), Ravi -0.6; Ravi left outside: anxiety Marta -2.3, Ravi +3.5 (of which 1.2 the outsider's own) |
| 13 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 14; anxiety Marta +2.0 (witness), Nadia +0.4, Pia +0.9 (witness), Ravi +2.8 |
| 13 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 14; anxiety Marta +0.8, Nadia +0.6 (witness), Pia +0.6, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -0.3 |
| 13 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 14; anxiety Marta +4.2, Nadia +1.3 (witness), Pia +1.8 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3; Ravi takes 2.0 of functioning from the other |
| 13 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 14 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 14 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 14 | Ana | WITHHOLD | Bruno | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 14 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 14 | Marta | TRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.4; anxiety Nadia +0.6 (witness), Ravi +1.5 (witness) |
| 14 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.9 (witness), Ravi +1.7 (witness) |
| 14 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 15; anxiety Marta +1.9 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +2.0; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 6.5 of the sender's anxiety bound into Pia–Ravi |
| 14 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.9 (witness) |
| 14 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 16; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.9 (witness), Ravi +3.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 15 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 15 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 15 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 15 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 16; anxiety Marta +0.7, Nadia +1.2 (witness), Pia +1.7 (witness), Ravi +1.8; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1; 10.4 of the sender's anxiety bound into Marta–Ravi |
| 15 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.8 (witness), Ravi +1.6 (witness) |
| 15 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 16; anxiety Marta +1.9 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +4.4; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Pia–Ravi functioning balance flips; Pia takes 2.0 of functioning from the other |
| 15 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 16; anxiety Marta +1.9 (witness), Nadia +0.6 (witness), Pia +1.1, Ravi +0.5; Ravi rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -5.6 |
| 15 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 17; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 16 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.2 (witness), Pia +1.7 (witness), Ravi +3.3 |
| 16 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.0, Nadia +1.2 (witness), Pia +1.7 (witness), Ravi +1.6 (witness) |
| 16 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 16 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 16 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 17; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Marta -0.5 |
| 16 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 17; anxiety Marta +0.7, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +5.8; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2 |
| 16 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 17; anxiety Marta +2.2 (witness), Nadia +0.4, Pia +0.9 (witness), Ravi +3.7; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 16 | Pia | TRIANGLE | Marta | Nadia, Ravi | arrives week 17; anxiety Marta -0.5, Nadia +0.5 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5; Marta left outside: anxiety Marta +2.4, Pia -1.6 (of which 0.8 the outsider's own) |
| 16 | Ravi | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 16 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 18; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Ravi takes 2.0 of functioning from the other |
| 17 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 17 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 17 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 17 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 18; anxiety Ana +0.3, Marta +0.2, Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 17 | Bruno | CUTOFF | Ana | — | arrives week 18; anxiety Ana +0.3, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 17 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 18; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.0 (witness) |
| 17 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.8 (witness), Ravi +1.8 (witness) |
| 17 | Pia | REDUCE_CUTOFF | Ravi | Marta, Nadia | arrives week 18; anxiety Marta +1.9 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.8; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Ravi's rejection -0.25; bound anxiety released to the third party: Marta +27.8 |
| 17 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 18; anxiety Marta +2.0 (witness), Nadia +0.1, Pia +0.8 (witness), Ravi +0.6; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -4.0 |
| 17 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 19; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +1.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 18 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 18 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 18 | Marta | TRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.4; anxiety Nadia +0.5 (witness), Ravi +1.4 (witness) |
| 18 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.8 (witness), Ravi +1.6 (witness) |
| 18 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 19; anxiety Marta +0.8, Nadia +0.5 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -1.6, Pia +1.6; 6.5 of the sender's anxiety bound into Marta–Pia |
| 18 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.8 (witness) |
| 18 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 19 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 19 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 20; anxiety Ana +0.3, Marta +2.4, Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9 |
| 19 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 19 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 20; anxiety Marta +0.7, Nadia +1.4, Pia +0.8 (witness), Ravi +1.5 (witness); 13.1 of the sender's anxiety bound into Marta–Nadia |
| 19 | Nadia | TRIANGLE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Marta +1.9 (witness), Pia +0.8 (witness) |
| 19 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 20; anxiety Marta +2.0 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.9; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0 |
| 19 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.9 (witness) |
| 19 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 19 | Bruno | SYMPTOM_ONSET | Ana | — | arrives week 20; symptom onset, social channel; anxiety Ana +1.9 |
| 20 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.2 (witness), Pia +1.6 (witness), Ravi +3.3 |
| 20 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.1, Nadia +1.2 (witness), Pia +1.6 (witness), Ravi +1.6 (witness) |
| 20 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 20 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 20 | Ana | CUTOFF | Bruno | — | arrives week 21; anxiety Ana +0.4, Bruno +0.1; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 20 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 20 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 21; anxiety Marta +0.7, Nadia +0.5 (witness), Pia +0.1, Ravi +1.4 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -0.2 |
| 20 | Nadia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 20 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 21; anxiety Marta +2.2 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +5.2; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9 |
| 20 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 21; anxiety Marta +1.8 (witness), Nadia -0.7, Pia +0.7 (witness), Ravi +0.5; Nadia left outside: anxiety Nadia +3.9, Ravi -2.6 (of which 1.3 the outsider's own) |
| 20 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 22; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Ravi takes 2.0 of functioning from the other |
| 21 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 21 | Ana | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 21 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 21 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 22; anxiety Marta +0.7, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.2 |
| 21 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.8 (witness), Ravi +1.7 (witness) |
| 21 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 22; anxiety Marta +2.2 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +1.8; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 21 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | arrives week 22; anxiety Marta +2.2 (witness), Nadia +0.7 (witness), Pia -0.5, Ravi +0.5; Pia takes 2.0 of functioning from the other |
| 21 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 23; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +1.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 21 | — | (system) | — | — | Pia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 22 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 22 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 23; anxiety Ana +0.3, Marta +2.3, Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 22 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 22 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | arrives week 23; anxiety Marta +0.7, Nadia -0.6, Pia +0.7 (witness), Ravi +1.6 (witness); Nadia holds a position: no functioning changes hands |
| 22 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 22 | Pia | TRIANGLE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.5 (witness), Ravi +1.4 (witness) |
| 22 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 23; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 22 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 24; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +3.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 23 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 23 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 23 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 24; anxiety Ana +0.3, Marta +0.5, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 23 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 23 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 24; anxiety Marta +0.7, Nadia +0.6 (witness), Pia +3.5, Ravi +1.6 (witness) |
| 23 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.7 (witness), Ravi +1.5 (witness) |
| 23 | Pia | PREVENT_ALIGNMENT | Ravi | Marta, Nadia | arrives week 24; anxiety Marta +2.0 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.2; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7 |
| 23 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 24; anxiety Marta +5.1, Nadia +1.4 (witness), Pia +1.7 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1; Ravi takes 2.0 of functioning from the other |
| 23 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 25; anxiety Marta +1.2 (witness), Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +2.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia takes 2.0 of functioning from the other |
| 24 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +3.2 |
| 24 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.9, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 24 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 24 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 24 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 24 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 25; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 24 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 25; anxiety Marta +0.7, Nadia +0.6 (witness), Pia +0.7, Ravi +1.4 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 24 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 25; anxiety Marta +2.2 (witness), Nadia +0.4, Pia +0.8 (witness), Ravi +4.4; calmer sender takes some anxiety: Nadia +0.0, Ravi -0.0 |
| 24 | Pia | TRIANGLE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.4 (witness) |
| 24 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 25; anxiety Marta +1.3, Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.2, Ravi +0.2; 11.5 of the sender's anxiety bound into Marta–Ravi |
| 24 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 26; anxiety Marta +1.2 (witness), Nadia +0.8 (witness), Pia +0.9 (witness), Ravi +2.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 25 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 25 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 25 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 26; anxiety Ana +0.3, Marta +2.5, Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 25 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 25 | Marta | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 25 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 26; anxiety Marta +2.0 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi -0.4; Ravi left outside: anxiety Nadia -1.3, Ravi +1.9 (of which 0.6 the outsider's own) |
| 25 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.6 (witness) |
| 25 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +1.2 (witness), Nadia +0.7 (witness), Pia +0.9 (witness) |
| 25 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 27; anxiety Marta +1.2 (witness), Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.30; the tie stays open) |
| 26 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 26 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 27; anxiety Ana +0.3, Marta +2.5, Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 26 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 26 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 27; anxiety Marta +0.7, Nadia +0.6, Pia +0.7 (witness), Ravi +1.5 (witness); 10.1 of the sender's anxiety bound into Marta–Nadia |
| 26 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.7 (witness), Ravi +1.5 (witness) |
| 26 | Pia | STAY-IN-CONTACT | Ravi | Marta, Nadia | arrives week 27; anxiety Marta +2.0 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi -0.1; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 26 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 27; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 26 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 28; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 27 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 28; anxiety Ana +0.3, Marta +2.5, Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 27 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 27 | Marta | WITHHOLD | Nadia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 27 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 28; anxiety Marta +1.9 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.9; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 27 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 28; anxiety Marta +1.9 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +3.1; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4 |
| 27 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | arrives week 28; anxiety Marta +1.9 (witness), Nadia +0.6 (witness), Pia -0.3, Ravi +0.5; Pia takes 2.0 of functioning from the other |
| 27 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 28 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +3.1 |
| 28 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.1, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 28 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 28 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 28 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 28 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 28 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 29; anxiety Marta +0.7, Nadia +1.4 (witness), Pia +1.6 (witness), Ravi +4.4; Marta takes 2.0 of functioning from the other |
| 28 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.8 (witness), Ravi +1.8 (witness) |
| 28 | Pia | PROVOKE | Ravi | Marta, Nadia | arrives week 29; anxiety Marta +2.1 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +4.2; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7 |
| 28 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.7; anxiety Marta +1.8 (witness), Pia +0.6 (witness) |
| 28 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 29 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 29 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 29 | Bruno | CUTOFF | Ana | — | arrives week 30; anxiety Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.25; the tie stays open) |
| 29 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 30; anxiety Marta +0.6, Nadia +1.4 (witness), Pia +1.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1; Ravi takes 2.0 of functioning from the other |
| 29 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 30; anxiety Marta +0.5, Nadia +0.4, Pia +0.7 (witness), Ravi +1.7 (witness); calmer sender takes some anxiety: Marta -0.1, Nadia +0.1; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 29 | Pia | STAY-IN-CONTACT | Ravi | Marta, Nadia | arrives week 30; anxiety Marta +1.8 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 29 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 30; anxiety Marta +5.4, Nadia +1.4 (witness), Pia +1.6 (witness), Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 29 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 31; anxiety Ana +0.3, Marta +4.5, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 30 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 31; anxiety Ana -0.7, Marta +0.4, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); Ana takes 2.0 of functioning from the other |
| 30 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 31; anxiety Marta +1.9 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia +0.0, Ravi -0.0; Ravi takes 2.0 of functioning from the other |
| 30 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 31; anxiety Marta +2.1 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +3.1; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7 |
| 30 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.5; anxiety Marta +1.3 (witness), Nadia +0.7 (witness), Pia +0.8 (witness) |
| 30 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 32; anxiety Marta +1.3 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +2.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia takes 2.0 of functioning from the other |
| 31 | Ana | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 31 | Bruno | CUTOFF | Ana | — | arrives week 32; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 31 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 32; anxiety Marta +0.8, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +0.7; 12.8 of the sender's anxiety bound into Marta–Ravi |
| 31 | Nadia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 31 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 32; anxiety Marta +2.2 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.7; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; 8.0 of the sender's anxiety bound into Pia–Ravi |
| 31 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.4; anxiety Marta +2.0 (witness), Pia +0.6 (witness) |
| 31 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 33; anxiety Marta +1.2 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia rejects the other on Ravi–Sofia (rejection +0.15; the tie stays open) |
| 32 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.3 (witness), Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +2.9 |
| 32 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.5, Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +1.5 (witness) |
| 32 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 32 | Ana | CUTOFF | Bruno | — | arrives week 33; anxiety Ana +0.4, Bruno +0.1; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 32 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 32 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 33; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.50; the tie stays open) |
| 32 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.8; anxiety Pia +0.7 (witness), Ravi +1.5 (witness) |
| 32 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 33; anxiety Marta +1.9 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Ravi left outside: anxiety Nadia -1.2, Ravi +1.8 (of which 0.6 the outsider's own) |
| 32 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 33; anxiety Marta +2.4 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +3.7; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0 |
| 32 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 33; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 32 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 34; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; 2.1 of the sender's anxiety bound into Ravi–Sofia |
| 33 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 33 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 33 | Ana | CUTOFF | Bruno | — | arrives week 34; anxiety Ana +0.4, Bruno +0.6; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 33 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 33 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 34; anxiety Ana -0.2, Marta +0.4, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); Ana holds a position: no functioning changes hands |
| 33 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 34; anxiety Marta +2.2 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +5.1; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2 |
| 33 | Pia | PROVOKE | Ravi | Marta, Nadia | arrives week 34; anxiety Marta +2.4 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +2.3; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9 |
| 33 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 34; anxiety Dr Halim +0.2, Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 33 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 35; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; 1.7 of the sender's anxiety bound into Ravi–Sofia |
| 34 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 35; anxiety Ana +0.3, Marta +2.6, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9 |
| 34 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 34 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 35; anxiety Marta +0.4, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -3.7 |
| 34 | Nadia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 34 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.4; anxiety Nadia +0.8 (witness), Ravi +1.7 (witness) |
| 34 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.6; anxiety Marta +2.3 (witness), Pia +0.7 (witness) |
| 34 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 36; anxiety Marta +1.0 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 35 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness) |
| 35 | Bruno | CUTOFF | Ana | — | arrives week 36; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open) |
| 35 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 36; anxiety Marta +0.7, Nadia +0.7 (witness), Pia +0.5, Ravi +1.6 (witness); 10.7 of the sender's anxiety bound into Marta–Pia |
| 35 | Nadia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 35 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 36; anxiety Marta +2.3 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +3.4; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 35 | Ravi | WITHHOLD | Sofia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 35 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 37; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1 |
| 36 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.1 (witness), Nadia +1.4 (witness), Pia +1.4 (witness), Ravi +3.1 |
| 36 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.3, Nadia +1.4 (witness), Pia +1.4 (witness), Ravi +1.5 (witness) |
| 36 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 36 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.8 (witness) |
| 36 | Bruno | CUTOFF | Ana | — | arrives week 37; anxiety Ana +0.5, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 36 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 37; anxiety Marta +0.7, Pia +0.7 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta +0.1, Nadia -0.1; 8.4 of the sender's anxiety bound into Marta–Nadia |
| 36 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 37; anxiety Marta +2.2 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 36 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 37; anxiety Marta +0.8, Nadia +0.7 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 36 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.6; anxiety Marta +2.2 (witness), Pia +0.7 (witness) |
| 36 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 38; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Ravi holds a position: no functioning changes hands |
| 37 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 37 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 37 | Marta | STAY-IN-CONTACT | Ana | Nadia, Pia, Ravi | arrives week 38; anxiety Ana -0.5, Marta +0.4, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 37 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 38; anxiety Marta -1.5, Nadia +0.5, Pia +0.6 (witness), Ravi +1.4 (witness); Marta takes 2.0 of functioning from the other |
| 37 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 38; anxiety Marta +1.2, Nadia +0.6 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.3, Pia +0.3; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 37 | Ravi | WITHHOLD | Pia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 37 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 38 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 38 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 38 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 39; anxiety Marta +0.7, Nadia -1.7, Pia +0.6 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3; Nadia left outside: anxiety Marta -1.5, Nadia +2.2 (of which 0.7 the outsider's own) |
| 38 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 39; anxiety Marta -2.1, Nadia +0.5, Pia +0.6 (witness), Ravi +1.4 (witness); Marta left outside: anxiety Marta +1.9, Nadia -1.3 (of which 0.6 the outsider's own) |
| 38 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 39; anxiety Marta +2.1 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +3.2; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7 |
| 38 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 39; anxiety Marta +2.1 (witness), Nadia +0.7 (witness), Pia +2.2, Ravi +0.5; 9.5 of the sender's anxiety bound into Pia–Ravi |
| 38 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 40; anxiety Marta +1.0 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 39 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 39 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 39 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.8 (witness) |
| 39 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 39 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 40; anxiety Marta +0.7, Nadia +1.5, Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta +0.2, Nadia -0.2; 7.4 of the sender's anxiety bound into Marta–Nadia |
| 39 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 40; anxiety Marta +0.6, Nadia +0.5, Pia +0.6 (witness), Ravi +1.3 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 39 | Pia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 39 | Ravi | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 39 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 40 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.4 (witness), Pia +1.4 (witness), Ravi +3.1 |
| 40 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.0, Nadia +1.4 (witness), Pia +1.4 (witness), Ravi +1.5 (witness) |
| 40 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 40 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 40 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 41; anxiety Dr Halim +0.1, Marta -0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Marta -0.5; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 40 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.3; anxiety Nadia +0.7 (witness), Ravi +1.6 (witness) |
| 40 | Nadia | PURSUE | Marta | Pia, Ravi | arrives week 41; anxiety Marta +2.2, Nadia +0.5, Pia +0.7 (witness), Ravi +1.4 (witness) |
| 40 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.4; anxiety Nadia +0.7 (witness), Ravi +1.6 (witness) |
| 40 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 41; anxiety Marta +2.2 (witness), Nadia +0.8 (witness), Pia +3.5, Ravi +0.5; Ravi takes 2.0 of functioning from the other |
| 40 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 41 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +0.7 (witness) |
| 41 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 41 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 42; anxiety Marta +0.7, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +1.5; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1; 7.3 of the sender's anxiety bound into Marta–Ravi |
| 41 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 42; anxiety Marta +1.9 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi -0.7; Ravi left outside: anxiety Nadia -1.4, Ravi +2.0 (of which 0.7 the outsider's own) |
| 41 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 42; anxiety Marta +2.0 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +1.5; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; 8.1 of the sender's anxiety bound into Pia–Ravi |
| 41 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 42; anxiety Marta +1.2, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +0.5 |
| 41 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 43; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; 5.7 of the sender's anxiety bound into Ravi–Sofia |
| 42 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 42 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 42 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 42 | Marta | STAY-IN-CONTACT | Ana | Nadia, Pia, Ravi | arrives week 43; anxiety Ana -0.2, Marta +0.4, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.7 (witness) |
| 42 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 43; anxiety Marta +0.5, Nadia +0.5, Pia +0.7 (witness), Ravi +1.3 (witness); Marta left outside: anxiety Marta +2.9, Nadia -1.9 (of which 1.0 the outsider's own) |
| 42 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 42 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 43; anxiety Marta +1.9 (witness), Nadia +0.7 (witness), Pia +0.7, Ravi +0.5; Ravi rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -1.3 |
| 42 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 43 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 43 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.7 (witness) |
| 43 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 43 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 44; anxiety Marta +0.4, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); Marta rejects the other on Ana–Marta (rejection +0.45; the tie stays open) |
| 43 | Nadia | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 43 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Marta +2.0 (witness), Nadia +0.7 (witness) |
| 43 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 44; anxiety Marta +5.7, Nadia +1.5 (witness), Pia +1.6 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.4, Ravi +0.4; Ravi takes 2.0 of functioning from the other |
| 43 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 45; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +2.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia takes 2.0 of functioning from the other |
| 43 | — | (system) | — | — | Dr Halim–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 44 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.9 (witness), Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +3.0 |
| 44 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.8, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +1.5 (witness) |
| 44 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 44 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 44 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 44 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.4; anxiety Pia +0.7 (witness), Ravi +1.2 (witness) |
| 44 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 45; anxiety Marta +1.9 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi -0.6; Ravi takes 2.0 of functioning from the other |
| 44 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 44 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.4; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia +0.8 (witness) |
| 44 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 46; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +2.5, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 45 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 45 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 45 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 46; anxiety Marta +0.7, Nadia +1.6 (witness), Pia +1.6 (witness), Ravi +0.6 |
| 45 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 46; anxiety Marta +1.9 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi -0.4; Ravi takes 2.0 of functioning from the other |
| 45 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 46; anxiety Marta +0.5, Nadia +0.7 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; Pia rejects the other on Marta–Pia (rejection +0.35; the tie stays open) |
| 45 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 46; anxiety Marta +2.0 (witness), Nadia +0.7 (witness), Pia +1.7, Ravi +0.5; 8.3 of the sender's anxiety bound into Pia–Ravi |
| 45 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 47; anxiety Marta +1.1 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +1.8, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Sofia takes 2.0 of functioning from the other |
| 46 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 46 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 46 | Bruno | CUTOFF | Ana | — | arrives week 47; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 46 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +3.4; anxiety Nadia +0.7 (witness), Ravi +1.3 (witness) |
| 46 | Nadia | CONFLICT | Marta | Pia, Ravi | arrives week 47; anxiety Marta +6.9, Nadia +0.5, Pia +0.8 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.3, Nadia +0.3 |
| 46 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Marta +2.2 (witness), Nadia +0.9 (witness) |
| 46 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 47; anxiety Marta +0.8, Nadia +1.6 (witness), Pia +1.5 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -1.0, Ravi +1.0; Marta left outside: anxiety Marta +1.9, Ravi -1.3 (of which 0.6 the outsider's own) |
| 46 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 48; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +2.0, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Sofia takes 2.0 of functioning from the other |
| 47 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 47 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 47 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 47 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 47 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 47 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 47 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 47 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 47 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.4; anxiety Pia +0.8 (witness), Ravi +1.4 (witness) |
| 47 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 48; anxiety Marta +0.6, Nadia +0.5, Pia +0.8 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.6, Nadia +0.6; Marta takes 2.0 of functioning from the other |
| 47 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Marta +2.2 (witness), Nadia +0.9 (witness) |
| 47 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | arrives week 49; anxiety Marta +1.1 (witness), Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +0.3, Sofia -0.7 |
| 47 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 48 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.1 (witness), Nadia +1.6 (witness), Pia +1.5 (witness), Ravi +2.8 |
| 48 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.1, Nadia +1.6 (witness), Pia +1.5 (witness), Ravi +1.4 (witness) |
| 48 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 48 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 48 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 49; anxiety Dr Halim +0.1, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7 |
| 48 | Marta | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 48 | Nadia | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 48 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 49; anxiety Marta +0.8, Nadia +0.7 (witness), Pia +0.5, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; Pia rejects the other on Marta–Pia (rejection +0.10; the tie stays open) |
| 48 | Ravi | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 48 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 48 | Nadia | SYMPTOM_ONSET | Marta, Ravi | Pia | arrives week 49; symptom onset, mental channel; anxiety Marta +3.2, Pia +0.8 (witness), Ravi +2.5 |
| 49 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 49 | Bruno | WITHHOLD | Ana | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 49 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +3.2; anxiety Nadia +0.7 (witness), Ravi +1.3 (witness) |
| 49 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 50; anxiety Marta +5.5, Nadia +0.6, Pia +0.8 (witness), Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4; Nadia takes 2.0 of functioning from the other |
| 49 | Pia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 49 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 50; anxiety Marta +3.7, Nadia +1.7 (witness), Pia +1.5 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.7, Ravi +0.7 |
| 49 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 51; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +2.6, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 50 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 50 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 50 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 50 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.6 (witness) |
| 50 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 50 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 51; anxiety Marta +0.7, Nadia -1.4, Pia +0.8 (witness), Ravi +1.5 (witness); Nadia left outside: anxiety Marta -2.9, Nadia +4.3 (of which 1.4 the outsider's own) |
| 50 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 50 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Marta +2.2 (witness), Nadia +0.9 (witness) |
| 50 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 51; anxiety Marta +1.3 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.1 |
| 50 | Sofia | PROVOKE | Ravi | Marta, Nadia, Pia | arrives week 52; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.9 (witness), Ravi +1.9, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 51 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 51 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 51 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 51 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 51 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 52; anxiety Marta +0.7, Nadia +1.5 (witness), Pia +1.4 (witness), Ravi -1.0; Ravi left outside: anxiety Marta -3.1, Ravi +4.6 (of which 1.5 the outsider's own) |
| 51 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 51 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Marta +2.3 (witness), Nadia +0.9 (witness) |
| 51 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | arrives week 52; anxiety Marta +1.8 (witness), Nadia -0.8, Pia +0.7 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Nadia holds a position: no functioning changes hands |
| 51 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 53; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.3, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 51 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |
| 52 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.6 (witness), Pia +1.5 (witness), Ravi +2.9 |
| 52 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.1, Nadia +1.6 (witness), Pia +1.5 (witness), Ravi +1.5 (witness) |
| 52 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 52 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 52 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 52 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 52 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 52 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.7 (witness) |
| 52 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 52 | Marta | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 52 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 53; anxiety Marta +0.5, Nadia +0.5, Pia +0.7 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.1, Nadia +0.1; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 52 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 53; anxiety Marta +3.8, Nadia +0.7 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7 |
| 52 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 53; anxiety Marta +2.1 (witness), Nadia +0.8 (witness), Pia +0.1, Ravi +0.5; 8.6 of the sender's anxiety bound into Pia–Ravi |
| 52 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 54; anxiety Marta +1.0 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6 |
| 53 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 53 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 53 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.3; anxiety Nadia +0.7 (witness), Ravi +1.3 (witness) |
| 53 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 54; anxiety Marta +1.7 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi -1.1; Ravi takes 2.0 of functioning from the other |
| 53 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Marta +2.2 (witness), Nadia +0.9 (witness) |
| 53 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | arrives week 55; anxiety Marta +1.0 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.3, Sofia +0.9; Sofia holds a position: no functioning changes hands |
| 53 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 55; anxiety Marta +1.0 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.4, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Sofia -1.4 |
| 54 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 55; anxiety Ana +0.3, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +1.0, Marta -1.0; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 54 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 54 | Marta | TRIANGLE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.3; anxiety Pia +0.7 (witness), Ravi +1.3 (witness) |
| 54 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 55; anxiety Marta +2.0 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +3.0 |
| 54 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 55; anxiety Marta +1.2, Nadia +0.7 (witness), Pia +0.5, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; Marta takes 2.0 of functioning from the other |
| 54 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 55; anxiety Marta +0.8, Nadia +1.5 (witness), Pia +1.4 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -1.0, Ravi +1.0; Ravi rejects the other on Marta–Ravi (rejection +0.50; the tie stays open) |
| 54 | Sofia | PROVOKE | Ravi | Marta, Nadia, Pia | arrives week 56; anxiety Marta +1.0 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +2.1, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 54 | — | (system) | — | — | Ravi–Sofia: rejection has faded; the tie is no longer cut off (ordinary) |
| 55 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 55 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 55 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 55 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 55 | Bruno | CUTOFF | Ana | — | arrives week 56; anxiety Ana +0.3, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open) |
| 55 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 56; anxiety Marta +0.4, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); Marta rejects the other on Ana–Marta (rejection +0.00; the tie stays open) |
| 55 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 55 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 56; anxiety Marta +1.0, Nadia +0.7 (witness), Pia +0.5, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Pia rejects the other on Marta–Pia (rejection +0.20; the tie stays open) |
| 55 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 56; anxiety Marta +1.9 (witness), Nadia +0.8 (witness), Pia +0.2, Ravi +0.5; 8.2 of the sender's anxiety bound into Pia–Ravi |
| 55 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 57; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.8 (witness), Ravi +0.4, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6 |
| 55 | — | (system) | — | — | Nadia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 56 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.9 (witness), Nadia +1.6 (witness), Pia +1.6 (witness), Ravi +2.8 |
| 56 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.8, Nadia +1.6 (witness), Pia +1.6 (witness), Ravi +1.4 (witness) |
| 56 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 56 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 56 | Dr Halim | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 57; anxiety Dr Halim +0.1, Marta +1.1 (witness), Nadia +0.8 (witness), Pia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.4, Ravi -0.4; Dr Halim rejects the other on Dr Halim–Ravi (rejection +0.50; the tie stays open) |
| 56 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +3.1; anxiety Nadia +0.7 (witness), Ravi +1.2 (witness) |
| 56 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 57; anxiety Marta +5.4, Nadia +0.5, Pia +0.8 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.3, Nadia +0.3; Nadia takes 2.0 of functioning from the other |
| 56 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 57; anxiety Marta +0.9, Nadia +0.7 (witness), Pia +0.5, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 56 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 57; anxiety Marta +6.4, Nadia +1.7 (witness), Pia +1.6 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.9, Ravi +0.9 |
| 56 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 58; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 57 | Ana | CUTOFF | Bruno | — | arrives week 58; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -0.7 |
| 57 | Bruno | CUTOFF | Ana | — | arrives week 58; anxiety Ana +0.7, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 57 | Marta | TRIANGLE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Marta +3.3; anxiety Nadia +1.5 (witness), Pia +1.5 (witness) |
| 57 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 58; anxiety Marta +1.9 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +3.1; 7.8 of the sender's anxiety bound into Nadia–Ravi |
| 57 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 58; anxiety Marta +1.9 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +0.0; calmer sender takes some anxiety: Pia +0.0, Ravi -0.0; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 57 | Ravi | WITHHOLD | Pia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 57 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 59; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 58 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 58 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 58 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 58 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 58 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 58 | Ana | CUTOFF | Bruno | — | arrives week 59; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 58 | Bruno | WITHHOLD | Ana | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 58 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.3; anxiety Nadia +0.7 (witness), Ravi +1.2 (witness) |
| 58 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 59; anxiety Marta +0.7, Nadia +0.5, Pia +0.8 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.8, Nadia +0.8; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 58 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 59; anxiety Marta +1.8 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; Pia rejects the other on Pia–Ravi (rejection +0.45; the tie stays open) |
| 58 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.3; anxiety Marta +1.8 (witness), Nadia +0.7 (witness) |
| 58 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 60; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6; Ravi takes 2.0 of functioning from the other |
| 59 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 59 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 59 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.3; anxiety Pia +0.7 (witness), Ravi +1.3 (witness) |
| 59 | Nadia | DISTANCE | Marta | Pia, Ravi | arrives week 60; anxiety Marta +0.6, Nadia +0.5, Pia +0.7 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.7, Nadia +0.7; 6.6 of the sender's anxiety bound into Marta–Nadia |
| 59 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 60; anxiety Marta +1.8 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +1.0; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; 8.1 of the sender's anxiety bound into Pia–Ravi |
| 59 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.7 (witness) |
| 59 | Sofia | REDUCE_CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 61; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.6, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia's rejection -0.25 |
| 60 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.6 (witness), Pia +1.6 (witness), Ravi +2.9 |
| 60 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.0, Nadia +1.6 (witness), Pia +1.6 (witness), Ravi +1.4 (witness) |
| 60 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 61; anxiety Ana +0.3, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +1.0, Marta -1.0; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 60 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 60 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 61; anxiety Marta +0.2, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 60 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 61; anxiety Marta +2.1 (witness), Nadia +0.5, Pia +0.8 (witness), Ravi +3.7; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 60 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 61; anxiety Marta +2.0 (witness), Nadia +0.8 (witness), Pia +0.5, Ravi +3.4; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia takes 2.0 of functioning from the other |
| 60 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | arrives week 61; anxiety Marta +0.6, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.7, Ravi +0.7; Marta takes 2.0 of functioning from the other |
| 60 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 62; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.2, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 61 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 61 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 61 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 61 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 61 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 61 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.0; anxiety Pia +0.7 (witness), Ravi +1.5 (witness) |
| 61 | Nadia | CONFLICT | Marta | Pia, Ravi | arrives week 62; anxiety Marta +6.4, Nadia +0.5, Pia +0.7 (witness), Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -0.6, Nadia +0.6 |
| 61 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 62; anxiety Marta +5.4, Nadia +0.7 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -1.2, Pia +1.2; Pia takes 2.0 of functioning from the other |
| 61 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.5; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.7 (witness) |
| 61 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 63; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 62 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.6 (witness) |
| 62 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 62 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 63; anxiety Marta +0.4, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.6 (witness); Marta rejects the other on Ana–Marta (rejection +0.20; the tie stays open) |
| 62 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 63; anxiety Marta +1.9 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.3; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Ravi takes 2.0 of functioning from the other |
| 62 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 63; anxiety Marta +2.1 (witness), Nadia +0.8 (witness), Pia +0.5, Ravi +4.2; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia takes 2.0 of functioning from the other |
| 62 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.5; anxiety Marta +1.0 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 62 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 64; anxiety Marta +1.0 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.7, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 63 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +1.4; anxiety Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.7 (witness) |
| 63 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 63 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 64; anxiety Marta +0.6, Nadia +0.8 (witness), Ravi +1.4 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -3.6 |
| 63 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 64; anxiety Marta +2.0 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +2.0; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 63 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 64; anxiety Marta +1.0, Nadia +0.8 (witness), Pia +0.4, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -1.1, Pia +1.1; Pia rejects the other on Marta–Pia (rejection +0.25; the tie stays open) |
| 63 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.5; anxiety Marta +1.1 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 63 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 65; anxiety Marta +1.1 (witness), Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +2.5, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia takes 2.0 of functioning from the other |
| 64 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +3.0 |
| 64 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.0, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +1.5 (witness) |
| 64 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 64 | Bruno | CUTOFF | Ana | — | arrives week 65; anxiety Ana +0.0, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 64 | Dr Halim | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 65; anxiety Dr Halim +0.1, Marta +1.3 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.6; calmer sender takes some anxiety: Dr Halim +0.6, Ravi -0.6; Dr Halim takes 2.0 of functioning from the other |
| 64 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 65; anxiety Marta +0.6, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi -0.0; Ravi left outside: anxiety Marta -2.9, Ravi +4.3 (of which 1.4 the outsider's own) |
| 64 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 65; anxiety Marta +1.9 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; 7.4 of the sender's anxiety bound into Nadia–Ravi |
| 64 | Pia | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 64 | Ravi | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 64 | Sofia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 65 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 65 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 65 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 65 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 65 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 65 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 65 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 65 | Bruno | CUTOFF | Ana | — | arrives week 66; anxiety Ana +0.4, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 65 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.2; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 65 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 66; anxiety Marta +1.8 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; 5.2 of the sender's anxiety bound into Nadia–Ravi |
| 65 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.2; anxiety Nadia +0.8 (witness), Ravi +1.7 (witness) |
| 65 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.7; anxiety Marta +2.0 (witness), Nadia +0.8 (witness) |
| 65 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 66 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 66 | Ana | WITHHOLD | Bruno | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 66 | Bruno | CUTOFF | Ana | — | arrives week 67; anxiety Ana +0.5, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 66 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +2.0; anxiety Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.9 (witness) |
| 66 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 67; anxiety Marta +1.7 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia +0.9, Ravi -0.9; 4.6 of the sender's anxiety bound into Nadia–Ravi |
| 66 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.2; anxiety Nadia +0.8 (witness), Ravi +1.6 (witness) |
| 66 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 67; anxiety Marta +0.5, Nadia +1.4 (witness), Pia +1.2 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.4, Ravi +0.4; 12.0 of the sender's anxiety bound into Marta–Ravi |
| 66 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 68; anxiety Marta +1.1 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +3.5, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 66 | — | (system) | — | — | Nadia–Ravi tie now distant |
| 67 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 67 | Bruno | CUTOFF | Ana | — | arrives week 68; anxiety Ana +0.5, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 67 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 68; anxiety Marta +0.6, Nadia +1.7 (witness), Pia +1.4 (witness), Ravi +4.5 |
| 67 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 68; anxiety Marta +2.0 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +4.5; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Nadia takes 2.0 of functioning from the other |
| 67 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.2; anxiety Nadia +0.8 (witness), Ravi +1.5 (witness) |
| 67 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.7; anxiety Marta +1.9 (witness), Nadia +0.8 (witness) |
| 67 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 69; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +3.2, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia takes 2.0 of functioning from the other |
| 67 | — | (system) | — | — | Dr Halim–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 67 | — | (system) | — | — | Marta–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 68 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +3.4 |
| 68 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.0, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +1.7 (witness) |
| 68 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 68 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 68 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 68 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 68 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 69; anxiety Marta +0.7, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 68 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 69; anxiety Marta +1.9 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5; 6.3 of the sender's anxiety bound into Nadia–Ravi |
| 68 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 69; anxiety Marta +0.2, Nadia +0.7 (witness), Pia +0.4, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -1.2, Pia +1.2; Pia rejects the other on Marta–Pia (rejection +0.25; the tie stays open) |
| 68 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 69; anxiety Marta +6.8, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.4, Ravi +0.4 |
| 68 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 70; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +2.2, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 69 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 69 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 69 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 69 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +3.5; anxiety Nadia +0.7 (witness), Ravi +1.7 (witness) |
| 69 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 70; anxiety Marta +2.2 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +4.6; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 69 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 70; anxiety Marta +2.2 (witness), Nadia +0.8 (witness), Pia +0.5, Ravi +6.2; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7 |
| 69 | Ravi | WITHHOLD | Sofia | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 69 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 71; anxiety Marta +1.3 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +3.7, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1 |
| 70 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.0 (witness) |
| 70 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 70 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +3.5; anxiety Nadia +0.7 (witness), Ravi +1.6 (witness) |
| 70 | Nadia | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 70 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 70 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 71; anxiety Marta +4.1, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.4, Ravi +0.4 |
| 70 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 72; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.7, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia rejects the other on Ravi–Sofia (rejection +0.35; the tie stays open) |
| 70 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 71, 72; symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +3.5, Nadia +2.6, Pia +2.3; anxiety Sofia +1.1 |
| 71 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 71 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 71 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 72; anxiety Marta +0.2, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +1.0 (witness) |
| 71 | Nadia | CONFLICT | Marta | Pia, Ravi | arrives week 72; anxiety Marta +7.0, Nadia +0.5, Pia +0.7 (witness), Ravi +1.9 (witness); calmer sender takes some anxiety: Marta -0.8, Nadia +0.8 |
| 71 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.2; anxiety Nadia +0.7 (witness), Ravi +1.7 (witness) |
| 71 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 72; anxiety Marta +5.0, Nadia +1.7 (witness), Pia +1.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.4, Ravi +0.4; Ravi takes 2.0 of functioning from the other |
| 71 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 73; anxiety Marta +1.3 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +3.0, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia takes 2.0 of functioning from the other |
| 72 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.1 (witness), Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +3.5 |
| 72 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.2, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +1.8 (witness) |
| 72 | Ana | WITHHOLD | Bruno | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 72 | Bruno | CUTOFF | Ana | — | arrives week 73; anxiety Ana +0.0, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 72 | Dr Halim | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 73; anxiety Dr Halim +0.1, Marta +1.2, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Dr Halim +0.8, Marta -0.8; Dr Halim takes 1.6 of functioning from the other |
| 72 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 73; anxiety Marta +0.7, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +0.5; Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -7.3 |
| 72 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 73; anxiety Marta +2.2 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 72 | Pia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 72 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.7; anxiety Marta +1.4 (witness), Nadia +0.9 (witness), Pia +0.8 (witness) |
| 72 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 73 | Ana | CUTOFF | Bruno | — | arrives week 74; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.6, Bruno -0.6; Ana rejects the other on Ana–Bruno (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ana -0.2 |
| 73 | Bruno | CUTOFF | Ana | — | arrives week 74; anxiety Ana +0.2, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 73 | Marta | WITHHOLD | Pia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 73 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 74; anxiety Marta +2.2 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Ravi takes 2.0 of functioning from the other |
| 73 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.2; anxiety Nadia +0.7 (witness), Ravi +1.6 (witness) |
| 73 | Ravi | WITHHOLD | Pia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 73 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 75; anxiety Marta +1.3 (witness), Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.6, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 74 | Ana | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 74 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 74 | Marta | TRIANGLE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.8; anxiety Pia +0.7 (witness), Ravi +1.6 (witness) |
| 74 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 75; anxiety Marta +1.0, Nadia +0.5, Pia +0.7 (witness), Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4; Marta takes 2.0 of functioning from the other |
| 74 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 75; anxiety Marta +2.5 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +3.2; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 74 | Ravi | TRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.7; anxiety Nadia +1.6 (witness), Pia +1.3 (witness) |
| 74 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 76; anxiety Marta +1.2 (witness), Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +3.1, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia takes 2.0 of functioning from the other |
| 75 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 75 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 75 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 75 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 75 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 75 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 75 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +0.9 (witness) |
| 75 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 75 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 76; anxiety Marta +0.2, Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +1.1 (witness) |
| 75 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 75 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 76; anxiety Marta +2.3 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +3.2; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7 |
| 75 | Ravi | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 75 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 77; anxiety Marta +1.2 (witness), Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +3.3, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia takes 2.0 of functioning from the other |
| 76 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.2 (witness), Nadia +1.7 (witness), Pia +1.4 (witness), Ravi +3.3 |
| 76 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.4, Nadia +1.7 (witness), Pia +1.4 (witness), Ravi +1.6 (witness) |
| 76 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +1.0 (witness) |
| 76 | Bruno | WITHHOLD | Ana | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 76 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.5; anxiety Pia +0.7 (witness), Ravi +1.7 (witness) |
| 76 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 77; anxiety Marta +2.2 (witness), Nadia +0.6, Pia +0.7 (witness), Ravi +5.0; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Nadia takes 2.0 of functioning from the other |
| 76 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.2; anxiety Nadia +0.8 (witness), Ravi +1.5 (witness) |
| 76 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 77; anxiety Marta +2.1 (witness), Nadia +0.9 (witness), Ravi +0.6; 13.5 of the sender's anxiety bound into Pia–Ravi |
| 76 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 78; anxiety Marta +1.3 (witness), Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +3.4, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia takes 2.0 of functioning from the other |
| 77 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +1.1 (witness) |
| 77 | Bruno | CUTOFF | Ana | — | arrives week 78; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open) |
| 77 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.5; anxiety Pia +0.7 (witness), Ravi +2.0 (witness) |
| 77 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 78; anxiety Marta +2.4 (witness), Nadia +0.5, Pia +0.8 (witness), Ravi +4.2 |
| 77 | Pia | TRIANGLE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.2; anxiety Nadia +0.7 (witness), Ravi +1.7 (witness) |
| 77 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.0; anxiety Marta +1.3 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 77 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 79; anxiety Marta +1.3 (witness), Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +3.1, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 78 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 78 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 78 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 78 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 78 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.5; anxiety Nadia +0.8 (witness), Ravi +1.9 (witness) |
| 78 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 79; anxiety Marta +2.2 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +1.0; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Ravi takes 2.0 of functioning from the other |
| 78 | Pia | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 78 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 79; anxiety Marta +2.0 (witness), Nadia +0.8 (witness), Pia +0.2, Ravi +0.7; 11.4 of the sender's anxiety bound into Pia–Ravi |
| 78 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 80; anxiety Marta +1.3 (witness), Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +2.4, Sofia +0.2; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia takes 2.0 of functioning from the other |
| 79 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 79 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 79 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 80; anxiety Marta +0.7, Nadia +1.6 (witness), Pia +1.3 (witness), Ravi +0.3; Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open) |
| 79 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 80; anxiety Marta +2.0 (witness), Nadia +0.6, Pia +0.7 (witness), Ravi +0.1; Ravi takes 2.0 of functioning from the other |
| 79 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 80; anxiety Marta +2.2 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +4.7; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia takes 2.0 of functioning from the other |
| 79 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 80; anxiety Marta +2.0 (witness), Nadia +0.0, Pia +0.7 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.2, Ravi +0.2; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -5.7 |
| 79 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 80 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.1 (witness), Nadia +1.7 (witness), Pia +1.4 (witness), Ravi +3.9 |
| 80 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.3, Nadia +1.7 (witness), Pia +1.4 (witness), Ravi +1.9 (witness) |
| 80 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 80 | Bruno | CUTOFF | Ana | — | arrives week 81; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 80 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 81; anxiety Dr Halim +0.1, Nadia +0.9 (witness), Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Dr Halim +0.9, Marta -0.9; the contact lands with Marta: systems perspective +0.10 |
| 80 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.5; anxiety Nadia +0.8 (witness), Ravi +1.8 (witness) |
| 80 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 81; anxiety Marta +6.1, Nadia +0.6, Pia +0.7 (witness), Ravi +2.2 (witness); calmer sender takes some anxiety: Marta -0.8, Nadia +0.8; Nadia takes 2.0 of functioning from the other |
| 80 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 81; anxiety Marta +2.2 (witness), Nadia +1.0 (witness), Pia +0.4, Ravi +4.3; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; Pia takes 2.0 of functioning from the other |
| 80 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 81; anxiety Marta +2.2 (witness), Nadia +1.0 (witness), Pia +0.2, Ravi +0.7; 10.6 of the sender's anxiety bound into Pia–Ravi |
| 80 | Sofia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 81 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 81 | Ana | CUTOFF | Bruno | — | arrives week 82; anxiety Ana +0.4, Bruno +0.1; calmer sender takes some anxiety: Ana +0.6, Bruno -0.6; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 81 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 81 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.7; anxiety Pia +0.7 (witness), Ravi +2.4 (witness) |
| 81 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 82; anxiety Marta +1.1, Nadia +0.6, Pia +0.7 (witness), Ravi +2.4 (witness); calmer sender takes some anxiety: Marta -2.6, Nadia +2.6; Marta takes 2.0 of functioning from the other |
| 81 | Pia | TRIANGLE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.1; anxiety Nadia +0.8 (witness), Ravi +1.9 (witness) |
| 81 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 82; anxiety Marta +2.5 (witness), Nadia +1.0 (witness), Pia +3.6, Ravi +0.7 |
| 81 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 83; anxiety Marta +1.2 (witness), Nadia +1.1 (witness), Pia +0.8 (witness), Ravi +4.1, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.9, Sofia +1.9; Sofia takes 2.0 of functioning from the other |
| 82 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 82 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 82 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 82 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 82 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.4; anxiety Pia +0.7 (witness), Ravi +2.4 (witness) |
| 82 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +2.2 (witness), Pia +0.7 (witness) |
| 82 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 83; anxiety Marta +2.1 (witness), Nadia +1.0 (witness), Pia +0.4, Ravi +0.9; calmer sender takes some anxiety: Pia +2.1, Ravi -2.1; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -3.3 |
| 82 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 83; anxiety Marta +2.1 (witness), Nadia +1.0 (witness), Pia +1.1, Ravi +0.7; 19.5 of the sender's anxiety bound into Pia–Ravi |
| 82 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 82 | — | (system) | — | — | Pia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 83 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +1.1 (witness), Pia +0.7 (witness), Ravi +1.4 (witness) |
| 83 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 83 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 84; anxiety Marta +0.7, Nadia +0.8 (witness), Ravi +2.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 83 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +2.4 (witness), Pia +0.7 (witness) |
| 83 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.1; anxiety Nadia +0.8 (witness), Ravi +2.0 (witness) |
| 83 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.9; anxiety Marta +2.0 (witness), Nadia +0.9 (witness) |
| 83 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 85; anxiety Marta +1.1 (witness), Nadia +1.0 (witness), Pia +0.7 (witness), Ravi +1.0, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.6, Sofia +1.6; Sofia rejects the other on Ravi–Sofia (rejection +0.40; the tie stays open) |
| 84 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.1 (witness), Nadia +1.8 (witness), Pia +1.2 (witness), Ravi +4.8 |
| 84 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.2, Nadia +1.8 (witness), Pia +1.2 (witness), Ravi +2.4 (witness) |
| 84 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 84 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 84 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 84 | Marta | STAY-IN-CONTACT | Ravi | Nadia, Pia | arrives week 85; anxiety Marta +0.7, Nadia +1.6 (witness), Pia +1.1 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2 |
| 84 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +2.4 (witness), Pia +0.8 (witness) |
| 84 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.1; anxiety Nadia +0.8 (witness), Ravi +2.0 (witness) |
| 84 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +2.3; anxiety Marta +1.1 (witness), Nadia +1.0 (witness), Pia +0.7 (witness) |
| 84 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 86; anxiety Marta +1.1 (witness), Nadia +1.0 (witness), Pia +0.7 (witness), Ravi +5.1, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.5, Sofia +1.5 |
| 84 | Nadia | SYMPTOM_ONSET | Marta, Ravi | Pia | arrives week 85; symptom onset, mental channel; anxiety Marta +3.4, Pia +0.8 (witness), Ravi +3.9 |
| 85 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +1.1 (witness), Pia +0.8 (witness), Ravi +1.4 (witness) |
| 85 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 85 | Marta | STAY-IN-CONTACT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.3; anxiety Pia +0.6 (witness), Ravi +2.4 (witness) |
| 85 | Nadia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 85 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.1; anxiety Nadia +0.9 (witness), Ravi +2.3 (witness) |
| 85 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +2.3; anxiety Marta +1.1 (witness), Nadia +1.0 (witness), Pia +0.7 (witness) |
| 85 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 86 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +1.0 (witness), Pia +0.7 (witness), Ravi +1.3 (witness) |
| 86 | Bruno | CUTOFF | Ana | — | arrives week 87; anxiety Ana +0.0, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 86 | Marta | STAY-IN-CONTACT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.3; anxiety Pia +0.6 (witness), Ravi +2.2 (witness) |
| 86 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 87; anxiety Marta +2.2 (witness), Nadia +0.6, Pia +0.7 (witness), Ravi +0.8; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 86 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 87; anxiety Marta +2.2 (witness), Nadia +1.0 (witness), Pia +0.4, Ravi +8.3; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 86 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 88; anxiety Marta +1.0 (witness), Nadia +0.9 (witness), Pia +0.6 (witness), Ravi +0.5; Ravi rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -3.6 |
| 86 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 87 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 87 | Ana | CUTOFF | Bruno | — | arrives week 88; anxiety Ana +0.4, Bruno +0.3; calmer sender takes some anxiety: Ana +0.2, Bruno -0.2; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ana -0.4 |
| 87 | Bruno | CUTOFF | Ana | — | arrives week 88; anxiety Ana +0.4, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Bruno -0.3 |
| 87 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 88; anxiety Marta +0.7, Nadia +0.9 (witness), Ravi +2.2 (witness); Marta rejects the other on Marta–Pia (rejection +0.10; the tie stays open) |
| 87 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 88; anxiety Marta +1.1, Nadia +0.6, Pia +0.6 (witness), Ravi +2.1 (witness); Marta holds a position: no functioning changes hands |
| 87 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 88; anxiety Marta +2.2 (witness), Nadia +1.0 (witness), Pia +0.4, Ravi +4.8; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 87 | Ravi | TRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.9; anxiety Nadia +1.6 (witness), Pia +1.1 (witness) |
| 87 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 87 | Ana | SYMPTOM_ONSET | Bruno, Marta | Nadia, Pia, Ravi | arrives week 88; symptom onset, physical channel; anxiety Bruno +2.7, Marta +2.0, Nadia +1.0 (witness), Pia +0.7 (witness), Ravi +1.3 (witness) |
| 88 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.9 (witness), Pia +1.2 (witness), Ravi +4.8 |
| 88 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.1, Nadia +1.8 (witness), Pia +1.2 (witness), Ravi +2.4 (witness) |
| 88 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 88 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 88 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 88 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 88 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 88 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 88 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +1.1 (witness), Pia +0.7 (witness), Ravi +1.4 (witness) |
| 88 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 88 | Dr Halim | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 89; anxiety Dr Halim +0.1, Marta +1.4 (witness), Nadia +1.0 (witness), Pia +0.7 (witness), Ravi +1.3; calmer sender takes some anxiety: Dr Halim +0.6, Ravi -0.6; Dr Halim takes 1.2 of functioning from the other |
| 88 | Marta | PROVOKE | Ravi | Nadia, Pia | arrives week 89; anxiety Marta +0.7, Nadia +1.8 (witness), Pia +1.2 (witness), Ravi +5.1 |
| 88 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +2.3 (witness), Pia +0.7 (witness) |
| 88 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +2.1; anxiety Nadia +0.8 (witness), Ravi +2.1 (witness) |
| 88 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.9; anxiety Marta +2.0 (witness), Nadia +0.9 (witness) |
| 88 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 88 | Pia | SYMPTOM_ONSET | Marta, Ravi | Nadia | arrives week 89; symptom onset, social channel; anxiety Marta +3.3, Nadia +0.9 (witness), Ravi +3.9 |
| 89 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 89 | Ana | CUTOFF | Bruno | — | arrives week 90; anxiety Ana +0.4, Bruno +0.3; calmer sender takes some anxiety: Ana +0.5, Bruno -0.5; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 89 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 89 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.3; anxiety Pia +0.6 (witness), Ravi +2.1 (witness) |
| 89 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +2.1 (witness), Pia +0.7 (witness) |
| 89 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 90; anxiety Marta +2.0 (witness), Nadia +0.9 (witness), Pia +0.4, Ravi +4.8; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4 |
| 89 | Ravi | TRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.9; anxiety Nadia +1.7 (witness), Pia +1.1 (witness) |
| 89 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 91; anxiety Marta +0.9 (witness), Nadia +0.9 (witness), Pia +0.6 (witness), Sofia +0.2; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia rejects the other on Ravi–Sofia (rejection +0.20; the tie stays open) |
| 90 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 90 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 90 | Bruno | CUTOFF | Ana | — | arrives week 91; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Bruno -0.1 |
| 90 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.2; anxiety Pia +0.6 (witness), Ravi +2.2 (witness) |
| 90 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +2.2 (witness), Pia +0.7 (witness) |
| 90 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 91; anxiety Marta +2.0 (witness), Nadia +0.9 (witness), Pia +0.4, Ravi +4.5; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4 |
| 90 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.9; anxiety Marta +2.0 (witness), Nadia +0.9 (witness) |
| 90 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 91 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 91 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.6; anxiety Nadia +1.1 (witness), Pia +0.7 (witness), Ravi +1.3 (witness) |
| 91 | Bruno | CUTOFF | Ana | — | arrives week 92; anxiety Ana +0.3, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 91 | Marta | PREVENT_ALIGNMENT | Ravi | Nadia, Pia | arrives week 92; anxiety Marta +0.6, Nadia +1.7 (witness), Pia +1.1 (witness), Ravi +0.7 |
| 91 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +2.0 (witness), Pia +0.6 (witness) |
| 91 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 92; anxiety Marta +1.9 (witness), Nadia +0.9 (witness), Pia +0.4, Ravi +3.4; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6 |
| 91 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.8; anxiety Marta +1.9 (witness), Nadia +0.9 (witness) |
| 91 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 92 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.9 (witness), Pia +1.3 (witness), Ravi +4.6 |
| 92 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.0, Nadia +1.9 (witness), Pia +1.2 (witness), Ravi +2.3 (witness) |
| 92 | Ana | CUTOFF | Bruno | — | arrives week 93; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +0.5, Bruno -0.5; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -0.0 |
| 92 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 92 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 93; anxiety Marta +0.6, Nadia +0.8 (witness), Ravi +2.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.20; the tie stays open) |
| 92 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +2.2 (witness), Pia +0.7 (witness) |
| 92 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 93; anxiety Marta +1.8 (witness), Nadia +0.9 (witness), Pia +0.4, Ravi +1.2; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; 10.9 of the sender's anxiety bound into Pia–Ravi |
| 92 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.8; anxiety Marta +1.8 (witness), Nadia +0.9 (witness) |
| 92 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 94; anxiety Marta +0.9 (witness), Nadia +0.9 (witness), Pia +0.6 (witness), Sofia +0.2; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; Ravi takes 2.0 of functioning from the other |
| 93 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 93 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 93 | Ana | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 93 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 93 | Marta | PROVOKE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.2; anxiety Pia +0.6 (witness), Ravi +2.2 (witness) |
| 93 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 94; anxiety Marta +0.8, Nadia +0.6, Pia +0.6 (witness), Ravi +2.2 (witness); calmer sender takes some anxiety: Marta -0.2, Nadia +0.2; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 93 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 94; anxiety Marta +0.3, Nadia +0.8 (witness), Pia +0.4, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -1.7, Pia +1.7; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 93 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.8; anxiety Marta +1.7 (witness), Nadia +0.8 (witness) |
| 93 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 95; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +0.1, Sofia +0.2; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia rejects the other on Ravi–Sofia (rejection +0.20; the tie stays open) |
| 94 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 94 | Ana | WITHHOLD | Bruno | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 94 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 94 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 95; anxiety Marta +0.6, Nadia +1.6 (witness), Pia +1.1 (witness), Ravi +0.7; 13.8 of the sender's anxiety bound into Marta–Ravi |
| 94 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +2.0 (witness), Pia +0.7 (witness) |
| 94 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 95; anxiety Marta +2.0 (witness), Nadia +1.0 (witness), Pia +0.4, Ravi +6.5; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2 |
| 94 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 95; anxiety Nadia +1.6 (witness), Pia +1.1 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.2, Ravi +0.2; Marta left outside: anxiety Marta +4.8, Ravi -3.2 (of which 1.6 the outsider's own) |
| 94 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 95 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 95 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 95 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 95 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 95 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 95 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 95 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.1 |
| 95 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 95 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.1; anxiety Pia +0.6 (witness), Ravi +2.1 (witness) |
| 95 | Nadia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 95 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.1; anxiety Nadia +0.9 (witness), Ravi +2.1 (witness) |
| 95 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 96; anxiety Marta +1.8 (witness), Pia +0.6 (witness), Ravi +0.7; 13.9 of the sender's anxiety bound into Nadia–Ravi |
| 95 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 96 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.9 (witness), Nadia +1.9 (witness), Pia +1.3 (witness), Ravi +4.3 |
| 96 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.8, Nadia +1.9 (witness), Pia +1.3 (witness), Ravi +2.1 (witness) |
| 96 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.1 |
| 96 | Bruno | CUTOFF | Ana | — | arrives week 97; anxiety Ana +0.1, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 96 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 97; anxiety Dr Halim +0.1, Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.50; the tie stays open) |
| 96 | Marta | WITHHOLD | Nadia | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 96 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 97; anxiety Marta +0.7, Nadia +0.6, Pia +0.6 (witness), Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -0.1, Nadia +0.1; Marta holds a position: no functioning changes hands |
| 96 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 96 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 98; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.5 (witness), Ravi +0.4; Ravi rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 96 | Sofia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 97 | Ana | WITHHOLD | Bruno | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 97 | Bruno | CUTOFF | Ana | — | arrives week 98; anxiety Ana +0.5, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 97 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 98; anxiety Marta +0.6, Nadia +1.8 (witness), Pia +1.2 (witness), Ravi +6.6 |
| 97 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +1.7 (witness), Pia +0.6 (witness) |
| 97 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 98; anxiety Marta +2.0 (witness), Nadia +1.0 (witness), Pia +0.4, Ravi +6.2; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia takes 2.0 of functioning from the other |
| 97 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 98; anxiety Marta +2.6, Nadia +1.8 (witness), Pia +1.2 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3 |
| 97 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 99; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.5 (witness), Sofia +0.2; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia rejects the other on Ravi–Sofia (rejection +0.20; the tie stays open) |
| 98 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 98 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 98 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 98 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.1 |
| 98 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 98 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.1; anxiety Pia +0.6 (witness), Ravi +2.2 (witness) |
| 98 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +1.7 (witness), Pia +0.5 (witness) |
| 98 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 98 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 100; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.5 (witness), Ravi +0.4; Ravi rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 98 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 100; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.5 (witness), Sofia +0.2; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 98 | — | (system) | — | — | Marta–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 99 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.1 |
| 99 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 99 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.1; anxiety Pia +0.6 (witness), Ravi +2.2 (witness) |
| 99 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 100; anxiety Marta +0.4, Nadia +0.6, Pia +0.6 (witness), Ravi +2.2 (witness); calmer sender takes some anxiety: Marta -0.2, Nadia +0.2; Nadia rejects the other on Marta–Nadia (rejection +0.00; the tie stays open) |
| 99 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.0; anxiety Nadia +0.9 (witness), Ravi +2.1 (witness) |
| 99 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 100; anxiety Marta +0.3, Nadia +1.7 (witness), Pia +1.1 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.4, Ravi +0.4; 13.1 of the sender's anxiety bound into Marta–Ravi |
| 99 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 100 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.9 (witness), Nadia +1.9 (witness), Pia +1.2 (witness), Ravi +4.6 |
| 100 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.8, Nadia +1.9 (witness), Pia +1.2 (witness), Ravi +2.3 (witness) |
| 100 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.1 |
| 100 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 100 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | arrives week 101; anxiety Ana +0.0, Marta +0.4, Nadia +1.0 (witness), Pia +0.6 (witness), Ravi +1.1 (witness); 13.1 of the sender's anxiety bound into Ana–Marta |
| 100 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.1; anxiety Marta +1.6 (witness), Pia +0.5 (witness) |
| 100 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 101; anxiety Marta +2.0 (witness), Nadia +1.0 (witness), Pia +0.4, Ravi +3.2; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6 |
| 100 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 101; anxiety Marta +5.1, Nadia +1.9 (witness), Pia +1.2 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.5, Ravi +0.5; Ravi takes 2.0 of functioning from the other |
| 100 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 101 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 102; anxiety Ana +0.3, Marta +0.0, Nadia +0.9 (witness), Pia +0.6 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 101 | Bruno | CUTOFF | Ana | — | arrives week 102; anxiety Ana +0.2, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 101 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.2; anxiety Pia +0.6 (witness), Ravi +2.1 (witness) |
| 101 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 101 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 101 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 102; anxiety Marta +1.7 (witness), Nadia +0.1, Pia +0.5 (witness), Ravi +0.7; 9.2 of the sender's anxiety bound into Nadia–Ravi |
| 101 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 101 | — | (system) | — | — | Ana–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 102 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.1 |
| 102 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 102 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +2.0; anxiety Nadia +0.9 (witness), Pia +0.6 (witness), Ravi +0.9 (witness) |
| 102 | Nadia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 102 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 103; anxiety Marta +2.0 (witness), Nadia +0.9 (witness), Pia +0.4, Ravi +0.9; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; 6.3 of the sender's anxiety bound into Pia–Ravi |
| 102 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | — |
| 102 | Sofia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 103 | Ana | OVERFUNCTION | Bruno | — | — |
| 103 | Bruno | CONFLICT | Ana | — | — |
| 103 | Marta | TRIANGLE | Ravi | Nadia, Pia | — |
| 103 | Nadia | CUTOFF | Marta | Pia, Ravi | — |
| 103 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | — |
| 103 | Ravi | TRIANGLE | Pia | Marta, Nadia | — |
| 103 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | — |
| 103 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |

104 weeks. The invariants were asserted at the end of every week.
