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
| config hash | `f853e082d890d1bbd21f70abaeaba468b15fb3f017cd42ca9998537b194d908f` |
| spec revision | 2.0, revision 28 |
| family instance | phase_c |
| activation | synchronous_activation v1 (synchronous) |
| visibility | household_conductance_visibility v1 |
| constants frozen | 2026-10-07 |
| constants changed since freeze | acting_age_years, attribution_rate, belief_appraisal_gain, birth_focus_gain, budget_per_undifferentiation, channel_protection, coach_retirement_age_years, coach_start_age_years, constitution_strength, death_transfer_share, declared_rejection, development_band, development_end_age_years, development_rate, development_start_age_years, estimator_breadth, estimator_rate, estimator_sd_weight, estimator_window, exchange_window, find_work_pressure_scale, find_work_rate, focus_rate, grief_extra_weeks, grief_min_weeks, grief_scale, income_loss_intensity, income_shortfall_share, launch_age_years, lock_in_gain, lock_in_turn, marital_distance_fraction, mortality_a, mortality_b, new_tie_investment_share, pattern_min_acts, pattern_projection_hold, pattern_window, perspective_channel_gain, perspective_steer_gain, practice_window, pseudo_self_relaxation_rate, refusal_cost_gain, rejection_cutoff_gain, rejection_fade, rejection_refusal_gain, rejection_refusal_threshold, rejection_relief, response_down_gain, response_relaxation_rate, response_up_gain, self_practice_gain, soothing_review_every, soothing_review_penalty, stress_floor_below_basic, stressor_persistence, system_adjust_gain, tie_wear_rate, usual_level_rate, wear_offset_per_act, witnessed_gain, working_age_years |

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
| 2 | Ravi | STAY-IN-CONTACT | Nadia | Marta, Pia | arrives week 3; anxiety Marta +1.3 (witness), Nadia -0.7, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3 |
| 2 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 4; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +2.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 1.2 of the sender's anxiety bound into Ravi–Sofia |
| 3 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 3 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 4; anxiety Ana +0.3, Marta +2.1, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.3, Marta -0.3; 2.0 of the sender's anxiety bound into Ana–Marta |
| 3 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 3 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 4; anxiety Marta +0.5, Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +1.1; calmer sender takes some anxiety: Marta +0.5, Ravi -0.5 |
| 3 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Pia +0.6 (witness), Ravi +1.7 (witness) |
| 3 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 4; anxiety Marta +3.1, Nadia +0.7 (witness), Pia +0.4, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5 |
| 3 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | arrives week 4; anxiety Marta +1.3 (witness), Nadia +1.2, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.0, Ravi +0.0; Nadia takes 2.0 of functioning from the other |
| 3 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 5; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +3.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia takes 2.0 of functioning from the other |
| 4 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +3.4 |
| 4 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +1.7 (witness) |
| 4 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 4 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 4 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 4 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 4 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 5; anxiety Marta +0.5, Nadia +1.0, Pia +0.6 (witness), Ravi +1.7 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3; Marta rejects the other on Marta–Nadia (rejection +0.45; the tie stays open) |
| 4 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 5; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +5.8; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Nadia takes 2.0 of functioning from the other |
| 4 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 5; anxiety Marta +1.4 (witness), Nadia +0.7 (witness), Pia +0.4, Ravi +0.7; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Ravi left outside: anxiety Pia -0.9, Ravi +1.4 (of which 0.5 the outsider's own) |
| 4 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 5; anxiety Marta +1.4 (witness), Nadia +0.7 (witness), Pia +2.8, Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 4 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 6; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; 1.4 of the sender's anxiety bound into Ravi–Sofia |
| 5 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 5 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 5 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +0.3, Marta +2.5, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana takes 2.0 of functioning from the other |
| 5 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 5 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +2.1, Marta +0.3, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.9 (witness) |
| 5 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 6; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.7; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Ravi left outside: anxiety Nadia -1.2, Ravi +1.8 (of which 0.6 the outsider's own) |
| 5 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 6; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.4; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Ravi left outside: anxiety Pia -1.1, Ravi +1.6 (of which 0.5 the outsider's own) |
| 5 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | arrives week 6; anxiety Marta +1.4 (witness), Nadia +1.0, Pia +0.6 (witness), Ravi +0.6; Nadia takes 2.0 of functioning from the other |
| 5 | Sofia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 5 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 6, 7; symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.5, Nadia +2.2, Pia +2.3; anxiety Sofia +1.3 |
| 6 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 6 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.1 |
| 6 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 6 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | arrives week 7; anxiety Marta +0.5, Nadia +0.6 (witness), Pia +3.5, Ravi +2.1 (witness); Marta takes 2.0 of functioning from the other |
| 6 | Nadia | PREVENT_ALIGNMENT | Ravi | Marta, Pia | arrives week 7; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.6; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4 |
| 6 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 7; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.4; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Ravi takes 2.0 of functioning from the other |
| 6 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 7; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 6 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 8; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +2.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 7 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 8; anxiety Ana +0.3, Marta +2.5, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana takes 2.0 of functioning from the other |
| 7 | Bruno | CUTOFF | Ana | — | arrives week 8; anxiety Ana +0.6, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 7 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | arrives week 8; anxiety Marta +0.5, Nadia -0.0, Pia +0.7 (witness), Ravi +1.8 (witness); calmer sender takes some anxiety: Marta +0.1, Nadia -0.1; Nadia takes 2.0 of functioning from the other |
| 7 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 8; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.0; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 7 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 8; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.0; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Pia–Ravi functioning balance flips; Ravi takes 2.0 of functioning from the other |
| 7 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | arrives week 9; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +0.4, Sofia +1.7 |
| 7 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +2.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1 |
| 8 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.9 |
| 8 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.9 (witness) |
| 8 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.3, Marta +2.6, Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 8 | Bruno | CUTOFF | Ana | — | arrives week 9; anxiety Ana +0.8, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 8 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Dr Halim +0.1, Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Ravi -0.7 |
| 8 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.5, Marta +0.3, Nadia +0.6 (witness), Pia +0.8 (witness), Ravi +0.9 (witness); Ana holds a position: no functioning changes hands |
| 8 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 9; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; 6.5 of the sender's anxiety bound into Nadia–Ravi |
| 8 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 9; anxiety Marta +5.3, Nadia +0.7 (witness), Pia +0.5, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Pia takes 2.0 of functioning from the other |
| 8 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 9; anxiety Marta +1.6 (witness), Nadia +0.6 (witness), Pia +2.8, Ravi +0.6 |
| 8 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 10; anxiety Marta +1.0 (witness), Nadia +0.7 (witness), Pia +0.9 (witness), Ravi +3.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia takes 2.0 of functioning from the other |
| 8 | Marta | SYMPTOM_ONSET | Ana, Dr Halim, Nadia, Pia, Ravi | — | arrives week 9; symptom onset, physical channel; anxiety Ana +1.5, Dr Halim +0.5, Nadia +2.0, Pia +2.6, Ravi +3.0 |
| 9 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 9 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 9 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 10; anxiety Ana +0.3, Marta +3.5, Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana takes 2.0 of functioning from the other |
| 9 | Bruno | CUTOFF | Ana | — | arrives week 10; anxiety Ana +0.7, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 9 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 10; anxiety Marta +0.6, Nadia +1.5, Pia +0.7 (witness), Ravi +1.7 (witness) |
| 9 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 10; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.0; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6 |
| 9 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 10; anxiety Marta +1.7 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.1; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -6.1 |
| 9 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 10; anxiety Marta +1.7 (witness), Nadia +0.6 (witness), Pia +3.5, Ravi +0.6 |
| 9 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 11; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +2.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 10 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 10 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 10 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 11; anxiety Marta +0.6, Nadia +0.6 (witness), Pia +3.5, Ravi +2.0 (witness); 10.8 of the sender's anxiety bound into Marta–Pia |
| 10 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 11; anxiety Marta +1.6 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.0; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Ravi left outside: anxiety Nadia -0.9, Ravi +1.3 (of which 0.4 the outsider's own) |
| 10 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 11; anxiety Marta +2.0 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +6.1; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Pia takes 2.0 of functioning from the other |
| 10 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 11; anxiety Marta +3.2, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.6 |
| 10 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 12; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +3.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1 |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 11 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 12; anxiety Ana +0.3, Marta +0.8, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 11 | Bruno | CUTOFF | Ana | — | arrives week 12; anxiety Ana +0.6, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 11 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 12; anxiety Marta +0.6, Nadia +0.7 (witness), Pia +4.1, Ravi +2.3 (witness) |
| 11 | Nadia | PREVENT_ALIGNMENT | Ravi | Marta, Pia | arrives week 12; anxiety Marta +1.6 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.9; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6 |
| 11 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 11 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 12; anxiety Dr Halim +0.0, Marta +1.1 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 11 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 13; anxiety Marta +1.0 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +1.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Sofia -0.1 |
| 12 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.9 (witness), Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +4.2 |
| 12 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.8, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +2.1 (witness) |
| 12 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +0.3, Marta +1.0, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; 3.5 of the sender's anxiety bound into Ana–Marta |
| 12 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 12 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +1.7, Marta +0.4, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.0 (witness) |
| 12 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 13; anxiety Marta +1.6 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.9; calmer sender takes some anxiety: Nadia +0.7, Ravi -0.7; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 12 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 12 | Ravi | PURSUE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.5; anxiety Marta +2.0 (witness), Nadia +0.6 (witness) |
| 12 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 14; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; 2.2 of the sender's anxiety bound into Ravi–Sofia |
| 12 | Pia | SYMPTOM_ONSET | Marta, Ravi | Nadia | arrives week 13; symptom onset, social channel; anxiety Marta +3.1, Nadia +0.7 (witness), Ravi +3.5 |
| 13 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 13 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 13 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 14; anxiety Marta +0.6, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.9; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; Ravi left outside: anxiety Marta -2.3, Ravi +3.5 (of which 1.2 the outsider's own) |
| 13 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.7 (witness), Ravi +1.9 (witness) |
| 13 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 14; anxiety Marta +6.9, Nadia +0.7 (witness), Pia +0.5, Ravi +2.4 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8 |
| 13 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 14; anxiety Marta +3.0, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.7 |
| 13 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 14 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 14 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 14 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 14 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 15; anxiety Ana +0.3, Marta +0.5, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Marta takes 2.0 of functioning from the other |
| 14 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 14 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 15; anxiety Marta +0.6, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +7.6; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3 |
| 14 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.8 (witness) |
| 14 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 15; anxiety Marta +1.9 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.1; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; 6.7 of the sender's anxiety bound into Pia–Ravi |
| 14 | Ravi | WITHHOLD | Sofia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 14 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 16; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.8 (witness), Ravi +4.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 15 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 15 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 15 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 16; anxiety Marta +0.6, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +4.0; calmer sender takes some anxiety: Marta +0.6, Ravi -0.6 |
| 15 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.7 (witness) |
| 15 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 16; anxiety Marta +1.8 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +6.3; calmer sender takes some anxiety: Pia +1.5, Ravi -1.5; Pia takes 2.0 of functioning from the other |
| 15 | Ravi | PURSUE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +1.8 (witness), Nadia +0.7 (witness) |
| 15 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 17; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.4, Sofia +1.4; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 16 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +4.0 |
| 16 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.4, Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +2.0 (witness) |
| 16 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 16 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 16 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 17; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6 |
| 16 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 17; anxiety Marta +0.6, Nadia +1.5 (witness), Pia +1.6 (witness), Ravi +3.2; calmer sender takes some anxiety: Marta +0.7, Ravi -0.7 |
| 16 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 17; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia +0.7, Ravi -0.7 |
| 16 | Pia | TRIANGLE | Marta | Nadia, Ravi | arrives week 17; anxiety Marta +1.0, Nadia +0.7 (witness), Pia +0.5, Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9; Marta left outside: anxiety Marta +2.0, Pia -1.3 (of which 0.7 the outsider's own) |
| 16 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 16 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 18; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 17 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 17 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 17 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 17 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 18; anxiety Ana +0.3, Marta +0.3, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 17 | Bruno | CUTOFF | Ana | — | arrives week 18; anxiety Ana +0.3, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 17 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 18; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +1.2 (witness) |
| 17 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.7 (witness), Ravi +2.1 (witness) |
| 17 | Pia | REDUCE_CUTOFF | Ravi | Marta, Nadia | arrives week 18; anxiety Marta +1.7 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +2.2; calmer sender takes some anxiety: Pia +1.4, Ravi -1.4; Pia's rejection -0.25; bound anxiety released to the third party: Marta +9.6 |
| 17 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 18; anxiety Marta +1.4 (witness), Nadia +1.7, Pia +0.6 (witness), Ravi +0.7; 13.0 of the sender's anxiety bound into Nadia–Ravi |
| 17 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 19; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 18 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 18 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 18 | Marta | PURSUE | Pia | Nadia, Ravi | arrives week 19; anxiety Marta +0.6, Nadia +0.7 (witness), Pia +1.8, Ravi +2.2 (witness) |
| 18 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 19; anxiety Nadia +0.4, Pia +0.6 (witness), Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -0.2, Nadia +0.2; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 18 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 19; anxiety Marta +0.8, Nadia +0.7 (witness), Pia +0.5, Ravi +2.2 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; 6.6 of the sender's anxiety bound into Marta–Pia |
| 18 | Ravi | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 18 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 19 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 20; anxiety Ana +0.3, Marta +2.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 19 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 19 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.8; anxiety Pia +0.7 (witness), Ravi +2.1 (witness) |
| 19 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 19 | Pia | REDUCE_CUTOFF | Ravi | Marta, Nadia | arrives week 20; anxiety Marta +1.7 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.2; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1 |
| 19 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.0; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 19 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 21; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Ravi takes 2.0 of functioning from the other |
| 19 | Bruno | SYMPTOM_ONSET | Ana | — | arrives week 20; symptom onset, social channel; anxiety Ana +2.0 |
| 19 | — | (system) | — | — | Pia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 20 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +4.1 |
| 20 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.4, Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +2.1 (witness) |
| 20 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 20 | Ana | CUTOFF | Bruno | — | arrives week 21; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 20 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 20 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 21; anxiety Marta +0.6, Nadia +0.6 (witness), Pia +1.0, Ravi +2.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -1.3 |
| 20 | Nadia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 20 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 21; anxiety Marta +0.4, Nadia +0.6 (witness), Pia +0.5, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; Marta takes 2.0 of functioning from the other |
| 20 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 21; anxiety Marta +1.8 (witness), Nadia +0.7 (witness), Pia +3.3, Ravi +0.7 |
| 20 | Sofia | REDUCE_CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 22; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia's rejection -0.25 |
| 21 | Ana | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 21 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 21 | Marta | PREVENT_ALIGNMENT | Ravi | Nadia, Pia | arrives week 22; anxiety Marta +0.5, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +1.3; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2 |
| 21 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Pia +0.8 (witness), Ravi +2.1 (witness) |
| 21 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.4; anxiety Nadia +0.6 (witness), Ravi +1.8 (witness) |
| 21 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 23; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.4; Ravi rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -5.4 |
| 21 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 23; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +3.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia takes 2.0 of functioning from the other |
| 22 | Ana | I-POSITION | Marta | Nadia, Pia, Ravi | arrives week 23; anxiety Ana +0.3, Marta +3.3, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 22 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 22 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 23; anxiety Marta +0.5, Nadia +0.6 (witness), Pia +3.0, Ravi +1.7 (witness); 7.9 of the sender's anxiety bound into Marta–Pia |
| 22 | Nadia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 22 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.4; anxiety Nadia +0.6 (witness), Ravi +1.7 (witness) |
| 22 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 23; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 22 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.8 (witness) |
| 23 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 24; anxiety Ana +0.3, Marta +0.5, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 23 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 23 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 24; anxiety Marta +0.5, Nadia +0.6 (witness), Pia +1.8, Ravi +1.8 (witness); 6.8 of the sender's anxiety bound into Marta–Pia |
| 23 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Pia +0.8 (witness), Ravi +2.2 (witness) |
| 23 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 24; anxiety Marta +1.8 (witness), Nadia +0.8 (witness), Pia +0.5, Ravi +5.4; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia takes 2.0 of functioning from the other |
| 23 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | arrives week 24; anxiety Marta -0.9, Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +0.7; Marta holds a position: no functioning changes hands |
| 23 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.7 (witness) |
| 23 | — | (system) | — | — | Nadia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 24 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +4.0 |
| 24 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.2, Nadia +1.3 (witness), Pia +1.4 (witness), Ravi +2.0 (witness) |
| 24 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 24 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 24 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 25; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Dr Halim +0.4, Marta -0.4; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 24 | Marta | PURSUE | Pia | Nadia, Ravi | arrives week 25; anxiety Marta +0.5, Nadia +0.6 (witness), Pia +1.6, Ravi +1.9 (witness); calmer sender takes some anxiety: Marta +0.1, Pia -0.1 |
| 24 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Pia +0.7 (witness), Ravi +2.2 (witness) |
| 24 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.3; anxiety Nadia +0.6 (witness), Ravi +1.9 (witness) |
| 24 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 25; anxiety Marta +4.6, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +0.7; Marta holds a position: no functioning changes hands |
| 24 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.8 (witness) |
| 25 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 25 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 25 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 26; anxiety Ana +0.3, Marta +2.0, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 25 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 25 | Marta | WITHHOLD | Dr Halim | — | held back DISPLACE: computed, not emitted; attention on the tie rises |
| 25 | Nadia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 25 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.3; anxiety Nadia +0.6 (witness), Ravi +1.9 (witness) |
| 25 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 26; anxiety Marta +1.7 (witness), Nadia +0.7 (witness), Pia +1.7, Ravi +0.7; 11.0 of the sender's anxiety bound into Pia–Ravi |
| 25 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 27; anxiety Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 26 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 27; anxiety Ana +0.3, Marta +2.0, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 26 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 26 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 27; anxiety Marta +0.5, Nadia +2.8, Pia +0.8 (witness), Ravi +2.3 (witness); calmer sender takes some anxiety: Marta +0.4, Nadia -0.4; Marta takes 2.0 of functioning from the other |
| 26 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Pia +0.8 (witness), Ravi +2.3 (witness) |
| 26 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 26 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 27; anxiety Marta +1.0 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 26 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 28; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.30; the tie stays open) |
| 27 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 28; anxiety Ana +0.3, Marta +1.9, Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 27 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 27 | Marta | WITHHOLD | Nadia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 27 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 28; anxiety Marta +1.5 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +5.5 |
| 27 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 28; anxiety Marta +1.6 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +1.7; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 27 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 29; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.4; Ravi rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -0.3 |
| 27 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 28 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.4 (witness), Pia +1.4 (witness), Ravi +4.2 |
| 28 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.4 (witness), Pia +1.4 (witness), Ravi +2.1 (witness) |
| 28 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 28 | Ana | CUTOFF | Bruno | — | arrives week 29; anxiety Ana +0.4, Bruno +0.4; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 28 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 28 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 29; anxiety Marta +0.5, Nadia +1.6 (witness), Pia +1.5 (witness), Ravi +6.1; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; Marta takes 2.0 of functioning from the other |
| 28 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Pia +0.8 (witness), Ravi +2.4 (witness) |
| 28 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 29; anxiety Marta +1.7 (witness), Nadia +0.8 (witness), Pia +0.5, Ravi +6.1; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Pia takes 2.0 of functioning from the other |
| 28 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 29; anxiety Marta +1.4 (witness), Nadia -0.9, Pia +0.6 (witness), Ravi +0.7; Nadia left outside: anxiety Nadia +4.8, Ravi -3.2 (of which 1.6 the outsider's own) |
| 28 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 29 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 29 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 29 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 29 | Bruno | CUTOFF | Ana | — | arrives week 30; anxiety Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.35; the tie stays open); impingement removed at once, anxiety Bruno -0.1 |
| 29 | Marta | STAY-IN-CONTACT | Ravi | Nadia, Pia | arrives week 30; anxiety Marta +0.5, Nadia +1.5 (witness), Pia +1.4 (witness), Ravi +1.3; calmer sender takes some anxiety: Marta +0.5, Ravi -0.5 |
| 29 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Pia +0.8 (witness), Ravi +2.9 (witness) |
| 29 | Pia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 29 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 30; anxiety Marta +0.2, Nadia +1.5 (witness), Pia +1.4 (witness), Ravi +0.8; Marta left outside: anxiety Marta +4.4, Ravi -2.9 (of which 1.5 the outsider's own) |
| 29 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 30 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 30 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 30 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 30 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 30 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 31; anxiety Ana +0.3, Marta +3.1, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 30 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 31; anxiety Marta +0.1, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +1.4 (witness) |
| 30 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 31; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +1.3; Ravi takes 2.0 of functioning from the other |
| 30 | Pia | STAY-IN-CONTACT | Ravi | Marta, Nadia | arrives week 31; anxiety Marta +1.6 (witness), Nadia +0.8 (witness), Pia +0.4, Ravi +0.3; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9 |
| 30 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 31; anxiety Marta +1.6 (witness), Nadia +0.8 (witness), Pia +2.7, Ravi +0.8; Pia holds a position: no functioning changes hands |
| 30 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 31 | Ana | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 31 | Bruno | CUTOFF | Ana | — | arrives week 32; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 31 | Marta | PROVOKE | Ravi | Nadia, Pia | arrives week 32; anxiety Marta +0.5, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +5.2; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1 |
| 31 | Nadia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 31 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 32; anxiety Marta +1.4 (witness), Nadia +0.8 (witness), Pia +0.4, Ravi +0.7; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -2.3 |
| 31 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 32; anxiety Marta +1.3 (witness), Nadia +2.2, Pia +0.6 (witness), Ravi +0.7; 12.9 of the sender's anxiety bound into Nadia–Ravi |
| 31 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 33; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia rejects the other on Ravi–Sofia (rejection +0.25; the tie stays open) |
| 31 | Nadia | SYMPTOM_ONSET | Marta, Ravi | Pia | arrives week 32; symptom onset, mental channel; anxiety Marta +2.4, Pia +0.8 (witness), Ravi +3.6 |
| 32 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.4 (witness), Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +4.3 |
| 32 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.9, Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +2.1 (witness) |
| 32 | Ana | CUTOFF | Bruno | — | arrives week 33; anxiety Ana +0.4, Bruno +0.3; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 32 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 32 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 33; anxiety Dr Halim +0.1, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.50; the tie stays open) |
| 32 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 33; anxiety Marta +0.5, Nadia +1.6, Pia +0.7 (witness), Ravi +2.4 (witness); calmer sender takes some anxiety: Marta +0.1, Nadia -0.1; 8.5 of the sender's anxiety bound into Marta–Nadia |
| 32 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 33; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +2.9 |
| 32 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 33; anxiety Marta +1.0, Nadia +0.7 (witness), Pia +0.4, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7 |
| 32 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 33; anxiety Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 32 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 34; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 33 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 33 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 33 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 33 | Ana | CUTOFF | Bruno | — | arrives week 34; anxiety Ana +0.4, Bruno +0.8; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 33 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 33 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +0.7; anxiety Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +1.3 (witness) |
| 33 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.5; anxiety Pia +0.7 (witness), Ravi +2.2 (witness) |
| 33 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 34; anxiety Marta +1.4 (witness), Nadia +0.7 (witness), Pia +0.4, Ravi +1.5; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; 6.8 of the sender's anxiety bound into Pia–Ravi |
| 33 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 34; anxiety Dr Halim +0.2, Marta +0.9 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 33 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.7 (witness), Pia +0.6 (witness) |
| 34 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 35; anxiety Ana +0.3, Marta +0.4, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; 2.6 of the sender's anxiety bound into Ana–Marta |
| 34 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 34 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 35; anxiety Marta +0.3, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.1 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -0.9 |
| 34 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 34 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 35; anxiety Marta +3.3, Nadia +0.8 (witness), Pia +0.4, Ravi +2.3 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6 |
| 34 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 35; anxiety Marta +1.3 (witness), Nadia +0.8, Pia +0.6 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia -0.1, Ravi +0.1; 9.4 of the sender's anxiety bound into Nadia–Ravi |
| 34 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 35 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +1.0 (witness) |
| 35 | Bruno | CUTOFF | Ana | — | arrives week 36; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open); impingement removed at once, anxiety Bruno -0.1 |
| 35 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | arrives week 36; anxiety Marta +0.5, Nadia +0.8 (witness), Pia +2.3, Ravi +2.4 (witness); Marta takes 2.0 of functioning from the other |
| 35 | Nadia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 35 | Pia | PREVENT_ALIGNMENT | Ravi | Marta, Nadia | arrives week 36; anxiety Marta +1.3 (witness), Nadia +0.7 (witness), Pia +0.4, Ravi -0.6; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6 |
| 35 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +0.7 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 35 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 36 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.4 (witness), Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +4.3 |
| 36 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.9, Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +2.1 (witness) |
| 36 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 36 | Bruno | CUTOFF | Ana | — | arrives week 37; anxiety Ana +0.5, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 36 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 37; anxiety Marta +0.5, Nadia +4.4, Pia +0.7 (witness), Ravi +2.2 (witness); calmer sender takes some anxiety: Marta +1.0, Nadia -1.0; Marta takes 2.0 of functioning from the other |
| 36 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.5; anxiety Pia +0.7 (witness), Ravi +2.2 (witness) |
| 36 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 37; anxiety Marta +0.5, Nadia +0.8 (witness), Pia +0.5, Ravi +2.2 (witness); calmer sender takes some anxiety: Marta -0.0, Pia +0.0; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -2.4 |
| 36 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 37; anxiety Marta +1.2 (witness), Nadia +0.3, Pia +0.6 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; 13.0 of the sender's anxiety bound into Nadia–Ravi |
| 36 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 37 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 37 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 37 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +0.7; anxiety Nadia +0.7 (witness), Pia +0.7 (witness), Ravi +1.3 (witness) |
| 37 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.6; anxiety Pia +0.7 (witness), Ravi +2.0 (witness) |
| 37 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 38; anxiety Marta +0.4, Nadia +0.8 (witness), Pia +0.5, Ravi +2.0 (witness); calmer sender takes some anxiety: Marta -0.3, Pia +0.3; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 37 | Ravi | WITHHOLD | Nadia | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 37 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 38 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 38 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 38 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 38 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 38 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.2; anxiety Nadia +0.7 (witness), Ravi +1.9 (witness) |
| 38 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.6; anxiety Pia +0.7 (witness), Ravi +2.1 (witness) |
| 38 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 38 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 39; anxiety Marta +1.2 (witness), Nadia +0.2, Pia +0.6 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia -0.7, Ravi +0.7; 9.4 of the sender's anxiety bound into Nadia–Ravi |
| 38 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 38 | — | (system) | — | — | Nadia–Ravi tie now distant |
| 39 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +1.1 (witness) |
| 39 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 39 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +0.7; anxiety Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.3 (witness) |
| 39 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 40; anxiety Nadia +0.5, Pia +0.7 (witness), Ravi +2.0 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Nadia -1.2 |
| 39 | Pia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 39 | Ravi | WITHHOLD | Dr Halim | — | held back DISPLACE: computed, not emitted; attention on the tie rises |
| 39 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 40 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +4.1 |
| 40 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.7, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +2.1 (witness) |
| 40 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 40 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 40 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 41; anxiety Dr Halim +0.1, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Dr Halim +0.4, Marta -0.4; Marta left outside: anxiety Dr Halim -0.3, Marta +0.4 (of which 0.1 the outsider's own) |
| 40 | Marta | PREVENT_ALIGNMENT | Ravi | Nadia, Pia | arrives week 41; anxiety Marta +0.4, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +0.0 |
| 40 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.6; anxiety Pia +0.6 (witness), Ravi +1.9 (witness) |
| 40 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 41; anxiety Marta +4.3, Nadia +0.8 (witness), Pia +0.5, Ravi +2.2 (witness); calmer sender takes some anxiety: Marta -0.3, Pia +0.3 |
| 40 | Ravi | PREVENT_ALIGNMENT | Nadia | Marta, Pia | arrives week 41; anxiety Marta +1.1 (witness), Nadia +0.1, Pia +0.6 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia -1.2, Ravi +1.2 |
| 40 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 41 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 41 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.0 (witness) |
| 41 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 41 | Marta | STAY-IN-CONTACT | Ravi | Nadia, Pia | arrives week 42; anxiety Marta +0.4, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi -0.1 |
| 41 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 42; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +2.4 |
| 41 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 42; anxiety Marta +1.2 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +1.5; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; 10.4 of the sender's anxiety bound into Pia–Ravi |
| 41 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 42; anxiety Marta +3.5, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1 |
| 41 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 42 | Ana | CUTOFF | Bruno | — | arrives week 43; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 42 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 42 | Marta | STAY-IN-CONTACT | Ana | Nadia, Pia, Ravi | arrives week 43; anxiety Ana -1.8, Marta +0.3, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +1.0 (witness) |
| 42 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.6; anxiety Pia +0.6 (witness), Ravi +1.8 (witness) |
| 42 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 42 | Ravi | CONFLICT | Nadia | Marta, Pia | arrives week 43; anxiety Marta +1.4 (witness), Nadia +5.2, Pia +0.7 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia -0.8, Ravi +0.8 |
| 42 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 42 | — | (system) | — | — | Pia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 43 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.9 (witness) |
| 43 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 43 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 44; anxiety Marta +0.3, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 43 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 43 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 44; anxiety Marta +1.2 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +1.4; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; 7.4 of the sender's anxiety bound into Pia–Ravi |
| 43 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 44; anxiety Marta +0.9 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 43 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.7 (witness), Pia +0.6 (witness) |
| 44 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.4 (witness), Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +4.0 |
| 44 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.7, Nadia +1.6 (witness), Pia +1.4 (witness), Ravi +2.0 (witness) |
| 44 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 44 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.9 (witness) |
| 44 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 44 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 45; anxiety Marta +0.5, Nadia +4.0, Pia +0.7 (witness), Ravi +2.0 (witness); calmer sender takes some anxiety: Marta +0.8, Nadia -0.8; Nadia holds a position: no functioning changes hands |
| 44 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 45; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.4; Ravi takes 2.0 of functioning from the other |
| 44 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 44 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.8; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 44 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 45 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 45 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 45 | Marta | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 45 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 46; anxiety Marta +1.4 (witness), Nadia +0.6, Pia +0.7 (witness), Ravi +0.3; Ravi takes 2.0 of functioning from the other |
| 45 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 46; anxiety Marta +1.4, Nadia +0.8 (witness), Pia +0.5, Ravi +1.8 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; 5.7 of the sender's anxiety bound into Marta–Pia |
| 45 | Ravi | CONFLICT | Nadia | Marta, Pia | arrives week 46; anxiety Marta +1.4 (witness), Nadia +6.1, Pia +0.7 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.6, Ravi +0.6 |
| 45 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 46 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 46 | Bruno | CUTOFF | Ana | — | arrives week 47; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open) |
| 46 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 47; anxiety Marta +0.5, Nadia +1.7 (witness), Pia +1.3 (witness), Ravi +1.0; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1; 6.9 of the sender's anxiety bound into Marta–Ravi |
| 46 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.8 (witness), Ravi +1.9 (witness) |
| 46 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 47; anxiety Marta -0.4, Nadia +0.8 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; Marta takes 2.0 of functioning from the other |
| 46 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 47; anxiety Marta +0.8, Nadia +1.7 (witness), Pia +1.3 (witness), Ravi +0.6; Ravi rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -0.3 |
| 46 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 48; anxiety Marta +0.6 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 47 | Ana | CUTOFF | Bruno | — | arrives week 48; anxiety Ana +0.4, Bruno +0.1; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 47 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 47 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 48; anxiety Marta +0.4, Nadia +3.4, Pia +0.8 (witness), Ravi +1.8 (witness); calmer sender takes some anxiety: Marta +1.5, Nadia -1.5 |
| 47 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.8 (witness), Ravi +1.8 (witness) |
| 47 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 48; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia +0.5, Ravi +0.8; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; 6.2 of the sender's anxiety bound into Pia–Ravi |
| 47 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.7; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 47 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 47 | — | (system) | — | — | Pia–Ravi tie now distant |
| 48 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +1.8 (witness), Pia +1.4 (witness), Ravi +3.4 |
| 48 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.6, Nadia +1.8 (witness), Pia +1.4 (witness), Ravi +1.7 (witness) |
| 48 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 48 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 48 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 49; anxiety Ana +0.3, Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 48 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 48 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 49; anxiety Dr Halim +0.1, Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Marta -0.5 |
| 48 | Marta | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 48 | Nadia | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 48 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 49; anxiety Marta +0.2, Nadia +0.8 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 48 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.7; anxiety Marta +0.6 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 48 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 49 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 49 | Bruno | WITHHOLD | Ana | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 49 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Marta +2.2; anxiety Nadia +1.8 (witness), Pia +1.4 (witness) |
| 49 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.8 (witness), Ravi +1.9 (witness) |
| 49 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 49 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 50; anxiety Marta +3.3, Nadia +1.8 (witness), Pia +1.4 (witness), Ravi +0.6 |
| 49 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 49 | — | (system) | — | — | Dr Halim–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 50 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 51; anxiety Ana +0.3, Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +1.0, Marta -1.0; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 50 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 50 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.2; anxiety Nadia +0.9 (witness), Ravi +1.7 (witness) |
| 50 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 50 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 51; anxiety Marta +1.2 (witness), Nadia +0.8 (witness), Pia +0.5, Ravi +2.4; calmer sender takes some anxiety: Pia +1.6, Ravi -1.6 |
| 50 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 51; anxiety Marta +0.9 (witness), Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 50 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 51 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 51 | Ana | CUTOFF | Bruno | — | arrives week 52; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 51 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 51 | Marta | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 51 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 51 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 52; anxiety Marta +1.2 (witness), Nadia +0.8 (witness), Pia +0.5, Ravi +0.3; calmer sender takes some anxiety: Pia +1.3, Ravi -1.3; 6.6 of the sender's anxiety bound into Pia–Ravi |
| 51 | Ravi | CONFLICT | Nadia | Marta, Pia | arrives week 52; anxiety Marta +1.5 (witness), Nadia +6.5, Pia +0.8 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia -0.1, Ravi +0.1 |
| 51 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 51 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |
| 52 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +1.8 (witness), Pia +1.4 (witness), Ravi +3.3 |
| 52 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.7, Nadia +1.7 (witness), Pia +1.4 (witness), Ravi +1.6 (witness) |
| 52 | Ana | REDUCE_CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 53; anxiety Ana +0.3, Marta +0.3, Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9; Ana's rejection -0.25; Marta's rejection -0.25 |
| 52 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 52 | Marta | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 52 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.7 (witness), Ravi +1.7 (witness) |
| 52 | Pia | TRIANGLE | Marta | Nadia, Ravi | arrives week 53; anxiety Nadia +0.8 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -1.3, Pia +1.3; Marta left outside: anxiety Marta +1.5, Pia -1.0 (of which 0.5 the outsider's own) |
| 52 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 53; anxiety Marta +1.2 (witness), Nadia +0.8 (witness), Pia +1.9, Ravi +0.5; 11.3 of the sender's anxiety bound into Pia–Ravi |
| 52 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 53 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 53 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 53 | Ana | WITHHOLD | Bruno | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 53 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 53 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 54; anxiety Marta +0.4, Nadia +1.8 (witness), Pia +1.4 (witness), Ravi +0.5 |
| 53 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 54; anxiety Marta +1.4 (witness), Nadia +0.6, Pia +0.7 (witness), Ravi -1.1; Ravi left outside: anxiety Nadia -2.4, Ravi +3.6 (of which 1.2 the outsider's own) |
| 53 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 54; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia +0.5, Ravi +1.3; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; 5.6 of the sender's anxiety bound into Pia–Ravi |
| 53 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 53 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 54 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 54 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 54 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 55; anxiety Ana +0.3, Marta +2.2, Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 54 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 54 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 55; anxiety Marta +0.4, Nadia +3.2, Pia +0.7 (witness), Ravi +1.6 (witness); calmer sender takes some anxiety: Marta +0.6, Nadia -0.6 |
| 54 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.7 (witness), Ravi +1.6 (witness) |
| 54 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 55; anxiety Marta -0.4, Nadia +0.8 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -1.1, Pia +1.1; Marta takes 2.0 of functioning from the other |
| 54 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 55; anxiety Marta +0.9 (witness), Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 54 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 55 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 55 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 56; anxiety Ana +0.3, Marta +0.9, Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 55 | Bruno | CUTOFF | Ana | — | arrives week 56; anxiety Ana +0.3, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.35; the tie stays open) |
| 55 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +1.3; anxiety Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 55 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 55 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 56; anxiety Marta +0.6, Nadia +0.8 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9; 6.7 of the sender's anxiety bound into Marta–Pia |
| 55 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 56; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +1.1, Ravi +0.5; 9.8 of the sender's anxiety bound into Pia–Ravi |
| 55 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 56 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +1.8 (witness), Pia +1.5 (witness), Ravi +3.2 |
| 56 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.6, Nadia +1.7 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 56 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 56 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 57; anxiety Ana +0.3, Marta +2.4, Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 56 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 56 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 57; anxiety Dr Halim +0.1, Nadia +0.8 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Marta left outside: anxiety Dr Halim -0.0, Marta +0.1 (of which 0.0 the outsider's own) |
| 56 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 57; anxiety Marta +0.4, Nadia +1.8 (witness), Pia +1.5 (witness), Ravi +0.7 |
| 56 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.8 (witness), Ravi +1.7 (witness) |
| 56 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 57; anxiety Marta +3.3, Nadia +0.9 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0 |
| 56 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 57; anxiety Dr Halim +0.1, Marta +0.8 (witness), Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 56 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 58; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.30; the tie stays open) |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 57 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 57 | Bruno | CUTOFF | Ana | — | arrives week 58; anxiety Ana +0.7, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 57 | Marta | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 57 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 58; anxiety Marta +1.3 (witness), Nadia +0.6, Pia +0.8 (witness), Ravi +1.9 |
| 57 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 58; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 5.7 of the sender's anxiety bound into Pia–Ravi |
| 57 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 59; anxiety Marta +0.6 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.15; the tie stays open) |
| 57 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 58 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 58 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 58 | Bruno | WITHHOLD | Ana | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 58 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.1; anxiety Nadia +0.8 (witness), Ravi +1.5 (witness) |
| 58 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.8 (witness), Ravi +1.8 (witness) |
| 58 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 59; anxiety Marta -0.1, Nadia +0.8 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -1.1, Pia +1.1; Marta takes 2.0 of functioning from the other |
| 58 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 59; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +1.0, Ravi +0.5; 7.4 of the sender's anxiety bound into Pia–Ravi |
| 58 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 59 | Ana | CUTOFF | Bruno | — | arrives week 60; anxiety Ana +0.4, Bruno +0.5; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -0.0 |
| 59 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.6 |
| 59 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.1; anxiety Nadia +0.9 (witness), Ravi +1.7 (witness) |
| 59 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.9 (witness), Ravi +1.8 (witness) |
| 59 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 60; anxiety Marta +1.0 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; 5.5 of the sender's anxiety bound into Pia–Ravi |
| 59 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +0.6 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 59 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 59 | — | (system) | — | — | Marta–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 60 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.2 (witness), Nadia +1.8 (witness), Pia +1.6 (witness), Ravi +3.2 |
| 60 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.5, Nadia +1.7 (witness), Pia +1.6 (witness), Ravi +1.6 (witness) |
| 60 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 60 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 61; anxiety Ana +0.3, Marta +2.3, Nadia +1.0 (witness), Pia +0.9 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.4, Marta -0.4 |
| 60 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 60 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 61; anxiety Marta +0.1, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.9 (witness) |
| 60 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.8 (witness), Ravi +1.7 (witness) |
| 60 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 61; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 60 | Ravi | CONFLICT | Nadia | Marta, Pia | arrives week 61; anxiety Marta +1.4 (witness), Nadia +5.8, Pia +0.9 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia -1.0, Ravi +1.0 |
| 60 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.9 (witness), Pia +0.8 (witness) |
| 61 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 61 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 61 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 61 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 61 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.1; anxiety Nadia +1.0 (witness), Ravi +1.8 (witness) |
| 61 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.9; anxiety Pia +0.8 (witness), Ravi +1.6 (witness) |
| 61 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 62; anxiety Marta +3.7, Nadia +1.0 (witness), Pia +0.5, Ravi +1.8 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0 |
| 61 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 62; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia +1.0, Ravi +0.5; 8.1 of the sender's anxiety bound into Pia–Ravi |
| 61 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 62 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 63; anxiety Ana +0.3, Marta +0.3, Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Marta takes 2.0 of functioning from the other |
| 62 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 62 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 63; anxiety Marta +0.3, Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -1.9 |
| 62 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 63; anxiety Marta +1.3 (witness), Nadia +0.6, Pia +0.8 (witness), Ravi +0.5; Ravi holds a position: no functioning changes hands |
| 62 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 63; anxiety Marta +1.3 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +4.1; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3 |
| 62 | Ravi | STAY-IN-CONTACT | Nadia | Marta, Pia | arrives week 63; anxiety Marta +1.3 (witness), Nadia +0.7, Pia +0.8 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia -1.3, Ravi +1.3 |
| 62 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.9 (witness), Pia +0.8 (witness) |
| 63 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 63 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 63 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 64; anxiety Marta +0.4, Nadia +3.0, Pia +0.8 (witness), Ravi +1.6 (witness); calmer sender takes some anxiety: Marta +0.7, Nadia -0.7 |
| 63 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 64; anxiety Marta +1.3 (witness), Nadia +0.6, Pia +0.8 (witness), Ravi +3.0 |
| 63 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 64; anxiety Marta +2.9, Nadia +1.0 (witness), Pia +0.5, Ravi +1.9 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7 |
| 63 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 64; anxiety Marta +1.2 (witness), Nadia +0.8 (witness), Pia +0.9, Ravi +0.5; 9.4 of the sender's anxiety bound into Pia–Ravi |
| 63 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 64 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.2 (witness), Nadia +1.8 (witness), Pia +1.6 (witness), Ravi +3.2 |
| 64 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.4, Nadia +1.8 (witness), Pia +1.6 (witness), Ravi +1.6 (witness) |
| 64 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 64 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 64 | Bruno | CUTOFF | Ana | — | arrives week 65; anxiety Ana +0.2, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 64 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 65; anxiety Dr Halim +0.1, Marta +0.7 (witness), Nadia +0.8 (witness), Pia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Ravi -0.5 |
| 64 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 65; anxiety Marta +0.4, Nadia +1.8 (witness), Pia +1.6 (witness), Ravi +1.4 |
| 64 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 65; anxiety Marta +1.3 (witness), Nadia +0.6, Pia +0.8 (witness), Ravi +3.0 |
| 64 | Pia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 64 | Ravi | WITHHOLD | Dr Halim | — | held back SPLIT: computed, not emitted; attention on the tie rises |
| 64 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.5 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 65 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 65 | Bruno | CUTOFF | Ana | — | arrives week 66; anxiety Ana +0.6, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 65 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 66; anxiety Marta +0.1, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.9 (witness) |
| 65 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 66; anxiety Marta +1.2 (witness), Nadia +0.6, Pia +0.8 (witness), Ravi +0.6; Ravi holds a position: no functioning changes hands |
| 65 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 66; anxiety Marta +2.5, Nadia +1.1 (witness), Pia +0.5, Ravi +1.9 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6 |
| 65 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | arrives week 66; anxiety Marta +1.1 (witness), Nadia +0.8 (witness), Pia -0.4, Ravi +0.5; Pia takes 2.0 of functioning from the other |
| 65 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 66 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +0.9 (witness) |
| 66 | Bruno | CUTOFF | Ana | — | arrives week 67; anxiety Ana +0.6, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 66 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 67; anxiety Ana +1.7, Marta +0.2, Nadia +0.9 (witness), Pia +0.8 (witness), Ravi +0.9 (witness) |
| 66 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 67; anxiety Marta +1.2 (witness), Nadia +0.6, Pia +0.8 (witness), Ravi +3.2 |
| 66 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 67; anxiety Marta +1.8, Nadia +1.0 (witness), Pia +0.5, Ravi +1.8 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; 7.2 of the sender's anxiety bound into Marta–Pia |
| 66 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 67; anxiety Marta +0.8 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 66 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.9 (witness), Pia +0.8 (witness) |
| 67 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 67 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.6 |
| 67 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 68; anxiety Marta +0.4, Nadia +1.7 (witness), Pia +1.5 (witness), Ravi +0.9; Ravi left outside: anxiety Marta -2.0, Ravi +3.0 (of which 1.0 the outsider's own) |
| 67 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 68; anxiety Marta +1.2 (witness), Nadia +0.6, Pia +0.7 (witness), Ravi +0.6; Ravi left outside: anxiety Nadia -2.0, Ravi +2.9 (of which 1.0 the outsider's own) |
| 67 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 68; anxiety Marta +1.1, Nadia +1.0 (witness), Pia +0.5, Ravi +1.8 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9 |
| 67 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 68; anxiety Marta +1.3 (witness), Nadia +0.9 (witness), Pia +3.1, Ravi +0.5 |
| 67 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 69; anxiety Marta +0.6 (witness), Nadia +0.9 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia rejects the other on Ravi–Sofia (rejection +0.45; the tie stays open) |
| 68 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.2 (witness), Nadia +1.8 (witness), Pia +1.5 (witness), Ravi +3.3 |
| 68 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.5, Nadia +1.8 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 68 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 68 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 68 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 68 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 68 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 68 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 68 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 68 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.6 |
| 68 | Marta | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 68 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 69; anxiety Marta +1.1 (witness), Nadia +0.6, Pia +0.7 (witness), Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 68 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 69; anxiety Marta +1.2, Nadia +1.0 (witness), Pia +0.5, Ravi +1.9 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6 |
| 68 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 69; anxiety Dr Halim +0.0, Marta +0.8 (witness), Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 68 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 69 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 69 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.6 |
| 69 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 70; anxiety Marta +0.4, Nadia +1.9 (witness), Pia +1.5 (witness), Ravi +3.1; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1 |
| 69 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 70; anxiety Marta +1.2 (witness), Nadia +0.6, Pia +0.7 (witness), Ravi +2.9 |
| 69 | Pia | TRIANGLE | Marta | Nadia, Ravi | arrives week 70; anxiety Marta +0.2, Nadia +1.0 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Marta left outside: anxiety Marta +2.1, Pia -1.4 (of which 0.7 the outsider's own) |
| 69 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.5; anxiety Marta +0.6 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 69 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 70 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 70 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 71; anxiety Ana +0.3, Marta +0.1, Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 70 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.6 |
| 70 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 71; anxiety Marta +0.4, Nadia +1.9 (witness), Pia +1.5 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2; Ravi left outside: anxiety Marta -1.5, Ravi +2.3 (of which 0.8 the outsider's own) |
| 70 | Nadia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 70 | Pia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 70 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 71; anxiety Marta +0.4, Nadia +1.9 (witness), Pia +1.5 (witness), Ravi +0.5 |
| 70 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 72; anxiety Marta +0.6 (witness), Nadia +0.9 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 71 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 71 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 71 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 71 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 72; anxiety Marta +0.1, Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.9 (witness) |
| 71 | Nadia | CONFLICT | Marta | Pia, Ravi | arrives week 72; anxiety Marta +2.4, Nadia +0.6, Pia +0.8 (witness), Ravi +1.7 (witness) |
| 71 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 72; anxiety Marta +2.3, Nadia +1.0 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4 |
| 71 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 72; anxiety Marta +2.4, Nadia +1.8 (witness), Pia +1.4 (witness), Ravi +0.5; 10.5 of the sender's anxiety bound into Marta–Ravi |
| 71 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 73; anxiety Marta +0.6 (witness), Nadia +0.8 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 72 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.2 (witness), Nadia +1.9 (witness), Pia +1.5 (witness), Ravi +3.1 |
| 72 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.5, Nadia +1.9 (witness), Pia +1.5 (witness), Ravi +1.5 (witness) |
| 72 | Ana | WITHHOLD | Bruno | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 72 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 72 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 73; anxiety Dr Halim +0.1, Nadia +0.8 (witness), Pia +0.7 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6 |
| 72 | Marta | WITHHOLD | Dr Halim | — | held back DISPLACE: computed, not emitted; attention on the tie rises |
| 72 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 73; anxiety Marta +1.2 (witness), Nadia +0.6, Pia +0.7 (witness), Ravi +3.1 |
| 72 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 72 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.5; anxiety Marta +0.5 (witness), Nadia +0.8 (witness), Pia +0.7 (witness) |
| 72 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 73 | Ana | CUTOFF | Bruno | — | arrives week 74; anxiety Ana +0.4, Bruno +0.3; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -0.1 |
| 73 | Bruno | CUTOFF | Ana | — | arrives week 74; anxiety Ana +0.1, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Bruno -0.3 |
| 73 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 74; anxiety Marta +0.4, Nadia +1.7 (witness), Pia +1.4 (witness), Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 73 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 74; anxiety Marta +1.2 (witness), Nadia +0.6, Pia +0.7 (witness), Ravi +0.7; Ravi takes 2.0 of functioning from the other |
| 73 | Pia | TRIANGLE | Marta | Nadia, Ravi | arrives week 74; anxiety Marta +0.6, Nadia +0.9 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5; Marta left outside: anxiety Marta +2.2, Pia -1.4 (of which 0.7 the outsider's own) |
| 73 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.4; anxiety Marta +0.6 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 73 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.9 (witness), Pia +0.7 (witness) |
| 74 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 74 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 74 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 74 | Ana | WITHHOLD | Bruno | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 74 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 74 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 75; anxiety Marta +0.4, Nadia +0.2, Pia +0.8 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta +0.5, Nadia -0.5; 8.6 of the sender's anxiety bound into Marta–Nadia |
| 74 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 75; anxiety Marta +1.2 (witness), Nadia +0.7, Pia +0.7 (witness), Ravi +0.6; 9.0 of the sender's anxiety bound into Nadia–Ravi |
| 74 | Pia | STAY-IN-CONTACT | Ravi | Marta, Nadia | arrives week 75; anxiety Marta +1.3 (witness), Nadia +1.0 (witness), Pia +0.5, Ravi -0.7; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 74 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 75; anxiety Marta +3.1, Nadia +2.0 (witness), Pia +1.5 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1; Ravi takes 2.0 of functioning from the other |
| 74 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 75 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 75 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 75 | Marta | STAY-IN-CONTACT | Ana | Nadia, Pia, Ravi | arrives week 76; anxiety Ana -1.8, Marta +0.3, Nadia +1.0 (witness), Pia +0.7 (witness), Ravi +0.7 (witness) |
| 75 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 75 | Pia | STAY-IN-CONTACT | Ravi | Marta, Nadia | arrives week 76; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +0.3; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3 |
| 75 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.8 (witness) |
| 75 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 77; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.8 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 75 | — | (system) | — | — | Ana–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 76 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +2.0 (witness), Pia +1.5 (witness), Ravi +2.7 |
| 76 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.7, Nadia +2.0 (witness), Pia +1.5 (witness), Ravi +1.3 (witness) |
| 76 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 77; anxiety Ana +0.3, Marta -1.3, Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 76 | Bruno | WITHHOLD | Ana | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 76 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 77; anxiety Marta +0.4, Nadia -1.1, Pia +0.7 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta +0.4, Nadia -0.4; Nadia left outside: anxiety Marta -1.6, Nadia +2.5 (of which 0.8 the outsider's own) |
| 76 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 77; anxiety Marta +1.2 (witness), Nadia +0.7, Pia +0.7 (witness), Ravi +0.4; Ravi left outside: anxiety Nadia -1.7, Ravi +2.6 (of which 0.9 the outsider's own) |
| 76 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 77; anxiety Marta +2.5, Nadia +1.0 (witness), Pia +0.5, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4 |
| 76 | Ravi | STAY-IN-CONTACT | Pia | Marta, Nadia | arrives week 77; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia -0.8, Ravi +0.4 |
| 76 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 78; anxiety Marta +0.6 (witness), Nadia +1.0 (witness), Pia +0.7 (witness), Ravi +0.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 76 | — | (system) | — | — | Marta–Nadia: rejection has faded; the tie is no longer cut off (ordinary) |
| 76 | — | (system) | — | — | Marta–Pia: rejection has faded; the tie is no longer cut off (ordinary) |
| 77 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 77 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 77 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 77 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 78; anxiety Ana +0.3, Marta +0.1, Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 77 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.6 |
| 77 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 78; anxiety Marta +0.4, Nadia +2.1, Pia +0.7 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Marta +0.4, Nadia -0.4; 6.9 of the sender's anxiety bound into Marta–Nadia |
| 77 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 78; anxiety Marta +1.2 (witness), Nadia +0.7, Pia +0.6 (witness), Ravi +0.4; 7.9 of the sender's anxiety bound into Nadia–Ravi |
| 77 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 78; anxiety Marta +2.6, Nadia +1.0 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4 |
| 77 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.7 (witness) |
| 77 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.7 (witness) |
| 78 | Ana | CUTOFF | Bruno | — | arrives week 79; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 78 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 78 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | arrives week 79; anxiety Marta +0.4, Nadia +1.0 (witness), Pia -0.8, Ravi +1.3 (witness); Pia takes 2.0 of functioning from the other |
| 78 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 79; anxiety Marta +1.1 (witness), Nadia +0.7, Pia +0.6 (witness), Ravi +0.3; Ravi takes 2.0 of functioning from the other |
| 78 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 79; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +0.2; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Ravi left outside: anxiety Pia -1.6, Ravi +2.4 (of which 0.8 the outsider's own) |
| 78 | Ravi | I-POSITION | Pia | Marta, Nadia | arrives week 79; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +4.8, Ravi +0.4; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 78 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 80; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.7 (witness), Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 79 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 79 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 79 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 79 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 79 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 80; anxiety Marta +0.5, Nadia +2.1 (witness), Pia +1.4 (witness), Ravi +1.8; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2 |
| 79 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 80; anxiety Marta +1.2 (witness), Nadia +0.7, Pia +0.6 (witness), Ravi +0.3; Ravi takes 1.8 of functioning from the other |
| 79 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 80; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; 8.6 of the sender's anxiety bound into Pia–Ravi |
| 79 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | arrives week 80; anxiety Marta -0.5, Nadia +2.1 (witness), Pia +1.4 (witness), Ravi +0.4; Marta takes 2.0 of functioning from the other |
| 79 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.2 (witness), Pia +0.7 (witness) |
| 80 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.4 (witness), Nadia +2.1 (witness), Pia +1.4 (witness), Ravi +2.6 |
| 80 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.8, Nadia +2.1 (witness), Pia +1.4 (witness), Ravi +1.3 (witness) |
| 80 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 80 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 80 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 80 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 81; anxiety Dr Halim +0.1, Nadia +1.0 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Marta -0.5; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 80 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 81; anxiety Marta +0.4, Nadia +1.1 (witness), Pia +1.3, Ravi +1.2 (witness); 7.5 of the sender's anxiety bound into Marta–Pia |
| 80 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 81; anxiety Marta +2.6, Nadia +0.8, Pia +0.7 (witness), Ravi +1.3 (witness); Nadia takes 2.0 of functioning from the other |
| 80 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 81; anxiety Marta +1.2 (witness), Nadia +1.0 (witness), Pia +0.5, Ravi +1.3; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 7.0 of the sender's anxiety bound into Pia–Ravi |
| 80 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 81; anxiety Marta +1.2 (witness), Nadia +1.0 (witness), Pia +2.4, Ravi +0.4; 8.0 of the sender's anxiety bound into Pia–Ravi |
| 80 | Sofia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 81 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 81 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 81 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 81 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 81 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 82; anxiety Marta +0.5, Nadia +1.2, Pia +0.7 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Marta +0.8, Nadia -0.8; 5.4 of the sender's anxiety bound into Marta–Nadia |
| 81 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 82; anxiety Marta +1.2 (witness), Nadia +0.7, Pia +0.6 (witness), Ravi +2.3; 7.9 of the sender's anxiety bound into Nadia–Ravi |
| 81 | Pia | TRIANGLE | Marta | Nadia, Ravi | arrives week 82; anxiety Marta +0.4, Nadia +0.9 (witness), Pia +0.5, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5; Marta left outside: anxiety Marta +1.7, Pia -1.1 (of which 0.6 the outsider's own) |
| 81 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 82; anxiety Marta +1.2 (witness), Nadia +0.2, Pia +0.6 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.9, Ravi +0.9; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -0.5 |
| 81 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.7 (witness) |
| 82 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 82 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 82 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.6 |
| 82 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 83; anxiety Marta +0.5, Nadia +0.6, Pia +0.7 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta +0.6, Nadia -0.6 |
| 82 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.5; anxiety Marta +1.2 (witness), Pia +0.6 (witness) |
| 82 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 83; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +2.3; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 5.1 of the sender's anxiety bound into Pia–Ravi |
| 82 | Ravi | PURSUE | Nadia | Marta, Pia | arrives week 83; anxiety Marta +1.2 (witness), Nadia +1.4, Pia +0.6 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.5, Ravi +0.5 |
| 82 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 83 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 83 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 83 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | arrives week 84; anxiety Marta +0.5, Nadia +0.9 (witness), Pia -1.1, Ravi +1.1 (witness); Pia takes 2.0 of functioning from the other |
| 83 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.5; anxiety Marta +1.2 (witness), Pia +0.6 (witness) |
| 83 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 84; anxiety Marta +0.3, Nadia +0.9 (witness), Pia +0.5, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Marta takes 2.0 of functioning from the other |
| 83 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 84; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +2.1, Ravi +0.4; 7.4 of the sender's anxiety bound into Pia–Ravi |
| 83 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 85; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Sofia rejects the other on Ravi–Sofia (rejection +0.25; the tie stays open) |
| 84 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.4 (witness), Nadia +2.1 (witness), Pia +1.4 (witness), Ravi +2.6 |
| 84 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.8, Nadia +2.1 (witness), Pia +1.4 (witness), Ravi +1.3 (witness) |
| 84 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 84 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 84 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 84 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 85; anxiety Marta +0.5, Nadia +2.0 (witness), Pia +1.3 (witness), Ravi +0.1; Ravi left outside: anxiety Marta -1.8, Ravi +2.6 (of which 0.9 the outsider's own) |
| 84 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.5; anxiety Marta +1.2 (witness), Pia +0.6 (witness) |
| 84 | Pia | TRIANGLE | Marta | Nadia, Ravi | arrives week 85; anxiety Marta +0.3, Nadia +0.9 (witness), Pia +0.5, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Marta left outside: anxiety Marta +1.5, Pia -1.0 (of which 0.5 the outsider's own) |
| 84 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 85; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia -1.4, Ravi +0.4; Pia left outside: anxiety Pia +2.0, Ravi -1.3 (of which 0.7 the outsider's own) |
| 84 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.1 (witness), Pia +0.7 (witness) |
| 85 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 85 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 85 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 85 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 85 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 85 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 85 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.6 |
| 85 | Marta | PURSUE | Pia | Nadia, Ravi | arrives week 86; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +0.6, Ravi +1.2 (witness) |
| 85 | Nadia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 85 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 86; anxiety Marta +2.5, Nadia +1.0 (witness), Pia +0.5, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6 |
| 85 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.7 (witness), Nadia +1.1 (witness), Pia +0.7 (witness) |
| 85 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.1 (witness), Pia +0.7 (witness) |
| 86 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 87; anxiety Ana +0.3, Marta +2.0, Nadia +1.1 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana takes 2.0 of functioning from the other |
| 86 | Bruno | CUTOFF | Ana | — | arrives week 87; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 86 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 87; anxiety Marta +0.5, Nadia +2.2, Pia +0.7 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Marta +0.6, Nadia -0.6; 7.0 of the sender's anxiety bound into Marta–Nadia |
| 86 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 87; anxiety Marta +1.2 (witness), Nadia +0.7, Pia +0.6 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Nadia -0.5 |
| 86 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 87; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +0.6; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; 5.4 of the sender's anxiety bound into Pia–Ravi |
| 86 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.7 (witness) |
| 86 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.0 (witness), Pia +0.7 (witness) |
| 87 | Ana | CUTOFF | Bruno | — | arrives week 88; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +0.7, Bruno -0.7; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ana -0.1 |
| 87 | Bruno | CUTOFF | Ana | — | arrives week 88; anxiety Ana +0.1, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Bruno -0.2 |
| 87 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | arrives week 88; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +4.0, Ravi +1.2 (witness); Marta takes 2.0 of functioning from the other |
| 87 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 88; anxiety Marta -1.3, Nadia +0.7, Pia +0.6 (witness), Ravi +1.2 (witness); Marta left outside: anxiety Marta +3.7, Nadia -2.5 (of which 1.2 the outsider's own) |
| 87 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 88; anxiety Marta +0.3, Nadia +1.0 (witness), Pia +0.5, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Marta takes 2.0 of functioning from the other |
| 87 | Ravi | STAY-IN-CONTACT | Marta | Nadia, Pia | arrives week 88; anxiety Marta -1.3, Nadia +1.9 (witness), Pia +1.3 (witness), Ravi +0.4 |
| 87 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +1.0 (witness), Pia +0.7 (witness) |
| 88 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +2.1 (witness), Pia +1.4 (witness), Ravi +2.6 |
| 88 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +2.1 (witness), Pia +1.4 (witness), Ravi +1.3 (witness) |
| 88 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 88 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 88 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 88 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 88 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 88 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 89; anxiety Dr Halim +0.1, Marta +0.8 (witness), Nadia +1.0 (witness), Pia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Ravi -0.5 |
| 88 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 89; anxiety Marta +0.4, Nadia +1.9 (witness), Pia +1.4 (witness); calmer sender takes some anxiety: Marta +0.2, Ravi -0.2; Ravi left outside: anxiety Marta -1.7, Ravi +2.6 (of which 0.9 the outsider's own) |
| 88 | Nadia | PURSUE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.5; anxiety Marta +1.3 (witness), Pia +0.7 (witness) |
| 88 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 89; anxiety Marta +2.4, Nadia +1.0 (witness), Pia +0.5, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4 |
| 88 | Ravi | CONFLICT | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +1.3 (witness), Pia +0.7 (witness) |
| 88 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.1 (witness), Pia +0.8 (witness) |
| 89 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 89 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 89 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 89 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 89 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 89 | Ana | CUTOFF | Bruno | — | arrives week 90; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 89 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.6 |
| 89 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 90; anxiety Marta +0.1, Nadia +1.0 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 89 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.5; anxiety Marta +1.2 (witness), Pia +0.7 (witness) |
| 89 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 90; anxiety Marta +1.1 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi -0.9; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; Ravi left outside: anxiety Pia -1.5, Ravi +2.2 (of which 0.7 the outsider's own) |
| 89 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 90; anxiety Marta +1.3, Nadia +1.8 (witness), Pia +1.4 (witness), Ravi +0.4; 7.0 of the sender's anxiety bound into Marta–Ravi |
| 89 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.2 (witness), Pia +0.9 (witness) |
| 90 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 90 | Bruno | CUTOFF | Ana | — | arrives week 91; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 90 | Marta | PROVOKE | Nadia | Pia, Ravi | arrives week 91; anxiety Marta +0.4, Nadia +4.8, Pia +0.8 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Marta +0.9, Nadia -0.9 |
| 90 | Nadia | PURSUE | Marta | Pia, Ravi | arrives week 91; anxiety Marta +1.9, Nadia +0.7, Pia +0.8 (witness), Ravi +1.2 (witness) |
| 90 | Pia | STAY-IN-CONTACT | Ravi | Marta, Nadia | arrives week 91; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi -1.4; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1 |
| 90 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 91; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +1.0, Ravi +0.4 |
| 90 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +1.2 (witness), Pia +0.9 (witness) |
| 91 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 92; anxiety Ana +0.3, Marta +0.2, Nadia +1.0 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 91 | Bruno | CUTOFF | Ana | — | arrives week 92; anxiety Ana +0.1, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 91 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 92; anxiety Marta +0.4, Nadia +1.8 (witness), Pia +1.3 (witness), Ravi +0.1; Ravi takes 2.0 of functioning from the other |
| 91 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.5; anxiety Marta +1.2 (witness), Pia +0.7 (witness) |
| 91 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 92; anxiety Marta +1.1 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +1.7; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; 8.0 of the sender's anxiety bound into Pia–Ravi |
| 91 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +1.2 (witness), Pia +0.7 (witness) |
| 91 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +1.3 (witness), Pia +0.9 (witness) |
| 92 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +2.1 (witness), Pia +1.6 (witness), Ravi +2.6 |
| 92 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.7, Nadia +2.1 (witness), Pia +1.6 (witness), Ravi +1.3 (witness) |
| 92 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 92 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 92 | Ana | CUTOFF | Bruno | — | arrives week 93; anxiety Ana +0.4, Bruno +0.1; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 92 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 92 | Marta | TRIANGLE | Pia | Nadia, Ravi | arrives week 93; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +0.5, Ravi +1.2 (witness); Pia left outside: anxiety Marta -2.1, Pia +3.2 (of which 1.1 the outsider's own) |
| 92 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 93; anxiety Marta +3.9, Nadia +0.7, Pia +0.8 (witness), Ravi +1.3 (witness); Marta holds a position: no functioning changes hands |
| 92 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 93; anxiety Marta +1.2 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +1.3; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 6.8 of the sender's anxiety bound into Pia–Ravi |
| 92 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.0; anxiety Marta +1.4 (witness), Pia +0.8 (witness) |
| 92 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +1.1 (witness), Pia +0.8 (witness) |
| 93 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 93 | Ana | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 93 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 93 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 94; anxiety Marta +0.5, Nadia +0.6, Pia +0.8 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Marta +0.8, Nadia -0.8; 7.1 of the sender's anxiety bound into Marta–Nadia |
| 93 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 94; anxiety Marta +0.4, Nadia +0.7, Pia +0.8 (witness), Ravi +1.2 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Nadia -1.9 |
| 93 | Pia | DETRIANGLE | Marta | Nadia, Ravi | arrives week 94; anxiety Marta +0.2, Nadia +1.0 (witness), Pia +0.5, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9 |
| 93 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 94; anxiety Marta -0.8, Nadia +1.8 (witness), Pia +1.3 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3; Marta left outside: anxiety Marta +2.8, Ravi -1.8 (of which 0.9 the outsider's own) |
| 93 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +1.2 (witness), Pia +0.9 (witness) |
| 94 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 94 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 94 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 94 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 94 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 94 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 95; anxiety Ana +0.3, Marta +0.0, Nadia +1.0 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Marta takes 2.0 of functioning from the other |
| 94 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 94 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 95; anxiety Marta +0.5, Nadia +1.9 (witness), Pia +1.4 (witness), Ravi +3.1; calmer sender takes some anxiety: Marta +0.0, Ravi -0.0; Marta takes 2.0 of functioning from the other |
| 94 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.5; anxiety Marta +1.3 (witness), Pia +0.7 (witness) |
| 94 | Pia | TRIANGLE | Marta | Nadia, Ravi | arrives week 95; anxiety Marta +0.2, Nadia +0.9 (witness), Pia +0.5, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5; Marta left outside: anxiety Marta +1.9, Pia -1.3 (of which 0.6 the outsider's own) |
| 94 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 95; anxiety Marta +3.5, Nadia +1.9 (witness), Pia +1.4 (witness), Ravi +0.4; 7.1 of the sender's anxiety bound into Marta–Ravi |
| 94 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 95 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 95 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 95 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 95 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 95 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.1; anxiety Pia +0.8 (witness), Ravi +1.3 (witness) |
| 95 | Nadia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 95 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 96; anxiety Marta +0.1, Nadia +0.9 (witness), Pia +0.5, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; 6.7 of the sender's anxiety bound into Marta–Pia |
| 95 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 96; anxiety Marta +1.2 (witness), Nadia +0.1, Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -1.3, Ravi +1.3; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 95 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 96 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +2.1 (witness), Pia +1.6 (witness), Ravi +2.6 |
| 96 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.6, Nadia +2.1 (witness), Pia +1.6 (witness), Ravi +1.3 (witness) |
| 96 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 96 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 96 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 97; anxiety Dr Halim +0.1, Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Marta -0.5; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.50; the tie stays open) |
| 96 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 97; anxiety Marta +0.4, Nadia +1.9 (witness), Pia +1.4 (witness), Ravi +0.5; Ravi takes 2.0 of functioning from the other |
| 96 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 97; anxiety Marta +0.3, Nadia +0.7, Pia +0.7 (witness), Ravi +1.2 (witness); Marta left outside: anxiety Marta +3.3, Nadia -2.2 (of which 1.1 the outsider's own) |
| 96 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 97; anxiety Marta +1.1 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi -1.3; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Ravi left outside: anxiety Pia -1.1, Ravi +1.6 (of which 0.5 the outsider's own) |
| 96 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 98; anxiety Marta +0.7 (witness), Nadia +1.2 (witness), Pia +0.9 (witness), Ravi +0.2; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 96 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.2 (witness), Pia +0.9 (witness) |
| 97 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 97 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 97 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 98; anxiety Ana +0.3, Nadia +0.9 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Marta takes 2.0 of functioning from the other |
| 97 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 97 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | arrives week 98; anxiety Marta +0.4, Nadia +0.9 (witness), Pia +0.2, Ravi +1.0 (witness); Pia takes 2.0 of functioning from the other |
| 97 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.5; anxiety Marta +1.4 (witness), Pia +0.8 (witness) |
| 97 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 98; anxiety Marta +1.1 (witness), Nadia +0.9 (witness), Pia +0.5, Ravi +1.8; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 5.7 of the sender's anxiety bound into Pia–Ravi |
| 97 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 98; anxiety Marta +0.9, Nadia +1.8 (witness), Pia +1.4 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3; 6.4 of the sender's anxiety bound into Marta–Ravi |
| 97 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 99; anxiety Marta +0.7 (witness), Nadia +1.1 (witness), Pia +0.8 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 98 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.1 |
| 98 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 98 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.2; anxiety Pia +0.7 (witness), Ravi +1.1 (witness) |
| 98 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Nadia +3.5; anxiety Marta +1.3 (witness), Pia +0.7 (witness) |
| 98 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 99; anxiety Marta +1.2 (witness), Nadia +1.0 (witness), Pia +0.5, Ravi -1.1; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Ravi takes 2.0 of functioning from the other |
| 98 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 99; anxiety Marta +1.2 (witness), Nadia +1.0 (witness), Pia +3.6, Ravi +0.4; Ravi takes 2.0 of functioning from the other |
| 98 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 100; anxiety Marta +0.6 (witness), Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +0.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Sofia rejects the other on Ravi–Sofia (rejection +0.10; the tie stays open) |
| 99 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.1 |
| 99 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 99 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.2; anxiety Pia +0.8 (witness), Ravi +1.1 (witness) |
| 99 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 100; anxiety Marta +0.3, Nadia +0.7, Pia +0.8 (witness), Ravi +1.1 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 99 | Pia | PREVENT_ALIGNMENT | Marta | Nadia, Ravi | arrives week 100; anxiety Nadia +0.9 (witness), Pia +0.5, Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9 |
| 99 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 100; anxiety Marta +3.7, Nadia +2.1 (witness), Pia +1.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.8, Ravi +0.8 |
| 99 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.1 (witness), Pia +0.9 (witness) |
| 100 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +2.1 (witness), Pia +1.7 (witness), Ravi +2.3 |
| 100 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.7, Nadia +2.1 (witness), Pia +1.7 (witness), Ravi +1.1 (witness) |
| 100 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 100 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 100 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.1 |
| 100 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 100 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 101; anxiety Ana +1.6, Marta +0.3, Nadia +1.1 (witness), Pia +0.8 (witness), Ravi +0.6 (witness) |
| 100 | Nadia | PURSUE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.5; anxiety Marta +1.3 (witness), Pia +0.8 (witness) |
| 100 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 101; anxiety Marta +1.3 (witness), Nadia +1.1 (witness), Pia +0.5, Ravi +1.6; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; Pia takes 2.0 of functioning from the other |
| 100 | Ravi | DETRIANGLE | Marta | Nadia, Pia | arrives week 101; anxiety Marta +0.1, Nadia +2.0 (witness), Pia +1.6 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.6, Ravi +0.6 |
| 100 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 101 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 102; anxiety Ana +0.3, Marta +1.2, Nadia +1.1 (witness), Pia +0.8 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 101 | Bruno | CUTOFF | Ana | — | arrives week 102; anxiety Ana +0.0, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 101 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.2; anxiety Pia +0.8 (witness), Ravi +1.2 (witness) |
| 101 | Nadia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 101 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 102; anxiety Marta +1.3 (witness), Nadia +1.0 (witness), Pia +0.5, Ravi -0.2; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1; Ravi left outside: anxiety Pia -1.5, Ravi +2.2 (of which 0.7 the outsider's own) |
| 101 | Ravi | STAY-IN-CONTACT | Marta | Nadia, Pia | arrives week 102; anxiety Marta -0.5, Nadia +1.9 (witness), Pia +1.4 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.6, Ravi +0.6 |
| 101 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +1.2 (witness), Pia +0.9 (witness) |
| 102 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 102 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 102 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.1 |
| 102 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 102 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 103; anxiety Ana +0.5, Marta +0.3, Nadia +1.0 (witness), Pia +0.8 (witness), Ravi +0.6 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -0.6 |
| 102 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +3.5; anxiety Marta +1.3 (witness), Pia +0.7 (witness) |
| 102 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 103; anxiety Nadia +0.9 (witness), Pia +0.5, Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Marta takes 2.0 of functioning from the other |
| 102 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | — |
| 102 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 103 | Ana | OVERFUNCTION | Bruno | — | — |
| 103 | Bruno | CUTOFF | Ana | — | — |
| 103 | Marta | TRIANGLE | Ravi | Nadia, Pia | — |
| 103 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | — |
| 103 | Pia | DISTANCE | Ravi | Marta, Nadia | — |
| 103 | Ravi | CUTOFF | Pia | Marta, Nadia | — |
| 103 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | — |
| 103 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |

104 weeks. The invariants were asserted at the end of every week.
