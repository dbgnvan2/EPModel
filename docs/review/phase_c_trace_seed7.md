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
| config hash | `0544614268435751a674cfc89e8c9c1ed45aa493457587701dc0d0f59373ca92` |
| spec revision | 2.0, revision 21 |
| family instance | phase_c |
| activation | synchronous_activation v1 (synchronous) |
| visibility | household_conductance_visibility v1 |
| constants frozen | 2026-10-07 |
| constants changed since freeze | declared_rejection, estimator_breadth, estimator_rate, estimator_sd_weight, estimator_window, pattern_min_acts, pattern_projection_hold, pattern_window, perspective_channel_gain, perspective_steer_gain, practice_window, refusal_cost_gain, rejection_cutoff_gain, rejection_fade, rejection_refusal_gain, rejection_refusal_threshold, rejection_relief, response_down_gain, response_relaxation_rate, response_up_gain, self_practice_gain, soothing_review_every, soothing_review_penalty, stress_floor_below_basic, stressor_persistence, system_adjust_gain, tie_wear_rate, wear_offset_per_act |

The standing load, the decay toward each person's floor, the relaxation of felt contact, symptom accumulation, investment, the reactive detectors and outside-ness run every week for everyone and are not listed line by line. Effects are shown beside the event that caused them.

| Week | Who | Move | Toward | Witnesses | What it did |
|---|---|---|---|---|---|
| 0 | Ana | WITHHOLD | Bruno | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 0 | Bruno | CUTOFF | Ana | — | arrives week 1; anxiety Ana +1.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 0 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 1; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.3, Marta -0.3 |
| 0 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | arrives week 1; anxiety Marta +0.5, Nadia +0.6 (witness), Pia -0.1, Ravi +1.3 (witness); Pia takes 2.0 of functioning from the other |
| 0 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 1; anxiety Marta -0.3, Nadia +0.5, Pia +0.6 (witness), Ravi +1.3 (witness); Marta left outside: anxiety Marta +0.5, Nadia -0.4 (of which 0.2 the outsider's own) |
| 0 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 1; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +3.8; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia takes 2.0 of functioning from the other |
| 0 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 1; anxiety Marta -0.6, Nadia +1.2 (witness), Pia +1.1 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1; Marta left outside: anxiety Marta +1.2, Ravi -0.8 (of which 0.4 the outsider's own) |
| 0 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 2; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +2.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Sofia takes 2.0 of functioning from the other |
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
| 1 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +3.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5 |
| 2 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 2 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 3; anxiety Ana +0.3, Marta -0.6, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.2, Marta -0.2 |
| 2 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 2 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 3; anxiety Marta +0.1, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.9 (witness) |
| 2 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Pia +0.5 (witness), Ravi +1.4 (witness) |
| 2 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 3; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +0.4, Ravi +4.8; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 2.8 of the sender's anxiety bound into Pia–Ravi |
| 2 | Ravi | STAY-IN-CONTACT | Nadia | Marta, Pia | arrives week 3; anxiety Marta +1.2 (witness), Nadia -0.7, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3 |
| 2 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 4; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +2.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 1.2 of the sender's anxiety bound into Ravi–Sofia |
| 3 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 3 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 4; anxiety Ana +0.3, Marta +2.0, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.2, Marta -0.2; 2.0 of the sender's anxiety bound into Ana–Marta |
| 3 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 3 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 4; anxiety Marta +0.5, Nadia +1.2 (witness), Pia +1.1 (witness), Ravi +1.1; calmer sender takes some anxiety: Marta +0.5, Ravi -0.5 |
| 3 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Pia +0.5 (witness), Ravi +1.4 (witness) |
| 3 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 4; anxiety Marta +3.2, Nadia +0.6 (witness), Pia +0.4, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5 |
| 3 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | arrives week 4; anxiety Marta +1.2 (witness), Nadia +1.2, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.0, Ravi +0.0; Nadia takes 2.0 of functioning from the other |
| 3 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 5; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +3.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia takes 2.0 of functioning from the other |
| 4 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +3.4 |
| 4 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +1.7 (witness) |
| 4 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 4 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 4 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 4 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 4 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 5; anxiety Marta +0.5, Nadia +1.0, Pia +0.5 (witness), Ravi +1.5 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3; Marta rejects the other on Marta–Nadia (rejection +0.45; the tie stays open) |
| 4 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 5; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +5.8; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Nadia takes 2.0 of functioning from the other |
| 4 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 5; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.4, Ravi +0.7; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Ravi left outside: anxiety Pia -1.0, Ravi +1.5 (of which 0.5 the outsider's own) |
| 4 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 5; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +2.8, Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 4 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 6; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +1.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; 1.9 of the sender's anxiety bound into Ravi–Sofia |
| 4 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 5, 6; symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.5, Nadia +2.3, Pia +2.2; anxiety Sofia +1.3 |
| 5 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 5 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 5 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +0.3, Marta +2.6, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana takes 2.0 of functioning from the other |
| 5 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 5 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +2.1, Marta +0.3, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 5 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 6; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.7; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Ravi left outside: anxiety Nadia -1.2, Ravi +1.7 (of which 0.6 the outsider's own) |
| 5 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 6; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.4; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Ravi left outside: anxiety Pia -1.1, Ravi +1.6 (of which 0.5 the outsider's own) |
| 5 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | arrives week 6; anxiety Marta +1.3 (witness), Nadia +1.0, Pia +0.6 (witness), Ravi +0.6; Nadia takes 2.0 of functioning from the other |
| 5 | Sofia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 6 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 6 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.1 |
| 6 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 6 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | arrives week 7; anxiety Marta +0.5, Nadia +0.5 (witness), Pia +3.5, Ravi +1.7 (witness); Marta takes 2.0 of functioning from the other |
| 6 | Nadia | PREVENT_ALIGNMENT | Ravi | Marta, Pia | arrives week 7; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.6; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 6 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 7; anxiety Marta +1.3 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +1.4; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Ravi takes 2.0 of functioning from the other |
| 6 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 7; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 6 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 8; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +2.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1 |
| 6 | Marta | SYMPTOM_ONSET | Ana, Dr Halim, Nadia, Pia, Ravi | — | arrives week 7; symptom onset, physical channel; anxiety Ana +1.5, Dr Halim +0.5, Nadia +2.1, Pia +2.3, Ravi +3.5 |
| 7 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 8; anxiety Ana +0.3, Marta +2.7, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana takes 2.0 of functioning from the other |
| 7 | Bruno | CUTOFF | Ana | — | arrives week 8; anxiety Ana +0.6, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 7 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | arrives week 8; anxiety Marta +0.5, Nadia +0.1, Pia +0.6 (witness), Ravi +1.6 (witness); calmer sender takes some anxiety: Marta +0.1, Nadia -0.1; Nadia takes 2.0 of functioning from the other |
| 7 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 8; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.4; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 7 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 8; anxiety Marta +1.3 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +1.1; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Pia–Ravi functioning balance flips; Ravi takes 2.0 of functioning from the other |
| 7 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | arrives week 9; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +1.7 |
| 7 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +2.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3 |
| 7 | Nadia | SYMPTOM_ONSET | Marta, Ravi | Pia | arrives week 8; symptom onset, mental channel; anxiety Marta +2.5, Pia +0.6 (witness), Ravi +3.2 |
| 7 | Pia | SYMPTOM_ONSET | Marta, Ravi | Nadia | arrives week 8; symptom onset, social channel; anxiety Marta +2.5, Nadia +0.5 (witness), Ravi +3.2 |
| 8 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.9 |
| 8 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.9 (witness) |
| 8 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.3, Marta +3.0, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 8 | Bruno | CUTOFF | Ana | — | arrives week 9; anxiety Ana +0.8, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 8 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Dr Halim +0.1, Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.8, Ravi -0.8 |
| 8 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.4, Marta +0.3, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); Ana holds a position: no functioning changes hands |
| 8 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 9; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.7, Ravi -0.7; 6.9 of the sender's anxiety bound into Nadia–Ravi |
| 8 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 9; anxiety Marta +0.9, Nadia +0.5 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; 8.3 of the sender's anxiety bound into Marta–Pia |
| 8 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | arrives week 10; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +3.0 |
| 8 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 10; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; 3.4 of the sender's anxiety bound into Ravi–Sofia |
| 8 | Bruno | SYMPTOM_ONSET | Ana | — | arrives week 9; symptom onset, social channel; anxiety Ana +2.0 |
| 9 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 9 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 9 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 10; anxiety Ana +0.3, Marta +3.5, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Ana takes 2.0 of functioning from the other |
| 9 | Bruno | CUTOFF | Ana | — | arrives week 10; anxiety Ana +0.7, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 9 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 10; anxiety Marta +0.6, Nadia -0.5, Pia +0.7 (witness), Ravi +1.5 (witness); Nadia left outside: anxiety Marta -3.0, Nadia +4.5 (of which 1.5 the outsider's own) |
| 9 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 10; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +3.9; calmer sender takes some anxiety: Nadia +0.8, Ravi -0.8 |
| 9 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 10; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +1.4, Ravi -1.4; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -1.6 |
| 9 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 10; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.3, Ravi +0.6; Pia left outside: anxiety Pia +4.9, Ravi -3.3 (of which 1.6 the outsider's own) |
| 9 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 11; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +2.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 10 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 10 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 10 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 10 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 10 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 10 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 10 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 11; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +2.9, Ravi +1.5 (witness); 11.7 of the sender's anxiety bound into Marta–Pia |
| 10 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 11; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 10 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 11; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +5.6; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Pia takes 2.0 of functioning from the other |
| 10 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 11; anxiety Marta +3.4, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.6 |
| 10 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 12; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +4.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1 |
| 10 | Ana | SYMPTOM_ONSET | Bruno, Marta | Nadia, Pia, Ravi | arrives week 11; symptom onset, physical channel; anxiety Bruno +2.8, Marta +1.8, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 11 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 12; anxiety Ana +0.3, Marta +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 11 | Bruno | CUTOFF | Ana | — | arrives week 12; anxiety Ana +0.6, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 11 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 12; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +4.8, Ravi +1.6 (witness) |
| 11 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 12; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.0; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 11 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 11 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 12; anxiety Marta +5.1, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.7 |
| 11 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 13; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Sofia -1.2 |
| 12 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.8 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +3.9 |
| 12 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.6, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +2.0 (witness) |
| 12 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +0.3, Marta +1.0, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; 4.2 of the sender's anxiety bound into Ana–Marta |
| 12 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 12 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +1.6, Marta +0.4, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 12 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 13; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.9; calmer sender takes some anxiety: Nadia +0.8, Ravi -0.8; 6.7 of the sender's anxiety bound into Nadia–Ravi |
| 12 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 12 | Ravi | PURSUE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +1.5 (witness), Nadia +0.5 (witness) |
| 12 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 14; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; 2.4 of the sender's anxiety bound into Ravi–Sofia |
| 13 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 13 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 13 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 13 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 13 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 14; anxiety Marta +0.6, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.8; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; Ravi left outside: anxiety Marta -2.3, Ravi +3.4 (of which 1.1 the outsider's own) |
| 13 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 13 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 14; anxiety Marta +5.7, Nadia +0.5 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6 |
| 13 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 14; anxiety Marta +1.1, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.7; Marta left outside: anxiety Marta +4.6, Ravi -3.1 (of which 1.5 the outsider's own) |
| 13 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 14 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 14 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 14 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 14 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 14 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 15; anxiety Ana +0.3, Marta +0.8, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9; Marta takes 2.0 of functioning from the other |
| 14 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 14 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 15; anxiety Marta +0.6, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +1.1; calmer sender takes some anxiety: Marta +0.0, Ravi -0.0; Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -3.4 |
| 14 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 14 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 15; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +1.0; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 14 | Ravi | WITHHOLD | Sofia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 14 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 16; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +4.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 15 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 15 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 15 | Marta | PREVENT_ALIGNMENT | Ravi | Nadia, Pia | arrives week 16; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +1.0; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4 |
| 15 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 15 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 16; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Pia rejects the other on Pia–Ravi (rejection +0.10; the tie stays open) |
| 15 | Ravi | PURSUE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 15 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 17; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +4.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 16 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +3.9 |
| 16 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +2.0 (witness) |
| 16 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 16 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 16 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 17; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6 |
| 16 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 17; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +4.0; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4 |
| 16 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 17; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 16 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 17; anxiety Marta +2.8, Nadia +0.5 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6 |
| 16 | Ravi | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 16 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 18; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3 |
| 17 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 17 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 17 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 17 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 18; anxiety Ana +0.3, Marta +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 17 | Bruno | CUTOFF | Ana | — | arrives week 18; anxiety Ana +0.3, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 17 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 18; anxiety Marta +0.2, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness) |
| 17 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 17 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 17 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 18; anxiety Marta +1.4 (witness), Nadia +1.5, Pia +0.6 (witness), Ravi +0.7; 12.9 of the sender's anxiety bound into Nadia–Ravi |
| 17 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 19; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 18 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 18 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 18 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | arrives week 19; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +0.1, Ravi +1.6 (witness); Pia takes 2.0 of functioning from the other |
| 18 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 18 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 19; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -0.7 |
| 18 | Ravi | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 18 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 20; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Ravi takes 2.0 of functioning from the other |
| 19 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 20; anxiety Ana +0.3, Marta +1.9, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 19 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 19 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 20; anxiety Marta +0.6, Nadia +2.5, Pia +0.6 (witness), Ravi +1.6 (witness); calmer sender takes some anxiety: Marta +0.2, Nadia -0.2; Nadia holds a position: no functioning changes hands |
| 19 | Nadia | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 19 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 20; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +1.4; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; 7.7 of the sender's anxiety bound into Pia–Ravi |
| 19 | Ravi | WITHHOLD | Dr Halim | — | held back SPLIT: computed, not emitted; attention on the tie rises |
| 19 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 21; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Ravi takes 2.0 of functioning from the other |
| 20 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +3.9 |
| 20 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +2.0 (witness) |
| 20 | Ana | CUTOFF | Bruno | — | arrives week 21; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 20 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 20 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 21; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +0.1, Ravi +1.5 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -4.4 |
| 20 | Nadia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 20 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 21; anxiety Marta +2.2, Nadia +0.5 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0 |
| 20 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.1; anxiety Marta +1.5 (witness), Nadia +0.5 (witness) |
| 20 | Sofia | REDUCE_CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 22; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia's rejection -0.05 |
| 21 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 21 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 21 | Ana | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 21 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 21 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 22; anxiety Marta +0.6, Nadia +1.0 (witness), Pia +1.2 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; Ravi left outside: anxiety Marta -1.5, Ravi +2.3 (of which 0.8 the outsider's own) |
| 21 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 22; anxiety Nadia +0.4, Pia +0.6 (witness), Ravi +1.4 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Nadia -1.3 |
| 21 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.4; anxiety Nadia +0.5 (witness), Ravi +1.4 (witness) |
| 21 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 23; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -5.7 |
| 21 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 23; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +3.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia takes 2.0 of functioning from the other |
| 21 | Sofia | SYMPTOM_ONSET | Ravi | Marta, Nadia, Pia | arrives week 23; symptom onset, physical channel; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.7 |
| 22 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 22 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 22 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 22 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 22 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 22 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 23; anxiety Ana +0.3, Marta +1.5, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; 2.8 of the sender's anxiety bound into Ana–Marta |
| 22 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 22 | Marta | PROVOKE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.9; anxiety Nadia +0.5 (witness), Ravi +1.4 (witness) |
| 22 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 22 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 23; anxiety Marta +0.2, Nadia +0.5 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 22 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 23; anxiety Marta +1.0 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 22 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 22 | — | (system) | — | — | Nadia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 23 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 24; anxiety Ana +0.3, Marta -0.7, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 23 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 23 | Marta | PROVOKE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.9; anxiety Nadia +0.5 (witness), Ravi +1.5 (witness) |
| 23 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.5 (witness) |
| 23 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 24; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; 7.3 of the sender's anxiety bound into Pia–Ravi |
| 23 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 24; anxiety Marta -0.9, Nadia +1.0 (witness), Pia +1.2 (witness), Ravi +0.6; Marta left outside: anxiety Marta +4.1, Ravi -2.7 (of which 1.4 the outsider's own) |
| 23 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 24 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.8 (witness), Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +3.7 |
| 24 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +1.9 (witness) |
| 24 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 24 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 24 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 24 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 24 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 24 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 25; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 24 | Marta | STAY-IN-CONTACT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.9; anxiety Nadia +0.5 (witness), Ravi +1.5 (witness) |
| 24 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.5 (witness) |
| 24 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 25; anxiety Marta +0.4, Nadia +0.5 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -1.2, Pia +1.2; Pia rejects the other on Marta–Pia (rejection +0.00; the tie stays open) |
| 24 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 25; anxiety Marta +3.1, Nadia +1.0 (witness), Pia +1.2 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.2, Ravi +0.2 |
| 24 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 25 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | arrives week 26; anxiety Ana +0.3, Marta +2.4, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 25 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 25 | Marta | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 25 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 25 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 26; anxiety Marta +0.5, Nadia +0.5 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open) |
| 25 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.0; anxiety Marta +1.5 (witness), Nadia +0.5 (witness) |
| 25 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 27; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 25 | — | (system) | — | — | Marta–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 26 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 26 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | arrives week 27; anxiety Ana +0.3, Marta +2.4, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 26 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 26 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 27; anxiety Marta +0.6, Nadia -0.9, Pia +0.6 (witness), Ravi +1.5 (witness); calmer sender takes some anxiety: Marta +0.1, Nadia -0.1; Nadia left outside: anxiety Marta -2.1, Nadia +3.1 (of which 1.0 the outsider's own) |
| 26 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.5 (witness) |
| 26 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 26 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 27; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 26 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 28; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 27 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 27 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 28; anxiety Ana +0.3, Marta +2.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 27 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 27 | Marta | WITHHOLD | Nadia | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 27 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 28; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.6 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 27 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 28; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.1; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; 6.6 of the sender's anxiety bound into Pia–Ravi |
| 27 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.8; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 27 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 28 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +3.6 |
| 28 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +1.8 (witness) |
| 28 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 28 | Ana | CUTOFF | Bruno | — | arrives week 29; anxiety Ana +0.4, Bruno +0.3; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 28 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 28 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 29; anxiety Marta +0.6, Nadia +1.0 (witness), Pia +1.2 (witness), Ravi +0.1; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1; Ravi left outside: anxiety Marta -2.2, Ravi +3.3 (of which 1.1 the outsider's own) |
| 28 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.5 (witness) |
| 28 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 29; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.3; calmer sender takes some anxiety: Pia +1.3, Ravi -1.3; 5.2 of the sender's anxiety bound into Pia–Ravi |
| 28 | Ravi | CONFLICT | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.0; anxiety Marta +1.5 (witness), Nadia +0.5 (witness) |
| 28 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 29 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 29 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 29 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 29 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 29 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 29 | Bruno | CUTOFF | Ana | — | arrives week 30; anxiety Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.35; the tie stays open); impingement removed at once, anxiety Bruno -0.0 |
| 29 | Marta | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 29 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 30; anxiety Marta +0.1, Nadia +0.4, Pia +0.6 (witness), Ravi +1.5 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 29 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 29 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 30; anxiety Marta -0.1, Nadia +1.0 (witness), Pia +1.2 (witness), Ravi +0.6; Marta left outside: anxiety Marta +3.8, Ravi -2.5 (of which 1.3 the outsider's own) |
| 29 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 30 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 30 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 30 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 30 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 31; anxiety Ana +0.3, Marta +3.8, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 30 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 31; anxiety Marta +0.2, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.0 (witness) |
| 30 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 31; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi -0.3; Ravi takes 2.0 of functioning from the other |
| 30 | Pia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 30 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.7; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 30 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 31 | Ana | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 31 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 31 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 32; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.2 (witness), Ravi -0.3; calmer sender takes some anxiety: Marta +0.0, Ravi -0.0; Ravi left outside: anxiety Marta -2.4, Ravi +3.6 (of which 1.2 the outsider's own) |
| 31 | Nadia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 31 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 32; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 31 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.9; anxiety Marta +1.5 (witness), Pia +0.6 (witness) |
| 31 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 32 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.8 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.4 |
| 32 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.7 (witness) |
| 32 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 32 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 32 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 32 | Ana | CUTOFF | Bruno | — | arrives week 33; anxiety Ana +0.4, Bruno +0.3; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 32 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 32 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 33; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.50; the tie stays open) |
| 32 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.9; anxiety Pia +0.6 (witness), Ravi +1.4 (witness) |
| 32 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 33; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +2.6; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 32 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 33; anxiety Marta -0.0, Nadia +0.5 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -1.2, Pia +1.2; Marta takes 2.0 of functioning from the other |
| 32 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 33; anxiety Marta +1.0 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 32 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 33 | Ana | CUTOFF | Bruno | — | arrives week 34; anxiety Ana +0.4, Bruno +0.8; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 33 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 33 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +0.8; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness) |
| 33 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 34; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +5.9; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2 |
| 33 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 34; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.5; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; 6.3 of the sender's anxiety bound into Pia–Ravi |
| 33 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 34; anxiety Dr Halim +0.1, Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 33 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 34 | Ana | I-POSITION | Marta | Nadia, Pia, Ravi | arrives week 35; anxiety Ana +0.3, Marta +3.4, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 34 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 34 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 35; anxiety Marta +0.3, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -4.1 |
| 34 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 34 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 35; anxiety Marta +4.4, Nadia +0.5 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -1.1, Pia +1.1 |
| 34 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | arrives week 35; anxiety Marta +1.4 (witness), Nadia +2.2, Pia +0.6 (witness), Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 34 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 35 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 35 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 35 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 35 | Ana | REDUCE_CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 36; anxiety Ana +0.3, Marta +0.4, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Marta's rejection -0.25 |
| 35 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 35 | Marta | TRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.8; anxiety Nadia +0.6 (witness), Ravi +1.4 (witness) |
| 35 | Nadia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 35 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 36; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.0; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; 6.1 of the sender's anxiety bound into Pia–Ravi |
| 35 | Ravi | WITHHOLD | Sofia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 35 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 36 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.2 |
| 36 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 36 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 36 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 36 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 37; anxiety Ana +0.3, Marta +3.6, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9 |
| 36 | Bruno | CUTOFF | Ana | — | arrives week 37; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.35; the tie stays open) |
| 36 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.8; anxiety Pia +0.6 (witness), Ravi +1.4 (witness) |
| 36 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 37; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +5.1; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 36 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 37; anxiety Marta +0.4, Nadia +0.6 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -1.4, Pia +1.4; 5.2 of the sender's anxiety bound into Marta–Pia |
| 36 | Ravi | PURSUE | Nadia | Marta, Pia | arrives week 37; anxiety Marta +1.4 (witness), Nadia +0.7, Pia +0.6 (witness), Ravi +0.5 |
| 36 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 37 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 37 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 37 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +0.8; anxiety Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.9 (witness) |
| 37 | Nadia | PURSUE | Marta | Pia, Ravi | arrives week 38; anxiety Marta +2.8, Nadia +0.4, Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4 |
| 37 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 38; anxiety Marta +5.6, Nadia +0.6 (witness), Pia +0.5, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -1.5, Pia +1.5 |
| 37 | Ravi | WITHHOLD | Pia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 37 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 38 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 38 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 38 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.8; anxiety Nadia +0.6 (witness), Ravi +1.4 (witness) |
| 38 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 39; anxiety Marta +5.1, Nadia +0.4, Pia +0.6 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.6, Nadia +0.6; Nadia takes 2.0 of functioning from the other |
| 38 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 39; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.0; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Ravi takes 2.0 of functioning from the other |
| 38 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 39; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Ravi +0.5; Ravi rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -0.1 |
| 38 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 39 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 39 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 40; anxiety Ana +0.3, Marta +4.1, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +1.1, Marta -1.1 |
| 39 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 39 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.0; anxiety Pia +0.7 (witness), Ravi +1.3 (witness) |
| 39 | Nadia | CONFLICT | Marta | Pia, Ravi | arrives week 40; anxiety Marta +3.3, Nadia +0.4, Pia +0.7 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.8, Nadia +0.8 |
| 39 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 39 | Ravi | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 39 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 40 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.8 (witness), Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +3.1 |
| 40 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +1.5 (witness) |
| 40 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 40 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 40 | Dr Halim | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 41; anxiety Dr Halim +0.1, Marta +1.0 (witness), Nadia +0.5 (witness), Pia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Ravi -0.7; Dr Halim rejects the other on Dr Halim–Ravi (rejection +0.50; the tie stays open) |
| 40 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 41; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.1; Ravi left outside: anxiety Marta -3.4, Ravi +5.1 (of which 1.7 the outsider's own) |
| 40 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 41; anxiety Marta +1.1, Nadia +0.4, Pia +0.7 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.9, Nadia +0.9; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 40 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 41; anxiety Marta +6.1, Nadia +0.5 (witness), Pia +0.5, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -1.7, Pia +1.7 |
| 40 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.6; anxiety Marta +1.5 (witness), Nadia +0.5 (witness) |
| 40 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 40 | — | (system) | — | — | Ana–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 40 | — | (system) | — | — | Nadia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 41 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 41 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 41 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 41 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 41 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 42; anxiety Ana +0.3, Marta +0.7, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +1.1, Marta -1.1; Marta takes 2.0 of functioning from the other |
| 41 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 41 | Marta | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 41 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 42; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +3.3; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 41 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 42; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Pia rejects the other on Pia–Ravi (rejection +0.40; the tie stays open) |
| 41 | Ravi | CONFLICT | Nadia | Marta, Pia | arrives week 42; anxiety Marta +1.5 (witness), Nadia +3.6, Pia +0.7 (witness), Ravi +0.5 |
| 41 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 43; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 42 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.1 |
| 42 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 42 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 43; anxiety Marta +0.2, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 42 | Nadia | PURSUE | Marta | Pia, Ravi | arrives week 43; anxiety Marta +3.6, Nadia +0.4, Pia +0.7 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.5, Nadia +0.5 |
| 42 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 42 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 43; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.5; Nadia left outside: anxiety Nadia +3.2, Ravi -2.1 (of which 1.1 the outsider's own) |
| 42 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 43 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 43 | Ana | WITHHOLD | Bruno | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 43 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 43 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 44; anxiety Marta +0.3, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -1.0 |
| 43 | Nadia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 43 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 44; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.2; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; 7.2 of the sender's anxiety bound into Pia–Ravi |
| 43 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 44; anxiety Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.6, Ravi +0.6; Marta left outside: anxiety Marta +2.7, Ravi -1.8 (of which 0.9 the outsider's own) |
| 43 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 44 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +3.1 |
| 44 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.4, Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +1.5 (witness) |
| 44 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 44 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 44 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 44 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 44 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 44 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 45; anxiety Dr Halim +0.0, Marta +0.2, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 44 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 45; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.2; Ravi takes 2.0 of functioning from the other |
| 44 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 44 | Ravi | WITHHOLD | Pia | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 44 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 45 | Ana | WITHHOLD | Bruno | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 45 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 45 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 46; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi -0.6; Ravi takes 2.0 of functioning from the other |
| 45 | Nadia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 45 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 46; anxiety Marta +0.4, Nadia +0.6 (witness), Pia +0.5, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -1.2, Pia +1.2; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 45 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 46; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Nadia left outside: anxiety Nadia +2.4, Ravi -1.6 (of which 0.8 the outsider's own) |
| 45 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 46 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 46 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.1 |
| 46 | Bruno | CUTOFF | Ana | — | arrives week 47; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 46 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.0; anxiety Nadia +0.6 (witness), Ravi +1.2 (witness) |
| 46 | Nadia | CONFLICT | Marta | Pia, Ravi | arrives week 47; anxiety Marta +6.4, Nadia +0.4, Pia +0.7 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.2, Nadia +0.2 |
| 46 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 47; anxiety Marta +0.4, Nadia +0.6 (witness), Pia +0.5, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -1.1, Pia +1.1; Marta takes 2.0 of functioning from the other |
| 46 | Ravi | STAY-IN-CONTACT | Marta | Nadia, Pia | arrives week 47; anxiety Marta +0.0, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.7, Ravi +0.7 |
| 46 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 48; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6; Sofia rejects the other on Ravi–Sofia (rejection +0.35; the tie stays open) |
| 47 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 47 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 47 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.1 |
| 47 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.9 |
| 47 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 48; anxiety Dr Halim +0.0, Marta +0.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 47 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 48; anxiety Marta +0.8, Nadia +0.4, Pia +0.7 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.2, Nadia +0.2; Marta takes 2.0 of functioning from the other |
| 47 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 48; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.6; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; 7.2 of the sender's anxiety bound into Pia–Ravi |
| 47 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 47 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 47 | — | (system) | — | — | Dr Halim–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 48 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.8 |
| 48 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.4, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.4 (witness) |
| 48 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.6; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 48 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 48 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 49; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7 |
| 48 | Marta | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 48 | Nadia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 48 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 49; anxiety Marta +0.3, Nadia +0.6 (witness), Pia +0.6, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -1.3, Pia +1.3; Pia rejects the other on Marta–Pia (rejection +0.35; the tie stays open) |
| 48 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 48 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 49 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 49 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.1 |
| 49 | Bruno | WITHHOLD | Ana | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 49 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 50; anxiety Marta +0.5, Nadia +1.2 (witness), Pia +1.4 (witness); Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -0.0 |
| 49 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 50; anxiety Marta +4.4, Nadia +0.5, Pia +0.7 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.0, Nadia +0.0; Nadia takes 2.0 of functioning from the other |
| 49 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 49 | Ravi | STAY-IN-CONTACT | Marta | Nadia, Pia | arrives week 50; anxiety Marta -1.0, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3 |
| 49 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 50 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.6; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 50 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.9 |
| 50 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 51; anxiety Marta +0.6, Nadia +0.6 (witness), Ravi +1.2 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 50 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 50 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 51; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi -0.2; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 50 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 51; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.1 |
| 50 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 51 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 51 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 51 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 52; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi -1.3; Ravi left outside: anxiety Marta -2.6, Ravi +3.9 (of which 1.3 the outsider's own) |
| 51 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 52; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi -0.8; Ravi takes 2.0 of functioning from the other |
| 51 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 52; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +2.2; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; 7.4 of the sender's anxiety bound into Pia–Ravi |
| 51 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 52; anxiety Marta +1.4 (witness), Pia +0.7 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia -0.4, Ravi +0.4; Nadia left outside: anxiety Nadia +2.7, Ravi -1.8 (of which 0.9 the outsider's own) |
| 51 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 51 | — | (system) | — | — | Dr Halim–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 51 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |
| 52 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.8 |
| 52 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.4, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.4 (witness) |
| 52 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 52 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 52 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 52 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 52 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 52 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 52 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 52 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 52 | Marta | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 52 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 53; anxiety Marta +0.7, Nadia +0.5, Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.0, Nadia +0.0; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 52 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.7; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 52 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 53; anxiety Marta +1.4 (witness), Nadia +0.2, Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.4, Ravi +0.4; Nadia left outside: anxiety Nadia +2.5, Ravi -1.7 (of which 0.8 the outsider's own) |
| 52 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 53 | Ana | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 53 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 53 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 54; anxiety Marta +0.6, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +2.0; 8.6 of the sender's anxiety bound into Marta–Ravi |
| 53 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 54; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +1.7 |
| 53 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 54; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.3; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; 6.1 of the sender's anxiety bound into Pia–Ravi |
| 53 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 55; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.30; the tie stays open) |
| 53 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 54 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 54 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 54 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 55; anxiety Marta +0.6, Nadia +0.6 (witness), Ravi +1.1 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 54 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 55; anxiety Marta -1.2, Nadia +0.5, Pia +0.7 (witness), Ravi +1.1 (witness); Marta left outside: anxiety Marta +2.3, Nadia -1.5 (of which 0.8 the outsider's own) |
| 54 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.7; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 54 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 55; anxiety Marta +2.8, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.0, Ravi +0.0; Ravi takes 2.0 of functioning from the other |
| 54 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 55 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 55 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 55 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 55 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness) |
| 55 | Bruno | CUTOFF | Ana | — | arrives week 56; anxiety Ana +0.3, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open) |
| 55 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 56; anxiety Marta +0.4, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 55 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 56; anxiety Marta +1.5 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.4; Ravi takes 2.0 of functioning from the other |
| 55 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +2.7; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 55 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 56; anxiety Marta +1.5 (witness), Nadia +0.6 (witness), Ravi +0.4; Ravi rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 55 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 56 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.8 (witness), Nadia +1.4 (witness), Pia +1.7 (witness), Ravi +2.6 |
| 56 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.6, Nadia +1.4 (witness), Pia +1.7 (witness), Ravi +1.3 (witness) |
| 56 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness) |
| 56 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 56 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 57; anxiety Dr Halim +0.1, Marta -0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 56 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 57; anxiety Marta +0.6, Nadia +1.2 (witness), Pia +1.4 (witness); Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -2.3 |
| 56 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 57; anxiety Marta +4.6, Nadia +0.5, Pia +0.7 (witness), Ravi +1.0 (witness); Nadia takes 2.0 of functioning from the other |
| 56 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +2.7; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 56 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 57; anxiety Marta +0.5, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.2, Ravi +0.2; Ravi rejects the other on Marta–Ravi (rejection +0.50; the tie stays open) |
| 56 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 57 | Ana | CUTOFF | Bruno | — | arrives week 58; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -0.7 |
| 57 | Bruno | CUTOFF | Ana | — | arrives week 58; anxiety Ana +0.7, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 57 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Marta +3.2; anxiety Nadia +1.2 (witness), Pia +1.4 (witness) |
| 57 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 58; anxiety Marta +1.6 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.8; 7.1 of the sender's anxiety bound into Nadia–Ravi |
| 57 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.7; anxiety Marta +1.6 (witness), Nadia +0.6 (witness) |
| 57 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 59; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 57 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 58 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 58 | Ana | CUTOFF | Bruno | — | arrives week 59; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.7, Bruno -0.7; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 58 | Bruno | WITHHOLD | Ana | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 58 | Marta | CONFLICT | Ravi | Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Marta +3.2; anxiety Nadia +1.2 (witness), Pia +1.4 (witness) |
| 58 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 59; anxiety Marta +5.5, Nadia +0.5, Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4; Nadia takes 2.0 of functioning from the other |
| 58 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.7; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 58 | Ravi | STAY-IN-CONTACT | Nadia | Marta, Pia | arrives week 59; anxiety Marta +1.6 (witness), Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.1, Ravi +0.1 |
| 58 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 59 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 59 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 59 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.4; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 59 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 60; anxiety Marta +3.6, Nadia +0.4, Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.7, Nadia +0.7; Nadia takes 2.0 of functioning from the other |
| 59 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.7; anxiety Marta +1.7 (witness), Nadia +0.6 (witness) |
| 59 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | arrives week 60; anxiety Marta +1.7 (witness), Nadia +0.6 (witness), Pia -0.1, Ravi +0.4; calmer sender takes some anxiety: Pia -0.2, Ravi +0.2; Pia takes 2.0 of functioning from the other |
| 59 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 60 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.1 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.5 |
| 60 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.1, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.2 (witness) |
| 60 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 60 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 61; anxiety Ana +0.3, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +1.1, Marta -1.1; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 60 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 60 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 61; anxiety Marta +0.2, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.7 (witness) |
| 60 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 61; anxiety Marta +1.1, Nadia +0.4, Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.8, Nadia +0.8; Marta takes 2.0 of functioning from the other |
| 60 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Marta +1.8 (witness), Nadia +0.5 (witness) |
| 60 | Ravi | CONFLICT | Nadia | Marta, Pia | arrives week 61; anxiety Marta +1.8 (witness), Nadia +3.8, Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.2, Ravi +0.2 |
| 60 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 61 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 61 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 61 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.4; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 61 | Nadia | CONFLICT | Marta | Pia, Ravi | arrives week 62; anxiety Marta +4.6, Nadia +0.4, Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.7, Nadia +0.7 |
| 61 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 61 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 62; anxiety Marta +1.7 (witness), Nadia +0.6 (witness), Pia +2.5, Ravi +0.4; calmer sender takes some anxiety: Pia -0.1, Ravi +0.1 |
| 61 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 62 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness) |
| 62 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 62 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +2.1; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness) |
| 62 | Nadia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 62 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 63; anxiety Marta +1.7 (witness), Nadia +0.6 (witness), Pia +0.5; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -1.6 |
| 62 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 63; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Nadia left outside: anxiety Nadia +2.4, Ravi -1.6 (of which 0.8 the outsider's own) |
| 62 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 63 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 63 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 63 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 63 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 63 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.4; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 63 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 64; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.5 |
| 63 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 63 | Ravi | CONFLICT | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +1.7 (witness), Nadia +0.6 (witness) |
| 63 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 64 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.1 (witness), Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +2.6 |
| 64 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.2, Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +1.3 (witness) |
| 64 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 64 | Bruno | CUTOFF | Ana | — | arrives week 65; anxiety Ana +0.2, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 64 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 65; anxiety Dr Halim +0.1, Marta +1.1 (witness), Nadia +0.6 (witness), Pia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Ravi -0.5; the contact lands with Ravi: systems perspective +0.10 |
| 64 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 65; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi -0.8; Ravi left outside: anxiety Marta -3.6, Ravi +5.5 (of which 1.8 the outsider's own) |
| 64 | Nadia | PREVENT_ALIGNMENT | Ravi | Marta, Pia | arrives week 65; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.6 |
| 64 | Pia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 64 | Ravi | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 64 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 65 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 65 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 65 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 65 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 65 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 65 | Bruno | CUTOFF | Ana | — | arrives week 66; anxiety Ana +0.6, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 65 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.4; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 65 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 66; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +2.6 |
| 65 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 65 | Ravi | STAY-IN-CONTACT | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +1.7 (witness), Nadia +0.6 (witness) |
| 65 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 66 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 66 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness) |
| 66 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 66 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +2.1; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness) |
| 66 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 67; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.6; 6.5 of the sender's anxiety bound into Nadia–Ravi |
| 66 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 67; anxiety Marta +0.5, Nadia +0.6 (witness), Pia +0.5, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -1.2, Pia +1.2; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 66 | Ravi | STAY-IN-CONTACT | Marta | Nadia, Pia | arrives week 67; anxiety Marta +0.1, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.8, Ravi +0.8 |
| 66 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 67 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 67 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 67 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 68; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.8; 13.5 of the sender's anxiety bound into Marta–Ravi |
| 67 | Nadia | PROVOKE | Ravi | Marta, Pia | arrives week 68; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +3.2; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 67 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 67 | Ravi | CONFLICT | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +1.7 (witness), Nadia +0.6 (witness) |
| 67 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 67 | — | (system) | — | — | Marta–Nadia: rejection has faded; the tie is no longer cut off (ordinary) |
| 68 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.1 (witness), Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +2.6 |
| 68 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.2, Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +1.3 (witness) |
| 68 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 68 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 68 | Ana | CUTOFF | Bruno | — | arrives week 69; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 68 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 68 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 69; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta +0.0, Ravi -0.0; Ravi holds a position: no functioning changes hands |
| 68 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 69; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +2.4; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 68 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 68 | Ravi | PREVENT_ALIGNMENT | Marta | Nadia, Pia | arrives week 69; anxiety Marta +0.2, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.4 |
| 68 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 69 | Ana | CUTOFF | Bruno | — | arrives week 70; anxiety Ana +0.4, Bruno +0.1; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 69 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 69 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 70; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.6; 8.8 of the sender's anxiety bound into Marta–Ravi |
| 69 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 70; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; 5.8 of the sender's anxiety bound into Nadia–Ravi |
| 69 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 69 | Ravi | WITHHOLD | Sofia | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 69 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 70 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.6; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness) |
| 70 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 70 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 71; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2; 5.5 of the sender's anxiety bound into Marta–Ravi |
| 70 | Nadia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 70 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 70 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 71; anxiety Marta -1.6, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.4; Marta left outside: anxiety Marta +3.5, Ravi -2.3 (of which 1.2 the outsider's own) |
| 70 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 71 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 71 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 71 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 71 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 71 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 71 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 71 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 71 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 71 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.5; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 71 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 72; anxiety Marta +0.5, Nadia +0.4, Pia +0.7 (witness), Ravi +1.1 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 71 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 72; anxiety Marta +0.9, Nadia +0.6 (witness), Pia +0.5, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.0, Pia +0.0; Pia rejects the other on Marta–Pia (rejection +0.15; the tie stays open) |
| 71 | Ravi | STAY-IN-CONTACT | Marta | Nadia, Pia | arrives week 72; anxiety Marta -2.7, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.4 |
| 71 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 73; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 71 | — | (system) | — | — | Marta–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 72 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.1 (witness), Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +2.6 |
| 72 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.3, Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +1.3 (witness) |
| 72 | Ana | WITHHOLD | Bruno | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 72 | Bruno | CUTOFF | Ana | — | arrives week 73; anxiety Ana +0.2, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.15; the tie stays open) |
| 72 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 73; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Marta -0.5 |
| 72 | Marta | PROVOKE | Ravi | Nadia, Pia | arrives week 73; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +2.7; calmer sender takes some anxiety: Marta +0.0, Ravi -0.0 |
| 72 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 73; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.9; 5.6 of the sender's anxiety bound into Nadia–Ravi |
| 72 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 73; anxiety Marta +1.7 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi -0.4; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; Ravi takes 2.0 of functioning from the other |
| 72 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 74; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2; Ravi rejects the other on Ravi–Sofia (rejection +0.10; the tie stays open) |
| 72 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 73 | Ana | CUTOFF | Bruno | — | arrives week 74; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.7, Bruno -0.7; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ana -0.7 |
| 73 | Bruno | CUTOFF | Ana | — | arrives week 74; anxiety Ana +0.7, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 73 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 74; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.1; Ravi holds a position: no functioning changes hands |
| 73 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 74; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi -1.0; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2 |
| 73 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 73 | Ravi | REDUCE_CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 75; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2, Sofia -1.7; Ravi's rejection -0.25; Sofia's rejection -0.25 |
| 73 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 74 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 74 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 74 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 74 | Ana | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 74 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 74 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 75; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +1.9; 7.9 of the sender's anxiety bound into Marta–Ravi |
| 74 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 75; anxiety Marta +1.7 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +4.1 |
| 74 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 75; anxiety Marta +1.7 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +0.2; calmer sender takes some anxiety: Pia +0.0, Ravi -0.0 |
| 74 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 75; anxiety Marta -0.1, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.2, Ravi +0.2; Marta left outside: anxiety Marta +3.1, Ravi -2.1 (of which 1.0 the outsider's own) |
| 74 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.4; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 75 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 75 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 75 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 75 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 75 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 75 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 75 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 76; anxiety Ana +0.3, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 75 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 75 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 76; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.5; Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -0.1 |
| 75 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 75 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 76; anxiety Marta +1.7 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +2.1; calmer sender takes some anxiety: Pia +0.0, Ravi -0.0 |
| 75 | Ravi | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 75 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 77; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 76 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.1 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.5 |
| 76 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.3, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.2 (witness) |
| 76 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 77; anxiety Ana +0.3, Marta +0.5, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 76 | Bruno | WITHHOLD | Ana | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 76 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 77; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.6; 8.7 of the sender's anxiety bound into Marta–Ravi |
| 76 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 77; anxiety Marta +1.8 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.2; 5.9 of the sender's anxiety bound into Nadia–Ravi |
| 76 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 77; anxiety Marta +0.9, Nadia +0.6 (witness), Pia +0.6, Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.2, Pia +0.2; Pia rejects the other on Marta–Pia (rejection +0.25; the tie stays open) |
| 76 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 77; anxiety Marta +1.8 (witness), Nadia +0.6 (witness), Pia +4.6, Ravi +0.4 |
| 76 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 78; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 77 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 78; anxiety Ana +0.3, Marta -0.6, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Marta takes 2.0 of functioning from the other |
| 77 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 77 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 78; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1; 6.2 of the sender's anxiety bound into Marta–Ravi |
| 77 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 78; anxiety Marta +1.8 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +3.2; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 77 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 78; anxiety Marta +1.1, Nadia +0.6 (witness), Pia +0.6, Ravi +1.0 (witness); Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open) |
| 77 | Ravi | STAY-IN-CONTACT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.2; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 77 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 78 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 78 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 78 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 78 | Ana | CUTOFF | Bruno | — | arrives week 79; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 78 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 78 | Marta | TRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +3.3; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 78 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 79; anxiety Marta +1.6 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.8; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 78 | Pia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 78 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 79; anxiety Marta +1.6 (witness), Nadia -0.3, Pia +0.7 (witness), Ravi +0.4; Nadia left outside: anxiety Nadia +2.9, Ravi -1.9 (of which 1.0 the outsider's own) |
| 78 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 80; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 79 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 79 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 79 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 79 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 79 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 80; anxiety Ana +0.3, Marta +0.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 79 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 79 | Marta | STAY-IN-CONTACT | Ravi | Nadia, Pia | arrives week 80; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi -1.2; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1 |
| 79 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 80; anxiety Marta +1.6 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.5 |
| 79 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 80; anxiety Marta +1.6 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +0.5; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -1.3 |
| 79 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.0; anxiety Nadia +1.1 (witness), Pia +1.4 (witness) |
| 79 | Sofia | PROVOKE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 79 | — | (system) | — | — | Pia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 80 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.0 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.5 |
| 80 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.0, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.2 (witness) |
| 80 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 80 | Bruno | CUTOFF | Ana | — | arrives week 81; anxiety Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.25; the tie stays open) |
| 80 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 81; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own); the contact lands with Marta: systems perspective +0.10 |
| 80 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 81; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +1.6; 10.0 of the sender's anxiety bound into Marta–Ravi |
| 80 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 81; anxiety Marta +4.7, Nadia +0.4, Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.0, Nadia +0.0; Nadia takes 2.0 of functioning from the other |
| 80 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 81; anxiety Marta +1.6 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +0.5; calmer sender takes some anxiety: Pia +0.0, Ravi -0.0; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 80 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 81; anxiety Marta +1.6 (witness), Nadia +0.6 (witness), Ravi +0.4; Ravi rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -1.0 |
| 80 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 81 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 81 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 81 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.2 |
| 81 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.9 |
| 81 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.5; anxiety Nadia +0.5 (witness), Ravi +1.0 (witness) |
| 81 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 82; anxiety Marta +1.8 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +3.8 |
| 81 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 82; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +0.6, Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.3, Pia +0.3; Pia rejects the other on Marta–Pia (rejection +0.05; the tie stays open) |
| 81 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 82; anxiety Marta +1.8 (witness), Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Nadia left outside: anxiety Nadia +2.9, Ravi -1.9 (of which 1.0 the outsider's own) |
| 81 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 82 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 82 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 83; anxiety Ana +0.3, Marta +3.9, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 82 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 82 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +3.5; anxiety Nadia +0.5 (witness), Ravi +1.0 (witness) |
| 82 | Nadia | PROVOKE | Ravi | Marta, Pia | arrives week 83; anxiety Marta +1.8 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.8 |
| 82 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Nadia +0.5 (witness), Ravi +1.0 (witness) |
| 82 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 83; anxiety Marta +1.8 (witness), Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Nadia left outside: anxiety Nadia +2.7, Ravi -1.8 (of which 0.9 the outsider's own) |
| 82 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 82 | — | (system) | — | — | Marta–Nadia: rejection has faded; the tie is no longer cut off (ordinary) |
| 83 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 83 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 83 | Marta | PROVOKE | Ravi | Nadia, Pia | arrives week 84; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +2.2 |
| 83 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 84; anxiety Marta +1.8 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.7; 7.6 of the sender's anxiety bound into Nadia–Ravi |
| 83 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Nadia +0.5 (witness), Ravi +1.0 (witness) |
| 83 | Ravi | PREVENT_ALIGNMENT | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.0; anxiety Marta +1.8 (witness), Nadia +0.5 (witness) |
| 83 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 85; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.25; the tie stays open) |
| 84 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.1 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.5 |
| 84 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.3, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.2 (witness) |
| 84 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 84 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.2 |
| 84 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 84 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 85; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.2; Ravi takes 2.0 of functioning from the other |
| 84 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 85; anxiety Marta +1.8 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 84 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Nadia +0.5 (witness), Ravi +1.0 (witness) |
| 84 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.0; anxiety Marta +1.8 (witness), Nadia +0.5 (witness) |
| 84 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +1.0 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 85 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 85 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 85 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 86; anxiety Ana +0.3, Marta +4.8, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9 |
| 85 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.9 |
| 85 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 86; anxiety Marta +0.8, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +3.0; Marta takes 2.0 of functioning from the other |
| 85 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 86; anxiety Marta +1.9 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 85 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 86; anxiety Marta +0.9, Nadia +0.5 (witness), Pia +0.6, Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Pia rejects the other on Marta–Pia (rejection +0.15; the tie stays open) |
| 85 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.2; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 85 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 86 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 86 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 87; anxiety Ana +0.3, Marta +0.5, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9; 3.7 of the sender's anxiety bound into Ana–Marta |
| 86 | Bruno | CUTOFF | Ana | — | arrives week 87; anxiety Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 86 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 87; anxiety Marta +0.7, Nadia +0.6 (witness), Pia +0.1, Ravi +1.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.35; the tie stays open); impingement removed at once, anxiety Marta -0.4 |
| 86 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 87; anxiety Marta +1.8 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; 5.5 of the sender's anxiety bound into Nadia–Ravi |
| 86 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 86 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.2; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 86 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 87 | Ana | CUTOFF | Bruno | — | arrives week 88; anxiety Ana +0.4, Bruno +0.0; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ana -0.3 |
| 87 | Bruno | CUTOFF | Ana | — | arrives week 88; anxiety Ana +0.3, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Bruno -0.0 |
| 87 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 88; anxiety Marta +0.7, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +2.9 |
| 87 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 88; anxiety Marta +0.8, Nadia +0.4, Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.7, Nadia +0.7; Marta takes 2.0 of functioning from the other |
| 87 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 88; anxiety Marta +1.8 (witness), Nadia +0.6 (witness), Pia +0.6; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; Pia rejects the other on Pia–Ravi (rejection +0.20; the tie stays open) |
| 87 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 88; anxiety Marta +1.8 (witness), Pia +0.7 (witness), Ravi +0.4; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -1.4 |
| 87 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 88 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.2 (witness), Nadia +1.4 (witness), Pia +1.7 (witness), Ravi +2.4 |
| 88 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.4, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.2 (witness) |
| 88 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 88 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 89; anxiety Ana +0.3, Marta +4.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +1.1, Marta -1.1 |
| 88 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 88 | Dr Halim | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 89; anxiety Dr Halim +0.1, Marta +1.1 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.3; calmer sender takes some anxiety: Dr Halim +0.6, Ravi -0.6; Dr Halim takes 2.0 of functioning from the other |
| 88 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 89; anxiety Marta +0.7, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.4; Ravi takes 2.0 of functioning from the other |
| 88 | Nadia | PURSUE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.3; anxiety Marta +1.7 (witness), Pia +0.7 (witness) |
| 88 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 89; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.6, Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; Pia rejects the other on Marta–Pia (rejection +0.15; the tie stays open) |
| 88 | Ravi | PREVENT_ALIGNMENT | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.0; anxiety Marta +1.7 (witness), Pia +0.7 (witness) |
| 88 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 88 | — | (system) | — | — | Marta–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 89 | Ana | CUTOFF | Bruno | — | arrives week 90; anxiety Ana +0.4, Bruno +0.1; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 89 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 89 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 90; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 89 | Nadia | DISTANCE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Marta +1.8 (witness), Pia +0.7 (witness) |
| 89 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 90; anxiety Marta +1.8 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +1.1; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1; 10.2 of the sender's anxiety bound into Pia–Ravi |
| 89 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 90; anxiety Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.8, Ravi +0.8; 8.1 of the sender's anxiety bound into Marta–Ravi |
| 89 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 90 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 90 | Bruno | CUTOFF | Ana | — | arrives week 91; anxiety Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 90 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 91; anxiety Marta +0.7, Nadia +0.6 (witness), Pia +0.1, Ravi +1.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.20; the tie stays open) |
| 90 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 91; anxiety Marta +1.8 (witness), Nadia +0.5, Pia +0.7 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 90 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 91; anxiety Marta +1.8 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +0.3; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1; 6.7 of the sender's anxiety bound into Pia–Ravi |
| 90 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 92; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 90 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 91 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 92; anxiety Ana +0.3, Marta +2.6, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +1.0, Marta -1.0 |
| 91 | Bruno | CUTOFF | Ana | — | arrives week 92; anxiety Ana +0.3, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 91 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 92; anxiety Marta +0.7, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.6 |
| 91 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Marta +1.8 (witness), Pia +0.7 (witness) |
| 91 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 92; anxiety Marta +1.8 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +0.4; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; 6.9 of the sender's anxiety bound into Pia–Ravi |
| 91 | Ravi | STAY-IN-CONTACT | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.0; anxiety Marta +1.8 (witness), Pia +0.7 (witness) |
| 91 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 92 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +2.2 (witness), Nadia +1.4 (witness), Pia +1.7 (witness), Ravi +2.4 |
| 92 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +4.4, Nadia +1.4 (witness), Pia +1.7 (witness), Ravi +1.2 (witness) |
| 92 | Ana | CUTOFF | Bruno | — | arrives week 93; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -0.0 |
| 92 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.9 |
| 92 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 93; anxiety Marta +0.7, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +1.9; 14.7 of the sender's anxiety bound into Marta–Ravi |
| 92 | Nadia | CONFLICT | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Marta +1.8 (witness), Pia +0.7 (witness) |
| 92 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 93; anxiety Marta +1.8 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +0.3; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; 5.4 of the sender's anxiety bound into Pia–Ravi |
| 92 | Ravi | PURSUE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.0; anxiety Marta +1.8 (witness), Nadia +0.6 (witness) |
| 92 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 93 | Ana | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 93 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 93 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 94; anxiety Marta +0.7, Nadia +2.9, Pia +0.7 (witness), Ravi +1.0 (witness); Marta takes 2.0 of functioning from the other |
| 93 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 94; anxiety Marta +0.4, Nadia +0.5, Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.0, Nadia +0.0; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Nadia -3.6 |
| 93 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 93 | Ravi | REDUCE_CUTOFF | Nadia | Marta, Pia | arrives week 94; anxiety Marta +1.8 (witness), Nadia -1.0, Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.6, Ravi +0.6; Nadia's rejection -0.25; Ravi's rejection -0.25; bound anxiety released to the third party: Marta +70.4 |
| 93 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 94 | Ana | WITHHOLD | Bruno | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 94 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 94 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 95; anxiety Marta +0.7, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +2.8; Marta takes 2.0 of functioning from the other |
| 94 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 95; anxiety Marta +1.7 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi -0.8; Ravi holds a position: no functioning changes hands |
| 94 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 95; anxiety Marta +1.7 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +0.3; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 5.9 of the sender's anxiety bound into Pia–Ravi |
| 94 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 95; anxiety Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -3.3, Ravi +3.3; 7.3 of the sender's anxiety bound into Marta–Ravi |
| 94 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 95 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 95 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 95 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.1; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 95 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 96; anxiety Marta +1.6 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi -0.8; Ravi holds a position: no functioning changes hands |
| 95 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 95 | Ravi | DETRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +1.6 (witness), Pia +0.7 (witness) |
| 95 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 95 | — | (system) | — | — | Ana–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 96 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.9 (witness), Nadia +1.5 (witness), Pia +1.7 (witness), Ravi +2.5 |
| 96 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.9, Nadia +1.5 (witness), Pia +1.7 (witness), Ravi +1.2 (witness) |
| 96 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 96 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 96 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 97; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Dr Halim +1.2, Marta -1.2; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.50; the tie stays open) |
| 96 | Marta | WITHHOLD | Pia | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 96 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 97; anxiety Marta +1.6 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +2.1; 6.2 of the sender's anxiety bound into Nadia–Ravi |
| 96 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 96 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +1.6 (witness), Nadia +0.6 (witness) |
| 96 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 97 | Ana | WITHHOLD | Bruno | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 97 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 97 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 98; anxiety Marta +0.6, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.6; Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open) |
| 97 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 98; anxiety Marta +1.6 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.3; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 97 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 98; anxiety Marta +1.6 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +2.7; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 97 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 98; anxiety Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -2.0, Ravi +2.0; Marta left outside: anxiety Marta +1.5, Ravi -1.0 (of which 0.5 the outsider's own) |
| 97 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 98 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 98 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 98 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 98 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 98 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 98 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 98 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 98 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 98 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.2; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 98 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 99; anxiety Marta +1.6 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi -1.6 |
| 98 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 98 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 100; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 98 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 100; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 99 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 99 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 99 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.2; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 99 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 100; anxiety Nadia +0.5, Pia +0.7 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -1.5, Nadia +1.5; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 99 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 100; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.6, Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -2.1, Pia +2.1; Pia rejects the other on Marta–Pia (rejection +0.25; the tie stays open) |
| 99 | Ravi | DISTANCE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Nadia +1.2 (witness), Pia +1.4 (witness) |
| 99 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 100 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.9 (witness), Nadia +1.5 (witness), Pia +1.7 (witness), Ravi +2.5 |
| 100 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.9, Nadia +1.5 (witness), Pia +1.7 (witness), Ravi +1.3 (witness) |
| 100 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 101; anxiety Ana +0.3, Marta +0.0, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +1.2, Marta -1.2; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ana -3.0 |
| 100 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 100 | Marta | I-POSITION | Ana | Nadia, Pia, Ravi | arrives week 101; anxiety Ana +2.1, Marta +0.4, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 100 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 101; anxiety Marta +1.6 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +1.1 |
| 100 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 101; anxiety Marta +1.6 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +1.4; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; 8.1 of the sender's anxiety bound into Pia–Ravi |
| 100 | Ravi | DISTANCE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Nadia +1.2 (witness), Pia +1.4 (witness) |
| 100 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 101 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 101 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 101 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 101 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 102; anxiety Ana +0.3, Marta +0.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +1.2, Marta -1.2; 3.5 of the sender's anxiety bound into Ana–Marta |
| 101 | Bruno | CUTOFF | Ana | — | arrives week 102; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 101 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.1; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 101 | Nadia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 101 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 102; anxiety Marta +1.6 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +1.1; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7 |
| 101 | Ravi | PREVENT_ALIGNMENT | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +1.6 (witness), Pia +0.7 (witness) |
| 101 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 102 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.2 |
| 102 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.9 |
| 102 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 103; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness) |
| 102 | Nadia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 102 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.9; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 102 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | — |
| 102 | Sofia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 103 | Ana | PURSUE | Bruno | — | — |
| 103 | Bruno | CUTOFF | Ana | — | — |
| 103 | Marta | STAY-IN-CONTACT | Ravi | Nadia, Pia | — |
| 103 | Nadia | CUTOFF | Ravi | Marta, Pia | — |
| 103 | Pia | CONFLICT | Ravi | Marta, Nadia | — |
| 103 | Ravi | TRIANGLE | Nadia | Marta, Pia | — |
| 103 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | — |
| 103 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |

104 weeks. The invariants were asserted at the end of every week.
