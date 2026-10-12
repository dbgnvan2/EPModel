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
| config hash | `4daaed251db3aa9483a70b29591066f358a77b40b72a687c960d45f90fcc8116` |
| spec revision | 2.0, revision 19 |
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
| 8 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.4, Marta +0.3, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); Ana takes 2.0 of functioning from the other |
| 8 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 9; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.7, Ravi -0.7; 6.9 of the sender's anxiety bound into Nadia–Ravi |
| 8 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 9; anxiety Marta +0.9, Nadia +0.5 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; 8.3 of the sender's anxiety bound into Marta–Pia |
| 8 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | arrives week 10; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +3.0 |
| 8 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 10; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; 3.4 of the sender's anxiety bound into Ravi–Sofia |
| 8 | Bruno | SYMPTOM_ONSET | Ana | — | arrives week 9; symptom onset, social channel; anxiety Ana +2.0 |
| 9 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 9 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 9 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | arrives week 10; anxiety Ana +0.3, Marta +3.1, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 9 | Bruno | CUTOFF | Ana | — | arrives week 10; anxiety Ana +0.7, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 9 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 10; anxiety Marta +0.6, Nadia +1.4, Pia +0.7 (witness), Ravi +1.5 (witness) |
| 9 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 10; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +3.9; calmer sender takes some anxiety: Nadia +0.8, Ravi -0.8 |
| 9 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 10; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +1.4, Ravi -1.4; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -1.6 |
| 9 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 10; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.3, Ravi +0.6; Pia left outside: anxiety Pia +4.9, Ravi -3.3 (of which 1.6 the outsider's own) |
| 9 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 11; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +2.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 10 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 10 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 10 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 10 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 10 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 11; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +3.0, Ravi +1.5 (witness); 12.4 of the sender's anxiety bound into Marta–Pia |
| 10 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 11; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Ravi left outside: anxiety Nadia -0.9, Ravi +1.4 (of which 0.5 the outsider's own) |
| 10 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 11; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +5.6; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Pia takes 2.0 of functioning from the other |
| 10 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 11; anxiety Marta +3.5, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1 |
| 10 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 12; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +4.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1 |
| 10 | Ana | SYMPTOM_ONSET | Bruno, Marta | Nadia, Pia, Ravi | arrives week 11; symptom onset, physical channel; anxiety Bruno +2.8, Marta +1.8, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 11 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 12; anxiety Ana +0.3, Marta +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 11 | Bruno | CUTOFF | Ana | — | arrives week 12; anxiety Ana +0.6, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 11 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 12; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +4.9, Ravi +1.6 (witness) |
| 11 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 12; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.0; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6; 6.4 of the sender's anxiety bound into Nadia–Ravi |
| 11 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 11 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 12; anxiety Marta +5.3, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.7 |
| 11 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 13; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Sofia -1.2 |
| 12 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.8 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +3.9 |
| 12 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.6, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +2.0 (witness) |
| 12 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +0.3, Marta +0.9, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ana -1.7 |
| 12 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 12 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +1.7, Marta +0.4, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 12 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 13; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.9; calmer sender takes some anxiety: Nadia +1.1, Ravi -1.1; 4.8 of the sender's anxiety bound into Nadia–Ravi |
| 12 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 12 | Ravi | PURSUE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +1.5 (witness), Nadia +0.5 (witness) |
| 12 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 14; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; 2.4 of the sender's anxiety bound into Ravi–Sofia |
| 13 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 13 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 13 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 13 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 14; anxiety Marta +0.6, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.9; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; Ravi left outside: anxiety Marta -2.3, Ravi +3.5 (of which 1.2 the outsider's own) |
| 13 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 13 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 14; anxiety Marta +5.8, Nadia +0.5 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6 |
| 13 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 14; anxiety Marta +1.2, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.7; Marta left outside: anxiety Marta +4.6, Ravi -3.1 (of which 1.5 the outsider's own) |
| 13 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 14 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 14 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 14 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 14 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 15; anxiety Ana +0.3, Marta +0.8, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9; Marta takes 2.0 of functioning from the other |
| 14 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 14 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 15; anxiety Marta +0.6, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +1.2; Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -3.4 |
| 14 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 14 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 15; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +1.0; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 14 | Ravi | WITHHOLD | Sofia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 14 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 16; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +4.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 15 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 15 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 15 | Marta | PREVENT_ALIGNMENT | Ravi | Nadia, Pia | arrives week 16; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +1.1; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4 |
| 15 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 15 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 16; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Pia rejects the other on Pia–Ravi (rejection +0.10; the tie stays open) |
| 15 | Ravi | PURSUE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 15 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 17; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +4.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 16 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +3.9 |
| 16 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +2.0 (witness) |
| 16 | Ana | CUTOFF | Bruno | — | arrives week 17; anxiety Ana +0.4, Bruno +0.5; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 16 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 16 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 17; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7 |
| 16 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 17; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +6.1; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4; Marta takes 2.0 of functioning from the other |
| 16 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 17; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +3.9; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6 |
| 16 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 17; anxiety Marta +2.9, Nadia +0.5 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6 |
| 16 | Ravi | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 16 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 18; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.4, Sofia +1.4 |
| 17 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 17 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 17 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 17 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 18; anxiety Ana +0.3, Marta +0.2, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 17 | Bruno | CUTOFF | Ana | — | arrives week 18; anxiety Ana +0.2, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Bruno -0.2 |
| 17 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 18; anxiety Marta +0.5, Nadia +1.8, Pia +0.6 (witness), Ravi +1.7 (witness); 8.8 of the sender's anxiety bound into Marta–Nadia |
| 17 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.7 (witness) |
| 17 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 17 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 18; anxiety Marta +1.3 (witness), Nadia -0.1, Pia +0.6 (witness), Ravi +0.7; Nadia left outside: anxiety Nadia +5.2, Ravi -3.5 (of which 1.7 the outsider's own) |
| 17 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 19; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +2.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2 |
| 18 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 18 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 18 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 19; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +6.3; calmer sender takes some anxiety: Marta +0.9, Ravi -0.9 |
| 18 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.7 (witness) |
| 18 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 19; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +0.5, Ravi +1.7 (witness); calmer sender takes some anxiety: Marta -0.2, Pia +0.2; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -0.0 |
| 18 | Ravi | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 18 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 20; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; Ravi takes 2.0 of functioning from the other |
| 19 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 20; anxiety Ana +0.3, Marta +0.8, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.4, Marta -0.4 |
| 19 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 19 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 20; anxiety Marta +0.5, Nadia +0.6, Pia +0.6 (witness), Ravi +1.8 (witness); calmer sender takes some anxiety: Marta +0.5, Nadia -0.5; 7.9 of the sender's anxiety bound into Marta–Nadia |
| 19 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 20; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.0; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 19 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 20; anxiety Marta +1.3 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.7; calmer sender takes some anxiety: Pia +1.3, Ravi -1.3; 8.6 of the sender's anxiety bound into Pia–Ravi |
| 19 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | arrives week 21; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.4, Sofia -0.6; Sofia takes 2.0 of functioning from the other |
| 19 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 21; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.3, Sofia +1.3; Ravi takes 2.0 of functioning from the other |
| 20 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +4.2 |
| 20 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.1, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +2.1 (witness) |
| 20 | Ana | CUTOFF | Bruno | — | arrives week 21; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 20 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 20 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.6; anxiety Nadia +0.5 (witness), Ravi +1.6 (witness) |
| 20 | Nadia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 20 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 21; anxiety Marta -0.4, Nadia +0.5 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5; Marta takes 2.0 of functioning from the other |
| 20 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +1.3 (witness), Nadia +0.5 (witness) |
| 20 | Sofia | REDUCE_CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 22; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +1.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; Sofia's rejection -0.05 |
| 21 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 21 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 21 | Ana | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 21 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 21 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 22; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +1.2; calmer sender takes some anxiety: Marta +1.0, Ravi -1.0; Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open) |
| 21 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.7 (witness), Ravi +1.7 (witness) |
| 21 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 22; anxiety Marta +4.1, Nadia +0.5 (witness), Pia +0.5, Ravi +1.7 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5; Pia takes 2.0 of functioning from the other |
| 21 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 23; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.4; Ravi rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -6.6 |
| 21 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 23; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +3.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia takes 2.0 of functioning from the other |
| 21 | Sofia | SYMPTOM_ONSET | Ravi | Marta, Nadia, Pia | arrives week 23; symptom onset, physical channel; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +2.0 |
| 22 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 23; anxiety Ana +0.3, Marta +1.5, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; 2.8 of the sender's anxiety bound into Ana–Marta |
| 22 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 22 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.5; anxiety Nadia +0.5 (witness), Ravi +1.7 (witness) |
| 22 | Nadia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 22 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 23; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +0.5, Ravi +1.7 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 22 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 23; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 22 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 23 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 24; anxiety Ana +0.3, Marta -0.9, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 23 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 23 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 24; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +7.9; calmer sender takes some anxiety: Marta +0.6, Ravi -0.6 |
| 23 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.8 (witness) |
| 23 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 24; anxiety Marta +1.3 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Pia rejects the other on Pia–Ravi (rejection +0.30; the tie stays open) |
| 23 | Ravi | TRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.6; anxiety Nadia +1.0 (witness), Pia +1.3 (witness) |
| 23 | Sofia | PROVOKE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 23 | — | (system) | — | — | Ana–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 24 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +4.3 |
| 24 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.1, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +2.1 (witness) |
| 24 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 24 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 24 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 24 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 24 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 24 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 24 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 24 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 25; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 24 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 25; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +3.3; calmer sender takes some anxiety: Marta +1.1, Ravi -1.1 |
| 24 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 25; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +7.1; calmer sender takes some anxiety: Nadia +0.7, Ravi -0.7 |
| 24 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 25; anxiety Marta +2.2, Nadia +0.5 (witness), Pia +0.5, Ravi +1.7 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5; 8.3 of the sender's anxiety bound into Marta–Pia |
| 24 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.4; anxiety Nadia +1.0 (witness), Pia +1.3 (witness) |
| 24 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 25 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 26; anxiety Ana +0.3, Marta +1.3, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 25 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 25 | Marta | WITHHOLD | Dr Halim | — | held back SPLIT: computed, not emitted; attention on the tie rises |
| 25 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 26; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.0; calmer sender takes some anxiety: Nadia +1.0, Ravi -1.0 |
| 25 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 26; anxiety Marta +3.7, Nadia +0.5 (witness), Pia +0.5, Ravi +1.7 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; Pia takes 2.0 of functioning from the other |
| 25 | Ravi | UNDERFUNCTION | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.5; anxiety Marta +1.3 (witness), Nadia +0.5 (witness) |
| 25 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 27; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.4, Sofia +1.4; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 26 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | arrives week 27; anxiety Ana +0.3, Marta +2.4, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 26 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 26 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 27; anxiety Marta +0.5, Nadia +0.3, Pia +0.6 (witness), Ravi +1.7 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3; 8.6 of the sender's anxiety bound into Marta–Nadia |
| 26 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.7 (witness) |
| 26 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 26 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 27; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 26 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 28; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 27 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 27 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 27 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 28; anxiety Ana +0.3, Marta +2.0, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 27 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 27 | Marta | WITHHOLD | Nadia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 27 | Nadia | DETRIANGLE | Ravi | Marta, Pia | arrives week 28; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 27 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 28; anxiety Marta +1.3 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.3; calmer sender takes some anxiety: Pia +1.5, Ravi -1.5; 6.9 of the sender's anxiety bound into Pia–Ravi |
| 27 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 27 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 28 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +4.2 |
| 28 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.2, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +2.1 (witness) |
| 28 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 28 | Ana | CUTOFF | Bruno | — | arrives week 29; anxiety Ana +0.4, Bruno +0.3; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 28 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 28 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 29; anxiety Marta +0.5, Nadia +1.0 (witness), Pia +1.2 (witness), Ravi +4.4; calmer sender takes some anxiety: Marta +1.0, Ravi -1.0 |
| 28 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.6 (witness), Ravi +1.7 (witness) |
| 28 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 29; anxiety Marta +1.3 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.6; calmer sender takes some anxiety: Pia +1.9, Ravi -1.9; 5.7 of the sender's anxiety bound into Pia–Ravi |
| 28 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.5; anxiety Marta +1.3 (witness), Nadia +0.5 (witness) |
| 28 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 29 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 29 | Bruno | CUTOFF | Ana | — | arrives week 30; anxiety Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.35; the tie stays open); impingement removed at once, anxiety Bruno -0.0 |
| 29 | Marta | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 29 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.6 (witness), Ravi +1.8 (witness) |
| 29 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 29 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 30; anxiety Nadia +1.0 (witness), Pia +1.2 (witness), Ravi +0.7; Ravi rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -4.2 |
| 29 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 30 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 31; anxiety Ana +0.3, Marta +3.4, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 30 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 31; anxiety Marta +0.2, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness) |
| 30 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 31; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +4.7; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 30 | Pia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 30 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 30 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 31 | Ana | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 31 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 31 | Marta | TRIANGLE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Marta +2.6; anxiety Nadia +1.0 (witness), Pia +1.2 (witness) |
| 31 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 32; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.4; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Ravi takes 2.0 of functioning from the other |
| 31 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 32; anxiety Marta +1.3 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.3; calmer sender takes some anxiety: Pia +1.5, Ravi -1.5; Pia rejects the other on Pia–Ravi (rejection +0.20; the tie stays open) |
| 31 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 32; anxiety Marta +1.3 (witness), Nadia +1.8, Pia +0.6 (witness), Ravi +0.7; 12.4 of the sender's anxiety bound into Nadia–Ravi |
| 31 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 33; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.10; the tie stays open) |
| 32 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +4.2 |
| 32 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.1, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +2.1 (witness) |
| 32 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 32 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 32 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 32 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 32 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 32 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 32 | Ana | CUTOFF | Bruno | — | arrives week 33; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 32 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 32 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 33; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.50; the tie stays open) |
| 32 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 33; anxiety Marta +0.5, Nadia +0.1, Pia +0.6 (witness), Ravi +1.6 (witness); calmer sender takes some anxiety: Marta +0.0, Nadia -0.0; 9.0 of the sender's anxiety bound into Marta–Nadia |
| 32 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 33; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +2.7 |
| 32 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 33; anxiety Marta +0.1, Nadia +0.5 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; Marta takes 2.0 of functioning from the other |
| 32 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 33; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 32 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 33 | Ana | CUTOFF | Bruno | — | arrives week 34; anxiety Ana +0.4, Bruno +0.8; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 33 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 33 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +0.7; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness) |
| 33 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 34; anxiety Marta +1.2 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.2; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 33 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 34; anxiety Marta +1.2 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.3; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; 6.6 of the sender's anxiety bound into Pia–Ravi |
| 33 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 34; anxiety Dr Halim +0.1, Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 33 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 34 | Ana | I-POSITION | Marta | Nadia, Pia, Ravi | arrives week 35; anxiety Ana +0.3, Marta +3.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 34 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 34 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 35; anxiety Marta +0.3, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -3.6 |
| 34 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 34 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 35; anxiety Marta +3.2, Nadia +0.5 (witness), Pia +0.5, Ravi +1.7 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7 |
| 34 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +1.2 (witness), Pia +0.6 (witness) |
| 34 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 35 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 35 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 35 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 35 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 35 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 35 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 36; anxiety Marta +0.5, Nadia +0.5 (witness), Ravi +1.7 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -3.0 |
| 35 | Nadia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 35 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 36; anxiety Marta +1.2 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.3; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; 6.1 of the sender's anxiety bound into Pia–Ravi |
| 35 | Ravi | WITHHOLD | Sofia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 35 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 36 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +4.0 |
| 36 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.9, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +2.0 (witness) |
| 36 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 36 | Bruno | CUTOFF | Ana | — | arrives week 37; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.35; the tie stays open) |
| 36 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 37; anxiety Marta +0.5, Pia +0.6 (witness), Ravi +1.7 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3; 7.0 of the sender's anxiety bound into Marta–Nadia |
| 36 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 37; anxiety Marta +1.2 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Nadia rejects the other on Nadia–Ravi (rejection +0.45; the tie stays open) |
| 36 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.5 (witness), Ravi +1.7 (witness) |
| 36 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +1.2 (witness), Pia +0.6 (witness) |
| 36 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 37 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 37 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 37 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +0.7; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness) |
| 37 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 37 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.5 (witness), Ravi +1.6 (witness) |
| 37 | Ravi | WITHHOLD | Pia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 37 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 38 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 38 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 38 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 38 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 39; anxiety Marta +0.5, Nadia +0.2, Pia +0.6 (witness), Ravi +1.7 (witness); calmer sender takes some anxiety: Marta +0.6, Nadia -0.6 |
| 38 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.7 (witness) |
| 38 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 39; anxiety Marta +1.2 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Ravi left outside: anxiety Pia -1.1, Ravi +1.6 (of which 0.5 the outsider's own) |
| 38 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +1.2 (witness), Nadia +0.5 (witness) |
| 38 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 39 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 39 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 39 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 39 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 39 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +0.7; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness) |
| 39 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.6 (witness), Ravi +1.6 (witness) |
| 39 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 39 | Ravi | WITHHOLD | Dr Halim | — | held back SPLIT: computed, not emitted; attention on the tie rises |
| 39 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 40 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +4.0 |
| 40 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.9, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +2.0 (witness) |
| 40 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 40 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 40 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 41; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Marta -0.5; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 40 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | arrives week 41; anxiety Marta +0.5, Nadia +0.5 (witness), Pia -0.8, Ravi +1.7 (witness); Pia takes 2.0 of functioning from the other |
| 40 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 41; anxiety Marta +0.1, Nadia +0.4, Pia +0.6 (witness), Ravi +1.7 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Nadia -0.7 |
| 40 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.5 (witness), Ravi +1.7 (witness) |
| 40 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +1.2 (witness), Nadia +0.5 (witness) |
| 40 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 41 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 41 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 41 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 41 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 42; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.2 (witness), Ravi +0.0; calmer sender takes some anxiety: Marta +0.9, Ravi -0.9; Ravi left outside: anxiety Marta -1.2, Ravi +1.9 (of which 0.6 the outsider's own) |
| 41 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 42; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +3.9; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 41 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 42; anxiety Marta +1.3 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +1.3, Ravi -1.3; Pia rejects the other on Pia–Ravi (rejection +0.30; the tie stays open) |
| 41 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 42; anxiety Marta +0.2, Nadia +1.1 (witness), Pia +1.2 (witness), Ravi +0.7; 12.5 of the sender's anxiety bound into Marta–Ravi |
| 41 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 42 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 42 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 42 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 42 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 42 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 42 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 42 | Marta | STAY-IN-CONTACT | Ana | Nadia, Pia, Ravi | arrives week 43; anxiety Ana -1.9, Marta +0.3, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 42 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Pia +0.6 (witness), Ravi +1.7 (witness) |
| 42 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 42 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +3.3; anxiety Marta +1.3 (witness), Pia +0.6 (witness) |
| 42 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 42 | — | (system) | — | — | Marta–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 43 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 43 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 43 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 43 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 44; anxiety Marta +0.3, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); Marta rejects the other on Ana–Marta (rejection +0.45; the tie stays open) |
| 43 | Nadia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 43 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 44; anxiety Marta +1.3 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; 7.5 of the sender's anxiety bound into Pia–Ravi |
| 43 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 44; anxiety Nadia +1.1 (witness), Pia +1.2 (witness), Ravi +0.7; Ravi rejects the other on Marta–Ravi (rejection +0.50; the tie stays open) |
| 43 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness) |
| 44 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +4.0 |
| 44 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +2.0 (witness) |
| 44 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 44 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 44 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | arrives week 45; anxiety Marta +0.2, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.1 (witness) |
| 44 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 45; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Ravi takes 2.0 of functioning from the other |
| 44 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 44 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.9; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 44 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 45 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 45 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 45 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 45 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 45 | Marta | TRIANGLE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Marta +2.5; anxiety Nadia +1.1 (witness), Pia +1.2 (witness) |
| 45 | Nadia | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 45 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 46; anxiety Marta +0.8, Nadia +0.6 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.8, Pia +0.8; 5.7 of the sender's anxiety bound into Marta–Pia |
| 45 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 46; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Ravi +0.6; Ravi rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 45 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 46 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 46 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 46 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 46 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 46 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 46 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 46 | Bruno | CUTOFF | Ana | — | arrives week 47; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 46 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 47; anxiety Marta +0.5, Nadia +0.6 (witness), Ravi +1.6 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 46 | Nadia | CONFLICT | Marta | Pia, Ravi | arrives week 47; anxiety Marta +3.3, Nadia +0.4, Pia +0.6 (witness), Ravi +1.6 (witness) |
| 46 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 47; anxiety Marta -0.9, Nadia +0.6 (witness), Pia +0.5, Ravi +1.6 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9; Marta takes 2.0 of functioning from the other |
| 46 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 47; anxiety Marta +2.9, Nadia +1.1 (witness), Pia +1.2 (witness), Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 46 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 47 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 47 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.9 |
| 47 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 48; anxiety Marta +0.5, Nadia +0.8, Pia +0.6 (witness), Ravi +1.5 (witness); calmer sender takes some anxiety: Marta +0.1, Nadia -0.1; 9.0 of the sender's anxiety bound into Marta–Nadia |
| 47 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 48; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +5.9; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 47 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Marta +1.3 (witness), Nadia +0.6 (witness) |
| 47 | Ravi | WITHHOLD | Sofia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 47 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 48 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.5 |
| 48 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.8 (witness) |
| 48 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 48 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 48 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 49; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Marta -0.5 |
| 48 | Marta | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 48 | Nadia | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 48 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.6 (witness), Ravi +1.5 (witness) |
| 48 | Ravi | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 48 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 49 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 49 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 49 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 49 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 49 | Bruno | WITHHOLD | Ana | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 49 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 50; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.3 (witness); calmer sender takes some anxiety: Marta +0.7, Ravi -0.7; Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -1.3 |
| 49 | Nadia | PURSUE | Marta | Pia, Ravi | arrives week 50; anxiety Marta +0.6, Nadia +0.4, Pia +0.6 (witness), Ravi +1.5 (witness) |
| 49 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 49 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 50; anxiety Marta +1.2, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.6; 9.7 of the sender's anxiety bound into Marta–Ravi |
| 49 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 51; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 49 | — | (system) | — | — | Dr Halim–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 50 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 50 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.9 |
| 50 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 51; anxiety Marta +0.5, Nadia -1.5, Pia +0.6 (witness), Ravi +1.5 (witness); calmer sender takes some anxiety: Marta +0.4, Nadia -0.4; Nadia left outside: anxiety Marta -1.4, Nadia +2.0 (of which 0.7 the outsider's own) |
| 50 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 50 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Marta +1.3 (witness), Nadia +0.6 (witness) |
| 50 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 51; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 50 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 51 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 51 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 51 | Ana | CUTOFF | Bruno | — | arrives week 52; anxiety Ana +0.4, Bruno +0.1; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 51 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 51 | Marta | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 51 | Nadia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 51 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 52; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +0.5; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 51 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.9; anxiety Marta +1.2 (witness), Pia +0.6 (witness) |
| 51 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 51 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |
| 52 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.5 |
| 52 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.8 (witness) |
| 52 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 52 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 52 | Marta | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 52 | Nadia | DISTANCE | Marta | Pia, Ravi | arrives week 53; anxiety Marta +3.2, Nadia +0.4, Pia +0.6 (witness), Ravi +1.5 (witness); 5.6 of the sender's anxiety bound into Marta–Nadia |
| 52 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.5 (witness), Ravi +1.4 (witness) |
| 52 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 54; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.15; the tie stays open) |
| 52 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 52 | — | (system) | — | — | Marta–Nadia: rejection has faded; the tie is no longer cut off (ordinary) |
| 53 | Ana | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 53 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 53 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 54; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.2; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1; 7.0 of the sender's anxiety bound into Marta–Ravi |
| 53 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 54; anxiety Marta +1.2 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.2; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2 |
| 53 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 54; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.0; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia rejects the other on Pia–Ravi (rejection +0.10; the tie stays open) |
| 53 | Ravi | WITHHOLD | Sofia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 53 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 55; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 54 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 54 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 54 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 54 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 55; anxiety Marta +0.5, Nadia -1.6, Pia +0.6 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta +0.2, Nadia -0.2; Nadia left outside: anxiety Marta -1.4, Nadia +2.1 (of which 0.7 the outsider's own) |
| 54 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 55; anxiety Marta +1.2 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +4.2 |
| 54 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.5 (witness), Ravi +1.4 (witness) |
| 54 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 55; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 54 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 55 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 55 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 55 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 55 | Bruno | CUTOFF | Ana | — | arrives week 56; anxiety Ana +0.2, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.30; the tie stays open) |
| 55 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | arrives week 56; anxiety Ana +0.7, Marta +0.3, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); 6.7 of the sender's anxiety bound into Ana–Marta |
| 55 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 56; anxiety Marta +1.2 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.3; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Ravi left outside: anxiety Nadia -0.9, Ravi +1.4 (of which 0.5 the outsider's own) |
| 55 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.5 (witness), Ravi +1.4 (witness) |
| 55 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 56; anxiety Marta +1.2 (witness), Nadia -1.8, Pia +0.6 (witness), Ravi +0.6; Nadia left outside: anxiety Nadia +3.8, Ravi -2.5 (of which 1.3 the outsider's own) |
| 55 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 56 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.5 |
| 56 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.7 (witness) |
| 56 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 56 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 56 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 57; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.4, Marta -0.4; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 56 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 57; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.3 (witness); calmer sender takes some anxiety: Marta +0.5, Ravi -0.5; Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open) |
| 56 | Nadia | PROVOKE | Marta | Pia, Ravi | arrives week 57; anxiety Marta +2.3, Nadia +0.4, Pia +0.6 (witness), Ravi +1.4 (witness) |
| 56 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.5 (witness), Ravi +1.4 (witness) |
| 56 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 57; anxiety Dr Halim +0.0, Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 56 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 58; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.15; the tie stays open) |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 57 | Ana | CUTOFF | Bruno | — | arrives week 58; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -0.6 |
| 57 | Bruno | CUTOFF | Ana | — | arrives week 58; anxiety Ana +0.6, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 57 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 58; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi -0.5; calmer sender takes some anxiety: Marta +0.4, Ravi -0.4; Ravi takes 2.0 of functioning from the other |
| 57 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 58; anxiety Marta +1.2 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +4.9; calmer sender takes some anxiety: Nadia +0.0, Ravi -0.0 |
| 57 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 58; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +4.7; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3 |
| 57 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 57 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 58 | Ana | CUTOFF | Bruno | — | arrives week 59; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.7, Bruno -0.7; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 58 | Bruno | WITHHOLD | Ana | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 58 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 59; anxiety Marta +0.5, Nadia +0.5 (witness), Pia +0.2, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta +0.1, Pia -0.1; 6.8 of the sender's anxiety bound into Marta–Pia |
| 58 | Nadia | DISTANCE | Marta | Pia, Ravi | arrives week 59; anxiety Marta +3.5, Nadia +0.4, Pia +0.6 (witness), Ravi +1.3 (witness); 5.4 of the sender's anxiety bound into Marta–Nadia |
| 58 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.5 (witness), Ravi +1.3 (witness) |
| 58 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 59; anxiety Marta +1.3 (witness), Pia +0.6 (witness), Ravi +0.5; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -5.1 |
| 58 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 59 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 59 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 59 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 59 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 60; anxiety Marta +0.5, Nadia +0.5 (witness), Pia +3.6, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta +0.1, Pia -0.1 |
| 59 | Nadia | PREVENT_ALIGNMENT | Marta | Pia, Ravi | arrives week 60; anxiety Marta -0.7, Nadia +0.4, Pia +0.6 (witness), Ravi +1.3 (witness) |
| 59 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 60; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +4.4; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3 |
| 59 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 59 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 60 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.3 |
| 60 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.2, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 60 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 60 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 61; anxiety Ana +0.3, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 60 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 60 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 61; anxiety Marta +0.2, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness) |
| 60 | Nadia | STAY-IN-CONTACT | Marta | Pia, Ravi | arrives week 61; anxiety Marta -1.3, Nadia +0.4, Pia +0.6 (witness), Ravi +1.3 (witness) |
| 60 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 61; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; 9.8 of the sender's anxiety bound into Pia–Ravi |
| 60 | Ravi | DISTANCE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.7; anxiety Nadia +1.1 (witness), Pia +1.3 (witness) |
| 60 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 62; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 61 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 61 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 61 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 62; anxiety Marta +0.5, Nadia -0.6, Pia +0.6 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta +0.2, Nadia -0.2; Nadia left outside: anxiety Marta -1.9, Nadia +2.8 (of which 0.9 the outsider's own) |
| 61 | Nadia | CONFLICT | Marta | Pia, Ravi | arrives week 62; anxiety Marta +5.6, Nadia +0.4, Pia +0.6 (witness), Ravi +1.4 (witness) |
| 61 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.5 (witness), Ravi +1.3 (witness) |
| 61 | Ravi | TRIANGLE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.7; anxiety Marta +1.3 (witness), Nadia +0.6 (witness) |
| 61 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 62 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 62 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 62 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 62 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 62 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 62 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 62 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 63; anxiety Marta +0.3, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); Marta rejects the other on Ana–Marta (rejection +0.20; the tie stays open) |
| 62 | Nadia | PURSUE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Marta +1.3 (witness), Pia +0.6 (witness) |
| 62 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 63; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +5.0; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9 |
| 62 | Ravi | PROVOKE | Nadia | Marta, Pia | arrives week 63; anxiety Marta +1.3 (witness), Nadia +1.5, Pia +0.6 (witness), Ravi +0.5 |
| 62 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 63 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 63 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 63 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 64; anxiety Marta +0.5, Nadia +0.5 (witness), Pia +0.3, Ravi +1.4 (witness); Marta rejects the other on Marta–Pia (rejection +0.30; the tie stays open) |
| 63 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 64; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.2; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Nadia -1.6 |
| 63 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.5 (witness), Ravi +1.4 (witness) |
| 63 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 63 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 64 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.3 |
| 64 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.1, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 64 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 64 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 64 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 65; anxiety Dr Halim +0.1, Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.6 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Ravi -0.7; the contact lands with Ravi: systems perspective +0.10 |
| 64 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 65; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.3 (witness); calmer sender takes some anxiety: Marta +0.4, Ravi -0.4; Ravi takes 2.0 of functioning from the other |
| 64 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 65; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 64 | Pia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 64 | Ravi | WITHHOLD | Dr Halim | — | held back SPLIT: computed, not emitted; attention on the tie rises |
| 64 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 65 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 65 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 65 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 65 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 65 | Marta | DISTANCE | Nadia | Pia, Ravi | arrives week 66; anxiety Marta +0.5, Nadia +2.1, Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta +0.0, Nadia -0.0; 8.1 of the sender's anxiety bound into Marta–Nadia |
| 65 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 66; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Nadia rejects the other on Nadia–Ravi (rejection +0.10; the tie stays open) |
| 65 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.5 (witness), Ravi +1.3 (witness) |
| 65 | Ravi | STAY-IN-CONTACT | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.6; anxiety Marta +1.4 (witness), Nadia +0.6 (witness) |
| 65 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 66 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.6 (witness) |
| 66 | Bruno | CUTOFF | Ana | — | arrives week 67; anxiety Ana +0.3, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 66 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 67; anxiety Ana +1.2, Marta +0.3, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.6 (witness) |
| 66 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 67; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; 6.4 of the sender's anxiety bound into Nadia–Ravi |
| 66 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 67; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +0.5, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.2, Pia +0.2; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 66 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 67; anxiety Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.5; Ravi rejects the other on Marta–Ravi (rejection +0.50; the tie stays open) |
| 66 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 67 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 67 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 67 | Marta | TRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +0.5 (witness), Ravi +1.3 (witness) |
| 67 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 68; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +2.8; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4 |
| 67 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.5 (witness), Ravi +1.3 (witness) |
| 67 | Ravi | CONFLICT | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.6; anxiety Marta +1.4 (witness), Nadia +0.6 (witness) |
| 67 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 68 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.1 |
| 68 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.5 (witness) |
| 68 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 68 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 68 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 68 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 68 | Marta | DISTANCE | Ravi | Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +1.1 (witness), Pia +1.3 (witness) |
| 68 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 69; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5; 5.7 of the sender's anxiety bound into Nadia–Ravi |
| 68 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 68 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 69; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 68 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 69 | Ana | CUTOFF | Bruno | — | arrives week 70; anxiety Ana +0.4, Bruno +0.1; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 69 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 69 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 69 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 70; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5; 4.2 of the sender's anxiety bound into Nadia–Ravi |
| 69 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 69 | Ravi | WITHHOLD | Sofia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 69 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 70 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.6 (witness) |
| 70 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 70 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 70 | Nadia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 70 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 70 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 71; anxiety Nadia +1.1 (witness), Pia +1.2 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1; 8.8 of the sender's anxiety bound into Marta–Ravi |
| 70 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 72; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6; Sofia rejects the other on Ravi–Sofia (rejection +0.40; the tie stays open) |
| 71 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 71 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 71 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 71 | Marta | CONFLICT | Nadia | Pia, Ravi | arrives week 72; anxiety Marta +0.5, Nadia +2.5, Pia +0.6 (witness), Ravi +1.3 (witness) |
| 71 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 72; anxiety Marta +0.1, Nadia +0.4, Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Nadia -4.2 |
| 71 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 71 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 72; anxiety Marta +0.1, Nadia +1.1 (witness), Pia +1.2 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.4, Ravi +0.4; 8.0 of the sender's anxiety bound into Marta–Ravi |
| 71 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 72 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.1 |
| 72 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.4, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 72 | Ana | WITHHOLD | Bruno | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 72 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 72 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 73; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6 |
| 72 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 72 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 73; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.8; 4.2 of the sender's anxiety bound into Nadia–Ravi |
| 72 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 72 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.5; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 72 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 73 | Ana | CUTOFF | Bruno | — | arrives week 74; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -0.1 |
| 73 | Bruno | CUTOFF | Ana | — | arrives week 74; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.15; the tie stays open); impingement removed at once, anxiety Bruno -0.2 |
| 73 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 74; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.2 (witness), Ravi -1.6; Ravi left outside: anxiety Marta -2.2, Ravi +3.4 (of which 1.1 the outsider's own) |
| 73 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 74; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi -0.1; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 73 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 73 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.5; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 73 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 74 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 74 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 74 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 74 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 74 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 74 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 74 | Ana | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 74 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 74 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Pia +0.6 (witness), Ravi +1.3 (witness) |
| 74 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 75; anxiety Marta -0.6, Nadia +0.4, Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4; Marta takes 2.0 of functioning from the other |
| 74 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 75; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2 |
| 74 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 75; anxiety Marta +0.1, Nadia +1.1 (witness), Pia +1.2 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3; 6.9 of the sender's anxiety bound into Marta–Ravi |
| 74 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 75 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.6 (witness) |
| 75 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 75 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.6; anxiety Pia +0.6 (witness), Ravi +1.3 (witness) |
| 75 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 75 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 76; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +2.6 |
| 75 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.5; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 75 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 76 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +3.1 |
| 76 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.2, Nadia +1.4 (witness), Pia +1.5 (witness), Ravi +1.5 (witness) |
| 76 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.6 (witness) |
| 76 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 76 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.6; anxiety Pia +0.6 (witness), Ravi +1.3 (witness) |
| 76 | Nadia | STAY-IN-CONTACT | Marta | Pia, Ravi | arrives week 77; anxiety Marta -0.3, Nadia +0.5, Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4 |
| 76 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 76 | Ravi | DISTANCE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.5; anxiety Marta +1.3 (witness), Pia +0.6 (witness) |
| 76 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 77 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 77 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 77 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 77 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.6 (witness) |
| 77 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 77 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.6; anxiety Pia +0.6 (witness), Ravi +1.3 (witness) |
| 77 | Nadia | PURSUE | Marta | Pia, Ravi | arrives week 78; anxiety Marta +2.2, Nadia +0.5, Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4 |
| 77 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 77 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.5; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 77 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 78 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 78 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 78 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 79; anxiety Marta +0.5, Nadia +1.2 (witness), Pia +1.2 (witness), Ravi +4.3 |
| 78 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 79; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.1; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Ravi takes 2.0 of functioning from the other |
| 78 | Pia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 78 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.5; anxiety Marta +1.3 (witness), Pia +0.6 (witness) |
| 78 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 80; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 79 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.5; anxiety Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.6 (witness) |
| 79 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 79 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 80; anxiety Marta +0.5, Nadia +1.2 (witness), Pia +1.2 (witness), Ravi +3.4; Marta takes 2.0 of functioning from the other |
| 79 | Nadia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 79 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 80; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +5.6; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 79 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 80; anxiety Nadia +1.2 (witness), Pia +1.2 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1; 11.8 of the sender's anxiety bound into Marta–Ravi |
| 79 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 79 | — | (system) | — | — | Marta–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 79 | — | (system) | — | — | Pia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 80 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.5 (witness), Pia +1.5 (witness), Ravi +2.9 |
| 80 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.2, Nadia +1.5 (witness), Pia +1.5 (witness), Ravi +1.5 (witness) |
| 80 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 80 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 80 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 81; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; the contact lands with Marta: systems perspective +0.10 |
| 80 | Marta | PURSUE | Pia | Nadia, Ravi | arrives week 81; anxiety Marta +0.5, Nadia +0.6 (witness), Pia +1.4, Ravi +1.3 (witness) |
| 80 | Nadia | DISTANCE | Marta | Pia, Ravi | arrives week 81; anxiety Marta +0.8, Nadia +0.5, Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.2, Nadia +0.2; 6.4 of the sender's anxiety bound into Marta–Nadia |
| 80 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 81; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.2; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 80 | Ravi | STAY-IN-CONTACT | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.5; anxiety Marta +1.2 (witness), Pia +0.6 (witness) |
| 80 | Sofia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 81 | Ana | CUTOFF | Bruno | — | arrives week 82; anxiety Ana +0.4, Bruno +0.3; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 81 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 81 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.5; anxiety Pia +0.6 (witness), Ravi +1.3 (witness) |
| 81 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 82; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.0; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 81 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 81 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | arrives week 82; anxiety Marta -0.1, Nadia +1.2 (witness), Pia +1.2 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3; Marta takes 2.0 of functioning from the other |
| 81 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 82 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 82 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 82 | Marta | TRIANGLE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.4; anxiety Pia +0.6 (witness), Ravi +1.3 (witness) |
| 82 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 83; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; 4.5 of the sender's anxiety bound into Nadia–Ravi |
| 82 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 82 | Ravi | PURSUE | Nadia | Marta, Pia | refused by Nadia (rejection +0.05); the sender bears the refusal: anxiety Ravi +2.7; anxiety Marta +1.2 (witness), Pia +0.6 (witness) |
| 82 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 83 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 83 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 83 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 83 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 83 | Marta | STAY-IN-CONTACT | Pia | Nadia, Ravi | arrives week 84; anxiety Marta +0.5, Nadia +0.6 (witness), Pia +0.1, Ravi +1.3 (witness) |
| 83 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 84; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +2.0; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 83 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 83 | Ravi | DISTANCE | Pia | Marta, Nadia | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.7; anxiety Marta +1.2 (witness), Nadia +0.6 (witness) |
| 83 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 85; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia rejects the other on Ravi–Sofia (rejection +0.25; the tie stays open) |
| 84 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.4 (witness), Nadia +1.5 (witness), Pia +1.5 (witness), Ravi +3.3 |
| 84 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.9, Nadia +1.5 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 84 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 84 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 84 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 85; anxiety Marta +0.5, Nadia +1.2 (witness), Pia +1.2 (witness), Ravi +0.7; calmer sender takes some anxiety: Marta +0.0, Ravi -0.0; Ravi left outside: anxiety Marta -1.9, Ravi +2.8 (of which 0.9 the outsider's own) |
| 84 | Nadia | TRIANGLE | Marta | Pia, Ravi | arrives week 85; anxiety Marta -0.3, Nadia +0.5, Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4; Marta left outside: anxiety Marta +1.5, Nadia -1.0 (of which 0.5 the outsider's own) |
| 84 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 84 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 86; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 84 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 85 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 85 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 85 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 85 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 85 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 85 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 85 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.9 |
| 85 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 86; anxiety Marta +0.5, Nadia +0.6 (witness), Pia +1.7, Ravi +1.3 (witness); 7.5 of the sender's anxiety bound into Marta–Pia |
| 85 | Nadia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 85 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 86; anxiety Marta +0.4, Nadia +0.6 (witness), Pia +0.5, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.3, Pia +0.3; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -0.9 |
| 85 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 85 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 86 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 87; anxiety Ana +0.3, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 86 | Bruno | CUTOFF | Ana | — | arrives week 87; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 86 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.4; anxiety Pia +0.6 (witness), Ravi +1.3 (witness) |
| 86 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 87; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +5.2; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4 |
| 86 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 86 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 86 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 87 | Ana | CUTOFF | Bruno | — | arrives week 88; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.7, Bruno -0.7; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ana -0.4 |
| 87 | Bruno | CUTOFF | Ana | — | arrives week 88; anxiety Ana +0.4, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 87 | Marta | STAY-IN-CONTACT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.4; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 87 | Nadia | PURSUE | Marta | Pia, Ravi | arrives week 88; anxiety Marta -0.1, Nadia +0.5, Pia +0.6 (witness), Ravi +1.3 (witness) |
| 87 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 88; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +5.9; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7 |
| 87 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 88; anxiety Marta +2.9, Nadia +1.2 (witness), Pia +1.2 (witness), Ravi +0.5; Ravi takes 2.0 of functioning from the other |
| 87 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 88 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.4 (witness), Nadia +1.5 (witness), Pia +1.5 (witness), Ravi +3.3 |
| 88 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.9, Nadia +1.5 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 88 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 88 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 88 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 88 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 88 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 88 | Dr Halim | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 89; anxiety Dr Halim +0.1, Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Dr Halim +0.8, Ravi -0.8; Dr Halim takes 2.0 of functioning from the other |
| 88 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 89; anxiety Marta +0.5, Nadia +1.2 (witness), Pia +1.2 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta +0.7, Ravi -0.7; Ravi left outside: anxiety Marta -1.8, Ravi +2.6 (of which 0.9 the outsider's own) |
| 88 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 89; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia +0.7, Ravi -0.7; 6.7 of the sender's anxiety bound into Nadia–Ravi |
| 88 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 89; anxiety Marta +0.8, Nadia +0.6 (witness), Pia +0.5, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.3, Pia +0.3; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 88 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.6; anxiety Marta +1.3 (witness), Pia +0.6 (witness) |
| 88 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 89 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 89 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 89 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 89 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 89 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 89 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 89 | Ana | CUTOFF | Bruno | — | arrives week 90; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 89 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 89 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 90; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.9 (witness) |
| 89 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 90; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia +0.7, Ravi -0.7; Nadia rejects the other on Nadia–Ravi (rejection +0.25; the tie stays open) |
| 89 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 90; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; 9.0 of the sender's anxiety bound into Pia–Ravi |
| 89 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 90; anxiety Marta +4.6, Nadia +1.2 (witness), Pia +1.2 (witness), Ravi +0.5 |
| 89 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 90 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 90 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 90 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.5; anxiety Nadia +0.6 (witness), Ravi +1.3 (witness) |
| 90 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 91; anxiety Marta +0.3, Nadia +0.5, Pia +0.6 (witness), Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.4, Nadia +0.4; Marta takes 2.0 of functioning from the other |
| 90 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 91; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; 6.2 of the sender's anxiety bound into Pia–Ravi |
| 90 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.6; anxiety Marta +0.6 (witness), Nadia +0.7 (witness), Pia +0.6 (witness) |
| 90 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.7 (witness), Pia +0.6 (witness) |
| 91 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 92; anxiety Ana +0.3, Marta +0.5, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 91 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 91 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 92; anxiety Marta +0.5, Nadia +1.3 (witness), Pia +1.2 (witness), Ravi +2.9; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1 |
| 91 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 92; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.3; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Ravi left outside: anxiety Nadia -1.1, Ravi +1.7 (of which 0.6 the outsider's own) |
| 91 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 92; anxiety Marta +1.2 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +5.0; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Pia takes 2.0 of functioning from the other |
| 91 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 92; anxiety Marta +1.2 (witness), Nadia +0.7 (witness), Pia +3.0, Ravi +0.5 |
| 91 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.7 (witness), Pia +0.6 (witness) |
| 92 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.4 (witness), Nadia +1.6 (witness), Pia +1.5 (witness), Ravi +3.3 |
| 92 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.9, Nadia +1.6 (witness), Pia +1.5 (witness), Ravi +1.6 (witness) |
| 92 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 92 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 92 | Ana | CUTOFF | Bruno | — | arrives week 93; anxiety Ana +0.4, Bruno +0.1; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 92 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 92 | Marta | STAY-IN-CONTACT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.4; anxiety Nadia +0.6 (witness), Ravi +1.4 (witness) |
| 92 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 93; anxiety Marta +0.3, Nadia +0.5, Pia +0.6 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.3, Nadia +0.3; Marta takes 2.0 of functioning from the other |
| 92 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 93; anxiety Marta +1.2 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; 7.6 of the sender's anxiety bound into Pia–Ravi |
| 92 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 93; anxiety Marta +1.2 (witness), Nadia +0.7 (witness), Pia +2.6, Ravi +0.6; 12.0 of the sender's anxiety bound into Pia–Ravi |
| 92 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.7 (witness), Pia +0.6 (witness) |
| 93 | Ana | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 93 | Bruno | CUTOFF | Ana | — | arrives week 94; anxiety Ana +0.3, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.15; the tie stays open) |
| 93 | Marta | PREVENT_ALIGNMENT | Nadia | Pia, Ravi | arrives week 94; anxiety Marta +0.5, Nadia -1.0, Pia +0.6 (witness), Ravi +1.4 (witness) |
| 93 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 94; anxiety Marta +0.3, Nadia +0.5, Pia +0.6 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.2, Nadia +0.2; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Nadia -0.1 |
| 93 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 94; anxiety Marta +0.3, Nadia +0.7 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -1.1, Pia +1.1; Pia rejects the other on Marta–Pia (rejection +0.10; the tie stays open) |
| 93 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 94; anxiety Marta +1.1 (witness), Nadia +0.3, Pia +0.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.2, Ravi +0.2; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -0.4 |
| 93 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.6 (witness), Nadia +0.7 (witness), Pia +0.6 (witness) |
| 93 | — | (system) | — | — | Pia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 94 | Ana | WITHHOLD | Bruno | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 94 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.9 |
| 94 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 95; anxiety Marta +0.5, Nadia +1.4 (witness), Pia +1.2 (witness), Ravi +4.4; Marta takes 2.0 of functioning from the other |
| 94 | Nadia | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 94 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 95; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +2.2; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; 6.0 of the sender's anxiety bound into Pia–Ravi |
| 94 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 95; anxiety Marta +0.7, Nadia +1.4 (witness), Pia +1.2 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1; Ravi rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -5.9 |
| 94 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 94 | — | (system) | — | — | Pia–Ravi tie now distant |
| 95 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 95 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 95 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 95 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 96; anxiety Marta +0.4, Nadia +0.7 (witness), Pia +0.0, Ravi +1.5 (witness); Marta rejects the other on Marta–Pia (rejection +0.45; the tie stays open); impingement removed at once, anxiety Marta -0.3 |
| 95 | Nadia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 95 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 96; anxiety Marta +0.3, Nadia +0.7 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; Pia rejects the other on Marta–Pia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Pia -0.0 |
| 95 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 96; anxiety Marta +1.8, Nadia +1.4 (witness), Pia +1.2 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.2, Ravi +0.2; 9.2 of the sender's anxiety bound into Marta–Ravi |
| 95 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 96 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +1.7 (witness), Pia +1.4 (witness), Ravi +3.7 |
| 96 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.7, Nadia +1.7 (witness), Pia +1.4 (witness), Ravi +1.8 (witness) |
| 96 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 96 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 96 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 97; anxiety Dr Halim +0.1, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.50; the tie stays open) |
| 96 | Marta | WITHHOLD | Pia | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 96 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 97; anxiety Marta +1.1 (witness), Nadia +0.6, Pia +0.6 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.35; the tie stays open) |
| 96 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 96 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 97; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +3.3, Ravi +0.6 |
| 96 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.5 (witness), Nadia +0.7 (witness), Pia +0.6 (witness) |
| 97 | Ana | WITHHOLD | Bruno | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 97 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.9 |
| 97 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Marta +2.2; anxiety Nadia +1.4 (witness), Pia +1.2 (witness) |
| 97 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 98; anxiety Marta +1.1 (witness), Nadia +0.6, Pia +0.6 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.05; the tie stays open) |
| 97 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 98; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +1.3; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; 6.2 of the sender's anxiety bound into Pia–Ravi |
| 97 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 98; anxiety Marta +1.5, Nadia +1.4 (witness), Pia +1.2 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.4, Ravi +0.4; 5.8 of the sender's anxiety bound into Marta–Ravi |
| 97 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 99; anxiety Marta +0.5 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Sofia rejects the other on Ravi–Sofia (rejection +0.45; the tie stays open) |
| 97 | — | (system) | — | — | Ana–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 98 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 98 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 98 | Marta | PURSUE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.2; anxiety Pia +0.6 (witness), Ravi +1.5 (witness) |
| 98 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 99; anxiety Marta +1.1 (witness), Nadia +0.6, Pia +0.6 (witness), Ravi -1.6; Ravi takes 2.0 of functioning from the other |
| 98 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 98 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 99; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia -0.4, Ravi +0.6; Pia left outside: anxiety Pia +1.4, Ravi -1.0 (of which 0.5 the outsider's own) |
| 98 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 100; anxiety Marta +0.5 (witness), Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 99 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 99 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 99 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 99 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 99 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.2; anxiety Nadia +0.7 (witness), Ravi +1.4 (witness) |
| 99 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 100; anxiety Nadia +0.6, Pia +0.6 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.1, Nadia +0.1; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 99 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 100; anxiety Nadia +0.7 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -1.1, Pia +1.1; Pia rejects the other on Marta–Pia (rejection +0.10; the tie stays open) |
| 99 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 100; anxiety Marta +4.1, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.8, Ravi +0.8 |
| 99 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.5 (witness), Nadia +0.7 (witness), Pia +0.6 (witness) |
| 100 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.3 (witness), Nadia +1.8 (witness), Pia +1.5 (witness), Ravi +3.5 |
| 100 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +2.7, Nadia +1.8 (witness), Pia +1.5 (witness), Ravi +1.7 (witness) |
| 100 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 100 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 100 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 101; anxiety Ana +1.4, Marta +0.3, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 100 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 101; anxiety Marta +1.1 (witness), Nadia +0.6, Pia +0.6 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.15; the tie stays open) |
| 100 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 101; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi +1.0; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; 6.1 of the sender's anxiety bound into Pia–Ravi |
| 100 | Ravi | CUTOFF | Marta | Nadia, Pia | arrives week 101; anxiety Marta +0.7, Nadia +1.5 (witness), Pia +1.2 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.5, Ravi +0.5; Ravi rejects the other on Marta–Ravi (rejection +0.50; the tie stays open) |
| 100 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 101 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 102; anxiety Ana +0.3, Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ana -1.6 |
| 101 | Bruno | CUTOFF | Ana | — | arrives week 102; anxiety Ana +0.2, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 101 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.2; anxiety Pia +0.6 (witness), Ravi +1.4 (witness) |
| 101 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 101 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 102; anxiety Marta +1.1 (witness), Nadia +0.7 (witness), Pia +0.5, Ravi -0.2; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6 |
| 101 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 102; anxiety Marta +1.1 (witness), Nadia +0.2, Pia +0.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.6, Ravi +0.6; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 101 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.3; anxiety Marta +0.5 (witness), Nadia +0.7 (witness), Pia +0.6 (witness) |
| 102 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 102 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 102 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 102 | Marta | DISTANCE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +1.3; anxiety Nadia +0.7 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 102 | Nadia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 102 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.4; anxiety Nadia +0.7 (witness), Ravi +1.4 (witness) |
| 102 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | — |
| 102 | Sofia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 103 | Ana | OVERFUNCTION | Bruno | — | — |
| 103 | Bruno | CUTOFF | Ana | — | — |
| 103 | Marta | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 103 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | — |
| 103 | Pia | DETRIANGLE | Ravi | Marta, Nadia | — |
| 103 | Ravi | CUTOFF | Nadia | Marta, Pia | — |
| 103 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | — |
| 103 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |

104 weeks. The invariants were asserted at the end of every week.
