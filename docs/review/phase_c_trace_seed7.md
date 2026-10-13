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
| config hash | `51afd8949828043298b39b44863153ae909d425c745b100ce5843376dead62d9` |
| spec revision | 2.0, revision 24 |
| family instance | phase_c |
| activation | synchronous_activation v1 (synchronous) |
| visibility | household_conductance_visibility v1 |
| constants frozen | 2026-10-07 |
| constants changed since freeze | channel_protection, constitution_strength, declared_rejection, estimator_breadth, estimator_rate, estimator_sd_weight, estimator_window, lock_in_gain, lock_in_turn, pattern_min_acts, pattern_projection_hold, pattern_window, perspective_channel_gain, perspective_steer_gain, practice_window, refusal_cost_gain, rejection_cutoff_gain, rejection_fade, rejection_refusal_gain, rejection_refusal_threshold, rejection_relief, response_down_gain, response_relaxation_rate, response_up_gain, self_practice_gain, soothing_review_every, soothing_review_penalty, stress_floor_below_basic, stressor_persistence, system_adjust_gain, tie_wear_rate, usual_level_rate, wear_offset_per_act |

The standing load, the decay toward each person's floor, the relaxation of felt contact, symptom accumulation, a symptom's hold on the family's anxiety (lock-in), investment, the reactive detectors and outside-ness run every week for everyone and are not listed line by line. Effects are shown beside the event that caused them.

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
| 2 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 3; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.9 (witness) |
| 2 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Pia +0.5 (witness), Ravi +1.4 (witness) |
| 2 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 3; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.4, Ravi +4.8; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 2.8 of the sender's anxiety bound into Pia–Ravi |
| 2 | Ravi | STAY-IN-CONTACT | Nadia | Marta, Pia | arrives week 3; anxiety Marta +1.3 (witness), Nadia -0.7, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3 |
| 2 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 4; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +2.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 1.2 of the sender's anxiety bound into Ravi–Sofia |
| 3 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 3 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 4; anxiety Ana +0.3, Marta +2.0, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.2, Marta -0.2; 2.0 of the sender's anxiety bound into Ana–Marta |
| 3 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 3 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 4; anxiety Marta +0.5, Nadia +1.2 (witness), Pia +1.1 (witness), Ravi +1.1; calmer sender takes some anxiety: Marta +0.5, Ravi -0.5 |
| 3 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.4; anxiety Pia +0.5 (witness), Ravi +1.4 (witness) |
| 3 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 4; anxiety Marta +3.2, Nadia +0.6 (witness), Pia +0.4, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5 |
| 3 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | arrives week 4; anxiety Marta +1.3 (witness), Nadia +1.2, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.0, Ravi +0.0; Nadia takes 2.0 of functioning from the other |
| 3 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 5; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +3.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia takes 2.0 of functioning from the other |
| 4 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +3.4 |
| 4 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +1.7 (witness) |
| 4 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 4 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 4 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.2 |
| 4 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 4 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 5; anxiety Marta +0.5, Nadia +1.0, Pia +0.5 (witness), Ravi +1.5 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3; Marta rejects the other on Marta–Nadia (rejection +0.45; the tie stays open) |
| 4 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 5; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +5.7; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; Nadia takes 2.0 of functioning from the other |
| 4 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 5; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.4, Ravi +0.7; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Ravi left outside: anxiety Pia -0.9, Ravi +1.4 (of which 0.5 the outsider's own) |
| 4 | Ravi | OVERFUNCTION | Pia | Marta, Nadia | arrives week 5; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +2.8, Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 4 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 6; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +1.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; 1.4 of the sender's anxiety bound into Ravi–Sofia |
| 5 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 5 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 5 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +0.3, Marta +2.4, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana takes 2.0 of functioning from the other |
| 5 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 5 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 6; anxiety Ana +2.1, Marta +0.3, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 5 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 6; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.6; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Ravi left outside: anxiety Nadia -1.1, Ravi +1.7 (of which 0.6 the outsider's own) |
| 5 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 6; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.4; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Ravi left outside: anxiety Pia -1.0, Ravi +1.6 (of which 0.5 the outsider's own) |
| 5 | Ravi | UNDERFUNCTION | Nadia | Marta, Pia | arrives week 6; anxiety Marta +1.3 (witness), Nadia +1.0, Pia +0.6 (witness), Ravi +0.6; Nadia takes 2.0 of functioning from the other |
| 5 | Sofia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 5 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 6, 7; symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.5, Nadia +2.2, Pia +2.3; anxiety Sofia +1.3 |
| 6 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 6 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.1 |
| 6 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 6 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | arrives week 7; anxiety Marta +0.5, Nadia +0.5 (witness), Pia +3.5, Ravi +1.7 (witness); Marta takes 2.0 of functioning from the other |
| 6 | Nadia | PREVENT_ALIGNMENT | Ravi | Marta, Pia | arrives week 7; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.6; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4 |
| 6 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 7; anxiety Marta +1.3 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +1.4; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; Ravi takes 2.0 of functioning from the other |
| 6 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 7; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.2 |
| 6 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 8; anxiety Marta +0.6 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +2.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 7 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 8; anxiety Ana +0.3, Marta +2.5, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana takes 2.0 of functioning from the other |
| 7 | Bruno | CUTOFF | Ana | — | arrives week 8; anxiety Ana +0.6, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 7 | Marta | UNDERFUNCTION | Nadia | Pia, Ravi | arrives week 8; anxiety Marta +0.5, Nadia -0.0, Pia +0.6 (witness), Ravi +1.6 (witness); calmer sender takes some anxiety: Marta +0.1, Nadia -0.1; Nadia takes 2.0 of functioning from the other |
| 7 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 8; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 7 | Pia | UNDERFUNCTION | Ravi | Marta, Nadia | arrives week 8; anxiety Marta +1.3 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Pia–Ravi functioning balance flips; Ravi takes 2.0 of functioning from the other |
| 7 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | arrives week 9; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +1.7 |
| 7 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +2.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 8 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +3.9 |
| 8 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.0, Nadia +1.3 (witness), Pia +1.5 (witness), Ravi +1.9 (witness) |
| 8 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.3, Marta +2.7, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 8 | Bruno | CUTOFF | Ana | — | arrives week 9; anxiety Ana +0.8, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 8 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Dr Halim +0.1, Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Ravi -0.6 |
| 8 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | arrives week 9; anxiety Ana +0.5, Marta +0.3, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); Ana holds a position: no functioning changes hands |
| 8 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 9; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.2; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; 6.1 of the sender's anxiety bound into Nadia–Ravi |
| 8 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | arrives week 9; anxiety Marta +5.3, Nadia +0.5 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -0.7, Pia +0.7; Pia takes 2.0 of functioning from the other |
| 8 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 9; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +2.8, Ravi +0.6 |
| 8 | Sofia | I-POSITION | Ravi | Marta, Nadia, Pia | arrives week 10; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +4.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 8 | Marta | SYMPTOM_ONSET | Ana, Dr Halim, Nadia, Pia, Ravi | — | arrives week 9; symptom onset, physical channel; anxiety Ana +1.5, Dr Halim +0.5, Nadia +2.0, Pia +2.6, Ravi +3.0 |
| 9 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 9 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 9 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 10; anxiety Ana +0.3, Marta +3.6, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana takes 2.0 of functioning from the other |
| 9 | Bruno | CUTOFF | Ana | — | arrives week 10; anxiety Ana +0.7, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 9 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 10; anxiety Marta +0.6, Nadia +1.5, Pia +0.6 (witness), Ravi +1.5 (witness) |
| 9 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 10; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +3.9; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 9 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 10; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +1.1; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; 7.9 of the sender's anxiety bound into Pia–Ravi |
| 9 | Ravi | PURSUE | Pia | Marta, Nadia | arrives week 10; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +3.5, Ravi +0.6 |
| 9 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 11; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +2.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 10 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 10 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 10 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 11; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +3.8, Ravi +1.5 (witness); 10.1 of the sender's anxiety bound into Marta–Pia |
| 10 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 11; anxiety Marta +1.6 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +1.0; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5; Ravi left outside: anxiety Nadia -0.9, Ravi +1.3 (of which 0.4 the outsider's own) |
| 10 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 11; anxiety Marta +1.6 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +1.0; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2; 6.6 of the sender's anxiety bound into Pia–Ravi |
| 10 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 11; anxiety Marta +2.9, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.6 |
| 10 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 12; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +2.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 11 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 11 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 12; anxiety Ana +0.3, Marta +1.0, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 11 | Bruno | CUTOFF | Ana | — | arrives week 12; anxiety Ana +0.6, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 11 | Marta | CONFLICT | Pia | Nadia, Ravi | arrives week 12; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +4.0, Ravi +1.5 (witness) |
| 11 | Nadia | PREVENT_ALIGNMENT | Ravi | Marta, Pia | arrives week 12; anxiety Marta +1.6 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4 |
| 11 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 11 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 12; anxiety Marta +5.2, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.6 |
| 11 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 13; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Sofia -0.1 |
| 12 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.9 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +3.6 |
| 12 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.8, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +1.8 (witness) |
| 12 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +0.3, Marta +0.9, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; 3.5 of the sender's anxiety bound into Ana–Marta |
| 12 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 12 | Marta | PURSUE | Ana | Nadia, Pia, Ravi | arrives week 13; anxiety Ana +1.7, Marta +0.4, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.8 (witness) |
| 12 | Nadia | DETRIANGLE | Ravi | Marta, Pia | arrives week 13; anxiety Marta +1.6 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.9; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 12 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 12 | Ravi | CUTOFF | Pia | Marta, Nadia | arrives week 13; anxiety Marta +1.6 (witness), Nadia +0.5 (witness), Pia +1.2, Ravi +0.6; Ravi rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -0.9 |
| 12 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 14; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +1.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; 2.0 of the sender's anxiety bound into Ravi–Sofia |
| 13 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 13 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 13 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 13 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 14; anxiety Marta +0.6, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.3; calmer sender takes some anxiety: Marta +0.0, Ravi -0.0; Ravi left outside: anxiety Marta -2.6, Ravi +3.9 (of which 1.3 the outsider's own) |
| 13 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.6 (witness), Ravi +1.5 (witness) |
| 13 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 14; anxiety Marta +6.7, Nadia +0.5 (witness), Pia +0.5, Ravi +1.5 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9 |
| 13 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 14; anxiety Marta +4.5, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 13 | Sofia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 14 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 14 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 14 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 14 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 14 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 14 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 15; anxiety Ana +0.3, Marta +0.9, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Marta takes 2.0 of functioning from the other |
| 14 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 14 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 15; anxiety Marta +0.7, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +6.2; calmer sender takes some anxiety: Marta +0.0, Ravi -0.0 |
| 14 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.6 (witness), Ravi +1.4 (witness) |
| 14 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Marta +1.7 (witness), Nadia +0.5 (witness) |
| 14 | Ravi | WITHHOLD | Sofia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 14 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 16; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +3.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 15 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 15 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 15 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 16; anxiety Marta +0.6, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +3.3; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2 |
| 15 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.6 (witness), Ravi +1.4 (witness) |
| 15 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.5; anxiety Marta +1.5 (witness), Nadia +0.5 (witness) |
| 15 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 16; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +1.0, Ravi +0.6; Pia left outside: anxiety Pia +4.6, Ravi -3.1 (of which 1.5 the outsider's own) |
| 15 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 17; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +3.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 16 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.8 (witness), Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +3.4 |
| 16 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.7, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +1.7 (witness) |
| 16 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 16 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 16 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 16 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 17; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.6 (witness), Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6 |
| 16 | Marta | CONFLICT | Ravi | Nadia, Pia | arrives week 17; anxiety Marta +0.6, Nadia +1.0 (witness), Pia +1.3 (witness), Ravi +2.6; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2 |
| 16 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 17; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.6 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 16 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 17; anxiety Marta +0.7, Nadia +0.5 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Marta takes 2.0 of functioning from the other |
| 16 | Ravi | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 16 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 18; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1 |
| 16 | Pia | SYMPTOM_ONSET | Marta, Ravi | Nadia | arrives week 17; symptom onset, social channel; anxiety Marta +3.1, Nadia +0.5 (witness), Ravi +2.9 |
| 17 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 17 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 17 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 18; anxiety Ana +0.3, Marta +0.6, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 17 | Bruno | CUTOFF | Ana | — | arrives week 18; anxiety Ana +0.3, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 17 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 18; anxiety Marta +0.2, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.9 (witness) |
| 17 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.7 (witness), Ravi +1.4 (witness) |
| 17 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 17 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 18; anxiety Marta +1.4 (witness), Nadia +0.1, Pia +0.7 (witness), Ravi +0.6; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -0.5 |
| 17 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 19; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +1.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 18 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 18 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 18 | Marta | PURSUE | Pia | Nadia, Ravi | arrives week 19; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +2.1, Ravi +1.4 (witness) |
| 18 | Nadia | DISTANCE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Pia +0.7 (witness), Ravi +1.4 (witness) |
| 18 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 19; anxiety Marta +2.5, Nadia +0.5 (witness), Pia +0.5, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; 7.0 of the sender's anxiety bound into Marta–Pia |
| 18 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | arrives week 20; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia -0.1 |
| 18 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 19 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 19 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 20; anxiety Ana +0.3, Marta +2.1, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 19 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 19 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 20; anxiety Marta +0.6, Nadia +2.7, Pia +0.7 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta +0.1, Nadia -0.1; Marta takes 2.0 of functioning from the other |
| 19 | Nadia | TRIANGLE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.0; anxiety Marta +1.4 (witness), Pia +0.7 (witness) |
| 19 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 19 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | arrives week 21; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +2.5; Sofia holds a position: no functioning changes hands |
| 19 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 21; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Ravi takes 2.0 of functioning from the other |
| 19 | Bruno | SYMPTOM_ONSET | Ana | — | arrives week 20; symptom onset, social channel; anxiety Ana +2.0 |
| 20 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.2 (witness), Pia +1.6 (witness), Ravi +3.4 |
| 20 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.2 (witness), Pia +1.6 (witness), Ravi +1.7 (witness) |
| 20 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 20 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 20 | Ana | CUTOFF | Bruno | — | arrives week 21; anxiety Ana +0.4, Bruno +0.2; calmer sender takes some anxiety: Ana +1.0, Bruno -1.0; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 20 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 20 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 21; anxiety Marta +0.5, Nadia +0.5 (witness), Pia +1.2, Ravi +1.4 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -0.4 |
| 20 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Marta +1.4 (witness), Pia +0.7 (witness) |
| 20 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 20 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 21; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.3, Ravi +0.6; Pia left outside: anxiety Pia +3.2, Ravi -2.2 (of which 1.1 the outsider's own) |
| 20 | Sofia | REDUCE_CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 22; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia's rejection -0.05 |
| 21 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 21 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 21 | Ana | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 21 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 21 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 22; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +2.3 |
| 21 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.7 (witness), Ravi +1.4 (witness) |
| 21 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 21 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | arrives week 23; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +1.7 |
| 21 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 23; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +1.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 2.8 of the sender's anxiety bound into Ravi–Sofia |
| 22 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 23; anxiety Ana +0.3, Marta +0.5, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; 2.6 of the sender's anxiety bound into Ana–Marta |
| 22 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 22 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 23; anxiety Marta +0.5, Nadia +0.5 (witness), Pia +3.0, Ravi +1.4 (witness); 7.1 of the sender's anxiety bound into Marta–Pia |
| 22 | Nadia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 22 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.5 (witness), Ravi +1.4 (witness) |
| 22 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 23; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 22 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 24; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +2.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 23 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 23 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 24; anxiety Ana +0.3, Marta +0.2, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 23 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 23 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 24; anxiety Marta +0.5, Nadia +0.5 (witness), Pia +2.3, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta +0.0, Pia -0.0; 7.6 of the sender's anxiety bound into Marta–Pia |
| 23 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.7 (witness), Ravi +1.4 (witness) |
| 23 | Pia | PROVOKE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 23 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 24; anxiety Marta +3.7, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.5; Ravi takes 2.0 of functioning from the other |
| 23 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 25; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +2.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia takes 2.0 of functioning from the other |
| 24 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +3.3 |
| 24 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +1.6 (witness) |
| 24 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 24 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 24 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 24 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +2.0 |
| 24 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 24 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 25; anxiety Dr Halim +0.1, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Marta -0.5; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 24 | Marta | OVERFUNCTION | Pia | Nadia, Ravi | arrives week 25; anxiety Marta +0.6, Nadia +0.5 (witness), Pia +4.7, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta +0.1, Pia -0.1; Marta takes 2.0 of functioning from the other |
| 24 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.7 (witness), Ravi +1.3 (witness) |
| 24 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.5 (witness), Ravi +1.3 (witness) |
| 24 | Ravi | DISTANCE | Marta | Nadia, Pia | arrives week 25; anxiety Marta +4.5, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.5; 10.3 of the sender's anxiety bound into Marta–Ravi |
| 24 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 26; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +2.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 25 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 25 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 25 | Ana | PROVOKE | Marta | Nadia, Pia, Ravi | arrives week 26; anxiety Ana +0.3, Marta +2.5, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 25 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 25 | Marta | WITHHOLD | Dr Halim | — | held back DISPLACE: computed, not emitted; attention on the tie rises |
| 25 | Nadia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 25 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Nadia +0.5 (witness), Ravi +1.4 (witness) |
| 25 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 26; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.3, Ravi +0.5; calmer sender takes some anxiety: Pia -0.1, Ravi +0.1; Pia left outside: anxiety Pia +2.4, Ravi -1.6 (of which 0.8 the outsider's own) |
| 25 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 27; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +2.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 26 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 26 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 26 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 27; anxiety Ana +0.3, Marta +3.0, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana takes 2.0 of functioning from the other |
| 26 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 26 | Marta | PURSUE | Nadia | Pia, Ravi | arrives week 27; anxiety Marta +0.5, Nadia +0.7, Pia +0.7 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3 |
| 26 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.7 (witness), Ravi +1.4 (witness) |
| 26 | Pia | TRIANGLE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 26 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 27; anxiety Marta +0.9 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.2 |
| 26 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 28; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +2.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 27 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 27 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 27 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 28; anxiety Ana +0.3, Marta +2.2, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 27 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 27 | Marta | WITHHOLD | Nadia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 27 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 28; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness); Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 27 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 27 | Ravi | DISTANCE | Sofia | Marta, Nadia, Pia | arrives week 29; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +1.2; 8.8 of the sender's anxiety bound into Ravi–Sofia |
| 27 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 29; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Ravi takes 2.0 of functioning from the other |
| 28 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +3.3 |
| 28 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.6 (witness) |
| 28 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 28 | Ana | CUTOFF | Bruno | — | arrives week 29; anxiety Ana +0.4, Bruno +0.3; calmer sender takes some anxiety: Ana +1.1, Bruno -1.1; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 28 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 28 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 29; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +2.2 |
| 28 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.7 (witness), Ravi +1.4 (witness) |
| 28 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 28 | Ravi | TRIANGLE | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.7; anxiety Marta +1.4 (witness), Pia +0.7 (witness) |
| 28 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 30; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6; Ravi takes 2.0 of functioning from the other |
| 29 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 29 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 29 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 29 | Bruno | CUTOFF | Ana | — | arrives week 30; anxiety Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.35; the tie stays open); impingement removed at once, anxiety Bruno -0.1 |
| 29 | Marta | WITHHOLD | Dr Halim | — | held back DISPLACE: computed, not emitted; attention on the tie rises |
| 29 | Nadia | CONFLICT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.1; anxiety Pia +0.7 (witness), Ravi +1.3 (witness) |
| 29 | Pia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 29 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 30; anxiety Marta +1.9, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.4, Ravi +0.4 |
| 29 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 31; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.2, Sofia +0.2; Ravi takes 2.0 of functioning from the other |
| 30 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 30 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 31; anxiety Ana +0.3, Marta +3.8, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +1.0, Marta -1.0 |
| 30 | Bruno | WITHHOLD | Ana | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | arrives week 31; anxiety Marta +0.2, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 30 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 31; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.3; Nadia rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 30 | Pia | TRIANGLE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 30 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | arrives week 32; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.2 |
| 30 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 32; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +2.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3; Sofia takes 2.0 of functioning from the other |
| 31 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 31 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 31 | Ana | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 31 | Bruno | CUTOFF | Ana | — | arrives week 32; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 31 | Marta | PURSUE | Ravi | Nadia, Pia | arrives week 32; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +2.6 |
| 31 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 32; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi -0.5; Ravi takes 2.0 of functioning from the other |
| 31 | Pia | CONFLICT | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Marta +1.4 (witness), Nadia +0.5 (witness) |
| 31 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 32; anxiety Marta +0.8, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.9, Ravi +0.9; Marta left outside: anxiety Marta +2.4, Ravi -1.6 (of which 0.8 the outsider's own) |
| 31 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 33; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +3.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5 |
| 32 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.8 |
| 32 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.4, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.4 (witness) |
| 32 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 32 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 32 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 32 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 32 | Ana | CUTOFF | Bruno | — | arrives week 33; anxiety Ana +0.4, Bruno +0.3; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 32 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 32 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 33; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Dr Halim -0.1 |
| 32 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 33; anxiety Dr Halim +0.1, Marta +0.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 32 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 33; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +3.8 |
| 32 | Pia | PURSUE | Marta | Nadia, Ravi | arrives week 33; anxiety Marta +3.0, Nadia +0.6 (witness), Pia +0.6, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5 |
| 32 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 33; anxiety Marta +6.1, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.8, Ravi +0.8 |
| 32 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 34; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6; 4.0 of the sender's anxiety bound into Ravi–Sofia |
| 33 | Ana | CUTOFF | Bruno | — | arrives week 34; anxiety Ana +0.4, Bruno +0.8; calmer sender takes some anxiety: Ana +0.9, Bruno -0.9; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 33 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 33 | Marta | DISPLACE | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +0.9; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 33 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Pia +0.7 (witness), Ravi +1.2 (witness) |
| 33 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Marta +1.4 (witness), Nadia +0.6 (witness) |
| 33 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Nadia, Pia | arrives week 34; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.1 |
| 33 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 35; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Sofia -0.1 |
| 33 | Nadia | SYMPTOM_ONSET | Marta, Ravi | Pia | arrives week 34; symptom onset, mental channel; anxiety Marta +2.9, Pia +0.7 (witness), Ravi +2.3 |
| 34 | Ana | I-POSITION | Marta | Nadia, Pia, Ravi | arrives week 35; anxiety Ana +0.3, Marta +3.6, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.9, Marta -0.9; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 34 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 34 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 35; anxiety Marta +0.3, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -4.4 |
| 34 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 34 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 35; anxiety Marta +6.2, Nadia +0.6 (witness), Pia +0.6, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5 |
| 34 | Ravi | STAY-IN-CONTACT | Marta | Nadia, Pia | arrives week 35; anxiety Marta +0.8, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.4, Ravi +0.4 |
| 34 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 36; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 35 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 35 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 35 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 35 | Ana | REDUCE_CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 36; anxiety Ana +0.3, Marta +0.5, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Marta's rejection -0.25 |
| 35 | Bruno | CUTOFF | Ana | — | arrives week 36; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.20; the tie stays open); impingement removed at once, anxiety Bruno -0.1 |
| 35 | Marta | DISTANCE | Pia | Nadia, Ravi | arrives week 36; anxiety Marta +0.6, Nadia +0.6 (witness), Pia +0.4, Ravi +1.2 (witness); 10.5 of the sender's anxiety bound into Marta–Pia |
| 35 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 35 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Marta +1.4 (witness), Nadia +0.6 (witness) |
| 35 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 35 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 37; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 36 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.8 |
| 36 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.4 (witness) |
| 36 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 37; anxiety Ana +0.3, Marta +0.5, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 36 | Bruno | CUTOFF | Ana | — | arrives week 37; anxiety Ana +0.5, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 36 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 37; anxiety Marta +0.6, Nadia -1.1, Pia +0.7 (witness), Ravi +1.2 (witness); calmer sender takes some anxiety: Marta +0.2, Nadia -0.2; Nadia left outside: anxiety Marta -1.7, Nadia +2.6 (of which 0.9 the outsider's own) |
| 36 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 37; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +4.2 |
| 36 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 37; anxiety Marta +1.1, Nadia +0.6 (witness), Pia +0.6, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.1, Pia +0.1; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -0.0 |
| 36 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 37; anxiety Marta -0.5, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.5; Marta left outside: anxiety Marta +3.6, Ravi -2.4 (of which 1.2 the outsider's own) |
| 36 | Sofia | PROVOKE | Ravi | Marta, Nadia, Pia | arrives week 38; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +2.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 37 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 37 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 37 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 37 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 37 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 37 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 37 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 37 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 37 | Marta | PROVOKE | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +1.7; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 37 | Nadia | STAY-IN-CONTACT | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Pia +0.7 (witness), Ravi +1.2 (witness) |
| 37 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 38; anxiety Marta +4.1, Nadia +0.6 (witness), Pia +0.6, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4 |
| 37 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 37 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 38 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 38 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 38 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.9; anxiety Nadia +0.6 (witness), Ravi +1.2 (witness) |
| 38 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Pia +0.7 (witness), Ravi +1.2 (witness) |
| 38 | Pia | PURSUE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Marta +1.4 (witness), Nadia +0.6 (witness) |
| 38 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 39; anxiety Marta +1.4 (witness), Nadia +0.0, Pia +0.7 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia -0.2, Ravi +0.2; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -2.5 |
| 38 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 40; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; 2.1 of the sender's anxiety bound into Ravi–Sofia |
| 39 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 40; anxiety Ana +0.3, Marta +3.3, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 39 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 39 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 40; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8 (witness) |
| 39 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 40; anxiety Marta +0.2, Nadia +0.4, Pia +0.7 (witness), Ravi +1.2 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 39 | Pia | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 39 | Ravi | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 39 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 41; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 40 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.8 |
| 40 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.4 (witness) |
| 40 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 40 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 40 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 40 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 40 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 40 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 41; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Marta left outside: anxiety Dr Halim -0.0, Marta +0.1 (of which 0.0 the outsider's own) |
| 40 | Marta | OVERFUNCTION | Ravi | Nadia, Pia | arrives week 41; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +3.6; Marta takes 2.0 of functioning from the other |
| 40 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Pia +0.7 (witness), Ravi +1.2 (witness) |
| 40 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 41; anxiety Marta +5.0, Nadia +0.6 (witness), Pia +0.6, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6 |
| 40 | Ravi | CUTOFF | Nadia | Marta, Pia | arrives week 41; anxiety Marta +1.4 (witness), Nadia +0.2, Pia +0.7 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Ravi rejects the other on Nadia–Ravi (rejection +0.50; the tie stays open) |
| 40 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 42; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Ravi takes 2.0 of functioning from the other |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 41 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 41 | Ana | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 41 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 41 | Marta | STAY-IN-CONTACT | Ravi | Nadia, Pia | arrives week 42; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.7 |
| 41 | Nadia | PURSUE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Marta +1.4 (witness), Pia +0.7 (witness) |
| 41 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 42; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +0.4; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 41 | Ravi | PURSUE | Marta | Nadia, Pia | arrives week 42; anxiety Marta +2.1, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1 |
| 41 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 43; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.20; the tie stays open) |
| 42 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 42 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 42 | Marta | OVERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +1.6; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 42 | Nadia | TRIANGLE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Pia +0.7 (witness), Ravi +1.2 (witness) |
| 42 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 42 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.3; anxiety Marta +1.4 (witness), Pia +0.7 (witness) |
| 42 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 44; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi -0.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 43 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 43 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 43 | Ana | WITHHOLD | Bruno | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 43 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 43 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 44; anxiety Marta +0.3, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -0.4 |
| 43 | Nadia | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 43 | Pia | DISTANCE | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +2.8; anxiety Marta +1.4 (witness), Nadia +0.6 (witness) |
| 43 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 44; anxiety Marta +4.5, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.2, Ravi +0.2; Marta holds a position: no functioning changes hands |
| 43 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 45; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.10; the tie stays open) |
| 44 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.8 |
| 44 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.4 (witness) |
| 44 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 44 | Ana | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 44 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 44 | Marta | UNDERFUNCTION | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +1.6; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 44 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.2; anxiety Marta +1.4 (witness), Pia +0.7 (witness) |
| 44 | Pia | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 44 | Ravi | PROVOKE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 44 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 46; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +2.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 45 | Ana | WITHHOLD | Bruno | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 45 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 45 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 46; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.5; Ravi takes 2.0 of functioning from the other |
| 45 | Nadia | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 45 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 46; anxiety Marta +0.7, Nadia +0.6 (witness), Pia +0.6, Ravi +1.2 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 45 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | arrives week 46; anxiety Marta +1.4 (witness), Nadia +3.0, Pia +0.7 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia -0.4, Ravi +0.4; Ravi takes 2.0 of functioning from the other |
| 45 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 47; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 45 | — | (system) | — | — | Dr Halim–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 46 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 46 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 46 | Bruno | CUTOFF | Ana | — | arrives week 47; anxiety Ana +0.1, Bruno +0.6; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 46 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.9; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 46 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 47; anxiety Marta +0.0, Nadia +0.5, Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.0, Nadia +0.0; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 46 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 47; anxiety Marta +0.3, Nadia +0.6 (witness), Pia +0.6, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Marta takes 2.0 of functioning from the other |
| 46 | Ravi | PREVENT_ALIGNMENT | Marta | Nadia, Pia | arrives week 47; anxiety Marta +0.7, Nadia +1.2 (witness), Pia +1.4 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.4, Ravi +0.4 |
| 46 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 48; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 47 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 47 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 47 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 48; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness) |
| 47 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.3; anxiety Pia +0.7 (witness), Ravi +1.1 (witness) |
| 47 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Pia +3.0; anxiety Marta +1.4 (witness), Nadia +0.6 (witness) |
| 47 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 47 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 48 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.4 (witness), Pia +1.8 (witness), Ravi +2.6 |
| 48 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.4 (witness), Pia +1.8 (witness), Ravi +1.3 (witness) |
| 48 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 49; anxiety Ana +0.3, Marta +1.0, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; 2.7 of the sender's anxiety bound into Ana–Marta |
| 48 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.8 |
| 48 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 49; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6 |
| 48 | Marta | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 48 | Nadia | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 48 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 49; anxiety Marta +0.7, Nadia +0.6 (witness), Pia +0.6, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4; Pia rejects the other on Marta–Pia (rejection +0.40; the tie stays open) |
| 48 | Ravi | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 48 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 49 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 49 | Bruno | WITHHOLD | Ana | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 49 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 49 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.3; anxiety Pia +0.7 (witness), Ravi +1.1 (witness) |
| 49 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 49 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 50; anxiety Marta +4.5, Nadia +1.2 (witness), Pia +1.5 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3; Marta holds a position: no functioning changes hands |
| 49 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 51; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +2.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 50 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | arrives week 51; anxiety Ana +0.3, Marta +0.3, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; 2.5 of the sender's anxiety bound into Ana–Marta |
| 50 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.8 |
| 50 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 50 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 50 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 51; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +2.4; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; Pia takes 2.0 of functioning from the other |
| 50 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 51; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.1 |
| 50 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 52; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +2.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia takes 2.0 of functioning from the other |
| 51 | Ana | CUTOFF | Bruno | — | arrives week 52; anxiety Ana +0.4, Bruno +0.0; calmer sender takes some anxiety: Ana +1.1, Bruno -1.1; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 51 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 51 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 52; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi -0.2; Ravi holds a position: no functioning changes hands |
| 51 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 51 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 52; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +1.4; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3; 8.5 of the sender's anxiety bound into Pia–Ravi |
| 51 | Ravi | STAY-IN-CONTACT | Marta | Nadia, Pia | arrives week 52; anxiety Marta +0.4, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.2, Ravi +0.2 |
| 51 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 53; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Ravi takes 2.0 of functioning from the other |
| 51 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |
| 52 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.4 (witness), Pia +1.7 (witness), Ravi +2.7 |
| 52 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.4 (witness), Pia +1.7 (witness), Ravi +1.3 (witness) |
| 52 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 53; anxiety Ana +0.3, Marta +0.8, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 52 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 52 | Marta | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 52 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.3; anxiety Pia +0.7 (witness), Ravi +1.1 (witness) |
| 52 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 53; anxiety Marta +1.1, Nadia +0.6 (witness), Pia +0.5, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9; 7.1 of the sender's anxiety bound into Marta–Pia |
| 52 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 53; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +3.6, Ravi +0.5 |
| 52 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 54; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 53 | Ana | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 53 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 53 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 54; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.4 (witness), Ravi +0.1; Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -1.3 |
| 53 | Nadia | TRIANGLE | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.3; anxiety Marta +1.3 (witness), Pia +0.7 (witness) |
| 53 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 54; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; 5.5 of the sender's anxiety bound into Pia–Ravi |
| 53 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 53 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 55; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +2.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 54 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 54 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 54 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 55; anxiety Ana +0.3, Marta +0.3, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 54 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 54 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 55; anxiety Marta +0.5, Nadia -0.8, Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3; Nadia left outside: anxiety Marta -1.5, Nadia +2.2 (of which 0.7 the outsider's own) |
| 54 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.3; anxiety Pia +0.7 (witness), Ravi +1.1 (witness) |
| 54 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 55; anxiety Marta -0.4, Nadia +0.6 (witness), Pia +0.5, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -1.0, Pia +1.0; Marta takes 2.0 of functioning from the other |
| 54 | Ravi | DETRIANGLE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.2; anxiety Nadia +1.1 (witness), Pia +1.4 (witness) |
| 54 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 56; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; 2.3 of the sender's anxiety bound into Ravi–Sofia |
| 54 | — | (system) | — | — | Ana–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 55 | Ana | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 56; anxiety Ana +0.3, Marta +1.9, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana takes 2.0 of functioning from the other |
| 55 | Bruno | CUTOFF | Ana | — | arrives week 56; anxiety Ana +0.2, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.30; the tie stays open) |
| 55 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 56; anxiety Marta +0.3, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -3.0 |
| 55 | Nadia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 55 | Pia | DISTANCE | Marta | Nadia, Ravi | arrives week 56; anxiety Marta +1.1, Nadia +0.6 (witness), Pia +0.6, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; 6.7 of the sender's anxiety bound into Marta–Pia |
| 55 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 56; anxiety Marta +1.3 (witness), Nadia +0.7, Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; 9.0 of the sender's anxiety bound into Nadia–Ravi |
| 55 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 56 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.5 (witness), Nadia +1.4 (witness), Pia +1.8 (witness), Ravi +2.7 |
| 56 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.1, Nadia +1.4 (witness), Pia +1.8 (witness), Ravi +1.3 (witness) |
| 56 | Ana | DISTANCE | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.3; anxiety Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness) |
| 56 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 56 | Dr Halim | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 57; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.3, Marta -0.3; Marta left outside: anxiety Dr Halim -0.3, Marta +0.5 (of which 0.2 the outsider's own) |
| 56 | Marta | UNDERFUNCTION | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 56 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.3; anxiety Pia +0.7 (witness), Ravi +1.1 (witness) |
| 56 | Pia | CONFLICT | Marta | Nadia, Ravi | arrives week 57; anxiety Marta +4.4, Nadia +0.6 (witness), Pia +0.6, Ravi +1.1 (witness) |
| 56 | Ravi | SPLIT | Dr Halim | Marta, Nadia, Pia | arrives week 57; anxiety Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.1 |
| 56 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 58; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +2.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 56 | — | (system) | — | — | Pia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 57 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 57 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 57 | Ana | CUTOFF | Bruno | — | arrives week 58; anxiety Ana +0.3; calmer sender takes some anxiety: Ana +0.4, Bruno -0.4; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -0.5 |
| 57 | Bruno | CUTOFF | Ana | — | arrives week 58; anxiety Ana +0.5, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 57 | Marta | PROVOKE | Ravi | Nadia, Pia | arrives week 58; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.5 (witness), Ravi +2.6; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2 |
| 57 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.3; anxiety Marta +1.3 (witness), Pia +0.7 (witness) |
| 57 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 58; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +0.4; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -0.5 |
| 57 | Ravi | WITHHOLD | Nadia | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 57 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 59; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Ravi takes 2.0 of functioning from the other |
| 58 | Ana | CUTOFF | Bruno | — | arrives week 59; anxiety Ana +0.3; calmer sender takes some anxiety: Ana +0.4, Bruno -0.4; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 58 | Bruno | WITHHOLD | Ana | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 58 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 58 | Nadia | PURSUE | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.3; anxiety Pia +0.7 (witness), Ravi +1.1 (witness) |
| 58 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 59; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +4.2; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3 |
| 58 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 59; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.9, Ravi +0.9; Nadia left outside: anxiety Nadia +2.7, Ravi -1.8 (of which 0.9 the outsider's own) |
| 58 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 59 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 59 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 59 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.7 |
| 59 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 59 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 59 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Nadia +2.3; anxiety Pia +0.7 (witness), Ravi +1.0 (witness) |
| 59 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 60; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +1.0; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; 9.1 of the sender's anxiety bound into Pia–Ravi |
| 59 | Ravi | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 59 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 61; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 60 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.4 (witness), Pia +1.8 (witness), Ravi +2.6 |
| 60 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.4 (witness), Pia +1.8 (witness), Ravi +1.3 (witness) |
| 60 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 60 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 61; anxiety Ana +0.3, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 60 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 60 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 61; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness) |
| 60 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 61; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +3.5 |
| 60 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 61; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +0.5; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; 7.1 of the sender's anxiety bound into Pia–Ravi |
| 60 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 61; anxiety Marta -0.5, Nadia +1.1 (witness), Pia +1.5 (witness), Ravi +0.4; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1; Marta left outside: anxiety Marta +3.1, Ravi -2.1 (of which 1.0 the outsider's own) |
| 60 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 62; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 61 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 61 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 61 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 61 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 61 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 61 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.7 |
| 61 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 61 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 62; anxiety Marta +0.5, Nadia +0.6 (witness), Ravi +1.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -0.5 |
| 61 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 62; anxiety Marta +0.3, Nadia +0.5, Pia +0.7 (witness), Ravi +1.0 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 61 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 62; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +3.3; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 61 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 61 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 63; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 62 | Ana | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 62 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 62 | Marta | CUTOFF | Ana | Nadia, Pia, Ravi | arrives week 63; anxiety Marta +0.3, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.5 (witness); Marta rejects the other on Ana–Marta (rejection +0.50; the tie stays open) |
| 62 | Nadia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 62 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 63; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.6, Ravi +1.2; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; 4.8 of the sender's anxiety bound into Pia–Ravi |
| 62 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 63; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia -2.0, Ravi +0.4; Pia left outside: anxiety Pia +3.2, Ravi -2.1 (of which 1.1 the outsider's own) |
| 62 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 64; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +1.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 63 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 63 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.7 |
| 63 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 63 | Marta | PURSUE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 63 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 64; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +0.6 |
| 63 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.9; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 63 | Ravi | OVERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 63 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 65; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.2, Sofia +1.2; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 64 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.4 (witness), Pia +1.8 (witness), Ravi +2.6 |
| 64 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.4 (witness), Pia +1.8 (witness), Ravi +1.3 (witness) |
| 64 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 64 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 64 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.7 |
| 64 | Bruno | CUTOFF | Ana | — | arrives week 65; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 64 | Dr Halim | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 65; anxiety Dr Halim +0.1, Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.4; calmer sender takes some anxiety: Dr Halim +0.7, Ravi -0.7; Dr Halim takes 2.0 of functioning from the other |
| 64 | Marta | UNDERFUNCTION | Ravi | Nadia, Pia | arrives week 65; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.5 (witness), Ravi +0.0; Ravi holds a position: no functioning changes hands |
| 64 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 65; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.7 (witness), Ravi +3.5; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Nadia takes 2.0 of functioning from the other |
| 64 | Pia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 64 | Ravi | WITHHOLD | Marta | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 64 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 66; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +1.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 65 | Ana | CUTOFF | Bruno | — | arrives week 66; anxiety Ana +0.3; calmer sender takes some anxiety: Ana +0.5, Bruno -0.5; Ana rejects the other on Ana–Bruno (rejection +0.15; the tie stays open); impingement removed at once, anxiety Ana -0.3 |
| 65 | Bruno | CUTOFF | Ana | — | arrives week 66; anxiety Ana +0.3, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 65 | Marta | TRIANGLE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Pia +0.7 (witness), Ravi +1.1 (witness) |
| 65 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 66; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia +0.0, Ravi -0.0; Ravi left outside: anxiety Nadia -1.3, Ravi +1.9 (of which 0.6 the outsider's own) |
| 65 | Pia | CONFLICT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.9; anxiety Nadia +0.5 (witness), Ravi +1.1 (witness) |
| 65 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 67; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open); impingement removed at once, anxiety Ravi -2.1 |
| 65 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 66 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 66 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 66 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 66 | Ana | WITHHOLD | Bruno | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 66 | Bruno | CUTOFF | Ana | — | arrives week 67; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 66 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Marta +1.6; anxiety Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.6 (witness) |
| 66 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 67; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; 6.9 of the sender's anxiety bound into Nadia–Ravi |
| 66 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.9; anxiety Nadia +0.5 (witness), Ravi +1.1 (witness) |
| 66 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | arrives week 67; anxiety Marta +4.3, Nadia +1.1 (witness), Pia +1.5 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.3, Ravi +0.3; Ravi takes 2.0 of functioning from the other |
| 66 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 66 | — | (system) | — | — | Marta–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 66 | — | (system) | — | — | Nadia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 67 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.7 |
| 67 | Bruno | CUTOFF | Ana | — | arrives week 68; anxiety Ana +0.1, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 67 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 68; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.5 (witness), Ravi +0.1; Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Marta -3.0 |
| 67 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 68; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.6; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2 |
| 67 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 68; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.6, Ravi +4.3; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 67 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 68; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +4.6, Ravi +0.4 |
| 67 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 69; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia rejects the other on Ravi–Sofia (rejection +0.20; the tie stays open) |
| 67 | Marta | SYMPTOM_ONSET | Ana, Dr Halim, Nadia, Pia, Ravi | — | arrives week 68; symptom onset, physical channel; anxiety Ana +1.3, Dr Halim +0.5, Nadia +2.2, Pia +2.9, Ravi +2.2 |
| 68 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.8 (witness), Ravi +2.7 |
| 68 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.3 (witness), Pia +1.8 (witness), Ravi +1.3 (witness) |
| 68 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.7 |
| 68 | Bruno | CUTOFF | Ana | — | arrives week 69; anxiety Ana +0.3, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 68 | Marta | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 68 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 69; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Ravi left outside: anxiety Nadia -1.2, Ravi +1.7 (of which 0.6 the outsider's own) |
| 68 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 69; anxiety Marta +0.5, Nadia +0.5 (witness), Pia +0.6, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 68 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.2; anxiety Nadia +1.1 (witness), Pia +1.5 (witness) |
| 68 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 68 | — | (system) | — | — | Pia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 69 | Ana | CUTOFF | Bruno | — | arrives week 70; anxiety Ana +0.3, Bruno +0.1; calmer sender takes some anxiety: Ana +0.6, Bruno -0.6; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Ana -0.0 |
| 69 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 69 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.05); the sender bears the refusal: anxiety Marta +2.8; anxiety Nadia +0.5 (witness), Ravi +1.1 (witness) |
| 69 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 70; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +3.0; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 69 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 70; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.6, Ravi +3.4; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4 |
| 69 | Ravi | WITHHOLD | Sofia | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 69 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 70 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ana +1.3; anxiety Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.5 (witness) |
| 70 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 70 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 71; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.5 (witness), Ravi -0.1; Ravi left outside: anxiety Marta -2.0, Ravi +3.0 (of which 1.0 the outsider's own) |
| 70 | Nadia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 70 | Pia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 70 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.2; anxiety Nadia +1.1 (witness), Pia +1.5 (witness) |
| 70 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 71 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 71 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 71 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 71 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 71 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.7 |
| 71 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 71 | Marta | TRIANGLE | Nadia | Pia, Ravi | arrives week 72; anxiety Marta +0.6, Nadia -1.6, Pia +0.7 (witness), Ravi +1.1 (witness); Nadia left outside: anxiety Marta -2.2, Nadia +3.3 (of which 1.1 the outsider's own) |
| 71 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 72; anxiety Marta +0.2, Nadia +0.4, Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.1, Nadia +0.1; Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 71 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 72; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.6, Ravi +3.2; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6 |
| 71 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.2; anxiety Nadia +1.1 (witness), Pia +1.5 (witness) |
| 71 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 73; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Sofia rejects the other on Ravi–Sofia (rejection +0.20; the tie stays open) |
| 72 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.8 (witness), Ravi +2.7 |
| 72 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.3 (witness), Pia +1.8 (witness), Ravi +1.3 (witness) |
| 72 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 72 | Ana | WITHHOLD | Bruno | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 72 | Bruno | CUTOFF | Ana | — | arrives week 73; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 72 | Dr Halim | OVERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 73; anxiety Dr Halim +0.1, Marta +0.8, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Dr Halim takes 2.0 of functioning from the other |
| 72 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 73; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.5 (witness), Ravi -0.1; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1; Ravi left outside: anxiety Marta -1.9, Ravi +2.8 (of which 0.9 the outsider's own) |
| 72 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 73; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +4.3; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 72 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 73; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.6, Ravi +1.8; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 72 | Ravi | PROVOKE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 72 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 73 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 73 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.7 |
| 73 | Bruno | CUTOFF | Ana | — | arrives week 74; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 73 | Marta | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 73 | Nadia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 73 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 74; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.6, Ravi +0.9; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open); impingement removed at once, anxiety Pia -0.2 |
| 73 | Ravi | STAY-IN-CONTACT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 73 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 74 | Ana | WITHHOLD | Bruno | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 74 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 74 | Marta | DISTANCE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.0; anxiety Nadia +0.5 (witness), Ravi +1.1 (witness) |
| 74 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 75; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +3.0; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 74 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 75; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.6, Ravi +2.5; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia takes 2.0 of functioning from the other |
| 74 | Ravi | CONFLICT | Nadia | Marta, Pia | arrives week 75; anxiety Marta +1.5 (witness), Nadia +2.9, Pia +0.7 (witness), Ravi +0.4 |
| 74 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 75 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.7 |
| 75 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 75 | Marta | TRIANGLE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +3.0; anxiety Pia +0.7 (witness), Ravi +1.1 (witness) |
| 75 | Nadia | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 75 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 76; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +2.1; calmer sender takes some anxiety: Pia +1.4, Ravi -1.4; Pia takes 2.0 of functioning from the other |
| 75 | Ravi | WITHHOLD | Marta | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 75 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 77; anxiety Marta +0.8 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -1.1, Sofia +1.1; Sofia rejects the other on Ravi–Sofia (rejection +0.10; the tie stays open) |
| 76 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.8 (witness), Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +2.8 |
| 76 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.7, Nadia +1.3 (witness), Pia +1.7 (witness), Ravi +1.4 (witness) |
| 76 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 76 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 76 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 76 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 77; anxiety Ana +0.3, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +1.3, Marta -1.3; Marta takes 2.0 of functioning from the other |
| 76 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 76 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 77; anxiety Marta +0.6, Nadia +0.5 (witness), Ravi +1.2 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 76 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 77; anxiety Marta +1.5 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.7; calmer sender takes some anxiety: Nadia +0.6, Ravi -0.6; 6.4 of the sender's anxiety bound into Nadia–Ravi |
| 76 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 77; anxiety Marta +1.5 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +2.2; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1 |
| 76 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 78; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.45; the tie stays open); impingement removed at once, anxiety Ravi -0.2 |
| 76 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 78; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Sofia rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 77 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 78; anxiety Ana +0.3, Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +1.2, Marta -1.2; Marta takes 2.0 of functioning from the other |
| 77 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 77 | Marta | TRIANGLE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.8; anxiety Pia +0.7 (witness), Ravi +1.2 (witness) |
| 77 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 78; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +4.3; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 77 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 78; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +2.2; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 77 | Ravi | WITHHOLD | Sofia | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 77 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 77 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 78, 79; symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.9, Nadia +2.2, Pia +2.6; anxiety Sofia +1.6 |
| 78 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 78 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 78 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 79; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.3 (witness); Marta rejects the other on Marta–Ravi (rejection +0.50; the tie stays open) |
| 78 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 79; anxiety Marta +1.3 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Ravi takes 2.0 of functioning from the other |
| 78 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 78 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 78 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 80; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 79 | Ana | UNDERFUNCTION | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 79 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 79 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 80; anxiety Marta +0.5, Nadia +1.1 (witness), Pia +1.3 (witness); Marta rejects the other on Marta–Ravi (rejection +0.30; the tie stays open) |
| 79 | Nadia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 79 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 80; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.8; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 8.4 of the sender's anxiety bound into Pia–Ravi |
| 79 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.4; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 79 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.7 (witness) |
| 79 | — | (system) | — | — | Ana–Marta: rejection has faded; the tie is no longer cut off (ordinary) |
| 80 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.4 (witness), Pia +1.6 (witness), Ravi +2.8 |
| 80 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.4 (witness), Pia +1.6 (witness), Ravi +1.4 (witness) |
| 80 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 80 | Bruno | CUTOFF | Ana | — | arrives week 81; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 80 | Dr Halim | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 81; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.7 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.7, Marta -0.7; the contact lands with Marta: systems perspective +0.10 |
| 80 | Marta | TRIANGLE | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 80 | Nadia | OVERFUNCTION | Marta | Pia, Ravi | arrives week 81; anxiety Marta +4.4, Nadia +0.5, Pia +0.7 (witness), Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.6, Nadia +0.6; Nadia takes 2.0 of functioning from the other |
| 80 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 81; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.3; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; 6.1 of the sender's anxiety bound into Pia–Ravi |
| 80 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 82; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 80 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 81 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 81 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 81 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 81 | Ana | CUTOFF | Bruno | — | arrives week 82; anxiety Ana +0.4, Bruno +0.1; calmer sender takes some anxiety: Ana +0.7, Bruno -0.7; Ana rejects the other on Ana–Bruno (rejection +0.10; the tie stays open) |
| 81 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 81 | Marta | CONFLICT | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.8; anxiety Pia +0.7 (witness), Ravi +1.1 (witness) |
| 81 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 82; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +4.4; calmer sender takes some anxiety: Nadia +0.0, Ravi -0.0 |
| 81 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.5 (witness), Ravi +1.1 (witness) |
| 81 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 83; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 81 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 83; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.10; the tie stays open); impingement removed at once, anxiety Sofia -0.3 |
| 82 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.10); the sender bears the refusal: anxiety Ana +1.9 |
| 82 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 82 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 83; anxiety Marta +0.6, Nadia +0.5 (witness), Ravi +1.1 (witness); Marta rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 82 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 83; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +1.5; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1; 5.7 of the sender's anxiety bound into Nadia–Ravi |
| 82 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 83; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.7; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; 5.5 of the sender's anxiety bound into Pia–Ravi |
| 82 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 84; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 82 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 83 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 83 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 83 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 84; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.3 (witness); Marta rejects the other on Marta–Ravi (rejection +0.20; the tie stays open) |
| 83 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 84; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +2.1; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; 5.2 of the sender's anxiety bound into Nadia–Ravi |
| 83 | Pia | OVERFUNCTION | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.5 (witness), Ravi +1.1 (witness) |
| 83 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 85; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.1; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 83 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 84 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +2.8 |
| 84 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.3 (witness), Pia +1.6 (witness), Ravi +1.4 (witness) |
| 84 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 84 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 84 | Bruno | WITHHOLD | Ana | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 84 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 85; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.3 (witness), Ravi +0.3; 11.1 of the sender's anxiety bound into Marta–Ravi |
| 84 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 85; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +3.2; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4 |
| 84 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 85; anxiety Marta +1.4 (witness), Nadia +0.5 (witness), Pia +0.5, Ravi +0.8; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; 6.4 of the sender's anxiety bound into Pia–Ravi |
| 84 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 86; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness), Ravi +0.3, Sofia +0.2; Ravi rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 84 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.5 (witness), Pia +0.7 (witness) |
| 84 | — | (system) | — | — | Pia–Ravi: rejection has faded; the tie is no longer cut off (ordinary) |
| 85 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 85 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 85 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 86; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.3 (witness); calmer sender takes some anxiety: Marta +0.0, Ravi -0.0; Marta rejects the other on Marta–Ravi (rejection +0.10; the tie stays open) |
| 85 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 86; anxiety Marta +1.4 (witness), Nadia +0.4, Pia +0.7 (witness), Ravi +0.3; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Ravi takes 2.0 of functioning from the other |
| 85 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 86; anxiety Marta +0.3, Nadia +0.5 (witness), Pia +0.5, Ravi +1.1 (witness); calmer sender takes some anxiety: Marta -0.9, Pia +0.9; Pia rejects the other on Marta–Pia (rejection +0.50; the tie stays open) |
| 85 | Ravi | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 85 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 85 | — | (system) | — | — | Pia–Ravi tie now distant |
| 86 | Ana | PURSUE | Marta | Nadia, Pia, Ravi | arrives week 87; anxiety Ana +0.3, Marta +1.0, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 86 | Bruno | CUTOFF | Ana | — | arrives week 87; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 86 | Marta | CONFLICT | Pia | Nadia, Ravi | refused by Pia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.9; anxiety Nadia +0.6 (witness), Ravi +1.1 (witness) |
| 86 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 87; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +2.3; 5.4 of the sender's anxiety bound into Nadia–Ravi |
| 86 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 87; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.9; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 6.1 of the sender's anxiety bound into Pia–Ravi |
| 86 | Ravi | WITHHOLD | Marta | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 86 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 87 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 87 | Bruno | CUTOFF | Ana | — | arrives week 88; anxiety Ana +0.2, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 87 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 88; anxiety Marta +0.6, Nadia +1.1 (witness), Pia +1.3 (witness); Marta rejects the other on Marta–Ravi (rejection +0.10; the tie stays open) |
| 87 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 88; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +2.6; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2 |
| 87 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 88; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +3.7; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 87 | Ravi | OVERFUNCTION | Nadia | Marta, Pia | arrives week 88; anxiety Marta +1.4 (witness), Nadia +2.4, Pia +0.6 (witness), Ravi +0.4; Ravi takes 2.0 of functioning from the other |
| 87 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 88 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.4 (witness), Pia +1.6 (witness), Ravi +2.7 |
| 88 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.4 (witness), Pia +1.6 (witness), Ravi +1.3 (witness) |
| 88 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 88 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 88 | Dr Halim | I-POSITION | Ravi | Marta, Nadia, Pia | arrives week 89; anxiety Dr Halim +0.1, Marta +0.9 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.6; calmer sender takes some anxiety: Dr Halim +0.6, Ravi -0.6; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 88 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 89; anxiety Marta +0.6, Nadia +1.2 (witness), Pia +1.3 (witness), Ravi +0.1; 10.8 of the sender's anxiety bound into Marta–Ravi |
| 88 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 89; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +1.2; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 88 | Pia | PREVENT_ALIGNMENT | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 88 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 89; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +1.2, Ravi +0.4; 8.7 of the sender's anxiety bound into Pia–Ravi |
| 88 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 89 | Ana | CONFLICT | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 89 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 89 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 90; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 89 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 90; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +2.7; 6.1 of the sender's anxiety bound into Nadia–Ravi |
| 89 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 90; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.6; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 89 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Nadia +1.2 (witness), Pia +1.3 (witness) |
| 89 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 90 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +1.9 |
| 90 | Bruno | CUTOFF | Ana | — | arrives week 91; anxiety Ana +0.2, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 90 | Marta | CUTOFF | Pia | Nadia, Ravi | arrives week 91; anxiety Marta +0.6, Nadia +0.6 (witness), Ravi +1.0 (witness); Marta rejects the other on Marta–Pia (rejection +0.35; the tie stays open) |
| 90 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 91; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +1.6 |
| 90 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 91; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.6; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3; 6.7 of the sender's anxiety bound into Pia–Ravi |
| 90 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 90 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 90 | — | (system) | — | — | Marta–Nadia: rejection has faded; the tie is no longer cut off (ordinary) |
| 91 | Ana | STAY-IN-CONTACT | Marta | Nadia, Pia, Ravi | arrives week 92; anxiety Ana +0.3, Marta -0.8, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 91 | Bruno | CUTOFF | Ana | — | arrives week 92; anxiety Ana +0.4, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 91 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 92; anxiety Marta +0.6, Nadia +1.2 (witness), Pia +1.3 (witness); 9.3 of the sender's anxiety bound into Marta–Ravi |
| 91 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 92; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 91 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 92; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 91 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 91 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 92 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.7 (witness), Nadia +1.5 (witness), Pia +1.6 (witness), Ravi +2.6 |
| 92 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.5, Nadia +1.5 (witness), Pia +1.6 (witness), Ravi +1.3 (witness) |
| 92 | Ana | CUTOFF | Bruno | — | arrives week 93; anxiety Ana +0.4, Bruno +0.4; calmer sender takes some anxiety: Ana +0.8, Bruno -0.8; Ana rejects the other on Ana–Bruno (rejection +0.05; the tie stays open); impingement removed at once, anxiety Ana -0.1 |
| 92 | Bruno | CONFLICT | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 92 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 93; anxiety Marta +0.6, Nadia +1.2 (witness), Pia +1.3 (witness); calmer sender takes some anxiety: Marta +0.1, Ravi -0.1; Marta rejects the other on Marta–Ravi (rejection +0.15; the tie stays open) |
| 92 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 93; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +1.9 |
| 92 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 93; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Pia rejects the other on Pia–Ravi (rejection +0.15; the tie stays open) |
| 92 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.10); the sender bears the refusal: anxiety Ravi +1.3; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 92 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 93 | Ana | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 93 | Bruno | CUTOFF | Ana | — | arrives week 94; anxiety Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.10; the tie stays open); impingement removed at once, anxiety Bruno -0.1 |
| 93 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | arrives week 94; anxiety Marta +0.2, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 93 | Nadia | CUTOFF | Marta | Pia, Ravi | arrives week 94; anxiety Marta +0.6, Nadia +0.5, Pia +0.6 (witness), Ravi +1.0 (witness); Nadia rejects the other on Marta–Nadia (rejection +0.50; the tie stays open) |
| 93 | Pia | CUTOFF | Marta | Nadia, Ravi | arrives week 94; anxiety Nadia +0.6 (witness), Pia +0.5, Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.5, Pia +0.5; Pia rejects the other on Marta–Pia (rejection +0.40; the tie stays open) |
| 93 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 94; anxiety Marta +1.4 (witness), Nadia +0.0, Pia +0.6 (witness), Ravi +0.4; calmer sender takes some anxiety: Nadia -0.1, Ravi +0.1; Nadia left outside: anxiety Nadia +2.7, Ravi -1.8 (of which 0.9 the outsider's own) |
| 93 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 94 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 94 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 94 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 95; anxiety Ana +0.3, Marta -0.2, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Marta takes 2.0 of functioning from the other |
| 94 | Bruno | CUTOFF | Ana | — | arrives week 95; anxiety Ana +0.2, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 94 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 95; anxiety Marta +0.6, Nadia +1.2 (witness), Pia +1.3 (witness); Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 94 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 95; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.8; Ravi left outside: anxiety Nadia -1.3, Ravi +2.0 (of which 0.7 the outsider's own) |
| 94 | Pia | CONFLICT | Ravi | Marta, Nadia | arrives week 95; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +3.3; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 94 | Ravi | OVERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.1; anxiety Nadia +1.2 (witness), Pia +1.3 (witness) |
| 94 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 95 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 95 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 95 | Bruno | UNDERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 95 | Marta | DISTANCE | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.7; anxiety Pia +0.6 (witness), Ravi +1.0 (witness) |
| 95 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 96; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.7; Ravi takes 2.0 of functioning from the other |
| 95 | Pia | DISTANCE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 95 | Ravi | UNDERFUNCTION | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +2.1; anxiety Nadia +1.2 (witness), Pia +1.3 (witness) |
| 95 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 96 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.5 (witness), Pia +1.6 (witness), Ravi +2.6 |
| 96 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.3, Nadia +1.5 (witness), Pia +1.6 (witness), Ravi +1.3 (witness) |
| 96 | Ana | DISTANCE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.0 |
| 96 | Bruno | DISTANCE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 96 | Dr Halim | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 97; anxiety Dr Halim +0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Dr Halim rejects the other on Dr Halim–Marta (rejection +0.50; the tie stays open) |
| 96 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 97; anxiety Marta +0.5, Nadia +1.3 (witness), Pia +1.3 (witness), Ravi -0.4; Ravi left outside: anxiety Marta -1.8, Ravi +2.7 (of which 0.9 the outsider's own) |
| 96 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 97; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.6; 6.4 of the sender's anxiety bound into Nadia–Ravi |
| 96 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 96 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 98; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.2; Ravi rejects the other on Ravi–Sofia (rejection +0.05; the tie stays open) |
| 96 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.7 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 97 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 97 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 97 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 97 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 97 | Ana | UNDERFUNCTION | Marta | Nadia, Pia, Ravi | arrives week 98; anxiety Ana +0.3, Marta -0.1, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Marta takes 2.0 of functioning from the other |
| 97 | Bruno | CUTOFF | Ana | — | arrives week 98; anxiety Ana +0.0, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 97 | Marta | CUTOFF | Ravi | Nadia, Pia | arrives week 98; anxiety Marta +0.5, Nadia +1.3 (witness), Pia +1.3 (witness); calmer sender takes some anxiety: Marta +0.4, Ravi -0.4; Marta rejects the other on Marta–Ravi (rejection +0.00; the tie stays open) |
| 97 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 98; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; 6.9 of the sender's anxiety bound into Nadia–Ravi |
| 97 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 98; anxiety Marta +1.4 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 9.9 of the sender's anxiety bound into Pia–Ravi |
| 97 | Ravi | PURSUE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.0; anxiety Nadia +1.3 (witness), Pia +1.3 (witness) |
| 97 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 99; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.50; the tie stays open) |
| 98 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.1 |
| 98 | Bruno | PURSUE | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 98 | Marta | SPLIT | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +0.8; anxiety Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 98 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 99; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2 |
| 98 | Pia | WITHHOLD | Marta | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 98 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | refused by Sofia (rejection +0.05); the sender bears the refusal: anxiety Ravi +1.2; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 98 | Sofia | CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 100; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia rejects the other on Ravi–Sofia (rejection +0.00; the tie stays open) |
| 99 | Ana | OVERFUNCTION | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.1 |
| 99 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 99 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | refused by Nadia (rejection +0.10); the sender bears the refusal: anxiety Marta +2.6; anxiety Pia +0.6 (witness), Ravi +1.0 (witness) |
| 99 | Nadia | DISTANCE | Marta | Pia, Ravi | arrives week 100; anxiety Marta +0.4, Nadia +0.5, Pia +0.6 (witness), Ravi +1.0 (witness); calmer sender takes some anxiety: Marta -0.2, Nadia +0.2; 6.0 of the sender's anxiety bound into Marta–Nadia |
| 99 | Pia | PURSUE | Marta | Nadia, Ravi | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Pia +2.6; anxiety Nadia +0.6 (witness), Ravi +1.0 (witness) |
| 99 | Ravi | DISTANCE | Marta | Nadia, Pia | refused by Marta (rejection +0.10); the sender bears the refusal: anxiety Ravi +2.0; anxiety Nadia +1.3 (witness), Pia +1.3 (witness) |
| 99 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 100 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | a 1-week spell, one week; anxiety Marta +1.6 (witness), Nadia +1.5 (witness), Pia +1.6 (witness), Ravi +2.5 |
| 100 | (from outside) | JOB_LOSS | Marta | Nadia, Pia, Ravi | a 1-week spell, one week; anxiety Marta +3.1, Nadia +1.5 (witness), Pia +1.6 (witness), Ravi +1.2 (witness) |
| 100 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 100 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 100 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 100 | Ana | PURSUE | Bruno | — | refused by Bruno (rejection +0.05); the sender bears the refusal: anxiety Ana +2.1 |
| 100 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.05); the sender bears the refusal: anxiety Bruno +2.7 |
| 100 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 101; anxiety Ana +1.3, Marta +0.3, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.5 (witness) |
| 100 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 101; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +3.6; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Nadia takes 2.0 of functioning from the other |
| 100 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 101; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.9; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6 |
| 100 | Ravi | DISTANCE | Marta | Nadia, Pia | refused by Marta (rejection +0.05); the sender bears the refusal: anxiety Ravi +2.0; anxiety Nadia +1.3 (witness), Pia +1.3 (witness) |
| 100 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.10); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 101 | Ana | CONFLICT | Marta | Nadia, Pia, Ravi | arrives week 102; anxiety Ana +0.3, Marta +2.8, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 101 | Bruno | CUTOFF | Ana | — | arrives week 102; anxiety Ana +0.1, Bruno +0.5; Bruno rejects the other on Ana–Bruno (rejection +0.05; the tie stays open) |
| 101 | Marta | FRAME_AMBIGUITY | Dr Halim | Nadia, Pia, Ravi | refused by Dr Halim (rejection +0.10); the sender bears the refusal: anxiety Marta +0.8; anxiety Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.7 (witness) |
| 101 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 102; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.5; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Ravi takes 2.0 of functioning from the other |
| 101 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 102; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +1.8; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6 |
| 101 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 102; anxiety Marta +1.3 (witness), Nadia -0.3, Pia +0.6 (witness), Ravi +0.4; Nadia left outside: anxiety Nadia +3.3, Ravi -2.2 (of which 1.1 the outsider's own) |
| 101 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | refused by Ravi (rejection +0.05); the sender bears the refusal: anxiety Sofia +1.6; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.6 (witness) |
| 102 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 102 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 102 | Ana | CUTOFF | Bruno | — | arrives week 103; anxiety Ana +0.4, Bruno +0.4; calmer sender takes some anxiety: Ana +0.7, Bruno -0.7; Ana rejects the other on Ana–Bruno (rejection +0.00; the tie stays open) |
| 102 | Bruno | OVERFUNCTION | Ana | — | refused by Ana (rejection +0.10); the sender bears the refusal: anxiety Bruno +2.7 |
| 102 | Marta | CONFLICT | Ana | Nadia, Pia, Ravi | arrives week 103; anxiety Ana +3.2, Marta +0.3, Nadia +0.6 (witness), Pia +0.6 (witness), Ravi +0.5 (witness) |
| 102 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 103; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.6 (witness), Ravi +0.4; Ravi left outside: anxiety Nadia -1.1, Ravi +1.7 (of which 0.6 the outsider's own) |
| 102 | Pia | CUTOFF | Ravi | Marta, Nadia | arrives week 103; anxiety Marta +1.3 (witness), Nadia +0.6 (witness), Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia rejects the other on Pia–Ravi (rejection +0.50; the tie stays open) |
| 102 | Ravi | WITHHOLD | Sofia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 102 | Sofia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 103 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 103 | Ana | DISTANCE | Bruno | — | — |
| 103 | Bruno | CUTOFF | Ana | — | — |
| 103 | Marta | TRIANGLE | Ravi | Nadia, Pia | — |
| 103 | Nadia | TRIANGLE | Marta | Pia, Ravi | — |
| 103 | Pia | DISTANCE | Ravi | Marta, Nadia | — |
| 103 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | — |
| 103 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | — |
| 103 | — | (system) | — | — | slow tick (yearly): 10 ties wore (conductance); every dyad a year older |

104 weeks. The invariants were asserted at the end of every week.
