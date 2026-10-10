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
| config hash | `d22146e329a3cb7adf59f995c1c0c8026a95f55a11e8782c65fadeb2a6618a38` |
| spec revision | 2.0, revision 12 |
| family instance | phase_c |
| activation | synchronous_activation v1 (synchronous) |
| visibility | household_conductance_visibility v1 |
| constants frozen | 2026-10-07 |
| constants changed since freeze | none |

The standing load, the decay toward each person's floor, the relaxation of felt contact, symptom accumulation, investment, the reactive detectors and outside-ness run every week for everyone and are not listed line by line. Effects are shown beside the event that caused them.

| Week | Who | Move | Toward | Witnesses | What it did |
|---|---|---|---|---|---|
| 0 | Ana | WITHHOLD | Marta | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 0 | Bruno | (holds) | — | — | fallback (hold): no legal act |
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
| 1 | Ana | CUTOFF | Marta | Nadia, Pia, Ravi | arrives week 2; anxiety Ana +0.3, Marta +0.4, Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Marta -0.4 |
| 1 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 1 | Marta | CUTOFF | Nadia | Pia, Ravi | arrives week 2; anxiety Marta +0.5, Nadia +0.8, Pia +0.5 (witness), Ravi +1.4 (witness); calmer sender takes some anxiety: Marta +0.3, Nadia -0.3; Marta–Nadia now cut off (no events; bond energy kept); impingement removed at once, anxiety Marta -0.6, Nadia -0.8 |
| 1 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 2; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi -0.6; Ravi left outside: anxiety Nadia -0.4, Ravi +0.7 (of which 0.2 the outsider's own) |
| 1 | Pia | UNDERFUNCTION | Marta | Nadia, Ravi | arrives week 2; anxiety Marta -0.2, Nadia +0.6 (witness), Pia +0.4, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4; Marta takes 2.0 of functioning from the other |
| 1 | Ravi | WITHHOLD | Nadia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 1 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 3; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +3.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5 |
| 2 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 2 | Ana | REDUCE_CUTOFF | Marta | Pia, Ravi | arrives week 3; anxiety Ana +0.3, Marta -0.4, Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta reopened |
| 2 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 2 | Marta | CONFLICT | Pia | Ravi | arrives week 3; anxiety Marta +0.5, Pia +4.9, Ravi +1.4 (witness) |
| 2 | Nadia | DISTANCE | Ravi | Marta, Pia | arrives week 3; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +4.9; 1.8 of the sender's anxiety bound into Nadia–Ravi |
| 2 | Pia | DISTANCE | Ravi | Marta, Nadia | arrives week 3; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +0.4, Ravi +4.8; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; 4.2 of the sender's anxiety bound into Pia–Ravi |
| 2 | Ravi | STAY-IN-CONTACT | Nadia | Marta, Pia | arrives week 3; anxiety Marta +1.2 (witness), Nadia -0.6, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia -0.1, Ravi +0.1 |
| 2 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 4; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +2.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; 1.2 of the sender's anxiety bound into Ravi–Sofia |
| 3 | Ana | PURSUE | Marta | Pia, Ravi | arrives week 4; anxiety Ana +0.3, Marta +2.2, Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.4, Marta -0.4 |
| 3 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 3 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 4; anxiety Marta +0.5, Nadia +1.2 (witness), Pia +1.1 (witness), Ravi -0.1; calmer sender takes some anxiety: Marta +0.6, Ravi -0.6; Ravi left outside: anxiety Marta -1.2, Ravi +1.8 (of which 0.6 the outsider's own) |
| 3 | Nadia | CONFLICT | Ravi | Marta, Pia | arrives week 4; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +5.5; calmer sender takes some anxiety: Nadia +0.5, Ravi -0.5 |
| 3 | Pia | TRIANGLE | Marta | Ravi | arrives week 4; anxiety Marta +0.8, Pia +0.4, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.4, Pia +0.4; Marta left outside: anxiety Marta +1.7, Pia -1.1 (of which 0.6 the outsider's own) |
| 3 | Ravi | DISTANCE | Pia | Marta, Nadia | arrives week 4; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +2.1, Ravi +0.6; 11.1 of the sender's anxiety bound into Pia–Ravi |
| 3 | Sofia | OVERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 5; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +3.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Sofia takes 2.0 of functioning from the other |
| 4 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | one-time effect; a 1-week spell, recorded; anxiety Marta +1.5 (witness), Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +3.4 |
| 4 | (from outside) | JOB_LOSS | Marta | Pia, Ravi | one-time effect; a 1-week spell, recorded; anxiety Marta +3.0, Pia +1.3 (witness), Ravi +1.7 (witness) |
| 4 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 4 | Ana | CONFLICT | Marta | Pia, Ravi | arrives week 5; anxiety Ana +0.3, Marta +3.6, Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 4 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 4 | Marta | SPLIT | Dr Halim | Pia, Ravi | arrives week 5; anxiety Marta +0.1, Pia +0.5 (witness), Ravi +0.9 (witness) |
| 4 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 5; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +0.6; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3 |
| 4 | Pia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 4 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | arrives week 6; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.4, Sofia +1.8 |
| 4 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 6; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +2.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; 2.4 of the sender's anxiety bound into Ravi–Sofia |
| 4 | Ravi | SYMPTOM_ONSET | Dr Halim, Marta, Nadia, Pia, Sofia | — | arrives week 5, 6; symptom onset, physical channel; anxiety Dr Halim +0.5, Marta +2.5, Nadia +2.4, Pia +2.2; anxiety Sofia +1.3 |
| 5 | Ana | PURSUE | Marta | Pia, Ravi | arrives week 6; anxiety Ana +0.3, Marta +2.2, Pia +0.5 (witness), Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 5 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 5 | Marta | CONFLICT | Ana | Pia, Ravi | arrives week 6; anxiety Ana +2.1, Marta +0.3, Pia +0.5 (witness), Ravi +0.8 (witness) |
| 5 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 6; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Ravi takes 2.0 of functioning from the other |
| 5 | Pia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 5 | Ravi | CONFLICT | Pia | Marta, Nadia | arrives week 6; anxiety Marta +1.2 (witness), Nadia +0.6 (witness), Pia +3.6, Ravi +0.6 |
| 5 | Sofia | WITHHOLD | Ravi | — | held back PURSUE: computed, not emitted; attention on the tie rises |
| 6 | Ana | DISTANCE | Marta | Pia, Ravi | arrives week 7; anxiety Ana +0.3, Marta +0.8, Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; 2.9 of the sender's anxiety bound into Ana–Marta |
| 6 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 6 | Marta | TRIANGLE | Pia | Ravi | arrives week 7; anxiety Marta +0.5, Pia +1.1, Ravi +1.4 (witness); Pia left outside: anxiety Marta -1.6, Pia +2.4 (of which 0.8 the outsider's own) |
| 6 | Nadia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 7; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +1.3; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2 |
| 6 | Pia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 6 | Ravi | DISPLACE | Dr Halim | Marta, Nadia, Pia | arrives week 7; anxiety Marta +0.8 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +0.2 |
| 6 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 8; anxiety Marta +0.6 (witness), Nadia +0.6 (witness), Pia +0.5 (witness), Ravi +1.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 6 | Marta | SYMPTOM_ONSET | Ana, Dr Halim, Pia, Ravi | Nadia | arrives week 7; symptom onset, physical channel; anxiety Ana +1.6, Dr Halim +0.5, Nadia +1.3 (witness), Pia +2.2, Ravi +2.9 |
| 7 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 7 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 7 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 7 | Ana | PURSUE | Marta | Pia, Ravi | arrives week 8; anxiety Ana +0.3, Marta +2.1, Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 7 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 7 | Marta | DISTANCE | Pia | Ravi | arrives week 8; anxiety Marta +0.5, Pia +1.5, Ravi +1.4 (witness); 8.7 of the sender's anxiety bound into Marta–Pia |
| 7 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 8; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +1.1; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Ravi takes 2.0 of functioning from the other |
| 7 | Pia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 7 | Ravi | CONFLICT | Sofia | Marta, Nadia, Pia | arrives week 9; anxiety Marta +0.6 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +0.3, Sofia +2.9 |
| 7 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Marta +0.6 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0; Ravi takes 2.0 of functioning from the other |
| 7 | Pia | SYMPTOM_ONSET | Marta, Ravi | Nadia | arrives week 8; symptom onset, social channel; anxiety Marta +2.5, Nadia +0.6 (witness), Ravi +2.9 |
| 8 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | one-time effect; a 1-week spell, recorded; anxiety Marta +1.5 (witness), Nadia +1.5 (witness), Pia +1.3 (witness), Ravi +3.4 |
| 8 | (from outside) | JOB_LOSS | Marta | Pia, Ravi | one-time effect; a 1-week spell, recorded; anxiety Marta +3.0, Pia +1.3 (witness), Ravi +1.7 (witness) |
| 8 | Ana | PURSUE | Marta | Pia, Ravi | arrives week 9; anxiety Ana +0.3, Marta +2.1, Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 8 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 8 | Dr Halim | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 9; anxiety Dr Halim +0.1, Marta +0.8 (witness), Nadia +0.7 (witness), Pia +0.5 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Ravi -0.6 |
| 8 | Marta | PURSUE | Ana | Pia, Ravi | arrives week 9; anxiety Ana +1.8, Marta +0.3, Pia +0.5 (witness), Ravi +0.7 (witness) |
| 8 | Nadia | PREVENT_ALIGNMENT | Ravi | Marta, Pia | arrives week 9; anxiety Marta +1.2 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +1.0; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4 |
| 8 | Pia | OVERFUNCTION | Marta | Ravi | arrives week 9; anxiety Marta +4.4, Pia +0.4, Ravi +1.4 (witness); calmer sender takes some anxiety: Marta -0.2, Pia +0.2; Pia takes 2.0 of functioning from the other |
| 8 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 10; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +0.3, Sofia +0.6; Ravi–Sofia now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -2.9, Sofia -3.1 |
| 8 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 10; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; 2.0 of the sender's anxiety bound into Ravi–Sofia |
| 8 | Nadia | SYMPTOM_ONSET | Ravi | Marta, Pia | arrives week 9; symptom onset, mental channel; anxiety Marta +1.2 (witness), Pia +0.5 (witness), Ravi +2.7 |
| 9 | Ana | PURSUE | Marta | Pia, Ravi | arrives week 10; anxiety Ana +0.3, Marta +2.2, Pia +0.5 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7 |
| 9 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 9 | Marta | DISTANCE | Pia | Ravi | arrives week 10; anxiety Marta +0.5, Pia +2.3, Ravi +1.3 (witness); 8.6 of the sender's anxiety bound into Marta–Pia |
| 9 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 10; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Ravi left outside: anxiety Nadia -0.9, Ravi +1.4 (of which 0.5 the outsider's own) |
| 9 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 10; anxiety Marta +1.3 (witness), Nadia +0.7 (witness), Pia +0.4, Ravi +4.5; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia takes 2.0 of functioning from the other |
| 9 | Ravi | CUTOFF | Sofia | Marta, Nadia, Pia | arrives week 11; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +0.3; Ravi–Sofia now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -1.5 |
| 9 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 11; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +1.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 10 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 10 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 10 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 10 | Ana | CUTOFF | Marta | Pia, Ravi | arrives week 11; anxiety Ana +0.3, Marta +0.7, Pia +0.5 (witness), Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -1.1, Marta -4.8 |
| 10 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 10 | Marta | UNDERFUNCTION | Pia | Ravi | arrives week 11; anxiety Marta +0.5, Pia +0.3, Ravi +1.4 (witness); Pia takes 2.0 of functioning from the other |
| 10 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 11; anxiety Marta +1.3 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.3, Ravi -0.3; Ravi takes 2.0 of functioning from the other |
| 10 | Pia | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 10 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 11; anxiety Marta -0.9, Nadia +1.4 (witness), Pia +1.0 (witness), Ravi +0.5; Marta left outside: anxiety Marta +3.6, Ravi -2.4 (of which 1.2 the outsider's own) |
| 10 | Sofia | REDUCE_CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 12; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi -0.5, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Ravi–Sofia reopened |
| 11 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 11 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Ravi |
| 11 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 11 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 11 | Ana | REDUCE_CUTOFF | Marta | Pia, Ravi | arrives week 12; anxiety Ana +0.3, Marta +0.6, Pia +0.5 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana–Marta reopened |
| 11 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 11 | Marta | UNDERFUNCTION | Pia | Ravi | arrives week 12; anxiety Marta +0.6, Pia +0.1, Ravi +1.3 (witness); Pia takes 2.0 of functioning from the other |
| 11 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 12; anxiety Marta +1.4 (witness), Nadia +0.6, Pia +0.5 (witness), Ravi +3.1; calmer sender takes some anxiety: Nadia +0.0, Ravi -0.0 |
| 11 | Pia | WITHHOLD | Marta | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 11 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 12; anxiety Marta +6.2, Nadia +1.4 (witness), Pia +1.0 (witness), Ravi +0.5 |
| 11 | Sofia | REDUCE_CUTOFF | Ravi | Marta, Nadia, Pia | arrives week 13; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +0.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 12 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | one-time effect; a 1-week spell, recorded; anxiety Marta +1.7 (witness), Nadia +1.7 (witness), Pia +1.2 (witness), Ravi +3.1 |
| 12 | (from outside) | JOB_LOSS | Marta | Pia, Ravi | one-time effect; a 1-week spell, recorded; anxiety Marta +3.3, Pia +1.2 (witness), Ravi +1.5 (witness) |
| 12 | Ana | OVERFUNCTION | Marta | Pia, Ravi | arrives week 13; anxiety Ana +0.3, Marta +3.4, Pia +0.5 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Ana takes 2.0 of functioning from the other |
| 12 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 12 | Marta | OVERFUNCTION | Ana | Pia, Ravi | arrives week 13; anxiety Ana +2.4, Marta +0.4, Pia +0.5 (witness), Ravi +0.6 (witness); Marta takes 2.0 of functioning from the other |
| 12 | Nadia | PURSUE | Ravi | Marta, Pia | arrives week 13; anxiety Marta +1.5 (witness), Nadia +0.6, Pia +0.5 (witness), Ravi +3.1; calmer sender takes some anxiety: Nadia +0.1, Ravi -0.1 |
| 12 | Pia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 12 | Ravi | TRIANGLE | Pia | Marta, Nadia | arrives week 13; anxiety Marta +1.5 (witness), Nadia +0.7 (witness), Pia -0.2, Ravi +0.5; Pia left outside: anxiety Pia +3.1, Ravi -2.0 (of which 1.0 the outsider's own) |
| 12 | Sofia | DISTANCE | Ravi | Marta, Nadia, Pia | arrives week 14; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +1.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; 1.9 of the sender's anxiety bound into Ravi–Sofia |
| 13 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 13 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 13 | Ana | DISTANCE | Marta | Pia, Ravi | arrives week 14; anxiety Ana +0.3, Marta +0.7, Pia +0.5 (witness), Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; 3.0 of the sender's anxiety bound into Ana–Marta |
| 13 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 13 | Marta | TRIANGLE | Ravi | Nadia, Pia | arrives week 14; anxiety Marta +0.6, Nadia +1.4 (witness), Pia +0.9 (witness), Ravi -0.4; Ravi left outside: anxiety Marta -2.2, Ravi +3.3 (of which 1.1 the outsider's own) |
| 13 | Nadia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 14; anxiety Marta +1.5 (witness), Nadia +0.6, Pia +0.5 (witness), Ravi +2.8; calmer sender takes some anxiety: Nadia +0.0, Ravi -0.0; Nadia takes 2.0 of functioning from the other |
| 13 | Pia | CUTOFF | Marta | Ravi | arrives week 14; anxiety Marta +0.8, Pia +0.4, Ravi +1.3 (witness); calmer sender takes some anxiety: Marta -0.6, Pia +0.6; Marta–Pia now cut off (no events; bond energy kept); impingement removed at once, anxiety Marta -1.2, Pia -0.4 |
| 13 | Ravi | TRIANGLE | Marta | Nadia, Pia | arrives week 14; anxiety Marta +1.1, Nadia +1.4 (witness), Pia +0.9 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.2, Ravi +0.2; Marta left outside: anxiety Marta +3.7, Ravi -2.5 (of which 1.2 the outsider's own) |
| 13 | Sofia | UNDERFUNCTION | Ravi | Marta, Nadia, Pia | arrives week 15; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9; Ravi takes 2.0 of functioning from the other |
| 13 | Ana | SYMPTOM_ONSET | Marta | Pia, Ravi | arrives week 14; symptom onset, physical channel; anxiety Marta +1.8, Pia +0.5 (witness), Ravi +0.6 (witness) |
| 14 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 14 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 14 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Pia–Ravi |
| 14 | Ana | UNDERFUNCTION | Marta | Ravi | arrives week 15; anxiety Ana +0.3, Marta +0.6, Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8; Marta takes 2.0 of functioning from the other |
| 14 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 14 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 15; anxiety Marta +0.6, Nadia +1.4 (witness), Pia +0.9 (witness), Ravi +1.8; 9.0 of the sender's anxiety bound into Marta–Ravi |
| 14 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 15; anxiety Marta +1.5 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +0.9; calmer sender takes some anxiety: Nadia +0.0, Ravi -0.0; Nadia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -6.0 |
| 14 | Pia | PURSUE | Ravi | Marta, Nadia | arrives week 15; anxiety Marta +1.5 (witness), Nadia +0.7 (witness), Pia +0.4, Ravi +2.0; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 14 | Ravi | UNDERFUNCTION | Sofia | Marta, Nadia, Pia | arrives week 16; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +0.3, Sofia -0.3; Sofia takes 2.0 of functioning from the other |
| 14 | Sofia | CONFLICT | Ravi | Marta, Nadia, Pia | arrives week 16; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.5 (witness), Ravi +3.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 15 | Ana | CUTOFF | Marta | Ravi | arrives week 16; anxiety Ana +0.3, Marta +0.5, Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -0.5, Marta -2.1 |
| 15 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 15 | Marta | TRIANGLE | Ravi | Pia | arrives week 16; anxiety Marta +0.6, Pia +0.9 (witness), Ravi -0.5; Ravi left outside: anxiety Marta -1.9, Ravi +2.9 (of which 1.0 the outsider's own) |
| 15 | Nadia | REDUCE_CUTOFF | Ravi | Marta, Pia | arrives week 16; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.5 (witness), Ravi +1.0; Nadia–Ravi reopened; bound anxiety released to the third party: Marta +1.8 |
| 15 | Pia | PURSUE | Ravi | Marta | arrives week 16; anxiety Marta +1.4 (witness), Pia +0.4, Ravi +2.2; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3 |
| 15 | Ravi | UNDERFUNCTION | Pia | Marta | arrives week 16; anxiety Marta +1.4 (witness), Pia -0.0, Ravi +0.5; Pia takes 2.0 of functioning from the other |
| 15 | Sofia | CONFLICT | Ravi | Marta, Pia | arrives week 17; anxiety Marta +0.7 (witness), Pia +0.4 (witness), Ravi +3.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.9, Sofia +0.9 |
| 16 | (from outside) | JOB_LOSS | Ravi | Marta, Pia | one-time effect; a 1-week spell, recorded; anxiety Marta +1.7 (witness), Pia +1.1 (witness), Ravi +3.1 |
| 16 | (from outside) | JOB_LOSS | Marta | Ravi | one-time effect; a 1-week spell, recorded; anxiety Marta +3.3, Ravi +1.5 (witness) |
| 16 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 16 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 16 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 16 | Ana | REDUCE_CUTOFF | Marta | Ravi | arrives week 17; anxiety Ana +0.3, Marta +0.4, Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana–Marta reopened |
| 16 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 16 | Dr Halim | STAY-IN-CONTACT | Marta | Ravi | arrives week 17; anxiety Dr Halim +0.1, Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6 |
| 16 | Marta | STAY-IN-CONTACT | Ravi | Nadia, Pia | arrives week 17; anxiety Marta +0.6, Nadia +1.4 (witness), Pia +0.9 (witness), Ravi +0.0; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1 |
| 16 | Nadia | TRIANGLE | Ravi | Marta, Pia | arrives week 17; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.4 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.2, Ravi -0.2; Ravi left outside: anxiety Nadia -1.0, Ravi +1.5 (of which 0.5 the outsider's own) |
| 16 | Pia | OVERFUNCTION | Ravi | Marta, Nadia | arrives week 17; anxiety Marta +1.4 (witness), Nadia +0.7 (witness), Pia +0.4, Ravi +4.0; calmer sender takes some anxiety: Pia +0.7, Ravi -0.7; Pia takes 2.0 of functioning from the other |
| 16 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | arrives week 18; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.4 (witness), Ravi +0.4, Sofia +1.3 |
| 16 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Nadia, Pia | arrives week 18; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.4 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -1.0, Sofia +1.0 |
| 17 | Ana | UNDERFUNCTION | Marta | Ravi | arrives week 18; anxiety Ana +0.3, Marta +0.5, Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Marta takes 2.0 of functioning from the other |
| 17 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 17 | Marta | UNDERFUNCTION | Ana | Ravi | arrives week 18; anxiety Marta +0.3, Ravi +0.8 (witness); Ana takes 2.0 of functioning from the other |
| 17 | Nadia | CUTOFF | Ravi | Marta, Pia | arrives week 18; anxiety Marta +1.4 (witness), Nadia +0.5, Pia +0.4 (witness), Ravi +0.8; calmer sender takes some anxiety: Nadia +0.4, Ravi -0.4; Nadia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Nadia -0.2, Ravi -1.7 |
| 17 | Pia | TRIANGLE | Ravi | Marta, Nadia | arrives week 18; anxiety Marta +1.4 (witness), Nadia +0.7 (witness), Pia +0.3, Ravi +0.9; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Ravi left outside: anxiety Pia -1.2, Ravi +1.8 (of which 0.6 the outsider's own) |
| 17 | Ravi | DISTANCE | Nadia | Marta, Pia | arrives week 18; anxiety Marta +1.4 (witness), Nadia +2.8, Pia +0.4 (witness), Ravi +0.6; 9.8 of the sender's anxiety bound into Nadia–Ravi |
| 17 | Sofia | PURSUE | Ravi | Marta, Nadia, Pia | arrives week 19; anxiety Marta +0.7 (witness), Nadia +0.7 (witness), Pia +0.4 (witness), Ravi +2.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6 |
| 18 | Ana | DISTANCE | Marta | Ravi | arrives week 19; anxiety Ana +0.3, Marta +0.4, Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; 2.4 of the sender's anxiety bound into Ana–Marta |
| 18 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 18 | Marta | PREVENT_ALIGNMENT | Ravi | Pia | arrives week 19; anxiety Marta +0.6, Pia +0.9 (witness), Ravi -0.1 |
| 18 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 18 | Pia | DISTANCE | Ravi | Marta | arrives week 19; anxiety Marta +1.4 (witness), Pia +0.3, Ravi +2.9; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; 5.7 of the sender's anxiety bound into Pia–Ravi |
| 18 | Ravi | UNDERFUNCTION | Sofia | Marta, Pia | arrives week 20; anxiety Marta +0.7 (witness), Pia +0.4 (witness), Ravi +0.4, Sofia +0.4; Sofia takes 2.0 of functioning from the other |
| 18 | Sofia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 20; anxiety Marta +0.7 (witness), Pia +0.4 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Ravi takes 2.0 of functioning from the other |
| 19 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 19 | Ana | UNDERFUNCTION | Marta | Ravi | arrives week 20; anxiety Ana +0.3, Marta +0.4, Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Marta takes 2.0 of functioning from the other |
| 19 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 19 | Marta | FRAME_AMBIGUITY | Dr Halim | Ravi | arrives week 20; anxiety Marta +0.2, Ravi +1.0 (witness) |
| 19 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 19 | Pia | UNDERFUNCTION | Ravi | Marta | arrives week 20; anxiety Marta +1.4 (witness), Pia +0.3, Ravi -0.4; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; Ravi takes 2.0 of functioning from the other |
| 19 | Ravi | OVERFUNCTION | Sofia | Marta, Pia | arrives week 21; anxiety Marta +0.7 (witness), Pia +0.4 (witness), Ravi +0.3, Sofia +2.6; Ravi takes 2.0 of functioning from the other |
| 19 | Sofia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 21; anxiety Marta +0.7 (witness), Pia +0.4 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Ravi takes 2.0 of functioning from the other |
| 20 | (from outside) | JOB_LOSS | Ravi | Marta, Pia | one-time effect; a 1-week spell, recorded; anxiety Marta +1.7 (witness), Pia +1.0 (witness), Ravi +3.6 |
| 20 | (from outside) | JOB_LOSS | Marta | Ravi | one-time effect; a 1-week spell, recorded; anxiety Marta +3.3, Ravi +1.8 (witness) |
| 20 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 20 | Ana | CONFLICT | Marta | Ravi | arrives week 21; anxiety Ana +0.4, Marta +3.3, Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.8, Marta -0.8 |
| 20 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 20 | Marta | DISTANCE | Ravi | Pia | arrives week 21; anxiety Marta +0.5, Pia +0.9 (witness), Ravi +2.6; 9.1 of the sender's anxiety bound into Marta–Ravi |
| 20 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 20 | Pia | OVERFUNCTION | Ravi | Marta | arrives week 21; anxiety Marta +1.3 (witness), Pia +0.4, Ravi +4.3; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; Pia takes 2.0 of functioning from the other |
| 20 | Ravi | OVERFUNCTION | Pia | Marta | arrives week 21; anxiety Marta +1.3 (witness), Pia +2.7, Ravi +0.6; Ravi takes 2.0 of functioning from the other |
| 20 | Sofia | PURSUE | Ravi | Marta, Pia | arrives week 22; anxiety Marta +0.7 (witness), Pia +0.4 (witness), Ravi +2.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 21 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 21 | Ana | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 21 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 21 | Marta | TRIANGLE | Ravi | Pia | arrives week 22; anxiety Marta +0.5, Pia +0.9 (witness), Ravi -0.4; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2; Ravi left outside: anxiety Marta -1.5, Ravi +2.3 (of which 0.8 the outsider's own) |
| 21 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 21 | Pia | PURSUE | Ravi | Marta | arrives week 22; anxiety Marta +1.3 (witness), Pia +0.4, Ravi +2.2; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8 |
| 21 | Ravi | UNDERFUNCTION | Sofia | Marta, Pia | arrives week 23; anxiety Marta +0.7 (witness), Pia +0.4 (witness), Ravi +0.3, Sofia +0.5; Sofia takes 2.0 of functioning from the other |
| 21 | Sofia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 23; anxiety Marta +0.7 (witness), Pia +0.4 (witness), Ravi +3.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia takes 2.0 of functioning from the other |
| 21 | Sofia | SYMPTOM_ONSET | Ravi | Marta, Pia | arrives week 23; symptom onset, physical channel; anxiety Marta +0.7 (witness), Pia +0.4 (witness), Ravi +1.5 |
| 22 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 22 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 22 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 22 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 22 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 22 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Pia |
| 22 | Ana | PURSUE | Marta | Ravi | arrives week 23; anxiety Ana +0.4, Marta +1.9, Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 22 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 22 | Marta | OVERFUNCTION | Ravi | Pia | arrives week 23; anxiety Marta +0.5, Pia +0.9 (witness), Ravi +3.8; calmer sender takes some anxiety: Marta +0.2, Ravi -0.2; Marta takes 2.0 of functioning from the other |
| 22 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 22 | Pia | DISTANCE | Ravi | Marta | arrives week 23; anxiety Marta +1.3 (witness), Pia +0.4, Ravi +1.4; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; 5.4 of the sender's anxiety bound into Pia–Ravi |
| 22 | Ravi | DISPLACE | Dr Halim | Marta, Pia | arrives week 23; anxiety Marta +0.9 (witness), Pia +0.4 (witness), Ravi +0.2 |
| 22 | Sofia | CONFLICT | Ravi | Marta, Pia | arrives week 24; anxiety Marta +0.6 (witness), Pia +0.4 (witness), Ravi +4.1, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 23 | Ana | UNDERFUNCTION | Marta | Ravi | arrives week 24; anxiety Ana +0.4, Marta +0.4, Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Marta takes 2.0 of functioning from the other |
| 23 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 23 | Marta | I-POSITION | Ravi | Pia | arrives week 24; anxiety Marta +0.5, Pia +0.9 (witness), Ravi +4.3; calmer sender takes some anxiety: Marta +0.3, Ravi -0.3; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 23 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 23 | Pia | UNDERFUNCTION | Ravi | Marta | arrives week 24; anxiety Marta +1.2 (witness), Pia +0.4, Ravi +0.4; calmer sender takes some anxiety: Pia +1.1, Ravi -1.1; Ravi takes 2.0 of functioning from the other |
| 23 | Ravi | DISTANCE | Marta | Pia | arrives week 24; anxiety Marta +2.1, Pia +0.9 (witness), Ravi +0.6; 11.9 of the sender's anxiety bound into Marta–Ravi |
| 23 | Sofia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 25; anxiety Marta +0.6 (witness), Pia +0.5 (witness), Ravi +2.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Sofia takes 2.0 of functioning from the other |
| 24 | (from outside) | JOB_LOSS | Ravi | Marta, Pia | one-time effect; a 1-week spell, recorded; anxiety Marta +1.5 (witness), Pia +1.1 (witness), Ravi +3.6 |
| 24 | (from outside) | JOB_LOSS | Marta | Ravi | one-time effect; a 1-week spell, recorded; anxiety Marta +3.0, Ravi +1.8 (witness) |
| 24 | Ana | CUTOFF | Marta | Ravi | arrives week 25; anxiety Ana +0.4, Marta +0.3, Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Marta -1.7 |
| 24 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 24 | Dr Halim | TRIANGLE | Marta | Ravi | arrives week 25; anxiety Dr Halim +0.1, Ravi +0.9 (witness); calmer sender takes some anxiety: Dr Halim +0.6, Marta -0.6; Marta left outside: anxiety Dr Halim -0.0, Marta +0.0 (of which 0.0 the outsider's own) |
| 24 | Marta | TRIANGLE | Ravi | Pia | arrives week 25; anxiety Marta +0.5, Pia +0.9 (witness), Ravi -0.0; Ravi left outside: anxiety Marta -2.0, Ravi +3.0 (of which 1.0 the outsider's own) |
| 24 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 24 | Pia | DISTANCE | Ravi | Marta | arrives week 25; anxiety Marta +1.2 (witness), Pia +0.4, Ravi +2.4; calmer sender takes some anxiety: Pia +0.8, Ravi -0.8; 5.0 of the sender's anxiety bound into Pia–Ravi |
| 24 | Ravi | CONFLICT | Marta | Pia | arrives week 25; anxiety Marta +4.4, Pia +0.9 (witness), Ravi +0.6; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1 |
| 24 | Sofia | CONFLICT | Ravi | Marta, Pia | arrives week 26; anxiety Marta +0.6 (witness), Pia +0.5 (witness), Ravi +2.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8 |
| 25 | Ana | REDUCE_CUTOFF | Marta | Ravi | arrives week 26; anxiety Ana +0.4, Marta +0.1, Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana–Marta reopened |
| 25 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 25 | Marta | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 25 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 25 | Pia | DETRIANGLE | Ravi | Marta | arrives week 26; anxiety Marta +1.2 (witness), Pia +0.4, Ravi +0.2; calmer sender takes some anxiety: Pia +1.2, Ravi -1.2 |
| 25 | Ravi | UNDERFUNCTION | Pia | Marta | arrives week 26; anxiety Marta +1.2 (witness), Pia +0.6, Ravi +0.6; Pia takes 2.0 of functioning from the other |
| 25 | Sofia | CONFLICT | Ravi | Marta, Pia | arrives week 27; anxiety Marta +0.6 (witness), Pia +0.4 (witness), Ravi +2.9, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 26 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 26 | Ana | PURSUE | Marta | Ravi | arrives week 27; anxiety Ana +0.4, Marta +1.6, Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 26 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 26 | Marta | UNDERFUNCTION | Ana | Ravi | arrives week 27; anxiety Ana -0.7, Marta +0.3, Ravi +0.8 (witness); Ana takes 2.0 of functioning from the other |
| 26 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 26 | Pia | UNDERFUNCTION | Ravi | Marta | arrives week 27; anxiety Marta +1.2 (witness), Pia +0.4, Ravi -0.1; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Ravi takes 2.0 of functioning from the other |
| 26 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Pia | arrives week 27; anxiety Marta +0.8 (witness), Pia +0.4 (witness), Ravi +0.2 |
| 26 | Sofia | CONFLICT | Ravi | Marta, Pia | arrives week 28; anxiety Marta +0.6 (witness), Pia +0.5 (witness), Ravi +2.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 27 | Ana | STAY-IN-CONTACT | Marta | Ravi | arrives week 28; anxiety Ana +0.4, Marta +0.3, Ravi +0.8 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 27 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 27 | Marta | WITHHOLD | Dr Halim | — | held back DISPLACE: computed, not emitted; attention on the tie rises |
| 27 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 27 | Pia | UNDERFUNCTION | Ravi | Marta | arrives week 28; anxiety Marta +1.2 (witness), Pia +0.4, Ravi -0.3; calmer sender takes some anxiety: Pia +0.9, Ravi -0.9; Ravi takes 2.0 of functioning from the other |
| 27 | Ravi | DISTANCE | Sofia | Marta, Pia | arrives week 29; anxiety Marta +0.6 (witness), Pia +0.5 (witness), Ravi +0.3, Sofia +0.7; 8.1 of the sender's anxiety bound into Ravi–Sofia |
| 27 | Sofia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 29; anxiety Marta +0.6 (witness), Pia +0.5 (witness), Ravi +0.8, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.8, Sofia +0.8; Ravi takes 2.0 of functioning from the other |
| 28 | (from outside) | JOB_LOSS | Ravi | Marta, Pia | one-time effect; a 1-week spell, recorded; anxiety Marta +1.5 (witness), Pia +1.1 (witness), Ravi +3.6 |
| 28 | (from outside) | JOB_LOSS | Marta | Ravi | one-time effect; a 1-week spell, recorded; anxiety Marta +3.0, Ravi +1.8 (witness) |
| 28 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 28 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 28 | Ana | CUTOFF | Marta | Ravi | arrives week 29; anxiety Ana +0.4, Marta +0.3, Ravi +0.7 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Marta -1.0 |
| 28 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 28 | Marta | PURSUE | Ravi | Pia | arrives week 29; anxiety Marta +0.5, Pia +1.0 (witness), Ravi +2.6 |
| 28 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 28 | Pia | UNDERFUNCTION | Ravi | Marta | arrives week 29; anxiety Marta +1.2 (witness), Pia +0.4, Ravi -0.1; calmer sender takes some anxiety: Pia +1.0, Ravi -1.0; Ravi takes 2.0 of functioning from the other |
| 28 | Ravi | DISTANCE | Pia | Marta | arrives week 29; anxiety Marta +1.2 (witness), Pia +2.0, Ravi +0.6; 5.7 of the sender's anxiety bound into Pia–Ravi |
| 28 | Sofia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 30; anxiety Marta +0.6 (witness), Pia +0.5 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi takes 2.0 of functioning from the other |
| 29 | Ana | REDUCE_CUTOFF | Marta | Ravi | arrives week 30; anxiety Ana +0.4, Marta +0.1, Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Ana–Marta reopened |
| 29 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 29 | Marta | UNDERFUNCTION | Ravi | Pia | arrives week 30; anxiety Marta +0.5, Pia +1.0 (witness), Ravi +0.5; Ravi takes 2.0 of functioning from the other |
| 29 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 29 | Pia | WITHHOLD | Ravi | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 29 | Ravi | DISTANCE | Marta | Pia | arrives week 30; anxiety Marta +2.2, Pia +1.0 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.6, Ravi +0.6; 4.1 of the sender's anxiety bound into Marta–Ravi |
| 29 | Sofia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 31; anxiety Marta +0.7 (witness), Pia +0.5 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3; Ravi takes 2.0 of functioning from the other |
| 30 | Ana | DISTANCE | Marta | Ravi | arrives week 31; anxiety Ana +0.4, Marta +0.2, Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; 3.1 of the sender's anxiety bound into Ana–Marta |
| 30 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 30 | Marta | DISTANCE | Ana | Ravi | arrives week 31; anxiety Ana +2.2, Marta +0.3, Ravi +0.6 (witness); 6.0 of the sender's anxiety bound into Ana–Marta |
| 30 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 30 | Pia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 30 | Ravi | CONFLICT | Sofia | Marta, Pia | arrives week 32; anxiety Marta +0.7 (witness), Pia +0.5 (witness), Ravi +0.3, Sofia +2.0 |
| 30 | Sofia | OVERFUNCTION | Ravi | Marta, Pia | arrives week 32; anxiety Marta +0.7 (witness), Pia +0.5 (witness), Ravi +2.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3; Sofia takes 2.0 of functioning from the other |
| 31 | Ana | WITHHOLD | Marta | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 31 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 31 | Marta | OVERFUNCTION | Ravi | Pia | arrives week 32; anxiety Marta +0.5, Pia +1.0 (witness), Ravi +3.9; Marta takes 2.0 of functioning from the other |
| 31 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 31 | Pia | PURSUE | Ravi | Marta | arrives week 32; anxiety Marta +1.3 (witness), Pia +0.4, Ravi +2.2; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2 |
| 31 | Ravi | CONFLICT | Pia | Marta | arrives week 32; anxiety Marta +1.3 (witness), Pia +2.7, Ravi +0.4 |
| 31 | Sofia | CONFLICT | Ravi | Marta, Pia | arrives week 33; anxiety Marta +0.6 (witness), Pia +0.5 (witness), Ravi +3.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6 |
| 32 | (from outside) | JOB_LOSS | Ravi | Marta, Pia | one-time effect; a 1-week spell, recorded; anxiety Marta +1.6 (witness), Pia +1.2 (witness), Ravi +2.7 |
| 32 | (from outside) | JOB_LOSS | Marta | Ravi | one-time effect; a 1-week spell, recorded; anxiety Marta +3.2, Ravi +1.3 (witness) |
| 32 | Ana | CUTOFF | Marta | Ravi | arrives week 33; anxiety Ana +0.4, Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana–Marta now cut off (no events; bond energy kept) |
| 32 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 32 | Dr Halim | CUTOFF | Marta | Ravi | arrives week 33; anxiety Dr Halim +0.1, Ravi +0.8 (witness); calmer sender takes some anxiety: Dr Halim +0.5, Marta -0.5; Dr Halim–Marta now cut off (no events; bond energy kept) |
| 32 | Marta | STAY-IN-CONTACT | Ana | Ravi | arrives week 33; anxiety Ana -2.3, Marta +0.3, Ravi +0.6 (witness) |
| 32 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 32 | Pia | PURSUE | Ravi | Marta | arrives week 33; anxiety Marta +1.2 (witness), Pia +0.4, Ravi +2.6; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 32 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Pia | arrives week 33; anxiety Marta +0.8 (witness), Pia +0.5 (witness), Ravi +0.1 |
| 32 | Sofia | CUTOFF | Ravi | Marta, Pia | arrives week 34; anxiety Marta +0.6 (witness), Pia +0.5 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7; Ravi–Sofia now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -5.1, Sofia -1.1 |
| 33 | Ana | REDUCE_CUTOFF | Marta | Ravi | arrives week 34; anxiety Ana +0.4, Marta +0.1, Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana–Marta reopened |
| 33 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 33 | Marta | CONFLICT | Ravi | Pia | arrives week 34; anxiety Marta +0.5, Pia +1.0 (witness), Ravi +4.4; calmer sender takes some anxiety: Marta +0.1, Ravi -0.1 |
| 33 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 33 | Pia | UNDERFUNCTION | Ravi | Marta | arrives week 34; anxiety Marta +1.2 (witness), Pia +0.4, Ravi +0.4; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Ravi takes 2.0 of functioning from the other |
| 33 | Ravi | FRAME_AMBIGUITY | Dr Halim | Marta, Pia | arrives week 34; anxiety Dr Halim +0.3, Marta +0.8 (witness), Pia +0.5 (witness), Ravi +0.1 |
| 33 | Sofia | CUTOFF | Ravi | Marta, Pia | arrives week 35; anxiety Marta +0.6 (witness), Pia +0.5 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6; Ravi–Sofia now cut off (no events; bond energy kept) |
| 34 | Ana | PURSUE | Marta | Ravi | arrives week 35; anxiety Ana +0.4, Marta +1.5, Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6 |
| 34 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 34 | Marta | CUTOFF | Ana | Ravi | arrives week 35; anxiety Marta +0.3, Ravi +0.6 (witness); Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Marta -1.5 |
| 34 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 34 | Pia | CONFLICT | Ravi | Marta | arrives week 35; anxiety Marta +1.2 (witness), Pia +0.4, Ravi +4.4; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4 |
| 34 | Ravi | OVERFUNCTION | Marta | Pia | arrives week 35; anxiety Marta +3.5, Pia +1.0 (witness), Ravi +0.5; calmer sender takes some anxiety: Marta -0.1, Ravi +0.1; Ravi takes 2.0 of functioning from the other |
| 34 | Sofia | REDUCE_CUTOFF | Ravi | Marta, Pia | arrives week 36; anxiety Marta +0.7 (witness), Pia +0.5 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.6, Sofia +0.6; Ravi–Sofia reopened |
| 35 | Ana | REDUCE_CUTOFF | Marta | Ravi | arrives week 36; anxiety Ana +0.4, Marta +0.1, Ravi +0.6 (witness); calmer sender takes some anxiety: Ana +0.6, Marta -0.6; Ana–Marta reopened |
| 35 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 35 | Marta | UNDERFUNCTION | Ravi | Pia | arrives week 36; anxiety Marta +0.5, Pia +1.0 (witness), Ravi +0.5; Ravi takes 2.0 of functioning from the other |
| 35 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 35 | Pia | UNDERFUNCTION | Ravi | Marta | arrives week 36; anxiety Marta +1.3 (witness), Pia +0.4, Ravi +0.5; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; Ravi takes 2.0 of functioning from the other |
| 35 | Ravi | WITHHOLD | Pia | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 35 | Sofia | REDUCE_CUTOFF | Ravi | Marta, Pia | arrives week 37; anxiety Marta +0.7 (witness), Pia +0.5 (witness), Ravi +0.7, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.7, Sofia +0.7 |
| 36 | (from outside) | JOB_LOSS | Ravi | Marta, Pia | one-time effect; a 1-week spell, recorded; anxiety Marta +1.6 (witness), Pia +1.2 (witness), Ravi +2.7 |
| 36 | (from outside) | JOB_LOSS | Marta | Ravi | one-time effect; a 1-week spell, recorded; anxiety Marta +3.2, Ravi +1.3 (witness) |
| 36 | Ana | DISTANCE | Marta | Ravi | arrives week 37; anxiety Ana +0.4, Marta +0.2, Ravi +0.5 (witness); calmer sender takes some anxiety: Ana +0.7, Marta -0.7; 2.1 of the sender's anxiety bound into Ana–Marta |
| 36 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 36 | Marta | CUTOFF | Ravi | Pia | arrives week 37; anxiety Marta +0.6, Pia +1.1 (witness), Ravi +0.4; Marta–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Marta -2.1, Ravi -2.2 |
| 36 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 36 | Pia | DISTANCE | Ravi | Marta | arrives week 37; anxiety Marta +1.4 (witness), Pia +0.4, Ravi +0.4; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; 6.4 of the sender's anxiety bound into Pia–Ravi |
| 36 | Ravi | OVERFUNCTION | Pia | Marta | arrives week 37; anxiety Marta +1.4 (witness), Pia +3.4, Ravi +0.4; Ravi takes 2.0 of functioning from the other |
| 36 | Sofia | STAY-IN-CONTACT | Ravi | Marta, Pia | arrives week 38; anxiety Marta +0.7 (witness), Pia +0.6 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5 |
| 37 | Ana | DISTANCE | Marta | — | arrives week 38; anxiety Ana +0.4, Marta +0.2; calmer sender takes some anxiety: Ana +0.7, Marta -0.7; 2.4 of the sender's anxiety bound into Ana–Marta |
| 37 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 37 | Marta | DISTANCE | Ana | — | arrives week 38; anxiety Ana +2.3, Marta +0.3; 6.0 of the sender's anxiety bound into Ana–Marta |
| 37 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 37 | Pia | CONFLICT | Ravi | — | arrives week 38; anxiety Pia +0.5, Ravi +3.4; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 37 | Ravi | UNDERFUNCTION | Sofia | Pia | arrives week 39; anxiety Pia +0.6 (witness), Ravi +0.2, Sofia -0.6; Sofia takes 2.0 of functioning from the other |
| 37 | Sofia | WITHHOLD | Ravi | — | held back CONFLICT: computed, not emitted; attention on the tie rises |
| 37 | — | (system) | — | — | Ana–Marta tie now distant |
| 38 | Ana | CONFLICT | Marta | — | arrives week 39; anxiety Ana +0.4, Marta +3.3; calmer sender takes some anxiety: Ana +0.4, Marta -0.4 |
| 38 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 38 | Marta | OVERFUNCTION | Ana | — | arrives week 39; anxiety Ana +1.8, Marta +0.3; Marta takes 2.0 of functioning from the other |
| 38 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 38 | Pia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 38 | Ravi | DISTANCE | Sofia | Pia | arrives week 40; anxiety Pia +0.6 (witness), Ravi +0.2, Sofia +1.6; 4.9 of the sender's anxiety bound into Ravi–Sofia |
| 38 | Sofia | PURSUE | Ravi | Pia | arrives week 40; anxiety Pia +0.6 (witness), Ravi +1.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5 |
| 39 | Ana | DISTANCE | Marta | — | arrives week 40; anxiety Ana +0.4, Marta +0.6; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; 3.0 of the sender's anxiety bound into Ana–Marta |
| 39 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 39 | Marta | DISTANCE | Ana | — | arrives week 40; anxiety Ana +1.7, Marta +0.3; 5.2 of the sender's anxiety bound into Ana–Marta |
| 39 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 39 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 39 | Ravi | WITHHOLD | Dr Halim | — | held back DISPLACE: computed, not emitted; attention on the tie rises |
| 39 | Sofia | UNDERFUNCTION | Ravi | Pia | arrives week 41; anxiety Pia +0.6 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi takes 2.0 of functioning from the other |
| 40 | (from outside) | JOB_LOSS | Ravi | Pia | one-time effect; a 1-week spell, recorded; anxiety Pia +1.4 (witness), Ravi +2.4 |
| 40 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +3.2 |
| 40 | Ana | CUTOFF | Marta | — | arrives week 41; anxiety Ana +0.4, Marta +0.3; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -1.8, Marta -1.7 |
| 40 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 40 | Dr Halim | STAY-IN-CONTACT | Ravi | Pia | arrives week 41; anxiety Dr Halim +0.1, Pia +0.6 (witness); calmer sender takes some anxiety: Dr Halim +0.3, Ravi -0.3 |
| 40 | Marta | UNDERFUNCTION | Ana | — | arrives week 41; anxiety Ana -0.6, Marta +0.3; Ana takes 2.0 of functioning from the other |
| 40 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 40 | Pia | CONFLICT | Ravi | — | arrives week 41; anxiety Pia +0.5, Ravi +4.0; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3 |
| 40 | Ravi | TRIANGLE | Pia | — | arrives week 41; anxiety Pia +0.3, Ravi +0.4; Pia left outside: anxiety Pia +1.5, Ravi -1.0 (of which 0.5 the outsider's own) |
| 40 | Sofia | WITHHOLD | Ravi | — | held back DISTANCE: computed, not emitted; attention on the tie rises |
| 41 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 41 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 41 | Ana | (holds) | — | — | fallback (hold): no legal act |
| 41 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 41 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 42; anxiety Ana -1.8, Marta +0.3; Ana–Marta reopened |
| 41 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 41 | Pia | PREVENT_ALIGNMENT | Ravi | — | arrives week 42; anxiety Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3 |
| 41 | Ravi | CUTOFF | Pia | — | arrives week 42; anxiety Pia +0.5, Ravi +0.4; Pia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Pia -1.2, Ravi -4.0 |
| 41 | Sofia | DISTANCE | Ravi | Pia | arrives week 43; anxiety Pia +0.6 (witness), Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3; 1.9 of the sender's anxiety bound into Ravi–Sofia |
| 42 | Ana | DISTANCE | Marta | — | arrives week 43; anxiety Ana +0.4, Marta +2.2; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; 1.1 of the sender's anxiety bound into Ana–Marta |
| 42 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 42 | Marta | CUTOFF | Ana | — | arrives week 43; anxiety Ana +0.3, Marta +0.3; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -0.3 |
| 42 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 42 | Pia | REDUCE_CUTOFF | Ravi | — | arrives week 43; anxiety Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; Pia–Ravi reopened; bound anxiety released to the third party: Marta +43.4 |
| 42 | Ravi | CONFLICT | Sofia | — | arrives week 44; anxiety Ravi +0.2, Sofia +1.7 |
| 42 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 43 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 44; anxiety Ana +0.4, Marta +0.1; calmer sender takes some anxiety: Ana +1.5, Marta -1.5; Ana–Marta reopened |
| 43 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 43 | Marta | (holds) | — | — | fallback (hold): no legal act |
| 43 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 43 | Pia | PURSUE | Ravi | — | arrives week 44; anxiety Pia +0.5, Ravi +2.3; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2 |
| 43 | Ravi | CONFLICT | Pia | — | arrives week 44; anxiety Pia +4.3, Ravi +0.4 |
| 43 | Sofia | I-POSITION | Ravi | Pia | arrives week 45; anxiety Pia +0.6 (witness), Ravi +2.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 44 | (from outside) | JOB_LOSS | Ravi | Pia | one-time effect; a 1-week spell, recorded; anxiety Pia +1.4 (witness), Ravi +2.4 |
| 44 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +3.3 |
| 44 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 44 | Ana | UNDERFUNCTION | Marta | — | arrives week 45; anxiety Ana +0.4, Marta +0.1; calmer sender takes some anxiety: Ana +1.3, Marta -1.3; Marta takes 2.0 of functioning from the other |
| 44 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 44 | Marta | DISTANCE | Ana | — | arrives week 45; anxiety Ana +2.4, Marta +0.3; 11.3 of the sender's anxiety bound into Ana–Marta |
| 44 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 44 | Pia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 44 | Ravi | STAY-IN-CONTACT | Sofia | Pia | arrives week 46; anxiety Pia +0.6 (witness), Ravi +0.2, Sofia -0.1 |
| 44 | Sofia | CONFLICT | Ravi | Pia | arrives week 46; anxiety Pia +0.6 (witness), Ravi +2.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 45 | Ana | UNDERFUNCTION | Marta | — | arrives week 46; anxiety Ana +0.4, Marta -0.4; calmer sender takes some anxiety: Ana +0.7, Marta -0.7; Marta takes 2.0 of functioning from the other |
| 45 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 45 | Marta | WITHHOLD | Ana | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 45 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 45 | Pia | CONFLICT | Ravi | — | arrives week 46; anxiety Pia +0.5, Ravi +4.1; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3 |
| 45 | Ravi | TRIANGLE | Pia | — | arrives week 46; anxiety Pia +0.7, Ravi +0.4; Pia left outside: anxiety Pia +1.8, Ravi -1.2 (of which 0.6 the outsider's own) |
| 45 | Sofia | OVERFUNCTION | Ravi | Pia | arrives week 47; anxiety Pia +0.6 (witness), Ravi +2.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Sofia takes 2.0 of functioning from the other |
| 46 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 46 | — | (system) | — | — | Marta–Pia–Ravi: inside pair Marta–Ravi |
| 46 | Ana | CUTOFF | Marta | — | arrives week 47; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana–Marta now cut off (no events; bond energy kept) |
| 46 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 46 | Marta | UNDERFUNCTION | Ana | — | arrives week 47; anxiety Ana -1.1, Marta +0.3; Ana takes 2.0 of functioning from the other |
| 46 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 46 | Pia | DISTANCE | Ravi | — | arrives week 47; anxiety Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3; 4.0 of the sender's anxiety bound into Pia–Ravi |
| 46 | Ravi | CUTOFF | Pia | — | arrives week 47; anxiety Pia +0.6, Ravi +0.4; Pia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Pia -2.1, Ravi -3.9 |
| 46 | Sofia | DISTANCE | Ravi | Pia | arrives week 48; anxiety Pia +0.6 (witness), Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; 1.8 of the sender's anxiety bound into Ravi–Sofia |
| 47 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 48; anxiety Ana +0.4, Marta -2.0; calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana–Marta reopened |
| 47 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 47 | Marta | (holds) | — | — | fallback (hold): no legal act |
| 47 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 47 | Pia | REDUCE_CUTOFF | Ravi | — | arrives week 48; anxiety Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; Pia–Ravi reopened; bound anxiety released to the third party: Marta +4.0 |
| 47 | Ravi | WITHHOLD | Sofia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 47 | Sofia | UNDERFUNCTION | Ravi | — | arrives week 49; anxiety Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Ravi takes 2.0 of functioning from the other |
| 48 | (from outside) | JOB_LOSS | Ravi | — | one-time effect; a 1-week spell, recorded; anxiety Ravi +2.4 |
| 48 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +3.2 |
| 48 | Ana | PURSUE | Marta | — | arrives week 49; anxiety Ana +0.4, Marta +1.4; calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 48 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 48 | Dr Halim | OVERFUNCTION | Ravi | Pia | arrives week 49; anxiety Dr Halim +0.1, Pia +0.6 (witness), Ravi +0.4; calmer sender takes some anxiety: Dr Halim +0.4, Ravi -0.4; Dr Halim takes 2.0 of functioning from the other; the contact lands with Ravi: systems perspective +0.10 |
| 48 | Marta | UNDERFUNCTION | Ana | — | arrives week 49; anxiety Ana -1.0, Marta +0.3; Ana takes 2.0 of functioning from the other |
| 48 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 48 | Pia | CONFLICT | Ravi | — | arrives week 49; anxiety Pia +0.5, Ravi +4.3; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5 |
| 48 | Ravi | WITHHOLD | Pia | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 48 | Sofia | UNDERFUNCTION | Ravi | Pia | arrives week 50; anxiety Pia +0.6 (witness), Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Ravi takes 2.0 of functioning from the other |
| 49 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 49 | Ana | CONFLICT | Marta | — | arrives week 50; anxiety Ana +0.4, Marta +3.4; calmer sender takes some anxiety: Ana +0.5, Marta -0.5 |
| 49 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 49 | Marta | DISTANCE | Ana | — | arrives week 50; anxiety Ana +2.3, Marta +0.3; 5.7 of the sender's anxiety bound into Ana–Marta |
| 49 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 49 | Pia | UNDERFUNCTION | Ravi | — | arrives week 50; anxiety Pia +0.5, Ravi +0.6; calmer sender takes some anxiety: Pia +0.6, Ravi -0.6; Ravi takes 2.0 of functioning from the other |
| 49 | Ravi | CUTOFF | Pia | — | arrives week 50; anxiety Ravi +0.4; Pia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -3.6 |
| 49 | Sofia | CONFLICT | Ravi | Pia | arrives week 51; anxiety Pia +0.6 (witness), Ravi +2.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 50 | Ana | UNDERFUNCTION | Marta | — | arrives week 51; anxiety Ana +0.4, Marta +0.3; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Marta takes 2.0 of functioning from the other |
| 50 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 50 | Marta | OVERFUNCTION | Ana | — | arrives week 51; anxiety Ana +1.7, Marta +0.3; Marta takes 2.0 of functioning from the other |
| 50 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 50 | Pia | REDUCE_CUTOFF | Ravi | — | arrives week 51; anxiety Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3; Pia–Ravi reopened |
| 50 | Ravi | DISPLACE | Dr Halim | — | arrives week 51; anxiety Ravi +0.1 |
| 50 | Sofia | PURSUE | Ravi | — | arrives week 52; anxiety Ravi +1.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 51 | Ana | CUTOFF | Marta | — | arrives week 52; anxiety Ana +0.4, Marta +0.3; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -1.9, Marta -2.0 |
| 51 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 51 | Marta | WITHHOLD | Ana | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 51 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 51 | Pia | PURSUE | Ravi | — | arrives week 52; anxiety Pia +0.5, Ravi +2.2; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4 |
| 51 | Ravi | PREVENT_ALIGNMENT | Pia | — | arrives week 52; anxiety Ravi +0.4 |
| 51 | Sofia | UNDERFUNCTION | Ravi | Pia | arrives week 53; anxiety Pia +0.6 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Ravi takes 2.0 of functioning from the other |
| 51 | — | (system) | — | — | slow tick (yearly) fired — nothing runs on it in Phase B |
| 52 | (from outside) | JOB_LOSS | Ravi | Pia | one-time effect; a 1-week spell, recorded; anxiety Pia +1.4 (witness), Ravi +2.3 |
| 52 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +3.0 |
| 52 | Ana | (holds) | — | — | fallback (hold): no legal act |
| 52 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 52 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 53; anxiety Ana -2.2, Marta +0.3; Ana–Marta reopened |
| 52 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 52 | Pia | CUTOFF | Ravi | — | arrives week 53; anxiety Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +0.5, Ravi -0.5; Pia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -2.0 |
| 52 | Ravi | DISTANCE | Sofia | Pia | arrives week 54; anxiety Pia +0.6 (witness), Ravi +0.2, Sofia +1.9; 2.7 of the sender's anxiety bound into Ravi–Sofia |
| 52 | Sofia | UNDERFUNCTION | Ravi | Pia | arrives week 54; anxiety Pia +0.6 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi takes 2.0 of functioning from the other |
| 53 | Ana | UNDERFUNCTION | Marta | — | arrives week 54; anxiety Ana +0.4; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Marta takes 2.0 of functioning from the other |
| 53 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 53 | Marta | STAY-IN-CONTACT | Ana | — | arrives week 54; anxiety Ana +0.3, Marta +0.3 |
| 53 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 53 | Pia | REDUCE_CUTOFF | Ravi | — | arrives week 54; anxiety Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia–Ravi reopened |
| 53 | Ravi | WITHHOLD | Sofia | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 53 | Sofia | CUTOFF | Ravi | — | arrives week 55; anxiety Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.2, Sofia +0.2; Ravi–Sofia now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -1.4 |
| 54 | Ana | I-POSITION | Marta | — | arrives week 55; anxiety Ana +0.4, Marta +2.3; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 54 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 54 | Marta | OVERFUNCTION | Ana | — | arrives week 55; anxiety Ana +3.7, Marta +0.3; Marta takes 2.0 of functioning from the other |
| 54 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 54 | Pia | CUTOFF | Ravi | — | arrives week 55; anxiety Pia +0.5, Ravi +0.4; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2; Pia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -0.5 |
| 54 | Ravi | SPLIT | Dr Halim | Pia | arrives week 55; anxiety Pia +0.6 (witness), Ravi +0.1 |
| 54 | Sofia | PURSUE | Ravi | Pia | arrives week 56; anxiety Pia +0.6 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.2, Sofia +0.2 |
| 55 | Ana | UNDERFUNCTION | Marta | — | arrives week 56; anxiety Ana +0.5, Marta +0.3; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Marta takes 2.0 of functioning from the other |
| 55 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 55 | Marta | CONFLICT | Ana | — | arrives week 56; anxiety Ana +4.9, Marta +0.3 |
| 55 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 55 | Pia | REDUCE_CUTOFF | Ravi | — | arrives week 56; anxiety Pia +0.5, Ravi +0.3; calmer sender takes some anxiety: Pia +0.1, Ravi -0.1; Pia–Ravi reopened |
| 55 | Ravi | SPLIT | Dr Halim | — | arrives week 56; anxiety Dr Halim +0.1, Ravi +0.1 |
| 55 | Sofia | (holds) | — | — | fallback (hold): no legal act |
| 56 | (from outside) | JOB_LOSS | Ravi | — | one-time effect; a 1-week spell, recorded; anxiety Ravi +2.1 |
| 56 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +2.7 |
| 56 | Ana | OVERFUNCTION | Marta | — | arrives week 57; anxiety Ana +0.5, Marta +2.3; calmer sender takes some anxiety: Ana +0.2, Marta -0.2; Ana takes 2.0 of functioning from the other |
| 56 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 56 | Dr Halim | STAY-IN-CONTACT | Ravi | Pia | arrives week 57; anxiety Dr Halim +0.1, Pia +0.6 (witness); calmer sender takes some anxiety: Dr Halim +0.3, Ravi -0.3 |
| 56 | Marta | UNDERFUNCTION | Ana | — | arrives week 57; anxiety Ana +0.6, Marta +0.3; Ana takes 2.0 of functioning from the other |
| 56 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 56 | Pia | CONFLICT | Ravi | — | arrives week 57; anxiety Pia +0.5, Ravi +3.7; calmer sender takes some anxiety: Pia +0.2, Ravi -0.2 |
| 56 | Ravi | DISPLACE | Dr Halim | Pia | arrives week 57; anxiety Dr Halim +0.3, Pia +0.6 (witness), Ravi +0.1 |
| 56 | Sofia | (holds) | — | — | fallback (hold): no legal act |
| 57 | Ana | CONFLICT | Marta | — | arrives week 58; anxiety Ana +0.4, Marta +3.0; calmer sender takes some anxiety: Ana +0.2, Marta -0.2 |
| 57 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 57 | Marta | WITHHOLD | Ana | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 57 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 57 | Pia | OVERFUNCTION | Ravi | — | arrives week 58; anxiety Pia +0.5, Ravi +2.2; calmer sender takes some anxiety: Pia +0.3, Ravi -0.3; Pia takes 2.0 of functioning from the other |
| 57 | Ravi | REDUCE_CUTOFF | Sofia | Pia | arrives week 59; anxiety Pia +0.6 (witness), Ravi +0.2, Sofia -1.9; Ravi–Sofia reopened |
| 57 | Sofia | (holds) | — | — | fallback (hold): no legal act |
| 58 | Ana | CONFLICT | Marta | — | arrives week 59; anxiety Ana +0.4, Marta +3.0; calmer sender takes some anxiety: Ana +0.3, Marta -0.3 |
| 58 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 58 | Marta | UNDERFUNCTION | Ana | — | arrives week 59; anxiety Ana +0.4, Marta +0.3; Ana takes 2.0 of functioning from the other |
| 58 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 58 | Pia | CUTOFF | Ravi | — | arrives week 59; anxiety Pia +0.5, Ravi +0.5; calmer sender takes some anxiety: Pia +0.4, Ravi -0.4; Pia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -3.8 |
| 58 | Ravi | DISTANCE | Pia | — | arrives week 59; anxiety Pia +2.7, Ravi +0.4; 1.8 of the sender's anxiety bound into Pia–Ravi |
| 58 | Sofia | (holds) | — | — | fallback (hold): no legal act |
| 59 | Ana | CUTOFF | Marta | — | arrives week 60; anxiety Ana +0.4, Marta +0.5; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -1.9, Marta -5.2 |
| 59 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 59 | Marta | DISTANCE | Ana | — | arrives week 60; anxiety Ana +0.3, Marta +0.3; 3.6 of the sender's anxiety bound into Ana–Marta |
| 59 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 59 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 59 | Ravi | STAY-IN-CONTACT | Sofia | — | arrives week 61; anxiety Ravi +0.2, Sofia -0.3 |
| 59 | Sofia | STAY-IN-CONTACT | Ravi | — | arrives week 61; anxiety Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 60 | (from outside) | JOB_LOSS | Ravi | — | one-time effect; a 1-week spell, recorded; anxiety Ravi +2.2 |
| 60 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +3.0 |
| 60 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 61; anxiety Ana +0.4, Marta +0.3; calmer sender takes some anxiety: Ana +0.2, Marta -0.2; Ana–Marta reopened |
| 60 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 60 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 61; anxiety Ana +0.4, Marta +0.3 |
| 60 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 60 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 60 | Ravi | SPLIT | Dr Halim | — | arrives week 61; anxiety Ravi +0.1 |
| 60 | Sofia | CUTOFF | Ravi | — | arrives week 62; anxiety Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia now cut off (no events; bond energy kept) |
| 61 | Ana | CUTOFF | Marta | — | arrives week 62; anxiety Ana +0.4, Marta +0.4; calmer sender takes some anxiety: Ana +0.2, Marta -0.2; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -3.6, Marta -0.5 |
| 61 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 61 | Marta | OVERFUNCTION | Ana | — | arrives week 62; anxiety Ana +3.6, Marta +0.3; Marta takes 2.0 of functioning from the other |
| 61 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 61 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 61 | Ravi | PURSUE | Sofia | — | arrives week 63; anxiety Ravi +0.2, Sofia +1.3 |
| 61 | Sofia | UNDERFUNCTION | Ravi | — | arrives week 63; anxiety Sofia +0.3; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3; Ravi takes 2.0 of functioning from the other |
| 62 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 63; anxiety Ana +0.4, Marta +0.2; calmer sender takes some anxiety: Ana +0.2, Marta -0.2; Ana–Marta reopened |
| 62 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 62 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 63; anxiety Ana +0.5, Marta +0.3 |
| 62 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 62 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 62 | Ravi | SPLIT | Dr Halim | — | arrives week 63; anxiety Dr Halim +0.1, Ravi +0.1 |
| 62 | Sofia | REDUCE_CUTOFF | Ravi | — | arrives week 64; anxiety Sofia +0.4; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3; Ravi–Sofia reopened |
| 63 | Ana | CUTOFF | Marta | — | arrives week 64; anxiety Ana +0.4, Marta +0.3; calmer sender takes some anxiety: Ana +0.2, Marta -0.2; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -3.4, Marta -0.3 |
| 63 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 63 | Marta | PROVOKE | Ana | — | arrives week 64; anxiety Ana +3.2, Marta +0.3 |
| 63 | Nadia | (holds) | — | — | fallback (hold): no legal act |
| 63 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 63 | Ravi | REDUCE_CUTOFF | Nadia | — | arrives week 64; anxiety Nadia -2.8, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Nadia–Ravi reopened; bound anxiety released to the third party: Marta +9.8 |
| 63 | Sofia | REDUCE_CUTOFF | Ravi | — | arrives week 65; anxiety Ravi +0.2, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3 |
| 64 | (from outside) | JOB_LOSS | Ravi | — | one-time effect; a 1-week spell, recorded; anxiety Ravi +2.1 |
| 64 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +2.9 |
| 64 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 65; anxiety Ana +0.4, Marta +0.2; calmer sender takes some anxiety: Ana +0.5, Marta -0.5; Ana–Marta reopened |
| 64 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 64 | Dr Halim | TRIANGLE | Ravi | Nadia | arrives week 65; anxiety Dr Halim +0.1, Nadia +0.7 (witness); calmer sender takes some anxiety: Dr Halim +0.3, Ravi -0.3; Ravi left outside: anxiety Dr Halim -0.0, Ravi +0.0 (of which 0.0 the outsider's own); the contact lands with Ravi: systems perspective +0.09 |
| 64 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 65; anxiety Ana +0.6, Marta +0.3 |
| 64 | Nadia | PURSUE | Ravi | — | arrives week 65; anxiety Nadia +0.5, Ravi -0.9 |
| 64 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 64 | Ravi | WITHHOLD | Dr Halim | — | held back FRAME_AMBIGUITY: computed, not emitted; attention on the tie rises |
| 64 | Sofia | STAY-IN-CONTACT | Ravi | Nadia | arrives week 66; anxiety Nadia +0.7 (witness), Ravi +0.1, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3 |
| 65 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle active |
| 65 | — | (system) | — | — | Dr Halim–Marta–Ravi: inside pair Dr Halim–Marta |
| 65 | Ana | CUTOFF | Marta | — | arrives week 66; anxiety Ana +0.4, Marta +0.2; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -0.7, Marta -0.2 |
| 65 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 65 | Marta | DISTANCE | Ana | — | arrives week 66; anxiety Ana +0.5, Marta +0.3; 4.2 of the sender's anxiety bound into Ana–Marta |
| 65 | Nadia | STAY-IN-CONTACT | Ravi | — | arrives week 66; anxiety Nadia +0.5, Ravi +0.1 |
| 65 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 65 | Ravi | DISTANCE | Sofia | Nadia | arrives week 67; anxiety Nadia +0.7 (witness), Ravi +0.2, Sofia +1.1; 2.1 of the sender's anxiety bound into Ravi–Sofia |
| 65 | Sofia | WITHHOLD | Ravi | — | held back CUTOFF: computed, not emitted; attention on the tie rises |
| 66 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 67; anxiety Ana +0.4, Marta +0.1; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta reopened |
| 66 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 66 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 67; anxiety Ana +0.5, Marta +0.3 |
| 66 | Nadia | PURSUE | Ravi | — | arrives week 67; anxiety Nadia +0.5, Ravi +1.7 |
| 66 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 66 | Ravi | CONFLICT | Nadia | — | arrives week 67; anxiety Nadia +5.0, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3 |
| 66 | Sofia | CUTOFF | Ravi | Nadia | arrives week 68; anxiety Nadia +0.7 (witness), Sofia +0.4; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3; Ravi–Sofia now cut off (no events; bond energy kept); impingement removed at once, anxiety Sofia -0.0 |
| 67 | Ana | CUTOFF | Marta | — | arrives week 68; anxiety Ana +0.4, Marta +0.2; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -0.6, Marta -0.2 |
| 67 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 67 | Marta | STAY-IN-CONTACT | Ana | — | arrives week 68; anxiety Ana +0.4, Marta +0.3 |
| 67 | Nadia | TRIANGLE | Ravi | — | arrives week 68; anxiety Nadia +0.5, Ravi +0.1; Ravi left outside: anxiety Nadia -0.8, Ravi +1.1 (of which 0.4 the outsider's own) |
| 67 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 67 | Ravi | CONFLICT | Sofia | Nadia | arrives week 69; anxiety Nadia +0.7 (witness), Ravi +0.2, Sofia +2.1 |
| 67 | Sofia | DISTANCE | Ravi | Nadia | arrives week 69; anxiety Nadia +0.7 (witness), Sofia +0.4; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3; 2.3 of the sender's anxiety bound into Ravi–Sofia |
| 68 | (from outside) | JOB_LOSS | Ravi | Nadia | one-time effect; a 1-week spell, recorded; anxiety Nadia +1.6 (witness), Ravi +2.1 |
| 68 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +2.9 |
| 68 | — | (system) | — | — | Dr Halim–Marta–Ravi triangle inactive |
| 68 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 68 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 68 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 69; anxiety Ana +0.4, Marta +0.1; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta reopened |
| 68 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 68 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 69; anxiety Ana +0.5, Marta +0.3 |
| 68 | Nadia | PURSUE | Ravi | — | arrives week 69; anxiety Nadia +0.5, Ravi +1.7 |
| 68 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 68 | Ravi | SPLIT | Dr Halim | Nadia | arrives week 69; anxiety Nadia +0.7 (witness), Ravi +0.1 |
| 68 | Sofia | REDUCE_CUTOFF | Ravi | Nadia | arrives week 70; anxiety Nadia +0.7 (witness), Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia reopened |
| 69 | Ana | CONFLICT | Marta | — | arrives week 70; anxiety Ana +0.4, Marta +2.8; calmer sender takes some anxiety: Ana +0.3, Marta -0.3 |
| 69 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 69 | Marta | STAY-IN-CONTACT | Ana | — | arrives week 70; anxiety Ana +0.4, Marta +0.3 |
| 69 | Nadia | OVERFUNCTION | Ravi | — | arrives week 70; anxiety Nadia +0.5, Ravi +2.9; Nadia takes 2.0 of functioning from the other |
| 69 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 69 | Ravi | WITHHOLD | Nadia | — | held back TRIANGLE: computed, not emitted; attention on the tie rises |
| 69 | Sofia | (holds) | — | — | fallback (hold): no legal act |
| 70 | Ana | I-POSITION | Marta | — | arrives week 71; anxiety Ana +0.4, Marta +2.9; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 70 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 70 | Marta | UNDERFUNCTION | Ana | — | arrives week 71; anxiety Ana +0.3, Marta +0.3; Ana takes 2.0 of functioning from the other |
| 70 | Nadia | UNDERFUNCTION | Ravi | — | arrives week 71; anxiety Nadia +0.5, Ravi +0.3; Ravi takes 2.0 of functioning from the other |
| 70 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 70 | Ravi | DETRIANGLE | Nadia | — | arrives week 71; anxiety Nadia +0.6, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.4, Ravi +0.4 |
| 70 | Sofia | CUTOFF | Ravi | Nadia | arrives week 72; anxiety Nadia +0.7 (witness), Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia now cut off (no events; bond energy kept); impingement removed at once, anxiety Sofia -0.7 |
| 71 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 71 | Ana | CUTOFF | Marta | — | arrives week 72; anxiety Ana +0.4, Marta +0.4; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -0.4, Marta -3.7 |
| 71 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 71 | Marta | DISTANCE | Ana | — | arrives week 72; anxiety Ana +0.3, Marta +0.3; 3.4 of the sender's anxiety bound into Ana–Marta |
| 71 | Nadia | CONFLICT | Ravi | — | arrives week 72; anxiety Nadia +0.5, Ravi +2.5 |
| 71 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 71 | Ravi | CUTOFF | Nadia | — | arrives week 72; anxiety Nadia +0.6, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.4, Ravi +0.4; Nadia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Nadia -1.5, Ravi -4.7 |
| 71 | Sofia | CUTOFF | Ravi | Nadia | arrives week 73; anxiety Nadia +0.7 (witness), Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia now cut off (no events; bond energy kept) |
| 72 | (from outside) | JOB_LOSS | Ravi | Nadia | one-time effect; a 1-week spell, recorded; anxiety Nadia +1.6 (witness), Ravi +2.1 |
| 72 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +3.0 |
| 72 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 73; anxiety Ana +0.4, Marta +0.3; calmer sender takes some anxiety: Ana +0.2, Marta -0.2; Ana–Marta reopened |
| 72 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 72 | Dr Halim | OVERFUNCTION | Ravi | — | arrives week 73; anxiety Dr Halim +0.1, Ravi +0.2; calmer sender takes some anxiety: Dr Halim +0.3, Ravi -0.3; Dr Halim takes 2.0 of functioning from the other |
| 72 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 73; anxiety Ana +0.3, Marta +0.3 |
| 72 | Nadia | REDUCE_CUTOFF | Ravi | — | arrives week 73; anxiety Nadia +0.5, Ravi +0.3; Nadia–Ravi reopened |
| 72 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 72 | Ravi | REDUCE_CUTOFF | Nadia | — | arrives week 73; anxiety Nadia +0.8, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.4, Ravi +0.4 |
| 72 | Sofia | REDUCE_CUTOFF | Ravi | — | arrives week 74; anxiety Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia reopened |
| 73 | Ana | CONFLICT | Marta | — | arrives week 74; anxiety Ana +0.4, Marta +3.2; calmer sender takes some anxiety: Ana +0.2, Marta -0.2 |
| 73 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 73 | Marta | UNDERFUNCTION | Ana | — | arrives week 74; anxiety Ana +0.3, Marta +0.3; Ana takes 2.0 of functioning from the other |
| 73 | Nadia | STAY-IN-CONTACT | Ravi | — | arrives week 74; anxiety Nadia +0.5, Ravi +0.3 |
| 73 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 73 | Ravi | UNDERFUNCTION | Nadia | — | arrives week 74; anxiety Nadia +0.5, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.5, Ravi +0.5; Nadia takes 2.0 of functioning from the other |
| 73 | Sofia | REDUCE_CUTOFF | Ravi | Nadia | arrives week 75; anxiety Nadia +0.6 (witness), Ravi +0.2, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 74 | Ana | UNDERFUNCTION | Marta | — | arrives week 75; anxiety Ana +0.4, Marta +0.5; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Marta takes 2.0 of functioning from the other |
| 74 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 74 | Marta | OVERFUNCTION | Ana | — | arrives week 75; anxiety Ana +3.2, Marta +0.3; Marta takes 2.0 of functioning from the other |
| 74 | Nadia | CUTOFF | Ravi | — | arrives week 75; anxiety Nadia +0.5, Ravi +0.3; Nadia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Nadia -0.8, Ravi -0.5 |
| 74 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 74 | Ravi | CUTOFF | Nadia | — | arrives week 75; anxiety Nadia +0.4, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.4, Ravi +0.4; Nadia–Ravi now cut off (no events; bond energy kept) |
| 74 | Sofia | PURSUE | Ravi | Nadia | arrives week 76; anxiety Nadia +0.6 (witness), Ravi +1.1, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 75 | Ana | DISTANCE | Marta | — | arrives week 76; anxiety Ana +0.4, Marta +0.4; calmer sender takes some anxiety: Ana +0.2, Marta -0.2; 3.3 of the sender's anxiety bound into Ana–Marta |
| 75 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 75 | Marta | DISTANCE | Ana | — | arrives week 76; anxiety Ana +0.4, Marta +0.3; 3.8 of the sender's anxiety bound into Ana–Marta |
| 75 | Nadia | REDUCE_CUTOFF | Ravi | — | arrives week 76; anxiety Nadia +0.5, Ravi +0.3; Nadia–Ravi reopened |
| 75 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 75 | Ravi | UNDERFUNCTION | Sofia | — | arrives week 77; anxiety Ravi +0.2, Sofia -0.9; Sofia takes 2.0 of functioning from the other |
| 75 | Sofia | CUTOFF | Ravi | — | arrives week 77; anxiety Ravi +0.1, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -0.7 |
| 76 | (from outside) | JOB_LOSS | Ravi | — | one-time effect; a 1-week spell, recorded; anxiety Ravi +2.3 |
| 76 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +2.9 |
| 76 | Ana | PURSUE | Marta | — | arrives week 77; anxiety Ana +0.4, Marta +1.7; calmer sender takes some anxiety: Ana +0.3, Marta -0.3 |
| 76 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 76 | Marta | OVERFUNCTION | Ana | — | arrives week 77; anxiety Ana +3.8, Marta +0.3; Marta takes 2.0 of functioning from the other |
| 76 | Nadia | DISTANCE | Ravi | — | arrives week 77; anxiety Nadia +0.5, Ravi +0.3; 2.3 of the sender's anxiety bound into Nadia–Ravi |
| 76 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 76 | Ravi | UNDERFUNCTION | Nadia | — | arrives week 77; anxiety Ravi +0.4; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Nadia takes 2.0 of functioning from the other |
| 76 | Sofia | DISTANCE | Ravi | Nadia | arrives week 78; anxiety Nadia +0.6 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; 1.4 of the sender's anxiety bound into Ravi–Sofia |
| 77 | Ana | PURSUE | Marta | — | arrives week 78; anxiety Ana +0.5, Marta +1.6; calmer sender takes some anxiety: Ana +0.2, Marta -0.2 |
| 77 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 77 | Marta | OVERFUNCTION | Ana | — | arrives week 78; anxiety Ana +4.3, Marta +0.3; Marta takes 2.0 of functioning from the other |
| 77 | Nadia | CUTOFF | Ravi | — | arrives week 78; anxiety Nadia +0.5, Ravi +0.3; Nadia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -0.4 |
| 77 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 77 | Ravi | UNDERFUNCTION | Nadia | — | arrives week 78; anxiety Ravi +0.4; calmer sender takes some anxiety: Nadia -0.2, Ravi +0.2; Nadia takes 2.0 of functioning from the other |
| 77 | Sofia | REDUCE_CUTOFF | Ravi | Nadia | arrives week 79; anxiety Nadia +0.6 (witness), Ravi +0.0, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia reopened |
| 78 | Ana | CONFLICT | Marta | — | arrives week 79; anxiety Ana +0.5, Marta +2.8; calmer sender takes some anxiety: Ana +0.1, Marta -0.1 |
| 78 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 78 | Marta | UNDERFUNCTION | Ana | — | arrives week 79; anxiety Ana +0.7, Marta +0.3; Ana takes 2.0 of functioning from the other |
| 78 | Nadia | REDUCE_CUTOFF | Ravi | — | arrives week 79; anxiety Nadia +0.5, Ravi +0.1; Nadia–Ravi reopened; bound anxiety released to the third party: Marta +2.3 |
| 78 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 78 | Ravi | FRAME_AMBIGUITY | Dr Halim | — | arrives week 79; anxiety Ravi +0.1 |
| 78 | Sofia | REDUCE_CUTOFF | Ravi | — | arrives week 80; anxiety Ravi +0.4, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 79 | Ana | DISTANCE | Marta | — | arrives week 80; anxiety Ana +0.5, Marta +0.4; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; 4.1 of the sender's anxiety bound into Ana–Marta |
| 79 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 79 | Marta | UNDERFUNCTION | Ana | — | arrives week 80; anxiety Ana +0.5, Marta +0.3; Ana takes 2.0 of functioning from the other |
| 79 | Nadia | STAY-IN-CONTACT | Ravi | — | arrives week 80; anxiety Nadia +0.5, Ravi +0.3 |
| 79 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 79 | Ravi | DISTANCE | Nadia | — | arrives week 80; anxiety Ravi +0.4; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; 3.1 of the sender's anxiety bound into Nadia–Ravi |
| 79 | Sofia | PURSUE | Ravi | Nadia | arrives week 81; anxiety Nadia +0.6 (witness), Ravi +1.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 80 | (from outside) | JOB_LOSS | Ravi | Nadia | one-time effect; a 1-week spell, recorded; anxiety Nadia +1.4 (witness), Ravi +2.6 |
| 80 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +2.7 |
| 80 | Ana | CUTOFF | Marta | — | arrives week 81; anxiety Ana +0.4, Marta +0.4; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -3.3, Marta -3.1 |
| 80 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 80 | Dr Halim | STAY-IN-CONTACT | Ravi | Nadia | arrives week 81; anxiety Dr Halim +0.1, Nadia +0.6 (witness); calmer sender takes some anxiety: Dr Halim +0.3, Ravi -0.3 |
| 80 | Marta | DISTANCE | Ana | — | arrives week 81; anxiety Ana +0.5, Marta +0.3; 3.3 of the sender's anxiety bound into Ana–Marta |
| 80 | Nadia | CONFLICT | Ravi | — | arrives week 81; anxiety Nadia +0.5, Ravi +4.1 |
| 80 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 80 | Ravi | UNDERFUNCTION | Nadia | — | arrives week 81; anxiety Nadia -0.1, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.4, Ravi +0.4; Nadia takes 2.0 of functioning from the other |
| 80 | Sofia | UNDERFUNCTION | Ravi | Nadia | arrives week 82; anxiety Nadia +0.6 (witness), Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Ravi takes 2.0 of functioning from the other |
| 81 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 82; anxiety Ana +0.4, Marta +0.2; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta reopened |
| 81 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 81 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 82; anxiety Ana +0.5, Marta +0.3 |
| 81 | Nadia | CUTOFF | Ravi | — | arrives week 82; anxiety Nadia +0.4, Ravi +0.4; Nadia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -3.4 |
| 81 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 81 | Ravi | STAY-IN-CONTACT | Nadia | — | arrives week 82; anxiety Ravi +0.4; calmer sender takes some anxiety: Nadia -0.2, Ravi +0.2 |
| 81 | Sofia | OVERFUNCTION | Ravi | Nadia | arrives week 83; anxiety Nadia +0.6 (witness), Ravi +2.1, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Sofia takes 2.0 of functioning from the other |
| 82 | Ana | DISTANCE | Marta | — | arrives week 83; anxiety Ana +0.4, Marta +0.3; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; 3.3 of the sender's anxiety bound into Ana–Marta |
| 82 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 82 | Marta | OVERFUNCTION | Ana | — | arrives week 83; anxiety Ana +3.9, Marta +0.3; Marta takes 2.0 of functioning from the other |
| 82 | Nadia | REDUCE_CUTOFF | Ravi | — | arrives week 83; anxiety Nadia +0.4, Ravi +0.4; Nadia–Ravi reopened; bound anxiety released to the third party: Marta +3.1 |
| 82 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 82 | Ravi | CUTOFF | Sofia | — | arrives week 84; anxiety Ravi +0.3; Ravi–Sofia now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -2.0 |
| 82 | Sofia | UNDERFUNCTION | Ravi | — | arrives week 84; anxiety Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi takes 2.0 of functioning from the other |
| 83 | Ana | CUTOFF | Marta | — | arrives week 84; anxiety Ana +0.5, Marta +0.3; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -5.7, Marta -0.3 |
| 83 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 83 | Marta | PURSUE | Ana | — | arrives week 84; anxiety Ana +2.8, Marta +0.3 |
| 83 | Nadia | DISTANCE | Ravi | — | arrives week 84; anxiety Nadia +0.4, Ravi +0.4; 2.4 of the sender's anxiety bound into Nadia–Ravi |
| 83 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 83 | Ravi | UNDERFUNCTION | Nadia | — | arrives week 84; anxiety Ravi +0.4; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Nadia takes 2.0 of functioning from the other |
| 83 | Sofia | CUTOFF | Ravi | Nadia | arrives week 85; anxiety Nadia +0.5 (witness), Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia now cut off (no events; bond energy kept) |
| 84 | (from outside) | JOB_LOSS | Ravi | Nadia | one-time effect; a 1-week spell, recorded; anxiety Nadia +1.3 (witness), Ravi +2.7 |
| 84 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +2.7 |
| 84 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 85; anxiety Ana +0.5, Marta +0.1; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana–Marta reopened |
| 84 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 84 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 85; anxiety Ana +0.7, Marta +0.3 |
| 84 | Nadia | CUTOFF | Ravi | — | arrives week 85; anxiety Nadia +0.4, Ravi +0.3; Nadia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -0.5 |
| 84 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 84 | Ravi | STAY-IN-CONTACT | Nadia | — | arrives week 85; anxiety Ravi +0.4; calmer sender takes some anxiety: Nadia -0.2, Ravi +0.2 |
| 84 | Sofia | REDUCE_CUTOFF | Ravi | Nadia | arrives week 86; anxiety Nadia +0.5 (witness), Ravi +0.0, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia reopened |
| 85 | Ana | CUTOFF | Marta | — | arrives week 86; anxiety Ana +0.5, Marta +0.2; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -3.2, Marta -0.2 |
| 85 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 85 | Marta | PURSUE | Ana | — | arrives week 86; anxiety Ana +2.8, Marta +0.3 |
| 85 | Nadia | REDUCE_CUTOFF | Ravi | — | arrives week 86; anxiety Nadia +0.4, Ravi +0.0; Nadia–Ravi reopened; bound anxiety released to the third party: Marta +2.4 |
| 85 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 85 | Ravi | REDUCE_CUTOFF | Sofia | — | arrives week 87; anxiety Ravi +0.3, Sofia -1.1 |
| 85 | Sofia | REDUCE_CUTOFF | Ravi | — | arrives week 87; anxiety Ravi +0.4, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3 |
| 86 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 87; anxiety Ana +0.5, Marta +0.1; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana–Marta reopened |
| 86 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 86 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 87; anxiety Ana +0.8, Marta +0.3 |
| 86 | Nadia | DISTANCE | Ravi | — | arrives week 87; anxiety Nadia +0.4, Ravi +1.0; 1.8 of the sender's anxiety bound into Nadia–Ravi |
| 86 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 86 | Ravi | PURSUE | Sofia | Nadia | arrives week 88; anxiety Nadia +0.5 (witness), Ravi +0.3, Sofia +1.8 |
| 86 | Sofia | STAY-IN-CONTACT | Ravi | Nadia | arrives week 88; anxiety Nadia +0.5 (witness), Ravi +0.2, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 87 | Ana | CONFLICT | Marta | — | arrives week 88; anxiety Ana +0.5, Marta +2.7; calmer sender takes some anxiety: Ana +0.3, Marta -0.3 |
| 87 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 87 | Marta | PURSUE | Ana | — | arrives week 88; anxiety Ana +2.8, Marta +0.3 |
| 87 | Nadia | DISTANCE | Ravi | — | arrives week 88; anxiety Nadia +0.4, Ravi +2.3; 3.0 of the sender's anxiety bound into Nadia–Ravi |
| 87 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 87 | Ravi | I-POSITION | Nadia | — | arrives week 88; anxiety Nadia +3.5, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.2, Ravi +0.2; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 87 | Sofia | PURSUE | Ravi | Nadia | arrives week 89; anxiety Nadia +0.5 (witness), Ravi +1.4, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5 |
| 88 | (from outside) | JOB_LOSS | Ravi | Nadia | one-time effect; a 1-week spell, recorded; anxiety Nadia +1.3 (witness), Ravi +2.7 |
| 88 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +2.7 |
| 88 | Ana | CUTOFF | Marta | — | arrives week 89; anxiety Ana +0.5, Marta +0.3; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -2.1, Marta -2.0 |
| 88 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 88 | Dr Halim | STAY-IN-CONTACT | Ravi | Nadia | arrives week 89; anxiety Dr Halim +0.1, Nadia +0.5 (witness); calmer sender takes some anxiety: Dr Halim +0.4, Ravi -0.4 |
| 88 | Marta | WITHHOLD | Ana | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 88 | Nadia | PURSUE | Ravi | — | arrives week 89; anxiety Nadia +0.4, Ravi -0.4 |
| 88 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 88 | Ravi | UNDERFUNCTION | Nadia | — | arrives week 89; anxiety Nadia +0.4, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.1, Ravi +0.1; Nadia takes 2.0 of functioning from the other |
| 88 | Sofia | PURSUE | Ravi | Nadia | arrives week 90; anxiety Nadia +0.5 (witness), Ravi +1.5, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 89 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 90; anxiety Ana +0.5, Marta +0.1; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta reopened |
| 89 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 89 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 90; anxiety Ana +0.7, Marta +0.3 |
| 89 | Nadia | OVERFUNCTION | Ravi | — | arrives week 90; anxiety Nadia +0.4, Ravi +3.9; Nadia takes 2.0 of functioning from the other |
| 89 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 89 | Ravi | SPLIT | Dr Halim | Nadia | arrives week 90; anxiety Nadia +0.5 (witness), Ravi +0.1 |
| 89 | Sofia | CUTOFF | Ravi | Nadia | arrives week 91; anxiety Nadia +0.5 (witness), Ravi +0.2, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5; Ravi–Sofia now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -1.9, Sofia -0.2 |
| 90 | Ana | CUTOFF | Marta | — | arrives week 91; anxiety Ana +0.5, Marta +0.2; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -4.6, Marta -0.2 |
| 90 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 90 | Marta | OVERFUNCTION | Ana | — | arrives week 91; anxiety Ana +4.3, Marta +0.3; Marta takes 2.0 of functioning from the other |
| 90 | Nadia | CUTOFF | Ravi | — | arrives week 91; anxiety Nadia +0.4, Ravi +0.5; Nadia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Nadia -0.9, Ravi -4.4 |
| 90 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 90 | Ravi | UNDERFUNCTION | Sofia | Nadia | arrives week 92; anxiety Nadia +0.5 (witness), Ravi +0.3; Sofia takes 2.0 of functioning from the other |
| 90 | Sofia | STAY-IN-CONTACT | Ravi | Nadia | arrives week 92; anxiety Nadia +0.5 (witness), Sofia +0.4; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3 |
| 91 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 92; anxiety Ana +0.5, Marta +0.1; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Ana–Marta reopened |
| 91 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 91 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 92; anxiety Ana +0.9, Marta +0.3 |
| 91 | Nadia | REDUCE_CUTOFF | Ravi | — | arrives week 92; anxiety Nadia +0.4, Ravi +0.6; Nadia–Ravi reopened; bound anxiety released to the third party: Marta +4.7 |
| 91 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 91 | Ravi | FRAME_AMBIGUITY | Dr Halim | — | arrives week 92; anxiety Ravi +0.1 |
| 91 | Sofia | REDUCE_CUTOFF | Ravi | — | arrives week 93; anxiety Ravi +0.3, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia reopened |
| 92 | (from outside) | JOB_LOSS | Ravi | — | one-time effect; a 1-week spell, recorded; anxiety Ravi +2.9 |
| 92 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +2.6 |
| 92 | Ana | CUTOFF | Marta | — | arrives week 93; anxiety Ana +0.5, Marta +0.1; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana–Marta now cut off (no events; bond energy kept); impingement removed at once, anxiety Ana -5.9, Marta -0.1 |
| 92 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 92 | Marta | I-POSITION | Ana | — | arrives week 93; anxiety Ana +5.4, Marta +0.3; assertion form (no systems perspective, or angry): extra impingement; claiming the position raises the sender's outward axis |
| 92 | Nadia | CUTOFF | Ravi | — | arrives week 93; anxiety Nadia +0.4, Ravi +0.4; Nadia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Nadia -2.8, Ravi -0.7 |
| 92 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 92 | Ravi | OVERFUNCTION | Nadia | — | arrives week 93; anxiety Nadia +2.8, Ravi +0.5; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Ravi takes 2.0 of functioning from the other |
| 92 | Sofia | REDUCE_CUTOFF | Ravi | Nadia | arrives week 94; anxiety Nadia +0.5 (witness), Ravi +0.6, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 93 | Ana | REDUCE_CUTOFF | Marta | — | arrives week 94; anxiety Ana +0.5, Marta +0.0; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana–Marta reopened |
| 93 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 93 | Marta | REDUCE_CUTOFF | Ana | — | arrives week 94; anxiety Ana +1.0, Marta +0.3 |
| 93 | Nadia | REDUCE_CUTOFF | Ravi | — | arrives week 94; anxiety Nadia +0.4, Ravi +0.5; Nadia–Ravi reopened |
| 93 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 93 | Ravi | CUTOFF | Sofia | — | arrives week 95; anxiety Ravi +0.3; Ravi–Sofia now cut off (no events; bond energy kept); impingement removed at once, anxiety Ravi -0.4 |
| 93 | Sofia | CUTOFF | Ravi | — | arrives week 95; anxiety Ravi +0.2, Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia now cut off (no events; bond energy kept) |
| 94 | Ana | UNDERFUNCTION | Marta | — | arrives week 95; anxiety Ana +0.5, Marta +0.0; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Marta takes 2.0 of functioning from the other |
| 94 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 94 | Marta | UNDERFUNCTION | Ana | — | arrives week 95; anxiety Ana +0.8, Marta +0.3; Ana takes 2.0 of functioning from the other |
| 94 | Nadia | TRIANGLE | Ravi | — | arrives week 95; anxiety Nadia +0.4, Ravi +0.4; Ravi left outside: anxiety Nadia -0.5, Ravi +0.7 (of which 0.2 the outsider's own) |
| 94 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 94 | Ravi | FRAME_AMBIGUITY | Dr Halim | Nadia | arrives week 95; anxiety Nadia +0.5 (witness), Ravi +0.1 |
| 94 | Sofia | UNDERFUNCTION | Ravi | Nadia | arrives week 96; anxiety Nadia +0.5 (witness), Sofia +0.3; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi takes 2.0 of functioning from the other |
| 95 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 95 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 95 | Ana | DISTANCE | Marta | — | arrives week 96; anxiety Ana +0.5, Marta +0.0; calmer sender takes some anxiety: Ana +0.2, Marta -0.2; 4.0 of the sender's anxiety bound into Ana–Marta |
| 95 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 95 | Marta | PROVOKE | Ana | — | arrives week 96; anxiety Ana +4.0, Marta +0.3 |
| 95 | Nadia | UNDERFUNCTION | Ravi | — | arrives week 96; anxiety Nadia +0.4, Ravi +0.3; Ravi takes 2.0 of functioning from the other |
| 95 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 95 | Ravi | SPLIT | Dr Halim | Nadia | arrives week 96; anxiety Dr Halim +0.2, Nadia +0.5 (witness), Ravi +0.1 |
| 95 | Sofia | REDUCE_CUTOFF | Ravi | Nadia | arrives week 97; anxiety Nadia +0.5 (witness), Ravi +0.1, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi–Sofia reopened |
| 96 | (from outside) | JOB_LOSS | Ravi | Nadia | one-time effect; a 1-week spell, recorded; anxiety Nadia +1.2 (witness), Ravi +2.9 |
| 96 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +2.6 |
| 96 | Ana | DISTANCE | Marta | — | arrives week 97; anxiety Ana +0.5, Marta +0.0; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; 2.8 of the sender's anxiety bound into Ana–Marta |
| 96 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 96 | Dr Halim | CUTOFF | Ravi | Nadia | arrives week 97; anxiety Dr Halim +0.1, Nadia +0.5 (witness); calmer sender takes some anxiety: Dr Halim +0.3, Ravi -0.3; Dr Halim–Ravi now cut off (no events; bond energy kept); the contact lands with Ravi: systems perspective +0.08 |
| 96 | Marta | WITHHOLD | Ana | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 96 | Nadia | OVERFUNCTION | Ravi | — | arrives week 97; anxiety Nadia +0.4, Ravi +3.7; Nadia takes 2.0 of functioning from the other |
| 96 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 96 | Ravi | OVERFUNCTION | Nadia | — | arrives week 97; anxiety Nadia +3.2, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; Ravi takes 2.0 of functioning from the other |
| 96 | Sofia | REDUCE_CUTOFF | Ravi | Nadia | arrives week 98; anxiety Nadia +0.5 (witness), Ravi +0.4, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.5, Sofia +0.5 |
| 97 | Ana | UNDERFUNCTION | Marta | — | arrives week 98; anxiety Ana +0.5; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; Marta takes 2.0 of functioning from the other |
| 97 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 97 | Marta | UNDERFUNCTION | Ana | — | arrives week 98; anxiety Ana +0.7, Marta +0.3; Ana takes 2.0 of functioning from the other |
| 97 | Nadia | OVERFUNCTION | Ravi | — | arrives week 98; anxiety Nadia +0.4, Ravi +3.4; Nadia takes 2.0 of functioning from the other |
| 97 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 97 | Ravi | DISTANCE | Nadia | — | arrives week 98; anxiety Nadia +1.1, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.3, Ravi +0.3; 4.8 of the sender's anxiety bound into Nadia–Ravi |
| 97 | Sofia | DISTANCE | Ravi | Nadia | arrives week 99; anxiety Nadia +0.5 (witness), Ravi +0.1, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; 1.7 of the sender's anxiety bound into Ravi–Sofia |
| 98 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 98 | Ana | DISTANCE | Marta | — | arrives week 99; anxiety Ana +0.5, Marta +1.1; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; 2.5 of the sender's anxiety bound into Ana–Marta |
| 98 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 98 | Marta | DISTANCE | Ana | — | arrives week 99; anxiety Ana +0.6, Marta +0.3; 2.8 of the sender's anxiety bound into Ana–Marta |
| 98 | Nadia | WITHHOLD | Ravi | — | held back OVERFUNCTION: computed, not emitted; attention on the tie rises |
| 98 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 98 | Ravi | OVERFUNCTION | Sofia | Nadia | arrives week 100; anxiety Nadia +0.5 (witness), Ravi +0.3, Sofia +2.5; Ravi takes 2.0 of functioning from the other |
| 98 | Sofia | DISTANCE | Ravi | Nadia | arrives week 100; anxiety Nadia +0.5 (witness), Ravi +0.3, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; 2.1 of the sender's anxiety bound into Ravi–Sofia |
| 99 | Ana | DISTANCE | Marta | — | arrives week 100; anxiety Ana +0.5, Marta +1.1; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; 3.1 of the sender's anxiety bound into Ana–Marta |
| 99 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 99 | Marta | PROVOKE | Ana | — | arrives week 100; anxiety Ana +3.8, Marta +0.3 |
| 99 | Nadia | CONFLICT | Ravi | — | arrives week 100; anxiety Nadia +0.4, Ravi +3.3 |
| 99 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 99 | Ravi | CUTOFF | Nadia | — | arrives week 100; anxiety Nadia +0.6, Ravi +0.5; calmer sender takes some anxiety: Nadia -0.4, Ravi +0.4; Nadia–Ravi now cut off (no events; bond energy kept); impingement removed at once, anxiety Nadia -1.5, Ravi -6.4 |
| 99 | Sofia | PURSUE | Ravi | Nadia | arrives week 101; anxiety Nadia +0.5 (witness), Ravi +1.1, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4 |
| 100 | (from outside) | JOB_LOSS | Ravi | Nadia | one-time effect; a 1-week spell, recorded; anxiety Nadia +1.2 (witness), Ravi +2.8 |
| 100 | (from outside) | JOB_LOSS | Marta | — | one-time effect; a 1-week spell, recorded; anxiety Marta +2.6 |
| 100 | Ana | DISTANCE | Marta | — | arrives week 101; anxiety Ana +0.5, Marta +0.1; calmer sender takes some anxiety: Ana +0.3, Marta -0.3; 3.9 of the sender's anxiety bound into Ana–Marta |
| 100 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 100 | Marta | CONFLICT | Ana | — | arrives week 101; anxiety Ana +5.6, Marta +0.3 |
| 100 | Nadia | REDUCE_CUTOFF | Ravi | — | arrives week 101; anxiety Nadia +0.4, Ravi +0.7; Nadia–Ravi reopened; bound anxiety released to the third party: Marta +4.8 |
| 100 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 100 | Ravi | CONFLICT | Sofia | — | arrives week 102; anxiety Ravi +0.3, Sofia +4.1 |
| 100 | Sofia | UNDERFUNCTION | Ravi | — | arrives week 102; anxiety Ravi +0.2, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.4, Sofia +0.4; Ravi takes 2.0 of functioning from the other |
| 101 | Ana | OVERFUNCTION | Marta | — | arrives week 102; anxiety Ana +0.5, Marta +1.2; calmer sender takes some anxiety: Ana +0.4, Marta -0.4; Ana takes 2.0 of functioning from the other |
| 101 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 101 | Marta | DISTANCE | Ana | — | arrives week 102; anxiety Ana +0.8, Marta +0.3; 4.1 of the sender's anxiety bound into Ana–Marta |
| 101 | Nadia | UNDERFUNCTION | Ravi | — | arrives week 102; anxiety Nadia +0.4, Ravi +0.5; Ravi takes 2.0 of functioning from the other |
| 101 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 101 | Ravi | TRIANGLE | Nadia | — | arrives week 102; anxiety Nadia -0.8, Ravi +0.4; calmer sender takes some anxiety: Nadia -0.4, Ravi +0.4; Nadia left outside: anxiety Nadia +1.1, Ravi -0.8 (of which 0.4 the outsider's own) |
| 101 | Sofia | PURSUE | Ravi | Nadia | arrives week 103; anxiety Nadia +0.5 (witness), Ravi +1.2, Sofia +0.4; calmer sender takes some anxiety: Ravi -0.3, Sofia +0.3 |
| 102 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 102 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Ravi |
| 102 | Ana | DISTANCE | Marta | — | arrives week 103; anxiety Ana +0.5, Marta +1.1; calmer sender takes some anxiety: Ana +0.2, Marta -0.2; 4.4 of the sender's anxiety bound into Ana–Marta |
| 102 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 102 | Marta | CONFLICT | Ana | — | arrives week 103; anxiety Ana +5.2, Marta +0.3 |
| 102 | Nadia | TRIANGLE | Ravi | — | arrives week 103; anxiety Nadia +0.4, Ravi +0.4; Ravi left outside: anxiety Nadia -0.6, Ravi +0.9 (of which 0.3 the outsider's own) |
| 102 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 102 | Ravi | PURSUE | Sofia | Nadia | — |
| 102 | Sofia | WITHHOLD | Ravi | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 103 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Marta–Nadia |
| 103 | Ana | CUTOFF | Marta | — | — |
| 103 | Bruno | (holds) | — | — | fallback (hold): no legal act |
| 103 | Marta | WITHHOLD | Ana | — | held back UNDERFUNCTION: computed, not emitted; attention on the tie rises |
| 103 | Nadia | CUTOFF | Ravi | — | — |
| 103 | Pia | (holds) | — | — | fallback (hold): no legal act |
| 103 | Ravi | CONFLICT | Sofia | Nadia | — |
| 103 | Sofia | CONFLICT | Ravi | Nadia | — |
| 103 | — | (system) | — | — | slow tick (yearly) fired — nothing runs on it in Phase B |

104 weeks. The invariants were asserted at the end of every week.
