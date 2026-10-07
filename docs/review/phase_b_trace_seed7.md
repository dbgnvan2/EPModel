# Run trace

> **What this is.** Output of a simulation that runs one theory's stated mechanisms
> over an invented family. It is a consistency engine for that theory — *if Bowen's
> account is right, what follows for a family shaped like this?* — not a measurement,
> not a prediction, and not a "virtual family" in the predictive sense (spec `M11.F.10`).
> It does not concern any real family or person, and nothing here is advice (`M11.F.9`).
> Every number in it comes from invented constants (`[I]`); only differences between
> two runs that share them carry meaning (`M0.4`). This run is scripted: no one in it
> chose anything (Phase B).

| Header | |
|---|---|
| seed | 7 |
| config hash | `80ac950c614d3c87754009ac6b1fa1968a89bbe6bcd8e82badce67040e175d32` |
| spec revision | 2.0, revision 10 |
| family instance | phase_b_reduced |
| activation | synchronous_activation v1 (synchronous) |
| visibility | household_conductance_visibility v1 |
| constants frozen | 2026-10-06 |
| constants changed since freeze | none |

The standing load and the decay toward each person's floor run every week for everyone and are not listed line by line. Effects are shown beside the event that caused them.

| Week | Who | Move | Toward | Witnesses | What it did |
|---|---|---|---|---|---|
| 0 | (from outside) | JOB_LOSS | Ravi | Marta, Nadia, Pia | one-time effect; a 34-week spell, recorded; anxiety Marta +10.0 (witness), Nadia +9.8 (witness), Pia +9.1 (witness), Ravi +10.3 |
| 0 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 0 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 1 | Ravi | CONFLICT | Marta | Nadia, Pia | arrives week 2; anxiety Marta +5.0, Nadia +4.9 (witness), Pia +4.5 (witness) |
| 2 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 3; anxiety Nadia +2.4 (witness), Pia +2.3 (witness), Ravi +2.6 |
| 3 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 4; anxiety Nadia +2.4 (witness), Pia +2.3 (witness), Ravi +2.6 |
| 4 | Ravi | TRIANGLE | Nadia | Marta, Pia | arrives week 5; anxiety Marta +3.8 (witness), Nadia +3.7, Pia +3.4 (witness) |
| 5 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 5 | Marta | DISTANCE | Ravi | Nadia, Pia | arrives week 6; anxiety Nadia +2.4 (witness), Pia +2.3 (witness), Ravi +2.6 |
| 5 | — | (system) | — | — | Marta–Ravi tie now distant |
| 6 | Marta | OVERFUNCTION | Nadia | Pia, Ravi | arrives week 7; anxiety Nadia +2.9, Pia +2.7 (witness), Ravi +3.1 (witness) |
| 9 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 10; anxiety Marta +2.0, Pia +1.8 (witness), Ravi +2.1 (witness) |
| 10 | (from outside) | TRIGGER | Ana–Bruno | — | standing load on Ana–Bruno spikes next week (multiplier +1) for 1 week; no event crosses the tie |
| 12 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 13; anxiety Marta +2.0 (witness), Pia +1.8 (witness), Ravi +2.1 |
| 15 | Nadia | UNDERFUNCTION | Marta | Pia, Ravi | arrives week 16; anxiety Marta +2.0, Pia +1.8 (witness), Ravi +2.1 (witness) |
| 18 | Nadia | UNDERFUNCTION | Ravi | Marta, Pia | arrives week 19; anxiety Marta +2.0 (witness), Pia +1.8 (witness), Ravi +2.1 |
| 23 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 23 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 24 | Ana | TRIANGLE | Marta | Nadia, Pia, Ravi | arrives week 25; anxiety Marta +1.8, Nadia +1.8 (witness), Pia +1.6 (witness), Ravi +1.8 (witness) |
| 25 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 25 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 25 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 28 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 28 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 31 | Marta | I-POSITION | Ravi | Nadia, Pia | arrives week 32; anxiety Nadia +2.4 (witness), Pia +2.3 (witness), Ravi +2.6 |
| 32 | — | (system) | — | — | Marta–Nadia–Ravi triangle active |
| 32 | — | (system) | — | — | Marta–Nadia–Ravi: inside pair Nadia–Ravi |
| 32 | — | (system) | — | — | Marta–Pia–Ravi triangle active |
| 33 | — | (system) | — | — | Marta–Nadia–Ravi triangle inactive |
| 33 | — | (system) | — | — | Marta–Pia–Ravi triangle inactive |
| 36 | Ravi | PURSUE | Sofia | Marta, Nadia, Pia | arrives week 38; anxiety Marta +0.9 (witness), Nadia +0.9 (witness), Pia +0.8 (witness), Sofia +0.9 |

40 weeks. The invariants were asserted at the end of every week; M6.I.6 is disabled until it is restated (`M4.G.2a`).
