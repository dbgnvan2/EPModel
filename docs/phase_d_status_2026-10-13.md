# Phase D, part I — status after the overnight run of 2026-10-12/13

> **Status: a report, not an approval.** The owner asked for the work to proceed overnight without waiting for
> input. Spec revisions 24–27 are Claude's recommendations, pending owner review; none is an owner decision. Every
> criterion result below is an ad hoc run on the final code (`final25.sh` / `final25.out` in the session scratchpad),
> not yet a record (plan step 12). Details, mutations and diagnoses: `docs/implementation_plan_phase_d.md`, steps 6–10.

## Build items

| Item | Status | Evidence |
|---|---|---|
| B1–B6 (speed, batched items, channel stamping, cutoff, slow tick, estimator) | done (before this night) | plan §2 |
| B7, the symptom channels (revision 24) | done | `src/bowen/engine/symptoms.py`; `tests/bowen/test_symptoms.py` |
| B8, beliefs as a channel (revision 25) | done | `src/bowen/engine/attribution.py`, `appraise.py`; `tests/bowen/test_attribution.py` |
| B9, projection: the tie-weighted cross-person term (revision 26) | partial: the term is built and unit-tested; no criterion shows it shaping a repertoire (`M11.C.2` fails) | `learner.py::cross_person_relief`; `tests/bowen/test_learner.py::test_m4d6e_the_cross_person_term_weighs_the_closer_tie_more` |
| B10, the `M11.G` readout | done | `src/bowen/readouts/evaluation.py`; `tests/bowen/test_evaluation.py` |
| B11, who acts | not done: no D-I criterion needs it; moved to part II | plan B11 |
| Revision 27, borrowed pseudo-self settles back | done | `moves.py::relax_pseudo_self`; `tests/bowen/test_moves.py` |

## The D-I gate, criterion by criterion

| Criterion | Status | Result (treatment − baseline) | Proved by its mutation |
|---|---|---|---|
| `M11.C.17` | done, weak | +0.0009 ± 0.0008 (p 0.011); was +0.021 at revision 23 | yes at revision 23 (landed contact removed); not rerun |
| `M11.C.18` | done | +0.015 ± 0.002 | yes (revision 23) |
| `M11.C.20` | done | +0.015 ± 0.002 | yes (revision 23's restated clause) |
| `M11.C.33` | done | +0.016 ± 0.002 | not rerun since revision 20 |
| `M11.C.30` | done | −21.4 ± 1.6 | yes |
| `M11.C.22` | done | mild +0.81 ± 0.38; severe −1.03 ± 0.44 | yes, each limb |
| `M11.C.31` | done | +22.1 ± 0.6 | yes (discarding breaks the ledger) |
| `M11.C.11` | partial | +2.63 ± 0.92 | conservation by the ledger; the direction limb has no mutant; the curve is the owner's (`M11.E`) |
| `M11.C.26` | done | +0.76 ± 0.04 | yes |
| `M11.C.36` | done | +3.07 ± 0.01 | yes |
| `M11.C.39` | done | +6.0 ± 1.1 | yes |
| `M11.C.24` | done | +11.9 ± 0.4 / −12.6 ± 0.5 | yes, both mutants |
| `M11.C.12` | not done: fails | −0.04 ± 0.24 | — |
| `M11.C.14` | not done: fails | −8.0 ± 1.9 against a bound of 5.6 | — |
| `M11.C.2` | not done: fails | +0.29 ± 2.20 (p 0.39) | — |
| `M11.C.40` | not done: fails clause 2 | clause 2 −0.0005 ± 0.041 | — |
| `M11.C.43` | not done: fails | −0.54 ± 0.71 | — |
| `M11.C.34` | not done: fails | −0.04 ± 0.10 | — |
| `M11.C.21` | not done: fails | −0.52 ± 1.32 | — |
| `M11.C.23` | not done: fails, reversed | −0.29 ± 0.08 | — |
| `M11.C.7` | not done: needs the coach's landing to read its position (`M8`) | — | — |
| `M11.C.9` | not done: needs sibling position to gate behaviour (`M1.A.14c`) | — | — |
| `M11.C.28` | not done: the sink allocation feeds no behaviour | — | — |
| `M11.D.20` | done | `tests/bowen/test_evaluation.py::test_m11d20_readout_ordering_matches_run_log_ordering` | — |

**Not done in part I:** the Phase D record (plan step 12, records over the D criteria), and the D-I completion
report. Part II (the life course, the 40-year run, `M11.C.6`, `.10`, `.15`, `.37`) is not started; it needs the owner's
P13–P16.

## For the owner, first

1. Revisions 24–27 as a whole (`TODO.md` lists each choice).
2. `M11.C.17`'s collapse: two probes put part of it on revision 27 and part on revision 24.
3. The eight failing criteria: each is the owner's question whether the criterion, its readout or the model is wrong;
   nothing was changed to make any pass.
4. `I-POSITION` completion in the triad fell from about 10 in 96 seeds to 4 in 192 after B7.
