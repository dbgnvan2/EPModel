# Phase D, part II (the life course): status, 2026-10-13

> **Status: a report, not an approval.** The owner approved the part II plan on 2026-10-13. The forms in spec revision 28
> are Claude's recommendation, pending owner review. Every verdict below comes from the generated record,
> `docs/phase_d2_ensemble_record.md`, or from a named test. Details and the `M11.C.15` diagnosis: plan §4.7
> (`docs/implementation_plan_phase_d.md`).

## Build steps (plan §4.3)

| Step | Status | Evidence |
|---|---|---|
| L0, a topology that can grow | done | `src/bowen/engine/state.py` (`other_families`, `add_person`, `add_tie`); `tests/bowen/test_life_course.py::test_m1d1_each_household_couple_has_its_own_family` |
| L1, the family data and starting ages as settings | done | `config/bowen/family_phase_d.md`; `test_m2a0j_ages_come_from_the_three_settings`, `test_m2b2_the_coach_is_tied_to_the_coachee_only` |
| L2, who acts (B11) | done | `src/bowen/engine/activation.py`; `test_m7c1f_a_child_under_ten_never_selects`, `test_m7c1f_a_child_under_ten_still_witnesses` |
| L3, chronic anxiety before 12 | done | `life_course.py::_childhood`; `test_m7b1a_child_chronic_reads_parents_and_witnessed_history` |
| L4, development from 5 to 20 | done | `test_m7e1f_level_never_leaves_two_percent_band`, `test_m7e1f_development_stops_at_twenty` |
| L5, mortality and the disposition at death | done | `life_course.py::die`; `test_m63b_*` to `test_m63g_*`, `test_m11c37_anxiety_conserved_across_death` |
| L6, lost income and `FIND_WORK` | done, with one renamed test: the act follows pressure rather than being learned (`M7.C.1k`) | `test_m63h_income_loss_lasts_until_someone_finds_work`, `test_m7c1k_find_work_follows_pressure_not_a_schedule` |
| L7, launch, partners, births, new families | done | `config/bowen/partners_phase_d.md`; `test_m7c1g_*`, `test_m7c1h_*`, `test_m7c1i_*`, `test_m64_*` |
| L8, the coach's retirement and successor | done | `test_m2a0k_successor_enters_at_retirement_and_learning_is_kept` |
| L9, the 40-year run and the `M16.T.3` re-run | done | `test_m3b1_forty_year_run_completes` (ensemble-marked); `test_m16t3_sink_does_not_change_a_life_course_run` |
| L10, criteria and record | done; three of five criteria fail | `src/bowen/ensemble/criteria_d2.py`; `docs/phase_d2_ensemble_record.md` |

## The criteria

| Criterion | Status | Result (treatment − baseline) | Proved by its mutation |
|---|---|---|---|
| `M11.C.37` | done | conserved across a death | yes: `test_m11c37_mutation_discarding_the_dead_persons_anxiety_turns_it_red` |
| `M11.C.15` | done, after the owner's decision on its arms (transfer kept, share 1) | death +1.25 ± 0.38 (p 2.1e-08); recovery +3.11 ± 0.35 (p 8.2e-18) | yes, ad hoc: the bound anxiety discarded at the death gives death −2.03 ± 0.29 (plan §4.7) |
| `M11.C.6` | not done: fails | +0.0177 ± 0.028 (p 0.894) | — |
| `M11.C.10` | not done: fails | −12.6 ± 28 (p 0.795) | — |
| `M11.C.2` (twelve-person family) | not done: fails | −0.13 ± 1.6 (p 0.456) | — |

Phase C and part I verdicts did not change: all three regenerations moved only hashes and header lines. Suite: 766
tests pass; 45 ensemble-marked tests are deselected.

## For the owner

1. *Decided 2026-10-13:* `M11.C.15` keeps `M6.3b`'s transfer at 1 ("keep the transfer in C.15, share at 1"). It
   failed with the share at 0 (death +0.21, p 0.135) and passes at 1.
2. **The ±2% band against `M11.C.6`.** At this band the effect `M11.C.6` looks for is at most about 0.8 points, and the
   test cannot separate it from noise. Which one changes: the band, the criterion, or neither?
3. **`M11.C.10` and `M11.C.2` (diagnosed, plan §4.7).**
   - `M11.C.10`'s arms do not stay apart: the declared cutoff ends within about 2 years, and the baseline's tie is cut
     within weeks. Two links of its chain have no mechanism.
   - `M11.C.2`'s parents spread projection evenly over the children, and nothing carries it to later load, which
     follows each child's age.
   - Each needs a design decision.
4. **A child's development**: the `M7.E.1f` rule (built), or the estimator bounded to the band.
5. Still open from part I: Marta–Iris, Teodor's level (36 built), and `M11.C.17`'s gate.
6. The defaults built in plan §4.5, each `[I]`: launch age 20, a Gompertz mortality with a median near 80, the partner
   rows, grief 20, income loss 15, working age 16, and a female successor coach.
7. **No setting in the app yet.** Starting ages are config settings (`M2.A.0j`), but the fam-diag app was not touched;
   exposing them there is deferred.
