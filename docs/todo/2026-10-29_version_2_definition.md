# Version 2's definition — the batch, 2026-10-29

Items found while building the version 2 definition batch (`docs/version_2_definition_plan.md` §2, spec revision 43),
on the owner's instruction of 2026-10-29 to proceed with Claude's recommendations.

- [ ] T-165 ⟨later⟩ **`M11.C.17`'s coached rise runs through the coach's I-position pull-up, which the estimator
  counts as the member's own functioning (found tracing `landed-contact-changes-nothing`'s survival, 2026-10-29; not
  decided).** With `perspective_gain` at 0 the coached arm still rises (ad hoc, 150 seeds: +0.0332 ± 0.028). The rise
  comes from `M5.D.5`'s pull-up when the coach's own `I-POSITION` resolves (`src/bowen/engine/iposition.py`,
  `resolve`): the member closes `pull_up_rate` of the gap to the coach's level, and the slow tick's estimator of own
  functioning (`src/bowen/engine/slow_tick.py`) takes out pseudo-self, `adjust_swing` and `floor_held` but not this
  pull-up, so it is credited to the member's basic level. On the four seeds traced, the person who rose was the one who
  received the coach's `I-POSITION`. Spec revision 43 names `coach-pull-up-removed` in the row, so the mutation proves
  the path; whether a pull-up toward the coach's level is "earned in behaviour" (`M11.C.17`, `FE11.6`) is a theory
  question for the owner.
  - 2026-10-30, Claude's recommendation 2026-10-30 under the owner's instruction of that night to proceed without stopping, pending the owner's review (`docs/version_2_definition_plan.md` §8, D5): **not earned.** `FE06.10` (`[K]`: a motivated member can go beyond the therapist's
    level; differentiation depends on knowledge, not on the coach's level) and `FE11.6` (`[B]`: the therapist may be a
    model; instruction makes the family dependent) both place a rise toward the coach's own level outside what
    `M11.C.17` counts, and `M5.D.5`'s reason the gain is solid self (both parties rise) does not hold for a mover
    outside the family. Recommended rule: `M5.D.5`'s pull-up does not apply when the mover is the external agent
    (the alternative, booking it as borrowed functioning the estimator removes, needs a new ledger). Moves the hash;
    `M11.C.17` is then expected to fail (`coach-pull-up-removed` reads −0.00003 ± 0.00017) and need a decision.
- [x] T-166 ⟨later⟩ **`tools/group_g_readout.py` still averages unmatched first cutoffs (the batch's `learning-qa`
  review, 2026-10-29; not fixed).** Its "mean run before the first cutoff" and "mean matched uncut run" columns read
  every first cutoff, so an unmatched one counts with a baseline of 0, the bias T-123 removed from `M11.C.48`. It is
  revision 36's before-and-after page, whose "before" side was measured on the old definition; filter its "after" side
  on `FirstCutoff.matched` with a note, or state the difference on the page. No rule changes.
  *Closed 2026-10-30:* the run columns (`M11.C.48`'s two arms, the median and the longest uncut run) read matched
  first cutoffs only, a new "unmatched" column counts the rest, and the header says the "before" section was
  measured on the earlier definition. Test, shown red first and proved by the filter's removal:
  `tests/test_group_g_readout.py::test_m11c48_t166_readout_leaves_unmatched_first_cutoffs_out_of_the_run_columns_and_counts_them`.
  The "after" section of `docs/review/group_g_readout.md` was regenerated on `da5981f6ec0b` (20 seeds per family).
- [x] T-167 ⟨now⟩ **The stage-end sweep reruns on spec revision 43's engine, `da5981f6ec0b` (the batch, 2026-10-30;
  not run yet).** Every record but `docs/version_2_sweep_record.md` was regenerated on it;
  `tests/test_version_2_sweep.py::test_s2_11_sweep_record_is_current` is a strict expected failure until the sweep
  reruns, split across both Macs (`docs/two_mac_sweep.md`; Mac Mini1 at 8 workers). `M11.C.48` now runs 200 seeds, so
  the sweep can no longer refuse it at the rerun (T-163); it fails centrally, so W4 does not count it.
  *Closed 2026-10-30:* the sweep reran on `da5981f6ec0b`, split across both Macs after a cross-check (`docs/two_mac_sweep.md`,
  "Not verified"). Only `M11.C.48` and `M11.C.48[phase_d]` moved: each holds 0 of 8, and neither is refused. The
  strict expected failure is gone from `test_s2_11_sweep_record_is_current`, and `docs/version_2_status.md`'s W4 is met.
