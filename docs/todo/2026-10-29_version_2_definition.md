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
- [ ] T-166 ⟨later⟩ **`tools/group_g_readout.py` still averages unmatched first cutoffs (the batch's `learning-qa`
  review, 2026-10-29; not fixed).** Its "mean run before the first cutoff" and "mean matched uncut run" columns read
  every first cutoff, so an unmatched one counts with a baseline of 0, the bias T-123 removed from `M11.C.48`. It is
  revision 36's before-and-after page, whose "before" side was measured on the old definition; filter its "after" side
  on `FirstCutoff.matched` with a note, or state the difference on the page. No rule changes.
- [ ] T-167 ⟨now⟩ **The stage-end sweep reruns on spec revision 43's engine, `da5981f6ec0b` (the batch, 2026-10-30;
  not run yet).** Every record but `docs/version_2_sweep_record.md` was regenerated on it;
  `tests/test_version_2_sweep.py::test_s2_11_sweep_record_is_current` is a strict expected failure until the sweep
  reruns, split across both Macs (`docs/two_mac_sweep.md`; Mac Mini1 at 8 workers). `M11.C.48` now runs 200 seeds, so
  the sweep can no longer refuse it at the rerun (T-163); it fails centrally, so W4 does not count it.
