# Version 2's definition — the batch, 2026-10-29

Items found while building the version 2 definition batch (`docs/archive/version_2_definition_plan.md` §2, spec revision 43),
on the owner's instruction of 2026-10-29 to proceed with Claude's recommendations.

- [x] T-199 ⟨later⟩ **`M11.5`'s note for `M11.C.42` says `M4.C.1d`'s contact is "returned to a relieved seeker"
  (found by T-171's learning-qa pass, 2026-11-06; wording only, not decided).** `M4.C.1d` returns contact to its
  sender when the delivery relieves its receiver, so the relieved one is the receiver. The note should say "returned to
  its sender by a relieved receiver", as T-171's note on `M11.C.42` does; a spec wording revision, for the owner.
  *Closed 2026-11-06 (spec revision 54, Claude's recommendation under the owner's instruction to proceed, pending the
  owner's review): the note now says the contact is returned to its sender by a relieved receiver.*
- [ ] T-172 ⟨later⟩ **Three minor items from the W5–W8 reviews (2026-11-01).** (a) `tests/test_requirements.py`'s
  `requirement_specifiers`, like `requirement_names`, raises on a `-r` or `--index-url` line, and the NumPy bound
  check rejects a valid `<3.0` or `<=3` (the safe direction, with a misleading message). (b) Nothing checks that the
  `ruff` `tests/test_lint.py` runs locally matches the pin (this Mac has 0.15.1; CI installs 0.17.3). (c)
  `tests/test_version_2_outputs.py` names its four readout pages instead of finding them, so a fifth would go
  unchecked.
- [x] T-171 ⟨later⟩ **Five version 1 disposition texts describe causes from before stage 3 (found summarising W5,
  2026-11-01; not decided).** `docs/version_2_status.md` shows version 1's text for `M11.C.2[twelve-person family]`
  (it says the pass is proved by removing focus on weakness; it is now carried by witnessing, `witness-path-removed`),
  `M11.C.29` (it describes the level-40 arm, not stage 3's level-60 arm with F's spell paused), and `M11.C.42`, `.44`
  and `.45` (causes found at Phase C's close, before group A). Version 1's file is frozen, so the fix is a version 2
  `note` per row stating the current cause, for the owner's review; no verdict changes. *Added 2026-11-05 (spec revision 51,
  T-195):* the spec's own `M11.C.2` row has the same staleness: its mutation cell and its revision 31 sentence say the
  witness path removed survives and focus on weakness removed turns the twelve-person row red, while the current
  mutation record has `witness-path-removed` red and `weakness-focus-and-steering-removed` surviving. *Closed 2026-11-06 (spec revision 53, Claude's recommendation under the owner's instruction of
  2026-11-05 to proceed, pending the owner's review):* five version 2 `note` rows in `docs/version_2_dispositions.md`
  (`M11.C.2`, where the twelve-person text is shown, `M11.C.29`, `M11.C.42`, `M11.C.44`, `M11.C.45`) state the
  current cause, or that it is not traced since group A; no verdict and no status changed. `tools/version_2_status.py`
  and `tools/spec_coverage.py` showed an applied version 2 note in place of version 1's disposition, which would have
  dropped the five accepted findings; each now shows the note beside it (tests in `tests/test_version_2_status.py` and
  `tests/bowen/test_spec_coverage.py`, shown red first). The spec's `M11.C.2` mutation cell states the current record,
  and a sentence after its revision 31 measurement says the pass needs the witness path.
- [x] T-170 ⟨now⟩ **The stage-end sweep reruns on spec revision 44's engine, `09ca2b46c7e0` (the batch, 2026-10-31;
  not run yet).** Every record but `docs/version_2_sweep_record.md` was regenerated on it;
  `tests/test_version_2_sweep.py::test_s2_11_sweep_record_is_current` is a strict expected failure until the sweep
  reruns, split across both Macs (`docs/two_mac_sweep.md`; Mac Mini1 at 10 workers, this Mac at 8 under `nice -n 15`),
  after a cross-check on `M11.C.16` and `M11.C.23`.
  *Closed 2026-11-01:* the sweep reran on `09ca2b46c7e0`, split across both Macs after a cross-check
  (`docs/two_mac_sweep.md`, "Not verified"). Six criteria moved: `M11.C.17` (8 of 8 to 3 of 8) and `M11.C.21` (4 of 8
  to 0 of 8), whose central verdicts are now FAIL and accepted findings (owner 2026-10-31); `M11.C.2` (2 to 3 of 8),
  `M11.C.7[joint over one spouse]` (4 to 3), `M11.C.2[twelve-person family]` (3 to 2) and `M11.C.10` (4 to 3). The
  strict expected failure is gone from `test_s2_11_sweep_record_is_current`, and `docs/version_2_status.md`'s W4 is met.
- [x] T-165 ⟨later⟩ **`M11.C.17`'s coached rise runs through the coach's I-position pull-up, which the estimator
  counts as the member's own functioning (found tracing `landed-contact-changes-nothing`'s survival, 2026-10-29; not
  decided).** With `perspective_gain` at 0 the coached arm still rises (ad hoc, 150 seeds: +0.0332 ± 0.028). The rise
  comes from `M5.D.5`'s pull-up when the coach's own `I-POSITION` resolves (`src/bowen/engine/iposition.py`,
  `resolve`): the member closes `pull_up_rate` of the gap to the coach's level, and the slow tick's estimator of own
  functioning (`src/bowen/engine/slow_tick.py`) takes out pseudo-self, `adjust_swing` and `floor_held` but not this
  pull-up, so it is credited to the member's basic level. On the four seeds traced, the person who rose was the one who
  received the coach's `I-POSITION`. Spec revision 43 names `coach-pull-up-removed` in the row, so the mutation proves
  the path; whether a pull-up toward the coach's level is "earned in behaviour" (`M11.C.17`, `FE11.6`) is a theory
  question for the owner.
  - 2026-10-30, Claude's recommendation 2026-10-30 under the owner's instruction of that night to proceed without stopping, pending the owner's review (`docs/archive/version_2_definition_plan.md` §8, D5): **not earned.** `FE06.10` (`[K]`: a motivated member can go beyond the therapist's
    level; differentiation depends on knowledge, not on the coach's level) and `FE11.6` (`[B]`: the therapist may be a
    model; instruction makes the family dependent) both place a rise toward the coach's own level outside what
    `M11.C.17` counts, and `M5.D.5`'s reason the gain is solid self (both parties rise) does not hold for a mover
    outside the family. Recommended rule: `M5.D.5`'s pull-up does not apply when the mover is the external agent
    (the alternative, booking it as borrowed functioning the estimator removes, needs a new ledger). Moves the hash;
    `M11.C.17` is then expected to fail (`coach-pull-up-removed` reads −0.00003 ± 0.00017) and need a decision.
  - 2026-10-31, owner 2026-10-31 (`docs/archive/version_2_definition_plan.md` §8, D5, as recommended): **not earned.**
    `M5.D.5`'s pull-up does not apply when the mover is the external agent; it replaces revision 43's
    `coach-pull-up-removed` clause in `M11.C.17` (D1). Built at spec revision 44 with T-161, tests first; `M11.C.17`
    is then expected to fail and is decided again.
  - 2026-10-31, built at spec revision 44 (`src/bowen/engine/iposition.py`, test
    `test_m5d5_t165_an_external_movers_resolve_pulls_no_one_up`, shown red first); `coach-pull-up-removed` retired.
    Regenerated on `09ca2b46c7e0` (Mac Mini1): `M11.C.17` fails, `coached_rise` −0.00003 ± 0.00017, p 0.43 (150
    seeds); in 136 of 150 seeds the two arms' coached rise is the same (`docs/archive/version_2_revision_44_changes.md`).
  - 2026-10-31, owner 2026-10-31, through `AskUserQuestion`: `M11.C.17`'s FAIL accepted as a finding
    (`docs/version_2_dispositions.md`); why the remaining path is flat is T-169. Closed.
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
- [x] T-168 ⟨later⟩ **`tests/test_stale_text.py`'s phrase misses the possessive form (the `learning-qa` review of the
  owner's step 7 decisions, 2026-10-31; not fixed).** `PHRASE` matches "pending owner review" and not "pending the
  owner's review", the form version 2's documents use, so `PENDING` was emptied on 2026-10-31 while the scan could
  not see that form; the review found three such sites the scan missed (`CLAUDE.md`'s revisions 42 and 43,
  `README.md`'s revision 43), fixed by hand the same day. Widen it to both forms; it will then flag MR-22's genuinely
  pending text (revision 40, the spec's status line and `docs/spec_revision_log.md`) and dated history entries, which
  need `PENDING` entries or an exemption rule. Changes no engine file.
  *Closed 2026-11-11 (T-224): moot, `tests/test_stale_text.py` was retired in `273398b`.*
- [ ] T-169 ⟨later⟩ **A coach's completed `I-POSITION` exchange still acts on the member (spec revision 44's
  `learning-qa` review, 2026-10-31; not decided).** Revision 44 (T-165, D5) takes away `M5.D.5`'s pull-up when the mover
  is the external agent. When that exchange completes (`complete`, `src/bowen/engine/iposition.py`), the member target's
  outside-ness axes are still multiplied by `1 − respect_gain`, the tie's `functioning_habit` is set to 0, the shared
  triangle takes `triangle_floor_decrement` and the budget is reduced (`M5.D.7`, `M1.C.5`); whether any of these reaches
  the member's estimated own functioning, and so `M11.C.17`'s coached rise, was not traced, and whether D5's reasoning
  covers them is the owner's. Two smaller items from the same review: `M5.D.5`'s "the payoff **MUST** propagate to
  subsequent movers" is not built (coverage partial), and when it is it must leave an external mover's sequence out;
  and `tests/bowen/test_iposition.py::test_m5d5_opposition_pulls_up_to_the_movers_level`'s last assertion (the mover
  above the mean) always holds, since the mover is set to 60, so the exact-formula assertion above it carries the
  test. No engine file changed.
