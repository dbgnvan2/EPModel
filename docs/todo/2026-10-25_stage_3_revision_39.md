# Version 2 stage 3 — open from revision 39's draft, 2026-10-25

Raised while drafting part 1 of spec revision 39 (`docs/spec_revision_39_draft.md`, group C) at plan step 4
(`docs/implementation_plan_version_2_stage_3.md` §3.8). Items the draft records and does not change.

- [ ] T-131 ⟨later⟩ **`KS00.9` against `M7.E.1f`'s parents' term (the stage 4 check, revision 39 part 1).** `KS00.9`
  `[K]`: "the basic levels of the parents establish the emotional climate". `M7.E.1f`'s parents' term reads the
  parents' functioning above basic (the estimator's reading), which `M1.A.5f` (P2) raises in a parent who displaces
  onto the child; in the joint probe the parents' term carries +0.026 ± 0.019 of `M11.C.6`'s +0.018 ± 0.025
  (`docs/PROPOSAL — DISPLACED COPING (P2).md` §6.1). The owner decided 2026-10-25 (the draft's §9 Q5, the recommended
  option): leave `M7.E.1f` as it is in revision 39 and record the finding for the stage 3 milestone review
  (`docs/milestone_review_brief.md`), which reads it under its theory-fidelity lens. A rule in which the parents' term
  leaves out what `M1.A.5f` gave the parent would be a new rule outside D18, with its own probe.
- [ ] T-132 ⟨later⟩ **`FE04.13`'s second kind of so-called I-position (the stage 4 check, revision 39 part 2).**
  `FE04.13` `[K]`: many so-called I-positions are attempts to get others to change or to pry oneself loose from
  intense situations. `M5.F.4`'s mixture (revision 39, D20) models the first as the demand share; the second,
  distance in disguise, has no share in the act and is `M11.C.29`'s other arm (`docs/spec_revision_39_draft.md` §17).
  Found, not fixed: no rule drafted. Recorded for the stage 3 milestone review (`docs/milestone_review_brief.md`).
- [ ] T-133 ⟨later⟩ **`M5.D.4a`'s anger check at `DEFINE` (the correctness pass on revision 39's draft).** `M5.D.4a`
  says an `I-POSITION` emitted while the mover's anger is above threshold executes as the assertion form; the code
  checks anger only when the act is selected (`src/bowen/engine/act.py`), not again when a sequence reaches `DEFINE`.
  Revision 39's `M5.F.4` describes the code as it is (`docs/spec_revision_39_draft.md` §13.1). Whether anger at
  `DEFINE` should also turn a said position into the assertion form is not drafted; recorded for the stage 3
  milestone review.
- [ ] T-134 ⟨later⟩ **`M6.I.4a`'s return with no lender is not counted (the step 7 `learning-qa` pass, 2026-10-26).**
  When a taker's capacity falls and no tie's balance names them the over-functioner, the amount due
  (`moves.return_lent_functioning`) has no one to go to and stays with the taker, as `M6.I.4a` allows ("what a cap
  stops stays with the taker"). Nothing records it, so a readout cannot tell "nothing returned, no lender" from
  "capacity did not fall". It is common early in a run: `initialise_functioning` sets each person's pseudo-self to
  their swing while every balance starts at 0. A count of the unreturned amount in the effect record, or a readout of
  it for `M11.C.21`, is a log change that moves the golden fingerprints; left for the owner's choice at the stage 3
  milestone review.
- [ ] T-135 ⟨later⟩ **`M11.C.10`'s scripted renewal outlasts a cutoff (the step 7 suite run, 2026-10-26).** The
  treatment renews each cutoff every `renew_every` = 13 weeks (`config/bowen/criteria.md`), and a cutoff's rejection
  (0.5) fades by `rejection_fade` 0.05 a week, so a renewal alone holds it about 10 weeks. Between renewals the cutoff
  is held only if the cutter refuses the other's contact. On the family-of-origin variant, seed 0, two years, the four
  ties were held every week (416) until revision 39's group C changed the run; since, Ravi's cutoff of Viktor lapses
  for 15 weeks (401). `M11.C.10` allows this ("a cutoff that lapsed is visible"), and
  `tests/bowen/test_phase_d2_criteria.py::test_m11c10_family_of_origin_arm_holds_both_parents_cutoffs` now asserts
  that bound, not the 416. Whether `renew_every` should be at most 10, so the scripted arm holds by itself, is a
  criterion-setting change that moves `M11.C.10`'s records; left for the owner.
- [ ] T-137 ⟨later⟩ **`M11.C.49` fails at its first settings (the step 8 targeted run, 2026-10-26).** On Mac Mini1, 150
  seeds (`docs/review/stage_3_step_8_criteria.md`): the act's effect on the family's summed excess over six weeks is
  +2.2 ± 3.0 higher at level 35 than at 60, and on the `CONFLICT` share of the F–M tie's moves +0.003 ± 0.028; neither
  resolves, so the composite limbs do not hold. The premise limb holds as reported (the target's felt impingement
  +0.085 ± 0.023), and the act completes less at 35 (−0.31 ± 0.074). The settings (`config/bowen/criteria.md`: 25
  weeks, `t0` 1, a six-week window) were set before the row first ran and are not to be tuned to it (`M10.B.4`); the
  row's mutants did not run, since it does not pass. Read with the other composite results at step 10's
  regeneration; left for the owner.
- [ ] T-138 ⟨later⟩ **`M11.C.29`'s genuine `I-POSITION` almost never completes after D21 (the step 8 targeted run,
  2026-10-26).** With the mover at level 60 and F's spell paused over the act's window (D21), the row fails:
  `third_symptom_weeks` +0.9 ± 3.7 and `budget` +0.04 ± 0.055 (100 seeds, `docs/review/stage_3_step_8_criteria.md`).
  In 30 seeds of the treatment arm, ad hoc: the scripted act is placed in 26 and owed a step in 4 (F already in a
  sequence); F's first sequence ends `prepare_failed` in 16, `aborted` in 8 and `did_not_land` in 1, and none ends in
  5; none completes. Over the run's 100 seeds the genuine arm completes in one (seed 82) and the control in three (22, 69,
  82), so `COMPLETED` is met and the budget limb reads one exchange. The pause takes
  F's own spell off `PREPARE`, but the excess that fails it is still there. Whether the arm needs another condition
  (M's spell over the same weeks, or a lower load) is a criterion-setting choice; left for the owner.
- [ ] T-139 ⟨later⟩ **Two arm settings read outside the run's parameters (the step 8 `learning-qa` pass, 2026-10-26;
  P4).** `act_window` in `src/bowen/ensemble/criteria.py` reads `prepare_ticks` from `load_constants()`, not from the
  run's parameters, so an arm that overrode `prepare_ticks` would pause the spell over the wrong weeks; none does, and
  the sweep edits a copy of `constants.md`, so the two agree today. `mover_at` sets the mover's systems perspective to
  1.0 in code (the owner's Q13 for `M11.C.49`, and `M11.C.29`'s and `M11.C.32`'s arms before it); a criteria setting
  in `config/bowen/criteria.md` would declare it. Neither changes a result now.
- [ ] T-140 ⟨later⟩ **`M5.C.1`'s gate and `M4.C.1d`'s returned contact (the step 9 `learning-qa` pass, 2026-10-26).**
  Under marital distance, `M5.C.1` stops a move addressed to a symptomatic child from improving them; the contact a
  relieved person returns to that child under `M4.C.1d` is not a move addressed to them, so it calms them. In a probe
  (a scratch copy), a symptomatic Nadia sends `STAY-IN-CONTACT` to a relieved Marta: Nadia's acute anxiety moves by
  −0.91 with `reciprocal_contact_share` at 0.5 and by +0.49 (the speaker echo only) at 0. The other direction holds
  too: a gated target, whose relief is set to 0, still counts as relieved by `M4.C.1`'s deviation and returns contact.
  The approved text is silent on how the two rules meet. The owner chose to keep the code as built (2026-10-26);
  whether to gate it (one line in `apply_appraisal`, a sentence in `M4.C.1d`, a test) is left for the owner's review.
- [ ] T-141 ⟨later⟩ **The sender's appraisal of returned contact reads the event's sign and attention (the step 9
  `learning-qa` pass, 2026-10-26; low).** `returned_contact` in `src/bowen/engine/appraise.py` signs the sender's
  appraisal with `sign(event.kind, event.source_position)`, the receiver's sign for the sender's position, and scales
  it by `attention_factor(sender, event, params)`, where `event.attention` describes the receiver's attention. Every
  sign in `config/bowen/` is +1 and nothing in `src/` sets an attention other than none, so neither changes a run now;
  a −1 sign in config would invert the sender's relief.
- [ ] T-142 ⟨later⟩ **Group A does not turn `M11.C.42` or `M11.C.45` green, and `M11.C.44` stops passing (the step 9
  targeted run, 2026-10-26).** On Mac Mini1 at engine hash `2a3b84e2d89a` (`docs/review/stage_3_step_9_criteria.md`):
  `M11.C.42`'s `triangle_reuse` −0.06 ± 0.36 over 50 seeds (revision 38's record −0.34 ± 0.35), so S3.12 is not met;
  `M11.C.45`'s `triangle_rate` +0.0013 ± 0.0055 over 150 (−0.0019 ± 0.0047), so S3.13 is not met; and `M11.C.44`'s
  `outside_inside_ratio` −0.020 ± 0.089 over 150, against a pass at +0.10 ± 0.092 in the record (S3.14 reports it
  beside version 1's verdict). Since none passes, their mutants (`reciprocal-contact-removed` among them) did not run.
  The `M11.C.27` cells keep their verdicts (three fail, `[unstable,remove_one]` passes), as do `M11.C.16` and
  `M11.C.41`'s four cells. The settings and `reciprocal_contact_share` (0.5, `[I]`) are not to be tuned to these rows
  (`M10.B.4`); read with step 10's regeneration and the sweep; left for the owner.
