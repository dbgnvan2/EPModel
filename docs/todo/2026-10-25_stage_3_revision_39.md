# Version 2 stage 3 — open from revision 39's draft, 2026-10-25

Raised while drafting part 1 of spec revision 39 (`docs/spec_revision_39_draft.md`, group C) at plan step 4
(`docs/implementation_plan_version_2_stage_3.md` §3.8). Items the draft records and does not change.

- [x] T-131 ⟨now⟩ **`KS00.9` against `M7.E.1f`'s parents' term (the stage 4 check, revision 39 part 1).** `KS00.9`
  `[K]`: "the basic levels of the parents establish the emotional climate". `M7.E.1f`'s parents' term reads the
  parents' functioning above basic (the estimator's reading), which `M1.A.5f` (P2) raises in a parent who displaces
  onto the child; in the joint probe the parents' term carries +0.026 ± 0.019 of `M11.C.6`'s +0.018 ± 0.025
  (`docs/PROPOSAL — DISPLACED COPING (P2).md` §6.1). The owner decided 2026-10-25 (the draft's §9 Q5, the recommended
  option): leave `M7.E.1f` as it is in revision 39 and record the finding for the stage 3 milestone review
  (`docs/milestone_review_brief.md`), which reads it under its theory-fidelity lens. A rule in which the parents' term
  leaves out what `M1.A.5f` gave the parent would be a new rule outside D18, with its own probe. *2026-10-28, the
  milestone review's triage: the owner chose option (b), `M1.A.5f`'s transfer on its own ledger, netted out of `d` only;
  a new rule with a probe, in the triage batch (moves the engine hash). The P2 proposal's `KS00.9` citation was corrected
  the same day (`KS00.9` is the anchor `a`, the parents' basic levels).* *Closed 2026-10-28 (spec revision 41, the
  triage batch): `M1.A.5f` books what it moves on each person's `displaced_swing` (the register's new row), and
  `M7.E.1f`'s parents' term reads the parents' functioning net of it; the child's own term keeps the loss (`FE06.3`).
  Probe before the build (`M11.C.6`'s arms, seeds 0–99): the parents' term's contribution +0.1535 ± 0.1963 at revision
  40 and +0.1180 ± 0.1664 with the batch. Regenerated (Phase D part II record): `M11.C.6` still fails, `development`
  +0.0282 ± 0.095, p 0.687 (+0.0534 ± 0.11 at revision 40). Tests, each shown red first:
  `tests/bowen/test_stress_response.py::test_m1a5f_t131_displaced_coping_is_booked_on_its_own_ledger` and
  `tests/bowen/test_life_course.py::test_m7e1f_t131_the_parents_term_leaves_out_displaced_coping`.*
- [x] T-132 ⟨later⟩ **`FE04.13`'s second kind of so-called I-position (the stage 4 check, revision 39 part 2).**
  `FE04.13` `[K]`: many so-called I-positions are attempts to get others to change or to pry oneself loose from
  intense situations. `M5.F.4`'s mixture (revision 39, D20) models the first as the demand share; the second,
  distance in disguise, has no share in the act and is `M11.C.29`'s other arm (`docs/spec_revision_39_draft.md` §17).
  Found, not fixed: no rule drafted. Recorded for the stage 3 milestone review (`docs/milestone_review_brief.md`). *Closed
  2026-10-28: the owner chose to record the second kind as not modelled; `M5.F.4` states `FE04.13` in full and
  `M11.C.29`'s row says its baseline is plain `DISTANCE` (spec revision 40).*
- [x] T-133 ⟨now⟩ **`M5.D.4a`'s anger check at `DEFINE` (the correctness pass on revision 39's draft).** `M5.D.4a`
  says an `I-POSITION` emitted while the mover's anger is above threshold executes as the assertion form; the code
  checks anger only when the act is selected (`src/bowen/engine/act.py`), not again when a sequence reaches `DEFINE`.
  Revision 39's `M5.F.4` describes the code as it is (`docs/spec_revision_39_draft.md` §13.1). Whether anger at
  `DEFINE` should also turn a said position into the assertion form is not drafted; recorded for the stage 3
  milestone review. *The milestone review's triage (its record, `2026-10-28_stage_3_milestone_review.md`): the owner chose option (b): at `DEFINE`, an angry mover's position is all
  demand (share 1) and the sequence ends; `M5.D.4a` restated to match. In the triage batch (moves the engine hash);
  `M11.C.32`, `M11.C.29` and `M11.C.49` rerun with the records.* *Closed 2026-10-28 (spec revision 41, the triage
  batch): the gate is read again when a sequence's `DEFINE` is said, and an angry mover then says the position all as
  demand and the sequence ends (`said_in_anger`); `M5.D.4a`, `M5.F.4` and `M5.F.2c` say so. Regenerated (Phase C
  ensemble record): `M11.C.32` passes, `assertion_form` +0.66 ± 0.25 (+1.33 ± 0.22 at revision 40) and `reached_peak`
  −0.07 ± 0.064; `M11.C.29` still fails (`third_symptom_weeks` +2.54 ± 4.4, p 0.87); `M11.C.49` still fails, its
  `family_excess` now holding (+3.27 ± 2.8, p 0.0158) and its `conflict_share` not (+0.00086 ± 0.026, p 0.468). What it
  does to `M5.D.3`'s first outcome is T-158. Test, shown red first:
  `tests/bowen/test_iposition.py::test_m5d4a_t133_an_angry_mover_at_define_says_it_all_as_demand_and_the_sequence_ends`.*
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
- [ ] T-143 ⟨later⟩ **Mutants that revision 39's regeneration no longer turns red (step 10, 2026-10-26).** From
  `docs/version_2_revision_39_changes.md`, against revision 38's records (`9ebf327`): in the Phase C mutation record
  `steepness-level-independent` now survives on `M11.C.41[heavy: lower level]`, `availability-level-inverted` survives
  and `availability-and-learner-removed` is reversed on `M11.C.16` (`steepness-inverted` turns it red where it
  survived), and `steepness-inverted` survives on `M11.C.38[-5 vs -10]`; the four `M11.C.44` mutants did not run,
  since `M11.C.44` fails (T-142). In the Phase D mutation record `M11.C.17` passes for the first time (FAIL → PASS in
  the Phase D ensemble record) and its one mutant, `landed-contact-changes-nothing`, survives, so that pass is not yet
  proved by mutation; `triangle-help-ignored` is reversed on `M11.C.46` where it was red. Each criterion that passes
  with a surviving mutant needs a mutant that turns it red before its pass counts as coverage (CLAUDE.md), or a reason
  the survivor is expected; read at the stage-end sweep (step 11) and the milestone review; left for the owner. *2026-10-27,
  step 11: the stage-end sweep held `M11.C.16`, `M11.C.41[heavy: lower level]` and `M11.C.17` in 8 of 8 samples each
  (`docs/version_2_sweep_record.md`); a fraction over the ranges is not a mutation proof, so each still waits on the
  milestone review.* *2026-10-28, the triage: the owner kept `M11.C.16` composite (the review's Q5); the documents'
  account of its mutants is T-148; each survivor still needs a mutant that turns it red, or a reason, after the triage
  batch regenerates the records.*
- [ ] T-144 ⟨later⟩ **`M11.C.49` has no declared sweep ranges (step 11, 2026-10-27).** `M11.C.49` is a mixed
  directional criterion, new at revision 39, and `config/sweep/sweep.md` declares no constants for it, so the stage-end
  sweep ran 34 keys, not the 35 the plan's §3.6 counted ("`M11.C.49` adds one Phase C criterion"). W4 is met without
  it, since it fails (S3.25 asks for a fraction for every *passing* directional criterion). Declaring its rows means
  choosing which `[I]` constants its mechanism reads (the I-position's demand share, `M5.F.4`; appraisal; contagion)
  and their ranges, the same judgment as T-125's batch 5; for the owner, before the next sweep that should carry it.
  The owner kept `M17.E.1` *partial* until it is swept (2026-10-27); the sweep record lists it as not swept.
- [x] T-145 ⟨now⟩ **`M11.C.21`'s scripted collapse falls between weeks, so `M6.I.4a` never sees it (step 12's
  `learning-qa` pass, 2026-10-27; major).** `run_tick` reads each person's level at the week's start inside the tick
  (`src/bowen/engine/tick.py:118`), and `arm_c21` drops both parents' levels in a `before` hook
  (`src/bowen/ensemble/criteria_d.py`, `collapse`), which runs between week t0 − 1 and week t0. No week holds the
  drop, so `return_lent_functioning` returns nothing for it, and `M11.C.21`, the criterion the
  `pseudo-self-return-removed` mutant targets, cannot show whether `M6.I.4a` works. Measured 2026-10-27 by an
  in-process probe in the session's scratchpad (nothing in the repository changed), 100 seeds each at the committed
  settings: as built, `target_rise` −0.0409 ± 0.44, p 0.889, FAIL (the Phase D ensemble record's row, reproduced);
  with the week's start read at the previous week's end (fix (a)), −0.870 ± 0.457, p ≈ 1.0, FAIL the other way. The
  return now happens, in week t0, and the readout's baseline is the child's level at the end of week t0
  (`watch` at tick t0 + 1), so what returns is left out of the rise and what follows is the child's settling back. A
  fix is therefore two decisions for the owner: (a) the week's start read at the previous week's end, kept on the
  run state, or (b) the drop moved inside the tick as an event; and where `M11.C.21`'s readout starts (the end of
  week t0 − 1 would count the return). Either moves the engine hash, so every record regenerates and the stage-end
  sweep reruns. Left for the milestone review's triage batch, so the records regenerate once with whatever else it
  finds (Claude's recommendation under the owner's overnight instruction, 2026-10-27, for the owner's review). Not
  T-134 (a return with no lender). *Closed 2026-10-29 (the triage batch, spec revision 41): fix (a), the owner's
  choice in the batch's brief. `run_tick` hands `M6.I.4a` each person's level at the end of the previous week, kept on
  the run state, and `M11.C.21`'s readout starts at the end of week t0 − 1 (Claude's recommendation under the owner's
  instruction of 2026-10-28, pending the owner's review). Tests, each shown red first:
  `tests/bowen/test_tick.py::test_m6i4a_t145_the_weeks_start_is_the_level_at_the_end_of_the_previous_tick`,
  `tests/bowen/test_tick.py::test_m6i4a_t145_a_fall_between_weeks_is_inside_the_next_week`,
  `tests/bowen/test_phase_d_criteria.py::test_m11c21_t145_the_readout_starts_at_the_end_of_week_t0_minus_1`.
  Regenerated on `6ce8be104767`: `M11.C.21` passes (`target_rise` +0.271 ± 0.32, p 0.0139, 100 seeds;
  `docs/phase_d_ensemble_record.md`), and `pseudo-self-return-removed` turns it red (−0.101 ± 0.21, p 0.68;
  `docs/phase_d_mutation_record.md`), so the mutant now shows whether `M6.I.4a` works. The arm's parents still
  over-function after they collapse (T-161).*
- [x] T-146 ⟨now⟩ **Displaced coping takes functioning from a receiver who refuses the act (step 12's `learning-qa`
  pass, 2026-10-27).** `stress_response.displace()` (`M1.A.5f`) moves functioning from every family member named in
  the selection's targets when the act is selected (update-order step 8), before delivery. `M1.B.3c`'s refusal is
  applied at delivery (`rejection.refuse`): a move to a person whose rejection of the sender is above
  `rejection_refusal_threshold` does not reach them, so `M6.I.4`'s exchange moves nothing, yet `displace()` has
  already taken up to `response_up_gain × above/100 × basic/100` from that person. `OVERFUNCTION` across a cut-off tie
  is legal (`REDUCE_CUTOFF` alone requires a severed tie; every other act needs a live one, and a cut-off tie is). `M1.A.5f` says "the
  act's receivers" and does not say whether a refused target is one, so the wording is the owner's first; the fix,
  if a refused target is not a receiver, skips it with `refuse()`'s own test, and moves the engine hash. Batched with
  T-145 for the milestone review's triage (Claude's recommendation, 2026-10-27, for the owner's review). *Closed
  2026-10-28 (spec revision 41, the triage batch): the owner answered through `AskUserQuestion` that a target who would
  refuse the act is not a receiver. `displace()` skips such a target with `rejection`'s own predicate, the one delivery
  applies, so the two cannot drift; `M1.A.5f` says so. Tests in `tests/bowen/test_stress_response.py`:
  `test_m1a5f_t146_a_target_who_would_refuse_is_not_a_receiver` and
  `test_m1a5f_t146_displacement_and_delivery_read_one_refusal_predicate`, each shown red first.*
- [x] T-147 ⟨now⟩ **The engine imports the policy at call time (step 12's correctness pass, 2026-10-27).**
  `src/bowen/engine/iposition.py`'s `demand_share` imports `self_channel_weight` from `src/bowen/policy/policy.py`
  inside the function, so `M5.F.4` and `M4.D.1a` read one definition of `w`; its docstring records this as
  deliberate, to avoid a load-time cycle. No run computes anything wrong. It inverts the layering (the policy imports
  the engine), and a rename in `policy.py` fails only when a `DEFINE` week is reached. Moving `self_channel_weight`
  into an engine module, with the policy importing it, moves the engine hash, so it waits for a batch that moves it
  anyway (T-145's). *Closed 2026-10-28 (the triage batch): `self_channel_weight` is defined in
  `src/bowen/engine/observe.py`; the policy and `demand_share` import it from there, so the engine imports nothing from
  the policy. `tests/bowen/test_engineering_gates.py::test_t147_the_engine_never_imports_the_policy` reads every
  engine module's imports (absolute, `from src.bowen import policy`, and relative) and was proved by mutation with a
  temporary module that imported the policy each way. No rule changed; the move is in the engine hash with the batch.*
