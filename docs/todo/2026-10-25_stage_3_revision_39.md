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
