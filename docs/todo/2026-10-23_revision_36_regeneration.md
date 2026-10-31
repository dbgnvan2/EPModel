# Revision 36's regeneration — open after the records, 2026-10-23

Raised when every record was regenerated once on the engine hash `66dcbb1e8ba5` (`CHANGELOG.md`, the entry of
2026-10-23; `docs/version_2_revision_36_changes.md`). Each verdict below is a composite in `M11.5`, so none is a
premise passing, and a changed one is a new result under revision 36's rules, kept beside the earlier verdict.

- [x] T-119 ⟨now⟩ **`M11.C.48`'s named mutant survives on both families:** `cutoff-blind-to-learned-value` (`CUTOFF`
  weighted as the best-valued act whenever it is legal) leaves `negative_run` at +0.149 ± 0.092 (p 0.00099) on the
  Phase B family and +0.181 ± 0.041 (p 1.4e-11) on the twelve-person family, from +0.43 ± 0.15 and +0.476 ± 0.06
  unmutated (`docs/phase_d_mutation_record.md`), while the criterion's row says the mutant must turn it red. The
  criterion passes, but the pass is not evidence (T-123: the baseline is biased toward a pass), so A11 of the G1
  proposal is not met. What the mutant leaves standing may be that artefact rather than a learned value; it is not known. Find it by mutation, then either a mutant that
  bites or an amended row — the row is the spec's, so that change is the owner's.
  *Closed 2026-10-30 (spec revision 43, Claude's recommendation under the owner's instruction of 2026-10-29, pending
  owner review):* with T-123's fix `M11.C.48` fails on both families (`docs/phase_d2_ensemble_record.md`), so its
  mutant is not run (`docs/phase_d_mutation_record.md`, "Not run"); the failure is an accepted finding in
  `docs/version_2_dispositions.md`, for the owner.
- [x] T-120 ⟨now⟩ **`M11.C.15`'s named mutant survives:** `death-bound-anxiety-discarded` leaves `death_rise` at
  +1.45 ± 0.69 (p 0.00017), from +0.128 ± 0.7 at stage 1, where it turned the criterion red. At revision 36 a death
  also loads the survivors with grief sized from the scale and the tie (`M6.3g`), so the rise at a death no longer
  rests on the bound anxiety's release alone; whether that is the whole account is not measured. The row says this
  mutant must turn the criterion red: a mutant that removes both, or an amended row (the owner's).
  *Closed 2026-10-30 (spec revision 43, Claude's recommendation under the owner's instruction of 2026-10-29, pending
  owner review):* the row names `death-all-bound-anxiety-discarded`, which also discards the dead person's
  distance-bound and triangle stock; it turns the criterion red (−1.32 ± 0.79, `docs/phase_d_mutation_record.md`).
  The first mutant survives and is reported. The arm runs with grief off, so the survivor was the stock the first
  mutant left, not grief.
- [x] T-121 ⟨now⟩ **`M11.C.11`'s named mutant survives:** `removed-member-hold-released-while-away` leaves
  `disturbance` at +1.30 ± 1.18 (p 0.0074), from −0.593 ± 1.37 at stage 1 (red); the unmutated effect grew from
  +3.37 ± 1.6 to +6.32 ± 1.3. The mutant still removes most of the effect; what remains, and which revision 36 rule
  carries it, is not traced.
  *Closed 2026-10-30 (spec revision 43, Claude's recommendation under the owner's instruction of 2026-10-29, pending
  owner review):* what remained was the reunion's return of the ties' distance-bound stock (`M1.D.2a` (c)). The row
  names `removed-member-holds-not-released-on-return`, which removes both, and it turns the criterion red (+0.426 ±
  1.1, p 0.17, `docs/phase_d_mutation_record.md`).
- [x] T-123 ⟨now⟩ **`M11.C.48`'s baseline counts a first cutoff with no matched uncut tie as 0 (learnings P7):**
  `first_cutoff_runs` gives such a first cutoff (no other tie with a contact in the window by that week) a baseline of
  0 and keeps its own run in the treatment arm, which biases the comparison toward a pass. Measured on the regenerated
  runs at the re-sweep of 2026-10-23: 139 of 700 first cutoffs on the Phase B family (100 seeds) and 284 of 749 on the twelve-person family (20 seeds); with those left out of both arms the difference is −0.28 ± 0.15 and −0.21 ± 0.18 (2 se) against +0.43 ± 0.16 and +0.44 ± 0.10 as built. So the pass recorded on 2026-10-23 is not
  evidence, and A11 of the G1 proposal is not met. Recommended rule, for the owner: such a first cutoff is left out of
  both arms and counted (`FirstCutoff` carries the count; the group G readout and the criterion's books report it);
  the spec row says what it contributes is undecided. The fix changes `src/bowen/readouts/negative_runs.py`, which
  is in the engine hash, so it goes with the next engine-hash move and one regeneration, batched with T-119 to T-121.
  *Stage 2 (2026-10-23) did not move the hash and did not make this fix; the version 2 status page shows `M11.C.48`
  passing with its mutant surviving, so W3 is not met on it.*
  *Closed 2026-10-30 (spec revision 43, Claude's recommendation under the owner's instruction of 2026-10-29, pending
  owner review):* an unmatched first cutoff is left out of both arms and counted (`FirstCutoff.matched`, `unmatched`,
  the report-only readout `unmatched_cutoffs`);
  `tests/bowen/test_negative_runs.py::test_m11c48_t123_an_unmatched_first_cutoff_is_left_out_of_both_arms_and_counted`.
  On `da5981f6ec0b` the criterion fails on both families (`docs/phase_d2_ensemble_record.md`). The group G readout
  still averages unmatched cutoffs (T-166).
- [ ] T-122 ⟨later⟩ **Four composite verdicts PASS → FAIL at revision 36, for the owner's review:** `M11.C.17`
  (`coached_rise` +0.000272 ± 0.00042, p 0.205, from +0.000888 ± 0.0013, p 0.0148), `M11.C.7[joint over one spouse]`
  (`deviation_fall` +0.0309 ± 0.056, p 0.177, from +0.0429 ± 0.052, p 0.0497), `M11.C.10[family of origin]`
  (`cut_off_tie_weeks` −31.8 ± 130, p 0.674, from +238 ± 150, p 0.00107) and `M11.C.2[twelve-person family]`
  (`symptom_load` +0.947 ± 1.6, p 0.118, from +3.42 ± 1.8, p 0.000236); and one FAIL → PASS, `M11.C.44`
  (`outside_inside_ratio` +0.103 ± 0.092, p 0.0298, from +0.00101 ± 0.096), whose mutant `deviation-inverted`
  survives, as it did when the same verdict moved at revision 29 (`docs/implementation_plan_phase_d.md` §4.8).
  Whether any of the four is a defect of revision 36's rules or a finding is the owner's call; the mutation rows that
  left the Phase D record with those passes are listed in the changes page. While `M11.C.10[family of origin]`
  fails, its mutation clause (`M4.D.6f` removed **MUST** turn it red) cannot be proved: a mutant runs only against
  a passing criterion, and `test_m11c10_family_of_origin_variant_is_proved_red_by_removing_m4d6f` now pins the
  missing row to that one cause (the re-sweep of 2026-10-23).
  - 2026-10-30, Claude's recommendation 2026-10-30 under the owner's instruction of that night to proceed without stopping, pending the owner's review (`docs/version_2_definition_plan.md` §8, D3): the four PASS → FAIL rows of revision 36 are back at the tag's verdict and no longer
    held. Two changed verdicts remain, both from revision 41: `M11.C.34` PASS, recommended as reviewed (`verdict
    reviewed` in `docs/version_2_dispositions.md`; its declared mutant turns it red; the sweep holds it in 1 of 8), and
    `M11.C.21` PASS, not recommended (a `note` row): its pass rests on T-161's scripted lending; T-161 (b) first. Open
    until the owner writes `owner` and a date in those rows.
