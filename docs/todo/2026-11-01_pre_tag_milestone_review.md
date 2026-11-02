# Version 2 pre-tag milestone review — the triage, 2026-11-01

The review is `docs/cycles/2026-11-01_milestone-review_v2-pre-tag.md` (mini2, branch `review/2026-11-01-pre-tag`,
commit `f19989c`, fast-forwarded into `main` on 2026-11-01). It reviewed `355031f` against `161da4e` and found 0
blocker, 3 major (MR-1 to MR-3) and 12 minor findings (MR-4 to MR-15); none was refuted. The owner answered its eight
questions on 2026-11-01 through `AskUserQuestion`, each as recommended:

1. MR-22 (MR-2): revision 40's restatement of `M13`'s Phase C exit is approved; `M11.C.49`'s verdict is reported as a
   finding, not required to pass.
2. W4's narrow mark (MR-13): W4 closes under D14's deferral. No narrow mark is read from 8 samples; the fractions stay as
   measured. Recorded in `docs/version_2_plan.md` §4 and on `docs/version_2_status.md`.
3. W8: the sign-off of 2026-11-01, given on Claude's recommendation, counts as the owner's review of the outputs.
4. `M11.C.34` (MR-3): its class stays mixed. Its disposition, its spec row and its `M11.5` row now say the pass runs
   through `M4.D.1a`'s premise share and the learned limb is untested; a learner-off mutant is T-173, after the tag.
5. `M11.C.48` (MR-6): the finding says the direction is reversed on both families; a seed with no matched first cutoff
   still enters as 0 against 0, which shrinks the mean difference and does not flip it. No rerun.
6. `M5.D.3` (MR-1): the spec requirement carries the finding note (T-158 (b)).
7. `VERDICT_NOTES` (MR-14): the truth check is run on planted false notes and must refuse each. Pinning the historical
   probe tools stays with T-157.
8. SP-1 (MR-9): `tests/asr_quotations.json`'s `min_words` is 2. The 15 spans it then flags are written sources or the
   project's own words, each allowlisted with its reason; none was an ASR quotation.

Fixed in this commit (spec revision 45, wording only; no file under `src/bowen/` or `config/bowen/` that a run reads
changed, engine hash `09ca2b46c7e0` unchanged): MR-1 (the explainer and the `constants_changes.md` row;
`config/bowen/constants.md` is in the engine hash, so its sentence waits for T-177), MR-2, MR-3, MR-4, MR-5 (spec `M5.D.5` and the explainer graded `[I]`), MR-6, MR-7 (and T-051
closed), MR-8 (`tools/spec_coverage.py` quotes version 2's dispositions), MR-9, MR-10, MR-11, MR-12, MR-13, MR-14. MR-15
and the review's open checks are below.

- [x] T-173 ⟨later⟩ **`M11.C.34`'s learned limb has no mutant (review MR-3 and Unverified 1–2, 2026-11-01).** Add to
  `tools/phase_d_mutation_record.py` a mutant that keeps `M4.D.1a`'s level-dependent share and removes the learned
  targeting (the policy's learner off, or blind to the bearer), and a one-off that fixes the weight only at the policy's
  call site, not in `M5.F.4`'s demand share. Both move no engine hash; the Phase D mutation record is regenerated for
  them alone. If the pass survives the learner-off mutant, the disposition says the result is the premise share's.
  **Done 2026-11-01** (spec revision 46, wording only; engine hash `09ca2b46c7e0` and the Phase D mutation record's
  `3e03e66ade9f` unchanged). Two mutants added to `tools/phase_d_mutation_record.py`, only they ran (150 seeds each,
  this Mac, 8 workers): `mixing-weight-blind-to-level-in-policy-only` red (−0.027 ± 0.13, p 0.35) and
  `learner-off-share-kept` red (+0.027 ± 0.055, p 0.88). The pass did not survive the learner-off mutant, so it is not
  the premise share's alone. The two reds from fixing the share (the declared mutant and the policy-only one) are
  weak: each interval contains the unmutated −0.113, so on a pass at p 0.039 they cannot be told from noise and the
  premise limb is not shown; the policy-only one shows the policy's read alone is enough to lose the pass, and
  `M5.F.4`'s read is not isolated. Learning off also makes `iposition._abort_act`'s draw uniform (correctness pass).
  An ad hoc count, not a record: for each of seeds 0–149 and each arm, `CRITERIA_D["M11.C.34"].arm(arm, seed,
  settings)`'s `automatic_reaction`, run in a copy of `src/`, `config/` and `tools/` with and without
  `learner-off-share-kept`'s edits (applied by `tools/mutant_runner.apply`), 8 processes. Unmutated, the others' extra
  automatic acts toward the bearer average +0.087 per seed at the low level and −0.027 at the high, 0 in 113 and 108
  of 150 seeds; learner off, +0.007 and +0.033, 0 in 147 and 136 of 150. So the reaction is small even unmutated, and
  learning off lowers the low arm's and raises the high arm's. Restated as Claude's recommendation, approved by the owner
  2026-11-02 (T-178): the disposition's row (the owner's review of the PASS verdict unchanged), the spec's `M11.C.34` row
  and its `M11.5` row (class mixed).
- [x] T-178 ⟨now⟩ **The owner's review of spec revision 46's reading of `M11.C.34` (T-173, 2026-11-01).** Revision 46
  supersedes revision 45's reading (the pass is the premise share's, the learned limb untested) with: the pass does not
  hold without learning and is not the premise share's alone, and the premise limb is not shown. Approve it, or say
  otherwise; then drop "pending the owner" from the spec's `M11.C.34` and `M11.5` rows, the disposition's row and the
  status lines, in one commit. **Approved by the owner 2026-11-02** through `AskUserQuestion`, as recommended; the
  markers dropped in one commit.
- [ ] T-174 ⟨later⟩ **`tools/group_g_readout.py` keeps no per-run results (review MR-15, 2026-11-01).** Cache each
  `one_run` in `docs/records_cache/` through `tools/record_cache`, keyed by the engine hash and `(family, seed)`, and add a
  render-only path, so a change of wording does not rerun 20 seeds per family.
- [ ] T-175 ⟨later⟩ **Hand-restated status goes stale (review systemic pattern 1, 2026-11-01; stage 3's pattern 1
  again).** Status lines and labels restate the spec's status line, the revision log or the dispositions, and each new
  revision makes some of them stale (MR-2, MR-4, MR-10, MR-11, MR-13). Make each point to one generated or test-checked
  source instead, extend `tests/doc_claims.json` to labels and revision ranges, and add the checklist to the documents
  a sweep rerun updates.
- [x] T-176 ⟨later⟩ **Why `M11.C.48`'s direction is reversed (review MR-6 and Unverified 3, 2026-11-01).** Trace whether
  comparing matched first cutoffs with the ties that stayed uncut in the same week produces the shorter run. Read only;
  the finding stands until the trace says otherwise.
  **Done 2026-11-02** (read only; engine hash `09ca2b46c7e0` unchanged). `tools/t176_negative_run_trace.py` reruns the
  criterion's 200 seeds on both families, reproduces the record's `negative_run` (−0.319 ± 0.104, −0.113 ± 0.062) and
  sets other baselines beside it (`docs/review/t176_negative_run_trace.md`). The comparison produces the reversal: the
  readout counts an uncut tie only if it had a contact by the cutoff's week, but counts a cut tie with none at a run of
  0 (329 of 1137 and 1419 of 4791 matched cutoffs). With the same condition on both arms the difference is
  +0.029 ± 0.136 on the Phase B family and +0.344 ± 0.078 on the twelve-person family. Matching on relation, or on
  whether the uncut tie is cut off later, does not remove it. On cut ties with a contact, the twelve-person family's
  run lengthens more than the uncut ties' over 13 and 52 weeks (+0.218 ± 0.078, +0.419 ± 0.141); the Phase B
  family's does not differ. The owner approved restating both disposition rows through `AskUserQuestion`
  (2026-11-02), and opened T-179.
- [ ] T-179 ⟨later⟩ **Whether `M11.C.48`'s readout reads both arms under the same contact condition (T-176,
  2026-11-02).** `first_cutoff_runs` counts an uncut tie only if it had a contact by the cutoff's week but counts a cut
  tie with none at a run of 0, and that produces the reversal (`docs/review/t176_negative_run_trace.md`). Decide whether
  to leave out cut ties with no contact, count uncut ties with no contact at 0 (on the twelve-person family this needs
  the week each tie to a person born in the run existed), or keep the readout. The design point in
  `docs/version_2_plan.md` (G1, "Ties with no contact in the window are reported, not dropped") was not followed: the
  readout drops uncut ties with no contact and reports no count of them. A change moves the M11.C.48 rows of the
  ensemble record and needs a spec revision, `docs/readouts_phase_c.md`'s "Negative runs" and the disposition rows with
  it.
- [ ] T-177 ⟨later⟩ **`config/bowen/constants.md:56–57` still says `hold_gain` makes aborts the usual outcome (review
  MR-1, 2026-11-01).** `tools/ensemble_record.py`'s `code_hash` reads every `config/bowen/*.md` but
  `constants_changes.md`, so a note there moves the engine hash; it was left out of revision 45. Add the sentence that
  since revision 41 aborts are not the usual outcome (T-158 (b)) with the next batch that moves the hash.
