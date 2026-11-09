# Changelog archive

The entries of `CHANGELOG.md` dated before 2026-10-19, the day version 2 began at the tag `version-1`, moved
here verbatim on 2026-10-22, the entries of 2026-10-19 and 2026-10-20 on 2026-10-23, and the entries of
2026-10-21 later on 2026-10-23, the earliest entry of 2026-10-22 (the SP-1 proposal's eight questions) on 2026-10-24, and the next (SP-1 built, with spec revision 34) later on 2026-10-24, and the next (Model ergonomics, Parts A–C) on 2026-10-25, and the next (Model ergonomics, Part D) later on 2026-10-25, and the next (Model ergonomics, Part E) later on 2026-10-25, and the next (Model ergonomics follow-ups) later on 2026-10-25, and the next (the closed and old parts of `TODO.md` and `CHANGELOG.md` archived) on 2026-10-26, and the next (`TODO.md` an index over records; a changelog index) later on 2026-10-26, and the next (Revision 35's dispositions approved by the owner) later on 2026-10-26, and the next two (Spec revision 36, of 2026-10-22, and Revision 36's records regenerated, of 2026-10-23) later on 2026-10-26, and the next five (from the cloud sessions' SessionStart hook to Stage 2: Phase C's directional criteria swept, all of 2026-10-23) later on 2026-10-26, and the next three (from the night's decisions listed for the owner to Stage 2: Phase D part I's directional criteria swept, all of 2026-10-23) on 2026-10-27, and the next (Stage 2: the sweep can run on two Macs, of 2026-10-23) later on 2026-10-27, and the next (The two-Mac cross-check run live, of 2026-10-23) on 2026-10-28, and the next four (from Stage 2: Phase D part II's sweep ranges declared to When the full sweep runs, the last of 2026-10-23) on 2026-10-29, and the next three (from Stage 2: Phase D part II's seven criteria swept across two Macs to The stage 2 milestone review triaged, spec revision 38, the earliest of 2026-10-24) later on 2026-10-29, and the next five (from Version 2 stage 3's implementation plan drafted to Rule: a finished session writes the next session's brief, the last of 2026-10-24) on 2026-10-30, and the next (Stage 3 step 3: P2 (displaced coping) drafted with F1, probed, for the owner's approval, the earliest of 2026-10-25) on 2026-10-31, and the next three (from The two-Mac driver reaches Mac Mini1 through the `macmini1` alias to Stage 3 step 3: P2 and F1 probed together on Mac Mini1, all of 2026-10-25) later on 2026-10-31, and the next (Stage 3 step 4: spec revision 39, part 1 drafted (group C), for the owner's approval, of 2026-10-25) on 2026-11-01, and the next (Stage 3 step 4: spec revision 39, parts 2 and 3 drafted; revision 39 approved by the owner, of 2026-10-25) later on 2026-11-01, and the next (Stage 3 step 5: the tests first, red, of 2026-10-25) later on 2026-11-01, and the next (Stage 3 step 6: the batch's first commit, T-130's wording, T-014 and T-129, of 2026-10-25) later on 2026-11-01, and the next (Stage 3 step 7: group C, displaced coping, the floor-held ledger and lent functioning returned, of 2026-10-26) on 2026-11-03, and the next (Stage 3 step 8: group F, the I-position as a mixture of self-statement and demand, of 2026-10-26) on 2026-11-04, and the next (Stage 3 step 9: group A, contact returned by a relieved receiver and TRIANGLE with distance, of 2026-10-26) later on 2026-11-04, and the next (Stage 3 step 10: the sweep's named ratio, one regeneration, the open batch closed, of 2026-10-26) on 2026-11-05, and the next (Stage 3 step 11: the stage-end sweep with the named ratio, of 2026-10-27) later on 2026-11-05, and the next (Stage 3 step 12: the reviews of stage 3's whole diff, of 2026-10-27) later on 2026-11-05, and the next (Stage 3 milestone review triage: spec revision 40, wording only, of 2026-10-28) later on 2026-11-05, and the next (Stage 3 milestone review triage batch: spec revision 41, one regeneration, of 2026-10-28) on 2026-11-06, and the next (Stage 3's stage-end sweep on revision 41's engine; spec revision 42, wording only, of 2026-10-29) later on 2026-11-06, and the next (Version 2's definition: the plan, and version 2's dispositions, of 2026-10-29) later on 2026-11-06, and the next (The version 2 definition batch: spec revision 43, one regeneration, of 2026-10-30) later on 2026-11-06, and the next (The stage-end sweep on revision 43's engine, of 2026-10-30) later on 2026-11-06, and the next (Version 2's definition, step 7: Claude's recommendations for the owner's review; T-166, of 2026-10-30) later on 2026-11-06, and the next (Version 2's definition, step 7: the owner's decisions D1–D6, of 2026-10-31) later on 2026-11-06, and the next (Spec revision 44: the owner's D4 and D5, one regeneration, of 2026-10-31) later on 2026-11-06, and the next (The stage-end sweep on revision 44's engine; the owner's decisions on `M11.C.21` and `M11.C.17`, of 2026-11-01) on 2026-11-07, and the next (Version 2's W5 to W8: the owner's sign-offs; T-151, of 2026-11-01) later on 2026-11-07, and the next three (Version 2 tagged, T-173's learned limb under two mutants, both of 2026-11-01, and T-176: why `M11.C.48`'s direction is reversed, a trace (2026-11-02)) on 2026-11-09, so that the live file stays short enough to read (`tests/test_doc_size.py` bounds its size). The heading below is the section they
were under; nothing in this file is reworded. Later entries are in `CHANGELOG.md`.

## [Unreleased] — 2026-10-11

### T-176: why `M11.C.48`'s direction is reversed, a trace (2026-11-02)

- **What.** `tools/t176_negative_run_trace.py` runs `M11.C.48`'s 200 seeds on both families as the criterion runs
  them (this Mac, `nice -n 15`, 8 workers, 19 min), reproduces the record's `negative_run` (−0.319 ± 0.104 and
  −0.113 ± 0.062) and sets the cut tie's run against other baselines (`docs/review/t176_negative_run_trace.md`). The
  readout counts an uncut tie only if it had a contact by the cutoff's week but counts a cut tie with none at a run of 0;
  with the same condition on both arms the difference is +0.029 ± 0.136 (Phase B family) and +0.344 ± 0.078
  (twelve-person family). Matching the uncut ties on relation does not remove the reversal. On cut ties with a
  contact, the twelve-person family's run lengthens more than the uncut ties' over 13 and 52 weeks; the Phase B
  family's does not differ. The review passes found the run-before rows carried the same bias; they were
  restricted to cut ties with a contact and the trace rerun once (20 min). The owner restated both
  disposition rows through `AskUserQuestion` and opened T-179 (whether the readout should read both arms alike). Read
  only: no rule, readout or record changed, and the engine hash (`09ca2b46c7e0`) did not move.
- **Documents.** `docs/review/t176_negative_run_trace.md` (new, generated; its "Reading" hand-written),
  `docs/version_2_dispositions.md`, `docs/version_2_status.md`, `docs/spec_index.jsonl`, `docs/spec_rules/`,
  `docs/spec_coverage.md` (regenerated), T-176's and T-179's record, `TODO.md`, `docs/TODO_archive.md`,
  `docs/SPEC_REVIEW_CHECKLIST.md`, `docs/REVIEWER_READING_LIST.md`, `docs/REVIEW_version_2_outputs.md` (a row for the
  new page, the owner's mark open), `CLAUDE.md` (the suite count); the test `tests/test_t176_negative_run_trace.py`,
  and the page added to `tests/test_version_2_outputs.py`'s framing list; the page listed as not compared in
  `tools/record_changes.py`, so each of the six change pages (`docs/version_2_stage_1_changes.md`,
  `docs/version_2_revision_*_changes.md`, regenerated) gains that one line and nothing else. The suite: 1295 tests,
  1293 pass and 2 strict expected failures.

### T-173: `M11.C.34`'s learned limb under two mutants; spec revision 46, wording only (2026-11-01)

- **What.** Two mutants for `M11.C.34` added to `tools/phase_d_mutation_record.py` (the pre-tag review's MR-3 and
  Unverified 1–2); only they ran (150 seeds each, this Mac). `mixing-weight-blind-to-level-in-policy-only` (the weight
  fixed at the policy's call site, `M5.F.4` untouched): red, −0.027 ± 0.13, p 0.35. `learner-off-share-kept` (learning
  off, `M4.D.1a`'s share kept): red, +0.027 ± 0.055, p 0.88. The pass does not hold without learning; the premise
  limb is not shown (both share-fixing reds lie within noise). Revision 45's reading is superseded, Claude's recommendation, approved by the owner
  2026-11-02 (T-178). Each mutant runs in a temporary copy: the engine hash (`09ca2b46c7e0`) and the Phase D mutation record's
  (`3e03e66ade9f`) did not move.
- **Documents.** `docs/phase_d_mutation_record.md` (regenerated), the spec's `M11.C.34` and `M11.5` rows and front
  matter, `docs/spec_revision_log.md`, `docs/version_2_dispositions.md`, `docs/version_2_status.md`,
  `docs/spec_index.jsonl`, `docs/spec_rules/`, `docs/spec_coverage.md` (regenerated), T-173's record, `TODO.md`,
  `README.md`, `CLAUDE.md`, `docs/theory/_STATUS.md`, `docs/SPEC_REVIEW_CHECKLIST.md`, `docs/REVIEWER_READING_LIST.md`;
  the review exports' revision label (`docs/review/phase_c_*_seed7.*`, regenerated); `docs/version_2_revision_44_changes.md`
  (regenerated; its prose, in `tools/record_changes.py`, names T-173's rows as not revision 44's). T-178 is the owner's
  review of the new reading. The oldest live entry (stage 3 step 6) moved to `docs/CHANGELOG_archive.md` verbatim, for
  `tests/test_doc_size.py`'s limit.

### Version 2 tagged (2026-11-01)

- **What.** The owner tagged `version-2` at `8c4e0bf` (annotated, pushed): spec revision 45, engine hash
  `09ca2b46c7e0`, W1–W8 holding, the pre-tag milestone review triaged. CI run 41369663069 green on Python 3.11–3.13.
- **Documents.** `docs/theory/_STATUS.md`, `docs/version_2_definition_plan.md` (status line, §7), `README.md`,
  `CLAUDE.md`. Nothing a run computes changed; the engine hash did not move.

### The pre-tag milestone review's triage: spec revision 45, wording only (2026-11-01)

- **What.** The review `docs/cycles/2026-11-01_milestone-review_v2-pre-tag.md` (branch `review/2026-11-01-pre-tag`,
  fast-forwarded into `main`) found 3 major and 12 minor findings. The owner answered its eight questions on 2026-11-01,
  each as recommended (`docs/todo/2026-11-01_pre_tag_milestone_review.md`): MR-22 approved; W4 closed under D14's
  deferral (no narrow mark from 8 samples); the W5 and W8 sign-offs count as the owner's review; `M11.C.34`'s pass
  restated as running through `M4.D.1a`'s premise share, its learned limb untested (T-173); `M11.C.48`'s finding says the
  direction is reversed; `M5.D.3` carries its finding note; the verdict-note check proved on planted notes; SP-1 at two
  words.
- **Spec revision 45** (wording only): the status line, `M5.D.3`'s finding note, `M5.D.5`'s coach rule graded `[I]`
  with its sources as context, `M11.C.34`'s row (revision 27's measurement marked superseded; revision 44's measurement,
  its mutant and the owner's reading added) and its `M11.5` row, MR-22 recorded as approved.
- **Review.** One `learning-qa` pass and one correctness pass. Fixed: a note added to `config/bowen/constants.md` moved
  the engine hash (that file is hashed), so it was taken out (T-177); 62 and 66 count sequences, not seeds; W4's note
  reads the sample count from `config/sweep/sweep.md`; SP-1's form A notes the two-word limit.
- **Code and tests.** `tools/spec_coverage.py` quotes version 2's dispositions in place of version 1's for the same
  key (`test_m141_version_2_disposition_replaces_version_1`, shown failing on version 1's rows alone).
  `tools/version_2_status.py` prints W4's deferral. `tests/asr_quotations.json`'s `min_words` is 2, with 15 spans
  allowlisted, each a written source or the project's own words. `test_m17a1_each_verdict_note_is_true_of_the_records_and_the_todo_list`
  refuses three planted false notes.
- **Filed.** T-173 (`M11.C.34`'s learner-off mutant), T-174 (`group_g_readout.py`'s cache), T-175 (status lines point
  to one source), T-176 (`M11.C.48`'s reversal traced), T-177 (`constants.md`'s `hold_gain` sentence). T-051 closed, superseded.
- **Suite.** 1293 tests: 1291 pass and 2 are strict expected failures (`M5.B.3`; `M5.D.3`, T-158); one test added (`test_m141_version_2_disposition_replaces_version_1`). Measured 2026-11-01 on this Mac with `python3 -m pytest tests/`; 52 `ensemble`-marked tests deselected.
- **Documents.** The spec, `docs/spec_revision_log.md`, `docs/model_explainer.md` (MR-1, MR-5, MR-12, its date),
  `config/bowen/constants_changes.md` (`hold_gain`'s note; no value changed; `constants.md` is in the engine hash, so
  its sentence waits for T-177),
  `docs/version_2_dispositions.md`, `docs/version_2_plan.md` §4, `docs/version_2_definition_plan.md`,
  `docs/theory/_STATUS.md`, `docs/implementation_plan_version_2_stage_3.md`, `docs/PROPOSAL — DISPLACED COPING (P2).md`,
  `docs/PROPOSAL — ASR QUOTATIONS (SP-1).md` (SP1.2), `docs/two_mac_sweep.md`, `docs/REVIEW_version_2_outputs.md`,
  `docs/REVIEWER_READING_LIST.md`, `docs/SPEC_REVIEW_CHECKLIST.md`, `README.md`, `CLAUDE.md`, `docs/todo/`, and the
  generated pages (`docs/version_2_status.md`, `docs/spec_coverage.md`, the spec indexes, `TODO.md`, and the three
  `docs/review/phase_c_*_seed7` pages, whose only change is the revision they stamp). Nothing a run
  computes changed and the engine hash did not move (`09ca2b46c7e0`), so no record was rerun and
  `docs/readouts_phase_c.md` needed no change. This file's earliest entry moved verbatim to `docs/CHANGELOG_archive.md`
  for the size limit.

### Version 2's W5 to W8: the owner's sign-offs; T-151 (2026-11-01)

- **What.** `docs/version_2_definition_plan.md` §5. W7: CI green on `5957648` on Python 3.11–3.13 (run 41302506920).
  T-151: `requirements.txt` pins `ruff==0.17.3` (what CI installed from about 2026-10-28; 0.17.2 before) and bounds NumPy to
  `>=1.26,<3` (the sweep's 1.26.4 and CI's 2.4.6 and 2.5.4 all pass); tests `tests/test_requirements.py::test_t151_*`,
  shown red first, and `test_m17e1_sweep_machines_pin_numpy_and_ci_does_not` restated to "not pinned". W6: the
  checklist, spec-consistency and record-current tests green.
- **W5, signed off by the owner** (2026-11-01, on Claude's recommendation): the results of stage 3's groups C, F and A.
  None moved its target criteria the way its proposal expected; each stands beside version 1's as measured, not tuned.
  T-171 filed: five version 1 disposition texts describe causes from before stage 3.
- **W8, signed off by the owner** (2026-11-01): `docs/REVIEW_version_2_outputs.md` lists every kind of output, version
  1's and version 2's, with its framing. `tests/test_version_2_outputs.py` checks that version 2's four readout pages
  carry "Not a real family (`M11.F.9`)" (shown failing on a page with the line removed).
- **Review.** One `learning-qa` pass and one correctness pass on the diff. Fixed: `ruff` 0.17.3 is what CI installed
  from about 2026-10-28, not on every run (0.17.2 on 2026-10-24); CLAUDE.md's line on where the changelog starts.
  T-172 filed with three minor items (parser limits, local `ruff` against the pin, the W8 test's fixed page list).
- **Suite.** 1292 tests: 1290 pass and 2 are strict expected failures (`M5.B.3`; `M5.D.3`, T-158). Measured 2026-11-01 on this Mac with `python3 -m pytest tests/`; 52 `ensemble`-marked tests deselected.
- **Documents.** `docs/REVIEW_version_2_outputs.md` (new), `docs/version_2_definition_plan.md` (status line, §5, §7),
  `docs/implementation_plan_version_2_stage_3.md` (W5, W8), `docs/two_mac_sweep.md`, `requirements-sweep.txt`'s
  comment, `docs/SPEC_REVIEW_CHECKLIST.md`, `docs/REVIEWER_READING_LIST.md`, `README.md`, `docs/theory/_STATUS.md`,
  `CLAUDE.md`, `docs/todo/` (T-151 closed, T-171 and T-172 filed), `TODO.md`, `docs/spec_coverage.md` (regenerated); this file's earliest entry moved verbatim to
  `docs/CHANGELOG_archive.md` for the size limit. No rule, constant, criterion or readout changed and
  the engine hash did not move, so the spec, the explainer, `docs/readouts_phase_c.md` and
  `config/bowen/constants_changes.md` needed no change.

### The stage-end sweep on revision 44's engine; the owner's decisions on `M11.C.21` and `M11.C.17` (2026-11-01)

- **What.** The rest of the revision 44 brief (T-170): the stage-end sweep rerun on `09ca2b46c7e0`, split across both
  Macs (`docs/two_mac_sweep.md`). The cross-check (`M11.C.16`, `M11.C.23` on both Macs, `--recompute`) found 491
  results equal and added 0. Mac Mini1 ran `M11.C.6`, `M11.C.10`, `M11.C.10[family of origin]` and `M11.C.15` at 10
  workers; this Mac ran the other 30 at 8 workers under `nice -n 15`, from a clean worktree. Both started at 17:13 and
  exited 0 (04:17 and 04:18); the merge added Mac Mini1's 47 results and found 491 equal.
- **Results** (`docs/version_2_sweep_record.md`). `M11.C.17` holds 3 of 8 samples (8 of 8 before) and `M11.C.21` 0 of
  8 (4 of 8 before); `M11.C.2` 3 of 8 (2), `M11.C.7[joint over one spouse]` 3 of 8 (4), `M11.C.2[twelve-person
  family]` 2 of 8 (3), `M11.C.10` 3 of 8 (4). Every other key's samples are unchanged.
- **The owner's decisions** (2026-10-31, through `AskUserQuestion`): `M11.C.21` and `M11.C.17`, each failing, are
  accepted findings; no rule writes either result. Recorded in `docs/version_2_dispositions.md` and in each row of the
  spec's `M11.5` table; T-122, T-161 and T-165 closed with them. `docs/version_2_status.md`: W1, W2, W3 and W4 met.
- **Tests.** `test_s2_11_sweep_record_is_current` lost its strict expected-failure mark.
- **Suite.** 1287 tests: 1285 pass and 2 are strict expected failures (`M5.B.3`; `M5.D.3`, T-158). Measured 2026-11-01 on this Mac with `python3 -m pytest tests/`; 52 `ensemble`-marked tests deselected.
- **Documents.** The spec (status line; the two `M11.5` rows' decisions), `docs/spec_revision_log.md`,
  `docs/version_2_dispositions.md`, `docs/two_mac_sweep.md` ("Not verified"), the plan's status line and §7,
  `docs/todo/` (T-122, T-161, T-165 and T-170 closed), `docs/REVIEWER_READING_LIST.md`, `README.md`,
  `docs/theory/_STATUS.md`, `CLAUDE.md`, `tests/doc_claims.json` (the status page and the sweep record listed as
  generated records with no status line, now that status lines name them); this file's earliest entry of 2026-10-25
  moved verbatim to `docs/CHANGELOG_archive.md` for the size limit; generated: the sweep record, the status page, the
  coverage page, the indexes, `docs/spec_rules/` and `TODO.md`. No rule, constant, criterion or readout changed and the engine hash did not move,
  so the explainer, `docs/readouts_phase_c.md` and `config/bowen/constants_changes.md` needed no change.

### Spec revision 44: the owner's D4 and D5, one regeneration (2026-10-31)

- **What.** The batch from the owner's D4 and D5 of 2026-10-31 (`docs/version_2_definition_plan.md` §8). D4 (T-161
  (b)): `arm_c21`'s scripted `OVERFUNCTION` toward C1 runs from week 0 to `t0 − 1` in both arms and ends at `t0`, so
  after it the parents act on their own policy (`src/bowen/ensemble/criteria_d.py`, `config/bowen/criteria.md`). D5
  (T-165): `M5.D.5`'s pull-up does not apply when the mover is the external agent (`src/bowen/engine/iposition.py`);
  revision 43's `coach-pull-up-removed` is retired from `tools/phase_d_mutation_record.py` and `M11.C.17`'s mutation
  clause restated. Tests first, each shown red: `test_m5d5_t165_an_external_movers_resolve_pulls_no_one_up`,
  `test_m11c21_t161_the_scripted_overfunction_ends_at_t0_in_both_arms`.
- **Regenerated once** on Mac Mini1 at 10 workers (12:57 to 16:35, every step exit 0, the diagnostic record
  included): engine hash `da5981f6ec0b` → `09ca2b46c7e0`. `M11.C.21` fails (−0.72 ± 1.09, p 0.94, the ad hoc value);
  `M11.C.17` fails (`coached_rise` −0.00003 ± 0.00017, p 0.43; the two arms are equal in 136 of 150 seeds). No other
  ensemble verdict changed; Phase C rows that reach Dr Halim's resolved `I-POSITION` moved slightly. Two mutant
  verdicts crossed their thresholds (`steepness-level-independent` now red on `M11.C.41[heavy: lower level]`;
  `weakness-focus-and-steering-removed` no longer red on `M11.C.2[twelve-person family]`, p 0.074 → 0.045). The
  seed-7 trace, chart and famdiag export changed only in their revision stamp; the golden fingerprints hold.
  `docs/version_2_revision_44_changes.md` lists every change.
- **Status.** `docs/version_2_status.md` regenerated: W1 is unmet on `M11.C.17`; `M11.C.21`'s FAIL is the tag's verdict
  again, so version 1's accepted finding applies, and its version 2 note (decided on PASS) is shown as not applied.
  Both go to the owner after the stage-end sweep (T-170), which this commit leaves to run:
  `test_s2_11_sweep_record_is_current` is a strict expected failure until then.
- **Review.** One `learning-qa` pass and one correctness pass on the batch before the regeneration; fixed: the status
  test no longer expects `M11.C.17`'s retired mutant, the T-161 test pins week `t0 − 1` (t0 = 5), the explainer's
  `RESOLVE` line, and the spec no longer says the rise runs through the landed contact "alone". T-169 filed (what a
  coach's completed exchange still does to the member, not traced).
- **Suite.** 1287 tests: 1284 pass, 3 strict expected failures (`M5.B.3`; `M5.D.3`, T-158; the sweep record, T-170).
- **Documents.** The spec (version and status lines, the count note, `M5.D.5`, `M11.C.17`, `M11.C.21`, their `M11.5`
  rows), `docs/spec_revision_log.md`, `docs/model_explainer.md` (§5.4 step 6, `RESOLVE`, the `M11.C.17` and `M6.I.4a`
  passages), `config/bowen/criteria.md`, `config/bowen/constants_changes.md` (no constant changed),
  `docs/version_2_definition_plan.md` (status line, §7), `docs/REVIEWER_READING_LIST.md`,
  `docs/SPEC_REVIEW_CHECKLIST.md`, `README.md`, `CLAUDE.md`, `docs/theory/_STATUS.md`, `tools/record_changes.py`,
  `tools/spec_coverage_overrides.json`, `tests/doc_claims.json`, the records of T-161, T-165, T-169 and T-170, and the
  regenerated records, `docs/version_2_status.md`, `docs/spec_coverage.md`, `docs/spec_index.jsonl`,
  `docs/spec_rules/`, `TODO.md` and this file's index; this file's three earliest entries of 2026-10-25 moved verbatim
  to `docs/CHANGELOG_archive.md` for the size limit, with one suite count's place in `tests/doc_claims.json`.
  `docs/readouts_phase_c.md` needed no change: no readout's definition changed.

### Version 2's definition, step 7: the owner's decisions D1–D6 (2026-10-31)

- **What.** The owner reviewed step 7 of `docs/version_2_definition_plan.md` (§8), each decision put one at a time
  through `AskUserQuestion` with the recommended option first, and took all six as recommended. D1: spec revisions
  41–43 approved as built, D5 replacing revision 43's `coach-pull-up-removed` clause in `M11.C.17`. D2: `M11.C.48`,
  `M11.C.48[phase_d]` and `M11.C.49` accepted as findings. D3 (T-122): `M11.C.34`'s pass reviewed; `M11.C.21`'s not,
  D4 first. D4 (T-161): form (b), the scripted `OVERFUNCTION` ends at `t0` in both arms. D5 (T-165): the coach's
  pull-up is not earned; `M5.D.5`'s pull-up will not apply to an external mover. D6 (T-158): (b), `M5.D.3` and its
  strict expected failure kept as a finding (closed). D4 and D5 move the engine hash: spec revision 44, the next batch.
- **Recorded.** `owner 2026-10-31` in the five rows of `docs/version_2_dispositions.md`; the revision 41–43 tags and
  status lines say "approved by the owner 2026-10-31"; sub-bullets in the records of T-122, T-158, T-161 and T-165.
  `docs/version_2_status.md` regenerated: W1 is met on four of the five rows and unmet on `M11.C.21`, which is decided
  again on the verdict revision 44 gives (T-122 stays open for it).
- **Fixed** (`tools/version_2_status.py`, outside the engine hash): an owner-made `note` no longer counts as the owner's
  review of a changed verdict. Under the old rule ("a disposition the owner made, of any kind") `M11.C.21`'s note,
  which records that the owner did not review its pass, made the page read "reviewed by the owner" and W1 met. Test
  `test_w1_v2_an_owners_note_does_not_review_a_changed_verdict`, shown red first. The page's header says so.
- **Review.** One `learning-qa` pass (narrow, on the diff): the fix sound; four stale texts fixed (the page header,
  `CLAUDE.md`'s and `README.md`'s revision 42 and 43 status, the plan's revision 43 tags line); T-168 filed (the
  stale-text scan cannot see "pending the owner's review").
- **Not changed.** No file in the engine hash (`da5981f6ec0b`); no record but the generated pages; no rule.
- **Suite.** 1285 tests: 1283 pass, 2 strict expected failures (`M5.B.3`; `M5.D.3`, T-158).
- **Documents.** The spec (status line, eight revision tags), `docs/spec_revision_log.md`, `docs/model_explainer.md`,
  `docs/readouts_phase_c.md`, `docs/version_2_definition_plan.md` (status line, §3's tests, §7, §8),
  `docs/version_2_dispositions.md`, `docs/version_2_revision_43_changes.md` and `tools/record_changes.py` (one phrase),
  `docs/REVIEWER_READING_LIST.md`, `docs/SPEC_REVIEW_CHECKLIST.md`, `tests/test_stale_text.py` (`PENDING` emptied),
  `CLAUDE.md` (suite count; revisions 42 and 43 approved), `README.md` (revision 43 approved), regenerated `docs/version_2_status.md`, `docs/spec_coverage.md`, `docs/spec_index.jsonl`,
  `docs/spec_rules/`, `TODO.md`, `docs/TODO_archive.md` and this file's index; this file's earliest entry of 2026-10-25
  moved verbatim to `docs/CHANGELOG_archive.md` for the size limit, with its suite count's place in `tests/doc_claims.json`.

### Version 2's definition, step 7: Claude's recommendations for the owner's review; T-166 (2026-10-30)

- **What.** The brief put step 7 of `docs/version_2_definition_plan.md` to the owner; the owner then said to proceed
  on Claude's recommendations without stopping. So each of the six decisions is **Claude's recommendation of
  2026-10-30, pending the owner's review**, in the plan's new §8 (D1–D6), and none is the owner's: W1 stays unmet on
  `M11.C.21`, `M11.C.34`, `M11.C.48`, `M11.C.48[phase_d]` and `M11.C.49` until the owner writes `owner` and a date.
- **The recommendations.** D1: revisions 41–43 approved as built. D2: the three accepted-finding rows stand. D3
  (T-122): `M11.C.34`'s pass reviewed as final (a `verdict reviewed` row in `docs/version_2_dispositions.md`);
  `M11.C.21`'s not (a `note` row), D4 first. D4 (T-161): form (b), the scripted lending stops at `t0` in both arms;
  measured ad hoc at 100 seeds on this engine, FAIL (−0.72 ± 1.09, p 0.94). D5 (T-165): the coach's pull-up is not
  "earned in behaviour" (`FE06.10`, `FE11.6`); `M5.D.5`'s pull-up should not apply to an external mover. D6 (T-158):
  (b) stands. D4 and D5 move the engine hash and wait for the owner, then go in one batch.
- **T-166** (closed): `tools/group_g_readout.py`'s run columns read matched first cutoffs only and a new "unmatched"
  column counts the rest; test written first and proved by mutation. The "after" section of
  `docs/review/group_g_readout.md` regenerated on `da5981f6ec0b`: 25 of 140 first cutoffs unmatched on the Phase C family and 271 of 748 on
  the Phase D family; on the matched ones the mean run before the cut is 1.504 against an uncut baseline of 1.842,
  and 1.268 against 1.352 (20 seeds per family; every other column unchanged).
- **Not changed.** No file in the engine hash (`da5981f6ec0b`); no record but the group G readout page; no spec text.
- **Documents.** `docs/version_2_definition_plan.md` (status line, §7, §8), `docs/version_2_dispositions.md` (status
  line, two rows), `docs/version_2_status.md` (regenerated), `docs/REVIEWER_READING_LIST.md` (both rows), `CLAUDE.md` (the suite count, 1284), `docs/spec_coverage.md` and `docs/spec_index.jsonl` (regenerated for the new test), the records
  of T-122, T-158, T-161, T-165 and T-166, `TODO.md` and this file (regenerated indexes).

### The stage-end sweep on revision 43's engine (2026-10-30)

- **What.** Step 6 of `docs/version_2_definition_plan.md` (T-167): the stage-end sweep rerun on `da5981f6ec0b`,
  split across both Macs (`docs/two_mac_sweep.md`). The cross-check (`M11.C.16`, `M11.C.23` on both Macs) found 486
  results equal and added 0. Mac Mini1 ran `M11.C.6`, `M11.C.10`, `M11.C.10[family of origin]` and `M11.C.15` at 8
  workers (07:58 to 20:07); this Mac ran the other 30 at 6 workers under `nice -n 15` (to 20:18), from a clean
  worktree. Both exited 0; the merge added Mac Mini1's 47 results and found 486 equal.
- **Results** (`docs/version_2_sweep_record.md`). Only `M11.C.48` and `M11.C.48[phase_d]` moved against the rerun of
  2026-10-29: each holds 0 of 8, and `M11.C.48` is no longer refused at the rerun, since it runs 200 seeds (T-163).
  Every other key's samples are unchanged, as expected: revision 43 changed `M11.C.48`'s readout and added mutants.
  `docs/version_2_status.md`: W2, W3 and W4 met; W1 unmet on `M11.C.21` and `M11.C.34` (verdicts not the tag's, for
  the owner's review, T-122), and `M11.C.48`, `M11.C.48[phase_d]` and `M11.C.49` (accepted findings pending the owner).
- **Tests.** `test_s2_11_sweep_record_is_current` lost its strict expected-failure mark.
- **Suite.** 1283 tests: 1281 pass and 2 are strict expected failures (`M5.B.3`; `M5.D.3`, T-158). Measured 2026-10-30 on this Mac with `python3 -m pytest tests/`; 52 `ensemble`-marked tests deselected.
- **Documents.** `docs/two_mac_sweep.md` ("Not verified"), the plan's §7, `docs/todo/` (T-167 closed; T-163's note),
  the reading list, `README.md`, `docs/theory/_STATUS.md`, `CLAUDE.md`; generated: the sweep record, the status page,
  the coverage page, the indexes and `TODO.md`. No rule, constant or criterion changed, and the engine hash did not
  move, so the spec, the explainer and `config/bowen/constants_changes.md` needed no change.

### The version 2 definition batch: spec revision 43, one regeneration (2026-10-30)

- **What.** Steps 3 to 5 of `docs/version_2_definition_plan.md`, built on the owner's instruction of 2026-10-29 to
  proceed with Claude's recommendations; every form is Claude's recommendation, pending the owner's review.
  `M11.C.48`: a first cutoff with no uncut tie in its week is left out of both arms and counted (`unmatched_cutoffs`,
  reported; T-123), and the criterion runs 200 seeds (T-163). Four Phase D mutants, each found by tracing what a
  surviving named mutant left: `removed-member-holds-not-released-on-return` (`M11.C.11`, T-121),
  `death-all-bound-anxiety-discarded` (`M11.C.15`, T-120), `coach-pull-up-removed` (`M11.C.17`) and
  `mixing-weight-blind-to-level` (`M11.C.34`, T-162). The spec's rows name them.
- **Withdrawn.** T-161 (`M11.C.21`'s scripted lending stopped at the collapse in the treatment arm only) was built and
  withdrawn before the regeneration: the batch's `learning-qa` review found the arms then differ by the lending as well
  as the collapse, and the lending alone carried the pass. A first `M11.C.15` mutant that also turned grief off was a
  no-op on an arm that runs with grief off, and was replaced. T-161 stays open for the owner, with both forms.
- **Regenerated once** on Mac Mini1 at 8 workers, 2026-10-29 23:05 to 2026-10-30 03:02, and the diagnostic record
  after it. The engine hash moved from `6ce8be104767` to `da5981f6ec0b`. `M11.C.48` fails on both families
  (`docs/phase_d2_ensemble_record.md`), so its mutant is not run and T-119 closes; each new mutant turns its criterion
  red (`docs/phase_d_mutation_record.md`), so `docs/version_2_status.md`'s W3 is met. What moved is
  `docs/version_2_revision_43_changes.md`. `M11.C.48`'s failure on each family is a recommended accepted finding in
  `docs/version_2_dispositions.md`. The Phase D record's verdict note on `M11.C.48` (T-123, T-119) is removed, since
  both items closed.
- **Found, not fixed.** T-165: `M11.C.17`'s coached rise runs through the coach's I-position pull-up, which the
  estimator credits as the member's own functioning; a theory question for the owner. T-166: `tools/group_g_readout.py`
  still averages unmatched first cutoffs.
- **Reviews.** One `learning-qa` pass (T-161 withdrawn, the grief mutant replaced, a test case added, T-166 filed) and
  one correctness pass on the revised code (no findings).
- **Tests.** `tests/bowen/test_negative_runs.py` (the unmatched case, proved by a mutant), the C.48 readout and seeds
  in `tests/bowen/test_phase_d2_criteria.py` and `tests/test_version_2_sweep.py`, `tests/test_record_changes.py`'s
  revision 43 page; the two verdict-note tests now check the renderer with a test note.
- **Suite.** 1283 tests: 1280 pass and 3 are strict expected failures (`M5.B.3`; `M5.D.3`, T-158; the sweep record, stale until the stage-end sweep reruns on `da5981f6ec0b`, T-167).
- **Documents.** The spec (revision 43: version and status lines, `M11.C.11`, `.15`, `.17`, `.34`, `.48`),
  `docs/spec_revision_log.md`, `docs/model_explainer.md`, `docs/readouts_phase_c.md`,
  `config/bowen/constants_changes.md` (no constant changed), `config/bowen/criteria.md`, the plan's status line and §7,
  the dispositions, `docs/todo/` (T-119, T-120, T-121, T-123, T-162, T-163 closed; T-161's forms; T-165, T-166, T-167 new),
  the checklist, the reading list, `README.md`, `docs/theory/_STATUS.md`, `CLAUDE.md`; generated: every record, the
  revision 43 changes page, the status page, the coverage page, the indexes and `TODO.md`. The remaining entries of
  2026-10-24 moved verbatim to `docs/CHANGELOG_archive.md`.

### Version 2's definition: the plan, and version 2's dispositions (2026-10-29)

- **What.** `docs/version_2_definition_plan.md`: each row `docs/version_2_status.md` holds under W1, W3 and W4, what
  would meet it, its test, one batch for everything that moves the engine hash (T-123, T-161, T-163), and the order of
  work. The owner asked for the plan and then, the same evening, for the work to go ahead on Claude's
  recommendations; every decision in it is Claude's recommendation, pending the owner's review.
- **Built (step 2).** `tools/version_2_status.py` reads a new `docs/version_2_dispositions.md` beside version 1's
  frozen file: a row names the verdict it was decided on and applies only while it stands; `verdict reviewed` records
  the owner's review of a verdict not the tag's. Its one row, `M11.C.49`'s failure accepted as a finding, is Claude's
  recommendation, so the page now shows *fail, acceptance pending the owner's review* and W1 is still unmet on it.
  The engine hash did not move (`6ce8be104767`); nothing a run computes changed.
- **Tests.** Five in `tests/test_version_2_status.py`; five mutants of the reader, each caught.
- **Suite.** 1281 tests: 1279 pass and 2 are strict expected failures (`M5.B.3`; `M5.D.3`, T-158).
- **Documents.** The plan and the dispositions file (new), `docs/version_2_status.md` and `docs/spec_coverage.md`
  (generated), `docs/SPEC_REVIEW_CHECKLIST.md`, `docs/REVIEWER_READING_LIST.md`, `CLAUDE.md` (the suite count).

### Stage 3's stage-end sweep on revision 41's engine; spec revision 42, wording only (2026-10-29)

- **What.** T-159: the constant sweep (`M17.E.1`) over the 34 declared directional keys of `config/sweep/sweep.md`,
  rerun on the triage batch's engine, `6ce8be104767`, split across both Macs (`docs/two_mac_sweep.md`). The engine
  hash did not move; nothing a run computes changed.
- **The cross-check.** The hash had moved since step 11's check, so `M11.C.16` (Phase C) and `M11.C.23` (Phase D) ran
  on both Macs first, 18 child runs and 2 run lengths each; the merge added 0 and found all 490 entries equal.
- **The split.** Mac Mini1 (8 workers) took `M11.C.6`, `.10`, `.10[family of origin]`, `.15`, `.48` and
  `.48[phase_d]`, from 08:12. This Mac took the other 28 (two done by the cross-check) under `nice` 15 at six workers,
  from a clean `git worktree` of `a4739f2`, because the main checkout holds uncommitted paper-material work (T-136).
  An unclean reboot of this Mac at 09:32 (cause not known) stopped its share. Restarted from the cache at 09:59, only
  the missing samples ran (95 first runs and 74 reruns), and it ended at 17:57, exit 0; the cache, written after
  every sample, kept the samples finished before the reboot.
- **Rebalanced.** From 17:57 a coordinator here took the keys Mac Mini1 had not started: `M11.C.48[phase_d]`
  (17:57–19:23), `M11.C.48` (refused, below) and `M11.C.15` (19:26–19:31). It stopped Mac Mini1 at 20:33, when
  Mini1 reached `M11.C.48`; between two of its checks Mini1 had run `M11.C.15` too. The merge added Mac Mini1's 35
  entries and found 505 equal, Mini1's `M11.C.15` and `M11.C.48` samples 0 and 1 among them. No worker was left on
  either Mac.
- **`M11.C.48` refused at the rerun.** Its sample 2's first run left `negative_run` at +0.255 ± 0.14, within two
  half-widths of zero, so `M17.A.1` needs a rerun at 200 seeds; the criterion fixes its own count at 100
  (`config/bowen/criteria.md`), so the sweep must refuse it (MR-12). The tool stopped the whole run there with
  `SystemExit` and could render no record. The owner chose, through `AskUserQuestion`, to show it refused:
  `tools/version_2_sweep.py` returns a refused result (the sample, both counts, the samples run before it), the
  record shows it *refused at the rerun* with no fraction, and `tools/version_2_status.py` shows that cell and keeps
  W4 unmet on it. Spec revision 42 adds one sentence to `M17.A.1` (wording only; the sentence pending the owner's
  review). T-163 holds the fix: raise its count with the next batch that moves the hash.
- **Results** (`docs/version_2_sweep_record.md`; `git diff a4739f2` on it shows each sample then and now). 267 first
  runs and 165 reruns at 200 seeds; none undetermined or errored. 20 keys held as many samples as at step 11. Moved,
  held of 8 at step 11 and now: `M11.C.21` 0 and 4, `M11.C.10` 3 and 4, `.10[family of origin]` 6 and 7, `.22` 7 and
  8, `.6` 0 and 1, `.7[joint over sided]` 7 and 8, `.2[twelve-person family]` 2 and 3; `M11.C.2` 4 and 2, `.11` 8 and
  7, `.15` 6 and 5, `.44` 3 and 2, `.7[one spouse over sided]` 7 and 6, `.41[lower level: heavier stress]` 1 and 0.
  Each is a measurement, not a verdict. `docs/version_2_status.md`'s W4 is unmet on `M11.C.48` alone (17 rows before,
  while the record was on the earlier engine).
- **Tests.** The refusal test rewritten (no rerun runs; the result carries the refusal, its last row marked); three
  sweep tests added (the record's refused section and cells, the refused row's cell count in both table layouts, a
  share's last line naming its refusals) and the status test extended (the refused cell, W4, the earlier-engine mark);
  `test_w4_s2_9_sweep_column_is_a_fraction_not_a_verdict` now patches the new reader too, so it stays independent of
  the real record; `test_s2_11_sweep_record_is_current` lost its mark. Thirteen mutants of the change were each caught.
- **Suite.** 1276 tests: 1274 pass and 2 are strict expected failures (`M5.B.3`; `M5.D.3`, T-158), measured
  2026-10-29 on this Mac with `python3 -m pytest tests/`; four tests added and one mark removed.
- **Reviews.** One `learning-qa` pass and one correctness pass on the code diff, each run while Mac Mini1 finished.
  No defect in the refusal path; fixed with tests: the refused sample's row showed its first run's PASS as its verdict
  (now a dash, with the first outcome in the Rerun cell, since under `M17.A.1` its verdict is the rerun's), the row
  marker and the ratio layout's cell count were untested, the status page's header did not explain a refused cell, a
  share's last line did not name its refusals, and `tools/record_changes.py` stated a result in hand-written prose
  (now general). T-163's first draft said `M11.C.48[phase_d]` sets no seeds of its own; it reads the same 100.
- **Filed.** T-163 (above); T-164 (the merge's `pgrep` guard matches the sweep's test file, so a test run made the
  merge refuse once; it fails safe). Closed: T-159.
- **Documents.** The spec (revision 42 and its status line), `docs/spec_revision_log.md`, T-001's next step, the spec index, rule views
  and `docs/spec_coverage.md` (generated), `docs/model_explainer.md` §18, `docs/two_mac_sweep.md` (status line, a
  refused criterion in a share, and this run under *Not verified*), the sweep record, `docs/version_2_status.md` and
  `docs/version_2_revision_41_changes.md` (generated), the TODO record, `TODO.md` and `docs/TODO_archive.md`,
  `README.md`, `CLAUDE.md`, `docs/REVIEWER_READING_LIST.md`, `docs/SPEC_REVIEW_CHECKLIST.md`,
  `docs/theory/_STATUS.md`, and `docs/implementation_plan_version_2_stage_3.md`'s status line, which still named
  step 13 as next: the review, its triage and this sweep are step 13, and every step of the plan is done. The run trace,
  its chart and the Family Diagram export (`docs/review/phase_c_*_seed7.*`) are re-rendered: they print the spec
  revision, and nothing else in them changed. `config/bowen/constants_changes.md` is unchanged: no constant changed.
- **Archived.** This entry took `CHANGELOG.md` past its 80,000-byte limit, so the three earliest entries of 2026-10-24
  moved to `docs/CHANGELOG_archive.md` verbatim (7,224 bytes, checked by script against the file at `a4739f2`);
  `CLAUDE.md` says so.

### Stage 3 milestone review triage batch: spec revision 41, one regeneration (2026-10-28)

- **What.** The hash-moving fixes from the stage 3 milestone review's triage (`docs/todo/2026-10-28_stage_3_milestone_review.md`), built as one batch on Mac Mini1, each test written first and shown red. T-131 (b) and T-133 (b) as the owner chose in the triage; T-145 (a) as the brief gave it; T-146 answered by the owner in this session through `AskUserQuestion` (a target who would refuse the act is not a receiver); T-147, T-148 and T-149 needed no decision. The forms are Claude's recommendations under the owner's instruction of 2026-10-28, pending the owner's review.
- **Rules (spec revision 41).** `M1.A.5f`: a target who would refuse the act, read when it is selected, is not a receiver, and what displaced coping moved is kept on each person's `displaced_swing` (the register's new row). `M7.E.1f`: the parents' term is net of `displaced_swing`. `M5.D.4a`, `M5.F.4`, `M5.F.2c`: anger is read again when a sequence's `DEFINE` is said; an angry mover says it all as demand and the sequence ends. `M6.I.4a`: the week's start is the level at the end of the previous week. `M11.C.21`'s readout starts at the end of week `t0 − 1`. The explainer follows each; `config/bowen/constants_changes.md` notes that no constant changed.
- **No rule changed.** T-147: `self_channel_weight` moved into `src/bowen/engine/observe.py`, and a gate test keeps the policy out of the engine's imports. T-149: `config/bowen/event_sizes.md`'s `FIND_WORK_LOAD` note. T-148: `tools/c16_report.py` computes §11's sentences on `M11.C.16`, the report's two `M11.C.16` rows and the spec's `M11.5` row from the mutant table.
- **Engine hash and records.** The hash moved once, from `2a3b84e2d89a` to `6ce8be104767`. Every record but the stage-end sweep was regenerated once on it, on Mac Mini1 at 8 workers (20:59–00:55, 3 h 57 min, every step exit 0); what moved is in `docs/version_2_revision_41_changes.md`. Verdicts that changed: Phase D part I, `M11.C.21`, `M11.C.34` and `M11.C.7[joint over one spouse]` FAIL → PASS, `M11.C.7[one spouse over sided]` PASS → FAIL; the Phase C sweep record, `M11.C.44` at `learning_rate` 0.1 PASS → FAIL; Phase C mutation, `steepness-inverted` on `M11.C.16` red → survived and on `M11.C.38[-10 vs -15]` survived → red; Phase D mutation, `stress-response-removed` on `M11.C.18` red → reversed, and `pseudo-self-return-removed` turns `M11.C.21` red (a new row, as the spec requires). Phase C's ensemble and Phase D part II: no verdict changed. Each is a new result under a new rule, reported beside revision 40's and never in its place (`M11.F`). The golden run-log fingerprints (`tests/bowen/test_engine_golden.py`) moved in both runs, as a change meant to alter behaviour moves them; the batch was measured as one, so the move is not attributed to one change.
- **Deferred.** The stage-end sweep reruns in the next session, split across both Macs (T-159); `test_s2_11_sweep_record_is_current` is a strict expected failure until then. `M5.D.3`'s "abort is the usual first outcome" no longer holds after T-133 (62 abort and 66 hold of 192 seeds, 118 said in anger); its test is a strict expected failure naming T-158, for the owner.
- **Reviews.** One `learning-qa` pass and one correctness pass on `sonnet`, run on the code while the records regenerated: eight defects, all in `tools/c16_report.py` and the tests, none in the engine; each fixed with a test.
- **Filed.** T-160 (the completion report's other rows quote Phase C's close), T-161 (`M11.C.21`'s parents keep over-functioning after they collapse), T-162 (`M11.C.34` passes and no mutant turns it red). Closed: T-131, T-133, T-145, T-146, T-147, T-148, T-149.
- **Documents.** The spec (revision 41, its `M11.5` row generated), `docs/spec_revision_log.md`, `docs/model_explainer.md`, `config/bowen/criteria.md`'s notes (in the engine hash, edited before it moved), `config/bowen/constants_changes.md`, `docs/phase_c_completion_report.md` (§11 and two rows, generated), CLAUDE.md, README.md, `docs/theory/_STATUS.md`, `docs/REVIEWER_READING_LIST.md`, `docs/SPEC_REVIEW_CHECKLIST.md`, the TODO records, and every generated index and record.
- **Archived.** This entry took the file past its 80,000-byte limit, so the last four entries of 2026-10-23 moved to `docs/CHANGELOG_archive.md` verbatim (5,393 bytes); `CLAUDE.md` now says entries from 2026-10-24 on.
- **Suite.** 1272 tests: 1269 pass and 3 are strict expected failures (`M5.B.3`; `M5.D.3`'s first outcome, T-158; the sweep record's currency, T-159); twelve tests added. Measured 2026-10-29 with `.venv/bin/python -m pytest tests/`.

### Stage 3 milestone review triage: spec revision 40, wording only (2026-10-28)

- **What.** The triage of `docs/cycles/2026-10-27_milestone-review_v2-stage-3.md` (plan §7, `milestone_review_brief.md`
  §7). The review branch `review/2026-10-28-v2-stage-3` (`78d5dd6`) was fast-forwarded into `main`. The owner answered
  its six questions through `AskUserQuestion` (Q1–Q5 each as recommended; Q6 as recommended under the owner's
  instruction "run everything using your own recommendations"): T-131 (b), T-133 (b), T-132 recorded as not modelled,
  MR-2 not an exception (reworded and swept), `M11.C.16` stays composite, the reading list's per-step rows cut to
  pointers. The decisions are in `docs/todo/2026-10-28_stage_3_milestone_review.md`.
- **Fixed (spec revision 40, wording only).** MR-2 (`M5.F.4`, `M5.F.2a` and eleven other requirements paraphrased where
  they reused transcript phrases; found by a three-word n-gram comparison against the quoted spans of the KB
  extractions and `_LECTURES_1979.md`), T-132 (`M5.F.4` states `FE04.13`'s two kinds; `M11.C.29`'s row names its plain
  `DISTANCE` baseline), MR-3, MR-4, MR-5, MR-6, MR-7, MR-8, MR-9, MR-10 (the T-130 page only), MR-11, MR-12
  (`tools/todo_index.py`'s **Now** heading), MR-13, MR-14, and the P2 proposal's `KS00.9` citation (T-131). MR-22
  restates an exit condition, not only wording: `M13`'s Phase C exit reports `M11.C.49`'s verdict as a finding instead
  of requiring a pass, as recommended, pending the owner's review. A `sonnet` review of the diff found hedges dropped
  in five paraphrases; they were restored.
- **Test added.** `tests/test_spec_consistency.py::test_mr13_current_engine_hash_is_the_one_the_documents_name`: the
  Phase C mutation record's engine hash must be named in CLAUDE.md, README.md and `_STATUS.md`. The reading list and
  the checklist joined `CITING_DOCS` (MR-13).
- **Filed.** T-148 (MR-1), T-149 (MR-10's config note, in the engine hash), T-150 (MR-2's follow-up: the explainer and
  a test), T-151–T-157 (MR-15–MR-21). T-131, T-133, T-145, T-146, T-147, T-148 and T-149 are **Now**: one hash-moving
  batch with one regeneration, not started here.
- **Not changed.** No file under `src/bowen/` or `config/bowen/`; the engine hash is `2a3b84e2d89a`, as before; no
  record was rerun.
- **Suite.** 1260 tests: 1259 pass and 1 is a strict expected failure (`M5.B.3`); one test added. Measured 2026-10-28
  with `python3 -m pytest tests/`.

### Stage 3 step 12: the reviews of stage 3's whole diff (2026-10-27)

- **What.** Plan step 12. One `learning-qa` pass (P1–P39) and one correctness pass on `sonnet`, in parallel, over
  `git diff 6e8c928 HEAD -- src tools tests config` (78 files); then a narrow `sonnet` re-sweep of the fix diff. The
  engine hash, `2a3b84e2d89a`, did not move; nothing a run computes changed and no record was rerun.
- **Found.** `learning-qa`: 2 (one major). The correctness pass: 5, all minor, and no defect in `src/bowen/`. The
  re-sweep of the fixes: 5 (the heading check narrower than its sibling; three `M6.I.4a` clauses still ungated; three
  nits in the gate wording and T-145's and T-146's records), all fixed.
- **Fixed.** `M6.I.4a`'s lender direction, a borrower who is no lender, its caps at 100 and 0, and a dead lender
  (`M6.3e`) had no case: a new test,
  `tests/bowen/test_moves.py::test_m6i4a_returns_only_to_lenders_within_the_caps_dead_or_alive`. Seven unit mutation
  gates in `tools/mutation_gate_stage_3.json` (those five clauses, and on `test_m6i4a_s3_4` the cap at what was lent
  and the proportional split), all proved (`docs/review/stage_3_unit_mutation_gate.md`, 25 of 25).
  `tools/sweep_config.py`'s `load_ratios` refuses any heading naming a ratio other than `## Named ratios`, which it
  read as no ratio (P2), as the sibling check for the not-run list already did; its test, in `test_s3_19`, fails with
  the check removed. The trace's assertion-form wording (`src/bowen/render/trace.py`, outside the hash) no longer
  names a path revision 39 removed.
- **Deferred, each moving the engine hash** (Claude's recommendation under the owner's overnight instruction,
  2026-10-27, for the owner's review). T-145 (major): `M11.C.21`'s scripted collapse runs in a `before` hook between
  weeks, after one week's start-of-week reading and before the next, so `M6.I.4a` never sees it. An in-process probe (scratchpad only, 100
  seeds each) reproduced the record's row as built (−0.0409 ± 0.44, p 0.889) and, with the week's start read at the
  previous week's end, gave −0.870 ± 0.457, p ≈ 1.0: the return then happens, in week t0, which the readout's
  baseline already includes, so the fix also needs the owner's decision on where the readout starts. T-146: displaced
  coping (`M1.A.5f`) takes functioning from a receiver who refuses the act (`M1.B.3c`); the wording is the owner's
  first. T-147: the engine imports the policy at call time (layering only). Fixing any of them means regenerating
  every record and rerunning the stage-end sweep, so they go to the stage 3 milestone review's triage batch to
  regenerate once, rather than a regeneration now and another after the review.
- **For the owner's review.** The deferral above, where the step 12 row asks that every finding be fixed; T-145's two
  choices (where the week starts, where `M11.C.21`'s readout starts).
- **Documents.** The plan's status line and step 12 row; `_STATUS.md`; the reading list; CLAUDE.md (the spec row,
  the suite line); `docs/SPEC_REVIEW_CHECKLIST.md`'s line for the unit mutation record; T-145, T-146, T-147; the
  generated indexes. No spec text changed, so the revision log did not; no constant changed, so
  `constants_changes.md` did not.
- **Cost.** `learning-qa` 163k tokens, 6.2 min; the correctness pass 253k tokens, 7.1 min; the re-sweep 109k tokens,
  1.8 min; the probe 1 min per arm.
- **Suite.** 1259 tests: 1258 pass and 1 is a strict expected failure (`M5.B.3`); one test added. Measured 2026-10-27
  on this Mac in a fresh worktree with the staged diff applied (`nice -n 15 python3 -m pytest tests/ -q -p
  no:cacheprovider`, exit 0, 5 min 29 s).

### Stage 3 step 11: the stage-end sweep with the named ratio (2026-10-27)

- **What.** Plan step 11 (S3.25). The constant sweep (`M17.E.1`) over all 34 directional keys of
  `config/sweep/sweep.md`, with the named ratio `tie_wear_to_transfer_ratio` (T-128), split across both Macs
  (`docs/two_mac_sweep.md`). The engine hash, `2a3b84e2d89a`, did not move; nothing a run computes changed.
- **The cross-check.** The hash had moved since stage 2's check, so `M11.C.16` (Phase C, the ratio entered) and
  `M11.C.23` (Phase D, not entered) ran on both Macs first, 19 child runs and 2 run lengths each; the merge added 0 and
  found all 452 entries equal.
- **The split.** Mac Mini1 (10 workers) ran `M11.C.6`, `.10`, `.10[family of origin]`, `.15`, `.48` and
  `.48[phase_d]`, 08:01–19:53, 48 first runs and 10 reruns. This Mac ran the other 28 (two already done by the
  cross-check), 08:03–17:05 under `nice` 15 with six workers, 208 first runs and 151 reruns, from a clean `git
  worktree` of the same commit, because another session's uncommitted files failed `check`'s clean-tree test. The
  merge added Mac Mini1's 64 entries (58 results, 6 run lengths) and found 452 equal. In all, 272 first runs and 164
  reruns at 200 seeds; none undetermined or errored; no worker left running on either Mac.
- **Where the ratio entered** (`docs/version_2_sweep_record.md`). 26 of 34 keys, each whose measured run is longer
  than 52 fast ticks; not the 8 at 52 or fewer: `M11.C.42` (40), the four `M11.C.27` cells (21), `M11.C.23` (52),
  `M11.C.34` (34) and `M11.C.43` (47). For the 15 keys whose central verdict passes, held overall, below the central
  ratio 1/52, and at or above it: `M11.C.16` 8 of 8, 6 of 6, 2 of 2; `M11.C.27[unstable,remove_one]` 6 of 8 (not
  entered); `M11.C.41[light: lower level]` 8 of 8, 3 of 3, 5 of 5; `M11.C.41[heavy: lower level]` 8 of 8, 4 of 4, 4 of
  4; `M11.C.7[joint over sided]` 7 of 8, 6 of 6, 1 of 2; `M11.C.7[one spouse over sided]` 7 of 8, 3 of 3, 4 of 5;
  `M11.C.11` 8 of 8, 5 of 5, 3 of 3; `M11.C.17` 8 of 8, 2 of 2, 6 of 6; `M11.C.22` 7 of 8, 4 of 5, 3 of 3;
  `M11.C.39` 8 of 8, 1 of 1, 7 of 7; `M11.C.2[twelve-person family]` 2 of 8, 0 of 4, 2 of 4; `M11.C.10[family of
  origin]` 6 of 8, 4 of 6, 2 of 2; `M11.C.15` 6 of 8, 3 of 3, 3 of 5; `M11.C.48` 8 of 8, 3 of 3, 5 of 5;
  `M11.C.48[phase_d]` 8 of 8, 4 of 4, 4 of 4. Measurements of the declared ranges, not verdicts (`M11.5`).
- **Status page.** W4 met; it was not met on 15 rows while the fractions were the earlier engine's, and no cell is
  now marked "earlier engine; not counted for W4". W1 (8 rows) and W3 (5 rows) stay unmet; W2 met. The page's list
  of records read calls it "Sweep record", not "Stage 2 sweep record".
- **Closed and opened.** T-128 closed. T-144 new: `M11.C.49`, a directional criterion new at revision 39, has no
  declared ranges and was not swept; it fails, so W4 does not need it, but `M17.E.1` asks for every directional
  criterion, so `M17.E.1` stays *partial* in `tools/spec_coverage_overrides.json` until it is swept (the owner,
  2026-10-27, asked when the review found it; the brief had said to mark it done). T-143 notes the sweep's fractions for
  its three criteria.
- **Tests.** `test_s2_11_sweep_record_is_current` loses its strict mark and `STEP_11`, and passes.
  `test_m17e1_an_undeclared_directional_criterion_is_listed_as_not_swept` is new, shown failing with the line left out
  and with every criterion listed (the second mutant survived the test's first version, whose pattern matched only
  directional classes).
- **Review fixes (learning-qa; the correctness pass found no mismatch).** The record showed only what was swept, so
  `M11.C.49`'s absence was visible only on the status page (P2): `tools/version_2_sweep.py` now lists every directional
  criterion of the ensemble records that `config/sweep/sweep.md` declares no ranges for. The record and the config's
  `sweep_samples` row said 8 samples time the sweep and measure no criterion, a sentence left from the pilot; both now
  say 8 is the pilot's count kept for the full sweeps, too few for a band or a *narrow* mark (D14). "All 34
  directional keys" read as all of them; now "declared". The revision 39 page's reason names the engine move as well
  as the ratio; a test docstring's "waits for step 11" removed.
- **Documents for the owner's full review** (the owner, during the run, 2026-10-27). README's Status section now says
  what each phase built and stopped restating dated verdicts (it listed `M11.C.2[twelve-person family]` as failing);
  `docs/model_explainer.md`'s `M11.C.17` passage (its mutant now survives, T-143) and a note that the verdicts it
  quotes are dated; the reading list's revision range, T-129's state, its snapshots row and its date;
  `docs/version_2_plan.md`'s MR-23 sentence.
- **For the owner's review.** `M11.C.49` not swept (T-144), where the plan's §3.6 counted it; the record's new line
  of undeclared criteria and its reworded 8-sample sentence; this Mac's share run
  from a worktree; the revision 39 changes page still does not compare the sweep record, its reason restated (the
  ratio changed the samples where it entered); the wall time stated before the run as about 11 h on each Mac, measured
  at 11 h 52 min on Mac Mini1 and 9 h 3 min here.
- **Documents.** The plan's status line, step 11 row and §3.6 time; `_STATUS.md`; the reading list; README;
  CLAUDE.md (the spec row, the suite line); the explainer's §18.7; `docs/two_mac_sweep.md`; `config/sweep/sweep.md`'s
  `sweep_samples` row; the revision 39 changes page; T-128, T-143, T-144; the sweep record, its cache, the status page
  and the generated indexes. No spec text
  changed, so the revision log did not; no constant changed, so `constants_changes.md` did not.
- **Suite.** 1258 tests: 1257 pass and 1 is a strict expected failure (`M5.B.3`); `test_s2_11` passes. Measured
  2026-10-27 on this Mac (`nice -n 15 python3 -m pytest tests/ -q -p no:cacheprovider`, exit 0, 5 min 46 s).

### Stage 3 step 10: the sweep's named ratio, one regeneration, the open batch closed (2026-10-26)

- **What.** Plan step 10, the batch's last commit. T-128 (`M17.E.1`'s named ratio) in the sweep tool, MR-23 (T-012),
  and every record regenerated once on the final engine hash, `2a3b84e2d89a`, which did not move: the sweep tool and
  `config/sweep/` are outside it. Spec revision 39 is now applied in full. The open batch (D31) is closed.
- **T-128 (D24–D26, Q14).** `config/sweep/sweep.md` gains `## Named ratios` with `tie_wear_to_transfer_ratio` =
  `tie_wear_rate` ÷ (`slow_tick_fast_ticks` × `calm_transfer_rate`), central 1/52, 1/208 to 4/52 on a log scale, `[I]`.
  `tools/sweep_config.py` loads it and refuses a ratio whose period constant has a `## Ranges` row (the ratio is
  derived at the period's central value). `tools/version_2_sweep.py` draws the ratio in place of `tie_wear_rate` for a
  criterion whose run is longer than 52 fast ticks, writes the ratio on each sample, and splits each such criterion's
  samples at the central value. A criterion's run length is measured, not listed: seed 0 of each arm at the central
  constants, the longest arm's fast ticks, cached under the engine hash and the measuring function's own source hash.
  `M17.E.1`'s sentence names the ratio. S3.16–S3.19 lose their strict marks and pass; S3.16 compares the row's ratio
  with one redrawn from the seed, after a mutant that ignored the numerator survived the first version.
- **MR-23 (T-012, closed).** `tools/ensemble_record.py` renders the Phase C ensemble record from
  `docs/records_cache/phase_c_ensemble.json`, as the Phase D records are rendered; its own file is out of its hash
  (the record prints `code_hash()` alone, `042166ee2147`, as both Phase D ensemble records do), and
  `tests/bowen/test_ensemble_record.py` compares the full render.
- **The regeneration.** Here: the Phase C ensemble record (4 min 31 s), mutation record (32 min 48 s), sweep record
  (8 min 11 s), diagnostic record (8 min 29 s), level occupancy (35 s), the group G readout's "after" section, Phase D
  part I (16 min 15 s) and the Phase D mutation record (93 min 23 s, six workers; step 7's four `M11.C.6` mutants run
  the 884-tick arms). Mac Mini1: Phase D part II (95 min 6 s), copied back, its `code_hash` checked. Then the seed-7
  trace and fam-diag export, `tools/c16_report.py`, and the generated indexes.
- **What moved** (`docs/version_2_revision_39_changes.md`, new, against `9ebf327`, the records revision 36's
  regeneration left and the batch kept). The status page: 6 of 63 rows differ from revision 38's.
  `M11.C.2[twelve-person family]` (symptom load +2.61 ± 1.6), `M11.C.7[one spouse over sided]`,
  `M11.C.10[family of origin]` (cut-off tie weeks +220 ± 146) and `M11.C.17` (coached rise +0.039 ± 0.030) go from
  FAIL to PASS; `M11.C.44` goes from PASS to FAIL (−0.020 ± 0.089, T-142); `M11.C.49` is new and fails (family excess
  +3.1 ± 2.9, `CONFLICT` share +0.003 ± 0.027, T-137). `M11.C.42` and `M11.C.45` still fail (T-142). The Phase C
  mutation record: 5 of 114 verdicts changed, and the four `M11.C.44` mutants no longer run. The Phase D mutation
  record: 1 of 11 changed, 8 rows new; `M11.C.17`'s one mutant survives, so its new pass is not yet proved by
  mutation. The sweep record: 3 of 54 settings changed, all on `M11.C.44`. T-143 lists the mutants that no longer turn
  red. Each is a new result under a new rule, reported beside revision 38's and version 1's verdicts (`M11.F`); no
  constant was tuned to a row (`M10.B.4`).
- **Pages.** `tools/record_changes.py` gains the revision 39 page; the revision 36 page is frozen at `9ebf327` (its
  later side was the working tree). The status page (S3.24) gains a column with revision 38's verdict and a summary
  line of the rows that differ; a sweep record built on an earlier engine (the stage 2 sweep record, until step 11)
  has each fraction marked "earlier engine; not counted for W4", and W4 does not count it.
- **The open batch closed.** `tests/open_batch.py`'s `OPEN` is None and the twelve marks are off; each of the eleven
  record-current tests and the `M13.4` record test passes. `test_s2_11_sweep_record_is_current` stays a strict
  expected failure until step 11 re-sweeps with the ratio, and now fails as expected only on a missing cache entry for
  the current engine (any other `KeyError` fails it). `test_m17e1_a_known_defect…` runs on a built record and passes.
- **Review fixes (learning-qa).** The stale sweep record had been read as current by the status page (now marked, not
  counted); the MR-3 guard had been disabled by an `xfail`; the run-length cache key lacked the measuring method; the
  Phase D criteria's `lru_cache`s were not cleared between run-length measurements; a swept period constant would have
  been silently held central. Each fixed with a test.
- **For the owner's review.** `sweep.md`'s preamble says the run length is measured where the draft's parenthesis
  named the criteria left out; strict expected failures go to 2, not 1 (`test_s2_11` waits for step 11's re-sweep);
  the revision 36 page frozen at `9ebf327`; `SweepConfig` a tuple subclass carrying the ratios; the status page's
  revision 38 column and the earlier-engine W4 rule; `docs/spec_revision_39_draft.md`'s status line corrected (it had
  said "not yet applied" since step 6); T-128 left open until step 11; Phase D part I and the Phase D mutation record
  run here rather than on Mac Mini1; the wall time stated after the run had started.
- **Documents.** The spec (line 4's status, `M17.E.1`), `config/sweep/sweep.md`, the explainer's §18.7, the revision
  log, the revision 39 draft's status, the plan's status line and step 10 row, `_STATUS.md`, the reading list, README,
  CLAUDE.md, `docs/two_mac_sweep.md`, `docs/SPEC_REVIEW_CHECKLIST.md`, the coverage overrides, T-012 (closed), T-128,
  T-143 (new); every generated record and index. No constant changed, so `constants_changes.md` did not.
- **Suite.** 1257 tests: 1255 pass and 2 are strict expected failures (`M5.B.3`; `test_s2_11` until step 11);
  the open batch's count is 0. Measured 2026-10-26 on this Mac.

### Stage 3 step 9: group A, contact returned by a relieved receiver and TRIANGLE with distance (2026-10-26)

- **What.** Plan step 9, the batch's fourth commit. Spec revision 39's group A lands with its code; the engine hash
  moved `b819660f58b9` → `2a3b84e2d89a`, and what a run computes changed.
- **P1 (`M4.C.1d`, D22).** When a delivery of an act with a positive contact component raises its target's felt
  contact and lowers the target's two-sided deviation (`M4.C.1`, impingement included, on the pre-batch state), the
  sender takes `reciprocal_contact_share × Δcontact × content_gain(sender)` as felt contact on the same tie and
  appraises it by their own deviation (`appraise.returned_contact`). It is not counted in `attended` (`M1.B.8`). The
  rule names no kind. `reciprocal_contact_share` is 0.5, `[I]` (`constants.md`, a `constants_changes.md` row).
- **P3 (`M4.D.3a`).** `TRIANGLE` is layer 0 in `config/bowen/event_kinds.md`, so its availability does not fall with
  functional level.
- **Tests.** S3.10 and S3.11 lose their strict marks and pass. Two more unit tests: the sender appraises the
  returned contact by their own deviation, and relief reads the whole delivery (Q11). The unit mutation gate adds
  S3.10's and S3.11's proofs (`review/stage_3_unit_mutation_gate.md`, 18 of 18 proved). The golden fingerprints were
  taken again with the module's own `_records` and `_fingerprint`; each of P1 and P3 alone moves both, and with both
  off step 8's values return.
- **Mutants.** Phase C `reciprocal-contact-removed` (`M11.C.42`); `one-sided-deviation` now also names `M11.C.45`.
  `tools/stage_3_targeted_run.py` matched a mutant's criteria exactly, so a mutant naming `M11.C.27[` or `M11.C.41` did
  not run on a passing cell of it; it now uses the Phase C record's rule (a test proves it), and the run was redone,
  its criterion rows identical to the first.
- **The suite run's fixes.** `tests/open_batch.py` cut a mutant's criteria by every quoted key in the file the batch
  found, so `one-sided-deviation`'s new `M11.C.45` (quoted there by other mutants) stayed and the Phase C mutation
  record's as-found render gained a line; it now reads the mutant's own definition (a test proves it).
  `test_m4d1h_decide_sets_intensity_by_force` needs a sequence-decided step: since group A seed 0 starts none in 30
  weeks, so it runs seed 3. `test_m16a7`'s after-freeze set adds `reciprocal_contact_share`. The `M11.5` notes for
  `M11.C.42` and `M11.C.45` drop the draft's opening "Revision 39:" (the history-in-body rule).
- **The targeted run** (Mac Mini1 at `2a3b84e2d89a`, `review/stage_3_step_9_criteria.md`; no record moved).
  `M11.C.42` FAIL: `triangle_reuse` −0.06 ± 0.36 (50 seeds). `M11.C.45` FAIL: `triangle_rate` +0.0013 ± 0.0055.
  `M11.C.44` FAIL: `outside_inside_ratio` −0.020 ± 0.089, where revision 38's record passes (T-142). `M11.C.27`: three
  cells fail and `[unstable,remove_one]` passes (`pair_deviation` −0.081 ± 0.05), and both its mutants turn it red, as in the record. `M11.C.16` PASS
  (`repertoire_entropy` −0.0385 ± 0.0062); four of its 24 named mutants turn it red, as in the record, but
  `steepness-inverted` is now red and `availability-level-inverted` now survives. `M11.C.41`: the two lower-level
  cells pass and the two heavier-stress cells fail, as in the record; on each passing cell four of nine named mutants
  turn it red, and `steepness-level-independent`, red on `[heavy: lower level]` in the record, now survives. S3.12 and S3.13 are not met.
- **For the owner's review.** `M4.C.1d`'s clause on a sender near or past the optimum reads "within its band or past
  it nothing", since the deviation's "too much" side is impingement; the register rows (`acute_anxiety`,
  `felt_contact`, base appraisal) and the explainer's sentence on it are additions to the draft. `M5.C.1`'s gate does
  not cover returned contact, kept as built by the owner's choice (T-140). T-141 (low) and T-142 recorded.
- **Review.** One `learning-qa` pass and one correctness pass (sonnet). Fixed: Q11's whole-delivery reading untested
  (test and gate added), the clause widened to the band, a vacuous assertion replaced by a fixture check, the
  constants_changes sweep clause restored, the status documents.
- **Documents.** The spec (`M4.C.1d`, `M4.D.3a`, `M1.C.1`, `M10.C.1a`, `M11.C.42`, the `M11.5` notes, the register,
  §0.3's count, the status line), the explainer (§5.2, §19.2, §19.4), `event_kinds.md`, `constants.md`,
  `constants_changes.md`, the revision log, the plan (status line, step 9 row), `_STATUS.md`, the reading list,
  `README.md`, `CLAUDE.md`, the proposal's requirement count, the checklist, T-140–T-142; the generated files
  regenerated.
- Suite: 1251 tests: 1234 pass, 5 strict expected failures, 12 open-batch expected failures (`python3 -m pytest tests/`, 2026-10-26).

### Stage 3 step 8: group F, the I-position as a mixture of self-statement and demand (2026-10-26)

- **What.** Plan step 8, the batch's third commit. Spec revision 39's group F lands with its code; the engine hash
  moved `b8e9ad1333e2` → `b819660f58b9`, and what a run computes changed.
- **A (`M5.F.4`, D20).** A genuine `I-POSITION`'s `DEFINE` event carries `demand_share` = `1 − w`, `w` the mover's own
  self-channel weight that week (`iposition.demand_share`); the angry assertion form carries 1. The share adds
  `(1 − w) × assertion_gain` to the impingement delivered (`appraise.py`), scales the target's debit by `w` and
  raises the mover's outward axis by `(1 − w) × assertion_evidence_gain` (`moves.iposition_effect`). The perspective
  threshold is retired (the owner's Q8): `assertion_perspective_threshold` leaves `config/bowen/constants.md`, with a
  `constants_changes.md` row, and is listed in `RETIRED` so the frozen constants of 2026-10-07 still load. A mover
  without perspective now starts a sequence like any other; the Phase B script's Marta begins hers at week 31 and
  says it at 35 (`config/bowen/script_phase_b.md`). `M11.C.32` counts the assertion form as share ≥ 1.
- **`M11.C.29`'s arm (D21).** F at `mover_level` in both arms, F's spell suspended over the act's window
  (`act_window`, `suspended`), M's kept; `completions` reported, and a row with no completion fails.
- **`M11.C.49`, new.** Four runs per seed on the triad, F at 35 and at 60, each with the act and with an unscripted
  control; the act starts at `DEFINE`, unprepared, by a `before` hook in `src/bowen/ensemble/criteria.py` (Q9);
  perspective 1.0 (Q13). Its `M11.5` row is Mixed; the Phase C exit condition names it.
- **Tests.** S3.6 and S3.8 lose their strict marks and pass; S3.7 is `ensemble`-marked. The unit mutation gate adds
  S3.6's and S3.8's proofs (`review/stage_3_unit_mutation_gate.md`, 11 of 11 proved). New: the open batch's rerun
  leaves out a criterion added since `9ebf327`, as step 7's pin does for mutants, so
  `test_m134_record_covers_every_criterion_and_names_what_is_not_built` is marked stale in the batch; the targeted-run
  tool sends each criterion to its phase's child; the trace renders an `I-POSITION` that starts its sequence.
- **Mutants.** Phase C (`tools/mutation_record.py`): `demand-share-removed`, `demand-share-inverted` (`M11.C.49`),
  `budget-reduction-removed` (`M11.C.29`), `iposition-debit-removed` (`M11.C.5`). Phase D
  (`tools/phase_d_mutation_record.py`): `stress-response-removed` (`M11.C.18`), `borrowed-pseudo-self-read-as-own`
  (`M11.C.20`), `dependence-gate-removed` (`M11.C.33`). The spec names these; no tool had them before.
- **The targeted run** (`tools/stage_3_targeted_run.py`, now taking Phase C and Phase D criteria, on Mac Mini1 at
  `b819660f58b9`, `review/stage_3_step_8_criteria.md`; no record moved). `M11.C.49` FAIL: `family_excess` +2.17 ± 3.0,
  `conflict_share` +0.003 ± 0.028; reported, `target_impingement` +0.085 ± 0.023 and `completions` −0.31 ± 0.074
  (T-137). `M11.C.29` FAIL: `third_symptom_weeks` +0.9 ± 3.7, and the genuine arm completes in 1 of 100 seeds (seed 82; the control, F
  withdrawing, in 3), T-138. `M11.C.32` PASS (+1.44 ± 0.2, −0.23 ±
  0.092); `M11.C.5` FAIL; `M11.C.18`, `.20`, `.33` PASS. Each passing criterion's mutants turn it red
  (`borrowed-pseudo-self-read-as-own` reverses `M11.C.20`); the mutants of the three failing criteria did not run.
- **Golden fingerprints** moved, both runs, and were taken again with `tests/bowen/test_engine_golden.py`'s own
  `_records` and `_fingerprint`.
- **Deviations, for the owner's review** (the plan's step 8 row): F's own `I-POSITION` withheld in every week of all
  four `M11.C.49` runs; a run with no move on the F–M tie reads a `CONFLICT` share of 0 and is counted; the open
  batch's rerun cuts criteria as well as mutants; `M11.C.29`'s completions read with a pseudo-move in
  `requires_moves`; `demand_share` imports the policy's `self_channel_weight` locally, to avoid an import cycle.
- **`M5.B.3`'s A6 test** (the owner, 2026-10-26: assert beyond noise). Group F made it pass by a margin inside its
  noise, so it now builds every arm of `range(30)` and asserts the paired mean less two standard errors is above 0;
  it stays a strict expected failure, its reason updated (62.78 against 62.78, +0.002 ± 0.94).
- **The open batch.** `tests/open_batch.py`'s `registry_as_found` reads the criteria the batch found, so the version 2
  status tests do not stop on `M11.C.49`, registered with no record row until step 10; a table is cut only through
  the module that defines it.
- **Review.** One `learning-qa` pass and one correctness pass (sonnet); every finding fixed: `M5.F.4`'s tail restored
  (an edit had cut it), the register's rows name the demand share, `arm_c49` raises unless only the act's arm says it,
  `demand_share` clamped at 0, the `before` hook keeps the begin record, `test_m13` checks a begun sequence delivers,
  the targeted run's page names a missing move. `act_window` reading `load_constants` and `mover_at`'s fixed
  perspective are recorded as T-139.
- **CHANGELOG.** The three oldest entries (revision 35's dispositions and spec revision 36, 2026-10-22; revision 36's
  records regenerated, 2026-10-23) moved to `docs/CHANGELOG_archive.md`, verbatim.
- Suite: 1247 tests: 1228 pass, 7 strict expected failures, 12 open-batch expected failures (`python3 -m pytest tests/`, 2026-10-26).

### Stage 3 step 7: group C, displaced coping, the floor-held ledger and lent functioning returned (2026-10-26)

- **What.** Plan step 7, the batch's second commit. Spec revision 39's group C lands with its code; the engine hash
  moved `f90285984f32` → `b8e9ad1333e2`, and what a run computes changed.
- **P2, displaced coping (`M1.A.5f`).** A loaded sender's act of a *displacing* kind moves the coping amount from its
  living family receivers' functional level and stress swing to the sender's, split evenly, each share bounded by the
  receiver's room above the stress floor and the sender's room below 100; a receiver holding a position toward the
  sender gives nothing (`stress_response.displace`, run before each person's own response). The kinds are a new
  `## Displacing` table in `config/bowen/stress_response.md`, `OVERFUNCTION` alone; loading fails on an unknown kind, a
  repeated row, or a kind both soothing and displacing (`load.load_displacing`, `EventKinds.displacing`).
- **F1, the floor-held ledger (`M1.A.5e`, `M1.A.4k`).** `Person.floor_held` takes what a bound stopped of a calm
  week's decay and is shed by `response_relaxation_rate` as far as the bound lets; the estimator leaves it out
  (`slow_tick.record_week`).
- **`M6.I.4a`.** A taker whose capacity fell by a cause other than the exchange returns the same share of their
  borrowed pseudo-self, with functional level, to their lenders (`moves.return_lent_functioning`, before
  `relax_pseudo_self`; the week's exchange read from its `functioning_shift` records by `moves.exchanged`).
- **The stress floor** is one function, `moves.stress_floor`, used by the stress response, displaced coping and both
  sides of `M6.I.4` (the `learning-qa` pass found it written three times).
- **Tests.** S3.1a–S3.1d and S3.4 lose their strict expected-failure marks and pass. New: the week's exchange left
  out of `M6.I.4a`'s fall, `exchanged` reading only the exchange, `floor_held` taking nothing in an unbound week,
  S3.1b's case of a sender below their usual level, the open batch's mutant pin, the targeted-run tool's render.
  `review/stage_3_unit_mutation_gate.md` (`tools/mutation_gate.py tools/mutation_gate_stage_3.json`): 9 of 9 gates
  proved, S3.1b split into one gate per guard.
- **Mutants.** Added to `tools/phase_d_mutation_record.py`: `displaced-coping-removed`, `displaced-coping-unguarded`,
  `floor-held-credited` (`M11.C.6`) and `pseudo-self-return-removed` (`M11.C.21`); `M11.C.2` added to
  `witness-path-removed`'s and `weakness-focus-and-steering-removed`'s criteria. The committed record is not
  regenerated (D31): `tests/open_batch.py` now pins each mutation tool's mutants, and their criteria, to those found at
  `9ebf327` during its rerun, so the committed render still reads as found.
- **The targeted run** (new `tools/stage_3_targeted_run.py`, on Mac Mini1 at `b8e9ad1333e2`; pages under
  `docs/review/`, no record moved). `M11.C.6` FAIL, `development` +0.041 ± 0.027 (revision 38: +0.089 ± 0.044):
  the child still develops more with projection than without; reported as read (Q4), not tuned. `M11.C.2[twelve-person
  family]` FAIL, +0.33 ± 1.8, p 0.39 (+0.95 ± 1.6). `M11.C.2` (Phase B family) PASS, +3.37 ± 3.3, p 0.046 (FAIL,
  +0.35 ± 3.6); its two mutants (`review/stage_3_step_7_c2_mutants.md`): `witness-path-removed` turns it red (+0.90 ± 1.9, p 0.2), `weakness-focus-and-steering-removed` does not (+3.31 ± 3.5, p 0.04), so here the pass is carried by witnessing, the reverse of what the spec's row records for the twelve-person family at revision 31; `M11.C.21` FAIL, `target_rise` +0.056 ± 0.11, p 0.22 (−0.068 ± 0.089). A failing criterion's
  mutants did not run (`M11.1d`). The first launch on Mini1 failed at import (plain `python3` there has no NumPy); the
  second was stopped when the review fixes moved the hash; the third is the one reported.
- **Golden fingerprints** moved, both runs, and were taken again with `tests/bowen/test_engine_golden.py`'s own
  `_records` and `_fingerprint`; the module docstring and the table say why (P23).
- **Review.** One `learning-qa` pass (about 120k tokens, 7 minutes): no conservation or bounds defect; five findings,
  four fixed (the generated indexes not rebuilt; S3.1b's calm case could not fail, now a below-usual sender and one
  gate per guard; the week's exchange left out of the fall was untested; the counts) and one recorded as T-134 (a
  return with no lender is not counted, which `M6.I.4a` allows; a log change for the owner). One correctness pass
  (`sonnet`, read-only, about 146k tokens, 7 minutes): the code matches the spec text; two rounding residues fixed
  (`floor_held` took about 1e-15 in weeks no bound stopped; a return of about 4e-15 when the exchange's addition and
  removal did not cancel, now ignored within `invariant_tolerance`), and the spec now says a receiver below the floor
  gives nothing and states the tolerance.
- **Deviations, for the owner's review** (the plan's step 7 row): a holder stays in the share's denominator, as
  `M1.A.5f` reads; receivers are family members, not the coach; the open-batch pin extended to mutants.
- **The suite run's fixes.** `M11.C.10`'s family-of-origin arm held 401 of its 416 tie-weeks on seed 0, not all:
  a renewal every 13 weeks outlasts a cutoff's rejection (about 10 weeks), and group C changed the refusals that
  held it between. `M11.C.10` counts a lapse by design, so the test now asserts held weeks above 0 and at most 416;
  whether `renew_every` should change is T-135, for the owner. The open batch's rerun cut `MUTANTS` in a tool that
  imports it (the diagnostic record's) by that tool's own file, emptying it; it now cuts by the defining file and
  applies that to every holder (`tests/test_open_batch.py`, proved failing on the old cut). The spec's reference
  count is 1027 (`M11.D.9`), §0.3 states 1063 MUSTs, the three new `docs/review/` pages are in `NOT_COMPARED`
  (the stage 1 and revision 36 change pages regenerated, one line each) and the targeted-run page is listed as
  having no status line (`tests/doc_claims.json`).
- Suite: 1242 tests: 1222 pass, 9 strict expected failures, 11 open-batch expected failures (`python3 -m pytest tests/`, 2026-10-26).
- Documents updated: the spec (status line, `M1.A.4k`, `M1.A.5e`, `M1.A.5f`, `M6.I.4a`, `M7.E.1f`, `M11.C.6`,
  `M11.C.21`, `M11.5`, the register), `config/bowen/stress_response.md`, `docs/model_explainer.md`,
  `docs/spec_revision_log.md`, the stage 3 plan (status line and step 7 row), `docs/theory/_STATUS.md`,
  `docs/REVIEWER_READING_LIST.md`, the checklist, README, `CLAUDE.md`, `docs/agent_model_proposal.html` (the requirement
  count, 583), the stage 3 TODO record (T-134); generated: the spec index, the rules view, `docs/spec_coverage.md`,
  the seed-7 renders, `TODO.md`. `config/bowen/constants_changes.md` is unchanged: no constant changed. The two oldest
  entries here (the closed and old parts of `TODO.md` and `CHANGELOG.md` archived; `TODO.md` an index over records)
  moved to `docs/CHANGELOG_archive.md` verbatim.

### Stage 3 step 6: the batch's first commit, T-130's wording, T-014 and T-129 (2026-10-25)

- **What.** Plan §3.6's edits in one commit, the first of stage 3's batch to move the engine hash
  (`66dcbb1e8ba5` → `f90285984f32`). No result changed: the golden fingerprint tests pass unchanged (S3.23), and the
  seed-7 trace, chart and fam-diag export differ only in the revision label. Spec revision 39 is applied in part.
- **T-130.** `config/bowen/event_sizes.md` no longer calls its values unchecked (they agree with Holmes and Rahe's
  Table 3, p. 216), and four item names follow the table's wording; `M1.F.6a` and the explainer's §11.5 say the same.
- **T-014** (revision 39 part 3, the draft's §21). The `M8` preamble, `M8.1`, `M8.5`, `M13.1`, `M11.C.7` and its
  `M11.5` row, and `M7.D.1a` carry the approved text. `live_positions.avoidance_available` and its test are removed
  (D27); `positions_live` is the one definition, with three callers (S3.20 green); fusion stays unbuilt (D28, `M8.1`
  partial). The explainer's §11.4 and §13.1, `_RESOLUTIONS.md` and version 2's plan follow.
- **T-129.** The fourteen `ruff check --select F` findings are fixed (unused imports and a dead assignment; the four
  names `policy.py` imports from `observe.py` are marked as re-exports), and `SOURCE_ALLOWED` in
  `tests/test_named_tests_resolve.py` is empty (S3.22 green). Docstrings in `moves.py`, `rejection.py`, `objects.py`
  and `invariants.py` are restated from the code (MR-2, MR-9, MR-13). T-127's status lines are updated.
- **The open batch** (Claude's recommendation, for the owner's review). D31 leaves the records keyed by the hash
  stale until step 10; CLAUDE.md says a red suite is never normal. New `tests/open_batch.py` marks the eleven tests
  that read those nine records: a failure is an expected failure only while the batch is open, the record is
  byte-identical to the file at `9ebf327`, and the test passes when rerun with the hash computed on the tree at
  `9ebf327` (`git archive` into pytest's temporary directory), so a failure the hash does not explain still fails and
  the renderer checks in two of the eleven still run. New `tests/test_open_batch.py` fails when some listed records are
  regenerated and others not, when a mark outlives the batch, and when the pinned hash is not the one a record prints.
  The gate test counts the marks as a third figure, each on a module-level test with no argument, collected once.
  `pytest.ini` adds `-rx`, so each expected failure's reason is printed.
- **Review.** One `learning-qa` pass (about 85k tokens, 4 minutes) found five issues, all fixed: the mark excused any
  failure while the record was unchanged (now only one the opening hash explains); it switched off the two renderer
  checks (now run under the opening hash); a regeneration in part stayed green (now all or none); expected failures
  were not listed in the summary (`-rx`); the gate counted decorator lines without checking what they decorate. One
  correctness pass (`sonnet`, read-only, about 115k tokens, 2 minutes) found the spec edits match the draft, and one
  wording point in `invariants.py`, fixed. A narrow re-check of the fix (`sonnet`, about 103k tokens, 2 minutes)
  found the gate's scan read the mark's own nested tests as loose marks (fixed), and three gaps, closed: the pinned
  hash with a tool is checked against a record's stamp, the patch is checked restored after a failed rerun, and the
  batch's commit is named in full.
- **Deferred.** MR-23 (T-012), a records-tool change, moves to step 10, before the one regeneration.
- Suite: 1236 tests: 1211 pass, 14 are strict expected failures and 11 are expected failures of the open batch.
  `EXPECTED_SPEC_REFERENCES` is 1016 (`M8.5`'s reference to `M5.C` went with the old text).
- Documents updated: the spec, `docs/spec_revision_log.md`, `config/bowen/event_sizes.md` and `contact_classes.md`,
  `docs/model_explainer.md`, `docs/theory/_RESOLUTIONS.md`, `docs/version_2_plan.md`, the stage 3 plan (status line
  and step 6 row), `docs/theory/_STATUS.md`, `docs/REVIEWER_READING_LIST.md`, the checklist, README, `CLAUDE.md`, the
  T-014, T-127, T-129 and T-130 records, `tools/spec_coverage_overrides.json`; generated: the spec index, the rules
  view, `docs/spec_coverage.md`, the seed-7 renders, `TODO.md`. The oldest entry here (the model ergonomics
  follow-ups) moved to `docs/CHANGELOG_archive.md` verbatim.

### Stage 3 step 5: the tests first, red (2026-10-25)

- Sixteen stage 3 tests for plan steps 6–10 are written before any engine change, each with its spec ID or S3 ID in
  its name: S3.1a–S3.1d (displaced coping, its config table, `M1.A.5e`'s `floor_held` ledger), S3.4 (lent
  functioning returned with lost capacity), S3.6 (`M5.F.4`'s demand share), S3.7 (the new I-position criterion), S3.8
  (`M11.C.29`'s arm), S3.10 (reciprocal contact), S3.11 (`TRIANGLE` in layer 0), S3.16–S3.19 (`M17.E.1`'s named ratio in the sweep tool), S3.20 (`positions_live`'s three
  callers) and S3.22 (`ruff check --select F` clean, new file `tests/test_lint.py`). Each was run with `--runxfail`
  and fails on an `AssertionError` that names the missing rule, none on an import error or a typo.
- Marking (Claude's recommendation, overnight, for the owner's review): the fifteen default-run tests are
  `xfail(strict=True, raises=AssertionError)` with a reason naming the step that turns each green, so a test that
  starts passing, or fails for any other reason, turns the suite red; S3.7 is `ensemble`-marked, out of the default
  run like every criterion test. The plan's step 5 row lists the names the tests fix for the build (among them
  `EventKinds.displacing`, `Person.floor_held`, `moves.return_lent_functioning`, `sweep_config.load_ratios`) and three
  readings: displaced coping's room above the floor is read as `functional_level − floor`, since the draft's parenthesis gives
  the floor itself (its wording is fixed at step 7); S3.8's window includes `t0 + prepare_ticks`; S3.19 checks
  `M17.E.1`, since `M10.C.1a` is unchanged (the draft's §20.2). §3.7's rows for S3.6, S3.7 and S3.19 are corrected to
  the draft (the mutant `demand-share-removed`; `M17.E.1`).
- The suite is 1232 tests: 1216 pass and 16 are strict expected failures (fifteen new expected failures); the
  `ensemble`-marked criterion tests rise from 51 to 52. Nothing under `src/bowen/` or `config/bowen/` changed, the spec
  did not change, and the engine hash stays `66dcbb1e8ba5`. `ruff check --select F` still reports T-129's 14 findings,
  none in the new tests.
- One correctness pass (`sonnet`, read-only, about 183k tokens, 6 minutes) found no test that would stay red once its
  rule is built as the draft says, and six weaknesses, five fixed: S3.10 gains an anxious sender, who takes in none of
  the returned contact; S3.6 runs at conductance 0.5, so the event's scale is not 1; S3.16's tolerance allows for the
  tool's six-digit values; S3.17 checks the 52-week boundary; S3.7 finds the criterion in any registry; S3.22 runs
  ruff from the interpreter's environment when installed there. The sixth (S3.1c expects failures the draft does not
  name) is recorded in the step 5 row for step 7's config text.
- `docs/spec_coverage.md` regenerated: `M1.A.5e` and `M5.F.4` read *partial* while their stage 3 tests are expected
  failures (an expected failure is not evidence), and the nine tests named for requirements revision 39 adds are listed
  as named for no spec ID until that text lands.
- Documents updated: the stage 3 plan (status line, step 5 row, §3.7's rows for S3.6, S3.7 and S3.19),
  `docs/theory/_STATUS.md`, `docs/REVIEWER_READING_LIST.md`, `CLAUDE.md` (the suite count and the ensemble count),
  `docs/spec_coverage.md` and `docs/spec_index.jsonl` (generated), `tests/doc_claims.json` (step 3's dated
  suite count of 1217) and this entry.

### Stage 3 step 4: spec revision 39, parts 2 and 3 drafted; revision 39 approved by the owner (2026-10-25)

- `docs/spec_revision_39_draft.md` gains part 2 (§11–§18) and part 3 (§19–§24). Part 2, group F: `M5.F.4` rewritten
  so a genuine `I-POSITION` is a self-statement, share `w` (the mover's self-channel weight), and a demand, share
  `1 − w`; the assertion form is the mixture at `w = 0`; `M5.F.2c`'s sentence; `M11.C.29`'s arm (D21); a new criterion after `M11.C.48`
  (four runs, the act's effect at level 35 against 60, each against its own control). Group A: a new requirement after `M4.C.1c`
  (contact that relieves its receiver returns in part to its sender, `reciprocal_contact_share` 0.5, `[I]`);
  `TRIANGLE` in layer 0 (`M4.D.3a`); `M11.C.42`'s and `M11.5`'s rows; `M1.C.1`'s `bound_anxiety` sentence. Part 3:
  T-128's named ratio in `M17.E.1` and `config/sweep/sweep.md`; T-014's `M8`, `M8.1`, `M8.5`, `M13.1`, `M11.C.7` and
  `M7.D.1a`; `M11.C.21`'s quoted effect reconciled with the records. Each block's config, explainer text, §19 check
  and stage 4 check against `_LEDGER.md` are in the draft. All 27 "replace" spans were checked by script against the
  revision 38 spec.
- The owner decided Q8–Q14 on 2026-10-25 through `AskUserQuestion`, each as recommended:
  `assertion_perspective_threshold` retired (Q8); the new criterion's act starts at `DEFINE` (Q9);
  `reciprocal_contact_share` 0.5 with its stated reasoning (Q10); relief judged on the whole delivery (Q11);
  `M1.C.1` *partial* with a spec sentence (Q12); the new criterion's mover at perspective 1.0 (Q13); the ratio enters only a
  criterion whose run is longer than one slow tick, refining D26 (Q14). The owner then approved revision 39 as a
  whole (parts 1–3), including part 1's reading of D19 (functioning returns with the lent pseudo-self). The spec stays at revision 38 until each
  group's text lands with its code, from step 6.
- One correctness pass over parts 2 and 3 (`sonnet`, read-only, about 250k tokens, 9 minutes) found 8 errors and 14
  lesser items. Twenty are fixed in the draft (two after Q13 and Q14). One is recorded as T-133 (`M5.D.4a`'s anger
  check at `DEFINE`). One was checked and left. `FE04.13`'s distancing share is recorded as T-132. Both are for the
  stage 3 milestone review (`docs/todo/2026-10-25_stage_3_revision_39.md`).
- The oldest entry (Model ergonomics, Part E) moved verbatim to `docs/CHANGELOG_archive.md` to keep this file under
  its size limit.
- Documents updated: the draft; the stage 3 plan's status line, step 4 row, §3.4's D26 paragraph and S3.17's row;
  `docs/theory/_STATUS.md`; `docs/REVIEWER_READING_LIST.md`; `docs/SPEC_REVIEW_CHECKLIST.md`; `CLAUDE.md`'s
  changelog sentence; the P2 proposal's status line; the TODO record. No engine change: nothing under `src/bowen/` or `config/bowen/` was edited, and
  the engine hash stays `66dcbb1e8ba5`.

### Stage 3 step 4: spec revision 39, part 1 drafted (group C), for the owner's approval (2026-10-25)

- The owner decided the displaced coping proposal's Q1–Q4 on 2026-10-25, each as recommended, through
  `AskUserQuestion`: `OVERFUNCTION` alone is displacing (Q1); `floor_held` is shed with the swings off the floor
  (Q2); P2 is conserved (Q3); revision 39 is drafted with P2 and F1 and `M11.C.6` reported as read (Q4, recorded in
  this session's first commit). T-130's two item names quote the paper ("Fired at work", "Wife begin or stop work").
- `docs/spec_revision_39_draft.md` (new) holds revision 39's part 1: the new displaced-coping requirement, `M1.A.5e`'s
  exception and `floor_held` ledger, `M1.A.4k` leaving `floor_held` out, `M7.E.1f`'s one sentence on what `d` reads,
  the new lent-functioning return (D19), `M11.C.6`'s and `M11.C.21`'s mutation columns and `M11.5` notes, the *displacing* table for
  `config/bowen/stress_response.md`, the explainer's text, T-130's wording for `M1.F.6a` and `event_sizes.md`, the §19
  check, and the stage 4 check against `KS04.7`, `FE01.1`, `FE06.3`, `FE07.4` and `KS00.9`. Each "replace" span was
  checked by script against the revision 38 text.
- One correctness pass over the draft (`sonnet`, read-only, about 65k tokens, 2 minutes) found 2 major and 5 minor
  items, all fixed in the draft: the lent-functioning return's capacity `s` written as `M6.I.4`'s formula with the `s_before` = 0 case,
  the week's timing and the dead lender's share, and what it adds to D19's wording (functioning moves with the
  returned pseudo-self; four copied forms) set out for the owner's review; displaced coping's floor clamped at 0, its three
  bounds listed and the receiver order stated, `FE06.3`'s difference written into its text, and its first step
  regraded (the passages `[T]`, the generalisation to `OVERFUNCTION` `[M]` `[I]`); `M7.E.1f`'s sentence says a
  negative system adjustment is read; §6's "Q6" names the T-130 review.
- From the stage 4 check, the owner decided two more questions as recommended (the draft's §9): Q5, `KS00.9` ("the
  basic levels of the parents establish the emotional climate") against `M7.E.1f`'s parents' term, which reads
  functioning that displaced coping raises in a parent who displaces onto the child, is recorded as T-131 for the stage 3
  milestone review and `M7.E.1f` is left as it is; Q6, each group's approved text is applied to the spec, config and
  explainer in the same commit as its code, so the spec stays at revision 38 until step 6.
- Parts 2 and 3 (groups F and A, T-128's `M17.E.1` and `M10.C.1a`, T-014's `M8`, `M8.1`, `M8.5`, `M13.1`, `M11.C.7`,
  `M7.D.1a`) are not drafted yet. Nothing under `src/bowen/` or `config/bowen/` changed and the spec is unchanged, so
  the indexes and the rules view were not regenerated; the engine hash is unchanged (`66dcbb1e8ba5`).
- Documents updated: the proposal (status line, §9), the plan (status line, D18's row, §3.8 step 4, §3.11),
  `_STATUS.md`, the reading list, the checklist, a new TODO record (`docs/todo/2026-10-25_stage_3_revision_39.md`,
  T-131) and `TODO.md`. `docs/version_2_plan.md` is unchanged: it carries no stage 3 status line, and step 4 is not
  done until revision 39 is approved.

### Stage 3 step 3: P2 and F1 probed together on Mac Mini1 (2026-10-25)

- The four F1 variants ran on Mac Mini1 (150 seeds, 10 workers, about 27 minutes), and their rows were joined by seed
  with the five earlier variants' rows here; `docs/review/stage_3_displaced_coping_probe.md` is regenerated from both
  and its Reading rewritten. `M11.C.6` expects a negative difference. With F1 the difference in `development` falls from
  +0.0886 ± 0.0439 to +0.0288 ± 0.0245; with P2 `OVERFUNCTION` and F1 to +0.0178 ± 0.0254, an interval that includes
  zero (the child's own term −0.0087 ± 0.0185, the parents' +0.0262 ± 0.0191). F1 kept gives the same within the
  intervals. No variant shows the decline; reported as read, nothing tuned.
- `docs/PROPOSAL — DISPLACED COPING (P2).md` gains §6.1 (the joint probe) and Q4 (draft revision 39 on that reading,
  recommended, or trace the parents' term first); §9's next step is revision 39's draft.
- The probe tool's summary no longer fails on a split run without `as_built` (the Mac Mini1 run's own render stopped
  there after writing its rows); the join test covers it. Nothing under `src/bowen/` or `config/bowen/` changed; the
  engine hash is unchanged (`66dcbb1e8ba5`).
- Documents updated: the probe page, the proposal (status line, §6.1, §9), the plan (status line, §3.3.1, §3.8 step 3),
  `_STATUS.md` and the reading list.

### Stage 3 step 3: F1 added to the displaced coping probe, for the joint run on Mac Mini1 (2026-10-25)

- `tools/stage_3_displaced_coping_probe.py` gains F1 (`docs/PROPOSAL — DISPLACED COPING (P2).md` §4.2) as four
  variants: F1 alone and with P2 `OVERFUNCTION`, each with the ledger shed in a calm week (§9 Q2's recommendation) or
  kept. F1 is put in by wrapping `tick.respond` and `tick.record_week` in the tool's process: what a bound held goes
  on a per-person `floor_held` ledger, which the estimator's reading leaves out. `--variants` runs a subset, and
  `--rows-in` joins several rows files by seed, so the five earlier variants' rows are reused, not rerun.
- Two tests added (`tests/test_stage_3_probes.py`): the F1 ledger holds at the floor and sheds off it (its floor bound
  proved by a mutant, run), and split runs join by seed. Suite: 1217 tests, 1216 pass and 1 strict expected failure.
- Nothing under `src/bowen/` or `config/bowen/` changed; the engine hash is unchanged (`66dcbb1e8ba5`). Documents
  updated: `CLAUDE.md`'s suite count; the probe page and the proposal follow with the run's numbers.

### The two-Mac driver reaches Mac Mini1 through the `macmini1` alias (2026-10-25)

- `tools/two_mac_sweep.py`'s `HOST` is now `macmini1`, the `~/.ssh/config` alias on this Mac (mini1.local, user
  davemini, a key of its own without a passphrase, IPv4 only). Mac Mini1 refused this Mac's default key, and its IPv6
  link-local route reset connections. `docs/two_mac_sweep.md` names the alias under *Before you start* and in the
  `fetch` step. A live `check` reached Mac Mini1 through it.
- `.gitignore` takes the eleven lines a security scan added on Mac Mini1 on 2026-10-25 (`node_modules/`, `.env.*`
  except `.env.example`, and backup copies such as `*.bak-*`), so that Mac Mini1's checkout is clean after it pulls;
  `check` refuses a split while either Mac has uncommitted changes to tracked files. No tracked file is ignored by them.
- Nothing under `src/bowen/` or `config/bowen/` changed; the engine hash is unchanged (`66dcbb1e8ba5`). Documents
  updated: `docs/two_mac_sweep.md`.

### Stage 3 step 3: P2 (displaced coping) drafted with F1, probed, for the owner's approval (2026-10-25)

- `docs/PROPOSAL — DISPLACED COPING (P2).md`: group C's mechanism, a new requirement beside `M1.A.5e`. A loaded parent's `OVERFUNCTION`
  toward someone meets their own load as coping does under `M1.A.5e` and moves the same amount of functional level to
  the receiver, conserved. Each step is graded and attributed; the §19 check, the constants (none new), a *displacing*
  table for `config/bowen/stress_response.md`, the `M11.5` classes, tests S3.1a–S3.1d, three mutants and the spec
  changes for revision 39 are set out. Q1–Q3 in its §9 are open for the owner.
- `tools/stage_3_displaced_coping_probe.py` (outside every `HASHED_TOOLS` list) puts P2 in by wrapping `tick.respond`
  in its own process and writes `docs/review/stage_3_displaced_coping_probe.md`: 150 seeds, five variants. P2 does not
  move `M11.C.6`'s direction at coping size: it lowers the child's own term and raises the parents' term by about as
  much (`development` +0.0886 ± 0.0439 as built, +0.0919 ± 0.0420 with P2 `OVERFUNCTION`). Of the own term's
  +0.0951, +0.0620 is functioning the stress response's floor holds while the swings decay (`stress_response.respond`),
  which the estimator credits to the child.
- Owner decision, 2026-10-25: P2 as drafted plus F1, the floor-held functioning put on a ledger (`Person.floor_held`)
  that the estimator nets out; the two are probed together on Mac Mini1 before revision 39. `M11.C.6` is reported as
  read and never tuned.
- The plan's §3.3.1, §3.7 (S3.1 replaced by S3.1a–S3.1d) and §3.8 are restated to match. One test added
  (`tests/test_stage_3_probes.py`). Nothing under `src/bowen/` or `config/bowen/` changed; the engine hash is unchanged
  (`66dcbb1e8ba5`).
- Documents updated: the proposal (new), the probe page (new), the plan and its status line, `_STATUS.md`, the reading
  list, and the checklist. The oldest live entry (Model ergonomics, Parts A–C, 2026-10-22) moved verbatim to
  `docs/CHANGELOG_archive.md` to keep this file under its size limit.
- Housekeeping the suite required: the new probe page added to `tools/record_changes.py`'s `NOT_COMPARED` and both
  changes pages regenerated; `docs/spec_coverage.md` regenerated; the 2026-10-24 entry's suite count listed as history
  in `tests/doc_claims.json`; `CLAUDE.md`'s suite count. Suite: 1215 tests, 1214 pass and 1 strict expected failure.

### Rule: a finished session writes the next session's brief (2026-10-24)

- `CLAUDE.md`, token budget: the owner's rule that a session whose work is done ends with a brief for the next session
  that can be pasted as written.

### T-130 checked: the event sizes agree with Holmes and Rahe (1967) (2026-10-24)

- The owner supplied a scan of the article (`Holmes1967SRRS.pdf`, journal pages 213–218; its embedded title is the
  article's DOI). All thirteen values in `config/bowen/event_sizes.md` agree with Table 3 on p. 216. Two item names
  differ from the paper's wording: `JOB_LOSS` ("Fired at work", where the config says "Dismissal from work") and
  `FIND_WORK_LOAD` ("Wife begin or stop work", where the config says "Spouse begins or stops work"). Two more differ
  by an article. `docs/review/stage_3_t130_event_sizes.md` gives the comparison value by value, with each rank.
- No value needs correcting. Dropping "unchecked" from the config, the spec (`M1.F.6a`) and the explainer, the
  paper's wording in the `item` column, and T-130's tick wait for step 6 (plan §3.6). The wording choice was decided
  as recommended, pending the owner's review. Nothing under `src/bowen/` or `config/bowen/` changed; the engine hash
  is unchanged (`66dcbb1e8ba5`).
- Documents updated: the review page, the plan's status line, `_STATUS.md`, the reading list, the checklist entry,
  and the T-130 note in its record.
- `c3abfb3` was pushed red: `test_d5b_status_claims_agree_with_each_documents_own_status` could not read the review
  page's status line, which was written as `**Status, 2026-…` instead of `**Status: …`. The background run's
  notification gave the exit code of the trailing `echo`, not pytest's. The fix commit restores the line's form.

### Stage 3 steps 1 and 2: the projection trace, the I-position probe, T-130 not yet checked (2026-10-24)

- **Step 2a, the projection trace** (`tools/stage_3_projection_trace.py`, in no `HASHED_TOOLS` list;
  `docs/review/stage_3_projection_trace.md`). `M11.C.6`'s own arm, both arms, seeds 0–149, 8 min 7 s at 10 workers on
  this Mac. Summed over the development steps at ages 6–20, treatment − baseline: `development` +0.0886 ± 0.0439 (the
  record's value, reproduced); the child's own term +0.0951 ± 0.0362; the parents' term −0.0072 ± 0.0296. The band
  bound 6 of 2250 steps in each arm. The step reads the estimator's functional level, which already leaves out
  pseudo-self (`slow_tick.py::record_week`, revision 23), so the exchange moves neither term; plan §3.1's reading
  was wrong there and is marked corrected. About half of the own-term difference is the child's stress swing
  (`M1.A.5e`), from age 14: +0.94 of +1.90 summed over ages. Read on the same weeks, P1 would move the parents' term
  from −0.30 / −0.31 to −5.05 / −5.28 (baseline / treatment), with a difference of −0.17 ± 0.56 (first-order, no
  re-run). **D18's condition was met.** The owner was asked again and chose P2, displaced coping (D18, revisited):
  P1 is dropped, and P2 is drafted for approval before revision 39.
- **Step 2b, the I-position probe** (`tools/stage_3_iposition_probe.py`, in no `HASHED_TOOLS` list;
  `docs/review/stage_3_iposition_probe.md`). On the triad, mover F at level 35 or 60, perspective 1.0, `I-POSITION`
  toward M scripted at week 1 (at week 4, F was angry in 13–14 of 40 seeds); no spell in weeks 1–5; F's own
  `I-POSITION` withheld every other week; a no-act control at each level; seeds 0–199, 25 s. Level 35 − level 60,
  both acting: the family's excess +7.40 ± 0.36, M's acute anxiety +0.78 ± 0.19. The controls differ by as much. The
  act's own effect at level 35 less at level 60 is −0.06 ± 0.32 on the family's excess, −0.04 ± 0.23 on M's acute
  anxiety and −0.005 ± 0.026 on the `CONFLICT` share. At level 35, 132 of 176 genuine sequences failed PREPARE
  (14 of 176 at level 60). Completions: 0 at level 35, 9 at level 60. **D20's condition was not met**, so D20 stands
  (build A) and was not put again. For the plan's new group F criterion (§3.7): a control or the `w = 1` mutant is needed, and the low-level
  mover's PREPARE window has to be calm enough to reach DEFINE.
- **Step 1, T-130: not checked.** The publisher refuses automated access (HTTP 403) and a search found only
  reproductions of the scale, not the article. `docs/review/stage_3_t130_event_sizes.md` records the attempts and
  the thirteen values to check. T-130 stays open and nothing in `config/bowen/event_sizes.md` changes; the owner
  will supply the PDF.
- Smoke tests: `tests/test_stage_3_probes.py`, two tests on seeds 0 and 0–2. `docs/spec_coverage.md` was regenerated:
  it lists the two tests under "Tests named for no spec ID" (267 → 269), and nothing else changed. The engine hash is
  unchanged (`66dcbb1e8ba5`); no record goes stale.
- Housekeeping the suite required: the oldest live changelog entry (SP-1 built, 2026-10-22) moved verbatim to
  `docs/CHANGELOG_archive.md`, checked by script against `HEAD` (the live file was 83,031 bytes against its 80,000
  limit); the three new pages added to `tools/record_changes.py`'s `NOT_COMPARED`, and both changes pages regenerated
  (one row each); `CLAUDE.md`'s suite count. Suite: 1214 tests, 1213 pass and 1 strict expected failure.
- Documents updated: the plan's status line, D18 and D20 rows and §3.3.1–§3.3.2 results; the checklist (three new
  pages); the reading list; `_STATUS.md`; the T-130 item's record. The spec, the explainer and the config are
  unchanged because nothing a run computes changed.

### Version 2 stage 3's implementation plan approved (2026-10-24)

- The owner approved the plan as written ("Approved: the stage 3 plan as written"). Its status line, the checklist,
  the reading list and `_STATUS.md` say so. Steps 1 (T-130) and 2 (the projection trace and the I-position probe)
  are next, in a session of their own.

### Version 2 stage 3's implementation plan drafted (2026-10-24)

- `docs/implementation_plan_version_2_stage_3.md`, a draft for owner approval: one proposal for each group the owner
  chose (C, projection; F, `I-POSITION` at a low level; A, triangle relief), `M17.E.1`'s named ratio (T-128), T-014's
  four questions, where T-129's edits land, every criterion with its test and mutation, the order of work and the
  compute. Three read-only agents gathered the group facts (about 440k subagent tokens, about 6 minutes in parallel).
- What the reading found: under projection `M11.C.6`'s child ends higher, not lower, because the development step
  averages the child's functioning with the parents', and the parents' term counts pseudo-self borrowed from the child;
  `M11.C.29` completes no genuine `I-POSITION` because its act falls on a spell week; a `TRIANGLE` relieves its seeker
  about a third of a `DISTANCE` and its contact never reaches the seeker; `Triangle.bound_anxiety` is never written.
- The owner decided D18–D31 the same day through `AskUserQuestion`, one at a time, each as recommended: group C, a
  trace then the parents' term net of borrowed pseudo-self, and lent functioning returned on a collapse; group F, a
  probe then the act weighted between self-statement and demand by level, `M11.C.29`'s mover at level 60 with F's
  spell paused, and a new criterion comparing a lower- and a higher-level mover; group A, contact returned to a relieved sender's tie and `TRIANGLE` moved to layer
  0; T-128, `tie_wear_rate` ÷ (52 × `calm_transfer_rate`) over a quarter to four times its central value on every
  directional criterion whose run spans a slow tick; T-014, the avoidance gate dropped, fusion left unbuilt, the
  coach's going along with a `SPLIT` declared, the illness size kept; one batch with one regeneration.
- Nothing a run computes changed; the engine hash is `66dcbb1e8ba5`. Documents updated: the plan,
  `docs/SPEC_REVIEW_CHECKLIST.md`, `docs/REVIEWER_READING_LIST.md`, `docs/theory/_STATUS.md`,
  `docs/implementation_plan_version_2.md`'s status line, and the T-014 and T-128 records (decided, open until built). This entry took `CHANGELOG.md` over its size limit, so
  the earliest entry of 2026-10-22 (the SP-1 proposal's eight questions) moved verbatim to `docs/CHANGELOG_archive.md`,
  and `CLAUDE.md`'s line on what the live file holds says so.

### The stage 2 milestone review triaged, spec revision 38 (2026-10-24)

- The review (`docs/cycles/2026-10-24_milestone-review_v2-stage-2.md`, branch `review/2026-10-24-v2-stage-2`) was
  fast-forwarded into `main`. Its triage is `docs/todo/2026-10-24_stage_2_milestone_review.md`: two agents checked
  the 22 unchallenged findings against their files and dropped none (16 hold as stated, 6 in part). The owner decided
  the review's seven questions, the batch scope, the class fixes, lint and MR-12, each as recommended.
- Spec revision 38, wording only: `M1.D.2b`'s decay and `M1.D.2c`'s reference level restated to the code (Q1);
  `M11.C.48` in `M11.4`'s exceptions (Q6); `M17.A.1`'s refusal placed before the rerun it refuses (MR-12);
  `M1.F.6a`'s source qualified (MR-14); `M10.A.1`, `M10.A.2`, `M10.C.2` (MR-27), the "Move physics" and
  `reference_level` register rows (MR-11, MR-8) and `M11.C.48`'s test names (MR-9) corrected; the log's revision 36
  and 37 tags say approved (MR-16).
- Tools: `tools/spec_coverage.py` no longer counts an expected failure as evidence, so `M5.B.3` is *partial* (MR-4);
  `M17.E.1` is *partial* in the overrides until its named ratio is swept (MR-1, T-128); `M1.B.3`'s override note
  re-derived (MR-10); `tools/version_2_status.py` labels a pass whose named mutant survived as such (MR-28: `M11.C.11`,
  `M11.C.15`, `M11.C.48`, `M11.C.48[phase_d]`); `tools/phase_d_record.py` and `tools/version_2_sweep.py` print
  "not evidence (T-123); named mutant survived (T-119)" beside `M11.C.48`'s verdict (MR-3), re-rendered from the
  cache with nothing rerun; `tools/version_2_sweep.py` refuses a criterion that fixes fewer seeds than the rerun count
  before running the rerun, not after (MR-12), and puts `--recompute`'s dropped results back in `try/finally` (MR-25).
- Tests: SP-B's gate, `tests/test_named_tests_resolve.py`, resolves every test a spec row or `Tests:` line names,
  with allowlists that can only shrink; `tests/test_group_g_readout.py` (MR-9); broken-table tests for the event-size
  and contact-class loaders (MR-23); the `M5.B.3` arm checks moved out of the expected failure (MR-24); ruff F items
  under `tools/` and `tests/` (MR-26); the suite-size gate reads "N tests: P pass and X is a strict expected failure"
  and checks the sum (MR-20).
- The batch's two review passes (one `learning-qa`, one correctness) found eight items, all fixed here: a
  `VERDICT_NOTES` note is now checked against the mutation records and `TODO.md`, and a key no record holds is
  refused; the suite-size gate checks its expected-failure figure against the marks and refuses "all N tests pass"
  while one exists; rule 2's "an expected failure is not evidence" holds under an override or a recorded verdict too,
  and `xfail_names` reads class, module, parameter, imperative and `async` forms; the gate's allowlist reason for
  `M1.F.2`'s and `M11.D.12`'s unwritten tests names T-004, not revision 38; the explainer's `M1.D.2c` line names
  three `DISTANCE`s and the declared start; the "Move physics" row lists its `reference_level` write; a sweep test
  shows a short-seeded criterion is swept while no sample needs a rerun. The seed-7 trace, chart and fam-diag export
  under `docs/review/` were regenerated for the revision number. The suite: 1212 collected, 1211 pass, 1 strict
  expected failure.
- Documents: the explainer (`REDUCE_CUTOFF`, the cutoff's end, the decay, the reference level, the event sizes'
  source, the detection limit, one KB phrase paraphrased, its date), `CLAUDE.md`, `README.md` (version 2 status page
  and generators), `docs/REVIEWER_READING_LIST.md`, `docs/SPEC_REVIEW_CHECKLIST.md`, `docs/theory/_STATUS.md`,
  `docs/theory/_SOURCE_QUALITY.md` (what counts as quotation, Q3), `docs/implementation_plan_version_2.md`,
  `docs/version_2_plan.md`, `docs/two_mac_sweep.md` (Q4), the Holmes–Rahe proposal, `docs/spec_revision_log.md`,
  `LEARNINGS.md` (Misses), T-004, T-011, T-012, T-014 and T-015; regenerated: `docs/spec_index.jsonl`,
  `docs/spec_rules/`, `docs/spec_coverage.md`, `docs/version_2_status.md`, `docs/phase_d2_ensemble_record.md`,
  `docs/version_2_sweep_record.md`, `TODO.md`.
- Not changed, by the owner's scope: `src/bowen/` and `config/bowen/`, so the engine hash is `66dcbb1e8ba5` as before.
  Their fixes (MR-2, MR-9, MR-13 and MR-26's docstrings and lint) wait for the next change that moves the hash
  (T-129). Opened: T-128 (`M17.E.1`'s ratio, stage 3's plan), T-129, T-130 (the event sizes against the paper).

### T-125 closed: the owner approved the fifty-three overnight decisions (2026-10-24)

- The owner approved all fifty-three decisions of T-125 and kept the branch `claude/confident-maxwell-vr5zol` (decision
  27). Each status phrase that stated one of them as pending the owner's review now reads "approved by the owner
  2026-10-24" with T-125; the T-122 verdict changes, a finding and not a decision, stay pending.
- Changed: `CLAUDE.md`, `README.md` (also: revision 36's branch is merged into `main`, not "not merged"),
  `config/sweep/sweep.md`, the spec (status line, two revision 36 tags, two revision 37 tags), `docs/PROPOSAL — CUTOFF
  ANXIETY, BRIDGING AND REPAIR.md`, `docs/PROPOSAL — RELEASE BY TIE (G2).md`, `docs/PROPOSAL — EVENT SIZES FROM THE
  HOLMES–RAHE SCALE.md`, `docs/REVIEWER_READING_LIST.md`, `docs/SPEC_REVIEW_CHECKLIST.md`,
  `docs/implementation_plan_version_2.md`, `docs/model_explainer.md`, `docs/readouts_phase_c.md`,
  `docs/theory/_STATUS.md`, `docs/todo/2026-08-22_blocking_the_next_phase.md`.
- T-125 ticked in `docs/todo/2026-10-23_overnight_decisions.md`; `TODO.md`, `docs/TODO_archive.md` and
  `docs/spec_rules/README.md` regenerated (`docs/spec_index.jsonl` unchanged). Nothing a run computes changed.
- Not changed: the status lines of `config/bowen/contact_classes.md` and `config/bowen/event_sizes.md`, which still say
  pending. The engine hash covers every `*.md` under `config/bowen/`, so editing them made every record stale; the
  edits were reverted and the hash is `66dcbb1e8ba5` as before. They are corrected at the next change that moves the
  hash (T-127).

### Stage 2: Phase D part II's seven criteria swept across two Macs (2026-10-24)

- Split by the part II record's seed counts: Mac Mini1 ran `M11.C.6`, `.10[family of origin]`, `.48` and `.48[phase_d]`
  (17:38:43–00:30:49, 36 child runs), this Mac `M11.C.2[twelve-person family]`, `.10` and `.15` (17:39:01–01:32:59, 34
  child runs, `nice` 20). A first start with all seven on Mac Mini1 was stopped at 17:38 before any sample finished.
  The merge added 36 results and found the 361 already here equal; no sample was undetermined or errored.
- Held of 8 (central verdict): `.2[twelve-person family]` 2 (FAIL), `.6` 0 (FAIL), `.10` 1 (FAIL), `.10[family of
  origin]` 5 (FAIL), `.15` 4 (PASS), `.48` 8 (PASS), `.48[phase_d]` 8 (PASS). Stage 2's full sweep is complete.
- The seven keys left `config/sweep/sweep.md`'s *Declared, not run* list; the sweep record, `docs/version_2_status.md`,
  `docs/spec_coverage.md` and the spec index were regenerated; plan §2.10, `_STATUS.md`, the reading list,
  `docs/two_mac_sweep.md`'s status line and T-001 updated. Nothing a run computes changed.

### When the full sweep runs: at a stage's end only (D17, 2026-10-23)

- The owner, 2026-10-23: "a full sweep only at a stage's end; between, sweep only the criteria a change targets."
  Recorded as D17 in plan §2.10, with its reason (every result is keyed by the engine hash, and the full sweep is
  about two days on two Macs) and what it gives up (a pass made fragile by a change aimed elsewhere is found only at
  the next full sweep). Also in `CLAUDE.md`'s review rules and `docs/theory/_STATUS.md`, whose order of work now says
  part II's run was started at 17:38, split across both Macs. Documents only; nothing a run computes changed.

### T-126 closed: `nodal_spike_intensity` stays out of the `M11.C.2` sets (2026-10-23)

- The owner, 2026-10-23: "keep D10's rule; nodal_spike_intensity stays out of both M11.C.2 sets." T-126 ticked;
  decisions 25, 32 and 47 stand as to this constant. The note is added to the plan's §2.10 judgment (3), to
  `config/sweep/sweep.md`'s preamble, to T-125's decision 47 and to `docs/theory/_STATUS.md`. No range row, code or
  record changed.

### Part II's judgments listed for the owner, and two loose ends (2026-10-23)

- T-125 gains batch 6, part II's eight constant-set judgments and the *Declared, not run* list (decisions 45–53; now
  fifty-three decisions in six batches), each pointing to the plan's §2.10 paragraph that holds it in full.
- T-126 (Now) asks the owner whether `nodal_spike_intensity` should be swept in both `M11.C.2` sets, where the nodal
  load acts but no row credits `M4.C.1c`. No code changed.
- `docs/CHANGELOG_archive.md`: a stray copy of this file's old end-of-file pointer, left between two archived entries,
  removed; no entry changed. `docs/theory/_STATUS.md` names decisions 45–53 and T-126.

### Stage 2: Phase D part II's sweep ranges declared, no run (2026-10-23)

- The owner, 2026-10-23: declare part II's ranges now and run them later, on Mac Mini1 alone at 8 samples. 81 rows
  added to `config/sweep/sweep.md` (now 327 rows, 78 constants, 34 criteria) for the part II record's seven composite
  keys: `M11.C.2[twelve-person family]` (15 constants), `M11.C.6` (6), `M11.C.10` and `M11.C.10[family of origin]` (17
  each), `M11.C.15` (2), `M11.C.48` and `M11.C.48[phase_d]` (12 each). `M11.C.47` is a premise and is not swept. Each
  range is half to double the central value within the file's bounds, every grade `[I]`; `development_band_support`'s
  low end is raised to 0.04, the top of `development_band`'s range, for the engine's invariant.
- The sets follow D10's rule as the file's preamble states it; the preamble's counts, grades and exclusion list are
  updated, and the plan's §2.10 lists each set and the eight numbered judgments, decided as recommended under the
  owner's instruction of 2026-10-22, pending the owner's review. Checked against the code: the estimator skips a child
  under 20, so `M11.C.6` takes `M7.E.1f`'s four development constants; births to the third generation fall inside
  `M11.C.2`'s ages read, so `birth_focus_gain` is in its twelve-person set; no differentiating exchange completed in
  20 seeds of `M11.C.15`'s three runs, so `exchange_budget_reduction` is out; `nodal_spike_intensity` acts in the
  twelve-person runs but no part II row credits it.
- The record is built from every declared key, so declaring without a run made
  `test_s2_11_sweep_record_is_current` raise on a missing result. `config/sweep/sweep.md` gains a section *Declared,
  not run* listing the seven keys; `tools/sweep_config.py::load_not_run` reads it strictly, and
  `tools/version_2_sweep.py` shows a listed key in the record with its ranges and no number, runs none of its samples
  unless `--criteria` names it, and refuses the record once a listed key has a cached result. Three tests (S2.5, S2.6),
  each failed by a mutant. `docs/version_2_sweep_record.md` was rendered with `--render-only`; the cache did not
  change. `tools/version_2_status.py::sweep_fractions` leaves a not-run key out, so the status page still shows it
  *not swept* and W4 still counts it (its test extended). The plan's §2.10 and `docs/two_mac_sweep.md` describe the
  list.
- The `learning-qa` review found three low items, all fixed: the record now says the cache holds no result of a
  not-run key (a share whose samples all errored caches nothing, so "none has run" was not what is checked); the
  sweep tests' helpers patch the not-run list with their own config; a misspelt *Declared, not run* heading raises
  instead of listing nothing (a mutant fails the test).
- Checked: the file parses, `tools/version_2_sweep.py::_checked` passes for each key and for the seven together, and
  200 joint samples of each key pass `EngineParams`' invariants. Nothing in `src/bowen/` changed (`engine_hash()`
  `66dcbb1e8ba5`, `phase_d_engine_hash()` `5c3865dcf8f9`); no record was run. Suite: 1195 collected, 1194 passed and
  1 xfailed (`CLAUDE.md`'s count updated).
- This entry took the file past its 80,000-byte limit, so the six entries of 2026-10-21 moved to
  `docs/CHANGELOG_archive.md` verbatim (13,354 bytes, checked as one unchanged span), and a pointer line now ends
  this file; `CLAUDE.md`, the checklist and the reading list now say entries before 2026-10-22. The plan's status
  line, `docs/theory/_STATUS.md` and `docs/REVIEWER_READING_LIST.md` now say the part II ranges are declared.

### The two-Mac cross-check run live (2026-10-23)

- With Mac Mini1 pulled to `8501fc0`, `python3 tools/two_mac_sweep.py check` found both Macs equal on the commit,
  clean files, both engine hashes, Python 3.12.3, NumPy 1.26.4 and arm64 (macOS 15.7.9 here, 15.7.8 there, printed).
  The cross-check, `start --criteria M11.C.16 --recompute --workers 10` on Mac Mini1 (14:28:38–14:31:45, 8 child
  runs, exit 0), then `fetch` and `tools/merge_cache.py`, reported 0 results added and 361 already there and equal:
  Mac Mini1 recomputed `M11.C.16`'s 8 results bit for bit. Not compared yet: a Phase D criterion and a 200-seed
  rerun. `docs/two_mac_sweep.md`'s *Not verified* and the plan's §2.10 updated; no code or record changed.
- A second cross-check, `M11.C.23` (Phase D part I), dropped its 9 cached results on Mac Mini1 and ran them again (8
  first runs and sample 1's rerun at 200 seeds, 14:40:40–14:41:56, exit 0); the merge again reported 0 added and 361
  equal, and Mac Mini1's cache file matched the committed one byte for byte. The owner chose, for part II: declare its
  seven criteria's ranges with no run started; run it later on Mac Mini1 alone at 8 samples.

### Stage 2: the sweep can run on two Macs (2026-10-23)

- Before part II's seven criteria (projected 23–40 h on one Mac), the constant sweep (`M17.E.1`) can be split between
  davemini2 and Mac Mini1 and the results combined (`docs/two_mac_sweep.md`, the how-to, listed in
  `docs/SPEC_REVIEW_CHECKLIST.md`; `docs/implementation_plan_version_2.md` §2.10). Nothing in `src/bowen/` changed;
  `engine_hash()` printed `66dcbb1e8ba5` before and after; no record was rerun. No split run has been made.
- `tools/version_2_sweep.py --criteria KEY ...`: checks every declared key first, refuses an undeclared one, runs only
  its share and writes only the cache with every entry kept, never the record. `--recompute` reruns the named
  criteria, the cross-check that the two Macs compute the same numbers. `--render-only` renders the record from the
  cache and runs nothing; a missing result stops it with nothing written.
- `tools/merge_cache.py OTHER`: the union of two cache files, refused with nothing written when one key holds two
  results, the NumPy versions differ or are missing, or a sweep is running on this Mac.
- The NumPy version is recorded in a cache file under `_meta` (`tools/record_cache.py`), outside the result keys; a
  cache without one is written without one. `docs/records_cache/sweep_v2.json` stamped `1.26.4`, the version all its
  361 results were made under. `requirements-sweep.txt` pins `numpy==1.26.4` for sweep machines; the sweep refuses
  another version or a cache with results and no version, and `--render-only` is not checked (CI runs NumPy 2).
  `requirements.txt` stays unpinned for CI's Python 3.13.
- `tools/two_mac_sweep.py` (`check`, `start`, `status`, `fetch`): checks Mac Mini1 against this checkout (commit,
  clean files, engine hashes, Python, CPU, NumPy), starts its share detached with its venv's Python, confirms it is
  running, logs `START`/`EXIT` lines, and copies its cache back to a new file. Tested with `subprocess.run` stubbed
  and the remote commands run under a local `sh`; not run against a live Mac Mini1 beyond its `pgrep` call.
- The `learning-qa` pass's seven findings fixed. Tests: `tests/test_sweep_two_macs.py`, 31 tests (43 with
  parameters), each shown failing under mutation (46 mutants, all killed). The suite: 1192
  tests (CLAUDE.md updated from 1149).

### Stage 2: Phase D part I's directional criteria swept (2026-10-23)

- The constant sweep (`M17.E.1`) run over the Phase D part I ensemble record's 13 directional keys (composite and
  mixed, `M11.5`; `M11.C.7`'s three pairs each a key) at 8 samples each, on the branch `stage-2-part-i-sweep`
  (`docs/implementation_plan_version_2.md` §2.10, D15's second step; the owner, 2026-10-23). 182 child runs — 104 first
  runs and 78 reruns at 200 seeds under the detection-limit rule (`M17.A.1`) — in 12,843 s of run time (3 h 34 min),
  in three segments at 10, 5 and 8 workers, the owner having asked for the Mac back during the run; projected 1.8 h at
  Phase C's ratio and 3.2 h at the pilot's, both at 10 workers. No sample undetermined or errored. Held of 8:
  `M11.C.2` 1, `M11.C.7[joint over one spouse]` 3, `M11.C.7[joint over sided]` 7, `M11.C.7[one spouse over sided]` 5,
  `M11.C.11` 8, `M11.C.12` 0, `M11.C.17` 4, `M11.C.21` 1, `M11.C.22` 6, `M11.C.23` 0, `M11.C.34` 0, `M11.C.39` 8,
  `M11.C.43` 0 — measurements of the declared ranges, not verdicts. W4 now lists three unmet rows, not seven
  (`docs/version_2_status.md`, regenerated).
- `config/sweep/sweep.md`: 125 rows for the 13 keys (246 rows, 60 constants, 27 criteria), every range `[I]`, and the
  preamble's rules for what is left out (no birth in a part I scenario; a constant an arm's declared condition is set
  against). The sets are judgments, listed in T-125's batch 5 (decisions 29–44) for the owner's review.
- Fixed: the sweep ran every criterion through `tools/mutant_runner.py`'s child, which knows only Phase C's criteria,
  so the first part I run raised `KeyError` on all 104 samples in 81 s. A criterion of a Phase D record now runs
  through `tools/phase_d_mutant_child.py` and is cached under `tools/phase_d_mutation_record.py`'s hash, printed in the
  record as `phase_d_engine_hash`; `tools/record_cache.run_cached` takes the child and the hash as arguments. Test:
  `tests/test_version_2_sweep.py::test_m17e1_phase_d_criterion_is_run_by_the_phase_d_child_under_its_hash`, shown
  failing with the routing removed and with the Phase C hash used for Phase D keys.
- Nothing in `src/bowen/` changed; the engine hash is `66dcbb1e8ba5` before and after. Documents: the plan's §2.10,
  `docs/model_explainer.md` (the sweep's two children), `docs/theory/_STATUS.md`, `docs/REVIEWER_READING_LIST.md`,
  T-001 and T-125.
- `CHANGELOG.md` passed its 80,000-byte limit with this entry: its three entries dated 2026-10-20, which the archive's
  header already counted as moved, went to `docs/CHANGELOG_archive.md` verbatim. `docs/spec_coverage.md` and the two
  index files regenerated for the new tests; `CLAUDE.md`'s suite count is 1149.
- From the `learning-qa` review (5 min 25 s): `tools/record_cache.run_cached` refuses a child other than Phase C's
  given without the hash its results are keyed by (P8); `docs/version_2_status.md` lists the sweep record's
  `phase_d_engine_hash` beside its `engine_hash`; two figures in the plan's §2.10 corrected. Tests:
  `test_m17e1_another_child_without_its_hash_is_refused`, `test_s2_11_records_read_shows_every_hash_a_record_keys_by`,
  each failing on the code before the fix.

### The symptom module's docstring restated from the code (2026-10-23)

- `src/bowen/engine/symptoms.py`'s module docstring still said the spill past the lock-in turn is "split equally".
  Since spec revision 36 (G2) the release and the spill are both split by bond energy among the living family members
  on interactive ties (`_release_shares`); ties all at zero split equally and an external agent takes none. The
  docstring now says so. `M7.D.2d` in `docs/spec_rules/M07.md`, `docs/model_explainer.md` and
  `docs/readouts_phase_c.md` already matched the code and did not change. No rule changed; the engine hash
  (`tools/record_cache.py`, docstrings excluded) is `66dcbb1e8ba5` before and after.

### The night's decisions listed in one place for the owner (2026-10-23)

- `docs/todo/2026-10-23_overnight_decisions.md`, one Now item (T-125): the twenty-eight decisions taken as recommended
  under the owner's instruction of 2026-10-22 ("proceed based on your recommendations - do not wait for my input"),
  in four batches — spec revision 36 (group G, the event sizes), the stage 2 plan and pilot (revision 37), the Phase C
  constant sweep, and the coordinating session's own calls (the superseded cloud G2 proposal, the changelog archive
  split) — each with where it is recorded. Ticking the item means the owner has reviewed the list.

### Stage 2: Phase C's directional criteria swept (2026-10-23)

- The constant sweep (`M17.E.1`) run over the Phase C ensemble record's 14 directional criteria (composite and mixed,
  `M11.5`) at 8 samples each, on the branch `stage-2-phase-c-sweep` (`docs/implementation_plan_version_2.md` §2.10,
  D15's first step): `python3 tools/version_2_sweep.py --workers 10`, 05:32:36 to 05:59:37 on this Mac, 1620 s wall
  (27 min 0 s) for 155 child runs — 96 first runs for the 12 criteria the pilot had not run and 59 reruns at 200
  seeds under the detection-limit rule (`M17.A.1`); the pilot's 24 results came from the cache. Per sample 16.9 s
  against the pilot's 29.8 s; 27 min against the 48 min the pilot's ratio projected for these 12 (56 min for 14).
- `config/sweep/sweep.md` gains 104 range rows (121 rows, 31 constants, 14 criteria), each half to double the central
  value within the file's bounds, every grade `[I]` (`defence_threshold` is the one constant the explainer names, §11.3,
  `[I]` there); its preamble states which constants a criterion's set holds and why, and the plan's §2.10 lists the
  judgments, each decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review.
- Held of 8, a measurement of each range and never a verdict: `M11.C.5` 0, `M11.C.16` 8, `M11.C.27` 2, 3, 6, 6
  (stable add, stable remove, unstable add, unstable remove), `M11.C.29` 0, `M11.C.41` 8, 8, 0, 1 (light: lower
  level; heavy: lower level; higher level: heavier stress; lower level: heavier stress), `M11.C.42` 0, `M11.C.44` 2,
  `M11.C.45` 0. No sample undetermined or errored. `docs/version_2_sweep_record.md` and `docs/version_2_status.md`
  regenerated; the status page's W4 now names the seven passing Phase D rows not yet swept.
- Nothing in `src/bowen/` changed; `engine_hash()` printed `66dcbb1e8ba5` before and after; no ensemble, mutation or
  diagnostic record was rerun. Documents: the plan's status line and §2.10, `docs/theory/_STATUS.md`,
  `docs/REVIEWER_READING_LIST.md`, T-001's text (`docs/todo/2026-08-22_blocking_the_next_phase.md`).
  `docs/model_explainer.md` §18.7 quotes the two rule values, which did not change, so it is unchanged.

### Stage 2's pilot built, with spec revision 37 (2026-10-23)

- The constant sweep of `M17.E.1` and the detection-limit rule of `M17.A.1`, built for stage 2's pilot on the branch
  `stage-2-plan-pilot` (`docs/implementation_plan_version_2.md` §2.10). `tools/version_2_sweep.py` draws joint
  samples of the `[I]` constants a criterion's mechanism reads over the ranges in `config/sweep/sweep.md` (new;
  outside `config/bowen/` because the engine hash covers that directory, D12), runs each as an edit of
  `constants.md` in a copy of the repository through `tools/mutant_runner.py`, reruns a sample whose first run left a
  gating effect within 2 half-widths of zero at a fixed 200 seeds (D8), and renders `docs/version_2_sweep_record.md`
  (new) from `docs/records_cache/sweep_v2.json` as *n of S* samples held — a measurement, never a verdict.
  `tools/sweep_config.py` (new) parses the config strictly. Nothing in `src/bowen/` changed; `engine_hash()` is
  `66dcbb1e8ba5` before and after; no ensemble, mutation or diagnostic record was rerun.
- **The pilot:** `M11.C.16` and `M11.C.42` at 8 samples, 10 workers: 476 s wall for 24 child runs (16 first runs, 8
  reruns at 200 seeds, all of them `M11.C.42`'s). `M11.C.16` held in 8 of 8 samples, `M11.C.42` in 0 of 8. **The
  projection:** 29.8 s per sample, 3.76 times the Phase C record's time per row; the full sweep over the 34
  directional rows at 8 samples is 44 h (part II alone 40 h; Phase C and part I 4.1 h; Phase C alone 56 min), so
  **the full sweep did not run** (the owner's 4-hour rule). D14 (the narrow mark: under half of at least 32 samples;
  not built yet), D15 (the full sweep not run; Phase C's 14 criteria next, 56 min) and D16 (no drafted `M17`
  requirement comes in) decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review.
- `docs/version_2_status.md` (new), from `tools/version_2_status.py` (new): every `M11.C` criterion with its verdict
  now, its verdict at the tag `version-1`, its mutation evidence, its sweep fraction and its status under W1–W4. At
  this commit W2 is met; W1 is not (9 rows: the five verdicts changed since the tag, T-122, and the four passing rows
  without a recorded mutation); W3 is not (`M11.C.11` and `M11.C.15`, whose mutants, red at the tag, survived at revision 36, and
  `M11.C.48` in both variants, T-123); W4 is not (11 passing directional rows not yet swept). The plan's draft said
  W3 was unmet on `M11.C.44`; it has `one-sided-deviation` red, and the plan is corrected (§2.9, §2.10).
- **Spec revision 37** (wording in `M17` only): `M17.A.1` gains the detection limit in a sweep, `M17.E.1` the sample
  rule (a joint seeded draw, each sample an edit of `constants.md` in a copy, *n of S* with undetermined and errored
  samples counted apart, a fraction never a pass or a fail, no premise swept). §0.3's count is 1055 MUSTs / 92 tests.
  `docs/spec_revision_log.md` has the entry; the indexes, the rules view and the coverage page are regenerated.
- Tests: `tests/test_version_2_sweep.py` (15) and `tests/test_version_2_status.py` (6), each shown failing under its
  mutation (plan §2.10); `tests/test_doc_checklist.py` now covers `config/sweep/`. The suite is 1146 tests.
- **The reviews** (plan §2.10, step 6), on the build commit's diff, 04:43:45 to 04:53:07: `learning-qa` 6 min 43 s and
  a correctness pass 8 min 05 s. Fixed in the next commit, each with a test: the frozen page's `M11.C.8` row was
  dropped by the reading pattern and no count checked it (P2); a changed row lost its version 1 disposition (now shown
  as not applied); a criterion with its own `seeds` below the fixed count would have been recorded as rerun at 200
  seeds without being so (now refused, `M17.A.1`); a `log` range on an integer constant was not refused; a premise late
  in the config was refused only after earlier samples ran; three tests were tied to the pilot's config; wording in
  the spec, the explainer, the config and the plan. One minor item recorded, not fixed: T-124. The re-sweep of the fix
  diff (sonnet, 2 min 20 s) found no blocking item; its minor items are fixed in the same commit (plan §2.10).
- Documents: the plan (status line, §2.4 W3, §2.9, §2.10), `model_explainer.md` §18.7 (the stage 2 build, the
  detection limit and the sweep, each `[I]`), `SPEC_REVIEW_CHECKLIST.md` (the config and the two pages),
  `REVIEWER_READING_LIST.md` (§2, §5, §7), `_STATUS.md`, `README.md`, `CLAUDE.md`, T-001, T-018 and T-123. Three
  renders that print the spec's revision were re-rendered with no other change: the seed-7 trace and chart
  (`tools/phase_c_trace.py`) and the famdiag export (`tools/famdiag_export.py`); the two changes pages
  (`tools/record_changes.py`) list the new pages as not compared.

### The stage 2 plan drafted (2026-10-23)

- `docs/implementation_plan_version_2.md` §2: version 2's stage 2 (the measurement stage, `docs/version_2_plan.md` §3
  items 2.1–2.5), in the stage 1 format. What the code shows (`M17`: 4 of 29 requirements built; the D9 sweep is six
  points, not `M17.E.1`'s fraction; a sweep sample is a `Mutant` over `constants.md` rows in a copy, so the engine hash
  does not move; the hash covers `config/bowen/*.md`, so the sweep's own config lives in `config/sweep/`); six decisions
  D8–D13, each decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review — D8 the
  detection-limit rule (a sample whose gating effect is within 2 half-widths is rerun at 200 seeds; `M17.A.1` wording,
  spec revision 37, the gating records adopt it at the next hash move), D9 the pilot's subjects (`M11.C.16` passing,
  `M11.C.42` an accepted finding in group A), D10 the scope (directional criteria; the constants each mechanism reads,
  half to double, log-uniform, drawn jointly; S = 8 for the pilot), D11 the sweep record and `docs/version_2_status.md`
  with its sweep column, D12 `config/sweep/sweep.md`, D13 none of `M17`'s drafted requirements until the pilot reports;
  thirteen criteria S2.1–S2.13 with their tests; the order of work ending in the end-of-stage milestone review; no cost
  estimated before the pilot's measurement. Committed alone, before the build, so the design can be read apart from it.
  Written overnight on the branch `stage-2-plan-pilot` under the owner's instruction of 2026-10-22.

### The entries of 2026-10-19 and 2026-10-20 archived (2026-10-23)

- The hook entry below took this file past its 80,000-byte limit (`tests/test_doc_size.py`, pushed red at `ad361d7` and
  fixed here), so the four entries of 2026-10-19 and 2026-10-20 (version 2 stage 1, the `/csdp` sweep, the version 1
  tag, the document-claim tests) moved to `docs/CHANGELOG_archive.md` verbatim, 11,246 bytes, checked byte for byte;
  the pointers in this file, `CLAUDE.md`, the checklist and the reading list now say entries before 2026-10-21.

### Cloud sessions: a SessionStart hook makes the suite runnable (2026-10-23)

- `.claude/hooks/session-start.sh`, run by `.claude/settings.json` at session start, does in a Claude Code cloud
  session what CI does on a blank machine: installs `requirements.txt` and fetches the tag `version-1` (a shallow
  clone leaves it out, and `tests/test_version_1_status.py` reads it). It exits at once outside a cloud session
  (`CLAUDE_CODE_REMOTE` unset). Written by a cloud session the owner started on 2026-10-23 (its branch
  `claude/confident-maxwell-vr5zol`, commit `6b74ba4`), cherry-picked onto `main`; that branch's own G2 proposal is
  superseded by `docs/PROPOSAL — RELEASE BY TIE (G2).md`, built at revision 36, and was not merged.

### Revision 36's records regenerated (2026-10-23)

- Every record regenerated once on the engine hash `66dcbb1e8ba5` (from `2842f0e8fea8`), on this Mac with 10 workers:
  the Phase C ensemble record 3 min 10 s, the Phase D part I record 12 min 7 s, part II 1 h 31 min, the Phase C
  mutation record 22 min 34 s, the Phase D mutation record 12 min 13 s, the sweep record 5 min 42 s, the diagnostic
  record 7 min 33 s, the level-occupancy record 35 s, the seed-7 trace, chart and family-diagram export and
  `docs/spec_coverage.md` under a minute together, and the after section of `docs/review/group_g_readout.md`
  1 min 34 s; 2 h 36 min of runs in all. `docs/version_2_revision_36_changes.md` lists every field that changed,
  generated by `tools/record_changes.py revision-36`, which now reads both sides from git refs (stage 1's regeneration
  `7aa9fc5` against the working tree; the stage 1 page is the tag against `7aa9fc5`, frozen, T-028 closed).
- **Verdicts that changed**, each a new result under revision 36's rules, reported beside the earlier verdict and
  never in its place; all composite in `M11.5` (`M11.C.7` mixed), so none is a premise passing. Phase C: `M11.C.44`
  FAIL → PASS (`outside_inside_ratio` +0.103 ± 0.092, p 0.0298, from +0.00101 ± 0.096). Phase D part I: `M11.C.17`
  PASS → FAIL (`coached_rise` +0.000272 ± 0.00042, p 0.205, from +0.000888 ± 0.0013, p 0.0148);
  `M11.C.7[joint over one spouse]` PASS → FAIL (`deviation_fall` +0.0309 ± 0.056, p 0.177, 100 seeds, from
  +0.0429 ± 0.052, p 0.0497, 150). Part II: `M11.C.10[family of origin]` PASS → FAIL (`cut_off_tie_weeks` −31.8 ± 130,
  p 0.674, from +238 ± 150, p 0.00107); `M11.C.2[twelve-person family]` PASS → FAIL (`symptom_load` +0.947 ± 1.6,
  p 0.118, from +3.42 ± 1.8, p 0.000236). New: `M11.C.48` PASS on both families (`negative_run` +0.43 ± 0.15,
  p 5.2e-07 on the Phase B family; +0.476 ± 0.06, p 6.9e-17 on the twelve-person family; 100 seeds each).
  `M11.C.44` and `M11.C.17` moved the same way at revision 29's build and back by revision 32
  (`docs/implementation_plan_phase_d.md` §4.8); both passes at stage 1 were marginal (p 0.0148, p 0.0497 for
  `M11.C.7`'s). Effects moved with the verdict held: 13 in Phase C, 8 in part I, 3 in part II. The four PASS → FAIL
  verdicts are T-122, for the owner's review.
- **Mutation.** Phase C: `steepness-level-independent` now turns `M11.C.41[heavy: lower level]` red and no longer
  `[light: lower level]`; `M11.C.44`, passing, gains `one-sided-deviation` (red) and `deviation-inverted`
  (**survived**, as at revision 29). Phase D: `triangle-help-ignored` now turns `M11.C.46` red (was survived); **three
  named mutants survive that the spec says must turn their criterion red** — `cutoff-blind-to-learned-value` on
  `M11.C.48` and `M11.C.48[phase_d]` (`negative_run` +0.149 ± 0.092 and +0.181 ± 0.041 under it), so A11 of the G1
  proposal is not met (and see the re-sweep below, T-123); `death-bound-anxiety-discarded` on `M11.C.15` (`death_rise` +1.45 ± 0.69, from +0.128 ± 0.7
  red); `removed-member-hold-released-while-away` on `M11.C.11` (`disturbance` +1.30 ± 1.18, from −0.593 ± 1.37 red)
  — T-119 to T-121 (`docs/todo/2026-10-23_revision_36_regeneration.md`). Five mutation rows left the Phase D record
  with their criteria's passes (`landed-contact-changes-nothing` on `M11.C.17` and `M11.C.7[joint over one spouse]`,
  `witnessed-cutoff-learning-removed` on `M11.C.10[family of origin]`, `witness-path-removed` and
  `weakness-focus-and-steering-removed` on `M11.C.2[twelve-person family]`): a mutant runs only against a passing
  criterion.
- **Sweep:** `credit_horizon 2 · M11.C.42` PASS → FAIL; `policy_temperature 2.0 · M11.C.44` FAIL → PASS; 28 effects
  moved. Level occupancy: all four rows' values changed.
- **The after readout** (`docs/review/group_g_readout.md`, 20 seeds a family): cutoffs over the run 793 → 457 on the
  Phase B family and 11,615 → 4,820 on the twelve-person family; the largest stock at a cut 975.9 → 330.7 and
  1,583 → 658.9 (the cap, 1583, was fixed from the before measurement); mean excess anxiety 23.9 → 22.4 and
  15.1 → 13.0; symptom weeks 0.36 → 0.30 and 0.25 → 0.17. The run before the first cutoff exceeds the matched uncut
  run on both families (2.06 against 1.58; 1.33 against 0.87).
- Documents: `docs/version_2_revision_36_changes.md` new; `docs/version_2_stage_1_changes.md` regenerated from git
  refs, frozen; the G1 proposal's status line (A11 partial); `docs/theory/_STATUS.md`; `docs/REVIEWER_READING_LIST.md`;
  `docs/SPEC_REVIEW_CHECKLIST.md`; `docs/spec_coverage.md`; the TODO record above; T-028 closed; `CLAUDE.md`'s suite
  count. No rule changed: this entry reports the measurement.
- **The regeneration's re-sweep** (`learning-qa` on sonnet, 11 min 25 s, 116,658 tokens, 45 tool uses) found: (1)
  `M11.C.48`'s baseline gave a first cutoff with no matched uncut tie (no other tie with a contact in the window) a run
  of 0 and kept the cutoff's own run in the treatment arm; measured on the regenerated runs, 139 of 700 first cutoffs on the Phase B family (100 seeds) and 284 of 749 on the twelve-person family (20 seeds); with those left out of both arms the difference is −0.28 ± 0.15 and −0.21 ± 0.18 (2 se) against +0.43 ± 0.16 and +0.44 ± 0.10 as built, so **the pass of
  `M11.C.48` is not evidence** and A11 is not met; the fix is a readout change that moves the engine hash, so it waits
  for the next move (T-123; `docs/readouts_phase_c.md`, the spec row, the explainer, the G1 proposal's status line and
  `docs/theory/_STATUS.md` say so). (2) `docs/phase_c_completion_report.md` §11 had not been regenerated with the
  records: regenerated (`tools/c16_report.py`). (3) `test_m11c10_family_of_origin_variant_is_proved_red_by_removing_m4d6f`
  read a mutation row that left with the variant's pass: it now pins the missing row to the variant's FAIL in the part II
  record (T-122). (4) Three coverage notes stated verdicts the regeneration changed (`M1.C.2` on `M11.C.44`, `M2.A.2` on
  `M11.C.2[twelve-person family]`, `M7.E.3` on `M11.C.10[family of origin]`), and `M6.2` was done through `M11.C.11`, now
  partial: corrected in `tools/spec_coverage_overrides.json`; `test_m141_accepted_finding_shows_its_disposition` now
  allows an accepted finding that passes under a later rule when its note shows the pass. (5) `test_negative_runs.py`'s
  late-contact case tested nothing (both candidate ties were declared): rewritten, and a test of two first cutoffs on one
  seed added; the readout module's `Tests:` line lists all six. (6) A comment in `tools/phase_d_mutation_record.py` gave
  the pre-regeneration numbers for `removed-member-hold-released-while-away`: updated. (7) `CLAUDE.md` said 49
  `ensemble`-marked tests; 51 with `M11.C.48`'s two. The first run of the Phase D mutation record exited at once (the part
  II record's rows were out of order until part II was rerun); every number above is from the one complete run.

### Spec revision 36 — version 2's group G and the Holmes–Rahe event sizes (2026-10-22)

- Built overnight on the branch `claude/recursing-bohr-2393a1` under the owner's instruction of 2026-10-22 ("proceed
  based on your recommendations"); every owner-level choice is decided as recommended, tagged pending the owner's
  review, and listed in the revision 36 report. Not merged into `main`.
- **G1** (`docs/PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md`, `M1.D.2b`, `M1.D.2c`, `M4.G.1`, `M5.B.3`, `M6.4`):
  a cut-off tie's held anxiety weighs on both members, is capped at 1583 on entering the cut, decays to a tenth of
  its peak over five years, is renewed to the peak by a negative contact and loses 80% on a positive one, which alone
  reopens the tie; distant and conflicted ties are repaired by thirds of a reference level; `CONFLICT` binds at 0.15;
  `REDUCE_CUTOFF` moves no stock to a third party. **A6, the third party's cost, is not shown**: its two-arm test is a
  strict expected failure and no cost was written.
- **G2** (`docs/PROPOSAL — RELEASE BY TIE (G2).md`, `M7.D.2a`): a symptom's release and spill are split by bond energy
  among the bearer's interactive ties to living family members; the coach takes none.
- **Event sizes** (`docs/PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md`, `M1.F.6a`, `M6.3g`, `M7.C.1l`,
  `M7.D.1a`; `config/bowen/event_sizes.md`): every stressor the engine raises is sized from the Holmes–Rahe ratios at
  6.9 units per LCU and the tie's reach; grief by the tie; the additions' loads for 12 weeks; a symptom onset loads
  the bearer and the family. Q3 of the proposal stands as decided and cites no source.
- **`M11.C.48`**, a composite in two variants: the run of negative contacts standing at a tie's first cutoff against
  the mean run standing in that week on the uncut ties (`src/bowen/readouts/negative_runs.py`; the readouts document's
  "Negative runs"); mutant `cutoff-blind-to-learned-value`. `docs/review/group_g_readout.md` measures 20 seeds before
  and after.
- Nine constants added to `config/bowen/constants.md`, each `[I]` with a `constants_changes.md` row; `grief_scale`
  retired. Illness loads travel on the `ENDOGENOUS` channel and are left out of the evaluation readout's stress term.
- Two reviews before the hash moved (learning-qa, 10 min 42 s; correctness, 11 min 52 s) found eleven and ten items;
  every code item is fixed in the same commit (`LAUNCH_LOAD` reaching the parents, the reference level on reopening,
  `FAMILY_ILLNESS` to members only, the `M11.C.48` baseline as the matched-week mean, a declared stock above the cap
  refused, the mutant's empty-case guard, the spec's `lcu_scale` anchor).
- Documents: the spec (front matter revision 36; `docs/spec_revision_log.md` has the revision), `docs/model_explainer.md`
  §11.5, `docs/readouts_phase_c.md`, the three proposals' status lines, `docs/theory/_STATUS.md`,
  `docs/REVIEWER_READING_LIST.md`, `docs/SPEC_REVIEW_CHECKLIST.md`, T-001 advanced and T-013 closed; the generated
  indexes and views regenerated.
- The engine hash moved once, from `2842f0e8fea8`; the records were regenerated once after it (the next entry).

### Revision 35's dispositions approved by the owner (2026-10-22)

- The owner read `docs/spec_revision_35_dispositions.md` and approved it in chat ("I have reviewed and approve the
  Spec_Revision_35 document"): all 262 rows stand, none reversed. Its status line, the reading list, the checklist, the
  plan's status, `_STATUS.md`, CLAUDE.md's spec row and the history list's intro (`tools/spec_rules_view.py`) now say
  so; TODO item T-005 is ticked. No spec text changed.

### `TODO.md` an index over records; a changelog index (2026-10-22)

- **What.** `TODO.md` is now generated: one line per open item — ID, date raised, title, link to its record — under
  **Now** (blocks version 2 stage 2 or the next commit) or **Later** (waits for a regeneration, a revision or an owner
  decision). The text of the 118 open items moved, verbatim, into 16 records under `docs/todo/`, one file per `TODO.md`
  section (named by the section's date, or the date its heading was committed where it had none), each item keeping
  its box and gaining a stable ID (`T-001` to `T-118`, in `TODO.md`'s order) and one tag. Five are ⟨now⟩: T-001 to
  T-004 (the "Blocking the next phase" section) and T-013 ("before group G's build starts"); 113 are ⟨later⟩. T-028
  (freeze the stage 1 changes page before any regeneration) was tagged ⟨later⟩ unsure, for the owner. Ticking an item
  in its record and running `python3 tools/todo_index.py` moves its line to `docs/TODO_archive.md`'s generated region;
  the items ticked before today stay in that file's static sections and have no records. `CHANGELOG.md` opens with a
  generated index of its `### ` entries (date, title), between marker comments; the entries are untouched.
- **The tool and the test.** `tools/todo_index.py` writes the three generated parts and never a record; it refuses an
  item without an ID or without exactly one of the two tags, and a duplicate ID. `tests/test_todo_index.py` checks
  the three are current, the records well formed, every open item indexed once, and that a run of the tool changes
  no record byte; each rule is also tried on a fixture that breaks it. `tests/test_doc_size.py`'s `TODO.md` limit is
  25 KB, the one-line form's (15 KB today); `CHANGELOG.md`'s stays 80 KB.
- **Verified.** Each record's items, with the ID and tag stripped and the headings put back a level, equal the
  corresponding `TODO.md` section at `eb7205f` line for line (script, before the commit).
- **Documents.** `CLAUDE.md`'s read-first paragraph, `docs/REVIEWER_READING_LIST.md`'s `TODO.md` row,
  `docs/SPEC_REVIEW_CHECKLIST.md` (the `docs/todo/` directory and the archive's new shape),
  `docs/milestone_review_brief.md` (L2 reads the index and the records it needs, not `TODO.md` whole),
  `docs/REVIEW_version_1_guide.md` and `docs/version_2_plan.md` (section pointers now name the record),
  `docs/TODO_archive.md`'s intro, and `tests/asr_quotations.json`'s allowlist entry for the quotation now in
  `docs/todo/2026-08-23_sources.md`. The owner's "Token budget" section for `CLAUDE.md` is drafted for approval, not
  added. `docs/spec_coverage.md` regenerated where the unmapped-test count changed.
- **Token budget, the instruction.** `CLAUDE.md` gained the section "Token budget — how to read this repo (owner rule,
  2026-10-22)": read by index, grep then a line range, module views over the spec, no whole-file reads of the big
  documents, wide reads to a subagent, generated summaries only, `tests/test_doc_size.py`'s limits as the enforcement,
  one task per session. Added here after the index session drafted it (owner, 2026-10-22: "make sure to set up
  instructions that this token efficient approach must be adhered to").

### The closed and old parts of `TODO.md` and `CHANGELOG.md` archived (2026-10-22)

- **What.** `TODO.md` (134 KB) and `CHANGELOG.md` (152 KB) were read in part by most sessions and grew without bound,
  the two files the model-ergonomics plan left growing. Every ticked item of `TODO.md` (112, with the section "Known
  defects in the frozen grid engine", closed whole at stage 1, moved whole) is now in `docs/TODO_archive.md`,
  verbatim, grouped by the month it was closed (104 in 2026-10, 3 in 2026-09, 1 in 2026-08: the latest date the item
  carries, or the commit that ticked it for the eleven that carry none); `TODO.md` holds the 118 open items in their
  order (82 KB). The `CHANGELOG.md` entries dated before 2026-10-19, the day version 2 began at the tag `version-1`,
  are in `docs/CHANGELOG_archive.md`, verbatim; that day's entries and later stay (60 KB), with a pointer at the
  bottom. Both moves were checked by script against `git show HEAD:TODO.md` and `HEAD:CHANGELOG.md`: nothing moved is
  reworded; two ticked items nested under open items lost their two-space indent, and the moved section's heading
  went one level down under its month.
- **The test.** `tests/test_doc_size.py`: `TODO.md` holds no ticked item closed more than 30 days ago (an undated
  ticked item fails too), each live file is under its limit (`TODO_LIMIT_BYTES` 100 KB, `CHANGELOG_LIMIT_BYTES`
  80 KB; 60 KB each was the suggested start, below the 82 KB of open items and the 62 KB of version 2 entries, so
  each limit starts above its file and is lowered when the file shrinks, never raised to pass), and each live file
  names its archive. The scan is checked on fixtures: an old ticked item, a recent one, an undated one, a nested one,
  an old open item, the latest date in the item winning, and the block ending at the next item, heading or blank line.
- **Documents.** The checklist (both archives under "Not for review", with the label "History, moved" added to
  `tests/doc_claims.json`'s unclassified labels), the reading list's `TODO.md`/`CHANGELOG.md` row, the milestone
  brief's step 6, `CLAUDE.md` (the archives, and the suite count 1039 → 1050), and
  `tests/test_version_1_outputs.py`'s allowed set (the archive carries the entry that added `M11.F.10`). Nineteen
  pointers that named a moved item by title, quotation or line number now name `docs/TODO_archive.md`:
  `plan_spec_coverage_phase_d.md` (3, one of them the stale "line 561"), `spec_revision_35_dispositions.md` (2),
  `spec_revision_log.md`, `plan_model_ergonomics.md`, `implementation_plan_version_2.md` (2),
  `REVIEW_version_1_guide.md` (4), `implementation_plan_phase_d.md` (3), `DECISIONS — CUTOFF SEMANTICS.md`,
  `PROPOSAL — RUN VISUALIZATION MAPPING.md` and `phase_d_status_2026-10-13.md`. Generic "recorded in `TODO.md`"
  pointers in dated plans and records are left as written. `PROPOSAL — ASR QUOTATIONS (SP-1).md`'s `TODO.md` line
  numbers (1046, 1098, 1035–1053) were already stale before this change and are left. Not touched: `version_1_status.md`
  and `version_1_dispositions.md` (frozen at the tag) and `spec_coverage.md` (generated; regenerated here only because its count of tests named for no spec ID
  rose from 219 to 229 with the new file). No rule changed;
  `engine_hash()` unchanged (`91f9a817`); no record rerun.

### Model ergonomics follow-ups: the ledger index reaches the extraction files; the missed history phrases (2026-10-22)

- **What.** Three of the four open `TODO.md` items the model-ergonomics work left, on the owner's instruction
  (2026-10-22: "tick 2. do 1 and 4"). Wording and tooling only; no rule changed, `engine_hash()` unchanged (`91f9a817`),
  no ensemble, mutation or sweep record rerun. Spec revision stays 35.
- **The ledger index reaches the extraction files** (plan §11 trial: `FE03.6` was unreachable). `tools/spec_index.py`
  indexes, besides `_LEDGER.md`'s entries, every `## … <ID> —` finding heading of the files that
  `tools/spec_index_rules.json` `extraction_sources` names (`docs/theory/family_evaluation/fe*.md`, `kerr_book/ks*.md`,
  `kb/kb*.md`, `_LECTURES_1979.md`): 875 records, 670 of them new, each with a `file` field; `chapter` is the file's
  title and `status` is empty for them. An ID defined twice, or an ID-like heading the `entry_id` pattern does not
  read, fails the run. The spec records' `ledger_mentions` follow (`M4.C.8` now lists `FE03.6`). Two tests added
  (`test_b4_every_extraction_finding_is_indexed_once_at_its_line`, `…_extraction_heading_the_pattern_does_not_read_is_refused`);
  two adapted to the `file` field. Documents: CLAUDE.md, README, the reading list, the checklist, the plan's B4,
  the milestone brief, `_STATUS.md`.
- **The missed history phrases.** Eight phrases added to `history_phrases` (`earlier form`, `The text it replaces`,
  `is replaced`, `The form approved at`, `It replaces`, `As first built`, `revision N removed`, `rev-N form`). The 44
  sentences they match, with the `*Revision N*` prefixes outside requirement bodies (`M2.A`'s table notes, `M8`'s
  preamble, `M11.E`'s `M11.C.9` row, `M13`'s phase table row B; the tool scans requirement blocks only, so their row
  ID is the section in parentheses), `M11.C.14`'s withdrawn two-sided bound and `M4.C.3a`'s pre-revision-24 floor,
  are disposed as rows 219–262 of `docs/spec_revision_35_dispositions.md` (19 moved, 25 rewritten), each bullet in
  `docs/spec_revision_log.md` under its revision, verbatim by construction (sliced from the spec before the edit).
  §0.3's count: 1043 → 1042 bolded MUST / MUST NOT, the withdrawn bound having left for the log. `history_in_body.md`
  lists 30 IDs, all "stays" rows. `test_e2` walks the 262 rows and reads a section-suffixed bullet ID.
- **Part D's commit history** stays as it is (owner, 2026-10-22); the TODO item is ticked.
- Suite: 1039 tests.

### Model ergonomics, Part E — the revision history out of the spec, revision 35 (2026-10-22)

- **What.** `docs/plan_model_ergonomics.md` Part E, on the owner's answer to its §8 Q2 (2026-10-22: "yes, as revision
  35"). Wording only; no rule changed, `engine_hash()` unchanged (`91f9a817`), no ensemble, mutation or sweep record
  rerun. The seed-7 trace, chart and fam-diag export were regenerated because their header carries the revision string
  (one line each changed). Spec revision 35.
- **E1.** New `docs/spec_revision_log.md`: the sections `### Revision 34` to `### Revision 5` and the front matter's
  status text (its record of revisions 10 to 34) moved there as they stood. The spec's `status:` now reads `APPROVED`
  with a pointer; a closing `## Revision history` section points at the log. §0.3's token count is recomputed (1043
  bolded MUST / MUST NOT; the five in the moved sections went with them). Five entries of `tests/asr_quotations.json`
  that allowed spans in the moved sections now name the log; `test_r31a12` reads the old status line in the log; the
  two tests that pinned the string `2.0, revision 34` now read the front matter; `test_d5b_the_scan_is_not_vacuous` no
  longer asserts a third status source, since the spec's status line was it.
- **E2.** The 213 sentences in 129 requirement bodies that `docs/spec_rules/history_in_body.md` listed, as 218 rows
  (five sentences held two narratives and got two rows), were disposed one by one: 35 stay (rule text, a recorded
  result or a cross-reference that matches a phrase; the row gives the reason), 32 moved to the log, 151 rewritten as
  the current rule only
  (a `*Revision N:*` prefix dropped, a title's "corrected at revision 7" clause dropped, the sentence as it stood
  recorded in the log under its revision). ID count, order and the bolded MUST / SHOULD count are unchanged. The log
  gained 183 bullets, in new `### Revision 13` … `### Revision 33` sections and sub-lists under the existing ones. The
  record is `docs/spec_revision_35_dispositions.md`, one row per sentence, applied as Claude's dispositions for the
  owner's review (the plan's E2 said one at a time with the owner; the owner chose the bulk option in §8 Q2). What the
  view's `history_in_body.md` lists now (30 IDs, 35 sentences) is what the pass left in place, each sentence a "stays"
  row, and its test matches them sentence by sentence.
- **Not covered by the phrase list** (noticed in the pass, recorded in `TODO.md`): narrative of other shapes still in
  bodies ("The earlier form …", "The text it replaces said …", a lower-case "*revision 11 removed …*"), `*Revision N*`
  prefixes in `M2.A`'s and `M8`'s preambles and two table rows, and `M11.C.14`'s withdrawn two-sided MUST NOT still
  standing before its one-sided restatement.
- **Documents.** CLAUDE.md's spec row, README's Status, `_STATUS.md`'s STATE line and a dated note, the checklist (the
  log and the dispositions record added; the stray clipping's entry removed with the file), the reading list (two rows),
  the plan's status line.
- **A slip in the Part D commits.** The log's first form (the moved sections, 493 lines) was staged by accident with
  Part D's first commit (`11c36bb`, the engine docstrings) and no checklist entry went with it, so
  `test_every_document_is_in_the_checklist` was red at the nine Part D commits (`11c36bb` to `fea842a`; confirmed
  by running the test in an extract of `11c36bb`), and `test_m11d8_spec_references_resolve` was red at the first eight
  (`11c36bb` to `6575009`: the code had 933 and then 938 references against the pinned 808, corrected at `fea842a`).
  `11c36bb`'s message says "docstrings only". The plan's D3 rule, the suite green after each commit, was not met for
  Part D. Both tests are green again at this commit. The branch is not pushed; whether to rewrite those commits is the
  owner's call (`TODO.md`).
- **Review fixes (2026-10-22, one learning-qa pass and one correctness pass over Parts D and E).** Two log bullets
  did not hold the sentence as it stood (`M1.A.4k`'s owner list of nodal events was new wording; `M11.C.14`'s Bowen
  quotation lost "any and" and its `ch04.md` citation): both restored verbatim from the spec at `716e5fe`. `M11.C.6`'s
  rewritten arm had dropped its post-hoc label, which `M11`'s reporting rule requires: restored. The spec's closing
  note ("Written against `model_explainer.md` v1.2 …") had gone to the log with the sections; it is the spec's and is
  back at its end. `test_c4` now matches each listed sentence to its own "stays" row and pins the counts to the
  record (it had checked IDs, which 14 of the 30 share with moved or rewritten rows); a new test walks all 218 rows against the
  spec and the log, proved failing under three mutations. `spec_references` (the `M11.D.9` count) now reads a `Spec:`
  block's continuation lines: 969 citations, not 938 (31 on wrapped lines were uncounted; all resolved). The two
  record tests compare the revision with the front matter read in the test, not with the loader's own value.
  `counterfeit.py`'s docstring keeps the equal-magnitude premise its conclusion needs. The "rule text" and "218
  sentences" claims corrected (35 kept rows are rule text, recorded results and cross-references; 213 sentences in
  218 rows); the reading list's spec line count; the log's intro says why revision 34's text calls revisions 13 to
  33 "not here". Cost: two passes, about 19 minutes wall time and 357k tokens.
- **Re-sweep of the fix commit** (`1328be8`; sonnet, narrow brief, about 5 minutes and 96k tokens): five items, none
  a blocker. The dispositions test looked for a moved sentence anywhere in the log, so deleting an ID's own bullet
  passed when another ID's bullet held the same words (row 95, `M7.B.1a`); it now looks under the row's own ID, proved
  by that mutation. A row with no uncut piece would have matched anything: asserted away. "16 of the 30" above was
  wrong (14 of the 30 kept IDs share an ID with a moved or rewritten row). `counterfeit.py`'s docstring quotes
  `FE03.1`'s "two sides of the same coin" rather than attributing "equal magnitudes" to it. A blank line in the log.
- **Suite.** 1036 collected after Part D (the pinned B3 test removed), 1037 after the review fixes (the dispositions
  test); `engine_hash()` `91f9a817` throughout.

### Model ergonomics, Part D — the code docstrings thinned (2026-10-22)

- **What.** `docs/plan_model_ergonomics.md` Part D, on the owner's answer to its §8 Q1 (2026-10-22, "thin them") and
  Q4 ("fix B3 now, in Part D"). Docstrings only, in `src/bowen/`: no code, comment or blank line changed, checked per
  file by comparing the syntax tree with its docstrings blanked against `HEAD`'s. `engine_hash()` is unchanged
  (`91f9a817`). One commit per package, the docstring bytes before and after in each commit message; in all,
  214,733 bytes of docstring became 203,303 (−11,430, 5%). The saving is small because D1 keeps what carries the
  contract: the `Purpose:`/`Spec:`/`Tests:` block, every formula, every grade statement and every sentence stating
  the current rule. What went is revision and approval narration ("*Revision 25 (approved 2026-10-16)*", "(revision
  18)" in a title or a `Spec:` line, "Phase C step 3", "the owner decided …", "measured before it …"), and rationale
  or quotations the cited ID's spec block holds. Three stale statements were corrected to what the code does
  (`slow_tick.run_slow_tick`'s life-stage note, `objects.Sex`'s "no mechanism reads it", `ensemble/criteria.py`'s claim
  that `M11.C.7` and `M11.C.14` are not built).
- **B3 closed.** The fifteen `Spec:` lines that cited a section heading now cite the numbered requirements the module
  or class implements; `#M13` was dropped where no requirement applies and the phase fact kept as prose.
  `test_b2c_every_id_cited_in_src_is_defined_by_the_spec` is no longer xfail; the pinned list and `test_b3_…` are gone.
  Long `Spec:` lines (12 or more IDs, D4) and the adjacent issues the pass found are in `TODO.md`, not fixed.
- **Moved here from docstrings** (D2: passages neither the spec nor this file held):
  - `engine/appraise.py` (`M4.C.8a`, `M1.A.9a`): the objectivity that gates attention's amplification was changed at
    Phase C step 3 to 1 − the maximum of the two outside-ness axes; step 2 had used 1 − their mean.
  - `engine/life_course.py` (`M7.C.1k`, 2026-10-13): a life event carries no sender and reaches its targets directly,
    because the 2026-10-13 ensemble crashed when `FIND_WORK` named its taker as sender and a sibling with no tie to
    them as a target.
  - `engine/life_course.py` (`M7.B.1a`, `M7.E.1f`, review 2026-10-13): three rules were fixed at that review — a dead
    parent's frozen values and the history that stopped at their death are not read as this year's; an orphan keeps
    the values they have and develops by their own functioning alone; an inactive holder, and an external agent, take
    no part in the life course.
  - `engine/activation.py` (`M7.C.1f`, `M2.A.0k`, revision 28): the activation component's `version` stayed "1" when
    acting age and inactivity were added, because a run that tracks no ages is unchanged.
  - `engine/recompute.py` (`M1.C.3`, revision 11, owner decision 2026-10-06): Phase B decided whether a triangle was
    active by a `tension_activation_threshold`; Phase C step 1 replaced it with the readout of recent `TRIANGLE` acts
    within `triangle_activity_window`, and no calm-system threshold remains in step 6.
  - `engine/visibility.py` (`M1.F.4`, `M3.D.4b`; the Phase B plan's decision D7): per-hop fidelity is the declared
    constant `per_hop_fidelity ** len(route)`, not a draw; the per-hop draw class `M3.D.4b` lists is declared in
    `draws.py` and has never been drawn.
  - `engine/moves.py` (`M6.I.4`, revision 27; review 2026-10-12): without the functioning-balance condition,
    pseudo-self relaxed between people who never exchanged (2–6% of what moved), and a starting gap would have relaxed
    from week 0; the condition was added at that review.
  - `ensemble/runner.py` (`M17.A.1`, decided 2026-10-08, `docs/phase_c_completion_report.md` §10): the baseline arm
    alone is the wrong ruler when the arms differ in spread, and a ruler must not depend on which arm is called
    baseline; the differences' own sd would make the rule a fixed seed count, since half-width over its own sd is
    1.96/√n. So the half-width is compared with `ensemble_precision` times the pooled seed-to-seed sd of the two arms,
    √((sd_baseline² + sd_treatment²) / 2).
  - `ensemble/criteria.py`, `reactive_over_chance` (`M11.C.41`, restated 2026-10-09, Q2 of `docs/DECISIONS — PHASE C
    FAILING.md`): the declared readout, reactive moves over all moves, conflicts with `M4.D.3a` — a lower level closes
    layers 1 and 2, which hold five of the seven reactive acts, so the share fell at a lower level even as acute
    anxiety rose. The first restatement (reactive selected over reactive offered) scored 1/N for a chooser at random,
    so it rose whenever the legal set shrank, and was replaced before it was relied on.
  - `ensemble/criteria_d2.py`, `_cut` (`M11.C.10`, measured 2026-10-14): a fresh declared cutoff fades out in about 10
    weeks, before a 13-week renewal, which is why the treatment arm renews it by a scripted reactive `CUTOFF` every
    `renew_every` weeks.
  - `ensemble/criteria_d.py`, `arm_c34` (`M11.C.34`, review 2026-10-12): the other members' automatic acts toward C1
    are read less the same count in the same run with no onset, because counting all their acts measured the
    family's level, not the reaction to the symptom.

### Model ergonomics, Parts A–C (2026-10-22)

- **What.** `docs/plan_model_ergonomics.md` (Parts A, B and C; D and E wait on the owner's answers to its §8).
  Everything new is generated from the spec, the ledger, the code and the tests by a tool and checked current by a
  test; nothing is hand-written. `engine_hash()` is unchanged (`91f9a817`) and no record was rerun.
- **A2, the suite count's history.** CLAUDE.md's "A red suite is never normal" bullet now gives the current count,
  the date and the command. Its dated history of counts, moved here as it stood: measured 2026-10-22 on this Mac after SP-1's six tests against new quotations of the ASR sources were added, `tests/test_asr_quotations.py`, and five for the fam-diag export keeping what the owner saved from the app; 1003 on 2026-10-20 after a test that compiles every tracked file on the oldest Python in CI's matrix was added, `tests/test_python_versions.py`; 1002 earlier that day after the coverage page's Part B added five tests, `docs/plan_spec_coverage_phase_d.md` §7; 997 earlier that day after its Part A and review added nine; 988 earlier that day after the document-claim tests were merged, adding 60, `docs/plan_doc_claim_tests.md`; 928 earlier that day after the fam-diag export's currency test was added; 927 earlier that day after the cold sweep's fixes added three tests; 924 earlier that day after the stage 1 `/csdp` sweep's fixes added seven tests; 917 on 2026-10-19 after version 2 stage 1's build, review fixes and one regeneration; 916 earlier that day after the `/csdp` sweep's fixes added three run-chart tests; 913 earlier that day after version 1's status page was frozen at the tag `version-1`, its eleven status tests now run at the tag by three; 921 on 2026-10-18 after the run chart was added beside the Phase C trace; 915 earlier that day after spec revision 32 with the owner's review and the nodal-event spike; 909 on 2026-10-17 after revision 32's first part; 886 earlier that day after `M11.C.21`'s reclassification and the capacity rule's no-span fix; 884 on 2026-10-16 after revision 31 part B; 868 earlier that day; 854 on 2026-10-15; 850 passed in CI run 38763412772 on Python 3.11, 3.12 and 3.13, before the Phase D mutation record's 4.
- **B1, B4: the indexes.** `tools/spec_index.py` writes `docs/spec_index.jsonl` (468 KB, one JSON line per spec
  ID: module, heading, lines, first sentence, withdrawn, grade tags, sources, ledger mentions, code, tests, status and
  note) and `docs/theory/ledger_index.jsonl` (51 KB, one line per ledger entry: title, status as written, labels,
  chapter, lines, the spec IDs it cites). The citation forms are in `tools/spec_index_rules.json`. The first sentence
  is generated, cut at 300 characters, never hand-written; where the first sentence is only an italic title the next
  comes with it. `tests/bowen/test_spec_index.py` (16 tests) checks both
  current, every ID once in order, the line spans, the code names, the status fields against `spec_coverage.py`, the
  ledger mentions against the ledger text, and each reader on a fixture.
- **B3.** Fifteen `Spec:` lines in `src/` cite a section heading of the spec (`M1.A`, `M13`), not a numbered
  requirement; listed in `TODO.md` for the owner's Q4, and the test that every cited ID is defined is xfail on exactly
  that list. Not fixed.
- **Documents.** CLAUDE.md's "Read first" paragraph, README's "Where to start", the reading list's §3, the milestone
  review brief's §3 step 6 and the checklist point at the indexes. `docs/spec_coverage.md` regenerated for the new
  tests. Suite: 1029 collected after Part B, 1036 after Part C, 1037 after the review fixes.
- **C1–C5: the current-rules view.** `tools/spec_rules_view.py` writes `docs/spec_rules/M00.md` to `M17.md`
  (one per `## M` module; `M11.F` and `M11.G` in `M11.md`), each opening with the non-normative line, plus a
  `README.md` with the sizes and `history_in_body.md`. Removed, mechanically and nothing else: the front matter's
  `status:` line, every `⟦…⟧` tag (551, 543 of them in the rules), every block-quote line (17) and the revision sections from the first
  `### Revision` heading (the spec's line 2117) to the end. No body sentence is removed. The eighteen module files
  are 432 KB (66,358 words) against the spec's 513 KB (78,938 words): 84% by bytes; the saving is the split, since
  a module is 3 to 100 KB. `history_in_body.md` lists 129 IDs whose bodies carry a revision narrative, by the
  phrases in `tools/spec_index_rules.json` — the input to Part E; the list over-includes by design (`previously`
  and `was changed` are rule text in a few bodies), and the owner's pass leaves those. `tests/bowen/test_spec_rules_view.py` (7 tests)
  checks the view current, every ID in exactly one file, every non-quote line of the spec present in order, and
  each removal and each phrase on a fixture. `docs/spec_rules/` is exempt from the ASR quotation test
  (`tests/asr_quotations.json`): it is the spec's own text, and the spec is scanned.
- **Review fixes, one commit after the two passes (learning-qa, correctness).** The ledger reader read a suffixed
  finding ID (`L10.11a`) as part of the entry before it: fixed, and a heading that looks like an entry the pattern
  does not read now stops the tool (P2); the ledger index has 205 entries, not 204. `withdrawn` is read from the
  defining line only, so `M11.4`, whose exceptions table says another ID is withdrawn, is no longer marked. A
  horizontal rule ends a block. Chapter-level citations (`Ch03`, `FE03`, `KS23`, `DSI`) are sources too. The history
  cut refuses to run if a module heading or a requirement follows it. The tag count here was a line count. A
  lower-case `at revision N` counts as a history phrase, which took the C4 list from 87 to 129 IDs.
- **A3, the leftover worktrees.** `claude/cranky-snyder-764f33`, `claude/quizzical-burnell-818aa7` and
  `docs-stale-fixes` (under `.claude/worktrees/`) are clean and have no commit that `main` lacks. Not deleted: the
  plan's §8 Q3 puts that to the owner.

### SP-1 built, cut down, with spec revision 34 (2026-10-22)

- **What.** The owner approved the 65 paraphrases in `docs/PROPOSAL — ASR QUOTATIONS (SP-1).md` §4 as written and
  asked for the build with spec revision 34. Partway through, the owner asked why so much of the work was on the
  transcripts and not the code, and cut the build down: the paraphrases, Q8's trace, the frozen-record notes, the
  default-suite test with its rules file, and revision 34's wording fixes now; the local transcript test, the
  extraction-note headers, the `kb` heading retitles and `TODO.md`'s two quotations deferred (`TODO.md`).
- **The paraphrases.** 47 entries in the spec, 8 in the explainer, 9 in the ledger and 1 in
  `agent_model_proposal.html`, each applied by exact match against the approved text. `M11.F.3` is restated from
  *Family Evaluation*'s Epilogue (Q3). `KB01` is credited to Bowen; `KB03` and `KB04` are cited as the interview (Q5).
- **Q8.** The *Family Evaluation* Ch3 passage is a new finding, `family_evaluation/fe03.md` · FE03.21, cited from
  `M4.D`, the spec's revision 9 entry and `model_explainer.md`. The finding says the passage ends a quoted run in the
  chapter whose speaker is not known.
- **Frozen records.** `DECISIONS FOR APPROVAL.md` and `proposal_rev11_emergence.md` carry the note that their ASR
  quotations are not the speakers' words.
- **The test.** `tests/test_asr_quotations.py` fails on a quoted span of four or more words in a paragraph, list item
  or table row that cites a 1979 tape or a Kerr–Bowen interview, outside the exempt files and the allowlist in
  `tests/asr_quotations.json` (16 exempt paths, 47 allowlisted spans, each with its reason or source). It failed on the
  tree before the paraphrases; mutant documents prove it catches a new quotation in markdown and in HTML (curly,
  straight and entity-encoded marks, and a curly single quotation with an apostrophe inside), and one beside a short
  quoted word or a stray mark. Quote marks are paired inside each block; HTML tags are blanked and entities decoded
  first. Its scope differs from the proposal's §5 (block, not paragraph; no bare "interview"; five more citation forms;
  no `kb/kbNN`, which `KBNN` already matches), and a test pins each citation form. Measured on `c893109` it catches 68
  of 87 and flags 13 of 64 kept rows (the proposal's new §9). `docs/theory/_STATUS.md` is scanned, not exempt: its 13
  flagged spans are the owner's words, the project's own wording and written sources, allowlisted. Other tests check
  that every rules entry is live and that the frozen records carry their note. `tests/doc_text.py` gains `blocks`.
- **Spec revision 34.** The SP-1 paraphrases; `M11.4` drops its count of exceptions (MR-10); M8's preamble, `M8.5` and
  `M13.1` say three mechanisms call the live-positions predicate (MR-12, D4); `M2.A.0i` records Dr Halim's declaration
  and Teodor's row drops "awaiting owner review" (MR-13, D6); the status line's closing clause is deleted (MR-15);
  `M11.1`'s scope is the `M11.C` criteria (D3). A revision 34 history entry lists them. Nothing a run computes
  changed, so no record was rerun.
- **Coverage.** `tools/spec_coverage_overrides.json`'s `M5.F.2a` note no longer quotes `KB06`; `docs/spec_coverage.md`
  regenerated.
- **The renders name the revision.** The seed-7 trace, its chart and the fam-diag export print the spec revision, so
  revision 34 made them stale; the trace and chart were regenerated (one line each), and two tests that pin the
  string (`test_log.py`, `test_render.py`) now say revision 34. A rerun of `tools/famdiag_export.py` would have undone
  the layout the owner saved from the app on 2026-10-20 and dropped what the app added (its colours, its settings and
  the owner's "Anxiety Level" category). The owner chose (2026-10-22) to have the tool keep them: a regeneration now
  keeps what the committed export holds for the same id at every depth — positions and sizes, the display name, and
  the app's colours and settings, the keys named in the tool's `APP_OWNED_KEYS` and `APP_ADDED_KEYS` (`keep_saved`).
  It reports each kept value that differs from the drawing, a count of the app's values kept, and each key, entry,
  duplicate id or misshapen value it drops; any other saved key the tool does not write is dropped and reported, so
  that no stale run output is kept. `--from-table` places everything by the layout again and reports each saved
  position it discards. Mutants (positional matching, app keys dropped, the flag ignored in `keep_saved` and in the
  command line, the discard, duplicate and shape reports, the count, a saved value replaced by an equal drawn one)
  each fail one of the five tests (`test_v5_regeneration_keeps_saved_positions`, `test_v5_write_export_keeps_and_reports`,
  `test_v5_keep_saved_reports_malformed_saves`, `test_v5_main_from_table_redraws`,
  `test_v5_app_key_sets_are_the_declared_ones`). `config/viz/famdiag.md`'s layout note says what is kept. The export is
  written with two-space indentation, as the app saves it, so a regeneration's diff shows what changed: regenerated
  from the owner's save, its content differs in the note's revision only; the diff also moves four keys in each of
  the three triangles, which the app writes in another order, and adds a final newline. The run itself did not
  change: the config hash is the same.
- **Found and recorded, not fixed** (`TODO.md`): Q7's six requirements, one decision each; `SP1-E06`'s check note
  claims text the approved version lacks; `SP1-S38` cites K10.9 twice; `SP1-S35` dropped the old note's explanation.
- **Documents:** the spec, the explainer, the ledger, `agent_model_proposal.html`, `fe03.md`, the two frozen records,
  the proposal (status line, §6, §9), `TODO.md`, `CLAUDE.md`, `README.md`, `docs/theory/_STATUS.md`, `config/viz/famdiag.md`,
  `docs/REVIEWER_READING_LIST.md`, `docs/SPEC_REVIEW_CHECKLIST.md` and `docs/spec_coverage.md`.
- Suite: 1014 tests.

### The SP-1 proposal's eight questions answered (2026-10-22)

- **What.** The owner answered Q1–Q8 of `docs/PROPOSAL — ASR QUOTATIONS (SP-1).md`, all as recommended, after a
  walk-through of each. The answers are in the proposal's §6. The 65 paraphrases in §4 are not yet approved, so
  nothing is applied and the build (§7, with spec revision 34) has not started.
- **The answers.** Q1: the KB11 quotation in `M10.C.2` and L16.3 is part of SP-1; DECISIONS A1 stays frozen.
  Q2: the frozen-record note as worded, `proposal_rev11_emergence.md` frozen the same way, the spec's revision 8
  and 9 history entries paraphrased. Q3: `M11.F.3` restated from the Epilogue (FE11.2), a requirement change the
  owner approved. Q4: the two Tape 1 passages paraphrased. Q5: Bowen named for `KB01`, the interview cited for
  `KB03` and `KB04`. Q6: form C, a citation test in the default suite and a local transcript test. Q7: the six
  requirements go to `TODO.md` one by one. Q8: see below.
- **Q8 traced.** Preparing the walk-through, the *Family Evaluation* Ch3 words the proposal could not trace were
  found in the book's Chapter 3 source text. The recommendation was revised before the owner answered: keep the
  quotation, add the passage to `fe03.md` as a finding and cite it, and use the book's own subject in place of
  "differentiated actions".
- **Documents:** the proposal (status line, §6, Q8, its scan table and SP1.1), `TODO.md`,
  `docs/theory/_STATUS.md`, `docs/REVIEWER_READING_LIST.md` and `docs/SPEC_REVIEW_CHECKLIST.md`. No code, config
  or record changed.

Entries dated before 2026-10-22 are in `docs/CHANGELOG_archive.md`, verbatim.

### The SP-1 proposal drafted (2026-10-21)

- **What.** `docs/PROPOSAL — ASR QUOTATIONS (SP-1).md` (owner decision D1): a paraphrase and citation for each
  quotation of the 1979 tapes and the Kerr–Bowen interviews in the spec, the explainer, the ledger and
  `agent_model_proposal.html`. Nothing is applied; it waits for the owner's review.
- **The scan, rerun.** Every tracked `.md` and `.html` file was matched against the transcripts. In those five
  documents it found 151 quoted rows: 87 ASR quotations (65 in the spec), and 64 kept (45 written-source quotations,
  19 not quotations). Two further scans found two transcript copies without quote marks and one more Hijack-set
  interview credited to Bowen by name.
- **How the paraphrases were made.** Three agents drafted one paraphrase per sentence from the transcript passage,
  and this session drafted the three items from the further scans; four more agents, none of which had drafted
  them, checked each against its passage. The check corrected 34 of the 65 entries, most often to restore a hedge or
  a condition that the current text had dropped.
- **Found on the way.** A quotation credited to *Family Evaluation* Ch3 in `M4.D` that is in no extraction and no
  transcript (Q8); six requirements whose passage supports less than they say (Q7); two citation errors (adjacent
  issues).
- **The test.** Three forms were measured. The proposal recommends a citation test in the default suite and a full
  transcript match as a local test, deselected like `ensemble`; a committed transcript fingerprint is not
  recommended, because the repository is public.
- **Documents:** `docs/SPEC_REVIEW_CHECKLIST.md` indexes the proposal; `docs/REVIEWER_READING_LIST.md`, `TODO.md` and
  `docs/theory/_STATUS.md` say it waits for the owner. No code, config or record changed.

### The stage 1 milestone review triaged (2026-10-21)

- **The review.** The second Mac ran the first milestone review over version 2 stage 1, from the tag `version-1`, as
  `docs/milestone_review_brief.md` describes. It found 26 findings (0 blocker, 4 major, 22 minor) in four themes:
  `docs/cycles/2026-10-21_milestone-review_v2-stage-1.md`, on branch `review/2026-10-21-v2-stage-1`. It cost 462,483
  subagent tokens and 13 to 14 minutes (the report says 13; its clock reads 11:54 to 12:08), plus writing.
- **The triage.** Three agents checked every finding against the repository, about 534k tokens and 12 minutes. None
  was refuted. 22 hold as stated and 4 in part:
  - MR-5: one of its quotes is not exact.
  - MR-10: the table has nine live rows, not six, and `M11.4b` already says the table supersedes the count.
  - MR-11: `config/bowen/policy.md` declares the fallback; only its trigger cannot occur yet.
  - MR-13: the Dr Halim half holds; Teodor's half needed the owner.
  - SP-1, the quoted ASR passages, is larger than reported: 103 quotations in four documents, and 240 more in the
    `kb/*.md` notes (`TODO.md`).
- **The owner's ten decisions**, each asked separately, each the recommended option:
  - D1: SP-1 goes to a proposal first, which lists each quotation with a paraphrase and its citation before any spec
    edit. Approved DECISIONS records stay frozen with a note. The proposal decides the review's Q1–Q3.
  - D2: the wording fixes outside the spec are made now; the spec's go into revision 34 with SP-1.
  - D3: `M11.1`'s scope is the `M11.C` criteria. The coverage page now says which "done" rows rest on a mutation.
  - D4: the live-positions predicate has three callers, not four: `avoidance_available` (the `M5.C` site) is
    defined and tested but not called. Revision 34 says so, and whether to build that gate goes on the stage 3 list.
    `M13.1` stays done.
  - D5: `M4.D.1f` stays done; its note points to `M1.A.10`.
  - D6: Teodor's basic level is settled at 36. The note is removed from the plan and `TODO.md` now, from the spec at
    revision 34, and from `config/bowen/family_phase_d.md` with group G's regeneration.
  - D7: MR-24 and MR-25 are fixed now. MR-23 waits for group G's regeneration. The triage said the
    `tools/ensemble_record.py` docstring fix would move the record's hash too; it does not, because the hash drops
    docstrings, so it was made now. MR-26 and the tag-copy status test (Q10) stay as they are, about a second each.
  - D8: the brief is approved, with two edits.
  - D9: `docs/REVIEW_version_1_guide.md` stays as the owner's pending review, with its stale lines fixed.
  - D10: Q7 (fusion) and Q8 (the coach and SPLIT) go on the stage 3 list. The Holmes–Rahe Q3 is asked again before
    group G's build.
- **Fixed in this batch:**
  - MR-2: the explainer's durations table no longer lists the withdrawn "eight to ten generations"; a note says why.
  - MR-6: the coverage page's headline says which "done" rows a mutation proves (`tools/spec_coverage.py`,
    regenerated).
  - MR-7: the explainer grades the one-year slow tick `[I]`, as the config does.
  - MR-9, MR-14 and MR-16 to MR-22: status lines, counts and labels in the reading list, the checklist, `_STATUS.md`, the
    version 1 review guide and the version 2 plans. MR-14's `tests/doc_claims.json` entry now names the section it
    excuses.
  - MR-11: `M4.D.1f`'s coverage note points to `M1.A.10`.
  - MR-24: `tools/version_1_status.py`'s docstring says `dispositions()` is still live.
  - MR-25: `test_s1_3a_kinds_without_act_classes_fail_where_a_class_is_read` now reaches all three reads of an act
    class with unclassed kinds (`coach_side`, and `legal_outcomes` for a family member and for an external agent) and
    expects `TypeError`. An `or ()` default at any one of them turns it red.
  - `tools/ensemble_record.py`'s docstring names its test as it is, `test_m134_ensemble_record_is_current`.
  - `TODO.md`: Teodor's level and the two cutoff-semantics items are closed (built 2026-10-11, option B). A section
    lists what the review left open.
  - The brief is approved; its cost line has this run's figure, and the orchestrator's spec read may run alongside the
    workers. `_STATUS.md` has a run-log entry for 2026-10-20 and 2026-10-21. The reading list's known problems name
    what the review left open.
- **This batch's review.** One learning-qa pass and one correctness pass, about 320k tokens, raised 16 items after
  overlaps, 6 of them medium. All were fixed here except an adjacent stale line in `_LECTURES_1979.md`, recorded in
  `TODO.md`. A narrow re-sweep of the fixes on Sonnet, about 124k tokens, confirmed them and raised three wording
  points, which were fixed without another sweep.
- **Not changed:** nothing under `src/bowen/` or `config/bowen/`, so no record changes.

### The milestone review brief's four questions decided (2026-10-21)

- **The owner decided §8's questions**, each asked separately and each answered with the recommended option: Q1, the
  milestones stay the end of stages 2 and 3, before any tag and before an outside review; Q2, the second Mac runs it
  by default; Q3, one worker per lens plus the challenge pass; Q4, the first review runs now, over stage 1 from the
  tag `version-1`. The brief's status line, §2, §3 and §8 record this; it is still a draft.
- CLAUDE.md and the checklist's entry no longer call the brief "pending owner review", so
  `tests/test_stale_text.py`'s `PENDING` is empty again.

### The milestone review brief revised after the second Mac's review of it (2026-10-21)

- **The review.** At the owner's request the second Mac reviewed the brief itself, ran its §3 commands, and pushed
  `docs/cycles/2026-10-21_brief-review_mini1.md` on the branch `review/brief-2026-10-21`: three major and nine minor
  findings. Each was checked here against the brief and the repo before triage.
- **Refuted as stated: MR-1** (the brief wrongly made the `KB` interviews no-verbatim). The rule is in
  `docs/theory/_SOURCE_QUALITY.md` ("MUST NOT be quoted verbatim") and the explainer's citation legend, so the brief
  was right. The review was misled because `_LEDGER.md`'s source table and CLAUDE.md gave only "cite the interview,
  not the man". Owner decision: both now also say "no verbatim quotation" (no rule change), and the brief names where
  the rule lives and adds `_SOURCE_QUALITY.md` to L1's slice.
- **Fixed in the brief, as found:** workers are read-only, so the orchestrator hands them the diff and runs anything
  that must be run (MR-2); L1, L2 and L3's slices gain the spec's `M10`–`M11`, the mutation records, the result
  documents and the claim tests they need (MR-3); one `learning-qa` pass and one correctness pass per batch, as
  CLAUDE.md says (MR-4); a second command for each Python version's CI result (MR-6); step 5's filter gains
  `requirements.txt`, `pytest.ini`, `.github/`, `README.md` and `CLAUDE.md`, but not the dated logs (`CHANGELOG.md`,
  `TODO.md`, `LEARNINGS.md`), which are history (MR-7), so Q4's figures are now 107
  files, about 7,400 lines added and 700 removed; two dates and a count in §1 (MR-8; its first part held only in
  part); CLAUDE.md's and the checklist's milestone summaries now match §2 (MR-9); a source for the token figure
  (MR-11); confidence, verified and found-by fields in §6 (MR-12); `git fetch origin --tags`.
- **Owner decisions (2026-10-21):** step 3 runs the suite once on an interpreter in CI's matrix, and a failure only
  outside the matrix does not stop the review (MR-5); minor findings are not challenged, and triage checks each
  before recording it (MR-10); Group G and the event sizes get no review of their own, the end-of-stage-2 review
  covers them (the review's question 2). §8's questions Q1–Q4 are still open.
- **Checked:** a narrow `learning-qa` fact-check on Sonnet over this diff (about 4 minutes, about 102k tokens) found
  six consistency items, among them §6's header still asking for two suite counts and §7 checking minor findings only
  when there were three or fewer; all were fixed.

### A brief for milestone critical reviews (2026-10-21)

- **`docs/milestone_review_brief.md`** (new, draft for owner review; owner request 2026-10-21): a critical review of
  the whole model at each milestone, run on either Mac by a session without the build history. Four lenses: theory
  fidelity (invented values presented as sourced, attribution, rules that write an outcome, premises reported as
  findings, `M11.F.9`), spec against code, documents, engineering drift, each with the files it reads. It covers how
  the review is set up (a worktree on its own branch, the suite on 3.11 and the default Python, CI for the reviewed
  commit, a baseline diff without deleted files or generated pages), the report's form in `docs/cycles/` (the
  `reviewer` skill's form, with its coverage matrix and refuted list kept), triage on the working machine, and four
  owner questions (§8). A narrow `learning-qa` pass on Sonnet checked every fact, path and command in the draft; its
  eight findings were fixed, among them a theory check that selected nothing (every constant is `[I]`), a CI check
  that read the latest run rather than the reviewed commit's, and a worktree branch that tracked `main`.
  `tests/test_spec_consistency.py` now checks the brief's spec IDs.
  Listed in `docs/SPEC_REVIEW_CHECKLIST.md` and the reading list's §9; CLAUDE.md points to it.
  `tests/test_stale_text.py`'s `PENDING` allows CLAUDE.md's one "pending owner review" for it; remove that entry
  when the owner decides.

### GitHub's default branch `main` brought up to date; CI's Python 3.11 job fixed (2026-10-21)

- **`main` fast-forwarded to `plan-phase-b`** (owner, 2026-10-20: "make sure github and main are in sync and up to
  date"). `main` had stopped at `40922e9` (2026-10-06, spec revision 11), and every later commit was on
  `plan-phase-b`. `main` was an ancestor of `plan-phase-b`, so no commit was rewritten and no merge commit was made.
- **New work goes on `main`** (owner decision 2026-10-21, chosen over keeping `plan-phase-b` and pushing both).
  `plan-phase-b` stays on GitHub as history and takes no new commits. CLAUDE.md states the rule.
- **Documents:** `README.md` and `docs/REVIEWER_READING_LIST.md` told a reader that `main` was stale and to read
  `plan-phase-b`. Both now point to `main`; the reading list records the change under its known problems, and its
  status line and checklist entry say the revised branch guidance is not yet reviewed by the owner.
- **CI's 3.11 job failed on `938c0a0`** (Part B of the coverage plan): `tests/bowen/test_spec_coverage.py` put a
  backslash inside an f-string expression, which Python accepts from 3.12 (PEP 701), so pytest stopped at collection
  on 3.11 while the local suite, on 3.12, passed. The expression was moved out of the f-string. New
  `tests/test_python_versions.py` compiles every tracked Python file with the oldest version in CI's matrix when that
  interpreter is installed (it is on this Mac), and skips when not; proved red on the old line. `pytest.ini` now adds
  `-rfEs`, so a skip prints its reason under `-q`. A runtime difference (a library call new in 3.12) is still caught
  only by CI's 3.11 job, or by running the suite with `python3.11`.
  `ast.parse(..., feature_version=(3, 11))` does not catch this case, so the test uses the real interpreter.
- Suite: 1003 tests, passing on Python 3.12 and 3.11 on this Mac.

### The coverage page judges every built ID; the overrides file renamed (2026-10-20)

- **`docs/spec_coverage.md`: Part B of `docs/plan_spec_coverage_phase_d.md`** (owner 2026-10-20: plan §6 approved).
  The 165 IDs that read "no test named for it; its module was built in Phase C" (or D), and the 74 earlier overrides,
  each got a judgment: the existing test that verifies it and how, *partial* with what is untested or missing, or *not
  done* — not built, or checked by review. Six subagents drafted; Claude read every proposed *done*'s test against its
  requirement and changed 11 drafts; the owner approved the list in six groups (the plan's §7 lists each group's IDs).
  575 IDs: 291 done, 136 partial, 148 not done (was 280, 64, 231). Of the earlier overrides, 13 fell from *done* to
  *partial*, 9 rose to *done* and 2 fell to *not done*; none still names Phase C or D as work to come. A clause that is
  only a SHOULD keeps an ID *partial* (`M16.D.2`, owner).
- **The learning-qa review of Part B** (eight items, the plan's §7): `M1.A.4` to *partial* (the code's rate constant
  against "rather than by a rate constant"; owner); `test_m4a5_self_generated_load_derives_from_basic_level` now pins
  "a function of `basic_level` only" and `M4.A.5` stays *done* (owner); two notes reworded (owner); B1.4 asks pytest
  for the ensemble tests and checks every `test_m11c…` citation; B1.1 refuses a partly built module's note; the
  old-name scan reads more of the repo; one reading-list sentence narrowed.
- **`tools/spec_coverage_phase_b.json` renamed `tools/spec_coverage_overrides.json`** (B3); the page's header names it
  in full. `test_m141_untested_id_is_not_read_as_unbuilt` now reads a fixture, since no built ID is left with that note.
- **Five tests**, each run red by its mutation: no built ID is left without a judgment; every override says why; a
  *done* override citing an ensemble test needs that criterion *done*; no override names a built phase as to come (the
  wordings in `tests/doc_claims.json`'s `coverage_future_phase`); the overrides file has one name.
- **Recorded in `TODO.md`, not fixed** (owner): where the spec and the code disagree, text that names a built phase as
  to come, two spec rows that name tests that do not exist, tests weaker than the ID they are counted for (`M6.3a`
  among them), fields never written, six mechanisms with no register row.

### The document-claim tests merged; the coverage page reads every record; the reading list approved (2026-10-20)

- **Merged `claude/cranky-snyder-764f33`** (the document-claim tests, the entry below) into `plan-phase-b`, owner
  2026-10-20. Four conflicts resolved: CHANGELOG.md and TODO.md keep both sides, CLAUDE.md's suite count is recomputed,
  and `docs/spec_coverage.md` is regenerated. Its D5b test then reported the reading list's checklist label "For an
  outside reviewer", which is neither open nor settled.
- **`docs/REVIEWER_READING_LIST.md` approved by the owner** 2026-10-20: its status line, and the checklist label, now
  "Approved, for an outside reviewer". §3, §7 and §8 updated for the coverage page; §8's fixed list adds the
  document-claim tests and the coverage page.
- **`docs/spec_coverage.md`: Part A of `docs/plan_spec_coverage_phase_d.md`** (owner 2026-10-20: build it now).
  `tools/spec_coverage.py` reads the Phase C and both Phase D ensemble records and both mutation records, Phase D's
  not-built list and the criterion run as a test (`M11.C.37`); the header names each record with the hash it states;
  the owner's version 1 dispositions are quoted in the notes; an ID with no test named for it says so and says whether
  its phase is built. `M11.C.16`'s hand-written override is removed, and a criterion whose spec row says its own
  mutation clause is not met (`M11.C.14`, `M11.C.16`, `M11.C.46`) is not done. 575 IDs: 280 done, 64 partial, 231 not
  done (was 274, 70, 231); six criteria moved from partial to done, and no other status changed. The record readers are
  the tool's own: the frozen `tools/version_1_status.py`'s part II reader refuses the record stage 1 regenerated. Five
  tests, each run red by its mutation (the plan's §5). Part B (an evidence mapping for each ID no test is named for) and
  the overrides file's rename: later, as their own task (`TODO.md`).
- **The learning-qa review of Part A** (six items): a test in `tests/bowen/test_ensemble_record.py` read the old
  record name; three generated texts that described the old reading were re-rendered from their records and cache
  (nothing ran); the page's header says the clause check reads only the row's annotation (`M11.C.1` and `M11.C.38` are
  proved only by a joint mutant and stay done); the marker reads "unmet" and "not yet met" too and is pinned to its three
  rows; orphan keys are refused, not dropped. Four more tests, each run red by its mutation. Three minor items in
  `TODO.md`.
- `docs/SPEC_REVIEW_CHECKLIST.md`: the family file is "the reference family the owner specified", the wording V7's
  review gave four other documents. `docs/REVIEW_version_1_guide.md`: its drafting-time inconsistencies 2–4 and the
  `M14` entry say what has been fixed since. `TODO.md`: version 2 stage 1 is built and pushed, not "build next".
- The owner's fam-diag layout for the seed-7 export committed at the owner's request (the currency test accepts a
  save from the app; a later run of `tools/famdiag_export.py` redraws the positions). The untracked backup copy, the
  revision 32 render byte for byte, deleted. `.claude/launch.json` and `.rca/` added to `.gitignore` (per-machine).
  The method-literature reading reports of 2026-10-15 in `papers/` committed as they were, in their own commit.
- Suite: 997 tests.

### A reading list for an outside reviewer; the fam-diag export brought up to date (2026-10-20)

- **`docs/REVIEWER_READING_LIST.md`** (new, draft for owner review): the documents that describe what the model does
  and how it works, in the order a reviewer new to the project should read them; which version to review (GitHub's
  default branch `main` stops at spec revision 11); and the known problems in the documents. Linked from the top of
  `docs/SPEC_REVIEW_CHECKLIST.md`. Compiling it on 2026-10-19 found six stale documents; five were fixed on
  `plan-phase-b` on 2026-10-19 and 2026-10-20. The sixth, `docs/spec_coverage.md`, waits on
  `docs/plan_spec_coverage_phase_d.md`, whose new step A6 updates the list when the page is regenerated. CLAUDE.md
  now says the list is updated in the same commit as any status it restates.
- **The seed-7 fam-diag export named spec revision 32.** Stage 1's regeneration refreshed the trace and chart but not
  `docs/review/phase_c_famdiag_seed7.json`, which nothing tested. Regenerated with `tools/famdiag_export.py`: only the
  page note's revision and config hash changed. `test_v5_committed_export_is_the_current_render` now fails when the
  file no longer holds what the tool draws, or holds a key the tool no longer writes. It ignores the keys a save
  from fam-diag changes or adds (named in the test), so the owner's layout, saved from the app into the working
  tree, still passes. That layout is not committed.

### Tests that fail when a document restates a changed fact (2026-10-20)

`docs/plan_doc_claim_tests.md`, approved by the owner 2026-10-20 (Q1: status claims checked both ways; Q2: built after
`docs-stale-fixes` was merged into `plan-phase-b`). Each check's self-tests were run red against an empty stub before
the check was written. Each check was then shown failing by one live edit to a real file, reverted, and the parts added
after the learning-qa review by nine code mutants (the plan's §12).

- **The spec revision where it is restated** (D1, `tests/test_doc_claims.py`): the largest revision named in the
  checklist's spec entry, CLAUDE.md's spec row, `_STATUS.md`'s `▶ STATE` line and README.md's Status section is the
  front matter's.
- **Repo paths in README.md and CLAUDE.md** (D2): each is tracked by git, or it is listed as retired for that document
  in `tests/doc_claims.json` and its own sentence or line says so. Paths under every top-level folder git tracks are
  read, `.github/` included.
- **`requirements.txt` against the imports** (D3, `tests/test_requirements.py`): the third-party top-level imports of
  every tracked `.py`, mapped to their distributions, equal the requirements less the tools `tests/doc_claims.json` lists
  as installed and not imported (`ruff`).
- **The suite-size pattern widened** (D4, `test_claimed_test_count_matches_the_suite`): it reads "N tests, all passing"
  and "N tests stay green", and no longer reads "N tests passed" as a claim. Two dated phrases are excused by name in
  the data file, and each must still be in its document.
- **"Pending owner review" in the current-state documents** (D5a, `tests/test_stale_text.py`): README.md, CLAUDE.md,
  the checklist, `_STATUS.md`'s `▶ STATE` line and each `docs/` status line.
- **Status claims against each document's own status line** (D5b, `tests/test_doc_status.py`), in both directions: the
  checklist's entry labels, README.md's sentences and other documents' status lines. The first status word in a
  sentence decides, and a negated settled word is not one. A label or status in words outside both lists is reported
  unless the data file lists it as carrying no review status.
- The word lists, the retired paths and the labels and documents with no review status are in `tests/doc_claims.json`
  (rule 9), and the shared reading helpers in `tests/doc_text.py`.
- **What the tests found, fixed:** three checklist labels still read "For review" for documents reviewed, complete or
  signed off (the review of revisions 24–30, the version 1 dispositions, the V7 outputs review); two checklist lines
  said "pending owner review" for items the owner had reviewed (the dispositions' rows, revisions 24–27); and the G1 and
  Holmes–Rahe proposals' own status lines still read "DRAFT, for owner review" though every question in each was
  decided on 2026-10-19.
- `docs/spec_coverage.md` regenerated: its list of tests named for no spec ID gains the 34 new test functions.
- Suite: 984 tests (60 new).

### Version 2 stage 1: one engineering batch (spec revision 33, 2026-10-19)

The owner decided D1–D7 as recommended (`docs/implementation_plan_version_2.md` §1.2). No rule changed. Each
criterion's test was shown failing under its mutation (§1.10).

- **The grid engine is retired** (`M13.2` amended, D1, D4): `src/engine.py`, `src/main.py`, `monte_carlo.py`,
  `tests/test_simulator.py` (37 tests) and `docs/model_config.md` are removed, and kept at the tag `version-1`.
  `tests/test_repo_layout.py` fails if any comes back or is imported.
- **Exact float totals** (S1.1, D2 (b)): every float `sum()` in `src/bowen`, the renderer's included, is a
  `math.fsum`, and every integer total a `count()` that refuses a float. The golden fingerprints did not move at 8
  significant digits.
- **The canonical state includes the stressors** (S1.2, `M16.T.3`), and after the review every other field of the
  run's state but the event kinds and the draw service; a test walks the fields.
- **The act classes come from `config/bowen/act_classes.md`** (S1.3, rule 9). A misspelt act, an empty class and a
  side-taking act that is not a family-to-external move fail at load; kinds built without the classes carry `None`.
- **Triangle activity is decided by presence** (`M8.5` amended, D3 (b)): the spec now says what the code did; no
  result changes. Whether a fused occupant should make a triangle inactive is left for stage 3.
- **Tooling:** a failed or timed-out mutant child is that row's errored outcome and leaves no process, nor does a
  stopped parent; a line cut short fails the child (S1.5a, in `tools/child_process.py`, outside the engine hash);
  `code_text` splits lines as `tokenize` does (S1.5b); the Phase B mutation record is frozen (S1.5c, D5); one helper
  parses a record's JSON block (S1.5d); `tools/phase_c_trace.py` loads the family once.
- **The renderer is out of the engine hash** (S1.8a), and no verdict imports it (S1.8b). **The Phase D ensembles
  are cached** in `docs/records_cache/` and their records re-render from the cache (S1.8c).
- **The trace shows what a symptom gives back** (S1.7a, D6 (a)): a lock-in release and the spill past the turn each
  have a row; the weekly hold stays summarised. **The D2 record prints each variant under its own key** (S1.7b).
- **Riders** (D7): no decided item is still described as pending owner review (`tests/test_stale_text.py`); the
  pattern readouts' `now`, built in `f916d55` (Phase D B2), is tested; the Phase D mutation record's hash ignores
  comments.
- **The step 6 review** (`learning-qa` and a correctness pass; plan §1.10): `DrawService.categorical` never returns a
  zero-weight index at the top edge; `tools/record_changes.py` matches the D2 rows by position, so `M11.C.2`'s
  twelve-person variant is compared, compares the level-occupancy record, and refuses a record not regenerated;
  `count` takes numpy booleans; the act-literal scan finds lists, tuples and one-act comparisons;
  `tests/test_repo_layout.py` covers `docs/model_config.md`; the explainer's §13.1 note says which caller of the
  live-position predicate reads a side.
- **Fix re-sweep** (`learning-qa`, narrow, over the review's fixes): a stop signal the parent ignores stays ignored,
  so a run under `nohup` outlives its terminal, and the handler acts once; `tools/record_changes.py`'s one-sided
  readouts and missing-hash refusal are tested; `config/bowen/act_classes.md` states the loader's two new refusals.
- **The records are regenerated once** on the new hash (S1.10a); `docs/version_2_stage_1_changes.md` lists every
  verdict that changed against the tag `version-1` (S1.10b). The run took 3 h 35 min and **no verdict changed, no
  effect moved and no value differs** in any compared record; the seed-7 trace gained only the release rows.
- **`/csdp` sweep** (step 8; three narrow passes on sonnet: `learning-qa` over what followed the step 6 review, a
  correctness pass over the re-sweep's fixes, and a security pass over the subprocess and cache code; plan §1.10):
  `tools/record_changes.py` compares every field of a row, not only verdicts and mean differences, and still finds
  no difference, refuses a readout named twice in a row and prints floats in full; a stop signal while a child is
  spawned no longer leaves it running, and one between setting the two handlers or during the restore loop no longer
  leaves them ignored; the cache is written whole, per process, and moved into place; a child line that is not a row
  is counted, not a `KeyError`. A narrow re-sweep of these fixes found six low items, all fixed.
- **Cold sweep** (2026-10-20; one `learning-qa` pass in a fresh session after the push; plan §1.10): a record tool
  now exits 1 on a RAISED or BROKEN row as well as a FAILED one (`tools/record_cache.py`; such a row is written into
  the record and never cached, so the record's test failed while the tool exited 0); `tools/level_occupancy.py`
  refuses to write its record unless the child printed one row for each of C.16's arms, and the record was rerun
  (only its hash line changed); `docs/version_2_stage_1_changes.md` lists the frozen Phase B mutation record,
  version 1's status page and the Phase B traces among what it does not compare, and a test fails when a page whose
  header says a tool generated it, or a committed file under `docs/review/`, is in neither list. C3, C4 and C9,
  older than stage 1, are recorded in `TODO.md`. A narrow re-sweep of these fixes found the coverage page and the
  test count stale, and four low items; all fixed.

### `/csdp` sweep of the 22 unpushed commits (2026-10-19)

- **Swept:** learning-qa over `b773efd...cef5e43` (7 findings), `/code-review` over the changed code (7), and a
  security review scoped to the range (none). Fixed: the index's and the two proposals' stale status lines; the
  Holmes–Rahe proposal's §2 and G1's bond-energy note now grade their cases (`KS22.3` and `KS22.8` are `[K-ext]`,
  Part III; `KS07.1` and `KS10.4` `[K]`), and §2's `KS22.3` is described as the source has it (one man, two partners'
  pregnancies, not two sisters); the guide's `M11.C.46` row; the plan's "four largest" stocks.
- **The run chart** (`tools/run_chart.py`) names a declared person alive in no sampled week as not drawn, draws a
  person the family file does not declare under their id, escapes every `<`, `>` and `&` in its data, and fills its
  template in one pass. A new test fails when the committed seed-7 trace or chart is not the current render. The
  chart's data changed only by the new `absent` key.
- **`tests/test_version_1_status.py`** no longer passes the parent's `PYTEST_*` options to the run at the tag.
  `tools/version_1_status.py`'s `main` is typed `NoReturn` (no change in behaviour).
- **Fix re-sweep** (learning-qa, narrow, over the fix commits): the chart's template fill now refuses any `__NAME__`
  it does not fill, not only the three it knows; the Holmes–Rahe proposal's `KS07.1` line, graded `[T]`, said the
  visit raised a symptom, and the source says it improved the wife's functioning. It is corrected, with a note beside
  the owner's Q3 decision, which cited the case. Its minor items are in `TODO.md`.
- Two code-review items not fixed, with reasons in `TODO.md`.

### Version 1 tagged and its status page frozen (2026-10-19)

- **The tag `version-1` marks `d62cfdc`** (the owner's choice; `docs/version_2_plan.md` stage 0.1). That commit records
  the owner's dispositions of 2026-10-19 and has the same engine, config and records as `a3e8169`. In a clean copy of
  it the whole suite passed (921 of 921 collected).
- **`docs/version_1_status.md` and `docs/version_1_dispositions.md` are frozen at the tag.**
  `tests/test_version_1_status.py` now checks that the tag marks `d62cfdc`, that both files are byte-identical to the
  tag's, and that the tag's own eleven status tests pass in a copy of the tagged tree. A missing tag fails the test.
  Each check was shown to fail under a mutation: the page edited, the wrong commit pinned, `docs/` left out of the
  copy, the tag missing, and a partial run at the tag. `tools/version_1_status.py` refuses to write. CI checks out
  with `fetch-depth: 0` so the tag is present.

---

### Version 1 complete (2026-10-18)

- The owner accepted `M11.C.6`'s failure as a finding ("Accept M11.C.6's failure as a finding, as recommended, and
  record it so version 1 is complete"). `docs/version_1_status.md` reads V1, V2 and V3 met; V4–V7 hold. By the
  version 1 definition's §4 question 8 the spec is frozen at version 1; later revisions start version 2.

### Spec revision 32, the owner's review: the nodal-event spike, `M11.C.6` restated (2026-10-17)

- **The owner reviewed revision 32 one decision at a time** (plan §4.12, "The owner's decisions"): every decision
  approved, every proposed disposition accepted except `M11.C.17` ("Diagnose coaching") and `M11.C.6` ("Trace it
  first"), and V7 signed off with a wording fix ("the owner's family" now reads "the reference family the owner
  specified" in the family file, `_STATUS.md` and the visualization proposal). Then: "continue with your
  recommendations … keep working until version 1 is completed".
- **`M4.C.1c`'s nodal-event spike built** (the owner's form, "cut ties, as TRIGGER"): a nodal event reaching a person
  spikes each of their cut-off ties (rejection held, both members alive) by `nodal_spike_intensity` (1.0, `[I]`) for one
  week, from the next tick (`event_effects.nodal_spike`; called at tick step 2, at a symptom onset and at removal by
  tolerance). From its review: a tie only not interactive is not spiked (a narrowing, Claude's recommendation pending
  owner review); ended spikes are pruned each tick (a 40-year run was about 50% slower without it, same results);
  `canonical_state` holds the spikes still to land. Tests: `test_m4c1c_*` (unit, adversarial, wiring, one-week and
  once-per-tie, removal). An `M11.C.4` mutant removes the spike alone. Phase B gate G7's mutant now removes the pruning
  too.
- **`M11.C.17` diagnosed:** no defect; the effect dies at the estimator's evidence test (own functioning stays below
  basic level in both arms). With the spike it passes, weakly.
- **Records.** Every record regenerated once (3 h 26 min; code hash `64e190be…`, Phase C `243ebedf…`). No Phase C verdict moved; `M11.C.17`,
  `M11.C.22` and two `M11.C.7` pairs turned green, each with a red recorded mutant; `M11.C.46`'s named mutant now
  survives at the edge (clause not met, recorded). The status page reads V2 and V3 met; V1 waits on `M11.C.6`, pending
  the owner.
- **`M11.C.6` traced and restated** (post hoc, pending the owner): the first arm picked by birth order; the new arm is
  one child with the parents' projection-type acts as declared against withheld. It fails the other way (ad hoc
  +0.096 ± 0.044). Recommended to accept as a finding, with a rule question on `M7.E.1f`.

### Spec revision 32: finishing version 1 (2026-10-17)

Built under the owner's instruction "proceed using your own recommendations"; every decision is Claude's
recommendation, pending owner review (plan §4.12, `docs/version_1_dispositions.md`).

- **The version 1 status table (V1).** `tools/version_1_status.py` writes `docs/version_1_status.md` from the three
  ensemble records, the two mutation records and the dispositions file. It refuses a failing premise (or a mixed
  criterion's failing premise limb) as a finding, a disposition that cannot apply, a row after a break in the table,
  and records that are stale against the registered criteria. 11 tests in `tests/test_version_1_status.py`. It found
  `M11.C.46`, a premise approved at revision 15 that no record listed.
- **The failing premises (V2).**
  - `M11.C.4`: the baseline now holds the tie open for the whole run; held open only to the act, the policy cut the tie
    in every baseline seed, so both arms held a cutoff at the nodal event. `M4.C.1c`'s nodal-event spike is not built,
    and the spec says so.
  - `M11.C.32`: the mover is declared at level 60 in both arms (post hoc). At 40, since revision 16, the calm arm never
    reached the peak.
  - `M11.C.29` and `M11.C.22`: reclassified mixed. `M11.C.29`'s budget limb is proved by a unit test; its time-course
    limb is untestable in the arm, since no genuine `I-POSITION` completes. `M11.C.22`'s mild limb is the net of
    lock-in and other mechanisms; the earlier statement that the capacity rule caused its failure is corrected (over
    600 seeds the rule shifts the limb by −0.15 ± 0.53, ad hoc).
- **Built.** `M11.C.46` (the seeker's relief with a third closer to the seeker against one closer to the partner);
  `M11.C.7` (the coach's position: a coach is on a member's side for `triangle_activity_window` weeks after that
  member's `SPLIT` toward him, and a sided coach's contact lands nothing, by `positions_live`). The gate acts in every
  run with a coach. `Forced.withheld` accepts a `KIND>target` label.
- **Proposed out of version 1.** `M11.C.9` (no rule reads sibling position) and `M11.C.8` (Phase E).
- **Recorded mutations (V3).** `M11.C.11`, `M11.C.15`, `M11.C.39`, `M11.C.46` and `M11.C.7` in the Phase D mutation
  record; `M11.C.4`'s two in the Phase C record.
- **V7.** `docs/REVIEW_version_1_outputs.md`, with its framing checks in `tests/test_version_1_outputs.py`.
- **Reviews** (one `learning-qa` pass, one correctness pass, before the regeneration): fixed `M11.C.46`'s readouts (a
  week early, and summing other transfers), `M11.C.32`'s mover not re-initialised, a no-op mutant target, a mutant edit
  that never ran, two imprecise mutant descriptions, a silent `withheld` label, weak tests, and the status tool's
  silent paths. Minor items are in `TODO.md`.
- **Records.** Every record regenerated once after the fix (3 h 34 min on 10 cores; code hash `43cd03b8…`, Phase C
  `2fc39cf9…`). Verdicts: `M11.C.4`, `M11.C.32` and `M11.C.46` pass; every `M11.C.7` pair fails on its deviation limb
  and holds on the landed-contacts limb; the others as before, Phase C family numbers moved by the coach's gate. The
  generated status page reads V2 met and V3 met; V1 waits on the owner's review of 23 proposed dispositions.
  Suite: 909.
- **A defect caught after the first regeneration was started.** `M11.C.46`'s module-level `SEEKER, PARTNER, THIRD`
  rebound `PARTNER`, which `M11.C.22`'s and `M11.C.11`'s readouts read as the nuclear four's c2, so both changed with
  no change to their own code. Found by comparing seed 0 of every Phase C and part I criterion against HEAD; the run
  was stopped and restarted after the fix. Every criterion without a coach now matches HEAD exactly; those on the Phase
  C family differ through the coach's gate. New test: no criteria module binds a top-level name twice
  (`tests/bowen/test_ensemble_record.py::test_criteria_modules_bind_no_top_level_name_twice`).

### Spec revision 31: the owner's review of revisions 24–30 applied (2026-10-16 and 2026-10-17)

*Added 2026-10-17: these entries were missing from the changelog when revision 31 was built. Written from the commit
messages of `6c95465`, `50c6eaf`, `224aa88`, `14b9df3`, `df33d68` and `1f0064f` and from plan §4.11.*

- **The review (2026-10-16).** The owner decided all 85 entries of `docs/REVIEW_spec_revisions_24-30.md`: 81 approved as
  built, 4 changed (`M14.A`'s `channel_focus` and `tolerance` rows; `M11.C.14` restated, option B; `M11.C.6` limit (a)
  restated, option A; `M7.E.1f`'s band). Version 1's condition V4 is met.
- **Part A, documents only.** The changed entries, the text fixes and the seven inconsistencies the form listed;
  every revision 24–30 tag reads "approved by the owner 2026-10-16"; the spec's version reads revision 31.
  `tests/test_review_rev31.py` pins each change.
- **Part B, code and config.** `M11.C.14` as two technique variants, each against a baseline in which the parents hold
  back on the same weeks, with a one-sided bound mode in the runner; its gate-removed mutant survives both (clause not
  met, recorded). `M7.E.1f`'s band opened by the parents' acts: ±2% with none, to +5% with supportive acts, to −10%
  with projection, limiting the step only. The criteria settings are graded `[I]` and the loader enforces it.
- **The owner's premise decisions.** A capacity rule in `M6.I.4` (what an over-functioner takes is limited by their
  own room above the stress floor); `M11.C.23` reclassified composite, its failure accepted as a finding; `M11.C.40`'s
  clause 2 reported and proved by a unit test. On 2026-10-17 the owner kept the capacity rule, accepted `M11.C.22`'s
  failure "for now", and had `M11.C.21` reclassified composite with its failure accepted.
- **Records.** Every record regenerated (about 3.5 h on 10 cores), then again on 2026-10-17 with `M11.C.21`'s class and
  the capacity rule's no-span fix (a taker exactly at basic level takes in full when there is no span). Suite: 886.

### Platform-independent fingerprints, for the red CI (2026-10-15)

- CI had been red on every push since 2026-10-09. Only the golden fingerprints failed, and on Python 3.11 the
  diagnostic record too. The cause was the last bits of floats: Python 3.12 changed how `sum` adds floats, and the C
  maths library differs between macOS and Linux. Measured on this Mac, Python 3.11 against 3.12 on the same seeds:
  no record differs in structure; 75,496 of 197,685 phase_d floats differ, by at most 1.7e-11 relative. NumPy 2.5.4
  against 1.26.4 changed nothing.
- `tests/bowen/test_engine_golden.py` now hashes canonical JSON with floats rounded to 8 significant digits. Both
  fingerprints moved; the run itself did not change. A new test checks that a 1e-12 change leaves the fingerprint alone
  and a 1e-6 change moves it. With rounding turned off it fails.
- `tools/diagnostic_record.py` takes its mean with `math.fsum`. The record is unchanged. 29 other `sum(...) / len(...)`
  means are left as they are and listed in TODO.
- Suite: 850 tests. CI run 38763412772 is green on Python 3.11, 3.12 and 3.13, the first since 2026-10-09.

### `M11.C.2` on the twelve-person family: no single revision 30 change explains the regression (2026-10-15)

- Each revision 30 change removed alone, 150 seeds, ad hoc: unmodified +0.70 ± 1.5, p 0.16 (reproduces the record); `M4.D.1h`'s force removed +0.70 ± 0.97, p 0.13; `M1.B.8b`'s compounding removed +1.12 ± 1.7, p 0.09; `M1.B.8b`'s reinforcement removed +1.11 ± 1.6, p 0.093; `M1.D.8a` removed +0.96 ± 1.5, p 0.14; `M1.B.3e`'s danger removed +0.08 ± 1.8, p 0.46 (`symptom_load`, seeds 0–149, one-sided signed-rank with the runner's margin; ad hoc, 2026-10-15).
- No single removal restores the pass. Removing the force narrows the spread to revision 29's (±0.95 then) without
  raising the mean; removing either focus term raises the mean to about +1.1 with the wider spread. Recorded in plan
  §4.10 and TODO.

### `M11.C.10[family of origin]`'s clause restated to `M4.D.6f`, and a Phase D mutation record (2026-10-15)

- The owner: "restate the clause to M4.D.6f and add the mutant". The spec's `M11.C.10` row now requires `M4.D.6f`
  removed to turn the family-of-origin variant red; `M1.D.8a` removed is reported. The main variant keeps `M1.D.8a`.
- New `tools/phase_d_mutation_record.py` writes `docs/phase_d_mutation_record.md` from
  `docs/records_cache/phase_d_mutation.json`. Its child, `tools/phase_d_mutant_child.py`, runs `CRITERIA_D2` in a mutated
  copy; it is a separate file so the Phase C records' hash does not move. The part II record names two criteria
  `M11.C.10`, so its rows are matched to `CRITERIA_D2` in order, each id checked.
- First mutant, `witnessed-cutoff-learning-removed` (`witnessed_cutoff_gain` 0.05 → 0): **red**, FAIL,
  `cut_off_tie_weeks` −75.9 ± 132, p 0.91, 200 seeds, the same values as the ad hoc run.
- Tests: `tests/bowen/test_phase_d_mutation_record.py` (the mutant applies once, the record is current, the variant
  is red under it, and every passing part II criterion has a mutant here or is named with where its mutation is
  reported: `M11.C.15` and `M11.C.47`, both ad hoc in the plan).
- Review fixes (learning-qa and a correctness pass): the cache key also covers `apply` and `run_in_copy`, which copy the
  repository and apply the edit; an engine exception under a mutant is reported as a red and not cached, as in Phase C,
  and finished results are kept if the machine fails; the spec's `M11.C.10` row states the restated clause in its
  Mutations sentence, not only after it. The record was rerun under the new key.

### `M11.C.10[family of origin]`: learning from a witnessed cutoff carries its pass (2026-10-15)

- The three remaining candidate mutations, 200 seeds each, ad hoc: `M4.D.6f` removed (`witnessed_cutoff_gain` 0)
  gives −75.9 ± 132 tie-weeks; `M1.B.3d`'s reactive rise removed (`reactive_cutoff_gain` 0) gives +163 ± 117;
  `M1.B.3e`'s danger removed (felt danger from the declared floor only, 0 in this family) gives +102 ± 126. Unmutated:
  +162 ± 120. Only removing `M4.D.6f` puts the mean below zero, so the children learning from the renewals they witness
  carries the pass. The criterion's own signed-rank test was not run on the mutants. The spec's clause still names
  `M1.D.8a`; whether to restate it is in TODO. Recorded in the spec's `M11.C.10` row, plan §4.10 and TODO.

### `M11.C.10[family of origin]`: its pass is not carried by `M1.D.8a` (2026-10-15)

- The variant's mutation, `M1.D.8a` removed (both gains 0), does not turn it red: +240 ± 116 tie-weeks (`child_cut_weeks` +49.7 ± 44.1; 200 seeds, ad hoc, 2026-10-15), against
  +162 ± 120 with it. The intervals overlap, so removing it is not shown to change the result either way. The spec
  requires this mutation to turn the variant red, so the variant's pass has no mutation proof and is not evidence for
  `M1.D.8a`. Something else carries the parents' held cutoffs to the children's rejection; not yet traced. Recorded in
  the spec's `M11.C.10` row, plan §4.10 and TODO.

### Spec revision 30 built: origin cutoff, danger, focus lock-in (2026-10-14)

- Built plan §4.10 (575 numbered requirements, 47 acceptance criteria): the cutoff a child grows up with raises their
  chronic anxiety and pull toward closeness at 20 (`M1.D.8a`); danger read from received intrusion, its contact cost,
  and a cutoff principled in proportion to it (`M1.B.3e`); an anxious act lands with more force (`M4.D.1h`, the
  owner's "force rises with anxiety"); anxious focus moves to where it already is (`M1.B.8b`, "anxious focus
  compounds"); `M11.C.10` read on ties to partners and own parents at a fixed 200 seeds, with a family-of-origin
  variant; the danger premise `M11.C.47` on a new fixture. Seven new `[I]` constants; one family-config key
  (`dangers`).
- The owner's decisions after the reviews: felt danger is held while rejection is held; focus compounding became a
  shift among the children that adds no investment, at `focus_compound_gain` 10 ("Shift, gain 10"), after the first
  form grew without limit and a ceiling removed the lock-in.
- Review fixes, each with a test: `M11.C.47`'s arms differ in one person and read the danger felt from them (most of
  it is the child's); the cached twelve-person run's move counts were shared between arms and double-counted; the
  `origin_cutoffs` key, which the spec forbids, is removed; a dead person's declared danger stops being a floor; weak
  tests strengthened and wiring tests added.
- Records regenerated (code hash `c41fea60…`, Phase C `6a8c31f0…`):
  - `M11.C.10[family of origin]` **passes**: +162 ± 120 tie-weeks (p 0.007, 200 seeds).
  - `M11.C.10` still **fails**: +6.6 ± 130 (p 0.43, 200 seeds).
  - `M11.C.47` passes (premise): felt danger +0.050 ± 0.017, principled cutoff +0.020 ± 0.008; but the weeks of
    rejection held toward the low-level person are fewer, −16.6 ± 4.9 (reported only).
  - `M11.C.2` on the twelve-person family now **fails**: +0.70 ± 1.5 (p 0.16), from a pass at revision 29.
  - `M11.C.17` FAIL → PASS (+0.00053 ± 0.00063, p 0.013); `M11.C.44` PASS → FAIL (−0.030 ± 0.11). `M11.C.6` fails
    (+0.070 ± 0.039); `M11.C.15` passes. No mutation proof lost by a passing criterion.
  - Mutations: `M11.C.47` with the intrusion read removed reads exactly 0 in every seed (100 seeds, ad hoc), so it
    cannot pass. The `M11.C.10[family of origin]` mutation is not yet run (run 2026-10-15: it does not turn the
    variant red; see the entry above); `M11.C.10` and `M11.C.2` fail, so theirs are moot.
- The `rca` skill (review convergence audit) added at `.claude/skills/rca/SKILL.md`, from the owner's trans-summary
  project; its first audit: MIXED / INSUFFICIENT DATA, and CI red on every push since 2026-10-09 (TODO).
- Suite: 849 tests. Spec, plan §4.10, explainer §8.3, constants docs, criteria config, the danger fixture, README,
  `_STATUS.md`, CLAUDE.md, the proposal and TODO updated.

### Spec revision 29 built: cutoff and focus (2026-10-14)

- Built plan §4.8, R1–R7: reactive and principled cutoff (`M1.B.3d`), learning from what follows a witnessed cutoff
  (`M4.D.6f`), chronic anxiety as the mean of the childhood years (`M7.B.1a`), focus on a child's real or perceived
  weakness (`M1.B.8a`), focus steering projection-type acts (`M4.D.1g`), `M11.C.10` with a held cutoff and a tie-week
  readout, and `M11.C.2` read by age. Five new `[I]` constants; two optional family-config keys (`principled_cutoffs`,
  `weaknesses`).
- Review fixes, each with a test: a declared weakness acts only through an anxious parent; the principled kind ends with
  the declared rejection; focus keeps each projection kind's total weight; one learned-value key format
  (`src/bowen/engine/value_keys.py`); range checks on the new constants; skipped scripted renewals counted.
- Records regenerated (code hash `eeef3517…`):
  - `M11.C.2` on the twelve-person family now **passes**: +1.31 ± 0.95 (p 0.005). Mutations (ad hoc): the witness path off +0.27 ± 0.75, `M1.B.8a` with `M4.D.1g` off +0.45 ± 0.99; neither interval excludes zero.
  - `M11.C.10` still **fails**: −63.1 ± 200 (p 0.71). Removing `M4.D.6f`, the reactive rise, or both, moves nothing
    detectable (ad hoc, 100 seeds each).
  - `M11.C.6` fails (+0.030 ± 0.024); `M11.C.15` passes (death +1.34 ± 0.40).
  - Elsewhere: `M11.C.44` FAIL → PASS, `M11.C.17` PASS → FAIL. No mutation proof lost.
- `I-POSITION`'s ensemble tests run with focus steering off by a declared override (TODO: they rest on about three
  completions). Two part II tests tightened (`M6.3b`, `M6.3e`).
- Suite: 812 tests. Spec, plan §4.8, explainer §8.1–8.2, the part II status report, constants docs, criteria config,
  README, `_STATUS.md`, CLAUDE.md and TODO updated.

### Spec revision 29 (570 numbered requirements, 46 acceptance criteria): cutoff and focus, drafted (2026-10-13)

- Applies the owner's decisions after the diagnosis:
  - `M11.C.10` holds the cutoff for the run and reads tie states.
  - Learning from a witnessed cutoff (`M4.D.6f`).
  - Reactive and principled cutoff, as a placeholder (`M1.B.3d`).
  - Chronic anxiety built over childhood (`M7.B.1a`).
  - A parent's focus on a child's real or perceived weakness (`M1.B.8a`, `M7.E.1c` at `t0`), and focus steering acts
    (`M4.D.1g`).
  - `M11.C.2` read by age.
- The forms are Claude's recommendation, pending owner review. Plan §4.8, awaiting approval. Nothing is built.

### Why `M11.C.10` and `M11.C.2` fail on the twelve-person family (2026-10-13, ad hoc diagnosis)

- `M11.C.10`: the declared Ravi–Dario cutoff ends within about 2 years, and the uncut baseline is cut within weeks, so
  the arms differ only at the start. Two of `M7.E.3`'s three links have no mechanism: witnessed-cutoff learning and
  origin cutoff as an input to intensity. Cut-off acts number about 775 per run, mostly generation 3 cutting their own
  parents.
- `M11.C.2`: the parents spread projection evenly over the children (about 80 acts each). Chronic anxiety reflects only
  the year at 11, and the level band leaves no room. Late load follows each child's age.
- Plan §4.7; owner decisions in TODO. No code changed.

### `M11.C.15` keeps the transfer at a death (2026-10-13, owner decision)

- The owner: "keep the transfer in C.15, share at 1". `M11.C.15`'s arms now zero only grief and lost income
  (`NO_GRIEF_OR_INCOME`, `src/bowen/ensemble/criteria_d2.py`); `death_transfer_share` stays at 1 and no criterion
  overrides it.
- `M11.C.15` now passes: death +1.25 ± 0.38 (p 2.1e-08), recovery +3.11 ± 0.35.
- Its mutation, the bound anxiety discarded at the death, turns it red (death −2.03 ± 0.29; ad hoc, plan §4.7).
- `canonical_state` now includes the life course, and a new test shows a change to the life course alone changes it.
- Every record was regenerated (code hash `4dbf3f0f…`); no other verdict moved. Suite: 766 tests.
- Spec `M6.3b`, plan §4.4 and §4.7, `config/bowen/criteria.md`, `config/bowen/constants.md`, the part II status report,
  TODO and README updated.

### Phase D part II built: the life course (2026-10-13)

*`M11.C.15`'s result below is superseded by the entry above: it passes since the owner kept the transfer.*

- **The life course** (spec revision 28, plan §4, steps L0–L10): a new module, `src/bowen/engine/life_course.py`. It
  runs only when a family declares one, so Phase C, part I and their golden runs are unchanged.
  - Ages, with acting from 10; a child's chronic anxiety before 12 and development from 5 to 20.
  - Launch; partners and births from `config/bowen/partners_phase_d.md`, each new couple getting a family of its own.
  - Mortality, and the disposition at death (`M6.3a`–`3i`).
  - Grief and lost income, and `FIND_WORK` (`M7.C.1k`).
  - The coach's retirement and successor.
  - Life events are `Mechanism.LIFE_EVENT`, with no sender.
- **Twelve-person family:** the coach is tied to the coachee only. Starting ages are three settings in
  `config/bowen/family_phase_d.md`.
- **Constants:** 23 new ones in `config/bowen/constants.md`, all `[I]`.
- **Criteria** (`src/bowen/ensemble/criteria_d2.py`) and their record, `docs/phase_d2_ensemble_record.md`
  (`python3 tools/phase_d_record.py --part 2`, 52 minutes on 10 cores).
  - `M11.C.37` passes, with its mutant, as a test.
  - Four fail: `M11.C.15` (death +0.212 ± 0.35, p 0.135; recovery +3.11, holds), `M11.C.6` (+0.018 ± 0.028),
    `M11.C.10` (−12.6 ± 28) and `M11.C.2` on the twelve-person family (−0.13 ± 1.6).
  - An ad hoc diagnosis puts `M11.C.15`'s death limb on its arm design, which discards the dead person's anxiety
    (plan §4.7). It is an owner decision; no arm was changed.
- **The `M16.T.3` re-run** on the life course passes (`test_m16t3_sink_does_not_change_a_life_course_run`).
- **Records:** every record was regenerated; no Phase C or part I verdict moved.
- **Suite:** 765 tests, plus 45 ensemble-marked tests.
- **Report:** `docs/phase_d2_status_2026-10-13.md`.

### Spec revision 28 (566 numbered requirements, 46 acceptance criteria): the life course (2026-10-13)

- Applies the owner's answers to P13–P15 (`docs/implementation_plan_phase_d.md` §1b): what happens at a death
  (`M6.3a`–`M6.3i`), finding work after a lost income (`M7.C.1k`), starting ages as settings (`M2.A.0j`), the coach's
  career and single tie (`M2.A.0k`, `M2.B.2`), acting from 10, launch, partners, births and mortality
  (`M7.C.1f`–`1j`), chronic anxiety before 12 (`M7.B.1a`), and a child's development from 5 to 20 (`M7.E.1f`). The
  forms are Claude's recommendation, pending owner review. Built the same day (above).

### The Phase D ensemble record (2026-10-13)

- **`tools/phase_d_record.py`** writes `docs/phase_d_ensemble_record.md`: every Phase D part I criterion built so far
  over adaptive ensembles, stamped with the code hash, and the three not built with the reason (plan step 12). A
  default-suite test (`tests/bowen/test_phase_d_record.py`) fails when it is stale. 12 PASS, 8 FAIL, the same verdicts
  as the ad hoc runs. The mutations are not yet a record. Suite: 711 tests.

### Spec revisions 25–27: beliefs as a channel, projection, and step 10's criteria (2026-10-12) — Claude's recommendations, pending owner review

The owner asked for the work to proceed overnight without waiting for input; none of these is an owner decision.
- **Revision 25, B8 (`M9.6`, `M9.7`, `M1.A.11c`, `M5.C.1`).** Each person holds where they believe the family's
  difficulty lies, written from moves sent or delivered, by rules in a new config file `config/bowen/attribution.md`;
  a readout (`src/bowen/readouts/attribution.py`) recovers the sink from it. A witness's appraisal reads its belief
  about the tie it watches. The family's anxious focus writes the symptom channel. The marital-distance gate blocks
  improvement in a symptomatic child. New constants (`[I]`): `attribution_rate` 0.05, `focus_rate` 0.05,
  `belief_appraisal_gain` 1.0, `marital_distance_fraction` 0.25. New field: `Person.attribution`.
- **Revision 26, B9 (`M4.D.6e`).** The cross-person learning term is weighted by the actor's bond energy on each tie.
- **Revision 27 (`M6.I.4`).** Borrowed pseudo-self settles back, conserved, once a pair's exchange has stopped for
  `exchange_window` weeks (8); new `pseudo_self_relaxation_rate` 0.05 (`[I]`). This changes every run in which lending
  stops.
- **`M10.B.5`.** Learning is switched off by an override the criterion declares (`scenario(overrides=...)`).
- **B10, `M11.G`.** A family-evaluation readout (`src/bowen/readouts/evaluation.py`): components 1–8, adaptiveness
  derived against stress, belief and truth apart, the framing carried; `M11.D.20`'s ordering check passes.
- **`M11.C.23`'s suppression term**: new field `Person.symptom_suppression` (1 is none; set by a criterion only).
- **Criteria (ad hoc on the final code, not yet a record).** Pass and proved by mutation: `M11.C.26`, `.36`, `.39`,
  `.24`. Fail: `M11.C.14`, `.2` (acts counted early, load read late), `.40` (its second clause), `.43`, `.34` (reactions
  to the onset against a no-onset run), `.21`, `.23` (reversed). Not built: `M11.C.7`, `.9`, `.28`, each needing a
  mechanism first.
- **`M11.C.17`'s coached rise nearly vanished** (+0.021 at revision 23, +0.0009 ± 0.0008 now; still PASS at p 0.011).
  Probes put part of it on revision 27's settling back and part on revision 24's integrand. Recorded, not tuned.
- **Reviews' fixes:** the marital-distance gate was on almost every week (being below the contact optimum is the
  resting state); it now needs felt contact below `marital_distance_fraction` (0.25, `[I]`) and reaches only the
  couple's children, not a symptomatic spouse. Pseudo-self settled between people who never exchanged; it now settles only on a tie with a
  functioning balance. Criterion overrides are applied before assembly. `M11.C.34`, `M11.C.2`, `M11.C.40` and
  `M11.C.26` readouts were corrected (see the plan, step 10); an empty entropy window no longer reads as zero. `TRIANGLE` no longer writes focus.
- **`I-POSITION` completion fell** in the triad fixture (about 10 in 96 seeds at revision 20, 4 in 192 now);
  `test_iposition.py`'s sequence test was raised to 192 seeds (P23).
- **Status report:** `docs/phase_d_status_2026-10-13.md`, every build item and D-I criterion as done, partial or not
  done.
- **Records regenerated.** `M11.C.44` moved PASS → FAIL again, on its threshold. No other Phase C verdict moved. The
  golden fingerprints moved (behaviour). Suite: 708 tests.
- **Test fixed:** `tests/bowen/test_phase_c_criteria.py`'s C.42 check counted any event a pair member sent toward the
  third, including a symptom onset, which is not a choice; it now counts moves only.

### Spec revision 24: the symptom channels, Phase D B7 (2026-10-12) — Claude's recommendation, pending owner review

The owner asked for the work to proceed overnight without waiting for input. Every choice here is Claude's
recommendation, labelled so in the spec, and none is an owner decision.
- **`M4.C.3a`, amended.** Symptom load builds from the excess above the person's usual level (revision 22's
  yardstick), not above the chronic floor. Measured before it: in ten-year runs of the nuclear-four fixture every
  person crossed the threshold in the first weeks and stayed there.
- **Built:** the expressed channel moved by a family focus and overridden by constitution (`M1.A.11c`, `M7.D.2`);
  mutual protection between channels (`M7.D.2c`); lock-in with its turn, through a new stock `Person.symptom_bound` in
  `M6.4`'s table (`M7.D.2a`, `M7.D.2d`); a remitted symptom's hold returns to the partners (`M1.F.9`); removal fired
  by the others' tolerance, suppressed by external decision ownership (`M7.D.4`). A remitted symptom's load is no
  longer credited to the family budget (it is not an anxiety stock). The symptom threshold is floored at
  `functional_level_floor`, as steepness is (`M1.A.6`), and every channel is checked for onset, each crossing its own
  event (`M7.D.1`).
- **New constants** (`[I]`): `constitution_strength` 0.5, `channel_protection` 1.0, `lock_in_gain` 0.2,
  `lock_in_turn` 3.0. **New fields:** `Person.channel_focus`, `constitution_strength`, `symptom_bound`;
  `Family.decision_ownership_external`.
- **`M11.C.22` restated** to read each limb against removing the same member with no symptom.
- **Result (ad hoc):** `M11.C.30`, `M11.C.22`, `M11.C.31` and `M11.C.11` pass. C.30 and C.22 turn red under their own
  mutations; C.31's mutant cannot run, because discarding the held anxiety breaks the `M6.I.6` ledger at the release
  week. `M11.C.12` fails on the dyad.
- **Test fixed:** `tests/bowen/test_famdiag_export.py` still counted `CUTOFF` acts as a cutoff's support; since
  revision 17 a cutoff is read from the other's refused approaches. Seed 3's run under this engine was the first to
  read a cutoff with no `CUTOFF` act in its block.
- **Records regenerated.** `M11.C.44` moved FAIL → PASS again: it sits on its threshold (see `TODO.md`). No other
  Phase C verdict moved. The golden fingerprints moved (behaviour). Suite: 683 tests.

### Spec revision 23: the estimator reads pseudo-self with its sign (2026-10-12)

- **`M1.A.4k`.** The estimator's history reads `functional_level` less pseudo-self with its sign: what was given way is
  added back, as what was borrowed was already left out (owner decision). The exchange shifts functional level in both
  people and leaves basic level alone.
- **Why.** A trace showed the load weighting barely mattered; leaving out only the borrowed part read the coached
  people about 9 points below basic level whatever they did (`docs/implementation_plan_phase_d.md`, step 6).
- **Result (ad hoc, 150 seeds).** **`M11.C.17` passes and is proved by mutation** (CLAUDE.md's bar: removing the
  landed contact turns it red): coached +0.021 ± 0.009. No minimum meaningful effect is declared, and the effect is
  small (about 0.02 points of basic level in ten years). In a 20-seed trace, 16 of 40 coached people had risen by year
  10 against 4 of 40 without a coach. C.33, C.18 and C.20 pass (+0.013 each); C.18 and C.20 each turn red under their
  own mutation, and C.33's own mutation (the slow-clock dependence gate) was not rerun.
- **New constants:** none.
- **Records regenerated:** no Phase C verdict moved. The golden fingerprints did not move; those runs do not exercise
  the estimator (see `tests/bowen/test_engine_golden.py`). Suite: 661 tests.

### Spec revision 22: calm is relative to the person (2026-10-12)

- **`M1.A.5e`, `M1.E.7`.** Loaded and calm are read against each person's usual level: their excess anxiety averaged
  over about a year (new `usual_level_rate`, 0.02, `[I]`; new Person fields `usual_excess`, `usual_weeks`). The stress response is
  driven by the excess above it; at or below it a person is calm, and what stress moved and the system's adjustment
  settle back. Self-discovered practice counts only above it. The policy's bands and the estimator's load weighting are
  unchanged (owner decisions).
- **Why.** A trace showed the Phase C family's own traffic holds the coached pair 30–35 points above chronic (the
  others 2–25), with or without nodal events, so against a fixed band of 5 the coached pair were never calm.
- **The usual level's start.** It is the plain mean of the weeks so far for the first year, then the one-year running
  average (owner decision, after review: started from week 0 alone, the first year was over-load).
- **Result (ad hoc, `docs/implementation_plan_phase_d.md`, step 6).** The stress response now settles, and coach-free
  people's functioning no longer erodes (+2.7 over ten years). Coached people still fall a little (−2.2), through giving
  way. The estimator's evidence stays near −9, and every criterion's readout is now tiny (about +0.002 a person):
  C.33, C.18 and C.20 pass the runner's test, and `M11.C.17`'s pass (+0.003) is not counted.
- **Records regenerated:** no Phase C verdict moved. Suite: 659 tests.
- **New constants:** `usual_level_rate`.

### Spec revision 21: holding a position stops the functioning exchange (2026-10-12)

- **`M1.B.5`.** An `OVERFUNCTION` or `UNDERFUNCTION` delivered to a person holding a position toward its sender moves
  no functioning, pseudo-self or balance, either way (owner decision). Holding a position is read from behaviour: a
  self-directed act or `WITHHOLD` toward that person in the last `practice_window` weeks, and no reactive act toward
  them since — any automatic act toward them, or triangling someone in about them (owner decision, after review)
  (`src/bowen/engine/moves.py::holds_position`, `external.record_stances`). Logged as `position_held`.
- **`M11.C.20`** reports the share of its borrowed arm's scripted support that a held position blocked (report only;
  owner decision: keep the rule, show it).
- **Why.** A trace showed coached people stopped over-functioning, and others then took charge of them more, taking
  functioning from them each time (`docs/implementation_plan_phase_d.md`, step 6).
- **Result (ad hoc).** `M11.C.33`, `M11.C.18`, `M11.C.20` pass, none of them carried by this change. `M11.C.17` passes the
  runner's test (+0.01 ± 0.01, p 0.02) but is **not counted**: the interval includes zero and no coached person's basic
  level rose in the trace. With stances ending on any reactive act, coached people fall as far as coach-free ones
  (−11.4 against −11.1); with soothing alone ending them they fell less (−8.0 against −12.3). Under constant load
  nobody's basic level rises; the owner's questions on C.17's load and the estimator's form remain.
- **Records regenerated.** `M11.C.44` moved back PASS → FAIL, as it had moved FAIL → PASS at revision 20: it sits on its
  threshold (see `TODO.md`). No other Phase C verdict moved. Suite: 656 tests.
- **New constants:** none.

### Spec revision 20: the system's adjustment settles back; `M11.C.18` and `M11.C.20` restated

From the owner's decisions of 2026-10-11:
- **The system's adjustment settles back** (`M1.E.7`). The gain the other person takes when a held stance works is
  kept in their `adjust_swing` and shed by `response_relaxation_rate` each calm week, as the stress-moved part is. While
  they are loaded it holds. The estimator leaves it out (`M1.A.4k`), as it leaves out borrowed pseudo-self: the
  owner chose this after the review showed a held adjustment could otherwise become a lasting basic rise.
- **It is a level, not a stack** (2026-10-12). Built first as a stack, it grew to 20–35 points under the criteria's
  steady load and carried `M11.C.17`'s pass (the +0.66 reported for `a35a962`, and revision 19's +0.63, were that
  defect). Now the other is up by at most `system_adjust_gain`, renewed while the stance works. A first fix also took
  falls off the adjustment; the review found that recorded each fall in two ledgers and let the stack return, so it was
  removed. Every change to functioning is now in exactly one ledger.
- **`M11.C.18` and `M11.C.20`'s mutation clauses** now name the mechanism that carries each: the stress response
  (`M1.A.5e`) for C.18, and the estimator reading functioning without borrowed pseudo-self (`M1.A.4k`) for C.20.
  Flattening `M1.A.4b`'s load weighting leaves both passing; the load weighting stays proved by its unit test.
- **Result (ad hoc, not yet in a record).** `M11.C.33`, `M11.C.18` and `M11.C.20` pass, and each restated mutation
  turns its criterion red. **`M11.C.17` fails**: in ten years nobody's basic level rises in either arm. A ten-year trace
  of the coached people (`docs/implementation_plan_phase_d.md`, step 6) shows coaching changes behaviour, but under
  load in every week functioning falls in both arms and the estimator's evidence stays far below zero. Open for the
  owner (`TODO.md`).
- **Records regenerated.** One Phase C verdict moved: `M11.C.44` FAIL → PASS (`outside_inside_ratio` +0.067 ± 0.078,
  p 0.055, became +0.101 ± 0.082, p 0.012). It sat on the edge before, and both its mutants survive, so by `M11.5` it
  is not proved (`docs/spec_coverage.md` keeps it partial). The mutation record's survivors went from 55 to 57: those
  two. Suite: 651 tests.
- **New constants:** none.

### Spec revision 19: the stress floor, self-discovered perspective, the system adjusting, the soothing review

From the owner's answers of 2026-10-11, after tracing why `M11.C.17` ran the wrong way:
- **Stress floor.** Stress, and giving way to a partner, cannot push functioning below basic level − 15
  (`M1.A.5e`). Without the floor, soothing spiralled and anxiety ran away in 11 of 20 four-year runs. With the floor on
  stress alone, 2 of 20 still did. With it on both, none of 20.
- **Self-discovered perspective** (`M1.E.7`). A self-directed act builds perspective only if it worked: the person held
  their position, and their reactivity on that tie came down over 4 weeks.
- **The system adjusts** (`M1.E.7`). When a held stance works, the other person functions up.
- **Perspective steers** (`M4.D.1a`; `config/bowen/perspective_steering.md`): away from triangling and blaming, toward
  detriangling.
- **The estimator ignores borrowed functioning** (`M1.A.4k`). It reads functioning less the positive pseudo-self that
  over-functioning took.
- **The soothing review** (`M4.D.6`). Now and then a person asks whether soothing helped, more often at a higher level
  and perspective. If it did not, soothing loses its learned appeal.
- **`M11.C.20`.** Perspective only in the basic arm.
- **`M11.C.17`.** Ten years, reading the people the coach reaches.
- **Result (ad hoc, 150 seeds).** All four estimator criteria pass, and each is turned red by removing the mechanism
  that carries it. The spec's load-weighting mutation leaves C.18 and C.20 passing, which was open for the owner (restated at revision 20).
- **Records regenerated:** no Phase C verdict moved. Suite: 647 tests.
- **New constants:** six `[I]`.

### Spec revision 18 (547 numbered requirements, 46 acceptance criteria): stress, coaching and self-work

From the owner's answers of 2026-10-11 (`docs/PROPOSAL — STRESS ON FUNCTIONING, COACHING AND SELF-WORK.md`, approved):
- **Stress moves functioning** (new `M1.A.5e`). Under load, coping (a self-directed act or holding back) raises
  `functional_level` and avoiding or soothing lowers it, each scaled by basic level. The soothing acts are listed in
  `config/bowen/stress_response.md`. The moved part settles back slowly when calm (`src/bowen/engine/stress_response.py`).
- **A stressor lasts** (`M1.F.6`): it weighs on its targets for its declared duration.
- **Systems perspective opens the self-directed channel** (`M4.D.1a`).
- **Perspective grows by a person's own practice** as well as by coaching (`M1.E.7`): 0.001 per self-directed act
  under load. At 0.01 nearly everyone saturated within ten years.
- **New constants:** six `[I]`.

### Spec revision 17 and the estimator's criteria

- **Revision 17** (owner, 2026-10-11):
  - nodal events are any major event on a person (`config/bowen/nodal_events.md`; new kinds `JOB_CHANGE`, `MOVE`,
    `SERIOUS_ILLNESS`);
  - cutoff is read from behaviour (`src/bowen/readouts/patterns.py`);
  - `M11.C.17` rewritten: a coach helps and is not required;
  - `M11.C.7`'s direction: joint > one spouse > sided.
- **The estimator's criteria**, in `src/bowen/ensemble/criteria_d.py`, each with a declared three-kind nodal load. Ad hoc
  run, not yet in a record:

  | Criterion | Verdict |
  |---|---|
  | `M11.C.33` | passes, and is proved by mutation |
  | `M11.C.18` | fails: +0.05 ± 0.11 |
  | `M11.C.20` | fails, reversed: the borrowed gain rises more |
  | `M11.C.17` | fails: no coach effect |
- **The causes,** recorded for the owner in the plan's step 6 result:
  - anxiety does not move `functional_level`;
  - the coach's effect does not reach the behaviour the estimator reads.

### Phase D B5 and B6: the slow tick and the basic-level estimator

- **B5, the slow tick** (`M3.B.1`; owner decisions P5, P6, P17).
  - **When:** it runs at the end of weeks 51, 103 and so on (week 0 is the start), after the fast tick's invariants,
    with its own invariant check.
  - **What it does:**
    - chronic anxiety falls only with a rise in `basic_level`;
    - ties wear by their members' chronic anxiety, offset by `STAY-IN-CONTACT`, with bond energy untouched;
    - `dyad_age` counts years.
- **B6, the estimator** (`M1.A.4a`–`4d`, `4h`, `4k`, `M7.A.1a`; owner decision P4). Over three years of weekly history
  it takes:
  - the load-weighted mean and spread of `functional_level`;
  - the breadth in kinds of nodal event;
  - the self-directed share under load.

  `basic_level` rises only on a loaded, stable, broad window, never for a dependent person, and never falls in D-I.
- Six `[I]` constants.
- *(Superseded below by revision 17's criteria, which carry three nodal kinds.)* **No rises yet, and that is not a result.** No scenario has more than one kind of nodal event, so the breadth gate is
  never met. Without a coach, several people already show positive evidence. Whether the coach-free arm stays flat
  (`M11.C.17`) is untested until a window holds three kinds (open risk in `TODO.md`).
- **Review fixes:**
  - a refused `STAY-IN-CONTACT` no longer offsets wear;
  - the history test exercises a stressor, the channel and the week's own level;
  - the one-year contact window is tested;
  - `dyad_age` is logged;
  - contact counts in the year it arrives.
- **Records regenerated:** no Phase C verdict moved. Suite: 627 tests.

### Spec revision 16 and Phase D B3/B4: cutoff as rejection

- **B3, channel stamping.** A policy act is logged with its kind's channel, so `I-POSITION`, `STAY-IN-CONTACT` and
  the other `self` acts are `self_directed`, not `automatic`. This changes the log only.
- **B4, cutoff as rejection** (owner decisions P1 and P2; spec `M1.B.3c`):
  - **The tie stays open.** A `CUTOFF` raises the cutter's `rejection`, and the tie stays interactive.
  - **Refusal.** Moves sent to the cutter are refused while their rejection is above 0.3. A refused move does not
    reach the cutter, raises their rejection, and costs the sender anxiety.
  - **Lowering rejection.** `REDUCE_CUTOFF` lowers both members' rejection. Rejection also fades by 0.05 a week, and
    `RECONCILIATION` clears it.
  - **The pull.** A cut tie rests at no contact for both members.
  - **Declared cutoffs.** Ana–Bruno and Ravi–Dario start open, with both members rejecting.
- **Spec revision 16 (owner, 2026-10-11): graded rejection.**
  - **Why:** as first built, refusal at any rejection above zero left 80–88% of tie-weeks cut off.
  - **The values:** the refusal threshold is 0.3, not the 0.5 first proposed, because at 0.5 one cutoff refused
    nothing.
- **Measured effects** (Phase C family, seeds 7, 1 and 3, two years; `docs/implementation_plan_phase_d.md` B4):
  - **Cut tie-weeks** are near the pre-B4 level.
  - **The nuclear family runs more anxious,** because a cut tie is no longer silent.
  - **The `I-POSITION` ensemble tests** now run on the triad fixture, where preparation can happen (owner, option a).
  - **The `M11.C.29` test** now asserts that every withdrawal is legal without the hold.
  - **The dyad symmetry test** uses 200 seeds.
- Seven `[I]` constants were added and logged in `constants_changes.md`. The golden fingerprints moved (B3, B4).
- **Review:** `learning-qa` and a correctness pass found 10 items, all fixed with tests. The new `M1.B.3c` invariant
  checks each week that a tie is cut off exactly while rejection is held.
- **Records regenerated under the B4 engine.**
  - **One Phase C verdict moved:** `M11.C.32`, PASS → FAIL. Its `reached_peak` difference shrank to −0.02 ± 0.022,
    because `PREPARE` now usually fails on anxiety in both arms.
  - Every other verdict is unchanged.
  - Suite: 609 tests.

### Spec revision 15 (546 numbered requirements, 46 acceptance criteria)

The owner approved `docs/PROPOSAL — PHASE D DEFINITIONS.md`:
- **C.14:** "management technique" is a parent's weekly act toward the symptomatic child (firmness, permissiveness,
  punishment, talking it out).
- **C.23:** the expressed-emotion term (hostility, criticism and over-involvement, read through appraisal) and the
  suppression term (a factor on symptom accumulation, 0.5).
- **C.28:** the mobility term (the decay of the activation memory that holds the family's focus).
- **New `M11.C.46`:** a triangle relieves its seeker in proportion to the third's help, so the mutant
  `triangle-help-ignored` must turn it red.

### Phase D engine work, 2026-10-10/11

- **B1:** windowed event-store indexes, so a run's cost per week is flat. Seeded runs are byte-identical; a 40-year
  run of the twelve-person family takes 12.9 s, against about 110 s.
- **B2:** the batched TODO items.
  - The trace framing is read from the run.
  - The engine hash is the same on every Python version.
  - The readouts take `now`.
  - The readouts' act sets are in `config/bowen/pattern_acts.md`.
  - One shared C.44/C.45 filter.
  - Review fixes for each.

## [Unreleased] — 2026-10-10

### Phase D plan approved; spec revision 14 (545 numbered requirements)

- `docs/implementation_plan_phase_d.md`, approved by the owner with decisions P1–P5, P7 and P17 (§1a). Phase D splits
  into D-I (foundations) and D-II (the life course).
- Revision 14 applies them:
  - `M1.B.3c`: cutoff is rejection, not severing.
  - `M4.C.1c`: the reconnection pull.
  - `M5.F.2c`: the counterfeit flip is gradual, crossing at 0.5.
  - `M1.A.4k`: the estimator's form.
  - `M7.B.1`: chronic anxiety is fixed and moves only with a rise in `basic_level`.
  - `M7.E.4`: chronic anxiety wears on ties; the bond-energy accrual is withdrawn.
  - `M2.A.0b`: `t0` is week 0, not a slow tick.
  - `M11.4` gains `M11.C.14`'s null.
  - `M11.C.13` moves to Phase E.
  - The blank line splitting the criteria table is removed.

### Added — run visualization (`docs/PROPOSAL — RUN VISUALIZATION MAPPING.md`, plan V1–V7, owner-approved)

- **The pattern readouts are the owner's** (V1, V2; `docs/readouts_phase_c.md`, `src/bowen/readouts/patterns.py`).
  - A pattern is 3 or more acts in 4 weeks (`pattern_window` 8 → 4, `pattern_min_acts` 2 → 3, `pattern_pole` retired;
    logged in `constants_changes.md`).
  - No interaction is an emerging level of distance. Cutoff is refusal of approaches, not absence.
  - Pursuit reads as fusion, over/underfunctioning or projection. Projection and fusion are new readouts.
  - `current_patterns`: a pattern ends only when a new one emerges.
  - No criterion reads these readouts, so no verdict moved.
- **Spec revision 13** (V3): `M2.A` is the owner's family. Bruno and Toma leave; Viktor (Sofia's husband) and Dario (Ravi's
  brother, cut off) join. Leo is declared male. Teodor's `basic_level` is proposed at 36. `M2.A.1`'s fixture is now Iris and
  Dario. New `config/bowen/family_phase_d.md`, not yet run; the Phase B and C families are unchanged.
- **`docs/DECISIONS — CUTOFF SEMANTICS.md`** (V4): should the engine's CUTOFF become rejection? Held for Phase D
  (owner, 2026-10-10, option C).
- **`tools/famdiag_export.py`** (V5) writes a run as a fam-diag diagram (`docs/review/phase_c_famdiag_seed7.json`).
  - One fam-diag year is one 4-week block, 2001–2026.
  - The drawing rules are in `config/viz/famdiag.md`, outside the engine hash.
  - Every line traces back to the run's acts (`tests/bowen/test_famdiag_export.py`).
- **EPModel Fam-Diag** (V6): a separate copy of fam-diag (`../epmodel-fam-diag`, port 5174, private repository
  `dbgnvan2/epmodel-fam-diag`), so the owner's fam-diag stays as it is.
  - A line marked `hideAfterEnd` hides after its end year, and a dated triangle shows only inside its span.
  - Triangles keep their notes and events on load.
- **Opened in fam-diag** (V7): blocks 1, 13 and 26 match the export, and the trace agrees at blocks 9 and 26.

## [Unreleased] — 2026-10-08

**Phase C built** (steps 0–16 of `docs/implementation_plan_phase_c.md`). **Its acceptance gate does not pass**;
the per-criterion status is in `docs/phase_c_completion_report.md`.

### Added

- **Phase C closed, 2026-10-09, with its gate failing** (owner decision; report §13). Done: C.1, C.3, C.19, C.25,
  C.32, C.35, C.38. Partial: C.16, C.27, C.41. Not done, recorded as findings: C.4, C.5, C.29, C.42, C.44, C.45. §13
  lists what the next phase inherits. Open: plan §9's human reviews and Bruno's weekly fallback.
- **The approved failing-criteria decisions, carried out 2026-10-09** (results at the top of
  `docs/DECISIONS — PHASE C FAILING.md`, report §12). All test restatements, logged as post hoc:
  - **S:** every scripted act's ties are held open, in both arms, from week 0 until the act (`Forced(held_open=…)`,
    `legal_outcomes(excluded=…)`), so every act is made; the ensemble record reports each arm's acts not made. The
    approved first choice, the latest week legal in every seed, measured as week 0 for every criterion, where every
    member is on the chronic floor and no relief can show, so it was not used.
  - **Q2:** `M11.C.41`'s readout is the reactive choice less the legal set's reactive share (`reactive_over_chance`, centred
    on a chooser at random); both level limbs now pass. **`M11.C.29`** counts the weeks the third person's symptom is active.
  - **Diagnostics:** `s-scripted-weeks`, `d7-rest-state` (members with several ties rest well above the floor at every level,
    from the members' own acts) and `q3-act-effects` (a `TRIANGLE` relieves the pair; `M4.D.6e`'s cross-person
    credit counts the third's cost). Summary tables now show the minimum.
  - Verdicts: `M11.C.3` passes more strongly (pair −5.27); `M11.C.41` passes in 2 of 4 cells; C.4, C.5, C.29, C.42,
    C.44, C.45 and C.27 fail.
  - **Owner decisions on the results:** X1 a finding; `M11.C.5`'s FAIL reported; `M11.C.4`'s nodal event and
    `M11.C.27`'s reading moved to 104 weeks (one declared value, run once): no effect remains in either, and
    C.27's one passing cell no longer passed. The owner then decided `M11.C.27` is read week by week, gating on
    the week the act lands: 1 of 4 cells passes. `M11.C.41`'s readout counts the policy's draws only. The owner's account of triangling became a proposal
    (`docs/PROPOSAL — TRIANGLE RELIEF AND CREDIT.md`), built the same day (next entry).

- **The triangle change, approved and built 2026-10-09** (`docs/PROPOSAL — TRIANGLE RELIEF AND CREDIT.md`; spec
  `M1.C.1`, `M4.D.6e`, `M11.C.3`). A `TRIANGLE` relieves only its seeker, scaled by `third_help` (the third's calm
  times their alignment with the seeker rather than the partner); the partner is not relieved. Its credit leaves the
  recruited third out of the cross-person term. `M11.C.3` now reads the seeker's change and passes; `M11.C.42` and
  `.45` still fail. New named mutants `triangle-help-ignored` (survives on C.3) and `triangle-credit-cross-person`
  (no passing criterion to run on). `M11.C.27` is read week by week, and `M11.C.41`'s readout counts policy draws
  only.

- **The engine hash ignores comments, docstrings and blank lines in Python source** (`code_text` in
  `tools/ensemble_record.py`, learnings P38): fixing prose in `src/bowen/` no longer reruns every record (about an
  hour). Docstrings are found with `ast` and comments with `tokenize`, by source position, which is stable across
  the Python versions CI runs. Config `.md` files stay hashed whole: their headings decide which table a row is in.
- **`seeker-relief-removed`**, a deletion mutant that proves `M11.C.3`'s seeker readout (the transfer and roles
  mutants reach it only through the third's).
- **Report §11's qualitative claims are generated** with its numbers: whether an inversion reverses `M11.C.16`, and
  whether removing its grounds turns it red, are chosen from the mutant results.

### Fixed

- **Report §11 said no inversion reverses `M11.C.16`**, a constant sentence in `tools/c16_report.py`; since the
  triangle change `availability-level-inverted` reverses it, and `c16-grounds-removed` survives. The class goes
  back to the owner (`TODO.md`).
- **`scripted_weeks` read past its own shortened runs** once an arm read weekly readouts; the scan now stops the
  arm after scanning.
- **A record built in the run that produced its results did not match its own re-render** (2026-10-09):
  `run_cached` returned the fresh row in the child's key order rather than the cached, normalised copy.
- **`tools/c16_report.py` dropped everything after report §11**; it now replaces §11 only.
- **A scripted act was counted as made when it was placed, not when it was emitted** (batch review, 2026-10-09):
  an act toward the actor's open I-POSITION sequence's other is rewritten to `STAY-IN-CONTACT` after selection.
  `Forced.outcomes` now classifies each act from the run's records (made, not legal, owed a step, dead,
  rewritten, not reached); the scripted criteria are found by running the arms, not from typed lists.
- **The first Q2 readout scored 1/N for a chooser at random**, so it rose whenever the legal set shrank (batch
  review); replaced by `reactive_over_chance` before it was relied on.
- **A held-open tie's redraw replaced a declared absence's redraw in the same week** (latent): one redraw now
  carries both.

### Added (earlier)

- **Diagnosis of the failing Phase C criteria, 2026-10-09** (`docs/plan_phase_c_failing_criteria.md`, approved).
  `tools/diagnostic_record.py` and `tools/probes.py` write `docs/phase_c_diagnostic_record.md`; the owner's decisions
  are asked in `docs/DECISIONS — PHASE C FAILING.md`. The main finding is a test defect: a criterion's scripted act is
  skipped (not legal that week, usually because the policy already cut the tie) in 25-60% of seeds, and a skipped
  seed adds a difference of exactly 0 (C.3, C.4, C.5, C.27, C.29, C.35, C.42). Also: in the triad fixture members sit
  above their chronic floor almost every week with or without the spell (a calm-state hypothesis, D7 proposed);
  C.41's level limb conflicts with `M4.D.3a`; the learner credits `TRIANGLE` less than other acts under §8's roles;
  C.5's reaction grows over years in the seeds where the act was made. A first draft of the memo drew conclusions
  from medians over unmade seeds and was revised after the batch review.
  `M11.C.7`, `.13` and `.14` moved to Phase D. Report §3.6's account of C.29 was wrong (the third person is
  symptomatic in every week, not none) and is corrected.
- **Cached records, 2026-10-09.** The mutation and sweep records cache each result under the engine hash (engine,
  config, `RULE_KEYS` and `tools/mutant_runner.py`, the child that computes results) plus the mutant's or setting's
  own edits (`tools/record_cache.py`, `docs/records_cache/`). A record's test re-renders it from its cache and
  compares, so labels, rendering and judging change without a rerun, one new mutant reruns only itself, and an
  engine change still reruns everything. Re-rendering takes under a second, where any tool edit had forced a 35-45
  minute regeneration. The seeded cache was checked by rerunning two mutants fresh: the record and cache came out
  byte-identical. The occupancy record no longer hashes the whole mutant list, the sweep no longer hashes the
  mutation tool, and report §11 is generated by `tools/c16_report.py` with a test, so a stale number fails the suite.
  An errored row is never cached (it reruns next time), a failure of the machine running the child (a broken worker
  pool, memory, the OS) stops the run instead of becoming a cached red, and the cache is flushed after each mutant so
  a late failure loses no finished result.
- **Fifth csdp sweep fixes, 2026-10-09.** A mutant whose replacement text is broken is **broken**, for every kind,
  and never proof; an engine exception under a mutant stays a red. The occupancy record gives each span its own
  member-week count. The sweep record renders a row without readouts. The band test checks both sides and the
  comparisons. Report §11 no longer says the cited rules "do not carry it without availability": either set alone
  keeps a passing result, and only removing both removes it.
- Phase C of the agent model: revision 11's two-sided appraisal and standing load, the belief store (`M9.8`),
  move physics, the policy (two channels, `WITHHOLD`, competing urges, keyed selection), the learner, the
  `I-POSITION` state machine, the external agent and landed contact, `M6.4`'s ledger with `M6.I.6` asserted every
  tick, sink allocation, pattern readouts (`docs/readouts_phase_c.md`), the ensemble runner (paired signed-rank,
  Holm, adaptive stopping), the engineering gates, and the fixture families.
- The criteria as paired arms (`src/bowen/ensemble/criteria.py`) and three generated, hash-stamped records:
  `docs/phase_c_ensemble_record.md`, `docs/phase_c_mutation_record.md` (named, deletion, sign-inverted, a joint
  level-blind, and representation mutants, each run on a temporary copy of the repository) and
  `docs/phase_c_sweep_record.md` (plan D9). Default-suite tests fail when any record is stale.
- Constants frozen 2026-10-07 (`config/bowen/constants_frozen.md`), before any criterion ran.

### Results

- At the frozen constants, as first built: 6 criteria pass and are proved by a mutant (`M11.C.1`, `.19`, `.25`, `.32`, `.35`, `.38`);
  `.42` and `.45` pass but reverse in the sweep; `.27` and `.41` pass in some cells; `.16` is undetermined at the
  cap; `.3`, `.4`, `.5`, `.29` and `.44` fail; `.7`, `.13` and `.14` are not built.
- `M11.5` corrected from the real mutants and dated in the spec: `M11.C.1` and `.38` first proposed as joint premises (decided later the same day: they stay premise, with the nine rules listed); `.45` proposed
  as a premise, withdrawn when it stopped passing after the `TRIANGLE` decision (it stays composite).

### Changed after the freeze, and reported

- Three numerical fixes to the policy (a shift-invariant softmax, no `WITHHOLD` weight with no automatic act, the
  withheld act drawn from within-channel weights), each with a unit test proved by reverting it.
- `M11.C.25`'s readout, which could not fail, corrected after its mutant survived.

### Fixed

- From the Hermes `learning-qa-sweep` gate of 2026-10-08 (`docs/cycles/2026-10-08_phase-c-qa-gate.md`, APPROVED with
  1 medium and 3 low findings): the mutation and sweep records now hash the tools that generate them (medium);
  coverage reads `NOT_BUILT` from the criteria module, not the record's prose; the belief store's
  "no `true_counterpart`" check is an AST name check tied to the function, with its own mutation test. The fourth
  (low) finding is in `TODO.md`.
- From the second gate (`docs/cycles/2026-10-08_phase-c-qa-gate-2.md`, APPROVED), and the first gate's fourth finding:
  every record's hash now covers `tools/ensemble_record.py`, whose `RULE_KEYS` feeds every verdict (A, medium); the
  sweep staleness test imports the sweep tool's `HASHED_TOOLS` (B, low); the criteria's declared settings moved,
  unchanged, to `config/bowen/criteria.md`, and `M11.D.2`'s magic-literal scan covers `criteria.py` (4, low). All
  three records regenerated; apart from the hash they are byte-identical. The suite then held 483 tests.
- From the third gate (`docs/cycles/2026-10-08_phase-c-qa-gate-3.md`, APPROVED, three low findings): a static test
  ties the criteria's `REQUIRED` settings to what each arm reads, in both directions; the `criteria.py` docstring
  no longer restates the spell's numbers; the mutation tool loads once in the tests. The suite then held 485 tests.
- From the fourth and fifth gates: a stale suite count in this file fixed, and the tests load both tools under their
  real module names (`tools.mutation_record`, `tools.sweep_record`).

### Decided

- **`TRIANGLE`'s roles (`M1.C.1`), 2026-10-08.** The target is the third party the act recruits and stands outside;
  the inside pair is the sender and its most strained partner. The spec already read it this way in `M1.F.1b`,
  `M11.C.3`, `M11.C.42` and `M6.4`; step 5 had built an alliance reading instead. A rule change after the freeze,
  made from the spec's text and reported (`docs/phase_c_completion_report.md` §8). `M11.C.3` now passes and is
  proved, including by a mutant that restores the old reading. `M11.C.42`, `.45`, one cell of `.27` and one of `.41`
  stopped passing; they are reported, not recovered. 7 criteria now pass and are proved, 7 fail.
- **What `M11.C.42` and `.45` test, 2026-10-08**, decided from the spec's text and committed before rerunning.
  C.42's baseline arm now makes the third member unavailable for one tick, instead of scripting `STAY-IN-CONTACT`,
  and its readout counts the whole pair's triangles. C.45's readout is `TRIANGLE` selections per person-week.
  Both still fail: C.42 is +0.26 (p 0.084), C.45 +0.0020 (p 0.20). Report §9.
- From the eighth gate (APPROVED, four test-strength findings): the absence and readout tests for `M11.C.42` and
  `.45` now compare every record before the absence, check that no triangle can form during it, and check the
  pair-wide count. Tests only; no record changes.
- **`M11.C.1` and `.38` stay premise, 2026-10-08.** No new class: every level-reading rule states the direction, so a
  pass is not a finding. The spec's `M11.5` rows now list the nine rules and name the joint level-blind mutant as the
  proof; plan §3's single-rule mutant cannot prove them.
- **The ensemble precision rule, 2026-10-08.** The stopping rule compares the half-width with 0.25 of the pooled sd of
  the two arms, not the baseline arm's (`ensemble_precision`'s unit changed, logged post hoc). Decided and committed
  before rerunning. `M11.C.16` now passes, but survives its named learner mutants, so its direction does not come
  from learning; `M11.C.41`'s [light: lower level] cell now fails instead of undetermined. Report §10.
- **`M11.C.16` stays composite, 2026-10-08.** The first check ran an availability inversion that saturated at the layer
  clamp over C.16's levels, so it removed availability instead of inverting it (csdp sweep). It was replaced, and all
  nine level-reading rules now run as deletion and inversion mutants on C.16. Inverting availability now cancels C.16
  (both readouts' intervals span zero) but does not reverse it; removing it halves the result. That inversion equals
  the deletion at level 20 and below, where the lowered arm spends 40.6% of member-weeks under it over the run and
  45.2% over the entropy's window (`docs/phase_c_level_occupancy.md`, a new generated record), so the cancellation
  holds for it alone. Removing the rules C.16's criterion row cites, `M4.C.1a` and `M1.C.3a`, leaves it passing
  (`c16-cited-grounds-removed`); removing them with `M4.D.3a`'s availability turns it red (`c16-grounds-removed`). Composite rests on
  reading only availability as naming a narrower repertoire; report §11 says why that reading is contestable. A reclassification to
  premise, which read that cancellation as `M11.1d`'s flip, was withdrawn the same day by owner decision. `M5.D.3`'s hold
  capacity never acts in C.16's runs. The learner changes the result's size, not its sign, so the row's rationale and
  mutation clause are not met, and coverage keeps it partial. The mutation record now gives each mutant's mean
  difference and half-width, marks a red mutant **reversed** only when every gating readout's interval lies on
  the opposite side (report-only and equivalence readouts do not gate), and marks a mutant whose replacement
  text is broken (an unimported name) **broken**, which coverage does not count as proof; an engine exception under
  a mutant stays a red. The sweep record says **opposite sign** for a sign-only
  reversal. Report §11.
- From the tenth gate (APPROVED, four low findings): `M11.5`'s premise definition now admits a result several rules
  state redundantly; the `M11.C.38` note says no single-rule mutant reds all three pairs (four of six red none); an
  "exactly zero" claim no committed record holds was removed; this file's `.45` entry corrected.

## [Unreleased] — 2026-10-06

Spec revision 10 approved, the Phase B plan approved, and **Phase B built** (`src/bowen/`).

### Added

- **Phase B of the agent model** — objects and identifiers, keyed draws, the event record and queue, the in-run event store, log records, the reduced family and the 40-week script from markdown, the M8 predicate, activation and visibility, the standing load, the base appraisal (`M4.C.1` only), structural events, consolidation, the M6 invariants (`M6.I.6` held back), the nine-step weekly loop, the persistence sink, the log header, the deterministic renderer, and `python3 -m src.bowen.run`. 291 tests after the `learning-qa` review's fixes; every automated Phase B exit criterion passes and is mutation-proved (`docs/phase_b_mutation_record.md`). Report: `docs/phase_b_completion_report.md`; coverage: `docs/spec_coverage.md`.
- `tools/mutation_gate.py` and `tools/spec_coverage.py`.
- Constants frozen 2026-10-06 (`config/bowen/constants_frozen.md`).

### Changed

- **Spec revision 12 approved, 2026-10-07** — the Phase C prerequisites, with the Phase C plan (`docs/implementation_plan_phase_c.md`, all recommendations as written). `M6.I.6` restated with a stock-and-flow table (`M6.4`); `M4.B.2` scoped row by row (`M4.B.2a`) and `M9.8`'s belief store moved into Phase C; `M11.C.17`, `.18`, `.20` moved to Phase D and `M7.D.1` into Phase C; `M5.D.6`'s next day is the next tick, with `M5.D.9` for multi-tick sequences; `M11.D.16`'s equivariance in distribution; `PROVOKE` added (`M5.B.6`, and to `config/bowen/event_kinds.md`); the review's Phase C contradictions resolved. Explainer §17.3 and six drifted passages corrected. 542 requirements, 45 criteria; Phase C gates 19.
- **Spec revision 11 approved, 2026-10-06** — set, random and emergent layers (`docs/proposal_rev11_emergence.md`; `model_explainer.md` §19). `M4.D.6a` reversed: the automatic channel learns from felt relief. `M4.C.1` re-derived as two-sided deviation from a felt contact optimum, with the standing load as its "too little" side. Core moves are single acts and Bowen's patterns are readouts (`M5.A.1a`). Rules that wrote an outcome made emergent; `M7.D.3` and `M7.E.3` withdrawn as mechanisms. `M4.C.10` (calmer contact lowers anxiety), `M4.B.3` (no look-ahead), four new criteria (`M11.C.42`–`M11.C.45`), and `M11.5`'s premise/composite/check classification. 538 requirements, 45 criteria. A `learning-qa` review's eleven findings were fixed the same day: stale references to the old `M4.D.6a`, `M4.D.3a` brought in line with amended `M4.D.3`, mutation targets moved to `M4.C.1a`, the cutoff cost made to build over time so `M11.C.4` holds, `tools/spec_coverage.py` gaining `exclude_evidence` and two guards (293 tests). Phase B code built against revision 10's `M4.C.1`, `M4.A.1` and `M1.C.3` no longer conforms; reworking it is a Phase C item.
- **Spec v2.0 revision 10 approved by the owner, 2026-10-06** — 519 requirements over 17 modules, 41 acceptance criteria. The terms are recorded in the spec's revision-10 section under *Approval*: the external review of 2026-09-23 is carried into the implementation plan rather than fixed first (its recommendation 9), `M11.4c` and `M4.G.3` kept, `M11.E`'s proposals accepted, revision-11 material held.
- **Phase B plan approved, 2026-10-06** (`docs/implementation_plan_phase_b.md`), all nine recommendations as written. The spec now holds 525 requirements: `M1.A.21` (`sex`), `M1.A.22` (`household_id`), `M2.3a` (the Phase B instance), `M3.C.2` (latency in whole ticks), `M4.G.2a` (`M6.I.6` held back until restated), `M14.A.4` and the filled `M14.A` register. `M4.C.1` moves into Phase B. Ravi's `basic_level` is 39, within `M2.A.0e`'s tolerance of Marta's 40.
- Phase B step 5: `M2.A` gains a `Sex` column and `M2.A.0i` (526 requirements). Ravi is male and Nadia and Pia female by owner decision; the rest follow the table's relationship words; Leo and Dr Halim are not stated until Phase D.
- `test_m11d17_register_has_no_orphans` and three mutation checks added to `tests/test_spec_consistency.py`; the suite-size guard now counts tests in subdirectories.
- `M11.F.10` added: the model may not be called a "virtual family" without `M11.F`'s framing. `model_explainer.md` §1.1 gives the comparison with virtual-cell models and what a virtual family would need.

## [Unreleased] — 2026-08-28

The sixth corpus source decided and applied, plus two modules that came out of working through what the model may and may not be asked. Documentation only; no code. Spec revisions 4, 5 and 6.

### Fixed

**From a failure-pattern sweep of the batch, 2026-08-28 — fifteen findings, all verified against the files and all fixed.** Four were high:

- **`M16.B.3` required logging to be a pure observer, contradicting `M16`'s own preamble twelve lines above**, which makes an agent's delayed log an *input* to a mechanism. Worse, its test gated **only Phase B** — the phase with no policy and no external agent, where it passes vacuously — and was not re-run in Phase C where `M16.D` makes it false. Split into an **event store** (always present, causally load-bearing) and a **persistence sink** (optional, pure); `M16.T.3` re-listed in Phases C and D, and `M16.T.6` added as its converse.
- **`M11.2` bounded ensemble testing at `M11.C.16`** while the table grew to 33, leaving seventeen criteria — including three distributional claims a single run cannot establish — outside a bound nobody had moved. Restated over the whole set.
- **`M11.3` shipped with eight counterexamples in the same commit.** It required every criterion to name a stressor; 8 of the 9 criteria added alongside it name none, and `M11.C.31` *requires a calm family by construction*. Scoped to discriminating criteria, with the conservation, identifiability and null classes named as outside it.
- **Seven `M11.C` criteria and `M16.T.4` carried a phase and gated nothing**, so Phases C and D could each be declared done with them unwritten — among them `M11.C.17`, which the spec calls "the model's strongest negative test", both estimator discriminators, and the only guard stopping an agent reading another agent's private hops. All eight listed; `M13.2a` and `M11.D.11` make the correspondence mechanical.

And eleven more: `M10.C.4`'s twelve magnitudes had **no permitted home** — forbidden in config, underivable, and `M11.D.2` requires one or the other, so the violating route was also the compliant-looking one (`M10.C.4b` names a checks module; `M11.D.12` guards it). **Nine prohibitions were written `No X MUST Y`**, which under §0.3 states a permission — including `M11.F.9(c)`, the clause the document marks as correctness rather than framing (`M11.D.13` now guards the form). §0.3 claimed every MUST carries a test: 616 MUSTs against 44 tests, restated honestly. `M0.4` pointed at an exception list that did not exist (`M11.4` now is one). `M11.F.5` never existed and the gap read as a withdrawal. `M11.C.25` and `M11.C.17` are nulls that pass when underpowered (`M11.4a`). `M16.C.3`'s rationale was stale inside its own commit. `M1.F.9` never named `binder_unavailable`. And the published proposal carried revision 4's requirement count through revisions 5 and 6 while `_STATUS.md` asserted the set was consistent.

- **`M11.F.9(c)`'s stated mechanism was wrong, though its conclusion stands.** It said fitting absorbs the model's error "on one side only"; under `M0.4` both arms share one parameter set, so nothing is absorbed asymmetrically. The real failure is **loss of independence, non-exchangeability across a regime boundary, and non-identifiability** — 60–90 free constants against a single realisation, so many parameter sets reproduce the history and disagree about the counterfactual, making the bias in the difference *unbounded* rather than merely unknown. A stronger objection than the one it replaced. The line "the better the fit, the less the counterfactual can be trusted" is now scoped to the overfitting regime, where it holds, rather than stated as a general law, where it does not.
- **`M5.D.4` was inverted, and had been since it was written.** The requirement said *anger* is the gate admitting the `I-POSITION` sequence to its peak. Ch13 says the opposite — "when he is finally able to maintain his course **without getting angry** at the opposition, the opposition does a final intense emotional attack" — and mentions anger exactly once in the whole chapter. `docs/theory/ch13.md` had it right in the body and wrong in the **heading** above it ("Anger gates the escalation"), and the heading is what reached the spec. Corrected in all three places, with `M11.C.32` added to hold it. Kerr 1988 (`FE04.4`) and Bowen 1988 (`FE11.19`) both agree with the chapter, about the same person's anger.
- **A phase exit condition that could not be checked.** Phase B was done when "the event log reads correctly" — an adjective naming no requirement, no test and no artifact, in a table where every other condition names one. It now names `M16.C.2` and `M16.T.1`.
- **Module ordering.** `M14` had ended up after `M15`.

### Fixed — the corpus-fidelity sweep

**Every quoted passage in the spec was checked against the primary sources** — 214 quotations, 160 verbatim in the corpus, 52 legitimate self-quotes of project documents, 0 unexplained. Five cold agents on a different model, reading the source texts and forbidden the extractions, plus a deterministic string match by the author.

- **One fabricated quotation.** `M1.A.4e` placed a paraphrase inside quotation marks and attributed it to the source as its own opening line; it was the evidence for revision 8's demotion. Fixed, and the demotion restated to what the source carries.
- **Modal inflation, the dominant class.** `can… or` → `MUST` with one branch erased; `suggested` → `required`; `perhaps having a role` → `MUST be`; `such as 55 or 60` → a bound; and a **rate** claim hedged with *"sort of"* turned into a **ceiling**.
- **Three scope shifts, introduced at revisions 7 and 8** — a subgroup phrase applied to a whole band, a societal trend grounding an individual mechanism, and a consequence presented as a quotation's when the corpus arguably says the opposite.
- **Evidence framing.** "An experiment and an independent replication" were two clinical accounts, one of them the author confirming what he had been told. "Two families — home movies in both" was one family. "Three statements, every one about the mover's anger" was two.
- **Attribution.** An ASR transcript attributed to Bowen personally against the project's own rule; a requirement grounded on **Andrew Solomon's** words, quoted by Kerr.
- **Propagated** to `model_explainer.md` and the published proposal, both of which carried the same band-scope over-read graded `[T]`.

Hedge-level findings are in `TODO.md` with reasons rather than fixed.

### Changed — what the estimator may read

Three points from the project owner, each checked against the corpus and each upheld.

- **The four "assessment domains" were genogram intake fields, promoted to an instrument.** `KS11.7`'s own opening line says they are what *"the family diagram records"*. Occupation, health history, longevity, courtship length — these are **outcomes**, they are **base-rate common** (a divorce or a job loss carries almost no information about who had it), and they confound the **load on a person** with their functioning. Demoted to a weak prior at import; they may no longer set an estimate. `M1.A.4e`
- **The corpus already corrected this and the spec had not carried it through.** `KS06.8`: *"In terms of solid self, one's personal life is where the rubber meets the road"*, and it is the intimate relationship *"and not how they manage themselves in their work lives"* that determines offspring level. The requirement cited that correction **underneath the table it invalidates**.
- **A three-tier reading replaces the flat list** (`M1.A.4g`): a single adverse event is mostly **load**; a chronic self-maintained pattern reads **functional** level, load-corrected; only a position held under pressure reads **basic** level.
- **The discriminator is which channel was selected under discomfort, at comparable load** (`M1.A.4h`). `M4.D.1a` already carries it — the automatic channel discharges anxiety *now*, the self-directed one holds a position *through* discomfort, and the mixing weight is a function of differentiation. Reaching for an affair, a substance or distance is the first; carrying it is the second. Ch13 states the affair case as mechanism. **Adhering to convictions under pressure and carrying discomfort are one observable, not two** — `M5.F.5` says so in a sentence — and every piece is already in the model.
- **The system's reaction to a symptom is now an estimator input, and it is the best-controlled one** (`M1.A.4i`, `M11.C.34`). One symptom, N observers, one moment: load is held constant **by construction**, so the comparison across members needs no correction. It is also the only strong reading on members who are *not* symptomatic.
- **`M1.E.2`'s three reaction detectors are generalised off the coach to every person** (`M1.A.19`). Ch10 extends the neutral third to "any third person", and the proposal's §10 already recorded the conclusion — *"one rule seen from four angles… Implement it once"*. One of the four had been implemented.
- **The circularity boundary is stated precisely rather than as a ban** (`M1.A.4j`): symptom **magnitude** may not feed the estimator, because the model generates it from level; the **channel selected under discomfort** must, because it reads policy rather than accumulated load. An earlier draft of this had banned symptoms outright, which would have discarded the best signal in the model.

### Fixed — the scale is a continuum

- **`M1.A.3` implemented a cliff where the corpus describes a band.** It required *"exactly one behavioural transition on `basic_level`, at 50"*. Raised by the project owner — Bowen's quadrants are bands across a continuum, not shift points — and upheld against the primary: Ch16's own wording is graded ("**begins** making **a few** decisions of its own"), Ch21 has **no regime change at 50 at all** and calls the transition **explicitly gradual**, the band scheme is present/absent/present across 1966/1972/1976, Kerr gives different edges entirely, and Bowen "subsequently dropped the term scale". The licence is now a **slope**: continuous in level, with the joint-decision domain lagging every other domain at every point. That ordering is the mechanism; 50 is a label.
- **Removing the step must not flatten the model** (`M1.A.3c`) — the corpus asserts **non-monotone** structure over the range, and a smooth ramp would erase it.
- **A band edge may no longer be a branch point anywhere** (`M1.A.3d`) — 25, 50, 60, 70, 75 — so the error cannot recur at the other four numbers. The capacity floor at 25 is regraded to a continuous falloff (`M1.A.4d`), applied **on the principle rather than on a fresh reading**, with its location left standing pending one.
- **"Regime boundary" corrected to "turning point"** throughout the sign-flip argument. The conclusion is unchanged — mis-estimating which side of a turning point you are on still reverses the sign — but the picture is a roll-off, not a cliff, and near the turn the effect is small and noisy.

**This is the second instance of one failure**, after `M5.D.4` at revision 4: a source *description* firmed up into a mechanism the source does not assert. Both were found by reading the primary rather than the extraction, and **neither was catchable by a regex** — a sweep for others is a reading task, logged in `TODO.md`.

### Fixed — second pass

**The fix commit was swept as its own range and returned fifteen more findings**, four high. That is the
expected result, not a surprise: a fix commit is the least-reviewed code in a change, and the project's
catalogue says a fix pass introduces defects at roughly the rate it removes them. Four of the fifteen were
**the first sweep's own diagnosed failure mode recurring inside its fix**:

- **The phase-gating guard went vacuum-green on a cosmetic reformat.** It accepted a phase cell only if it
  was exactly `B`/`C`/`D`/`E`; bolding all 32 remaining cells made it check **zero** criteria and pass. One
  row was already being dropped silently. It now asserts it read every gateable row.
- **The inverted-prohibition scan was narrower than the class it named** — capital `No` plus a separately
  bolded `**MUST**` — so four instances survived, including **`M11.D.12`, the guard written for another
  finding, in the form this finding forbids.**
- **Two corrections landed in the spec only.** `model_explainer.md` still said logging is a pure observer,
  which corrected `M16.B.3` contradicts, and `_STATUS.md` still carried the repudiated "on one side only"
  mechanism — while asserting the document set was consistent.
- **`M11.4`'s "exhaustive" exception list was contradicted three times in the same document**, once by a
  table fifteen lines below it. Same shape as the `M11.3` defect it was written alongside.

Also: `M16.T.6` required disabling what `M16.B.3` forbids (now an experimental override, the pattern
`M10.C.4b` already uses); all six `M16.T` criteria named tests that do not embed their ID, against §0.4;
`M10.C.4b`'s third home cleared `M0.3` and `M10.1` but not `M10.B.1` or `M11.D.2`, which are now scoped to
engine-read constants; `M11.3`'s "discriminating" was undefined and named a table column that does not
exist; §0.3's coverage figures were measured at the parent commit and stale on arrival (now asserted by
`M11.D.14`); `M11.C.8`'s Test cell is `—`, a counterexample to §0.3's replacement claim; and
`M11.F.9(c)` said the bias was "**unbounded**", which does not follow — the difference is *unidentified*,
and an unsupported strengthening is precisely the error §10 records the first corpus pass making.

Four low findings were graded and **backlogged rather than fixed**, with reasons, in `TODO.md`.

### Added
- **`tests/test_spec_consistency.py`** — ten guards over the document set, each mutation-proved by reverting the fix it protects and confirming red. They assert what a failure-pattern sweep had to find by hand: IDs unique and machine-extractable, every cross-reference resolving in seven documents, no prohibition written in the inverted form §0.3 reads as a permission, every criterion gating the phase its own table row names, `M11.2` stated over the whole criteria set rather than a numeric range, and the requirement and suite counts in `README`/`CLAUDE.md`/`_STATUS.md`/the proposal reconciling to the artifacts they describe. They read documents only — nothing imports `src/`.
- **`M15` — the family-diagram import contract.** A Phase E capability specified early, because the diagram application that would feed it is under the project owner's control and its export format is cheaper to fix before it exists than after. Structure imports as **values**; ratings import as **ranges**; the readout is an **envelope**, never a point direction. Four adapter traps named, including a warmth rating wired to a valence-blind `investment` (which would invert the sign on exactly the conflicted families the model is for) and a genogram's single "distant" line, which covers both a rupture and a resolved low-contact tie — the discrimination `M1.B.3` calls the most consequential in this part of the theory. Fixture mode included: anonymised real topologies beat hand-built ones wherever a criterion depends on asymmetry.
- **`M16` — the run log and the readable trace.** Six requirements already depended on a log and none of them said what one is. Effects and the selection rationale are recorded beside the events, so the log is a causal trace rather than a list; belief writes are tagged apart from ground truth; the engine emits and the caller persists; logging is a pure observer. A **deterministic renderer** — template, no language model — is built in Phase B, and a per-agent **delayed view** serves the fourth form of a landed coaching contact, defaulting to the corpus's six months.
- **`M11.F.9`** — three clauses on speaking about a real family. The third is a *correctness* requirement, not a framing one: no counterfactual may be reported from parameters tuned to reproduce a known history, because the parameters are then chosen conditional on the trajectory and the fit is non-identifiable, so the counterfactual difference is not identified by the data. *(This entry originally carried the mechanism as "on one side only", which the re-sweep showed to be wrong; see Fixed.)*
- **~20 requirements from *Family Evaluation***, the largest being that the belief layer becomes a **channel** (`M9.6`, `M9.7`) rather than a parallel store — chronic anxiety runs on what *might be*, and Bowen's own correction of "projection" has the transfer running through descriptions.
- **`M10.C.4`** — twelve magnitudes stated in the corpus, admitted as **checks on an output** and never as parameters, with `M10.C.3a` clarifying that the existing prohibition is on the parameter and not on the quantity.
- **`M11.G`** — the eight-component family evaluation readout, with the source's ninth and tenth components (therapeutic focus, prognosis) explicitly barred.
- **Nine acceptance criteria**, `M11.C.25`–`M11.C.33`, and five for the log, `M16.T.1`–`M16.T.5`.
- **`docs/model_explainer.md` §17** — what the model can be asked, and the two ways the answer goes wrong. Marked as method rather than corpus.

### Changed
- **`docs/DECISIONS — FAMILY EVALUATION.md` closed.** All nine items decided 2026-08-27; the file is now the record of how each was decided.
- **`M1.A.11c`** — constitution sets the symptom channel as a *prior*; a family-focus term can shift it; a constitutional-strength term can override the shift. `model_explainer.md` §3.9 is no longer provisional.
- **`M1.A.3b`** — the transition at 50 keeps its licence implementation and gains an awareness *readout*. Neither is reduced cognitive capacity: what differs is the strength of the emotional circuits over the cognitive ones, and a low-level agent argues just as fluently.
- **`M7.A.1a`** — the financial-dependence gate now binds the slow clock, not only the `I-POSITION` move.
- **`M12.2` reviewed and left standing in full.** Kerr 1988 names the *locus* of the first unknown, not the rule; its own hedge is "determined largely by".
- **`M12.5`** — power and punishment barred as mechanisms. The only entry in that module stated as a prohibition by the author rather than inferred by the project.
- Spec is **427 requirements over 16 modules**, 34 acceptance criteria, 0 duplicate IDs, 0 dangling cross-references. Every `M…` reference in seven documents resolves, and all of that is now asserted by `tests/test_spec_consistency.py` rather than by a script run once.

### Learned
- **A summary heading is a lossy artifact, and a requirement sourced from one can invert without anything looking wrong.** The `M5.D.4` inversion above is the worked instance: the quoted sentence was right, the heading over it was not, and only the heading travelled.
- **Initial conditions are not nuisance parameters.** Invented constants cancel between two arms of a counterfactual because they do not interact with the intervention. Initial conditions are *what the intervention acts on*, so their error crosses a regime boundary and flips the sign instead of shifting both arms together. Five such boundaries are named at `M15.D.4`; one of them, `M5.C.1a`, is a SAFETY property with recorded harms. **Direction is the least robust output under mis-specified inputs, not the most.**

## 2026-08-22 — corpus alignment

The corpus-alignment batch. All 22 chapters of *Family Therapy in Clinical Practice* have now been read twice and the project's documents have been brought into line with what the source actually says.

### Added
- **`docs/model_explainer.md`** — every object, field, move, gate, invariant, clock and test in the agent model: what it is for, what it does, and which chapter it implements. 216 claims, each graded `[T]` textual / `[M]` mechanism / `[D]` direction / `[#]` stated quantity / `[I]` invented / `[X]` contested.
- **`docs/bowen_agent_model_spec_v2.md`** — the v2 specification, v2.0-draft, **awaiting approval**. 212 requirement definitions over 14 modules; 16 acceptance criteria and 10 engineering criteria, each with a named test and a mutation target; four criteria flagged as not code-testable in Phases B–D with a human-review proposal each.
- **`docs/theory/_RESOLUTIONS.md`** — the two contradictions carried out of pass 2, both resolved against the primary source.
- **`docs/theory/`** brought under version control: 22 chapter extractions, the ledger, the convergences, the corrections and the status file.
- Ledger entry **L21.11**, surprise as a third secrecy mechanism — present in the source and in neither pass.
- Spec requirement **M11.D.10**, requiring every ID to be machine-extractable.

### Changed
- **`docs/theory/_LEDGER.md`** rewritten and made self-contained: all 109 pass-1 IDs preserved and re-verdicted, 40 new pass-2 entries, all ten correction batches folded in. Withdrawn findings kept as tombstones with their reason.
- **`docs/agent_model_proposal.html`** corrected throughout rather than annotated. The undifferentiation budget has three sinks, not four. Reactivity is derived, not stored. The move repertoire is no longer claimed closed. The event record carries source position, route, fidelity and latency. Acceptance tests: eleven became fifteen. **Phase B's four-person nuclear spike is withdrawn** — cutoff with the families of origin is an *input* to intensity, so a model closed at the household cannot compute its own driving term.
- **`docs/bowen_individual_family_model_spec.md`** frozen as a v1.2 historical record of the grid engine, with a header table of the nine things the corpus contradicts in it. It is not the base for v2.
- **`CLAUDE.md`** rewritten for the pivot. Axiom 1's vectorisation mandate is scoped to the frozen grid engine — at N ≈ 12 it pushes the design the wrong way.
- **`README.md`** expanded from two lines.

### Fixed
- Two flaky tests, `test_spouse_dysfunction_asymmetric_penalty` and `test_triangle_mechanism_attaches_and_releases_circle`. Each constructed an unseeded `Simulator` and asserted on a family whose randomly drawn size did not satisfy the code path's precondition; the first was measured failing 6 times in 12 runs, with the asymmetry it names never exercised on a failing draw. Verified 36 passed across 12 consecutive full-suite runs; the suite is 37 after the working-directory guard below.
- `.gitignore`, whose last line read `.DS_Store (for Mac users)` — a literal filename, so `.DS_Store` was never ignored.
- Spec-internal ID drift: invariants were defined as `I1` and referenced as `M6.I1`, criteria as `C4` and referenced as `M11.C4`. A traceability scan would have resolved 19 of 216 references to nothing and reported a lower count rather than an error.

### Withdrawn
Five mechanisms proposed on the first reading and removed on the second, recorded rather than deleted because a merely-absent mechanism gets re-invented: the permission scalar, ally cancellation, the durable projection target queue, distance as a depletable relief stock, and "content is noise" as an argument for anything.
