# Version 2 stage 3 milestone review — the triage, 2026-10-28

The review is `docs/cycles/2026-10-27_milestone-review_v2-stage-3.md` (mini2, branch `review/2026-10-28-v2-stage-3`,
commit `78d5dd6`, fast-forwarded into `main` on 2026-10-28). It found 0 blocker, 1 major (MR-1) and 21 minor findings;
none was refuted. The owner answered the review's six questions on 2026-10-28 through `AskUserQuestion`, Q1–Q5 one at
a time, each as recommended; Q6 was taken as recommended under the owner's instruction of the same day ("run
everything using your own recommendations"):

1. T-131: option (b), a separate ledger for `M1.A.5f`'s transfer, netted out of `M7.E.1f`'s `d`; the P2 proposal's
   `KS00.9` citation corrected now.
2. T-133: option (b), anger checked at `DEFINE`, share 1.
3. T-132: the second kind of so-called I-position recorded as not modelled (spec revision 40; T-132 closed).
4. MR-2: not an exception; `M5.F.4` and `M5.F.2a` reworded and the spec swept for other transcript phrases (revision 40).
5. MR-1: `M11.C.16` stays composite; MR-1's generator fix is T-148.
6. MR-4: the reading list's per-step rows cut to pointers at the plan's status line.

Fixed at revision 40 (wording only, no file under `src/bowen/` or `config/bowen/` edited, engine hash `2a3b84e2d89a`
unchanged): MR-2, MR-3, MR-4, MR-5, MR-6, MR-7, MR-8, MR-9, MR-11, MR-12, MR-13 (two documents added to `CITING_DOCS`
and `test_mr13_current_engine_hash_is_the_one_the_documents_name`), MR-14, MR-22, and MR-10 in the T-130 page. The
items below are what is left. The hash-moving ones (T-131, T-133, T-145, T-146, T-147, T-149) go in one batch, so the
records are regenerated once.

- [ ] T-148 ⟨now⟩ **`M11.C.16`'s fixed prose against its regenerated table (review MR-1, major; the owner kept the
  class composite, 2026-10-28).** `tools/c16_report.py` writes three sentences of `phase_c_completion_report.md` §11 as
  literal text (the learning bullet, "No other single rule removes the result", the "What does not hold" bullet) that
  the current mutant table contradicts: `availability-and-learner-removed` is reversed (+0.002 ± 0.002) and
  `steepness-inverted` is red. Rows :49 and :541 of the report and spec `M11.5`'s `M11.C.16` row misstate which mutants
  survive. Fix: make each sentence conditional on the mutant results, as INVERSION is, with a branch for a red or
  reversed single-rule inversion; generate or test rows :49 and :541 and the `M11.5` row from the same results; correct
  the docstring's "cannot leave a false sentence behind". Done with the triage batch, after the records regenerate,
  so the sentences are computed from the final table.
- [ ] T-149 ⟨now⟩ **`config/bowen/event_sizes.md`'s `FIND_WORK_LOAD` note (review MR-10).** The note says the load goes to
  the person who finds work; the code, the spec and the explainer send it to the household, at the self reach. The
  T-130 page was corrected 2026-10-28; the config note waits for the triage batch, since `config/bowen/` is in the
  engine hash.
- [ ] T-150 ⟨later⟩ **Transcript phrases outside the spec, and a test for them (review MR-2, 2026-10-28).** Revision 40
  reworded every distinctive run of three or more words that the spec shared with a quoted span of the KB extractions
  or `_LECTURES_1979.md` (an n-gram comparison; theory vocabulary such as "differentiation of self" and common phrases such as "level of anxiety" kept). The same
  comparison at four words finds about 14 lines in `docs/model_explainer.md` (for example its lines 837, 996, 2119);
  other documents were not checked. `tests/test_asr_quotations.py` reads only quoted spans of four words or more, so it
  cannot find unquoted reuse. To decide: sweep the explainer the same way, and either add a test built on the n-gram
  comparison (with an allowlist of theory vocabulary in `tests/asr_quotations.json`) or record why the test stays as it is.
- [ ] T-151 ⟨later⟩ **CI installs whatever `ruff` and `numpy` are current (review MR-15, 2026-10-28).** Pin `ruff` to the
  version that passed CI run 40630271972 and bound numpy's major version in `requirements.txt`; bump each on purpose,
  in its own commit. numpy's draws feed the golden fingerprints.
- [ ] T-152 ⟨later⟩ **A probe test's "as built" assertion cannot fail (review MR-16, 2026-10-28).**
  `tests/test_stage_3_probes.py:80–81`: the `as_built` variant never installs the displacement wrapper, so its tallies
  are 0 by construction. Delete it, or check the engine's own displacement in the `as_built` treatment arm.
- [ ] T-153 ⟨later⟩ **`test_d31_every_listed_record_is_marked_and_no_other` asserts nothing while the batch is closed
  (review MR-17, 2026-10-28).** Add `else: assert _marked() == []` (`tests/test_open_batch.py:61–63`).
- [ ] T-154 ⟨later⟩ **The family-of-origin arm's held-weeks check was loosened to "more than none" (review MR-18,
  2026-10-28).** `tests/bowen/test_phase_d2_criteria.py:130`: until T-135 is decided, a floor that catches a collapse
  (for example `>= 0.9 * 416`), or `== 401` pinned with the golden fingerprints.
- [ ] T-155 ⟨later⟩ **The projection-trace tool prints the estimator's formula from before revision 39 (review MR-19,
  2026-10-28).** `tools/stage_3_projection_trace.py:191–193` leaves out `− floor_held` (`slow_tick.py:129`). Add the
  term, or have `main()` refuse unless the engine hash is the page's stamped one.
- [ ] T-156 ⟨later⟩ **The i-position probe copies the criterion's mover setup (review MR-20, 2026-10-28).**
  `tools/stage_3_iposition_probe.py:65–70` should call `criteria.mover_at(...)` and keep only its documented spell
  difference local.
- [ ] T-157 ⟨later⟩ **Probe-page tests for frozen pages take 17% of the default suite (review MR-21, 2026-10-28).** Mark
  the two slow tests in `tests/test_stage_3_probes.py` (the displaced-coping probe, 39 s, and the trace, 18 s)
  `ensemble` or `slow`, and update the suite count in CLAUDE.md in the same commit.
