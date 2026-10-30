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

- [x] T-148 ⟨now⟩ **`M11.C.16`'s fixed prose against its regenerated table (review MR-1, major; the owner kept the
  class composite, 2026-10-28).** `tools/c16_report.py` writes three sentences of `phase_c_completion_report.md` §11 as
  literal text (the learning bullet, "No other single rule removes the result", the "What does not hold" bullet) that
  the current mutant table contradicts: `availability-and-learner-removed` is reversed (+0.002 ± 0.002) and
  `steepness-inverted` is red. Rows :49 and :541 of the report and spec `M11.5`'s `M11.C.16` row misstate which mutants
  survive. Fix: make each sentence conditional on the mutant results, as INVERSION is, with a branch for a red or
  reversed single-rule inversion; generate or test rows :49 and :541 and the `M11.5` row from the same results; correct
  the docstring's "cannot leave a false sentence behind". Done with the triage batch, after the records regenerate,
  so the sentences are computed from the final table. *Closed 2026-10-28 (the triage batch): `tools/c16_report.py`
  computes every sentence of §11 that names a mutant result (the learning bullet, the single-rule sentences, "What
  does not hold", the inversion and redundancy sentences) and writes the report's two `M11.C.16` rows and the spec's
  `M11.5` row from the same results; it stops, writing nothing, on a result it has no wording for, a representation
  mutant that is not unchanged, a duplicated mutant row, a sweep record with no `M11.C.16` row, or `learner-disabled`
  turning C.16 red. Run on the regenerated records: `steepness-inverted` now survives, `availability-and-learner-removed`
  is reversed (+0.002 ± 0.002). Tests: `tests/bowen/test_ensemble_record.py::test_t148_c16_sentences_follow_the_mutant_results`
  (each false sentence the reviews found, fed its result) and `::test_m115_report_section_11_is_generated` (the three
  rows).*
- [x] T-149 ⟨now⟩ **`config/bowen/event_sizes.md`'s `FIND_WORK_LOAD` note (review MR-10).** The note says the load goes to
  the person who finds work; the code, the spec and the explainer send it to the household, at the self reach. The
  T-130 page was corrected 2026-10-28; the config note waits for the triage batch, since `config/bowen/` is in the
  engine hash. *Closed 2026-10-28 (the triage batch): the note says the load goes to the household, at the self reach.
  `tests/bowen/test_event_sizes.py::test_m7c1l_t149_the_find_work_note_says_where_the_load_lands`, red on the old note.*
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
- [ ] T-158 ⟨now⟩ **`M5.D.3`'s "abort is the usual first outcome" no longer holds after T-133 (the triage batch,
  2026-10-28).** Since spec revision 41 a mover angry when a sequence reaches `DEFINE` says the position as a demand and
  the sequence ends there (`M5.D.4a`), so the movers who reach the first opposition are calmer. Over
  `tests/bowen/test_iposition.py`'s 192 seeds: 62 abort at the first opposition and 66 hold (125 and 106 at revision 40),
  118 sequences end said in anger, and 21 complete. `test_m5d3_abort_is_the_usual_first_outcome` is a strict expected
  failure naming this item; its claim was not changed. To decide: (a) count a position said in anger as one of
  `M5.D.3`'s abort branches, since `M5.D.4a` calls them what discharging anger looks like, and restate `M5.D.3`'s place
  ("at the first opposition") to match; (b) keep `M5.D.3` as it is and record the failure as a finding; or (c) revisit
  where T-133's gate is read. Claude's recommendation under the owner's instruction of 2026-10-28, pending the owner's
  review: (b) until the owner decides, since (a) and (c) change a rule.
- [x] T-159 ⟨now⟩ **The stage-end sweep on the triage batch's engine (2026-10-28).** The batch moved the engine hash to
  `6ce8be104767`; every record but `docs/version_2_sweep_record.md` was regenerated once on it. The sweep
  (`tools/version_2_sweep.py`, 34 keys) reruns in the next session, split across both Macs as at step 11
  (`docs/two_mac_sweep.md`); at 8 workers on one Mac it would take about 22 hours. Until it does,
  `tests/test_version_2_sweep.py::test_s2_11_sweep_record_is_current` is a strict expected failure naming this item,
  failing only on a missing cached result for the current engine. Claude's recommendation under the owner's instruction
  of 2026-10-28, pending the owner's review: step 10's precedent, rather than holding the batch's push for a day.
  *Closed 2026-10-29: the sweep ran on `6ce8be104767`, split across both Macs, after a cross-check on `M11.C.16` and
  `M11.C.23` (the merge added 0 and found 490 entries equal). 267 first runs and 165 reruns at 200 seeds; none
  undetermined or errored. 33 keys have a fraction; `M11.C.48` is refused at the rerun at sample 2 (`M17.A.1`, spec
  revision 42; T-163). `docs/version_2_sweep_record.md` is current, its test passes without a mark, and
  `docs/version_2_status.md`'s W4 is unmet on `M11.C.48` alone.*
- [x] T-163 ⟨later⟩ **`M11.C.48` fixes its own seeds at 100, so the sweep refuses it at the rerun (found in T-159's
  sweep, 2026-10-29; not fixed).** On `6ce8be104767` sample 2's first run leaves `negative_run` at +0.255 ± 0.14,
  within two half-widths of zero, so `M17.A.1` needs a rerun at 200 seeds; `config/bowen/criteria.md` fixes the
  criterion's own count at 100, so the sweep refuses it there and shows it refused (revision 42, the owner's choice of
  2026-10-29). Samples 0 to 2 held; no fraction is formed, and `docs/version_2_status.md` keeps W4 unmet on it. To
  decide with the next batch that moves the engine hash: raise its count to 200 or more (the criteria table is in the
  hash), then sweep it again (`tools/version_2_sweep.py --criteria M11.C.48`). `M11.C.48[phase_d]` reads the same
  100 seeds (a variant key reads its base ID's section) and swept 8 of 8, because none of its samples needed the
  rerun; the raised count applies to it too.
  *Closed 2026-10-30 (spec revision 43, Claude's recommendation under the owner's instruction of 2026-10-29, pending
  owner review):* the criterion's seeds are 200 (`config/bowen/criteria.md`);
  `tests/test_version_2_sweep.py::test_m11c48_t163_own_seeds_reach_the_detection_limit`. It is swept again at the
  stage-end sweep on `da5981f6ec0b`.
- [ ] T-164 ⟨later⟩ **The merge's running-sweep guard matches the sweep's test file (found in T-159's merge,
  2026-10-29; not fixed).** `tools/merge_cache.py` refuses while `pgrep -f '[v]ersion_2_sweep.py'` finds a process,
  and that pattern also matches `python3 -m pytest tests/test_version_2_sweep.py`, so a test run makes the merge
  refuse. It fails safe (nothing written, and a second try after the tests end merged). `tools/two_mac_sweep.py`'s
  `status` uses the same pattern on Mac Mini1. To fix: match `tools/version_2_sweep.py` and test that a test run is
  not taken for a sweep.
- [ ] T-160 ⟨later⟩ **`phase_c_completion_report.md`'s other rows quote the numbers of Phase C's close (found building
  T-148, 2026-10-28; not fixed).** The report opens by saying every number below comes from a committed, generated
  record. Since T-148, `M11.C.16`'s two rows and §11 are generated from the current records (`tools/c16_report.py`);
  the other rows of §1 and §2 still quote the numbers of Phase C's close (for example `M11.C.3`'s seeker −1.29 and third
  +1.10), which the records have moved away from since. To decide: generate those rows from the records as T-148 does,
  or restate the opening as the report's state at Phase C's close, pointing at `docs/version_2_status.md` for where each
  criterion stands now.
- [ ] T-161 ⟨later⟩ **`M11.C.21`'s parents keep over-functioning toward the child after they collapse (found building
  T-145, 2026-10-28; not fixed).** `arm_c21` (`src/bowen/ensemble/criteria_d.py`) scripts both parents'
  `OVERFUNCTION` toward C1 every `act_every` weeks for the whole run, the weeks after the collapse included, so the
  treatment arm measures the return of lent functioning against lending that goes on. `M11.C.21` asks whether the
  target rises when both overfunctioners' capacity falls; whether collapsed parents should keep acting as projecting
  parents is the arm's design, not a rule, and changing it moves the engine hash (the arm is under
  `src/bowen/`). To decide with the next batch that moves the hash.
  - 2026-10-29, the version 2 definition batch (`docs/version_2_definition_plan.md` §2): built as Claude's
    recommendation (the treatment's lending stops at `t0`, the baseline's runs on) and **withdrawn before the
    regeneration** on the batch's `learning-qa` review. That form makes the arms differ by the scripted lending as well
    as the collapse, and the lending alone carries the pass, which is a rule writing the outcome (§19). Ad hoc, 40 seeds
    (1–40), `target_rise` at the criterion's settings: baseline 0.21; treatment with the drop 5.37; treatment with the
    lending stopped and **no** drop 6.14; as built at revision 41, treatment −0.04 (the review's measurement,
    `scratchpad` script, not a record). With that form `pseudo-self-return-removed` reads 4.92 against 0.21, so the
    named mutant would survive. The two forms left for the owner: (a) keep the arm as it is (lending in both arms for
    the whole run), or (b) stop the lending at `t0` in **both** arms, so the arms differ by the drop alone; ad hoc, (b)
    reads 5.37 against 6.14, a fail. Claude's recommendation: (b), since it is the only form whose arms differ by the
    collapse alone; it reverses the owner's arm design of 2026-10-16 and turns the verdict, so it waits for the owner.
- [x] T-162 ⟨later⟩ **`M11.C.34` passes at revision 41 and no mutant turns it red (found in revision 41's
  regeneration, 2026-10-29; not fixed).** At revision 40 it failed and its failure was accepted as a finding (version
  1's disposition, owner 2026-10-17); regenerated on `6ce8be104767` it passes (`automatic_reaction` −0.113 ± 0.13, p
  0.0388, `docs/phase_d_ensemble_record.md`). No mutant in either mutation record targets it, so
  `docs/version_2_status.md` holds it under W3 (every passing composite has a recorded mutation) and
  `docs/spec_coverage.md` lists it as partial. Adding a mutant (`tools/phase_d_mutation_record.py`) moves no rule;
  which rule to remove, and the owner's review of the changed verdict (T-122), come first.
  *Closed 2026-10-30 (spec revision 43, Claude's recommendation under the owner's instruction of 2026-10-29, pending
  owner review):* the declared mutant is built, `mixing-weight-blind-to-level` (`M4.D.1a`'s mixing weight fixed at
  the scale's middle), and turns the criterion red (−0.04 ± 0.12, p 0.26, `docs/phase_d_mutation_record.md`). The
  owner's review of the changed verdict (T-122) is still open.
