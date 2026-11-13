# Whole-repo review — 2026-11-11

Items from `REVIEW-EPModel-2026-11-11.md` (repo root), a read-only review of the code, tests and documents. The
report's finding ID is in brackets. "Fix" items have a known fix; "Decide" items need the owner's answer first.
The Decide items T-228 to T-233 and the carried-over T-241 were decided on 2026-11-12; T-234 to T-242 carry the work
they set. Each item's own line says whether it is done. Items marked *verified* survived the review's challenge pass against the code.

## Fix — engine and criteria

- [x] T-214 ⟨now⟩ **Fix: beliefs created at a pairing, birth or succession are one shared object (2026-11-11).** [B1,
  *verified*] `src/bowen/engine/life_course.py:226-233` assigns one `TieBelief` to every entry, and
  `beliefs.update_beliefs` mutates in place, so one person's belief update changes others'. Build one object per
  entry; add a test that updates one belief after a birth and asserts no other entry moved. Moves the engine hash and
  every life-course record, so it goes in the next batch that moves the hash.
  *Done 2026-11-12: one `TieBelief` per holder at every entry; spec revision 59, the engine batch (T-001 step 3; `docs/plan_engine_batch_2026-11-12.md`), criterion 1.*
- [x] T-215 ⟨later⟩ **Fix: two I-POSITION exits skip their cleanup (2026-11-11).** [C2, *verified*]
  `iposition.py:212-214` (`mover_died` in RESOLVE leaves the target's pull-up) and `:400-401` (a twice-deferred
  abort ends without `restore_prior_balance`). No test names either exit. Moves the engine hash.
  *Done 2026-11-12: (a) a mover dead in `RESOLVE` reverts the pull-up, logged; (b) an abort ended by a second deferral restores the prior balance; spec revision 59, the engine batch (T-001 step 3; `docs/plan_engine_batch_2026-11-12.md`), criteria 2 and 3.*
- [x] T-216 ⟨later⟩ **Fix: the required-move gate reads the treatment arm only (2026-11-11).** [C3, *verified*]
  `src/bowen/ensemble/runner.py:207-213`. A move scripted in the baseline (`M11.C.18`, `M11.C.20`, `M11.C.23`) is not
  gated. No current verdict is wrong because of it.
  *Done 2026-11-12: `Criterion.requires_moves_baseline`; `M11.C.18`, `C.20`, `C.23` split by arm; spec revision 59, the engine batch (T-001 step 3; `docs/plan_engine_batch_2026-11-12.md`), criterion 5.*
- [x] T-217 ⟨later⟩ **Fix: exclusion counts recorded by arms reach no verdict or record (2026-11-11).** [C4,
  *verified*] `runner.py:84-95`; keys at `criteria_d2.py:393, 404`, `criteria.py:733-735`. Add a counts field to
  `Verdict` or make each a report-only readout. Includes `M11.C.6` reading a dead child as 0.0, `M11.C.25` passing
  with no pole formed, and `M11.C.26` booking one of its three runs.
  *Done 2026-11-12: `Verdict.counts` per arm, rendered in the ensemble and Phase D records when present; spec revision 59, the engine batch (T-001 step 3; `docs/plan_engine_batch_2026-11-12.md`), criterion 6. The records show them once step 4 regenerates them.*
- [x] T-218 ⟨later⟩ **Fix: the log header's `config_hash` omits config that changes a run (2026-11-11).** [C5,
  *verified*] `src/bowen/scenario/header.py:27-55` leaves out event-kind components, the accommodating and displacing
  sets, contact classes, event sizes, `family.life` and the policy rules.
  *Done 2026-11-12: `config_hash` hashes every loaded config input; spec revision 59, the engine batch (T-001 step 3; `docs/plan_engine_batch_2026-11-12.md`), criterion 7.*
- [x] T-219 ⟨later⟩ **Fix: minor engine and loader defects, one batch (2026-11-11).** [Minor] `consolidate.py:51-55`
  logs a snap up to the floor as positive `acute_decay`; `appraise.py:332-357` gates on the reference family only;
  `moves.py:192-194` `_desc` prefers the longer id; `iposition.py:327` logs the unclamped gain; `act.py:113-163`
  degrades I-POSITION when `params` is None; loaders accept `nan`, `inf` and out-of-range values, and a NaN
  `invariant_tolerance` switches the invariants off; a bad binder target fails mid-run without file and line; the
  `partners:` path is unchecked; `M6.I.2` and the tie half of `M6.I.8` cannot fire; `fast_tick_weeks` is read by no
  code; the two definitions of "automatic" differ (`criteria.py:172` against `event_kinds.md`); act sets live in
  Python and are duplicated in config (`sinks.py:40-45`, `criteria.py:172, 812`, `iposition.py:95`).
  *Done 2026-11-12: items (1) to (12), one guard each; spec revision 59, the engine batch (T-001 step 3; `docs/plan_engine_batch_2026-11-12.md`), criterion 8. (7) checks a triangle-position binder's three members only, since triangles form during the run; (2) does not filter a couple who have both died (noted, not changed).*

## Fix — tests

- [x] T-220 ⟨now⟩ **Fix: delete `tests/test_review_rev31.py` (2026-11-11).** [T1, *verified*] Six blocks assert
  exact prose, which effort budget rule 5 bans. 159 lines, 16 tests.
  *Done 2026-11-11: deleted; its one structural check (every row of `config/bowen/criteria.md` graded `[I]`) moved to `tests/bowen/test_config.py::test_m0_1_criteria_settings_graded`.*
- [x] T-221 ⟨later⟩ **Fix: delete the closed open-batch mechanism (2026-11-11).** [T2, *verified*]
  `tests/open_batch.py`, `tests/test_open_batch.py` and the hooks in `conftest.py`, `test_version_2_status.py:30-38`
  and `test_spec_consistency.py:404-437`. About 540 lines, 11 tests; `OPEN` has been None since 2026-10-26.
  *Done 2026-11-11: `tests/open_batch.py`, `tests/test_open_batch.py`, the `conftest.py` fixture and the `test_version_2_status.py` hooks removed.*
- [x] T-222 ⟨later⟩ **Fix: remove tests that pin hand-kept numbers or re-check fixed history (2026-11-11).** [T3,
  T4] The suite count in `CLAUDE.md`, the exact `Spec:` reference count, the 140-name constant set in
  `bowen/test_log.py:147-212`, the revision and requirement count restated across documents; the `version-1` tag
  re-run, the nine rebuilt revision pages, `pytest --collect-only` inside the suite. Goes with T-225.
  *Done 2026-11-11: removed the suite-count test with its `pytest --collect-only` and the D4 phrasing tests, D1's revision checks, the requirement-count test, MR-13's three-document hash check (the hash is read into `docs/STATUS.md`), and the `version-1` tag re-run; the constant set in `bowen/test_log.py` is computed from the two registers; only the live revision page is rebuilt. Kept by the owner's choice of 2026-11-11: the exact `Spec:` count (`M11.D.9` requires it; the rule goes to the Decide session).*
- [ ] T-223 ⟨later⟩ **Fix: small test gaps and leftovers (2026-11-11).** [Minor] Add a cross-process determinism test
  of a policy run under a changed hash salt; delete `bowen/test_log.py:31-43` (vacuous by its docstring); archive the
  five closed probe tools and their tests; replace report-fragment assertions with assertions on the numbers; pin
  CI's NumPy and add a manual `-m ensemble` job; `tools/c2_matched_probe.py` has no test.
  *Partly done 2026-11-11: added `tests/bowen/test_determinism.py::test_m11d5_policy_run_same_log_across_hash_salts`. Kept `bowen/test_log.py:31-43`: it checks that the sink writes every record, and the frozen Phase B mutation gate and two `Tests:` lines name it. Not done: the probe tools' archive (with T-226's move), the report-fragment assertions, the CI NumPy pin (1.26.4 has no wheel for Python 3.13; whether CI runs at all is T-233), the manual `-m ensemble` job, a test for `tools/c2_matched_probe.py`.*

## Fix — documents

- [x] T-224 ⟨now⟩ **Fix: stale status lines and pointers (2026-11-11).** [D1, D2, P5] Recheck each against `HEAD`
  first; they were read while another session was editing. `docs/archive/proposal_social_cues.md:3-6` and
  `docs/archive/spec_revision_55_draft.md:3-4` say not built; `SPEC_REVIEW_CHECKLIST.md:218, 262, 264` and
  `REVIEWER_READING_LIST.md:196-199` repeat it. The sweep record's hash is `0b363cb3186c`, not `927550dfcb8a`
  (`REVIEWER_READING_LIST.md:421, 435`, `CHANGELOG.md:118-120`). `model_explainer.md:2332` says 10 of 13 audited rows
  survive; the record shows 9. `version_2_status.md:114` contradicts itself on `M11.C.48[phase_d]`. T-111 and T-168
  point at things that no longer hold. The checklist files the current status page under "not for review".
  *Done 2026-11-11, except P5 (`version_2_status.md:114`), which goes to the Decide session by the owner's choice of 2026-11-11: the status lines of the social-cues proposal and the revision 55 draft, the checklist's and the reading list's rows, the sweep record's hash (`0b363cb3186c`), the explainer's audit count (9 of 13), the six current records moved into the checklist's Part 3. T-111 and T-168 closed.*
- [x] T-225 ⟨now⟩ **Fix: keep revision history in one place (2026-11-11).** [D3, D4] It is hand-restated in
  `CLAUDE.md:28, 96`, `README.md:54`, `REVIEWER_READING_LIST.md:3, 226-436`, the spec's line 4,
  `docs/theory/_STATUS.md:3` and `spec_revision_log.md:3`. Keep it in the revision log and the changelog; elsewhere
  one sentence and a pointer, from one generated status file with one test. Cuts `CLAUDE.md` by about half. Relates
  to T-175.
  *Done 2026-11-11 (option A, the owner's choice): `tools/status.py` writes `docs/STATUS.md`, checked by `tests/test_status.py`; the status texts of the spec, the revision log, `_STATUS.md`, `README.md`, `CLAUDE.md` and the reading list are one sentence and a pointer, and the old texts are in `docs/spec_revision_log.md` under "Status text moved here 2026-11-11 (T-225)", verbatim. The explainer's §11.3-11.5 stay (T-227).*
- [x] T-226 ⟨later⟩ **Fix: archive finished-work documents and delete grid-engine leftovers (2026-11-11).** [D5,
  Minor] About 55 files to an archive: 15 finished plans, 20 applied proposals, decisions and drafts, past reviews
  and phase reports, eight frozen comparison pages, 17 Phase C QA-gate logs. Delete `gemini.md`,
  `monte_carlo_results.txt`, `data/` and six grid-engine documents. Check `tests/test_doc_claims.py`,
  `tests/test_doc_checklist.py` and `tools/record_changes.py` for path dependencies first. The list is in the report.
  *Partly done 2026-11-11: the six grid-engine documents, `gemini.md`, `monte_carlo_results.txt` and `data/` removed (`git rm`; all at the tag `version-1`). The archive move goes with the engine batch (the owner's choice of 2026-11-12): 17 candidates are cited by hashed `config/bowen/*.md` prose (6), by version 1's tag-checked files (2), by the LLM experiment's code and prompts (3) and by generated records (6), and that batch moves the hash and regenerates every record anyway.*
  *Done 2026-11-12: the archive move: 49 of the 60 finished documents and the 17 Phase C QA-gate logs moved to `docs/archive/` (`docs/archive/cycles/`), live references rewritten by script; the 11 kept in place are listed in the plan's criterion 14. Moved unedited, so their own relative links to documents still in `docs/` no longer resolve. The 12 filenames with spaces were not renamed.*
- [ ] T-227 ⟨later⟩ **Fix: comments and README (2026-11-11).** [Minor] 416 "revision N" comments in 44 engine files
  carry history where the current rule belongs (comments are outside the hash); explainer §11.3-11.5 are "what
  revision N changed" sections; README "Running" lacks a policy run; 12 filenames have spaces and em dashes.
  *Partly done 2026-11-11: 374 of the 416 "revision N" provenance tags removed from comments in `src/bowen/` (the engine hash did not move; one mutant's site, `displaced-coping-removed` in `stress_response.py`, kept as it was); the 42 left are narrative. README "Running" gives the scripted and policy runs. Not done: the explainer's §11.3-11.5 (`config/sweep/sweep.md` and the explainer itself cite them by number) and the 12 filenames with spaces (with T-226's move).*

## Decide

- [x] T-228 ⟨later⟩ **Decide: a second I-POSITION during a running sequence (2026-11-11).** [C1, *verified*]
  `act.py:124-130` overwrites the running sequence with no `end` record; a RESOLVE-stage pull-up is never reverted.
  `M5.D.9` does not say whether it is illegal, redirected or a restart. May bear on T-158.
  *Decided by the owner 2026-11-12 through `AskUserQuestion`: "Illegal (Recommended)". While a sequence runs, I-POSITION toward anyone else is not a legal move: one sequence per person at a time. Built by T-234.*
- [x] T-229 ⟨later⟩ **Decide: what the headline should say about learning (2026-11-11).** [P1, P6] Of 13 audited
  passing rows, 9 pass with learning off; five need it and two of those are robust; 216 invented constants stand
  behind them. Whether the status page's summary shows the learning-dependent tally beside the pass count.
  *Decided by the owner 2026-11-12 through `AskUserQuestion`: "Add the tally (Recommended)". `tools/version_2_status.py` adds, beside the pass count, a generated line read from the emergence audit record: how many passing rows need learning and how many of those are robust. Built by T-235.*
- [x] T-230 ⟨later⟩ **Decide: act payoffs set what is learned (2026-11-11).** [P2] DISTANCE relieves more at once
  than TRIANGLE inside the 3-week horizon, so triangling is not learned (`M11.C.42`, `C.44`, `C.45`). Whether to
  sweep `distance_binding_rate` against `triangle_transfer_rate` on those criteria.
  *Decided 2026-11-12 as recommended, under the owner's message of 2026-11-12, "continue with your recommendations", sent while T-230 was asked: sweep, report only. After the engine batch, a targeted two-constant sweep of `distance_binding_rate` against `triangle_transfer_rate` on `M11.C.42`, `C.44` and `C.45` (`tools/version_2_sweep.py --criteria`), recorded as a sensitivity result. The constants are not moved to make the rows pass (`M10.B.4`). Built by T-236.*
- [x] T-231 ⟨later⟩ **Decide: the learning signal (2026-11-11).** [P3, *verified* as code] (a) It includes ordinary
  decay with no baseline (`tick.py:134, 243`, `learner.py:130`); a mutant against `M11.C.39`, `C.16`, `C.34` would
  show what that carries. (b) It reads others' true anxiety and functional level (`learner.py:120, 215-253`);
  `M4.B.2` does not say whether learning is exempt from the perception boundary.
  *Decided 2026-11-12 as recommended, under the owner's message of 2026-11-12, "continue with your recommendations", sent while T-230 was asked: measure before changing a rule. (a) A mutant that subtracts the expected decay toward the chronic floor from the felt relief, run on `M11.C.39`, `C.16` and `C.34`. (b) `M4.B.2` states the learner's exemption from the perception boundary as an `[I]` modelling choice, and a mutant that reads the learner's perceived values instead of others' true anxiety and level runs on `M11.C.10[family of origin]`; whether to close the exemption is decided on that result. Built by T-237.*
- [x] T-232 ⟨later⟩ **Decide: a command that runs the learning family (2026-11-11).** [P4, *verified*]
  `src/bowen/run.py` runs the scripted Phase B family only; `tools/phase_c_trace.py` runs the Phase C family. Nothing
  runs the twelve-person family with the life course and renders it.
  *Decided 2026-11-12 as recommended, under the owner's message of 2026-11-12, "continue with your recommendations", sent while T-230 was asked: yes. A tool in `tools/` (outside the engine hash) runs the twelve-person family with the life course under the policy and renders its trace, reusing `tools/phase_c_trace.py`'s renderer; README "Running" names it. Built by T-238.*
- [x] T-233 ⟨later⟩ **Decide: smaller questions from the review (2026-11-11).** Whether `settle_functioning`
  (`moves.py:370-390`) is act physics or a written pattern; whether `M11.C.55`'s reversal with learning off, partly
  by definition, counts as evidence of learning; whether a within-run criterion needs a minimum share of non-zero
  pairs; whether the LLM experiment is finished, so its code, tests and the `anthropic` dependency can go; whether a
  tick's records should be emitted when its invariants fail; whether sibling ties and sibling position belong in the
  reference family; whether the engine hash should be split per criterion [C6]; whether CI's `@v7` actions exist and
  CI is green.
  *Decided 2026-11-12 as recommended, under the owner's message of 2026-11-12, "continue with your recommendations", sent while T-230 was asked. (1) `settle_functioning` is set act physics, required by `M1.B.5` and `M1.B.6`; it is listed among the acknowledged set mechanisms in `M11.5` and explainer §19. (2) The part of `M11.C.55`'s reversal with learning off that holds by definition (a tie counts as a miss) is not evidence of learning; the audit row says so. (3) Every within-run criterion first reports its share of non-zero pairs; a minimum is decided after the shares are seen. (4) The LLM experiment is not finished while T-201 is open; its code, tests and the `anthropic` dependency stay until T-201 closes, then go. (5) A tick's records are emitted before an invariant failure is raised, with a record of the failure, so the log holds the failing tick; logging stays a pure observer. (6) Sibling ties and sibling position are not added in version 2; they need a proposal with ledger sources first. (7) The engine hash is split per criterion after the tag, as its own tools batch, so the stage-end sweep and the tag cite one hash. (8) The `@v7` actions exist: CI runs on all three Pythons. CI on `main` was red from 2026-11-11 on one test, `tests/experiments/llm_agents/test_cache.py::test_lx25_committed_pilot_replays_fixed`, which compared gzip bytes that differ between platforms; changed 2026-11-12 to compare the decompressed content. Items (1), (2), (3) and (5) are T-239; (4), (6) and (7) are T-240.*

## Carried over from the documents-and-tests batch (`bda211e`)

- [x] T-241 ⟨now⟩ **Decide: `M11.C.48[phase_d]`'s status-page row, and `M11.D.9`'s exact count (2026-11-12).** [P5]
  `docs/version_2_status.md:114` shows the row as proved by `cutoff-blind-to-learned-value` turning red and counts it
  toward W3, while the audit shows the row passes with all learning removed and the disposition text
  (`docs/version_2_dispositions.md:36`) still says the mutant survives. [T3] `M11.D.9` requires the exact `Spec:`
  reference count that `tests/bowen/test_engineering_gates.py` pins (`EXPECTED_SPEC_REFERENCES = 1097`).
  *Decided 2026-11-12 as recommended, under the owner's message of 2026-11-12, "continue with your recommendations", sent while T-230 was asked. P5: the disposition text is rewritten after step 4's regeneration,
  against the Phase D mutation record and the emergence audit as regenerated there. It says that the mutant's red
  (the mutant makes `CUTOFF` top-weighted whenever it is legal) does not show that learning carries the row, since the
  row passes with learning removed. W3 counts the row only if the mutant is red in the regenerated record. `M11.D.9`:
  the exact count is replaced by a check that the scan covers every `.py` file under `src/bowen/` and that every
  reference resolves; `EXPECTED_SPEC_REFERENCES` goes, and `M11.D.8` and `M11.D.9` change at the next spec revision.
  Built by T-242.*

## Fix — from the owner's decisions of 2026-11-12

- [x] T-234 ⟨now⟩ **Fix: I-POSITION is illegal while a sequence runs (2026-11-12).** From T-228. Mask I-POSITION
  toward anyone other than the running sequence's target in legality; state it in `M5.D.9`; a test that a person in
  RESOLVE cannot open a second sequence and that the target's pull-up is followed up or reverted. Moves the engine
  hash; goes in the engine batch. Check whether it changes T-158's expected failure.
  *Done 2026-11-12: `I-POSITION` toward anyone but the running sequence's target is not legal, and a scripted one raises; spec revision 59, the engine batch (T-001 step 3; `docs/plan_engine_batch_2026-11-12.md`), criterion 4. `M5.D.3`'s counts moved to 55 abort, 64 hold, 122 said in anger.*
- [ ] T-235 ⟨now⟩ **Fix: the status page's summary shows the learning-dependent tally (2026-11-12).** From T-229.
  Generated by `tools/version_2_status.py` from the emergence audit record and checked by the existing status test.
  Built after step 4's regeneration, so it reads the regenerated audit. No engine change.
- [ ] T-236 ⟨now⟩ **Sweep: `distance_binding_rate` against `triangle_transfer_rate` on `M11.C.42`, `C.44`, `C.45`
  (2026-11-12).** From T-230. After the engine batch, recorded as a sensitivity result; no constant is moved by it.
- [x] T-237 ⟨now⟩ **Mutants: the learning signal's decay baseline and its perception exemption (2026-11-12).** From
  T-231. (a) Subtract the expected decay toward the chronic floor from the felt relief; run on `M11.C.39`, `C.16`,
  `C.34`. (b) Read the learner's perceived values in `cross_person_relief` and the level terms; run on
  `M11.C.10[family of origin]`. State the exemption in `M4.B.2` as `[I]`. The mutants live in
  `tools/mutant_runner.py`, which is in the engine hash, so they go in the engine batch.
  *Partly done 2026-11-12: `M4.B.2` states the exemption as `[I]`; the two mutants (`relief-less-expected-decay`, `learner-reads-own-state-only`) are defined in the two record tools at spec revision 59, the engine batch (T-001 step 3; `docs/plan_engine_batch_2026-11-12.md`), criterion 12, each applying once. Run 2026-11-12 at step 4 on `95368d6ddf18`: `relief-less-expected-decay` survives on `M11.C.16` (`docs/phase_c_mutation_record.md`) and on `M11.C.39` (`docs/phase_d_mutation_record.md`). Not run on `M11.C.34` or, for `learner-reads-own-state-only`, on `M11.C.10[family of origin]`: both criteria fail unmutated on this hash, so the record lists the two mutants under criteria that do not pass. Closed as run; what the two new FAILs mean is the owner's.*
- [ ] T-238 ⟨later⟩ **Build: a trace command for the twelve-person family with the life course (2026-11-12).** From
  T-232. A tool in `tools/`, outside the hash, reusing `tools/phase_c_trace.py`'s renderer; README "Running" names it.
- [x] T-239 ⟨now⟩ **Fix: the engine-batch parts of T-233 (2026-11-12).** List `settle_functioning` among the
  acknowledged set mechanisms (`M11.5`, explainer §19); note in the audit that part of `M11.C.55`'s reversal with
  learning off holds by definition; report each within-run criterion's share of non-zero pairs (with T-217's counts);
  emit a tick's records before an invariant failure is raised, with a failure record.
  *Done 2026-11-12: (a) `settle_functioning` listed in `M11.5` and explainer §19; (b) `M11.5`'s `M11.C.55` note, printed by the audit renderer beside a reversed row; (c) every row reports its share of non-zero pairs; (d) a failing tick logs its records and an `InvariantRecord` before it raises; spec revision 59, the engine batch (T-001 step 3; `docs/plan_engine_batch_2026-11-12.md`), criteria 6, 9, 10, 11.*
- [ ] T-240 ⟨later⟩ **Later parts of T-233 (2026-11-12).** Remove the LLM experiment's code, tests and the
  `anthropic` dependency once T-201 closes; split the engine hash per criterion after the tag (C6), as its own tools
  batch; a proposal with ledger sources before sibling ties or sibling position enter the reference family.
- [x] T-242 ⟨now⟩ **Fix: P5's disposition text and `M11.D.9`'s scan check (2026-11-12).** From T-241. The
  disposition is rewritten after step 4's regeneration; the scan-coverage check replaces `EXPECTED_SPEC_REFERENCES`,
  with `M11.D.8` and `M11.D.9` changed at the next spec revision.
  *Partly done 2026-11-12: `M11.D.8` and `M11.D.9` amended and the scan check built (`EXPECTED_SPEC_REFERENCES` removed) at spec revision 59, the engine batch (T-001 step 3; `docs/plan_engine_batch_2026-11-12.md`), criterion 13. P5's text rewritten 2026-11-12 after step 4's regeneration (`docs/version_2_dispositions.md`, `M11.C.48[phase_d]`), approved by the owner 2026-11-13.*
