# Whole-repo review — 2026-11-11

Items from `REVIEW-EPModel-2026-11-11.md` (repo root), a read-only review of the code, tests and documents. The
report's finding ID is in brackets. "Fix" items have a known fix; "Decide" items need the owner's answer first.
Nothing here is built or decided. Items marked *verified* survived the review's challenge pass against the code.

## Fix — engine and criteria

- [ ] T-214 ⟨now⟩ **Fix: beliefs created at a pairing, birth or succession are one shared object (2026-11-11).** [B1,
  *verified*] `src/bowen/engine/life_course.py:226-233` assigns one `TieBelief` to every entry, and
  `beliefs.update_beliefs` mutates in place, so one person's belief update changes others'. Build one object per
  entry; add a test that updates one belief after a birth and asserts no other entry moved. Moves the engine hash and
  every life-course record, so it goes in the next batch that moves the hash.
- [ ] T-215 ⟨later⟩ **Fix: two I-POSITION exits skip their cleanup (2026-11-11).** [C2, *verified*]
  `iposition.py:212-214` (`mover_died` in RESOLVE leaves the target's pull-up) and `:400-401` (a twice-deferred
  abort ends without `restore_prior_balance`). No test names either exit. Moves the engine hash.
- [ ] T-216 ⟨later⟩ **Fix: the required-move gate reads the treatment arm only (2026-11-11).** [C3, *verified*]
  `src/bowen/ensemble/runner.py:207-213`. A move scripted in the baseline (`M11.C.18`, `M11.C.20`, `M11.C.23`) is not
  gated. No current verdict is wrong because of it.
- [ ] T-217 ⟨later⟩ **Fix: exclusion counts recorded by arms reach no verdict or record (2026-11-11).** [C4,
  *verified*] `runner.py:84-95`; keys at `criteria_d2.py:393, 404`, `criteria.py:733-735`. Add a counts field to
  `Verdict` or make each a report-only readout. Includes `M11.C.6` reading a dead child as 0.0, `M11.C.25` passing
  with no pole formed, and `M11.C.26` booking one of its three runs.
- [ ] T-218 ⟨later⟩ **Fix: the log header's `config_hash` omits config that changes a run (2026-11-11).** [C5,
  *verified*] `src/bowen/scenario/header.py:27-55` leaves out event-kind components, the accommodating and displacing
  sets, contact classes, event sizes, `family.life` and the policy rules.
- [ ] T-219 ⟨later⟩ **Fix: minor engine and loader defects, one batch (2026-11-11).** [Minor] `consolidate.py:51-55`
  logs a snap up to the floor as positive `acute_decay`; `appraise.py:332-357` gates on the reference family only;
  `moves.py:192-194` `_desc` prefers the longer id; `iposition.py:327` logs the unclamped gain; `act.py:113-163`
  degrades I-POSITION when `params` is None; loaders accept `nan`, `inf` and out-of-range values, and a NaN
  `invariant_tolerance` switches the invariants off; a bad binder target fails mid-run without file and line; the
  `partners:` path is unchecked; `M6.I.2` and the tie half of `M6.I.8` cannot fire; `fast_tick_weeks` is read by no
  code; the two definitions of "automatic" differ (`criteria.py:172` against `event_kinds.md`); act sets live in
  Python and are duplicated in config (`sinks.py:40-45`, `criteria.py:172, 812`, `iposition.py:95`).

## Fix — tests

- [ ] T-220 ⟨now⟩ **Fix: delete `tests/test_review_rev31.py` (2026-11-11).** [T1, *verified*] Six blocks assert
  exact prose, which effort budget rule 5 bans. 159 lines, 16 tests.
- [ ] T-221 ⟨later⟩ **Fix: delete the closed open-batch mechanism (2026-11-11).** [T2, *verified*]
  `tests/open_batch.py`, `tests/test_open_batch.py` and the hooks in `conftest.py`, `test_version_2_status.py:30-38`
  and `test_spec_consistency.py:404-437`. About 540 lines, 11 tests; `OPEN` has been None since 2026-10-26.
- [ ] T-222 ⟨later⟩ **Fix: remove tests that pin hand-kept numbers or re-check fixed history (2026-11-11).** [T3,
  T4] The suite count in `CLAUDE.md`, the exact `Spec:` reference count, the 140-name constant set in
  `bowen/test_log.py:147-212`, the revision and requirement count restated across documents; the `version-1` tag
  re-run, the nine rebuilt revision pages, `pytest --collect-only` inside the suite. Goes with T-225.
- [ ] T-223 ⟨later⟩ **Fix: small test gaps and leftovers (2026-11-11).** [Minor] Add a cross-process determinism test
  of a policy run under a changed hash salt; delete `bowen/test_log.py:31-43` (vacuous by its docstring); archive the
  five closed probe tools and their tests; replace report-fragment assertions with assertions on the numbers; pin
  CI's NumPy and add a manual `-m ensemble` job; `tools/c2_matched_probe.py` has no test.

## Fix — documents

- [ ] T-224 ⟨now⟩ **Fix: stale status lines and pointers (2026-11-11).** [D1, D2, P5] Recheck each against `HEAD`
  first; they were read while another session was editing. `docs/proposal_social_cues.md:3-6` and
  `docs/spec_revision_55_draft.md:3-4` say not built; `SPEC_REVIEW_CHECKLIST.md:218, 262, 264` and
  `REVIEWER_READING_LIST.md:196-199` repeat it. The sweep record's hash is `0b363cb3186c`, not `927550dfcb8a`
  (`REVIEWER_READING_LIST.md:421, 435`, `CHANGELOG.md:118-120`). `model_explainer.md:2332` says 10 of 13 audited rows
  survive; the record shows 9. `version_2_status.md:114` contradicts itself on `M11.C.48[phase_d]`. T-111 and T-168
  point at things that no longer hold. The checklist files the current status page under "not for review".
- [ ] T-225 ⟨now⟩ **Fix: keep revision history in one place (2026-11-11).** [D3, D4] It is hand-restated in
  `CLAUDE.md:28, 96`, `README.md:54`, `REVIEWER_READING_LIST.md:3, 226-436`, the spec's line 4,
  `docs/theory/_STATUS.md:3` and `spec_revision_log.md:3`. Keep it in the revision log and the changelog; elsewhere
  one sentence and a pointer, from one generated status file with one test. Cuts `CLAUDE.md` by about half. Relates
  to T-175.
- [ ] T-226 ⟨later⟩ **Fix: archive finished-work documents and delete grid-engine leftovers (2026-11-11).** [D5,
  Minor] About 55 files to an archive: 15 finished plans, 20 applied proposals, decisions and drafts, past reviews
  and phase reports, eight frozen comparison pages, 17 Phase C QA-gate logs. Delete `gemini.md`,
  `monte_carlo_results.txt`, `data/` and six grid-engine documents. Check `tests/test_doc_claims.py`,
  `tests/test_doc_checklist.py` and `tools/record_changes.py` for path dependencies first. The list is in the report.
- [ ] T-227 ⟨later⟩ **Fix: comments and README (2026-11-11).** [Minor] 416 "revision N" comments in 44 engine files
  carry history where the current rule belongs (comments are outside the hash); explainer §11.3-11.5 are "what
  revision N changed" sections; README "Running" lacks a policy run; 12 filenames have spaces and em dashes.

## Decide

- [ ] T-228 ⟨later⟩ **Decide: a second I-POSITION during a running sequence (2026-11-11).** [C1, *verified*]
  `act.py:124-130` overwrites the running sequence with no `end` record; a RESOLVE-stage pull-up is never reverted.
  `M5.D.9` does not say whether it is illegal, redirected or a restart. May bear on T-158.
- [ ] T-229 ⟨later⟩ **Decide: what the headline should say about learning (2026-11-11).** [P1, P6] Of 13 audited
  passing rows, 9 pass with learning off; five need it and two of those are robust; 216 invented constants stand
  behind them. Whether the status page's summary shows the learning-dependent tally beside the pass count.
- [ ] T-230 ⟨later⟩ **Decide: act payoffs set what is learned (2026-11-11).** [P2] DISTANCE relieves more at once
  than TRIANGLE inside the 3-week horizon, so triangling is not learned (`M11.C.42`, `C.44`, `C.45`). Whether to
  sweep `distance_binding_rate` against `triangle_transfer_rate` on those criteria.
- [ ] T-231 ⟨later⟩ **Decide: the learning signal (2026-11-11).** [P3, *verified* as code] (a) It includes ordinary
  decay with no baseline (`tick.py:134, 243`, `learner.py:130`); a mutant against `M11.C.39`, `C.16`, `C.34` would
  show what that carries. (b) It reads others' true anxiety and functional level (`learner.py:120, 215-253`);
  `M4.B.2` does not say whether learning is exempt from the perception boundary.
- [ ] T-232 ⟨later⟩ **Decide: a command that runs the learning family (2026-11-11).** [P4, *verified*]
  `src/bowen/run.py` runs the scripted Phase B family only; `tools/phase_c_trace.py` runs the Phase C family. Nothing
  runs the twelve-person family with the life course and renders it.
- [ ] T-233 ⟨later⟩ **Decide: smaller questions from the review (2026-11-11).** Whether `settle_functioning`
  (`moves.py:370-390`) is act physics or a written pattern; whether `M11.C.55`'s reversal with learning off, partly
  by definition, counts as evidence of learning; whether a within-run criterion needs a minimum share of non-zero
  pairs; whether the LLM experiment is finished, so its code, tests and the `anthropic` dependency can go; whether a
  tick's records should be emitted when its invariants fail; whether sibling ties and sibling position belong in the
  reference family; whether the engine hash should be split per criterion [C6]; whether CI's `@v7` actions exist and
  CI is green.
