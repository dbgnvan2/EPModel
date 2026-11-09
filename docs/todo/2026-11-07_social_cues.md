# Social cues and signal channels — 2026-11-07

Items from `docs/proposal_social_cues.md`, decided by the owner on 2026-11-07 (Q1–Q11, each as recommended).

- [x] T-202 ⟨later⟩ **Build the social cues proposal, after T-183's cross-check (2026-11-07).** The owner decided
  Q1–Q11 of `docs/proposal_social_cues.md` on 2026-11-07, each as recommended: the four cues are readings that move
  `M4.C.1`'s two drives, not new drives (Q1); disapproval acts on both sides, impingement larger (Q2); another's
  distress on impingement, graded `[X]` (Q3); `SC-5` kept at `[I]` (Q4); sensitivity and meaning accuracy as two
  terms, the imagined part of a cue meaning only (Q5); cues engine-only for the LLM agents (Q8); a private hop drops
  the nonverbal channels, a witness reads all four (Q9); the owner's summary file's unfound quotations not used (Q10).
  Steps: (1) a ledger search for age and dependency before `SC-12` (Q7); (2) a spec revision with `SC-1` to `SC-12`;
  (3) `SC-2` and `SC-3` behind a switch off by default, the golden fingerprints holding; (4) the rest, in one
  engine-hash batch with one regeneration of the records (Q11). Switching it on by default is a later owner decision. *2026-11-07:* step 1 (the ledger search) done and the plan
  written, `docs/plan_t202_social_cues.md`; the owner approved it the same day with D1–D6 each as recommended
  (`SC-12` not built, T-204; `SC-9` on a stressor-load proxy; the emergence audit and sweep records xfail until the
  stage end). Step 2: `docs/spec_revision_55_draft.md`, approved by the owner 2026-11-08 (R55-Q1, R55-Q2 as recommended);
  the build started. *Done 2026-11-09:* steps 3 and 4 built as spec revision 55, the switch off by default; the engine
  hash moved from `0b363cb3186c` to `927550dfcb8a` and the records were regenerated once on it, with no existing row
  changed (`docs/version_3_revision_55_changes.md`). Recorded as found (the owner, 2026-11-08): `M11.C.50`, `M11.C.51`,
  `M11.C.53` and `M11.C.54` fail; `M11.C.52` passes at p 0.046 and its named mutant turns it red (T-208);
  `M11.C.55` passes and its named mutant reverses it (T-206). The departures from the draft and the choices taken as
  recommended under the owner's overnight instruction of 2026-11-08, pending the owner's review, are in
  `docs/spec_revision_log.md` (revision 55). Open: T-205, T-206, T-207, T-208; the switch's default is the owner's
  later decision.
- [ ] T-203 ⟨later⟩ **`KS10.1`'s uncertainty term as its own proposal (2026-11-07).** Uncertainty about whether and
  when a cue will come intensifies the reaction (`KS10.1`; `KS09.8`'s intermittent withdrawal). Left out of
  `docs/proposal_social_cues.md` by the owner's Q6 (2026-11-07): a separate mechanism, variance in the other's
  response rather than its value, with its own criterion.
- [ ] T-204 ⟨later⟩ **`KS09.6`'s parent-side separation as its own proposal (2026-11-07).** T-202's ledger search
  (`docs/plan_t202_social_cues.md` §2) found that a young child's dependency falling as it matures is supported
  (`KS09.6`, `K14.5`) with the cause on the parents' side: the parents tone down their togetherness urges as the child
  matures, and failure to do so prolongs dependency; `KS09.6` rejects developmental stages. The owner decided
  2026-11-07 (plan D5) not to build `SC-12`'s age term on the child, and to take this mechanism up as a separate
  proposal. `KS09.8`'s inverted projection (a child overfunctioning toward a dysfunctional parent) bears on it.
- [ ] T-205 ⟨later⟩ **Wire `M11.G.5`'s perceived-cue half into a report (2026-11-08).** `ReadingTap` in
  `src/bowen/readouts/roles.py` (the mean perceived attention and approval cue per pair, when the social cues are on)
  is tested by `tests/bowen/test_roles_readout.py` and has no other caller: no record, criterion or report runs it, so
  `M11.G.5` is built in code and reported nowhere (T-202's learning-qa pass, 2026-11-08). Wire it into the trace or the
  evaluation readout once the owner decides the switch's default, or record why not.
- [ ] T-206 ⟨later⟩ **Whether `M11.C.55`'s relief predictor is more than the policy's own input (2026-11-08).** The
  predictor is the child's learned value on the supplying kinds toward the parent, which is what the policy selects
  by (`M4.D.6`), so a pass may show only that the policy follows its own values (T-202's learning-qa pass, P7). A
  probe: the same comparison with learning removed (`M10.B.5`'s switch), which should leave the two predictors equal.
  `M11.5`'s row says its probe decides whether it is informative; until that probe runs, the row is not a finding.
- [ ] T-207 ⟨now⟩ **How the LLM experiment's committed runs follow an engine-hash move (2026-11-08).** T-202 moved the
  engine hash; the runs under `experiments/llm_agents/runs/` record `0b363cb3186c`. With `social_cues_enabled` off the
  engine computes the same runs (the golden fingerprints hold), but a replay restamps the hash into the summary, and
  `replay_all` marks a run whose summary hash changed `not_replayable`. Three tests are marked `xfail(strict=True)`
  until the owner decides (`test_lx25_committed_pilot_replays_fixed`, `test_lx10_run_record_fields`,
  `test_lx16_report_is_current`): restamp by replay with the experiment's hash rule changed to read behavior, not the
  hash; leave the runs on the old engine and say so in the report; or rerun live (a cost decision). Taken as
  recommended under the owner's overnight instruction, pending owner review: the xfail marks; no live call, no run
  rewritten.
- [ ] T-208 ⟨later⟩ **`M11.C.52`'s pass and its named mutant (2026-11-09).** On revision 55's engine `M11.C.52` passes
  with `escalation` -23.8 ± 24.4 (p 0.046, 100 seeds), and `cue-attribution-discount-removed` turns it red with
  -20.7 ± 25.4: the mutant leaves the effect near the same size and is caught because the interval crosses 0, not
  because the direction goes. The pass is not separated from the threshold, and the mutant is not shown to remove what
  the criterion reads. A probe: more seeds on both arms and on the mutant, and the same comparison with the cues'
  nonverbal channels set equal (`M4.C.1f`). Until then the row is a composite pass at the margin, not a finding.
