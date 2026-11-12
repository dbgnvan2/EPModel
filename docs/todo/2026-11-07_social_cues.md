# Social cues and signal channels — 2026-11-07

Items from `docs/archive/proposal_social_cues.md`, decided by the owner on 2026-11-07 (Q1–Q11, each as recommended).

- [x] T-202 ⟨later⟩ **Build the social cues proposal, after T-183's cross-check (2026-11-07).** The owner decided
  Q1–Q11 of `docs/archive/proposal_social_cues.md` on 2026-11-07, each as recommended: the four cues are readings that move
  `M4.C.1`'s two drives, not new drives (Q1); disapproval acts on both sides, impingement larger (Q2); another's
  distress on impingement, graded `[X]` (Q3); `SC-5` kept at `[I]` (Q4); sensitivity and meaning accuracy as two
  terms, the imagined part of a cue meaning only (Q5); cues engine-only for the LLM agents (Q8); a private hop drops
  the nonverbal channels, a witness reads all four (Q9); the owner's summary file's unfound quotations not used (Q10).
  Steps: (1) a ledger search for age and dependency before `SC-12` (Q7); (2) a spec revision with `SC-1` to `SC-12`;
  (3) `SC-2` and `SC-3` behind a switch off by default, the golden fingerprints holding; (4) the rest, in one
  engine-hash batch with one regeneration of the records (Q11). Switching it on by default is a later owner decision. *2026-11-07:* step 1 (the ledger search) done and the plan
  written, `docs/archive/plan_t202_social_cues.md`; the owner approved it the same day with D1–D6 each as recommended
  (`SC-12` not built, T-204; `SC-9` on a stressor-load proxy; the emergence audit and sweep records xfail until the
  stage end). Step 2: `docs/archive/spec_revision_55_draft.md`, approved by the owner 2026-11-08 (R55-Q1, R55-Q2 as recommended);
  the build started. *Done 2026-11-09:* steps 3 and 4 built as spec revision 55, the switch off by default; the engine
  hash moved from `0b363cb3186c` to `927550dfcb8a` and the records were regenerated once on it, with no existing row
  changed (`docs/archive/version_3_revision_55_changes.md`). Recorded as found (the owner, 2026-11-08): `M11.C.50`, `M11.C.51`,
  `M11.C.53` and `M11.C.54` fail; `M11.C.52` passes at p 0.046 and its named mutant turns it red (T-208);
  `M11.C.55` passes and its named mutant reverses it (T-206). The departures from the draft and the choices taken as
  recommended under the owner's overnight instruction of 2026-11-08, pending the owner's review, are in
  `docs/spec_revision_log.md` (revision 55). Open: T-205, T-206, T-207, T-208; the switch's default is the owner's
  later decision.
- [ ] T-203 ⟨later⟩ **`KS10.1`'s uncertainty term as its own proposal (2026-11-07).** Uncertainty about whether and
  when a cue will come intensifies the reaction (`KS10.1`; `KS09.8`'s intermittent withdrawal). Left out of
  `docs/archive/proposal_social_cues.md` by the owner's Q6 (2026-11-07): a separate mechanism, variance in the other's
  response rather than its value, with its own criterion.
- [ ] T-204 ⟨later⟩ **`KS09.6`'s parent-side separation as its own proposal (2026-11-07).** T-202's ledger search
  (`docs/archive/plan_t202_social_cues.md` §2) found that a young child's dependency falling as it matures is supported
  (`KS09.6`, `K14.5`) with the cause on the parents' side: the parents tone down their togetherness urges as the child
  matures, and failure to do so prolongs dependency; `KS09.6` rejects developmental stages. The owner decided
  2026-11-07 (plan D5) not to build `SC-12`'s age term on the child, and to take this mechanism up as a separate
  proposal. `KS09.8`'s inverted projection (a child overfunctioning toward a dysfunctional parent) bears on it.
- [x] T-205 ⟨later⟩ **Wire `M11.G.5`'s perceived-cue half into a report (2026-11-08).** `ReadingTap` in
  `src/bowen/readouts/roles.py` (the mean perceived attention and approval cue per pair, when the social cues are on)
  is tested by `tests/bowen/test_roles_readout.py` and has no other caller: no record, criterion or report runs it, so
  `M11.G.5` is built in code and reported nowhere (T-202's learning-qa pass, 2026-11-08). Wire it into the trace or the
  evaluation readout once the owner decides the switch's default, or record why not.
  *Closed 2026-11-09 (the owner, 2026-11-09, "go with your recommendations"):* `social_cues_enabled` stays off by
  default, because neither cue pass is yet a finding about the cues (T-206, T-208,
  `docs/review/2026-11-09_sc_pass_probes.md`). `M11.G.5`'s perceived-cue half stays out of the reports until the
  isolating arm of T-209 exists; `ReadingTap` keeps its unit test.
- [x] T-206 ⟨later⟩ **Whether `M11.C.55`'s relief predictor is more than the policy's own input (2026-11-08).** The
  predictor is the child's learned value on the supplying kinds toward the parent, which is what the policy selects
  by (`M4.D.6`), so a pass may show only that the policy follows its own values (T-202's learning-qa pass, P7). A
  probe: the same comparison with learning removed (`M10.B.5`'s switch), which should leave the two predictors equal.
  `M11.5`'s row says its probe decides whether it is informative; until that probe runs, the row is not a finding.
  *Probed 2026-11-09 (`docs/review/2026-11-09_sc_pass_probes.md`, `tools/sc_pass_probe.py`; open, closing is the
  owner's call):* with learning off the two predictors do not become equal: every learned value stays 0, the relief
  predictor names no one (0 hits) and the row fails (−0.080 ± 0.047, 150 seeds). With it on, the past twenty years'
  supply predicts the window only at chance (0.633 hits per seed, chance 0.633) while the year-20 learned value
  predicts it (1.067), and birth order is below chance (0.307; 0.080 with learning off). The pass is consistent with
  selection following learned value; it is not a finding that the supplier is learned rather than given. A wording for
  `M11.5`'s row and two criterion changes were proposed in the record; the wording is applied at spec revision 56 and the criterion changes are held (below).
  *Decided 2026-11-09 (the owner: "go with your recommendations", and for the criterion changes "Hold for next
  batch"):* the record's wording is in `M11.5`'s row (spec revision 56). Both criterion changes are decided: compare
  the relief predictor with chance and with prior supply, not with birth order; read the relief at a week before the
  window that is not the window's own start. They wait for the next batch that moves the engine hash, because the
  criterion's code and settings are in it and every record would rerun. Then: regenerate once, record the verdict as
  found with no tuning, and close this item. Open until then.
  *Built 2026-11-10 (spec revision 57, the version 2 definition plan's step 2):* `M11.C.55` now reads two directional
  readouts, the relief predictor against the main supplier over the weeks before the window and against the run's own
  chance, with birth order and the supplied count report-only, and reads the relief `relief_lead_weeks` (52) before
  the window. The engine hash moved from `927550dfcb8a` to `d3779684dd53`. The verdict is read at the plan's step 4's
  regeneration, as found; this item closes then.
  *Closed 2026-11-11 (the plan's step 4's regeneration, `docs/archive/version_3_revision_57_changes.md`):* `M11.C.55`
  passes as found, with no tuning, on 150 seeds: `predicted_main_supplier` +0.38 ± 0.157 (p 6.38e-06) against the
  supply over the weeks before the window, and `predicted_over_chance` +0.38 ± 0.111 (p 6.62e-09) against the run's
  own chance; `birth_order_hits` is report-only at +0 ± 0. Its named mutant `supply-by-birth-order` turns it red,
  where at revision 55 it reversed it. At revision 55 the single `predicted_main_supplier` readout stood at
  +0.76 ± 0.146 against birth order, so the pass holds under a reading that compares something else. What the pass
  means, and whether `M11.5`'s row changes, is the owner's at the plan's step 6; T-212 asks whether the two
  predictors should be read to the same week.
- [x] T-207 ⟨now⟩ **How the LLM experiment's committed runs follow an engine-hash move (2026-11-08).** T-202 moved the
  engine hash; the runs under `experiments/llm_agents/runs/` record `0b363cb3186c`. With `social_cues_enabled` off the
  engine computes the same runs (the golden fingerprints hold), but a replay restamps the hash into the summary, and
  `replay_all` marks a run whose summary hash changed `not_replayable`. Three tests are marked `xfail(strict=True)`
  until the owner decides (`test_lx25_committed_pilot_replays_fixed`, `test_lx10_run_record_fields`,
  `test_lx16_report_is_current`): restamp by replay with the experiment's hash rule changed to read behavior, not the
  hash; leave the runs on the old engine and say so in the report; or rerun live (a cost decision). Taken as
  recommended under the owner's overnight instruction, pending owner review: the xfail marks; no live call, no run
  rewritten.
  *Decided 2026-11-09 (the owner: "Replay by behavior (Recommended)"); closed 2026-11-09:* replay counts a run when
  its replay reaches the committed run's behavior (the summary without `engine_hash` and `summary_hash`, and the
  choices), and refuses one whose behavior differs, whatever the hash; a committed summary off its own hash, a replay
  ending in another status, and a complete run without its summary are refused
  (`experiments/llm_agents/summary.py::replay_differs`; tests `test_lx8_t207_*`, the behavior check proved by
  mutation). The engine hash stays recorded as provenance: `run.json` keeps the live run's, `summary.json` takes the
  engine it was replayed on. `runner replay` restamped all 31 replayed runs from `0b363cb3186c` to `927550dfcb8a`
  from their cached responses, with no live call: only those two summary fields moved. The three xfail marks are
  removed (`docs/plan_v3_llm_agent_experiment.md` §8).
- [x] T-208 ⟨later⟩ **`M11.C.52`'s pass and its named mutant (2026-11-09).** On revision 55's engine `M11.C.52` passes
  with `escalation` -23.8 ± 24.4 (p 0.046, 100 seeds), and `cue-attribution-discount-removed` turns it red with
  -20.7 ± 25.4: the mutant leaves the effect near the same size and is caught because the interval crosses 0, not
  because the direction goes. The pass is not separated from the threshold, and the mutant is not shown to remove what
  the criterion reads. A probe: more seeds on both arms and on the mutant, and the same comparison with the cues'
  nonverbal channels set equal (`M4.C.1f`). Until then the row is a composite pass at the margin, not a finding.
  *Probed 2026-11-09 (`docs/review/2026-11-09_sc_pass_probes.md`, `tools/sc_pass_probe.py`; open, closing is the
  owner's call):* on 500 seeds the pass holds clear of the test (−17.0 ± 10.1, p 0.0018), and the named mutant
  survives (−12.5 ± 10.2, p 0.031): it removes a detected +4.5 ± 1.5 of the effect, paired, and a detected residual
  remains. With the nonverbal channels equal the difference is −9.3 ± 8.6, p 0.053, not detected; with the cues off
  −22.2 ± 8.3, no different from the recorded effect (paired −5.2 ± 8.5). The pass is not a finding about
  self-attribution by channel alone; by `M11.5`'s row the row now comes back to the owner. A wording for the row was proposed in the record and is applied at spec revision 56 (below).
  *Closed 2026-11-09 (the owner, 2026-11-09, "go with your recommendations", relayed by another
  Claude session):* recorded as found: the pass holds, its named mutant survives, and the pass is not a finding about
  self-attribution by channel alone. The record's wording is in `M11.5`'s row and the criterion's row (spec revision
  56). An arm or mutant that isolates the by-channel discount is T-209.
- [ ] T-209 ⟨later⟩ **An arm or mutant that isolates `M4.C.6`'s by-channel discount in `M11.C.52` (2026-11-09).**
  T-208's probe (`docs/review/2026-11-09_sc_pass_probes.md`): the named mutant `cue-attribution-discount-removed`
  removes a detected +4.5 ± 1.5 of the pass and a detected −12.5 ± 10.2 remains, carried by `systems_perspective`'s
  other paths. Build a mutant that removes every path but the by-channel discount, or a baseline arm that has the
  other paths and lacks the discount, so that the residual paths no longer carry the pass (the owner, 2026-11-09).
  `M11.G.5`'s perceived-cue half enters the reports when this exists (T-205). Also read here, at more seeds than
  100, the emergence audit's carriers for `M11.C.52` under L0: on 2026-11-11 only C2's interval lay clear of L0's
  mean (T-213 (c), the owner 2026-11-11).
- [ ] T-210 ⟨later⟩ **`M11.C.55` under frozen learning (2026-11-09).** Learning on up to the relief's read week, off
  from then: does the window's supply follow the values held at that week without further learning? Left open by the
  owner, 2026-11-09; not run (`docs/review/2026-11-09_sc_pass_probes.md`, "What this record does not cover").
- [ ] T-211 ⟨later⟩ **Why birth order predicts the main supplier below chance on the twelve-person family
  (2026-11-09).** The first-born is the main supplier in years 20–30 for 0.307 parents per seed against chance 0.633
  with learning on, and 0.080 with it off (T-206's probe). Candidates: age at the window, leaving the parents'
  household. Left open by the owner, 2026-11-09; not run.
- [ ] T-212 ⟨later⟩ **Whether `M11.C.55`'s two predictors should be read to the same week (2026-11-10).** Spec
  revision 57 reads the relief `relief_lead_weeks` (52) before the window and counts the supply it is compared with
  over every week up to the window's start, so the prior-supply predictor sees 52 weeks the relief predictor does not:
  the arms differ in a second condition, the class T-194 and T-195 recorded (learnings P7). The asymmetry strengthens
  the baseline, so it runs against a pass and a pass stays readable, but a fail is not attributable to the predictor
  alone. `M11.5`'s row names it. The alternative is to count the supply over `[0, start − relief_lead_weeks)` as well,
  one line in `supplier_run`, which moves the engine hash and so waits for a batch that does. Raised by the
  learning-qa and correctness passes on revision 57's diff, 2026-11-10; the owner's call.
