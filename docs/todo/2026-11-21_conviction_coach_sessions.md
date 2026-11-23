# Spec revision 67 — conviction, and the coach acting only in sessions (2026-11-21)

Built from `docs/proposal_conviction_and_coach_sessions.md`, approved by the owner 2026-11-21. Open:

- [x] T-266 ⟨now⟩ **Regenerate every record on revision 67's engine (2026-11-21).** The batch moved the engine hash
  from `4aaa9029f63a`, so every engine-keyed record (ensemble, mutation, sweep, level occupancy, diagnostic, Phase D
  parts I and II, Phase D mutation, emergence audit, sensitivity, group G readout, T-176 trace) is stale. Mac Mini1's
  queue `~/regen67.sh` was started on 2026-11-21 at 20:07 (Mini1 time) at the revision 67 commit `0545f33`, engine
  `9321b8523500`, in `~/ProjectsMini1/EPModel-regen67`, logging to `~/regen67.log`; when it ends, merge its records here,
  remove each strict-xfail mark naming T-266 as its record is regenerated, and report each verdict move to the owner
  with T-254's seed-block rule. The LLM experiment is frozen at revision 64 (T-265) and is not in the queue.
  **Closed 2026-11-22.** All 12 steps exited 0 (queue done 14:44 Mini1 time); the records are merged and the
  eleven T-266 marks removed. The verdict moves are in `docs/version_3_revision_67_changes.md` (revision 66's page
  frozen at `41b14c3` and archived) and listed for the owner in T-267. The emergence audit's scope follows the
  verdicts: 13 rows to 11 (`M11.C.12`, `M11.C.51`, `M11.C.7[one spouse over sided]` leave, `M11.C.17` enters).

- [ ] T-267 ⟨now⟩ **Revision 67's verdict moves (2026-11-22).** From `docs/version_3_revision_67_changes.md` (T-266),
  for the owner. Criteria: `M11.C.51` PASS→FAIL (p 0.048→0.081), `M11.C.17` FAIL→PASS (p 0.10→0.047), `M11.C.12`
  PASS→FAIL (p 0.042→0.062), `M11.C.7[one spouse over sided]` PASS→FAIL (p 0.008→0.092). Each now sits with p
  between about 0.01 and 0.2, so under T-254's rule none is read as a change of the model until fresh, disjoint seed
  blocks on revisions 66 and 67 show it (`tools/t253_seed_blocks.py`, Mini1). Mutants: `level-blind · M11.C.16`
  red→survived; `coach-position-blind · M11.C.7[joint over sided]` red→survived (deviation fall p 0.0008→0.00006, outside
  the band). Sweep: `M11.C.50` PASS→FAIL at `learning_rate` 0.1 and 0.4, `M11.C.54` FAIL→PASS at `policy_temperature`
  2.0. Level occupancy: 4 rows' values moved. None was acted on.
