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

- [x] T-267 ⟨now⟩ **Revision 67's verdict moves (2026-11-22).** From `docs/version_3_revision_67_changes.md` (T-266),
  for the owner. Criteria: `M11.C.51` PASS→FAIL (p 0.048→0.081), `M11.C.17` FAIL→PASS (p 0.10→0.047), `M11.C.12`
  PASS→FAIL (p 0.042→0.062), `M11.C.7[one spouse over sided]` PASS→FAIL (p 0.008→0.092). Each now sits with p
  between about 0.01 and 0.2, so under T-254's rule none is read as a change of the model until fresh, disjoint seed
  blocks on revisions 66 and 67 show it (`tools/t253_seed_blocks.py`, Mini1). Mutants: `level-blind · M11.C.16`
  red→survived; `coach-position-blind · M11.C.7[joint over sided]` red→survived (deviation fall p 0.0008→0.00006, outside
  the band). Sweep: `M11.C.50` PASS→FAIL at `learning_rate` 0.1 and 0.4, `M11.C.54` FAIL→PASS at `policy_temperature`
  2.0. Level occupancy: 4 rows' values moved. None was acted on.
  **Seed blocks 2026-11-22** (`docs/review/2026-11-22_t267_seed_blocks.md`, from `tools/t267_seed_blocks.py` on Mac
  Mini1: five fresh disjoint blocks of the record's size from seed 43000, revisions 66 and 67; pooled p over the blocks).
  None of the nine moves is a change of the model.
  *`M11.C.51`, PASS to FAIL: noise.* No block passes on either engine (pooled `distance_conflict_share` +0.0001, p 0.70,
  at revision 66; +0.002, p 0.39, at revision 67). Revision 66's PASS at p 0.048 was seed noise, as T-264 found; the
  FAIL is the verdict both engines give.
  *`M11.C.17`, FAIL to PASS: not a change; the effect holds on both engines.* Each engine passes 2 of 5 blocks, with
  pooled `coached_rise` +0.0023 (p 5e-5) at revision 66 and +0.0022 (p 9e-5) at revision 67. The record's move is the
  threshold at 150 seeds, where the effect is too small to pass reliably.
  *`M11.C.12`, PASS to FAIL: not a change; the effect holds on both engines.* Pooled `marital_conflict` is +1.16 (p
  0.0003) at revision 66 and +1.15 (p 0.0001) at revision 67, with 1 and 2 of 5 blocks passing. The record's move is
  the threshold at 100 seeds.
  *`M11.C.7[one spouse over sided]`, PASS to FAIL: not a change; the effect holds on both engines.* Pooled
  `deviation_fall` is +0.066 (p 0.0003) at revision 66 and +0.058 (p 0.002) at revision 67, with 1 and 2 of 5 blocks
  passing, and `landed_contacts` holds on both (+10.4 and +10.5). The record's move is the threshold at 150 seeds.
  *`coach-position-blind` on `M11.C.7[joint over sided]`, red to survived: noise; the criterion still catches it.*
  Pooled `landed_contacts` is +0.02 (p 0.48) at revision 66 and +0.04 (p 0.44) at revision 67, so the mutant is caught
  in 5 of 5 and 4 of 5 blocks; the record's +0.62 (p 0.0098) does not hold. A probe (40 seeds an arm) found a small new
  path: the sided coach starts more of its own I-POSITION sequences, and under `M5.D.9b` their steps take session weeks
  and push out about 0.5 contacts a run (0.16 at revision 66), too few to change the landed count.
  *`level-blind` on `M11.C.16`, red to survived: noise.* The mutant is caught in all 10 blocks (pooled
  `repertoire_entropy` +0.0004, p 0.68, at revision 66; −0.0003, p 0.36, at revision 67). Revision 67's new level read,
  conviction's solid part, is one the mutant already fixes at 50, so no level effect leaks through.
  *Sweep: `M11.C.50` at `learning_rate` 0.1 and 0.4, and `M11.C.54` at `policy_temperature` 2.0: noise.* No block passes
  on either engine (pooled p 0.96 and 0.97 at 0.1, 0.26 and 0.26 at 0.4, 0.62 and 0.78 at 2.0). Revision 66's two
  `M11.C.50` passes and revision 67's `M11.C.54` pass were seed noise, and both criteria fail at the central setting.
  *Closed 2026-11-25 by the owner: the finding is accepted; none of revision 67's nine moves is a change of the model.*
