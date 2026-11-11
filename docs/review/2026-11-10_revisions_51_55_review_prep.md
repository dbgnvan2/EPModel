# Spec revisions 51–55: the parts pending the owner's review, prepared

**Status: prepared 2026-11-10; decided by the owner 2026-11-10 (§10); written into the spec, the revision log and the status lines at spec revision 58 (2026-11-11), after the version 2 definition plan's step 4 landed.** Prepared at spec revision 57,
engine hash `d3779684dd53`, commit `3d84065`. This document changes no rule, class, verdict, record or status line.
A recommendation is Claude's; a status line changes only when the owner decides the item.

## How to use this

There are 21 items, numbered by revision, and 4 findings this review made that are on no pending list (§7). Each
item says what changed, what evidence it rests on, the result of five checks, whether revision 56 or 57 has since
changed it, and a recommendation with the cost of reversing it. §8 is the one table.

**Suggested order.** Six entries need a real decision: R53-1, R53-2, R55-14, F-1, F-2 and F-4. Three items need no
decision because the owner has since decided them or made the choice personally: R55-4, R55-11 and R55-12. The other fifteen
are routine and are recommended for approval as they stand; F-3 is optional.

**The five checks**, named in each item by these letters:

- **(a)** Is a premise reported as a finding (`M11.5`)?
- **(b)** Does an arm differ in more than the condition compared (learnings P7)?
- **(c)** Is a `[K-ext]` claim attributed to Bowen?
- **(d)** Is a constant presented as sourced when it is `[I]`?
- **(e)** Is anything quoted verbatim from an ASR source (`Tape N`, `KB`)?

**Where the numbers come from.** Every number below is copied from a committed record or from the todo record's
probe table, named beside it. None is computed here. Where this document compares two record numbers (for example,
says two intervals overlap), that is a reading of the two numbers shown, and it says so.

**Two limits on all of it.**

1. Every committed record still stands on revision 55's engine (`927550dfcb8a`; Phase D mutation record
   `fa37a4c64ee1`). Revision 57 moved the hash, and the records are regenerated at the version 2 definition plan's
   step 4. Revision 57 changed only `M11.C.48`'s and `M11.C.55`'s readouts, so the other rows are expected to
   reproduce, but that is not verified until step 4.
2. The probes of T-196 and T-197 did not commit their per-seed rows (F-3). Their numbers are checked here against
   the tables in `docs/todo/2026-11-03_version_3_emergence_audit.md`, not against an artifact.

## What is not in scope

These parts of revisions 51–55 were decided by the owner and are not reviewed again here:

- Revision 51's wording itself (T-194's four answers, the owner through `AskUserQuestion`, 2026-11-05). Only its
  order is pending (R51-1).
- The decision to run T-196 and T-197 (the owner, 2026-11-05). What is pending is folding their findings into the
  spec (R52-1, R52-2).
- Revision 55's rules: the proposal's Q1–Q11 (2026-11-07), the plan's D1–D6 (2026-11-07), and the draft as a whole
  with R55-Q1 and R55-Q2 (2026-11-08). What is pending is the seven departures from the draft, six other choices,
  and one consequence left for review (R55-1 to R55-14).
- Revision 55's results, "recorded as found (the owner, 2026-11-08)", and revisions 56 and 57.

---

## 1. Revision 51 (2026-11-05, T-194, T-195)

### R51-1 — the wording was written before the two probes ran

1. **What changed.** Nothing in a rule. T-195's wording (`M11.5`'s rows for `M11.C.2` and `M11.C.48`, and each
   criterion's row) was written at revision 51, ahead of T-196 and T-197, which were then open. The brief had put the
   order to the owner, who was away; Claude took "wording first, probes folded in later".
2. **Evidence.** `docs/spec_revision_log.md`, revision 51; `docs/todo/2026-11-03_version_3_emergence_audit.md`,
   T-195. No record number rests on the order.
3. **Checks.** (a)–(e): not applicable; an order of work, not a claim.
4. **Since.** Revision 52 folded both probes in a day later, so the order has no remaining effect. It had one cost:
   revision 51 wrote "among them" before the tie classes, which the probe showed was wrong (the baseline holds only
   the 1.0 and 0.6 ties), and revision 52 removed it.
5. **Recommendation: approve.** Nothing is left to reverse. Reversing costs nothing and does not move the hash.

---

## 2. Revision 52 (2026-11-06, T-196, T-197, T-198)

### R52-1 — T-196's probe folded into `M11.5`'s row for `M11.C.48` (the Phase B family)

1. **What changed.** `M11.5`'s row for `M11.C.48` gained: matched on the cut tie's class no difference is detected;
   the cut and baseline ties are acted on by both members about equally often (0.77 against 0.78); at every matched
   cutoff all the uncut baseline ties are cut later in the run, so a never-cut baseline cannot be formed. Revision
   51's "among them" was removed. The criterion's row is unchanged. Wording in `M11` only.
2. **Evidence.** The record: `docs/phase_d2_ensemble_record.md`, `M11.C.48` FAIL, 200 seeds, `negative_run`
   +0.0286 ± 0.14, p 0.108. The probe (`tools/c48_carrier_probe.py --family phase_c`, 200 seeds, T-196's table):
   the row reproduced at +0.029 ± 0.136 on 808 matched first cutoffs; matched on class −0.004 ± 0.186; share of
   ties both members act on, cut / baseline, 0.772 / 0.779; baseline ties cut later in the run, 1711 of 1711.
   `M11.5`'s three statements match the table.
3. **Checks.** (a) No: the row is composite, and the wording says the fail is *not* a finding about the run before
   a cutoff, which under-claims rather than over-claims. (b) Yes, and that is what the wording records: the arms
   differ in tie class and the baseline ties are all cut later. One gap: the class matching reads 572 of the 808
   cutoffs (the rest have no uncut tie of their class), and `M11.5`'s sentence does not give that count (learnings
   P2). (c), (d), (e): no theory claim, constant or quotation.
4. **Since.** Revision 57 matched the baseline on how many members act on the tie, on both families. `M11.5`'s
   sentence "on the Phase B family … the arms are unmatched in the same way" is now only partly true: sides are
   matched, class and cut-later are not. Revision 57 says both rows are reread at the regeneration, and that the
   owner's acceptance of the Phase B fail (2026-11-03) stops applying if the verdict moves. T-196's own
   sides-matched reading on the old engine was +0.057 ± 0.154 (the probe's number, not a verdict).
5. **Recommendation: approve.** The wording matches the probe and claims less, not more. Have the Phase B sentence
   reread at the plan's step 4 with the regenerated row, and add "572 of 808 cutoffs" then. Reversing is wording in
   `M11` only and does not move the hash.

### R52-2 — T-197's probe folded into `M11.5`'s row for `M11.C.2` (the twelve-person family)

1. **What changed.** `M11.5`'s row for `M11.C.2` gained: the most-projected child is more often the eldest, Leo
   (8 at week 0, observed under 20 for about 12 years); with which child is picked held, no direction is detected
   over the three children (+0.17 ± 0.47 per act per year, against +0.73 ± 0.47 unmatched); between the two
   children born into the run a direction remains with their early state held, at the edge of detection
   (+0.80 ± 0.63); the pass is not separated from which child is picked. Wording in `M11` only.
2. **Evidence.** The record: `docs/phase_d2_ensemble_record.md`, `M11.C.2[twelve-person family]` PASS, 150 seeds,
   `symptom_load` +2.22 ± 1.5, p 0.0019. The probe (`tools/c2_matched_probe.py`, T-197's table): the 150 seeds
   reproduce the record (+2.221 ± 1.536); on 600 seeds the row is +1.35 ± 0.82; Leo is the most-projected child in
   286 seeds and the least-projected in 134; the three numbers in `M11.5` are the table's 600-seed column.
3. **Checks.** (a) No: composite, and the pass is stated as not separated from which child is picked. (b) Yes, and
   that is what the wording records. Two things the row leaves out that the todo record states: the +0.17 and
   +0.73 intervals overlap and the probe does not test their difference, so "against" is not a detected drop; and
   the 600-seed row (+1.35 ± 0.82) is smaller than the record's 150-seed row. (c), (d), (e): none.
4. **Since.** Revision 53 restated the same criterion's mutation cell (R53-1). Revisions 56 and 57 did not touch
   `M11.C.2`.
5. **Recommendation: approve.** The conclusion ("not separated") is the weak one the numbers support. Optional,
   wording only: add that the two slopes are not tested against each other. Reversing does not move the hash.

---

## 3. Revision 53 (2026-11-06, T-171)

### R53-1 — `M11.C.2`'s mutation cell and the sentence after its revision 31 measurement

1. **What changed.** The criterion row's mutation cell now says that on the twelve-person family
   `witness-path-removed` turns the pass red and `weakness-focus-and-steering-removed` does not. The revision 31
   sentence (focus removed turns it red; the pass is carried by the parents' focus) stays as a dated measurement,
   and a sentence after it says the current record has the reverse, "so the pass needs the witness path and is not
   shown to be carried by the parents' focus". Wording in `M11` only.
2. **Evidence.** `docs/phase_d_mutation_record.md`, 150 seeds each, on `M11.C.2[twelve-person family]`:
   `witness-path-removed` FAIL, `symptom_load` +0.216 ± 1.8, p 0.42, **red**;
   `weakness-focus-and-steering-removed` PASS, +1.26 ± 1.5, p 0.045, **survived**. The revision 31 numbers kept in
   the row: focus removed +0.73 ± 1.8 (red), witness path removed +1.72 ± 1.5 (survived).
3. **Checks.** (a) No. (b) Not an arm question, but a related one. The restated cell is correct as a statement of
   the record. The added sentence goes further than the record: "needs the witness path" rests on one mutant
   falling on one side of the test and the other falling on the other side at p 0.045. Read side by side, each
   mutant's interval contains the other's mean, in the current record and in revision 31's, where the verdicts were
   the other way round. Neither record compares the two mutants with each other. T-197's 600-seed value for the
   unmutated row (+1.35 ± 0.82) is also about where the surviving mutant sits (+1.26 ± 1.5). So the record does not
   show which path carries the pass; revision 31's sentence made that claim one way and revision 53's makes it the
   other way. **Whether the carrier changed between revision 31 and now, or these are two noisy readings of the
   same thing, cannot be assessed from the records.** A paired comparison of the two mutants on the same seeds, at
   more seeds, would settle it (T-197 measured about 9 minutes per 150 seeds on 8 workers for the unmutated row).
   (c), (d), (e): none.
   A second point: the cell now holds revision 31's claim and its reversal one after the other. Revision 35 moved
   such dated narratives out of requirement bodies into `docs/spec_revision_log.md`.
4. **Since.** Revisions 56 and 57 did not touch it. The mutation record is regenerated at step 4.
5. **Recommendation: approve with a change.** Keep the restated mutation cell. Replace "so the pass needs the
   witness path and is not shown to be carried by the parents' focus" with what the record shows: which mutant
   turned it red and which survived, with their numbers, and that the two are not separated from each other. Move
   the revision 31 mutation sentence to the revision log. Both are wording in the spec, with `docs/spec_index.jsonl`
   and `docs/spec_rules/M11.md` regenerated; the hash does not move. Reversing revision 53 outright would restore a
   cell the record contradicts, so it is not recommended.

### R53-2 — the version 2 note on `M11.C.2` in `docs/version_2_dispositions.md`

1. **What changed.** A `note` row, `decided` cell "Claude's recommendation 2026-11-06, pending owner review". It
   says version 1's reason names focus on weakness as what proves the twelve-person pass, that the current record
   has the reverse, "so the pass needs the witness path (`M1.F.5`) and is not shown to be carried by focus on
   weakness", that neither family's comparison is separated from which child is picked, and that the Phase C
   family's fail stands as version 1 gives it.
2. **Evidence.** As R53-1; the Phase C family's row is `docs/phase_d_ensemble_record.md`, `M11.C.2` FAIL, 150
   seeds, `symptom_load` −0.623 ± 3.1, p 0.671.
3. **Checks.** As R53-1: the same sentence, the same over-reading of one red and one survived mutant.
4. **Since.** Unchanged by revisions 56 and 57. `docs/version_2_status.md` is generated from this file.
5. **Recommendation: approve with the same change as R53-1**, so the note and the spec say the same thing. Wording
   in a dispositions file, the status page regenerated; the hash does not move.

### R53-3 — the version 2 notes on `M11.C.29`, `M11.C.42`, `M11.C.44` and `M11.C.45`, and how the status tools show a note

1. **What changed.** Four `note` rows in `docs/version_2_dispositions.md`, each "pending owner review", each
   saying version 1's reason predates stage 3 and giving the current state. `tools/version_2_status.py` and
   `tools/spec_coverage.py` now show a version 2 note beside version 1's disposition, not in place of it; before,
   an applied note would have dropped the five accepted findings from the page (tests in
   `tests/test_version_2_status.py` and `tests/bowen/test_spec_coverage.py`).
2. **Evidence**, each checked against the record it names:

   | Row | The note's claim | The record |
   |---|---|---|
   | `M11.C.29` | no genuine `I-POSITION` completes, so the precondition is unmet | `docs/phase_c_ensemble_record.md`: FAIL, 50 seeds, `completions` +0 ± 0, "move never occurred" |
   | `M11.C.42` | still fails after group A; the current cause is not traced | FAIL, 50 seeds, `triangle_reuse` −0.06 ± 0.39, p 0.613 |
   | `M11.C.44` | passed in revision 38's record with `one-sided-deviation` red and `deviation-inverted` surviving; fails since group A | `docs/version_2_revision_39_changes.md`: PASS → FAIL, and both mutant rows under "Rows only at revision 38"; now FAIL, 150 seeds, `outside_inside_ratio` +0.0221 ± 0.096, p 0.344 |
   | `M11.C.45` | still fails; `one-sided-deviation` names it and does not run while it fails | FAIL, 150 seeds, `triangle_rate` +0.00136 ± 0.0054, p 0.296; `docs/phase_c_mutation_record.md` lists the mutant for it |

3. **Checks.** (a) No: each is a failing composite, and three notes say outright that the current cause is not
   traced. (b) Not applicable. (c), (d), (e): none. Not checked here: T-138's trace, which the `M11.C.29` note
   cites as not rerun on the current engine.
4. **Since.** Revision 54 corrected `M11.5`'s wording for `M11.C.42` to agree with this note (R54-1). Revisions 56
   and 57 did not touch these rows.
5. **Recommendation: approve.** Each note agrees with its record and claims no cause it has not traced. What they
   leave open is whether to trace the three causes, which is a work decision, not a wording one. Reversing removes
   four rows and the tools' change; it does not move the hash.

---

## 4. Revision 54 (2026-11-06, T-199)

### R54-1 — `M11.5`'s note for `M11.C.42` names the relieved party correctly

1. **What changed.** "returned to a relieved seeker" became "returned to its sender by a relieved receiver".
   Wording in `M11` only.
2. **Evidence.** `M4.C.1d` (spec line 719): a move with positive contact that "relieves the target" returns a share
   of the contact the target took in to "the sender's felt contact". The relieved party is the receiver.
3. **Checks.** (a)–(e): none applies. `M4.C.1d`'s `reciprocal_contact_share` is tagged `[I]` in its own rule.
4. **Since.** Unchanged.
5. **Recommendation: approve.** The old wording contradicted the rule it cited. Reversing restores an error; no
   hash move either way.

---

## 5. Revision 55 (2026-11-08, T-202): the seven departures from the approved draft

Checks (c), (d) and (e) for revision 55 as a whole are in §7's note under F-2, since they concern the rules, not
these choices. In this section and the next, "checks: none" means none of the five applies to the choice.

| Item | What changed | Evidence | Checks | Since (revisions 56, 57) | Recommendation; cost of reversing |
|---|---|---|---|---|---|
| **R55-1** | The supply readout is `M11.G.5`, not the draft's `M11.G.4` | `M11.G.4` already names another rule (the evaluation carries `M11.F`'s framing block; spec line 1594) | none | unchanged | **Approve.** The draft's ID was taken. Renumbering is wording plus test names; no hash move |
| **R55-2** | The 31 constants are rows of `config/bowen/constants.md`, not a new `config/bowen/social_cues.md` | 31 rows cite the cue rules, each `[I]`, each logged in `constants_changes.md` (lines 223 on) | (d) no: every row is `[I]` | unchanged | **Approve.** One register and one parser. Moving them to a new file changes `config/bowen/` and **moves the hash** |
| **R55-3** | `M11.C.55`'s window is stated in run years (`window_from_year` 20 of 30) | `config/bowen/criteria.md` line 288, `[I]` | (d) no | revision 57 kept it and added `relief_lead_weeks` 52, `[I]` | **Approve.** It matches how the Phase D part II runs count. Restating it is a criterion setting in `config/bowen/` and **moves the hash** |
| **R55-4** | `M11.C.52`'s mutant is `cue-attribution-discount-removed` | the revision log records it as the owner's pick, 2026-11-08 | none | revision 56 recorded that it survives at 500 seeds (−12.5 ± 10.2, p 0.031) and opened T-209 | **No decision needed**: the owner's own pick, and T-209 holds what is open |
| **R55-5** | The criteria's rows name their tests in full | the six rows in `docs/spec_rules/M11.md` | none | unchanged | **Approve.** No hash move |
| **R55-6** | The cue-reading draw is a register row of `M3.D.4b` ("Cue perception", dyad-keyed) | `docs/spec_rules/M03.md` line 52 | none | unchanged | **Approve.** `M3.D.4b` requires every draw class to be in that table. No hash move for the wording |
| **R55-7** | The work landed as one commit (`a19a690`), not through the open-batch machinery | `tests/open_batch.py`: `OPEN` stays None | none | revision 57 did the same | **Approve.** Nothing to reverse; the commit exists |

---

## 6. Revision 55: the six other choices and the one consequence left for review

| Item | What changed | Evidence | Checks | Since (revisions 56, 57) | Recommendation; cost of reversing |
|---|---|---|---|---|---|
| **R55-8** | `M11.C.55`'s relief predictor names the child with the highest summed relief toward the parent, and no child on a tie. The first build also required the value to be above 0, which left most parents unnamed; that was removed as a defect | `docs/review/2026-11-09_sc_pass_probes.md`, T-206's table: as recorded the relief names 1.993 of 2 parents per seed; with learning off it names 0.000 | (b) the tie rule is why learning-off reads as a fail by definition, which the probe record states | revision 57 kept the rule ("a tie for the highest relief … is a miss") and changed what the predictor is compared with | **Approve.** Naming no one on a tie avoids a tie-break by birth order or position. Changing it edits the criterion and **moves the hash** |
| **R55-9** | `cues.py`'s 2.0 is on the structural-literal list | `tests/bowen/test_engineering_gates.py` line 81: a uniform on [0, 1) mapped to [−1, 1) for `M4.C.12`'s spread | (d) no: it is arithmetic, not a parameter | unchanged | **Approve.** Promoting it to config would **move the hash** for no gain |
| **R55-10** | `RunState.cue_readings` is outside the canonical state | `src/bowen/engine/state.py` line 191; it is emptied at each week's appraisal (`appraise.py` line 467) and filled as a per-delivery cache (`cues.py` lines 202, 227). Read here from the code; nothing was run | none | unchanged | **Approve.** It is a within-week cache of values the seed determines. Adding it to the canonical state changes `src/bowen/` and **moves the hash** |
| **R55-11** | The emergence audit and the version 2 sweep were not rerun; three strict expected failures stand (the two is-current tests and the audit's scope test, 13 rows against the plan's 11) | `docs/version_3_revision_55_changes.md`, "Not compared here"; `tests/test_emergence_audit.py` line 107 | none | **Superseded.** The owner's answers of 2026-11-09 to the definition plan: Q7 (a), the audit's L0 and carriers rerun after the regeneration; the sweep reruns at step 7 | **No decision needed.** The marks clear at the plan's steps 5 and 7 |
| **R55-12** | The LLM experiment's three tests marked strict expected failures; no live call, no run rewritten | `docs/todo/2026-11-07_social_cues.md`, T-207 | none | **Superseded.** The owner chose "Replay by behavior" (2026-11-09); T-207 closed and the marks came off. Two of its record tests are expected failures again under revision 57, for the regeneration, a different reason | **No decision needed** |
| **R55-13** | The new unit tests are named requirement first (`test_m4c12_sc6_…`); the plan and the unit mutation record were renamed to match | `tools/spec_coverage.py` reads a test's ID from the start of its name; 26 such tests in `tests/bowen/test_social_cues.py` | none | unchanged | **Approve.** Required by the coverage tool. No hash move |

### R55-14 — with the cues on, the words part of a received move is no longer reattributed (`M4.C.6`)

1. **What changed.** With `social_cues_enabled` on, an agent with `systems_perspective` attributes to its own prior
   emission only the nonverbal part of a received move, in proportion to its own recent nonverbal leak; the words
   part (the move kind's components) is not reattributed. So such an agent, facing a sender whose nonverbal
   channels are neutral, loses the reattribution it has with the cues off. This follows the approved draft (its
   line 158); the revision log lists it for the owner's review as a consequence, not as a departure.
2. **Evidence.** `docs/review/2026-11-09_sc_pass_probes.md`, T-208, `M11.C.52` on 500 seeds: as recorded
   −17.0 ± 10.1 (p 0.0018); with the cues off, under the whole-event rule, −22.2 ± 8.3 (p 5.9e-7); paired, the
   change is −5.2 ± 8.5, so no difference is detected between the cues off and on. With the nonverbal channels
   equal the effect is −9.3 ± 8.6 (p 0.053). The named mutant removes +4.5 ± 1.5 of the effect and survives.
3. **Checks.** (a) No: `M11.C.52` is composite, and revision 56 already says its pass is not a finding about
   self-attribution by channel alone. (b) The probe record lists five other reads of `systems_perspective` that may
   carry the rest; which do is not measured (T-209). (c) No: `KS17.3` is `[K]` and is cited as Kerr's. (d) No: the
   share's form is tagged `[I]`. (e) No: `KS17.3` is a written source. **Not assessable from the records:** the
   effect of this rule outside `M11.C.52`'s dyad. Every other criterion that uses `systems_perspective` runs with
   the cues off.
4. **Since.** Revision 56 wrote T-208's probe into `M11.5` and kept the switch off by default (T-205). Revision 57
   did not touch it.
5. **Recommendation: approve as built, and decide it again with T-209.** It is what the approved draft says, no
   loss is detected where it was measured, and the switch is off. It should be settled before any decision to turn
   the switch on. Reversing changes `src/bowen/engine/appraise.py` and **moves the hash**; runs with the switch off
   would not change, but every record would be regenerated.

---

## 7. Found in this review, on no pending list

### F-1 — two rules' text sits under the wrong ID in the spec (a departure from the approved draft, not listed)

In `docs/bowen_agent_model_spec_v2.md` since revision 55's commit (`a19a690`):

- `M4.C.1e` ends at line 721 with a colon. `M4.C.1f` follows at line 723. The cue-weight table and the paragraph
  "The four cues are readings of a received signal …" (lines 725–732) belong to `M4.C.1e` and sit under `M4.C.1f`.
- `M4.C.14`'s lead sentence is line 770. `M4.C.15` follows at line 772. `M4.C.14`'s two bullets and its closing
  paragraph (lines 774–777) sit under `M4.C.15`.

The approved draft (`docs/spec_revision_55_draft.md`, lines 101–133 and 204–221) has both in the right order. No
word is missing and nothing a run computes is affected. The generated views carry the error:
`docs/spec_index.jsonl` gives `M4.C.1e`'s sources and its `[D]`, `[X]`, `[T]` tags to `M4.C.1f`, shows `M4.C.1e`
with no source, and shows `M4.C.14` with no tag and no source; `docs/spec_rules/M04.md` shows the same order.

**Recommendation: fix.** Move the two blocks back under their own rules and regenerate the two views. Wording
position only; the spec is not in the engine hash, so it does not move.

### F-2 — `KS23.1` is graded `[K]` in the cue list's sources; its chapter is `[K-ext]`

The paragraph that lists the sources for the four cues (now under `M4.C.1f`, see F-1) reads "`KS01.7`, `KS07.6`,
`KS09.4`, `KS10.1`, `KS23.1` (`[T]` `[K]`)". `docs/theory/kerr_book/ks23.md` opens with the rule that the chapter is
Kerr's own proposed addition and everything in it is graded `[K-ext]` unless it is a cited external study. `KS23.1`
restates the four cues, which the spec already cites from `KS10.1` as `[K]`. The claim is attributed to Kerr, not to Bowen, so
check (c) is not failed; the grade is wrong by the extraction file's own rule.

**Recommendation: fix.** Either drop `KS23.1` from that list (the other four carry it) or tag it `[K-ext]`. Wording
only; no hash move. Not checked: the spec's other citations of `KS23.1`, which predate revision 55.

**The rest of checks (c), (d), (e) for revision 55's rules:**

- (c) The rules cite `FE03.3`, `KS01.7`, `KS07.6`, `KS08.4`, `KS09.4`, `KS10.1`, `KS17.3` as `[K]` and `ch18.md`
  line 42 as `[B]`. None attributes a Kerr claim to Bowen. No cited entry but `KS23.1` is in a `[K-ext]` chapter.
- (d) All 31 constants are `[I]` in the register. `M4.C.11` states that its direction and size are both `[I]` and
  that no extraction entry supports it; `M4.C.1e`'s distress side is `[X]`. No constant is presented as sourced.
- (e) Sixteen quoted phrases in `M1.F.1c`, `M4.C.1e`, `M4.C.6`, `M4.C.11` to `M4.C.15` and `M9.9` were each found,
  word for word, in a written source's extraction file (`fe03.md`, `ks01.md`, `ks07.md`, `ks08.md`, `ks09.md`,
  `ks10.md`, `ks17.md`, `ch18.md`). None is from a tape or an interview. `M4.C.5`, a line stamped revision 55,
  mentions the KB05 interview in paraphrase. Not searched for quotations: the sentences revision 55 added to nine
  other older lines (`M1.A.14b`, `M1.F.1`, `M1.F.4`, `M4.C.1a`, `M4.C.9`, `M7.E.1d`, `M10.C.1a`, `M13`'s two rows and
  the register row for `cue_beliefs`).

### F-3 — the per-seed rows of T-196's and T-197's probes are not committed

`M11.5`'s rows for `M11.C.48` and `M11.C.2` cite numbers from these two probes. Both todo entries say "rows:
scratchpad only, not committed". The later probe record (`docs/review/sc_pass_probes/`) committed its rows.
Revision 57 changed `M11.C.48`'s readout, so rerunning `tools/c48_carrier_probe.py` today reads a different
baseline and will not reproduce T-196's table.

**Recommendation: optional.** Accept the numbers as dated measurements, or rerun each probe once at the commit
before `3d84065` (about 7 minutes for T-196; about 9 minutes per 150 seeds for T-197, as each todo entry measured)
and commit the rows. No hash move.

### F-4 — `M11.C.53`'s too-little limb reads like a premise limb, and it is the limb reported as holding

`M11.5` classes `M11.C.53` as composite. Its first limb is that the other spouse's too-little deviation on the
marital tie is higher in the arm with the stressor. `M4.C.15` lowers the attention cue with the sender's stressor
load, and `M4.C.1e` has a lower attention cue lower felt contact by `cue_attention_contact_weight`. Two rules, read
together, nearly state the limb. `M11.5` defines a premise as a result a rule states or nearly states, and requires
a mixed criterion to report each limb under its own class, as `M11.C.49` does.

The record (`docs/phase_c_ensemble_record.md`): `M11.C.53` FAIL; `too_little_on_marital_tie` +0.01007 ± 0.00661,
p 7.98e-05; `pursue_conflict_toward_stressed` +0.01 ± 0.19, p 0.416. The revision log reports this as "its
too-little limb holds, its pursuit limb does not". Across the six sweep settings listed in
`docs/version_3_revision_55_changes.md` (from `docs/phase_c_sweep_record.md`), the too-little limb's mean is positive
at every one and the pursuit limb's is not.

`M11.5`'s note already says the arms also differ in the stressed spouse's own anxiety and moves, and that only the
named mutant shows whether `M4.C.15` carries it. The mutant (`cue-attention-load-removed`) does not run while the
row fails, so **this cannot be assessed from the records.** What would settle it: run that mutant on the too-little
limb alone, as a probe outside the hash, the way T-208 probed `M11.C.52`. If the mutant removes the limb, it is a
premise limb.

**Recommendation: decide the class after that probe; until then, do not report the too-little limb as a partial
finding.** A class change is wording in `M11.5` and does not move the hash. The same question was not examined for
`M11.C.50`, `M11.C.51` and `M11.C.54`, which fail on their only compared readout.

---

## 8. The table

| Item | Revision | Subject | Recommendation | Moves the hash? |
|---|---|---|---|---|
| R51-1 | 51 | wording written before the probes | Approve | No |
| R52-1 | 52 | T-196's probe in `M11.5` (`M11.C.48`, Phase B family) | Approve; reread at step 4 | No |
| R52-2 | 52 | T-197's probe in `M11.5` (`M11.C.2`, twelve-person family) | Approve | No |
| R53-1 | 53 | `M11.C.2`'s mutation cell and added sentence | Approve with a change | No |
| R53-2 | 53 | version 2 note on `M11.C.2` | Approve with the same change | No |
| R53-3 | 53 | version 2 notes on `M11.C.29`, `.42`, `.44`, `.45`; notes shown beside dispositions | Approve | No |
| R54-1 | 54 | `M11.5`'s note for `M11.C.42` | Approve | No |
| R55-1 | 55 | readout numbered `M11.G.5` | Approve | No |
| R55-2 | 55 | constants in `constants.md` | Approve | Reversing would |
| R55-3 | 55 | `M11.C.55`'s window in run years | Approve | Reversing would |
| R55-4 | 55 | `M11.C.52`'s mutant | No decision needed (the owner's pick; T-209) | — |
| R55-5 | 55 | criteria rows name tests in full | Approve | No |
| R55-6 | 55 | cue draw as a row of `M3.D.4b` | Approve | No |
| R55-7 | 55 | one commit, no open batch | Approve | No |
| R55-8 | 55 | relief predictor names no child on a tie | Approve | Reversing would |
| R55-9 | 55 | `cues.py`'s 2.0 as a structural literal | Approve | Reversing would |
| R55-10 | 55 | `cue_readings` outside canonical state | Approve | Reversing would |
| R55-11 | 55 | audit and sweep not rerun; three expected failures | No decision needed (superseded by Q7 (a) and step 7) | — |
| R55-12 | 55 | LLM experiment's expected failures | No decision needed (superseded by T-207) | — |
| R55-13 | 55 | tests named requirement first | Approve | No |
| R55-14 | 55 | words part not reattributed with the cues on | Approve as built; decide again with T-209 | Reversing would |
| F-1 | 55 | `M4.C.1e` and `M4.C.14` text under the wrong IDs | Fix | No |
| F-2 | 55 | `KS23.1` graded `[K]`, chapter is `[K-ext]` | Fix | No |
| F-3 | 52 | probe rows not committed | Optional | No |
| F-4 | 55 | `M11.C.53`'s too-little limb may be a premise limb | Probe, then decide the class | No |

**Counts over the 21 pending items:** approve 16 (one with a reread at step 4, one to decide again with T-209);
approve with a change 2; no decision needed 3; reverse 0. Approving as recommended moves the hash for no item; six
approvals would move it if reversed. **Over the 4 findings:** fix 2, optional 1, probe then decide 1; none moves the
hash.

## 9. What this review did not do

- It ran no model code and regenerated no record. Code was read for R55-10 only.
- It did not reread the owner-decided parts listed under "What is not in scope".
- It did not check `docs/model_explainer.md`, `docs/readouts_phase_c.md` or `CHANGELOG.md` against these revisions.
- It did not check T-138's trace (R53-3) or the spec's citations of `KS23.1` from before revision 55 (F-2).

## 10. The owner's decisions (2026-11-10)

Asked in this order, in one dialog, each with "Other" available.

| Item | Decision |
|---|---|
| R51-1, R52-1, R52-2, R53-3, R54-1, R55-1, R55-2, R55-3, R55-5, R55-6, R55-7, R55-8, R55-9, R55-10, R55-13 | **Approved** as they stand (15 items) |
| R53-1, R53-2 | **Approved with the change** recommended in R53-1: keep the restated mutation cell; replace "so the pass needs the witness path …" with the two mutants' numbers, which turned the pass red and which survived, and that the two are not separated from each other; move the revision 31 mutation sentence to `docs/spec_revision_log.md`. The version 2 note says the same |
| R55-14 | **Approved as built; decided again with T-209**, before any decision to turn `social_cues_enabled` on |
| R55-4, R55-11, R55-12 | No decision needed (the owner's own pick, or superseded), as §8 gives |
| F-1 | **Fix**: move `M4.C.1e`'s table and paragraph and `M4.C.14`'s bullets and closing paragraph back under their own rules |
| F-2, F-3, F-4 | Not selected; not acted on |

**Done at spec revision 58, 2026-11-11, after step 4 landed** (step 4 edited the same status files on its own branch,
`claude/nifty-allen-37cdf4`). All of it was wording only, and the engine hash did not move (`d3779684dd53`). No todo
record is settled by these decisions; T-209 stays open and now also carries R55-14:

1. Write R53-1's change into `M11.C.2`'s row in `docs/bowen_agent_model_spec_v2.md`, and R53-2's into its note in
   `docs/version_2_dispositions.md`, with that row's `decided` cell set to the owner, 2026-11-10; regenerate
   `docs/version_2_status.md`.
2. Fix F-1 in the spec, then regenerate `docs/spec_index.jsonl` and `docs/spec_rules/`.
3. Change "pending the owner's review" to the owner's approval of 2026-11-10 in the entries for revisions 51–55 in
   `docs/spec_revision_log.md`, in the spec's status line, in `CLAUDE.md` and in `docs/REVIEWER_READING_LIST.md`.
4. Add a changelog entry, and close any todo record these decisions settle.

