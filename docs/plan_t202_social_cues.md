# Plan: build the social cues proposal (T-202)

**Status: APPROVED by the owner 2026-11-07, with D1–D6 each as recommended (§7); not built.** Step 2's draft,
`docs/spec_revision_55_draft.md`, was approved by the owner 2026-11-08; the build started. No requirement, code, config or record
changed, and the engine hash (`0b363cb3186c`) did not move. The proposal is `docs/proposal_social_cues.md`, decided by
the owner 2026-11-07 (Q1–Q11, each as recommended). This plan covers T-202's four steps
(`docs/todo/2026-11-07_social_cues.md`): (1) the ledger search for `SC-12`, done here (§2); (2) spec revision 55 as a
draft for approval; (3) `SC-2` and `SC-3` behind a switch that is off by default; (4) the rest, in one engine-hash
batch with one regeneration of the records. Switching the cues on by default is a later owner decision and is not in
this plan.

The plan stops for the owner three times: after this plan (§7, D1–D6), after the spec revision draft, and at the end
with the composite criteria's results.

---

## 1. What reading the code changed

Five facts from the code that the proposal did not have. Each changes a criterion or adds a decision.

1. **`life_energy` is not built.** `Person.life_energy_ratio` exists and is always unset; `M6.I.3` has no code and
   `invariants.py` says "life energy is not built". `iposition.py` already uses a contact proxy for the same reason.
   `SC-9` ("attention is bounded by the relationship share of `life_energy`") has nothing to bound it with. **D3.**
2. **The words "channel" and "attention" are taken.** `Event.channel` is `M4.D.1a`'s selection channel, and
   `event_kinds.md` has a `channel` column for it. `Event.attention` is `M4.C.8`'s focus (feeling or intellect). The
   new field is `Event.signals`, the four carriers are *signal channels*, and the cue is `Cue.ATTENTION` in a new
   config file, `config/bowen/social_cues.md`, never a column of `event_kinds.md`.
3. **The golden fingerprints hash the whole log** (`tests/bowen/test_engine_golden.py`). With the switch off, the log
   must be byte-identical, so `Event.signals` is `None` when the switch is off and is written to the log only when it
   is on. `SC-3`'s check ("an Event without the field is refused") therefore applies with the switch on.
4. **Draws are counter-based and keyed** (`draws.py`, `M3.D.4a`). The perception draws `SC-6` and `SC-7` need go in a
   new draw class, `cue_perception`, made only when the switch is on. No other stream moves, so the off path draws
   nothing new.
5. **Two hash-keyed records are expensive.** Besides the ensemble and mutation records (about 4 h across both Macs at
   revision 47), the emergence audit record (`docs/emergence_audit_record.md`, 6–17 h) and the version 2 sweep record
   (about 11 h) both check the engine hash. Any edit under `src/bowen/` or `config/bowen/` makes all of them stale,
   even with the switch off. **D4.**

---

## 2. Step 1 done: the ledger search for age and dependency (Q7)

A search of the six sources (subagent, then the three entries below read here by line range):

| Entry | Grade | What it says, paraphrased | Bears on |
|---|---|---|---|
| `KS09.6` (`kerr_book/ks09.md` 95–107) | `[K]` `[T]` | separation is gradual and runs both ways; developmental stages are "not a useful way" to describe it; **the parents' toning down their togetherness urges as the child matures drives the child's separating**, and failure to do so prolongs dependency | the direction (dependency falls as the child matures) and the **mechanism**, which is on the parents' side, not an age term on the child |
| `K14.5` (`kb/kb14.md` 46–50) | `[K]` interview `[M]` | the infant–caretaker symbiosis is a normal state | the direction only |
| `KS09.8` (`kerr_book/ks09.md` 117–123) | `[K]` `[T]` | a child fused with a dysfunctional mother is tuned to her moods and feels responsible for her happiness | **against** "a young child's capacity to supply is small" |
| `FE11.4`, `FE10.14` | `[B]` / `[K]` | which child carries the load follows the parents' immaturity and sibling position | the load falls by projection and position, which the model already makes emerge |
| `FE09.11`, `FE07.22`, `L10.11a` | `[K]`, `[B]` | dependence ends with financial dependence; an adopted child's level is weighted by time in each family | already carried by `M7.C.1g` and `M1.A.15`; not about attention |

**What the corpus supports:** the direction that a young child is more dependent and that this falls as it matures
(`KS09.6`, `K14.5`), with the cause on the parents' side.

**What it does not support:** any magnitude or age curve; any rule that a young child's capacity to supply is small
(`KS09.8` runs against it); an age term on the child itself (`KS09.6` rejects stages and puts the cause in the
parents' separating).

**What the model already does with age:** `M7.C.1f` — no one selects a move before 10, so a young child already
supplies nothing by a move; `M7.B.1a` (chronic anxiety fixed at 12), `M7.E.1f` (basic level develops from 5 to 20),
`M7.C.1g` (leaving home at 20).

So `SC-12`'s capacity half is already carried by `M7.C.1f`, and its weight half, as the proposal writes it (an age
term on the child), runs against `KS09.6`'s causal order. **D5.**

---

## 3. Acceptance criteria, the test for each, and the mutant that must turn it red

Unit premises and checks are proved red by a new gate file, `tools/mutation_gate_social_cues.json`, run by
`tools/mutation_gate.py` into `docs/review/social_cues_unit_mutation_gate.md`. Composites are new ensemble criteria
`M11.C.50`–`M11.C.55` in a new module `src/bowen/ensemble/criteria_sc.py`, with their settings in
`config/bowen/criteria.md` and named mutants in `tools/mutant_runner.py`. **Both arms of every composite run with the
switch on** (a per-criterion override), so the criteria test the cues while every other record runs with them off.

| ID | Test | Mutant that must make it red | Class |
|---|---|---|---|
| **SC-1** (`M4.C.1e`) | `tests/bowen/test_social_cues.py::test_sc1_m4c1e_disapproval_raises_inward_impingement` — one delivery, the perceived approval cue −1 against 0, perception error off: the receiver's felt impingement change higher in the first | approval's impingement weight set to 0 | premise |
| | `::test_sc1_m4c1e_cues_only_move_contact_and_impingement` — the reading returns a (Δcontact, Δimpingement) pair and writes no new state field the policy reads | a cue written to a new `Person` field the policy reads | check |
| **SC-2** (`M4.C.1f`) | `tests/bowen/test_engine_golden.py::test_b1_seeded_run_log_is_unchanged`, its fingerprints **not edited** — the switch off against the current engine | the switch ignored (the cue path always on) | check |
| | `test_social_cues.py::test_sc2_m4c1f_reduction_with_equal_channels` — switch on, four channels equal to the words, no leak, no perception error: every move kind's (Δcontact, Δimpingement) equals the switch-off value | the word-cue baseline not subtracted (see D1) | check |
| **SC-3** (`M1.F.1`) | `test_social_cues.py::test_sc3_m1f1_move_without_signals_is_refused_when_on`; `::test_sc3_m1f1_signals_absent_and_unlogged_when_off` | the refusal removed | check |
| SC-3, Q9 clause (`M1.F.4`, `M4.C.9`) | `::test_sc3_q9_private_hop_carries_words_only`; `::test_sc3_q9_witness_reads_all_four` | the hop keeps tone; the witness reads words only | check |
| **SC-4**(i) (`M4.C.13`) | `::test_sc4_m4c13_anxious_sender_leak_raises_receiver_impingement` — same move, same words, the sender's acute anxiety high against low | leak set to 0 | premise |
| **SC-4**(ii) | `M11.C.50` — `KS08.4`'s two arms over the following weeks: the receiver's voluntary contact toward the sender higher in the low-anxiety arm | leak set to 0 | composite |
| **SC-5** (`M4.C.11`) | `::test_sc5_m4c11_nonverbal_share_rises_with_receiver_anxiety` | the weight held constant | premise |
| **SC-6**(i) (`M4.C.12`) | `::test_sc6_m4c12_neutral_cue_read_as_threat_more_at_low_level` — every sender neutral, 2000 keyed draws per arm | bias set to 0 | premise |
| | `::test_sc6_m4c12_presence_is_never_imagined` — no signal sent on a channel: perceived presence 0 at every level | error allowed onto presence | check |
| **SC-6**(ii) | `M11.C.51` — **changed from the proposal's arms, D2:** same level, the receiver's starting belief that the sender disapproves set against neutral; over the run, the receiver's share of `DISTANCE` and `CONFLICT` on that tie higher in the first | bias set to 0 | composite |
| **SC-7** (`M4.C.14`, `M4.C.1a`) | `::test_sc7_m4c14_no_term_reaches_zero_at_the_top_level` | — (a check against the config's floors) | check |
| | `::test_sc7_m4c14_lower_level_detects_a_small_leak_more_often` | sensitivity not scaled by level | premise |
| | `::test_sc7_m4c14_lower_level_reads_threat_into_neutral_more_often` | meaning error not scaled by level | premise |
| **SC-8** (`M4.C.6`) | `M11.C.52` — two anxious agents on one tie; one arm gives one of them perspective: the tie's escalation lower in that arm | attribution reads the word channel instead of the nonverbal ones | composite; **risk, D2** |
| **SC-9** (`M1.A.10`, `M6.I.3`; a proxy, D3) | `M11.C.53` — a work stressor on one spouse against none: the other spouse's too-little deviation on the tie and its `PURSUE` or `CONFLICT` rate higher | attention sent not reduced by the sender's load | composite |
| **SC-10** (`M9.9`) | `M11.C.54` — a parent's starting belief about one child set false-negative against neutral, the child's own state equal: that child's later `functional_level` lower (gated); its share of supplying moves toward the parent reported, not gated (D2) | the belief not sent as leak | composite |
| **SC-11** (`M1.A.14b`, `M11.G`) | `tests/test_repo_layout.py::test_sc11_no_role_field` — no config key or `Person` field names a role; the role words live in `config/lint/role_labels.md` (outside `config/bowen/`, so editing them does not move the hash) | a `caretaker` field added | check |
| | `M11.C.55` — across seeds on the twelve-person family, the child who becomes the main supplier to a parent is predicted better by that child's learned relief from supplying than by birth order | supply weighted by birth order | composite |
| **SC-12** | **not built in T-202** (D5, decided 2026-11-07); follow-up T-204 | | |

Every criterion test name carries its `SC` ID and its spec ID. The new readout for `SC-11`, `src/bowen/readouts/roles.py`,
is computed after the run and never read by the policy; a test checks that no module under `src/bowen/engine/` imports
it (the pattern of the existing readouts).

---

## 4. Order of work, what moves the engine hash, and what it costs

| Step | Work | Moves the engine hash | Where | Time |
|---|---|---|---|---|
| 0 | This plan; the owner approves it and D1–D6 | no | here | — |
| 1 | The ledger search (§2) | no | done | — |
| 2 | Spec revision 55 as `docs/spec_revision_55_draft.md`: `SC-1`–`SC-11` (and `SC-12` per D5), each with grade, source and `M11.5` class; the parameter register rows; the owner approves it | no (a draft file) | here | about 2 h |
| 3 | Reopen `tests/open_batch.py` (`OPEN`, `OPENED_AT` = the commit before, `AT_OPEN` = the records' hashes then). `SC-3` and `SC-2`: `Event.signals`, `config/bowen/social_cues.md` with `social_cues_enabled: false`, the reduction; tests written first and run red; golden fingerprints unchanged | **yes** | here | about 3 h |
| 4a | `SC-1`, `SC-4`, `SC-6`, `SC-7` (the reading, the leak, the perception error, the level terms); unit tests first; the gate file's mutants run | yes, again | here | about 5 h |
| 4b | `SC-5`, `SC-8`, `SC-9`, `SC-10`, `SC-11` (and `SC-12` per D5); `criteria_sc.py`, the named mutants | yes, again | here | about 6 h |
| 4c | **Probe before regenerating:** each composite and its mutant on 40 seeds, this Mac. A composite that fails, or whose mutant survives, goes back to the owner before step 6 (the `SC-8` risk, D2) | no | here, 8 workers | about 1 h |
| 5 | Documents (§6); the spec revision applied from the approved draft | yes (config prose) | here | about 3 h |
| 6 | **One regeneration** of the hash-keyed records (D4) | — | Mac Mini1 + here, or here alone | see below |
| 7 | `tools/record_changes.py` (`docs/version_3_revision_55_changes.md`), the generated indexes, `tools/c16_report.py`, the status page; close the open batch | no | here | minutes |
| 8 | One `learning-qa` pass and one correctness pass over the batch's diff; fixes; the suite; commit to `main`, push | no | here | about 1 h |

Steps 3, 4a, 4b and 5 each move the engine hash; step 6 regenerates once, on the final hash. Between, the records are
stale by decision, and the open batch marks the tests that read them (the machinery stage 3 used, D31). The golden
fingerprints must hold at every commit, because the switch is off in all of them.

**The regeneration (step 6).** With the switch off, every existing row must come out identical apart from its hash
line; the regeneration doubles as `SC-2`'s reduction check at ensemble scale, and `tools/record_changes.py` must show
no moved row. Times from revision 47's run:

- **Mac Mini1 reachable** (`ssh macmini1`, 10 workers; it timed out again at this plan's check, 2026-11-07): Phase D
  part II and the Phase D mutation record there; the Phase C records, Phase D part I, level occupancy, the group G
  "after" section, the trace and the fam-diag export here. About 4 h wall, plus the new `M11.C.50`–`M11.C.55` rows and
  their mutants, estimated 1–2 h and measured at step 4c before the run starts.
- **This Mac alone** (`nice -n 15`, 8 workers): about 6 h for the existing records plus 1–2 h for the new rows,
  **about 7–8 h**, run in the background, split in two sittings if needed.
- **The emergence audit and the sweep** are not in either figure (D4).

---

## 5. Hard rules this plan keeps

- No assigned role and nothing read from a role label (`M1.A.14b`): `SC-11`'s lint and readout.
- Every new constant is `[I]`, lives in `config/bowen/social_cues.md` or `constants.md`, and has a register row in the
  spec whose grade matches (the constants parser refuses a mismatch).
- Kerr is `[K]` or `[K-ext]`, never Bowen. Only `ch18` is `[B]`.
- Every criterion is a direction between two arms. `M11.C.54`'s "different" supply share is reported, not gated,
  because "different" is not a direction (D2).
- The suite: `python3 -m pytest tests/`, no xdist; the exit line read and the count stated.

---

## 6. Documents the batch updates

The spec (revision 55, its status line), `docs/spec_revision_log.md`, `docs/model_explainer.md` (a new section for the
cues and signal channels, with grades), `docs/readouts_phase_c.md` (the role readout), `config/bowen/constants_changes.md`,
`config/bowen/criteria.md`, every regenerated record, `docs/version_3_revision_55_changes.md` (new),
`docs/SPEC_REVIEW_CHECKLIST.md`, `docs/REVIEWER_READING_LIST.md`, `docs/theory/_STATUS.md`, `CHANGELOG.md` (within its
size limit), `CLAUDE.md` (the suite count and the spec status), the proposal's status line, T-202's record, `TODO.md`
(regenerated), and this plan's status line.

---

## 7. The owner's decisions

**Decided 2026-11-07, through `AskUserQuestion`: D1–D6 each as recommended, and the plan approved.** D5's follow-up
is T-204 (`docs/todo/2026-11-07_social_cues.md`). The text below is the decision as put.

- **D1 — how a cue reading reduces to the current components.** Recommended: **the deviation form**. Each move kind
  keeps its contact and impingement in `event_kinds.md`; the felt change is those components plus the cue weights
  times (the perceived cue − the word cue). With every channel equal to the words and no error, the difference is zero,
  so `SC-2` holds by construction and `event_kinds.md` stays the one source of each kind's effect. The other form,
  a full four-cue vector per kind chosen so the weights reproduce its two components, adds 64 invented numbers (16 move kinds, four cues each)
  and makes `SC-2` a fit to be checked kind by kind. The proposal's `SC-2` mutant ("one cue weight perturbed") cannot
  turn the deviation form red, so its mutant becomes "the word-cue baseline not subtracted".
- **D2 — four criteria changed from the proposal.** Recommended, all four:
  (a) `SC-6`(ii) compares a receiver's belief, not two levels: level also moves steepness, band and gain, so a
  low-against-high-level arm would pass with the bias zeroed (the lesson of T-194 and T-195);
  (b) `SC-8` is probed at step 4c before the regeneration, because `M4.C.6` already attributes part of every received
  event, and the criterion may pass on that rule alone with its named mutant surviving; if it does, the criterion
  comes back to the owner;
  (c) `M11.C.54` gates only on the child's `functional_level` and reports the supply share;
  (d) `M11.C.55` compares two predictors of the main supplier across seeds on the twelve-person family.
- **D3 — `SC-9` while `life_energy` is not built.** Recommended: **a proxy** — the attention cue a person sends on its
  word channel falls with its current exogenous stressor load, graded `[I]`, in the way `iposition.py` already uses a
  contact proxy, and stated in the spec as a proxy until `M6.I.3` is built. Other choices: defer `SC-9` until
  `life_energy` is built; or build `life_energy` first (a premise change of its own, outside T-202).
- **D4 — the expensive records.** Recommended: regenerate the ensemble, mutation, diagnostic and other records once
  (step 6), and mark the emergence audit's and the sweep's record-current tests `xfail(strict=True)` with a reason
  naming this batch, to be rerun at the stage end or when the owner decides the switch's default (that decision moves
  every record anyway). The suite would then have 4 strict expected failures. With the switch off their rows cannot
  change, which the step 6 regeneration shows on every other record. The other choice is to rerun both now: about
  6–17 h and about 11 h more machine time.
- **D5 — `SC-12`.** Recommended: **do not build it in T-202.** Its capacity half is already `M7.C.1f` (no moves before
  10); its weight half, as an age term on the child, runs against `KS09.6`, which puts the cause in the parents'
  toning down their togetherness. Record a follow-up proposal for `KS09.6`'s parent-side mechanism instead. Other
  choices: build `SC-12` at `[I]` as written; or build a parent-side form now at `[I]` (a parent's leak and word
  cues toward a child fall as the child matures).
- **D6 — Mac Mini1.** It timed out again. Recommended: start step 6 on Mini1 if it answers then, and otherwise run
  on this Mac alone (about 7–8 h), not wait for it.

---

## 8. Adjacent issues found, not fixed

- `M1.A.10` and `M6.I.3` are in the spec with no code; `invariants.py` checks only that the unset value is in range.
  The spec index shows `M1.A.10` with code and no test, which reads as built.
- `KS09.8`'s inverted projection (a child overfunctioning toward a dysfunctional parent) is in the ledger as a model
  consequence ("`M7.E` currently runs parent→child only"); no open item covers it (T-203 cites `KS09.8` only for its
  intermittent withdrawal).
