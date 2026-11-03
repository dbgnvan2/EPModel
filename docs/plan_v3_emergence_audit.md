# Plan: the emergence audit (version 3, part 1; T-182)

**Status: approved by the owner as written, 2026-11-03, with §9's Q1–Q7 each as recommended (through
`AskUserQuestion`). Written 2026-11-03 on `main` at `5c7fc9b` (spec revision 47, engine hash `0b363cb3186c`,
Phase D `020160abb6db`). Nothing is built.** No edit under `src/bowen/` or `config/bowen/` is planned: every ablation
is a mutant applied to a temporary copy of the repository, as the mutation records already do
(`tools/mutant_runner.py`). §9 lists what the owner decided.

## 1. Why

The project's core rule is that patterns emerge and are not written (`CLAUDE.md`; `docs/model_explainer.md` §19):
conflict, distance, cutoff, projection and fixed triangles must come from agents that learn from felt relief, and a rule
that writes one of these outcomes is a defect. Today this is checked one criterion at a time, by the mutants a
criterion's spec row names (`M11.1d`; `tools/mutation_record.py`, `tools/phase_d_mutation_record.py`; W3 on
`docs/version_2_status.md`). A named mutant removes the mechanism the row names. It does not ask whether the result
needs learning at all.

The agents act inside a fixed act list and hand-set terms: the capacity gate on newer acts (`M4.D.3a`), the mixing
weight between the two channels (`M4.D.1a`), focus and perspective steering (`M4.D.1g`, `M4.D.1a`), the loaded-tie
gate (`M4.D.3b`), the dependence gate (`M5.C.1`) and the legal set's structural preconditions (`M4.D.1e`). Any of
these can carry an outcome without anyone intending it. T-180 is a live case: `M11.C.48[phase_d]` passes at revision
47, and its mutant `cutoff-blind-to-learned-value`, which makes `CUTOFF` ignore its learned value, survives. So the run
of negative contacts before a cutoff does not come from what was learned about `CUTOFF`.

The audit asks the question for the whole model at once. For every criterion that passes, it reruns the criterion with
learning removed and reports whether the result survives. For each result that survives, it finds which non-learning
term carries it.

A survivor is not a defect by itself. `M11.5` defines a composite as one that needs several mechanisms acting
together, "usually through learning or through time", and some composites are carried by time alone (for example
`M11.C.27`, from `M4.C.1`'s two-sided shape, and `M11.C.28`, through `M4.C.3`'s integrator). The audit's output is
evidence for triage. The defect test is `M11.1d`'s: does a rule in the carrying set name the criterion's readout or
outcome as its own target? The owner decides that for each survivor (§9, Q5).

## 2. What "learning removed" means: the ablations

Learning from felt relief has four paths in the engine, all in `src/bowen/engine/learner.py`. Every value they write
is a learned value of the automatic channel (`Person.learned_values`). Two places read those values: the policy's
automatic weights (`src/bowen/policy/policy.py`, `channel_weights`) and `M5.F.4`'s abort draw
(`src/bowen/engine/iposition.py`, `_abort_act`). Values start empty, and a missing value reads as 0.

| ID | Ablation | Edit (in the copy) | Removes | Leaves in place |
|---|---|---|---|---|
| **L0** | all learning off | `learning_rate` 0.2 → 0.0 and `witnessed_cutoff_gain` 0.05 → 0.0 (`config/bowen/constants.md`) | every path below. No learned value moves from 0, so every automatic act starts and stays at the same value, and the abort draw is uniform. The soothing review cuts a share of a positive value, so it changes nothing. | availability (`M4.D.3a`), the mixing weight (`M4.D.1a`), focus and perspective steering, the loaded-tie and dependence gates, the legal set, the self-directed channel's position score (`M5.F.5`), competing urges (`M4.D.1d`), and every state dynamic outside the policy: appraisal, transfer, stocks, beliefs, the estimator, `systems_perspective`'s rehearsal |
| **L1** | own relief not credited (`M4.D.6a`) | in `learner.py`, `felt = relief.get(pid, 0.0) + params.cross_person_weight * …` → `felt = params.cross_person_weight * …` | the actor's own felt change in anxiety from the signal | the cross-person term, witnessed-cutoff learning, the soothing review |
| **L2** | cross-person term off (`M4.D.6e`) | `cross_person_weight` 0.5 → 0.0 | the target's and witnesses' change in anxiety from the signal: the projection loop's internalisation path | own relief, witnessed-cutoff learning, the soothing review |
| **L3** | witnessed-cutoff learning off (`M4.D.6f`) | `witnessed_cutoff_gain` 0.05 → 0.0 | a witness's learning from what followed a cutoff | own relief, the cross-person term, the soothing review |
| **L4** | soothing review off (`M4.D.6`) | `soothing_review_penalty` 0.5 → 0.0 | a person's learning that soothing did not help | own relief, the cross-person term, witnessed-cutoff learning |

L0 is the audit's main question, and the same overrides as `LEARNING_OFF` (`src/bowen/ensemble/criteria_d.py`), which
`M11.C.39` and `M11.C.40` use as their baseline. L1–L4 say which learning path a result needs, when L0 turns it red.
L0 is the same edit as the existing mutants `learning-off-in-both-arms` and `learner-off-share-kept` (Phase D mutation
record). The audit keeps its own cache, so it runs them again; on the same engine its rows for `M11.C.39` and
`M11.C.34` must agree with those mutants' rows, and the build checks that.

Considered and not used: holding the values at 0 only where the policy reads them, with the learner still running.
Nothing else in the engine reads the values, so this differs from L0 only in `_abort_act`'s draw, and L0 is a config
edit that cannot break.

Not learning, and so not ablated: `M1.F.5`'s witness appraisal (`witness_weight` moves a witness's anxiety, which is
appraisal), beliefs (`M9.6`), the rehearsal of `systems_perspective` (`M1.A.9`), the accommodation stock (`M4.D.5a`),
lock-in (`M7.D.2a`), habituation (`M4.G.3`, which only scales the learning signal and so is removed with it under L0).

## 3. Which criteria, and the verdicts

**In scope:** every row that passes in `docs/phase_c_ensemble_record.md`, `docs/phase_d_ensemble_record.md` or
`docs/phase_d2_ensemble_record.md` and whose class is **composite** or **mixed** (`M11.5`). The tool reads the rows
and classes from the records' JSON blocks at build time; no criterion list is written into it.

- **Premises are out.** A premise passing confirms that the code renders the spec and is never a finding (`M11.5`,
  `M10.C.5`). **Checks are out**: they verify an instrument, a conservation law or the architecture.
- **A mixed criterion is judged on its composite limbs only.** Its premise limbs (`tools/version_1_status.py`,
  `premise_limbs()`) are run and shown, but they do not decide the verdict, as on the status page.
- **Exclusions are stated, never silent.** A row whose two arms differ only in learning cannot pass with learning
  removed, by construction: `M11.C.39` (learning on against `LEARNING_OFF`). It is not a finding. It is used as the
  positive control (below). Each exclusion and its reason is a row in `config/audit/emergence_audit.md`, and the record
  prints "N of M passing rows audited; K excluded" with each excluded row and why.

On revision 47's records (`5c7fc9b`) the in-scope rows are 13, with `M11.C.39` as the positive control: `M11.C.16`,
`M11.C.27[unstable,remove_one]`, `M11.C.41[light: lower level]`, `M11.C.41[heavy: lower level]` (Phase C);
`M11.C.22`, `M11.C.11`, `M11.C.34`, `M11.C.7[joint over one spouse]`, `M11.C.7[joint over sided]` (part I);
`M11.C.15`, `M11.C.10[family of origin]`, `M11.C.2[twelve-person family]` and `M11.C.48[phase_d]` (part II; it
passes at revision 47, with W3 unmet on it, T-180). The set is whatever the regenerated records say when the
audit is built.

**Verdicts**, per row and ablation, the same as `tools/mutation_record.py`'s `judge`, so they read the same way across
records:

- **survives**: the criterion still passes with the path removed. The result does not need that learning.
- **red**: it stops passing (FAIL or UNDETERMINED at the seed cap). The result needs that learning.
- **reversed**: red, and every gating readout's interval lies wholly on the side opposite its direction. Removing the
  learning flips the result rather than removing it.

Two outcomes are not verdicts and are rerun next time, never cached: **broken** (the edit's text does not apply: no
evidence) and **not run** (the child failed or timed out). An engine exception under an ablation is a red, as in the
mutation records.

**Controls.** The build refuses to render the record if a control fails, because then the ablation is not doing what
it says:

- *positive control:* `M11.C.39` under L0 must be red. Its treatment arm differs from its baseline only in learning,
  so if it survives, L0 did not remove learning.
- *negative control:* `M11.C.3` (a premise, `M1.C.1`'s transfer, the physics of an act) under L0 must survive. If it
  goes red, L0 changed something beyond learning.

The controls are shown in their own section and are never findings.

**Seeds.** Each run uses the records' adaptive rules unchanged (`M17.A.1`: blocks of 50, cap 500, the same precision),
so a verdict under an ablation compares directly with the verdict in the ensemble record (§9, Q4).

## 4. Tracing a survivor to what carries it

For each row that survives L0, the tool runs a second set of joint mutants: L0 plus one **carrier candidate**. A
candidate whose removal turns the survivor red is a **carrier**. The record names each carrier, or says "no carrier
among N candidates".

The candidates are the policy's non-learning terms, and the criterion's own named mutants:

| ID | Candidate | Edit (with L0, in the copy) | Rule |
|---|---|---|---|
| C1 | every capacity layer available | `availability` returns 1.0 | `M4.D.3a` |
| C2 | the mixing weight blind to level, in the policy only | as the mutant `mixing-weight-blind-to-level-in-policy-only` | `M4.D.1a` |
| C3 | focus steering off | `focus_steer_gain` 1.0 → 0.0 | `M4.D.1g` |
| C4 | perspective steering off | `perspective_steer_gain` 0.5 → 0.0 | `M4.D.1a` |
| C5 | loaded-tie gate off | `_loaded_gate` returns 1.0 | `M4.D.3b` |
| C6 | competing urges off | `competing_urge_gain` 0.5 → 0.0 | `M4.D.1d` |
| C7 | the criterion's own mutants | every mutant that `tools/mutation_record.py` or `tools/phase_d_mutation_record.py` names for the row, joined with L0 | the row's `M11.5` entry |

The dependence gate (`M5.C.1`) is covered by C7 where a row names it (`dependence-gate-removed`). The legal set's
structural preconditions (a tie to a living person; a closed triad for `TRIANGLE`) define what an act is and are not
mutated; the record lists them as not tested.

For each survivor the record also shows, from the row's existing `M11.1d` listing in `M11.5`, the rules the spec says
the result rests on, beside the carriers found, so the owner can see where the two disagree.

T-180 is the first row this answers: `M11.C.48[phase_d]` under L0, and if it survives, C1–C7 against it.

## 5. Where it lives, and the engine hash

| What | Path |
|---|---|
| tool | `tools/emergence_audit.py` — `ABLATIONS`, `CARRIERS` (as `tools/mutant_runner.Mutant`), `in_scope()`, `build()`, `render()` |
| exclusions and controls | `config/audit/emergence_audit.md` — rows with reasons, outside `config/bowen/` as `config/sweep/sweep.md` is |
| cache | `docs/records_cache/emergence_audit.json` (`tools/record_cache.py`'s `Cache`, `result_key`) |
| record | `docs/emergence_audit_record.md`, ending in a JSON block read by `tools/record_json.py` |
| test | `tests/test_emergence_audit.py` |

The ablation and carrier texts live in the tool beside the other mutants, as the mutation records keep theirs; they are
code edits, not editorial content. The exclusion reasons are editorial and go in the config file (global rule 9).

Phase C rows run through Phase C's child (`tools/mutant_runner.py`), Phase D rows through
`tools/phase_d_mutant_child.py`, chosen by which registry holds the key (`CRITERIA`, `CRITERIA_D`, `CRITERIA_D2`).

**The engine hash.** Each result is keyed by the hash the Phase D mutation record uses (`code_hash` of `src/bowen` and
`config/bowen`, both children, and the source of `apply` and `run_in_copy`) and by the mutant's edits. Nothing new
enters the engine hash: the audit adds no code that changes what a run computes. It only applies edits through
`apply`, which is already in the hash. The tool, the config file and the record stay out, so a change to a label, the
renderer or an exclusion reason reruns nothing (learnings P38). When the engine hash moves, the record goes stale like
every other record, and an open batch can carry its current-test under `stale_in_open_batch` (`tests/open_batch.py`).

The audit is not wired into `docs/version_2_status.md` or the stage conditions yet (§9, Q6).

## 6. Acceptance criteria and their tests

IDs are `EA-n`. If the owner adds the spec requirement (§9, Q1), it is `M11.1f`, and the tests are named
`test_m111f_ea<n>_…`. "Proved by" is the mutation that must turn each test red before it counts. The build commit
records each mutation run. Tests EA-3 to EA-8 compare two arms (with and without an ablation) and assert a direction.
The rest are tool checks, like the existing record tests, and are flagged as such.

| ID | Criterion | Test | Proved by |
|---|---|---|---|
| EA-1 | Scope is the passing composite and mixed rows of the three records, no more | `test_m111f_ea1_scope_is_passing_composite_and_mixed` (fixture records with a passing premise, check, failing composite, passing composite and passing mixed row) | drop the class filter: the premise and check enter |
| EA-2 | Every ablation and carrier applies exactly once to the current tree (`M11.1a`) | `test_m111f_ea2_every_ablation_applies_exactly_once` | change one `old` string by one character |
| EA-3 | L0 removes learning: on a short run (Phase B family, seed 0, 40 ticks), the count of nonzero learned values is lower under L0 than unmodified | `test_m111f_ea3_l0_leaves_fewer_learned_values` | an L0 whose edit is a comment |
| EA-4 | L1 removes own relief: with only the actor's relief nonzero, an act's learned value moves less under L1 than unmodified | `test_m111f_ea4_l1_credits_less_own_relief` | an L1 that edits the cross-person term instead |
| EA-5 | L2 removes the cross-person term: with only the target's relief nonzero, the value moves less under L2 than unmodified | `test_m111f_ea5_l2_credits_less_cross_person_relief` | swap L1's and L2's edits |
| EA-6 | L3 removes witnessed-cutoff learning: fewer `learned_value[CUTOFF…]` effect rows from a witnessed cutoff under L3 than unmodified, same seed | `test_m111f_ea6_l3_fewer_witnessed_cutoff_updates` | L3 edits `witness_window` instead |
| EA-7 | L4 removes the soothing review: fewer `soothing_review` effect records under L4 than unmodified, same seed | `test_m111f_ea7_l4_fewer_soothing_reviews` | L4 edits `soothing_review_every` to its current value |
| EA-8 | The controls: the record's positive control (`M11.C.39`, L0) is red and its negative control (`M11.C.3`, L0) survives; `build` raises if either is not | `test_m111f_ea8_controls_hold_in_the_record`, `test_m111f_ea8_build_refuses_a_failed_control` (fixture cache) | an L0 that changes nothing: the positive control survives |
| EA-9 | Verdicts are `judge`'s: survives, red, reversed; broken and not run are never cached | `test_m111f_ea9_verdicts_follow_judge` (synthetic rows) | invert the reversed test's sign |
| EA-10 | A mixed row's verdict reads its composite limbs only | `test_m111f_ea10_premise_limb_does_not_decide_a_mixed_row` (fixture: premise limb flips, composite holds → survives) | gate on every limb |
| EA-11 | Carriers run only for L0 survivors, and the record names each one that turns the survivor red, or "no carrier among N" | `test_m111f_ea11_carriers_named_only_for_survivors` (fixture cache) | run carriers for every row |
| EA-12 | Every in-scope row has a verdict under every ablation, or a stated non-verdict; the record's "N of M audited, K excluded" matches | `test_m111f_ea12_no_row_dropped_silently` | skip excluded rows without printing them |
| EA-13 | The cache key is the engine hash and the edits; a label change reruns nothing and an edit change does | `test_m111f_ea13_key_is_engine_and_edits` | put `what` into the key |
| EA-14 | The committed record equals the render from the cache, and a missing result raises `KeyError` | `test_m111f_ea14_emergence_audit_record_is_current` | change one byte of the record |
| EA-15 | The record never reports a premise or check as audited, and prints each row's class (`M11.5`) | `test_m111f_ea15_class_shown_premise_never_audited` | include a premise row |
| EA-16 | The plan, the record and the config file are listed in `docs/SPEC_REVIEW_CHECKLIST.md` | `tests/test_doc_checklist.py` (existing) | remove the record's line |

Not testable as code, flagged for human review:

- **EA-H1**, the triage of each survivor, the `M11.1d` question (does a carrying rule name the outcome): the owner,
  item by item, in this plan's T-item.
- **EA-H2**, `docs/REVIEWER_READING_LIST.md`'s row is accurate: read at the build's review.

Integration only, not covered by CI: the full audit run. CI checks the record against the committed cache (EA-14).
It does not rerun it.

## 7. Order of work and cost

**Order.** (1) Tool, config file and tests EA-1, -2, -9 to -16 on fixtures; each test's mutation run. (2) EA-3 to
EA-7, short runs in a copy. (3) The pilot: L0 on `M11.C.48[phase_d]` (T-180), on the two controls, and on two short
Phase C rows. It is timed, and the per-row cost below is corrected from it before the full run is started. (4) L0 on
every in-scope row. (5) L1–L4 on every in-scope row. (6) Carriers on L0's survivors. (7) Render, review, documents,
commit, push. Steps (4)–(6) wait for T-181's full sweep (about 11 h across both Macs), so that both runs use
the same engine hash and the two Macs are not loaded at once.

**Cost, projected from the records and the cache.** The cache stores seeds per result, not seconds, so the projection
is in seed-ticks: seeds × the longest arm's ticks (`measure_run_length`, cached in
`docs/records_cache/sweep_v2.json`). It is calibrated on a regeneration with timings: Phase D part II, 1.81 million
seed-ticks (counted on today's record) in 95 min on Mac Mini1 (`CHANGELOG.md`, stage 3 step 10, 2026-10-26), about
19,000 seed-ticks a minute.

| Row | Ticks | Seeds at pass | Seed-ticks |
|---|---|---|---|
| `M11.C.10[family of origin]` | 2080 | 200 | 416,000 |
| `M11.C.2[twelve-person family]` | 2080 | 150 | 312,000 |
| `M11.C.48[phase_d]` | 2080 | 200 | 416,000 |
| `M11.C.7`, two pairs | 520 | 100, 150 | 130,000 |
| the other 9 rows, and the two controls | 21–112 | 50–150 | about 100,000 |
| **one ablation, at the passing seed counts** | | | **about 1.37 million: about 72 min on one Mac** |

A row that an ablation turns red often runs to the cap of 500 seeds, so a red row costs up to 2.5 to 5 times its
passing cost. One ablation therefore costs between about 70 min and about 3.5 h of one Mac's time. The three 2080-tick
rows are 83% of it.

| Step | One-Mac time | Split |
|---|---|---|
| pilot (3) | about 30–60 min | this Mac, `nice -n 15`, 8 workers |
| L0 (4) | 1.2–3.5 h | the three 2080-tick rows on Mac Mini1 (`ssh macmini1`, `~/ProjectsMini1/EPModel/.venv/bin/python`, 10 workers); the rest here |
| L1–L4 (5) | 4.7–14 h | the same split; about 2.5–7 h of wall time on each Mac |
| carriers (6) | about 7 × each survivor's cost; with three survivors of mean 100,000 seed-ticks, about 2 h | split by row |

The split and the merge follow `docs/two_mac_sweep.md` (both Macs on the same pushed commit; Mini1's cache copied back
with `tools/merge_cache.py`; nothing committed from Mini1). Kill orphan workers after each share. The cost is stated
again from the pilot's timings before step (4) starts, and the owner approves it then.

Tokens: one `learning-qa` pass and one correctness pass on the build diff, about 250,000 tokens together, from the
last batches' figures.

## 8. Documents

This commit: this plan; `docs/SPEC_REVIEW_CHECKLIST.md` (the plan listed); `docs/REVIEWER_READING_LIST.md` §7 (one
row); the T-item in `docs/todo/2026-11-03_version_3_emergence_audit.md`; `TODO.md` regenerated.

The build commit:

- **the spec**, if Q1 is yes: `M11.1f` added beside `M11.1d`, as version 3's first revision (revision 48, wording in
  `M11` only; the spec is not in the engine hash). `docs/spec_revision_log.md`, `docs/spec_index.jsonl` and
  `docs/spec_rules/M11.md` regenerated (`tools/spec_index.py`, `tools/spec_rules_view.py`).
- `docs/model_explainer.md` §18 and §19: one paragraph on what the audit asks and how to read a survivor.
- `docs/SPEC_REVIEW_CHECKLIST.md`: the record and `config/audit/emergence_audit.md`.
- `docs/REVIEWER_READING_LIST.md` §5: the record, beside the mutation records.
- `CHANGELOG.md`, `docs/theory/_STATUS.md`'s STATE line, this plan's status line.
- `docs/spec_coverage.md` regenerated, if it reads the new test names.

`config/bowen/constants_changes.md` is not touched: no constant changes.

## 9. What the owner decides

**Decided 2026-11-03:** the owner answered Q1–Q7 through `AskUserQuestion`, each as recommended, and approved the
plan as written. `M11.1f` goes into the spec at the build, as version 3's first revision.

Each is written for `AskUserQuestion`, with the recommendation first. "Other" is always open.

**Q1. Should the audit become a spec requirement?**
- *Yes, as `M11.1f`, version 3's first revision (recommended).* The audit is a check on every criterion, as `M11.1d`
  is, and the spec is the contract the records are reviewed against. Wording only; the engine hash does not move.
- *No, a tool only.* Faster, but its rule is then stated nowhere the reviews read.

**Q2. Which ablations?**
- *L0 and all four paths, L1–L4 (recommended).* L0 answers whether a result needs learning; L1–L4 say which path it
  needs. About 6–17 h of machine time in all.
- *L0 only.* About a fifth of the cost. A red row then says only that it needs some learning.
- *L0, L2 and L3.* The two paths the spec ties to named results (`M11.C.2`'s internalisation, `M11.C.10`'s witnessed
  cutoff).

**Q3. How is a survivor traced?**
- *Joint mutants: L0 plus each policy term C1–C6 and the row's own mutants C7 (recommended).* Evidence by removal,
  the same kind as the mutation records.
- *The `M11.5` listing only, no runs.* Free, but it is the spec's reading, not a measurement.
- *Joint mutants and a per-act probe of each survivor* (`tools/probes.py`: which act's share carries the difference).
  More to read; useful if C1–C7 find no carrier.

**Q4. Seeds under an ablation?**
- *The records' adaptive rules, cap 500 (recommended).* Verdicts compare directly with the ensemble records. A red
  row costs more.
- *A fixed count, each row's passing seed count.* Cheaper and fixed in cost; a row can read UNDETERMINED where more
  seeds would decide it.

**Q5. What happens to a survivor?**
- *It is a finding for triage, one T-item per survivor, decided by `M11.1d`'s test (does a carrying rule name the
  outcome?); no rule changes in the audit's batch (recommended).*
- *A survivor with a carrier among C1–C6 is reclassified as a premise in `M11.5`'s table, in the same batch.*
  Quicker, but it changes classes before anyone has read the carriers.

**Q6. Does the audit join the stage conditions?**
- *Not yet. A separate record, and a column on `docs/version_2_status.md` after the first triage (recommended).*
- *Now, as a new condition W5: every passing composite has an L0 verdict.*

**Q7. When does it run?**
- *The pilot (§7 step 3) once the tool is built, the rest after T-181's full sweep, on the same engine hash, split
  across both Macs (recommended).* The pilot is small, answers T-180 sooner and corrects the cost before the full run.
- *All of it after T-181.* One machine schedule, and T-180 waits.

## 10. Adjacent issues found, not fixed

- `tools/phase_d_mutation_record.py` and `tools/mutation_record.py` each keep their `MUTANTS` and their own scope
  reading. The audit reads both lists for C7. A third reader of the same lists suggests one registry of mutants by
  criterion; not part of this plan.
- The records' caches store seeds but no run time, so every cost estimate (this one, the sweep's) is projected from
  ticks. Storing each result's wall time in the cache would make projections measurements; it would not change a key.
