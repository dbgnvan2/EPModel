# Implementation plan — Phase D

> **Status: draft for owner approval, 2026-10-10.** No Phase D code until this plan is approved (planning rules).
> Spec: `docs/bowen_agent_model_spec_v2.md`, revision 13. Phase C closed with its gate failing (owner decision,
> `docs/phase_c_completion_report.md` §13). Spec IDs are used verbatim. The inventories behind this plan were made on
> 2026-10-10 from the spec, `src/bowen`, `TODO.md` and the Phase C documents; their findings are cited where used.

## 0. Scope, and the one recommendation that shapes the rest

**What `M13` assigns to Phase D:**
- M7 the slow clock, including `M7.A.1a`, `M7.D.2c/2d` and `M7.E.4`;
- `M9` beliefs as a channel (`M9.6`, `M9.7`);
- the twelve-person reference family;
- the three symptom channels and endogenous events;
- `M11.G`'s readout schema.

**Its gate:**
- 28 criteria, each over an ensemble with its mutation run: `M11.C.2, .6, .7, .9, .10, .11, .12, .13, .14, .15, .17,
  .18, .20, .21, .22, .23, .24, .26, .28, .30, .31, .33, .34, .36, .37, .39, .40, .43`;
- `M11.D.20`;
- the `M16.T.3` re-run;
- a 40-year three-generation run that completes.

**What exists today.** Almost none of it.
- The slow tick only writes a log record (`engine/tick.py:182`).
- `M9.8`'s belief store exists and is read by the policy only.
- `M4.D.6e`'s cross-person learning term was built in Phase C (`learner.py:108`).
- Symptoms accumulate in one channel.
- `child_id` exists.
- The Phase D family loads and runs under today's engine, but no one is born, dies, launches or partners, and no
  `basic_level` moves.

**What the owner has added for Phase D (2026-10-10):**
- cutoff as rejection, not severing (`docs/DECISIONS — CUTOFF SEMANTICS.md`, option B);
- a pull toward reconnecting, since severed ties stay severed and that produced the long silences (Marta–Nadia
  102 of 104 weeks);
- projection as a model mechanism, not only a readout.

**Recommendation R0: split Phase D into two gated parts.** The spec makes Phase D one gate. But half its criteria need
births, deaths, partnering and a 40-year run, and the other half run on fixture families with the slow clock alone.
Building the life course first would hold every criterion back behind the hardest, most decision-dependent work.
- **D-I, foundations:**
  - engine speed;
  - cutoff and reconnection;
  - the slow clock and estimator;
  - symptom channels;
  - beliefs as a channel;
  - the `M11.G` schema;
  - the criteria that run on fixtures or on the Phase C family.
- **D-II, the life course:**
  - births, childhood, launch, partnering, death;
  - the twelve-person family over 40 years;
  - the generational criteria.

Each part gets a completion report. The `M13` gate stays one gate; it passes when both parts do. **Owner decision:**
approve R0, or keep one undivided Phase D.

## 1. Owner decisions before code

None of these can be settled from the corpus or the code. Each lists what it blocks and what I recommend.
Decisions marked **D-I** block the first part; **D-II** decisions can wait until D-I is built.

| # | Decision | Blocks | Recommendation |
|---|---|---|---|
| **P1** | **Cutoff as rejection (option B).** The tie stays interactive. A per-tie rejection intensity rises with each refusal and falls with sustained contact. While it holds, the cutter's appraisal of what the other sends is replaced by rejection. Needs a spec revision (`M1.B.3`, `M5.B.3`, `M6.I.7`). **D-I** | C.10, the silences, every record | Approve B, with the rejection intensity `[I]` and decaying with sustained contact |
| **P2** | **A pull toward reconnecting.** Today `REDUCE_CUTOFF` carries no learned value without systems perspective, so a severed tie stays severed (`TODO.md`, Bruno and the Phase C silences). Options: (a) the "too little" side of the deviation (`M4.C.1c`) keeps growing while a tie is cut, until approaching relieves it; (b) a separate pull term. **D-I** | P1's effect; Dario's tie to Ravi | (a): it uses an existing mechanism and writes no outcome |
| **P3** | **The `M5.F.2` threshold** (re-explained 2026-10-10): keep 0.5 on both axes, set other lines, or make the flip gradual. **D-I** | C.34; every I-POSITION result | Gradual (c), crossing at 0.5 `[I]` |
| **P4** | **The estimator** (`M1.A.4a`, `M1.A.4b`): its form, window and breadth count (`[I]`, with the sensitivity sweep `M1.A.4c` requires). Also when transmission hands over to it (review S2). **D-I** | C.6, C.17, C.18, C.20, C.33 | Load-weighted mean and variance of `functional_level` over a 3-slow-tick window, plus a breadth count of distinct nodal situations, all `[I]`; declared before any criterion reads it |
| **P5** | **Chronic anxiety: fixed or drifting.** `M7.B.1` fixes it in childhood; `M1.A.7a` and `M3.B.1` derive it every slow tick (review B). **D-I** | the slow-tick step | Fixed for everyone past the fixation age at `t0`; derived for people born during the run (`M2.A.0a` already says this) |
| **P6** | **`M7.E.4` against `M1.B.4`.** "Accrual on bond energy" against bond energy decaying at or near zero. **D-I** | tie deterioration | Conductance decays on the slow tick, faster at lower level; bond energy is left as `M1.B.4` has it |
| **P7** | **Undefined terms in criterion rows.** C.23's "EE term" and suppression term; C.28's "allocation-mobility term"; C.14's "technique"; C.13's "community". **D-I** for C.14, C.23, C.28 | those criteria | I draft a definition of each from its source for approval. C.13 moves out of Phase D until a community is modelled (it has no host) |
| **P8** | **Nulls with no `M11.4` exception.** C.13 and C.14 assert "no effect", but `M11.4`'s exception list is closed; C.17 needs a declared margin. **D-I** | C.14, C.17 | Add both to `M11.4` with an equivalence margin, minimum detectable effect and power declared (`M11.4a`) |
| **P9** | **C.7's direction.** "The topology arms MUST differ" names no direction (`M0.4`). **D-I** | C.7 | Declare it: the arm in which the coach is triangled into the marital pair shows less change in the pair than the arm in which he holds contact with each separately |
| **P10** | **C.9: keep or demote** (`M11.E`). Its stale `M11.E` row says its effect size is `[I]`; the restated three-arm form needs none. **D-I** | C.9 | Keep, as the three-arm mid-peak comparison |
| **P11** | **What C.42 and C.45 test, and C.16's mutation clause** (Phase C report §13). **D-I** | carry-over criteria | Owner question; Phase D does not rebuild them unless decided |
| **P12** | **A criterion that proves the triangle change** (report §13 item 5). **D-I** | the triangle mutants | I draft one reading the seeker's relief against the third's help |
| **P13** | **`M6.3`, what happens to a dead person's bound anxiety.** **D-II** | C.15, C.37, the 40-year run | Owner question. The review asks for it before Phase D is planned in full; it can wait for D-II |
| **P14** | **Births, partners and life stage.** Birth prior spread and the age the estimate is taken (`M7.E.1`'s named plan items); where generation 3's partners come from (the spec is silent); when a person starts and stops acting. **D-II** | C.6, C.10, the 40-year run | Partners enter as new people drawn at the pairing tolerance (`M2.A.0e`, `M2.A.0f`); children act from a declared age `[I]` |
| **P15** | **Family data.** Dr Halim's sex and values (spec says 55, "not stated"; config says 60, male); Teodor at 36; Marta–Iris "resolved, low contact" (config says ordinary); Halim tied to both spouses where `M2.B.2` says "whichever member is being coached". **D-II** | the twelve-person runs | Set the spec to the config (60, male, both ties), Marta–Iris resolved; owner confirms |
| **P16** | **Bowen's two falsifiers and three chapter items** (`L09.2`, `L16.2`, `L15.1`, `L03.4`, `L03.5`; revision 12 P10). **D-II** | — | Add as criteria in a later revision, not this plan |
| **P17** | **`t0` and the first slow tick.** `M2.A.0b` calls `t0` the first slow tick; today it fires at the end of week 51. **D-I** | the slow-tick step | Week 0 is a slow tick |

## 2. Build decisions (D-I)

### B1. Engine speed first
- **The problem.** A 2,080-week run of the twelve-person family is estimated at 105–110 s, and the cost per tick grows
  with run length (measured 2026-10-10: 3.0 s for the first 208 weeks, 9.8 s for the fifth). At the current ensemble
  settings that is about 3 hours per 50-seed block.
- **The cause.** Nine mechanisms scan, and some sort, the whole event history every tick: `consolidate.py:34`,
  `reactive.py:72`, `observe.py:110`, `event_store.py:53`, `recompute.py:76`, `appraise.py:142`,
  `outside_ness.py:86`, `sinks.py:58`, `learner.py:67`.
- **The fix.** Give the event store windowed indexes: by tick, by tie and by recipient. Each mechanism then reads only
  its window, and the cost per tick becomes flat.
- **Test.** A golden test: the same seed gives byte-identical final state and log before and after (`M11.D.16`'s method).
  A timing check stays out of the suite.

### B2. The batched engine items, in the first engine change
These were all marked "next engine change" and were missed by two regenerations on 2026-10-10:
- the trace renderer's Phase B framing (`render/trace.py:43`);
- `code_text`'s two hash gaps (`tools/ensemble_record.py`);
- a `now` argument for the pattern readouts;
- the readouts' act sets moved to config (rule 9);
- the shared C.45 predicate.

Each gets the test its TODO entry names.

### B3. Channel stamping
`act.py:128` stamps every policy act `AUTOMATIC`, including `I-POSITION` and `STAY-IN-CONTACT`. `M1.A.4h` (the
estimator's primary signal) and `M1.A.11c` both read the channel. Stamp the channel the policy selected. A test that a
self-directed selection is logged self-directed.

### B4. Cutoff as rejection and the reconnection pull (P1, P2)
- Per the decision memo. New tie field: `rejection`, owned by the cutter's side.
- `apply_delivered_cutoffs` no longer sets `interactive = False`.
- Appraisal reads rejection.
- `REDUCE_CUTOFF` lowers rejection.
- The pattern readouts' cutoff definition is unchanged; it already reads refusals.
- **Tests:**
  - a cut tie stays interactive;
  - repeated refusals raise rejection;
  - sustained contact lowers it;
  - a silent tie reads as distance, not cutoff;
  - the seed-7 silences shrink (a readout, not an assertion);
  - P2's pull: a cut tie's "too little" deviation rises with time, and an approach relieves it.

### B5. The slow-tick step
- A `slow_tick(state, params, emitter)` phase runs after the fast tick on every slow tick (P17). It has its own
  invariant check, since `M6.I.5` forbids `basic_level` moving inside a fast tick. Any stock it changes is logged in
  `M6.4`'s ledger.
- It carries:
  - the estimator (B6);
  - chronic-anxiety drift for people born in the run (P5);
  - `M7.E.4`'s conductance decay (P6);
  - `dyad_age`;
  - the life-stage update hook, which stays empty until D-II.

### B6. The estimator (`M1.A.4a`–`4c`, `M7.A.1a`; P4)
- A per-person history of `functional_level` and the load that tick imposed, kept in state.
- The estimator reads that history on the slow tick and writes `basic_level`.
- Financially dependent people get no rise (`M7.A.1a`).
- The window and breadth are config, with the sensitivity sweep.
- **Tests:**
  - a benign window does not raise `basic_level`;
  - a nodal-loaded, stable window does;
  - a dependent twin does not.

### B7. Symptom channels (`M7.D.2`, `2a`–`2d`, `M7.D.4`, `M1.A.11c`)
- Load can move between the three channels.
- Lock-in lowers family anxiety until a severity peak, then raises it.
- Load in one channel lowers onset hazard in the others.
- Removal fires off the others' tolerance.
- Every shape is `[I]` and declared.

### B8. Beliefs as a channel (`M9.6`, `M9.7`, `M9.1`)
- **Attribution store:** each person's view of where the family's difficulty lies, written from witnessed events.
- **Appraisal reads it:** appraisal reads the receiver's belief about the sender and the situation, not the true
  state.
- **Projection gate:** the projection clause of `M4.B.2a` and `M5.C.1`'s "system calm" row read through belief.
- **Ties created mid-run** get beliefs (needed in D-II).

### B9. Projection in the model (the owner's D4)
`M4.D.6e`'s term exists. What is missing:
- a cross-person term that weighs the *parent's* calming specifically, where today it is the mean over all witnesses;
- a declared test that the projection target's repertoire shapes to the parent's relief (C.2's mechanism);
- the mutant C.2's `M11.5` note implies, with `M4.D.6e` removed, run alongside the witness-path mutant.

### B10. The `M11.G` readout schema
- Components 1–8 of the family evaluation; 9 and 10 are forbidden (`M11.G.1`).
- Adaptiveness is derived against stress.
- The `M11.F` framing travels with every readout.
- Belief and truth are reported separately.
- Built in `src/bowen/readouts/`, outside the engine (`M11.D.22`).

### B11. Who acts
- Activation reads a declared life stage instead of selecting every living person (`activation.py:32`).
- Dr Halim's session schedule moves from `policy_source.py:55` into the family or the script.
- Needed in D-I so that fixture families can hold a member inactive.

## 3. The D-I gate

Criteria that need only the slow clock, fixtures or the Phase C family. Each runs over an ensemble with its mutation
run, an `M11.1d` audit (deletion plus sign-inverted mutant) and the fallback-rate flag (`M11.D.18`).

| Criterion | Family | Test | Mutation that must turn it red | Blocked by |
|---|---|---|---|---|
| C.2 | Phase C family (Nadia) | `test_m11c2_symptom_concentrates_on_projection_target` | the witness path (`M1.F.5`); also `M4.D.6e` removed | B9, B7 |
| C.7 | Phase C family with Dr Halim | `test_m11c7_topology_not_coach_skill` | `M8.2`/`M8.3` predicate made position-blind | P9, `M8` predicates |
| C.9 | sibling fixture | `test_m11c9_position_effect_peaks_midscale` | `M1.A.14c`'s gating made additive | P10; propensity reading sibling position |
| C.11 | nuclear four | `test_m11c11_removal_three_phase_shape` | `M6.I.6` | B7; conservation limb under the `M11.4` exception; the curve shape goes to human review (`M11.E`) |
| C.12 | dyad | `test_m11c12_symptom_relief_raises_conflict` | released bound anxiety discarded (`M6.I.1`) | B7 |
| C.14 | nuclear four | `test_m11c14_technique_null_under_marital_distance` | `M5.C.1`'s marital-distance gate removed | P7, P8; the gate |
| C.17 | Phase C family with Dr Halim | `test_m11c17_no_basic_level_drift_without_coach` | `systems_perspective` allowed to rise spontaneously | B6, P8, `M7.E.4` |
| C.18 | any | `test_m11c18_estimator_separates_cutoff_from_iposition` | `M1.A.4b`'s load weighting flattened | B6, B3 |
| C.20 | any | `test_m11c20_estimator_separates_borrowed_from_basic` | as C.18 | B6 |
| C.21 | nuclear four | `test_m11c21_reciprocity_inverts_on_projector_collapse` | `M6.I.4`'s transfer made one-way | — |
| C.22 | nuclear four | `test_m11c22_symptom_lock_in_is_non_monotone` | `M7.D.2a` flattened and `M7.D.2d` removed | B7 |
| C.23 | nuclear four | `test_m11c23_expressed_emotion_dominates_suppression` | the EE term zeroed | P7 |
| C.24 | dyad | `test_m11c24_whisper_of_nature_reverses_on_dissolution` | the `M6.I.4` transfer zeroed | — |
| C.26 | nuclear four | `test_m11c26_sink_recoverable_from_belief_configuration` | `M9.6`'s attribution write randomised | B8 |
| C.28 | nuclear four | `test_m11c28_sink_mobility_is_protective` | the mobility term pinned | P7 |
| C.30 | any | `test_m11c30_channels_are_mutually_protective` | `M7.D.2c`'s cross-channel term zeroed | B7 |
| C.31 | calm family | `test_m11c31_binder_removal_releases_bound_anxiety` | the held anxiety discarded (`M1.F.9`) | — (mechanism built) |
| C.33 | dyad | `test_m11c33_dependence_gate_blocks_basic_level_rise` | `M7.A.1a`'s gate removed | B6 |
| C.34 | any | `test_m11c34_system_reaction_to_symptom_discriminates_level` | `M1.A.19`'s detectors made level-independent | B3, P3 |
| C.36 | triad | `test_m11c36_policy_reads_belief_not_true_state` | `M4.B.2` reads the true state | B8 |
| C.39 | triad | `test_m11c39_pattern_outlasts_spell_only_with_learning` | `M4.D.6` disabled in both arms | a test-declared override (`M10.B.5`) |
| C.40 | triad | `test_m11c40_leaked_reinforcement_raises_self_directed_entropy` | reinforcement reaching the self-directed channel | B3 |
| C.43 | triad | `test_m11c43_learned_buffer_removal_costs_more` | `M4.C.10`'s downward term removed | C.43's inputs reconciled with `M4.D.6d` and `M1.F.9` |
| `M11.D.20` | — | `test_m11d20_readout_ordering_matches_run_log_ordering` | — | B10 |

**Moved out of Phase D:** C.13. It has no community to host it (P7). It returns when one is modelled.

## 4. The D-II gate (planned in detail after D-I)
- **Criteria:** C.6, C.10, C.15, C.37, and C.2 on the full twelve-person family.
- **Re-runs:** the `M16.T.3` re-run with the slow clock, Dr Halim and the belief channel active.
- **The 40-year run:** 2,080 weeks and 40 slow ticks, carrying everything `M3.B.1` lists.
- **Blocked by:** P13, P14, P15; the life-course mechanisms (`M7.C.1`–`1e`, `M7.E.1`–`1e`); several `Family` objects
  (today only `ravi_marta` has a budget and sinks, `family.py:238`); and topology that can change (`state.py:55`).

## 5. Build order (D-I)

1. **The owner decides** P1–P12 and P17 (one session; I bring drafts for P4, P7, P9 and P12).
2. **Spec revision 14**, applying those decisions.
3. **B1 engine speed**, golden-tested, and **B2 batched items**. The first engine change; one regeneration.
4. **B3 channel stamping.**
5. **B4 cutoff and reconnection.** All records change meaning. Regenerate; report what moved. The Phase C verdicts are
   re-read under the new engine as information, not as a gate.
6. **B5 slow tick, B6 estimator.** Then C.18, C.20, C.33, C.17.
7. **B7 channels.** Then C.11, C.12, C.22, C.30, C.31.
8. **B8 beliefs.** Then C.26, C.36, C.14.
9. **B9 projection.** Then C.2.
10. **B11 activation, the `M8` predicates.** Then C.7, C.9, C.21, C.24, C.23, C.28, C.34, C.39, C.40, C.43.
11. **B10 `M11.G`.** Then `M11.D.20`.
12. **Ensemble, mutation and sweep records.** One review pass and one correctness pass per batch (`CLAUDE.md`).
    Then the D-I completion report.

Each step: tests first, then code, then docs.

## 6. Compute budget
- **Phase C as a baseline.** A full regeneration took about 45 minutes.
- **Risk.** Phase D adds 23 criteria at D-I, each with a mutant and an `M11.1d` audit. Without B1 the
  twelve-person runs of D-II would cost about 30 hours per criterion at the ensemble cap.
- **Requirement.** B1 is a prerequisite for D-II, and worth doing first for D-I.
- **Cache.** Records stay cached by engine hash (`tools/record_cache.py`). Only changed mutants and settings rerun.

## 7. What cannot be made code-testable in Phase D (`M11.E`)
| Item | Proposal |
|---|---|
| `M5.F.2`'s threshold (P3) | the sign flip across it is tested; the threshold is the owner's |
| C.11's curve shape | the directions are tested; the curve is reviewed against `_LEDGER.md` L08.4 by the owner |
| C.9 | kept as the three-arm comparison if P10 approves; otherwise a readout |
| The Phase C trace review (plan §9) | still the owner's; open |
| `M11.C.8` | Phase E (unchanged) |

## 8. Adjacent issues found, not fixed
- **Spec: a blank line splits the criteria table** (line 1241), so rows C.26–C.45 sit outside the table under its
  header. Fix in revision 14 and check `M11.D.11`'s scan.
- **Spec: C.40's test name describes the mutant, not the claim.** Rename in revision 14.
- **Spec: C.10's "learning from witnessed cutoff" has no mechanism.** `M4.D.6` reinforces only the actor's own
  moves. Decide in revision 14 whether C.10 runs through `M1.D.8` alone.
- **Spec: C.6's shape limit (a), "within about ten generations",** cannot be observed in a three-generation run.
  Recorded as a readout in D-II.
- **Spec: `M4.D.6e` is said to "stay Phase D"** (`M5.A.1a`'s revision 13 note) but was built in Phase C. B9 is what
  remains; the note will say so.
- **Tools, from `TODO.md`:**
  - record caches trusted, not proved;
  - the occupancy record checked by hash only;
  - `record_cache.py`'s error matching;
  - duplicated record regexes;
  - `run_in_copy`'s child failure;
  - probe issues.
  None blocks Phase D. Handled when the record tools are next touched.
- **Paper-material candidates** held since 2026-09-27 (`TODO.md`): not in this plan.

## 9. How completion will be reported
A status report per criterion (done / partial / not done, with the test and record that prove it), as for Phase C. A
regenerated `docs/spec_coverage.md`. The completion report says which Phase C findings changed under the new engine.
