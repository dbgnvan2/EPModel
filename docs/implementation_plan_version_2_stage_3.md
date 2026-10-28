# Version 2 stage 3: implementation plan

**Status: APPROVED by the owner 2026-10-24 ("Approved: the stage 3 plan as written"). Step 4 done 2026-10-25: spec revision 39 (part 1, group C and T-130's wording; part 2, groups F and A; part 3, T-128 and T-014) is in `docs/spec_revision_39_draft.md`, written from the owner's answers the same day (Q1–Q14, each as recommended; Q14 refines D26: the ratio enters a criterion whose run is longer than one slow tick), and the owner approved it as a whole the same day; each group's approved text lands in the spec with its code (Q6). Step 3, 2026-10-25: P2 was drafted (`docs/PROPOSAL — DISPLACED COPING (P2).md`, decided by the owner 2026-10-25, Q1–Q4 as recommended) and probed; it does not move `M11.C.6` at coping size, and the owner chose P2 plus F1, the floor-held functioning on a ledger (D18, third reading). P2 and F1 were probed together on Mac Mini1 the same day: the difference falls from +0.089 to +0.018 ± 0.025, unresolved, and no variant shows the decline `M11.C.6` expects. Step 2 done 2026-10-24: the trace met D18's condition and the owner chose P2 (D18, revisited); the probe did not meet D20's, so D20 stands. Step 1 done 2026-10-24: T-130's thirteen values all agree with the paper's Table 3, and two item names differ from its wording; the edits wait for step 6 (`docs/review/stage_3_t130_event_sizes.md`). Step 5 done 2026-10-25: the sixteen stage 3 tests for steps 6–10 are written and each is shown failing on its missing rule; fifteen are strict expected failures in the default run and `M11.C.49`'s is `ensemble`-marked (the step 5 row). Step 6 done 2026-10-25: the batch's first commit, §3.6's edits (T-130's wording, T-014's revision 39 part 3 text, T-129's ruff F fixes with `SOURCE_ALLOWED` emptied, and T-127's wording); S3.20 and S3.22 green, S3.23 holds (the golden fingerprints pass unchanged); the engine hash moved 66dcbb1e8ba5 → f90285984f32. The records keyed by the hash stay stale until step 10 (D31), and the eleven tests that read them are marked by `tests/open_batch.py` as expected failures of the open batch, which `tests/test_open_batch.py` keeps strict on both ends; MR-23 (T-012) moves to step 10, before the one regeneration (the step 6 row). Step 7 done 2026-10-26, the batch's second commit: group C built (P2 as `M1.A.5f`, F1's `floor_held`, `M6.I.4a`); S3.1a–S3.1d and S3.4 green; the engine hash moved f90285984f32 → b8e9ad1333e2 and the golden fingerprints were taken again; the targeted run (`review/stage_3_step_7_criteria.md`): `M11.C.6` fails, +0.041 ± 0.027, reported as read (Q4); `M11.C.2` passes on the Phase B family, fails on the twelve-person family; `M11.C.21` fails, +0.056 ± 0.11 (the step 7 row). Step 8 done 2026-10-26, the batch's third commit: group F built (`M5.F.4`'s demand share with the threshold retired, `M5.F.2c`, `M11.C.29`'s arm, `M11.C.49`); S3.6 and S3.8 green, S3.7's test written and red (`M11.C.49` fails, T-137); the engine hash moved b8e9ad1333e2 → b819660f58b9 and the golden fingerprints were taken again; the targeted run (`review/stage_3_step_8_criteria.md`): `M11.C.49` fails (family excess +2.2 ± 3.0, `CONFLICT` share +0.003 ± 0.028), `M11.C.29` fails, its genuine arm completing in 1 of 100 seeds, `M11.C.32`, `M11.C.18`, `M11.C.20` and `M11.C.33` pass and each of their mutants turns them red, `M11.C.5` still fails (the step 8 row). Step 9 done 2026-10-26, the batch's fourth commit: group A built (P1 as `M4.C.1d`, P3 as `M4.D.3a`'s `TRIANGLE` at layer 0, `reciprocal_contact_share` 0.5); S3.10 and S3.11 green; the engine hash moved b819660f58b9 → 2a3b84e2d89a and the golden fingerprints were taken again; the targeted run (`review/stage_3_step_9_criteria.md`): `M11.C.42` fails (`triangle_reuse` −0.06 ± 0.36) and `M11.C.45` fails (`triangle_rate` +0.0013 ± 0.0055), so S3.12 and S3.13 are not met; `M11.C.44` fails (−0.020 ± 0.089), where revision 38's record passes (S3.14, T-142); the `M11.C.27` cells, `M11.C.16` and `M11.C.41`'s cells keep their verdicts (the step 9 row). Step 10 done 2026-10-26, the batch's last commit: T-128 built (`M17.E.1`'s named ratio `tie_wear_to_transfer_ratio` in the sweep tool and its configuration, drawn for a criterion whose measured run is longer than 52 fast ticks); S3.16–S3.19 green; MR-23 (T-012) built; the engine hash did not move (2a3b84e2d89a: the sweep tool is outside it); every record regenerated once on it and the open batch closed (D31); the changes page (`version_2_revision_39_changes.md`) and the status page (S3.24, revision 38's verdict beside each): 6 of the status page's 63 rows differ from revision 38's: `M11.C.2[twelve-person family]`, `M11.C.7[one spouse over sided]`, `M11.C.10[family of origin]` and `M11.C.17` FAIL → PASS, `M11.C.44` PASS → FAIL, and `M11.C.49` new and failing; five Phase C mutation verdicts and one Phase D changed, and `M11.C.17`'s new pass has a surviving mutant (T-143) (the step 10 row). Step 11 done 2026-10-27: the stage-end full sweep with the ratio, split across both Macs; the ratio entered 26 of the 34 keys; W4 met; T-128 closed; `M17.E.1` stays *partial* until `M11.C.49` is swept (T-144, the owner 2026-10-27) (the step 11 row). Next: step 12.** The owner decided D18–D31 on 2026-10-24, each through
`AskUserQuestion`, one at a time, and each answer was the recommended option (§3.2). No engine change is built.
Written from `docs/version_2_plan.md` §3 "Stage 3 — Rule questions the owner chooses", the TODO items T-014, T-128,
T-129 and T-130, and the stage 2 milestone review's triage (`docs/todo/2026-10-24_stage_2_milestone_review.md`).
Every code fact in §3.1 was checked against `main` at `7fa32b5` (spec revision 38, engine hash `66dcbb1e8ba5`).
Stages 1 and 2 are in `docs/implementation_plan_version_2.md`; this stage has its own file so that one stays
readable. Section numbers continue that file's: this is §3.

The version 2 plan's rule holds: each stage that changes a run goes proposal → spec revision → implementation plan →
build, each approved before the next. This plan carries the three proposals (§3.3) and the plan together, so the
owner approves both at once. Spec revision 39 is then drafted from them and approved before any engine code (§3.8,
step 4).

## 3.0 What this stage covers

1. One proposal for each group the owner chose on 2026-10-19 (Q2): **C**, projection; **F**, `I-POSITION` at a low
   level; **A**, triangle relief. Each starts from the owner's note in `docs/version_2_plan.md`.
2. **T-128:** `M17.E.1`'s named ratio, for the owner's decision before stage 3's full sweep.
3. **T-014:** the four stage 3 questions (the avoidance gate, fusion, the coach and `SPLIT`, `M7.D.1a`'s illness
   size).
4. **T-129:** the `src/bowen` docstring and lint edits that wait for the first change that moves the engine hash, with
   T-127 and T-012's MR-23, which wait on the same trigger.
5. Every criterion by ID with its test and mutation, the order of work, the compute, and what no test can check.

Groups B, D and E stay accepted findings (Q2). `M15`, calibration and Phase F stay out of version 2.

## 3.1 What the code shows (checked 2026-10-24)

Three read-only agents gathered the group facts (about 440k subagent tokens, about 6 minutes in parallel). The two
claims each recommendation rests on were re-checked by hand.

### Group C — projection

- **The development rule** (`M7.E.1f`, `src/bowen/engine/life_course.py:369-422`). From age 5 to 20 the child's basic
  level moves each year by `development_rate × d` (0.1, `[I]`), where `d` is the mean of two terms: the child's own
  year-mean `functional_level − basic_level`, and the living parents' mean functional level minus their mean basic
  level (`:407-410`, checked by hand). The band `a(1 − 0.02 − 0.08p)` to `a(1 + 0.02 + 0.03s)` only limits the step
  (`:414-420`); the projection share `p` lowers the floor and nothing else.
- **`M11.C.6` does not fail because of the floor.** The band bound 0 times in 180 child-runs
  (`docs/implementation_plan_phase_d.md:1970-1972`), and widening `development_band_projection` to 0.19 in the stage 2
  sweep changed nothing. Under projection the child ends **higher**: +0.0886 ± 0.044 at revision 38, 0 of 8 samples
  held, every sample positive (+0.043 to +0.118, `docs/version_2_sweep_record.md`).
- **Why it ends higher.** *(Corrected 2026-10-24 by the trace, §3.3.1: the step reads the estimator's functional
  level, net of pseudo-self, so neither term moves with the exchange; the lift is in the child's own term.)* An
  `OVERFUNCTION` moves functional level and pseudo-self from the child to the parent,
  conserved (`moves.py:313-336`). The child's own term falls by the amount; the parents' term rises by the amount
  divided by the number of living parents. So the step credits the child with part of what the parent took, and the
  relief projection gives the parent lifts the parents' term further. `pseudo_self` is one number per person, positive
  when borrowed and negative when lent (`objects.py:164`); it does not record from whom.
- **`M11.C.21`** reads the child's rise after both scripted over-functioning parents collapse. The child has lent down
  to its own floor (`basic − stress_floor_below_basic`) in both arms, and nothing returns lent functioning while the
  exchange continues: `relax_pseudo_self` acts only after `exchange_window` weeks with no exchange (`moves.py:367`),
  and the collapse lowers the parents without reversing the flow. Revision 38: −0.0682 ± 0.089, 1 of 8 held.
- **The projection dose is small:** about 3.5 more projection-type acts a year in the treatment arm
  (`docs/phase_d2_ensemble_record.md:15`).
- **Corpus.** FE07.4 `[K][T][M]` and KS04.7 `[T][M]`: the child goes along because it gets a calmer parent, and comes
  to look to the parent to solve problems, which keeps the parent over-functioning. FE06.3 `[K][T]`: the parents give
  up self to the child but less than the child does; the child's functioning is the most impaired. KS00.9 `[T]`: the
  parents' levels set the family climate. `model_explainer.md` §19 places projection in the emergent layer. No entry
  gives a per-act magnitude.

### Group F — `I-POSITION` at a low level

- **Level enters indirectly, in four places; there is no level gate.** Anger reads felt impingement above a band that
  narrows with functional level (`contact.py:47-52`); hold capacity is `hold_gain × functional_level × efficacy`
  (`iposition.py:129-134`); every move delivers `assault_gain × outward` extra impingement, and the outward axis starts
  at `0.8 × (1 − basic_level/100)` (`appraise.py:190`, `outside_ness.py:16-17`); the self-directed channel's weight is
  `(FL/100)^2 + 0.5 × perspective × (1 − that)` (`policy.py:183-193`, `M4.D.1a`). The genuine/assertion choice
  (`iposition.py:120-126`) is binary: perspective below 0.3 or anger.
- **A weak form of the owner's direction already exists:** at initial values a level-40 mover's genuine act carries
  about 0.34 impingement and a level-60 mover's about 0.26 (agent's arithmetic from the constants; the outward axis
  drifts over a run).
- **Why `M11.C.29` completes 0 of 100.** The arm scripts the act at week 4 (`config/bowen/criteria.md`, `M11.C.29`
  `t0` 4) and the spell is a `JOB_LOSS` of 120 to both parents every 4 weeks from week 4 (checked by hand). The act
  falls on a spell week and every 4-week PREPARE window holds another: 58 seeds failed PREPARE on excess 20.7 ± 0.85
  against `defence_threshold` 15, and 37 ran as the assertion form because the mover was angry
  (`docs/version_1_dispositions.md:31`). The unit tests already raised their mover to level 60 for the same reason
  (`tests/bowen/test_iposition.py:376-401`), and `M11.C.32` uses level 60.
- **Corpus.** KB12 interview (K12.3) `[T]`: a low-level self can hold an I-position but it is selfish, dogmatic and
  forceful; the manner shows the level. KS05.5 `[K][T]`: less differentiated people tend to take reactive I-positions
  that polarize the system and raise anxiety. KS06.11 `[K][T]` and FE04.13 `[K][T]`: many so-called I-positions are
  attempts to get others to change. `[B]` support (L10.12, FE11.19) covers dogmatism, not a gradient by level. None is
  `[K-ext]`; none gives a magnitude.

### Group A — triangle relief

- **Why a `TRIANGLE` relieves less.** `DISTANCE` binds `min(excess, 0.3 · s · excess)` into the tie's stock
  (`moves.py:90-121`). `TRIANGLE` moves `min(excess, 0.3 · capacity · s · third_help · excess)` onto the third's acute
  anxiety (`moves.py:204-237`): the same base rate times three factors below 1 (`capacity` about 0.6 at level 40,
  `third_help` about 0.5 or less, `1 − intensity_floor`). Measured at Phase C close: −1.29 against −4.97 one week after.
- **Contact.** A `TRIANGLE`'s contact (+0.3) lands only on the receiver's felt contact; the sender gets only the
  `speaker_echo` cost (`appraise.py:290-301`). The learner credits a `TRIANGLE` by the seeker's own change in acute
  anxiety alone; the recruited third is excluded from the cross-person term (`learner.py:91-94`, checked by hand).
- **Availability.** `TRIANGLE` sits in layer 2 of `M4.D.3a`'s ordering (`config/bowen/event_kinds.md:57`), an `[I]`
  placement the config says has no source; availability falls with functional level, so a spell makes triangles
  less available.
- **The sweep shows no dose-response.** `M11.C.42` samples with `triangle_transfer_rate` from 0.17 to 0.42 range from
  −0.155 to +0.185 with no trend; raising that constant alone is not a fix.
- **`Triangle.bound_anxiety` is never written** (`objects.py:299`), though `M1.C.1` says a triangle must hold it.
  Recorded in §3.13; the owner chose not to build it in this stage (D22).
- **Corpus.** L10.3 `[B]` and L17.1 `[B]`: anxiety dilutes over nodes; a tense twosome recruits a third. KS03.1
  `[K][T][M]`: the insiders are calmed by the alliance. FE06.2 `[K][T]`: at moderate anxiety the outsider is actively
  recruited. KS15.10 `[K][T]` and FE06.20 `[K][T]`: triangles are older than language and three people invariably
  gravitate to two insiders and an outsider. No source compares a triangle's relief with distance's.

### T-128, T-014, T-129

- **T-128.** `M17.E.1`'s sentence comes from `papers/DESIGN_LESSONS_model_design_papers_2026-09-17.md` §2.4: in
  Castellano et al.'s coevolving networks, the probability of rewiring a tie against changing state is a control
  parameter with a transition. In this model, tie attenuation is `tie_wear_rate` (`M7.E.4`, conductance lost per slow
  tick, `slow_tick.py:204`) and anxiety moving between people is `calm_transfer_rate` (`M4.C.10`, the share of the gap
  a calmer sender closes per contact). No criterion's sweep set holds `tie_wear_rate` today; `calm_transfer_rate` is in
  `M11.C.43`'s only (`config/sweep/sweep.md:319`).
- **T-014.** `avoidance_available` is defined and tested in `live_positions.py` and called nowhere (`M8`'s preamble).
  No caller passes a fusion map to `positions_live`, so `M8.1`'s fused-occupant branches never run. A `SPLIT` toward
  the coach always sides it (`external.coach_side`, `M8.3`). Every symptom onset takes the `OWN_ILLNESS` and
  `FAMILY_ILLNESS` sizes on all three channels (`M7.D.1a`).
- **T-129.** The edits are listed in the triage record: `moves.py:63-68`'s module docstring (MR-2); the stale test name
  at `rejection.py:62` (MR-9); `objects.py:247-248`, `:277`, `rejection.py:10` and `invariants.py:26` (MR-13); ruff F
  items in `src/bowen` and `tools/level_occupancy.py:139`, then a ruff test (MR-26). `ruff` is in `requirements.txt`.

## 3.2 Decisions (the owner, 2026-10-24)

Each was put through `AskUserQuestion`, one at a time, recommendation first; each answer was the recommended option.

| ID | Question | Decided |
|---|---|---|
| D18 | Group C's mechanism | A per-term trace of the development step first (no engine change), then **P1**: the parents' term reads their functioning net of borrowed pseudo-self. P2 (displaced coping) is the fallback if the trace shows P1 is not enough. **Revisited 2026-10-24 after the trace** (`docs/review/stage_3_projection_trace.md`: the parents' term contributes −0.0072 ± 0.0296 of the +0.0886, the child's own term +0.0951 ± 0.0362): **P2**, drafted as group C's proposal for the owner's approval before revision 39; P1 is dropped; the half of the own-term difference the stress swing does not explain is traced while drafting, with no engine change. **2026-10-25, after the P2 probe** (`docs/review/stage_3_displaced_coping_probe.md`: P2 at coping size leaves `M11.C.6` at +0.0919 ± 0.0420, and the other half is functioning the stress floor holds, +0.0620 ± 0.0308): **P2 as drafted plus F1** (the floor-held functioning on a ledger the estimator leaves out), the two probed together on Mac Mini1 before revision 39; `M11.C.6` reported as read, never tuned. **2026-10-25, after the joint probe** (+0.018 ± 0.025 with P2 and F1, the proposal's §6.1): the owner decided Q4 as recommended, "draft revision 39 with P2 and F1 as recommended, reporting M11.C.6 as read"; the parents' term is not traced first. The same day, Q1–Q3 as recommended (`OVERFUNCTION` alone; `floor_held` shed with the swings; conserved), and, from the draft, Q5 (`KS00.9` against `M7.E.1f`'s parents' term recorded as T-131, `M7.E.1f` left) and Q6 (each group's approved text lands with its code) |
| D19 | `M11.C.21` | **Lent functioning returns** when the taker's functional level falls, in proportion to the capacity lost, on `M6.I.4`'s conserved quantity |
| D20 | Group F's mechanism | A probe first (no engine change), then **A**: the act is part self-statement, part demand, weighted by the mover's own self-channel weight. **Probe run 2026-10-24** (`docs/review/stage_3_iposition_probe.md`): the direction does not hold clearly with no rule change (the act's effect on the family's excess differs by level by −0.06 ± 0.32), so D20 stands and was not put again |
| D21 | `M11.C.29`'s arm | Mover at **level 60**; **F's spell paused** for the act's PREPARE window in both arms, M's kept; a reported `completions` readout |
| D22 | Group A's mechanism | **P1** (reciprocal contact, general physics) **and P3** (`TRIANGLE` to layer 0) |
| D23 | What P1 and P3 do not reach | `M11.C.27[add_third]` (both) and the calm-state finding **stay accepted findings** unless the new rules move them, each new result reported beside version 1's; `M11.C.44` keeps `TRIANGLE` as an inside act |
| D24 | T-128's two rates | **`tie_wear_rate` ÷ `calm_transfer_rate`**, named per week |
| D25 | The ratio's range | **A quarter to four times** its central value, log-uniform |
| D26 | Where the ratio is swept | **Every directional criterion whose run spans at least one slow tick**, as an exception to T-126's rule made by `M17.E.1`'s own sentence; the record also splits each fraction below and above the ratio's central value |
| D27 | The avoidance gate | **Drop it**: remove `avoidance_available` and its test, restate `M8` and `M13.1` for three places, record it as a version 3 candidate |
| D28 | Fusion | **Leave unbuilt and say so**: `M8.1` marked partial with the reason; to version 3's list |
| D29 | The coach and `SPLIT` | **Intended**; declared as a premise of `M11.C.7`'s sided arm |
| D30 | `M7.D.1a`'s illness size | **Kept on all three channels**; `M7.D.1a` gains a sentence saying why |
| D31 | Batching | **One batch, one regeneration**: probes first, one spec revision, each group built in its own commit with only its criteria run after it, one full regeneration at the end |

## 3.3 The three proposals

Each states the mechanism that would produce the owner's direction, the spec text it changes at revision 39, the
check that no rule writes the outcome (`model_explainer.md` §19), the `M11.5` class of each criterion it touches, and
the stage 4 check of the requirements it rests on against `_LEDGER.md` (Q7: scoped to what stage 3 touches).

### 3.3.1 Group C — projection pushes development down

**The owner's note:** "project DOES impact development, that's the problem."

**Step 0, the trace (D18).** Before any rule is written, a per-term trace of `d` for Nadia in both of `M11.C.6`'s arms
over ages 5–20: the child's own term, the parents' term, the parents' borrowed pseudo-self, the child's load, and the
counts of coping and soothing (`M1.A.5e`). It runs from a tool outside the engine hash (`tools/`, not in any
`HASHED_TOOLS`), writes `docs/review/stage_3_projection_trace.md`, and moves no record. **If the trace shows the
parents' term is not what lifts the child**, P1 is not built and the owner is asked again before revision 39, with P2
(displaced coping, KS04.7, FE01.1) as the fallback.

**The trace's result (2026-10-24, 150 seeds; `docs/review/stage_3_projection_trace.md`).** The condition was met. The
parents' term contributes −0.0072 ± 0.0296 to `M11.C.6`'s +0.0886 ± 0.0439 and the child's own term +0.0951 ± 0.0362.
The step reads the estimator's functional level, which already leaves out pseudo-self (`slow_tick.py::record_week`),
so a conserved exchange moves neither term. About half of the own-term difference is the child's stress swing
(`M1.A.5e`'s coping and soothing) from age 14. Read on the same weeks, P1 would move the parents' term from about −0.3
to about −5 in both arms. The owner chose P2 the same day (D18, revisited); P1 below is kept as the record of what
was proposed and is not built.

**P1, the mechanism (not built: D18 revisited).** `M7.E.1f`'s parents' term reads each living parent's functional level **minus the pseudo-self
they hold borrowed** (`max(0, pseudo_self)`), against their basic level. A parent's own functioning still sets the
climate (KS00.9); functioning they hold because someone gave way to them does not count toward a child's
development. The child's own term is unchanged, so what the child gave away still lowers it. Because `pseudo_self` is
one number per person, P1 subtracts all borrowed pseudo-self, from the spouse as well as the children; that reading
("the parents' own functioning") is what P1 means, and revision 39 says so. No new constant.

**P2, the mechanism (D18 revisited; drafted 2026-10-25 for the owner's approval, `docs/PROPOSAL — DISPLACED COPING
(P2).md`).** New `M1.A.5f`: in a week a person is loaded (`M1.A.5e`: load above their own usual level), an act of a
kind listed as *displacing* in `config/bowen/stress_response.md` (proposed: `OVERFUNCTION`) meets that load through
the act's receiver. The amount coping would have given the sender moves from the receiver's functional level to the
sender's, on both stress swings, conserved; a receiver holding a position toward the sender gives nothing, and the
receiver's floor bounds it. `KS04.7`, `FE01.1` `[K]`: the mother is calmer because she sees the anxiety as coming from
outside herself, and the child ends in the patient position. It writes no pseudo-self, so the development step,
which nets pseudo-self out, reads it. No new constant in the recommended form; one new config list. The proposal's §6
holds the probe, and its §9 the owner's open questions.

**The P2 probe (2026-10-25, 150 seeds; `docs/review/stage_3_displaced_coping_probe.md`, from
`tools/stage_3_displaced_coping_probe.py`, outside every `HASHED_TOOLS`).** P2 at coping size does not move the
direction: `M11.C.6` reads +0.0919 ± 0.0420 with `OVERFUNCTION` and +0.0853 ± 0.0398 with `PURSUE` too, against
+0.0886 as built; the child's own term falls and the parents' term rises by about as much. Forms in which the sender
gains nothing do not move it either. The half of the own-term difference the stress swing does not carry is
functioning on no ledger, +0.0620 ± 0.0308: what the stress floor holds while the swings decay, credited by the
estimator as the child's own. The owner chose, the same day, P2 as drafted plus F1, probed together on Mac Mini1
before revision 39 (D18, third reading). That probe (the proposal's §6.1): with F1 the difference is +0.029 ± 0.025,
and with P2 and F1 +0.018 ± 0.025, an interval that includes zero; no variant shows the expected decline. The owner
decided Q4 there the same day: revision 39 goes ahead on that reading, `M11.C.6` reported as read.

**F1, the floor-held functioning (the owner, 2026-10-25).** Each week a bound stops functional level from moving
while the swings decay (`stress_response.respond`), the amount is added to a new ledger, `floor_held`; the estimator's
reading (`M1.A.4k`) leaves it out, as it leaves out pseudo-self; off the floor it is shed with the swings. No new
constant. The proposal's §4.2 states the trade-off against the review of 2026-10-11.

**`M11.C.21`, the mechanism (D19).** New `M6.I.4a`: when a person holding borrowed pseudo-self loses functional level
by any cause other than an exchange, the share of their borrowed pseudo-self equal to the share of their capacity lost
(the room above the stress floor, `moves.py:322-324`'s `capacity`) returns to the people who lent it. Because
`pseudo_self` does not record lenders, it returns to the tie partners holding lent pseudo-self, in proportion to what
each holds; the sum stays conserved. No new constant. Revision 39 states it as the mirror of `functioning_shift`'s
capacity scaling, which the owner chose for `M11.C.21` on 2026-10-16.

**Spec changes at revision 39** (restated for P2 and F1, 2026-10-25). New `M1.A.5f` (P2); `M1.A.5e`'s "any other act
leaves it" gains the exception, and its floor paragraph F1's ledger; `M1.A.4k` leaves out `floor_held`; `M7.E.1f` one sentence, no rule change (the step reads the stress swing and nets out
pseudo-self); new `M6.I.4a`; `M1.A.5c` if it describes the ledgers; `M11.5`'s `M11.C.6` row (P2 supplies its sign);
`config/bowen/stress_response.md` (the *displacing* table); `model_explainer.md` §§ on the stress response, development
and pseudo-self; `config/bowen/constants.md` unchanged unless the owner chooses a separate size (the proposal's §9).

**§19 check** (restated for P2). P2 states what one act does, for every sender and receiver; it names no projection,
no child and no outcome. Who sends `OVERFUNCTION` to whom, under what load, and whether the receiver holds a position
are decided by focus (`M4.D.1g`, `M1.B.8a`), learning and the run. One risk is stated in the proposal's §5: `M11.C.6`'s
arms differ only in the acts P2 acts on, so P2 supplies the direction's sign; whether it outweighs the lift the trace
found is read, not set; the probe shows it cancels at coping size. F1 records functioning a bound kept and changes
no act, so it supplies no sign. `displaced-coping-removed` and `floor-held-credited` must each move it back. `M6.I.4a` returns a conserved quantity on a
cause the rule does not choose (a collapse); whether the reciprocity inverts is read, not set.

**Classes (`M11.5`).**

| Criterion | Class | Why |
|---|---|---|
| `M11.C.6` | composite | the decline needs focus, learned projection, the parents' load at the time, the child's own responses and P2 together; P2 supplies its sign (the proposal's §5) |
| `M11.C.2`, `M11.C.2[twelve-person family]` | composite | unchanged; reported, not targeted |
| `M11.C.21` | composite | the inversion needs the collapse, `M6.I.4a` and the scripted exchange; `M6.I.4a` alone returns functioning but does not decide whether the child rises past the arm without collapse |
| new unit tests of P2 and F1 (S3.1a–S3.1d) | premise (unit) | a single rule produces each, so each confirms the code and is never reported as a finding |
| new unit test of `M6.I.4a` | premise (unit) | as above |

**Stage 4.** Check `M1.A.5e`, `M1.A.5f`, `M1.A.4k`, `M7.E.1f`, `M6.I.4`, `M1.A.5c` and `M11.C.6`'s row against FE01.1, FE06.3,
FE07.4, KS04.7 and KS00.9, and fix in revision 39 what is found.

### 3.3.2 Group F — at a low level an I-position comes out as a demand

**The owner's note:** "the lower the "level" the more an I position comes out as conflictual demand. The lower the
LEVEL the more anxiety it creates."

**Step 0, the probe (D20).** One genuine `I-POSITION` from a level-35 and a level-60 mover, same target and third, same
seeds, the act week and its PREPARE window clear of any spell. Read the target's acute anxiety and the family's summed
excess over the next 6 weeks, the `CONFLICT` share on the mover's tie, and completions. Outside the engine hash; writes
`docs/review/stage_3_iposition_probe.md`. It tells how much of the direction the existing outward-axis term already
gives. **If the probe shows the direction holds clearly with no rule change**, the owner is asked again whether to build
A or only the criterion (option B).

**The probe's result (2026-10-24, 200 seeds; `docs/review/stage_3_iposition_probe.md`).** The condition was not met,
so D20 stands. With both movers acting, the level-35 family has more excess than the level-60 family (+7.40 ± 0.36),
but the no-act controls differ by as much: the act's own effect on the family's excess is −1.42 ± 0.32 at level 35
and −1.36 ± 0.27 at level 60 (difference in differences −0.06 ± 0.32), and the same holds for the target's acute
anxiety and the `CONFLICT` share. At level 35, 132 of 176 genuine sequences failed PREPARE. For `M11.C.49`'s arm
design: a control or the `w = 1` mutant is needed to separate the act from level, and the low-level mover needs a
PREPARE window calm enough to reach DEFINE.

**A, the mechanism.** When a genuine `I-POSITION` reaches DEFINE, the act is a mixture: a self-statement share `w` and
a demand share `1 − w`, where `w` is the mover's own self-channel weight (`(FL/100)^2 + 0.5 × perspective × (1 −
that)`, `M4.D.1a`, `policy.py:183-193`). The demand share adds `(1 − w) × assertion_gain` to the impingement
delivered, scales the debit of the target's felt contact by `w`, and raises the mover's outward axis by `(1 − w) ×
assertion_evidence_gain`. At perspective 1.0, `w` is about 0.58 at level 40 and 0.68 at level 60. No new constant: it
reuses `assertion_gain` and `assertion_evidence_gain` and derives the share from level, as `M10.A` prefers. The anger
gate (`M5.D.4a`) stays; `M5.F.4`'s binary perspective gate becomes this continuous mixture, which `M5.F.2c` ("the flip
is gradual … MUST NOT be a step") already asks for.

**`M11.C.29`'s arm (D21).** Mover F at basic and functional level 60; F's spell suspended from `t0` to `t0 +
prepare_ticks` in both arms; M's spell kept, so the third still carries load (T-020); a reported `completions`
readout, so under `M11.1f` the row fails if no move completes. The arm settings go in `config/bowen/criteria.md`
(moves the hash; it rides in group F's commit).

**Spec changes at revision 39.** `M5.F.4` (the mixture); `M5.F.2c`'s note that the step is gone; `M11.C.29`'s arm; new
criterion `M11.C.49` (§3.7); `model_explainer.md` §5.3 and §11.3.

**§19 check.** A writes the act's content — how much of it lands as pressure — which is the set layer, the same class
as `assertion_gain` and `assault_gain` today. It writes no response. Whether the target answers with `CONFLICT`,
`PURSUE` or `DISTANCE` comes from the target's learned values; whether the family's anxiety rises comes from appraisal.
So "the target feels more pressure from a lower-level mover" is a premise, and "the family's anxiety and conflict rise"
are findings, and the report keeps them apart.

**Classes (`M11.5`).**

| Criterion | Class | Why |
|---|---|---|
| `M11.C.49`, the target's felt impingement limb | premise | A alone produces it; reported as confirming the code |
| `M11.C.49`, the family-excess and `CONFLICT`-share limbs | composite | they need A, appraisal, the target's learned response and contagion |
| `M11.C.29` | mixed (unchanged since revision 32) | the budget limb is proved by a unit test; the symptom limb is composite |
| `M11.C.32` | premise (unchanged) | the anger gate; must still pass |
| `M11.C.18`, `.20`, `.33`, `.5`, `.16`, `.17` | unchanged | reported; any changed verdict is a new result under a new rule |

**Confound, declared in `M11.C.49`'s row:** level also moves the contact band and hold capacity, so the two arms differ
in more than the mixture. The mutant that sets `w` to 1 isolates A's part.

**Stage 4.** Check `M5.D`, `M5.F.2c` and `M5.F.4` against KS05.5, KS06.11, FE04.13, L10.12 and the KB12 interview, and
grade the new rule `[K]` with the interview cited, never as Bowen's.

### 3.3.3 Group A — seeking out another relieves, so it is learned

**The owner's note:** "Triangling - seek out other, is a natural human behavior."

**P1, the mechanism (D22).** New `M4.C.1d`, general move physics: when a receiver appraises an approach act's contact
as relief (its two-sided `M4.C.1` deviation falls), a share `reciprocal_contact_share` (`[I]`, new) of the contact
delivered returns to the **sender's** felt contact on that tie, and the sender appraises it through their own
deviation. It applies to every approach act, not to `TRIANGLE` alone. A seeker whom anxiety has put on the "too
little" side of their optimum (`M4.C.1b`) feels the returned contact as relief; a calm seeker near the optimum feels
little or a cost. The learner already credits the seeker's own change in anxiety, so the relief reaches the value of a
`TRIANGLE` (and of any approach act) without any change to the learner.

**P3 (D22).** `TRIANGLE` moves from layer 2 to layer 0 of `M4.D.3a`'s ordering in `config/bowen/event_kinds.md`, beside
`DISTANCE` and `CUTOFF`. The layer-2 placement is an `[I]` choice the config says has no source; KS15.10 and FE06.20
read triangling as old and automatic. This removes a gate by which falling functional level makes triangles less
available under load. No new constant.

**Spec changes at revision 39.** New `M4.C.1d`; `M4.D.3a`'s table note; `config/bowen/event_kinds.md`'s note on the
layers; `reciprocal_contact_share` in `config/bowen/constants.md` and `constants_changes.md`, graded `[I]`, its value
set before the acceptance suite runs (`M10.B.4`) and stated with its reasoning, not tuned to a criterion; `M10.C.1a`'s
list; `model_explainer.md` §§ on contact and triangles.

**§19 check.** Nothing sets a `TRIANGLE`'s relief or compares it with `DISTANCE`'s. The returned contact is ordinary
contact, appraised by the sender's own two-sided deviation, so it relieves only when the sender's state makes
closeness wanted. Whether `TRIANGLE` is reused, and whether triangles are quieter when calm, are learned from felt
relief. A rule that raised `triangle_transfer_rate` or added a bonus to a `TRIANGLE`'s learned value to reach parity
would write the outcome; neither is proposed. P3 removes an invented gate rather than adding a preference.

**Classes (`M11.5`).**

| Criterion | Class | Why |
|---|---|---|
| `M11.C.42` | composite | reuse needs P1, appraisal, the learner and the third's response |
| `M11.C.45` | composite | quiet-when-calm needs P1's two-sidedness, the spell and learning |
| `M11.C.44` | composite (unchanged) | `TRIANGLE` stays an inside act (D23); a changed verdict is reported, not prevented |
| `M11.C.27[stable,add_third]`, `[unstable,add_third]` | composite (unchanged) | stay accepted findings unless they move (D23) |
| new unit test of `M4.C.1d` | premise (unit) | never a finding |

**Stage 4.** Check `M1.C.1`, `M4.C.1`, `M4.D.3a` and `M5.A.1` against L10.3, L17.1, KS03.1, FE06.2, FE06.20 and
KS15.10.

## 3.4 T-128 — `M17.E.1`'s named ratio

**The ratio (D24).** `tie_wear_to_transfer_ratio` = `tie_wear_rate` ÷ (`slow_tick_fast_ticks` × `calm_transfer_rate`):
tie wear per week against calm transfer per contact. `slow_tick_fast_ticks` (52) is read from `constants.md`, never
written as a literal. Central value 0.05 ÷ (52 × 0.05) = 1/52 ≈ 0.0192. The units differ (per week against per
contact); the record says so beside every fraction.

**Where it is named.** `M10.C.1a`'s list gains the ratio, graded `[I]`, with its two rates and `M17.E.1`'s sentence as
the reason. `M17.E.1` names it.

**Its range (D25).** In `config/sweep/sweep.md`, a new table of named ratios: name, numerator, denominator, the
constant that converts their periods, low, high, scale, grade, spec. Low 1/208 ≈ 0.0048, high 4/52 ≈ 0.0769, `log`.
Each sample draws the ratio log-uniformly and `calm_transfer_rate` from its own range (0.025–0.1), and derives
`tie_wear_rate` from the two, so each sample's ratio is exact. **A consequence for the owner to see before the full
sweep:** the derived `tie_wear_rate` then spans 0.00625 to 0.4, one eighth to eight times its central value, wider than
the half-to-double rule every other row follows; it stays inside its admissible bound of 1. The parser rejects a
ratio whose derived numerator could leave that bound.

**Where it is swept (D26).** Every directional criterion whose run is longer than `slow_tick_fast_ticks` fast ticks, so
that a slow tick fires before its last tick (the owner's Q14, 2026-10-25, refining "at least"), since tie wear acts only
at a slow tick. The tool reads each criterion's run length and lists the criteria it added the ratio
to and the ones it did not, with the reason. This is an exception to T-126's rule (a constant that acts and that no
row credits is left out), made by `M17.E.1`'s own sentence, and `sweep.md`'s preamble says so. The sample count stays
at 8, so the ratio adds no runs. For each criterion the record reports *n of S* overall and the counts below and above
the ratio's central value, so a direction that flips across it shows. `M17.E.1` stays *partial* in
`tools/spec_coverage_overrides.json` until the stage-end sweep has run with the ratio; then it is *done*.

**The tool change** is in `tools/version_2_sweep.py`, which is outside the engine hash; a ratio draw changes a sample's
edits, so the sweep cache keys stay correct without a hash change.

## 3.5 T-014 — the four questions

- **The avoidance gate (D27).** Remove `avoidance_available` and its unit test from `live_positions.py`; restate `M8`'s
  preamble ("written for three places"), `M8.5` and `M13.1`; `docs/version_2_plan.md`'s list and `M5.C` (where it
  names the gate) say it is not built and is a version 3 candidate. Close T-011's `live_positions.py:1` note. Edits
  `src/bowen`, so it rides in the batch's first commit (§3.6).
- **Fusion (D28).** `M8.1` is marked *partial* in `tools/spec_coverage_overrides.json` with the reason: the engine has
  no fusion state, and fusion is a readout the engine must not read. `M8.1`'s text gains one sentence saying the
  fused-occupant branch is not reached. To version 3's list. No engine change.
- **The coach and `SPLIT` (D29).** `M11.C.7`'s row and its `M11.5` class row state that the sided arm needs the coach
  drawn in, and that the coach's always going along is a declared premise of the arm, since a decline would make the
  coach's skill a variable the criterion holds fixed. Wording only.
- **`M7.D.1a` (D30).** One sentence: every onset keeps the "personal injury or illness" size on all three channels,
  because the explainer treats the channels as substitutable (`model_explainer.md`, "the same deficit presents
  interchangeably as physical illness, mental illness, or social dysfunction"), so the channel a symptom lands in does
  not change the family's load by a rule; the size stays `[I]`. Wording only.

## 3.6 T-129 and the other edits that wait for an engine-hash change

**Where they go.** The first change in this stage that moves the engine hash is the batch's first commit (§3.8, step
6). That commit holds, and only holds:

- T-129: MR-2 (`moves.py:63-68`), MR-9 (`rejection.py:62`, then empty SP-B's `src/bowen` allowlist as each name is
  fixed), MR-13 (`objects.py:247-248`, `:277`, `rejection.py:10`, `invariants.py:26`, including "exactly"), MR-26 (the
  ruff F items in `src/bowen` and `tools/level_occupancy.py:139`, and a new `tests/test_lint.py` running `ruff check
  --select F` over `src/`, `tools/` and `tests/`);
- T-127: the two status lines in `config/bowen/contact_classes.md` and `config/bowen/event_sizes.md`;
- T-012's MR-23: the Phase C ensemble record moved onto `docs/records_cache/`, rendered from the cache, its tool out of
  its own hash, the full render compared in `tests/bowen/test_ensemble_record.py`;
- D27's removal of `avoidance_available`;
- T-130's corrected values, if T-130 (step 1) finds any.

**It changes no result.** The golden fingerprint fixtures (learnings P23) must be unchanged by this commit; that is the
proof it moves the hash without changing what a run computes (unless T-130 corrected a value, in which case the commit
says so and the fixtures are regenerated in it). Group C's rule then lands in the next commit.

**What it makes stale.** Every result keyed by the engine hash: the Phase C ensemble record, the Phase D part I and
part II ensemble records (`phase_d_ensemble.json`, `phase_d2_ensemble.json`), the Phase C and Phase D mutation records
(`mutation.json`, `phase_d_mutation.json`), the D9 sweep record (`sweep.json`), the diagnostic record
(`diagnostic.json`), the level-occupancy record, the seed-7 trace, chart and family-diagram export,
`docs/spec_coverage.md`, the after section of `docs/review/group_g_readout.md`, and the stage 2 sweep
(`sweep_v2.json`, `docs/version_2_sweep_record.md`). The record-current tests turn red until they are regenerated.

**What rerunning them costs.** Measured at revision 36's regeneration on this Mac with 10 workers (`CHANGELOG.md`,
2026-10-23): 2 h 36 min of runs for every record (the ensemble records 3 min 10 s, 12 min 7 s and 1 h 31 min; the
mutation records 22 min 34 s and 12 min 13 s; the D9 sweep 5 min 42 s; the diagnostic 7 min 33 s; the rest under 3
minutes). Under D31 this runs **once**, at the end of the batch (step 10), not after this commit. The stage 2 sweep is
not rerun between: under D17 the full sweep runs only at the stage's end (step 11). Its measured cost at stage 2 was
27 min (Phase C, 12 criteria), 3 h 34 min (part I) and 6 h 52 min plus 7 h 54 min (part II, split across both Macs):
about 18 h 47 min of machine time, about 8 h wall when part II is split. That is a measurement at revision 36's code,
not a projection for revision 39's.

## 3.7 Criteria and their tests

Each test name carries the criterion's ID and a spec ID. Acceptance tests assert a direction of difference between two
arms on identical seeds, never an absolute threshold on a model result, and each counts as coverage only once the
named mutation has turned it red (or reversed it) in a record. A premise passing confirms the code and is never
reported as a finding (`M11.5`). Provisional IDs: `M1.A.5f` (and F1's `floor_held`, inside `M1.A.5e` and `M1.A.4k`), `M6.I.4a`, `M4.C.1d` and `M11.C.49` are new at revision 39.

| ID | Criterion | Test | Proved failing by |
|---|---|---|---|
| S3.1a (`M1.A.5f`, D18 revisited) | premise: a loaded sender's `OVERFUNCTION` moves the coping amount from the receiver's functional level and stress swing to the sender's; the pair's total functional level is unchanged | `tests/bowen/test_stress_response.py::test_m1a5f_s3_1a_displaced_coping_moves_functioning_from_receiver_to_sender` | `displaced-coping-removed` |
| S3.1b (`M1.A.5f`) | premise: nothing moves when the sender is calm, when the receiver holds a position toward the sender, or when the receiver is at their floor | `tests/bowen/test_stress_response.py::test_m1a5f_s3_1b_no_displacement_when_calm_held_or_at_the_floor` | `displaced-coping-unguarded` (the three guards removed) |
| S3.1c (`M1.A.5f`) | check: the displacing kinds are read from `config/bowen/stress_response.md`, and a kind listed both soothing and displacing fails at load | `tests/bowen/test_config.py::test_m1a5f_s3_1c_displacing_kinds_come_from_config` | the list moved into code |
| S3.1d (`M1.A.5e`, `M1.A.4k`, F1) | premise: a calm week at the floor with a positive swing adds what the floor held to `floor_held`, leaves `functional_level − stress_swing − adjust_swing − floor_held` unchanged, and the estimator's reading leaves it out; off the floor it is shed with the swings | `tests/bowen/test_stress_response.py::test_m1a5e_s3_1d_floor_held_functioning_is_on_a_ledger` | `floor-held-credited` (the estimator reads `floor_held` as own functioning) |
| S3.2 (`M11.C.6`) | composite: the child's basic level at 20, minus the anchor, is lower with projection than with every projection-type act toward her withheld | `tests/bowen/test_phase_d2_criteria.py::test_m11c6_multigenerational_decline_in_projection_line` (existing) | `displaced-coping-removed`, `floor-held-credited` (both new) and `development-fixed-at-anchor` (existing), each in the Phase D mutation record |
| S3.3 (`M11.C.2`, both variants) | composite: symptoms concentrate on the child who received most projection-type acts | the existing tests | `witness-path-removed`, `weakness-focus-and-steering-removed` (existing), run if the row passes |
| S3.4 (`M6.I.4a`, D19) | premise: when a taker's functional level falls by a non-exchange cause, the borrowed pseudo-self returns to its lenders in proportion to the capacity lost, and the family's total pseudo-self is unchanged | `tests/bowen/test_moves.py::test_m6i4a_s3_4_lent_functioning_returns_with_lost_capacity` | `pseudo-self-return-removed` |
| S3.5 (`M11.C.21`) | composite: the child rises more after both projecting parents collapse than in the arm without collapse | `tests/bowen/test_phase_d_criteria.py::test_m11c21_reciprocity_inverts_on_projector_collapse` (existing) | `pseudo-self-return-removed`, added to the Phase D mutation record (the spec's named mutant, "make `M6.I.4`'s transfer one-way", has no entry today) |
| S3.6 (`M5.F.4`, D20) | premise: a lower-level mover's genuine act delivers more impingement and debits less felt contact than a higher-level mover's, by `1 − w` | `tests/bowen/test_iposition.py::test_m5f4_s3_6_demand_share_falls_with_level` | `demand-share-removed` (`w` := 1; named `demand-share-level-blind` before revision 39's text) |
| S3.7 (new `M11.C.49`, D20) | two arms, mover at level 35 against 60, same seeds, act clear of the spell: the family's summed excess over 6 weeks is higher and the `CONFLICT` share on the mover's tie is higher at 35 (composite limbs); the target's felt impingement is higher at 35 (premise limb, reported); completions are reported | `tests/bowen/test_phase_c_criteria.py::test_m11c49_low_level_iposition_raises_family_anxiety` | `demand-share-removed` turns the composite limbs red; `demand-share-inverted` (`w` := 1 − `w`) reverses them |
| S3.8 (`M11.C.29`, D21) | arm property: mover at level 60; F's spell absent from `t0` to `t0 + prepare_ticks` in both arms; M's spell present; a `completions` readout reported | `tests/bowen/test_phase_c_criteria.py::test_m11c29_s3_8_act_window_clear_of_the_spell` | the pause removed (the arm test fails) |
| S3.9 (`M11.C.29`) | mixed: the third's symptom weeks are fewer after a genuine move than after distance in disguise; the row fails if `completions` is 0 (`M11.1f`) | `tests/bowen/test_phase_c_gate.py::test_m11c29_relief_and_differentiation_differ_in_time_course` (existing) | the budget limb by `test_m6i1_only_a_completed_exchange_reduces_the_budget` (existing); the symptom limb by `budget-reduction-removed`, added to the Phase C mutation record |
| S3.10 (`M4.C.1d`, D22) | premise: a receiver who appraises an approach act as relief returns `reciprocal_contact_share` of its contact to the sender; a receiver who appraises it as a cost returns none; the same for `TRIANGLE`, `STAY-IN-CONTACT` and `PURSUE` | `tests/bowen/test_appraise.py::test_m4c1d_s3_10_relieved_receiver_returns_contact` | `reciprocal-contact-removed` (share 0); `reciprocal-contact-triangle-only` fails the act-generality limb |
| S3.11 (`M4.D.3a`, D22) | check: `TRIANGLE` is layer 0, so its availability does not fall with functional level | `tests/bowen/test_policy.py::test_m4d3a_s3_11_triangle_is_layer_zero` | the config row set back to layer 2 |
| S3.12 (`M11.C.42`) | composite: a relieving triangle is reused more than in the arm where the third is unavailable at `t0` | `tests/bowen/test_phase_c_gate.py::test_m11c42_relieving_triangle_is_reused` (existing) | `reciprocal-contact-removed` (new), and the existing `learner-disabled`, `learner-inverted`, `triangle-help-ignored`, `triangle-credit-cross-person`, run if the row passes |
| S3.13 (`M11.C.45`) | composite: the `TRIANGLE` rate per person-week is lower when calm than under the spell | `tests/bowen/test_phase_c_gate.py::test_m11c45_triangles_quiet_when_calm` (existing) | `relief-tension-independent` (existing); `one-sided-deviation` (contact always relieves) |
| S3.14 (`M11.C.44`, `M11.C.27[add_third]` ×2, D23) | composite, unchanged: reported beside version 1's verdict | the existing tests | the existing mutants, run if a row passes |
| S3.15 (`M11.5`, §19) | every criterion this stage touches has a class row; `M11.C.49`'s row lists its premise limb apart from its composite limbs; the engine imports nothing from `readouts` | `tests/test_spec_consistency.py` (class rows); `tests/bowen/test_engine_purity.py` (no readout import; extended if it does not already check this) | a class row deleted; a readout imported into `src/bowen/engine` |
| S3.16 (`M17.E.1`, D24, D25) | a sample draws the ratio log-uniformly within its range and derives `tie_wear_rate` exactly from it and the sampled `calm_transfer_rate`, with 52 read from `constants.md` | `tests/test_version_2_sweep.py::test_m17e1_s3_16_ratio_drawn_and_numerator_derived` | the numerator drawn independently; the period written as a literal |
| S3.17 (`M17.E.1`, D26) | the ratio enters every directional criterion whose run is longer than one slow tick and no other (Q14), and the record lists both groups | `tests/test_version_2_sweep.py::test_m17e1_s3_17_ratio_scope_by_run_length` | the ratio added to every criterion |
| S3.18 (`M17.E.1`, D26) | the record gives *n of S* and the counts below and above the ratio's central value, and no cell prints PASS or FAIL | `tests/test_version_2_sweep.py::test_m17e1_s3_18_record_splits_by_ratio` | the split dropped; a fraction rendered as a verdict |
| S3.19 (`M17.E.1`, D25) | the ratio is named in `M17.E.1` (`M10.C.1a` is unchanged, revision 39's §20.2) and has a `sweep.md` row with a grade; a ratio whose derived numerator could leave its bound fails at load | `tests/test_version_2_sweep.py::test_s3_19_ratio_row_parsed_strictly` | the bound check removed |
| S3.20 (`M8`, D27) | `positions_live` has exactly three callers and no uncalled helper beside it; `M8`'s preamble says three places | `tests/bowen/test_live_positions.py::test_m8_s3_20_predicate_has_three_callers` | `avoidance_available` restored |
| S3.21 (`M8.1`, D28; `M11.C.7`, D29; `M7.D.1a`, D30) | the coverage override marks `M8.1` partial with its reason; the three spec sentences exist | `tests/bowen/test_spec_coverage.py` (override present); `tests/test_spec_consistency.py` | — (wording) |
| S3.22 (T-129) | `ruff check --select F` is clean over `src/`, `tools/` and `tests/`; SP-B's `src/bowen` allowlist is empty | `tests/test_lint.py::test_s3_22_ruff_f_clean` (new file); SP-B's gate test, `tests/test_named_tests_resolve.py` (`SOURCE_ALLOWED`) | red before the fixes (shown at step 6) |
| S3.23 (T-129) | the batch's first commit changes no result | the golden fingerprint tests, unchanged by that commit | — |
| S3.24 (W1–W4) | every record regenerated once on the batch's final hash; the changes page lists every changed field; the status page shows each new verdict beside version 1's and revision 38's; every row has a status | the record-current tests; `tools/record_changes.py revision-39`; `tests/test_version_2_status.py` (S2.8–S2.11) | the existing ones |
| S3.25 (W4) | every passing directional criterion, `M11.C.49` among them, has a stage-end sweep fraction | `tests/test_version_2_status.py::test_w4_s2_9_sweep_column_is_a_fraction_not_a_verdict` (existing) | the existing one |

## 3.8 Order of work

| Step | Work | Depends on | Moves the hash |
|---|---|---|---|
| 1 | This plan approved. T-130: check `config/bowen/event_sizes.md`'s values against Holmes and Rahe (1967)'s table; correct or drop "unchecked" (the correction waits for step 6) | — | no |
| 2 | The projection trace (§3.3.1) and the I-position probe (§3.3.2), each from a tool outside the engine hash; results written to `docs/review/` and `CHANGELOG.md` | 1 | no |
| 3 | If step 2 changes a proposal (D18's or D20's condition), the owner is asked again, one decision at a time. **Done 2026-10-24–25:** the owner chose P2 (D18 revisited); P2 drafted and probed (`docs/PROPOSAL — DISPLACED COPING (P2).md`, `docs/review/stage_3_displaced_coping_probe.md`): it does not move `M11.C.6` at coping size, and the owner chose P2 plus F1 (D18, third reading). P2 and F1 probed together on Mac Mini1 2026-10-25 (the proposal's §6.1): +0.018 ± 0.025, unresolved. **Next:** step 4 | 2 | no |
| 4 | Spec revision 39 drafted from §3.3–§3.5 and the stage 4 checks: `M1.A.5e`, `M1.A.5f`, `M1.A.4k`, `M7.E.1f`, `M6.I.4a`, `M5.F.4`, `M5.F.2c`, `M11.C.29`, `M11.C.49`, `M4.C.1d`, `M4.D.3a`, `M10.C.1a`, `M17.E.1`, `M8`, `M8.1`, `M8.5`, `M13.1`, `M11.C.7`, `M7.D.1a`, the `M11.5` rows; `config/bowen/stress_response.md`'s *displacing* table; the revision log, the indexes, the rules view. **Approved by the owner before step 5.** Written in `docs/spec_revision_39_draft.md`; applied to the spec, config and explainer with each group's code (the owner's Q6, 2026-10-25). **Done 2026-10-25:** parts 1–3 drafted, the owner's Q1–Q14 decided as recommended, and revision 39 approved by the owner as a whole | 3 | no (wording) |
| 5 | Tests first: S3.1a–S3.1d, S3.4, S3.6–S3.8, S3.10, S3.11, S3.16–S3.20, S3.22 red. **Done 2026-10-25:** sixteen tests written, each shown failing on its missing rule with an `AssertionError` that names it (run with `--runxfail`), none on an import error or a typo. **Marking:** the fifteen default-run tests are `xfail(strict=True, raises=AssertionError)`, each with a reason naming the step that turns it green (step 6: S3.20, S3.22; step 7: S3.1a–S3.1d, S3.4; step 8: S3.6, S3.8; step 9: S3.10, S3.11; step 10: S3.16–S3.19), so a test that starts passing, or fails for any reason but an assertion, turns the suite red; S3.7 (`M11.C.49`) is `ensemble`-marked and out of the default run, like every criterion test, since it runs an ensemble. **Names the tests fix for the build:** `EventKinds.displacing` and `load.load_displacing(kinds, path)` (a `## Displacing` table in `stress_response.md`, columns `kind`, `why it meets the sender's load through the receiver`); `Person.floor_held`; `moves.return_lent_functioning(state, params, week_start)`; `reciprocal_contact_share` in the engine parameters; `mover_level` 60 in `M11.C.29`'s settings and a `completions` readout; `sweep_config.load_ratios(text, central)`, returning one entry per ratio with `name`, `numerator`, `denominator`, `period`, `low`, `high` and `grade`; a sweep row's `ratio`, a result's `ratio_split` (`below` and `above`, each `held` and `counted`), the record's scope lines (`entered`, and `not entered` with the run length and `slow_tick_fast_ticks`) and two summary columns named for below and above. **Readings recorded for the build:** `M1.A.5f`'s "room above its floor" is read as `functional_level − floor` (the draft's parenthesis gives the floor itself, `max(0, basic_level − stress_floor_below_basic)`; fix the wording at step 7); S3.8's window includes `t0 + prepare_ticks`; S3.19 checks `M17.E.1`, since the draft's §20.2 leaves `M10.C.1a` unchanged, and S3.6's and S3.7's mutant is the draft's `demand-share-removed`: §3.7's rows are corrected to match. S3.1c also expects a `ConfigError` for an unknown kind and for a repeated row, as `load_soothing` gives; the draft's §4 names only a kind both soothing and displacing, so step 7's config preamble says all three. S3.7 finds `M11.C.49` in any of the three criterion registries; S3.17 checks the boundary with `M11.C.23`, a 52-week run, not entered. Suite: 1232 tests, 1216 pass and 16 are strict expected failures; the ensemble-marked criterion tests rise from 51 to 52 | 4 | no |
| 6 | **Commit 1 of the batch, the first to move the hash:** §3.6's edits; S3.20, S3.22 green; S3.23 holds. **Done 2026-10-25:** T-130's wording in `config/bowen/event_sizes.md` (four item names to Table 3's, "unchecked" dropped); T-014's revision 39 part 3 text in the `M8` preamble, `M8.1`, `M8.5`, `M13.1`, `M11.C.7` and its `M11.5` row, `M7.D.1a` (and `M1.F.6a`), `avoidance_available` and its test removed, `positions_live` the one definition with three callers; T-129's fourteen ruff F findings fixed and `SOURCE_ALLOWED` empty; T-127's status lines. Engine hash 66dcbb1e8ba5 → f90285984f32; the golden fingerprint tests pass unchanged (S3.23) and the seed-7 renders differ only in the revision label. **Deviations, Claude's recommendations for the owner's review:** (a) the record-current tests D31 leaves red until step 10 are marked by `tests/open_batch.py` (`stale_in_open_batch`): an expected failure only while `OPEN` is set, the record is byte-identical to the file at `9ebf327`, and the test passes when rerun with the hash computed on the tree at `9ebf327`, checked by `tests/test_open_batch.py` (all records as found or none), so the suite stays green and the gate line counts them as a third figure; (b) MR-23 (T-012), a records-tool change, moves to step 10, before the one regeneration; (c) the approved draft text carries a ⟦rev39⟧ tag in the spec. Suite: 1236 collected, 1211 pass, 14 strict expected failures, 11 expected failures of the open batch | 5 | **yes** |
| 7 | **Group C:** P2 (`M1.A.5f`, the *displacing* list), F1 (`floor_held`) and `M6.I.4a`; S3.1a–S3.1d, S3.4 green; `displaced-coping-removed`, `displaced-coping-unguarded` and `floor-held-credited` added to the Phase D mutation record; then only `M11.C.6`, `M11.C.2` (both) and `M11.C.21` run, with their mutants (S3.2, S3.3, S3.5). **Done 2026-10-26:** `EventKinds.displacing` from a `## Displacing` table in `config/bowen/stress_response.md` (`OVERFUNCTION` alone; loading fails on an unknown kind, a repeated row, or a kind both soothing and displacing); `stress_response.displace` before the per-person response; `Person.floor_held`, written in calm weeks and left out by the estimator; `moves.return_lent_functioning` before `relax_pseudo_self`, the week's exchange read from its `functioning_shift` records (`moves.exchanged`); the stress floor in one function (`moves.stress_floor`). The spec takes group C's text (`M1.A.5e`, `M1.A.5f` with the step 5 wording fix, `M1.A.4k`, `M7.E.1f`, `M6.I.4a`, `M11.C.6`'s and `M11.C.21`'s mutation columns and `M11.5` rows, the register). Four mutants added to the Phase D mutation record's tool (the three named and `pseudo-self-return-removed`, an edit in `moves.py` so one mutant proves S3.4 and S3.5), and `M11.C.2` added to `witness-path-removed`'s and `weakness-focus-and-steering-removed`'s criteria. Unit proofs: `review/stage_3_unit_mutation_gate.md`, 9 of 9. Engine hash f90285984f32 → b8e9ad1333e2; the golden fingerprints moved (both runs) and were taken again with the test module's own functions. **The targeted run** (`review/stage_3_step_7_criteria.md`, Mac Mini1, not a record): `M11.C.6` FAIL, `development` +0.041 ± 0.027 (revision 38: +0.089 ± 0.044), reported as read (Q4), not tuned; `M11.C.2[twelve-person family]` FAIL, +0.33 ± 1.8 (+0.95 ± 1.6); `M11.C.2` PASS, +3.37 ± 3.3, p 0.046 (FAIL, +0.35 ± 3.6), its two mutants (`review/stage_3_step_7_c2_mutants.md`): `witness-path-removed` turns it red (+0.90 ± 1.9, p 0.2), `weakness-focus-and-steering-removed` does not (+3.31 ± 3.5, p 0.04), so here the pass is carried by witnessing, the reverse of what the spec's row records for the twelve-person family at revision 31; `M11.C.21` FAIL, `target_rise` +0.056 ± 0.11, p 0.22 (−0.068 ± 0.089). The mutants of a failing criterion did not run (`M11.1d`). **Deviations, Claude's recommendations for the owner's review:** (a) a receiver who holds a position stays in the share's denominator, as `M1.A.5f`'s text reads (the step 3 probe left them out); (b) receivers are living family members (`Role.MEMBER`), not the coach; (c) `M6.I.4a` states the taker is not taken below 0, and a fall within `invariant_tolerance` is rounding; `M1.A.5f` states a receiver below the floor gives nothing; (d) `tests/open_batch.py` also pins the mutation tools' mutants, and their criteria, to those found at `9ebf327`, so the mutants step 7 adds are left for step 10 as D31 says; (e) T-134 recorded (`M6.I.4a`'s return with no lender is not counted). Suite: 1242 tests: 1222 pass, 9 strict expected failures, 11 open-batch expected failures (`python3 -m pytest tests/`, 2026-10-26) | 6 | **yes** |
| 8 | **Group F:** A, `M11.C.29`'s arm, `M11.C.49`; S3.6–S3.8 green; then only `M11.C.49`, `M11.C.29`, `M11.C.32`, `M11.C.5`, `M11.C.18`, `.20`, `.33` run, with their mutants (S3.7, S3.9). **Done 2026-10-26:** `iposition.demand_share` (`1 − w`, `w` the mover's self-channel weight) carried on the `DEFINE` event as `Event.demand_share`, 1 for the angry assertion form; it adds `(1 − w) × assertion_gain` to the impingement delivered (`appraise.py`), scales the target's debit by `w` and raises the mover's outward axis by `(1 − w) × assertion_evidence_gain` (`moves.iposition_effect`); `assertion_perspective_threshold` retired (kept in `RETIRED` so the frozen constants load); `M11.C.32` counts share ≥ 1. `M11.C.29`'s arm: F at `mover_level`, F's spell suspended over `act_window(t0)`, the `completions` readout and the `(I-POSITION completed)` required move. `M11.C.49`: four runs per seed, the act started at `DEFINE` by a `before` hook (the owner's Q9), F's own `I-POSITION` withheld in every week of all four runs, the F–M tie held open to `t0`. S3.6 and S3.8 green, their strict marks removed; S3.7's test is `ensemble`-marked and red, as `M11.C.49` fails (T-137). Unit gates for S3.6 and S3.8 added (`review/stage_3_unit_mutation_gate.md`, 11 of 11 proved). Mutants added: Phase C `demand-share-removed`, `demand-share-inverted` (`M11.C.49`), `budget-reduction-removed` (`M11.C.29`), `iposition-debit-removed` (`M11.C.5`); Phase D `stress-response-removed` (`M11.C.18`), `borrowed-pseudo-self-read-as-own` (`M11.C.20`), `dependence-gate-removed` (`M11.C.33`). Engine hash b8e9ad1333e2 → b819660f58b9; golden fingerprints taken again with the module's own `_records` and `_fingerprint`. Targeted run on Mac Mini1 (`review/stage_3_step_8_criteria.md`): `M11.C.49` FAIL (`family_excess` +2.17 ± 3.0, `conflict_share` +0.003 ± 0.028; `target_impingement` +0.085 ± 0.023 and `completions` −0.31 ± 0.074 reported), `M11.C.29` FAIL (`third_symptom_weeks` +0.9 ± 3.7; the genuine arm completes in 1 of 100 seeds, T-138), `M11.C.32` PASS, `M11.C.5` FAIL, `M11.C.18`, `.20`, `.33` PASS; each passing criterion's mutants turn it red; the mutants of the failing three did not run (`M11.1d`). S3.7 is not met. **Deviations, for the owner's review:** F's own `I-POSITION` withheld in every run of `M11.C.49`; no move on the tie reads a `CONFLICT` share of 0, counted; the open batch's rerun also cuts criteria added since `9ebf327`; the Phase B script's `I-POSITION` now starts a sequence (week 31, `DEFINE` at 35) | 7 | yes |
| 9 | **Group A:** P1, P3, `reciprocal_contact_share`; S3.10, S3.11 green; then only `M11.C.42`, `.45`, `.44`, `.27` (all four cells) and `M11.C.16`, `.41` run, with their mutants (S3.12–S3.14) **Done 2026-10-26:** `appraise.returned_contact`: when a delivery of an act with a positive contact component and a positive contact change lowers its target's two-sided deviation (`M4.C.1`, impingement included, on the pre-batch state), the sender takes `reciprocal_contact_share × Δcontact × content_gain(sender)` as felt contact on the same tie and appraises it by their own deviation, through `M4.B`'s steepness and reattribution; it is not counted in `attended` (`M1.B.8`). `TRIANGLE` set to layer 0 in `config/bowen/event_kinds.md`. S3.10 and S3.11 green, their strict marks removed; two more unit tests (`test_m4c1d_sender_appraises_the_returned_contact_by_their_own_deviation`, `test_m4c1d_relief_reads_the_whole_delivery`). Unit gates for S3.10 and S3.11 added (`review/stage_3_unit_mutation_gate.md`, 18 of 18 proved). Mutant added: Phase C `reciprocal-contact-removed` (`M11.C.42`); `one-sided-deviation` now also names `M11.C.45`. `tools/stage_3_targeted_run.py` now names a criterion's cells by the Phase C record's rule (a name ending in `[` names each cell it begins, a bare name each of its cells), so the `M11.C.27[` mutants and `M11.C.41`'s run on its passing cells; the run was redone with it. `tests/open_batch.py` cuts a mutant's criteria by its own definition in the file the batch found, not by the whole file (`one-sided-deviation`'s new `M11.C.45`). Engine hash b819660f58b9 → 2a3b84e2d89a; golden fingerprints taken again with the module's own `_records` and `_fingerprint`. Targeted run on Mac Mini1 (`review/stage_3_step_9_criteria.md`): `M11.C.42` FAIL (`triangle_reuse` −0.06 ± 0.36, 50 seeds), `M11.C.45` FAIL (`triangle_rate` +0.0013 ± 0.0055), `M11.C.44` FAIL (`outside_inside_ratio` −0.020 ± 0.089; PASS in revision 38's record), `M11.C.27` three cells FAIL and `[unstable,remove_one]` PASS (`pair_deviation` −0.081 ± 0.05) and both its mutants turn it red, `M11.C.16` PASS (`repertoire_entropy` −0.0385 ± 0.0062) with four of its 24 named mutants red, as against four in revision 38's record but not the same four (`steepness-inverted` now red, `availability-level-inverted` now survives), `M11.C.41`'s two lower-level cells PASS and the two heavier-stress cells FAIL, as in the record, and on the passing cells four of nine named mutants turn each red (in the record `steepness-level-independent` also turned `[heavy: lower level]` red; it now survives); the mutants of the failing rows did not run (`M11.1d`). S3.12 and S3.13 are not met (T-142). **Deviations, for the owner's review:** `M4.C.1d`'s clause on a sender near or past the optimum reads "within its band or past it nothing", since the deviation's "too much" side is impingement; the register rows and the explainer's sentence on it are added; `M5.C.1`'s gate does not cover returned contact, kept as built by the owner's choice (T-140) | 8 | yes |
| 10 | The sweep tool's ratio (§3.4); S3.16–S3.19 green. **One full regeneration** on the final hash, in the background; the changes page; the status page (S3.24) **Done 2026-10-26:** the ratio (`tie_wear_rate` ÷ (`slow_tick_fast_ticks` × `calm_transfer_rate`), central 1/52, 1/208 to 4/52 on a log scale) under `config/sweep/sweep.md`'s `## Named ratios`, loaded by `tools/sweep_config.py` (which refuses a swept period constant), drawn in place of `tie_wear_rate` and its samples split at the central value for a criterion whose run, measured on seed 0 of each arm at the central constants and cached under the engine hash, is longer than one slow tick; `M17.E.1`'s sentence; S3.16–S3.19 green, their strict marks removed. MR-23 (T-012): the Phase C ensemble record rendered from `docs/records_cache/phase_c_ensemble.json`. The engine hash did not move. One regeneration on 2a3b84e2d89a: the Phase C records and the Phase D mutation record here, Phase D part II on Mac Mini1 (95 min), part I here (16 min), the Phase D mutation record 93 min. `tools/record_changes.py` gained the revision 39 page (`docs/version_2_revision_39_changes.md`, against `9ebf327`; the revision 36 page frozen at `9ebf327`); the status page gained a revision 38 column, and a sweep record built on an earlier engine is marked and not counted for W4. 6 of the status page's 63 rows differ from revision 38's: `M11.C.2[twelve-person family]`, `M11.C.7[one spouse over sided]`, `M11.C.10[family of origin]` and `M11.C.17` FAIL → PASS, `M11.C.44` PASS → FAIL, and `M11.C.49` new and failing. The open batch is closed: `OPEN` None, the twelve marks off. `test_s2_11_sweep_record_is_current` stays a strict expected failure until step 11 (T-128 closes there) | 9 | no |
| 11 | **The stage-end full sweep** with the ratio, split across both Macs as at stage 2 (S3.25); `M17.E.1` marked *done* **Done 2026-10-27:** a cross-check on `M11.C.16` and `M11.C.23` (both Macs; the merge added 0 and found 452 equal), then Mac Mini1 ran `M11.C.6`, `.10`, `.10[family of origin]`, `.15`, `.48` and `.48[phase_d]` (08:01–19:53, 58 child runs) and this Mac the other 28 (08:03–17:05, 359 child runs, `nice` 15, 6 workers); 272 first runs and 164 reruns at 200 seeds in all, none undetermined or errored. The ratio entered 26 keys, the 8 whose run is 52 fast ticks or fewer not; `test_s2_11` lost its strict mark; W4 met, no fraction marked as an earlier engine's. `M11.C.49` has no declared ranges and was not swept (T-144); it fails, so W4 does not need it, but `M17.E.1` asks for every directional criterion, so the owner kept `M17.E.1` *partial* (2026-10-27) | 10 | no |
| 12 | Reviews before the push: one `learning-qa` pass and one correctness pass over the whole diff; every finding fixed; a narrow re-sweep on `sonnet`; three or fewer minor items go to a TODO record | 10 (11 may run meanwhile) | no |
| 13 | Push; then the stage 3 **milestone review** (`docs/milestone_review_brief.md`) in a session without this build's history; triage after it | 11, 12 | — |

Steps 7–9 each run only their group's criteria (D17, D31), so each change's effect is seen on its own before the next
lands; each step's targeted results are written to `CHANGELOG.md` with the hash they ran on. The order C, F, A puts
group A's P1 last because it reaches every approach act in every run, so it would confound the targeted runs of the
other two if it landed first. A targeted result on an intermediate hash is a step's check, not a record; only step
10's regeneration and step 11's sweep go into the records.

## 3.9 Compute

- **Steps 2 and 3:** the trace and the probe, minutes each (a handful of seeds of one Phase D part II child and one
  Phase C triad).
- **Steps 7–9, targeted runs at 10 workers, per step:** from revision 36's per-row times, Phase C criteria take seconds
  to a minute each, Phase D part I criteria about 30 s each, part II criteria about 11 min each (1 h 31 min for 8
  rows). Step 7 is dominated by `M11.C.6` and `M11.C.2[twelve-person family]`, about 25 min with their mutants' first
  runs; steps 8 and 9 under 15 min each. These are estimates from measured per-row times, labelled as estimates.
- **Step 10:** one full regeneration, 2 h 36 min at revision 36 (§3.6), plus `M11.C.49`'s rows.
- **Step 11:** about 18 h 47 min of machine time at stage 2 (§3.6), about 8 h wall split across both Macs; the ratio
  adds no samples; `M11.C.49` adds one Phase C criterion. Measured again at step 11, not projected further.
  *Measured 2026-10-27:* 11 h 52 min wall on Mac Mini1 (six criteria, 10 workers) and 9 h 3 min here (28, six
  workers under `nice`), after the cross-check; `M11.C.49` was not swept (T-144).

## 3.10 How a changed result is reported

Each criterion whose verdict changes under revision 39 is reported as a new result under a new rule, beside version
1's verdict and revision 38's, never as version 1's finding withdrawn (`docs/version_2_plan.md` §3, stage 3). A
criterion that turns green is a composite passing, with its mutation recorded before it counts (W3). A premise
passing (S3.1, S3.4, S3.6, S3.10, `M11.C.49`'s impingement limb) confirms the code and is never reported as a finding.
`M11.C.27[add_third]` and the calm-state finding stay accepted findings unless they move (D23). The four PASS → FAIL
verdicts of revision 36 that await the owner (T-122) are re-verdicted by step 10 and shown as changed; this stage does
not decide them.

## 3.11 Documents each step updates

- **Step 1:** this plan; `docs/SPEC_REVIEW_CHECKLIST.md`; `docs/REVIEWER_READING_LIST.md`; `docs/theory/_STATUS.md`;
  `docs/implementation_plan_version_2.md`'s status line (pointer); `CHANGELOG.md`; the TODO records of T-014 and
  T-128 (decided, built later). T-130's result in `config/bowen/event_sizes.md` waits for step 6.
- **Step 4:** `docs/spec_revision_39_draft.md`, which holds the text below until each group's code lands with it (Q6); then the spec, `docs/spec_revision_log.md`, `docs/spec_index.jsonl`, `docs/spec_rules/`,
  `config/bowen/stress_response.md` (P2's *displacing* table), `docs/model_explainer.md`, `docs/version_2_plan.md` (stage 3's status), `config/sweep/sweep.md`'s preamble.
- **Steps 6–9:** `config/bowen/constants.md` and `constants_changes.md` (`reciprocal_contact_share`),
  `config/bowen/event_kinds.md` (P3), `config/bowen/criteria.md` (`M11.C.29`, `M11.C.49`), `docs/readouts_phase_c.md`
  (`M11.C.49`'s readouts), `tools/spec_coverage_overrides.json`, `docs/spec_coverage.md`, the golden fixtures if a
  result changes, the TODO records of T-011, T-012, T-127, T-129 and T-130 ticked.
- **Steps 10–11:** every record, `docs/version_2_status.md`, `docs/version_2_sweep_record.md`, the revision 39 changes
  page, `CHANGELOG.md`, `CLAUDE.md`'s test count, `docs/REVIEWER_READING_LIST.md`.

## 3.12 Not code-testable

- **Whether a rule writes the outcome (§19).** S3.15 checks the classes and that the engine reads no readout; whether
  P2 (`M1.A.5f`), F1, `M6.I.4a`, A and `M4.C.1d` are the set layer and not a written pattern is a judgment. Proposed review: the owner
  reads revision 39's four rules, one at a time, against `model_explainer.md` §19 at step 4; the milestone review
  (step 13) reads them under its theory-fidelity lens.
- **The stage 4 corpus checks.** Each is a reading of `_LEDGER.md` entries against requirement text, done at step 4
  and listed in the revision log with what was found; reviewed by the owner with revision 39.
- **The value of `reciprocal_contact_share`.** `[I]`, set before the suite runs (`M10.B.4`); the owner approves the
  value and its stated reasoning at step 4, and the stage-end sweep reports how far each direction survives over its
  range.
- **W5:** owner sign-off on each group's proposal (this plan's approval) and on its result (after step 11).
- **W8:** no output presents the model as speaking about a real family; owner review, as in version 1.

## 3.13 Adjacent issues found while planning, not fixed

- **`Triangle.bound_anxiety` is never written** (`objects.py:299`), though `M1.C.1` says a triangle must hold it; its
  release paths (`event_effects.py:142-147`, `life_course.py:572-577`) release zero. The owner chose P1 and P3 without
  P2 (D22). Proposed: `M1.C.1` marked *partial* in the coverage override with the reason, at step 4.
- **`M11.C.21`'s quoted effect.** The spec quotes +0.15 ± 0.19; the status page shows −0.0906 ± 0.1 at the tag and
  −0.0682 ± 0.089 at revision 38. To reconcile in revision 39's wording.
- **The binary assertion gate** (`iposition.py:120-126`) is a step, which `M5.F.2c` forbids for act effects. A (D20)
  removes it; recorded here so the milestone review can check it went.
- **`M11.C.44` against `M11.C.45`.** With `TRIANGLE` an inside act, more triangles under a spell lower `M11.C.44`'s
  ratio while `M11.C.45` wants more of them. Kept by D23; a changed verdict is reported.
- **W1's nine rows** (`docs/version_2_status.md`: the four PASS → FAIL verdicts of revision 36 in T-122, `M11.C.44`'s
  FAIL → PASS, and `M11.C.11`, `M11.C.15` and `M11.C.48` in both variants) block W1 whatever stage 3 does; they need
  the owner's decisions before version 2's status page can be final.
- **The ratio's derived `tie_wear_rate`** spans one eighth to eight times its central value (§3.4), wider than every
  other row's half-to-double. Within D25's decision; shown here so the owner sees it before the full sweep.
