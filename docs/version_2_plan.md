# Version 2 of the agent model: a plan

**Status: DRAFT 2026-10-18, for owner review.** Nothing in it is decided. No spec revision and no code until the
owner answers §5. Drafted from `TODO.md` ("Revision 32" and the older open sections) and `docs/version_1_status.md`.

Version 1 is complete (2026-10-18, `docs/version_1_definition.md` §2b). By that definition's question 8 the spec is
frozen at v2.0 revision 32, and later revisions start version 2. This plan proposes what version 2 is for, what it
contains, in what order, and how "finished" is checked, in the same form as the version 1 definition.

## 1. Where version 1 left the model

From `docs/version_1_status.md` (generated, current against the records): 36 rows pass, 1 passes by a test, 19 fail
and are accepted as findings, 4 are out of scope.

### 1.1 The 19 accepted findings, grouped by stated cause

The grouping is Claude's reading of each row's disposition, not a recorded decision. Each group is one candidate rule
question for version 2.

| Group | Criteria | Cause, as the dispositions state it |
|---|---|---|
| A. Triangling is not learned | `M11.C.42`, `M11.C.44`, `M11.C.45`, `M11.C.27[stable,add_third]`, `M11.C.27[unstable,add_third]` | A `TRIANGLE` relieves its seeker less than a `DISTANCE`, so it is not reused; it changes anxiety, not contact; there is no calm state near the floor |
| B. One act's consequences wash out | `M11.C.5`, `M11.C.27[stable,remove_one]`, `M11.C.43` | No settled change-back reaction over 60 weeks; removing one steadies for about four weeks, then unsettles; losing a calming member costs no more |
| C. Projection does not push | `M11.C.2`, `M11.C.6`, `M11.C.21` | Projection-type acts open `M7.E.1f`'s band and do not push development down; the child is at its floor in both arms |
| D. The arms barely differ, or the channel is not selected | `M11.C.10`, `M11.C.12`, `M11.C.23`, `M11.C.7[one spouse over sided]` | The held cutoff adds little to an input already high; more anxiety does not select `CONFLICT`; decay outweighs hostility |
| E. Reactive choice reads level, not anxiety | `M11.C.41[higher level: heavier stress]`, `M11.C.41[lower level: heavier stress]`, `M11.C.34` | The mixing weight reads functional level; no selection reads `M1.A.19`'s detectors |
| F. Precondition not met | `M11.C.29` | No genuine `I-POSITION` completes in the arm (0 of 100 seeds) |

Group F connects to an open `TODO.md` item that reaches every `I-POSITION`: since revision 16, 0 of 200 seeds reach the
peak in `M11.C.32`'s calm arm at level 40, so `hold_gain` and `defence_threshold` may be sized for the calmer model.

### 1.2 Loose ends from version 1, still pending owner review

These do not block version 1 (the status page reads V1–V3 met), but each is Claude's recommendation:

1. Four disposition notes on criteria that now pass since the nodal-event spike: `M11.C.7[joint over one spouse]`,
   `M11.C.7[joint over sided]`, `M11.C.17`, `M11.C.22` (`docs/version_1_dispositions.md`).
2. `M11.C.46`'s mutation clause is not met at the edge (`triangle-help-ignored` survives, +0.16 ± 0.23 against +1.45
   unmutated). Accept as for `M11.C.14`, or restate the readout (post hoc).
3. The nodal spike's narrowing: a cut tie is one holding rejection, so a retired coach's or removed member's tie is
   not spiked.

### 1.3 What version 1 has not measured

Every pass in version 1 is at one point in the space of invented constants. The register holds about 187 `[I]`
entries in `config/bowen/constants.md` (a `grep -c` of the tag, so a count of lines, not of distinct constants).
`M11.4f` says a comparison with a mechanism switched off **MUST**, in Phase E, be run across `M17.E.1`'s constant sweep
before its direction is relied on. No criterion has been swept. So version 1's passes say the code does what its rules
say at the declared constants; they do not yet say the direction holds across the range the constants could take.

## 2. What version 2 is for — the recommendation

**Recommended:** version 2 measures how far version 1's results hold, then revisits the rule questions the owner
chooses, in that order. The real-family import and the language layer stay out.

Reasons for this order:
- The sweep is the step the spec already requires (`M11.4f`, `M17.E.1`) before a direction is relied on.
- It sorts the accepted findings: a finding that passes across much of the range is a constant question, not a rule
  question, and should not be answered with a new rule.
- A rule change moves the engine hash and reruns every record (about 1 hour 50 minutes on 10 cores, measured at plan §4.8; §4.10 estimated about 2 hours). A sweep
  run after the rule changes would be run once; run before, it informs which rules to change. The plan runs a
  **pilot** sweep first so the order can be revisited when its cost is measured.
- Rule changes made to turn a failing criterion green are the fit risk `M11.F.9` names. Measuring first gives the
  owner the evidence to choose rules on their theory, not on which criterion they move.

## 3. Stages

Each stage that changes what a run computes follows the project's sequence: proposal → spec revision → implementation
plan → build → review → check off, each approved before the next. Spec IDs for new work are assigned in those
revisions, not here.

### Stage 0 — Freeze version 1 and close its loose ends (no engine change)

- **0.1** Mark the version 1 record so later regenerations cannot change it. Today `tests/test_version_1_status.py`
  fails when `docs/version_1_status.md` differs from what `tools/version_1_status.py` renders from the current
  records, so the first engine change in version 2 turns it red or rewrites it. Proposed: a git tag `version-1` at
  `a3e8169`; `docs/version_1_status.md` kept as committed, with its test changed to check it against the records at the
  tag (or its staleness test retired with a "frozen" status line). Owner question Q8.
- **0.2** The owner's decisions on §1.2's three items, recorded in `docs/version_1_dispositions.md` and `TODO.md`.
- **0.3** Render-only fixes that need no rerun (`docs/records_cache/`): the D2 record prints `M11.C.10`'s two variants
  under one id; and the trace hides lock-in. `src/bowen/render/trace.py` lists lock-in among the mechanisms that run
  every week and are not shown line by line, so in the seed-7 trace week 74 shows Marta +226.4 and nothing of the 222.4
  her lock-in spill moved on to five partners (§3, group G). Proposed: show a lock-in transfer above a declared size,
  or any spill past the turn, as its own row. Regenerates the seed-7 trace and its chart only.

### Stage 1 — One engineering batch that moves the engine hash once

Collected from `TODO.md` so the records regenerate once (owner rule, learnings P38). None is meant to change a result;
any that does is reported as a measured change.
- **1.1** Means over floats taken as `sum(...) / len(...)` in the engine (`reactive.py`, `appraise.py`, `iposition.py`,
  `slow_tick.py`, `life_course.py`, `beliefs.py`, `learner.py`) moved to `math.fsum`; the likely source of the 3.11
  against 3.12 drift.
- **1.2** `canonical_state` includes `state.stressors`.
- **1.3** Rule 9: `SIDE_TAKING` (`external.py`) and `FAMILY_TO_EXTERNAL` (`policy.py`) move to config together; the
  readouts' act sets (`APPROACH`, `PROJECTION_ACTS`, the over/under lists) likewise.
- **1.4** `recompute.py`'s triangle activity treats the coach as an unfused occupant against `M8.5`; decide and align.
- **1.5** Tooling that does not move the hash but affects the regeneration loop: a child failure in `run_in_copy`
  aborts the whole record run; `code_text`'s two latent gaps and its split against `tokenize`; the Phase B mutation
  record cannot be regenerated (gate G2's fragment); record-parsing duplication (about six copies of the regex).
- **1.6** Retire the frozen grid engine, if the owner says so (Q6): `src/engine.py`, its tests and its purity
  violation (`sim_audit.csv`). Version 1 passing its acceptance tests is the condition `CLAUDE.md` set for this.

### Stage 2 — Measurement (Phase E core)

- **2.1** `M17.E.1`'s constant sweep, scoped (Q3): for each criterion, the `[I]` constants its mechanism reads, over
  declared ranges, reporting the fraction of samples in which the asserted direction holds. Ranges are proposed by
  Claude with each one's grade and reviewed by the owner before any run.
- **2.2** A **pilot** first: one passing criterion and one accepted finding, a small sample, timed. Its wall time sets
  the full sweep's cost, which is reported to the owner before the full sweep runs. No cost is estimated here without
  that measurement.
- **2.3** The detection-limit rule (`TODO.md`, revision 32): a fixed seed count when an effect is within a set
  multiple of the precision. It changes `M17.A.1` for every criterion, so it is decided before the sweep.
- **2.4** A generated sweep record and a column for it on the version 2 status page.
- **2.5** Which of `M17`'s other drafted requirements come in (`M17.A.2` time-resolved power, `M17.B.1` principal
  strata, `M17.E.3` invariance intervals, `M17.E.6` an alternative activation regime) is Q3's second part. The held
  method proposals in `TODO.md` ("Held for a later revision — paper material") stay held unless the owner names one.

### Stage 3 — Rule questions the owner chooses

One proposal per chosen group from §1.1, written after the sweep reports. Candidates, with what each would change:
- **Group C, projection:** should projection push a child's development down (`M7.E.1f`, "the acts open room, they
  do not push")? The owner raised this at `M11.C.6`.
- **Group F, `I-POSITION` at level 40:** resize `hold_gain` and `defence_threshold`, or change the rule; then
  `M11.C.29`'s arm redesigned (an act week clear of the spell, a mover able to prepare).
- **Group A, triangle relief:** why a `TRIANGLE` relieves less than a `DISTANCE`, and whether a calm state near the
  floor should exist.
- **Groups B, D, E** as the owner chooses.
- **Mechanisms the spec states and the code lacks:** `M1.E.7e`'s lending of pseudo-self by a sided coach;
  `Family.tolerance` declared and never updated (`M7.D.4`); lock-in that does not make a symptom chronic.
- **`M11.C.9`, sibling position** — a new mechanism whose profiles the spec leaves open; recommended for version 2 by
  the version 1 disposition. The theory work questions whether it earns a place at all (Ch13 omits it).
- **A multi-generation check** that chronic anxiety only ratchets up (`M7.B.1a`); a readout over a longer run, not a
  rule.
- **Learning by level, and asking "what can I do differently?"** — raised by the owner 2026-10-18, not one of §1.1's
  groups: `docs/PROPOSAL — LEARNING BY LEVEL, ASKING WHAT TO DO DIFFERENTLY.md` (draft). It touches groups E and F.
- **Group G, how bound stocks are released** — found in a trace, not from a criterion (the seed-7 Phase C run, the
  `M11.C.1` baseline arm, week 74; `docs/review/phase_c_trace_seed7.md`, `docs/review/phase_c_chart_seed7.html`;
  reported by the session that built the chart, 2026-10-18). In week 73 Ravi chose `REDUCE_CUTOFF` toward Pia; the tie
  held 226.4 of `distance_bound_anxiety` from 73 weeks of `DISTANCE`, and `reduce_cutoff` (`M5.B.3`) gave all of it to
  Marta, the only third party in a triad on that tie. Her active symptom went from 1.06× to 6.47× its threshold, past
  lock-in's turn, and lock-in (`M7.D.2a`, `M7.D.2d`) released her symptom-bound stock (95.3) and spilled her excess
  above usual (222.4) to her five interactive partners in equal shares, the coach among them. Two rule questions:
  - **G1. A tie-bound stock has no bound and no decay.** `distance_bound_anxiety` is only added to (`moves.py:115`) and
    only emptied whole: by `release_bound_distance` on `REDUCE_CUTOFF` (`moves.py:124–127`) and on reconciliation
    (`event_effects.py:100`), when the binder is unavailable (`event_effects.py:137`) and at death
    (`life_course.py:542`). `REDUCE_CUTOFF` releases the whole stock though it lowers rejection only by
    `rejection_relief` × intensity. At the end of week 73 the seed's stocks were 319.2, 226.4, 135.0 and 85.5 on four
    ties. Should a bound stock settle back, be capped, or release in proportion to the cutoff reduced?
  - **G2. Releases split equally, ignoring the tie.** `release_bound` and the lock-in spill divide by the number of
    interactive partners (`symptoms.py:147`, `symptoms.py:198`). The coach, whose tie to Marta has conductance 0.3 and
    bond energy 10 (`config/bowen/family_phase_c.md`), took the same 63.6 as Nadia. The learner's cross-person term
    already weights by bond energy (`M4.D.6e`), and `M1.E.7e` requires the coach's tie to stay thin. Weight by bond
    energy or conductance? Should a family member's symptom release reach an external agent at all?

  Both change what every run with a `DISTANCE` or an active symptom computes, so if the owner takes group G it is a
  candidate to build **before** the stage 2 pilot, so the sweep does not measure a release rule about to change (Q2).

Each changed rule keeps the version 1 result beside the version 2 result in the status page. A criterion that turns
green under a new rule is reported as a new result under a new rule, not as version 1's finding withdrawn.

### Stage 4 — Corpus fidelity, scoped to what stage 3 touches

The 2026-08-28 items in `TODO.md` (requirements sourced from an extraction heading, descriptions firmed up into
mechanisms, trimmed hedges and qualifiers, `KS05.2` on the capacity floor, the source behind `M5.F.2`'s threshold, a cold
sweep) are spec-text corrections. Proposed: before each stage 3 proposal, check the requirements it rests on against
`_LEDGER.md`, and fix what is found in the same revision. The full backlog stays in `TODO.md` unless the owner wants it
done as one revision (Q7).

### Out of version 2 (proposed)

- `M15`'s import of real families, the envelope output, and `M11.C.8` against published rates (Q5). The two export
  changes `M13.3` names as cheapest-first (`M15.A.4` intervals, `M15.C.2` rupture against resolved distance) are on the
  owner's application and can be made at any time without waiting for a version.
- Calibrating any `[I]` constant to data.
- Phase F, the language layer.
- `M11.C.13` (no community in the model) and `M11.C.28` (the sink allocation feeds no behaviour), unless a stage 3 rule
  makes either testable.

## 4. The definition of version 2 (proposed)

Version 2 is finished when all of the following hold. W1–W3 and W6–W8 carry version 1's conditions forward; W4 and W5
are new.

| # | Condition | How it is checked |
|---|---|---|
| W1 | Every `M11.C` criterion, including any added in version 2, has a final status, as V1. | the version 2 status page (generated) and a test that no criterion lacks one |
| W2 | Every premise and check passes, as V2. | the same page, by class |
| W3 | Every passing composite has a recorded mutation, as V3. | the record tests (`test_m111d_*`) |
| W4 | Every passing directional criterion has a sweep result: the fraction of samples in which its direction holds over declared ranges, with a pass below a declared fraction marked **narrow** (the fraction is the owner's, Q4). | a generated sweep record and its column on the status page; a test that every passing row has one |
| W5 | Each finding group the owner chose (Q2) has a proposal the owner decided, and if a rule changed, its result is recorded beside version 1's. | the status page's version 1 column; owner sign-off on each proposal (not code-testable) |
| W6 | The documents agree with the code, as V5. | `tests/test_doc_checklist.py`, `tests/test_spec_consistency.py`, the record-current tests |
| W7 | The suite and CI are green on every supported Python version, as V6. | `.github/workflows/tests.yml` on the last commit |
| W8 | No output presents the model as speaking about a real family (`M11.F.9`), as V7. | owner review of the outputs (not code-testable) |

**Not code-testable:** W5's sign-off and W8, as in version 1. Proposed review: the same one-decision-at-a-time form the
owner used for revisions 24–30 and revision 32.

**Tooling this needs:** `tools/version_1_status.py` generalised to a version 2 page (or a sibling tool) that also reads
the sweep record and shows the version 1 status beside each row. Built in stage 2.

## 5. Questions for the owner

1. **What is version 2 for?** (a) measure how far version 1's results hold, then the rule questions you choose
   (recommended, §2); (b) rule questions only, no sweep; (c) the real-family import (`M15`); (d) something else.
2. **Which finding groups (§1.1) should version 2 revisit?** Recommended: C (projection), F (`I-POSITION` at level 40,
   since it reaches every `I-POSITION`) and A (triangling, five criteria). The rest stay accepted findings. Group G
   (how bound stocks are released, found in the seed-7 trace) is not a finding group; recommended in, and built
   before the stage 2 pilot, since it reaches every run that binds anxiety.
3. **Sweep scope.** Per criterion, only the constants its mechanism reads (recommended), or every `[I]` constant? And
   which of `M17.A.2`, `M17.B.1`, `M17.E.3`, `M17.E.6` come in? Recommended: none until the pilot reports.
4. **The "narrow" fraction for W4.** Recommended: decide after the pilot, from what the measured distribution looks
   like, and record it before the full sweep runs so it is not chosen after seeing the results.
5. **Is the `M15` import in version 2?** Recommended: no; version 3.
6. **Retire the frozen grid engine in version 2?** Recommended: yes, as stage 1.6.
7. **Corpus-fidelity backlog:** scoped to what stage 3 touches (recommended), or one full revision?
8. **Freezing version 1:** a git tag `version-1` at `a3e8169`, pushed, and `docs/version_1_status.md` kept as the
   record (recommended)? A tag push is outward-facing, so it waits for your yes.
9. **The spec file.** Version 2's revisions continue in `docs/bowen_agent_model_spec_v2.md` as revision 33 onward, with
   its status line saying version 2 begins at revision 33 (recommended), or a new file? Note the naming: the file's
   "v2.0" is the agent-model spec's own version, not the model's version 2.
10. **§1.2's three items** — the four disposition notes, `M11.C.46`'s mutation clause, and the nodal spike's narrowing.

## 6. Order of work after approval

| Step | Depends on | Changes the engine hash | Regeneration |
|---|---|---|---|
| Stage 0 (freeze, loose ends, render-only fixes) | Q8, Q10 | no | render only |
| Stage 1 (engineering batch) | Q6 | yes | one full (about 2 hours, 10 cores, plus the mutation records) |
| Group G, if taken (proposal → revision → build) | stage 1, Q2 | yes | one full |
| Stage 2.2 pilot | stage 1, Q3 | no | pilot only; timed and reported before 2.1 |
| Stage 2.1 full sweep | pilot, Q3, Q4 | no | cost from the pilot |
| Stage 3 proposals | stage 2 record, Q2 | per proposal | one full per batch of approved rules |
| Stage 4 checks | inside each stage 3 proposal | — | — |

Stage 1 and stage 2's design (ranges, the sweep record's format) can proceed in parallel; the pilot waits for stage 1
so it does not run on a hash that is about to move.

## 7. Adjacent issues found while drafting, not fixed

- `docs/version_1_dispositions.md`'s status line reads "complete" while four rows are notes "pending owner review"
  (§1.2). Notes carry no status by that file's own rule, so the page is right; the wording may mislead a reader.
- `tests/test_version_1_status.py` will go red at the first engine change in version 2 (stage 0.1 addresses it).
- The constant count in §1.3 is a count of tagged lines, not of distinct constants; the sweep's design step counts them
  properly from the register.
