# Version 2 of the agent model: a plan

**Status: APPROVED 2026-10-19.** Drafted 2026-10-18 from `TODO.md` ("Revision 32" and the older open sections) and
`docs/version_1_status.md`. The owner answered §5's ten questions and approved the plan on 2026-10-19: Claude's
recommendation on every question except Q2, where the owner chose the groups with notes of their own (stage 3). The
owner also decided that group G is built before the stage 2 pilot. Each stage that changes a run still goes proposal
→ spec revision → implementation plan → build, each approved before the next.

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

These do not block version 1 (the status page reads V1–V3 met). Each was Claude's recommendation; **the owner decided
all three on 2026-10-19 (Q10)**: the four notes accepted, `M11.C.46`'s clause accepted as for `M11.C.14`, and the
narrowing kept.

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

**Recommended, and decided by the owner 2026-10-19 (Q1):** version 2 measures how far version 1's results hold, then revisits the rule questions the owner
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

*Done 2026-10-19: 0.1 and 0.2 done, 0.3 moved to stage 1.*


- **0.1** Mark the version 1 record so later regenerations cannot change it. Today `tests/test_version_1_status.py`
  fails when `docs/version_1_status.md` differs from what `tools/version_1_status.py` renders from the current
  records, so the first engine change in version 2 turns it red or rewrites it. Proposed: a git tag `version-1` at
  `a3e8169`; `docs/version_1_status.md` kept as committed, with its test changed to check it against the records at the
  tag (or its staleness test retired with a "frozen" status line). Owner question Q8. **Decided 2026-10-19:** the
  tag `version-1`, pushed at stage 0, and the status page kept as the record. *Done 2026-10-19.* The owner moved the
  tag to `d62cfdc`, the commit that records stage 0.2 (same engine, config and records as `a3e8169`), and chose the
  test's form: `tests/test_version_1_status.py` checks that the tag marks `d62cfdc`, that the committed page and
  `docs/version_1_dispositions.md` are byte-identical to the tag's, and runs the tag's own status tests in a copy of
  the tagged tree. `tools/version_1_status.py` refuses to write, and CI checks out with `fetch-depth: 0` so the tag is
  there.
- **0.2** The owner's decisions on §1.2's three items, recorded in `docs/version_1_dispositions.md` and `TODO.md`.
  *Done 2026-10-19*, with `docs/version_1_status.md` regenerated (render only).
- **0.3** ~~Render-only fixes that need no rerun (`docs/records_cache/`)~~ *Moved to stage 1 (1.7), owner 2026-10-19.
  Neither is render-only. The D2 record's renderer is `tools/phase_d_record.py`, which is in the Phase D records' own
  `code_hash`, and those records are not cached, so changing it reruns parts I and II. The trace renderer is
  `src/bowen/render/trace.py`, and `code_hash` covers every `.py` file under `src/bowen`, so changing it moves the
  engine hash and every record reruns.* The two fixes: the D2 record prints `M11.C.10`'s two variants
  under one id; and the trace hides lock-in. `src/bowen/render/trace.py` lists lock-in among the mechanisms that run
  every week and are not shown line by line, so in the seed-7 trace week 74 shows Marta +226.4 and nothing of the 222.4
  her lock-in spill moved on to five partners (§3, group G). Proposed: show a lock-in transfer above a declared size,
  or any spill past the turn, as its own row. ~~Regenerates the seed-7 trace and its chart only.~~ *Regenerates with
  stage 1's run.*

### Stage 1 — One engineering batch that moves the engine hash once

*Implementation plan approved 2026-10-19 (`docs/implementation_plan_version_2.md` §1). Not yet built. Checking the items against the code found that 1.6 needs `M13.2` amended, that 1.1's drift comes from every
float `sum()` and not only means, and that 1.4 changes results in coached arms. The owner decided the plan's D1–D7
the same day, all as recommended: revision 33 carries `M13.2` and `M8.5`; every float `sum()` moves to `math.fsum`;
the code's triangle activity is kept and `M8.5` amended to match.*

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
- **1.6** Retire the frozen grid engine (Q6; **decided 2026-10-19: yes**): `src/engine.py`, its tests and its purity
  violation (`sim_audit.csv`). Version 1 passing its acceptance tests is the condition `CLAUDE.md` set for this.
- **1.7** Stage 0.3's two fixes (moved here, owner 2026-10-19): the lock-in transfer as its own trace row, and
  `M11.C.10`'s two variants under their own keys in the D2 record. Both regenerate with this stage's one run.
- **1.8** Render fixes made cheap (owner 2026-10-19): take `src/bowen/render/` out of `code_hash`, and cache the Phase D
  ensemble results in `docs/records_cache/` as the Phase C records' are, so a later change to the trace renderer or a
  record renderer re-renders in seconds. No verdict reads the trace renderer: only `src/bowen/run.py`'s `--trace` and
  the trace tools import it. Changing the hash's definition makes every record stale, so it belongs in this stage's run.

### Stage 2 — Measurement (Phase E core)

- **2.1** `M17.E.1`'s constant sweep, scoped (Q3; **decided 2026-10-19**): for each criterion, the `[I]` constants
  its mechanism reads, over declared ranges, reporting the fraction of samples in which the asserted direction holds. Ranges are proposed by
  Claude with each one's grade and reviewed by the owner before any run.
- **2.2** A **pilot** first: one passing criterion and one accepted finding, a small sample, timed. Its wall time sets
  the full sweep's cost, which is reported to the owner before the full sweep runs. No cost is estimated here without
  that measurement.
- **2.3** The detection-limit rule (`TODO.md`, revision 32): a fixed seed count when an effect is within a set
  multiple of the precision. It changes `M17.A.1` for every criterion, so it is decided before the sweep.
- **2.4** A generated sweep record and a column for it on the version 2 status page.
- **2.5** Which of `M17`'s other drafted requirements come in (`M17.A.2` time-resolved power, `M17.B.1` principal
  strata, `M17.E.3` invariance intervals, `M17.E.6` an alternative activation regime) is Q3's second part:
  **decided 2026-10-19, none until the pilot reports.** The held
  method proposals in `TODO.md` ("Held for a later revision — paper material") stay held unless the owner names one.

### Stage 3 — Rule questions the owner chooses

One proposal per chosen group from §1.1, written after the sweep reports.

**Chosen by the owner 2026-10-19 (Q2): groups C, F and A.** B, D and E stay accepted findings. Group G is in as
well, by the owner's decisions on G1 and the event sizes, and is built before the stage 2 pilot. The owner's notes,
which each proposal starts from, in the owner's words:
- **C:** "project DOES impact development, that's the problem."
- **F:** "the lower the "level" the more an I position comes out as conflictual demand. The lower the LEVEL the more
  anxiety it creates."
- **A:** "Triangling - seek out other, is a natural human behavior."

Each proposal states which mechanism produces the owner's direction and checks that no rule writes the outcome
(`model_explainer.md` §19). Candidates, with what each would change:
- **Group C, projection:** should projection push a child's development down (`M7.E.1f`, "the acts open room, they
  do not push")? The owner raised this at `M11.C.6`.
- **Group F, `I-POSITION` at level 40:** resize `hold_gain` and `defence_threshold`, or change the rule; then
  `M11.C.29`'s arm redesigned (an act week clear of the spell, a mover able to prepare).
- **Group A, triangle relief:** why a `TRIANGLE` relieves less than a `DISTANCE`, and whether a calm state near the
  floor should exist.
- **Groups B, D, E:** not chosen (Q2); they stay accepted findings.
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
    `rejection_relief` × intensity. At the end of week 73 the seed's four largest stocks were 319.2, 226.4, 135.0 and
    85.5 (seven ties held more than 1). Should a bound stock settle back, be capped, or release in proportion to the cutoff reduced?

    **The owner's answer, 2026-10-18** (given in the chart session, verbatim): "Cutoff doesn't grow forever, it would
    reach a thresshold and become chronic. I would rank it around that of a divorce, which like a divorce disappates
    over time, but not to zero, down to 10% of peak after 5 years. Bridging the cutoff could be negative or positive.
    Negative, takes it back to what it was at the peak. positive, takes 80% of the remaining anxiety away."

    Reading, to confirm with the owner before a proposal is written:
    - **Growth stops at a peak.** The stock rises to a threshold and then becomes chronic, with no further accumulation.
    - **Size, "around a divorce".** Divorce is not modelled (`config/bowen/nodal_events.md`), so the peak needs an
      anchor: a declared `[I]` cap in anxiety points, or the nearest modelled loss, a death (`M6.3a`). Death's grief is
      a stressor intensity (`grief_scale` 20 at bond energy 100, `M6.3g`), not an anxiety stock, so it is not the same
      unit and cannot be copied across directly. `docs/PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md` (draft, 2026-10-18) gives an anchor:
      divorce is 73 life change units, about 504 intensity units at its recommended 6.9 per unit, roughly 13 anxiety
      points delivered at functional level 40. A spouse's death through grief delivers about 17 points at level 40 over
      two years, against the 226 points week 74 released onto one person. The peak should be stated in one unit (anxiety
      points, the stock's unit); converting from the scale goes through the delivered-load step that proposal sets out.
    - **Decay to 10% of peak after 5 years (260 weeks).** If exponential, the weekly factor is 0.1^(1/260) ≈ 0.99118,
      about 0.88% a week, a half-life of about 78 weeks. "Not to zero" is read as a floor at 10% of peak; it could
      instead be a point on a decay that continues.
    - **Bridging** (`REDUCE_CUTOFF`, and presumably reconciliation): negative returns the stock to its peak; positive
      removes 80% of what remains.

    **The owner's answers to the four questions, 2026-10-18** (verbatim): "1) Negative - the content event is negative
    - in the real world it would an argument / rejection / anger/ fighting. Positive - kinder words spoke, request to
    resume contact, agreement to maintain contact. 2) Anxiety can just dissapate, as when a threat is removed. It
    comes from the relationship tie not others, since the cutoff and interaction would be "felt" as a kind of threat.
    3. PRinciple of 2) applies to any tie - positive outcome - anxiety dissapates, 4) only anxiety related to the
    cutoff."

    Reading:
    - **Negative or positive is the content of the contact across the tie.** Negative: argument, rejection, anger,
      fighting. Positive: kinder words, a request to resume contact, an agreement to keep it. This is decided by what
      the people do, so it does not write the outcome (§19). Proposed mapping, Claude's, to confirm: negative is a
      `CONFLICT` or `CUTOFF` across the tie, or a `REDUCE_CUTOFF` the other refuses (not checked: today the act always lowers both rejections, so whether it can be refused under `M1.B.3c` is for the proposal to establish); positive is a `REDUCE_CUTOFF` the
      other accepts, a `STAY-IN-CONTACT`, or a reconciliation event (`M4.A.3`).
    - **Removed anxiety dissipates; it goes to no one.** The 80% a positive bridge takes away leaves the model, as when
      a threat is removed. A return to peak is felt from the tie itself, as threat, not taken from other people. Both
      are a source and a sink of the tie, so `M6.4`'s table gains them by name, as it has decay.
    - **Any tie.** A positive outcome dissipates bound anxiety on any tie, not only a cut-off one.
    - **Only the anxiety related to the cutoff** is governed by this rule: the tie's bound stock, not the rejection
      people hold (which keeps its own fade, `rejection_fade`).

    Consequence, and a conflict to decide: today `REDUCE_CUTOFF` hands the whole stock to the triangle's third party,
    which is what produced week 74's 226 points on Marta. Under these answers that transfer goes. But `M5.B.3` says the
    transient cost of bridging **MUST** land on a third party whose distancing is being stripped, sourced from the
    corpus (`L22.6`, a `[T]` finding; owner decision 2026-10-07, P9). The answers and `L22.6` can both hold if the
    third party's cost is the loss of their distancing outlet, not the stock: the bridged tie's stock dissipates, and
    the third party, who can no longer bind anxiety in that distance, carries more of their own. Recommended, Claude's:
    keep `M5.B.3`'s third-party cost in that form, where it emerges from the lost outlet, and drop the transfer of the
    stock. **Decided by the owner, 2026-10-18:** "Yes: keep M5.B.3's third-party cost as the loss of the third party's
    distancing outlet, and drop the transfer of the bound stock." The proposal restates `M5.B.3`'s last sentence to
    this form, still citing `L22.6`, and a test shows a third party's own anxiety rising when the outlet is lost
    without any of the stock reaching them.

    **Decided by the owner, 2026-10-18:** the act mapping above, "yes to Act Mapping"; and the peak cap and the
    five-year decay apply to the cut-off tie only, "I was referring to the cutoff tie". A cut-off tie is one where
    either side holds rejection (`M1.B.3c`), the same test the nodal spike uses. On other ties a positive contact still
    dissipates bound anxiety (answer 3), with no cap and no decay.

    **The owner's answers to the last three, 2026-10-18** (verbatim): "1. it stays there. A close family member cutoff
    has a residual effect. 2. on a non cutoff tie, it is just a tie, so current processes apply. a) + or - contact
    could be a triangle effort. b) or could be related to creating Distance, Conflict, but if the tie is already
    "distant" or "conflict" it moves 33% + or - base on the idea that repeated contacts are need to repair. At some
    point neg contacts push distance or conflict to cutoff. 3. Rejection anxiety might fade to zero, but a positive
    contact is needed to repair the tie. "stored anxiety" would keep it's value"

    Reading:
    - **The 10% is a floor.** A cut-off tie's stock decays to 10% of its peak by year 5 and stays there: a cutoff from
      a close family member leaves a residual effect.
    - **A tie that is not cut off is just a tie; current processes apply.** G1's peak, decay and 80% removal do not
      reach it. A contact on it may be a triangling effort (the current `TRIANGLE` mechanics), or may create distance
      or conflict (the current `DISTANCE` and `CONFLICT`).
    - **A distant or conflicted tie moves a third per contact.** On a tie already `DISTANT` or `OPEN_CONFLICT`, each
      positive contact moves it 33% of the way back toward ordinary and each negative contact 33% further, because
      repair takes repeated contacts. Enough negative contacts take it to cutoff.
    - **Repair takes a positive contact.** A tie stays cut off after rejection fades to zero, until a positive contact
      repairs it. Its stored anxiety keeps its value meanwhile.

    What this changes in the code, found while recording it:
    - `lower_rejection` (`src/bowen/engine/rejection.py`) returns a tie to its state before the cut as soon as neither
      side holds rejection. Under answer 3 it stays `CUT_OFF` until a positive contact, so "cut off" can no longer be
      read as "holds rejection". The nodal-event spike's narrowing (§1.2 item 3) uses that same test and would follow.
    - `OPEN_CONFLICT` is declared (`objects.py`) and never entered by the engine. `DISTANT` is entered by a run of
      `DISTANCE` acts (`consolidate.py`, hardening) and never left except by reconciliation. Answer 2 needs a way into
      `OPEN_CONFLICT` (proposed, Claude's: a run of `CONFLICT` acts, as `DISTANT` has) and a way back from both.

    Questions for the proposal:
    - **What the 33% measures.** Proposed, Claude's: a per-tie repair measure between ordinary and cutoff, where three
      positive contacts take a distant or conflicted tie back to ordinary and three negative contacts reach cutoff.
    - **How negative contacts reach cutoff without writing it** (§19: cutoff is emergent). Today a tie is cut off when
      a person holds any rejection at all (`Relationship.rejected()`: any value above 0, `M1.B.3c`), which people choose
      or learn. Proposed, Claude's: a tie becomes cut off when a person's rejection reaches
      `rejection_refusal_threshold` (0.3, where moves start to be refused), not at any rejection; a negative contact on
      a distant or conflicted tie raises the receiver's rejection by a third of that threshold. Cutoff then follows from
      rejection the person holds, not from a rule that sets the tie's state. This changes `M1.B.3c`'s definition of a
      cut-off tie, as answer 3 does. The owner to say whether that is the meaning.

    **The owner's answer, 2026-10-18** (verbatim): "A cutoff is a extreme form of distance. it takes >15 rejections or
    10 conflicts in a row to create that level of "distance". Perhaps you need to related to level of anxiety induced
    because cutoff is in response to high levels of anxiety and the person wanting relief. The pattern is this - the
    relationship creates a lot of anxiety, which grows to the point, where cutting off lowers anxiety vs. the
    relationship growing it. The cutoff brings some relief, but not 100%, because the person doesn't want to cutoff,
    they are forced to by the anxiety."

    This replaces the two proposals just above. **Withdrawn:** the cut-off line at `rejection_refusal_threshold`, and a
    negative contact adding a third of it. Revised reading and recommendation, Claude's, to confirm:
    - **Cutoff stays a choice driven by anxiety, not a count.** A person cuts off when the anxiety the tie generates
      has grown until cutting off relieves more than staying. That is how the model is meant to produce cutoff already:
      `CUTOFF` is an act selected and learned from felt relief (§19), and `M4.C.1c` makes its relief partial, since it
      removes the "too much" side at once while the "too little" side builds over time. So no new rule is needed for
      the pattern itself; negative contacts across the tie feed it by raising the anxiety the tie generates.
    - **The owner's figures become a check, not a rule.** A readout counts, on each tie, the negative contacts in a
      row before its first cutoff: rebuffs (read as `DISTANCE` toward the person, to confirm) and `CONFLICT`s. It is
      reported against the owner's reference (more than 15 rebuffs or 10 conflicts in a row). A rule that cut a tie
      off at the 10th conflict would write the pattern. Whether the readout gates, as a composite criterion asserting
      that cut-off ties had longer negative runs than ties not cut off, is the owner's.
    - **The model is likely far from the reference now.** `TODO.md` records ties cut within weeks (`M11.C.10`'s uncut
      baseline) and the couple's tie in the Phase C family held cut off in 82–89% of last-year weeks. If the readout
      confirms that cutoff comes after a few contacts rather than ten or more, the cause is how fast a tie's anxiety
      grows against how much a cutoff relieves, which is the stage 2 sweep's question (`rejection_cutoff_gain`,
      `M4.C.1`'s deviation function), not a new rule.
    - **Still to confirm:** that repair stays at three positive contacts for a distant or conflicted tie (answer 2b's
      33%), with the negative side now governed by anxiety as above; and that the readout is wanted before G1 is built,
      so G1's effect on it can be seen.

    **Decided by the owner, 2026-10-18:** "Make it a composite criterion: cut-off ties had longer runs of negative
    contacts before cutoff than ties not cut off." A new `M11.C` criterion in version 2's first spec revision (its ID
    is assigned there). Composite, because cutoff is emergent (§19). Design points for the proposal, Claude's:
    - **Its form differs from most criteria.** It compares two groups of ties within the same runs, not two arms, as
      `M11.C.2` compares children. It asserts the direction only; the owner's 15 and 10 are reported beside it as the
      owner's reference, graded `[I]`, and do not gate.
    - **The comparison must not be biased by its own selection.** A cut-off tie's run is counted up to the cutoff; a
      tie never cut off would otherwise be counted over the whole run and have more weeks to build a long run.
      Proposed: for each cutoff, compare the run before it against the longest run on the same seed's uncut ties over
      the same weeks, paired per seed. Ties with no contact in the window are reported, not dropped.
    - **What counts as negative.** `CONFLICT` and rebuffs received across the tie; rebuffs read as `DISTANCE` toward
      the person, confirmed by the owner 2026-10-18 ("yes"). A refused move counts too once the tie holds rejection, but by then it is cut off.
    - **Its mutation.** A mutant in which cutoff no longer depends on the anxiety the tie has built (the "too much" side
      of `M4.C.1` read as zero when choosing `CUTOFF`, or a fixed `CUTOFF` propensity) should turn it red. If no mutant
      bites, the criterion is not counted as covered (W3).
    - **Where it is expected to stand.** From the `TODO.md` evidence above, cutoff now comes quickly, so the criterion
      may fail at version 1's code. That would be a finding to report, measured before G1 is built and again after.

    **Confirmed by the owner, 2026-10-18:** repair of a distant or conflicted tie takes three positive contacts, a
    third each (answer 2b), with the negative side governed by the anxiety the tie generates; and rebuffs are read as
    `DISTANCE` toward the person. G1's questions are answered except the readout's timing (before G1 is built, so its
    effect can be seen; Claude's recommendation). **Drafted 2026-10-19:** `docs/PROPOSAL — CUTOFF ANXIETY, BRIDGING AND
    REPAIR.md`; its Q1–Q11 all decided by the owner 2026-10-19. By its Q9 it is built in one batch with G2 and the
    event sizes.
  - **G2. Releases split equally, ignoring the tie.** `release_bound` and the lock-in spill divide by the number of
    interactive partners (`symptoms.py:147`, `symptoms.py:198`). The coach, whose tie to Marta has conductance 0.3 and
    bond energy 10 (`config/bowen/family_phase_c.md`), took the same 63.6 as Nadia. The learner's cross-person term
    already weights by bond energy (`M4.D.6e`), and `M1.E.7e` requires the coach's tie to stay thin. Weight by bond
    energy or conductance? Should a family member's symptom release reach an external agent at all?

- **Event sizes from the Holmes–Rahe scale** — `docs/PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md` (draft, the owner's request 2026-10-18): base
  sizes for exogenous events from the scale's life change units, scaled by bond energy. The scale is an outside rating,
  graded `[I]`, not Bowen. Its other findings: marriage, birth, retirement and a child leaving home add no anxiety
  today; job loss and a family member's illness are light against the scale; lost income after an earner's death is
  heavy and has no end; a person's own symptom onset loads only their partners. Its Q7 recommends building it with
  G1, since G1's peak is sized against divorce, so the two share one regeneration. **Decided by the owner
  2026-10-19** (its Q1–Q8): the scale sets an ordering of base sizes at 6.9 per LCU; additions carry load for 12
  weeks; a person's own illness loads them; the mapping as drafted; lost income stays outside the scale; built in one
  batch with G1 and G2.

  Group G and the event sizes change what every run with a `DISTANCE`, an active symptom or an exogenous event computes, so it is built
  **before** the stage 2 pilot, so the sweep does not measure a release rule about to change (decided by the owner
  2026-10-19).

Each changed rule keeps the version 1 result beside the version 2 result in the status page. A criterion that turns
green under a new rule is reported as a new result under a new rule, not as version 1's finding withdrawn.

### Stage 4 — Corpus fidelity, scoped to what stage 3 touches

The 2026-08-28 items in `TODO.md` (requirements sourced from an extraction heading, descriptions firmed up into
mechanisms, trimmed hedges and qualifiers, `KS05.2` on the capacity floor, the source behind `M5.F.2`'s threshold, a cold
sweep) are spec-text corrections. Proposed: before each stage 3 proposal, check the requirements it rests on against
`_LEDGER.md`, and fix what is found in the same revision. The full backlog stays in `TODO.md` unless the owner wants it
done as one revision (Q7). **Decided 2026-10-19: scoped to what stage 3 touches.**

### Out of version 2 (approved 2026-10-19)

- `M15`'s import of real families, the envelope output, and `M11.C.8` against published rates (Q5; **decided 2026-10-19: version 3**). The two export
  changes `M13.3` names as cheapest-first (`M15.A.4` intervals, `M15.C.2` rupture against resolved distance) are on the
  owner's application and can be made at any time without waiting for a version.
- Calibrating any `[I]` constant to data. The Holmes–Rahe proposal is close to this: its Q1 asks whether the scale may
  set an ordering of base sizes, graded `[I]`, without counting as calibration. If the owner says yes, that is the one
  exception, and it is stated in this bullet. **The owner said yes, 2026-10-19.** The one exception: the Holmes–Rahe
  scale sets the ordering of exogenous event base sizes, at 6.9 intensity units per LCU, every size graded `[I]` and
  never cited as Bowen. No other constant is fitted to outside data in version 2.
- Phase F, the language layer.
- `M11.C.13` (no community in the model) and `M11.C.28` (the sink allocation feeds no behaviour), unless a stage 3 rule
  makes either testable.

## 4. The definition of version 2 (approved 2026-10-19)

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

All ten answered by the owner on 2026-10-19.

1. **What is version 2 for?** (a) measure how far version 1's results hold, then the rule questions you choose
   (recommended, §2); (b) rule questions only, no sweep; (c) the real-family import (`M15`); (d) something else.
   **Answered: (a).**
2. **Which finding groups (§1.1) should version 2 revisit?** Recommended: C (projection), F (`I-POSITION` at level 40,
   since it reaches every `I-POSITION`) and A (triangling, five criteria). The rest stay accepted findings. Group G
   (how bound stocks are released, found in the seed-7 trace) is not a finding group; recommended in, and built
   before the stage 2 pilot, since it reaches every run that binds anxiety. **Answered: C, F and A, with the owner's
   notes quoted in stage 3.** B, D and E stay accepted findings. Group G is built before the stage 2 pilot (asked
   separately the same day).
3. **Sweep scope.** Per criterion, only the constants its mechanism reads (recommended), or every `[I]` constant? And
   which of `M17.A.2`, `M17.B.1`, `M17.E.3`, `M17.E.6` come in? Recommended: none until the pilot reports.
   **Answered: per criterion; none of the four until the pilot reports.**
4. **The "narrow" fraction for W4.** Recommended: decide after the pilot, from what the measured distribution looks
   like, and record it before the full sweep runs so it is not chosen after seeing the results. **Answered: after the
   pilot, recorded before the full sweep.**
5. **Is the `M15` import in version 2?** Recommended: no; version 3. **Answered: no; version 3.**
6. **Retire the frozen grid engine in version 2?** Recommended: yes, as stage 1.6. **Answered: yes, as stage 1.6.**
7. **Corpus-fidelity backlog:** scoped to what stage 3 touches (recommended), or one full revision? **Answered:
   scoped.**
8. **Freezing version 1:** a git tag `version-1` at `a3e8169`, pushed, and `docs/version_1_status.md` kept as the
   record (recommended)? A tag push is outward-facing, so it waits for your yes. **Answered: yes, the tag pushed at
   stage 0.** The commit was then moved to `d62cfdc` (§7; stage 0.1).
9. **The spec file.** Version 2's revisions continue in `docs/bowen_agent_model_spec_v2.md` as revision 33 onward, with
   its status line saying version 2 begins at revision 33 (recommended), or a new file? Note the naming: the file's
   "v2.0" is the agent-model spec's own version, not the model's version 2. **Answered: the same file, revision 33
   onward**; its status line changes at revision 33.
10. **§1.2's three items** — the four disposition notes, `M11.C.46`'s mutation clause, and the nodal spike's narrowing.
    **Answered: the four notes accepted; `M11.C.46`'s clause accepted as for `M11.C.14`; the narrowing kept.**

## 6. Order of work after approval

| Step | Depends on | Changes the engine hash | Regeneration |
|---|---|---|---|
| Stage 0 (freeze, loose ends; 0.3 moved to stage 1) | Q8, Q10 | no | render only |
| Stage 1 (engineering batch) | Q6 | yes | one full (about 2 hours, 10 cores, plus the mutation records) |
| Group G and the event sizes, taken (owner 2026-10-19; proposals → one revision → build) | stage 1, Q2, G2's proposal (G1's and the event-size proposal's questions are all decided) | yes | one full, shared |
| Stage 2.2 pilot | stage 1, group G built (owner 2026-10-19), Q3 | no | pilot only; timed and reported before 2.1 |
| Stage 2.1 full sweep | pilot, Q3, Q4 | no | cost from the pilot |
| Stage 3 proposals | stage 2 record, Q2 | per proposal | one full per batch of approved rules |
| Stage 4 checks | inside each stage 3 proposal | — | — |

Stage 1 and stage 2's design (ranges, the sweep record's format) can proceed in parallel; the pilot waits for stage 1
so it does not run on a hash that is about to move.

## 7. Adjacent issues found while drafting, not fixed

- ~~`docs/version_1_dispositions.md`'s status line reads "complete" while four rows are notes "pending owner review"
  (§1.2).~~ *Resolved 2026-10-19: the owner decided the four notes, and the status line says so.*
- ~~**The tag's commit (found 2026-10-19, after Q8 was answered).**~~ *Resolved 2026-10-19: the owner chose `d62cfdc`
  and a test that renders at the tag (stage 0.1).* `a3e8169` precedes stage 0.2, so at that commit
  the status page still shows the four notes as pending and has no `M11.C.46` note. If stage 0.1's test checks the
  committed page against what the tool renders at the tag, the two will differ. Either the tag marks the commit that
  records stage 0.2, or the test compares against the page as it stands at the tag. Stage 0.1 asks the owner which
  before the tag is pushed.
- ~~`tests/test_version_1_status.py` will go red at the first engine change in version 2 (stage 0.1 addresses it).~~
  *Resolved 2026-10-19 by stage 0.1: the test now reads the tag, not the working tree.*
- The constant count in §1.3 is a count of tagged lines, not of distinct constants; the sweep's design step counts them
  properly from the register.
