# Implementation plan — Phase D

> **Status: approved by the owner, 2026-10-10 ("Proceed"), with the decisions in §1a.** P9 decided 2026-10-11; P13–P16 answered 2026-10-13 (§1b), with two P15 points still open. **Part II planned 2026-10-13 (§4) and approved by the owner the same day ("approved"); L0–L9 built 2026-10-13; L10's criteria built and run: `M11.C.37` and, after the owner's decision on its arms, `M11.C.15` pass; `M11.C.6`, `.10` and `.2` fail (§4.7). Spec revision 29 and its plan (§4.8) drafted 2026-10-13 after the owner's decisions on `.10` and `.2`, approved by the owner ("i approve proceed") and built 2026-10-14: `M11.C.2` on the twelve-person family now passes; `.10` and `.6` still fail (§4.8). §4.9 diagnoses `.10`; revision 30's plan (§4.10) approved and built 2026-10-14: the family-of-origin variant of `.10` passes, `.10` still fails, and `.2` on the twelve-person family fails again.** **Built:** B1 and B2 (2026-10-10/11); B3 and B4 (2026-10-11, spec revision 16); B5 and B6 (2026-10-11); B7, B8 and B9 and part of step 10 (2026-10-12, spec revisions 24–27, Claude's recommendation pending owner review: the owner asked for the work to proceed overnight without waiting for input).
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
| **P17** | **`t0` and the first slow tick.** `M2.A.0b` calls `t0` the first slow tick; today it fires at the end of week 51. **D-I** | the slow-tick step | *Decided: week 0 is the start, not a step (§1a)* |

## 1a. Owner decisions, 2026-10-10

- **P1: yes.** Cutoff is rejection; the tie stays interactive.
- **P2: agreed.** The "too little" side grows while a tie is cut, and approaching relieves it.
- **P3: agreed.** A gradual flip, crossing at 0.5 `[I]`.
- **P4: okay.** The estimator as recommended.
- **P5, the owner's words:** "chronic anxiety stays fixed unless level of differentiation goes up, thus it's fixed for
  some time. It wears on ties, which is why the other emotional patterns act up from time to time, often triggered by
  other stressors."
  - Chronic anxiety is fixed and changes only when `basic_level` rises: it falls with a rise, never drifts on its own,
    and does not rise when `basic_level` falls.
  - **It wears on ties.** `M7.E.4`'s slow-clock deterioration is driven by chronic anxiety.
  - Patterns flare when a stressor meets a worn tie. That needs no new rule: a worn tie has less slack, so a spell
    pushes it past where the patterns are relieving.
  - This also settles **P6**: conductance wears down on the slow tick at a rate set by the two members' chronic anxiety,
    offset by contact. Bond energy stays as `M1.B.4` has it.
- **P7: agreed.**
  - I draft definitions of C.14's "technique", C.23's EE and suppression terms, and C.28's mobility term, for approval
    before those criteria are built.
  - C.13 leaves Phase D.
- **P9: decided 2026-10-11.** Yes, with the one-spouse arm. The owner: "if both are working to improve, couples go
  faster. But a principle of systems applies — change 1 part and the system will shift, so it is effective to work with
  1 person. In both cases the coach must not take sides." C.7's direction: joint > one spouse > sided (spec revision 17).
- **P17: week 0 is the starting point, not a step.** The first slow tick is at the end of week 51, as the code has it.
  `M2.A.0b`'s "`t0`, the first slow tick" is corrected in revision 14: `t0` is the start.
- **Read as approved by "Proceed", unless the owner says otherwise:**
  - R0's split into D-I and D-II;
  - P8 (C.14 and C.17 added to `M11.4` with declared margins);
  - P10 (C.9 kept as the three-arm comparison);
  - P11 (C.42, C.45 and C.16 not rebuilt in Phase D);
  - P12 (I draft the triangle criterion).

## 1b. Owner decisions for part II, 2026-10-13 (P13–P16)

The owner's words, then Claude's reading of them. The readings are for the spec revision that applies them, and the
open points are the owner's.

- **P13, death (`M6.3`).** "Every family member has some 'tie' with anxiety bound up to the dead person. The dead
  person has some level of chronic anxiety. Like chemical bond energy — it gets redistributed. A death is a stressor
  on individuals as well … a major life stressor. If that person was an income source that source is now gone and
  that's a stressor."
  1. The dead person's chronic anxiety is redistributed to the living, in proportion to their ties to the dead person.
  2. The death itself is a stressor on each person, in proportion to their tie (grief). It dissipates over one to
     three years.
  3. If the dead person was a source of income, its loss is a financial stressor. "Finances have to get resolved
     somehow." **Answered 2026-10-13:** "family members have to find a job. They have to figure it out for
     themselves. Trying equals finding, but at 20% less income." **Reading:** no rule restores the income. A living
     member of the dead person's household can take an act of looking for work; the act always succeeds, and the
     household's income from then on is 80% of what was lost. The remaining 20% stays a smaller financial stressor.
     Who looks, and when, is learned like any other act (`M0.4`): it is not scheduled.
  4. Everyone's stress going up stresses the system reciprocally: no separate rule, the raised anxiety reaches the
     others through the ties as it does now.
- **P14, births, partners, life stage.**
  1. A child's basic level starts at age 5 and develops each year, levelling off at age 20, ending within ±2% of the
     parents' level depending on circumstances and on how the parents are functioning: the parents functioning up
     raises it, functioning down lowers it. **Reading:** from 5 to 20 the child's level moves within ±2% of the
     parents' mean basic level, pushed by the parents' functioning and the child's own; spec revision 28 writes this as
     a rule in place of the estimator (`M7.E.1f`, Claude's recommendation). **Answered 2026-10-13:** "Level x .02, not two more points": the band is ±2% of the parents'
     level (about ±0.8 points at 40).
  2. Partners: provide "settings" for partners: a partner is drawn from declared settings in config, not invented by
     the engine. **Reading (no objection 2026-10-13):** a config table of the people who can enter as partners.
  3. A person starts acting at 10 and stops at death.
- **P15, family data.** Dr Halim is female, age 35, retires at 68, and a new coach then starts over.
  - **The coach is tied to the coachee only (answered 2026-10-13).** This matches `M2.B.2` ("whichever member is
    being coached"); the config's tie to both spouses is wrong and changes.
  - **Starting ages (answered 2026-10-13):** "1 for 3rd generation's youngest child, 30 for second, 55 for third
    [the grandparents]. Allow these as settings in app." **Reading:** three settings, one per generation: the youngest
    third-generation child starts at 1, the second generation at 30, the first at 55. Each generation's other members
    keep their current age gaps to that anchor; for the second and first generations the anchor is the generation's
    eldest. At the current gaps that gives Pia 1, Nadia 4, Leo 9; Ravi 30, Marta 28, Dario 27, Iris 25; Teodor 55,
    Ana 52, Viktor 52, Sofia 50. "Settings in app": the three anchors go in the family config; showing them in the
    run visualization is a separate change (this repo does not touch the fam-diag app).
  - **Still open:** Teodor's basic level 36 (Claude's recommendation stands: 34 broke `M2.A.0e`'s pairing with Ana);
    and whether Marta–Iris is "resolved, low contact" or ordinary.
- **P16:** agreed. Bowen's two falsifiers and the three chapter items become criteria in a later revision.

### P9 in detail: what C.7 tests, and the direction proposed

- **The claim.** C.7's claim is "position, not skill". Whether coaching helps depends on where the coach stands in the
  family's emotional process, not on how good the coach is.
- **The test.** Hold every one of Dr Halim's parameters fixed and change only the topology, who talks to whom in whose
  presence.
- **The predicate.** `M8.2` and `M8.3` supply it. A coach in the *outside position* (in contact with both spouses, taking
  neither side) counts as a live position. A coach who *has taken a side* has fused into one of the two and does not
  count.
- **What the spec leaves out.** It says only that "the topology arms MUST differ". That names no direction, so a result
  either way would pass. `M0.4` forbids that.
- **The arms.** Same coach parameters and the same session schedule in each:
  - **Outside, joint:** sessions with both spouses; Dr Halim stays in contact with each and sides with neither.
  - **Outside, one spouse:** sessions with Ravi alone; Dr Halim still sides with neither. In Bowen's practice a coach can
    work through one person if he stays outside.
  - **Sided:** sessions with both spouses, but Dr Halim is drawn in on Marta's side (`TRIANGLE` toward him from her,
    and he goes along).
- **Proposed direction.** The couple's reactivity on the marital tie falls more in both outside arms than in the sided
  arm. The two outside arms are not required to differ.
  - Readout: the spouses' combined `M4.C.1` deviation on their own tie over the coaching weeks.
- **Mutation.** Make the `M8` predicate position-blind (a sided coach still counts as live). The outside and sided arms
  then come out alike, and the test turns red.
- **What the owner decides:**
  - whether this is the right reading of "position, not skill";
  - whether the one-spouse arm belongs in it.

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
- **Build note (2026-10-11).** Each policy act belongs to one channel, declared in `config/bowen/event_kinds.md`, so
  the channel an act is stamped with is its kind's: `self` kinds are logged `self_directed`, the rest `automatic`. An
  `I-POSITION` sequence's steps are stamped the same way. Nothing reads the channel until B6, so B3 changes the log
  and the golden fingerprints but not what a run computes. It shares B4's regeneration.

### B4. Cutoff as rejection and the reconnection pull (P1, P2)
- Per the decision memo. New tie field: `rejection`, owned by the cutter's side.
- `apply_delivered_cutoffs` no longer sets `interactive = False`.
- Appraisal reads rejection.
- `REDUCE_CUTOFF` lowers rejection.
- The pattern readouts' cutoff definition is unchanged. *Corrected 2026-10-11 (review):* it does not read the
  engine's refusals. It counts `CUTOFF` acts sent after an approach (`config/bowen/pattern_acts.md`). Whether it
  should also count the engine's `refusal` records is open for the owner (`TODO.md`).
- **Tests:**
  - a cut tie stays interactive;
  - repeated refusals raise rejection;
  - sustained contact lowers it;
  - a silent tie reads as distance, not cutoff;
  - the seed-7 silences shrink (a readout, not an assertion);
  - P2's pull: a cut tie's "too little" deviation rises with time, and an approach relieves it.
- **Build notes (2026-10-11, read from `M1.B.3c`, `M4.C.1c` and `M5.B.3`; each choice `[I]`, for the owner to correct):**
  - **The field.** `Relationship.rejection`, per member, in [0, 1]. A tie is cut while either member holds rejection
    above zero; `tie_state` reads `cut_off` exactly then and returns to `ordinary` when both reach zero. The tie stays
    interactive throughout.
  - **A delivered `CUTOFF`** raises the sender's rejection by `rejection_cutoff_gain` × strength and clears the
    sender's own felt impingement at once (the immediate relief, `M11.C.4`). The target appraises the act as any other.
  - **A refusal.** A move delivered to a person who holds rejection toward its sender, other than `CUTOFF` and
    `REDUCE_CUTOFF`, is refused. It is not appraised by, and has no effect on, the cutter. The cutter's rejection rises
    by `rejection_refusal_gain` × strength, and the sender bears the refusal: acute anxiety
    `refusal_cost_gain` × strength × conductance × steepness. Each refusal is logged by name (`M6.4`). Witnesses
    still see the move. Exogenous and symptom events are not moves and are not refused.
  - **Accepted contact.** `REDUCE_CUTOFF` is the move the cutter accepts. Its delivery lowers both members' rejection
    by `rejection_relief` × strength, so the actor's own and the cutter's fall together (`M5.B.3`).
  - **The pull.** While a tie is cut, resting contact is zero for both members, so felt contact relaxes toward none
    and the "too little" side grows (`M4.C.1c`). An accepted approach restores resting contact once rejection is gone.
    No pull term is written.
  - **Legality.** Every act is legal on a cut tie, since it is interactive. `REDUCE_CUTOFF` stays legal only on a cut
    tie. An `I-POSITION` sequence still cannot run across one.
  - **Declared cutoffs.** A family row with `tie_state` `cut_off` is interactive, and both members start with
    `declared_rejection`. Ana–Bruno (Phase C) and Ravi–Dario (Phase D) are declared this way. Which member is the
    cutter is not stated in either family, so both hold it.
- **Measured before regenerating (2026-10-11, Phase C family, seeds 7, 1 and 3, 104 weeks). Held for the owner.**

  | | before B4 | B4 as built |
  |---|---|---|
  | tie-weeks cut off, of 1,040 | 591–699 | 835–911 |
  | mean excess anxiety, first / last 26 weeks | 7.8–9.7 / 5.1–7.1 | 12.9–13.6 / 16.2–17.3 |
  | `CUTOFF` acts | 48–62 | 97–118 |
  | `REDUCE_CUTOFF` acts | 47–70 | 1–3 |
  | refusals | — | 353–376 |

  - **Why:** rejection falls only through `REDUCE_CUTOFF`. Before B4 that act was chosen often because it was the
    only act legal across a severed tie. Now every act is legal on a cut tie, so `REDUCE_CUTOFF` competes with all of
    them, and a self-directed act weighted by functional level is rarely drawn. Cut ties therefore stay cut.
  - **Variants tried (same seeds):**
    - the cutter's own contact also lowering rejection: no change;
    - rejection fading by 5% a week: no change, since only "above zero" has any effect;
    - no repeat `CUTOFF` toward a person already rejected: 15–16 cutoffs, but still 828–911 cut tie-weeks;
    - refused only above 0.5, with accepted contact wearing rejection down: 715–867 cut tie-weeks.
  - **Consequences in the suite:**
    - the `I-POSITION` ensemble (`test_m5d3`, `test_m5d8`): PREPARE now fails on excess anxiety in 1,830 of 1,862
      attempts;
    - `M11.C.29`'s vacuity check: no seed skips a withdrawal without the hold.
- **Owner decision 2026-10-11: graded rejection (spec revision 16).** Built with threshold 0.5 and fade 0.05 a week.
  Measured on the same seeds:
  - cut tie-weeks 542–592, back to the pre-B4 level;
  - mean excess 11–13, still above the pre-B4 5–10.
- **Found after building (2026-10-11), decided below:**
  - **A flaw in the values I proposed.** One `CUTOFF` raises rejection to exactly 0.5, and refusal needs more than
    0.5, so a single cutoff refuses nothing.
    - Threshold 0.3: one cutoff refuses for about four weeks, and each refusal renews it by two. Cut tie-weeks
      601–647; excess 13–15.
    - Threshold 0.25: cut tie-weeks 588–666; excess 13–17.
  - **Family anxiety stays higher under every B4 variant.** It does not depend on the refusal values or the refusal
    cost (cost 0: unchanged).
    - Mean excess per person over three seeds, before → after: Ravi 12.9 → 21.7, Marta 14.0 → 20.5, Nadia 7.8 → 14.4,
      Pia 8.6 → 17.7, Bruno 1.9 → 11.7.
    - The likely cause: a cut tie is no longer silent. Before, severing stopped every act on the tie. Now automatic
      acts keep crossing it, which is the owner's definition (cutoff is not absence of interaction).
    - The marital tie is cut less: 70 tie-weeks against 214.
  - **The `I-POSITION` ensemble** (`test_m5d3`, `test_m5d8`): Ravi's excess is above `defence_threshold` (15) most
    weeks, so PREPARE fails in 1,907 of 1,941 attempts and no sequence completes. Those two tests fail.
- **Owner decision 2026-10-11:**
  - **The threshold is 0.3.**
  - **Option (a):** the more anxious family is the emergent result.
  - **Test changes that follow:**
    - the `I-POSITION` ensemble tests run on the triad fixture. Measured there over 24 seeds: 21 aborts, 13 holds,
      4 completed sequences;
    - `M11.C.29`'s hold test asserts that every withdrawal is legal without the hold;
    - the dyad symmetry test uses 200 seeds (over 1,000 seeds: difference 0.03, standard error 0.14).
- **Review and regeneration (2026-10-11):**
  - `learning-qa` and a correctness pass found 10 items. All are fixed with tests:
    - refused moves were still recorded as delivered;
    - `I-POSITION` was legal across a cut tie;
    - a trace line was misplaced;
    - `REDUCE_CUTOFF` reopened worry edges;
    - an ended cutoff erased a distant tie.
    A new invariant, `M1.B.3c`, holds each week. A narrow re-sweep was clean.
  - **The Phase C verdicts re-read under the B4 engine (information, not a gate). One moved:** `M11.C.32`, PASS →
    FAIL.
    - Its `reached_peak` difference shrank from −0.09 ± 0.10 to −0.02 ± 0.022.
    - Under B4 the calm arm also rarely reaches `PEAK`, since `PREPARE` fails on anxiety (above).
    - Every other verdict is unchanged.

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
- **Build notes (2026-10-11; each choice `[I]`, for the owner to correct):**
  - **Where it runs.** At the end of every 52nd fast tick, after that tick's invariants, as its own step. A
    slow-tick invariant check runs after it:
    - `basic_level` only rose;
    - nobody financially dependent rose;
    - `chronic_anxiety` never rose.
  - **Chronic anxiety (P5, `M7.B.1`).** It changes only when `basic_level` rises, keeping its share of the distance to
    the top of the scale: `chronic × (100 − new) / (100 − old)`. It never drifts and never rises. Nobody is born in a
    D-I run, so P5's derivation for the newborn is a hook for D-II.
  - **Tie wear (P6, `M7.E.4`).**
    - **The rate:** each slow tick, conductance falls by `tie_wear_rate × (mean chronic anxiety of the two) / 100` of
      itself.
    - **The offset:** that loss is reduced by `wear_offset_per_act` for each `STAY-IN-CONTACT` sent on the tie that
      year, down to none.
    - **Not covered:** completed exchanges also offset in the spec, but they are not counted yet, since no exchange
      is logged per tie.
    - Bond energy is untouched (`M1.B.4`).
  - **`dyad_age`** counts slow ticks. Nothing reads it yet.

### B6. The estimator (`M1.A.4a`–`4c`, `M7.A.1a`; P4)
- A per-person history of `functional_level` and the load that tick imposed, kept in state.
- The estimator reads that history on the slow tick and writes `basic_level`.
- Financially dependent people get no rise (`M7.A.1a`).
- The window and breadth are config, with the sensitivity sweep.
- **Tests:**
  - a benign window does not raise `basic_level`;
  - a nodal-loaded, stable window does;
  - a dependent twin does not.
- **Build notes (2026-10-11; each choice `[I]`, for the owner to correct):**
  - **The history.** Kept per person, for each fast tick of the last `estimator_window` slow ticks:
    - `functional_level` at the end of the week;
    - the load carried that week: excess acute anxiety over the chronic floor, at selection;
    - which channel the person acted in (self-directed, `WITHHOLD`, or automatic). A scripted act counts by its
      kind, since a criterion arm that scripts `I-POSITION` or exchanges is declaring that behaviour (C.18, C.33);
    - the kinds of nodal event that reached them (revision 17: any major event on a person, `config/bowen/nodal_events.md`).
  - **The estimate, on each slow tick once the window is full** (so from the third slow tick):
    - load-weighted mean `m` and standard deviation `sd` of `functional_level`, weighted by load (`M1.A.4b`);
    - breadth `b`: the number of distinct *kinds* of nodal event in the window. A stressor that recurs is one
      situation: the Phase C spell, a job loss every two weeks, would otherwise count 78 "situations" in three years;
    - self-directed share under load `q`: the load-weighted share of weeks acted in the self-directed channel or
      withheld (`M1.A.4h`'s primary signal);
    - evidence `e = m − estimator_sd_weight × sd − basic_level`.
  - **The update.** `basic_level` rises by `estimator_rate × (basic_level / 100) × q × e` only if all of these hold:
    - `e > 0`;
    - `b ≥ estimator_breadth`;
    - the person is not financially dependent (`M7.A.1a`).

    The factor `basic_level / 100` is `M1.A.4d`'s falloff toward the bottom, with no hard gate.
  - **`basic_level` does not fall in D-I.** A fall belongs to the child's estimator in D-II (`M7.E.1`). With
    `functional_level` mostly at or below `basic_level` under load, a two-way estimator would pull every adult down
    each year.
  - **A known limit, for C.18 and C.20.** Anxiety does not move `functional_level` in the engine. Only borrowing
    (over- and underfunctioning) and `I-POSITION` exchanges and pull-ups do. So load weighting can separate a
    borrowed gain from a basic one only where the borrowing reverts under load. C.18 and C.20 will show whether it
    does. If not, that is a finding for the owner, not something to tune.
  - **Sensitivity (`M1.A.4c`):** window and breadth join the sweep record when C.18 is built, since no criterion reads
    the estimator before then.
  - **Review (2026-10-11).** No scenario has more than one kind of nodal event, so no rise has happened. That is the
    breadth gate, not a result. Without a coach, evidence is already positive for several people, so C.17 must be
    built with three nodal kinds in both arms (`TODO.md`).
  - **Fixes from the review:**
    - only a `STAY-IN-CONTACT` that reached the other offsets tie wear;
    - the history test exercises a stressor and the channel;
    - `dyad_age` is logged.
    - contact is counted in the year it arrives.
  - **Records regenerated (2026-10-11):** no Phase C verdict moved from the B4 records.

### Step 6 result: the estimator's four criteria (2026-10-11, spec revision 17)

Built in `src/bowen/ensemble/criteria_d.py`. Every arm carries the declared nodal load (three kinds;
`config/bowen/criteria.md`) and runs 208 weeks.

The adaptive ensemble, with the same rules as Phase C. These numbers come from an ad hoc run, not a committed record;
they join a record when the Phase D record is built (step 12). The run was repeated after the review fixes:
- C.33's arms now differ only at the slow tick;
- C.17's nodal load goes to every member.

| Criterion | Verdict | Readout, treatment − baseline (± half-width) | Mutation |
|---|---|---|---|
| `M11.C.33` dependence gate (premise) | **PASS**, 150 seeds | rise +0.23 ± 0.09 | gate and its invariant removed: arms equal (+0.003 ± 0.03), FAIL, so proved. With the gate alone removed, the slow-tick invariant raises. |
| `M11.C.18` binder against the real thing (check) | **FAIL**, 150 seeds | rise +0.05 ± 0.11 (p 0.06) | not run |
| `M11.C.20` borrowed against basic (check) | **FAIL, reversed**, 150 seeds | rise −3.05 ± 0.60: the borrowed arm rises more | not run |
| `M11.C.17` a coach helps (composite) | **FAIL**, 150 seeds | mean rise −0.03 ± 0.03: no coach effect | not run |

**Why the three fail, as far as the code shows (for the owner, not tuned):**
- **C.20, and most of C.18: anxiety does not move `functional_level`.** This is the limit B6 recorded.
  - A gain borrowed by over/underfunctioning stays after the support is withdrawn, because nothing under load pulls it
    back.
  - The estimator reads that steady, elevated functioning as evidence.
  - The spec has the missing relationship: `M1.A.5`, and `M1.A.4d`'s source, "their functional levels of
    differentiation can change as the level of chronic anxiety fluctuates"; `M1.A.5c`, a borrowed rise is pseudo-self
    and "does not survive withdrawal of the source".
  - Building it means `functional_level`'s swing falls under load, more at a lower basic level. That is an engine
    change, for the owner to approve.
- **C.17: the coach cannot reach what the estimator reads.**
  - A landed contact raises `systems_perspective`.
  - The estimator reads functioning, load and the self-directed share.
  - The self-directed channel's weight reads `functional_level` only (`M4.D.1a`). Perspective changes only a few
    gates (`M4.D.3b`, `M5.F.4`).
  - So coaching barely changes behaviour the estimator sees. How a coach's effect should reach behaviour is for the
    owner.

**After spec revisions 18 and 19 (2026-10-11), on the final engine, after the review fixes — superseded below.** The
system's adjustment there added to functioning with no ledger and no settling back, and stacked; these numbers, C.17's
above all, were carried by it. Ad hoc runs, not yet in a record; 150 seeds each (C.17: 100):

| Criterion | Verdict | Readout | The mutation that turns it red (run before the review fixes) |
|---|---|---|---|
| `M11.C.33` | **PASS** | +0.18 ± 0.04 | the slow-clock gate with its invariant (FAIL, +0.003) |
| `M11.C.18` | **PASS** | +0.10 ± 0.04 | the stress response removed (FAIL, +0.03 ± 0.05) |
| `M11.C.20` | **PASS** | +0.13 ± 0.04 | borrowed functioning read again (FAIL, reversed −1.08 ± 0.48) |
| `M11.C.17` | **PASS** | coached +0.63 ± 0.32 | the coach's landed contact removed (FAIL, −0.45 ± 0.38) |

**For the owner.** The spec names `M1.A.4b`'s load weighting as the mutation for `M11.C.18` and `M11.C.20`. With it
flattened, both still pass: C.18 +0.13 ± 0.05, C.20 +0.15 ± 0.05. With the self-directed share removed they also still
pass. What carries them is the mechanism in the table: the stress response for C.18, and the estimator ignoring
borrowed functioning for C.20. By `M11.5` a survivor does not prove the criterion. Either the mutation clause is
restated to the mechanism that carries each, or the load weighting is shown to matter in some other test.

**After spec revision 20 as first pushed (`a35a962`, 2026-10-11) — superseded below.** The adjustment there stacked,
and these numbers, C.17's above all, were carried by that defect. The owner
chose to restate the mutation clause of `M11.C.18` and `M11.C.20` to the mechanism that carries each. Ad hoc runs on
the revision 20 engine, not yet in a record (the Phase D record is step 12):

| Criterion | Verdict | Readout | Mutation (rerun at revision 20) |
|---|---|---|---|
| `M11.C.33` | **PASS**, 150 seeds | +0.18 ± 0.04 | not rerun |
| `M11.C.18` | **PASS**, 100 seeds | +0.12 ± 0.05 | coping and soothing removed, the calm settling kept: **FAIL**, +0.02 ± 0.03 (p 0.22), 150 seeds |
| `M11.C.20` | **PASS**, 150 seeds | +0.13 ± 0.04 | borrowed functioning read again: **FAIL, reversed**, −0.80 ± 0.46, 100 seeds |
| `M11.C.17` | **PASS**, 100 seeds | coached +0.66 ± 0.30; family +0.22 ± 0.09 (reported) | not rerun |

Each mutation turns only its own criterion red: with coping and soothing removed `M11.C.20` still passes (+0.06 ± 0.04),
and with borrowed functioning read `M11.C.18` still passes (+0.39 ± 0.27). With the load weighting flattened (every
week of the window weighted equally, loaded or not) both still pass: C.18 +1.61 ± 0.58, C.20 +1.92 ± 0.44. A first
C.18 mutant that switched off the whole stress response step, the calm settling with it, failed only narrowly
(+0.03 ± 0.03, p 0.06); the review of 2026-10-11 pointed out it removed more than the clause names, so the narrower
mutant above is the one the clause describes. Both are single ad hoc runs on the runner's seeds; the step-12 record
confirms them.

**Revision 20 completed (2026-10-12): the estimator leaves the adjustment out, and the adjustment is a level.**

- **What went wrong first:** under the criteria's steady nodal load nobody is ever calm (the trace below: loaded in
  97–100% of weeks), so the stacked adjustment never settled and grew to 20–35 points. The estimator read it as each
  person's own functioning. The owner chose (a), leave it out. With the stack still in place and left out, C.17's
  coached rise fell to +0.02 ± 0.02.
- **The fix:** the adjustment is a level of at most `system_adjust_gain`, renewed while the stance works. A first
  version also took falls through giving way or soothing off the adjustment. The review of 2026-10-12 found that
  recorded each fall in two ledgers, so functioning drifted above where its ledgers put it and a renewed stance refilled
  the adjustment: the stack came back through the stress swing and pseudo-self, and C.17 passed on it (+0.52 ± 0.16).
  It was removed; every change to functioning is now in exactly one ledger
  (`tests/bowen/test_stress_response.py::test_m1e7_every_change_lands_in_one_ledger_so_calm_returns_functioning_to_its_start`).

Ad hoc, the revision 20 engine as committed, the runner's seeds:

| Criterion | Verdict | Readout | Its mutation |
|---|---|---|---|
| `M11.C.33` | **PASS**, 150 seeds | +0.05 ± 0.01 | not rerun |
| `M11.C.18` | **PASS**, 150 seeds | +0.05 ± 0.01 | coping and soothing removed: **FAIL**, 0.00 ± 0.00, 50 seeds |
| `M11.C.20` | **PASS**, 150 seeds | +0.05 ± 0.01 | borrowed pseudo-self read again: **FAIL, reversed**, −0.74 ± 0.30, 100 seeds |
| `M11.C.17` | **FAIL**, 150 seeds | coached −0.00 ± 0.00; family −0.00 ± 0.00 (reported) | landed contact removed: FAIL, 0.00 ± 0.00, 50 seeds — not a test while the criterion is red |

With coping and soothing removed nobody's basic level rises in either arm of C.18 or C.20, so the stress response
carries both (C.20 fails under it too). Reading borrowed pseudo-self leaves C.18 passing (+0.16 ± 0.19).

**The trace of the coached people (owner request, 2026-10-12).** `M11.C.17`'s ten years, 20 seeds per arm, the two
people the coach is in contact with; yearly means (ad hoc script, not a record):

| Year | Arm | Perspective | Self-directed weeks | Soothing weeks | Functioning | Stress swing | Estimator's evidence | People who rose |
|---|---|---|---|---|---|---|---|---|
| 1 | coach-free | 0.001 | 0.15 | 0.51 | 39.6 | −1.6 | — (window not full) | 0/40 |
| 1 | coached | 0.06 | 0.14 | 0.45 | 37.6 | −1.4 | — | 0/40 |
| 4 | coach-free | 0.006 | 0.16 | 0.54 | 35.6 | −6.2 | −10.7 | 0/40 |
| 4 | coached | 0.30 | 0.23 | 0.44 | 32.0 | −4.5 | −12.4 | 0/40 |
| 7 | coach-free | 0.011 | 0.14 | 0.54 | 34.2 | −10.3 | −13.5 | 0/40 |
| 7 | coached | 0.50 | 0.31 | 0.42 | 31.6 | −6.3 | −14.0 | 0/40 |
| 10 | coach-free | 0.015 | 0.11 | 0.57 | 30.1 | −13.9 | −15.4 | 0/40 |
| 10 | coached | 0.63 | 0.37 | 0.35 | 27.9 | −6.4 | −14.7 | 0/40 |

What it shows:
- **Coaching changes behaviour.** Perspective climbs to 0.63; self-directed weeks rise from 0.14 to 0.37 while the
  coach-free arm's fall to 0.11; soothing weeks fall; the stress-moved swing ends at −6.4 against −13.9.
- **Functioning still falls in both arms**, from about 40 to 28–30, toward the stress floor (basic − 15 = 24.5).
  Everyone is under load in essentially every week, so the stress response never settles back. By year 10 the
  coached people's functioning (27.9) sits 11.6 below basic: 6.4 of it is the stress swing and about 5 is elsewhere.
  The coach-free people's (30.1) sits 9.4 below basic with a stress swing of −13.9, so about 4.5 elsewhere is up. Where
  "elsewhere" is (giving way and pseudo-self, the `I-POSITION` pull, the adjustment) is not yet traced.
- **So the estimator never sees a rise.** Its evidence (load-weighted own functioning, less one standard deviation,
  less basic level) is −10.7 to −15.4 in both arms; nobody rises in ten years.
- **Questions for the owner** (`TODO.md`): whether C.17's declared load (a nodal event to every member every few weeks,
  so nobody is ever calm) is the right test bed; why the coached people's functioning falls despite coping more —
  the next trace is the ledgers other than the stress swing; and whether the estimator's form (`M1.A.4k`), which needs
  the loaded mean above basic level, can register improvement that shows as falling less rather than rising.

**Where the coached people's fall comes from (owner request, 2026-10-12).** The same ten years, 20 seeds per arm, the
two coached people; each change to their `functional_level` summed by the mechanism that logged it (ad hoc scripts,
not a record; nothing changed functioning without a log row):

| Ten-year change, per person | Coach-free | Coached |
|---|---|---|
| Stress response: down (soothing) | −17.5 | −15.9 |
| Stress response: up (coping) | +3.7 | +9.5 |
| Giving way, net (`M1.B.5`'s pseudo-self exchange) | +3.5 | −6.2 |
| The system adjusts | +1.0 | +1.0 |
| **Total** | **−9.4** | **−11.7** |

Coaching gains 7.4 points through the stress response and loses 9.7 through giving way. The giving-way loss is all with
people other than the spouse (between the two spouses, both coached, it nets to zero). Split by which act started it:

| Functioning moved, per person, with people other than the spouse | Coach-free | Coached |
|---|---|---|
| Gave: someone else's `OVERFUNCTION` toward them | −62.8 | −79.7 |
| Gave: their own `UNDERFUNCTION` | −25.5 | −19.4 |
| Took: someone else's `UNDERFUNCTION` | +64.7 | +69.0 |
| Took: their own `OVERFUNCTION` | +27.1 | +23.8 |

What it shows:
- **The coached people stop over-functioning, as the owner described improved functioning** ("one person, in defining
  self, chooses to stop overfunctioning for the other"): their own `OVERFUNCTION` acts fall from 63 to 45 in ten years,
  and `UNDERFUNCTION` from 56 to 47. They also make fewer `CONFLICT`, `DISTANCE` and `TRIANGLE` acts, and many more
  `STAY-IN-CONTACT` (14 → 58) and `DETRIANGLE` (4 → 15).
- **The others then take charge of them more** (−62.8 → −79.7). Every delivered `OVERFUNCTION` moves functioning from
  the person it is aimed at to the one who makes it, whatever the person aimed at is doing: holding a position does not
  stop it. So stepping back from over-functioning costs functioning in the model.
- **That is the coached fall.** The stress response's gain from coaching (+7.4) is outweighed by being
  over-functioned for (−16.8 more), less the over-functioning they gave up.
- **For the owner:** whether an `OVERFUNCTION` aimed at a person who is holding a position should move functioning
  from them at all (the exchange takes two, `M1.B.5`), or whether this is the change-back reaction the system goes
  through before it adjusts (`M1.E.7`), which here never settles because nobody is ever calm.

**After spec revision 21 (2026-10-12): holding a position stops the exchange.** The owner chose (1): an `OVERFUNCTION`
or `UNDERFUNCTION` delivered to someone holding a position toward the sender moves nothing (`M1.B.5`). After the
review, the owner chose that any reactive act toward that person ends the stance — any automatic act toward them, or
triangling someone in about them — not only a soothing one. Ad hoc, same scripts and seeds, on the engine as
committed:

| | Coach-free | Coached |
|---|---|---|
| Ten-year change in functioning, per coached person | −11.1 | −11.4 |
| of which giving way, net | +2.7 | −4.7 |
| of which the stress response, net | −14.8 | −7.7 |
| Gave to someone else's `OVERFUNCTION` (people other than the spouse) | −57.8 | −61.6 |
| Estimator's evidence, year 10 | −15.9 | −14.7 |
| People whose basic level rose, any year | 0 of 40 | 0 of 40 |

| Criterion | Runner's verdict | Readout | Its mutation |
|---|---|---|---|
| `M11.C.33` | PASS, 150 seeds | +0.05 ± 0.01 | not rerun |
| `M11.C.18` | PASS, 150 seeds | +0.05 ± 0.01 | coping and soothing removed: FAIL, 0.00 ± 0.00, 50 seeds |
| `M11.C.20` | PASS, 150 seeds | +0.05 ± 0.01; scripted support blocked by a held position 10% ± 2% (report only) | borrowed pseudo-self read: FAIL, reversed, −0.67 ± 0.26, 100 seeds |
| `M11.C.17` | PASS, 150 seeds, **not counted** | coached +0.01 ± 0.01 (p 0.02); family +0.00 ± 0.00 | landed contact removed: FAIL, −0.00 ± 0.00 |

With holding a position switched off (`holds_position` always false), C.33, C.18 and C.20 still pass (+0.05 each) and
C.17 fails (−0.00 ± 0.00, p 0.69): revision 21 carries none of the first three. With borrowed pseudo-self read, C.18
now fails too (+0.08 ± 0.14).

What it shows:
- **C.17's pass is not evidence.** Its interval includes zero, and in the trace's 20 seeds per arm no coached person's
  basic level rose in any year. It passes the runner's signed-rank test on a handful of small rises.
- **Ending a stance on any reactive act takes back most of the first version's gain.** The coached people still make
  many automatic acts toward the people they took a stance with, so their stances end often; they fall as far as the
  coach-free (−11.4 against −11.1), where with soothing alone ending a stance they fell less (−8.0 against −12.3).
- **The blocker is unchanged:** everyone is loaded in nearly every week, functioning falls in both arms, and the
  estimator's evidence stays at −15 to −16. The owner's first and third questions (C.17's declared load, and whether
  the estimator can register falling less) are what is left.

**What keeps everyone above the calm band (owner request, 2026-10-12).** First, nodal events less often (the owner's
"make C.17 calmer"): every 8, 16, 26 or 52 weeks, or none, the coached people are loaded in 100% of weeks, in both arms,
and nobody's basic level rises (ad hoc, 20 seeds per arm). So the load is not the nodal events. Then each week's change
in the coached people's acute anxiety, years 8–10, summed by the mechanism that logged it (ad hoc, 20 seeds per arm,
nodal events as C.17 declares them; per person-week; nothing unlogged):

| | Coach-free | Coached |
|---|---|---|
| Excess anxiety at selection (the calm band's edge is 5) | 32.6 | 36.3 |
| Appraisal of others' acts (over-functioning, conflict, distance, cutoff, pursuit, …) | +3.0 | +3.4 |
| Appraisal of their own acts | +1.4 | +1.6 |
| Appraisal of the nodal events | +0.5 | +0.5 |
| Standing load (`M4.A.1`) | +1.3 | +1.2 |
| Refusals (`M1.B.3c`) | +1.0 | +0.7 |
| Released cut-off anxiety (`REDUCE_CUTOFF`) | +0.7 | +1.3 |
| Competing urges, triangle transfer | +0.5 | +0.4 |
| Decay (`acute_decay_rate` 0.2 of the excess a week) | −6.6 | −7.3 |
| Bound by distance, calm contact, cutoff | −1.8 | −1.8 |

What it shows:
- **No single source; the family's own traffic.** About 8.4 points a week flow in (9.1 coached), spread over many
  acts; decay takes 0.2 of the excess a week and binding about 1.8, so the excess settles near 30–35, six to seven
  times the calm band's edge. The nodal events are about 0.5 of it.
- **The calm band cannot be reached even in a quiet family.** `config/bowen/constants.md` records that the standing load
  alone was set to hold a person "a few points above their chronic floor": its 1.2–1.3 a week over a decay of 0.2 is an
  excess of about 6–7, already above the band's edge of 5. Two `[I]` constants disagree: nobody in this family can ever
  be calm, so the stress response never settles back and the system's adjustment never fades.
- **Coaching adds load here:** the coached people reopen cut-off ties more (`REDUCE_CUTOFF`, releasing what the cutoff
  held) and sit higher (36.3 against 32.6).
- **For the owner:** whether "calm" should be read against each person's own usual level rather than a fixed 5 points,
  whether the bands should be set from the operating point the engine actually runs at, or whether the family's
  traffic itself should carry less anxiety.

**After spec revision 22 (2026-10-12): calm is relative to the person.** The owner chose (1): loaded and calm are read
against each person's usual level, and the stress response is driven by the excess above it; the policy's bands and the
estimator's weighting are unchanged. After the review, the owner chose that the usual level is the plain mean of the
weeks so far for its first year (started from week 0 alone it sat near 0.5 and the first year was over-load), then a
running average over about a year (`usual_level_rate` 0.02). Ad hoc, same scripts and seeds, on the engine as
committed:

| Ten-year change in functioning, per coached person | Coach-free | Coached |
|---|---|---|
| Stress response, net | −3.7 | −3.6 |
| Giving way, net | +2.4 | −3.1 |
| The system adjusts | +4.0 | +4.5 |
| **Total** | **+2.7** | **−2.2** |
| Stress-moved swing, year 10 | −0.4 | −0.1 |
| Estimator's evidence, year 10 | −8.6 | −9.9 |
| People whose basic level rose, by year | 1 of 40 in years 7–8, 3 in years 9–10 | 4 of 40 in year 9, 5 in year 10 |

| Criterion | Runner's verdict | Readout |
|---|---|---|
| `M11.C.33` | PASS, 150 seeds | +0.002 ± 0.001 |
| `M11.C.18` | PASS, 150 seeds | +0.002 ± 0.001 |
| `M11.C.20` | PASS, 150 seeds | +0.002 ± 0.001; support blocked 10% ± 2% (report only) |
| `M11.C.17` | PASS, 150 seeds, **not counted** | coached +0.003 ± 0.003 (p 0.003) |

What it shows:
- **The stress response now settles.** The stress-moved swing stays near zero (−0.1 to −0.4, against −8 to −15 at
  revision 21), and the system's adjustment settles and renews (+4.0 to +4.5 over ten years). Coach-free people's
  functioning no longer erodes (+2.7).
- **The coached people still fall a little, through giving way** (net −3.1 against +2.4): they are taken over more,
  as the earlier trace found, and their stances end on any reactive act.
- **Rises begin late in both arms.** From year 9 the coached arm has more people rising (4–5 of 40) than the coach-free
  (3), but the estimator's evidence stays far below zero (−8.6 and −9.9 at year 10) although coach-free functioning
  sits near basic (42 against 39.5): its reading (about 33) leaves out positive pseudo-self and weights the most loaded
  weeks most.
- **The criteria's readouts are now tiny** (+0.002 a person): with the background load no longer counted, coping moves
  functioning very little, so the estimator has little to read. C.33, C.18 and C.20 pass the runner's test; none of the
  four shows a rise that would count as evidence.

**Why the estimator reads about 9 points below basic level (owner request, 2026-10-12).** Each coached person's last
estimator window (years 8–10) at the year-10 slow tick, the reading rebuilt five ways, each less one standard
deviation, against basic level 39.5 (ad hoc, 20 seeds per arm, revision 22 engine):

| Reading | Coach-free: mean, sd, evidence | Coached: mean, sd, evidence |
|---|---|---|
| Plain mean of whole functioning | 41.8, 4.4, −2.0 | 37.5, 4.2, −6.1 |
| Plain mean of own functioning | 33.4, 2.3, −8.4 | 32.5, 2.5, −9.6 |
| Load-weighted, whole | 41.5, 4.2, −2.3 | 37.1, 4.2, −6.6 |
| **Load-weighted, own — the estimator** | **33.2, 2.3, −8.6** | **32.2, 2.5, −9.9** |
| Load-weighted, own, with functioning lent added back | 39.1, 0.3, −0.6 | 39.6, 0.2, −0.1 |

Over the window, positive pseudo-self averages 7.8 coach-free and 4.4 coached; negative pseudo-self −5.8 and −7.2.

What it shows:
- **The load weighting barely matters** (0.2–0.5 points either way).
- **The asymmetry in what is left out is the gap.** The estimator subtracts borrowed functioning (positive pseudo-self,
  revision 19) but keeps functioning lent: a person who gave way is read at their lowered level, and a person who took
  over has the gain removed. Pseudo-self is conserved within a pair (`M6.I.4`), so across a family the reading sits
  below basic level by about the pseudo-self in play. Subtracting pseudo-self with its sign — leaving out both what was
  borrowed and what was lent — puts own functioning at basic level (39.1 and 39.6) with almost no spread, and the
  evidence near zero (−0.6 coach-free, −0.1 coached).
- **With that reading the coached people sit slightly ahead** (39.6 against 39.1), but the evidence is still not above
  zero: a rise needs own functioning held above basic by more than its spread, and with the background load no longer
  counted, coping now moves functioning very little.
- **For the owner:** should the estimator also add back functioning lent, so that it reads functioning with the whole
  pseudo-self exchange left out? The owner's revision-19 words were about over-functioning ("pseudo-self looks
  better"); they did not address the person who gave way.

**After spec revision 23 (2026-10-12): the estimator reads pseudo-self with its sign.** The owner chose to add back what
was given way. The estimator's reading of the coached people now sits at basic level (39.1 coach-free, 39.6 coached,
spread 0.2–0.3), where before it sat near 33. Ad hoc, same scripts and seeds:

| Criterion | Runner's verdict | Readout | Its mutation |
|---|---|---|---|
| `M11.C.33` | PASS, 150 seeds | +0.013 ± 0.002 | not rerun |
| `M11.C.18` | PASS, 150 seeds | +0.013 ± 0.002 | coping and soothing removed: **FAIL**, 0.000 ± 0.001 (p 0.29) |
| `M11.C.20` | PASS, 150 seeds | +0.013 ± 0.002; support blocked 10% ± 2% (report only) | pseudo-self of both signs read: **FAIL, reversed**, −0.87 ± 0.23. Borrowed pseudo-self read, given way still added back (the clause as restated): **FAIL, reversed**, −0.85 ± 0.23 |
| `M11.C.17` | **PASS**, 150 seeds | coached **+0.021 ± 0.009**; family +0.006 ± 0.003 | landed contact removed: **FAIL**, 0.000 ± 0.001 (p 0.09). Borrowed pseudo-self read, given way still added back: **FAIL, reversed**, −0.76 ± 0.61 |

| People whose basic level had risen, by year (of 40; 20-seed trace, two coached people per seed) | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| Coach-free | 0 | 0 | 1 | 4 |
| Coached | 1 | 3 | 7 | 16 |

The estimator's evidence at year 10 is −0.6 coach-free and −0.1 coached, and the coached arm's has been closing since
year 7.

What it shows:
- **`M11.C.17` passes, and it is proved by mutation** (CLAUDE.md's bar for coverage). The interval excludes zero and
  removing the coach's landed contact turns it red. In the 20-seed trace coached people rise from year 4, four times as
  many as without a coach by year 10. No minimum meaningful effect is declared: the runner's margin is 0.1 of the
  baseline's standard deviation.
- **The effect is small**: about 0.02 points of basic level per coached person in ten years, on a 0–100 scale. Most of a
  rise still has to come from own functioning held above basic level by more than its spread.
- **C.17 also depends on leaving borrowed pseudo-self out**: reading it, with what was given way still added back, turns
  C.17 red and reversed (−0.76 ± 0.61).
- **What carries it is the coach, not the stress response.** With coping and soothing removed C.17 still passes
  (+0.039 ± 0.010); with holding a position switched off it still passes (+0.017 ± 0.004).
- **C.20 with coping and soothing removed** passes the runner on a readout of 0.000 (p 0.04) — a degenerate pass;
  its own clause, reading borrowed pseudo-self, turns it red.

### B7. Symptom channels (`M7.D.2`, `2a`–`2d`, `M7.D.4`, `M1.A.11c`)
- Load can move between the three channels.
- Lock-in lowers family anxiety until a severity peak, then raises it.
- Load in one channel lowers onset hazard in the others.
- Removal fires off the others' tolerance.
- Every shape is `[I]` and declared.

#### B7 design (spec revision 24, 2026-10-12): Claude's recommendation, pending owner review

The owner asked for the project to be carried forward overnight without waiting for input (2026-10-12). Every choice
below is Claude's recommendation, labelled as such in the spec and the CHANGELOG, and none is an owner decision.

**What was found first.** In ten-year policy runs of the nuclear-four fixture (3 seeds, ad hoc), every person crossed
the symptom threshold within the first weeks and stayed there, at 4 to 31 times the threshold. The integrand is excess
above the chronic floor (`M4.C.3a`), and the family's own traffic holds everyone 15–25 points above it. The calm-band
trace for revision 22 found the same thing at 30–35 points in the Phase C family. The threshold was set (Phase C step
2) to sit above "the integral a steady few points of excess reach, so onset needs a sustained excursion". The steady
level is no longer a few points, so that intent no longer holds. With every channel saturated, no symptom criterion
can tell its arms apart.

1. **The integrand reads excess above the person's usual level** (`M4.C.3a`, amended). It is the same yardstick
   revision 22 gave the stress response. It is still time above a floor, not peak, so many resolved excursions and one
   unresolved excursion still differ. A family's steady state is no longer a symptom in itself; a sustained rise above
   it is. *Cost:* a rise that lasts long enough becomes usual (about a year), so a symptom with nothing holding it
   remits. What can hold it is item 4.
2. **Substitutable channels and the family's focus** (`M7.D.2`, `M1.A.11b`, `M1.A.11c`).
   - Each person gets `channel_focus`, one weight per channel, 0 by default, and `constitution_strength` in [0, 1],
     0.5 by default `[I]`.
   - The expressed channel is the one with the largest
     `constitution_strength × [c = channel_prior] + (1 − constitution_strength) × channel_focus[c]`. Ties go to the
     prior.
   - With no focus the prior is expressed (`M1.A.11b`). A focus can move it (`M1.A.11c` term 2). A constitution
     strength of 1 overrides any focus (term 3).
   - **Not built here:** what writes `channel_focus` from the family's attention under anxiety. That is a belief, so it
     is B8's. Until then a criterion or scenario sets it.
3. **Mutual protection** (`M7.D.2c`). What enters channel *c* each week is divided by
   `1 + channel_protection × Σ_{c' ≠ c} load[c'] / threshold` (`channel_protection` 1.0, `[I]`). Load already carried
   in another channel lowers the hazard on this one.
4. **Lock-in, with its turn** (`M7.D.2a`, `M7.D.2d`).
   - Each person gets a new stock, `symptom_bound`: anxiety the family's focus on their symptom holds.
   - Each week, for a bearer with severity `x = load[expressed] / threshold`, the stabilising coefficient is
     `s(x) = lock_in_gain × x × (1 − x / lock_in_turn)` (`lock_in_gain` 0.05, `lock_in_turn` 2, both `[I]`). It
     rises to a peak at `x = lock_in_turn / 2`, is zero at the turn, and is negative past it.
   - **Below the turn** (`s > 0`): each partner on an interactive tie moves `s × max(0, above_usual)` of their acute
     anxiety into the bearer's `symptom_bound`. This is a transfer, logged. The family calms around the symptom, and
     the bound stock adds to the bearer's integrand, which fosters chronicity (KS23.1).
   - **Past the turn** (`s < 0`): the bound stock is released back to the partners at `|s|` of it a week. It is a
     transfer, logged, and the configuration destabilises.
   - **Removal** (`INSTITUTIONALIZE`) makes the bearer's ties non-interactive, so both directions stop. Below the turn
     the family loses what was holding its anxiety, so removal is resisted. Past the turn the release stops, so removal
     brings relief. Both limbs come from the same curve; no rule writes either.
   - **Remission** (`binder_unavailable` on a symptom channel, `M1.F.9`): the channel's load returns to the family
     budget, as built. Its `symptom_bound` returns to the partners who are on interactive ties, split equally. That is
     what `M11.C.12` reads (`M6.I.1`'s release, carried by their learned patterns).
   - `M6.4` gains the stock `Person.symptom_bound`, with its transfers.
5. **Removal fired by tolerance** (`M7.D.4`, `M1.D.5`).
   - The family gets `tolerance`, per person, `None` by default, meaning never.
   - Each week, if some member has an active symptom, and the mean above-usual excess of the remaining members on an
     interactive tie to that member exceeds their tolerance, the member with the most symptom load is removed
     (`INSTITUTIONALIZE`, endogenous). This depends on the others' anxiety, not on the bearer's severity.
   - Removal is suppressed while an external agent holds the family's decision ownership (`M5.C.1`).
   - No D-I criterion depends on it, and it is pinned by a unit test only.

**As built, four changes from the draft above.**
- **The lock-in constants were set when the code was written, before any run:** `lock_in_gain` 0.2 and
  `lock_in_turn` 3, not 0.05 and 2. An active symptom starts at severity 1. A turn at 2 would put the peak at 1,
  the onset point; at 3 the peak, 1.5, lies inside the active range, and a gain of 0.2 makes the peak share 0.15.
- **Removal fires when any one partner passes their own tolerance**, not on the partners' mean: `M7.D.4` names each
  member's tolerance (`M1.D.5`: per agent).
- **Past the turn** the symptom also stops holding the bearer's own anxiety: the same share of the bearer's excess
  above usual spills to the partners, split equally (a transfer, logged as `lock_in`). Giving back only what had been
  held was too little for a severe symptom to unsettle anything, since a symptom held past the turn from the start
  has held nothing.
- **The hold does not feed the bearer's integrator.** Feeding it made every bearer's symptom climb to the turn and
  stay there. So lock-in calms the family but does not by itself make a symptom chronic (recorded in `TODO.md`).

**Step 7 result (2026-10-12, spec revision 24; ad hoc runs of `CRITERIA_D` with the declared settings, not yet a
record).** On the nuclear-four fixture, except `M11.C.12` on the dyad, as §3 plans.

| Criterion | Arms | Runner's verdict | Readout (treatment − baseline) | Its mutation |
|---|---|---|---|---|
| `M11.C.30` | C1's focus on the social channel in both; the treatment also carries 0.9 of the threshold in the mental channel, held | **PASS**, 50 seeds | social load −20.8 ± 1.7 | cross-channel term zeroed: **FAIL**, 0.0 |
| `M11.C.22` | removal of C1 at week 52, read against removal with no symptom; C1 held at 1.5 (mild) and 4.5 (severe) | **PASS**, 150 seeds | mild +0.33 ± 0.36 (p 0.027); severe −0.80 ± 0.44 | lock-in flattened to its peak, no turn: **FAIL** (the severe limb, +0.50); its stabilising half removed: **FAIL** (the mild limb, 0.0) |
| `M11.C.31` | the spouses' tie holds 20 of distance-bound anxiety; a `binder_unavailable` names it at week 10 | **PASS**, 150 seeds | budget +21.9 ± 0.6 | the held anxiety discarded: the run raises `M6.I.6` at the release week |
| `M11.C.11` | C1 removed at week 52; the treatment brings C1 back at week 104 | **PASS**, 150 seeds | the return's disturbance exceeds the removal's by +2.35 ± 0.86 (first run, with both windows one week early: +1.76 ± 0.85) | the conservation limb is the ledger, asserted every tick; the direction limb has no declared mutant. Probes: reunion not restoring contact, +1.61 ± 0.84 (first run); a symptom ending returning nothing, +1.21 ± 0.73 |
| `M11.C.12` | spouse A holds a mild symptom (1.5) to week 52; the treatment remits it then | **FAIL**, 50 seeds | marital `CONFLICT` 0.0 ± 0.24 | not run |

The numbers above are from `mut24b.sh` and `c12_trace_dyad.py` (scratch, 2026-10-12), run after both reviews' fixes.

What it shows:
- **`M11.C.30` and `M11.C.22` pass and are proved by mutation**, each of C.22's limbs by its own half of the curve.
  They are premises: they confirm the code does what M7.D.2c and M7.D.2a/2d say, and are not findings. **`M11.C.31`
  passes**; its mutant cannot run, because discarding the held anxiety breaks the `M6.I.6` ledger.
- **`M11.C.22`'s mild limb is weak**: the interval reaches below zero, and it passes on its one-sided test. The
  severe limb is clearer. **Unrestated, the mild limb fails**: removing a mildly symptomatic member still relieves the
  family a little (−0.09, against −0.83 with no symptom, 30 seeds ad hoc); it relieves it less.
- **`M11.C.11` passes, and about half its direction is the removed member's hold coming back.** A removed bearer has no
  partners, so their symptom's hold is frozen while they are away and released into the return's window; with that
  release removed the return still exceeds the removal (+1.21 ± 0.73). What carries the rest is not identified. Its
  declared mutation is the conservation invariant, which every run asserts. Its curve shape stays with the owner
  (`M11.E`).
- **`M11.C.12` fails, and the traces say why.**
  - On the dyad (20 seeds, the 26 weeks after week 52): A's symptom held 33.5 at remission (median; 26–47), half of
    it returned to B (the other half to A's parent). B's mean excess over those weeks was 13.8 in the remission arm
    against 10.3 in the arm where the symptom continued. But the spouses' acts toward each other barely moved:
    `CONFLICT` 55 against 56, while `PURSUE` (63 → 75) and `UNDERFUNCTION` (52 → 65) rose. More anxiety in this
    engine does not select conflict in particular.
  - It was first run on the nuclear four (100 seeds, FAIL, −0.16 ± 0.35). Its trace (40 seeds) showed the husband's
    excess rising (17.2 → 20.9), the remitted symptom returning (42 onsets against 6: relapse on cure), and between
    the spouses `CONFLICT` falling (57 → 38) while `TRIANGLE` (46 → 61) and `OVERFUNCTION` (54 → 66) rose.
  - The released anxiety goes into the family's learned patterns, as `M7.D.3` (withdrawn) requires; they are not
    conflict. Nothing has been changed to make it pass.
- **`M11.C.22` was restated at revision 24 to read against a control arm.** Removing anyone relieves this family a
  little (−0.83 ± 0.76 with no symptom, 30 seeds ad hoc), so "resists removal" is read against that.

### B8. Beliefs as a channel (`M9.6`, `M9.7`, `M9.1`)
- **Attribution store:** each person's view of where the family's difficulty lies, written from witnessed events.
- **Appraisal reads it:** appraisal reads the receiver's belief about the sender and the situation, not the true
  state.
- **Projection gate:** the projection clause of `M4.B.2a` and `M5.C.1`'s "system calm" row read through belief.
- **Ties created mid-run** get beliefs (needed in D-II).

#### B8 design (spec revision 25, 2026-10-12): Claude's recommendation, pending owner review

The owner asked for the work to proceed overnight without waiting for input. Every choice below is Claude's
recommendation, labelled so in the spec and CHANGELOG, and none is an owner decision.

1. **Attribution: where each person believes the family's difficulty lies** (`M9.6`, `M9.1`).
   - Each person gets `attribution`, a weight per family member in [0, 1], summing to at most 1. It starts empty: no
     one is blamed.
   - It is written in step 3 from moves the person sent or was delivered (as target or witness, after fidelity).
     Nothing undelivered writes it.
   - Each such move names a locus by its kind, in config (`config/bowen/attribution.md`, editorial content):
     - `CONFLICT`: the other party, as each side sees it, so each spouse says the other; a witness splits it between
       the two.
     - `OVERFUNCTION` and `PURSUE`: the target, the one being looked after or chased.
     - `UNDERFUNCTION`: the sender, the one giving way.
     - `DISTANCE` and `CUTOFF`: the other party.
     - Every other kind names no locus.
   - Each observation moves the weights toward the locus by `attribution_rate × fidelity` (0.05, `[I]`): every
     weight decays by that share and the locus gains it. That is order-independent within a tick, as the tie
     beliefs are.
   - **No rule writes the three patterns.** "Each says the other", "both say the same one" and "both say the children"
     follow from which acts the family makes.
2. **The readout recovers the sink from belief, and the belief from the sink** (`M9.6`; `src/bowen/readouts/`,
   outside the engine).
   - Read the spouses' top loci:
     - each is the other: marital conflict;
     - both are the same spouse: spouse dysfunction;
     - both are the same child, or both are children: child projection;
     - anything else: none.
   - Read in the other direction, the largest sink allocation predicts the pattern.
3. **Appraisal reads belief** (`M9.7`).
   - A witness appraises an event between two people it is not party to scaled by
     `1 + belief_appraisal_gain × believed tension` of that tie (`M9.8`'s belief; `belief_appraisal_gain` 1.0,
     `[I]`). The tie's true state is never read.
   - Events addressed to the person already read the person's own tie, which is its own state (`M4.B.2`).
4. **The family's focus writes the channel** (`M1.A.11c` term 2, left open by B7).
   - When a person above their usual level sends a move to someone, the target's `channel_focus` for that kind's
     channel gains `focus_rate` (0.05, `[I]`).
   - The kinds and their channels are in config (`config/bowen/attribution.md`):
     - caretaking (`OVERFUNCTION`): physical;
     - worry and pursuit (`PURSUE`, `TRIANGLE`): mental;
     - blame (`CONFLICT`): social.
   - Every focus decays by `focus_rate` a week. What the family attends to under anxiety moves the channel; the
     constitution term still overrides it.
5. **The marital-distance gate** (`M5.C.1`).
   - While both spouses sit below their felt contact optimum on the marital tie, a move targeting a member with an
     active symptom produces no improvement in that member: its appraisal cannot lower their acute anxiety, and calm
     contact (`M4.C.10`) does not reach them.
   - It is engine-side, not a policy read, so it reads the tie's true state.
   - **Premise:** `M11.C.14` is premise class, and this gate is the rule its null rests on.

**Not in B8:** the projection clause of "system calm" (B9's), and beliefs for ties created mid-run (D-II).

**As built, as drafted.**

**Step 8 result (2026-10-12; ad hoc runs on the final code of revisions 25–27, not yet a record; `final25.sh` and
`final25.out` in the session scratchpad, run after both reviews' fixes).**

| Criterion | Fixture and arms | Runner's verdict | Readout | Its mutation |
|---|---|---|---|---|
| `M11.C.26` | nuclear four driven to each sink in turn for 52 weeks (scripted acts every 2 weeks); the baseline reads the same runs against the next driving's sink | **PASS**, 150 seeds | beliefs recover the sink +0.76 ± 0.04 (the reverse reading is the same comparison here, so it is reported, not tested) | attribution write randomised: **FAIL**, +0.01 ± 0.02 |
| `M11.C.36` | triad; the spouses exchange `CONFLICT` weekly for 4 weeks; the treatment's witness believes their tie tense (0.8) at the start | **PASS**, 50 seeds | the witness's appraisal +3.07 ± 0.01 | appraisal reads the tie's true state: **FAIL**, 0.0 |
| `M11.C.14` | nuclear four, the spouses' tie held at distance, C1 starting symptomatic; both parents' firmness against talking it out every 2 weeks | **FAIL**, 100 seeds (a null) | C1's symptom load, talking − firmness, −8.0 ± 1.9 against a bound of 5.6 | the gate removed: −6.7 ± 1.9, also outside (bound 5.1) |

What it shows:
- **`M11.C.26` and `M11.C.36` pass and are proved by mutation.** C.26 is a premise and C.36 a check: they confirm that
  the code reads belief as `M9.6` and `M9.7` say. C.26's beliefs are the attribution table read back through scripted
  acts, which is what a premise is.
- **`M11.C.14` fails.** The gate stops any improvement in the symptomatic child, but firmness still impinges, and once
  the symptom ends talking it out calms the child; with the gate removed the arms differ less, not more. The null needs
  either a gate that also blocks harm, or a reading of "zero independent effect" that allows it; that is the owner's.
- **The gate, as first built, was on almost every week** (review 2026-10-12: being below the contact optimum is the
  resting state). It now needs both spouses' felt contact below `marital_distance_fraction` (0.25, `[I]`) of their
  optimum, and reaches only the couple's children (a symptomatic spouse is not gated); in ordinary 208-week runs it is on 26–32 weeks in the nuclear four
  and 18–70 in the Phase C family (3 seeds each, `gate_rate.py`).

### B9 design (spec revision 26, 2026-10-12): Claude's recommendation, pending owner review

- **The cross-person term is weighted by the tie** (`M4.D.6e`): the others' relief after an act is weighted by the
  actor's bond energy on its tie to each, where it was their plain mean. A child's tie to a parent (bond 50) outweighs
  a thin tie to a witness, and a person with no tie to the actor counts for nothing. "Her calming is the
  reinforcement" without any rule naming a parent.
- **`M11.C.2` as a composite**: the Phase C family under the policy for ten years, with the declared nodal load on the
  parents. Each run counts the projection-type acts (`OVERFUNCTION`, `PURSUE`, `TRIANGLE`) each child received from the
  parents; the treatment reads the most-projected child's mean symptom load, the baseline (same run) the
  least-projected child's.

**Step 9 result (ad hoc, on the final code).**

| Criterion | Runner's verdict | Readout | Its mutations |
|---|---|---|---|
| `M11.C.2` | **FAIL**, 100 seeds | the later symptom load of the child who received the most projection-type acts early, less the least: +0.29 ± 2.20 (p 0.39) | not needed |

- **`M11.C.2` fails.** Read over the same weeks it first passed (+1.73, then +1.31 ± 1.41 at p 0.045), but a
  same-window comparison cannot tell projection raising load from load attracting projection (review 2026-10-12).
  With acts counted in the first five years and load read in the last five, the most-projected child carries no more
  load (+0.29 ± 2.20, p 0.39). The cross-person term (revision 26) is built and tested on its own (`test_learner.py`), but no criterion yet
  shows it shaping a child's repertoire.

### Step 10 (part): criteria on existing mechanisms, and revision 27 (2026-10-12): Claude's recommendation, pending owner review

**Revision 27, borrowed pseudo-self settles back** (`M6.I.4`, `M1.A.5c`). `M11.C.24` (the whisper of nature) cannot
pass without it: nothing returned pseudo-self once the exchange stopped. On a tie the pair has exchanged on (its
functioning balance is not zero), whose over-functioning member carries borrowed pseudo-self and the other lent, and
between whom no over/under-functioning was sent for `exchange_window` weeks (8, `[I]`),
`pseudo_self_relaxation_rate` (0.05, `[I]`) of the smaller moves back each week, with functional level, conserved.
This changes every run in which pseudo-self is lent and the lending stops. *Review 2026-10-12:* as first built it
paired any opposite-signed tie, so pseudo-self moved between people who never exchanged; the balance condition
fixes that.

**`M10.B.5`, learning switched off by a declared override**: `scenario(..., overrides={"learning_rate": 0})`, never
a config default.

| Criterion | Fixture and arms | Runner's verdict | Readout | Its mutation |
|---|---|---|---|---|
| `M11.C.39` | triad; a 26-week spell on both spouses, then 52 calm weeks; learning off in the baseline | **PASS**, 100 seeds | `DISTANCE`/`CUTOFF` acts in the calm weeks +6.0 ± 1.1; time to remission +2.7 ± 2.6 weeks (reported) | learning off in both: **FAIL**, 0.0 |
| `M11.C.40` | Phase C family, its declared spell, 208 weeks; learning off in the baseline | **FAIL**, 150 seeds | clause 1, automatic entropy fall +0.027 ± 0.006 (holds); clause 2, the automatic difference less the self-directed one, −0.0005 ± 0.041 (p 0.49, does not hold); no empty windows | learning off in both: **FAIL**, 0.0 |
| `M11.C.43` | triad; F under a 26-week spell with M available (treatment) or absent from F's choices (baseline); M unavailable to F 13 weeks later | **FAIL**, 150 seeds | F's rise on losing M −0.54 ± 0.71 | not run |
| `M11.C.34` | nuclear four 10 below and 10 above the fixture's level; the same onset in C1 at week 26 | **FAIL**, 150 seeds | the others' automatic acts toward C1 over 8 weeks, onset less no onset, higher arm −0.04 ± 0.10 | not run |
| `M11.C.21` | nuclear four; both parents' functional level −10 at week 26 | **FAIL**, 100 seeds | C1's rise to the end of the run −0.52 ± 1.32 | not run |
| `M11.C.23` | nuclear four; C1's symptom remits at week 4; parents' `CONFLICT` with suppression 0.5 against `STAY-IN-CONTACT` without | **FAIL, reversed**, 150 seeds | relapses −0.29 ± 0.08 | not run |
| `M11.C.24` | dyad; A takes charge and B gives way every 2 weeks for 52 weeks, then the tie is dissolved (treatment) or the exchange goes on | **PASS**, 150 seeds | B's change +11.9 ± 0.4; A's −12.6 ± 0.5 | transfer zeroed: **FAIL**; revision 27's settling removed: **FAIL** (B −0.13 ± 0.26) |

- **Pass and proved by mutation:** `M11.C.39`, `M11.C.24`.
- **Fail:** `M11.C.40` (its second clause), `M11.C.43`, `M11.C.34`, `M11.C.21`, `M11.C.23` (reversed). Each is
  recorded with its numbers; nothing was changed to make any pass.
- **`M11.C.34` first passed (−0.18) on a readout that counted all the others' acts with no onset fired**; the review
  found both. Fired, and counted as reactions to C1 against a no-onset run, the levels do not differ.

**Why `M11.C.17`'s coached rise collapsed (trace, 2026-10-13; `c17_landing.py`, `trace_c17.py`, 20 seeds; ad hoc).**

| Code | Coached people's binders failing (share of weeks) | Coach contacts landed (of about 66) | Perspective at year 10 | C.17 coached rise (150 seeds) |
|---|---|---|---|---|
| Revision 23 | 99% | 19.3 | 0.63 | +0.021 ± 0.009 |
| Revision 24 (B7) | 43% | 8.0 | 0.34 | +0.0056 ± 0.0023 |
| Revision 27 | 38% | 7.6 | 0.32 | +0.0009 ± 0.0008 |

- A coach's contact lands only when the recipient's binders are failing: an active symptom, or symptom load near its
  threshold (`M1.E.7`, "pain is necessary").
- At revision 23 every person was symptomatic from the first weeks, the saturation revision 24 removed, so the coach's
  gate was open 99% of the time. C.17's revision 23 pass was carried by that saturation.
- With symptoms episodic, the gate is open about 40% of the time. Landings fall by more than half, so perspective
  reaches half the level. The coached people then have fewer self-directed weeks (0.31 against 0.43) and the
  estimator's evidence never turns positive: 3 of 40 coached people risen by year 10, against 16.
- Revision 24 accounts for about three quarters of the fall; revisions 25–27 (chiefly 27's settling back) for the rest.
- *Decided 2026-10-16 (review of revisions 24–30): it keeps reading an active symptom.* Was, for the owner: whether "pain is necessary" should read an active symptom as it does, or the person's load above
  their usual level (as the stress response does since revision 22). Nothing has been changed.

**Every Phase D criterion on the final code of revision 27** (`final25.out`, 2026-10-12; ad hoc):

| Criterion | Verdict | Readout (treatment − baseline) |
|---|---|---|
| `M11.C.18` | PASS | +0.015 ± 0.002 |
| `M11.C.20` | PASS | +0.015 ± 0.002 |
| `M11.C.33` | PASS | +0.016 ± 0.002 |
| `M11.C.17` | PASS, weak | coached +0.0009 ± 0.0008 (p 0.011); +0.021 ± 0.009 at revision 23 |
| `M11.C.30` | PASS | −21.4 ± 1.6 |
| `M11.C.22` | PASS | mild +0.81 ± 0.38; severe −1.03 ± 0.44 |
| `M11.C.31` | PASS | +22.1 ± 0.6 |
| `M11.C.11` | PASS | +2.63 ± 0.92 |
| `M11.C.12` | FAIL | −0.04 ± 0.24 |
| `M11.C.26`, `.36`, `.39`, `.24` | PASS | above |
| `M11.C.14`, `.2`, `.40`, `.43`, `.34`, `.21`, `.23` | FAIL | above |

- **`M11.C.17`'s coached rise nearly vanished** between revisions 23 and 27. Two probes: without revision 27's
  settling back it is +0.005 ± 0.002; with the symptom integrand read above the chronic floor again (before revision
  24) +0.0018 ± 0.0010. So several of tonight's changes together shrink it; none alone restores it. Recorded, not
  tuned (`c17probe.out`).

- **Not built this night: `M11.C.7`, `M11.C.9`, `M11.C.28`.** Each needs a mechanism first:
  - C.7: the coach's landing reads nothing of the coach's position (`M8`'s predicate exists but nothing calls it);
  - C.9: sibling position does not gate any behaviour (`M1.A.14c`);
  - C.28: the sink allocation feeds no behaviour, so pinning its mobility can change no symptom.


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

#### B10 as built (2026-10-12): Claude's recommendation, pending owner review
- `src/bowen/readouts/evaluation.py` emits components 1–8 of `M11.G.1` from a finished run, and never 9 or 10.
- Adaptiveness is derived against stress, and is undefined with no stress.
- Belief and truth are reported apart, and the framing travels with the evaluation.
- The multigenerational projection line is reported as not computable in part I.
- Tests: `tests/bowen/test_evaluation.py`, including `M11.D.20`'s ordering check against the run log.

### B11. Who acts
- Activation reads a declared life stage instead of selecting every living person (`activation.py:32`).
- Dr Halim's session schedule moves from `policy_source.py:55` into the family or the script.
- Needed in D-I so that fixture families can hold a member inactive.
- **Not built (2026-10-12).** No D-I criterion built so far needs a member held inactive. It is needed for the life
  course (D-II: children who do not yet act, the dead), so it moves there.

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
| C.17 | Phase C family with Dr Halim | `test_m11c17_coach_helps_and_is_not_required` | `systems_perspective` allowed to rise spontaneously | B6, P8, `M7.E.4` |
| C.18 | any | `test_m11c18_estimator_separates_cutoff_from_iposition` | the stress response's coping and soothing removed (`M1.A.5e`; restated at revision 20) | B6, B3 |
| C.20 | any | `test_m11c20_estimator_separates_borrowed_from_basic` | the estimator reads borrowed pseudo-self again, what was given way still added back (`M1.A.4k`; restated at revisions 20 and 23) | B6 |
| C.21 | nuclear four | `test_m11c21_reciprocity_inverts_on_projector_collapse` | `M6.I.4`'s transfer made one-way | — |
| C.22 | nuclear four | `test_m11c22_symptom_lock_in_is_non_monotone` | `M7.D.2a` flattened and `M7.D.2d` removed | B7 |
| C.23 | nuclear four | `test_m11c23_expressed_emotion_dominates_suppression` | the EE term zeroed | P7 |
| C.24 | dyad | `test_m11c24_whisper_of_nature_reverses_on_dissolution` | the `M6.I.4` transfer zeroed | — |
| C.26 | nuclear four | `test_m11c26_sink_recoverable_from_belief_configuration` | `M9.6`'s attribution write randomised | B8 |
| C.28 | nuclear four | `test_m11c28_sink_mobility_is_protective` | the mobility term pinned | P7 |
| C.30 | any | `test_m11c30_channels_are_mutually_protective` | `M7.D.2c`'s cross-channel term zeroed | B7 |
| C.31 | nuclear four, no stressor | `test_m11c31_binder_removal_releases_bound_anxiety` | the held anxiety discarded (`M1.F.9`) | — (mechanism built) |
| C.33 | dyad | `test_m11c33_dependence_gate_blocks_basic_level_rise` | `M7.A.1a`'s gate removed | B6 |
| C.34 | any | `test_m11c34_system_reaction_to_symptom_discriminates_level` | `M1.A.19`'s detectors made level-independent | B3, P3 |
| C.36 | triad | `test_m11c36_policy_reads_belief_not_true_state` | `M4.B.2` reads the true state | B8 |
| C.39 | triad | `test_m11c39_pattern_outlasts_spell_only_with_learning` | `M4.D.6` disabled in both arms | a test-declared override (`M10.B.5`) |
| C.40 | triad | `test_m11c40_leaked_reinforcement_raises_self_directed_entropy` | reinforcement reaching the self-directed channel | B3 |
| C.43 | triad | `test_m11c43_learned_buffer_removal_costs_more` | `M4.C.10`'s downward term removed | C.43's inputs reconciled with `M4.D.6d` and `M1.F.9` |
| `M11.D.20` | — | `test_m11d20_readout_ordering_matches_run_log_ordering` | — | B10 |

**Moved out of Phase D:** C.13. It has no community to host it (P7). It returns when one is modelled.

## 4. Part II (D-II): the life course — plan, 2026-10-13

> **Status: approved by the owner 2026-10-13 ("approved"); built the same day (§4.7).** It builds spec revision 28,
> which applies the owner's P13–P15 answers (§1b). The forms in revision 28 are Claude's recommendation, pending owner
> review; the steps below build them as written unless the owner changes them.

### 4.1 What part II must deliver
- **Criteria:** `M11.C.6`, `.10`, `.15`, `.37`, and `M11.C.2` on the twelve-person family. Each runs over an ensemble
  with its mutation run, as in part I.
- **Re-runs:** the `M16.T.3` re-run with the slow clock, Dr Halim and the belief channel active.
- **The 40-year run:** 2,080 weeks and 40 slow ticks of the twelve-person family that completes, with every invariant
  holding, carrying everything `M3.B.1` lists.
- **Carried from part I:** B11 (who acts), which the life course needs.

### 4.2 What exists, and what has to change first
- `RunState` holds one `Family` (`state.py:51`), and only the `ravi_marta` family has a budget and sinks
  (`family.py:238`). Partnering creates families, so the state needs a set of them, keyed by household.
- The people and ties are fixed at assembly. Births and partners add both during a run, so the people, ties, triads
  and the event store's indexes must accept additions. Identifiers come from `M1.A.20`, never a counter.
- Activation selects every living person (`activation.py:34`). It must read age (`M7.C.1f`).
- No `Person` carries an age. The loader reads ages (`family.py:144`) and drops them after checking the fixation age.

### 4.3 Build steps

Each step: tests first, then code, then the documents it changes (spec status, explainer, config docs, records).

| Step | What | Spec | Tests (each name carries the spec ID) |
|---|---|---|---|
| **L0** | Topology that can grow: several `Family` objects keyed by household; people, ties and triads added in a run; `Person.age_weeks`, advanced on the slow tick; `Person.earner` | `M1.D.1`, `M1.A.20`, `M7.C.1f` | `test_m1d1_each_household_couple_has_its_own_family`; `test_m1a20_a_birth_identifier_is_stable_across_arms`; `test_m7c1f_age_advances_on_the_slow_tick`; the golden run unchanged until L1 (`test_engine_golden.py`) |
| **L1** | The family data: the three starting-age settings and per-person gaps; Dr Halim female, 35; one coach tie, to the coachee (`M11.C.7`'s joint arm keeps both); the children's chronic anxiety derived and their levels at the parents' mean; Leo in the household; an `earner` column. The golden run is re-recorded here, once | `M2.A.0j`, `M2.A.0k`, `M2.B.2` | `test_m2a0j_ages_come_from_the_three_settings`; `test_m2a0j_changing_a_setting_changes_only_ages`; `test_m2b2_the_coach_is_tied_to_the_coachee_only`; `test_m2a_phase_d_family_matches_spec_table` (updated with the spec's Age column) |
| **L2** | Who acts (B11): activation from `acting_age`; younger children perceive, appraise and witness | `M7.C.1f`, `M3.E.1` | `test_m7c1f_a_child_under_ten_never_selects`; `test_m7c1f_a_child_under_ten_still_witnesses` |
| **L3** | Chronic anxiety before 12 | `M7.B.1a` | `test_m7b1a_child_chronic_reads_parents_and_witnessed_history`; `test_m7b1a_chronic_is_fixed_at_the_fixation_age` |
| **L4** | A child's development from 5 to 20 | `M7.E.1f`, `M7.E.1` | `test_m7e1f_level_held_at_parents_mean_before_five`; `test_m7e1f_parents_functioning_up_raises_the_level`; `test_m7e1f_level_never_leaves_two_percent_band` (adversarial: a long run of high functioning must stop at the band); `test_m7e1f_development_stops_at_twenty` |
| **L5** | Mortality and the disposition at death | `M7.C.1j`, `M6.3a`–`3g`, `M6.I.4` | `test_m7c1j_age_at_death_drawn_once_at_entry`; `test_m7c1j_hazard_rises_with_age`; `test_m63b_no_share_to_an_external_agent`; `test_m63b_falls_back_to_the_household`; `test_m63b_anxiety_split_by_bond_energy`; `test_m63c_tie_and_triangle_holdings_go_to_the_living`; `test_m63d_investment_redistributed_over_other_ties`; `test_m63e_balances_settle_back_after_a_death`; `test_m63f_sinks_naming_the_dead_return_to_the_family`; `test_m63g_grief_lasts_one_to_three_years_by_bond_energy`; `test_m11c37_anxiety_conserved_across_death` |
| **L6** | Lost income and `FIND_WORK` | `M6.3h`, `M7.C.1k` | `test_m63h_income_loss_lasts_until_someone_finds_work`; `test_m7c1k_find_work_leaves_twenty_percent_and_sets_earner`; `test_m7c1k_find_work_unavailable_without_a_loss`; `test_m7c1k_find_work_is_learned_not_scheduled` (across seeds, who takes it and in which week vary, and a member whose earlier goal-directed acts were relieved takes it sooner) |
| **L7** | Launch, partners from settings, births, new families | `M7.C.1g`–`1i`, `M7.C.1a`, `M6.4` | `test_m64_entry_and_new_budget_are_logged_sources`; `test_m7c1g_launch_moves_household_and_ends_dependence`; `test_m7c1h_partner_level_within_pairing_tolerance`; `test_m7c1h_partner_comes_only_from_settings`; `test_m7c1h_new_couple_gets_a_family_with_derived_budget`; `test_m7c1i_birth_ties_redistribute_investment`; `test_m7e1c_birth_situations_add_focus_not_a_target` |
| **L8** | The coach's retirement and successor | `M2.A.0k` | `test_m2a0k_successor_enters_at_retirement_and_learning_is_kept` |
| **L9** | The 40-year run: completes, invariants hold every tick, timed outside the suite; the `M16.T.3` re-run | `M3.B.1`, `M16.T.3` | `test_m3b1_forty_year_run_completes` (one seed, ensemble-marked); the `M16.T.3` test re-pointed |
| **L10** | The criteria, the D record and the completion report | §4.4 | §4.4 |

**Order and dependencies.** L0 → L1 → L2 → (L3, L4) → L5 → L6 → L7 → L8 → L9 → L10. L1 is the one change that alters the
golden run; every later step that changes a run is behind a config flag for the Phase C family and the fixtures, so
part I's records do not move. One regeneration of the part I record at the end, not one per step.

### 4.4 The criteria

| Criterion | Arms | Readout | Mutation that must turn it red |
|---|---|---|---|
| `M11.C.37` | one run with a death (an `M11.4` exception, like C.31) | total anxiety before and after `M6.3b`–`3c`, to `M6.1`'s tolerance | the dead person's anxiety discarded |
| `M11.C.15` | three arms from one state with spouse dysfunction carried by Marta: Marta dies; Marta recovers (her binder removed, `M1.F.9`); neither. Grief and lost income are set to zero in every arm (`M10.B.5`). *Amended 2026-10-13 by the owner ("keep the transfer in C.15, share at 1"): `M6.3b`'s transfer stays at 1; it was 0 at first (§4.7)* | the living members' summed excess above usual over the next 52 weeks; both treatment arms above the control | the bound anxiety discarded at the death |
| `M11.C.6` | the twelve-person family, 40 years | the change in basic level from parents to child by age 20, for the child who received the most projection-type acts against the others | each child's level fixed at the anchor (`M7.E.1f` removed) |
| `M11.C.10` | with the Ravi–Dario cutoff at `t0`; without it (the tie ordinary) | cut-off ties (rejection, `M1.B.3c`) in generation 3 by year 40 | `M1.D.8`'s family-of-origin input removed; and, separately, learning from witnessed cutoff removed |
| `M11.C.2` | the twelve-person family | as in part I: projection-type acts counted early, symptom load read late | the witness path (`M1.F.5`) |
| `M16.T.3` | — | the re-run with the slow clock, Dr Halim and the belief channel | — |

**Three limits, stated now so they are not found later:**
- **C.6 cannot see three generations in 40 years.** Generation 4 is born after the partner years (about year 20 on the
  draft rows) and is not 20 by year 40. Part II tests one transmission, generation 2 to 3, and reports generation 4's
  levels as a readout. Shape limit (a), a fall from top to bottom within about ten generations, **cannot be met under the ±2% band** (about 18 points in ten generations at most). This is an open owner decision (spec `M7.E.1f`), not a readout.
- **C.6's effect is small by the owner's rule.** The band is ±2% of the parents' level, about ±0.8 points at 40, so a
  child can end at most 0.8 points from its parents, and two siblings at most 1.6 points apart. The test asserts a direction, and the ensemble may need more
  seeds; the cost is measured at L9 before it is promised.
- **C.10's witnessed-cutoff learning has no mechanism** (§8). `M4.D.6` reinforces only the actor's own moves, and
  `M4.D.6e`'s cross-person term reads relief, not cutoff. The second mutation may have nothing to remove; if so, C.10 runs
  through `M1.D.8` alone and the report says so.

### 4.5 What the owner still decides, and the defaults part II builds meanwhile

| Point | Default built unless the owner says otherwise |
|---|---|
| Marta–Iris: "resolved, low contact" or ordinary | ordinary, as the config has it |
| The ±2% band against `M11.C.6`'s traversal limit (a): which one changes | neither; (a) is reported as failing by construction |
| A child's development: the rule in `M7.E.1f`, or the estimator bounded to the band | the `M7.E.1f` rule |
| Teodor's basic level | 36 (34 broke `M2.A.0e`'s pairing with Ana) |
| `M11.C.17`'s gate: load above usual, or an active symptom | an active symptom, as now (part I) |
| Launch age | 20 (`M7.C.1g`) |
| When people die | a Gompertz hazard, median near 80 (`M7.C.1j`) |
| The partner rows: who, when, how many children | drafted in `config/bowen/partners_phase_d.md` at L7 for the owner to edit: Leo pairs at 28, Nadia at 27, Pia at 29; two children each, at 2 and 5 years after pairing |
| Grief size, income-loss size, working age | 20, 15, 16 (all `[I]`) |
| The new coach's sex | female, as Dr Halim |

### 4.6 Compute
- A part I criterion runs on families of four or fewer for about 5 years. Part II's runs are 40 years of 11 to about 20
  people.
- B1 made the cost per tick flat. The cost of one 40-year run is measured at L9, and the ensemble budget for C.6, C.10
  and C.2 is stated from that measurement before those criteria run.

### 4.7 What was built, and what the criteria showed (2026-10-13)

**Build steps.** Every test named in §4.3 exists in `tests/bowen/test_life_course.py`, with two departures:
- **L6:** `test_m7c1k_find_work_is_learned_not_scheduled` was built as
  `test_m7c1k_find_work_follows_pressure_not_a_schedule`. Revision 28 (`M7.C.1k`) makes `FIND_WORK` a goal-directed act of
  the life course taken with a probability that rises with the member's load above their usual level, not a move the
  policy learns. Who takes it and when still varies across seeds and follows each member's state.
- **L9:** the `M16.T.3` re-run is a new test, `test_m16t3_sink_does_not_change_a_life_course_run`. It runs the twelve-person
  family for 22 years, through the first pairing and birth, with the sink on and off, and compares the final state
  including the life course, which `canonical_state` now holds.

L0 to L8 are done. For L9, the 40-year run completes with every invariant holding: `test_m3b1_forty_year_run_completes`
is ensemble-marked and covers one seed, and the three 40-year ensembles (300 seeds in all) completed with no crash. The
Phase C golden run and every Phase C and part I verdict are unchanged: `state.life` is None unless a family declares a
life course, and both regenerations changed only hashes and header lines.

**Departures from §4.4 in the criteria as built** (all declared in `src/bowen/ensemble/criteria_d2.py` before any ran):
- **`M11.C.15` runs on the nuclear-four fixture,** not on Marta in the twelve-person family. The mother carries a symptom
  held below lock-in. The read is the living others' mean excess over `after` weeks (`config/bowen/criteria.md`), not
  a sum over 52 weeks.
- **`M11.C.10` counts cut-off acts** sent by the third generation and their children, not cut-off ties at year 40.
- **Mutations:** `M11.C.37`'s is a test,
  `test_m11c37_mutation_discarding_the_dead_persons_anxiety_turns_it_red`, through the real `die`; `M11.C.15`'s was
  run ad hoc (below). The other three criteria fail, so they have no mutation to prove.

**Verdicts** (`docs/phase_d2_ensemble_record.md`, code hash `4dbf3f0f…2eab`, after the owner's `M11.C.15` decision below):

| Criterion | Verdict | Seeds | Result (treatment − baseline) | Proved by its mutation |
|---|---|---|---|---|
| `M11.C.37` | PASS (test, with its mutant) | — | conserved to `M6.1`'s tolerance | yes, a test |
| `M11.C.15` | PASS | 100 | death +1.25 ± 0.38 (p 2.1e-08); recovery +3.11 ± 0.35 (p 8.2e-18) | yes, ad hoc: death −2.03 ± 0.29 |
| `M11.C.6` | FAIL | 50 | +0.0177 ± 0.028 (p 0.894): the most-projected child's development is not lower | — |
| `M11.C.10` | FAIL | 100 | −12.6 ± 28 (p 0.795) | — |
| `M11.C.2` (twelve-person family) | FAIL | 150 | −0.13 ± 1.6 (p 0.456) | — |

**Compute** (wall time from the regeneration log, 10 cores): `M11.C.6`'s 50 seeds took 378 s, `M11.C.10`'s 100 seeds
1,518 s, and `M11.C.2`'s 150 seeds 1,084 s. The part II record took 50 minutes in all.

**`M11.C.15`: the first design, the diagnosis and the owner's decision.**
- **First design.** As first built, every arm set `death_transfer_share` to 0, so that only the release of what was
  bound acted. Then `die` discards the mother's own acute anxiety. What her symptom held is released by
  `release_bound`, as at a remission, so that part is the same in both treatments, and the death arm also loses her as
  an anxious member. The record at that design (code hash `9337f5d8…ffff`) read death +0.212 ± 0.35 (p 0.135), FAIL.
- **Diagnosis** (ad hoc, 30 seeds): the death rise was +1.17 (sd 2.0) with the share at 1, against +0.22 (sd 2.0) at 0.
- **Decision.** The owner, 2026-10-13: "keep the transfer in C.15, share at 1". Grief and lost income stay off in
  every arm (`NO_GRIEF_OR_INCOME` in `criteria_d2.py`), and `death_transfer_share` is no longer overridden by any
  criterion.
- **Mutation** (ad hoc, 100 seeds; script not kept in the repository). §4.4 names "the bound anxiety discarded at the
  death". The mutant zeroes the mother's symptom-bound anxiety and what her ties and triangles held just before `die`.
  The death limb went from +1.25 ± 0.38 to −2.03 ± 0.29 (95%), so it turns red. The unmutated run reproduced the
  record's number exactly. The recovery limb does not pass through `die`, so this mutant does not touch it.

**`M11.C.6`'s size.** Under the ±2% band a child can end at most about 0.8 points from the parents' level (§4.4). The
difference measured is 0.018 (p 0.89), and the test cannot tell the band from the absence of an effect. That is the
open band decision in §4.5.

**Diagnosis of `M11.C.10` and `M11.C.2`** (2026-10-13, ad hoc, not a record). Twenty seeds of the twelve-person family
over 40 years, both `M11.C.10` arms; the `M11.C.2` numbers come from the treatment arm, which is the family as declared.
A second run tracked the Ravi–Dario tie week by week for 10 years on 10 seeds. The scripts were not kept in the
repository.

*`M11.C.10`: the arms do not stay different, and two of the three links have no mechanism.*
- **The condition does not hold.** In the treatment the declared cutoff ended after 28 to 97 weeks (10 seeds). In the
  baseline the uncut tie was cut within 3 to 57 weeks, by week 14 in 8 of 10 seeds: Ravi and Dario cut each other on
  their own. Over 40 years the tie was cut for 16.9 years (sd 9.0) in the treatment and 17.45 (sd 6.4) in the baseline,
  and Ravi's mean excess was 21.18 against 21.21. After the first year or two the arms run the same process.
- **Links two and three of `M7.E.3`'s chain are not built.**
  - Link two: witnessing a cutoff teaches nothing. `M4.D.6` reinforces only the actor's own moves, and `M4.D.6e`'s
    cross-person term reads relief, not cutoff (§4.4 foresaw this).
  - Link three: nothing reads a family of origin's cutoff as an input to intensity. `M1.D.8` is built only as
    requiring the ties to exist (`scenario/family.py`). In the engine a cut-off state is read only on that tie
    (`observe.py`, `rejection.py`, `consolidate.py`).
  - The one path left runs through the parents' anxiety, and Ravi's anxiety does not differ between the arms.
- **The readout is mostly noise.** Generation 3 and their children sent 770 cut-off acts per run in the treatment
  (sd 84) and 782 in the baseline (sd 97). Most went to their own parents: in seed 0, Nadia cut Ravi 142 times. The
  paired difference ranged from −217 to +215. A cut-off act is a common short-term reaction here, not a lasting
  cutoff, so counting acts measures reactivity to the parents.

*`M11.C.2`: no child becomes the focus, and nothing carries early projection into later load.*
- **The parents spread projection evenly.** In the first 20 years the three children received on average 79.7 (Leo),
  83.0 (Nadia) and 79.7 (Pia) projection-type acts: `PURSUE`, `TRIANGLE` and `OVERFUNCTION` toward a child. The most
  projected child was Nadia in 10 seeds and Leo or Pia in 5 each, so the treatment picks a child largely by chance.
- **No lasting channel.**
  - Chronic anxiety (`M7.B.1a`) is recomputed each year from the parents' mean plus `witnessed_gain` × that year's
    load, and fixed at 12. It reflects only the year lived at 11, not the childhood. Its correlation with projection
    received was −0.26 (Leo), −0.04 (Nadia) and −0.01 (Pia).
  - The basic level at 20 was 39.31, 39.33 and 39.35: the ±2% band leaves no room.
  - A child's acute anxiety in childhood decays and does not reach the readout 20 years later.
- **Late load follows age.** Mean symptom load in years 20 to 40 was 24.9 (Leo, 29 to 49), 22.8 (Nadia) and 15.4 (Pia,
  21 to 41). The window catches each child at a different point in adult life: Leo pairs in year 19, Pia in year 28.
  Across all child-seeds, projection received and late load correlated at −0.13.
- Part I's `M11.C.2` also fails. Whether it fails for the same reasons was not checked here.

*What would make each testable* (owner decisions; nothing is changed):
- **`M11.C.10`:**
  - Hold the cutoff as a declared condition for the run, as `M11.C.15` holds a symptom.
  - Read lasting cut-off ties (`M1.B.3c` rejection) rather than acts. §4.4 named that readout; the build counted acts.
  - Decide whether to build link two or link three. Either is a spec change.
- **`M11.C.2`:**
  - Read each child's load over the same ages, not the same calendar years.
  - Decide whether chronic anxiety should integrate childhood load rather than take one year (`M7.B.1a`, a spec
    change).
  - Ask why the parents' anxious focus (`M1.A.11c`) does not settle on one child. This was not diagnosed here.

**`canonical_state`** now includes the life course. `test_m16t3_canonical_state_sees_the_life_course` shows that a
change to `state.life` alone changes it, so the `M16.T.3` re-run would see a sink that touched the life course.

### 4.8 Revision 29: cutoff and focus — plan, 2026-10-13

> **Status: approved by the owner ("i approve proceed"); built 2026-10-14. Results under "Built" at the end of this
> section.** It builds spec revision 29,
> which applies the owner's decisions after the diagnosis in §4.7:
> "hold the cutoff for the run and read tie states. But - add learning by witnessing cutoff and a place holder for the
> impact of cut-off a increasing anxiety (reactive cutoff / imaturity) or a principled cutoff for safety (abusive
> relationship for example). Child load is by age, chronic Anxiety builds overchildhood. Parents anxious focus CAN get
> set on one child e.g. child has some real or percieved weakness."
> The forms are Claude's recommendation, pending owner review.
>
> **Owner's answers on the defaults, 2026-10-13:** the gains okay; "no child named, it just evolves"; the other four
> situations not at `t0`, okay; "Principled cutoff should teach that this is good action (it's functioning up)";
> "Premise of witness cutoff is wrong. It depends on what the witness sees happen to the cutoff person and others"; the
> mutation rule okay; "builds over childhood - correct". `M4.D.6f` was rewritten to learn from what follows a
> witnessed cutoff, and the table below records each answer.

**What changes in every run, and what does not.**
- **Every run:** `M1.B.3d`'s reactive rise, `M4.D.6f`'s learning, `M1.B.8a`'s perceived weakness and `M4.D.1g`'s
  steering. Phase C has cutoffs and parents with children, so every golden run is re-recorded, and every Phase C and
  part I verdict may move. They are read after one regeneration and reported as information, as at B4. None is a gate.
- **Life-course runs only:** `M7.B.1a`'s childhood mean, and a declared `weakness` at `t0` (the reference family declares
  none).

**Build steps.** Each step: tests first, then code, then the documents it changes.

| Step | What | Spec | Tests (each name carries the spec ID) |
|---|---|---|---|
| **R1** | Chronic anxiety as the mean of every observed childhood year | `M7.B.1a` | `test_m7b1a_chronic_integrates_every_childhood_year` (adversarial: a child with a hard early childhood and a calm year at 11 ends higher than a child calm throughout); `test_m7b1a_a_child_present_at_t0_counts_from_t0` |
| **R2** | Reactive and principled cutoff: a `cutoff_kind` tie column (default reactive); the weekly reactive rise as a logged `M6.4` source; no rise and no "too little" deviation on a principled tie | `M1.B.3d` | `test_m1b3d_reactive_cutoff_raises_the_cutters_anxiety`; `test_m1b3d_a_less_differentiated_cutter_pays_more`; `test_m1b3d_principled_cutoff_raises_nothing_and_feels_no_loss`; `test_m1b3d_a_policy_cutoff_is_always_reactive`; `test_m1b3d_only_the_declared_member_is_exempt`; `test_m64_reactive_cutoff_rise_is_logged_by_name` (`M6.4`'s table, amended); `test_m4c1c_no_too_little_term_on_a_principled_cutters_tie` |
| **R3** | Learning from what follows a witnessed cutoff: a record per witness, read `witness_window` weeks later, weighted by the witness's bond to each person | `M4.D.6f` | `test_m4d6f_a_family_that_functions_better_after_a_cutoff_teaches_cutoff`; `test_m4d6f_a_family_that_functions_worse_teaches_against_it` (adversarial: the same witnessed cutoff, opposite outcomes, opposite learning); `test_m4d6f_a_principled_cutoff_that_lets_the_cutter_function_up_teaches_it_is_good`; `test_m4d6f_closer_ties_weigh_more`; `test_m4d6f_the_kind_is_not_read`; `test_m4d6f_a_child_at_home_witnesses_a_parents_cutoff` (the visibility check below) |
| **R4** | A child's declared `weakness`; perceived weakness while a parent is anxious; `M7.E.1c`'s situations at `t0` | `M1.B.8a`, `M7.E.1c` | `test_m1b8a_a_declared_weakness_adds_focus_at_t0`; `test_m1b8a_perceived_weakness_adds_focus_only_while_anxious`; `test_m1b8a_equal_children_get_equal_focus` (adversarial: no child is named); `test_m7e1c_a_declared_weakness_applies_at_t0`; `test_m1b8a_an_only_child_gains_no_perceived_focus` |
| **R5** | Focus steers the automatic channel's targets | `M4.D.1g` | `test_m4d1g_focus_raises_the_chance_an_act_lands_on_that_tie`; `test_m4d1g_the_self_channel_is_not_steered`; `test_m4d1g_withdrawing_kinds_are_not_steered` (adversarial: a cut tie with high investment draws no extra `CUTOFF`); `test_m1b8a_focus_raises_the_focused_childs_load` (the loop closes only with `M4.D.1g` on); golden runs re-recorded here, once |
| **R6** | The criteria: `M11.C.10` with a held cutoff, an open baseline and a tie-state readout; `M11.C.2` on the twelve-person family read by age | `M11.C.10`, `M11.C.2` | `test_m11c10_the_treatment_holds_the_cutoff_and_the_baseline_holds_it_open`; `test_m11c10_reads_cut_off_tie_weeks`; `test_m11c2_twelve_reads_load_by_age`. The mutations run ad hoc and are reported in this plan, as in part I |
| **R7** | One regeneration of every record; the mutations; the documents; one `learning-qa` pass and one correctness pass; the report | — | the records' staleness tests |

**Order.** R1 → R2 → R3 → R4 → R5 → R6 → R7. R2 to R5 change every run, so the regeneration waits for R6 and runs once.

**The criteria as amended.**

| Criterion | Arms | Readout | Mutation that must turn it red |
|---|---|---|---|
| `M11.C.10` (composite) | treatment: the Ravi–Dario cutoff held for the run, renewed by a scripted reactive `CUTOFF` every `renew_every` weeks (a setting shorter than the fade below the refusal threshold); baseline: the tie ordinary and held open for the run | cut-off tie-weeks held by generation 3 and the children born to them in the run | `M4.D.6f` and `M1.B.3d`'s reactive rise both removed; each alone reported |
| `M11.C.2` (twelve-person family) | as now: the child who received the most projection-type acts against the one who received the fewest | acts counted per year under `projection_until_age` (20); load as the mean over ages `load_from_age` to `load_to_age` (20 to 30) | the witness path (`M1.F.5`); and `M1.B.8a` with `M4.D.1g` removed |

`M11.C.6` and `M11.C.15` are rerun unchanged in the regeneration. `M11.C.6` reads chronic anxiety's new form through
development.

**Risks, stated now.**
- **A parent's cutoff may not reach the children as a witnessed event.** Visibility (`M1.F.1b`) decides who witnesses a
  `CUTOFF` from Ravi to Dario, who lives elsewhere. R3's last test checks it. If the children at home do not witness
  it, I stop and report before changing visibility.
- **Focus could lock onto one child in every family.** `M1.B.8a` and `M4.D.1g` feed each other. The tests show focus
  stays equal between equal children. After R5, the Phase C family's acts per tie are compared before and after, and
  any shift is reported.
- **Every Phase C and part I verdict may move** (above). If one flips, it is reported, not tuned.
- **`M11.C.10` may come out reversed.** A held reactive cutoff costs the cutter (`M1.B.3d`), and if the family functions
  worse after each renewal, the children learn against cutoff. That would be a finding about the forms, reported as
  such.
- **`renew_every` sets how often the children witness a cutoff.** The `M11.C.10` effect through `M4.D.6f` scales with
  it, so its value is stated with the result.
- **A child who learns that cutoff is good learns `CUTOFF`, which is reactive when they select it.** Only a declared
  cutoff is principled. The spec states the gap (`M4.D.6f`).

**What the owner still decides, and the defaults built meanwhile.**

| Point | Default built unless the owner says otherwise |
|---|---|
| `reactive_cutoff_gain`, and scaling it by immaturity, (100 − `basic_level`) ÷ 100 | *Accepted by the owner:* 0.1 a week at full rejection, scaled (`[I]`); the excess it adds over a year of held cutoff is measured at R2 and reported |
| Does a principled cutoff teach a witnessing child? | *Decided by the owner:* yes, "that this is good action (it's functioning up)". `M4.D.6f` reads what follows, so it teaches through the cutter functioning up |
| What a witness learns from (`M4.D.6f`) | *Decided by the owner:* "what the witness sees happen to the cutoff person and others". Built as the change in functional level of the cutter, the person cut off and the others the witness is tied to, over `witness_window` (13 weeks, `[I]`), weighted by the witness's bond to each. Functioning is read as it is, not through belief |
| Toward whom the learned `CUTOFF` value moves | everyone the witness holds a tie to |
| `witnessed_cutoff_gain`, `perceived_weakness_gain`, `focus_steer_gain` | *Accepted by the owner:* 0.05 (per point of functional change), 0.01, 1.0 (`[I]`); each effect is measured at its step and reported |
| `M11.C.10`'s baseline | the tie held open for the run, rather than left to the run |
| `M11.C.10`'s mutation | *Accepted by the owner:* both paths removed must turn it red; each alone reported |
| A declared weakness in the reference family | *Decided by the owner:* none, "it just evolves" |
| Perceived weakness reads the children's true functioning, not a parent's belief (`M9`) | true functioning |
| `M11.C.2`'s ages | projection under 20; load over ages 20 to 30 |
| `M7.E.1c`'s other four situations (firstborn, firstborn of a sex, high stress, last-born) at `t0` | *Accepted by the owner:* not applied: no history before `t0` is declared. Applying them would raise the initial focus on Leo and Pia |
| Who holds a principled cutoff | one named member of a tie; the other still feels the cut |
| Every policy-selected `CUTOFF` is reactive | yes |
| Which acts focus steers (`M4.D.1g`) | the `projection` row of `config/bowen/pattern_acts.md` (`OVERFUNCTION`, `PURSUE`, `TRIANGLE`); withdrawing kinds are not steered |
| Chronic anxiety "builds over childhood" | *Confirmed by the owner:* the mean of the yearly loads over childhood, not a running sum: a sum would rise with the years observed, so a child present at `t0` aged 9 would get less than one born in the run |
| `renew_every` | 13 weeks (four renewals a year), shorter than the fade below the refusal threshold; checked at R6 |

**Compute.** One full regeneration took about 1 hour 50 minutes on 10 cores (twice today). Each `M11.C.10` mutant is
about 25 minutes (100 seeds, two arms, 40 years). Total about 3 hours, plus the build.

**Built, 2026-10-14.** Steps R1–R7 are built. The records were regenerated once after the review fixes (code hash
`eeef3517…`). Every number below comes from a generated record or is marked ad hoc.

*The criteria* (`docs/phase_d2_ensemble_record.md`; ± is the half-width, p after Holm):

| Criterion | Verdict | Result (treatment − baseline) | Mutations (ad hoc, 95% half-widths) |
|---|---|---|---|
| `M11.C.2` (twelve-person family, by age) | **PASS** (150 seeds) | `symptom_load` +1.31 ± 0.95, p 0.005 | witness path off (`M1.F.5`) +0.27 ± 0.75; `M1.B.8a` and `M4.D.1g` off +0.45 ± 0.99 (150 seeds each). Neither interval excludes zero, so each removal loses the effect. Run ad hoc, not through the runner's Holm test |
| `M11.C.10` (held cutoff) | **FAIL** (100 seeds) | `cut_off_tie_weeks` −63.1 ± 200, p 0.71 | both paths off +104.8 ± 192.7; `M4.D.6f` off +37.7 ± 181.2; `M1.B.3d`'s reactive rise off +65.7 ± 169.9 (100 seeds each). None differs from the record or from zero: neither path has a detectable effect on this readout |
| `M11.C.6` | **FAIL** (100 seeds) | `development` +0.0301 ± 0.024, p 0.98 | — |
| `M11.C.15` | **PASS** (100 seeds) | `death_rise` +1.34 ± 0.40, p 1.3e-08; `recovery_rise` +3.37 ± 0.29 | unchanged from §4.7 |

- **`M11.C.10`'s treatment held the cutoff while both ends lived.** 1,866 of 16,000 scheduled renewals were not made:
  860 because the cutter had died, 983 "not legal", 22 owed a step and 1 rewritten. On seed 2 (checked ad hoc, one seed), all 12
  "not legal" weeks came after Dario's death: the forced source labels a dead *target* "not legal" (TODO).
- **Other verdicts that moved** (information, not gates): `M11.C.44` FAIL → PASS (`outside_inside_ratio` +0.152 ±
  0.093, p 0.0047); `M11.C.17` PASS → FAIL (`coached_rise` +1.43e-05 ± 0.00029, p 0.23, from +0.000881, p 0.011). In
  the mutation record no proof was lost: `M11.C.44` gains `one-sided-deviation` (red) and `deviation-inverted`
  (survived); `steepness-level-independent` now turns `M11.C.41[heavy]` red; `c16-grounds-removed` moves from red to
  reversed.

*The review fixes* (one `learning-qa` pass, one correctness pass, one narrow re-sweep of the fixes; each fix has a test):
- **A declared weakness acts through the anxious parent only.** It adds `weakness` × 100 to the weekly shortfall, from
  `t0`, rather than to `M7.E.1c`'s birth situations, which multiply investment that is zero at birth (spec `M1.B.8a`,
  `M7.E.1c` aligned). The pressure is the load *above usual*, as the spec now says.
- **The principled kind belongs to the declared rejection.** It ends when that rejection fades to zero, at a reunion,
  or when the holder's own policy cuts off on the tie (`rejection.end_principled`).
- **Focus steering keeps each projection kind's total weight** (`M4.D.1g`). As first built it also raised it, which
  `M11.C.2` and `M11.C.6` count. *Measured (ad hoc, Phase C, 20 seeds, 104 weeks):* the more-projected child's share
  was 0.580 off, 0.626 with the first form, 0.578 after the fix. So `M11.C.2`'s pass does not come from steering
  raising the rate of projection acts.
- **One key format** for learned values (`src/bowen/engine/value_keys.py`), used by the policy, the learner and the
  `I-POSITION` abort.
- **The five new constants are range-checked**; `M11.C.10`'s skipped renewals and the weeks the cutter held rejection
  are counted in the arm's bookkeeping.
- **Two part II tests were tightened**, not loosened: `test_m63b_no_share_to_an_external_agent` isolates `M6.3b`
  (`M6.3c` lets an external agent receive released stocks, an owner question in TODO), and
  `test_m63e_balances_settle_back_after_a_death` reads the tie itself; its setup had set no balance.

*`I-POSITION`.* With steering on, `test_m5d8_success_usually_follows_several_failures` fails in the triad ensemble
(0 completions against 3 in 192 seeds, within chance). Steering is switched off there by a declared override; the
statistics rest on a handful of completions (TODO).

### 4.9 After revision 29: why `M11.C.10` still fails, and what comes next — 2026-10-14

> **Status: diagnosis done (ad hoc); the proposals below await the owner's decisions. No code until approved.**

**Diagnosis of `M11.C.10`** (ad hoc, 20 seeds per arm, 40 years; script outside the repo, not a record):

| | Baseline (held open) | Treatment (cutoff held) |
|---|---|---|
| Readout: cut-off tie-weeks, mean (sd) | 8,580 (925) | 8,550 (865) |
| … generation 3 rejecting their own parents | 4,671 | 4,681 |
| … rejecting their own children | 1,457 | 1,428 |
| … partners and partners' kin | 1,163 | 1,181 |
| Generation 3's own `CUTOFF` acts per run | 871 | 902 |
| Witnessed-cutoff learning updates per run (rows written) | 9,389 | 11,027 |
| Children's mean learned `CUTOFF` value, years 10 / 20 / 40 | −0.66 / −0.63 / −0.24 | −0.42 / −0.90 / −0.30 |

Paired difference, treatment − baseline: −31 (sd 1,179).

1. **The treatment is a small addition to a large stream.** The children witness the family's own cutoffs throughout
   the run: about 9,400 learning updates without the treatment. The held cutoff adds about 1,600, roughly a sixth more.
2. **What the children learn from witnessing is against cutoff, in both arms.** The family functions worse after a
   cutoff, so `M4.D.6f` lowers the value (−0.2 to −0.9). The extra witnessing pushes the treatment the wrong way for
   `M11.C.10`, which is the reversal the plan named as a risk.
3. **The learned value is not what drives how often the children cut off.** Their values are negative, yet they make
   about 900 cutoffs a run, a few more in the treatment (902 against 871). Cutoff follows load and the reconnection
   pull more than learning.
4. **The reactive cost reaches the children weakly.** It raises the cutter's anxiety (about +1 point of excess over
   5 years, §8.2 of the explainer), and a child's chronic anxiety integrates the family's load only until 12.
   Leo, Nadia and Pia are 9, 4 and 1 at `t0`.
5. **The readout is mostly noise for this question.** More than half of it is generation 3 rejecting their own
   parents, at about the same level in both arms, with a run-to-run sd near 900. At 100 seeds the test can see about
   ±200 tie-weeks; at 400 seeds about ±115.

In short: the model has no path by which a parent's cutoff makes a child more likely to cut off. Witnessing teaches the
opposite, and the cost path ends when the child turns 12. The corpus's mechanism is `M1.D.8`: the degree of cutoff in
the family of origin raises the intensity of the next generation's relationships. That is not built as a direct input.

**Options for `M11.C.10`** (the owner decides; none is built):
- **(a) Build `M1.D.8`'s input.** The degree of cutoff a person grew up with, read from the rejection their parents
  held over their childhood, raises their intensity (`[I]` gain). It is a starting condition, not a written outcome:
  how much cutoff follows is still left to the run. *Recommended.*
- **(b) A stronger treatment:** the parents cut off from their own parents (generation 1), the family-of-origin
  cutoff Bowen describes, instead of Ravi from his brother.
- **(c) A readout of generation 3's own families only** (their partners and children), leaving out their rejection of
  their parents, which is half the readout's noise. This changes the criterion after seeing it fail, so it needs the
  owner's reason, not mine.
- **(d) More seeds** (400). This narrows the test but cannot make a missing mechanism appear.

**Point 4, a self-chosen principled cutoff.** The owner (2026-10-14): "refusing contact with a person that is abusive
or is a serious threat (criminal, drug addict) where contact is very risky for one's safety." The engine has no notion
of physical danger; `M5.F.1`'s "assault" is an intrusive move, not violence. Proposed (spec revision 30):
- **`danger`**, a person field from 0 to 1, declared in the family config (optional, like `weakness`). The engine
  never infers it.
- **Contact with a dangerous person costs safety:** each accepted move from them adds `danger_gain` × `danger` to the
  recipient's acute anxiety (`[I]`), logged as a `M6.4` source. Cutting them off removes the cost at once, so learning
  that cutoff relieves follows from the run, not from a rule.
- **A cutoff toward a dangerous person is principled in proportion:** the reactive cost and the "too little" term are
  multiplied by (1 − `danger`). At `danger` 1 it is fully principled, at 0 fully reactive. This replaces the
  config-only `principled_cutoffs` key as the way a person chooses it; the declared key stays for a cutoff made before
  `t0`.
- Tests: an adversarial pair (the same cutoff toward a dangerous and a safe person); the cost of contact; learning
  that follows; the reference families declare no danger, so no record moves.

**Point 5, the open items to complete.** From TODO, in order:

| # | Item | Kind |
|---|---|---|
| 1 | `M11.C.10`: option (a), (b), (c) or (d) | owner decision, then build |
| 2 | Point 4: `danger` and the self-chosen principled cutoff | build after approval |
| 3 | `M11.C.17` (coaching) flipped to FAIL at revision 29: diagnose | diagnosis |
| 4 | `M11.C.6` and the ±2% band | owner decision (open since 2026-10-13) |
| 5 | `I-POSITION` tests: raise the triad seeds so they run with steering on | build |
| 6 | A scripted act toward a dead target labelled "dead", not "not legal" | build |
| 7 | Range checks on the revision 28 life-course constants | build |
| 8 | `acts_per_year` for a child who dies young; `_pick` counting the children it drops | build |
| 9 | Focus steering's size (`focus_steer_gain` 1.0 has no measurable effect in Phase C) | *decided 2026-10-16 (review of revisions 24–30): stays at 1* |
| 10 | Part II owner points: Marta–Iris, Teodor's level 36, `M11.C.17`'s gate, the plan §4.5 defaults | owner decisions |

Items 5 to 8 change the engine hash, so they are batched with 1 and 2 before one regeneration.

### 4.10 Revision 30: origin cutoff, danger, and focus that locks in — plan, 2026-10-14

> **Status: approved by the owner 2026-10-14 ("approved, partner and own parents, b as second variant"); built
> 2026-10-14 with the owner's two choices on danger and focus (below, "Built").**
> The spec text is written in the build's first step and committed with the code. The forms are Claude's recommendation, pending owner review.

**The owner's decisions (2026-10-14), quoted.**
- `M11.C.10`: "a parent cutting off represents a lower level of differentiation and a higher level of intensity in a
  system. Both of these should bear on the children to lower their level of functioning and increase their tendency
  to fuse (pursuit a lot). a) yes b) that's a different arm c) It applies to one's partner's and parents, less to
  one's children. d) seeds are expensive - 200 instead."
- Danger: "Agreed - Agents, like people, can becoming demanding to the point of being abusive, either because of
  tremendous stress or low level of differentiation. this might need a seed with a low level of differentiation and
  higher stressors."
- Focus: "In a family focus can get stuck onto mainly one child, once it starts, it tends to grow."
- `M6.3c`: "okay" (recorded in the spec, 2026-10-14).

**Measured before planning** (ad hoc, 20 seeds, the twelve-person family's first 20 years). The top child's share of
the parents' projection-type acts rises from 0.41 to 0.50 with steering off and from 0.40 to 0.54 with it on. Over
the same years the top child's share of the parents' *attention* falls from 0.62 to 0.47. Part of the rise in acts is
mechanical: Leo leaves home at year 11. So focus does not lock in now. Attention grows only from what a parent
appraises on a tie (`M1.B.8`), not from the parent's own anxious acts, so nothing closes the loop.

**What is built.**

| Step | What | Spec (new or amended) | Tests (each name carries the spec ID) |
|---|---|---|---|
| **S1** | **Origin cutoff as an input** (`M1.D.8`, option a). For each child, the *origin cutoff* is the mean, over the weeks from birth (or `t0`) to `development_end_age` (20), of the largest rejection either living parent holds on any tie. At 20 it sets two inputs for the rest of their life: chronic anxiety rises by `origin_cutoff_anxiety_gain` × origin cutoff (intensity), and the person's contact optimum on every tie is multiplied by 1 + `origin_fusion_gain` × origin cutoff (fusion). Functioning falls through the added load, as for any load, and pursuit follows from the larger "too little" gap; neither is written. | `M1.D.8a` (new) | `test_m1d8a_a_child_of_a_cutting_off_parent_carries_more_chronic_anxiety`; `test_m1d8a_and_seeks_more_closeness` (adversarial: the same family with the parent's rejection at zero gives neither); `test_m1d8a_read_from_the_parents_held_rejection_not_from_a_label`; `test_m1d8a_set_once_at_twenty` |
| **S2** | **`M11.C.10`'s readout and seeds** (options c, d). The readout becomes the tie-weeks on which generation 3 hold rejection toward their partner or their own parents. Rejection of their own children is reported, not tested. Seeds: 200, a per-criterion setting. | `M11.C.10` amended | `test_m11c10_reads_rejection_toward_partners_and_parents`; `test_m11c10_runs_200_seeds` |
| **S3** | **"A different arm"** (option b, as read): a second variant, `M11.C.10[family of origin]`, in which both parents' cutoffs from their own parents are held for the run, against the same ties held open. Reported with its own verdict. | `M11.C.10` amended | `test_m11c10_family_of_origin_arm_holds_both_parents_cutoffs` |
| **S4** | **Danger, declared or read from behaviour** (`M1.B.3e`, new). A person's *felt danger* from another is a slow average (`danger_window` weeks) of the impingement they receive from that person above `danger_threshold`, scaled 0 to 1. A declared `danger` (family config, optional) sets a floor. A demanding, intrusive person is a low-level or heavily stressed one by the existing mechanisms, so danger emerges from stress and level; nothing names an abuser. Accepted contact from that person costs `danger_gain` × felt danger in acute anxiety (an `M6.4` source). A cutoff toward them is principled in proportion: `M1.B.3d`'s reactive cost and `M4.C.1c`'s "too little" term are multiplied by (1 − felt danger). | `M1.B.3e` (new), `M1.B.3d` amended | `test_m1b3e_felt_danger_rises_with_sustained_intrusion`; `test_m1b3e_contact_with_a_dangerous_person_costs_the_recipient`; `test_m1b3e_a_cutoff_toward_danger_is_principled_in_proportion` (adversarial: the same cutoff toward a safe person pays in full); `test_m1b3e_danger_is_never_named` |
| **S5** | **The danger fixture** (the owner's "seed"): a small family in `config/bowen/fixtures/` with one person at a low level under repeated stressors, and its matched arm at a higher level with fewer stressors. Acceptance direction: felt danger toward that person, and principled cutoff of them, are higher in the low-level, stressed arm. A premise check: it confirms the code, not a finding. | `M11.C.47` (new, premise) | `test_m11c47_low_level_and_stress_breed_danger` (ensemble); its mutation: the impingement read removed |
| **S6** | **Focus that locks in** (`M1.B.8b`, new). Each projection-type act a person sends adds `focus_reinforce_gain` × the act's intensity ÷ `intensity_scale` to their investment on that tie. With `M4.D.1g`'s steering this closes the loop: attention brings acts, and acts bring attention. Noise decides which child it settles on; no child is named. | `M1.B.8b` (new) | `test_m1b8b_a_projection_act_adds_to_the_senders_attention`; `test_m1b8b_focus_concentrates_and_grows` (ensemble, measured among the children still at home so leaving home is not counted); `test_m1b8b_no_child_is_favoured_across_seeds` (adversarial: each child is the focus in some seeds) |
| **S7** | **The batch fixes** (plan §4.9, items 5–8): `I-POSITION`'s triad seeds raised until the tests pass with steering on; a scripted act toward a dead target labelled "dead"; range checks on the revision 28 constants; `acts_per_year` for a child who dies young, and `_pick` counting what it drops. Also diagnose `M11.C.17`'s flip (a report, no code). | — | each with its own test |
| **S8** | One regeneration, the mutations (`M11.C.10`: `M1.D.8a` removed must turn it red, and each of its two inputs alone is reported; `M11.C.2`: `M1.B.8b` removed is reported), the documents, one `learning-qa` pass and one correctness pass, the report | — | the records' staleness tests |

**Order.** S1 → S2 → S3 → S4 → S5 → S6 → S7 → S8. S1, S4, S6 and S7 change every run, so the regeneration waits.

**Defaults built unless the owner says otherwise** (all `[I]`):

| Point | Default |
|---|---|
| Origin cutoff window | birth (or `t0`) to 20, the years at home |
| `origin_cutoff_anxiety_gain`, `origin_fusion_gain` | 10 points of chronic anxiety and +50% optimum at an origin cutoff of 1; measured at S1 and reported |
| Generation 2 at `t0` | origin cutoff 0 unless declared (optional family metadata), since their childhood is not simulated |
| `M11.C.10`'s readout: "one's partner's and parents" | *Decided by the owner:* "partner and own parents" |
| "That's a different arm" | *Decided by the owner:* "b as second variant" |
| `danger_window`, `danger_threshold`, `danger_gain` | 26 weeks; the impingement of a `CONFLICT` at policy intensity; 2 points per accepted move at danger 1 |
| `focus_reinforce_gain` | sized at S6 so that, with no other change, the top child's attention share grows over the years at home; reported |

**Risks.**
- S1 and S6 change every life-course run, so `M11.C.2`, `M11.C.6` and `M11.C.15` may move. They are reported, not
  tuned.
- S6 could lock focus in every family too hard, or onto a child for no reason the run shows. The adversarial test
  checks that no child is favoured across seeds.
- S4 adds a source of anxiety to every run in which someone is intrusive enough. The reference families declare no
  danger, so any change comes from behaviour, and it is reported.
- `M11.C.10` may still fail. S1 is the corpus's mechanism, so a failure would be a finding about the sizes or the run
  length.

**Compute.** One regeneration (about 2 hours). `M11.C.10` at 200 seeds is about 1 hour per variant, two variants.
Its mutations at 200 seeds, three runs, about 3 hours. About 7 hours of runs in all, plus the build.

**Built, 2026-10-14: what the build found, and the owner's two choices.** All numbers ad hoc (scripts outside the
repo), unless they come from a record.
- **Danger did not emerge as planned (S4, S5).** The plan assumed a low-level or stressed person already intrudes more.
  Measured on the danger fixture (10 seeds, 5 years): the peak 26-week intrusion was 0.14 to 0.17 whether the couple
  was at level 50 and calm or at level 15 under a stressor every 2 weeks, because every act had the same force, one a
  week. *The owner chose, 2026-10-14:* "Force rises with anxiety" — `M4.D.1h`. Unbounded, force and the other's
  anxiety fed each other and a run diverged within 23 weeks, so it saturates. Read from anxiety above usual it barely
  moved, because the usual level follows a chronically stressed person up; so it reads excess. At `force_anxiety_gain`
  2: felt danger 0.003 (calm, level 50) against 0.125 (level 15, stressed); at 4 the calm couple reached 0.046, so 2 is
  built. `M11.C.47` passed at that setting in the build, but with both spouses lowered and danger read across the
  whole household; the correctness review found that against S5 and the spec (one person, danger felt from them). With
  one spouse lowered and stressed (8 seeds, ad hoc): felt danger from them +0.036 ± 0.082 (sd) over the baseline,
  mostly the child's; their partner's alone was +0.001. The criterion now reads danger felt from that person by anyone
  else at home.
- **Felt danger faded once a cutoff worked** (correctness review): refused moves are never received, so the cutoff
  turned reactive by itself within about 26 weeks. *The owner decided, 2026-10-14:* "hold felt danger while rejection is
  held". Built in `M1.B.3e`.
- **Focus did not lock in under S6's form.** Attention from one's own projection acts alone changed nothing at any
  setting tried (steering 1 to 20, reinforcement 0 to 5; 20 seeds, 10 years): attention leaks 10% a week, and a
  parent's one act a week is spread over about eight ties. *The owner chose, 2026-10-14:* "Anxious focus compounds" —
  added to `M1.B.8b`. At `focus_compound_gain` 2 the top child's share of attention was 0.44, 0.85 and 0.76 in years
  1, 3 and 9, against 0.41, 0.62 and 0.54 without it; at 5 and above one child took all attention within 3 years, so 2
  is built. *Then (learning-qa):* that form had no ceiling — investment 14,000 median, 4 million max over 10 years
  (20 seeds, the four fixture, children alike, ad hoc). *The owner:* "saturate the investment, pick a sensible
  constant". A ceiling of 25 to 100, on each tie or on the parent's total, bounded it but left the top share at 0.59
  to 0.67, as without compounding (0.61): the lock-in had come from the runaway. *The owner chose* "Shift, gain 10":
  the term moves the parent's attention among the children toward the larger share and adds none. Built at 10: share
  0.87, 0.84, 0.70 and 0.69 in years 1, 3, 5 and 10; peak investment 51 median, 94 max. No new constant.
  `focus_reinforce_gain` is 1. The acts follow only weakly at steering 1 (act share 0.46 against 0.56 at 20),
  so `focus_steer_gain` is left at the owner-accepted 1 and reported.
- **`M11.C.17` (S7).** Run on the revision 29 code (150 seeds): switching off any one revision 29 gain leaves the
  coached rise near zero; switching off all four reproduces revision 28's +0.000881 ± 0.000828 exactly. No single
  mechanism flipped it. The pass was fragile: under a thousandth of a level point, its interval barely clear of zero.
- **`I-POSITION` (S7).** With steering on and `M4.D.1h`, the triad's mover (level 40) completed 1 sequence in 192 seeds
  and failed PREPARE 11,725 times. The tests are about the sequence, so the mover is set at level 60: 27 completions.
  The steering override is gone. The mode assertion in `test_m5d8_success_usually_follows_several_failures` was
  removed: first successes spread over attempts 1 to 62 and the only repeated value was 1, so the mode measured ties.
- **`M11.C.10[family of origin]`.** A fresh scripted cutoff fades out in about 10 weeks, before a 13-week renewal, so
  the variant starts each held cutoff at `declared_rejection`, as the family declares its own.

**Results** (records regenerated 2026-10-14, code hash `c41fea60…`; numbers from `docs/phase_d2_ensemble_record.md`,
`docs/phase_d_ensemble_record.md` and `docs/phase_c_ensemble_record.md`):

| Criterion | Verdict | Readout, treatment − baseline (95%) | Seeds |
|---|---|---|---|
| `M11.C.10` | FAIL | `cut_off_tie_weeks` +6.64 ± 130, p 0.43 | 200 |
| `M11.C.10[family of origin]` | **PASS** | `cut_off_tie_weeks` +162 ± 120, p 0.007 | 200 |
| `M11.C.47` (premise) | PASS | `felt_danger` +0.050 ± 0.017; `principled_cutoff` +0.020 ± 0.008; `held_weeks` −16.6 ± 4.9 (reported) | 150 |
| `M11.C.2[twelve-person family]` | FAIL | `symptom_load` +0.70 ± 1.5, p 0.16 (passed at revision 29, +1.31 ± 0.95) | 150 |
| `M11.C.17` | PASS | `coached_rise` +0.00053 ± 0.00063, p 0.013 | 150 |
| `M11.C.44` | FAIL | `outside_inside_ratio` −0.030 ± 0.11 (passed at revision 29) | 100 |

- The record prints both `M11.C.10` rows under the one id; the second row is the family-of-origin variant (TODO).
- In the variant, 35,688 of 128,000 scripted renewals were toward a dead grandparent ("target dead").
- `M11.C.47`: the low-level, stressed person is felt as more dangerous and the cutoff toward them is more principled,
  but they are cut off on fewer weeks than the calm baseline person. Reported, not explained yet.
- Mutations: `M11.C.47` with the intrusion read removed reads 0 in every seed (100 seeds, ad hoc). The variant's
  `M1.D.8a`-removed mutation (run 2026-10-15) does **not** turn it red: +240 ± 116 tie-weeks (`child_cut_weeks` +49.7 ± 44.1; 200 seeds, ad hoc, 2026-10-15), against +162 ± 120
  with it; the intervals overlap. The variant's pass is therefore not proved by its mutation and is not evidence for
  `M1.D.8a`. The other candidates (run 2026-10-15): `M4.D.6f` removed (`witnessed_cutoff_gain` 0) gives −75.9 ± 132 tie-weeks; `M1.B.3d`'s reactive rise removed (`reactive_cutoff_gain` 0) gives +163 ± 117; `M1.B.3e`'s danger removed (felt danger from the declared floor only, 0 in this family) gives +102 ± 126 (`child_cut_weeks` +48.4 ± 45.3, +37.3 ± 46.7, +18.4 ± 42.3; 200 seeds each, ad hoc, 2026-10-15). Only `M4.D.6f` removed puts the mean below zero, so learning from the witnessed renewals carries the pass. The owner restated the variant's clause to `M4.D.6f` (2026-10-15), and the mutant is now in `docs/phase_d_mutation_record.md` (`tools/phase_d_mutation_record.py`): **red**, FAIL, p 0.91, 200 seeds. `M11.C.10` and `M11.C.2` fail, so their mutations are moot. *`M11.C.2[twelve-person family]` traced 2026-10-15*, each revision 30 change removed alone: unmodified +0.70 ± 1.5, p 0.16 (reproduces the record); `M4.D.1h`'s force removed +0.70 ± 0.97, p 0.13; `M1.B.8b`'s compounding removed +1.12 ± 1.7, p 0.09; `M1.B.8b`'s reinforcement removed +1.11 ± 1.6, p 0.093; `M1.D.8a` removed +0.96 ± 1.5, p 0.14; `M1.B.3e`'s danger removed +0.08 ± 1.8, p 0.46 (`symptom_load`, seeds 0–149, one-sided signed-rank with the runner's margin; ad hoc, 2026-10-15). No single removal brings the pass back.

### 4.11 Revision 31: apply the owner's review of revisions 24–30 — plan, 2026-10-16

> **Status: approved by the owner 2026-10-16 ("I approve revision 31's plan. Build part A now, and batch part B with
> the V2 premise fixes"). Part A built 2026-10-16; part B built 2026-10-16 with the owner's V2 premise decisions, and every record regenerated
once (R31.B6: about 3.5 h of runs on 10 cores, the part II record 1 h 41 min of it; the first attempt was stopped after the reviews found two code
defects). R31.B7 was added
> 2026-10-16 after approval, from the owner's decision on the band; its form approved the same day (2026-10-16: "Approve R31.B7's form as written (shares of the parents' acts to the child, acts open room but don't push)").** The review is
> `docs/REVIEW_spec_revisions_24-30.md` (all 85 entries decided 2026-10-16: 82 approved as built, 3 changed; then
> `M7.E.1f`'s band, changed by the owner the same day, R31.B7). This plan applies the changes, the text fixes recorded
> on approved entries, the seven inconsistencies the form lists, and the tags. Apart from R31.B7 it adds no mechanism
> and changes no constant.

**Why two parts.** The records' hash (`tools/ensemble_record.py::code_hash`) covers every file under `src/bowen/` and
`config/bowen/` except `constants_changes.md`. A one-line docstring fix there makes every ensemble, mutation and
diagnostic record stale. So the work splits:

- **Part A, documents only.** The spec, the explainer, `constants_changes.md`, this plan and the version 1
  definition. No record moves, no run. Can be built and committed at once after approval.
- **Part B, code and config.** Moves the hash, so every record is regenerated once. **Recommended: batch part B with
  the version 1 premise fixes** (`M11.C.21`, `M11.C.23`, `M11.C.40`, V2 of `docs/version_1_definition.md`), so the
  records are regenerated once for both (owner rule, 2026-10-09: batch before any slow rerun).

**The owner's decisions applied (2026-10-16, quoted from the review form).** Revision 24: "as recommended". Revision
25: "as recommended, with B for M11.C.14". Revisions 26 and 27: "as recommended, keeping the active-symptom reading
for M11.C.17". Revision 28: "continue to finish the review process", read as "as recommended" with option A for the
band (the form records the reading). Revisions 29 and 30: "as recommended".

#### Part A — documents (no record moves)

| ID | Change | Where | Verified by |
|---|---|---|---|
| R31.A1 | `channel_focus`'s register row says it is written from the moves delivered (engine-side attention), not "from belief" (inconsistency 1) | spec `M14.A`, line 1630 | `tests/test_review_rev31.py::test_r31a1_channel_focus_row_not_from_belief` |
| R31.A2 | `tolerance`'s register row says Family, one declared value per member (inconsistency 2) | spec `M14.A`, line 1679 | `test_r31a2_tolerance_row_per_member` |
| R31.A3 | `M11.C.6` limit (a) restated (option A): keep "**MUST NOT** complete in one or two generations"; drop "**MUST** be achievable within roughly ten". `M7.E.1f`'s "open conflict" sentence records the owner's decision of 2026-10-16, made after the review on the corrected account (the band also ruled out `FE07.7`'s 5–10-point step and `FE08.1`'s quantum jump): the band opens with the parents' acts toward the child, ±2% with none, to +5% with supportive acts, to −10% with projection (built as R31.B7). The explainer marks the traversal span (`KS11.1`, `FE08.2`) `[X]` contested and states what the new band allows | spec `M11.C.6`, `M7.E.1f`; `model_explainer.md` | `test_r31a3_c6_limit_a_restated`, `test_r31a3_explainer_marks_traversal_contested` |
| R31.A4 | `M11.C.14` restated (option B): under high marital distance, **no technique lowers the child's symptom load against a no-technique arm**; the difference between techniques is reported, not tested. `M11.4`'s exceptions row for `M11.C.14` changes from a two-sided null to a one-sided bound. Its mutation clause: the marital-distance gate removed **MUST** turn at least one technique variant red | spec `M11.C.14` row, `M11.4` table; explainer line 1134 | `test_r31a4_c14_row_restated` (text); the criterion itself in part B |
| R31.A5 | `M4.D.6f`'s gap sentence updated: since `M1.B.3e`, a chosen cutoff toward a person felt as dangerous is principled in proportion, so the gap is narrower than stated | spec `M4.D.6f` | `test_r31a5_m4d6f_gap_mentions_felt_danger` |
| R31.A6 | `M7.E.1c` records the owner's decision (plan §4.8): the other four situations are not applied at `t0` (inconsistency 4) | spec `M7.E.1c` | `test_r31a6_m7e1c_not_open` |
| R31.A7 | `M11.C.10`'s row: the superseded revision 11 and revision 29 mutation clauses struck | spec `M11.C.10` | `test_r31a7_c10_superseded_clauses_struck` |
| R31.A8 | `M11.C.2`'s "are to be added" removed; the settings are in `criteria.md` (inconsistency 5) | spec `M11.C.2` | `test_r31a8_c2_settings_not_to_be_added` |
| R31.A9 | Every revision 24–30 tag reading "pending owner review" becomes "approved by the owner 2026-10-16", or for the three changed entries "changed by the owner 2026-10-16, revision 31"; the revision 28 tags with no source (inconsistency 7) get the same tag as the rest of revision 28. Counted 2026-10-16: 64 in the spec, 7 in the explainer, 10 in `constants_changes.md` | spec, explainer, `constants_changes.md` | `test_r31a9_no_pending_review_tags_for_rev24_30` |
| R31.A10 | `M7.C.1g`'s "The owner has not stated a launch age" becomes the owner's approval of 20 (2026-10-16) | spec `M7.C.1g` | `test_r31a10_launch_age_approved` |
| R31.A11 | This plan: §4.9 item 9 (`focus_steer_gain`) closed at 1; step 10's "For the owner" on `M11.C.17`'s gate closed (active symptom kept) | `implementation_plan_phase_d.md` | read in review |
| R31.A12 | Status lines: the spec's frontmatter `version` becomes revision 31 with part A built (it still reads revision 29, stale since revision 30 was built: found 2026-10-16). Every run log's header reads it (`M16.A.1`), and two tests pin the string (`tests/bowen/test_log.py::test_m16a1_header_hashes_the_resolved_config`, `tests/bowen/test_render.py::test_m16c5_trace_carries_header_and_framing`), so they change with it. The form and the version 1 definition (V4) updated | spec frontmatter, the two tests, form, `version_1_definition.md` | `test_r31a12_spec_status_names_revision_31` and the two tests above |

Inconsistency 3 (`TRIANGLE`'s channel) needs no spec change: the owner approved the config as built; the B8 design
draft above stays as the draft it was.

#### Part B — code and config (the hash moves; one regeneration)

| ID | Change | Where | Verified by |
|---|---|---|---|
| R31.B1 | `symptoms.py`'s stale docstring ("a belief, and is B8's") says `channel_focus` is written from the acts delivered | `src/bowen/engine/symptoms.py:26` | `test_r31b1_symptoms_docstring_not_belief` |
| R31.B2 | The criteria settings carry a grade, `[I]` (`M11.C.2` ages, `renew_every`, `seeds`, the `M11.C.47` fixture values) (inconsistency 6) | `config/bowen/criteria.md` | `test_r31b2_criteria_settings_graded` |
| R31.B3 | "pending owner review" in the headers of `attribution.md` and `criteria.md`, and in the record renderer's reworded line (`tools/phase_d_record.py`, `REWORDED`, `REWORDED_2`), updated | config, tool | `test_r31a9_no_pending_review_tags_for_rev24_30` extended to these files |
| R31.B4 | `M11.C.14` as restated: two variants, `M11.C.14[firmness]` and `M11.C.14[talking it out]`. Each baseline is the same fixture with no scripted technique (the parents under their own policy); each treatment adds the technique. Readout `child_symptom_load`, one-sided: the variant passes if the interval's lower end lies above −(the equivalence bound). The former firmness-against-talking comparison is kept as a report-only row. The runner gains a one-sided bound mode for a `Readout` | `src/bowen/ensemble/criteria_d.py`, `src/bowen/ensemble/runner.py` | `tests/bowen/test_phase_d_criteria.py::test_m11c14_no_technique_lowers_load_under_marital_distance` (gate); adversarial unit test `tests/bowen/test_ensemble.py::test_m11c14_one_sided_bound_fails_when_interval_reaches_below` (an interval whose mean is inside but whose lower end crosses the bound must fail) |
| R31.B5 | `M11.C.14`'s mutation: the gate removed must turn at least one variant red. Added to `docs/phase_d_mutation_record.md` | `tools/phase_d_mutation_record.py` | the record's `test_m111d_*` tests |
| R31.B7 | **`M7.E.1f`'s band opens with the parents' acts** (the owner, 2026-10-16: "No attention +-2%"; "supportive, responsible behaviour attention - up to 5% growth"; "projection up to 10% negative"). *Form, Claude's recommendation, approved (the owner, 2026-10-16: "Approve R31.B7's form as written (shares of the parents' acts to the child, acts open room but don't push)"):* from 5 to 20, count the acts either parent sends the child: projection-type (`OVERFUNCTION`, `PURSUE`, `TRIANGLE`, the `projection` set of `config/bowen/pattern_acts.md`) and supportive (the self-directed channel's moves addressed to the child: `STAY-IN-CONTACT`, `I-POSITION`, `DETRIANGLE`, `PREVENT_ALIGNMENT`, `REDUCE_CUTOFF`, `PROVOKE`, from `config/bowen/event_kinds.md`). With `p` and `s` the shares of all parental acts to the child so far that were projection and supportive (0 when there were none), the yearly clip becomes `a × (1 − 0.02 − 0.08 p)` to `a × (1 + 0.02 + 0.03 s)`. The development rule itself (`development_rate × d`) is unchanged: the acts open room, they do not push. Two new constants, the owner's values, `[I]`: `development_band_projection` 0.10 and `development_band_support` 0.05; `development_band` 0.02 stays as the no-attention band. **Known weakness of the share form:** a child who receives few acts, all projection, gets the full −10% room; an alternative weights the shares by the child's share of the parents' acts to all their children. Changes every run in which a child develops (part II: `M11.C.6`, `M11.C.2[twelve-person family]`, `M11.C.10`) | `src/bowen/engine/life_course.py` (development), `config/bowen/constants.md`, `constants_changes.md` | `tests/bowen/test_life_course.py::test_m7e1f_no_parental_acts_keeps_two_percent_band`, `::test_m7e1f_projection_opens_room_to_fall_to_ten_percent`, `::test_m7e1f_support_opens_room_to_rise_to_five_percent`; adversarial: `::test_m7e1f_acts_open_room_but_do_not_push` (a child who receives only projection acts, while the parents function above basic, does not fall) |
| R31.B6 | Every record regenerated once, after part B and the batched premise fixes | `tools/phase_d_record.py --part 1/2`, `tools/ensemble_record.py`, the mutation records | the record-current tests |

**Risk stated before the build.** If the gate-removed mutant does not turn either `M11.C.14` variant red (that is,
no technique lowers the child's load even without the gate), the restated criterion would pass for a reason other
than the gate. The build stops there and reports to the owner. It does not change the criterion to make it bite.

**Compute.** Part A: none. Part B: one regeneration of every record. Revision 30's plan estimated about 2 hours for
the ensemble records plus about 1 hour per `M11.C.10` variant at 200 seeds; the mutation records rerun in full
because they key on the same hash. Not measured as one figure; it will be timed and reported. `M11.C.14`'s two
variants add two short runs (52 weeks, the nuclear four).

#### Part B built, and the batched premises diagnosed — for the owner, 2026-10-16

**Built (not committed until the records regenerate):** R31.B1, B2, B3, B4, B5 and B7, each with the tests the table
names. The fast suite passes except the record-current tests, which go red because the hash moved. Two changes the table
did not list: (1) "pending owner review" also appeared in eleven code comments and docstrings for revisions 24–30
(`criteria_d.py`, `criteria_d2.py`, `symptoms.py`, `attribution.py`, `objects.py`, `learner.py`, `appraise.py`,
`moves.py`, `life_course.py`, the two readouts). They now read "approved by the owner 2026-10-16", since the hash moves
anyway. (2) `criteria.md`'s loader now refuses a row whose grade is not `[I]`. One adjacent item, not changed:
`config/bowen/family_phase_d.md` line 19 still reads "awaiting owner review" for revision 13. Every number below is ad
hoc: run on this working tree, in-process or in a copy, never cached and never a verdict.

**`M11.C.14`: the stated risk was reached in a different form. The build stops here for the owner.** As built, both
variants **fail**: firmness lowers the child's load by 6.4 ± 3.6 against no technique, talking it out by 17.3 ± 2.9
(100 seeds). With the gate removed: −5.6 ± 3.8 and −16.3 ± 3.1. The gate is not what decides the result. What the
diagnosis found (40 seeds, the child's acute anxiety by mechanism):
- *Displacement.* A scripted act is the parent's whole act for that week. The no-technique baseline is the parents under
  their own policy, and they send the child about 30 acts a year (`CONFLICT` 3.3, `CUTOFF` 3.6, `DISTANCE` 4.0,
  `TRIANGLE` 3.0, …). Scripting the technique every second week removes about half of those.
- With the parents unable to address the child under their own policy in **both** arms (150 seeds): firmness −1.3 ± 1.5
  (passes the one-sided bound, 3.8); talking it out −7.1 ± 1.3 (fails). The gate changes neither (−1.2, −7.0). Talking
  it out's remaining effect is still displacement, through what the child **witnesses**. On scripted weeks the parents'
  own acts toward each other are replaced, so the child receives 66 ± 8 less appraisal load as a witness. A further
  7.4 ± 1.2 of standing load goes, as contact moves the parent–child tie toward its optimum. Neither is an improvement
  addressed to the child, which is all `M5.C.1`'s gate blocks.
- Options: **(a)** the baseline scripts the parents **holding back** (`WITHHOLD`) on the same weeks, so both arms lose
  the same acts of their own (`Forced` must accept a selection with no target; not measured). **(b)** Read the
  displacement as part of the model: a parent who spends the week talking with the child is not fighting with the
  spouse. Then talking it out does lower the load, and `M11.C.14` is a finding, not a premise (reclassified by the
  owner). **(c)** Widen `M5.C.1`'s gate to the tie's contact and standing load. This changes a rule to make a criterion
  pass, and gating contact was tried in 2026-10 without bringing the result nearer the bound (`TODO.md`).
  **Recommended: (a)**, then run once and report. If talking it out still lowers the load under (a), the remaining path
  is contact, and (b) is the honest reading.

**`M11.C.21` (premise, fails −1.79 ± 1.65 in the record).** Diagnosis (100 seeds, after the drop):
- *The precondition is weak.* At the drop the child has lent on average only 1.1 points of functioning (pseudo-self
  −1.08 ± 0.87), and 43% of seeds have lent none. At most about a point can come back.
- *The collapse does not reach the exchange.* The drop is written to the parents' stress ledger, as `M11.C.21`'s arm
  states it. Nothing in `functioning_shift` reads the over-functioner's capacity. A parent 10 points down takes just as
  much when they over-function (−3.6 ± 0.8 from the child, against −1.4 ± 0.5 without the drop, as they send it more
  often). The giver's floor (`stress_floor_below_basic`) means the collapsed parents give less when the child takes over:
  4.1 against 4.7, and the child's `OVERFUNCTION` moves nothing 0.16 times per seed against 0.02. All the exchange acts
  rise after the drop, in both directions. Net, the child gives more (−2.1 ± 1.6 through the exchange, against −0.1 ±
  1.5). The direction runs against the criterion, and it follows from two rules, not from noise.
- Options: **(a)** a new rule: what an over-functioner can take is limited by their own capacity, the mirror of the
  giver's floor (`M6.I.4`; the owner's form; then rerun). This is the reading of the first instance: "a schizophrenic
  brother becoming the family's functional member when both parents collapsed". **(b)** Restate the arm on the second
  instance, "a projection target's symptoms fading after both parents died". First build a lent balance (the parents
  over-function toward the target for a declared span, as `M11.C.24` does), then end the parents' over-functioning
  (death, or their exchange acts withheld). Revision 27's settling-back returns what was lent. This largely repeats
  `M11.C.24`. **(c)** Withdraw `M11.C.21` as unproducible by the current rules. **Recommended: (a)**. It is the only
  option that tests reciprocity rather than settling-back, but it adds a mechanism, so the owner sets its form.

**`M11.C.23` (premise, fails reversed −0.67 ± 0.09).** Its direction is **not** set by the suppression size. A grid over
suppression 1.0 to 0.5 (150 seeds) shows high-EE with suppression against low-EE without: +0.01 ± 0.11 at 1.0 (no
medication), −0.09 at 0.9, −0.28 at 0.8, −0.42 at 0.7, −0.54 at 0.6, −0.67 at 0.5. No suppression below 1 passes. What
decides it: in the model, hostility barely raises relapse at all. The EE effect at equal suppression ranges from +0.007
± 0.109 to +0.107 ± 0.098, while medication's effect is +0.68 ± 0.09. One explanation was tested and ruled out: the
child's usual level adapting to constant hostility. Holding it fixed changed nothing (identical counts). Why
`CONFLICT` and `STAY-IN-CONTACT` produce nearly the same relapse count is not traced.
- Options: **(a)** trace why the EE term does not reach relapse before deciding (one diagnosis run, like those above).
  **(b)** Strengthen the EE arm (more frequent or more intense hostility, both `[I]`). This would be tuning a size until
  the criterion passes, and a pass would then confirm the chosen size, not the model. **(c)** Reclassify as a composite
  (it compares the magnitudes of two mechanisms, which `M11.5` treats as a finding) and accept the failure as a finding
  under V1. **(d)** Restate as the 2×2's within-row directions (EE raises relapse at each suppression level). Measured,
  these are weak and mostly within noise. **Recommended: (a)**, then choose between (c) and a rule change.

**`M11.C.40` (premise, clause 2 fails: +0.042 ± 0.044, p 0.05).** Clause 2's intent holds in the mean. Learning does not
move the self-directed channel's entropy (between-arm difference −0.005 ± 0.054, 100 seeds), against +0.046 ± 0.008 for
the automatic channel. The failure is noise: the self-directed channel has about 13 acts per 26-week window over six
kinds, so its per-seed difference averages 0.22 in size, four times the automatic signal. **More important, the readout
cannot detect `M4.D.6d`'s removal here.** A joint mutant that lets reinforcement reach the self-directed channel
(`policy.py`, three literal edits; `I-POSITION` left out because a sequence start emits no event) moves that channel's
difference only to +0.015 ± 0.049. A restated null would pass under the mutant, and so would the clause as written
(FAIL, p 0.09 for clause 2).
- Options: **(a)** keep clause 1 as the criterion. Prove clause 2 by the existing unit test
  (`tests/bowen/test_learner.py::test_m4d6d_self_directed_channel_is_never_reinforced`) and report its ensemble reading
  only. **(b)** Find a fixture with enough self-directed acts (a coached or higher-level family) and longer windows,
  then check the mutant bites before restating. **(c)** Raise the seed count. This does not help: the mutant shows no
  bite to detect. **Recommended: (a)**. If the owner wants the ensemble to carry clause 2, then (b), which is not yet
  measured.

**The owner's decisions, 2026-10-16** (answers to the four questions above, each the recommended option):
`M11.C.14` (a), "Baseline holds back": the baseline scripts the parents holding back on the same weeks; run once and
report. `M11.C.21` (a), "Taking capped by capacity": a new `M6.I.4` rule whose form the owner sets; then one rerun.
`M11.C.23` (a), "Trace it first": one diagnosis run, then the owner chooses between reclassifying and a rule change.
`M11.C.40` (a), "Unit test proves clause 2": clause 1 stays the ensemble criterion; clause 2 rests on
`tests/bowen/test_learner.py::test_m4d6d_self_directed_channel_is_never_reinforced`, and its ensemble reading is reported.

**Built after the decisions, and what each showed (ad hoc, 100–150 seeds, 2026-10-16):**
- `M11.C.14` (a): the baseline scripts a targetless `WITHHOLD` on the technique's weeks (`Forced` now accepts it and
  books it as made when the actor emits no move). Both variants **pass**: firmness +8.2 ± 3.0, talking it out −4.5 ± 2.2
  (one-sided bound about 7). **The gate-removed mutant survives both** (+9.5 ± 3.0, −3.6 ± 2.3): the plan's stated stop
  condition. Cause: in this fixture the gate has almost nothing to block. Over a run it zeroes 0.8–1.4 points of relief
  to the child (one or two deliveries), against 28–99 points of load delivered while it is on, and no calming transfer.
  The pass shows that no technique lowers the load here; it does not show that the gate is the reason. The hold is
  coping in the model, as talking it out is; firmness is not.
- `M11.C.21` (a): `functioning_shift` scales what the over-functioner takes by
  `clip((f − (b − stress_floor_below_basic)) / stress_floor_below_basic, 0, 1)`
  (`tests/bowen/test_moves.py::test_m6i4_collapsed_taker_takes_in_proportion_to_room`, and the adversarial
  `::test_m6i4_capacity_does_not_limit_a_taker_above_basic`). The rerun: +0.26 ± 1.35 (was −1.79 ± 1.65). **Still
  fails**, now in the stated direction. The precondition is unchanged: the child has lent about a point at the drop.
- `M11.C.23` (a), traced (60 seeds, the child's acute anxiety by mechanism after the remission): hostility delivers
  41 ± 10 more appraisal load than talking it out. Two things cancel it: higher anxiety decays faster (−33 ± 10), and
  the child gets relief by cutting off (−7.5 ± 1.1) and distancing. Mean anxiety ends 2.8 points higher, and symptom
  load is no higher (0.74 against 0.76 of threshold). Talking it out every second week loads the child with 83% of
  hostility's appraisal load. Even against a hold-back arm, hostility does not raise relapse (+0.05 ± 0.11 without
  suppression). Without suppression relapse is about 0.8 per seed in 48 weeks in every arm, close to the most a
  52-week run allows.
- `M11.C.40` (a): `automatic_less_self` is report-only; clause 1 gates
  (`tests/bowen/test_phase_d_criteria.py::test_m11c40_clause_2_reported_and_proved_by_the_unit_test`).

**The owner's second decisions, 2026-10-16** (each the recommended option, then "proceed with your recommendations"):
`M11.C.14`, "Fixture the gate acts in"; `M11.C.21`, "Build the lent balance first"; `M11.C.23`, "Reclassify, accept
finding". The capacity rule's form: "Proportional to room".
- `M11.C.14`: a fixture with calmer parents was tried, lowering the parents' chronic and acute anxiety by 10, 20 and 30
  (100 seeds each, a fresh process per setting). Removing the gate never turned a variant red; it made talking it out
  slightly *worse* for the child (−3.0 → +0.4 at 30). No further fixture was searched, since that would be fishing for
  a mutant that bites. The hold-back restatement is kept and passes. The mutation record shows the gate-removed mutant
  as **survived**, and the spec marks the mutation clause **not met**, for the owner.
- `M11.C.21`: the lent balance was built in two forms, both reported.
  - First form: the parents over-function toward the child every second week until the drop, in both arms. The child
    has lent 14.5 ± 0.5 at the drop; the result is −0.88 ± 1.05. The scripted over-functioning ends at the drop in both
    arms, so lent functioning settles back in both (+6.6, +5.3). It settles back less in the collapse arm because more
    exchange acts pause the settling.
  - Second form, kept: the row says "both projecting parents", so they over-function throughout, in both arms, and the
    arms differ only by the collapse. Result: +0.15 ± 0.19 (p 0.09, 50 seeds): **fails**. The child has lent down to its
    own floor in both arms, so the capacity rule has nothing to change. The correctness review notes that with taking
    in both arms the difference measures the capacity rule; that is the mechanism the row names (`M6.I.4`). It also
    notes that capacity reads functional level, which already includes what the taker has lent, so a person who has
    lent takes back more slowly (for the owner). **Open for the owner: accept as a finding, or withdraw.** Neither needs
    a rerun.
- `M11.C.23`: reclassified as a composite; the failure is accepted as a finding, with the trace above as the reason.

**Reviews of the code diff, 2026-10-16** (one `learning-qa` pass, one correctness pass, before the regeneration that
counts):
- Fixed:
  - The band clipped the level itself, so a share that fell over childhood moved the level with no development push
    (both reviews; reproduced: +1.58 points while the rule pulled down). The band now only limits the step
    (`tests/bowen/test_life_course.py::test_m7e1f_a_narrowing_band_does_not_move_the_level`).
  - The taker's floor lacked the `max(0, …)` clamp the giver's and stress's have
    (`tests/bowen/test_moves.py::test_m6i4_a_low_basic_taker_at_zero_takes_nothing`).
  - The record rendered a one-sided bound as if two-sided, and the mutation child dropped the bound. Both now carry it.
  - `passing_part_1` read an absent id as failing; it now refuses a stale part I record.
  - `Readout` refuses `bound_side` on a directional readout (`test_m11c14_bound_side_only_on_a_null`).
- The first regeneration was stopped after the Phase C ensemble record, and everything reruns once.
- Minor, recorded in `TODO.md`:
  - a scripted hold is booked "made" when its actor emitted no move, inferred rather than read from its selection
    record;
  - `bound_side = −1` has no caller and no test.

**The owner's decision on the capacity rule, 2026-10-17** (the owner, 2026-10-17: "Keep the capacity rule. Accept M11.C.22 failing for now and apply M11.C.21's reclassification at the next code change"). The rule stays; `M11.C.22`'s failure is accepted for now; `M11.C.21` becomes a composite with its failure accepted as a finding, applied in the code with the next code change.

**Nothing has been changed to make any criterion pass.** Once the owner has decided, the decisions are built and the
records regenerate once (R31.B6; estimate in the status line of this section when started).

**Documents updated by the build.** Part A: the spec, `model_explainer.md`, `constants_changes.md`, this plan, the
review form, `version_1_definition.md`, `TODO.md`. Part B: `config/bowen/criteria.md`, `config/bowen/attribution.md`,
and the regenerated records (`docs/readouts_phase_c.md` does not mention `M11.C.14`, checked 2026-10-16).

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
    *2026-10-13:* the ensemble record is built (`tools/phase_d_record.py`, `docs/phase_d_ensemble_record.md`); the
    mutation and sweep records for the D criteria are not, and their mutations are reported ad hoc in steps 6–10.
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
