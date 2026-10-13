# Implementation plan — Phase D

> **Status: approved by the owner, 2026-10-10 ("Proceed"), with the decisions in §1a.** P9 decided 2026-10-11; P13–P16 wait for D-II. **Built:** B1 and B2 (2026-10-10/11); B3 and B4 (2026-10-11, spec revision 16); B5 and B6 (2026-10-11); B7, B8 and B9 and part of step 10 (2026-10-12, spec revisions 24–27, Claude's recommendation pending owner review: the owner asked for the work to proceed overnight without waiting for input).
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
