# The 2026-11-23 digest: candidate review, and the items it raises

Review of the twenty-five candidates in the six papers read in full from the 2026-11-23 method-literature digest
(`papers/SWEEP_READING_REPORTS_2026-11-23.md`; plain-language version in
`papers/SPEC_CANDIDATES_plain_language_2026-11-23.md`). The batch's holding item was T-270 in
`2026-09-27_paper_material.md`; this record closes it and lists what it decomposes into.

**What this review is.** A read of each candidate against the current spec (revision 67, `docs/bowen_agent_model_spec_v2.md`)
asking four questions, in order: (1) is the gap real — does the spec already carry it? (2) is it in scope for this
project — a rule about mechanism, inference validity or reporting, and not an imported scoring device? (3) is it blocked —
by a theory decision only the owner can take, by another candidate, or by an unbuilt module? (4) is it worth the running
cost? Every candidate here had already passed its reader's overlap check (each report's "already covered, not proposed"
block names the near-neighbour requirement IDs), so nothing below is a candidate the reader flagged as a duplicate. The
verification done for this review was: the proposed requirement IDs do not already exist (`M17.A.5`, `M16.A.3d`,
`M16.A.3e`, `M16.A.11`, `M17.G.4`, `M11.1h`, `M11.2a` — all zero hits at revision 67), and the wording each candidate
rests on is absent (`guardrail`, `not_elicited`, `resampling unit`, `exposure`, `eligible denominator` — zero hits each;
`UNDETERMINED` exists as a verdict, four hits).

**Outcome.** Nothing is built, and nothing enters the spec from this review. Twenty-five candidates become twenty-three
items: two pairs are the same rule written twice and are merged, no candidate is rejected on merit, and three carry an
owner decision ahead of them (the failable-repair theory question behind AL1–AL3, MG3's replication cost, MG2's
vocabulary). Twelve are cheap (test, reporting rule or record change); two are real engine work (AL1, AP1); the rest sit
between. All are held for a later revision on the same terms as the earlier batches: nothing here is a requirement yet.

| Candidate | Item | Verdict |
|---|---|---|
| AL1 staged, failable repair | T-271 | include — the batch's one structural gap; owner theory decision first |
| AL2 repair aimed at the strained attribute | T-272 | include — depends on AL1 |
| AL3 graded taboo register, bypassed vs addressed | T-273 | include — the same theory decision; rewrites `M1.B.10a` |
| AL4 successes as a rate over opportunities | T-274 | include |
| AL5 name the resampling unit | T-275 | include, **merged with PW4** |
| US1 fault injection over declared inputs | T-276 | include |
| US2 control retention, no composite scores | T-277 | include |
| US3 per-seed attribution of the acting member | T-278 | include |
| US4 exposure-adjusted readouts | T-279 | include, **merged with PW3** |
| US5 plausibility never enters a verdict | T-280 | include — zero implementation |
| AP1 per-tie cue spread | T-281 | include — engine work; declare its joint effect with `M4.C.16` |
| AP2 bounded change amplification | T-282 | include — `[I]` throughout, invented premise; after AP1 |
| AP3 three rungs for an adaptive mechanism | T-283 | include |
| AP4 read-and-updated quantities read from last tick | T-284 | include |
| AP5 counterpart and joint readouts, channel path | T-285 | include |
| PS1 how many persons carry a difference | T-286 | include |
| PS2 act selected against act delivered | T-287 | include — cues-on runs only |
| MG1 decision census, one class per person-tick | T-288 | include |
| MG2 record the pre-mask first choice | T-289 | include — owner decision on the readout's vocabulary |
| MG3 confirmatory replication on a fresh block | T-290 | include — owner decision on cost |
| MG4 append-only records, no bare superseded verdict | T-291 | include — cheapest protective rule |
| PW1 composition dimension never "unaudited" | T-292 | include |
| PW2 one guardrail readout per criterion | T-293 | include |
| PW3 reach vs per-event effect | T-279 | merged into US4's item — one denominator rule |
| PW4 the family is the resampling unit | T-275 | merged into AL5's item — one resampling rule |

Three conventions hold across all twenty-three items. They are `⟨later⟩` and enter no revision until the owner assigns
them. None creates a constant that could be fitted: the two that add one (AP2's gain, US2's retention share) declare it in
`M10` before a run, as the spec already requires of `[I]` values. And none asserts on prose: every item is to be checked
by a structural test over records the engine already writes, or by a comparison in a report.

---

- [ ] T-271 ⟨later⟩ **Failable staged repair, not a scripted reconciliation.** AL1: a repair of a damaged tie must run as
  an ordered sequence (attend to the shift → say what the damage was → the damaging party owns its part → re-agree how to
  go on), where each stage advances only on content the other member produced in that stage's own terms, assent alone
  does not advance a stage, and a stage that receives no such content inside its window closes unresolved with the tie
  keeping the strain it held. Lands in `M5` as a state machine beside `M5.D`'s `I-POSITION`, with the precedent `M5.D.9`
  already sets for a sequence that takes priority over `M4.D.1`'s weekly selection; the stage gate reads delivered events,
  so it touches `M4.C.1` and `M1.F.1a`. **This is the batch's one structural gap:** `RECONCILIATION` (`M4.A.2`) is a
  scripted event that empties the tie's stock in one firing (`M1.B.4`, `M1.D.2a`–`D.2c`), so repair in EPModel cannot fail
  while `M5.D.2a` requires the mover's own sequence to be able to — and "smoothed over" and "worked through" are the same
  event. Evidence: `papers/SWEEP_READING_REPORTS_2026-11-23.md` Part 1, AL1 (paper §3.4, Tables 23–24, App. E Fig. 11,
  Algorithm 2; 55 of 57 major ruptures resolve through the ladder, Table 7). **Owner decision first:** whether a repair
  attempt the theory allows to fail, and the pseudo-resolution/working-through distinction behind it, is a claim the
  owner wants the model to make. Cost: moderate to high — a per-tie repair record, a stage cursor, windows in fast ticks,
  a priority rule; the windows and stage count are `[I]`.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-272 ⟨later⟩ **A repair aimed at the wrong attribute does not repair.** AL2: a repair move must be aimed at the
  attribute of the tie that carries the strain — `felt_contact`, `functioning_balance` or `areas_of_joint_activity`
  (`M1.B.9`) — and a mis-aimed repair must not reduce the strain and should be appraised by its receiver as a further
  negative event on the tie. Lands in `M4.C.1`'s appraisal against the strained attribute and in `M1` as a field naming
  which attribute is strained. Cheap once T-271 exists: one tie field and an appraisal branch, no constants beyond the
  mis-aim penalty (`[I]`). Sharpens `M1.B.9`/`M1.B.5`, which carry the three attributes but only one repair event.
  Evidence: Part 1, AL2 (Table 23 caption, Table 25, Fig. 10's `re_rupture`). **Depends on T-271.**
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-273 ⟨later⟩ **A graded taboo register, with bypassed and addressed as different states.** AL3: `taboo_set`
  (`M1.B.10`) must become a per-subject grade rather than set membership, and a strain on a subject must be recorded as
  either **bypassed** — the triggering move dropped or redone more gently, lowering that subject's sensitivity one level
  only and never ending it — or **addressed** — the specific concern named and the feared response disconfirmed, closing
  that subject permanently for that tie, so the same move no longer registers. Lands in `M1.B.10`, written by `M4.C.1`'s
  appraisal and, once it exists, by T-271's final stage. Corrects a coarseness: `M1.B.10a` returns a taboo by purposeful
  mention under self-control — one path, no grade, no closure — and `M1.D.2c`'s repair by thirds is a quantity on the
  tie, not a state of the subject. Every `M11.C` readout that counts taboo entries must read the grade instead of
  membership, and `M1.B.10a` must be rewritten as the addressed mode rather than kept beside it or the two will
  contradict. Evidence: Part 1, AL3 (paper §3.2, Table 11, Algorithm 1 lines 25–26). Carries the same owner theory
  decision as T-271.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-274 ⟨later⟩ **Every success is a rate over the chances there were.** AL4: a readout that counts an outcome a
  situation may or may not elicit — a repair after a rupture, a position held under pressure, contact after a cut-off —
  must report per arm how many such occasions arose, must report a run where none arose as `not_elicited` rather than as a
  zero, and must count an occasion that arose and drew nothing as a failure rather than pooling it. Where an arm changes
  how many attempts are made, each revised attempt must be scored against its original objective so added attempts cannot
  add successes. Lands in `M11.G` and `M17.B`; the opportunity count is an observer over `M16`'s event stream, so `M16.B`
  holds. Closes a hole: `M11.C.48` compares the run standing on a tie at its first `CUTOFF` and has no defined behaviour
  where no tie is cut off; `M17.A.4`'s margin and `M17.C.1`'s variance components do not reach it. Evidence: Part 1, AL4
  (paper Fig. 16, App. E's objective-linkage rule, Table 6's
  124/228 against a baseline's 18). Decides one thing: `not_elicited` does not count toward a criterion, and the count is
  printed beside the rate.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-275 ⟨later⟩ **Name what a confidence interval is resampled over; the family is the unit.** AL5 merged with PW4:
  every Phase E interval must name its resampling unit; where the design is nested (seeds inside an `M15` family draw,
  seeds inside a `D0` cell) the interval must be computed on the outer unit with the naive per-seed interval printed
  beside it, and a contrast whose verdict crosses the declared margin between the two units is reported `UNDETERMINED`,
  whichever unit favours it. From the other side: any readout computed over more than one person or tie must declare that
  clustering, an `n` for such a readout must count families and not persons or person-weeks, and a test that treats
  persons or weeks within one family as independent must not be used. Lands in `M17.A` (a field in `M17.C.1`) and a new
  `M17.G.4`, binding `M11.4`'s within-run comparisons too. The defect is live: `M17.C.1` draws families as distributions
  and reports four variance components, but nothing says which unit an interval is computed over; `M17.A.1`'s rule about
  a verdict moving between two *engines* catches the source's case only because both p values happen to land in its 0.01–0.2
  band — a wrong unit gives a wrong interval at p = 0.001 as well. Twelve persons × 2,080 ticks is one family, and a
  per-person standard error would report precision the model does not have. Evidence: Parts 1 and 6, AL5 (App. I:
  +0.022 reads p = 0.068 or p = 0.035 by unit, both printed) and PW4 (App. B.7's 10,000 whole-team resamples; App. C's
  statement that generic categorical tests require verification for repeated-person designs). Cost: analysis-layer only,
  plus carrying the nesting from `M15`'s import into the runner.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-276 ⟨later⟩ **Fault injection over an arm's declared inputs.** US1: the acceptance suite must include a
  fault-injection set that alters one arm's declared inputs while every mechanism runs unchanged — a scripted-act
  substitution for one named member, a belief initial-state substitution (`M9.9`), a hard-constraint removal, a run
  truncation, an external-agent magnitude substitution, and a transient-reaction substitution asking the estimator to
  absorb a one-off reaction into `basic_level` — and each `M11.C` criterion must declare before the run which faults it is
  expected to detect, with a coverage count printed beside its verdict on the rule "any readout, at any of `M17.B`'s
  levels, detects it once". Lands as a new `M11.1h` plus a harness section; imported-family faults ride `M15.B.1`'s
  ranges. Corrects an asymmetry: EPModel's mutation family is structural — delete (`M11.1`), invert (`M11.1d`), sever
  (`M11.1b`), re-encode (`M11.1c`), permute order (`M11.D.16`) — and none of them perturbs *what an arm was told to be*
  while the machinery runs, which is the defect class the source shows good-looking metrics miss. A fault is a model
  change under `M11.4f`, so it is admissible as a harness test only and must never be reported as a counterfactual
  result. Evidence: Part 2, US1 (paper §4.3, Table 5, §6.1.1's 6/10, 5/10, 6/10 against 10/10, §A.3.2). Cost: moderate —
  each fault needs a declared entry channel (`M17.D.3`).
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-277 ⟨later⟩ **Control retention at a fixed mutant survival, and no composite scores.** US2: for each `M11.C`
  criterion report, beside the mutant verdict, the fraction of unperturbed seeds still satisfying the criterion's margin
  at the threshold admitting 5% of fault-injected seeds; a criterion retaining less than a declared `[I]` share of its
  controls is reported **screening but not discriminating** and may not be cited as evidence for its mechanism; and no
  aggregate score across criteria or readout families is reported at any level. Lands as `M17.A.5` (this item takes the
  number; see T-290, which needs the next free one) with the reporting half in `M17.G` and the prohibition beside
  `M11.5`. This is the missing false-alarm side: `M11.1` proves a test can go red and `M11.4a`'s equivalence bound governs
  only nulls, so nothing today measures whether a criterion goes red for the right seeds. Evidence: Part 2, US2 (paper
  §4.4, §6.1.2 Tables 1–3 with persona retention 0.10/0.17 against 0.68/0.77, Table 15's composite AUROC 0.37/0.29/0.26
  below chance for approve-broken-work, §5.4). Cheap per criterion: the fault set is T-276's, and both sides are runs the
  suite already makes. One new `[I]` constant, declared in `M10`.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-278 ⟨later⟩ **Classify every seed by which member the mechanism acted on.** US3: a criterion that names a member
  must, per seed, classify the run as **named member** (the mechanism acted on the named member), **other member**, or
  **none**, by a declared log predicate over `M16` records rather than a mean over persons; it passes only when the
  named-member count exceeds the other-member count by the declared margin, and the other-member count is reported with
  the verdict. Lands in `M17.B.1`, whose seed classification is `same / converted / reversed` on the arm difference and
  never says which person moved. `M17.D.2(a)` demands the assertion be about the named member but checks it through the
  rival arm's shape, so a criterion can satisfy the wording while the mechanism lands elsewhere in 45% of seeds. Evidence:
  Part 2, US3 (paper §3.2 L3, §5.3, §6.1.1's 0.70 → 1.00 aggregation effect). Cost: one log predicate per criterion;
  `M11.G.5` already attributes moves per pair.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-279 ⟨later⟩ **Denominators: exposure, eligibility, and which of the three carries a difference.** US4 merged
  with PW3: any readout counting events, days in a state or a distribution over time is expressed per unit of exposure
  (person-weeks at risk, weeks in the state) with the denominator reported, and a run shortened by a death, a cutoff or a
  short horizon is not compared against one with more exposure without that adjustment; and every cumulative readout
  reports the eligible denominator (persons or weeks in which the mechanism could fire), the count converted and the
  per-eligible rate, stating which of the three carries the between-arm difference — a difference carried by the eligible
  denominator alone is not the mechanism acting more strongly. Land in `M17.B` and `M16.A` (exposure), extending
  `M11.1e` from asymmetric-constant mechanisms to every cumulative readout, with `M16.A.8`'s record already separating
  per-event magnitudes from counts. Both halves are live defects: over forty years with mortality, arms differ in
  person-weeks by construction, so a count per run rewards whichever arm lost members earlier; and `M11.C.10`, `M11.C.16`,
  `M11.C.48` and `M11.C.55` can each pass on reach alone. Evidence: Part 2, US4 (paper §6.1.2, Table 2, Table 10) and
  Part 6, PW3 (App. B.5's 97.3–100% against 30.9–75.2% eligible while agreeing conditionally at 92.1% vs 95.6%).
  Arithmetic on records already written; the one caution is that exposure adjustment can hide a mechanism whose effect is
  to end exposure, so the denominator is reported rather than silently divided.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-280 ⟨later⟩ **A plausibility judgement never enters a verdict.** US5: any check that a run, trace or narrator
  line "looks right" — by the owner, a reviewer or a language model — is recorded as a plausibility note, names its
  reader, and contributes to no `M11.C` verdict and no claim grade; where the spec already defers to human review (the
  curve shape of `M11.C.11`, the counterfeit threshold of `M5.F.2`) the review is recorded as choosing a declared `[I]`
  value, not as confirming a result. Lands in `M11.5` and `M11.F`'s framing block, binding `M16`'s renderer wherever a
  trace is shown. Zero implementation; the cost is political, because it forbids counting a comfort the project may want
  — reading the narrator line and saying the family behaves like a family. It does not forbid the reading. Evidence: Part
  2, US5 (paper App. D.2, Table 13: paired identification of a defective replay at 0.56 accuracy against a 0.50 floor).
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-281 ⟨later⟩ **A cue's scale is the tie's own, not the family default.** AP1: with the social cues on, each tie
  carries per partner and per cue a running spread of the cues the receiver perceives (an exponential variance over an
  `[I]` window), and the felt change a cue makes is expressed in units of that spread, so the same absolute departure
  costs more on a tie that never varies and less on one that always does; the spread is estimated from the perceived
  vector only, after fidelity, is floored at `M4.C.14`'s level-derived sensitivity, and its rate is slow enough that a
  monotone deterioration inside one window still registers. Lands in `M4.C` with the state on `Relationship` beside
  `M1.B.3e`, which has the same shape. This is the batch's one other piece of engine work and it fills a stated
  asymmetry: `M9.9` gives a per-tie cue *location* and `M4.C.14` a generic *slope*, so a habitually flat partner and a
  habitually volatile one are read with the same sensitivity today. Four risks to write into the requirement: it must read
  as a statistic of delivered events rather than a free per-person parameter (`M10.A.1`); reference drift, where a slow
  real deterioration is absorbed into the spread, needs the slow-rate clause and a test; reading the *sent* rather than the
  *perceived* signal would hand the receiver a channel to the sender's true cue and defeat `M4.B.2` and `M11.C.36`; and
  the joint effect with `M4.C.16`'s 12-week dispersion must be declared rather than left to the arithmetic, since dividing
  by the spread and multiplying by its dispersion push opposite ways. Evidence: Part 3, AP1 (paper §IV eq. (8), §II.C,
  §VI.A.1's 5.7× output range; strength ARGUED — no independent affect criterion exists in the source).
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-282 ⟨later⟩ **A change in a cue amplifies the deviation it makes, and the amplification is bounded.** AP2: with
  the cues on, the felt change a delivered cue makes is scaled by `1 + cue_change_gain × |P|`, where `P` is that cue's
  signed change on the tie since the previous delivery to the same receiver from the same sender, normalised as `M4.C`
  normalises intensity; the scaling applies to the existing two-sided deviation and adds no drive, and a tie held at a
  constant low value for a window accumulates nothing from it. Lands as an amendment to `M4.C.1e` with the previous
  delivery's value on `Relationship`. Closes a gap: `M4.C.1` is a level appraisal, so a tie sitting at a poor level and
  one that has just fallen to it are indistinguishable in the week of the fall; `M4.C.16` is the nearest rule and reads
  dispersion, not the first difference, and says nothing about direction. Evidence: Part 3, AP2 (paper §IV eqs. (6)–(7),
  §VII). Lowest confidence in the batch: it is `[I]` throughout, with no corpus source, so it must be presented as an
  invented premise the way `M4.C.11` is and not as a Bowen claim. **After T-281**, whose spread is a second intensifier
  on the same quantity — declare the joint effect and sweep the pair together (`M17.E.1`), and run the reference family
  for runaway conflict (a run-length check, not a new criterion).
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-283 ⟨later⟩ **An adaptive mechanism is tested in three rungs.** AP3: where a criterion or arm credits a
  mechanism that both calibrates and keeps adapting, Phase E runs three arms — off, calibrated through a declared window
  then frozen, adapting throughout — and reports the off→frozen and frozen→adapting differences separately; where the
  frozen arm carries the whole difference, continuing adaptation is reported **not discriminating** with the consequence
  `M17.D.2(c)` already names for a rival arm. Lands as a sixth named control arm in `M17.D`, the switch beside
  `M10.B.5`, the window declared in `M10.B` before the suite and swept under `M17.E.1`. Repairs a gap in the arm
  vocabulary: `M17.D.1`'s five control arms contain no "adaptive mechanism held at its calibrated value", and `M10.B.5`
  gives on/off only, so every EPModel result about the learner is all-or-nothing while `M4.D.6`, AP1's spread and
  `M9.9`'s smoothing are all two-stage. Evidence: Part 3, AP3 (paper §VIII.A's first limitation and §V.C's admission that
  its comparison evaluates training and continued adaptation together).
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-284 ⟨later⟩ **A quantity that is read and updated is read from the previous tick's end.** AP4: any quantity both
  read by appraisal and updated by it — `M9.9`'s believed cue vector, AP1's spread, `M4.C.16`'s variance, `M1.B.3e`'s
  felt-danger average — is read in tick *t* at its value as of the end of *t−1*, and is not updated by any delivery
  appraised in *t* until every delivery in that batch has been appraised, so two deliveries in one tick are read through
  the same value whatever order the batch is processed in. Lands in `M3` as a generalisation of `M1.F.8`, asserted under
  `M4.G.2`, with a permutation test (swap two deliveries' order within a tick; the tick's outcome must not change). Closes
  a silent ordering channel: `M1.F.8` covers the state a delivery is appraised against, not a quantity the appraisal
  reads and then writes, so an implementation can legitimately update `M9.9`'s smoothing after each delivery and make the
  second delivery's felt change depend on queue order — deterministic and wrong, which `M3.D.1` and `M11.D.5` do not
  catch. Very low cost. The one real risk is over-breadth: the rule must list the exemptions the tick is supposed to move
  within the batch (a tie's `bond_energy`, `M4.C.1d`'s reciprocal return of acute anxiety), or it forbids coupling the
  spec deliberately has. Evidence: Part 3, AP4 (paper Algorithm 1 lines 4–6 and 27, §IV on keeping learning off the
  interaction path).
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-285 ⟨later⟩ **Report the counterpart and a joint measure, and test a named channel as a path.** AP5: a result
  asserting that a change raises one member's readout reports beside it that readout for every other member of the ties
  that member acted on, plus at least one named joint readout (`M11.G`'s evaluation or `M11.C.52`'s total anxiety); and
  where the mechanism names an intermediate channel, the report states whether that channel moved in the asserted
  direction on the same seeds, and the mechanism is not described as supported if it did not. Lands in `M11.F` (framing
  that travels with an output) and `M17.G` (ensemble). Closes the largest inference hole in the criterion form: `M0.4`'s
  directional criteria include several that assert one member's readout (`M11.C.2`, `M11.C.4`, `M11.C.48`) and nothing
  requires the counterpart's number to appear, or that a named channel be checked rather than inferred from two
  co-moving arm differences. Evidence: Part 3, AP5 (paper Table V: +0.022 for the agent, −0.008 and p = 0.495 for the
  participant, Nash distance null; Table III: the agent's concession rate 38.9% in both arms while the human moved 28.1%
  → 37.4%; §VI.E: the named route null at ab = 0.0001, 13 of 14 exploratory models null, and the mechanism still
  described as supported). Two wording guards: name the joint readout's component, not the eight-part composite, or it
  becomes a box ticked by a schema; and ask for the channel's direction on the same seeds, not an estimated indirect
  effect, permitting `UNDETERMINED` — a twelve-agent ensemble has no power for mediation.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-286 ⟨later⟩ **Say how many persons carried a difference.** PS1: for every `M11.C` criterion whose readout is a
  move count, a move share or a tie-level quantity, the verdict record reports per arm the number of distinct persons and
  distinct ties contributing to the asserted difference, and a passing difference carried by fewer than a declared number
  of persons in either arm is reported as carried by those persons and not as a family-level result. Lands in `M11`'s
  verdict record beside `M11.D.18`, reading `M4.E.1`'s event stream through `M16.A` (no engine change, `M16.B.3` holds),
  extending `M17.B.5`'s per-person structure to the verdict itself. The distinction it protects is one the theory cares
  about: `M11.C.28` already refuses symptom count as a differentiation readout, but nothing stops "the coached arm
  produces more `CUTOFF`" from being a statement about Sam read as a statement about the family. Write it as a report
  field, not a criterion, and let the threshold flag rather than fail, or a legitimately single-member criterion
  (`M7.D`'s projection) is flagged routinely. Evidence: Part 4, PS1 (paper Tables 3–4: Top-1 flat at 0.891–0.913 across
  four held-out clusters while teleport recall runs 0.13–0.26, and the authors' own reading that 10–14 people per group
  are descriptive diagnostics rather than causal effects, §4.5; Table 1's constant predictor at 0.739 beside a model at
  0.754).
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-287 ⟨later⟩ **Separate the act selected from the act delivered.** PS2: where `M4.C.1f`'s social cues are on, the
  run log and `M11.G`'s readout report, per ordered pair and move kind, the count and share of deliveries in which what
  the receiver appraised differed in kind from what the sender emitted (a private hop arriving with tone, face and
  posture empty; a `TRIANGLE` arriving at a different third party), and no criterion attributes its direction to
  selection alone above a declared share without a selection-only arm to separate them. Lands as an observer record class
  in `M16.A` over `M1.F.1`/`M4.E.1` (Phase D) and in `M11.G`'s schema with the guard beside `M11.1d` (Phase E). Scoped to
  cues-on runs, where `M1.F.4`'s per-hop fidelity is drawn; with the cues off the readout is near-vacuous, and `M4.C.1f`
  already requires an identical log then, so the scoping costs nothing. It is a precondition for reading `M4.C.7` and
  `M1.F.3` at all: if a third of deliveries arrive as a different event, a move-level arm difference is partly a
  difference in what arrived, not in what was chosen. Evidence: Part 4, PS2 (paper Table 5, §3.7, §4.5: Top-1 0.904
  beside DPO completion 0.332 and a move-frequency gap of 0.520). `M16.A.2` already logs `fidelity` and both timestamps,
  so the raw material exists and is unused.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-288 ⟨later⟩ **One recorded outcome class per person-tick, and a forced outcome is never a choice.** MG1: for
  each living person each fast tick the log contains exactly one outcome record, classed as a move chosen by the scored
  policy, a move chosen by the tie-break, a fallback (`M4.D.1f`), a sequence step deferred under `M5.D.9`, or `WITHHOLD`
  chosen by the policy; the ensemble report prints the census per person and per arm with living person-ticks equal to
  the sum of classes; a person-tick with no record or two fails the run as a log defect; and every move marginal,
  transition matrix and criterion readout carries tie-break and fallback outcomes as their own row rather than counting
  them as the move they resolve to, with `M11.D.18`'s rate read from the same census. Lands as `M16.A.3d` with the
  reporting half in `M11.D` and the census as an annex of `M11.G.1`. Corrects a latent defect: provenance is recorded per
  record (`M16.A.3c`) and the rate flagged (`M11.D.18`), but a criterion still pools those outcomes into the move they
  resolved to, so a gate that closes — a tie cut off, a third party unreachable — can read as a family that stopped
  pursuing. Evidence: Part 5, MG1 (paper §6.1, Table 3: 22/288 and 8/288 failed calls recorded as "no valid order", one
  fault moving both the economic and the compliance metric upward). Keep `WITHHOLD` a chosen class and state the
  distinction once in `M4.D.1` rather than re-deriving it in the report.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-289 ⟨later⟩ **Record the choice the policy reached for before the mask.** MG2: alongside the legal set
  (`M16.A.3b`), the selection record carries the score and rank of the outcome the policy would have ranked first before
  `M4.D.1e`'s mask, computed by the same scoring function and drawing nothing; where that outcome has no target under
  the current structure the score is recorded undefined rather than zero and those ticks counted separately; the ensemble
  report gives per move and per person the rate at which the pre-mask first choice was illegal and which legal outcome was
  enacted, and labels the quantity a **score**, never a probability. Lands as `M16.A.3e` with the readout in `M17.B`
  beside the transition matrix. This is the family-model analogue of the source's violation ledger and the theoretically
  more interesting half: a family whose structure forbids what its members reach for is a different claim about triangles
  and cutoffs than a family whose members do not reach, and the spec can currently report only the second. Cost: medium —
  scoring off the legal set needs every term of `M4.D.1` defined outside it, and `M4.D.1f`'s fallback exists because some
  scores are non-finite, so undefined is a frequent declared class; it must consume no draw (`M3.D.4a`'s per-event
  keying). Evidence: Part 5, MG2 (paper §3, §5). **Owner decision:** whether "the move the person was heading for and
  could not make" belongs in the family-evaluation readout or is a trace-level observation only — it has theory content,
  but it also invites reading a score as an intention, and the model has no intentions.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-290 ⟨later⟩ **A verdict is provisional until it replicates on a fresh seed block.** MG3: every `M11.C` verdict
  is reported from a primary block and confirmed on a confirmatory block disjoint from every seed the criterion has
  already used, sized before it is drawn and run after the primary verdict is recorded but before it is reported as a
  finding; both blocks' ranges, sizes and per-seed differences are recorded separately; a criterion holding on one and not
  the other is reported **NOT-REPLICATED** — neither pass nor fail, and never described as supported — and a mutant that
  turns a criterion red on one block only does not establish that the mechanism carries it. Lands in `M17.A` with the new
  verdict word in `M11.4`'s exceptions table beside `UNDETERMINED`; no engine change beyond seed bookkeeping in
  `M16.A.1`'s header. Corrects an assumption in `M17.A.1`: a precision-based stopping rule reads one seed stream and stops
  when the interval narrows, which is exactly where the running estimate sits furthest from zero, and with heavy-tailed
  per-seed differences a small block can look decisive. It is one-sided — only passes need confirming. Evidence: Part 5,
  MG3 (paper §6.2, Table 4: identical prompts, seeds and settings, zero failed calls, Factory bullwhip 6.9 → 113.4 and
  Supplier 8.0 → 134.2 across a rerun; the abstract's own statement that three seeds did not survive a rerun). `M5.D.3`'s
  T-158 already applies this to one criterion, so the project already believes it. **Owner decision on cost:** confirming
  every passing criterion roughly doubles the wall clock of the Phase C/D ensembles — all criteria a report cites as a
  finding (the reader's recommendation), only criteria that gate a phase, or only those above an effect margin?
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-291 ⟨later⟩ **Ensemble records are append-only, and a superseded verdict is never quoted bare.** MG4: each
  ensemble record is persisted under its own identity and never overwritten, carrying `M16.A.1`'s header plus the seed
  range and the criterion's verdict as rendered; a record replacing an earlier one names the earlier identity and the
  declared reason (a spec revision, a defect fix, a changed readout, a changed seed set); any document quoting a verdict
  cites the record identity, and a verdict from a superseded record is labelled superseded where it is quoted. Lands as
  `M16.A.11` with the citing rule in `M11.F`'s framing block; the engine still writes nothing (`M16.B.1`), so this is the
  runner's and the repository's obligation. The cheapest rule in the batch and the one that protects every other finding:
  the project already regenerates records often and annotates them by hand in prose ("Regenerated at revision 32",
  "superseded at revision 12"), and `M11.F.9(c)`'s independence requirement cannot be audited after the fact without it.
  Evidence: Part 5, MG4 (paper §6.3(iv) on keeping superseded runs and an experiment log; §6.2 as a worked retraction).
  Scope it to records carrying a verdict or cited by a report, so a routine regeneration is not a clerical tax.
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-292 ⟨later⟩ **A composite criterion's composition dimension may not read "unaudited".** PW1: a composite or
  mixed criterion (`M11.5`) to be reported as a finding must carry the family-composition dimension in its `M17.G.1` audit
  record as **held**, **sensitive** or **reversed** over at least one structurally different instance — a second family
  under `M2`, a redraw of `D0` with a different topology or sibling structure, or an `M15` import — and never as
  **unaudited**; a direction that holds on the reference family and reverses or vanishes on the other instance is reported
  as a property of that composition, not as a mechanism finding. Lands as `M17.G.1a`, read with `M11.5` and
  `M15.D.2`–`M15.D.4`, arriving in Phase E under `M13.3`. Corrects a default: `M17.G.1` admits "unaudited" for every
  dimension today, so a composite pass can be reported as a finding with its composition never perturbed. The second
  instance must be declared before the first result is seen (`M10.B.4`), and "reversed" is a reportable outcome, never a
  reason to try a third. Evidence: Part 6, PW1 (paper App. B.2: group–priority association Cramér's V .928 → .063 without
  persona descriptions; §4.2/App. B.9: the acceptance contrast +17.25 → −15.50 pp, reversing the ranking).
  ⟦batch: paper-method proposals (2026-11-23)⟧
- [ ] T-293 ⟨later⟩ **One guardrail readout per criterion, with a bound declared in advance.** PW2: each `M11.C`
  criterion names, beside its primary directional readout, at least one guardrail readout — a quantity its own mechanism
  does not target, computed on the same arms — with a bound graded `[I]` and recorded in config before the run; a pass
  whose guardrail breaches its bound is reported as a pass with a stated cost and not as support for the mechanism. Lands
  as `M11.2a` beside `M11.2` and `M11.5`, with the bound in `M10.B`. `guardrail` occurs zero times in the spec, and the
  live instances are the criteria where the mechanism has a known price: `M11.C.49` (a low-level `I-POSITION` raises
  family anxiety; the mover's own estimated `basic_level` must not fall as the price), `M11.C.4` (cutoff relieves now, and
  family total anxiety at the next nodal event is currently the criterion's second limb rather than a bound), `M11.C.12`
  (the couple's differentiation must not be what paid for the symptom relief). Guards against the obvious failure mode:
  the bound is declared before the first run (`M10.B.4`) and the guardrail must be a quantity the mechanism does not
  target, which also keeps it outside `M11.1d`'s outcome-directive audit. Evidence: Part 6, PW2 (paper §2.1's three
  outcome classes, App. B.1 Table 2, App. B.9).
  ⟦batch: paper-method proposals (2026-11-23)⟧
