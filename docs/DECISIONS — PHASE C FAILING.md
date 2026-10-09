# Decisions — Phase C's failing criteria

> Written 2026-10-09, step 3 of `docs/plan_phase_c_failing_criteria.md`, and revised the same day after the batch
> review. **Decided 2026-10-09: the owner approved every recommendation** (S (a)/(b), X1 via D7, Q2 (a), Q3 (b), Q4/Q5 rerun after S).
> Evidence: `docs/phase_c_diagnostic_record.md` (generated, cached, never a verdict); `d0-…` to `d6-…` name its
> sections. The fixture's level comes from `config/bowen/fixtures/triad.md`, and the act layers from
> `config/bowen/event_kinds.md`; every other number is from the record, rounded. Nothing in the engine, config or
> criteria has changed.

## Results of the approved steps (2026-10-09)

Numbers are from `docs/phase_c_ensemble_record.md` and `docs/phase_c_diagnostic_record.md` (sections named), both
regenerated after the changes below. Every change is a pre-declared test restatement, logged as post hoc in
`config/bowen/criteria.md` and at the readout in `src/bowen/ensemble/criteria.py`.

**S — done, by (b) for every criterion, not (a).**
- (a) was measured first (`s-scripted-weeks`, 500 seeds, no tie held). The column is the first week at which the
  arm's acts would not all be made, or t0 + 1 if never. Its minimum is 1 in every row but one, so the latest week
  legal in every seed is week 0. The exception is `M11.C.29`'s baseline (four withdrawals), where the minimum is 0:
  no week works for it.
- Week 0 was not used. Every member starts on their chronic floor (checked on all three fixtures), and decay lifts
  anyone below it back, so no relief can show there. One run with every act at week 0 was made and its verdicts
  were seen before it was discarded (`M11.C.3`'s pair relief came out at +0.09); it is not kept.
- So the fallback applies everywhere. The declared weeks stand, and in both arms the ties the scripted acts cross
  are held open from week 0 until the act (`M11.C.29`: until its last withdrawal). `CUTOFF` across a held tie
  is not in either member's legal set (`Forced`, `legal_outcomes(excluded=…)`).
- The ensemble record now reports each arm's scripted acts not made, and why, beside the verdict. "Made" means
  emitted as scripted, or an I-POSITION sequence begun (`Forced.outcomes`); an act placed in the week's selections
  can still be rewritten afterwards. Every act is made in every seed except `M11.C.29`'s: 12 of 600 in the
  baseline (8 owed a step, 4 rewritten) and 8 of 150 in the treatment (5 owed a step, 3 rewritten). None was
  illegal. Its actor has full systems perspective and can be inside their own I-POSITION sequence with the same
  person, so on a step week they select nothing, and an act toward that person is rewritten to `STAY-IN-CONTACT`
  (`M5.D.9`).

**Verdicts after S, Q2 and the C.29 readout** (before → after):

| Criterion | Before | After |
|---|---|---|
| `M11.C.3` | PASS, pair −2.22, third +4.02 | PASS, pair −5.27 ± 0.51, third +8.85 ± 0.64 |
| `M11.C.4` | FAIL, now −1.02, later +0.07 | FAIL, now −2.55 ± 0.32 holds; later −0.59 ± 1.0, p 0.84 |
| `M11.C.5` | FAIL | FAIL, reaction +8.45 ± 16 (p 0.10); third's load +6.12 ± 10 (p 0.08) |
| `M11.C.27` | 1 of 4 cells | 1 of 4: only `[unstable,remove_one]` passes; `[stable,remove_one]` is −0.24 ± 0.06, the opposite way |
| `M11.C.29` | FAIL, readout pinned at 0 | FAIL, budget +0.04 ± 0.10; symptom weeks −0.23 ± 0.36 (p 0.37) |
| `M11.C.35` | PASS | PASS, +0.33 ± 0.06 |
| `M11.C.41` level limbs | FAIL (share fell) | **PASS**, `reactive_over_chance` +0.065 ± 0.005 and +0.067 ± 0.005 |
| `M11.C.41` stress limbs | FAIL | FAIL, `reactive_over_chance` −0.0008 and +0.0016 |
| `M11.C.42` | FAIL, +0.26 | FAIL, +0.12 ± 0.40 |

**X1 (D7, `d7-rest-state`) — members with several ties have no calm state near the floor; recommend (b), a
finding.**
- With no spell and no scripted act, the triad's f, m and c rest at a mean excess of 8.4–8.9 over the floor at
  their own level (40), within 5 points of it in 15–28% of weeks. The excess falls as level rises but stays well
  above the floor: 9–14 at level 20, 7–8 at 60, 6.6–7.5 at 80, 5.5–6.7 at 95, and never within 5 points in more
  than 30% of weeks.
- On `family_phase_c` at its own levels, the parents rest at 11.9–12.4 and the others at 4.5–10.3.
- Members with one tie rest near the floor: the triad's grandparents at about 3, Bruno (one cut tie) at 0.3–2.7.
- The excess comes from the members' own acts on one another: appraisal of delivered acts holds about +8, relief
  from cutoff and distance about −3 to −6, competing urges about +1.5. The standing load the constants were
  designed around (`M4.A.5`'s self term and the ties' "too little" side) holds only about 1.5–2.7.
- So (a), a calmer declared starting state, would not help: every run already starts on the floor, and the
  excess re-forms from the policy's own acts within the 20-week burn-in.
- What this means for `M11.C.44`/`.45`: the calm arm is the triad's resting state, about 8–9 above the floor, and
  the declared spell adds about 2 (`d5-spell-effect`). The arms differ by little. Whether the spell is too light is
  a question about a declared `[I]` input. Changing it now would be tuning after the freeze, so it is left to the
  owner (TODO).

**Q2 — done.** `M11.C.41`'s readout is now `reactive_over_chance`: per selection, whether the act chosen was
reactive, less the reactive share of that selection's legal set, averaged. A chooser at random scores 0 at any set
size. The first restatement (reactive selected over reactive offered) was found by the batch review to score 1/N for
such a chooser, so it rose whenever the legal set shrank; it was replaced before being relied on. Both level limbs
pass. The stress limbs do not move.

**`M11.C.29`'s readout — restated.** It now counts the weeks the third person's symptom is active (onset,
`M1.A.6`, until the load falls below the re-arm fraction of the threshold). The old count, weeks with any symptom
accumulation, was positive in every week above the floor, and D7 shows the model never rests on it. The new
readout moves (−0.23 ± 0.36) but does not separate the arms.

**Q3 (`q3-act-effects`, 50 seeds, `M11.C.3`'s week and holds) — the act relieves the pair; the credit does not
reward it.**
- One week after a `TRIANGLE`, against `STAY-IN-CONTACT`: sender −3.6, partner −1.6, third +8.9. That is what
  `M1.C.1` states, so its transfer needs no fix.
- `DISTANCE` relieves the sender more (−6.0) and costs the partner +1.8. `CONFLICT` and `PURSUE` relieve no one.
- The learner credits an act with the actor's own relief plus `cross_person_weight` (0.5) times the others' mean
  relief (`M4.D.6e`). The recruited third's +8.9 enters that mean, so a `TRIANGLE` (own −3.6, third +8.9) is
  credited well below a `DISTANCE` (own −6.0, partner +1.8). This is why triangling is not reinforced
  (`M11.C.42`, `.45`), and it is a property of `M4.D.6e` under §8's reading, not a defect in the act.
- **Owner decision needed:** (a) report it as a finding, or (c) exempt the recruited third from the cross-person
  credit (a spec change to `M4.D.6e`). Recommendation: (a). The spec's cross-person credit is
  deliberate, and exempting one role for one act would write the outcome the criterion tests.

**Q4/Q5 — rerun under S.** With every act made, nothing is diluted any more. `M11.C.4`'s later limb
(−0.59 ± 1.0) and three of `M11.C.27`'s cells still fail, one of them the opposite way: they are now class C,
findings about the model, unless the owner wants a longer declared horizon (B). `M11.C.5` points the claimed way on
both readouts (p 0.10 and 0.08), but its verdict is a converged FAIL, not UNDETERMINED. The approved rule allowed a
longer horizon only if it stayed undetermined, so none was tried; choosing one after seeing this result would be
tuning. Owner decision (TODO).

## The owner's decisions on the results (2026-10-09)

1. **Q3:** the owner described how triangling works (either of a pair may seek their own third; the seeker's
   anxiety goes down if the third is a positive experience; a third aligned with the partner, or very anxious,
   does not help). The model differs in what the act relieves and in what the seeker learns from. Written up as a
   spec proposal for approval: `docs/PROPOSAL — TRIANGLE RELIEF AND CREDIT.md`. Nothing is built yet.
2. **X1:** accepted as a finding. Members with several ties rest well above the chronic floor at every level,
   from their own acts; the declared spell is not revisited.
3. **`M11.C.5`:** its converged FAIL at the declared horizon is reported; no longer horizon (the approved rule).
4. **`M11.C.4`'s later limb and `M11.C.27`:** a longer horizon allowed. Rule, fixed before running: one value,
   the longest any Phase C criterion declares (104 weeks), run once. Result: no effect remains.
   - `M11.C.4`'s later limb, nodal event at week 104: +0.14 ± 0.99 (p 0.76). Its "now" limb still holds (−2.55).
   - `M11.C.27`, read 104 weeks after the act: every cell within ±0.05 of zero, none passes. This includes
     `[unstable, remove one]`, which passed at 3 weeks (−0.28 ± 0.06) and is now +0.05 ± 0.06.
   - So a single act's effect on the pair's deviation is gone by two years, and a cutoff's later cost does not
     show at a nodal event within two years. Both are findings at this horizon. Whether `M11.C.27` should be read
     at 3 weeks (its passing cell) or 104 is the owner's to say; the record holds the declared 104.

## Summary (diagnosis, before the steps above)

| Criterion | Status | Cause, as far as the record shows | Decision |
|---|---|---|---|
| all criteria that script an act | **A** | The scripted act is skipped in 25–60% of seeds, because it is not legal that week; a skipped seed adds a difference of exactly 0 | **S** (first) |
| `M11.C.4` later limb | open, after S | Over the 8 seeds where the cutoff was made, the later cost is mixed (median −0.9 at week 30, +0.4 at weeks 120 and 240) | S, then perhaps B |
| `M11.C.5` | open, after S; perhaps **B** | Over the made seeds the target's median reaction rises with the run (−13, +26, +60 at 60, 120, 260 weeks), but only about half the seeds are positive and the means disagree | S, then Q5 |
| `M11.C.27` | open, after S | 30–40% of seeds skipped; the made-seed subsets are not paired across arms; read at 3 weeks only | S |
| `M11.C.42` | **C**, and S | The learner credits `TRIANGLE` less than other acts; 35% of seeds skipped | S, Q3 |
| `M11.C.45` | **C**, and X1 | The same credit gap; the calm arm may not be calm | Q3, X1 |
| `M11.C.29` | **A**, and S, X1 | The readout is pinned at its maximum (the third person is above their floor in 79 of 80 samples); 25–50% skipped | S, X1, then a readout |
| `M11.C.44` | open, X1 | The spell barely changes any act count | X1 |
| `M11.C.41` level limb | **A** | The readout conflicts with `M4.D.3a`: 5 of the 7 reactive acts sit in layers a lower level closes | Q2 |
| `M11.C.41` stress limb | possibly **B**, X1 | It points the right way, but the intervals cross zero at 150 seeds | X1 |

A = test defect, B = underpowered or too short, C = a finding about the model (plan §0). **This revision withdraws
three claims of the first version:** "no criterion is B", "a cutoff is a net relief at every horizon", and "TRIANGLE
does not relieve the sender". The first two came from medians over seeds in which the cutoff was never made. The
third read a learning credit as a causal effect.

---

## S — the scripted act is often not made (decide first)

**Evidence** (`d0-scripted-acts`, 20 seeds per arm). A criterion's scripted act is made only if it is legal that week (`M4.D.1e`); otherwise it is skipped and counted. It was skipped in:

| Criterion | Seeds skipped |
|---|---|
| `M11.C.3` (passing) | 9–10 of 20 |
| `M11.C.4` | 12 of 20, in both arms |
| `M11.C.5` | 5 of 20 |
| `M11.C.29` | 5 of 20 in the treatment arm; in the baseline arm, 3 fully and 7 partly |
| `M11.C.35` (passing) | 7–8 of 20 |
| `M11.C.42` | 7 of 20 |
| `M11.C.27` | 6–8 of 20 per cell |
| `M11.C.32` | 0 of 20 |

**The cause, from a hand check outside the record.** In C.4 seeds 0–5, the selection record at week 8 offered `REDUCE_CUTOFF` toward `b` in the five skipped seeds: the policy had already cut the tie, so a scripted `CUTOFF` was not legal, and the "no cutoff" baseline was cut too. The diagnostic record itself shows only the skip counts; why each other criterion's act is skipped is not yet measured.

**What it means.** In a skipped seed both arms run the same history, and the difference is exactly 0. Every affected criterion's effect is diluted toward zero by its skip rate. C.4 runs at about 40% of its seeds. The two passing criteria pass despite this; their effect sizes are understated.

**Options.**
- (a) **Script the act where it is legal in every seed.** Choose each criterion's scripted week as the latest week at which the act is legal in all seeds, measured by a diagnostic before any rerun and then declared.
- (b) **Hold the affected tie open until the scripted week.** Declare that the scripted actor's policy cannot cut that tie before t0. This changes the scenario, not the model.
- (c) **Analyse only seeds where the act was made in both arms**, and report the skip rate. This is not recommended: whether the act is legal depends on the seed's history, so the kept seeds are a selected sample.

**Recommendation: (a)**, with (b) for any criterion where no week works. Both are pre-declared test restatements (A), logged as post hoc. The made/skipped counts already appear in each arm's `moves` in the ensemble record, but no readout used them. The ensemble record should also report each criterion's skip rate beside its verdict, so the defect stays visible.

## X1 — is there a calm state? (hypothesis; D7 needed)

**Evidence, triad fixture only** (level 40, the fixture of C.29, C.42, C.44 and C.45):
- Every member is above the chronic floor (any positive excess) in 79 of 80 tick-start samples, in both arms; the first sample is the initial state.
- The third person's peak symptom load averages 217, against an onset threshold averaging 63 at that peak (`d1-c29-third-person`).
- With the declared spell removed, this is unchanged: 79 of 80, and an average 188 against 60 (`d1-c29-third-person-calm`).
- In C.45's calm arm, members are above the floor in 98.8% of samples, at a mean acute anxiety of about 54 against a floor of 45; the spell arm is about 56 (`d5-spell-effect`).
- The spell barely changes act counts: outside acts 63.0 against 62.5, inside acts 33.1 against 34.4, triangles 11.2 against 11.3 (`d5-c44-act-counts`).

**What it might mean.** In the triad, the calm arm may not be calm, so calm-against-spell criteria compare two tense states. Two limits on that:
- "Above floor" has no tolerance, and the probe has not measured by how much.
- The `family_phase_c` fixture (C.41, C.5) has not been measured at all.

**Recommendation: D7 before deciding.** D7 would measure:
1. the mean excess over the floor with a declared tolerance, on both fixtures;
2. what holds acute anxiety above the floor at rest: `M4.A.5`'s standing load, against each tie's deviation;
3. whether any level in range rests near the floor.

Then decide between (a) restating the arms with a calmer declared starting state and (b) reporting it as a finding. Reporting it as a finding would mean the resting state is a property of the model.

## Q2 — `M11.C.41`'s level limb conflicts with `M4.D.3a`

**Evidence.**
- At a lower level, mean acute anxiety rises as claimed: +16 and +18 in the two "lower level" cells (`d2-c41-unmodified`).
- The reactive share falls: −0.013 and −0.017.
- With availability held level-independent, the share rises: +0.003 and +0.010; the heavy cell then passes (`d2-c41-availability-fixed`).

**Cause.** Of the seven reactive acts, only `CUTOFF` and `DISTANCE` are in layer 0. `OVERFUNCTION` and `UNDERFUNCTION` are in layer 1, and `CONFLICT`, `PURSUE` and `TRIANGLE` in layer 2. `M4.D.3a` closes layers 1 and 2 as level falls, while the self-directed acts stay available.

**Options.**
- (a) Restate the readout as reactive acts selected over reactive acts offered by the legal set.
- (b) Accept it as a finding: a lower level narrows reactivity to its oldest forms.
- (c) Revisit `M4.D.3a`'s layering.

**Recommendation: (a).** C.41 runs no scripted act, so S does not affect it.

## Q3 — the learner credits `TRIANGLE` less than other acts (`M11.C.42`, `M11.C.45`)

**Evidence** (`d3-triangle-relief`, 20 seeds per arm; learned values per person and key):
- Under the current roles (report §8), a `TRIANGLE`'s closed learning signal averages −2.1 to −3.5, against −0.5 to −0.9 for other automatic acts.
- 23–32% of triangles are followed by relief, against 44–47% of other acts.
- Its learned values end at about −0.9 to −1.0, against −0.4 to −0.5.
- Under step 5's alliance reading (`d3-triangle-relief-old-roles`), triangles are credited like any other act (42–46% relieved).
- With `cross_person_weight` at 0 (`d3-triangle-relief-own-relief-only`), the gap narrows: 36–47% against 41–43%. In C.45's baseline the learned values are then close (−0.35 against −0.31).

**What it shows, and what it does not.** This is the credit the learner assigns: the actor's change in acute anxiety over the horizon, which includes decay and everything else that happened in those weeks. It explains why triangling is not learned under the current roles, which is enough for C.42 and C.45 to fail. It does not show what the act itself does. The run with the cross-person weight at 0 is a different trajectory, not a decomposition, so "part of the cost is the third's distress" is an inference.

**Options.**
- (a) Report it as a finding about §8's reading.
- (b) Measure the act's own effect with a paired forced-act comparison, as C.3 does, and fix `M1.C.1`'s transfer if it does not relieve the twosome as the spec states.
- (c) Exempt the recruited third from `M4.D.6e`'s cross-person credit.

**Recommendation: (b)** after S, since C.42 also skips 35% of its seeds.

## Q4 — `M11.C.4`'s later limb and `M11.C.27`: decide after S

**Evidence, made seeds only** (`d6-horizons`; C.4 made its cutoff in 8 of 20 seeds):
- The cutoff relieves now: mean −3.2, median −3.1.
- Family anxiety at the nodal event, treatment minus baseline, has median −0.9, −1.8, +0.4 and +0.4 at weeks 30, 60, 120 and 240, with 38–50% of seeds positive. At week 240 the mean is +24.8, driven by one seed at +206. Eight seeds cannot settle that.
- C.27 is read 3 weeks after its act. In its remove-one cells (14 of 20 seeds made in both arms, the same seeds), the cutoff removes the "too much" side of the pair's deviation at once (f 0.105 → 0, m 0.155 → 0.005), while the "too little" side rises by less (f 0.169 → 0.199, m 0 → 0.045), so the sum falls (`d4-c27-deviation-terms`). In the add-third cells the made seeds differ between arms (14 against 12), so those are not paired. No longer horizon was tried.

**Recommendation:** rerun both under S's fix. Only then consider a longer declared horizon (B) or a finding (C).

## Q5 — `M11.C.5`: likely too short

**Evidence, made seeds only** (15 of 20):
- The target's reaction, treatment minus baseline, has median −13, +26 and +60 at 60, 120 and 260 weeks, but only 47%, 53% and 53% of seeds are positive. The means disagree (−15, −64, +252) and are driven by heavy tails (largest seed +2,950 at 260 weeks).
- The third person's load has median −28, −11 and +61, positive in 27%, 47% and 73% of seeds.
- The third person's load may build over years. The target's reaction is not settled by 15 seeds.

**Recommendation:** after S, rerun C.5 over its full ensemble at the declared horizon. Consider one declared longer horizon (B) only if that rerun is still undetermined. C.5 runs on `family_phase_c`, which X1's evidence does not cover.

---

## What the owner is asked to decide

1. **S:** (a) script each act at a week where it is legal in every seed (recommended), (b) hold the tie open, or (c) made-seed analysis (not recommended).
2. **X1:** run D7 first (recommended).
3. **Q2:** (a) the share of offered reactive acts (recommended), (b) a finding, or (c) revisit `M4.D.3a`.
4. **Q3:** (b) a paired forced-act measurement of `TRIANGLE`'s own effect, after S (recommended), (a) a finding, or (c) exempt the third.
5. **Q4 and Q5:** rerun after S; then decide on a longer horizon for C.5 only if it is still undetermined.

**Corrections made while diagnosing.**
- Report §3.6 said the third person in C.29 accumulates no symptom weeks. They are above zero in every week of both arms. Corrected in the report and TODO.
- The first version of this memo drew Q4 and "no criterion is B" from medians over unmade seeds. Withdrawn above.
