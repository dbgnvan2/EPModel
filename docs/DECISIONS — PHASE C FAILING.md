# Decisions — Phase C's failing criteria

> Written 2026-10-09, step 3 of `docs/plan_phase_c_failing_criteria.md`. **Awaiting the owner's decisions.**
> Evidence: `docs/phase_c_diagnostic_record.md` (generated, cached, never a verdict). The rows named below as
> `d1-…` to `d6-…` are its sections. Nothing in the engine, config or criteria has changed. Every number here is
> copied from that record and rounded.

## Summary

| Criterion | Outcome | One-line cause | Decision needed |
|---|---|---|---|
| `M11.C.29` | **A**, blocked by X1 | The readout is pinned at its maximum: everyone is above threshold all run | X1, then a readout |
| `M11.C.41` level limb | **A** | The readout conflicts with `M4.D.3a`: 5 of the 7 reactive acts sit in layers a lower level closes | Q2 |
| `M11.C.41` stress limb | blocked by X1 | The spell adds little anxiety to a system already above its floor | X1 |
| `M11.C.44` | blocked by X1 | The spell changes no act counts | X1 |
| `M11.C.45` | **C**, and X1 | `TRIANGLE` does not relieve; there is also no calm arm | Q3, X1 |
| `M11.C.42` | **C** | `TRIANGLE` relieves the sender no more than other acts, so it is not learned | Q3 |
| `M11.C.4` later limb | **C** | A cutoff's cost never outweighs its relief, at any horizon to 240 weeks | Q4 |
| `M11.C.27` remove-one cells | **C** | The same: a cutoff removes impingement at once; "too little" grows slowly | Q4 |
| `M11.C.27` add-third cells | **C** | Adding a third barely moves the pair | Q4 |
| `M11.C.5` | **C**, perhaps X1 | No consistent reaction at 60, 120 or 260 weeks | Q5 |

A = test defect, B = underpowered or too short, C = a finding about the model (plan §0). **No criterion is B.**
Every horizon that was lengthened (C.4 to 240 weeks, C.5 to 260) left the median difference at 0.

---

## X1 — the model has no calm state at the fixtures' levels (most consequential)

**Evidence.**
- In M11.C.29's triad, every member is above their chronic floor in 79 of 79 weeks, and their peak symptom load is about 3 times the onset threshold: the third person at 217 against 63 (`d1-c29-third-person`).
- Removing the declared spell changes almost nothing: still 79 of 79 weeks, and the third person's load is 188 against 60 (`d1-c29-third-person-calm`).
- In M11.C.44/.45's "calm" arm, members are above the floor in 98.8% of weeks, at a mean acute anxiety of about 54 against a chronic floor of 45. The spell arm is 56 (`d5-spell-effect`).
- Act counts are the same in both arms: outside acts 63 against 62.5, inside acts 33 against 34, triangles 11.2 against 11.3 (`d5-c44-act-counts`).

**What it means.** Every criterion that compares calm against a spell (C.44, C.45, C.41's stress limb), or that reads a symptom's time course (C.29), compares two saturated states. They cannot pass whatever the mechanisms do. Either the fixtures start too tense, or the model has no reachable resting state near the floor at these levels (about 40).

**Options.**
- (a) **Restate the arms (A).** Give these criteria a declared starting state that rests near the floor (for example, a higher level or a less tense starting tie). Declare it before the rerun, and log it as post hoc.
- (b) **Treat it as a finding (C).** The resting state is set by `M4.A.5`'s standing load and the tie appraisal (`M4.C.1`), and restating the fixtures would hide it.

**Recommendation: first one more diagnostic (D7, minutes)**, then choose. D7 would decompose what holds acute anxiety above the floor at rest (the standing load, against each tie's deviation), and would test whether any level in range reaches a calm state. If one does, (a) is honest. If none does, it is (b), and the spec or model question goes ahead of every criterion above.

## Q2 — `M11.C.41`'s level limb conflicts with `M4.D.3a`

**Evidence.**
- At a lower level, mean acute anxiety rises as claimed: +16 and +18 in the two "lower level" cells (`d2-c41-unmodified`).
- But the reactive share falls: −0.013 and −0.017.
- With availability held level-independent, the share rises: +0.003 (light) and +0.010 (heavy; this cell then passes) (`d2-c41-availability-fixed`).

**Cause.** Of the seven reactive acts, only `CUTOFF` and `DISTANCE` are in layer 0. `OVERFUNCTION` and `UNDERFUNCTION` are in layer 1, and `CONFLICT`, `PURSUE` and `TRIANGLE` in layer 2. `M4.D.3a` closes layers 1 and 2 as level falls, so a lower level removes most reactive options, while the self-directed acts stay. The criterion and `M4.D.3a` pull against each other within the spec.

**Options.**
- (a) Restate the readout as the reactive share of what the legal set offered: reactive acts selected, over reactive acts available. Declared before the rerun.
- (b) Accept it as a finding: in this model, a lower level narrows reactivity to its oldest forms.
- (c) Revisit `M4.D.3a`'s layering of the reactive acts.

**Recommendation: (a)**, because the criterion's claim is about the pressure toward reactivity, and (a) measures that without undoing `M4.D.3a`. (b) is defensible if the owner reads Bowen as saying that the poorly differentiated have fewer forms of reactivity, not more of it.

## Q3 — `TRIANGLE` does not relieve the sender under the spec's roles (`M11.C.42`, `M11.C.45`)

**Evidence** (`d3-triangle-relief`, 20 seeds per arm):
- Under the current roles (the 2026-10-08 `TRIANGLE` decision, report §8), the learner credits a `TRIANGLE` with a mean signal of −2.1 to −3.5, against −0.5 to −0.9 for other automatic acts.
- 23–32% of triangles relieve, against 44–47% of other acts.
- Its learned value ends at about −0.8 to −1.1, against −0.3 to −0.4.
- Under step 5's alliance reading (`d3-triangle-relief-old-roles`), triangles look like any other act: 42–46% relieve.
- With only the actor's own relief credited (`cross_person_weight` 0; `d3-triangle-relief-own-relief-only`), the gap narrows but stays: signal −0.5 to −1.5 against −0.3 to −0.5, and 36–47% relieve against 41–43%.

**What it means.**
- Part of the cost is the loaded third person's distress, credited through `M4.D.6e`.
- The rest is that the sender's own relief from triangling is no better than from other acts.
- A learner that repeats what relieves therefore cannot learn to reuse triangles (C.42), or to use them more under a spell (C.45). The criteria fail because, as specified, the act is not relieving.

**Options.**
- (a) Report it as a finding about §8's reading.
- (b) The spec intends a triangle to relieve the twosome (`M1.C.1`'s transfer). If so, the transfer as built does not achieve that, and fixing it to do what `M1.C.1` states is implementing the spec, not tuning. Diagnose the transfer's size and route first.
- (c) Exempt the recruited third from `M4.D.6e`'s cross-person credit.

**Recommendation: (b), diagnosis first.** The corpus is clear that triangling relieves the twosome. So a model in which it does not is more likely a defect in the transfer than a finding. (c) changes a rule's scope, so it is the owner's call on the theory.

## Q4 — a cutoff is a net relief at every horizon (`M11.C.4` later limb, `M11.C.27`)

**Evidence.**
- With C.4's nodal event at 30, 60, 120 and 240 weeks, the median difference in family anxiety is 0 at every horizon, with 15–20% of seeds positive. The mean at 240 weeks (+9.9) comes from one seed (`d6-horizons`).
- In C.27's remove-one cells, the cutoff drives the "too much" side of the pair's deviation to about 0 (f 0.10 → 0.03, m 0.11 → 0.003). The "too little" side rises only modestly (f 0.21 → 0.23, m 0.09 → 0.13). Net deviation falls, where the stable cell expects a rise (`d4-c27-deviation-terms`).
- Adding a third (the add-third cells) moves the pair by a few hundredths on either side.

**What it means.** `M4.C.1c`'s accrual on a severed tie never builds enough to outweigh the relief of losing the impingement. This is one mechanism behind two criteria's failures, and a longer run does not change it (not B).

**Options.**
- (a) Report it as a finding.
- (b) If `M4.C.1c` is meant to make a cutoff's cost exceed its relief over time ("cutoff trades now against later"), then the accrual as built does not, and fixing it is implementing the spec. Diagnose first.
- (c) Restate C.27's remove-one cells. A cutoff may not be the right operationalisation of "removing one".

**Recommendation: (b), diagnosis first**, because the spec's own wording for C.4 states the trade. The add-third cells are the same question as Q3: if a triangle does not move the pair, it does not stabilise or destabilise it.

## Q5 — `M11.C.5`: no consistent reaction

**Evidence** (`d6-horizons`). Target reaction: the median difference is 0 at 60, 120 and 260 weeks, with 35–40% of seeds positive and heavy tails (the mean swings from −48 to +189). The third person's load: median −15, 0 and +31, with 20–55% positive.

**Recommendation: decide after X1.** The target is above its floor throughout (X1), so a "reaction" is hard to tell apart from the baseline. If X1 is restated, rerun C.5 under it. Otherwise report it as a finding: the change-back ladder (`M5.E.0`) does not form.

---

## What the owner is asked to decide

1. **X1:** run D7 first (recommended), or decide (a) or (b) now.
2. **Q2:** (a) the share of the offered reactive acts (recommended), (b) a finding, or (c) revisit `M4.D.3a`.
3. **Q3:** (b) diagnose `M1.C.1`'s transfer (recommended), (a) a finding, or (c) exempt the third from cross-person credit.
4. **Q4:** (b) diagnose `M4.C.1c`'s accrual (recommended), (a) a finding, or (c) restate C.27's remove-one cells.
5. **Q5:** after X1 (recommended).

**Corrections made while diagnosing.** Report §3.6 said the third person in C.29 accumulates no symptom weeks. In fact they are symptomatic in every week of both arms; the +0 was a difference, not a count. Corrected in the report and TODO.
