# Suggested enhancements from three silicon-sampling papers, in plain language

Plain-language companion to `SWEEP_READING_REPORTS_2026-10-05b.md`. That file has the requirement wording, the evidence
and the spec references. None of these is in the spec. Each is a proposal for the owner, held for revision 11 like the
earlier batches.

The three papers study "silicon sampling": using a language model to stand in for survey respondents. EPModel does not
do this, and no LLM is in its decision path (`M3.D.6`). What transfers is how these papers check whether a simulated
population can be trusted, and the ways they found it fails.

- **Total Simulated Survey Error** (Sen et al.) lists where a simulated survey can go wrong, stage by stage.
- **Before You Poll** (Wali & Tayyab) tests whether simulated people change opinion in the same direction as real
  people after the same briefing.
- **Silicon Sampling Country Assumptions** (Wang) shows that simulated respondents answer from their country label, not
  their own attributes.

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children
Sam (15) and Lily (10). They show how a proposal would work. They are not results.

---

## 1. Tests

### SC1. Test that each person acts from their own history, not the family average *(from Silicon Sampling)*
**What it proposes:** `M1.A.7` already says a person's reactivity must not be computed from a family average, but
nothing tests it. Add a mutant that replaces each person's own inputs (reactivity, witnessed history, position) with
the family mean, keeping the family's overall level unchanged. Criteria that depend on differences between members,
such as symptoms concentrating on the projection target (`M11.C.2`), must fail under it. Every criterion is then
labelled as either needing within-family differences or working at family level. A family-level criterion cannot be
cited as evidence for a mechanism that works through one person's history.
**Why:** In the paper, the country label carried almost all of the match with real data. Individual answers were mostly
noise (95% of within-country variation was run-to-run noise).
**Example:** Replace Sam's and Lily's witnessed histories with their average. If Sam still ends up the projection target,
something other than his own history is choosing him.

### SC2. Split person-level results into "between families" and "within a family" *(from Silicon Sampling)*
**What it proposes:** For each person-level readout, report how much variation lies between families and how much lies
between members of the same family. Also report how much of the within-family difference repeats across random seeds
when the starting family is held fixed. That share should be higher in the reference family than in a family whose
members start identical.
**Why:** If the difference between siblings changes with every seed, it is noise, not position.
**Example:** Run the same family under 20 seeds. If Sam ends up worse off than Lily in 19, the difference is reproducible.
If it is Sam in 10 and Lily in 10, it is not.

### SC3. Every rating scale names its two ends in words *(from Silicon Sampling)*
**What it proposes:** Every rating imported from the family diagram must name both ends of its scale ("1 = cut off,
5 = fused"), and the importer rejects a rating without them. Every readout states which end means "more". A test
imports a reverse-coded fixture and checks it produces the same internal state.
**Why:** In the paper, 10 of the 11 results that came out backwards were on reverse-coded items. Naming the ends fixed it.
A silently reversed scale flips the sign of a result.
**Example:** One diagram rates closeness 1 = very close; another rates it 1 = distant. Without named ends, Tom and Ann's
tie imports as close in one and distant in the other.

### BP2. An inert coach arm, to separate "a third person arrived" from "coaching landed" *(from Before You Poll)*
**What it proposes:** Each test that compares a coach arm with a no-coach arm also runs a third arm. In it the coach is
present with the same timing, contacts and ties, but coach quality is null. The result must hold against this inert
arm, not only against the no-coach arm. The inert arm's difference from no-coach is reported as the effect of the form
alone.
**Owner question:** Can coach quality (`M1.E.7d`) be set to null as an ordinary setting? If it can only be done by
changing the model, the arm becomes a structure test under `M11.4f`.
**Why:** In the paper, irrelevant text in the same format moved the models' answers 3.6 times as much as the real
briefing moved humans. In EPModel, adding a person creates triangles, which can relieve a couple whatever the coach does.
**Example:** Ann starts seeing a coach. Her marriage calms. Was it the coaching, or just that she now has someone outside
the marriage to talk to?

### TS2. Check each result under a second reasonable way of measuring it *(from Total Simulated Survey Error)*
**What it proposes:** Before a criterion is run, declare its main readout and at least one alternative way of measuring
the same thing from the corpus. Report the direction under both. If they disagree, record the result as sensitive. The
main readout cannot be changed after results are seen, and readouts the corpus marks as traps (such as counting
symptoms) cannot be used as alternatives.
**Why:** In the paper, the "best" model changed when the scoring measure changed.
**Example:** Does coaching reduce Sam's symptoms? Measured as time until symptoms appear, yes. Measured as total symptom
load over ten years, perhaps not. Both should be reported.

---

## 2. Reporting rules

### BP1. Before claiming a result for an imported family, re-run the tests it depends on for that family *(from Before You Poll)*
**What it proposes:** A result reported for an imported family, or any family other than the reference family, comes with
the acceptance tests its mechanism depends on, re-run on that family and marked held, reversed or undetermined. If a
supporting test reverses there, the result is flagged.
**Owner question:** Is the list of supporting tests worked out from `M11.1d`'s minimal rule sets, or declared by hand?
**Why:** The model is tested on the reference family. A mechanism that behaves correctly there may not behave correctly
in a family on the other side of a turning point.
**Example:** A coaching result for an imported family relies on triangles relieving a pair. If that test reverses in
this family, the coaching result is flagged.

### BP3. Say which parts of a corpus intervention the arm actually implements *(from Before You Poll)*
**What it proposes:** Each test whose expected direction comes from the corpus account of an intervention lists the
parts of that account the arm implements and the parts it leaves out. A pass or failure applies only to the parts
implemented.
**Why:** In the paper, humans had a weekend of group discussion; the models got only the briefing.
**Example:** The corpus describes coaching as a series of sessions, a relationship with the coach and the family's own
follow-through. If the arm models only the sessions, the result says nothing about the rest.

### BP4. Keep "the starting family looks right" separate from "the change goes the right way" *(from Before You Poll)*
**What it proposes:** Checks that the simulated population matches corpus distributions are reported in their own
section, apart from direction-of-change results. Neither may be cited as support for the other. Could be folded into
CB1 or OM1.
**Example:** Ann and Tom's imported family matches the corpus spread of differentiation. That says nothing about whether
coaching works in their family.

### BP5. Look at which runs came out backwards *(from Before You Poll; low priority)*
**What it proposes:** For seeds where the result reversed, report how reversals are distributed across tests and whether
they cluster in certain starting families. A two-humped distribution must not be summarised by its average.
**Why:** It tells whether reversals come from particular families (a turning point) or from particular tests.

### SC4. Test each part of an imported diagram both added alone and removed alone, and with wrong values *(from Silicon Sampling; amends OM2)*
**What it proposes:** OM2 removes each part of an import (structure, tie states, dated events, ratings) and sees whether
the result changes. This adds two arms: the part added alone to a blank family, and the part filled from a different
family. If wrong values give the same result as true ones, the model is not reading that part.
**Why:** In the paper, the country label moved results far more when added alone than when removed, because other
variables already carried the same information. A wrong label made results worse than no label.
**Example:** Give the model Tom and Ann's family structure with the Smith family's tie ratings. If the result is the same,
the ratings are not doing anything.

### SC5. Compare sibling outcomes with a trivial predictor *(from Silicon Sampling)*
**What it proposes:** For tests that credit history or position for a difference between members, also report how well
two simple rules predict the same ordering: ranking by starting differentiation, and the projection-target rule applied
at birth. If either rule matches the simulation, the relationship process is not credited. Could be folded into SC1.
**Why:** In the paper, a simple average of neighbouring countries matched or beat the model.
**Example:** If Sam ends up worst off, and Sam also started with the lowest level, the simulation may only have re-sorted
the children by their starting values.

### TS1. A result cannot carry a high claim grade with a whole stage unchecked *(from Total Simulated Survey Error)*
**What it proposes:** Group the robustness checks in each result's audit record into five stages: who is in the family,
the mechanisms, the arm definition, how it is measured, and comparison with corpus bounds. A result with a whole stage
unchecked is graded exploratory at most.
**Why:** The paper found the largest effect in one stage (who the simulated people are), and that choices across stages
interact.

### TS3. Flag tests that check the same corpus sentence the mechanism was built from *(from Total Simulated Survey Error)*
**What it proposes:** Record which corpus passage each test comes from and which passage each rule it depends on comes
from. Where they are the same, the pass is reported as "the passage was implemented", not as something the model
derived.
**Why:** In the paper, the persona variables and the target came from the same survey, which inflated success.
**Example, verified in the spec:** `M11.C.25` (pole assignment is independent of sex) and `M2.A.0g` (the rule that
makes it so) quote the same sentence from FE07.3. Passing that test shows the rule was coded, nothing more.

### TS4. No weighting runs toward the corpus *(from Total Simulated Survey Error)*
**What it proposes:** Seeds or starting families must not be weighted by how well they match corpus bounds or
distributions. Only weights declared before the run are allowed, and the unweighted result is shown beside them.
**Why:** PW3 forbids filtering runs and PD4 forbids selecting them. Weighting is the same move done gradually. In the
paper, reusing weights built for another purpose made results slightly worse.

---

## 3. Narrator line only (does not enter the v2 spec)

- **TS-X1.** Narrator results record the exact model version, cut-off date, settings and number of runs, never use an
  unversioned API, and state whether any narrated case resembles a published corpus case from before the model's cut-off.
- **BP-X1.** Test a narrator on change across an event, with the engine as the reference: does the narrated change go the
  same way as the engine's? Include an inert-event control, and fail a narrator that matches overall but reverses for
  one class of move.
- **BP-X2.** Before any LLM persona is used for maturity or reactivity, give it the same stressor before and after, plus
  an inert control and a swapped target (own parent against a peer), and re-test any fix separately for each model.
- **SC-X1.** Narrator recovery is reported within role or age groups and against a baseline that uses the label only.
  Otherwise a narrator can pass by answering from "15-year-old" rather than from Sam's state.
- **SC-X2.** Any number scale given to an LLM names its two ends in words, and reverse-coded items are reported apart.

---

## Owner questions raised by this batch

1. **BP2.** Can coach quality be set to null as an ordinary setting?
2. **BP1.** Is the list of supporting tests derived from `M11.1d`, or declared by hand?
