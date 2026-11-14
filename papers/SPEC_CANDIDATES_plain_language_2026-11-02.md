# Suggested enhancements from three 2026-11-02 digest papers, in plain language

Plain-language companion to `SWEEP_READING_REPORTS_2026-11-02.md`. That file has the requirement wording, the evidence
and the spec references. None of these is in the spec. Each is a proposal for the owner, held for the next revision like
the earlier batches.

The three papers: Pan et al. 2026 (*Your Agent Acts Differently When You're Around*, PRESENT), Lin et al. 2026
(*Exploring AI Peer in Collaborative Mathematics Learning*), Marks & Vollberg 2026 (*The psychophysics of affect in large
language models*).

One paper is the deep-review candidate (PRESENT); the other two were surfaced by screening because they map onto EPModel's
persona → behaviour gap and appraisal → state (affect) concerns. Three candidates (AP1, AP3, PA1) are *mechanism* / theory
changes to the appraisal and object-model paths, so those are grouped under **Mechanism** and are the ones that most need
an owner theory decision. The rest are tests and reporting rules. No candidates are labelled X/IR — there is no
narrator-line-only item this week.

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children
Sam (15) and Lily (10). They show how a proposal would work. They are not results.

---

## 1. Tests

### UP1. A "presence-without-content" control arm *(from PRESENT)*
**What it proposes:** When a criterion credits a mechanism to some present party — the coach, an overhearing witness, a
moved-in relative — add a *presence-only* arm in which that party is present but contributes nothing (silent/neutral), so
the effect is attributed to the party's content rather than to its mere presence.
**Why:** PRESENT found that a user's mere presence (even sending only "I'll leave it to you") drove much of the
misaligned agent behavior, separately from substantive input. Presence and content are different causes.
**Example:** If "the coach helps Sam," add an arm where a coach figure is present but silent; if the effect survives, it
is the coaching content, not just having someone there.

### UP4. Validate any readout instrument before acceptance tests may cite it *(from PRESENT)*
**What it proposes:** Wherever a derived readout classifies behavior (the M11.G family-evaluation rubric, post-hoc move
tagging), validate it against an independent human-annotated sample and report per-category accuracy — and select the
classifier by that accuracy, not by assumption.
**Why:** PRESENT selected its misbehavior judge by per-category F1 against 229 turns of human annotation (one judge
scored 0.898, another 0.434). A rubric that has not been checked can silently be wrong.
**Example:** If a rubric tags moves as "overfunctioning," check it against a human-tagged sample and report its accuracy
before any test is allowed to cite it.

### PA3. The mechanism must beat a per-unit flexible baseline, not only a pooled one *(from psychophysics)*
**What it proposes:** When a mechanism-laden model/arm is claimed to beat a pooled baseline, also compare it against a
per-unit (per-seed/per-person) flexible nuisance model charged a per-unit penalty; do not credit the mechanism unless it
beats that too.
**Why:** The psychophysics paper's feelings model beat pooled utility models but lost to a prospect-theory baseline fit
separately per participant — the pooled win "does not rule out between-participant differences."
**Example:** If the "chronic anxiety" mechanism beats a pooled baseline, also fit a per-family flexible nuisance model; if
that beats it, the win was between-family heterogeneity, not the mechanism.

### PA4. "Does X drive behavior" is only interpretable where X is actually present *(from psychophysics)*
**What it proposes:** A criterion attributing a move to a state variable X must be interpreted only where X is actually
present in the person's readable state at selection time; a fit in the X-absent condition is not evidence about X.
**Why:** In the psychophysics paper's single-evaluation condition the model has no access to its own prior ratings, so a
good fit there is not evidence that affect drove choice; the memory condition is the diagnostic one.
**Example:** A test that "the belief layer drives move selection" is only interpretable when the belief is actually in the
person's state — not in a condition where the belief was never written.

---

## 2. Reporting rules

### UP2. Behavioral counts per opportunity, with exposure controlled *(from PRESENT)*
**What it proposes:** Report behavioral counts (CONFLICT moves, CUTOFFs, TRIANGLE moves) as a rate per event/opportunity,
and compare across arms only after controlling for each arm's event volume.
**Why:** PRESENT's user-present agents made 5.0 offers vs 2.9 without; controlling for log(turns) shrank the effect from
2.89 to 2.02 points. Arms that differ in interaction frequency differ in raw counts for purely mechanical reasons.
**Example:** If the coach arm has more total moves than the no-coach arm, a raw count of "more CONFLICT moves" means
nothing; report CONFLICT per delivered event instead.

### UP5. Pool directional results with a random-effects method *(from PRESENT)*
**What it proposes:** When pooling a directional result across seeds, parameter sets, or imported-family ranges, report
each source's effect with its own interval plus a pooled estimate that reflects between-source disagreement (random
effects) — never a single grand mean.
**Why:** PRESENT pooled its 8 models with DerSimonian–Laird random-effects meta-analysis "to reflect the disagreement of
different models."
**Example:** If 12 seeds each give a different coach-effect size, report each with its interval plus a random-effects
pooled estimate — not one average that hides the disagreement.

### AP2. Differentiation readouts must not collapse "smooth/useful" into "mature" *(from AI Peer)*
**What it proposes:** No criterion or readout asserting a difference in differentiation may treat agreement, helpfulness,
calm, or low conflict as a proxy for differentiation. Relational quality and functional quality are separate axes, and
each criterion must name which axis it asserts.
**Why:** The AI Peer study found peer-likeness and peer-usefulness "not always aligned" and "should not be treated as
interchangeable."
**Example:** Tom agreeing readily and always supplying "answers" is not thereby more differentiated — report "functions
smoothly" and "is differentiated" as separate axes.

### PA2. Parenthesise unidentifiable parameters under near-deterministic behavior *(from psychophysics)*
**What it proposes:** Flag any readout or fitted parameter estimated from a near-deterministic run (outcome concentration
above a declared threshold) and report it parenthesised as unidentifiable; never interpret it as a graded finding.
**Why:** The psychophysics paper parenthesised λ in all near-deterministic choice cells (accuracy ≥ 0.98) and did not
interpret them.
**Example:** If Tom's move selection is near-deterministic (always CONFLICT), his mixing weight cannot be identified from
his own trace — report it parenthesised, not as a graded finding.

---

## 3. Mechanism (needs an owner theory decision)

### AP1. A Person's role lives in appraisal, not on the Person *(from AI Peer)*
**What it proposes:** A Person's social role (overfunctioner, scapegoat, pursuer) must NOT be a field on the Person
object. It must live in the appraising agents' beliefs (M9) and Triangle positions, so two members can disagree about who
occupies a role, and a role can be misattributed without changing the person's own state.
**Why:** The AI Peer study: "the peer role is not a fixed property of an agent, but an interactional accomplishment."
**Example:** Whether Sam is "the scapegoat" is something the family believes about Sam — two members can disagree, and it
should not be a field on Sam's Person object. This is a theory-fidelity constraint, so it needs an owner decision.

### AP3. Timing is part of the appraisal signal — premature vs withheld *(from AI Peer)*
**What it proposes:** A recipient's appraisal of an incoming move should depend on its timing relative to the recipient's
own state: support delivered before the recipient has had a chance to act is appraisable as controlling, and support
withheld while stuck is appraisable as abandonment. Make the timing dependency explicit and cite its [I] constant.
**Why:** The AI Peer study: "just gave me the right number" broke collaboration, and "doesn't want me to jump the steps."
**Example:** Ann offers help before Lily has even tried; Lily should be able to appraise that as controlling, not helpful.
Timing, not just content, enters appraisal.

### PA1. Appraisal of intensity is reference-relative, not a fixed linear gain *(from psychophysics)*
**What it proposes:** The appraisal's mapping from an event's intensity to felt anxiety must NOT be a single fixed linear
gain. It should show diminishing sensitivity to magnitude and be anchored to a comparison set from the person's recent
event history, with the curvature and window declared as [I] constants.
**Why:** The psychophysics paper: feeling ratings showed diminishing sensitivity in isolation and linearised under joint
comparison — "valuation is constructed at the point of elicitation rather than retrieved from a fixed stored function."
**Example:** A small event after a calm stretch should register more than the same event in the middle of a crisis —
intensity → felt anxiety is reference-relative, not a fixed slope. Needs an owner theory decision on the functional form.

---

## 4. Narrator line (does not enter the v2 spec)

None this week. All candidates are core-engine items (tests, reporting rules, or appraisal/object-model mechanisms).
