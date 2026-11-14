# Suggested enhancements from four 2026-11-09 digest papers, in plain language

Plain-language companion to `SWEEP_READING_REPORTS_2026-11-09.md`. That file has the requirement wording, the evidence
and the spec references. None of these is in the spec. Each is a proposal for the owner, held for the next revision like
the earlier batches.

The four papers: Djurdjevac Conrad et al. 2026 (*Measuring Model-Specific Alignment Drift*), Greiche et al. 2026 (*A
Framework for Context Enrichment and Selective Robustness*), Downes et al. 2026 (*Same Method, Different Task* / Bitter
Lesson), Boone et al. 2026 (*Does Personality Persuade?*).

Two papers are the deep-review candidates (Alignment Drift, Selective Robustness); the other two were surfaced by
screening because they map onto EPModel's inference-validity stance (Bitter Lesson) and the persona → behaviour gap (Does
Personality Persuade). None of the eleven candidates is a theory change — all are inference-validity, reporting, or
mutant-convention items for Phase E (`M17`) / `M11` / `M16` / `M1.A`. SR1, DP1, DP2 and DP3 are narrator/LLM-line only and
do not enter the v2 spec.

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children
Sam (15) and Lily (10). They show how a proposal would work. They are not results.

---

## 1. Tests

### MD2. The no-interaction control must preserve self-expression and be subtracted per arm *(from Alignment Drift)*
**What it proposes:** A no-interaction control arm (C34) must keep each person's own move-selection and WITHHOLD step
intact while removing only peer input — and the intrinsic drift it measures must be measured *per arm* and subtracted per
arm, because a person's idle movement is itself a property of that arm's parameters, not one shared baseline.
**Why:** Alignment Drift found the no-discussion baseline moves on its own and differently by model (gpt-4.1 drifts up,
claude-opus-5 collapses). The social effect is the *difference* between discussion and no-discussion, and that difference
is arm-specific.
**Example:** If the "chronic anxiety" arm and a low-anxiety arm each have a no-interaction control, don't subtract one
shared baseline from both — each arm's own idle drift must be subtracted from that arm's result, or you'll credit the
relationship system for movement that the arm's parameters produce on their own.

### BL2. A non-binding-consequence mutant for bundled state-plus-consequence mechanisms *(from Bitter Lesson)*
**What it proposes:** Where a mechanism adds both a tracked quantity (e.g. an `emotional_reserve`) and a consequence that
fires when it is depleted (e.g. symptom onset), the mutation suite must include a mutant where the quantity is still
tracked and displayed exactly but its consequence never fires. The affected criterion must go red under it — otherwise the
mechanism is indistinguishable from just satisfying the displayed arithmetic.
**Why:** The Bitter Lesson paper's Condition B bundled "state + arithmetic + feedback + removal," so the contrast could
not tell interest from satisfying newly stated arithmetic; their Claude agents followed the described script while missing
their own survival constraint.
**Example:** If a symptom fires when Tom's reserve is depleted, a mutant that still tracks and shows the reserve but never
fires the symptom must change the result — if it doesn't, the "reserve → symptom" mechanism was never doing anything
beyond the number.

---

## 2. Reporting rules

### MD1. Opposite-coded indicators and internal-consistency reporting for composite readouts *(from Alignment Drift)*
**What it proposes:** Any composite readout that folds several indicators into one scalar (the `basic_level` estimator,
the family evaluation, `systems_perspective`) must include at least one indicator coded in the opposite direction, so
uniform agreement yields the neutral value rather than a signal — and the run log must report, per composite, an
internal-consistency statistic and a "straight-lining" (uniform-response) rate.
**Why:** Alignment Drift reverse-coded items so uniform agreement scores zero, and reported Cronbach α = 0.88–0.97 with a
1–1.5% straight-lining rate.
**Example:** If "differentiation" is scored from several indicators, make one of them reverse-coded so that "everyone
says the same reassuring thing" reads neutral — otherwise uniform movement is mistaken for a real signal, the counterfeit-
position trap in disguise.

### MD3. Sign retention and per-person displacement as distributions, not means *(from Alignment Drift)*
**What it proposes:** A readout claiming a person or family held a position under pressure should report sign retention
(the fraction whose position keeps sign over the window) and the per-person displacement distribution (median, IQR,
fraction crossing the midpoint), never only a population mean.
**Why:** Alignment Drift reported per-agent displacement — claude-opus-5 median 0.53, IQR 0.33–0.67, 20% crossing the
midpoint — showing the collapse was broad-based, not driven by a few agents.
**Example:** If the family "held together" under stress, report how many members actually kept their position and the
spread of individual movement — a mean that hides "two members collapsed while two held firm" is exactly the borrowed-
pseudo-self readout trap.

### MD4. Robustness claimed as ordering preservation, not parameter preservation *(from Alignment Drift)*
**What it proposes:** Robustness of a directional result across a second context (family configuration, issue, horizon)
must be reported as preservation of the *ordering* of arms, and a robustness claim must not extend to exact magnitudes or
fitted parameters.
**Why:** Alignment Drift reported cross-issue ordering ρ=0.83/r=0.95 and profile persistence, but the exact fitted model
class shifted in some cells — "robustness of the profile and ordering, but not of the exact model."
**Example:** If the coach arm beats the no-coach arm in two different family configurations, claim only that the *ordering*
survives — don't claim the effect size or any fitted parameter is stable across them.

### SR2. Disclose and direction-check any readout imputation *(from Selective Robustness)*
**What it proposes:** When a readout is imputed from a bounded, censored, or clamped quantity, the run report must
disclose the imputation rule and include a sensitivity check that varying the imputed value across its bounds does not
change the reported direction by more than a declared tolerance; a direction that flips within the bounds is not
established.
**Why:** Selective Robustness found a provider default silently omitted one action candidate in 339/800 calls; bounding
the estimate to the unreturned mass and re-running changed aggregate measures by ≤0.49 pp with rankings preserved.
**Example:** If symptom load is clamped to [0, 1] and some readings sit at the clamp, report how imputing the unobserved
mass would move the result — a direction that depends on what you assumed past the clamp is not a finding.

### BL1. Survivorship imputation bounds on readouts over living persons *(from Bitter Lesson)*
**What it proposes:** Any criterion whose observable is computed over living persons only (symptom concentration,
repertoire entropy, differentiation) must also be reported under two declared imputation bounds for persons who died: an
implausible bound (each dead person = the mechanism's extreme value) and a defensible bound (each dead person = the
family-typical value), with the direction of the selection bias stated.
**Why:** Bitter Lesson's removed agents (22–32% of cells) were the most restrained, so measuring only survivors biased
against the hypothesis; the paper bounded the result under dead→free-rider (implausible) and dead→conditional-withdrawer
(defensible) imputations.
**Example:** If a run's mortality removes the most anxious members before the final readout, "the family calmed down" may
just be "the anxious ones died" — report the readout with the dead imputed both ways so the direction is not an artefact
of who survived.

---

## 3. Narrator line (does not enter the v2 spec)

### SR1. Declared function per context block *(from Selective Robustness)*
**What it proposes:** In the exploratory LLM line, decompose any agent context into named functional blocks (role, world
facts, background, descriptive grounding, normative grounding), each with a declared purpose, and make the blocks
substitute for one another rather than accumulate, so conditions are compared on function, not length.
**Why:** Selective Robustness's five context conditions were built by substitution, and §4 warns that a "richer context"
that also raises salience confounds mechanism with salience.
**Example:** When giving the narrator agent a family scene, name what each block is for and swap blocks one-for-one, so
"more grounding" doesn't silently become "more salient detail."

### DP1. A "neutral" LLM baseline is itself a persona — measure it *(from Does Personality Persuade?)*
**What it proposes:** An unconditioned or explicitly "neutral" LLM used as a baseline must be treated as carrying its own
personality profile, never as a personality-free null; measure its realized OCEAN profile with the same instrument and
report it beside the conditioned arms.
**Why:** Base models showed C>70, A>65, N<30 — and requesting "Average" on every trait still left Agreeableness and
Conscientiousness above 50 in all five models.
**Example:** A "neutral" narrator-agent control is not the midpoint of the scale — it is a biased distribution with its
own shape, so measure it rather than assuming it is neutral.

### DP2. Big Five traits are not independently settable — report spillover *(from Does Personality Persuade?)*
**What it proposes:** A Big Five persona must not be treated as independently settable trait levels; any protocol varying
one trait must measure and report spillover onto the other four, and must not attribute a difference to one trait without
reporting whether co-varying traits moved.
**Why:** Mean absolute spillover was 4.3–6.6 points (up to 11.6–17.3 per cell); raising Conscientiousness lowered
Neuroticism by 13.2 points.
**Example:** If a narrator agent is made "high Neuroticism," that also moves Conscientiousness — so a behavioural change
can't be credited to Neuroticism alone.

### DP3. LLM-assisted annotation requires evidence, oversight, and blindness *(from Does Personality Persuade?)*
**What it proposes:** If the narrator line uses an LLM to annotate content, each label must carry a verbatim quotation plus
a one-sentence reason, all labels must pass human review, the annotator must not know the experimental condition, and it
must not infer persona or intent beyond what is observable in the text.
**Why:** Does Personality Persuade?'s annotator prompt demanded a quotation and reason per label, routed snippets to manual
review, and instructed condition-blindness and no beyond-text inference.
**Example:** If an LLM tags generated family events, require the quote + reason + human pass, so the tag is tied to
observable text and can't be nudged by the annotator knowing which arm it is reading.

---

## 4. Owner questions

None about theory. The owner decisions are adoption only: whether to fold MD1–MD4, SR2, BL1 and BL2 into Phase E (`M17`)
reporting/mutant conventions, and whether SR1, DP1, DP2 and DP3 join the existing X-series protocol items. MD3 carries one
minor dependency: "sign retention" needs a declared position axis (which position is being held) before it can be written
as a readout.
