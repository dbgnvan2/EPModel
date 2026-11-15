# Suggested enhancements from six 2026-10-26 digest papers, in plain language

Plain-language companion to `SWEEP_READING_REPORTS_2026-10-26.md`. That file has the requirement wording, the evidence
and the spec references. None of these is in the spec. Each is a proposal for the owner, held for the next revision like
the earlier batches.

The six papers: Wu et al. 2026 (*AgentMonopoly*), Bai et al. 2026 (*Consistent but Biased: Chair Personas*), Kang et al.
2026 (*What Makes a Social Agent's Interactions Effective*, HICO-Bench), Siahkamari et al. 2026 (*To Manipulate or Not*),
Kwok et al. 2026 (*Beyond Small Talk*, TraDE), Yonezawa 2026 (*Affective Viscosity*).

Most of this batch is inference-validity material — tests and reporting rules — the same kind as the 2026-10-12 batch.
Three candidates (CP3, AV1, AV2) are *mechanism* changes to the anxiety→move and symptom/appraisal paths, so those are
grouped under **Mechanism** and are the ones that most need an owner theory decision. HC4 is for the exploratory
narrator/LLM line only and does not enter the v2 spec.

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children
Sam (15) and Lily (10). They show how a proposal would work. They are not results.

---

## 1. Tests

### AM1. Per-seed reduction — no pseudo-replication *(from AgentMonopoly)*
**What it proposes:** Any criterion that counts within-run events (per-tick moves, per-event appraisals, weekly anxiety
samples) must first reduce them to one number per seed before comparing across seeds. The seed is the sampling unit; a
week's worth of ticks is not ~2,080 independent draws.
**Why:** AgentMonopoly's authors warn that "actions within a game affect one another," so many logged actions do not
replace more independent games. EPModel's runs are ~40 years of weekly ticks; treating each tick as an independent
observation inflates N into the millions and manufactures "significance."
**Example:** A criterion counts how often Tom overfunctions. Ten seeds × 2,000 ticks is not 20,000 independent
observations; it is 10 observations (one per seed), each an average over that seed's trajectory.

### AM2. Balance the intervention's position across seeds *(from AgentMonopoly)*
**What it proposes:** If an arm puts an intervention on a specific person or position (the coach, the focus child, one
side of a dyad), either rotate that position across seeds or report the result separately by position — and never claim
a mechanism-only result from an arm that fixes the position.
**Why:** AgentMonopoly's persona always played at seat P1, and its own authors concede they "cannot rule out an
advantage from turn order." A fixed-position intervention conflates the mechanism with the family role.
**Example:** If the coach always works with the same child, a result "the coach helps" could really be "this child was
already the most responsive." Rotate which child the coach works with across seeds.

### AM3. Pre-declare the intervention before looking at the baseline *(from AgentMonopoly)*
**What it proposes:** An arm's intervention must be written down before the baseline arm's behavior is observed. If the
intervention was chosen to reverse or amplify something seen in the baseline, call it calibrated and do not attribute its
effect to the mechanism alone.
**Why:** AgentMonopoly's personas were "selected after examining baseline behavior" and "written as the opposite of how
its model plays on its own" — so the persona effect cannot be separated from the model. This is fitting by another name.
**Example:** If you notice Ann cuts off under stress and then design the "differentiated Ann" arm to be the opposite of
that, the arm is calibrated, not evidence that differentiation does the work.

### HC1. A timing/order mutant *(from HICO-Bench)*
**What it proposes:** For any mechanism whose effect lands in time (event latency, spell onset, when the learning credit
is applied), the mutation suite must include a mutant that keeps the same magnitudes but shuffles their order across
ticks. A directional criterion whose direction depends on timing must go red on that mutant.
**Why:** HICO-Bench's shuffled-credit control dropped Verified Success (26.25 vs 28.33), proving timing matters beyond
magnitude. A direction should be a property of *when* the mechanism acts, not an artefact of the schedule.
**Example:** If "the coach's early intervention lowers year-40 symptom load," shuffle the intervention to late in the
run; if the direction survives, it was not the timing doing the work.

### HC2. Cross-arm state isolation *(from HICO-Bench)*
**What it proposes:** Each counterfactual arm must be computed as a pure function of its own inputs and seed — no value
from arm B may ever be read by an agent in arm A; arms are compared only through their separate, completed logs.
**Why:** HICO-Bench keeps "alternate-world state and undisclosed partner-private facts… never injected into actor or
reference-policy prompts." If a runner leaks one arm's state into the other, the "difference between arms" is
contaminated.
**Example:** The no-coach arm must not, at any tick, read a value that only exists because the coach arm ran. Two runs
that shared state are not two independent counterfactuals.

### HC3. Divergence-tick reporting *(from HICO-Bench)*
**What it proposes:** Record, per seed, the tick at which two arms' trajectories first diverge beyond a declared
tolerance, and summarize the distribution. A result whose arms diverge only at the final consolidation tick must be
flagged as not attributable to an early-acting mechanism.
**Why:** HICO-Bench locates credit at the divergence turn after disclosure. If a "mechanism" only shows up at the very
end of a run, it is not the mechanism claimed.
**Example:** If the coach arm and no-coach arm run identically for 39 years and only split in year 40, do not credit an
early intervention — something else moved at the end.

### DS1. Endpoint-matched history probe *(from TraDE)*
**What it proposes:** At least one acceptance test must present two arms with an *identical* current event/state that
differ only in the preceding trajectory (a graded build-up into that state vs. a flat history held at that state), and
assert the direction of a downstream difference, with a mutant that deletes the history-carrying mechanism making the
test go red.
**Why:** TraDE matched disclosure depth at the endpoint and still found history effects (119/832 significant at matched
depth). It rules out the artifact that "any matched current state produces identical behavior."
**Example:** Ann faces the same upsetting event in two arms; in one the event is the peak of a long build-up, in the
other it comes out of nowhere. If her reaction differs, history is doing real work.

### MN3. Verify an assigned condition actually materialized *(from To Manipulate or Not)*
**What it proposes:** Any arm that imposes an initial condition or assigned role must include a check that the condition
actually obtained at initialization, and must not assume a fixed numeric starting state for it.
**Why:** The paper's assigned stances produced the right *ordering* but not the expected endpoint values, and the prior
varied by model — "an instruction toward an assigned stance is not an invariant numerical starting state."
**Example:** An arm assigns Sam the "scapegoat" role; the runner must verify the role took effect before a result is
reported, not just assume the assignment is equivalent to the state.

### CP2. Report directional effects by situational-strength stratum *(from Chair Personas)*
**What it proposes:** For any criterion asserting a differentiation- or policy-driven direction, report the effect
separately for ambiguous/mid-scale vs. near-extreme conditions, and expect the effect to be largest in the ambiguous
stratum; flag a flat-across-strata effect as not matching the prediction.
**Why:** The autocratic-chair effect was largest on borderline cases (+44 pp) and smallest on clear ones (+28 pp). A
mechanism that should matter most when the situation is genuinely ambiguous but shows up strongest when it is obvious is
doing something else.
**Example:** Differentiation should matter most when the family faces a genuinely ambiguous choice, not when the answer
is obvious. If the effect is flat, that is a warning.

---

## 2. Reporting rules

### AM4. Multi-channel disposition readout *(from AgentMonopoly)*
**What it proposes:** A dispositional claim about a person or family (generous, overfunctioning, differentiated) must be
asserted over two or more behavior channels; a single-channel claim must say so.
**Why:** AgentMonopoly's Mistral accepted 67% of trades but waived rent only 3% — "generous" in one channel, not in
others. A trait read from one channel is not a trait.
**Example:** "Ann is overfunctioning" must be supported by more than her move count — e.g. her move mix plus her symptom
channel — or the report must say "single-channel."

### CP1. Chance-corrected agreement; bias is not reliability *(from Chair Personas)*
**What it proposes:** When reporting that arms "agree" or a result is "robust" across seeds, report the full outcome
distribution per arm and the agreement expected by chance from each arm's own outcome shares. Never present raw agreement
as reliability; a result that saturates to one side is a property of the rule design (bias), not evidence of reliability.
**Why:** The autocratic persona raised raw across-run agreement from 50% to 85% — but chance expectation was 81% and
Fleiss κ did not rise. The "reliability" was one ruling dominating. A saturating rule looks reliable without being so.
**Example:** If the coach arm produces "withdraw" in 90% of seeds, that is a biased rule, not a robust mechanism — report
the chance level beside the agreement.

### MN1. Two-channel position readout with a divergence flag *(from To Manipulate or Not)*
**What it proposes:** A readout that asserts a position changed must name the channel (enacted move vs. belief-layer
position) and, where both exist, report both and flag their divergence.
**Why:** The paper's direct ratings and conversational state-taking matched exactly only ~33% of turns and "cannot be
treated as interchangeable measurements." Picking whichever channel shows the expected direction is cherry-picking.
**Example:** If Ann's enacted moves say "distance" but her belief layer says "still close to Tom," report both and flag
the split, not just the one that supports the claim.

### MN2. Do not count an actor's own movement as influence on the other *(from To Manipulate or Not)*
**What it proposes:** A readout attributing an effect on B to A's act must measure B's change and separately report A's
own movement, which must not be counted as influence on B.
**Why:** The paper's opponents moved more in absolute terms than discussants, yet that was accommodation, not persuasion
— "a larger shift by the nominal persuader need not mean it has persuaded its counterpart."
**Example:** If Tom overfunctions more while Ann underfunctions, do not read Tom's extra effort as "helping" Ann — his own
rise is accommodation, not influence on her.

### MN4. Per-agent stability is not pairwise convergence *(from To Manipulate or Not)*
**What it proposes:** A convergence/steady-state claim must be about the between-agent quantity (gap, functioning balance,
tie state); late-window per-agent stability must not be read as convergence between agents.
**Why:** ~70–80% of the paper's agents gave identical ratings in late rounds while same-model and cross-model pairs moved
in *opposite* directions — "individual late-turn stability and convergence between participants are distinct properties."
**Example:** Tom and Ann each settle into stable behavior while the distance between them keeps growing. That is not
convergence.

### DS2. Retention-ratio decay readout with a sign-reversal flag *(from TraDE)*
**What it proposes:** For any history-conditioned effect significant at the end of an episode, report a retention ratio
(effect at each later tick ÷ effect at episode end) and flag any sign reversal as a direction change, not retention.
**Why:** TraDE tracked how effects decayed (median 58% after four shallow turns) and excluded sign reversals. Making
persistence/decay a number makes "chronic" falsifiable instead of asserted.
**Example:** After a stress event raises Ann's anxiety, report how much of that rise survives each later year — and if
the effect flips sign, call it a reversal, not persistence.

### DS3. Confound register for history summaries *(from TraDE)*
**What it proposes:** When a result is attributed to a specific history mechanism (accumulated exposure vs. peak
intensity vs. recency/order), name which of these the design cannot separate, and do not attribute the effect to one
when the others are confounded.
**Why:** TraDE could not "determine whether depth, duration, or recency was responsible" even after showing exposure
dominates peak and order.
**Example:** "Chronic anxiety, not acute spikes, drives cutoff" must state whether the design can actually separate
accumulated load from peak intensity and recency.

### DS4. Family/seed as the unit of independence; below-band reporting *(from TraDE)*
**What it proposes:** Cluster and bootstrap all ensemble statistics at the level of the sampled unit (the family/seed),
not the individual agent; do not count a seed as passing when the effect magnitude is smaller than that seed's bootstrap
band.
**Why:** TraDE's small samples showed trajectory variation below the bootstrap CI width for most behaviors and reported
it. Aggregating across ~12 agents treats them as independent replicates when they are one interacting family.
**Example:** The 12 people in one family are not 12 independent samples. If a per-seed effect is smaller than its own
uncertainty band, that seed is not a pass.

---

## 3. Mechanism (needs an owner theory decision)

### CP3. Avoidance resolves to deferral or buck-passing, not inaction *(from Chair Personas)*
**What it proposes:** Anxiety-driven avoidance must not default to inaction. Under threat or decisional conflict,
avoidance resolves to deferral (DISTANCE plus reassurance-seeking/checking) or buck-passing (TRIANGLE — moving the
decision onto a third party), and buck-passing is gated on accountability salience plus an available third party.
**Why:** The avoidant chair deferred 95% of the time and chose inaction only 3%, and buck-passing appeared only when the
rule named it. Bowen says anxiety produces distance and triangles, not paralysis.
**Example:** When Ann is anxious about Sam's behavior, she should withdraw or pull a grandparent in — not simply freeze.
This is a theory claim, so it needs an owner decision on whether Bowen really rules out paralysis.

### AV1. Expression gated by time-integrated load, not instantaneous anxiety *(from Affective Viscosity)*
**What it proposes:** A symptom onset or escalation move should be gated by time-integrated unresolved load (the M4.C.3a
integrator — excess acute anxiety above the person's usual level), not by instantaneous acute anxiety alone, so a single
spike cannot fire an expression the accumulated state does not sustain.
**Why:** Affective Viscosity argues immediate expression is "psychologically unnatural"; a delayed escalation reads more
authentic than instant boundary-setting. Symptoms should be chronic binding, not an instant mirror of acute anxiety.
**Example:** One bad week should not flip Sam's symptom channel; only a sustained excess above his usual level should.
This strengthens the "visible emotion is not the measure of anxiety" trap (M1.A.0).

### AV2. Appraisal sensitizes to repeated same-source events *(from Affective Viscosity)*
**What it proposes:** Appraisal of an event should be modulated by accumulated unresolved affect from prior events with
the same sender and move type within a declared window, so a repeated identical provocation escalates the response until
discharge rather than producing an identical response each time.
**Why:** Affective Viscosity: agents "respond differently to the same event after repeated encounters," with a discomfort
index that accumulates with a decay factor.
**Example:** The fifth time Tom makes the same dismissive remark, Ann should react more strongly than the first — not
identically — because the accumulated affect has not discharged. The window and constant are invented and labelled.

---

## 4. Narrator line (does not enter the v2 spec)

### HC4. Inter-judge agreement for the LLM line *(from HICO-Bench)*
**What it proposes:** In the exploratory LLM line, any evaluation using an LLM judge must report cross-judge agreement
(Cohen's κ) against a second judge on a fixed stratified subset, should prefer a rule-based/programmatic check wherever
the target is checkable, and must record that LLM-judge agreement is not evidence of correctness.
**Why:** HICO-Bench's human annotators agreed 100% with the rule-based feasibility check but only 82% with the LLM-judged
integrity check, and two judges agreed at κ = 0.783. Rule-based checks beat LLM judges — which is itself the M3.D.6
rationale.
**Example:** When the LLM line evaluates whether a generated agent "acts mature," a second judge and a κ number are
required, and the result is not proof of correctness — same treatment as the X-series protocol items.
