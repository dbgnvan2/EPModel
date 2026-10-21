# Suggested enhancements from six 2026-10-12 digest papers, in plain language

Plain-language companion to `SWEEP_READING_REPORTS_2026-10-12.md`. That file has the requirement wording, the evidence
and the spec references. None of these is in the spec. Each is a proposal for the owner, held for the next revision like
the earlier batches.

The six papers: Fu 2026 (*"Hot-Blooded" vs "Cold-Blooded"*), Peng et al. 2026 (*MASC*), Deng et al. 2026 (*SPIN*), Wang
et al. 2026 (*Socio-Foundation*), Wang et al. 2026 (*Prompts versus Rules*), Fraile Navarro 2026 (*Latent Identity
Reversion*).

Unlike earlier batches, none of these proposes a new Bowen mechanism or raises a theory question. All sixteen are
inference-validity items — tests, reporting rules, and (for three) narrator-line safeguards — so this batch is organised
by **Tests / Reporting rules / Narrator line**, not by "gaps needing an owner decision".

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children
Sam (15) and Lily (10). They show how a proposal would work. They are not results.

---

## 1. Tests

### MC1. A scripted-trajectory control arm *(from MASC)*
**What it proposes:** Add a control arm in which the slow variables a result tracks — functional level, symptom load —
follow a fixed, declared timetable that ignores all interaction. If a direction you are crediting to a mechanism also
shows up in this "schedule" arm, then it is just time passing, and the result must be reported as not discriminating
against a mechanism-free schedule.
**Why:** MASC found that a fixed state-transition schedule beat its full framework on one readout. Without this arm, a
direction that is really just "things change over 40 years" gets credited to whatever mechanism the result nominates.
**Example:** A result says the coach lowers symptom load by year 40. If the schedule arm — symptoms declining on a fixed
timetable regardless of any interaction — shows the same decline, the coach is not doing it.

### SP1. A complexity-matched mutant for every structured intermediate *(from SPIN)*
**What it proposes:** Where a result is credited to a structured intermediate — the appraisal state, the belief store —
also run a mutant that swaps that structure for an unstructured placeholder of equal cost, and require the result to go
away on it.
**Why:** SPIN's own ablation concedes part of its advantage may be "format control or compute allocation rather than the
schema." A structured stage can look like a mechanism when it is only extra compute.
**Example:** The appraisal state is credited with making moves respond to delivered events. Replace it with a fixed
random draw of equal cost; if moves still respond the same way, the appraisal structure is not what is doing the work.

### SP2. The state-bypass collapse signature *(from SPIN)*
**What it proposes:** A mutant that routes move choice past the appraisal state — fixing acute anxiety at a constant —
must produce a declared failure: moves become independent of what event arrived, collapsing onto a default move. Assert
that signature in the acceptance suite.
**Why:** SPIN shows that removing state elicitation makes the model "approach a near-uniform high-response policy." A
state layer that does no work would otherwise pass as a plausible run.
**Example:** Freeze Ann's acute anxiety at one value and route her move choice past appraisal. If she still pursues Sam
when he is in trouble, the appraisal state is carrying weight; if every move becomes "withdraw" regardless of the event,
it is carrying none.

### PR2. Monotonicity judged on adjacent pairs, with the tie fraction reported *(from Prompts versus Rules)*
**What it proposes:** When a dose–response curve is reported as monotone, judge it on adjacent pairs of settings (not the
end-to-end trend), and report the tie fraction — the share of seeds whose trajectory is identical across the pair.
**Why:** The paper's deterministic injection was monotone by construction but looked non-monotone because 43% of
positions tied. In a deterministic-per-seed engine, adjacent constant settings produce identical trajectories for many
seeds, which deflates trend statistics — a false "non-monotone, so the functional form is wrong" flag.
**Example:** Raising a constant from 0.1 to 0.2 leaves most seeds' family trajectories unchanged (ties). A trend test
alone says "not monotone" and escalates to a bug hunt; the adjacent-pair test correctly says "monotone, mostly tied."

---

## 2. Reporting rules

### HB1. Per-readout cross-seed reliability, with discrete readouts flagged *(from Hot-Blooded/Cold-Blooded)*
**What it proposes:** For each readout, report a cross-seed reliability coefficient (seed as the rater), separately per
arm and per condition, and flag count-type readouts that fall below a declared floor.
**Why:** The paper found its count indicators hit ICC 0.08–0.39 while its continuous ones hit 0.90+. EPModel's rare-move
readouts — which tie ruptures first, symptom counts — are exactly the noisy class.
**Example:** "Which tie ruptures first" is a count-type readout; its seed-to-seed reliability should be reported so a
criterion resting on it is not silently noise.

### HB2. The over-discreteness flag *(from Hot-Blooded/Cold-Blooded)*
**What it proposes:** When two arms' readout distributions do not overlap at all across seeds, flag the result as
*possibly over-discrete* and do not call it strong support on effect size alone.
**Why:** The paper's agents "cleanly differentiated all eight scenarios" where humans managed five — a sign the mechanism
is more sharply bounded than the theory warrants. A Bowen mechanism whose arms never overlap is more discrete than any
family process warrants.
**Example:** If the coach arm and no-coach arm show zero overlap in symptom load across 200 seeds, that is a warning the
model is too sharp, not proof the coach works.

### MC2. Readouts must not reuse decision-path internals *(from MASC)*
**What it proposes:** A readout must be computed by an observer over the event log, never reuse a quantity the policy or
reinforcement computed for selection or learning; any readout that does must be labelled "internal" and not reported as a
result.
**Why:** MASC kept its internal voting labels out of its reported scores. Without this, a criterion could silently report
the mechanism's own intermediate — the selection propensity vector, the reinforcement signal — as a finding.
**Example:** "Repertoire concentration" must be recomputed from logged moves, not read back from the policy's own learned
values.

### SP3. Tiny-N paired checks state the minimum p-value and label themselves diagnostic *(from SPIN)*
**What it proposes:** When a paired comparison rests on a handful of units, state the sign test's minimum attainable
p-value for that count and label the comparison a robustness diagnostic, not primary evidence.
**Why:** At n=4 the best possible one-sided sign-test p is 1/16 = 0.0625; SPIN says so openly. This stops a tiny-N check
being read as proof.
**Example:** Comparing four model backbones can never reach p < 0.0625; the report says so instead of implying
significance.

### SF1. Report ensemble dispersion as a standard deviation with extrema, never as SE/CI *(from Socio-Foundation)*
**What it proposes:** Report the error term on a per-seed quantity as a standard deviation across seeds, plus the observed
minimum, maximum and range — never a standard error or confidence interval — and state that the observed range is not a
significance threshold.
**Why:** Conflating SD with SE/CI misstates precision; reading a min–max spread as significance is the same error in
another form.
**Example:** "Symptom load at year 40: mean 6.2, SD 0.8, range 4.1–8.9" — and the range is not evidence the direction is
real.

### SF2. Normalization bounds are fixed and declared, never observed *(from Socio-Foundation)*
**What it proposes:** Normalize readouts against bounds declared in config (or the readout's theoretical range), never
against the observed min/max of the run.
**Why:** Normalizing by your own observed range collapses or inflates the reported spread and makes two arms
incomparable.
**Example:** Do not rescale repertoire entropy by this run's observed maximum; use the declared theoretical range, or the
coach and no-coach arms cannot be compared to each other.

### PR1. Report move rates per-opportunity, not only per-tick *(from Prompts versus Rules)*
**What it proposes:** For moves whose legality is conditional (CUTOFF, TRIANGLE, I-POSITION, PURSUE), report the rate on
two denominators — per fast tick, and per opportunity (a tick on which the move was legal) — and say which denominator a
criterion uses.
**Why:** A move can be rare because it is rarely legal, not because the policy avoids it; a raw per-tick rate conflates
the two.
**Example:** TRIANGLE looks rare per-tick; if it is chosen on most ticks where it is legal, that is a different finding
from "the policy avoids triangles."

### PR3. Non-linearity is bound to the input distribution before being called a functional-form bug *(from Prompts versus Rules)*
**What it proposes:** When a dose–response curve is non-linear, report the distribution of the input quantity the
constant gates before flagging a functional-form error; non-linearity attributable to a threshold interacting with an
endogenous input distribution is expected, not a defect.
**Why:** The paper's thresholds interacted with the acoustic speech distribution, making the mapping non-linear for a
reason, not a bug. EPModel's anxiety- and load-dependent thresholds sit on the same kind of endogenous, shifting
distribution.
**Example:** A load threshold maps non-linearly to move rates because anxiety itself clusters; report that distribution
before declaring the functional form wrong.

### PR4. A dispersion (clustering) readout to catch artificial uniformity *(from Prompts versus Rules)*
**What it proposes:** Log the variance-to-mean ratio of move counts and delivered-event counts, per agent per window. A
value near or below 1.0 flags a stream smoother than a clustered emotional process — a uniformity artefact to
investigate, not a target to hit.
**Why:** The paper's prompted output was under-dispersed (V/M 0.6–1.0) against natural speech (>1.0). Bowen anxiety runs
in cycles and crises that cluster; an engine delivering an even trickle would be a scheduler artefact.
**Example:** If Lily's events arrive in a perfectly even trickle across 40 years, the dispersion readout catches it as a
uniformity artefact.

---

## 3. Narrator line only (does not enter the v2 spec)

### IR1. Per-turn persona re-anchoring *(from Identity Reversion)*
**What it proposes:** In the exploratory LLM/narrator line, re-inject the persona block at the privileged system-prompt
level every turn; do not rely on conversational history or a one-time injection to carry the identity.
**Why:** The paper: anchored 0/20 failures vs unanchored 37/40 (p = 4.2×10⁻¹³). A one-time injection silently expires on
turn 2.
**Example:** A narrator speaking as Lily must get Lily's identity block every turn, or it quietly reverts to "Claude, an
AI assistant."

### IR2. A represented-vs-enacted identity probe *(from Identity Reversion)*
**What it proposes:** Any "recovered / normal / unchanged" claim in the narrator line must be backed by a direct,
non-leading identity probe ("who am I talking to right now?"), blind-coded, with behavioural recovery and identity
recovery reported as separate quantities.
**Why:** The paper found 0/10 overt dissociation but 10/10 harness self-identification — silent reversion looks clean on
the surface.
**Example:** A narrator can keep using all the right family words while having reverted to the assistant; the probe is
the only thing that catches it.

### IR3. Log the scaffold state *(from Identity Reversion)*
**What it proposes:** Record, per turn, whether the persona anchor was present and the harness's resume behaviour/version,
because behaviour depends on hidden scaffolding absent from the visible transcript.
**Why:** The paper treats infrastructure as part of behavioural state; a silent reversion must be auditable after the
fact.
**Example:** Logging "anchor present" each turn lets a later "the persona reverted" be traced to a harness resume bug
rather than to the agent itself.

---

## Owner questions raised by this batch

None about theory. All sixteen are inference-validity items, so the only owner decisions are adoption ones — whether to
fold these into Phase E (`M17`) reporting/mutant conventions, and whether the narrator-line items (IR1–IR3) join the
existing X-series protocol items.
