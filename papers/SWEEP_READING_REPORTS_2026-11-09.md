# Reading reports: four papers, full reads (2026-11-09 digest)

Produced 2026-11-09 at the owner's request, from the 2026-11-09 method-literature digest. Two papers are the deep-review
candidates the sweep flagged in `DIGEST.md`/`INDEX.md` (individual/small-group ABM) — *Measuring Model-Specific Alignment
Drift in LLM-Based Opinion Dynamics* (2611.03980) and *A Framework for Context Enrichment and Selective Robustness in LLM
Social Agents* (2611.06914). Two more were surfaced by screening because they map directly onto EPModel core concerns:
*Same Method, Different Task: What the Bitter Lesson Does Not License in Social Simulation* (2611.05717, the "stakes, not
emergence" argument) and *Does Personality Persuade?* (2611.06572, persona control vs observable influence). All four were
read in full from `pdftotext -layout` output by four readers working under
`sweep_readers_brief_2026-09-19/READER_TASK.md`. Each reader checked for overlap by keyword search in the current spec
(`docs/bowen_agent_model_spec_v2.md`), `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, the earlier
`SPEC_CANDIDATES_plain_language_*.md` files, and `docs/todo/2026-09-27_paper_material.md`'s held-for-later-revision
section. Reports are reproduced as written; only heading levels were changed. SR1, DP1, DP2 and DP3 are narrator/LLM-line
only and do NOT enter the v2 spec.

**Status: none of these candidates is in the spec.** A plain-language explanation with examples is in
`SPEC_CANDIDATES_plain_language_2026-11-09.md`.

| Part | Paper | Candidates |
|---|---|---|
| 1 | Djurdjevac Conrad et al. 2026, *Measuring Model-Specific Alignment Drift in LLM-Based Opinion Dynamics* (2611.03980v1) | MD1, MD2, MD3, MD4 |
| 2 | Greiche et al. 2026, *A Framework for Context Enrichment and Selective Robustness in LLM Social Agents* (2611.06914v1) | SR1, SR2 |
| 3 | Downes et al. 2026, *Same Method, Different Task: What the Bitter Lesson Does Not License in Social Simulation* (2611.05717v1) | BL1, BL2 |
| 4 | Boone et al. 2026, *Does Personality Persuade? Investigating Personality Trait Prompting and Persuasion in Large Language Models* (2611.06572v1) | DP1, DP2, DP3 |

---

# Part 1: Measuring Model-Specific Alignment Drift in LLM-Based Opinion Dynamics (2611.03980v1)

## A. Report

**1. Simulation formalism — synchronous rounds, measurement sealed from theory.**
- [PAPER] §2: each of T rounds has "a synchronous compose phase" (every agent writes a statement conditioned on the previous round) and a survey phase (every agent re-answers the instrument). n=10, T=20 (R=5 replicates, some R=3), six LLMs.
- [PAPER] §2: measurement is "kept separate from theory" — no opinion-dynamics model enters the measurement path, exposure is "never gated by opinion distance," and "the storage layer refuses non-LLM data to the fitting code."
- [INFERENCE] This is a synchronous scheme (the exact regime DESIGN_LESSONS §2.1 flags); it works here only because the measurement is a fixed instrument and the fit is applied after the fact. Nothing new for EPModel's M3, but the measurement/seal design (§7 below) transfers.

**2. Agent architecture — one-sentence personas, no built-in anchor.**
- [PAPER] §2, App. C: fixed one-sentence persona plus initial stance xᵢ(0)∈[−1,1]; "no opinion anchor is built into the agents"; own recent opinions are resubmitted only under a framing condition (anchored-strict/soft/persuadable).
- [PAPER] §3, Table 3: fitted Friedkin–Johnsen anchoring weights λ≈0.87–0.94 quantify how strongly an agent "retains for its own position" — but this is a fitted readout, not a built-in parameter.

**3. Interaction/network structure — three exposure conditions, emergent homophily.**
- [PAPER] §2: plenary (all hear all), selective attention (each agent names, "from prose alone, the peers it takes seriously" and hears only that subset), and group identity (labels tied to initial stance).
- [PAPER] §3, Table 1: homophily index = mean opinion distance to dropped vs retained peers. Only gpt-4.1 (+0.485±0.032) and claude-sonnet-5 (+0.327±0.024) show stance-homophilous attention; gpt-5.6-sol is the opposite (−0.039±0.008). App. E: from a neutral start the homophily signal does not emerge (+0.002) — selective attention "maintains and structures pre-existing disagreement but does not create it."
- [INFERENCE] Attention is driven by the agent's own naming (a behaviour), never by a computed opinion distance — the distance is only read out afterward. That is a design choice EPModel could mirror (§B MD-seal note).

**4. Initialization — stances spaced, then held or collapsed.**
- [PAPER] App. C: firm-but-uncommitted roster has ten personas with opinions "evenly spaced on [−0.8, +0.8]" (initial dispersion 0.47); committed roster seeds interior stances |xᵢ(0)|≤0.7; neutral roster seeds all at xᵢ(0)=0.
- [PAPER] §3, Table 4: with committed personas and no peers, "all five LLMs keep every agent on its assigned side (S=1.00)" — the collapse under discussion is interaction-induced, not an initial-condition effect.

**5. Calibration and validation — internal model selection, no human ground truth.**
- [PAPER] §3, Table 3, App. D: fits a battery (persistence, DeGroot, Friedkin–Johnsen, bounded-confidence, common-drift attractor, pole-directed polarize) by one-step least squares with BIC selection; "the battery recovers known generating models and parameters on synthetic data." FJ wins 5/5 for gpt-4.1, gpt-5.6-sol, claude-opus-5; attractor 4/5 for gemini-3.7-flash; polarize 5/5 for claude-sonnet-5; claude-sonnet-4-5 "unresolved" (∆BIC_next<1).
- [PAPER] §3, App. B, Table 2: robustness reported as cross-issue ordering (Spearman ρ=0.83, Pearson r=0.95) and profile persistence at n=20, T=40 — "robustness of the profile and ordering, but not of the exact model at every scale."
- [PAPER] App. A: instrument internal consistency Cronbach α=0.88–0.97, pro-vs-reverse subscale r=−0.83 to −0.95, straight-lining 1–1.5%. There is no human validation anywhere; all checks are internal.

**6. Failure modes / biases / limitations.**
- [PAPER] §3, Table 4, App. E: the no-discussion baseline is not stable — without peers gpt-4.1 diverges (0.47→0.60) while claude-opus-5 collapses (0.47→0.14). Intrinsic movement is itself model-specific, so the social component is the *difference* between discussion and no-discussion values.
- [PAPER] abstract, §3: alignment drift is model-specific — gpt-4.1 keeps clusters apart, "claude-sonnet-5 holds them partially, and gpt-5.6-sol and claude-opus-5 abandon them"; under discussion claude-opus-5's sign retention falls S=1.00→0.60.
- [PAPER] Limitations: single axis; "one LLM per society" (homogeneous); English, proprietary models, 3–5 replicates; "snapshots can change"; "jump filtering, prompts, and interaction geometry may shift profiles"; "omits multi-dimensional value trade-offs and emotional dynamics."

**7. Software engineering — a sealed measurement layer.**
- [PAPER] §2: three separations hold by construction: no dynamics model in the measurement path, exposure never gated by opinion distance, storage refuses non-LLM data to the fitter. Grade parsing succeeded on 100% of surveys.
- [INFERENCE] The "storage refuses data to the fitter" pattern is a pure-observer rule in the M16.B sense; the "exposure never gated by the measured coordinate" is the one piece not obviously already in EPModel (see §B).

**8. Small-N / emotional / long-horizon.**
- [PAPER] §4, Limitations: n=10 is small-N but opinion-only; the paper states outright it "omits … emotional dynamics" and covers only two economic-policy issues, horizon 20→40 rounds.
- [INFERENCE] Nothing here models a family, a triangle, conserved emotional quantities, or years/decades. The transferable material is method, not mechanism — consistent with DESIGN_LESSONS §4.

**Not transferable / cautions.** The fitted classical models (FJ, DeGroot, BC) are already mapped elsewhere (SWEEP_READING_REPORTS_2026-09-27, candidate J4) and EPModel forbids fitting to known histories (M11.F.9); the homophily index and sign-retention require a single scalar stance axis EPModel does not possess. Caution for C34: the paper's own no-discussion control "removes both peer input and statement composition … does not isolate peer exposure" — the same conflation risk applies to a control arm that zeroes conductance while also stopping move-selection. Caution: every headline number is 3–5 replicates of proprietary English models; treat model-specific magnitudes as illustrative, not stable.

## B. Candidate additions to the EPModel spec

**MD1 — opposite-coded indicators and internal-consistency reporting for composite readouts.**
- **Proposed requirement:** Any composite readout that reduces multiple indicators to one scalar — the `basic_level` estimator (M1.A.4), the family evaluation (M11.G), and `systems_perspective` (M1.A.18a) — MUST include at least one indicator coded in the opposite direction, so that uniform movement or acquiescence across indicators yields the neutral value rather than a signal; and the run log (M16) MUST report, per composite, an internal-consistency statistic and a uniform-response ("straight-lining") rate.
- **Where it would live:** M1.A.4 / M11 (readout form) + M16 (the two statistics as log columns).
- **Evidence:** §2 and App. A — "reverse-coded items force internal consistency, so uniform agreement scores x=0"; Cronbach α=0.88–0.97, pro-vs-reverse r=−0.83 to −0.95, straight-lining 1–1.5%. SHOWN.
- **What it changes:** generalises the two-sided neutrality readout already present for `systems_perspective` (M1.A.18a: blame *or* praise is a loss) to a rule over all composite readouts, and adds a reporting rule EPModel lacks (no "internal consistency" or "straight-lining" requirement exists). Guards the counterfeit-position and mid-scale readout traps (brief §"Known readout traps"); improves inference validity.
- **Cost / risk:** low; the statistics are pure functions of state, no determinism conflict (M3.D.4/5). The consistency statistic and the straight-lining threshold are new `[I]` constants to label. Applies only where multiple indicators are aggregated; a single engine quantity (e.g. `functional_level`) has no "items," so the rule is vacuous there and MUST say so rather than force a synthetic decomposition. No conflict with M11.F.9 (descriptive stats, not fitted parameters) or M16.B (observer-only).

**MD2 — the no-interaction control must preserve self-expression and be subtracted per arm.**
- **Proposed requirement:** A no-interaction control arm (C34) MUST keep each agent's own move-selection and WITHHOLD step intact while removing peer input only, and the intrinsic drift it measures MUST be measured per arm and subtracted per arm — intrinsic (no-interaction) movement is itself a function of the arm's policy and parameters, not one shared baseline.
- **Where it would live:** Phase E (extends C34's two-level no-interaction control).
- **Evidence:** §2, §3, App. E — the no-discussion baseline moves on its own and *differently by model*: gpt-4.1 0.47→0.60 vs claude-opus-5 0.47→0.14; the social component is the discussion-minus-no-discussion difference (−0.26 for gpt-4.1, +0.09 for claude-sonnet-5). The paper's own limitation names the conflation: the control "removes both peer input and statement composition." SHOWN (per-model intrinsic drift) + ARGUED (conflation).
- **What it changes:** refines C34. C34's "delivery suppressed" variant already keeps selection while cutting delivery, so the self-expression half is partly covered; the *per-arm subtraction* (intrinsic drift is arm-specific) is new and prevents attributing a parameter set's own idle drift to the relationship system. Improves inference validity.
- **Cost / risk:** low; consistent with M0.4's one-parameter-set discipline so long as the no-interaction arm is run for each arm's configuration, not once. No conflict with determinism; no fitting. Risk is only combinatorial cost (a control per arm).

**MD3 — sign retention and per-person displacement reported as distributions, not means.**
- **Proposed requirement:** Any readout asserting that a person or family held a position under pressure SHOULD report sign retention (the fraction of persons whose position keeps sign over the window) and the per-person displacement distribution (median, IQR, fraction crossing the midpoint), never only a population mean.
- **Where it would live:** Phase E readouts / M16 run log.
- **Evidence:** Table 4 (sign retention S) and App. E "Per-agent displacement" — claude-opus-5 median displacement 0.53 (IQR 0.33–0.67), "20% cross the midpoint," vs no-discussion 0.22 with no crossings; the collapse is "broad-based rather than driven by a few agents." SHOWN.
- **What it changes:** a reporting rule that makes §2.2's "distribution, not mean" concrete for position-holding readouts, and gives the I-POSITION / differentiation readout a falsifiable population-level summary (fraction retaining sign). Improves inference validity.
- **Cost / risk:** low; reporting only. Requires choosing a position axis EPModel does not currently have — sign retention is well-defined for a scalar coordinate, so it transfers as a *pattern* for whichever position-like readout exists, not as a literal axis.

**MD4 — robustness claimed as ordering preservation, not parameter preservation.**
- **Proposed requirement:** Robustness of a directional result across a second context (family configuration, issue, or horizon) MUST be reported as preservation of the *ordering* of arms, and a robustness claim MUST NOT extend beyond that ordering to exact magnitudes or fitted parameters.
- **Where it would live:** Phase E (ensemble runner / counterfactual arms).
- **Evidence:** §3 and App. B — cross-issue ordering ρ=0.83/r=0.95 and profile persistence at n=20, T=40, but the exact BIC class "shifts in some cells"; the paper claims "robustness of the profile and ordering, but not of the exact model." SHOWN + ARGUED.
- **What it changes:** a reporting discipline that keeps EPModel's directional claims honest under a second context; sharpens §2.1's "which directional results survive an alternative regime" into "survive as ordering." Improves inference validity.
- **Cost / risk:** low; reporting only. No conflict with any stated rule. Weakest of the four — it is a phrasing discipline more than a mechanism.

**Already covered (not written up):**
- "Measurement independent of the dynamics model / storage refuses data to the fitter" → M16.B (pure observer) + M1.A.4j (circularity boundary) + arm-blindness (C-note from PIMMUR) already cover the seal; the only new piece is the exposure-never-gated-by-measurement rule, folded into the MD1/MD2 discussion.
- "Unresolved" model-selection outcome (∆BIC<1 → no winner claimed) → covered by UNDETERMINED (C26, M17.A.1) and E-RM(c) "not discriminating."
- "Model choice is a first-class validity variable; report and vary it like demographics" → covered by DESIGN_LESSONS §3.3 and M3.D.6 (LLM line only; no v2 decision-path change).

---

# Part 2: A Framework for Context Enrichment and Selective Robustness in LLM Social Agents (2611.06914v1)

## A. Report

1. **Simulation formalism — one-shot, memoryless, single agent** [PAPER]. One focal LLM agent (emergency-supply manager, Hospital A) makes a single simultaneous-choice Prisoner's-Dilemma-like decision; "Every request had no conversational history" (§B.1). Full factorial 2 worlds × 5 contexts × 4 surfaces = 40 cells per model, 10 calls/cell, 800 calls total (§B.1). Payoffs T=5>R=3>P=1>S=0 (§B.1). No ticks, no dynamics, no event queue, no memory — no formalism transfers to a tick/event engine. [INFERENCE] This is a static elicitation, not a simulation.

2. **Agent architecture** [PAPER]. Role+setting prompt only; decision read from token log-probabilities at the first output position, up to 20 candidates, t/(t+r) transfer probability (§B.3). temperature 0.7, reasoning effort none, 64-token output limit, zero retries (§B.3). No reflection, memory, learning, or heterogeneity beyond the two models (GPT-5.6 Sol and Terra).

3. **Interaction structure** [PAPER]. One dyad: two hospitals, symmetric payoffs, simultaneous choice, "You cannot communicate" (§B.2). No triads, no networks, no cascades.

4. **Initialization — no content.** The paper never discusses seeds, initial conditions, or sensitivity to them.

5. **Calibration / validation — the paper's substance is a validation criterion** [PAPER]. "Selective robustness" = surface invariance (stability across four meaning-preserving surface forms) + world responsiveness (regional-minus-local transfer, direction prespecified before observing behavior, §2.2). Both reported with 95% bootstrap CIs (10,000 resamples of the 10 calls/cell, §B.3). Table 2: descriptive/normative grounding raise world responsiveness in both models (Sol 85.9/92.0; Terra 57.2/77.1 pp) and cut Sol's surface sensitivity to 3.7/3.9 pp but leave Terra's at 13.2/15.0 pp. Explicitly "intended to illustrate the framework's utility, not to establish a general effect" (§1) and "serves as an internal diagnostic" (§4).

6. **Failure modes / limitations** [PAPER]. (a) Surface-form brittleness: Terra local normative, alternative prose yields 89.9% transfer vs "at most 0.9%" for the other three surfaces (§3, Table 3); echoes Ye et al.'s 76 pp shifts (§1). (b) Model dependence: Sol and Terra differ in robustness under identical conditions (Fig. 1, §3). (c) "Neither property is sufficient on its own": low surface sensitivity "can also arise if the model simply ignores the social context," and strong responsiveness "may be overtly brittle" (§2.2). (d) Magnitude-vs-direction: a model at 100%/0% transfer "would follow the expected direction but exaggerate the size" (§4). (e) Salience confound: grounded prompts "add more detail about the social-world distinction," so grounding is not separated from increased salience (§4). (f) No human raters checked surface-form semantic equivalence or unintended normative force (§4).

7. **Software-engineering lessons** [PAPER]. Code and data released (§1). Modular frozen prompt scaffold — fixed opening + social-context block + fixed decision block — so "all 40 prompts can be reconstructed" (§B.2). Measurement discipline: a provider top-p default of 0.98 silently omitted one action candidate in 339/800 calls; handled by bounding the estimate to the unreturned probability mass, and a sensitivity analysis showed aggregate measures change ≤0.49 pp with "all context rankings within each model" preserved (§B.3).

8. **Small-N / long-horizon / family / emotion — no content.** One focal agent, one decision, one scenario, one model family (§4). Nothing on families, emotion, stress, or horizons.

**Not transferable / cautions.** This is an LLM-validation paper; its central objects (surface form, prompt context) do not exist in the rule-based v2 engine (`M3.D.6`). The one-shot memoryless design has no dynamics, no conserved quantities, no triangles. Its transferable value is confined to (a) the exploratory LLM line and (b) validation-discipline patterns that the design lessons already contain. The 89.9%-vs-≤0.9% anomaly and the Sol/Terra split are further instances of §3.3's prompt/model-sensitivity findings, not new mechanisms.

## B. Candidate additions to the EPModel spec

**SR1 — declared function per context block (narrator/LLM line only).**
- **Proposed requirement.** For the exploratory LLM/narrator line, any context supplied to an agent MUST be decomposed into named functional blocks (role/task, minimal world facts, background, descriptive decision grounding, normative grounding), each with a declared purpose, and the blocks MUST substitute for one another rather than accumulate, so conditions are compared on function rather than length.
- **Where it would live.** Phase F / exploratory notes (X-series); not a v2 module.
- **Evidence.** §2.1 and Table 1 (five conditions, substitution rule); §4 "researchers should specify what each part of the supplied context is meant to do" — ARGUED.
- **What it changes.** Protocol discipline for the LLM line only; no engine change. Improves inference validity of LLM-line results (a "richer context" intervention that also raises the salience of the world distinction confounds mechanism with salience — §4's confound).
- **Cost / risk.** Nil for v2. No conflict with `M3.D.6` (this constrains prompt construction, not the decision path).

**SR2 — disclosure and direction-preservation check under readout imputation.**
- **Proposed requirement.** When a reported readout is imputed from a bounded, censored, or clamped quantity, the run report MUST disclose the imputation rule and MUST include a sensitivity analysis showing that varying the imputed value across its bounds changes the reported direction by no more than a declared tolerance; a direction that flips within the bounds MUST be reported as not established.
- **Where it would live.** M16 run log + Phase E reporting (M17), readout side.
- **Evidence.** §B.3: top-p 0.98 default omitted a candidate in 339/800 calls; interval-midpoint imputation; sensitivity analysis ≤0.49 pp with rankings preserved — SHOWN.
- **What it changes.** Adds a reporting rule; inference validity. Partly overlaps C31 (threshold-margin log), which addresses threshold *comparisons*; this addresses readout *imputation* (clamping/rounding/omitted-mass), the same class as §2.7's clamping artefact.
- **Cost / risk.** Low; post-hoc, observer-side. No conflict with `M3.D.4`/`M3.D.5` (analysis after the run) or `M16.B` (pure observer).

**Already covered (no write-up):**
- Selective-robustness pairing (invariance + directional response; "neither property suffices," the invariant-but-inert failure) → C21 + design-lessons §2.6 sever-the-input mutants + `M0.4` direction criteria.
- Magnitude-vs-direction caveat ("follow the expected direction but exaggerate the size"; internal diagnostic not fidelity) → `M0.4` / brief's "direction of difference, never an absolute threshold."
- Prespecifying the expected direction before observing behavior → `M0.4` / `M11.C` direction criteria.
- Substitution-not-accumulation as clean comparison design → `M11` one-rule-at-a-time mutation / C22 minimal rule sets.
- Model-dependence of robustness (Sol vs Terra) → design-lessons §3.3 (prompt/model sensitivity).

---

# Part 3: Same Method, Different Task: What the Bitter Lesson Does Not License in Social Simulation (2611.05717v1)

## A. Report

**1. Simulation formalism.** [PAPER] The pilot (§4.1, App. A) is a round-based common-pool resource: four agents share a buffer (initial 200, capacity 300) over 8 periods, each submitting a requisition of 0–36 units per period; all agents act each period (synchronous, no simultaneity rule discussed). A partial-restock rule floors the buffer at zero then restocks 20% of the gap if combined requisitions <50% of pre-extraction level, else 10%. Collapse (removal of all agents at buffer zero) "never fires": 0.10×300=30>0. Fixed temperature 1.0; no seeding discussion (LLM calls are non-deterministic). [INFERENCE] Horizon of 8 periods vs EPModel's ~2,080 ticks; nothing on timescale or determinism transfers.

**2. Agent architecture.** [PAPER] Agents are frozen LLM predictors: "a frozen model reacts to a changing balance in its context, but nothing it experiences updates its policy" (§4.1). §5 separates three things — persistent consequences (state carrying forward), represented utility, and learning from returns (policy updated by outcomes) — and states the pilot tests only the first. No reflection, no in-episode memory, no heterogeneity by construction.

**3. Interaction/network structure.** [PAPER] Groups of four, but the environment has no payoff externalities: "requisitions are always granted in full … there is no channel by which one agent's behavior affects another's outcome" (§4.6). The "group" is not interdependent; there is nothing to free-ride on. [INFERENCE] This is the paper's own central limitation, and it undercuts its headline result.

**4. Initialization.** [PAPER] Reserve starts at 50, buffer at 200/300 (§4.1, App. A); no sensitivity to initial conditions is reported. Persona/preference diversity "was not manipulated," so the mode-collapse rival is untested (§4.6, §5).

**5. Calibration/validation.** [PAPER] Hypotheses and analysis plan frozen 17 Aug 2026, pre-registered (§4.1, App. A). Primary (B-schedules show a floor near operating cost at low others' withdrawal) supported for Llama (−12.1 units, 95% CI 10.2–13.8) and Nova (−8.8, CI 5.8–11.4), not Claude (CI −2.5 to +1.9) — 2 of 3 (§4.3). Secondary (TVD(B,human)<TVD(A,human)): Nova marginal 0.483→0.440, Llama flat 0.500, Claude worse; human baseline from a different game, qualitative only (§4.4, App. E). Free riders are 0.00 in every cell; 92%/100%/95% are one cooperative type; humans ~30% free riders (Fischbacher et al.) (§4.4, App. B Table 2). Bootstrap CIs resample groups (n=15/cell), not agents (§4.2, App. D).

**6. Failure modes and negative results.** [PAPER] (a) End-game exploitation: Claude A spikes +17.7 units above mid-game mean, Llama A +9.2; B suppresses (Claude B +6.0, Llama B +0.6 ns) — Claude runs the cooperative script only while the stated rationale holds (§4.2). (b) Nova's inverted trajectory: extraction rises +0.65/period in B; 32% insolvency by period 8 (Claude 22%, Llama 25%) (§4.2). (c) Claude bankruptcy: 13/60 B-agents insolvent, mean requisition 13.4 below the 13.75 survival threshold — it followed the described script and missed its own survival constraint (§4.5). (d) Buffer=30 is a mechanical attractor of the restock rule, not coordination — scripted always-max/always-20/random-uniform agents converge to 30 in 2–4 periods (§4.5, App. G Table 7). (e) Condition B is a bundled treatment (state + arithmetic + feedback + removal); the missing A′ control would isolate stakes (§4.6, §5). (f) SEs computed per agent "ignore that agents in the same group are correlated, so they are too narrow" (§4.2).

**7. Software-engineering lessons.** [PAPER] Pre-registration of the analysis plan (App. A); scripted-agent baselines to establish an attractor is mechanical (App. G Table 7); survivorship imputation bounds for missing respondents (App. F); an implementation-check figure verifying the manipulation binds (App. D Fig. 5); total API cost logged — 4,768 calls, 4.2M input tokens, 343K output tokens (App. G).

**8. Small-N, emotion, family, long horizon.** [PAPER] Groups of four, 540 agents total (60/condition/model, 135 groups). Nothing on families, emotional process, or years/decades. [INFERENCE] The transferable content is method-level (interest/stakes as constitutive of a mechanism; survivorship; bundled-treatment isolation), not any family or emotional mechanism.

**Not transferable / cautions.** (1) Nearly all empirical findings are about LLM agents; EPModel excludes an LLM from the decision path (`M3.D.6`), so they support the design-lessons §8.8 conclusion and add nothing to v2's mechanisms. (2) The TVD-to-human calibration is unavailable to EPModel (no validation data). (3) The headline result (zero free riders) is, by the paper's own §4.6 analysis, predicted by the environment's absence of a dilemma as much as by disinterest — the pilot "cannot rule out other explanations," so treat the empirical work as suggestive, not a test.

## B. Candidate additions to the EPModel spec

**BL1 — Survivorship imputation bounds on any readout computed over living persons.**
- **Proposed requirement:** Any `M11.C` criterion whose observable is computed over living persons only — symptom concentration (`M11.C.2`), repertoire entropy (`M16.A.9`, `M11.C.16`), functioning-balance or differentiation readouts — **MUST** also be reported under two declared imputation bounds for persons who died during the run: an implausible bound counting each dead person as the mechanism's extreme value, and a defensible bound counting them as the family-typical value. The report **MUST** state the direction of the selection bias (whether excluding the dead works for or against the asserted direction).
- **Where it would live:** M11 acceptance tests; M16 run log.
- **Evidence:** App. F and §4.6 (SHOWN as technique): removed B-agents (22–32% of cells) were the most restrained (mean requisition 11.7–13.4 vs 16.0–17.7 for survivors), so measuring only survivors biased against the hypothesis; the paper bounds TVD under dead→free-rider (0.250, implausible) and dead→conditional-withdrawer (≥0.44, defensible) imputations. Transfer to EPModel's mortality readouts is INFERENCE.
- **What it changes:** Adds a reporting rule; improves inference validity. `M6.3`/C15 already dispose of a dead person's *quantities*; this addresses the distinct question of who is in the readout's *denominator*, so a direction is not an artefact of who died.
- **Cost / risk:** Low — pure reporting, no engine change, no new constants beyond declared imputation values. No conflict with M3.D.4/M3.D.5, M3.D.6, or M11.F.9 (imputations are declared, not fitted); consistent with M16.B's pure observer.

**BL2 — Non-binding-consequence mutant (bundled-treatment isolation; the paper's A′ control).**
- **Proposed requirement:** Where a mechanism adds both a tracked state quantity and a consequence that reads it (e.g. `emotional_reserve` gating symptom onset, `M4.D.5d`; any stock whose depletion fires a removal), the mutation suite **MUST** include a **non-binding-consequence mutant**: the quantity is still tracked, updated and displayed exactly, but its consequence never fires. The affected criterion **MUST** go red under it, else the mechanism is indistinguishable from constraint-satisfaction on the displayed arithmetic.
- **Where it would live:** M11 mutation protocol (M11.D); M10 config (declared experimental override, following `M10.C.4a`'s pattern).
- **Evidence:** §4.6 and §5 (ARGUED): B "bundles state, arithmetic, feedback, and removal," so the B–A contrast "cannot distinguish it from simply satisfying newly stated arithmetic"; the proposed A′ control is "identical arithmetic and feedback but non-binding consequences." The Claude bankruptcy result (§4.5, SHOWN) — an agent satisfying the described script while missing its live constraint — is the exact confound this mutant isolates.
- **What it changes:** Adds a test (a named mutant form); improves both theory fidelity (selection responds to a *consequential* quantity, not a number) and inference validity. It is the complement of C17 (which preserves the input's magnitude while destroying its state-contingency): this mutant preserves the quantity *and its display* while removing only its consequence.
- **Cost / risk:** Low–moderate — one more mutant per consequential mechanism. No conflict with M3.D.4/M3.D.5 (a deterministic override), M3.D.6, or M11.F.9 (a test, not a parameter). Care required not to conflate it with deleting the mechanism outright, which would also remove the tracking.

**Already covered (not written up):** "the type distribution is the phenomenon, not aggregate agreement" — C34's homogeneous-family arm, X2's heterogeneity floor, and E-RM already require criteria to assert *which* member and the shape, not presence; `M11.C.2` already names the member. Pre-registration of hypotheses/analysis — C23 (constants frozen before tests) and the spec's pre-declared `M11.C` criteria. Mechanical-attractor/scripted-null baselines — design-lessons §7.5 null model, E-RM, C34's no-interaction arms, §2.7 functional-form artefacts. "Persistent state ≠ interest; heterogeneity requires learning-from-outcomes" — supports `M4.D.6`; the learning-off arm is already C18. "Stakes, not emergence" — supports `M3.D.6` and the rule-based design; adds no new requirement.

---

# Part 4: Does Personality Persuade? Investigating Personality Trait Prompting and Persuasion in Large Language Models (2611.06572v1)

## A. Report

**1. Simulation formalism** — None. [PAPER] This is not a simulation: a factorial calibration study plus a within-subjects human experiment. The only "scheduling"-adjacent content is measurement hygiene — each of the 120 IPIP-NEO-120 items answered in a separate request with no carried context, 10 repetitions per profile (243 profiles × 120 × 10 = 291,600 requests per model-format pair, §3.1). [INFERENCE] Nothing transfers to `M3` clocks/ordering.

**2. Agent architecture** — [PAPER] The "persona" is a system-prompt fragment, not an architecture: CAT format (Low/Average/High) vs NUM (20/50/80). NUM "suggests a precision that the model cannot deliver" — numeric targets missed systematically (+11.6 for A=20, +11.4 for C=80, −5.2 for N=80; §3.2). [PAPER] Controllability is strongly model/configuration-dependent: direct effects 42.3–73.5 points for the five ≥20B reasoning models vs 2.2–38.2 for Gemma4 e4B and non-reasoning variants (§3.3, Table 4); size and reasoning both matter (Gemma4 e4B violated Low<Avg<High ordering in 64% of Neuroticism contexts).

**3. Interaction/network structure** — [PAPER] Only dyadic human↔agent persuasion dialogues (128 conversations, 597 agent messages, §5.3). No network, no inter-agent dyads, no triangles. [INFERENCE] Nothing transfers.

**4. Initialization** — [PAPER] The agent's persona is initialized from the participant's own IPIP-NEO-120 (Aligned) or inverted (Mismatched); Neutral and unconditioned Base are the other arms (§4.1). [PAPER] Result: initialization-by-alignment did not matter (item 6). [INFERENCE] A rare clean instance of "initial condition is a free variable that didn't move the outcome."

**5. Calibration and validation** — [PAPER] The paper's core move: "treat the prompt as the specification, the profile measured with the IPIP-NEO-120 as the observed behavior, and examine the difference between them" (§3). Reliability measured with ICC(1,1) and ICC(1,10): .928–.997 for capable models, down to .681 for Gemma4 e4B; a single generation deviated only 1.4–5.1 points from its profile mean (§3.3, Table 2). [PAPER] What they could not validate: N=32, four dilemmas, 8 stance changes, generalization limits conceded (§7). [PAPER] An LLM-assisted labeling procedure with human oversight (§5.3) — not a validated instrument (see DP3).

**6. Failure modes / negative results** — [PAPER] Headline null: personality alignment did NOT significantly affect perceived persuasiveness (Friedman χ²(3)=2.88, p=.410, Kendall's W=.030), stance switching (Cochran's Q=2.40, p=.494), or confidence change (Q=6.86, p=.077) (§5.2). Stance changed in only 8/128 conversations (6.25%); descriptively the result was *opposite* H1 (Mismatched M=41.44 highest, Aligned M=34.41 lowest). [PAPER] Unconditioned models are not neutral: base C>70 and A>65 for five models, N<30 for three; Gemma4 26B C=99.8, Mistral C=92.4 (Table 1). [PAPER] Cross-trait spillover (item below). [PAPER] Homogenization: logical appeals in 88.2–100% of agent messages across all conditions/dilemmas (§5.3). [PAPER] One co-worker Base conversation triggered a safety refusal, yielding no persuasion strategies (§5.3). [PAPER] Frustration observed in 22 conversations, most in Aligned/Mismatched (Table 15).

**7. Software engineering** — [PAPER] Local inference via Ollama v0.33.3 chosen explicitly to avoid "updates to hosted models that would compromise reproducibility" (§3.1); temperature 0.7 applied uniformly; full prompts in Appendix A for reproducibility. [PAPER] Graeco-Latin 4×4 square counterbalances condition against dilemma and position, with adaptive assignment to equalize cell sizes (§4.1). [PAPER] LLM annotation prompt requires a verbatim evidence quotation + one-sentence reason per label, and manual review/correction of snippets (App A.1).

**8. Small-N / emotion / long-horizon** — [PAPER] N=32 small-N human study; participants' own OCEAN traits showed no significant association with susceptibility to persuasion (all Holm-adjusted p=1.000, §5.2). [PAPER] Confidence change is more frequent than stance change (43.75%/50% Aligned/Mismatched vs 31.25%/25% Base/Neutral) and sometimes reinforces the original position — "persuasive influence cannot be adequately characterized by stance switching alone" (§6). [PAPER] Higher AI persuasion literacy associated with fewer confidence changes (ρ=−.441, adjusted p=.046). [INFERENCE] No family, stress, or multi-year dynamics; nothing transfers to the core engine.

**Not transferable / cautions.** The paper is entirely on the exploratory LLM line; it has zero content for the rule-based v2 engine (no dyads, no conserved quantities, no years-long horizon, no small-N family). The persuasion result is a human-perception study at N=32 with rare stance changes — under-powered for the effect it sought, by its own admission (§7); treat the null as "no detectable effect," not "no effect." The LLM-assisted labeling is not a validated instrument and should not be cited as one. The within-subjects human design (counterbalancing, paired sign/McNemar tests) is a human-experiment technique, not a simulation technique.

## B. Candidate additions to the EPModel spec

**DP1 — An "unconditioned / neutral" LLM baseline is itself a persona and must be measured, not assumed neutral**
- **Proposed requirement.** In the exploratory LLM-agent line, any arm that uses an unconditioned or explicitly "neutral" LLM as a baseline or control MUST be treated as carrying its own personality profile, never as a personality-free null. The arm's realized OCEAN profile MUST be measured with the same instrument used for conditioned arms and reported beside them, and a claim of "no persona" MUST NOT rest on an explicitly requested Average/neutral profile, which does not remove the model's default Agreeableness/Conscientiousness elevation.
- **Where it would live.** Exploratory LLM notes (sharpens the X4 neutral-persona contrast, candidates file §8.8).
- **Evidence.** §3.3 + Table 1: base models show C>70, A>65, N<30 for three of five; Gemma4 26B C=99.8, Mistral C=92.4; requesting Average for every trait moved scores toward midpoint but "Agreeableness and Conscientiousness remained above 50 in all five models (up to 68.1 and 67.2)". Strength SHOWN. Partly covered by design lessons §3.2 (pull toward agreeable) and X4 — the new bit is that the *neutral-request* arm also fails to neutralize.
- **What it changes.** Corrects/strengthens X4: the neutral distribution is itself biased and off-midpoint, so it must be measured rather than treated as the midpoint of the scale. Improves inference validity of every LLM-line comparison against a "baseline."
- **Cost / risk.** No engine change, no new constants. Consistent with `M3.D.6` (further reason LLMs stay out of the decision path). Reporting rule only.

**DP2 — Prompted Big Five traits are not independently settable: cross-trait spillover must be measured and reported**
- **Proposed requirement.** In the exploratory LLM-agent line, a Big Five persona MUST NOT be treated as a set of independently settable trait levels. Any protocol that varies one trait MUST measure and report spillover onto the other four (mean absolute change per trait), and MUST NOT attribute a behavioral difference to a single trait without reporting whether co-varying traits also moved.
- **Where it would live.** Exploratory LLM notes (alongside X2's heterogeneity floor).
- **Evidence.** §3.3 + Fig. 2: mean absolute spillover 4.3–6.6 points (7.3–10.7% of mean direct effect), individual cells up to 11.6–17.3; largest signed spillovers — raising Conscientiousness lowered Neuroticism by 13.2, raising Openness raised Extraversion 11.0, Agreeableness raised Openness 9.8, Extraversion lowered Neuroticism 9.2. Strength SHOWN.
- **What it changes.** Adds a measurement/reporting rule; prevents attributing an outcome to one trait when the manipulation moved others. Improves inference validity of persona-line results. New — no existing candidate or design-lesson item names spillover.
- **Cost / risk.** Low; exploratory line only; no engine change. No conflict with stated rules.

**DP3 — LLM-assisted content annotation requires forced evidence, human oversight, and condition-blindness**
- **Proposed requirement.** If the exploratory/narrator line ever uses an LLM to classify or annotate generated content, the annotation protocol MUST require each label to carry a verbatim quotation from the target message plus a one-sentence reason, MUST route all labels through human review/correction, MUST forbid the annotator from using knowledge of the experimental condition, and MUST forbid inferring persona or intent beyond what is observable in the text.
- **Where it would live.** Exploratory LLM notes (analysis/narration line); related to design lessons §3.4, which already forbids LLM-as-judge — this is the *contract* if an LLM is used as an annotator anyway.
- **Evidence.** §5.3 + Appendix A.1: annotator prompt demands a "short quotation from target message" and a one-sentence reason per label, returns snippets "to enable manual review and correction," and instructs "Do not use knowledge of the experimental condition" and "do not infer the AGENT's personality… beyond what is observable in the text." Strength SHOWN as a practised protocol (though the resulting annotations were not independently validated — item 6).
- **What it changes.** Adds a process/reporting rule for any LLM-assisted labeling; improves inference validity of content analysis on the narrator line.
- **Cost / risk.** Low; no engine change. Must stay downstream of the decision path to respect `M3.D.6`; does not license an LLM in the decision path itself.

**Already covered (one line each):**
- *Validate the realized persona per model, not the prompt; controllability varies with size/reasoning* → design lessons §3.3 + X1/X2/X3 (per-model sensitivity and the persona-validation protocol are already there).
- *Controllability ≠ influence (the persuasion null) as a spec requirement* → X1 already requires a closed-action behavioral readout beside any questionnaire readout; the paper's N=32 null is recorded in Section A item 6 as a caution, not a new requirement.
- *Reproducibility via local/frozen models* → design lessons §3.1 (a hosted model cannot satisfy `M3.D.5`).
- *ICC(1,1) vs ICC(1,10) variance decomposition* → design lessons §2.6 "separate the variance sources."
