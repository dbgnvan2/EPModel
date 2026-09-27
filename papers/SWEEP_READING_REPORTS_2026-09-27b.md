# Reading reports: second batch of the 2026-09-27 alerts ("agent based model, social simulation")

Produced 2026-09-27. The owner's alert export (`~/Downloads/agent based model_ social simulation _ 2026-09-27 - summaries.pdf` and its CSV listing, 8 papers) was screened on its summaries; *What People Almost Did* had no summary and was screened on its abstract. Four papers were read in full from `pdftotext -layout` output by readers working under `sweep_readers_brief_2026-09-19/READER_TASK.md`, with spec revision 10, `SPEC_CANDIDATES_from_preprints_2026-09-20.md` and `SWEEP_READING_REPORTS_2026-09-27.md` checked for overlap by keyword search. Reports are reproduced as written.

**Status: none of these candidates is in the spec.** Plain-language explanation with examples: `SPEC_CANDIDATES_plain_language_2026-09-27b.md`.

| Part | Paper | Candidates |
|---|---|---|
| 1 | Akbar, Platnick, Alirezaie, Rahnama & Pentland, *Bayesian Belief Layer for Controllable Opinion Dynamics in LLM Agents*, arXiv 2609.21997v2 | BB1–BB5 |
| 2 | Li, Chen, Fung, Rossi & Li, *Mind or Message? Auditing Theory of Mind in Multi-Agent Social Simulation*, arXiv 2609.24146v1 | MM1–MM4 |
| 2 | Kim & Boggust, *What People Almost Did: Evaluating LLM Social Simulations Beyond Behavioral Fit*, arXiv 2609.20055v1 (position paper, no experiments) | AD1–AD3 |
| 3 | Tareaf, *Diverse Minds, Divided Networks?*, arXiv 2609.12444v1 | TM1–TM6, X10 |

Screened on summary only, no candidate derived: 2609.28609 (AdvRole; also in the first batch), 2609.21857 (personality-tuned LLMs), 2609.16436 (interpreting and steering LLM agents), 2608.29803 (LLM vs human persuasion judgments — slight agreement, kappa 0.08–0.18; further support for M3.D.6, nothing to add).


---

# Part 1: Akbar et al. 2026 (Bayesian Belief Layer)

## Reader report: Akbar, Platnick, Alirezaie, Rahnama & Pentland 2026, "Bayesian Belief Layer for Controllable Opinion Dynamics in LLM Agents" (arXiv 2609.21997v2, 22 Sep 2026)

**Scope.** I read the whole text file (531 lines: body, references, Appendices A–E). Figures survive only as captions. Equations are partly garbled but can be reconstructed.

I checked spec revision 10 by keyword search and by reading these passages: M1.A.4a–j, M1.F.1–F.5, M3.D.6, M4.C.1, M4.C.9, M9.1–M9.8, M11.4c–f, M11.D.19–21, M16.A, M16.C, M16.E and M17.D.2–4. The spec has no hits for "closed-form", "closed form", "analytic", "DeGroot", "Friedkin", "stubborn", "Spearman" or "reconstruct". The design lessons and candidate files I read are those named in the task.

### A. Report

**1. Simulation formalism**
- [PAPER] **State and time.**
  - Each agent's stance on one concept is a Beta(α, β) pseudo-count pair, held outside the LLM. Credence is b = α/(α+β) (§3.1, App. E).
  - One synthetic topic with a binary stance pair: rail versus roads in the fictional town of "Aldenvale" (App. D).
  - Round-robin turns. Each round, every agent speaks once. Each utterance is appraised once, and that single number e reaches every listener (§4). All listeners get the same evidence, so the listener's own state enters only through its own κ and history.
- [PAPER] **The update (Eq. 1).**
  - α ← α0 + γ(α − α0) + we, and β ← β0 + γ(β − β0) + w(1 − e).
  - Prior strength κ = α0 + β0 is the only per-agent knob. The forgetting factor γ is fixed globally at 0.7, and w = 1.
  - App. E.2 gives the exact one-step identity b′ = b + η(e − b) + ρ(b0 − b), which is a Friedkin–Johnsen (FJ) step. η falls as κ rises. The anchor term ρ exists only when γ < 1.
- [PAPER] **Sequential engine, synchronous reference.** The engine applies one utterance at a time. The FJ reference is the synchronous mean-field fixed point b* = (I − ΛW)⁻¹(I − Λ)b0 (App. E.3). The authors call the mismatch small and cite R² = 0.93–0.99 (Limitations).
- [PAPER] **Determinism.** Generator temperature is 0.9 and appraiser temperature is 0. No reproducibility claim is made. The authors note that two LLM calls per utterance limit exact reproduction (Limitations).
- [PAPER] **An exact identity.** The total pseudo-count n is a known function of (κ, γ, number of utterances heard), independent of what was said (App. E.1). That is what allows κ to be inverted event by event from the log (App. E.4).

**2. Agent architecture: owner interest (a)**
- [PAPER] **Division of labour.**
  - The generator LLM renders the current belief as speech. The belief enters the prompt as a nine-bin verbal descriptor, with "no numbers appear in any prompt" (App. D).
  - A separate appraiser LLM reads each heard utterance and returns p_plus in [0, 1]. After calibration this is e.
  - Only e reaches the belief, through the deterministic Bayesian step (§3.2).
  - The paper's own summary: belief follows a transparent rule, and how the agent speaks "is decided by the LLM" (§1).
- [INFERENCE] **This is not the same pattern as EPModel's "render, don't decide" narrator.**
  - The belief is decided outside the LLM, which matches.
  - But the LLM's rendering, read back by another LLM, is the only channel through which one agent influences another. So the language layer sits inside the influence path, closing a loop of state → text → appraised scalar → state.
  - EPModel's narrator is strictly downstream: M16.E.1 keeps narration off the ensemble path, and M3.D.6 keeps LLMs out of decisions. BCA is "render, and let the reading move state."
- [INFERENCE] **The paper's own results show what that loop costs.**
  - In the pliable limit, consensus lands at 0.15 for claude-sonnet-4-6 and 0.01 for Llama-4-Scout, against a reference of 0.50 (§5.2, Table 2, Fig. 4b). The cause is one-sided misreadings that add up over rounds.
  - This is direct evidence for keeping M16.E.1 as it is.
- [INFERENCE] **What BCA adds for EPModel is the round-trip as a measurement instrument, used offline.**
  - EPModel can take the paper's prescribe-then-recover test and the channel transfer curve.
  - It should not take the feedback.

**3. Interaction and network**
- [PAPER] N = 20 on a complete graph with uniform weights W_ij = 1/(N−1) (§4, App. E.3). There is no tie dynamics, no witness distinction, and every listener hears every utterance.
- [PAPER] **Three regimes from one knob (§5.2, Fig. 2, Table 2).**
  - κ → 0 (run at κ = 0.1) gives DeGroot consensus. Spread falls by three orders of magnitude.
  - Stubborn camps (κ = 16) at the extremes with pliable κ = 1 agents between them give persistent FJ disagreement. Per-agent final beliefs match the FJ fixed point at R² = 0.93–0.99.
  - A committed minority (κ → ∞) against a free κ = 4 majority, swept from 5% to 40% committed, moves the majority's mean smoothly with no tipping point.
- [PAPER] **Two structural results proved and then confirmed.**
  - Exact Bayes (γ = 1) forces consensus on any connected graph, whatever κ is assigned (App. E.3).
  - An affine update cannot produce a tipping point: "Tipping requires a nonlinearity" (App. E.5).
  - The forgetting ablation (Table 1, gpt-5.4-mini, 5 seeds): at γ = 1, FJ variance falls from 0.053 to 10⁻⁴, R² against the fixed point falls from 0.97 to −1.1, κ relative error rises from 0.49 to 1.41, and the share of well-conditioned events falls from 0.76 to 0.12.

**4. Initialization**
- [PAPER] Initial stances come from the prior means b0 = α0/κ. The DeGroot reference is "the true initial mean of 0.50" (§5.2).
- [PAPER] γ = 0.7 was chosen by an oracle sweep with no LLM calls, before any paid run, and held fixed across models and regimes (App. A, Fig. 3). FJ cross-agent variance rises from 9×10⁻⁵ at γ = 1 to 0.056 at γ = 0.7.
- [PAPER] The topic is fictional so that no model brings a strong trained-in prior (App. D). The results therefore "do not speak to debates" where models do have one (Limitations).

**5. Calibration and validation: owner interests (a), (b), (d)**
- [PAPER] **Recoverability, the core test (§5.1, Eq. 3, App. E.4).**
  - κ was swept over {0.5, 1, 2, 4, 8, 16, 32}: 14,000 utterances and 266,000 listener events per model.
  - For each event, the exact one-step identity is inverted with the speaker's latent belief s in place of the appraised e.
  - Inverting with e would return κ "by construction". With e, the inversion returns the prescribed κ to numerical precision on all four models, which is an exactness check of the code.
  - With s, any remaining error is attributable to the generate-then-appraise channel.
  - Events where the listener already nearly agrees (|s − b| ≤ 0.05) are discarded as uninformative. Retention is reported by condition, rising from 0.45 at κ = 0.5 to 0.92 at κ = 32, so the authors say the filter is not selective (App. A).
- [PAPER] **Recovery results (Table 2, Table 3, Fig. 5).**
  - Rank order is exact: Spearman 1.0 on every model, and "for every individual seed", not only in pooled medians (Table 2 caption).
  - Magnitudes are attenuated uniformly: κ = 32 is recovered as 25.3–30.0, and κ = 0.5 as −0.1 to 0.3.
  - Relative error is 0.27–0.49.
  - Channel alignment (Pearson of e against s) is 0.94–0.97.
- [INFERENCE] **Spearman 1.0 is an easy test at this spacing.** The seven levels are a factor of 2 apart. Per-person differences in EPModel will be much finer.
- [PAPER] **Closed-form references for owner interest (b).**
  - DeGroot consensus at the initial mean.
  - FJ fixed point b* = (I − ΛW)⁻¹(I − Λ)b0, with λ_i = w/(κ_i(1−γ) + w). R² is computed directly against b* with no refitting and no intercept (App. E.3).
  - A "stylised full-following reference" for the minority case, described as "not a proven bound" (App. E.5).
- [PAPER] **Appraiser calibration, owner interest (d) (§4, App. B).**
  - 100 utterances, LLM-generated to cover all stance bands.
  - Each has a soft label y in [0, 1] from a single human annotator, with a one-line rationale.
  - Temperature scaling p_cal = σ(logit(p+)/τ), with one τ per model, fitted by cross-entropy on 80 items and evaluated on 20 held-out items.
  - Validation ECE: 0.101 → 0.052 (gpt-5.4-mini), 0.103 → 0.046 (gpt-5.4), 0.040 → 0.034 (Llama-4-Scout), 0.081 → 0.033 (claude-sonnet-4-6).
  - Fitted τ = 1.51–2.15.
  - Each fit is frozen and reused, and is independent of κ and γ.
- [PAPER] **What calibration cannot do (App. C).** Temperature scaling "rescales confidence symmetrically about the neutral point", so directional bias passes straight through. The authors restate this in the Limitations.
- [PAPER] **Separate behaviour audit (App. C).**
  - Each round, each agent gives a 0–100 self-report through a separate probe that "never enters any belief update".
  - It correlates with the hidden belief at r ≈ 0.98–0.99, over 18,000 reports per model.
- [INFERENCE] That probe is a self-report from the same model family. It shows the belief governs what the agent says, not that the text is valid to an outside reader.
- [PAPER] **Validation they could not do.**
  - No human trajectories.
  - The calibration set is the only human-labelled data: one annotator, 20 held-out items, one topic.
  - Rendering and reading are not separated. Cross-appraising one model's utterances with another's appraiser is "left to future work" (App. C).

**6. Failure modes and biases: owner interest (c)**
- [PAPER] **Channel transfer curves (App. C, Fig. 4a).**
  - GPT models: symmetric exaggeration, with moderate stances arriving as more extreme on both sides.
  - Llama-4-Scout: a uniform shift toward a−.
  - claude-sonnet-4-6: exaggerates only the a− side (a mild 0.35 arrives as 0.24) and is roughly faithful on the a+ side.
- [PAPER] **Why the shape matters.**
  - Symmetric distortion cancels across a balanced population. GPT consensus lands at 0.44–0.50.
  - One-sided distortion adds a push in the same direction every round. Pliable agents "have nothing to resist it with."
  - Stubborn agents are re-anchored every step, so FJ lands near theory on all four models: "stubbornness absorbs channel bias" (Fig. 4c caption).
- [PAPER] An end-to-end LLM simulation "would report these displaced consensuses as findings" (§5.2).
- [INFERENCE] **For EPModel's narrator.**
  - Directional narrator bias does not compound in EPModel, because narration never feeds back.
  - It will still distort every narrated trace in a consistent, model-specific direction.
  - A one-sided shift toward composure is the Leins et al. drift, seen cross-sectionally rather than over time.

**7. Software engineering**
- [PAPER] Every belief change is a logged event. The log carries s and e per utterance, so the channel can be audited after the run. Code, prompts and logged run data are released (§1).
- [PAPER] The mechanism parameter γ was fixed by a no-LLM sweep before the language layer ran. The self-report probe is a pure observer.

**8. Small-N, emotion, family, long horizon**
- [PAPER] **Scale, owner interest (e).** N = 20; 5 seeds × 20 rounds per condition; 4 models; two LLM calls per utterance (§4, Limitations). No emotion, family or long horizon.
- [INFERENCE] **Narrator cost at EPModel scale.**
  - Narrating one EPModel run at this call density is about 12 persons × 2,080 ticks, or roughly 25,000 renderings plus the same number of readings.
  - Any narrator acceptance test has to sample ticks and persons, not whole ensembles.
  - Five seeds would not meet EPModel's ensemble standard for anything reported as a result.

**Not transferable / cautions**
- The Beta pseudo-count belief is averaging. It conserves nothing and pulls back toward baseline symmetrically. These are the same objections J4 records against FJ (M6.I.4, M7.A.1). It is not a candidate for EPModel's M9 update.
- The complete graph with one broadcast e per utterance has no witness geometry (compare M4.C.9) and no tie state.
- The closed-form references exist because the update is affine. EPModel's gated, nonlinear policy has no global closed form. References are available only in limits and for sub-mechanisms (BB4).
- One fictional topic, binary stances, a single annotator, and 20 held-out calibration items.
- The "no tipping" result is a theorem about affine updates. It is not an empirical finding about minorities.

### B. Candidate additions to the EPModel spec

**BB1. Narrator recoverability test: an independent reader must recover the engine's ordering of persons from narrated text alone**
- **Proposed requirement.**
  - Before any Phase F narration is shown, the project MUST run a prescribe-then-recover test.
  - Select sampled ticks at which the engine's ordering of persons on a declared quantity is known from the run log (for example acute anxiety, symptom load, or the move's channel).
  - Narrate each person at those ticks.
  - A reader MUST produce a per-person score or ranking from the narrated text only, with no access to the log or the narrator prompt, and MUST be a different model from the narrator, or a human.
  - The test MUST assert rank agreement (Spearman or Kendall) against the log ordering, reported per seed and per tick, not only pooled. The pass floor is `[I]`.
  - Pairs whose engine values fall within one descriptor bin or a declared margin MUST be excluded as uninformative, and the retention fraction MUST be reported per condition.
  - Magnitude recovery MUST NOT be a pass criterion.
  - Recovery MUST be scored against the engine's value, never against the words the narrator was handed. Scoring against the handed words recovers the input by construction.
- **Where it would live.** Phase F, M16.E. It is the narrator counterpart of M11.D.20, which covers only the deterministic readout against the log.
- **Evidence.** §5.1, Eq. 3, App. E.4, Table 3, Fig. 5 (SHOWN in the paper: rank exact on all seeds and models, magnitudes attenuated, ill-conditioned events filtered with retention reported). Transfer to a narrator is INFERENCE.
- **What it changes.** Adds a test. It is the first acceptance test in the spec that a narrator can fail on content, rather than on labelling (M16.E.3). It protects inference validity: a narrated trace that cannot carry the engine's ordering must not be shown as a reading of the run.
- **Cost / risk.** Moderate: two LLM calls per sampled rendering, plus a validated reader. No conflict with M3.D.6 or M16.B, because this is offline and downstream. Owner-facing caution: at the paper's factor-of-2 spacing, rank recovery was easy. Test it at the finer spacing EPModel will actually have.

**BB2. Narration channel transfer curve, per narrator model, with direction separated from scale**
- **Proposed requirement.**
  - For each narrator model and each persona type (at minimum high-expression and low-expression profiles), the project MUST bin the engine value of each narrated quantity and plot the reader's mean score against it.
  - The report MUST classify the deviation as symmetric scale distortion, uniform shift, or one-sided distortion.
  - Any reader calibration, such as temperature scaling, MUST NOT be reported as correcting directional bias.
  - Where feasible, rendering and reading SHOULD be separated by crossing at least two narrators with at least two readers.
- **Where it would live.** Phase F, M16.E, beside X6.
- **Evidence.**
  - Fig. 4a and App. C: every model has its own curve shape (SHOWN).
  - App. C: calibration is symmetric and cannot fix directional bias (ARGUED, with the ECE results SHOWN).
  - The crossed design is the paper's stated future work (INFERENCE).
- **What it changes.** Adds a reporting rule. X6 finds drift over turns under a constant state. BB2 finds distortion across the state range at one time, which X6 cannot see. It also names model identity as a factor with a measurable signature (design lessons §3.3).
- **Cost / risk.** Moderate, since it needs a reader LLM. Low conflict. It partly overlaps X6, X7 (rater separation) and X8 (target is engine state). What is new is the cross-sectional curve and the split between direction and scale.

**BB3 (exploratory LLM line only). If text ever feeds state, run an oracle arm and report the displacement as channel bias**
- **Proposed requirement.** If any exploratory LLM-agent build lets an appraised reading of generated text change agent state (the BCA loop), it MUST:
  - hold that state outside the LLM;
  - log both the sender's latent value and the appraised value per message;
  - run an oracle arm in which the same deterministic update consumes the sender's latent value instead of the appraised one, with seeds matched;
  - report the displacement between the two arms as channel bias, never as a simulation finding.
  - Constants of the update MUST be fixed in the oracle arm before any LLM run.
- **Where it would live.** Exploratory notes (X-line). It has no v2 counterpart, because M16.E.1 and M3.D.6 forbid the loop.
- **Evidence.**
  - Fig. 4b and 4c, Table 2: displaced consensus of 0.01 and 0.15 against 0.50 (SHOWN).
  - App. A: the no-LLM oracle sweep fixed γ first (SHOWN as method).
  - Table 1: the ablation (SHOWN).
- **What it changes.** A protocol for the exploratory line. For v2 it is further evidence for keeping M16.E.1.
- **Cost / risk.** Doubles LLM runs in that line only. None for v2.

**BB4. Limit-case reference tests for the engine harness**
- **Proposed requirement.** Phase B/C MUST include a small suite of limit-case tests. In each, the engine is put in a configuration where a sub-mechanism reduces to a form with a hand- or closed-form-computable answer, and the engine MUST match that answer to numerical tolerance, not to a statistical fit. At minimum:
  - (a) all conductance set to zero: each person's acute anxiety MUST follow its own decay rule exactly;
  - (b) a single dyad driven by `ScriptedSource` with fixed moves: M4.C.1's increment (intensity × conductance / functional_level, with route and fidelity) MUST match the hand computation per event;
  - (c) with J4's FJ-averaging mutant switched on: the synchronous engine MUST reach b* = (I − ΛW)⁻¹(I − Λ)b0, with W taken from conductance. With the anchor weight set to zero, it MUST reach DeGroot consensus.
  - Each test MUST state the limit it exercises. These are tests of implementation, not of theory, and MUST NOT be reported as validation.
- **Where it would live.** M11.D, with a Phase B/C gate.
- **Evidence.**
  - §5.2 and App. E.3: each regime is checked against its closed-form reference, R² computed without refitting (SHOWN).
  - App. E.4: exact inversion to numerical precision as a code check (SHOWN).
  - Transfer is INFERENCE.
- **What it changes.** Adds tests. It answers owner interest (b): EPModel has no analytic sanity checks at parameter limits (no keyword hits). Because M3.D.1 is synchronous, (c) should match the mean-field fixed point exactly, where the paper's sequential engine got only R² 0.93–0.99. (c) therefore exercises the scheduler, the order-free batching of M1.F.8 and conductance weighting in one test with a known answer. It also validates J4's mutant before J4 relies on it.
- **Cost / risk.** Low for (a) and (b). (c) depends on J4 being adopted. The M6 conservation invariants are already exact references, so they are already covered.

**BB5. A criterion that asserts a threshold, jump or bistability must name its nonlinearity and fail under a linearised mutant**
- **Proposed requirement.** Every M11.C criterion whose assertion is a discontinuity, tipping, quantum jump or bistable outcome MUST name the nonlinearity in M4–M7 that produces it. The mutation suite MUST include a mutant that replaces that nonlinearity with its affine counterpart, and the criterion MUST turn red under it.
- **Where it would live.** M11, mutation protocol, reported under M11.4f.
- **Evidence.** App. E.5 proves an affine update cannot tip, and §5.2 and Fig. 2d confirm it (SHOWN). Transfer is INFERENCE.
- **What it changes.** Adds a test. It ties a claimed discontinuity to the mechanism that must produce it.
- **Cost / risk.** Moderate. It partly overlaps C13 (habituation, or a proof of non-bistability), M10.C.4a (quantum-jump condition ablation) and J4. What is new is the general rule and the affine-mutant form. It could be folded into J4.

**Already covered (no write-up)**
- State held outside the LLM, with the LLM kept out of decisions: M3.D.6, M16.E.1, design lessons §3.6 and §7.6.
- A deterministic renderer kept alongside any narrator: M16.C.1, M16.E.3, design lessons §7.6.
- Mechanism constants fixed before the language layer or tests run: C23 and M16.A.7.
- A pure-observer probe that never writes state: M16.B.
- Two renderings of one state agree in ordering: M11.D.20 (deterministic readout only; BB1 extends it to the narrator).
- Full state profile sent on every narrator call, with no numbers in the prompt: X5, which is compatible with the paper's nine-bin descriptor.
- Rater different from the narrator, and fidelity target taken from engine state: X7 and X8.
- Model and temperature as experimental factors: design lessons §3.3.
- Reporting what a filter excludes, as retention by condition: global rule P2/P9 (applied inside BB1).
- Belief separate from truth, with a signed discrepancy: M9.1, M9.5, M16.A.5a and C41.
- Circularity of recovering an input from its own product: M1.A.4j already states it for the estimator. The paper's s-versus-e device is the same principle applied to a channel, and BB1 uses it.
- The FJ form as a rival mechanism: J4. BB4(c) and BB5 depend on it or extend it.

### C. Search terms

- "Belief Engine" configurable inspectable stance dynamics multi-agent LLM deliberation (Yang et al. 2026, arXiv 2605.15343)
- "parameter recovery" agent-based model "prescribe" OR "simulation-based calibration"
- "ID-RAG" identity retrieval-augmented generation persona coherence
- "LLM stance classifier" calibration "temperature scaling" expected calibration error
- "committed minority" OR "zealots" opinion dynamics "tipping point" affine
- "Friedkin-Johnsen" "fixed point" asynchronous OR sequential update convergence
- "verbalized probability" LLM "stance" round-trip OR "text-to-state" fidelity
- "LLM narrator" simulation log faithfulness OR "data-to-text" "ordinal consistency"
- "analytical benchmark" OR "limiting case" verification agent-based model "docking"
- "consensus collapse" LLM populations "model-inherent bias" (Chuang 2024; Taubenfeld 2024)

---

# Part 2: Li et al. 2026 (Mind or Message?) and Kim & Boggust 2026 (What People Almost Did)

## Reading reports: Li et al. 2026 ("Mind or Message?") and Kim & Boggust 2026 ("What People Almost Did")

Both papers were read in full from the pdftotext output. The spec cross-checks below come from grepping `docs/bowen_agent_model_spec_v2.md` at revision 10. The overlap check also covered C1–C42 and X1–X4 in the preprint candidates file, and J1–J4, SV1–SV4, PD1–PD4 and X5–X9 in the 2026-09-27 sweep reports.

---

## Paper 1: Li, Chen, Fung, Rossi & Li 2026, "Mind or Message? Auditing Theory of Mind in Multi-Agent Social Simulation" (arXiv 2609.24146v1)

The paper is 7 pages plus appendices A–C. The model families are not named in the text. Every author's affiliation is given as "DailyAdvance".

### A. Report

**1. Simulation formalism**
- [PAPER] §3.2, App. A. Scenario structure:
  - There are 40 scenarios in two framings: an employment offer and a supplier contract.
  - Each scenario has 5 issues, each with 5 levels.
  - Each side has a private point table that sums to 100, with pairwise-distinct weights.
  - The five issue types appear exactly once each: two integrative (weighted asymmetrically), one distributive (equal weight, opposed), one compatible (same direction for both sides) and one filler.
  - Reservation value is 30% of each side's maximum.
  - The Pareto frontier is computed by full enumeration. The text says "all 55 complete packages", which does not match 5^5, so it may be garbled.
- [PAPER] §3.3. Negotiation is turn-taking. Each turn returns a message, a complete package and an accept flag in a fixed structured format. A deal closes when one side accepts the other's standing proposal.
- [PAPER] App. C. Scenario generation is seeded and deterministic. Runs are resumable and skip records that already exist.

**2. Agent architecture**
- [PAPER] §3.3. Each agent sees the public issue sheet, its own table, its reservation value and the transcript so far. There is no other memory.
- [PAPER] §3.3. Disclosure is either honest or strategic, where the agent is explicitly permitted to misrepresent its priorities.
- [PAPER] §4.2. Stated confidence does not mark errors. Across 2,560 probes it spans only 0.35–0.65. It averages 0.537 when the top priority is right and 0.520 when wrong, a gap of 0.018 [0.013, 0.022].

**3. Interaction and information distortion: the core result**
- [PAPER] §3.4, Fig. 1. The audit freezes every transcript first. It then runs 2,880 probes over nine arms. Each probe holds the evidence about the partner byte-identical and changes one factor:
  - B: the reader's own stake. The mirror arm sets the reader's table equal to the partner's; the inverse arm reverses it.
  - C: tone (assertive or accommodating), with offers stored structurally and re-rendered identically.
  - D: a seniority label (senior or junior).
  - E: the direction of recursion (what the partner believes about the agent).
- [PAPER] Arm B uses an "analyst" reader. Swapping the participant's own table would make the transcript inconsistent with that participant's moves, so an uninvolved reader holding their own stake reads the same transcript instead. An analyst base arm controls for the framing.
- [PAPER] §4.3, Fig. 2b. Swapping only the reader's own table:
  - moves the inferred partner weight vector by 6.5 points [6.0, 7.0];
  - moves the inferred top priority by 15.0 pp [10.9, 19.1];
  - moves rank correlation by 0.119 [0.096, 0.144].
- [PAPER] §4.3. Accuracy is 66.6% in the mirror arm against 51.6% in the inverse arm. Projecting oneself onto the partner gives the right answer in the mirror arm "without any inference having occurred".
- [PAPER] §4.4. The surface cues move the reading much less:
  - Tone moves top-priority accuracy by 5.3 pp [1.9, 9.1].
  - The label moves it by 0.0 pp [-2.8, 2.8].
  - The senior label does raise the estimated reservation value, by 1.10 points [0.78, 1.44], although the label carries no information about it.
  - The authors' summary: where evidence underdetermines the answer, "the agent falls back on itself".
- [PAPER] §4.5, arm E. An agent predicts its partner's stated belief about it 72.5% of the time, while that belief is itself correct only 51.2% of the time. Scored against the agent's own true priority, the hit rate is 68.8%. Their reading is that agents keep a shared representation of the conversation, not an independent model of the partner's private state.

**5. Calibration and validation**
- [PAPER] §4.1. Fluent but inefficient:
  - Agreement in 96.2% of 160 dyads, in a mean of 5.94 turns, with 0 protocol failures.
  - Only 0.7% of deals are Pareto-optimal.
  - 20.5% of available joint value is destroyed.
  - The compatible issue is missed in 76.6% of deals.
  - Honest vs strategic disclosure: 19.5% vs 21.6% value loss, a difference of 2.1 pp [-0.8, 4.8].
- [PAPER] §4.1. The exact baseline is a random package drawn from the set both sides would accept. It is Pareto-optimal 1.1% of the time, destroys 24.1% of joint value and captures the compatible optimum 24.8% of the time. Agent minus baseline:
  - joint value destroyed: -3.7 [-5.1, -2.4];
  - Pareto: -0.4 [-1.1, 0.9];
  - compatible issue: -1.5 [-8.0, 5.6].

  So on the two measures that need a model of the partner, the agents are indistinguishable from random.
- [PAPER] §4.2, Table 1. Base partner model:
  - top priority 51.2% against a 20% chance level;
  - Kendall τ 0.491;
  - compatible issue identified 36.9%;
  - reservation error 22.9 points.

  Participant and analyst arms do not differ: -2.5 pp [-5.3, 0.3].
- [PAPER] §3.5. Intervals are 95% bootstrap (2,000 resamples at dyad level, paired). They are not corrected for multiple comparisons.
- [PAPER] §4.6. Splitting dyads at the median partner-model accuracy: 20.0% vs 21.3% value destroyed, and 26.7% vs 18.8% compatible issue captured, over 154 dyads. The authors state this is descriptive, not causal.
- [PAPER] §6. Projection appears in all 4 cells of model × seat, between 13.8 and 17.5 points. All probes were answered by a single reader model.

**6. Failure modes and limitations**
- [PAPER] §5, §6:
  - The partner model is elicited by verbal report and may diverge from what drove the moves.
  - The scenarios are synthetic and regular.
  - The study covers dyads only and short horizons.
  - Seniority was used instead of demographic labels by design, so the label null does not generalise to demographic labels.
- [PAPER] §5. The authors recommend reporting partner-modelling accuracy separately for aligned and opposed preference pairs, because similar characters inflate apparent theory-of-mind competence.
- [PAPER] §5. The failure is silent. Deals are nearly always reached, so the value loss is invisible to anyone reading the transcript or collecting a satisfaction rating afterwards.

**7. Software-engineering lessons**
- [PAPER] §3.2, App. A. Structural assertions are run over 200 freshly generated scenarios before the study starts, and any failure stops it before any agent is called.
- [PAPER] App. B. Tone rewrites are checked by comparing the multiset of numerals with the original. The result is recorded per probe, and the gate fails above a fixed mismatch threshold.
- [PAPER] App. C. Every reported number is written by the analysis script into a macro file, so no number in the text is typed by hand. Two verification gates are released: one over source and frozen records, one over the compiled manuscript.

**8. Small N, emotion, family**
- [PAPER] The unit is the dyad. There is no emotion variable. §4.1 attributes the incompatibility bias to a representational gap, noting the agents "never had to overcome a competitive emotion".

**The owner's three questions**

(a) Egocentric projection. [INFERENCE]
- **Li's notion.** Under evidence that underdetermines the answer, the perceiver's own preferences are imputed to the other. It is a static cognitive default measured at one moment ("curse of knowledge", egocentric anchoring). It has no anxiety term, no function for the system and no effect on the target.
- **Bowen's family projection process** (M7.E.1c–d, M9.7). It is:
  - anxiety-driven;
  - aimed at a selected child;
  - a false definition that "makes the false conception come true" (M7.E.1d);
  - run "through descriptions" (FE11.3).

  It changes the target over years. It is a process of the relationship system, not an inference error. **The two should not be conflated.**
- **The closer spec analogue is at the appraisal level.**
  - M4.C.6's corpus instance: the tense expression he saw on her face "reflected his own" expression. That is the perceiver's own output read as the other's state, which is Li's structure.
  - M4.C.5's perception-side readout ("hearing her as being critical").
  - M9.7: chronic anxiety runs on "what might be".
- **What transfers is the test design, not the mechanism.** M9.8 names only delivered events, after per-hop fidelity, as writers of a person's belief about a tie. I found no requirement stating whether the receiver's own state enters that belief write. Li's arm B is the design that would establish whatever rule is chosen. See MM1.

(b) Audit method. [INFERENCE]
- EPModel already works this way. Mutation arms, identical seeds and M11.C.36 (two arms differing in one belief with the true state identical) are one-factor probes on fixed ground truth. M4.C.7 even describes itself as "content identical by construction".
- The engine's readouts are deterministic functions of the log. Probing them adds little beyond the existing M11.D.20 and C21-type checks.
- The method does add something in two places:
  - the belief-write path (MM1, MM2);
  - the optional Phase F narrator, where the frozen log is the fixed evidence and engine state is ground truth known by construction (MM3).

(c) Fluent agreement with low joint value. [INFERENCE]
- This matches the Bowen readout trap in which a calm dyad is paid for in another sink.
- **Already covered** by J1 (report the displacement term beside every low-conflict readout), M11.F.6 (two-sidedness) and M11.G.1–G.2 (sink readouts).
- Li's random-acceptable-package baseline corresponds to the design-lessons §7.5 null model and C34's control arms. Already covered.

**Not transferable / cautions**
- Every number is about LLM text agents in negotiation. None may become an EPModel constant or bound.
- The partner model is a verbal report (§6).
- The results come from a single reader model.
- Intervals are not multiplicity-corrected.
- The paper supports M3.D.6: fluent transcripts hid a representational gap.

### B. Candidate additions

**MM1. Frozen-inbox own-state probe for belief writes and appraisal**
- **Proposed requirement.** M11 **MUST** include a probe pair in which a `ScriptedSource` delivers byte-identical events to one receiver in two arms that differ only in the receiver's own state (acute or chronic anxiety; `systems_perspective`). The pair **MUST** report the difference in the belief the receiver writes about the sender and the tie (M9.8), and in the M4.C.5 perception-side readout.
  - The direction the theory predicts: higher anxiety or lower `systems_perspective` gives a more threatening reading of identical events (M4.C.5, M4.C.6, M9.7).
  - A third arm **MUST** change only a role label on the sender, and **MUST** show no difference (M1.A.14b).
- **Where it would live.** M11.C, beside M11.C.36. The owner must also decide whether M9.8's write rule reads receiver state at all.
- **Evidence.** Li §3.4 arm B, §4.3 (the probe design is SHOWN as method). The direction comes from the spec's own corpus lines (M4.C.5, M4.C.6). The transfer is INFERENCE.
- **What it changes.** It adds a test and exposes a gap in the mechanism. If belief writes do not read receiver state, the anxiety-distorts-perception claim lives only in appraisal and never in the belief store. That bears on theory fidelity. The label-null arm tests M1.A.14b, which I found stated as a rule with no test.
- **Cost / risk.** Low as a test. If a receiver-state term is added to belief writes, its magnitude is `[I]`. It must not be framed as the family projection process, and M9.2 still binds: the loaded reading is not assumed to be the false one. No conflict with M3.D or M16.B.

**MM2. Stratify the belief–truth discrepancy by perceiver–target similarity**
- **Proposed requirement.** M17.B.3's signed belief–truth discrepancy **SHOULD** be reported separately for perceiver–target pairs that are similar and dissimilar on the quantity the belief concerns. An egocentric write rule looks accurate when the two are alike.
- **Where it would live.** Phase E (M17.B.3), using M16.A.5a.
- **Evidence.** Li §4.3 (mirror 66.6% vs inverse 51.6%) and §5's recommendation to report aligned and opposed pairs separately (SHOWN). The transfer is INFERENCE.
- **What it changes.** A reporting rule. Spouses are matched on `basic_level` (M2.A.0c), so a projection-type belief error would be masked between spouses and would show up between parent and child. Pooling the two hides exactly the case the theory is about. Improves inference validity.
- **Cost / risk.** Trivial, observer-side. The similarity cut is `[I]`.

**MM3. Frozen-log single-factor probes for the Phase F narrator (X-line)**
- **Proposed requirement.** Before any narrator is used, the project **MUST** run it on a frozen log under paired arms that each change exactly one input field:
  - one person's anxiety value;
  - one role or identity label;
  - the rendering register.

  Claims extracted from the narration **MUST** be scored against engine state. The label arm and the register arm **MUST** be null on state claims. The state arm **MUST** move in the declared direction. Every numeral in the input profile **MUST** survive into the output, and a mismatch rate above a declared threshold fails the gate.
- **Where it would live.** Phase F, M16.E.3.
- **Evidence.** Li §3.4 and App. B (numeral-multiset gate; tone rewrite with offers fixed), SHOWN as method.
- **What it changes.** It adds an acceptance gate for the narrator.
- **Overlap.** Partly covered by X6 (constant-state drift), X7 (rater separation) and X8 (fidelity target is engine state). The new parts are the one-factor arms, the label-null arm and the numeral check.
- **Cost / risk.** Moderate: it needs an extractor, which cannot be an LLM judge scoring itself (X7). Consistent with M3.D.6 and M16.E.

**MM4. A pre-run structural gate over draws from D0**
- **Proposed requirement.** Before any ensemble, the Phase E runner **MUST** draw a declared number K of initial states from `D0` (M17.C.1) and assert its declared structural properties. Examples: spouses matched within the declared band; pole independent of sex; every person's legal set non-empty at tick 0 (M4.D.1e). Any failure **MUST** stop the run before any tick executes.
- **Where it would live.** Phase E (M17.C.1).
- **Evidence.** Li §3.2 and App. A: assertions over 200 fresh scenarios, and failure stops the study (SHOWN as practice).
- **What it changes.** A validity gate, since a malformed initial state is otherwise seen only as odd results.
- **Overlap.** Partly covered by M11.D.17 (static register checks) and M17.C.3 (structural totals verified per arm). What is new is sampling D0 itself before running.
- **Cost / risk.** Low.

**Already covered (one line each)**
- Surface harmony vs hidden cost: already covered by J1 and M11.F.6.
- Exact no-mechanism baseline: already covered by design lessons §7.5 and C34 (M17.D control arms).
- Numbers generated by script, not transcribed: already covered by Prasad's single-script regeneration (candidates file §9) and M16.
- Seeded, resumable runs: already covered by M3.D.4–5.
- Stated confidence as uninformative: LLM line only; X1–X4 and design lessons §3.4 already cover it.

### C. Search terms
- "egocentric projection" theory of mind LLM agents counterfactual probe
- "curse of knowledge" perspective taking agent-based model
- "frozen transcript" counterfactual audit social simulation
- "incompatibility bias" fixed-pie perception simulation
- perception distorted by anxiety computational model interpersonal appraisal
- "partner model" latent ground truth multi-agent evaluation
- attribution of own state to other hostile attribution bias model
- second-order belief "what the other thinks of me" simulation
- label invariance counterfactual fairness agent simulation

---

## Paper 2: Kim & Boggust 2026, "What People Almost Did: Evaluating LLM Social Simulations Beyond Behavioral Fit" (arXiv 2609.20055v1)

This is a short position paper of about 4 pages of text. **It reports no experiments, no data and no measurements.** Its only figure-like number is a background Pew statistic about teens' Instagram use.

### A. Report

**2. Agent architecture: reasoning traces**
- [PAPER] §1. A reasoning trace may be:
  - generated before the action;
  - stored as structured state;
  - elicited afterwards;
  - reconstructed from logs.

  Whichever way it is obtained, it is "a hypothesis about the process".
- [PAPER] §3. The unit of analysis is the proposed scenario–reasoning–action triple. The paper says current benchmarks keep actions and discard the traces.

**What "what people almost did" means operationally** (the owner's question)
- [PAPER] The phrase appears only in the Conclusion, as rhetoric: simulations should preserve what people "considered, feared, and almost did". **The paper does not operationalise it** as near-miss alternatives, choice distributions, margins or deliberation logs. Its operational object is the reasoning trace inside the triple.
- [PAPER] §2 gives a taxonomy of ways behaviour hides process:
  - **Missing actions.** Preference falsification (Kuran): someone who weighed protesting and stayed home looks the same as someone who never considered it. The running example is a teen who drafts a post and deletes it.
  - **Ambiguous actions.** Equifinality (different reasons, same act; Janis's groupthink) and multifinality (one reason, different acts: privacy fear drives both oversharing and withdrawal).
  - **Shifting composition.** The aggregate rate is stable while who acts, and why, changes.
- [PAPER] §3. Representational adequacy is defined relative to a claim, a population and a scenario. It is distinguished from:
  - faithfulness, i.e. whether a trace reflects the model's computation (traces can be post-hoc rationalisations: Turpin 2023, Lanham 2023);
  - representational alignment.
- [PAPER] §3. Ground-truth reasoning should be recognisable to the population, consistent with the domain literature, and distributed correctly. The example of failure: a simulation that reproduces the silence rate but puts constrained silence on the wrong teens.
- [PAPER] §4. Recommendations: state the claim type (behavioural vs process); match evidence to the claim; preserve the triple. It also warns that process knowledge can be misused and calls for governance.
- [PAPER] §5. Measurement is posed as an open problem. The one concrete method suggested: edit individual rationales in a trace and check whether the action changes. Probing, causal mediation and interchange interventions are listed as options.

**5. Validation**
- [PAPER] None performed. The paper is ARGUED throughout.

**Not transferable / cautions**
- The paper is about LLM traces. EPModel's rule-based engine has no trace-faithfulness problem, because its "reasoning" is the computation itself: M16.A.3 records the propensity vector and the draw.
- The adequacy ground truth (population self-report) does not exist for EPModel, and M11.F.9 forbids fitting to it.
- What transfers is the taxonomy (missing actions, equifinality, shifting composition) as a checklist for readouts.

**Owner's question: should M4 move selection be logged and evaluated on the alternatives and their margins?** [INFERENCE, based on the spec text]
- **Logging of alternatives is largely covered already**:
  - M16.A.3 requires the propensity vector over the repertoire and the resolving draw.
  - M16.A.3b logs the legal set.
  - M16.A.3c logs fallback and tie-break provenance.
  - M16.A.10 logs threshold margins.
  - M4.D.1d already uses "an entropy or margin term over the propensity distribution".
- **Three things are not covered:**
  1. The identity of the move a `WITHHOLD` withheld. M4.D.1b says it is "computed, detected, and not emitted". The spec lists "a `WITHHOLD` count in readouts" among design lessons "noted, not specified".
  2. Tests that assert on the propensity distribution rather than on sampled moves.
  3. Decomposing a move count by the process that produced it.

### B. Candidate additions

**AD1. The withheld move is a logged outcome with its identity, and a readout**
- **Proposed requirement.** Every `WITHHOLD` selection record **MUST** carry the move type the automatic channel computed and did not emit (M4.D.1b), with that move's propensity. Readouts **MUST** report, per person and window, withheld counts by move type beside emitted counts.
- **Where it would live.** M16.A.3 (a new sub-item) and M11.G.
- **Evidence.** Kim §2 "missing actions" (ARGUED). The corpus instance already quoted at M4.D.1b ("I caught myself and stopped"). The rest is INFERENCE.
- **What it changes.** A logging and reporting rule. A person who withheld PURSUE forty times looks identical, in emitted moves, to one who never had the urge, and the M4.D.1c distinction (WITHHOLD vs I-POSITION trajectory) cannot be read without this. Improves theory fidelity and inference validity.
- **Cost / risk.** Trivial, observer-side (M16.B.3). It must be labelled as not a maturity measure (same as M16.A.9).

**AD2. Complexity-ordering criteria asserted on the propensity distribution, not only on sampled moves**
- **Proposed requirement.** The criterion for M4.D.3a (rising anxiety slides selection down the complexity ordering) **MUST** assert a direction on the propensity-weighted mean rank along the ordering, computed from the M16.A.3 record. It **MAY** also assert on selected-move counts, but not only on those.
- **Where it would live.** M11.C (the criterion for M4.D.3a and M4.D.3).
- **Evidence.** Kim §2: behaviour underdetermines process; a sampled act is a lossy record (ARGUED). Transfer is INFERENCE.
- **What it changes.** It adds a test. A shift in probability mass can happen without changing the modal or sampled move. Asserting on the logged distribution detects the mechanism at far lower seed variance, and a mutant that flattens the ordering should turn it red. Improves inference validity.
- **Cost / risk.** Low; the data is already required. The softmax temperature is `[I]` and decides how visible the shift is in sampled moves. That is a further reason to test on the distribution.

**AD3. Move counts decomposed by channel, act form and dominant score term (equifinality and shifting composition)**
- **Proposed requirement.** Where the propensity score is separable into the M4.D.2 terms (acute anxiety, `functional_level`, tie state, triangle position, repertoire), the selection record **SHOULD** carry each term's contribution to the chosen outcome's score. Any readout or criterion reporting a move count **MUST** also report its composition:
  - by channel mixture (M1.F.1a);
  - for `I-POSITION`, by executed form (genuine vs M5.F.4 assertion form);
  - where the terms are logged, by dominant term.

  At least one M11 test **SHOULD** construct two arms with equal total `I-POSITION` counts and different composition, and assert that the readout distinguishes them.
- **Where it would live.** M16.A.3, M11.G and M11.
- **Evidence.** Kim §2 "ambiguous actions" and "shifting composition" (ARGUED). Transfer is INFERENCE.
- **What it changes.** A reporting rule and a test. Counterfeit I-positions (M5.F) are the Bowen instance of equifinality: same act label, opposite process. A count readout without composition would merge them.
- **Overlap.** Partly covered by M16.A.9 (entropy per channel), M5.F.2b and M11.C.19 (counterfeit axis), and M17.B.5 (per-channel transition column). The per-term contribution and the composition rule for count readouts are new.
- **Cost / risk.** Low if the score is additive. If it is not separable, use a per-term ablation difference, which costs more. Observer-side.

**Already covered (one line each)**
- Preserving the scenario–reasoning–action triple: already covered by M16.A.2–A.4 (event, selection rationale, effects beside cause).
- Editing one rationale and checking whether the action changes: already covered by the M11 mutation protocol, M11.C.36 and C17 (matched-magnitude severing mutants).
- A narrated reason is not the computation (faithfulness): already covered by M16.E.2–E.3 (narration traceable to log lines) and X5–X8. One sharpening worth one line in M16.E.3: a narrated reason for a move must name only terms present in that move's selection record.
- "State the claim type": already covered by M11.F and M17.G.1's claim grade.

### C. Search terms
- "preference falsification" agent-based model withheld action
- equifinality multifinality agent-based simulation validation
- "representational adequacy" LLM simulation
- near-miss decisions choice margin logging agent model
- softmax policy "propensity distribution" test versus sampled action
- inhibited response "urge to act" computational model restraint
- "process validity" versus "outcome validity" agent-based model
- chain-of-thought faithfulness post-hoc rationalization social simulation
- composition shift stable aggregate behaviour simulation readout

---

# Part 3: Tareaf 2026 (Diverse Minds, Divided Networks)

## Reader report: Tareaf 2026, "Diverse Minds, Divided Networks?" (arXiv 2609.12444v1)

I read the full text: the main article (pp. 1–41) and Additional file 1 (S1–S21). For the long contrast table (S17) and the prompt appendix (S14) I skimmed, spot-checked rows and read all the prose in them. Figures survive only as captions and as stray numbers. It is a single-author preprint and has not been peer reviewed. §6.2 calls it a follow-up to prior work "in this journal", which suggests it was submitted to EPJ Data Science.

### A. Report

**What the paper is.** TraitMix treats the Big Five composition of an LLM-agent society as the controlled independent variable. Traits are drawn as θᵢ ∼ TN₅(μ, Σ), a five-dimensional normal truncated to [0,1]⁵. The mean μ sets trait **level** and Σ sets trait **heterogeneity** (§3.1, Def. 2). Each society has 100 agents on a Barabási–Albert graph (m = 3), with a recommender feed and follow/unfollow actions. Two outcome families are measured in the same runs: polarization (six measures) and collective intelligence (estimation and hidden-profile tasks). There are 991 runs across six models, six topics and 5–8 seeds (Table 2, S13).

**1. Simulation formalism**
- [PAPER] Each round: every agent activates independently with probability ρ = 0.4, gets a ranked feed of f = 10 posts (Eq. 1: opinion proximity + popularity/recency, with a boost for followed authors), and picks exactly one of {post, reply, like, follow, unfollow, pass}. Memory holds the last k = 10 own actions. Private probes run every P = 5 rounds, T = 30 rounds in all (§3.2, §4.2).
- [PAPER] Opinions are never inferred from posts. They are elicited "privately, outside the platform" on a −3 to +3 scale, so belief is measured apart from willingness to post (§3.2).
- [PAPER] Trait sampling is deterministic given the condition and seed, so every run's realised trait matrix can be rebuilt without re-running it (§3.5, S2). Seeds are matched across conditions and contrasts are paired (§4.5).
- [INFERENCE] The split between belief and expression matches EPModel's belief layer and its counterfeit-position trap. It is already in the spec (M9, M11.F.6), so it adds nothing new.

**2. Agent architecture / heterogeneity**
- [PAPER] Traits enter the prompt as graded text descriptions at nine levels (§3.3, S14).
- [PAPER] **Level and spread are separate factors, but truncation couples them.** A distribution centred at 0.8 with σ = 0.15 is clipped, so its realised spread is smaller. For that reason the realised moments are reported for every run and used as a covariate (§3.1, §3.5).
- [PAPER] Realised dispersion predicts opinion variance within a condition (b = 6.67, p < 10⁻⁴). The size is small: about 0.07 predicted against an observed 0.84, "about one twelfth" of the largest effect (§5.1, Fig. 4).

**3. Interaction structure, and owner interests (a) and (c)**
- [PAPER] Polarization is split into **dispersion** (variance, extremity, bimodality) and **segregation** (assortativity, E–I echo closure, cross-cutting reply rate) (§3.4.1).
- [PAPER] Heterogeneity pushes the two in opposite directions. In the E2 heterogeneity experiment, realised trait SD correlates with opinion variance at r = +0.966, with extremity at +0.870, with cross-cutting at +0.795, and with echo closure at −0.810 (Table after §5.4, Fig. 8).
- [PAPER] The homogeneous society (σ = 0.05) has opinion variance −0.772, cross-cutting −0.123 and closure +0.313 against baseline (Table 5). The paper's phrase is "not a moderate society but a consensual echo chamber" (§5.4).
- [PAPER] High Agreeableness gives variance −0.465, extremity +0.216 and cross-cutting −0.113. Agreeable societies "converge on a more extreme position" and stop replying across the divide (§5.2, Table 4). High Agreeableness ends at a 98:2 split, high Neuroticism at 51:49 (Fig. 2).
- [PAPER] **Mechanical dependence is admitted and tested.** When variance collapses, few opposing pairs remain, so closure rises by construction. The measures correlate (r = +0.74 variance–cross-cutting; −0.76 variance–closure). Regressing each segregation measure on variance and adding condition still adds substantial R² (0.60 cross-cutting, 0.63 closure), so the measures are "related but not redundant" (§5.2). Closure is "never interpreted without variance alongside it" (§3.4.1, §6.4).
- [PAPER] **Rates are reported with their denominators** (§5.2, S11). A near-zero cross-cutting rate could mean agents declined to reply, or that few opposing pairs were left. Under high Agreeableness, 4,069 opposing pairs remained and 598 replies were sent, but only 25 crossed the divide (baseline: 72 of 435). The opportunity existed and was not taken. In the homogeneous condition the opposing-pair pool thins to 14.5% of possible pairs, and the paper marks that rate as resting on a smaller base.
- [PAPER] Two measures failed and are reported anyway. Assortativity "discriminates nothing" (SD 0.038, no condition moves it by more than 0.024). Bimodality is not independent of the others (§5.7).
- [PAPER, caution the paper does not stress] In Qwen2.5-14B the homogeneous condition moves opinion variance (−0.409) but not closure (+0.035, n.s.) or cross-cutting (−0.057, n.s. after Holm) (S17). At n = 5 the "echo chamber" half of the headline therefore rests mainly on the primary model.
- [INFERENCE, (a)] Big Five heterogeneity is not differentiation. Agreeableness is not togetherness and Neuroticism is not chronic anxiety. The structural point does carry over: a low value on one face of a construct (low dispersion) can come with a high value on another face (closure), so a low reading is not evidence of health. EPModel already states this in textual form:
  - M11.3: cohesion from togetherness and cohesion from individuality are indistinguishable when the system is calm.
  - M11.F.6 lists "an apparently fine marriage" and "reported closeness" among its traps.
  - M5.C.1a names the "peace-agree" family, "whose differences are obliterated quickly". That is the Bowen analogue of the consensual echo chamber, and the reactive family is the analogue of the neurotic two-camp split.
  - J1 already pairs low conflict with a displacement term.
- [INFERENCE, (a), spouse pairing] The analogy to M2.A.0c/e runs the wrong way. In TraitMix, homogeneity is an input the experimenter sets. In EPModel, spouses match on `basic_level` by theory, to ±1 point, at every level. Within-couple spread is fixed by construction, and sibling spread is an **output** of projection (M2.A.2). Setting sibling spread as an input would pre-empt the mechanism. Only founding-generation, between-couple spread could be a legitimate input factor. M17.D.1(c)'s homogeneous-family arm is correctly framed as a diagnostic null, not a finding.

**4. Initialization, and owner interest (e)**
- [PAPER] Composition is the experimental variable, and every other aspect is held fixed (Def. 2). The design has five parts (§4.4):
  - E1: one trait at a time at 0.2 or 0.8, the others at 0.5.
  - E2: σ ∈ {0.05, 0.15, 0.25}, plus a condition built from human norms.
  - E3: a 3 × 3 grid crossing Openness and Agreeableness.
  - E5: a robustness audit.
  - Replications and ablations.
- [PAPER] Statistical power "resides in the number of conditions and seeds, not in the number of agents", because each society is one observation (§4.2).
- [PAPER] Scale ablation at N = 200, three seeds: somewhat more dispersed and less closed, and "no conclusion depends on scale" (§5.9).
- [PAPER] The human-norm condition (norms from 320,128 respondents) was the most insular in the study, and the paper says it inherits that sample's self-selection (§5.4, §6.4).
- [PAPER] Unplanned internal replication: three conditions had identical compositions and independent seeds, and their means differ by at most 0.052 in opinion variance (§5.1).

**5. Calibration and validation, and owner interests (b) and (d)**

*The induction gate (§3.3, §5.1, §5.10, Table 8, S7–S8)*
- [PAPER] The IPIP-NEO-120 is given to every persona configuration, separately for each model, before any results are read.
- [PAPER] The criteria are rank (Spearman ρ ≥ 0.60) **and** magnitude: the gap between the highest and lowest targeted levels must be at least 1 point on the 5-point scale, for each manipulated trait.
- [PAPER] The magnitude criterion was added **after** a model ordered its configurations at ρ = 0.96 but separated the extremes by only 0.71 points, and its behaviour was flat. The paper says openly that this criterion was not pre-specified, and defends it on three grounds: it is independent of the outcomes, it is applied to every model, and it agrees with a behavioural quantity (the range of condition means was 0.09 against 0.45–2.06 in admitted models).
- [PAPER] Two models were excluded and both are reported. Qwen2.5-3B failed on rank. Llama-3.2-3B failed on magnitude, and excluding it removes a surface that would otherwise contradict the interaction (§5.10, S7).
- [PAPER, §6.4] The gate measures questionnaire validity only. A model can pass by "reading its own instructions". The behavioural trait scorer was a lexical-marker proxy, "close to circular", so expressed-trait drift is not reported.

*Circularity ablations (§5.1, S10)*
- [PAPER] **Probe anchors.** The opinion probe told the agent its previous answer and the feed average, which is influence built into the instrument. Removing the anchors gave ρ = 0.729 against the published effects, with 29 of 36 signs kept. Extremity rises, three low-magnitude effects flip, and the headline effects hold.
- [PAPER] **Recommender proximity term.** The feed ranked posts by latent-opinion distance, the same variable the outcomes are computed on. Removing that term gave ρ = 0.975, with 33 of 36 signs kept.
- [PAPER] A variant using the most recently **posted** stance, which a real platform could observe, gave cross-cutting −0.113, the same as published (S17).
- [PAPER, from S16, not remarked on by the paper] The ablations move baseline **levels** a lot even where the effects hold. Baseline opinion variance is 0.809 without anchors against 1.263. Baseline cross-cutting is 0.306 without the proximity term against 0.165.
- [INFERENCE] The probe-anchor case is the LLM form of an impure observer, which M16.B already forbids. The proximity term reading latent opinion is a "god-view" mechanism, which C9 already covers. The finding that levels move under ablation while effects survive supports EPModel's rule of differencing two arms rather than reading levels.

*Other controls*
- [PAPER] **Neutral filler topic** ("Pineapple belongs on pizza"). In the primary model, filler variance is unrelated to contested-topic extremity (r = −0.061). In Qwen2.5-14B they correlate at +0.367, so extremity there measures response style. This explains why extremity fails to replicate (§6.3).
- [PAPER] **Six-topic replication** (§5.3). Five named sign reversals, all where the two-topic estimate was noisy. Immigration has low "purchase": between-condition spread divided by seed noise is 1.6, against a mean of 3.4 for the other topics. Composition explains 53.2% of the variance in opinion variance and topic 12.3%. A two-topic asymmetry the paper had earlier reported (raising a trait matters, lowering does not) was withdrawn.

*Interaction, owner interest (b) (§5.5, Table 6, §5.10)*
- [PAPER] The crossover: when μA = 0.2, raising Openness moves variance from 0.53 to 1.38. When μA = 0.8, it moves variance from 0.97 to 0.44. Openness "has no interpretable main effect".
- [PAPER] The interaction is fitted on the **nine cell means** (b = −3.86, p = 0.030), because run-level p-values would treat within-cell noise as replication.
- [PAPER] In the primary model the estimate depends on a few cells: two cells have Cook's D > 1, and p < 0.05 survives in only 3 of 9 leave-one-out fits.
- [PAPER] Replication carries the result: Qwen2.5-14B (b = −3.25, 9 of 9 leave-one-out fits) and Qwen2.5-32B (b = −0.98, 7 of 9).
- [PAPER] Qwen2.5-7B is **additive** (b = −0.02), even though it responds most strongly of all (range 2.06). It also lacks the heterogeneity effect: r = −0.33 against +0.97 in the primary model. The traits' joint effects fail together in that model, while their one-at-a-time effects hold.
- [PAPER] Trait main effects replicate near chance (sign agreement 6/10, 4/10). The paper's words: "What transfers is the moderation, not the magnitudes and not the main effects" (§5.10).

*Statistics (§4.5)*
- [PAPER] Paired t-tests and mixed models with a seed random intercept. Holm correction within each experiment family × metric. dz, Hedges' g and Cliff's δ for every contrast, with 10,000-resample bootstrap intervals.
- [PAPER] A pre-declared seed extension from 5 to 8, applied to **every** condition in a family, not just the flagged ones.
- [PAPER] The Wilcoxon floor is stated: the two-sided p cannot fall below 2/2ⁿ.
- [PAPER] Baselines and standardisation are within model. The robustness audit (3 seeds) is analysed by sign stability only.

**6. Failure modes and negative results**
- [PAPER] The pre-to-post accuracy change is a null (F = 0.35, p = 0.995), because private estimates barely move (§5.6).
- [PAPER] The hidden-profile task is at a floor (0.12 in the primary model, about 0 in Qwen2.5-14B). There are 33 undefined contrasts, left empty rather than imputed (S17, S20).
- [PAPER] Collective-intelligence effects are unstable under perturbation (3–7 of 9 cells keep their sign) and are labelled exploratory (Table 7).
- [PAPER] Of four polarization–accuracy associations, three are withdrawn after partialling on the diversity-prediction identity. Cross-cutting survives (+0.409 falls to +0.240), but within-condition r = +0.107 is not significant (§5.8).
- [PAPER] Hidden-profile non-association is shown by an equivalence test (TOST, |r| < 0.20).

**Internal inconsistencies I found**
- The S12 ledger gives the probe-anchor ablation as ρ = +0.620 and the recommender ablation as +0.979. The main text gives 0.729 and 0.975.
- §6.4 cites "the attenuation we observe with scale". §5.10 says there is no systematic relationship with scale and that an earlier attenuation claim was not supported.
- S3 still states that lowered traits produced no significant contrasts, an asymmetry §5.2 withdraws.
- S16 contains an `e3mistral7` row (one cell, empty metrics) that is missing from the run accounting in Table 2 and S13.
- S21 claims agents are never told they are in an experiment, but every probe says "This is a private research survey". S21 also points to prompts "in Section S2"; they are in S14.
- Small dz and effect mismatches between text and ledger: −2.37 vs −2.33; −3.31 vs −3.24; homogeneous variance −0.783 in the §5.4 text vs −0.772 in Table 5.

**7. Software engineering**
- [PAPER] Every table and figure is produced by a pipeline that reads the run-level file, and "No value was transcribed by hand". A results ledger (S12) gives the file, column and aggregation for each quoted number (§4.6).
- [PAPER] Configurations that need normative statistics refuse to run until those statistics are supplied with their source, which keeps placeholders out of runs.
- [PAPER] The model client aborts a run when the share of failed calls passes a threshold, rather than substituting null actions (§4.6).
- [PAPER] Models are served locally to fix their versions. The stack is pinned (vLLM 0.8.5.post1). One run takes about 3 minutes on one RTX 4090 (§4.1–4.2).

**8. Small-N, family, emotional state, long horizon**
- [PAPER] Nothing on families, dyads, triads or horizons longer than 30 rounds. N is 100–200 (§6.4 Scope).

**Not transferable / cautions**
- The Big Five constructs, the recommender, the collective-intelligence tasks and the human-norm condition have no EPModel analogue.
- The polarization measures are population statistics over 100 agents. With 11 family members, variance and E–I indices over ties are not meaningful.
- The headline "echo chamber" result is weaker outside the primary model.
- The interaction estimate in the primary model depends on a few cells.
- The magnitude gate was added after the data were seen, although it is well defended.

### B. Candidate additions to the EPModel spec

**Already covered (one line each)**
- Paired seeds across arms: M3.D.4a / M17.B.1.
- Homogeneous-family null arm: M17.D.1(c).
- Sign stability across design perturbations, with an unaudited column: M17.G.1.
- Two reference configurations: M17.E.4.
- Pairwise additivity residual for pairs the theory names: M17.E.5.
- Floor on the measure at the baseline: M11.4d.
- Equivalence-bound nulls: M11.4a.
- Holm-style multiplicity and paired rank statistic: M11.4e / M17.A.3.
- Seed extension in blocks: M17.A.1.
- Observer that injects influence (probe anchors): M16.B.
- Mechanism reading true latent state (proximity term): C9.
- Low-conflict readout paired with displacement: J1.
- "Calm cohesion is ambiguous": M11.3, M11.F.6.
- Peace-agree vs reactive family: M5.C.1a.
- Level-over-difference reporting: M0.4.
- Constants frozen before tests: M10.B.4.
- Configuration fails loudly: M10.B.2.

**TM1. Realised initial moments logged per seed; level and spread declared separately in D0**
- **Proposed requirement:** `D0` (M17.C.1) MUST declare the level and the spread of each person attribute it varies (`basic_level`, `chronic_anxiety`) as separate parameters. The runner MUST log each seed's **realised** initial moments: family mean and SD of `basic_level` by generation, and the within-couple gap. Any arm contrast that varies level MUST report the realised spread in both arms and MUST enter it as a covariate. Spread MAY be varied only across founding couples, never within a couple (M2.A.0e) or across siblings born in the run (an M2.A.2 output).
- **Where:** Phase E (M17.C.1), M16.A.
- **Evidence:** §3.1, §3.5, §5.1, S2. Truncation couples level and spread, and realised spread explained about 1/12 of the largest effect (SHOWN). The EPModel transfer is INFERENCE: bounds on 0–100, the ±1 spouse tolerance and the declared correlations will also make realised spread differ from declared spread.
- **What it changes:** Adds a reporting rule. Improves inference validity, and fidelity through the no-sibling-spread-input clause.
- **Cost / risk:** Low; observer-side. No rule conflict. PARTLY covered by M17.C.1, which has a distribution and correlations but no realised-moment log and no level/spread separation.

**TM2. Manipulation-check gate for any arm that assigns a person attribute**
- **Proposed requirement:** Before a directional result from an arm that differs in an assigned person attribute is analysed, Phase E MUST check that the manipulation took effect over the readout window, using two criteria declared in advance and graded `[I]`:
  - (a) rank: the arms are ordered correctly on the realised attribute (for `basic_level`, the M7.A.1 estimate, not the assigned value);
  - (b) magnitude: the arms are separated by at least a declared margin, both on that attribute and on its proximal behaviour (for example the self-directed channel share, M16.A.9).
  
  An arm that fails MUST be reported as "manipulation not realised", never as a null about the mechanism. The gate's result MUST be reported whether it passes or fails.
- **Where:** Phase E (new M17 clause), M11.4.
- **Evidence:** §3.3, §5.10, Table 8. A model at ρ = 0.96 with 0.71-point separation produced flat behaviour. Rank without magnitude is insufficient (SHOWN). The EPModel transfer is INFERENCE: assigned `basic_level` is re-estimated from `functional_level` history, and borrowing inflates functional level (M1.A.5c), so assigned and realised can diverge within a run.
- **What it changes:** Adds a test and a reporting rule. Improves validity.
- **Cost / risk:** Low. The margins are `[I]` and MUST be frozen under M10.B.4. The paper's own gate was tightened after seeing the data, and the freeze prevents that here. PARTLY covered by M17.E.2 dose–response and DL §2.6 "regress moves on inputs".

**TM3. Composite readouts emitted by component, with declared mechanical dependencies and opportunity denominators**
- **Proposed requirement:** Every M11.G readout, and any readout named for a construct the corpus describes as having more than one face (fusion, tension, closeness/distance), MUST be emitted as its components. An acceptance criterion stated on it MUST name which component and which direction it asserts. Where one component bounds another by construction, the dependency MUST be declared and that component MUST be reported beside it, never alone. Every rate readout (CONFLICT per tense tie-week, TRIANGLE per activation, I-POSITION per admitted occasion) MUST report its numerator and its opportunity count taken from the M16.A.3b legal set, so that "no opportunity" is distinguishable from "opportunity not taken".
- **Where:** M11.G, M16.A, Phase E reporting.
- **Evidence:** §3.4.1, §5.2, §5.7, S11. Dispersion and segregation move in opposite directions. The paper tested non-redundancy by regression, declared the closure dependency, and used denominators (4,069 opposing pairs, 25 of 598 replies crossing) to separate refusal from absence (SHOWN).
- **What it changes:** Generalises J1 from one pair to all composite readouts, and adds the denominator rule. Improves validity, and fidelity via M11.F.6's two-sidedness.
- **Cost / risk:** Low. M16.A.3b already logs the legal set. PARTLY covered by J1, M11.G.1 component 2 (already a vector), M11.F.6 and M17.F.1(c).

**TM4. Crossover check for direction tests, with cell-level inference and an additive-substitution mutant**
- **Proposed requirement:** For each M11.C criterion, the moderators the theory names (at minimum `basic_level` level and `chronic_anxiety` level, alongside M17.E.5's constant pairs) MUST be declared in advance, and the criterion MUST be run at **three or more** levels of each. The sign of the arm difference MUST be reported per level; a sign that reverses within the declared range MUST be reported as conditional, not as a property of the mechanism. Interaction terms MUST be estimated on cell-level summaries, with a leave-one-cell-out check, and never with seeds treated as replicates of the design. Where the theory states an interaction, a mutant that replaces the product form with an additive one MUST turn the criterion red.
- **Where:** Phase E (M17.E.4/E.5 extension), M11 mutation protocol (M11.4f).
- **Evidence:** §5.5, Table 6, §5.10. Agreeableness sets the sign of Openness. The primary-model estimate was influence-driven (3 of 9 leave-one-out fits), and main effects replicated at chance while the moderation replicated (SHOWN). One model was purely additive and lost every joint effect (SHOWN). The mutant is INFERENCE.
- **What it changes:** Answers owner question (b). Yes, but targeted at declared moderators, not full factorials. Two reference configurations (M17.E.4) cannot tell a crossover from attenuation, so three levels are needed. Improves validity and fidelity.
- **Cost / risk:** Moderate compute (a factor of about 3 per moderator), affordable at N = 11. Moderators must come from the corpus, not from main effects already seen. The paper picked its crossed pair after E1, which is a selection risk. No rule conflict.

**TM5. A negative-control readout per criterion**
- **Proposed requirement:** Each M11.C criterion SHOULD name at least one readout that the theory says its manipulation does not move. The ensemble report MUST show that readout's arm difference beside the criterion's own. Movement beyond the equivalence margin (M11.4a) MUST be flagged as a possible global-gain artefact, such as an intensity scale that raises every readout.
- **Where:** M11.C / M11.4, Phase E.
- **Evidence:** §3.5, §6.3. The neutral filler topic separated a response-style artefact (r = +0.367 in one model, −0.061 in the other) from a real effect (SHOWN). The rule-based analogue is INFERENCE.
- **What it changes:** Adds a test. Improves validity.
- **Cost / risk:** Low compute. The risk is that the corpus rarely names an outcome as unaffected. Where none can be sourced, the criterion should say so rather than pick one, since an invented negative control is itself `[I]`. NEW.

**TM6. Results ledger: every reported number is generated and traceable**
- **Proposed requirement:** Every quantity in a Phase E report MUST be emitted by the analysis pipeline from the run-level records, and a ledger MUST give, for each quantity, its source file, field, aggregation and seed set. Hand-transcribed values MUST NOT appear.
- **Where:** Phase E reporting, M16.
- **Evidence:** §4.6 and S12 (practice SHOWN). Even with the ledger, this paper's text and ledger disagree in at least four places (ρ 0.729 vs 0.620, and the dz values listed above), which argues for a check that compares report text with the ledger.
- **What it changes:** Adds a reporting rule and a check. Improves validity, and applies the owner's global rules 1–4.
- **Cost / risk:** Low. NEW as far as my keyword search shows (no ledger or transcription rule in spec rev10).

**X-line note (Phase F narrator, not v2).** Most of this is covered by X1–X9. One addition, X10: an induction gate for any narrator persona, run per model before use.
- Rank and magnitude criteria, as in TM2.
- A behavioural check, because questionnaire validity can be passed by the model reading its own prompt (§6.4).
- A neutral-content probe, to detect response style (§6.3).
- A threshold on failed calls that aborts the run instead of substituting null output (§4.6).

### C. Search terms

- "personality composition" "agent-based" polarization heterogeneity
- "dispersion" "segregation" polarization measures dissociation
- "Krackhardt E-I index" echo chamber
- "manipulation check" "agent-based model" induced attribute
- "negative control outcome" simulation study
- "diversity prediction theorem" collective accuracy partialling
- "hidden profile" Stasser Titus simulation
- "response surface" interaction "cell means" "leave-one-out" simulation experiment
- "Serapio-Garcia" psychometric personality language models induction validity
- "Tosato" personality measurement instability LLM conversation history
