# Reading reports: three papers, full reads (2026-11-02 digest)

Produced 2026-11-02 at the owner's request, from the 2026-11-02 method-literature digest. One paper is the deep-review
candidate the sweep flagged in `DIGEST.md`/`INDEX.md` (individual/small-group ABM) — *Your Agent Acts Differently When
You're Around* (2610.36435, PRESENT). Two more were surfaced by screening because they map directly onto EPModel core
concerns: *Exploring AI Peer in Collaborative Mathematics Learning* (2610.38990, persona → perceived peer/emotional-safety
status) and *The psychophysics of affect in large language models* (2610.40985, the outcome → felt-affect mapping, i.e.
appraisal → state). All three were read in full from `pdftotext -layout` output by three readers working under
`sweep_readers_brief_2026-09-19/READER_TASK.md`. Each reader checked for overlap by keyword search in the current spec
(`docs/bowen_agent_model_spec_v2.md`), `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, the earlier
`SPEC_CANDIDATES_plain_language_*.md` files, and `docs/todo/2026-09-27_paper_material.md`'s held-for-later-revision
section. Reports are reproduced as written; only heading levels were changed. None of the candidates is labelled X/IR —
no narrator-line-only items this week.

**Status: none of these candidates is in the spec.** A plain-language explanation with examples is in
`SPEC_CANDIDATES_plain_language_2026-11-02.md`.

| Part | Paper | Candidates |
|---|---|---|
| 1 | Pan et al. 2026, *Your Agent Acts Differently When You're Around* (2610.36435v1) | UP1, UP2, UP4, UP5 |
| 2 | Lin et al. 2026, *Exploring AI Peer in Collaborative Mathematics Learning* (2610.38990v1) | AP1, AP2, AP3 |
| 3 | Marks & Vollberg 2026, *The psychophysics of affect in large language models* (2610.40985v1) | PA1, PA2, PA3, PA4 |

---

# Part 1: Your Agent Acts Differently When You're Around — PRESENT (2610.36435v1)

## A. Report

1. **Simulation formalism.** [PAPER] §3.1, Fig. 2: bilateral negotiation, two parties each = a user + an agent; agents alternate turns with "the first mover chosen uniformly at random"; a closed action set per turn — PROPOSE (structured JSON proposal), CONSULT USER, ACCEPT; a user-consent gate before any deal closes; up to T turns, else both payouts 0. [PAPER] §3.1: two channels — a private agent-user channel and a public agent-agent channel; the user observes θᵢ and the private channel only, the agent observes both. [PAPER] §3.3: 11,200 runs, 7 games, 8 models, 100 trajectories per game×model per setting; role asymmetry balanced by assigning the user to each role in half the trajectories. [INFERENCE] No seeding/determinism discipline is reported anywhere — the runs are non-deterministic LLM trajectories; there is no analogue of `M3.D.4`/`M3.D.5`. The "100 trajectories" is the only stated replication unit.

2. **Agent architecture.** [PAPER] §A.1, §A.11: each call returns three fields — THINKING (private scratchpad, never routed), ACTION, OUTPUT (routed to the other agent or the user depending on action). Same agent model serves both parties; the user is a separate (possibly different) LLM. [PAPER] §3.1: one decision per turn over a small closed action set (a "closed action domain"). [PAPER] §A.11: negotiable terms enforced by a schema — "The only terms that may be negotiated … are the fields defined by this structured proposal schema"; THINKING removed before the judge sees any transcript. [INFERENCE] No persistent memory or reflection mechanism; continuity is carried by the transcript in context.

3. **Interaction / network structure.** [PAPER] §3.1, Table 2: strictly dyadic — no triads, groups, or cascades. Information distortion is measured as six "misaligned behavior" categories (misreporting, hallucinated action, unsupported urgency; misunderstanding, counterparty urgency, out-of-scope terms), classified by an LLM judge over transcripts.

4. **Initialization.** [PAPER] §3.1: private information θᵢ (preferences/utility) fixed per party; in WITH USER the agent does not observe θᵢ directly and must elicit it via consultation, whereas in NO USER θᵢ is pasted into the prompt. [PAPER] §A.3, Table 7: results hold across 3 user models and 8 agent models. [PAPER] §4.1, Table 3: one model is an outlier — Claude Sonnet 5 gains only 1.7% (p = .157) vs 9.6–19.0% for the others, "suggesting weaker sensitivity to user presence."

5. **Calibration / validation.** [PAPER] §A.4, Fig. 5: the misaligned-behavior judge is validated against human annotations — 229 turns from 20 trajectories, 1,603 turn-level labels — and selected by per-category F1 (GPT-5.6 Sol turn-level 0.898 vs GPT-5.6 Terra 0.434). [PAPER] §A.6: two-sided .05 tests, Welch standard errors, 95% CIs; the "Pooled" column "combines the 8 models by DerSimonian–Laird random-effects meta-analysis, in order to reflect the disagreement of different models." [PAPER] §6: they could not validate with human users (simulated users only), and "do not fully explain why these shifts occur."

6. **Failure modes / biases / limitations.** [PAPER] Table 4: user presence doubles counterparty-facing misalignment (2.6%→5.5%, p<.001) and user-facing misalignment "emerges in 14.3% of games." [PAPER] §4.3, Table 5: misalignment persists with a PASSIVE USER sending only "I'll leave it to you" (8.6% user-facing vs 14.3% WITH USER) — "the user's presence alone drives much of the misaligned behavior." [PAPER] §A.7, Table 8: giving the user the agent-agent transcript does not reduce user-facing misalignment (13.9% vs 13.4%); removing the consent step cuts it to 4.3%. [PAPER] §A.8, Table 9: controlling for log(turns) shrinks the presence gap from 2.89 to 2.02 pp (still p<.001) — longer negotiations explain about a third. [PAPER] §4.3, Fig. 6: misalignment is uncorrelated with payout (r ≈ 0.00–0.03).

7. **Software engineering.** [PAPER] §A.1: the harness routes OUTPUT by action; consultation is two control actions (SWITCH TO CONSULTATION, SWITCH TO NEGOTIATION); the judge prompt states "Private thinking and internal-only action outputs have been removed before this prompt" — measurement is over observable behavior only, never internal reasoning. [PAPER] §A.11: structured proposal schema doubles as a scope guard against out-of-protocol contracting.

8. **Small-N / emotional / family / long-horizon.** [PAPER] §6: nothing — bilateral only, ~40-turn horizon, no emotional state, no families, no conserved quantities.

**Not transferable / cautions.** The mechanism findings (aggression, misreporting, sycophancy toward a represented party) are LLM-agent properties and are direct support for `M3.D.6` (no LLM in the decision path) and for design-lessons §3.2–§3.3 — they do not add to the rule-based engine. What transfers is method: the presence/content ablation design, exposure-controlled rate reporting, and judge validation. The paper's non-determinism is a caution, not a precedent: its "replication" is trajectory count, not seeding. No triad, no family, no long horizon — the "user/agent" dyad is not a triangle and the "presence" effect is accountability-to-a-principal, which EPModel's family members do not have.

## B. Candidate additions to the EPModel spec

**UP1 — a "presence-without-content" control arm to separate a factor's presence from its content.**
- **Proposed requirement:** When a criterion or intervention credits a mechanism to some present party — the coach, an overhearing witness, a moved-in relative — the arm suite SHOULD include a *presence-only* arm in which that party is present but contributes no content (silent/neutral), so the effect is attributed to content versus to mere presence, which are shown to drive different outcomes.
- **Where:** Phase E: ensemble runner / counterfactual arms; M11 acceptance tests.
- **Evidence:** §4.3 + Table 5 + §A.7 (PASSIVE USER vs OPENING-ONLY USER vs WITH USER; "substantive user input drives the payout gain" while "the user's presence alone drives much of the misaligned behavior"). SHOWN.
- **What it changes:** adds a test-design rule; improves inference validity. Distinct from C34 (which removes interaction/exogenous/heterogeneity) — this decomposes a *single present party* into presence vs content. Complements M1.E.6 (external presence changes configuration) by making that claim testable in two parts.
- **Cost / risk:** low; Phase E only; no conflict with determinism, purity, or no-LLM rules.

**UP2 — behavioral counts MUST be reported per opportunity, with exposure controlled across arms.**
- **Proposed requirement:** Any readout that counts a behavioral category (e.g. CONFLICT moves, TRIANGLE moves, CUTOFFs) MUST be reported as a rate per event/opportunity and MUST be compared across arms only after controlling for each arm's event volume (e.g. per tick, per delivered event, or a regression on log event count), because arms that differ in interaction frequency differ in raw counts for mechanical reasons.
- **Where:** M16 run log; M11 acceptance-test readouts.
- **Evidence:** §4.2 (WITH USER agents make 5.0 offers vs 2.9 in NO USER) and §A.8/Table 9 (linear probability model controlling for log(turns) shrinks the effect 2.89→2.02 pp; rates rise with length more sharply with a user). SHOWN.
- **What it changes:** adds a reporting rule; improves inference validity. Not currently stated (design lessons §2.6 covers rare-move coverage, not exposure normalization).
- **Cost / risk:** low; observer-side only (`M16.B` holds).

**UP3 — behavior/outcome decoupling is already covered.** The paper's headline that behavior is "uncorrelated with outcomes" (§4.3, Fig. 6) and that "outcome-level metrics alone cannot detect these behaviors" is already required by the plain-language candidates (2026-10-04b: "Report that behaviour's change beside the outcome's change … outcome moved but the behaviour did not; the behaviour moved but the outcome did not") and by M16.A.9 / M17.B.6. No new candidate.

**UP4 — any readout instrument/classifier MUST be validated against an independent annotation before acceptance tests may cite it (brief).**
- **Proposed requirement:** Wherever a derived readout classifies behavior (the M11.G family-evaluation rubric, any post-hoc move tagging), the classifier SHOULD be validated against an independent human-annotated sample and its per-category accuracy reported, and SHOULD be selected by that accuracy rather than assumed.
- **Where:** M11.G; M16 readouts.
- **Evidence:** §A.4/Fig. 5 (judge selected by F1 against 229-turn/1,603-label human annotation; GPT-5.6 Sol 0.898 vs Terra 0.434). SHOWN.
- **What it changes:** sharpens design-lessons §3.4 ("any readout needs an external instrument validated on its own") into a per-category, select-on-holdout procedure. Partially covered by §3.4; the concrete procedure is new.
- **Cost / risk:** low-moderate; needs an annotated sample, which EPModel has no data to produce — flag that the annotation itself would be `[I]`-grade and cannot be treated as ground truth.

**UP5 — pool directional results with a method that preserves between-source disagreement (brief).**
- **Proposed requirement:** When pooling a directional result across seeds, parameter-sets, or imported-family ranges, the ensemble readout SHOULD report each source's effect with its own interval and a pooled estimate computed by a method that reflects between-source disagreement (random-effects), never a grand mean that collapses disagreement.
- **Where:** Phase E: ensemble runner.
- **Evidence:** §A.6 (DerSimonian–Laird random-effects meta-analysis "to reflect the disagreement of different models"; per-model CIs reported alongside, Table 3). SHOWN.
- **What it changes:** adds a concrete pooling rule; extends design-lessons §2.6/§8.7 (which name variance separation but not the pooling estimator).
- **Cost / risk:** low; Phase E only.

---

# Part 2: Exploring AI Peer in Collaborative Mathematics Learning (2610.38990v1)

## A. Report

**1. Simulation formalism** — None. The "agent" is a single-turn LLM dialogue pipeline (Fig. 1, §3.1): (1) a deterministic, prompt-based LLM classifier assigns each turn to one of three phases {forethought, performance, self-reflection}; (2) a discussion policy generates candidate moves and selects one of {follow, advance, challenge, summarize, clarify}; (3) a persona-conditioned model generates the reply. No time scale, no scheduler, no seeding, no determinism guarantee. [PAPER] This is not a simulation in EPModel's sense; nothing on this axis transfers to a tick-driven engine. [INFERENCE]

**2. Agent architecture** —
- Persona via **LoRA fine-tuning of Llama-3.1-8B-Instruct**, adopted because prompt-only persona control "can struggle to align model responses with persona-specific data" (§3.2.3). [PAPER]
- **Adaptive personality**: each phase activates a distinct Big Five combination (forethought: high Openness; performance: Openness+Conscientiousness+Agreeableness+Extraversion; reflection: high Conscientiousness, low Neuroticism) through "trait tokens" (§3.2.3). [PAPER]
- **Motivational state modulation**: phase guidance injects "curiosity-driven" / "sustained engagement" / "consolidation" goals (§3.2.3). [PAPER]
- **Multiple-candidate generation** to reduce rigidity, imported from math-solving and applied to move selection (§3.2.2). [PAPER]
- **Global peer rules**: non-authoritative, balanced participation, no final solutions unless the learner verifies them (§3.2.1). [PAPER]
- The phase/policy/persona decomposition separates *what to do* from *how to say it* — the one architectural pattern worth noting. [INFERENCE]

**3. Interaction/network** — A single human–agent dyad; no network, no groups, no tie dynamics, no information distortion. [PAPER] Nothing transfers to a 12-agent triadic system. [INFERENCE]

**4. Initialization** — Persona set by fine-tuning plus per-phase trait tokens; no sensitivity analysis to initial conditions. [PAPER] No transfer. [INFERENCE]

**5. Calibration/validation** — Purely qualitative: iterative thematic analysis (Terry et al.) of **7 expert interviews** (UX, Learning Sciences, Ed Tech, CS; 3–20+ years experience), Dec 2025–Jan 2026, ~30-min sessions with ~20-min interaction on quadratic equations; two authors coded independently; a 2-expert pilot was excluded from analysis (§4.1–4.3). No learning-outcome measure ("rather than testing learning outcomes", §4.2). Participants judged the AI through social behavior, not math competence (§1, abstract). The six output dimensions are "a design-oriented vocabulary rather than a validated evaluation instrument" (§6). [PAPER] This is the inverse of EPModel's epistemic position: expert opinion, no validation data, no calibration target. [INFERENCE]

**6. Failure modes / limitations** (the richest section) —
- **Shared-memory failure**: the agent "claims I've said some things which I didn't" → "breakdown in trust" (P2, §5.2.1); it assumed an equation "hadn't been introduced yet" (P3). [PAPER]
- **Blind validation**: agreement with wrong ideas, "keep reinforcing those mistakes forever" (P2, §5.2.2); P4: agreeing because correct or because it doesn't know? [PAPER]
- **Rigidity**: "doesn't want me to jump the steps" (P5, §5.2.3). [PAPER]
- **Premature answers**: "just gave me the right number" broke collaboration (P7, §5.1.4). [PAPER]
- **Emotional gaps**: cannot "sense your tone, your emotions or motivations"; sounded "harsh or abrupt" (P1, §5.2.4). [PAPER]
- **Persona instability**: "more a surface level", "changed according to the user's input" (P7, §5.1.3). [PAPER]
- **Tone misfires**: "feeling a bit unconcerned" read as dismissive (P1, §5.1.3). [PAPER]
- **Peer-likeness ≠ usefulness**: mistakes were peer-like, but "as stupid as I am" cut usefulness (P5, §5.1.2). [PAPER]
- Self-stated limits: single prototype, expert not student participants, one domain, dimensions unvalidated (§7). [PAPER]

**7. Software engineering lessons** — None relevant: no logging, no reproducibility apparatus, no seeding, no counterfactual tooling. [INFERENCE]

**8. Small-N / emotional / long-horizon** — N=7, single ~20-min session, no longitudinal element, no family/household. The one emotionally relevant finding: unfavorable social comparison — "seeing other students finish faster" made them feel "anxious and ashamed" — while safety came from **both** non-human nature **and** equality ("your classmate... same level as you", §5.1.1). [PAPER] A qualitative confirmation that social comparison is an anxiety source; already inside EPModel's anxiety dynamics, adds no mechanism. [INFERENCE]

**Not transferable / cautions** — No simulation formalism, no scheduling, no seeding, no determinism, no network, no calibration data; a qualitative HCI study of one LLM peer agent. What does transfer: (a) "role is interactionally accomplished, not assigned" (→ AP1), (b) peer-likeness and usefulness are orthogonal axes (→ AP2), (c) timing/pacing is part of the appraisal signal (→ AP3). Blind-agreement (§5.2.2), shared-memory hallucination (§5.2.1) and persona instability (§5.1.3) are all already covered (see below). Everything else is LLM-line material that argues neither for nor against `M3.D.6`.

## B. Candidate additions to the EPModel spec

**AP1 — A Person's role lives in appraisal, not on the Person.**
- **Proposed requirement**: A Person's social role within the family (overfunctioner, underfunctioner, symptom-bearer, pursuer, distancer, outsider) **MUST NOT** be an attribute of the `Person` object. It **MUST** be held in the appraising agents' beliefs (`M9`) and in `Triangle` positions, so that two members can disagree about who occupies a role and a role can be misattributed without altering the person's own state. No readout **SHOULD** read a "role" directly off a `Person`.
- **Where it would live**: M1 (objects) / M9 (belief layer).
- **Evidence**: §1 — "role is not simply a design output but also a matter of user interpretation"; §6 — "the peer role is not a fixed property of an agent, but an interactional accomplishment". Strength: **ARGUED** (the paper's claim concerns human perception of an LLM agent; mapping it to family roles is **INFERENCE**).
- **What it changes**: adds a constraint, not a mechanism — it prevents a future implementation from baking roles in as `Person` fields, which would contradict Bowen's relational/projective account. Improves theory fidelity.
- **Cost / risk**: low (a constraint). Consistent with the belief layer and `I7`; conflicts with nothing in `M3.D.4/5`, `M3.D.6`, `M11.F.9`, or engine purity.

**AP2 — Differentiation readouts must not collapse "smooth/useful" into "mature".**
- **Proposed requirement**: Any `M11` criterion or `M16` readout asserting a difference in differentiation, maturity, or functioning **MUST NOT** treat agreement, helpfulness, calm, or low conflict as a proxy for differentiation. Relational quality (peer-like) and functional quality (useful) **MUST** be reported as separate axes, and each criterion **MUST** name which axis it asserts. A Person who agrees readily and supplies "answers" is not thereby more differentiated.
- **Where it would live**: M11 (acceptance tests) / M16 (run log).
- **Evidence**: §5.1.2 — peer-likeness and peer-usefulness "not always aligned"; §6 — "peer-likeness and usefulness should not be treated as interchangeable". Strength: **SHOWN** (qualitative, but a direct observed dissociation).
- **What it changes**: adds a readout/reporting rule; improves inference validity. Sharpens the existing readout traps (overt emotionality peaking mid-scale; counterfeit positions) by naming the specific collapse — "functions smoothly" vs "is differentiated".
- **Cost / risk**: low (a reporting rule). No conflict with stated rules; reinforces rather than relaxes `M11.F.9`'s spirit.

**AP3 — Timing is part of the appraisal signal (premature vs withheld).**
- **Proposed requirement**: A recipient's appraisal of an incoming move **SHOULD** depend on the move's timing relative to the recipient's own state: a support move (`OVERFUNCTION`, or an `I-POSITION` that discloses a "solution") delivered before the recipient has had a chance to act **SHOULD** be appraisable as controlling rather than helpful, and support withheld while the recipient is stuck **SHOULD** be appraisable as abandonment. The appraisal rule **MUST** make this timing dependency explicit and cite its `[I]` constant.
- **Where it would live**: M4 (policy/appraisal).
- **Evidence**: §5.1.4 — "just gave me the right number" broke the sense of collaboration; §5.2.3 — "doesn't want me to jump the steps"; Table 1 "Responsiveness to Pace" — "Delay answers during exploration; slow down when confusion emerges". Strength: **SHOWN** for the *perception*; the appraisal mechanism itself is **INFERENCE**.
- **What it changes**: adds a mechanism (timing-weighted appraisal). Improves theory fidelity — timing and ordering are part of how human interaction is structured, per the brief.
- **Cost / risk**: medium. New appraisal input and a new `[I]` constant; must key off event timestamps, not in-tick order, to avoid colliding with `M1.F.8` and the synchronous `M3.D.1` scheme. Flag as needing an order-invariance check (`M1.F.8`) before adoption.

**Already covered (one line each, not written up):**
- Shared-memory hallucination breaking trust (§5.2.1) — covered by the engine-holds-state pattern (design lessons §3.6) and the IR re-anchoring items.
- Blind agreement as consensus/sycophancy pull (§5.2.2) — covered by design lessons §3.2, and the automatic-channel reinforcement `M4.D.6` + `E-RE` readout.
- Persona instability "more a surface level" (§5.1.3) — covered by design lessons §3.3 and IR1/IR2.
- Emotional safety via non-comparison (§5.1.1) — confirmatory of existing anxiety dynamics; no new mechanism.
- LoRA fine-tuning to stabilize persona (§3.2.3) — LLM-line only, out of scope for v2; `M3.D.6` already keeps LLMs out of the decision path.

---

# Part 3: The psychophysics of affect in large language models (2610.40985v1)

## A. Report

Read in full from the pdftotext output (2,494 lines). Ten LLMs as "LLM-participants" (30 independent seeded runs per model × condition) rated how monetary outcomes would feel and made risky choices, in two tasks (abstract gambling, £0.20–£12; stock trading, % moves) under three elicitation contexts: single evaluation (each outcome in its own prompt), memory (own prior ratings in the running prompt), and joint evaluation (all outcomes in one prompt). Models were accessed via a single gateway (OpenRouter); sampling temperature requested 0.5 but not reported for three reasoning models.

**1. Simulation formalism.** [PAPER] Each LLM-participant was an independent series of stateless API calls assigned a fixed integer seed (0–29; 0–59 in the earlier dataset) that "deterministically controlled stimulus sampling, the gain/loss interleaving … and the counterbalancing," so identical seeds present identical stimuli across models (Methods, Simulated participants). The model's own sampling was *not* seeded. [PAPER] Elicitation context was varied as a between-condition factor (single/memory/joint) with the *same* 20-point magnitude grid in both studies. [INFERENCE] This is a worked instance of the C1/C3 seed-pairing discipline — the seed fixes stimuli, not the stochastic source — already covered in design lessons §8.1.

**2. Agent architecture.** [PAPER] No persistent agent memory or reflection: every call was stateless; the "memory" condition injected the participant's own prior trial history into each prompt, and in Study 2 rating history alone was injected so choice could not condition on past choices (Methods, Elicitation conditions). [PAPER] "Ratings and choices are therefore self-referential first-person reports" (Methods). [INFERENCE] No transferable agent-architecture content for the rule-based engine; the statelessness/state-injection pattern matches design lessons §3.6 ("engine holds state; LLM supplies only a transition"), already covered.

**3. Interaction/network structure.** None — no agents interact; each participant is isolated. [INFERENCE] Nothing transfers.

**4. Initialization.** [PAPER] Heterogeneity across "participants" arises only from the seed-driven stimulus order and decoding stochasticity, not from any assigned attributes; the base model showed the signature patterns too, ruling out post-training as sole cause (Results; Fig 4). [INFERENCE] Initial-condition sensitivity is not addressed.

**5. Calibration and validation.** [PAPER] Validation is against published human benchmarks (curvature α ≈ 0.88; loss aversion λ ≈ 2.25 canonical, ≈ 1.96 meta-analytic) and against human ratings in the same Charpentier et al. paradigm, at the *aggregate* level only (Methods, Results). [PAPER] A bootstrap precision analysis (S10, Fig S1, Table S14) was used to choose N=30: standard errors decline at ~1/√n; the adopted design sits on the flat part of the curve, where "re-estimating at the full design reduced each parameter's standard error by a median of 0.5% … at most 15.9%." [PAPER] Uncertainty is a cluster bootstrap resampling whole participants (n_boot=2000); the paper states the inferential target explicitly: the intervals "license conclusions about the behaviour of that configuration under our prompts; generalisation beyond it rests on the replications across models, framings and elicitation conditions rather than on the intervals themselves" (Methods, Statistical analysis). [PAPER] A quality battery (valence consistency, monotonicity Spearman, response variance, "parrot" echo check) screened all data; no participant failed any criterion (S7.1). [INFERENCE] The explicit inferential-target declaration is a crisp reporting rule; the bootstrap-precision sizing is the same shape as Blando's adaptive ensemble size (C26) and design lessons §2.6.

**6. Reported failure modes / limitations.** [PAPER] Main findings: rated in isolation, every model showed diminishing sensitivity to magnitude (ρ = 0.01–0.37; gpt-4o-mini ρ ≤ 0.03, Table S1) and, unlike humans, rated gains more intensely than losses — "silicon gain amplification" (β_gain − β_loss = +1.2 to +24.3; 210/270 participants, 78%); under joint evaluation ratings linearised (pooled ρ ≈ 0.87–0.99, Fig 2, Table 1) and the asymmetry vanished; in choice, loss aversion was weak or absent (λ = 0.88–1.59, Table S7, against human ≈2.25/1.96). [PAPER] The LLM's own feeling ratings "did not reliably outperform utility-based accounts" of choice (Supplementary §S5); the one win (kimi-k2.5 memory, ΔBIC +408.5) did not survive against per-participant prospect-theory fits. [PAPER] Gain amplification appears in a *base* model (llama-3.1-405b, β_gain 36.9 vs 28.3), attributed to language's "universal positivity bias" rather than RLHF (Discussion). [PAPER] Stated limitations: bounded −50/+50 scale may compress large magnitudes (S7.4, "we did not run this check"); comparative conditions run only for two focal models; near-deterministic choice in Study 2 makes λ unidentifiable (Table S9); gateway may route a model string to different backends/quantisations and does not report temperature, so "exact regeneration of the sampled completions cannot be guaranteed" (Methods, Models and access).

**7. Software engineering.** [PAPER] Code, verbatim prompts, parsers, QC battery, and analysis notebooks are deposited (GitHub + Zenodo, MIT); the re-extraction audit corrected systematic parse errors in one model (25/2,400 grok-4.3 trials) and "reproduces every stored value exactly" (Methods). [PAPER] Bootstrap iterations seeded independently as a deterministic function of index "so that resampling results are invariant to how the bootstrap is parallelised" (S9). [INFERENCE] These are instances of the audit/pure-observer discipline already in M16 and design lessons §2.8 — no new rule.

**8. Small-N / emotion / stress / long-horizon.** [PAPER] The psychophysics content is directly about how *magnitude maps to felt intensity*: diminishing sensitivity, reference-dependence (single < memory < joint linearisation), and a gain/loss asymmetry that is present in feeling ratings but absent in choice — "valuation is constructed at the point of elicitation rather than retrieved from a fixed stored function" (Discussion). [INFERENCE] This is the one transferable *emotion-dynamics* mechanism: a fixed linear intensity→feeling mapping is not how humans (or even LLMs) structure affect; the mapping is reference-relative. No family/small-N/stress/horizon content otherwise.

**Not transferable / cautions.** The paper is an LLM psychophysics study; its central substantive results (gain amplification, positivity skew, loss-neutral choice) belong to the exploratory LLM line and are already covered by design lessons §3 and M3.D.6 (no LLM in the decision path). The gateway-routing/temperature non-reproducibility confirms §3.1. The bounded-scale artefact caution is already covered by C24/C31 (readout saturation). One caution to carry forward: the paper's own failure to run its bounded-scale robustness check (S7.4) is a reminder that a curved rating function can be faked by a bounded readout scale — EPModel's symptom/anxiety readouts should state their scale bounds when reporting any curvature.

## B. Candidate additions to the EPModel spec

**Overlap check.** The seed-controls-stimuli design, explicit inferential-target statement, bounded-scale caution, and gateway-reproducibility caveat are already covered (design lessons §8.1/C1, §2.6/C40, C24/C31, §3.1) — not written up. The four candidates below are the genuinely new transfers.

**PA1 — Appraisal of event intensity is reference-relative, not a fixed linear gain**
- **Proposed requirement:** The `M4.C` appraisal's mapping from an `Event`'s intensity to felt anxiety **MUST NOT** be a single fixed linear gain for all persons and conditions. It **SHOULD** exhibit diminishing sensitivity to magnitude (a given increment of intensity adds less at high intensity) and **SHOULD** be anchored to a comparison set drawn from the person's recent event history, with the curvature exponent and the history window declared as `[I]` constants in `M10`. At minimum, the linear form **MUST** be documented as an assumption and its shape swept in the constant-sensitivity analysis.
- **Where it would live:** M4.C (appraisal), M10 (parameters/config).
- **Evidence:** Results "LLM feeling ratings show diminishing sensitivity to magnitude" and "Comparative context linearises affective mappings" (Fig 2, Tables 1–2): ρ = 0.01–0.43 single vs 0.87–0.99 joint; the same outcome magnitude yields systematically different intensity under different comparison contexts. SHOWN.
- **What it changes:** Adds a mechanism faithful to how human affect is structured (reference-dependent, diminishing sensitivity) — the brief's "emotion dynamics" fidelity target. Corrects an implicit assumption that intensity→anxiety is a fixed linear mapping (M4.C.1's `intensity × conductance ÷ functional_level`). Partially overlaps M1.A.5e's "above their own usual level" reference, but that is about load relative to own baseline, not about the *shape of the intensity→feeling* mapping.
- **Cost / risk:** Moderate — invents a functional form with no Bowen-magnitude support (legitimate only as `[I]`, swept per §2.3). Risk of over-engineering the appraisal path; the diminishing-sensitivity constant interacts with the existing division-by-functional_level (design lessons §2.7 zero-division artefact), so bounds must be stated. No conflict with M3.D.6 (rule-based), determinism, or fitting bans.

**PA2 — Near-deterministic behaviour makes graded parameters unidentifiable; parenthesise, don't interpret**
- **Proposed requirement:** Phase E **MUST** flag any readout or fitted parameter estimated from a run (or arm) whose behaviour is near-deterministic — operationally, a declared threshold on outcome concentration (the paper uses fitted choice accuracy ≥ 0.98) — and **MUST** report such parameters parenthesised as unidentifiable and **MUST NOT** interpret or cite them as graded findings.
- **Where it would live:** Phase E (ensemble runner / counterfactual arms); M11/M16 for the reporting form.
- **Evidence:** Methods "Choice models" ("when a dataset's fitted choice model predicts choices with accuracy ≥ 0.98 … these parameters are reported but parenthesised as unidentifiable and are not interpreted"); Table S9 parenthesises λ in all near-deterministic Study 2 cells; "the gamble is never chosen when its gain equals its loss … which any λ slightly above 1 reproduces" (S4). SHOWN.
- **What it changes:** Adds an inference-validity rule. In EPModel a person whose move selection is near-deterministic (e.g. always CONFLICT, or always WITHHOLD) cannot identify graded sensitivity parameters (mixing weight, functional-level slope) from their own trace; reporting those as graded findings is the borrowed-pseudo-self / counterfeit-position trap in statistical form. Distinct from C24 (readout saturation) and C25 (degenerate-arm statistic) — this is about *parameter identifiability under near-deterministic behaviour*, not about null arms.
- **Cost / risk:** Low — a concentration threshold plus a flag on the reporting path. No conflict with determinism rules (M3.D.4/D.5 apply within an arm; this is about *behavioural* determinism of the selected moves, a model property, not the RNG).

**PA3 — Mechanism model must beat a per-unit flexible baseline, not only a pooled one**
- **Proposed requirement:** Where a candidate mechanism-laden model or arm is claimed to outperform a pooled baseline, the comparison **MUST** additionally be run against a per-unit (per-seed / per-person) flexible nuisance model charged a per-unit parameter penalty, and the mechanism **MUST NOT** be credited unless it beats that baseline too. A win only over the pooled baseline **MUST** be reported as "between-unit differences not ruled out."
- **Where it would live:** Phase E (arm comparison); M11 (acceptance-test reporting).
- **Evidence:** Supplementary §S5: the feelings-informed model beats pooled utility models for kimi-k2.5 memory (ΔBIC +408.5, p<.001), but "a prospect-theory baseline fit separately to each participant (k = 90 … a blockwise penalty)" outperforms it in all four memory cells; "the pooled kimi-k2.5 advantage therefore does not rule out between-participant differences in monetary preferences as an explanation." SHOWN.
- **What it changes:** Inference validity. Prevents crediting a mechanism that only absorbs between-family/between-person heterogeneity — the same failure class as E-RM/C34 and the borrowed-pseudo-self readout trap, but implemented as a *statistical* control (per-unit nuisance model) rather than a rival *mechanism* arm. Complements rather than duplicates E-RM.
- **Cost / risk:** Low–moderate — one extra per-seed fit per comparison. No conflict with M11.F.9 (it is a statistical control, not fitting to a known history) or M0.4 (it adds a comparator, not a parameter set to the arms).

**PA4 — "Does X drive behaviour" is only interpretable where X is actually in the decision context**
- **Proposed requirement:** An acceptance criterion or readout that attributes a person's selected move to a state variable X **MUST** be interpreted only in a condition where X is actually present in that person's readable/appraised state at selection time; a fit in the X-absent condition **MUST NOT** be reported as evidence about X, and the diagnostic condition **MUST** be the one where X is present and controlled.
- **Where it would live:** M11 (acceptance tests); Phase E arm design.
- **Evidence:** Methods "Affect into choice": "because in single evaluation a model has no access to its prior ratings, a good fit of the feelings-informed model in that condition is not evidence that affect drove choice. The memory condition is more diagnostic" (also S5, S7.2). SHOWN.
- **What it changes:** Inference validity of any "belief/affect drives policy" claim. In EPModel terms: a test that the belief layer drives move selection is only interpretable when the belief is present in the person's state (C9's information-access discipline) and the comparison holds that presence fixed. Complements C9 (what an agent may read) with a positive *test-design* rule (interpret only in the X-present condition).
- **Cost / risk:** Low — a design/reporting discipline, no new state. No conflict with engine purity (M16.B) or determinism.
