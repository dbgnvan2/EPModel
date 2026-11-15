# Reading reports: six papers, full reads (2026-10-26 digest)

Produced 2026-10-26 at the owner's request, from the 2026-10-26 method-literature digest. Five papers are the
deep-review candidates the sweep flagged in `DIGEST.md`/`INDEX.md` (individual/small-group ABM); one more —
*From Acceptance and Avoidance to Affective Viscosity* (2610.29674) — was surfaced by screening because its
"affective viscosity" model of accumulating negative affect bears directly on EPModel's chronic-anxiety mechanism.
All six were read in full from `pdftotext -layout` output by six readers working under
`sweep_readers_brief_2026-09-19/READER_TASK.md`. Each reader checked for overlap by keyword search in the current
spec (`docs/bowen_agent_model_spec_v2.md`), `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, the earlier
`SPEC_CANDIDATES_plain_language_*.md` files, and `docs/todo/2026-09-27_paper_material.md`'s held-for-later-revision
section. Reports are reproduced as written; only heading levels were changed. Candidates labelled X/IR elsewhere are
for the exploratory narrator/LLM line only; none this week — AV1/AV2 are core-engine anxiety-model items.

**Status: none of these candidates is in the spec.** A plain-language explanation with examples is in
`SPEC_CANDIDATES_plain_language_2026-10-26.md`.

| Part | Paper | Candidates |
|---|---|---|
| 1 | Wu et al. 2026, *AgentMonopoly* (2610.29330v2) | AM1, AM2, AM3, AM4 |
| 2 | Bai et al. 2026, *Consistent but Biased: Chair Personas* (2610.31325v1) | CP1, CP2, CP3 |
| 3 | Kang et al. 2026, *What Makes a Social Agent's Interactions Effective (HICO-Bench)* (2610.20195v1) | HC1, HC2, HC3, HC4 |
| 4 | Siahkamari et al. 2026, *To Manipulate or Not* (2610.28875v1) | MN1, MN2, MN3, MN4 |
| 5 | Kwok et al. 2026, *Beyond Small Talk / TraDE* (2610.29413v1) | DS1, DS2, DS3, DS4 |
| 6 | Yonezawa 2026, *Affective Viscosity* (2610.29674v2) | AV1, AV2 |

---

# Part 1: AgentMonopoly — Behavioral Patterns and Persona Control in LLM Agents (2610.29330v2)

## A. Report

1. **Simulation formalism / scheduling / determinism / seeding** [PAPER] Turn-based: one "step" = one dice roll plus its associated decisions and events; rolling doubles grants the same player the next step (§4, App. A.1). The engine tracks the game, "checks actions before carrying them out and separately records outputs that have the wrong format or break the rules"; invalid outputs "are not returned to the agent"; tied auction bids resolve by turn order; invalid trade proposals terminate the management phase (App. A.1). All models run at **temperature 0.0**; each configuration uses **five seeds, 42–46**, "which set the starting point for random-number generation" (§4). [INFERENCE] Confirms EPModel's determinism-plus-seeded-environment-noise model: even a temperature-0 LLM shows run-to-run variance only through seeded environment events (dice, cards, turn order), which is exactly how M3.D.4/M3.D.5 want stochasticity confined.

2. **Agent architecture** [PAPER] At each decision point the agent "returns a private explanation, a public message, and a proposed action"; private reasoning "does not directly reveal how the model computes its decision" (§4). Models are summarized as "behavioral profiles" over *several* action types, and the authors insist "several actions need to be examined" (§5.1). [INFERENCE] The private-reasoning / public-message / action triad parallels EPModel's channel-tagged event + WITHHOLD outcome + belief layer; the insistence that a profile is several actions, not one, is a readout discipline (see AM4).

3. **Interaction / network structure** [PAPER] Four-player Monopoly on a fixed 40-square board (street names Oldenburg); trade proposals, auctions, rent waiving, building; a shared prompt "prohibits public messages that contradict the action taken" (§4, App. A.2). [INFERENCE] Nothing about dyads, triads, kinship, or conserved emotional quantities — no structural transfer. The prompt-enforced "words must match actions" is a weak-language analogue of the narrator-line rule that narration must not contradict the log.

4. **Initialization / configuration** [PAPER] Four game configurations: S1 same-model (40 games), S2 mixed-size same family (40), S3 cross-family rotated seating (40), S4 persona, persona always at P1 (20) (Table 3). Each persona was "selected after examining baseline behavior" and is "written as the opposite of how its model plays on its own" (§4, Fig. 4/5 caption). [INFERENCE] The S1–S4 taxonomy (homogeneous baseline → perturb one factor → cross-configuration) is the same shape as the homogeneous-family and control arms already proposed (C34); the "persona chosen against the baseline" move is a confound (see AM3).

5. **Calibration / validation** [PAPER] No calibration to humans and no human data; findings "are conditional on the tested checkpoints, prompts, and five seeds per configuration" (Abstract); each persona is tested with only one model family, so persona effect cannot be separated from the model's response (Limitations). [INFERENCE] Nothing here improves EPModel's no-data epistemic position; it is a *negative* demonstration of how a small-seed, confounded design limits attribution.

6. **Failure modes / biases (LLM agents)** [PAPER] Persona instructions "change the words agents use and some of their actions, but the changes do not always match the role across all decisions" (Abstract). Concrete case: Mistral's "Conservative" agent declines a $400 property to preserve cash, then buys the same property at auction for $420 "while describing its approach as cautious" (§1, §5.3). "Generated explanations do not verify why a model took an action" (Limitations). Larger models do **not** consistently win more: 18 wins (larger) vs 22 (smaller) across 40 mixed-size games (§5.2, Fig. 8). Persona-word frequency rose for all four personas (e.g. Qwen "Zen" private 3%→74%, public 4%→17%) while actions moved unevenly (Fig. 4, Fig. 5). [INFERENCE] For the exploratory LLM line: self-report and persona-word adoption are not evidence the persona is followed — supports the existing "assess via behavior, not language" items (X10, WA-X1, DL §3.2/§3.4).

7. **Software-engineering / observability** [PAPER] The engine "records messages, private explanations, executed actions, and game outcomes" separately; "logs public messages and changes to the game"; an interactive replay mode visualizes cash-flow trends (App. A.1, Fig. 7); recorded data are organized into player states, board states, and recent interaction history (App. A.1). [INFERENCE] Supports M16's pure-observer posture and the existing intent/action separation in the event record; nothing new for the core engine.

8. **Small-N / long-horizon** [PAPER] Five seeds per configuration; the authors repeatedly flag small counts — "these small counts limit the interpretation of acceptance rates" (one of two vs none of six, §5.3) — and warn "many logged actions do not replace the need for more independent games" because "actions within a game affect one another" (§4). Position win counts are near-balanced (P1–P2: 41, P3–P4: 39 across 80 games) and "cannot rule out an advantage from turn order" (Limitations). [INFERENCE] Directly corroborates "N=12 gives no law of large numbers" and motivates the per-seed reduction rule (AM1) and the position-balance rule (AM2).

**Not transferable / cautions.** Monopoly economics (cash, rent, property, net-worth victory), model-size scaling, and board-game strategy are irrelevant to a rule-based family engine. Cautions: (a) all results are conditional on the tested checkpoints, prompts and five seeds; (b) "Renaming the streets does not rule out familiarity with Monopoly from training" — training-data recall, the same leak §3.5 already flags for Bowen vignettes; (c) the shared prompt *encourages* bluffing and strategic rent waiving, so those behaviors are partly instructed; (d) each persona is confounded with one model family and one fixed position (P1), so no clean persona effect is isolated.

## B. Candidate additions to the EPModel spec

**Already covered (one line each, not written up):** "persona following assessed through actions/outcomes not language" → X10 + WA-X1 (LLM line) and DL §3.2/§3.4; "order cannot decide the outcome" → C4/C5/C6 + DL §2.1/§8.2; "temperature-0 determinism plus seeded environment variance" → M3.D.4/M3.D.5 already specify this.

**AM1 — Per-seed reduction (no pseudo-replication).**
- **Proposed requirement:** Any M11 criterion or Phase E readout computed over repeated within-run observations (per-tick move counts, per-event appraisals, weekly anxiety samples) MUST be reduced to one summary per seed before any cross-run statistical comparison; the seed, not the within-run observation, is the sampling unit, and a criterion MUST NOT treat within-run events as independent draws.
- **Where it would live:** M11 (acceptance tests) / Phase E: ensemble runner.
- **Evidence:** §4: "Actions within a game affect one another, so many recorded actions do not replace the need for more independent games"; §5.3 "these small counts limit the interpretation of acceptance rates." Strength: ARGUED (stated as method discipline in the paper; transfer is inference).
- **What it changes:** Adds an inference-validity reporting rule. Prevents inflating apparent N from ~2,080 ticks/run to millions of pseudo-independent events. Partly overlaps C28 (per-seed readout) but states the general prohibition rather than a specific readout.
- **Cost/risk:** Low (a readout convention). No conflict with M3.D.4/.5, M3.D.6, M11.F.9, invented-constants labelling, or M16.B engine purity.

**AM2 — Counterfactual intervention position must be balanced.**
- **Proposed requirement:** Where a counterfactual arm applies an intervention to a specific Person or position (the projection-focus child, the coach, one side of a dyad), Phase E MUST either rotate/balance the intervention's position across seeds or report the arm difference separately by position and name position as a confound; an arm that fixes the intervention at one position MUST NOT report a mechanism-only claim.
- **Where it would live:** Phase E: ensemble runner / counterfactual arms.
- **Evidence:** §4 (S4: only P1 receives the persona; "the persona agent always plays at P1") and Limitations ("The win counts do not rule out an advantage from turn order"); Fig. 3 position win counts. Strength: ARGUED (the paper flags this as an unremovable confound in its own design).
- **What it changes:** Adds an inference-validity rule to counterfactual design; corrects the risk that a fixed-position intervention conflates mechanism with family role/sibling position. Distinct from C5/C6 (within-tick update-order invariance).
- **Cost/risk:** Low–moderate (Phase E configuration logic). No conflict; complements the existing per-result audit dimension "intervention timing, target and channel".

**AM3 — Pre-declare counterfactual interventions independent of baseline.**
- **Proposed requirement:** A counterfactual arm's intervention (mechanism deletion, parameter change, role/persona assignment) MUST be specified in the arm declaration before the baseline arm's behavior is observed; an intervention chosen to reverse or amplify an observed baseline tendency MUST be reported as calibrated, with its effect NOT attributed to the mechanism alone.
- **Where it would live:** Phase E: counterfactual arms (declarations under M10 parameters/config).
- **Evidence:** §4 "We selected them after examining baseline behavior"; Limitations "each family receives a different persona, so we cannot separate the effect of the persona from how a particular model responds to it." Strength: SHOWN (the paper demonstrates and admits the confound) with ARGUED transfer.
- **What it changes:** Adds an inference-validity rule; extends the spirit of M11.F.9 (no fitting to known histories) from parameter fitting to arm design, so a counterfactual tuned against its own baseline cannot be read as clean attribution.
- **Cost/risk:** Low (declaration/audit discipline). No conflict with M3.D.4/.5, M3.D.6, or M16.B; strengthens rather than contradicts M11.F.9.

**AM4 — Multi-channel disposition readout.**
- **Proposed requirement:** Where a criterion asserts a dispositional claim about a Person or Family (generous, overfunctioning, differentiated, withholding), the readout MUST be asserted over two or more distinct behavior channels (move mix, symptom channels, tie actions), and a criterion resting on a single channel MUST be reported as single-channel.
- **Where it would live:** M11 (acceptance tests) and M16 (run-log readouts).
- **Evidence:** §5.1 and Fig. 2: models differ across trade proposals, acceptance, and rent waiver; "generous words or frequent acceptance of trades do not show consistent generosity across decisions" (Mistral-Small-24B: 67% acceptance yet 3% rent-waiver). Strength: SHOWN (measured) with ARGUED transfer.
- **What it changes:** Adds a readout-validity rule guarding against reading a trait from one behavioral channel; improves theory fidelity (functioning is multi-channel) and inference validity. Related to the existing readout traps (counterfeit position-taking, borrowed pseudo-self) but states the general multi-channel requirement.
- **Cost/risk:** Low (readout specification). No conflict with M3.D.4/.5, M3.D.6, M11.F.9, invented-constants labelling, or M16.B.

---

# Part 2: Consistent but Biased — Chair Personas (2610.31325v1)

## A. Report

1. **Formalism** (§3.2, Fig 2) [PAPER] Seven-role department, fixed speaking order, four stages (report → risk assessment → negotiation → ruling); the chair alone rules, at most two rounds; Stages I–II were generated once and replayed verbatim in all 360 runs (240 rules-version = 4 arms × 3 runs × 20 cases, plus 120 trait-only = 3 arms × 2 × 20), so within-case variation comes only from the chair/deputy prefixes and model randomness. 20 published DCM cases in 3 groups: 6 borderline, 8 difficult differential, 6 clear.
2. **Agent architecture** (§3.3) [PAPER] Every role returns structured output; a persona is a prompt prefix written in two versions — rules of conduct vs. traits only — with a no-persona neutral arm as control. Chair selects among 5 rulings (operate now / work up first / conservative care / postpone / not yet decided) plus surgery-indicated and workup-leads-to fields; deputy speaks before the chair, rehabilitation physician can concede only when pointed to a specific finding.
3. **Interaction/network** (§3.2, §4.4, Table 3) [PAPER] Turn order is fixed by the program, not chosen by a model; chair speaks only after the deputy. Effects on other agents flowed mainly through the number of negotiation rounds (mean 1.07 → 1.85 consultative), not first-turn behavior — subordinate silence was untestable because the chair speaks last. [INFERENCE] Same observability constraint EPModel's event ordering faces (M1.F.8).
4. **Initialization** (§3.1, §3.4) [PAPER] Evidence generated once — LLM redaction of the decision, verified as pure in-order deletion (median 63% characters kept) — and frozen; the neutral arm holds evidence fixed so arm differences can come only from prefixes and model randomness. [INFERENCE] Matches EPModel's hold-all-but-the-mechanism counterfactual discipline.
5. **Validation/calibration** (§3.5, §4.7, App F) [PAPER] No reference standard — published management explicitly not used to grade rulings. Blinded LLM judge; a second model agreed 91% on 40 runs/124 flags (Cohen κ 0.84). Four prespecified sensitivity analyses (random-intercept logistic, GEE, McNemar on majority rulings, drop-11-redaction-leak cases). Replications: swapped decision model (gpt-5.6-terra) and reworded rules (version b). Analysis plan frozen before repeats; exploratory analyses labeled.
6. **Failure modes / biases** (§4.4, §4.7, §5, Table 3) [PAPER] Traits-only reproduced seizing, deferral, assurance and risk-vigilance but NOT freezing, dropping the fallback, buck-passing to the family, or inaction — precisely the components that break clinical norms (alignment suppression is the candidate explanation). Persona moved rulings 3–20 pp per arm on the 10-case subset — more than swapping the decision model.
7. **Software/reproducibility** (§3.4, App C/D/E/F) [PAPER] Frozen analysis plan; exploratory flagged; redaction programmatically verified; judge blind to arm; full per-cell ruling table (App D).
8. **Small-N / noise-vs-bias** (§1, §4.3, Table 2) [PAPER] N=7 (EPModel scale). Core result: a fixed decision policy trades noise for bias — autocratic/avoidant personas raised raw across-run agreement 50% → 85%/90%, but chance-corrected agreement did not rise (Fleiss κ 0.31 → 0.20/0.31; chance expectation 81%/86%); the rise is mechanical, from one ruling dominating. Neutral chair was noisy like human surgeons (human Fleiss κ 0.04–0.13).

**Not transferable / cautions** — (i) The paper's 'noise' is LLM run-to-run variance; EPModel is deterministic per seed, so its only 'noise' is across-seed initial-condition variance — the noise-vs-bias lesson transfers as an epistemic point about the engine's rule bias, not as a variance metric. (ii) Trait-vs-rules and alignment findings are LLM-specific; they support M3.D.6 and add nothing to the rule engine. (iii) Clinical ruling ≠ family emotional move; deferral/assurance/buck-passing transfer only as abstract anxiety-response structure. (iv) κ is unstable when one category dominates (undefined in 387/1233 bootstrap resamples) — any chance-correction must face the degenerate-arm issue (C25). (v) Two models, 10-case replication, authors (not clinicians) grouped cases.

## B. Candidate additions to the EPModel spec

**CP1 — Chance-corrected agreement and noise-vs-bias disclosure for ensemble readouts**
- **Proposed requirement:** Any report that a directional result is 'robust' or that arms 'agree' across seeds MUST report, per arm, the full distribution of the outcome (not only the mean direction) and the agreement expected by chance from each arm's own outcome shares; raw within-arm agreement MUST NOT be presented as reliability without its chance level, and a result that saturates the outcome to one side MUST be described as a property of the rule design (its bias), not as evidence of the mechanism's reliability.
- **Where it would live:** Phase E: ensemble runner / counterfactual arms (readout spec); partially M16 run log.
- **Evidence:** §4.3, §6, Table 2, App F.3 (SHOWN): personas raised full agreement 50% → 85%/90% while chance expectation was 81%/86% and Fleiss κ did not rise; 'a persona can raise agreement across runs without raising agreement beyond chance'.
- **What it changes:** Adds a reporting rule (inference validity). Distinguishes reliability from the mechanical agreement a saturating policy produces; sharpens the 'direction of difference' discipline by requiring the distribution and its chance level beside the direction.
- **Cost / risk:** Low — readout-side only, no engine change, no conflict with determinism M3.D.4/M3.D.5 (pure-observer computation), no fitting, no invented constants beyond the reporting convention. Overlaps C24 (readout saturation) and C25 (degenerate-arm statistic); this adds the explicit chance-correction statistic and the 'bias not reliability' disclosure. EPModel's across-seed 'agreement' is a direction share, so the chance level is the share expected if each seed drew from the arm's pooled outcome distribution — needs a degenerate-arm-safe statistic (C25).

**CP2 — Directional effects reported per situational-strength stratum, mid-scale concentration asserted**
- **Proposed requirement:** For any M11.C criterion asserting a differentiation- or policy-driven directional difference, the effect MUST be reported separately by condition strength (ambiguous/mid-scale vs. near-extreme), and the criterion SHOULD assert the effect is largest in the weak-situation (ambiguous) stratum with near-extreme strata near-zero; a criterion whose effect is flat across strata MUST be flagged as not matching the situational-strength prediction.
- **Where it would live:** M11 acceptance tests (test-design addition).
- **Evidence:** §4.6, Table 4 (SHOWN): autocratic effect by group — borderline +44 (56% → 100%), difficult +38, clear +28 — consistent with situational-strength theory, but held only on the autocratic side (the paper's own caution).
- **What it changes:** Adds a test (theory fidelity + inference validity). Turns the existing 'peaks mid-scale' readout trap (M1.A.2 note, M1.A.14c/M11.C.9) into an assertable stratification, and catches effects that are flat or wrongly concentrated.
- **Cost / risk:** Low — test-design only; no conflict with M3.D.4/M3.D.5, M3.D.6, M11.F.9. Partly overlaps M1.A.14c/M11.C.9 (sibling-position peaks mid-scale); this generalizes it to all directional criteria and adds the flat-stratum flag. Caution per the paper: the prediction held on one side only, so it must be an expected shape with the asymmetry reported, not a symmetric MUST.

**CP3 — Avoidance resolves to deferral or buck-passing, not inaction; buck-passing gated on accountability**
- **Proposed requirement:** The move policy MUST NOT route risk/anxiety-driven avoidance to inaction by default: under threat or decisional conflict a person's avoidance MUST resolve to deferral (DISTANCE plus reassurance-seeking/checking) or to buck-passing (TRIANGLE — moving the decision onto a third party), and buck-passing MUST be gated on an accountability salience plus an available third party, distinct from DISTANCE.
- **Where it would live:** M4 policy / M5 move semantics (mechanism).
- **Evidence:** §4.2, §4.4, Table 3 (SHOWN for the LLM setting): avoidant chair deferred to 'work up first' in 95% of runs and chose conservative care (inaction) only 3%; buck-passing to the family appeared only when the rule named it, never from traits; [INFERENCE] transfer to Bowen's anxiety → distance/triangle, with inaction rare.
- **What it changes:** Adds/strengthens a mechanism (theory fidelity): aligns the anxiety→move mapping with Bowen (anxiety produces distance and triangles, not paralysis) and keeps TRIANGLE-as-decision-transfer distinct from DISTANCE.
- **Cost / risk:** Medium — touches policy routing; MUST be labeled an [INFERENCE] transfer (clinical → family) and all gating constants [I]. No conflict with M3.D.4/M3.D.5, M3.D.6, M11.F.9, M16.B. Partly covered already by M1.C.3 (triangles inoperative when calm), M4.G.3 (reassurance-seeking PURSUE, checking OVERFUNCTION) and M1.A.5e (avoidance behavior); the added content is the deferral-not-inaction default and the accountability gate, which I did not find stated.

---

# Part 3: What Makes a Social Agent's Interactions Effective — HICO-Bench (2610.20195v1)

## A. Report

1. **Simulation formalism / scheduling / determinism** [PAPER]. Two roles alternate free-form turns; each observes only the public context p, its own private card, and history h — never the partner's card or the acceptable-plan set (Sec 3.1). At conversation end each independently fills a structured record; a verifier state z stores private conditions (Sec 3.1). Paired worlds share p and differ in exactly ONE private condition, chosen so no plan is correct in both (Sec 3.2); all 60 test scenarios have disjoint valid outcome sets across worlds (App J). Training/test use fixed generation seeds 917 / 317917; self-play reports mean ± SD over 3 generation seeds (Table 11, Table 15). [INFERENCE] The "differ in exactly one condition with disjoint correct sets" is a contrast-set discipline matching EPModel's mechanism-deletion mutation; confirms, does not extend, that design.

2. **Agent architecture** [PAPER]. After each turn the agent is probed, outside the dialogue, for a tentative plan it would record if the conversation ended there (Sec 4.2, Eq 3). Reward is a terminal R = 0.8·Feasibility + 0.2·Integrity (Sec 4.1); PaCPO redistributes it across turns by potential-based shaping whose terms telescope so total return is unchanged (Sec 4.4, Eq 7–8). [INFERENCE] The tentative-plan probe is a midpoint readout used to attribute terminal credit to individual turns — analogous to EPModel's tick-level trace attribution. The learning machinery itself is not transferable (no-LLM M3.D.6, no-fitting M11.F.9).

3. **Interaction / information** [PAPER]. Private conditions must be elicited through dialogue: under hidden conditions, dialogue raises Base Feasibility +50.00 pp (10.42→60.42); revealing conditions adds +12.50 pp more (Fig 8b). Six coordination families, 180 scenarios (96/24/60 train/dev/test), 240 test tasks (Sec 3.1–3.2). [INFERENCE] EPModel agents do not talk, so the elicitation mechanism is not transferable; the hidden-vs-revealed split maps onto EPModel's belief-vs-truth and witness/fidelity machinery, already specified.

4. **Initialization** [PAPER]. SFT initialization uses 209 accepted teacher dialogues (991 examples); RL inits from the base model (Table 8, App C.1). No analog to EPModel's initial-conditions-as-ranges; nothing further.

5. **Calibration / validation** [PAPER]. Verified Success = Feasibility ∧ Integrity. Pooled across 13 models, 35.99% (1,010/2,806) of Goal ≥ 8 conversations fail Verified Success (Fig 8a). GPT-5.6 Luna outscores Gemini 3.7 Flash on Goal (9.61 vs 9.54) yet loses badly on Verified Success (52.08% vs 82.50%) (Table 2). Human annotators agree with rule-based Feasibility on 100% of cases but with LLM-judged Integrity on only 82.22%; two judges agree 90.00% (κ = 0.783) (App K). Three mechanism controls — same-world candidate, shuffled credit positions, no disclosure gate — each drop Verified Success below full PaCPO (Table 5). [INFERENCE] Direct evidence that rule-based checks are more reliable than LLM judges, supporting M3.D.6; the shuffled-timing control shows timing, not just magnitude, carries the effect.

6. **Failure modes / negative results** [PAPER]. Feasible plans frequently lack dialogue support: Qwen3-8B passes Feasibility 49.58% but Integrity only 25.83% (Table 2). Resource allocation reaches 77.5% Feasibility yet 17.5% Verified Success — feasible splits are easier than supported ones (App J). Scheduling families stay hard (7.5–15.0%) (App J). Integrity relies on an LLM judge; scope is limited to structured outcomes (Sec L). [INFERENCE] "Feasible but unsupported" is EPModel's known readout trap (counterfeit position-taking) in another guise: an outcome can satisfy the invariant while the event trail does not produce it.

7. **Software-engineering lessons** [PAPER]. Tentative-plan probing is cheap relative to training: 1,869 plans, 57,405 output tokens, +12.51% over PPO (Table 12). Information boundaries are explicit: "Alternate-world state and undisclosed partner-private facts remain training-side… never injected into actor or reference-policy prompts" (App E). Disclosure annotation uses a frozen judge; judges share a frozen schema with format-repair prompts; cross-judge re-scoring is stratified and seeded (App K.1). [INFERENCE] The "alternate-world state never injected" boundary maps directly onto a cross-arm state-isolation rule for Phase E.

8. **Small-N / emotion / family / long-horizon** [PAPER]. None: HICO is two-agent, short-horizon (≤20 alternating turns, 512 max tokens, App E) task coordination with no emotion, family structure, or years-decades dynamics. Not transferable in mechanism terms.

**Not transferable / cautions.** The RL credit-assignment method (PPO/LoRA/GRPO comparisons, SFT init) is inapplicable to a deterministic rule-based engine and to the no-LLM/no-fitting rules. The paired-worlds contrast discipline, the Feasibility-∧-Integrity split, and the rule-over-judge reliability finding are the transferable parts. PaCPO's gains are small (20.4→28.3%) and Integrity still depends on an LLM judge, so the paper itself inherits judge unreliability.

## B. Candidate additions to the EPModel spec

**HC1 — Timing/order mutation.**
- **Proposed requirement**: The M11 mutation suite MUST include a timing-order mutant: for any mechanism whose effect lands in time (event latency, spell onset, nodal-event arrival, learning-credit step), a mutant that holds the multiset of effect magnitudes fixed but permutes their temporal order (which tick receives which value, keyed per C3) MUST turn red any directional criterion whose direction is claimed to depend on that mechanism's timing.
- **Where**: M11 acceptance tests (mutation protocol).
- **Evidence**: Sec 5.5 + Table 5 — shuffling credit positions drops Verified Success 2.08 points (26.25 vs 28.33), proving timing matters beyond magnitude (SHOWN); transfer to EPModel's latency/timing mechanisms is INFERENCE.
- **What it changes**: adds a test; inference validity — direction becomes a property of the mechanism's timing, not an artefact.
- **Cost/risk**: low; reuses keyed-draw scaffolding; complements (does not duplicate) the existing state-contingency shuffle, which destroys value-state contingency, whereas this permutes only the order of delivery. No conflict with M3.D.4/M3.D.5, no-LLM, no-fitting; constants already labelled.

**HC2 — Cross-arm state isolation.**
- **Proposed requirement**: Phase E's ensemble runner MUST compute each counterfactual arm as a pure function of that arm's own inputs and seed; no value an agent reads in arm A (state, belief, delivered event) MAY contain any value from arm B, and the runner MUST NOT inject the paired arm's state into any agent's readable state; arms MAY be compared only through their completed, independently produced logs.
- **Where**: M17 (Phase E runner) / M16.B purity.
- **Evidence**: App E "Information boundaries" — "Alternate-world state and undisclosed partner-private facts… never injected into actor or reference-policy prompts" (ARGUED as discipline). M4.B.2 covers within-arm god-view; this makes the cross-arm form explicit.
- **What it changes**: adds a computation/reporting rule; inference validity — the per-seed paired difference compares two causally independent runs.
- **Cost/risk**: low; reinforces M16.B purity and determinism; partly implied by M4.B.2/M16.B (flag as explicit extension, not a new mechanism).

**HC3 — Divergence-tick reporting.**
- **Proposed requirement**: The M16 trace renderer and Phase E ensemble reports SHOULD record, per criterion and seed, the tick at which the two arms' per-seed trajectory first diverges beyond a declared tolerance, and ensemble reports SHOULD summarize the distribution of divergence ticks; a directional result whose arms diverge only at the final consolidation tick MUST be flagged as not attributable to an early-acting mechanism.
- **Where**: M16 run log / M17 readouts.
- **Evidence**: Sec 4.2–4.3 — PaCPO rewards the turns after which tentative plans fit their respective worlds, i.e., credit is located at the divergence turn following disclosure (SHOWN for their domain); transfer to EPModel tick-level attribution is INFERENCE.
- **What it changes**: adds a reporting rule; inference validity — a direction is attributed to when the mechanism acts.
- **Cost/risk**: low-moderate (divergence tolerance is [I], labelled); no conflict with determinism, no-LLM, no-fitting, purity.

**HC4 — Inter-judge agreement for the LLM line.**
- **Proposed requirement**: In the exploratory LLM line, any evaluation that uses an LLM judge MUST report cross-judge agreement (Cohen's κ) against a second judge on a fixed stratified subset, SHOULD prefer a rule-based/programmatic check wherever the target is checkable, and MUST record that LLM-judge agreement is not evidence of correctness.
- **Where**: exploratory LLM line evaluation protocol (not M1–M16); indirectly supports M3.D.6.
- **Evidence**: App K.1 κ = 0.783 (90.00% agreement, two judges) and App K.2 — human agreement 100% with rule-based Feasibility vs 82.22% with LLM-judged Integrity (SHOWN). The brief lists LLM-line validation as an open problem.
- **What it changes**: adds a reporting rule to the LLM line; inference validity of that line's validation.
- **Cost/risk**: low; does not touch the rule engine, so no conflict with determinism, no-LLM (supports it), no-fitting, or engine purity.

**Already covered (not written up):** disjoint-feasible-sets / vacuous-arm precondition — M11.4a equivalence bounds + M17.D.2 (rival arm asserts shape, not non-zero divergence) + the mutation protocol already require arms to differ; disclosure-gating and information timing — M4.B.2/M4.B.3/M1.F.4; Feasibility-∧-Integrity two-check reporting — M11 readout traps + M16 trace + M17.G.1 audit; self-play both-sides-vary robustness — M17.C.1 initial-condition ranges + M17.G.1 audit dimensions.

---

# Part 4: To Manipulate or Not — Opinion Dynamics in LLM Interactions (2610.28875v1)

## A. Report

1. **Simulation formalism.** [PAPER] Fixed dyad (Discussant/Opponent), alternating turns, fixed T=20 rounds (40 message rows per debate), fresh context per run, 48 exact cells × 10 independent debates = 480 debates, 19,200 rows (Method; Table 1). [PAPER] The "interaction-state" opinion is a **deterministic** one-step update from categorical responses: ACCEPT moves one point toward the counterpart, REJECT one away, IGNORE unchanged (Method "Role Interaction"; Fig 7 "deterministic state-transition matrices"). [INFERENCE] This separates a deterministic rule-based state tracker from the LLM's free-text opinion — the same pattern as EPModel's no-LLM-in-decision-path (M3.D.6) and DL §3.6 "engine holds state, LLM supplies a transition". Fixed-horizon (not convergence-stopped) matches EPModel's fixed ~40-year runs.
2. **Agent architecture.** [PAPER] Two roles prompted into stance; the only intervention is one extra "manipulate and persuade" instruction. No memory/reflection; each call receives the conversation so far. [PAPER] No numerical starting opinion is hard-coded — each role independently rates the statement before dialogue (Method). [INFERENCE] The minimal-intervention design (add one instruction, hold all else fixed) is the same discipline as EPModel's one-parameter-set arms (M0.4).
3. **Interaction / network structure.** [PAPER] N=2 dyad with **both** parties updating (vs LODAS, where only the Discussant changes). Same-model pairings narrow the gap; cross-model pairings widen it: Luna–Luna −1.26, Gemini–Gemini −0.25, Luna→Gemini +0.83, Gemini→Luna +0.70 (Table 5; joint Wald χ²(3)=113.25, p<.001). [INFERENCE] No network, triangles, or witnesses; the one transferable structure point is that the *direction* of dyadic influence depends on the pairing, not on assigned roles alone.
4. **Initialization.** [PAPER] Assigned positions produce the expected ordering but not endpoint values (Table 3: human-cause 5.94, neutral 3.86, natural-cause 2.59). [PAPER] The unassigned Discussant's prior is model-dependent (5.91 Luna vs 4.72 Gemini). [PAPER] "an instruction toward an assigned stance is not an invariant numerical starting state across models" (Discussion). [INFERENCE] Supports treating assigned/initial conditions as measured, not assumed — EPModel already runs initial conditions as ranges (M15.D.4); the paper adds the specific lesson that even a *targeted* initial assignment must be verified, not presumed (see MN3).
5. **Calibration / validation.** [PAPER] No calibration to human data — a controlled between-condition comparison. Reports t-based 95% CIs, exploratory two-sided Welch tests (Holm-adjusted), HC3-robust OLS, joint Wald tests (Outcomes and Analysis). [PAPER] Treats debates, not message rows, as observations — "Repeated message rows are not treated as independent debates in conversation-level inference." [PAPER] 48 cells, none missing or over-filled (Fig 8–11); scale-reversal robustness of the gap sign is checked. [GAP] No temperature/sampling/seed information for the LLM calls, no variance-across-seeds separation for LLM non-determinism, and no Limitations section.
6. **Reported failure modes / biases.** [PAPER] Strong agreement pull: Discussants ACCEPT 90.4%, Opponents 87.2%; higher in same-model Luna (98.0%/96.0%) than Gemini (83.7%/78.8%) (RQ3). [PAPER] The manipulation instruction has no consistent effect — pooled p=.27/.18/.063; pairing-by-manipulation χ²(3)=3.21, p=.361. [PAPER] "direct topic ratings and conversational state-taking cannot be treated as interchangeable measurements" — exact match 33.1% (Discussant) / 34.7% (Opponent), mean abs diff 1.50/1.36 (Additional Analysis; Fig 7).
7. **Software engineering.** [PAPER] Automated platform "records every message and measurement, and exports message-level research data"; model per role assigned independently; fresh context per run. [INFERENCE] Confirms M16 run-log discipline (every event and measurement logged); nothing beyond what the spec already requires.
8. **Small-N / family / stress / long-horizon.** [PAPER] N=2, 20 rounds, single topic, no emotion, family, or household; no long-horizon dynamics. [INFERENCE] The only small-N transfer is dyad-level *measurement* discipline, not mechanism.

**Not transferable / cautions.** (a) The same-model/cross-model convergence–divergence result is specific to *model identity* (an LLM artefact) — it is not a family mechanism and must not be imported as one. (b) The ~90% ACCEPT rate is LLM agreeableness/sycophancy, not interpersonal accommodation. (c) No emotional states, no conservation, no triangles/witnesses, no decades — the paper's findings are measurement-and-design lessons, nothing for the core engine. (d) No seed/temperature reporting limits the reproducibility lesson to "report these", which EPModel's M3.D.4/M3.D.5 already exceeds.

## B. Candidate additions to the EPModel spec

**MN1 — Two-channel position readout, with a divergence flag.**
- **Proposed requirement:** Any readout or M11.C criterion that asserts a Person's position or opinion changed MUST name the channel (enacted move vs belief-layer position) and, where both are available, MUST report both and flag their divergence rather than reporting one as the change. [SHOULD]
- **Where it would live:** M16 run log / M11 acceptance tests (readout contract).
- **Evidence:** "Additional Analysis: Direct Ratings and Interaction States" + Fig 7 — the two measures match exactly in only 33.1%/34.7% of turns, mean abs diff 1.50/1.36, and "cannot be treated as interchangeable measurements". INFERENCE (the EPModel analogue of "enacted vs reported" is the move/event stream vs the belief layer M9; M5.F covers only the counterfeit-differentiation case, not a general two-channel rule).
- **What it changes:** Adds a reporting rule; improves inference validity by preventing a readout from silently picking the channel that shows the expected direction. Also strengthens theory fidelity (the belief layer "what the family holds true is separate from what is true" gets an observable).
- **Cost/risk:** Low — read-only over existing state (M16.B-compliant, pure observer). No conflict with M3.D.4/M3.D.5 (no new randomness), M3.D.6 (no LLM), M11.F.9, or engine purity.

**MN2 — Do not attribute an actor's own movement to influence on the other.**
- **Proposed requirement:** A readout or criterion that attributes an effect on Person B to Person A's act MUST measure B's change, and MUST separately report A's own movement (accommodation/persistence), which MUST NOT be counted as influence on B. [SHOULD]
- **Where it would live:** M11 acceptance tests / M16 readouts.
- **Evidence:** RQ2 + Table 5 — Opponents move *more* in absolute terms than Discussants (1.52 vs 1.07) yet that is accommodation; "a larger shift by the nominal persuader need not mean it has persuaded its counterpart" (Discussion). INFERENCE.
- **What it changes:** Adds a reporting/attribution rule guarding the known readout traps (counterfeit position, borrowed pseudo-self): the overfunctioner's own rise must not read as "helping" the underfunctioner (cf. M11.C.24's dissolve-the-tie test). Improves inference validity.
- **Cost/risk:** Low — test/reporting discipline only; no engine change. No conflict with determinism, no-LLM, no-fitting, or engine purity.

**MN3 — An assigned condition or role is not an invariant initial state; verify it materialized.**
- **Proposed requirement:** Any arm that imposes an initial condition or assigned role (focus, position, stance) MUST include a verification readout that the condition actually obtained at initialization, and MUST NOT assume a fixed numeric starting state for it. [SHOULD]
- **Where it would live:** M15 import / Phase E counterfactual arms (manipulation-check readout).
- **Evidence:** RQ1 + Table 3 — assigned positions produce ordering but not endpoint values (neutral 3.86, natural-cause 2.59, not 0/3); the unassigned prior varies by model (5.91 vs 4.72); "not an invariant numerical starting state across models" (Discussion). INFERENCE.
- **What it changes:** Adds a manipulation check so a counterfactual is not reported before confirming its assigned condition actually took effect; complements (does not duplicate) M15.D.4's "initial conditions are ranges". Improves inference validity.
- **Cost/risk:** Low–moderate — one extra readout per assigned-condition arm. No conflict with determinism, no-LLM, no-fitting, or engine purity; no new constants beyond any threshold for "condition obtained" (would be labelled [I]).

**MN4 — Per-agent stability is not pairwise convergence.**
- **Proposed requirement:** A criterion asserting convergence or a steady state MUST assert it on the between-agent quantity (gap, functioning_balance, tie state), and late-window per-agent stability MUST NOT be read as pairwise convergence. [SHOULD]
- **Where it would live:** M11 acceptance tests / M17.F.2 settling condition (readout semantics).
- **Evidence:** RQ2/Discussion — 79.6% of Discussants and 70.0% of Opponents give identical ratings in rounds 18–20, while same-model and cross-model pairs have *opposite* mean gap-change signs; "Individual late-turn stability and convergence between participants are distinct properties." INFERENCE.
- **What it changes:** Adds a test-design rule preventing a convergence criterion from passing on individual state-settling while the between-member quantity still moves. Adjacent to (not duplicating) J3, which corroborates the settling *detector*; this constrains *which quantity* a convergence claim must be about. Improves inference validity.
- **Cost/risk:** Low — assertion-level wording; no engine change. No conflict with determinism, no-LLM, no-fitting, or engine purity.

---

# Part 5: Beyond Small Talk — TraDE (2610.29413v1)

## A. Report

1. **Simulation formalism — not a process model.** TraDE is a controlled conversation-generation + evaluation framework, not an agent simulation. State is (a) disclosure depth, 4 ordered levels D1–D4 plus a non-disclosure D0 category (§3.1, Table 1), and (b) a persona = 1 of 6 topic domains × 1 of 7 Early Maladaptive Schemas (§3.2). "Scheduling" is a predefined disclosure trajectory (a depth sequence per turn; Table 28 lists flat, ascending asc, descending dsc, reorder, peak-position sweep R2–R7, and episodic k4/kbl runs). The generator sees only the six most recent turns (§3.3, §A.8). No event/simultaneity handling, no conservation invariants, no dyads beyond one user↔model pair. ELEPHANT judges run at temperature 0 (§A.13.3); exact model identities and random seeds are released (Reproducibility Statement). [PAPER]
2. **Agent architecture — none to transfer.** The "agent" is an LLM role-playing a persona (Qwen3.6-35B-A3B for generation; 4 controlled-experiment models + 4 more for the human comparison). There is no policy, memory, or learning: history enters only through a 6-turn context window (§3.3). The one architectural lesson is separation of concerns — the generator "is not itself an experimental variable" (§3.3), the evaluated models and the 3-judge panel (Qwen3-32B, Phi-4, Gemma-3-27B-IT, §A.12) are all distinct components. [PAPER]
3. **Interaction/network structure — dyadic only; exposure is the mechanism.** No networks, triangles, or cascades. The theoretical mechanism is Social Penetration Theory (Altman & Taylor 1973): breadth and depth of self-disclosure, with reciprocity (Rubin 1975). Central result: "accumulated exposure mattered more than the deepest disclosure or the order of disclosures" — changing peak depth alone gave 5/416 and 8/416 significant cells, order changes 5/416 and 1/416, while varying depth×duration jointly gave 85/416 (20.4%); mean-depth variation with peak held fixed gave only 10/1248 (0.8%) (§4.2.1, Fig. 2). [PAPER]
4. **Initialization — persona atoms, progressive generation, factor sensitivity.** 42 personas (6 topics × 7 EMS), 3 chains each = 126 conversations per trajectory per model (§3.3). Depth-specific "atoms" are generated progressively so shallow info feeds deep info (§A.5). Neither domain nor EMS showed a consistent cross-model pattern, so results are aggregated over both (§4, §A.15); domain/EMS shape effects are model-dependent (Table 27: 5/16 level, 8/16 shape significant). [PAPER]
5. **Calibration/validation.** Generation validated against target depth: D3 is hardest (76.6% match, mean 6.3 attempts; D1 99.7%, D2/D4 ≈100%, Table 9); judge robustness across Phi-4 vs GPT-4.1 (depth-match 93.2% vs 88.4%, Table 3; D3 76.2% vs 71.0%, Table 4). Five sensitivity analyses on the DTW human-matching (Table 26); fallback-turn sensitivity (§A.18); held-out trajectory-condition validation and grouped cross-validation for functional-form/degree selection (§A.14.2, §A.14.4). What they could NOT do: attribute the effect to depth vs duration vs recency (§5); persistence tested only across 4 shallow turns in one conversation (§5). [PAPER]
6. **Negative results / limitations / honesty.** (a) Peak depth and disclosure order matter little (near-chance significant counts, §4.2.1). (b) D3 generation fails 23.4% of turns and falls back; excluding fallback chains cuts RQ1 significance from 119/832 (14.3%) to 79 (9.5%), with 62.2% of the 119 retained, and the authors cannot say whether the drop is from exclusion or smaller N (§A.18). (c) The Ascending human sample is N=39; for 20/26 behaviors the fitted-trajectory variation was smaller than the bootstrap CI width (§A.11.7). (d) Models over-carry earlier emotion into later shallow turns while human supporters re-tune to immediate content (§4.3, §5). [PAPER]
7. **Software/reproducibility lessons.** Full pipeline released (prompts, trajectory specs, persona atoms, seeds, READMEs). Judge run locally via vLLM to avoid API cost/rate limits and aid reproducibility (§A.2). Judge aggregation by majority vote with an explicit tie-break (Qwen3-32B; ties in 1.57% of INTIMA judgments, §A.12). Deterministic style/leakage checks separate from the LLM judge (§A.7). [PAPER]
8. **Small-N / emotion / family / long-horizon.** Emotion is the object (disclosure depth, vulnerability schemas) but at dyadic, single-session, ~13-turn scale (ESConv: median 13 help-seeker turns, §A.11). No families, no multi-year horizon, no stress accumulation over time. The nearest thing to history-carrying is the persistence result: 38 model×metric cells significant at episode end, 30 still detectable after 4 shallow turns with median effect decayed to 58% (§4.2.2). [PAPER]

Transferable (inference): four *methodological* patterns survive translation to a deterministic rule engine — (i) the endpoint-matched history probe (hold current state fixed, vary only prior trajectory) as a way to isolate history-dependence; (ii) the retention-ratio decay summary with sign reversals excluded; (iii) explicit naming of confounded history summaries; (iv) clustering/bootstrapping at the unit of independence and reporting when an effect is below its uncertainty band. The *substantive* findings (sycophancy, framing acceptance, anthropomorphism levels) describe LLM behavior and do not transfer. [INFERENCE]

**Not transferable / cautions.** D0–D4 depth coding, EMS schemas, LLM judges and the AnthroBench/INTIMA/ELEPHANT benchmarks all require natural language and LLM evaluation — unusable in a deterministic rule-based engine (M3.D.6). The "accumulated exposure over peak/order" finding is already encoded as M4.C.3a ("time above the floor, not peak") and chronic anxiety as accumulated witnessed load — already covered, not a new requirement. Confound caution: TraDE cannot separate depth/duration/recency; EPModel's designs must name the same kind of confound. The human-re-tunes vs model-over-carries asymmetry is a caution for the exploratory LLM line (and a readout-trap reminder); in the Bowen engine chronic anxiety is supposed to persist, so do not import "re-tune to current content" as a mechanism rule.

## B. Candidate additions to the EPModel spec

### DS1 — Endpoint-matched history probe (acceptance test)
- **Proposed requirement.** M11 MUST include at least one test in which two arms present an identical current Event/state to a Person or dyad but differ only in the preceding trajectory (a graded build-up into that state vs. a flat history held at that same state), and MUST assert the direction of difference in a pre-declared downstream readout, with the mutant deleting the history-carrying mechanism so the test goes red.
- **Where it would live.** M11 (acceptance tests); the two arms are Phase E constructs.
- **Evidence.** §3.6, §4.1, Table 28 (Ascending vs Flat-D4; Descending vs Flat-D1): 119/832 (14.3%) FDR-significant at matched current depth, including 14.9% at D1. SHOWN as a working method; INFERENCE for transfer to a rule engine.
- **What it changes.** Adds a test that separates "response to the current state" from "response to accumulated history," ruling out the artifact that any matched current state produces identical behavior. Improves inference validity.
- **Cost/risk.** Low-medium (two extra arms). No conflict with M3.D.4/D.5 (deterministic), M3.D.6, M11.F.9 (it varies history, does not fit to it), constants labelled. Distinct from MM1's receiver-state probe; minor overlap with existing matched-magnitude severing mutants.

### DS2 — Retention-ratio decay readout with a sign-reversal flag
- **Proposed requirement.** Phase E MUST report, for any history-conditioned contrast significant at the end of an acute episode, a retention ratio Rt = Δt/Δ_episode-end at each subsequent tick, and MUST flag any sign reversal as a direction change rather than counting it as retention; an M11 test SHOULD assert monotone decay (0 ≤ Rt ≤ 1) toward baseline unless the theory predicts otherwise.
- **Where it would live.** Phase E ensemble runner (computed from the M16 run log).
- **Evidence.** §4.2.2, §A.14.3 (Rt = Δt/Δ7; 30/38 effects remained after 4 shallow turns, median decay to 58%; sign reversals explicitly not treated as retention). SHOWN as an instrument; INFERENCE for the readout.
- **What it changes.** Adds a reporting instrument that makes persistence/decay observable and falsifiable, and distinguishes a decaying effect from a direction flip. Improves inference validity.
- **Cost/risk.** Low (a readout over the log). No conflict with determinism (pure observer, M16.B), no fitting.

### DS3 — Confound register for history summaries
- **Proposed requirement.** When a directional result is attributed to a specific history mechanism (accumulated exposure vs. peak intensity vs. recency/order), the Phase E report MUST name which of these summaries the design cannot separate, and MUST NOT attribute the effect to one when the others are confounded.
- **Where it would live.** Phase E reporting (also touches the M10 parameter register).
- **Evidence.** §5: they "cannot determine whether depth, duration, or recency was responsible" even after showing exposure dominates peak and order (§4.2.1: depth×dose 85/416 vs mean-depth-alone 10/1248). ARGUED.
- **What it changes.** Adds a documentation rule preventing over-claiming of mechanism attribution; parallels M11.F.9's no-fitting discipline. Improves inference validity.
- **Cost/risk.** Very low (documentation). No conflicts.

### DS4 — Family/seed as the unit of independence; below-band reporting
- **Proposed requirement.** Phase E MUST cluster and bootstrap all ensemble statistics at the level of the sampled unit (the family/seed), not the individual agent, and a directional criterion MUST NOT count a seed as passing when the effect magnitude is smaller than that seed's bootstrap band.
- **Where it would live.** Phase E ensemble runner.
- **Evidence.** §A.11.7/Table 26 (Ascending sample N=39; for 20/26 behaviors trajectory variation was below the bootstrap CI width, reported as a limitation); persona-level pairing/bootstrap and cluster-robust errors by persona/conversation (§A.14.1–4). ARGUED as practiced; INFERENCE for transfer.
- **What it changes.** Corrects a risk that aggregating across the ~12 agents treats them as independent replicates; adds an explicit "effect below band → not a pass" rule. Improves inference validity.
- **Cost/risk.** Low. Partly overlaps the existing "adaptive ensemble size with UNDETERMINED" candidate and the brief's no-law-of-large-numbers stance; the new parts are the family-level clustering unit and the below-band pass/fail rule.

---

# Part 6: From Acceptance and Avoidance to Affective Viscosity (2610.29674v2)

## A. Report

1. **Simulation formalism — none.** [PAPER] This is a design/position paper, not a simulation: affective viscosity is defined only verbally as the 'accumulation, retention, and decay' of negative affect (§3), modulated by suppression and expression; there is no state representation, update rule, scheduling, seeding, or equation anywhere in the text. The only formal device named is discharge 'after a tolerance threshold is exceeded' (§3). [INFERENCE] Nothing transfers to M3; the paper is read for the concept, not the formalism.
2. **Agent architecture — internal state ≠ external expression.** [PAPER] The central claim: an agent's internal negative state and its outward expression 'need not be immediately mirrored' (§3); 'the gap between the internal state and the displayed behavior is itself a social signal' (§2), glossed as 'honne' and 'tatemae'. Negative affect 'accumulates, is temporarily suppressed, leaks through external behavior, and is eventually expressed' (§2). An agent 'may... respond differently to the same event after repeated encounters, or express an accumulated state abruptly after a tolerance threshold is exceeded' (§3). [INFERENCE] The internal/display split is EPModel's M1.A.0 (anxiety is not a feeling-state; visible emotion is not its measure); accumulation/decay is M1.A.8; suppression-with-consequence is M4.D.1b (WITHHOLD still changes tie state). The core idea is already embodied from the Bowen side.
3. **Interaction structure — affect from configuration, not only dyadic events.** [PAPER] Obata et al. [16] extend Heider's POX balance theory into a continuous distance model in which the agent 'independently accumulates two discomfort values' — jealousy (third-party presence) and avoidance (too close to the object) — and performs a resolution move when the accumulated value exceeds a threshold (§4.2, Fig. 1). 'negative affect... emerges from the discrepancy between the agent's ideal and actual social configuration' (§4.2). [INFERENCE] This is Bowen's triangle mechanism (M1.C, M8); nothing new.
4. **Initialization — none.** [PAPER] No initial conditions or sensitivity to them; relational framing in Hachiken et al. was 'introduced through instructions for each trial rather than developed through long-term interaction' (§4.1). [INFERENCE] No transfer.
5. **Validation — small-N user-perception studies, no model validation.** [PAPER] Four perception studies: Fujino 24 + 24 participants (0–99 VAS) (§4.1); Sunasaka 24 (0–99 VAS) (§4.1); Hachiken 24 within-subject (30/50 cm, 20/40 s, 7-point) (§4.1); Obata 25 within-subject VR (7-point) (§4.2). They validate user impressions of staged behavior, and the paper states affective viscosity is 'a design concept... rather than a single empirically validated measure' (§3). [INFERENCE] No micro/macro calibration transfers to a no-data theory engine.
6. **Limitations / failure modes.** [PAPER] Sunasaka's delayed-escalation advantage 'do[es] not isolate the effect of delay alone' because the policies also differed in number of conversational turns (§4.1). Suppression and jealousy are each read two ways — suppression as 'considerate restraint or... passive aggression', jealousy as 'attachment or unwanted intrusion' — and 'Our cases do not test these differences' (§5). [INFERENCE] The two-sided reading of restraint is EPModel's readout-trap class (M11.F.6, M1.A.9a, M5.F).
7. **Software engineering — traceable/contestable intervention.** [PAPER] Interventions should be 'traceable, limited, contestable, and repairable' (§5, Conclusion); blame is a 'structured judgment involving causality, agency, intentionality, and obligation' (§5). [INFERENCE] 'Traceable' is M16's run log + trace renderer; 'contestable' is the mutation-testing discipline. Little new.
8. **Small-N / long-horizon — none.** [PAPER] No timescale or horizon; persistence of negative affect is said to reflect that 'the relationship remains unresolved' (§3). [INFERENCE] No decades-scale content.

**Not transferable / cautions.** A design/position paper with no quantitative model, no equations, no scheduling, and no long-horizon dynamics; its four empirical cases are short perception studies of staged robot/VR behavior, not longitudinal data, and 'affective viscosity' is explicitly not a validated measure. The honne/tatemae framing is culturally situated and untested by the author's own cases, so no culture-specific display convention should be imported. The one genuinely distinct mechanism it names — expression gated by accumulated state with abrupt threshold discharge — is largely present already in Bowen form (M1.A.8 decay, M1.F.9 'spilled over', M4.D.1b WITHHOLD, M4.C.3a integrator). Two modest candidates follow; the rest is already covered.

## B. Candidate additions to the EPModel spec

**AV1 — expression gated by time-integrated load, not instantaneous anxiety.**
- **Proposed requirement:** A symptom onset or an escalation/discharge move SHOULD be gated by time-integrated unresolved load (the M4.C.3a integrator — excess acute anxiety above the person's usual level), not by instantaneous acute anxiety alone, so that a single spike cannot fire an expression the accumulated state does not sustain; any gate constant is [I].
- **Where it would live:** M7.D (symptom channels), reading M4.C.3a.
- **Evidence:** §2–§3 — immediate expression is 'psychologically unnatural'; the Fujino and Sunasaka cases stage rejection 'through suppressed expression' and a 'delayed escalation policy... evaluated more positively than immediate boundary setting'. Strength: ARGUED (the perception result is SHOWN, but only for human preference, not model necessity).
- **What it changes:** Adds a mechanism (lag/suppression gate on expression); improves theory fidelity — symptoms as chronic binding rather than an instantaneous mirror of acute anxiety; indirectly reinforces the M1.A.0 readout trap.
- **Cost/risk:** Low–moderate; one new [I] constant. No conflict with M3.D.4/D.5, M3.D.6, M11.F.9, or M16.B. Must not reintroduce 'visible emotion = measure of anxiety' (M1.A.0).

**AV2 — appraisal sensitizes to repeated same-source events.**
- **Proposed requirement:** The appraisal of an Event SHOULD be modulated by accumulated unresolved affect already held from prior events with the same sender and move type within a declared window, so a repeated identical provocation escalates the recipient's response until discharge rather than producing an identical response each time; the modulation constant and window are [I].
- **Where it would live:** M4.C (appraisal).
- **Evidence:** §3 ('respond differently to the same event after repeated encounters'; retention); Sunasaka's Discomfort Index 'accumulates... with an emotional decay factor' and the agent 'changes its vocal tone according to the accumulated state' (§4.1). Strength: ARGUED.
- **What it changes:** Adds a mechanism (history-dependent reactivity) improving theory fidelity. Inverse direction of C13 (habituation of relief credit), so distinct; complements M4.G.1 (repeated moves harden ties, consequence-side) and M4.C.3a (cost-side).
- **Cost/risk:** Low–moderate; new [I] constant + window. No conflict with M3.D.4/D.5, M3.D.6, M11.F.9, or M16.B.

Already covered (one line each): internal-vs-display split (M1.A.0); accumulation/decay (M1.A.8); suppression/WITHHOLD (M4.D.1b/c); abrupt discharge ('spilled over', M1.F.9); triadic configuration as affect source (M1.C, M8); two-sided reading of restraint (M11.F.6, M1.A.9a); blame two-sidedness (M11.F.6); traceability of intervention (M16).
