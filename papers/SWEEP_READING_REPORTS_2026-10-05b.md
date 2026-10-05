# Reading reports: three silicon-sampling papers, full reads (2026-10-05, batch b)

Produced 2026-10-05 at the owner's request. The Argyle reader in `SWEEP_READING_REPORTS_2026-10-05.md` found that these
three later silicon-sampling papers had never been read in full: spec revision 10 records 2609.15849 and 2609.16395 as
"read at abstract level only; no requirement derived", and `INDEX.md` listed 2609.10280 as `new`. All three were read in
full from `pdftotext -layout` output by three readers working under `sweep_readers_brief_2026-09-19/READER_TASK.md`.
All three readers were interrupted once by an API overload error and resumed; each re-read the text it had not
finished. Each reader checked for overlap by keyword search in spec revision 10,
`SPEC_CANDIDATES_from_preprints_2026-09-20.md`, all earlier report files including the Argyle report (OM1–OM3), and
TODO.md's revision-11 held section. Reports are reproduced as written. Only heading levels were changed.

**Status: none of these candidates is in the spec.** A plain-language explanation with examples is in
`SPEC_CANDIDATES_plain_language_2026-10-05b.md`. The spec's revision-10 note that two of these papers were read at
abstract level is now out of date; it is left unchanged because revision 10 is under external review.

| Part | Paper | Candidates |
|---|---|---|
| 1 | Sen et al. 2026, *Total Simulated Survey Error* (2609.10280v1) | TS1–TS4, TS-X1 |
| 2 | Wali & Tayyab 2026, *Before You Poll with LLMs* (2609.15849v1) | BP1–BP5, BP-X1, BP-X2 |
| 3 | Wang 2026, *Silicon sampling answers with country-level assumptions* (2609.16395v1) | SC1–SC5, SC-X1, SC-X2 |

Candidates labelled `-X` are for the exploratory narrator line only and do not enter the v2 spec.

**Checked by the coordinator.** TS3's example of a criterion and a mechanism sourced from the same passage was verified:
`M11.C.25` and `M2.A.0g` both cite `fe07.md` · FE07.3 and quote the same sentence.

**Correction found by the BP reader.** The four-way taxonomy in *Before You Poll* (match / overshoot / reversal /
rigidity) is not new to the project: Wang et al. 2608.06485, read in full in `SWEEP_READING_REPORTS_2026-09-20.md`,
already classifies persona shifts the same way and is the source of X1 and X2. What is new is the content-trigger
control, use-site diagnosis and the treatment-equivalence point.

---

# Part 1: Error taxonomy

## Reader report: Sen, Ahnert, von der Heyde, Lasser, Weiß & Strohmaier, "Total Simulated Survey Error: Designing and Diagnosing Survey Responses from Large Language Models", arXiv 2609.10280v1

**Scope.** I read every line of the pdftotext output (lines 1–2504). That covers §1–§8, the reference list, and Appendix §9.1–§9.3: the TSE background, the case-study appendix with Figures 7–10, the prompts, the LLM-judge prompts, and the full Table 2 checklist. Work was interrupted by an API error part-way through. I re-read all five chunks from line 1 afterwards, so no chunk was read only once.

**What was garbled or missing in the text:**
- Figures 1, 2, 3 and 6 survive only as captions and box labels. Figure 3's stage and error labels are readable.
- Figures 4, 5 and 7–10 survive only as axis labels, row labels, subgroup means and adjusted R² values. Every regression coefficient, confidence interval and per-model bar is lost, so results that exist only in those plots are reported here from the prose.
- Table 1's bold, blue and red marking of the "best" variant is lost.
- The paper has its own slips:
  - §5 says it varies "one design dimension at a time while holding others constant" (l. 716). The design actually run is a full factorial of 2 × 4 × 6 × 3 × 2 = 288 configurations (l. 768).
  - The appendix gives the paper a different title, "…An Evaluation Framework for LLM-generated Survey Responses" (l. 1507).
  - Figure 8's caption says the error bars are standard errors, but its axes say 95% CI.
  - A sentence in §9.2.2 breaks off mid-way ("e.g., The patterns…", l. 1841).
  - On weighting (§3.3, l. 623), the paper cites Danielmeier & Ullsperger 2011, whose title is about post-error adjustments in cognition. [INFERENCE: probably a mis-citation, judged from the reference title only.]
  - The LLM-judge user prompt always quotes the v1 question wording, including when it codes responses to v2 (l. 2171).
- The eight ideology subgroups in Fig. 7 sum to 4,154 of the 4,779 respondents. The remaining 625 are not mentioned. [INFERENCE: my arithmetic.]
- `papers/INDEX.md` lists the paper as `new` and `DIGEST.md` puts it "out of scope on level grounds". The authors and arXiv ID in both match the title page.

**What I read of the project:**
- In full: `READER_TASK.md`, `EPMODEL_BRIEF.md`, and Part 3 of `SWEEP_READING_REPORTS_2026-10-05.md` (Argyle et al.: OM1–OM3, OM-X1).
- `DESIGN_LESSONS_model_design_papers_2026-09-17.md`: §0–§1, §2.1, §2.6, §2.9, §4 and §5, plus a grep of the whole file for fidelity, silicon, valid, homogen, calibrat and error.
- In the spec (rev10), in full: M11 preamble and M11.1–M11.4f, the whole M11.C table, M11.D, M11.E, M11.F and M11.G; M15.A–E; M17.A–G. I also read M10.B.4, M10.C.3–C.5 and M2.A.0g.
- The full candidate text of the nearest held proposals: CB1–CB3, CD1–CD4, WA1–WA5, PW1–PW3, TM1–TM6, AQ1–AQ3, PD4, NG1–NG3, X1–X10 and OM1–OM3.

**Keyword searches.**
- Terms: fidelity, silicon, survey, error, total, TSE, bias, variance, coverage, nonresponse, measurement, specification, processing, homogen, compress, dispersion, valid, multiverse, preregist, reliability, test-retest, inconsequential, ground truth, context drift, temporal validity, single metric, reweight, adjustment, weighted, construct, operationali, and the paper's ID and authors.
- Files searched:
  - the spec;
  - `SPEC_CANDIDATES_from_preprints_2026-09-20.md`;
  - all six `SWEEP_READING_REPORTS_*.md` files;
  - `TODO.md`'s "Spec revision 11 — held" section;
  - `DESIGN_LESSONS`, `INDEX.md` and `DIGEST.md`.
- Results: "Total survey error", TSE, nonresponse, specification error, processing error, context drift and inconsequential have zero hits anywhere. "Reweight" appears only in the Argyle report. The spec has no rule on weighting ensemble members.

---

### A. Report

**1. Simulation formalism.** [PAPER] There is no time, no interaction and no state between queries.
- §2 gives the formal pipeline. A simulated response is ŷ = f_LLM(t, f_persona(f_map(u))), where t is the task, f_map maps a population unit to a simulation unit, and f_persona turns that unit into a prompt. The survey statistic is f_adj(Σ f_proc(ŷ)).
- The case study runs each configuration over 5 seeds. In total that is 4,779 respondents × 144 pre-adjustment configurations × 5 = 3,440,880 responses (§5, l. 768–770).
- Adjustment is deterministic, so its reliability is not tested (fn. 13).

**2. Agent architecture.**
- [PAPER] The "agent" is an instruction-tuned LLM prompted with a persona and a task (§2, Fig. 2).
- [PAPER] Two framings sit side by side (§9.2.3):
  - The system prompt casts the model as "a political scientist predicting vote choice".
  - The user prompt is a first-person persona, either declarative or in interview format.
- [INFERENCE] That mixes prediction and role-play. Because it was held constant across all arms, any articulation error it causes is invisible to the study.

**4. Initialisation and sensitivity to it.** [PAPER, §5, Table 1, Fig. 5, Fig. 8]
- Persona composition is the largest and most consistent effect. Demographics plus attitudes give F1 0.624 and TVD 0.147; demographics alone give F1 0.434 and TVD 0.27 (direct format).
- This is the only finding that holds on both metrics and in nearly all subgroups (App. 9.2.2).
- Persona format is a smaller effect: direct format beats interview format (F1 0.624 vs 0.593 with attitudes).
- The two question wordings show no significant difference (F1 0.52 vs 0.518).
- [PAPER, §5] The authors note the strongest persona variables (party identification) are close to the target (vote choice), and that both came from the same 2024 survey. They call real use, where the target is missing, a different setting.

**5. Calibration and validation (the main content).**
- [PAPER, §3, Fig. 3] **The TS2E taxonomy.** Each error is a gap δ between artifacts at successive stages, with an example per stage.

  | Stage | Error | Gap |
  |---|---|---|
  | Measurement, designer | Specification | construct → question, δ(c, q) |
  | Measurement, designer | Articulation | question → task, δ(q, t) |
  | Representation, designer | Persona construction | population → persona set |
  | Measurement, LLM-inherent | Response generation | oracle LLM → actual, given the task |
  | Representation, LLM-inherent | Persona simulation | oracle LLM → actual, given the persona |
  | Measurement, after fielding | Response processing | raw response → processed |
  | Representation, after fielding | Adjustment | unweighted → reweighted |

- [PAPER, §3.2] The "oracle LLM" is a stand-in for an error-free simulator. The inherent errors are defined against it. In practice the authors substitute human survey data, because no oracle exists (§5, l. 719–728).
- [PAPER, §1, §6.2] **The argument for decomposing.** Prior work reports one aggregate similarity score, which can hide distorted variance, subgroups and downstream relationships. The paper separates errors that are the designer's choice from errors inherent to the instrument, and errors about *what* is measured from errors about *whom*, because "the two call for different remedies" (§6.2). ARGUED.
- [PAPER, §3.2, §7] **Reliability is distinct from sensitivity.**
  - Reliability is variation in output for identical input across runs.
  - Sensitivity is variation under "minute and inconsequential" input changes.
  - The paper recommends assessing both, in addition to validity.
- [PAPER, §4] **Three evaluation fallacies.**
  - *Ground truth fallacy.* The reference data carry their own errors and may be contaminated: in the model's training data, or written with LLM help.
  - *Single best metric fallacy.* No one metric suffices. High performance may not carry over to downstream estimates; effect sizes and directions should be compared.
  - *Context drift fallacy.* Validating in one setting and simulating in a "similar" one ("validate-then-simulate") assumes a similarity nobody has measured.
- [PAPER, §5, Table 1, Fig. 4] **Case-study numbers.**
  - Overall F1 is 0.519 ± 0.1 and TVD 0.23 ± 0.08 (SD across configurations).
  - The best configuration scores F1 0.686 / TVD 0.097 and the worst F1 0.308 / TVD 0.489, which the authors describe as high sensitivity.
  - F1 and TVD rankings correlate at ρ = −0.84, but the best LLM differs by metric: Qwen-30B on F1, OLMo-32B on TVD. The authors call this the single-best-metric fallacy in action (App. 9.2.2).
  - Per-configuration spread across the 5 seeds is described as "generally stable". It is shown only in Fig. 4, which is lost.
- [PAPER, Fig. 5, Fig. 8] **Subgroups.**
  - Mean F1 by group: moderates 0.38, slightly conservative 0.45, liberal 0.72, extremely conservative 0.73.
  - The best LLM overall is not the best for the conservative groups.
  - Design choices explain less variance where performance is worst. On TVD, adjusted R² is 0.30 for moderates and 0.19 for slightly conservative, against 0.72 overall.
- [PAPER, App. 9.2.2, Figs. 9–10] **Interactions.** Adding LLM × design-choice interaction terms raises adjusted R² from 0.93 to 0.96 (F1, overall), from 0.67 to 0.81 (F1, moderates) and from 0.19 to 0.54 (TVD, slightly conservative). The authors read this as LLM and design choice interacting systematically.
  - [INFERENCE] With about 50 predictors on 288 observations, part of that rise is expected whatever the cause.
  - [INFERENCE] Adjusted and unadjusted configurations share the same raw LLM outputs, so the 288 observations are not independent.
- [PAPER, Fig. 7] All six LLMs under-predict non-voters, most heavily in the moderate and conservative groups, where non-voting is most common.

**6. Failure modes, biases and limitations.**
- [PAPER, §5] Reweighting with the ANES pre-election survey weights slightly *lowered* performance (F1 0.53 → 0.508). Only the slightly liberal group benefited.
- [PAPER, §3.3] The authors explain this through adjustment error: survey weights correct human representation errors, and the LLM's errors need not have the same structure.
- [PAPER, §5] Choosing the best variant as each regression's reference category makes every other coefficient negative by construction. The text states this.
- [PAPER, §7 Limitations] The authors say the case study illustrates the framework and should not be generalised. A design choice that is not significant here may be significant elsewhere.
- [PAPER, §7] Post-training may reduce a model's ability to simulate fringe groups. They name this a "trade-off between realism and assistiveness".
- [PAPER, §4, §7] Proprietary models can change or be withdrawn without notice, which threatens both reliability and reproducibility.

**7. Software engineering and reproducibility.**
- [PAPER, §9.3, Table 2] A documentation checklist runs across the lifecycle:
  - goal of the simulation, degree of substitution, and stakes;
  - construct, with the estimand stated;
  - target population and subgroups, including the time period the findings apply to;
  - exact question wording and variations;
  - simulation unit, sample size, and sampling procedure with seed;
  - persona format, attributes and their rationale;
  - exact prompts;
  - number of runs;
  - model version and hash, knowledge cut-off, open vs proprietary, base vs instruct, temperature, access mode, decoding parameters;
  - processing method (LLM judge or other classifier);
  - source and method of weights;
  - reference data and its limits, metric and its rationale, subgroup analysis, inter-run variance.
- [PAPER, fn. 11] The judge model is deliberately a different model from the simulators, to avoid self-preference.
- [PAPER, §7] The authors suggest preregistration "for the next LLM or survey wave" as a guard against selecting among high-variance configurations.
- [PAPER] Code is public (l. 772).

**8. Small-N, family, emotion, long horizon.** None.
- [PAPER, §7] The authors say errors in generative agent-based models over many rounds, such as feedback loops, need their own framework, and leave that open.

**Mapping the TS2E stages onto EPModel's pipeline** [INFERENCE]

| TS2E stage | EPModel counterpart | Already addressed by | Gap |
|---|---|---|---|
| Specification (construct → question) | corpus construct → `M11.G` component or `M11.C` readout | `M11.F.6`, `M11.G.2`, CB1 (layer), TM3 (components), RP2 | No requirement that a direction be checked under a second admissible readout of the same construct (TS2) |
| Articulation (question → task) | readout definition → code; arm definition → declared channels | `M11.D.20`, TM6, `M17.D.3`, NG2 | — |
| Persona construction (population → personas) | reference family / `D0` / `M15` import | `M15.B`, `M15.B.4`, `M17.C.1`, PW1–PW3, TM1, OM2 | — |
| Persona simulation and response generation (inherent instrument limits) | the mechanism set and its `[I]` constants; no executable oracle exists | `M17.E.1`, `M17.D.2`, `M10.C.5`, `M11.1d`, `M11.D.18–19`, `M11.1f` | Whether a criterion's check comes from the same passage as the mechanism it tests is not recorded (TS3) |
| Response processing | state → readout classification and aggregation | `M11.D.20`, CD4, `M17.B.2`, `M17.B.4` | — |
| Adjustment | weighting of seeds or `D0` draws | PW3 (no filter by `M10.C.4`), PD4 (no selection) | Weighting is not covered (TS4) |
| Ground truth fallacy | corpus bounds and illustrations as reference | `M10.C.3`, `M10.C.4`, `M17.G.3`, AQ2, `M11.F.9(c)` | — |
| Single best metric fallacy | one readout, one aggregate | CB2, OM1 (strata) | the readout dimension itself (TS2) |
| Context drift fallacy | a direction shown on one family carried to another | `M15.D.2` (every import is run), `M17.E.4`, OM2 | — |
| Reliability / sensitivity | seed spread / inconsequential re-encodings | `M11.D.5`, `M17.A.1` / `M11.1c`, `M11.D.16`, `M17.E.6` | — |
| "Total" error | per-result audit record | `M17.G.1`, CD4 | Nothing ties claim grade to coverage of every stage (TS1) |

**What this paper adds beyond OM1–OM3 and the other fidelity reads** [INFERENCE]
- Argyle supplies fidelity *criteria*. Kutzner (CB) supplies *layers*. This paper supplies a *stage-by-stage error ledger*, split by measurement vs representation and by designer vs inherent.
- Mapped onto EPModel, most stages are already covered by scattered requirements. What is missing is:
  - the readout-operationalisation stage;
  - weighting of ensemble members;
  - the provenance overlap between a check and the mechanism it checks;
  - a rule that a result may not carry a high claim grade while a whole stage is unaudited.
- The adjustment finding is the one new SHOWN negative result relevant to EPModel: reusing weights built for another error structure made things slightly worse.

**Not transferable / cautions**
- **The oracle-LLM construction has no EPModel counterpart.** There is no error-free simulator of a family, and the corpus is not one. Using corpus vignettes as the oracle would be the tuning `M10.C.3` and `M11.F.9(c)` forbid.
- **All of the paper's errors are quantified against human data.** EPModel has none, so only the *taxonomy* and the reporting rules transfer, never the error magnitudes.
- **Multiverse regression of performance on design choices**, with the best variant as reference, ranks configurations. In EPModel the counterpart would select a configuration, which PD4 and AQ2 forbid. Design dimensions belong in `M17.G.1`'s audit as robustness checks, not in a ranking.
- **Mixed-LLM samples** (§5, "leverage the differential simulation capabilities") would amount to choosing an instrument per subgroup after seeing results. That is a fitting move and does not transfer.
- **The case study is one survey, one construct and one election**, with personas and target drawn from the same survey wave. Its numbers say nothing about simulated time or interaction.

---

### B. Candidate additions to the EPModel spec

**TS1. Stage coverage before claim grade: a result's audit record covers every pipeline stage, or its claim grade is capped.**
- **Proposed requirement.** `M17.G.1`'s design dimensions **MUST** each be assigned to one of five stages:
  - *representation*: initial persons and ties, topology, family composition, `D0` or import ranges;
  - *mechanism*: appraisal and belief rules, `[I]` constants, activation and same-tick aggregation, tick length;
  - *arm definition*: intervention timing, target and channel;
  - *measurement*: readout operationalisation (TS2), estimator windows, analysis thresholds (CD4), aggregation statistic;
  - *evaluation*: comparison with `M10.C.4`'s bounds (`M17.G.3`).

  The audit record **MUST** show, per stage, whether any dimension was perturbed. A result with a whole stage unaudited **SHOULD NOT** be given a claim grade above *exploratory*. Where it is, the record **MUST** name the unaudited stage.
- **Where it would live.** `M17.G.1`, Phase E.
- **Evidence.**
  - §1, §3 and §6.2 argue that one aggregate score hides stage-specific error and that stages need different remedies. ARGUED.
  - §5 shows the largest effect sitting in one stage (persona construction, F1 0.434 vs 0.624), and shows design choices interacting (adjusted R² 0.67 → 0.81 for moderates). SHOWN for the LLM case.
  - The transfer is INFERENCE.
- **What it changes.** It adds a reporting rule. Today `M17.G.1` lists dimensions with held / sensitive / unaudited per dimension, but nothing stops a result graded *intervention* from having audited only mechanism-side dimensions. This improves inference validity.
- **Overlap.** It builds on `M17.G.1` and CD4 (which adds analysis thresholds), and is the grouping into which TS2's new dimension falls. Not covered elsewhere.
- **Cost / risk.** Low: classification of existing audit fields. The stage assignment is editorial and should live in config with the dimension list. No conflict with `M3.D.4`/`M3.D.5`, `M3.D.6`, `M11.F.9` or `M16.B`.

**TS2. Readout operationalisation as an audited dimension: each directional result is checked under at least two declared readouts of the same corpus construct.**
- **Proposed requirement.** For each `M11.C` criterion, before it is run, the test **SHOULD** declare its primary readout and at least one alternative admissible readout of the same corpus construct, each citing its ledger source. Examples:
  - time to symptom threshold vs time-integrated symptom load;
  - end-of-window value vs settled-window mean (`M17.F.2`);
  - share of seeds vs mean paired difference.

  Phase E **MUST** report the direction under each declared readout. Disagreement **MUST** be recorded in `M17.G.1` as *sensitive* on a new "readout operationalisation" dimension. The primary readout **MUST NOT** be changed after results are seen (`M10.B.4`; AQ1).

  Alternatives that the corpus marks as traps (`M11.F.6`'s list, `M1.A.3c` overt emotionality, `M11.C.28` symptom count) **MUST NOT** be admitted as alternatives.
- **Where it would live.** `M11.C` test declarations; `M17.G.1`, Phase E.
- **Evidence.**
  - §4 defines the single-best-metric fallacy and asks that downstream effect sizes and directions be compared. ARGUED.
  - §5 and App. 9.2.2: the best LLM and the best response handling change when the metric changes from F1 to TVD, and only persona composition holds on both. SHOWN.
  - The transfer from metric to readout is INFERENCE.
- **What it changes.** It adds a test dimension. Today:
  - TM3 makes a criterion name which *component* it asserts;
  - CB1 makes it name which *layer* it reads;
  - CD4 varies *thresholds*;
  - `M11.D.20` checks that two *renderings* agree.

  None asks whether the direction survives a different, equally admissible operationalisation of the same construct. That is the specification-error stage, and nothing in the spec currently audits it. This improves inference validity and, through the trap exclusion, theory fidelity.
- **Overlap.** Partly covered by CB1, TM3, CD4, `M11.D.20` and RP2. The new element is the alternative readout as a declared, audited dimension.
- **Cost / risk.** Low compute, since the same runs are re-read. The risk is a readout search, where alternatives are added until one agrees. The pre-declaration clause and the `M10.B.4` freeze are the guard. Where the corpus admits only one readout, the criterion should say so rather than invent an alternative, which would itself be `[I]`.

**TS3. Check-source independence: a criterion whose check comes from the same corpus passage as the mechanism it tests is reported as an implementation check.**
- **Proposed requirement.**
  - For each `M11.C` criterion, the register **MUST** record the ledger IDs that source the criterion and the ledger IDs that source each rule in its `M11.1d` minimal set.
  - Where a criterion and a rule in its minimal set rest on the same passage, the criterion **MUST** be marked *same-source* in `M10.C.5`'s premise column. Its pass **MUST** be reported as confirming that the passage was implemented, not as a consequence derived by the model.
  - A verified instance: `M11.C.25` and `M2.A.0g` both rest on `fe07.md` · FE07.3 and quote the same sentence.
- **Where it would live.** `M10.C.5`, next to `M11.1d`; reporting in `M17.G.1`.
- **Evidence.**
  - §5 concedes that the persona variables and the target came from the same survey, and that party identification and vote choice are strongly correlated, so in-sample success overstates real use. ARGUED.
  - §4's contamination form of the ground-truth fallacy makes the same point: the reference was already inside the instrument. ARGUED.
  - The transfer is INFERENCE.
- **What it changes.** It adds a reporting rule. `M11.1d` catches a rule that *names* the outcome, and `M10.C.5` catches a single rule whose *inversion* flips the criterion. Neither catches a criterion that passes because the check and the mechanism transcribe the same sentence. The pass is real, but it shows that the passage was implemented, not that the theory has a consequence. This improves inference validity, and stops implementation checks being presented as theory results.
- **Overlap.** Partly covered by `M11.1d`, `M10.C.5` and AQ1 (held-out audit after revision). The provenance comparison is new.
- **Cost / risk.** Low: register bookkeeping from citations the spec already carries. Many same-source criteria are legitimate, for example `M11.4`'s nulls and conservation checks. The rule labels them; it does not remove them. No conflict.

**TS4. No post-hoc weighting of seeds or initial-condition draws.**
- **Proposed requirement.** Phase E **MUST NOT** weight seeds, `D0` draws or sweep samples by their agreement with `M10.C.4`'s bounds, `M15.B.4`'s corpus distribution, or any other target.
  - The only admissible weights are a sampling measure declared in `D0` or in PD4's design before the run.
  - A report that uses one **MUST** show the unweighted result beside it.
- **Where it would live.** `M17.C.1`, alongside PW3 and PD4.
- **Evidence.**
  - §3.3 defines adjustment error and argues that weights built for one error structure need not fit another. ARGUED.
  - §5: reusing the ANES weights lowered F1 from 0.53 to 0.508, and helped only one subgroup. SHOWN for the LLM case.
  - The transfer is INFERENCE.
- **What it changes.** It closes a gap. PW3 forbids *filtering* draws by the bounds and PD4 forbids *selecting* a sample, but weighting is the continuous form of both and is not named. A weighted ensemble pulled toward the corpus distribution is a soft fit, which leans toward `M11.F.9(c)`. This improves inference validity.
- **Overlap.** Partly covered by PW3, PD4, `M15.B.4` (check only) and `M10.C.4` ("checks, never parameters").
- **Cost / risk.** Negligible: it is a prohibition. No conflict.

**Already covered (one line each)**
- Reliability vs sensitivity: `M11.D.5` and `M17.A.1` (seeds); `M11.1c`, `M11.D.16` and `M17.E.6` (inconsequential re-encodings, order, activation regime).
- Context drift: `M15.D.2` runs every imported family itself and never infers it from a similar one; `M17.E.4` and OM2.
- Ground-truth fallacy: `M10.C.3`, `M10.C.4`, `M17.G.3` and AQ2.
- Subgroup failure hidden by an aggregate: CB2 and OM1.
- Design-choice interactions: `M17.E.5` and TM4.
- Preregistration: `M10.B.4`, `M17.A.4`, WA4 and AQ1.
- Documentation checklist: `M16.A`, `M17.G.1`, TM6 and AQ3.
- Processing error: deterministic readouts, `M11.D.20` and CD4.
- Rare outcome under-predicted (non-voters, Fig. 7): `M11.1f` and `M11.D.19`.

**Narrator / LLM line only (does not enter v2; `M3.D.6` stands)**

**TS-X1. Narrator runs carry the TS2E documentation fields, a pinned model, and a contamination statement for corpus cases.**
- **Proposed protocol item.**
  - Any reported narrator result **MUST** record Table 2's model, decoding and processing fields: exact model version or checkpoint hash, knowledge cut-off, base vs instruct, access mode, temperature and decoding parameters, number of runs, and inter-run variance.
  - It **MUST NOT** rest on a model reached through an unversioned API.
  - Where narrated material resembles a published corpus case, the record **MUST** state whether that case predates the model's cut-off, because recall of a published case can pass for simulation (`DESIGN_LESSONS` §3.5).
- **Evidence.**
  - §4 and §7 (API models change or are withdrawn; training-data contamination inflates apparent performance). ARGUED.
  - Table 2 is the checklist. Footnote 11 keeps the judge separate from the simulator, practice SHOWN.
- **Overlap.** Partly covered by X7 (judge ≠ narrator), X1 (retest floor and paraphrase), `DESIGN_LESSONS` §3.1 and §3.5. New here: pinning the version as a condition of reporting, and the cut-off-versus-corpus statement.

---

# Part 2: Dynamic fidelity

## Reader report: Wali & Tayyab, "Before You Poll with LLMs: A Deliberative Diagnostic Framework", arXiv 2609.15849v1 (14 Sep 2026, cs.CL; Lahore University of Management Sciences)

**Scope.** I read every line of the pdftotext output (lines 1–928). That covers the abstract, §1–§9, Limitations, Ethics, Acknowledgments, the references, Appendices A–E (prompt templates in Figs. 3–5) and Tables 1–15. An API overload interrupted the session, so after resuming I re-read the whole paper from line 1.

**What was garbled or missing in the text:**
- Figures 1 and 2 survive only as captions. Table 4 carries Figure 2's numbers.
- The two-column layout interleaves the abstract and §1, but the text is readable.
- Appendices C–E are short stubs that point to Tables 9–15.
- Table 14's caption says "|∆| < 0.1*", but its rows use thresholds 1, 2, 3 and 5 on the 0–10 scale. Under the largest threshold, 86.8% of human responses count as "no meaningful change". I cannot reconcile the caption with the rows.
- The paper contradicts itself on the claim it motivates with. §2.1 says GPT-5.1 "reproduces pre-deliberation opinions". Table 8 shows its pre-poll means 1.2 and 3.2 points below the human means on Q16A and Q16B. Limitation (7) concedes that pre-poll fidelity was "not directly" evaluated. So the paper never shows that a model passes static fidelity and fails dynamic fidelity. It shows dynamic failure only.

**What I read of the project.**
- In full: `READER_TASK.md` and `EPMODEL_BRIEF.md`.
- `DESIGN_LESSONS_model_design_papers_2026-09-17.md`: §1, §2.6, §2.7, §3 (all), §4, §7.5, §7.6 and §8.
- Part 3 of `SWEEP_READING_REPORTS_2026-10-05.md` (Argyle et al.; OM1–OM3, OM-X1).
- In the spec (rev10), in full: M11 preamble, M11.1–M11.1f, M11.2, M11.3, M11.4–M11.4f, the M11.C table through C.41, M11.D, M11.E, M11.F, M11.G, M15 (A–E) and M17 (A–G).
- Also from the spec: M1.E.7–M1.E.8, and the revision-10 note at line ~1942. That note lists 2609.15849 as "read at abstract level only; no requirement derived". This report replaces that status.
- `TODO.md`'s "Spec revision 11 — held" section. Line 148 records this paper as a catalogue gap.
- Candidate write-ups checked for overlap:
  - CB1–CB3, CD1–CD4, WA1–WA5, PW1–PW3 (`-10-04.md`);
  - AQ1–AQ3, RP1–RP2, NG1–NG3, EV1–EV4 (`-10-04b.md`);
  - TM1–TM6, BB1–BB5, MM1–MM4 (`-09-27b.md`);
  - X5–X9 (`-09-27.md`);
  - X1–X4 and C42 (`SPEC_CANDIDATES_from_preprints_2026-09-20.md`), plus the Wang et al. 2026 report in `-09-20.md`, which is the source of X1/X2.

**Keyword searches.**
- Terms: fidelity, dynamic, deliberat, diagnos, silicon, survey, poll, homogen, compress, dispersion, valid, before. Later I added sham, inert, irrelevant, attention, treatment, multicomponent, sycophan, overshoot, reversal, rigidity and "reference family".
- Files: the spec, the candidates file, all six `SWEEP_READING_REPORTS_*.md` files and `TODO.md`.
- Results in the spec:
  - "poll", "silicon", "sycophan" and "sham" have zero hits.
  - "fidelity" hits only per-hop event fidelity (`M1.F.4`).
  - "deliberat" hits only ordinary English.
  - Outside the spec, "sham" and "multicomponent" have zero hits anywhere.

**Correction to the coordinator's framing.** The four-way taxonomy (match / overshoot / reversal / rigidity) is not new to the project. Wang et al. 2026 (2608.06485), already read in full in `-09-20.md`, classifies LLM persona responses to life events as in-band, reversed, under-shift and overshoot, with numbers. That is the source of X1 and X2. What is new in this paper is listed under "What this paper adds" below.

---

### A. Report

**1. Simulation formalism.** Nothing transfers.
- [PAPER, §4.2, Table 7] Each response is one stateless API call at temperature 0, with output capped at 10 tokens so the model returns a bare integer from 0 to 10.
- [PAPER, §4.2] Pre and post questions are asked independently. The model never sees its own earlier answer. The authors say a conditioned design "measures a different capacity", and they leave it to future work.
- [PAPER] No seeds are reported.
- [PAPER, Limitation (9)] There is no repeated-prompting floor at T = 0. The authors say one "would provide this directly".

**2. Agent architecture.**
- [PAPER, Fig. 3] A persona is a fixed template of 13 A1R demographic attributes, with party in the first sentence.
- [PAPER, Figs. 4–5] Variants tested: no party sentence; party named explicitly in the question; persona placed before the briefing; and a multi-agent prompt that injects peer responses between the briefing and the question.
- [PAPER] There is no memory, no state and no learning.

**3. Interaction / network.** The paper treats this as exploratory only (§7, Table 5, Fig. 5).
- Peer injection lowered GPT-5.1's reversal rate from 80% to 20%.
- The same change raised Gemini's overshoot from 6.5× to 18× human magnitude.
- [PAPER] The authors' conclusion: no intervention fixes direction and magnitude together, and the same intervention "can help one model while hurting another".

**4. Initialisation and sensitivity to it.**
- [PAPER, §6.3, Table 8] LLM pre-poll means differ from the human ones before any briefing: GPT-5.1 Q16B is 4.47 against 7.67.
- [PAPER] The diagnostic deals with this by using the within-persona shift ∆ = Post − Pre, which "absorbs each persona's baseline".
- [PAPER, Table 12] Label-swap 2×2 on Gemini (N = 326): both the party label and the other demographics move ∆.
  - The label effect is asymmetric: 0.38 points within Democrat-demographic personas (p < 0.001), 0.13 within Republican-demographic personas (p = 0.27).
  - The demographic effect is 0.23 and 0.28 points.

**5. Calibration and validation (the main content).**
- [PAPER, §3.1] **The four-phase protocol:** pre-poll; identical balanced briefing to humans and personas; post-poll; then compare ∆ between humans and LLMs. Each question is classified as **match**, **overshoot** (same direction, larger), **reversal** (opposite direction) or **rigidity** (near-zero change).
- [PAPER, §3.2] **The human baseline:** America in One Room, 2019. 526 voters recruited by NORC address-based probability sampling. 72 of 107 items kept; the exclusions are itemised in §4.3. Its value as a reference is a known direction: deliberation reduces hostility toward the other party.
- [PAPER, Table 1] **Outgroup results** (∆ on 0–10; human −0.21, CI [−0.33, −0.09]):

  | Model | ∆ | Classification |
  |---|---|---|
  | GPT-5.1 | +0.50 | reversal |
  | Gemini 2.0 Flash | −1.36 | overshoot |
  | Claude Sonnet 4.5 | −1.04 | overshoot |
  | Llama 3.3 70B | −1.42 | overshoot |
  | DeepSeek V3 | +0.02 | rigidity |

  - Four of five model–human comparisons survive Bonferroni correction at α = 0.0038. DeepSeek's does not (p = 0.051).
  - Claude, Llama and DeepSeek were run on 100-persona subsamples.
- [PAPER, Table 14] **Rigidity:** at |∆| < 1, DeepSeek has 97.2% near-zero responses against 24.5% for humans. The paper calls this "4.0×", although 97.2 ÷ 24.5 ≈ 4.0 is correct; the rigidity classification is stable at every threshold.
- [PAPER, §5.2, Table 2] **Selectivity:** GPT-5.1 reverses on 80% of outgroup items (4/5) but 26% of policy items (12/47). Fisher OR = 11.67, p = 0.027, which does **not** survive correction. The authors rest the selectivity claim on three convergent experiments instead.
- [PAPER, §6.1, Table 4] **Content-trigger control:** replacing the policy briefing with irrelevant Wikipedia text (pasta, chess, gardening) gave ∆ = −0.75 for both GPT-5.1 and Gemini. GPT-5.1's reversal disappeared, but both models still moved 3.6 times the human response to the real briefing.
- [PAPER, §6.2] **Target swap:** the same persona and briefing produce opposite failure modes depending only on whose group the question is about.
  - GPT-5.1: outgroup ∆ +0.24, ingroup ∆ −1.09 (t = 12.29).
  - Gemini: the inverse (t = −12.60).
- [PAPER, §6.3, Table 13] **Stratification:** GPT-5.1's outgroup ∆ is positive in every stratum of 10 of 11 non-party attributes. The exception is marital status: separated (n = 11) and divorced (n = 80) personas shift in the human direction.
- [PAPER, §6.3] **Temperature:** at T = 0.7 Gemini's overshoot persists (∆ −1.36 → −1.55; paired r = 0.77).
- [PAPER, §8.1] **The practitioner protocol:**
  1. Build a calibration sample from a small deliberative poll in the target domain.
  2. Run matched personas through identical briefings.
  3. Apply the taxonomy. Do not deploy a model that reverses on identity-relevant items. If it overshoots, apply calibration scaling.
  4. Test for selectivity, because aggregate metrics can pass while the most important items reverse.

**6. Failure modes, biases, limitations.**
- [PAPER, §5.1] Every model shows "substantially less response variance than humans".
- [PAPER, Tables 9–10] Per-persona reversal is bimodal: 26.5% of personas never reverse, 31.6% always reverse, mean 52.6%. Reversal is not a stable persona trait: mean ϕ across items is 0.08.
- [PAPER, §6.4] The authors' account is "self-sycophancy": conformity to the model's own stereotype of the persona. They state that behavioural evidence cannot separate this from learned conditional associations.
- [PAPER, Limitations] Their own list:
  - one poll, US, 2019;
  - instruction-tuned models only;
  - prompt sensitivity not explored exhaustively;
  - Independents excluded from the target-swap test;
  - no variability floor;
  - proprietary APIs with opaque moderation;
  - (11) the human and LLM treatments are not equivalent: humans had multi-component deliberation, the LLMs had the briefing alone.
- [INFERENCE] Limitation (11) weakens every "reversal" call. The human direction belongs to a different treatment.

**7. Software engineering and reproducibility.**
- [PAPER, Tables 6–7] A registry of all 21 conditions (over 340,000 queries). Per model, it records the API string, endpoint, temperature, token cap, reasoning setting, a 900 s timeout, 5 retries with exponential backoff, and prompt caching on the briefing block. Runs were made in December 2025 and January 2026.
- [PAPER, §4.1] Data and registry are released on Zenodo.
- [PAPER, Appendix C] Statistics: mixed effects with crossed random intercepts for persona and question (statsmodels). One singular fit is flagged, with an unpaired fallback.

**8. Small-N, family, emotion, long horizon.** None. Marital status appears only as a persona attribute (Table 13). The "dynamic" in dynamic fidelity is a single pre/post step with no time axis.

**What this paper adds beyond what is on file** [INFERENCE]
- **The content-trigger control.** An intervention of the same form with inert content produced a response 3.6 times the human response to the real treatment. Nothing on file has a sham-content arm. `M11.D.15` is zero-magnitude and byte-identical, `M11.1b` is a mutant on appraisal inputs, and TM5 is on the readout side.
- **Use-site diagnosis.** Run the diagnostic on the domain and population of intended use, and expect failure to be item- and target-specific. Nothing on file asks that the `M11.C` criteria be re-checked on the initial-condition class a reported result concerns.
- **Treatment non-equivalence** as an explicit threat to a direction comparison (Limitation 11).
- **The static/dynamic split itself.** For EPModel this mostly confirms the design: the `M11.C` criteria are dynamic by construction under `M0.4`, while `M10.C.4`, `M15.B.4` and OM1 are static.

**Not transferable / cautions**
- §8.1 step 3, "apply calibration scaling estimated from the sample", is fitting to observed outcomes. EPModel has no calibration sample, and `M11.F.9(c)` forbids reporting from tuned parameters. Overshoot also has no counterpart: the corpus supplies no magnitudes, so EPModel can classify only reversal, rigidity and direction-match, never overshoot.
- The human-baseline design (matched personas from a real panel) has no EPModel counterpart. Building one would mean scoring the model against real families, which `M11.F.9(a)` rules out.
- The selectivity statistic rests on five outgroup items, and the paper's static-pass claim is contradicted by its own Table 8. Cite the taxonomy and the control designs, not the headline rates.
- Self-sycophancy is a property of LLM generation. EPModel's engine has no counterpart (`M3.D.6`).

---

### B. Candidate additions to the EPModel spec

**BP1. Use-site diagnosis: a result reported for an initial-condition class other than the reference family carries the supporting `M11.C` criteria re-run on that class.**
- **Proposed requirement.** A Phase E intervention result reported over an imported family (`M15`) or over a `D0` class other than the `M2` reference family **MUST** be accompanied by the `M11.C` criteria its mechanism chain depends on, re-run over that same class (as an envelope for an import, `M15.D.2`). Each is marked **held**, **reversed** or **UNDETERMINED** (`M17.A.1`).
  - The dependency list **MUST** be declared before the run, from `M11.1d`'s minimal rule sets.
  - Where a supporting criterion reverses on that class, the result **MUST** be flagged as resting on a mechanism the model does not reproduce there, and **MUST NOT** be reported without the flag.
- **Where it would live.** `M17.G.1` (as an audit-record column), `M15.D` (a new `M15.D.5`), and `M11.1d` for the dependency map.
- **Evidence.**
  - §8.1 steps 1 and 4: the protocol is run "in the target domain", and aggregate success can hide reversal on the items that matter. ARGUED.
  - §6.2 and Table 2: the same model passes on one item class and reverses on another, so a pass in one place does not carry over. SHOWN, for LLMs.
  - The transfer is INFERENCE. It fits EPModel because `M15.D.4` already names five turning points where the sign of a mechanism flips with initial conditions.
- **What it changes.** It adds a reporting rule and fills a gap.
  - The acceptance suite is written against the reference family and fixtures.
  - `M17.E.4` (two reference configurations) and `M17.G.1`'s "family composition" dimension ask whether the *result's own* direction survives. Neither asks whether the mechanisms it relies on still behave as the corpus says on that class.
  - It improves inference validity.
- **Overlap.** Partly covered:
  - `M17.E.4`, `M17.G.1`, `M15.D.3` (flip location);
  - CB2 (position strata within a family);
  - OM2 (import against a no-import baseline);
  - PW1 (`D0` coverage of turning-point sides).

  None re-runs the supporting criteria on the reporting class.
- **Cost / risk.**
  - Moderate compute: the relevant subset of the suite per reporting class. The dependency map is editorial work.
  - It **MUST NOT** become a filter that selects classes where criteria hold (PD4, AQ2's no-selection rule).
  - No conflict with `M11.F.9`, since nothing is fitted, or with `M3.D.4`/`M3.D.5`/`M3.D.6`/`M16.B`.

**BP2. A form-matched inert arm for every intervention-presence criterion.**
- **Proposed requirement.** Every `M11.C` criterion whose arms differ by the presence of an intervention — an external agent (`M11.C.17`, `M11.C.7`) or a delivered knowledge event (the held SV2 answer) — **SHOULD** also be run against a third arm. In that arm the intervention is present with matched timing, contact schedule, tie formation and event intensity, but its credited content is inert through a declared channel (`M17.D.3`); for a coach, this means coach quality at its null value under `M1.E.7d`.
  - The asserted direction **MUST** hold between the intervention arm and the inert arm, not only between the intervention arm and the absent arm.
  - The inert arm's own difference from the absent arm **MUST** be reported as the form effect.
- **Where it would live.** Phase E, `M17.D.1` as a fifth control arm (e).
- **Evidence.** §6.1 and Table 4: irrelevant content with identical prompt structure gave ∆ = −0.75 in both models, 3.6 times the human response (−0.21) to the real treatment. The form of an intervention moved the outcome by itself. SHOWN for LLMs; the transfer is INFERENCE.
  - The EPModel route to a form effect is concrete. Adding a Person with ties creates triangles, and triangling relieves the pair (`M11.C.3`).
  - `M1.E.7e` already names a pseudo-self lending path that `M11.C.17` "must not be able to exploit".
- **What it changes.** It adds a control arm, and separates "a third person arrived" from "a contact landed". This improves both inference validity and theory fidelity: Bowen theory credits differentiation to landed contact, not to added contact (`M1.E.7`).
- **Overlap.** Partly covered:
  - `M17.D.2`, the rival-mechanism arm. It is scoped to between-member differentiation criteria and works by disabling the mechanism, which makes it an `M11.4f` structure test.
  - `M11.D.15`: zero-magnitude placebo, byte-identical.
  - `M11.1b`: a mutant, not an arm.
  - TM5: readout side.

  New here: an inert arm reached through a declared channel, so it stays an ordinary arm rather than an `M11.4f` model change.
- **Cost / risk.**
  - Adds one arm per intervention criterion. The null value of coach quality is `[I]`.
  - Owner question: is coach quality a declared per-agent attribute that can be set to null? If it can be reached only by disabling landing, the arm falls under `M11.4f` and must be reported as a structure test.
  - No rule conflict.

**BP3. Treatment-equivalence declaration for corpus-sourced intervention directions.**
- **Proposed requirement.** Every `M11.C` criterion whose direction comes from a corpus account of an intervention or nodal event **MUST** list the components of that account that the arm implements and those it omits. For example, the corpus describes coaching as a sequence of sessions, a relationship to the coach, and the family's own follow-through (`M1.E.7c`–`M1.E.8`, FE11.6). A direction match or failure **MUST** be reported as concerning the implemented components only.
- **Where it would live.** The `M11.C` preamble, with the listing in each criterion's row or in `model_explainer.md` §18.
- **Evidence.** Limitation (11): A1R participants had moderated small groups and expert plenaries, while the LLMs got the briefing alone, so divergence "may therefore reflect treatment context". ARGUED. The transfer is INFERENCE.
- **What it changes.** It adds a reporting rule and stops a partial implementation being read as a test of the whole corpus claim. This improves theory fidelity and how results are framed.
- **Overlap.** Partly covered:
  - `M17.D.3` (channels declared);
  - `M17.F.1` (a shock must run through the model's own machinery);
  - RP2 (proximal behaviour named);
  - the SV2 answer (arm defined by the knowledge event).

  None asks for an implemented/omitted component list against the corpus source.
- **Cost / risk.** Low; editorial only. No conflict.

**BP4. Static-distribution checks and direction-of-change results are reported apart, and neither is cited as support for the other.**
- **Proposed requirement.** Reports **MUST** carry corpus-distribution checks (`M10.C.4`, `M15.B.4`, `M17.G.3`, OM1 if adopted) in a section labelled *static*, separate from `M11.C` direction results. A static match **MUST NOT** be cited as evidence for a direction-of-change claim, and a static misfit **MUST NOT** be cited as refuting one.
- **Where it would live.** `M11.F` (framing) and `M17.G`.
- **Evidence.** Abstract, §2.1 and §9: static and dynamic fidelity are separable axes. ARGUED. The paper's own demonstration of the separation is incomplete (Table 8 against §2.1; Limitation (7)), so this rests on the argument alone.
- **What it changes.** A framing rule. It closes the route by which "the imported family matches the corpus distribution" (`M15.B.4`) could be read as licensing an intervention result on it. This improves inference validity.
- **Overlap.** Partly covered by CB1 (layers: belief / latent / enacted, which is a different axis), `M10.C.4`'s "checks, never parameters" and OM1. A short clause; it could be folded into CB1 or OM1.
- **Cost / risk.** Negligible. No conflict.

**BP5 (low priority). Reversed seeds are tabulated across criteria and against `D0` draws.**
- **Proposed requirement.** Phase E **SHOULD** report, for the seeds `M17.B.1` classifies as **reversed**:
  - the per-criterion distribution of the reversed share;
  - the pairwise association of reversal across criteria;
  - the association of reversal with declared `D0` strata.

  A bimodal distribution **MUST NOT** be summarised by its mean (`M17.B.4`'s rule, applied here).
- **Where it would live.** `M17.B`.
- **Evidence.** Tables 9 and 10: per-persona reversal is bimodal (modes at 0% and 100%, mean 52.6%), and cross-item ϕ averages 0.08. Together these separate a unit-level cause from item-specific content. SHOWN as method; the transfer is INFERENCE.
- **What it changes.** A diagnostic readout. It tells whether reversals come from particular initial conditions (pointing at a turning point, `M15.D.4`) or from particular criteria. This improves inference validity.
- **Overlap.** Largely covered by `M17.B.1`, `M17.B.4`, PW1 and CB2. New only in the cross-criterion association. Drop it if the owner wants fewer readouts.
- **Cost / risk.** Negligible; analysis only, seeded under AQ3. No conflict.

**Already covered (one line each)**
- Within-unit differencing absorbs baseline mismatch: `M0.4`, `M3.D.4a` coupled arms, `M17.B.1`.
- Classification of the arm difference into direction-match, reversal and rigidity: `M17.B.1` (same/converted/reversed), `M17.A.4` (margin), `M17.A.1` (UNDETERMINED), `M11.4a` (equivalence bound).
- Rigidity classification checked at several thresholds (Table 14): CD4.
- Response variance compressed against a reference: WA1 and CD1 for the engine; X2 and WA-X1 for LLMs.
- Direction checked within every stratum of non-manipulated attributes (Table 13): CB2, OM1.
- Target swap, ingroup against outgroup (§6.2): CB2's inside-pair/outside-member stratum and `M11.C.35`.
- Polarity-reversed item as an internal check (Q16D): `M11.F.6` two-sidedness and `M17.D.5` both signs.
- Label-swap 2×2 (Table 12): considered and not proposed. `M11.C.36` already varies belief with truth fixed, and `M15.A.8` confines the identified-patient label to the belief layer. A full belief × truth crossing would need an owner ruling on how much label-driven response is theory (projection) and how much is artefact.
- Registry of conditions, model-version and transport record: TM6, `M16.A.1`, `M16.A.7`, AQ3.
- Temperature and model as factors: DESIGN_LESSONS §3.3, `M17.E.6`.

**Narrator / LLM line only (does not enter v2; `M3.D.6` stands)**

**BP-X1. Dynamic recovery test for any Phase F narrator, with the engine as baseline.**
- **Proposed protocol item.** On frozen logs, a reader blind to the log narrates each person at a pre tick and a post tick that bracket a nodal event or arm divergence, and scores the narrated change. The sign of the narrated change **MUST** be compared with the engine's change and classified as match, reversal or rigidity.
  - An irrelevant-content control — the post tick rendered with the event's content replaced by an inert event of the same form — **MUST** be run, and its narrated change reported as the narrator's form effect.
  - Results **MUST** be reported per move class and per tie relation (inside pair / outside member), not only pooled. A narrator that matches pooled but reverses in a class **MUST** fail.
- **Evidence.** §3.1 taxonomy, §6.1 content trigger, §5.2 and §8.1 step 4 selectivity. SHOWN for LLM personas. The engine-as-baseline transfer is INFERENCE.
- **Overlap.** Related items, all different:
  - BB1: cross-sectional ordering at sampled ticks.
  - BB2: transfer curve at one time.
  - X6: drift under constant state.
  - X1: retest floor for persona change claims.

  New here: recovery of change across an event, the inert-content control, and the per-class selectivity failure rule.

**BP-X2. Any LLM persona used to represent changing maturity or reactivity is run through a pre/post diagnostic with a content-trigger control and a target swap before use.**
- **Proposed protocol item.** This extends X1.
  - Before an LLM persona (for example "a 15-year-old brat") is used, the same persona **MUST** be given an identical stressor or input in a pre/post design.
  - Add an inert-content control of the same form, and a version in which the target of the response is swapped (own parent against a peer, for example).
  - Report each item class as match, reversal or rigidity against whatever directional reference exists. Without one, report directions only and say so.
  - A persona that changes as much under inert content as under the real input **MUST** be reported as not responding to content.
  - Any mitigation (chain-of-thought, persona-first ordering, multi-agent context) **MUST** be re-tested per model, because Table 5 shows the same mitigation improving one model and worsening another.
- **Evidence.** §6.1, §6.2, Table 5. SHOWN.
- **Overlap.** X1 (retest floor, paraphrase, behavioural readout), X2 and WA-X1 (dispersion), X3/L1 (range ceiling). New here: the inert-content control, the target swap, and per-model re-testing of mitigations.

**Questions for the owner**
1. BP2: is coach quality (`M1.E.7d`) a declared per-agent attribute that can be set to null? This decides whether the inert arm is an ordinary arm or an `M11.4f` structure test.
2. BP1: should the dependency list for an intervention result be derived from `M11.1d`'s minimal rule sets, or declared by hand per result?

---

# Part 3: Group-level defaults

## Reader report: Wang, "Silicon sampling answers with country-level assumptions, not individual attitudes: Cross-national evidence from the European Social Survey", arXiv 2609.16395v1

**Scope.** I read every line of the pdftotext output (lines 1–1605): Abstract, §1–§5.4, Declarations, the reference list, and Appendices A–M, including Tables A1–A6. The paper has one author, Chuyao Wang (LSE). The text has no arXiv date line.

**What was garbled or missing in the text:**
- Figures 1–7 and A1–A4 survive only as captions, and the per-item rankings exist only in those plots. Per-block values are said to be "provided with the figure data", which is not in the text.
- Table A6 has one wrapped row (Llama 3P) but is readable.
- The paper has its own slips. The Declarations contain unfilled placeholders ("[Confirm: …]", "[Insert the funding statement …]"). §4.1 cites the parse loss in Llama 3P as 23%, while App. L gives 77.28% parsed. These agree, but §3.2 rounds coverage to 76%.

**What I read of the project.**
- In full: `READER_TASK.md`, `EPMODEL_BRIEF.md`, and Part 3 of `SWEEP_READING_REPORTS_2026-10-05.md` (Argyle: OM1–OM3, OM-X1).
- `DESIGN_LESSONS`: §0–§1, §3, §4, §7.4 and §7.10, chosen from a grep for fidelity, silicon, valid, homogen, heterogen and group.
- In the spec (rev10): M1.A.7–M1.A.7a, M1.D (all), M4.C, M7.E.1–M7.E.3, M10 (A, B, C.1), M11.1–M11.4 (the M11.4 table), M11.D, M11.E, M11.F, M11.G, M15 (A–E), M17 (A–G), and the revision-10 note at about line 1942. That note records this paper as "read at abstract level only; no requirement derived".
- For overlap, the full write-ups of: WA1–WA5 and WA-X1; CB1–CB3; CD1–CD4; PW1–PW3; TM1–TM6; MM1–MM4; NG1–NG3; EV2–EV4; J4; X1–X4 (`SPEC_CANDIDATES` §5); X5–X9 by title. Also `TODO.md`'s "Spec revision 11 — held" section in full.

**Keyword searches.** Terms: country, group-level, average, family mean, class-level, within/between/across-family, variance, decompos, ICC, multilevel, hierarch, homogen, stereotyp, silicon, ecological, Simpson, pooled, dispersion, compress, polarity, endpoint, reverse-cod. Files searched: the spec, `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, all six `SWEEP_READING_REPORTS_*.md`, and `TODO.md`.

What the searches found:
- **M1.A.7.** The spec's only family-average prohibition is M1.A.7. I found no test or `M11.C` criterion that covers it.
- **Polarity.** No requirement mentions scale polarity, endpoints or reverse coding.
- **Variance components.** The spec's only variance decomposition is M17.C.1's four components (seed, initial condition, spell timing, constants). It works at the family level and has no between-family / within-family split.

---

### A. Report

**1. Simulation formalism.** None that transfers.
- [PAPER, §3.2] Each simulated respondent is one stateless generation. Temperature is 0.7. The subsample uses seed 888, generation is unseeded, and the replicate arm uses seed 889.
- [PAPER, §3.2] There are 685 respondents per country × 30 countries × 42 items. That is 3.5 M attempts in the main 2×2 and 34.6 M across all 43 arms.

**2. Agent architecture.**
- [PAPER, §3.3] The agent is an LLM conditioned on a first-person backstory built from 20 ESS variables through fixed value-label dictionaries. Party is generic.
- [PAPER, §3.2] The paper compares a first-person framing ("adopt the persona") with a third-person one ("predict how this person would respond").
- [PAPER, App. C] An earlier 27-variable backstory leaked outcomes. For example, `inprdsc` had pooled r 0.854 with the leaked variable and 0.038 without it. The seven overlapping variables were removed.

**4. Initialisation and sensitivity to it (the backstory experiment).**
- [PAPER, §3.4] Design: 10 blocks. Each non-base block is tested at two positions:
  - added alone to base + country ("earliest possible entry");
  - removed alone from the full profile ("latest possible exit").

  The paper says the two positions bound intermediate positions only if the effect is monotone, which it does not test.
- [PAPER, §4.2] Adding the country sentence to a three-variable base raises median rbc (between-country correlation per item) from −0.03 to 0.52. Other results:
  - The same 0.52 appears with no demographics at all.
  - Stating age as a birth year changes nothing.
  - The label's effect is +0.73 (Fisher z) when added, against +0.20 when removed, because the full profile already carries country information through its other variables.
- [PAPER, §4.2] More information lowers recovery:
  - Political identity added to the base gives −0.12.
  - The socioeconomic block gives −0.13 added and −0.06 removed; household income alone reproduces most of that.
  - Income items recover worse with income in the profile.
- [PAPER, §3.4, §4.2, Fig. 3] **Swapped-label arm.** Every country label is replaced with a wrong one. Scored against the respondents' own country:
  - forward-item median rbc is 0.34, against 0.40 with no label;
  - the paired median difference is −0.096 [−0.142, −0.003].

  Scored against the named country, the arm gains only +0.012 [−0.068, 0.122]. So the label is read, but it does not stand in for the named country.
- [PAPER, App. I] A classifier recovers the country from the no-country full profile 28.7% of the time, against a 3.3% baseline (28.99% in a replicate). The no-country arm is therefore "not a zero-information control".

**5. Calibration and validation (main content).**
- **Levels of recovery.** [PAPER, §3.5]
  - Aggregate: rbc across 30 country means, per item.
  - Individual: rwc within country, per country-item cell.
  - A pooled correlation across all 1,260 cells is reported only as a "labeled counterexample".
- **Pooled versus per-item.** [PAPER, §4.1] Under Qwen 1P the pooled r is 0.822, but the median per-item rbc is 0.443. 20 of 42 items exceed 0.5, and 11 items are inverted (down to −0.82).
- **Inversions come from scale direction.** [PAPER, §4.1, §4.2, Fig. 5, Table A5]
  - 10 of the 11 inversions are among the 13 reverse-coded items. Direction alone explains 66% of the variance in rbc (R² 0.658, rising to 0.746 after pre-declared exclusions).
  - In a 2×2 of label × verbal endpoints, the triple difference is +0.42 [0.13, 0.79] (Fisher z). It replicates in Llama at +0.45, and gives +0.16 with the two largest movers dropped.
  - The forward–reverse gap closes from +0.39 to −0.03.
  - [PAPER, §5.2] The paper's conclusion: "Prompt format is therefore part of the measurement instrument."
- **Individual level.** [PAPER, §4.3]
  - Median rwc is at most 0.028 in every condition. A five-predictor linear benchmark reaches 0.210.
  - First- and third-person framing do not differ: 0.027 against 0.029 (p = 0.785).
  - **In a replicate arm, 95% of the model's within-country variance is run-to-run noise.**
  - Anchoring is the only manipulation that raises rwc, from 0.010 to 0.079.
- **Aggregate and individual recovery dissociate.** [PAPER, §4.3, Fig. 7]
  - Across countries, profile correlation and mean rwc correlate at −0.21 (90% CI [−0.49, 0.10]).
  - 232 of 435 country pairs reverse order between the two rankings. Switzerland vs Israel: 0.91 vs 0.75 on aggregate, 0.014 vs 0.077 individually.
  - [PAPER, §5.3] The paper links this to the ecological fallacy, citing Robinson 1950 and Firebaugh 1978.
- **Rank versus level and spread.** [PAPER, §4.1, App. F, Table A5]
  - Every country's mean is biased upward, by 0.36–1.40 points.
  - Between-country dispersion ratio (model to human) is 0.28. The within-country SD ratio is 0.57–0.61, against a prespecified 0.9–1.1 band.
  - Lin's concordance has a median of 0.036. 87% of the agreement attainable at the observed ranking is lost to bias and compression.
- **Comparators that use no LLM.** [PAPER, §4.1, Table A5]
  - The average of regional neighbours gives median rbc 0.55 against the model's 0.44, with normalised error 0.056 against 0.180. It is better on level for 41 of 42 items.
  - Adding log GDP raises the comparator to 0.62.
  - The paper calls these comparators "a standard of adequacy".
- **Noise floor.** [PAPER, §3.5, §4.4] Two identical unseeded runs differ by a median 0.060 rbc per item (0.079 Fisher z). Every interpreted effect is read against this floor.
- **Thresholds.** [PAPER, §3.5] Thresholds were fixed "before any counts were taken". The direction coding was frozen from the codebook before results were seen.

**6. Failure modes, biases, limitations.**
- [PAPER, §5.1] The model compresses spread, shifts every country the same way, and orders countries without matching the distances between them.
- [PAPER, §5.1] The paper cannot tell calibrated country knowledge from stereotype activation. Leaning on the label "could be sound cue use".
- [PAPER, §4.4, App. L–M] Llama 3P lost 22.7% of responses, concentrated on wide scales. Restricting to items with >80% coverage moves median rbc from +0.17 to about 0, so the well-covered subset is biased, not cleaner. Four alternative parse rules move medians by at most 0.013.
- [PAPER, §5.4] Limits: English prompts, generic parties, one survey programme, two 7–8B models.

**7. Software engineering and reproducibility.**
- [PAPER, Table A5] 17 robustness checks, each with a column for whether any conclusion changed. One did.
- [PAPER, App. M] Re-parsing 11.3 M raw responses also checked the accumulator; the worst disagreement in a country mean was 1.8 × 10⁻¹⁵.
- [PAPER, §3.3] Replication materials are public.

**8. Small-N, family, emotion, long horizon.** None.

**What this paper adds beyond the fidelity papers already on file** [INFERENCE]
- **A group-level default that explains most of the aggregate fit while individuals carry nothing.** This is shown with a design that separates the label from composition (add-alone, remove-alone, swapped label) and is replicated in two models. Argyle (OM) shows compression and modal collapse. Kutzner (CB2) argues that subgroup error can offset. Neither isolates a group label as the source of the fit.
- **Measured aggregate–individual dissociation on the same units.** CB2 and OM1 reason about pooling. This paper measures that ranking units by fit at one level gives no information about fit at the other.
- **A polarity mechanism for sign inversion.** It is tested experimentally and is not on file anywhere.

**Not transferable / cautions**
- **Silicon sampling has no EPModel counterpart.** M3.D.6 stands, and this paper is further support for it: individual-level persona outputs here are mostly sampling noise.
- **The non-LLM comparator scores against real survey data.** EPModel has none, so "beats the comparator on accuracy" cannot be ported. Only the structural idea transfers: a trivial predictor built from information already held, used as a reference.
- **Level accuracy and concordance need true magnitudes.** The corpus supplies none, so OM1's sign-only form is the right ceiling. Lin's concordance does not transfer.
- **Some family-level drivers are correct by design.** `M1.D.7` ambient anxiety, the `M1.D.1` budget and `M1.D.4a` capacity are family-level on purpose. The paper's failure is a person-level output computed from a group default, not the existence of group-level inputs. Any test built from this paper must leave the family-level quantities the theory specifies untouched.
- **Correlation with t0 attributes is not a defect in itself.** The theory says `basic_level` should predict a great deal (M7.E.1). Predictability from a person's label is informative only for criteria that credit history or position.

---

### B. Candidate additions to the EPModel spec

**SC1. Group-mean substitution mutants for person-level inputs: the executable test of M1.A.7, plus a per-criterion classification of what carries it.**
- **Proposed requirement.** The `M11` mutation suite **MUST** include two group-mean substitution mutants:
  - (a) **Family mean.** Each person's per-person input is replaced by the mean of that input over their family at the same tick. The inputs are `programmed_reactivity` (M1.A.7), accumulated witnessed history (M1.F.5), and the person's functioning-position term in M1.A.7a.
  - (b) **Class mean.** The same replacement uses the mean over a declared position class: generation, or spouse/child.

  Both mutants **MUST** preserve the family-level mean and remove only within-family variation. Family-level fields (`M1.D.1`, `M1.D.4a`, `M1.D.7`) are untouched. The rules for using them:
  - `M11.C.2`, and any criterion whose readout names a member (`M17.D.2(a)`), **MUST** turn red under (a).
  - Every `M11.C` criterion **MUST** be classified, in its verdict, as **requires within-family variation** (red under (a)) or **carried at family level** (green under (a)).
  - A criterion in the second class **MUST NOT** be cited as evidence for a mechanism that acts through individual history or position.
- **Tests.** `test_m1a7_family_mean_substitution_turns_m11c2_red`; `test_m11_criteria_classified_by_family_mean_mutant`.
- **Where it would live.** M11 mutation protocol (beside `M11.1b`), reported under `M11.4f`. The classification goes in the `M17.G.1` audit record.
- **Evidence.** SHOWN in the paper:
  - a group label carries nearly all aggregate recovery (§4.2: −0.03 → 0.52);
  - individual-level recovery stays near zero (§4.3: rwc ≤ 0.028; 95% of within-country variance is noise).

  The transfer to a substitution mutant is INFERENCE. M1.A.7 is a prohibition with no test; my search found none.
- **What it changes.** It adds a test for an existing requirement and a reporting rule. It catches the EPModel form of the paper's failure: a person behaving from a family or position default while their own witnessed history is inert. This improves theory fidelity (M1.A.7, M7.E.1b: "siblings grow up in the same family but in different triangles") and inference validity.
- **Overlap.** Partly covered:
  - `M11.1b` permutes an input across persons. That keeps within-family variance and scrambles who gets what. This mutant removes the variance and keeps the mean, which is the paper's between/within split.
  - J4 replaces the exchange with averaging between partners, a different object.
  - EV3 attributes `basic_level` pathways.
  - `M17.D.2` disables a named mechanism.

  None of these tests M1.A.7, and none classifies criteria by level.
- **Cost / risk.** Low: two mutant switches. The class definition is `[I]` and must be frozen under `M10.B.4`. Mutants are model changes under `M11.4f`, and the engine stays arm-blind (`M17.D.3`), so there is no conflict with `M3.D.4`/`M3.D.5`, `M3.D.6`, `M11.F.9` or `M16.B`.

**SC2. Two-level decomposition of every person-level readout, with how reproducible within-family differences are across seeds at a fixed initial state.**
- **Proposed requirement.** For each person-level `M11.G.1` component (1, 4, 5, and per-person symptom and anxiety), Phase E **MUST** report two parts separately:
  - the **between-family** part: variance of family means across `D0` draws;
  - the **within-family** part: variance of persons around their family mean.

  It **MUST** also report, for a declared number of `D0` draws each run under several seeds, the share of within-family variance that is **reproducible across seeds**: a person-identity intraclass correlation at a fixed initial state. That share **MUST** be reported for the reference family and for the homogeneous-family arm (`M17.D.1(c)`).
  - A direction **SHOULD** be asserted: the reproducible share is higher in the reference family.
  - Results at the two levels **MUST** be reported separately. A ranking of families on a family-level readout **MUST NOT** be used to choose families or seeds for a person-level claim.
- **Where it would live.** Phase E, `M17.C` (as an extension of `M17.C.1`'s variance components) and `M17.B`.
- **Evidence.** SHOWN in the paper:
  - 95% of within-country model variance was run-to-run noise (§4.3);
  - aggregate and individual recovery correlated at −0.21 across countries, with 232 of 435 pairs reversed (Fig. 7).

  ARGUED: relations among group means need not hold among individuals (§5.3). The transfer is INFERENCE.
- **What it changes.** It adds a readout and a test. `M17.C.1` decomposes variance by source (seed, initial condition, spells, constants) at family level, not by level. Without this, a family whose members differ only by seed noise looks the same as one whose members differ by position, which is the paper's failure in EPModel terms.
- **Overlap.** Partly covered:
  - WA1: within-run split-half reliability and an identical-chooser reference, but no across-seed reproducibility at a fixed initial state.
  - CD2: within-seed computation and the distribution of who is most affected.
  - CB2: position strata.
  - OM1: pooled against within-stratum signs across families.

  New here: the two-level split and the person-identity ICC across seeds.
- **Cost / risk.** Moderate compute (draws × seeds). It is observer-side. Theory caution: `M7.E.1c` makes projection-target selection history-dependent, so some seed dependence is legitimate. The direction is therefore asserted only against the homogeneous arm, never as an absolute. The ICC margin is `[I]`. No rule conflict.

**SC3. Every imported rating and every readout declares its polarity by named endpoints; a reverse-coded fixture must import to the same state.**
- **Proposed requirement.**
  - Every rating exported under `M15.A.4` **MUST** name both scale endpoints in words, for example "1 = cut off, 5 = fused". The importer **MUST** reject a rating whose polarity is not declared.
  - Every `M11.G` component and every readout an `M11.C` criterion asserts on **MUST** declare which end is "more", in words, in the readout register.
  - Test: an import fixture whose ratings are reverse-coded with declared polarity **MUST** produce internal state identical to the forward-coded fixture. A variant with polarity stripped **MUST** be rejected.
- **Test.** `test_m15a4_reverse_coded_fixture_imports_identically`.
- **Where it would live.** `M15.A.4` (an amendment), `M11.G.1`, and the readout register.
- **Evidence.** SHOWN in the paper:
  - 10 of 11 inversions are on reverse-coded items, and direction explains 66% of item-level variance (§4.1);
  - verbal endpoints remove the inversion, with triple difference +0.42 [0.13, 0.79], replicated in a second model (§4.2, Table A5).

  The transfer is INFERENCE. The import is a contract on an application the owner controls, and rating scales on genograms run in either direction.
- **What it changes.** It adds a constraint and a test. `M15.C.1` already warns about one sign inversion (warmth wired to investment). This generalises the guard to the convention that causes the whole class of inversions: an undeclared scale direction between producer and consumer. A silent inversion flips the sign of a directional result, which `M15.D.4` says is the least robust output.
- **Overlap.** Partly covered:
  - `M15.A.4` requires "the definition of the scale point", which does not say direction;
  - `M15.C.1`, for one field;
  - `M11.1c` re-encoding mutants, which rescale but do not reverse.

  My keyword search found no polarity rule.
- **Cost / risk.** Low. It is a schema field plus one fixture. No rule conflict.

**SC4. An amendment to OM2: test each imported field class at two positions, and add a swapped-value arm.**
- **Proposed requirement.** When OM2's import-information ablation is run, each imported field class (topology, tie kinds and states, dated events, ratings) **MUST** be tested at two positions:
  - **added alone** to the no-import baseline;
  - **removed alone** from the full import.

  Both effects **MUST** be reported. The report **MUST** also include a **swapped-value arm**, in which the field class is filled from a different declared family or permuted among persons. The arm's result **MUST** be reported against both the true import and the removed form.
  - If swapped ≈ true, the field is not read.
  - If swapped is worse than removed, the field is read but not used as a stand-in for the family it came from.

  Where a removed field can be partly reconstructed from the remaining fields (for example tie state from dated events), the report **MUST** say that the removed arm is not a zero-information control.
- **Where it would live.** `M15.D` (with OM2), Phase E.
- **Evidence.** SHOWN in the paper:
  - the add/remove asymmetry of +0.73 against +0.20, caused by redundancy with other blocks (§4.2);
  - the swapped-label placebo at −0.096 [−0.142, −0.003] (Fig. 3);
  - residual identifiability of 28.7% against 3.3% (App. I).

  The transfer is INFERENCE.
- **What it changes.** It adds a test design. OM2 as written removes each field class only from the full import. That understates any field that is redundant with others, and cannot tell "ignored" from "read and used".
- **Overlap.** OM2 (amended here), `M15.D.3`, NG1 (assign vs condition), NG2 (tick-0 diff).
- **Cost / risk.** Compute is roughly 3× OM2's. The donor family for the swap must be declared and must not be chosen by result (PD4, `M10.B.4`). No conflict with `M11.F.9`, since nothing is fitted.

**SC5. A reference predictor from t0 labels for person-level readouts.**
- **Proposed requirement.** For each `M11.C` criterion that credits individual history or position for a difference among members (`M11.C.2`, and the `M7.E.1b` sibling-difference claims), Phase E **SHOULD** report, per seed, the rank agreement between the readout's within-family ordering and two declared rules that use no dynamics:
  - (a) ordering by t0 `basic_level`;
  - (b) the `M7.E.1c` closed-list rule evaluated at birth.

  Where the simulated ordering is matched by either rule within `M17.A.4`'s margin across the ensemble, the result **MUST** be reported as label-predictable, and the relationship process **MUST NOT** be credited for it.
- **Where it would live.** Phase E, `M17.B`, beside `M17.D.2`.
- **Evidence.** SHOWN in the paper: an average of neighbours that uses no LLM matches the model on ranking (0.55 against 0.52) and beats it on level for 41 of 42 items (§4.1). The paper frames it as "a standard of adequacy" (§5.4). The transfer is INFERENCE.
- **What it changes.** It adds a reporting rule and a cheap baseline. `M17.D.2`'s rival arm needs a model change and a run. This is analysis-only, and it catches the case where per-person dynamics only re-sort persons along their t0 labels.
- **Overlap.** Partly covered:
  - `M17.D.2`, the rival-mechanism arm;
  - EV3, pathway attribution;
  - SC1, which tests inputs while this tests outputs.

  I would fold SC5 into SC1's classification if the owner prefers fewer items.
- **Cost / risk.** Low. Theory caution: `M7.E.1` predicts that the projection target ends lower, so agreement with rule (a) after the fact is expected for that readout. The rule must be applied to the ordering at t0, and only for criteria that explicitly credit history. The margin is `[I]`. No rule conflict.

**Already covered (one line each)**
- Noise floor from identical replicate runs (§3.5): CD1's same-arm reference, `M17.A.4`'s margin, `M11.D.15`'s placebo arm.
- Pooled figure shown only as a counterexample; per-item results with inversions counted (§3.5, §5.2): OM1 (per-cell signs, matches and misses together), `M17.G.3`, CB2 (weakest stratum).
- Thresholds and coding frozen before results: `M10.B.4`, CD4, WA4.
- Robustness table with a "conclusion changed?" column: `M17.G.1` audit record.
- Loss concentrated on some items, where restricting to well-covered items biases the result (§4.4): `M17.F.1(c)`, `M11.D.18` (fallback rate per move and per person).
- First- vs third-person framing null: no engine counterpart.
- Compression of spread as a validity dimension: CB3, TM1, WA1.

**Narrator / LLM line only (does not enter v2; `M3.D.6` stands)**

**SC-X1. Narrator and persona recovery is reported within label strata and against a label-only baseline.**
- **Proposed protocol item.** BB1's recoverability test, and any check that a narrator or persona tracks per-person state, **MUST** report recovery:
  - pooled;
  - **within** each role or age label (parent, child, "15-year-old");
  - against a baseline that predicts the ordering from the label alone, with no narration.

  Recovery that is matched by the label-only baseline, or that disappears within strata, **MUST** be reported as the narrator answering from the label. A swapped-label arm (correct state, wrong role label) **MUST** be run, and the effect of the wrong label on state claims reported.
- **Evidence.** SHOWN: the label carries the aggregate signal while within-group recovery is about 0 (§4.2–§4.3); a wrong label lowers recovery below no label (Fig. 3).
- **Overlap.** Partly covered:
  - MM3 asserts the label arm is null on state claims;
  - BB1 recovers ordering, pooled;
  - X2 and WA-X1 cover dispersion;
  - CD-X1 covers between-agent differences.

  New here: the within-label recovery requirement and the label-only baseline. Without them, BB1 can pass on role labels alone.

**SC-X2. Any numeric scale given to an LLM names its endpoints in words, and the format is reported with the result.**
- **Proposed protocol item.** This covers questionnaire readouts in the maturity experiments, or a narrator asked for a number.
  - Every numeric scale **MUST** carry verbal endpoints.
  - Any reverse-coded item **MUST** be reported separately.
  - The response format **MUST** be stated with every result.
- **Evidence.** SHOWN: a format change reverses the direction of the ranking on reverse-coded items in two models (§4.2, Table A5).
- **Overlap.** Partly covered by X1 (closed-action readout beside questionnaires) and X10. New here: the format sets the sign, not just the noise.
