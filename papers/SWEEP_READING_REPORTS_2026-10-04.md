# Reading reports: the five papers from the 2026-09-28 sweep

Produced 2026-10-04 at the owner's request ("review these papers for items that should be added to the overall spec"). The 2026-09-28 sweep (`DIGEST.md`) had screened all five on their abstracts and flagged none as a deep-review candidate. This round reads each in full from `pdftotext -layout` output. Each reader worked under `sweep_readers_brief_2026-09-19/READER_TASK.md` and checked for overlap by keyword search in four places: spec revision 10, `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, both `SWEEP_READING_REPORTS_2026-09-27*.md` files, and the "Spec revision 11 — held" section of `TODO.md`. The reports are reproduced as written. Only the heading levels were changed.

**Status: none of these candidates is in the spec.** Plain-language explanation with examples: `SPEC_CANDIDATES_plain_language_2026-10-04.md`.

| Part | Paper | Candidates |
|---|---|---|
| 1 | Ezaki, Imura & Nishinari, *Warned alike, AI agents avoid the less-crowded road while people take it*, arXiv 2609.30883v1 | WA1–WA5, WA-X1 |
| 2 | Chae, Choi, Lee & Im, *Cultural Divergence Preservation*, arXiv 2609.29928v1 | CD1–CD4, CD-X1 |
| 3 | Kutzner, Kacperski, de Molière, Chidichimo, Jung, Wallis & He, *Consequential Behaviour and Representational Fairness in the Validation of Synthetic Research*, arXiv 2609.27690v2 (position paper, no data) | CB1–CB3 |
| 4 | Guo, Song, Yao, Zhang, Yi, Xie & Zhou, *Bayesian Personalized Value Alignment* (BaCVA), arXiv 2609.28942v1 | BV1–BV3 |
| 5 | Qraitem, Saenko & Plummer, *PERSONAWEAVER*, arXiv 2609.26629v1 | PW1–PW3, PW-X1 |

Candidates labelled `-X` are for the exploratory narrator line only and do not enter the v2 spec.


---

# Part 1: Ezaki, Imura & Nishinari 2026 (Warned alike)

## Reader report: Ezaki, Imura & Nishinari 2026 (Warned alike)

### Scope

I read all of Ezaki, Imura & Nishinari, arXiv 2609.30883v1: main text, Materials and Methods, Supplement S1–S10, Tables S1–S14 and the figure captions. The source was the 2,368-line pdftotext file, read in four chunks. Figures survive only as captions and in-text numbers. Before reading the paper I read `READER_TASK.md`, `EPMODEL_BRIEF.md` and `DESIGN_LESSONS_model_design_papers_2026-09-17.md` in full.

Spec revision 10 passages I read directly:
- `M3` in full, including `M3.D.4a/b` and `M3.E`.
- `M4.D.1`–`M4.D.1f` and `M4.D.6`–`M4.D.6e`.
- `M1.B.11`, `M1.F.1`–`M1.F.7`, `M9.4`.
- `M10.B.4`, `M10.C.1`.
- `M11.C.2`, `M11.C.16`, `M11.C.38`, `M11.C.40`, `M11.G`.
- `M16.A.1`–`M16.A.10`.
- `M17` in full.

Keyword searches, all returning 0 hits in the spec: lockstep, herd, dispersion, split-half, binomial, broadcast, screening, confirmatory, disjoint/held-out seeds. Terms with hits that I followed up: synchron, homogen, heterogen, simultaneous, oscillat, variance, burden.

I ran the same searches over:
- `SPEC_CANDIDATES_from_preprints_2026-09-20.md` (C1–C42, X1–X4).
- `SWEEP_READING_REPORTS_2026-09-27.md` (J, SV, PD, X5–X9).
- `SWEEP_READING_REPORTS_2026-09-27b.md` (BB, MM, AD, TM).
- The "Spec revision 11 — held" section of `TODO.md`.

### A. Report

**What the paper is.** It is a congestion game: two identical roads, and travel time on a road is 20 + 80·(share of drivers on it) minutes. Four broadcast conditions were tested:
- F0: no report.
- F1: yesterday's numbers.
- F2: a tip naming the less-crowded road.
- F3: the tip plus one sentence warning that others will also switch.

Populations: 50 GPT agents (gpt-5.4-mini), each a stateless call per round, plus 480 human participants on Prolific in 12 all-human rooms and 24 mixed human–agent rooms of 20 seats.

**1. Simulation formalism**
- [PAPER] Synchronous rounds: all agents choose at once, and feedback is immediate (Fig. 1A). The round-1 split is random; choices begin in round 2. Analysis windows are rounds 6–60, or 6–40 for 20 seats (Results, Methods).
- [PAPER] The seed fixes the initial routes, traits, label mapping and the baseline's draws. No provider sampling seed was set, so a repeated seed reproduces the setup, not the outputs (Methods; S1).
- [PAPER] Ties are handled explicitly. When the previous round's counts were equal, there is no "less-crowded road", so that round is excluded from tip-following summaries and kept in I, M and travel time. Tie counts are reported (S4). The environment's tie-break always named physical road 0 (S2).
- [PAPER] Eq. 1 is an exact identity. Mean cost = 60 + 160[(mean tip-following − ½)² + Var(tip-following)]. It splits cost into a "lean" term and a "volatility" term (Results; S5; largest residual 5×10⁻¹⁴ min).

**2. Agent architecture and heterogeneity (the owner's main question)**
- [PAPER] The agents' only individual difference was a time-sensitivity trait drawn from U(0.8, 1.2), rendered as one of three phrases. Each agent also saw its own five-day history (S1).
- [PAPER] Under F3, 98 of 100 agents followed the tip on fewer than 10% of rounds: mean 0.03, SD 0.03. The SD for identical independent choosers at the same mean is 0.03. Split-half r was 0.14 under F1 and 0.23 under F3 (Fig. 3D; S4). So agent dispersion was what sampling noise alone would give, and individual tendencies barely carried over between halves of a session.
- [PAPER] Humans were more dispersed: SD 0.25 under F1 and 0.27 under F3, against a reference of 0.09. Their tendencies were reproducible: split-half r = 0.73 and 0.81. Mean tip-following was 0.51 under both messages (Fig. 3D).
- [PAPER] Under the reasoning setting, agent dispersion was *below* the independent-chooser reference: SD 0.044 and 0.041 against 0.059 and 0.054 (S9).
- [PAPER] Under F3, agents ignored their own experience. P(switch) was 0.03 after a slow day and 1.00 after a fast day. For humans it was 0.47 and 0.43 under F3, and 0.73 and 0.22 in mixed rooms (Fig. 3E; Results).
- [PAPER] The source of human heterogeneity is not settled by the paper. The authors call dispersed tendencies "one plausible route" to balance and say the design "does not isolate their contribution" (Discussion). Observed tendencies "combine persistent individual differences, learning and shared room history" (Results). The questionnaire shows a mixture of strategies. Under F1, 35 said they took the faster-looking road, 62 the other road expecting others to switch, and 17 "it depended" (Table S6).
- [PAPER] Strategic anticipation is therefore present in humans too, but spread across people rather than shared. Humans were not instructed to anticipate others; agents were (S7). That confound is the authors' own stated limit (Discussion).
- [PAPER] The authors' account of the agent result: "A common model supplies similar decision rules" (Discussion), so one forecast produces the same anticipatory response in every recipient, "even when their personal histories differ".

**3. Interaction and influence**
- [PAPER] There is no direct communication. Coupling runs only through congestion and the shared broadcast (Discussion).
- [PAPER] Even without a broadcast (F0), shared feedback produced lockstep oscillation. All 29 agents on the slow road switched, giving a 2:48 split. M = 0.99 and travel time was 96 min (Results; Fig. 1C).
- [PAPER] A rule-based multinomial-logit learner with heterogeneous β (0.10·U(0.8, 1.2)) was also tested (Methods; Table S1).
  - With private experience only, it was near balance: I = 0.06.
  - Adding the shared numbers (F1) made it oscillate: I = 0.44, M = 0.88, 90.7 min.
  - So rule-based agents with modest parameter heterogeneity also align once they share one input signal.
- [PAPER] Selective exposure was non-monotone. In 20-agent populations, median I at 0, 5, 10, 15 and 20 warned agents was 0.23, 0.14, 0.20, 0.35 and 0.47. A linear mix of the two pure populations' responses under-predicted imbalance near the minimum: 0.03 predicted against 0.14 observed (Fig. 2D–E; Table S7).
- [PAPER] In mixed rooms, humans moved onto the road the agents avoided. Human tip-following was 0.65, 0.83 and 0.91 at K = 5, 10 and 15 agents. At K = 15, human seats averaged 43.5 min and agent seats 80.2 min. Of 240 participants, 129 averaged under 60 min; no agent did (Fig. 5).
- [PAPER] Summary sentence: "A better group average can also hide an unequal burden" (Abstract).

**4. Initialisation**
- [PAPER] Switching the message at round 31 overrode the established state in all seeds, in both directions (F3→F1 and F1→F3; Table S9). The authors say this "do[es] not establish weak initial-condition dependence in general" (Results).

**5. Calibration, validation and robustness**
- [PAPER] Prespecification:
  - Two confirmatory claims, a classifier, seeds 3–12, prompts and model were fixed in a release manifest with SHA-256 hashes of 29 files (commit 31300fb).
  - Screening runs used a separate seed (90) and are excluded from the analyses (S4).
  - The frozen classifier was fixed in advance: I ≥ 0.25, M ≤ 0.20, and at least ten consecutive rounds with the same majority road. It is extended with oscillation, near-equilibrium and partial labels (S4).
- [PAPER] Tests: exact two-sided sign tests on 10 paired seeds (all ten in one direction gives p = 0.00195). A claim's p is the maximum over its components, then Holm-adjusted (S4).
- [PAPER] The rule-based comparison is a difference-in-differences: (F0 − F1)_LLM − (F0 − F1)_MNL on the same seeds (S4).
- [PAPER] The drift control repeated F3 four weeks later with prespecified drift flags (|ΔI| > 0.05, |ΔM| > 0.10). None was triggered (S1).
- [PAPER] Mixed-group inference uses a block sign-flip test, p = 1/256. Grouping blocks by shared recruitment window leaves only 16 and 4 configurations, with p = 0.0625 and 0.25. The authors report this weakening openly (S8).
- [PAPER] Robustness is bounded, not uniform. Paraphrase 2 did not freeze (83 min). Claude Haiku and Gemini shifted in the same direction but never froze. Reasoning removed freezing. GPT-6 Luna at medium reasoning froze under the tip alone (Fig. 2G; S9; Tables S13–S14).

**6. Failure modes and limits**
- [PAPER] All prompts told agents to consider others' reactions, which may favour anticipatory responses (Discussion).
- [PAPER] Human and agent benchmarks were run at different times with different incentives.
- [PAPER] Mixed-room composition was associated with recruitment order. Some rooms were completed after outcomes had been seen (Discussion; S8).
- [PAPER] Reason codes describe the text agents generated, not their internal computation (Results; S4).

**7. Software engineering**
- [PAPER] Hashed release manifest; client-side request audit; preserved records of aborted attempts; a `VALIDATION.json` that recomputes every reported quantity (S4, S8, S10). The reproducibility archive is not public yet (S10).

**8. Small N**
- [PAPER] 20-seat rooms, with exact independent-chooser references for N = 20: I = 0.088 and 62.0 min (Statistics). Nothing on families, emotion or long horizons.

**Not transferable / cautions**
- [INFERENCE] The game rewards anti-coordination, which a family system does not. Bowen theory may predict *more* alignment of reactions in more fused families. I have not checked `_LEDGER.md` on this, so it is an owner question, not a finding. Any EPModel test must assert a difference between arms, never "no alignment".
- [INFERENCE] Effect sizes, the 64 → 95 min result and the human numbers are not calibration targets for anything in EPModel.
- [INFERENCE] The seat-type cost finding is the paper's version of "the average hides who carries it". EPModel already builds that into its readout. **Already covered** by `M11.G.1` component 1, `M11.C.2`, the `M1.D` sinks and J1 (held).
- [INFERENCE] The switch experiments (the collective state follows the message currently in force) are **already covered** by SV2 (held) and `M17.D.5` (both directions of a perturbation).
- [INFERENCE] The frozen classifier with prespecified thresholds is **already covered** by `M17.B.2`.
- [INFERENCE] Paired sign tests with multiplicity correction are **already covered** by `M17.A.3`, `M11.4e` and C25.
- [INFERENCE] The drift-control repeat is **same as X6** (held), on the narrator line.
- [INFERENCE] Prompt sentences acting as interventions is **already covered** by DESIGN_LESSONS §3.3.

**Does EPModel's rule-based design risk the same thing?** [INFERENCE] Partly.
- Against it: per-edge latency (`M1.B.11`), per-hop fidelity (`M1.F.4`), witness position (`M1.F.5`), per-person `basic_level` and reactivity, and per-actor selection draws (`M3.D.4b`) all spread out both the input and the response.
- For it:
  - Some inputs reach every person at once and enter every propensity vector: standing load, societal anxiety and family-keyed exogenous spells (`M3.D.4b`).
  - The paper's rule-based baseline aligned under exactly that kind of shared input, despite ±20% parameter heterogeneity.
  - The spec has the homogeneous-family arm (`M17.D.1(c)`) but no statistic that would show whether the reference family's members behave more distinctly than identical people would. Nothing in the spec would detect lockstep.

### B. Candidate additions

**WA1. Between-person dispersion and split-half reliability of move tendencies, against an identical-chooser reference**
- **Proposed requirement.** Phase E **SHOULD** report, per seed and inside `M17.F.2`'s settled window:
  - each person's tendency (the share of selected outcomes in a declared move class, at minimum the self-directed-channel share);
  - the between-person SD of that tendency against the SD expected for identical independent choosers at the family mean;
  - the split-half correlation of tendencies.

  An acceptance criterion **SHOULD** assert that both quantities are higher in the reference family than in the homogeneous-family arm (`M17.D.1(c)`).
- **Where.** Phase E, `M17.B` (readout) and `M17.D.1(c)` (assertion). The log side needs nothing new: `M16.A.3` and `M1.F.1a` suffice.
- **Evidence.**
  - Fig. 3D and S4 (SHOWN in the paper): agent dispersion equalled the identical-chooser reference with split-half r 0.14–0.23; humans were at about 3× the reference with r 0.73–0.81. The agents' three-phrase trait did not produce persistent differences.
  - S9: dispersion below the reference also occurs.
  - Transfer to EPModel is INFERENCE.
- **What it changes.** Adds a test that parameter heterogeneity actually yields persistent individual behaviour, which is what the LLM trait failed to do. Improves inference validity: a criterion that "holds" in a family whose members behave interchangeably is weaker than it looks.
- **Overlap.** Partly covered by:
  - X2 (between-persona SD, LLM line only).
  - TM2 (manipulation check between arms, not between persons in one family).
  - `M11.C.38` (ordering across runs).
  - `M17.B.5` and `M16.A.9`, which are descriptive and have no reference.

  New here: the within-family comparison with the identical-chooser reference, and split-half reliability.
- **Cost / risk.** Low; observer-side, so it is consistent with `M16.B`. At N ≈ 12 a per-seed correlation is noisy, so it should be reported as a distribution across seeds (`M17.B.4`).
  - Theory caution: functioning position changes within a run (`M1.A.7a`), so low split-half reliability over decades may be legitimate. Use windows inside a settled period.
  - The reference is a sampling benchmark, not a model, and the margin is `[I]`.
  - No conflict with `M3.D.4`/`M3.D.5`, `M3.D.6` or `M11.F.9`.

**WA2. Alignment of responses to a shared input, with an onset-spread readout**
- **Proposed requirement.** For every input that reaches several persons in the same tick (exogenous spell onset, societal-anxiety step, nodal event), the run log **SHOULD** record each exposed person's first selected outcome and its tick after the input. Phase E **SHOULD** report two things per input class, per seed:
  - the concentration of first responses across exposed persons (1 − normalised entropy over the `M5` repertoire);
  - the spread of onset ticks.

  A criterion **SHOULD** assert that concentration is lower, and onset spread higher, in the reference family than in the homogeneous-family arm. `M17.B.6`'s regression **SHOULD** add the shared-input term as a separate regressor, so that the weight of shared input against person-specific state is reported.
- **Where.** `M16.A` (record) and Phase E `M17.B`/`M17.D.1(c)`.
- **Evidence.**
  - Table S1 (SHOWN): the rule-based MNL learner with heterogeneous β went from near balance (I = 0.06) to oscillation (I = 0.44, M = 0.88) once given one shared numeric signal.
  - Fig. 1C (SHOWN): GPT F0 lockstep, M = 0.99.
  - Fig. 3E (SHOWN): under F3, own experience had no effect on choice.
  - The authors attribute alignment to shared input meeting similar decision rules (Discussion; ARGUED).
  - Transfer is INFERENCE.
- **What it changes.** Adds a readout and a test for lockstep, which the spec cannot currently detect. Improves inference validity, because a mechanism result produced by a whole family moving as one unit on a shared input is a different claim from one produced by the relationship process. It may also improve theory fidelity; see the owner question below.
- **Overlap.** Partly covered by:
  - `M17.B.6`, whose conformity terms measure response to *received moves*, not co-selection driven by a common input.
  - `M1.B.11`, which staggers delivery of events but not of family-wide inputs.
  - PD3 (held), on staggering the slow tick only.
- **Cost / risk.** Low to moderate: a new observer record and no engine change.
  - Owner question before any direction is asserted: does the corpus say members of a more fused family react more alike to the same event? If so, the criterion should assert that alignment *rises* as `basic_level` falls, not merely reference > homogeneous. Check `_LEDGER.md` first.
  - The concentration margin is `[I]`.

**WA3. Within-run mean and within-run temporal variance reported as separate arm differences**
- **Proposed requirement.** For every family-level readout used in an `M11.C` criterion, Phase E **SHOULD** report, per seed, the within-window mean and the within-window variance over ticks, and the paired arm difference on each. A criterion that asserts a mean difference **MUST** also report the variance difference, and **SHOULD** flag arms where the two move in opposite directions.
- **Where.** Phase E, `M17.B`.
- **Evidence.**
  - Eq. 1, Fig. 3C, S9 (SHOWN): for Claude and Gemini, the warning raised the lean term by 12.5 and 10.7 min and lowered the volatility term by 9.8 and 5.4 min, in every pair. The net cost change was small and hid two opposed effects.
  - The exact identity depends on the quadratic cost. The general reporting rule is INFERENCE.
- **What it changes.** A reporting rule; improves inference validity. It is theory-relevant because `M5.E.6` and `M11.C.5` describe change-back as a damped oscillation, so an intervention can lower mean tension while raising volatility.
- **Overlap.** Partly covered by `M17.B.2` (an oscillating regime label, but no variance term), `M17.F.1(a)` (trajectory through a shock only) and J1 (a different decomposition).
- **Cost / risk.** Low, observer-side. No rule conflict. EPModel readouts have no quadratic identity, so the two terms are reported side by side, not as a sum.

**WA4. Development seeds disjoint from acceptance seeds, declared before the suite runs**
- **Proposed requirement.** The seed range used while writing code and choosing `[I]` constants **MUST** be declared and **MUST** be disjoint from the seed range used by the acceptance suite and Phase E ensembles. The acceptance seed range **MUST** be recorded in the `M16.A.1` header beside `M10.B.4`'s frozen values before the suite first runs. A result computed on a development seed **MUST NOT** be reported as acceptance evidence.
- **Where.** `M10.B.4`, `M16.A.1`, `M17.A.1`.
- **Evidence.** S4 (practice SHOWN): screening at seed 90 used the preceding code revision; confirmatory seeds 3–12 were fixed in a hashed manifest before the first confirmatory call and had not been run before. That this prevents seed-level overfitting is ARGUED and INFERENCE.
- **What it changes.** A reporting and process rule; improves inference validity. It closes a gap `M10.B.4` leaves open: constants can be frozen and still have been chosen while watching the very seeds later used to test them.
- **Overlap.** Partly covered by `M10.B.4`/`M16.A.7` (constants only) and SV4 (held; counts versions, not seeds).
- **Cost / risk.** Negligible. It interacts with `M17.A.1`'s adaptive blocks: the acceptance range must be open-ended or capped, and declared either way. No conflict with `M3.D.4a`; keyed draws make seed ranges cleanly separable.

**WA5. Sweep the number of recipients of an intervention event; do not interpolate from none-versus-all**
- **Proposed requirement.** Where arms differ in which family members receive an intervention event (the coaching-knowledge event in the held owner answer to SV2), Phase E **SHOULD** run the arm at every recipient count from 0 to the number of eligible members, or at four or more declared counts. It **SHOULD** report the response curve and its residual against a linear mix of the zero-recipient and all-recipient arms, and flag non-monotonicity.
- **Where.** Phase E, `M17.E`, next to the SV2 arm definition.
- **Evidence.** Fig. 2D–E and Table S7 (SHOWN): non-monotone in the number warned, with the minimum near 30% exposure. Linear mixing and an independent-choice benchmark both under-predicted imbalance near the minimum (0.03 and 0.09 against 0.14 observed). Transfer is INFERENCE.
- **What it changes.** Adds a test design; improves inference validity. Mixed exposure in a coupled system is not the average of the pure cases, and "who receives it" is already an arm dimension under the owner's SV2 answer.
- **Overlap.** Partly covered by `M17.E.2` (dose–response over an `[I]` constant) and `M11.C.38`/C16 (monotonicity over person parameters). A recipient count is neither. Also partly covered by SV2 (held), which defines the arm but not the sweep.
- **Cost / risk.** Compute scales with family size. With about 12 members, subsets must be declared rather than enumerated, and *which* members receive the event matters as much as how many, so stratify by position. No rule conflict. Recipients must be chosen by declaration, not by the engine (`M17.D.3`).

**WA-X1. LLM persona populations: report dispersion against an identical-chooser reference and split-half reliability, not only means** (exploratory LLM line only)
- **Proposed protocol item.** Any test of LLM personas (for example "a 15-year-old brat") **MUST** report each agent's behavioural tendency on a closed action set, the between-agent SD against the identical-chooser reference at the same mean, and split-half reliability. A persona trait that does not raise either above the reference **MUST** be reported as not realised before any mean effect is interpreted. Every sentence shared by all agents' prompts **MUST** be treated as an experimental factor.
- **Evidence.**
  - Fig. 3D and S4 (SHOWN): the trait rendered as three phrases produced SD equal to the reference and r 0.14–0.23.
  - S9: under-dispersion appeared with reasoning.
  - One shared sentence (F3) moved cost from 64 to 95 min and froze 10/10 runs.
- **Overlap.** Partly covered by X2 (between-persona SD; no reference or reliability) and DESIGN_LESSONS §3.3 (prompt sensitivity). Not for the v2 spec; `M3.D.6` stands.

### Questions for the owner

1. Does the corpus say members of a more fused family respond more alike to the same event? This decides whether WA2 asserts only reference > homogeneous, or a direction in `basic_level`.
2. Should WA1 and WA2 be one readout or two?
3. Which family-wide inputs, if any, should be staggered per person (societal anxiety, spell onset)? Is the current simultaneous application an `[I]` choice to be tested in the way PD3 proposes for the slow tick?

---

# Part 2: Chae, Choi, Lee & Im 2026 (Cultural Divergence Preservation)

## Reader report: Chae, Choi, Lee & Im 2026 (Cultural Divergence Preservation)

### Scope

I read all 802 lines of the `pdftotext` output of Chae, Choi, Lee & Im, arXiv 2609.29928v1, dated 24 Sep 2026. That covers the body (§1–5), the Limitations, and Appendices A–E with Tables 1–11. Fig. 2 survives only as its caption and prose, but its numbers are restated in Tables 5–6. I read `EPMODEL_BRIEF.md` and `DESIGN_LESSONS_model_design_papers_2026-09-17.md` in full.

I checked overlap by keyword search in four places:
- **Spec revision 10:** `M17` read in full (A.1–G.3). I also read `M11.G.1`–`G.4`, `M15.D.1`–`D.4`, `M16.A.7`–`A.10` and `M16.B.1`–`B.3`, `M16.E.1`–`E.3`, and the rows or clauses for `M11.C.2`, `M11.1c`, `M11.4d`, `M11.4e`, `M1.A.12`, `M1.A.20` and `M2.A.0`.
- **`SPEC_CANDIDATES_from_preprints_2026-09-20.md`:** C2, C21, C24, C25, C39, X1–X4.
- **The two 2026-09-27 report files:** J2, BB4, TM1–TM6, MM3, X5–X10.
- **`TODO.md`:** the "Spec revision 11 — held" section.

Search terms: JSD/Jensen, flatten, caricature, bootstrap, leave-one, centroid, dispersion, finite-sample, cutoff, fixture, Laplace, concentration, narrat, verbaliz, membership, alive/death across arms.

Statements of the form "the spec has no X" rest on those searches, not on a full read of all 427 requirements.

---

### A. Report

**1. Simulation formalism.** There is none.
- [PAPER] This is one-shot survey generation. There is no time, no interaction and no state (§4.1).
- [PAPER] 300 synthetic respondents per country × method × model. Generation seeds were 42, 1 and 2 (§4.1, App. E.3).
- [PAPER] The controlled perturbation is deterministic and "requires no simulation seed" (App. C).

**2. Agent architecture.** [PAPER] Three persona-prompting methods (App. A):
- **Cultural:** country of birth and residence only.
- **PHub-I:** a one-line PersonaHub persona expanded into an eight-field profile.
- **DeepP-I:** about 25 attributes sampled from a fixed taxonomy and rendered by a template.

Four open-weight models were used: Gemma-3-4B, Qwen3.5-9B, Qwen3.5-27B and Llama-2-13B (§4.1).

**3. Interaction/network structure.** None.

**4. Initialization.** [PAPER] DeepP-I draws its attributes from a taxonomy "not conditioned on the target country". The authors say its flattening may partly reflect this sampling design rather than the model (Limitations).

**5. Calibration and validation.**
- **Metric** [PAPER, §3.1–3.2, Eqs. 1–2]:
  - CD is the mean, over items and countries, of the base-2 JSD between each country's histogram and the equal-weight cross-country centroid.
  - CDP = CD_synthetic / CD_human. Below 100% means flattening; above 100% means caricature.
  - Country-wise fidelity (JSD to the matched human distribution) and CD use the same divergence function but compare different pairs of distributions.
  - The human reference "must be recomputed" when the country set, item set or scale changes (§3.2).
- **Validating the metric by controlled perturbation** [PAPER, §4.2, Fig. 2, Tables 5–6]:
  - Each country's histogram is moved toward or away from the centroid by p̄ + γ(p − p̄).
  - 20% mixing leaves 62.0% of human CD on WVS and 63.7% on Big Five, at a JSD cost of 0.45% and 0.09% of the median LLM error.
  - 40% mixing leaves 34.2% and 35.8%.
  - Complete collapse costs only 9.33% (WVS) and 2.28% (Big Five) of the median LLM JSD.
  - Amplification to γ = 1.5 raises CDP to 231–254% at 0.6–3.9% of the median JSD.
  - Locally, JSD to the original grows roughly as (γ−1)² (App. C).
  - The perturbation is monotone in both directions.
- **Audit** [PAPER, Table 1]:
  - DeepP-I has the lowest raw WD in 7 of 8 blocks and the lowest nWD and JSD in 6, yet CDP is 14–59% in every block.
  - Cultural is 184–501%.
  - PHub-I is 44–99%.
  - At country level, Cultural is over-divergent in 33 of 36 cells and DeepP-I under-divergent in 33 of 36 (App. D, Table 7; bands are below 50% and above 150%).
- **Uncertainty** [PAPER, App. E.1, Table 8]:
  - Respondent-level bootstrap, B = 2,000, resampling whole respondents within each country.
  - Resampling adds finite-sample histogram noise, which raises CD. The "Shift" column is negative for every PHub-I and DeepP-I condition (16 of 16).
  - The DeepP-I gap excludes zero in 7 of 8 blocks.
  - BCa intervals are degenerate on Big Five because the point estimate sits at an extreme of the bootstrap distribution, so percentile intervals are treated as primary.
  - Leave-one-country-out removes the same country from both sides ("matched omission"). The DeepP-I gap stays positive in all 8 blocks.
- **Threshold sensitivity** [PAPER, App. E.2, Table 9]: with the discordance cutoff t at 0.4, 0.5 and 0.6, the count of discordant conditions is 31, 37 and 42, and the set of methods represented does not change. The 50% and 150% bands of Table 7 were not varied.
- **Seeds and weights** [PAPER]:
  - Three seeds. CDP ranges per condition are 0.5–33.2 points, and the only class change is Gemma PHub-I on WVS, at 98.9–103.5% (E.3, Table 11).
  - Survey weighting changes human CD by −0.80% (E.4).
- **What could not be done** [PAPER, Limitations]:
  - CDP is magnitude-only and blind to which side of the centroid a country falls on, so an inverted country pattern can leave it unchanged.
  - It cannot separate culture from survey year, translation, response style or sample composition.
  - The authors: "CDP should therefore not be used as a general quality score" (Limitations).

**6. Failure modes.**
- [PAPER] Good country-wise fidelity and preserved between-country divergence can rank methods in opposite orders (§4.3, §5). "Low distance ≠ preserved cultural divergence" (Fig. 1).
- [PAPER] Among the four lowest-JSD conditions per domain, 6 of 8 have CDPc below 50% (Table 10).
- [PAPER] Persona richness does not protect against flattening. Minimal country prompting produces caricature (Table 1).

**7. Software engineering.**
- [PAPER] Human CD is computed once and reused, and is recomputed when the set changes (§3.2).
- [PAPER] Leave-one-out removes the same unit from both sides of the comparison (App. E.1).
- [PAPER] Weighting is checked as a separate sensitivity (E.4).

**8. Small-N, emotion, family, long horizon.** [PAPER] None. Groups are 3–6 countries of about 486–8,584 human respondents each (Table 4).

**Not transferable / cautions.**
- [INFERENCE] EPModel has no human reference, so CDP itself has no counterpart. The nearest analogue is "divergence in arm B relative to baseline arm A", which is a difference between arms, not fidelity. It must not be called "preservation".
- [INFERENCE] The scale differs completely: hundreds of respondents per group against 11 members, one move per week.
- [INFERENCE] The source is a non-peer-reviewed v1 preprint, with three seeds, four open-weight models, English prompts only, and three countries for Big Five.
- [INFERENCE] The main contribution, "fidelity metrics miss between-group divergence", applies to EPModel only where EPModel aggregates something. The four candidates below are those places.
- [INFERENCE] The magnitude-only limitation is already handled. `M17.D.2(a)` requires naming which member is affected, which is the directional check the authors defer to future work.
- [INFERENCE] The BCa degeneracy is already covered by `M11.4e`'s rank-based paired statistic for degenerate arms.

---

### B. Candidate additions

**CD1. A divergence readout between arms carries a same-arm noise reference and a signed companion.**
- **Proposed requirement.**
  - Wherever `M17.B.5` reports a Jensen–Shannon divergence between arms, the report MUST also give the same statistic between two independent seeds of the same arm, at matched move counts and window length. The arm divergence MUST be stated relative to that reference.
  - JSD MUST NOT be the test statistic of a directional criterion. A criterion on move distributions MUST assert a signed change in named move shares.
- **Where it would live.** An amendment to `M17.B.5`. It could also be folded into J2, which proposes a null distribution for structural readouts over ties and triangles but does not cover `M17.B.5`'s move-distribution JSD.
- **Evidence.**
  - Finite-sample noise inflates a divergence measured against a centroid, and that bias pushes the measured synthetic-to-human gap toward zero (App. E.1, Table 8 Shift column). SHOWN, for their data.
  - JSD grows about as (γ−1)² near zero, so small real shifts register as near-zero divergence (App. C). SHOWN.
  - JSD is blind to direction (Limitations). ARGUED.
  - My own computation [INFERENCE], on an invented 10-category move distribution with Laplace smoothing. Two samples drawn from the same distribution give a mean JSD of:
    - 0.037 bits at 52 moves (1 person-year);
    - 0.011 bits at 260 moves;
    - 0.003 bits at 1,040 moves.
    
    A real 5-point share shift has a true JSD of 0.0036 bits, and a 10-point shift 0.0145 bits. At one person over five years, sampling noise alone averages three times a real 5-point shift.
- **What it changes.** It adds a reporting rule and corrects a readout. Without a reference, every per-seed paired JSD is positive and a rank test on it rejects trivially. This improves inference validity.
- **Cost / risk.**
  - Low: extra seeds that the ensemble already has.
  - Under `M3.D.4a`'s coupled draws, an arm with no effect gives exactly zero until the trajectories diverge. The independent-seed reference is therefore an upper bound on the noise floor, and the report should say so.
  - No rule conflict.

**CD2. Between-member readouts are computed within seed, over a matched member set.**
- **Proposed requirement.**
  - Any readout of how family members differ from one another MUST be computed within each seed and then summarised across seeds. This covers concentration of symptom or anxiety (`M11.C.2`; `M11.G.1` components 1, 2 and 4) and per-person `M16.A.9` entropy compared across members.
  - Such a readout MUST NOT be computed from per-member values that were first averaged across seeds by person identifier.
  - Where the identity of the most-affected member varies across seeds, its distribution MUST be reported.
  - When arms are compared on such a readout, both arms MUST use the same member set over the window: anyone dead, unborn or below the `M1.A.12` membership threshold in either arm is removed from both, and the removed persons MUST be listed.
- **Where it would live.** `M17.B` (per-seed readouts), with a fixture test in `M11`.
- **Evidence.**
  - Mixing distributions toward their centroid removes divergence faster than linearly: 20% mixing leaves 62% (WVS) and 63.7% (Big Five); 40% mixing leaves 34% and 36% (§4.2, Table 5). SHOWN.
  - Averaging seeds in which projection lands on different children is a convex mixing of the same kind, so a family with full concentration in every seed can look uniform across children. INFERENCE.
  - The reference is recomputed when the set changes (§3.2), and matched omission is used in leave-one-out (App. E.1). SHOWN as method.
- **What it changes.**
  - It adds a reporting rule and a test: `test_cd2_seed_pooling_does_not_flatten_member_concentration`. The fixture is two seeds with the projection target on child A in one and child B in the other. Pooling by identifier must turn the test red.
  - It closes a gap. `M17.D.2(a)` names which member, but nothing says how per-member results are aggregated across seeds, and mortality and the derived membership in `M1.A.12` let member sets differ between arms. `M1.A.20` gives stable identifiers, not matched sets.
  - It improves both theory fidelity and inference validity.
- **Cost / risk.** Low and observer-side. Matched omission can discard the most affected person, for example when the projection target dies in one arm. That case must be reported, not silently dropped (global rule P2).

**CD3. A positive-control fixture for each criterion's readout.**
- **Proposed requirement.** Each readout that an `M11.C` criterion asserts on MUST have a deterministic fixture test. In it, the property the readout measures is attenuated and amplified by declared factors on synthetic log input, and the readout MUST move monotonically in both directions.
- **Where it would live.** `M11` test protocol, alongside `M17.B.4`.
- **Evidence.** The paper validates its own metric this way: a deterministic γ grid, both directions, monotone response, no seed (§4.2, Fig. 2, App. C). SHOWN as method.
- **What it changes.** It adds tests. It catches readouts that are insensitive near zero (as in CD1) or saturated, before they are used to decide a direction.
- **Overlap.** PARTLY covered:
  - `M17.B.4` is one instance (the bimodal time-to-event fixture).
  - `M11.1c` tests that readouts do not move under re-encoding, which is the opposite check.
  - `M11.4d` handles a readout already at a bound.
  - BB4 tests engine sub-mechanisms, not readouts; TM5 is a negative control.
- **Cost / risk.** Low. No rule conflict.

**CD4. Analysis-layer thresholds are reported at neighbouring values.**
- **Proposed requirement.**
  - Every `[I]` threshold in the analysis layer that classifies a seed or decides a direction MUST be frozen before the run (`M10.B.4`). That includes `M17.A.4`'s margin, `M17.A.1`'s δ, `M17.B.2`'s regime bounds, `M17.D.4`'s floor, `M17.F.2`'s drift threshold and `M11.4d`'s bound.
  - Each result MUST be reported at the declared value and at declared values either side.
  - `M17.G.1`'s list of dimensions gains "analysis thresholds".
- **Where it would live.** `M17.G.1`, with each threshold declared in `M10`.
- **Evidence.** E.2 and Table 9: 31, 37 and 42 conditions at t = 0.4, 0.5 and 0.6, with the methods represented unchanged. SHOWN as practice. The paper's own 50%/150% bands in Table 7 were not varied, which shows the gap.
- **What it changes.** It adds a reporting rule. `M17.E.3` gives invariance intervals for engine thresholds in policy and appraisal only. Nothing found covers thresholds in the analysis layer. This improves inference validity.
- **Cost / risk.** Low: reclassification of existing runs, with no re-runs. No conflict.

**CD-X1. A narrator preserves between-agent differences, checked in both directions against the deterministic renderer (X-line only).**
- **Proposed requirement.**
  - Before any Phase F narrator is used, the divergence between agents of narrated trait readouts MUST be compared, on the same frozen log, with the same divergence computed from `M16.C`'s deterministic renderer or from targets derived from engine state, and reported as a ratio.
  - A ratio well below 1 (flattening) and a ratio well above 1 (caricature) both fail.
  - Per-agent fidelity MUST NOT be reported as evidence that between-agent differences are preserved.
- **Where it would live.** Phase F narrator acceptance (`M16.E.3`).
- **Evidence.**
  - SHOWN (Table 1): DeepP-I gets the best WD in 7 of 8 blocks with CDP of 14–59%, and Cultural reaches 184–501%.
  - SHOWN (Table 10): 6 of the 8 lowest-JSD conditions have CDPc below 50%.
  - This bears directly on the owner's "15-year-old brat" vs "shy 10-year-old" question. The richest persona template flattened most.
  - Caution: the authors attribute part of that to attributes not conditioned on country (Limitations).
- **What it changes.** It adds a narrator gate. It extends X2, which tests flattening only, against each persona's own baseline SD. CD-X1 adds the caricature direction, uses the engine or renderer as the reference, and adds the fidelity disclaimer. It complements X8 (per-agent targets) and MM3.
- **Cost / risk.** Moderate: it needs an extractor that is not an LLM judging itself (X7). Consistent with `M3.D.6` and `M16.E`.

**Already covered (one line each).**
- Magnitude-only divergence cannot show who moved: `M17.D.2(a)`.
- Degenerate bootstrap intervals and a paired statistic on per-seed differences: `M11.4e`, `M17.A.3`.
- Too few seeds: `M17.A.1`.
- Stable identifiers across arms: `M1.A.20` (C2).
- Readout at a bound: `M11.4d`.

---

# Part 3: Kutzner et al. 2026 (Consequential Behaviour)

## Reader report: Kutzner et al. 2026 (Consequential Behaviour)

### Scope

I read the whole paper: all 990 lines of the `pdftotext -layout` output, covering the abstract, §1–§7, Fig. 1, Tables 1–2, the Declarations and the reference list. The paper reports no data and runs no experiments. Its Declarations say "No new data were created or analysed in this article." I also read `READER_TASK.md`, `EPMODEL_BRIEF.md` and `DESIGN_LESSONS_model_design_papers_2026-09-17.md` in full.

I checked overlap by keyword search. In the spec (rev10) I read in full: M5.F, M11.F, M11.G, M17 (all of it), the M11.4 exceptions table, M11.D.15, M2.A.0a, M2.A.1, M1.A.4a, M1.B.3, M9.5, M16.A.5a, M16.A.9, M10.B.4, M10.C.2a, Revision 8, and the M11.C criteria headlines. Elsewhere I searched `SPEC_CANDIDATES_from_preprints_2026-09-20.md` (C4, C28, C34, C39, X2), `SWEEP_READING_REPORTS_2026-09-27.md` (J1–J4, SV1–SV4, PD1–PD4, X5–X9), `SWEEP_READING_REPORTS_2026-09-27b.md` (BB1–BB5, MM1–MM4, AD1–AD3, TM1–TM6, X10), and TODO.md's "Spec revision 11 — held" section. Search terms included: worst, subgroup, stratum/strata, per-person, self-report, stated, intention, enacted, dispersion, flatten, collapse, homogen, placebo, childhood, founder. "Worst" and "subgroup" have no hits in any of these files that bear on this proposal.

---

### A. Report

**What the paper is.** It is a position and framework paper on validating LLM "synthetic survey respondents" at population scale. It contains no simulation, no agents interacting over time, and no results of its own [PAPER, Declarations]. Four of the seven authors are founders or employees of a company that sells synthetic-audience simulations, and the other three were funded by it [PAPER, Declarations]. Most of the brief's items 1–4 and 8 therefore have no content.

**5. Calibration and validation (the main content)**

- [PAPER, §1.2, Fig. 1] What people say and what they do are separate measurable quantities. Persona self-report can match human self-report and still fail to predict behaviour. Fig. 1 says about half of people who intend to act do so (citing Sheeran & Webb). The authors expect models to "tighten the coupling" between self-reports and action (§1.2), so that personas at best behave "as if people did what they said."
- [PAPER, §1.2] They give a figure of 6–18% for the share of studies in flagship journals that measure actual behaviour (citing Doliński 2018).
- [PAPER, Table 1, §2] The framework has one criterion and four diagnostic levels:
  - **C**: does the persona set predict observed behaviour, per stratum?
  - **L0 location**: do marginals match?
  - **L1 dispersion**: is the spread neither compressed nor inflated?
  - **L2 response process**: context, order and format effects.
  - **L3 structure**: covariance, factor structure and invariance.
  - **E** is cross-cutting: do manipulations move the synthetic population "as they move people"? Its pass criterion is that the direction and approximate size of the human effect are reproduced.
- [PAPER, §2] Validity belongs to an interpretation and a use, not to an instrument (citing Messick, Cronbach & Meehl). Each level licenses a different claim. The acceptable discrepancy is to be declared "before inspecting correspondence."
- [PAPER, §2.1] Baselines must be declared (demographic base rates and an unconditioned model, not chance). Calibration must hold within each stratum. Behavioural predictions use held-out outcomes that were not available during persona construction.
- [PAPER, §2.2 L1] Variance compression is "where models fail most consistently": the mean is right but the distribution is too narrow ("distribution collapse or flattening").
- [PAPER, §2.2 L3] A structure "cleaner than human data" is to be treated as a failure.
- [PAPER, §2.3] Within-persona counterfactuals (duplicate a persona and assign each copy to a different condition) raise four problems:
  1. Implementing a treatment through the prompt can change who the model thinks the respondent is ("user drift"), which confounds the contrast.
  2. Output is stochastic, so a same-condition null distribution is a precondition for reading any contrast.
  3. The contrast is a conditional effect for a profile, not an individual effect.
  4. Human benchmarks are themselves noisy.
- [PAPER, §3.1] Aggregate accuracy is a weighted average dominated by the large groups. A sample can be right on the mean and "wrong for every subgroup" when the errors offset. The rules that follow:
  - strata are declared ex ante from the decision context, not from the data;
  - intersections are reported where sample size allows;
  - worst-group performance is reported beside the average;
  - unpopulated cells are flagged and kept (Table 2 item 7).
- [PAPER, §3.1] Three mechanisms cause subgroup failure: the model learned the population statistic without the group structure; training-data coverage is thin for small groups; and helpful models "cannot easily be instructed to mimic incapacity" (citing Kumar et al. 2025). The last one makes personas over-predict uptake where people do not know about an option or cannot manage the process.
- [PAPER, §3.1] Rectification with a small human sample reduces a 24–86% bias to below 5% (citing Krsteski et al.). A subgroup with thirty human respondents cannot support a strong claim either way.
- [PAPER, §3.2] The three justice dimensions are turned into quantities:
  - **distributional**: how prediction error is spread across pre-specified subgroups;
  - **procedural**: documentation of model, version, prompt, temperature, draws per persona and benchmark;
  - **recognition**: preserved within-group variance, since removing dispersion counts as representational harm.
- [PAPER, §4] Strata should track behavioural constraints (for example, whether a household has a private charging point) rather than demographics.
- [PAPER, §7] The paper makes four testable predictions, none of which it tests:
  1. Structure (L3) will be at most weakly related to behavioural prediction (C).
  2. Where verbal fidelity carries behavioural information, it will be through L2 rather than L3.
  3. Aggregate fidelity will overstate worst-subgroup fidelity.
  4. Direction will be reproduced more reliably than size.
- [INFERENCE] For a model with no data and a no-fitting rule (M11.F.9), C and L0–L3 cannot be reached by EPModel at all. The only level with a counterpart is E, read against corpus-stated directions instead of human effects. This is what M0.4 already does, with direction only. Prediction 4 (untested) points the same way as M0.4's choice of direction over magnitude. Table 2 item 10, an explicit statement when no behavioural criterion was tested, is already met by M11.F.9(a) and M11.G.3. The paper's "use claimed" (exploratory / substitutive / predictive) corresponds to M17.G.1's claim grade (exploratory / mechanism / intervention).
- [INFERENCE] On "measure what the theory says people DO": at the level of a single agent, the spec already builds in an intention–behaviour gap.
  - The paper's own reference band is empirical. EPModel's counterpart is a corpus-stated relation: M5.F.2a says asserting a differentiated state is negative evidence for it.
  - M5.F.4's assertion-form I-POSITION and M4.D.1b's withheld move are stated-versus-enacted distinctions.
  - `basic_level` is an estimator over enacted functioning (M1.A.4a, Revision 8), not a stored self-description.
  - The remaining gap is at the readout level (CB1).
- [INFERENCE] On representational fairness: in EPModel, some positions being worse off is theory content (the projection focus, M11.C.2; the outside member, M11.C.3), not a defect. The transferable question is narrower: is any position systematically moved by the mechanism for a reason the theory does not name, and do family-level aggregates hide position-level reversals? The M6 conservation invariants make offsetting by construction likely, so this is a real exposure (CB2, CB3).

**6. Limitations the paper states** [PAPER, §6]
- Training-data contamination cannot be separated out.
- Subgroup analysis is bounded by the human data available.
- Behavioural criteria have their own problems: public posting is performed behaviour, and administrative records are shaped by the systems that produce them.
- Model versions change, so validity evidence has "a short shelf life."
- Personas may need to be specified "as persons in situations rather than as stable characters." Interacting agents' coordination patterns are left to later work.

**7. Software engineering / reporting** [PAPER, Table 2] There is a 12-item reporting checklist. Item 11 is the same-condition null alongside any contrast. Item 12 asks who ran the validation, their financial ties to the system, and "what was locked before results were seen." [INFERENCE] EPModel already has: M10.B.4 (constants frozen before the acceptance suite), M17.A.4 (pre-declared margin), M11.D.15 (placebo arm), M11.4a (equivalence bounds, the same idea as the paper's "equivalence band"), and the ledger and version-lineage proposals TM6 and SV4.

**8. Small-N, family, emotion, long horizon.** None. Households appear only as subgroup labels in the EV-tariff example (§4).

**Not transferable / cautions**
- Everything stated against human benchmarks — calibration, AUC, base rates, rectification, human test-retest bands — needs data EPModel does not have and must not be fitted to (M11.F.9(c)). Do not import any of it as a validation target.
- The justice framing must not be carried over literally. A position that the theory says carries the cost must not be "corrected" for fairness. That would put an outcome directive into the mechanism, which M11.1d's outcome-directive audit exists to catch.
- The paper has a vendor conflict of interest. Its predictions are untested, and the claims it cites (Krsteski, Kumar, Barrie & Cerina) are secondhand here. Cite them as ARGUED at most.

---

### B. Candidate additions

**CB1. Every M11.G component and every M11.C criterion names the layer it reads (belief, latent state, or enacted events), and a result on one layer is not reported as a result on another.**
- **Proposed requirement.** Each M11.G.1 component and each M11.C criterion **MUST** declare which layer it is computed from:
  - **held** — belief, M9;
  - **latent** — true state: anxiety, bond energy, `functional_level`;
  - **enacted** — the M4.E.1 event log: moves emitted, contact counts, WITHHOLD counts.

  A finding on one layer **MUST NOT** be reported as a finding on another. Components 1, 5 and 8 **SHOULD** emit their enacted-layer counterpart beside the latent value. The test uses the M2.A.1 fixture (Iris and Bruno: comparable contact frequency, different bond energy). Component 5 on the enacted layer **MUST** fail to separate them within M11.4a's equivalence bound, and on the latent layer **MUST** separate them.
- **Where.** M11.G, M11.C preamble, M16.A.
- **Evidence.** Fig. 1, §1.2 and Table 2 items 1 and 10 say self-report and behaviour are different targets, and a claim on one "is not extended to behaviour" (ARGUED). Transfer to three layers is INFERENCE. The theory supports it: M1.B.3's four tie states "differ in events and energy independently," and M11.F.6's traps ("reported closeness", "geographic distance") are cross-layer readings.
- **What it changes.** Adds a reporting rule and a test. Component 5 is currently computed from bond energy alone, a quantity no real evaluator can see, and nothing stops a report presenting it as observed cutoff. Improves inference validity and fidelity to M1.B.3.
- **Overlap.** PARTLY covered:
  - M9.5 and M11.G.4 (belief vs truth: two layers, no enacted layer);
  - M17.B.3 (belief–truth discrepancy);
  - AD1 (withheld moves);
  - TM3 (composites by component).

  New here: the third layer, the no-cross-layer-claim rule, and the fixture test.
- **Cost / risk.** Low. Observer-side, so M16.B holds. No new constants beyond the equivalence bound, which M11.4a already requires.

**CB2. Directional results asserted on a family aggregate are also reported per declared position stratum, with the weakest stratum and any sign reversal shown.**
- **Proposed requirement.** For every M11.C criterion asserted on a family-level aggregate (for example M11.C.1, M11.C.4, M11.C.17, M11.C.41), Phase E **MUST** report the per-seed paired arm difference for each position stratum declared before the acceptance suite runs (frozen under M10.B.4). Strata are defined by structural position, not by identity labels:
  - generation;
  - spouse or child;
  - projection focus or sibling;
  - inside pair or outside member of the active triangle;
  - over- or under-functioning pole;
  - financially dependent (M11.C.33);
  - married-in;
  - external agent;
  - founder or born in the run.

  The report **MUST** show the stratum with the weakest direction beside the aggregate, **MUST** flag any stratum whose sign reverses, and **MUST** list strata with too few occurrences as unpopulated rather than drop them. It **MUST** use M11.4e's multiplicity correction. Where the theory predicts opposite signs by position, an aggregate-only criterion is a defect.
  - Test: `test_cb2_reversed_stratum_is_flagged_when_aggregate_holds`, on a synthetic fixture.
- **Where.** Phase E (M17.B), using M16.A. It would also define the "person stratum" that M17.B.5 uses without defining it.
- **Evidence.** §3.1: a sample can match a population mean "while still being wrong for every subgroup" when errors offset. The same section asks for strata declared ex ante, worst group beside average, and intersections. Table 2 item 7: unpopulated cells retained. §4: strata defined by behavioural constraint. All ARGUED. Transfer is INFERENCE, and it is stronger here than in the paper's setting: M6 conserves and redirects anxiety, so family totals can stay flat while positions move in opposite directions by construction.
- **What it changes.** Adds a reporting rule. Improves inference validity. "Worst" here means the weakest arm difference, not the worst outcome. A projection focus being worse off is theory, not unfairness.
- **Overlap.** NEW as far as my search shows. Related items:
  - MM2 (stratifies belief error by similarity);
  - TM4 (moderator levels, not positions);
  - M17.G.1 (audits family composition as a dimension, not positions within it);
  - M11.C.25 (one invariance, sex).
- **Cost / risk.** Low compute. With about 12 persons, most strata hold 1–3 people, so results are close to per-person and need the ensemble's power. Report per-stratum results as descriptive except for the reversal flag. No rule conflict.

**CB3. A comparison across generations must not compare supplied founder state with simulated state without showing that the result does not depend on the founders' declared spread.**
- **Proposed requirement.** Any criterion or readout that compares a quantity between generations whose state was supplied at t0 (M2.A.0a: all twelve founders) and generations produced by the dynamics **MUST** be repeated over a declared sweep of D0's founder spread (M17.C.1). If the direction depends on that spread, it **MUST** be reported as conditional. This applies to M11.C.6(b)'s widening spread of `basic_level` across generations and to M11.G component 4's level by generation. Where the horizon allows, the comparison **SHOULD** also be made between two simulated generations only.
- **Where.** M11.C.6, M11.G.1, Phase E (M17.C.1, M17.E).
- **Evidence.** §2.2 L1 and §3.2 make dispersion a validity dimension separate from location, and treat lost within-group variance as a failure in its own right (ARGUED). The specific confound is INFERENCE from M2.A.0a, which says founder childhoods "are never simulated" and their values are supplied. M11.C.6(b) asserts that spread widens, so the result partly depends on how narrow a spread was declared for the founders.
- **What it changes.** Corrects a hidden dependence in an existing criterion. Improves inference validity.
- **Overlap.** PARTLY covered:
  - TM1 (logs realised initial spread and uses it as a covariate);
  - M17.F.2 (settling condition);
  - DESIGN_LESSONS §2.5 (initial-state compatibility).

  None of these names the supplied-versus-simulated comparison inside M11.C.6(b).
- **Cost / risk.** Low to moderate (one swept D0 parameter). The sweep range is `[I]` and must be frozen under M10.B.4. No conflict with M11.F.9, since nothing is fitted.

**Already covered (one line each)**
- Same-condition null before any contrast (Table 2 item 11; §2.3): M11.D.15 placebo arm, C4, M17.B.1.
- User drift, where the treatment changes the persona (§2.3): M17.D.3 arm-blindness, TM2's manipulation check, and the owner's SV2 answer (arms differ only by the knowledge event).
- Acceptable discrepancy declared in advance (§2): M17.A.4, M11.4a.
- Claim level and intended use declared (Table 2 items 1–2): M17.G.1's claim grade.
- Explicit downgrade when no behavioural criterion exists (Table 2 item 10): M11.F.9(a), M11.G.3.
- What was locked before results were seen (Table 2 item 12): M10.B.4, SV4.
- Invariance to identity attributes the theory treats as irrelevant: M11.C.25 (sex), MM1's label-null arm (M1.A.14b), C5 permutation (M1.F.8), M1.A.20 (stable identifiers).
- Persons in situations rather than stable characters (§6): already the design (load-dependent `functional_level`, M1.A.4g).

**Narrator / LLM line**
- No CB-X candidates. The relevant point is §3.1's citation of Kumar et al.: helpful models cannot be instructed to act incapable, so an LLM persona of a low-maturity person will over-represent competent responses. This is already covered by X10 (a behavioural check because questionnaire validity can be passed by reading the prompt), BB1, X7, X8, and DESIGN_LESSONS §3.2 and §8.8 (pull toward agreeableness, benevolence bias). It is further support for M3.D.6.

---

# Part 4: Guo et al. 2026 (Bayesian Personalized Value Alignment)

## Reader report: Guo et al. 2026, "Bayesian Personalized Value Alignment" (BaCVA), arXiv 2609.28942v1

### Scope

I read the whole paper from the pdftotext file, lines 1–1,941: the body (§1–5), Limitations, Ethics, references, Appendix A (ELBO derivation and prompts, Figs. 8–13), Appendix B (datasets, metrics, baselines, implementation, human evaluation) and Appendix C (C.1–C.11, Tables 6–20). Most figures came through only as captions and broken numbers. I do not report numbers from Figs. 4–7 except where the prose states them.

Before writing, I read EPMODEL_BRIEF.md and DESIGN_LESSONS_model_design_papers_2026-09-17.md in full.

**What I read of the spec (rev 10).** M1.A in full: M1.A.0–M1.A.20, especially M1.A.3/3a/3b/3c, M1.A.4a/4b/4c/4h, M1.A.5/5a/5b/5c, M1.A.7/7a and M1.A.14a–d. Also M1.F.1–M1.F.9, M4 in full (M4.A–M4.G), M10.A–M10.C.1, the list of M11.C/M11.D criteria (M11.C.1–C.41, M11.1b, M11.4e), M14, and M17 in full.

**Overlap searches.** I keyword-searched the spec, SPEC_CANDIDATES_from_preprints_2026-09-20.md, SWEEP_READING_REPORTS_2026-09-27.md, SWEEP_READING_REPORTS_2026-09-27b.md, and the "Spec revision 11 — held" section of TODO.md. Terms: Lewin, context, situation, domain, salience, profile, within-person, mixing weight, joint-decision, task demand, sever.

All statements that the spec "has no X" rest on those keyword searches, not on a line-by-line read of all 427 requirements.

---

### A. Report

#### What the paper is
[PAPER] It is an inference-time method for one LLM. For a user and a question, it predicts a 10-dimensional Schwartz value vector, each dimension scored 1–5 (§3.1).

The model is built from:
- **Two priors:**
  - a person prior: a static value profile scored by an LLM from the user's self-description (Fig. 8);
  - a scenario prior: value salience extracted from a "universal" answer to the question, generated by an aligned LLM, GPT-5-nano (§3.3, Eq. 3).
- **Two views:**
  - View 1, scenario-anchored: Softmax(s_x + z_u), a personal residual added to the scenario prior (Eq. 6–7);
  - View 2, person-anchored: Softmax(s_u + z_x), a context residual added to the person prior (Eq. 8–9).
- **A gate α** that mixes the two views (Eq. 5, 10). The gate is itself an LLM prompt; α is read from the token-0/1 logits (Fig. 13, App. A.3.2).
- **Training:** variational inference. The loss is MSE to the values of the preferred answer, plus λ1 times the KL to each prior (Eq. 11).

**The motivation is explicitly Lewin's field theory.** [PAPER] §1 and §3.2 say behaviour is jointly shaped by personal dispositions and situational constraints, and also cite cognitive appraisal theory. In Fig. 1(a) the same user ranks Self-Direction above Conformity, yet answers the other way when the scenario imposes strict protocols.

#### 2. Agent architecture: stable disposition versus context-dependent expression
- [PAPER] **A fixed profile does worst.** The static-profile baseline (PersonValue) scores accuracy 56.56% on PRISM, below the non-personalised DirectAnswer at 67.05%. BaCVA scores 81.72% (Table 1). MAE is 2.969 for PersonValue, 1.858 for DirectAnswer and 0.728 for BaCVA. The authors conclude that static profiles cannot represent "users' context-dependent preferences" (§4.2).
- [PAPER] **Ablations on PRISM (Table 2), MAE / accuracy:**

  | Configuration | MAE | Accuracy |
  |---|---|---|
  | Full model | 0.728 | 81.72 |
  | Without scenario view | 0.819 | 76.32 |
  | Without person view | 0.798 | 77.06 |
  | Without gated fusion (fixed average) | 0.762 | 79.44 |
  | Without prior constraint | 0.800 | 78.18 |

  GOOD shows the same ordering.
- [PAPER] **Error by profile–answer conflict (Table 18, App. C.9).** Test samples were split into thirds by how far the chosen answer's values sit from the static profile (n = 136 / 149 / 135):
  - ValuesRAG MAE rises 0.801 → 0.899 → 1.163;
  - BaCVA MAE rises 0.640 → 0.678 → 0.956.

  Both get worse as context pulls away from the profile. The gap is largest in the high-conflict third, where BaCVA has 17.8% lower MAE.
- [PAPER] **The gate.** Fig. 6 and §4.4 say the gate gives more weight to whichever view dominates in a subset (personal-dominant vs scenario-dominant). The figure's numbers did not extract cleanly.
- [PAPER] **The KL weight is non-monotone (Table 6, PRISM MAE):** 0.7848 at λ1 = 0, 0.7344 at λ1 = 0.1, 0.9405 at λ1 = 1. Too weak a tie to the prior and too strong a tie both hurt (App. C.1).
- [INFERENCE] **Mapping to EPModel.** The paper's split is the one EPModel already makes, in more detail:
  - **Stable disposition:** `basic_level` (M1.A.2, M1.A.4a) and `programmed_reactivity`. M1.A.7a states the split outright: the M1.A.7 value is "a *disposition*", the M1.A.7a value "a *state*".
  - **Context-dependent state:** `functional_level` = basic + swing (M1.A.5a); `chronic_anxiety` derived from field and position (M1.A.7a); `acute_anxiety`; and the basic versus functional relationship balance on each tie (M1.A.5b).
  - **Context-dependent expression of a stable trait, in three places:**
    - the intellect's licence is weaker over the joint-decision domain at every level (M1.A.3);
    - marriage reclassifies decisions into that domain "and changes neither person" (M1.A.3a);
    - sibling-position effects are suppressed in the relational domain but stay available on a task demand (M1.A.14d).

  **EPModel does not hold a fixed per-person profile.** The paper's central criticism therefore does not apply as a correction. What it does expose is in B below: the spec names contexts it never represents.
- [INFERENCE] **The two views mirror EPModel's two channels.** View 1 is anchored in what the surrounding norm demands; View 2 is anchored in the person. That is the same shape as M4.D.1a's automatic channel ("driven by the relationship system") and self-directed channel ("driven by the person"), with α in the place of the mixing weight. In the paper α depends on context (x and both views' uncertainties), and replacing it with a fixed average costs MAE 0.034 and 2.3 accuracy points (Table 2). This is a structural parallel only. Which way context should move EPModel's mixing weight is a theory question this paper cannot answer (BV3).

#### 4. Initialisation
- [PAPER] Personal profiles come from users' self-descriptions scored by an LLM. The Limitations section warns that users may "curate, simplify, or embellish their profiles".
- [INFERENCE] This matches EPModel's existing caution on imported data: M15.B uses ranges, and M1.A.4e keeps intake fields as a weak prior only. Nothing new.

#### 5. Calibration and validation
- [PAPER] **Training and data.** One training seed (42) for the main runs and for the E2E split (App. B.3, C.5; Table 12). Data is split 80/20: PRISM 2,131 instances, GOOD 2,696. No seed-to-seed variance is reported anywhere. Table 1 marks p < 0.05 without naming the test.
- [PAPER] **Human evaluation.** 50 samples, two raters. When the raters disagree the sample counts as a tie: 46% win, 32% tie, 22% loss (Fig. 3, App. B.4).
- [PAPER] **Extractor validation.** GPT-5-nano both builds the training targets and supplies the scenario prior. Against human labels it scores MAE 0.056 at profile level and 0.120 at answer level (Table 4).
  - Rebuilding the test targets with other extractors keeps BaCVA ahead of ValuesRAG: MAE 0.826 vs 1.052 with Gemini-3-Flash, 0.842 vs 1.066 with Qwen3-235B (Table 15).
  - The three scenario-prior estimators agree within one point 89.15% ± 1.28% of the time. Downstream MAE is 0.756 ± 0.027 across them (Table 16).
- [PAPER] **Prompt sensitivity.** Different prompt designs for the universal answer agree with the default only 0.738–0.806 of the time, yet final MAE moves only 0.715–0.743 (Table 8).
- [INFERENCE] **"Universal" means an aligned LLM's answer.** The scenario prior is defined as what an RLHF-aligned model says is normative (§3.3; the Limitations call it "an aligned-LLM approximation"). The gate prompt also asks the LLM whether personal reasoning risks "overfitting to idiosyncratic user traits" (Fig. 13), which pushes toward the norm. This is the agreeableness and consensus pull in DESIGN_LESSONS §3.2, built into the method.

#### 6. Failure modes reported
- [PAPER] **Qualitative cases.** In Fig. 17, prompt-steered baselines list every profile dimension mechanically ("Mechanically lists all v_U dimensions") instead of adapting to the situation.
- [INFERENCE] This is the "trait recited rather than enacted" failure from DESIGN_LESSONS §3.2.

#### 7. Software engineering
- [PAPER] **Amortised versus sampled posterior.** A single amortised deterministic pass scores MAE 0.728. A 10-sample Monte Carlo posterior scores 0.786, at similar latency (158.8 ms vs 148.0 ms per sample) (Table 20).
- Nothing transferable beyond that.

#### 1, 3, 8. Formalism, interaction, small-N and family
- [PAPER] None. Single-turn questions, one user at a time, no time dimension, no interaction between agents.
- Family content is limited to two PRISM example questions about relatives (Fig. 17 Case B; Table 5).

#### Not transferable / cautions
- **No Bowen content.** Schwartz values, Moral Foundations and the Daily-Dilemma value system are non-Bowen taxonomies. None of them may enter `docs/theory/` or be cited as theory.
- **Lewin's field theory is cited here as motivation, not tested.** An earlier reader judged that the Lewin P/E split fails for EPModel because a person's environment is the other agents (SWEEP_READING_REPORTS_2026-09-27.md, line 310). I agree for whole runs: context is endogenous there. It still works inside a constructed test that holds the context fixed (BV2).
- **Every number is LLM-scored against LLM-scored targets** on preference data. None of it is a magnitude or a calibration target for EPModel.
- **The gate is an LLM call.** Any analogue in EPModel would have to be rule-based (M3.D.6).
- **Single seed, no reported variance, 50-item human study.** Do not cite these numbers as precedent.

---

### B. Candidate additions

**Already covered, one line each:**
- Fixed profile versus disposition plus state: M1.A.5/5a, M1.A.7/7a, M1.A.5b.
- The disposition crossed with stress load as two factors: M11.C.41.
- The same emitted act read differently depending on what drove it: M1.F.1a's channel field.
- A policy insensitive to its context inputs: same as M11.1b (C17, matched-magnitude severing mutants).
- Non-monotone response to how strongly behaviour is tied to the disposition (Table 6): M17.E.2(b) and M11.C.38 already require monotonicity to be reported and checked.
- Narrator gets the full state for each call, not a person-level profile: same as X5 (27 report); the narration contract in DESIGN_LESSONS §7.6.
- Profile-only and context-only baselines as arms (the paper's single-view ablations): the C34 / M17.D.1 control-arm family. EPModel's arms remove mechanisms rather than inputs, and M11.1b covers inputs.

#### BV1. Give "situation", "decision domain" and "demand type" a representation the policy and estimator can read

- **Proposed requirement.** Every context term that a requirement makes behaviour depend on **MUST** be a declared, typed field on the object the policy or estimator reads, with its values listed in config. Three such terms exist:
  - the decision domain in M1.A.3/M1.A.3a: joint-decision / shared life course versus other;
  - the demand type in M1.A.14d: relational versus task;
  - the "situation" that M1.A.4a/4b's estimator counts toward its breadth (M1.A.4c, M10.B.3).

  The M14.A register **MUST** list each one: who writes it, and which mechanisms read it. A requirement whose context term has no representation **MUST** be reported as not implementable rather than approximated.
- **Where it would live.** M1.F, as an Event field, or a new M1 sub-part; the M14.A register; M10.B for the value lists.
- **Evidence.**
  - [PAPER] The paper's main result is that context must be an explicit input. The static-profile baseline drops to 56.56% (Table 1), and error grows with profile–context conflict (Table 18). Strength: SHOWN for its own task.
  - [INFERENCE] The EPModel gap itself. M1.F.1 lists the Event fields and none is a domain or demand. A keyword search found no definition of "situation", "decision domain" or "demand type" anywhere in the spec. Yet M10.B.3 asks for sensitivity to "choosing 30 situations rather than 12", and M1.A.3a says marriage "reclassifies a large class of decisions". Nothing in the spec says what a decision is in a model whose units are moves on ties.
- **What it changes.** It fixes a gap the current design has: three stated requirements and one estimator have no state to read. It improves theory fidelity, because the requirements become implementable as written, and inference validity, because the breadth count becomes a defined quantity whose sensitivity can be measured.
- **Cost / risk.**
  - Low code cost.
  - Needs a **theory decision by the owner**:
    - What is a "situation" for the estimator: a tie, a tie crossed with nodal events, or a domain?
    - Are M1.A.3's decision domain and M1.A.14d's demand type one axis or two?
    - How does a move on a tie carry a decision domain at all?

    The paper supplies none of these answers and must not be used to fill them.
  - No conflict with M3.D.4–6, M11.F.9 or M16.B. The value lists are editorial content, so they belong in config (M10.B.1).

#### BV2. Within-person context-contrast criteria for M1.A.3 and M1.A.14d, with a static-profile mutant

- **Proposed requirement.** Two new M11.C criteria **SHOULD** each hold one person fixed (same seed, same `basic_level`, same `programmed_reactivity`, same ties) and differ only in the context field from BV1:
  - **(a)** At equal level and load, the self-directed channel's share over joint-decision demands **MUST** be lower than its share over other demands (M1.A.3).
  - **(b)** Under raised acute anxiety, sibling-position-typical behaviour **MUST** appear more often on a task demand than on a relational demand (M1.A.14d).

  Each criterion's mutation **MUST** replace the context-dependent term with that person's average across contexts, which is a fixed profile. The test **MUST** go red under it.
- **Where it would live.** M11.C; the mutation goes in the M11.1 mutation suite.
- **Evidence.**
  - [PAPER] Fig. 1(a) uses the same design (one profile, two scenarios, opposite expression), and the PersonValue baseline is the static-profile mutant (Table 1). Strength: SHOWN in the paper's domain; transferring it is INFERENCE.
  - [INFERENCE] My spec search found no M11 criterion and no `test_` name tied to M1.A.3 or M1.A.14d. M14.1 will eventually require tests for them, but nothing states their form. Every existing person-versus-condition criterion compares families or arms with different persons or loads (M11.C.1, C.9, C.41). None holds the person fixed and varies the context.
- **What it changes.** Adds tests. A policy that collapses to a per-person profile would fail it, so it improves inference validity. It is directional (two arms, M0.4) and needs no magnitude.
- **Cost / risk.**
  - Depends on BV1.
  - M1.A.14d's source is graded `[D]`, n≈6, no control. If the owner treats that requirement as weak, (b) should be a SHOULD with that grade attached.
  - The mutant must keep the person's mean across contexts so it is not the same as M11.1b's shuffle. If it turns out to equal M11.1b in practice, drop it and keep only the criteria.

#### BV3. State what M4.D.1a's mixing weight reads (owner question, then one requirement)

- **Proposed requirement.** M4.D.1a **MUST** state which variables the mixing weight between the automatic and self-directed channels reads: `basic_level` only, agency (M10.A.1a), `functional_level`, or any of these together with acute anxiety. It **MUST** also state whether M1.A.4h's "mixing weight read backwards" means that coefficient or the share of channels actually selected. If the coefficient reads only slow variables, the spec **MUST** say that all context dependence of the channel split passes through M4.D.3's anxiety weighting of the reactive moves.
- **Where it would live.** M4.D.1a, M1.A.4h, M10.A.1.
- **Evidence.**
  - [PAPER] A context-dependent gate beats a fixed average (Table 2), and the gate shifts toward the dominant view in each subset (§4.4, Fig. 6). Strength: SHOWN for the paper's task; relevance to EPModel is INFERENCE.
  - [INFERENCE] The spec text does not settle the question:
    - M4.D.1a says only "a function of differentiation";
    - M10.A.1a derives agency from `basic_level`;
    - M1.A.4h reads the weight "at comparable load", which suggests the selected share depends on load;
    - M4.D.3 raises reactive-move weight with anxiety, without saying whether that happens inside the automatic channel or through the mix.

    The two readings give different estimator behaviour. If the weight reads `functional_level`, the estimator reads the swing, not the basic level, unless load is matched.
- **What it changes.** It clarifies the spec (the policy is currently ambiguous) and protects M1.A.4h's estimator from reading state as disposition. It improves both theory fidelity and inference validity.
- **Cost / risk.**
  - No code yet, so the cost is a sentence.
  - It is a **theory decision**. Whether anxiety shifts the channel split itself, and in which direction, needs corpus support from `_LEDGER.md`, not this paper. Nothing here supplies a direction or a magnitude.
  - A gate in the paper's style read from an LLM is excluded by M3.D.6.

#### Narrator line (BV-X)
None. A narrator should get context (tie, triangle position, current anxiety) for each call, not just a profile. That is already X5 and DESIGN_LESSONS §7.6. Two of the paper's features argue against using an aligned LLM to supply context salience at all, which is consistent with DESIGN_LESSONS §3.2:
- the "universal" prior is an aligned model's normative answer (§3.3);
- the gate prompt pushes toward the norm (Fig. 13).

---

# Part 5: Qraitem, Saenko & Plummer 2026 (PERSONAWEAVER)

## Reader report: Qraitem, Saenko & Plummer 2026 (PERSONAWEAVER)

### Scope

I read the full `pdftotext` output of Qraitem, Saenko & Plummer, *PERSONAWEAVER* (arXiv 2609.26629v1). That covers the body (§1–6), Tables 1–4, the figure captions for Figs 1–9, and Appendices A–F. The text is not truncated: it runs through Appendix F and the Fig. 9 caption. Two-column layout scrambles the order on pages 1–2, but no text is missing. The figures did not survive extraction, so none of the Fig. 3 distribution values are in the text. Only the medians in App. F and the Fig. 7 adherence cells survive. I also read `READER_TASK.md`, `EPMODEL_BRIEF.md` and all of `DESIGN_LESSONS_model_design_papers_2026-09-17.md`.

In the spec (rev 10) I read in full: `M2`/`M2.A` (reference family, `M2.A.0c–h`), `M15` (all of it), `M16.E` and `M17` (all of it), and `M3.D.4a`. I keyword-checked `M10.C.4`, `M11.1c`, `M11.C.25`, `M11.D.17`, `M11.D.22` and `M14.A`. For overlap I searched `SPEC_CANDIDATES_from_preprints_2026-09-20.md` (C10, C21, C32, C34, C42, X1–X4), both `SWEEP_READING_REPORTS_2026-09-27*.md` (J2, PD4, MM4, TM1, BB1, BB2, X5–X10) and TODO.md's "Spec revision 11 — held" section. Search terms: coverage, archetyp, stereotyp, homogen, D0, initial cond, reject, feasib, turning point, Latin, topolog, hand-built, narrat, adherence, surface, demograph.

### A. Report

**What the paper is.** [PAPER] An LLM procedural character generator. It builds "world" attributes in one step and assigns behaviour in a separate step, by sampling from two fixed hand-curated banks: 8 moral positions and 8 reactions to questions (§3, Table 1). It is evaluated on 10 settings × 100 characters × 3 narration models, against WorldWeaver, WorldWeaver + Diverse and PersonaHub (§4.1). No agent interaction, no dynamics, no time. Only items 2, 4, 5, 6 and 8 of the brief have content.

**2. Agent architecture.** [PAPER] A character is a static card: 10 world attributes, one moral position and one reaction type (Eq. 1). Behaviour is enforced only by prompt guidance (§3). No memory, learning or interaction.

**4. Initialisation (the paper's whole subject).**
- [PAPER] **Factorisation.** The world module is told to leave out personality, values, moral positions and interaction styles (§3, "World-Building Module"). Behaviour comes only from the banks. The stated reason is that asking the LLM for complete characters "tends to recover archtypes of each setting" (§3).
- [PAPER] **Sample-and-Mix (Eq. 1).** Each of the 10 world axes is drawn uniformly from 30 LLM-generated options. The moral and reaction entries are drawn uniformly from their banks. All draws are independent and with replacement. The authors say this "maximizes entropy over each bank" and covers the Cartesian product (§3).
- [INFERENCE] The coverage claim holds only for the marginals. 100 cards per setting cannot cover 30¹⁰ × 64 combinations. Joint coverage in a high-dimensional initial space is not reachable by sampling, so coverage has to be stated on named low-dimensional projections.
- [PAPER] **Consistency repair.** After mixing, an LLM at temperature 0 changes fields that are contradictory or impossible in the setting (e.g. a child working as an engineer). It is "not allowed to modify an unlikely group of attributes" (§3). No count of repairs is reported.
- [PAPER] **Limitation 3 (§6).** Equal-probability sampling was chosen to study coverage. The authors say the distribution a real use wants "likely would" differ by setting.

**5. Validation and diagnostics.**
- [PAPER] **Behavioural coverage is measured on realised responses, not on card text.** Every character answers the same 10 Social Chemistry norms on a 4-point scale and the same 10 ConvAI2 questions (Table 3). Replies are classified into refusal, deflection or compliance by Qwen 3.6 27B at temperature 0 (§4.2, App. A). Each model–method cell has 10,000 responses (Fig. 3 caption).
- [PAPER] **Main finding.** The baselines concentrate on agreement and compliance even though their world descriptions vary: "varied world descriptions therefore do not translate" into varied behaviour (§4.2).
- [PAPER] **A generic diversity instruction is not enough.** WorldWeaver + Diverse makes behavioural language more explicit but stays centred on prosocial concepts (Fig. 4, §4.2). Characters were generated in batches of 10 with earlier profiles in context and an instruction to differ from them (§4.1).
- [PAPER] **Archetypality score.** An LLM judge rates each card from 1 (unconventional) to 5 (archetypal), methods anonymised and shuffled. PersonaWeaver's median is 2; baseline medians are 3 or 4 (Fig. 6; App. F).
- [PAPER] **Plausibility.** One human annotator rated 50 GPT-4o cards per method: 4.98, 5.00, 4.98, and 4.82 for PersonaWeaver (§4.3).
- [PAPER] **Adherence check.** An LLM judge scored whether each reply follows its assigned reaction (Fig. 7, App. C). GPT-4o is at least 87.2% in every category. GPT-5.6 Luna follows deflection only 30.6% and meta-commentary only 13.3% of the time.
- [PAPER] The validation does not include seeds, repeated runs, confidence intervals, significance tests or human validation of the judges.

**6. Failure modes and limitations.**
- [PAPER] Homogenisation is attributed to maximum-likelihood training and assistant alignment (§1, §2). This matches design lessons §3.2.
- [PAPER] Narrator models differ: Luna silently drops two of the eight assigned categories (Fig. 7).
- [PAPER] Only two behavioural dimensions were tested; emotional regulation is named as missing (§6).
- [PAPER] One inconsistency: the text says Qwen adherence is "91.3% for every Qwen category" (App. C), but the Fig. 7 cell for Qwen compliance reads 91.2.
- [PAPER] Fig. 9 drops cards with fewer than two explicit world attributes and gives no count of how many.
- [INFERENCE] The judge (Qwen 3.6 27B) is from the same model family as one of the narrators (Qwen 3.5).

**8. Small-N, family, emotion, long horizon.** [PAPER] Nothing. "Family background" appears only as a candidate world axis (§3). The only emotion content is the sentiment-classifier result (§4.3, App. D).

**Not transferable / cautions**
- The LLM-judge measures (conventionality, adherence, reaction class) are not instruments for EPModel (design lessons §3.4).
- Independent uniform sampling (Eq. 1) is the wrong default for EPModel. `M17.C.1` already requires `D0` to state correlations: spouses matched on `basic_level`, sibling-order structure. The paper's approach is what C32 warns against.
- The split between demographic and behaviour-driving attributes does not map cleanly. In EPModel, sex is not a surface label: it enters sibling-position complementarity (`M2.A.0f`). Only display names are surface-only, and `M15.A.1` and `M2.A.0h` already treat them as opaque.
- "More diverse" is not a goal for EPModel. The theory restricts which families are admissible: the species distribution (`M10.C.4`, `M15.B.4`) and spouse matching. The useful lesson is narrower: varying the descriptors does not show that the behaviour varies.

### B. Candidate additions

**PW1. Coverage of initial-condition draws on the behaviour-relevant dimensions, starting with which side of each turning point a draw falls**
- **Proposed requirement.** For every `D0` (`M17.C.1`), the Phase E runner **MUST** report how its draws are spread over each turning point named in `M15.D.4` (`M11.C.14` marital distance; `M11.C.27` stable/unstable twosome; `M5.F.2` the `outside_ness` threshold; `M7.D.2d` severity; `M5.C.1a` peace-agree vs reactive):
  - the fraction of draws on each side and within a declared band of the turn, evaluated at t0 and again at the end of the settling window (`M17.F.2`);
  - the measure under which these fractions are computed.

  A `D0` whose draws lie mostly on one side of a turning point **MUST** have every directional result reported as conditional on that side. Wide spread in attribute values **MUST NOT** be reported as coverage.
- **Where.** Phase E, `M17.C`, with log fields in `M16.A`.
- **Evidence.** SHOWN in the paper: varied card descriptions produce concentrated behaviour (Fig. 3, §4.2). The paper therefore measures diversity on realised responses, not on descriptors. The transfer is INFERENCE: `M15.D.4` says error in initial conditions reverses the sign across a turning point, so the coverage that matters is coverage of turning-point sides, not of attribute ranges.
- **What it changes.** Adds a reporting rule. Improves inference validity: it can show that an ensemble covered only one regime.
- **Overlap.** PARTLY covered:
  - TM1 logs realised moments (mean and SD), not regime occupancy.
  - MM4 asserts structural properties of `D0`, not coverage.
  - `M15.B.3`'s coverage figure is about the ordinal map.
  - PD4 requires a declared measure for constants only; the measure clause here is its counterpart for `D0`.
  - `M17.B.2` classifies outcomes per seed, not initial occupancy.
- **Cost / risk.** Low, observer-side. Some turning points (e.g. `outside_ness`) are defined only on dynamic state, so the t0 evaluation may be undefined for them and must say so. No conflict with `M3.D.4a`, `M11.F.9` or `M16.B`.

**PW2. Family structure as a separate factor drawn from a declared bank**
- **Proposed requirement.** Phase E `D0` **SHOULD** treat family composition and kinship topology as a factor drawn independently of person attributes, from a declared bank of structural configurations. The bank covers sibship sizes, which lineage and generation holds a cut-off, which spouse's family of origin is distant, and launched vs at-home children. It **MUST NOT** use the single `M2` reference family with attribute ranges laid over it. Each bank entry **MUST** be either an `M15.E` anonymised topology or listed as `[I]`. Results **MUST** be reported per configuration as well as pooled.
- **Where.** Phase E, `M17.C.1`, using `M15.E`.
- **Evidence.**
  - SHOWN in the paper: asking for diversity while generating (WorldWeaver + Diverse, with earlier profiles in context) still returns the model's default concepts (Fig. 4, §4.1–4.2). Factorised sampling gives lower archetypality medians: 2 vs 3–4 (Fig. 6, App. F).
  - INFERENCE: a person hand-building families has the same pull toward a default. `M15.E.2` already says hand-built families "come out more balanced than real ones".
- **What it changes.** Moves topology from a single fixed fixture to a declared factor. Improves both fidelity (the theory's asymmetric families get covered) and inference validity.
- **Overlap.** PARTLY covered:
  - `M17.C.1` puts triangle topology, not kinship composition, in `D0`.
  - `M17.G.1` only marks "topology; family composition" as perturbed or unaudited.
  - `M17.G.2` enumerates triads only, and says the twelve-person family stays on `M15`'s ranges.
  - `M17.C.3` covers totals held fixed when arms differ in topology.
- **Cost / risk.** Moderate: compute rises by the bank size, and each entry needs `M2.2` validity (three generations, at least one cut-off tie). Do not confuse this with `M11.C.7`'s topology arms: here topology is an initial-condition factor shared by both arms, never the variable the arms differ in. No rule conflict.

**PW3. Infeasible `D0` draws: exclude only what is impossible, repair in a declared deterministic way, and count the repairs**
- **Proposed requirement.** `D0` **MUST** list its hard exclusions. Each **MUST** be an impossibility citing a requirement, for example a spouse gap outside `M2.A.0e`'s tolerance, or a pre-fixation-age agent with supplied `chronic_anxiety` (`M2.A.0a`). Rarity **MUST NOT** be a reason for exclusion, and `M10.C.4`'s bounds **MUST NOT** be used as a filter on draws. A draw that breaks an exclusion **MUST** be handled by a declared deterministic transform, or `D0` must be built so the exclusion cannot occur. Resampling until valid **MUST NOT** be used. The log **MUST** count repairs per exclusion ("N of M draws").
- **Where.** Phase E, `M17.C.1`; log in `M16.A`.
- **Evidence.**
  - SHOWN as practice: the paper's consistency repair fixes impossible combinations and is barred from fixing unlikely ones (§3).
  - INFERENCE: without that bar, a feasibility filter narrows `D0` toward the conventional family. Resampling-until-valid is the rejection sampling `M3.D.4a` forbids for engine draws, and it breaks keyed pairing across arms.
  - The paper does not count repairs; requiring the count follows global rule 6 (P2/P9).
- **What it changes.** Adds a constraint and a logging rule. Improves inference validity, and protects fidelity at the theory's unusual edges.
- **Overlap.** PARTLY covered:
  - `M3.D.4a` (no rejection sampling) applies to engine draws, not `D0`.
  - MM4 is a pre-run gate over `D0` that stops the run; it says nothing on per-draw handling or on rarity.
  - `M10.C.4` ("checks, never parameters") supports the bounds clause but does not mention `D0` filtering.
- **Cost / risk.** Low. No conflict.

**Already covered (one line each)**
- Separating surface attributes from behaviour-driving ones: covered by `M14.A.1` (classification register), `M15.A.1`/`M2.A.0h` (display names opaque), `M2.A.0g`/`M15.C.3` (sex out of pole assignment) and C5/`M1.F.8` (permutation equivariance). Sex is not surface in EPModel (`M2.A.0f`).
- Homogeneous-family control: same as `M17.D.1(c)`.
- Declared sampling measure for constants: same as PD4.
- Realised versus declared initial spread: same as TM1.

**Narrator line only**

**PW-X1. Per-category adherence audit for categorical narrated state**
- **Proposed requirement.** Before any narrator model is used, the project **MUST** report a confusion matrix of engine move type (the nine moves plus WITHHOLD) against the move type a validated non-LLM-judge reader recovers from the narration, per model and per category. Any category whose adherence is below a declared `[I]` floor **MUST** be either barred from narration for that model or flagged in every narrated output. An aggregate adherence figure **MUST NOT** be the pass criterion.
- **Where.** Phase F, `M16.E`.
- **Evidence.** SHOWN: on aggregate, one model followed most reaction types while following deflection 30.6% and meta-commentary 13.3% of the time; another model followed every category at 87.2% or more (Fig. 7, App. C).
- **What it changes.** Adds a test that catches a narrator silently softening CUTOFF or CONFLICT into gentler moves. This is the categorical form of BB2 and matches the silent-filtering finding in design lessons §2.8 (TIS).
- **Overlap.** PARTLY covered by BB1 (rank recovery), BB2 (transfer curve per model; continuous quantities only) and X6/X8.
- **Cost / risk.** Moderate. Requires a reader that is not an LLM judge (X7). Consistent with `M3.D.6` and `M16.E`.

**PW-X2.** A generic "be diverse" prompt does not move LLM defaults (Fig. 4): already covered by design lessons §3.2, §3.6 and X5, where the engine holds state and the narrator only renders it.
