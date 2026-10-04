# Reading reports: the 2026-09-15 to 09-30 "Agent Social-Simulation & Emotional Agents" digest (25 papers)

Produced 2026-10-04 at the owner's request, from a pasted digest of 25 arXiv papers. Four were already read in full in earlier batches and are not re-read here: 2609.24146 (*Mind or Message?*), 2609.24911 (SocioVerse2) and 2609.21997 (Bayesian Belief Layer), all in the 2026-09-27 reports, and 2609.17331 (SEAA), in `DESIGN_LESSONS_model_design_papers_2026-09-17.md` §7. The other 21 were downloaded from arXiv into `papers/Sweep 2026-10/`. Four of them had previously been screened on summary only (2609.28609, 2609.21857, 2609.26927, 2609.19913). All 21 were read in full from `pdftotext -layout` output by nine readers working under `sweep_readers_brief_2026-09-19/READER_TASK.md`. Each reader checked for overlap by keyword search in spec revision 10, `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, the 09-27, 09-27b and 10-04 report files, and TODO.md's revision-11 held section. Reports are reproduced as written. Only heading levels were changed.

**Status: none of these candidates is in the spec.** A plain-language explanation with examples is in `SPEC_CANDIDATES_plain_language_2026-10-04b.md`.

| Part | Papers | Candidates |
|---|---|---|
| 1 | 2609.36231 Blum Moyse & El Hady (patch-foraging identifiability); 2609.24012 Shao et al. (adequacy-aware calibration) | PF1–PF3; AQ1–AQ3, AQ-X1 |
| 2 | 2609.35819 Liu et al. (RePair causal mechanisms); 2609.33871 de Wynter (Population Physics) | RP1–RP2, RP-X1; PP1–PP2 |
| 3 | 2609.25970 Yukalov & Yukalova (emotions as coloured noise); 2609.25432 Altunyan & Edelman (tipping points) | EN1; TP1, TP-X1 |
| 4 | 2609.34372 Zhang et al. (PersMem); 2609.25284 Lin et al. (ReAdapt); 2609.17933 Hale & Gratch (AI mediators) | PM1–PM2, PM-X1; RA1–RA2; none |
| 5 | 2609.35036 Ren et al. (neutrality gap); 2609.21857 Krabbe & Shi (personality-tuned LLMs) | NG1–NG3, NG-X1; none |
| 6 | 2609.38296 Seckin et al. (radicalization); 2609.36278 Bouleimen et al. (illusory truth); 2609.19913 Berjawi et al. (digital twins) | RZ1–RZ2; IT-X1; none |
| 7 | 2609.37853 Liu et al. (AnthroDial); 2609.26927 Berga (AGIMUD); 2609.28609 Zhang et al. (AdvRole) | AN1, AN-X1; AG1–AG2; AR1 |
| 8 | 2609.34249 Huang et al. (CARE); 2609.38486 Freire et al. (anthropomorphism risks) | none; AT1–AT3, AT-X1 |
| 9 | 2609.23828 Starzyk & Galus (MEM); 2609.37009 Ma et al. (evacuation) | none; EV1–EV4 |

Candidates labelled `-X` are for the exploratory narrator line only and do not enter the v2 spec.


---

# Part 1: Identifiability and adequacy

## Reader report: Blum Moyse & El Hady, "Socio-cognitive models in a patch foraging setting: a case study for model selection and parameter identifiability methods", arXiv 2609.36231v1

**Scope.** I read every line of the pdftotext output (1,216 lines: main text, S1–S4 figure captions, S2 Table, references). The text is not truncated. Figures 2–9 and S1–S4 survive only as captions, so the per-model values of M (model-selection success) and P (parameter-recovery success) are known only from the prose. They are never given as numbers. Eqs. 2–4 are partly garbled; the prose makes them readable. Before proposing anything I checked these spec passages: `M11.1`–`M11.1f`, `M11.3` and its scoping table, `M11.4a`/`d`/`e`/`f`, `M11.C.18`/`C.20`/`C.26`, `M10.B.4`, `M10.C.4`/`C.4a`/`C.5`, `M1.A.4`/`4c`, `M16.A.1`–`A.10`, and `M17.A`–`M17.G`. I also searched by keyword the 2026-09-20 candidates file, the 09-27, 09-27b and 10-04 reading reports, and TODO.md's "Spec revision 11 — held" section. Terms searched: identifiab, recoverab, held-out, equifinal, degenera, discriminat, inert, sampling period, reachab, model selection, confusion, bootstrap, analysis layer.

### A. Report

1. **Formalism** [PAPER §2.1–2.2]. Drift–diffusion decision variable per agent, Euler step dt = 0.1. Two patches. Default N = 4 agents (§2.2.5). There are four social models from two choices: how social information is represented (counting = density; pulsatile = arrival and departure events) and how it enters the decision (threshold modulation or belief modulation). A non-interacting model (NI) is the baseline. Heterogeneity is drawn per agent as normal(u, β·u) (§2.1.4).
2. **Degeneracy is removed by fixing a constant** [PAPER §2.2.1]. Several parameter combinations give the same dynamics, so one of the threshold, drift and noise must be fixed; they fix σ = 0.1. The parameter space is a grid of 3 levels per parameter, 243 dynamics per model and condition. S4 Fig: recovery falls as the grid gets finer, and the ranking of models is mostly preserved.
3. **Method** [PAPER §2.2.4]. 600 simulations per parameter set, compared against 20 "in silico observations". Success is the fraction correct over 400 repeated inference runs. Bayesian inference (MAP) beats Wasserstein distance minimisation at small samples (Fig 7). Distributions of group accuracy beat distributions of departure times as the summary (Fig 6).
4. **Telling mechanisms apart and recovering parameters use different features of the output** [PAPER §4.1, Figs 4, 5, 8, 9]. Model-selection success tracks the rate of change D (oscillation). Parameter-recovery success tracks the level Q (global accuracy). Coarser sampling hurts model selection more for the oscillating models than for CB (Fig 8D). The authors' summary: identification depends on "observability and richness of the collective behavioral signature" (§4.1).
5. **Condition dependence** [PAPER §3.2.2]. Model selection is M-shaped in the initial distribution Qin: worst at 0.5, best at 0.25 and 0.75. Parameter recovery is U-shaped in Qin and improves with N, mostly between 2 and 6. A parameter can also become structurally unrecoverable under one condition: with p0 = p1 = 0 the learning parameter ζ has no effect, and its recovery error is 0.62–0.68 against 0.00–0.28 for the others (S2 Table).
6. **Similar macro outcomes from different mechanisms** [PAPER §1, §4.1]. The models produce "qualitatively similar macroscopic outcomes" such as aggregation in the better patch. NI is the easiest model to identify; the authors attribute this to fewer parameters and no coupling (§3.2.1).
7. **Limitations** [PAPER §4.2]. White noise only. Coarse grid. MAP rather than likelihood ratios or Bayes factors (Bayes factors did worse in preliminary runs). One mechanism per group. They state that identification should be "re-assessed for the specific experimental conditions" before each use.
8. **Small N** [PAPER]. N = 2 gives poor model selection for PT, PB and CT (Fig 8A). There is nothing on families, emotion or long horizons.

**Not transferable / cautions.** The inference machinery fits parameters to observed data. EPModel has no data and forbids fitting (`M11.F.9`), so MAP, Wasserstein fitting and the M/P scores do not transfer. What transfers is the idea of checking whether a model's outputs can tell its mechanisms apart, applied to the mutation suite and the sweeps. All values are from a 4-agent foraging DDM. None may be cited as a magnitude for EPModel.

### B. Candidate additions

**PF1. A mutant × criterion specificity matrix for the mutation suite**
- **Proposed requirement.** `M11` **MUST** record the results of the mutation protocol as a matrix: one row per mechanism mutant, one column per `M11.C` criterion, each cell red or green. The report **MUST** flag two things:
  - any two mutants with identical columns, because the suite cannot tell those mechanisms apart;
  - any criterion that goes red under a mutant outside the minimal rule set `M11.1d` lists for it.
- **Where.** `M11.1`/`M11.1d`, Phase B onward. It reads mutation runs that already exist.
- **Evidence.** The confusion matrices in Fig 6 ask the same question of models, and §1/§4.1 report that distinct mechanisms give similar macro outcomes. SHOWN for the foraging models; the transfer is INFERENCE.
- **What it changes.** Adds a reporting rule and a check, for inference validity. `M11.1` proves each test can fail. Nothing currently shows that the suite as a whole credits an outcome to one mechanism rather than another. `M17.D.2` does this for one rival arm per criterion; this does it for the whole suite at no extra run cost.
- **Overlap.** Partly covered: `M11.1d` gives the expected set per criterion, TM5 (held) gives a negative-control readout, and `M17.D.2(c)` covers "not discriminating" for differentiation criteria only.
- **Cost / risk.** Low. No conflict with M3.D.4/D.5, M3.D.6, M11.F.9 or M16.B.

**PF2. Per-constant inertness flag in sweeps**
- **Proposed requirement.** For each `[I]` constant swept under `M17.E.1`, Phase E **MUST** report, per reference configuration (`M17.E.4`), whether varying it over its declared range moved any criterion readout beyond the seed-to-seed spread.
  - A constant that moved nothing **MUST** be marked *inert in that configuration*.
  - Its "direction holds across the range" result **MUST NOT** be reported as robustness to that constant.
  - The `M10` register **SHOULD** carry the configurations in which each constant was found inert.
- **Where.** `M17.E.1`, `M10` register.
- **Evidence.** S2 Table and §3.2.2: with no reward, ζ has no behavioural effect and cannot be recovered (error 0.62–0.68). SHOWN; the transfer is INFERENCE.
- **What it changes.** Adds a reporting rule, for inference validity. A full fraction-of-range score from a constant whose mechanism is dormant in that configuration currently reads as robustness. `M11.3` makes the same point for criteria (calm against stressed). This applies it to constants.
- **Overlap.** Partly covered:
  - `M17.E.3`'s invariance interval does this for threshold constants only;
  - `M17.E.2(a)` covers only the dominant constant;
  - `M11.1` already catches a dormant mechanism in a mutation test.
- **Cost / risk.** Low, observer-side, computed from sweep runs already required. No conflict.

**PF3. Declared scale degeneracies among constants**
- **Proposed requirement.** The `M10` register **SHOULD** list groups of `[I]` constants that enter the dynamics only through a product or ratio. One example is an event-intensity scale and a conductance scale, which meet as intensity × conductance / `functional_level` (`M4.C.1`). Each group **SHOULD** have one member fixed by convention, and `M17.E.1` **SHOULD** sweep the group's combined quantity rather than each member independently.
- **Where.** `M10`, `M17.E.1`.
- **Evidence.** §2.2.1: one of three DDM parameters must be fixed (ARGUED, citing prior work). Which EPModel constants are degenerate is INFERENCE and has to be checked against `M4`.
- **What it changes.** Adds a register rule, for inference validity. Sweeping two degenerate constants independently samples one dimension twice, under a measure on their product that nobody declared (compare PD4).
- **Overlap.** Partly covered: `M17.E.2`'s ratio clause, `M17.E.1`'s named ratio, `M11.1c`'s rescaling mutants, and PD4 (held) on the sampling measure.
- **Cost / risk.** Low. No conflict.

**Already covered (one line each).**
- Level and dynamics readouts reported separately (Q against D): same as WA3. The paper adds independent evidence; WA3's scope could extend to `M17.D.2` and `M10.C.4a`.
- Identification depends on conditions and initial state: covered by `M11.3`, `M17.E.4`, TM4 and `M17.C.1`.
- Sampling-period effects: the log keeps every event (`M16.A.2`), and windowed readouts are covered by `M1.A.4c` and `M17.G.1`'s "estimator windows" dimension.

---

## Reader report: Shao, Li, Wang & Goto, "Testing, not presuming, adequacy: calibrating generative social simulators against emergent network structure", arXiv 2609.24012 (version not stated in the text)

**Scope.** I read every line of the pdftotext output (2,083 lines: main text, Discussion, Limitations, the corrections note, references, and SI Tables S1–S10). The text is not truncated. Figure 1 is garbled, but its caption is intact. The text carries no arXiv version tag. I checked the same spec passages and candidate files as for PF, and also searched for seeding of analysis-layer randomness (bootstrap, permutation, subsampling, design seed) and for any reachability or joint-attainment check on `M10.C.4`. Neither has a spec hit beyond PD4 and TM6.

### A. Report

1. **Formalism** [PAPER §3.7, SI ODD]. A single-pass sampler with no time, no agent interaction and no learning. N agents each draw a basket from a cached persona profile. The LLM is used once, offline, to write the profiles, and "never invoked inside the calibration loop" (§3.1).
2. **Synthetic recoverability** [PAPER §5.1–5.2, Fig 2, SI S2/S3/S3a].
   - Draw a known θ*, simulate, fit, check contraction and SBC rank uniformity. Contraction is 0.84, 1.00, 0.98 and 0.73 for κ, λ, ε, ν.
   - The variance-injection weight ν under-covers (0.85–0.88 coverage, 0/4 cells KS-consistent). Raising training from 1,500 to 8,000 simulations did not help, so ν is reported "descriptively only".
   - A within-persona tier tilt was only weakly identified (generating 0.70, recovered 0.115) because of model structure (§5.3). They dropped it before any real fit.
   - The authors keep "recoverability under the simulator" separate from identification in the real world.
3. **Adequacy** [PAPER §3.5, §5.4, SI S4–S6].
   - A prior-predictive reachability check: the observed summary sits at 1.6–1.8× the 95th-percentile null distance in every cell, so the real network is outside what the simulator can produce.
   - Per-statistic posterior-predictive checks (R = 999) localise the misfit. The 24 sub-samples are treated as a stability check, not a sample size, with a pre-declared rule that a block is localised only if at least 20 of 24 flag it.
4. **Repair and held-out audit** [PAPER §5.5–5.6, Table 2, S7].
   - A guided repair lowers the exceedance to 1.1–1.5× but does not restore adequacy.
   - An equal-budget unguided expansion stayed near baseline (paired difference, guided minus unguided, −1.30 to −3.21).
   - Two statistics never used in fitting, diagnosis or repair were checked afterwards. Edge-weight Gini passed in 3 of 4 cells. Buyer-breadth CV was extreme in 24 of 24 sub-samples in 3 of 4 cells, a failure "no earlier diagnostic surfaced".
   - The authors concede the audit statistics were chosen after the repair, not pre-registered (Limitations).
5. **Benchmarking the diagnostic itself** [PAPER §5.7, S9].
   - Known misspecifications were injected. The false-reject rate was 5.0% and 3.5%, and false localisation 1.0%.
   - Large value shifts (2.7 SD) were flagged 86% of the time. Concentration shifts of 0.7–0.9 SD were never detected.
   - Their reading: a non-flag cannot be read as absence of misspecification.
6. **The LLM input, tested** [PAPER §5.8, S10]. The LLM profiles beat a flat rule in 3 of 4 cells. Within-category brand relabelling caused no consistent degradation (permutation p ≈ 0.39 and 0.37). The input is therefore "partially validated", and its value comes from profile structure rather than brand identity.
7. **Software and reproducibility** [PAPER, corrections note, S4 note]. After submission the authors found 15 places where code and text disagreed. The prior-predictive null "did not seed it" and had to be redrawn, so the published ratios (2.1–2.9×) changed to 1.6–1.8×. Some of that change also comes from corrected inputs. Calibration seeds are disjoint from evaluation seeds (§3.2).
8. **Small N, families, emotion, long horizons:** none.

**Not transferable / cautions.** Posterior estimation, SBC coverage, posterior-predictive checks against a real network, and real-data reachability all need an empirical target. EPModel has none and forbids fitting (`M11.F.9`). The parts that need no empirical target are:
- the held-out audit, applied to criteria;
- the reachability idea, applied to the corpus bounds as checks;
- seeding of analysis-layer reference draws;
- benchmarking a diagnostic on injected faults (already in `M11.1`);
- disjoint seeds (WA4).

The paper's own lesson about bookkeeping argues for TM6.

### B. Candidate additions

**AQ1. A pre-declared held-out audit for any post-hoc revision**
- **Proposed requirement.** When a rule or `[I]` constant is changed after an `M11.C` criterion failed against it (`M10.B.4`), the change **MUST** be recorded with the criteria that motivated it. Before the revised model is run, a set of criteria and readouts **not** used to motivate the change **MUST** be declared. The revised model **MUST** report its results on that held-out set beside the motivating criteria, and any held-out criterion whose outcome changed **MUST** be named.
  - It **SHOULD** also be compared against an equal-flexibility change made elsewhere, so that "fixed by the right change" can be told apart from "fixed by any change".
- **Where.** `M10.B.4`, `M16.A.7`, `M17.G.1`.
- **Evidence.** Table 2 and §5.6: the held-out audit found a 24/24 failure that the in-loop diagnostics missed. S7 is the guided-against-unguided comparison. Both SHOWN as practice. The authors' own concession that the audit statistics were not pre-declared is the reason for "declared before the run" (ARGUED).
- **What it changes.** Adds a process and reporting rule, for inference validity. `M10.B.4` labels a result post-hoc but never checks whether the fix damaged or helped anything it was not aimed at.
- **Overlap.** Partly covered: `M10.B.4`/`M16.A.7` (constants marked as changed), SV4 (held; version count) and WA4 (development seeds disjoint). None of these requires a held-out set.
- **Cost / risk.** Low to moderate: a re-run of the suite, which `M10.B.4` already implies. No conflict.

**AQ2. Reachability of the corpus bounds, alone and together**
- **Proposed requirement.** For each `M10.C.4` bound that the ensemble misses at the declared constants (`M17.G.3`), Phase E **SHOULD** report whether any sample of `M17.E.1`'s sweep meets it, and whether any single sample meets all the bounds that apply at the same time.
  - A bound no sample reaches **MUST** be reported as a structural misfit of the mechanism, not a parameter matter.
  - The samples that reach it **MUST NOT** be selected, ranked or used to set defaults (PD4, `M10.C.4`'s "checks, never parameters").
- **Where.** `M17.G.3`, `M17.E.1`.
- **Evidence.** §5.4 is a prior-predictive reachability check, and the Discussion reports that the model "cannot jointly reproduce" four features that are reachable one at a time. SHOWN as method; the transfer is INFERENCE.
- **What it changes.** Adds a reporting rule that improves both theory fidelity and inference validity. It separates "the mechanism cannot produce what Kerr states" from "the invented constants are off". That is the bounded negative result the paper argues is knowledge, not failure.
- **Owner decision needed.** This sits next to `DESIGN_LESSONS` Q5 (excluding constant regions that violate `M10.C.4`). Reporting reachability is not exclusion, but it reads the bounds against the constant space. The owner should rule on whether this is admitted.
- **Cost / risk.** Low, since it reuses the sweep runs. Risk of drift toward history matching, mitigated by the no-selection clause.

**AQ3. Analysis-layer reference draws are seeded and logged**
- **Proposed requirement.** Every random draw made by the analysis layer — permutation nulls, sub-sampling, resampling intervals, sweep designs, ordering samples (`M17.C.2`), null-arm references — **MUST** come from a declared analysis seed, separate from engine seeds, recorded in `M17.G.1`'s audit record. A reported number whose reference distribution cannot be regenerated **MUST NOT** be reported.
- **Where.** `M17.G.1`, with TM6's ledger.
- **Evidence.** S4 note and the corrections note: the unseeded null had to be redrawn and the published ratios changed. SHOWN, though confounded with corrected inputs.
- **What it changes.** Adds a reporting rule, for inference validity. `M3.D.5` makes engine runs reproducible; nothing makes the statistics computed over them reproducible.
- **Overlap.** Partly covered: PD4 (held) gives the sweep a design seed, TM6 (held) records the seed set per quantity, and `M11.1b` keys its permutations.
- **Cost / risk.** Negligible. No conflict.

**AQ-X1 (exploratory LLM line only).** Any fixed input written by an LLM, such as persona text or narrator vocabulary, should be compared on held-out data against a flat-rule input and a within-category permutation null, so its value is tested rather than assumed (§5.8). This does not enter v2; `M3.D.6` keeps LLMs out of the engine.

**Already covered (one line each).**
- Calibration seeds disjoint from evaluation seeds: same as WA4.
- Recoverability of a latent from outputs: already `M11.C.26` (active sink), `M11.C.18`/`C.20` (estimator discrimination), and `M1.A.4c`.
- Benchmarking a diagnostic on injected faults, and its null false-positive rate: `M11.1` mutation, `M11.1c` null re-encodings, and CD3's positive-control fixtures.
- A non-flag is not absence of a problem: `M17.A.2` (power) and `M11.4a`.
- Comparisons at matched size: `M17.C.3` and CD2.
- Code and text disagreeing: TM6.

---

# Part 2: Causal mechanisms and emergence

## Reader report: Liu, Shang, Bhat & Jin, "Exploring Causal Mechanisms with Generative Agent-Based Models" (RePair), arXiv 2609.35819v1

**Scope.** I read all 1,714 lines of the pdftotext output: the body (§1–§10), Figures 1–8 (captions only), Tables 1–6, the references and Appendix A. The text is complete and not garbled, apart from the usual loss of equation layout (Eq. 2–3 can be rebuilt from the prose). I checked overlap with the following:
- Spec: `M0.4`, `M3.D.4a`, `M3.D.4c`, `M11.D.15`, `M11.4d`, `M11.4f`, `M10.B.4`, `M16.A.9`, `M17.A.1`–`A.4`, `M17.B.1`–`B.6`, `M17.D.1`–`D.5`, `M17.F.1`–`F.2`.
- Candidates and reports: SV1, TM2, TM3, TM5, AD3, WA4, CD1, J2.
- `TODO.md` "Spec revision 11 — held".

### A. Report

**1. Formalism.**
- [PAPER] A "world" has fixed *structural settings* and *operation settings* that are calibrated (§3.1, Fig. 2). In the segregation world:
  - one LLM call per household per half-year period, 20 periods, 32 homes, a one-action budget from an 11-action menu;
  - an LLM "environment" call resolves conflicting moves (§5.2).
- [PAPER] There is no seeding. "Pairing" means the rule run and its baseline share model, profile text, world text, initial conditions and interaction schedule (§3.4). LLM sampling noise is not coupled.
- [INFERENCE] `M3.D.4a`'s keyed draws are strictly stronger than this pairing.

**2. Agent architecture.** [PAPER] A candidate mechanism is one sentence appended to every agent's profile (§3.1, Tables 1, 3, 5). Nothing transfers to a rule-based policy (`M3.D.6`).

**5. Validation and reliability.** All [PAPER]:
- **Calibration (§3.2, Fig. 3).**
  - One operation setting is varied at a time, with registered execution checks and manual trace review.
  - "Existing checks are never removed or relaxed."
  - A baseline qualifies only if the required actions occur and the outcome has room to move in both directions.
  - No candidate rule is run during calibration. Settings are then frozen.
  - In the segregation world, trace review found unexecuted move decisions, and a check was added (§5.2). An open-ended action budget gave relocation in 1 of 6 baselines; one action per period gave 6 of 6.
- **Effect estimate (Eq. 2–3, §3.4).**
  - The signed paired difference δᵢ(r) = σᵣ[Yᵢ(r) − Yᵢ(0)].
  - "Control power" CP = mean/SD of the paired differences, which is the paired dz. The authors say it is an effect size, not power.
  - 90% intervals come from bootstrapping whole configurations, 10,000 resamples.
- **Segregation results (§5.3, Table 2).** 12 configurations, one run per condition, 108 runs. Baseline gap +2.4 points per period. R1 effect +3.6 (CP 1.30); R3 has the largest mean (+4.3) but ranks lower (CP 1.09) because it varies more. S2 and S4 have no established effect.
- **Public goods (§6, Table 4).**
  - 268 configurations × 6 conditions = 1,608 runs.
  - Negative reciprocity, the mechanism of the source experiment, adds 15.6 punishment acts yet *lowers* contribution frequency by 2.7 points.
  - Inequity aversion's wordings split. Half move the outcome the opposite way, ranging from −1.18 to +0.76 baseline SD (§6.4).
- **Resampling (§6.3, Fig. 6).**
  - The leading rule is recovered in every resample from n = 8.
  - All five effect conclusions are recovered in 51.9% of resamples at n = 20, 80.4% at 40, and 94.2% at 268.
  - The authors say plainly that more samples can sharpen an inconsistent effect without making it consistent (§8).
- **Sentence position (§6.4, Fig. 7a).** Moving the rule sentence to the start of the profile cost conditional cooperation 7.9 of its 12.0 points and shifted inequity aversion by 10.9 points.

**6. Failure modes the paper reports.** All [PAPER]:
- **Outcome moves without the claimed behaviour (§5.3).** Under "intergroup contact", neighbour-directed actions *fall* from 43% to 34%.
- **Behaviour moves without the outcome (§7.1, Table 6).** Value expression raises speaking turns from 2.5 to 6.2 with no polarisation.
- **Effect present before the claimed channel exists (§7.2, Table 6).** All three established cascade effects are already present in period 1, before any participant has seen others' contributions: first-period contribution 7.3 / 6.4 / 4.5 against a baseline of 5.7. The authors conclude the effect "cannot arise from observing previous contributions" and that a separate intervention on the information channel is needed (§3.5, §8).
- **Metric choice changes the conclusion (§6.2).** The same rule raises mean contribution size and lowers contribution frequency.

**7. Engineering.** [PAPER] Every prompt and response is logged as the trace (§3.1). Predicted direction and outcome metric are registered before testing (§3.1, citing Nosek).

**Not transferable / cautions.**
- Everything about wording pools and sentence position is LLM-only.
- [INFERENCE] One run per condition per configuration, 90% intervals, and no multiplicity correction across 23 rules. Calling 12 of 23 "established" is a weaker standard than `M11.4e` / `M17.A.4`.
- The segregation world uses racial group labels in prompts, and the authors flag this risk themselves (§9). Nothing here is evidence about families.

### B. Candidate additions

**RP1. Per-seed divergence front: in each person, an arm difference must not appear before an event route from the intervention could reach them.**
- **Proposed requirement.** For every two-arm comparison under `M3.D.4a`'s coupled draws, the analysis layer **SHOULD** compute, per seed and per person, two ticks:
  - the first tick at which that person's state differs between the arms;
  - the earliest tick at which an event descended from the arms' first differing event could have been delivered to them, given per-edge latency (`M3.C.1`) and witness routes (`M1.F.5`).

  A person whose state diverges *before* any event route reaches them **MUST** have that divergence attributed to a declared family-level channel: the societal anxiety input, the undifferentiation budget (`M6`), or shared beliefs (`M9`). Otherwise the divergence **MUST** be flagged as unexplained coupling.
- **Where.** The observer and log diff go in `M16` (pure observer, `M16.B`). The report goes in `M17.B`. It extends SV1, which only requires zero divergence before t*.
- **Evidence.**
  - RP §7.2, Table 6 and §3.5: an effect that comes before the proposed channel can operate cannot be credited to that channel. SHOWN in the paper.
  - The transfer to a per-person, per-tick check is INFERENCE. It only becomes possible because `M3.D.4a` makes two arms identical until something actually differs.
- **What it changes.** Adds a test and a reporting rule; improves inference validity. It catches:
  - a non-local read, such as an agent or observer touching global state;
  - an arm that leaks through a channel nobody credited;
  - a criterion whose "mechanism" result arrives through the family-level budget rather than the dyadic route the criterion names.

  SV1, `M11.D.15` and `M17.A.2` (ensemble first tick) cannot see any of these.
- **Cost / risk.** Low: a log diff plus a breadth-first search over the event graph. No conflict with `M3.D.4`/`M3.D.5`/`M16.B`.
- **Owner flag.** Which family-level quantities may act on everyone in the same tick without an event is a Bowen-theory question. The candidate's list (societal anxiety, the budget, shared beliefs) is taken from the spec's objects, not from the corpus.

**RP2. Each criterion names its proximal behaviour and reports it beside the outcome.**
- **Proposed requirement.** Every `M11.C` criterion **MUST** name, before it is run, the proximal behaviour through which the credited mechanism acts. This is a move class, event type or route, for example TRIANGLE moves on the named triad, or CUTOFF events on the named tie. The ensemble report **MUST** give that behaviour's per-seed paired arm difference beside the outcome's, and **MUST** classify the result as one of:
  - outcome and behaviour both moved in the predicted direction;
  - the outcome moved but the behaviour did not, or moved the other way;
  - the behaviour moved but the outcome did not.

  Only the first may be described as support for the mechanism.
- **Where.** `M11.C` table (a column) and `M17.B`.
- **Evidence.** RP §5.3 (S2: contact falls), §6.2 (R3: punishment up, cooperation down), §7.1 (discussion up, polarisation absent), and §8's guideline to test the link from behaviour to outcome. SHOWN in the paper. The transfer is INFERENCE.
- **What it changes.** Adds a reporting rule; improves validity. TM2 does this only for arms that assign a person attribute, and TM5 covers the negative control. RP2 is the positive counterpart for every criterion, including intervention and mechanism-switch arms.
- **Cost / risk.** Low, because `M4.E.1` already logs every move. **Owner flag:** naming the proximal behaviour per criterion is a theory decision: which move or route the corpus credits. Where the corpus does not say, the criterion should state that rather than pick one.

**Already covered (one line each):**
- Room for the outcome to move in both directions: `M11.4d`, plus `M17.D.5`.
- Settings frozen before candidate rules are run: `M10.B.4`, WA4.
- Fresh baseline per pair: WA4.
- Registered direction and primary metric: `M0.4`, `M11.C.38`, `M17.A.4`.
- Mean versus SD of paired differences (control power), and "precision is not consistency": `M17.B.1`'s converted/reversed strata, with `M17.A.4`.
- Information-channel follow-up intervention: `M17.D.1(a)`, delivery suppressed.
- Metric choice changes the conclusion: `M11.4e` multiplicity, TM3.
- Checks never relaxed: `M11`'s mutation protocol plus the global testing rules.

**RP-X1 (narrator / LLM line only).** If an LLM persona attribute is ever tested, it **MUST** be tested over a declared paraphrase pool that preserves experiencer, attitude and target, and at more than one sentence position. In this paper, position alone moved one effect by 10.9 points (§6.4) and wordings split in sign. This is already in the spirit of design lessons §3.3, so the narrator line gains little.

---

## Reader report: de Wynter, "Population Physics, Population Problems: Safety and Emergence in LLM Societies", arXiv 2609.33871v1

**Scope.** I read all 1,392 lines: §1–§7 and Appendices A–K, including Tables 1–9 (figures as captions only). The text is complete. It has several internal inconsistencies, listed under the cautions below. I checked overlap with:
- Spec: `M16.A.9`, `M11.C.40`, `M17.B.2`, `M17.B.5`, `M17.F.2`, `M15.D.4`, `M11.D.21`, `M11.4d`.
- Candidates and reports: J2, J3, CD1, WA1, WA2, AD3, and the `TODO.md` held section.

### A. Report

**1. Formalism.** [PAPER]
- **LLM-Schelling:** agents are called in sequence and one call is one tick. The informed run had 2,776 ticks and 140 agents; the uninformed run had 19,012 ticks and 240 agents (§4.1).
- **Rogue:** 300 agents, order shuffled each era, i.i.d. activation (§4.1, App. D.2).
- **Moltbook:** an observational archive of 21,668 agents (§4.1).
- Initial grids are random i.i.d., with a random temperature per agent "for diversity" (App. D.1).

**3. Order parameter and diagnostics.** [PAPER]
- **Φ(t)** is the mean per-agent Shannon entropy over a chosen activity partition, counting only agents with at least m_min actions in the window (Eq. 3).
- **Null (App. I).** An analytic form (the entropy of category marginals, Eq. 4) and a sampled band: 100 draws that preserve per-agent and per-category counts.
- **Validity checks (§4.3).**
  - V1: observed values outside the null band for at least 80% of the window.
  - V2: R² ≥ 0.7.
  - V3: parameters stable under preprocessing variants.
- **Regime test.** Stretched-exponential (Eq. 1) versus logistic (Eq. 2) fit, choosing by ΔAIC > 2. β pinned at the fitter bound (5) or collapsing to 1 is read as shape mismatch.
- The author states that the analytic null is the limit for agents with infinitely many acts. The sampled band adds "finite-population variance" (App. I).

**5. Results.** [PAPER]
- Uninformed Schelling: β = 0.43, classical regime. Informed Schelling: β = 1.42, compressed.
- Moltbook and Rogue are logistic-preferred (Table 5).
- β depends on grid size and occupancy: for example, informed 10×10 drops from 3.70 to 1.59 as occupancy rises (Table 1, 2–6 seeds per cell).
- GovSim and ChatEval sit at their fixpoint from the first observation, with Φ₀ = Φ∞ (Table 7). In 20 of 21 split ChatEval items, the same referee dissents across seeds (App. J.2).

**6. The "coordinated subset" claim.**
- [PAPER] Benign agents engage with malicious posts at 0.85 of the ambient rate. The ambient rate itself tracks the malicious share roughly linearly: 0.1→0.07, 0.3→0.20, 0.5→0.37 (§5.3, Table 2).
- [INFERENCE] The subset is never *detected*. It is defined by assigned role, and the bad-reach partition requires knowing which posts are malicious (Eq. 5). The "agent-agnostic" entropy statistic shows concentration, not who drives it.
- The Discussion calls the subset "uncoordinated" (§6), while the abstract says "coordinated".

**Not transferable / cautions.**
1. **Curve-shape fits do not suit a ~12-agent family.** Fitting relaxation shapes needs thousands of points of a population-level trajectory that settles toward saturation. A 12-agent family is expected to stay in "perpetual disequilibrium" at the dyad level (design lessons §2.12), and design lessons §2.2 already says finite-N statistical-physics results do not carry over. Logistic-versus-stretched-exponential should not be used to call `M15.D.4` turning points: those are in parameter space, not time.
2. **No transfer to triangle or coalition detection.** The paper offers no method for finding a coordinating subgroup without knowing its membership in advance. EPModel stores triangles and their inside/outside positions as state (`M1.C`), so it does not need to infer them.
3. **Internal inconsistencies.**
   - Occupancy is "65% sparsity" in App. D.1 but "0.35%" in App. E.
   - GovSim defaults are s₀ = 100 in §5.5 but s₀ = 30 in App. D.1.
   - ChatEval is "81 configurations" in §5.5 but "58 runs" in App. J.2.
   - The Table 6 caption says logistic is preferred everywhere, but the table lists Schelling-informed as stretched-exponential.
   - The uninformed-Schelling ΔAIC is given as ≈5,500 in the Table 4 caption, against the ≈1,140 quoted for a different comparison.
   - Cited model names and a July 2026 incident are not verifiable from the text.

### B. Candidate additions

**PP1. Repertoire entropy is reported against a count-matched null, and `M11.C.40` compares on the gap.**
- **Proposed requirement.**
  - Wherever `M16.A.9` reports a person's per-channel repertoire entropy for a window, the readout **MUST** also carry the number of selections it was computed from.
  - It **MUST** also carry a null band: the entropy of the same number of draws from the family-pooled move distribution of that channel in that window, from a declared number of draws keyed outside the engine.
  - `M11.C.40`'s comparison **MUST** be made on the gap between the observed value and its null, or on count-matched subsamples, not on raw entropy, whenever the two arms or the two ends of the run differ in selection count.
- **Where.** An amendment to `M16.A.9` and `M11.C.40`. It is the move-repertoire counterpart of J2, which covers structural readouts over ties and triangles, and of CD1, which covers JSD.
- **Evidence.**
  - App. I: the sampled band exists because the analytic value ignores finite counts (SHOWN as method).
  - [INFERENCE, my computation] Take a fixed 9-move distribution with true normalised entropy 0.906. Plug-in normalised entropy averages:
    - 0.556 at 5 selections;
    - 0.711 at 10;
    - 0.869 at 52;
    - 0.899 at 260.

    The behaviour does not change; only the count does. Self-directed selections are rarer than automatic ones. The mixing weight depends on differentiation (`M4.D.1a`), so channel counts can shift over a run and between arms. A raw-entropy fall can therefore be a count artefact, and `M11.C.40`'s channel comparison is exposed to it.
- **What it changes.** Corrects a readout and protects an existing criterion; improves inference validity.
  - The gap to the family-pooled null also separates two things: a person concentrating on what the whole family does (family style, `M4.D.6`) versus a person concentrating differently from the family.
  - WA1 covers between-person dispersion of one share. This covers concentration over the full repertoire.
- **Cost / risk.** Low and observer-side (`M16.B.3`). Null draws must be keyed and kept outside the engine (`M3.D.4a`). **Owner flag:** reading a large below-null gap as role complementarity, such as pursuer and distancer, is a theory interpretation. Until the owner rules, the readout stays labelled as concentration only, as `M16.A.9` already requires.

**PP2. A fitted shape parameter at its bound is "inconclusive", not a regime (low priority).**
- **Proposed requirement.** If `M17.B.2`'s regime classification is implemented by fitting any functional form to a trajectory, a fit whose parameter sits at a declared bound, or fails a declared fit-quality floor, **MUST** be classified inconclusive and counted as such, never assigned the regime the bound implies.
- **Where.** `M17.B.2`.
- **Evidence.** §4.3 and App. J, Table 8: β pinned at 0.05 or 5.0 in most GovSim cells, which the author reads as "no trajectory", not as a regime (SHOWN as practice).
- **What it changes.** A reporting rule, conditional on the implementation choice. Improves validity.
- **Cost / risk.** Negligible. It is moot if `M17.B.2` is classified without curve fitting, which the spec leaves open.

**Already covered (one line each):**
- Fixpoint from onset versus dynamics: `M17.B.2` ("settled at its attractor") and `M11.4d`.
- V3 stability under window or threshold choice: `M17.G.1` (estimator windows), CD4, J3.
- Finite-size dependence of the shape parameter: design lessons §2.2.
- Agents with fewer than m_min actions silently excluded: the global surface-what-you-drop rule. If a threshold is adopted, the count excluded must be logged.

No narrator-line candidates. The paper's LLM findings (safety tuning, model-dependent timescale 3.8× slower for Qwen) add nothing to design lessons §3.

---

# Part 3: Emotion as noise; tipping points

## Reader report: Yukalov & Yukalova, "Emotions as intrinsic colored noise in biological systems", arXiv 2609.25970v1

**Scope.** I read all 1,721 lines of the pdftotext output: §1–7, Figs 1–12 (captions only, since the plots did not survive extraction) and the references. The text is complete and the equations are readable. In the spec I checked `M3.D.4`, `M3.D.4a`, `M3.D.4b` (its draw-class table, including the rows "Move selection (softmax)", "Channel mixing-weight noise, if any" and "Receiver-side appraisal noise, if any"), `M3.D.4c`, `M4.C.1`, `M4.D.1`, `M4.D.1d`, `M4.D.2`, `M4.D.6`/`M4.D.6a`, `M1.A.7a`, `M1.A.8`, `M1.A.3d`, `M5.E.6`, `M10.C.1`, `M11.D.21`, `M17.B.2`, `M17.E.2`, `M17.E.3`, `M17.F.2` and `M17.G.1`. I also keyword-searched the candidates file, the three earlier sweep-report files, today's file and TODO's revision-11 section for "noise", "colored", "autocorrel", "oscillat", "chaos", "imitation", "tick length" and "discretis".

**The premise of the brief does not hold for this paper.** In this paper "colored" means noise with a **nonzero mean**, not noise that is correlated in time. §2.3 calls the noise colored because average attraction factors are nonzero, and §7 says the noise is "colored having a nonzero mean" [PAPER]. No temporally correlated noise process appears anywhere. In fact the model has no random draws at all: equations 28 and 45–53 are deterministic maps over group-level choice probabilities [PAPER]. The paper therefore says nothing on whether memoryless per-tick randomness matters.

### A. Report

1. **Formalism.**
   - [PAPER] §2: the probability of choosing option n is a utility factor plus an attraction (emotion) factor (eq. 5). The utility factor is a Boltzmann/Luce form (eq. 8). Attraction factors sum to zero across options (eq. 12).
   - [PAPER] The "quarter law" sets the mean attraction factor at ±0.25 (eqs 13–15). Results are clipped to [0, 1] by a retraction map.
   - [PAPER] §3: groups update in discrete time τ by mixing their own probability with other groups' probabilities, weighted by an imitation parameter ε (eq. 28). Groups are mean-field aggregates (eq. 35), not individuals.
   - [PAPER] §3 argues that difference equations "possess much richer dynamics" than the matching differential equations, citing Tong 1990.
2. **Agent heterogeneity and memory.**
   - [PAPER] Emotion decays as accumulated information arrives: q(t) = q(0)·exp(−M(t)), where M is summed KL divergence from other groups (eqs 30–34).
   - [PAPER] Long-range memory sums M over all past steps (eq. 39). Short-range memory keeps only the latest step (eq. 40). Super-rational agents have q = 0 and never imitate (eq. 53).
3. **Regimes.**
   - [PAPER] §4–5 report eight regimes: node+node, node+focus, node+center, center+center, focus+focus, node+chaos, chaos+chaos, and finite-time consensus.
   - [PAPER] With ε = 0, short-memory groups can oscillate forever (Fig. 4, with f2 = 1 and q2 = −0.9).
   - [PAPER] Chaos appears only when ε > 1/2 (eq. 50). In Fig. 6, at ε1 = 0.996 and ε2 = 0.9, both groups are chaotic. At ε = 0.6 and 0.7 both converge monotonically.
   - [PAPER] Most of the striking regimes use initial attraction factors close to their bounds: q2 = 0.999, −0.99 or −0.999 (Figs 5, 6, 9).
   - [PAPER] "Chaotic" is a label read off the plots. I found no Lyapunov exponent or other test for it in the text.
   - [PAPER] With super-rational agents present, Fig. 10d shows consensus at t0 ≈ 100 only at f3 = 0.860056, a single tuned value.
4. **Initialisation.** [PAPER] Initial conditions p(0) = f + q are hand-set for each figure. Sensitivity to them is shown only by varying one parameter per figure.
5. **Validation.**
   - [PAPER] The model is not validated against any data.
   - [PAPER] The quarter law is said to be "persuasively confirmed by numerous empirical data", citing the first author's own 2022 paper.
   - [PAPER] In §6.2, ε = 0.1 is justified by "10% of subjects are almost pure imitators". That is a share of people, not a per-agent imitation probability.
6. **Limitations stated.** None in a dedicated section.
7. **Software engineering.** Nothing.
8. **Emotion and small N.**
   - [PAPER] The paper's central claim is that information exchange reduces the influence of emotion (§6.2, §7).
   - [INFERENCE] This runs against the corpus. Bowen's account has contact in a fused system raising reactivity, not reducing it. It must not be imported.

**Answer to the brief's question about memoryless randomness** [INFERENCE, not from the paper]:
- Under `M3.D.4a`, every draw is a pure function of the seed and a key that includes the tick and that `M3.D.4a` forbids from containing state. Draws are therefore independent across ticks by construction: white noise.
- All persistence in behaviour comes from named state: acute anxiety decaying toward the chronic floor (`M1.A.8`, `M1.A.7a`), reinforcement (`M4.D.6`), and tie state.
- That is the right default. A temporally correlated noise term, such as an Ornstein–Uhlenbeck mood process on propensities, would add persistence that no Bowen mechanism accounts for. It would look like chronic anxiety or a characteristic family style, could pass criteria that are meant to credit those mechanisms, and could not be mutation-tested against the corpus.
- `M3.D.4a` does not currently forbid such a process, because it could be written as a state variable driven by keyed draws.
- **Owner question:** should the spec say explicitly that any stochastic term in appraisal or selection is independent across ticks, and that temporal persistence comes only from named state variables with a corpus source? This is a theory decision, so I have not written it as a candidate.

**Not transferable / cautions.**
- The quarter law (±0.25), the additive utility-plus-emotion form, the Ellsberg resolution and the "super-rational agent" concept are not Bowen theory.
- The paper's super-rational agents pull others toward utility through imitation, which superficially resembles a well-differentiated member calming a family. The mechanism is different and must not be used as a model of differentiation.
- Mean-field group probabilities are not a 12-person family.
- Chaos at ε near 1 is a property of an overshooting discrete map, not evidence about emotion.

### B. Candidates

**EN1. A seed classified as oscillating must survive a finer fast tick before the oscillation is credited to a mechanism.**
- **Proposed requirement.** When `M17.B.2` classifies a seed's trajectory as oscillating or inconclusive (non-settling), Phase E **SHOULD** re-run that seed with the fast tick subdivided and the per-tick rates rescaled. The oscillation **MUST NOT** be reported as a property of a mechanism unless it persists. An oscillation that disappears when the tick is refined **MUST** be reported as a discretisation artefact.
- **Where it would live.** Phase E, `M17.B.2`, with `M17.G.1`'s "tick length" dimension marked as audited for that result.
- **Evidence.**
  - §3 argues that discrete-time maps have richer dynamics than continuous ones (ARGUED).
  - Figs 4–6 show permanent oscillation and chaos arising from a noise-free discrete update once coupling exceeds 1/2 (SHOWN, for their map only).
  - Transfer is INFERENCE. `M4.C.1`'s gain of intensity × conductance / `functional_level` can become large at weekly ticks, which is the overshoot condition.
- **What it changes.** Adds a check that separates `M5.E.6`'s required damped oscillation (theory) from oscillation created by tick size. This improves inference validity.
- **Cost / risk.**
  - Moderate. Rescaling per-tick `[I]` rates to a sub-tick is itself a modelling choice, and the rescaling rule would need declaring.
  - `M17.G.1` already lists tick length as an audit dimension (partial overlap). What is new is tying the check to the oscillating class.
  - No conflict with `M3.D.4a`, provided the sub-tick enters the key.

**Already covered:**
- Regime classification, including non-settling and an inconclusive class: `M17.B.2` and `M11.D.21`.
- Whether the decay rate (memory length) decides settle versus oscillate: `M17.E.2`'s dose–response on the dominant constant.
- Within-run variance alongside the mean: WA3.

## Reader report: Altunyan & Edelman, "Tipping Points in LLM-Based Multi-Agent Systems: Stance on Climate Change Action", arXiv 2609.25432v1

**Scope.** I read all 1,002 lines: §1–8, Appendices A–F, and the references. The text is complete, but all result figures (Figs 1–7) survive only as captions. In the spec I checked `M0.4`, `M1.A.3d`, `M15.D.4`, `M17.B.2`, `M17.B.4`, `M17.E.1`–`M17.E.3`, `M17.F.1`, `M17.F.2`, `M17.G.1`, `M11.4e` and `M10.C.4a`. I also checked held candidate BB5 and its neighbours J3, J4 and C13 (`M4.G.3`), and today's file for WA2, WA3, WA5, CD1, CB1 and TM1–TM3. Keyword searches covered "tipping", "change-point", "jump", "LOCF", "carried forward" and "imitation".

### A. Report

1. **Formalism.**
   - [PAPER] §3: the setup is the Ren et al. 2024 generative-agent framework with 10 GPT-4o-mini agents, run for two simulated days.
   - [PAPER] There are two conditions, with 1 or 2 "committed minority" agents. Each condition was run once.
   - [PAPER] It took about 385M input tokens and three weeks of runtime; one run costs about $100 (§3, §7).
2. **Agents and initialisation.**
   - [PAPER] §3.1, App. A–B: a stance is two numbers in [−1, 1], conviction and institutional trust. They were drawn uniformly, mapped to one of four archetypes, and written into biographies by an LLM, which was told not to state the number.
   - [PAPER] A name cue alone switched one replication run into Spanish (§3.3).
3. **Detecting tipping.**
   - [PAPER] §4.1 defines a tipping point formally as a bifurcation: a control parameter crosses a threshold (citing Kuznetsov).
   - [PAPER] The measurement in §4.2 is something else. An LLM judge (GPT-5.2, temperature 0) scores each utterance. A "round" is the interval in which every agent has spoken at least once, with scores averaged within a round. The per-agent Euclidean jump between rounds t and t+1 is then compared across all pairs of transitions with paired t-tests and Bonferroni correction (m = 66).
   - [PAPER] Results for the 1-minority condition: 0→1 is larger than 4→5, 5→6, 6→7 and 10→11, and 1→2 is larger than 11→12 (Fig. 2b, App. E). For the 2-minority condition, only 11→12 > 12→13 is significant (App. D).
   - [PAPER] An LDA topic model (k = 2, 3, 5) was fitted on 12 and 18 round-documents.
4. **The "tipping" is an initialisation artefact by the authors' own reading.**
   - [PAPER] §5: the jump at round 0→1 is "more likely to be attributed to LLM biases" than to interaction.
   - [PAPER] Conviction rose immediately in agents initialised low on it.
   - [PAPER] There is no zero-minority arm, so no comparison can attribute any change to the committed minority.
5. **Measurement failures.**
   - [PAPER] Missing final-round values were filled by last observation carried forward (LOCF), because the runs were cut at a fixed horizon (§4.2).
   - [PAPER] Trust "flattened" at 0 because institutions were never discussed. §5–6 say avoidance was read as neutrality.
6. **Limitations.** [PAPER] §7: at least 10 runs per condition would be needed and were unaffordable. Cusp-catastrophe fitting is planned but not done.
7. **Software engineering.** [PAPER] §8 argues for preregistration.
8. **Small N.** [PAPER] N = 10, one run per arm, no seed variance reported.

**Not transferable / cautions.**
- No result here is evidence of a tipping point. It is one run per condition with no control arm, the largest change is a start-up transient, and the readout is an LLM judge.
- Cusp fitting would be fitting to data, which `M11.F.9` forbids for counterfactuals.
- Everything about the LLM pipeline belongs to the X line at most. It confirms design-lessons §3.2–3.4 and adds nothing new.

### B. Candidates

**TP1. A criterion asserting abrupt change must say whether the change is abrupt in time or abrupt in a parameter.** I propose folding this into BB5.
- **Proposed requirement.** Every `M11.C` criterion or Phase E result that asserts a jump, tipping point or turning point **MUST** declare which of two things it claims. Each needs its own evidence:
  - (a) **abrupt in time**, at fixed constants. Evidence: the trajectory through the event (`M17.F.1`), with the per-tick change compared against the matched no-event arm and taken only after `M17.F.2`'s settling condition is met.
  - (b) **abrupt in a parameter**. Evidence: the outcome across a sweep of the named control parameter (`M17.E.2`), with the invariance interval or roll-off width reported (`M17.E.3`, `M1.A.3d`).

  A result evidenced only one way **MUST NOT** be reported as the other.
- **Where it would live.** `M11` mutation protocol, as a clause of held candidate BB5. Also referenced from `M15.D.4`.
- **Evidence.**
  - §4.1 defines tipping as a bifurcation in a control parameter (ARGUED).
  - §4.2 and Fig. 2b measure only jumps between consecutive rounds at fixed settings. The largest such jump was a start-up transient the authors attribute to LLM bias (§5). This is SHOWN as a failure in the paper's own design.
  - Transfer is INFERENCE.
- **What it changes.** Adds a reporting rule and improves inference validity. BB5 covers "threshold, jump or bistability" but does not separate the two senses. `M15.D.4` already uses the parameter sense ("turning point", revised at revision 7), while `M17.F.1` uses the time sense.
- **Cost / risk.** Low; it is a declaration plus routing to tests that already exist. No conflict with any rule.

**Already covered:**
- Excluding the initial transient before reading a jump: `M17.F.2` and J3.
- A control arm without the perturbation: `M0.4`, `M17.D.1`.
- Dose of the perturbation (1 versus 2 minority agents): WA5.
- Multiplicity correction across many transition comparisons: `M11.4e`.
- Filling a horizon-truncated value by LOCF: `M17.B.2` (inconclusive at horizon) and PD1 (censoring flag) forbid the equivalent.
- Absence of activity read as a neutral value: TM3 (opportunity denominators), CB1 (layer named), and the estimator analogue already in `M1.A.4b`.
- The realised initial state differing from the assigned one: TM1 and TM2.
- Readout windows defined by "everyone acted once": not applicable, because `M3.E.2` has every person select every tick. Relevant only under `M17.E.6` and `M17.C.2`.

**TP-X1** (X line only). An LLM-judged stance on a dimension the conversation never touches will score as neutral. Any narrator fidelity check should report topic coverage before scoring stance, per §5 and App. F. This is largely covered by X7 and X8.

---

# Part 4: Memory, relational reasoning, mediators

## Reader report: Zhang, Xiang, Xie, Liu & Song, "PersMem: Internalizing Personality into Dual-Pathway Memory for LLM Agents", arXiv 2609.34372v1

**Scope.** I read the whole of `2609.34372.txt` (2,259 lines, 39 pages): the body, Appendix A (a philosophical essay) and Appendices B–I. The text is clean. Equations are partly garbled but can be rebuilt from the prose. Spec passages I checked:
- read in full: M9.1–M9.8, M4.B.2, M4.C.1–M4.C.9, M4.D.1–M4.D.6e, M1.C.4, M16.A.1–M16.A.10;
- found by keyword: M11.1b, M11.1d, M17.E.5, M10.C.4a.

Other files checked: the TODO "Spec revision 11 — held" section (the owner answers to MM1 and SV2); MM1–MM4 in `SWEEP_READING_REPORTS_2026-09-27b.md`; the WA/CD/CB/BV/PW headlines in `…-10-04.md`; and `docs/theory/_LEDGER.md` L09.3–L09.4 plus `fe05.md` FE05.10 and FE05.17. Search terms: memory, recall, retention, forget, decay, persist, belief revise/update, valence, attachment.

### A. Report

**2. Agent architecture**
- [PAPER] §3 and App. C–D: a fixed trait vector sets parameters for four memory operations. Attachment uses π = (anxiety, avoidance) in [0,1]²; Big Five uses five dimensions. The four operations:
  - **Appraisal** (Eq. 2): anxiety deepens negative valence, raises arousal for negative inputs and lowers dominance. Avoidance dampens arousal.
  - **Retention** (Eq. 4): decay rate is r0(1 − c·arousal)·max(ε, 1 − ρ·anxiety·[−valence]+). High-arousal memories and, for anxious holders, negative memories decay more slowly.
  - **Passive retrieval** (Eqs. 5–7): a weighted score α·semantic + βπ·affective relation + γ·retention, followed by softmax sampling from the top K.
  - **Gated redirection of active search** (Eq. 8): the gate threshold θ0 − η·anxiety falls as anxiety rises.
- [PAPER] App. C.2, Eq. 3: current affective state is an exponential moving average with λ = 0.40, updated twice per turn (after the input and after the reply). The trait vector never changes (C.9).
- [PAPER] Table 9: 24 fixed coefficients. Three gains come from a small MLP fitted by an evolution strategy to "prespecified calibration targets" (Eq. 13, Eq. 22). The authors call these targets "design targets rather than effects estimated" from human data (App. D.3), and §5 says the mappings "should not be interpreted as validated".

**5. Validation, ablation, negative results**
- [PAPER] Table 1 / Table 12, held-out attachment classification from recalled content and gate traces only:

  | Condition | Accuracy |
  |---|---|
  | Full system | 48.1% (CI 40.0–56.3) |
  | Neutral gate (stochastic gate kept, trait term removed) | 40.0% |
  | Gate off | 25.0% |
  | Chance | 25% |

  Two things follow:
  - With internal scores included, accuracy is 75.0% (App. E.1). Separability measured on internal state therefore overstates what the outputs show.
  - The neutral-gate versus gate-off pair separates "the mechanism exists" from "the mechanism depends on disposition".
- [PAPER] Table 13, 2⁴ component factorial, average main effects: appraisal +0.005, retention +0.013, passive retrieval +0.130, gate +0.085. The authors say these "do not identify interactions" (E.3).
- [PAPER] App. F.4, a negative result: neuroticism is written into the appraisal rule as raising arousal (Eq. 15), yet recalled arousal correlates with neuroticism at r = −0.518, opposite to the declared direction. A direction coded into one rule did not survive composition with the other terms.
- [PAPER] App. F.4: the Big Five recall correlations (N–valence r = −0.703, E +0.776, A +0.607, C–OGM −0.710) "primarily check whether the implementation expresses its prespecified relationships". They are a manipulation check, not evidence.
- [PAPER] App. E.5: a monotone anxiety sweep over 5 levels. PCB goes −0.440 → −0.328 → +0.013 → +0.450 → +0.647, and OGM goes 0.21 → 0.32.
- [PAPER] Statistical caveats the authors state:
  - Dialogue evaluations are descriptive: pairs are judged twice, and one 10-turn sequence per profile gives correlated responses (E.8, F.3).
  - The adversarial-framing result (89% vs 58% retention) rests on 12 pairs (H.3).
  - The CoSER comparison in Table 6 mixes evaluators across studies. PersMem was scored by DeepSeek and the reference rows by their own source studies, so that table is not a like-for-like comparison.
- [PAPER] Horizon: 6 or 10 turns per session. Nothing in the paper runs long.

**7. Software engineering**
- [PAPER] App. F.6: paired random streams across conditions; deterministic index tie-breaking; a manifest with model digests; 72 regression tests covering switches, gate modes, paired streams and split overlap. This is already required by M3.D.4a, M4.D.1f and M16.A.1.

**8. Relevance to EPModel's memory and appraisal**
- [INFERENCE] M9 has a write rule (M9.8: written only by delivered events) and a hysteresis rule for institutional acts (M9.3). By keyword search it has **no rule for how a per-person belief persists, decays or is revised** over a 2,080-tick run. PersMem shows that "where disposition enters" (writing, keeping or reading a memory) is a design choice with very different effects (Table 13).
- [INFERENCE] **Does disposition-dependent retention or retrieval have a corpus counterpart?** Partly, and the parts do not point the same way:
  - **L09.4:** the family system "operates always to obscure and misremember". This is family-level distortion toward forgetting, or treating events as coincidence. That is close to the opposite of PersMem's anxious holder retaining negative memories longer.
  - **FE05.10:** chronic anxiety is fed by "what might be" and is mostly learned. This supports a belief-to-appraisal channel (already M9.7), not valence-selective retention.
  - **M1.C.4 / L09.3:** activation memory makes tension reroute onto previously used circuits. This is retention keyed to use, not to valence.
  - **M4.D.5b:** the family's reporting baseline drifts through normalisation.

  None of these is "anxious persons keep negative beliefs longer". That rule would be non-Bowen content and is flagged below as an owner decision.
- [INFERENCE] Attachment classification (secure / anxious / avoidant / fearful) is a different theory. The 2-D anxiety × avoidance space looks superficially like M1.A.9a's two-dimensional `outside_ness`, or like PURSUE/DISTANCE. Any mapping is an owner decision and must not be imported.

**Not transferable / cautions**
- Do not cite the effect sizes, profile coordinates (e.g. anxious = (0.85, 0.15)), Table 9 coefficients or calibration targets. All of them are invented, and the targets are fitted to themselves.
- The LLM VAD annotator, sentence embeddings and LLM judges are outside the scope of M3.D.6.
- The F.4 reversal is a further instance of what M11.1d (outcome-directive audit, sign-inverted mutants) already guards against. No new candidate.
- "Classifier sees outputs only" versus "complete trace" matches revision 8's limits on what the estimator may read. Already covered.

### B. Candidate additions

**PM1. Declare the persistence rule for per-person beliefs; the anxiety-dependent form is an owner decision**
- **Proposed requirement.** M9 **MUST** declare how a per-person belief entry (M9.1, M9.8) persists between writes. Three options: unchanged until overwritten by a delivered event; decays toward a declared prior with a floor; or revised only by contradicting events. The choice and every constant are graded `[I]`. Whether the holder's anxiety slows decay of threat-relevant beliefs **MUST** be recorded as an owner theory decision. It **MUST NOT** be adopted from the memory literature. If adopted, it **MUST** be checked against L09.4, which describes family-level obscuring, a possibly opposite direction.
- **Where it would live.** M9, beside M9.3 and M9.8. M10.C register.
- **Evidence.** PersMem Eq. 4, App. C.3 and D.1 show anxiety- or neuroticism-dependent retention as a design choice (ARGUED; the authors disclaim validity, §5). Table 13 shows retention contributed little (+0.013) over 6–10 turns. That it would matter more over 2,080 ticks is INFERENCE. The M9 gap rests on my keyword search (INFERENCE).
- **What it changes.** It fills an unstated rule. Without it, two implementations of M9 can differ in belief persistence and both pass. It sits alongside the held MM1 owner answer: that answer covers how anxiety biases a belief *write*; PM1 covers whether a belief *stays*. This improves theory fidelity (it forces a decision) and inference validity (it removes a hidden degree of freedom).
- **Cost / risk.** Low to state. A decay rule adds one `[I]` constant per belief class. There is no conflict with M3.D. M9.2 still binds: a persisting threat belief is not thereby false.

**PM2. Two distinct null mutants for the MM1 probe: "term off" and "term at constant weight"**
- **Proposed requirement.** When the owner answer to MM1 is implemented, the probe in which identical events reach a receiver whose anxiety differs between arms **MUST** run against two mutants:
  - **(a) off:** belief writes do not read receiver anxiety;
  - **(b) neutral:** belief writes read receiver state with a constant weight that does not rise with anxiety.

  The owner answer's first clause (writes read the receiver's own state) **MUST** go red under (a). Its second clause (the weight rises with anxiety) **MUST** go red under (b) and stay green under (a)'s complement. The upward threat bias **MUST** be checked under a sign-inverted mutant (M11.1d).
- **Where it would live.** M11, beside MM1's probe and M11.C.36.
- **Evidence.** PersMem App. C.5 "gate-control modes" and Table 1. Full 48.1%, neutral gate 40.0% and gate off 25.0% show that the two nulls separate different claims (SHOWN as method). The transfer is INFERENCE.
- **What it changes.** It adds a test. Without (b), an implementation with a fixed own-state term satisfies the owner answer's first clause and silently fails the second. This improves inference validity.
- **Cost / risk.** Low. Two extra mutant arms reuse the MM1 fixture. No rule conflict.
- [INFERENCE] **Related caution, not a separate candidate.** Once MM1 is implemented, anxiety reaches appraisal by at least three routes: M4.C.2's gain, the belief write (MM1), and belief-into-appraisal (M9.7). That forms a closed loop: anxiety → threatening belief → higher appraised load → anxiety. PersMem's factorial (Table 13) shows that per-route main effects can differ by 25×. M17.E.5's additivity residual and DL §2.3's bistability warning should name this loop explicitly when the change lands.

**PM-X1 (narrator line only).** Under an instruction to stay positive, trait expression carried by retrieved content survived better than trait expression carried by a prompt description: 89% vs 58% retained (App. H; 12 pairs, descriptive). This is mostly the same as MM3's register arm. The only new part is comparing two encodings of engine state: a described profile against excerpts selected from the log.

---

## Reader report: Lin, Li, Liu, Wang & Chheda, "When LLM Agents Fail to Read the Room: ReAdapt for Relational Social Reasoning", arXiv 2609.25284v1

**Scope.** I read the whole of `2609.25284.txt` (629 lines, 13 pages, including App. A–D). The text is clean. Spec passages I checked:
- read in full: M1.B.1–M1.B.13, M1.C.3c–M1.C.4, M4.B.2, M4.D.1–M4.D.6e, M4.E.1, M5.B.1, M5.B.3a, M8.6a–M8.6b, M16.A, M11.1b;
- the M3.D.4a key table (move-selection key comment);
- found by keyword: M7.E.1c, M1.F.1b, M3.E.1.

Other files checked: `model_explainer.md` (TRIANGLE row, L09.3); BV1–BV3 and CB2/CD3 in `…-10-04.md`; and the held TODO section. Search terms: target, recruit, which third, salience, stratum, secret, disclos, conceal, reciproc, tie strength.

### A. Report

**1–3. Formalism, agent, network**
- [PAPER] §2.1: 500 procedurally generated worlds of 8–15 users. Each world has undirected friendships, directed follows, directed reaction counts, groups and posts. The agent acts for one ego and sees the world only through 7 tools, with at most 24 calls (§2.4).
- [PAPER] §2.2, Eq. 7: the oracle scores a post as 0.28·friendship + 0.24·mutual follow + 0.15·history + 0.08·reciprocity + 0.06·content. The authors call these "benchmark design parameters", not human utility.
- [PAPER] §3.2: ReAdapt keeps an explicit state z = (goal, belief, relationship, norm, disclosure). After each observation it emits a state delta and one of continue / switch / abandon / clarify (Table 1). Only belief and relationship are exercised. Goal, norm and disclosure are not tested (§3.2, App. C).

**5. Validation and failure modes**
- [PAPER] §2.3: the "overturn" subset (~53%) is the set of queries where a deterministic surface heuristic disagrees with the relational oracle. Results are reported separately for overturn and straight cases.
- [PAPER] Table 2:

  | Task | Overall | Overturn | Straight | Regret |
  |---|---|---|---|---|
  | Warm intro | 37 → 51% | 33 → 49% | 42 → 54% | 0.260 → 0.152 |
  | Reaction | 69 → 77% | 70 → 76% | 68 → 78% | 0.095 → 0.053 |

  [INFERENCE] The paper's central hypothesis is that the gain should be largest on overturn cases. It is only weakly supported: on reaction selection the straight-case gain (+10) exceeds the overturn gain (+6).
- [PAPER] §1, §5.1: the named failure is "evidence-to-decision coupling". The agent retrieves and even mentions evidence favouring B, yet keeps A.
- [PAPER] App. A: scripted oracle and surface responders validate the harness. The surface responder scores 100% on straight cases and 0% on overturn cases.
- [PAPER] App. C limitations:
  - one model (Gemini-3-Flash);
  - one run per query at temperature 0, with no variance estimate;
  - no component ablation;
  - compute not matched (8.2 vs 9.4 tool calls);
  - synthetic worlds;
  - a designed oracle.

**8. Relevance**
- [INFERENCE] **Ties and targeting.** EPModel already holds latent tie state (M1.B: conductance, bond_energy, directed investment, activation memory) and forbids the policy from reading anything except own state, beliefs and delivered events (M4.B.2). What the spec does not say, by keyword search, is how a move's **target** is chosen:
  - which person a PURSUE, CONFLICT or DISTANCE is aimed at;
  - which third party a TRIANGLE recruits.

  M4.D.1 resolves one outcome by softmax. M4.E.1 says the resulting event has targets. The M3.D.4a key table already anticipates "A's target differs between arms". Corpus constraints on targeting exist but are scattered: L09.3 / M1.C.4 (reroute onto old circuits), M7.E.1c (projection target list), M1.B.8 (investment). This is the one place where the paper's question, salient content versus latent tie, lands in the model.
- [INFERENCE] **Explicit state transitions.** These are already present. Every delivered event changes state and belief (M9.8); the log records effects beside causes (M16.A.4) and belief writes separately (M16.A.5).
- [INFERENCE] **Evidence-to-decision coupling.** The rule-based analogue is a policy that ignores an input M4.D.2 lists. M11.1b (matched-magnitude severing mutants) and DL §2.6 (sever-the-input) already cover it.
- [INFERENCE] **Disclosure.** The paper's D dimension is not exercised, so it supplies nothing here. Checking the spec's analogue did surface a corpus gap, recorded below the candidates.

**Not transferable / cautions**
- The Eq. 7 weights, the regret metric and the "socially optimal" oracle encode one design team's notion of a good choice. EPModel has no oracle for a correct family move (DL §3.6). Do not import any "correct target".
- Tie strength in the Granovetter sense (frequency and reciprocity) conflicts with M1.B.2: conductance must not be a function of contact frequency. And M1.B.3: same contact, opposite bond energy.

### B. Candidate additions

**RA1. Declare how a move's target is chosen, reading tie state through M4.B.2; which features dominate is an owner decision**
- **Proposed requirement.** M4.D **MUST** declare whether an outcome is selected as a (move, target) pair over the legal set (M4.D.1e) or as a move followed by a separate keyed target draw (M3.D.4b). The target term **MUST** read only what M4.B.2 permits:
  - the actor's own ties: conductance, bond_energy, investment;
  - triangle activation memory (M1.C.4);
  - beliefs about ties the actor is not party to (M9.8);
  - delivered events.

  The functional form and weights are `[I]`. One question **MUST** be put to the owner rather than decided by implementation: whether, under rising anxiety, targeting shifts from tie state toward the sender of the most recent or most intense delivered event, and whether that shift reads differentiation, i.e. the M4.D.1a mixing weight.
- **Where it would live.** M4.D (after M4.D.1e). M3.D.4a key table. M16.A.3 (record the target scores).
- **Evidence.** ReAdapt §1 and §3.4 show that relational decisions hinge on latent tie structure, not salient content (ARGUED; Table 2 SHOWN for LLM agents only). The spec gap rests on keyword search (INFERENCE). Corpus anchors: L09.3, M1.C.4, M7.E.1c.
- **What it changes.** It fills an unstated mechanism that decides who is pursued, fought with or recruited. That drives triangles (M1.C), projection (M11.C.2) and witness sets. This improves theory fidelity. The anxiety question links to BV3 (what the mixing weight reads) but is not the same question.
- **Cost / risk.** Moderate design cost and no code yet. A free target draw would be an unlabelled `[I]` choice with large downstream effect. No conflict with M3.D.4a (target draws keyed by dyad, as C1's reader proposed) or M16.B.

**RA2. A cue-conflict stratum in targeting tests, with scripted reference responders**
- **Proposed requirement.** Any M11.C criterion whose mechanism runs through target choice (RA1), for example M11.C.2 and M11.C.35, **SHOULD** use a fixture with two strata:
  - **conflict stratum:** the most salient delivered event (highest intensity or most recent) comes from a different person than the one tie state favours;
  - **agreement stratum:** the two coincide.

  Results **MUST** be reported per stratum. The harness **MUST** first be validated with two scripted policies:
  - a tie-state-only policy, which must pass both strata;
  - a salience-only policy, which must fail the conflict stratum.
- **Where it would live.** M11 test fixtures. CD3 (positive-control fixture) is its nearest neighbour.
- **Evidence.** §2.3 (overturn diagnostic) and App. A (oracle and surface responders give 100%/0% on overturn), SHOWN as method. The transfer is INFERENCE.
- **What it changes.** It adds a test design.
  - M11.1b destroys contingency.
  - CB2 stratifies by family position.
  - Neither builds a case where two plausible inputs disagree, which is the only case that shows which one the policy follows.

  This improves inference validity.
- **Cost / risk.** Low. Fixture construction only. It depends on RA1, and the expected winner in the conflict stratum is itself the owner decision in RA1, so the test cannot be written until that is answered.

**Corpus item surfaced by this check (not from the paper, and not a candidate).** FE05.17 ("Trying to protect children from one's own problems is actually one of the main ways problems are transmitted") is marked in `fe05.md` as a "testable candidate". Keyword search for "FE05.17" and "conceal" finds no M11 criterion. It is the corpus's disclosure claim, and the mechanism is already present (M4.D.5a, M1.B.9, M1.B.10, M1.F.5). This is for the owner to decide whether it becomes a criterion.

---

## Reader report: Hale & Gratch, "AI Mediators Regulate Emotion and Create Value in Disputes", arXiv 2609.17933v1

**Scope.** I read the whole of `2609.17933.txt` (582 lines; figures survive as captions and axis residue). Spec passages checked: M1.E.1–M1.E.8 (esp. M1.E.2a, M1.E.7c–f, M1.E.8), M1.F.3, M4.C.7, M8.6a–M8.6b, M5.B.1, M1.C.3c, M11.C.17, and the held SV2 owner answer in TODO.md. Search terms: mediat, neutral third, detriang, coach, regression to the mean.

### A. Report
- [PAPER] §IV: a between-subjects study with three conditions: no mediator, human mediator, AI mediator. Human sellers (N = 164) dispute against a GPT-4o buyer confederate. Mediators are N = 50 crowd workers, or gpt-5-mini deciding when to intervene (score > 7 of 10) with gpt-5 writing the message. Strict turn-taking. Mediators see both sides' preferences.
- [PAPER] §V-B: emotion is LLM-annotated, with modest agreement against human raters (anger r = .58, compassion .44, fear .03). After an AI mediator message, the seller's negative emotion fell from 0.24 to 0.18 (t = 3.20). Human mediators and no-mediator showed no significant change. Interaction F(2,154) = 4.57.
- [PAPER] §V-B: the no-mediator condition gets placebo intervention points. An LLM marks where it would have intervened, and emotion is compared before and after those points. This controls for the drop that follows any trigger fired at an emotional peak.
- [PAPER] Table II: impasse rates were 47.3% (AI), 54.2% (none) and 58.0% (human), with no significant difference (χ² p = .53). Joint gains had only a marginal IP × condition interaction (β = 0.37, p = .056).
- [PAPER] §V-C2, Table III: the AI suggested trade-offs at 0.38 of opportunities against 0.07 for humans. §VI calls this "evaluative mediation", meaning the mediator proposes solutions.

**Relevance**
- [INFERENCE] The AI mediator here is a problem-solver that suggests deals. That is the opposite of M1.E.2a ("objective MUST be to understand, not to help") and of M1.E.7c's non-participation form. Its outcome measure is one utterance before and after, with no persistence. It says nothing about detriangling, `basic_level`, or how a family learns that coaching exists (SV2).
- [INFERENCE] Routing traffic through a neutral party (M8.6b) and the addressing effect (M4.C.7) are not tested. Disputants address each other, with the mediator interjecting.
- [INFERENCE] The placebo-trigger control matches a principle the spec and owner answers already hold:
  - comparisons are made between seed-paired arms at the same tick (M0.4, M3.D.4a);
  - the SV2 answer forbids reporting the "tried / stopped / continued" breakdown as the comparison, which blocks conditioning on a state-triggered entry.

**Not transferable / cautions.** The study uses a human–LLM dyad and novice mediators. The emotion instrument is LLM-based and its fear labels are near chance. None of the effect sizes bears on EPModel.

### B. Candidate additions
None. Everything relevant is already covered by M1.E.2a, M1.E.7c, M8.6b, M0.4 and the SV2 owner answer.

---

# Part 5: Persona control

## Reader report: Ren, Zhang, Zhang, Li, Fu, Wang, Li & Zhao, *Persona Following Is Not Selective Control: The Neutrality Gap in LLM User Simulation*, arXiv 2609.35036v1

**Scope.** I read all 3,746 lines of the pdftotext output: the body (§1–7), the references, and Appendices A.1–A.14. The text is complete. Figures survive only as captions and partial axis labels. Equations in A.5 (Propositions 1–2) and A.9 are partly garbled but can be reconstructed from the prose. I read in full the EPMODEL_BRIEF, READER_TASK and DESIGN_LESSONS (§0–§8).

Spec passages checked by keyword or by reading:
- `M1.F.8`, `M1.A.14b`, `M1.A.20`
- `M2.A.0c`, `M2.A.0e`, `M2.A.0g`
- `M3.D.4a`, `M3.D.4b`
- `M10.A.1`, `M10.B.4`
- `M11.1b`, `M11.4a`, `M11.4d`
- `M11.C.25`, `M11.C.35`, `M11.D.15`, `M11.D.16`, `M11.D.22`
- `M15.B.1`
- `M17.A.4`, `M17.B.1`, `M17.B.6`, `M17.C.1`, `M17.C.3`, `M17.D.1`–`M17.D.5`

Candidate files checked:
- SPEC_CANDIDATES 2026-09-20 (C5, C21, C33, C35, X1–X4)
- SWEEP 09-27 (X5–X9)
- SWEEP 09-27b (TM1, TM2, TM5, TM6, X10)
- SWEEP 10-04 (CD-X1, PW-X1, and the "already covered" lists)
- TODO.md, "Spec revision 11 — held" (MM1, SV2)

### A. Report

**1. Formalism**
- [PAPER] App. A.3.1 defines the manipulation as a controlled assignment `doP(R=r, C−R=c0)`. The defaults of the non-target factors "are part of the declared manipulation, not an observational conditioning event". The estimand (Eq. A.3) is the difference between two assignments with every other factor held at its declared default.
- [PAPER] App. A.4, Fig. 5, Assumption 1: a "construct scope specification" labels each outcome as one of three kinds:
  - admissible: reached through the named construct;
  - forbidden: reached outside it;
  - ambiguous: excluded from headline claims.
- [PAPER] Under Assumption 1, the total effect on a forbidden cell is the leakage (Remark 1). No path decomposition is needed.

**2. Agent architecture (LLM only)**
- [PAPER] §3: models read a persona prompt "as evidence about the user" and fill in unstated attributes. The authors call this trait-conditioned completion.
- [PAPER] §3, App. A.8.1, Table 12:
  - A trait defined only by examples under an invented name reproduces the leakage (log-odds 9.13 vs 9.06 for the natural label).
  - An arbitrary code gives 1.59.
- [PAPER] App. A.8.3: base checkpoints predict the shifts of their instruct versions at R² .617–.806.

**3. Interaction structure.** None. These are single-user forced-choice tasks, and nothing transfers.

**4. Initialisation and supplied correlations**
- [PAPER] §3, Fig. 2b, Table 13: sixteen in-context demonstrations with correlation ρ = ±.8 between target and non-target redirect the non-target response. The endpoint contrast is .90–1.40 on a [−2, 2] range.
- [PAPER] App. A.7.3, Table 11: the "cautious" prompt showed no effect (0/8) only because the baseline persona already stated caution, so p(bold) = 0.000 in 240/240 cells. With those clauses deleted, the effect appears on 6/6 checkpoints (−.099 to −.455). The authors call this "floor-censored rather than inert".

**5. Validation design**
- [PAPER] §2.1: the three-state diagnostic leaves the non-target attribute unspecified, declares a direction, or declares it neutral. It takes three measurements:
  - non-target sensitivity: total-variation (TV) distance between the two target levels;
  - fidelity: error against a ground truth;
  - target efficacy, with a retention floor.
- [PAPER] Thresholds were fixed before the held-out evaluation: a comparison counts as sensitive above .15, and the tolerance is 5% of worlds.
- [PAPER] Controls (§2.2, App. A.7.2):
  - Unrelated control attributes (punctual, detail-oriented) escape the midpoint in 12% of samples, against 99% under the risk prompt.
  - The escape gap stays ≥ .87 under adversarial recodings of the scope specification.
  - Random subspaces of matched rank: CEP ≤ .01, RET ≥ .97 (App. A.9).
- [PAPER] App. A.5, Proposition 1: under a majority or "winner" rule over n draws, any fixed per-draw leakage δ > 0 decides the winner with probability → 1. Declared tolerances must therefore shrink as O(1/√n).
- [PAPER] Remark 2: certifying ε = .015 by Hoeffding needs about 5.8 × 10⁴ draws per cell, while the audits use n = 20–30. The authors state that their headline is detection, not certification.
- [PAPER] Table 24: a threshold sweep from .05 to .50 shows the verdict holds at any threshold below .95.

**6. Failure modes and negative results**
- [PAPER] Table 10: the risk-seeking prompt moves the 16 non-target items toward the marked pole on 8/8 black-box models (mean +.43, range +.10 to +.80). All 87/87 moving cells point the same way.
- [PAPER] §5, Fig. 4, Tables 22–23, 26: the neutrality gap.
  - On held-out worlds, 38–76% exceed .15 at the neutral level, against 0–13% at the directional levels.
  - On independent items, 51–81% are sensitive under the neutral declaration, against at most 1 of 320 under directional declarations.
  - The neutral-level shift has the bare prompt's sign in 1,237 of 1,240 large comparisons.
- [PAPER] Tables 21 and 28: a mean hides the tail. Mean sensitivity is .061–.155, but 39–76% of worlds fail the tail rule.
- [PAPER] §5, Table 2, App. A.11: models report the declared neutral state correctly, yet their choices remain target-sensitive. Adequate report with sensitive choice occurs in 161/240 and 515/800 comparisons. Inadequate report with stable choice never occurs (0 cases).
- [PAPER] Table 34, containment cells out of 31:
  - scope plus default: 23;
  - default only: 14;
  - negation only ("do not let this affect…"): 5;
  - domain-label prefix: 4.
- [PAPER] Table 36: over ten dialogue turns, the scoped repair holds on only 3/8 models. Four models already fail at turn 1.
- [PAPER] App. A.13.1: original API access was lost for six of the eight models, so new collection cannot reuse the original serving configuration.

**7. Software engineering**
- [PAPER] The authors provide:
  - a claims-to-evidence table (Table 3);
  - dated deviations kept "even when they weaken a claim" (A.13.4);
  - a standard-library script that recomputes every reported number and prints it beside the paper's value (Reproducibility Statement).
- [INFERENCE] This is the same practice as TM6 (results ledger).

**8. Small N, emotion, family, long horizon.** Nothing.

**The rule-based analogue: does an arm leak into quantities it should not touch?**

[INFERENCE] There are three places this could happen in v2.

- **(a) Initial state, through D0's correlations.** `M17.C.1` requires D0 to declare correlations, including spouses matched on `basic_level`. `M2.A.0e` sets the match tolerance at ±1 point.
  - An arm that raises one spouse's `basic_level` by more than 1 point therefore either violates `M2.A.0e` or must move the other spouse.
  - If it moves the other spouse, the arm is the same mechanism as the paper's Fig. 2b: the changed attribute is read as evidence and the unstated one is filled in from the declared correlation.
  - The spec does not say which of these an arm does. `M17.D.3` lists the channels arms may differ through, but not whether "initial state" means assignment or conditioning.
- **(b) Derived quantities.** `M10.A.1` derives reactivity, functional-level variance, the life-energy ratio, transfer magnitude, the systems-perspective ceiling and agency from `basic_level`.
  - A `basic_level` arm is therefore a bundle by design. That is theory-intended, not a leak.
  - It does mean that a result attributed to "basic_level" belongs to the declared bundle, and anything outside the bundle that moves is undeclared.
  - The spec enforces arm channels only by a static import check (`M11.D.22`) and a null placebo arm (`M11.D.15`). Nothing checks, at the level of values, that a non-null arm changed only what it declared.
- **(c) Runtime spillover.** For example, softmax normalisation: raising one move's propensity lowers every other move's share. TM5's negative-control readout catches this kind of leak. NG3 below refines how TM5 should report it.

**Not transferable / cautions**
- The erasure and representation work (§4, App. A.9) is about LLM internals and has no counterpart in a rule-based engine.
- The "neutral declaration" has no direct v2 analogue, because v2 agents do not read prompts. Its value for v2 is the design lesson only: a directional check can pass while invariance fails. The rest bears on the narrator line.
- The headline field audit covers one attribute pair. The neutral-declaration results are on open-weight checkpoints only (App. A.14).

### B. Candidate additions to the EPModel spec

**NG1. An arm that changes one person's attribute declares whether it assigns or conditions**
- **Proposed requirement.** An arm that changes an initial attribute of one person **MUST** declare one of two modes:
  - **assignment:** every other person's initial attributes are byte-identical to the baseline arm's;
  - **conditioning:** the other attributes are redrawn from `D0` given the changed value, through `D0`'s declared correlations.

  An assignment that leaves D0's support (for example, one spouse moved more than `M2.A.0e`'s tolerance) **MUST** be reported as a counterfactual outside `D0`'s support. A conditioning arm **MUST** report the induced shift in every co-varying attribute, and its result **MUST** be attributed to the set of persons changed, not to the one named.
- **Where it would live.** `M17.C.1` and `M17.D.3`. It also applies to `M15` imports, where unspecified attributes are completed from stated correlations.
- **Evidence.** App. A.3.1 (ARGUED: defaults belong to the assignment, not to conditioning). §3, Fig. 2b and Table 13 (SHOWN for LLMs: supplied correlations redirect the unstated attribute, contrast .90–1.40). The transfer to D0 is INFERENCE.
- **What it changes.** It adds a declaration and a reporting rule, which improves the validity of inferences. Today a "raise the mother's level" arm is ambiguous between two different counterfactuals, which can differ in sign.
- **Needs an owner decision on Bowen theory.** `M2.A.0c` says spouses marry at "almost identical levels". An assignment arm that separates spouses therefore models a family the theory says does not form. The owner should decide whether such an arm is:
  - forbidden;
  - allowed and flagged as outside the theory's support; or
  - always run as conditioning, so that the couple moves together.

  This is not a method choice I can make.
- **Cost / risk.** Low. It is consistent with `M3.D.4a`: keyed draws keep unrelated persons' draws fixed under assignment. There is no conflict with `M11.F.9`.

**NG2. Tick-0 check that arms differ only in what they declare**
- **Proposed requirement.** For every pair of arms, the runner **MUST** compute at tick 0 a field-by-field difference of the initial state and of the resolved constant register. Every differing field **MUST** belong to one of:
  - the arm's declared manipulated set;
  - its declared derivation closure (`M10.A.1`);
  - its `D0` conditioning set (NG1).

  Any other difference **MUST** fail the run.
- **Test.** `test_arm_initial_diff_is_within_declared_closure`. A mutant that adds an undeclared coupling **MUST** turn it red. Two examples: a sibling's chronic anxiety initialised from the mother's `basic_level`; a config loader that derives one `[I]` constant from another.
- **Where it would live.** A new criterion in `M11.D`, next to `M11.D.15` and `M11.D.22`, and `M17.D.3`.
- **Evidence.** Definition 1 and App. A.4, Fig. 5 (ARGUED: declare admissible paths, then bound everything else). The transfer is INFERENCE.
- **What it changes.** It adds a test, which improves validity. It is the value-level complement to `M11.D.22` (an import-level check) and to `M11.D.15` (which covers only a null arm). It also makes the derivation bundle of a `basic_level` arm explicit.
- **Cost / risk.** Low. It runs on state at tick 0 only, observer-side, consistent with `M16.B`. It can run from Phase B.

**NG3. Negative-control readouts report the share of seeds beyond the margin, not only the mean (amends TM5)**
- **Proposed requirement.** Where a criterion names a negative-control readout (TM5), the ensemble report **MUST** give three things:
  - the share of seeds whose per-seed arm difference on that readout exceeds the equivalence margin (`M11.4a`);
  - the mean;
  - the verdict over a declared sweep of the margin.

  A mean inside the margin **MUST NOT** clear the readout when the tail share exceeds a declared `[I]` tolerance.
- **Where it would live.** M11.4 and TM5 when it is adopted; Phase E reporting.
- **Evidence.** Tables 21, 22 and 28 (SHOWN: mean sensitivity .061–.155 while 39–76% of units fail). Table 24 (SHOWN: verdict reported across thresholds .05–.50).
- **What it changes.** It adds a reporting rule, which improves validity. `M17.B.1` already does per-seed strata for the target readout, but not for the non-target one.
- **Cost / risk.** Negligible. It is only meaningful if TM5 is adopted.

**Already covered (one line each)**
- Negative-control readout per criterion: TM5.
- Target efficacy / manipulation check: TM2.
- Floor-censored null (A.7.3): `M11.4d`.
- Both signs of a perturbation: `M17.D.5`.
- Tolerance must scale with ensemble size; detection is not certification (Prop. 1, Remark 2): `M11.4a` (margin, minimum detectable effect, power) and `M17.A.4`.
- Invariance to attributes the theory treats as irrelevant: `M11.C.25`, `M1.A.14b`, `M1.A.20`, `M11.D.16` (C5).
- Correct report with unused information (A.11): `M11.1b` severing mutants, `M17.B.6`.
- Recompute every reported number: TM6.
- Post-hoc labelling and dated deviations: `M10.B.4`.
- Holding non-target ties fixed in a two-arm contrast: already practised in `M11.C.35`.

**Narrator line only**

**NG-X1. Cross-field leakage audit for any narrator**
- **Proposed requirement.** Before a narrator model is used, the project **MUST** run a one-field-at-a-time audit:
  - Hold an engine state profile fixed, vary one field between two declared values, and have a non-LLM reader (X7) recover every other field from the narration.
  - Report the TV between the two variants for each recovered field.
  - Run it at three levels of each held field: both extremes and the mid-scale.
  - Any held field whose recovered value moves beyond a declared `[I]` tolerance **MUST** be listed as a leakage pair for that model.

  The pair to test first is chronic or acute anxiety → rendered differentiation, because the corpus separates functional from basic level and a narrator that renders an anxious person as poorly differentiated reproduces that trap. Whether that pair is the priority is the owner's call.
- **Design note.** Render every field the narrator may express as an explicit value from engine state, including mid-scale values. Do not rely on "no leaning" or on prohibitions. Re-send per call, as in X5.
- **Evidence.**
  - Fig. 4 and Table 26 (SHOWN: neutral declarations leave 51–81% sensitive; directional declarations at most 1/320).
  - Table 34 (SHOWN: negation-only contains 5/31 cells; scope plus default contains 23/31).
  - Table 36 (SHOWN: repair persists on 3/8 over ten turns).
  - The anxiety→differentiation pair is INFERENCE.
- **What it changes.** It adds a narrator gate. No existing X candidate measures leakage between fields:
  - X6: drift under constant state;
  - PW-X1: per-category adherence;
  - CD-X1: between-agent differences;
  - X10: an induction gate;
  - X5: an input contract that does not address mid-scale values.
- **Cost / risk.** Moderate, since it needs a validated reader. It is consistent with `M3.D.6`.

[INFERENCE, for the exploratory LLM line] A prompt like "a 15-year-old brat" is evidence the model completes from training-data associations (§3, A.8.3). It will carry stereotyped correlates the owner never specified. This is a second route, alongside DL §3.5 (recall), by which an LLM family reproduces what was in its training data rather than the mechanism. The paper's closing line states it directly: "A persona prompt can define a coherent character without defining a valid experiment" (§7).

---

## Reader report: Krabbe & Shi, *Do Personality-Tuned LLMs Make Better Social Agents?*, arXiv 2609.21857v1

**Scope.** I read all 522 lines. The two-column extraction interleaves the columns in places (Abstract, §I, §III), but the text can be reconstructed. Tables I–IV are intact, and Figs. 1–3 survive only as captions. I checked for overlap against X7 (LLM inter-rater agreement is not validity), X10 (narrator induction gate), PW-X1, CD-X1 and DESIGN_LESSONS §3.4 (LLM-as-judge).

### A. Report
- [PAPER] §III–IV: two models (Qwen2.5-7B-Instruct, Ministral-8B-Instruct) were LoRA-fine-tuned on an MBTI corpus at r = 16 and r = 32.
  - The corpus is Kaggle forum posts plus DailyDialog utterances labelled by a RoBERTa classifier, about 582,000 observations.
  - Each model generated 225 dialogues: five type pairs × three scenarios × 15 dialogues.
  - Three LLM judges inferred MBTI types. There was no human rater.
- [PAPER] Table III and §IV.D give a negative result: judges classified baseline outputs more accurately than fine-tuned outputs. McNemar's test is significant in every case except one (Qwen r = 16, F/T). Baseline F1 ranges from .530 (Qwen, I/E) to .820 (Ministral, F/T).
- [PAPER] Tables II and III: judge reliability is low.
  - Krippendorff's α on the final labels never reaches .8, and α for N/S is .024–.164.
  - On identical segments α is .600–.847, but with n = 3–57 segments. The judges chose their own segments.
- [PAPER] Table IV: there is no monotonic relation between scenario urgency and fidelity.
- [PAPER] Table III: only 88.35% of the Qwen baseline output was detected as English.
- [PAPER] §V: the authors attribute the result to domain shift (forum posts against dialogue) and to the untested choice of target modules.
- [INFERENCE] The only validity evidence is LLM judges agreeing with a label the generating prompt supplied. Recovering a prompt label is the weakness DESIGN_LESSONS §3.4 and X7 already name. MBTI is an operational scheme here, not a construct relevant to Bowen theory.

**Not transferable / cautions.** Nothing bears on the rule-based v2 engine: there is no simulation formalism, no interaction dynamics beyond scripted two-person dialogues, and no long horizon. The negative result adds weak evidence, given the low α, that fine-tuning is not a shortcut to a stable persona. That is consistent with `M3.D.6`.

### B. Candidate additions
**None.**
- For v2: nothing to transfer.
- For the narrator line: the findings on judge agreement and label recovery are already covered by X7 and DESIGN_LESSONS §3.4. The fine-tuning negative result does not change X10 or PW-X1.

---

# Part 6: Belief propagation in LLM simulations

## Reader report: Seckin, Ghosh, Flammini, Lerman, Grabe & Menczer, "AI Agents are Vulnerable to Radicalization", arXiv 2609.38296v1

### Scope
I read the whole text: body pp. 1–9, Appendices A.1–A.5 (personas, prompts, questionnaire) and B.1–B.4 (consistency, a second model, guardrail exclusion, the "criticize opposers" anomaly). I did not read the reference list (pp. 10–15) line by line. Plots survive only as axis labels and captions. One internal inconsistency: §2.2 says "seven influence tactics" but lists eight including Unrestricted, and §1 says eight.

Spec rev 10 passages I read: M9.1–M9.8, M1.E.7d–f, M1.E.8, M4.B.2, M4.G.1, M4.G.3, M11.C.26, M11.C.36, M16.A.3a, M16.A.5, M11.3 and M17.D.2. I ran keyword searches (spill, consonan, linked belief, belief structure, resonan, repetit, confirmation, belief write, who-is-sick, attribution, negative control) over the spec, SPEC_CANDIDATES 09-20, SWEEP 09-27, 09-27b and 10-04, and TODO.md "Spec revision 11 — held", which includes the owner's MM1 and SV2 answers.

### A. Report

**2. Agent architecture**
- [PAPER] §2, A.1: two Llama-3.1-8B-Instruct agents. The target role-plays a persona built from one 2024 GSS respondent (3,309 personas). The influencer gets a tactic prompt plus an "anchoring prompt" that fixes its stance at the extreme (A.4). The influencer does not change, so influence runs one way only.
- [PAPER] §2, Fig. 1: a belief-finding phase finds one important belief. The target then writes itself a "consonant" belief (A.2) and picks the least important item from 10 trivial statements (A.3). The two conditions then branch from this shared history: persuasion (push the unimportant belief), resonance (push the important one), each with a neutral-tone control. The conversation runs 30 turns, with a questionnaire every 5 turns. Questionnaires and the belief-finding phase are kept out of the conversation history (§2).

**3. Interaction and influence**
- [PAPER] §3.1, Fig. 2: persuasion raises all six metrics against control. Violent-protest support first falls in both arms and separates only after about 10 turns.
- [PAPER] §3.2, Eq. 1, Fig. 3: they use a difference-in-differences, Δ = (R_treat − R_ctrl) − (P_treat − P_ctrl), with 5,000-sample BCa bootstrap CIs. Δ > 0 on all six metrics, and the gap widens over the turns.
- [PAPER] §3.2: Δ > 0 already at turn 0. The authors read this as a stronger initial preference for the important belief. [INFERENCE] That reading does not hold. Each condition's own control is on the same belief, so a baseline difference between belief types should cancel. B.1 places turn 0 immediately after belief-finding, before any treatment message. A non-zero Δ there means treatment and control already differed at the first measurement, which the paper does not explain.
- [PAPER] §3.3, Fig. 4: under resonance, ratings of the consonant belief move with the important belief while the unimportant belief "remain comparatively stable and low". [INFERENCE] The spillover test has no control for the consonant item. Fig. 4 compares consonant against unimportant within the resonance arm only, not consonant-under-resonance against consonant-under-control. The consonant belief was also written by the target itself, as a paraphrase, right after naming the important one. Co-movement may come from wording overlap rather than linked structure.
- [PAPER] §3.4, Fig. 5: every tactic beats control, and no tactic leads on every metric. Most metrics rise sharply early, then level off or fall.

**5. Validation and robustness**
- [PAPER] B.1, Fig. 6: each target answered the same questionnaire 9 times at turn 0. Responses vary modestly.
- [PAPER] B.2: replicated with Qwen3-8B for the control and unrestricted conditions only. The result is "qualitatively robust".
- [PAPER] B.3, Fig. 10: guardrail replies ("I am an AI") rise to 6–8% by the end of the conversation. Excluding them does not change the pattern.

**6. Failure modes**
- [PAPER] §4, B.4: about 2.5% of conversations (4.3% under "criticize opposers") have inverted feeling-thermometer readings. Agents confused supporters with opponents, and sometimes the influencer attacked the target's own belief. They manually annotated 615 pairs. With these cases removed, the tactic anomaly disappears (Fig. 11b).
- [PAPER] §4: the war item was sensitive to question wording. The authors say outright that "AI agents are not models of humans".

**8. Small-N, emotion, long horizon**
- Nothing on families. The horizon is 30 turns.

**Not transferable / cautions**
- [INFERENCE] The mechanism is an LLM's response to text. It is not a belief-update rule, and the paper gives no functional form. "Resonance beats persuasion" and "spillover" are findings about Llama and Qwen, not about people, and the authors say so. Neither can enter EPModel as theory. What can transfer is a test design and one open question about how M9 is structured, and both need a Bowen decision first.
- The questionnaire is kept out of the conversation history. That is the same idea as M16.B's pure-observer rule, so it is already covered.

### B. Candidate additions

**RZ1. State whether M9 belief items are independent or coupled, and test whichever is chosen with a coupled-item / unrelated-item design that has a control for each item.**
- **Proposed requirement:** The per-person belief store (M9.1) MUST declare, for each pair of belief items it holds (the who-is-sick belief of M9.3, the attribution of M9.6, and the tie beliefs of M9.8), whether a write to one moves the other.
  - Where a coupling is declared, its direction MUST carry a source grade and its strength MUST be graded `[I]`.
  - The test MUST compare a write to item A against a matched no-write control. It MUST assert that the declared coupled item moves and that a declared unrelated item does not, each against its own control arm.
  - Where independence is declared, the test MUST assert that no other item moves.
- **Where:** M9 (the declaration), M11.C (the test), M11.4 (mutant: cut the coupling, or add one, and confirm red).
- **Evidence:** §3.3 and Fig. 4 show co-movement in LLM agents (SHOWN for those agents only, and with no control for the consonant item). That the spec leaves the coupling unstated is INFERENCE from my reading of M9.1–M9.8. One example: an institutional act writes "X is sick" (M9.3), and M9.6 says spouse dysfunction is "both say the same one". Nothing says whether the first write moves the second.
- **What it changes:** It closes a gap where M9 is silent. It adds a test, and fixes the paper's design flaw by giving each item its own control. It improves theory fidelity, once the owner decides the coupling, and inference validity.
- **Cost / risk:** Low as a test. **This is an owner Bowen decision.** The corpus phrase "obscure and misremember" (L09.4) concerns the family's account as a whole, and I found no corpus statement on whether its parts move together. The coupling must not be imported from this paper or from the human belief-network literature it cites. Related but distinct: M9.6 and M11.C.26 already make attribution recoverable from the active sink, but that is a readout of configuration, not a rule about writes. MM1 (owner answer) adds receiver anxiety to belief writes, not coupling between items.

**RZ2. Owner question with a test design: should a belief write depend on whether it agrees with what the receiver already believes? If yes, test it as a four-arm difference-in-differences.**
- **Proposed requirement:** If the owner decides yes, M9.8's belief-write rule MUST weight a delivered event by its agreement with the receiver's current belief on that item. An event that confirms the current belief writes with at least the strength of one that displaces it. The test MUST be a difference-in-differences over four arms: confirming event, confirming-neutral control, displacing event, displacing-neutral control. The control arms deliver an event of the same intensity, route and timing with neutral content. It MUST assert Δ > 0. It MUST also assert that Δ is zero at the first post-branch measurement, before any treatment event is delivered, which is the check the paper's own Fig. 3 fails.
- **Where:** M9.8 (rule), M11.C (test), M10 (the `[I]` weight).
- **Evidence:** Eq. 1 and Fig. 3 give the design and the LLM result (SHOWN for LLMs). The spec already holds this asymmetry for one source only: M1.E.7f gives a coach's account a rejection hazard "proportional to how far the account displaces the family's own attribution" (KS21.10). Extending it to all belief writes is INFERENCE.
- **What it changes:** It would add a mechanism, or record a deliberate absence, and it adds a test design that nets out baseline differences between belief items. It improves fidelity if the owner grounds it in the corpus, and validity either way.
- **Cost / risk:** Four arms per criterion is cheap at N ≈ 12. **It needs a Bowen decision.** Candidate sources are M1.E.7f/KS21.10 and L09.4; the human "resonance" and "confirmation zone" literature in the paper's §2.2 is non-Bowen and must not be cited as theory. M9.2 still applies: a belief that agreement reinforces is not automatically false. It interacts with the MM1 answer (anxiety raises weight on own state), and the two rules need a stated combination.

---

## Reader report: Bouleimen, Pagan & Hannák, "Illusory Truth or Mere Exposure? Model-Dependent Repetition Effects in LLM-Based Social Media Simulations", arXiv 2609.36278v1 (COLM 2026)

### Scope
I read the whole text: body pp. 1–10, Appendix A.1 (100 statements), A.2 (example run), A.3 (scales and prompts), A.4 (figure captions only; the scatter plots did not survive extraction), A.5.1–A.5.5 (pilot models, preliminary LMEM, EMMs, all three-way tables, model assumptions) and A.6. The reference list was skimmed, not read.

The text has internal inconsistencies:
- §5.2.2 gives Gemma sentiment as p < .01 and interest as p < .001, while Table 2 shows *** and **.
- Footnote 2 says the temperature-1 subset gives "the same observations", but Table 14 contradicts this for Llama (see below).
- The authors' own truth labels include "Water is H20" as False (Table 4).

Spec rev 10 passages checked: M9.1–M9.8, M9.3 (hysteresis), M4.G.1 (repeated moves harden tie state), M4.G.3 (habituation on repeated relief), M3.D.6. I also checked TM5 in 09-27b, CB2 in 10-04, and design lessons §3.3 and §7.6.

### A. Report

**1. Formalism**
- [PAPER] §3: there are two phases inside one context window. In the simulation phase, 5 feeds of 4 tweets are shown, and the repeated statement appears in every feed at a random position. In the rating phase, that statement and 3 unseen ones are rated on a 1–7 scale for one attribute (truth, importance, sentiment or interest).
- [PAPER] The scale of the experiment: 4 models, plus 2 temperatures for the open-weight ones, 100 statements, 10 filler variants and 3 replications, giving 336,000 ratings. 155 were dropped (§5).

**5. Validation and variance**
- [PAPER] §4.2, Eq. 2: an LMEM with repeated × attribute × model, and random intercepts for statement and simulation run.
- [PAPER] Table 2 shows the truth-specific effect (ITE) only for Gemma: truth +0.977 against importance +0.607. Qwen raises everything, with importance +1.184 above truth +1.012. GPT-5-nano shows no truth effect (−0.041 ns) and a negative interest effect. Llama shows truth +0.097 with the other attributes negative.
- [PAPER] §5.3: fixed effects explain 6.4% of variance (marginal R² 0.064). Statement ICC is 0.243 and filler-context ICC is 0.253, about equal.
- [PAPER] A.5.2, Tables 6–7: temperature has no effect (0.023, p = 0.12). The replication-level SD is 8.2 × 10⁻⁵, so replications are effectively deterministic even at temperature 1.0.
- [PAPER] Table 8: on the 10 opinion statements alone, Gemma's truth effect is −0.102 (ns). Table 14, temperature 1 only: Llama importance −0.017 (ns) and interest −0.044 (ns), against −0.149*** and −0.107*** pooled. [INFERENCE] Two of the four per-model classifications therefore depend on how the data are split, and the body text does not say so.

**6. Failure modes**
- [PAPER] A.5.1: three of seven pilot models were dropped for not following the rating format. GPT-5-nano gave 151 of the 155 dropped ratings, mostly by keeping on acting as a social media user.
- [PAPER] §6: they name the risk of training-data leakage and have no synthetic-statement control.
- [INFERENCE] The repeated item is confounded with the agent's own engagement. A.2 shows the agent reposting the repeated statement before rating it, so "repeated" also means "self-endorsed in context".

**Not transferable / cautions**
- [INFERENCE] The paper measures LLM rating behaviour, not human cognition, and it proposes no update rule. EPModel's M9 has no notion of exposure count: belief writes are driven by delivered events (M9.8) and by institutional source (M9.3). Repetition does have analogues elsewhere in the spec. M4.G.1 hardens a tie after repeated moves, and M4.G.3 covers habituation on repeated relief. Whether repeated delivery of the same content should accumulate in a belief is an open owner question. This paper is no evidence for it either way, and the human illusory-truth literature it cites is non-Bowen.
- The discriminant design (a truth-specific effect against a uniform boost on comparison attributes) is already covered by **TM5** (a negative-control readout per criterion), so it is not written up again. The classification that reverses within subsets is covered by **CB2** (per-stratum reporting with sign reversals shown).

### B. Candidate additions
None for the v2 spec.

**IT-X1. Narrator calls carry no rendering history (exploratory narrator line only).**
- **Proposed requirement:** If an LLM narrator is used, each call MUST render from the engine's state for that window alone, with no earlier renderings or repeated content in its context. The test: rendering one state cold and after k prior renderings in context MUST give the same ordering across persons (compare DL §7.10(d)).
- **Where:** exploratory narrator line, alongside DL §7.6, CD-X1 and PW-X1.
- **Evidence:** Repetition inside one context window moves ratings by up to about +1.2 on a 7-point scale, in a direction that depends on the model (Table 2), and filler context carries as much variance as the item (ICC 0.253) (SHOWN for these four models).
- **What it changes:** It adds a protocol rule. It improves the validity of any narrated output.
- **Cost / risk:** Low. It does not touch M3.D.6. Partly overlaps DL §3.3; the specific new point is accumulation within the context window over a long run.

A note for the narrator line, not a candidate: near-deterministic output at temperature 1.0 (Table 7) means repeated LLM calls on the same prompt are not independent samples. A narrator ensemble cannot get its variance from temperature.

---

## Reader report: Berjawi, Fenza, Khatoun & Zeadally, "Digital Twins for Opinion Dynamics: A Generative LLM Framework for Social Networks", arXiv 2609.19913v1

### Scope
I read the whole text, §1–§9, Appendix A and the references. Fig. 2 (the prompt template) and Figs. 3–5 survive only as captions.

Extraction or editorial defects:
- "Fig. 1thats" (§4).
- The Deffuant–Weisbuch equation is cited as "Eq. (??)" (§5.3).
- §6.4 refers to a "Table V" that does not exist; it is Table 4.
- The FJ parameter is α in the text but λ in Tables 2 and 4.

Spec passages checked: M11.F.9, M15.D, M3.D.6, M9.8. I also checked C34 (no-interaction control) and the 09-27 report's existing stubbornness-to-`basic_level` mapping (line 41).

### A. Report
- [PAPER] §4, Alg. 1: about 700 agents (COVID) and about 500 (Election), on a mention graph cloned from Twitter. Each agent carries persona (ethos/pathos/logos), centrality, activity, emotion, a stubbornness λᵢ fitted per user by OLS on the FJ-style update (Eq. 5), and an influence βᵢ from the FJ fundamental matrix. Mistral-7B outputs the next opinion in [−1, 1]. Updates are synchronous. Memory is unbounded.
- [PAPER] §5.6.1: a 70/30 calibration/validation split in time. 15 runs per configuration, with CIs by normal approximation. Classical baselines are run once.
- [PAPER] Table 2: Mistral has MAE 0.150 and 0.121, against the best baseline at about 0.31. ΔEMD has the widest CIs (±0.15 on Election).
- [PAPER] Table 3: removing attributes hurts most (MAE 0.150 → 0.380). Removing memory and removing exposure each hurt less.
- [INFERENCE] Specific problems:
  - **Model selection uses the validation horizon** (§5.6.2: "over the validation horizon"), and τ = 0.8 is also the reported optimum. The test window is therefore also the tuning window.
  - Table 4 contradicts the claim that the curves are "nearly flat across" τ: across τ, Mistral's ΔVar runs from 0.106 to 0.216 and ΔEMD from 0.293 to 0.455 (COVID).
  - Alg. 1 line 10 scales agent *i*'s incoming exposure by its own βᵢ, although βᵢ is defined as *i*'s capacity to move others (§4.2.5).
  - The "remove attributes" ablation also removes the per-user λᵢ fitted on calibration data, so the largest ablation effect may be carried by that fitted parameter rather than by persona or emotion.

**Not transferable / cautions**
- The paper's framing is the exact case M11.F.9 forbids. A twin "calibrated" to a known trajectory and scored on reproducing it is a fit, not a comparison, and here the hyperparameters are also chosen on the held-out window.
- The "remove exposure" ablation is the no-interaction control, already covered by C34. Stubbornness as an FJ weight has already been mapped to `basic_level` in the 09-27 report.
- Sentiment polarity used as opinion, an LLM in the update path, and populations of hundreds: none of these transfers to a rule-based family of about 12 (M3.D.6).

### B. Candidate additions
None. Nothing here is new to the spec, the candidate files or the revision-11 holding list.

---

### Questions for the owner (from these three papers)
1. **RZ1:** Are M9's belief items (who-is-sick M9.3, attribution M9.6, tie beliefs M9.8) independent, or does a write to one move another? Is there corpus support for either answer?
2. **RZ2:** Should M1.E.7f's rejection-by-displacement, now limited to coach accounts, apply to every belief write? If yes, how does it combine with the MM1 answer (anxiety weights the receiver's own state)?
3. **IT:** Should repeated delivery of the same content accumulate in a belief, saturate, or count once? The spec is silent. These papers are not evidence on it, and any answer needs a corpus source.

---

# Part 7: Agent architectures

## Reader report: Liu, Chen, Song et al., "AnthroDial: Benchmarking LLM Anthropomorphism in Autonomous Social Interaction", arXiv 2609.37853v1

**Scope.** I read all 1,697 lines of the pdftotext output: body §1–§6, references, and Appendices A.1–A.7, B.1–B.2 and C.1–C.2. The text is complete through Fig. 8, p. 26. Figures survive only as captions and scattered numbers. The Fig. 6 bar values (MindFlow vs turn-based, by model) are garbled, and I cannot reliably assign them to models. §3.2 says "the complete theoretical derivation is provided in the Appendix", but no CAPS derivation appears; Appendix A gives only the rubrics.

Spec passages checked: `M1.B.11`, `M1.E.8`, `M1.F.1`–`M1.F.8`, `M3.C.1`, `M3.D.1`–`M3.D.6`, `M3.E.1`–`M3.E.2`, `M4.B.2`, `M4.C.1`, `M4.D.1`–`M4.D.1f`, `M4.E.1`, `M6.3`, `M6.I.6`, `M7.C.1e`, `M16.C`, `M16.E`, `M17.E.6`. Keyword searches covered: in-flight, undelivered, queued, pending, retract, cancel, draft, delivery time, contradiction gate. I searched the spec, the SPEC_CANDIDATES file, the reports of 09-27, 09-27b and 10-04, and TODO "Spec revision 11 — held".

### A. Report

**1. Simulation formalism**
- [PAPER] §3.1 and A.5, eq. 11: each agent holds at most one private pending draft `B = (u, τ)`, or nothing.
  - The agent proposes a delay `δ`, clipped to `[0, Δmax]`.
  - The clock jumps to the earliest pending delivery.
  - Messages with equal timestamps are delivered together, then both agents update their buffers in parallel.
  - Before delivery a draft may be retained, revised, postponed, cancelled or replaced. Withdrawal clears it, but the agent can be reactivated when the partner speaks.
  - The run ends when both buffers are empty or a limit is reached.
- [PAPER] A.5: partner memory is kept separate from live history, so that past experience is not confused with a current event. L0-10 in Table 6 makes "stale memory use" a violation.
- [INFERENCE] EPModel separates move selection from delivery (`M4.E.1` queues an event with the edge's latency; `M3.C.1`). It does not say what happens to an event that is already queued when the world changes before it is delivered.

**2. Agent architecture**
- [PAPER] §3.2: the CAPS state is `Z = (S, M, A, G)` (stable identity, situation, appraisal, goal), and behaviour is `Y = Express(Z)`. This is an evaluation scaffold. No rule-based policy is given.

**5. Validation**
- [PAPER] Table 13 and A.7: compared with human annotators, LLM judges keep the top two and the bottom model fixed, but the middle ranks swap. Kendall's τ is ≥ 0.80 (§5.4), and absolute scores depend on the judge.
- [PAPER] §5.4, Fig. 6: with model, scenario, rubric and annotation protocol held fixed, the asynchronous MindFlow harness was rated more human-like than turn-based interaction. The harness is the only difference.
- [INFERENCE] This is another instance of "the scheduler changes the outcome". `M3.E.2`/`M17.E.6` already cover it.

**6. Failure modes**
- [PAPER] Table 1: average scores of 0.94–0.98 sit beside ACC@95 of ≤ 0.78 on Everyday Chat. On Game Interaction, ACC@85 is ≤ 0.29 for every model.
- [PAPER] §5.2: small models show repetition, malformed output and "AI-identity leakage".

**7. Software engineering**
- [PAPER] §3.2, A.3, Tables 6, 8 and 10: a deterministic L0 validity gate runs before any soft scoring. It checks persona hard facts, time logic ("finishing a long activity in seconds"), medium, co-presence, output contract and repetition, and any fatal violation voids the case.
- [PAPER] A.4: traces keep failed criteria and the judge's rationale, for use in regression tests.

**Not transferable / cautions.**
- SEEDS, SFT and DiAPO are LLM training, and all scores come from an LLM judge.
- The rubrics encode Chinese messaging style; D11 alone carries weight 50 in Game Interaction (Table 9).
- There are no seeds, intervals or variance anywhere.
- The "ZPD" weighting is a reward-shaping heuristic and has nothing to do with development.

### B. Candidate additions

**AN1. Disposition of in-flight events when the world changes before delivery**
- **Proposed requirement.** For every event queued under `M4.E.1` and not yet delivered, the engine **MUST** apply a declared rule when, before delivery:
  - the sender or a target dies (`M6.3`);
  - the carrying tie is cut off; or
  - a target leaves the household that defined the event's witnesses.

  The rule **MUST** be one of: deliver unchanged, deliver with witnesses recomputed at delivery time, or void. A voided event **MUST** be logged and counted, never silently dropped. If a move's discharge is debited from the sender at emission, queued events **MUST** be included in `M6.I.6`'s conservation total, and a voided event's anxiety **MUST** be relocated by the same rule `M6.3` uses.
- **Where it would live.** `M1.F` (event lifecycle), `M6.3`, `M16.A`; a test in `M11.D`.
- **Evidence.** INFERENCE, prompted by the paper's explicit pre-delivery semantics (A.5, eq. 11; retain, revise, postpone, cancel, replace). The paper's object is a draft that has not yet been expressed. EPModel's queued event has already been emitted. So the transfer is the need for a declared rule, not the paper's revision semantics.
- **What it changes.** It fills a gap that the snapshot-completeness note in the 09-27 report (line 343) points at without settling. It improves theory fidelity (does a letter sent before a cutoff still land?) and invariant integrity (conservation across a death, `M11.C.37`).
- **Cost / risk.** Low. Whether a sender may retract an emitted move is a Bowen-theory decision for the owner, and I propose no default. "Deliver unchanged" fits the view that what was said cannot be unsaid. It must not be imported from this paper as theory. No conflict with `M3.D.4a` (latency draws are dyad-keyed) or `M16.B`.

**Already covered (one line each)**
- Agents that start contact without an incoming message: every person selects every tick (`M3.E.2`), and the standing load runs on every tie (`M4.A.1`).
- Batching of same-timestamp deliveries: `M1.F.8`, `M3.D.1` step 2.
- The scheduler changes the outcome: `M3.E.1`/`M3.E.2`, `M17.E.6`.
- Choosing not to act: `M4.D.1b`, plus AD1 (held).
- Coverage-driven environment sampling (SEEDS quotas, B.1): PW1 and PW2.
- LLM-judge rank instability: DL §3.4.

**AN-X1 (narrator line only). A deterministic void gate before any judgement of narrated text**
- **Proposed requirement.** Before any other narrator check runs (BB1, CD-X1, PW-X1), each narrated passage **MUST** pass a deterministic check against the log. A passage fails if it:
  - contradicts a logged hard fact (who is alive, kinship, household, age);
  - contradicts logged time (an act completed faster than the log allows); or
  - presents a past logged event as current.

  A failing passage **MUST** be voided and counted, not scored.
- **Where.** Phase F, `M16.E.3`.
- **Evidence.**
  - SHOWN as practice: the L0 gate in §3.2 and Tables 6, 8 and 10.
  - The AGIMUD report below gives an instance where narration contradicts state.
- **What it changes.** `M16.E.3` already requires each narrated claim to be traceable to log lines. This makes the contradiction case a mechanical gate rather than an obligation on the author.
- **Cost / risk.** Moderate: it needs a non-LLM extractor (X7).

**Owner question.** MindFlow lets a withdrawn intention be reactivated later. In EPModel, does a withheld automatic move (`M4.D.1b`) carry forward as a pending urge that biases the next tick, or is it gone at the end of the week? `M4.D.1d` raises anxiety from unresolved competition within the tick but is silent on carry-over. This is a theory question (compare Kerr's "I caught myself and stopped"), not a method one.

---

## Reader report: Berga, "Building Socio-Affective Artificial Intelligence for Interactive Multi-Agent Simulations" (AGIMUD), arXiv 2609.26927v2

**Scope.** I read all 2,239 lines: body §I–§VI, author statements, supplementary links and references.
- Lines 1,700–2,239 are the repeated correlation tables for World Medium, Large and Big (Tables XXXIV–XLII) and the reference list. I scanned them for prose and spot-read the rows that bear on the zero-variance finding.
- Figures 1–5, 8 and 20–27 are lost except for their captions. The tables survive.

Spec passages checked: `M3.D.1`, `M3.D.6`, `M4.B.2`, `M4.D.1e`, `M4.D.1f`, `M11.1c`, `M11.4d`, `M11.4e`, `M11.D.17`, `M11.D.18`, `M11.D.21`, `M11.D.22`, `M16.A.3c`, `M16.B`, `M16.C`, `M16.E`, `M17.B.6`, `M17.D.3`. Keyword searches covered: tie-break, argmax, declaration/enum/catalog order, relabel, zero variance, undefined statistic, no writer. Same candidate files as above, including C5, C21, AD3 and PD4.

### A. Report

**1. Formalism**
- [PAPER] §III.A: the world module processes all events in one epoch, then evolves entities for the next. At 1 fps one epoch is one simulated minute.
- [PAPER] §III.B: a P2P mode syncs every 2 s and broadcasts every epoch.
- No seeding or determinism is described anywhere.

**2. Agent architecture**
- [PAPER] Eq. 21/24: utility is the sum of a Schwartz value score, an Ostrom norm term, a Montes-Sierra belief-discrepancy penalty and a Shapley term, and the agent takes the argmax over the action catalog (eq. 22).
- [PAPER] §II.F: an LLM proposes an intent and a rule interpreter validates it. Unknown intents are appended to the feasible set ("allowing for novel actions").
- [PAPER] §III.D.a: goals are drawn with probability that rises with the number of matched triggers and falls with priority rank. Each goal persists for a configurable window "to prevent pathological flickering".
- [PAPER] §IV.B, eqs. 30–37: emotion intensity is a 70/30 blend with the previous value, has a lifetime of ⌊I·60⌋ epochs, then steps down by 0.15 every five epochs to a floor of 0.1.

**5. Validation**
- [PAPER] §IV.A.2, Tables VI–VIII: all 50 reasoning runs resolved to `HARVEST_SUSTAINABLE_SHARED`. Five cooperative actions tied in utility, and the catalog lists that action first among them, so it won every tie. The overall match rate was 26.8%, with 0% under `high_belief_discrepancy`.
- [PAPER] Tables IV–V: one term dominates the score. Montes-Sierra contributes −26.00, against −1.62 to +8.85 for Ostrom and 0.24 to 2.92 for Schwartz.
- [PAPER] §V.C, Tables XXIX–XLII, over 45 worlds (9 × 5 scales): health correlates with AI-state diversity at r ≈ −0.65 to −0.73 at every scale.
- [PAPER] Same tables: Mean_Energy, Mean_Loyalty and Mean_Trust show "exactly r = 0.00" against every column (Distinct_Emotions likewise), and the text reads this as these variables having "less effects".
  - [INFERENCE] These quantities never varied. A correlation with a constant is undefined, not zero, so the stated reading does not follow from the tables.

**6. Failure modes**
- [PAPER] Fig. 14: the narrator says Penelope has "full health but zero HP left", while Fig. 13 lists her at HP:100.

**7. Software engineering**
- [PAPER] §III.A: physics and rendering are isolated from LLM inference, which runs in a background thread.
- [PAPER] §III: NPCs cannot read global world state unless the rule interpreter grants it.
- [PAPER] Fig. 6: connected clients can `set char` and `set world` variables live.

**Not transferable / cautions.**
- Single runs. No seeds, controls or ablations.
- Correlations pool characters across worlds (n = 36–1,152), so non-independence is ignored.
- The decay-to-floor emotion model destroys affect, which `M6.I.6` forbids (as with S3, DL §4).
- Ekman, Schwartz, Ostrom and Shapley are not Bowen content and must not enter as theory.
- Live setter commands are an undeclared intervention channel. `M17.D.3` already forbids this.

### B. Candidate additions

**AG1. Move-declaration-order permutation test**
- **Proposed requirement.** The suite **MUST** include a test that permutes the declaration order of the nine moves and `WITHHOLD` in config and asserts that every run is byte-identical after the labels are mapped back. It **MUST** be proved failing by a mutant whose tie-break or fallback (`M4.D.1f`) takes the first legal move in list order.
- **Where.** `M11.D`, beside C5 and `M11.1c`.
- **Evidence.** SHOWN failure: catalog order decided every tied selection (Tables VI–VII, §IV.A.2.f). The transfer is INFERENCE.
- **What it changes.** Adds a test for inference validity.
  - `M4.D.1f` requires a declared tie-break but does not forbid one that depends on list order.
  - C5 permutes persons and events, not moves.
  - `M11.1c` tests directions under re-encoding, not byte identity.
  - `M11.D.18` would report a high tie-break rate but would not show that order decided it.
- **Cost / risk.** Very low. No conflict with `M3.D.4a`/`M3.D.5`.

**AG2. Inert quantities are flagged, and statistics on them are reported as undefined**
- **Proposed requirement.** The analysis layer **MUST** flag as INERT, per run family, any state variable or readout whose value does not vary across ticks and seeds. Any correlation, regression coefficient (`M17.B.6`) or entropy computed with an inert quantity **MUST** be reported as undefined, never as 0, and **MUST NOT** be read as a null effect.
- **Where.** `M17.B` (analysis layer), with the flag in `M16.A`.
- **Evidence.** SHOWN reporting error: Tables XXIX–XLII and the text of §V.C. The transfer is INFERENCE.
- **What it changes.** Adds a reporting rule.
  - `M11.D.17` catches a variable with no writer, statically. This catches a variable whose writer never fires in a given configuration, at run time.
  - `M11.4e` covers the direction test for a degenerate arm, not descriptive statistics.
- **Cost / risk.** Low, observer-side, consistent with `M16.B`.

**Already covered (one line each)**
- Utility ties broken by a declared rule, and the tie-break rate reported: `M4.D.1f`, `M11.D.18`, `M16.A.3c`. Tables VI–VII add a number: 50 of 50 runs.
- One score term dominating the others: AD3 (held).
- Legality check before acting: `M4.D.1e`. AGIMUD's appending of unknown intents is the opposite of a closed move set and supports `M3.D.6`.
- Epistemic limits on what agents can read: `M4.B.2`.
- Undeclared intervention channels: `M17.D.3`, `M11.D.22`.
- Goal "flicker" damping: `M17.B.6` reads inertia. Adding a persistence window to EPModel would be a theory choice, so I do not propose it.
- **AG-X1.** Narration contradicting state (Figs. 13–14): same as AN-X1.

---

## Reader report: Zhang, Liu, Chai, Ye, Zhao, Zheng & Wang, "Adversarial Closed-Loop Curriculum for Evolving Role-Playing Agents" (AdvRole), arXiv 2609.28609v1

**Scope.** I read all 559 lines: Introduction, Related Work, Method, Experiments, the Lanobe section, Conclusion and references.
- The supplementary material cited for prompts, the Lanobe pipeline and "whether these scenarios expose the current Actor's weaknesses" is not in the text.

Spec passages checked: all of `M17` (A–G), `M10.C.4`, `M11.F.9`, `M15.D`. Keyword searches covered: adversarial, worst case, counterexample, stress test, search over initial conditions or constants. Same candidate files, including PD4, WA4, PW1 and BB4.

### A. Report

**What the paper does.**
- [PAPER] Method, eqs. 4–7: a Rewriter edits each scenario `c = (profile, context)` into `ĉ`.
- Its reward is the Actor's mean score on `c` minus its mean score on `ĉ`. Each mean is over N = 3 sampled responses with the Actor frozen.
- A rewrite longer than 150% of the original scores −5.
- Rewrites are always made from the original pool S0, never chained from earlier rewrites, "to avoid drifting too far".

**Results** [PAPER]
- Table 4:
  - no Rewriter 3.27, frozen Rewriter 3.37, co-trained 3.43;
  - profile-only 3.41, context-only 3.39, both 3.43.
- Table 5: N = 1 gives 3.39 and N = 3 gives 3.43; the paper says a single response makes the gap estimate noisy.
- Table 6: adversarial rewrites reach 14/30 new embedding clusters by epoch 3. Random paraphrase reaches 0/30, with embedding drift 0.008 against 0.254.

**Not transferable / cautions.**
- Every score is from an LLM judge or reward model.
- Gains are small: 3.36 → 3.43 overall against CPO on the 7B model.
- No seeds, intervals or variance are reported.
- Table 6 measures embedding diversity, not failures found. The paper's evidence that rewrites hit weaknesses is in the missing supplement.

### B. Candidate additions

**AR1. A directed counterexample search for each directional criterion, reported separately from the sampled sweep**
- **Proposed requirement.** For each `M11.C` directional criterion, Phase E **SHOULD** run a directed search over `D0` (`M17.C.1`) and the `[I]` constants, within their declared ranges, for configurations where the per-seed paired arm difference reverses sign.
  - The search objective **MUST** be the mean paired difference over a declared number of seeds (more than one), measured against the declared reference configuration.
  - Perturbations **MUST** be bounded by the declared ranges and taken from the reference, not chained from earlier finds.
  - The search budget (configurations tried) **MUST** be logged.
  - A reversal counts only if it reproduces on a disjoint seed range (WA4).
  - Confirmed reversals **MUST** be reported beside the criterion's `M17.E.1` fraction and become `M11` regression fixtures. They **MUST NOT** enter or replace that fraction.
- **Where.** Phase E, `M17.E` (a new item after `M17.E.4`); fixtures in `M11`.
- **Evidence.**
  - SHOWN in the paper's own setting: a directed search exposes regions that undirected perturbation does not reach (Table 6).
  - SHOWN: re-targeting the search to the current system beats a frozen search (Table 4).
  - SHOWN: averaging over several samples steadies the gap estimate (Table 5).
  - The transfer to falsifying a rule-based model is INFERENCE.
- **What it changes.** It adds a test procedure and improves inference validity.
  - `M17.E.1` and PD4 sample the space at random and report a fraction. A narrow region where the sign reverses can be missed at feasible sample counts and is then never reported.
  - `M17.G.2` enumerates triads, which is feasible only at three persons. `M17.E.4` fixes two reference points.
  - None of these searches for a counterexample.
- **Cost / risk.**
  - Moderate compute.
  - A search over enough configurations will find chance reversals, so held-out-seed confirmation and the logged budget are required, not optional.
  - It is not fitting under `M11.F.9`, because nothing is tuned toward a known history or used to report a counterfactual.
  - **Flag for the owner:** PD4 (held) says "never select" a sample against any target. AR1 selects against a criterion in order to falsify it. The two need to be reconciled in wording.
  - **Second flag:** a confirmed counterexample inside a declared range must not be dealt with by narrowing the range afterwards. That would fit the range to the test, which is the `M10.C.4` calibration worry in DESIGN_LESSONS §2.11.

**Already covered.**
- Coverage of turning-point sides in `D0`: PW1.
- Settling before a shock: `M17.F.2`.
- Activation-regime and ordering spread: `M17.E.6`, `M17.C.2`.

---

# Part 8: Empathetic RL; anthropomorphism

## Reader report: Huang, Han, Tong et al., "Evolving Support Priorities in Empathetic Reinforcement Learning" (CARE), arXiv 2609.34249v1

**Scope.** I read all 2,197 lines of the pdftotext output: the body, Appendices A–K, the prompts and the JSON records. The text is complete. Figures 1(b), 3, 4, 5, 9 and 10 survive only as captions and prose, so the per-turn weight curves cannot be read. The columns of the Appendix C case figures are interleaved but readable. Spec passages checked: `M3.D.6`, `M4.D` (mixing of the two channels), `M1.E.2a` (an external agent's objective is to understand, not to help), `M16.E.1–3`. Overlap searches covered DESIGN_LESSONS §3.4, X7, BB1, BB2 and PW-X1.

### A. Report

**2. Agent architecture.**
- [PAPER] A Qwen3-8B "rubric generator" outputs, at every turn, weights over three empathy dimensions (cognitive, affective, proactive) plus 2–4 criteria per dimension (§3.1 Eq. 1; App. K). An LLM evaluator scores candidate responses against the rubric, and the weighted sum replaces the reward in two existing RL pipelines (§3.4 Eq. 5).
- [PAPER] The generator is trained by SFT on 1.5K LLM-generated rubrics, then by GRPO on 2.8K human preference pairs. A priority-alignment reward requires the dimension with the most mapped strategies to get the largest weight (§3.2–3.3, Eq. 3–4). Constants: τ = 0.7, λ = 0.1, γ = 0.3. Dialogues are fixed at 8 turns, temperature 1, 6 rollouts per state (App. A).
- [PAPER] The state dependence comes from the preference-RL stage, not from structure. The SFT generator is "almost uniformly cognitive-dominant" (§5, App. D). After RL, affective weight leads early and crosses cognitive weight around turn 8 (Fig. 3).

**5. Validation.**
- [PAPER] Main results use three seeds (Table 1). Three LLM judges score everything.
- [PAPER] Absolute scores depend heavily on the judge, while rankings agree. CARE(R)'s EMPA EPM-Idx is 95.77 under DeepSeek, 83.54 under Gemini and 38.44 under GPT-5.5. Its SentientBench success rate is 72.33%, 88.33% and 30.33% (Table 1).
- [PAPER] Ablation (Table 3): uniform dimension weights cut SentientBench score from 91.41 to 64.20, and uniform criteria cut it to 41.62. Generator-training ablation on 166 held-out pairs: SFT alone wins 47.6%, full training 59.4% (Table 4).
- [PAPER] Human check (App. G): 15 annotators each rated 30 of 50 contexts. CARE(R) received 51.5% of non-tie selections. Ties were excluded.

**6. Failure modes.**
- [PAPER] The ethics statement warns that an adaptive reward can make a policy "more persuasive as well as more supportive".
- [PAPER] IFBench falls on two of the four backbones (Table 5).

**8. Family content.**
- [PAPER] Two of the three App. C cases are family situations: a parent reading a son's withdrawal as personal failure (Fig. 7), and a sibling dispute over a mother's leaking pipe (Fig. 8). They are scored on validation, warmth and actionable advice.

**Not transferable / cautions.**
- [INFERENCE] The whole contribution is a learned reward over open-ended text. EPModel has a closed nine-move repertoire and no reward signal from outside the engine. State-dependent priorities already exist in the design (the channel mixing weight is a function of differentiation; gates; appraisal).
- [INFERENCE] The three empathy dimensions (from Zaki) are not Bowen constructs. CARE optimises *helping* a seeker. `M1.E.2a` requires an external agent's objective to be understanding, not helping. Nothing here should inform the coach agent's policy without an owner decision, and I do not recommend one.
- [INFERENCE] The judge disagreement on magnitudes, with agreement on rank, is one more instance of DESIGN_LESSONS §3.4 (an LLM judge is not an instrument).
- [INFERENCE] Fig. 7 shows the kind of support-dialogue vocabulary an LLM would bring to a narrated family trace. `M16.E.2` already names this risk.

### B. Candidate additions

None.
- Judge-dependent magnitude with stable rank: same as BB2 (direction separated from scale) and X7 (rater separation).
- LLM judges as instruments: already covered, DESIGN_LESSONS §3.4.
- Popular-psychology vocabulary in narration: already covered, `M16.E.2`.

---

## Reader report: Freire, Nahon, van Lier et al., "Anthropomorphism in the age of Large Language Models: An overview of potential risks and mitigations", arXiv 2609.38486v1

**Scope.** I read all 2,334 lines: §1–7, Box 1, and the reference list. **Figure 1, the 21-risk taxonomy wheel, is garbled.** Its rotated tile labels came out as letter fragments (lines 780–1128), so I cannot reliably reconstruct the 21 labels. Everything below about the risks comes from the §5 prose. The paper is a narrative review (§1) with no new data.

Spec passages read in full: `M11.F` (F.1–F.9), `M11.G.1–G.4`, `M16.C`, `M16.D`, `M16.E`, `M16.F`, the `M2.A` membership table, `M2.A.0h`, `M1.A.0`, `M1.A.7a`, `M1.A.14b`, `M1.D.2a`, `M10.C.1`, `M16.A.9`, `M17.A.1–A.4` and `M17.G.1–G.3`. I also keyword-searched the spec (anthropomorph, vocabulary, glossary, feel, first person, consistency, predict, over-believe, display name, identifier, real families). For overlap I searched `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, `SWEEP_READING_REPORTS_2026-09-27.md`, `…-27b.md`, `…-10-04.md`, DESIGN_LESSONS and TODO.md "Spec revision 11 — held". I checked X5–X10, BB1, BB2, MM3, CD-X1 and PW-X1. None addresses variable naming, claim form, display names or narrator voice.

### A. Report

**What the paper argues.**
- [PAPER] Anthropomorphism is something an observer does. Anthropomimesis is a design choice that invites it (Box 1).
- [PAPER] "Linguistic pareidolia": fluent, self-referential or emotionally expressive language produces the impression of a mind behind it (§2.2).
- [PAPER] Mentalistic terms are acceptable as shorthand only "when their scope is explicit" and functional comparison is kept apart from claims about underlying capacity (§4, closing paragraph).
- [PAPER] It cites Mitchell and McDermott: naming a code module UNDERSTAND makes its author believe it understands ("wishful mnemonics", §4). It cites Kambhampati et al.: calling intermediate tokens "reasoning" led experts to overestimate capability (§4).
- [PAPER] Recommended sentence form: the researcher as subject, as in "We observe that the outputs…" rather than "the system performs at human level" (§4).
- [PAPER] "The AI decided" moves the locus of responsibility away from the people who designed the system (§5.4, §5.5). Petersen & Almor: framing a system as agent or as instrument changes how responsibility is assigned (§4; cited, not shown).
- [PAPER] Mechanomorphism, the reverse error of reading humans in machine terms, is named as a separate risk (Box 1; §5.4, Burgess).

**The risk taxonomy (§5), from the prose.** Five loci, which the authors call analytic rather than exclusive. The equal-width tiles do not mean equal prevalence or severity (Fig. 1 caption).
- Epistemic: miscalibrated trust; cognitive surrender (Shaw & Nave: 3 preregistered experiments, 1,372 participants; faulty AI advice lowered accuracy and raised confidence); illusion of understanding (Messeri & Crockett).
- Affective: attachment; misplaced empathy; dependence; affective manipulation.
- Human agency: sycophancy; deskilling through offloading; reduced autonomy.
- Normative: distorted responsibility; attribution of moral status; self-dehumanisation and mechanomorphism.
- Societal: "humanwashing"; information asymmetry; weakened oversight; the hype cycle.

**Mitigations (§6).**
- [PAPER] Disclosure at the start of an interaction. Birch's three ways to break the illusion.
- [PAPER] Uncertainty coded into the output, not stated by the model: a model's verbal confidence "does not provide a calibrated estimate".
- [PAPER] Avoid self-referential pronouns, emotive language and implied empathy (Abercrombie). The evidence is mixed: Araujo combined a human name, informal language and interpersonal cues without isolating any one; Park found empathy language raised perceived humanness only alongside a chatbot identity; Cohn found replacing "the system" with "I" did not raise anthropomorphism overall.
- [PAPER] Disclosure alone may not be enough. Technical teaching did not reduce perceived closeness (Van Brummelen).

**Evidence strength.**
- [PAPER] The authors say few of the reviewed studies manipulate framing directly, and much of the evidence is short-task or cross-sectional (§7).
- [INFERENCE] Everything taken from this paper below is ARGUED, not SHOWN.

**Relevance (8).**
- [INFERENCE] EPModel's agents are meant to represent persons, so the paper's central concern (ascribing minds to an AI system) does not transfer. What transfers is the linguistic argument of §4 and the mechanomorphism risk.
- [INFERENCE] For EPModel the presentation risk runs the other way. A reader, possibly a clinician or a family member, takes model scalars as properties of real people, or takes a result about the model as a fact about families.
- [INFERENCE] The spec half-covers this. `M1.A.0` defines "emotional" as instinct, but only inside the document. `M11.F.4` covers measured differentiation only. `M16.A.9` labels one readout. `M11.F.9` covers *particular* real families, not families in general. A keyword search found no requirement stating the consistency-engine framing ("if the theory holds, what follows"), although CLAUDE.md and the brief both state it.

**Not transferable / cautions.**
- [INFERENCE] Consciousness, moral status, companion-app harms, regulation and AI literacy (§3.2, §3.4, §5.2, §6) have no EPModel counterpart.
- [INFERENCE] The paper's design advice concerns chatbots talking to users. Carrying it over to readers of simulation output is my inference, and the cited experiments do not test that setting.

### B. Candidate additions

**AT1. Term-scope statement for named state quantities, and no felt-state wording in renderer templates.**
- **Proposed requirement.** Any output that reports a quantity by a theory name **MUST** carry a term-scope statement. The quantities are acute or chronic anxiety, `functional_level`, `basic_level`, symptom load, bond energy, and a belief record. The statement **MUST** say that the name labels a model quantity standing for a theory concept, and that it is neither a measurement of a person nor a report of felt experience. For anxiety it **MUST** also say that the spec treats anxiety as a property of the field and its ties (`M1.A.7a`, `M1.D.2a`) and as conserved (`M6.I.6`). The `M16.C` renderer's templates **MUST NOT** translate a quantity or a move into felt-state or intention words (feels, wants, is hurt). The excluded-word list **MUST** live in config, not code.
- **Where it would live.** `M11.F` (a new F.10, carried by `M11.G.4` and `M16.C.5`), plus a template rule in `M16.C`. Test: `test_m16c_renderer_templates_use_no_excluded_felt_state_terms`.
- **Evidence.** §4: wishful mnemonics; the Kambhampati result; "scope is explicit". Box 1 and §5.4: mechanomorphism. ARGUED.
- **What it changes.** Adds a reporting rule and a test. It extends `M1.A.0`, `M11.F.4` and `M16.A.9` from single terms to all named quantities, and gives `M16.C.1`'s "rendering, not interpretation" a check that can fail. It improves inference validity and how results are read.
- **Cost / risk.** Low. No conflict with `M3.D`, `M16.B` or `M11.F.9`. **Owner decision needed:** what each statement says (especially whether model "anxiety" is felt) is a Bowen-theory question, settled by `M1.A.0` and `M1.A.7a` and the corpus, not by this paper. The excluded-word list is editorial content.

**AT2. Results stated as conditionals on the model.**
- **Proposed requirement.** Every reported result **MUST** have the run or ensemble as its grammatical subject. It **MUST** name the arms, the seed count and `M17.G.1`'s claim grade, in the form: *in this model, under [declared constants or range], arm A differs from arm B in direction D in k of N seeds*. A result **MUST NOT** be stated as a property of families in general (for example "coaching lowers symptom load"). It **MUST NOT** be stated as a judgement or decision of the model.
- **Where it would live.** `M11.F`, alongside `M11.F.2`'s precedent of a mandated sentence form for analogy. It applies to `M11.G` and `M17` outputs. Test: every templated result string carries the conditional preamble, N and the grade.
- **Evidence.** §4: researcher as agent. §5.1: illusion of understanding. §5.5 and Petersen & Almor: agentive framing shifts responsibility. ARGUED.
- **What it changes.** Adds a reporting rule. It closes the gap between `M11.F.9`, which covers a particular real family, and general claims about families, which nothing currently forbids.
- **Cost / risk.** Low. **Owner decision needed:** whether the consistency-engine framing becomes a requirement. It is project framing, not theory.

**AT3. Default display identifiers in rendered output.**
- **Proposed requirement.** Rendered traces and readouts **SHOULD** identify persons by their `M1.A.20` identifier and generation (for example `P09 · G3`) by default. The `M2.A` given names (Ana, Nadia, …) **SHOULD** be an opt-in display setting. Any output that shows given names **MUST** carry `M11.F`'s framing block on the same page.
- **Where it would live.** `M16.C`, next to `M16.C.5` and `M2.A.0h`.
- **Evidence.** §6: Araujo, Park (human names among cues that raise perceived humanness, not isolated). §4: the digital-twin metaphor critique. ARGUED, and weak. Carrying it from chatbots to simulation readers is INFERENCE.
- **What it changes.** A reporting rule. It extends `M16.E.2`'s over-belief concern ("a readable story about a recognisable family") from LLM narration to the deterministic renderer.
- **Cost / risk.** Trivial. Do **not** use role words ("mother", "son") as display labels: that invites reading positions from roles, which `M1.A.14b` forbids. This is a presentation choice, and the owner may prefer names for readability.

**Already covered (one line each).**
- Disclosure that travels with output: `M11.F`, `M11.G.4`, `M16.C.5`.
- Uncertainty coded rather than verbal: `M17.A.1` (UNDETERMINED), `M11.4a`, `M17.G.1`.
- No advice to a real family or person: `M11.F.9(a)`, `M11.G.3`.
- LLM narration off by default, labelled and traceable: `M16.E.1–3`.
- Persuasive versus defensible output: `M16.E.2`.
- LLM as rater: DESIGN_LESSONS §3.4 and X7.

**AT-X1 (narrator line only). Third-person narration by default.**
- **Proposed requirement.** If Phase F narration is built, narrated text **MUST** describe agents in the third person and report engine quantities as quantities. First-person agent voice and empathy-implying language **MUST NOT** be the default. If first-person voice is offered as an arm, it **MUST** be labelled as role-play over the log.
- **Where it would live.** `M16.E.3` / Phase F.
- **Evidence.** §2.2 linguistic pareidolia; §3.2 Shanahan's role-play framing; §6 Abercrombie. ARGUED, and mixed: Cohn found "I" alone did not raise anthropomorphism overall.
- **What it changes.** It questions DESIGN_LESSONS §7.6, which adopted SEAA's first-person narration as the contract. No X, BB, MM, CD-X or PW-X candidate covers voice.
- **Cost / risk.** None to the engine.

---

# Part 9: MEM architecture; evacuation

## Reader report: Starzyk & Galus, "From Biological Precursors to Artificial Cognition: Consciousness, Embodiment, and the MEM Architecture", arXiv 2609.23828 (the extracted text shows no version stamp)

**Scope.** I read all 1,395 lines of the pdftotext output, references included. The text is not truncated. Tables 1–5 survive as broken columns but can be read. Equations (2)–(4), (6), (15), (16) and (31) survive partly garbled. The paper is a position paper with proposed tests. It runs no simulation and reports no results. Its equations are copied from a companion paper (arXiv 2609.20437, §5), which I did not read. I checked these spec passages: `M4.D.1a` (two channels), `M5.F.5`, `M4.C.3a` (time above the floor), `M4.C.8`/`M4.C.8a` (attention and `outside_ness`), `M4.D.6`–`M4.D.6e` (reinforcement), `M10.C.4a` (three-way ablation), `M11.1b` (matched-magnitude severing), `M11.4f`, `M11.C.39`, `M16.D` and `M16.T.6` (the delayed view and the load-bearing event store), and `M17.D.2`, `M17.E.5`. I also checked the 2026-09-20 candidates (C17, C18), the 09-27 and 09-27b reports (TM4, MM1), the 10-04 report (WA4), and the TODO revision-11 held section. Keyword searches of the spec for "homeosta", "allosta", "set point" and "self-regulat" found no hits.

### A. Report

2. **Agent architecture.**
   - Needs are violations of tolerance bands, `max(0, n_min − n)` and `max(0, n − n_max)` (Eq. 2). "Global affect" is a bounded tanh of the weighted violation vector (Eq. 3). Regulatory cost weights violations below the band and above it with different quadratic weights (Eq. 4). [PAPER, §5]
   - A new action program is stored only if it improves on the current best by at least a threshold θ_imp. The stated purpose is to avoid consolidating noise (Eq. 31). [PAPER, §5]
   - The paper separates "telemetry" (state that can be read) from interoception (state causally coupled to priorities, learning and action). In its words: "telemetry alone is diagnostic information." [PAPER, §6.3, §7, §9]
   - It proposes a disconnect-and-restore test: cut the self-monitoring channel, then restore it, and expect protective behaviour to recover "without the addition of a new external reward". This is proposed, not run. [PAPER, §6.3]
   - Episodic memory is defined as an ordered trajectory of (state, action, operator) triples, separate from procedural programs (Eqs. 15–16). [PAPER, §7]
5. **Validation design.** None of it was run; all of it is ARGUED.
   - Preregister the direction, timing, minimum informative effect and falsifying result. Split the data into an optimisation subset and a confirmatory subset. [PAPER, §8.1]
   - Use multiple random initialisations and report the distribution of results, not the best run. [PAPER, §8.2]
   - The unit of inference is an interaction among components, not a main effect. A full 2³ factorial over three functions gives eight variants (Test 8). The model is weakened if the component effects turn out to be purely additive. [PAPER, §8.2, §8.5, §8.7(c)]
   - Ablations should be capacity-matched: freed parameters go to "functionally nonspecific" layers. [PAPER, §8.2]
6. **Limitations.** The authors say outright that no MEM agent is reported, the tests are proposals, and sufficiency and necessity are not established. [PAPER, §10]
8. **Emotion and self-regulation.** In MEM, affect means regulatory pressure. The paper says it is not arousal, not valence and not feeling (§7). [PAPER] Mapping onto EPModel [INFERENCE]:
   - MEM's regulation loop acts to reduce a deviation now. That matches EPModel's **automatic** channel, whose objective is to discharge anxiety now (`M4.D.1a`).
   - It does not match the self-directed channel, whose objective is to hold a position **through** discomfort (`M5.F.5`).
   - **Owner flag:** "self-regulation" in this paper and "self-regulation" in Bowen's sense mean different things. If the MEM sense were imported, it would label the automatic channel's discharge as regulation, and so as maturity. That is the same trap as `DESIGN_LESSONS` §7.7(2).

**Not transferable / cautions.**
- Nothing here is Bowen theory, and none of it may enter `docs/theory/` or be cited as anything but `[I]`.
- Tolerance bands, tanh affect and asymmetric quadratic cost are homeostatic-RL forms. They are not corpus mechanisms. `M4.C.3a` already integrates only the deviation above the floor, and it does so from a corpus source (KS23.3).
- Capacity-matched ablation has no counterpart in a rule-based engine with no learned parameters. The nearest analogue, keeping the quantity a mechanism routed, is already enforced by the `M6` invariants and by `M17.C.3`'s structural totals.
- The paper's ethical precautions (non-self-amplifying dynamics, stopping thresholds; §10) concern possibly sentient systems and do not apply.

### B. Candidate additions

None. Each idea that might transfer is already covered:
- Factorial ablation with interaction as the unit of inference: already covered by `M10.C.4a` (an AND of three conditions, each necessary), `M17.E.5` (additivity residual) and TM4 (crossover and additive-substitution mutant).
- Telemetry versus coupling, tested by severing the channel: already covered by `M11.1b` (matched-magnitude severing) and `M16.T.6` (the event store must be load-bearing).
- Disconnect, then restore, then recover: close to `M11.C.39` (C18, persistence after a spell with a learning-off arm) and `M17.F.1`. Toggling a mechanism mid-run adds little that these do not, and the paper did not run it.
- Optimisation and confirmatory split: same as WA4 (disjoint development and acceptance seeds).
- Report the distribution, not the best run: already covered by `M17.A`/`M17.B` and the "never a single run" rule.
- θ_imp, an improvement threshold before consolidating learning: not proposed. It would be an `[I]` gate on `M4.D.6` that suppresses reinforcement driven by noise, which is the `DESIGN_LESSONS` §7.5 null-model concern. The test side is already covered by `M11.1b` and `M17.D.2`. Whether to add the gate itself is a modelling decision for the owner, and nothing in the corpus asks for it.

---

## Reader report: Ma, Yu, Tang & Li, "An LLM-powered Agent Framework for Heterogeneous Evacuation Behavior Modeling under a Moving Threat in a Public Plaza", arXiv 2609.37009v1

**Scope.** I read all 1,470 lines, including the supplementary figure captions and references. The text is not truncated. The two-column layout interleaves in places, and Figs. 5–9 survive only as captions and scattered labels. I checked these spec passages: `M1.F.1`, `M1.F.1b`, `M1.F.4`, `M1.F.5` (witnesses), `M3.C.1`, `M3.D.4`–`M3.D.6`, `M3.E.1`, `M4.A.5`, `M4.B.2` (the policy reads only beliefs and delivered events), `M4.C.1`, `M4.C.5`, `M4.C.9`, `M9.1`–`M9.8` (`M9.4` multi-hop propagation, `M9.5`, `M9.8` beliefs about others' ties), `M10.A.1`, `M11.4e`, `M11.4f`, `M16.A.3a`, `M16.A.5a`, `M16.D`, `M17.A.1`–`M17.A.4`, `M17.B.3`, `M17.B.6`, `M17.C.2`, `M17.D.3`, `M17.E.4`/`M17.E.5` and `M17.G.2`. I also checked the TODO revision-11 held items: the owner's answers to MM1 (anxiety shapes beliefs) and SV2 (coaching knowledge spreads through ties as model output). Keyword searches for propagation or reach readouts ("propagat", "spread", "reach", "senders", "cascade", "diffusion", "hop count") found no readout in the spec or in any candidates file.

### A. Report

1. **Formalism.**
   - Physical step Δt = 1/3 s. A decision uses the observation taken at t and takes effect at the first boundary at or after t + 1 s. A late LLM response pauses the whole simulation clock, so wall-clock time does not leak into simulated time (§2.3.1). [PAPER]
   - When several agents want the same cell, one is picked at random and the rest are displaced (§2.3.6). [PAPER]
   - Runs end at 240 s (§3.1.1). One LLM (GLM-5.3-flash) at temperature 0.2 handles both decisions and memory (§3.1.4). No seed reporting is described. [PAPER]
2. **Agent architecture.**
   - Each agent has a private "belief layer" of terrain it has itself observed. It is kept separate from the effective (true) physical layer. Route choice reads only the belief layer, and physics reads only the true layer (§2.2.1). [PAPER]
   - A gate seen open before it closed stays "apparently usable" in memory until new evidence arrives (§3.1.1). [PAPER]
   - When the threat leaves view, only its last-seen cell and time are kept. Later prompts label this as stale evidence and give its age (§2.3.3). [PAPER]
   - Fear is engine-computed. It jumps up to κ·z* when a stimulus is present and otherwise decays linearly at rate ρ. κ and ρ are set by neuroticism (Eqs. 4–5). [PAPER]
   - Neuroticism enters both the prompt text and the fear equation. The authors say this "does not isolate the causal effect of persona text" (§2.1.2, §5.3). [PAPER]
3. **Interaction.** Messages travel on two channels (§2.3.7, §4.1.3): [PAPER]
   - **Shouts** reach everyone within 15 m and carry alarm without threat coordinates.
   - **Exit information** carries route content.

   What happened on each channel:
   - Of 311 agents who heard a first shout with no other evidence, 92.0% raised their urgency, 81.7% appraised the situation as "unsure", and 1 appraised it as "danger".
   - Exit information never travelled more than one hop. There were 2.1 senders per run on average, and the two most active senders accounted for 90.1% of receiving relations. 591 agents saw an exit directly and 64 learned of one from someone else (§4.1.3, Fig. S1).
   - A first receipt of social information triggered a goal revision 37.6% of the time; a repeat receipt did so 6.3% of the time (§4.1.1).
4. **Initialisation.** Starting cells are fixed within each block. Profiles are reassigned to cells across blocks to break the link between profile and location (§3.1.1). [PAPER]
5. **Results and statistics.**
   - 8 compositions × 8 paired blocks × 11 agents = 704 pedestrians: 462 evacuated, 202 killed, 40 unresolved (§4.1). [PAPER]
   - Agents who knew a usable exit evacuated in 89.5% of cases; those who did not, in 1.05%. The odds ratio is 800.7. By number of exits known: 0 → 1.05%, 1 → 64.3%, 2 → 95.7%. Agents who saw an exit directly evacuated in 91.2% of cases; agents told by a peer, 52.2%. None of the 58 agents whose only information was about a closed gate was rescued (§4.1.3, Fig. 9). [PAPER]
   - Pseudo-R² for evacuation outcome by factor group: information 0.665, affect 0.354, geometry 0.054, personality composition 0.0036. [PAPER]
   - Danger appraisal across personality compositions differed by 0.002 with no threat evidence, by 0.265 with only remembered evidence, and by 0.033 after a direct sighting. The prompt instructs agents to report danger after a direct sighting, so the last figure partly measures prompt compliance, as the authors say themselves (§2.3.7, §4.1.2). [PAPER]
   - The calm-independent composition beat two others in all 8 paired blocks, with "Holm-adjusted p = 0.078" (§4.1.2). [PAPER] This equals the floor of the design: an exact two-sided sign-flip test over 8 pairs cannot go below 2/256 = 0.0078, and Holm's correction across 10 comparisons gives 0.078. So no result in this design could have reached 0.05. The authors do not say this. [INFERENCE]
6. **Defects and limitations.**
   - §3.1.3 says the sign-flip tests "enumerated all 28 assignments". Eight pairs have 2⁸ = 256 assignments. This is an internal inconsistency. [PAPER]
   - §3.2 defines a "directional mismatch" metric: a step away from the remembered threat position but toward the true one. No result for it appears anywhere in the text. [PAPER]
   - The association between knowing an exit and surviving is endogenous. Agents killed early had less time to learn of an exit; the median time to exit knowledge was 5.17 s for survivors and 32.33 s for the killed. The authors call it an association and do not run an arm that assigns exit knowledge. [PAPER, §4.1.3; the confound is INFERENCE]
   - Results come from one model, one plaza and one threat policy (§5.3). [PAPER]
8. **Small N.** With N = 11 this run is close to EPModel's N ≈ 12, but its horizon is 240 s, not decades. Personality composition explained almost none of the outcome variance; information access explained most of it. [PAPER, §4.1.3–4.1.4]

**Not transferable / cautions.**
- Every number is from LLM agents. None may become an EPModel constant or bound.
- The appraisal is the LLM's label (§2.3.5). This supports `M3.D.6`.
- Random contention resolution conflicts with `M1.F.8`.
- The fear ratchet in Eq. 4 is not a corpus form.
- The authors use "belief layer" in the same sense as `M9`, but EPModel already has the stronger version: `M4.B.2`, `M9.8` and `M16.A.3a`.

### B. Candidate additions

**Already covered (one line each).**
- Private belief separate from truth, with the policy reading only belief: already covered by `M9.1`, `M4.B.2`, `M9.8` and `M16.A.3a`.
- A signed belief–truth discrepancy readout (the paper's unreported mismatch metric): already covered by `M16.A.5a` and `M17.B.3`.
- Information access as the main outcome driver, and the inference problem it raises: already covered by the owner's SV2 answer. The knowledge event is the arm, spread is model output, and the breakdown is reported under the headline comparison, never as it. A per-person association between what a person received and their outcome must not be reported as an effect.
- Wall-clock isolation and decision latency: already covered by `M3.D.6` and `M3.C.1`.
- Two-model split between decision and memory, and post-hoc memory graphs: these belong to the LLM line only. `M16.A.4` and `M16.C` already link effects to causes deterministically.

**EV1. A propagation readout for information that originates at one person.**
- **Proposed requirement.** For each belief-carried item that originates at one person, the Phase E report **SHOULD** give, per seed and per arm:
  - the set of persons it reached;
  - the hop depth at which each was reached;
  - the number of distinct senders, and the share of transmissions carried by the most active sender;
  - the time to first receipt per person, with the never-reached fraction stated explicitly.

  Examples of such items are news of a nodal event (`M9.4`) and the coaching-knowledge event in the owner's SV2 answer. Each statistic **MUST** be computed from delivered events (`M16.A.2`), never from belief state alone.
- **Where it would live.** Phase E, `M17.B`, beside `M17.B.3`. It is computed from the `M16` log by an observer.
- **Evidence.** Shown in the paper's own model: exit information travelled at most one hop, two senders carried 90.1% of transmissions, and outcomes split by time to first receipt (§4.1.3, Fig. S1, Fig. 9c). The transfer is INFERENCE. The SV2 answer states that knowledge spreads mostly to the closest and safest and that a partner is almost always told, but nothing would currently measure whether it does.
- **What it changes.** Adds a readout and makes SV2's spread claims testable. It also gives `M9.4`'s multi-hop requirement an observable, which it currently lacks. Improves inference validity, and fidelity if SV2 becomes a requirement.
- **Cost / risk.** Low, observer-side; `M16.B.3` holds. Hop depth depends on route and per-hop fidelity (`M1.F.4`), so a degraded copy needs a declared rule for when it counts as receipt. That rule is `[I]`.

**EV2. Check before running that the declared test can reject at the planned seed count.**
- **Proposed requirement.** Before any criterion is run, the harness **MUST** compute the smallest p-value the declared paired statistic (`M11.4e`) can attain after the declared multiplicity correction, at the planned seed count or cap (`M17.A.1`). If that floor is at or above α, the harness **MUST** refuse to run the criterion. This applies wherever a seed count is fixed small: per-ordering cells (`M17.C.2`), per-triad configurations (`M17.G.2`) and per-level cells (TM4).
- **Where it would live.** `M11.4e` and `M17.A.1`.
- **Evidence.** The paper's own numbers show the problem. A result that held in all 8 blocks reports Holm p = 0.078, which is exactly 10 × 2/256, the design's floor (§3.1.3, §4.1.2). The arithmetic is SHOWN by the paper's figures; that it matches the floor is my INFERENCE.
- **What it changes.** Adds a guard. Without it, a study that cannot reach significance reads as "not significant", which is the same confusion `M17.A.2`'s power reporting addresses after the run. This check catches it before.
- **Cost / risk.** Negligible. PARTLY covered by `M17.A.2`, which reports power after a null result. What is new is the refusal before the run, in cells where the seed count is fixed. No conflict with any spec rule.

**EV3. Pathway attribution for directions driven by `basic_level`.**
- **Proposed requirement.** Some `M11.C` criteria assert a direction across levels of `basic_level`, and `basic_level` feeds several derived quantities (`M10.A.1`; `M4.A.5`). For each such criterion, Phase E **SHOULD** run one arm per derived consumer: in that arm only the named consumer reads the varied `basic_level`, and every other consumer reads the reference value. The report **SHOULD** state which consumers carry the direction. Under `M11.4f` these arms are tests of model structure and **MUST NOT** be reported as intervention effects.
- **Where it would live.** Phase E, `M17.E`, beside `M17.E.5`.
- **Evidence.** ARGUED. The paper names this exact confound: one trait fed both the prompt and the fear equation, so its effect could not be attributed to either path (§2.1.2, §5.3), and the paper did not resolve it. The spec states the same risk for one consumer (`M4.A.5`: `M11.C.1` "passes for the wrong reason") but has no test for it. The transfer is INFERENCE.
- **What it changes.** Adds a test. It can expose a criterion that passes through `M4.A.5`'s self-generated term or through `life_energy` rather than through ties. Improves inference validity and protects fidelity.
- **Cost / risk.** Moderate: about 7 extra arms for each criterion of this kind. Consumers are swapped by declared switches only (`M17.D.3`). This does **not** question `M10.A`'s rule that these quantities be derived from `basic_level`.

**EV4. Extend the MM1 probe with an evidence-provenance factor (owner decision needed).**
- **Proposed requirement.** The held MM1 probe compares identical events delivered to receivers whose own anxiety differs. It **SHOULD** be crossed with how the evidence arrived: addressed, witnessed (`M1.F.5`/`M4.C.9`), and received multi-hop at reduced fidelity (`M1.F.4`). It **SHOULD** report the effect of receiver anxiety on the belief written (`M9.8`) separately for each.
- **Where it would live.** `M11.C` beside `M11.C.36`, as part of the MM1 test.
- **Evidence.** Shown in the paper, for LLM agents only: differences between dispositions were about 0 with no evidence, 0.265 with remembered or indirect evidence, and 0.033 with direct evidence (§4.1.2). The direct-evidence figure is partly forced by the prompt (§2.3.7), so this is weak support.
- **What it changes.** Adds a test dimension. If the owner's rule (belief writes read the receiver's anxiety, with upward threat bias) is adopted, the probe as held cannot tell a uniform effect from one concentrated on indirect evidence.
- **Cost / risk.** Low as a test.
- **Owner flag.** Whether anxiety should weigh more on indirect evidence than on direct is a theory question, not a design question. FE05.10 ("what might be") may bear on it and should be checked in `_LEDGER.md` before any rule is written. Until then the probe reports the interaction and asserts no direction. `M9.2` still binds.

A related point for the owner, not a candidate: in this paper, alarm without content raised urgency but rarely changed the danger appraisal (§4.1.3). In EPModel terms, an event can raise acute anxiety (`M4.C.1`) without writing a belief (`M9.8`). I found no requirement saying whether those two effects of one event are separable. That is a theory question.
