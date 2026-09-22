# Reading reports: the fourteen full-read preprints from the 2025-09 to 2026-09 sweep

Date: 2026-09-20. PDFs in `Model Design papers/Sweep 2026-09/`; catalogue entries in `PREPRINT_SWEEP_2025-09_to_2026-09.md`; consolidated candidates in `SPEC_CANDIDATES_from_preprints_2026-09-20.md`.

How these were produced. Each PDF was converted to text (pdftotext; figures survive only as captions; some equations are garbled and the readers say so where it matters). Five readers each took two or three papers, read `EPMODEL_BRIEF.md`, `DESIGN_LESSONS_model_design_papers_2026-09-17.md` and a written task, then wrote for each paper (A) a report on the eight questions in the brief, with every finding tagged [PAPER] (stated or shown in the paper) or [INFERENCE] (the reader's transfer to EPModel) and cited to a section, table or figure, and (B) candidate additions to the spec with a proposed requirement, target module, evidence strength (SHOWN / ARGUED / INFERENCE), what it changes, and cost or risk. The readers did not have the 427 requirements, only the brief and the design lessons; their "already covered" judgements are re-checked against spec v2 in the candidates file, which supersedes section B of each report where they differ.

> **Missing input, noted 2026-09-21.** `EPMODEL_BRIEF.md` and the readers' written task (`READER_TASK.md`) were working files given to the five readers and were never committed. Neither exists in this repository, its history, or anywhere searched on davemini2; the session that produced them did not run on this machine, so they may survive on the other Mac. They have **not** been reconstructed, because a reconstruction would be a guess presented as a source. What is lost is the ability to check what the readers were told about EPModel. What is not lost: every "already covered" judgement the readers made was re-checked against spec v2 by text search in `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, which supersedes the readers' coverage calls where they differ — so no candidate's status rests on the brief alone. If the files are found, commit them to `papers/` and remove this note.

Direct quotations from the papers have been replaced by paraphrases; paper-defined labels are kept in backticks. Numbers, citations and tags are as the readers wrote them.

Contents:

1. Buffalo, Pearson & Klein 2026 — event-keyed hashing for common random numbers (2603.11084); Blando et al. 2026 — statistical model checking of the Island Model (2604.04543); Kurz 2025 — equivalent bounded-confidence processes (2512.18016)
2. Sachdeva & van Nuenen 2025 — interaction protocol in multi-agent debate (2510.10002); Li & Tao 2026 — AI agents alone are not sufficient for social simulation (2603.00113); Buitrago López et al. 2026 — should LLM agents decide in social simulations (2606.12369)
3. He 2026 — VISA description protocol (2607.28027); Kalluri 2026 — trust development in small human–robot teams (2603.01189); Holland et al. 2026 — agent-based dynamics of criminal propensity with witnesses (2607.29546)
4. Prasad 2026 — transdiagnostic disorder-like phenotypes in RL agents (2607.07753); Wang et al. 2026 — do AI personas grow (2608.06485); Li et al. 2026 — benevolence bias in LLMs (2608.24912)
5. Ye et al. 2026 — TRAILS robustness audits (2605.18890); Zhou et al. 2026 — PIMMUR principles (2509.18052)

---

# Part 1

# Reader report: Buffalo/Pearson/Klein 2026, Blando et al. 2026, Kurz 2025

Files read in full: `txt/2026_Event-Keyed_Hashing_CRN_Buffalo_2603.11084.txt`, `.../2026_Statistical_Model_Checking_Island_Model_Blando_2604.04543.txt`, `.../2025_Equivalent_Bounded_Confidence_Processes_Kurz_2512.18016.txt`. Figures survive only as captions (Kurz's influence-graph drawings and Buffalo's Fig. 1–2 DAGs are lost except for labels); Kurz's fractions are garbled by extraction but the numerical claims quoted below are legible in prose.

---

# Paper 1. Buffalo, Pearson & Klein (2026), "Realizing Common Random Numbers: Event-Keyed Hashing for Causally Valid Stochastic Models", arXiv 2603.11084 [stat.ME]

## A. Report

**1. Simulation formalism (seeding, determinism, counterfactual arms).**
- [PAPER] §1, eq. (1): paired counterfactual estimator variance is `(Var Y1 + Var Y0 − 2 Cov)/M`; CRN works only if the same draw index means the same modelled event in both arms. Stateful PRNGs (Mersenne Twister, PCG, xoshiro; §3) break this whenever an intervention changes control flow: any conditional draw shifts every later draw index (the paper calls this draw indexing that depends on the execution path, §3.1).
- [PAPER] §3.2, Listing 1, Fig. 2, eq. (7): the program-level SCM contains an endogenous draw-index variable `K2 = 2 + I1`, a spurious causal path from person 1's outcome to person 2's noise. Proposition 1 and Corollary 3.1 (§3.3): seed-matched CRNs with stateful PRNGs fail to produce valid couplings whenever arms alter execution paths.
- [PAPER] §1: partitioning into separate streams per event class (Stout & Goldie 2008; L'Ecuyer 2002) is only a coarse-grained mitigation; within a class the dependence persists and choosing the granularity requires anticipating all execution-path changes.
- [PAPER] Definition 2.1 (execution invariance): `U_e = g(s, event_id_e)` with `event_id_e` required to be the same in every scenario and independent of what has executed before. §2.3: conditionally occurring events (present in one arm only) are fine; the failure is when events present in both arms get different noise.
- [PAPER] §2.1 parenthetical: rejection-sampling algorithms consume a variable number of uniforms, which creates misalignment except when the uniforms feeding the sampler are themselves keyed by event.
- [PAPER] §4.1, Listing 2: remedy = counter-based RNG (Philox/Threefry, Salmon et al. 2011) called as `CBRNG(seed, hash("infection", i))`; a second purpose label (`incubation`) for a second draw on the same agent. Purely functional, random access, repeated call with same key returns same value.
- [PAPER] §5: Philox ~2x slower than Mersenne Twister, Threefry ~5% faster (citing Salmon Table 2), with the caveat that the cost of computing keys may worsen these comparisons; counter-based generators parallelise naturally.

**Event identity across arms (slot-keyed vs dyad-keyed), §4.2.**
- [PAPER] Defining a stable key amounts to deciding which events in different worlds count as the same event; the paper treats this as a substantive modelling decision that cannot be automated (references the transworld-identity debate, Lewis 1986).
- [PAPER] Worked case: patient i has a scheduled encounter at time t; the worker present is j in baseline but k in the intervention arm. Two coherent keyings:
  - **Slot-keyed**, eq. (8)–(9): `event_id = (t, i, r)` (patient i's r-th encounter on day t); partner identity enters only through modelled state `S_e` and hence `p_e(S_e)`. Counterfactual question: does i get infected at that encounter given its modelled determinants.
  - **Dyad-keyed**, eq. (10)–(11): `event_id = (t, i, j)`; changing partner j→k changes the noise (`U_ij ≠ U_ik`). The individual-level counterfactual is well-posed only when both arms involve the same partner; otherwise only averages (ATEs) are defined.
- [PAPER] The choice encodes an assumption about exchangeability: slot-keying assumes partners are exchangeable conditional on modelled state (all partner heterogeneity is in `S_e`); dyad-keying allows partner-specific residual variation. Stateful PRNGs make the draw index the de facto event identity without anyone choosing it.
- [PAPER] §5: a spectrum: a simple event key carries a strong exchangeability assumption, while a key that is complex enough behaves nearly like comparing independently seeded runs.

**Agent identifiers when births differ across arms, §4.3 third principle.**
- [PAPER] Global mutable ID counters have the same defect as stateful PRNGs: if a parent dies in one arm, a child born in the other arm shifts all later IDs. Remedy: founders get indices 1..N0 in all arms; offspring get `id_child = h(id_parent, k)`, k = k-th offspring of that parent, while noting that other keying schemes, such as ones that include the time of birth, could suit some models better.

**Other key-design rules, §4.3.**
- [PAPER] Granularity: omitting the time index from a daily infection key collapses all daily trials into one event (effectively fixing the agent's susceptibility for its whole lifetime).
- [PAPER] Semantic reuse: draw each event a single time and store the value; accidental collisions negligible at 128-bit; debug builds can check that no key is used twice.
- [PAPER] Isolation: never put an endogenous summary (e.g. incidence on day t) in a key; history-dependence of risk must enter via `p_e(S_e)`, not via noise identity.
- [PAPER] Conditionally occurring events have a well-defined key whether or not they are queried; their existence never shifts other draws.

**Appendix A, consequences of violation.**
- [PAPER] Statistical efficiency: `Cov(Y1, Y0)` can even become negative, so seed-matching can increase variance.
- [PAPER] Counterfactual coherence: individual treatment effects end up comparing two unrelated chance events instead of measuring a causal effect on one unit.
- [PAPER] Auxiliary analyses: (i) one-at-a-time sensitivity sweeps: perturbing one parameter can trigger a new draw and break downstream matching; (ii) Sobol first-order indices pick up variance that is an artefact rather than real; (iii) mediation analysis (direct vs indirect effect via intervening on an intermediate variable) is contaminated by the spurious paths.
- [PAPER] §2.2, Table 1: with execution invariance, each event's `U_e` places it in a principal stratum (always/preventable/never); with ordered scenarios (increasing efficacy) the strata are nested in a consistent order, so a stronger intervention either leaves an outcome unchanged or turns it into a protected one; it never does the reverse.
- [PAPER] §2.3: a placebo arm (mechanistically identical, extra draws only) that diverges proves a violation; identical outcomes do not by themselves establish that invariance holds.

**5–7. Validation, limitations, software.** [PAPER] No experiments; the paper is a formal argument with a toy listing. No measured variance-reduction figures for a real ABM. [PAPER] §1 explicitly: within a fixed scenario a stateful-PRNG ABM is still a valid probabilistic model; the defect is only in across-arm coupling.

**Not transferable / cautions.** [INFERENCE] No content on agents, families or emotion; entirely about RNG plumbing and counterfactual semantics. The 128-bit collision remark assumes a good mixing function; a weak hash (Python `hash()` on tuples is salted per process and NOT stable across runs) would break M3.D.5 — the implementation must use a fixed, documented mixing function (e.g. Philox with the key derived from a stable byte-encoding of the tuple). The slot-vs-dyad choice is a modelling decision the paper refuses to make for you.

## B. Candidate additions

**B1.1 Counter-based, event-keyed RNG; no stateful generator in the engine.**
- Proposed: "Every stochastic draw in the engine MUST be computed as a pure function `g(seed, key)` of the run seed and a canonical event key by a counter-based generator (Philox or Threefry); the engine MUST NOT hold a mutable generator state. Distribution sampling MUST use a fixed number of keyed uniforms (inverse-transform), never rejection sampling."
- Where: M3 (clocks/ordering/determinism), next to M3.D.4/M3.D.5.
- Evidence: §3.3 Prop. 1, §4.1 Listing 2, §2.1 rejection-sampling note. SHOWN (formal) for the coupling failure; ARGUED for the remedy's sufficiency.
- What it changes: corrects a latent defect. Byte-identical determinism (M3.D.5) alone does not make two-arm seed-paired differences valid; the arm that adds or removes a mechanism will change control flow, so per-seed paired differences (the brief's trusted output, and the design-lessons per-seed paired differences technique) would partly measure noise re-alignment. Also makes the design-lessons §2.3 constant sweep and any Sobol decomposition (App. A) well-posed. Improves inference validity only.
- Cost/risk: low-moderate. NumPy ships Philox; a thin wrapper `draw(seed, key_tuple, purpose, index)` is all that is needed. Conflicts: none; it strengthens M3.D.4/5 and engine purity (M16.B: no state to snapshot for forks). Snapshot-and-fork (design lessons Q6) becomes a pure state snapshot with no RNG state.

**B1.2 Stable identifiers for persons, ties and triangles across arms.**
- Proposed: "Person identifiers MUST be stable across counterfactual arms: founders receive fixed identifiers from the family import; a person born during the run receives `id = h(parent_ids, birth_order)`; identifiers MUST NOT be allocated from a run-time counter. Tie identifiers are the unordered pair of person identifiers; triangle identifiers the sorted triple."
- Where: M1 (objects) and M15 (import).
- Evidence: §4.3 third principle. ARGUED (with the birth-order example).
- What it changes: prerequisite for B1.1. In a 40-year three-generation run, births and deaths differ between arms (mortality is a slow-tick output), so any counter-based ID would drift. Also makes trace-renderer comparisons across arms line up (M16).
- Cost: trivial if done at design time; painful to retrofit. No conflicts.

**B1.3 Declared key composition for every draw class, with the slot/dyad choice stated.**
- Proposed: "The spec MUST list every stochastic draw class in the engine (policy sampling, mixing-weight noise, per-hop fidelity, per-edge latency, witness overhearing, exogenous spell onset and duration, mortality, symptom onset) and for each state its key: `(tick, actor_id, purpose, k)` for actor-slot-keyed draws, `(tick, actor_id, partner_id, purpose, k)` for dyad-keyed draws, `(family, spell_class, occurrence_index)` for spells. Keys MUST contain only structural identity (tick, ids, purpose label, within-event index), never a state quantity."
- Where: M3 (with a table), referenced by Phase E.
- Evidence: §4.2 eqs. (8)–(11) and the exchangeability argument; §4.3 first and second principles. ARGUED.
- What it changes: adds a reporting/design rule; makes explicit what counts as the same event across arms. [INFERENCE] Concrete recommendation for EPModel: move SELECTION is actor-slot-keyed `(tick, actor, select)` so that what A does this week is the coupled event even if the target differs across arms; target-specific draws (fidelity per hop, appraisal noise at the receiver, witness overhearing) are dyad-keyed, because the TRIANGLE move picking a different third party is exactly the paper's worker-j-vs-k case, and the tie is the object that carries residual chance in Bowen terms. Exogenous spells keyed by occurrence index `k` give principal-strata ordering (Table 1): an arm with higher hazard can only bring the k-th spell forward, never reshuffle it. Symptom/illness onset keyed `(tick, person, symptom, channel)` makes more-load-implies-same-or-earlier-symptom a per-seed monotone coupling, which is what directional acceptance tests want.
- Cost: documentation plus discipline in the wrapper. No conflicts. Risk: over-fine keys approach independent seeds (§5), losing the pairing; the spec should say which keys are deliberately coarse.

**B1.4 Placebo-arm execution-invariance test and single-query caching.**
- Proposed: "M11 MUST include a placebo test: an arm that enables an extra mechanism at zero magnitude (extra draws, no state change) MUST reproduce the baseline trajectory byte-for-byte for every seed. The engine MUST query each event key at most once per run and cache the value; debug builds MUST assert on a repeated key."
- Where: M11 (acceptance tests) and M3.
- Evidence: §2.3 placebo argument; §4.3 semantic-reuse rule. ARGUED.
- What it changes: adds a cheap test that would fail loudly under any stateful-PRNG regression, and makes the existing mutation-test discipline (delete the mechanism, confirm red) measure the mechanism rather than noise re-alignment. Inference validity.
- Cost: trivial. Caveat from the paper: passing the placebo test does not prove invariance.

**B1.5 Per-seed principal-strata readout (Phase E).**
- Proposed: "Phase E SHOULD report, for each criterion and each seed, the paired arm difference and classify each seed as same/converted/reversed (principal strata); a directional criterion holds when converted seeds dominate reversed ones by the pre-declared margin."
- Where: Phase E.
- Evidence: §2.2 Table 1 and App. A (the `counterfactual coherence` consequence). ARGUED; the per-seed readout is INFERENCE.
- Changes: sharpens the existing difference-between-arms-over-ensembles rule; pairs naturally with the design-lessons pre-declared minimum effect size. Already partly covered by per-seed paired differences (design lessons §2.6); the addition is the three-way classification and the requirement that the pairing be valid (B1.1).
- Cost: nil beyond B1.1.

---

# Paper 2. Blando, Fagiolo, Giachini, Vandin & Ivanaj (2026), "Statistical Model Checking of the Island Model", EPTCS 443 (MARS'26), arXiv 2604.04543

## A. Report

**1. Formalism.** [PAPER] §5.2–5.3, Table 1: N = 20 agents on a T×T grid, 201 steps, synchronous per-step update (signal transmission, type transitions, movement). Agents are miners/imitators/explorers. Original model was one monolithic MATLAB script; the refactor had to control the random seeds so runs were independent and had to keep the sequence of random draws in the same order so that the new code behaved identically to the original (§5, subsection on the original implementation).

**2. Agent architecture.** [PAPER] Three discrete types with rule-based transitions (§5.3, Fig. 1): miner→explorer with probability ε; miner→imitator on receiving a stronger signal; explorer→miner on discovery; imitator→miner on arrival. No memory beyond `Past_skills` (§6.3).

**3. Interaction.** [PAPER] eq. (3): signal reception probability `w_ij = m/Σ1[miner] · exp(−ρ d_ij)`, Manhattan distance; ρ sets knowledge locality.

**5. Statistical method (the transferable content).**
- [PAPER] §4, §7.1: MultiVeStA adds simulation batches (block size 30) until the CI width at every queried time point is below δ (δ = 1, 95% confidence); time regions where variance is higher automatically get correspondingly more runs. Requires IID simulations (§4, subsection on MultiVeStA).
- [PAPER] §6.2, Listing 1: the simulator exposes exactly `reset(seed)`, `next`, `eval(obs)`, plus `setParams` once at startup; the checker drives the loop externally.
- [PAPER] §7.4: Welch's t-test at each time step between two parameterisations, α = 0.05, with statistical power reported. 6 of 7 pairwise comparisons reject at t = 201 with power > 0.85; ρ = 3.0 vs 5.0 does not reject at any step with power = 1.0, read as saturation.
- [PAPER] §7.4.1: α ∈ {0.9, 1.0, 1.1}; α = 1.1 needed 60 runs from t = 131 on (higher variance in the super-linear regime); the others converged at 30. α = 1.0 vs 1.1 shows equal means for t ∈ [11, 71] and rejects from t = 81 on.
- [PAPER] §7.4.2: only ϕ ∈ {0, 0.1} converged; larger ϕ values gave variance too high to reach the δ = 1 precision. ϕ = 0.1 vs 0: 8.1 vs 7.4 log-GDP at t = 201, power > 0.99; not rejected for t ∈ [11, 41].
- [PAPER] §7.2, Fig. 2: mid-run intervention arm (ε set to 0 at t = 50) vs sustained arm: plateau ≈ 10.4 vs ≈ 22.7 at t = 201.
- [PAPER] §5, subsection motivating SMC: the model is path-dependent with multiple regimes (growth, stagnation, lock-in), variance between simulation runs is large, and small parameter changes can switch regime.
- [PAPER] §5 end: analysis limited to aggregate observables, expected trajectories, one-parameter-at-a-time sweeps; distributions at the agent level and interactions between parameters are explicitly out of scope. §8: sweeping several parameters at once would need experimental designs that sample the space more selectively.

**6. Limitations.** [PAPER] Only means with CIs; no distributional readouts; no seeds-vs-parameters variance split; validation is reproduction of the original paper's stylized facts (Figs. 1a, 6d, 8 of Fagiolo & Dosi 2003), not data.

**7. Software.** [PAPER] §6.1: Model/Agent class split; the analysis driver lives outside the model (cf. chiSIM/EMEWS in the design lessons). Future work (§8): process mining on agent traces to uncover decision patterns that aggregate observables cannot show.

**8. Small N.** [PAPER] N = 20 default, which is close to EPModel's scale; the paper does not discuss finite-size effects.

**Not transferable / cautions.** [INFERENCE] Welch's t-test is an unpaired test on independent samples; under CRN-paired arms (Paper 1) the right test is on per-seed differences, so MultiVeStA's built-in comparison is the wrong tool for EPModel's paired arms unless the observable is itself the paired difference. Nothing on agents' internal states, ties, or emotion. The economic content is irrelevant.

## B. Candidate additions

**B2.1 Adaptive ensemble size with a declared precision, and `undetermined` as a third test outcome.**
- Proposed: "For each M11 directional criterion, Phase E MUST add seeds in blocks until the confidence interval of the per-seed arm difference has half-width below a declared δ, up to a declared cap; the run log MUST record the seed count used. A criterion whose interval has not converged at the cap is reported as UNDETERMINED, not as pass or fail."
- Where: Phase E; M11 (outcome vocabulary); M16 (log).
- Evidence: §4 and §7.1 (block-based convergence), §7.4.2 (non-convergence of ϕ ≥ 0.2 reported as a result). SHOWN as practice; the third outcome is INFERENCE.
- Changes: adds a reporting rule; addresses the design-lessons point that ensemble size otherwise decides the result (Röchert). Inference validity.
- Cost: low. No conflicts; complements the pre-declared minimum effect size (which sets δ).

**B2.2 Time-resolved arm comparison with power.**
- Proposed: "Phase E SHOULD test the arm difference at each reporting tick and report the first tick at which the pre-declared direction holds, together with the power of the test where the null is not rejected."
- Where: Phase E.
- Evidence: §7.4 (per-step tests, first-rejection ticks t = 81, ρ saturation with power 1.0). SHOWN as practice.
- Changes: adds a readout. The design lessons already ask for windowed trajectories of slow variables; the addition is a formal no-difference result that carries its power, so that arms-did-not-differ is distinguishable from ensemble-too-small.
- Cost: low. Caution: with paired arms use a paired test, not Welch.

**B2.3 Engine driver interface `reset(seed)/next/eval`.** Already covered by M16.B engine purity and the caller-supplied log interface; one line only: the Blando interface is the minimum surface an external ensemble/SMC driver needs, and EPModel's engine should expose nothing beyond it.

**B2.4 Refactor equivalence.** [INFERENCE] Their regression test that keeps random draws in the original order is what one needs with a stateful PRNG; under B1.1 the equivalent is that the same seed and the same keys give the same trajectory, which M3.D.5 already gives. Already covered.

---

# Paper 3. Kurz (2025), "Equivalent bounded confidence processes", arXiv 2512.18016 [physics.soc-ph]

## A. Report

**1–3. Formalism and interaction.** [PAPER] §2 (A)–(H): n agents, discrete synchronous time, real-valued opinion, confidence ε; agent i averages all j with `|x_j − x_i| ≤ ε` (eq. 5). The influence graph `G(t)` (eq. 6) is undirected (Lemma 1.3). Process is deterministic given `(X(0), ε)`. Order-preserving (Lemma 1.1), width non-increasing (Lemma 1.2), affine-invariant (Lemma 1.5). Freezing time bounds: `F(n) ∈ Ω(n²)`, `F(n) < 4n(n²+1)`.

**What is exactly enumerable for small n (the question asked).**
- [PAPER] Lemma 2–3: for a fixed `X(0)`, the ε axis splits into finitely many half-open intervals `[ε_j, ε_{j+1})` with identical trajectories inside each; the breakpoints (the paper's `ε-switches`) are computed by a breadth-first algorithm: at each step insert all pairwise opinion distances of the current profile into the current interval partition, until freezing. Switches are rational if `X(0)` is rational. Example 1 (n = 4, X(0) = (2,3,4,5)): switches 1, 13/12, 3/2, 2, 3. No lower bound on interval length is given, but each has positive length (§2 after Lemma 3).
- [PAPER] Lemma 4: for fixed n, finitely many graph-equivalence classes (sequences of influence graphs) exist. Lemma 7: whether a labelled graph can be an influence graph is an LP feasibility problem (maximize slack Δ; realizable iff Δ* > 0); Example 6–7: a *sequence* of graphs is checked by chaining LPs with the averaging rule as equality constraints (eqs. 32–35), solvable in exact rational arithmetic. Lemmas 10–11 prune redundant constraints using left/right-neighbour indices.
- [PAPER] Influence graphs are unit interval graphs (Definition 5, §3), so the candidate set is small: Table 1 gives, for n = 12, 51,505 unlabeled unit interval graphs (29,624 connected) versus 165,091,172,592 unlabeled graphs; for n = 10: 4,502 vs 12,005,168. Stars with ≥ 4 vertices are never influence graphs (Fig. 4, footnote 6). Lemma 12: a gap > ε between neighbours in 1D is permanent, so the process decomposes into independent sub-processes (1D only).
- [PAPER] Transition graphs over influence graphs computed for n = 3 (Fig. 3, 8 labelled graphs) and n = 4 (Fig. 7, 9 unit interval graphs); refined to weighted influence graphs over clusters of equal opinion (Def. 6, Fig. 8–9) so that freezing is detectable (final state = no edges).
- [PAPER] Exact extremal values obtained by this enumeration: `F(1..7) = 0, 1, 2, 5, 7, 9, 12`; n = 5 has a unique maximal graph sequence, n = 6 exactly two, n = 7 exactly two (`G4^1 G2^4 G3^3 G1^4`, `G4^1 G2^5 G3^2 G1^4`). Maximum fragmentation from a connected start: `S(1..4) = 1`, `S(5) = S(6) = 2` (proved via Lemmas 13–14), `S(7) = S(8) = 3`, `S(9) = S(10) = 4`; for n = 10 the seven possible four-cluster size vectors are listed (§5). Constructions: `S(5k+2) ≥ 2k+2`, `S(4k+9) ≥ 2k+3`; Conjecture 2: `S(n)/n → 1/2`. Equidistant starts do not maximise freezing time for 5 ≤ n ≤ 14 (footnote 9).
- [PAPER] Example 5 (interval-valued inputs, n = 4, `x_i ∈ [i+0.9, i+1.1]`, `ε ∈ [0.9, 1.1]`): freezing time can be anything in 0..5, components 1..4, final width 0 to 3.2; in effect nearly any outcome is possible, though this would differ for larger n or tighter intervals.
- [PAPER] §2 after Example 4: BC processes are extremely sensitive to tiny numerical perturbations; the graph-equivalence view is offered as the robust description when inputs are only known as intervals.
- [PAPER] §6: slightly increasing ε can raise freezing time and cluster count (non-monotone); Hegselmann 2023 reports consensus turning into three clusters; a jump to four is open. Higher dimensions: LPs still work under ℓ1/ℓ∞, Euclidean needs semidefinite programs (Remark 2); Lemma 12 fails in 2D.

**4–7.** [PAPER] Initialisation is the whole object of study (initial profile up to affine map). No stochasticity, no validation, no software details beyond a mention that an LP solver and an exact rational simplex were used.

**Not transferable / cautions.** [INFERENCE] Everything rests on three properties EPModel lacks: one-dimensional ordered state, deterministic dynamics, and a linear update conditional on the discrete interaction pattern. Consensus/fragmentation are not Bowen readouts. The OEIS counts apply to unit interval graphs only, not to EPModel's kinship graph.

**Could the method apply to a threshold-gated move rule with 12 agents?** [INFERENCE]
- The structural idea does transfer: if, given the discrete skeleton of a tick (which threshold comparisons fired, which move each of 12 persons selected, which targets), the continuous state update is affine, then the question of whether this sequence of skeletons is realizable from some initial state or some constant vector is an LP, and the trajectory is piecewise constant in every threshold constant. That is exactly the ε-switch result generalised.
- Full enumeration does not transfer. Per fast tick the skeleton is 12 move choices from 9 plus target choices (≥ 9^12 ≈ 2.8×10^11 labelled patterns before targets), with no unit-interval or ordering structure to prune it, over ~2,080 ticks, with stochastic selection and exogenous spells. Kurz's exhaustive searches top out at n = 7 for freezing time and n = 10 for fragmentation in a system whose per-step pattern space is ≤ 51,505 for n = 12.
- Three partial uses are feasible: (a) **ε-switch diagnostic for [I] thresholds**: for a fixed seed and initial state, record at every tick the signed margin of every threshold comparison; the smallest margin per constant gives the interval around its current value inside which the trajectory is provably identical. This turns the design-lessons constant sweep into a statement that this directional result holds for θ in [a, b) for these seeds, and exposes constants that sit within numerical noise of a switch. (b) **Local reachability**: for one triangle (n = 3) over one or two ticks, enumerate which move-pattern sequences are realizable at all under the policy's threshold gates, LP-style; use it as a coverage check for M11 (which moves can ever fire from which states), the analogue of YuLan's action-graph reachability. (c) **Interval-valued initial conditions** (Example 5) is the same message as M15.D.4 (imported families run as ranges); Kurz gives it a sharp form: with n = 4 and ±10% input uncertainty the outcome set is essentially everything.
- Nonlinearities in EPModel (division by functional level, products in bond energy) break the LP; they would need piecewise-linear surrogates or SDP/nonlinear feasibility, which the paper does not cover.

## B. Candidate additions

**B3.1 Threshold-margin log and per-constant invariance interval.**
- Proposed: "For every [I] constant used as a threshold in policy or appraisal, the engine SHOULD emit, per tick, the signed margin of each comparison against it; Phase E MUST report for each directional criterion the interval of each threshold constant within which every seed's trajectory is unchanged (the minimum positive and negative margins over the run)."
- Where: M16 (log record) and Phase E; M10 (parameter register column `invariance interval`).
- Evidence: Lemma 3 and its algorithm (SHOWN, for the BC model); extension to EPModel is INFERENCE.
- Changes: adds a reporting rule and a cheap diagnostic; makes the design-lessons constant sweep exact rather than sampled for threshold-type constants; flags results that depend on a comparison decided by rounding (design lessons §2.7). Inference validity.
- Cost: low (one extra log record class; logging remains a pure observer, M16.B). No conflicts.

**B3.2 Reachability check of move patterns for a single triangle.**
- Proposed: "M11 SHOULD include a static reachability test: for a three-person sub-family over one fast tick, enumerate the move assignments that the M4 policy can produce from any state within the declared ranges, and assert that every one of the nine moves and WITHHOLD is reachable; unreachable moves MUST be listed."
- Where: M11.
- Evidence: Lemma 7 and the transition graphs of Figs. 3, 7, 9 (SHOWN for BC; construction for EPModel is INFERENCE).
- Changes: adds a test; closes the silently-filtered-behaviour-class and rare-move-coverage risks from the design lessons at the policy level rather than by ensemble sampling. Theory fidelity and inference validity.
- Cost: moderate if the policy is piecewise-linear (LP); if not, fall back to dense sampling of the state box, which weakens `unreachable` to `not observed`. No conflicts.

**B3.3 Interval-valued initial conditions.** Already covered by M15.D.4 (imported families run as ranges, outputs as envelopes); Example 5 is supporting evidence that at n ≈ 4–12 a ±10% input band can span the whole outcome space, not a new requirement.

**B3.4 Exact arithmetic.** [INFERENCE, one line] Kurz's rational-arithmetic remark is not worth adopting; EPModel's determinism rule is byte-identity, not exactness, and B3.1 catches the cases where float rounding decides a threshold.


---

# Part 2

# Reader report, round 2: Sachdeva & van Nuenen 2025; Li & Tao 2026; Buitrago López et al. 2026

All three read in full from the pdftotext files in `txt/`. Figures survive only as captions and axis text; the numbers below come from prose and tables, not from plots, except where a heatmap printed its cell values (Lopez Figs 3, 5, 7).

---

# 1. Sachdeva & van Nuenen, "Interaction Protocol Shapes Moral Judgment in Multi-Agent Debate" (arXiv 2510.10002v4, COLM 2026)

## A. Report

**1. Simulation formalism (protocols, exactly as defined)** [PAPER, §2.2, Fig. 1, App. N]
- Task: 1,000 Reddit AITA dilemmas (Jan–Mar 2025, selected as the 1,000 highest-entropy of 3,272), five categorical verdicts (YTA, NTA, ESH, NAH, INFO). Three models: GPT-4.1, Claude 3.7 Sonnet, Gemini 2.0 Flash, all at temperature 1 (App. H). Orchestrated with autogen. More than 30,000 debates total (15,000 main + 15,000 open-source).
- **Synchronous protocol**: Round 1, each model gives its verdict on its own, seeing only the dilemma. If verdicts differ, each model is shown the other model's Round 1 output and prompted for Round 2; models again respond independently and simultaneously. Repeat until verdicts agree or 4 rounds are reached. The prompt informs the agent that it and the other agent will pick their verdicts at the same time (App. N).
- **Round-robin protocol**: within a round, the model in position n sees the verdicts of the n − 1 models that answered before it in that round, then responds. Same stopping rule (consensus or 4 rounds). The order is fixed for the whole debate and the agent is told its position (the template states whether it goes first or second, App. N Jinja template). Run in pairwise (both orders) and three-way (all 6 orderings) variants (§3.3, App. B).
- No seeding or replication: the authors state that each experiment was run a single time (§4 Limitations). No stochastic control beyond the fixed temperature.

**2. Agent architecture: what inertia and conformity measure** [PAPER, §2.4, Eq. 2, App. L]
- A multinomial (softmax) model of the verdict y of model m on dilemma d in round r:
  logit[y=v] = θ_mv + φ_dv + α_m·1[v = v_{m,r−1}] + γ_prev,m·n^prev_{vd,r} + γ_within,m·n^within_{vd,r}
- θ_mv: model m's baseline preference for verdict v; φ_dv: fixed effect of dilemma d on v (both centred across verdicts for identifiability).
- **Inertia α_m**: the added log-odds of selecting v when the model had chosen that same verdict in round r − 1. Binary indicator.
- **Conformity γ_prev,m**: increase in log-odds per unit count of v among verdicts seen in previous rounds; **γ_within,m**: per unit count of v among verdicts already rendered in the current round. The paper notes that γ_within,m does not apply under the synchronous protocol (it is identically 0 there), so the within-round term isolates the round-robin order effect.
- Model selection (App. L, Table 12): AIC fixed effects only 92,232.78; +inertia 85,263.29; +global conformity 82,612.56; +separate global conformity 81,362.60; +model-specific conformity (the paper's model) 77,734.30; +inertia×conformity interactions 75,640.46 (ΔAIC −2,093.83, i.e. better still). The interaction terms δ are all positive, which the authors read as conformity and inertia reinforcing one another (App. L, Table 13). A debate-level random intercept has fitted SD 0.275 (Table 14); adding either term reduces coefficient magnitudes but the between-model differences largely survive.

**Table 1 estimates (log-odds, 95% CI, odds ratio)** [PAPER, §3.4]:
| Parameter | Estimate | 95% CI | OR |
|---|---|---|---|
| α_GPT | 2.12 | [2.01, 2.28] | 8.31 |
| α_Claude | 1.48 | [1.41, 1.55] | 4.39 |
| α_Gemini | 1.02 | [0.93, 1.10] | 2.76 |
| γ_prev,GPT | 0.26 | [0.19, 0.33] | 1.30 |
| γ_prev,Claude | 0.39 | [0.35, 0.43] | 1.47 |
| γ_prev,Gemini | −0.01 | [−0.03, 0.03] | 0.99 |
| γ_within,GPT | 2.22 | [2.08, 2.34] | 9.20 |
| γ_within,Claude | 0.03 | [−0.01, 0.06] | 1.03 |
| γ_within,Gemini | 1.65 | [1.54, 1.79] | 5.21 |

Headline: GPT-4.1 is the most inertial model under the synchronous protocol and the most conformist under round-robin (the authors describe a basic tension between a model's inertia and its conformity, §3.4). Claude shows essentially no within-round conformity. DeepSeek-V3.2 (§3.6): α = 2.29, γ_prev = −0.126, γ_within = −0.698 (inertial and anti-conformist).

**3. Interaction structure: results per protocol**
- Synchronous, pairwise [PAPER, §3.1, Fig. 2a-b]: Round-1 agreement Claude–GPT 66.1%, GPT–Gemini 53.6%, Claude–Gemini 53.0%; consensus in later rounds 24.5% / 29.0% / 38.5%; no consensus within 4 rounds 9.4% / 17.4% / 11.5%. Change-of-verdict (CoV) rate = fraction of dilemmas in which a model changed its Round-1 verdict: Claude 28.2% vs GPT 3.1%; Gemini 33.3% vs Claude 34.1%; GPT 0.6% (six debates) vs Gemini 41.2%. Round-1 NTA share: GPT 78.8% and 84.9%; Claude 55.6%, 55.4%; Gemini 51.9%, 50.9%; Gemini YTA 33.1%, 35.2%.
- Round-robin, pairwise [PAPER, §3.3, Fig. 2c-d]: consensus rates rose substantially. **Order effect**: Claude–GPT with Claude first ended in one round close to 90% of the time; reverse order (GPT first) ended in one round only 40%. Gemini second conformed to either partner (~90% one-round). GPT's CoV rate was higher than in the synchronous setting.
- Round-robin, three-way [PAPER, §3.3, App. B Figs 8–10]: consensus was reached in almost every dilemma. Debates beginning with Claude ended in round 1 nearly 90% for either ordering of the other two. GPT pulled more than 70% of dilemmas to NTA when it went first and Claude third; that effect vanished when Claude was in second position (Fig. 9). CoV rates depended on a model's own position and on its position relative to the others; GPT's CoV rate rose significantly when it came right after Claude (Fig. 10).
- Debate archetypes [PAPER, App. M, Table 15]: final verdict distributions in Claude/Gemini vs GPT debates mostly tracked GPT's own distribution; holdout rates GPT NTA→None 0.233 (vs Claude), 0.293 (vs Gemini); Gemini YTA→None 0.291. The most inertial agent's Round-1 verdict anchors the outcome.
- Value alignment [PAPER, §3.2, Fig. 5, Tables 8–10]: Jaccard similarity of up-to-5-value sets is higher in agreeing rounds (0.420–0.484) than disagreeing (0.274–0.324); in initially-disagreeing debates that reached consensus, similarity rose 30–60% (e.g. Claude–Gemini 0.268 → 0.431); in no-consensus debates 6–17%. The paper's `inherited values` effect: a model that changes verdict adopts values its opponent used (Fig. 4d-f). Consensus was accompanied by convergence of the stated justification, not only the verdict.

**4. Initialization / baseline** [PAPER, App. D, Tables 4–5]: each model was also run solo three times on all 1,000 dilemmas. GPT and Claude solo distributions match their synchronous Round-1 distributions (self-agreement GPT 0.849–0.874, Claude 0.71–0.777). Gemini did not: solo NTA 0.263–0.271 vs 0.509–0.519 in the debate setting; solo ESH 0.243–0.249 vs 0.063–0.074; self-agreement 0.532–0.544. The debate framing alone, before any message was exchanged, changed Gemini's baseline.

**5. Calibration/validation** [PAPER]: no ground truth by design (the verdicts are subjective, so there is no ground truth to check them against). The LLM value judge (Gemini 2.5 Flash) was validated by Jaccard vs a human on 100 items 0.533; vs itself 0.638; at T=0 0.644; vs GPT-5 0.547; vs Kimi K2.5 0.529 (Table 7). Fig. 5 was reproduced with the GPT-5 judge (Fig. 22). Bootstrapped 95% CIs across dilemmas; Mann-Whitney U tests. Single run per experiment.

**6. Failure modes / steerability** [PAPER, §3.5, App. C, Tables 2–3]: removing the `goals` section of the prompt had no effect on CoV (Fig. 13). A `balanced` prompt (consensus equal to correctness) raised CoV for all models, most for GPT, yet the ranking of the models stayed the same; no-consensus rates original/balanced/adversarial: Claude–GPT 0.094/0.062/0.184, Claude–Gemini 0.115/0.129/0.154, GPT–Gemini 0.174/0.073/0.239. Prompting to emphasise the value `Empathy and understanding` raised its usage 22.6–48.9% (Table 3) with debate structure unchanged. Conclusion: the models' behaviour relative to one another survives prompt edits. Llama 3.1 8B failed consensus 28–31% vs 70B 8–15%, despite the highest CoV (45%); 8B changed verdict in 21% of failed debates vs 5–8% for others (§3.6).

**7. Software engineering**: autogen orchestration; code and data public; system prompts printed in full (App. N). Nothing else.

**8. Small-N / emotion / family**: N = 2 or 3 agents. The dilemmas are family and couple conflicts (App. O examples: a new half-sibling; a partner's Disneyland ultimatum), but the agents are judges of a family, not members of one. Nothing on emotion dynamics or long horizons (max 4 rounds).

**Not transferable / cautions**
- Every experiment run once; single value taxonomy; the authors concede the models are already out of date (§4). The agents are told the protocol and their position in it, which no EPModel person is.
- The inertia/conformity model is fitted to LLM outputs; the coefficients say nothing about humans and must not be used as constants.
- The 4-round cap makes `no consensus` partly a horizon artefact.

## B. Candidate additions

**S1. Ordering-spread report for any sequential regime.**
- *Proposed requirement*: If Phase E runs an activation regime in which persons select or receive in sequence, the runner MUST, for each seed, run every ordering of the sequence (or a seeded sample of orderings of declared size) and MUST report, per acceptance criterion, the spread of the directional result across orderings as a variance component distinct from the across-seed spread.
- *Where*: Phase E (ensemble runner); reporting rule into M16.
- *Evidence*: SHOWN. §3.3 and Fig. 2c: same two agents, same dilemmas, first-round consensus 40% vs ~90% depending only on who goes first; Fig. 9: >70% NTA in one ordering, effect absent in another.
- *What it changes*: reporting rule; inference validity. Extends design-lessons §2.1 (which asks for an alternative regime) by requiring that the ordering within a sequential regime be treated as an experimental factor, not fixed silently.
- *Cost/risk*: low (loop over permutations; 12! is infeasible, so a seeded sample). No conflict with M3.D.4/5 if the ordering is itself seeded and logged.

**S2. Inertia/conformity decomposition as a post-run diagnostic with mutation-proved direction.**
- *Proposed requirement*: The analysis layer SHOULD fit, per person, a multinomial model of the selected move on (a) an indicator that the same move was selected last fast tick (inertia α), (b) the count of each move type received or witnessed in the current batch (γ_within) and in earlier ticks (γ_prev), with person and tick fixed effects. Acceptance tests MAY assert directions on these coefficients, e.g. zeroing all conductance MUST drive γ to ~0; deleting the self-directed channel MUST reduce α for high-`basic_level` persons.
- *Where*: M11 (tests) and M16 (renderer emits the counts).
- *Evidence*: SHOWN as method (Eq. 2, Table 1, App. L nested-model comparison); the mapping to EPModel channels is INFERENCE. The paper shows the two quantities are statistically separable and interact positively.
- *What it changes*: adds a diagnostic that reads the two policy channels off behaviour rather than off internal state; improves inference validity. Partially overlaps design-lessons §2.6 (regress selected moves on the inputs shown); the addition is the specific own-past vs witnessed-others split and the within-batch vs earlier-tick split.
- *Cost/risk*: low-moderate (a fitting step outside the engine; engine purity untouched). Risk: with 9 moves and ~2,080 ticks per person the fit is feasible, but rare moves will have wide CIs; report CIs.

**S3. Two-level no-interaction control.**
- *Proposed requirement*: The control arms SHOULD include both (i) all conductance zero and (ii) ties intact but event delivery suppressed, so that drift produced by standing tie load (bond energy ÷ `functional_level`) is separated from drift produced by events.
- *Where*: Phase E.
- *Evidence*: INFERENCE from App. D, Tables 4–5: Gemini's distribution shifted under the debate framing alone, before any message; the analogue in EPModel is a tie exerting load without traffic.
- *What it changes*: sharpens the no-interaction control already proposed in design-lessons §2.6 (flag: partly covered).
- *Cost/risk*: low.

Already covered, not written up: solo baseline (= no-interaction control, §2.6 of lessons); the finding that relative behaviours persist under prompting supports M3.D.6 for the LLM line without a spec change.

---

# 2. Li & Tao, "AI Agents Alone Are Not (Yet) Sufficient for Social Simulation" (arXiv 2603.00113v2)

Position paper; no experiments, no numbers of its own. Everything below is ARGUED unless marked.

## A. Report

**1. Simulation formalism: the tuple** [PAPER, Def. 4.1, Eq. 1]
Sim ≜ (E, G, C, {M_i, O_i, A_i, P_i, U_i, R_i}_{i=1..N}, D0, Sch, Vis, Tr), fixed N and horizon T:
- E environment state (policy parameters, institutions, resources, enforcement); G graph state g_t; C context c_t (exogenous conditions: policies, events, scenario).
- Per agent: M_i mental state (beliefs, preferences, memory); O_i observation; A_i action space; P_i policy, a_it ~ P_i(·|o_it, m_it; ℓ_i) where ℓ_i is the LLM configuration; U_i mental-state update m_{it+1} = U_i(m_it, o_it, a_it, v_t, e_t, g_t, c_t); R_i reward/consequence function mapping actions to social consequences.
- D0: distribution over initial conditions (e0, g0, m0^{1:N}) ~ D0(·|c0); it is required to carry the population's composition, the correlations among attributes, the starting network structure and the agents' epistemic priors.
- **Sch** (Scheduler): I_t = Sch(e_t, g_t, c_t) ⊆ {1..N}, i.e. which agents may act, and at which times.
- **Vis** (Visibility): (o_t^{1:N}, v_t) ~ Vis(·|e_t, g_t, c_t), v_t a visibility object (e.g. exposure graph) fixing which agent sees which information.
- **Tr** (Transition): (e_{t+1}, g_{t+1}) ~ Tr(·|e_t, g_t, c_t, a_t^{1:N}).
- Step loop per t: (1) sample observations and v_t via Vis; (2) sample active set via Sch; (3) each active agent samples an action (inactive agents no-op); (4) update mental states via U_i; (5) transition environment and graph via Tr.
- Claimed payoff (§4): information asymmetry lives in Vis instead of being replaced by an assumed context shared by everyone; ordering lives in Sch, which keeps artefacts of turn-taking from being mistaken for social dynamics; interventions become controlled edits to the environment state and mechanisms instead of changes made only in the prompt.

**Mapping to EPModel (my judgement, INFERENCE, against the brief; `exists` means the brief describes it, not that I read the 427 requirements):**
| Tuple element | EPModel counterpart | Status |
|---|---|---|
| E | `Family` (undifferentiation budget, societal anxiety, calendar, shared beliefs, ledger) | exists |
| G | `Relationship` ties + persistent `Triangle` topology | exists |
| C | exogenous spells with `exogenous` flag | exists |
| M_i | `Person` state incl. beliefs, inbox | exists |
| O_i | popped inbox batch + witnessed events | exists |
| A_i | nine moves + WITHHOLD (closed set) | exists |
| P_i, ℓ_i | two-channel rule policy (M4); no ℓ_i by M3.D.6 | exists; ℓ_i deliberately absent |
| U_i | appraisal + automatic-channel learning | exists |
| R_i | no utility/incentive mapping; consequences are conserved anxiety, invariants, belief writes with hysteresis | different by design, not missing |
| Tr | tie dynamics, cutoff flag, slow-tick drift | exists |
| **Sch** | fixed: every person every fast tick (M3.D.1) | exists as a hard-coded choice, **not as a named object**; the (e,g,c)-dependence (state can restrict who acts) is absent |
| **Vis** | targets, witnesses, route, per-hop fidelity, per-edge latency | exists and is richer than Li's (distortion); **unclear whether witnesses are derived from state or set by the sender** |
| **D0** | imported families run as ranges; outputs are envelopes (M15) | partly exists; **attribute correlation structure and D0-variance as a reported component are not described** |

**2–3. Agent architecture and interaction** [PAPER, §3.1–3.2.2]: Mismatch 1: role-play plausibility ≠ behavioural validity; prompt personas suffer from three named problems, `underspecification`, `stereotype completion` and `prompt sensitivity`. Mismatch 2: outcomes are often produced by the joint dynamics of agents and their environment, not messaging; an interaction-only simulator may be unable to represent an intervention at all, or will only be able to express it as an improvised prompt instruction. §3.2.2: a simulator is only properly defined once it states its interaction kernel, meaning which agents can affect which others, over which channels, and how those interactions change persistent state. Dialogue-first designs blur the line between communication and state transition and tacitly assume exchanges that are synchronous and highly observable. The authors hold that scheduling and synchronisation are components of the model itself, not merely engineering decisions.

**4. Initialization** [PAPER, §3.2.3]: three points. (a) Matching marginals is insufficient; drawing attributes independently of one another, without care, can destroy the joint distribution that underlies them; two initialisations with identical aggregates can embody quite different mechanisms. (b) Scale is an epistemic choice; finite-size effects can inflate run-to-run variance and so produce apparent treatment effects that disappear at larger scales (cites 500-agent Törnberg, 1,000/300-core HiSim, million-level ElectionSim). (c) A mismatch the paper labels as confusing information availability with information awareness: broadcasting a shock as prompt text erases the important distinction between information existing and agents actually being aware of it.

**5. Validation** [PAPER, §3.2.1, Action 2]: current practice is validation by demonstration, with little quantification of how much runs vary from one another. Action 2 prescribes three checks: behaviour under constraints; **mechanism sensitivity** (deliberate edits to C, Vis, Sch, or Tr should produce changes that are stable and consistent in direction); **counterfactual stability** (comparative conclusions should hold up across random seeds, rewordings of prompts, and other implementation choices that ought not to matter), achieved through ablating mechanisms, negative controls, and counterfactual sweeps that change a single component at a time, with results reported as distributions (for example uncertainty intervals and a variance decomposition).

**6. Failure modes**: selective reporting of trajectories; metric gains that reflect polished language rather than behaviour; rule layer dominating so that LLM agents are reduced to narrative wrappers around it (§3.2.2); the worry that structure makes the model rigid is answered by observing that removing structure does not remove assumptions but moves them into prompts, scheduling and interfaces (§5).

**7. Software engineering** [PAPER, Action 1]: Vis, Sch, Tr and R ought to be specified as versioned, inspectable artefacts accompanied by logged traces, so that outside auditors can reconstruct the constraints, incentives and information frictions in force at every step. Action 3: treat uncertainty as a primary output, and report sensitivity to D0, Vis, Sch, Tr and the LLM configuration.

**8. Small-N / family**: nothing. Households not mentioned.

**Not transferable / cautions**: no evidence of its own; every claim is a citation or an argument. The formulation assumes an LLM policy (ℓ_i) and a reward R_i, neither of which EPModel has. The scale discussion is about down-sampling a large population; EPModel's N=12 is fixed by the theory, so only the moral that a single run is never enough carries.

## B. Candidate additions

**L1. Scheduler and visibility as named, versioned engine objects.**
- *Proposed requirement*: The engine MUST implement activation (which persons select a move in a given fast tick) and visibility (which persons are targets or witnesses of an event, and with what latency and fidelity) as two separately named components with a declared interface; the run-log header MUST record the identity and version of each. The default activation, "all persons every fast tick", MUST be recorded as such rather than left implicit.
- *Where*: M3 (clocks/ordering), M1 (events), M16 (log header).
- *Evidence*: ARGUED (Def. 4.1, Action 1). 
- *What it changes*: software structure and reporting; makes design-lessons §2.1 (alternative regime) a swap rather than a rewrite; lets an auditor reconstruct who could act and who could see from the log alone. Improves inference validity.
- *Cost/risk*: low if done at design time; none of M3.D.4/5, M3.D.6, M16.B is touched.

**L2. Order-permutation invariance test, with per-object RNG streams.**
- *Proposed requirement*: (a) The acceptance suite MUST include a test that, for a fixed seed, permutes the iteration order of persons in the select-all step and the delivery order of events within every same-tick batch, and asserts a byte-identical final state and (order-normalised) log. The test MUST be proved failing by replacing the commutative batch reduction with a sequential one. (b) To make (a) achievable, every stochastic draw MUST come from a per-person (or per-object) substream derived from the run seed and the object's stable id, never from a single stream consumed in iteration order.
- *Where*: M3 (determinism) for (b); M11 for (a).
- *Evidence*: Li Action 2(3), ARGUED (conclusions should survive implementation choices that ought not to matter); Sachdeva §3.3 SHOWN that uncontrolled ordering changes outcomes qualitatively. (b) is INFERENCE: a commutative reduction does not save M1.F.8 if the RNG is consumed in loop order.
- *What it changes*: adds a test that operationalises M1.F.8 and M3.D.5 together; corrects a likely implementation trap. Inference validity.
- *Cost/risk*: low. No conflict; it strengthens M3.D.4/5.

**L3. Initial-condition distribution D0 with declared joint structure and its own variance component.**
- *Proposed requirement*: Initial state for any run family MUST be specified as a distribution D0 over (person attributes, tie states, triangle topology, beliefs) that declares its correlation structure (e.g. any theory-stated pairing of spouses' `basic_level`, or ordering of children's levels), not as independent per-attribute ranges. The ensemble runner MUST draw initial states from D0 and MUST report D0-variance separately from seed variance, exogenous-spell-timing variance and constant-sweep variance.
- *Where*: M15 (import) and M10 (config); Phase E for the reporting rule.
- *Evidence*: ARGUED (§3.2.3: sampling attributes independently can destroy the underlying joint distribution; Action 3: report sensitivity to D0).
- *What it changes*: partly covered by M15.D.4 (initial conditions not nuisance) and design-lessons §2.5/§2.6; the addition is (i) correlation structure as a required part of D0 and (ii) D0 as a fourth named variance source. Both theory fidelity (if the corpus states such correlations) and inference validity.
- *Cost/risk*: moderate (a small sampler and a variance-decomposition step). No conflict with M11.F.9 as long as D0 is not tuned to a known history.

**L4. Witness set produced by the visibility rule, not chosen by the sender.** (Flag: check spec.)
- *Proposed requirement*: The set of witnesses of an event MUST be computed by the visibility component from tie and household state (co-residence, tie conductance, route), and MUST NOT be a free field set by the sender's policy; a sender's choice of audience, where the theory needs it, MUST be expressed as a move (e.g. TRIANGLE targeting a third party), not as witness selection.
- *Where*: M1 (events), M4 (policy).
- *Evidence*: ARGUED (Vis is a function of state in Def. 4.1); mapping is INFERENCE. The brief lists `witnesses` as an Event field; I could not tell whether it is derived or authored.
- *What it changes*: if witnesses are currently authored by the sender, this moves an assumption out of the policy into an auditable mechanism (theory fidelity: who overhears is a property of the household, not of the speaker's intent).
- *Cost/risk*: low; possibly already how it is done.

Already covered, not written up: information availability vs awareness (the belief layer and `exogenous`-flagged Events already separate these, provided spell onsets enter as Events rather than as global flags, which is worth a one-line check); mechanism ablation and one-component-at-a-time sweeps (mutation tests; design-lessons §2.3); intervention channels (design-lessons §2.8).

---

# 3. Buitrago López, Pastor-Galindo & Ruipérez-Valiente, "Should LLM Agents Decide in Social Simulations?" (arXiv 2606.12369v1)

## A. Report

**1. Simulation formalism and the reference policy** [PAPER, §3.1–3.2]
- Synthetic microblogging OSN from the authors' own framework [4]: 1,000 agents, directed homophily-generated graph (fixed across conditions), action space A = {read, like, share, reply, post, follow, unfollow}. The simulation advances by activating agents one after another: at each step one agent is activated, its local context is derived from the current network state, and one valid action is selected and executed (read exposes content; like/share/reply need available content; follow/unfollow need valid candidates). 10,000 action-selection steps per configuration.
- **Reference policy** = a finite state machine realised as a first-order Markov model. Each state is an action; the next action is sampled from a transition matrix conditioned on current state and user type u: p_ij^(u) = P(A_{t+1} = a_j | A_t = a_i, U = u), Σ_j p_ij^(u) = 1. A **contextual mask** (has the agent read a post; are like/share/reply possible; are there follow/unfollow candidates; should repeated post/reply be down-weighted) is applied and the result renormalised to what the paper calls the `final normalized probabilities` (described in the v3 prompt, §3.2). User types: Passive 547 (54.7%), Socializer 221 (22.1%), Debater 135 (13.5%), Advanced 97 (9.7%).
- FSM reference counts by type (Fig. 3, FSM row; read/like/share/reply/post/follow/unfollow): Advanced N=870: 468/44/71/74/189/13/11. Debater N=1,520: 900/92/146/213/134/20/15. Passive N=5,910: 5,124/421/167/88/40/55/15. Socializer N=1,700: 967/196/174/153/158/42/10. Read dominates every type; follow/unfollow are rare (≤ 1.5%).
- Decoding for all LLM runs: temperature 0.7, top_p 0.9, top_k 40; local vLLM on a 10-core Xeon 2.3 GHz, 100 GB RAM server. No seeds reported; one run per configuration.

**2. Agent architecture: the three prompt strategies** [PAPER, §3.2, Table 1]
- v1 Base (213 words): metadata, state, constraints, actions with definitions, only the barest behavioural framing; the prompt instructs the model not to try to maximise the diversity of its actions.
- v2 Guided (670 words): five empirical rules (consume before reacting; creation is unequal; actions differ in effort; roles shape engagement; follow/unfollow are ordinary candidates) plus calibration rules discouraging over-selection of post/reply. The authors characterise it as a deliberate, explicit intervention in the decision process.
- v3 Probabilistic (538 words): the LLM is handed the FSM's own transition row, the mask, and the final normalised probabilities, told to treat them as strong priors on behaviour but not to always pick the max, ignore them, or balance.

**3. Alignment metric and results** [PAPER, §3, §4.1–4.4, Table 2]
- Jensen–Shannon divergence, base-2 (range [0,1]), Laplace smoothing α = 10⁻⁶, over the global action distribution after 10,000 steps; also weighted per-user-type averages.

| Prompt | Model | Global JSD | Time (s) | s/action | × FSM |
|---|---|---|---|---|---|
| FSM | – | 0.000 | 7 | 0.0007 | 1.0 |
| v1 | LLaMA 3.1 8B | 0.359 | 946 | 0.0946 | 135.1 |
| v1 | GPT-OSS 20B | 0.113 | 2,172 | 0.2172 | 310.3 |
| v1 | Mistral Small 3.2 24B | 0.045 | 2,989 | 0.2989 | 427.0 |
| v2 | LLaMA | 0.223 | 3,089 | 0.3089 | 441.3 |
| v2 | GPT-OSS | 0.672 | 6,334 | 0.6334 | 904.9 |
| v2 | Mistral | 0.055 | 9,360 | 0.9360 | 1,337.1 |
| v3 | LLaMA | 0.278 | 1,816 | 0.1816 | 259.4 |
| v3 | GPT-OSS | 0.035 | 2,990 | 0.2990 | 427.1 |
| v3 | Mistral | 0.132 | 5,794 | 0.5794 | 827.7 |

- Means: across the 9 configurations JSD 0.212; by prompt v3 0.148, v1 0.172, v2 0.317; by model Mistral 0.077, GPT-OSS 0.273, LLaMA 0.287. Best prompt differs per model (v2 for LLaMA, v3 for GPT-OSS, v1 for Mistral): no single prompting strategy comes out best for every model. Slowdown: mean 563.3×, minimum 135.1×; the configuration that is most faithful is not the one that is most efficient.
- Weighted per-type averages: LLaMA v2 0.243 (best), v1 0.390; GPT-OSS v3 0.087, v1 0.120, v2 0.676; Mistral v1 0.060, v2 0.105, v3 0.165. Per type: LLaMA v3 Advanced 0.108, Socializers 0.181, but Passive 0.420; v2 Debaters 0.158, Passive 0.234. GPT-OSS v3 Passive 0.027 but Advanced 0.180, Debaters 0.166, Socializers 0.179; v2 0.660–0.686 for every type.
- Concrete distortions (heatmap cells, Figs 3, 5, 7): LLaMA v1 Passive read 5,124 → 1,071, post 40 → 1,810, reply 88 → 1,652; GPT-OSS v2 Passive follow 55 → 5,139 (follow dominated every type); Mistral v2 Passive read 6,077 of 6,080 (everything else collapsed); LLaMA v3, given the exact probabilities, still Passive read 1,052 vs 5,124 and reply 1,605 vs 88. Follow/unfollow are driven to 0 in most LLM cells (e.g. LLaMA all prompts Advanced/Debater/Socializer follow = 0).

**4. Initialization**: profiles with age, personality, occupation, interests, user type; homophily graph; fixed across conditions (§3.1). No sensitivity to it reported.

**5. Validation**: alignment only against the authors' own FSM, not against human data; distributions only (the global figure pools all user types together). What they could not do: sequences/transition structure are not compared (the reference is first-order Markov but only marginals are scored); no seeds, no CIs, no repetition; effect on downstream diffusion left to future work (§5).

**6. Failure modes** [PAPER]: adding guidance to the prompt can itself induce systematic biases in the actions chosen (abstract); v2 made GPT-OSS worst of all nine (0.672) and Mistral narrowest; one and the same prompt brought one model nearer to the FSM while pushing another further from it; the authors warn against assuming that an LLM's action selection follows the intended policy merely because the prompt spells that policy out (§5). Rare actions vanish.

**7. Software engineering**: a framework in which choosing the action and generating the text are separate stages [4], which is what made the swap-in comparison possible; the FSM is the only component changed.

**8. Small-N / emotion / family**: none. Heterogeneity is one categorical type per agent controlling a transition matrix, loosely analogous to differentiation level controlling the policy mixing weight.

**Not transferable / cautions**: single run per configuration and no variance, so the JSD differences between e.g. 0.035 and 0.045 are unranked noise; the FSM itself is invented, so `alignment` is alignment with an assumption; open-weight models only; distributions are pooled over the whole run, so drift over time is invisible.

**Inference for the LLM line** (mine): an EPModel run is 12 × ~2,080 ≈ 25,000 decisions. At the FSM rate (0.0007 s) that is ~17 s; at the observed LLM rates (0.09–0.94 s/decision) it is 39 min to 6.5 h per run, before ensembles. And v3 shows that handing an LLM the engine's distribution does not make it a faithful sampler, so the design-lessons §3.6 pattern (engine samples; LLM at most estimates) is the only one of the two directions with evidence.

## B. Candidate additions

**P1. Legality mask computed and logged before selection.**
- *Proposed requirement*: For each person each fast tick, the policy MUST compute the set of legal moves from current tie and triangle state (e.g. CUTOFF requires a live tie; TRIANGLE requires a reachable third party; PURSUE requires a non-cutoff target) and MUST select only over that set with renormalised weights; the mask MUST be written to the run log so that a move that was never legal is distinguishable from one that was legal and never chosen.
- *Where*: M4 (policy), M16 (log record).
- *Evidence*: SHOWN as practice (§3.2 FSM contextual mask and renormalisation); the rare-action collapse in LLM rows (follow/unfollow → 0) SHOWN. Requirement wording is INFERENCE.
- *What it changes*: makes the rare-move coverage test (design-lessons §2.6) interpretable, and moves the question of what is possible right now out of the policy into an auditable mechanism. Both theory fidelity and inference validity.
- *Cost/risk*: low. No conflict.

**P2. Move-transition structure as a logged readout.**
- *Proposed requirement*: The trace renderer SHOULD emit, per person and per arm, the first-order move-transition count matrix (which move followed which), and ensemble reports SHOULD compare arms on transition structure as well as on move marginals.
- *Where*: M16 (renderer), Phase E.
- *Evidence*: INFERENCE from the paper's own gap: its reference policy is Markov in the move sequence, yet only marginals were scored. A pursue–distance cycle or a triangle hand-off is a transition property that marginal counts cannot show.
- *What it changes*: adds a readout; inference validity (a directional test on marginals can pass while the sequence structure the theory predicts is absent).
- *Cost/risk*: low (a 10×10 count per person). Pure observer, so M16.B is respected.

**P3. Divergence metric for arm comparison.**
- *Proposed requirement*: Where an acceptance criterion concerns the distribution of moves, the ensemble report SHOULD include base-2 Jensen–Shannon divergence with Laplace smoothing between arms' move distributions, per person stratum, with a per-seed paired distribution rather than a single pooled value.
- *Where*: Phase E.
- *Evidence*: SHOWN as method (§3, Table 2); the per-seed pairing is from design-lessons §2.6 (OASIS), not this paper.
- *What it changes*: adds a scalar summary that is bounded and symmetric; reporting only. Modest value; the directional criteria remain primary.
- *Cost/risk*: trivial.

Supports existing rules without a spec change: M3.D.6 (no LLM in the decision path) is directly supported by the JSD table and the 135–1,337× cost; nothing here argues for relaxing it.

---

## Cross-paper note for the owner

The three papers converge on one point the spec already half-holds: **who acts when, and who sees what, are model content, not plumbing.** Sachdeva shows it empirically (ordering alone moves first-round consensus from 40% to ~90%), Li & Tao name the objects (Sch, Vis) and demand they be versioned and logged, and Lopez shows what happens when the decision component is swapped for one whose policy is not inspectable (mean JSD 0.212, 563× slower). The cheapest concrete additions are L1 + L2 (named scheduler/visibility objects, per-object RNG streams, an order-permutation test proved failing by mutation) and P1 (logged legality mask); the reporting additions S1, L3 and P2 belong in the Phase E specification.

---

# Part 3

# Reader report, round 2: VISA (He 2026), Kalluri 2026, Holland et al. 2026

Read in full: READER_TASK.md, EPMODEL_BRIEF.md, DESIGN_LESSONS.md, and the three pdftotext files. *(READER_TASK.md and EPMODEL_BRIEF.md were not preserved — see the note under "How these were produced". DESIGN_LESSONS.md is presumably `DESIGN_LESSONS_model_design_papers_2026-09-17.md` as it stood before its §8, which was written from these reports.)* All statements about what EPModel "has" rest on the brief and design lessons, not on a read of the 427 requirements; each "already covered" judgement should be checked against the spec.

---

# Paper 1. He, "VISA: A Structured Description Protocol for ABMs Towards Machine Reproducibility" (arXiv 2607.28027)

## A. Report

**1. Simulation formalism (what the protocol requires to be stated).** [PAPER]
- Eight tables, four agent-level, four model-level (§3.1, Fig. 1). Fields:
  - **T1 Agent**: Name, Set (calligraphic symbol), Instances, Category ∈ {Environment, Space, Decision-maker, Passive}, Description, Quantity (N fixed, n variable) (§3.3.1). Environment = manages other agents' lifecycle and computes aggregates; Space = active agent owning topology and answering neighbour queries, explicitly not treated as a passive container; only things that behave become agents; output records and lookup data are not agents (§3.3.1, App. D.1.6).
  - **T2 Variable**: Variable, Symbol (endogenous symbols carry index t), Type ∈ {Exog.-homogeneous, Exog.-heterogeneous, Endog.-decision, Endog. non-decision}, Data Type, Value (= `Input` → T6a, or the f-IDs that write it), Unit, Desc (§3.3.2, Fig. 2).
  - **T3 Sensing**: quasi-matrix, rows = observers (all non-passive types), columns = observed; cell = ∗ (all), ∅, or a list of variables; diagonal = what an instance may observe of its own-type peers; self-observation implicit; passive agents are columns only (§3.3.3).
  - **T4 Internal Function**: ID (f-n), Function, Method, Decision Basis (from T2/T3), Self-state Update, External Effect (other agents' endogenous variables, or instance create/remove), Ref (§3.3.4). The paper's principle is that the sole purpose of any internal function is to update endogenous variables.
  - **T5 Associated Data**: ID (d-n), Title, Type ∈ {Empirical, Literature, Generated}, Temporal ∈ {Static, Dynamic}, Source, Collection ∈ {Survey, Administrative, Sensor, Experimental, Computational}, Pre-processing ∈ {None, Selected, Aggregated, Transformed}, #Rec, Availability ∈ {Open, Restricted, Private} (§3.4.1). T5 may be empty; then every T6a source is `Author` (App. A.1.5, B.1.5).
  - **T6a Input**: Symbol, Value/Distribution, Data source (d-ID or Author), Derivation ∈ {Direct, Estimated, Computed, Assumed}, Algorithm, Ref. **T6b Output**: Symbol, Indicator, Formula, Data Type, Unit, Frequency (integer k ≥ 1 = sample every k steps; −1 = terminal only), Desc (§3.4.2).
  - **T7a Schedule** (the one asked for): Step, Agent, ID, Function, Exec. mode, Condition. Exec. mode ∈ {Synchronous (all observe start-of-step state, lockstep write-back), Sequential (fixed deterministic order, later instances see earlier writes; order given in parentheses, e.g. descending order of w_i), Random-order (reshuffled each step with the seeded RNG), Asynchronous (event/time-driven, e.g. a Poisson process with rate λ)}. Conditional activation is NOT a mode; it goes in Condition. Staged execution (sense → decide → update, each phase completed by all before the next) is described as a composite pattern assembled from the four basic modes, applied where within-step information contamination has to be excluded, and treated as a local organising principle rather than a fixed global structure (§3.4.3). **T7b Termination**: ID (c-n), Indicator (T2 variable or T6b output; t exempt), Condition, Description, Value Source/Ref, Termination logic (boolean combination) (Table 7b).
  - **T8 Validation**: ID (v-n), level ∈ {Agent, Model, Output} (HAV framework, He et al. 2026), Validation object, Benchmark data, Method, Indicator (test statistic), Passing cond., Ref (§3.4.4).
- Code-generation skill, exec-mode patterns (App. D.3.6): Synchronous = take a snapshot, compute, then swap it in; Random-order = shuffle with seeded RNG; Asynchronous = priority queue of activation times. The appendix names the execution mode as the largest single cause of irreproducibility. Safeguards: one seeded RNG threaded through every draw, `--seed` flag, seed in run log, T3 matrix reproduced as a docstring on the sensing layer (D.3.8).
- Reproduction standard used in §5: a qualitative signature that does not depend on the random seed, 3 NetLogo runs vs 3 Python seeds, min–max bands (Figs. 3–5). Rebellion: peaks near 300, ~30 episodes per 1000 ticks. Wolf stride reversal 1.0→0.78 (penalty on) vs 1.0→1.18 (off). One criterion (v1, sheep stride → 1) was reproduced in direction only, not in magnitude.

**The 19 consistency rules** (Tables 9, 11), with my judgement of EPModel's status per the brief/design lessons [INFERENCE unless noted]:

| Rule | Content | EPModel status |
|---|---|---|
| r1 | every variable has one of 4 leaf types; endogenous symbols carry t | Not addressed as a check. Constants are labelled [I] (≈ Derivation "Assumed"); state variables are fast/slow but not exog/endog/decision-typed |
| r2 | unique function IDs | Requirements have IDs; functions/mechanisms not known to be enumerated |
| r3 | every function writes ≥1 endogenous variable | Stronger analogue exists: every mechanism must be proved live by mutation (M11) |
| r4 | all functions in one Step share one Exec. mode | Satisfied by construction: M3.D.1 is one uniform staged synchronous scheme (design lessons §2.1). VISA would record it as Synchronous composite plus Asynchronous delivery (per-edge latency M3.C.1) |
| r5 | peer-sensing diagonal stated | Partially: what one Person can observe of another is defined by Event delivery (targets, witnesses, per-hop fidelity), not by a stated matrix |
| r6 | passive agents have no state/functions | Satisfied: external agents are full Persons, no passive agents in the brief |
| r7 | one observer row per active type | n/a (one agent type, Person; Family ≈ Environment agent; Triangle topology ≈ Space agent) |
| r8 | variable-count types need create/remove functions | Unclear. Mortality exists on the slow tick; births over 40 years unstated in brief; disposition of a deceased Person's conserved quantities unstated in brief (see candidate V3) |
| r9 | every active type owns ≥1 function | Satisfied (each Person selects one move per fast tick) |
| r10 | every variable is observable somewhere or never read | Not addressed as a check |
| r11 | every endogenous variable has a named writer; no dangling references | Not addressed as a check |
| r12/r13 | self-writes only to own endogenous vars; cross-agent writes only to others' endogenous vars or create/remove | Satisfied in a stronger structural form if, as the brief says, Events are the only cross-agent channel and invariants constrain what an exchange may do |
| r14 | a function's Decision Basis must be self-attributes or T3-authorized observations | Partially: "agents act on beliefs" is the intent; no stated check that the policy cannot read another Person's true state (candidate V2) |
| r15 | every input has a value; every value corresponds to a declared variable; outputs computable from state | Satisfied in spirit by M10 config, provided every [I] constant is declared there |
| r16 | every function scheduled; no phantom schedule entries | Partially: M3 orders phases; unknown whether every mechanism is placed in the phase order |
| r17 | termination indicators are declared quantities | Satisfied trivially (fixed ~40-year horizon) |
| r18 | validation objects are state variables or outputs | Satisfied: M11 criteria are on readouts; readout traps are recognised |
| r19 | empirical references resolve to data records | Partially: the 12 corpus bounds (M10.C.4) would be Literature-type T5 records; whether they are catalogued as such is unknown |

**2. Agent architecture.** [PAPER] Endog.-decision vs Endog. non-decision distinguishes what a decision function sets directly from what is computed as a consequence (§3.3.2). A parameter shared by several decision-maker types is declared on each, value listed once in T6a, and never stored on the Environment agent as a convenience (D.1.6). §6.2: RL/LLM agents fit T4 as an input–output contract; the residual hazard, that model versions drift over time and that inference is itself stochastic, cannot be removed by any description protocol; what VISA does is confine the irreproducibility to one named, versioned component.

**3. Interaction structure.** [PAPER] Topology is owned by an explicit Space agent that answers neighbour queries (SensingNet holds A_t; Rebellion Grid pushes vision lists n_{i,t} to each agent as an external effect, App. A.1.4). The sensing layer is generated as getters plus an access-control wrapper that enforces the T3 matrix, with an optional strict mode (D.3.7 step 5).

**4. Initialization.** [PAPER] Initial values are not part of the schedule loop; they are T6a inputs (§3.4.3). Heterogeneous fixed-at-birth traits (risk aversion R_i ~ U(0,1); inherited stride) are Exog.-heterogeneous, not endogenous, because no function on that agent writes them (App. B.1.2). No burn-in concept.

**5. Calibration/validation.** [PAPER] T8 passing conditions in the example are absolute (p > 0.05, β < 0, Table 8). Case studies use invariant checks as T8 rows (A_t + J_t + Q_t = N_A every t; 0 ≤ e ≤ M_max). The AnyLogic case: 29 types/240 functions consolidated to 4 sets/14 functions; 19/19 pass; reproduction blocked by a proprietary movement library and unavailable data, which the authors count as a finding in its own right rather than a failure of VISA (§5, App. C.1.10). No quantitative evidence that VISA beats ODD; the authors list that as future work (§7).

**7. Software engineering.** [PAPER] Authoring order T1→T8 because later tables consume earlier identifiers (D.1.3). Project layout maps tables to files: config (T6a), data loaders (T5), agents (T1/T2/T4), sensing layer (T3), model step loop (T7), collectors honouring Frequency (T6b), post-run validation (T8) (D.3.4). Pitfalls: a wrong exec mode alters results without any visible error; in-place update where T7a says Synchronous; unseeded randomness (D.3.9).

**Not transferable / cautions.** Nothing on emotion, families, small N or long horizons. VISA's own reproduction evidence is 3 seeds with visual overlay, below EPModel's standard. T8 assumes absolute passing conditions; EPModel's direction-only, two-arm, mutation-proved criteria have no VISA slot and are stricter. The four exec modes do not cover EPModel's batched same-tick events (M1.F.8); that would be a fifth mode or a documented composite.

## B. Candidate additions (VISA)

**V1. Specification register for state and mechanisms (extends design lessons §2.9).**
- *Proposed requirement*: The spec's object model (M1) MUST be accompanied by a register in which every state variable of Person, Relationship, Triangle and Family is classified as exogenous-homogeneous, exogenous-heterogeneous, endogenous-decision or endogenous-derived, with the mechanism(s) that write it; every mechanism MUST list what it reads, what it writes on its owner, and what it writes on other objects; and the phase order (M3) MUST list each mechanism with its execution mode (synchronous batch, latency-delivered, slow-tick) and trigger condition. A check MUST fail on any variable with no writer, any mechanism not placed in the phase order, and any mechanism reading a quantity its owner cannot observe.
- *Where*: M1 (objects), M3 (ordering), M10 (config), M16 (documentation); the check in M11.
- *Evidence*: VISA r1, r3, r10, r11, r14, r16 (Tables 9, 11) and the AnyLogic case where the register located the reproduction barrier (§5). ARGUED (no experiment shows the register improves reproduction).
- *What it changes*: adds a documentation rule and a static check; inference validity (reproducibility, locatability of assumptions). Design lessons §2.9 proposed a constants register only; this covers state and mechanisms.
- *Cost/risk*: low, a table plus a script; no conflict with M3.D.4–6, M11.F.9 or M16.B.

**V2. Information-access rule as a testable requirement.**
- *Proposed requirement*: The policy (M4) MUST compute a Person's move from only (a) that Person's own state, (b) that Person's beliefs, and (c) Events delivered to that Person's inbox as target or witness, after per-hop fidelity; it MUST NOT read another Person's true state or an undelivered Event. An acceptance test MUST include a mutant in which the policy reads true state in place of belief and MUST go red on at least one belief-layer criterion.
- *Where*: M4, M11.
- *Evidence*: VISA r14 and the strict-mode sensing layer (D.3.7); ARGUED. Possibly already covered by "agents act on beliefs" (brief); if the spec states it as a rule with a test, mark "already covered".
- *What it changes*: converts an intent into a mutation-provable constraint; theory fidelity (belief ≠ truth) and inference validity.
- *Cost/risk*: low; the mutant is one switch. No conflict.

**V3. Disposition of a Person at death (r8 analogue).**
- *Proposed requirement*: The consolidation module (M6) MUST name the mechanism that removes a Person at death and MUST state where that Person's anxiety, bond energy on each tie, functioning-balance debts, triangle positions and share of the undifferentiation budget go, so that the conservation invariants and "no exit from the field" (I7) remain checkable across a death. If births occur within the 40-year horizon, the creating mechanism MUST be named likewise.
- *Where*: M1, M6, M11 (invariant test spanning a death event).
- *Evidence*: VISA r8/r13 (a variable-count type must have named create/remove functions and their external effects); INFERENCE that the spec leaves this open (the brief mentions mortality but not disposition).
- *What it changes*: closes a possible hole in the invariants; theory fidelity (Bowen: the dead remain in the emotional field through the multigenerational ledger) and inference validity (a leak at death would fake "destroyed" anxiety).
- *Cost/risk*: small if the ledger already absorbs it; needs an owner decision on whether death is an exit.

Already covered, one line each: T6b Frequency per readout ≈ design lessons §2.8 (windowed trajectories); single seeded RNG threaded through all draws ≈ M3.D.4/5; T7 exec-mode as an assumption to test ≈ design lessons §2.1.

---

# Paper 2. Kalluri, "Agent-Based Simulation of Trust Development in Human-Robot Teams" (arXiv 2603.01189)

## A. Report

**1. Simulation formalism.** [PAPER] NetLogo 6.4.0, 33×33 continuous toroidal space; 1 tick ≈ 1 minute; runs of 2,000 ticks (~33 h) (§4.1). Per-tick sequence of nine phases in fixed order: state update (stress, battery, tenure) → task seeking → task execution → collaboration matching → communication → trust update → task completion → task generation → metrics (§4.3). No statement on within-phase agent order, seeding or determinism. Teams of 2–10 agents; scenarios use 5 humans and 5 robots (§6.1).

**2. Agent architecture.** [PAPER] Human state: trust θ_i(t) ∈ [0,100], stress σ_i(t), expertise ε_i, tenure τ_i(t), propensity Φ_i, workload w_i(t). Robot: reliability ρ_j, transparency ψ_j, capability κ_j, warmth ω_j, communication frequency γ_j, all fixed. Task: difficulty d_k, collaboration requirement χ_k ∈ {0,1}, visibility v_k ∈ [0,100] (Table 2). No learning; agent capabilities are fixed for the whole run (§7.4).
- **Trust update rule**: the paper gives NO equation. What it states: parameters trust gain rate α = 1.5, loss rate β = 2.0, asymmetry factor λ = 1.5 (Table 2/3); visibility moderates the size of each trust update (§3.2); transparency changes how much robot actions and communication move trust (§3.2); trust update radius 10, described in Table 3 as the range within which an outcome can be observed, i.e. humans within radius 10 of a task outcome update on it (a witness rule) [INFERENCE from Table 3 wording]; longer tenure makes the trust change after any single outcome smaller (§2.5), rate 0.002 (Table 3). Stress: accumulation η = 0.1, recovery δ = 5.0, threshold 70 (Table 3).
- **Asymmetry mechanism**: negative outcomes are set up to have 1.5 times the effect on trust that positive outcomes have (§5.2). Note an internal inconsistency: β/α = 2.0/1.5 = 1.33, yet λ = 1.5 is listed separately, and Table 3 says β was calibrated so that the asymmetry ratio falls in 1.3–1.7; which quantity implements the 1.5× is not resolvable from the text.
- **Repair mechanism**: transparency and communication are framed as the mechanisms that can, up to a limit, rebuild trust after a violation (§2.1.6); in phase 5 robots communicate with nearby humans with some probability (§4.3), communication radius 5 (Table 3). No equation; the `three strikes` limit (§2.1.4) is cited from the literature, not stated as implemented.

**4. Initialization.** [PAPER] §4.2: trust = initial-trust ± propensity adjustment; expertise N(60,15) bounded [20,100]; propensity N(mean, sd); stress U(0,30); robot reliability = parameter ± N(0,5), transparency ± N(0,5), warmth ± N(0,10), capability U(60,90); task difficulty N(mean,20) bounded [10,100], collaboration Bernoulli(rate + difficulty adjustment), visibility U(40,100).

**5. Calibration and validation.** [PAPER]
- Validation target: Hancock et al. (2021) meta-analytic r values (Table 1; note the paper gives the meta-analysis as 69 studies with N = 7,769 in §2.1.5 and as 142 studies with N = 7,458 in §2.2). Design: for each of 8 antecedents, 10 levels × 50 replications = 500 observations, model r vs benchmark CI (§5.1). Result (Table 4): interval validity 4/8 (reliability 0.59 vs 0.60; transparency 0.38; communication 0.41; propensity 0.15 pass; warmth 0.34, expertise 0.14, tenure 0.06 fail low; collaboration 0.39 fails high vs 0.27 [0.15, 0.38]). Ordinal validity: Spearman ρ = 0.833 on the ranking of effect sizes (§5.1, Fig. 3).
- **OFAT**: 8 parameters, each swept across its full range with others at default; η² on trust / task success / productivity (Table 6): reliability .35/.93/.89; communication .20/.01/.00; transparency .17/.03/.03; collaboration .17/.01/.01; warmth .13/.02/.01; expertise .02/.32/.71; propensity .04/.01/.01; tenure .01/.03/.02.
- **Full factorial**: 3⁴ over reliability {40,70,90%}, transparency {30,60,90%}, communication {20,50,80%}, collaboration {20,50,80%}, 30 replications per cell, n = 2,430 runs (§5.3). ANOVA (Table 7): main effects η² R .171, C .131, T .080, L .072 (Σ .454); two-way Σ .010, three-way Σ .009, four-way .004; residual .524. Only R×T (p = .024), C×L (p = .005), R×T×C (p = .023) significant, each η² < .005: the authors read the effects as mostly additive, with little synergy.
- Scenarios: 5 scenarios × 50 replications, 2,000 ticks (§6.1, Table 8). Table 9: trust 38.2 (Trust Recovery) to 73.2 (Unreliable Robot); task success 33.4% to 63.8%; productivity 2.15 to 4.29; calibration error 8.93 to 52.05; asymmetry ratio 0.069 to 0.552.
- **Per-event vs cumulative asymmetry** (§5.2, Table 5): asymmetry ratio = cumulative trust loss ÷ cumulative trust gain over a run. By reliability: 30% → 0.53 (SD 0.15); 50% → 0.37 (0.11); 70% → 0.22 (0.07); 90% → 0.13 (0.04); overall 0.31 (0.18). None approach the 1.50 benchmark despite the 1.5× per-event rule: the authors conclude that how often events occur moderates how much of the per-event asymmetry shows up cumulatively, and that a per-event asymmetry does not by itself guarantee a cumulative one while repair mechanisms are still operating (Abstract, §7.1). [INFERENCE] Much of this is arithmetic: with success probability p and a 1.5× loss weight, the expected ratio is 1.5(1−p)/p, i.e. 0.64 at p = 0.7 and 0.17 at 0.9, close to the observed 0.22 and 0.13; but at p = 0.3 the expected 3.5 vs observed 0.53 shows that repair, visibility gating and the [0,100] bounds also suppress losses. The paper does not decompose these.
- Variance: 52.4% residual attributed to stochastic dynamics and to factors possibly left out of the model (§7.4); seed variance not separated from anything.

**6. Failure modes / negative results.** [PAPER] 4/8 antecedents outside CI (§7.4). Trust–performance decoupling: highest productivity at lowest trust; highest trust at lowest success (Table 9). `Calibration error` = |subjective trust − objective capability| proposed as a diagnostic distinct from magnitude (§7.1). No learning; cannot validate trajectory shapes, only end-point correlations (§7.4).

**8. Small N, long horizon.** [PAPER] 10 agents, ~33 simulated hours; nothing on years.

**Not transferable / cautions.** Trust in a fixed-reliability robot is not a family tie. The central update equation is absent from the paper, so no functional form can be borrowed. The asymmetry parameter was calibrated to a target (Table 3) and then the target is reported as a validation result (Table 5): this is the practice M11.F.9 forbids, and here it failed anyway and was reframed as a boundary condition on the mechanism. The 3⁴ factorial with 30 replications and n = 2,430 makes every main effect p < .001 regardless of size (design lessons §2.6, "pre-declared minimum effect size").

## B. Candidate additions (Kalluri)

**K1. Ordinal acceptance criteria.**
- *Proposed requirement*: Where the corpus supplies an ordering of effect strengths across three or more mechanisms (e.g. which of several moves or stressors shifts a readout most), an acceptance criterion MUST assert the rank order of the per-seed paired arm differences across those mechanisms, not only the sign of each; the criterion passes when the observed ordering matches the corpus ordering in a pre-declared majority of seeds.
- *Where*: M11 (criteria), Phase E (ensemble runner computes the ranking per seed).
- *Evidence*: §5.1, Fig. 3: interval validity 4/8 but ordinal validity ρ = 0.833; the authors offer the dual test as a methodological template for others to reuse. SHOWN as a technique; its fit to EPModel is INFERENCE (the brief says the corpus supplies "directions, orderings and mechanisms but almost no magnitudes", which is exactly the evidence an ordinal test uses).
- *What it changes*: adds a test class that uses more of the corpus than sign tests do; theory fidelity and inference validity. Mutation-provable: deleting the mechanism ranked first must break the ordering.
- *Cost/risk*: needs ≥3 comparable arms per criterion; no conflict with any rule; no magnitudes assumed.

**K2. Frequency-controlled test of asymmetric mechanisms, and a readout that separates per-event magnitude from event frequency.**
- *Proposed requirement*: For every mechanism whose [I] constants are asymmetric (a loss or escalation rate larger than the corresponding gain or recovery rate), the run log MUST record per-event magnitudes and event counts separately, and the acceptance test for the asymmetry MUST hold event counts equal across the compared arms (or condition on them), so that a cumulative asymmetry is attributable to the per-event rule and not to the frequency of triggering events.
- *Where*: M16 (log fields), M11 (test design), Phase E.
- *Evidence*: Table 5 and §7.1: a 1.5× per-event rule produced cumulative ratios 0.07–0.55, varying monotonically with failure frequency. SHOWN.
- *What it changes*: adds a reporting rule and a test-design rule; inference validity. It also names a readout trap for EPModel: a "cutoff is easier than reconnection" or "anxiety rises faster than it settles" result read from cumulative totals confounds rule and frequency.
- *Cost/risk*: low; counts are already in the event log if M16 logs every Event.

**K3. Belief–truth discrepancy readout.**
- *Proposed requirement*: For each belief in the family belief layer that has a true-state counterpart, the run log MUST allow a signed discrepancy (belief − truth) to be computed per tick, and Phase E SHOULD report its distribution and trajectory per arm alongside the outcome readouts, so that arms in which beliefs and outcomes move in opposite directions are identified rather than averaged away.
- *Where*: M16, Phase E.
- *Evidence*: §6.3, Table 9: trust and task success decouple across scenarios; calibration error 8.9–52.1 distinguishes them. SHOWN in the paper; transfer to the belief layer is INFERENCE.
- *What it changes*: adds a readout; supports the brief's own trap list (counterfeit position-taking, biased differentiation estimates). Zero mechanism change.
- *Cost/risk*: low. Possibly partly covered if M16 already logs belief writes with the true value; check.

Already covered: OFAT/factorial as a constant sweep ≈ design lessons §2.3; witness radius with visibility weight ≈ EPModel witnesses with per-hop fidelity.

---

# Paper 3. Holland, Saynor, Svingen, Luo, "Agent-based dynamics of criminal propensity" (arXiv 2607.29546)

## A. Report

**1. Simulation formalism.** [PAPER] N agents, propensity C_i(t) ∈ [−1,1] (−1 most criminal, +1 least, 0 neutral), discrete t (§3). Each step: a random even number n(t) > 0 of agents form n(t)/2 random pairs; the remaining N − n(t) agents are each randomly allocated an interaction to witness, with no cap on how many witnesses an interaction may have (§3). Every agent therefore acts every step, either as a participant or a witness [INFERENCE from the assignment rule]. A whole reciprocal exchange (push and push-back) plus all witnesses' responses is counted as a single time step, not as three consecutive ones (§3). Updates are deterministic given the random pairing and valence draws; valence of each interaction is Bernoulli(0.5), independent of the agents (§5). Boundedness of C in [−1,1] is a theorem (4.1), not a clamp. Python, 100 agents, 10,000 steps (§5). The regime map (Fig. 6) fixes C(0), the pairings at every step and every interaction's valence, then varies only r and P over a 20×20 grid: 400 runs on common random numbers (§5.2).

**2. Agent architecture (the exact mechanism).** [PAPER; equation text is partly garbled in extraction, reconstructed from the surrounding prose and the Deffuant comparison]
- Per-agent constants: positive reciprocity r_i⁺, negative reciprocity r_i⁻, retribution r_i^e, all in [0,1], fixed; perception of environment P_i ∈ [−1,1], fixed (which the authors identify as the model's main simplifying assumption, §3).
- **Participant rule** (eqs 2–3). Weight w_ij = ± (r_i^±/4)(1 − |C_i|)|C_j − P_i|, sign = valence of the interaction (same sign for both parties; w_ij ≠ w_ji). If r_i^± |C_j − P_i| > |P_i| (the `shock` exceeds the magnitude of the agent's PoE): C_i ← C_i + w_ij |C_j − C_i|; otherwise the **attractor term**. Reading of the top line: the direction of change is the interaction's valence, not the relative position of the two agents; a positive interaction moves both parties in the positive direction and a negative one moves both in the negative direction (§3), which makes one party Deffuant and the other anti-Deffuant. Magnitude = own reciprocity × openness (1 − |C_i|) × shock relative to own baseline × distance between the parties.
- **Witness rule** (eqs 4–5). Witness i of interaction (j,k): w_{i,jk} = ± (r_i^e/4)(1 − |C_i|)·(|C_j − P_i| + |C_k − P_i|)/2, sign = the valence of the witnessed interaction. If r_i^e (|C_j − P_i| + |C_k − P_i|)/2 > |P_i|: C_i ← C_i + w_{i,jk}·(|C_j − C_i| + |C_k − C_i| + |C_k − C_j|)/3; otherwise the attractor term. So a witness (a) uses its own retribution constant, distinct from the participants' reciprocity constants; (b) takes the mean shock over both parties; (c) scales by the mean of three pairwise distances, including the parties' distance from each other; (d) moves in the direction of the valence, with weight |w| ≤ 1/2.
- **Attractor term** (both rules, eq 6): C_i ← C_i + (1 − |P_i|)(1 − |C_i|)(P_i − C_i), a weighted average of C_i and P_i; drift is small when either P_i or C_i is extreme (the authors' gloss: agents at the extremes are stubborn). The threshold |P_i| means an agent with a neutral PoE is moved by any interaction, an agent with an extreme PoE by almost none (§3, §6). Corollary 4.2: C = ±1 iff it was ±1 the step before; ±1 are `sinks` (§6).
- Theorems: 4.3 persistent supra-threshold same-valence interactions → C → ±1; 4.4 persistent sub-threshold interactions → C → P_i.
- **Oscillation condition** (§4.2, eqs 18–20), necessary not sufficient: the shock threshold must be met repeatedly, which requires r_i^± > 0 and P_i ∈ (−r/(1−r), r/(1−r)) for 0 < r ≤ 1/2, and any P_i ∈ [−1,1] for r > 1/2; empty for r = 0; same with r^e for the witness channel. With r, r^e ~ truncated N(1/2) independent, more than half the agents meet the condition per channel and fewer than a quarter cannot oscillate at all. Oscillation is explained as a balance between how often the agent is pushed towards an extreme and how often it is pulled back towards its PoE (§6). Amplitude grows with r and r^e (Fig. 5); the authors state that these two parameters govern the size of the fluctuation, not its direction (§6).

**3. Interaction / population size.** [PAPER] Random mixing, no network. With a universal PoE P: far from 0 → consensus at P (Fig. 3a, within 100 steps); P ≈ 0 → polarisation to ±1 (Fig. 3b): the authors point out that universal neutrality, the condition one would naively expect to yield consensus, is precisely what produces the most division (§7). **How N enters** (§5.2, Fig. 6): polarising band of P is [0, ~0.06] at N = 10 and [0, ~0.04] at N = 100 (axis ranges of Fig. 6a/b); the critical P rises with r and falls with N; near the boundary the outcome flips between polarisation and consensus several times as P changes marginally (Fig. 6a at r = 0.75; 6b at r = 1). Authors' explanation: self-averaging; in a small population a handful of early excursions to the extremes can have an outsized effect on the interaction environment everyone else faces (§6). The polarising band narrows as the population grows and widens as reciprocal and retributive tendencies strengthen (Abstract).

**4. Initialization.** [PAPER] r^±, r^e, P_i, C_i(0) drawn independently from truncated normals; r_i⁺ = r_i⁻ assumed in simulations (§5). With P ≈ 0 and all C_i ≈ 0 polarisation is unlikely because shocks are small; it is the influence of agents whose propensities start as outliers that turns this latent susceptibility into actual division (§6): initial outliers matter.

**5. Calibration/validation.** [PAPER] None: the parameters have no empirical calibration, and the authors say the results are to be read as claims about qualitative regimes only (§6). Fig. 6 is one run per grid point with fixed draws; `inconclusive` is an explicit outcome class when neither consensus nor polarisation occurs within 10,000 steps. Fidelity check practised: two results that RRM already contains (extremity → stubbornness; neutral PoE → maximal exposure) are presented as a check that the formalisation is faithful to the verbal theory, not as new findings (§6); results the verbal theory could not generate are listed separately.

**6. Limitations.** [PAPER] Valence independent of propensity, so no crime rate and no feedback; P fixed, defended as valid over short enough time-scales, of the order of a day; oscillations are individual-level, not the population-level free-rider cycle RRM posits (§2.1, §6). Future: P_i(t) as the mean of network neighbours, with time-scale separation (propensity fast, perception slow); memory expected to reduce extremes (§7).

**8. Small N / long horizon.** [PAPER] N = 10 is one of the two regime maps (Fig. 6a); no long-horizon or emotional content beyond the attractor.

**Not transferable / cautions.** Random mixing with no persistent ties; valence exogenous; a single scalar per agent. The pull to ±1 as absorbing sinks is a functional-form property (design lessons §2.7 class) and should not be imported. Equation (3)'s top line is garbled in the text; my reading (absolute distance, valence sets sign) follows §3's prose and eqs (8)–(9); verify against the PDF before quoting.

## B. Candidate additions (Holland)

**H1. Witness appraisal computed from the witness's own state and its ties to both parties.**
- *Proposed requirement*: A witness's appraisal of an Event MUST be computed from the witness's own state (reactivity, functional level), the witness's ties to BOTH the sender and each target, and the intensity of the exchange between them, using a witness-specific weighting constant [I]; it MUST NOT be a scaled copy of a target's appraisal. The direction of the witness's response MUST follow the move type of the observed exchange.
- *Where*: M1 (Event/witness semantics), M4 (appraisal), M11 (test: a mutant that replaces witness appraisal with the target's appraisal scaled by fidelity must go red on a triangle-formation criterion).
- *Evidence*: eqs (4)–(5): distinct parameter r^e, mean shock over both parties, mean of three pairwise distances including the parties' own distance; §4.2 and §5: this channel alone suffices for oscillation and polarisation. SHOWN formally in their model; transfer is INFERENCE. The brief says witnesses "also appraise" but not how.
- *What it changes*: adds or sharpens a mechanism; theory fidelity (Bowen: the third party's involvement depends on its ties to both members of the anxious dyad, which is what makes triangles rather than diluted dyads).
- *Cost/risk*: moderate: one more appraisal path with its own [I] constant; must respect per-hop fidelity and no LLM; no conflict.

**H2. Separate seeded random streams per stochastic source (common random numbers across arms).**
- *Proposed requirement*: The engine MUST draw each stochastic source (exogenous spell timing and content, policy tie-breaking, any initial-condition sampling) from its own seeded stream derived deterministically from the run seed, so that two arms of a counterfactual run under the same seed receive the identical exogenous sequence even when the intervention changes the number of draws in another stream. Phase E MUST report per-seed paired differences computed under this pairing.
- *Where*: M3 (determinism, D.4/D.5), Phase E.
- *Evidence*: §5.2, Fig. 6: 400 runs sharing C(0), pairings and valences so that only r and P differ; SHOWN as practice, the stream-separation design is INFERENCE. Design lessons §2.6 asks for per-seed paired differences but does not state how to keep arms paired when draw counts diverge.
- *What it changes*: strengthens determinism into arm-comparability; inference validity (removes seed-desynchronisation noise from the reported difference).
- *Cost/risk*: low if done before Phase B; retrofitting is expensive. Consistent with M3.D.5 and M16.B.

**H3. Per-seed trajectory regime classification with an explicit "inconclusive at horizon" class, and a non-absorbing-bound check.**
- *Proposed requirement*: Phase E MUST classify each bounded slow and fast state trajectory per seed as settled at a bound, settled at its attractor, oscillating, or inconclusive at the horizon, and report the fractions per arm; a directional criterion MUST NOT count inconclusive seeds as passes. An acceptance test MUST show that no state bound (functional_level, anxiety channels, conductance) is absorbing unless the theory says it is (cutoff may be; a test starting a Person at a bound must show the mechanism can leave it).
- *Where*: Phase E, M11, M6 (bound semantics).
- *Evidence*: Fig. 6's three-class outcome including `inconclusive`; Corollary 4.2 (±1 are sinks) and §6 (an agent near an extreme is much harder to move back). ARGUED/SHOWN in their model; the absorbing-bound risk for EPModel is INFERENCE, extending design lessons §2.7 (absorbing zero) to absorbing extremes.
- *What it changes*: adds a reporting rule and a test; inference validity (design lessons §2.2: at N = 12 there are metastable states with lifetimes, so "settled" needs a horizon-qualified definition).
- *Cost/risk*: low; classification is a post-run script over the M16 log.

**H4. Endogenous-only control arm.**
- *Proposed requirement*: Phase E SHOULD include an arm with all exogenous spells removed (societal anxiety constant, no nodal events beyond those the calendar generates from within), so that persistent non-settling in the reference family can be attributed to the relationship mechanisms rather than to external forcing.
- *Where*: Phase E.
- *Evidence*: §6: the model is claimed to demonstrate persistent non-convergence in the absence of any external driver, as a substantive claim about the mechanisms. ARGUED; INFERENCE for EPModel.
- *What it changes*: adds a control arm complementary to the no-interaction arm (design lessons §2.6); inference validity.
- *Cost/risk*: low; one config switch.

Report only, no candidate: the N = 10 vs N = 100 result (Fig. 6) is direct evidence that at EPModel's population size the polarise/consense boundary is wider and flips under marginal constant changes; it supports the constant-sweep requirement of design lessons §2.3 and adds nothing new to it. The attractor's "extremity → stubbornness" threshold is not transferable as is: in Bowen, low differentiation increases reactivity, so susceptibility is monotone in differentiation rather than symmetric about a neutral point.

Files read: txt/2026_VISA_Description_Protocol_He_2607.28027.txt ; .../2026_Trust_Development_Small_Teams_ABM_Kalluri_2603.01189.txt ; .../2026_Criminal_Propensity_Witnesses_ABM_Holland_2607.29546.txt

---

# Part 4

# Reader report: Prasad 2026; Wang et al. 2026; Li et al. 2026

Read in full from pdftotext. Figures survive only as captions; Li et al.'s Figs 2–4 are garbled and results from them rest on prose. Tags: [PAPER] = stated/shown in the paper; [INFERENCE] = mine.

---

# 1. Prasad 2026, "A Transdiagnostic Space of Disorder-Like Phenotypes in RL Agents" (arXiv 2607.07753v2)

## A. Report

**2. Agent architecture / appraisal-signal architecture** [PAPER]
- AG-PPO (Fig. 1, Method): a shared conv encoder feeds an actor (3 actions) and a critic; the critic additionally receives a 6-d appraisal vector ζ. A separate next-reward-estimation (NRE) network predicts r_t from (o_{t-1}, a_{t-1}) and supplies the anticipation appraisal.
- Six appraisals, each in (0,1), computed online from state the agent already has (Eqs. 3–8; supplement `Formulation`): motivational relevance (Manhattan distance to goal), goal congruence (Euclidean distance), certainty (complement of squashed policy entropy), novelty (squashed KL of policy from uniform), coping potential (fraction of known threats *outside* the egocentric view), anticipation (1 − NRE error).
- ζ plays two roles: concatenated into the critic (value estimation is appraisal-informed) and used in reward shaping (Eq. 2). Only anxiety and mania act *through* ζ; the other five knobs act on environment reward directly, while the critic nonetheless continues to receive ζ in every configuration.
- Limitation stated: certainty and anticipation depend on the evolving policy, so the shaped reward is nonstationary, although the authors report that in practice the phenotypes settle (Limitations).

**The seven knobs** (Table 1, Table 5, Method) [PAPER]. Exactly one term of Eq. 2 (or γ) is active per disorder.
| Knob | Mechanism | Dose grid |
|---|---|---|
| Anxiety | penalty ∝ (1 − coping potential): a threat in view is penalised → avoidance | w = 0.01, 0.03, 0.1, 0.3 |
| Mania | penalty on *high* coping potential → threat seeking; same knob, opposite sign | same |
| OCD | checkpoint-return bonus ελ^k on k-th return, λ = 0.5 (diminishing reassurance), bounded total ε/(1−λ) | 0.1–0.6 |
| Depression | per-step effort cost on forward moves | 0.01–0.1 |
| Impulsivity | γ = 1 − ε (steeper discounting) | ε = 0.05–0.4 |
| Addiction | non-habituating drug-tile bonus whose cumulative value can exceed the goal | 0.01–0.1 |
| PTSD | shock on a trauma tile on the short route; long safe detour exists | 0.05–0.4 |

**5. Assay design** [PAPER]
- One primary assay per disorder fixed before running (risky-goal choice, death rate, checking rate, forward-action fraction, near-reward choice, drug occupancy, trauma distance); secondary assays in supplement (thigmotaxis, freezing, stereotypy, turnarounds, revisits, visitation entropy, stress index with weights 0.25/0.05/0.1/0.2/0.35/0.05, which the authors say were inherited from their base model).
- 10 seeds per configuration (30 for anxiety; 5 for rescue; 15/10 for exposure); mean ± 95% CI across seeds; 40 evaluation episodes on held-out seeds under the stochastic policy. 1,375 runs total.
- Four controls: standard PPO, critic-noise (random vector replaces ζ), PPO+RND, appraisal critic with no shaping (the ε = 0 point of each curve). None reproduces a phenotype (Table 3, Table 9); RND over-explores into lava (success 0.06 on LavaGap).
- Inclusion criterion fixed before analysis: only runs with success ≥ 50% enter symptom analysis; every aggregate is reported together with the count of seeds that contributed to it.
- Reproducibility: seeds, dose grids and criterion fixed in configuration; every run writes a self-describing result file; all tables/figures regenerated by one script.

**Dose–response** (Table 2, Table 7) [PAPER]: anxiety risky-goal choice 0.73 → 0.43 → 0.20 → 0.10 → 0.00 with task success preserved; mania death rate 0 → 0 → 0 → 0.26 → 0.70; OCD checking 0 → 0.00 → 0.16 → 0.26 → 0.29; depression forward fraction 0.81 → 0.73 → 0.71 → 0.00 → 0.00; impulsivity near-reward 0.46 → 0.61 → 0.56 → 1.00 → 0.94; addiction drug occupancy 0 → 0.02 → 0.00 → 0.50 → 0.76 (vulnerability threshold near ε = 0.05); PTSD trauma distance 2.20 → 5.76. Mania, OCD, depression are monotone on all four threat grids.
- [INFERENCE] The paper's claim that all seven disorders show graded, monotone dose–response is not literally true of Table 2: impulsivity 0.61 → 0.56 and 1.00 → 0.94, addiction 0.02 → 0.00, are non-monotone within CI. Several curves are step-like (depression 0.71 → 0.00 between ε2 and ε3), which is threshold behaviour, not a smooth dose–response.

**Appraisal-contingency ablation** (Fig. 3, Fig. 12) [PAPER]: holding the anxiety penalty at the severe dose, the appraisal entering it is corrupted three ways that preserve penalty magnitude and frequency: *shuffle* (permute across the batch each step), *random* (uniform noise), *shift* (redirect to another appraisal dimension). Intact knob: risky-route avoidance 1.00, threat distance 6.64. All three corruptions: avoidance 0.00–0.20, threat distance 3.0–3.5, non-overlapping CIs over 10 seeds. On LavaGap the assay is saturated and no arm differs, reported as the assay's limit.

**Negative result** (Experiments; supplement A.10) [PAPER]: penalising low certainty *reduced* checking. A checkpoint bonus without habituation was bistable: ignored below a threshold, an unbounded loop above it, with no intermediate graded regime. Diminishing reassurance (λ^k) produced the graded curve.

**Remission and graded exposure** (Table 4, 11, 12; Fig. 13) [PAPER]
- Warm-start from the severe model; continue training with the knob removed vs kept, 5 seeds. Mania, OCD, addiction remit to 0.00 (controls 0.53, 0.23, 0.68). Anxiety and PTSD: feared-route use 0.20 ± 0.39 passive vs 0.00 control; depression 0.20 ± 0.03 (partial); impulsivity 0.00 vs 0.11 ± 0.22 (resists). Authors' explanation: because the safe or myopic policy continues to succeed, the agent is never exposed again to the evidence that would disconfirm it.
- Exposure with response prevention: a penalty (coef. 0.5) on the *avoidance* route annealed to zero. Feared-route choice rises to 0.93 ± 0.13 (anxiety, 15 seeds) and 0.90 ± 0.20 (PTSD, 10 seeds); persists after the penalty fades. A simple reward for reaching the feared route failed because an avoidant policy never visits that route, so the reward is never collected.
- [INFERENCE] The passive-removal CIs (±0.39 at n = 5) include zero; the remit/resist dissociation is shown clearly for the three remitting disorders and for exposure, but the `resists` cells rest on 5 seeds with wide intervals.

**Non-additive interaction** (Tables 13–14, Fig. 14; 10 seeds per cell) [PAPER]
- Mania × impulsivity: mania alone (dose 0.1/0.3) gives death rate 0.70/0.82; any impulsivity (0.2 or 0.4) gives 0.00 in every cell. Max interaction residual 0.82. Explanation: dying in lava requires a committed multi-step approach a myopic agent will not undertake.
- Anxiety × depression (readout = risky choice): (0,0) 0.50; depression 0.02 with anxiety 0 gives 0.80; anxiety 0.1 or 0.3 gives 0.10/0.00; depression 0.05 gives 0.00 everywhere. The authors' reading is that severe depression masks the anxiety readout: the agent stops acting, so avoidance cannot be expressed. Residual 0.50.

**Other results** [PAPER]: PCA of the assay vector separates disorders without labels (silhouette 0.31, Fig. 8). Fine-tuning from the healthy checkpoint with a knob on re-creates each phenotype in 150k steps (a quarter of the budget), 300k for mania/depression/addiction. Critic-representation CKA vs healthy: checking 0.56, mania 0.75, anxiety 0.87. 3D MiniWorld transfer: depression and addiction dissociate cleanly; anxiety is a null because the baseline CNN agent is already fully avoidant (safe-route 1.00). Main text says 5 seeds / 60 runs for MiniWorld; supplement says 3 seeds / 36 runs (Table 15, `Compute`) — internal inconsistency.

**6. Limitations stated** [PAPER]: the phenotypes are offered as computational analogues of the disorders, not as clinically equivalent conditions; labels not validated against human or animal data; affective geometry depends on the chosen appraisal taxonomy; the 2D projection discards compulsivity.

**Judgement: what transfers to testing an anxiety/reactivity parameter in a rule-based model** [INFERENCE]
- The RL machinery (learned value, PPO, CKA, PCA of learned phenotypes) does not transfer; EPModel has no gradient learning and anxiety is a conserved state, not a reward term.
- What transfers is the *experimental grammar* around a single graded parameter: (a) a pre-declared primary readout per parameter; (b) a ≥4-level sweep with a monotonicity claim, not just two arms; (c) a matched-magnitude contingency-severing control, which is stronger than delete-the-mechanism because it rules out the possibility that any input of that size would do; (d) a pre-fixed inclusion rule with contributing-seed counts; (e) reporting saturated readouts as assay limits; (f) 2-D dose grids with an additivity residual; (g) the general lesson that a relief/bonus mechanism without habituation is bistable.
- The remit/resist result maps structurally onto EPModel: the automatic channel *learns* (family style) and DISTANCE/CUTOFF remove the stimulus that would disconfirm the learned response. If that is implemented, patterns should persist after a spell ends without any persistence mechanism being written in, which is a testable emergent property and a mutation target (disable learning → pattern remits). The anxiety × depression masking (a withdrawn agent cannot express avoidance) is the same class as the brief's *overt emotionality peaks mid-scale* readout trap.

**Not transferable / cautions**: the seven disorders themselves; anything requiring training budgets; the claim of monotonicity is stronger than the tables support; rescue statistics rest on 5 seeds; MiniWorld seed count is inconsistent between main text and supplement.

## B. Candidate additions

**P1. Graded-parameter monotonicity tests.** For each Person parameter the theory treats as graded (basic_level, chronic anxiety), M11 SHOULD include at least one test that sweeps the parameter over ≥4 declared levels across the ensemble and asserts a monotone ordering of a pre-declared primary readout, with the ordering violated in the mutant. Where: M11 (+ Phase E runner). Evidence: Table 2/7, SHOWN in RL; transfer is INFERENCE. Changes: adds a test class; improves theory fidelity (the corpus supplies orderings, and a two-arm direction test cannot detect a non-monotone or threshold shape such as depression's 0.71 → 0.00). Cost: ~4× ensemble runs per parameter; no conflict with M3/M11 rules. Distinct from design-lessons §2.3 (which asks whether a *direction* survives a constant sweep, not whether the response is ordered).

**P2. Matched-magnitude contingency-severing mutants.** For each M4 appraisal input that the spec says selection depends on, the mutation suite SHOULD include a variant that preserves the input's per-tick magnitude distribution but destroys its state-contingency (seeded permutation across persons or ticks), and the affected acceptance criterion MUST go red under it. Where: M11. Evidence: Fig. 3 / Fig. 12, SHOWN. Changes: adds a test; inference validity (rules out the alternative that any load of that size produces the pattern). Extends design-lessons §2.6 `sever-the-input` (EconAgent removed the input; Prasad keeps its size). Cost: low; permutation must use the run seed to keep M3.D.5.

**P3. Persistence-after-spell test with a learning-off arm.** M11 SHOULD include a test that a relational pattern established during an exogenous anxiety spell (DISTANCE frequency, conductance loss, CUTOFF) persists after the spell ends in the arm where the automatic channel learns, and remits in the arm with learning disabled; the report MUST show the time-to-remission distribution per seed. Where: M11, with the learning switch in M10. Evidence: Tables 4/11/12, SHOWN in RL; the mechanism's presence in EPModel is INFERENCE. Changes: adds a test of an emergent property (no persistence rule is written in) and a mutation target; theory fidelity. Cost: low. Caution: Prasad's `resists` cells have wide CIs; the test's value is in EPModel's own ensemble, not in the precedent.

**P4. Readout-saturation rule.** Every acceptance test MUST record the baseline arm's position within the readout's attainable range, and a null result on a readout whose baseline sits at a bound MUST be reported as an assay limit, not as evidence about the mechanism. Where: M11 + M16 reporting. Evidence: LavaGap ablation and MiniWorld anxiety nulls, SHOWN. Changes: reporting rule; inference validity. Cost: trivial.

**P5. Pairwise additivity residual.** Phase E SHOULD, for parameter pairs the theory claims interact (e.g. chronic anxiety × basic_level; functioning_balance × conductance), run a 2-D grid and report the maximum residual against the additive prediction from the two 1-D sweeps. Where: Phase E. Evidence: Tables 13–14, SHOWN. Changes: reporting rule over an existing sweep; partly covered by a Latin-hypercube sweep (design lessons §2.11), which samples the joint space but does not report interaction explicitly. Cost: moderate (grid size).

**P6. Habituation on repeated relief.** Any M4 term that pays anxiety relief on a repeated identical move within a window (reassurance-seeking PURSUE, checking-type OVERFUNCTION) SHOULD decay geometrically with repetition count, or be shown by sweep not to be bistable (ignored below a threshold, absorbing above it). Where: M4, constant in M10 labelled [I]. Evidence: negative result, SHOWN. Changes: functional-form guard; theory fidelity and inference validity; same class as design lessons §2.7. Cost: low. Conditional: I do not know whether such a term exists in v2.

Already covered: self-describing per-run files and single-script regeneration (M16 run log); pre-declared direction per test (M11); `never a single run`.

---

# 2. Wang et al. 2026, "Do AI Personas Grow?" (arXiv 2608.06485v1)

## A. Report

**2/4. Design and initialization** [PAPER] (§3, App. B)
- 100 personas: 2 genders × 5 regions × 10 personality types; each type sets one Big Five trait `very high` or `very low`, others moderate, rendered as behavioural descriptions with no trait labels. 11 life events (6 occupational, 4 social, 1 health) with expected direction priors from Specht (2017) (Table 1); 27 of 55 event–trait pairs have a definite prior.
- Pipeline per persona × event: BFI-44 baseline (temp 0) → event notification + first-person reflection (temp 0.7) → BFI-44 again in a 4-message context. Δ = post − baseline. 11 main models + 3 open-weight; ~12,100 trajectories; each condition starts a fresh conversation. Non-reasoning mode.

**5. Validation** [PAPER] (§3.6, §6, Tables 5, 9, 10; 8 models)
- No-event retest floor: 0.025–0.080 BFI units; event+reflection 0.100–0.237, 1.6×–9.0× above floor; paired excess CI above zero for all 8.
- Paraphrase robustness: sign agreement 80.0–92.7% across the 55 cells; Spearman 0.825–0.956.
- Convergence with scenario decisions (counterbalanced parallel forms): ρ = 0.003–0.105; sign agreement 48.4–62.7%; only 3 of 8 CIs exclude zero.
- Retention after three unrelated turns: ρ 0.329–0.713; 62.6–85.3% keep direction.
- Statistics: persona-level direction classification with ε = 0.1; Wilson CIs; BH-FDR; persona-cluster bootstrap.

**Key results** [PAPER]
- RQ1: movement is widespread but untargeted: median pct_moved 0.44–0.84 for pairs with a human prior vs 0.42–0.82 without; within-model gap < 0.05 (§4.2, Table 11).
- RQ2: DC%pair 48.1–70.4%; of 27 definite pairs, 14 match and 13 reverse (Table 3). Retirement reversed by every model (match 0.0–38.9%, median 11.5%). Agreeableness weakest (30–58%), which the authors interpret as a built-in bias toward making personas more agreeable in the wake of major events. Magnitude: only 11.0–16.4% of responses fall in the human band [0.035, 0.14]; 20.8–40.2% reversed; 15.1–54.0% under-shift; 9.9–31.6% overshoot. Fig. 1: median |Δ| = 0.02, roughly an order of magnitude (~10×) below the human band.
- RQ3: across-strata SD median 0.044; 93.2% of cells < 0.10; no gender or region Fisher test survives FDR.
- RQ4: σ_LLM centred at 0.19 vs human 0.5–0.8; 99.8% of cells below 0.5; baseline across-persona SD ≈ 0.7, so personas are well differentiated at baseline but respond alike.
- BFI-Adapt composite 0.044–0.348 (Table 2); pooling across pairs with opposite priors reorders 9 of 11 models (Table 13).
- Limitations (§8): pre–post only; cannot see acute-to-adaptation trajectory; delayed measurement is three turns, which the authors acknowledge is nowhere near the multi-year horizons of the human studies.

**Implications for the owner's LLM-agent question** [INFERENCE]
- Holding a persona: yes, at baseline (SD 0.7 between personas, κ̄ 0.58–0.86 item stability).
- Changing over time: the change is real (above retest floor, paraphrase-stable, retained over three turns) but generic, tiny, homogenised, and biased toward agreeableness. A persona given a stressor moves the way all personas move, not the way that persona would. Neuroticism does rise after negative events (chronic illness N 49/14, divorce 66/14), so *more reactive after a stressor* is reachable, but *less agreeable after a stressor* mostly is not (13 of 27 reversals cluster on unemployment, divorce, new relationship).
- The self-report/behaviour gap (ρ ≤ 0.105) means a BFI-type readout of an LLM persona says little about what it would *do*; a maturity readout would need a behavioural instrument. This reinforces design lessons §3.4.
- The paper's control structure (retest floor, paraphrase, behavioural convergence, delayed retention) is the minimum protocol before any claim that an LLM persona "changed".

**Not transferable / cautions**: nothing about the rule-based engine; human priors are coarse meta-analytic bands; every step is a single API turn with no interaction between agents; horizon is three turns.

## B. Candidate additions (exploratory LLM line only; none for spec v2)

**W1. Change-claim protocol for LLM personas.** Any claim that an LLM persona's maturity/reactivity changed MUST be reported against a no-event retest floor for the same persona and model, under at least one independent paraphrase of the stressor, and with a behavioural (closed-action) readout alongside any questionnaire readout. Where: exploratory line notes, not M1–M16. Evidence: §6, SHOWN. Changes: inference validity for the exploratory question. Cost: 3–4× calls. No conflict with M3.D.6 because the LLM stays outside the engine.

**W2. Heterogeneity floor.** An LLM-persona experiment SHOULD report the between-persona SD of the response and treat SD collapse (Wang: 0.19 vs baseline 0.7) as a failure of the persona layer, before interpreting any mean shift. Evidence: RQ4, SHOWN. Already partly covered by design lessons §3.2 (homogenisation); the specific test (compare response SD to baseline SD) is new.

---

# 3. Li et al. 2026, "Analyzing and Correcting Benevolence Bias in LLMs" (arXiv 2608.24912v1)

## A. Report

**Design** [PAPER] (Methods; Fig. 1)
- 406 value-laden items from ANES, GSS, WVS and a prospect-theory replication, decoupled from survey context by an LLM rewrite, labelled into six categories (prosociality, social desirability, harm aversion, benevolent interpretation, emotional softening, fairness optimism) via LLM pre-screen → expert review → blinded doctoral-rater consensus.
- Human reference: real respondents' demographic profiles condition the LLM; the synthetic PMF is compared with the empirical human distribution. Metrics: BTB (weighted mass shift toward the benevolent pole; Eq. 12) and BWR (per-question win rate vs the human answer; Eq. 13). 18 models.

**Key numbers** [PAPER]
- 83/108 BTB cells positive, 75/108 BWR > 0.5; mean BTB 0.027, BWR 0.527. Social desirability positive for all 18 (BWR 0.565), harm aversion 17/18 (0.569); emotional softening no shift (BWR 0.494) (Table 1).
- Grows with size in Qwen3 0.6B → 32B and Qwen2.5; instruction tuning raises BTB at 4 of 5 sizes; capability explains little (R² = 0.16); reasoning mode reduces but leaves a residual. Rankings agree across surveys (ρ 0.54–0.63).
- Language (English/Chinese) and framing change size, never sign; direct > role-play > prediction.
- Malicious persona (antisocial, self-interested, harm-tolerant rewrite, 3 models): lowers win rate on fairness optimism, emotional softening, benevolent interpretation, but is unable to drive answers beneath the human baseline on social desirability, prosociality, harm aversion; Qwen3-32B stays above 0.5 on all four datasets.
- Contrastive calibration: divide the persona-conditioned next-token distribution by the neutral-persona distribution^α and renormalise (Eq. 1); α ≈ 0.5 brings all six categories into a band around the human baseline; α = 1 overshoots. Needs logits.
- Self-correction prompt: directed but partial; stays above 0.5. Temperature: BTB flat within ±0.01.
- Limitations: multiple-choice items only; Western/global datasets; two languages; alignment recipe not traceable.

**Implications for the owner's question** [INFERENCE]
- The `15-year-old brat` is precisely a persona at the less-prosocial, more harm-tolerant end. The malicious-persona result says aligned models can reach that end on emotional softening and fairness optimism (anger, cynicism are producible) but not on prosociality, social desirability or harm aversion, and the ceiling is a property of post-training that prompting does not move. So an LLM adolescent can be sullen but will not be reliably unkind or careless of harm.
- The one thing that did move the distribution is output-level: contrastive calibration against a neutral-persona call. That requires a closed option set with logits, which EPModel's nine moves are, and a human reference distribution, which EPModel does not have. The neutral-persona call is itself a no-persona control arm, the same logic as the design-lessons no-interaction control.
- Emotional softening showing *no* mean bias is a small favourable point for reactivity personas, with the caveat that it was measured on survey items, not on conflict behaviour.

**Not transferable / cautions**: survey-item measurement; the human reference is demographic-profile conditioning, so the `human baseline` is itself LLM-mediated at the individual level; nothing about interaction, memory, or time.

## B. Candidate additions (exploratory LLM line only)

**L1. Persona-range ceiling test before any reactive persona is used.** Before an LLM persona is used to represent a low-maturity agent, the experiment MUST run a deliberately antisocial rewrite of the same persona on a closed-choice battery and report whether it crosses the neutral-persona baseline on prosociality and harm aversion; if it cannot, that region of the maturity scale is declared unreachable for that model. Where: exploratory line. Evidence: malicious-persona result, SHOWN. Changes: a feasibility test; inference validity. Cost: low.

**L2. Neutral-persona contrast over the closed move set.** If an LLM is ever used to propose a distribution over the nine moves (design lessons §3.6, *estimate a distribution, then let the engine sample*), the pipeline SHOULD obtain a matched neutral-persona distribution per call and report the persona-specific component (Eq. 1 with α declared and labelled [I]), keeping the sample draw in the seeded engine. Where: exploratory line; interface note for M3.D.6. Evidence: Eq. 1 and the α sweep, SHOWN on surveys; transfer to moves is INFERENCE. Caution: without a human reference, α cannot be calibrated, so the correction is a diagnostic, not a fix. No conflict with determinism if the LLM outputs are cached as inputs.

Already covered: sycophancy/agreeableness pull (design lessons §3.2); temperature and prompt as experimental factors (§3.3); LLM-as-judge (§3.4).

---

## Summary across the three

Prasad supplies method for the rule-based engine: a per-parameter dose sweep with a pre-declared readout, a matched-magnitude contingency mutant, a persistence-after-spell test with a learning switch, a saturation rule, and an additivity residual (P1–P6). Wang and Li answer the exploratory question in the negative for the reactive end of the scale: aligned LLM personas differ at baseline, drift generically and slightly toward agreeableness after events, respond alike across personas, express self-reported change weakly in behaviour, and cannot be pushed below the human baseline on prosociality or harm aversion by prompting. Neither addresses maturation over years; Wang's horizon is three turns. Both support M3.D.6.

Files read: txt/2026_Transdiagnostic_Disorder_Phenotypes_RL_Prasad_2607.07753.txt ; .../2026_Do_AI_Personas_Grow_Wang_2608.06485.txt ; .../2026_Benevolence_Bias_Li_2608.24912.txt

---

# Part 5

# Paper 1: Ye, Cao, Chen & Ferrara (2026), "Stop Drawing Scientific Claims from LLM Social Simulations Without Robustness Audits" (TRAILS), arXiv 2605.18890v1

## A. Report

**1. Simulation formalism** [PAPER]
- Position paper with two controlled case studies (§4). Shared protocol (App. C.1): gpt-5.2 primary; replicated on claude-haiku-4-5, gemini-2.5-flash, deepseek-v3; N = 30 independent runs per condition with distinct seeds; the simulation run is treated as the unit of analysis; temperature 0.3, top-p 0.95; two-sided Mann–Whitney U chosen because several outcomes are bounded, discrete, or show no variance at the run level; Holm correction within each metric; Cohen's d reported with 0.2/0.5/0.8 as descriptive guides.
- Prisoner's Dilemma (§4.1, C.2): T = 10 rounds, simultaneous C/D, canonical payoffs (3,3)/(0,5)/(5,0)/(1,1); single-agent vs four fixed policies (TitForTat, Random, AlwaysCooperate, AlwaysDefect) and two-agent LLM-vs-LLM. Prompt order randomized across rounds so as to reduce persistent positional bias; unparseable response after one retry falls back to DEFECT (C.2.1, E.1.7).
- Echo chamber (§4.2, C.3): 100 agents, stance on AI regulation on a 5-point scale, **stance frozen for the whole run** so perturbations test structural echo chambers (who interacts with whom) rather than belief updating; 15 rounds; fixed power-law degree sequence (mean degree 8, exponent 2.4); activation probability 0.3 per round with at least one agent forced active; neighbours-only feed of 5 posts; memory window 2 rounds; actions POST/REPOST/REPLY/DO NOTHING; invalid targets converted to post or silence.
- Structural perturbations built to isolate one factor: homophily varied by degree-preserving double-edge swaps into bands h ∈ [0.02,0.10], [0.12,0.20], [0.22,0.30] with the degree sequence held fixed; hub assignment varied while holding both the degree sequence and the medium homophily band fixed (C.3.3).

**2. Agent architecture** [PAPER] LLM agents with persona prompt, JSON action output. Nothing rule-based. §3.1 distinguishes **design-level** perturbations (model, topology, protocol) from **representation-level** perturbations (same setup, different text presentation).

**3. Interaction / network** [PAPER] Echo chamber results (§4.2, gpt-5.2):
- Homophily (P1): final interaction-network stance assortativity 0.142 → 0.247 → 0.287, all pairwise p ≤ 0.002; weighted same-group edge ratio 0.462 [0.454, 0.470] (mostly cross-stance) vs 0.542 [0.534, 0.551] and 0.538 [0.522, 0.553] (majority within-stance). The authors note that a modest rise in initial network assortativity suffices to cross 0.5.
- Hub assignment (P2): no significant change in assortativity; same-group ratio > 0.5 in every condition, strongest with pro-regulation hubs 0.759 [0.739, 0.779], weakest with anti-regulation hubs 0.690 [0.672, 0.707]. Hubs change how strong the effect is, not whether it occurs.
- Activation probability 0.3 vs 0.5 and memory window 2 vs 4 rounds: no significant effect on either metric. Feed size 5 → 10: assortativity 0.247 → 0.277 (p = 0.014); same-group ratio 0.542 → 0.568 (p = 0.003) (P3–5; App. D.2, Fig. 8).
- Cross-model echo-chamber results exist only as boxplots (Figs 21–25); no numbers in text.

**4. Initialization** [PAPER] Only in the sense of `initial network homophily` and `hub assignment` above, plus the TRAILS-R initial-seed-turn perturbation (same stance, slightly different opening statement) (Table 3). [INFERENCE] The homophily result is an initial-condition sensitivity: a small shift in initial assortativity moved the system across a qualitative boundary (0.5 ratio).

**5. Calibration / validation / robustness** [PAPER]
- PD persona format (P1, §4.1, Fig. 2): three semantically equivalent formats PLAIN (prose), DESCRIPTIVE (bullets), TABULAR (key–value). Single-agent vs AlwaysCooperate: TABULAR earns 1.46 more points/round than DESCRIPTIVE (95% CI [1.14, 1.78], d = 2.30, p < 0.001) and 2.00 more than PLAIN (p < 0.001); TABULAR least cooperative against all four policies. Two-agent (both agents same format): DESCRIPTIVE cooperates **76 pp** less than PLAIN and 73 pp less than TABULAR (p < 0.001).
- Per model, same persona-format perturbation, two-agent cooperation gap (Finding Summary, §4.2; App. D.3): gpt-5.2 76 pp; claude-haiku-4-5 ≈ 77 pp; gemini-2.5-flash ≈ 36 pp; deepseek-v3 ≈ 1 pp. The authors conclude that model identity is itself a dimension of robustness.
- Game framing (P2, CANONICAL / MORALIZED `cooperate fairly`/`exploit` / RISK `safer but vulnerable`): MORALIZED raises cooperation vs both others; in two-agent also raises payoff. Magnitudes only in Fig. 6 (not recoverable from text).
- Memory representation (P3, 2×2 table/narrative × ±summary statistics): most shifts in payoff stay under 0.2 on the 0–5 scale and most shifts in cooperation rate stay under 0.1.
- Summary: sensitivity is spread unevenly across dimensions (persona format equilibrium-flipping; framing, homophily, hubs, feed size smaller but consistent; memory representation, memory window, activation probability small or null) and across model families.
- Three prioritization heuristics (§5): align audits with the claim's mechanism; a sweep over several reasonable alternatives on one axis is stronger evidence than just two prompt variants; audit across model families before claiming generality.
- Claim ladder (§3.1): exploratory probe → mechanism claim → policy/intervention claim, each requiring stronger audit; simulation types goal-structured / theory-guided open-ended / emergence-driven open-ended; domain stakes as third axis.

**6. Failure modes / limitations** [PAPER] Representation-level sensitivity is particularly easy to miss because most studies take for granted that equivalent textual representations behave the same way (§3.1). If a finding fails a TRAILS-R perturbation the authors say it should be reported as sensitive to the interface. §6 concedes that ABMs likewise depend on their initialization and parameters, and points to multiverse analysis and specification-curve analysis as the social-science analogues. Cost objection answered by grading audits to claim strength and by transparently reporting which dimensions were tested and which were not audited.

**7. Software / reporting** [PAPER] App. C.4: decision-level rows (condition, seed, round, actions, payoffs, parse-validity indicator, parse note, raw output, full prompt) plus run-level summaries, run metadata, final agent states. Reporting call (§6): state scenario, stakes and claim type; report the perturbations tested, the findings that stayed stable, the findings that proved sensitive, and the dimensions left unaudited. Code released.

**8. Small-N / emotion / family / long horizon** [PAPER] Nothing. Horizons are 10 and 15 rounds; two-agent PD is the only small-N case.

**The full TRAILS taxonomy (Tables 1–3)** [PAPER]

TRAILS-D (design-level), three levels, eight dimensions, with a `what to perturb` column:
- Micro / **Model substrate**: model family, size, base vs instruction-tuned, alignment, temperature, top-p, random seed, safeguards.
- Micro / **Agent specification**: demographics, ideology, personality, goals, prior beliefs, real-data-grounded vs synthetic personas, relationship attributes.
- Micro / **Internal state and cognition**: beliefs, attitudes, emotions, needs, moral values, reasoning style, reflection rules, belief updating, preference updating.
- Micro / **Memory and temporality**: last-k memory, summarized memory, episodic memory, retrieval rules, forgetting, reflection frequency, temporal granularity.
- Meso / **Interaction protocol**: turn order, fixed or free turns, dyadic or group interaction, available actions.
- Meso / **Intervention design**: moderator type, target selection, timing, frequency, rule-based vs LLM-based intervention.
- Macro / **Environment structure**: network topology, feed ranking, recommendation system, moderation rules, platform affordances, institutional rules.
- Macro / **Population and scale**: number of agents, demographic distribution, ideological balance, heterogeneity, real vs synthetic population construction.

TRAILS-R (representation-level), five categories, 14 perturbations:
- **Representational format**: formatting (prose vs bullets vs table, bold, numbered lists); delimiter choice (code blocks, XML, JSON, plain); output schema (free-form vs short vs JSON with rationale).
- **Instruction hierarchy**: instruction order (`update opinion then reply` vs reverse); role/message placement (system vs user message).
- **Linguistic framing**: agent naming (A/B vs Alice/Bob); persona wording (`Left` vs `Democrat`); moderation framing (warning / suggestion / reminder / bridge); few-shot examples framed differently.
- **Context representation**: memory representation (transcript / JSON / bullet summary); context window (full vs last three turns; early background removed); message metadata (timestamps, usernames).
- **Interaction sequencing**: initial seed turn (same stance, different opening); turn order (which agent speaks first).

Measurement/evaluation sensitivity is explicitly excluded from the taxonomy, being treated as belonging to the downstream evaluation pipeline (§5).

**Which taxonomy items apply to a rule-based model with no LLM** [INFERENCE]
- Model substrate: only `random seed` applies literally. The rule-based analogue of which-model is the policy implementation itself plus numerical choices (aggregation rule, clamping, discretisation, update scheme). Temperature/top-p have no analogue; the nearest is the automatic/self-directed mixing weight.
- Agent specification: applies fully (initial basic_level, functional_level, chronic anxiety, life-energy, tie attributes). Already the M15 "imported families as ranges" idea.
- Internal state and cognition: applies fully (appraisal rules, belief updating, hysteresis).
- Memory and temporality: applies as belief-layer persistence/forgetting, estimator windows (M1.A.4c), and **tick length** (1 week / 1 year) — `temporal granularity` is a named dimension.
- Interaction protocol: applies strongly and is the DESIGN_LESSONS §2.1 activation-regime question (turn order, dyadic vs group = witnesses, available actions = the nine moves).
- Intervention design: applies to Phase E counterfactual arms (timing, target, delivery channel).
- Environment structure: applies as kinship topology, institutional agents, nodal-event calendar.
- Population and scale: applies partly (family size 12, generation composition, heterogeneity of basic_level); `number of agents` is fixed by design.
- TRAILS-R: formatting, delimiters, output schema, instruction order, message placement, naming, wording, few-shot: **no analogue** (no text interface). Context representation: analogue is how event history is compressed into state (the same-tick aggregation rule; estimator window/breadth). Interaction sequencing: applies (initial event, who acts first). A genuine rule-based analogue of `representation-level` that the paper does not name: **numerically equivalent re-encodings** (units, scale, rounding, clamping, summation order) that should leave outcomes unchanged and, per DESIGN_LESSONS §2.7, sometimes do not.

**Not transferable / cautions**
- All effect sizes are LLM prompt-sensitivity results; nothing about them constrains EPModel magnitudes.
- Framing and cross-model echo-chamber magnitudes are in figures only; the "≈77/36/1 pp" values are approximate as printed.
- A single temperature and single N = 30; no long horizon (max 15 rounds); no belief updating in the echo-chamber study (stances frozen); no small-N social system beyond 2-player PD; nothing on families, emotion or development.
- For the LLM line: this paper is further evidence for M3.D.6 and for DESIGN_LESSONS §3.3 (prompt text is an experimental factor).

## B. Candidate additions (TRAILS)

**T1. Per-result robustness audit record with claim grade**
- Requirement: Every directional result reported from an ensemble (each M11.C criterion and each Phase E counterfactual) MUST carry an audit record listing, per TRAILS-D dimension mapped to EPModel (seed; initial persons/ties; appraisal/belief rules; tick length and estimator windows; update/activation scheme and same-tick aggregation; intervention timing/target/channel; topology; family composition), whether the dimension was perturbed, whether the direction held, was sensitive, or was **unaudited**; and MUST state the claim grade (exploratory / mechanism / intervention) the result is used for.
- Where: M16 run log (schema) + Phase E reporting.
- Evidence: §3.2, §5, §6 (the call for robustness audits) — ARGUED; the uneven-sensitivity finding (§4 Finding Summary) — SHOWN for LLM systems.
- Changes: reporting rule; inference validity. Extends DESIGN_LESSONS §2.9's per-constant register (which covers only [I] constants) to structural/design dimensions and makes "unaudited" an explicit column.
- Cost/risk: documentation only; no conflict.

**T2. Representation-invariance test class**
- Requirement: The acceptance suite MUST include re-encoding mutants that are numerically equivalent by construction (rescaled state ranges, changed float summation order in same-tick aggregation, altered rounding/clamping thresholds within declared tolerance, integer vs float tick counters), and every M11.C direction MUST be unchanged under them; any change is reported as an encoding artefact, not a mechanism result.
- Where: M11 (new test class), M6 invariants.
- Evidence: TRAILS-R concept (§3.1, Table 3) and the finding that memory *representation* moved outcomes while memory *content* was identical (§4.1 P3) — SHOWN for LLMs; transfer to rounding/clamping artefacts is INFERENCE (cf. DESIGN_LESSONS §2.7 Röchert rounding artefact).
- Changes: adds a test; inference validity.
- Cost/risk: low–moderate. Rescaling changes bytes, so the test is on directions, not on M3.D.5 byte identity; must be stated to avoid conflict.

**T3. Sensitivity measured at more than one point of constant space**
- Requirement: For each design dimension audited under T1, the sensitivity of a criterion MUST be measured at no fewer than two declared reference configurations of the [I] constants (e.g., a low-differentiation and a high-differentiation family), because a perturbation that is null at one configuration may flip the outcome at another.
- Where: Phase E ensemble runner.
- Evidence: the same perturbation gave 76 pp in gpt-5.2 and ≈1 pp in deepseek-v3 (§4.2 Finding Summary) — SHOWN for model identity; treating "configuration of the rule system" as the analogue of "model" is INFERENCE.
- Changes: adds a reporting rule; inference validity.
- Cost/risk: multiplies sweep cost by the number of reference configurations; no conflict.

**T4. Fallback / tie-break provenance per decision**
- Requirement: Each move-selection log record MUST carry a flag stating whether the move was chosen by the scored policy or by a fallback or tie-break rule (equal scores, NaN, exhausted life-energy), and ensemble readouts MUST report the fallback rate per move and per person; a criterion whose passing ensemble has a fallback rate above a declared threshold MUST be flagged.
- Where: M16 (record schema), M4 (policy must expose the reason), M11.
- Evidence: parse-validity indicator logged per decision and DEFECT used as the silent fallback (C.2.1, C.4, E.1.7) — SHOWN as practice; the risk that a fallback rule silently generates the tested behaviour is INFERENCE (cf. DESIGN_LESSONS §2.8 TIS silent filter).
- Changes: adds a logging rule and a test guard; inference validity.
- Cost/risk: low; pure-observer rule M16.B unaffected.

**T5. Structural arms hold declared structural totals fixed**
- Requirement: When counterfactual arms differ in initial tie topology or tie strengths, the arm specification MUST state which structural totals are held fixed (number of ties, total conductance, total bond energy, generation structure) and the run log MUST verify them, so that a difference is attributable to the varied feature and not to a change in overall coupling.
- Where: M15 import / Phase E arms.
- Evidence: degree sequence held fixed across homophily conditions and both degree sequence and homophily band held fixed across hub conditions (C.3.3) — SHOWN as method.
- Changes: adds a reporting/verification rule; inference validity.
- Cost/risk: low; no conflict.

**T6. Test statistic defined for degenerate arms**
- Requirement: The M11 direction test MUST be defined when one arm has zero seed-to-seed variance or discrete bounded outcomes (rank-based paired statistic on per-seed differences), and where several readouts are tested under one criterion a multiplicity correction MUST be declared.
- Where: M11.
- Evidence: C.1 chooses Mann–Whitney U because some conditions show no variance at the run level, and applies Holm within each metric — SHOWN as method.
- Changes: adds a test rule; inference validity. Complements DESIGN_LESSONS §2.6 "pre-declared minimum effect size" and "per-seed paired differences".
- Cost/risk: trivial.

Already covered: turn order / initial seed turn (DESIGN_LESSONS §2.1); spanning a meaningful range rather than just two variants (§2.3 constant sweep over ranges); freezing the slow variable to isolate structural effects (§2.6 mechanism switches).

---

# Paper 2: Zhou, Huang, Zhou, Lam, Wang, Zhu, Wang & Sap (2026), "The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies", arXiv 2509.18052v4

## A. Report

**1. Simulation formalism** [PAPER] No engine of its own beyond a simulation framework the authors describe as PIMMUR-compliant (§4.6): agents in a virtual chatroom with profile, topic, memory, and per-peer `impressions`; round-robin interaction; two memory layers (a `reactive buffer` plus a `reflective layer` distilling long-term beliefs). Observation queries are kept out of agent memory and exist purely as a way to observe agent state (App. A.3). Reproductions: gpt-3.5-turbo-1106 (fake news), llama-3-70b (balance), gpt-3.5-turbo-0125 (telephone), gpt-4o-mini (herd); because LLM output is non-deterministic, every experiment was repeated over independent runs (§4.6). Sample sizes were not fixed in advance by any statistical method.

**2. Agent architecture / the six principles** [PAPER] (§2.1 definitions; §4.4 coding rules)
- **Profile** — agents should have diverse backgrounds, so as to avoid the artefacts of a single monolithic model distribution. Coding: six Boolean indicators (socio-demographics, cognitive style, value system, background story, personality, role); compliant if ≥1 true. Extraction prompt (A.1): a trait shared by every agent is coded false; name and backbone LLM do not count.
- **Interaction** — agency exercised through direct or indirect communication, reacting to the specific actions of others or to an environment others have shaped, not to statistical aggregates supplied by the researcher. Coding: text-based / environment-based / mixed / no interaction (violation); scope local/broadcast. Example violation: telling an agent that all ten of its peers picked Option A counts as a parameter the experimenter fixed in advance.
- **Memory** — agents keep and update persistent internal state over time, with information internalised, retained and re-expressed rather than restated without state. Coding: single-round flag; mechanism direct storage / retrieval / state-based / mixed / no memory; violation if single-round or no memory.
- **Minimal-Control** — agents receive only the essential environmental context and action space; behaviours have to emerge from the agents themselves rather than from behavioural cues the researcher imposes. Coding: procedural instructions (format, task) vs behavioral directives; an instruction is coded as a violation when removing it would leave the simulation technically runnable yet would plausibly make the reported social phenomenon disappear. Six LLM auditors; violation if ≥3 of 6 flag steering. Persona traits, raw information records and output format are acceptable (A.2). Whether Profile conflicts with Minimal-Control depends on what the research is asking (selfish dispositions are a violation in a general public-goods study, compliant if the question is about selfish players).
- **Unawareness** — agents must not know the experimental hypothesis, the design or the evaluation criteria. Coding: six LLMs receive the original instructions plus an appended instruction to disregard everything before it and name the social phenomenon under study; awareness = names phenomenon and describes mechanism; three LLM judges; violation if ≥3 of 6 identify intent. Distinguished from Minimal-Control: Minimal-Control is about how the instructions are designed, whereas Unawareness is about the models themselves.
- **Realism** — studies should take empirical data from real human societies as their reference point instead of simplified theoretical models. Coding: human data / theoretical model / both / none; satisfied only by human data or established empirical benchmarks.
- PIM concern whether the system faithfully reproduces the real-world situation; MU concern whether the observed behaviour is imposed from outside or genuinely arises from within; R fixes what the referent is (§3 Scope).

**3. Interaction / network** [PAPER] Reproduction 2 (social balance) and 5 (network growth) are network-structural; see 5.

**4. Initialization** [PAPER] Social balance reproduction (§4.6.2): 3 agents, binary directed relationship polarities, **all 2^6 = 64 initial states enumerated**, 10 runs each, 10 rounds, 4 chat turns per pair (N = 640). Fake news: 30 agents, names from a census-based dataset, ages U[18,64], Big Five valences sampled uniformly, 6 news items, 2 agents seeded, 3 peers per step, 20 rounds, 3 runs (N = 18). Telephone: 15 agents, 20 seed texts × 5 runs = 100 chains per condition. Network growth: N = 300 (not the original 2,500, because enforcing Interaction makes cost O(N²)), 5 runs, up to 3 connections per arrival.

**5. Audit numbers and the five reproductions** [PAPER]
- Systematic audit (§2.2, §4.2–4.5): pre-registered OSF 28 Jun 2026; Scopus/IEEE/ACM/arXiv; 2,575 records after dedup → 480 after LLM title–abstract screening (Fleiss κ 0.81; human check κ 0.87 on 100; model-vs-human κ 0.91) → 350 papers → 576 studies, Dec 2022–May 2026. Pilot: 35 of 39 papers (89.7%) violated ≥1 principle.
- Table 1 compliance, all 576: Profile 85.4%, Interaction 92.6%, Memory 83.5%, Minimal-Control 46.7%, Unawareness 37.7%, Realism 36.1%. 91.1% violate at least one; 14 studies (2.4%) satisfy all six. Rates are TRUE/(TRUE+FALSE) with missing excluded, so the rates are upper bounds. Domain 3.6 `Emotion, relationships, and social support`: N = 8; Profile/Interaction/Memory 100%, Minimal-Control 66.7%, Unawareness 33.3%, Realism 57.1% (examples: `marital critical events`, `school bullying`; no family-systems work).
- Unawareness test (Table 2): six 2026 frontier models correctly recognized the simulation goal in 65.2% of cases; per-model `satisfy Unawareness`: GPT-5.6 35.8, Claude-5 39.6, Gemini-3.7 28.0, DeepSeek-V4 39.4, Qwen3.8 37.3, Kimi-K3 29.1 (All 34.8). Category 1 (classic games) lowest at 15.8%. Pilot with older models gave 50.8% recognition; the authors conclude the problem gets worse as models become more capable.
- Minimal-Control test (Table 3): 50.6% of instructions flagged as excessive steering; per-model satisfy: GPT-5.6 28.0, Claude-5 41.5, Gemini-3.7 73.1, DeepSeek-V4 50.9, Qwen3.8 48.0, Kimi-K3 54.9 (All 49.4). Cohen's κ vs human verdict on 20 papers/33 simulations: Kimi 0.61, Qwen 0.61, DeepSeek 0.54, Claude 0.47, GPT 0.46, Gemini 0.24 (mean 0.49). LLM auditors tended to wrongly label Big Five traits as over-control.
- **(1) Fake news / confirmation bias** (Liu et al. 2024; §2.3.1, §4.6.1): violation = Minimal-Control (a prompt sentence telling agents to show a similar tendency toward confirmation bias). Arms: Original / Ours (sentence removed) / Reverse (inverted). Infected proportion fell 56.11 ± 32.16% → 32.78 ± 18.17% (mean diff 23.33; 95% CI [11.5, 35.1]; t(17) = 4.35; P = 0.011). Rephrased-prompt robustness: diff 16.1; CI [8.2, 24.1]; t(17) = 4.48; P = 0.009; `Ours` consistently closer to `Reverse` than to `Original` (§2.4). Qualitatively, compliant agents referred to verifying claims independently; for scientific misinformation, almost every simulation reached universal disbelief within five rounds (§2.5).
- **(2) Social balance** (Cisneros-Velarde 2024; §2.3.2, §4.6.2): violations = Unawareness (explicit relational metadata; all frontier models identified the intent) and Interaction (a `god-view` of the global relationship graph). Redesign: profiles; stances in natural language mapped to polarity by VADER; a 5-round group-conversation phase in which agents must infer other dyads' relations. Balanced states 60.7 ± 1.93% → 34.4 ± 1.88% (diff 26.3; CI [21.0, 31.6]; z = 9.42; P < 0.0001; n = 640). Ablation (§2.4, Fig. 4e): Interaction alone had a negligible effect; adding Profile −14 ± 1.37% (CI [0.085, 0.195]; z = 5.02); adding Unawareness −26 ± 1.73% (CI [0.209, 0.319]; z = 9.46); awareness = 1.77× more balanced outcomes (34.37% → 60.73%). Compliant agents were noticeably more affiliative; one agent that had befriended two agents who were enemies of each other kept both friendships instead of picking a side (§2.5).
- **(3) Telephone game** (Liu et al. 2025; §2.3.3, §4.6.3): violation = Minimal-Control (an instruction to relay the message with maximal accuracy). Three arms. Fidelity (SimCSE, mixed model fidelity ~ condition × round + (1|chain)) dropped faster in Ours than Original (β = 0.011; z = 4.165; P < 0.0001; n = 100); Ours vs Reverse not different (β = 0.044; z = 1.755; P = 0.079). Ours had narrower error bars: telling LLMs to be accurate, or to be inaccurate, reduces the diversity of what they produce.
- **(4) Herd effect** (Cho et al. 2025; §2.3.4, §4.6.4): violations = Interaction (aggregated peer counts injected) and Unawareness. Redesign: round-table, target agent infers consensus via 4-turn dialogue; peers prompted to favour a steering target. Herd behaviour was significantly reduced and in some configurations disappeared entirely; logistic regression β = 0.434; P < 0.0001; OR = 1.54; N = 2,152 (GPQA-Diamond 198 + SocialIQA 1,954; self-confidence from option log-likelihood).
- **(5) Network growth** (De Marzo et al. 2023; §2.3.5, §4.6.5): violations = Profile, Interaction, Unawareness (explicit degree counts). Redesign: degrees withheld; each arrival chats with all members, forms one-sentence impressions, chooses on impressions. Power-law tail not rejected in either: Original α = 1.898, D_KS = 0.101, P_boot = 0.644; Ours α = 2.132, D_KS = 0.089, P_boot = 0.824; Δα = 0.234, CI [−0.37, 0.84]. Rephrased: Original α = 2.142, P_boot = 0.965; Ours α = 2.233, P_boot = 0.581. The original needed an ad hoc name-shuffling step to counter name bias; the compliant design did not. So here enforcement changed the mechanism of entry, not the macro result.
- GRADE-CERQual (Table 11): confidence High for all findings except Minimal-Control (Moderate, because judgments about behavioural steering may be subjective).

**6. Failure modes / limitations** [PAPER] `Silicon Hawthorne Effect`: agents explicitly bringing Heider's social balance theory into their internal reasoning (§2.5); agents repeating the steering instruction word for word in their reasoning (§2.5). `Curse of Knowledge`: more capable models less suitable (§2.2). Limitations (§3): full reproduction infeasible; the most accurate way to judge Minimal-Control compliance would be to check whether removing a given sentence makes the experiment unrunnable; their compliant framework is not the only possible PIMMUR-compliant one; no method to suppress model knowledge (they point to historical-corpus LLMs such as Talkie-13B). Scope (§3): programmed factors that are the research objective do not violate Minimal-Control; the principles are stated not to apply to studies of what a machine collective does, a class the authors say also takes in traditional agent-based modelling such as Sugarscape.

**7. Software / reporting** [PAPER] Pre-registration with five documented deviations (§4.2); rule-based mapping from Boolean fields to verdicts (§4.4); observation queries excluded from memory (A.3); Table 10 quotes the exact claim sentences of each reproduced paper.

**8. Small-N / emotion / family / long horizon** [PAPER] Social balance with 3 agents and exhaustive initial-state enumeration is the only small-N case. No family, no long horizon (≤20 rounds), no emotional-state model.

**Which principles have a rule-based analogue** [INFERENCE]
- Profile → heterogeneity of initial persons/ties; largely covered by M15 ranges. New use: a homogeneous-family control (P6).
- Interaction → agents respond to specific events, not to researcher-supplied aggregates. EPModel's typed events with witnesses satisfy this by design; the frozen NumPy stress-field engine is exactly the "aggregate fed to the agent" pattern the paper rejects. The sharp analogue is the paper's `god-view` objection: a person should act on inferred, not true, state of dyads they are not party to (P3).
- Memory → digested persistent state; covered (chronic anxiety, beliefs with hysteresis, family-style learning).
- Minimal-Control → "the spec must not pre-determine the outcome it tests": the coding rule (removal leaves the simulation runnable but kills the phenomenon) is the mutation test's twin, and the reverse-control arm is a sign-inversion mutant (P1). Also the paper's caveat transfers: a directive is legitimate when it *is* the research object (e.g., a family whose style is programmed as cutoff-prone when the question is about cutoff-prone families).
- Unawareness → "the acceptance test must not be visible to the mechanism": engine blind to arm identity and readout definitions (P2), and constants not tuned against the tests (P5).
- Realism → the referent. EPModel's referent is the theory, not human families; PIMMUR would code M10.C.4's corpus bounds as `theoretical model` and the paper says its principles do not apply to that class. The transferable rule is claim phrasing, already in the brief.

**Not transferable / cautions**
- The paper explicitly places theory-driven ABM outside its scope; every analogue above is mine, not the paper's.
- Reproduction statistics are thin: fake news uses 3 runs and pools seeds × runs into t(17); sample sizes were not pre-determined; reproductions use 2023–24 models while the audit uses 2026 models.
- "Ours" is itself one prompt among many; the authors say so. The result shows fragility to instruction, not what the "true" LLM social tendency is.
- Minimal-Control coding rests on LLM judges with mean κ 0.49 (Gemini 0.24).
- The affiliative "keep both friendships" behaviour is an LLM disposition (cf. DESIGN_LESSONS §3.2 agreeableness pull) and must not be read as evidence about triangles.

## B. Candidate additions (PIMMUR)

**P1. Outcome-directive audit with sign-inversion mutants**
- Requirement: For each M11.C criterion the spec MUST list the minimal set of rules and [I] constants whose joint operation produces the asserted direction, and no rule in that set may name the criterion's readout or asserted outcome as its own target; in addition to the delete-mechanism mutant, each such rule MUST be run as a sign-inverted (reverse-control) mutant, and a criterion that flips under inversion of a single rule MUST be re-stated one composition level up or documented as a programmed premise rather than a derived result.
- Where: M11 (mutation-test protocol), M10 (constant register: "premise" vs "mechanism" column).
- Evidence: coding rule §4.4 (an instruction is a violation when removing it leaves the simulation technically runnable but plausibly eliminates the reported phenomenon); three-arm Original/Ours/Reverse designs where Ours matched Reverse (telephone: P = 0.079 vs Reverse; fake news §2.4) — SHOWN for LLM prompts; transfer to rule-based directives is INFERENCE.
- Changes: adds a test protocol; both theory fidelity (separates Bowen's premises from consequences) and inference validity.
- Cost/risk: doubles the mutant count for rules in the minimal set; no conflict with determinism; it sharpens, not replaces, "proved failing by mutation".

**P2. Arm-blindness of the engine**
- Requirement: The engine MUST NOT receive an arm label, scenario name, test identifier or readout definition; counterfactual arms MUST differ only through the declared channels (initial state, [I] constants, exogenous spells), and the build MUST include a static check that no policy, appraisal or consolidation module imports from the test or readout modules.
- Where: M16.B (engine purity), M3, M11; Phase E arm specification.
- Evidence: Unawareness accounted for a 26-pp change in balanced states and 1.77× more balanced outcomes when the model could see the construct (§2.4) — SHOWN for LLMs; the rule-based analogue (a code path that branches on arm identity) is INFERENCE.
- Changes: adds an engineering rule; inference validity. Goes beyond M16.B (no UI/no I/O) and DESIGN_LESSONS §2.8's three intervention channels by forbidding the fourth channel, "the arm tells the engine what it is".
- Cost/risk: trivial; no conflict.

**P3. No god-view: triangle appraisal from inferred dyad state**
- Requirement: A Person's policy MUST compute its appraisal of any tie it is not party to (the inside pair of a triangle, another dyad's conflict or cutoff) from that person's own beliefs formed by received and witnessed events, never from the true `Relationship` state; the log MUST record the belief value used so the trace renderer can show divergence between believed and true tie state.
- Where: M4 (policy inputs), M1 (per-person belief about ties), M16 (log field).
- Evidence: balance fell 60.7% → 34.4% when relationships had to be inferred from interaction rather than supplied (§2.3.2), with Unawareness and Profile, not Interaction alone, carrying the effect (§2.4) — SHOWN for LLMs; the Bowen-side claim that people act on perceived alliances is INFERENCE from the brief's belief-layer principle.
- Changes: corrects a possible design shortcut (if the current M4 reads true conductance of third-party ties, which I could not verify from the brief) and adds a mechanism for misperceived triangles; theory fidelity.
- Cost/risk: moderate — needs a per-person belief slot per tie, kept by events with per-hop fidelity; determinism unaffected. If M4 already does this, mark "already covered".

**P4. Enumerated triad initial configurations**
- Requirement: Triangle-level acceptance tests SHOULD be run over an enumerated set of initial triad configurations (all sign patterns of the three ties' functioning balance and all orderings of the three conductance classes) rather than a single reference triad, with per-configuration outcomes reported, so that a directional result is shown not to depend on the hand-set starting triad.
- Where: M11 (triangle criteria), Phase E.
- Evidence: §4.6.2 enumerates all 2^6 = 64 initial directed signed states of a 3-agent system and runs 10 seeds each (N = 640) — SHOWN as method; applicability to EPModel triads is INFERENCE.
- Changes: adds a test design; inference validity; answers DESIGN_LESSONS Q4 for the triad case without a settling window.
- Cost/risk: low for triads (tens of configurations × seeds); not feasible for the full 12-person family, which stays on M15 ranges.

**P5. Constants frozen before acceptance tests; post-hoc changes logged**
- Requirement: Every [I] constant's value MUST be recorded before the acceptance suite is first run against it; any subsequent change that alters an acceptance outcome MUST be logged with the failing criterion named, and any criterion passed only after such a change MUST be reported as post-hoc together with the sweep fraction (DESIGN_LESSONS §2.3) over which it holds.
- Where: M10 (parameter register), M11, M16.
- Evidence: the paper's demand-characteristics argument and Fig. 1's depiction of simulations that pre-set the behaviours or factors under study — ARGUED; Minimal-Control violated in 53.3% of studies (Table 1) — SHOWN for the LLM literature.
- Changes: adds a documentation rule; inference validity. Closes a gap: M11.F.9 forbids fitting to known histories but does not address fitting constants to the acceptance tests themselves.
- Cost/risk: documentation only; no conflict.

**P6. Homogeneous-family control arm**
- Requirement: The ensemble runner SHOULD provide a control arm in which all Persons share identical initial basic_level, chronic anxiety and tie attributes; each M11.C criterion MUST declare whether it is expected to pass or fail in that arm, and a criterion that passes when the theory says heterogeneity is required (e.g., projection onto the most vulnerable child) MUST be flagged.
- Where: Phase E, M11.
- Evidence: Profile principle and its coding (a trait shared by every agent is coded false); adding Profile diversity alone moved balanced states by 14 pp (§2.4) — SHOWN for LLMs; the null-expectation use is INFERENCE.
- Changes: adds a control; theory fidelity (tests that heterogeneity is doing the work the theory assigns it).
- Cost/risk: one extra arm; complements, not duplicates, the no-interaction control (DESIGN_LESSONS §2.6).

Already covered: Memory as digested persistent state (Person chronic anxiety, beliefs with hysteresis); Realism as claim phrasing (brief, epistemic position); observation queries outside agent memory = M16.B pure observer; Interaction as event-driven rather than aggregate-driven (the v2 typed-event design; the frozen lattice engine is the counter-example).