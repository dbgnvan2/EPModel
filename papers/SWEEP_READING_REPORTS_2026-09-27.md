# Reading reports: four preprints from the 2026-09-27 "Agent Simulation" alert

Produced 2026-09-26. The owner's alert export (`~/Downloads/Agent Simulation _ 2026-09-27 _4_ - summaries.pdf`, 16 papers) was screened on its summaries. Four papers were chosen for a full read; the other twelve were not read beyond the summary and **no candidate is derived from them**. Each paper was read in full from `pdftotext -layout` output by a separate reader under `sweep_readers_brief_2026-09-19/READER_TASK.md`, with the rev-10 spec and `SPEC_CANDIDATES_from_preprints_2026-09-20.md` checked for overlap by keyword search. The reports are reproduced as the readers wrote them, except that candidate labels were made unique across this file.

**Status: none of these candidates is in the spec.** They are proposals for the owner, in the same standing as C1–C42 before revision 10.

| Part | Paper | Candidates | Where they would live |
|---|---|---|---|
| 1 | Jiang et al., *Hybrid Coevolutionary Opinion Games*, arXiv 2609.27639v1 | J1–J4 | M11.G readouts, Phase E nulls, M17.F.2 settling, M11 mutation suite |
| 2 | Zhang et al., *SocioVerse2*, arXiv 2609.24911v1 | SV1–SV4 | M11.D prefix invariance, M17.D intervention arm, M15 import point-in-time rule, M16.A.1 version lineage |
| 3 | Nair-Turkich, Campbell & Geard, *Partnership dynamics in dynamic network ABMs*, arXiv 2609.17622v1 | PD1–PD4 | M16.A tie-episode record, M10 hazard shapes, M3.B slow-tick phase, M17.E.1 Latin hypercube |
| 4 | Leins et al., *Prompting Against Persona Drift*, arXiv 2609.24532v1 | X5–X9 (exploratory LLM line only) | Phase F narrator (M16.E) |

Screened on summary only, not read: 2609.28942 (BaCVA), 2609.28609 (AdvRole), 2609.26927 (AGIMUD), 2609.25186 (LLM mental-health survey), 2609.22522 (cosine similarity), 2609.22517 (SOPHIE 2.0), 2609.21925 (ChatGPT panic autoethnography), 2609.21626 (Self-Meta-Evolve), 2609.19913 (Digital twins for opinion dynamics; already in INDEX), 2609.17496 (Fuse), 2609.16344 (PAIR; already in INDEX). 2609.17331 (SEAA) was already digested in `DESIGN_LESSONS_model_design_papers_2026-09-17.md` §7.


---

# Part 1: Jiang et al. 2026

## Jiang et al. 2026, "Hybrid Coevolutionary Opinion Games" (arXiv 2609.27639v1): reader report

Scope. I read the whole paper text (lines 1–1293, including the references). Figures survive only as captions and scattered axis labels. The equations came through partly garbled but can be reconstructed. The spec was checked at revision 10, by keyword search and by reading the relevant passages: M1.A.5–M1.A.5d, M6.I.4, M7.A.1, M11.4a, M11.4e, M11.F.6, M11.G.1–G.2, M16.A.9 and M17.A–M17.G.

### A. Report

**1. Simulation formalism**

- [PAPER] The state is the directed graph plus the previous text records of the LLM agents (§3.2). Every agent has out-degree fixed at K = 5, and N = 50.
- [PAPER] The update is synchronous. All agents update their opinions, then every out-neighbourhood is rebuilt (§3.3, §4.3.4). There is one step clock, so tie updates and opinion updates run at a 1:1 ratio.
- [PAPER] Rewiring is memoryless. The transition kernel "does not depend on S(t)" because each step rebuilds all neighbourhoods from current opinions and does not edit the previous edge set (§5.1, Eq. 9). The initial topology therefore affects only the first update (§4.3.2).
- [PAPER] Ties at the K-th neighbour are broken by a dedicated seeded generator, "so ties do not fall to the lowest agent index" (§5.1). The reported correlation between in-degree and agent index is −0.037 (§6.4).
- [PAPER] Determinism is claimed only for the numerical path. The α = 1 path is "bit-for-bit reproducible" (§6.4). The LLM path uses temperature 0 and a fixed vLLM seed (§4.3.2), but the paper does not claim it reproduces.

**2. Agent architecture: (a) the Friedkin–Johnsen anchor**

- [PAPER] Each agent has a fixed intrinsic opinion s_i and an expressed opinion z_i (§3.1). The numerical agents (Type-C) take the myopic cost minimiser (Eq. 1):
  z_i ← (Σ_j z_j + ρ_i K s_i) / (K(1+ρ_i))
  This is a convex combination of the neighbour mean and the agent's own anchor. The weight on the anchor is ρ/(1+ρ). With ρ_i ~ U(0.3, 0.7), drawn independently of stance (§4.3.1), that weight is about 0.23–0.41. The anchor s_i never changes. Numerical agents start at z(0) = s.
- [PAPER] The authors say the difference between agent types comes down to "where the anchor sits" (§7.1). For numerical agents the prior is a term in the objective, "a restoring force". For LLM agents it is "a sentence in a prompt".
- [INFERENCE] Mapped onto EPModel, s corresponds to `basic_level`, z to `functional_level`, and ρ to a stubbornness term that differentiation would set. The structural parts of the analogy do not carry over, for three reasons:
  - The Friedkin–Johnsen update is averaging. It does not conserve anything. When i moves toward j, j loses nothing, so there is no lender. M6.I.4 requires that one spouse's functional gain equal the other's loss.
  - The anchor is a symmetric quadratic restoring force. M7.A.1 forbids exactly this: recovery toward baseline "MUST NOT be a symmetric restoring force".
  - The anchor is fixed for ever. `basic_level` is recomputed on the slow tick.
  - Conclusion: Friedkin–Johnsen gives a clean contrast form, not a mechanism for borrowing and lending. See candidate J4.
- [PAPER] LLM agents (Type-L) are Phi-4 (14.7B, 4-bit AWQ). Each makes one call per step and returns three fields in the order reasoning, opinion, memory. The memory field is rewritten each step and capped at 120 words (§3.2, §4.3.3). The persona is written once by Phi-4 and then held fixed.
- [PAPER] The model is never asked for a numerical self-rating. An external RoBERTa regressor (r = 0.926 against held-out human judgements, §4.2) scores the opinion text. The stated reason is that LLMs regress to the centre of a scale and anchor on round values (§4.3.3). This is cited, not shown in this paper.

**3. Interaction and network: (d) rewiring**

- [PAPER] In the experiments each agent links to the five peers whose expressed opinion is nearest to its own *intrinsic* opinion (Eq. 2). This is the β → ∞ limit of a Gibbs kernel exp(−β·d). Rewiring is "a kernel of the environment, not an action any agent chooses" (§3.3). It happens every step, after all opinion updates.
- [INFERENCE] So the ties are chosen by the anchor, while the displacement comes through those same ties. Because no tie has history, the ratio of tie timescale to state timescale (design lessons §2.4) is not varied. It is fixed at 1:1 and effectively infinitely fast. Nothing here bears on EPModel ties with near-zero bond decay (M1.B.4) or on "no exit from the field".
- [PAPER] Churn: the fraction of neighbours retained per step plateaus at 0.913 ± 0.069 at α = 0 (about 8.7% of ties replaced per step) and at 0.990 ± 0.004 at α = 1 (§6.5.1, Fig. 4a).
- [PAPER] The modularity null: 44% of the raw modularity at α = 1 (Q = 0.763 against a degree-preserving null of 0.336) comes from the constant out-degree the K-NN rule imposes (§6.5.4). All claims use the null-corrected Q_norm. The null-corrected values are still 18–27 SD above the null.

**4. Initialization**

- [PAPER] Stances come from 5,199 r/politics comments (April 2019), stratified into five strata with thresholds ±0.25 and ±0.75 (§4.2). Each agent keeps its own comment's continuous score as s_i. Initial splits are 54/46 (gun control) and 68/32 (abortion).
- [PAPER] Personas and ρ are identical across the two topics, and corr(s_gun, s_abortion) = +0.881. The authors therefore say the two topics "are not independent replicates" and compare them with paired tests (§6.1).
- [PAPER] An LLM agent's z(0) is produced by generating opening text from its persona plus the *stratum keyword*, then scoring that text (§4.3.1). It is not set to s_i.
- [INFERENCE] LLM agents therefore start displaced from their anchor before any interaction. The paper never reports conformity at t = 0, so part of the displacement term may be an initial-condition artefact (see Cautions).
- [PAPER] The three initial topologies (BA, ER, WS) are matched on mean degree (3.84, 3.84, 4.00) and differ in clustering (0.179, 0.079, 0.390) (§6.2).

**5. Calibration, validation, statistics: (b), (c), (e)**

- [PAPER] **Design (e).** 9 α levels × 3 topologies × 10 seeded graphs × 2 topics = 540 runs, all completed (§4.3.2, §6.2). Seeds vary topology and type assignment. The sampled agents, anchors, personas and ρ are held fixed. The confidence intervals therefore exclude sampling variability of the agents (§4.3.2, §7.3 item 2).
- [PAPER] **Is it two-arm?** No. It is a nine-level dose sweep. Adjacent α levels are compared with Welch's t-test plus Benjamini–Hochberg correction (§6.3). The stated reason is that the SD of PoA (Price of Anarchy) "falls by two orders of magnitude" from α = 0 to α = 1.
- [INFERENCE] The graphs are "crossed" with α, so pairing by graph seed was available but not used.
- [PAPER] Equivalence of topologies is tested with TOST (two one-sided tests) against δ = 0.2, fixed in advance. The margin is absolute, so it is 17.5% of PoA at α = 1 but 3.6% at α = 0 (§6.3). Equivalence is established only at α = 1 (p < 0.0001). At other levels the data cannot exclude a difference of about 0.5 (§6.5.5, Table 4).
- [PAPER] **Convergence criterion (e).** A run stops at the first exact recurrence of the edge set, or when the relative change in C_out (the neighbour-retention rate) between two windows of W = 10 steps stays below ε = 0.01 for 5 steps. Ten more steps are then recorded, with T_max = 120 and P_max = 20 (§4.4.3). Runs are classified as fixed point, limit cycle:p, plateau, or none.
- [PAPER] The authors caution that these labels "do not by themselves establish a Nash equilibrium" (§4.4.3).
- [PAPER] Three measures that the stopping rule never reads (normalized Hamming distance, spectral distance, 1 − DeltaCon) all flatten at about t = 25–30. That is ahead of the median t_conv of 46, so the rule's own statistic is "the last of them to register it" (§6.5.1, Fig. 4b). All 540 runs converged.
- [PAPER] Outcomes are read at each run's own t_conv + 10. Fixed indices t = 20 and t = 35 are kept as robustness checks (§4.4.4). The metrics are flat about 10 steps in (Fig. 5).
- [PAPER] **Social-cost decomposition (c).** Eq. 3 splits social cost into Σ(z_i − z_j)² over neighbours ("disagreement") plus Σ ρ_i K (z_i − s_i)² ("conformity", meaning displacement from one's own anchor). From α = 1 to α = 0 (Table 3):
  - disagreement rises 3.3× (0.956 → 3.136);
  - conformity rises 13.2× (0.183 → 2.422);
  - the conformity share rises from 16.1% to 43.6%, falling monotonically across all nine levels as α increases.
- [PAPER] PoA is 5.558 ± 0.309 at α = 0 and 1.139 ± 0.005 at α = 1, the latter close to the 9/8 bound (§6.5.3). There is no significant step up to α = 0.375 (q > 0.12). Every step from α = 0.5 onward is significant (q < 0.04) (Table 2).
- [PAPER] **Polarization reading (b).** Pz rises from 0.304 to 0.597 and Q_norm from 0.218 to 0.423 as α goes from 0 to 1 (§6.5.4). The authors' reading is that the less polarized population is the more susceptible one:
  - "Weaker polarization is a symptom, not an improvement" (§7.1).
  - Low polarization "is what a population looks like when its members do not hold their ground" (§7.1).
  - Read alone, polarization "points the wrong way".
- [INFERENCE] This matches EPModel's readout traps (M11.F.6: "an apparently fine marriage", "absence of adolescent rebellion"). The paper's contribution is to pair the observable that looks healthy with a displacement-from-anchor term, and to show that the two move in opposite directions.

**6. Failure modes and limits: (f)**

- [PAPER] The authors list their own limits (§7.3):
  - one LLM only, so it is unsettled whether the gap is "this model's particular disposition to be persuaded";
  - no sampling variability in the agents;
  - topology equivalence shown only for α = 1;
  - the modularity resolution limit is not corrected.
- [PAPER] Two results have no mechanism, and the paper says so:
  - mixed populations reach an exact fixed point more often (62% at α = 0.875 against 18% at α = 0 and 12% at α = 1);
  - settling time is non-monotone in α (52.2 → 21.7 → 25.1 steps) (§6.5.1, Fig. 3).
- [PAPER] Theorem 1 assumes Optimistic Gradient Ascent (OGA) dynamics, and the authors state that "the simulated agents do not run OGA" (§4.4.3). The abstract's claim that convergence "licenses" reading a PoA therefore rests on an empirical plateau in graph change, not on an established equilibrium.
- [PAPER] **Bearing on M3.D.6 (f).** When the anchor is given as prompt text, it does not act as a restoring force under uniform peer input (§7.1). LLM runs settle about twice as slowly (52.2 vs 25.1 steps) and churn about 9× more ties per step. Only the numerical path is claimed reproducible.
- [INFERENCE] This is one more instance supporting "no LLM in the decision path". A persona-conditioned stance is not a stable basic level, and "holding a position" cannot be delegated to a prompt. It fits the pull-toward-agreeableness findings in design lessons §3.2 and candidates-file §5. The paper gives no LLM runtime or cost figures.

**7. Software and engineering**

- [PAPER] Code is public (§6.4), and all 540 runs passed a schema-conformance audit of per-step output. Reasoning traces were kept in full. The structural measures are logged at every step whether or not the rule reads them (§3.5).

**8. Small-N, emotion, family, long horizon**

- [PAPER] None of these. N = 50, runs of at most 120 steps, political stance only. The note that near-balanced initial splits (54/46) settle later because more agents sit near the K-NN boundary (§6.5.1) is a small-N boundary effect of the kind in design lessons §2.2.

**Not transferable / cautions**

- K-NN rewiring to strangers, rebuilt each step with no edge history, has no EPModel analogue.
- PoA needs a planner optimum over a convex cost, and EPModel has none.
- The RoBERTa stance pipeline is irrelevant.
- [INFERENCE] **Three confounds weaken the "susceptibility" reading:**
  1. LLM agents are scored with the ρ_i weight in the conformity term. Section 4.3.1 assigns ρ to "each Type-C agent", while §3.4 and §6.1 apply ρ to all agents. The prompt carries s_i but, as described, not ρ. So LLM agents are scored against an objective they are not given.
  2. LLM agents start displaced from their anchor, because z(0) is generated from a stratum keyword, not set to s_i.
  3. Numerical agents' z is exact, but LLM agents' z carries regressor error (r = 0.926). Error in z inflates the displacement term for LLM agents only.

  None of these is separated in the paper. The direction of the result may survive, but its size (13.2×) should not be quoted as a property of language updating.

### B. Candidate additions to the EPModel spec

**J1. Report the displacement term beside every low-conflict readout**

- **Proposed requirement.** Wherever M11.G.1 emits a quantity that reads as health when low (overt conflict in the marital-conflict sink, count of CONFLICT moves, tie tension), the evaluation MUST also emit, for the same family and window, the family's aggregate displacement from basic level. This is Σ over persons of (`functional_level` − `basic_level`), split into borrower and lender sides per M1.A.5c. The evaluation MUST also emit the displacement term's share of the two combined.
- **Acceptance test.** An M11 test SHOULD construct two arms: one lower-`basic_level` family with fewer CONFLICT moves and more borrowing, and one comparison family. The test asserts that the readout does not order them by conflict alone.
- **Where it would live.** M11.G, with the log side in M16.A.
- **Evidence.** §3.4 Eq. 3, Table 3, §7.1. Paper-level strength: SHOWN, that the two terms move in opposite directions. Transfer: INFERENCE.
- **What it changes.** Adds a reporting rule and a test. Improves inference validity by making one of M11.F.6's readout traps computable.
- **Cost / risk.** Low: an observer-side sum, consistent with M16.B. No new constants. It must not be presented as a maturity score, and it needs the same labelling as M16.A.9.
- **Overlap.** PARTLY covered by M11.F.6 (two-sidedness), M11.G.1 components 2–3 (sink proportion, budget occupancy) and M11.G.2. What is new is the paired emission and its share.

**J2. Constraint-preserving null for structural readouts**

- **Proposed requirement.** Any structural readout over ties or triangles that Phase E reports (for example, concentration of cutoffs along a lineage, or the share of triangle activations on one triad) MUST be reported both raw and against a null distribution. The null is a permutation that preserves the constraints the rules impose: the kinship graph, the number of ties per person, and the event and move counts per person. The report MUST state what fraction of the raw value the null reproduces.
- **Where it would live.** Phase E (M17.B), with log fields in M16.A.
- **Evidence.** §3.5 Eq. 7 and §6.5.4: 44% of raw modularity came from the constant out-degree the rule imposes (SHOWN).
- **What it changes.** Adds a reporting rule. Improves validity, because a structural pattern forced by the model's own constraints is not reported as a finding.
- **Cost / risk.** Low to moderate: post-run permutations. There is no conflict with M3.D.x. The permutation draws must be keyed outside the engine.
- **Overlap.** Not covered. M17.D.1 control arms re-run the model; this is a readout-level null that needs no re-run.

**J3. Corroborate the settling condition with measures the rule does not read**

- **Proposed requirement.** M17.F.2's settling detection SHOULD be corroborated by at least two drift measures of the same state that the stopping rule does not read. The report MUST state their agreement or disagreement with the rule. A readout taken at the settled point SHOULD also be reported at two or more fixed ticks declared in advance, as a robustness check.
- **Where it would live.** M17.F.2.
- **Evidence.** §4.4.3–4.4.4, §6.5.1, Fig. 4b: three unread measures flattened before the rule fired. Reading at t = 20 and t = 35 in addition to each run's own evaluation point (SHOWN as method).
- **What it changes.** Adds a test and a reporting rule to an existing requirement. Improves validity, because a single `[I]` window and threshold otherwise decides where the transient ends.
- **Cost / risk.** Low, observer-side. The extra measures' windows are also `[I]`.
- **Overlap.** PARTLY covered by M17.F.2, and by M17.B.2 (regime classes, which correspond to the paper's fixed point, limit cycle, plateau and none).

**J4. Averaging-anchor substitution mutant for `functional_level`**

- **Proposed requirement.** The M11 mutation suite SHOULD include one mutant that replaces the conserved borrowing and lending exchange with a Friedkin–Johnsen form: `functional_level` becomes a convex combination of the partners' levels and the person's own `basic_level`, with a symmetric quadratic pull and no conservation. Every criterion that depends on M6.I.4 or M1.A.5c (M11.C.20 at minimum) MUST turn red under it. A criterion that stays green MUST be reported as not distinguishing exchange from averaging.
- **Where it would live.** M11, mutation protocol, reported under M11.4f.
- **Evidence.** §3.2 Eq. 1 and §7.1 give the form. That the form is forbidden in EPModel is a spec fact (M7.A.1, M6.I.4). The test is INFERENCE.
- **What it changes.** Adds a test. It answers the owner's question (a): Friedkin–Johnsen is the obvious neighbouring model and the one a reviewer would propose, so the suite should show that the criteria can tell it apart from the specified mechanism. Improves both fidelity and validity.
- **Cost / risk.** Moderate: one alternative update function behind a mutant switch. ρ is `[I]`. No rule conflict, since mutants are changes to the model under M11.4f.
- **Overlap.** PARTLY covered by C21 (representation-invariance mutants) and M17.D.2 (rival-mechanism arm). What is new is naming this specific rival form.

**Already covered (no write-up):**

- TOST with a margin fixed in advance: M11.4a. One addition worth noting there is that an absolute margin changes relative size across arms (3.6% to 17.5% here), so the margin's relative size should be stated at each arm's scale.
- Welch test under unequal variances: superseded by M17.A.3 and M11.4e, which require paired per-seed differences.
- Regime classification: M17.B.2 / C29.
- Seeded tie-breaks: M3 / C12.
- Both topics treated as paired because they share agents: M3.D.4a coupling.

### C. Search terms

- "Friedkin-Johnsen model" stubbornness "innate opinion"
- "coevolutionary opinion formation game"
- "price of anarchy" "opinion formation"
- "adaptive coevolutionary networks" (Gross & Blasius)
- "expressed and private opinions" OR "EPO model"
- "social influence network theory" susceptibility
- "susceptibility to persuasion" "opinion dynamics" anchoring
- "degree-preserving null model" modularity
- "temporal correlation coefficient" "time-varying graphs"
- "homophily" "network rewiring" "opinion dynamics" timescale
- "LLM agents" "opinion drift" persona consistency
- "sycophancy" "multi-agent" "opinion dynamics"
- "polarization" "disagreement" "Chitra Musco" OR "Musco" "polarization-disagreement index"
- "Markov game" "opinion dynamics" convergence
- "bounded confidence" "stubborn agents" "zealots"

---

# Part 2: Zhang et al. 2026 (SocioVerse2)

## SocioVerse2 (Zhang et al. 2026, arXiv 2609.24911v1): reader report

I read the whole text, appendices included. None of the seven cases models a family, a dyad, emotion, or a horizon longer than 75 steps. Most of the reported numbers come from earlier companion papers rather than from new SocioVerse2 runs. What carries over to EPModel is the fork and intervention formalism (§3.3), the point-in-time discipline (§4.3), and the paper's own admission about which version gets reported (§6.1).

### A. Report

**1. Simulation formalism**

- [PAPER] Each step is B = f(P, E) and E(t+1) = g(E(t), B(t)) (eqs 1, 2, 4). A step runs five phases in a fixed order: exogenous update, observe, decide, apply, record (§3.2.1, Fig. 3). Actions within a round are applied synchronously at the end of the round. Phases 2–4 can repeat for K interaction rounds.
- [PAPER] In the Chicago example, 93 households intend to move and 37 moves execute under budgets and caps (§3.2.1). The paper does not say how the 37 are chosen.
  - [INFERENCE] That is an unstated order or tie-break rule, the class covered by M1.F.8, M11.D.16 and M4.D.1f.
- [PAPER] Interventions enter only in the exogenous phase, "before any agent observes", and go through the same call as the regular schedule (§3.3, Fig. 4a).
- [PAPER] The population is built "deterministically under a declared seed". Identifiers are "never reused" (§3.2.1, §4.2).

**2. Agent architecture**

- [PAPER] Memory is a bounded window of the agent's own recent actions: 8 steps by default, rebuilt exactly from the panel table (§3.2.2).
- [PAPER] There are four behaviour functions: rule-f (the "parity baseline"), LLM-f, RL-f, and hybrid-f (§3.2.2). All return the same typed action: type, payload, rationale, and a source label of LLM, rule, fallback or replay.
- [PAPER] Hybrid-f in Case 2 uses 300 LLM core users and 700 rule users. The two tiers are coupled by "mirror agents", whose attitudes are overwritten each step from the core user's scored message. There is no reverse influence (§5.2.2).

**3. Interaction structure**

- [PAPER] Observation is projected along two axes, modality (physical or information) and scope (macro or local), with audience selectors. In the Chicago example a ward notice addressed to another district is not seen (§3.2.1).
- [PAPER] Messages travel on a mediated bus and are delivered by neighbour lookup, from earlier rounds of the step and from the previous step (§3.2.2).
- [INFERENCE] This is already covered by C6/C7 (M3.E, M16.A.1a).

**4. Initialisation**

- [PAPER] The Population MCP routes requests across five pools, joins records, reweights with IPF to target marginals, and synthesises uncovered attributes "conditional on" existing demographics. Every attribute records its source (§4.2, Fig. 7). About 10.4M records are indexed.
- [PAPER] In Chicago, 241 archetypes are weighted into 19,235 agents. Each run starts from the 2010 census with 15% of households displaced, which lowers the Black–White dissimilarity index (DBW) from 0.835 to about 0.72 (§5.3.1).

**5. Calibration and validation** (item (e) below covers strength)

- [PAPER] Results are read from four angles in a fixed order: micro alignment, level fidelity, trajectory fidelity, mechanism (§5.1, Fig. 9).
- [PAPER] Table 11 discloses repetition per case. Synthetic tasks get 3–10 seeds. The three forecasting cases are each "single calibrated run" or "single frozen configuration".

**6. Failure modes and limitations** (all [PAPER], §6.1)

- LLM-f is weakest where exact numbers or continuous control are needed (NaSch, Boids).
- Cost: about 275K calls, 350M input tokens, for one 15-step Chicago run.
- Branch contrasts are "simulation-internal contrasts under the model's assumptions".
- "The human side of the loop": the report gives the accepted version, "not the tree of versions".

**7. Software engineering**

- [PAPER] The study state is σ = (P, E0:T, f, Θ), where Θ holds the metrics, the resource manifest and the seed. An edit is a_k = (c, u), and it produces one of three outcomes (eq. 8):
  - no change;
  - a **branch**: edits E only, holds P, f and seed fixed, and inherits the parent's history;
  - a **version**: may edit any component, and runs fresh.
- [PAPER] The version manifest records id, parent, note, creation time, and for branches the source version and fork step. Each version snapshots the whole workspace: code, bundles, trajectory, reports (§3.4, Fig. 5).
- [PAPER] Every artifact is typed and validated on write and on read. The pipeline has seven skills with gates between them. Four researcher decisions are always logged (§4.1, Table 2).
- [PAPER] The environment is resolved at build time; "the run itself fetches nothing" (Fig. 8, §4.3).

**8. Small N, family, stress, long horizon**

- [PAPER] Nothing on any of these. The smallest population is Boids at N = 40. Horizons are 14–75 steps. Households are archetype units with no internal dynamics.

### Owner's specific checks

**(a) Fork mechanics.**
- [PAPER] An intervention is δ = (t*, op). A branch is built by **replay**, not by copying state (§3.3, eqs 5–7):
  - P and E0 are rebuilt from the same bundles and seed;
  - for steps 1 to t*−1 the parent's scheduled events are activated and its stored actions are read back from the panel ("no decision calls"), labelled `replay`, and pushed into memory windows;
  - the decision model is called only from t* on.
- [PAPER] Replay is exact because g is a function of the recorded actions. The same mechanism extends a finished run and resumes an interrupted one.
- [PAPER] In the Fig. 4b example the control is itself a branch forked from the parent, not the parent run.
- [PAPER] On randomness the paper says only that P, f and the seed are "the same objects in both branches". It then states the difference is "attributable to δ alone" (eq. 7).
  - [INFERENCE] That claim does not hold for a stochastic f. After the fork, both branches draw fresh LLM samples (temperature 0.7 in Cases 1 and 3), and nothing couples those draws across branches.
  - [INFERENCE] Replay also skips the parent's decision draws, so a stateful generator in a branch would sit at a different position at t* than the parent's did. Branch-versus-branch comparison is aligned at t*. Branch-versus-parent comparison is not.
  - [INFERENCE] No case reports an ensemble of branch pairs or any variance of Δ. The only branch shown is the 8-step illustration in Fig. 4b.
- [INFERENCE] Mapping to EPModel: the shared prefix matches M0.4's two-arm logic. Post-fork coupling is weaker than M3.D.4a's keyed draws. At N ≈ 12 and about 2,080 ticks with no LLM, rerunning from tick 0 under M3.D.5 makes replay unnecessary. What does transfer is a testable assertion that the arms' prefixes are identical (SV1).
- [INFERENCE] The branch/version split is the same line EPModel draws in M11.4f: a mechanism-off comparison is a different model (a version), not an intervention arm.

**(b) The study as editable state.**
- [PAPER] What is versioned: the full workspace, with lineage in the manifest (§3.4). Provenance: pool snapshots versioned, signals archived by year-month vintage, a manifest pinning both, and the trajectory store recording which events bundle each run used (§4.4).
- [PAPER] The reported result of each case is "the version its researchers accepted" (§5.1). §6.1 concedes that the tree of versions leading to it is not reported.
  - [INFERENCE] That is the problem M10.B.4 addresses, but here it extends past constants to code, population, schedule and readouts (SV4).
- [INFERENCE] The manifest fields fit M16.A.1's header. A workspace snapshot adds nothing that git plus the config hash does not already give EPModel.

**(c) Point-in-time guarantees.**
- [PAPER] The Event MCP answers queries parameterised by (year, month) and returns only the vintage available then. Every payload is stamped with source, series and vintage. The data is materialised at build time (§4.3).
- [PAPER] An intervention broadcast added later is grounded by the same as-of query, so the treated branch keeps "the same information discipline" as its parent.
- [PAPER] Target values are kept out of the prompt: target-month CCI values in §5.4.1, and PMI/ISM values in §5.4.2. Beyond that, the paper relies on post-knowledge-cutoff windows.
- [PAPER] Several cases feed ground truth back into the output anyway:
  - CCI: "Behavioral Inertia Alignment" blends the forecast with the previous month's official value at weights 0.4, 0.6 and 0.7 (Table 14).
  - Car market: KBA observations enter a post-hoc calibration (pseudo-count 30, α = 0.45, 0.03 per-cell residual cap), and the authors label the result a reconstruction (§5.4.3, A.8).
- [INFERENCE] EPModel has no environment data, so the vintage machinery itself does not apply. The discipline does apply in two places:
  - **M15 import.** A family diagram carries dated transitions after t0 (M15.A.3) and dated ratings (M15.A.6). The spec does not say whether events dated after t0 are inputs or held-out outcomes, or whether a rating made with hindsight may set the state at t0. Either path puts the known history into the run, which is M11.F.9(c) by another route (SV3).
  - **Inside the engine.** Nothing may read a scheduled future input before its tick (SV1).

**(d) Canonical ABM reproduction.**
- [PAPER] Case 1 covers 10 models: NaSch, Boids, Social Force, Sugarscape, Minority Game, Axelrod IPD, Schelling (N = 810), Civil Violence, SIR rumour, Hegselmann–Krause (Table 12).
- [PAPER] Agreement is a consistency score: 1 minus the RMS relative deviation of a small endpoint metric vector (Table 13) from the **ten-run mean** of the rule model.
  - Rule control against its own mean: 0.911.
  - GPT-4o 0.898, DeepSeek-V3 0.900, Qwen3 0.885, each over 3 seeds.
- [PAPER] Trajectory agreement rests on two qualitative figures (HK at ε = 0.15 gives three clusters; SIR on Watts–Strogatz, Fig. 18) and one tempo probe: Schelling fully satisfied by step 12 with scalar context, step 18 with enriched context.
- [PAPER] LLM conditions score **above** the control on Opinion Dynamics and Minority Game, because "LLM runs sit closer to the ten-run rule mean" (§5.2.1).
  - [INFERENCE] So the metric rewards reduced variance. A run that always landed on the reference mean would score 1.0. The score cannot tell a reproduced mechanism from a reproduced mean.
- [PAPER] Case 3 extends Schelling to Chicago against five rule references. Case 2 uses five opinion-dynamics ABMs as the rule tier.
- [PAPER] Case 1 and Case 2 numbers are adapted from companion papers [43] and [11], not re-run in SocioVerse2 (captions of Fig. 10 and Fig. 11; Table 7 note).

**(e) Strength of the validation claims.**
- **Case 2.**
  - [PAPER] Hybrid beats pure rules on correlation in 15/15 pairings. Corr is computed over 14 points, averaged over 3 runs. Stance F1 is 0.34–0.37 against accuracy of 0.90–0.97 (Table 6).
  - [PAPER] Table 7's hybrid rows are the configurations with the best correlation or the lowest bias, and A.3 says the best model is chosen per scenario "based on macro-level trajectory fit".
  - [INFERENCE] That is selection on the reported metric.
- **Case 3.**
  - [PAPER] About half the Black–White gap is recovered in 5 seeds, and the tract-level R² is 0.79. The stopping point is set by hand-set anti-overshoot constraints; removing them overshoots.
  - [PAPER] The claim that asymmetries arise "without any hand-coded preference rule" conflicts with archetypes that carry calibrated "ideal own-group percentages" given to the LLM (§5.3.1, A.4).
  - [INFERENCE] With about 85% of households never displaced, much of the R² is inherited from the initial state.
- **Case 4.**
  - [PAPER] Single training seed. The checkpoint is chosen by a Pareto and lexicographic rule over the three objectives that are then reported; the text says the improvement "follows from the checkpoint-selection rule". Winner-set F1 improves by +0.006.
- **Case 6.**
  - [PAPER] Post-cutoff MAE 0.892 against persistence 0.894. Direction 58.6% against consensus 51.7%.
  - [PAPER] The earlier kernel and the mechanical-threshold twin both reach 65.5% direction (Table 16). A homogeneous-persona ablation lowers level MAE by 0.171 (Table 17).
  - [INFERENCE] The 58.6% versus 51.7% gap is about 2 calls out of 29.
- **Cases 5 and 7.**
  - [PAPER] Each is a single run, and the ground truth is blended into the output (see (c)).
- **Overall.** Agreement is endpoint-level or calibrated, with few seeds. No counterfactual contrast is validated or reported with its variance.

**(f) Not transferable / cautions**

- The Lewin split into P and E fails for EPModel. There, each person's environment is the other agents, and a coach is a Person with ties (M1.E), so an intervention is not "an operation on E with P and f fixed".
- MCP data services, persona pools, IPF, LLM or RL behaviour, archetype batching, nowcasting and the skill pipeline with human gates have no role in a rule-based, data-free model of 12 agents.
- Replay-based forking solves an LLM-cost problem that EPModel does not have.
- Do not cite these cases as precedent for evidential standard. Single calibrated runs, selection on validation fit, and blending in the target are all things M11.F.9 and M17 forbid.
- For the exploratory LLM line only: any rule-versus-LLM agreement score has to compare distributions, not distance to the reference mean, or a variance-collapsed agent outscores the reference.

### B. Candidate additions to the EPModel spec

**Already covered:**
- Decision source label (LLM, rule, fallback, replay): M16.A.3c / C12.
- Audience- and scope-limited observation: M3.E / C6, C7.
- Identifiers never reused: M1.A.20 / C2.
- Seed-deterministic construction: M3.D.5.
- Run reads nothing external: M16.B.
- Ablation treated as a different model: M11.4f.
- Calibrated reconstruction reported as a fit: M11.F.9(c), last sentence.
- Misfits reported beside fits: M17.G.3.
- Single-run disclosure: M17.A.1.

### SV1. Prefix invariance: arms do not diverge before the tick where they first differ

- **Proposed requirement.** For any two arms that differ only in a declared exogenous input first active at tick t*, the logs **MUST** be byte-identical for every tick before t*, and the engine **MUST NOT** read any scheduled input before its activation tick. If Phase E ever forks from a snapshot rather than rerunning from tick 0, then:
  - a fork taken at t* with the identity operation **MUST** reproduce the parent byte for byte to the horizon;
  - the control arm **MUST** be produced by the same fork path as the treated arm, not taken from the parent run.
- **Where it would live.** M11.D, beside M11.D.15; referenced from M17.
- **Evidence.**
  - §3.3 eqs 5–6: branches reproduce the parent's rows exactly up to t* (SHOWN as design, Fig. 4b).
  - Fig. 4b and eq. 7: the control is itself a branch (SHOWN as design).
  - §4.3: as-of access holds the treated branch to the same information discipline (ARGUED).
  - The in-engine look-ahead form and the snapshot-completeness form are INFERENCE.
- **What it changes.** Adds a test; improves inference validity.
  - It catches an engine that reads the future schedule, for example an agent anticipating a nodal-calendar entry or normalising spell hazards over the whole horizon.
  - It catches a key-composition defect in M3.D.4a that shifts pre-t* draws.
  - In the snapshot form, it catches incomplete state copies: in-flight events on per-edge latency, the M16.D event store, I-POSITION state machines, open spells, belief hysteresis.
  - Neither M11.D.5 (one arm against itself) nor M11.D.15 (an extra mechanism at zero magnitude over the whole run) can see these defects.
- **Cost / risk.** Low: one pair of runs and a log diff. No conflict with M3.D.4–6 or M16.B.

### SV2. The mid-run intervention arm, declared as data

- **Proposed requirement.** A Phase E intervention arm **MUST** be declared as δ = (t*, operation):
  - the operation is restricted to M17.D.3's channels;
  - it enters as an ordinary exogenous Event in the exogenous step of t*, never as a direct state edit;
  - it is validated against declared targets before the run;
  - its activation is written to the run log at t* with its declaration.
- Every arm specification **MUST** also state whether it is:
  - a **branch**: shares the prefix, differs only in exogenous input from t*, and is the only kind reportable as the effect of an intervention on a family; or
  - a **version**: differs from tick 0 in initial state, constants or mechanism; runs from tick 0; M11.4f applies where a mechanism differs.
- For a branch, readouts **MUST** be reported as the paired difference for ticks from t* on, and SV1 applies.
- **Where it would live.** Phase E, M17.D; with M16.A for the activation record.
- **Evidence.**
  - §3.3, Fig. 4a: an intervention is a declared, validated, logged entry (SHOWN as design).
  - §3.4, eq. 8: branch versus version (ARGUED).
  - Revision 10's own notes record that M17.D.3 "has no mid-run perturbation".
- **What it changes.** Fills that stated gap. It also makes explicit which arms may be read as interventions on a family and which only test structure. Improves inference validity.
- **Cost / risk.** Low. Consistent with M17.F.1, which requires a shock to be an ordinary nodal event, and with M17.D.3 arm-blindness (δ is data the engine executes, not an arm label).
- **Open question for the owner.** An intervention that adds a coach mid-run changes the set of Persons. The spec has to say whether such a person is present from t0 with a dormant tie, or whether the arm counts as a version.

### SV3. Import has a point-in-time rule: later-dated information is held out, not used as input

- **Proposed requirement.** The importer **MUST** classify every dated item as an exogenous input or a held-out outcome, using M14.A's register.
  - Items the model generates endogenously **MUST NOT** be scheduled as inputs. These include cutoff, divorce, symptoms, illness, job loss, identified patient and tie-state transitions.
  - Nothing dated after t0, or after an arm's t*, may set the state at that tick.
  - Every rating **MUST** carry both the date it was made and the period it describes. A rating made after that period is retrospective: it **MUST** be flagged and imported as a range at least as wide as a contemporaneous one.
  - Any comparison of a run against held-out outcomes **MUST** be reported as a check under M11.F.9, never used to adjust the state at t0.
- **Where it would live.** M15.A and M15.B.
- **Evidence.**
  - §4.3: vintage-safe as-of access with provenance stamps (SHOWN as design).
  - §5.4.1 and §5.4.2: targets excluded from agent inputs (SHOWN as practice).
  - §5.4.3 and Table 14 are the counter-example: ground truth blended into output makes the result a reconstruction, as the authors themselves say (SHOWN).
  - The transfer to family import is INFERENCE.
- **What it changes.** Corrects a gap: M15.A.3 imports transitions after t0 without saying what they are for. It generalises M15.A.8's rule (identified patient is an output) to all endogenous outcomes. Improves both theory fidelity and inference validity.
- **Cost / risk.** Low to moderate: one classification column, plus a second date on each rating in the export contract. It supports M11.F.9(c) and is in no conflict with it.

### SV4. Version lineage and the count of versions tried

- **Proposed requirement.**
  - The M16.A.1 header **MUST** carry a study-version identifier and its parent.
  - Every ensemble report **MUST** state how many versions of the study state were run against the same criterion before the reported one, and which component each changed: rules, D0, constants, spell schedule, readout definition.
  - A result from a version chosen after its outcome was inspected **MUST** be marked post-hoc.
- **Where it would live.** M16.A.1 and M17.G.1.
- **Evidence.**
  - §3.4: version manifest with id, parent, note, time and fork step (SHOWN as design).
  - §5.1 and §6.1: the paper reports the accepted version and concedes the tree is not reported (ARGUED; a stated limitation).
- **What it changes.** A reporting rule; improves inference validity.
- **Overlap.** Partly covered. M10.B.4 and M16.A.7 cover changes to `[I]` constants only, and M17.G.1 covers perturbation dimensions but not the selection history. This extends the same rule to code and to readout choices.
- **Cost / risk.** Low: git supplies the lineage. No conflict with any stated rule.

### C. Search terms

- "inject, fork, compare" multi-agent simulation (ref [10], Lee et al. 2025)
- "counterfactual branching" agent-based simulation
- "replay-based branching" simulation
- "common random numbers" agent-based model counterfactual
- "look-ahead bias" "data vintage" backtest simulation
- "point-in-time" "real-time data" nowcasting agent-based
- "simulation provenance" experiment versioning reproducibility
- "checkpoint restart" agent-based model reproducibility
- "longitudinal" agent-based simulation "panel" trajectory
- "trajectory fidelity" social simulation validation
- "WhatIf" interactive exploration social simulation policy (ref [9])
- "AgentSociety 2" executable social science (ref [8])
- "reforms as experiments" simulation counterfactual (Campbell, ref [26])
- "consistency score" LLM agents canonical ABM benchmark (ref [43])
- "hybrid" LLM rule-based "opinion dynamics" mirror agents (ref [11])

---

# Part 3: Nair-Turkich et al. 2026

## Reader report: Nair-Turkich, Campbell & Geard 2026, "Modelling sexual partnership dynamics and population heterogeneities in agent-based dynamic network models" (arXiv 2609.17622v1)

I read the whole text, including Appendices A–B and the supplement (S1–S5: the ODD protocol, the design rationale and the trimmed Python listings). Figures survive only as captions. I checked overlaps with spec v2 rev10 by keyword search only, not by a full read.

### A. Report

**1. Simulation formalism**
- [PAPER] The time step is one day (§2.4.5, Table 6). Each run is 1,875 steps (about 5.1 years) with N = 15,000 agents. Six processes run in a fixed order each day: ageing, sexual debut, removal, replacement, formation, dissolution (S2.4). The stated reason for the fixed order is to stop an agent forming and dissolving a partnership on the same day. A partnership cannot dissolve on the day it forms. The text is inconsistent on the boundary: §2.4.2 says `d >= 1`, S2.8.2 says `d > 1`.
- [PAPER] Formation is sequential. Eligible agents are processed "in random order". Each makes one Bernoulli attempt per day, and at most one new partnership forms per agent per step. An agent whose attempt finds no partner stays in the pool as a passive candidate only (S2.8.1).
- [PAPER] Dissolution is one Bernoulli trial per active partnership per day, evaluated in one vectorised call (S2.8.2, Listing 2).
- [PAPER] Birthdays are staggered. Each agent gets an offset drawn from DiscreteUniform(0, 364) "to avoid synchronised ageing events" (§2.2.2, eq. 2). No artefact from synchronised ageing is shown; this is only the stated reason.
- [PAPER] All draws go through one seeded `default_rng(seed)`, so that "a given (config, seed) pair is fully reproducible" (S4.1). The SIS listing (S4.5) calls the global `np.random.rand`, which does not go through that seeded generator.
- [PAPER] Replacement agents get the identifier "one greater than the previous maximum" (§2.4.4). Array slots are reused, with an id-to-slot map (S4.1).

**2. Agent architecture and heterogeneity (owner item b)**
- [PAPER] Fixed attributes are sex and orientation. Time-varying attributes are age, sexually-active status and the partner set (Table S1). The ODD entries for Adaptation, Objectives, Learning and Prediction all say "None" (S2.5). Agents do not change their behaviour in response to their partnership history.
- [PAPER] Each rate is a baseline for a reference stratum times an orientation multiplier times an age multiplier (eqs 8–9; 36 strata). The age multiplier is a youth boost β for ages 16–24 and exp(−κ(g−2)) above 34.
- [PAPER] Each agent draws X_i from NegBin(r = 0.5, p = 0.5) and gets the multiplier η_i = 1 + X_i/E[X_i] ≥ 1 (eq. 13). Formation and dissolution get separate, independent draws. The authors say explicitly that there is no correlation between an agent's propensity to form partnerships and its propensity to dissolve them (S2.8.3). Probabilities are clipped to [1e-4, 0.99].
- [INFERENCE, my arithmetic] With those constants P(X = 0) = 0.5^0.5 ≈ 0.707. So about 71% of agents have η exactly 1, the rest have η ≥ 3, and E[η] = 2 by construction. The "heavy tail" is mostly a point mass plus a jump. The calibrated baselines absorb the factor of 2.
- [PAPER] Concurrency is a fixed trait assigned at initialisation. A share θ_conc of agents are eligible, each with a cap K_i = max(2, Poisson(λ = 2)); everyone else is strictly monogamous (§2.3, eq. 7). The stated reason is that a fixed trait keeps the mechanism identifiable (S3).
- [INFERENCE] With λ = 2, P(K = 2) ≈ 0.68, so most concurrency-eligible agents are capped at two partners.

**3. Interaction and ties (owner item a)**
- [PAPER] Formation (§2.4.1, eq. 10, S2.8.1):
  - A daily Bernoulli draw at the stratum rate times η.
  - Partners must be compatible by sex and orientation.
  - The partner is picked with a Gaussian age-assortative weight, σ = 4 years (fixed, not calibrated; S3).
  - If no compatible candidate exists, no partnership forms.
- [PAPER] Dissolution is duration-dependent (eqs 11–12). The hazard is p_diss(d) = p_base,s,o,g · (1 + d/α)^−γ with α = 1500 days and γ = 2. α and γ are fixed and were not in the LHS (Table 6). The justification is empirical (Nelson et al. 2010). S3 argues that a constant daily probability would imply a memoryless duration distribution.
- [INFERENCE, my arithmetic]
  - The form is a Lomax/Pareto-II decay, not a Weibull, despite the paper's "Weibull-like" label.
  - The hazard multiplier is f(365) ≈ 0.65, f(1500) = 0.25 and f(1875) ≈ 0.20. Within the run the hazard never gets near the "near-zero" the text claims; that is only a long-horizon limit.
  - Figure S4's caption swaps α and γ.
- [PAPER] The duration hazard is the only memory in the model. Rates otherwise depend only on stratum and multiplier (Discussion, limitations).
- [PAPER] When an agent ages out at 75, the surviving partner keeps the partnership as an "external" partnership. It stays at risk of dissolution under the same hazard (§2.4.4). In Listing 1 it also counts toward a concurrency-eligible agent's cap (`partner_count + len(external_partner)`). External partnerships are excluded from transmission (S2.8.6).
- [PAPER] In the trimmed Listing 2, a partnership's base dissolution probability is taken from `breakage_probs_arr[i]` for the lower slot index `i < j` only. The prose says the risk depends on "the participating agents". Because slots are reused (S4.1), which partner sets the rate is arbitrary. This is the trimmed code; the full repository may differ.
- **Transfer to marriage, divorce, separation and pairing [INFERENCE]:**
  - The hazard is an empirical, stratified event generator with no mechanism behind it. Importing it would conflict with the brief's rule that population statistics are a calibration target, never an event generator.
  - What transfers is three things. First, the modelling choice that ending rules have a hazard shape that must be declared (S3's memoryless argument). Second, the tie-episode record with a censoring flag (Table S2). Third, the treatment of a tie to a departed person as persisting and still occupying capacity; this matches the owner's guidance under M6.3.
  - The Gaussian kernel is a soft alternative to M2.A.0e's hard ±1 tolerance. It gives a declared distribution of partner differences (Fig. S5). I am not proposing it, because M2.A.0e is an owner amendment.

**4. Initialisation, burn-in and replicates (owner item d)**
- [PAPER] All agents start with empty partner sets (S2.6). Initial ages are drawn uniformly over 16–74 by block allocation; debut is sampled from a cumulative table (Table S3).
- [PAPER] There is no burn-in period for the partnership network. Outputs such as mean partner counts over about five years are computed over the whole run from t = 0. The disease is seeded at step 51 (Table 7).
- [INFERENCE] The duration hazard's timescale is α = 1500 days, so the network is far from stationary at day 51. The early days of every run are relaxation from an empty network, and they are counted in the outputs.
- [PAPER] Partnerships still active at the end of the run are flagged as censored (Table S2). The paper does not say how censored partnerships enter the mean durations reported by age group (Fig. 7b, Discussion). [INFERENCE] Truncation at 1,875 days and the start from empty both bias those means.
- [PAPER] Outputs are "averaged over 100 simulations" (§3.1, Table 6). Table 2's caption says 1000 runs, which contradicts that.
- [PAPER] The SIS results are 100 runs on one randomly selected network. The authors state that this variation "reflects the stochasticity in disease seeding and transmission alone" and not network variation (§2.6). This is a correct statement of which variance source is being reported.

**5. Calibration and sensitivity (owner items c and e)**
- [PAPER] The LHS design has 60,000 samples per scenario, generated with the same seed so the two scenarios (θ_conc = 0 and 0.15) get identical parameter sets (§2.5.1).
- [PAPER] About 16 dimensions are sampled:
  - 2 baseline probabilities;
  - 5 orientation multipliers × {formation, dissolution}, each over [0.5, 6.0] (Table 3);
  - β over [2, 4] and κ over [0.1, 1.0], each × {formation, dissolution} (Table 4).
  
  The range of the baseline probabilities is not given in the text I have. The marginal distribution within each range is not stated anywhere.
- [PAPER] The loss is the MSE over 12 sex × age cells per orientation, averaged over the three orientations (eqs 15–16). The targets are mean partner counts in the past five years only (Table 1). S2.5 says degree, duration and concurrency statistics are "used as model calibration targets", which contradicts §2.5.2.
- [PAPER] Selection is the single lowest-MSE sample in each scenario: id 2777 (global MSE 0.466) and id 557 (0.540) (Table 5). Figure S2 shows the top three. The paper gives no acceptance threshold, no count of acceptable sets, no parameter uncertainty and no identifiability analysis. It never says how many replicates were run per LHS sample; each simulation records "seed values", which suggests one run per sample.
- [PAPER] Two targets, same-sex males aged 25–34 and 35–44, were excluded as outliers because they did not follow the monotone age pattern. S2.8.8 shows that exclusion was decided after seeing that those cells dominated the MSE.
- [PAPER] The best-fit parameters differ between the two scenarios by ratios from 0.33 to 3.03. For example, the same-sex female formation scale is 1.804 at θ_conc = 0 and 0.595 at 0.15 (Table 5). The authors attribute calibration differences "primarily ... to the inclusion of concurrency".
- [INFERENCE] The paper cannot support that attribution:
  - The winner is the minimum over 60,000 noisy, probably single-seed runs, so its identity is itself a random variable.
  - Mean counts alone cannot separate formation rates from dissolution rates, since a monogamous agent's count depends on both.
  - The large parameter swings at similar MSE look like equifinality, not an effect of concurrency.
- [PAPER] There is no sensitivity analysis. The fixed constants (α, γ, σ, λ, r, p, the clip bounds) are never varied against outputs. Figures S3–S6 show only how each constant changes its own input distribution. Concurrency is tested at two levels only. Sensitivity work is deferred to future work (Discussion).
- [PAPER vs INFERENCE] The central comparison is confounded. The paper says of monogamous agents across scenarios that "the behaviour of these agents is identical across both scenarios" (§3.3, Fig. 12), and reports a 6–8× rise in their risk. But each arm runs on its own separately fitted parameter set (Table 5), so monogamous agents do not have identical rates across the arms. This is the failure M11.F.9(c) and M11.4f exist to prevent: each arm is fitted separately, so the invented constants no longer cancel.
- **Reusability for EPModel [INFERENCE].** The LHS design itself is reusable without the fitting step: stratified samples over each dimension, and the same design applied to both arms. What must be dropped is the MSE loss, the argmin selection, post-hoc target exclusion and per-arm refitting. EPModel runs are about 2,080 ticks × 12 agents, which is cheap, so several seeds per LHS sample are affordable. The paper does not do that.

**7. Software engineering**
- [PAPER] The model is documented with a full ODD protocol (Grimm 2020) in S2. Every partnership is logged with partner ids, demographics at formation, start, end, duration, a censoring flag and an external flag (Table S2). Network snapshots are rebuilt from that flat log (S2.8.5). Code is public; two of the three repository URLs listed are the same.
- [PAPER] Other editorial defects: references to nonexistent sections ("Sections 2.19–2.21", "Section 2.13"), the captions of Figs 20 and 21 both say "bisexual", and Fig. S4's axes are mislabelled.

**8. Small N, families and long horizons**
- [PAPER] None of these are covered. The model has no households and no emotional state, and the horizon is five years.

**Not transferable, and cautions**
- Everything population-scale does not transfer to a 12-agent family: the 36 strata, NATSAL targets, degree distributions, shortest paths, bisexual "bridging", ego networks, SIS prevalence, and the finding that monogamous agents face risk created elsewhere in the network. A 12-person family has no strata to calibrate and no network large enough for path statistics.
- The empirical duration hazard and the formation rates must not enter EPModel as event generators.
- The paper practises four things EPModel forbids: best-fit selection, per-arm refitting, post-hoc target exclusion and reporting a counterfactual from fitted parameters. Cite it only for the LHS design, the censoring-flagged episode record and the staggered-phase idea, not for its standard of evidence.

### B. Candidate additions

**PD1. A tie-episode record with a censoring flag, and duration readouts that account for censoring**
- **Proposed requirement:** The log MUST emit one record per tie episode: marriage, cutoff spell, distance spell, and any other declared episode kind. Each record carries the start tick, end tick, end cause (death, rupture, reunion, horizon) and a right-censoring flag. Any duration or time-to-rupture readout over an ensemble MUST treat censored episodes as censored (a survival curve or hazard by tie age), and MUST NOT report a mean over completed episodes only.
- **Where:** M16.A, plus Phase E (M17.F).
- **Evidence:** Table S2 records censoring (SHOWN, as a method). Fig. 7b and the Discussion report mean durations with no stated treatment of censored partnerships, which is the failure this would prevent (INFERENCE).
- **What it changes:** Adds a reporting rule. Improves inference validity. It also gives the owner a way to see whether a tie-age dependence of marital rupture emerges from the mechanism, rather than building one in.
- **Cost / risk:** Low; the record is observer-side, so M16.B holds. Partly covered: M17.F.1(c) requires the fraction of seeds in which the event does not happen for shock time-to-event readouts, and DL §2.2 asks for time-to-event distributions. Neither covers tie episodes or end cause.

**PD2. Every duration draw declares its distribution and hazard shape**
- **Proposed requirement:** For each spell class in M1.F.6, and for the mortality hazard in M7.C.1, the M10 register MUST state the duration or hazard distribution and its shape (constant, decreasing or increasing with elapsed time), graded `[I]`. Phase E SHOULD run each criterion under at least one alternative shape. A shape is a structural choice, which a numeric sweep of its parameters does not test.
- **Where:** M10, plus M17.E.
- **Evidence:** S3 argues that a constant rate implies memoryless durations (ARGUED). α and γ were fixed and never varied (Table 6), and Fig. S4 shows how much the curve depends on them (SHOWN, but for input curves only).
- **What it changes:** Adds a reporting rule and a sweep dimension. Improves inference validity.
- **Cost / risk:** Low to moderate. It is consistent with M1.F.6's ban on per-tick stressor draws.

**PD3. Stagger the slow-tick phase per person, or test the synchronous version as an assumption**
- **Proposed requirement:** Either (a) each person SHOULD carry a fixed slow-tick phase offset (birth tick for persons born in the run, a declared offset for founders), so that annual updates are staggered; or (b) M3.B.1's synchronous annual update MUST be graded `[I]`, and M17.E.6 MUST include a staggered-phase regime.
- **Where:** M3.B, plus M17.E.6.
- **Evidence:** The paper staggers birthdays "to avoid synchronised ageing events" (§2.2.2) (ARGUED; no artefact shown). DL §2.1 on synchrony artefacts is the stronger support (INFERENCE).
- **What it changes:** Would change M3.B.1, which says the slow tick fires every 52 fast ticks. Under that rule, drift, life stage, mortality and chronic anxiety change for all twelve people in the same tick. Improves inference validity.
- **Cost / risk:** Option (a) amends M3.B.1 and needs an owner decision. Option (b) is cheap. Both are compatible with M3.D.4a keyed draws.

**PD4. The M17.E.1 sweep uses a declared Latin hypercube with a declared sampling measure, paired arms, several seeds per sample, and no selected sample**
- **Proposed requirement:** M17.E.1 MUST do all of the following:
  - Draw its constant samples as a Latin hypercube with a declared sample count and a logged design seed.
  - Declare the marginal measure for each constant over its range. Multipliers and ratios SHOULD be reported under both uniform and log-uniform measures.
  - Evaluate each sample in both arms over a declared number of seeds, so each sample yields a per-seed paired arm difference.
  - Report the fraction of samples in which the direction holds, with its interval.
  - Never select, rank or report a "best" sample against any target, including the M10.C.4 corpus bounds.
- **Where:** Phase E (M17.E.1), plus M10.
- **Evidence:**
  - The paper applies one identical 60,000-sample design across both scenarios (§2.5.1) (SHOWN, as a method).
  - It does not state the marginal measure. [INFERENCE] If the Table 3 ranges [0.5, 6] were sampled uniformly, about 91% of the mass sits above 1. A "fraction of range" is therefore only defined relative to a declared measure.
  - One stochastic run per sample plus argmin selection gave parameters that swing 0.33–3.03× at similar loss (Table 5) (SHOWN).
- **What it changes:** Turns the LHS design, which the rev10 notes record as "noted, not specified", into a stated design. It also closes a gap in M17.E.1, which has no sampling measure.
- **Cost / risk:** Moderate compute, which is affordable at N = 12. The ban on a best sample restates M11.F.9(c) and M10.C.4 ("checks, never parameters"). Reporting a fraction conditional on satisfying the bounds would be history matching, which is DL §2.11's open question for the owner; I do not propose it here.

**Already covered (one line each)**
- Arms refitted separately and then compared (the paper's Fig. 12 confound): M0.4, M11.F.9(c), M11.4f.
- Target exclusion decided after seeing the fit: M10.B.4 (C23).
- No burn-in; readouts start from an empty initial state: M17.F.2 and DL §2.5.
- Replacement identifiers allocated from a run-time counter: M1.A.20 (C2).
- A single stateful generator, and the global RNG in the SIS code: M3.D.4a (C1).
- The dyad rate taken from the lower-slot partner (Listing 2): DL §7.10(b) permutation-equivariance and M11.D.16 would detect it. The listing is a concrete instance of what those tests catch.
- A tie to a departed person persists and occupies capacity: M6.3 (C15) and the owner's 2026-09-22 guidance.
- Independent per-agent multipliers with undeclared correlation: M17.C.1 (C32).
- Separate variance sources (the SIS runs vary disease draws only): M17.C.1 and DL §2.6.
- Random-order sequential formation: M17.E.6 and C30.

### C. Search terms

These come from the paper's vocabulary and its cited literature. The last two do not appear in the paper; they are my additions from the wider field.

- "pair formation model" (Kretzschmar & Heijne 2017)
- "duration-dependent dissolution hazard"
- "partnership formation and dissolution rates"
- "serial monogamy agent-based model"
- "concurrent partnerships dynamic network model"
- "temporal network model" partnership turnover
- "Latin hypercube sampling" "partial rank correlation coefficient" (Marino et al. 2008)
- "global uncertainty and sensitivity analysis" agent-based
- "equifinality" agent-based model calibration
- "identifiability" agent-based model parameters survey targets
- "right-censored" relationship duration simulation
- "ODD protocol" "structural realism" (Grimm et al. 2020)
- "marriage duration" divorce hazard microsimulation (my addition)
- "separable temporal exponential random graph" STERGM partnership dynamics (my addition)

---

# Part 4: Leins et al. 2026

## Leins et al. 2026, "Prompting Against Persona Drift" (arXiv 2609.24532v1): reader report

### A. Report

**What the study is.** [PAPER] The paper is about LLM student personas in education. It ran 1,200 conversations of 28 turns each. The design was 6 conditions × 4 LLMs × 2 ADHD intensities × 25 seeds (§3.3). The persona models were GPT-5.5, Claude Sonnet 5, DeepSeek V4 Flash and Qwen 3.6 35B, all at default inference parameters (Table 5). The same four models also acted as judges. DeepSeek additionally played the conversation partner and the monitor.

The scenario is a student telling a friend about the school day. The partner is prompted as an attentive listener who does not restate the persona (Table 7). Checkpoint k falls after turn 4k, giving seven checkpoints. Checkpoint 1 (turn 4) is each run's own reference point (§3.3, Fig. 2).

**1. Simulation formalism**
- [PAPER] Each conversation is an independent run with no shared context (§3.3).
- [PAPER] An intervention is a transient user-role message beginning "System:". It goes into the persona agent's next reply only. The partner never sees it, and it is not saved to the transcript or kept in later context (§3.5 "Delivery").
- [PAPER] "Seeds" are not defined anywhere, and all models ran at default temperature (Table 5 note).
- [INFERENCE] This is not reproducible in the M3.D.5 sense.

**2. Agent architecture**
- [PAPER] The persona is a system prompt. The high- and moderate-intensity prompts differ only in frequency words ("often/frequently" vs "sometimes/occasionally", Table 6 note).
- [PAPER] The six conditions (Table 1) were:
  - static full-persona reinjection after checkpoints 2, 4 and 6 (3 interventions)
  - static two-sentence "reflective reminder" on the same schedule
  - adaptive versions of both
  - adaptive "behavior-specific instruction"
  - no-intervention control
- [PAPER] Adaptive conditions could intervene 0–5 times.

**3. Interaction structure.** A single dyad with a neutral partner. Nothing else transfers here.

**4. Initialisation**
- [PAPER] Initial Scale C means were 26.2–27.5 across conditions (Table 2).
- [PAPER] One-way ANOVA at checkpoint 1: F(5,1194) = 0.77, p = .570. The authors say this "does not establish baseline equivalence".
- [PAPER] A covariate-adjusted model still found adaptive reinjection slightly below control at checkpoint 1 (p = .039, §4.2).

**5. Measurement and validation (owner's checks a–e and g)**

**(a) Two-pipeline separation** [PAPER] (§1, §3.1)
- The "observer pipeline measures and reports" behaviour. A separate "correction pipeline schedules and injects prompts" in every intervention condition.
- Adaptive conditions also read observer output to decide *whether* to intervene.
- In the behaviour-specific condition, a third LLM (the monitor) receives three things (§3.5, Fig. 1):
  - the full persona
  - the last four messages
  - an item-by-item table of target vs current observer ratings

  The target is the checkpoint-1 four-judge average; the current value is the triggering checkpoint's average. The monitor writes a 2–4-sentence corrective instruction.
- Judges "see only the segment being rated and are blind to the persona instructions" (§3.4).

**How strict the separation is** [PAPER + INFERENCE]
- It is a separation of process roles, not of information.
- In condition (e) the measurement drives the correction content directly. The authors say the same CAARS items informed both correction and outcome, "so the mechanism may have optimized expression toward the measurement instrument" (§5.4).
- The same models served as persona agents and as judges. The authors note this creates dependencies that a multi-model panel "reduces but does not remove" (§5.6).
- DeepSeek holds four roles at once (Table 5).

**(b) Drift measure and instrument** [PAPER] (§3.4, §4.1)
- The instrument is the Conners' Adult ADHD Rating Scales (CAARS), observer form, Scale C. It has 18 DSM-IV items rated 0–3, for a total of 0–54.
- Four LLM judges rate each non-overlapping four-turn segment. Item ratings are averaged without rounding, then summed.
- Reliability: ICC(3,1) = .899 and ICC(3,4) = .973.
- The only "validation" of the LLM rater is agreement among LLM judges. §5.6 says this "does not establish human validity" and that human ratings are needed.
- Drift is measured against the run's own checkpoint-1 score, "not fidelity to the intended persona" (§5.6). An inaccurate early expression can become the target that gets preserved.
- The adaptive trigger fires on an absolute deviation from the checkpoint-1 score of at least a threshold. The threshold is the standard deviation of checkpoint-1 scores across the 25 control runs, averaged over models: 4.31 (high) and 4.36 (moderate) points (§3.5).

**(c) Effect sizes; none eliminates drift** [PAPER] (Table 3, Table 2)

The control's post-knot slope is about −2.39 points per checkpoint.

| Condition | Slope | Reduction vs control |
|---|---|---|
| Static reinjection | −1.56 | 35% |
| Static reflective reminder | −1.86 | 22% |
| Adaptive reinjection | −1.47 | 38% |
| Adaptive reflective reminder | −1.74 | 27% |
| Behaviour-specific instruction | −0.32 [−0.50, −0.13] | 87% |

- All contrasts with control had Bonferroni-adjusted p < .001.
- First-to-last median change was −21.0 for control (27.5 → 7.6) and −6.9 for behaviour-specific instruction (26.6 → 18.9) (Table 2).
- "None eliminated drift" (Abstract). All intervention trajectories stayed negative (§5.1).
- The behaviour-specific condition bundles several things that cannot be separated: context, item targets, generated text and an extra LLM. It was tested only with adaptive timing, and there is no static behaviour-specific arm (§5.4, §5.6).

**(d) Static vs adaptive timing** [PAPER] (Table 4, §5.2)
- Timing: static − adaptive = −0.11 [−0.29, 0.08], p = .769. No overall difference.
- Adaptive timing was better only at checkpoints 4 and 6 (checkpoint-factor analysis).
- Adaptive timing did not reduce the number of interventions. Medians were 5 (reminder), 4 (reinjection) and 3 (behaviour-specific), against a fixed 3 for static (Fig. 3).
- The authors' explanation: drift-triggered intervention "is necessarily reactive", and the threshold was crossed too often to be selective. They limit the result to "this particular trigger policy".
- Content: reinjection beat the reminder, −0.28 [−0.46, −0.10], p = .008. The authors call this "tentative".
- Timing × content interaction: p = 1.000.
- Intervention count in the behaviour-specific arm is endogenous (a successful correction lowers the chance of another trigger), so it is not evidence of efficiency (§5.4).

**(e) How drift develops over 28 turns** [PAPER] (§3.6, §4.1, Table 8, Table 10, §5.1, §5.6)
- **Direction:** expressed ADHD symptoms fall. The persona becomes more composed. Fig. 1's example correction describes the agent as "wrapping up too neatly and calmly".
- **Shape:** on control data, a piecewise model with a knot at checkpoint 2 fit best (AIC 8705.79). It beat linear (8892.98; χ²(1) = 189.19), quadratic (8773.36) and logarithmic (8723.85). There is a steep early drop from turn 4 to turn 8, then a steady linear decline.
  - Caveat: the knot was placed for design reasons ("interventions delivered after that checkpoint can first affect checkpoint 3").
- **Continuation:** the control "continued declining through turn 28". Whether it levels off is unknown, and longer horizons are listed as future work.
- **Sawtooth under static schedules** (Table 10): between checkpoints with no prompt, mean change was −4.77 (reminder) and −5.19 (reinjection). Between checkpoints with a prompt it was −0.55 and +0.82. Each reinjection restores the persona and then it decays again.
  - The no-prompt category includes the first-interval drop, so position in the conversation is confounded.
- **Frequency** (§4.6, Tables 11–13): a frequent schedule (5 prompts) was slightly smoother and declined slightly less. For example, checkpoint 6 was +4.59 [2.94, 6.21] higher for reinjection. These were separate runs, analysed descriptively only.
- **Cited mechanisms** (§2.2): attention decay, with drift in as few as eight rounds (Li et al. 2024). Sycophancy, which may pull models away from personas that "depart from socially expected behavior".

**(g) Persona intensity** [PAPER] (§3.2, §4.5, §5.3, Table 9, §5.6)
- Intensity was treated as a replication condition, "rather than a formal independent variable".
- The paper does not report a test of whether control drift rate differs by intensity. Per-intensity control slopes are not given in the text.
- Low-intensity and default personas were excluded because prior work "show[s] negligible drift" there. So the value of intervention when drift is weak "remains unknown".
- Stratified results:
  - High intensity: all five interventions beat control.
  - Moderate intensity: reflective reminders were not significant (p = .058 and .089). Reinjection and behaviour-specific instruction were.
  - Slope differences are larger at high intensity, e.g. behaviour-specific instruction 2.254 vs 1.892.
- The authors' hypothesis is that a generic reminder can reactivate a salient persona but gives too little guidance for "subtler behavioral expression", which sycophancy may override more easily. They state that a content-by-intensity interaction was not tested.

**6. Failure modes and limits** [PAPER] (§5.5, §5.6)
- "Stability is not validity." Stabilisation can preserve an inaccurate or stereotyped profile.
- Not evaluated: naturalness, stereotyping, interaction quality, cost.
- Scope: one scenario, one persona family, four models, 28 turns.
- Several mixed-model fits were singular. The authors say the focal estimates were stable across optimisers and uncorrelated refits (§3.6, B.1).

**7. Software and method lessons** [PAPER] (§3.6)
- The functional form was chosen on control data before the planned tests.
- Bonferroni families were pre-specified.
- Checkpoint-1 balance was checked, and a null result there was explicitly "not interpreted as evidence of equivalence".
- Checkpoint-factor analyses were used to show effects hidden by the averaged slope.
- [INFERENCE] These are reasonable Phase E reporting habits. They are not new relative to C27 or C25.

**8. Small-N, emotion, long horizon**
- The horizon is 28 turns (14 persona replies). EPModel runs about 2,080 ticks.
- [INFERENCE] The paper says nothing about decades. It does show that within a single rendered exchange, a low-regulation persona decays toward composure starting from about the first eight turns.

**(f) Implications for an age- and profile-specific narrator** [INFERENCE]
- **The drift direction is the problem.** A reactive persona (the "15-year-old brat") is off the socially expected default in the same way ADHD expression is. The expected artefact is rendered calming. In a narrated trace this would read as de-escalation or maturation that the engine did not produce, which is counterfeit maturation.
  - Design lessons §3.2 predicted this ("regress toward composure"). This paper shows it for one persona family across four current models.
- **The shy 10-year-old is untested.** The paper only measured an over-expressed trait. For a low-expression persona the drift direction could be toward more talkativeness, or there may be little drift, as with the excluded low-intensity personas. This needs its own measurement.
- **Re-send the full profile on every call.** Full-persona reinjection beating a pointer to the system prompt, plus the sawtooth, argue for sending the full engine-derived profile each time. Don't rely on accumulated prose to carry the persona. The sawtooth also means that on any schedule, fidelity varies with time since the last reinjection. That variation would appear in the narrative as mood variation.
- **EPModel escapes the anchor problem.** EPModel has an exogenous target: the engine state for that tick. The paper's weakest point is anchoring to the model's own first expression (§5.6).
- **Keep the scoring instrument away from the correction step.** If a monitor-and-correct loop is used, the §5.4 Goodhart risk applies directly. The instrument used to score fidelity must not be the one fed to the corrector.
- **Stability does not establish validity.** A stably rendered 15-year-old may be a stable stereotype.

**Not transferable / cautions**
- One scenario and one clinical persona family. The persona models doubled as their own raters.
- No human validation. "Validated instrument" refers to CAARS with humans, not to LLM judges applying it to LLM text.
- 28 turns only. Default temperatures and undefined seeds.
- The 87% figure comes from a bundled, adaptive-only, instrument-targeted condition. Nothing here bears on the v2 decision engine beyond supporting M3.D.6 and M16.E.

---

### B. Candidate additions (X-line, Phase F narrator under M16.E)

**X5. Narrator receives the full state profile on every call; no persona carried in context**
- **Proposed requirement:** Each narration call MUST receive the complete persona-and-state profile for that agent and tick (age, life stage, differentiation-derived descriptors, current anxiety, chosen move, tie context), derived deterministically from the run log. The narrator MUST NOT depend on earlier narrated prose in its context window to maintain the persona. Where a multi-turn rendered exchange is produced, the full profile MUST be re-sent on every turn.
- **Where it would live:** Phase F narrator (M16.E), as the input contract; an extension of design lessons §7.6.
- **Evidence:**
  - SHOWN: reinjection beats a generic reminder (Table 4, p = .008, tentative); the control declines continuously; the sawtooth under a 3-prompt schedule (Table 10).
  - ARGUED: attention decay as the mechanism (§2.2).
  - INFERENCE: that a stateless per-call design removes most of the problem. The paper did not test it.
- **What it changes:** adds an input contract for the narrator. It improves the validity of any narrated trace.
- **Cost / risk:** low. Consistent with M3.D.6 and M16.E.3.

**X6. Constant-state rendering drift audit**
- **Proposed requirement:** Before any narrator is used, the project MUST run a constant-state audit. Hold one agent's engine state fixed, narrate K consecutive turns or ticks (K at least 28, the paper's horizon), have a blind rater score the target traits, and report the slope. Any trend under constant state MUST be reported as narrator drift, and its direction MUST be declared per persona type, both high-expression (reactive adolescent) and low-expression (shy child). A narrated change in reactivity MUST NOT be presented as agent change unless it exceeds this floor.
- **Where it would live:** Phase F narrator acceptance, next to M11.D.20.
- **Evidence:**
  - SHOWN: control drift of −21.0 median over 28 turns, toward composure (Table 2, Fig. 1).
  - SHOWN: low-intensity personas excluded for negligible drift, so the effect is persona-dependent (§3.2).
- **What it changes:** adds a test and a reporting rule. It guards theory fidelity, because a composure artefact would look like maturation.
- **Cost / risk:** moderate, since it needs an LLM rater. Partly overlaps X1, which sets a no-event retest floor for persona change claims. X6 applies the same idea to the rendering layer under fixed engine state, so it could be folded into X1 as a narrator clause.

**X7. Rater separation and instrument hold-out for narrated output**
- **Proposed requirement:** Any rater of narrated output MUST:
  - see only the rendered segment
  - be blind to engine state and to the narrator prompt
  - be a different model from the narrator

  If a correction step uses rater output, it MUST use a different instrument or item set from the one used to report fidelity. Inter-rater agreement among LLM judges MUST NOT be reported as validity.
- **Where it would live:** Phase F narrator evaluation (M16.E.3's traceability clause).
- **Evidence:**
  - ARGUED, limitations-grade: §5.4 (optimising toward the instrument); §5.6 (same models as agents and judges; ICC does not establish human validity).
  - SHOWN: judge blinding and ICC .973 among LLM judges (§4.1), which is exactly the result that should not be over-read.
- **What it changes:** adds a reporting and validity rule. Extends design lessons §3.4 with the Goodhart point and the point that ICC is not validity.
- **Cost / risk:** low to moderate. No conflict with EPModel rules.

**X8. Fidelity target is the engine state, never the narrator's own early output**
- **Proposed requirement:** Any narrator fidelity or drift measure MUST compare the rendered expression with a target derived from engine state for that tick. It MUST NOT use a run-local anchor taken from the narrator's first renderings. Where engine state changes, the target changes with it.
- **Where it would live:** Phase F narrator evaluation.
- **Evidence:** ARGUED. §5.6: the checkpoint-1 anchor measures stability, "not fidelity"; §5.5: stabilising can preserve an inaccurate profile.
- **What it changes:** corrects a measurement design that would otherwise be copied from this paper. It also keeps real engine-driven change (e.g. a de-escalating adolescent) from being "corrected" back toward the first rendering.
- **Cost / risk:** low. It requires a declared mapping from engine state to expected rendered traits, which is itself an [I] artefact and must be labelled so.

**X9. If a correction loop exists: fixed schedule as baseline, matched budgets, endogenous counts**
- **Proposed requirement:** Any drift-correction loop in the narrator MUST be compared against a fixed-schedule full-profile reinjection baseline with a matched intervention budget. Realised intervention counts MUST NOT be reported as efficiency. Time since the last correction MUST be logged per rendered segment, so that sawtooth variation is not read as emotional variation.
- **Where it would live:** Phase F narrator.
- **Evidence:**
  - SHOWN: adaptive timing no better (p = .769) and not sparser (Fig. 3).
  - ARGUED: intervention frequency is endogenous (§5.4); budgets should be matched (§5.2).
- **What it changes:** a reporting rule. Largely moot if X5 is adopted, so it is low priority.
- **Cost / risk:** low.

**A v2-spec candidate: none.**
The mirror of the paper's measurement–control separation in the v2 engine is already covered by three existing items:
- M16.B: logging as a pure observer; same seed reaches the same final state.
- C35: arm-blindness, with a static check that no policy, appraisal or consolidation module imports test or readout modules.
- M11.D.20: two renderings of one state agree in ordering.

The paper's one sharper point is that the instrument that feeds control must not be the one that scores the outcome. It has no v2 analogue, because no readout feeds the engine.

---

### C. Search terms

- "persona drift" multi-turn LLM
- "behavioral drift" LLM persona simulation
- "instruction drift" OR "instruction (in)stability" language model dialogs
- "split-softmax" system prompt attention decay
- "context equilibria" multi-turn LLM interactions
- "persona consistency" multi-turn reinforcement learning
- "temporal persona stability" LLM simulation
- "simulated students" tutoring dialogues fidelity
- "long context instruction following" reinstruction
- "self-reminder" LLM identity anchoring
- "stateful role-playing" persona memory retrieval long narrative
- "agreement sycophancy" interaction context
- "LLM-as-judge" validity human raters behavioral rating scale
- "stability is not validity" LLM persona
- "executive control" transformer attention contextual interference
