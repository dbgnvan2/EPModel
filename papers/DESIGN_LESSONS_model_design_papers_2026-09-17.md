# What the model-design papers say about how to build EPModel

Compiled 2026-09-17; SEAA addendum 2026-09-19 (§7); revised 2026-09-20 after full reads of the fourteen sweep papers (§8 and `SPEC_CANDIDATES_from_preprints_2026-09-20.md`). Method literature only. This file is separate from the Bowen corpus in `docs/theory/` and makes no claim about what the corpus says.

## 0. How this was produced, and how far to trust it

**Reading set.** 30 papers: the 29 PDFs in `papers/Model Design papers/` plus the Light Society paper (arXiv 2506.12078v2). Eight references from the original list were not obtained and are not covered: Schelling 1969, Axelrod 1997, Epstein 1999, Palmer et al. 1999, Xia et al. 2011, Das et al. 2014, Bikhchandani et al. 1992, Granovetter 1978.

**Who read what.** Nine sub-agents each read a group of papers in full from `pdftotext` output and reported findings with section, table or figure references, tagging each statement as stated by the paper or as their own inference. I read Light Society myself. I did not read the other 29 line by line. I spot-checked 22 of the quotations used below against the text files with `grep`; all 22 were found. That is a check that the words exist, not that every paraphrase is right.

**Known defects in the inputs.**
- Figures did not survive text extraction. Results that exist only in a plot are reported from captions and prose.
- `JEL-v2.0.pdf` is the INET Oxford Working Paper 2022-10 (21 June 2022), not the typeset 2025 JEL article. It has unresolved citation placeholders and a missing Table 2.
- Röchert et al.: the initialisation and update pseudocode tables were images and were lost. Scheduling statements for that paper rest on prose only.
- The AgentTorch text has no appendices A–D, which the paper refers to.
- Readers flagged internal inconsistencies in OASIS (follow probability 0.2 vs 0.1), YuLan-OneSim (figure captions contradict the body), ElectionSim (text vs Table 10), Axtell 2016 (Table 1 "uniform" vs 4% activation), Mou et al. 2024 (">72% on all three datasets" vs Roe 0.6665), Ghaffarian (step numbering), Röchert (a CI that excludes its own mean), WarAgent (Table 2 vs Table 3).

**What I read of EPModel.** `README.md`, `CLAUDE.md`, `docs/theory/_STATUS.md`, `docs/model_explainer.md` §1–2, §5.1–5.2, §8–12, spec v2 `M3` in full, and keyword searches of the rest of spec v2. Statements below of the form "the spec has no X" rest on a keyword search, not a full read of 427 requirements. Check them before acting.

**Addendum, 2026-09-19.** A 31st paper, SEAA (arXiv 2609.17331v1), was read after this file was first
compiled and is covered in **§7**, which has its own provenance note. It was read by me directly rather
than by a sub-agent, and it is a non-peer-reviewed preprint. §1–§6 below were not rewritten for it; where
§7 sharpens something in §2, it says which subsection.

**Tags used below.** `Paper:` = the paper states or shows it. `Inference:` = my or a reader's judgement about EPModel. Spec IDs are EPModel's own.

---

## 1. Summary

1. None of the 30 papers models a family, a triangle, conserved emotional quantities, or development over decades. Castellano et al. do not cover structural (Heider) balance or triad dynamics. The transferable material is about simulation method: update schemes, small-N behaviour, test design, parameter handling, logging, and engine interfaces.
2. The single most consequential finding for the implementation plan is that **the update scheme changes results qualitatively**, and spec `M3.D.1` is a synchronous scheme. Three independent sources say this (§2.1). The spec has no requirement that tests the 34 acceptance criteria under an alternative scheme.
3. Most other lessons land in **Phase E** (ensemble runner, counterfactual arms, distribution readouts), which the spec deliberately leaves unspecified. They are inputs to that specification, not corrections to Phases B–D.
4. For the exploratory LLM-agent line, the papers support spec `M3.D.6` (no LLM in the decision path). They document non-reproducibility, a pull toward agreeable and consensual output that persona prompts do not remove, large sensitivity to prompt wording and temperature, invalid LLM-as-judge scoring, and recall of training data presented as simulation. No paper addresses maturation or an operational definition of maturity.
5. Several papers practise what EPModel forbids: single runs, fitting then reporting counterfactuals, visual validation. Where they are cited below as precedent, it is for a specific technique, not for their evidential standard.

---

## 2. Lessons for the v2 rule-based engine and the implementation plan

### 2.1 The update scheme is an assumption, and it is not currently tested

Paper:
- Axtell & Farmer §II.A.2: Huberman and Glance showed Nowak and May's spatial-cooperation results "were artifacts of their synchronous updating mechanism". §III.C.2: "Perfect synchrony should be avoided"; on whether results are independent of execution order, "that is generally not the case". Table 3 compares uniform, random and Poisson-clock activation. They also state there is little understanding of how to choose between regimes and almost no empirical data on real activation.
- Castellano et al. §III.E (Sznajd model): under synchronous update an agent receiving opposite instructions is "frustrated" and keeps its state; "Such frustration hinders consensus". §III.B.3 and §X: on heterogeneous graphs, which member of a pair is chosen first (direct vs reverse voter) changes consensus time from logarithmic to exponential in N under slow rewiring.
- Axtell 2016 §4: synchronous hardware produced "computational artifacts that had no meaningful interpretation".
- Counter-examples of order dependence left unexamined: LMAgent Algorithm 2 (agents act in fixed index order), EconAgent §2.3 (random sequential consumption under scarcity), OASIS §2.5 (wall-clock order leaks into simulated timestamps).
- A pattern that avoids the problem: JUNE eq. 5.2 computes infection hazard as a sum over all infectious group members, so the within-step result does not depend on agent order.
- Light Society §4.1–4.2: min-heap over `(time, priority, sequence)`, same-`(time, priority)` events popped as one batch grouped by kind.

Inference for EPModel:
- `M3.D.1` is perceive-all, select-all, act-all. That is the synchronous end of the spectrum. Per-edge latency (`M3.C.1`) adds asynchrony in delivery but not in selection.
- `M1.F.8` requires that order within a tick cannot decide the outcome. The implementation plan needs the resolution rule stated: what a person's appraisal does with two same-tick events that pull in opposite directions. The Sznajd result says that rule is itself a modelling choice with system-level consequences. A commutative aggregation (JUNE's sum) satisfies `M1.F.8` by construction. A heap with a `sequence` tie-breaker does not, unless the batch is reduced order-free after popping.
- Candidate Phase E requirement: run the acceptance criteria under at least one alternative activation regime (seeded random-sequential selection, or Poisson clock) and report which directional results survive. The papers offer no principled way to pick the "right" regime, so the chosen one should be graded `[I]` like any other invented choice.

### 2.2 At N = 12 there are no phases, only metastable states with lifetimes

Paper:
- Castellano et al. §II.A: phase transitions are defined only as N → ∞; "finiteness of N must play therefore a crucial role". §III.B.1: any finite voter system reaches consensus, in d > 2 "only because of a large random fluctuation". §III.B.3: finite-N coexistence states are metastable with size-dependent lifetimes. §III.E: time-to-absorption distributions are broad.
- Bonabeau 2002, diffusion example (Fig. 3): with low initial counts takeoff is slower "because of significant fluctuations".
- Axtell & Farmer §IV.B: with few agents it is hard to assess the character of distributions. Light Society §3 and GenSim §2.2 obtain stability from N and say small populations drown effects in noise.

Inference: this agrees with the explainer's "never a single run". It adds one thing: readouts such as "which tie ruptures first, and when" (§10) should be reported as time-to-event distributions across seeds, not as means. Mean-field results, scaling exponents and critical points from the statistical-physics literature do not carry to 12 agents and should not be cited for EPModel.

### 2.3 Feedback plus relaxation gives bistability; direction tests can flip under a constant sweep

Paper:
- Castellano et al. §VII.A (Bonabeau hierarchy model): winner/loser feedback with forgetting gives an egalitarian-to-hierarchical transition, discontinuous with coexisting metastable states for some parameters, and the stationary state "is very sensitive to the choice of the initial conditions".
- Axtell & Farmer §II.B.4: in a two-agent rule-based model the leverage cycle "suddenly appears at full amplitude" as a parameter varies, with no external noise.
- Castellano et al. §IV.B: Axelrod's frozen multicultural states are destroyed by any small noise, and the effect depends on N. §III.G: partial resistance to a disagreeing majority turns a continuous transition discontinuous.

Inference:
- Over/underfunctioning reciprocity (`functioning_balance`, "directed, bistable") has the same structure as the hierarchy model: positive feedback on a directed dyadic quantity opposed by decay. Expect wide ensemble envelopes and dependence on initial balance. This matches the project's own finding that initial conditions are not nuisance parameters (`M15.D.4`).
- The spec requires sensitivity analysis only for the `basic_level` estimator's window and breadth (`M1.A.4c`, `M10.B.3`). Candidate Phase E requirement: for each `M11.C` criterion, sample the `[I]` constants over declared ranges and report the fraction of samples in which the asserted direction holds. A criterion that holds only in a narrow band of invented constants is a weaker result than one that holds across the range, and the current design cannot tell them apart.
- Any outcome that looks frozen should be re-run with a small perturbation before it is reported as a property of the mechanism.

### 2.4 The ratio of tie dynamics to state dynamics is a control parameter

Paper: Castellano et al. §X. With probability φ an agent rewires a discordant link, otherwise it changes state. Large φ fragments the network into internally uniform components; there is a transition at a critical φ. Deletion-only variants split the population into two opposed groups. Results where a node can "find other agents" depend on rewiring to strangers.

Inference: distance, conductance change and `CUTOFF` are tie attenuation and deletion. Only the attenuation and deletion results transfer, because EPModel has no exit from the field (`I7`) and no rewiring to strangers. The rates at which ties harden or attenuate, relative to the rate at which anxiety moves, belong in the constant sweep of §2.3 as a named ratio.

### 2.5 Initial states must be compatible with the dynamics

Paper: Axtell & Farmer §IV.H: if invented initial agent states are not compatible "with each other and with the inherent dynamics of the model", the model "will generate transient behaviors"; they call this "an open problem". Axtell 2016 Table 1 and §4: an "initial transient passes" before the stationary macrostate. Röchert et al. Fig. 6 report initial and final distributions side by side, which exposed drift produced by the update rule alone (ambivalent share 9.5% → 17.5% or 3.2% with no opinion leaders present).

Inference: my keyword search found no burn-in or settled-window concept in the spec. I do not know how the twelve-person reference family's initial tie and triangle state is constructed. If it is set by hand, early ticks may be relaxation toward a state the dynamics can sustain, and an intervention applied during that window confounds the two. Options are a declared settling window excluded from readouts, or generating initial state by running the engine forward. This is a question for the owner, not a finding.

### 2.6 Test design for the acceptance criteria

Each item is a technique taken from a paper, with the reason.

| Technique | Source | Why it matters here |
|---|---|---|
| **No-interaction control arm** | Chuang et al. §3.5: controls showed opinions drift toward truth "when agents do not communicate" | Run with all conductance zeroed. Whatever drift remains is produced by the policy and clocks alone and must be subtracted before attributing anything to the relationship system |
| **Both signs of every perturbation** | OASIS §3.3.2: up-treated arm matched humans, down-treated arm did not | A directional match on one side can hide an asymmetric failure on the other |
| **Sever-the-input mutants**, not only delete-the-mechanism | EconAgent Fig. 4: without perception, results were "too stable" and looked plausible | A run can pass because agents are insensitive. Mutants that cut an appraisal input test that selection responds to what the spec says it responds to |
| **Rare-move coverage** | Mou et al. 2024 Table 2 and 5: stance accuracy 0.90–0.97 with macro-F1 0.34–0.37 under extreme class imbalance | A directional test should fail if `CUTOFF` or `I-POSITION` never occurs in the ensemble |
| **Pre-declared minimum effect size** | Röchert et al. Tables 4–6: 1000 replicates give CIs so narrow that every difference is "significant" | Test 1's "significant majority of seeds" needs a stated margin, or ensemble size decides the result |
| **Per-seed paired differences** | OASIS §3.2: averaged error on one instance "could be balanced out" by another | Report the distribution of arm differences, not the difference of arm means. *Revised 2026-09-20:* the pairing is only valid if the same seed produces the same chance events in both arms, which a single stateful generator (`M3.D.4`) does not guarantee once an arm changes control flow; see §8.1 and candidate C1 |
| **Separate the variance sources** | Light Society §2.4.1 isolates seed noise from LLM noise. JUNE's 14 realizations vary parameters, not seeds, and never separate the two. Mou et al. App. C.2 hold core-agent output fixed and re-run only the rest, which understates variance | Report seed variance, exogenous-spell-timing variance and constant-sweep variance separately. Re-draw every stochastic source in every seed |
| **Regress selected moves on the inputs shown** | EconAgent Eq. 14, Table 1: savings significant for consumption in 100/100 agents, interest rate for work in 31/100 | A post-run check that each move's selection depends on the state variables `M4` says it depends on |
| **Mechanism switches with a baseline arm** | Röchert et al.: each mechanism "can be switched on and off"; all effects reported against a baseline | Same logic as mutation testing, available at run time rather than only in the test suite |
| **Report misfits beside fits** | Axtell 2016 Fig. 23: model "slightly over-predicts" young-firm survival, "too few separations" | Applies to the twelve corpus bounds in `M10.C.4` |

### 2.7 Functional-form artefacts to check for

Paper:
- Ghaffarian et al. Eq. 1: utility is a product of terms, so any zero term makes every such agent identical at U = 0.
- Röchert et al., "Updating function": the ambivalent state exists only because values are rounded; exact equality is "otherwise not feasible".
- Barabási & Albert, Model B: in a fixed-N system a monotone accumulation rule with no decay runs to saturation.
- Ghaffarian et al. §3.3: a sensitivity sweep gave 99%, 98%, 104%, 97%, non-monotone and within a few points, reported without a test. A reader judged it plausibly noise read as an effect.

Inference: the standing load is bond energy ÷ `functional_level` (`M3.D.1` step 1), and incoming anxiety is divided by functional level. Division by a quantity that can approach zero is the same class of artefact as the absorbing zero. I have not read how the spec bounds it. Any discretisation or clamping that creates ties or equalities should be documented as a modelling decision.

### 2.8 Engine interface, interventions and logging

Paper:
- UGI §4.3–4.4: all control "unified into modifying the agent's trip list" through one call; separate read-only sense calls; push subscriptions on typed triggers instead of polling.
- AgentSociety §5.6: three intervention channels: configuration before the run, state manipulation during it, and an exogenous message. §3.6: two linked memory streams, an Event Flow of what happened and a Perception Flow whose nodes are each "linked to one or more nodes in the Event Flow".
- JUNE §6, Table 4: policies are date-bounded objects that modify parameters, with a table mapping each real policy to how it was implemented.
- RecAgent Supp. A.4: run five rounds, fork, edit one profile in one branch, run five more, compare. The sandbox is "intervenable and resettable".
- chiSIM §4–5: full event history logged so the simulation is a source of "otherwise unobserved" information (tracing back to patient zero); constant state separated from dynamic state; the ensemble and calibration driver (EMEWS) lives outside the model.
- Agent Hospital Fig. 9 and App. C.3: reporting accuracy per 1,000-case segment as well as cumulatively exposed a regression between 12,000 and 14,000 cases that the cumulative curve hid.
- TIS §5.2: an architectural filter silently removed a whole class of behaviour (greetings) from the output distribution.
- WarAgent §4.1.3–4.1.4, Listing 4: a pairwise relation matrix plus a per-agent record, each agent holding its own copy, with a rule-based function that renders state to text. §4.1.2: a checker validates each action's legality and preconditions before it reaches the world.
- AgentScope §2.1: agents expose `reply` and `observe`; `observe` "processes incoming messages without generating a direct reply". Every message has a unique id and timestamp.

Inference:
- My keyword search found no snapshot, fork or checkpoint requirement in the spec. Two-arm counterfactuals that share history up to an intervention tick need it, or each arm must replay from tick 0. Byte-identical determinism (`M3.D.5`) makes replay valid, so this is a cost decision for Phase E, not a correctness gap. *Revised 2026-09-20:* with event-keyed draws (candidate C1) a fork carries no generator state, so a snapshot is a plain state copy.
- The three intervention channels map onto EPModel's arms: initial-condition arm, mid-run perturbation, exogenous spell. Naming them in the Phase E spec, with JUNE's mapping table, would make each counterfactual's mechanism of entry explicit.
- Appraisal and belief-write log records that cite the event IDs they appraised would let the trace renderer show the path from an event to a belief. `M16` already separates belief writes; I did not check whether it links them to event IDs. IDs must derive from the seed, not from a UUID, to keep `M3.D.5`.
- `observe` versus `reply` is the same split as witness versus target (§9.4 of the explainer).
- The explainer already treats `WITHHOLD` as an outcome that changes tie state. The TIS result is a reason to make sure it is also a logged outcome with its own count in readouts.
- Slow variables should be reported as windowed trajectories as well as end states.
- Several papers' observers are not pure: YuLan-OneSim collects data by sending events to agents (§3.3.2); AgentSociety and RecAgent "interview" agents mid-run and do not say whether that writes to memory. `M16.B`'s pure-observer rule is stricter than anything in this literature.

### 2.9 Documentation

Paper: YuLan-OneSim §3.2.1 formalises scenarios with the ODD protocol (Grimm et al.). Ghaffarian et al. use ODD+D and give a table of variable, description, initialisation and data source (Table A1). Röchert et al. Table 3 lists symbol, meaning and explored parameter space, and never varied their threshold constants. Axtell & Farmer mention ODD only in a glossary.

Inference: a keyword search found no mention of ODD in the spec. ODD is the reporting format ABM reviewers expect, and the spec already contains its ingredients. A parameter register with a column stating whether each `[I]` constant has ever been varied, and over what range, would record exactly what Röchert et al. omitted. *Added 2026-09-20:* He's VISA protocol (2607.28027) is a stricter alternative to ODD with 19 machine-checkable consistency rules; candidate C10 proposes a state-and-mechanism register on that model, and candidates C31 and C36 add an invariance-interval column and a per-result audit record.

### 2.10 Candidate functional shapes

These are shapes only. Every constant would be `[I]`.

- **Latitude of acceptance and rejection** (social judgement model, Mou et al. 2024 App. A.4): assimilation when the other's position is close, repulsion when far. A rule-based form for reactivity thresholds.
- **Survival-model durations** (TIS §3.4, Cox proportional hazards with covariates): consistent with the explainer's rule that exogenous input arrives as spells with a duration.
- **Forgetting with a floor** (RecAgent; LMAgent Eq. 3): recall probability falls with recency but has a floor set by importance, so old important items stay recallable. A candidate for persistence of nodal events in the belief layer.
- **Two opposed components stored separately** (Röchert et al.: "Ambivalent attitudes are different from neutral attitudes"): relevant to `M4.D.1d`, where simultaneous approach and withdraw urges are anxiogenic, and to the readout trap that overt emotionality peaks mid-scale.
- **Internal state separate from visible action** (Castellano et al. §III.G, Martins' CODA model: agents observe only others' actions): a formal precedent for counterfeit moves (§5.6) and the belief layer.
- **A fixed budget divided across contacts** (Castellano et al. §III.G, Indekeu): effective coupling falls as the number of ties rises. Loosely analogous to `life_energy`.

### 2.11 Parameter handling without data

Paper: JUNE §6.3 and App. E use Bayes linear emulation with history matching: maximin Latin hypercube, 3 waves of 125 runs, an implausibility measure that includes an explicit "structural model discrepancy" term, and the goal of finding "the set of all input parameter values that give acceptable matches" rather than a best fit. They report that most of 18 fitted parameters were not individually identifiable and that two traded off strongly. Axtell & Farmer §III.H.1 describe qualitative models whose parameters are "simply invented and tested through experimentation" until patterns appear.

Inference: the Latin-hypercube design and the practice of reporting trade-offs transfer directly to the constant sweep in §2.3. The emulator does not; it pays off only when a run takes hours. Whether history matching itself transfers is a decision I cannot make for the project: ruling out regions of constant space because they violate the twelve corpus bounds in `M10.C.4` is a form of calibration, and `M10.C.4` admits those bounds "as checks, never as parameters". It is not the same as fitting a known family history (`M11.F.9`), but it moves in that direction. Flagged for the owner.

### 2.12 Statements that support existing decisions without adding to them

- Axtell & Farmer §I.A: "Each realization of an ABM is a sufficiency theorem"; they recognise a class of models "illustrating a particular mechanism" with no empirical ambition. This is the consistency-engine framing.
- Bonabeau 2002, "Issues with ABM": outcomes of models with soft human factors "should be interpreted purely at the qualitative level". His five conditions for when ABM is appropriate include heterogeneous interaction topology, which is the argument for leaving the lattice.
- Castellano et al. §I and §XI: "a striking unbalance between empirical evidence and theoretical modelization"; an election fit held only at "a certain, carefully chosen, time".
- AgentTorch Table 1: with a fixed query budget, 300 LLM agents scaled up gave higher error than a plain heuristic over 8.4M agents (unemployment MSE 56.98 vs 41.05; infection MSE 4311.70 vs 2914.73). Agency without the system's structure was worse than structure without agency.
- AgentSociety §4.1: numerical bookkeeping is kept in rule-based code because LLMs "cannot guarantee absolute accuracy". Axtell 2016: a stationary macrostate coexists with agent-level "perpetual disequilibrium", so invariants belong at family level and dyads should not be expected to settle.

---

## 3. Lessons for the exploratory LLM-agent line

Spec `M3.D.6` already says an LLM must not appear in the decision path. The papers support that and bear on the September question of whether an LLM agent can hold "a 15-year-old brat" or mature over time.

### 3.1 Reproducibility, cost and horizon

Paper:
- Light Society §2.4.1: with 50% live LLM calls and a fixed seed, run-to-run CV was "about two orders of magnitude larger" than with none.
- Agent Hospital App. C.2: a trend change near 30,000 cases "may caused by the API update from OpenAI".
- WarAgent §6.1.4: anonymized runs "are non-stable when using different random seeds".
- RecAgent: "After 15 rounds of execution, the believability scores for all the questions are lower than 4"; cost per round rises as memory accumulates.
- EconAgent §3.2, §4.1: a 20-year monthly run works with one month of verbatim memory ("L = 1 works") plus a quarterly reflection. Continuity is carried by environment state, not agent memory. App. B: about 30 dollars and 2 hours per run for 100 agents.
- Axtell & Farmer §III.B.4: "Rich cognitive models face challenges of intelligibility"; it becomes unclear whether an outcome is due to the cognitive model, the interactions, or both.

Inference: a hosted model cannot satisfy `M3.D.5`. A 40-year run is about 2,080 ticks; the longest LLM-per-step horizon in this set is 240–300 steps with near-zero agent memory. If an LLM layer is ever used, maturity and relationship state have to be engine variables rendered into each prompt, not things the model is expected to remember.

### 3.2 The pull toward agreeable, consensual, accurate output

Paper:
- Chuang et al. Table 1: agents converge toward the true framing regardless of persona. Under false framing with no induced bias, mean bias B = −1.33 ± 0.17. When all ten agents started by endorsing a false claim they ended at B = −1.3 (Fig. 5). Induced confirmation bias raises diversity monotonically with its prompted strength (0.60 → 0.87 → 1.24).
- Mou et al. 2024 §5.2 and Limitations: agents "struggle to generate non-supportive content"; RLHF makes them "more polite, articulate and respectful" than real users.
- ElectionSim Fig. 9b: simulated answer distributions are more concentrated than real ones (higher HHI).
- OASIS Finding 3: agents pile on to dislikes more than humans do.
- Light Society §2.4.6: cosine diversity fell 36% in seven simulated days.
- Mou et al. survey §4.2.1: LLMs "reach consensus even amidst inconsistencies".

Inference: a prompted reactive adolescent will probably regress toward composure over repeated turns, independent of any mechanism. That should be measured first in a no-interaction control. A reader also noted that in Chuang et al. App. N an agent cites the scientific consensus and then keeps the opposite belief "because" of its confirmation bias, which reads as the trait being recited rather than enacted. That is the LLM version of the counterfeit-position trap.

### 3.3 Sensitivity to wording, temperature, model and language

Paper:
- ElectionSim Table 11: prompt format alone moves voting-subset Macro-F1 between 64.38 and 80.85.
- Mou et al. 2024 Table 8: temperature 0 → 1 moves a macro correlation from 0.7043 to 0.4363; temperature "can make a huge difference".
- WarAgent §6.3: a few sentences of system prompt decide whether war occurs in round 1 or never. Tripling or cutting a numeric capability by two-thirds gave "no obvious" change, while removing one historical grievance removed the war.
- BASES Table 5: two profiles differing in one personality word gave distinctly different behaviour.
- Light Society §2.4.2: changing communication language shifts stance-change rate by up to 4.7 points with topic-dependent sign.
- SocioVerse Table 3: model ranking reverses by domain.
- EconAgent §3.1: prompts verbalise state with loaded words ("you became unemployed", "shortage of goods"), which "enables the agent to recognize the risks".

Inference: prompt text, temperature and model are experimental factors on the same footing as seeds, and two arms may never differ in them. Loaded wording puts theory into a place where it cannot be mutation-tested.

### 3.4 LLM-as-judge is not an instrument

Paper: LMAgent Table II: human raters scored LMAgent 4.30, real humans 4.60, random 3.08. GPT-4 scored LMAgent 3.96, real humans 3.43, random 3.48, ranking random behaviour above real humans. GenSim Limitations: LLM-judge scores were never checked against humans because of cost. SocioVerse §4.3: one "ground truth" was itself produced by prompting LLMs. Chuang et al. App. I and N: an external classifier was "more reliable than self-reported ratings" and was validated separately (84% agreement with the human majority on 100 items).

Inference: an LLM cannot be the rater of maturity or differentiation. Any readout needs an external instrument validated on its own, and the explainer §10.1 already argues that even a good estimate of differentiation is biased by construction.

### 3.5 Recall presented as simulation

Paper: WarAgent §3.4 and §6.1.4 anonymize countries because de-anonymized runs converge for all models (alliance score 97.43 for each), stable across seeds, which the authors read as recall. ElectionSim §4.2: removing time tags risks "pseudo-predictions". EconAgent §4.5: one COVID sentence in the prompt reproduces the 2020 unemployment surge. AgentTorch §8 calls for LLM knowledge to be "time-bound".

Inference: Bowen theory and its clinical vignettes are in LLM training data. An LLM family that produces triangling, cutoff and projection may be reciting the theory. For a project whose purpose is to test what follows if the theory is right, that is circular, and it is the same failure as tuning to a known history.

### 3.6 If an LLM layer is built anyway

Patterns the papers support:
- **Engine holds state; the LLM supplies only a transition.** S3 §4.3 keeps emotion as a discrete state updated as "a Markov process" and asks the LLM for the next state. AgentSociety keeps accounts in rule code.
- **One decision per call over a closed action set.** BASES §2.2.2: a monolithic prompt produced "arbitrary click behaviors"; separate prompts per decision fixed it. The Mou et al. survey §3.1.4 notes a closed action domain makes "responses predictable". The nine moves are such a set.
- **Estimate a distribution, then let the engine sample.** AgentTorch Eq. 7 queries the LLM M times per archetype to estimate an action probability and draws from it; larger M "significantly improves" results (Fig. 6). Seeding stays in the engine.
- **Rule-based legality checking** between LLM output and the world (WarAgent's secretary, §4.1.2).
- **Cap retrieved memory.** Agent Hospital App. C.6: 3 cases and 4 experiences were best; retrieving more "degrades performance".
- **Periodic reflection stabilises early transients but homogenises.** EconAgent Fig. 4: without reflection, early inflation anomalies near 15%; the example reflection ends in generic prudent advice.
- **Change stored as explicit, inspectable state.** Agent Hospital App. B.3: proposed rules are validated and "Otherwise, the rule will be discarded", with the base LLM frozen. This is a design for an agent that changes over time without weight updates. It depends on a ground-truth label for every decision. EPModel has no oracle for a correct family move, so the September question of what "more mature" means operationally is not answered by it.

The one favourable statement: AgentTorch §7 says that "for small populations with high-personalized interventions", LLM-as-agent "can be viable". The paper gives no evidence for that regime.

The Mou et al. survey has no method for maturation over decades or for defining maturity; it evaluates personas with BFI/MBTI questionnaires and LLM judges. The open problems listed in the September conversation remain open in this literature.

---

## 4. What this reading set does not contain

- Nothing on families as relational systems. Households appear as co-location containers (chiSIM, JUNE), as atomic agents (Ghaffarian), or as a tie label (AgentSociety). JUNE treats multi-generational households as a residual category "filled last".
- Nothing on triads or structural balance.
- No model in which an emotional quantity is conserved and redirected. S3's emotion decays to zero by design.
- No validation method for a theory-driven model without data beyond one-at-a-time sensitivity sweeps, which Röchert et al. label "validation".
- Nothing on replication, docking or model-to-model comparison beyond a citation in Axtell & Farmer.

Literature that may fill these gaps, named from my own knowledge and **not checked in this session**: dynamics of structural balance on triads (Antal, Krapivsky and Redner, mid-2000s); the ODD protocol papers (Grimm et al. 2006, 2010, 2020); pattern-oriented modelling (Grimm et al. 2005); history matching for stochastic simulators (Vernon, Goldstein and colleagues); model alignment or "docking" (Axtell, Axelrod, Epstein and Cohen 1996). Verify existence and relevance before adding to `INDEX.md`.

---

## 5. Relevance of each paper to EPModel

High = changes or sharpens a design decision. Medium = a usable technique or a specific caution. Low = little or nothing transfers.

| Paper | Relevance | Main use |
|---|---|---|
| Axtell & Farmer 2022 (INET WP; JEL 2025) | High | Activation regimes, sufficiency-theorem framing, initial transients, intelligibility |
| Castellano, Fortunato & Loreto 2009 | High | Synchronous frustration, finite-N metastability, noise fragility, coevolving ties, bistability |
| Chuang et al. 2024 | High (LLM line) | Convergence to truth regardless of persona; no-interaction control; external instrument |
| JUNE, Aylett-Bullock et al. 2021 | Medium–High | Order-free within-step aggregation, non-implausible parameter sets, non-identifiability, policy objects |
| Light Society, Guan et al. 2026 | Medium–High | Event-queue formalism close to EPModel's; separating noise sources |
| EconAgent, Li et al. 2024 | Medium | Two clocks over 20 years; state in environment; sever-the-input ablation; decision regression |
| WarAgent, Hua et al. 2023 | Medium | Small-N typed actions, state matrix plus renderer, legality checker, recall tests; how not to run 3-run arms |
| AgentTorch, Chopra et al. 2024 | Medium | Agency vs structure (Table 1); distribution-estimate pattern |
| RecAgent, Wang et al. 2025 | Medium | Fork-and-compare; believability decay; forgetting with a floor |
| Röchert et al. 2022 | Medium | Mechanism switches, parameter table, two-component ambivalence, rounding artefact, over-powered CIs |
| AgentSociety, Piao et al. 2025 | Medium | Intervention taxonomy; linked event and perception streams; rule-based bookkeeping |
| Mou, Wei & Huang 2024 | Medium | Select/message/update decomposition; SJ model; temperature sensitivity; imbalance |
| OASIS, Yang et al. 2024 | Medium | Three-arm design; asymmetric failure; averaging hides error |
| Agent Hospital, Li et al. 2024 | Medium (LLM line) | Gated auditable learning store; windowed reporting; API drift |
| Bonabeau 2002 | Medium | When ABM is appropriate; qualitative interpretation |
| chiSIM, Macal et al. 2018 | Medium | Hybrid time-stepped and discrete-event; full event log; driver outside engine |
| Axtell 2016 | Medium | Activation defines the time unit; synchrony artefacts; report misfits |
| UGI, Xu et al. 2023 | Low–Medium | Narrow control surface and read-only queries. No results reported |
| LMAgent, Liu et al. 2024 | Low–Medium | Evidence against LLM-as-judge. Single seed, fixed agent order |
| ElectionSim, Zhang et al. 2024 | Low–Medium (LLM line) | Homogenisation measured by HHI; prompt-format sensitivity; leakage |
| S3, Gao et al. 2023 | Low–Medium | Ordinal emotion state outside the LLM; source-weighted messages |
| BASES, Ren et al. 2024 | Low–Medium (LLM line) | One decision per call; one-word persona sensitivity |
| YuLan-OneSim, Wang et al. 2025 | Low–Medium | ODD use; round vs tick mode; action-graph reachability check |
| TIS, Zhang et al. 2024 | Low | Hazard-model durations; silent filtering of a behaviour class |
| Mou et al. 2024 survey | Low | Vocabulary (micro/macro/system evaluation; closed vs open action domain). No critical analysis |
| SocioVerse, Zhang et al. 2025 | Low | Initial conditions dominate (Table 4); circular ground truth |
| Ghaffarian et al. 2021 | Low | ODD+D and variable-source table; absorbing-zero artefact; stating what could not be validated |
| AgentScope, Gao et al. 2024 | Low | `reply`/`observe` split. No experiments |
| GenSim, Tang et al. 2025 | Low | Repeat-10 fluctuation metric. Four-page demo |
| SEAA, Liu 2026 (arXiv 2609.17331v1) | Medium–High | Dose–response ablation shape; shock-and-recover protocol; repertoire-entropy and self–other-gap readouts; a mechanism-free null model for emergent differentiation; downstream-narration contract. Preprint, not peer reviewed — see §7 |
| Barabási & Albert 1999 | None | Only the ablation logic (Models A and B). Do not use preferential attachment for kinship ties |

---

## 6. Questions this raises for the project owner

1. What does a person's appraisal do with two same-tick events that pull in opposite directions, and is that rule stated in the spec? (§2.1)
2. Should Phase E require the acceptance criteria to be re-run under an alternative activation regime? (§2.1)
3. Should Phase E require a sweep over `[I]` constants, reporting for each criterion the fraction of the range in which its direction holds? (§2.3)
4. How is the reference family's initial tie and triangle state constructed, and is there a settling window before interventions and readouts? (§2.5) *2026-09-20:* still open; §7.10(e) gives the testable form, and candidates C29, C32 and C42 bear on it.
5. Is excluding regions of constant space that violate the `M10.C.4` bounds permitted, or does it conflict with "checks, never parameters"? (§2.11)
6. Is snapshot-and-fork wanted in Phase E, or will arms replay from tick 0? (§2.8) *2026-09-20:* either way, candidate C1 is needed first, or the two arms are not comparing the same chance events.
7. Is a no-interaction control arm (all conductance zero) already among the 34 criteria? I did not find one by keyword. (§2.6) *2026-09-20:* candidate C34 proposes three named control arms, including a two-level no-interaction control.

Questions 8–17 are in **§7.11**; 8–15 come from the SEAA addendum and 16–17 from reconciling it with §8. Questions 13 and 14 are answered in **§8.1** and **§8.2**.

---

## 7. Addendum, 2026-09-19 — SEAA (arXiv 2609.17331v1)

**Provenance and trust.** Liu, X., *Self-Emergence Agent Architecture: Behavior-Inertia HMM, Reflexive
Metacognition, and Social-Contrastive Self-Modeling*, arXiv:2609.17331v1, 16 September 2026. Single
author, independent researcher, preprint, **not peer reviewed**. Unlike the 30 papers in §0–§6 this one I
read myself, from the PDF: pages 1–19 in full (body, figures, tables, Appendix A prompts, the start of
Appendix B). Pages 20–25 are the remaining verbatim transcripts and reproducibility notes and I did not
read them. Nothing below rests on a sub-agent report.

**Tags** as in §0: `Paper:` = the paper states or shows it; `Inference:` = my judgement about EPModel.
Everything in this section is **method literature and is `[I]` by construction**. None of it is corpus
material, none of it belongs in `docs/theory/_LEDGER.md`, and none of it may be cited in
`model_explainer.md` as anything but `[I]`.

### 7.1 What the paper does

Paper: three components in one closed loop — social observation → action → reflection → **parameter
edit** → differentiated action. (i) Each agent's disposition is an *editable* HMM transition matrix
`P_t` over `K = 4` latent states {calm, alert, impulsive, pessimistic}. (ii) A Reflexion-style verbal
reflection is parsed into a direction matrix and applied as `P_{t+1} = Normalize(P_t + η·Δ(R_t))`
(Eq. 2) rather than stored as text. (iii) `N = 5` initially identical agents observe each other's
behaviour. The claimed contribution is that the object being updated is a *personality carrier*, not a
competence carrier, and that no prior public work closes the loop reflection → parameters →
differentiation (Table 1).

The headline numerical result (§5.7.2, Table 3, `T = 600`, 30 seeds) is that initially identical agents
diverge: pairwise transition-matrix Frobenius distance 0.00 → 1.91, self-model distance 0.13 → 0.51,
"determinism" `1 − H(P)/log K` 0.25 → 0.84, self–other gap 0.08 → 0.35, each against a control that
computes the same reflection and never applies it.

**The mechanism prototype is language-model-free**, and this is what makes the paper readable as method
rather than as an LLM demo. Stripped to §5.7.1 it is: a per-agent preference vector `V`, a reward
`r_t = cos(b_z, n_t) − 0.80 + ε` measuring the fit between the enacted behaviour and that agent's own
random-walking "experience need", the update `V_{t+1}[z] = clip(V_t[z] + η·tanh(2r_t))`, and
`P_t = softmax(log P_base + βV_t)`. That is a positive feedback loop on a preference vector with a
per-agent idiosyncratic driver. The LLM appears only downstream, as a renderer.

### 7.2 Dose–response as the reporting standard for an `[I]` constant

Paper (§5.7.5, Fig. 5): they sweep the one constant the mechanism hangs on,
η ∈ {0, 0.01, 0.025, 0.05, 0.1, 0.2}, 10 seeds per point, everything else fixed, and report three
properties rather than one. The effect is **absent at zero** (η = 0 reproduces the flat control exactly,
so the mechanism is necessary); it is **monotone** in η (divergence 0 → 1.07 → 1.91 → 1.99 → 2.04;
determinism 0.25 → 0.37 → 0.76 → 0.83 → 0.84); and the operating point used throughout the paper sits
on the **saturated plateau, not on the rising edge**.

Inference for EPModel: this is a sharper form of the sweep already proposed in §2.3, and it is
complementary to `M10.C.4a` rather than a duplicate of it. `M10.C.4a` is a presence/absence factorial —
it establishes that each of the three quantum-jump conditions is *necessary*, which is the harder and
more important claim. What it cannot show is **where the chosen value of an `[I]` constant sits on the
response curve**. A direction that holds at the declared default and collapses 20% either side is a
different result from one that holds across a plateau, and the current design reports both the same
way. The three-part shape — *absent at zero, monotone through the middle, saturated at the operating
point* — is a reporting standard worth adopting verbatim, because each part fails differently: no effect
at zero and the constant is inert; non-monotone and the functional form is wrong (cf. §2.7); on the
rising edge and the result is a knife-edge artefact of an invented number.

Caution: this only works for a criterion with **one dominant** `[I]` constant. Where a direction depends
on a ratio of two rates (§2.4) the curve is over the ratio, and that has to be named before it can be
swept.

### 7.3 The contingency-shock protocol

Paper (§5.7.4, Fig. 4): run the standard dynamics to consolidation (300 steps), then **reverse the
payoff of whatever state each agent settled on** — `r_t ← −|r_t|` whenever the agent occupies its own
`z*` — and measure four things over 30 seeds: occupancy of the shocked state (0.94 → 0.06 for the edit
condition; 0.40 → 0.25 for the control, which has no mechanism to revise), the **switch rate** (150/150
agents abandon the shocked state), the **time-to-switch distribution** (median 152 steps, plotted as a
histogram, Fig. 4d), and a rigidity index that **dips and then recovers** (0.84 → ~0.37 → 0.80).

Inference for EPModel: three things transfer, in descending order of value.

1. **The dip-and-recover shape is the falsifiable signature, not the switch.** Fleeing a punished state
   and never re-settling, and fleeing it and re-stabilising on a new one, are different outcomes that a
   single end-of-run number cannot separate. Anywhere the model claims reorganisation after a nodal
   event, the readout should be the trajectory of a concentration measure through the event, not its
   value after it.
2. **Time-to-event as a distribution.** This is §2.2's lesson with a worked instance: the paper reports
   a histogram over 150 agents rather than a mean, and the histogram is visibly broad and skewed.
   `M11.C.4` — `CUTOFF` drops acute anxiety immediately and raises family total anxiety **at the next
   nodal event** — is a claim about *when*, currently phrased as a direction at one point. The shock
   protocol is the shape that makes the *when* measurable.
3. **The control is what makes the shock interpretable.** Their control cannot revise, so its occupancy
   is unchanged by the shock; the difference between arms is attributable to the edit and to nothing
   else. This is `M0.4`'s discipline arriving at the same place independently.

Inference, and a warning: the shock in this paper is *exogenous and instantaneous*, applied to a scalar
payoff by the experimenter. EPModel's nodal events are internal to the family and propagate
(`M9.4`, `M7.C.1c`). The protocol transfers as a **measurement design**, not as an event model.

### 7.4 Two readouts worth taking, both `[I]`

Paper (§5.7.2): two of the four "operational signatures of self-emergence" are computable from state
EPModel already keeps.

1. **Repertoire entropy** — their "personality determinism", `1 − H(P) / log K`, one minus the
   normalised entropy of the transition row, rising as the agent's behaviour concentrates on one state.
2. **Self–other gap** — `‖m_t − m̄_t‖`, the distance between an agent's EMA self-model and its EMA model
   of the group mean, rising as the agent's behaviour separates from the group's.

Inference for (1): `M4.D.6` requires that moves that worked before are reinforced, "so that a family
develops a characteristic style", and **names no readout for it**. Entropy over the nine-move repertoire
(`M5.A.1`) is that readout. It is nearly free: `M4.E.1` already records every selected move as an event,
and `M1.F.1a` already puts the selecting **channel** on the event record, so it can be computed *per
channel*. That gives a direct test of `M4.D.6d` — reinforcement operates on the automatic channel only —
as a **signature rather than an assertion**: automatic-channel repertoire entropy falls over a run while
self-directed-channel entropy does not. Nothing in the 34 criteria currently checks this, and a
reinforcement implementation that leaked into the self-directed channel would pass all of them
(`M4.D.6c` says as much for the degenerate-policy case). This is the only item in §7 that touches
Phases B–D, and it touches them as a read-only readout over the existing event log.

Inference for (2): attractive as a readout, **dangerous as a target**. A cut-off member's behaviour
separates from the family's too, so the self–other gap scores a binder and the real thing the same way —
exactly the failure `M11.C.18` and `M11.C.20` exist to catch for the `basic_level` estimator. It should
not enter the model without the same discriminating treatment: two arms brought to the same gap by
different routes must be shown to separate on something else. Until then it is a description of
behavioural distance and must be labelled as one.

### 7.5 The null model EPModel should have to beat

This is the most useful thing in the paper and the authors do not present it as a finding.

Paper: with **no family, no triangle, no projection, no role assignment and no persona**, five agents
running a bare positive-feedback loop on a preference vector, differing only in an idiosyncratic random
walk, reliably produce (a) stable differentiated dispositions, ~2.8 of 5 locking onto *distinct*
dominant states (§5.7.2); (b) first-person self-narratives that cite specific contrasts with specific
others (§5.8, Table 4); and (c) in the LLM layer, a reproducible **social topology** — a consensus hub
and a unanimously-named outlier, outlier share 0.77 vs 0.33 for controls over 6 seeds (§5.9.1, Fig. 8).
Differentiation, self-description and social structure all emerge from reinforcement plus observation
alone.

Inference: EPModel will produce sibling differentiation — `M11.C.2`'s concentration of anxiety on one
child is meant to. SEAA is a demonstration that **the bare fact of differentiation is not evidence for
the projection mechanism**, because a mechanism-free reinforcement loop delivers it, with large effect
sizes, from identical starts. `M4.D.6` *is* such a loop, and it is inside EPModel. So an acceptance
criterion of the form "the arms differ in how the children turn out" is satisfiable by the reinforcement
loop alone, and this is the same failure class as `M4.D.6c`: a mechanism that shifts both arms together
is invisible to criteria that hold the policy fixed across arms.

The fix is a **rival-mechanism control arm** — the same engine with projection concentration disabled
and `M4.D.6` reinforcement left running — and criteria that assert the **shape** of the outcome (*which*
child, tracking the mother's focus and the `M10.C.4a` conditions) rather than its presence. This is a
different thing from `M10.C.4a`'s ablation: that removes a condition *from the jump mechanism* and asks
whether the jump survives; this removes *the mechanism* and asks whether the observable survives without
it. Both are needed and neither substitutes for the other.

### 7.6 The language-layer contract, if Phase F ever narrates

Paper (§5.8, §5.9, Appendix A): three properties, and all three are worth copying.

- **Narration is strictly downstream of state.** The LLM is handed a JSON objective profile (dominant
  state, 3-D self-model, group mean, signed gaps, switch count, step of last switch) and asked to speak
  in the first person consistently with it. It computes nothing and decides nothing.
- **Each agent's renderer sees only that agent's own profile** — "the LLM receives no other agent's
  private trajectory" (§5.8). Epistemic scope is enforced at the call boundary, not by prompt
  instruction.
- **A deterministic verbalizer is released alongside**, rendering the same profile as text with no model
  call, and the paper shows both layers give the same ordering (Jaccard overlap 0.57 SEAA vs 0.77
  control for the verbalizer; 0.267 vs 0.391 for the hosted model, §5.9). Every number therefore
  reproduces offline.

Inference: this is the concrete form of `M3.D.6` (no LLM in the decision path) extended to the output
path, and it answers §3.1's reproducibility objection without giving up the narration. The deterministic
fallback is the load-bearing part: it is what lets a result be checked by someone without an API key,
and it is a direct answer to §3.3's model- and temperature-sensitivity findings. Prompts are in
Appendix A and are short enough to adapt.

### 7.7 What to reject

1. **The HMM carrier.** `K = 4` discrete latent states with a per-agent transition matrix is precisely
   the flattening the pivot exists to undo (`CLAUDE.md`, on Axiom 1 and on per-agent, per-tie and
   per-triangle state). Take the loop shape; do not take the representation.
2. **"Consolidation" is not maturation, and the paper's headline signature is the opposite of
   differentiation.** In SEAA, becoming a self means becoming *more* deterministic, more locked and less
   responsive: determinism rises 0.25 → 0.84 and this is reported as success. In Bowen terms that is
   increased automatic functioning. If §7.4's entropy readout is adopted it must be labelled **rigidity**
   and must never be read as maturity, and the same applies to the self–other gap. The one place the
   paper's own framing wobbles on this is Agent 4's "mid-life personality transition" (§5.7.3), which is
   a preference vector changing sign twice.
3. **The effect sizes are tautological and must not be cited.** Cohen's `d` of 9.19 and `p < 10⁻¹³`
   (Table 3) are measured against a control that is the identical system **with the update switched
   off** — η = 0, i.e. no dynamics at all. That is a manipulation check, not a discovery, and a
   near-perfect separation is what a working positive feedback loop is *defined* to produce. `M0.4`'s
   two-arm design, where both arms share one parameter set and one mechanism, is strictly stronger.
   Nothing in Table 3 is a calibration target and nothing in it is precedent.
4. **`N = 5`, one topic, one model family.** The paper says so itself (§6.3): a single hosted model
   family (DeepSeek-Chat), a single deliberation topic, a text-only society with no embodiment, and a
   language-to-parameter mapping `Δ(R_t)` that "requires careful tuning for stability". The social
   topology in §5.9.1 may carry that model's stylistic biases; the authors name cross-model replication
   as the most important next step and have not done it.
5. **Attribution.** Method literature, `[I]`, `papers/` not `docs/theory/`. See §7's preamble.

### 7.8 Candidate Phase E requirements, drafted

Draft text, **not in the spec**. Provisional IDs `E-DR`, `E-SH`, `E-RM` are placeholders pending Phase E
numbering. Each is written in the spec's own form so it can be lifted if approved. `E-RE` in §7.9 is the
one item that is not Phase E.

---

**`E-DR` — dose–response reporting for the dominant `[I]` constant of a direction criterion.**

Every `M11.C` criterion whose asserted direction depends on a single dominant `[I]` constant **MUST** be
re-run over a declared sweep of that constant and **MUST** report the response curve, not a single point.
The constant and its sweep range **MUST** be declared in `M10` alongside the constant itself. The report
**MUST** state three properties separately:

- **(a) Necessity.** The effect **MUST** be absent at the null value of the constant (zero, or the value
  that disables the mechanism). A criterion whose direction still holds with its own mechanism disabled
  is a failing criterion and **MUST** be raised as such, not reported as robust.
- **(b) Monotonicity.** The effect **MUST** be reported as monotone or non-monotone over the range. A
  non-monotone response **MUST** be flagged as a functional-form finding (§2.7), not smoothed.
- **(c) Position of the operating point.** The report **MUST** state whether the declared default sits on
  a plateau or on a rising edge, by the fraction of the effect attained at the default relative to the
  range maximum.

Where a direction depends on a **ratio** of two `[I]` rates rather than on one constant (§2.4), the ratio
**MUST** be named in `M10` and the sweep **MUST** be over the named ratio. A criterion whose dominant
constant cannot be identified **MUST** be reported as not covered by this requirement rather than
silently omitted.

`E-DR` **MUST NOT** be read as replacing `M10.C.4a`. That requirement establishes that each quantum-jump
condition is necessary; `E-DR` establishes where an invented number sits on its own curve. A criterion
may satisfy one and fail the other.

*Tests:* `test_edr_null_value_removes_effect`, `test_edr_reports_monotonicity`,
`test_edr_flags_rising_edge_operating_point`.
*Falsification:* pinning the sweep to a single point, or reporting only the end-of-range value, **MUST**
turn these red.

---

**`E-SH` — the shock-and-recover protocol for any criterion asserting reorganisation after a nodal
event.**

Any `M11.C` criterion that asserts a change *following* a nodal event — `M11.C.4` is the first — **MUST**
be run as a shock protocol and **MUST** report a trajectory, not an end-of-run difference. The protocol
is: run both arms to a declared settling condition; apply the nodal event at a fixed tick in both arms;
continue for a declared post-event window.

The readout **MUST** include, for both arms:

- **(a)** the trajectory of the affected quantity **through** the event, at a resolution fine enough to
  show a transient — an end-of-window value alone is a failing readout;
- **(b)** the **time-to-event distribution** across seeds for whatever the criterion says should happen
  (the tie ruptures, the symptom appears, the anxiety relocates), reported as a distribution with its
  spread, **never** as a mean (§2.2);
- **(c)** the fraction of seeds in which it does not happen at all within the window, reported explicitly
  rather than dropped from the denominator.

Where a concentration or rigidity measure is available (`E-RE`), the report **MUST** distinguish
**destabilisation without re-settling** from **reorganisation** — a fall that does not recover, against a
fall that recovers to a comparable level on a different configuration. A criterion that cannot
distinguish these **MUST** say so.

The shocked quantity **MUST** be an ordinary nodal event propagating through the model's own machinery
(`M7`, `M9.4`). An exogenous edit applied directly to an agent's state **MUST NOT** be used to stand in
for one.

*Tests:* `test_esh_reports_trajectory_through_event`, `test_esh_time_to_event_is_a_distribution`,
`test_esh_reports_non_occurrence_fraction`.
*Falsification:* collapsing (b) to a mean, or excluding non-occurring seeds from the denominator, **MUST**
turn these red.

---

**`E-RM` — the rival-mechanism control arm.**

Every `M11.C` criterion whose observable is **differentiation between family members** — `M11.C.2` is the
first — **MUST** be run against a third arm in which the mechanism the criterion attributes the outcome
to is **disabled** while `M4.D.6` reinforcement continues to run. The criterion **MUST** assert the
**shape** of the outcome, not its presence.

- **(a)** The criterion **MUST** name which member the mechanism predicts will be affected, and the
  assertion **MUST** be about that member, in a way the rival arm can fail. "The members diverge" is a
  failing criterion under this requirement; "the member carrying the projection focus diverges, and the
  siblings do not" is not.
- **(b)** The rival arm **MUST** be reported even when it also produces divergence. A rival arm that
  reproduces the *magnitude* but not the *shape* is the expected result and is informative; suppressing
  it is not permitted.
- **(c)** If the rival arm reproduces the shape as well, the criterion **MUST** be reported as **not
  discriminating**, and the mechanism it tests **MUST NOT** be described in any output as supported by
  it.

`E-RM` is distinct from `M10.C.4a`. That requirement removes a *condition* from the jump mechanism and
asks whether the jump survives; `E-RM` removes the *mechanism* and asks whether the observable survives
without it. Both are required. `E-RM` is the arm-level counterpart of `M4.D.6c`: a mechanism that moves
both arms together is invisible to any criterion that holds it fixed across them.

*Tests:* `test_erm_rival_arm_is_run_and_reported`,
`test_erm_criterion_asserts_which_member_not_that_members_differ`,
`test_erm_non_discriminating_criterion_is_reported_as_such`.
*Falsification:* an assertion that only requires non-zero divergence between members **MUST** turn the
second test red.

---

**Where these sit against the C-numbers (added 2026-09-20).** The candidates file
(`SPEC_CANDIDATES_from_preprints_2026-09-20.md`) was written from fourteen other papers and overlaps
these four drafts in places. The two sets are **not** rival proposals; where they overlap the C-number is
better sourced, because it rests on a paper that did the thing rather than on transfer from SEAA.

| Draft | Overlaps | How to resolve |
|---|---|---|
| `E-DR` | **C16** (monotonicity over ≥4 levels), **C24** (readout saturation), **C31** (per-constant invariance interval) | Keep `E-DR` for the `[I]` constants *behind a criterion*; C16 is the same shape applied to *person parameters*. `E-DR`(b) is C16's assertion. `E-DR`(c) is C24 plus C31 — and for a **threshold** constant C31 is strictly better than sampling a curve, because the switch points are computable from the margins encountered rather than sampled |
| `E-SH` | **C18** (persistence after a spell, with a learning-off arm), **C29** (regime classification) | No C-number covers the trajectory-through-the-event readout or the non-occurrence fraction. Keep `E-SH`; take C29's inconclusive class for part (b) |
| `E-RM` | **C34** (three named control arms) | Distinct: C34's arms remove *interaction*, *exogenous input* or *heterogeneity*; `E-RM` removes **the mechanism the criterion credits** while leaving `M4.D.6` reinforcement running. Add it to C34's list rather than proposing it separately |
| `E-RE` | **C39** (move-transition matrix and JSD between arms) | Complementary. C39 is richer on structure and says nothing about the **channel split**, which is the whole point of `E-RE` and the only thing that tests `M4.D.6d`. Fold `E-RE` in as a per-channel entropy column on C39's readout |

### 7.9 The one Phase B–D item

**`E-RE` — repertoire entropy, per channel.** Not a Phase E requirement; a readout over state the engine
already holds, and the only item in §7 that bears on the core.

The run log (`M16`) **SHOULD** carry, per agent and per window, the normalised entropy of the
distribution of selected moves over the `M5` repertoire, computed **separately for the `AUTOMATIC` and
`SELF_DIRECTED` channels** using the channel already recorded on every event by `M1.F.1a`. It is a
description of concentration and **MUST** be labelled as such — it is **not** a measure of
differentiation, maturity or `basic_level`, and a falling value means behaviour has narrowed, not that
anyone has matured.

Its purpose is to give `M4.D.6` an observable. The predicted signature is that automatic-channel entropy
falls over a run as a family develops a characteristic style, while self-directed-channel entropy does
not, because `M4.D.6d` forbids reinforcement of the self-directed channel. An implementation whose
reinforcement leaked across channels would satisfy every current criterion and would show up here.

This requires **no new state** and **no engine change** — it is computed from the `M4.E.1` event stream
by an observer, consistent with `M16.B` (logging is a pure observer; same seed with and without it
reaches the same final state).

*Test:* `test_ere_leaked_reinforcement_raises_self_directed_channel_entropy` — an implementation that
reinforces both channels **MUST** turn it red. This is a mutation test and **MUST** be proved failing by
mutation before it counts as coverage.

### 7.10 Testing ideas the paper suggests, beyond §7.8

These are test designs rather than requirements. Each is derived from something the paper does; none of
them is a claim the paper makes about EPModel.

**(a) A disabled mechanism MUST be bit-identical to baseline, not merely similar.** Paper: the η = 0 arm
reproduces the control *exactly* — matrix divergence 0.00 ± 0.00 over 30 seeds (Table 3), not a small
number. Inference: that exactness is usable as a harness self-test rather than as a result. Any arm that
disables a mechanism **should** be bit-identical to baseline under the same seed, and the failure it
catches is **RNG leakage**: a disabled mechanism that still consumes random draws shifts every downstream
stochastic decision, so the arms diverge for reasons unrelated to the mechanism and the divergence reads
as an effect. The remedy is a named RNG substream per mechanism, so that switching one off does not move
the others. This bears on `E-RM` and on `M10.C.4a`, both of which work by turning mechanisms off and
comparing, and it is a dirty-state problem in the sense of the global testing rules — the shared stream
is state persisting across the thing under test.
*Test:* `test_disabled_mechanism_leaves_rng_stream_unchanged`.
*Revised 2026-09-20 (§8.1):* **the test stands; the remedy above does not.** Buffalo et al. §1 classify
per-mechanism substreams as a coarse mitigation — within a stream the dependence persists, and choosing
the granularity requires anticipating every execution-path change in advance. The remedy is counter-based
draws keyed by a stable event identity (candidate **C1**), and this test is candidate **C4**'s placebo
test. The sentence "the remedy is a named RNG substream per mechanism" is superseded and should not be
acted on.

**(b) Permutation invariance over a symmetric family.** Paper: the entire design rests on agents starting
identical, so that any later difference is emergent by construction (§4.7, §5.7.1 — "identical for all",
"initialized identically"). Inference: inverted, this is a bug detector. Construct a deliberately
symmetric family, permute the agent identifiers, and assert the outcome permutes with them. If it does
not, **agent index order is leaking into the result** — the concrete form of the order-dependence worry
in §2.1, and the failure visible in LMAgent's fixed index order and OASIS's wall-clock leak. This is an
executable test of `M1.F.8` (order within a tick **MUST NOT** decide the outcome), which is currently
asserted with nothing behind it. A symmetric family is artificial; the test does not need a realistic
one, and its artificiality is the point.
*Test:* `test_m1f8_symmetric_family_outcome_is_permutation_equivariant`.
*Extended 2026-09-20 (§8.2):* Q14 answered **yes — write it now**. Candidate **C5** adds two further
forms, batch-order permutation and a sequential-reduction mutant. One trap the §7.10(b) text does not
name: a commutative batch reduction does **not** satisfy `M1.F.8` if a stateful generator is consumed in
loop order, which C1 removes. Sachdeva & van Nuenen §3.3 put a number on why this matters — speaking
order alone moved first-round consensus from ~40% to ~90%.

**(c) Floor tests where a null arm exists, not only direction tests.** Paper: the controls' lock-in rate
is **0 across all seeds** (§5.7.2), and 0.00 of control groups produce a unanimous outlier against 0.77
for SEAA (§5.9.1). Those are absence claims, which are strictly stronger than "A > B". Inference: a
direction test passes when *both* arms show the phenomenon and one shows more of it; a floor test fails
in that case. Wherever a mechanism-disabled arm exists, the criterion **should** assert that the
observable is absent — as a declared upper bound over N seeds, not a literal zero, since the model is
stochastic. `E-RM`'s rival arm is the first place this applies.
*Test:* `test_null_arm_observable_stays_below_declared_floor`.
*Extended 2026-09-20:* candidate **C34** names three control arms to apply this to — no-interaction at
two levels (zero conductance; ties intact with delivery suppressed), endogenous-only, and a homogeneous
family in which each criterion declares in advance whether it is expected to fail. C34's homogeneous arm
is the sharper form of the same idea: a criterion that still passes where the theory says heterogeneity
is required is flagged.

**(d) Two renderings of the same state MUST agree in ordering.** Paper: the deterministic verbalizer and
the hosted model give different magnitudes but the same ordering (0.57 vs 0.77 for one, 0.267 vs 0.391
for the other, §5.9) — and the paper uses that agreement as evidence the result is not a renderer
artefact. Inference: `M11.G`'s family-evaluation readout and the `M16` run log are two views of one
state. An ordering computed from the readout **should** match the ordering computed from the raw log.
This is cheap and it catches renderer drift, which rots silently because nothing else reads the readout.
*Test:* `test_m11g_readout_ordering_matches_run_log_ordering`.
*Related 2026-09-20:* candidate **C21** (representation-invariance mutants) is the general form — a
result that changes when the encoding changes but the content does not is an artefact of the encoding.

**(e) Assert the settling condition; do not count ticks.** Paper: 300 steps are burned to consolidation
before the shock is applied (§5.7.4), and consolidation is verified — occupancy of the dominant state
reaches 0.94 before `t = 300`, it is not assumed. Inference: §2.5 already warns about initial transients;
the testable form is that the readout quantity's **drift over the trailing window** is below a declared
threshold, so that a run taking its readout while still in transient **fails** rather than quietly
reporting the transient as a result. This is a precondition for `E-SH`: a shock applied before settling
measures the transient, not the reorganisation.
*Test:* `test_esh_readout_refuses_to_report_before_settling_condition_is_met`.

**(f) A bimodality guard on any time-to-event readout.** Paper: the time-to-switch histogram (Fig. 4d,
150 agents) is broad and skewed, and a mean would have described it badly. Inference: §2.2 says these
readouts should be distributions; the way to hold that at the code level rather than in prose is a
synthetic **bimodal** fixture that the reporting path must not collapse to a single number. This is the
`[I]`-fixture discipline of P23 applied to a summary statistic.
*Test:* `test_time_to_event_report_does_not_collapse_a_bimodal_fixture`.
*Related 2026-09-20:* candidate **C29** is the reporting-side counterpart — per-seed regime
classification with an explicit inconclusive class, rather than a summary that has to pick one mode.

**Priority.** (a), (b) and (e) are testable against the engine as specified and need no Phase E. (b) is
the one to write first: it tests an invariant the spec states and nothing currently checks, and unlike the
others it can fail today.
*Confirmed 2026-09-20:* §8.2 reaches the same conclusion on (b) from Sachdeva & van Nuenen and from Li &
Tao, independently of SEAA. (a) is reclassified: the test is still Phase-B-testable, but its remedy is a
design decision (**C1**, P1) that has to be settled before Phase B rather than a test to be written.

### 7.11 Open questions this addendum raises

8. Should `E-DR`'s dose–response reporting be adopted as the standard form of §2.3's constant sweep, or
   are they separate Phase E requirements? (§7.2)
9. `M11.C.4` asserts a deferred cost "at the next nodal event". Is that a claim about *whether* or about
   *when*, and if the latter, is a time-to-event distribution the intended readout? (§7.3)
10. Is a rival-mechanism arm — projection disabled, reinforcement running — acceptable within `M0.4`'s
    one-parameter-set discipline, or does disabling a mechanism count as a second parameter set? (§7.5)
11. Should repertoire entropy per channel be added to the `M16` run log now, given that it needs no new
    state and gives `M4.D.6d` its only observable? (§7.4, §7.9)
12. If Phase F narrates, is the deterministic-verbalizer-plus-model pattern — with the epistemic scope
    enforced at the call boundary — the intended contract? (§7.6)
13. Does each mechanism draw from its own named RNG substream today, or from one shared stream? If
    shared, `M10.C.4a`'s ablation arms differ by more than the ablated condition. (§7.10a)
    **Answered 2026-09-20 (§8.1): neither.** Substreams are a partial fix; the answer is counter-based
    draws keyed by a stable event identity (C1), which corrects `M3.D.4`. The question is closed; the
    open part is C3's modelling decision — what counts as "the same event" across arms.
14. `M1.F.8` requires that order within a tick cannot decide the outcome, and nothing currently tests it.
    Should the permutation-equivariance test be written now, against a symmetric family, rather than
    waiting for Phase E? (§7.10b) **Answered 2026-09-20 (§8.2): yes**, and C5 adds two further forms of
    it. C1 is a prerequisite — without it a commutative reduction still fails the test.
15. Where a criterion has a mechanism-disabled arm, should it assert a **floor** on that arm — the
    observable is absent — rather than only a direction between arms? (§7.10c) **Still open**; C34
    supplies the arms to assert it on, and C25 the statistic for an arm with zero variance.

16. Do `E-RM` and C34 become one requirement — a single list of named control arms, including the
    rival-mechanism arm — or two? (§7.8 mapping table)
17. `E-RE`'s per-channel entropy is the only proposed test of `M4.D.6d`. Fold it into C39's readout, or
    keep it as its own `M16` column? (§7.8 mapping table, §7.9)

---

## 8. Revisions after the 2026-09 sweep (full reads, 2026-09-20)

Fourteen preprints from the twelve-month sweep were read in full by five sub-agents from `pdftotext` output (reports in `SWEEP_READING_REPORTS_2026-09-20.md`, with the same provenance limits as §0); the proposals drawn from them are consolidated, re-checked against spec v2 by keyword search, and prioritised in `SPEC_CANDIDATES_from_preprints_2026-09-20.md` (candidates C1–C42, X1–X4). Where a full read changes a statement in §1–§7, the change is recorded here and the affected line above carries a dated note.

**8.1 Seed pairing across arms is not guaranteed by a single generator, and per-mechanism substreams are only a partial fix (changes §2.6, §2.8; answers Q13).** Buffalo, Pearson & Klein 2026 (2603.11084) show formally (§3.3) that seed-matched runs with a stateful generator fail to couple the two arms of a counterfactual whenever the intervention alters the execution path, because every later draw index shifts. Spec `M3.D.4` (one seeded generator threaded explicitly) is exactly that design; byte-identical logs (`M3.D.5`) are a within-arm property and do not help. The paper also addresses §7.10(a)'s proposed remedy directly (§1): separate streams per event class are a coarse mitigation, because within a class the dependence persists and choosing the granularity requires anticipating every execution-path change. The full remedy is a counter-based generator keyed by a stable event identity (tick, stable object identifiers, purpose, index), with stable person identifiers across arms and no rejection sampling. Q13's answer is therefore: neither a shared stream nor per-mechanism substreams; keyed draws. This is the one finding of the sweep that corrects a stated requirement rather than adding to Phase E. Candidates C1–C4; §7.10(a)'s bit-identical-disabled-arm test is candidate C4's placebo test.

**8.2 Ordering is model content, with a number attached (strengthens §2.1; bears on Q14).** Sachdeva & van Nuenen 2025 (2510.10002) §3.3: with everything else fixed, the order in which two agents spoke moved first-round consensus from about 40% to nearly 90%. Li & Tao 2026 (2603.00113) name the scheduler and the visibility object as parts of the model to be versioned and logged. §7.10(b)'s permutation test should be written now (Q14: yes), and with the caution that a commutative batch reduction does not save `M1.F.8` if a stateful generator is consumed in loop order, which 8.1 removes. Candidates C5, C6, C30.

**8.3 Information access needs a rule and a mutant, not an intent (new).** He 2026 (VISA, 2607.28027) rule r14 and Zhou et al. 2026 (PIMMUR, 2509.18052) §2.3.2 make the same point from different directions: what an agent may read must be declared and enforced. The spec has the belief layer (`M9`) but no prohibition on reading another person's true state and no test that would notice. Candidate C9.

**8.4 Witness appraisal (new mechanism).** Holland et al. 2026 (2607.29546) eqs 4–5 give a witness rule with its own constant and a dependence on the witness's relation to both parties. Spec `M1.F.5` says witnesses appraise; `M4.C.1` does not say which tie's conductance applies. This is the one place the sweep proposes a mechanism rather than a test. Candidate C8.

**8.5 Object lifecycle (new).** VISA rules r8 and r13 ask for named create and remove functions with declared external effects. Mortality is in the spec (`M3.B.1`, `M7.C.1`, `M11.C.15`); the disposition of a dead person's conserved quantities is not. Candidate C15.

**8.6 Test-design additions to §2.6 and §7.10.** From Prasad 2026 (2607.07753): four-or-more-level monotonicity sweeps with a pre-declared primary readout (the same form as §7.2's dose–response, applied to person parameters); matched-magnitude contingency-severing mutants, stronger than the sever-the-input row in §2.6 because they keep the input's size; a persistence-after-spell test with a learning-off arm; a readout-saturation rule; an additivity residual. From Kalluri 2026 (2603.01189): ordinal criteria (rank order across three or more mechanisms) and frequency-controlled tests of asymmetric rules, since a 1.5× per-event asymmetry produced cumulative ratios of 0.07–0.55 depending only on event frequency. From TRAILS (2605.18890): re-encoding mutants, a fallback-provenance flag per decision, structural totals held fixed across arms, a paired rank statistic for degenerate arms with a declared multiplicity correction, and a per-result audit record with an unaudited column. From PIMMUR: sign-inversion mutants with a premise/mechanism register, constants frozen before the acceptance suite, arm-blindness of the engine, enumerated triad initial configurations. From Buitrago López et al. 2026 (2606.12369): a logged legality mask before selection. Candidates C11–C25, C33, C35–C37, C42.

**8.7 Ensemble statistics for Phase E (extends §2.6, §2.8, §7.8).** Blando et al. 2026 (2604.04543) practise adaptive ensemble size (add blocks until the interval at every reporting time is below a declared width) and time-resolved comparison with power reported; the reader adds UNDETERMINED as a third outcome. Holland's regime map at N = 10 versus N = 100 shows the polarise/consense boundary widening and flipping under marginal constant changes at small N, which is §2.2's point with a number; the reader proposes per-seed regime classification with an inconclusive class and a non-absorbing-bound test, which is the reporting side of §7.10(e)'s settling condition. Kurz 2025 (2512.18016) gives the exact form of §2.3's constant sweep for threshold-type constants: the trajectory is piecewise constant in each threshold, and the switch points are computable from the margins encountered, so an invariance interval per constant can be logged rather than sampled. Candidates C26–C29, C31.

**8.8 The LLM line (confirms §3, adds nothing to v2).** Buitrago López et al.: mean JSD 0.212 between LLM action choices and the intended policy, no prompt best across models, 135–1,337× slower. Wang et al. 2026 (2608.06485): personas differ at baseline but respond to life events alike, with changes about ten times smaller than human bands and a pull toward agreeableness. Li et al. 2026 (2608.24912): a benevolence bias that adversarial personas cannot push below the human baseline on prosociality or harm aversion. All three support `M3.D.6`. Protocol items for the exploratory notes only (X1–X4 in the candidates file).

**Not changed.** §2.2, §2.4, §2.5, §2.7, §2.10–2.12, §3 and §7.1–7.9 stand as written; the sweep papers add instances, not corrections.

**8.9 Reconciliation pass, 2026-09-21.** §8 was written against §1–§7 but the dated notes it calls for were applied only to §2.6, §2.8, §2.9 and questions 4, 6 and 7. This pass completed them: §7.10(a) now records that its proposed remedy is superseded by C1 while its test survives as C4's placebo test; §7.10(b)–(f) carry the C-numbers that extend them; §7.8 gains a mapping table placing `E-DR`, `E-SH`, `E-RM` and `E-RE` against C16/C24/C31, C18/C29, C34 and C39, so the two proposal sets do not compete; and questions 13–15 carry their answers, with 16 and 17 added for the two overlaps the mapping table leaves undecided. No claim in §7 or §8 was rewritten — only cross-referenced.

