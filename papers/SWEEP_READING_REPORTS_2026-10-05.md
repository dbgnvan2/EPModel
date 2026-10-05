# Reading reports: three early method papers, full reads (2026-10-05)

Produced 2026-10-05 at the owner's request. These three papers were in `INDEX.md` as `read` (abstract only) since the
2026-09-17 catalogue, and were cited in `docs/conversation_agents_maturity_emotion_2026-09.md`, mostly from memory.
They were read in full from `pdftotext -layout` output by three readers working under
`sweep_readers_brief_2026-09-19/READER_TASK.md`. Each reader checked for overlap by keyword search in spec revision 10,
`SPEC_CANDIDATES_from_preprints_2026-09-20.md`, the 09-27, 09-27b, 10-04 and 10-04b report files, and TODO.md's
revision-11 held section. Each reader also checked the conversation's citation of its paper against the text. Reports are
reproduced as written. Only heading levels were changed.

The other four papers in the same 2026-09-17 group (Persona Vectors 2507.21509, Emotion Concepts 2604.07729, SEAL
2506.10943, Emergent Misalignment 2502.17424) were not read: they concern LLM internals, and under `M3.D.6` they could
only produce narrator-line items.

**Status: none of these candidates is in the spec.** A plain-language explanation with examples is in
`SPEC_CANDIDATES_plain_language_2026-10-05.md`.

| Part | Paper | Candidates |
|---|---|---|
| 1 | Redish 2004, *Addiction as a Computational Process Gone Awry* (Science 306:1944) | TD1–TD3 |
| 2 | Park et al. 2023, *Generative Agents* (2304.03442v2) | GA1–GA4, GA-X1, GA-X2 |
| 3 | Argyle et al. 2022, *Out of One, Many* (2209.06899v1) | OM1–OM3, OM-X1 |

Candidates labelled `-X` are for the exploratory narrator line only and do not enter the v2 spec.

**Corrections to the September conversation found by these reads** (a correction note has been added to that file):
- Redish does **not** model "a short horizon learns the vice". He states discounting is not the fundamental reason; the
  drug's uncompensable effect on the error signal is. The horizon claim belongs to the delay-discounting literature.
- Park et al.'s relationships did not emerge "entirely" from memory, retrieval and reflection: seed paragraphs authored
  pre-existing relationships, the crush behind the "date" was user-set, planning is a third component, and
  "relationship" was measured as mutual awareness.
- Argyle et al.: compression of spread and collapse onto modal categories are already in this paper's own appendix
  (Table 16), not only in later literature; the authors do not discuss them.

**Correction to the catalogue found by the Argyle reader.** The three later silicon-sampling papers (2609.10280,
2609.15849, 2609.16395) have not been read in full; spec revision 10 records two as abstract-only and `INDEX.md` lists
the third as `new`.

---

# Part 1: Reinforcement

## Reader report: Redish, "Addiction as a Computational Process Gone Awry", Science 306:1944 (2004)

**Scope.** I read all 848 lines of the pdftotext output: the three-page report (lines 1–331), the supporting online material (lines 333–848: model, Table S1, Figs S1–S7, simulation details) and both reference lists. Lines 294–331 also carry the opening of an unrelated article that shares the journal page (Mehlmann et al., GPR3). I ignored it. Some of the text is garbled:
- Greek letters come out as Latin letters (δ→"d", γ→"g", η→"h"), "=" comes out as "0" and ">" as "9".
- The supplement drops decimal points ("0025" is 0.025, "0001 ≤ γ ≤ 0999" is 0.001–0.999, "0.99n" is 0.99^n, "105 time-steps" is probably 10^5).
- Figures survive only as captions, axis labels and printed slopes.
- Eq. 4's max() bracket is broken, but the text restates it.

The paper does not report the number of simulations per point or any seeds. Supporting files read: `READER_TASK.md`, `EPMODEL_BRIEF.md`, and `DESIGN_LESSONS_model_design_papers_2026-09-17.md` (all headings, plus every passage on reinforcement, learning and horizon: §3.1, §7.4, §7.8–7.10, §8.6). Spec passages read in full: M1.A.4g, M1.A.4h, M1.A.5, M1.A.5a, M4.D.1–M4.D.6e (including M4.D.1a, M4.D.5c, M4.D.6a–6e), M4.G.1–M4.G.3, M7.D.1–M7.D.4, M10.B.5, M10.C.1, M10.C.1a, M11.4f, M11.C.4, M11.C.5, M11.C.12, M11.C.16, M11.C.17, M11.C.22, M11.C.29, M11.C.39, M11.C.40, M12.1 and the revision-10 decision table. Also read: KS23.2 in `docs/theory/kerr_book/ks23.md`, `docs/conversation_agents_maturity_emotion_2026-09.md` §4 (lines 151–169, 214–218), and TODO.md's "Spec revision 11 — held" section plus its "A horizon that changes with age" item (lines 125–136). I keyword-searched the spec, `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, `SWEEP_READING_REPORTS_2026-09-20/-09-27/-09-27b/-10-04/-10-04b.md` and TODO.md for: reinforc, discount, horizon, habituat, relief, prediction error, dopamin, addict, substance, temporal difference, TD learn, value function, extinction, elasticit, sensitiz, reward, relapse, reinstat, unlearn, unbounded / without bound, saturat, learning rate, hyperbolic. Main hits: C13 / M4.G.3 (habituation), C18 / M11.C.39 (persistence after a spell), E-RE / M11.C.40, M10.B.5, and nothing on update form, extinction or discount shape.

### A. Report

**1. Simulation formalism**
- [PAPER] Main text eqs. 1–3, SOM pp. 2–3: the agent lives in a discrete, partially observable semi-Markov state space. Each state has a dwell-time distribution, an observation, a reward R(s) and a drug term D(s). Transitions happen either by the agent's action or when the dwell time runs out.
- [PAPER] Eq. 2 is the reward-error signal: δ = γ^d[R(S_l) + V(S_l)] − V(S_k), where d is the time spent in S_k. Eq. 3 updates the value: V(S_k) ← V(S_k) + η·δ.
- [PAPER] Learning stops once the value correctly predicts the reward. The value then "compensates" for the reward, and δ = 0 (p. 1944).
- [PAPER] Eq. 4 is the drug case: δ = max(γ^d[R + V(S_l)] − V(S_k) + D(S_l), D(S_l)). The value function cannot cancel the drug term, so δ ≥ D > 0 always, and the values of states that lead to the drug "approach infinity" (p. 1945).
- [PAPER] Time is discrete. Determinism and seeding are not discussed.

**2. Agent architecture**
- [PAPER] SOM p. 3. The agent is a population of 1,000 sub-agents ("μAgents"). Each holds a believed state, a believed dwell time, a fitness and its own discount factor γ_i.
  - Each step, a sub-agent survives with probability equal to its fitness.
  - A rejected sub-agent is replaced by a copy of a fitter one. The copy takes the state and dwell time but keeps its own γ_i.
- [PAPER] Hyperbolic discounting overall comes from a mixture of exponential discounters, with γ_i uniform on 0.001–0.999 (Table S1; Fig. S6 shows hyperbolic discounting for natural rewards).
- [PAPER] Action selection has three steps (SOM eqs. S1–S4):
  - benefit = V(S_l) + E[R(S_l)] − V(s_i), averaged over sub-agents;
  - an action is selected in proportion to its benefit;
  - a soft-max with m = 4 decides whether to take it.
  - Other parameter: η = 0.05.
- [PAPER] The paper stresses that "δ is not equivalent to pleasure". δ is the gap between what was expected and what was observed (p. 1944). The value function maps to "wanting", not "liking" (p. 1946).
- [PAPER] The paper says the qualitative results do not depend on the TDRL variant (SOM p. 2), and that McClure et al.'s selection rule gave "qualitatively similar results" (SOM p. 4). This is stated, not shown.

**3. Interaction / network.** None. A single agent faces a fixed task world.

**4. Initialisation.** [PAPER] Each run starts in the action-available state S0 (Fig. S1). Initial values are not reported.

**5. Calibration and validation**
- [PAPER] Validation is qualitative. The paper reproduces three directions from the literature:
  - drug choice depends on prior experience and on the size of the alternative reward (Fig. 1);
  - drug and natural rewards are both sensitive to cost, but drug is less elastic (Fig. 2: slope −1.5 natural, −0.8 drug);
  - drug elasticity falls with experience while natural elasticity does not (Fig. S4: drug slopes −1.27, −1.05, −0.79, −0.61 over action windows 100–500; natural slopes −1.45 to −1.43 throughout).
- [PAPER] It makes three testable predictions: no blocking with cocaine, a double dopamine signal in experienced users (Fig. 3, supported by Phillips et al. 2003), and vouchers working best early.
- [PAPER] Dose sweep: D = 0.010, 0.025, 0.040 (Fig. S2). A higher D raises drug choice and changes the shape of the response to the alternative. This is a graded-parameter sweep of the kind M11.C.38 asks for.
- [PAPER] Nothing is fitted. No seed counts or variance are reported beyond scatter dots of individual runs.
- [PAPER] The author calls unbounded value growth "highly unlikely" in biology (p. 1947). Adding a dopamine-effectiveness factor of 0.99^n per drug receipt gave "similar properties" with all values finite. That result is described, not plotted.

**6. Failure modes and limitations**
- [PAPER] Not modelled: sensitisation, tolerance, withdrawal, titration, compensation, and differences between individuals (p. 1946).
- [PAPER] On extinction (p. 1947): simple value decay "does not model extinction very well", especially reinstatement after extinction. The paper says this needs additional components such as expanding the state space. This is argued, with a citation to Bouton (2002), and not simulated.
- [PAPER] The model rests on cocaine. Nicotine and opiates are covered by an argument, not by simulation.

**7. Software / engineering.** [PAPER] One shared parameter table across all simulations (Table S1), and each simulation is specified by its own state-space figure. Nothing on logging or reproducibility.

**8. Small-N, emotion, long horizon.** [PAPER] No families. Runs are 500–1,000 actions or about 10^5 time-steps. Nothing on development or years.

**Checking the September 2026 citation (conversation §4, line 151)**
- **"Addiction as a TD-learning pathology": accurate.** That is the paper's thesis.
- **"A TD learner with a short horizon reliably learns the vice … Redish's 2004 model … treat[s] it exactly this way": not supported, and the paper argues the opposite.**
  - Redish says hyperbolic discounting "is not the fundamental reason" the agent is trapped. He locates the cause in the drug's effect on the dopamine signal (p. 1946).
  - He sets his model against the discount-based accounts: rational addiction (Becker & Murphy), which assumes exponential discounting, and Ainslie's crossover effect from hyperbolic discounting.
  - In his simulations the discount distribution is the same in the drug and natural-reward arms (Table S1). The only thing that differs between the arms is D(s).
  - The "short horizon learns the vice" idea belongs to the delay-discounting literature that the conversation cited from memory (its line 218 already says this). It should not be attributed to Redish.
  - Minor: the only Bickel paper Redish cites (Bickel & Marsch 2001, ref 28) is cited for cost inelasticity, not for discounting.
- **Horizon is not what makes the learning go wrong here.** [INFERENCE] In this paper the vice is learned because the error signal can never be predicted away. A longer horizon would not fix that. With a clean signal, TD reaches a finite, reward-matched value at any discount mix (p. 1945).

**Bearing on M4.D.1a's [I] decision ("not reachable by lengthening a reinforcement horizon")**
- [INFERENCE] It cuts neither way.
  - The paper shows a learner can be trapped in a worse choice for a reason no horizon change cures. That agrees with M4.D.1a's last sentence, but for a different reason (a corrupted signal, not an objective that lies outside the channel).
  - It also shows that a TD learner with a clean signal finds the reward-optimal choice. So if differentiation were inside the automatic channel's objective, a horizon argument would apply.
  - M4.D.1a's claim rests entirely on the target being outside that objective, and the paper has nothing that corresponds to that. The `[I]` grade should stand, and this paper should not be cited for the claim or against it.
- [INFERENCE] TODO lines 132–136 (a horizon that changes with age) get no support here. The paper's discount factors are fixed for each sub-agent and never change with experience or age.
- [INFERENCE] **M4.D.6a's argument has an assumption it does not state.** It says a short-horizon relief proxy makes "every agent converge on CUTOFF by construction". That holds if the update adds relief cumulatively. Under a prediction-error update, CUTOFF's learned value would level off at a finite value. Choice would then depend on the relative values of the moves, as in Fig. 1, so the short horizon still pushes toward CUTOFF, but not necessarily to a full collapse. The direction M11.C.16 tests survives either way. The "by construction" wording depends on an update form the spec never states.

**Not transferable / cautions**
- Every number in the paper is a neuroscience constant about drugs and dopamine: D = 0.025, η = 0.05, m = 4, 1,000 sub-agents, γ on 0.001–0.999, 0.99^n, and the elasticity slopes. None of them is a family-model magnitude, and none may enter the M10 register as sourced.
- Neither dopamine nor drugs has an EPModel counterpart. M1.A.4g already treats substance use as a chronic `functional_level` pattern, and nothing here changes that.
- "Vouchers work best early" is an intervention-timing prediction for drug use. Transferring it to coaching would need a corpus source.
- The particle-filter belief state belongs to a perception model that EPModel does not need.

### B. Candidate additions to the EPModel spec

**Already covered (one line each):**
- Reinforcement on the automatic channel only: M4.D.6d.
- The learning switch and ablations: M10.B.5, M11.C.39, M11.C.40.
- The signal is not a feeling-state: consistent with M1.A.0 and M5.F.5.
- Geometric decay of a relief term on repetition: M4.G.3 / C13 (Redish's 0.99^n bound has the same form).
- The forbidden relief proxy and the horizon test: M4.D.6a, M11.C.16.
- A graded dose sweep: M11.C.38.

**TD1. The M4.D.6 update form is declared, and learned weights stay finite**
- **Proposed requirement.**
  - M4.D.6's update **MUST** be declared in config as one of two forms, labelled `[I]` with its learning rate:
    - **compensable**: the reinforcement is the signal minus the move's current learned expectation, so a fully predicted outcome stops reinforcing;
    - **cumulative with a declared bound**: an effectiveness factor that decays with repetition.
  - Each automatic-channel move's learned weight **MUST** stay finite under a constant signal.
  - Test: hold a constant signal on one move and assert its learned weight levels off. A mutant that adds the raw signal with no expectation and no bound **MUST** turn the test red.
- **Where it would live.** M4.D.6 (new item, M4.D.6f); M10.C.1a (learning rate and update form); unit test under M11.D.
- **Evidence.** Eqs. 2–4 and Fig. 1 show the update form alone decides between a finite level and unbounded growth, with the discount mix held the same. The bounded variant (p. 1947) is reported, not plotted. SHOWN, within the paper's model. Transfer: INFERENCE.
- **What it changes.**
  - It corrects an under-specification. M4.D.6b fixes the horizon but not the update form. A cumulative update can collapse the repertoire whatever the horizon, so M11.C.16's declared-horizon arm could still collapse.
  - It probably also settles M4.G.3's open question (C13, "review later"). A compensable update stops paying for a relief that has become predicted, so a separate habituation term is unnecessary.
  - Entrenchment the theory wants still comes from named stocks: M4.G.1 tie hardening and M4.D.5a's accommodation stock, which TD1 does not touch.
  - Improves inference validity, and theory fidelity because family style comes from a named mechanism.
- **Cost / risk.** Low. No conflict with M3.D.4/5, M3.D.6, M11.F.9 or M16.B. The spec's wording at M4.D.6a ("converges on CUTOFF by construction") should be softened to state the update form it assumes.

**TD2. A remitted learned pattern is suppressed, not erased**
- **Proposed requirement.** When an automatic-channel pattern remits (after a spell ends, or after a landed coach contact), the remission **MUST NOT** be implemented only as decay of that pattern's learned weight to baseline. The learned weight **SHOULD** persist under suppression, so that renewed load brings the pattern back faster than it was first acquired. Test, two arms with identical seeds:
  - arm 1: spell A teaches a pattern, the pattern remits, then spell B arrives;
  - arm 2: no spell A, then the same spell B;
  - arm 1 **MUST** show a shorter latency to the pattern in spell B. A pure-decay mutant **MUST** turn the test red.
- **Where it would live.** M4.D.6 or M4.G; new M11.C row next to M11.C.39; Phase D.
- **Evidence.**
  - Paper: simple value decay fails to model reinstatement after extinction (p. 1947). ARGUED, not simulated, with a citation to Bouton.
  - Corpus side: KS23.2 (regression as reversion to an intact older system, a reversible functional change) and M4.D.3a. Both are graded `[K-ext]`/`[M]` and **must not be attributed to Bowen**.
  - Transfer: INFERENCE.
- **What it changes.** Adds a mechanism constraint and a test. M11.C.39 covers persistence after a spell, but nothing covers a pattern coming back after it has remitted. Theory fidelity (regression returns the old pattern) and inference validity (a coached arm's gains are not counted as permanent unlearning).
- **Cost / risk.** Moderate: it needs a second learned quantity or a suppression term. **Needs an owner decision.** The corpus support is Kerr's extension, and the answer interacts with M1.E.7, since a landed coach contact raises `systems_perspective` and does not unlearn the automatic channel. No conflict with any stated rule.

**TD3. The discount form is declared, and M11.C.16 is run under two forms**
- **Proposed requirement.** The M4.D.6b horizon **MUST** be declared with its discount form as well as its length: exponential, or a declared mixture of exponentials, which gives hyperbolic discounting. M11.C.16's two directions **MUST** hold under both forms, at the same declared mean horizon, before either result is relied on.
- **Where it would live.** M4.D.6b; M10.C.1a (the form, `[I]`); M11.C.16 or Phase E's M17.E.1 sweep.
- **Evidence.**
  - A mixture of exponential discounters produces hyperbolic discounting (SOM p. 3, Fig. S6). SHOWN within the model.
  - Animals discount hyperbolically, and Ainslie's crossover depends on discount shape. Both are cited, not shown.
  - M4.D.6a's argument is a timing argument (immediate relief, deferred cost), which is the case where discount shape can reverse a preference. INFERENCE.
- **What it changes.** Adds a robustness rule for a functional form, of the class in design lessons §2.7. It also applies to the TODO horizon-as-state idea, if adopted: an age-varying horizon would need its form declared too. Inference validity.
- **Cost / risk.** Low: one extra arm pair. No rule conflict. The 0.001–0.999 γ range is the paper's own and **must not** be imported.

No narrator-line (TD-X) candidates. The paper has no LLM content.

---

# Part 2: Memory and belief

## Reader report: Park, O'Brien, Cai, Morris, Liang & Bernstein, "Generative Agents: Interactive Simulacra of Human Behavior", arXiv 2304.03442v2

**Scope.** I read all 1,366 lines of the pdftotext output: §1–9, the references, Appendix A and Appendix B. The text is complete in substance. These parts are damaged:
- Figures 1–9 survive only as captions, and the emoji glyphs in §3.1.1 are lost.
- The two-column layout interleaves Appendix B. The B.5 reflection questions and Klaus's answers appear before the B.5 heading (lines 1308–1349), and the file ends on the B.5 intro sentence.
- A footnote-marker artefact ("11It was good talking…", line 954) is harmless.

Spec passages I read in full:
- `M1.F` (F.1–F.9), all of `M3` including `M3.D.4a`–`M3.D.6` and `M3.E`
- `M4.B`, `M4.C` including `M4.C.9`, `M4.D`, `M4.G`
- `M6` (table and `M6.1`–`M6.3`), `M9.1`–`M9.8`, all of `M16` (A–F), all of `M17` (A–G)
- the revision-10 list of design lessons "noted, not specified" (around spec line 1936)
- ledger `L09.3`–`L09.4`

Other files checked:
- `READER_TASK.md` and `EPMODEL_BRIEF.md` in full.
- `DESIGN_LESSONS` §0–§5 and §7.1–§7.3, grepped for Park, Smallville, memory, reflection and retriev. This paper is not in the 30-paper DL set: there are no Park or Smallville hits, though its PDF is in `papers/`.
- The TODO "Spec revision 11 — held" section, including the MM1 and SV2 owner answers and the open question PM1.
- BB1–BB5 and MM1–MM4 (`…09-27b`), X5–X9 (`…09-27`), and PM1, PM2, PM-X1, RZ1, RZ2, EV1–EV4 and AN-X1 (`…10-04b`).

Keyword counts covered the spec, `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, all four `SWEEP_READING_REPORTS` files and TODO.md. Terms: memory, retriev, recency, importance, reflect, consolidat, persist, decay, forget, diffus, spread, plan, salien, rehears. "retriev", "recency", "forget", "diffus" and "rehears" have no hits in the spec.

### A. Report

**1. Simulation formalism**
- [PAPER] §5: a sandbox server keeps a JSON record of each agent's location, action and object. At each time step it applies agent output, moves agents and updates object states. It also sends every agent and object within a preset visual range into each agent's memory.
- [PAPER] App. A: the architecture "runs sequentially", at one real second per game minute. The length of a sandbox time step is not stated.
- [PAPER] The paper has no seeds, no determinism and no repeated runs. The end-to-end evaluation is one run of two game days (§7).
- [PAPER] §4.3: plans have three levels, decomposed recursively: a day of 5–8 chunks, then hour-long chunks, then 5–15-minute actions. App. A: only the high-level plan is made in advance, and the near future is decomposed just in time.

**2. Agent architecture**
- [PAPER] §4.1, memory stream: each record holds a natural-language description, a creation timestamp and a most-recent-access timestamp. Observations, reflections and plans all live in the stream, and all are retrievable.
- [PAPER] §4.1, retrieval score = α·recency + α·importance + α·relevance:
  - Each term is min-max scaled to [0,1], and all α are set to 1.
  - **Recency** decays exponentially per game hour "since the memory was last retrieved", with factor 0.995. The clock is keyed to last access, not to creation.
  - **Importance** is an LLM rating from 1 to 10, made once when the record is created. Examples: 2 for cleaning a room, 8 for asking a crush out.
  - **Relevance** is the cosine similarity of embeddings to a query.
  - The top-scoring records that fit the context window are passed to the model.
- [PAPER] §4.2, reflection:
  - It fires when the summed importance of recent events exceeds 150, which happened "roughly two or three times a day".
  - The 100 most recent records produce 3 questions. Each question is used as a retrieval query, and the model then produces 5 insights that cite record numbers.
  - Each insight is stored with pointers to the records it cites.
  - Reflections can cite earlier reflections, so they form trees.
  - Nothing says reflections are ever revised or removed.
- [PAPER] §4.3.1–4.3.2: a reaction or dialogue turn is conditioned on a summary retrieved with two queries: the observer's relationship with the observed entity, and that entity's current status. There is no relationship variable; a "relationship" is retrieved text.
- [PAPER] App. A: the cached "[Agent's Summary Description]" is re-synthesised "at regular intervals" from three retrieval queries: core characteristics, current occupation, and feeling about recent progress.

**3. Interaction and network structure**
- [PAPER] §3.1.1, §5: agents perceive others within a visual range. The architecture decides whether to walk by or talk, and dialogue is generated turn by turn until one agent ends it.
- [PAPER] §3.1.2: a user can command an agent by speaking as its "inner voice".
- [INFERENCE] The visual-range rule computes the audience from the environment and not from the sender's choice. That is the same principle as `M1.F.1b`.

**4. Initialization**
- [PAPER] §3.1: each agent starts from one authored paragraph, split at semicolons into seed memories. The paragraphs **include pre-existing relationships and opinions**, for example John Lin's ties to the Moores, and that he "thinks Sam Moore is a kind and nice man".
- [PAPER] §3.4.3: the user set Isabella's party intent **and** Maria's crush on Klaus.
- [PAPER] §5.1: each agent also starts with an environment tree covering its home, its workplace and the shops it commonly visits.
- [PAPER] Nothing about sensitivity to initial conditions is reported.

**5. Calibration and validation**
- [PAPER] Controlled evaluation, §6:
  - 100 Prolific raters, within-subjects. Each ranked five conditions on one random question from each of five categories (25 questions in total, App. B).
  - Ranks were converted to TrueSkill ratings:

    | Condition | TrueSkill μ |
    |---|---|
    | Full architecture | 29.89 |
    | No reflection | 26.88 |
    | No reflection, no planning | 25.64 |
    | Crowdworker-authored | 22.95 |
    | No memory | 21.21 |

  - Kruskal–Wallis H(4) = 150.29, p < 0.001. Every Dunn–Holm pair differs except crowdworker against no memory. They report d = 8.16 between the full architecture and no memory.
- [PAPER] §6.2: the ablations were **not re-simulated**. Every condition answered from the memory built by the full architecture. The authors chose this because re-simulating would make runs diverge, and they call the result one that would "likely represent a conservative estimate".
- [PAPER] End-to-end, §7.1.2:
  - Knowledge of Sam's candidacy went from 1 agent (4%) to 8 (32%).
  - Knowledge of the party went from 1 (4%) to 13 (52%).
  - Network density went from 0.167 to 0.74. Density is the share of pairs that both answer yes to "Do you know of <name>?"
  - 6 of 453 awareness answers (1.3%) were hallucinated. Each "yes" was checked against the memory stream.
  - 12 agents other than Isabella heard of the party, and 5 came. Of the 7 who did not, 3 cited conflicts and 4 were interested but did not plan to go.
- [PAPER] §8.2: models and hyperparameters were not varied, and the timescale was short. The crowdworker baseline is not "maximal human performance", and four crowdworker answer sets were regenerated (§6.2).

**6. Failure modes and limitations**
- [PAPER] §6.5.2, memory failures:
  - **Retrieval failure:** Rajiv says he has not followed the election, though he heard of it.
  - **Fragment:** Tom remembers that he plans to discuss the election at the party, but not that the party exists.
  - **Embellishment:** Isabella adds an "announcement tomorrow" that never happened.
  - **World-knowledge leak:** Adam Smith is described as the author of *Wealth of Nations*.
- [PAPER] §7.2, behaviour failures:
  - Location choices get less typical as the number of known places grows.
  - Physical norms are missed: a one-person dorm bathroom, shops closed after 5 pm.
  - Instruction tuning makes agents overly formal and "overly cooperative". Isabella rarely says no, and her stated interests drift toward what others suggest.
- [PAPER] §8.2–8.3:
  - Running 25 agents for two days cost "thousands of dollars" and took multiple days.
  - Named risks: prompt hacking, and "memory hacking", where a conversation convinces an agent of an event that never occurred.
  - Biases of the underlying model are inherited.
  - Platforms should keep "an audit log of the inputs and generated outputs".

**7. Software engineering**
- [PAPER] §5.1: the world is a containment tree rendered to text. Each agent holds its own subtree, which can go stale until the agent revisits the area.
- [PAPER] App. A: summaries are cached; decomposition is done just in time; batching dialogue and partial re-planning are suggested.
- [PAPER] Public code and a demo are linked (footnote 2).

**8. Small-N, family, emotion, long horizon**
- [PAPER] The Lin family of three appears only in the "day in the life" vignette (§3.3). The architecture has no emotion, stress or anxiety state. The longest horizon is two game days.
- [INFERENCE] Nothing here bears on decades or on family process. What transfers is structural: how a memory store retains, consolidates and records provenance, and how evaluation was designed.

**Conversation check, `docs/conversation_agents_maturity_emotion_2026-09.md` §2, line 79 and reference line 207**

| Claim in the conversation | What the text says | Verdict |
|---|---|---|
| Relationships "emerged entirely from the memory-retrieval-reflection architecture" | The paper names three components: memory, reflection **and planning**. Seed paragraphs author pre-existing relationships and opinions (§3.1). Density starts at 0.167, not 0. | **Overstated.** "Entirely" is wrong, and planning is omitted. |
| One agent asked another on a date | Maria invites Klaus to join her at the party (§3.4.3). The paper calls this asking out on a date (§1, abstract). The user set Maria's crush on Klaus as a seed. | **The date was seeded**, not emergent from an unseeded state. |
| Valentine's party organised through the social network | Isabella invites friends and customers she meets. 12 others heard of it and 5 attended. | Broadly correct. The paper is internally inconsistent: the abstract says "starting with only a single user-specified notion", but §3.4.3 lists two user-set seeds. |
| "relationships … emerged" | "Relationship" is measured as both agents answering yes to "Do you know of <name>?" (§7.1.1). | It measures mutual awareness, not attachment or bond. |
| 25 agents on one frozen model; weights contributed nothing agent-specific | The model is gpt-3.5-turbo with no fine-tuning described, and 25 agents is correct. But §7.2 attributes over-cooperation, formality and Isabella's interest drift to instruction tuning, and §6.5.2 shows world knowledge leaking in. | Literally defensible, but the weights did shape relational behaviour, just not per agent. |

**Not transferable / cautions**
- d = 8.16 appears to use TrueSkill's posterior σ (about 0.7), which measures uncertainty in a rating. It is not the spread of responses, so it is not a population effect size [INFERENCE from §6.5.1].
- Believability rankings measure rater preference. The human-authored condition ranked below two ablated architectures, so the measure rewards something other than human-likeness [INFERENCE].
- One run, one model, no seeds, no constant varied. 0.995, 150, the 1–10 scale and α = 1 are engineering choices with no reported sweep. In EPModel they would be `[I]`.
- The retrieval bottleneck exists because of the context window. The "misremembering" it produces is an engineering artefact and says nothing about the emotional function of forgetting.
- Relationship formation as acquaintance density does not apply to a fixed family with no exit from the field (`M6.I.7`).

### B. Candidate additions

**GA1. The persistence rule (held PM1) must declare its clock and its normalisation, and decay needs its own null arm**
- **Proposed requirement.** When PM1 is decided, `M9` **MUST** state for each belief class:
  - (a) which clock persistence runs on: time since the belief was written, or time since it was last read by the policy (`M4.B.2`) or last confirmed by a delivered event;
  - (b) that the retention value of an item is computed from that item alone, and **MUST NOT** be normalised across the store.

  If any decay is adopted, a criterion or readout that attributes belief loss or distortion to emotional process (`L09.4`) **MUST** be compared against a **decay-only arm**: the same persistence rule, with every anxiety dependence switched off. The decay constant and clock are `[I]`.
- **Where it would live.** `M9`, beside PM1. `M10.C` register. The arm goes in `M11.C` / `M17.D`.
- **Evidence.**
  - §4.1: recency keys on last retrieval, and min-max scaling makes every score relative to the rest of the store. This is a design statement (ARGUED), never ablated or swept.
  - §6.5.2: retrieval and decay alone produce fragments and omissions (SHOWN qualitatively).
  - That use-keyed retention has a nearby corpus precedent in `M1.C.4` / `L09.3` (tension reroutes onto "old preestablished circuits") is INFERENCE; that precedent is about triangles, not beliefs.
- **What it changes.**
  - It adds a fourth option to PM1's three. A read-keyed clock means beliefs that drive selection persist and unread beliefs fade. That is a feedback loop, and `M17.E.1` and BB5 should sweep it.
  - It forbids store-wide normalisation. Under normalisation, one write would change the persistence of every other item, an undeclared coupling of the kind RZ1 asks to be stated.
  - The decay-only arm separates misremembering produced by forgetting from misremembering produced by emotional process. This matters because `L09.4` describes active, functional obscuring, while this paper shows that neutral decay alone yields omissions.
  - Improves inference validity, and theory fidelity once the owner rules.
- **Cost / risk.** Low. One declaration and one arm. Both clocks are deterministic, so there is no conflict with `M3.D.4`/`M3.D.5`. Whether to key on reads is an owner decision. `M9.2` binds: a belief that persisted is not thereby true.

**GA2. Derived beliefs must declare their sources, their trigger and whether they are revised when sources change**
- **Proposed requirement.** If any belief item is computed from other belief items or from accumulated events rather than written directly by a delivered event, `M9` **MUST** declare three things for it. Candidates include `M9.6`'s attribution of where the difficulty lies, and `M9.3`'s who-is-sick belief.
  - (a) its source items;
  - (b) its trigger: the slow-tick clock, or an accumulation threshold over delivered events;
  - (c) whether it is revised, left standing or decayed when its sources change or decay.

  If an accumulation trigger is declared, an `M11.C` probe **MUST** compare two arms with equal elapsed time and different delivered-event load. The count of derivations **MUST** differ, and **MUST NOT** differ under a clock-triggered mutant. The threshold is `[I]`.
- **Where it would live.** `M9` (declaration), `M11.C` (probe). It touches `M4.G` / `M7`.
- **Evidence.**
  - §4.2: reflection is load-triggered (summed importance above 150), recursive, and append-only, with no revision described (ARGUED, as design).
  - Fig. 8: removing reflection lowers believability from 29.89 to 26.88 (SHOWN, for believability only).
  - §6.5.2, Tom's fragment: a derived item outlives the source it depended on (SHOWN, one case).
  - The transfer is INFERENCE.
- **What it changes.**
  - It fills a silent choice. Clock-triggered consolidation gives a crisis year and a calm year one update each; load-triggered consolidation gives the crisis year more. That is a theory claim either way.
  - Clause (c) decides whether a family can hold a conclusion after its evidence has faded. That reads directly on `L09.4`'s "misremember" and on `M9.3`'s hysteresis.
  - It improves theory fidelity by forcing the decision.
- **Overlap.** PARTLY covered by RZ1 (whether items are coupled). GA2 adds the direction of derivation, the trigger and the staleness rule. Fold it into RZ1 if preferred.
- **Cost / risk.** Low to state; this is an owner Bowen decision. The 150 threshold and the trigger form **MUST NOT** be imported as sourced.

**GA3. Belief-write records cite the delivered events that produced them, with a per-write check**
- **Proposed requirement.**
  - Every belief-write record (`M16.A.5`) **MUST** carry the identifiers of the delivered events it was computed from, and, under GA2, of the source belief items. Identifiers **MUST** be seed-derived.
  - An `M11.D` test **MUST** assert that every cited event was delivered to the writer, as target or witness, at or before the write tick.
  - A mutant that writes a belief from an undelivered event **MUST** turn the test red.
- **Where it would live.** `M16.A.5` (record), `M11.D` (test). It supports `M9.8`, `M4.B.2` and the `M16.C` renderer.
- **Evidence.**
  - §4.2: each reflection stores "pointers to the memory objects that were cited".
  - §7.1.1: every positive interview answer was verified by locating its source dialogue in the stream, which found 6 of 453 hallucinated (SHOWN as practice).
  - Transfer is INFERENCE.
- **What it changes.**
  - Revision 10 lists "appraisal and belief records citing event IDs" as noted but not specified, because it had no testable form. This supplies one.
  - It makes `M9.8`'s "written only by events delivered" checkable on every write, not only through `M11.C.36`'s mutant.
  - It lets the trace renderer show the path from an event to a belief.
  - Combined with GA1, it makes a belief whose source events have all decayed visible as a readout.
  - Improves inference validity.
- **Cost / risk.** Low. The record is emitted, not written (`M16.B.1`). It must not feed back into state, so `M16.T.3` holds. There is no conflict with `M3.D.5`.

**GA4. Every mechanism-disabled arm must declare when the mechanism was removed**
- **Proposed requirement.** Every arm that disables a mechanism (`M10.C.4a`, `M17.D.1`, `M17.D.2`, the `M11` mutation suite) **MUST** declare its onset:
  - from tick 0; or
  - from a fork tick, on a history produced with the mechanism on.

  A fork-onset result **MUST NOT** be reported as the mechanism's total contribution. It **MUST NOT** be called a conservative bound unless the from-tick-0 arm was also run and is larger.
- **Where it would live.** `M17.D`, reported under `M11.4f`. It applies only if Phase E adopts snapshot-and-fork.
- **Evidence.** §6.2: the ablations removed components only when the agents answered, on memory accrued by the full architecture. The "conservative estimate" claim is ARGUED and untested. A frozen history can make an ablated arm look better or worse than a re-run would (INFERENCE).
- **What it changes.** It adds a reporting rule. A fork-onset ablation measures what the mechanism does given a history it helped build, not what it contributes to that history. The two answer different questions and would otherwise be pooled silently. MM1's frozen-inbox probe is legitimately fork-onset by design and would carry the label.
- **Cost / risk.** Negligible. No rule conflict. Under `M3.D.4a` a fork is a plain state copy.

**GA-X1 (narrator line only). Human believability preference is not evidence of fidelity.** Any human rating of Phase F narration **MUST** score recovery of engine state (BB1), never a preference for one rendering over another. Evidence: §6.5.1. Human-authored answers (μ 22.95) ranked below two ablated LLM architectures (25.64 and 26.88), so raters rewarded detail and fluency rather than human authorship. This partly overlaps BB1, X7 and `M16.E.2`. What is new is that human raters fail this way too, not only LLM judges.

**GA-X2 (narrator line only). Count unsupported additions separately from contradictions.** AN-X1 voids passages that contradict the log. Narration should also report the rate of claims that are absent from the log rather than contradicted by it, and classify name-triggered world-knowledge leaks as their own category. Evidence: §6.5.2 (the invented announcement; Adam Smith). This is partly covered by `M16.E.3`'s traceability clause and by the display-name item in `…10-04b`.

**Already covered (one line each)**
- State held outside the LLM, with no LLM in decisions: `M3.D.6`, `M16.E.1`, DL §3.6.
- An audit log of inputs and outputs: `M16.A`.
- A propagation readout that verifies each receipt against the log: EV1.
- Over-cooperation and drift of interests toward others' suggestions: DL §3.2, X6.
- The audience computed from the environment and not chosen by the sender: `M1.F.1b`, `M3.E.1`.
- A per-agent world model that goes stale and differs from truth: `M9.1`, `M9.8`.
- Plan commitment across ticks: the `I-POSITION` state machine (`M5.D`). Day-plan decomposition does not transfer.
- Fork-and-compare: DL §2.8 (RecAgent).
- Cost and horizon limits: DL §3.1.

**What this paper adds to or subtracts from the held items**
- **PM1:** it adds the read-keyed clock and the no-normalisation rule (GA1). It subtracts any claim that 0.995 or importance-weighted decay is supported: neither was varied. It gives **no** support to an anxiety-dependent direction, because the architecture has no affect. On `L09.4`, its forgetting is functionally neutral, which is why GA1 asks for a decay-only reference arm.
- **PM2:** nothing new. GA1's decay-only arm is the persistence-side counterpart of PM2(b).
- **MM1–MM4:** nothing. There is no anxiety, no perceiver state and no initial-state distribution.
- **BB1–BB5:** narrator side only; GA-X1 extends BB1's rationale to human raters. Nothing for BB4 or BB5, apart from noting that GA1's read-keyed rule is a feedback loop BB5 should cover.
- **RZ1:** GA2 is a specific, directional form of coupling.
- **EV1:** the paper's measure is a coarse, two-time-point version of it.

**Owner questions raised**
1. Should belief persistence run on time since write, or time since last use (GA1)?
2. Are any belief items derived rather than written, and does a derived belief stand after its evidence fades (GA2)?

---

# Part 3: Fidelity of simulated populations

## Reader report: Argyle, Busby, Fulda, Gubler, Rytting & Wingate, "Out of One, Many: Using Language Models to Simulate Human Samples", arXiv 2209.06899v1

**Scope.** I read every line of the pdftotext output (lines 1–2702): §1–§9, Appendices A–E, Tables 1–17, the figure captions and the reference list.

**What was garbled or missing in the text:**
- Figures 2, 3, 4, 8, 9 and 11 survive only as axis labels and captions. Figure 7 is lost. Results that exist only in those plots are reported here from the prose.
- Table 11's columns are interleaved but readable.
- The paper has its own slips:
  - "1,1471 model queries" (App. E).
  - "5,914 in 20112" (App. E).
  - Cross-references to "Appendix 3" and "Appendix 5", but the appendices are lettered.
  - App. B.3 and D.2 cite "Figure 4" and "Figure 6 in the main text" for what are Figures 3 and 4.
  - The age variable is V161267 in D.1 but V161247 in Table 11.
  - C.4 calls a 6B-parameter model "the largest member of the GPT-Neo family", while GPT-J has its own entry in the Fig. 9 legend.
- `papers/INDEX.md` files the PDF as `2022_Out_of_One_Many_Ahn_2209.06899.pdf`. The first author is Argyle. There is no Ahn.

**What I read of the project.** In full: `READER_TASK.md`, `EPMODEL_BRIEF.md`, and `DESIGN_LESSONS_model_design_papers_2026-09-17.md` §0–§1, §3, §4, §7.6–§7.7 and §8. In the spec (rev10) I read in full:
- M10 (including M10.C.1–M10.C.5);
- M11.1–M11.4f, the whole M11.C table, M11.D, M11.E, M11.F (including M11.F.9) and M11.G;
- M15 (A–E);
- M17 (A–G and the revision-10 note).

**Keyword searches.** Terms: fidelity, silicon, subgroup, conditioning, backstor, Turing, pattern correspond, homogen, stereotyp, compress, dispersion, valid, survey, Argyle, "Out of One". Later I also searched associat, correlat, ablat, uninformative, blind, Simpson, pooled and marginal. Files searched:
- the spec;
- `SPEC_CANDIDATES_from_preprints_2026-09-20.md`;
- `SWEEP_READING_REPORTS_2026-09-27.md`, `-09-27b.md`, `-10-04.md` and `-10-04b.md`;
- `TODO.md`'s "Spec revision 11 — held" section;
- `INDEX.md` and `DIGEST.md`.

In the spec, "fidelity" hits only per-hop event fidelity (`M1.F.4`), a different sense of the word. Silicon, subgroup, conditioning, backstory, Turing and survey have zero hits in the spec.

**Correction to the brief.** The three later silicon-sampling papers have not been read in full:
- Spec rev10 (line ~1942) records 2609.16395 and 2609.15849 as "read at abstract level only; no requirement derived".
- `INDEX.md` lists 2609.10280 as `new`.
- "Silicon" has zero hits in every `SWEEP_READING_REPORTS` file.

So there are no silicon-sampling proposals to duplicate. The nearest full reads are:
- Kutzner et al. (CB1–CB3, whose L0–L3/E framework descends from this paper);
- Chae et al. (CD1–CD4, CD-X1);
- Ezaki et al. (WA1–WA5, WA-X1);
- Qraitem et al. (PW1–PW3, PW-X1);
- the held X1–X10.

I checked overlap against all of those, and against TM1–TM6, AQ1–AQ3, BB1–BB2 and NG1–NG3.

---

### A. Report

**1. Simulation formalism.** None that transfers.
- [PAPER] Each "silicon subject" is one stateless API query conditioned on one backstory. There is no time, no interaction and no state carried between queries (§5, §6, App. C.1, D.1).
- [PAPER] Generation settings by study:
  - Study 1 samples 128 tokens.
  - Study 2 reads the probability of a single next token, so temperature is "irrelevant" (C.1).
  - Study 3 samples 5 tokens at temperature 0.7.
- [PAPER] No seeds are reported. Temperature 0.7 "was not tuned" (App. A).

**2. Agent architecture.**
- [PAPER] The agent is the conditional distribution p(output | context) (§2).
- [PAPER] Conditioning is a first-person backstory built from template fragments in a fixed order. A missing variable drops its fragment (B.1, C.1).
- [PAPER] Ordinal values are mapped to words. For example, ages 40–60 map to "old" and incomes of $15k–50k map to "poor" (B.1).
- [PAPER] Study 3 uses a mock interview: eleven real ANES answers condition the model, which then predicts the twelfth (D.1, Fig. 10, Table 11).
- [PAPER] There is no memory, no reflection and no learning.

**4. Initialisation and sensitivity to it.**
- [PAPER, C.3, Fig. 8] Ablation on 2016 vote. Results:
  - No single backstory element accounts for all the predictive power.
  - Party predicts better than ideology.
  - Adding State or Political Interest "mildly hurt performance".
  - With party and ideology both removed, the eight remaining demographic elements together beat any single element.
  - The "no backstory" arm gives every subject the same prediction. The caption calls this "essentially equivalent to random chance". [INFERENCE] That is wrong: it equals the majority share, not chance.
- [PAPER] The authors state that "no attempt was made to optimize the template". They also say the token sets "were not tuned" (C.1, C.3).

**5. Calibration and validation (the main content).**
- [PAPER, §3] **Algorithmic fidelity** is defined as the degree to which patterns of relationships between ideas, attitudes and contexts in the model "mirror" those in human sub-populations. There are four criteria:
  - **C1, Social Science Turing Test**: generated responses are indistinguishable from parallel human texts.
  - **C2, Backward Continuity**: humans can infer key elements of the conditioning context from the response.
  - **C3, Forward Continuity**: responses proceed naturally from the context, reflecting its form, tone and content.
  - **C4, Pattern Correspondence**: responses reflect the relationships between ideas, demographics and behaviour seen in comparable human data.
- [PAPER, §3] The authors set no thresholds: "We do not propose specific metrics or numerical thresholds." Their standard is repeated support across data sources, measures and groups. Failing on one criterion lowers confidence; failing on more than one lowers it further.
- [PAPER, §3] They state that fidelity "does not imply that the model can simulate a specific individual". They also say known language-model shortcomings still apply.
- [PAPER, Study 1, §5, B.2–B.3]
  - Design: 2,873 Lucid raters, each list rated about 3 times, with the source hidden.
  - Turing task: raters judged 61.7% of human lists and 61.2% of GPT-3 lists to be human-written (p = .44).
  - Content: lists mentioning traits were 72.3% (human) vs 66.5% (GPT-3); lists rated extreme were 39.8% vs 41.0%.
  - Backward continuity: raters guessed the writer's party correctly for 60.1% of human lists and 52.8% of GPT-3 lists, against 33% chance. The 7.3-point gap is significant (p < .001; Table 3 coefficient −0.073).
  - GPT-3 lists were longer: mean 7.78 words (max 97) against 4.54 (max 15) (B.1).
- [PAPER, Study 2, §6, Table 1, Tables 8–10] Vote choice in the 2012, 2016 and 2020 ANES:
  - Republican vote share, GPT-3 vs ANES: 0.391 vs 0.404 (2012), 0.432 vs 0.477 (2016), 0.472 vs 0.412 (2020).
  - Whole-sample tetrachoric correlations: 0.90, 0.92, 0.94.
  - Pure independents: tetrachoric 0.31, 0.41 and 0.02. In 2020 κ was 0.02 and ICC 0.03.
  - Weak partisans: 0.71–0.84.
  - The authors attribute the independents result to independents being hard to predict (ARGUED, citing the political-science literature).
  - The 2020 wave lies outside GPT-3's training corpus, and correspondence held.
- [PAPER, C.2] The authors caution that κ and tetrachoric correlation are unreliable when more than 95% of cases fall in one cell. Examples: Liberals have proportion agreement 0.95 but κ 0.25–0.51; Blacks have 0.97 but κ 0.31.
- [PAPER, Study 3, §7, D.2, Tables 12–14, Fig. 4]
  - Method: Cramér's V between each ANES input variable and the GPT-3-predicted output, compared with the same pair in the human data.
  - The reported mean difference is −0.026.
  - My count of Tables 12–14: GPT-3's association is weaker in 76 of 110 pairs, stronger in 27 and equal in 7.
  - The largest gaps: political interest → discusses politics 0.40 vs 0.16; ideology → church attendance 0.28 vs 0.07; party ID → 2016 vote 0.48 vs 0.37.
  - A fully synthetic version (Fig. 11) is described as "highly similar". It is shown only as a plot.
- [PAPER, D.2 vs §6] **A tension in the evaluation.** Study 3 says it does not evaluate individual-level correspondence, because two draws from one distribution need not match. Study 2's proportion-agreement figures are per-respondent matches.
- [PAPER, D.3.2, Table 17] Temperature changes the result:
  - At 0.001 the mean error is +0.059 and the maximum +0.700, so associations are overstated.
  - At 0.7 the mean error is −0.026; at 1.0 it is −0.031.
  - Each setting was run once and "not select[ed] … for best fit".
- [PAPER, C.4, Fig. 9] Across five language-model families, performance rises with parameter count and also depends on the training corpus.

**6. Failure modes, biases and limitations.**
- [PAPER, Table 16] The Study 3 marginals of GPT-3's predictions are well off the human ones:

  | Quantity | ANES | GPT-3 |
  |---|---|---|
  | Age, mean | 50.1 | 35.5 |
  | Age, SD | 17.6 | 12.6 |
  | Age, min | 18 | 0 |
  | Patriotism, SD | 1.30 | 0.90 |
  | White | 80.3% | 97.4% |
  | Hispanic | 8.9% | 0.1% |
  | Graduate degree | 19.6% | 0.2% |
  | Male | 48.1% | 75.9% |
  | "Other" vote | 7.8% | 52.3% |
  | Trump vote | 43.8% | 24.5% |

  The main text does not discuss any of this.
- [PAPER, D.3.2] At temperature 0.001, "GPT-3 identified all respondents as white."
- [PAPER, Table 15, D.2.1] GPT-3 missing or non-compliant rates reach 23.8% (vote), 22.6% (ideology) and 14.6–14.8% (several items). Analysis is restricted to complete cases: 1,782 of 4,270 respondents.
- [PAPER, §6] The authors describe a mild overall bias against one candidate each year.
- [PAPER, §9] They name misuse risks: targeting groups for misinformation, manipulation and fraud.

**7. Software engineering and reproducibility.**
- [PAPER, App. A, C.1] Token sets are collapsed (e.g. {Donald, donald, Trump, …}) and then renormalised.
- [PAPER, B.1] Outputs were extracted with regular expressions plus "light manual post-processing".
- [PAPER, App. E] Costs: $29 (Study 1), $75 (Study 2), $1,428 (Study 3). Only final runs are costed, and "additional runs were performed as part of the experimental rhythm".
- [INFERENCE] So the no-selection statement covers the temperature runs, not the whole development history.

**8. Small-N, family, emotion, long horizon.** None. "Discusses politics with family and friends" appears only as a conditioning variable.

**What this paper adds beyond the later fidelity papers already on file** [INFERENCE]
- It is the source of the four criteria. Kutzner's L0–L3/E levels, and the "dynamic fidelity" in 2609.15849's abstract, extend them.
- Its own appendix already holds the evidence that later papers report as findings:
  - compression of spread (Table 16);
  - collapse onto the modal category, at both default and near-greedy temperature;
  - associations weakened at default temperature and strengthened at near-greedy temperature.
- That last point is a mechanism-level detail I have not seen in the held candidates: the sign of the association error depends on the sampling temperature.

**Check of `docs/conversation_agents_maturity_emotion_2026-09.md` §2 against the text**
1. "Persona-conditioned models reproduce some subgroup opinion distributions." This is a fair, cautious summary. The authors claim more: proper conditioning will "accurately emulate response distributions from a wide variety of human subgroups" (Abstract). The evidence is uneven: pure independents in 2020 come out near zero (tetrachoric 0.02), and the Study 3 marginals are badly skewed (Table 16). The conversation's "some" is closer to the data than the abstract is.
2. "Compressed and stereotyped … from … the subsequent literature" is partly wrong about where the evidence first appears. The compression is visible in this paper's own Table 16 (the age and patriotism SDs, modal-category collapse on race and education), and D.3.2 reports everyone coded White at near-greedy temperature. **The authors never discuss it** and never use the words compressed or stereotyped about their silicon subjects.
3. On stereotyping, the paper's data do not support a general claim of exaggeration:
   - Study 1's topic is human stereotypes of partisans, and the authors count reproducing them as fidelity.
   - GPT-3 lists were *less* diagnostic of the writer's party (52.8% vs 60.1%), and extremity ratings were equal (41.0% vs 39.8%).
   - At default temperature, GPT-3's associations were mostly *weaker* than the human ones (76 of 110 pairs).
   - Over-strong associations appear only at temperature 0.001.

   The accurate statement is: compressed spread and collapse onto modal categories are shown in this paper's appendix; stereotype-like over-association is shown only under near-deterministic sampling.
4. Bibliographic detail: Political Analysis 31(3) is the later journal version, which I did not see. The arXiv v1 is dated 14 Sep 2022 (header) and 16 Sep 2022 (title page).

**Not transferable / cautions**
- **C1, the Turing test, has no EPModel counterpart.** There are no human texts to be indistinguishable from. Scoring rendered traces against the corpus's clinical vignettes would tune to illustrations the corpus bounds. That is the fitting `M11.F.9(c)` and `M10.C.3` forbid.
- **Silicon sampling reweights the model's conditionals by a real population's distribution of conditions (§4).** For EPModel that would be calibration against data the project does not have. `M15.B.4` already does the only admissible version: it checks an imported population against the corpus distribution, as a check.
- **§8 argues silicon samples can be used before or instead of human data, once fidelity is established.** The EPModel counterpart of that move, reading model output as a finding about families, is what `M11.F.9(a)` forbids.
- **The paper's evidence is correlational agreement with one survey programme in one country.** It says nothing about simulated time, interaction or development, so it gives nothing on the maturity questions in the September conversation.

---

### B. Candidate additions to the EPModel spec

**OM1. Cross-sectional pattern correspondence: the signs of associations across a `D0` ensemble are checked against the associations the corpus states, both pooled and within strata.**
- **Proposed requirement.** Phase E **SHOULD** emit, for every `D0` ensemble (`M17.C.1`), the sign matrix of rank associations among the `M11.G.1` components and `basic_level`, computed across families at a declared readout tick. It **MUST** compare each cell with a corpus-stated sign where the corpus states one; cells with no corpus statement are unscored.
  - Each cell **MUST** be reported both pooled and within each declared `D0` stratum (at least the founders' `basic_level` band).
  - A pooled sign that reverses within a majority of strata **MUST** be flagged.
  - Matches and misses **MUST** be reported together (`M17.G.3`).
  - Pairs the corpus says are non-monotone (`M1.A.3c` overt emotionality, `M7.D.2d`) or trap-shaped (`M11.F.6`, `M11.C.28` symptom count) **MUST** be entered as unscored, not as a sign.
  - The list of corpus-stated signs is editorial. It **MUST** live with `M10.C.4`'s checks (`M10.C.4b`), graded `[T]`/`[#]` with its attribution grade.
- **Where it would live.** Phase E, `M17.B`, next to `M10.C.2a` (which admits exactly one association, DSI–anxiety). It is a readout, not an `M11.C` criterion, so it needs no `M11.4` exception.
- **Evidence.**
  - Criterion 4 and Study 3 (§7, Fig. 4, Tables 12–14) are the method: compare the whole association structure, not the marginals. SHOWN as practice.
  - §4 names Simpson's paradox as the reason conditional and pooled patterns can differ. ARGUED.
  - The transfer to corpus-stated signs is INFERENCE.
- **What it changes.** It adds a fidelity check that two-arm criteria cannot do. The `M11.C` criteria are interventional and pairwise. A model can pass every one and still produce across-family co-variation the corpus contradicts, for example through `D0`'s declared correlations. This improves theory fidelity and, through the strata, inference validity.
- **Overlap.** Partly covered:
  - `M10.C.2a`: one association, used as a check.
  - `M11.C.26`: identifiability, a mapping and not a sign matrix.
  - CB2: strata by position within a family, not across `D0`.
  - Kutzner's L3 (structure) was judged out of reach by the CB reader. I think a sign-only version is reachable without data.
- **Cost / risk.** Low compute, since it reuses Phase E ensembles. The editorial work is extracting the stated signs, and the corpus will supply few. It **MUST NOT** be used to select constants (`M10.C.4`: "checks, never parameters"). It sits next to AQ2's open owner question about reading bounds against the constant space. No conflict with `M3.D.4`/`M3.D.5`, `M3.D.6` or `M16.B`.

**OM2. Import-information ablation: a result on an imported family is reported beside a no-import baseline and with each imported field class removed in turn.**
- **Proposed requirement.** A Phase E result over an imported family (`M15`) **MUST** be reported beside the same arms run on a declared no-import baseline: the `M2` reference family, or a declared generic `D0`.
  - It **MUST** also be re-run with each imported field class replaced in turn by its no-information form: topology, tie kinds and states, dated events, and ratings. The no-information form is the reference value or the full declared range.
  - The report **MUST** name the field classes on which the envelope's direction depends.
  - Where the imported envelope does not differ from the baseline envelope beyond `M17.A.4`'s margin, the report **MUST** say that the import did not change the result. The result **MUST NOT** then be described as specific to that family's structure.
- **Where it would live.** `M15.D` (a new `M15.D.5`), Phase E.
- **Evidence.**
  - C.3 and Fig. 8 run a no-backstory arm, single-element arms and leave-one-out arms. They show that no single element carries the prediction and that some elements reduce accuracy. SHOWN for the paper's language model.
  - The transfer is INFERENCE.
- **What it changes.** It adds a reporting rule. Today `M15.D.2`–`D.4` say whether a direction survives the imported ranges and which quantity it flips on. Nothing says whether the imported information mattered at all. A result identical to the generic family's says nothing about this family, and presenting it as if it did leans toward the reading `M11.F.9(a)` forbids. This improves inference validity.
- **Overlap.** Partly covered:
  - `M15.D.3`: a flip within ranges, not removal of a field class.
  - `M17.D.1(c)`: homogeneous-family arm.
  - `M17.E.4`: two reference configurations.
  - PW2: topology bank.
  - None of these compares an import against a no-import baseline.
- **Cost / risk.** Compute rises by (number of field classes + 1) times. The no-information ranges are `[I]` and must be frozen under `M10.B.4`. Full-range replacement must not filter by `M10.C.4`'s bounds (PW3). No rule conflict.

**OM3. Blind arm identification before any `M11.E` human review.**
- **Proposed requirement.** Where `M11.E` assigns a criterion to human review (`M11.C.11`'s curve, `M5.F.2`'s threshold, `M11.C.9`), the reviewer **MUST** first receive the rendered traces (`M16.C`) or readout plots of both arms for several seed pairs, with arm labels, parameter values and seeds removed.
  - The reviewer **MUST** record which arm they judge to be which, and for `M11.C.11` whether the three phases are present, before the labels are revealed.
  - The identification rate **MUST** be reported with the review. A review whose blind identification is not better than chance **MUST** be reported as not supporting the criterion.
- **Where it would live.** `M11.E`, using `M16.C`.
- **Evidence.**
  - Study 1 (§5, B.2): raters blind to source judged content and inferred the writer's party. Backward continuity (C2) is defined as a human recovering the conditioning input from the output. SHOWN as method.
  - The transfer is INFERENCE.
- **What it changes.** It adds a test protocol. Today `M11.E`'s human reviews are unblinded, so a reviewer who knows which arm should show the effect can confirm it. This improves inference validity.
- **Overlap.** Partly covered:
  - X7: blind raters, for the narrator line only.
  - BB1 and PW-X1: machine recovery from narration.
  - `M11.D.20`: machine agreement of orderings.
  - None of these blinds the v2 human-review items.
- **Cost / risk.** Low to moderate, in owner time. With one expert reviewer, the identification rate must come from several seed pairs and should be reported with its count. Review outcomes must not drive constant changes without `M10.B.4`'s post-hoc label (and AQ1 if adopted). No conflict with `M3.D.6`: the reviewer is human.

**Already covered (one line each)**
- Forward continuity (output follows from the conditioning): deterministic engine, the `M11.C` directions, TM2's manipulation-check gate, NG2's tick-0 check.
- Repeated support across measures and groups instead of one threshold: `M17.G.1` audit record, CB1 (layers), CB2 (position strata), RP2 (proximal behaviour), `M17.A.4`.
- Distributions, not individuals: `M11.2` (ensemble-only criteria), `M17.B`.
- Agreement statistics unreliable when one cell dominates (C.2): `M11.4e` (rank-based paired statistic for degenerate arms), `M11.4d`.
- Robustness to the sampling-temperature analogue: softmax temperature is in `M10.C.1` and swept under `M17.E.1`; dispersion under WA1.
- No tuning of templates or token sets, and no selection of runs: `M10.B.4`, PD4 (no selected sample), WA4 (disjoint seeds).
- Complete-case analysis dropping 58% of cases: `M17.F.1(c)` (non-occurrence kept in the denominator), global rule 6.
- Reweighting skewed marginals toward a true population: only the check form is admissible, and `M15.B.4` has it.

**Narrator / LLM line only (does not enter v2; `M3.D.6` stands)**

**OM-X1. Persona populations: held-out attribute recovery per category, at two or more sampling temperatures.**
- **Proposed protocol item.** Any LLM persona population **MUST** be tested with Study 3's leave-one-out design: hold out each specified attribute in turn and have the model produce it from the rest.
  - It **MUST** report the produced distribution per category, including rare categories, against the specified distribution.
  - It **MUST** run at no fewer than two temperatures, one near-greedy, and report the sign of the association error at each.
  - A category produced at a small fraction of its specified rate **MUST** be reported as unrealised for that model.
- **Evidence.** SHOWN:
  - Table 16: Hispanic 0.1% against 8.9%; graduate degree 0.2% against 19.6%; age SD 12.6 against 17.6.
  - D.3.2: all respondents coded White at temperature 0.001.
  - Table 17: associations overstated at 0.001 (+0.059) and understated at 0.7 (−0.026).
- **Overlap.** Partly covered by X2 (between-persona SD), WA-X1 (identical-chooser reference), CD-X1 (flattening and caricature), PW-X1 (per-category adherence for move types) and X4 (neutral-persona contrast). New here: rare-category collapse in the persona's own attributes, and the finding that temperature sets the sign of the association error.
