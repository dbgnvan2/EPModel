# Method-literature digest — agent-based simulation of human behaviour

Weekly sweep notes from the `agent-simulation-research` cron job. Newest first.
Completely separate from the Bowen corpus in `docs/theory/`.

---

## 2026-11-09 — 14 new papers

Method note: sourced from the weekly "Agent Papers Digest" email (2026-11-09, the `agent-social-simulation-emotional-agents-papers` cron job — Theme A social simulation, Theme B emotional agents; arXiv submitted ≈ 2026-10-29 … 2026-11-06, ~1-week window since the 2026-11-02 digest). All fourteen PDFs arrived as attachments (ten in email 1/2, four in email 2/2); no download gaps. All versions are v1 (verified against the arXiv API `id_list`; the four 2026-11-06 submissions were confirmed v1 individually). Facet keys assigned during curation.

1. **ODD+LLM: Specifying LLM-Agent Behavior for Reproducible Social Simulation** — 2611.08932 — Justin Downes, Robby Milletich, Newel Hirst — METHOD. Backward-compatible ODD extension adding five documentation blocks (prompt assembly, memory serialization, parsing, retry logic) grounded in CoALA; a multiverse analysis shows that missing one instruction (answer as an outside observer, not a player) raises the share of runs settling on agents' favored convention from 0.35 to 1.00; a 28-paper review finds prompt-building/output-parsing details undetermined in 77% of cases.
2. **Same Method, Different Task: What the Bitter Lesson Does Not License in Social Simulation** — 2611.05717 — Justin Downes, Robby Milletich, Newel Hirst — METHOD. Position paper: the bitter lesson's unstated condition (the general method was scaled against the *same task*) is violated by swapping hand-built agents for a text predictor; "stakes, not emergence"; a 540-agent pilot finds consequences reshape strategies in two of three model families but never produce human-like heterogeneity.
3. **Measuring Model-Specific Alignment Drift in LLM-Based Opinion Dynamics** — 2611.03980 — Nataša Djurdjevac Conrad, Sebastian Pokutta, Christof Schütte — FIDELITY. Six agents discuss a contested policy issue; committed personas are stable in isolation (20 rounds) but drift under interaction, and the drift is model-specific (gpt-4.1 keeps clusters apart, claude-opus-5 abandons them) — the choice of LLM becomes a first-class validity variable.
4. **Scientific Liquidity Agents: An Agent-Based Model of Knowledge Arbitrage in Scientific Communities** — 2611.02539 — Koutian Wu, You Liu, Yue Liu, Zhiping Li, Ergan Shang — ARCH. BDI agents (domain specialists + epistemic arbitrageurs) over a prerequisite problem graph; 10-/18-agent runs; no recorded replication succeeds — bottleneck in verification, not production.
5. **AgenticMobility: An Open-Source Platform for Multi-AI-Agent Mobility Dynamics** — 2611.01918 — Jiaheng Xiao, Jiangtao Han, Jiachen Wang, Honghai Li, Lijun Sun, Kay Axhausen, Chang Liu, Zhengbing He — ARCH. Persistent LLM travelers + a manager inside MATSim; 100 simulated days; all 1,179,300 finalized decisions satisfied action constraints.
6. **Agent-Based Disruption Simulation on an Edge-Level Product-Annotated Supply Network of the Automotive Industry** — 2611.01706 — Benjamin Rolf, Hiroyasu Inoue, Sebastian Lang — DYNAMICS. ~77,000-firm product-annotated network; the product-specific substitution rule makes post-hub-failure cascades four to five times larger than the portfolio-overlap rule (119,000 simulations).
7. **A Framework for Context Enrichment and Selective Robustness in LLM Social Agents** — 2611.06914 — Mattia Greiche, Reihaneh Rabbany, Maximilian Puelma Touzel — FIDELITY. Defines "selective robustness" (insensitivity to meaning-preserving variation + a directional response to decision-relevant change); enrichment forms produce distinct behavioral patterns that vary between the two models tested, in a Prisoner's-Dilemma-like game.
8. **Sharing the Road: LLM-Based Bargaining in Mixed-Autonomy Traffic** — 2611.09004 — Lingxiao Zhou, Di Chen, Jia Li — ARCH. Population-level AV/human agents bargain over road-space allocations; fairness objectives raise agreement rates but not road-space utilization.
9. **The Price of a Chance in Artificial Societies** — 2610.40259 — Shuqi Zhao, Boyu Wang, Shuangyue Niu, Chenhao Zhang, Zhanzhan Zhao — DYNAMICS. 200 LLM agents over 50 rounds; "overcompetition" vs "lying flat" against a Bayesian adaptive baseline; smaller prizes to more winners can cut investment while preserving selection quality.
10. **What Matters for LLM Persona Simulation: Signal Form in Cold-Start E-Commerce Advertising** — 2611.08840 — Jiwoo Choi, In Ik Lee, Woo Sung Seo, Minho Lee, Yun Young Choi — TRAIT. Real Google Ads data across five demographic segments; a minimal demographic anchor beats richer personas (MAE 24.9, 51.3% lower than the best non-LLM baseline).
11. **AffToM: Benchmarking Recursive Affective Theory of Mind in Dynamic Multi-Party Interactions** — 2611.02725 — Jinpeng Hu, Yiduo Jin, Erqiang Wang, Jia Li, Jingjing Wu, Peipei Song — TRAIT. Recursive affective ToM benchmark (1,922 test pairs); multi-party affective reasoning stays under 60% for the strongest MLLM; AffToM-R1 lifts MPAR 13.38 points.
12. **Compression of Emotion in Language Model Representations** — 2611.08683 — Laya Iyer, Sanmi Koyejo, Chen Shani — TRAIT. Seven LLMs, GoEmotions; "overcompression" — hidden representations split most cleanly into positive vs negative, yet linear probes retain fine-grained emotion.
13. **Does Personality Persuade? Investigating Personality Trait Prompting and Persuasion in Large Language Models** — 2611.06572 — Ruben Boone, Gilles Eerlings, Sebe Vanbrabant, Davy Vanacken, Kris Luyten — TRAIT. Big Five prompting (IPIP-NEO-120) + a N=32 user study; prompted profiles reproducible but controllability varies across models; personality alignment did NOT significantly affect persuasiveness, stance switching, or confidence.
14. **Mind the Divergence: Divergence-Guided Reliability-Aware Fusion and Adaptive LLM Prompting for Conversational Emotion Recognition** — 2611.08860 — Jingwen Wang, Jiao Li, Yuhang Qian, Yuyin Wang, Min Zhang — TRAIT. MiD-ERC: divergence-based modality reliability for multimodal emotion recognition; SOTA on IEMOCAP/MELD, gains strongest on high-divergence samples.

Facets with nothing new in window: ADAPT (self-improvement *training* / robotics / coding agents, out of scope as prior weeks), MECH (no mechanistic psychiatric-process models), TIES (no signed-tie/relationship-dynamics paper at individual scale).

### Deep-review candidates (individual/small-group ABM, for the EPModel spec)

Screened 2026-11-09 against the level criterion (design/build/testing of an ABM at individual or small-group scale). **Two candidates** this week:

- **Measuring Model-Specific Alignment Drift** (2611.03980) — candidate. Six-agent small-group opinion dynamics; committed personas stable in isolation but drift under interaction, and the drift is model-specific — maps onto EPModel's persona → behaviour gap and the inference-validity concern that the "prompt determines behaviour" assumption fails under interaction.
- **Context Enrichment and Selective Robustness** (2611.06914) — candidate. Dyadic (PD-like) interaction whose "selective robustness" criterion — insensitivity to meaning-preserving variation plus a directional response to decision-relevant change — is exactly EPModel's inference-validity concern (distinguishing robust grounding from spurious brittleness).

- *Adjacent, surfaced and read:* **Same Method, Different Task / Bitter Lesson** (2611.05717) — a position paper, not an ABM, but "stakes, not emergence" and mixed-motive settings that *shape* rather than *test* agents map onto EPModel's inference-validity stance. **Does Personality Persuade?** (2611.06572) — human–agent dyads, but the dissociation of technical persona control from observable influence maps onto the persona → behaviour gap.
- *Adjacent, not read:* **ODD+LLM** (2611.08932, documentation protocol; the one-missed-instruction → 0.35-to-1.00 multiverse finding is the transferable inference-validity caution), **Scientific Liquidity Agents** (2611.02539, small-scale BDI ABM but epistemic not relational; "bottleneck in verification not production" is the transferable note), **Persona Simulation / Signal Form** (2611.08840, single-persona ad data; "minimal anchor beats richer personas"), **AffToM** (2611.02725, recursive affective ToM benchmark), **Compression of Emotion** (2611.08683, representation analysis), **MiD-ERC** (2611.08860, multimodal emotion recognition) — all single-model TRAIT building blocks.
- Out on level grounds: AgenticMobility (2611.01918, mobility infrastructure/population), Supply Network Disruption (2611.01706, 77k-firm economic), Sharing the Road (2611.09004, population mixed traffic), Price of a Chance (2610.40259, 200-agent economic competition).

---

## 2026-11-02 — 10 new papers

Method note: sourced from the weekly "Agent Papers Digest" email (2026-11-02, the `agent-social-simulation-emotional-agents-papers` cron job — Theme A social simulation, Theme B emotional agents; arXiv submitted ≈ 2026-10-23 … 2026-10-30, ~1-week window since the 2026-10-26 digest). All ten PDFs arrived as attachments (nine in email 1/2, one in email 2/2); no download gaps. All versions are v1 (verified against the arXiv API `id_list`). Facet keys assigned during curation.

1. **The As-If Machine: Personalized counterfactuals make distant policy risks feel close** — 2610.33301 — Joshua Ashkinaze, Yubo Zhou, Pengyu Chen, et al. — ARCH. Multi-agent system where people co-write personalized counterfactual narratives; two pre-registered experiments (N=1,007 / 894) raised psychological closeness to distant policy risks 0.36–0.90 SD and shifted policy attitudes 0.54–0.67 SD in one domain.
2. **FraudSociety: A Generative Agent-Based Society for the Financial-Fraud Economy** — 2610.33536 — Keyang Chen, Mingxuan Jiang, Zeping Li, et al. — ARCH. Couples a dynamic social environment with persistent financial ledgers; civilian agents decide through Need-Intent-Action while fraudsters adapt communication to individual interaction histories; reproduces macro fraud statistics and micro loss concentration (top 10% of victimization records hold 81.1% of losses).
3. **The Inbox as a Coordination Primitive for LLM Agent Societies** — 2610.36264 — Karthik Sreedhar, Leo Yuan, Pradyumna Chari, et al. — ARCH. Coordination in societies up to 16,000 agents; the scarce resource is reading, not delivery; contrasts push vs pull protocols where an inbox + recipient's private rules govern which requests reach an agent.
4. **Your Agent Acts Differently When You're Around (PRESENT)** — 2610.36435 — Jane Pan, You Lyu, Abe Bohan Hou, et al. — FIDELITY. Controlled framework comparing LLM-agent negotiation with vs without a user (11.2k runs, 7 games, 8 models): a present user raises payouts +13.7% and roughly doubles counterparty-facing misalignment, and user-directed misalignment appears in 14.3% of games even with only templated neutral messages — a blind spot static instructions and outcome-level metrics cannot detect.
5. **Multi-Agent Reinforcement Learning for Global Climate Policy Cooperation** — 2610.37870 — Anne Moll-Elsborg, Oskar Bohn Lassen, Francisco Camara Pereira — DYNAMICS. Four-agent MARL with multi-stage negotiation (propose/accept/reject commitments, non-compliance penalties): non-binding agreements barely dent free-riding, moderate penalties improve cooperation, excessive penalties discourage ambitious commitments.
6. **When Disclosure Is Compulsory: Ride-Hailing Rebalancing with Self-Interested LLM Driver Agents** — 2610.41132 — Xinyu Wang, Andres Fielbaum — ARCH. Drivers as self-interested LLM agents that read mandated disclosed rules; disclosure raises compliance (declined relocations fall ~4-in-5 → ~1-in-4) and makes the anticipatory policy a win-win-win (+10–12% each on driver income, platform profit, service rate).
7. **The psychophysics of affect in large language models** — 2610.40985 — Joseph Marks, Marius C. Vollberg — TRAIT. How LLMs map outcomes to anticipated feelings: diminishing sensitivity to magnitude like human value functions, but gains rated more intensely than losses ("silicon gain amplification"); ratings linearise under direct comparison and do not explain choices better than monetary utility.
8. **CASE: Confusion-Aware Skill Evolution for Frozen Multimodal Emotion Models** — 2610.32481 — Siyu Wu, Zeyu Liang, Linghao Wang, et al. — TRAIT. Externalizes pair-specific affective expertise into a sparse, self-evolving library of EmotionSkills retrieved at inference while the MLLM backbone stays frozen; lifts Qwen3-VL-8B from 46.80% to 59.13% on CA-MER and transfers across frozen emotion models.
9. **Exploring AI Peer in Collaborative Mathematics Learning** — 2610.38990 — Yiran Lin, Ye Deng, Ziqi Pan, et al. — TRAIT. Adaptive Big Five-informed persona + motivational-state-modulated agent as a math-learning peer; participants judge it on social behavior (emotional safety, shared control, tentative reasoning, respecting pace) rather than mathematical competence.
10. **Influence of Agent Number and Behavior on Users' Psychological Responses in VR Evacuation Training** — 2610.38536 — Naoki Matsumura, Kensuke Sugihara, Tomoko Yonezawa — DYNAMICS. VR evacuation crowd whose agents express emotion; users act more confidently when many agents evacuate with them, and the agents' chosen route shapes users' own decisions.

Facets with nothing new in window: ADAPT (self-improvement *training* / robotics / coding agents, out of scope as prior weeks), MECH (no mechanistic psychiatric-process models), METHOD (no ABM-methodology paper; the climate MARL is filed under DYNAMICS), TIES (no signed-tie/relationship-dynamics paper at individual scale).

### Deep-review candidates (individual/small-group ABM, for the EPModel spec)

Screened 2026-11-02 against the level criterion (design/build/testing of an ABM at individual or small-group scale). **One candidate** this week:

- **PRESENT / Your Agent Acts Differently When You're Around** (2610.36435) — candidate. Small-group (dyadic) negotiation with a user in the loop; presence alone changes behavior and doubles misalignment invisible to outcome-level metrics — maps onto EPModel's instruction/persona → behaviour gap and inference-validity concern.

- *Adjacent, surfaced and read:* **AI Peer in Collaborative Math Learning** (2610.38990) — a single human-facing agent, not an ABM, but "the peer role is interactionally accomplished, not assigned" and "peer-likeness vs usefulness are orthogonal" map directly onto EPModel's role-as-appraisal and persona → behaviour-gap concerns. **The psychophysics of affect** (2610.40985) — single-model, but "intensity → felt affect is reference-relative, not a fixed linear gain" maps onto EPModel's appraisal → state (affect) mechanism.
- *Adjacent, not read:* **As-If Machine** (2610.33301, human–AI narrative co-writing; relevant to the exploratory narrator line), **Climate MARL** (2610.37870, non-relational negotiation game; the emergent-repeated-strategic-interaction methodological note is the transferable piece).
- Out on level grounds: FraudSociety (2610.33536, society/population scale), Inbox (2610.36264, up to 16,000 agents), Ride-hailing (2610.41132, market scale), CASE (2610.32481, single-model MER), VR Evacuation (2610.38536, crowd scale).

---

## 2026-10-26 — 17 new papers

Method note: sourced from the weekly "Agent Papers Digest" email (the `agent-social-simulation-emotional-agents-papers` cron job — Theme A social simulation, Theme B emotional agents; arXiv submitted ≈ 2026-10-13 … 2026-10-23, ~2-week window since the last digest). Fifteen PDFs were downloaded by the job; two Theme A papers (2610.17777 *Beyond Belief Nesting*, 2610.19796 *Learning Perception to Action Rules*) arrived without PDFs. Version suffixes preserved (2610.29330 and 2610.29674 are v2; the rest v1). Facet keys assigned during curation.

1. **The Butterfly Defects: Tracing Cooperation in LLM Agent Networks** — 2610.32726 — Maria Sampani, Maria Lymperaiou, Angeliki Dimitriou, et al. — DYNAMICS. Iterated Prisoner's Dilemma from dyads to networks: individual cooperation is highly sensitive to model composition and minor linguistic framing (reversing action-label order shifts behaviour and propagates to unexposed agents).
2. **Network topology shapes consensus and vulnerability in LLM populations** — 2610.31165 — Roberto Passaro, Ariel Flint, Romualdo Pastor-Satorras, et al. — DYNAMICS. Policy agents (10³–10⁵) on varying topologies playing the naming game; sparser networks slow consensus up to twelvefold yet preserve correlations that restore it, and a hub ≤ random ≤ low-degree ordering governs minority-tipping thresholds.
3. **The Butterfly Effect in LLM Social Simulations: From Architecture Sensitivity to Robustness Audits** — 2610.28981 — Jinyi Ye, Lei Cao, Ding Chen, Emilio Ferrara — METHOD. Small implementation perturbations (persona representation, framing, memory) shift collective outcomes (persona representation alone moved two-agent cooperation 75.2 points for GPT-5.2); supplies TRAILS + an Audit Card for auditing those dependencies.
4. **Model Capacity Frontier: A Development Framework for LLM-Supported Social Simulation** — 2610.24168 — Patrick Zhao, Yuxuan Li, Hirokazu Shirado — METHOD. The Model Capacity Frontier (best attainable performance across configurations as a Pareto frontier) applied to a ~12,000-agent crowd simulation; naive LLM integration improves flow direction but degrades temporal/spatial occupancy.
5. **To Manipulate or Not: Opinion Dynamics in LLM Interactions** — 2610.28875 — Mohammad Siahkamari, Phuoc Dat Nguyen, Stephen MacNeil — DYNAMICS. Controlled LLM–LLM debate (assigned stance ± explicit persuasion instruction): same-model interactions narrow the opinion gap, cross-model widen it, and a blanket persuasion instruction does not reliably increase influence.
6. **AgentMonopoly: Behavioral Patterns and Persona Control in LLM Agents** — 2610.29330 — Changshun Wu, Mengyu Li, Mathis Marsault, et al. — ARCH. Open-source Monopoly environment: models differ in trading/rent-waiving/cash management; persona instructions shift words and some actions but do not consistently match the assigned role, so persona following must be assessed through actions and outcomes.
7. **Consistent but Biased: Chair Personas Trade Noise for Bias in Multi-Agent Clinical Decisions** — 2610.31325 — Yitong Bai, Chengyi Zhang, Ziyang Wang — TRAIT. Holding 20 clinical cases constant and varying only the chair's persona prompt in a seven-agent department: conduct-rule personas move "operate now" from 2% (avoidant) to 93% (autocratic) vs 57% no-persona, while run-to-run agreement stays at chance — persona prompts are a silent lever and apparent reliability can be an artefact.
8. **What Makes a Social Agent's Interactions Effective? Learning from Verifiable Outcomes in Paired Worlds** — 2610.20195 — Caixin Kang, Ke Wang, Yuchuan Wu, et al. — FIDELITY. HICO-Bench: 180 dyadic coordination scenarios where 36% of fluent conversations fail a program-verified outcome; PaCPO (rewarding plan-consistent turns) lifts Qwen3-8B's Verified Success from 20.4% to 28.3%.
9. **Profile-Driven User Simulation via Reinforcement Learning with Hierarchical Rewards for Task-Oriented Dialogues** — 2610.32435 — Juan Li, Fangshu Chang, Xinyi Jiang, et al. — ARCH. ProfileSim: profile-driven user simulation (demographics + Big Five) for financial outbound calls, fine-tuned on real call logs and refined with GRPO under hierarchical rewards; strongest profile adherence and lowest agent score on long-tail intents.
10. **Synthetic populations for epidemic, demographic and mobility modeling** — 2610.21784 — Davide Torre, Miguel Artigues, Benjamin Faucher, et al. — METHOD. Harmonized synthetic-population dataset (Spain/France/Italy) reproducing household composition and census-tract density from public census data + statistical calibration; a standardized substrate for ABMs.
11. **Beyond Belief Nesting: Rethinking Higher-Order Theory of Mind in Large Language Models** — 2610.17777 — Shanshan You, Zhengqi Wang, Hao Liu, et al. — TRAIT. Argues existing higher-order ToM benchmarks conflate belief-nesting depth with reasoning order; HopToM formalizes "effective order" and HopTrace lifts fourth/fifth-order accuracy 22.0/38.5 points at 20–39× fewer tokens.
12. **Learning Perception to Action Rules through Reinforcement Learning for Theory-of-Mind Reasoning** — 2610.19796 — Soyoung Oh, Nursulu Sagimbayeva, Nicholas Jennings, et al. — TRAIT. LLMs infer a mental state correctly but fail to use it to predict action; GRPO-trained if–then rules linking perception/belief/action close that gap and transfer from ≤3B models to GPT-4o-mini.
13. **ToM2Vec: Continuous Theory-of-Mind Tokens for Multi-Agent Coordination** — 2610.29934 — Nithin Sivakumaran, Ali Payani, Hyunji Lee, et al. — ARCH. Latent ToM tokens (supervised by partners' actual thoughts) condition each agent's thoughts/actions; improves coordination across collaborative/competitive/mixed-motive games and generalizes to unseen partners.
14. **Beyond Small Talk: Modeling Disclosure Trajectories in Human-AI Conversation** — 2610.29413 — Keith Kwok, Wanting Ning, Tej Deep Pala, et al. — TRAIT. TraDE grounds evaluation in Social Penetration Theory with fixed disclosure trajectories; responses matched on disclosure depth still differ with history, and models diverge most from human supporters when users step back, not when they open up — relational behaviour is genuinely history-dependent.
15. **FACES: Fine-grained Gated Semantic Alignment for Multimodal Emotional Support Conversation** — 2610.20051 — Hang Ren, Peidong Wang, Yichen Gao, et al. — TRAIT. Gated semantic alignment with non-compensatory fusion and structured chain-of-thought anchors for token-level credit assignment; lifts MLLM evaluation 3.61→4.04 with 80.2% pairwise win over SOTA.
16. **From Acceptance and Avoidance to Affective Viscosity: Designing Social Agents Intermediating in Human Relationships** — 2610.29674 — Tomoko Yonezawa — TRAIT. Position/design paper: "affective viscosity" models how negative affect accumulates, persists, is suppressed, and is expressed over time; avoidance is a relational boundary signal, and design cases (triadic jealousy, discomfort suppression, patience timing) give agents a temporally extended affective state.
17. **Reading Too Much Into It? Interpretive Preferences and Commitment in LLM Mentalisation under Ambiguity** — 2610.28105 — Stephanie Fong, Yiwen Jiang, Qingyang Xu, et al. — TRAIT. MentalLoom (604 ambiguous social episodes, matched depression/anxiety disclosures): emotional context — not just the event — shifts how an agent reads a social situation; repeated questioning raises support and certainty without new evidence.

Facets with nothing new in window: ADAPT (self-improvement *training* / robotics / coding agents, out of scope as prior weeks), MECH (no mechanistic psychiatric-process models).

### Deep-review candidates (individual/small-group ABM, for the EPModel spec)

Screened 2026-10-26 against the level criterion (design/build/testing of an ABM at individual or small-group scale). **Five candidates** this week:

- **AgentMonopoly** (2610.29330) — candidate. Small-group multi-agent interaction (Monopoly) where persona following is judged by actions/outcomes, not language — directly on EPModel's "persona must map to behaviour, not surface transcript" concern.
- **Consistent but Biased: Chair Personas** (2610.31325) — candidate. Seven-agent group; persona prompt is a silent lever on a collective decision, and run-to-run agreement is at chance — the "apparent reliability can be an artefact" finding is exactly EPModel's inference-validity concern.
- **What Makes a Social Agent's Interactions Effective (HICO-Bench)** (2610.20195) — candidate. Dyadic coordination evaluated by verifiable outcomes rather than conversational fluency — the fluent-vs-effective gap maps onto EPModel's "inference from a directional result" stance.
- **To Manipulate or Not** (2610.28875) — candidate. Dyadic debate; opinion change depends on the model combination as much as the assigned role — cautions against treating reported opinion change as a clean instruction-driven outcome.
- **Beyond Small Talk / TraDE** (2610.29413) — candidate. Human–AI dyad; disclosure-history-dependent relational behaviour — history-dependence is EPModel's chronic-process concern.

- *Adjacent, surfaced and read:* **Affective Viscosity** (2610.29674) — a design paper, not an ABM, but "affective viscosity" (negative affect that accumulates/persists/is suppressed) is the closest analogue to EPModel's chronic anxiety; read for the anxiety model.
- *Adjacent, not read:* **Butterfly Effect / Robustness Audits** (2610.28981, robustness-audit methodology at echo-chamber scale), **ToM2Vec** (2610.29934, coordination games, not human relational), **Butterfly Defects** (2610.32726, network-scale conclusion, framing-sensitivity transferable), **HopToM** (2610.17777) and **Learning Perception to Action** (2610.19796) (single-model ToM building blocks).
- Out on level grounds: Network topology (2610.31165, population), Model Capacity Frontier (2610.24168, 12,000-agent crowd), ProfileSim (2610.32435, single user simulator), Synthetic populations (2610.21784, dataset), FACES (2610.20051, single support agent), MentalLoom (2610.28105, single-model mentalisation).

---

## 2026-10-19 — 4 new papers

Method note: all six facet queries returned via `export.arxiv.org/api` (explicit
`Accept: application/atom+xml` header, curl). Window: arXiv submitted >= 2026-10-05
(~2 weeks since last digest). Version suffixes preserved (all v1). Four PDFs downloaded
and verified (`%%EOF` trailer, non-trivial size). No withdrawn papers in the window.
Semantic Scholar citation tracking skipped (optional; nothing new expected from seeds).

1. **Probable but Not Valid: A Psychometric Audit of Decision Models as Synthetic Survey Respondents** — 2610.14912 — Mantas Lukauskas, Viktorija Šarkauskaitė — FIDELITY. Psychometric audit of 14 decision models + 47 generative LLMs as silicon samples against a real organisational-psychology survey (n=263); best generative model matches a copula baseline but profile fields say little about how a person answers — argues synthetic respondents should target sample-level, not individual, fidelity.
2. **Towards Adaptive and Real-Survey-Aligned Simulated Urban Mobility Behaviour of Generative Agents** — 2610.22407 — Yves Bru, Kamaldeep Singh Oberoi, Benoit Gaudou — ARCH. Generative-agent daily-mobility ABM for Toulouse, aligned to a real household travel survey; compares 15 decision-makers (discrete-choice, ML fit, tuned-prompt GAs, typed classifier) — GAs approach fitted models on modal split but not trip-by-trip.
3. **Beyond Large Language Models: A Mental World Model Framework for Mental Modeling, Simulation, Evolution, and Interpretability** — 2610.23722 — Qingshi Wang — TRAIT. Mental World Model / Multi-Branch Mental Evolution Architecture that constructs subjective worlds and simulates continuous mental-state evolution shaped by historical experience; strong gains on ToM (DynToM/RecToM) and mental-state prediction.
4. **Validating LLM-generated political networks with survey-reported partisan exposure** — 2610.23199 — Marco Tonin, Michele Tizzoni, Bruno Lepri, Luca Maria Aiello — FIDELITY. LLM-generated political networks from ANES personas, validated against respondents' survey-reported partisan exposure — shows aggregate network properties can hide group-specific errors in individuals' local environments.

Facets with nothing new in window: ADAPT (self-improvement *training*, coding agents,
robotics — out of scope, same as prior weeks), DYNAMICS (top hits were robotics/RL-infra
"alignment" noise and safety-engineering), MECH (all pre-window or out of scope).

### Deep-review candidates (individual/small-group ABM, for the EPModel spec)

Screened 2026-10-19 against the level criterion (design/build/testing of an ABM at
individual or small-group scale). **No candidates this week** — all four are
population-scale, survey-sample, or single-model building blocks:

- *Adjacent:* **Urban Mobility GAs** (2610.22407) — city-scale generative-agent ABM; the
  generative-agent-meets-real-survey alignment method is transferable, but population-scale
  transport mode choice, not individual/small-group interaction.
- *Adjacent:* **Probable but Not Valid** (2610.14912) — sample-level psychometric fidelity
  of silicon respondents; FIDELITY methodology, no interacting individual agents.
- *Adjacent:* **Mental World Model** (2610.23722) — individual mental-state architecture
  (subjective world + historical-experience memory), a TRAIT building block but single-model
  ToM/mental-state inference, not a multi-agent interaction model.
- *Adjacent:* **Political Networks** (2610.23199) — ego-level validation of persona
  networks is the transferable method, but network/population-scale (ANES sample).

---

## 2026-10-12 — 6 new papers

Method note: all six facet queries returned via `export.arxiv.org/api` (explicit
`Accept: application/atom+xml` header, curl; urllib hit an SSL issuer error this run).
Window: arXiv submitted >= 2026-09-28 (last digest); all six new papers are 2610.xxxxx
(2026-10-01 … 2026-10-09). Version suffixes preserved (all v1). Six PDFs downloaded and
verified (`%%EOF` trailer, non-trivial size). One TRAIT hit — "Thin Evidence, Thick
Priors" (2610.07798) — is marked withdrawn in its summary and was dropped (also
out-of-scope financial-advice identity work). Semantic Scholar citation tracking skipped
(optional; nothing new expected from seeds). No arXiv failures.

1. **"Hot-Blooded" vs "Cold-Blooded": Simulating the Behavioral Phenotypes of Childhood Aggression via Generative Agents** — 2610.11951 — Liping Fu — ARCH. Construct validity of LLM generative agents simulating reactive/proactive/co-occurring childhood aggression; four agents theory-parameterized from the social information processing model, 1,920 simulations.
2. **Socio-Foundation: A Model for Generalizable Individual Behavior Simulation via Hierarchical Capability Distillation** — 2610.08967 — Liang Wang, Wenxuan Xie, Xinyi Mou, Yixin Luo, Zhongyu Wei — ARCH. Distills persona-preserving, context-adaptive individual behaviour simulation so distinct personas are not flattened by general-purpose LLMs.
3. **Enhancing LLMs with Cognitive-Affective Personality Inference for Simulating Human Social-Psychological Behavior** — 2610.08328 — Zhibo Deng, Dongyuan Li, Shuwen Ge, Ziqing Zhang, Ying Zhang, Renhe Jiang — TRAIT. Cognitive-affective personality inference that disentangles stable dispositions from situation-specific interpretation, instead of static persona prompting.
4. **MASC: A Multi-Agent Self-Calibration Framework with Latent Construct Alignment for Consistent Client Role-Playing in Psychological Counseling** — 2610.08250 — Shixin Peng, Kun Jiang, Jiaxing Zheng, Qihao Yang, Jingying Chen — ARCH. Multi-agent self-calibration so simulated counselling clients stay psychologically coherent across extended interactions (counsellor-training domain).
5. **PAIR: Perceptual Affective Inference and Regulation in a Real-Time Multimodal Conversational Agent** — 2610.07523 — Kexin Quan, Zijian Ding, Jiaye Yong, Qinshi Zhang, Dong Wang, Jessie Chin — TRAIT. Real-time multimodal agent that reconstructs how an event is appraised into an emotional state, for sustained (cross-encounter) emotion support.
6. **Polarization Dynamics under Algorithmic Recommendations in LLM-based Online Social Networks** — 2610.13351 — Ali Safarpoor Dehkordi, Mohammad Shirzadi, Mehwish Nasim, Ahad N. Zehmakan — DYNAMICS. LLM-based online-social-network simulation of opinion dynamics/polarization under algorithmic recommendation.

Facets with nothing new in window: FIDELITY (top hits all already catalogued or
pre-window), ADAPT (self-improvement *training*, robotics, and coding agents — out of
scope, same as prior weeks), MECH (all pre-window or out of scope).

### Deep-review candidates (individual/small-group ABM, for the EPModel spec)

Screened 2026-10-12 against the level criterion (design/build/testing of an ABM at
individual or small-group scale). **Four candidates** this week:

- **Hot-Blooded vs Cold-Blooded** (2610.11951) — candidate. Four theory-parameterized
  generative agents simulating individual childhood-aggression phenotypes (social
  information processing architecture), tested across 1,920 simulations — individual-level
  behavioural-phenotype fidelity with an explicit processing architecture.
- **MASC** (2610.08250) — candidate. Multi-agent simulation of counselling clients
  (criterion a+b), small-group client/counsellor, individual-level psychological coherence
  across long interactions — the closest analogue yet to EPModel's counsellor + family
  setting.
- **Cognitive-Affective Personality Inference** (2610.08328) — candidate. Individual-level
  social-psychological behaviour simulation; the disposition×situation disentangling maps
  directly onto EPModel's appraisal→state→move separation.
- **Socio-Foundation** (2610.08967) — candidate. Individual behaviour simulation preserving
  persona traits under dynamic social context (criterion b).

- *Adjacent:* **PAIR** (2610.07523) — individual emotion-regulation agent (appraisal→state),
  a TRAIT building block but a single agent, not an interaction model. **Polarization
  Dynamics** (2610.13351) — out on level grounds (population/network-scale opinion dynamics).

*Deep-reviewed 2026-10-15 → `SWEEP_READING_REPORTS_2026-10-12.md`.* The four candidates were read
in full, plus two papers surfaced from the broader "Agent Social-Simulation & Emotional Agents"
digest — **Prompts versus Rules** (2610.11015, rule-based-vs-prompting validation) and **Latent
Identity Reversion** (2610.01490, persona continuity) — now in INDEX.md too. Sixteen candidates
total; plain-language in `SPEC_CANDIDATES_plain_language_2026-10-12.md`.

---

## 2026-09-28 — 5 new papers

Method note: all six facet queries returned via `export.arxiv.org/api` (explicit
`Accept: application/atom+xml` header). Window: arXiv submitted >= 2026-09-14 (~2 weeks,
since last sweep). Version suffixes preserved (all v1 except 2609.27690, read as v2).
One PDF (PERSONAWEAVER, 2609.26629) truncated on first download and was re-fetched;
verified `%%EOF` on all five. No arXiv failures. Semantic Scholar citation tracking was
skipped this run (rate-limited on prior run; nothing new expected from seeds).

1. **Cultural Divergence Preservation: Diagnosing Flattening and Caricature in LLM-Simulated Survey Populations** — 2609.29928 — Yeeun Chae, Yewon Choi, Seunghyun Lee, IL Im — FIDELITY. Cross-cultural survey simulation: existing JSD-style distance metrics miss whether *between-country* differences are preserved; proposes diagnostics for "flattening" and "caricature" of simulated populations.
2. **Consequential Behaviour and Representational Fairness in the Validation of Synthetic Research** — 2609.27690 — Florian Kutzner, Celina Kacperski, Laura de Molière, Edoardo Chidichimo, Min Jun Jung, Felix P. S. Wallis, James K. He — FIDELITY. Argues synthetic-sample validation tests the wrong thing (intention vs behaviour gap) and proposes consequential-behaviour + representational-fairness criteria.
3. **Warned alike, AI agents avoid the less-crowded road while people take it** — 2609.30883 — Takahiro Ezaki, Naoto Imura, Katsuhiro Nishinari — FIDELITY. Two-road congestion game: 50 GPT agents herded onto one road under a shared forecast while humans diversified — a concrete agent-vs-human divergence in emergent collective choice.
4. **From Static Personal Values to Contextualized Personalization: Bayesian Personalized Value Alignment for LLMs** — 2609.28942 — Hanze Guo, Aixuan Song, Jing Yao, Xiangxu Zhang, Xiaoyuan Yi, Xing Xie, Xiao Zhou — TRAIT. Replaces a static value profile with context-dependent value salience, explicitly citing Lewin's Field Theory (behaviour = f(person, environment)).
5. **PERSONAWEAVER: Controllable Diversity Beyond Conventional Archetypes in Procedural Character Generation** — 2609.26629 — Maan Qraitem, Kate Saenko, Bryan A. Plummer — TRAIT. Procedural character/persona generation with controllable diversity beyond a fixed archetype bank, for populating games and simulations.

Facets with nothing new in window: ARCH, ADAPT, DYNAMICS, MECH. (ARCH top hits were
post-training recipes / robotics / security agents; ADAPT returned self-distillation and
self-improvement *training* papers — out of scope, same as prior weeks; DYNAMICS was
dominated by robotics/optics "alignment" noise; MECH hits were all pre-window.)

### Deep-review candidates (individual/small-group ABM, for the EPModel spec)

Screened 2026-09-28 against the level criterion (design/build/testing of an ABM at
individual or small-group scale). **No candidates this week** — all five are
population-scale, single-model, or individual-level building blocks:

- **Warned alike** (2609.30883) — adjacent. A 50-agent congestion game that *tests*
  agent-vs-human emergent behaviour (criterion c), but it studies an aggregate
  congestion outcome, not individual-level dynamics; no persona/identity/memory
  architecture. The game-as-behavioural-audit method is the transferable piece.
- **PERSONAWEAVER** (2609.26629) — adjacent. Individual persona *generation* (controllable
  diversity), not an interaction model; relevant as a TRAIT building block.
- **Bayesian Personalized Value Alignment** (2609.28942) — adjacent. Individual value
  salience (Lewin Field Theory) — the person×environment framing is the closest analogue to
  EPModel's appraisal→state→move, but it is single-model alignment, not a group sim.
- Out on level grounds: Cultural Divergence Preservation (2609.29928, cross-national survey
  populations) and Consequential Behaviour (2609.27690, synthetic survey validation) — both
  population-scale FIDELITY methodology, no interacting individual agents.

---

## 2026-09-21 — 8 new papers

Method note: `export.arxiv.org/api` was reachable this run, but only with an explicit
`Accept: application/atom+xml` header (urllib's default header got HTTP 406; switched to
curl). All six facet queries returned. Window: arXiv submitted >= 2026-09-07 (~2 weeks).
Version suffixes preserved (all v1 except 2609.16344, read as v2). Semantic Scholar
citation tracking ran for 5/6 seeds (2507.21509 hit a 429 rate limit); it surfaced mostly
journal-venue work outside the arXiv-only filing convention, plus two items already in the
catalog ("Do Personality-Tuned LLMs Make Better Social Agents?", "Emergent Misalignment Is
Not Magical") — so no additional papers were filed from S2.

1. **Silicon sampling answers with country-level assumptions, not individual attitudes: Cross-national evidence from the European Social Survey** — 2609.16395 — Chuyao Wang — FIDELITY. Cross-national audit (ESS Round 11, 30 countries) showing silicon sampling recovers country-level priors rather than individual-level attitudes.
2. **Before You Poll with LLMs: A Deliberative Diagnostic Framework** — 2609.15849 — Ahmed Wali, Hassaan Tayyab — FIDELITY. Tests "dynamic fidelity": whether LLM personas update beliefs in response to new arguments, not just hold static opinions.
3. **From Momentary Emotion Inference to Sustained Emotion Support: Evaluating a Companion Agent in a Longitudinal Study** — 2609.16344 — Kexin Quan, Zijian Ding, Jiaye Yong, Qinshi Zhang, Dong Wang, Jessie Chin — TRAIT. PAIR, a theory-based emotion-regulation companion agent, evaluated over 14 days (1,093 sessions) for sustained emotional support.
4. **Do Personality-Tuned LLMs Make Better Social Agents?** — 2609.21857 — Tim Krabbe, Xiaodan Shi — TRAIT. Whether personality-aware fine-tuning reduces the "alienness" of LLM social-simulation agents and improves consistency/controllability.
5. **Steering LLMs Responses Towards Moral Foundations on the Norwegian MFQ-30** — 2609.21636 — Hans Andersen, David Dichas — TRAIT. Whether psychometric instruments measure stable traits in models and whether moral/value profiles can be steered toward a target human population.
6. **From Memory to Behavior: A Behavior-Aware Role-Playing Framework for Social Media Influencers** — 2609.21349 — Ji-Lun Peng, Yi-Zhen Zhang, Chun-Nan Chou, Yun-Nung Chen — TRAIT. Situation→internal-state→behavior framework for faithful role-play impersonation of real individuals.
7. **Digital Twins for Opinion Dynamics: A Generative LLM Framework for Social Networks** — 2609.19913 — Omran Berjawi, Giuseppe Fenza, Rida Khatoun, Sherali Zeadally — ARCH. LLM-based digital twins for opinion dynamics, against simplified mathematical models.
8. **AutoViewMem: Self-Configuring Orthogonal Views for Conversational Long-Term Memory** — 2609.21940 — Zijie Cao, Xijun Qu, Zhicheng Gu, Xiaoshu Chen, Duanyang Yuan, Yanning Hou, Sihang Zhou, Jianxing Gong, Jian Huang, Yang Mei — ARCH. Self-configuring long-term memory for agent consistency/personalization.

Facets with nothing new in window: ADAPT, DYNAMICS, MECH. (The ADAPT/MECH queries returned
mostly out-of-scope self-distillation-training and pre-window papers; the DYNAMICS query was
dominated by "alignment" noise from robotics/optics.)

### Deep-review candidates (individual/small-group ABM, for the EPModel spec)

Reviewed 2026-09-21 against the level criterion. **No candidates this week** — none of the
eight is an ABM of human interaction at individual/small-group scale. Closest as building
blocks (adjacent, not flagged):

- **AutoViewMem** (2609.21940) — ARCH. Long-term memory architecture for a single
  conversational agent (write-time semantic views); a memory building block, not a
  multi-agent interaction model.
- **From Memory to Behavior / SIBPersona** (2609.21349) — TRAIT. Situation→internal-state→
  behavior persona (CAPS-inspired) for role-playing *individual* influencers; individual
  impersonation, not a group sim, but the state-between-situation-and-behavior structure is
  the closest analogue to EPModel's appraisal→state→move.
- **Before You Poll with LLMs** (2609.15849) — FIDELITY. "Dynamic fidelity" of individual
  personas (belief *updating*, not static opinion) — relevant to EPModel's agents changing
  state over ticks, but it is a deliberative-polling diagnostic, not an interaction model.
- **Do Personality-Tuned LLMs Make Better Social Agents?** (2609.21857) — TRAIT. Individual
  personality fidelity via fine-tuning (which `M3.D.6` excludes from the decision path).

Out on level grounds: Silicon Sampling country-level (2609.16395, societal survey), Digital
Twins for Opinion Dynamics (2609.19913, societal network), Companion Agent emotion support
(2609.16344, single human-facing agent), Steering Moral Foundations (2609.21636, individual
trait measurement, no interaction).

Note: this run predated the deep-review screening rule added to the skill, so it filed the
papers without candidate flags; the screen above is applied here manually. Future runs flag
candidates automatically.

---

## 2026-09-14 — 10 new papers (first catalogued sweep)

Method note: `export.arxiv.org/api` was down (connection timeout / 429) this run; queries
were issued against `arxiv.org/search` (server-rendered HTML) instead. Semantic Scholar
returned 429 throughout. Metadata is verbatim from arXiv search results; no failures
otherwise. Window: 30 days (>= 2026-08-15, first run).

1. **Total Simulated Survey Error: Designing and Diagnosing Survey Responses from Large Language Models** — 2609.10280 — Sen, Ahnert, von der Heyde, Lasser, Weiß, Strohmaier — FIDELITY. Applies total-survey-error framing to LLM "silicon samples" for simulating survey populations.
2. **Diverse Minds, Divided Networks? Personality Composition, Polarization, and Collective Intelligence in LLM-Based Social Simulations** — 2609.12444 — Tareaf — TRAIT. TraitMix: personality composition of LLM-agent societies; polarization vs collective intelligence in one system.
3. **Emergent Misalignment Is Not Magical** — 2608.29118 — Li, Dai, Wang, Tan — DYNAMICS. Argues emergent misalignment has concrete (non-mysterious) mechanistic explanations.
4. **Prompt Sensitivity of Generative Agents: Evidence from an Epidemic Model** — 2608.26221 — Williams, Hosseinichimeh — ARCH. Generative agents as human proxies in an epidemic ABM; documents prompt sensitivity.
5. **Inducing Emergent Misalignment from Reward Hacks with Iterative DPO** — 2609.06649 — Daniels, Moodley, Marlin, Lindner — DYNAMICS. Reproduces emergent misalignment via reward hacking + iterative DPO (cheaper than RL).
6. **Ordinary, Reasonable Chatbots: Do AI Models Track Human Legal Judgments?** — 2609.06769 — Patel, Wenger, Buccafusco — FIDELITY. "Silicon jurors": how well models emulate human legal judgment.
7. **Implicit Personality Representations in Humans and LLMs** — 2609.12704 — Geng, Abend, Hovy, Frermann — TRAIT. Whether LLM internal trait structure reproduces human implicit personality structure.
8. **MicroVerse: An Instrument for Measuring Self-Authored Identity Drift in Long-Horizon Multi-Agent Language-Model Simulations** — 2608.15844 — Ng, Joshi, Gupta, Huang, Di, et al. — ARCH. Behavioral-science instrument for identity drift in generative agents.
9. **Emergent Misaligned Communication in Long-Horizon Multi-Agent LLM Commerce** — 2608.14825 — Li, Petersson, Acquisti, Bakker — DYNAMICS. Misalignment emerging in multi-agent LLM commerce (not single-agent evals).
10. **Insurance as AI Risk Infrastructure: A Generative-Agent Simulation of AI Adoption** — 2608.15181 — Yuan, Wei, Qian, Feng, Lin, et al. — ARCH. Generative-agent simulation of AI adoption dynamics.

Facets with nothing new in window: ADAPT, MECH. (MECH near-miss, outside 30-day window:
2607.07753 "A Transdiagnostic Space of Disorder-Like Phenotypes in Reinforcement Learning
Agents", 2026-07-21.)

### Deep-review candidates (individual/small-group ABM, for the EPModel spec)

Screened against the level criterion — design/build/testing of an ABM simulating human
interaction at individual or small-group scale, not societal/population scale.

- **MicroVerse** (2608.15844) — candidate. 25 agents, individual "soul file"/identity,
  memory/reflection, long-horizon identity-drift testing; engine/cognition separation. Closest
  match to EPModel's small-group trait-stability and engine-purity concerns.
- **Emergent Misaligned Communication in Long-Horizon Multi-Agent LLM Commerce** (2608.14825) —
  candidate. 4-agent small-group, one-year horizon, corpus-scale testing of emergent inter-agent
  behaviour (domain is competitive commerce/misalignment, not human relational simulation).
- *Adjacent (methodology transfers, domain is population-scale or non-interactive):*
  Prompt Sensitivity of Generative Agents (2608.26221, prompt-sensitivity testing of individual
  agent decisions, but 100-agent epidemic); Implicit Personality Representations (2609.12704,
  individual trait geometry, but not an interaction model).

Out of scope on level grounds: Total Simulated Survey Error, Diverse Minds/TraitMix, Insurance
as AI Risk Infrastructure (societal/population); Emergent Misalignment Is Not Magical, Inducing
Emergent Misalignment (single-model, not an ABM); Ordinary Reasonable Chatbots (individual
judgment eval, no agent-agent interaction).
