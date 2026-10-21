# Method-literature digest — agent-based simulation of human behaviour

Weekly sweep notes from the `agent-simulation-research` cron job. Newest first.
Completely separate from the Bowen corpus in `docs/theory/`.

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
