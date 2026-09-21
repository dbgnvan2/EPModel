# Method-literature digest — agent-based simulation of human behaviour

Weekly sweep notes from the `agent-simulation-research` cron job. Newest first.
Completely separate from the Bowen corpus in `docs/theory/`.

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
