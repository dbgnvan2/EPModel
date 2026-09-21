# Method-literature digest — agent-based simulation of human behaviour

Weekly sweep notes from the `agent-simulation-research` cron job. Newest first.
Completely separate from the Bowen corpus in `docs/theory/`.

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
