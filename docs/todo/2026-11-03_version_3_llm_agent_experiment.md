# Version 3, part 2 — the LLM-agent experiment, 2026-11-03

Items from planning the LLM-agent experiment (`docs/plan_v3_llm_agent_experiment.md`), the second of the two new
utilities the owner decided version 3 has (2026-11-03).

- [ ] T-183 ⟨later⟩ **The LLM-agent experiment: build the approved plan (2026-11-03; approved 2026-11-06).** The
  plan's seven decisions (§9, Q1–Q7) on the arms and contamination tests, what the agent is told and the plain-label
  wording, step and scale, the model, the budget cap and cache storage, the comparison, and whether the spec gains a
  version 3 section were decided by the owner on 2026-11-06, each as recommended. §11 steps 1–6 built 2026-11-06
  under `experiments/llm_agents/`, tested offline (`tests/experiments/llm_agents/`, LX.1–LX.20), reviewed, and the
  A0 baseline run (`docs/llm_agent_experiment_report.md`). The plain-label and prompt
  wording (`experiments/llm_agents/config/act_labels.md`, `prompt_text.md`, Q2) approved by the owner 2026-11-06, with
  A4 hiding a tie's strain and a pair's believed tension as well as the person's own tension (built the same day,
  LX.7 widened), and the build's three extras accepted. Step 7's smoke call LX.L1 made 2026-11-06 on the
  owner's go-ahead: one call to `claude-haiku-5-5`, `end_turn`, the choice parsed, $0.000113 against the $10 pool
  (`experiments/llm_agents/runs/smoke-A1-s1/`). Open: the pilot LX.L2, the main phase and the cross-check, each on
  the owner's go-ahead. Found at the smoke call, not fixed: the record's `sampling` line cites API version
  `2026-10-21`, but the request went out at the SDK's `anthropic-version` `2023-06-01`; check the claim before the
  pilot.
