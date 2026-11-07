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
  (`experiments/llm_agents/runs/smoke-A1-s1/`). The smoke call's `sampling` wording corrected 2026-11-06 before the
  pilot (the client sends `2023-06-01`; the request unchanged), and the prices checked against the pricing page. The
  pilot LX.L2 run 2026-11-06 on the owner's go-ahead: A1, seed 1, 26 weeks, `complete`, 182 calls, 0 re-asks,
  fallbacks, retries or failures, $0.029851 (`experiments/llm_agents/runs/pilot-A1-s1/`); input mean 1,590 and max
  2,337 tokens, output 10; `estimate_output_tokens` restated 50 → 10, the main phase $5.97 against $9.97 left. Found
  at the pilot, for the owner: the model chooses by list position (six of seven members give the same option id
  almost every week, the first "hold back" option; options are numbered in a fixed order, `K1` first). The owner
  decided 2026-11-06: fix the option order first (a proposal for a per-person, per-round shuffle, for the owner's
  approval), then a second pilot, then the main phase. The proposal written 2026-11-06 as the plan's §13 (one
  shuffle per person per round over acts and holds, keyed by seed, week and person outside the engine's draws; ids
  the shuffled positions; the first pilot kept replayable through a recorded `option_order`; LX.21–LX.26; a position
  readout per arm; a second pilot the size of LX.L2, about $0.04), its seven questions answered by the owner the same
  day, each as recommended. §13 built 2026-11-06, no live call: `option_order` `shuffled` in `config/llm.md` and in
  the run record, the shuffle drawn outside the engine's draws, `_select` mapping the id through the order shown, the
  phase `pilot2`, the position readout in the report, tests LX.21–LX.26 each proved failing by its named mutation; the
  first pilot replays in the fixed order to its committed summary hash. Its readout: 173 same-option against 148
  same-id pairs of 175, position 1 at 0.011 against 0.036, every choice on `K1`: in the fixed order position and
  content are not told apart. The owner decided 2026-11-06 that §13.5's "effect gone" test uses the stronger rule
  (same-option above same-id, same-id not above chance, position 1 and the last position not above the mean of
  `1/N`; plan §13.5). The second pilot run 2026-11-06 on the owner's go-ahead: `pilot2-A1-s1`, `complete`, 182 calls,
  0 re-asks, fallbacks, retries or failures, $0.029782; every run replays `complete`, `pilot-A1-s1` at `d94482be4672`.
  Position 1 at 0.066 against 0.036, same id 7 against 6.4 by chance and 108 same-option pairs of 175, holds 0.951,
  `K1` at 0.033: the fixed-id habit and the `K1` preference are gone, a lean toward the top of the list is not, and
  the effect does not count as gone. The owner decided 2026-11-06 to proceed to the main phase as is, the lean to
  be reported and controlled for in the analysis (each choice's position is recorded), and recorded for later review
  as T-200. The owner approved the main phase's cost 2026-11-07 ($5.97 estimated, `main` pool, Haiku); it ran
  the same day: 20 runs `complete`, 29,115 calls, 0 re-asks, fallbacks, retries or failures, $5.087992 (the pool at
  $5.147736 of $10); every run replays. A position effect in 0 of 5 seeds of each arm; A3 follows labels over mechanics
  in 5 of 5 seeds; A4 keeps A2's direction for every pattern (distance 4 of 5); results in the report and plan §11
  step 7. Open: the cross-check, on the owner's go-ahead.
- [ ] T-200 ⟨later⟩ **The LLM agents' remaining lean toward the top of the option list (2026-11-06).** The second
  pilot (`pilot2-A1-s1`, plan §13.5), in the shuffled order, put 12 choices at position 1 against 6.6 expected by
  chance, and 32 in the first two positions against 13.3 (counted from `choices.jsonl.gz`; the report shows the
  first tenth of the list at 0.214). The stronger "effect gone" rule the owner adopted for §13.5 fails only on this
  lean (and on same id 7 against 6.4 by chance, less than one pair); the fixed-id habit and the `K1` preference are
  gone. The owner decided 2026-11-06 to proceed to the main phase as is: the order is shuffled, so the lean favours
  no option, and each choice's position is recorded, so the lean can be reported and controlled for in the analysis;
  this item is for later review. A gap: `report.py::position_readout` computes the approved rule (its "Position
  effect" column), not the stronger one, which is only in the plan's text. Possible fixes: say in the prompt that the
  order is random; show fewer options per call; or loosen the rule. The main phase (2026-11-07) showed a position
  effect in 0 of 5 seeds of each arm, position 1 between 0.026 and 0.049 against about 0.036, so the lean did not show
  at that scale; the stronger rule was not computed per run.
