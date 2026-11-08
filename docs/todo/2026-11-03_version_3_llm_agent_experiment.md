# Version 3, part 2 — the LLM-agent experiment, 2026-11-03

Items from planning the LLM-agent experiment (`docs/plan_v3_llm_agent_experiment.md`), the second of the two new
utilities the owner decided version 3 has (2026-11-03).

- [x] T-183 ⟨later⟩ **The LLM-agent experiment: build the approved plan (2026-11-03; approved 2026-11-06).** The
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
  step 7. The owner chose `claude-sonnet-5-5` for the cross-check and approved its cost 2026-11-07 ($23.88
  estimated, `cross_check` pool); it ran the same day: 4 runs `complete`, 5,820 calls, 0 re-asks, fallbacks, retries
  or failures, 10 output tokens a call, $20.526398 (the pool at $20.526398 of $35, both pools $25.674134 of $45); every
  run replays. For A1 and A2 the direction against A0 is the same as the main phase's on the same seed for every
  pattern and readout in 2 of 2 seeds, except A2's distance (1 of 2); no position effect in the 4 runs; A1 holds back
  less on the second model (0.446, 0.635 against 0.916 to 0.965). The report's cross-check table (LX.27) is Claude's
  addition, pending the owner's review.
  **Closed 2026-11-07 by the owner, through `AskUserQuestion`, every question as recommended**, at the closing review
  (`docs/version_3_llm_experiment_closing_review.md`, no live call): T-183 closed at layer 1; no layer 2 (plan §6); no
  other model family now, revisited only if T-201 shows A1 responds to tension; LX.27 approved; T-200 closed with no
  change; T-201 kept open, its no-cost check run. Spend at close: $25.674134 of $45.
- [x] T-200 ⟨later⟩ **The LLM agents' remaining lean toward the top of the option list (2026-11-06).** The second
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
  **Closed 2026-11-07 by the owner, no change** (closing review §4.1): under the approved rule no position effect in
  any of the 24 main and cross-check runs; counted offline from the report's table, the stronger rule reads "not gone"
  in 21 of the 24 by small margins in both directions, since it has four strict conditions and no tolerance; it was
  written for the 182-call pilot and is not used on full-length runs. The report keeps the approved rule only.
- [ ] T-201 ⟨later⟩ **Whether A1's holding back responds to tension, or to the words "hold back" (2026-11-07).** In the
  main phase A1, the opaque arm and the only one where felt tension is the agent's sole guide, chose `hold back` in
  about 92–97% of its choices (the pilots: 0.989 in the fixed order, 0.951 shuffled); A2–A4 almost never did. A1's
  act labels are opaque (`K1`…`K16`), but its withhold option is rendered in plain English, `hold back from <option>`
  (plan §3.1), so the model may choose it because the words read as the safe choice, not because holding back lowered
  its tension. No arm separates the two: A4 hides tension for A2's labels only, and there is no A1 without tension.
  This bears on what A1 can show about emotional process (owner's question, 2026-11-07). Checks: (1) A1's hold share
  by week over the 208 weeks and against its tension history, from the committed `choices.jsonl.gz` (no live call;
  learning from relief would show a trend); (2) if needed, an A1 run with tension hidden, or with the withhold option
  given an opaque token, which needs a plan change and the owner's cost go-ahead.
  **New evidence and check (1), 2026-11-07, at T-183's closing review** (§4.2 there). On `claude-sonnet-5-5`, A1 holds
  back in 0.446 and 0.635 of its choices, and its choices fall on act token `K1` in 0.990 and 0.855 of them (report),
  though `K1` stands for a different act in each seed: a preference for the token's name. Check (1), run on the
  owner's go-ahead, offline, from the committed `choices.jsonl.gz`: on Haiku A1 holds back in 23 or 24 of its first 28
  choices in every seed, and its hold share by 52-week block stays level (0.893 to 0.973); on Sonnet it starts near half
  and moves with no shared direction over the two seeds; in every run the persons split between a higher and a lower
  later half. No run shows holding back growing with experience. This fits a starting preference for the option's
  text more than a learned response to tension; a week-by-week response to tension is not ruled out. Open: the tension
  per week (an offline replay from the cache), and check (2).
