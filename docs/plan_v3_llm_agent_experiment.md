# Plan: the LLM-agent experiment (version 3, part 2; T-183)

**Status: approved by the owner 2026-11-06, Q1–Q7 as recommended (§9). §11 steps 1–6 built 2026-11-06 under
`experiments/llm_agents/` (its README), with tests LX.1–LX.20 offline and the A0 baseline run. The owner approved the plain-label and prompt wording 2026-11-06 (Q2, §9), with A4 showing no tension number of
any kind (§2) and the build's three extras accepted. Step 7's smoke call (LX.L1) was made 2026-11-06 on the owner's go-ahead and
passed (§11), and the pilot (LX.L2) the same day: complete, $0.029851, the estimate restated from its usage (§11);
its choices showed a list-position effect, and the owner decided to fix the option order first: the fix (§13) was proposed and its seven questions answered 2026-11-06, each as recommended, and built the same day with tests LX.21–LX.26 (§10, §11), the first pilot still replaying; the second pilot (§13.5) was run 2026-11-06 on the owner's go-ahead, judged by the stronger rule the owner adopted the same day: complete, $0.029782, the K1 and same-id effect gone but position 1 still above the mean of 1/N, so the effect does not count as gone and the main phase waits for the owner (§13.5); the main phase and the cross-check each wait for the owner's go-ahead.** Written 2026-11-03 on `main` at `5c7fc9b` (spec revision 47, engine
hash `0b363cb3186c`). The owner decided on 2026-11-03 that version 3 has two new utilities: this experiment and an
emergence audit of the learning model (`docs/plan_v3_emergence_audit.md`, T-182). This plan settles what the experiment is, how it keeps the
theory out of the agents, its scale and cost, how a run replays, how results are compared, how calls are hardened,
and its acceptance criteria. §9 holds the seven decisions, each with its recommendation, and the owner's answers.

## 1. What it is, and what it is not

**The question.** Suppose each family member is an LLM agent that sees only what it feels and what happens to it. Do
conflict, distance, cutoff, over/underfunctioning, projection and fixed triangles appear as they do with the model's
learning agents, on the same readouts?

**It is a separate, small experiment.** It is not a replacement for the model of record. The model's agents
(`src/bowen/`, `M4.D.1`, `M4.D.6`) stay the model of record. Each person draws one act a week by a weighted random
draw, from values learned from felt relief that start equal. That is reproducible by seed and auditable by mutation,
and the two-arm acceptance tests depend on both. The four reasons given against LLM agents as the model of record
still stand, and this plan does not argue against them:

1. An LLM has read about families, therapy and Bowen theory, so it may reproduce what text says families do. That
   writes the pattern in through training (`CLAUDE.md`, "Patterns emerge; they are not written"). §3 is about this.
2. Bowen's emotional system is automatic, and an LLM deliberates in words. §4 turns the model's thinking off and asks
   for the choice alone. That narrows the gap and does not close it.
3. One criterion needs about 10 million decisions. This experiment makes about 30,000 (§5).
4. An LLM cannot be mutated to prove a mechanism. The experiment's findings are therefore **descriptive**. No
   finding of this experiment is evidence for or against a mechanism of the model.

**Where it lives.** It lives in `experiments/llm_agents/`, a sibling of `src/` and `tools/`. It reads the model's family
definitions, constants and readouts through their public functions. It never edits, and never writes under,
`src/bowen/` or `config/bowen/`. The engine hash (`tools/record_cache.py::engine_hash`, over `src/bowen` and
`config/bowen`) therefore does not move when the experiment is built or run.

**How it plugs in, with no engine change.** The engine takes its selections from a `Source`
(`src/bowen/engine/tick.py::Source`, `scheduled` and `selections`). The model's own source is
`src/bowen/scenario/policy_source.py::PolicySource`, which calls `observe` and then `decide` for each active person.
The experiment adds `LLMSource`, which implements the same protocol. For each active family member it calls `observe`
(`M4.B.2`, the person's own state, beliefs and inbox) and `legal_outcomes` (`M4.D.1e`, the legal set). It renders a
prompt from these, takes one option back, and builds the `Selection` the way `decide` does. Intensity comes from
`policy.force`, and a `WITHHOLD` names the act it holds back. Everything else in the tick is the model's own:
appraisal, felt change, consolidation, symptoms, the readouts. `M16.B` holds, because the engine still performs no
file I/O. The source is the caller's object, and its network and cache I/O happen in the experiment package, which
`src/bowen/` never imports.

**What the LLM replaces, and what it does not.** It replaces `decide` for the seven family members. That includes the
two-channel mix (`M4.D.1a`), the softmax over learned values, and competing urges (`M4.D.1d`, `urge` is 0). The
learner still runs. It registers only `DecidedBy.POLICY` selections that carry a value key (`learner.register_acts`).
`LLMSource` emits `DecidedBy.SCRIPTED` selections with no value key, so no learned value is written and none is read.
The external agent, Dr Halim, keeps the model's policy in every arm (Q3). `M4.D.3a`'s capacity gate is kept as an
*offer*: an act in layer `k` is offered to the agent with probability `availability(...)`, by a keyed draw with its own
purpose (`llm_offer`). This is the one place the LLM arm uses the draw service. Keyed draws do not shift each other
(`M3.D.4b`), so the learning-model arm is unaffected (LX.19).

## 2. Arms

All arms run the same family, horizon and engine seeds.

| Arm | Who decides | Act labels the agent sees | Felt change shown | Role |
|---|---|---|---|---|
| A0 | the model's policy (`PolicySource`) | none | n/a | the comparison |
| A1 | LLM | **opaque tokens**, permuted per seed (`K1`…`K16`), no description | yes | **primary** |
| A2 | LLM | **plain descriptions** with no theory vocabulary (§3.2) | yes | contamination probe |
| A3 | LLM | A2's descriptions **shuffled** onto other acts' mechanics (§3.3) | yes | contamination test |
| A4 | LLM | A2's descriptions | **no** (no tension line, no history of tension, no tie strain, no believed tension) | contamination test |

A1 is the closest analogue to the learning model. The learning agents know nothing about an act except what followed
it. With opaque labels, an LLM agent can only learn the same way, from what followed. A2–A4 exist to measure how much
of what an LLM agent does comes from what words mean to it rather than from consequences.

## 3. Keeping the theory out (the main risk)

### 3.1 What the agent is told, and what it is not

**Told** (system prompt, fixed for the run). It is one of several participants in a situation that repeats each round.
Each round it chooses exactly one option from a list. It feels a level of tension from 0 to 100. It wants to feel less
tension. It cannot see what the others feel. It answers with its choice only.

**Told** (each round, inside `<round>` tags, generated by the engine and never free text):

- the round number;
- its own tension (`acute_excess`, rounded) and the change since its last round (A4: neither);
- for each tie: the other person's token (`P1`…`Pn`, permuted per seed), how strained the tie feels to it (its own
  `deviation`, rounded; A4: not shown), and whether contact is open, broken off by rejection (`severed`) or stopped
  (`cut_off`);
- what it believes about other ties, including ties it is not part of (`beliefs`, believed tension and contact; A4:
  contact only);
- what reached it this round: "P4 did K7 toward you" or "you saw P2 do K3 toward P5" (`inbox`, target or witness);
- its own last `history_rounds` (12) rounds: what it chose and, after each, the change in its tension (A4: choices
  only);
- the options this round: one line per legal outcome, as label and target, plus `hold back from <option>` for
  `WITHHOLD` when an automatic act is offered.

**Not told:** names, sex, ages, generations, households, or relations (spouse, parent, child, sibling, professional);
that it is in a family; that anyone is a coach; the theory or any of its terms; the act kinds' real names; any other
person's tension or level; its own functional level or basic level (the offer gate carries capacity); its own learned
values (there are none); any goal except less tension; that the run is looking for patterns.

**The model's cross-person credit is not given.** The learner credits an act with the felt change of its target and
witnesses, weighted by bond (`M4.D.6e`). The agent sees only its own tension. That matches "sees only what it feels".
It is also a known difference from A0, and the report states it (§6).

### 3.2 Plain labels (A2–A4)

Each act has a short plain description: "seek more closeness with", "pull back from", "argue with", "take over a task
for", "let … take over a task for you", "turn to … about the other one", "stop all contact with", "say plainly where you
stand to", "keep in touch with, calmly", "step out of the middle between … and another", "decline to take sides with",
"reopen contact with", "stir things up with", and three for the moves toward an outside person. These are editorial
content, so they go in `experiments/llm_agents/config/act_labels.md` (rule 9), not in code. Their wording was approved by
the owner 2026-11-06, as drafted (Q2).

### 3.3 The shuffled arm (A3)

A3 shows A2's descriptions, each attached to another act's mechanics. The permutation is a fixed derangement per seed,
and it stays within groups that share a legality rule, so a label never shows up where its own act could not. The groups
are the tie acts (`PURSUE`, `DISTANCE`, `CONFLICT`, `OVERFUNCTION`, `UNDERFUNCTION`, `CUTOFF`, `I-POSITION`,
`STAY-IN-CONTACT`, `PROVOKE`), the triad acts (`TRIANGLE`, `DETRIANGLE`, `PREVENT_ALIGNMENT`) and the moves toward an
outside person (`SPLIT`, `FRAME_AMBIGUITY`, `DISPLACE`). `REDUCE_CUTOFF` is alone in its group and keeps its label.
`I-POSITION`'s two extra gates (a severed tie, `M5.C.1`'s dependence) can make its label rarer than its partner's. The
report states this.

### 3.4 How contamination is tested

| Test | What it measures | Read as |
|---|---|---|
| **Vocabulary lint** (LX.4) | Every prompt rendered in the test suite, every arm, every person, over a real-length run with a stub chooser, contains no term from `experiments/llm_agents/config/banned_terms.md`. That list covers the theory's terms, family and role words, the act kinds' names, the family file's names, and words such as "therapy" and "anxiety". | a gate: a build that fails it does not run |
| **Opaque arm leak** (LX.5) | No A1 prompt contains any plain description. | a gate |
| **Label-following** (A3 against A2) | Per seed: the rank correlation between A3's and A2's act frequencies **by label**, and the same **by mechanics**. | If by-label is larger in most seeds, the agent follows the words, not the consequences, so A2's patterns are counted as prior-driven. |
| **Feedback removed** (A4 against A2) | Per seed: whether each pattern's direction against A0 is the same in A4 as in A2. | Where removing felt change changes nothing, the pattern did not come from felt relief. |
| **Opaque against plain** (A1 against A2) | Which patterns appear in A2 and not in A1. | A pattern that needs the plain words to appear is counted as prior-driven. |

None of these tests has a threshold. Each is reported as a direction with the number of seeds in which it holds.

## 4. The model call

- **Model (Q4).** Recommended: `claude-haiku-5-5` for every arm, plus a cross-check with `claude-sonnet-5-5` on A1 and
  A2 with 2 seeds. Model IDs live in `experiments/llm_agents/config/llm.md`, never in code (L1). Before any run, each
  ID is checked against the Models API.
- **Thinking off, choice only.** This keeps the agent closer to an automatic response (reason 2 above).
  `claude-haiku-5-5` takes `thinking: {type: "disabled"}` at effort `low`. `claude-sonnet-5-5` takes
  `thinking: {type: "between_tools"}` (`{type: "disabled"}` is a 400 on it). `claude-opus-5-5` cannot turn thinking
  off at any effort, which is one reason it is not recommended (Q4).
- **Structured output.** `output_config.format` is a JSON schema `{"choice": <enum of this round's option ids>}`, so
  an option outside the legal set cannot come back. The parsed choice is still checked against the legal set (L4).
  `max_tokens` is 64, from config.
- **Sampling.** `temperature`, `top_p` and `top_k` are not sent. The client sends API version `2023-06-01` (the SDK's
  default, SDK 0.78.0), at which `claude-haiku-5-5` and `claude-sonnet-5-5` return a 400 for any non-default value, a
  rule of those models; at version `2026-10-21` any value is a 400 on every model. The sampling is therefore the
  model's default and cannot be seeded. The run record says so (LX.10). (Corrected 2026-11-06 before the pilot: the
  first wording gave `2026-10-21` as the reason, which is not the version the client sends; the request did not
  change. The smoke call's record keeps the first wording, as it was made.)
- **Prompt caching.** The system prompt is first and fixed. If it is shorter than the model's minimum cacheable prefix,
  it is not cached, and the estimate in §5 assumes it is not.
- **No batch API.** Each round depends on the last, so the batch API would mean 208 sequential batches. Instead, calls
  within a round are independent (`PolicySource` selects every person from the state at step 7) and run concurrently,
  up to `max_concurrency` from config.

## 5. Step, scale and cost

**The step is the engine's: one week.** The brief suggested monthly steps. A monthly decision would mean holding one
act for four weeks, which is a different policy from the one A0 runs, and the engine cannot tick monthly without a
change under `src/bowen/`. Recommended: weekly (Q3).

**Scale (Q3).** The family is `phase_c` (`src/bowen/ensemble/criteria.py::FAMILIES`). It has eight people: three
generations of family members plus Dr Halim. The horizon is 208 weeks, the ensemble's four-year horizon
(`config/bowen/criteria.md`). There are 5 engine seeds. That is 7 LLM calls a week, about 1,456 a run.

**Tokens per call (estimate, to be checked in the pilot).** Input is about 2,000 tokens: the system prompt about 350,
the state and ties about 300, beliefs about 150, the history about 200, and the options up to about 600 (about 60
options for the family member with the most ties). Output is about 50 tokens. Both are rounded up.

**Prices.** These come from the `claude-api` skill's model table (cached 2026-10-06), read on 2026-11-03. They go into
`config/llm.md` with that date and are checked against the pricing page before the first live run.

| Model | Input $/MTok | Output $/MTok |
|---|---|---|
| `claude-haiku-5-5` (prompts ≤ 100K) | 0.10 | 0.50 |
| `claude-sonnet-5-5` | 2.00 | 10.00 |
| `claude-opus-5-5` | 4.00 | 20.00 |

**Cost.**

| Phase | Runs | Calls | Per call | Total |
|---|---|---|---|---|
| Pilot: A1, 1 seed, 26 weeks, Haiku | 1 | ~182 | $0.000225 | < $0.05 |
| Main: A1–A4 × 5 seeds, Haiku | 20 | ~29,120 | $0.000225 | ~$6.60 |
| Cross-check: A1, A2 × 2 seeds, Sonnet | 4 | ~5,824 | $0.0045 | ~$26.20 |
| *(not recommended)* Main on Opus 5.5, thinking at `low` | 20 | ~29,120 | ≥ $0.009 before thinking tokens | ≥ $262 |
| A0, the model's policy | 5 | 0 | $0 | $0 |

Wall time: about 1–2 s a round with concurrent calls, so 5–7 min a run. The runs can go in parallel up to the rate
limit. Rate limits are not quoted here. They are read from each response's `anthropic-ratelimit-*` headers, and a 429
is retried after `retry-after` (§7).

**Hard budget cap (Q5).** Recommended caps: $10 for the pilot and main phase, $35 for the cross-check, $45 in all. They
live in `config/llm.md`. Cost is computed from each response's `usage` and the price table.

- **Before a phase:** the runner prints the estimate (calls × tokens × price). It refuses to start if the estimate is
  over the cap.
- **During a phase:** before each round it checks `spent + this round's worst case` (every call at `max_tokens`
  output, its input (text and output schema) at `worst_case_chars_per_token`, or at the costliest call seen so far if
  that is more, and every call counted twice for a re-ask; a call already in the cache counts nothing) against the
  cap. If that would go over, it stops before the round. Every run then in progress is marked
  `stopped_at_budget`. The record gives weeks completed out of planned, and the money spent; the phase's own record,
  `runs/phase-<name>.json`, gives the runs not started (P9: what was dropped is announced). As built: what was spent
  is summed from the response caches, every response paid for, so a run that crashed before its record was written
  still counts; and the estimate is checked against the total cap as well as the pool's.
- **A stopped run is not a result.** Its responses stay cached, so it can resume once the cap is raised. It is left out
  of the comparison, and the report says "N of M runs complete" for each arm.

## 6. How results are compared

**The same readouts, by the same functions.** For every run of every arm, the report computes:

- the `M11.G.1` evaluation, components 1–8 (`src/bowen/readouts/evaluation.py::evaluate`), with `M11.F`'s framing
  block (`M11.G.4`);
- each tie's and triad's pattern readings over time (`src/bowen/readouts/patterns.py::tie_readings`,
  `current_patterns`, `fixed_triangle`), by the definitions declared in `docs/readouts_phase_c.md` and
  `config/bowen/pattern_acts.md`.

**Layer 1 (this plan): directions, not thresholds.** For each pattern and each LLM arm X, the report gives:

- whether the pattern appears at all: in how many seeds at least one tie holds it, under the readout's own declared
  definition;
- the share of tie-weeks holding it, in X and in A0, with the spread over seeds;
- the direction of X against A0, with the number of seeds in which it holds;
- the contamination tests of §3.4.

No claim uses an absolute threshold. Every sentence of the report that quotes a number is generated by
`experiments/llm_agents/report.py` and checked current by a test (the `tools/c16_report.py` precedent).

**Layer 2 (not in this plan; Q6).** Composite `M11.C` criteria could be rerun with `LLMSource` in both of their arms,
to ask whether each asserted direction holds under LLM agents. Each criterion doubles the calls. The selection rule
would be: composite (`M11.5`), on `phase_c`, and with a readout that does not read the policy's learned values (so
`M11.C.42` and `M11.C.48` are out). Recommended: decide after layer 1's results, in a plan of its own.

**What the report never says.** It never says the model or the LLM speaks about a real family, or that an LLM agent
is a person (`M11.F.9`). It never says the experiment confirms or refutes a mechanism (§1, reason 4). It never treats
A0's results as ground truth that the LLM arms should match. Both are models. It states the differences in what each
arm's agents receive: A1–A4 get no cross-person credit, no two-channel mix, no competing urges, history in place of
learned values, and Halim on the policy.

## 7. Hardening

- **Layers (L9).** The prompt renderer is pure: no network, no file I/O. It is tested offline. The client layer is the
  only code that imports `anthropic`, and it imports it lazily, so the rest of the package and every test import
  without it.
- **Timeout, retry, backoff (E1, E5, P5).** The timeout is from config (60 s). The SDK's own retries are off
  (`max_retries=0`) so that every retry goes through one loop that logs it. That loop retries 429, 5xx, timeouts and
  connection errors, up to 5 attempts, with exponential backoff and jitter, capped at 60 s, honouring `retry-after` (up to `retry_after_max_seconds`, 300 s, as built).
  It does not retry 400, 401, 403 or 404: the run stops with the error, and the method, status and a response snippet
  are logged, never the key (E4, E7).
- **Transient is not terminal (P1).** If retries run out, the run is marked `incomplete_transient` and stops. No choice
  is recorded for that person that week. A rerun replays the cached rounds and continues from the first uncached call.
- **Failed parse (L4).** These are `stop_reason` `refusal` (recorded with its `stop_details.category`) and
  `max_tokens`, a choice missing from the response, or a choice outside the legal set. Each is `failed_to_parse`. There
  is one re-ask, which is cached separately. If the re-ask also fails, the week's selection is a `FALLBACK` hold,
  flagged `llm_failed_to_parse`. A run with more than `max_fallback_share` (1%) of its selections as such holds is
  marked `invalid` and left out of the comparison. The counts are reported for every run.
- **The key (S1).** It comes from `ANTHROPIC_API_KEY` or an `ant auth` profile, never from the repo, config or CI. Live
  calls need an explicit `--live` flag. Without it the runner only replays. A test runs `git grep` over the repo for
  key-shaped strings.
- **The dependency (S7).** `anthropic` is added to `requirements.txt` at an exact version, read from PyPI at build time.
  `tests/test_requirements.py::test_d3_requirements_match_the_imports` reads every tracked `.py` file, so the import
  has to be listed. CI installs it and never calls it.
- **Tests never call the API.** Every test injects a stub client. A test asserts that the live client is built only
  under `--live`, so `tests/` passes on a blank machine with no key and no network.

## 8. Reproducibility

- **The response cache.** Every response is stored under
  `sha256(model_id, arm, seed, person_token, week, attempt, canonical_json(request_body))`. The run key is part of the
  hash, so two runs that happen to render the same prompt never share an answer. Otherwise a cache hit would quietly
  remove the variation between seeds (P8, LX.9).
- **Storage (Q5).** Each run's cache is `experiments/llm_agents/runs/<run_id>/responses.jsonl.gz`, committed, so any
  machine can replay a run without a key. Each entry holds the key, the response text, `usage`, `stop_reason`, the
  returned `model`, the request id, and the time. Full prompts are not stored, because at about 10 KB each the cache
  would reach about 290 MB. Only their hashes are stored, plus every person's full prompt in weeks 0, 1 and the last
  week, for audit.
- **Replay.** `--replay` never calls the client. A missing key means the rendered prompt has changed: the engine,
  family, labels or template moved. The run is then marked `not_replayable` and stops, and nothing is filled in.
  As built: a change that leaves every prompt the same but moves what follows still hits the cache, so replay also
  compares the replayed summary with the committed one, and a difference marks the run `not_replayable`, naming the
  recorded hashes that moved. Only `complete` and `invalid` runs are replayed; a stopped or incomplete run never
  fetched its later weeks, and keeps its resumable status.
- **The run record** (`runs/<run_id>/run.json`) holds:
  - the requested model ID and the returned `model`;
  - the API version and SDK version;
  - the effort, the thinking setting and `max_tokens`;
  - sampling, recorded as "default; not settable";
  - the date and time;
  - the engine hash, the experiment's own code hash, the template hash and the label-file hash;
  - the arm, the seed and the permutations;
  - the order the options were numbered in, `option_order` (`fixed` or `shuffled`, §13); a record written before
    the field is read as `fixed`;
  - the status: `complete`, `stopped_at_budget`, `incomplete_transient`, `invalid`, `not_replayable` or
    `stopped_permanent` (a 400, 401, 403 or 404 stopped it, §7; named at build);
  - the counts: calls, re-asks, fallbacks, retries and cost.
- **Purity.** With a replayed `LLMSource`, the same seed with the log sink on and off reaches the same final state
  (`M16.B`, LX.18).

## 9. Decisions for the owner

| # | Decision | Recommended |
|---|---|---|
| Q1 | The arms and contamination tests (§2, §3.4) | A0–A4 as in §2, with A1 (opaque labels) as the primary arm |
| Q2 | What the agent is told (§3.1), and the wording of the plain labels (§3.2) | As §3.1: own tension and its change, ties by token, beliefs, inbox, 12 rounds of history, goal "less tension", capacity carried by the offer gate. The label wording is reviewed by the owner before any run. |
| Q3 | Step and scale (§5) | Weekly (the engine's tick), `phase_c`, 208 weeks, 5 seeds, Dr Halim on the model's policy in every arm |
| Q4 | The model (§4) | `claude-haiku-5-5` with thinking off for all arms, plus a `claude-sonnet-5-5` cross-check (A1, A2 × 2 seeds). `claude-opus-5-5` is not recommended: it cannot turn thinking off, and it costs at least 40 times as much. |
| Q5 | Budget cap and cache storage (§5, §8) | Caps of $10 (pilot and main) and $35 (cross-check), $45 in all. The response cache is committed, gzipped, per run. |
| Q6 | Comparison (§6) | Layer 1 only. Layer 2 (rerunning composite criteria) is decided after layer 1, in its own plan. |
| Q7 | Does the spec gain a version 3 section? | Not now. The experiment is outside the model of record, so this plan is its contract. If the owner later wants its results to stand as evidence about the model, a spec section is written then, with `M11.F.9` in full. |

**Decided 2026-11-06.** The owner chose every recommendation above, Q1–Q7, through AskUserQuestion. Each live phase of
§11 step 7 still waits for the owner's go-ahead.

**Q2's wording, decided 2026-11-06 after the build, through AskUserQuestion, each as recommended.** `act_labels.md` and
`prompt_text.md` are approved as drafted. A4 hides every tension number, not only the person's own: a tie's strain and a
pair's believed tension move after the agent's own acts, so showing them would show felt change (§2's intent); ties,
their contact state and believed contact stay, since contact is not felt tension. The build's three extras are accepted:
`prompt_text.md` and `report_text.md`, the status `stopped_permanent`, and the standard spells in every arm (LX.17).

## 10. Acceptance criteria and tests

Test files go under `tests/experiments/llm_agents/`. Every test name carries its ID. A gate criterion (LX.4–LX.7) is
proved failing by a mutation (a banned word placed in a template; a description placed in an A1 label) before it
counts.

| ID | Criterion | Test |
|---|---|---|
| LX.1 | Nothing under `src/bowen/` or `config/bowen/` imports or names `experiments/`, and the experiment writes only under `experiments/llm_agents/runs/` and its report | `test_isolation.py::test_lx1_engine_never_imports_the_experiment`, `::test_lx1_writes_only_under_runs` |
| LX.2 | `LLMSource` implements `Source`, gives one selection per active living family member a week, and gives Halim the model's policy | `test_source.py::test_lx2_one_selection_per_active_member`, `::test_lx2_external_agent_keeps_the_policy` |
| LX.3 | Every chosen option is in `legal_outcomes`, and one outside it is `failed_to_parse` | `test_source.py::test_lx3_choice_is_always_legal`, `::test_lx3_illegal_choice_fails_to_parse` |
| LX.4 | No rendered prompt in any arm contains a banned term | `test_prompts.py::test_lx4_no_banned_term_in_any_arm` (real-length run, stub chooser) and `::test_lx4_a_banned_term_is_caught` (adversarial, P7) |
| LX.5 | A1 prompts carry no plain description, and their labels are permuted per seed | `test_prompts.py::test_lx5_opaque_arm_has_no_description`, `::test_lx5_labels_differ_between_seeds` |
| LX.6 | A3's permutation is a derangement inside each legality group | `test_prompts.py::test_lx6_shuffle_is_a_derangement_within_groups` |
| LX.7 | A4 prompts carry no tension number or change: not the person's own tension, its change, a past round's change, a tie's strain or a pair's believed tension; contact stays | `test_prompts.py::test_lx7_no_feedback_arm_shows_no_tension`, `::test_lx7_render_without_feedback_drops_every_number_of_tension` |
| LX.8 | Replay never calls the client, and a cache miss under replay marks the run `not_replayable` | `test_cache.py::test_lx8_replay_never_calls`, `::test_lx8_miss_is_not_replayable` |
| LX.9 | Two runs with an identical prompt keep separate entries (dirty state, P8) | `test_cache.py::test_lx9_run_key_separates_entries` |
| LX.10 | The run record has every field of §8 | `test_record.py::test_lx10_run_record_fields` |
| LX.11 | The estimate refuses a phase over the cap, and a run stops before the round that would cross it, announcing what it dropped (real-scale fixture where the cap bites, P9) | `test_budget.py::test_lx11_estimate_over_cap_refuses`, `::test_lx11_stops_before_crossing` |
| LX.12 | 429, 5xx, timeouts and connection errors are retried with logged backoff. When retries run out, the run is `incomplete_transient`, with no choice recorded, and it resumes from the cache. | `test_client.py::test_lx12_transient_is_retried`, `::test_lx12_exhausted_is_not_a_result`, `::test_lx12_resumes_from_cache` |
| LX.13 | 400, 401, 403 and 404 are not retried, and they stop the run | `test_client.py::test_lx13_permanent_is_not_retried` |
| LX.14 | A refusal, `max_tokens`, or an unparseable response gets one re-ask and then a flagged fallback. A run over the fallback share is `invalid`, and the counts are reported. | `test_client.py::test_lx14_failed_parse_path` |
| LX.15 | No key in the repo; the live client is built only under `--live` | `test_isolation.py::test_lx15_no_key_in_repo`, `::test_lx15_live_client_only_when_asked` |
| LX.16 | The report is generated and current. It carries the framing block, uses no absolute threshold, and says "N of M complete" for each arm. | `test_report.py::test_lx16_report_is_current`, `::test_lx16_framing_and_counts` |
| LX.17 | A0 run through the experiment reaches the same final state as the ensemble's own assembly for that seed | `test_source.py::test_lx17_a0_matches_the_ensemble` |
| LX.18 | A replayed LLM run reaches the same final state with logging on and off | `test_source.py::test_lx18_logging_is_a_pure_observer` |
| LX.19 | The offer gate's draw has its own purpose and changes no other draw | `test_source.py::test_lx19_offer_draw_is_isolated` |
| LX.20 | The report never names a real family or presents an agent as a person (`M11.F.9`) | `test_report.py::test_lx20_framing_block_present` |
| LX.21 | The option order differs across rounds and seeds for one person, and the same `(seed, week, person)` gives the same order in every arm and every process (§13.1) | `test_prompts.py::test_lx21_order_differs_across_rounds_and_seeds`, `::test_lx21_order_is_shared_across_arms` |
| LX.22 | The shuffle requests no engine draw, and a chooser that picks by option text reaches the same final state and choices under `shuffled` and `fixed` | `test_source.py::test_lx22_order_requests_no_engine_draw`, `::test_lx22_order_changes_no_outcome` |
| LX.23 | Every option is offered exactly once, and acts and holds interleave | `test_prompts.py::test_lx23_every_option_offered_once`, `::test_lx23_acts_and_holds_interleave` |
| LX.24 | Answering id `i` selects the option shown as `i`, in every round of a real-length stub run | `test_source.py::test_lx24_id_maps_to_the_option_shown` |
| LX.25 | The run record carries `option_order`; a record without it replays as `fixed`, and the committed `pilot-A1-s1` replays `complete` with its committed summary hash | `test_record.py::test_lx25_record_carries_option_order`, `test_cache.py::test_lx25_committed_pilot_replays_fixed` |
| LX.26 | The report's position readout (§13.4) is generated per LLM arm and tells position from content (adversarial, P7) | `test_report.py::test_lx26_position_readout_per_arm`, `::test_lx26_position_and_content_choosers_differ` |

LX.21–LX.26 were each proved failing by the mutation §13.3 names, 2026-11-06: one shuffle key on the seed only, the
arm added to the key, an engine draw under its own purpose taken each round, the last option dropped, acts and holds
shuffled as two blocks, `_select` indexing the fixed order, a missing `option_order` read as `shuffled`, and both
repeat kinds counted by id. Each failed its named test by an assertion, and the code was restored.

**Untested by the suite (live API only).** These are flagged rather than covered by a mock:

- **LX.L1.** The real API accepts the request shape: the schema enum, the thinking setting and effort. This is checked
  by one owner-approved smoke call (cost < $0.01) before the pilot, and recorded in the run record. Checked
  2026-11-06: accepted, parsed, $0.000113 (§11 step 7).
- **LX.L2.** The real model's outputs, and the token sizes behind the estimate in §5. These are checked in the pilot,
  and the estimate is restated from the pilot's `usage` before the main phase. Checked 2026-11-06: the pilot
  completed; input mean 1,590 and max 2,337 tokens, output 10 on every call; the estimate restated (§11 step 7). The
  outputs show a list-position effect, reported to the owner (§11 step 7). The second pilot (2026-11-06, shuffled
  order, §13.5): complete, 182 calls, $0.029782; the effect does not count as gone, reported to the owner.

## 11. Order of work

1. Config: `llm.md`, `act_labels.md`, `banned_terms.md`. Then the pure renderer with LX.4–LX.7.
2. `LLMSource` with a stub chooser: LX.2, LX.3, LX.17, LX.18, LX.19, LX.1.
3. Client layer: the cache, retries, the budget and the run record. LX.8–LX.15.
4. The runner CLI: arms, seeds, `--live`, `--replay`, the estimate.
5. `report.py`: LX.16, LX.20.
6. Reviews: one `learning-qa` pass and one correctness pass over the diff, with `security-review` for the client layer.
   Then commit, with the suite green. No live call has been made up to this step.
7. Live, each step only on the owner's go-ahead: the smoke call (LX.L1), then the pilot (LX.L2), then the main phase,
   then the cross-check. Each phase is committed with its cache, its run records and the regenerated report.
   **The smoke call, 2026-11-06:** one call to `claude-haiku-5-5` with `thinking: disabled` at effort `low` and the
   schema enum was accepted (HTTP 200, `stop_reason` `end_turn`), its answer `{"choice": "19"}` parsed to a legal
   option, no retry and no re-ask, and its cost, $0.000113 (1,082 input and 10 output tokens), was logged against the
   `main` pool. Both model IDs were found by the Models API first. The record and the cache are
   `experiments/llm_agents/runs/smoke-A1-s1/`. LX.L1 holds.
   **The pilot, 2026-11-06:** before it, the prices in `config/llm.md` were checked against the pricing page (they
   match) and §4's sampling wording was corrected (no change to the request). One live run, A1, seed 1, 26 weeks,
   on `claude-haiku-5-5`: `complete`, 182 calls, all `end_turn`, 0 re-asks, 0 fallbacks (share 0.0 against 0.01), 0
   retries, 0 failures, $0.029851 against the estimate's $0.041 (`experiments/llm_agents/runs/pilot-A1-s1/`). Input
   per call: mean 1,590, max 2,337, level at about 1,700 after week 15; output 10 on every call. Sized offline for
   208 weeks with the stub chooser and scaled by the pilot's tokens per character, the four arms' means are about
   1,716 (A1), 2,068 (A2), 2,128 (A3) and 1,742 (A4). Restated: the per-call estimate is 2,000 input (unchanged) and
   10 output tokens (was 50), $0.000205 a call on Haiku; the main phase is 29,120 calls, about $5.97, against the
   main pool's $9.97 left; the cross-check 5,824 calls, about $23.88, against $35 (its output not yet measured). §3:
   the lint finds no banned term in the 21 stored prompts. The choices show a list-position effect: six of the seven
   members give the same option id almost every week, the first "hold back from" option, and two chose option 1 (the
   first act, `K1`, which in seed 1 is `CUTOFF`) in week 0. The option order is fixed (`prompts.py::ordered_options`).
   No prompt was changed; whether to change the order before the main phase is the owner's decision. LX.L2 holds for
   the token sizes.
   **The option order, 2026-11-06:** the owner decided to shuffle the options before the main phase and run a second
   pilot; the fix is §13, its questions answered as recommended. **Built 2026-11-06** (no live call): `option_order`
   in `llm.md` (`shuffled`) and in the run record; the shuffle in `prompts.py::arranged`, drawn by
   `arms.py::option_rng`; the prompt carries the numbered options' mechanics (`Prompt.specs`), and `_select` maps the
   id through them; the phase `pilot2`; each LLM run's choices with their positions in `choices.jsonl.gz`, and the
   position readout in the report; tests LX.21–LX.26 (§10). `runner replay` replays `pilot-A1-s1` in the fixed order
   to its committed summary hash `d94482be4672` and wrote its `choices.jsonl.gz`. The readout of that run: 182 choices,
   position 1 at 0.011 against a mean 1/N of 0.036, 148 same-id and 173 same-option pairs of 175, holds 0.989, every
   choice on token `K1`, 0.978 of choices in the seventh tenth of the list. In the fixed order the same option keeps
   nearly the same place, so this does not tell position from content; the second pilot does.
   **The second pilot, 2026-11-06:** on the owner's go-ahead, judged by the stronger rule the owner adopted before the
   run (§13.5). `pilot2-A1-s1`: `complete`, 182 calls, 0 re-asks, fallbacks, retries or failures, $0.029782; every run
   replays `complete`, `pilot-A1-s1` at its summary hash `d94482be4672`. Position 1 at 0.066 against 0.036, same id 7
   against 6.4 by chance and 108 same-option of 175 pairs, holds 0.951, K1 at 0.033. The `K1` preference and the
   fixed-id habit are gone; a lean toward the top of the list is not, so the effect does not count as gone (§13.5) and
   the main phase waits for the owner's decision.

## 12. Documents

- **Added now:** this plan, listed in `docs/SPEC_REVIEW_CHECKLIST.md` and `docs/REVIEWER_READING_LIST.md` §7, and
  T-183 in `docs/todo/2026-11-03_version_3_llm_agent_experiment.md`, indexed in `TODO.md`.
- **Added at build:**
  - `experiments/llm_agents/README.md`;
  - its config files: the plan's three, and `prompt_text.md` and `report_text.md`, added at build so that no
    prompt or report wording is in code (rule 9);
  - `docs/llm_agent_experiment_report.md`, the generated report.

  The checklist covers `docs/` and `config/bowen/`. The build extends `tests/test_doc_checklist.py` to
  `experiments/llm_agents/config/`, so the experiment's editorial files are indexed too.
- **Not changed:** the spec (Q7), `docs/model_explainer.md` and `config/bowen/`. The experiment adds nothing to the
  model of record.
- **Adjacent issues found, not fixed:** during an I-POSITION sequence the engine makes a family member's act toward
  the sequence's other as `STAY-IN-CONTACT` (`src/bowen/engine/iposition.py`, `M5.D.9`), so in a few rounds the act
  that runs is not the one the agent chose, and the prompt does not say so. Found 2026-11-06 by LX.24's record check;
  the same in either option order; engine behaviour, unchanged here.

## 13. The option order (proposed and decided 2026-11-06, built 2026-11-06)

**Why.** The pilot (LX.L2, §11 step 7) showed the model choosing by list position. `prompts.py::ordered_options`
numbers the options in a fixed order: acts by target and label, then every `hold back from` line. Six of the seven
members gave the same id almost every week, and it was the first hold; P2 and P4 chose option 1, the first act, in
week 0. Of 175 week-to-week pairs, 148 repeat the same id; 9 distinct ids in 182 choices. With a fixed order the
same id and the same option cannot be told apart, so the pilot cannot say how much is position and how much is
content. In seed 1 the first hold is always `hold back from: K1 …`, so the token `K1` is confounded with position
too. The owner decided 2026-11-06 to fix the order before the main phase, then run a second pilot. This section is
the fix; the owner's answers to its seven questions, each taken as recommended, are in 13.6.

### 13.1 The shuffle

- **One shuffle per person per round**, over acts and holds together (interleaved, not two blocks), so neither kind
  has a fixed first line.
- **Presorted by mechanics first** (kind, target, third, hold), not by the arm's label text, then shuffled. The draw
  is keyed by `(seed, week, person token)` and not by the arm, so two arms in the same state show the same order, and
  position is not a difference between arms (A3 against A2 stays paired on position).
- **The draw** is `arms.rng`'s kind of generator, `sha256("experiments.llm_agents|option_order|seed|week|token")`,
  outside the engine's draw service. It requests no engine draw, so it moves none, as LX.19 holds for `llm_offer`.
- **Ids are the shuffled positions**, `1`…`N`. The schema's enum, the parse against the legal ids and the answer
  line are unchanged; only the id-to-option map changes each round. `_select` maps the id through the same shuffled
  order `render` numbered, never through `ordered_options`' fixed order. The history lines show the option's text,
  not its id, so they are unaffected.
- **Config, not code** (rule 8): `option_order` in `experiments/llm_agents/config/llm.md`, `shuffled` from the
  second pilot on. `fixed` remains a value, for replaying the first pilot (13.2). The run record carries it.
- **A1's tokens stay** `K1`…`K16`, permuted per seed. The second pilot reports the choice share by token number; a
  `K1` preference that survives the shuffle goes back to the owner.

### 13.2 Replay and the cache

The shuffled prompt differs from the fixed one, so every new run has new prompt hashes and new cache keys; the run id
is in the key too, so the second pilot (`pilot2-A1-s1`) never reads the first pilot's entries. The first pilot is
kept replayable, not retired: a run record without `option_order` is read as `fixed`, `runner replay` renders that
run in the fixed order, its 182 keys hit, and replay's existing summary comparison (§8) must find the committed
summary hash. The record's `code_hash` moves (the experiment's code changed) and stays as it was written; replay
already judges by the summary, not by that hash. The engine hash does not move: nothing under `src/bowen/` or
`config/bowen/` changes. The smoke record is skipped by replay, as now.

### 13.3 Acceptance criteria

Each test is proved failing by the named mutation before it counts. Built 2026-11-06 and in §10, each proved failing by
its mutation.

| ID | Criterion | Test | Mutation that must fail it |
|---|---|---|---|
| LX.21 | The order differs across rounds and across seeds for one person, and the same `(seed, week, person)` gives the same order in every arm and every process | `test_prompts.py::test_lx21_order_differs_across_rounds_and_seeds`, `::test_lx21_order_is_shared_across_arms` | key the draw on the seed only (one order every week); add the arm to the key |
| LX.22 | The shuffle requests no engine draw: over a real-length stub run, the set of draw keys asked of `state.draws` is the same with `shuffled` and `fixed`, and a chooser that picks by option text reaches the same final state and choices under both | `test_source.py::test_lx22_order_requests_no_engine_draw`, `::test_lx22_order_changes_no_outcome` | draw the order from `state.draws` under any purpose |
| LX.23 | Every option is offered exactly once (the rendered list is a permutation of the legal options), and acts and holds interleave: over a real-length run some round has a hold before an act | `test_prompts.py::test_lx23_every_option_offered_once`, `::test_lx23_acts_and_holds_interleave` | drop the last option after the shuffle; shuffle acts and holds as two blocks |
| LX.24 | Answering id `i` selects the option whose text is shown as `i`, in every round of a real-length stub run | `test_source.py::test_lx24_id_maps_to_the_option_shown` | `_select` indexes the fixed `ordered_options` order |
| LX.25 | The run record carries `option_order` (LX.10's field list grows); a record without it replays as `fixed`, and the committed `pilot-A1-s1` replays `complete` with its committed summary hash | `test_record.py::test_lx25_record_carries_option_order`, `test_cache.py::test_lx25_committed_pilot_replays_fixed` | read a missing field as `shuffled` |
| LX.26 | The report's position readout (13.4) is generated per LLM arm, and it tells position from content: a stub that always answers `1` shows same-id repeats above same-option repeats and a position-1 share above the mean of `1/N`; a stub that picks by content shows neither (adversarial, P7) | `test_report.py::test_lx26_position_readout_per_arm`, `::test_lx26_position_and_content_choosers_differ` | count both repeat kinds by id |

### 13.4 The position readout in the report

Per LLM arm and seed, generated by `report.py`, a direction with no threshold (§6): the share of choices at
position 1 against the mean of `1/N` over the rounds; the chosen position as a share of the list, in tenths; the
week-to-week pairs that repeat the same id against those that repeat the same option (by mechanics); the share of
holds; and for A1 the share by token number. A position effect reads as same-id repeats above same-option repeats
and a position-1 share above `1/N`. The report says in how many seeds of each arm it holds.

As built: the readout is read from each complete LLM run's `runs/<run_id>/choices.jsonl.gz` (one line per family
member's week: the id, the list's length, the option's mechanics and, in A1, its act token), kept out of the summary
so no summary hash moves. The report gives one row per run, the pilots included, and counts the seeds with a position
effect over the main phase's runs of each arm. Pairs are one person's consecutive weeks, both with a choice.

**For the owner, before the second pilot (from the build's review, 2026-11-06).** The rule above has no chance
baseline and looks only at position 1. Run offline on 26-week A1 stub runs: a chooser that picks at random scored
"effect" in 3 of 8 runs (same id 9 against same option 2, position 1 at 0.044 against 0.036), and a chooser that
always answers the last option scored "no effect" with 152 same-id pairs of 175. The readout therefore also shows,
as context the rule does not use, the same-id pairs a position-blind choice would give by chance (the sum of `1/N`
over the pairs), the share at the last position, and the flagged weeks. A stronger rule would compare same-id
repeats with that chance figure and look at any fixed position, not only the first; whether to adopt it for §13.5's
"effect gone" test was the owner's decision. **The owner adopted it for §13.5, 2026-11-06** (13.5). The report's
"Position effect" column, which the main-phase seed count uses, still applies the rule as approved above.

### 13.5 The second pilot

The same as LX.L2: A1, seed 1, 26 weeks, `claude-haiku-5-5`, 182 calls, about $0.04 (182 × $0.000205), from the
`main` pool's $9.97; the main phase's $5.97 then leaves about $3.96. Run only on the owner's go-ahead, after the
build and its review. **The effect counts as gone if**, over its 175 week-to-week pairs, same-option repeats
outnumber same-id repeats, and the position-1 share is not above the mean of `1/N`. A preference for holds, or for
one act, that persists under the shuffle is a content preference, not a position effect: it is reported, not
counted against the fix. If the effect is not gone, the main phase waits and the result goes to the owner.

**The stronger rule (the owner, 2026-11-06, before the run).** The effect counts as gone only if all four hold:
same-option repeats above same-id repeats; same-id repeats not above the same-id repeats a position-blind choice
would give by chance (the sum of `1/N` over the pairs); the position-1 share not above the mean of `1/N`; and the
last-position share not above the mean of `1/N`. It is stricter than the approved rule, so a near-random result can
read "not gone" and go to the owner, never the reverse.

**The second pilot, 2026-11-06, on the owner's go-ahead.** `pilot2-A1-s1`, `option_order` `shuffled`: `complete`, 182
calls, 0 re-asks, fallbacks, retries or failures, $0.029782 against the estimate's $0.0373
(`experiments/llm_agents/runs/pilot2-A1-s1/`). From the report's position table: position 1 at 0.066 against a mean
`1/N` of 0.036 (12 choices against 6.6 by chance), the last position at 0.022; same id 7 against 6.4 by chance and
108 same-option pairs of 175; holds 0.951; by act token K14 0.280, K10 0.187, K16 0.176, K4 0.170, K11 0.088, K1
0.033, the rest 0.022 or less. The report's own "Position effect" column reads "no". Against the first pilot (148
same-id pairs, every choice on `K1`), the fixed-id habit and the `K1` preference are gone, and the choices follow
content. **The effect does not count as gone**, under either rule: position 1 is above the mean of `1/N` (both rules),
and same id is above chance, 7 against 6.4 (the stronger rule; a difference of less than one pair). A lean toward the
top of the list remains: the first tenth of the list takes 0.214 of choices and the second 0.143 (report), and, counted
offline from `choices.jsonl.gz` and not in the report, the first two positions take 32 choices against 13.3 by chance.
The main phase waits; the result goes to the owner. The hold share, 0.951, is a content preference and is reported,
not counted against the fix.

### 13.6 The owner's answers (2026-11-06, each as recommended)

1. Layout: one shuffle over acts and holds together.
2. Ids: the shuffled positions `1`…`N`.
3. The first pilot: kept replayable through a recorded `option_order`, a missing field read as `fixed`.
4. The position readout: in the report, per LLM arm.
5. The draw key: `(seed, week, person token)`, not the arm.
6. A1's tokens: kept; the second pilot measures a `K1` preference.
7. The second pilot: the same size as LX.L2.
