# Plan: the LLM-agent experiment (version 3, part 2; T-183)

**Status: draft for approval. Nothing is built.** Written 2026-11-03 on `main` at `5c7fc9b` (spec revision 47, engine
hash `0b363cb3186c`). The owner decided on 2026-11-03 that version 3 has two new utilities: this experiment and an
emergence audit of the learning model (`docs/plan_v3_emergence_audit.md`, T-182). This plan settles what the experiment is, how it keeps the
theory out of the agents, its scale and cost, how a run replays, how results are compared, how calls are hardened,
and its acceptance criteria. §9 holds the decisions for the owner, each with a recommendation.

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
| A4 | LLM | A2's descriptions | **no** (no tension line, no history of tension) | contamination test |

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
  `deviation`, rounded), and whether contact is open, broken off by rejection (`severed`) or stopped (`cut_off`);
- what it believes about other ties, including ties it is not part of (`beliefs`, believed tension and contact);
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
content, so they go in `experiments/llm_agents/config/act_labels.md` (rule 9), not in code. **Their wording is for the
owner's review (Q2)** before any run.

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
- **Sampling.** Since API version `2026-10-21`, `temperature`, `top_p` and `top_k` return a 400 on every model. The
  sampling is therefore the model's default and cannot be seeded. The run record says so (LX.10).
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
  output) against the cap. If that would go over, it stops before the round. Every run then in progress is marked
  `stopped_at_budget`. The record gives weeks completed out of planned, the runs not started, and the money spent (P9:
  what was dropped is announced).
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
  connection errors, up to 5 attempts, with exponential backoff and jitter, capped at 60 s, honouring `retry-after`.
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
- **The run record** (`runs/<run_id>/run.json`) holds:
  - the requested model ID and the returned `model`;
  - the API version and SDK version;
  - the effort, the thinking setting and `max_tokens`;
  - sampling, recorded as "default; not settable";
  - the date and time;
  - the engine hash, the experiment's own code hash, the template hash and the label-file hash;
  - the arm, the seed and the permutations;
  - the status: `complete`, `stopped_at_budget`, `incomplete_transient`, `invalid` or `not_replayable`;
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
| LX.7 | A4 prompts carry no tension number or change | `test_prompts.py::test_lx7_no_feedback_arm_shows_no_tension` |
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

**Untested by the suite (live API only).** These are flagged rather than covered by a mock:

- **LX.L1.** The real API accepts the request shape: the schema enum, the thinking setting and effort. This is checked
  by one owner-approved smoke call (cost < $0.01) before the pilot, and recorded in the run record.
- **LX.L2.** The real model's outputs, and the token sizes behind the estimate in §5. These are checked in the pilot,
  and the estimate is restated from the pilot's `usage` before the main phase.

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

## 12. Documents

- **Added now:** this plan, listed in `docs/SPEC_REVIEW_CHECKLIST.md` and `docs/REVIEWER_READING_LIST.md` §7, and
  T-183 in `docs/todo/2026-11-03_version_3_llm_agent_experiment.md`, indexed in `TODO.md`.
- **Added at build:**
  - `experiments/llm_agents/README.md`;
  - its three config files;
  - `docs/llm_agent_experiment_report.md`, the generated report.

  The checklist covers `docs/` and `config/bowen/`. The build extends `tests/test_doc_checklist.py` to
  `experiments/llm_agents/config/`, so the experiment's editorial files are indexed too.
- **Not changed:** the spec (Q7), `docs/model_explainer.md` and `config/bowen/`. The experiment adds nothing to the
  model of record.
- **Adjacent issues found, not fixed:** none.
