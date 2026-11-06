# LLM-agent experiment (T-183)

The contract is `docs/plan_v3_llm_agent_experiment.md`. This directory is outside the model of record: nothing under
`src/bowen/` or `config/bowen/` names it, it changes no engine rule, and the engine hash does not move with it (LX.1).
The report it generates is `docs/llm_agent_experiment_report.md`.

**Status (2026-11-06).** Plan steps 1–6 are built and tested offline. No live call has been made. A0 (the model's own
policy, no call, no cost) has been run for its five seeds and is in the report. The wording in `config/act_labels.md`
and `config/prompt_text.md` was approved by the owner 2026-11-06 (Q2); arm A4 shows no tension number of any kind (no
own tension or change, no tie strain, no believed tension). Each live phase (the smoke call LX.L1, the pilot
LX.L2, the main phase, the cross-check) waits for the owner's go-ahead.

## What it does

Each week of a `phase_c` run, every living family member's outcome is chosen by a language model instead of the
policy. The model is shown only its own tension, its ties by token, its beliefs, what reached it, its last rounds and
a numbered list of options. The options are the policy's own legal set (`policy.legal_outcomes`), narrowed by the
offer gate (`policy.availability`, one keyed draw per kind under the purpose `llm_offer`). The external agent stays on
the policy. Arms A0–A4 are in the plan's §2.

## Modules

| Module | What it is |
|---|---|
| `config.py` | Reads the five config files strictly; an unknown or missing entry is refused |
| `arms.py` | The arms, and each seed's person tokens, opaque act tokens and A3's derangement |
| `prompts.py` | The pure renderer (no network, no file I/O), and the banned-term lint |
| `source.py` | `LLMSource`, the engine `Source` for A1–A4; `build` assembles a run as the ensemble does |
| `client.py` | The request, the cache lookup, the one retry loop, the parse and the one re-ask; the only module that imports `anthropic`, lazily |
| `cache.py` | The response cache, one gzipped JSON-lines file per run |
| `budget.py` | The estimate before a phase, and the worst case before each round |
| `record.py` | `run.json`, with every field of the plan's §8 |
| `summary.py` | A run's summary, by the model's own readout functions |
| `report.py` | Renders the report from the complete runs' summaries |
| `runner.py` | The command line |
| `files.py` | The only place the package writes: `runs/` and the report |

## Config (editorial content, rule 9)

| File | What it holds | Status |
|---|---|---|
| `config/llm.md` | Models, thinking and effort, prices, retries, caps, scale, phases | decided by the owner 2026-11-06 |
| `config/act_labels.md` | Each act's plain label and what others see (A2–A4) | approved by the owner 2026-11-06 (Q2) |
| `config/prompt_text.md` | Every fixed word of the prompt | approved by the owner 2026-11-06 (Q2) |
| `config/banned_terms.md` | The lint's terms; the family file's names are added at check time | built |
| `config/report_text.md` | The report's fixed sentences | built |

The plan named three config files. `prompt_text.md` and `report_text.md` were added at build so that no prompt or
report wording lives in code; the owner accepted both 2026-11-06.

## Commands (from the repo root)

```
python3 -m experiments.llm_agents.runner estimate main    # calls × tokens × price, against the cap
python3 -m experiments.llm_agents.runner run baseline     # A0: no call, no cost
python3 -m experiments.llm_agents.runner smoke --live     # LX.L1, one call: only on the owner's go-ahead
python3 -m experiments.llm_agents.runner run pilot --live # LX.L2: only on the owner's go-ahead
python3 -m experiments.llm_agents.runner replay           # every committed run, from its cache; never calls
python3 -m experiments.llm_agents.runner report           # regenerate docs/llm_agent_experiment_report.md
```

Without `--live` a phase that calls a model is refused. The key is the SDK's own, `ANTHROPIC_API_KEY` in the
environment; it is never in the repo, the config or a log.

## A run's files

`runs/<phase>-<arm>-s<seed>/` holds `run.json` (the record), `summary.json` (what the report reads),
`responses.jsonl.gz` (every response, keyed by run and request) and `prompts_audit.jsonl.gz` (the full prompts of the
weeks in `audit_prompt_weeks`). `runs/phase-<name>.json` records a phase's runs and the runs it did not start. A
run's status is one of `complete`, `stopped_at_budget`, `incomplete_transient`,
`invalid`, `not_replayable` or `stopped_permanent` (a 4xx the retry loop does not retry; the plan's §7 says such a
run stops, and this status names it). Only a complete run is a result. What a pool has spent is summed from the
response caches. `replay` replays complete and invalid runs only, and marks a run `not_replayable` if a cached
response is missing or its replayed summary differs from the committed one.

## Tests

`tests/experiments/llm_agents/`, one test or more per acceptance criterion LX.1–LX.20, each named for its ID. No test
calls the API: the choosers are stubs and the SDK's errors are mapped through a stand-in module. LX.L1 and LX.L2 are
live-only and are checked by the smoke call and the pilot.
