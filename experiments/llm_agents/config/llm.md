# LLM-agent experiment — the model call, the scale and the budget

> The settings of `docs/plan_v3_llm_agent_experiment.md` §4, §5, §7 and §8, as the owner decided them on 2026-11-06
> (Q3, Q4, Q5: all as recommended). Read by `experiments/llm_agents/config.py`; an unknown or missing setting is
> refused at load. Nothing here is under `config/bowen/`, so no edit here moves the engine hash.
>
> * **Models** (L1). Model IDs live here and nowhere in code. Before any live phase the runner checks each ID it will
>   use against the Models API and stops if one is not found. `claude-haiku-5-5` takes `thinking: disabled` at effort
>   `low`; `claude-sonnet-5-5` takes `between_tools`, the setting that turns its thinking off (`disabled` is a 400 on
>   it). Sampling (`temperature`, `top_p`, `top_k`) is not sent, so it is the model's default and
>   cannot be seeded: at API version `2023-06-01`, the one the SDK sends, both models return a 400 for any non-default
>   value (a model rule), and at `2026-10-21` any value is a 400 on every model; the run record says so.
> * **Prices** are dollars per million tokens, read from the `claude-api` skill's model table (cached 2026-10-06) on
>   `prices_read`; they are checked against the pricing page before the first live run (LX.L2).
> * **Estimate** (§5): `estimate_input_tokens` and `estimate_output_tokens` are the per-call estimate, restated
>   2026-11-06 from the pilot's `usage` (LX.L2): input 2,000 (the pilot's mean 1,590, max 2,337 over 26 weeks of A1;
>   the four arms' 208-week mean, sized offline, about 1,910, rounded up), output 10 (every one of the pilot's 182
>   calls; the plan's first figure was 50). The cross-check model's output is not yet measured. **Worst case** before a round (§5): each call's input is its prompt's length
>   in characters divided by `worst_case_chars_per_token` (2, below English text's usual 3–4, so the bound is high),
>   its output is `max_tokens`, and every call is counted twice, for a possible re-ask.
> * **Phases** (§11 step 7). `smoke` is one call (LX.L1); `pilot` is A1, one seed, 26 weeks (LX.L2); `main` is A1–A4
>   × 5 seeds; `cross_check` is A1 and A2 × 2 seeds on the second model. `baseline` is A0, the model's own policy: it
>   makes no call and costs nothing. A phase's `pool` names its cap: `main` ($10) holds the smoke call, the pilot and
>   the main phase, `cross_check` ($35) the cross-check; `cap_total_usd` ($45) bounds both.
> * **Scale** (Q3): the family is `phase_c`, the run is `horizon_weeks` weekly ticks, the standard spell of the
>   ensemble (`config/bowen/criteria.md`'s `spell`: a `JOB_LOSS` to each parent on the declared weeks) runs in every
>   arm, as in the ensemble's baseline arms, so A0 is the ensemble's own assembly (LX.17).
> * `option_order` (§13.1, decided 2026-11-06): `shuffled` numbers each round's options in an order drawn per
>   (seed, week, person token), over acts and holds together, outside the engine's draws; `fixed` is the order the
>   first pilot ran in (acts by target and label, then the holds), kept only so that run replays. A run record
>   carries the order it ran in; a record without one is read as `fixed`. `pilot2` is the second pilot (§13.5): the
>   same as `pilot`, run in the shuffled order.
> * `max_fallback_share` (§7): a run with more of its LLM selections as `llm_failed_to_parse` holds is `invalid`.
> * `audit_prompt_weeks` (§8): the weeks whose full prompts are stored for audit; `last` is the run's last week.
> * `evaluation_window_weeks` and `evaluation_readouts` (§6): the window of the `M11.G.1` evaluation and the numeric
>   readouts of it the report tabulates.

| setting | value |
|---|---|
| `model_main` | claude-haiku-5-5 |
| `model_cross_check` | claude-sonnet-5-5 |
| `thinking.claude-haiku-5-5` | disabled |
| `effort.claude-haiku-5-5` | low |
| `thinking.claude-sonnet-5-5` | between_tools |
| `effort.claude-sonnet-5-5` | low |
| `price_input.claude-haiku-5-5` | 0.10 |
| `price_output.claude-haiku-5-5` | 0.50 |
| `price_input.claude-sonnet-5-5` | 2.00 |
| `price_output.claude-sonnet-5-5` | 10.00 |
| `prices_read` | 2026-11-03 |
| `max_tokens` | 64 |
| `timeout_seconds` | 60 |
| `max_attempts` | 5 |
| `backoff_base_seconds` | 1 |
| `backoff_cap_seconds` | 60 |
| `retry_after_max_seconds` | 300 |
| `max_concurrency` | 7 |
| `max_fallback_share` | 0.01 |
| `history_rounds` | 12 |
| `estimate_input_tokens` | 2000 |
| `estimate_output_tokens` | 10 |
| `worst_case_chars_per_token` | 2 |
| `cap_main_usd` | 10 |
| `cap_cross_check_usd` | 35 |
| `cap_total_usd` | 45 |
| `family` | phase_c |
| `horizon_weeks` | 208 |
| `spells` | standard |
| `audit_prompt_weeks` | 0, 1, last |
| `evaluation_window_weeks` | 52 |
| `option_order` | shuffled |
| `evaluation_readouts` | intensity.mean_excess, intensity.budget_occupancy, stress.exogenous_events, adaptiveness.automatic_acts, adaptiveness.reactivity_per_stress |
| `phase.baseline` | arms=A0; seeds=1,2,3,4,5; weeks=208; model=none; pool=none |
| `phase.smoke` | arms=A1; seeds=1; weeks=1; model=claude-haiku-5-5; pool=main; calls=1 |
| `phase.pilot` | arms=A1; seeds=1; weeks=26; model=claude-haiku-5-5; pool=main |
| `phase.pilot2` | arms=A1; seeds=1; weeks=26; model=claude-haiku-5-5; pool=main |
| `phase.main` | arms=A1,A2,A3,A4; seeds=1,2,3,4,5; weeks=208; model=claude-haiku-5-5; pool=main |
| `phase.cross_check` | arms=A1,A2; seeds=1,2; weeks=208; model=claude-sonnet-5-5; pool=cross_check |
