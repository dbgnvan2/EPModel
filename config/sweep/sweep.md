# EPModel version 2 — the constant sweep's rules and ranges (stage 2, `M17.E.1`)

> What `tools/version_2_sweep.py` reads: the sweep's own rules and, for each swept criterion, the `[I]` constants its
> mechanism reads with the range each is sampled over. **The engine never reads this file.** It is outside
> `config/bowen/` because the engine hash covers every `*.md` there (`tools/ensemble_record.py::code_hash`), and a
> sweep's ranges are an input to a measurement of the model, not to the model
> (`docs/implementation_plan_version_2.md` §2.1 item 4, D12). Parsed strictly: an unknown constant, a range that does
> not bracket the constant's central value in `config/bowen/constants.md`, a range outside its declared bound, a
> scale other than `log`, `linear` or `integer`, or a row without a grade raises.
>
> Every value here is `[I]`: a modelling or measurement decision with no source (`M0.2`). Each range is half to
> double the central value, the rule the Phase C plan's D9 sweep used, bounded where the constant has an admissible
> bound: a share or weight is capped at 1, the value at which what it scales is not discounted; a count of ticks is
> rounded to whole ticks and never below 1. A `log` range is sampled log-uniformly, so the central value is its
> midpoint unless the range is capped (then the cap cuts the upper half short); an `integer` range uniformly over its
> whole values.
>
> **Grades.** The grade column gives the constant's grade as `docs/model_explainer.md` gives it, found by grep of the
> constant's name. None of the ten constants below (seventeen rows) is named there by its code name (checked 2026-10-23), so each
> is `[I]` until proven otherwise, as `config/bowen/constants.md` already grades it; the column says so.
>
> Decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review
> (`docs/implementation_plan_version_2.md` §2.2, D8, D10, D12).

## Rules

| constant | value | grade | unit | why |
|---|---|---|---|---|
| `sweep_samples` | 8 | [I] | samples per criterion | the pilot's count (D10): enough to time the runs, not a measurement of a criterion |
| `sweep_seed` | 2 | [I] | generator seed | with the criterion's key, seeds the draw of its samples; a longer draw begins with the shorter one |
| `detection_limit_multiple` | 2.0 | [I] | half-widths | D8 (`M17.A.1`): a first run whose gating effect is within this many half-widths of zero is run again |
| `detection_limit_seeds` | 200 | [I] | seeds | D8: the fixed seed count of that second run, taken as one block; the count the owner chose for `M11.C.10` |

## Ranges

| criterion | constant | low | high | scale | bound | grade | why it is read |
|---|---|---|---|---|---|---|---|
| `M11.C.16` | `learning_rate` | 0.1 | 0.4 | log | — | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.D.6`: both arms learn; the share of the gap to the signal taken per update |
| `M11.C.16` | `credit_horizon` | 2 | 6 | integer | — | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.D.6b`: the ticks over which felt relief is credited |
| `M11.C.16` | `credit_discount` | 0.35 | 1.0 | log | 1 | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.D.6`: the weight per tick of age; 1 is no discount |
| `M11.C.16` | `cross_person_weight` | 0.25 | 1.0 | log | 1 | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.D.6e`: the others' relief in the signal; 1 weights it as the actor's own |
| `M11.C.16` | `policy_temperature` | 0.5 | 2.0 | log | — | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.D.1`: the softmax over learned values that the readout's entropy is taken from |
| `M11.C.16` | `habituation_rate` | 0.35 | 1.0 | log | 1 | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.G.3`: relief kept per earlier repetition; 1 is no habituation |
| `M11.C.16` | `habituation_window` | 4 | 16 | integer | — | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.G.3`: the ticks a repetition counts over |
| `M11.C.42` | `learning_rate` | 0.1 | 0.4 | log | — | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.D.6`: whether the `TRIANGLE`'s relief is learned |
| `M11.C.42` | `credit_horizon` | 2 | 6 | integer | — | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.D.6b`: whether the relief falls within the credited ticks |
| `M11.C.42` | `credit_discount` | 0.35 | 1.0 | log | 1 | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.D.6`: the weight per tick of age; 1 is no discount |
| `M11.C.42` | `cross_person_weight` | 0.25 | 1.0 | log | 1 | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.D.6e`: the third's and the partner's change in the signal |
| `M11.C.42` | `policy_temperature` | 0.5 | 2.0 | log | — | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.D.1`: how sharply a learned value turns into a selection |
| `M11.C.42` | `habituation_rate` | 0.35 | 1.0 | log | 1 | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.G.3`: a repeated `TRIANGLE`'s relief is habituated |
| `M11.C.42` | `habituation_window` | 4 | 16 | integer | — | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M4.G.3`: the ticks a repetition counts over |
| `M11.C.42` | `triangle_transfer_rate` | 0.15 | 0.6 | log | 1 | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M1.C.1`: the share of each insider's excess the third takes — the relief the learner reads |
| `M11.C.42` | `outsider_positional_gain` | 0.25 | 1.0 | log | — | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M1.C.1`: the third's positional anxiety per unit transferred — the cost of being recruited |
| `M11.C.42` | `triangle_activity_window` | 2 | 8 | integer | — | [I] — not named in `model_explainer.md`; `constants.md` grades it [I] | `M1.C.3`: how long a triangle stays active, which the reuse readout counts |
