# EPModel v2 — constants register

> Every numeric constant the engine reads, with its grade (spec `M10.1`, `M0.1`).
> `[I]` means invented: a modelling decision with no source. An `[I]` value must
> never be described as sourced, derived from Bowen, or theoretically grounded (`M0.2`).
> Parsed strictly by `src/bowen/scenario/constants.py`: an unknown key, a missing
> key, a grade that differs from the spec's, or any unrecognised line raises (`M10.B.2`).
> `frozen_at` is set before the acceptance suite first runs against these values;
> any later change that alters an acceptance outcome is logged (`M10.B.4`).
>
> `chronic_anxiety_fixation_age_years` has no source. `M2.A.0a` states that every
> agent in the reference family is past it, and the youngest is 14, so it must be
> at most 14. 12 was chosen at Phase B step 5.

frozen_at: unset

| key | value | grade | unit | spec |
|---|---|---|---|---|
| `fast_tick_weeks` | 1 | [I] | week | `M3.A.1` |
| `slow_tick_fast_ticks` | 52 | [I] | fast tick | `M3.B.1` |
| `invariant_tolerance` | 1e-9 | [I] | anxiety unit | `M6.1` |
| `spouse_basic_level_tolerance` | 1.0 | [I] | basic-level point | `M2.A.0e` |
| `chronic_anxiety_fixation_age_years` | 12 | [I] | year of age | `M2.A.0a` |
