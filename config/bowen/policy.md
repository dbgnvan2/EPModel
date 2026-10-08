# EPModel v2 — policy rules

> The policy's tie-break and fallback rules (spec `M4.D.1f`: both `[I]`, both declared in config).
> Parsed strictly; a rule the policy does not implement is refused at load (`src/bowen/policy/rules.py`).
>
> * `tie_break: equal_probability` — equal scores take equal probability in the softmax, so no tie-break
>   draw is ever made and no selection is flagged `tie_break`.
> * `fallback: hold` — on an empty legal set or a non-finite weight the person holds: nothing is emitted,
>   no automatic act is computed, and the selection is flagged `fallback` (`M16.A.3c`). `life_energy`
>   (`M6.I.3`) is not built in Phase C, so it cannot be exhausted; the rule covers it when it is.

| rule | value | spec |
|---|---|---|
| `tie_break` | equal_probability | `M4.D.1f` |
| `fallback` | hold | `M4.D.1f` |
