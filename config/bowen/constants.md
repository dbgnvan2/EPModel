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
>
> `activation_regime` is `[I]` (`M3.E.2`): every person selects every fast tick.
> `per_hop_fidelity` stands in for the per-hop fidelity draw of `M3.D.4b` in
> Phase B, which makes no draws (plan decision D7). 0.8 was chosen at step 7.
>
> The step 8 constants (from `standing_load_gain` down) were chosen together so that,
> on the Phase B family, the standing load alone holds each person a few points above
> their chronic floor: steady excess = standing load ÷ `acute_decay_rate`. `M4.G.1`'s
> "three withdrawals" is the spec's own number, still `[I]`. `bond_energy_decay_rate`
> is 0 because `M1.B.4` requires decay at or near zero. `functional_level_floor` only
> stops a division by zero at `functional_level` 0, which `M1.A.2` allows.

frozen_at: unset
activation_regime: synchronous

| key | value | grade | unit | spec |
|---|---|---|---|---|
| `fast_tick_weeks` | 1 | [I] | week | `M3.A.1` |
| `slow_tick_fast_ticks` | 52 | [I] | fast tick | `M3.B.1` |
| `invariant_tolerance` | 1e-9 | [I] | anxiety unit | `M6.1` |
| `spouse_basic_level_tolerance` | 1.0 | [I] | basic-level point | `M2.A.0e` |
| `chronic_anxiety_fixation_age_years` | 12 | [I] | year of age | `M2.A.0a` |
| `per_hop_fidelity` | 0.8 | [I] | fraction kept per private hop | `M1.F.4` |
| `standing_load_gain` | 0.2 | [I] | anxiety per tick per bond-energy unit per functional-level unit | `M4.A.1` |
| `interactive_standing_fraction` | 0.5 | [I] | fraction | `M4.A.3` |
| `functional_level_floor` | 1.0 | [I] | functional-level point | `M4.C.1` |
| `acute_decay_rate` | 0.2 | [I] | fraction of excess per tick | `M1.A.8` |
| `route_damping` | 0.5 | [I] | gain per neutral third | `M1.F.3` |
| `hardening_run_length` | 3 | [I] | consecutive moves | `M4.G.1` |
| `bond_energy_decay_rate` | 0.0 | [I] | fraction per tick | `M1.B.4` |
| `tension_activation_threshold` | 5.0 | [I] | anxiety above the chronic floor | `M1.C.3` |
| `involvement_membership_threshold` | 0.5 | [I] | involvement units | `M1.A.12` |
