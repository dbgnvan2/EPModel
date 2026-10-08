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
>
> **Phase C step 1, 2026-10-07 — spec revision 11's two-sided appraisal (`M4.C.1`–`M4.C.1c`, plan D2).**
> `standing_load_gain` keeps its value and now scales the "too little" side of the appraisal function.
> The constants from `appraisal_gain` to `impingement_relaxation_rate` and `triangle_activity_window` are new;
> `interactive_standing_fraction` and `tension_activation_threshold` are retired. Every change is logged in
> `constants_changes.md`. The new values were chosen so that, on the Phase B family, magnitudes stay on the
> Phase B scale: `intensity_scale` 100 makes a stressor's appraisal equal Phase B's (intensity ÷ functional
> level); an interactive tie resting at half its optimum (`interactive_resting_contact` 0.5) carries roughly
> Phase B's interactive standing load. `triangle_activity_window` 4 keeps a triangle active for four weeks
> after a `TRIANGLE` act. All are `[I]`; Phase C re-freezes at step 14.
>
> **Phase C step 2, 2026-10-07 — the rest of the appraisal** (`M4.C.2`–`M4.C.10`, `M4.C.3`, `M7.D.1`,
> `M1.A.19`, `M1.B.8`). The constants from `defence_threshold` to `investment_leak_rate` are new and logged.
> Chosen so that a witness takes about half of what Phase B gave it, below the person addressed; content stops
> getting through at 15–30 points of excess anxiety; and the symptom threshold (1.5 × functional level, about 60
> for the Phase B adults) sits above the integral a steady few points of excess reach (excess ÷ `symptom_leak_rate`),
> so onset needs a sustained excursion. All `[I]`.
>
> **Phase C step 3, 2026-10-07 — outside-ness** (`M1.A.9`, `M1.A.9a`, `M4.C.5`, `M5.F`). The six constants from
> `initial_impingement_scale` down are new and logged. At 0.8, the Phase B adults (basic level about 40) start
> near 0.5 on both axes, at the gate's thresholds: the gate passes or fails on behaviour, not on the start.
> `hollow_gain` and `assault_gain` at 0.5 keep act identity smaller than the move's own components. All `[I]`.
>
> **Phase C steps 4–7, 2026-10-07** — beliefs (`belief_rate`), move physics (`distance_binding_rate` …
> `pseudo_self_transfer_gain`), the policy (`self_channel_exponent` … `policy_intensity`) and the learner
> (`learning_rate` … `habituation_window`). Each is new and logged in `constants_changes.md`. All `[I]`.
>
> **Phase C step 8, 2026-10-07 — the `I-POSITION` state machine** (`M5.D`, `M5.E`, `M5.F.4`, `M1.C.5`), from
> `prepare_ticks` to `debit_gain`. All `[I]`. **`exchange_gain` is small on purpose (`M5.D.7a`):** half a point
> of functional level per completed exchange, against basic levels around 40, and `basic_level` is never
> written by an exchange. No realistic number of exchanges in a run should read as differentiation: the KB
> interviews call that reading "grotesque". **`hold_gain` was chosen against `M5.D.3`**, so that aborts are the
> usual outcome at first opposition; `constants_changes.md` records the measurement.

frozen_at: 2026-10-06
activation_regime: synchronous

| key | value | grade | unit | spec |
|---|---|---|---|---|
| `fast_tick_weeks` | 1 | [I] | week | `M3.A.1` |
| `slow_tick_fast_ticks` | 52 | [I] | fast tick | `M3.B.1` |
| `invariant_tolerance` | 1e-9 | [I] | anxiety unit | `M6.1` |
| `spouse_basic_level_tolerance` | 1.0 | [I] | basic-level point | `M2.A.0e` |
| `chronic_anxiety_fixation_age_years` | 12 | [I] | year of age | `M2.A.0a` |
| `per_hop_fidelity` | 0.8 | [I] | fraction kept per private hop | `M1.F.4` |
| `standing_load_gain` | 0.2 | [I] | anxiety per tick per unit of "too little" deviation per unit of steepness | `M4.A.1` |
| `appraisal_gain` | 3.0 | [I] | anxiety per unit change in deviation per unit of steepness | `M4.C.1` |
| `intensity_scale` | 100.0 | [I] | event intensity per unit component | `M4.C.1` |
| `contact_band_max` | 0.2 | [I] | deviation tolerated at functional level 100 | `M4.C.1a` |
| `anxiety_togetherness_gain` | 0.5 | [I] | relative rise of the optimum per 100 points of excess anxiety | `M4.C.1b` |
| `interactive_resting_contact` | 0.5 | [I] | share of the optimum | `M4.C.1c` |
| `contact_relaxation_rate` | 0.1 | [I] | share of the gap to resting contact closed per tick | `M4.C.1c` |
| `impingement_relaxation_rate` | 0.3 | [I] | share of felt impingement shed per tick | `M4.C.1` |
| `functional_level_floor` | 1.0 | [I] | functional-level point | `M4.C.1` |
| `acute_decay_rate` | 0.2 | [I] | fraction of excess per tick | `M1.A.8` |
| `route_damping` | 0.5 | [I] | gain per neutral third | `M1.F.3` |
| `hardening_run_length` | 3 | [I] | consecutive moves | `M4.G.1` |
| `bond_energy_decay_rate` | 0.0 | [I] | fraction per tick | `M1.B.4` |
| `triangle_activity_window` | 4 | [I] | fast ticks | `M1.C.3` |
| `defence_threshold` | 15.0 | [I] | points of excess anxiety | `M4.C.2` |
| `witness_weight` | 0.5 | [I] | share of an overheard exchange | `M4.C.9` |
| `speaker_echo_gain` | 0.2 | [I] | share of what was addressed | `M4.C.7` |
| `reappraisal_window` | 4 | [I] | fast ticks | `M4.C.6` |
| `attention_gain` | 0.5 | [I] | relative change in appraisal | `M4.C.8` |
| `perspective_anxiety_scale` | 10.0 | [I] | points of excess anxiety | `M4.C.4` |
| `calm_transfer_rate` | 0.05 | [I] | share of the anxiety gap per unit conductance | `M4.C.10` |
| `symptom_leak_rate` | 0.1 | [I] | share of the integral per tick | `M4.C.3a` |
| `symptom_threshold_gain` | 1.5 | [I] | integral units per functional-level point | `M7.D.1` |
| `symptom_rearm_fraction` | 0.5 | [I] | share of the threshold | `M7.D.1` |
| `symptom_event_intensity` | 100.0 | [I] | event intensity | `M7.D.1` |
| `reactive_rate` | 0.2 | [I] | share of the gap per tick | `M1.A.19` |
| `investment_leak_rate` | 0.1 | [I] | share per tick | `M1.B.8` |
| `initial_impingement_scale` | 0.8 | [I] | impingement at basic level 0 | `M1.A.9` |
| `outside_ness_rate` | 0.1 | [I] | share of the gap per tick | `M1.A.9` |
| `hollow_gain` | 0.5 | [I] | share of contact hollowed at full inward impingement | `M5.F.1` |
| `assault_gain` | 0.5 | [I] | impingement added per unit outward impingement | `M5.F.1` |
| `outside_ness_threshold_outward` | 0.5 | [I] | outward impingement | `M5.C.1` |
| `outside_ness_threshold_inward` | 0.5 | [I] | inward impingement | `M5.C.1` |
| `belief_rate` | 0.3 | [I] | share of the gap to an observation per tick, at full fidelity | `M9.8` |
| `self_channel_exponent` | 2.0 | [I] | exponent on functional_level / 100 | `M4.D.1a` |
| `policy_temperature` | 1.0 | [I] | learned-value units | `M4.D.1` |
| `anxiety_band_low` | 5.0 | [I] | points of excess anxiety | `M4.D.3` |
| `anxiety_band_high` | 15.0 | [I] | points of excess anxiety | `M4.D.3` |
| `capacity_level_per_layer` | 20.0 | [I] | functional-level points per layer | `M4.D.3a` |
| `competing_urge_gain` | 0.5 | [I] | points of acute anxiety at maximal entropy | `M4.D.1d` |
| `withhold_investment_gain` | 0.2 | [I] | share of attention per unit scaled intensity | `M4.D.1b` |
| `loaded_tie_threshold` | 0.2 | [I] | deviation units | `M4.D.3b` |
| `policy_intensity` | 100.0 | [I] | intensity units | `M4.D.1` |
| `learning_rate` | 0.2 | [I] | share of the gap to the signal per update | `M4.D.6` |
| `credit_horizon` | 3 | [I] | fast ticks | `M4.D.6b` |
| `credit_discount` | 0.7 | [I] | weight per tick of age | `M4.D.6` |
| `cross_person_weight` | 0.5 | [I] | weight on the others' mean relief | `M4.D.6e` |
| `habituation_rate` | 0.7 | [I] | relief kept per earlier repetition | `M4.G.3` |
| `habituation_window` | 8 | [I] | fast ticks | `M4.G.3` |
| `prepare_ticks` | 4 | [I] | fast ticks | `M5.D.2a` |
| `rehearsal_rate` | 0.05 | [I] | share of each axis per tick | `M1.A.9` |
| `assertion_perspective_threshold` | 0.3 | [I] | systems perspective | `M5.F.4` |
| `anger_threshold` | 0.3 | [I] | too-much deviation | `M5.D.4` |
| `assertion_gain` | 0.5 | [I] | impingement per unit scaled intensity | `M5.F.4` |
| `assertion_evidence_gain` | 0.05 | [I] | outward impingement per assertion | `M5.F.2a` |
| `opposition_window` | 3 | [I] | fast ticks | `M5.E.3` |
| `hold_gain` | 0.3 | [I] | excess anxiety per point of functional level at full efficacy | `M5.D.3` |
| `stall_limit` | 4 | [I] | fast ticks | `M5.D.4` |
| `hold_window` | 6 | [I] | fast ticks | `M5.D.2` |
| `pull_up_rate` | 0.2 | [I] | share of the level gap | `M5.D.5` |
| `exchange_gain` | 0.5 | [I] | functional-level points | `M5.D.7a` |
| `triangle_floor_decrement` | 0.1 | [I] | share of routing capacity | `M1.C.5` |
| `respect_gain` | 0.2 | [I] | share of each axis | `M5.E.8` |
| `debit_gain` | 0.3 | [I] | felt contact per unit scaled intensity | `M5.E.7` |
| `session_interval_weeks` | 8 | [I] | fast ticks | `M1.E.8` |
| `landing_rate` | 0.15 | [I] | probability per contact at full coach quality | `M1.E.7` |
| `delayed_view_bonus` | 1.0 | [I] | added share of the landing rate | `M1.E.7c` |
| `binder_failure_fraction` | 0.8 | [I] | share of the symptom threshold | `M1.E.7` |
| `contact_optimum` | 4 | [I] | contacts in the window | `M1.E.8` |
| `contact_window` | 26 | [I] | fast ticks | `M1.E.8` |
| `perspective_gain` | 0.1 | [I] | share of the gap to 1 | `M1.E.7` |
| `delayed_view_weeks` | 26 | [I] | fast ticks | `M16.D.2` |
| `sink_window` | 8 | [I] | fast ticks | `M1.D.1` |
| `sink_rate` | 0.1 | [I] | share of the gap per tick | `M1.D.1` |
| `exchange_budget_reduction` | 2.0 | [I] | budget units per completed exchange | `M6.I.1` |
| `pattern_window` | 8 | [I] | fast ticks | `M5.A.1a` |
| `pattern_min_acts` | 2 | [I] | acts per member in the window | `M5.A.1a` |
| `pattern_pole` | 0.5 | [I] | functioning habit | `M5.A.1a` |
| `pattern_fixed_activations` | 3 | [I] | consecutive activations | `M5.A.1a` |
| `distance_binding_rate` | 0.3 | [I] | share of excess bound per unit scaled intensity | `M1.D.2a` |
| `triangle_transfer_rate` | 0.3 | [I] | share of each insider's excess passed per unit scaled intensity | `M1.C.1` |
| `outsider_positional_gain` | 0.5 | [I] | positional anxiety per unit transferred | `M1.C.1` |
| `balance_push_gain` | 0.2 | [I] | balance per unit scaled intensity | `M1.B.5` |
| `balance_settle_rate` | 0.05 | [I] | share of the gap to the pole per tick | `M1.B.5` |
| `balance_harden_rate` | 0.02 | [I] | share of the gap per tick | `M1.B.6` |
| `reversal_asymmetry` | 0.5 | [I] | share of an upward push lost at a fully hardened habit | `M1.B.7` |
| `pseudo_self_transfer_gain` | 2.0 | [I] | functional-level points per unit scaled intensity | `M6.I.4` |
| `involvement_membership_threshold` | 0.5 | [I] | involvement units | `M1.A.12` |
