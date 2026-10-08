# EPModel v2 — changes to frozen constants

> Every change to a constant after `frozen_at` in `constants.md` (spec `M10.B.4`).
> A change that alters an acceptance outcome names the failing criterion, and a
> criterion that passes only after such a change is reported as **post-hoc**.
> `tests/bowen/test_phase_b_gate.py::test_m10b4_constants_frozen_before_suite` fails if
> a value differs from `constants_frozen.md` without a row here.

| key | frozen_value | new_value | date | criterion | post_hoc |
|---|---|---|---|---|---|
| `standing_load_gain` | 0.2 | 0.2 (meaning changed: scales the "too little" side, M4.C.1c) | 2026-10-07 | — (spec revision 11 replaced the mechanism; Phase C step 1) | no |
| `interactive_standing_fraction` | 0.5 | retired | 2026-10-07 | — (spec revision 11 replaced the mechanism; Phase C step 1) | no |
| `tension_activation_threshold` | 5.0 | retired | 2026-10-07 | — (spec revision 11 replaced the mechanism; Phase C step 1) | no |
| `appraisal_gain` | — | 3.0 | 2026-10-07 | — (spec revision 11 replaced the mechanism; Phase C step 1) | no |
| `intensity_scale` | — | 100.0 | 2026-10-07 | — (spec revision 11 replaced the mechanism; Phase C step 1) | no |
| `contact_band_max` | — | 0.2 | 2026-10-07 | — (spec revision 11 replaced the mechanism; Phase C step 1) | no |
| `anxiety_togetherness_gain` | — | 0.5 | 2026-10-07 | — (spec revision 11 replaced the mechanism; Phase C step 1) | no |
| `interactive_resting_contact` | — | 0.5 | 2026-10-07 | — (spec revision 11 replaced the mechanism; Phase C step 1) | no |
| `contact_relaxation_rate` | — | 0.1 | 2026-10-07 | — (spec revision 11 replaced the mechanism; Phase C step 1) | no |
| `impingement_relaxation_rate` | — | 0.3 | 2026-10-07 | — (spec revision 11 replaced the mechanism; Phase C step 1) | no |
| `triangle_activity_window` | — | 4 | 2026-10-07 | — (spec revision 11 replaced the mechanism; Phase C step 1) | no |
| `defence_threshold` | — | 15.0 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `witness_weight` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `speaker_echo_gain` | — | 0.2 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `reappraisal_window` | — | 4 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `attention_gain` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `perspective_anxiety_scale` | — | 10.0 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `calm_transfer_rate` | — | 0.05 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `symptom_leak_rate` | — | 0.1 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `symptom_threshold_gain` | — | 1.5 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `symptom_rearm_fraction` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `symptom_event_intensity` | — | 100.0 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `reactive_rate` | — | 0.2 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `investment_leak_rate` | — | 0.1 | 2026-10-07 | — (new mechanism; Phase C step 2) | no |
| `initial_impingement_scale` | — | 0.8 | 2026-10-07 | — (new mechanism; Phase C step 3) | no |
| `outside_ness_rate` | — | 0.1 | 2026-10-07 | — (new mechanism; Phase C step 3) | no |
| `hollow_gain` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 3) | no |
| `assault_gain` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 3) | no |
| `outside_ness_threshold_outward` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 3) | no |
| `outside_ness_threshold_inward` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 3) | no |
| `belief_rate` | — | 0.3 | 2026-10-07 | — (new mechanism; Phase C step 4) | no |
| `distance_binding_rate` | — | 0.3 | 2026-10-07 | — (new mechanism; Phase C step 5) | no |
| `triangle_transfer_rate` | — | 0.3 | 2026-10-07 | — (new mechanism; Phase C step 5) | no |
| `outsider_positional_gain` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 5) | no |
| `balance_push_gain` | — | 0.2 | 2026-10-07 | — (new mechanism; Phase C step 5) | no |
| `balance_settle_rate` | — | 0.05 | 2026-10-07 | — (new mechanism; Phase C step 5) | no |
| `balance_harden_rate` | — | 0.02 | 2026-10-07 | — (new mechanism; Phase C step 5) | no |
| `reversal_asymmetry` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 5) | no |
| `pseudo_self_transfer_gain` | — | 2.0 | 2026-10-07 | — (new mechanism; Phase C step 5) | no |
| `self_channel_exponent` | — | 2.0 | 2026-10-07 | — (new mechanism; Phase C step 6) | no |
| `policy_temperature` | — | 1.0 | 2026-10-07 | — (new mechanism; Phase C step 6) | no |
| `anxiety_band_low` | — | 5.0 | 2026-10-07 | — (new mechanism; Phase C step 6) | no |
| `anxiety_band_high` | — | 15.0 | 2026-10-07 | — (new mechanism; Phase C step 6) | no |
| `capacity_level_per_layer` | — | 20.0 | 2026-10-07 | — (new mechanism; Phase C step 6) | no |
| `competing_urge_gain` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 6) | no |
| `withhold_investment_gain` | — | 0.2 | 2026-10-07 | — (new mechanism; Phase C step 6) | no |
| `loaded_tie_threshold` | — | 0.2 | 2026-10-07 | — (new mechanism; Phase C step 6) | no |
| `policy_intensity` | — | 100.0 | 2026-10-07 | — (new mechanism; Phase C step 6) | no |
| `learning_rate` | — | 0.2 | 2026-10-07 | — (new mechanism; Phase C step 7) | no |
| `credit_horizon` | — | 3 | 2026-10-07 | — (new mechanism; Phase C step 7) | no |
| `credit_discount` | — | 0.7 | 2026-10-07 | — (new mechanism; Phase C step 7) | no |
| `cross_person_weight` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 7) | no |
| `habituation_rate` | — | 0.7 | 2026-10-07 | — (new mechanism; Phase C step 7) | no |
| `habituation_window` | — | 8 | 2026-10-07 | — (new mechanism; Phase C step 7) | no |
| `prepare_ticks` | — | 4 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `rehearsal_rate` | — | 0.05 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `assertion_perspective_threshold` | — | 0.3 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `anger_threshold` | — | 0.3 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `assertion_gain` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `assertion_evidence_gain` | — | 0.05 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `opposition_window` | — | 3 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `hold_gain` | — | 0.3 | 2026-10-07 | — (new mechanism; Phase C step 8). **Chosen against a requirement:** at 0.5 most movers held at first opposition (86 of 113 over 40 seeds), contrary to `M5.D.3`'s "the usual outcome"; at 0.3 most abort (62 of 110). Measured on the reference family with Ravi given systems perspective 1.0 | no |
| `stall_limit` | — | 4 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `hold_window` | — | 6 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `pull_up_rate` | — | 0.2 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `exchange_gain` | — | 0.5 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `triangle_floor_decrement` | — | 0.1 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `respect_gain` | — | 0.2 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `debit_gain` | — | 0.3 | 2026-10-07 | — (new mechanism; Phase C step 8) | no |
| `session_interval_weeks` | — | 8 | 2026-10-07 | — (new mechanism; Phase C step 9) | no |
| `landing_rate` | — | 0.15 | 2026-10-07 | — (new mechanism; Phase C step 9) | no |
| `delayed_view_bonus` | — | 1.0 | 2026-10-07 | — (new mechanism; Phase C step 9) | no |
| `binder_failure_fraction` | — | 0.8 | 2026-10-07 | — (new mechanism; Phase C step 9) | no |
| `contact_optimum` | — | 4 | 2026-10-07 | — (new mechanism; Phase C step 9) | no |
| `contact_window` | — | 26 | 2026-10-07 | — (new mechanism; Phase C step 9) | no |
| `perspective_gain` | — | 0.1 | 2026-10-07 | — (new mechanism; Phase C step 9) | no |
| `delayed_view_weeks` | — | 26 | 2026-10-07 | — (new mechanism; Phase C step 9) | no |
| `sink_window` | — | 8 | 2026-10-07 | — (new mechanism; Phase C step 10) | no |
| `sink_rate` | — | 0.1 | 2026-10-07 | — (new mechanism; Phase C step 10) | no |
| `exchange_budget_reduction` | — | 2.0 | 2026-10-07 | — (new mechanism; Phase C step 10) | no |
| `pattern_window` | — | 8 | 2026-10-07 | — (new readout; Phase C step 11) | no |
| `pattern_min_acts` | — | 2 | 2026-10-07 | — (new readout; Phase C step 11) | no |
| `pattern_pole` | — | 0.5 | 2026-10-07 | — (new readout; Phase C step 11) | no |
| `pattern_fixed_activations` | — | 3 | 2026-10-07 | — (new readout; Phase C step 11) | no |
| `ensemble_block` | — | 50 | 2026-10-07 | — (ensemble rule; Phase C step 12) | no |
| `ensemble_cap` | — | 500 | 2026-10-07 | — (ensemble rule; Phase C step 12) | no |
| `ensemble_precision` | — | 0.25 | 2026-10-07 | — (ensemble rule; Phase C step 12) | no |
| `ensemble_margin` | — | 0.1 | 2026-10-07 | — (ensemble rule; Phase C step 12) | no |
| `ensemble_alpha` | — | 0.05 | 2026-10-07 | — (ensemble rule; Phase C step 12) | no |
| `fallback_flag_rate` | — | 0.2 | 2026-10-07 | — (ensemble rule; Phase C step 12) | no |
| `equivalence_margin` | — | 0.5 | 2026-10-07 | — (ensemble rule; Phase C step 12) | no |
| `contact_excess_exponent` | — | 2.0 | 2026-10-07 | — (promoted from a literal by M11.D.2; Phase C step 13) | no |
