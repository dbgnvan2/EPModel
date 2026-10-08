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
