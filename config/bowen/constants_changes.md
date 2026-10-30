# EPModel v2 — changes to frozen constants

> Every change to a constant after `frozen_at` in `constants.md` (spec `M10.B.4`).
> A change that alters an acceptance outcome names the failing criterion, and a
> criterion that passes only after such a change is reported as **post-hoc**.
> `tests/bowen/test_phase_b_gate.py::test_m10b4_constants_frozen_before_suite` fails if
> a value differs from `constants_frozen.md` without a row here.

> **Re-frozen for Phase C on 2026-10-07** (`constants_frozen.md`). The rows above that date record the
> changes from the Phase B snapshot (`constants_frozen_phase_b.md`) made while Phase C was built, before any
> Phase C criterion ran. Rows dated after the re-freeze are changes against the Phase C snapshot.

> **Version 2 stage 1 (spec revision 33, 2026-10-19) changed no constant**, so it has no row here. The act classes it
> moved out of the code into `act_classes.md` (`M8.3`, `M5.B.4`, `M5.B.5`) are lists of act names, not constants, and
> their contents did not change.
>
> **The version 2 definition batch (spec revision 43, 2026-10-29) changed no constant**, so it has no row here.
> `M11.C.48`'s seed count (100 → 200, T-163) is a criterion setting in `criteria.md`, not a constant; `M11.C.48`'s
> readout (T-123) is code. The four new Phase D mutants (`M11.C.11`, `M11.C.15`, `M11.C.17`, `M11.C.34`) edit the
> engine in a temporary copy only.
>
> **The stage 3 triage batch (spec revision 41, 2026-10-28) changed no constant**, so it has no row here. Its rules
> (`M1.A.5f`, `M7.E.1f`, `M5.D.4a`, `M6.I.4a`) read existing constants: `rejection_refusal_threshold`,
> `response_relaxation_rate`, `assertion_gain` and `assertion_evidence_gain` at the all-demand end.

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
| `ensemble_precision` | 0.25 | 0.25 (unit changed: pooled seed-to-seed sd of the two arms, was the baseline arm's) | 2026-10-08 | — (owner request; `docs/phase_c_completion_report.md` §10, decided before rerunning) | yes |
| `pattern_window` | 8 | 4 | 2026-10-10 | — (owner decision D1, 2026-10-10, `docs/PROPOSAL — RUN VISUALIZATION MAPPING.md`: the owner's pattern definitions; no criterion reads the pattern readouts) | no |
| `pattern_min_acts` | 2 | 3 (unit changed: acts on the tie, was acts per member) | 2026-10-10 | — (owner decision D1, 2026-10-10, `docs/PROPOSAL — RUN VISUALIZATION MAPPING.md`: the owner's pattern definitions; no criterion reads the pattern readouts) | no |
| `pattern_pole` | 0.5 | retired | 2026-10-10 | — (owner decision D1, 2026-10-10, `docs/PROPOSAL — RUN VISUALIZATION MAPPING.md`: the owner's pattern definitions; no criterion reads the pattern readouts) | no |
| `pattern_projection_hold` | — | 12 | 2026-10-10 | — (owner, 2026-10-10: projection holds until 12 turns without evidence; no criterion reads the pattern readouts) | no |
| `rejection_cutoff_gain` | — | 0.5 | 2026-10-11 | — (new mechanism; Phase D B4, owner decisions P1 and P2: cutoff as rejection) | no |
| `rejection_refusal_gain` | — | 0.1 | 2026-10-11 | — (new mechanism; Phase D B4, owner decisions P1 and P2: cutoff as rejection) | no |
| `rejection_relief` | — | 0.25 | 2026-10-11 | — (new mechanism; Phase D B4, owner decisions P1 and P2: cutoff as rejection) | no |
| `refusal_cost_gain` | — | 1.0 | 2026-10-11 | — (new mechanism; Phase D B4, owner decisions P1 and P2: cutoff as rejection) | no |
| `declared_rejection` | — | 1.0 | 2026-10-11 | — (new mechanism; Phase D B4, owner decisions P1 and P2: cutoff as rejection) | no |
| `rejection_refusal_threshold` | — | 0.5 | 2026-10-11 | — (new mechanism; spec revision 16, owner decision 2026-10-11: graded rejection) | no |
| `rejection_refusal_threshold` | 0.5 | 0.3 | 2026-10-11 | — (owner decision 2026-10-11: at 0.5 a single cutoff, exactly 0.5, refused nothing; before any record was regenerated) | no |
| `rejection_fade` | — | 0.05 | 2026-10-11 | — (new mechanism; spec revision 16, owner decision 2026-10-11: graded rejection) | no |
| `estimator_window` | — | 3 | 2026-10-11 | — (new mechanism; Phase D B5/B6, owner decisions P4–P6) | no |
| `estimator_breadth` | — | 3 | 2026-10-11 | — (new mechanism; Phase D B5/B6, owner decisions P4–P6) | no |
| `estimator_sd_weight` | — | 1.0 | 2026-10-11 | — (new mechanism; Phase D B5/B6, owner decisions P4–P6) | no |
| `estimator_rate` | — | 0.5 | 2026-10-11 | — (new mechanism; Phase D B5/B6, owner decisions P4–P6) | no |
| `tie_wear_rate` | — | 0.05 | 2026-10-11 | — (new mechanism; Phase D B5/B6, owner decisions P4–P6) | no |
| `wear_offset_per_act` | — | 0.1 | 2026-10-11 | — (new mechanism; Phase D B5/B6, owner decisions P4–P6) | no |
| `perspective_channel_gain` | — | 0.5 | 2026-10-11 | — (new mechanism; spec revision 18, owner decision 2026-10-11: coaching or a person's own practice opens the self-directed channel) | no |
| `self_practice_gain` | — | 0.001 | 2026-10-11 | — (new mechanism; spec revision 18, owner decision 2026-10-11: coaching or a person's own practice opens the self-directed channel) | no |
| `stressor_persistence` | — | 0.5 | 2026-10-11 | — (new mechanism; spec revision 18, owner decision 2026-10-11: stress moves functioning up or down) | no |
| `response_up_gain` | — | 0.5 | 2026-10-11 | — (new mechanism; spec revision 18, owner decision 2026-10-11: stress moves functioning up or down) | no |
| `response_down_gain` | — | 0.5 | 2026-10-11 | — (new mechanism; spec revision 18, owner decision 2026-10-11: stress moves functioning up or down) | no |
| `response_relaxation_rate` | — | 0.02 | 2026-10-11 | — (new mechanism; spec revision 18, owner decision 2026-10-11: stress moves functioning up or down) | no |
| `stress_floor_below_basic` | — | 15.0 | 2026-10-11 | — (new mechanism; spec revision 19, owner decision 2026-10-11) | no |
| `practice_window` | — | 4 | 2026-10-11 | — (new mechanism; spec revision 19, owner decision 2026-10-11) | no |
| `perspective_steer_gain` | — | 0.5 | 2026-10-11 | — (new mechanism; spec revision 19, owner decision 2026-10-11) | no |
| `system_adjust_gain` | — | 1.0 | 2026-10-11 | — (new mechanism; spec revision 19, owner decision 2026-10-11: the system adjusts to a held stance) | no |
| `soothing_review_every` | — | 12 | 2026-10-11 | — (new mechanism; spec revision 19, owner decision 2026-10-11: "is soothing helping me?") | no |
| `soothing_review_penalty` | — | 0.5 | 2026-10-11 | — (new mechanism; spec revision 19, owner decision 2026-10-11: "is soothing helping me?") | no |
| `usual_level_rate` | — | 0.02 | 2026-10-12 | — (new mechanism; spec revision 22, owner decision 2026-10-12: calm relative to the person) | no |
| `constitution_strength` | — | 0.5 | 2026-10-12 | — (new mechanism; spec revision 24, B7, Claude's recommendation, approved by the owner 2026-10-16: the channel prior against a family focus) | no |
| `channel_protection` | — | 1.0 | 2026-10-12 | — (new mechanism; spec revision 24, B7, Claude's recommendation, approved by the owner 2026-10-16: channels are mutually protective) | no |
| `lock_in_gain` | — | 0.2 | 2026-10-12 | — (new mechanism; spec revision 24, B7, Claude's recommendation, approved by the owner 2026-10-16: lock-in) | no |
| `attribution_rate` | — | 0.05 | 2026-10-12 | — (new mechanism; spec revision 25, B8, Claude's recommendation, approved by the owner 2026-10-16: attribution) | no |
| `focus_rate` | — | 0.05 | 2026-10-12 | — (new mechanism; spec revision 25, B8, Claude's recommendation, approved by the owner 2026-10-16: the family's focus) | no |
| `marital_distance_fraction` | — | 0.25 | 2026-10-12 | — (new mechanism; spec revision 25, Claude's recommendation, approved by the owner 2026-10-16: the marital-distance gate) | no |
| `belief_appraisal_gain` | — | 1.0 | 2026-10-12 | — (new mechanism; spec revision 25, B8, Claude's recommendation, approved by the owner 2026-10-16: appraisal reads belief) | no |
| `pseudo_self_relaxation_rate` | — | 0.05 | 2026-10-12 | — (new mechanism; spec revision 27, Claude's recommendation, approved by the owner 2026-10-16: borrowed pseudo-self settles back) | no |
| `exchange_window` | — | 8 | 2026-10-12 | — (new mechanism; spec revision 27, Claude's recommendation, approved by the owner 2026-10-16) | no |
| `acting_age_years` | — | 10 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `development_start_age_years` | — | 5 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `development_end_age_years` | — | 20 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `development_band` | — | 0.02 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `development_rate` | — | 0.1 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `witnessed_gain` | — | 0.5 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `launch_age_years` | — | 20 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `working_age_years` | — | 16 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `find_work_rate` | — | 0.1 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `find_work_pressure_scale` | — | 10.0 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `grief_scale` | — | 20.0 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `grief_min_weeks` | — | 52 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `grief_extra_weeks` | — | 104 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `income_loss_intensity` | — | 15.0 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `income_shortfall_share` | — | 0.2 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `mortality_a` | — | 0.0000657 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `mortality_b` | — | 0.085 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `budget_per_undifferentiation` | — | 1.49 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `coach_start_age_years` | — | 35 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `coach_retirement_age_years` | — | 68 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `new_tie_investment_share` | — | 0.2 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `birth_focus_gain` | — | 0.1 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `death_transfer_share` | — | 1.0 | 2026-10-13 | — (new mechanism; spec revision 28, the life course, plan part II) | no |
| `reactive_cutoff_gain` | — | 0.1 | 2026-10-14 | — (new mechanism; spec revision 29, plan §4.8; owner-accepted default) | no |
| `witnessed_cutoff_gain` | — | 0.05 | 2026-10-14 | — (new mechanism; spec revision 29, plan §4.8; owner-accepted default) | no |
| `witness_window` | — | 13 | 2026-10-14 | — (new mechanism; spec revision 29, plan §4.8) | no |
| `perceived_weakness_gain` | — | 0.01 | 2026-10-14 | — (new mechanism; spec revision 29, plan §4.8; owner-accepted default) | no |
| `focus_steer_gain` | — | 1.0 | 2026-10-14 | — (new mechanism; spec revision 29, plan §4.8; owner-accepted default) | no |
| `origin_cutoff_anxiety_gain` | — | 10.0 | 2026-10-14 | — (new mechanism; spec revision 30, plan §4.10) | no |
| `origin_fusion_gain` | — | 0.5 | 2026-10-14 | — (new mechanism; spec revision 30, plan §4.10) | no |
| `danger_window` | — | 26 | 2026-10-14 | — (new mechanism; spec revision 30, plan §4.10) | no |
| `danger_threshold` | — | 0.25 | 2026-10-14 | — (new mechanism; spec revision 30, plan §4.10) | no |
| `danger_gain` | — | 2.0 | 2026-10-14 | — (new mechanism; spec revision 30, plan §4.10) | no |
| `focus_reinforce_gain` | — | 1.0 | 2026-10-14 | — (new mechanism; spec revision 30, plan §4.10) | no |
| `force_anxiety_gain` | — | 2.0 | 2026-10-14 | — (new mechanism; spec revision 30, plan §4.10, the owner's choice of 2026-10-14) | no |
| `focus_compound_gain` | — | 2.0 | 2026-10-14 | — (new mechanism; spec revision 30, plan §4.10, the owner's choice of 2026-10-14) | no |
| `focus_compound_gain` | 2.0 | 10.0 | 2026-10-14 | the form became a shift among the children that adds no investment: learning-qa found the first form unbounded (peak 14,000 median, 4 million max, 20 seeds, 10 years, ad hoc), a ceiling bounded it but removed the lock-in, and the owner chose "Shift, gain 10" | no |
| `lock_in_turn` | — | 3.0 | 2026-10-12 | — (new mechanism; spec revision 24, B7, Claude's recommendation, approved by the owner 2026-10-16: lock-in's turn) | no |
| `development_band_projection` | — | 0.10 | 2026-10-16 | — (new mechanism; spec revision 31, R31.B7: `M7.E.1f`'s band opened by the parents' acts, the owner's value: "projection up to 10% negative") | no |
| `development_band_support` | — | 0.05 | 2026-10-16 | — (new mechanism; spec revision 31, R31.B7, the owner's value: "supportive, responsible behaviour attention - up to 5% growth") | no |
| `nodal_spike_intensity` | — | 1.0 | 2026-10-17 | — (new mechanism; spec revision 32, plan §4.12: `M4.C.1c`'s nodal-event spike, the owner's form of 2026-10-17, "cut ties, as TRIGGER") | no |
| `conflict_binding_rate` | — | 0.15 | 2026-10-22 | — (new mechanism; spec revision 36, group G1 `M1.D.2c`, the owner's decisions of 2026-10-19 where the row says so, else `[I]`) | no |
| `cutoff_stock_cap` | — | 1583.0 | 2026-10-22 | — (new mechanism; spec revision 36, group G1 `M1.D.2b`, the owner's decisions of 2026-10-19 where the row says so, else `[I]`) | no |
| `cutoff_threat_weight` | — | 0.00312 | 2026-10-22 | — (new mechanism; spec revision 36, group G1 `M1.D.2b`, the owner's decisions of 2026-10-19 where the row says so, else `[I]`) | no |
| `cutoff_decay_weeks` | — | 260 | 2026-10-22 | — (new mechanism; spec revision 36, group G1 `M1.D.2b`, the owner's decisions of 2026-10-19 where the row says so, else `[I]`) | no |
| `cutoff_floor_share` | — | 0.1 | 2026-10-22 | — (new mechanism; spec revision 36, group G1 `M1.D.2b`, the owner's decisions of 2026-10-19 where the row says so, else `[I]`) | no |
| `positive_bridge_dissipation` | — | 0.8 | 2026-10-22 | — (new mechanism; spec revision 36, group G1 `M1.D.2b`, the owner's decisions of 2026-10-19 where the row says so, else `[I]`) | no |
| `tie_repair_steps` | — | 3 | 2026-10-22 | — (new mechanism; spec revision 36, group G1 `M1.D.2c`, the owner's decisions of 2026-10-19 where the row says so, else `[I]`) | no |
| `grief_scale` | 20.0 | retired | 2026-10-22 | — (spec revision 36 sized grief from the Holmes–Rahe table, `M1.F.6a`, `M6.3g`; `grief_weight` replaces it) | no |
| `grief_weight` | — | 1.0 | 2026-10-22 | — (new mechanism; spec revision 36, `M6.3g`: grief's size is its table size times this) | no |
| `lcu_scale` | — | 6.9 | 2026-10-22 | — (new mechanism; spec revision 36, `M1.F.6a`, the owner's anchor of 2026-10-19) | no |
| `assertion_perspective_threshold` | 0.3 | retired | 2026-10-26 | — (spec revision 39, `M5.F.4`, the owner's D20 of 2026-10-24 and Q8 of 2026-10-25: perspective enters an `I-POSITION`'s form through the self-channel weight, and no threshold decides it; stage 3 step 8) | no |
| `reciprocal_contact_share` | — | 0.5 | 2026-10-26 | — (new mechanism; spec revision 39, `M4.C.1d`, the owner's D22 of 2026-10-24 and Q10 of 2026-10-25; stage 3 step 9. `[I]`: no source gives a share; 0 is revision 38's model (nothing returns) and 1 would return to the sender all the receiver took in, which no source states either; 0.5, the middle of that interval, set before any criterion ran under it (`M10.B.4`), not tuned to a criterion; its half-to-double sweep range, 0.25 to 1.0, reaches the bound, so the stage-end sweep reports each direction over that whole span) | no |
