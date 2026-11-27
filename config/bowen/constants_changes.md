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
> **Spec revision 71 (2026-11-26, T-297) changed no constant.** It adds the `coach_lending` class to `act_classes.md`,
> a list of act names (`STAY-IN-CONTACT`, `I-POSITION`, `OVERFUNCTION`, `TRIANGLE`): the moves of a sided coach that
> lend, which until now were all its moves, `CUTOFF` included.
>
> **Spec revision 70 (2026-11-25, T-295 and E2) adds one constant**, `nonverbal_reattribution_enabled` (1, `[I]`, a
> switch under `M4.C.6`): 0 removes only the by-channel discount, for `M11.C.52[by channel]`'s baseline arm (T-209).
> At its default no run changes. T-019's coach lending reuses `pseudo_self_transfer_gain` and
> `pseudo_self_relaxation_rate`; no other value changed.
>
> **Spec revision 69 (2026-11-24, T-269) changed no constant.** A divorce is built in its own week by `DivorceSource`.
>
> **Spec revision 68 (2026-11-23, T-261) changed no constant.** It adds the `capacity_consuming` class to
> `act_classes.md`, a list of act names holding the one act the code named (`CONFLICT`).
>
> **Spec revision 63 (2026-11-16, T-254) changed no constant in `constants.md`.** `criteria.md`'s `M11.C.10` `seeds`
> rose from 200 to 600 (the owner: "C.10 to 600 seeds"): a criterion setting, the seed count of a fixed-count
> criterion, not a model constant; no other value changed.
>
> **Spec revision 62 (2026-11-15, T-247 and T-248) changed no constant**, so it has no row here. The lenders' share of
> a conversion (`M1.A.5c`) is the converted amount split as `M6.I.4a`'s return splits it; it adds none.
>
> **Spec revision 61 (2026-11-14, T-243 and T-004) changed no constant**, so it has no row here. `M1.A.5c`'s conversion
> adds none (the fall is the rise, at most what is borrowed). `constants.md`'s note on `M1.A.4c`'s sweep and
> `act_classes.md`'s note on `M1.E.3`/`M1.E.4` were reworded; no value changed.
>
> **Spec revision 59 (2026-11-12, the engine batch, T-001 step 3) changed no constant's value.** Its one row retires
> `fast_tick_weeks` (1), removed from `constants.md`: no code read it, and `M3.A.1` makes the fast tick one week by
> definition, so it is a retired key that fails at load if written (T-219 (10)); the frozen snapshots keep it as they
> were taken. The register now rejects a non-finite or negative value. `act_classes.md` gains four classes moved out of
> the code with the members the code held (`reactive`, `iposition_abort`, `position_outside`, `position_inside`, T-219
> (11), (12)); they are lists of act names, not constants.
>
> **Spec revision 47 (2026-11-02; T-179, the owner's option (a), with T-177) changed no constant**, so it has no row
> here. `M11.C.48`'s readout (a first cutoff on a tie with no contact by its week left out of both arms and counted) is
> code. T-177 added a sentence to `constants.md`'s note on `hold_gain` (since revision 41 aborts are not the usual
> outcome, T-158); the value of `hold_gain` did not change.
>
> **Spec revision 44 (2026-10-31; T-161 (b) and T-165, the owner's D4 and D5) changed no constant**, so it has no row
> here. `M5.D.5`'s pull-up reads `pull_up_rate` as before and skips an external mover; `M11.C.21`'s scripted acts end at
> `t0` with its settings in `criteria.md` unchanged.
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
| `hold_gain` | — | 0.3 | 2026-10-07 | — (new mechanism; Phase C step 8). **Chosen against a requirement:** at 0.5 most movers held at first opposition (86 of 113 over 40 seeds), contrary to `M5.D.3`'s "the usual outcome"; at 0.3 most abort (62 of 110). Measured on the reference family with Ravi given systems perspective 1.0. Superseded: since revision 41 (T-133) aborts are not the usual outcome (over 192 seeds on the triad, 62 sequences abort at the first opposition and 66 hold), kept as a finding (T-158 (b), owner 2026-10-31) | no |
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
| `pattern_window` | 8 | 4 | 2026-10-10 | — (owner decision D1, 2026-10-10, `docs/archive/proposal_run_visualization_mapping.md`: the owner's pattern definitions; no criterion reads the pattern readouts) | no |
| `pattern_min_acts` | 2 | 3 (unit changed: acts on the tie, was acts per member) | 2026-10-10 | — (owner decision D1, 2026-10-10, `docs/archive/proposal_run_visualization_mapping.md`: the owner's pattern definitions; no criterion reads the pattern readouts) | no |
| `pattern_pole` | 0.5 | retired | 2026-10-10 | — (owner decision D1, 2026-10-10, `docs/archive/proposal_run_visualization_mapping.md`: the owner's pattern definitions; no criterion reads the pattern readouts) | no |
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
| `social_cues_enabled` | — | 0 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.1f`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_attention_contact_weight` | — | 0.10 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.1e`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_attention_impingement_weight` | — | 0.0 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.1e`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_approval_contact_weight` | — | 0.04 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.1e`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_approval_impingement_weight` | — | 0.08 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.1e`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_expectations_contact_weight` | — | 0.0 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.1e`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_expectations_impingement_weight` | — | 0.08 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.1e`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_distress_contact_weight` | — | 0.0 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.1e`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_distress_impingement_weight` | — | 0.06 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.1e`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `unmet_expectation_weight` | — | 0.05 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.1e`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_leak_gain` | — | 0.6 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.13`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_leak_expectations_share` | — | 0.5 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.13`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_leak_distress_share` | — | 0.5 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.13`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_tone_share` | — | 1.0 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.13`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_face_share` | — | 1.0 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.13`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_posture_share` | — | 0.5 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.13`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `nonverbal_weight_base` | — | 0.5 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.11`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `nonverbal_weight_anxiety_gain` | — | 0.3 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.11`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `nonverbal_weight_cap` | — | 0.9 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.11`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_sensitivity_floor` | — | 0.3 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.14`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_sensitivity_level_gain` | — | 0.6 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.14`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_bias_floor` | — | 0.02 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.14`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_bias_level_gain` | — | 0.15 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.14`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_bias_anxiety_gain` | — | 0.15 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.12`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_bias_belief_gain` | — | 0.3 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.12`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_spread_floor` | — | 0.05 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.14`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `cue_spread_level_gain` | — | 0.2 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.14`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `attention_load_gain` | — | 0.5 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.15`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `attention_load_scale` | — | 0.5 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M4.C.15`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `belief_cue_rate` | — | 0.1 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M9.9`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `belief_leak_gain` | — | 0.5 | 2026-11-08 | — (new mechanism; spec revision 55, T-202, `M9.9`: the social cues, behind `social_cues_enabled`, off by default; `[I]`, the owner's approval of the draft 2026-11-08) | no |
| `fast_tick_weeks` | 1 | retired | 2026-11-12 | — (spec revision 59, T-219 (10), `M3.A.1`: no code read it; the fast tick is one week by definition) | no |
| `coach_side_window` | — | 4 | 2026-11-17 | — (spec revision 64, T-021, `M8.3`: how long an external agent stays on a member's side, split from `triangle_activity_window` at the same value, so tuning `M1.C.3` no longer moves `M11.C.7`; `[I]`, Claude's recommendation, pending owner review) | no |
| `belief_anxiety_scale` | — | 10.0 | 2026-11-19 | — (new mechanism; spec revision 65, T-082, `M9.8`: a receiver's own anxiety biases its reading of a tie's tension upward; `[I]`, Claude's recommendation, pending owner review) | no |
| `cue_uncertainty_gain` | — | 1.0 | 2026-11-19 | — (new mechanism; spec revision 65, T-203, `M4.C.16`: the variance of a sender's cues scales the reaction to them, behind `social_cues_enabled`; `[I]`, Claude's recommendation, pending owner review) | no |
| `cue_uncertainty_window` | — | 12 | 2026-11-19 | — (new mechanism; spec revision 65, T-203, `M4.C.16`: the ticks that variance is read over; `[I]`, Claude's recommendation, pending owner review) | no |
| `parent_pull_rate` | — | 0.02 | 2026-11-19 | — (new mechanism; spec revision 65, T-204, `M7.E.1g`: a parent's togetherness pull toward a child eases with the child's age, more slowly with the parent's chronic anxiety; `[I]`, Claude's recommendation, pending owner review) | no |
| `reactive_extra_max` | — | 3.0 | 2026-11-19 | — (new mechanism; spec revision 66, `M4.D.3c`: anxiety raises the number of reactive acts a week; `[I]`, the owner's approval 2026-11-19 of `docs/proposal_reactivity_conflict_divorce.md`; 0 restores revision 65) | no |
| `reactive_half_anxiety` | — | 100.0 | 2026-11-19 | — (new mechanism; spec revision 66, `M4.D.3c`; `[I]`, `M4.D.1h`'s force midpoint) | no |
| `conflict_capacity_cost` | — | 1.0 | 2026-11-19 | — (new mechanism; spec revision 66, `M4.D.3d`: conflict consumes capacity; `[I]`, the owner 2026-11-19) | no |
| `importance_max` | — | 20.0 | 2026-11-21 | — (new mechanism; spec revision 67, T-259, `M5.D.3b`: a position that is not a line draws its importance from [0, this], in points of conviction; `[I]`, Claude's pick under the owner's approval of `docs/proposal_conviction_and_coach_sessions.md` 2026-11-21, pending owner review) | no |
| `line_importance` | — | 60.0 | 2026-11-21 | — (new mechanism; spec revision 67, T-259, `M5.D.3b`: the importance of a position that is one of the person's lines; `[I]`, Claude's pick under the owner's approval of `docs/proposal_conviction_and_coach_sessions.md` 2026-11-21, pending owner review) | no |
| `line_tendency_max` | — | 0.2 | 2026-11-21 | — (new mechanism; spec revision 67, T-259, `M5.D.3b`: each person's line tendency is drawn once from [0, this]; `[I]`, Claude's pick under the owner's approval of `docs/proposal_conviction_and_coach_sessions.md` 2026-11-21, pending owner review) | no |
| `tie_discount` | — | 0.5 | 2026-11-21 | — (new mechanism; spec revision 67, T-259, `M5.D.3b`: conviction falls by this share at bond energy 100 on the tie to the target; `[I]`, Claude's pick under the owner's approval of `docs/proposal_conviction_and_coach_sessions.md` 2026-11-21, pending owner review) | no |
| `hold_gain` | 0.3 | 0.3 | 2026-11-21 | — (spec revision 67, T-259, `M5.D.3b`: the unit changes from per point of functional level to per point of conviction; the value is kept, not tuned) | no |
| `nonverbal_reattribution_enabled` | — | 1 | 2026-11-25 | — (new switch; spec revision 70, T-209, `M4.C.6`: 0 removes only the by-channel discount, for `M11.C.52[by channel]`'s baseline arm; at its default no run changes; `[I]`, Claude's recommendation 2026-11-25, pending owner review) | no |
