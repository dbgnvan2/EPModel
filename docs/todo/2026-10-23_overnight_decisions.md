# The night of 2026-10-22/23 — decisions taken as recommended, for the owner's review

The owner's instruction, 2026-10-22, verbatim: "you can run all night - proceed based on your recommendations - do not
wait for my input." Every decision the night's work would otherwise have put to the owner was taken as Claude
recommended and recorded, where it lands, with the words "decided as recommended under the owner's instruction of
2026-10-22, pending the owner's review". This record is the one place that lists them all. Ticking the item means the
owner has reviewed the list; a decision the owner reverses becomes its own item or proposal.

- [x] T-125 ⟨now⟩ **Owner review of the overnight decisions (2026-10-23):** fifty-three decisions in six batches; batches
  1–4 on `main` at the push of 2026-10-23 06:10 (`146c6bd`), batch 5 with the part I sweep's merge of 2026-10-23,
  batch 6 with part II's declared ranges (`c955620`).
  *Approved in full by the owner 2026-10-24; decision 27: the branch `claude/confident-maxwell-vr5zol` is kept.*

  **Batch 1 — group G and the Holmes–Rahe event sizes, spec revision 36** (branch `claude/recursing-bohr-2393a1`,
  commits `e52adbc`, `0b02fd3`, `da4e512`, `7b1b622`; engine hash `2842f0e8fea8` → `66dcbb1e8ba5`; every record
  regenerated once, 2 h 36 min; `docs/version_2_revision_36_changes.md` holds the changed results beside version 1's):
  1. G2 Q1: a released symptom hold is split by bond energy, not conductance (`docs/PROPOSAL — RELEASE BY TIE (G2).md` §8).
  2. G2 Q2: no release reaches an external agent such as the coach; family members only (same §8).
  3. G2, death: a dead member's tie takes no share; with no member on an interactive tie the hold stays with the bearer
     (spec `M7.D.2a`).
  4. Event sizes Q3, asked again as the stage 1 review's D10 required: E2 (additions carry load for 12 weeks) stands as
     the owner decided it on 2026-10-19, cites no source and is graded `[I]`; the correction that `KS07.1` shows an
     addition moving functioning, not adding load, is kept beside it (the event-size proposal's status line).
  5. `M11.C.48`'s baseline: the run standing on each uncut tie in the cutoff's week, averaged, not the span-longest run
     (G1 proposal §4 as-built note; `docs/readouts_phase_c.md`).
  6. The contact classes, Claude's reading of G1's D4–D6 and D12 (`config/bowen/contact_classes.md`).
  7. Nine `[I]` constants: `conflict_binding_rate` 0.15, `cutoff_stock_cap` 1583, `cutoff_threat_weight` 0.00312,
     `cutoff_decay_weeks` 260, `cutoff_floor_share` 0.1, `positive_bridge_dissipation` 0.8, `tie_repair_steps` 3,
     `lcu_scale` 6.9, `grief_weight` 1.0; `grief_scale` retired (`docs/spec_revision_log.md`, revision 36, Constants).
  8. Removed: `REDUCE_CUTOFF`'s transfer of the bound stock to third members, and the restoration of `uncut_state` when
     rejection fades (revision log, Removed).
  9. G1's A6 left not shown, as a strict expected failure (`test_m5b3_third_party_cost_is_the_lost_outlet`); no cost was
     written (revision log, Not shown).
  10. The build's two review fixes before the hash moved (revision log, "From the build's two reviews").
  Findings of this batch, not decisions, each its own item: five composite criteria changed verdict against version 1
  (T-122: `M11.C.44` FAIL→PASS; `M11.C.17`, `M11.C.7[joint over one spouse]`, `M11.C.10[family of origin]`,
  `M11.C.2[twelve-person family]` PASS→FAIL); three named mutants survive (T-119 to T-121); `M11.C.48`'s recorded
  PASS is not evidence, a readout fix for the next hash move (T-123).

  **Batch 2 — stage 2's plan and pilot, spec revision 37** (branch `stage-2-plan-pilot`, commits `497e7ba`, `36069fd`,
  `9475d59`, `44ef318`; hash unchanged; `docs/implementation_plan_version_2.md` §2):
  11. D8, the detection-limit rule: a sweep sample whose first run leaves any gating directional effect within 2.0
      half-widths of zero is rerun at a fixed 200 seeds; the verdict is the second run's; both counts are recorded; a
      criterion with its own seed count below 200 is refused (spec `M17.A.1`, revision 37; `config/sweep/sweep.md`).
  12. D9, the pilot's subjects: `M11.C.16` (passing) and `M11.C.42` (an accepted finding).
  13. D10, the sweep's scope and ranges: the `[I]` constants a criterion's mechanism reads, half to double the central
      value with the declared bounds, joint seeded draws, 8 samples per criterion.
  14. D11, the sweep record and the version 2 status page: a fraction "n of S held", never a verdict; a verdict changed
      since the tag is not final (`docs/version_2_sweep_record.md`, `docs/version_2_status.md`).
  15. D12, the sweep's config at `config/sweep/sweep.md`, outside the engine hash.
  16. D13, no further `M17` requirement for stage 2.
  17. D14, the narrow mark (under half of at least 32 counted samples), defined, not built.
  18. D15, the full sweep not run: the pilot projected 44 h at 8 samples (part II alone 40 h); Phase C's 14 criteria
      next (projected 56 min), then part I's 13 (3.2 h) in a session of its own; part II's seven need a faster arm or a
      smaller sample, the owner to choose.
  19. D16, none of the drafted `M17.A.2`, `M17.B.1`, `M17.E.3`, `M17.E.6` comes in.
  The pilot's measurement: `M11.C.16` held in 8 of 8 samples; `M11.C.42` in 0 of 8 (it fails at the central values too).

  **Batch 3 — the Phase C constant sweep** (branch `stage-2-phase-c-sweep`, commit `146c6bd`; hash unchanged; 14
  criteria, 155 child runs, 27 min against 48 min projected; `config/sweep/sweep.md`'s preamble and plan §2.10):
  20. The swept set is the Phase C ensemble record's composite and mixed rows, 14, at 8 samples.
  21. The learner's seven constants are in a criterion's set where its row or its `M11.5` class row names learning or the
      readout counts selected acts over the run (`M11.C.5`, `M11.C.45`, `M11.C.44`), and out where the readout is one week
      after a scripted act (`M11.C.27`) or the row credits no learned mechanism (`M11.C.29`, `M11.C.41`).
  22. `M11.C.29`'s set: `exchange_budget_reduction`, the third's symptom onset and end constants, `defence_threshold`.
  23. `M11.C.41`'s set: the mixing weight, the bands, the stress response's two gains and floor, `acute_decay_rate`;
      `response_relaxation_rate`, `usual_level_rate` and the perspective gains left out.
  24. `M11.C.45` leaves out `M4.C.1`'s seven (swept under `M11.C.44`) and `triangle_activity_window`.
  25. `nodal_spike_intensity` and `functional_level_floor` are in no set.
  26. `anxiety_band_high`'s low end is 10, the top of `anxiety_band_low`'s range, so every joint sample keeps low ≤ high.
  The fractions held are in `docs/version_2_sweep_record.md`; four criteria held in 0 of 8, two in 8 of 8.

  **Batch 4 — the coordinating session's own calls:**
  27. A cloud session the owner started on 2026-10-22 drafted a second G2 proposal, `PROPOSAL — SYMPTOM RELEASE BY TIE
      (G2).md`, on the branch `claude/confident-maxwell-vr5zol` (Q1–Q5 open; the same recommendations as the built one).
      It was not merged, since the built proposal supersedes it; its SessionStart hook for cloud sessions was
      cherry-picked (`2cad4de`). The branch is left on GitHub for the owner to delete or keep.
  28. `CHANGELOG.md` passed its 80,000-byte limit and the four entries of 2026-10-19 and 2026-10-20 moved to the
      archive verbatim (`e3ca2b6`), after a push that went out with that test red (`ad361d7`) and was fixed in the
      next commit.
  The stage 2 planning session requested in the cloud at about 22:30 on 2026-10-22 had pushed nothing by 03:40; the
  plan was written here instead. If a branch from it appears on GitHub, it is superseded.

  **Batch 5 — the Phase D part I constant sweep** (branch `stage-2-part-i-sweep`; hash unchanged;
  13 keys, 182 child runs, 3 h 34 min of run time against 1.8 h and 3.2 h projected; `config/sweep/sweep.md`'s preamble and plan §2.10). The owner's instruction of 2026-10-23 was to run
  it; the owner chose to have its judgment calls decided as Claude recommended and listed here.
  29. The swept set is the Phase D part I record's composite and mixed rows, 13 keys (`M11.C.7`'s three pairs each a
      key), at 8 samples. Part II's seven stay out (D15).
  30. The learner's seven are in `M11.C.2`, `M11.C.12`, `M11.C.34`, `M11.C.39` and `M11.C.43`, whose rows or class rows
      name learning or a readout of selected acts, and out of the other eight.
  31. `M11.C.7`'s three pairs share one set: `M4.C.1`'s seven, `nodal_spike_intensity`, and the coach's
      `landing_rate`, `perspective_gain` and `binder_failure_fraction`, which every arm reads at the same value.
  32. `nodal_spike_intensity` is in `M11.C.7` and `M11.C.17`, whose arms run nodal events; it stays out of Phase C's
      sets (batch 3, decision 25).
  33. `M11.C.2`'s set adds the focus constants (`perceived_weakness_gain`, `focus_steer_gain`, `focus_reinforce_gain`,
      `focus_compound_gain`), `investment_leak_rate`, `witness_weight` and `symptom_leak_rate`; `birth_focus_gain` is out,
      since the part I scenarios have no births.
  34. `M11.C.11`'s set: lock-in's gain and turn, the symptom's threshold, re-arm and leak, `standing_load_gain`, the
      contact relaxation pair and `acute_decay_rate`.
  35. `lock_in_turn` is out of `M11.C.12` and `M11.C.22`: each arm's held severity is declared against the turn, so
      moving it would change the arm rather than the mechanism.
  36. `M11.C.17` leaves out `estimator_window` and `estimator_breadth`: its nodal load is declared to meet them.
  37. `M11.C.21`'s set: the pseudo-self transfer and relaxation, `exchange_window`, the stress response's two gains,
      its relaxation and its floor.
  38. `M11.C.22`'s set: `lock_in_gain`, `standing_load_gain`, the contact relaxation pair and `acute_decay_rate`.
  39. `M11.C.23`'s set: the appraisal pair, `impingement_relaxation_rate`, `investment_leak_rate`, the symptom's
      threshold, re-arm and leak, and `acute_decay_rate`. The arm's `suppression` (0.5 in `config/bowen/criteria.md`) is
      an arm setting, not a constant, and is not swept.
  40. `M11.C.34`'s set: the learner's seven, `self_channel_exponent`, `capacity_level_per_layer` and
      `symptom_event_intensity`; `M1.A.19`'s `reactive_rate` is out, since no selection in the criterion reads its
      detectors.
  41. `M11.C.39`'s set adds `witnessed_cutoff_gain`, `witness_window` and `distance_binding_rate`; `witness_window`'s low
      end is 7, half of its central 13 rounded up to a whole week.
  42. `M11.C.43`'s set: the learner's seven, `calm_transfer_rate` and `acute_decay_rate`.
  43. Bounds: `binder_failure_fraction` (central 0.8) runs 0.4 to its bound 1.0, not to double; the perspective gains
      are capped at 1: above 1 the channel weight passes 1 and the steering factor can turn negative
      (`src/bowen/policy/policy.py`, lines 193 and 289).
  44. The sweep's first run raised `KeyError` on all 104 part I samples in 81 s: `tools/mutant_runner.py`'s child
      knows only Phase C's criteria. A criterion of a Phase D record is now run by `tools/phase_d_mutant_child.py` and
      cached under that record's hash (the child and the source of `apply` and `run_in_copy`), which the sweep record
      prints as a second line, `phase_d_engine_hash`; Phase C's results kept their keys and were not rerun.
  The fractions held are in `docs/version_2_sweep_record.md`; four keys held in 0 of 8, two in 8 of 8. Reruns at 200
  seeds took 54% of the seeds; the owner kept D8's 200 seeds on 2026-10-23 when told so.

  **Batch 6 — the Phase D part II ranges, declared and not run** (on `main`, `c955620`; hash unchanged; the owner
  decided on 2026-10-23 to declare only; `docs/implementation_plan_version_2.md` §2.10, "Part II's ranges declared",
  holds each judgment in full):
  45. The learner's seven are in `M11.C.2[twelve-person family]`, both `M11.C.10` variants and both `M11.C.48`
      variants, and out of `M11.C.6` and `M11.C.15`, whose gating readouts are states and whose rows credit no learner.
  46. `birth_focus_gain` is in `M11.C.2[twelve-person family]`: it acts at the third generation's births, and in seed 1
      Nadia's first child is born inside the ages the readout averages.
  47. `nodal_spike_intensity` is in no part II set, though it acts in the twelve-person runs: no part II row credits
      `M4.C.1c` (the owner kept this, 2026-10-23, T-126).
  48. `M11.C.6` takes `M7.E.1f`'s four development constants and the estimator's `estimator_sd_weight` and
      `estimator_rate`; the estimator's window and breadth and the two development ages are out.
  49. `development_band_support`'s low end is 0.04, the top of `development_band`'s range, for the engine's invariant.
  50. Both `M11.C.10` variants take `M1.B.3c`'s five rejection constants (the readout, weeks of held rejection, is
      that derived state); `declared_rejection` and `refusal_cost_gain` are out; `witness_window` is in at 7 to 26.
  51. `M11.C.15` takes `lock_in_gain` and `acute_decay_rate` only; `lock_in_turn`, `exchange_budget_reduction` (no
      differentiating exchange completed in 20 seeds of its three runs), `death_transfer_share` and the grief and
      income constants are out.
  52. Both `M11.C.48` variants take `M1.D.2b`'s five cutoff constants; `M1.B.3c`'s are out.
  53. The sweep config gained a *Declared, not run* list: a listed key is shown in the record with its ranges and no
      number, runs only when `--criteria` names it, and stops the record once it has a cached result.

- [x] T-126 ⟨now⟩ **Should `nodal_spike_intensity` be swept where the nodal load acts?** Phase D's `M11.C.2` (part I)
  and `M11.C.2[twelve-person family]` (part II) both run under the nodal load, and the spike acts there (662 spikes in
  seed 1's 40 twelve-person years, 2026-10-23). Neither set holds it, because no row of theirs credits `M4.C.1c`
  (decisions 25, 32 and 47). `M11.C.7` and `M11.C.17` do hold it. Owner's question: add it to both `M11.C.2` sets
  (one range row each in `config/sweep/sweep.md`; part I's `M11.C.2` would then be rerun at 8 samples, part II's runs
  with the rest of part II), or keep D10's rule that only a credited mechanism is swept. No code change either way.
  *Closed 2026-10-23: the owner, verbatim: "keep D10's rule; nodal_spike_intensity stays out of both M11.C.2 sets."
  Decisions 25, 32 and 47 stand as to this constant; nothing in `config/sweep/sweep.md` changed.*

- [x] T-127 ⟨later⟩ **Two status lines in `config/bowen/` still say T-125's decisions are pending.**
  `config/bowen/contact_classes.md` and `config/bowen/event_sizes.md` state their choices as pending the owner's
  review; the owner approved them on 2026-10-24 (T-125). The engine hash covers every `*.md` under `config/bowen/`
  (`tools/ensemble_record.py::code_hash`), so changing those lines alone would make every record stale (tried and
  reverted 2026-10-24). Fix them in the same commit as the next change that moves the engine hash, which reruns the
  records anyway.
  *Closed 2026-10-25 at stage 3 step 6: both lines say approved by the owner 2026-10-24 (T-125).*
