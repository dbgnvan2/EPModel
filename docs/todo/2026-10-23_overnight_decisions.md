# The night of 2026-10-22/23 — decisions taken as recommended, for the owner's review

The owner's instruction, 2026-10-22, verbatim: "you can run all night - proceed based on your recommendations - do not
wait for my input." Every decision the night's work would otherwise have put to the owner was taken as Claude
recommended and recorded, where it lands, with the words "decided as recommended under the owner's instruction of
2026-10-22, pending the owner's review". This record is the one place that lists them all. Ticking the item means the
owner has reviewed the list; a decision the owner reverses becomes its own item or proposal.

- [ ] T-125 ⟨now⟩ **Owner review of the overnight decisions (2026-10-23):** twenty-eight decisions in four batches, all on
  `main` at the push of 2026-10-23 06:10 (`146c6bd`).

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
