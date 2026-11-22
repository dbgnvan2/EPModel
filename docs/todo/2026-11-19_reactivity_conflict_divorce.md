# Spec revision 66 — reactive amount, conflict as consumption, divorce (2026-11-19)

Built as revision 65 on the branch `claude/reactivity-divorce`, merged as revision 66 on the final engine batch, from `docs/proposal_reactivity_conflict_divorce.md`, approved by the owner
2026-11-19. Open:

- [x] T-258 ⟨now⟩ **Regenerate every record on revision 66's engine (2026-11-19).** The batch moved the engine hash, so
  the ensemble, mutation, sweep, level-occupancy, diagnostic, Phase D (parts I and II), Phase D mutation, emergence
  audit and sensitivity records, and the LLM experiment's pilot cache and report, are stale. Each test that reads one is
  a strict expected failure naming T-258; remove each mark as its record is regenerated (the test then passes, or turns
  red as an unexpected pass). Criteria that read act counts or conflict (`M11.C.1`, `M11.C.16`, `M11.C.19`, the
  triangle and cutoff criteria) may change verdict: report each move to the owner, with T-254's seed-block rule. This one regeneration also serves revision 65 (the final engine batch): its own
  run on Mac Mini1 was stopped after Phase D part II and part I (the owner, 2026-11-19), and those two records are rerun
  here.
  *Closed 2026-11-20: Mac Mini1's queue (`~/regen66.sh` at `0aece2f`) ran 13 steps; 12 exited 0. The emergence audit
  exited 1 because its pilot row `M11.C.48[phase_d]` no longer passes; the row left `config/audit/emergence_audit.md`'s
  pilot table and the audit was rerun alone. Every record but the LLM experiment's is regenerated and its mark removed;
  the verdict moves are `docs/version_3_revision_66_changes.md`, for the owner in T-264. The LLM runs do not replay:
  T-265.*
- [ ] T-259 ⟨now⟩ **No I-POSITION sequence completes in the triad at revision 66 (2026-11-19).** With `M4.D.3c`'s extra
  reactive acts, `tests/bowen/test_iposition.py::attempts_over_seeds` (192 seeds, mover at level 60) draws no
  completed sequence (27 at revision 30), so `test_m5d8_success_usually_follows_several_failures` is a strict expected
  failure; `test_m5d3_abort_is_the_usual_first_outcome`, a strict expected failure since revision 41 (T-158), passes
  again. For the owner: whether a sequence in progress should shield the mover from extra reactive acts toward the
  sequence's other (they are already made as STAY-IN-CONTACT, `M5.D.9`, which makes an automatic extra a
  self-directed act that takes a stance and opens a `self_practice` window, contrary to `M4.D.3c`'s automatic channel;
  the correctness review of 2026-11-19), or the fixture's mover level should rise, or
  the finding stands.
- [x] T-260 ⟨now⟩ **Run the predictions registered in the proposal's §4 (2026-11-19).** P1–P4 on the probe of
  2026-11-19, no constant tuned afterwards; report each as met or not met.
  *Closed 2026-11-19: run the same day on the Cowork build (revision 64 plus this change, before the merge), no constant tuned (720 runs). P2 met; P1 met for intensity,
  not for counts; P3 not met (r = .09 [−.02, .20]); P4 not met. The proposal's §4.1 has the table.*
- [ ] T-261 ⟨later⟩ **Revision 66's minor review items (2026-11-19).** From the learning-qa and correctness passes, not
  fixed: (1) the `_apart` household suffix (`event_effects.py`) and the `"CONFLICT"` kind name (`stress_response.py`)
  are literals; a set of capacity-consuming kinds belongs in `event_kinds.md` (P4). (2) `reactive_extras` stops early
  when nobody is left to act toward, and nothing records the drawn count against the made one (P2). (3)
  `divorce_events` does not check that either spouse is alive. (4) `consume_conflict` charges a `CONFLICT` that is
  refused or never lands; whether it should is the owner's call. (5) `divorce_events` treats any non-spouse in a
  parent–child tie as a child, correct only because the loader keeps nuclear ties. (6) No test runs a divorce through
  `run_tick` and the trace renderer.
- [ ] T-262 ⟨now⟩ **An external agent makes an `I-POSITION` sequence step between sessions (2026-11-19).** At revision
  66, in `tests/experiments/llm_agents/test_source.py::test_lx2_external_agent_keeps_the_policy` (stub run A1, seed 1,
  208 weeks), the coach's policy chooses `I-POSITION` at a session and `M5.D.9`'s sequence makes its `DEFINE` step
  at week 124, which is not a session week; `M1.E.8` gives an external agent no outcome between sessions. The
  combined trajectory reaches a path neither revision 65 nor the Cowork build alone did; the gap is older. The test is
  a strict expected failure. For the owner: whether an external agent's sequence steps wait for its next session, or
  an external agent never opens a sequence.
- [x] T-263 ⟨now⟩ **The extra acts raise a two-year anxiety peak, and why P4 failed (2026-11-19, corrected 2026-11-20).**
  Revision 66's extra acts (`M4.D.3c`) do not make anxiety climb without bound, as this item first said: with no
  scripted events (Phase C family, 312 weeks, 20 seeds, a grid of `reactive_extra_max` 0–3, `reactive_half_anxiety`
  100 and 300, `conflict_capacity_cost` 0 and 1) the members' mean excess peaks in years 2–3 and is back by year 6 in
  every setting. The constants set the peak's height: at level −20, about 1,500 at the current constants against 170
  with both off (146 and 24 at level 0). The probe of T-260 read weeks 20–103, the rising phase; there Marta's excess was
  327 against 33 at revision 64, and one run reached 9408 and underflowed `M4.D.1d`'s entropy (fixed in `0aece2f`).
  Bounding a no-event run is already met, so no constant was re-set: a magnitude target would be invented.
  P4 (a divorce raises the former spouses' conflict) fails at week 20 and again at week 260, after the peak (40 seeds,
  52 weeks each side, no spell): divorced minus not, the spouses' conflict is −0.006 ± 0.014 a week at level 0 and
  +0.000 ± 0.010 at level −20 (2 se). The cause is that the marital tie is cut off before the divorce: in 99% of weeks
  20–103 and in every week of year 5, with or without it, and the spouses' conflict is then near zero (0.02 a week).
  The divorce's parts, each run alone at week 20: the `DIVORCE` load lands (+10 to 15 acute anxiety for 12 weeks) but
  leaves the window's mean unchanged; the `SEPARATION` lowers every member's excess by about 40% (by more than half
  for Ravi at week 260), because the leaver no longer witnesses or is witnessed in the household. The sinks reading
  the old marital tie and the load reaching the children are not the cause; a contact channel through the children
  would not raise conflict on a tie already quiet. For the owner: whether the spouses' cutoff within two years in
  almost every run is the model's finding or a defect to trace (it is 28–50% of weeks 0–19 at revision 64 and 52–72%
  at 66), before P4 is read again.
  **Traced 2026-11-20** (Phase C family, no events, 156 weeks, 20 seeds a level). The cutoff is older than revision
  66: at revision 64 every run cuts the marital tie (median week 16 at level 0, 9 at −20), by a spouse's own `CUTOFF`; it is cut
  80% of the time at level 0 (92% at −20). Entry and exit are unequal: one `CUTOFF` (rejection 0.5) cuts the tie
  (`M1.B.3c`), and it reopens only when rejection has faded to zero, about ten quiet weeks, and then a positive
  contact is accepted (`M1.D.2b`); the only positive contacts are `STAY-IN-CONTACT` and `REDUCE_CUTOFF`, both `self`
  channel, which the spouses send across the cut tie 0.002–0.015 times a week. At revision 64, with `CUTOFF` across
  the cut tie at 0.06 a week, it reopens 5 times in 60 run-years at level 0; at revision 66 the spouses send it 0.59 a
  week (0.17 their own choice, 0.43 extra acts, which are automatic only and so never the repair), and it never
  reopens. Removing `CUTOFF` across a cut tie from the extra acts alone leaves it cut 91% with no reopening, the own
  choice staying at 0.18. Raising every level by 40 lowers the cut share (65%, against 29% at revision 64), so the
  direction by level holds. For the owner: whether a cut whose rejection has faded ends without a `self` contact
  (revision 36 decided it does not), or the spouses' cutoff stands as the model's result and P4 is dropped.
  **Decided 2026-11-20 (the owner): revision 36's rule is kept; the spouses' cutoff stands as the model's result and
  P4 is dropped.** No rule or constant changes.
- [x] T-264 ⟨now⟩ **Revision 66's verdict moves (2026-11-20).** From `docs/version_3_revision_66_changes.md` (T-258), for
  the owner. Ensemble rows, PASS to FAIL: `M11.C.18`, `M11.C.20` and `M11.C.33` (`rise` is +0 ± 0 in every pair; in
  `M11.C.20` 783 of 900 treatment moves are not legal), `M11.C.22` (the severe removal's effect turns positive),
  `M11.C.15` (`death_rise` −109), `M11.C.10[family of origin]`, `M11.C.2[twelve-person family]` and
  `M11.C.48[phase_d]`; FAIL to PASS: `M11.C.6`, `M11.C.23`, `M11.C.27[unstable,add_third]`. Under T-254's rule these
  are not yet read as changes of the model, a p on one side lying between about 0.01 and 0.2: `M11.C.10` (p 0.031 to
  0.80), `M11.C.17` (0.018 to 0.10), `M11.C.49` (to 0.21), `M11.C.12` (to 0.042), `M11.C.44` (to 0.036), `M11.C.51`
  (to 0.048), and the Phase D mutants on `M11.C.46` (0.17), `M11.C.7[one spouse over sided]` (0.017) and `M11.C.55`
  (0.049); fresh, disjoint seed blocks (`tools/t253_seed_blocks.py`, on Mac Mini1) decide them. Also: the
  `availability-level-inverted` mutant's treatment arm is 84% of member-weeks in the every-layer deletion band (98.5%
  in C.16's window, against 40% and 44% at revision 64), so `test_m111a_availability_inversion_acts_over_c16s_run` is
  a strict expected failure: the inversion now acts as a deletion.
  **Traced 2026-11-21** (probes on the triad fixture, seeds 0–9, both engines; no rule or constant changed).
  *The zero rise in `M11.C.18`, `M11.C.20` and `M11.C.33`.* At revision 66 F's load-weighted functional level over
  the estimator's window is about 27 against a basic level of 40, so the evidence (`M1.A.4k`) is negative in all 20 of
  F's readings in every arm (at most −8.9; at revision 64 it straddles 0, at most +0.46) and no arm ever rises. Both
  arms read `rise` 0 in every seed, so the criteria have nothing to compare and their failure says nothing about
  `I-POSITION` or borrowing. The harness is sound: with `conflict_capacity_cost` 0 for the arm (a scratch override)
  the treatment arm rises in 10 of 10 seeds in all three (difference +0.30 to +0.36), and with only
  `reactive_extra_max` 0 no arm rises, so `M4.D.3d`'s cost, enlarged by `M4.D.3c`'s extra `CONFLICT`s, is the cause.
  *The cut tie.* The F–M tie is cut in most weeks on both engines (seed 0: 198 of 208 weeks at revision 66, 187 at
  revision 64), so most scripted `I-POSITION`s are not legal on both (revision 64's record: 4180 of 5250 in
  `M11.C.18`). Revision 66 raises that share (`M11.C.20`'s 783 of 900), but with the cost off the treatment arm rises
  in every seed with 332 of 350 scripted acts still not legal. The cut ties are not what zeroes the rise.
  *`availability-level-inverted`.* The same fall puts the unmodified `M11.C.16` treatment arm at functional level 20
  or below in 84.7% of member-weeks (98.6% in the window), against 42.3% (46.4%) at revision 64
  (`docs/phase_c_level_occupancy.md`). In that band the inverted rule equals the deletion on every layer, so the
  mutant acts as a deletion because the engine's levels fell, not because of the mutant. For the owner: whether
  `M4.D.3d`'s cost is meant to hold functioning this far below basic level for years, since the estimator cannot
  raise anyone in that state.
  **Seed blocks 2026-11-21** (`docs/review/2026-11-21_t264_seed_blocks.md`, from `tools/t264_seed_blocks.py` on Mac
  Mini1: fresh disjoint blocks of the records' size from seed 40000, revisions 64 and 66; pooled p over the blocks).
  *`M11.C.12`, FAIL to PASS: an effect.* Pooled over 300 fresh seeds `marital_conflict` rises by +1.47 ± 0.73 (p
  0.0008) at revision 66 and by +0.02 (p 0.36) at revision 64, where no block passes. At the record's 100 seeds only 1
  of 3 revision-66 blocks passes, so the effect is new but the verdict at that size is borderline.
  *`M11.C.17`, PASS to FAIL: not a change of verdict; the effect weakens.* Revision 66 passes 2 of 3 blocks (pooled
  `coached_rise` +0.0016, p 0.007) and revision 64 3 of 3 (+0.0029, p 6e-7). The record's FAIL is the threshold; the
  effect about halves.
  *`M11.C.44`, FAIL to PASS: not an effect of revision 66.* Both engines pass 2 of 3 blocks with pooled p at most
  0.0014, so revision 64's FAIL (p 0.71) was seed noise. Revision 66's effect is about twice revision 64's (+0.17
  against +0.08).
  *`M11.C.49`, PASS to FAIL: the effect weakens and keeps its direction.* Revision 64 passes 3 of 3 blocks
  (`family_excess` +4.0, p 2e-7); revision 66 passes 1 of 3, pooled `family_excess` +2.7 (p 0.008) and
  `conflict_share` +0.022 (p 0.004). The smaller effect is real; at 150 seeds the verdict is unstable.
  *`M11.C.51`, FAIL to PASS: noise.* Revision 66 passes none of 3 blocks (pooled p 0.15), nor does revision 64 (p
  0.80). The record's PASS at p 0.048 does not hold.
  *`triangle-help-ignored` on `M11.C.46`, reversed to red: the catch holds, the label is noise.* The mutant is caught
  in all 6 blocks on both engines. Revision 64's "reversed" came from a negative mean that its fresh blocks do not
  show (pooled +0.13).
  *`coach-position-blind` on `M11.C.7[one spouse over sided]`, red to survived: an effect.* At revision 66 both gating
  readouts hold pooled (`deviation_fall` +0.092, p 0.008; `landed_contacts` +1.6, p 3e-11), so the criterion passes
  under the mutant; at revision 64 `deviation_fall` is −0.03 (p 0.95) and the mutant is caught. No fixed block reaches
  PASS in 100 seeds (each ends UNDETERMINED, short of the precision rule), so the record's survival at 100 seeds sits
  at that rule's edge.
  *`supply-by-birth-order` on `M11.C.55`, red to survived: an effect.* Revision 66 passes 3 of 3 blocks under the
  mutant (pooled `predicted_main_supplier` +0.32, p 9e-14) and revision 64 none (−0.56, p 1). The record's p 0.049
  understated it: at revision 66 `M11.C.55` no longer catches the mutant.
  *`M11.C.10`, PASS to FAIL: an effect.* On one fresh block of 600 seeds (one block, not three: a 600-seed block takes
  1.75 to 3.7 hours on Mini1) `cut_off_tie_weeks` differs by −53 ± 59 (p 0.98) at revision 66 and +42 ± 76 (p 0.09)
  at revision 64. Revision 66's difference is negative on the record's seeds too (−30), and revision 64's positive
  (+67), so the sign changed with the engine; revision 64's own pass was borderline (T-253).
  *Closed 2026-11-21 (owner): the finding stands. `M4.D.3d`'s cost, with `M4.D.3c`'s extra `CONFLICT`s, holding
  functioning below basic level is the model's result at revision 66; no rule or constant changes. The verdict moves
  above stand as the records show them.*
- [ ] T-265 ⟨now⟩ **The LLM experiment's runs do not replay on revision 66's engine (2026-11-20).** The replay in T-258
  refused every committed run: the A0 baseline's behavior changed with the engine, and the A1–A4 runs' rendered prompts
  changed, so their cached responses miss. The committed runs were left as they were (their `run.json` would read
  `not_replayable`), and `test_lx16_report_is_current` and `test_lx25_committed_pilot_replays_fixed` stay strict
  expected failures naming this item. For the owner: whether the A0 baseline is rerun on the new engine and whether
  the LLM arms are run again live (a paid run), or the experiment is frozen at revision 64.
