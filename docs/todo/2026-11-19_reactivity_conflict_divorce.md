# Spec revision 66 — reactive amount, conflict as consumption, divorce (2026-11-19)

Built as revision 65 on the branch `claude/reactivity-divorce`, merged as revision 66 on the final engine batch, from `docs/proposal_reactivity_conflict_divorce.md`, approved by the owner
2026-11-19. Open:

- [ ] T-258 ⟨now⟩ **Regenerate every record on revision 66's engine (2026-11-19).** The batch moved the engine hash, so
  the ensemble, mutation, sweep, level-occupancy, diagnostic, Phase D (parts I and II), Phase D mutation, emergence
  audit and sensitivity records, and the LLM experiment's pilot cache and report, are stale. Each test that reads one is
  a strict expected failure naming T-258; remove each mark as its record is regenerated (the test then passes, or turns
  red as an unexpected pass). Criteria that read act counts or conflict (`M11.C.1`, `M11.C.16`, `M11.C.19`, the
  triangle and cutoff criteria) may change verdict: report each move to the owner, with T-254's seed-block rule. This one regeneration also serves revision 65 (the final engine batch): its own
  run on Mac Mini1 was stopped after Phase D part II and part I (the owner, 2026-11-19), and those two records are rerun
  here.
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
