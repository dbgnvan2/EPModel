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
