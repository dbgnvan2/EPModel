# Blocking the next phase

- [x] T-001 ⟨now⟩ **Version 2, next:** ~~stage 0~~ *done 2026-10-19: the `version-1` tag at `d62cfdc`, pushed, with the status
  page frozen and checked against it; the dispositions; the render-only fixes moved to stage 1 (owner).* Then stage 1
  (the engineering batch, with the grid engine retired, stage 0.3's two fixes, and the trace renderer out of the hash;
  *plan drafted 2026-10-19, `docs/implementation_plan_version_2.md` §1; D1–D7 decided and the plan approved by the owner the same day; built 2026-10-19, pushed 2026-10-20 with CI green*);
  ~~*first, from the stage 1 milestone review (owner, 2026-10-21): the SP-1 proposal and spec revision 34, in the
  review's section below;*~~ *built 2026-10-22 at spec revision 34, cut down by the owner; what was deferred is in
  the review's section below;* ~~the G2 proposal, so group G (G1, G2, the Holmes–Rahe sizes) can go to one revision and be
  built before the stage 2 pilot;~~ *the G2 proposal written and group G built 2026-10-22 at spec revision 36 on the
  branch `claude/recursing-bohr-2393a1`, every record regenerated once, each choice decided as recommended under the
  owner's instruction of 2026-10-22, approved by the owner 2026-10-24 in T-125 (the revision 36 report);* ~~then the stage 2 plan~~ *drafted 2026-10-23, `docs/implementation_plan_version_2.md` §2, D8–D13 decided as recommended
  under the owner's instruction of 2026-10-22, approved by the owner 2026-10-24 (T-125);* ~~then its pilot (§2.4 steps 2–7, the same
  branch, `stage-2-plan-pilot`)~~ *built 2026-10-23 on that branch at spec revision 37: `M11.C.16` 8 of 8, `M11.C.42` 0 of 8 at 8 samples; the full sweep projected at 44 h did not run; D14–D16 decided as recommended, approved by the owner 2026-10-24 in T-125 (§2.10)*; ~~then Phase C's 14 directional criteria at 8 samples (D15)~~ *swept 2026-10-23 on the branch `stage-2-phase-c-sweep`, 27 min, §2.10*; ~~then part I's 13 in a session of its own~~ *swept 2026-10-23 on the branch `stage-2-part-i-sweep`, 3 h 34 min of run time, §2.10; its constant sets in T-125's batch 5*; ~~next part II's seven~~ *swept 2026-10-23–24, split across both Macs, 7 h 54 min on the longer share, §2.10*; ~~then proposals for groups C, F and A, each with the owner's note in the plan's §5 Q2~~ *stage 3 (`docs/archive/implementation_plan_version_2_stage_3.md`): decided by the owner and built 2026-10-24–26 at spec revision 39, the milestone review 2026-10-27, its triage 2026-10-28 (revisions 40 and 41), and the stage-end sweep rerun on revision 41's engine 2026-10-29 (T-159, revision 42); every step of that plan done*; next, version 2's definition (`docs/version_2_plan.md` §4): on 2026-10-29 `docs/version_2_status.md` had W1 unmet on 8 rows, W3 on 6 and W4 on 1 (`M11.C.48`, T-163), and the owner's answer on `M5.D.3` (T-158) is open. *Met at the tag `version-2` (2026-11-01, `docs/archive/version_2_definition_plan.md`); T-158 closed 2026-10-31. Reopened by revisions 47 and 55: at `2bc3c1f` W1 is unmet on 5 rows, W3 on 1 and W4 on 16; the plan to meet them again on the current engine is `docs/archive/implementation_plan_version_2_definition.md` (drafted 2026-11-09; its §8 answered by the owner the same day, each as recommended; step 2, the batch, next).* *Met again 2026-11-11 on `d3779684dd53`: every step of that plan done,
  and `docs/version_2_status.md` reads W1–W4 met (the four cue rows decided by the owner 2026-11-11; the stage-end sweep
  merged from both Macs). Next: the owner's call on a tag (that plan's §10, Q1 (a)), with T-213 (b) for the next spec
  revision and the 2026-11-11 review's items T-214 to T-233. The owner decided 2026-11-11, each as recommended, through
  `AskUserQuestion`: no tag until after the engine batch (T-214 and the other fixes that move the engine hash, one
  regeneration, then the stage-end sweep); first the documents-and-tests batch that moves no hash (T-220 to T-227),
  then the owner's answers on T-228 to T-233, then the engine batch. Step 2 done 2026-11-12: T-228 and T-229 answered by the owner, the rest
  decided as recommended under the owner's "continue with your recommendations"; the work they set is T-234 to T-242
  (`docs/todo/2026-11-11_whole_repo_review.md`). Step 3 done 2026-11-12: the engine batch built as spec revision 59
  (`docs/plan_engine_batch_2026-11-12.md`), the engine hash moved from `d3779684dd53` to `95368d6ddf18`, and
  the record-current tests are strict expected failures until the records are regenerated. Step 4 done 2026-11-12: every record regenerated once on
  `95368d6ddf18` (`docs/version_3_revision_59_changes.md`); `M11.C.34` and `M11.C.10[family of origin]` moved PASS → FAIL, so W1 waits on the owner's call on `M11.C.10[family of origin]` (`M11.C.34`'s FAIL is final under version 1's accepted finding); T-237 run, T-242's P5 text drafted, approved by the owner 2026-11-13; 5 of the LLM experiment's 31 runs are now `not_replayable` (accepted as recommended, approved by the owner 2026-11-13). The stage-end sweep done 2026-11-13 on `95368d6ddf18`, split across both Macs (`docs/version_2_sweep_record.md`): W2–W4 met, W1 not met on `M11.C.10[family of origin]` alone. 2026-11-13: after a probe (a detection-limit outcome of its fixed seeds, no one rule's effect) the owner accepted its FAIL as a finding (`docs/version_2_dispositions.md`), so W1–W4 are met. The owner decided 2026-11-13, as recommended, no tag: Q1 (a) of 2026-11-09 stands (`docs/archive/implementation_plan_version_2_definition.md` §10). Done.*
- [x] T-002 ⟨now⟩ **`/csdp` sweep of 2026-10-19, items not fixed** (each with its reason; the last three are the fix re-sweep's
  minor items, recorded under the owner's three-or-fewer rule).
  - The owner's Holmes–Rahe Q3 ("additions carry load", decided 2026-10-19) cited `KS07.1`, which §2 then misdescribed
    as a visit that raised a symptom; the source says it improved the wife's functioning. The decision stands with a
    correction note beside it. The owner should look at it again before revision 33 is written. *Not asked before
    revision 33; the owner decided 2026-10-21 (stage 1 milestone review, Q14) that it is asked before group G's
    build starts. Group G was built 2026-10-22 under the owner's overnight instruction; the decision stands, cites no
    source, and is in the revision 36 report's decisions for the owner's review (T-013).*
  - `test_m16c1_chart_names_the_absent_and_draws_the_undeclared` puts its undeclared person in every sampled week, so
    the leading-`None` path for someone who enters mid-run is not exercised; the `PYTEST_*` filter in
    `tests/test_version_1_status.py` has no test of its own (a wrong option fails that test loudly).
  - The byte-equality checks on the committed trace, chart and frozen page were measured on macOS only; watch the
    first CI run on Linux after the push.
  - `tools/version_1_status.py` in the working tree is no longer run by any test: `main()` refuses, and
    `tests/test_version_1_status.py` runs the tag's copy. It is kept, not reduced to a stub, because the plan names it
    as the starting point for version 2's status page. Whoever writes that page starts from the tag's copy
    (`git show version-1:tools/version_1_status.py`) or puts the new tool under its own test first. Since 2026-10-20
    `tools/spec_coverage.py` imports its `dispositions()` parser (the dispositions file is frozen with it), so
    `tests/bowen/test_spec_coverage.py` runs that one function; its record readers no longer fit the part II record.
  *Closed 2026-11-14 (owner): the Holmes–Rahe citation is under T-013; Linux CI has been green since; the two test gaps are
  minor and dropped; `tools/version_1_status.py` is kept, as `tools/version_2_status.py` and `tools/spec_coverage.py` import it.*
- [ ] T-003 ⟨later⟩ **Coverage page, minor, from the Part A review (2026-10-20), not fixed.** (1) `tools/spec_coverage.py`'s
  `BUILT_PHASES` (A–D) is a constant nothing checks: when Phase E is built, the notes keep saying "Phase E builds it
  and is not built yet" until it is edited. (2) A spec row amended by appending keeps its old "not met", so a criterion
  whose clause is later met stays *partial* until the row is reworded (the safe side). (3)
  `test_m141_header_names_each_record_read` finds records by the `*_ensemble_record.md` and `*_mutation_record.md`
  names and the `"criterion"` field; a record named or shaped otherwise would pass it unread.
- [x] T-004 ⟨now⟩ **Coverage page, from Part B's drafting (2026-10-20), not fixed** (owner 2026-10-20: record them here). Each ID's
  note in `docs/spec_coverage.md` gives the detail. *Closed 2026-11-14 at spec revision 61: every sub-item below was
  fixed or decided; what was left open is T-244 to T-248.*
  - *Spec and code disagree; the owner decides which stands.* `M1.A.7`: the parents'-mean term against "MUST NOT be
    computed from a family average". ~~`M1.A.18`: the 0.3 threshold makes a genuine I-POSITION impossible below it,
    against "graded, not a hard gate".~~ *resolved at revision 39 (Q8: the threshold retired); taken off 2026-10-28, stage 3 review MR-9.* `M1.A.5c` against `M6.I.4`: a one-sided fall of pseudo-self against its
    conservation across an exchange. `M7.E.1e` against `M7.E.1f` (owner, 2026-10-13): `M7.E.1f`'s `d` uses the
    parents' global `functional_level`. `M1.A.4`: the estimator's rise is scaled by `estimator_rate`, against "rather
    than by a rate constant". `M1.C.4`: the learner keys values by triad and never reads `activation_memory`.
    `M1.A.18c` and `M1.A.18d`: the perspective term applies to every act, and per-tie attenuation is not read by the
    `M4.D.3b` gate or `M5.F.4`.
    *Checked and decided 2026-11-14 (spec revision 59, engine `95368d6ddf18`; owner's answers):*
    - `M1.A.7`: not a contradiction — the parents' mean is `M1.A.7a`'s field term under `M7.B.1a`; spec follows code (revision 60).
    - `M1.A.5c` against `M6.I.4`: not a contradiction, an unbuilt clause; the code's `M6.I.4` check would trip if built; build it (T-243).
    - `M7.E.1e` against `M7.E.1f`: still holds; spec follows code (revision 60).
    - `M1.A.4`: still holds; spec follows code, `estimator_rate` named in the rule (revision 60).
    - `M1.C.4`: partly mis-recorded (no policy term is what the spec requires); `activation_memory` is never read; spec follows code, a readout only (revision 60).
    - `M1.A.18c`: mis-recorded — `M4.D.1a` puts perspective in every selection's weight and nothing writes it into `functional_level`.
    - `M1.A.18d`: still holds; code follows spec (T-243).
  - *Text that names a built phase as work to come.* The error in `src/bowen/engine/event_effects.py::_release_binder`
    ("is built in Phase C"); `config/bowen/act_classes.md` (`M1.E.3`'s and `M1.E.4`'s acts "are Phase D and not yet
    moves"); `src/bowen/engine/beliefs.py`'s docstring (the family-level store as Phase D work, `M9.1`);
    `src/bowen/render/trace.py`'s refusal of a `BeliefWriteRecord` with a "Phase D" message (`M16.A.5`);
    `config/bowen/constants.md`'s promise that the estimator sweep "is added when `M11.C.18` first reads the
    estimator" — it now reads it and no sweep exists (`M1.A.4c`, `M10.B.3`); `docs/archive/implementation_plan_phase_c.md`
    §10 says `M1.B.10` and `M4.D.5a` were built as written, and no code builds either.
  - *Spec rows that name a test that does not exist.* `M1.F.2` (`test_m1f2_same_kind_moves_receiver_on_opposite_sides`)
    and `M11.D.12` (`test_m11d12_corpus_magnitudes_are_not_config_keys`).
  - *Tests weaker than the ID they are counted for.* `M6.3a` is *done* by
    `test_m63a_a_death_with_no_living_mourner_queues_no_nodal_event`, which checks none of its clauses — rule 2 of
    `tools/spec_coverage.py` counts any test named for an ID. `test_m16c5_trace_carries_header_and_framing`'s needle
    "invented" also matches "invented family", so it does not guard the "invented constants" sentence (`M10.C.1`).
    The `M4.D.3a` test lowers basic and functional level together (`M4.A.5`'s did too, and was fixed in Part B's
    review). Removing `beliefs_used=` from `policy.decide` or `act._record` fails no test (`M16.A.3a`). No test fails
    if the runner uses an unpaired test or skips Holm's correction (`M17.A.3`).
  - *Fields and records that exist and are never written.* ~~`TieState.OPEN_CONFLICT` (`M1.B.3`)~~ *written since
    revision 36 (`consolidate.py`, a run of three `CONFLICT`s); taken off 2026-10-24, stage 2 review MR-10*; `Person.beliefs`
    (`M1.A.16`); `Person.programmed_reactivity` (`M10.A.1`); the engine never emits `BeliefWriteRecord`, so belief
    writes are not separable from ground-truth changes in a run's log (`M16.A.5`, `M16.A.5a`); the spec register names
    writers for two `M1.A.5b` fields that do not exist; `Triangle.bound_anxiety` (`M1.C.1`, the owner's D22: no act binds anxiety into a `Triangle`; added 2026-10-28, stage 3 review MR-6).
  - *Register gaps.* Six mechanisms called from `src/bowen/engine/tick.py` have no register row (`M14.A.2`,
    `M14.A.3`): `stressor_load`, `respond`, `reactive_cutoff_cost`, `update_attribution`, `update_focus`,
    `update_danger`.
  - *Outside Part B's scope.* `M12.1`–`M12.5` and `M12.2a` still read "no test named for it (a prohibition, checked by
    review in every phase)", and `M14.2` "no test named for it (every phase)"; the plan's §3 covered only the notes
    that named Phase C or D.

- [x] T-243 ⟨later⟩ **Engine batch: two clauses from T-004 the owner kept (2026-11-14).** *Built 2026-11-14 at spec revision 61,
  with `Person.programmed_reactivity` removed.* Both move the engine hash; build in
  the next batch that does. (1) `M1.A.5c`: a self-directed functional rise lowers pseudo-self, converting toward solid;
  today it goes to `stress_swing` (`stress_response.py`), and `invariants.py`'s `M6.I.4` check (total pseudo-self unchanged
  every tick) must be narrowed to exchange records or the fall logged as its own record. (2) `M1.A.18d`: the `M4.D.3b`
  gate (`policy.py`, `obs.systems_perspective`) and `M5.F.4`'s weight (`iposition.py`) read
  `appraise.py::effective_perspective` on the target tie, not the held value. Optional in the same batch: remove the
  unwritten `Person.programmed_reactivity` field (`M1.A.7`, revision 60).
- [ ] T-244 ⟨later⟩ **`M11.D.12`: the checks module (2026-11-14).** `M10.C.4b`'s checks module for `M10.C.4`'s twelve
  corpus magnitudes, and `test_m11d12_corpus_magnitudes_are_not_config_keys` with it. A guard by value alone would flag
  ordinary 5s and 40s; it needs the magnitudes named. From T-004, owner 2026-11-14.
- [ ] T-245 ⟨later⟩ **`M1.A.4c`: the estimator's sensitivity sweep (2026-11-14).** `estimator_window` and
  `estimator_breadth` are read by `M11.C.18` and other criteria, and the stage-end sweep leaves both out
  (`config/sweep/sweep.md`); no sweep over them exists. From T-004's `constants.md` note.
- [ ] T-246 ⟨later⟩ **An event in flight is still delivered to a person who has died (2026-11-14).** Found writing
  `M6.3a`'s test: a move emitted before a death is logged as a `DeliveredRecord` to the dead person a tick later. It
  moves none of their state (appraisal, beliefs and attribution skip the dead), but the log shows a delivery `M6.3a`
  says the dead no longer perceive. Decide: drop it at delivery, or log it as undeliverable.
- [ ] T-247 ⟨later⟩ **`M1.A.5c`: a lender's lent pseudo-self has no way back once the borrower converts (2026-11-14).**
  Both review passes of revision 61. `relax_pseudo_self` and `return_lent_functioning` (`src/bowen/engine/moves.py`)
  move lent pseudo-self back only while the borrower holds a positive amount, and the conversion leaves the lender
  unchanged (the owner's choice, 2026-11-14), so a borrower who converts all of a loan leaves the lender at the lent
  amount for the rest of the run, and the estimator (`slow_tick.py`, `functional_level − pseudo_self`) credits the
  lender with it as own level. **Decided (owner, 2026-11-14):** lower the lenders' lent with the conversion, split as
  the return splits it, in the same `pseudo_self_conversion` record. Not built in revision 61; it goes in the next
  engine batch with T-248.
- [ ] T-248 ⟨later⟩ **Revision 61's low review findings (2026-11-14).** Each a hash-moving change, kept for the next
  engine batch: `invariants.pseudo_self_held` admits a `pseudo_self_conversion` record of either sign (count only
  falls); `beliefs.with_truth` leaves `true_value` `None` silently when a `tie_beliefs[...]` subject matches no tie
  (raise); a newborn's resting tie beliefs (`life_course._introduce`) are written with no `BeliefWriteRecord`;
  `with_truth` reads the truth at step 3, before step 4's appraisal of the same deliveries (state the timing in
  `M16.A.5a`, or move the call); `"belief"` and `"attribution"` in `trace.SYSTEM_SUMMARISED` are dead entries.
