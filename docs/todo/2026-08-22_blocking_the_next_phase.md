# Blocking the next phase

- [ ] T-001 ⟨now⟩ **Version 2, next:** ~~stage 0~~ *done 2026-10-19: the `version-1` tag at `d62cfdc`, pushed, with the status
  page frozen and checked against it; the dispositions; the render-only fixes moved to stage 1 (owner).* Then stage 1
  (the engineering batch, with the grid engine retired, stage 0.3's two fixes, and the trace renderer out of the hash;
  *plan drafted 2026-10-19, `docs/implementation_plan_version_2.md` §1; D1–D7 decided and the plan approved by the owner the same day; built 2026-10-19, pushed 2026-10-20 with CI green*);
  ~~*first, from the stage 1 milestone review (owner, 2026-10-21): the SP-1 proposal and spec revision 34, in the
  review's section below;*~~ *built 2026-10-22 at spec revision 34, cut down by the owner; what was deferred is in
  the review's section below;* the G2 proposal, so group G (G1, G2, the Holmes–Rahe sizes) can go to one revision and be
  built before the stage 2 pilot; then proposals for groups C, F and A, each with the owner's note in the plan's §5 Q2.
- [ ] T-002 ⟨now⟩ **`/csdp` sweep of 2026-10-19, items not fixed** (each with its reason; the last three are the fix re-sweep's
  minor items, recorded under the owner's three-or-fewer rule).
  - The owner's Holmes–Rahe Q3 ("additions carry load", decided 2026-10-19) cited `KS07.1`, which §2 then misdescribed
    as a visit that raised a symptom; the source says it improved the wife's functioning. The decision stands with a
    correction note beside it. The owner should look at it again before revision 33 is written. *Not asked before
    revision 33; the owner decided 2026-10-21 (stage 1 milestone review, Q14) that it is asked before group G's
    build starts.*
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
- [ ] T-003 ⟨now⟩ **Coverage page, minor, from the Part A review (2026-10-20), not fixed.** (1) `tools/spec_coverage.py`'s
  `BUILT_PHASES` (A–D) is a constant nothing checks: when Phase E is built, the notes keep saying "Phase E builds it
  and is not built yet" until it is edited. (2) A spec row amended by appending keeps its old "not met", so a criterion
  whose clause is later met stays *partial* until the row is reworded (the safe side). (3)
  `test_m141_header_names_each_record_read` finds records by the `*_ensemble_record.md` and `*_mutation_record.md`
  names and the `"criterion"` field; a record named or shaped otherwise would pass it unread.
- [ ] T-004 ⟨now⟩ **Coverage page, from Part B's drafting (2026-10-20), not fixed** (owner 2026-10-20: record them here). Each ID's
  note in `docs/spec_coverage.md` gives the detail.
  - *Spec and code disagree; the owner decides which stands.* `M1.A.7`: the parents'-mean term against "MUST NOT be
    computed from a family average". `M1.A.18`: the 0.3 threshold makes a genuine I-POSITION impossible below it,
    against "graded, not a hard gate". `M1.A.5c` against `M6.I.4`: a one-sided fall of pseudo-self against its
    conservation across an exchange. `M7.E.1e` against `M7.E.1f` (owner, 2026-10-13): `M7.E.1f`'s `d` uses the
    parents' global `functional_level`. `M1.A.4`: the estimator's rise is scaled by `estimator_rate`, against "rather
    than by a rate constant". `M1.C.4`: the learner keys values by triad and never reads `activation_memory`.
    `M1.A.18c` and `M1.A.18d`: the perspective term applies to every act, and per-tie attenuation is not read by the
    `M4.D.3b` gate or `M5.F.4`.
  - *Text that names a built phase as work to come.* The error in `src/bowen/engine/event_effects.py::_release_binder`
    ("is built in Phase C"); `config/bowen/act_classes.md` (`M1.E.3`'s and `M1.E.4`'s acts "are Phase D and not yet
    moves"); `src/bowen/engine/beliefs.py`'s docstring (the family-level store as Phase D work, `M9.1`);
    `src/bowen/render/trace.py`'s refusal of a `BeliefWriteRecord` with a "Phase D" message (`M16.A.5`);
    `config/bowen/constants.md`'s promise that the estimator sweep "is added when `M11.C.18` first reads the
    estimator" — it now reads it and no sweep exists (`M1.A.4c`, `M10.B.3`); `docs/implementation_plan_phase_c.md`
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
  - *Fields and records that exist and are never written.* `TieState.OPEN_CONFLICT` (`M1.B.3`); `Person.beliefs`
    (`M1.A.16`); `Person.programmed_reactivity` (`M10.A.1`); the engine never emits `BeliefWriteRecord`, so belief
    writes are not separable from ground-truth changes in a run's log (`M16.A.5`, `M16.A.5a`); the spec register names
    writers for two `M1.A.5b` fields that do not exist.
  - *Register gaps.* Six mechanisms called from `src/bowen/engine/tick.py` have no register row (`M14.A.2`,
    `M14.A.3`): `stressor_load`, `respond`, `reactive_cutoff_cost`, `update_attribution`, `update_focus`,
    `update_danger`.
  - *Outside Part B's scope.* `M12.1`–`M12.5` and `M12.2a` still read "no test named for it (a prohibition, checked by
    review in every phase)", and `M14.2` "no test named for it (every phase)"; the plan's §3 covered only the notes
    that named Phase C or D.
