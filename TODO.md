# TODO

Deferred and adjacent items. Each carries enough context to act on later without the conversation that produced it.
Open items only: a ticked item is moved, verbatim, to `docs/TODO_archive.md` within 30 days (`tests/test_doc_size.py`).

## Blocking the next phase

- [ ] **Version 2, next:** ~~stage 0~~ *done 2026-10-19: the `version-1` tag at `d62cfdc`, pushed, with the status
  page frozen and checked against it; the dispositions; the render-only fixes moved to stage 1 (owner).* Then stage 1
  (the engineering batch, with the grid engine retired, stage 0.3's two fixes, and the trace renderer out of the hash;
  *plan drafted 2026-10-19, `docs/implementation_plan_version_2.md` §1; D1–D7 decided and the plan approved by the owner the same day; built 2026-10-19, pushed 2026-10-20 with CI green*);
  ~~*first, from the stage 1 milestone review (owner, 2026-10-21): the SP-1 proposal and spec revision 34, in the
  review's section below;*~~ *built 2026-10-22 at spec revision 34, cut down by the owner; what was deferred is in
  the review's section below;* the G2 proposal, so group G (G1, G2, the Holmes–Rahe sizes) can go to one revision and be
  built before the stage 2 pilot; then proposals for groups C, F and A, each with the owner's note in the plan's §5 Q2.
- [ ] **`/csdp` sweep of 2026-10-19, items not fixed** (each with its reason; the last three are the fix re-sweep's
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
- [ ] **Coverage page, minor, from the Part A review (2026-10-20), not fixed.** (1) `tools/spec_coverage.py`'s
  `BUILT_PHASES` (A–D) is a constant nothing checks: when Phase E is built, the notes keep saying "Phase E builds it
  and is not built yet" until it is edited. (2) A spec row amended by appending keeps its old "not met", so a criterion
  whose clause is later met stays *partial* until the row is reworded (the safe side). (3)
  `test_m141_header_names_each_record_read` finds records by the `*_ensemble_record.md` and `*_mutation_record.md`
  names and the `"criterion"` field; a record named or shaped otherwise would pass it unread.
- [ ] **Coverage page, from Part B's drafting (2026-10-20), not fixed** (owner 2026-10-20: record them here). Each ID's
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

## Model ergonomics — open after Parts A–C, 2026-10-22

- [ ] **Revision 35's dispositions, for the owner's review (Part E, 2026-10-22):** `docs/spec_revision_35_dispositions.md`,
  218 rows for 213 sentences (35 stay, 32 moved, 151 rewritten; rows 219–262 added the same day for the missed
  phrases, the item below), applied as Claude's dispositions under the owner's bulk
  decision (§8 Q2). The review of 2026-10-22 found two log bullets whose quotation was not the sentence as it stood
  (`M1.A.4k`'s owner list and `M11.C.14`'s Bowen quotation), both restored verbatim, and one post-hoc label dropped from
  `M11.C.6`'s arm, restored; the E2 pass was an agent's, so the owner's read of the rewritten rows is the check.
  The "stays" rows marked unsure carry the reason in the last column. A reversed row is one edit in the spec and one
  bullet back out of `docs/spec_revision_log.md`.
- [ ] **D4, long `Spec:` lines (Part D, 2026-10-22; the plan says record, do not split):** `engine/objects.py` 56 IDs
  (its classes `Person` 21, `Relationship` 17), `engine/life_course.py` 21, `engine/appraise.py` 18,
  `engine/iposition.py` 18, `ensemble/criteria.py` 19 (a registry, one per criterion), `policy/policy.py` 16,
  `engine/invariants.py` 15, `engine/log_records.py` 14, `engine/events.py` 12, `engine/symptoms.py` 12,
  `engine/slow_tick.py` 12, `scenario/family.py` 12. A line this long may mean the module does too much.
- [ ] **Adjacent issues found by Part D's docstring pass, not fixed (2026-10-22):**
  - `engine/draws.py` declares the `per_hop_fidelity` draw class (`M3.D.4b`'s row) but nothing draws it;
    `visibility.fidelity_for` uses the constant. The spec's table row or the engine is out of step.
  - `engine/live_positions.py` cites `#M13.1`, a phase-plan requirement the code does not implement (the class B3
    removed elsewhere; it is a defined ID, so the test passes).
  - `engine/consolidate.py`'s module `Spec:` line omits `#M4.C.1c`, which `consolidate()` cites.
  - `engine/recompute._counter_moves`: an inner `if` on the kind is always true after the `continue` above it.
  - `engine/iposition._abort_act` re-imports `math` inside the function.
  - `objects.Role` (MEMBER/EXTERNAL) and `events.Role` (TARGET/WITNESS) share a name; a reading hazard only.
  - `scenario/header.py::resolved_config` hashes neither `kinds.components` nor `kinds.accommodating`, so a change
    to a kind's components or its `accommodates` flag leaves `config_hash` unchanged; check against `M16.A.1`.
  - `scenario/policy_source.py` implements `M1.E.8` (its comment says so) without citing it.
  - `src/bowen/__init__.py` names `docs/implementation_plan_phase_b.md` as the build order; stale since version 2.
  - `run.py` composes a `ScriptedSource` only; `PolicySource` is not reachable from the composition root.
  - `readouts/evaluation.py` line 103 emits phase narration inside a readout value ("from Phase D part II").
  - `ensemble/counterfeit.py` cites `M5.C.1` for "the gate's two thresholds" while `M5.C.1`'s first sentence is
    the general gate rule; check the ID is the one meant.
  - Comments (out of Part D's scope) still carry decision narration: `engine/observe.py` line 58, `engine/danger.py`
    line 78, `ensemble/runner.py` line 81, `ensemble/criteria.py` lines 47–48, 111–119, 146, `ensemble/criteria_d.py`
    line 430, and the `params.py` / `scenario/constants.py` field comments.
  - `engine/contact.py` said "revision-12 finding 6" where the spec (`M4.C.1c`) records revision 11; the phrase is
    gone from the docstring, but CHANGELOG may carry the same label.

## Version 2 stage 1 milestone review — open after the triage, 2026-10-21

The review is `docs/cycles/2026-10-21_milestone-review_v2-stage-1.md` (the second Mac, branch
`review/2026-10-21-v2-stage-1`). The triage refuted none of its 26 findings: 22 hold as stated and 4 in part. The
owner's ten decisions and what this batch fixed are in `CHANGELOG.md`. Open:

- [ ] **SP-1, the parts the owner deferred on 2026-10-22** (the build was spending more on the transcripts than on
  the model; the proposal's §7 and §9):
  - SP1.3: the local test, marked `transcripts` and deselected by default, that matches quoted spans against the
    transcript folder named by `EPMODEL_TRANSCRIPTS`, with its line in `docs/milestone_review_brief.md`.
  - SP1.6: the header note on each extraction note in the proposal's §3 (their quotations only locate passages), and
    the six headings in `kb02.md`–`kb15.md` that name Bowen as the speaker of a Hijack-set interview.
  - This file's two ASR quotations: the backlog item on hedges trimmed off the front of quotations (2026-08-28)
    quotes spec phrases from the tapes and interviews, which revision 34 paraphrased, so that item can close or
    narrow; the other is in the item on locating the rest of the interview series, and is on
    `tests/asr_quotations.json`'s allowlist until it is paraphrased. The default-suite test does not catch the first.
  - The test's known limits (its docstring; the proposal's §9): it misses a quotation whose citation sits in another
    paragraph, list item or table row, or that has none; one set as a block quote without quote marks; one whose
    marks do not pair (a curly opening closed by a straight one); one after a stray quote mark in its own block; one
    longer than 800 characters; one in straight single quotes or guillemets; one beside a citation form not in the
    rules (`KB-04`, `T6.8`, "tape five"); an HTML entity in a Markdown file is not decoded; only tracked `.md` and
    `.html` files are read. The local test (SP1.3) is the check for most of those.
- [ ] **SP-1 Q7: six requirements whose passage supports less than they say** (owner, 2026-10-22: each its own
  decision; the paraphrases keep to the passage and the requirement's MUST is unchanged at revision 34; the
  proposal's Q7 table):
  - `M11.C.11` (SP1-S40): Tape 6 ranks a long absentee's return as probably the strongest of the additions, and says a
    family generally reacts more to a loss; it does not by itself support the return exceeding the removal.
    `_LECTURES_1979.md` T6.8 makes the same step. Revision 34's paraphrase now says so inside the criterion's own row.
  - `M1.C.3a` (SP1-S16): Tape 5 speaks of the intensity of symptoms and relationship patterns, not of triangles.
    Decide with the next item's `SP1-E06`, which cites the same passage for triangles.
  - `M1.E.7` (SP1-S24): the KB10 passage is about a society accepting what it can no longer deal with; its use as the
    condition for a rise in `systems_perspective` is the spec's analogy.
  - `M5.B.3a` (SP1-S32): KB08's statement on de-selfing is limited to a family where everyone does it, and the same
    interview recommends near-full disclosure between spouses as an exercise.
  - `M5.C.1a` (SP1-S34): KB08's warning was given to one family before it entered the research, not as a general
    warning against the protocol.
  - `M1.B.10a` (SP1-S15): Tape 6 states a closed subject's reopening as a wish, and says couples usually cannot
    reopen it themselves by the time they seek help; it supports the MUST as a possibility, not a common outcome.
- [ ] **Found while building SP-1 (2026-10-22), not fixed:**
  - `SP1-E06`, the explainer's §6.1 triangle table (`routing_capacity`): the proposal's check note says the text now
    says that applying Tape 5 to triangles is the project's inference (K04.15); the approved text does not say it.
    Pair with `M1.C.3a` above.
  - `SP1-S38` (`M10.A.2`) cites K10.9 twice, once inline and once in its source arrow.
  - `SP1-S35` (`M5.D.7a`) dropped the old note's explanation of why the attribution changed.
  - `FE03.21`: the block quotation in `M4.D` ends a quoted run in the chapter whose speaker is not known; it is cited
    as Kerr's chapter until it is checked against the printed book (`family_evaluation/fe03.md`).
- [ ] **From the SP-1 build's review (2026-10-22), not fixed:**
  - Six exempt paths in `tests/asr_quotations.json` hide no finding today (`CHANGELOG.md`, `LEARNINGS.md`, `papers/`,
    `_KERR_INTERVIEWS.md`, `_PASS2_CORRECTIONS.md`, `_SOURCE_QUALITY.md`). Kept: each is exempt by the proposal's §3
    for what it is (a dated record, the owner's files, extraction notes), so a quotation added there later should
    not fail. `test_sp1_4` checks that each path exists, not that it hides anything.
  - `test_sp1_4` does not catch two allowlist entries for the same file and span.
  - The test scans tracked files only, so a new document fails it only once it is added to git. Scanning untracked
    files would scan the owner's untracked files too.
  - `src/bowen/engine/live_positions.py:1` says the predicate is used in four places; revision 34 says three callers
    (M8's preamble). The file is in the engine hash, so the fix waits for group G's regeneration.
  - **Owner decision:** `docs/theory/family_evaluation/` and `docs/theory/kerr_book/` are exempt from the test as
    written-source notes, but four of their cross-references quote an ASR source: `ks07.md:159` (KB07 · K07.4),
    `ks19.md:148` (KB04), `fe10.md:303` (KB06) and `fe07.md:38` (Tape 2). Paraphrase them, or hold these notes like
    the `kb` notes under D1 (their quotations only locate passages), or list their spans and drop the folder
    exemptions so a fifth is caught. The exemption reasons now say so.
  - `tools/famdiag_export.py` keeps the app keys it names (positions and sizes, colours, settings). Any other key the
    app writes when the owner edits a field the tool does not draw (a birth date, say) is dropped, with a report
    line. The tool writes in the app's format, so a key name cannot tell an owner's edit from a value an earlier
    version of the tool wrote; keeping every unknown key would keep stale run output. Owner edits to fields the tool
    draws (a name, a line's notes) are redrawn by design. The file is written in place, not atomically; it is in git.
- [ ] **With group G's regeneration.** Each of these edits moves a record hash, so it waits for a change that
  reruns the records anyway (owner, D6 and D7):
  - MR-23: move the Phase C ensemble record onto `docs/records_cache`, as Phase D's records are: render it from the
    cache, take the tool out of its own hash, and compare the full render in `tests/bowen/test_ensemble_record.py`.
  - `config/bowen/family_phase_d.md:19`: drop "awaiting owner review" from Teodor's 36.
- [ ] **Before group G's build starts:** ask the owner the Holmes–Rahe Q3 again (D10, the review's Q14; see
  "Blocking the next phase").
- [ ] **For the stage 3 question list** (owner, D4 and D10):
  - Build the `M5.C` avoidance gate (`avoidance_available`), or drop it from M8 and `M13.1`.
  - Q7, fusion: no caller passes a fusion map, so `positions_live`'s fused-occupant branches never run. The spec
    leaves open whether a fused occupant makes its triangle inactive.
  - Q8, the coach and SPLIT: `external.coach_side` has the coach always go along with a SPLIT (`M11.C.7`, classed
    Mixed). Is that intended?
- [ ] **SP-2: extend the document-claim tests** to the places where this review found stale prose: closing clauses
  of status lines, the checklist's `- [ ]` items, the reading list's body, lists of standing decisions, counts in
  prose, and the status notes in the reference-family table.
- [ ] **Adjacent, not fixed** (found by the batch's learning-qa pass): `docs/theory/_LECTURES_1979.md:46` still says
  twelve of twenty-three items were probed and the rest are outstanding. Its own "Revision 1" section and
  `_STATUS.md`'s file map say all 23 were probed.

## Revision 32 (finishing version 1) — open after the build, 2026-10-17

Plan §4.12 holds the evidence. Built under the owner's instruction "proceed using your own recommendations"; every
decision is Claude's recommendation, pending owner review.
- [ ] **A genuine `I-POSITION` at level 40 almost never completes** since revision 16 (0 of 200 seeds reach the peak in
  `M11.C.32`'s calm arm at level 40). `hold_gain` and `defence_threshold` may be sized for the calmer model before
  cutoff became rejection. A rule decision for every `I-POSITION`.
- [ ] **Criteria near the runner's detection limit.** `M11.C.22`'s mild limb has about a 78% pass chance at the runner's
  stop; others may be similar. A general rule (a fixed seed count when the effect is within a set multiple of the
  precision) would change `M17.A.1` for all criteria.
- [ ] **`M1.E.7e`'s lending of pseudo-self by a sided coach is not built**: a sided coach's contact lands nothing.
- [ ] **`M11.C.29` could be redesigned** (the owner accepted its failure 2026-10-17; optional, for version 2): its genuine `I-POSITION` never completes in the arm, so
  its time-course limb is untested. A redesign needs an act week clear of the spell and a mover able to prepare; the
  diagnosis found nothing that completes without weakening the third person's symptom (plan §4.12).
- [ ] **Minor, from the revision 32 reviews (2026-10-17), not fixed:**
  - ~~`SIDE_TAKING` (`external.py`) is a classification kept in code, as `FAMILY_TO_EXTERNAL` is in `policy.py`; rule 9
    puts it in config. Move both together.~~ *Moved at version 2 stage 1 (S1.3), with `EXTERNAL_REPERTOIRE`, to
    `config/bowen/act_classes.md`.*
  - How long a coach stays on a side borrows `triangle_activity_window`, so tuning `M1.C.3` changes `M11.C.7` too.
  - `arm_c7` scripts the `SPLIT` at `session − 1`; it stays inside the window at the landing tick only while the coach
    tie's latency is 2 or less (it is 1).
  - For V3 the status tool counts a red on any gating readout of a mixed criterion; it does not know which limb a
    mutant reached. No passing mixed criterion depends on this today.
  - ~~Adjacent, not fixed: `recompute.py`'s triangle activity treats the coach as an unfused occupant, though `M8.5` says
    the same predicate decides triangle activity.~~ *Settled at version 2 stage 1 (owner D3 (b)): `M8.5` amended at
    revision 33 to say presence decides; `test_m8_5_s1_4_a_sided_coach_leaves_its_triangle_active` pins it.*

## Spec revision 30 (origin cutoff, danger, focus lock-in) — open after the build, 2026-10-14

Plan §4.10 holds the evidence.
- [ ] **`M11.C.10`'s input barely differs between its arms** (correctness review, 3 seeds, 12 years, ad hoc). The
  origin cutoff of the first gen-3 child was 0.79 (open) against 0.87 (held), and 0.80 against 0.92 in the
  family-of-origin variant: parents already hold some rejection on some tie most weeks, and the held cutoff adds 0.08
  to 0.12. This limits what 200 seeds can detect.
- [ ] **`M11.C.17` passes by a fragile margin, not by one mechanism** (ad hoc, 2026-10-14, the revision 29 worktree):
  switching off any one of the four revision 29 gains leaves the coached rise at about 0; switching off all four
  reproduces revision 28's +0.000881 ± 0.000828.
- [ ] **Minor, from the fix re-sweep (2026-10-14):** `update_danger`'s docstring puts the dead-person sentence in the
  middle of the paragraph on cost (move it to the end; left during the regeneration so the engine hash would not move);
  `arm_c47` reads only people with a tie to the person, so a removed tie would drop someone from both readouts
  silently (not possible on the fixture today).
- [ ] **Adjacent, not fixed (2026-10-19):** the ad hoc probes in `tools/probes.py` (lines 139, 141, 268, 458) still
  take float means with the builtin `sum`. No record reads them; move them to `math.fsum` when the file is next
  touched.
- [ ] **Adjacent, not fixed (stage 1 review, 2026-10-19):** the Phase D ensembles (`tools/phase_d_record.py`) run
  with no timeout, and `tools/diagnostic_record.py:136` and `tools/level_occupancy.py:145` stop when their child
  fails, where the Phase C records keep the rows and go on (S1.5a). Both run once per regeneration; give them the
  Phase C handling when either is next touched. *The cold sweep (2026-10-20) added to `tools/level_occupancy.py` a
  check that the child printed one row for each of C.16's arms, so a short record is no longer written; a failed
  child still stops it.*
- [ ] **Adjacent, not fixed (stage 1 review, 2026-10-19):** the act-literal scan, widened to lists, tuples and one-act
  comparisons, finds act names in `src/bowen/ensemble/criteria.py`, `criteria_d.py` and `criteria_d2.py` (the acts
  each criterion reads), as well as `NEVER_REFUSED` (`rejection.py`) and `ABORT_ACTS` (`iposition.py`). Rule 9 asks
  whether each belongs in config; the criteria are definitions, not engine rules, and are for a later stage.
- [ ] **Owner: freeze `docs/version_2_stage_1_changes.md` at stage 1** (`/csdp` sweep, step 8, 2026-10-20). The page
  compares the tag `version-1` with the records as they are now, so a later regeneration fails
  `test_s1_10b_changes_page_is_current`, and rerunning `tools/record_changes.py` would rewrite the page under stage
  1's title with the later stage's changes. Freezing it needs a ref to read stage 1's records from, such as a tag at
  the stage 1 commit (the owner chose the tag `version-1`); then the tool reads both sides from git and the test
  stays true. Until then: do not rerun the tool after stage 1.
- [ ] **Minor, from the `/csdp` sweep (step 8, 2026-10-20), not fixed:** (1) a stop signal while
  `tools/mutant_runner.py:run_in_copy` copies the repository, or while its temporary directory is removed, leaves a
  `bowen-mutant-*` directory in `$TMPDIR` (no process is left; the handlers are set only around the child), as does
  a late stop passed on to the default handler, which ends the process at once (`_stop_signals_raise`'s docstring);
  (2) two runs of one record tool at once overwrite each other's cache file (`tools/record_cache.py`), the last save
  winning; (3) the timed-out path's second `communicate()` in `tools/child_process.py` has no timeout, which would hang
  only if a worker left the child's process group, and none calls `setsid`.
- [ ] **Adjacent, not fixed (`/csdp` fix re-sweep, 2026-10-20):** the generated records are written in place with
  `write_text` (`tools/ensemble_record.py:211`, `tools/level_occupancy.py:148` and the other record tools), and
  later tools read them (`mutation_record.passing`, `sweep_record.central`, `phase_d_mutation_record.passing`). A
  kill mid-write fails the next tool loudly and costs a rerun of minutes, so this is lower priority than the cache
  (now written whole and moved into place); give them the same write when any is next touched.
- [ ] **Adjacent, not fixed (cold sweep, 2026-10-20, C3):** `code_hash` (`tools/ensemble_record.py:54-60`) hashes
  each `config/bowen/*.md` file as raw bytes; only `constants_changes.md` is skipped. Rewording a config file's prose,
  such as the text under the table in `config/bowen/act_classes.md`, changes the engine hash, so every record reruns
  (P38). The Python files are hashed without comments and docstrings; the config files could be hashed by what the
  loader reads from them. Older than stage 1 (PLAUSIBLE: no such edit has been made yet).
- [ ] **Adjacent, not fixed (cold sweep, 2026-10-20, C4):** the worker pools that `tools/phase_d_record.py` and
  `tools/ensemble_record.py` start in their own process (`src/bowen/ensemble/runner.py:116`, a
  `ProcessPoolExecutor`) are not killed as a group when the parent gets a stop signal, as `tools/child_process.py`
  does for a mutant child (S1.5a), so after a SIGTERM to the parent, workers can run on until their current task
  ends. Older than stage 1 (PLAUSIBLE: not reproduced).
- [ ] **Owner: a row that raises on every run is never cached** (cold sweep, 2026-10-20, after the C1 fix). A RAISED
  row is a red under `tools/mutation_record.py`'s `judge` (the criterion can no longer pass), and the sweep and
  diagnostic records print its error as its result, but `tools/record_cache.py` caches no errored row. So a mutant,
  sweep setting or diagnostic variant that makes the engine raise every time leaves its record stale for good: the
  tool now exits 1 and says so, and the record's test fails. No committed record holds a
  RAISED or BROKEN row (checked 2026-10-20). Caching a RAISED row whose cause is the model, not the machine, would
  let such a record be committed; what counts as evidence is the owner's choice.
- [ ] **Minor, from the document-claim tests' fix re-sweep (2026-10-20), not fixed** (`docs/plan_doc_claim_tests.md`;
  recorded under the owner's three-or-fewer rule). (1) D5b's `classify` (`tests/test_doc_status.py`) reads "Yet to be
  approved", "To be decided by the owner", "Not yet fully approved", "Not formally approved" and "Isn't approved" as
  settled: a negator is matched only when followed by fillers and then a settled word. A document whose own status says
  "To be decided" would agree with a checklist label "Decided" and nothing would be reported. No status in the repo
  reads that way today. Fix: add "to be" and "yet to be" to the open list, and adverbs and "isn't" to the negation lists
  in `tests/doc_claims.json`, with classification cases. (2) D2's roots (`tests/test_doc_claims.py` `path_scan`) are
  the five project folders and the top-level folders git tracks, so a path under a whole top-level folder that was
  deleted (`papers/x.pdf` after `papers/` is gone) is not read and not reported. Fix: take root candidates from the
  folders the documents name as well.
- [ ] **`M11.C.47`: the dangerous person is cut off on fewer weeks** (−16.6 ± 4.9 `held_weeks`, 150 seeds), though felt
  danger and the principled part are higher. Trace why before reading the premise as a finding about cutoff.
- [ ] **`M11.C.2` on the twelve-person family fails again at revision 30** (+0.70 ± 1.5, p 0.16; +1.31 ± 0.95 at
  revision 29). *Traced 2026-10-15, each revision 30 change removed alone:* unmodified +0.70 ± 1.5, p 0.16 (reproduces the record); `M4.D.1h`'s force removed +0.70 ± 0.97, p 0.13; `M1.B.8b`'s compounding removed +1.12 ± 1.7, p 0.09; `M1.B.8b`'s reinforcement removed +1.11 ± 1.6, p 0.093; `M1.D.8a` removed +0.96 ± 1.5, p 0.14; `M1.B.3e`'s danger removed +0.08 ± 1.8, p 0.46 (`symptom_load`, seeds 0–149, one-sided signed-rank with the runner's margin; ad hoc, 2026-10-15). No single removal brings
  the pass back. Removing the force gives revision 29's spread with revision 30's mean (the two means agree to three
  figures with different spreads; measured, not explained); removing either focus term gives back most of the mean
  with the wider spread. Next, if wanted: force and both focus terms removed together, about 10 minutes. Under the
  version 1 definition this composite can also be accepted as a finding.
- [ ] **Phase D mutation record, three minor items** (re-review of 7c8a0fb, 2026-10-15): (1) ~~`engine_hash` reads
  `inspect.getsource` of `apply` and `run_in_copy`, comments included, so a docstring edit there reruns the record~~
  *(fixed at version 2 stage 1, rider (iv): read through `source_text`)*;
  (2) `build()` does not flag an `AD_HOC` entry that now has a mutant or no longer passes (only the test checks the
  first); (3) the spec's `M11.C.10` row still opens with the revision 11 and revision 29 mutation clauses, superseded by
  the later amendments but not struck (item 3 is in revision 31's plan, approved in the review 2026-10-16).
- [ ] **Seven inconsistencies in the spec found while drafting the review form** (listed in
  `docs/REVIEW_spec_revisions_24-30.md`, "Inconsistencies found while drafting"): `channel_focus` "from belief" (line
  1630) against `M1.A.11c`; `tolerance` Family field (line 1679) against per person (`M7.D.4`); `TRIANGLE`'s channel in
  the B8 plan against `attribution.md`; `M7.E.1c` "open owner decision" (line 1084) against plan line 1355; `M11.C.2`
  settings "to be added" (line 1314) though present; criteria settings ungraded; revision 28 tags with no source. Fix
  with the review's revision, so the spec moves once. *2026-10-16:* all seven are in revision 31's plan (§4.11); the
  `TRIANGLE` one needs no spec change (the owner approved the config as built).
- [ ] **rca: record findings and commit review fixes separately** (rca audit, 2026-10-14): with fixes folded into the
  feature commit, the audit cannot run a reproducer at the parent. Have csdp write `.rca/findings.md`.
- Dismissed with reason (correctness review, 2026-10-14): intrusion is read by `tie_change` after the week's move
  effects and landed contacts, not on the state appraisal saw. Both read the same function; the spec now says which
  state.

## Spec revision 29 (cutoff and focus) — open after the build, 2026-10-14

Plan §4.8 holds the evidence.
- [ ] **Focus steering has no measurable effect in Phase C** after the review made it a redistribution (share 0.578
  against 0.580 off, ad hoc, 20 seeds). Whether `focus_steer_gain` 1.0 is too small to matter is for the owner.
- Dismissed with reason (correctness review, 2026-10-14): `weakness_focus` reads the usual level after this week's
  `update_usual`. The symptom channels and `find_work` read it the same way at step 8 and 9, so it is consistent.

## Phase D part II — open after the build, 2026-10-13

Plan §4.7 and `docs/phase_d2_status_2026-10-13.md` hold the evidence.
- [ ] **Owner: the ±2% band against `M11.C.6`'s traversal limit (a)**, and development as `M7.E.1f`'s rule against
  the estimator bounded to the band. `M11.C.6` measured +0.018 ± 0.028 under a band of about ±0.8 points.
- [ ] **Owner: `M11.C.10`.** The declared cutoff ends within about 2 years, and the uncut baseline is cut within weeks,
  so the arms run the same process. Witnessed-cutoff learning and origin cutoff as an input to intensity (`M1.D.8`)
  have no mechanism. The act count is dominated by generation 3 cutting their own parents (about 775 per run). Hold
  the cutoff? Read tie states? Build link two or three? (plan §4.7)
- [ ] **Owner: `M11.C.2` on the twelve-person family.** Projection is spread evenly (about 80 acts per child). Chronic
  anxiety takes only the year at 11, and the band fixes the level. Late load follows age (Leo 24.9, Pia 15.4). Read
  load over matched ages? Make chronic anxiety integrate childhood? Why does focus not settle on one child?
  (plan §4.7)
- [ ] **Chronic anxiety ratchets across generations.** A child's chronic anxiety is derived from the living parents'
  values and its own witnessed history and then fixed at 12 (`M7.B.1a`); nothing in the run lowers it again for later
  generations. Check over several generations whether it only rises; the 40-year run is too short to see it.
- [ ] **Starting ages as settings in the fam-diag app.** The owner asked for them as settings in the app; they are
  config settings (`M2.A.0j`) and the app was not touched (standing instruction). Deferred.

## Phase D — open from B3/B4, 2026-10-11

- [ ] **`M11.C.44` moved FAIL → PASS at revision 20 (2026-10-12), with both mutants surviving.** p went from 0.055 to
  0.012 on an engine change that does not touch the outside position (the system's adjustment as a level). A criterion
  that flips on an unrelated change is on its threshold; with `one-sided-deviation` and `deviation-inverted` both
  surviving, the pass is not evidence (`M11.5`). At revision 21 it moved back to FAIL. For the owner when Phase C's gate is
  next looked at.
  Revision 24 (2026-10-12): it moved FAIL → PASS again with B7's symptom channels. Still on its threshold. Revisions 25–27 moved it back to FAIL.
- [ ] **Review of revision 20's completion, minor (2026-10-12; the correctness pass, which read the version with
  `shed_adjustment`, since removed):**
  - At the stress floor a calm week decays `adjust_swing` while functioning cannot fall, so the estimator's `own`
    is overstated by at most `response_relaxation_rate × system_adjust_gain` (0.02). Intended, as for `stress_swing`.
  - No test pins a partial top-up (0 < `adjust_swing` < `system_adjust_gain`) in `external.self_practice`.
  - The once-per-stance gate is now mostly redundant with the level (a top-up at the gain adds nothing); it only limits
    renewal to once per `practice_window`.
- [ ] **After revision 31 part B (2026-10-16), for the owner** (plan §4.11 holds the evidence):
  - ~~**Keep or revert `M6.I.4`'s capacity rule.**~~ *Decided 2026-10-17: kept; `M11.C.22`'s failure accepted for now.* It did nothing for `M11.C.21` (+0.15 ± 0.19; +0.24 ± 1.36 without it),
    made the twelve-person `M11.C.2` pass (+3.14 ± 1.7, proved red by removing weakness focus and steering), and turned
    the premise `M11.C.22` (mild limb) and the composite `M11.C.17` red. With the rule removed each gives its earlier
    number exactly. Kept overnight as the owner's decision; reverting costs one regeneration (about 2.5 hours).
  - ~~**`M11.C.21`: apply the reclassification with the next code change**~~ *Done 2026-10-17, with the header and the
    `span == 0` fix below; every record regenerated once (plan §4.11).* (owner, 2026-10-17: composite, failure
    accepted as a finding).
  - **`M11.C.14`'s mutation clause is not met:** the gate-removed mutant survives both variants; the gate blocks about a
    point of relief per run in this fixture, and calmer parents (up to 30 points) did not change that.
  - ~~**Config prose waiting for the next hash move:**~~ *Done 2026-10-17.* `config/bowen/criteria.md`'s header does not yet describe
    `M11.C.21`'s `act_every` or `M11.C.14`'s hold-back baseline. Config `.md` files are hashed byte for byte, unlike
    Python, whose comments and docstrings are not, so a header edit reruns every record.
  - Minor, from the reviews (2026-10-16): a scripted hold is booked "made" when its actor emitted no move, inferred
    rather than read from its selection record; `bound_side = −1` has no caller and no test.
  - Minor, from the fix-loop re-sweep (2026-10-16): ~~at `span == 0` (basic level 0, or `stress_floor_below_basic` 0)
    the capacity rule gives a taker exactly at basic 0 rather than "full at or above basic"~~ *(fixed 2026-10-17, two
    tests)*; `test_m11c14_gate_removed_…` checks the spec by a literal phrase.
  - ~~Adjacent, not fixed: `config/bowen/constants_changes.md` lists the five revision 28 rows twice~~ *(checked
    2026-10-17: each row appears once now)*; `config/bowen/family_phase_d.md` line 19 reads "awaiting owner review" for
    Teodor's 36 (revision 13). *Checked 2026-10-17: accurate, the spec's family table and plan P15 still list it as
    open.* ~~**For the owner: Teodor's basic level 36 or 34.**~~ *Settled at 36 (owner, 2026-10-21, stage 1 milestone
    review triage, D6). The Phase D plan's note was updated then; the spec's "awaiting owner review" goes at
    revision 34, and `family_phase_d.md:19`'s with group G's regeneration, because a config edit moves the hash.*
- [ ] **Failing criteria from steps 8–10, each the owner's question; nothing was changed to make any pass:**
  - `M11.C.14`: the gate blocks improvement but not harm, so firmness and talking it out still differ. *Decided
    2026-10-16 (review, option B):* restate it to "no technique lowers the child's load against a no-technique arm";
    the difference between techniques is reported. Revision 31.
  - *Revision 31 (2026-10-16):* `M11.C.14` and `M11.C.40` pass as decided; `M11.C.23` is a composite, accepted as a
    finding; `M11.C.21` still fails (see the item above).
  - *Version 1 (2026-10-16):* `M11.C.21`, `M11.C.23` and `M11.C.40` are failing **premises**, so V2 requires each to be
    fixed or restated by the owner before version 1. `M11.C.21` and `M11.C.40` need a diagnosis run first;
    `M11.C.23`'s direction is set by two `[I]` sizes, so bring the owner options. `M11.C.43` is a composite: the
    owner accepts it as a finding or not (V1).
  - `M11.C.40`: per seed, the automatic channel's entropy difference does not exceed the self-directed one's.
  - `M11.C.43`: no larger cost on losing a calming member after learned reliance.
  - `M11.C.21`: the target does not rise when both parents collapse.
  - `M11.C.23`: reversed; halving symptom accumulation outweighs the hostility.
- [ ] **`M11.C.2` fails** once projection acts are counted early and load read late (+0.29 ± 2.20, p 0.39); read over the
  same weeks it passed, which cannot separate cause from attraction. No criterion yet shows the tie-weighted
  cross-person term (revision 26) shaping a child's repertoire.
- [ ] **`M11.C.34` fails** on reactions to the onset against a no-onset run; and its declared mutant cannot bite in any
  case: no selection reads `M1.A.19`'s detectors.
- [ ] **`M11.C.17`'s coached rise nearly vanished** with revisions 24–27 (+0.021 → +0.0009). Traced 2026-10-13 (plan, step
  10): the coach's contact lands only when binders fail (an active symptom), which at revision 23 was 99% of weeks only
  because everyone was saturated; now about 40%, so landings and perspective halve. Owner's question: should the gate
  read load above the usual level instead? *Decided 2026-10-16 (review): no, it keeps reading an active symptom or
  load near threshold.* The weak pass stands and needs a recorded mutation for V3. *Superseded: `M11.C.17` fails at revision 31 (−0.00008 ± 0.0013); its failure is proposed for acceptance (`docs/version_1_dispositions.md`).*
- [ ] **Reviews of revisions 25–27 (2026-10-12), not fixed, with reasons:**
  - **`I-POSITION` sequences now rarely complete.** In the triad fixture, completion fell from about 10 in 96 seeds
    (revision 20) to 2 in 96 after B7, and 4 in 192 now. `tests/bowen/test_iposition.py::test_m5d8_*` was raised to
    192 seeds and still rests on four sequences (P23). Why B7 lowered it is not traced.
  - The marital-distance gate blocks a symptomatic child's improvement, but the move's contact and impingement
    still land on the tie and persist. Gating them as well made `M11.C.14` no closer to its null (measured: −8.4 to
    −10.9 over 20 seeds), so it was left.
  - `update_focus` reads the sender's true excess above usual: the family's focus is engine-side attention, not a
    belief any policy reads (the spec says so).
  - A refused move never reaches its sender's own attribution.
  - `arm_c43` makes M unavailable to F, not F to M: M can still approach F in the loss window.
  - `arm_c26` and `arm_c22` return one scenario's records, so their bookkeeping covers one run of several.
  - `M11.C.36` measures appraisal only; its row says "appraisal and selection".
  - `relax_pseudo_self` runs tie by tie, so a person relaxed on two ties in one week sees the first change before
    the second: deterministic, but not `M1.F.8`'s batch rule.
- [ ] **Phase D record, review of 2026-10-13, minor:** ~~`tools/spec_coverage.py` reads only the Phase C
  ensemble record, so `docs/spec_coverage.md` gives the Phase D criteria no verdict~~ *(fixed 2026-10-20, the coverage
  plan's Part A)*; the README's and the status report's lists of passing and failing D criteria are still not checked
  against `docs/phase_d_ensemble_record.md` by a test.
- [ ] **B7 (spec revision 24, 2026-10-12): for the owner's review.** Every B7 choice is Claude's recommendation, made
  overnight at the owner's request: the symptom integrand read above the usual level, the channel weights, mutual
  protection, the lock-in curve and its spill past the turn, removal by tolerance, and `M11.C.22`'s control arm.
- [ ] **`M11.C.12` fails** (B7, 2026-10-12): released anxiety raises the other spouse's excess, but more anxiety does
  not select `CONFLICT` in particular (dyad, 20 seeds: 55 against 56 conflict acts between the spouses). Whether the criterion, the
  readout or the policy is wrong is the owner's question; nothing was changed to make it pass.
- [ ] **Lock-in does not make a symptom chronic** (B7): the hold does not feed the bearer's integrator, because
  feeding it made every bearer climb to the turn and stay there. `KS23.1`'s "fosters it becoming chronic" is not
  produced.
- [ ] **Correctness pass on B7, minor (2026-10-12), not fixed:**
  - On the dyad, a remission's returned hold is split between the spouse and the bearer's parent, so half of it
    reaches the marriage.
  - The hold is one stock per person, not per channel: remitting one channel returns all of it, though another
    channel may still be active.
  - Lock-in runs in person order: a partner of two bearers gives more to the first, and one bearer's spill is
    counted in the next one's excess. Deterministic, but not symmetric between people.
  - `M11.C.30` reads mean social *load*, a stand-in for the onset hazard its row names.
  - `arm_c11` and `arm_c12` have no fast test; a window or arm error shows only in the ensemble run.
- [ ] **`Family.tolerance` is declared, never updated** (B7): `M7.D.4`'s "tolerance update" in the register is not
  built, and no fixture declares a tolerance, so removal by tolerance never fires in any run.
- [ ] **Review of revision 23, minor (2026-10-12):** `functional_level` is clamped at 100 (and at the stress floor) by the
  stress response, the I-POSITION gain and the system's adjustment, while pseudo-self is not, so a person at a bound
  can read slightly off basic level in the estimator. The exchange itself clamps its amount and is exact.
- [ ] **Adjacent, found in the review of revision 23 (2026-10-12), not fixed:** the `M1.A.5c` table says a sustained
  self-directed rise moves pseudo-self **down** ("converting toward solid"). Nothing implements that; only
  `initialise_functioning` and `functioning_shift` write pseudo-self. If it is implemented, the signed estimator reading
  (revision 23) would read each fall in pseudo-self as a rise in own functioning, on top of any rise in
  `functional_level` itself. Decide the two together.
- [ ] **Review of revision 20, minor (2026-10-11):** no full-tick test that a calm partner sheds part of the gain in the
  week it arrives (`self_practice` runs before `respond` in `tick.py`; consistent with the rule, not pinned).
- [ ] **Adjacent, not fixed (review of revision 21, 2026-10-12): two checks are written out more than once.** "Self-directed
  or a hold" (`kind == WITHHOLD or channels.get(kind) == SELF_CHANNEL`) appears in `external.py` (twice),
  `stress_response.py` and `slow_tick.py`; "held their position" is `moves.holds_position` and, separately,
  `external.self_practice`'s soothing check (which judges only the first target). One shared helper each would keep them
  from drifting apart.
- [ ] **Pre-existing unused imports in `tests/bowen/test_engineering_gates.py`:** `clamp_unit`, `dataclasses`,
  `EventStore`, `legal_outcomes`. Found while checking the B3/B4 diff. Not touched, since they predate it.

## Phase C — open after the build, 2026-10-08

Phase C steps 0–16 are built. **Its gate does not pass** (`docs/phase_c_completion_report.md`). These need a person.

- [ ] **`M11.C.42` and `.45` stopped passing after the `TRIANGLE` decision.** *What each tests decided 2026-10-08
  (report §9): C.42's baseline arm and readout, and C.45's readout, brought to the spec's text. Rerun: both still
  fail. Current figures (`docs/phase_c_ensemble_record.md`, 2026-10-10): C.42 +0.08 ± 0.39 (p 0.21); C.45
  −0.0008 ± 0.0044 per person-week (p 0.69). The 2026-10-08 figures were C.42 +0.26 and C.45 +0.0020. What remains is whether the learner should produce these
  effects at all, which is a model question, not a test question.* Both were already fragile in the sweep.
  C.42's reuse is −0.17 and it fails at every sweep setting; C.45 has the right sign, without significance.
  Decide whether either criterion's arms still test what it claims, now that the act loads the recruited third.
  Report §3.8 and §8.
- [ ] **Owner: `M11.C.16`'s class rests on a reading.** Composite holds if only `M4.D.3a` states a narrower repertoire;
  if `M4.C.1a`'s KS03.2 text or `M1.C.3a`'s "stays fixed" is read as stating it, `M11.5`'s redundancy clause would make
  it a premise. Removing those two rules alone leaves C.16 passing (`c16-cited-grounds-removed`). Removing them with
  availability now also survives (`c16-grounds-removed`, −0.006 ± 0.005), while the availability inversion reverses it
  (`docs/phase_c_mutation_record.md`). Occupancy figures are in `docs/phase_c_level_occupancy.md`. Report §11.
- [ ] **The record caches are trusted, not proved.** A test re-renders each record from `docs/records_cache/` and
  compares, but nothing binds a cached result to a run: editing a value in the cache and re-rendering passes. The
  key hashes the inputs, not the result. Only a rerun checks it (learning-qa, 2026-10-09).
- [ ] **`docs/phase_c_level_occupancy.md` is checked by its hash line only**, not re-rendered from cached rows like
  the mutation and sweep records, so a hand-edited table would pass (learning-qa, 2026-10-09).
- [ ] **`tools/record_cache.py`'s infrastructure-error match is by exact class name** (`INFRASTRUCTURE_ERRORS`,
  line 43). An error of the machine raised in the child under a class not named there, a subclass
  (`FileNotFoundError`, `BrokenPipeError`, `PermissionError`) or a base (`BrokenExecutor`, the base of
  `BrokenProcessPool`), is classed RAISED, an uncached red, instead of FAILED; `KeyboardInterrupt` in
  the list is dead (it is not an `Exception`) (learning-qa, 2026-10-09; found again by the cold sweep, 2026-10-20,
  C9). Since the cold sweep's C1 fix such a row makes the tool exit 1, so it is no longer silent, but the record
  shows a red where the machine failed. Older than stage 1. *The 2026-10-09 entry's last part is fixed: since S1.5a
  a child that prints no row for a requested criterion gives a FAILED row, not `None`.*
- [ ] **Diagnostic probes, minor (batch review 2026-10-09).** `c44_act_counts` re-runs `arm_spell_triad`'s scenario
  instead of calling the arm; probes sample at tick start, so the last tick is missed; `summarise` takes its columns
  from the first group and assumes every row has the same keys. None changes a current number. The
  "no rows" and "names no criterion" exits of `tools/diagnostic_record.py` have no test.
- [ ] **`M11.C.16`'s mutation clause is not met.** The spec requires that disabling `M4.D.6` turn it red, and it does
  not. Decide whether to restate the criterion so that a learned contribution is what it tests (for example, hold
  level-reading availability fixed and compare learning on against off), or to accept it as a level result.
- [ ] **The triangle change is not proved by an acceptance criterion** (2026-10-09): `triangle-help-ignored`
  survives on `M11.C.3` (a direction, not a size), and `triangle-credit-cross-person` has no passing criterion.
  Both rules have unit tests. A criterion that reads the size of the seeker's relief against the third's help, or
  `M11.C.42`/`.45` passing, would prove them.
- [ ] **Re-sweep notes, 2026-10-09** (fix range `da57f87..996bb7a`; deferred, not fixed now):
  1. ~~`reactive_over_chance` averaged in `FALLBACK` selections~~ *done 2026-10-09: policy draws only; the
     verdicts did not change.*
  2. Three probes (`c27_deviation_terms`, `horizons`; `tools/probes.py`) call `made(criteria)` without records,
     so they count placed acts as made. Diagnostics only; pass the records at the next probe edit.
  3. Low: a scripted week the run reached but never offered would read "not reached"; the scripted-criteria
     scans stop at an arm's first `scenario` call (every arm calls it once today).
- [ ] **Plan §9's human reviews:**
  - ~~`docs/readouts_phase_c.md`~~ *the owner gave the definitions, 2026-10-10 (visualization plan V1).*
  - **The 104-week Phase C trace is not reviewed** (owner, 2026-10-10: viewing the seed-7 diagram was not the trace
    review). Read `docs/review/phase_c_trace_seed7.md`, or the diagram in EPModel Fam-Diag.
  - **`M5.F.2`'s threshold is open.** The owner found the first explanation unclear (2026-10-10), so it was
    re-explained.
- [ ] **fam-diag, found in review, not fixed** (pre-existing; neither affects File > Open of an export):
  - merge import collapses two episodes of the same pattern on one pair into one line (`utils/diagramMerge.ts`);
  - `FamilyCutoffArc` ignores the timeline (`DiagramCanvas.tsx`).
- [ ] **What the seed-7 picture shows**, for the owner's review. Most parent–child and marital ties go silent
  for months: Marta–Nadia has no act for up to 102 weeks, Marta–Ravi for 67. No cutoff pattern forms, although there are 62
  CUTOFF acts: no one refuses 3 approaches in 4 weeks. Sofia → Ravi projection holds for 20 blocks.
- [ ] **Not built:** `M11.C.7` (needs `M8.2`/`M8.3`'s predicates and a direction), `.13` (needs a community),
  `.14` (needs `M5.C.1`'s marital-distance gate); two of `M11.1c`'s four re-encodings (rescaled state range,
  integer against float ticks); `M11.1b`'s severing mutants; plan D9's "dominant constant" (§3 names none).
- [ ] **Ninth gate's notes, 2026-10-08** (`docs/cycles/2026-10-08_phase-c-qa-gate-9.md`, APPROVED, nothing medium or
  higher): (1) `tick_of` in `tests/bowen/test_phase_c_criteria.py` does not handle `LogHeader`, which `run_tick` never
  emits; make it total or assert the record type. (2) The C.45 rate check still copies the readout's filter; share one
  predicate from `criteria.py` the next time the records are regenerated anyway, since moving it into `src` changes
  the hash. (The third note, a stale date in `CLAUDE.md`, is fixed.)

## Spec revision 10 — deferred at the owner's review, 2026-09-22

Revision 10 (branch `spec-rev10-draft`) folds the method literature into the spec. The owner answered its
fourteen questions on 2026-09-22; these are the ones left for later. The answers table is in the spec's
revision-10 section.

- [ ] **Disposition at death (C15, spec `M6.3`).** Decide where a dead person's anxiety, bond energy,
  functioning-balance debts, positions and budget share go. Owner's guidance, not yet a requirement:
  relationship energy behaves like chemical bond energy and does not disappear when a person dies. That agrees
  with `M1.B.4` (bond energy decays at or near zero) and `M6.I.7` (no exit from the field). `M11.C.37` cannot be
  written until this is decided.
- [ ] **`E-RM` and C34 — one requirement or two (DESIGN_LESSONS Q16).** Currently `M17.D.1`(d) plus `M17.D.2`.

## Held for a later revision — paper material, 2026-09-27 (labelled "revision 11" until 2026-10-07)

> **Relabelled 2026-10-07 (spec revision 12, P12).** Revision 11 became the emergence revision and revision 12 the
> Phase C prerequisites. Nothing in this section is part of either. It waits for the owner to assign it to a revision.
>
> **For the Phase D plan (revision 12, P10):** criteria for Bowen's two falsifiers of the budget (`L09.2`, `L16.2`) and
> `L15.1`'s delayed symptom after a loss; decide `L03.4`'s two-rung avoidance ladder and `L03.5`'s illness in the mover.
>
> **Phase C plan prerequisites from revision 11:** rework Phase B's base appraisal (`appraise_base.py`, now `M4.C.1`–`M4.C.1b`),
> standing load (`standing_load.py`, now `M4.C.1c`) and triangle activity (`recompute.py`'s `tension_activation_threshold`
> and `test_m1c3_triangles_are_inoperative_when_calm`, now amended `M1.C.3`). Define the pattern readouts `M5.A.1a`
> requires before any criterion reads them. Decide the five rule groups revision 11 marked for review.

The owner decided on 2026-09-27 that nothing from the 2026-09-27 paper batches enters revision 10, which is
under external review. All of it waits for revision 11. None of it is a requirement yet.

**One exception, by owner instruction on 2026-10-06:** `M11.F.10` (the "virtual family" term may not be used
without `M11.F`'s framing) was entered in the spec text directly, marked ⟦proposed rev11 · owner decision
2026-10-06⟧. **Approved with revision 10 the same day**, so it is no longer a revision-11 item. Rationale: `model_explainer.md` §1.1.

- [ ] **Method proposals from the two 2026-09-27 batches.** 36 candidates. First batch: J1–J4, SV1–SV4, PD1–PD4,
  X5–X9, in `papers/SWEEP_READING_REPORTS_2026-09-27.md`. Second batch: BB1–BB5, MM1–MM4, AD1–AD3, TM1–TM6, X10,
  in `papers/SWEEP_READING_REPORTS_2026-09-27b.md`. Plain-language versions with examples are in
  `papers/SPEC_CANDIDATES_plain_language_2026-09-27.md` and `…-27b.md`. Candidates labelled X are for the
  exploratory narrator line only and do not enter the v2 spec.
- [ ] **Owner answer: anxiety shapes beliefs (answers MM1).** Anxiety makes a person's beliefs about others more
  subjective. It also biases risk assessment upward, so the same event is read as more threatening. For revision 11:
  - belief writes (`M9.8`, and the per-person belief store) read the receiver's own anxiety, not only the events
    delivered;
  - the weight on the receiver's own state rises with anxiety;
  - the threat bias points one way, upward;
  - direction graded `[user]`, strength graded `[I]`;
  - check `M4.C.5` and FE05.10 for corpus support before raising the grade;
  - `M9.2` still binds: a threat-biased reading is not automatically false.

  MM1's probe (identical events, only the receiver's anxiety differs, plus a label-only null arm) becomes the test.
- [ ] **Owner answer: how coaching enters a family (answers SV2).**
  - **Coach.** Present from t0 in every run.
  - **Knowledge.** A family learns that coaching exists only through an outside event, which reaches one or more
    named people at a declared tick.
  - **What makes an arm.** Arms differ by that knowledge event, meaning whether it happens, when, or who receives it.
    If both runs receive the same event on the same tick, it is shared history, not an arm, and not what the
    comparison tests.
  - **After the event.** Everything is the family's own dynamics: whether to try coaching, how many sessions, and
    whether to stop once relief comes or continue. The owner expects lower-differentiated people to stop once
    relieved more often, graded `[user]`; corpus support for relief lowering motivation is ledger L14.6 and L14.2.
  - **Spread.** Once one person knows, knowledge spreads through ordinary ties, mostly to those closest and safest.
    A partner is almost always told. Spread is therefore an output of the model, not part of the arm definition. The
    exception is an arm in which the person is instructed not to tell their partner: that instruction is declared
    as part of the knowledge event.
  - **Reporting.** The headline result compares runs where the family learned about coaching with runs where it
    did not. Never tried, tried and stopped, and continued are reported underneath as a breakdown, never as the
    comparison itself.
  - **Where it sits in the spec.** Next to `M1.E.7f` (family-relative entry threshold and rejection hazard) and
    `M1.E.8` (low optimum for contact frequency).
- [ ] **Method proposals from the two 2026-10-04 batches.** Added 2026-10-04 by owner decision; held for revision 11
  on the same terms as the 2026-09-27 batches. 59 candidates, including narrator-line items.
  - **Batch a.** The five papers from the 2026-09-28 sweep: WA1–WA5, CD1–CD4, CB1–CB3, BV1–BV3, PW1–PW3, plus
    WA-X1, CD-X1 and PW-X1. In `papers/SWEEP_READING_REPORTS_2026-10-04.md`.
  - **Batch b.** 21 papers from the owner's 25-paper digest of 2026-09-15 to 09-30. The other four had already been
    read. Candidates: PF1–PF3, AQ1–AQ3, RP1–RP2, PP1–PP2, EN1, TP1, PM1–PM2, RA1–RA2, NG1–NG3, RZ1–RZ2, AN1,
    AG1–AG2, AR1, AT1–AT3, EV1–EV4, plus AQ-X1, RP-X1, TP-X1, PM-X1, NG-X1, IT-X1, AN-X1 and AT-X1. In
    `papers/SWEEP_READING_REPORTS_2026-10-04b.md`.
  - **Plain-language versions.** `papers/SPEC_CANDIDATES_plain_language_2026-10-04.md` and `…-04b.md`.
  - **X-labelled candidates.** As before, these are for the exploratory narrator line only.
  - **Interactions with earlier proposals.** TP1 should be folded into BB5. PM2 and EV4 extend the MM1 test. EV1 and
    WA5 build on the SV2 answer. AR1's wording must be reconciled with PD4's "never select".
- [ ] **Open owner questions from the 2026-10-04 batches.** Each one blocks the candidate named. The full list is at
  the end of each plain-language file.
  - **Silent mechanisms** (theory decisions):
    - **RA1.** How a move's target is chosen, and whether anxiety shifts targeting toward the sender of the latest
      upset.
    - **PM1.** How long a per-person belief persists between writes. L09.4 may point the opposite way from the memory
      literature.
    - **BV1.** What a "situation", a "decision domain" and a "demand type" are. `M1.A.3`, `M1.A.3a`, `M1.A.14d` and
      the `M1.A.4c` estimator read them, and nothing represents them.
    - **BV3.** What `M4.D.1a`'s mixing weight reads.
    - **AN1.** Whether an event in flight is delivered after the sender dies or the tie is cut off.
    - **RZ1.** Whether belief items are coupled.
    - **RZ2.** Whether confirming writes are stronger than contradicting ones.
    - **WA2.** Whether more fused families react more alike to one event.
    - **EV4.** Whether anxiety weighs more on indirect evidence.
    - **Carry-over.** Whether a withheld move carries over to the next tick.
    - **EN.** Whether any stochastic term may persist across ticks.
  - **Arm and framing decisions:**
    - **NG1.** Whether an arm may separate spouses' levels against `M2.A.0c`.
    - **AT1, AT2.** Whether model "anxiety" is described as felt, and whether the consistency-engine framing becomes
      a requirement.
  - **Corpus item found while checking, not from a paper.** FE05.17 (protecting children from one's own problems
    transmits them) is marked testable in `fe05.md`, but no `M11` criterion covers it.
- [ ] **Owner questions from the 2026-09 maturity-and-emotion conversation.** Added 2026-10-05. These four ideas
  come from Claude's reasoning in `docs/conversation_agents_maturity_emotion_2026-09.md` §4, not from a paper or
  the corpus. Each is `[I]` until corpus support is found. None is a requirement, and each needs an owner decision
  before it can become a proposal.
  - **Goals and shared resources.** Each person holds weighted goals (closeness, achievement, money, autonomy,
    relief), and a family-level ledger of time, money and attention that pursuing those goals uses up. Conflict comes
    from goals competing for the same resources. Check against `M1.A.10` (`life_energy` split between relationship
    and goal-directed activity) before adding anything; it may already be the place for this.
  - **Three cost terms on an act.** An act carries relief now, a cost to the actor later, and a cost to others
    (tie tension, others' goals blocked). Drinking is the example: relief now, cost later. Qualifications are the
    reverse: cost now, payoff later. Check against `M1.A.4g` (substance use as a chronic `functional_level`
    pattern), `M4.D.6a` (reversed at revision 11: the automatic channel now learns from felt relief) and `M7.D`.
  - **An outside world with delayed returns.** School, jobs and legal risk return outcomes after a delay, with some
    chance involved. Without this, "qualifications pay off" can never come out of a run. This would be a new
    exogenous source; check it against `M1.E` and the scripted-event source.
  - **A horizon that changes with age.** The reinforcement horizon is short in adolescence, lengthens through the
    twenties, and nodal events can shift it. At present `M4.D.6b` declares one fixed horizon. This bears on
    `M4.D.1a`'s design decision that a longer horizon cannot reach differentiation: the decision stands, and a
    horizon that changes with age would act on the automatic channel only (`M4.D.6d`). The Redish full read
    (2026-10-05) gives this no support: his discount factors are fixed and never change with age or experience, and
    he states discounting is not the cause of addiction in his model. Any support would have to come from elsewhere.
- [ ] **Method proposals from the 2026-10-05 full reads.** Added 2026-10-05 by owner request; held for revision 11 on
  the same terms as the earlier batches. Three papers that had been catalogued at abstract level since 2026-09-17:
  Redish 2004 (TD1–TD3), Park et al. 2023, Generative Agents (GA1–GA4, GA-X1, GA-X2), Argyle et al. 2022, Out of One,
  Many (OM1–OM3, OM-X1). In `papers/SWEEP_READING_REPORTS_2026-10-05.md`; plain-language version in
  `papers/SPEC_CANDIDATES_plain_language_2026-10-05.md`.
  - **Interactions with earlier proposals.** GA1 adds to PM1. GA2 is close to RZ1 and could be folded into it. TD1
    may make `M4.G.3`'s separate habituation term unnecessary, and shows `M4.D.6a`'s "converges on CUTOFF by
    construction" assumes an additive update the spec never states. *That clause was removed at revision 11, when `M4.D.6a` was reversed; TD1's point about the update form still applies to `M4.D.6`.* OM1 sits next to `M10.C.2a`.
  - **Owner questions.** TD2: is an old pattern held down or erased when it goes quiet? GA1: does belief fading run on
    time since written or time since last used? GA2: does a derived belief outlast its evidence?
  - **Catalogue gap found by the Argyle reader.** The three later silicon-sampling papers (2609.10280, 2609.15849,
    2609.16395) had not been read in full. *Closed 2026-10-05: read in batch b, below.*
- [ ] **Method proposals from the 2026-10-05 batch b (silicon sampling).** Added 2026-10-05 by owner request; held for
  revision 11 on the same terms. Total Simulated Survey Error 2609.10280 (TS1–TS4, TS-X1), Before You Poll 2609.15849
  (BP1–BP5, BP-X1, BP-X2), Silicon Sampling Country Assumptions 2609.16395 (SC1–SC5, SC-X1, SC-X2). In
  `papers/SWEEP_READING_REPORTS_2026-10-05b.md`; plain-language version in
  `papers/SPEC_CANDIDATES_plain_language_2026-10-05b.md`.
  - **Interactions with earlier proposals.** SC4 amends OM2. SC5 could be folded into SC1. BP4 could be folded into
    CB1 or OM1. TS4 extends PW3 and PD4 from filtering and selection to weighting. BP2's inert arm builds on the SV2
    answer. TS2's readout alternatives must exclude the `M11.F.6` traps.
  - **Owner answer: BP2's inert arm (answered 2026-10-05).** Coach quality, one of `M1.E.7d`'s three landing
    conditions, is a declared setting that can be set to null. The inert arm is therefore an ordinary arm, set through
    a declared channel (`M17.D.3`). It is not a model change under `M11.4f`. For revision 11:
    - coach quality is a declared per-agent attribute, and its null value is `[I]`;
    - the inert arm matches the coach arm on timing, contact schedule, tie formation and event intensity, and differs
      only in coach quality;
    - it is added as a fifth control arm, (e), in `M17.D.1`.
  - **Owner answer: BP1's dependency list (answered 2026-10-05).** The list of `M11.C` criteria an intervention result
    depends on is **derived** from `M11.1d`'s minimal rule sets, not declared by hand. For revision 11:
    - a criterion is on the list when its minimal rule set shares a rule with the mechanism chain the result runs
      through;
    - the list is computed before the run and logged with the result in `M17.G.1`;
    - BP1 therefore depends on `M11.1d`'s rule sets being complete for every criterion.
  - **At approval of revision 10 or in revision 11.** The spec's line recording 2609.15849 and 2609.16395 as "read at
    abstract level only" is out of date. Left unchanged while revision 10 is under review.
- [ ] **Method proposals from the 2026-10-12 batch.** Added 2026-10-17 by owner request; held for revision 11 on the
  same terms as the earlier batches. Six papers read in full from the 2026-10-12 method-literature digest: Fu 2026,
  *"Hot-Blooded" vs "Cold-Blooded"* (2610.11951; HB1, HB2), Peng et al. 2026, *MASC* (2610.08250; MC1, MC2), Deng et
  al. 2026, *SPIN — Cognitive-Affective Personality Inference* (2610.08328; SP1, SP2, SP3), Wang et al. 2026,
  *Socio-Foundation* (2610.08967; SF1, SF2), Wang et al. 2026, *Prompts versus Rules* (2610.11015; PR1, PR2, PR3,
  PR4), and Fraile Navarro 2026, *Latent Identity Reversion* (2610.01490; IR1, IR2, IR3, narrator/LLM line only). The
  four deep-review candidates flagged in `DIGEST.md` were Hot-Blooded, MASC, SPIN and Socio-Foundation (PAIR 2610.07523
  was adjacent and not read in full); *Prompts versus Rules* and *Latent Identity Reversion* were surfaced by screening
  the wider digest. In `papers/SWEEP_READING_REPORTS_2026-10-12.md`; plain-language version in
  `papers/SPEC_CANDIDATES_plain_language_2026-10-12.md`.
  - **Scope note.** Unlike the earlier batches, none of the sixteen candidates is a theory change — all are
    inference-validity, reporting, or mutant-convention items for Phase E (`M17`). HB1/HB2 (per-readout reliability and
    the over-discreteness flag), MC1/MC2 (scripted-trajectory control arm; readouts must not reuse decision-path
    internals), SP1/SP2/SP3 (complexity-matched mutants; state-bypass collapse signature; tiny-N paired checks state
    the minimum p-value), SF1/SF2 (report dispersion as SD with extrema, never SE/CI; normalization bounds fixed and
    declared), PR1/PR2/PR3/PR4 (per-opportunity move rates; monotonicity on adjacent pairs with tie fraction; bound
    non-linearity to the input distribution; dispersion readout for artificial uniformity).
  - **Narrator-line items (do not enter the v2 spec).** IR1 (per-turn persona re-anchoring), IR2
    (represented-vs-enacted identity probe), IR3 (log the scaffold state) — same treatment as the X-series.
  - **Owner questions.** None about theory. The only owner decisions are adoption ones: whether to fold these into
    Phase E (`M17`) reporting/mutant conventions, and whether IR1–IR3 join the existing X-series protocol items.

## Arising from the 2026-08-28 batch

- [ ] **Audit the spec for requirements sourced from an extraction *heading* rather than a quoted sentence.** `M5.D.4` was inverted for exactly this reason: `docs/theory/ch13.md` carried the correct quotation in its body under a heading that said the opposite, and only the heading travelled into the spec. The heading is fixed and the requirement corrected, but **nothing has checked whether there are others of the same shape.** The check is mechanical: for each requirement citing a chapter, confirm the claim appears in a *quoted* passage in the extraction, not only in a section title or a bolded gloss. Highest risk in the earliest requirements, written when the extractions were newest.
- [ ] **Two changes to the family-diagram application, both needing information the diagram does not currently hold.** Neither can be recovered afterwards, which is why `M13.3` names them as the ones to make first.
  - `M15.A.4` — export the **interval** a rater would defend, with the scale point defined, not a bare `3` on a 1–5 scale. The model divides by, differences and thresholds these values; a rank supports none of that.
  - `M15.C.2` — split the diagram's single **"distant"** line into *rupture* and *resolved low-contact*. `M1.B.3` calls telling those apart the most consequential discrimination in this part of the theory: same contact frequency, opposite bond energy. Until the app emits it, bond energy on any distant tie imports as a free range.
### Deferred from the 2026-08-28 re-sweep — graded below medium, backlogged rather than fixed

The re-sweep of the fix commit returned fifteen findings. Four high and seven medium were fixed; these four
were graded low and are recorded instead, because each fix is new unreviewed surface and the project's own
rule is to bound the loop rather than trade one defect class for another.

- [ ] **`docs/theory/` is outside the reference guard, and has a dangling reference.**
  `family_evaluation/fe03.md` cites an `M0` requirement numbered **one past the last one §0 defines** (§0 runs
  `M0.1`–`M0.4`). `fe03.md` and `fe08.md` also cite `M1.E.7c`'s numbered sub-forms as though they were IDs —
  the spec's own uses of that notation were corrected at revision 5, the extractions were not. Adding the
  directory to `CITING_DOCS` turns the suite red until all three are fixed, so do both together.
  *(The literal tokens are deliberately not written out here: `TODO.md` is inside `CITING_DOCS`, so quoting a
  dangling ID as an example makes the guard fail on the note describing it. The guard is right to be strict —
  an exclusion mechanism would be the loophole that later hides a real one.)*
- [ ] **`_REF` truncates compound references.** `M7.D.2c/2d` yields only `2c`; `M1.D.7i–l` only `7i`. Neither
  is currently dangling, so nothing is wrong today — but the scan is narrower than it reads, and
  `_STATUS.md` describes it without that qualifier.
- [ ] **Name the stressor per criterion when the tests are written.** `M11.3` now places the obligation on the
  test rather than the criterion's prose, which is honest but defers it: 27 of 33 criteria are unclassified
  as discriminating or not. Classify them when each test is written, and consider a marking column then —
  `M11.3` originally required one and the table has no such column.

- [ ] **Sweep the spec for other descriptions firmed up into mechanisms.** Two are now known — `M5.D.4`
  (anger, revision 4) and `M1.A.3` (the transition at 50, revision 7) — and both were caught by reading the
  primary, not the extraction. **Neither was catchable by a regex**, which is why this is a reading task and
  not a guard. The shape to look for: a requirement whose source is a *descriptive* passage, whose hedges
  ("begins", "a few", "tends to", "usually") the requirement dropped, and which the spec then states as a
  threshold, a switch, or an exact count. Highest risk where a single chapter is the only source.
- [ ] **Re-read `KS05.2` on the capacity floor at 25.** `M1.A.4d` was regraded to a continuous falloff at
  revision 7 **on the principle rather than on a reading** — its source says "they lack the flexibility to
  make basic change", which is a description of a group and the same grammatical shape as Ch16's band. The
  falloff is applied; the *location* stands pending the primary. It is the one number `M1.A.3d` admits as
  structural, and it is admitted provisionally.
- [ ] **Re-read the source behind `M5.F.2`'s `outside_ness` threshold.** "Below threshold the same move
  produces the opposite sign" is the same shape as the transition at 50 — a threshold on a continuous
  quantity, doing load-bearing work in the sign-flip argument at `M15.D.4`. It may be sound; it has not been
  checked with this question in mind.

### Backlogged from the corpus-fidelity sweep, 2026-08-28 — graded LOW, recorded not fixed

The sweep checked **every** quoted passage in the spec: 214 total, 160 verbatim in the corpus, 52 legitimate
self-quotes of project documents, 0 unexplained. Integrity and modal findings are fixed; these are the
hedge-level remainder, held back because each fix is new unreviewed surface.

- [ ] **Ellipses that remove a qualifier.** `M1.A.19` drops "**would probably**" and a trailing condition
  ("as long as discussion touched emotional issues") from Ch10's neutral-third sentence. `M1.A.14c` drops
  "**provided his spouse functions in a reciprocal opposite way**", which is a condition on the whole claim.
  `M1.A.5d` splices two passages ~5,700 characters apart into one quotation and drops a "per se, **but…**"
  that states the harmful case. `M11.F.3a` truncates a Bowen epigraph with no ellipsis shown.
- [ ] **Hedges trimmed off the front of quotations.** `M1.A.13a` ("**I believe** the most important cue"),
  `M1.D.7b` ("**I believe** a spectrum of problems"), `M1.A.18d` ("not observed it **fully**"),
  `M8.6a` ("**as many as** fifty or sixty" flattened to "fifty to sixty").
- [ ] **Third parties quoted by an author, unmarked.** `M7.C.1c` rests on **Sylvia Nasar's** thought, quoted
  by Kerr. (`M4.D.5c`'s Andrew Solomon attribution is fixed; this one is the same shape.)
- [ ] **A scope narrower than the requirement.** `M5.B.3a`'s band is stated of "**the alcoholic person**";
  `M1.B.10` quotes a taboo-set claim made of **spouses** and applies it to every tie; `M15.D.4`'s "they could
  do no wrong" comes from the inpatient schizophrenia project, the most severe families studied.
- [ ] **`M4.D.3a`'s four-element evolutionary ordering has three sourced slots and one unsourced** —
  conflict's position between cooperation and dominant-adaptive is nowhere given in the source.
- [ ] **`M1.A.5a` quotes a phrase absent from the book it cites** ("more mature, more responsible for self",
  cited to Kerr 2019 ch6; nearest analogue is in ch24 and says something different). Its decomposition is
  tagged as user-derived, so the model claim survives — the citation does not.
- [ ] **`M2.A.0e`'s "almost identical" reads as Kerr-2019-grounded** and is in fact from KB Interview #14.
  Kerr 2019 says spouses marry at the "**same**" basic levels — stricter than the ±1 tolerance built on it.
- [ ] **`M5.C.1a` borrows severity from an adjacent intervention class.** The three catastrophes are real and
  correctly sourced to *forcing conjoint contact*; the recorded consequence of the peace-agree/reactive
  mismatch itself is relational rupture. Careful drafting, suggestive juxtaposition.
- [ ] **`M1.A.4d`'s source sentence is gate-shaped.** Kerr 2019: "**People above 25** on the continuum can
  make basic changes." Revision 7 replaced the gate with a continuous falloff on principle; the requirement
  should say plainly that the source's own wording is the gate it forbids.

- [ ] **Run a cold sweep over this batch.** The 2026-08-28 sweep was warm — the same session that wrote the documents commissioned it. It found fifteen defects, which says the sweep works, not that the set is clean. The project's own rule is that a falling finding count from self-review is not a stopping condition, and that a pass which does not know the change's history changes the *distribution* of findings. `/csdp --cold-sweep` over `4e79697..HEAD` from a fresh session.
- [ ] **The sweep covered one failure family only.** It reports itself as not covering: theoretical fidelity (it did not open `_LEDGER.md`, the `family_evaluation/` extractions, or Ch13's primary text), whether the ~88 new requirements are individually implementable, whether the specified model is coherent, and the reasoning in the DECISIONS document. **The `M5.D.4` correction rests on one re-read of one chapter by the context that made the correction** — a second reader on the sources is the highest-value follow-up.
- [ ] **Decide whether `M16`'s renderer output format is worth fixing in the spec.** `M16.C.2` says what a rendered line must carry and points at the proposal's §4.2 table as the shape, but does not fix a format. That is deliberate for now — the format is cheaper to settle against a running Phase B than in advance — but it should be settled *before* a second consumer exists, or the two will drift.
- [ ] **`M11.F.9(c)` deserves a worked demonstration before anyone relies on it.** The claim is that fitting one arm of a counterfactual to a known history breaks the error cancellation `M0.4` depends on. It is argued, not demonstrated. Phase E could show it directly: take a synthetic family, fit one arm to its own history, and measure how far the counterfactual moves against an unfitted control. If the effect is small the requirement is over-strong; if it is large it is the most important sentence in `M11.F`.

## Repository hygiene

- [ ] **No `LEARNINGS.md`.** The project convention is a repo-local fix log for repo-specific lessons, with generic patterns going to `~/.claude/standards/learnings.md`. Candidates: the ID-drift incident recorded in `M11.D.10`; the two unseeded flaky tests; and from 2026-08-28, **the `M5.D.4` heading inversion** — a requirement sourced from a summary heading rather than the sentence beneath it, which inverted silently and survived four spec revisions. The last of those may be general enough for the global catalogue rather than the repo-local log.

## Documentation hygiene, deferred

- [ ] **`§n` cross-references are ambiguous across documents.** All 50 in the v2 spec resolve against `model_explainer.md`, but several sentences address other documents ("proposal §9", "the frozen spec's §5.4") and `§5.4` exists in both. A traceability scan over `§` cannot tell the targets apart. Prefix them with the document.

## Sources

- [ ] **Locate the rest of the Kerr–Bowen interview series.** The folder holds #1 of a series; Kerr closes by saying later tapes "will concentrate on some of the more specific areas of the theory". Bowen names two he wanted covered: the difference between *distance* and *differentiation*, and how much of behaviour is intellectually directed versus emotionally reactive. An interview format with Kerr probing specific concepts is the highest-value form this material could take.
- [ ] **Probe the remaining 11 validation items against the 1979 lectures.** Tape 6 especially — it is the direct counterpart to Ch15, where this project withdrew both the seven-rung severity ladder and the "hidden dependence network", and a whole lecture on family reaction to death is the natural place to test both withdrawals.

## Carried from the theory work

- [ ] **`M11.C.9`'s and sibling position's status generally.** Ch13 omits sibling position entirely, and its effect size has no source. Consider whether it earns a place in the model at all.
- [ ] **Convergence C1 needs its scope re-derived from the chapter texts**, not from the pass-1 summaries. It was the headline finding at eleven chapters; Ch22 supplies a direct counterexample, so it likely survives only as a claim about *symptom relief obtained without structural change*.
- [ ] **The ego-mass terminology arc must be re-derived.** Ch08 actively retains and defends "undifferentiated family ego mass" against the timeline's claim that it was discarded at Ch05. The arc is messier than used → discarded → revived → abandoned.
- [ ] **When the basic level is fixed is genuinely unresolved** (`model_explainer.md` §13.3). Decided for the model — slow-moving with a ratchet, not frozen — but that is a modelling decision over a real disagreement, not a resolution of it.
