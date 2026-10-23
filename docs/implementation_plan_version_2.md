# Version 2: implementation plan

**Status: stage 1 APPROVED 2026-10-19; stage 2 plan DRAFTED 2026-10-23 and its pilot BUILT 2026-10-23 on the branch
`stage-2-plan-pilot` (§2.10), decisions D8–D16 taken as recommended under the owner's instruction of 2026-10-22, pending
the owner's review (§2.2, §2.10); the full sweep did not run (projected 44 h at 8 samples, over the 4-hour rule, §2.10); Phase C's 14 directional
criteria SWEPT 2026-10-23 on the branch `stage-2-phase-c-sweep` at 8 samples, 27 min (§2.10); the tooling to split the sweep across two Macs BUILT 2026-10-23 (§2.10).** The owner
decided D1–D7 on 2026-10-19, every one as recommended (§1.2), and approved the stage 1 plan. **Built, reviewed and regenerated 2026-10-19 (steps 0–7, §1.10): no verdict changed against
the tag `version-1`. The `/csdp` sweep's fixes (step 8) were made 2026-10-20 and pushed the same day with the owner's
approval; CI run 39449517708 passed on Python 3.11, 3.12 and 3.13 (S1.10c). A cold sweep's fixes were made
2026-10-20 (§1.10) and pushed the same day with the owner's approval (`0d86b29`); CI run 39482917825 passed on
Python 3.11, 3.12 and 3.13.** Written from
`docs/version_2_plan.md` §3 (stage 1) and the `TODO.md` entries it collects, each checked against the code on
2026-10-19. Later stages add their own sections here, as the Phase D plan did.

## 1. Stage 1: one engineering batch that moves the engine hash once

Stage 1's eight items (1.1–1.8) are listed in `docs/version_2_plan.md` §3. None is meant to change a result; any that
does is reported as a measured change (§1.6). This section adds what the code shows, the decisions the owner makes
before code, the criteria and their tests, the order, and the one run.

### 1.1 What the code shows (checked 2026-10-19)

Mapping each item to the code found five things that change the stage as the version 2 plan wrote it.

1. **Retiring the grid engine conflicts with the spec.** `M13.2` reads: "The frozen grid engine and its tests
   **MUST** stay green throughout." The owner's answer to Q6 cannot be built without amending it, so stage 1 needs a
   spec revision (D1).
2. **The 3.11 against 3.12 drift comes from every float `sum()`, not only from means.** Python 3.12 changed the builtin
   `sum()` over floats to compensated summation, so any float `sum()` can differ in its last bits between 3.11 and
   3.12, whether or not it is divided by a length. `src/bowen` has 129 lines with a builtin `sum(` call outside `render/`
   (engine 57, policy 8, readouts 7, ensemble 57). Some are integer counts, which are exact on every version. The
   float means item 1.1 names are about six lines (`reactive.py:187`, `appraise.py:254`, `iposition.py:340`,
   `learner.py:167`, `beliefs.py:145–146`). The float sums that are not means include the selection weights
   (`draws.py:201`, `policy.py:395`), the policy's channel totals, the budget and invariant totals, and
   `slow_tick.py`'s weighted means (lines 153–159), which use the builtin `sum` although the module already uses
   `math.fsum` elsewhere. (D2)
   `math.fsum` removes the Python-version source of drift. It does not remove the platform source: the C maths library
   (`exp`, `log`, `sqrt`) differs between macOS and Linux (`docs/TODO_archive.md`, the CI entry of 2026-10-15), so the golden
   fingerprints keep their rounding to 8 significant digits.
3. **Item 1.4 changes results; it is not an alignment.** `recompute_triangles` (`src/bowen/engine/recompute.py`)
   counts live positions from presence only. The only fused occupant the running engine builds is a coach who has
   taken a side (`external.coach_live`); no caller passes `visibility`'s `fusion` map, so family members are never
   fused into one another in a run. Aligning the code with `M8.5` as written therefore changes one thing: a triangle
   that holds a sided coach reads inactive. That changes `activation_memory` and every readout of triangle activity
   in a coached arm (`M11.C.7`, `.17`, `.18`, `.20`, `.33`). (D3)
4. **Three `TODO.md` entries are already done in the code** and should be closed, not rebuilt: the trace's framing
   sentence (`src/bowen/render/trace.py:45–47` chooses `SCRIPTED` or `POLICY`); `code_text`'s two latent gaps (fixed
   at Phase D plan B2, as its docstring says); the readouts' act sets (`config/bowen/pattern_acts.md`, loaded at
   `src/bowen/readouts/patterns.py:37`). Step 0 confirms each with a test before the entry is closed.
5. **Item 1.5's `run_in_copy` entry is partly done.** `tools/record_cache.py:run_cached` and
   `tools/phase_d_mutation_record.py:build` already keep the rows a child finished before it failed. What remains is
   that `run_in_copy` (`tools/mutant_runner.py:77`) raises `SystemExit` when its child fails, so one failed mutant
   still stops the record, and, by the `TODO.md` entry, a timed-out child's worker pool may be left running. Step 0
   checks the second against a real timed-out child.

### 1.2 Decisions for the owner before code

**D1 — A spec revision for stage 1.**
- (a) Stage 1 carries a short revision 33: `M13.2` amended to say the grid engine was retired at stage 1 and is kept
  at the tag `version-1`, and `M8.5` amended if D3 is (b). Group G's revision becomes 34.
- (b) No revision now: 1.6 waits for group G's revision, and 1.4 is recorded as an open mismatch until then.

*Recommended: (a).* **Decided 2026-10-19 (owner): (a).** The spec and the code then agree at stage 1's commit, as the documentation rule requires. The
grid engine is outside the engine hash, so either choice leaves the run unchanged.

**D2 — The scope of 1.1.**
- (a) As written: the float means in the seven named engine modules (about six lines).
- (b) Every float `sum()` in `src/bowen` outside `render/`, moved to `math.fsum`. Integer counts move to a small helper
  `count()` that refuses a float, and a test fails on any builtin `sum()` left in `src/bowen`.

*Recommended: (b).* **Decided 2026-10-19 (owner): (b).** (a) leaves most of the drift in place (learnings P5: fix the class, not the instance). Both move
the engine hash, and neither adds run time.

**D3 — Item 1.4: does a sided coach make its triangle inactive?**
- (a) Align the code with `M8.5` as written: a triangle that holds a sided coach is inactive. Results change in the
  coached arms. By the version 2 plan's sequence, a change of result belongs in a proposal.
- (b) Amend `M8.5` to say what the code does: whether a triangle is active is decided by presence; a coach's side is
  read for whether the coach's contact lands (`M11.C.7`) and for witness positions, not for whether the triangle is
  active. No result changes.
- (c) Leave both, and record the mismatch in `TODO.md`.

*Recommended: (b), with the theory question offered to stage 3 if you want it as a rule question.* **Decided
2026-10-19 (owner): (b).** There is a case each
way. For active: a coach who has taken a side is inside a triangle with that member, which is what being triangled
means. For inactive: under `M8` a fused position leaves two live positions, and the spec reads such a group as one
that cannot move its tension around. Stage 1 is not the place to settle that, because either answer to (a) is a rule
choice that changes results.

**D4 — Item 1.6: what is removed.**
**Decided 2026-10-19 (owner): remove as listed.** Recommended: remove from the tree `src/engine.py`, `src/main.py`, `monte_carlo.py`, `tests/test_simulator.py` (37
tests) and `docs/model_config.md` (the grid engine's configuration; step 1.6 confirms nothing in `src/bowen` or
`tools/` reads it). `tests/bowen/test_engine_purity.py` drops `src.engine` and `src.main` from its module list. Keep
`docs/bowen_individual_family_model_spec.md` as the frozen record, with a status line saying the engine it describes
was removed at stage 1 and is at the tag `version-1`. `CLAUDE.md`'s grid-engine rules (the purity violation, Axiom 1,
family sizes drawn at construction) become one line saying the engine is retired. Everything removed stays in git
history and at the tag. The two stray `sim_audit.csv` files are untracked output, so git cannot restore them; they
are listed for you to delete, not deleted by Claude.

**D5 — The Phase B mutation record (part of 1.5).**
- (a) Make it regenerable: restate gate G2's expectation against today's renderer. The gate would then run against
  today's engine, so it would no longer reproduce Phase B's close.
- (b) Freeze it, as version 1's status page is frozen: a status line saying it is the record of Phase B's close, and
  `tools/mutation_gate.py` refuses to write.

*Recommended: (b).* **Decided 2026-10-19 (owner): (b).**

**D6 — Item 1.7: which lock-in transfers get their own row in the trace.**
- (a) Every release: what a symptom held going back when it ends or passes its turn, and the spill to its partners
  past the turn (the `s < 0` branch and the ended-symptom branch of `symptoms.py`). The weekly hold stays summarised.
  No new constant.
- (b) (a), plus any weekly hold above a declared size, which needs a new `[I]` display constant in `config/viz/`.

*Recommended: (a).* **Decided 2026-10-19 (owner): (a).** Week 74's missing 222.4 is a spill past the turn, so (a) shows it.

**D7 — Riders: `TODO.md` entries that say "with the next engine change".** **Decided 2026-10-19 (owner): all five in.**
Recommended in:
- (i) the stale "pending owner review" text: ten sites in `src/`, `config/` and `tools/phase_d_record.py`, found by
  `grep -rn 'pending owner review' src config tools`, each checked against the owner's record before it is reworded
  (`tools/version_1_status.py`'s matches are its rules, not stale text);
- (ii) the pattern readouts take a `now` argument instead of reading `state.tick`, so `tools/famdiag_export.py`
  stops stepping the tick back and restoring it;
- (iii) `code_text` builds its lines as `tokenize` does (`io.StringIO(source).readlines()`, not `str.splitlines`);
- (iv) the Phase D mutation record's own hash reads `apply` and `run_in_copy` through `code_text`, not
  `inspect.getsource`, so a comment there does not rerun that record;
- (v) `tools/phase_c_trace.py` loads the family file once.

Recommended out: the coach's time on a side borrowing `triangle_activity_window` (a new constant, so a rule
question); `M11.C.29`'s redesign; the `M11.C.2` and `M11.C.47` traces. Each stays in `TODO.md`.

### 1.3 Criteria and their tests

Each test name carries the criterion's ID, and a spec ID where one applies. Each is shown failing under the named
mutation before it counts. The table is written for the recommended answers; another answer changes its row.

| ID | Criterion | Test | Proved failing by |
|---|---|---|---|
| S1.0 | The three `TODO.md` entries in §1.1 item 4 are done | `tests/bowen/test_render.py::test_s1_0_framing_names_a_policy_run`; `tests/bowen/test_ensemble_record.py::test_s1_0_code_text_keeps_code_on_a_docstring_line`; `tests/bowen/test_patterns.py::test_s1_0_act_sets_come_from_config` (each reused if an equivalent test exists) | reverting each fix on a copy |
| S1.1a | No builtin `sum()` in `src/bowen` outside `render/` | `tests/bowen/test_float_sums.py::test_s1_1a_no_builtin_sum_in_bowen` (an `ast` scan) | one builtin `sum` put back |
| S1.1b | `count()` refuses a float | `tests/bowen/test_float_sums.py::test_s1_1b_count_refuses_a_float` | `count` as the builtin `sum` |
| S1.2 | The canonical state includes `state.stressors` (`M16.T.3`) | `tests/bowen/test_log.py::test_m16t3_s1_2_canonical_state_sees_the_stressors`: two states that differ only in a stressor give different strings | the `stressors` key removed |
| S1.3a | `SIDE_TAKING` and `FAMILY_TO_EXTERNAL` are loaded from `config/bowen/act_classes.md` (rule 9) | `tests/bowen/test_config.py::test_s1_3a_act_classes_come_from_config`: each module attribute equals the loaded set, and neither module holds a literal set of act names | the literal put back |
| S1.3b | A misspelled act in that file fails at load (`M10.B.2`) | `tests/bowen/test_config.py::test_s1_3b_an_undeclared_act_fails_to_load` | the check removed |
| S1.4 | A triangle with a sided coach and a recent `TRIANGLE` act is active, while `coach_live` is false for it (`M8.5` as amended) | `tests/bowen/test_mechanisms.py::test_m8_5_s1_4_a_sided_coach_leaves_its_triangle_active` | the coach's side passed to `recompute_triangles` |
| S1.5a | A failed or timed-out child is that row's `ERRORED` outcome; the record run continues; a timed-out child's processes are gone | `tests/bowen/test_mutant_runner.py::test_s1_5a_a_failed_child_does_not_stop_the_record`, `::test_s1_5a_a_timed_out_child_leaves_no_process` (stub children, real processes, seconds) | `SystemExit` restored; the process-group kill removed |
| S1.5b | `code_text` splits lines as `tokenize` does | `tests/bowen/test_ensemble_record.py::test_s1_5b_code_text_keeps_a_line_separator_inside_a_string` (a U+2028 inside a string literal) | `str.splitlines` restored |
| S1.5c | The Phase B mutation record is frozen | `tests/bowen/test_phase_b_gate.py::test_s1_5c_phase_b_record_is_frozen`: the tool refuses to write and the record's status line says frozen | the refusal removed |
| S1.5d | One helper parses a record's JSON block | `tests/bowen/test_record_json.py::test_s1_5d_one_record_parser`: the block's pattern appears once across `tools/` and `tests/` | a second copy pasted back |
| S1.6 | The grid engine is retired | `tests/test_repo_layout.py::test_s1_6_the_grid_engine_is_retired`: the four files are gone and no file imports `src.engine`; `tests/test_doc_checklist.py` covers `docs/model_config.md` | `src/engine.py` restored |
| S1.7a | A lock-in release, and the spill past the turn, each have their own trace row (`M16.T.1`) | `tests/bowen/test_render.py::test_m16t1_s1_7a_a_release_has_its_own_row` (built records); the committed seed-7 trace's record-current test then covers week 74 | `lock_in` returned to the summarised set |
| S1.7b | The D2 record prints each variant under its own key | `tests/bowen/test_phase_d_record.py::test_s1_7b_variants_print_under_their_own_keys`: `M11.C.10` and `M11.C.10[family of origin]` are separate rows, as is `M11.C.2`'s twelve-person variant | rendering `Criterion.id` |
| S1.8a | `src/bowen/render/` is not in `code_hash` | `tests/bowen/test_ensemble_record.py::test_s1_8a_render_is_not_hashed`: a code edit under `render/` on a copy leaves the hash unchanged | `render/` included |
| S1.8b | No verdict reads the renderer | `tests/bowen/test_engine_purity.py::test_s1_8b_only_run_imports_render`: outside `render/`, only `src/bowen/run.py` imports `src.bowen.render` | an import added to a readout |
| S1.8c | The Phase D ensemble results are cached in `docs/records_cache/`, keyed as Phase C's are, and the Phase D records re-render from the cache | `tests/bowen/test_phase_d_record.py::test_s1_8c_records_render_from_the_cache`; dirty state (P8): `::test_s1_8c_a_stale_entry_is_rerun` | the key without the engine hash |
| S1.9 | The riders (D7) | (i) `tests/test_stale_text.py::test_s1_9_no_pending_review_text_for_decided_items`; (ii) `tests/bowen/test_patterns.py::test_s1_9_readouts_read_now`; (iii) is S1.5b; (iv) `tests/bowen/test_phase_d_mutation_record.py::test_s1_9_engine_hash_ignores_comments`; (v) no new test: the trace's record-current test covers it | (i) one stale string put back; (ii) `now` ignored; (iv) `inspect.getsource` restored |
| S1.10a | Every record is regenerated once on the new hash and matches its record-current test | the existing record-current tests | — |
| S1.10b | Each criterion's verdict, effect and p at the tag `version-1` against stage 1's records, in a generated page | `tools/record_changes.py` writes `docs/version_2_stage_1_changes.md`; `tests/test_record_changes.py::test_s1_10b_changes_page_is_current` | a hand edit to the page |
| S1.10c | The suite passes, and CI passes on 3.11, 3.12 and 3.13 | `python3 -m pytest tests/`; `.github/workflows/tests.yml` on the pushed commit | — |

The golden fingerprint fixtures change with S1.1 (learnings P23). They are regenerated in the same commit with the
reason in each fixture's docstring, and their test is shown failing against the old fixture first.

### 1.4 Order of work

| Step | Work | Moves the hash |
|---|---|---|
| 0 | Confirm §1.1 items 4 and 5 (S1.0); close the three done `TODO.md` entries | no |
| 1 | Tooling outside the hash: S1.5a, S1.5c, S1.5d, S1.9 (v); the grid engine (S1.6) | no |
| 2 | The hash's definition: S1.8a, S1.8b | yes (definition) |
| 3 | Engine and hashed tools: S1.1, S1.2, S1.3, S1.4's test, S1.5b, S1.9 (i), (ii), (iv) | yes |
| 4 | Renderers, now outside the hash: S1.7a, S1.7b; the Phase D cache, S1.8c | no |
| 5 | Spec revision 33 (D1) and every document in §1.7 | no |
| 6 | One `learning-qa` pass and one correctness pass over the whole diff; every finding fixed before the run | — |
| 7 | The run (S1.10a) in the background; then S1.10b, `docs/spec_coverage.md`, the C16 report, the seed-7 trace and chart | — |
| 8 | Report every changed verdict to the owner (§1.6); then `/csdp` | — |

Step 6 comes before the run so that a review finding in the engine does not cost a second run (`CLAUDE.md`, "batch
fixes before any rerun that is still slow").

### 1.5 Compute

One full regeneration: about 2 hours on 10 cores, plus the mutation records (`docs/version_2_plan.md` §6). It runs
once, at step 7, with the estimate stated before it starts. After S1.8, a change to the trace renderer or a Phase D
record's renderer re-renders in seconds.

### 1.6 How a changed result is reported

S1.1 changes the last bits of many sums, and the selection draws read some of them, so a seed's trajectory can change
from the week a draw lands on a boundary. Verdicts are expected to hold, but that is a prediction, not a measurement.
`docs/version_2_stage_1_changes.md` lists every criterion whose verdict changed and every effect that moved by more
than its stated precision. A changed verdict is reported to the owner before stage 1 is checked off. It is a measured
change, not a defect to be fixed in this stage; if the owner wants its cause, each item's effect can be separated with
ad hoc runs of that criterion alone.

### 1.7 Documents each step updates

- `docs/bowen_agent_model_spec_v2.md`: revision 33 (`M13.2`, `M8.5`), its status line (version 2 begins at revision
  33, Q9).
- `docs/model_explainer.md`: the grid engine's retirement, the canonical state's coverage, triangle activity (D3).
- `config/bowen/act_classes.md` (new) and `config/bowen/constants_changes.md`.
- `docs/bowen_individual_family_model_spec.md`: its status line.
- `docs/SPEC_REVIEW_CHECKLIST.md`: this plan, the changes page, `act_classes.md`, `docs/model_config.md` removed.
- `docs/phase_b_mutation_record.md`: its status line (D5).
- `docs/version_2_plan.md` (stage 1's status line), `TODO.md`, `CHANGELOG.md`, `README.md`, `CLAUDE.md` (the
  grid-engine rules, the suite count), `docs/theory/_STATUS.md`.
- Every regenerated record, `docs/spec_coverage.md`, and the seed-7 trace and chart.

### 1.8 Not code-testable

- D3's theory question, if the owner takes it to stage 3.
- Whether a measured change of verdict is acceptable: the owner reads `docs/version_2_stage_1_changes.md`.
- Whether the trace reads well with the new lock-in rows: the owner reads week 74 of the seed-7 trace.

### 1.9 Adjacent issues found while planning, not fixed

- `math.fsum` of an empty iterable is `0.0` where the builtin `sum` gives `0`. Where an empty total is printed, a
  record can change from `0` to `0.0`. The regeneration shows any such line; none is expected to change a verdict.
- `M1.F.1b` and `M3.E.1` say visibility uses `M8.5`'s predicate with fusion, but no caller passes a fusion map, so
  family members are never fused in a run. This is outside stage 1. It belongs with D3's question if the owner takes
  that to stage 3.
- `docs/version_2_plan.md` item 1.1 names `slow_tick.py` and `life_course.py`; both already use `math.fsum` for some
  sums, and `slow_tick.py` still uses the builtin `sum` for others. D2 (b) covers both.
- `docs/TODO_archive.md`'s rule 9 entry (the readouts' act sets) is half done: the sets are in config, and only the engine's two
  classifications remain (S1.3).

### 1.10 Build record (2026-10-19)

Steps 0–5 are built: commits `423bbed` (steps 0–2), `b6ca902` (step 3), `36fad6c` (step 4) and `4bea435` (step 5's
documents). The step 6 review's fixes follow the table.
Each criterion's test was shown failing under its mutation on a copy of the repository before it counted; the
mutations are edits applied by a scratch tool and reverted, and none is kept in the tree.

| ID | Test | Mutation | Result |
|---|---|---|---|
| S1.0 | `tests/bowen/test_render.py::test_b2_framing_says_who_chose_the_acts` (reused) | the framing always says scripted | fails |
| S1.0 | `tests/bowen/test_ensemble_record.py::test_b2_code_text_drops_only_docstrings_and_comments` (reused) | a docstring's whole line dropped; blank lines inside a string stripped | fails, each |
| S1.0 | `tests/bowen/test_patterns.py::test_s1_0_act_sets_come_from_config` | the `APPROACH` literal put back with the same values (the older config test passed under it) | fails |
| S1.1a | `tests/bowen/test_float_sums.py::test_s1_1a_no_builtin_sum_in_bowen` | `reactive.py`'s `math.fsum(others)` back to `sum(others)` | fails |
| S1.1b | `tests/bowen/test_float_sums.py::test_s1_1b_count_refuses_a_float` | `count` as the builtin `sum` | fails |
| S1.2 | `tests/bowen/test_log.py::test_m16t3_s1_2_canonical_state_sees_the_stressors` | the `stressors` key removed | fails (and failed before the fix) |
| S1.3a | `tests/bowen/test_config.py::test_s1_3a_act_classes_come_from_config` | the family→external literal put back in `policy.py` | fails |
| S1.3b | `tests/bowen/test_config.py::test_s1_3b_an_undeclared_act_fails_to_load` | the declared-move check removed | fails |
| S1.4 | `tests/bowen/test_mechanisms.py::test_m8_5_s1_4_a_sided_coach_leaves_its_triangle_active` | the coach's side passed to `recompute_triangles` | fails |
| S1.5a | `tests/bowen/test_mutant_runner.py::test_s1_5a_a_failed_child_does_not_stop_the_record` | the `SystemExit` restored | fails |
| S1.5a | `tests/bowen/test_mutant_runner.py::test_s1_5a_a_timed_out_child_leaves_no_process` | the process group's kill replaced by `child.kill()` | fails at its 40 s deadline |
| S1.5b | `tests/bowen/test_ensemble_record.py::test_s1_5b_code_text_keeps_a_line_separator_inside_a_string` | `str.splitlines` restored | fails (and failed before the fix) |
| S1.5c | `tests/bowen/test_phase_b_gate.py::test_s1_5c_phase_b_record_is_frozen` | the refusal to write removed | fails |
| S1.5d | `tests/bowen/test_record_json.py::test_s1_5d_one_record_parser` | the block's pattern pasted back into `tools/sweep_record.py` | fails |
| S1.6 | `tests/test_repo_layout.py::test_s1_6_the_grid_engine_is_retired` | `src/engine.py` restored | fails |
| S1.7a | `tests/bowen/test_render.py::test_m16t1_s1_7a_a_release_has_its_own_row` | `lock_in` back in the summarised set | fails |
| S1.7b | `tests/bowen/test_phase_d_record.py::test_s1_7b_variants_print_under_their_own_keys` | the verdict keeps `Criterion.id` | fails |
| S1.8a | `tests/bowen/test_ensemble_record.py::test_s1_8a_render_is_not_hashed` | the `render/` exclusion removed | fails |
| S1.8b | `tests/bowen/test_engine_purity.py::test_s1_8b_only_run_imports_render` | an import of the renderer added to `readouts/patterns.py` | fails |
| S1.8c | `tests/bowen/test_phase_d_record.py::test_s1_8c_records_render_from_the_cache` | the cache never read | fails |
| S1.8c | `tests/bowen/test_phase_d_record.py::test_s1_8c_a_stale_entry_is_rerun` (P8) | the key without the engine hash | fails |
| S1.9 (i) | `tests/test_stale_text.py::test_s1_9_no_pending_review_text_for_decided_items` | one reworded site put back as "pending owner review" | fails |
| S1.9 (ii) | `tests/bowen/test_patterns.py::test_s1_9_readouts_read_now` | the readouts ignore `now` | fails |
| S1.9 (iv) | `tests/bowen/test_phase_d_mutation_record.py::test_s1_9_engine_hash_ignores_comments` | raw `inspect.getsource` restored | fails |

S1.9 (iii) is S1.5b, and (v) is covered by the trace's record-current test. S1.10a–c are steps 7 and 8.

**Where the build differs from the plan.**
- **The golden fingerprints did not move.** §1.3 expected S1.1 to change them. Measured with S1.1–S1.4 in place, both
  seeded runs give the same fingerprint at 8 significant digits, so no fixture was regenerated and there was no old
  fixture to show failing. `tests/bowen/test_engine_golden.py`'s docstring records this.
- **S1.3a removed the module attributes** instead of keeping attributes equal to the loaded sets. The engine reads
  `kinds.side_taking` from the run's kinds and the policy reads `family_to_external` and `external_repertoire` from
  the kinds it is given, so no copy of a set can drift. The test checks the loaded classes, the absence of a literal
  in either module, and that the two policy classes are the lists `M5.B.4` and `M5.B.5` name.
- **S1.5a's code was first put in `tools/mutant_runner.py`**, which is in the engine hash, so it moved the hash at
  step 1. The step 6 review moved it to `tools/child_process.py`, outside the hash: how a child is run does not change
  what it computes (`CLAUDE.md`, learnings P38). S1.5a still moved the hash this stage, since `run_in_copy` in
  `tools/mutant_runner.py` changed; from now on a fix to how a child is run does not. No record was regenerated
  between steps, so the run at step 7 is still the only one.
- **S1.9 (ii) was already built.** The pattern readouts took `now`, and `tools/famdiag_export.py` stopped stepping
  `state.tick`, in `f916d55` (Phase D B2, 2026-10-10). Stage 1 added only the test.
- **S1.1's scan covers `render/` too.** §1.3 left the renderer out, as no verdict reads it. The review found its two
  sums (a float total and a count in `render/trace.py`); they are now a `math.fsum` and a `count`, and the scan reads
  every file in `src/bowen`.
- **The Phase D cache's engine part is the hash the Phase D records print** (`code_hash()` with no tool), not
  `engine_hash()`, which adds `tools/mutant_runner.py`; the Phase D ensembles do not run through that tool. The key is
  otherwise built as Phase C's are (`result_key`).

**The step 6 review (2026-10-19).** One `learning-qa` pass and one correctness pass over `a9ee812..4bea435`. Every
finding was fixed before the run:
- **`tools/record_changes.py` (S1.10b).** It matched the tag's D2 rows only where a key repeated, so `M11.C.2`, which
  the tag prints for the twelve-person family, was never compared. It now matches the D2 record's rows by position,
  each to a current key naming the same criterion. It also compares the level-occupancy record. It lists each record
  it does not compare, with the reason. It counts a readout found on only one side as a difference. And it refuses a
  record whose hash is the tag's, because that record was not regenerated (P35).
- **S1.5a.** A failed child's error output is printed in full, not only its first line. The code moved out of the
  hash (above). A SIGTERM or SIGHUP to the parent kills the child's process group, because a child in its own
  session gets no signal sent to the parent. A JSON line cut short fails the child instead of stopping the record on
  a decoding error.
- **`DrawService.categorical`** fell back to the last index when rounding at the top edge passed every running sum.
  When the last weight was zero, that index had no weight. It now returns the last index with weight, as
  `policy._invert` does.
- **`canonical_state` left out six fields of the run's state.** These are the event queue, the self-directed acts and
  witnessed cutoffs awaiting judgement, `adjusted`, `stances` and the estimator's history (`M16.T.3`). They are now
  in, and a test walks every `RunState` field. `NOT_CANONICAL` names the two left out, the event kinds and the draw
  service, each with its reason.
- **The act classes (S1.3).** `EventKinds`' three classes default to `None`, so kinds built without them fail where a
  class is read; an empty set would have put every act in no class. The loader refuses a class with no acts and a
  side-taking act that is not a family-to-external move. `ACT_CLASSES` moved to `engine/events.py`, beside the
  fields it names.
- **The act-literal scan (S1.0, S1.3a)** now finds list and tuple displays of act names, and a comparison with one
  act's name, as well as sets.
- **`count`** takes numpy booleans, a counted condition over an array.
- **`tests/test_repo_layout.py`** fails if `docs/model_config.md` comes back. Before, only the documentation index
  caught it.
- **The explainer's §13.1 note** said the live-position predicate does not decide whether a triangle is active. It
  now says which of its three callers is fed a side (only `external.coach_live`) and which are fed presence only.
- **This record and the CHANGELOG** said stage 1 built S1.9 (ii); it was built in `f916d55` (above).

Each new test was shown failing under its mutation, in the same way as the table above:

| Test | Mutation | Result |
|---|---|---|
| `tests/test_record_changes.py::test_s1_10b_the_tags_d2_rows_are_matched_by_position` | the D2 rows matched by key, not position | fails |
| `tests/test_record_changes.py::test_s1_10b_a_record_not_regenerated_is_refused` | the unchanged-hash check removed | fails |
| `tests/test_record_changes.py::test_s1_10b_a_changed_verdict_and_a_moved_effect_are_listed` (P7) | any movement counted, not only past the tag's half-width | fails |
| `tests/test_record_changes.py::test_s1_10b_a_value_record_lists_a_changed_value` | a value row's nested values not read | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_failed_child_does_not_stop_the_record` | the failure not printed; the `ChildFailed` re-raised | fails, each |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_timed_out_child_leaves_no_process` (re-proved in `child_process.py`) | the process group's kill replaced by `child.kill()` | fails at its 40 s deadline |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_stopped_parent_leaves_no_process` | the stop-signal handler never set | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_line_cut_short_is_reported` | the cut-line check removed | fails |
| `tests/bowen/test_draws.py::test_m3d4a_categorical_never_returns_a_zero_weight_index_at_the_top_edge` | the fallback back to the last index | fails (and failed before the fix) |
| `tests/bowen/test_log.py::test_m16t3_canonical_state_covers_every_field` | the `queue` key removed | fails |
| `tests/bowen/test_config.py::test_s1_3a_kinds_without_act_classes_fail_where_a_class_is_read` | the default back to an empty set | fails |
| `tests/bowen/test_config.py::test_s1_3b_an_undeclared_act_fails_to_load` | the empty-class check removed; the side-taking check removed | fails, each |
| `tests/bowen/test_patterns.py::test_s1_0_the_scan_finds_each_form_of_act_literal` | sets only; comparisons not read; a tuple assigned to names reported | fails, each |
| `tests/bowen/test_float_sums.py::test_s1_1b_count_refuses_a_float` | a numpy boolean not converted | fails |
| `tests/bowen/test_float_sums.py::test_s1_1a_the_scan_finds_a_sum` | the `render/` exclusion put back in the scan | fails |
| `tests/bowen/test_float_sums.py::test_s1_1a_no_builtin_sum_in_bowen` | `render/trace.py`'s `math.fsum` back to `sum` | fails |
| `tests/test_repo_layout.py::test_s1_6_the_grid_engine_is_retired` | `docs/model_config.md` restored from the tag | fails |

**The fix re-sweep (2026-10-19).** One narrow `learning-qa` pass (sonnet) over `08a2f71`, the review's fixes, found
one major item and five minor ones, all fixed before the run:
- **Major: the stop handler replaced an ignored SIGHUP**, so a run under `nohup` would stop when its terminal
  closed. A signal the parent ignores now stays ignored. The handler acts once, so a second signal cannot interrupt
  the kill, and `Popen` is inside the `try` that kills the group. A signal that arrives inside `Popen` itself, after
  the fork, can still leave the child running; that window is narrowed, not closed (closed at step 8, below).
- The S1.5a bullet above claimed the hash no longer moved for S1.5a; `run_in_copy` itself changed, so it did.
- `test_s1_3a_act_classes_come_from_config`'s docstring now says the scan sees strings, not sets built from name
  constants (`NEEDS_TRIAD`).
- `tools/record_changes.py`'s one-sided readouts and its refusal of a record with no hash line now have tests.
- `config/bowen/act_classes.md` states the two refusals the loader gained.

| Test | Mutation | Result |
|---|---|---|
| `tests/bowen/test_mutant_runner.py::test_s1_5a_an_ignored_stop_signal_stays_ignored` | an ignored signal given the handler | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_second_stop_signal_cannot_interrupt_the_kill` | the handler not made one-shot | fails |
| `tests/test_record_changes.py::test_s1_10b_a_readout_on_one_side_only_is_listed` | a one-sided readout not listed | fails |
| `tests/test_record_changes.py::test_s1_10b_a_record_not_regenerated_is_refused` | the missing-hash check removed | fails |

**The run (step 7, 2026-10-19).** One full regeneration, from 20:00 to 23:36 (3 h 35 min, against §1.5's estimate of
about 2 hours on 10 cores plus the mutation records). Every step exited 0 and no row is FAILED. Times:

| Step | Time |
|---|---|
| Phase C ensemble | 217 s |
| Phase C mutation | 1547 s |
| Sweep | 391 s |
| Level occupancy | 35 s |
| Diagnostic | 449 s |
| Phase D ensembles, part I | 764 s |
| Phase D ensembles, part II (D2) | 5655 s |
| Phase D mutation | 3866 s |

`docs/version_2_stage_1_changes.md` (S1.10b) compares the rows of the records in its table with the tag `version-1`'s,
every field (since the step 8 sweep; before, verdicts and mean differences only): **no verdict changed, no effect
moved, and no value differs at all**, across 24 Phase C ensemble rows, 25 Phase D ensemble rows, 6
D2 rows, 114 Phase C mutation rows, 14 Phase D mutation rows, 54 sweep rows and 4 level-occupancy rows. The page lists
the records and generated pages it does not compare under "Not compared here". The records outside the page were
checked by hand against the tag:
- `docs/phase_c_diagnostic_record.md` changed only its hash line.
- In the Phase D records, every row's full JSON (keys sorted) is the tag's, apart from the criterion labels S1.7b
  changed. The rest of their diff is key order, from rendering through the cache (S1.8c), and `M11.C.9`'s note,
  "pending owner review" → "decided by the owner 2026-10-17" (S1.9 (i)).
- The seed-7 trace gained the lock-in release rows (S1.7a); its header's config hash and revision and the sentence
  that names the system rows changed with them, and no other row changed. The chart's series are the tag's; its
  `RUN` block differs in the config hash, the revision and the `absent` list its renderer added after the tag.
- `docs/review/phase_c_famdiag_seed7.json` was not regenerated, because the working copy holds the owner's
  uncommitted edits to it. Built from the current engine and compared with the committed export, it differs only in
  its note, which names revision 32 and the tag's config hash.
- `docs/phase_c_completion_report.md` §11 regenerated unchanged.

Stage 1 therefore changed no computed result. §1.3 expected S1.1 (exact float totals) and S1.4 (triangle activity) to
move some; at these seeds and arm sizes, none did.

**The `/csdp` sweep (step 8, 2026-10-20).** The batch's one full `learning-qa` pass and correctness pass were step 6,
so, by the owner's rule on sizing reviews (`CLAUDE.md`), step 8 sent only what came after them through three narrow
passes on sonnet: a `learning-qa` re-sweep of `08a2f71...7aa9fc5` (the fix re-sweep's items and step 7's documents,
generated records left out), a correctness pass over `d361646`'s code, and a security pass over the branch's
subprocess, cache and record-parsing code (`tools/child_process.py`, `mutant_runner.py`, `record_cache.py`,
`record_changes.py`, `phase_d_record.py`, `record_json.py`), which no earlier pass had covered. The program had been
run end to end on the final code: the regeneration started 24 s after `d361646` was committed. Fixed, each with a
test shown failing under its mutation:
- **Medium: `tools/record_changes.py` compared only verdicts and mean differences**, while the page, this record, the
  CHANGELOG and the commit said no value differed. It now compares every field of a row on both sides (each readout's
  p and half-width, the seed count, the flags), lists each field that differs, says "No change" only when none does,
  counts a NaN on both sides as no difference and a mean difference that became NaN as moved, and refuses a record
  with no row on both sides. Rerun, it finds no difference in any record: the claim now rests on the tool, not on a
  separate check.
- **A stop signal while `Popen` returned left the child running** (found by the correctness and security passes, the
  window the fix re-sweep had narrowed): a stop that arrives during the spawn now raises once `child` is set, so the
  `finally` kills the group. The stop handlers are set inside the `try`, and a signal during the restore loop goes
  on to the restored handler instead of raising with the others left ignored. A stop in the instant after the body
  ends and before the restore begins still raises with them ignored; nothing catches it, so the process exits.
- **The cache was written in place**, so a kill mid-write left a file the next run could not read; it is written
  beside the cache and moved into place. A decoded child line with no criterion is counted and its id is FAILED,
  instead of stopping the record with a `KeyError`.
- Tests for two of `d361646`'s fixes that had none (a handler not set from Python restored as the default; a `Popen`
  that fails raises its own error), and the one-shot test now checks every stop signal, not only the one sent.
- This section's run record: the mutation row was labelled "mutation and sweep" and left out the sweep's own 391 s,
  the estimate was cited from the wrong section, and the trace and chart claims named less than their diffs held.
- `docs/version_2_stage_1_changes.md` lists the fam-diag export among what it does not compare, with why.

| Test | Mutation | Result |
|---|---|---|
| `tests/test_record_changes.py::test_s1_10b_every_field_is_compared` | seed count and flags not compared; "No change" ignoring differing fields; NaN unequal to NaN; NaN not moved | fails, each |
| `tests/test_record_changes.py::test_s1_10b_a_record_not_regenerated_is_refused` | no refusal when no row is compared | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_stop_signal_during_the_spawn_still_kills_the_child` | the spawn not deferred | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_stop_signal_while_restoring_reaches_the_restored_handler` | no pass-on while restoring | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_stop_signal_while_setting_the_handlers_restores_both` | handlers set outside the `try` | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_handler_not_set_from_python_is_restored_as_the_default` | `None` restored as is | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_child_that_cannot_start_raises_its_own_error` | the `child is None` guard removed | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_second_stop_signal_cannot_interrupt_the_kill` | only the signal that came ignored | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_cache_write_cut_short_leaves_the_last_whole_file` | the cache written in place | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_failed_child_does_not_stop_the_record` | a line with no criterion read as a row | fails |

The fix commit (`66de943`) had one narrow `learning-qa` re-sweep on sonnet: no high or medium item, six low ones in
its code, all fixed (each test shown failing under its mutation, listed below): a readout named twice in one row is
refused, since the second hid the first; the difference tables print a float in full, so a last-bit change shows;
a child line is a row only with a text criterion and outcome and any error as text (one with no outcome, or a null
error, had stopped the record); the cache's write in progress has one file per process, and git ignores it; the
pass-on of a late signal is stated where it ends the process at once (the default handler), and the claim about the
handlers' restore is narrowed to what the code does. Generated records still written in place are recorded as
adjacent.

| Test | Mutation | Result |
|---|---|---|
| `tests/test_record_changes.py::test_s1_10b_every_field_is_compared` | a repeated readout name accepted; floats rounded to 6 digits | fails, each |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_failed_child_does_not_stop_the_record` | a line with a criterion and nothing else read as a row | fails |
| `tests/bowen/test_mutant_runner.py::test_s1_5a_a_cache_write_cut_short_leaves_the_last_whole_file` | the cache written in place | fails |

Recorded in `TODO.md`, not fixed: the changes page is not frozen at stage 1 (a later regeneration would rewrite it
under stage 1's title; freezing it needs a ref to read stage 1's records from, which is the owner's choice, as the
tag `version-1` was); a stop during the repository copy or its cleanup leaves a `bowen-mutant-*` directory in the
temporary directory (no process); two runs of one record tool at once overwrite each other's cache; the timed-out
path's second `communicate()` has no timeout (not reachable: no worker leaves the group). Checked and left: the
mutants' file paths are not confined to the copy, but every one is a constant in the repository.

**Adjacent issues found while building, not fixed** (each is in `TODO.md` or below):
- Act sets still written in the code outside the two classes S1.3 moved: `NEVER_REFUSED` (`rejection.py`),
  `ABORT_ACTS` (`iposition.py`), the sets in `sinks.py` and `moves.py`, and `NEEDS_TRIAD` in `policy.py` (rule 9).
- `tools/probes.py` still takes float means with the builtin `sum` (lines 139, 141, 268, 458). It is outside
  `src/bowen` and no verdict reads it.
- `monte_carlo_results.txt`, the grid engine's output, is still tracked.
- `tools/mutation_gate_phase_b.json` names `tests/test_simulator.py`; it is an input to the frozen Phase B record
  (D5) and is left as it was.
- `src/sim_audit.csv`, an untracked output of the retired grid engine, is the owner's to delete.
- The widened act-literal scan finds act names in `ensemble/criteria.py`, `criteria_d.py` and `criteria_d2.py`.
  These are criterion definitions naming the acts each criterion reads, not engine rules; whether they belong in
  config is for a later stage (rule 9).
- The Phase D ensembles (`tools/phase_d_record.py`) run with no timeout, and `tools/diagnostic_record.py:136` and
  `tools/level_occupancy.py:145` stop when their child fails, where the Phase C records go on (S1.5a).

**Cold sweep (2026-10-20).** After the push, one `learning-qa` pass in a fresh session, which had not seen the
earlier passes' findings, swept `579170c...3d70d3e`. It found nine items, C1–C9: none high, C1 and C2 medium, the
rest low or plausible. The warm passes had recorded four (C2, C6, C7, C8) and missed five (C1, C3, C4, C5, C9) and
the level-occupancy half of C2. It confirmed that no verdict changed against the tag. Fixed, each test shown
failing under its mutation on a copy:
- **C1 (medium):** `tools/record_cache.py`'s `exit_status` counted only FAILED rows, so a run with a RAISED or
  BROKEN row (written into the record, never cached) exited 0 while the record's test failed. Every errored row now
  gives exit 1, and the message says that rerunning will not change a row that raises every time.
- **C5:** `docs/version_2_stage_1_changes.md` now lists the frozen Phase B mutation record and version 1's status
  page among what it does not compare, with why; a test fails when a page under `docs/` whose header says a tool
  generated it, or a committed file under `docs/review/`, is in neither list. The fix's re-sweep found the Phase B
  seed-7 traces in neither list too; they are listed. The page was regenerated.
- **Level occupancy:** `tools/level_occupancy.py` refuses to write its record unless the child printed one row for
  each of C.16's arms, in order. The tool is in its record's hash, so the record was rerun (35 s): only its hash
  line changed.

| Test | Mutation | Result |
|---|---|---|
| `tests/bowen/test_ensemble_record.py::test_m111d_a_raised_or_broken_row_exits_nonzero` | only FAILED rows counted, as before | fails |
| `tests/test_record_changes.py::test_s1_10b_every_generated_record_is_compared_or_listed_as_not` | `docs/version_1_status.md` taken out of `NOT_COMPARED`; the Phase B traces taken out | fails, each |
| `tests/bowen/test_ensemble_record.py::test_m111a_level_occupancy_refuses_a_short_child` | the row check skipped | fails |

Recorded in `TODO.md`, not fixed, each older than stage 1: C3, `config/bowen/*.md` is hashed as raw bytes, so a
prose edit reruns every record; C4, the worker pools that `tools/phase_d_record.py` and `tools/ensemble_record.py`
start are not killed as a group on a stop signal; C9, errors of the machine are matched by exact class name, so an
`OSError` subclass is a RAISED row (already in `TODO.md` since 2026-10-09; the entry is brought up to date). Also
recorded, for the owner: a row that raises on every run is never cached, so its record can never be committed.

These fixes had one narrow `learning-qa` re-sweep on sonnet (about 8 minutes, about 115k tokens). It found two high
items, both stale artifacts that turned the suite red: `docs/spec_coverage.md` did not list the new tests, and the
test count in `CLAUDE.md` was 924, not 927; both are updated. Its four low items are fixed: the C5 test now also
covers every committed file under `docs/review/`, which found the Phase B seed-7 traces in neither list (they are
listed); the claims about that test say which pages it sees; the counts above name each item; the `TODO.md` entry
for the owner names sweep settings and diagnostic variants as well as mutants, and the C9 entry no longer calls
`BrokenExecutor` a subclass.

## 2. Stage 2: measurement — the constant sweep, its pilot, the detection-limit rule and the version 2 status page

Stage 2's five items (2.1–2.5) are listed in `docs/version_2_plan.md` §3. None changes what a run computes: the engine
hash stays `66dcbb1e8ba5` through the stage, and no ensemble, mutation or diagnostic record is rerun. The stage
measures how far each directional result holds when the invented constants its mechanism reads are moved over declared
ranges (`M17.E.1`), after a timed pilot sets the cost. Every decision below is **decided as recommended under the
owner's instruction of 2026-10-22, pending the owner's review** ("you can run all night - proceed based on your
recommendations - do not wait for my input"); the owner's answers to Q3 and Q4 of `docs/version_2_plan.md` §5
(2026-10-19) stand and are not re-decided here.

### 2.1 What the code shows (checked 2026-10-23)

1. **Of `M17`'s 29 requirements, four have code and tests** (`docs/spec_index.jsonl`, `docs/spec_coverage.md` rows
   583–611): `M17.A.1` done (the adaptive stop, `src/bowen/ensemble/runner.py::run_criterion`, five `test_m17a1_*`
   tests), `M17.A.3` partial (the paired statistic and Holm's step), `M17.A.4` done (the margin), `M17.D.3` done (the
   engine is arm-blind). The other 25, `M17.E.1` among them, have no code and no test: "Phase E builds it and is not
   built yet". Stage 2 builds `M17.E.1`'s sweep and the one sentence of `M17.A.1` that D8 adds; nothing else in `M17`.
2. **A sweep already exists, and it is not `M17.E.1`'s.** `tools/sweep_record.py` (Phase C plan D9) moves three of
   the learner's constants one at a time to half and double their central value and reports each composite
   criterion's verdict there: 54 rows, 5 min 42 s at revision 36's regeneration (`CHANGELOG.md`, 2026-10-23). It
   answers "does the verdict survive at these six points", not "over what fraction of the joint range does the
   direction hold", and it gives no fraction. It stays as it is: its settings are in the engine hash through
   `HASHED_TOOLS`, so changing it would rerun it.
3. **How a sweep run reads the `[I]` constants without touching `src/bowen/` or the hash.** `tools/mutant_runner.py`
   copies `src/`, `config/` and `tools/` to a temporary directory, applies a `Mutant`'s edits there, and runs
   `run_criterion` in a child whose constants are the copy's `config/bowen/constants.md` (`load_constants()`); the
   result is keyed by the engine hash of the working tree and the edits (`tools/record_cache.py::result_key`). A
   sweep sample is therefore a `Mutant` whose edits are the sampled values written into that file's rows, as the D9
   sweep's `row_mutant` already does for one row; a joint sample uses `Mutant.also` for the further rows. The runner's
   stopping rules are rows of the same file (`ensemble_block`, `ensemble_cap`, `ensemble_precision`), so a fixed seed
   count is an edit too (block and cap both set to the count: `run_criterion` runs one block and stops). The working
   tree is never edited and `tools/mutant_runner.py` does not change, so `engine_hash()` is unchanged.
4. **The engine hash covers every `config/bowen/*.md`** (`tools/ensemble_record.py::code_hash`, with
   `constants_changes.md` the one exclusion). A new file there — the sweep's own ranges and rules — would move the hash
   and make every record stale. The sweep's config therefore lives outside `config/bowen/` (D12). The engine never
   reads it; only `tools/` does.
5. **What the pilot needs that does not exist:** a driver that reads declared ranges, draws joint samples, runs each
   as a cached `Mutant`, applies the detection-limit rule and reports the fraction (D8, D10); a sweep record rendered
   from a cache (D11); a version 2 status page with a sweep column (D11), since `docs/version_2_status.md` does not
   exist and `tools/version_1_status.py` is frozen at the tag. Its `record_rows()` no longer runs on `main`'s records
   (`phase_d2_ensemble_record.md` prints each variant under its own key since S1.7b and has `M11.C.48`'s two rows, so
   its in-order check against `CRITERIA_D2` fails); it is frozen, so this is not a defect, and the version 2 tool reads
   the records' keys directly.
6. **The directional criteria, as the records stand at revision 36** (`class` composite or mixed; `M11.5`): 34 rows —
   14 in the Phase C record, 13 in Phase D part I, 7 in part II. 12 pass (`M11.C.16`, `.44`,
   `.27[unstable,remove_one]`, `.41[light: lower level]`, `.41[heavy: lower level]`, `.22`, `.11`, `.39`,
   `.7[joint over sided]`, `.15`, `.48`, `.48[phase_d]`) and 22 fail. Five of the 34 changed verdict at revision 36
   (T-122); `M11.C.44`'s pass has no recorded mutation yet, and `M11.C.48`'s pass is not evidence (T-123).
7. **Record times at revision 36's regeneration, 10 workers** (`CHANGELOG.md`, 2026-10-23): the Phase C ensemble
   record 3 min 10 s for 24 rows; Phase D part I 12 min 7 s for 25; part II 1 h 31 min for 8. These are the only
   measured per-run costs; §2.5 uses them with the pilot's measured ratio and not before.

### 2.2 Decisions

Each is taken as recommended. The owner's review may reverse any of them; a reversal of D8 or D10 reruns the sweep's
cached results (their keys carry the edits), nothing else.

**D8 — The detection-limit rule (plan item 2.3; T-018).** `M11.C.22`'s mild limb had about a 78% pass chance at the
runner's stop (T-018): when an effect is near the precision the stop is set to, the adaptive stop makes the verdict
itself a draw, and a sweep that counts verdicts would count that noise as a fraction.
- (a) In the sweep: a sample whose first run, under `M17.A.1`'s adaptive stop, leaves any gating readout's mean
  difference within a declared multiple *k* of that readout's interval half-width is run again at a declared fixed
  seed count *N*, and the sample's verdict is the second run's; both seed counts are recorded. No rerun when the first
  run already used *N* seeds or more (a criterion with its own fixed count, `M11.C.10`, `M11.C.48`). *k* and *N* are
  `[I]`. `M17.A.1` gains one sentence stating the rule and that the sweep applies it now while the gating records
  adopt it at the next engine-hash move (the runner would have to change). Spec revision 37, wording in `M17` only.
- (b) Every sample at a fixed seed count, no adaptive stop.
- (c) No rule; the adaptive stop as it is, and the noise reported.

*Recommended: (a), with k = 2 and N = 200.* **Decided as recommended under the owner's instruction of 2026-10-22,
pending the owner's review.** (a) is the rule T-018 describes and the version 2 plan's item 2.3 names; it costs a
second run only where the first could not tell the direction. (b) costs four blocks on every sample where most stop
at one. (c) leaves the fraction partly a measurement of the stop. k = 2: an effect smaller than twice the half-width
is within about four standard errors of zero, where a stop at the first converged block decides the sign on little.
N = 200 is four blocks of 50, the count the owner chose for `M11.C.10` ("200 instead", revision 30), and at 200 seeds
the half-width is 0.14 of the differences' spread (1.96/√200), against the 0.25 of the arms' pooled spread the
condition asks, so a rerun converges unless the differences spread 1.8 times more than the arms do; a rerun that does
not converge is UNDETERMINED and counted as such. Neither number has a source; both are `[I]` in the sweep's config (D12). The gating
records are not rerun under the rule in this stage: that is an engine-side change, and T-018 stays open for it,
landing with T-123's fix at the next hash move.

**D9 — The pilot's two subjects (plan item 2.2).** One passing criterion and one accepted finding, each directional,
each with a verdict that did not move at revision 36 (T-122 lists the five that did), each on a family whose runs
are cheap enough that the pilot measures the sweep and not one long record.
- The passing criterion: **`M11.C.16`** (composite, Phase C, PASS at version 1 and at revision 36 with
  `repertoire_entropy` −0.0435 ± 0.0061; proved by four red mutants, `level-blind` among them). Its arms differ only in level and
  both learn, so the constants its mechanism reads are the learner's and the selection's (`M4.D.6`, `M4.D.6b`,
  `M4.D.6e`, `M4.D.1`, `M4.G.3`), the same family the D9 sweep moved one at a time, so the two sweeps can be read
  together.
- The accepted finding: **`M11.C.42`** (composite, Phase C, FAIL at version 1 and at revision 36 with `triangle_reuse`
  −0.34 ± 0.35; the owner's disposition of 2026-10-17: a `TRIANGLE` relieves its seeker less than a `DISTANCE`, so it
  is not learned and not reused). It is in group A (triangling), which the owner chose for stage 3 (Q2), so whether
  the finding holds across the learner's and the triangle-relief constants (`M1.C.1`, `M1.C.3`) or only at the central
  point is what stage 3's proposal needs to know first.
- Not chosen: `M11.C.44` (moved FAIL → PASS at revision 36, no recorded mutation); `M11.C.39` (passing, Phase D part I,
  slower, and its arms differ by a mechanism switch rather than by constants); the part II criteria (each run is
  minutes, so a pilot on one would measure one record's cost).

**Decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review.**

**D10 — The sweep's scope, ranges and sampling (plan item 2.1; Q3 decided per criterion).**
- *Which criteria.* The directional criteria (composite and mixed), passing and failing alike: a failing criterion's
  fraction tells stage 3 whether the finding is a property of the range or of the central point. Premises and checks
  are not swept: a premise passing confirms the code (`M11.5`), and a fraction on one would be read as a finding.
- *Which constants.* For each criterion, the `[I]` constants read by the mechanism its spec row credits and by the
  learner where both arms learn, declared in the sweep's config (D12) with the spec ID each constant is registered
  under in `config/bowen/constants.md`. Each constant's grade is taken from `docs/model_explainer.md` where a grep of
  its name finds it; **none of the pilot's sixteen constants is named there by its code name** (checked 2026-10-23:
  `grep -n` of each name in the explainer returns nothing), so each is `[I]` until proven otherwise, as
  `constants.md` already grades it, and the config says so in its grade column. The nine constants revision 36 added
  are `[I]` by their `constants.md` entries. The pilot's sets: `M11.C.16` — `learning_rate`, `credit_horizon`,
  `credit_discount`, `cross_person_weight`, `policy_temperature`, `habituation_rate`, `habituation_window`;
  `M11.C.42` — those seven and `triangle_transfer_rate`, `outsider_positional_gain`, `triangle_activity_window`.
- *Ranges.* Half to double the central value, the rule the D9 sweep used, bounded by the constant's admissible
  interval: a share or weight is capped at 1 (the value at which the thing it scales is not discounted), and an integer
  count of ticks is rounded to whole ticks and never below 1. Each range is written in the config with its bound and
  the reason for any cap. All `[I]`.
- *Sampling.* Each sample draws every constant in the criterion's set at once (the joint space `M17.E.1` asks about),
  from a generator seeded by a declared sweep seed and the criterion's key: a multiplicative constant log-uniformly
  over its range, so the central value is the range's midpoint; an integer uniformly over its whole values. Samples
  are drawn one at a time in order, so a longer sweep's first *S* samples are the pilot's and its cached results are
  reused.
- *The pilot's sample count.* S = 8 per criterion, `[I]`. Enough to time sixteen child runs and the reruns the rule
  triggers; not enough to read as a measurement of either criterion, and the record says so.
- *`M17.E.1`'s named ratio* (tie hardening or attenuation against the rate anxiety moves) is not in the pilot. It
  needs a ratio declared beside the two rates in `M10`, which is spec wording and a range row, and it is listed in
  §2.3 as not done; it is declared with the full sweep's ranges (§2.4 step 7).

**Decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review.**

**D11 — The sweep record, the version 2 status page and their tests (plan item 2.4; W1, W4).**
- `docs/version_2_sweep_record.md`, generated by `tools/version_2_sweep.py` from `docs/records_cache/sweep_v2.json`
  (a `Cache("sweep_v2")`, keyed as every record is). One row per swept criterion: class, the central verdict from its
  ensemble record, the constants and ranges, *held* n of S, *undetermined*, *reruns at the fixed count*, *errored*
  (counted and excluded from the denominator, never dropped — learnings P2; a `FAILED` row is not cached and is rerun
  next time, P1). Then one table per criterion with every sample's values, seeds, rerun flag and verdict, and the
  machine-readable block every record carries. The header says in its first lines that a fraction is a measurement
  of the range and not a verdict, and no cell of the summary table prints PASS or FAIL for a fraction.
- `docs/version_2_status.md`, generated by `tools/version_2_status.py` on the model of `tools/version_1_status.py`:
  one row per `M11.C` criterion or variant with its class, record, verdict now, verdict at the tag `version-1` (read
  from the frozen page's table), mutation evidence (the red or reversed mutants of the two mutation records), the
  **sweep column** (*n of S held* for a swept criterion; *not swept* for a directional one not yet swept; *—* with
  the class for a premise or check), and its status under W1–W4. Dispositions come from `docs/version_1_dispositions.md`
  where the verdict is the tag's; a row whose verdict changed at revision 36 shows *changed at revision 36, pending
  the owner's review (T-122)* and is not final. W4's *narrow* mark waits for the fraction the owner chooses (Q4: after
  the pilot), so the column shows the fraction and no mark until then.
- Tests in `tests/test_version_2_sweep.py` and `tests/test_version_2_status.py` (§2.3). The checklist and the
  reading list name both pages.

**Decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review.**

**D12 — Where the sweep's constants and ranges live.** `config/sweep/sweep.md`, two tables in the form of
`config/bowen/constants.md`: the sweep's rules (`sweep_samples`, `sweep_seed`, `detection_limit_multiple`,
`detection_limit_seeds`, each with a grade) and the ranges (criterion, constant, low, high, scale, grade, why it is
read). Parsed strictly by the tool (an unknown constant, a range outside the admissible bound, a row without a grade
raises). Not under `config/bowen/`, because the engine hash covers that directory (§2.1 item 4) and the engine never
reads this file. `docs/SPEC_REVIEW_CHECKLIST.md` lists it and `tests/test_doc_checklist.py` covers the directory.
*Alternative:* a `.json` under `config/bowen/`, which the hash's `*.md` glob would miss; rejected because the repo's
graded constants are markdown tables with a reason column, and a file in that directory that the hash does not
cover invites a second one that it should. **Decided as recommended under the owner's instruction of 2026-10-22,
pending the owner's review.**

**D13 — `M17`'s drafted requirements (plan item 2.5; Q3's second part).** None of `M17.A.2`, `M17.B.1`, `M17.E.3`,
`M17.E.6` comes in with the pilot: the owner decided none until the pilot reports. The pilot's report (§2.10) says
whether its results needed any of them to be read, and that recommendation, if it is to bring one in, is a further
decision there. **Decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review.**

**Decided after the pilot, recorded in §2.10:** D14 the *narrow* fraction (Q4: after the pilot, before the full sweep
runs); D15 the full sweep's sample count and whether it runs in this session (the brief's limit: at most 4 hours on
this Mac with 10 workers); D16 whether any drafted `M17` requirement is recommended in.

### 2.3 Criteria and their tests

Each test name carries the criterion's ID and a spec ID where one applies. Each is shown failing under the named
mutation before it counts (§2.10). Acceptance tests here assert a direction of difference or a property of the
arithmetic, never an absolute threshold on a model result; a sweep fraction is reported as a measurement and no test
turns one into a verdict.

| ID | Criterion | Test | Proved failing by |
|---|---|---|---|
| S2.1 (`M17.A.1`, D8) | A sample whose first run leaves a gating readout's effect within k half-widths is rerun at N seeds, the verdict is the rerun's, both counts are recorded; a clear effect is not rerun; a first run of N seeds or more is not rerun | `tests/test_version_2_sweep.py::test_m17a1_s2_1_a_near_limit_sample_is_rerun_at_the_fixed_count`, `::test_m17a1_s2_1_a_clear_effect_is_not_rerun`, `::test_m17a1_s2_1_a_fixed_seed_criterion_is_not_rerun` (stub children; no model run) | the condition inverted; the rerun removed; the seed check removed |
| S2.2 (`M17.E.1`) | The fraction counts a sample as holding only when its verdict is PASS; UNDETERMINED and FAIL do not count; an errored sample is counted apart and excluded from the denominator, never dropped (P2) | `tests/test_version_2_sweep.py::test_m17e1_s2_2_fraction_counts_only_passing_samples_and_reports_errors` | UNDETERMINED counted as holding; the errored count dropped |
| S2.3 (D10) | Every sample lies in its declared range, an integer constant is whole, a log-scale draw is log-uniform (its median over many draws is the central value), the draw is seeded (same seed, same samples) and the first S of a longer draw are the shorter draw's | `tests/test_version_2_sweep.py::test_s2_3_samples_stay_in_range_are_whole_and_seeded`, `::test_s2_3_a_longer_draw_begins_with_the_shorter_one` (P7: a linear draw on a log range puts the median off the centre and fails) | the bound clipped instead of checked; the generator unseeded; the integer rounding removed |
| S2.4 (P8) | A cached result on a stale engine hash is not reused: every sample runs again and saving the cache drops the old entries | `tests/test_version_2_sweep.py::test_s2_4_a_cached_result_on_a_stale_hash_is_not_reused` | the key without the engine hash |
| S2.5 (P38) | The record renders from the cache without running anything; without workers a missing result raises rather than runs; a label change in the renderer reruns nothing | `tests/test_version_2_sweep.py::test_s2_5_record_renders_from_the_cache_and_a_missing_result_raises` | the cache not read |
| S2.6 (rule 9, P4; D12) | Every rule and range comes from `config/sweep/sweep.md` with a grade; every constant named there is a row of `constants.md`; a range outside the admissible bound, a missing grade or an unknown constant fails at load; the tool holds no range or rule literal | `tests/test_version_2_sweep.py::test_s2_6_sweep_config_is_loaded_strictly_with_grades`, `::test_s2_6_no_range_or_rule_literal_in_the_tool` | a check removed; a literal pasted into the tool |
| S2.7 (`M17.E.1`) | A sample's edits are the sampled values written into the constants file's rows, exactly once each, and a rerun's edits add the two stopping rows | `tests/test_version_2_sweep.py::test_m17e1_s2_7_a_sample_edits_each_constant_row_once` | a row matched more than once |
| S2.8 (W1) | Every `M11.C` criterion in the spec's table has a row on the version 2 status page and no row lacks a status | `tests/test_version_2_status.py::test_w1_s2_8_every_criterion_has_a_status` | one criterion dropped from the page |
| S2.9 (W4, `M11.5`) | The sweep column shows *n of S* for a swept criterion, *not swept* for an unswept directional one and no fraction for a premise or check; no cell of that column prints PASS or FAIL | `tests/test_version_2_status.py::test_w4_s2_9_sweep_column_is_a_fraction_not_a_verdict` | a fraction rendered as PASS above a threshold |
| S2.10 (W1) | A row whose verdict is not the tag's is not final and says so; a row whose verdict is the tag's carries version 1's disposition | `tests/test_version_2_status.py::test_w1_s2_10_a_changed_verdict_is_not_final` | the tag's verdict ignored |
| S2.11 | The two pages are what their tools render (record-current) | `tests/test_version_2_sweep.py::test_s2_11_sweep_record_is_current`; `tests/test_version_2_status.py::test_s2_11_status_page_is_current` | a hand edit to either page |
| S2.12 | The engine hash is unchanged by the stage | the record-current tests of every existing record (`test_m134_ensemble_record_is_current`, `test_m111d_mutation_record_is_current`, `test_d9_sweep_record_is_current`, the Phase D record tests) fail if it moves; `engine_hash()[:12]` printed before and after (§2.10) | — |
| S2.13 (revision 37) | `M17.A.1` states D8's rule and where it applies; `M17.E.1` says a sample's verdict is `M17.A.1`'s and the fraction is a measurement beside the gating verdict; the spec's status line, `docs/spec_revision_log.md`, the indexes and the rules view agree | `tests/test_spec_consistency.py`, `tests/test_doc_claims.py::test_d1_spec_revision_restatements_match_the_front_matter`, the index-current tests | — |
| W1 | every criterion has a final status | S2.8, S2.10 | — |
| W2 | every premise and check passes | the status page's conditions block, S2.11 | — |
| W3 | every passing composite has a recorded mutation | the existing `test_m111d_*` tests; the page's W3 condition names the passing rows without one: `M11.C.11` and `M11.C.15`, whose mutants, red at the tag, survived at revision 36 (`version_2_revision_36_changes.md`), and `M11.C.48` in both variants (T-123) | — |
| W4 | every passing directional criterion has a sweep result | S2.9; the page's W4 condition names the passing rows not yet swept, met only when the full sweep has run | — |
| `M17.E.1`'s named ratio | not done in this stage's pilot (D10); declared with the full sweep's ranges | — | — |

### 2.4 Order of work

| Step | Work | Depends on | Moves the hash |
|---|---|---|---|
| 1 | This section, its decisions, and the documents of §2.7 that describe the plan; committed alone | — | no |
| 2 | Tests first: S2.1–S2.7 red | 1 | no |
| 3 | `config/sweep/sweep.md`; `tools/version_2_sweep.py` (the driver and the renderer); S2.1–S2.7 green | 2 | no |
| 4 | Spec revision 37 (`M17.A.1`, `M17.E.1`; S2.13), the revision log, the indexes, the rules view, the coverage page | 3 | no (wording) |
| 5 | **The pilot**, timed, in the background: `M11.C.16` and `M11.C.42` at S = 8; the record rendered; its wall time and the per-sample ratio written into §2.10 and `CHANGELOG.md` | 3 | no |
| 6 | `tools/version_2_status.py` and `docs/version_2_status.md` with the sweep column; S2.8–S2.11 green | 3, 5 | no |
| 7 | The projection (§2.5) from step 5's measurement; D14–D16 recorded in §2.10. **If the projected full sweep is at most 4 hours on this Mac with 10 workers:** declare every directional criterion's constants and ranges in the config (with the named ratio), run it in the background, render the record and the page. **If more:** stop after the pilot; the projection and the recommended split go to the owner | 5, 6 | no |
| 8 | Reviews before the push: one `learning-qa` pass and one correctness pass over the whole diff; every finding fixed; a narrow re-sweep on `sonnet`; three or fewer minor items left go to a TODO record | 7 | no |
| 9 | Push; then the end-of-stage-2 **milestone review** (`docs/milestone_review_brief.md` §2) in a separate session without this build's history, writing its report to `docs/cycles/` on its own branch; triage after it | 8 | — |

Step 4 comes before the pilot so the pilot runs under the rule as the spec states it, and after step 3 so the spec
sentence describes code that exists. Step 7's threshold is the coordinating brief's, not a cost estimate: no cost is
estimated before step 5's measurement.

### 2.5 Compute

Nothing in this stage reruns an existing record. The pilot is 16 samples (two criteria at S = 8), each a child run of
one Phase C criterion with 10 workers, plus a rerun at 200 seeds for every sample the rule catches; its wall time is
measured at step 5 and written in §2.10 before anything else runs. The full sweep's projection is then: for each
record, that record's measured time per row at revision 36's regeneration (§2.1 item 7) × the pilot's measured ratio
of a sample's time to a central run's × S × the number of directional rows in it (14, 13 and 7), summed. The ratio
and the per-row times are measurements; the product is a projection and is labelled one. Part II's rows (each about
11 minutes at the central setting) are expected to dominate it; whether the full sweep runs this session is step 7's
rule.

### 2.6 How a changed result is reported

No run in this stage changes the engine hash, so no verdict changes: the gating verdicts are revision 36's and stay
so. The sweep adds a measurement beside them — *n of S samples in which the direction held over the declared ranges*
— on the sweep record and in the status page's sweep column, next to the revision 36 verdict and the version 1 verdict.
A fraction is never written as a pass or a fail, and the pilot's two fractions at S = 8 are not read as a measurement
of either criterion: the record says what S was. A premise passing is never a finding (`M11.5`), and premises are not
swept. If a sample run raises, the row is `RAISED` with its cause, reported and not cached; it is not a verdict on the
model at that point, and a sample whose constants put the engine outside its invariants is a finding about the range,
written in the record's errored count. The five verdicts that changed at revision 36 (T-122) are shown on the status
page as changed and pending the owner's review; this stage does not decide them.

### 2.7 Documents each step updates

- Step 1: this file (its status line; the stage 1 status stays); `CHANGELOG.md` (an entry at the top, then
  `python3 tools/todo_index.py`); `docs/theory/_STATUS.md` (state and Next); `docs/REVIEWER_READING_LIST.md` (§7's
  row for this plan); `docs/todo/2026-08-22_blocking_the_next_phase.md` (T-001's text advanced; not ticked).
- Steps 3–4: `config/sweep/sweep.md` (new; `docs/SPEC_REVIEW_CHECKLIST.md` lists it); `docs/bowen_agent_model_spec_v2.md`
  (revision 37, `M17.A.1`, `M17.E.1`, the status line), `docs/spec_revision_log.md`, then `python3 tools/spec_index.py
  && python3 tools/spec_rules_view.py && python3 tools/spec_coverage.py tools/spec_coverage_overrides.json
  docs/spec_coverage.md`; the revision restatements (`README.md`'s Status, `CLAUDE.md`'s spec row, `_STATUS.md`'s
  state line, the checklist's spec entry); `docs/model_explainer.md` §18 (the sweep built for stage 2's pilot and the
  detection-limit rule, each sentence graded).
- Steps 5–7: `docs/version_2_sweep_record.md` and `docs/version_2_status.md` (new, generated; the checklist and the
  reading list §5 name them); §2.10 of this file; `CHANGELOG.md`; `CLAUDE.md`'s suite count; `docs/theory/_STATUS.md`;
  `docs/todo/2026-10-17_revision_32.md` (T-018's text: the sweep's part built, the gating records' part open);
  `docs/todo/2026-10-23_revision_36_regeneration.md` (T-123: where its fix lands); the TODO record of what the reviews
  leave.
- No readout is added, so `docs/readouts_phase_c.md` does not change; no constant the engine reads changes, so
  `config/bowen/constants_changes.md` does not change.

### 2.8 Not code-testable

- Whether D8–D16 stand: the owner's review of the numbered list in the build report, one decision at a time.
- Whether the pilot's two fractions mean anything at S = 8: they do not, and the record says so; the owner reads them
  as a timing run.
- Whether a sample that raised is a finding about the range or a defect: the owner reads its cause in the record.
- W5 (stage 3) and W8 (the owner's review of the outputs) are not this stage's.

### 2.9 Adjacent issues found while planning, not fixed

- `tools/version_1_status.py::record_rows` fails on `main`'s part II record (§2.1 item 5). The tool is frozen and only
  its `dispositions()` is live, so nothing reads the failing function; a reader who imports it to list the records
  gets `SystemExit`. Left as it is: the version 2 tool has its own reader.
- `tools/sweep_record.py`'s docstring still calls its six points `M17.E.1`'s sweep ("Spec: ... #M17.E.1"). It is the
  D9 sweep; `M17.E.1`'s is this stage's. Changing the docstring would move that record's `HASHED_TOOLS` hash and rerun
  it (5 min 42 s); left until that record is rerun for another reason.
- W3 is not met on four passing rows at revision 36: `M11.C.11` and `M11.C.15`, whose mutants
  (`removed-member-hold-released-while-away`, `death-bound-anxiety-discarded`), red at the tag, survived at the regeneration
  (`version_2_revision_36_changes.md`), and `M11.C.48` in both variants, whose mutant survives (T-123). `M11.C.44`,
  FAIL → PASS at revision 36 (T-122), has `one-sided-deviation` red, so W3 is met on it; the draft of this plan said
  the opposite and was corrected when the status tool was built. The status page shows all of this. Deciding the
  four is the owner's (stage 3 or a disposition), not this stage's.
- The runner's precision condition (`_converged`) depends on the seed count alone once the two arms' spreads are
  alike, so the adaptive stop converges at the first block on most criteria and the "adaptive" size is in practice 50,
  100 or 150. This is what makes T-018 bite; it is recorded here for the owner, not changed (an engine change).
- `docs/phase_c_sweep_record.md`'s table shows `M11.C.44`'s central verdict PASS and its row at `learning_rate` 0.1
  FAIL; a one-at-a-time sweep already shows that criterion's verdict is a property of the central point.
  The stage 2 sweep will say so with a fraction; nothing to fix.

### 2.10 Build record (2026-10-23)

Steps 1–6 are built on the branch `stage-2-plan-pilot`: the plan alone in `497e7ba`, then one build commit (the
sweep driver, its config, the status tool, their tests, spec revision 37 and the documents) and the review fixes.
Nothing in `src/bowen/` changed; `engine_hash()` printed `66dcbb1e8ba5` before the build and after it. No ensemble,
mutation or diagnostic record was rerun. Each criterion's test was shown failing under its mutation in place, with the
original restored by the scratch script's `try/finally` and checked equal afterwards; none is kept in the tree.

| ID | Test | Mutation | Result |
|---|---|---|---|
| S2.1 | `tests/test_version_2_sweep.py::test_m17a1_s2_1_a_near_limit_sample_is_rerun_at_the_fixed_count` | `near_limit` always False | fails |
| S2.1 | `…::test_m17a1_s2_1_a_clear_effect_is_not_rerun` | `near_limit` always True | fails |
| S2.1 | `…::test_m17a1_s2_1_a_fixed_seed_criterion_is_not_rerun` | the `seeds < fixed` guard removed | fails |
| S2.1 | `…::test_m17a1_s2_1_a_criterion_fixing_its_own_seeds_below_the_count_is_refused` (review fix) | the refusal of a rerun that did not run at the fixed count removed | fails |
| S2.2 | `…::test_m17e1_s2_2_fraction_counts_only_passing_samples_and_reports_errors` | errored samples kept in the denominator | fails |
| S2.3 | `…::test_s2_3_samples_stay_in_range_are_whole_and_seeded` | the draw seeded without the criterion's key | fails |
| S2.3 | `…::test_s2_3_a_longer_draw_begins_with_the_shorter_one` | the sample count folded into the seed | fails |
| S2.4 | `…::test_s2_4_a_cached_result_on_a_stale_hash_is_not_reused` (P8) | `run_cached`'s key without the engine hash (`tools/record_cache.py`) | fails |
| S2.5 | `…::test_s2_5_record_renders_from_the_cache_and_a_missing_result_raises` | a cache-only build runs a child | fails |
| S2.5 | `…::test_s2_5_a_premise_is_never_swept` | the class check removed | fails |
| S2.6 | `…::test_s2_6_sweep_config_is_loaded_strictly_with_grades` | the bracketing check removed (`tools/sweep_config.py`) | fails |
| S2.6 | `…::test_s2_6_no_range_or_rule_literal_in_the_tool` | `detection_limit_seeds` read as the literal 200 | fails |
| S2.6 | `…::test_s2_6_sweep_config_is_loaded_strictly_with_grades` (review fix) | the scale-against-`SCHEMA`-kind check removed; (re-sweep) the check made one-directional | fails, fails |
| S2.6 | `…::test_s2_6_explainer_quotes_the_config_values` (review fix) | the explainer's multiple changed to 3.0 | fails |
| S2.7 | `…::test_m17e1_s2_7_a_sample_edits_each_constant_row_once` | an integer written as `6.0` | fails |
| S2.8 | `tests/test_version_2_status.py::test_w1_s2_8_every_criterion_has_a_status` | one unbuilt criterion dropped from the page | fails |
| S2.9 | `…::test_w4_s2_9_sweep_column_is_a_fraction_not_a_verdict` | a fraction of one half or more rendered as PASS | fails |
| S2.10 | `…::test_w1_s2_10_a_changed_verdict_is_not_final` | the tag's verdict ignored; (review fix) the version 1 disposition on a changed row not marked *not applied* | fails |
| S2.10 | `…::test_w1_s2_10_the_tag_page_is_read_whole` (review fix) | the tag page's pattern requires a non-empty class cell (drops `M11.C.8`) | fails |
| S2.10 | `…::test_w1_s2_10_a_tag_row_with_no_row_now_and_a_misplaced_disposition_stop_the_tool` (re-sweep) | the `gone` check removed; the `misplaced` check removed | fails, fails |
| S2.11 | `tests/test_version_2_sweep.py::test_s2_11_sweep_record_is_current`, `tests/test_version_2_status.py::test_s2_11_status_page_is_current` | record-current tests; a hand edit to either page fails them by construction | — |
| S2.12 | the existing record-current tests, and `engine_hash()` printed before and after | — | `66dcbb1e8ba5` both times |
| S2.13 | `tests/test_doc_claims.py`, `tests/test_spec_consistency.py` at revision 37 | — | pass |

**The pilot (step 5).** `python3 tools/version_2_sweep.py --workers 10`, started 04:11:23 and finished 04:19:19 on
2026-10-23 on this Mac: **476 s wall** for 24 child runs — 16 first runs (8 samples × 2 criteria) and 8 reruns at
200 seeds, every one of `M11.C.42`'s samples and none of `M11.C.16`'s. `M11.C.16` held in **8 of 8** samples;
`M11.C.42` held in **0 of 8** (`docs/version_2_sweep_record.md`). Neither number is a verdict (`M11.5`); at 8 samples
they time the sweep and say nothing about the width of a band. No sample was undetermined or errored.

**The projection (step 7, §2.5's method).** Per sample, the pilot cost 29.8 s against the Phase C record's 7.9 s per
row at the same 10 workers (190 s / 24 rows), a ratio of 3.76: each sample is its own child (a copy of the repository
and a fresh import), and half the samples were rerun at 200 seeds. Applying that ratio to each record's measured
time per row (Phase C 7.9 s, part I 29.1 s, part II 682.5 s; `CHANGELOG.md` 2026-10-23) over the directional rows
(14, 13, 7): the full sweep at 8 samples is **44 h**, part II alone 40 h; Phase C and part I together 4.1 h; Phase C
alone 56 min. At 32 samples, 176 h. **Over the owner's 4-hour rule, so the full sweep did not run**; the sweep record
holds the pilot only. The projection assumes the pilot's rerun rate (one in two) and part II's per-row time at the
sweep's sampled constants; both are measured on two Phase C criteria only, and part II's figure is the least certain.

**D14 — the narrow fraction (W4's mark).** *Recommended:* a criterion is marked *narrow* when the direction held in
fewer than half of the counted samples, and the mark is read only from a sweep of at least 32 samples; below that
the column shows the fraction and no mark. The pilot's 8 samples cannot show a band, so the mark is not built in
stage 2 and its rule is not in `config/sweep/sweep.md`; it goes in, with its test, with the first sweep that has 32
samples. **Decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review.**

**D15 — the full sweep.** *Recommended:* not run in this session (44 h, above). The next step within the rule is
Phase C's 14 directional criteria at 8 samples (56 min), then part I's 13 (3.2 h) in a session of its own; part II's
seven at 8 samples (40 h) need the engine's part II arms to get faster first or a smaller sample, and the owner should
choose which. **Decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review.**

**D16 — the drafted requirements (D13 re-decided).** *Recommended:* none of `M17.A.2`, `M17.B.1`, `M17.E.3` and
`M17.E.6` comes into stage 2. The pilot showed no need: both criteria gave a verdict at every sample, and the one
sweep rule the pilot needed (the detection limit) is `M17.A.1`'s. **Decided as recommended under the owner's
instruction of 2026-10-22, pending the owner's review.**

**The Phase C sweep (D15's first step; 2026-10-23, the branch `stage-2-phase-c-sweep`).**
`python3 tools/version_2_sweep.py --workers 10`, started 05:32:36 and finished 05:59:37 on 2026-10-23 on this Mac:
**1620 s wall (27 min 0 s; the tool's integer figure — the timestamps span 1621 s)** for 155 child runs — 96 first runs (8 samples × the 12 Phase C directional criteria the
pilot had not run) and 59 reruns at 200 seeds under the detection-limit rule (`M17.A.1`); the pilot's 24 results were
read from the cache and nothing was rerun. The 14 criteria are the Phase C ensemble record's composite and mixed rows
(`M11.5`), the count §2.1 item 6 gives. Their constants and ranges are the 104 rows added to `config/sweep/sweep.md`
(121 rows, 31 constants, 14 criteria), each range half to double the central value within the file's bounds, every
grade `[I]`: one constant, `defence_threshold`, is named in `docs/model_explainer.md` (§11.3) and graded `[I]` there;
the other 30 are not named there by their code name; every constant in `config/bowen/constants.md` is graded `[I]`
(checked 2026-10-23), so no constant a mechanism reads was left out of a set for carrying another grade. No sample
was undetermined or errored. Nothing in `src/bowen/` changed; `engine_hash()` printed `66dcbb1e8ba5` before and after;
no ensemble, mutation or diagnostic record was rerun. The fractions, from `docs/version_2_sweep_record.md`:

| Criterion | Central verdict | Constants swept | Held of 8 | Reruns at 200 |
|---|---|---|---|---|
| `M11.C.5` | FAIL | the learner's seven, `debit_gain` | 0 | 8 |
| `M11.C.16` (pilot) | PASS | the learner's seven | 8 | 0 |
| `M11.C.27[stable,add_third]` | FAIL | `M4.C.1`'s seven | 2 | 2 |
| `M11.C.27[stable,remove_one]` | FAIL | `M4.C.1`'s seven | 3 | 3 |
| `M11.C.27[unstable,add_third]` | FAIL | `M4.C.1`'s seven | 6 | 3 |
| `M11.C.27[unstable,remove_one]` | PASS | `M4.C.1`'s seven | 6 | 3 |
| `M11.C.29` | FAIL | `exchange_budget_reduction`, `symptom_threshold_gain`, `symptom_rearm_fraction`, `symptom_leak_rate`, `defence_threshold` | 0 | 8 |
| `M11.C.41[light: lower level]` | PASS | the level-and-stress ten | 8 | 0 |
| `M11.C.41[heavy: lower level]` | PASS | the level-and-stress ten | 8 | 0 |
| `M11.C.41[higher level: heavier stress]` | FAIL | the level-and-stress ten | 0 | 8 |
| `M11.C.41[lower level: heavier stress]` | FAIL | the level-and-stress ten | 1 | 8 |
| `M11.C.42` (pilot) | FAIL | the learner's seven, `triangle_transfer_rate`, `outsider_positional_gain`, `triangle_activity_window` | 0 | 8 |
| `M11.C.44` | PASS | `M4.C.1`'s seven, the learner's seven | 2 | 8 |
| `M11.C.45` | FAIL | `triangle_transfer_rate`, `outsider_positional_gain`, the learner's seven | 0 | 8 |

The learner's seven are `learning_rate`, `credit_horizon`, `credit_discount`, `cross_person_weight`,
`policy_temperature`, `habituation_rate`, `habituation_window`; `M4.C.1`'s seven are `appraisal_gain`,
`intensity_scale`, `impingement_relaxation_rate`, `contact_band_max`, `anxiety_togetherness_gain`,
`interactive_resting_contact`, `contact_relaxation_rate`; the level-and-stress ten are `appraisal_gain`,
`contact_band_max`, `self_channel_exponent`, `capacity_level_per_layer`, `anxiety_band_low`, `anxiety_band_high`,
`acute_decay_rate`, `response_up_gain`, `response_down_gain`, `stress_floor_below_basic`. Every fraction is a
measurement of the declared range at 8 samples and not a verdict (`M11.5`); a premise passing is never a finding, and
no premise was swept; at 8 samples no *narrow* mark is read (D14). Two measurements the owner may want beside the
verdicts: `M11.C.44`, PASS at the central setting, held in 2 of 8 samples (the one-at-a-time D9 sweep had shown its
verdict moving at `learning_rate` 0.1, §2.9); `M11.C.27[unstable,add_third]`, FAIL at the central setting, held in
6 of 8.

*Time against the projection.* Per sample 16.9 s (1620 s over 96 first runs, the reruns included) against the
pilot's 29.8 s; the 12 criteria took 27 min against the 48 min the pilot's ratio projects for them (56 min × 12/14,
the pilot's two being cached). A sample cost 2.14 times the Phase C record's 7.9 s per row, against the pilot's
3.76, with reruns on 59 of 96 samples (61%) against the pilot's one in two. Both figures are measurements on Phase C
runs; at Phase C's ratio part I's 13 criteria at 8 samples scale to 1.8 h (3.2 h × 2.14 / 3.76), and part II's seven
to 23 h — projections, since no part I or part II row has run at sampled constants.

*The constant sets are judgments* (D10's rule applied; the config's preamble states the rule, each row's last column
the reason), each **decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review**:
(1) the learner's seven are in a set where the criterion row or its class row names learning (`M11.C.5`: `M5.E.0`;
`M11.C.45`: `M1.C.3`'s "`M4.D.6` does not reinforce it") or the readout counts selected acts over the run
(`M11.C.44`), and out where the readout is taken one week after a scripted act (`M11.C.27`) or the row credits no
learned mechanism (`M11.C.29`, `M11.C.41`); (2) `M11.C.29`'s set holds `exchange_budget_reduction` (`M6.I.1`, the
limb the row credits, report-only in the gate), the onset and end of the third's symptom (`M7.D.1`, `M4.C.3a`) that
the gating readout counts, and `defence_threshold` (`M4.C.2`), the arm's precondition named in its status note;
(3) `M11.C.41`'s set reads the mixing weight (`M4.D.1a`, `M4.D.3a`), the bands (`M4.D.3`), the stress response's two
gains and floor (`M1.A.5e`) and `acute_decay_rate` (`M1.A.8`), and leaves out `response_relaxation_rate` and
`usual_level_rate` (the return, not the fall) and the perspective gains (no arm sets perspective); (4) `M11.C.45`'s
set leaves out `M4.C.1`'s seven (the disposition names the calm state as a cause, but the row credits `M1.C.1` and
`M1.C.3`; `M11.C.44` sweeps them) and `triangle_activity_window` (the readout counts selections, not activity);
(5) `nodal_spike_intensity` (no nodal event in these scenarios) and `functional_level_floor` (a numerical guard) are
in no set; (6) `anxiety_band_high`'s low end is 10, the top of `anxiety_band_low`'s range, not half its central value,
so every joint sample keeps the engine's low ≤ high; a sample outside `EngineParams`'s invariants would be an
errored sample (`tools/mutant_runner.py` turns the engine's `ValueError` into one), and the record's errored count
is 0 on every row; (7) the sample count stays 8 (D15), so the record's header sentence that
these fractions time the sweep and do not measure a criterion stands for every row.

**The Phase D part I sweep (D15's second step; 2026-10-23, the branch `stage-2-part-i-sweep`).** The owner,
2026-10-23: "Run part I's 13-criterion sweep now, as recommended, on a branch, and merge it to main when green." The
13 keys are the Phase D part I record's composite and mixed rows (`M11.C.7`'s three pairs each a key), at 8 samples.
Their constants and ranges are the 125 rows added to `config/sweep/sweep.md` (246 rows, 60 constants, 27 criteria),
each range half to double the central value within the file's bounds, every grade `[I]`. The first run, started
08:30:57, raised `KeyError` on all 104 samples in 81 s and cached nothing (*Where the build differs*, below). After the
fix, `python3 tools/version_2_sweep.py` ran in three segments, stopped twice at the owner's request to free the Mac:
08:35:56–09:26:23 at 10 workers, 09:26:28–09:52:43 at 5 workers at the lowest priority (`nice` 20), and
09:52:46–12:10:07 at 8 workers at `nice` 20 with a video call running. **12,843 s of run time (3 h 34 min)**, 3 h 39 min
from the first start to the end, for 182 child runs: 104 first runs and 78 reruns at 200 seeds under the
detection-limit rule (`M17.A.1`); each stop lost the sample then running, which was run again. Phase C's 179 cached
results were read and not rerun. No sample was undetermined or errored. Nothing in `src/bowen/` changed;
`engine_hash()` printed `66dcbb1e8ba5` before and after (the Phase D results are keyed by
`tools/phase_d_mutation_record.py`'s hash, `5c3865dcf8f9…`); no ensemble, mutation or diagnostic record was rerun.
The fractions, from `docs/version_2_sweep_record.md`:

| Criterion | Central verdict | Constants swept | Held of 8 | Reruns at 200 |
|---|---|---|---|---|
| `M11.C.2` | FAIL | the learner's seven, the four focus gains, `investment_leak_rate`, `witness_weight`, `symptom_leak_rate` | 1 | 8 |
| `M11.C.7[joint over one spouse]` | FAIL | the coaching eleven | 3 | 8 |
| `M11.C.7[joint over sided]` | PASS | the coaching eleven | 7 | 7 |
| `M11.C.7[one spouse over sided]` | FAIL | the coaching eleven | 5 | 8 |
| `M11.C.11` | PASS | `lock_in_gain`, `lock_in_turn`, the symptom's threshold, re-arm and leak, `standing_load_gain`, the contact relaxation pair, `acute_decay_rate` | 8 | 1 |
| `M11.C.12` | FAIL | the learner's seven, `lock_in_gain`, `acute_decay_rate` | 0 | 8 |
| `M11.C.17` | FAIL | the coach's three, the self-practice pair, `system_adjust_gain`, the estimator's two, the perspective gains, `nodal_spike_intensity` | 4 | 5 |
| `M11.C.21` | FAIL | the pseudo-self pair, `exchange_window`, the stress response's two gains, its relaxation and floor | 1 | 8 |
| `M11.C.22` | PASS | `lock_in_gain`, `standing_load_gain`, the contact relaxation pair, `acute_decay_rate` | 6 | 8 |
| `M11.C.23` | FAIL | the appraisal pair, `impingement_relaxation_rate`, `investment_leak_rate`, the symptom's threshold, re-arm and leak, `acute_decay_rate` | 0 | 1 |
| `M11.C.34` | FAIL | the learner's seven, `self_channel_exponent`, `capacity_level_per_layer`, `symptom_event_intensity` | 0 | 8 |
| `M11.C.39` | PASS | the learner's seven, `witnessed_cutoff_gain`, `witness_window`, `distance_binding_rate` | 8 | 0 |
| `M11.C.43` | FAIL | the learner's seven, `calm_transfer_rate`, `acute_decay_rate` | 0 | 8 |

The coaching eleven are `M4.C.1`'s seven, `nodal_spike_intensity`, `landing_rate`, `perspective_gain` and
`binder_failure_fraction`; the coach's three are the last three of those; the other groups are named as above. Every
fraction is a measurement of the declared range at 8 samples and not a verdict (`M11.5`); a premise passing is never a
finding, and no premise was swept; at 8 samples no *narrow* mark is read (D14). Measurements the owner may want beside
the verdicts: `M11.C.7[one spouse over sided]`, FAIL at the central setting, held in 5 of 8; `M11.C.17`, FAIL at the
central setting and changed since the tag (T-122), held in 4 of 8; `M11.C.22`, PASS, held in 6 of 8. With these rows
W4 lists three unmet rows, not seven (`docs/version_2_status.md`): `M11.C.15`, `M11.C.48` and `M11.C.48[phase_d]`.

*The reruns.* 78 of the 104 first runs (75%, against the Phase C sweep run's 59 of 96, 61%) left a gating effect within 2 half-widths of zero,
and 53 within one; the rerun changed the verdict in 6 of the 78. Reruns took 15,600 of the 29,050 seeds run (54%). A
first run stops on precision against the seed-to-seed spread (`src/bowen/ensemble/runner.py::_converged`), not on
distance from zero, so a small effect stops near zero; from 50–150 seeds to 200 the half-width narrowed by about a
fifth (median ratio 0.84 over the reruns' gating readouts), so the rerun settles only a sample near the line. Told this cost on 2026-10-23, the owner chose to keep D8's
200 seeds for this run. A rerun repeats the first run's seeds, which are deterministic; running only the missing ones
would give the same numbers with about a third fewer seeds in this run (the first runs' seeds inside the 78 reruns), and is not built.

*Time against the projection.* 3 h 34 min of run time against 1.8 h at Phase C's ratio and 3.2 h at the pilot's, both
projected at 10 workers on an otherwise idle Mac; 2 h 43 min of this run was at 5 or 8 workers at the lowest priority
beside other work, so the comparison is not like for like. Per sample 123.5 s (12,843 s over 104 first runs, the reruns
included) against Phase C's 16.9 s; per child run 70.6 s against Phase C's 10.5 s.

*The constant sets are judgments* (D10's rule applied; the config's preamble states the rule, each row's last column
the reason), each **decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review**,
the owner choosing on 2026-10-23 to have them decided and listed in T-125 (batch 5, decisions 29–44): (1) the learner's
seven are in `M11.C.2`, `M11.C.12`, `M11.C.34`, `M11.C.39` and `M11.C.43`, whose rows or class rows name learning or a
readout of selected acts, and out of the other eight, whose readouts are states; (2) `M11.C.7`'s three pairs share
one set, the coach's constants included, which every arm reads at the same value; (3) `nodal_spike_intensity` is in
`M11.C.7` and `M11.C.17`, which run under nodal load; (4) `birth_focus_gain` is in no set (no birth in a part I
scenario), nor `M1.A.19`'s `reactive_rate` in `M11.C.34` (no selection there reads its detectors); (5) a constant an
arm's declared condition is set against is left out: `lock_in_turn` in `M11.C.12` and `M11.C.22`, `estimator_window`
and `estimator_breadth` in `M11.C.17`; (6) `M11.C.23`'s arm setting `suppression` is not a constant and is not swept;
(7) `witness_window`'s low end is 7, half of 13 rounded up to a whole week; (8) `binder_failure_fraction` (0.8) runs
0.4 to its bound 1.0, and the perspective gains are capped at 1, above which the channel weight passes 1 and the
steering factor can turn negative (`src/bowen/policy/policy.py`, lines 193 and 289); (9) a Phase D criterion's
results are keyed by the Phase D mutation record's hash, printed as a second line of the record.

**The two-Mac tooling (2026-10-23, the branch `claude/sharp-gauss-30cf8a`, before part II).** Part II's seven
criteria are projected at 23–40 h on one Mac, so the sweep can now run on this Mac and Mac Mini1 at once
(`docs/two_mac_sweep.md`). `tools/version_2_sweep.py --criteria KEY ...` checks every declared key first (central
record, directional class), refuses an undeclared key, runs only its share and writes only the cache, every entry
kept (`Cache.flush`, not `Cache.save`, which keeps only the keys a run read and would drop the other share's
results); it never writes the record, because a partial record would replace the whole one and fail S2.11 until
the merge. `--recompute` drops the named criteria's cached results and runs them again (a failed rerun keeps the
earlier result), the cross-check that two Macs compute the same numbers: a split by criterion shares no key, so the
merge's conflict check could not otherwise fire. `--render-only` renders the record from the cache and runs
nothing. `tools/merge_cache.py` writes the union of two cache files and refuses, writing nothing, when one key holds
two results, the files' NumPy versions differ or either records none, or a sweep is running on this Mac (it would
overwrite the merge at its next write). The NumPy version is recorded in the cache file under `_meta`, outside the
result keys, so no cached result is orphaned. The committed cache was stamped `1.26.4` once, on 2026-10-23, by
`check_numpy` before it was made to refuse a cache holding results and no version; every one of its 361 results was
made under that version (this Mac's Python 3.12, NumPy installed 2024-05-29). `requirements-sweep.txt` pins
`numpy==1.26.4` for sweep machines; a run under another version, or another than its cache's, is refused, and
`--render-only` is not checked, so CI stays green on NumPy 2 (`requirements.txt` stays unpinned: Python 3.11–3.13,
and 1.26.4 has no 3.13 wheel). `tools/two_mac_sweep.py` checks that Mac Mini1 matches (commit, clean tracked files,
both engine hashes, Python, CPU, NumPy against the pin; macOS reported), checks the keys here before any SSH, starts
the share detached with the venv's Python and `START`/`EXIT` lines in its log, refuses when a run is going or
`pgrep` fails, confirms the run is alive after 15 s, and copies the cache back to a new file. The `learning-qa`
pass found seven items (no cross-machine check; `start` not confirming the run; no exit status in the log; a merge
lost under a running sweep; an unstamped cache stamped silently; untested render-only and retry paths; `pgrep`
failing open), all fixed. Tests: `tests/test_sweep_two_macs.py`, 31 tests (43 with parameters), each shown failing
under mutation (46 mutants, all killed). Nothing in `src/bowen/` changed, `engine_hash()` printed `66dcbb1e8ba5`
before and after and `phase_d_engine_hash()` `5c3865dcf8f9`; no record was rerun. Not verified: the cross-check
has not been run, so bit-identity across the two Macs (both arm64, Python 3.12.3, NumPy 1.26.4; macOS 15.7.9 and
15.7.8) is untested, and the driver has not run against a live Mac Mini1 beyond its `pgrep` call.

**Where the build differs from the plan.**
- **"Changed at revision 36" is written "changed since the tag `version-1`"** (D11). The tool compares the current
  record with the frozen page and cannot know at which revision a verdict moved; the status cell names the tag and
  T-122, and the page's header says T-122 is where the changes are listed. The data is the same.
- **W3's unmet rows are `M11.C.11`, `M11.C.15` and `M11.C.48` (both variants), not `M11.C.44`** (§2.4's W3 row and
  §2.9, corrected when the status tool first ran). `M11.C.44` has `one-sided-deviation` red in the Phase C mutation
  record; the two Phase D criteria's mutants, red at the tag, survived at revision 36.
- **`tools/version_2_status.py` repeats version 1's status rules** (about 90 lines of `tools/version_1_status.py`)
  instead of importing them, because the frozen tool's `statuses()` reads its own `record_rows()`, which fails on
  `main`. Its live readers (the spec's table, the mutation records, the not-built lists, the dispositions) are
  imported. If the frozen tool is ever unfrozen, the rules can be shared.
- **The sweep config's tables are checked by header**, not by cell count: a table whose header is not the declared
  column names raises, so a column added or reordered cannot be read as another.
- **`docs/records_cache/sweep_v2.json`** holds the pilot's 24 results; a `FAILED` result is never written to it.
- **A Phase D criterion is run by the Phase D child** (2026-10-23, found by the part I sweep's first run). The driver
  ran every sample through `tools/mutant_runner.py`, whose child knows only Phase C's `CRITERIA`, so the first part I
  run raised `KeyError` on all 104 samples in 81 s and cached nothing. A criterion of the Phase D part I or part II
  record now runs through `tools/phase_d_mutant_child.py` and its results are cached under
  `tools/phase_d_mutation_record.py`'s hash (that child and the source of `apply` and `run_in_copy`); the record prints
  it as `phase_d_engine_hash`. Phase C's results keep their keys. `tools/record_cache.run_cached` takes the child and
  the hash as arguments. Test: `tests/test_version_2_sweep.py::test_m17e1_phase_d_criterion_is_run_by_the_phase_d_child_under_its_hash`.
- **Three renders moved with the revision number only:** the seed-7 trace and chart and the famdiag export print the
  spec's revision, so revision 37 re-rendered them (the same seed, the same engine; `git diff` shows the one line).
  `tools/record_changes.py` lists the two new pages under *not compared*, which re-rendered both changes pages.

**The reviews (step 6).** Both ran in parallel on the build commit's diff, 04:43:45 to 04:53:07 on 2026-10-23:
`learning-qa` (6 min 43 s) and a correctness pass without build context (8 min 05 s). Every finding was fixed in the
following commit, with a test for each fix shown failing under its mutation (the table above, *review fix*):

- `tag_verdicts` read the frozen page with a pattern that required a non-empty class cell, so `M11.C.8` (not built, no
  class) was dropped silently (P2); and no reconciliation checked the count read against the page's rows. Fixed: the
  pattern allows an empty cell, and a row count that does not match the rows read stops the tool.
- A changed row (`M11.C.7[joint over one spouse]`, `M11.C.17`, `M11.C.44`) dropped its version 1 disposition from the
  page. Fixed: it is shown as *version 1 disposition, not applied (the verdict changed)* with its text, so the owner
  sees what the tag had decided; it is never applied to the new verdict.
- A criterion with its own `seeds` in `criteria.md` below the fixed count (`M11.C.48`, 100) would ignore the block and
  cap written into the copy, so its rerun would run the same 100 seeds again and be recorded as a rerun at 200.
  Fixed: a rerun whose seed count is not the fixed count stops the sweep, naming the criterion; the spec says so
  (`M17.A.1`).
- `near_limit` reads directional limbs only; an equivalence limb (direction 0) has no detection limit of this kind.
  The docstring, `M17.A.1` and the explainer now say so, and `M17.A.1` says an unconverged rerun is UNDETERMINED.
- The sweep config did not check a range's scale against the constant's `SCHEMA` kind, so a `log` range on an integer
  constant would have drawn floats. Fixed: the scale must agree with the kind.
- `build` checked each criterion's class only as its turn came, so a premise late in the config was refused after
  earlier samples had run. Fixed: every class is checked before any sample runs.
- Tests tied to the pilot's config (`set(ranges) == {…}`, a `next(…)` over log ranges, a sample-edit count that
  assumed no draw equals the central value) were loosened to hold for any config; a new test checks that the
  explainer's quoted multiple and count are the config's.
- Wording: *red mutants survived* read as if the mutants were red now; it is *mutants, red at the tag, survived at
  revision 36* here, in §2.4, §2.9 and the CHANGELOG. `config/sweep/sweep.md`'s header no longer says a capped log
  range's central value is its midpoint, and the fixed count is one block, not four of 50.
- `M17.E.1`'s first sentence now says *directional* `M11.C` criterion, which is what the rest of the requirement and
  the tool do.

One minor item was not fixed and is recorded (T-124): the Phase D mutation record's *engine_hash* label is that tool's
own digest (the code hash plus its child functions), not `record_cache.engine_hash()`; version 1 did the same and
the two are not compared anywhere.

**The re-sweep of the fix diff** (`learning-qa`'s role on sonnet, 2 min 20 s, 114,302 tokens, 27 tool uses) found no
blocking item and these minor ones, fixed in the same commit: the *not applied* note said *the verdict changed* even
for a key with no row at the tag (now *no row at the tag*); a constant in `constants.md` but not in `SCHEMA` was
reported as not a constant (now its own message); the scale-against-kind rule was tested in one direction only (a
float constant on an integer scale added); the explainer did not say an equivalence limb is not read (added); the
`gone` and `misplaced` checks of `tools/version_2_status.py` had no test (one added, each shown failing). Not taken:
the tool's `sys.path` insert before importing `src.bowen.scenario.constants`, which every tool here does.
